//! Presenting the canvas: the screen frame and export (§6.3, §15.6). What is *in*
//! a frame is the draw list's (`engine::draw_list`); this is where one is asked for,
//! cached, and composited.
//!
//! **One path serves all three consumers**, which is what keeps them from
//! disagreeing about what the document looks like. [`Engine::render_view`] takes a
//! [`RenderSpec`] and somewhere to put the pass-A attachments, and every caller differs
//! only in those: the screen renders through the session's view with chrome, the
//! navigator's miniature through a planned rect without it, and an export through the
//! same planned rect into a texture it then reads back.

use super::Engine;
use super::presenter::{Attachments, DrawKey};
use crate::Result;
use crate::device::context::GpuContext;
use crate::document::DocState;
use crate::error::{ExportError, Produces};
use crate::gpu::composite::{CompositeScene, Offscreen, Output, SelectionOutline};
use crate::image::RgbaImage;
use crate::view::{Extent2, ViewTransform};
use stark_model::document::{GuideScene, LayerId};

/// What sits under the paint when rendering (§15.6).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Background {
    /// The document's substrate color, lit and textured by the canvas substrate —
    /// what the screen shows.
    #[default]
    Substrate,
    /// Nothing: the paint's own visible alpha becomes the image's alpha, for a
    /// cut-out PNG. The substrate composite is skipped entirely, so bare canvas is
    /// genuinely absent rather than white-and-invisible.
    Transparent,
}

/// Whether on-canvas affordances (the selection outline) are drawn. Screen: yes.
/// Export: never — chrome is a thing to draw *with*, not a thing to ship
/// (§15.6).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Chrome {
    Shown,
    Hidden,
}

/// **What one render draws**, whole: the framing, how much of the document, what sits
/// under the paint, and whether chrome goes over it.
///
/// One value rather than five arguments, because the five travel together through every
/// entry point on this path — `render_into`, `export_view` and the two probes all hand
/// the same set down two levels.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(super) struct RenderSpec {
    /// The framing, whose `viewport` is the target's size.
    view: ViewTransform,
    /// A single layer to draw **alone**, its blend mode, clip and opacity dropped —
    /// see [`DrawList::layer_alone`](super::draw_list::DrawList::layer_alone), which
    /// settles what that means (§18.0.2). `None` is the document, which is every render
    /// but a layer thumbnail's (§14.6).
    only: Option<LayerId>,
    background: Background,
    chrome: Chrome,
    content: Rendered,
}

impl RenderSpec {
    /// What the screen draws: the whole document through `view`, on its substrate, with
    /// chrome, following the gesture in flight. The two probes read back exactly this,
    /// which is what makes them a picture of the screen rather than of a path beside it.
    fn screen(view: ViewTransform) -> Self {
        Self {
            view,
            only: None,
            background: Background::Substrate,
            chrome: Chrome::Shown,
            content: Rendered::Live,
        }
    }
}

/// Which document a render draws: the one being *shown*, or the committed one alone.
///
/// The screen wants [`Rendered::Live`], which is what makes a stroke visible as it is
/// drawn. A render standing in for the *state of the work* wants
/// [`Rendered::Committed`], since following an in-flight stroke would mean
/// re-rendering at pointer rate.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Rendered {
    /// The committed document with every in-flight gesture — this client's and
    /// each peer's — and any unlogged drag preview drawn over it
    /// (§17.6). What the screen shows.
    #[default]
    Live,
    /// The committed document alone: no in-flight stroke, no drag preview.
    Committed,
}

/// How large an exported image is, relative to the frame's canvas-space size
/// (§15.6). Resolution is a property of the *output*, not of the
/// artwork, which is why the frame stores only a canvas-space rect.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ExportScale {
    /// A multiple of the frame's canvas size — 1× is one canvas px per image px.
    Factor(f32),
    /// An exact width in image px; the height follows the frame's aspect.
    Width(u32),
    /// The largest scale whose output fits inside a box of image px, both axes
    /// respected — what a *preview* of the whole piece asks for. Scales *up* as
    /// happily as down: the overview shows the whole of a piece at a glance.
    ///
    /// Distinct from a 1× plan the caller then scales itself, because a piece wider
    /// than the device's texture limit (`Presenter::max_dim`) would fail *that* query
    /// for a render it was never going to make.
    Fit(Extent2),
}

/// What an export will produce, before producing it — so a dialog can show the
/// pixel size the user is about to get.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ExportPlan {
    /// The canvas-space rect being exported.
    pub min: stark_model::geom::Vec2,
    pub max: stark_model::geom::Vec2,
    /// Output size in image px.
    pub size: Extent2,
    /// Image px per canvas px.
    pub zoom: f32,
}

impl ExportPlan {
    /// The view this plan renders through: centred on the rect, at `zoom` = its scale,
    /// with the plan's pixel size as the viewport.
    ///
    /// Both consumers of a planned rect — [`Engine::export`] and
    /// [`Engine::render_into`] — derive their view here, so the two cannot disagree
    /// about the framing.
    pub fn view(&self) -> ViewTransform {
        ViewTransform {
            center: (self.min + self.max) * 0.5,
            zoom: self.zoom,
            // Upright and unmirrored, whatever angle the artist has the canvas at:
            // turning the easel is a way of *looking* at the piece, and a file — or
            // the navigator's overview — shows the piece rather than the easel
            // (§18.1.2).
            rotation: 0.0,
            flip_h: false,
            viewport: self.size,
        }
    }
}

/// A flat 2-D texture to render into off-screen: an export's target, or the HDR
/// probe's. Everything but the label, the format, the size and the usage is the same
/// for both: a single mip, a single sample, no view formats.
fn offscreen_target(
    gpu: &GpuContext,
    label: &str,
    format: wgpu::TextureFormat,
    size: Extent2,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size.width,
            height: size.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

/// The usage every offscreen target on this path is made with: drawn into, then copied
/// out.
const OFFSCREEN_USAGE: wgpu::TextureUsages =
    wgpu::TextureUsages::RENDER_ATTACHMENT.union(wgpu::TextureUsages::COPY_SRC);

/// What the two [`Probe`](crate::testing::Probe) readbacks render through, **kept
/// across calls** (§9): pass A's attachments and the texture they resolve into.
///
/// **Why the memory is parked here and not inside [`Offscreen`].** `Offscreen`'s
/// bargain is that a one-shot must not be parked — an 8192-px export's attachments are
/// half a gigabyte, and leaving them resident for the session is worse than paying for
/// them twice. The probes are the opposite case: a suite calls them ~1000 times at one
/// steady viewport, and building the accumulator trio afresh each time costs an order
/// of magnitude more than the frame it then composites. Keeping it *behind the probe*
/// settles the two at once, and it is sound only there: these are `testing`-only entry
/// points, so the parked memory belongs to a test harness and no app path can reach it.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
pub(super) struct ProbeTarget {
    /// Pass A's attachments, kept unconditionally: a
    /// [`Compositor`](crate::gpu::composite::Compositor) resizes its own on a change of
    /// target, and follows a colour-space rebuild.
    attachments: Offscreen,
    target: Option<ProbeTexture>,
}

/// The kept readback target, with the two things that invalidate it.
#[cfg(not(target_arch = "wasm32"))]
struct ProbeTexture {
    format: wgpu::TextureFormat,
    size: Extent2,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

#[cfg(not(target_arch = "wasm32"))]
impl ProbeTarget {
    /// The attachments, the target and its view, ready for a `size` render into
    /// `format` — the target rebuilt if and only if one of those two has moved, which
    /// is all that can make it wrong.
    ///
    /// Three borrows off one `&mut self` because a render needs the attachments
    /// mutably while it draws into the view beside them.
    fn parts(
        &mut self,
        gpu: &GpuContext,
        format: wgpu::TextureFormat,
        size: Extent2,
    ) -> (&mut Offscreen, &wgpu::Texture, &wgpu::TextureView) {
        let Self {
            attachments,
            target,
        } = self;
        if target
            .as_ref()
            .is_none_or(|t| t.format != format || t.size != size)
        {
            *target = None;
        }
        let kept = target.get_or_insert_with(|| {
            let texture =
                offscreen_target(gpu, "stark probe target", format, size, OFFSCREEN_USAGE);
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            ProbeTexture {
                format,
                size,
                texture,
                view,
            }
        });
        (attachments, &kept.texture, &kept.view)
    }
}

/// Which display a probe's frame is composited for — the only thing the two differ in
/// beyond the format they read (§6.5).
#[cfg(not(target_arch = "wasm32"))]
#[derive(Copy, Clone)]
enum ProbeDisplay {
    /// SDR whatever the screen is showing, as any export is (§15.6).
    Export,
    /// The screen's own display, which for the half-float read is the whole point.
    Screen,
}

/// How much of the viewport [`Engine::show_piece`] leaves clear around the piece, as
/// a fraction of each axis on each side. Not zero, unlike the fit an *export* makes:
/// a piece flush with all four window edges reads as one that carries on past them.
const SHOW_PIECE_MARGIN: f32 = 0.04;

impl Engine {
    /// Render the current canvas (preview if stroking, else committed) into
    /// `target`, through this engine's own pan/zoom (§6.4).
    pub fn render(&mut self, target: &wgpu::TextureView) {
        self.render_view(target, RenderSpec::screen(self.view), Attachments::Screen);
    }

    /// Render the document through `view` into a target that is **not** the engine's
    /// own substrate — a second substrate showing the same document (§11), the
    /// navigator's miniature being the consumer.
    ///
    /// `into` holds the pass-A attachments (see [`Offscreen`]); a consumer drawing
    /// repeatedly keeps them, so a refresh allocates nothing at all. `target` must
    /// carry the format [`target_format`](Self::target_format) reports and be
    /// `view.viewport` in size. No chrome: a selection outline belongs to the
    /// substrate you are painting on, not to a thumbnail of the piece.
    pub fn render_into(
        &mut self,
        into: &mut Offscreen,
        target: &wgpu::TextureView,
        view: ViewTransform,
        background: Background,
        content: Rendered,
    ) {
        self.render_view(
            target,
            RenderSpec {
                view,
                only: None,
                background,
                chrome: Chrome::Hidden,
                content,
            },
            Attachments::Surface(into),
        );
    }

    /// The texture format this engine's pipelines render to. A frontend configuring a
    /// second substrate for [`render_into`](Self::render_into) has to match it.
    pub fn target_format(&self) -> wgpu::TextureFormat {
        self.shared.target_format
    }

    /// Gather what a frame is made of — the document, its draw list and the chrome
    /// over it — and hand it to the
    /// [`Presenter`](super::presenter::Presenter) to composite.
    ///
    /// The seam export needs: exporting a frame is rendering at `frame.rect × scale`,
    /// centred on the frame, at `zoom = scale` — **the same path the screen takes**,
    /// so what is written is what was seen.
    ///
    /// Private, with [`Engine::export`] and [`Engine::render_into`] as the two
    /// consumers: what a caller may choose is a [`RenderSpec`] and where the attachments
    /// live, never whether chrome is drawn (it is, for the screen alone).
    fn render_view(
        &mut self,
        target: &wgpu::TextureView,
        spec: RenderSpec,
        attachments: Attachments<'_>,
    ) {
        // Everything a painted frame costs on the CPU, from the fold through to the
        // last command encoded. The frontend's own `frame` span sits outside it and
        // adds the substrate acquire and the present, so the difference between the
        // two rows is what the *page* costs on top of what the engine does.
        crate::timing::span!("render.view");
        // The fold is rebuilt lazily (`Engine::mark_live_stale`), and this is the read
        // that services it: once per frame painted, whatever arrived since.
        if matches!(spec.content, Rendered::Live) {
            self.flush_live();
        }
        // **Borrowed, not cloned.** The document is read off two fields, and the
        // compositor is a third — so the frame holds the very `DocState` the timeline
        // holds, where `Engine::presented` would have borrowed all of `self` and left
        // nothing to composite with.
        let doc = match spec.content {
            Rendered::Live => self.preview.presented(self.timeline.current()),
            Rendered::Committed => self.timeline.current(),
        };
        // Only what this view can show (§6.3); the list is otherwise every populated
        // tile of every visible layer, whatever the viewport. Rebuilt only when
        // something it is a function of has moved (`DrawKey`).
        let key = DrawKey {
            doc_revision: self.doc_revision,
            epoch: self.preview.epoch(),
            fold: self.preview.fold(),
            content: spec.content,
            only: spec.only,
            visible: spec.view.visible_tiles(),
        };
        // Instrumented because the cache is the whole claim: this row's *count* against
        // `render.view`'s says how often the key actually moved, so a rebuild that
        // stopped being rare shows up here before it shows up as a dropped frame.
        // Braced because a timing span runs to the end of the block it is opened in.
        let groups = {
            crate::timing::span!("render.drawlist");
            self.presenter
                .draw_list(key, doc, self.shared.color_space.as_ref())
        };

        // Chrome never reaches a file: an exported image gets no selection outline
        // (§15.6). Keyed on `chrome`, deliberately *not* on the background — a
        // substrate export is still an export.
        let outlines: Vec<(crate::document::Selection, Option<[f32; 3]>)> = match spec.chrome {
            Chrome::Hidden => Vec::new(),
            Chrome::Shown => self.visible_selections(doc),
        };
        let outlines: Vec<SelectionOutline<'_>> = outlines
            .iter()
            .map(|(selection, tint)| SelectionOutline {
                selection,
                tint: *tint,
            })
            .collect();
        // Chrome, on the outlines' argument: a guide is a thing to draw *with*, so an
        // export or a miniature never carries one (§20.4). Derived fresh per render —
        // the camera math is a handful of products, and a cached copy would shadow
        // the session's state.
        let guide_scenes: Vec<GuideScene> = match spec.chrome {
            Chrome::Hidden => Vec::new(),
            // The document holds the guides and the session holds whose eye is shut
            // (§20.5), combined once in `Session::shown_guides`. The cursor is
            // per-client for the same reason — the rays drawn through every guide are
            // handed in rather than known to a camera (§20.9) — and is the one the
            // wire carries, so what a peer sees the guides through is what this client
            // does.
            Chrome::Shown => {
                let cursor = self.presence.cursor();
                self.session
                    .shown_guides(doc)
                    .map(|g| g.camera.scene(cursor))
                    .collect()
            }
        };
        let scene = CompositeScene {
            // Document state (§15.5), so the substrate a piece was painted on travels
            // with it rather than with whichever frontend renders it.
            substrate: self.shared.color_space.rgb_to_latent(doc.substrate_color()),
            // Off `doc`, which is the *previewed* document when one is up — so an
            // unlogged scale drag re-lights the substrate at pointer rate (§6.4).
            substrate_uv_scale: doc.substrate().uv_scale(),
            groups: &groups,
            outlines: &outlines,
            transparent: spec.background == Background::Transparent,
            guides: &guide_scenes,
            // What display the picture is for (§6.5): the screen's own setting for the
            // screen and a surface beside it, SDR for anything bound for a file — an
            // export never reads the screen's.
            output: match attachments {
                Attachments::Screen | Attachments::Surface(_) => self.presenter.output(),
                Attachments::Export(_) => Output::SDR,
            },
        };
        self.presenter
            .present(target, spec.view, scene, attachments);
    }

    /// [`Probe::render_to_image`](crate::testing::Probe::render_to_image): the
    /// export path, read back blocking. WebGPU has no blocking poll (`gpu::readback`).
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn render_to_image(&mut self) -> RgbaImage {
        // 8-bit whatever the screen is (§15.6): the compositor renders an 8-bit target
        // SDR, so this format and `ProbeDisplay::Export` together are what make it one.
        let format = crate::gpu::composite::export_format(self.shared.target_format);
        self.probe_frame(format, ProbeDisplay::Export, |gpu, target, size| {
            let pixels = crate::device::readback::read_rgba8_blocking(gpu, target, size);
            RgbaImage::from_target_bytes(size.width, size.height, pixels, format)
        })
    }

    /// [`Probe::render_to_floats`](crate::testing::Probe::render_to_floats): the
    /// screen's passes, read back blocking.
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn render_to_floats(&mut self) -> Vec<f32> {
        let format = self.shared.target_format;
        assert_eq!(
            format,
            wgpu::TextureFormat::Rgba16Float,
            "render_to_floats reads a half-float screen, and this engine presents to {format:?}",
        );
        self.probe_frame(format, ProbeDisplay::Screen, |gpu, target, size| {
            crate::device::readback::read_rgba16f_blocking(gpu, target, size)
        })
    }

    /// The two probes' shared body: the screen's own frame — same view, same
    /// background, same chrome — into the [`ProbeTarget`] kept on the presenter, then
    /// `read` over it (§9).
    ///
    /// The target comes out of the presenter and goes back in around the render, since
    /// that render borrows the engine whole
    /// ([`take_probe`](super::presenter::Presenter::take_probe)).
    #[cfg(not(target_arch = "wasm32"))]
    fn probe_frame<T>(
        &mut self,
        format: wgpu::TextureFormat,
        display: ProbeDisplay,
        read: impl FnOnce(&GpuContext, &wgpu::Texture, Extent2) -> T,
    ) -> T {
        let spec = RenderSpec::screen(self.view);
        let size = spec.view.viewport;
        let mut kept = self.presenter.take_probe();
        let (attachments, target, target_view) = kept.parts(&self.shared.gpu, format, size);
        let attachments = match display {
            ProbeDisplay::Export => Attachments::Export(attachments),
            ProbeDisplay::Screen => Attachments::Surface(attachments),
        };
        self.render_view(target_view, spec, attachments);
        let out = read(&self.shared.gpu, target, size);
        self.presenter.keep_probe(kept);
        out
    }

    /// Render through an explicit view into a freshly allocated offscreen texture,
    /// ready to be read back — an export's half of what
    /// [`probe_frame`](Self::probe_frame) does with a kept one.
    fn render_offscreen(
        &mut self,
        into: &mut Offscreen,
        spec: RenderSpec,
    ) -> (wgpu::Texture, Extent2) {
        let size = spec.view.viewport;
        // 8-bit whatever the screen is (§15.6): the compositor renders an 8-bit target
        // SDR, so this format is the whole of what makes an export one.
        let target = offscreen_target(
            &self.shared.gpu,
            "stark export target",
            crate::gpu::composite::export_format(self.shared.target_format),
            size,
            OFFSCREEN_USAGE,
        );
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
        // The caller's attachments, not the substrate's — see [`Attachments`].
        self.render_view(&target_view, spec, Attachments::Export(into));
        (target, size)
    }

    /// What exporting `frame` at `scale` would produce, without producing it —
    /// so a dialog can show the pixel size before committing to the render.
    ///
    /// `frame` names a **matte layer** whose rect is the piece
    /// (§15.6). With no frame it falls back to the painted bounds, and on an empty
    /// canvas to the current viewport, so export always has *something* to mean.
    pub fn export_plan(&self, frame: Option<LayerId>, scale: ExportScale) -> Result<ExportPlan> {
        let (min, max) = self.export_rect(frame);
        let (w, h) = (max.x - min.x, max.y - min.y);
        if !(w.is_finite() && h.is_finite()) || w < 1.0 || h < 1.0 {
            return Err(ExportError::TooSmall {
                width: w,
                height: h,
            }
            .into());
        }
        let zoom = match scale {
            ExportScale::Factor(f) => f,
            ExportScale::Width(px) => px as f32 / w,
            ExportScale::Fit(into) => (into.width as f32 / w).min(into.height as f32 / h),
        };
        if !(zoom.is_finite() && zoom > 0.0) {
            return Err(ExportError::BadScale.into());
        }
        // Round rather than truncate, so a 1× export of a 100.5-px frame is 101
        // rather than silently dropping most of a pixel off two edges.
        let size = Extent2::new(
            (w * zoom).round().max(1.0) as u32,
            (h * zoom).round().max(1.0) as u32,
        );
        let limit = self.presenter.max_dim();
        if size.width > limit || size.height > limit {
            return Err(ExportError::OverLimit {
                what: Produces::Export,
                size,
                limit,
            }
            .into());
        }
        Ok(ExportPlan {
            min,
            max,
            size,
            zoom,
        })
    }

    /// Render a frame to a CPU-side image (§15.6).
    ///
    /// **The same path the screen takes** — every visible layer composited through
    /// the media pass — with the view centred on the frame at `zoom = scale`. Nothing
    /// is special-cased: a frame matte covers only *outside* its rect, which is
    /// clipped away here, so it contributes nothing to its own export, while a
    /// substrate matte is inside and contributes exactly what it should.
    ///
    /// Renders immediately and returns a future for the **readback**, which is the
    /// only asynchronous part (§7 — on WebGPU `mapAsync` settles only when the
    /// browser's event loop runs, so there is no way to block on it).
    ///
    /// Deliberately *not* an `async fn`: one would hold `&mut self` across the
    /// readback, and a frontend taking that borrow from a shared cell would panic with
    /// `AlreadyBorrowedMut` when the UI re-rendered mid-await. The future owns a cloned
    /// [`GpuContext`] and the target texture and
    /// touches the engine not at all, so the borrow ends when `export` returns.
    ///
    /// `content` chooses whether the in-flight gesture is in the picture: a file
    /// export takes [`Rendered::Live`], since that is what the artist is looking at.
    /// Anything refreshed per *committed* change wants [`Rendered::Committed`] and
    /// probably [`render_into`](Self::render_into).
    ///
    /// `into` is where the render's attachments live — **beside** the substrate rather
    /// than in it, so the screen's are never touched; whether its own outlive the call
    /// is the caller's choice (see [`Offscreen`]).
    ///
    /// **Two `Result`s, and they answer different questions.** The outer one is the
    /// request: a frame too small, a size past the device's limit — refused before
    /// anything is drawn, and answerable by asking for something else. The inner one
    /// is the *readback*, which can only fail by the GPU failing underneath it (§5),
    /// and is reported rather than panicked on because the action log survives what
    /// the device does not.
    ///
    /// ```text
    /// let readback = { engine.write().export(&mut own, frame, scale, bg, content)? }; // borrow ends
    /// let image = readback.await?;
    /// ```
    pub fn export(
        &mut self,
        into: &mut Offscreen,
        frame: Option<LayerId>,
        scale: ExportScale,
        background: Background,
        content: Rendered,
    ) -> Result<impl std::future::Future<Output = Result<RgbaImage>> + use<>> {
        // Ahead of the render, so a frame too small or too large to export is
        // refused before anything is drawn — and with the message that names the
        // *frame*, since `export_view`'s checks are then satisfied by construction.
        let plan = self.export_plan(frame, scale)?;
        self.export_view(into, plan.view(), None, background, content)
    }

    /// Render through an **explicit** view to a CPU-side image —
    /// [`export`](Self::export) with the framing chosen by the caller instead of
    /// derived from the document.
    ///
    /// This takes the view whole — `view.viewport` is the output size — for a caller
    /// that means *this* rect at *this* pixel size, such as a preset thumbnail. The
    /// same borrow bargain as `export` applies: the returned future owns what it
    /// reads, so drop the engine borrow before awaiting.
    ///
    /// No chrome, like every render that is not the screen's (§15.6).
    ///
    /// `only` names a layer to render **alone** — its own paint, with its blend mode,
    /// clip and opacity dropped, making the result an identity card for the layer
    /// rather than a picture of its contribution (see `RenderSpec`'s `only`). The
    /// layer panel's thumbnails are the consumer (§14.6); `None` renders the
    /// document. A layer that is hidden or fully transparent draws nothing at all, so
    /// a caller that would rather keep its last picture than show a blank one should
    /// ask before rendering.
    ///
    /// Errors mirror [`export_plan`](Self::export_plan)'s: a degenerate or non-finite
    /// view, or a viewport past the device's texture limit.
    pub fn export_view(
        &mut self,
        into: &mut Offscreen,
        view: ViewTransform,
        only: Option<LayerId>,
        background: Background,
        content: Rendered,
    ) -> Result<impl std::future::Future<Output = Result<RgbaImage>> + use<>> {
        // The same question the view's own mutators ask before storing anything
        // (`ViewTransform::usable`): a caller-supplied view has not passed through
        // them, so it is asked at the door it comes in by.
        if !view.usable() {
            return Err(ExportError::UnusableView.into());
        }
        let size = view.viewport;
        let limit = self.presenter.max_dim();
        if size.width == 0 || size.height == 0 || size.width > limit || size.height > limit {
            return Err(ExportError::OverLimit {
                what: Produces::Render,
                size,
                limit,
            }
            .into());
        }
        // The hover mark is a hypothesis about the *next* stroke rather than work
        // (§18.1.10), so like any other chrome it may never reach a file. Dropped
        // from the fold rather than excluded per-render, because the fold is one
        // cached document; the next hover report re-seeds it.
        if matches!(content, Rendered::Live) && self.session.clear_hover() {
            self.mark_live_stale();
        }
        let (target, size) = self.render_offscreen(
            into,
            RenderSpec {
                view,
                only,
                background,
                content,
                chrome: Chrome::Hidden,
            },
        );
        // Captured, not read through `self`: the future deliberately does not
        // borrow the engine.
        let gpu = self.shared.gpu.clone();
        let format = target.format();
        Ok(async move {
            let pixels = crate::device::readback::read_rgba8(&gpu, &target, size).await?;
            Ok(RgbaImage::from_target_bytes(
                size.width,
                size.height,
                pixels,
                format,
            ))
        })
    }

    /// The canvas-space rect an export covers: the named frame, else the painted
    /// bounds, else the viewport.
    fn export_rect(
        &self,
        frame: Option<LayerId>,
    ) -> (stark_model::geom::Vec2, stark_model::geom::Vec2) {
        self.piece_rect(frame).unwrap_or_else(|| {
            // Everything the viewport shows — a *bound* under rotation, which is the
            // safe direction: an export with nothing painted and no frame should not
            // crop tighter than what the artist is looking at.
            self.view.visible_bounds()
        })
    }

    /// The canvas-space rect that **is the piece**: the named frame's, else the
    /// painted bounds, else `None` (§15.6).
    ///
    /// The rule without the last resort, because its two askers want different ones:
    /// an export has to write *something*, so it frames what you are looking at, while
    /// [`ViewCommand::ShowPiece`](crate::command::ViewCommand::ShowPiece) on a document
    /// with neither paint nor frame should leave the view alone rather than zoom the
    /// window onto itself. Shared so the two cannot disagree about where the piece
    /// ends.
    pub(super) fn piece_rect(
        &self,
        frame: Option<LayerId>,
    ) -> Option<(stark_model::geom::Vec2, stark_model::geom::Vec2)> {
        let doc = self.timeline.current();
        // An `Everything` matte has no rect and so defines no frame: naming one falls
        // through to the painted bounds — a substrate is under the picture, not a crop
        // of it (§15.6). The rect is placed by the layer's translation (§14.12), so
        // export frames the hole where the compositor draws it.
        if let Some(id) = frame
            && let Some(rect) = doc
                .layer(id)
                .and_then(|l| l.matte_region()?.translated(l.translation.as_vec2()).rect())
        {
            return Some(rect);
        }
        let (min, max) = doc.bounds().tile_range()?;
        let t = stark_model::geom::TILE_SIZE as f32;
        Some((
            stark_model::geom::Vec2::new(min.x as f32 * t, min.y as f32 * t),
            stark_model::geom::Vec2::new((max.x + 1) as f32 * t, (max.y + 1) as f32 * t),
        ))
    }

    /// Put the whole piece on screen: the rect an export of `frame` would write,
    /// centred and fitted to the viewport (§15.6) —
    /// [`ViewCommand::ShowPiece`](crate::command::ViewCommand::ShowPiece).
    ///
    /// A document with nothing painted and no frame does not move the view: there is
    /// no piece to show yet, and the honest answer to "show me it" is to stay put.
    pub(super) fn show_piece(&mut self, frame: Option<LayerId>) {
        if let Some((min, max)) = self.piece_rect(frame) {
            self.view.show_rect(min, max, SHOW_PIECE_MARGIN);
        }
    }

    /// The selection masks to outline, and whose each is (§17.3).
    ///
    /// `DocState` holds a selection for every actor that ever made one, since replay
    /// needs them all: the log decides what exists, presence what could be shown, and
    /// `show_peer_selections` whether it is — a second contour over the artwork being
    /// a preference rather than a fact about the drawing.
    fn visible_selections(
        &self,
        doc: &DocState,
    ) -> Vec<(crate::document::Selection, Option<[f32; 3]>)> {
        let mut out = Vec::new();
        let mine = doc.selection_of(self.actor());
        if mine.is_active() {
            out.push((mine, None));
        }
        if self.show_peer_selections {
            for peer in self.peers.iter() {
                let theirs = doc.selection_of(peer.actor);
                if theirs.is_active() {
                    out.push((theirs, Some(peer.color)));
                }
            }
        }
        out
    }
}
