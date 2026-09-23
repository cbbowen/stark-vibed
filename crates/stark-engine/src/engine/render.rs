//! Presenting the canvas: the screen frame and export (§6.3, §15.6). What is *in*
//! a frame is the draw list's (`engine::draw_list`); this is where one is asked for,
//! cached, and composited.
//!
//! **One path serves all three consumers**, which is what keeps them from
//! disagreeing about what the document looks like. [`Engine::render_view`] takes a
//! view, a substrate and somewhere to put the pass-A attachments, and every caller
//! differs only in those: the screen renders through the session's view with chrome,
//! the navigator's miniature through a planned rect without it, and an export through
//! the same planned rect into a texture it then reads back.

use super::Engine;
use super::draw_list::DrawList;
use crate::Result;
use crate::document::DocState;
use crate::error::{ExportError, Produces};
use crate::gpu::composite::{
    CompositeGroup, CompositeScene, CompositorPipeline, Offscreen, Output, SelectionOutline,
};
use crate::image::RgbaImage;
use crate::view::{Extent2, ViewTransform};
use stark_model::document::{GuideScene, LayerId};
use stark_model::geom::TileRect;
use std::sync::Arc;

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

/// Which [`Compositor`](crate::gpu::composite::Compositor) a render's offscreen
/// attachments come from: the screen's own, or the caller's — one per target, since
/// sharing one would resize the other's away.
enum Attachments<'a> {
    /// The screen's own, cached across frames ([`Engine::compositor`]).
    Screen,
    /// The caller's, for a second surface on the same screen — the navigator's
    /// miniature — presented as the screen is (§6.5). Whether they outlive the call
    /// is the caller's choice, since only it knows whether the render repeats — see
    /// [`Offscreen`].
    Surface(&'a mut Offscreen),
    /// The caller's, for a picture bound for a file or the CPU: an export, a
    /// thumbnail, a golden. Rendered [`Output::SDR`] whatever the screen is showing,
    /// which keeps the screen's headroom and gamut out of a file (§6.5, §15.6).
    Export(&'a mut Offscreen),
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
    /// than [`max_export_dim`] would fail *that* query for a render it was never
    /// going to make.
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

/// Largest exported edge, in px: **the device's own texture limit**, so a stray
/// zero-ish frame or a huge scale is reported as an error rather than surfacing as a
/// wgpu validation panic.
///
/// Asked of the device rather than fixed at a literal, because there is no one
/// number: the frontend requests `wgpu::Limits::default()` while the headless device
/// ([`GpuContext::headless`]) asks only for the engine's own minimums
/// ([`MAX_TEXTURE_DIM_2D`](crate::gpu::context::MAX_TEXTURE_DIM_2D)), so any literal
/// agrees with one of them and not the other. Asked once, by the compositor.
///
/// [`GpuContext::headless`]: crate::gpu::context::GpuContext::headless
fn max_export_dim(compositor: &CompositorPipeline) -> u32 {
    compositor.max_dim()
}

/// How much of the viewport [`Engine::show_piece`] leaves clear around the piece, as
/// a fraction of each axis on each side. Not zero, unlike the fit an *export* makes:
/// a piece flush with all four window edges reads as one that carries on past them.
const SHOW_PIECE_MARGIN: f32 = 0.04;

impl Engine {
    /// Render the current canvas (preview if stroking, else committed) into
    /// `target`, through this engine's own pan/zoom (§6.4).
    pub fn render(&mut self, target: &wgpu::TextureView) {
        self.render_view(
            target,
            self.view,
            None,
            Background::Substrate,
            Chrome::Shown,
            Rendered::Live,
            Attachments::Screen,
        );
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
            view,
            None,
            background,
            Chrome::Hidden,
            content,
            Attachments::Surface(into),
        );
    }

    /// The texture format this engine's pipelines render to. A frontend configuring a
    /// second substrate for [`render_into`](Self::render_into) has to match it.
    pub fn target_format(&self) -> wgpu::TextureFormat {
        self.shared.target_format
    }

    /// Render through an **explicit** view rather than the session's, choosing what
    /// sits under the paint and whether on-canvas chrome is drawn (§6.4, §15.6).
    ///
    /// The seam export needs: exporting a frame is rendering at `frame.rect × scale`,
    /// centred on the frame, at `zoom = scale` — **the same path the screen takes**,
    /// so what is written is what was seen.
    ///
    /// `only` names a single layer to draw **alone**, its blend mode, clip and opacity
    /// dropped — see [`DrawList::layer_alone`], which settles what that means
    /// (§18.0.2). `None` is the document, which is every render but a layer
    /// thumbnail's (§14.6).
    ///
    /// Private, with [`Engine::export`] and [`Engine::render_into`] as the two
    /// consumers: what a caller may choose is a view, how much of the document, a
    /// substrate and where the attachments live, never whether chrome is drawn (it is,
    /// for the screen alone).
    #[expect(
        clippy::too_many_arguments,
        reason = "every argument is a distinct type, so the transposition the lint guards cannot be written"
    )]
    fn render_view(
        &mut self,
        target: &wgpu::TextureView,
        view: ViewTransform,
        only: Option<LayerId>,
        background: Background,
        chrome: Chrome,
        content: Rendered,
        attachments: Attachments<'_>,
    ) {
        // Everything a painted frame costs on the CPU, from the fold through to the
        // last command encoded. The frontend's own `frame` span sits outside it and
        // adds the substrate acquire and the present, so the difference between the
        // two rows is what the *page* costs on top of what the engine does.
        crate::timing::span!("render.view");
        // The fold is rebuilt lazily (`Engine::mark_live_stale`), and this is the read
        // that services it: once per frame painted, whatever arrived since.
        if matches!(content, Rendered::Live) {
            self.flush_live();
        }
        // Cloned rather than borrowed — a handful of `Arc` bumps (§5.1) — to release
        // the borrow of `self`: the draw list is rebuilt through `&mut self` and the
        // compositor is borrowed mutably at the end.
        let doc = match content {
            Rendered::Live => self.presented().clone(),
            Rendered::Committed => self.timeline.current().clone(),
        };
        // Only what this view can show (§6.3); the list is otherwise every populated
        // tile of every visible layer, whatever the viewport. Rebuilt only when
        // something it is a function of has moved (`DrawKey`).
        //
        // Instrumented because the cache is the whole claim: this row's *count*
        // against `render.view`'s says how often the key actually moved, so a rebuild
        // that stopped being rare shows up here before it shows up as a dropped frame.
        // Braced because a timing span runs to the end of the block it is opened in.
        let key = DrawKey {
            doc_revision: self.doc_revision,
            epoch: self.preview.epoch(),
            fold: self.preview.fold(),
            content,
            only,
            visible: view.visible_tiles(),
        };
        // An owned handle rather than a borrow out of the memo, so the compositor can
        // be borrowed mutably at the end without the list being copied out of the way
        // first — why `Engine::draw_list` hands back a share.
        let groups = {
            crate::timing::span!("render.drawlist");
            self.draw_list(key, &doc)
        };

        // Chrome never reaches a file: an exported image gets no selection outline
        // (§15.6). Keyed on `chrome`, deliberately *not* on the background — a
        // substrate export is still an export. Read off the owned `doc`, so the
        // compositor can be borrowed mutably at the end.
        let outlines: Vec<(crate::document::Selection, Option<[f32; 3]>)> = match chrome {
            Chrome::Hidden => Vec::new(),
            Chrome::Shown => self.visible_selections(&doc),
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
        let guide_scenes: Vec<GuideScene> = match chrome {
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
                    .shown_guides(&doc)
                    .map(|g| g.camera.scene(cursor))
                    .collect()
            }
        };
        // What display the picture is for (§6.5): the screen's own setting for the
        // screen and a surface beside it, SDR for anything bound for a file — an
        // export never reads the screen's.
        let output = match attachments {
            Attachments::Screen | Attachments::Surface(_) => self.compositor_pipeline.output(),
            Attachments::Export(_) => Output::SDR,
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
            transparent: background == Background::Transparent,
            guides: &guide_scenes,
            output,
        };
        // The three compositing passes and the draws inside them, encoded and
        // submitted (§6.3). CPU time to *record* them: WebGPU offers no timestamp
        // query on the web, so nothing here can say what the GPU then spent executing
        // them — the frontend's `Renderer::gpu_behind` is that signal.
        crate::timing::span!("render.composite");
        match attachments {
            Attachments::Screen => {
                self.compositor
                    .render(&self.compositor_pipeline, target, view, scene)
            }
            Attachments::Surface(into) | Attachments::Export(into) => into
                .get(&self.compositor_pipeline)
                .render(&self.compositor_pipeline, target, view, scene),
        }
    }

    /// [`Probe::render_to_image`](crate::testing::Probe::render_to_image): the
    /// export path, read back blocking. WebGPU has no blocking poll (`gpu::readback`).
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn render_to_image(&mut self) -> RgbaImage {
        // One render per call, so nothing is kept: the attachments are allocated here
        // and dropped with this `Offscreen`.
        let (target, size) = self.render_offscreen(
            &mut Offscreen::default(),
            self.view,
            None,
            Background::Substrate,
            Chrome::Shown,
            Rendered::Live,
        );
        let pixels = crate::gpu::readback::read_rgba8_blocking(&self.shared.gpu, &target, size);
        RgbaImage::from_target_bytes(size.width, size.height, pixels, target.format())
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
        let view = self.view;
        let size = view.viewport;
        let target = self.offscreen_target(
            "stark hdr probe target",
            format,
            size,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
        self.render_view(
            &target_view,
            view,
            None,
            Background::Substrate,
            Chrome::Shown,
            Rendered::Live,
            // The screen's own display, which is the whole of what this reads back.
            Attachments::Surface(&mut Offscreen::default()),
        );
        crate::gpu::readback::read_rgba16f_blocking(&self.shared.gpu, &target, size)
    }

    /// Render through an explicit view into an offscreen texture, ready to be read
    /// back. Split out so the blocking and async readbacks share every step but
    /// the wait.
    fn render_offscreen(
        &mut self,
        into: &mut Offscreen,
        view: ViewTransform,
        only: Option<LayerId>,
        background: Background,
        chrome: Chrome,
        content: Rendered,
    ) -> (wgpu::Texture, Extent2) {
        let size = view.viewport;
        // 8-bit whatever the screen is (§15.6): the compositor renders an 8-bit target
        // SDR, so this format is the whole of what makes an export one.
        let target = self.offscreen_target(
            "stark export target",
            crate::gpu::composite::export_format(self.shared.target_format),
            size,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
        // The caller's attachments, not the substrate's — see [`Attachments`].
        self.render_view(
            &target_view,
            view,
            only,
            background,
            chrome,
            content,
            Attachments::Export(into),
        );
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
        let limit = max_export_dim(&self.compositor_pipeline);
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
    /// [`GpuContext`](crate::gpu::context::GpuContext) and the target texture and
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
    /// rather than a picture of its contribution ([`DrawList::layer_alone`] settles
    /// what that means). The
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
        let limit = max_export_dim(&self.compositor_pipeline);
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
        let (target, size) =
            self.render_offscreen(into, view, only, background, Chrome::Hidden, content);
        // Captured, not read through `self`: the future deliberately does not
        // borrow the engine.
        let gpu = self.shared.gpu.clone();
        let format = target.format();
        Ok(async move {
            let pixels = crate::gpu::readback::read_rgba8(&gpu, &target, size).await?;
            Ok(RgbaImage::from_target_bytes(
                size.width,
                size.height,
                pixels,
                format,
            ))
        })
    }

    /// [`DrawList`], **memoized**: the draw list for `key`, built afresh or taken from
    /// [`Engine::draw_cache`] (C4).
    ///
    /// **Why the cache earns its place.** Building the list clones a
    /// `TilePairHandle` per visible tile, per layer, plus a `Vec` per layer, and the
    /// visible tile count scales as 1/zoom² — so a zoomed-out multi-layer document
    /// pays ~10⁵ atomic bumps a frame to produce a list identical to the last one.
    ///
    /// **A single-layer list is never cached.** The memo holds one list, so a
    /// thumbnail pass (one layer per row, interleaved with screen frames) would evict
    /// the screen's list N times over — and such a key can never be hit twice anyway,
    /// every row naming a different `only`.
    ///
    /// Takes the key and a borrowed `doc` so the caller can compute one while it still
    /// holds the other. The list comes back as an `Arc` rather than a reference into
    /// the memo, so holding it does not hold a borrow of the engine.
    pub(super) fn draw_list(&self, key: DrawKey, doc: &DocState) -> Arc<[CompositeGroup]> {
        let build = || {
            let walk = DrawList::new(self.shared.color_space.as_ref(), key.visible);
            match key.only {
                Some(id) => walk.layer_alone(doc, id),
                None => walk.document(doc),
            }
            .into()
        };
        if key.only.is_some() {
            return build();
        }
        self.draw_cache.get_or_build(key, build)
    }

    /// A flat 2-D texture to render into off-screen: an export's target, or the HDR
    /// probe's. Everything but the label, the format, the size and the usage is the
    /// same for both: a single mip, a single sample, no view formats.
    pub(super) fn offscreen_target(
        &self,
        label: &str,
        format: wgpu::TextureFormat,
        size: Extent2,
        usage: wgpu::TextureUsages,
    ) -> wgpu::Texture {
        self.shared
            .gpu
            .device
            .create_texture(&wgpu::TextureDescriptor {
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

/// What a built draw list is a function of — the whole of it, which is what makes
/// caching one sound (§6.3, C4). See [`Memo`](super::Memo) for the rule this is one
/// of three keys under.
///
/// Every term is already counted for another reason, and that is the point: nothing
/// here is a new notion of "has it changed", only the existing ones read together.
///
/// - `doc_revision` moves on every **committed** change — a commit, an undo, a
///   merged remote action, a load.
/// - `epoch` moves whenever the document the previews are drawn over is *replaced*:
///   an unlogged drag preview installed or dropped, and a commit. So it is *wider*
///   than `doc_revision` rather than a second, narrower term; both are named because
///   a key names the terms its value depends on, not the smallest set that happens to
///   cover them.
/// - `fold` moves whenever the live fold is built or dropped (`Preview::rebuild`); a
///   rebuild with nothing to drop leaves it alone. A stroke in flight commits nothing
///   and replaces no document, so neither counter above stirs while one is being
///   drawn, and a list keyed without this would hold the frame at the moment the pen
///   went down. It is the term the two roster keys do not need: they project what is
///   *shown*, and the fold is not that (`super::ShownKey`).
/// - `content` is which document is being drawn at all. `Live` and `Committed` differ
///   by exactly the in-flight gesture, so at one instant they are two different lists.
/// - `only` and `visible` are the two arguments that shape the list.
///
/// **What is deliberately absent is the view.** A draw list is built against the
/// *tile rect* a view can reach, not the view, so panning within one tile — or
/// rotating, or supersampling — hits the same key.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct DrawKey {
    doc_revision: u64,
    epoch: u64,
    fold: u64,
    content: Rendered,
    only: Option<LayerId>,
    visible: Option<TileRect>,
}
