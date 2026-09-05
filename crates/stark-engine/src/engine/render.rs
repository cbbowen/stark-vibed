//! Presenting the canvas: the compositor's draw list, the screen frame, and export
//! (§6.3, §15.6).
//!
//! One path serves all three consumers, which is what keeps them from disagreeing
//! about what the document looks like. [`Engine::render_view`] takes a view, a
//! substrate and somewhere to put the pass-A attachments, and every caller differs only
//! in those: the screen renders through the session's view with chrome, the
//! navigator's miniature through a planned rect without it, and an export through
//! the same planned rect into a texture it then reads back. "Export" was a
//! screenshot of the viewport for exactly as long as `render` read `session.view`
//! instead of taking one.

use super::Engine;
use crate::Result;
use crate::document::{CompositeParams, DocState, Layer, LayerContent};
use crate::error::{ExportError, Produces};
use crate::gpu::{
    CompositeGroup, CompositeItem, CompositeScene, FilterDraw, GpuContext, MatteDraw, Offscreen,
    Output, SelectionOutline,
};
use crate::image::RgbaImage;
use crate::view::{Extent2, ViewTransform};
use stark_model::document::{GradientParcel, GuideScene, LayerId};
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
    /// cut-out PNG. A real branch in the media pass rather than an alpha tweak —
    /// the substrate composite is skipped entirely, so bare canvas is genuinely
    /// absent rather than white-and-invisible.
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

/// Which [`Compositor`](crate::gpu::Compositor) a render's offscreen attachments come
/// from.
///
/// Compositing runs through pass-A attachments the size of the target, so *whose*
/// they are decides who pays for a resize. The substrate's are kept from frame to
/// frame; anything rendered beside them is a different size and brings its own, so
/// the screen's are never resized out from under it — and never rebuilt on the next
/// frame to recover. That mattered as soon as something rendered off-screen
/// *repeatedly*: the navigator's miniature is one render per edit, and sharing the
/// substrate's attachments made it two rebuilds of window-sized textures and a full
/// recomposite per edit.
enum Attachments<'a> {
    /// The screen's own, cached across frames ([`Engine::compositor`]).
    Screen,
    /// The caller's, for a second surface on the same screen — the navigator's
    /// miniature — presented as the screen is (§6.5). Whether they outlive the call
    /// is decided by whoever knows whether the render repeats — see [`Offscreen`].
    Surface(&'a mut Offscreen),
    /// The caller's, for a picture bound for a file or the CPU: an export, a
    /// thumbnail, a golden. Rendered [`Output::SDR`] whatever the screen is showing,
    /// which is what keeps the screen's headroom and gamut out of a file (§6.5,
    /// §15.6).
    Export(&'a mut Offscreen),
}

/// Which document a render draws: the one being *shown*, or the committed one
/// alone.
///
/// The distinction only exists because a render can be asked for while a gesture
/// is in flight. The screen wants [`Rendered::Live`] — that is what makes a stroke
/// visible as it is drawn. A render that stands in for the *state of the work*
/// wants [`Rendered::Committed`]: it is refreshed when the document changes, so
/// following the in-flight stroke would mean re-rendering at pointer rate to show
/// something that is already on screen at full size.
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
    /// respected — what a *preview* of the whole piece asks for.
    ///
    /// It exists so asking that question does not require answering a harder one
    /// first. Asking for a 1× plan purely to learn the rect's size and then scaling
    /// that oneself means a piece wider than [`max_export_dim`] fails the query for a
    /// render it was never going to make, and the miniature quietly stops refreshing
    /// at the size where an overview starts to matter most.
    ///
    /// Scales *up* as happily as down: the overview's job is to show the whole of a
    /// piece at a glance, and a 60 px sketch shown at 60 px says less than the empty
    /// panel around it.
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
    /// The plan *is* the view, in other words, which is why both things that render a
    /// planned rect — writing a file ([`Engine::export`]) and drawing the navigator's
    /// miniature ([`Engine::render_into`]) — derive it here rather than each spelling
    /// out the same three lines and drifting.
    pub fn view(&self) -> ViewTransform {
        ViewTransform {
            center: (self.min + self.max) * 0.5,
            zoom: self.zoom,
            // Upright and unmirrored, whatever angle the artist has the canvas at:
            // turning the easel is a way of *looking* at the piece, and a file — or
            // the navigator's overview, which frames itself the same way — shows the
            // piece rather than the easel (§18.1.2).
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
/// Asked of the device rather than fixed at a number, because the number was wrong in
/// the dangerous direction. The frontend requests `wgpu::Limits::default()` while the
/// headless device ([`GpuContext::headless`]) asks only for the engine's own minimums
/// ([`MAX_TEXTURE_DIM_2D`](crate::gpu::context::MAX_TEXTURE_DIM_2D)), so a literal
/// agreed with one of them by coincidence and not the other: written against the
/// frontend's 8192, a 4096-px export passed the check on a headless device capped at
/// 2048 and then asked for a texture it was never granted. A guard that has to be
/// kept in step with a limit it does not read is a guard that is already out of step
/// somewhere — and the two limits have since moved *again*, which is the point.
///
/// It also lets the ceiling *rise*: the adapters this runs on report far more than
/// 8192 (32768 is common), so a frontend that requests more gets more, and this
/// follows it with nothing to update.
///
/// [`GpuContext::headless`]: crate::gpu::GpuContext::headless
fn max_export_dim(gpu: &GpuContext) -> u32 {
    gpu.device.limits().max_texture_dimension_2d
}

/// How much of the viewport [`Engine::show_piece`] leaves clear around the piece, as
/// a fraction of each axis on each side.
///
/// Not zero, unlike the fit an *export* makes: a file is the piece and nothing else,
/// while a view of it is a thing on an easel, and a piece flush with all four window
/// edges reads as one that carries on past them. Small enough that the margin is a
/// breath rather than a mount — the picture is still what the window is mostly
/// showing.
const SHOW_PIECE_MARGIN: f32 = 0.04;

impl Engine {
    /// Render the current canvas (preview if stroking, else committed) into
    /// `target`, through the session's own pan/zoom (§6.4).
    pub fn render(&mut self, target: &wgpu::TextureView) {
        self.render_view(
            target,
            self.session.view,
            None,
            Background::Substrate,
            Chrome::Shown,
            Rendered::Live,
            Attachments::Screen,
        );
    }

    /// Render the document through `view` into a target that is **not** the engine's
    /// own substrate — a second substrate showing the same document (§11).
    ///
    /// The navigator's miniature is the consumer: an overview of the whole piece is a
    /// second view of the canvas, and once it has somewhere to draw there is no reason
    /// for it to travel through the CPU. Reaching it through [`export`](Self::export)
    /// instead — render, copy back, hand the browser a `<canvas>` full of bytes — is
    /// this same render plus a frame of latency and a megabyte of pixels in transit.
    ///
    /// `into` holds the pass-A attachments (see [`Offscreen`]); a consumer drawing
    /// repeatedly keeps them, so a refresh allocates nothing at all. `target` must
    /// carry the format [`target_format`](Self::target_format) reports and be
    /// `view.viewport` in size — a substrate texture configured to match.
    ///
    /// No chrome: a selection outline belongs to the substrate you are painting on, not
    /// to a thumbnail of the piece.
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
    /// sits under the paint and whether on-canvas chrome is drawn (§6.4,
    /// §15.6).
    ///
    /// This is the seam export needs: exporting a frame is rendering at
    /// `frame.rect × scale`, centred on the frame, at `zoom = scale` — the same
    /// path the screen takes, so what is written is what was seen. `render` reading
    /// `session.view` instead of taking one is exactly what made "export" a
    /// screenshot of the viewport.
    ///
    /// `only` names a single layer to draw **alone**, its blend mode, clip and opacity
    /// dropped — see [`composite_groups`](Self::composite_groups), which decides what
    /// that means and has done since the eyedropper needed it (§18.0.2). `None` is the
    /// document, which is every render but a layer thumbnail's (§14.6).
    ///
    /// Private, with [`Engine::export`] and [`Engine::render_into`] as the two
    /// consumers: what a caller may choose is a view, how much of the document, a
    /// substrate and where the attachments live, never whether chrome is drawn (it is,
    /// for the screen alone) nor how the two are wired together.
    ///
    /// Over the arity lint by one, and left that way: **every parameter here is a
    /// distinct type**, so the arrangement the lint guards against — two arguments of
    /// one type, silently transposed — cannot be written. If a *second* `Option<LayerId>`
    /// ever arrives (a frame and a layer are both one), these stop being independent
    /// choices and become a "what to draw" value worth naming.
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
        // adds the substrate acquire and the present, so the difference between the two
        // rows is what the *page* costs on top of what the engine does.
        crate::timing::span!("render.view");
        // The fold is rebuilt lazily (`Engine::mark_live_stale`), and this is the
        // read that services it: once per frame painted, whatever arrived since.
        if matches!(content, Rendered::Live) {
            self.flush_live();
        }
        // Cloned rather than borrowed — a handful of `Arc` bumps (§5.1) — because it
        // buys back the borrow of `self`, and everything below wants that: the draw
        // list is rebuilt through `&mut self`, and the compositor is borrowed mutably
        // at the end. Owning the document once is cheaper than the two dances that
        // paid for it piecemeal.
        let doc = match content {
            Rendered::Live => self.presented().clone(),
            Rendered::Committed => self.timeline.current().clone(),
        };
        // Only what this view can show (§6.3). The draw list is otherwise every
        // populated tile of every visible layer, whatever the viewport — and it is
        // rebuilt only when something it is a function of has moved ([`DrawKey`]).
        //
        // Instrumented because the cache is the whole claim: this row's *count*
        // against `render.view`'s says how often the key actually moved, and a live
        // stroke moves it every frame by way of `Preview::fold`. A rebuild that
        // stopped being rare would show up here long before it showed up as a
        // dropped frame. Braced, because a timing span runs to the end of the block
        // it is opened in and what is being timed is this call rather than the rest
        // of the render (`timing::span!`).
        let key = DrawKey {
            doc_revision: self.doc_revision,
            epoch: self.preview.epoch(),
            fold: self.preview.fold(),
            content,
            only,
            visible: view.visible_tiles(),
        };
        // Held as an owned handle rather than borrowed out of the memo, which is what
        // lets the compositor be borrowed mutably at the end without the list having
        // to be copied out of the way first — the same bargain the outlines and the
        // guide scenes below strike, and the reason [`Engine::draw_list`] hands back
        // a share instead of a reference.
        let groups = {
            crate::timing::span!("render.drawlist");
            self.draw_list(key, &doc)
        };

        // The substrate is document state now (§15.5), so the substrate a
        // piece was painted on travels with it instead of living in whichever
        // frontend happened to render it.
        let bg = self.shared.color_space.rgb_to_latent(doc.substrate_color);
        // The substrate is opaque paint under everything, so its per-unit opacity is
        // 1; the residual target carries the same number (§6.7).
        let bg_channels = [bg.lat[0], bg.lat[1], bg.lat[2], 1.0];
        let bg_resid = [bg.res[0], bg.res[1], bg.res[2], 0.0];
        // Chrome never reaches a file: an exported image gets no selection outline
        // (§15.6). Keyed on `chrome`, deliberately *not* on the
        // background — a substrate export is still an export, and tying the two
        // together silently leaked the outline into every opaque PNG.
        //
        // Read off the owned `doc` above rather than through `self`, which is what
        // lets the compositor be borrowed mutably at the end without the list, the
        // outlines and the guides each having to be copied out of the way first.
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
        // Chrome, on the same argument as the outlines: a guide is a thing to
        // draw *with*, so an export or a miniature never carries one (§20.4).
        // Derived fresh per render — the camera math is a handful of products,
        // and a cached copy would shadow the session's state.
        let guide_scenes: Vec<GuideScene> = match chrome {
            Chrome::Hidden => Vec::new(),
            // The document holds the guides and the session holds whose eye is
            // shut (§20.5), so what is on screen is the two combined — one filter,
            // written once, in `Session::shown_guides`.
            //
            // The pointer is the session's too, and for the same reason: it is
            // per-client, so the rays it draws through every guide are handed in
            // here rather than being a thing a camera knows (§20.9).
            Chrome::Shown => {
                let cursor = self.session.cursor();
                self.session
                    .shown_guides(&doc)
                    .map(|g| g.camera.scene(cursor))
                    .collect()
            }
        };
        // Read as a **field**, not through an accessor: a `&self` method borrows the
        // whole engine, and the compositor is borrowed mutably three lines down.
        // Rust splits disjoint fields and not method calls, which is the whole of why
        // this is written out.

        // What display the picture is for (§6.5): the screen's own setting for the
        // screen and a surface beside it, SDR for anything bound for a file.
        let output = match attachments {
            Attachments::Screen | Attachments::Surface(_) => self.compositor_pipeline.output(),
            Attachments::Export(_) => Output::SDR,
        };
        let scene = CompositeScene {
            substrate_color: bg_channels,
            substrate_resid: bg_resid,
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
        // submitted (§6.3). CPU time to *record* them, like every row here — WebGPU
        // offers no timestamp query on the web, so nothing in this module can say
        // what the GPU then spent executing them. The signal for that is the
        // frontend's frame-skip counter (`Renderer::gpu_behind`), which is the
        // honest place for it: a queue that will not drain is what being GPU-bound
        // looks like from the CPU's side.
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

    /// One committed tile's channels, straight off the GPU — **height and alpha
    /// without the lit composite in between** (§6.1).
    ///
    /// `None` if that layer has no paint tile at that coordinate, which includes every
    /// layer that is not paint at all. Reads the *committed* document, so a caller
    /// mid-gesture is asking about the state before the live tail.
    ///
    /// **`pub` for the suite and nothing else**, on [`render_to_image`](Self::render_to_image)'s
    /// terms. Every conservation, opacity and erase claim in the suite was a proxy
    /// through tonemapping before this: the assertions read image darkness and said so
    /// in a comment, because there was no way to ask a tile what it held. A proxy
    /// through the media pass, the blend and the tonemap cannot separate "height was
    /// not conserved" from "the light changed", and §6.1 is a claim about the first.
    #[cfg(not(target_arch = "wasm32"))]
    #[doc(hidden)]
    pub fn tile_channels(
        &self,
        layer: stark_model::document::LayerId,
        coord: stark_model::geom::TileCoord,
    ) -> Option<crate::gpu::TileChannels> {
        Some(
            self.document()
                .layer(layer)?
                .tiles()?
                .get(&coord)?
                .read_channels(&self.shared.gpu),
        )
    }

    /// Render the current canvas to a CPU-side image at the viewport size (§9). The
    /// backbone of golden tests. The target uses the engine's configured format, so it
    /// matches on-screen rendering.
    ///
    /// Blocking, and therefore **native-only**: WebGPU has no blocking poll, so this
    /// shape cannot work on the web (see `gpu::readback`). The frontend uses
    /// [`export`](Self::export), which awaits the map.
    ///
    /// **`pub` for the suite and nothing else** — an integration test is a separate
    /// crate, so a diagnostic it reads has to be public (`testing`). Hidden from the
    /// docs to say so.
    #[cfg(not(target_arch = "wasm32"))]
    #[doc(hidden)]
    pub fn render_to_image(&mut self) -> RgbaImage {
        // One render per call, so nothing is kept: the attachments are allocated here
        // and dropped with this `Offscreen`.
        let (target, size) = self.render_offscreen(
            &mut Offscreen::default(),
            self.session.view,
            None,
            Background::Substrate,
            Chrome::Shown,
            Rendered::Live,
        );
        let pixels = crate::gpu::readback::read_rgba8_blocking(&self.shared.gpu, &target, size);
        RgbaImage::from_target_bytes(size.width, size.height, pixels, target.format())
    }

    /// Render the current canvas through the **screen's** passes into a half-float
    /// texture and read it back as `f32` — four per texel, rows top-down, in the
    /// transfer the engine's [`Output`](crate::Output) names (§6.5). Only for an
    /// engine built on an `Rgba16Float` target. `pub` for the suite and nothing else,
    /// on [`render_to_image`](Self::render_to_image)'s terms.
    #[cfg(not(target_arch = "wasm32"))]
    #[doc(hidden)]
    pub fn render_to_floats(&mut self) -> Vec<f32> {
        let format = self.shared.target_format;
        assert_eq!(
            format,
            wgpu::TextureFormat::Rgba16Float,
            "render_to_floats reads a half-float screen, and this engine presents to {format:?}",
        );
        let view = self.session.view;
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
            crate::gpu::export_format(self.shared.target_format),
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
        let limit = max_export_dim(&self.shared.gpu);
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
    /// This is the same path the screen takes — every visible layer composited
    /// through the media pass — just with the view centred on the frame at
    /// `zoom = scale`. Nothing is special-cased: a frame matte covers only
    /// *outside* its rect, which is clipped away here, so it contributes nothing
    /// to its own export, while a substrate matte is inside and contributes exactly
    /// what it should.
    /// Renders immediately and returns a future for the **readback**, which is the
    /// only asynchronous part (§7 — on WebGPU `mapAsync` settles only
    /// when the browser's event loop runs, so there is no way to block on it).
    ///
    /// Deliberately *not* an `async fn`. An `async fn` would hold `&mut self` for
    /// the whole readback, and a frontend must take that borrow from a shared cell
    /// — so the engine would stay locked across an await during which the UI
    /// re-renders and tries to read it, panicking with `AlreadyBorrowedMut`. This
    /// shape ends the borrow when `export` returns: the returned future owns a
    /// cloned [`GpuContext`] (cheap — the handles are reference-counted) and the
    /// target texture, and touches the engine not at all.
    ///
    /// `content` chooses whether the in-flight gesture is in the picture: a file
    /// export takes [`Rendered::Live`], since that is what the artist is looking at.
    /// (Anything refreshed per *committed* change wants [`Rendered::Committed`]
    /// instead — see [`render_into`](Self::render_into), which is the shape that
    /// suits a render repeated on a cadence rather than written to a file.)
    ///
    /// `into` is where the render's attachments live. It renders **beside** the
    /// substrate rather than into it, so it never touches the screen's; whether its own
    /// outlive the call is the caller's call, and the caller is the only one who knows
    /// (see [`Offscreen`]) — a `&mut Offscreen::default()` for a one-shot, a held one
    /// for a render that repeats.
    ///
    /// **Two `Result`s, and they answer different questions.** The outer one is the
    /// request: a frame too small, a size past the device's limit — refused before
    /// anything is drawn, and answerable by asking for something else. The inner one
    /// is the *readback*, which can only fail by the GPU failing underneath it
    /// (§5), and is not answerable at all — but is reported rather than panicked on,
    /// because the action log survives what the device does not and a caller told
    /// this can still save the file.
    ///
    /// ```text
    /// let readback = { engine.write().export(&mut own, frame, scale, bg, content)? }; // borrow ends
    /// let image = readback.await?;
    /// ```
    /// **This is [`export_view`](Self::export_view) through the plan's own view**,
    /// which is the whole of what "export a frame" adds: [`ExportPlan::view`] already
    /// exists so that the two things which render a planned rect derive it in one
    /// place, and having said that, exporting one is not a second render path. The
    /// tail — render off-screen, hand back a future that owns what it reads — is
    /// otherwise written out twice, down to the borrow bargain the doc comment above
    /// explains.
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
    /// `export` answers "the piece, at a scale": its rect comes from a frame or the
    /// painted bounds, tile-aligned in the fallback. A preset thumbnail asks the
    /// question the other way round — *this* rect, at *this* pixel size — and
    /// deriving that from `export_rect` would crop to whichever tiles the test
    /// stroke happened to land in. So this takes the view whole: `view.viewport` is
    /// the output size, and the same borrow bargain as `export` applies — the
    /// returned future owns what it reads, so the caller drops its engine borrow
    /// before awaiting.
    ///
    /// No chrome, like every render that is not the screen's (§15.6).
    ///
    /// `only` names a layer to render **alone** — its own paint, with its blend mode,
    /// clip and opacity dropped, which is what makes the result an identity card for
    /// the layer rather than a picture of its contribution
    /// ([`composite_groups`](Self::composite_groups) settles what that means, and
    /// settled it for the eyedropper first). The layer panel's thumbnails are the
    /// consumer (§14.6); `None` renders the document, which is what every other caller
    /// wants. A layer that is hidden or fully transparent draws nothing at all, so a
    /// caller that would rather show the last picture it had than a blank one should
    /// ask before rendering rather than after.
    ///
    /// Errors mirror [`export_plan`](Self::export_plan)'s: a degenerate or
    /// non-finite view, or a viewport past the device's texture limit, is reported
    /// rather than surfacing as a wgpu validation panic.
    pub fn export_view(
        &mut self,
        into: &mut Offscreen,
        view: ViewTransform,
        only: Option<LayerId>,
        background: Background,
        content: Rendered,
    ) -> Result<impl std::future::Future<Output = Result<RgbaImage>> + use<>> {
        // The same question the view's own mutators ask before storing anything
        // ([`ViewTransform::usable`]), rather than a second spelling of it here: a
        // caller-supplied view has not passed through them, so it is asked once, at
        // the door it comes in by.
        if !view.usable() {
            return Err(ExportError::UnusableView.into());
        }
        let size = view.viewport;
        let limit = max_export_dim(&self.shared.gpu);
        if size.width == 0 || size.height == 0 || size.width > limit || size.height > limit {
            return Err(ExportError::OverLimit {
                what: Produces::Render,
                size,
                limit,
            }
            .into());
        }
        // No chrome: a selection outline or any other on-canvas affordance is a
        // thing to draw *with*, never a thing to ship. The hover mark (§18.1.10)
        // is the same statement made of paint — a hypothesis about the *next*
        // stroke, not work — and it may never reach a file. Dropped rather than
        // excluded per-render, because the fold is one cached document; honestly
        // so, since a moment worth exporting is not one the hand is painting in,
        // and the next hover report re-seeds it.
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

    /// [`composite_groups`](Self::composite_groups), **memoized**: the draw list for
    /// `key`, built afresh or taken from [`Engine::draw_cache`] (C4).
    ///
    /// **Why this is worth a cache at all.** Building the list clones a
    /// `TilePairHandle` per visible tile, per layer — an atomic increment in and a
    /// decrement out — plus a `Vec` per layer. The visible tile count scales as
    /// 1/zoom², so a zoomed-out multi-layer document was paying ~10⁵ of those every
    /// frame to produce a list identical to the last one. A canvas nobody is editing
    /// or panning now pays nothing.
    ///
    /// **A single-layer list is never cached.** The memo holds one list, and a
    /// thumbnail pass renders one layer at a time with a sleep between rows while the
    /// canvas keeps painting frames — so each thumbnail evicted the screen's list and
    /// the next screen frame rebuilt it from nothing, N times over for N layers, which
    /// is exactly the ~10⁵ handle clones the cache exists to stop paying. Such a key
    /// can never be hit twice anyway: every row names a different `only`. Deliberately
    /// *not* a second slot keyed on `only` either — the navigator refreshes per commit,
    /// and at that instant `doc_revision` has already moved the screen's key, so the
    /// eviction that slot would prevent costs nothing.
    ///
    /// Takes the key rather than deriving it so the caller can compute it while it
    /// still holds `doc` — and takes `doc` borrowed for the same reason. The list comes
    /// back as an `Arc` rather than a reference into the memo so that holding it does
    /// not hold a borrow of the engine: the compositor is borrowed mutably a few lines
    /// after the call, and the whole of what the `RefCell` bought would go back out
    /// through a `Ref` guard that had to be kept alive to read the slice.
    pub(super) fn draw_list(&self, key: DrawKey, doc: &DocState) -> Arc<[CompositeGroup]> {
        if key.only.is_some() {
            return self.composite_groups(doc, key.only, key.visible).into();
        }
        self.draw_cache.get_or_build(key, || {
            self.composite_groups(doc, key.only, key.visible).into()
        })
    }

    /// The compositor's draw list for `doc`, bottom-to-top: every visible layer's
    /// tiles and mattes, each tagged with its layer opacity, cut into blend groups
    /// (§18.0.4, §14.7).
    ///
    /// Consecutive layers that need no isolation share one `Run` — they compose
    /// correctly against each other and against everything below under
    /// premultiplied "over", so a document that uses no blend modes, no clipping
    /// and no groups produces exactly one `Run` and the compositor's work is
    /// unchanged. Anything else becomes a group of its own, because its mode and
    /// its clip are both defined against *what is underneath it*, which means it
    /// has to be composited in isolation first.
    ///
    /// A layer that **carries** others is a group, and composites as a `Stack`:
    /// its own content at the bottom, then each carried layer merging into what is
    /// beneath it *within the group* (§14.2). The group as a whole
    /// then merges outward through its own — that is, its base's — blend mode,
    /// clip and opacity.
    ///
    /// Within a run this is an *ordered* item list rather than a flat tile list
    /// because a matte has to composite at its own place in the stack — a frame over
    /// the painting, a substrate under it (§15.4.4). The compositor
    /// re-batches consecutive tiles into one instanced draw, so an all-paint document
    /// costs nothing for it.
    ///
    /// `only` restricts the list to a single layer — the eyedropper's
    /// sample-one-layer option (§18.0.2). It means that layer's *own*
    /// paint: what it carries is left out, and its mode, clip and opacity go with
    /// it, since a sample is of the paint that is there rather than of the part of
    /// it that survives its surroundings. Sharing this with rendering is what makes
    /// a sample come off the same stack the screen draws.
    /// `visible` is the view-AABB cull (§6.3): only tiles it names are built into
    /// the draw list. `None` culls nothing — see [`ViewTransform::visible_tiles`].
    ///
    pub(super) fn composite_groups(
        &self,
        doc: &DocState,
        only: Option<LayerId>,
        visible: Option<TileRect>,
    ) -> Vec<CompositeGroup> {
        if let Some(id) = only {
            // Fully transparent is the one exception to dropping the opacity below,
            // and this filter is now the whole of it rather than an optimization: a
            // layer turned all the way down contributes nothing to the document, so
            // sampling it answers "nothing here" — the same answer bare canvas gives —
            // instead of reporting paint that is switched off. Everywhere above zero
            // the setting says nothing about what the paint *is*, so nothing about
            // what a sample of it reports; at zero it is not a fainter statement of
            // the same thing, it is the absence of one.
            //
            // Hidden reads the same way, and for the reason it does everywhere else:
            // a sample comes off the same stack the screen draws (§18.0.2).
            let Some(layer) = doc.layer(id).filter(|l| l.is_shown()) else {
                return Vec::new();
            };
            let items = self.layer_items(layer, visible);
            return if items.is_empty() {
                Vec::new()
            } else {
                // **All three composite params are dropped**, opacity included: a
                // sample is of the paint that is there, not of the part of it that
                // survives its surroundings, and a layer's opacity is exactly such a
                // surrounding — it says how much of this layer the *document* shows,
                // which is the question the other two pick sources ask. Turning a
                // layer down does not turn its paint into a paler paint, so
                // "sample this layer" must answer the same color at any setting.
                //
                // The pick already reported that color, because it divides by the
                // coverage it sums and the opacity cancels (`mean_channels`). What
                // dropping it changes is where the pick answers **at all**: a faded
                // layer's coverage was scaled down towards `PICK_MIN_OPACITY`, so a
                // thin glaze on a layer at 20% could report "nothing here" while the
                // same paint at 100% reported its color.
                vec![CompositeGroup::leaf(CompositeParams::IDENTITY, items)]
            };
        }
        // The root stack has nothing under its first member, by definition — see
        // `composite_stack`'s `under`.
        self.composite_stack(doc.root().iter(), visible, false)
    }

    /// The draw list for an eyedropper source (§18.0.2) —
    /// [`composite_groups`](Self::composite_groups) for the whole-document and one-layer
    /// questions,
    /// plus the two scoped ones: a group's interior, and the document cut above a
    /// layer. Here rather than in `engine::pick` because it is draw-list
    /// arithmetic: everything it does is a restriction of `composite_stack`'s
    /// walk, and keeping the restrictions beside the walk is what keeps a sample
    /// coming off the same stack the screen draws.
    pub(super) fn pick_groups(
        &self,
        doc: &DocState,
        source: super::pick::PickSource,
        visible: Option<TileRect>,
    ) -> Vec<CompositeGroup> {
        use super::pick::PickSource;
        match source {
            PickSource::Composite | PickSource::CompositeOverSubstrate => {
                self.composite_groups(doc, None, visible)
            }
            PickSource::Layer(id) => self.composite_groups(doc, Some(id), visible),
            PickSource::Group { layer, below } => self.group_interior(doc, layer, below, visible),
            PickSource::Below(layer) => self.below_groups(doc, layer, visible),
        }
    }

    /// The interior of the group that carries `layer`: the carrier's own content at
    /// the bottom, then the members — all of them, or with `below` only those up to
    /// and including `layer` (`PickSource::Group`).
    ///
    /// This is `composite_stack`'s group branch with the carrier's outward params
    /// dropped instead of applied — the same trade `composite_groups` makes for one
    /// layer, for the same reason: the params say how the group meets what is
    /// beneath it, and beneath it is what this source excludes. The members keep
    /// theirs, because a sibling's mode against the base is part of what the group
    /// looks like *inside*. A layer in the root stack reads the root as its group,
    /// which makes this the whole document — `Composite`, built by the same walk.
    ///
    /// A carrier that is hidden or turned all the way down contributes nothing to
    /// the screen, so its interior answers nothing — the `Layer` source's rule. The
    /// layer itself is only the anchor: hidden or not, its *group* is still what is
    /// being asked about, and the member walk already skips it like the screen does.
    fn group_interior(
        &self,
        doc: &DocState,
        layer: LayerId,
        below: bool,
        visible: Option<TileRect>,
    ) -> Vec<CompositeGroup> {
        let (base, members): (Option<&Layer>, &rpds::Vector<Layer>) = match doc.carrier_of(layer) {
            Some(cid) => {
                let Some(carrier) = doc.layer(cid).filter(|l| l.is_shown()) else {
                    return Vec::new();
                };
                (Some(carrier), &carrier.carries)
            }
            // `carrier_of` answers `None` for a root layer and for a layer that
            // does not exist, and only the first has a group to sample.
            None if doc.contains_layer(layer) => (None, doc.root()),
            None => return Vec::new(),
        };
        // The anchor is in `members` by construction — `carrier_of` just said so.
        let end = match members.iter().position(|l| l.id == layer) {
            Some(at) if below => at + 1,
            _ => members.len(),
        };
        let own = base.map_or_else(Vec::new, |b| self.layer_items(b, visible));
        let mut groups = Vec::new();
        if !own.is_empty() {
            groups.push(CompositeGroup::leaf(CompositeParams::IDENTITY, own));
        }
        let under = !groups.is_empty();
        groups.extend(self.composite_stack(members.iter().take(end), visible, under));
        groups
    }

    /// The document with everything above `layer` switched off, bottom of the tree
    /// up to and including the layer itself (`PickSource::Below`).
    ///
    /// The chain of carriers from the root down to the layer is the only part of
    /// the tree that is *partially* included, so the walk follows exactly that
    /// path: whole stacks beneath each ancestor, then the ancestor cut above the
    /// next link ([`stack_below`](fn@Self::stack_below)).
    fn below_groups(
        &self,
        doc: &DocState,
        layer: LayerId,
        visible: Option<TileRect>,
    ) -> Vec<CompositeGroup> {
        if !doc.contains_layer(layer) {
            return Vec::new();
        }
        // Root-first: the last link is the layer itself.
        let mut path = vec![layer];
        let mut cur = layer;
        while let Some(carrier) = doc.carrier_of(cur) {
            path.push(carrier);
            cur = carrier;
        }
        path.reverse();
        self.stack_below(doc.root(), &path, visible, false)
    }

    /// One stack's worth of [`below_groups`](fn@Self::below_groups): everything
    /// beneath `path[0]` whole, then `path[0]` itself — whole when it is the target,
    /// cut above `path[1]` when it is an ancestor carrying the rest of the chain.
    ///
    /// The ancestor's own composite params are **kept** and applied to the partial
    /// group, unlike the one-layer and group-interior sources — because this source
    /// asks what the screen would show with the upper layers hidden, and hiding a
    /// member does not lift the group's mode, clip or opacity off what remains. An
    /// ancestor that is itself hidden or turned off takes the whole chain with it,
    /// exactly as it does on screen; the layers beneath it still answer.
    fn stack_below(
        &self,
        layers: &rpds::Vector<Layer>,
        path: &[LayerId],
        visible: Option<TileRect>,
        under: bool,
    ) -> Vec<CompositeGroup> {
        let Some(&head) = path.first() else {
            return Vec::new();
        };
        let Some(at) = layers.iter().position(|l| l.id == head) else {
            return Vec::new();
        };
        if path.len() == 1 {
            // The target itself: it and everything beneath it in this stack,
            // whole — the ordinary walk, stopped early.
            return self.composite_stack(layers.iter().take(at + 1), visible, under);
        }
        let mut groups = self.composite_stack(layers.iter().take(at), visible, under);
        let ancestor = layers.get(at).expect("position() names an element");
        if !ancestor.is_shown() {
            return groups;
        }
        // `composite_stack`'s group branch, with the carried stack cut by the rest
        // of the path. An ancestor is a carrier, and a filter never carries
        // (§21.2), so the filter branch cannot arise here.
        let own = self.layer_items(ancestor, visible);
        let carried = self.stack_below(&ancestor.carries, &path[1..], visible, !own.is_empty());
        let Some(group) = group_of(ancestor.composite, own, carried) else {
            return groups;
        };
        // Through the same merge the draw list uses. This pushed unmerged before,
        // which is pixel-identical — more direct runs, same picture — but it meant the
        // two walks could disagree about the shape of what they built.
        push_merging(&mut groups, group);
        groups
    }

    /// One stack's worth of groups — the root's, or a layer's carried stack.
    ///
    /// `under` says whether something already composites beneath this stack's first
    /// member: false for the document's own stack, and for a carried stack whether
    /// the **base's own content** draws anything. That is §14.1's algorithm read
    /// back — a group's members composite over the base's content, so inside a group
    /// the base is what lies beneath the bottom member.
    ///
    /// Only a **filter** asks (§21.2), which is why this is a `bool` handed down
    /// rather than the base's items handed down: everything else in this walk is
    /// defined against what it is drawn *into*, and a filter is the one thing
    /// defined against what has already been drawn.
    fn composite_stack<'a>(
        &self,
        layers: impl Iterator<Item = &'a Layer>,
        visible: Option<TileRect>,
        under: bool,
    ) -> Vec<CompositeGroup> {
        let mut groups: Vec<CompositeGroup> = Vec::new();
        for layer in layers {
            // Hiding a layer hides what it carries: the group is the layer
            // (§14.3), so its visibility is the group's.
            if !layer.is_shown() {
                continue;
            }
            // A **filter layer** rewrites what is already composited beneath it *in
            // its own stack* (§21.2) — the same set a clip reads, which is what makes
            // "filter just this layer" the single gesture of carrying it onto that
            // layer rather than a scoping mode of its own. It never carries anything
            // itself: the state refuses to attach children to one
            // (`DocState::cannot_carry`), so this branch is the whole of what a
            // filter can be.
            //
            // Two ways it reaches nothing, and both drop it from the draw list rather
            // than encoding a pass that provably cannot change a texel: **nothing is
            // beneath it here** (the foot of a stack, or a stack whose lower members
            // were all culled), or the filter is at its **neutral** setting, which is
            // what a freshly added one holds (§21.3).
            //
            // The first of those is also where a **clip** would have nothing to say
            // (§14.4.3) — a filter with no backdrop is already the identity, so the
            // one place the two properties could disagree is a pass that is not
            // encoded at all.
            if let Some(f) = layer.filter() {
                debug_assert!(
                    layer.carries.is_empty(),
                    "a filter never carries (§21.2) — the state refuses the arrangement",
                );
                if (under || !groups.is_empty()) && !f.is_neutral() {
                    groups.push(CompositeGroup::filter(FilterDraw::new(f, layer.composite)));
                }
                continue;
            }
            let own = self.layer_items(layer, visible);
            let carried = self.composite_stack(layer.carries.iter(), visible, !own.is_empty());
            // An empty layer is dropped rather than given a group. For `Normal`
            // that only saves a loop; for a blend mode or a clip it saves two
            // render passes that provably compute the identity, which is what
            // keeps a stack of empty glow layers free. A layer that carries
            // something visible is not empty, whatever its own content.
            //
            let Some(group) = group_of(layer.composite, own, carried) else {
                continue;
            };
            push_merging(&mut groups, group);
        }
        groups
    }

    /// What one layer's own content draws, without what it carries.
    ///
    /// Every item comes out at **opacity 1**, and that is not a stub: the layer's
    /// opacity is a [`CompositeParams`] field, so it arrives with the other two and
    /// is folded in — or not — by [`CompositeGroup::leaf`], which is the one place
    /// that decision is made (§14.7). This function is deliberately not told what a
    /// layer's opacity is; tagging items with it here as well is how a group's base
    /// gets faded twice.
    ///
    /// Paint is culled to `visible` (§6.3); a matte is not. A matte's rect can be
    /// the *hole* in a frame, whose fill covers everything outside it (§15.4.4), so
    /// there is no box to test it against — and there is at most one per layer, so
    /// there would be nothing to win.
    fn layer_items(&self, layer: &Layer, visible: Option<TileRect>) -> Vec<CompositeItem> {
        match &layer.content {
            LayerContent::Paint(tiles) => {
                // The layer's frame (§14.12): its tiles are keyed in it, the view
                // rect is on the canvas, and this is the one place the two meet —
                // the cull asks in the frame, the item answers on the canvas.
                let d = layer.translation;
                let offset = stark_model::geom::Vec2::new(d.x as f32, d.y as f32);
                culled(tiles.map(), visible_in_frame(visible, d))
                    .map(|(coord, handle)| CompositeItem::Tile {
                        coord,
                        origin: coord.origin() + offset,
                        handle: handle.clone(),
                        opacity: 1.0,
                    })
                    .collect()
            }
            LayerContent::Matte { region, paint } => {
                // The layer's frame again (§14.12): the rect and the gradient
                // axis are stated in it, and this is where both land on the
                // canvas.
                let d = layer.translation;
                let offset = stark_model::geom::Vec2::new(d.x as f32, d.y as f32);
                let rect = match region.translated(offset).rect() {
                    Some((min, max)) => [min.x, min.y, max.x, max.y],
                    // The whole plane: the shader never reads the rect (`flags`
                    // routes past it), so zeros rather than sentinels.
                    None => [0.0; 4],
                };
                let flags = match region {
                    stark_model::document::MatteRegion::OutsideRect { .. } => 0.0,
                    stark_model::document::MatteRegion::Everything => 1.0,
                };
                // sRGB in the log, working-space channels on the GPU — the same
                // conversion the brush color gets, so a matte means the same
                // color in an Oklab and a Mixbox document. A gradient converts
                // every stop the same way, once per item build, and the shader
                // interpolates in the working space (§22.4).
                let (channels, resid, ramp) = match paint {
                    stark_model::document::Parcel::Solid(color) => {
                        let l = self.shared.color_space.rgb_to_latent(*color);
                        (
                            [l.lat[0], l.lat[1], l.lat[2], 1.0],
                            [l.res[0], l.res[1], l.res[2], 0.0],
                            None,
                        )
                    }
                    stark_model::document::Parcel::Gradient(GradientParcel { gradient, axis }) => {
                        let mut ramp = stark_shaders::mirror::matte::Ramp::default();
                        let stops = gradient.stops();
                        ramp.p[0] = stops.len() as f32;
                        ramp.axis = match axis.translated(offset) {
                            stark_model::document::GradientAxis::Linear { from, to } => {
                                [from.x, from.y, to.x, to.y]
                            }
                            stark_model::document::GradientAxis::Radial { center, radius } => {
                                ramp.p[1] = 1.0;
                                [center.x, center.y, radius, 0.0]
                            }
                        };
                        // Indexed by the stop's own position, and bounded by
                        // `Gradient`'s invariant rather than by a check here: a ramp
                        // holds at most `gradient::MAX_STOPS`, which is the length
                        // `matte.wesl` declares `stop_c` at. Truncating instead would
                        // hide a broken invariant behind a wrong picture; this way it
                        // is loud, at the site.
                        for (i, stop) in stops.iter().enumerate() {
                            let l = self.shared.color_space.rgb_to_latent(stop.color);
                            ramp.stop_c[i] = [l.lat[0], l.lat[1], l.lat[2], stop.t];
                            ramp.stop_r[i] = [l.res[0], l.res[1], l.res[2], 0.0];
                        }
                        ([0.0; 4], [0.0; 4], Some(Box::new(ramp)))
                    }
                };
                vec![CompositeItem::Matte(MatteDraw {
                    rect,
                    flags,
                    channels,
                    resid,
                    opacity: 1.0,
                    ramp,
                })]
            }
            // A filter draws no items at all: it is a pass over what the *stack* has
            // built, not content of its own, so `composite_stack` gives it a group
            // rather than asking here. That is also what the eyedropper's
            // sample-one-layer option reads (§18.0.2) — a filter layer's own content
            // is nothing, and sampling it reports nothing rather than reporting the
            // picture it happens to be sitting over.
            LayerContent::Filter(_) => Vec::new(),
        }
    }

    /// A flat 2-D texture to render into off-screen: an export's target, or one of the
    /// eyedropper's two sample attachments. Everything but the label, the format, the
    /// size and the usage is the same for all of them — a single mip, a single sample,
    /// no view formats — and was written out once per call site until it wasn't.
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
            self.session.view.visible_bounds()
        })
    }

    /// The canvas-space rect that **is the piece**: the named frame's, else the
    /// painted bounds, else `None` (§15.6).
    ///
    /// The rule itself, without the last resort, because its two askers want
    /// different last resorts and only one of them is "the viewport". An export has
    /// to write *something*, so it frames what you are looking at; framing the view
    /// on a document with neither paint nor frame has nothing to frame and should
    /// leave the view alone — falling back the same way would zoom the window onto
    /// itself. Shared so that what a file would hold and what
    /// [`ViewCommand::ShowPiece`](crate::command::ViewCommand::ShowPiece) puts on
    /// screen cannot come to disagree about where the piece ends.
    pub(super) fn piece_rect(
        &self,
        frame: Option<LayerId>,
    ) -> Option<(stark_model::geom::Vec2, stark_model::geom::Vec2)> {
        let doc = self.timeline.current();
        // An `Everything` matte has no rect and so defines no frame: naming one
        // falls through to the painted bounds, the same answer as no frame at
        // all — a substrate is under the picture, not a crop of it (§15.6).
        // The rect is placed by the layer's translation (§14.12), so export
        // frames the hole where the compositor draws it.
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
            self.session.view.show_rect(min, max, SHOW_PIECE_MARGIN);
        }
    }

    /// The selection masks to outline, and whose each is (§17.3).
    ///
    /// `DocState` holds a selection for every actor that ever made one, because
    /// replay needs them all; only the actors actually *here* are candidates. The log
    /// decides what exists, presence decides what could be shown — and
    /// `show_peer_selections` decides whether it is, since a second contour over the
    /// artwork is a preference rather than a fact about the drawing.
    fn visible_selections(
        &self,
        doc: &DocState,
    ) -> Vec<(crate::document::Selection, Option<[f32; 3]>)> {
        let mut out = Vec::new();
        let mine = doc.selection_of(self.actor());
        if mine.is_active() {
            out.push((mine, None));
        }
        if self.session.show_peer_selections {
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

/// The entries of `map` that `visible` admits, **walking whichever side is smaller**.
///
/// The cull is the same set either way — `TileRect::contains` and a probe of
/// `TileRect::coords` agree by construction — so the only question is which walk to
/// pay for, and the two differ by orders of magnitude in opposite directions.
///
/// Scanning the layer and filtering is right when the viewport admits more tiles than
/// the layer holds: every zoomed-out frame, which is the case the cull was written
/// for (§6.3), where the visible count scales as 1/zoom². It is badly wrong the other
/// way. At 1:1 on a large painting the viewport holds a few dozen tiles and a layer
/// holds thousands, and `HashTrieMap::iter` walks the whole trie to discard nearly all
/// of it — per layer, per frame, so a document's paint cost the frame rate whether or
/// not any of it was on screen.
///
/// Probing costs one hash lookup per *visible* tile instead, so the frame follows the
/// viewport. Which is what a cull is supposed to buy, and what the scan quietly was
/// not buying.
///
/// `None` claims everything: the box could not be measured (a non-finite view, or one
/// so far out that whole tiles leave the `i32` grid), and an optimization that cannot
/// see its input must do nothing rather than guess — see [`ViewTransform::visible_tiles`].
/// The view's tile rect, restated in a layer frame placed at `frame` on the
/// canvas (§14.12): what is visible of a translated layer's **local** tiles.
/// Rounded outward — a frame off the tile grid lands the rect astride tiles, and
/// a cull may only ever keep too much. Saturating, so a frame at the integer
/// horizon degrades to a wider rect rather than wrapping to the far side.
fn visible_in_frame(
    visible: Option<TileRect>,
    translation: stark_model::geom::IVec2,
) -> Option<TileRect> {
    if translation == stark_model::geom::IVec2::ZERO {
        return visible;
    }
    let rect = visible?;
    let t = stark_model::geom::TILE_SIZE as i64;
    // The rect moves by −frame: floor for the low edge, ceil for the high one.
    // In `i64`, like `sources_in`'s spans: the frame is clamped at the funnel
    // (`FRAME_LIMIT`), but a cull may not rest a wrap away from a value one
    // unclamped writer could someday hand it.
    let lo = |d: i32| (-(d as i64)).div_euclid(t);
    let hi = |d: i32| (-(d as i64) + t - 1).div_euclid(t);
    let edge = |base: i32, shift: i64| {
        (base as i64 + shift).clamp(i32::MIN as i64, i32::MAX as i64) as i32
    };
    Some(TileRect {
        min: (
            edge(rect.min.0, lo(translation.x)),
            edge(rect.min.1, lo(translation.y)),
        ),
        max: (
            edge(rect.max.0, hi(translation.x)),
            edge(rect.max.1, hi(translation.y)),
        ),
    })
}

fn culled<V>(
    map: &rpds::HashTrieMap<stark_model::geom::TileCoord, V>,
    visible: Option<TileRect>,
) -> Box<dyn Iterator<Item = (stark_model::geom::TileCoord, &V)> + '_> {
    match visible {
        Some(rect) if rect.count() < map.size() as u64 => Box::new(
            rect.coords()
                .filter_map(move |c| map.get(&c).map(|v| (c, v))),
        ),
        Some(rect) => Box::new(
            map.iter()
                .filter(move |(coord, _)| rect.contains(**coord))
                .map(|(coord, v)| (*coord, v)),
        ),
        None => Box::new(map.iter().map(|(coord, v)| (*coord, v))),
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
///   merged remote action, a load (`Engine::committed_changed`).
/// - `epoch` moves whenever the document the previews are drawn over is *replaced*:
///   the unlogged drag preview being installed or dropped (`Preview::set_doc`, the
///   only way to move that slot, which invalidates) — and a commit, which invalidates
///   too. So it is *wider* than `doc_revision` rather than a second, narrower term.
///   Both are named because a key names the terms its value depends on, not the
///   smallest set that happens to cover them: reading the wide one alone would make
///   this key rest on `committed_changed`'s implementation rather than on its promise.
/// - `fold` moves whenever the live fold is *rebuilt* (`Preview::rebuild`) — a stroke
///   in flight commits nothing and replaces no document, so neither counter above
///   stirs while one is being drawn, and a list keyed without this would hold the
///   frame at the moment the pen went down. This is the term the two roster keys do
///   not have and do not need: they project what is *shown*, and the fold is not that
///   (`super::ShownKey`).
/// - `content` is which document is being drawn at all. `Live` and `Committed`
///   differ by exactly the in-flight gesture, so at one instant they are two
///   different lists — and the navigator's miniature asks for the second while the
///   canvas asks for the first.
/// - `only` and `visible` are the two arguments that shape the list.
///
/// **What is deliberately absent is the view.** A draw list is built against the
/// *tile rect* a view can reach, not the view, so panning within one tile — or
/// rotating, or supersampling — hits the same key. That is `ViewTransform::visible_tiles`'
/// conservatism paying for itself twice.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct DrawKey {
    doc_revision: u64,
    epoch: u64,
    fold: u64,
    content: Rendered,
    only: Option<LayerId>,
    visible: Option<TileRect>,
}

/// **Where a layer's composite params go** (§14.4.3, §14.7) — the one statement of
/// the rule, and `None` for a layer that draws nothing at all.
///
/// - A **leaf** is the whole of what it draws, so its params are its run's.
/// - A **group's base** is a *member* of the group, not the group. Its own content
///   composites with `IDENTITY` and the layer's params are applied once, to the
///   composited whole. Every one of the three would be wrong applied twice: a blend
///   mode would combine the base with itself, a clip would clip the base to its own
///   coverage, and an opacity would fade it to `a²` — which is what happened for as
///   long as the three were carried separately and the item builder tagged them with
///   one of them.
///
/// The empty case answers `None` rather than an empty group because **the cull can
/// empty a layer that has paint**, so it fires for a document scrolled away from as
/// well as one not yet painted on. That is sound on the identity above rather than on
/// the two cases coinciding: `blend_common.wesl::merge` with a transparent source is
/// `cb` exactly — `cs.a` is 0, so both source terms vanish and the aux sum adds
/// nothing — for every mode and both clip states.
///
/// Asked in two walks, which is why it is a function: `composite_stack` builds the
/// draw list, and `stack_below` builds the restriction of it a `PickSource::Below`
/// sample comes off. Written out at each, the second keeps the old rule when the first
/// changes.
fn group_of(
    params: CompositeParams,
    own: Vec<CompositeItem>,
    carried: Vec<CompositeGroup>,
) -> Option<CompositeGroup> {
    if own.is_empty() && carried.is_empty() {
        return None;
    }
    Some(if carried.is_empty() {
        CompositeGroup::leaf(params, own)
    } else {
        let mut members = Vec::with_capacity(carried.len() + 1);
        if !own.is_empty() {
            members.push(CompositeGroup::leaf(CompositeParams::IDENTITY, own));
        }
        members.extend(carried);
        CompositeGroup::stack(params, members)
    })
}

/// Push `group`, merging it into the run below when neither side needs isolating —
/// the fast path, and the reason an ordinary document is one group.
///
/// `as_direct_run_mut` is the test and the run in one, which is what makes this
/// total. Asking `is_direct` of both sides and then re-matching the two
/// `GroupContent`s behind an `if let` with no else leaves a gap: a group answering
/// "direct" while holding something other than a `Run` would be counted as merged and
/// then silently dropped.
fn push_merging(groups: &mut Vec<CompositeGroup>, mut group: CompositeGroup) {
    let merged = match (
        groups
            .last_mut()
            .and_then(CompositeGroup::as_direct_run_mut),
        group.as_direct_run_mut(),
    ) {
        (Some(items), Some(more)) => {
            items.append(more);
            true
        }
        _ => false,
    };
    if !merged {
        groups.push(group);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stark_model::geom::TileCoord;

    /// **The two walks are the same cull.** [`culled`] picks between scanning the
    /// layer and probing the viewport purely on which is cheaper, so the one thing
    /// that must never depend on that choice is the answer — a disagreement would drop
    /// paint from the picture at exactly one zoom level, which is the kind of bug that
    /// gets blamed on the compositor for a week.
    ///
    /// Driven over map/rect pairs that put the branch on both sides of its own
    /// threshold, including the equality case where it flips.
    #[test]
    fn both_culling_walks_pick_the_same_tiles() {
        let map: rpds::HashTrieMap<TileCoord, i32> = (0..40)
            .map(|i| (TileCoord::new(i % 7, i / 7), i))
            .collect::<Vec<_>>()
            .into_iter()
            .fold(rpds::HashTrieMap::new(), |m, (c, v)| m.insert(c, v));

        let sorted = |mut v: Vec<(TileCoord, i32)>| {
            v.sort_by_key(|(c, _)| (c.y, c.x));
            v
        };
        // Both routes, asked directly, so the test does not depend on which one
        // `culled` would have chosen for a given rect.
        let by_scan = |rect: TileRect| {
            sorted(
                map.iter()
                    .filter(|(c, _)| rect.contains(**c))
                    .map(|(c, v)| (*c, *v))
                    .collect(),
            )
        };
        let by_probe = |rect: TileRect| {
            sorted(
                rect.coords()
                    .filter_map(|c| map.get(&c).map(|v| (c, *v)))
                    .collect(),
            )
        };

        for rect in [
            TileRect::ALL,
            TileRect::EMPTY,
            // Wholly inside the painted region, wholly outside, and straddling it —
            // the three ways a viewport can sit over a painting.
            TileRect {
                min: (1, 1),
                max: (3, 3),
            },
            TileRect {
                min: (50, 50),
                max: (60, 60),
            },
            TileRect {
                min: (-4, 2),
                max: (2, 9),
            },
            // A single tile, which is the probe branch at its most lopsided.
            TileRect {
                min: (3, 2),
                max: (3, 2),
            },
        ] {
            // The scan is the reference: it is defined for every rect, including
            // [`TileRect::ALL`].
            let want = by_scan(rect);
            let got = sorted(culled(&map, Some(rect)).map(|(c, v)| (c, *v)).collect());
            assert_eq!(got, want, "culled disagreed on {rect:?}");
            // The probe is only asked where probing is *finite*, which is the size
            // guard inside `culled` stated from the outside. `TileRect::ALL` counts
            // 1.8e19 tiles, so walking its coords is not a slow test but a hung one —
            // which is exactly the case the guard exists to keep the renderer out of,
            // and this loop hit it before the guard was mirrored here.
            if rect.count() <= 10_000 {
                assert_eq!(want, by_probe(rect), "the walks disagree on {rect:?}");
            }
        }
        // An unmeasurable box claims everything, which is the safe direction.
        assert_eq!(culled(&map, None).count(), map.size());
    }

    /// **"An ordinary document is one `Run`"**, which `composite_groups` claims and
    /// nothing checked.
    ///
    /// It is the whole argument for cutting the draw list into groups at all: a
    /// document that uses no blend modes, no clipping and no groups must cost the
    /// compositor exactly what it did before groups existed, or the feature is a tax
    /// on everyone who does not use it. That is measurable, so it is measured — the
    /// habit `a_trim_never_drops_below_the_epochs_peak_demand` and
    /// `a_scope_hands_its_scratch_back_as_it_goes` already keep, extended to a
    /// *performance* claim rather than a correctness one (C8).
    ///
    /// Asked of the grouping rule directly rather than through an `Engine`, so it
    /// needs no GPU: what decides a run boundary is [`group_of`] plus [`push_merging`],
    /// and both are functions of `CompositeParams`.
    ///
    /// Through those two rather than a copy of them. This test used to re-implement
    /// the merge — the same `as_direct_run_mut` pair, written out again — so it pinned
    /// a transcription of the rule and would have gone on passing if the rule itself
    /// changed underneath it.
    #[test]
    fn plain_layers_merge_into_one_run() {
        use crate::document::CompositeParams;
        use stark_model::document::BlendMode;

        // A stand-in item; what is being tested is how many groups the merge
        // leaves, not what is in them.
        let items = || {
            vec![CompositeItem::Matte(MatteDraw {
                rect: [0.0; 4],
                flags: 1.0,
                channels: [0.0; 4],
                resid: [0.0; 4],
                opacity: 1.0,
                ramp: None,
            })]
        };
        let plain = CompositeParams::IDENTITY;

        // Six ordinary layers, through the very walk `composite_stack` runs.
        let mut groups: Vec<CompositeGroup> = Vec::new();
        for _ in 0..6 {
            let g = group_of(plain, items(), Vec::new()).expect("a layer with an item draws");
            push_merging(&mut groups, g);
        }
        assert_eq!(
            groups.len(),
            1,
            "plain layers did not collapse into one run"
        );

        // And a layer that *does* need isolating breaks the run — otherwise the
        // test above would pass on a rule that merged everything unconditionally,
        // which would composite blend modes against the wrong backdrop.
        let isolating = CompositeParams {
            blend: BlendMode::Multiply,
            ..plain
        };
        push_merging(
            &mut groups,
            group_of(isolating, items(), Vec::new()).expect("a layer with an item draws"),
        );
        assert_eq!(
            groups.len(),
            2,
            "a blended layer merged into the run below instead of breaking it",
        );
    }

    /// A layer that draws nothing produces no group at all — the empty-skip both
    /// walks make, which is what lets a culled-away document cost nothing.
    #[test]
    fn a_layer_with_nothing_to_draw_makes_no_group() {
        use crate::document::CompositeParams;
        assert!(group_of(CompositeParams::IDENTITY, Vec::new(), Vec::new()).is_none());
    }
}
