//! Presenting the canvas: the compositor's draw list, the screen frame, and export
//! (§6.3, §15.6).
//!
//! One path serves all three consumers, which is what keeps them from disagreeing
//! about what the document looks like. [`Engine::render_view`] takes a view, a
//! ground and somewhere to put the pass-A attachments, and every caller differs only
//! in those: the screen renders through the session's view with chrome, the
//! navigator's miniature through a planned rect without it, and an export through
//! the same planned rect into a texture it then reads back. "Export" was a
//! screenshot of the viewport for exactly as long as `render` read `session.view`
//! instead of taking one.

use super::Engine;
use crate::document::{CompositeParams, DocState, Layer, LayerContent, LayerId};
use crate::geom::{Extent2, TileRect, ViewTransform};
use crate::gpu::{
    CompositeGroup, CompositeItem, CompositeScene, FilterDraw, GpuContext, MatteDraw, Offscreen,
    SelectionOutline,
};
use crate::image::RgbaImage;
use crate::{EngineError, Result};

/// What sits under the paint when rendering (§15.6).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Background {
    /// The document's substrate color, lit and textured by the canvas weave —
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

/// Which [`Compositor`] a render's offscreen attachments come from.
///
/// Compositing runs through pass-A attachments the size of the target, so *whose*
/// they are decides who pays for a resize. The surface's are kept from frame to
/// frame; anything rendered beside them is a different size and brings its own, so
/// the screen's are never resized out from under it — and never rebuilt on the next
/// frame to recover. That mattered as soon as something rendered off-screen
/// *repeatedly*: the navigator's miniature is one render per edit, and sharing the
/// surface's attachments made it two rebuilds of window-sized textures and a full
/// recomposite per edit.
enum Attachments<'a> {
    /// The surface's own, cached across frames ([`Engine::compositor`]).
    Surface,
    /// The caller's, so whether they outlive the call is decided by whoever knows
    /// whether the render repeats — see [`Offscreen`].
    Offscreen(&'a mut Offscreen),
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
    pub min: crate::geom::Vec2,
    pub max: crate::geom::Vec2,
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
/// the dangerous direction. `wgpu::Limits::default()` caps 2D textures at 8192 and the
/// frontend requests that, so on the app's device the two agreed by coincidence; the
/// headless device ([`GpuContext::headless`]) requests only `downlevel_defaults` —
/// 2048, the web/WebGL2 floor — and there a 4096-px export passed a check written
/// against 8192 and then asked for a texture the device was never granted. A guard
/// that has to be kept in step with a limit it does not read is a guard that is
/// already out of step somewhere.
///
/// It also lets the ceiling *rise*: the adapters this runs on report far more than
/// 8192 (32768 is common), so a frontend that requests more gets more, and this
/// follows it with nothing to update.
///
/// [`GpuContext::headless`]: crate::gpu::GpuContext::headless
fn max_export_dim(gpu: &GpuContext) -> u32 {
    gpu.device.limits().max_texture_dimension_2d
}

impl Engine {
    /// Render the current canvas (preview if stroking, else committed) into
    /// `target`, through the session's own pan/zoom (§6.4).
    pub fn render(&mut self, target: &wgpu::TextureView) {
        self.render_view(
            target,
            self.session.view,
            Background::Substrate,
            Chrome::Shown,
            Rendered::Live,
            Attachments::Surface,
        );
    }

    /// Render the document through `view` into a target that is **not** the engine's
    /// own surface — a second surface showing the same document (§11).
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
    /// `view.viewport` in size — a surface texture configured to match.
    ///
    /// No chrome: a selection outline belongs to the surface you are painting on, not
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
            background,
            Chrome::Hidden,
            content,
            Attachments::Offscreen(into),
        );
    }

    /// The texture format this engine's pipelines render to. A frontend configuring a
    /// second surface for [`render_into`](Self::render_into) has to match it.
    pub fn target_format(&self) -> wgpu::TextureFormat {
        self.target_format
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
    /// Private, with [`Engine::export`] and [`Engine::render_into`] as the two
    /// consumers: what a caller may choose is a view, a ground and where the
    /// attachments live, never whether chrome is drawn (it is, for the screen alone)
    /// nor how the two are wired together.
    fn render_view(
        &mut self,
        target: &wgpu::TextureView,
        view: ViewTransform,
        background: Background,
        chrome: Chrome,
        content: Rendered,
        attachments: Attachments,
    ) {
        // The fold is rebuilt lazily (`Engine::mark_live_stale`), and this is the
        // read that services it: once per frame painted, whatever arrived since.
        if matches!(content, Rendered::Live) {
            self.flush_live();
        }
        let doc = match content {
            Rendered::Live => self.presented(),
            Rendered::Committed => self.timeline.current(),
        };
        // Only what this view can show (§6.3). The draw list is otherwise every
        // populated tile of every visible layer, whatever the viewport.
        let groups = self.composite_groups(doc, None, visible_tiles(view));

        // The substrate is document state now (§15.5), so the ground a
        // piece was painted on travels with it instead of living in whichever
        // frontend happened to render it.
        let bg_channels = self.color_space.rgb_to_channels(doc.background);
        let bg_resid = {
            let r = self.color_space.rgb_to_resid(doc.background);
            [r[0], r[1], r[2], 0.0]
        };
        // Chrome never reaches a file: an exported image gets no selection outline
        // (§15.6). Keyed on `chrome`, deliberately *not* on the
        // background — a substrate export is still an export, and tying the two
        // together silently leaked the outline into every opaque PNG.
        //
        // Own the masks (a handful of `Arc` bumps) so the borrow of `doc` — and with
        // it of `self` — ends before the compositor is borrowed mutably below.
        let outlines: Vec<(crate::document::Selection, Option<[f32; 3]>)> = match chrome {
            Chrome::Hidden => Vec::new(),
            Chrome::Shown => self.visible_selections(),
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
        let guide_scenes: Vec<crate::guides::GuideScene> = match chrome {
            Chrome::Hidden => Vec::new(),
            Chrome::Shown => self
                .session
                .guides
                .iter()
                .filter(|g| g.visible)
                .map(|g| g.scene())
                .collect(),
        };
        let scene = CompositeScene {
            background: bg_channels,
            background_resid: bg_resid,
            groups: &groups,
            outlines: &outlines,
            transparent: background == Background::Transparent,
            guides: &guide_scenes,
        };
        match attachments {
            Attachments::Surface => {
                self.compositor
                    .render(&self.compositor_pipeline, target, view, scene)
            }
            Attachments::Offscreen(into) => into.get(&self.compositor_pipeline).render(
                &self.compositor_pipeline,
                target,
                view,
                scene,
            ),
        }
    }

    /// Render the current canvas to a CPU-side image at the viewport size
    /// (§9). The backbone of golden tests. The target uses the engine's
    /// configured format, so it matches on-screen rendering.
    /// Blocking, and therefore **native-only**: WebGPU has no blocking poll, so
    /// this shape cannot work on the web (see `gpu::readback`). The frontend uses
    /// [`export`](Self::export), which awaits the map.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn render_to_image(&mut self) -> RgbaImage {
        // One render per call, so nothing is kept: the attachments are allocated here
        // and dropped with this `Offscreen`.
        let (target, size) = self.render_offscreen(
            &mut Offscreen::default(),
            self.session.view,
            Background::Substrate,
            Chrome::Shown,
            Rendered::Live,
        );
        let pixels = crate::gpu::readback::read_rgba8_blocking(&self.gpu, &target, size);
        RgbaImage::from_target_bytes(size.width, size.height, pixels, self.target_format)
    }

    /// Render through an explicit view into an offscreen texture, ready to be read
    /// back. Split out so the blocking and async readbacks share every step but
    /// the wait.
    fn render_offscreen(
        &mut self,
        into: &mut Offscreen,
        view: ViewTransform,
        background: Background,
        chrome: Chrome,
        content: Rendered,
    ) -> (wgpu::Texture, Extent2) {
        let size = view.viewport;
        let target = self.offscreen_target(
            "stark export target",
            self.target_format,
            size,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
        // The caller's attachments, not the surface's — see [`Attachments`].
        self.render_view(
            &target_view,
            view,
            background,
            chrome,
            content,
            Attachments::Offscreen(into),
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
            return Err(EngineError::Export(format!(
                "frame is too small to export ({w:.0} × {h:.0} canvas px)"
            )));
        }
        let zoom = match scale {
            ExportScale::Factor(f) => f,
            ExportScale::Width(px) => px as f32 / w,
            ExportScale::Fit(into) => (into.width as f32 / w).min(into.height as f32 / h),
        };
        if !(zoom.is_finite() && zoom > 0.0) {
            return Err(EngineError::Export("export scale must be positive".into()));
        }
        // Round rather than truncate, so a 1× export of a 100.5-px frame is 101
        // rather than silently dropping most of a pixel off two edges.
        let size = Extent2::new(
            (w * zoom).round().max(1.0) as u32,
            (h * zoom).round().max(1.0) as u32,
        );
        let limit = max_export_dim(&self.gpu);
        if size.width > limit || size.height > limit {
            return Err(EngineError::Export(format!(
                "export is {} × {} px; this device's limit is {limit}",
                size.width, size.height
            )));
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
    /// to its own export, while a ground matte is inside and contributes exactly
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
    /// surface rather than into it, so it never touches the screen's; whether its own
    /// outlive the call is the caller's call, and the caller is the only one who knows
    /// (see [`Offscreen`]) — a `&mut Offscreen::default()` for a one-shot, a held one
    /// for a render that repeats.
    ///
    /// ```ignore
    /// let readback = { engine.write().export(&mut own, frame, scale, bg, content)? }; // borrow ends
    /// let image = readback.await;
    /// ```
    /// **This is [`export_view`](Self::export_view) through the plan's own view**,
    /// which is the whole of what "export a frame" adds: [`ExportPlan::view`] already
    /// exists so that the two things which render a planned rect derive it in one
    /// place, and having said that, exporting one is not a second render path. The
    /// tail — render off-screen, hand back a future that owns what it reads — was
    /// written out twice, down to the borrow bargain the doc comment above explains.
    pub fn export(
        &mut self,
        into: &mut Offscreen,
        frame: Option<LayerId>,
        scale: ExportScale,
        background: Background,
        content: Rendered,
    ) -> Result<impl std::future::Future<Output = RgbaImage> + use<>> {
        // Ahead of the render, so a frame too small or too large to export is
        // refused before anything is drawn — and with the message that names the
        // *frame*, since `export_view`'s checks are then satisfied by construction.
        let plan = self.export_plan(frame, scale)?;
        self.export_view(into, plan.view(), background, content)
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
    /// Errors mirror [`export_plan`](Self::export_plan)'s: a degenerate or
    /// non-finite view, or a viewport past the device's texture limit, is reported
    /// rather than surfacing as a wgpu validation panic.
    pub fn export_view(
        &mut self,
        into: &mut Offscreen,
        view: ViewTransform,
        background: Background,
        content: Rendered,
    ) -> Result<impl std::future::Future<Output = RgbaImage> + use<>> {
        if !(view.zoom.is_finite() && view.zoom > 0.0 && view.center.is_finite()) {
            return Err(EngineError::Export("view must be finite".into()));
        }
        let size = view.viewport;
        let limit = max_export_dim(&self.gpu);
        if size.width == 0 || size.height == 0 || size.width > limit || size.height > limit {
            return Err(EngineError::Export(format!(
                "render is {} × {} px; this device's limit is {limit}",
                size.width, size.height
            )));
        }
        // No chrome: a selection outline or any other on-canvas affordance is a
        // thing to draw *with*, never a thing to ship.
        let (target, size) = self.render_offscreen(into, view, background, Chrome::Hidden, content);
        // Captured, not read through `self`: the future deliberately does not
        // borrow the engine.
        let gpu = self.gpu.clone();
        let format = self.target_format;
        Ok(async move {
            let pixels = crate::gpu::readback::read_rgba8(&gpu, &target, size).await;
            RgbaImage::from_target_bytes(size.width, size.height, pixels, format)
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
    /// the painting, a ground under it (§15.4.4). The compositor
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
    /// the draw list. `None` culls nothing — see [`visible_tiles`].
    ///
    /// [`visible_tiles`]: fn@visible_tiles
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
            let Some(layer) = doc
                .layer(id)
                .filter(|l| l.visible && l.composite.opacity > 0.0)
            else {
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
            if !layer.visible || layer.composite.opacity <= 0.0 {
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
            // **The cull can empty a layer that has paint**, so this now fires for
            // a document scrolled away from as well as for one not yet painted on.
            // That is sound on the identity above rather than on the two cases
            // happening to coincide: `blend_common.wesl::merge` with a transparent
            // source is `cb` exactly — `cs.a` is 0, so both source terms vanish and
            // the aux sum adds nothing — for every mode and both clip states.
            if own.is_empty() && carried.is_empty() {
                continue;
            }
            // **Where the layer's composite params go**, and the only place the
            // question is asked (§14.4.3, §14.7):
            //
            // - A **leaf** is the whole of what it draws, so its params are its run's.
            // - A **group's base** is a *member* of the group, not the group. Its own
            //   content composites with `IDENTITY` and the layer's params are applied
            //   once, to the composited whole. Every one of the three would be wrong
            //   applied twice: a blend mode would combine the base with itself, a clip
            //   would clip the base to its own coverage, and an opacity would fade it
            //   to `a²` — which is what happened for as long as the three were carried
            //   separately and the item builder tagged them with one of them.
            let mut group = if carried.is_empty() {
                CompositeGroup::leaf(layer.composite, own)
            } else {
                let mut members = Vec::with_capacity(carried.len() + 1);
                if !own.is_empty() {
                    members.push(CompositeGroup::leaf(CompositeParams::IDENTITY, own));
                }
                members.extend(carried);
                CompositeGroup::stack(layer.composite, members)
            };
            // Merge into the run below when neither side needs isolating — the
            // fast path, and the reason an ordinary document is one group.
            //
            // `as_direct_run_mut` is the test and the run in one, which is what makes
            // this total. Asking `is_direct` of both sides and then re-matching the
            // two `GroupContent`s behind an `if let` with no else leaves a gap: a
            // group answering "direct" while holding something other than a `Run`
            // would be counted as merged and then silently dropped.
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
            LayerContent::Paint(tiles) => tiles
                .map()
                .iter()
                .filter(|(coord, _)| visible.is_none_or(|r| r.contains(**coord)))
                .map(|(coord, handle)| CompositeItem::Tile {
                    coord: *coord,
                    handle: handle.clone(),
                    opacity: 1.0,
                })
                .collect(),
            LayerContent::Matte { region, paint } => {
                let rect = match region.rect() {
                    Some((min, max)) => [min.x, min.y, max.x, max.y],
                    // The whole plane: the shader never reads the rect (`flags`
                    // routes past it), so zeros rather than sentinels.
                    None => [0.0; 4],
                };
                let flags = match region {
                    crate::document::MatteRegion::OutsideRect { .. } => 0.0,
                    crate::document::MatteRegion::Everything => 1.0,
                };
                // sRGB in the log, working-space channels on the GPU — the same
                // conversion the brush color gets, so a matte means the same
                // color in an Oklab and a Mixbox document. A gradient converts
                // every stop the same way, once per item build, and the shader
                // interpolates in the working space (§22.4).
                let (channels, resid, ramp) = match paint {
                    crate::document::MattePaint::Solid(color) => (
                        self.color_space.rgb_to_channels(*color),
                        {
                            let r = self.color_space.rgb_to_resid(*color);
                            [r[0], r[1], r[2], 0.0]
                        },
                        None,
                    ),
                    crate::document::MattePaint::Gradient { gradient, axis } => {
                        let mut ramp = stark_shaders::mirror::matte::Ramp::default();
                        let stops = gradient.stops();
                        ramp.p[0] = stops.len() as f32;
                        ramp.axis = match axis {
                            crate::document::GradientAxis::Linear { from, to } => {
                                [from.x, from.y, to.x, to.y]
                            }
                            crate::document::GradientAxis::Radial { center, radius } => {
                                ramp.p[1] = 1.0;
                                [center.x, center.y, *radius, 0.0]
                            }
                        };
                        for (i, stop) in stops.iter().enumerate() {
                            let c = self.color_space.rgb_to_channels(stop.color);
                            let r = self.color_space.rgb_to_resid(stop.color);
                            ramp.stop_c[i] = [c[0], c[1], c[2], stop.t];
                            ramp.stop_r[i] = [r[0], r[1], r[2], 0.0];
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
        self.gpu.device.create_texture(&wgpu::TextureDescriptor {
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
    fn export_rect(&self, frame: Option<LayerId>) -> (crate::geom::Vec2, crate::geom::Vec2) {
        let doc = self.timeline.current();
        // An `Everything` matte has no rect and so defines no frame: naming one
        // falls through to the painted bounds, the same answer as no frame at
        // all — a ground is under the picture, not a crop of it (§15.6).
        if let Some(id) = frame
            && let Some(rect) = doc
                .layer(id)
                .and_then(|l| l.matte_region())
                .and_then(|r| r.rect())
        {
            return rect;
        }
        if let Some((min, max)) = doc.bounds().tile_range() {
            let t = crate::geom::TILE_SIZE as f32;
            return (
                crate::geom::Vec2::new(min.x as f32 * t, min.y as f32 * t),
                crate::geom::Vec2::new((max.x + 1) as f32 * t, (max.y + 1) as f32 * t),
            );
        }
        // Everything the viewport shows — a *bound* under rotation, which is the
        // safe direction: an export with nothing painted and no frame should not
        // crop tighter than what the artist is looking at.
        self.session.view.visible_bounds()
    }

    /// The selection masks to outline, and whose each is (§17.3).
    ///
    /// `DocState` holds a selection for every actor that ever made one, because
    /// replay needs them all; only the actors actually *here* are candidates. The log
    /// decides what exists, presence decides what could be shown — and
    /// `show_peer_selections` decides whether it is, since a second contour over the
    /// artwork is a preference rather than a fact about the drawing.
    fn visible_selections(&self) -> Vec<(crate::document::Selection, Option<[f32; 3]>)> {
        let doc = self.presented();
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

/// The tiles a render of `view` can show — the **view-AABB cull** (§6.3).
///
/// Pass A places a tile as the quad `[origin, origin + TILE_SIZE]` and lets the
/// rasterizer clip it, so a tile outside this rect covers no pixel of a
/// viewport-sized target and building a draw for it produces nothing. Skipping it
/// is therefore a pure subtraction: same pixels, less work.
///
/// The bound is conservative twice over, which is the direction that cannot crop a
/// picture. [`ViewTransform::visible_bounds`] is the AABB of the *rotated* viewport,
/// so it covers more canvas than is really on screen; and [`TileRect::covering`]
/// then floors to whole tiles. A supersampled render sees the same rect —
/// [`ViewTransform::supersampled`] scales zoom and viewport together, leaving the
/// canvas region fixed — so culling against the caller's view is consistent with
/// the draw's.
///
/// `None` when the box cannot be measured (a non-finite view, or one so far out
/// that whole tiles fall off the `i32` grid). That is the "claim everything" answer
/// [`TileRect::covering`] leaves to its callers: culling is an optimization, and an
/// optimization that cannot measure its input must do nothing rather than guess.
pub(super) fn visible_tiles(view: ViewTransform) -> Option<TileRect> {
    let (lo, hi) = view.visible_bounds();
    TileRect::covering(lo, hi, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{TILE_SIZE, TileCoord, Vec2};

    /// An upright, unmirrored view of `size` at `zoom`, centred on `center`.
    fn view(center: Vec2, zoom: f32, size: Extent2) -> ViewTransform {
        ViewTransform {
            center,
            zoom,
            ..ViewTransform::identity(size)
        }
    }

    /// The tile a canvas point falls in.
    fn tile_at(c: Vec2) -> TileCoord {
        TileCoord::new(
            (c.x / TILE_SIZE as f32).floor() as i32,
            (c.y / TILE_SIZE as f32).floor() as i32,
        )
    }

    /// **The cull must never crop**, which is the whole risk it carries: it runs on
    /// the export path as well as the screen, so a bound one tile too tight would
    /// silently drop the edge of a saved image rather than fail.
    ///
    /// Asked the way the renderer asks it — walk the pixels the viewport actually
    /// shows, map each back to canvas space, and require its tile to be in the draw
    /// list — rather than by re-deriving the bound, which would only restate the
    /// implementation. That is also what makes it meaningful under rotation and
    /// mirroring, where the visible region is not the box `visible_bounds` returns.
    fn every_pixel_on_screen_keeps_its_tile(v: ViewTransform) {
        let rect = visible_tiles(v).expect("an ordinary view is measurable");
        let (w, h) = (v.viewport.width as f32, v.viewport.height as f32);
        for sy in 0..=32 {
            for sx in 0..=32 {
                let screen = Vec2::new(sx as f32 / 32.0 * w, sy as f32 / 32.0 * h);
                let canvas = v.screen_to_canvas(screen);
                let tile = tile_at(canvas);
                assert!(
                    rect.contains(tile),
                    "screen {screen:?} shows canvas {canvas:?}, in {tile:?}, which                      the cull dropped",
                );
            }
        }
    }

    #[test]
    fn the_cull_keeps_every_tile_the_viewport_shows() {
        let ts = TILE_SIZE as f32;
        let centre = Vec2::new(ts * 1.5, ts * 0.5);
        for size in [
            Extent2::new(64, 64),
            Extent2::new(3 * ts as u32, ts as u32), // wide: worst case under a turn
            Extent2::new(1920, 1080),
        ] {
            for zoom in [0.05, 0.5, 1.0, 8.0] {
                let upright = view(centre, zoom, size);
                every_pixel_on_screen_keeps_its_tile(upright);
                for rotation in [0.3, std::f32::consts::FRAC_PI_4, 2.1, -1.0] {
                    every_pixel_on_screen_keeps_its_tile(ViewTransform {
                        rotation,
                        ..upright
                    });
                }
                every_pixel_on_screen_keeps_its_tile(ViewTransform {
                    flip_h: true,
                    ..upright
                });
            }
        }
    }

    /// And it does cull — otherwise the test above would pass a `visible_tiles` that
    /// simply answered [`TileRect::ALL`].
    #[test]
    fn the_cull_drops_what_the_viewport_cannot_reach() {
        let ts = TILE_SIZE as f32;
        // Strictly inside tile (0, 0): a viewport two px shy of a tile, centred on
        // it, so no edge lands on a tile boundary.
        let v = view(
            Vec2::new(ts * 0.5, ts * 0.5),
            1.0,
            Extent2::new(ts as u32 - 2, ts as u32 - 2),
        );
        let rect = visible_tiles(v).expect("measurable");
        assert!(rect.contains(TileCoord::new(0, 0)));
        for c in [
            TileCoord::new(-1, 0),
            TileCoord::new(1, 0),
            TileCoord::new(0, -1),
            TileCoord::new(0, 1),
            TileCoord::new(7, 7),
        ] {
            assert!(!rect.contains(c), "{c:?} is nowhere near the viewport");
        }
    }

    /// Supersampling scales zoom and viewport together, so it must name the same
    /// tiles — the draw list is built against the caller's view and drawn against
    /// the supersampled one, and a disagreement would crop only when zoomed out.
    #[test]
    fn supersampling_does_not_move_the_cull() {
        let ts = TILE_SIZE as f32;
        let v = view(Vec2::new(ts * 2.0, ts * 2.0), 0.4, Extent2::new(700, 500));
        let plain = visible_tiles(v).expect("measurable");
        for n in [2, 3, 4] {
            assert_eq!(
                visible_tiles(v.supersampled(n)),
                Some(plain),
                "{n}x supersampling changed which tiles are visible",
            );
        }
    }

    /// A view the bound cannot measure culls **nothing** rather than guessing. An
    /// optimization that cannot see its input has to do no harm, and the harm here
    /// would be an empty picture.
    #[test]
    fn an_unmeasurable_view_culls_nothing() {
        let size = Extent2::new(800, 600);
        for bad in [f32::NAN, f32::INFINITY, -f32::INFINITY] {
            assert_eq!(visible_tiles(view(Vec2::new(bad, 0.0), 1.0, size)), None);
        }
        // Far enough out that whole tiles stop fitting an i32 index.
        assert_eq!(visible_tiles(view(Vec2::splat(1e30), 1.0, size)), None);
    }
}
