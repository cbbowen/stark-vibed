//! The wgpui surface the engine paints into (§6.4, §11).
//!
//! The web frontend's `Renderer` builds the device, binds a `wgpu::Surface` to a
//! `<canvas>` and configures it. None of that happens here: **wgpui owns the device**
//! and hands out a double-buffered pair of textures instead, so what this holds is a
//! [`WgpuSurfaceHandle`] and an [`Engine`] built on the device behind it. The engine
//! renders straight into the back buffer and the swap is a pointer swap — the same
//! bargain the browser canvas makes, with no readback and no encode.

use stark_engine::Extent2;
use stark_engine::command::{InputCommand, ViewCommand};
use stark_engine::{Engine, GpuContext, ObservableState, Output, Transfer, ViewTransform};
use stark_ui::prefs::Hdr;
use wgpui::{WgpuSurfaceHandle, Window};

/// The format the engine renders through and the transfer the window reads it in
/// (§6.5), read off the swapchain wgpui configured (`vendor/wgpui/VENDORING.md`,
/// patch 6): the engine's texels are composited into it unconverted. **Never an
/// sRGB format**: the media pass encodes the transfer itself.
fn target_for(window: &Window) -> (wgpu::TextureFormat, Transfer) {
    match window.surface_color_space() {
        Some(wgpu::SurfaceColorSpace::ExtendedSrgbLinear) => {
            (wgpu::TextureFormat::Rgba16Float, Transfer::Linear)
        }
        _ => (wgpu::TextureFormat::Rgba8Unorm, Transfer::Srgb),
    }
}

/// One surface, one engine, and the resize that keeps them agreeing.
pub struct Renderer {
    surface: WgpuSurfaceHandle,
    engine: Engine,
    /// The viewport the engine was last told about, in device px — see
    /// [`paint`](Self::paint), which is where it is corrected.
    viewport: (u32, u32),
    /// How the window reads the surface's texels (§6.5), fixed with the format.
    transfer: Transfer,
    /// The navigator's miniature, once something has asked for one (`crate::navigator`).
    ///
    /// A **second surface**, not a picture: the engine renders the committed document
    /// straight into it and the swap is a pointer swap, exactly as the canvas's is —
    /// one document seen twice. So nothing here holds pixels, and a refresh costs a
    /// render rather than a render plus a readback.
    overview: Option<Overview>,
}

/// The miniature's surface and the pass-A attachments it renders through.
///
/// Its own [`Offscreen`](stark_engine::Offscreen) rather than the screen's, because
/// those are the size of the *target*: sharing them would have the miniature resize
/// the canvas's attachments away on every frame and back again on the next.
struct Overview {
    surface: WgpuSurfaceHandle,
    targets: stark_engine::Offscreen,
    /// The surface size the last render covered, in device px — the canvas keeps the
    /// same number under the same argument ([`viewport`](Renderer::viewport)).
    drawn: (u32, u32),
}

impl Renderer {
    /// Bind a surface to `window` and build an engine on the device behind it.
    ///
    /// `None` when wgpui is not on its wgpu renderer, which is the one platform
    /// answer that leaves nothing to paint with — the view reports it rather than
    /// panicking, for the reason the web frontend's `StartupFailure` gives.
    pub fn new(window: &Window) -> Option<Self> {
        let (width, height) = device_pixels(window);
        let (format, transfer) = target_for(window);
        let surface = window.create_wgpu_surface(width, height, format)?;
        // The engine is *given* its wgpu resources (CLAUDE.md), and here that is
        // forced rather than chosen: the handle carries a device and a queue and
        // nothing else. It also replaces wgpui's device callbacks with the engine's
        // (`GpuContext::from_parts`), which is the right way round — the engine is
        // what has to stop issuing work when the device dies.
        let gpu = GpuContext::from_parts(surface.device().clone(), surface.queue().clone());
        let mut engine = Engine::new(gpu, format, Extent2::new(width, height));
        // The transfer is the surface's from the first frame; `apply_hdr` moves only
        // the headroom.
        engine.process(ViewCommand::SetOutput(Output::new(transfer, 1.0)));
        Some(Self {
            surface,
            engine,
            viewport: (width, height),
            transfer,
            overview: None,
        })
    }

    /// Whether the window can show anything above white (§6.5).
    pub fn hdr_capable(&self) -> bool {
        self.transfer.is_hdr()
    }

    /// Tell the engine what the window is (§6.5): the surface's transfer — stated
    /// even with the switch off, since `Command::ToggleHdr`'s `enabled` reads it —
    /// and the headroom: the display's where reported (`Window::display_headroom`),
    /// `choice`'s where not, and 1 with the switch off.
    pub fn apply_hdr(&mut self, choice: Hdr, display_headroom: Option<f32>) {
        let headroom = if choice.on && self.hdr_capable() {
            display_headroom.unwrap_or_else(|| choice.clamped_headroom())
        } else {
            1.0
        };
        self.engine
            .process(ViewCommand::SetOutput(Output::new(self.transfer, headroom)));
    }

    /// Send a command to the engine — the **only** way to move engine state through a
    /// `Renderer` (§4), deliberately, and for the reason the web frontend's
    /// `Renderer::process` spells out: a named `set_*` beside it is a second spelling
    /// that skips whatever the first one also did.
    pub fn process(&mut self, command: impl Into<InputCommand>) {
        self.engine.process(command);
    }

    /// The engine's cheap UI-facing projection, read back after each command (§5).
    ///
    /// Cheap by construction — the layer roster is shared rather than copied — but
    /// not free, so the view keeps the answer rather than asking per frame.
    pub fn observe(&self) -> ObservableState {
        self.engine.observe()
    }

    /// Serialize the document — the action log, not the pixels (§8).
    ///
    /// `resolvable` names content the *opener* is expected to already have, so it is
    /// left out of the file. This frontend passes none: see `crate::files`.
    pub fn save_bytes_resolvable(
        &self,
        resolvable: &[stark_model::AssetId],
    ) -> stark_engine::Result<Vec<u8>> {
        self.engine.save_bytes_resolvable(resolvable)
    }

    /// What `file` names that neither it carries nor this engine holds — settle it
    /// before [`load_document`](Self::load_document), which refuses otherwise.
    pub fn unresolved_content(
        &self,
        file: &stark_model::DocumentFile,
    ) -> Vec<stark_model::AssetNeed> {
        self.engine.unresolved_content(file)
    }

    /// Replace the document by replaying a loaded log (§8) — its whole undo history
    /// comes back with it, because the file *is* the history.
    ///
    /// Fails leaving the open document untouched, which is what makes a refused file
    /// cost nothing.
    pub fn load_document(&mut self, file: &stark_model::DocumentFile) -> stark_engine::Result<()> {
        self.engine.load_document(file)
    }

    /// Import a brush shape, returning the content id that names it (§6.6).
    ///
    /// Content-addressed, so importing the same image twice is free and lands on the
    /// same id — which is what lets a shipped stamp, a library entry and a stamp that
    /// arrived in a save file all be one asset.
    pub fn import_brush_id(&self, png: &[u8]) -> Result<stark_model::AssetId, String> {
        self.engine.import_brush(png).map_err(|e| e.to_string())
    }

    /// The canonical bytes of an imported asset — for a card's picture, and for
    /// seeding a session so peers can fetch it by hash.
    pub fn asset_bytes(&self, id: stark_model::AssetId) -> Option<Vec<u8>> {
        self.engine.asset_bytes(id)
    }

    /// Import a canvas substrate's height map (§6.4).
    ///
    /// `Result` rather than the web frontend's `Option`-and-a-log: this frontend has
    /// somewhere to put a failure a person can act on (`Canvas::report`), and a
    /// substrate that would not import is exactly the kind they should hear about.
    pub fn import_substrate(&mut self, png: &[u8]) -> Result<stark_model::SubstrateId, String> {
        self.engine.import_substrate(png).map_err(|e| e.to_string())
    }

    /// Take a substrate's bytes under the id that asked for them.
    ///
    /// Re-derives the id and refuses bytes that do not match, which is what makes a
    /// catalog file that changed out from under a document a caught error rather than
    /// a deposit through the wrong substrate (§6.4).
    pub fn accept_substrate(
        &mut self,
        id: stark_model::SubstrateId,
        png: &[u8],
    ) -> Result<(), String> {
        self.engine
            .accept_substrate(id, png)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    /// The canonical height map of a loaded substrate.
    pub fn substrate_bytes(&self, id: stark_model::SubstrateId) -> Option<Vec<u8>> {
        self.engine.substrate_bytes(id)
    }

    /// Render a picture and hand back a future for its readback (§15.6).
    ///
    /// The future does **not** borrow the renderer, which is the point: the caller
    /// can go back to painting while the GPU→CPU copy is in flight. A one-shot — the
    /// attachments are allocated for this render and dropped with it, so a large
    /// export does not park its buffers for the session.
    pub fn export(
        &mut self,
        frame: Option<stark_model::document::LayerId>,
        scale: stark_engine::ExportScale,
        background: stark_engine::Background,
        content: stark_engine::Rendered,
    ) -> stark_engine::Result<
        impl std::future::Future<Output = stark_engine::Result<stark_engine::RgbaImage>> + use<>,
    > {
        self.engine.export(
            &mut stark_engine::Offscreen::default(),
            frame,
            scale,
            background,
            content,
        )
    }

    /// Take a picture's bytes under the id that asked for them (§23) — the third
    /// store an arriving asset can belong in, beside a shape and a substrate.
    ///
    /// Only a peer ever supplies one: no build ships a picture, so this has no file
    /// path and no library behind it (`crate::collab`).
    pub fn accept_picture(&self, expected: stark_model::AssetId, png: &[u8]) -> Result<(), String> {
        self.engine
            .accept_picture(expected, png)
            .map_err(|e| e.to_string())
    }

    // --- collaboration (§12) — thin engine delegates for the session glue in
    // `crate::collab`. The web frontend's `Renderer` carries the same run, and the
    // two lists are short for the same reason: what a session *is* belongs to the
    // engine, and what a link is belongs to `stark_ui::collab`. What is left over
    // here is delegation. ---

    /// Convert the current document into a shared one, authored as `identity`.
    pub fn start_collaboration(&mut self, identity: impl Into<stark_engine::Identity>) {
        self.engine.start_collaboration(identity);
    }

    /// Replace the document with a joined session's log — or `Err`, leaving this
    /// client's own document alone, when the session is in a color space this build
    /// cannot render (§6.7).
    pub fn join_collaboration(
        &mut self,
        file: &stark_model::DocumentFile,
        identity: impl Into<stark_engine::Identity>,
    ) -> stark_engine::Result<()> {
        self.engine.join_collaboration(file, identity)
    }

    /// Leave a shared session: keep the canvas and its history, stop broadcasting.
    pub fn end_collaboration(&mut self) {
        self.engine.end_collaboration();
    }

    /// Snapshot the document — the full shared log with the assets it references,
    /// which is what a joining peer is served.
    pub fn document_file(&self) -> stark_model::DocumentFile {
        self.engine.document_file()
    }

    /// Every imported asset's canonical bytes, for seeding a session's blob store.
    pub fn all_asset_bytes(&self) -> Vec<(stark_model::AssetId, Vec<u8>)> {
        self.engine.all_asset_bytes()
    }

    /// Integrate one remote action; `true` if it was new.
    pub fn merge_remote(&mut self, action: stark_model::document::Action) -> bool {
        self.engine.merge_remote(action)
    }

    /// Drain locally-committed actions awaiting broadcast.
    pub fn take_outbox(&mut self) -> Vec<stark_model::document::Action> {
        self.engine.take_outbox()
    }

    /// Whether [`take_presence`](Self::take_presence) would do anything — a `&self`
    /// test, so an idle frame of a shared session takes no mutable borrow.
    pub fn presence_due(&self, now: f64) -> bool {
        self.engine.presence_due(now)
    }

    /// Drain this client's presence latch, and expire peers gone quiet (§17.5). The
    /// frame is `None` when there is nothing new to say; `repaint` reports that the
    /// expiry took a departed peer's paint off the canvas.
    pub fn take_presence(&mut self, now: f64) -> stark_engine::PresenceTick {
        self.engine.take_presence(now)
    }

    // No `leaving_presence` here, and its absence is a fact about the frontend rather
    // than an oversight: leaving a session is not an act this window offers yet, so
    // there is nothing to say goodbye *at*. A session ends when the process does, and
    // peers drop this client on the presence timeout instead of at once. The engine
    // has the farewell whenever a Leave arrives to send it.

    /// Integrate a peer's presence; `true` when the **canvas** needs repainting.
    ///
    /// Narrower than "anything changed": a moved cursor or a switched layer is chrome,
    /// and this frontend draws none of it yet — so here the narrow answer is the only
    /// one, and a remote pointer move costs nothing at all.
    ///
    /// `now` dates the frame for expiry — the caller's clock, because the engine's own
    /// only advances when something drains it.
    pub fn merge_presence(
        &mut self,
        actor: stark_model::document::ActorId,
        frame: stark_model::PeerFrame,
        now: f64,
    ) -> bool {
        self.engine.merge_presence(actor, frame, now)
    }

    /// The view a pointer position is mapped through.
    pub fn view(&self) -> ViewTransform {
        self.engine.view()
    }

    /// The handle the element composites. Cloned per frame, which costs two atomic
    /// bumps — the element wants it by value.
    pub fn surface(&self) -> WgpuSurfaceHandle {
        self.surface.clone()
    }

    /// Whether the element has resized the surface out from under the viewport the
    /// engine was last told about.
    ///
    /// A repaint the frame owes even though no command asked for one: the resize
    /// happens in the element's *prepaint*, so the frame that caused it has already
    /// drawn the old picture stretched to the new size.
    pub fn resized(&self) -> bool {
        self.surface.size() != self.viewport
    }

    // --- the navigator's miniature (§11) --------------------------------------

    /// What a miniature of the whole piece would be, at the largest size that fits
    /// `into` — **exactly what an export would write** (§15.6), because it is the same
    /// call: [`Engine::export_plan`] answers the rect, and the plan it returns *is*
    /// the view the miniature renders through. So the overview cannot come to
    /// disagree with the picture a file would hold.
    ///
    /// `None` on an unpainted, unframed canvas, where the rect the engine would fall
    /// back to is the *viewport* — a picture of the window presented as the piece.
    /// An unbounded canvas with nothing on it has no overview, and saying so is the
    /// honest answer.
    pub fn overview_plan(
        &self,
        frame: Option<stark_model::document::LayerId>,
        into: Extent2,
    ) -> Option<stark_engine::ExportPlan> {
        self.engine
            .export_plan(frame, stark_engine::ExportScale::Fit(into))
            .ok()
    }

    /// Draw the miniature `plan` describes into the overview surface and swap it.
    ///
    /// The surface is made on the first call, at the plan's own size; from then on the
    /// element resizes it from its own bounds, one frame behind — so the view is given
    /// the surface's *actual* size as its viewport rather than the plan's. The two
    /// differ by a pixel of rounding in the steady state and by a whole aspect on the
    /// frame after the piece is reshaped, and taking the target's word for it is what
    /// keeps that frame a slightly wider crop rather than a stretched picture. That
    /// resize also *discards* what was drawn, which is what
    /// [`overview_resized`](Self::overview_resized) is for.
    ///
    /// The **committed** document, over the substrate: an overview is a picture of the
    /// piece as it stands, and following the stroke in hand would mean compositing
    /// every tile in the document at pointer rate to show what the canvas beside it is
    /// already showing full size.
    pub fn paint_overview(&mut self, window: &Window, plan: &stark_engine::ExportPlan) -> bool {
        if self.overview.is_none() {
            // `None` where wgpui is not on its wgpu renderer, which is the same
            // answer the canvas's own surface gives and is reported the same way:
            // nothing to draw the miniature with, so it is simply not drawn.
            let Some(surface) =
                window.create_wgpu_surface(plan.size.width, plan.size.height, self.format())
            else {
                return false;
            };
            self.overview = Some(Overview {
                surface,
                targets: stark_engine::Offscreen::default(),
                // Nothing drawn into it yet, and no size it could have been drawn at.
                drawn: (0, 0),
            });
        }
        let Some(ov) = self.overview.as_mut() else {
            return false;
        };
        let Some((target, size)) = ov.surface.back_view_with_size() else {
            return false;
        };
        let mut view = plan.view();
        view.viewport = Extent2::new(size.0, size.1);
        self.engine.render_into(
            &mut ov.targets,
            &target,
            view,
            stark_engine::Background::Substrate,
            stark_engine::Rendered::Committed,
        );
        ov.drawn = size;
        ov.surface.swap_buffers();
        true
    }

    /// Whether the element has resized the miniature's surface out from under the
    /// picture drawn into it — the miniature's [`resized`](Self::resized), and
    /// sharper: a resized surface is two **new** textures, so what was drawn is not
    /// stretched, it is gone. Nothing else asks for it back, since the refresh policy
    /// is otherwise keyed on the document's revision and a resize does not move that.
    pub fn overview_resized(&self) -> bool {
        self.overview
            .as_ref()
            .is_some_and(|ov| ov.surface.size() != ov.drawn)
    }

    /// The miniature's handle, for the element that composites it — `None` until
    /// something has painted one.
    pub fn overview_surface(&self) -> Option<WgpuSurfaceHandle> {
        self.overview.as_ref().map(|ov| ov.surface.clone())
    }

    /// Let the miniature go: its surface leaves the registry with the last handle
    /// (`WgpuSurfaceHandle`), so putting the navigator away really does give the
    /// textures back rather than keeping a second copy of the piece on the GPU.
    pub fn drop_overview(&mut self) {
        self.overview = None;
    }

    /// The format both surfaces are configured in — the engine's own, since its texels
    /// are composited unconverted (§6.5).
    fn format(&self) -> wgpu::TextureFormat {
        self.surface.format()
    }

    // --- lighting (§6.3) ------------------------------------------------------

    /// Whether `id`'s bytes have already been decoded and prefiltered, so a switch to
    /// it costs nothing.
    pub fn environment_loaded(&self, id: stark_engine::EnvironmentId) -> bool {
        self.engine.environment_loaded(id)
    }

    /// Hand an environment's HDR bytes over, readying it *without* switching to it —
    /// the switch is a command, and this is not (§4).
    pub fn register_environment(
        &mut self,
        id: stark_engine::EnvironmentId,
        hdr: Vec<u8>,
    ) -> Result<(), String> {
        self.engine
            .register_environment(id, hdr)
            .map_err(|e| e.to_string())
    }

    /// Render the canvas into the back buffer and swap it to the front.
    ///
    /// `swap_buffers` rather than `present`: this runs inside a frame wgpui is
    /// already building, so the swap is what that frame composites, and the
    /// `request_present` the other half of `present` would add asks for a further
    /// frame nobody needs.
    pub fn paint(&mut self) {
        // Both at once, so a concurrent resize cannot hand back a view of one size
        // and dimensions of another.
        let Some((target, size)) = self.surface.back_view_with_size() else {
            return;
        };
        // The viewport is taken from the buffer rather than from the window, because
        // the buffer is what the render lands in: a view that disagreed with its
        // target would put every stroke somewhere other than under the pointer.
        if size != self.viewport {
            self.viewport = size;
            self.engine
                .process(ViewCommand::Resize(Extent2::new(size.0, size.1)));
        }
        self.engine.render(&target);
        self.surface.swap_buffers();
    }
}

/// The window's drawable area in **device px** — what a surface is sized in, where
/// every [`Pixels`](wgpui::Pixels) the layout speaks in is logical.
fn device_pixels(window: &Window) -> (u32, u32) {
    let scale = window.scale_factor();
    let size = window.viewport_size();
    // Floored at 1: a zero-sized surface is not a texture, and a minimized window
    // is an ordinary state rather than a failure.
    let px = |v: wgpui::Pixels| ((f32::from(v) * scale).round() as u32).max(1);
    (px(size.width), px(size.height))
}
