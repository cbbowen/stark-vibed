//! WebGPU surface rendering (§6.4, §11).
//!
//! The engine renders directly into the canvas's `wgpu::Surface` texture — no
//! readback, no encode. A [`Renderer`] bundles the surface and the engine; the
//! app stores it in a signal, requests a paint after each command (coalesced to
//! one [`Renderer::paint`] per animation frame —
//! [`request_paint`](crate::state::request_paint)), and calls
//! [`Renderer::resize`] when the canvas (window) changes size.

use stark_engine::command::Tool;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::platform::Canvas;
use stark_engine::ViewTransform;
use stark_engine::command::ViewCommand;
use stark_engine::command::{InputCommand, InputSample};
use stark_engine::{Engine, EnvironmentId, GpuContext, ObservableState};
use stark_model::AssetNeed;
use stark_model::ColorSpaceId;
use stark_model::SubstrateId;
use stark_model::geom::Extent2;

pub const CANVAS_ID: &str = "stark-canvas";

/// How many painted frames may still be executing on the GPU before
/// [`request_paint`](crate::state::request_paint) skips a frame instead of
/// submitting another.
///
/// This is the back-pressure the surface cannot give on the WebGPU backend:
/// `present` is a no-op there, `PresentMode::Fifo` and
/// `desired_maximum_frame_latency` are dead values, and `get_current_texture`
/// hands out a fresh canvas texture every frame without blocking — so nothing
/// stops a paint per rAF from deepening the GPU queue without bound whenever a
/// frame's work exceeds the frame budget. Two is what
/// `desired_maximum_frame_latency` would have asked for: enough depth that the
/// CPU and GPU pipeline instead of alternating, while a GPU more than two
/// frames behind sheds presentation frames rather than queueing them.
/// Skipping is safe because ingestion is decoupled from presentation — samples
/// keep reaching the fitter per event, and the first fold after a skip shows
/// exactly what the skipped folds would have (`Engine::flush_live`).
const MAX_FRAMES_IN_FLIGHT: u32 = 2;

/// Owns the canvas surface and the painting engine.
pub struct Renderer {
    canvas: Canvas,
    /// The two handles a *surface* is made from, kept on this side because they are
    /// this frontend's business and not the engine's ([`stark_engine::GpuContext`]):
    /// the app binds three `<canvas>` elements over one device — the painting
    /// canvas, the navigator's miniature and the brush editor's preview — and each
    /// of them needs the instance to create the surface and the adapter to ask what
    /// that surface can do. Both are cheap `Arc` handles, and wgpu does not require
    /// either to be kept alive; keeping them is how a *second* surface stays
    /// possible after the first is bound.
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    engine: Engine,
    /// The shapes bundled with the app (`crate::builtins`), each imported once
    /// its bytes are fetched (§6.6), keyed by the name that module gives it.
    /// A short list looked up by name a handful of times per frame at most —
    /// a `Vec` beats a map, and keeps gallery order.
    builtins: Vec<(&'static str, stark_model::AssetId)>,
    /// The canvas substrates bundled with the app (`crate::substrates`), each imported
    /// once its height map is fetched (§6.4), keyed by the name that module gives
    /// it.
    ///
    /// The same shape as `builtins` above and for the same reason: a substrate is
    /// named by the hash of its image, so its id is only knowable once the bytes
    /// have arrived, and this is where the answer is remembered. A document may
    /// perfectly well be on a substrate that is in no catalog — one a peer brought —
    /// which is why this is a *display* index and never the source of truth.
    substrates: Vec<(&'static str, SubstrateId)>,
    /// The Navigator panel's canvas and everything that draws into it — `None` until
    /// the panel mounts one ([`Renderer::attach_overview`]).
    overview: Option<Overview>,
    /// The compositing attachments the Layers panel's thumbnails render through
    /// ([`Renderer::export_layer`]).
    ///
    /// Kept rather than allocated per call, unlike [`export`](Self::export)'s, and the
    /// decision is here rather than at the call site for the reason `Offscreen`'s own
    /// doc gives: whether a slot outlives its call is the *caller's* to state, and this
    /// is the caller that knows. A file export happens once and may be enormous; a
    /// thumbnail is 64 px, is rendered once per layer, and is rendered again on the
    /// next commit — so allocating and dropping a pair per row would be the cost of
    /// the feature.
    layer_thumbs: stark_engine::Offscreen,
    /// Painted frames whose GPU work has not yet completed (see
    /// [`MAX_FRAMES_IN_FLIGHT`]). Incremented per [`paint`](Self::paint),
    /// decremented by the `on_submitted_work_done` callback that paint registers
    /// — an atomic behind an `Arc` because the callback must be `Send` and may
    /// not touch a signal.
    frames_in_flight: Arc<AtomicU32>,
}

/// A second WebGPU surface showing the same document: the Navigator panel's canvas,
/// the surface bound to it, and the compositing attachments the miniature renders
/// through (`panels::navigator`).
///
/// The miniature is a *rendered surface*, not an image the UI carries — the same
/// bargain the painting canvas makes (§11). It began as an `export`: render
/// to an offscreen texture, copy the pixels back to the CPU, hand them to a 2D canvas
/// through `ImageData`. Every part of that after "render" existed only because the
/// miniature had nowhere of its own to draw, and giving it a surface deleted all of
/// it — the GPU→CPU copy and its frame of latency, the pixel buffer held in a signal,
/// and the imperative repaint that had to be re-run whenever the element remounted.
///
/// One document, two surfaces, on one device: exactly what the brush editor's preview
/// canvas already does ([`Renderer::shared`]), except that this one shares the
/// *engine* too, so it is a second view of the real painting rather than a second
/// painting.
struct Overview {
    /// Kept so the drawing buffer can be resized with the surface: the miniature's
    /// pixel size follows the piece's aspect, not the window's.
    canvas: Canvas,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    /// Pass A's attachments, kept between refreshes — see [`stark_engine::Offscreen`].
    /// This render repeats for as long as the panel is open, so it is the one that has
    /// to reuse them; step 2 of this design is what makes a refresh allocate nothing.
    targets: stark_engine::Offscreen,
}

/// A collaborator, as the chrome draws them (§17.4).
///
/// Deliberately not the engine's [`Peer`](stark_engine::Peer): that carries the
/// in-flight gesture, which is a whole stroke path and is the *canvas's* business,
/// not the DOM's. What the chrome needs is who is here, where they are, and where
/// they are working.
///
/// Kept out of [`ObservableState`] for a related
/// reason: `obs` drives the entire component tree and is refreshed after every
/// command, while this changes thirty times a second whenever anybody moves.
#[derive(Clone, PartialEq, Debug)]
pub struct PeerInfo {
    pub actor: stark_model::document::ActorId,
    pub name: String,
    pub color: [f32; 3],
    pub active_layer: stark_model::document::LayerId,
    pub cursor: Option<stark_model::Vec2>,
}

impl PeerInfo {
    /// The peer's color as a CSS `rgb(...)`, for chips and cursors.
    pub fn css_color(&self) -> String {
        let [r, g, b] = self
            .color
            .map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8);
        format!("rgb({r},{g},{b})")
    }

    /// A one- or two-character badge: the name's initials, so a chip reads as a
    /// person rather than as a colored dot.
    pub fn initials(&self) -> String {
        self.name
            .split_whitespace()
            .filter_map(|w| w.chars().next())
            .take(2)
            .collect::<String>()
            .to_uppercase()
    }
}

impl Renderer {
    /// Send a command to the engine — the **only** way to move engine state through
    /// a `Renderer`, deliberately.
    ///
    /// **No named `set_*` wrappers sit beside it.** A one-line
    /// `engine.process(…)` per setting is a second spelling of a command, and the
    /// second spelling is the one that skips
    /// [`state::dispatch`](crate::state::dispatch): a panel reaching one through the
    /// renderer signal moves engine state without refreshing the observable projection
    /// the chrome reads back, so its own control re-renders showing the *previous*
    /// value and stays wrong until some unrelated command happens to refresh `obs`
    /// (§4, §7). Core declines to expose such a setter for the canvas surface for the
    /// same reason (`Engine::apply_document_substrate`).
    ///
    /// Frontend code holding an `AppState` calls `state::dispatch`, which is this
    /// plus that refresh, the repaint and the outbox flush. This is for the callers
    /// that own a `Renderer` outright and have no chrome to keep in step: app
    /// startup, and the brush editor's private preview engine.
    pub fn process(&mut self, command: impl Into<InputCommand>) {
        self.engine.process(command);
    }

    /// Replay a full stroke as one commit — a single render, no per-sample
    /// live-preview refresh (see `Engine::replay_stroke`).
    pub fn replay_stroke(&mut self, tool: Tool, samples: &[InputSample]) {
        self.engine.replay_stroke(tool, samples);
    }

    /// Replay a full stroke with a caller-chosen jitter seed, so repeated
    /// replays of the same samples keep the same color dynamics and dither
    /// (see `Engine::replay_stroke_seeded`).
    pub fn replay_stroke_seeded(
        &mut self,
        tool: Tool,
        samples: &[InputSample],
        seed: u64,
        rope: f32,
    ) {
        self.engine.replay_stroke_seeded(tool, samples, seed, rope);
    }

    pub fn observe(&self) -> ObservableState {
        self.engine.observe()
    }

    pub fn view(&self) -> ViewTransform {
        self.engine.view()
    }

    /// The in-flight tow, for the string overlay (§6.11).
    pub fn tow_string(&self) -> Option<stark_engine::TowString> {
        self.engine.tow_string()
    }

    /// Whether the engine holds a hover mark (§18.1.10) — peeked before taking
    /// one down, so the clear costs a command and a repaint only when there is
    /// one to remove (`input::clear_hover_mark`).
    pub fn hover_held(&self) -> bool {
        self.engine.hover_held()
    }

    /// What the stroke in flight has snapped to (§6.9), for the tour's count of
    /// shape-assisted strokes (§24.2). Only answers before the gesture ends.
    pub fn assisted(&self) -> Option<stark_engine::Assisted> {
        self.engine.assisted()
    }

    /// The document's current color space (§6.7).
    pub fn color_space(&self) -> ColorSpaceId {
        self.engine.color_space()
    }

    /// Start a fresh document in `color_space`, on `surface` (§6.7).
    /// The color space cannot be changed any other way — see
    /// [`Engine::new_document`].
    pub fn new_document(&mut self, color_space: ColorSpaceId, surface: SubstrateId) {
        if let Err(e) = self.engine.new_document(color_space, surface) {
            // Unreachable from the picker, which only offers spaces this build
            // carries — but a `Renderer` is also driven from tests and scripts, and
            // the engine leaves the open document alone when it refuses, so saying so
            // is better than looking like the New silently did nothing.
            tracing::error!(?color_space, "cannot open a document in this space: {e}");
        }
    }

    /// The document's current canvas surface (§6.4).
    pub fn substrate(&self) -> SubstrateId {
        self.engine.substrate()
    }

    /// Import a canvas substrate from a fetched height map, returning the id that
    /// names it (§6.4). The id comes *out of* the bytes, so it can only be had
    /// once they have arrived.
    pub fn import_substrate(&mut self, png_bytes: &[u8]) -> Option<SubstrateId> {
        match self.engine.import_substrate(png_bytes) {
            Ok(id) => Some(id),
            Err(e) => {
                tracing::warn!("canvas substrate failed to import: {e}");
                None
            }
        }
    }

    /// Take in a substrate that arrives already named — from a peer — checking the
    /// bytes against the id that asked for them (§12.4).
    pub fn accept_substrate(&mut self, id: SubstrateId, png_bytes: &[u8]) {
        if let Err(e) = self.engine.accept_substrate(id, png_bytes) {
            tracing::warn!("remote canvas substrate rejected: {e}");
        }
    }

    /// The canonical height map of a loaded substrate — for seeding a live session so
    /// peers can fetch it by hash.
    pub fn substrate_bytes(&self, id: SubstrateId) -> Option<Vec<u8>> {
        self.engine.substrate_bytes(id)
    }

    /// Import a bundled substrate from its fetched bytes, caching its id under the
    /// name `crate::substrates` gives it. Idempotent: a repeat import is free
    /// (content-addressed) and simply refreshes the entry.
    pub fn load_substrate(&mut self, name: &'static str, png_bytes: &[u8]) -> Option<SubstrateId> {
        let id = self.import_substrate(png_bytes)?;
        self.substrates.retain(|(n, _)| *n != name);
        self.substrates.push((name, id));
        Some(id)
    }

    /// The substrate `name` resolved to, or `None` while its height map has yet to be
    /// fetched and imported.
    pub fn builtin_substrate(&self, name: &str) -> Option<SubstrateId> {
        self.substrates
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, id)| *id)
    }

    /// Whether an environment's bytes are loaded (the procedural `Neutral` always is).
    pub fn environment_loaded(&self, id: EnvironmentId) -> bool {
        self.engine.environment_loaded(id)
    }

    /// Register frontend-fetched HDR bytes for a lighting environment (§6.3) — or
    /// `Err` if they are not an HDR this build can read, in which case nothing is
    /// stored and the canvas keeps the light it has.
    pub fn register_environment(
        &mut self,
        id: EnvironmentId,
        hdr_bytes: Vec<u8>,
    ) -> stark_engine::Result<()> {
        self.engine.register_environment(id, hdr_bytes)
    }

    /// The surface's current size in CSS pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Where the history playhead stands and how far it can travel, in actions —
    /// `None` when the history is not this client's alone to walk
    /// (§18.2.4).
    pub fn scrub_range(&self) -> Option<(usize, usize)> {
        self.engine.scrub_range()
    }

    /// A caption per action across the whole scrub range, oldest first.
    pub fn scrub_labels(&self) -> Vec<&'static str> {
        self.engine.scrub_labels()
    }

    /// Serialize the document — the action log, not the pixels (§8) — leaving out
    /// content this build ships with, which it resolves again on open (§8's
    /// version 6). There is no fat-save wrapper because the frontend has no use
    /// for one: it is the app that owns the assets it is declining to carry.
    pub fn save_bytes_resolvable(
        &self,
        resolvable: &[stark_model::AssetId],
    ) -> stark_engine::Result<Vec<u8>> {
        self.engine.save_bytes_resolvable(resolvable)
    }

    /// What `file` names that neither it carries nor this engine already holds —
    /// settle it before [`Renderer::load_document`].
    pub fn unresolved_content(&self, file: &stark_model::DocumentFile) -> Vec<AssetNeed> {
        self.engine.unresolved_content(file)
    }

    /// Replace the document by replaying a loaded log (§8).
    ///
    /// Fails, leaving the open document untouched, if anything the log names is still
    /// unresolved — so the settle above is not advisory.
    pub fn load_document(&mut self, file: &stark_model::DocumentFile) -> stark_engine::Result<()> {
        self.engine.load_document(file)
    }

    /// What exporting would produce, without producing it (§15.6).
    pub fn export_plan(
        &self,
        frame: Option<stark_model::document::LayerId>,
        scale: stark_engine::ExportScale,
    ) -> stark_engine::Result<stark_engine::ExportPlan> {
        self.engine.export_plan(frame, scale)
    }

    /// Render a frame and return a future for its readback (§15.6).
    ///
    /// The future does **not** borrow the renderer, which is the whole point: the
    /// caller can drop its write guard before awaiting, so the UI is free to
    /// re-render (and read the renderer) while the GPU→CPU copy is in flight.
    ///
    /// A one-shot: the attachments are allocated for this render and dropped with it,
    /// which is what keeps a 4× export of a large frame from parking its
    /// several-hundred-megabyte pair for the rest of the session.
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

    /// Render **one layer alone** through `plan`'s view, for a Layers panel row
    /// (§14.6). Returns a future for the readback, on [`export`](Self::export)'s
    /// borrow bargain — the caller drops its write guard before awaiting.
    ///
    /// The layer's blend mode, clip and opacity are dropped by the isolate this
    /// renders through (`Engine::export_view`), so a row shows the paint that is
    /// there rather than the part of it the document lets through. Cut out rather
    /// than over the substrate, so a row says where the layer *has* paint.
    ///
    /// `plan` is the caller's, and deliberately: it is the same plan the navigator
    /// frames its miniature with, so an overview and a row cannot come to disagree
    /// about where the piece is.
    pub fn export_layer(
        &mut self,
        layer: stark_model::document::LayerId,
        plan: &stark_engine::ExportPlan,
    ) -> stark_engine::Result<
        impl std::future::Future<Output = stark_engine::Result<stark_engine::RgbaImage>> + use<>,
    > {
        self.engine.export_view(
            &mut self.layer_thumbs,
            plan.view(),
            Some(layer),
            stark_engine::Background::Transparent,
            stark_engine::Rendered::Committed,
        )
    }

    /// Bind the Navigator panel's `<canvas>` as a second surface onto this engine's
    /// device, ready for [`paint_overview`](Self::paint_overview).
    ///
    /// Called from the panel's `onmounted`, and again on every remount: a closed panel
    /// takes its element with it, and the element is what a surface is bound to, so
    /// the old one is dropped here rather than reused. Nothing is measured — the
    /// miniature's size comes from the piece's proportions, not from layout, so the
    /// drawing buffer is sized on the first paint instead.
    ///
    /// Configured to the engine's own target format, not to a format picked from this
    /// surface's capabilities: the engine's pipelines are built for one format, and a
    /// second surface that chose differently would fail validation rather than merely
    /// look wrong. The format the main canvas settled on is available here (both
    /// surfaces are canvases on the same adapter, so it is in this one's caps too).
    pub fn attach_overview(&mut self, canvas: Canvas) {
        let surface = match self.instance.create_surface(canvas.surface_target()) {
            Ok(surface) => surface,
            Err(e) => {
                tracing::warn!("navigator surface unavailable: {e}");
                return;
            }
        };
        let caps = surface.get_capabilities(&self.adapter);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.engine.target_format(),
            // Zero says "not configured yet", and cannot collide with a real plan
            // size (a plan's edges are floored at 1), so the first paint always
            // configures before it asks for a texture.
            width: 0,
            height: 0,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::default(),
        };
        self.overview = Some(Overview {
            canvas,
            surface,
            config,
            targets: stark_engine::Offscreen::default(),
        });
    }

    /// Draw the miniature `plan` describes straight into the Navigator's surface and
    /// present it. `false` if no canvas is attached (the panel is closed).
    ///
    /// The committed document, over the substrate: an overview is a picture of the
    /// piece as it stands, and it is refreshed per commit, so following the stroke in
    /// hand would mean re-rendering at pointer rate to show what the canvas beside it
    /// is already showing full size.
    ///
    /// Synchronous, which is the whole point of the surface: there is no readback to
    /// await, so a refresh is one render and a present.
    pub fn paint_overview(&mut self, plan: &stark_engine::ExportPlan) -> bool {
        use wgpu::CurrentSurfaceTexture::{Suboptimal, Success};
        let Some(ov) = self.overview.as_mut() else {
            return false;
        };
        let size = plan.size;
        if (ov.config.width, ov.config.height) != (size.width, size.height) {
            ov.canvas.set_buffer_size(size.width, size.height);
            ov.config.width = size.width;
            ov.config.height = size.height;
            ov.surface.configure(&self.engine.gpu().device, &ov.config);
        }
        let frame = match ov.surface.get_current_texture() {
            Success(frame) | Suboptimal(frame) => frame,
            // Timeout/Outdated/Lost: skip it. The next committed edit repaints, and a
            // miniature one revision stale is not worth a retry loop.
            _ => return false,
        };
        let target = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.engine.render_into(
            &mut ov.targets,
            &target,
            plan.view(),
            stark_engine::Background::Substrate,
            stark_engine::Rendered::Committed,
        );
        self.engine.gpu().queue.present(frame);
        true
    }

    /// Sample the canvas color at `at` — the eyedropper (§18.0.2).
    ///
    /// The same borrow bargain as [`Renderer::export`], and it matters more here:
    /// the sample is taken mid-gesture, so the caller has to be able to drop its
    /// write guard before awaiting or the UI's own re-render will panic on a
    /// renderer it still holds borrowed.
    pub fn pick_color(
        &mut self,
        at: stark_model::Vec2,
        options: stark_engine::PickOptions,
    ) -> impl std::future::Future<Output = Option<[f32; 3]>> + use<> {
        self.engine.pick_color(at, options)
    }

    /// Sample a gradient along a traced path — the gradient capture (§22.2).
    /// The same borrow bargain as [`Renderer::pick_color`].
    pub fn pick_gradient(
        &mut self,
        path: &[stark_model::Vec2],
        options: stark_engine::PickOptions,
    ) -> impl std::future::Future<Output = Option<stark_model::Gradient>> + use<> {
        self.engine.pick_gradient(path, options)
    }

    /// Which layer's paint the canvas shows at `at` — the layer carry's hit test
    /// (§16.11). The same borrow bargain as [`Renderer::pick_color`], and it
    /// matters here for that reason exactly: this one opens a drag, so the
    /// gesture goes on driving the renderer while the readback is in flight.
    pub fn pick_layer(
        &mut self,
        at: stark_model::Vec2,
    ) -> impl std::future::Future<Output = Option<stark_model::document::LayerId>> + use<> {
        self.engine.pick_layer(at)
    }

    /// A bundled shape's content id, once its bytes have been imported.
    pub fn builtin(&self, name: &str) -> Option<stark_model::AssetId> {
        self.builtins
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, id)| *id)
    }

    // --- collaboration (§12) — thin engine delegates for the
    // session glue in `collab.rs`. ---

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

    /// Leave a shared session (keep the canvas and its history, stop broadcasting).
    pub fn end_collaboration(&mut self) {
        self.engine.end_collaboration();
    }

    /// Snapshot the document (full shared log + referenced assets).
    pub fn document_file(&self) -> stark_model::DocumentFile {
        self.engine.document_file()
    }

    /// Integrate one remote action; returns whether it was new.
    pub fn merge_remote(&mut self, action: stark_model::document::Action) -> bool {
        self.engine.merge_remote(action)
    }

    /// Drain locally-committed actions awaiting broadcast.
    pub fn take_outbox(&mut self) -> Vec<stark_model::document::Action> {
        self.engine.take_outbox()
    }

    /// Whether [`take_presence`](Self::take_presence) would do anything — a `&self`
    /// test, so an idle tick of the presence pump takes no mutable borrow.
    pub fn presence_due(&self, now: f64) -> bool {
        self.engine.presence_due(now)
    }

    /// A counter that changes whenever the peer roster does.
    pub fn peers_revision(&self) -> u64 {
        self.engine.peers_revision()
    }

    /// Drain this client's presence latch, and expire peers gone quiet
    /// (§17.5). The frame is `None` when there is nothing new to
    /// say; `repaint` reports that the expiry took paint off the canvas.
    pub fn take_presence(&mut self, now: f64) -> stark_engine::PresenceTick {
        self.engine.take_presence(now)
    }

    /// The farewell frame, so peers drop this client at once on leave.
    pub fn leaving_presence(&mut self) -> stark_model::PeerFrame {
        self.engine.leaving_presence()
    }

    /// Integrate a peer's presence; returns whether the canvas needs repainting.
    ///
    /// Narrower than "anything changed": a moved cursor or a switched layer is
    /// chrome, picked up from [`peers_revision`](Self::peers_revision) instead.
    /// `now` dates the frame for expiry — the caller's clock, because the engine's
    /// own only advances when the presence pump has something to drain.
    pub fn merge_presence(
        &mut self,
        actor: stark_model::document::ActorId,
        frame: stark_model::PeerFrame,
        now: f64,
    ) -> bool {
        self.engine.merge_presence(actor, frame, now)
    }

    /// Everyone else in the session, for the peer chrome (§17.4).
    pub fn peers(&self) -> Vec<PeerInfo> {
        self.engine
            .peers()
            .map(|p| PeerInfo {
                actor: p.actor,
                name: p.name.clone(),
                color: p.color,
                active_layer: p.active_layer,
                cursor: p.cursor,
            })
            .collect()
    }

    /// Import a remote peer's brush image so its strokes render faithfully.
    pub fn import_brush(&self, png_bytes: &[u8]) {
        if let Err(e) = self.engine.import_brush(png_bytes) {
            tracing::warn!("remote brush import failed: {e}");
        }
    }

    /// Import a user's brush-shape image, returning its content id — the
    /// error surfaces to the import UI rather than a log line.
    pub fn import_brush_id(&self, png_bytes: &[u8]) -> Result<stark_model::AssetId, String> {
        self.engine
            .import_brush(png_bytes)
            .map_err(|e| e.to_string())
    }

    /// Import a picture to place (§23), returning the content id the action names it
    /// by — [`import_brush_id`](Self::import_brush_id)'s sibling, error and all: the
    /// caller has an import to report failure on, where a remote arrival has only a
    /// log line.
    pub fn import_picture_id(&self, png_bytes: &[u8]) -> Result<stark_model::AssetId, String> {
        self.engine
            .import_picture(png_bytes)
            .map_err(|e| e.to_string())
    }

    /// Install a picture that arrives already named — off a peer, or out of a
    /// document's own bundle. Refused if the bytes are not the picture the id asked
    /// for, which is where a wrong photograph would otherwise be placed silently.
    pub fn accept_picture(&self, id: stark_model::AssetId, png_bytes: &[u8]) {
        if let Err(e) = self.engine.accept_picture(id, png_bytes) {
            tracing::warn!("remote picture install failed: {e}");
        }
    }

    /// The canonical PNG bytes of one imported brush asset, if loaded.
    pub fn asset_bytes(&self, id: stark_model::AssetId) -> Option<Vec<u8>> {
        self.engine.asset_bytes(id)
    }

    /// Every imported brush asset, to seed a session's asset mirror.
    pub fn all_asset_bytes(&self) -> Vec<(stark_model::AssetId, Vec<u8>)> {
        self.engine.all_asset_bytes()
    }

    /// Import a bundled shape from its fetched bytes, caching its id under the
    /// name `crate::builtins` gives it. Idempotent: a repeat import is free
    /// (content-addressed) and simply refreshes the entry.
    pub fn load_builtin(&mut self, name: &'static str, png_bytes: &[u8]) {
        match self.engine.import_brush(png_bytes) {
            Ok(id) => {
                self.builtins.retain(|(n, _)| *n != name);
                self.builtins.push((name, id));
            }
            Err(e) => tracing::warn!("built-in shape “{name}” failed to import: {e}"),
        }
    }

    /// Match the surface + engine viewport to a new canvas size (CSS pixels).
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 || (width == self.config.width && height == self.config.height)
        {
            return;
        }
        self.canvas.set_buffer_size(width, height);
        self.config.width = width;
        self.config.height = height;
        self.surface
            .configure(&self.engine.gpu().device, &self.config);
        self.engine
            .process(ViewCommand::Resize(Extent2::new(width, height)));
    }

    /// Re-measure the canvas element and match the surface to it. A no-op when it
    /// already agrees.
    ///
    /// The size [`finish_init`] seeds from is a *guess*. It is read one animation
    /// frame in, which is not the same thing as the stylesheet having applied: until
    /// it does, `.paint-canvas` is not in force and the element measures the canvas's
    /// intrinsic 300×150 rather than the window. Nothing corrects that on its own,
    /// because the only correction is the DOM resize observer, and it reports through
    /// [`crate::state::resize`], which can act only once the renderer signal is
    /// published — while everything between `init` and that publish is a *network
    /// fetch* (shape assets, the substrate's height map, the environment HDR). The
    /// corrected size therefore lands squarely inside the window where it is dropped,
    /// and the viewport keeps a size the canvas has not had since the first frame:
    /// the view's `half()` is off by the difference, so every stroke lands away from
    /// the pointer until something else changes the layout.
    ///
    /// So the seed is treated as provisional and this re-reads the element at the
    /// first moment a resize could no longer be missed — the statement immediately
    /// before the renderer is published, with no `await` between the two.
    pub fn sync_to_canvas(&mut self) {
        let (width, height) = self.canvas.laid_out_size();
        self.resize(width, height);
    }

    /// Whether the device is still usable — the `&self` request form of what
    /// [`ObservableState::gpu_failure`] projects (§5).
    ///
    /// For the one caller that has no projection to hand: the paint loop
    /// (`state::schedule_paint`). Everything else asks the projection, which is
    /// what the chrome mounts its report on — see [`crate::failure`].
    pub fn gpu_healthy(&self) -> bool {
        self.engine.gpu().health().is_ok()
    }

    /// Whether the GPU still owes the work of [`MAX_FRAMES_IN_FLIGHT`] painted
    /// frames — the signal for [`request_paint`](crate::state::request_paint) to
    /// skip a frame rather than deepen the queue. Because submissions on one
    /// queue complete in order, a paint's completion also vouches for every
    /// submission before it (commit renders, fills), so queue depth from
    /// non-paint work is counted too, one frame later.
    pub fn gpu_behind(&self) -> bool {
        self.frames_in_flight.load(Ordering::Relaxed) >= MAX_FRAMES_IN_FLIGHT
    }

    /// Render the current canvas straight into the surface texture and present.
    pub fn paint(&mut self) {
        use wgpu::CurrentSurfaceTexture::{Suboptimal, Success};
        // Its own row because it is the one part of a frame that is not Stark's
        // work: acquiring a surface texture is where a browser compositor makes the
        // page wait, and folded into `frame` that wait would read as time the engine
        // spent. On WebGPU it should be free — `get_current_texture` never blocks
        // there — so a row that grows is a finding rather than a cost.
        let frame = {
            stark_engine::timing::span!("frame.acquire");
            match self.surface.get_current_texture() {
                Success(frame) | Suboptimal(frame) => frame,
                // Timeout/Outdated/Lost/etc.: skip; the next command repaints.
                _ => return,
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.engine.render(&view);
        // Count this frame against the in-flight budget until the GPU finishes
        // it. Registered after the render's submit, so the callback fires once
        // everything this paint queued has executed. The WebGPU spec resolves
        // the underlying promise even on device loss, so the count cannot wedge.
        let in_flight = Arc::clone(&self.frames_in_flight);
        in_flight.fetch_add(1, Ordering::Relaxed);
        self.engine.gpu().queue.on_submitted_work_done(move || {
            in_flight.fetch_sub(1, Ordering::Relaxed);
        });
        // A no-op on the web, where the canvas is presented by the page — measured
        // anyway, and cheaply, because "present is free here" is a claim about wgpu's
        // WebGPU backend that a version bump could quietly stop being true.
        stark_engine::timing::span!("frame.present");
        self.engine.gpu().queue.present(frame);
    }
}

/// Why the app could not start (§5, `crate::failure`).
///
/// **A different fact from `ObservableState::gpu_failure`**, and the difference is
/// what earns it a type of its own: that one is a device that *died*, with a
/// document behind it that outlives it and is worth saving. This is a device that
/// never arrived — there is no engine, no document and nothing to offer. What the
/// two share is that the canvas will never take a mark, which is why both reports
/// are the same surface.
///
/// It was `expect` on all three arms until the review that named it. That is
/// defensible for `create_surface`, which fails only if the element is not a
/// canvas, and indefensible for the other two: a browser without WebGPU is the
/// single most likely way this app fails for a first-time visitor — Safari before
/// 26, Firefox without the flag, a blocklisted driver, any headless browser — and
/// a panic inside the startup task killed the task and nothing else, leaving the
/// chrome up over a blank canvas with the explanation in the console.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StartupFailure {
    /// The page's `<canvas>` would not give a WebGPU surface.
    Surface(String),
    /// No adapter answered — what a browser with no WebGPU at all looks like
    /// from here, and the common case by a wide margin.
    Adapter,
    /// An adapter answered but would not give a device at the limits the engine
    /// needs ([`GpuContext::minimum_required_limits`]).
    Device(String),
}

impl std::fmt::Display for StartupFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StartupFailure::Surface(why) => write!(f, "no WebGPU surface: {why}"),
            StartupFailure::Adapter => write!(f, "no WebGPU adapter"),
            StartupFailure::Device(why) => write!(f, "no WebGPU device: {why}"),
        }
    }
}

/// Asynchronously create the WebGPU device, configure the surface to the
/// canvas's current size, and build the engine (§7).
///
/// Fallible on all three of the browser's answers rather than panicking on them
/// — see [`StartupFailure`] for why that is not merely tidiness.
pub async fn init(canvas: Canvas) -> Result<Renderer, StartupFailure> {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::BROWSER_WEBGPU;
    let instance = wgpu::Instance::new(desc);

    let surface: wgpu::Surface<'static> = instance
        .create_surface(canvas.surface_target())
        .map_err(|e| StartupFailure::Surface(e.to_string()))?;

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..Default::default()
        })
        .await
        .map_err(|_| StartupFailure::Adapter)?;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("stark web device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default()
                .or_better_values_from(&GpuContext::minimum_required_limits()),
            experimental_features: wgpu::ExperimentalFeatures::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
        .map_err(|e| StartupFailure::Device(e.to_string()))?;

    let gpu = GpuContext::from_parts(device, queue);
    Ok(finish_init(canvas, instance, adapter, surface, gpu).await)
}

impl Renderer {
    /// Build a second [`Renderer`] on this one's device: a new surface bound to
    /// `canvas` plus an engine of its own that **shares** this engine's expensive
    /// state — every compiled pipeline, the imported brush shapes, and the decoded
    /// substrate and environment caches (`Engine::new_sharing`). The preview document
    /// stays fully isolated from the real one; it opens on this document's substrate,
    /// under this canvas's lighting and media parameters, so a stroke on it reads
    /// exactly as it would here — with nothing re-fetched and nothing re-decoded.
    ///
    /// Synchronous, and callable only once this renderer exists — which is also the
    /// only time it makes sense: a preview is a preview *of* this canvas. The caller
    /// should await a layout frame ([`platform::next_frame`](crate::platform::next_frame))
    /// before this, so the canvas
    /// measures as laid out rather than at its 300×150 intrinsic size; the measure
    /// here is still only a seed, corrected by [`Renderer::sync_to_canvas`] before
    /// anything is placed against it.
    ///
    /// Configured to this engine's own target format rather than a format picked
    /// from the new surface's capabilities, exactly as
    /// [`attach_overview`](Self::attach_overview) is and for the same reason: the
    /// shared pipelines are built for one format, and a second surface that chose
    /// differently would fail validation rather than merely look wrong.
    pub fn shared(&self, canvas: Canvas) -> Renderer {
        let (width, height) = canvas.laid_out_size();
        canvas.set_buffer_size(width, height);
        let surface: wgpu::Surface<'static> = self
            .instance
            .create_surface(canvas.surface_target())
            .expect("create preview canvas surface");
        let caps = surface.get_capabilities(&self.adapter);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.engine.target_format(),
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::default(),
        };
        surface.configure(&self.engine.gpu().device, &config);
        let engine = Engine::new_sharing(&self.engine, Extent2::new(width, height));
        Renderer {
            canvas,
            instance: self.instance.clone(),
            adapter: self.adapter.clone(),
            surface,
            config,
            engine,
            // The name → id indexes ride along: the ids are content-addressed and
            // the assets behind them are shared, so the donor's answers are this
            // engine's answers.
            builtins: self.builtins.clone(),
            substrates: self.substrates.clone(),
            overview: None,
            layer_thumbs: stark_engine::Offscreen::default(),
            frames_in_flight: Arc::new(AtomicU32::new(0)),
        }
    }

    /// The expensive half of this renderer's engine, on its own
    /// (`stark_engine::EngineShared`) — the device, the compiled pipelines, the brush
    /// assets and the decoded substrates.
    ///
    /// **It outlives this renderer's borrow, which is the point.** A consumer that
    /// only wants to *render* something — a preset thumbnail — previously had to hold
    /// the whole live `Renderer` to reach these, so it could not be built until one
    /// existed and had to be created lazily inside the loop that used it. This clones
    /// for a handful of refcount bumps and can simply be kept.
    pub fn engine_shared(&self) -> stark_engine::EngineShared {
        self.engine.shared()
    }
}

/// Tail of [`init`]: size the drawing buffer, pick the surface format, configure,
/// and build the engine. (A *second* renderer never comes through here — it is built
/// synchronously by [`Renderer::shared`], on the first engine's format and state.)
async fn finish_init(
    canvas: Canvas,
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    surface: wgpu::Surface<'static>,
    gpu: GpuContext,
) -> Renderer {
    // Size the drawing buffer to the canvas's laid-out size (CSS pixels). We
    // measure the *element*, not the window, so an embedded/sub-window canvas
    // works too, and we do it here — after the async device setup and a layout
    // frame — rather than up front, where the unstyled 300×150 intrinsic size is
    // all there is to read.
    //
    // A frame is not a *guarantee* that the stylesheet (linked via
    // `document::Stylesheet`) has applied, though, so this is a seed and not the
    // answer: the caller re-reads the element with `Renderer::sync_to_canvas` just
    // before publishing the renderer, which is where the guarantee actually is.
    // Everything after that is handled by `onresize`.
    crate::platform::next_frame().await;
    let (width, height) = canvas.laid_out_size();
    canvas.set_buffer_size(width, height);

    // Pick a non-sRGB format: the media pass already encodes display sRGB, so an
    // sRGB surface would double-encode (§6.5).
    let caps = surface.get_capabilities(&adapter);
    let format = caps
        .formats
        .iter()
        .copied()
        .find(|f| !f.is_srgb())
        .unwrap_or(caps.formats[0]);

    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width,
        height,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
        color_space: wgpu::SurfaceColorSpace::default(),
    };
    surface.configure(&gpu.device, &config);

    let engine = Engine::new(gpu, format, Extent2::new(width, height));
    Renderer {
        canvas,
        instance,
        adapter,
        surface,
        config,
        engine,
        builtins: Vec::new(),
        substrates: Vec::new(),
        overview: None,
        layer_thumbs: stark_engine::Offscreen::default(),
        frames_in_flight: Arc::new(AtomicU32::new(0)),
    }
}
