//! The engine: owns the GPU, session, and timeline; turns commands into state
//! and renders the canvas (DESIGN.md §7).
//!
//! For the MVP this exposes a synchronous [`Engine::process`]. The asynchronous
//! actor loop and reactive `ObservableState` channel (DESIGN.md §7) wrap this
//! same core in a later step.

use std::sync::Arc;

use crate::Result;
use crate::assets::{AssetId, AssetStore};
use crate::colorspace::{ColorSpace, ColorSpaceId};
use crate::command::{DocCommand, GestureCommand, InputCommand, ViewCommand};
use crate::document::{
    Action, ActionId, ActionKind, ActorId, ApplyCtx, BrushParams, BrushShape, CanvasBounds,
    DocState, Layer, LayerId, LinearTimeline, ReplicatedTimeline, SelectionMode, StrokeRecord,
    Timeline, Tool, effective_actions,
};
use crate::geom::{Extent2, ViewTransform};
use crate::gpu::tile::MASK_FORMAT;
use crate::gpu::{
    Compositor, Environment, EnvironmentId, GpuContext, Registry, SelectionRenderer,
    StrokeRenderer, StrokeSpans, Surface, SurfaceId, TilePairHandle, TilePool,
};
use crate::image::RgbaImage;
use crate::io::DocumentFile;

/// The starting layer present in every new document.
const ROOT_LAYER: LayerId = LayerId(0);

/// A layer's presentation properties, for the UI's layer panel (DESIGN.md §11).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerInfo {
    pub id: LayerId,
    pub blend: crate::document::BlendMode,
    pub opacity: f32,
    pub visible: bool,
}

/// A cheap, UI-facing projection of engine state (DESIGN.md §7). Published to
/// the frontend so it can render chrome reactively without touching pixels.
#[derive(Clone, Debug)]
pub struct ObservableState {
    pub can_undo: bool,
    pub can_redo: bool,
    pub is_stroking: bool,
    pub tool: Tool,
    pub brush: BrushParams,
    pub view: ViewTransform,
    pub bounds: CanvasBounds,
    pub active_layer: LayerId,
    /// Layers bottom-to-top.
    pub layers: Vec<LayerInfo>,
    /// Whether a selection is masking the canvas (DESIGN.md §6.8) — drives the
    /// "Deselect"/"Invert" affordances and the selection indicator.
    pub has_selection: bool,
    /// How the next selection gesture will combine with the current selection.
    pub selection_mode: SelectionMode,
    /// Edge softness (canvas px) the next selection gesture will apply.
    pub selection_feather: f32,

    // --- view settings (per-client, never historized) ---------------------
    //
    // Projected here for the same reason as `tool` and `brush`: a frontend that
    // has to read these back off the engine ends up keeping its own copy, and a
    // copy seeded from `Default` rather than from the engine goes stale the
    // moment anything else changes them (DESIGN.md §4).
    /// Media/lighting parameters of the painterly pass (DESIGN.md §6.3).
    pub media: crate::gpu::MediaParams,
    /// The HDR lighting environment in use (DESIGN.md §6.3).
    pub environment: EnvironmentId,

    // --- document properties fixed at creation ----------------------------
    /// The document's colour space. Immutable for the document's life — changing
    /// it means starting a new document ([`Engine::new_with_color_space`]).
    pub color_space: ColorSpaceId,
    /// The physical canvas surface (DESIGN.md §6.4).
    pub surface: SurfaceId,
}

/// Colour the live tail is drawn in under the `debug-unfrozen` feature.
/// Full-opacity magenta: it has to read against paint of any hue, and against the
/// stroke's own colour in particular.
#[cfg(feature = "debug-unfrozen")]
const DEBUG_UNFROZEN_COLOR: [f32; 4] = [1.0, 0.0, 1.0, 1.0];

/// The part of the in-flight stroke that has stopped changing, already composited
/// onto the committed document.
///
/// A live stroke is re-rendered on every pointer move, and rendering it costs
/// (segments × tiles it covers) — both of which grow with its length, so a long
/// stroke gets quadratically expensive to keep drawing. But the fitter freezes
/// control points behind the pointer and never revises them
/// ([`PathFitter::frozen_spans`](crate::path::PathFitter::frozen_spans)), so the
/// spans they determine are final: render them once, keep the result, and each move
/// only has to draw the short live tail over it. Work per move then follows the tail
/// rather than the stroke.
struct FrozenHead {
    /// How many leading spans `state` already has drawn on it.
    spans: usize,
    /// Arc length at the end of those spans — where the tail's `dist` resumes.
    /// Not recoverable from `spans` alone, and the `drain` falloff and colour
    /// dynamics both read it (see `gpu::stroke::StrokeSpans`).
    dist: f32,
    /// The brush state the tail must resume from, for a stroke that runs the
    /// sequential stamp loop (`lift`/`deposit`/`charge`). `None` for the swept path,
    /// which carries nothing between segments. See
    /// [`ToolState`](crate::gpu::stroke::ToolState).
    tool: Option<crate::gpu::stroke::ToolState>,
    state: DocState,
}

pub struct Engine {
    gpu: GpuContext,
    target_format: wgpu::TextureFormat,
    color_space: Arc<dyn ColorSpace>,
    /// The GPU subsystems an action needs in order to apply itself — the tile
    /// pool, the stroke renderer, the asset store and the selection rasterizer —
    /// held as the `history::Action::Context` (DESIGN.md §5).
    ///
    /// Stored rather than built per call. `history`'s `Context` is an owned
    /// associated type, so there is nothing to hand it a borrow of; this used to be
    /// rebuilt by cloning all four on *every* commit, undo, redo and remote merge —
    /// tens of `Arc` bumps plus a `HashMap` allocation each time, for a value that
    /// only changes when the colour space is rebuilt.
    ///
    /// They live only here: the engine reaches them through `self.apply` too, so
    /// there is one copy rather than the engine's plus the context's.
    /// `selection` is colour-space independent (a mask is one coverage channel
    /// whatever the paint is), so unlike the pool and the stroke renderer it
    /// survives a rebuild.
    apply: ApplyCtx,
    compositor: Compositor,
    /// The physical canvas surface (relief) and the bytes registered for
    /// it. Colour-space-independent, so it survives colour-space rebuilds. Which
    /// surface is in use is a *cache* of `document().surface`, kept in step by
    /// [`Engine::apply_document_surface`] — the document is the source of truth
    /// (DESIGN.md §6.4).
    surface: Registry<SurfaceId>,
    /// The surface the action log starts from, written to `CanvasMeta` and used to
    /// seed the document. Plays the same role as `CanvasMeta::color_space`: it
    /// describes the empty document that the log is replayed onto, and is not
    /// itself a logged change.
    initial_surface: SurfaceId,
    /// The HDR lighting environment (image-based lighting) and its registered
    /// bytes. A *view* setting — not historized, colour-space-independent — so it
    /// survives rebuilds and switching it never touches the document (§6.3).
    environment: Registry<EnvironmentId>,
    timeline: Box<dyn Timeline>,
    session: crate::session::Session,
    /// Live preview of the in-flight stroke, composited in place of the
    /// committed state while painting (DESIGN.md §6.2). `None` when idle.
    preview: Option<DocState>,
    /// The settled head of the in-flight stroke, already rendered (see
    /// [`FrozenHead`]).
    frozen_head: Option<FrozenHead>,
    /// Raw pointer reports of the in-flight stroke, dumped on release under the
    /// `debug-unfrozen` feature so a misfit stroke can be replayed as a test.
    debug_samples: Vec<crate::command::InputSample>,
    actor: ActorId,
    clock: u64,
    next_layer: u64,
    /// Locally-committed actions awaiting broadcast to peers (DESIGN.md §12.4).
    /// Only populated in a shared session (`outbox_enabled`), and drained by the
    /// transport via [`Engine::take_outbox`]; solo mode never accumulates.
    outbox: Vec<Action>,
    outbox_enabled: bool,
}

impl Engine {
    /// Build an engine that presents to `target_format` (a surface format, or a
    /// test target), in the default Oklab color space. Takes wgpu handles per
    /// GOALS §Inputs.
    pub fn new(gpu: GpuContext, target_format: wgpu::TextureFormat, viewport: Extent2) -> Self {
        Self::new_with_color_space(gpu, target_format, viewport, ColorSpaceId::Oklab)
    }

    /// Build an engine in a chosen color space (DESIGN.md §6.7).
    pub fn new_with_color_space(
        gpu: GpuContext,
        target_format: wgpu::TextureFormat,
        viewport: Extent2,
        color_space: ColorSpaceId,
    ) -> Self {
        let color_space = color_space.make();
        // Fresh documents start on the procedural flat surface; image-backed
        // surfaces are registered later by the frontend (DESIGN.md §6.4).
        let surface = Registry::<SurfaceId>::new(&gpu, SurfaceId::default());
        // Lighting starts on the procedural studio environment; image HDRs are
        // registered later by the frontend (DESIGN.md §6.3).
        let _environment_id = EnvironmentId::default();
        let environment = Registry::<EnvironmentId>::new(&gpu, EnvironmentId::default());
        let selection = SelectionRenderer::new(&gpu);
        let (pool, stroke, compositor) = build_gpu(GpuBuild {
            gpu: &gpu,
            target_format,
            viewport,
            cs: &color_space,
            surface: surface.current(),
            environment: environment.current(),
            selection: &selection,
        });
        let assets = AssetStore::new(gpu.clone());

        let initial = DocState::with_layer(ROOT_LAYER);
        let timeline: Box<dyn Timeline> = Box::new(LinearTimeline::new(initial));
        let session = crate::session::Session::new(ViewTransform::identity(viewport), ROOT_LAYER);

        Self {
            gpu,
            target_format,
            color_space,
            apply: ApplyCtx {
                pool,
                stroke,
                assets,
                selection,
            },
            compositor,
            initial_surface: surface.id(),
            surface,
            environment,
            timeline,
            session,
            preview: None,
            frozen_head: None,
            debug_samples: Vec::new(),
            actor: ActorId::SOLO,
            clock: 0,
            next_layer: 1,
            outbox: Vec::new(),
            outbox_enabled: false,
        }
    }

    /// Apply one input command (DESIGN.md §4).
    ///
    /// One-way by construction: nothing comes back. Reads go through
    /// [`Engine::observe`]; anything that must answer is a request (see
    /// [`command`](crate::command)).
    pub fn process(&mut self, command: impl Into<InputCommand>) {
        let command = command.into();
        // The frozen head caches pixels composited onto *a particular document* for
        // *a particular stroke*, and neither of those is visible from the head
        // itself. It is only ever legitimately reused across consecutive gesture
        // moves; anything else may have replaced the document under it - a commit,
        // an undo, a remote merge, a layer switch - or started a different stroke
        // entirely, and reusing it then composites the live tail onto a stale
        // canvas. Dropping it on every other command costs one rebuild and rules
        // out the whole class of staleness rather than enumerating the ways it
        // arises.
        if !matches!(command, InputCommand::Gesture(GestureCommand::To { .. })) {
            self.frozen_head = None;
        }
        match command {
            InputCommand::Gesture(c) => self.process_gesture(c),
            InputCommand::Doc(c) => self.process_doc(c),
            InputCommand::View(c) => self.process_view(c),
        }
    }

    /// The press-drag-release lifecycle. One path for both kinds of tool
    /// (DESIGN.md §6.8): the selection tools build an op where the brush builds a
    /// stroke, and both preview through the same `preview` DocState.
    fn process_gesture(&mut self, command: GestureCommand) {
        match command {
            GestureCommand::Start { tool, sample } => {
                if tool.is_selection() {
                    self.session.start_selection(tool, sample.pos);
                    self.refresh_selection_preview();
                } else {
                    let seed = self.clock;
                    self.session.start_stroke(tool, sample, seed);
                    self.debug_samples.clear();
                    self.debug_samples.push(sample);
                    self.refresh_preview();
                }
            }
            GestureCommand::To { sample } => {
                if self.session.is_selecting() {
                    self.session.selection_to(sample.pos);
                    self.refresh_selection_preview();
                } else {
                    self.session.stroke_to(sample);
                    if cfg!(feature = "debug-unfrozen") {
                        self.debug_samples.push(sample);
                    }
                    self.refresh_preview();
                }
            }
            // The one edge that produces document state.
            GestureCommand::End => {
                if self.session.is_selecting() {
                    if let Some(op) = self.session.end_selection() {
                        self.commit(ActionKind::Select(op));
                    }
                } else if let Some(rec) = self.session.end_stroke() {
                    self.log_debug_samples();
                    self.commit(ActionKind::CommitStroke(rec));
                }
                self.preview = None;
            }
            GestureCommand::Cancel => {
                self.session.cancel_stroke();
                self.preview = None;
            }
        }
    }

    /// Document-state mutations: every arm here either commits an action or
    /// navigates the history that holds them.
    fn process_doc(&mut self, command: DocCommand) {
        match command {
            DocCommand::Undo => {
                self.preview = None;
                // Shared sessions log undo as an action peers can order
                // (DESIGN.md §5.4, §12.3); solo falls back to navigation.
                if let Some(target) = self.timeline.undo_as_action() {
                    self.commit(ActionKind::Undo(target));
                } else {
                    self.timeline.undo(&mut self.apply);
                }
                self.apply_document_surface();
            }
            DocCommand::Redo => {
                self.preview = None;
                // Redo is an `Undo` of an `Undo` in a shared session.
                if let Some(target) = self.timeline.redo_as_action() {
                    self.commit(ActionKind::Undo(target));
                } else {
                    self.timeline.redo(&mut self.apply);
                }
                self.apply_document_surface();
            }
            DocCommand::Select(op) => self.commit(ActionKind::Select(op)),
            DocCommand::InvertSelection => self.commit(ActionKind::InvertSelection),
            DocCommand::SetSurface(id) => {
                if id != self.document().surface {
                    self.commit(ActionKind::SetSurface(id));
                    self.apply_document_surface();
                }
            }
            DocCommand::AddLayer { above } => {
                let id = LayerId(self.next_layer);
                self.next_layer += 1;
                self.commit(ActionKind::AddLayer { id, above });
                // A freshly added layer becomes the active painting target.
                self.session.active_layer = id;
            }
            DocCommand::RemoveLayer(id) => {
                self.commit(ActionKind::RemoveLayer(id));
                // Keep the active layer valid after removal.
                if self.session.active_layer == id
                    && let Some(first) = self.document().layers.iter().next()
                {
                    self.session.active_layer = first.id;
                }
            }
            DocCommand::SetLayerBlend(id, blend) => {
                self.commit(ActionKind::SetLayerBlend(id, blend))
            }
            DocCommand::SetLayerOpacity(id, opacity) => {
                self.commit(ActionKind::SetLayerOpacity(id, opacity))
            }
            DocCommand::SetLayerVisible(id, visible) => {
                self.commit(ActionKind::SetLayerVisible(id, visible))
            }
            DocCommand::MoveLayer { id, above } => self.commit(ActionKind::MoveLayer { id, above }),
        }
    }

    /// View-state mutations: nothing here is logged, replicated, or reachable by
    /// undo.
    fn process_view(&mut self, command: ViewCommand) {
        match command {
            ViewCommand::SetTool(tool) => {
                // Switching away mid-gesture abandons it rather than committing a
                // half-dragged marquee.
                self.session.cancel_stroke();
                self.preview = None;
                self.session.tool = tool;
            }
            ViewCommand::SetBrush(brush) => {
                self.session.brush = brush;
                self.refresh_preview();
            }
            ViewCommand::Pan { delta } => {
                // Grab-and-drag: content follows the cursor, so the view center
                // moves opposite by the drag delta (converted to canvas units).
                self.session.view.center -= delta / self.session.view.zoom;
            }
            ViewCommand::Zoom { anchor, factor } => {
                self.session.view.zoom_about(anchor, factor);
            }
            ViewCommand::Resize(viewport) => {
                self.session.view.viewport = viewport;
            }
            ViewCommand::SetSelectionMode(mode) => self.session.selection_mode = mode,
            ViewCommand::SetSelectionFeather(feather) => {
                self.session.selection_feather = feather.max(0.0)
            }
            ViewCommand::SetActiveLayer(id) => {
                if self.document().layer_index(id).is_some() {
                    self.session.active_layer = id;
                }
            }
            ViewCommand::SetMediaParams(params) => self.compositor.set_media(params),
            ViewCommand::SetEnvironment(id) => self.set_environment(id),
        }
    }

    /// Replay a whole recorded stroke as a single commit: start → samples →
    /// end, skipping the per-sample live-preview refresh. `refresh_preview`
    /// re-renders the entire in-flight stroke after every sample — right for
    /// interactive drawing (the user must see each move), but O(n²) across a
    /// replay where nothing is presented in between. This renders the stroke
    /// exactly once, at commit. Used by the brush editor's test-stroke replay.
    pub fn replay_stroke(&mut self, tool: Tool, samples: &[crate::command::InputSample]) {
        self.replay_stroke_seeded(tool, samples, self.clock);
    }

    /// [`Engine::replay_stroke`] with an explicit jitter `seed` instead of the
    /// Lamport clock. Replaying the same samples repeatedly advances the clock
    /// (each replay commits), so the seed — and with it the colour dynamics and
    /// dither — changes on every replay. A caller re-rendering *one* stroke to
    /// show the effect of a brush change (the brush editor's preview) wants the
    /// jitter held fixed, so only the edited parameter moves.
    pub fn replay_stroke_seeded(
        &mut self,
        tool: Tool,
        samples: &[crate::command::InputSample],
        seed: u64,
    ) {
        let mut it = samples.iter();
        let Some(first) = it.next() else { return };
        self.session.start_stroke(tool, *first, seed);
        for s in it {
            self.session.stroke_to(*s);
        }
        if let Some(rec) = self.session.end_stroke() {
            self.commit(ActionKind::CommitStroke(rec));
        }
        self.preview = None;
    }

    /// Render the current canvas (preview if stroking, else committed) into
    /// `target`, clearing to `background` first (DESIGN.md §6.4).
    pub fn render(&mut self, target: &wgpu::TextureView, background: wgpu::Color) {
        let doc = self
            .preview
            .as_ref()
            .unwrap_or_else(|| self.timeline.current());

        // Gather populated tiles bottom-to-top, skipping hidden layers and
        // tagging each tile with its layer opacity. Normal-blend layers compose
        // correctly under premultiplied "over"; richer blend modes (which need
        // per-layer isolation) are a follow-up.
        let mut tiles: Vec<(crate::geom::TileCoord, TilePairHandle, f32)> = Vec::new();
        for layer in doc.layers.iter() {
            if !layer.visible || layer.opacity <= 0.0 {
                continue;
            }
            for (coord, handle) in layer.tiles.iter() {
                tiles.push((*coord, handle.clone(), layer.opacity));
            }
        }

        let bg_channels = self.color_space.rgb_to_channels([
            background.r as f32,
            background.g as f32,
            background.b as f32,
        ]);

        let view = self.session.view;
        let selection = doc.selection.clone();
        self.compositor
            .render(target, view, bg_channels, &tiles, &selection);
    }

    /// Render the current canvas to a CPU-side image at the viewport size
    /// (DESIGN.md §9). The backbone of golden tests and export. The target uses
    /// the engine's configured format, so it matches on-screen rendering.
    pub fn render_to_image(&mut self, background: wgpu::Color) -> RgbaImage {
        let size = self.session.view.viewport;
        let target = self.gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("stark export target"),
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        self.render(&view, background);
        let pixels = crate::gpu::readback::read_rgba8(&self.gpu, &target, size);
        RgbaImage::new(size.width, size.height, pixels)
    }

    /// Snapshot the document as a saveable [`DocumentFile`] (DESIGN.md §8),
    /// bundling the brush-shape assets that strokes actually reference (§6.6).
    pub fn document_file(&self) -> DocumentFile {
        let actions = self.timeline.clone_actions();
        let mut referenced = std::collections::HashSet::new();
        for action in &actions {
            if let ActionKind::CommitStroke(rec) = &action.kind
                && let BrushShape::Stamp(id) = rec.brush.shape
            {
                referenced.insert(id);
            }
        }
        let assets = self
            .apply
            .assets
            .all_bytes()
            .into_iter()
            .filter(|(id, _)| referenced.contains(id))
            .collect();
        let mut file = DocumentFile::new(actions);
        file.canvas.color_space = self.color_space.id();
        file.canvas.surface = self.initial_surface;
        file.assets = assets;
        file
    }

    /// Serialize the document to the compact on-disk container (DESIGN.md §8).
    pub fn save_bytes(&self) -> Result<Vec<u8>> {
        self.document_file().to_bytes()
    }

    /// Replace the document by replaying a loaded file's action log. The full
    /// undo timeline is available afterwards — undo-after-load (DESIGN.md §8).
    pub fn load_document(&mut self, file: &DocumentFile) {
        self.frozen_head = None;
        // The surface the log starts from, before `reset_document` seeds with it.
        // Replayed `SetSurface` actions move it from there (DESIGN.md §6.4).
        self.initial_surface = file.canvas.surface;
        self.reset_document();
        // Match the document's color space before replaying (DESIGN.md §6.7).
        if file.canvas.color_space != self.color_space.id() {
            self.rebuild_gpu_for(file.canvas.color_space);
        }
        // Brush assets must be available before replaying strokes that use them.
        for (_, bytes) in &file.assets {
            if let Err(e) = self.apply.assets.insert_bytes(bytes) {
                eprintln!("skipping unreadable brush asset: {e}");
            }
        }
        // Replay only the *effective* sequence: a file saved from a shared
        // session is the full log, including `Undo` actions and the actions
        // they suppress (DESIGN.md §12.3). A solo load flattens those away.
        for action in effective_actions(&file.actions) {
            self.replay_one(action);
        }
        self.resync_counters(&file.actions);
        // Whatever the replayed log left the document on.
        self.apply_document_surface();
    }

    /// Decode and load a container produced by [`Engine::save_bytes`].
    pub fn load_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        let file = DocumentFile::from_bytes(bytes)?;
        self.load_document(&file);
        Ok(())
    }

    /// Replay a document, invoking `on_frame` with the rendered image after each
    /// action — a timelapse (DESIGN.md §8). Ends with the document fully loaded.
    pub fn replay_timelapse(
        &mut self,
        file: &DocumentFile,
        background: wgpu::Color,
        mut on_frame: impl FnMut(RgbaImage),
    ) {
        self.reset_document();
        for (_, bytes) in &file.assets {
            let _ = self.apply.assets.insert_bytes(bytes);
        }
        for action in effective_actions(&file.actions) {
            self.replay_one(action);
            on_frame(self.render_to_image(background));
        }
        self.resync_counters(&file.actions);
    }

    /// A snapshot of UI-facing state (DESIGN.md §7).
    pub fn observe(&self) -> ObservableState {
        let doc = self.timeline.current();
        let layers = doc
            .layers
            .iter()
            .map(|l| LayerInfo {
                id: l.id,
                blend: l.blend,
                opacity: l.opacity,
                visible: l.visible,
            })
            .collect();
        ObservableState {
            can_undo: self.timeline.can_undo(),
            can_redo: self.timeline.can_redo(),
            is_stroking: self.session.is_stroking(),
            tool: self.session.tool,
            brush: self.session.brush,
            view: self.session.view,
            bounds: doc.bounds,
            active_layer: self.session.active_layer,
            layers,
            has_selection: doc.selection.is_active(),
            selection_mode: self.session.selection_mode,
            selection_feather: self.session.selection_feather,
            media: self.compositor.media(),
            environment: self.environment.id(),
            color_space: self.color_space.id(),
            surface: self.document().surface,
        }
    }

    /// The current committed document state.
    pub fn document(&self) -> &DocState {
        self.timeline.current()
    }

    /// The GPU context this engine renders with (for surface/readback setup).
    pub fn gpu(&self) -> &GpuContext {
        &self.gpu
    }

    /// The current pan/zoom view (for mapping pointer input to canvas space).
    pub fn view(&self) -> ViewTransform {
        self.session.view
    }

    /// The current media/lighting parameters (DESIGN.md §6.3).
    pub fn media_params(&self) -> crate::gpu::MediaParams {
        self.compositor.media()
    }

    /// Import a brush-shape image (PNG bytes), returning its content id for use
    /// in `BrushParams::shape = BrushShape::Stamp(id)` (DESIGN.md §6.6).
    pub fn import_brush(&self, png_bytes: &[u8]) -> Result<AssetId> {
        self.apply.assets.import(png_bytes)
    }

    // --- collaboration (DESIGN.md §12) -----------------------------------
    //
    // The engine stays network-agnostic: it owns the merge semantics (the
    // `ReplicatedTimeline`) and these hooks; `stark-net` owns the wire.

    /// Whether this engine is in a shared session (replicated timeline active).
    pub fn is_shared(&self) -> bool {
        self.outbox_enabled
    }

    /// This engine's author id for new actions.
    pub fn actor(&self) -> ActorId {
        self.actor
    }

    /// Start sharing the **current** document as `actor` (the host side).
    /// Converts the linear history into a [`ReplicatedTimeline`] over the same
    /// log. Solo-authored actions ([`ActorId::SOLO`]) are rewritten to `actor`
    /// — done once, before any peer has seen them — so the sharer can still
    /// undo their pre-share strokes (undo targets *my* actions, §12.3).
    pub fn start_collaboration(&mut self, actor: ActorId) {
        if self.is_shared() {
            return;
        }
        let mut log = self.timeline.clone_actions();
        for a in &mut log {
            if a.id.actor == ActorId::SOLO {
                a.id.actor = actor;
            }
        }
        let ctx = &mut self.apply;
        let initial = DocState::with_layer(ROOT_LAYER);
        self.timeline = Box::new(ReplicatedTimeline::from_log(actor, initial, log, ctx));
        self.actor = actor;
        self.outbox_enabled = true;
        self.preview = None;
    }

    /// Join a shared session (the peer side): replace the document with the
    /// session's **full** log — including `Undo` actions, which the replicated
    /// timeline resolves — and author future actions as `actor`.
    pub fn join_collaboration(&mut self, file: &DocumentFile, actor: ActorId) {
        // The surface the shared log starts from; replayed `SetSurface` actions
        // move it from there, exactly as in `load_document` (DESIGN.md §6.4).
        self.initial_surface = file.canvas.surface;
        self.reset_document();
        if file.canvas.color_space != self.color_space.id() {
            self.rebuild_gpu_for(file.canvas.color_space);
        }
        for (_, bytes) in &file.assets {
            if let Err(e) = self.apply.assets.insert_bytes(bytes) {
                tracing::warn!("skipping unreadable brush asset: {e}");
            }
        }
        let ctx = &mut self.apply;
        let initial = DocState::with_layer(ROOT_LAYER).with_surface(self.initial_surface);
        self.timeline = Box::new(ReplicatedTimeline::from_log(
            actor,
            initial,
            file.actions.clone(),
            ctx,
        ));
        self.resync_counters(&file.actions);
        self.actor = actor;
        self.outbox_enabled = true;
        // Whatever the replayed log left the document on.
        self.apply_document_surface();
    }

    /// Leave a shared session: stop queueing broadcasts. The replicated
    /// timeline (and the shared log) stays — editing continues solo on the
    /// same canvas, and a later [`Self::start_collaboration`] re-shares it.
    pub fn end_collaboration(&mut self) {
        self.outbox.clear();
        self.outbox_enabled = false;
    }

    /// Integrate an action authored by a peer (DESIGN.md §12.1). Idempotent —
    /// duplicates are rejected by id. Advances the Lamport clock past the
    /// remote action so future local ids order after everything seen.
    pub fn merge_remote(&mut self, action: Action) -> bool {
        // Replaces the document the frozen head was composited onto.
        self.frozen_head = None;
        self.clock = self.clock.max(action.id.lamport + 1);
        if let ActionKind::AddLayer { id, .. } = &action.kind {
            self.next_layer = self.next_layer.max(id.0 + 1);
        }
        let ctx = &mut self.apply;
        let merged = self.timeline.merge(action, ctx);
        // The live preview is rendered over the committed state; re-base it if
        // a remote stroke landed mid-gesture.
        if merged && self.session.is_stroking() {
            self.refresh_preview();
        }
        // A peer may have switched the surface (DESIGN.md §6.4).
        self.apply_document_surface();
        merged
    }

    /// Drain locally-committed actions awaiting broadcast (empty when solo).
    pub fn take_outbox(&mut self) -> Vec<Action> {
        std::mem::take(&mut self.outbox)
    }

    /// Every imported brush asset (id + canonical PNG bytes) — used to seed a
    /// transport session's asset mirror so peers can fetch any brush a future
    /// stroke references (DESIGN.md §12.4).
    pub fn all_asset_bytes(&self) -> Vec<(AssetId, Vec<u8>)> {
        self.apply.assets.all_bytes()
    }

    fn commit(&mut self, kind: ActionKind) {
        let action = Action {
            id: self.next_action_id(),
            kind,
        };
        let ctx = &mut self.apply;
        self.timeline.push(action.clone(), ctx);
        if self.outbox_enabled {
            self.outbox.push(action);
        }
    }

    /// The document's color space id (DESIGN.md §6.7).
    pub fn color_space(&self) -> ColorSpaceId {
        self.color_space.id()
    }

    /// Start a fresh, empty document in `color_space`, on `surface`.
    ///
    /// The **only** way to choose a colour space, and deliberately so: the channel
    /// layouts differ between spaces, so existing tiles cannot be reinterpreted and
    /// changing it can never preserve a document. Modelling it as a setter hid that
    /// — every caller was really asking for a new document (DESIGN.md §6.7).
    ///
    /// Takes `&mut self` rather than being an associated function because
    /// frontend-provided *resources* survive: imported brush assets, and the
    /// registered surface and environment bytes. Those belong to the app, not to
    /// the document, and re-fetching them on every New would be gratuitous.
    pub fn new_document(&mut self, color_space: ColorSpaceId, surface: SurfaceId) {
        self.frozen_head = None;
        self.initial_surface = surface;
        self.reset_document();
        self.rebuild_gpu_for(color_space);
        self.apply_document_surface();
    }

    /// The document's current surface (DESIGN.md §6.4). Change it with
    /// [`crate::command::DocCommand::SetSurface`].
    pub fn surface(&self) -> SurfaceId {
        self.document().surface
    }

    /// Whether `id` is ready to use — `Flat` always is; an image-backed surface
    /// is ready once its bytes have been [`register_surface`](Self::register_surface)ed.
    pub fn surface_loaded(&self, id: SurfaceId) -> bool {
        self.surface.is_loaded(id)
    }

    /// Provide (frontend-fetched) image bytes for a surface. If it's the one in
    /// use, the surface is rebuilt so the bytes take effect immediately.
    pub fn register_surface(&mut self, id: SurfaceId, png_bytes: Vec<u8>) {
        if self.surface.register(&self.gpu, id, png_bytes) {
            self.apply_surface();
        }
    }

    /// Bring the GPU-side surface in line with the document's, rebuilding it if the
    /// document moved to a different one — after a commit, an undo, a load, or a
    /// remote merge. A no-op when unchanged, which is the common case.
    ///
    /// There is deliberately no public `set_surface`: the surface is document state
    /// (DESIGN.md §6.4), so it changes by logging an action like anything else.
    fn apply_document_surface(&mut self) {
        let id = self.document().surface;
        if self.surface.set(&self.gpu, id) {
            self.apply_surface();
        }
    }

    /// Rebind the current surface in the media pass — the only thing that samples
    /// it. No pipeline or pool rebuild, no document reset.
    fn apply_surface(&mut self) {
        self.compositor.set_surface(self.surface.current().clone());
    }

    /// The current lighting environment (DESIGN.md §6.3).
    pub fn environment(&self) -> EnvironmentId {
        self.environment.id()
    }

    /// Whether `id` is ready — `Studio` always is; an HDR environment is ready once
    /// its bytes have been [`register_environment`](Self::register_environment)ed.
    pub fn environment_loaded(&self, id: EnvironmentId) -> bool {
        self.environment.is_loaded(id)
    }

    /// Provide (frontend-fetched) HDR bytes for an environment. If it's the one in
    /// use, it's rebuilt so the bytes take effect immediately.
    pub fn register_environment(&mut self, id: EnvironmentId, hdr_bytes: Vec<u8>) {
        if self.environment.register(&self.gpu, id, hdr_bytes) {
            self.apply_environment();
        }
    }

    /// Switch the lighting environment. A view setting, so this never touches the
    /// document — it just re-lights the canvas on the next render. Image
    /// environments fall back to the procedural studio until their bytes arrive.
    ///
    /// Private: reached through
    /// [`crate::command::ViewCommand::SetEnvironment`].
    fn set_environment(&mut self, id: EnvironmentId) {
        if self.environment.set(&self.gpu, id) {
            self.apply_environment();
        }
    }

    /// Rebind the current environment in the media pass.
    fn apply_environment(&mut self) {
        self.compositor
            .set_environment(self.environment.current().clone());
    }

    /// Rebuild the GPU subsystems (pool/stroke/compositor) for `id`. Assumes the
    /// document is already empty (no tiles of the old format are referenced).
    fn rebuild_gpu_for(&mut self, id: ColorSpaceId) {
        let cs = id.make();
        let (pool, stroke, compositor) = build_gpu(GpuBuild {
            gpu: &self.gpu,
            target_format: self.target_format,
            viewport: self.session.view.viewport,
            cs: &cs,
            surface: self.surface.current(),
            environment: self.environment.current(),
            selection: &self.apply.selection,
        });
        self.color_space = cs;
        self.apply.pool = pool;
        self.apply.stroke = stroke;
        self.compositor = compositor;
    }

    /// Reset to an empty document (one root layer) before a load/replay. Also
    /// leaves any shared session: the caller (UI/transport) tears down the
    /// network side; `join_collaboration` re-enables after its reset.
    fn reset_document(&mut self) {
        self.timeline = Box::new(LinearTimeline::new(
            DocState::with_layer(ROOT_LAYER).with_surface(self.initial_surface),
        ));
        self.preview = None;
        self.clock = 0;
        self.next_layer = 1;
        self.session.cancel_stroke();
        self.session.active_layer = ROOT_LAYER;
        self.actor = ActorId::SOLO;
        self.outbox.clear();
        self.outbox_enabled = false;
    }

    /// Commit one already-built action onto the timeline (replays its GPU work).
    fn replay_one(&mut self, action: Action) {
        let ctx = &mut self.apply;
        self.timeline.push(action, ctx);
    }

    /// After loading, advance the id counters past everything in the log so new
    /// edits get fresh, monotonic ids.
    fn resync_counters(&mut self, actions: &[Action]) {
        let mut max_lamport = None;
        let mut max_layer = 0u64;
        for a in actions {
            max_lamport = Some(max_lamport.map_or(a.id.lamport, |m: u64| m.max(a.id.lamport)));
            if let ActionKind::AddLayer { id, .. } = &a.kind {
                max_layer = max_layer.max(id.0);
            }
        }
        self.clock = max_lamport.map_or(0, |m| m + 1);
        self.next_layer = max_layer + 1;
    }

    fn next_action_id(&mut self) -> ActionId {
        let id = ActionId {
            lamport: self.clock,
            actor: self.actor,
        };
        self.clock += 1;
        id
    }

    /// Re-render the in-flight stroke onto a CoW copy of the committed state.
    /// Uses the exact stroke path that a commit/replay would (DESIGN.md §6.2),
    /// so live and committed pixels match.
    fn refresh_preview(&mut self) {
        let Some(rec) = self.session.preview_record() else {
            self.preview = None;
            self.frozen_head = None;
            return;
        };
        let frozen = self.session.frozen_spans();
        let head = match self.frozen_head.take() {
            // Advance the head over the spans that have settled since last time.
            Some(head) if head.spans <= frozen => head,
            // Nothing yet, or the fit went backwards (a new stroke): start over from
            // the committed document, with a fresh (uncharged) brush.
            _ => FrozenHead {
                spans: 0,
                dist: 0.0,
                tool: None,
                state: self.timeline.current().clone(),
            },
        };
        let head = if frozen > head.spans {
            self.advance_head(head, &rec, frozen)
        } else {
            head
        };

        let all = crate::path::span_count(rec.path.len());
        let tail = StrokeSpans {
            range: head.spans..all,
            dist: head.dist,
        };
        // The diagnostic recolours only what this move actually redrew, so the seam
        // between tinted and untinted paint *is* the freezing boundary. Build-time
        // only (the `debug-unfrozen` feature): a shipping build has no code path that
        // paints the tail in anything but the stroke's own colour.
        #[cfg(feature = "debug-unfrozen")]
        let tinted = {
            let mut r = rec.clone();
            r.brush.color = DEBUG_UNFROZEN_COLOR;
            r
        };
        #[cfg(feature = "debug-unfrozen")]
        let tail_rec = &tinted;
        #[cfg(not(feature = "debug-unfrozen"))]
        let tail_rec = &rec;
        // The tail reaches the end of the stroke, so the state it leaves the brush in
        // is handed to nobody — it is thrown away and rebuilt from the head on the next
        // move, which is exactly what makes the tail re-renderable.
        let (preview, _) = self.render_span_range(&head.state, tail_rec, tail, head.tool.as_ref());
        self.preview = Some(preview);
        self.frozen_head = Some(head);
    }

    /// Dump the finished stroke's raw input as a pasteable Rust literal.
    ///
    /// A misfit seen in the app is otherwise unreproducible: the fit depends on the
    /// exact sequence of pointer reports — their spacing carries the pen's speed,
    /// which is what the density policy and the freezing both key off — and no
    /// synthetic curve stands in for a real hand. This turns one into a test case.
    fn log_debug_samples(&mut self) {
        if !cfg!(feature = "debug-unfrozen") || self.debug_samples.is_empty() {
            return;
        }
        // Positions *and* the pen channels. Position alone is not the input: pressure
        // sizes the brush and tilt steers it, both are fitted as their own least-squares
        // channels, and a capture without them cannot reproduce a fault in either.
        let mut lit = String::from("&[");
        for (i, s) in self.debug_samples.iter().enumerate() {
            if i > 0 {
                lit.push(',');
            }
            lit.push_str(&format!(
                "[{:.2},{:.2},{:.3},{:.3},{:.3}]",
                s.pos.x, s.pos.y, s.pressure, s.tilt.x, s.tilt.y
            ));
        }
        lit.push(']');
        tracing::info!(
            samples = self.debug_samples.len(),
            "raw stroke [x,y,pressure,tiltx,tilty]: {lit}"
        );
        self.debug_samples.clear();
    }

    /// Draw spans `head.spans..frozen` onto the frozen head, so the next move need
    /// not draw them again.
    fn advance_head(&self, head: FrozenHead, rec: &StrokeRecord, frozen: usize) -> FrozenHead {
        let spans = StrokeSpans {
            range: head.spans..frozen,
            dist: head.dist,
        };
        // The renderer reports where it stopped rather than the caller recomputing it:
        // arc length is accumulated along the *emitted* polyline, and only the renderer
        // knows the budget it flattened at (a dynamics brush may have coarsened it), so
        // a second measurement here could hand the tail a distance the head never
        // reached — which `drain` and the colour-dynamics noise would both show.
        let (state, carry) = self.render_span_range(&head.state, rec, spans, head.tool.as_ref());
        FrozenHead {
            spans: frozen,
            dist: carry.dist,
            // `None` from a range means "unchanged", not "reset" — a range with no
            // geometry runs nothing and leaves the brush as it found it.
            tool: carry.tool.or(head.tool),
            state,
        }
    }

    /// Render one span range of the in-flight stroke over an arbitrary base, resuming
    /// the brush from `tool` and reporting what the next range must resume from.
    ///
    /// Uses the same entry point a commit does (`StrokeRenderer::render_range`), so
    /// the live preview and the `Action::apply` that replaces it draw the same pixels.
    fn render_span_range(
        &self,
        base: &DocState,
        rec: &StrokeRecord,
        spans: StrokeSpans,
        tool: Option<&crate::gpu::stroke::ToolState>,
    ) -> (DocState, crate::gpu::stroke::StrokeCarry) {
        let carry_only = |dist| crate::gpu::stroke::StrokeCarry { dist, tool: None };
        let Some(idx) = base.layer_index(rec.layer) else {
            return (base.clone(), carry_only(spans.dist));
        };
        let layer = base.layer_at(idx).clone();
        let (tiles, carry) = self.apply.stroke.render_range(
            crate::gpu::stroke::StrokeScene {
                pool: &self.apply.pool,
                assets: &self.apply.assets,
                base: &layer.tiles,
                selection: &base.selection,
            },
            rec,
            spans,
            tool,
        );
        (base.with_layer_at(idx, Layer { tiles, ..layer }), carry)
    }

    /// Rasterize the in-flight selection gesture onto a copy of the committed state,
    /// so the outline follows the drag. Uses the same op the commit will (§6.8), so
    /// what is previewed is what lands.
    fn refresh_selection_preview(&mut self) {
        let Some(op) = self.session.preview_selection() else {
            self.preview = None;
            return;
        };
        let base = self.timeline.current();
        self.preview = self
            .apply
            .selection
            .apply(&self.apply.pool, &base.selection, &op)
            .map(|selection| base.with_selection(selection));
    }
}

/// Build the GPU subsystems whose layout/shaders depend on the color space.
/// What the colour-space-dependent GPU subsystems are built from.
///
/// Grouped because they are always supplied together: the pool, stroke renderer and
/// compositor are torn down and rebuilt as a set whenever the colour space changes
/// (DESIGN.md §6.7), and `surface` / `environment` / `selection` are precisely the
/// pieces that *survive* that rebuild and have to be handed back in.
struct GpuBuild<'a> {
    gpu: &'a GpuContext,
    target_format: wgpu::TextureFormat,
    viewport: Extent2,
    cs: &'a Arc<dyn ColorSpace>,
    surface: &'a Surface,
    environment: &'a Environment,
    selection: &'a SelectionRenderer,
}

fn build_gpu(b: GpuBuild<'_>) -> (TilePool, StrokeRenderer, Compositor) {
    let GpuBuild {
        gpu,
        target_format,
        viewport,
        cs,
        surface,
        environment,
        selection,
    } = b;
    // Selection masks are pooled and recycled like paint (DESIGN.md §6.8), so their
    // format joins the pool's free lists.
    let pool = TilePool::new(
        gpu.clone(),
        [cs.color_format(), cs.aux_format(), MASK_FORMAT],
    );
    let stroke = StrokeRenderer::new(gpu, cs.clone(), selection.clone());
    let compositor = Compositor::new(
        gpu,
        target_format,
        viewport,
        cs.as_ref(),
        surface.clone(),
        environment.clone(),
    );
    (pool, stroke, compositor)
}

/// Convenience for tests/tools: build an engine on a headless device.
pub async fn headless_engine(
    target_format: wgpu::TextureFormat,
    viewport: Extent2,
) -> Result<Engine> {
    headless_engine_with(target_format, viewport, ColorSpaceId::Oklab).await
}

/// Headless engine in a chosen color space (DESIGN.md §6.7).
pub async fn headless_engine_with(
    target_format: wgpu::TextureFormat,
    viewport: Extent2,
    color_space: ColorSpaceId,
) -> Result<Engine> {
    let gpu = GpuContext::headless().await?;
    Ok(Engine::new_with_color_space(
        gpu,
        target_format,
        viewport,
        color_space,
    ))
}
