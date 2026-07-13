//! The engine: owns the GPU, session, and timeline; turns commands into state
//! and renders the canvas (DESIGN.md §7).
//!
//! For the MVP this exposes a synchronous [`Engine::process`]. The asynchronous
//! actor loop and reactive `ObservableState` channel (DESIGN.md §7) wrap this
//! same core in a later step.

use std::sync::Arc;

use crate::assets::{AssetId, AssetStore};
use crate::colorspace::{ColorSpace, ColorSpaceId};
use crate::command::InputCommand;
use crate::document::{
    effective_actions, Action, ActionId, ActionKind, ActorId, ApplyCtx, BrushParams, BrushShape,
    CanvasBounds, DocState, Layer, LayerId, LinearTimeline, ReplicatedTimeline, StrokeRecord,
    Timeline, Tool,
};
use crate::geom::{Extent2, ViewTransform};
use crate::gpu::{
    Compositor, Environment, EnvironmentId, GpuContext, StrokeRenderer, Surface, SurfaceId,
    TilePairHandle, TilePool,
};
use crate::image::RgbaImage;
use crate::io::DocumentFile;
use crate::Result;

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
}

pub struct Engine {
    gpu: GpuContext,
    target_format: wgpu::TextureFormat,
    color_space: Arc<dyn ColorSpace>,
    pool: TilePool,
    stroke: StrokeRenderer,
    assets: AssetStore,
    compositor: Compositor,
    /// The physical canvas surface (tooth + relief). Color-space-independent, so
    /// it survives color-space rebuilds (DESIGN.md §6.4).
    surface: Surface,
    /// Which surface is loaded — a document property, saved in `CanvasMeta`.
    surface_id: SurfaceId,
    /// Frontend-provided image bytes for non-`Flat` surfaces, keyed by id. The
    /// engine embeds none; the frontend fetches and registers them at runtime
    /// (DESIGN.md §6.4). Missing bytes fall back to `Flat`.
    surface_assets: std::collections::HashMap<SurfaceId, Vec<u8>>,
    /// The HDR lighting environment (image-based lighting). A *view* setting — not
    /// historized, color-space-independent — so it survives rebuilds and switching
    /// it never touches the document (DESIGN.md §6.3).
    environment: Environment,
    environment_id: EnvironmentId,
    /// Frontend-provided HDR bytes for non-procedural environments, keyed by id
    /// (the engine embeds none; the frontend fetches them at runtime).
    environment_assets: std::collections::HashMap<EnvironmentId, Vec<u8>>,
    timeline: Box<dyn Timeline>,
    session: crate::session::Session,
    /// Live preview of the in-flight stroke, composited in place of the
    /// committed state while painting (DESIGN.md §6.2). `None` when idle.
    preview: Option<DocState>,
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
        let surface_id = SurfaceId::default();
        let surface = Surface::flat(&gpu);
        // Lighting starts on the procedural studio environment; image HDRs are
        // registered later by the frontend (DESIGN.md §6.3).
        let environment_id = EnvironmentId::default();
        let environment = Environment::studio(&gpu);
        let (pool, stroke, compositor) =
            build_gpu(&gpu, target_format, viewport, &color_space, &surface, &environment);
        let assets = AssetStore::new(gpu.clone());

        let initial = DocState::with_layer(ROOT_LAYER);
        let timeline: Box<dyn Timeline> = Box::new(LinearTimeline::new(initial));
        let session = crate::session::Session::new(ViewTransform::identity(viewport), ROOT_LAYER);

        Self {
            gpu,
            target_format,
            color_space,
            pool,
            stroke,
            assets,
            compositor,
            surface,
            surface_id,
            surface_assets: std::collections::HashMap::new(),
            environment,
            environment_id,
            environment_assets: std::collections::HashMap::new(),
            timeline,
            session,
            preview: None,
            actor: ActorId::SOLO,
            clock: 0,
            next_layer: 1,
            outbox: Vec::new(),
            outbox_enabled: false,
        }
    }

    /// Apply one input command (DESIGN.md §4).
    pub fn process(&mut self, command: InputCommand) {
        match command {
            InputCommand::StartStroke { tool, sample } => {
                let seed = self.clock;
                self.session.start_stroke(tool, sample, seed);
                self.refresh_preview();
            }
            InputCommand::StrokeTo { sample } => {
                self.session.stroke_to(sample);
                self.refresh_preview();
            }
            InputCommand::EndStroke => {
                if let Some(rec) = self.session.end_stroke() {
                    self.commit(ActionKind::CommitStroke(rec));
                }
                self.preview = None;
            }
            InputCommand::CancelStroke => {
                self.session.cancel_stroke();
                self.preview = None;
            }
            InputCommand::Undo => {
                self.preview = None;
                // Shared sessions log undo as an action peers can order
                // (DESIGN.md §5.4, §12.3); solo falls back to navigation.
                if let Some(target) = self.timeline.undo_as_action() {
                    self.commit(ActionKind::Undo(target));
                } else {
                    let mut ctx = self.apply_ctx();
                    self.timeline.undo(&mut ctx);
                }
            }
            InputCommand::Redo => {
                self.preview = None;
                // Redo is an `Undo` of an `Undo` in a shared session.
                if let Some(target) = self.timeline.redo_as_action() {
                    self.commit(ActionKind::Undo(target));
                } else {
                    let mut ctx = self.apply_ctx();
                    self.timeline.redo(&mut ctx);
                }
            }
            InputCommand::SetTool(tool) => self.session.tool = tool,
            InputCommand::SetBrush(brush) => {
                self.session.brush = brush;
                self.refresh_preview();
            }
            InputCommand::Pan { delta } => {
                // Grab-and-drag: content follows the cursor, so the view center
                // moves opposite by the drag delta (converted to canvas units).
                self.session.view.center -= delta / self.session.view.zoom;
            }
            InputCommand::Zoom { anchor, factor } => {
                self.session.view.zoom_about(anchor, factor);
            }
            InputCommand::SetActiveLayer(id) => {
                // Session state, like tool selection — never historized.
                if self.document().layer_index(id).is_some() {
                    self.session.active_layer = id;
                }
            }
            InputCommand::AddLayer { above } => {
                let id = LayerId(self.next_layer);
                self.next_layer += 1;
                self.commit(ActionKind::AddLayer { id, above });
                // A freshly added layer becomes the active painting target.
                self.session.active_layer = id;
            }
            InputCommand::RemoveLayer(id) => {
                self.commit(ActionKind::RemoveLayer(id));
                // Keep the active layer valid after removal.
                if self.session.active_layer == id
                    && let Some(first) = self.document().layers.iter().next()
                {
                    self.session.active_layer = first.id;
                }
            }
            InputCommand::SetLayerBlend(id, blend) => {
                self.commit(ActionKind::SetLayerBlend(id, blend))
            }
            InputCommand::SetLayerOpacity(id, opacity) => {
                self.commit(ActionKind::SetLayerOpacity(id, opacity))
            }
            InputCommand::SetLayerVisible(id, visible) => {
                self.commit(ActionKind::SetLayerVisible(id, visible))
            }
            InputCommand::MoveLayer { id, above } => {
                self.commit(ActionKind::MoveLayer { id, above })
            }
        }
    }

    /// Replay a whole recorded stroke as a single commit: start → samples →
    /// end, skipping the per-sample live-preview refresh. `refresh_preview`
    /// re-renders the entire in-flight stroke after every sample — right for
    /// interactive drawing (the user must see each move), but O(n²) across a
    /// replay where nothing is presented in between. This renders the stroke
    /// exactly once, at commit. Used by the brush editor's test-stroke replay.
    pub fn replay_stroke(&mut self, tool: Tool, samples: &[crate::command::InputSample]) {
        let mut it = samples.iter();
        let Some(first) = it.next() else { return };
        self.session.start_stroke(tool, *first, self.clock);
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
        let doc = self.preview.as_ref().unwrap_or_else(|| self.timeline.current());

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

        let bg_channels = self.color_space.rgb_to_channels([background.r as f32, background.g as f32, background.b as f32]);

        let view = self.session.view;
        self.compositor.render(target, view, bg_channels, &tiles);
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
            .assets
            .all_bytes()
            .into_iter()
            .filter(|(id, _)| referenced.contains(id))
            .collect();
        let mut file = DocumentFile::new(actions);
        file.canvas.color_space = self.color_space.id();
        file.canvas.surface = self.surface_id;
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
        self.reset_document();
        // Match the document's surface before replaying — it affects deposition
        // (DESIGN.md §6.4). Update the id first; the rebuild below picks it up.
        if file.canvas.surface != self.surface_id {
            self.surface_id = file.canvas.surface;
            self.surface = build_surface(&self.gpu, self.surface_id, &self.surface_assets);
            self.rebuild_gpu_for(self.color_space.id());
        }
        // Match the document's color space before replaying (DESIGN.md §6.7).
        if file.canvas.color_space != self.color_space.id() {
            self.rebuild_gpu_for(file.canvas.color_space);
        }
        // Brush assets must be available before replaying strokes that use them.
        for (_, bytes) in &file.assets {
            if let Err(e) = self.assets.insert_bytes(bytes) {
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
            let _ = self.assets.insert_bytes(bytes);
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

    /// Resize the viewport (e.g. when the window/canvas changes size). The
    /// compositor's offscreen targets follow on the next render (DESIGN.md §6.4).
    pub fn resize(&mut self, viewport: Extent2) {
        self.session.view.viewport = viewport;
    }

    /// The current media/lighting parameters (DESIGN.md §6.3).
    pub fn media_params(&self) -> crate::gpu::MediaParams {
        self.compositor.media()
    }

    /// Tune the media/lighting parameters of the painterly pass (DESIGN.md §6.3).
    pub fn set_media_params(&mut self, params: crate::gpu::MediaParams) {
        self.compositor.set_media(params);
    }

    /// Import a brush-shape image (PNG bytes), returning its content id for use
    /// in `BrushParams::shape = BrushShape::Stamp(id)` (DESIGN.md §6.6).
    pub fn import_brush(&self, png_bytes: &[u8]) -> Result<AssetId> {
        self.assets.import(png_bytes)
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
        let mut ctx = self.apply_ctx();
        let initial = DocState::with_layer(ROOT_LAYER);
        self.timeline = Box::new(ReplicatedTimeline::from_log(actor, initial, log, &mut ctx));
        self.actor = actor;
        self.outbox_enabled = true;
        self.preview = None;
    }

    /// Join a shared session (the peer side): replace the document with the
    /// session's **full** log — including `Undo` actions, which the replicated
    /// timeline resolves — and author future actions as `actor`.
    pub fn join_collaboration(&mut self, file: &DocumentFile, actor: ActorId) {
        self.reset_document();
        if file.canvas.surface != self.surface_id {
            self.surface_id = file.canvas.surface;
            self.surface = build_surface(&self.gpu, self.surface_id, &self.surface_assets);
            self.rebuild_gpu_for(self.color_space.id());
        }
        if file.canvas.color_space != self.color_space.id() {
            self.rebuild_gpu_for(file.canvas.color_space);
        }
        for (_, bytes) in &file.assets {
            if let Err(e) = self.assets.insert_bytes(bytes) {
                tracing::warn!("skipping unreadable brush asset: {e}");
            }
        }
        let mut ctx = self.apply_ctx();
        let initial = DocState::with_layer(ROOT_LAYER);
        self.timeline = Box::new(ReplicatedTimeline::from_log(
            actor,
            initial,
            file.actions.clone(),
            &mut ctx,
        ));
        self.resync_counters(&file.actions);
        self.actor = actor;
        self.outbox_enabled = true;
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
        self.clock = self.clock.max(action.id.lamport + 1);
        if let ActionKind::AddLayer { id, .. } = &action.kind {
            self.next_layer = self.next_layer.max(id.0 + 1);
        }
        let mut ctx = self.apply_ctx();
        let merged = self.timeline.merge(action, &mut ctx);
        // The live preview is rendered over the committed state; re-base it if
        // a remote stroke landed mid-gesture.
        if merged && self.session.is_stroking() {
            self.refresh_preview();
        }
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
        self.assets.all_bytes()
    }

    fn commit(&mut self, kind: ActionKind) {
        let action = Action {
            id: self.next_action_id(),
            kind,
        };
        let mut ctx = self.apply_ctx();
        self.timeline.push(action.clone(), &mut ctx);
        if self.outbox_enabled {
            self.outbox.push(action);
        }
    }

    /// The document's color space id (DESIGN.md §6.7).
    pub fn color_space(&self) -> ColorSpaceId {
        self.color_space.id()
    }

    /// Switch color space, clearing the canvas (channel layouts differ, so
    /// existing tiles can't be reinterpreted). For a fresh document or a UI
    /// toggle (DESIGN.md §6.7).
    pub fn set_color_space(&mut self, id: ColorSpaceId) {
        self.reset_document();
        self.rebuild_gpu_for(id);
    }

    /// The document's current surface (DESIGN.md §6.4).
    pub fn surface(&self) -> SurfaceId {
        self.surface_id
    }

    /// Whether `id` is ready to use — `Flat` always is; an image-backed surface
    /// is ready once its bytes have been [`register_surface`](Self::register_surface)ed.
    pub fn surface_loaded(&self, id: SurfaceId) -> bool {
        id == SurfaceId::Flat || self.surface_assets.contains_key(&id)
    }

    /// Provide (frontend-fetched) image bytes for a surface. If it's the one in
    /// use, the surface is rebuilt so the bytes take effect immediately.
    pub fn register_surface(&mut self, id: SurfaceId, png_bytes: Vec<u8>) {
        self.surface_assets.insert(id, png_bytes);
        if id == self.surface_id {
            self.surface = build_surface(&self.gpu, id, &self.surface_assets);
            self.apply_surface();
        }
    }

    /// Switch the canvas surface **in place** — the document is preserved
    /// (DESIGN.md §6.4). The surface is view-time today: the weave shows through the
    /// media pass (`thickness = height − surface`, relief normals), and the
    /// deposition tooth gate is a pass-through stub, so existing paint simply
    /// re-reads against the new weave. Image surfaces fall back to `Flat` until
    /// their bytes are registered.
    ///
    /// NOTE: the chosen surface is still saved with the document (`CanvasMeta`,
    /// §8). If a real tooth gate returns, mid-document switches would make replay
    /// non-reproducible — the choice would then need to be historized as an action.
    pub fn set_surface(&mut self, id: SurfaceId) {
        if id == self.surface_id {
            return;
        }
        self.surface_id = id;
        self.surface = build_surface(&self.gpu, id, &self.surface_assets);
        self.apply_surface();
    }

    /// Rebind the current surface in the subsystems that sample it (the sweep's
    /// tooth gate and the media pass) — no pipeline or pool rebuild, no document
    /// reset.
    fn apply_surface(&mut self) {
        self.stroke.set_surface(&self.surface);
        self.compositor.set_surface(self.surface.clone());
    }

    /// The current lighting environment (DESIGN.md §6.3).
    pub fn environment(&self) -> EnvironmentId {
        self.environment_id
    }

    /// Whether `id` is ready — `Studio` always is; an HDR environment is ready once
    /// its bytes have been [`register_environment`](Self::register_environment)ed.
    pub fn environment_loaded(&self, id: EnvironmentId) -> bool {
        id == EnvironmentId::Studio || self.environment_assets.contains_key(&id)
    }

    /// Provide (frontend-fetched) HDR bytes for an environment. If it's the one in
    /// use, it's rebuilt so the bytes take effect immediately.
    pub fn register_environment(&mut self, id: EnvironmentId, hdr_bytes: Vec<u8>) {
        self.environment_assets.insert(id, hdr_bytes);
        if id == self.environment_id {
            self.environment = build_environment(&self.gpu, id, &self.environment_assets);
            self.compositor.set_environment(self.environment.clone());
        }
    }

    /// Switch the lighting environment. A view setting, so this never touches the
    /// document — it just re-lights the canvas on the next render. Image
    /// environments fall back to the procedural studio until their bytes arrive.
    pub fn set_environment(&mut self, id: EnvironmentId) {
        if id == self.environment_id {
            return;
        }
        self.environment_id = id;
        self.environment = build_environment(&self.gpu, id, &self.environment_assets);
        self.compositor.set_environment(self.environment.clone());
    }

    /// Rebuild the GPU subsystems (pool/stroke/compositor) for `id`. Assumes the
    /// document is already empty (no tiles of the old format are referenced).
    fn rebuild_gpu_for(&mut self, id: ColorSpaceId) {
        let cs = id.make();
        let (pool, stroke, compositor) = build_gpu(
            &self.gpu,
            self.target_format,
            self.session.view.viewport,
            &cs,
            &self.surface,
            &self.environment,
        );
        self.color_space = cs;
        self.pool = pool;
        self.stroke = stroke;
        self.compositor = compositor;
    }

    /// Reset to an empty document (one root layer) before a load/replay. Also
    /// leaves any shared session: the caller (UI/transport) tears down the
    /// network side; `join_collaboration` re-enables after its reset.
    fn reset_document(&mut self) {
        self.timeline = Box::new(LinearTimeline::new(DocState::with_layer(ROOT_LAYER)));
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
        let mut ctx = self.apply_ctx();
        self.timeline.push(action, &mut ctx);
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

    fn apply_ctx(&self) -> ApplyCtx {
        ApplyCtx {
            pool: self.pool.clone(),
            stroke: self.stroke.clone(),
            assets: self.assets.clone(),
        }
    }

    /// Re-render the in-flight stroke onto a CoW copy of the committed state.
    /// Uses the exact stroke path that a commit/replay would (DESIGN.md §6.2),
    /// so live and committed pixels match.
    fn refresh_preview(&mut self) {
        let Some(rec) = self.session.preview_record() else {
            self.preview = None;
            return;
        };
        self.preview = Some(self.render_stroke(&rec));
    }

    /// Produce the DocState that committing `rec` would yield, without touching
    /// history. Shared by live preview here and `Action::apply` via the renderer.
    fn render_stroke(&self, rec: &StrokeRecord) -> DocState {
        let base = self.timeline.current();
        let Some(idx) = base.layer_index(rec.layer) else {
            return base.clone();
        };
        let layer = base.layer_at(idx).clone();
        let tiles = self.stroke.render(&self.pool, &self.assets, &layer.tiles, rec);
        base.with_layer_at(idx, Layer { tiles, ..layer })
    }
}

/// Build the GPU subsystems whose layout/shaders depend on the color space.
/// Build the surface texture for `id` from the registry: `Flat` is procedural;
/// image surfaces use their registered bytes, falling back to `Flat` if absent.
fn build_surface(
    gpu: &GpuContext,
    id: SurfaceId,
    assets: &std::collections::HashMap<SurfaceId, Vec<u8>>,
) -> Surface {
    match id {
        SurfaceId::Flat => Surface::flat(gpu),
        other => match assets.get(&other) {
            Some(bytes) => Surface::load(gpu, bytes),
            None => {
                tracing::warn!("surface {other:?} has no registered bytes; using flat");
                Surface::flat(gpu)
            }
        },
    }
}

fn build_gpu(
    gpu: &GpuContext,
    target_format: wgpu::TextureFormat,
    viewport: Extent2,
    cs: &Arc<dyn ColorSpace>,
    surface: &Surface,
    environment: &Environment,
) -> (TilePool, StrokeRenderer, Compositor) {
    let pool = TilePool::new(gpu.clone(), [cs.color_format(), cs.aux_format()]);
    let stroke = StrokeRenderer::new(gpu, cs.clone(), surface.clone());
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

/// Build the environment for `id`: `Studio` is procedural; image environments use
/// their registered HDR bytes, falling back to the procedural studio if absent.
fn build_environment(
    gpu: &GpuContext,
    id: EnvironmentId,
    assets: &std::collections::HashMap<EnvironmentId, Vec<u8>>,
) -> Environment {
    match id {
        EnvironmentId::Studio => Environment::studio(gpu),
        other => match assets.get(&other) {
            Some(bytes) => Environment::load(gpu, bytes),
            None => {
                tracing::warn!("environment {other:?} has no registered bytes; using studio");
                Environment::studio(gpu)
            }
        },
    }
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
    Ok(Engine::new_with_color_space(gpu, target_format, viewport, color_space))
}
