//! The engine: owns the GPU, session, and timeline; turns commands into state
//! and renders the canvas (DESIGN.md §7).
//!
//! For the MVP this exposes a synchronous [`Engine::process`]. The asynchronous
//! actor loop and reactive `ObservableState` channel (DESIGN.md §7) wrap this
//! same core in a later step.

use crate::command::InputCommand;
use crate::document::{
    Action, ActionId, ActionKind, ActorId, ApplyCtx, BrushParams, CanvasBounds, DocState, Layer,
    LayerId, LinearTimeline, StrokeRecord, Timeline, Tool,
};
use crate::geom::{Extent2, ViewTransform};
use crate::gpu::{GpuContext, Presenter, StrokeRenderer, TileHandle, TilePool};
use crate::Result;

/// The starting layer present in every new document.
const ROOT_LAYER: LayerId = LayerId(0);

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
}

pub struct Engine {
    gpu: GpuContext,
    pool: TilePool,
    stroke: StrokeRenderer,
    presenter: Presenter,
    timeline: Box<dyn Timeline>,
    session: crate::session::Session,
    /// Live preview of the in-flight stroke, composited in place of the
    /// committed state while painting (DESIGN.md §6.2). `None` when idle.
    preview: Option<DocState>,
    actor: ActorId,
    clock: u64,
    next_layer: u64,
}

impl Engine {
    /// Build an engine that presents to `target_format` (a surface format, or a
    /// test target). Takes wgpu handles per GOALS §Inputs.
    pub fn new(gpu: GpuContext, target_format: wgpu::TextureFormat, viewport: Extent2) -> Self {
        let pool = TilePool::new(gpu.clone());
        let stroke = StrokeRenderer::new(&gpu);
        let presenter = Presenter::new(&gpu, target_format);

        let initial = DocState::with_layer(ROOT_LAYER);
        let timeline: Box<dyn Timeline> = Box::new(LinearTimeline::new(initial));
        let session = crate::session::Session::new(ViewTransform::identity(viewport), ROOT_LAYER);

        Self {
            gpu,
            pool,
            stroke,
            presenter,
            timeline,
            session,
            preview: None,
            actor: ActorId(0),
            clock: 0,
            next_layer: 1,
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
                let mut ctx = self.apply_ctx();
                self.timeline.undo(&mut ctx);
            }
            InputCommand::Redo => {
                self.preview = None;
                let mut ctx = self.apply_ctx();
                self.timeline.redo(&mut ctx);
            }
            InputCommand::SetTool(tool) => self.session.tool = tool,
            InputCommand::SetBrush(brush) => {
                self.session.brush = brush;
                self.refresh_preview();
            }
            InputCommand::Pan { delta } => {
                // Drag the canvas under a fixed viewport.
                self.session.view.center -= delta / self.session.view.zoom;
            }
            InputCommand::Zoom { factor, .. } => {
                self.session.view.zoom = (self.session.view.zoom * factor).max(1e-3);
            }
            InputCommand::AddLayer { above } => {
                let id = LayerId(self.next_layer);
                self.next_layer += 1;
                self.commit(ActionKind::AddLayer { id, above });
            }
            InputCommand::RemoveLayer(id) => self.commit(ActionKind::RemoveLayer(id)),
            InputCommand::SetLayerBlend(id, blend) => {
                self.commit(ActionKind::SetLayerBlend(id, blend))
            }
        }
    }

    /// Render the current canvas (preview if stroking, else committed) into
    /// `target`, clearing to `background` first (DESIGN.md §6.4).
    pub fn render(&mut self, target: &wgpu::TextureView, background: wgpu::Color) {
        let doc = self.preview.as_ref().unwrap_or_else(|| self.timeline.current());

        // Gather populated tiles bottom-to-top. True per-blend-mode compositing
        // is step 4; for now layers stack with premultiplied "over".
        let mut tiles: Vec<(crate::geom::TileCoord, TileHandle)> = Vec::new();
        for layer in doc.layers.iter() {
            for (coord, handle) in layer.tiles.iter() {
                tiles.push((*coord, handle.clone()));
            }
        }

        let view = self.session.view;
        self.presenter.render(target, view, background, &tiles);
    }

    /// A snapshot of UI-facing state (DESIGN.md §7).
    pub fn observe(&self) -> ObservableState {
        ObservableState {
            can_undo: self.timeline.can_undo(),
            can_redo: self.timeline.can_redo(),
            is_stroking: self.session.is_stroking(),
            tool: self.session.tool,
            brush: self.session.brush,
            view: self.session.view,
            bounds: self.timeline.current().bounds,
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

    fn commit(&mut self, kind: ActionKind) {
        let action = Action {
            id: self.next_action_id(),
            kind,
        };
        let mut ctx = self.apply_ctx();
        self.timeline.push(action, &mut ctx);
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
        let tiles = self.stroke.render(&self.pool, &layer.tiles, rec);
        base.with_layer_at(idx, Layer { tiles, ..layer })
    }
}

/// Convenience for tests/tools: build an engine on a headless device.
pub async fn headless_engine(
    target_format: wgpu::TextureFormat,
    viewport: Extent2,
) -> Result<Engine> {
    let gpu = GpuContext::headless().await?;
    Ok(Engine::new(gpu, target_format, viewport))
}
