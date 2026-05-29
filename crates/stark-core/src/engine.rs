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
use crate::gpu::{Compositor, GpuContext, StrokeRenderer, TileHandle, TilePool};
use crate::image::RgbaImage;
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
    target_format: wgpu::TextureFormat,
    pool: TilePool,
    stroke: StrokeRenderer,
    compositor: Compositor,
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
        let compositor = Compositor::new(&gpu, target_format, viewport);

        let initial = DocState::with_layer(ROOT_LAYER);
        let timeline: Box<dyn Timeline> = Box::new(LinearTimeline::new(initial));
        let session = crate::session::Session::new(ViewTransform::identity(viewport), ROOT_LAYER);

        Self {
            gpu,
            target_format,
            pool,
            stroke,
            compositor,
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
        self.compositor.render(target, view, background, &tiles);
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

    /// Tune the media/lighting parameters of the painterly pass (DESIGN.md §6.3).
    pub fn set_media_params(&mut self, params: crate::gpu::MediaParams) {
        self.compositor.set_media(params);
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
