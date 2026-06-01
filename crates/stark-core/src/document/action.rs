//! Actions: committed, deterministic, replayable document mutations (DESIGN.md §4).
//!
//! An [`Action`] is the unit the timeline stores/replays and (later) the unit
//! serialized to disk. Every action carries a globally-unique [`ActionId`] so
//! the same records work unchanged in a future replicated, multi-peer log
//! (DESIGN.md §4, §12) — we pay that tiny cost from the first commit.

use serde::{Deserialize, Serialize};

use super::layer::{BlendMode, Layer, LayerId};
use super::state::DocState;
use crate::command::InputSample;
use crate::gpu::stroke::StrokeRenderer;
use crate::gpu::tile::TilePool;

/// Identifies the author of an action: one local user, or a peer (DESIGN.md §4).
/// Maps to an iroh `NodeId` when collaborating; a fixed value when solo.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ActorId(pub u64);

/// Globally-unique action id; also the total order key `(lamport, actor)`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ActionId {
    pub lamport: u64,
    pub actor: ActorId,
}

/// The painting tool that produced a stroke. A single brush for now; tools
/// become an open registry later (DESIGN.md §10).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tool {
    Brush,
}

/// The brush tip shape (DESIGN.md §6.6).
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum BrushShape {
    /// Procedural soft disc; `hardness` controls the falloff.
    Round,
    /// A sampled coverage mask, referenced by content id (an imported image).
    Stamp(crate::assets::AssetId),
}

/// How a brush interacts with paint already on the canvas (DESIGN.md §6.2,
/// "Wet mixing & brush dynamics"). This is the pluggable fidelity axis: `Dry` is
/// the one-way model; `Mixer` smears existing wet paint; future `Bleed`/`Fluid`
/// tiers slot in here.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub enum BrushDynamics {
    /// No canvas pickup — paint flows one way and thins via `drain`. The default,
    /// so existing strokes, goldens, and saved files are unchanged.
    #[default]
    Dry,
    /// Mixer/smudge: lift wet paint under the brush into a reservoir and carry it,
    /// so the stroke smears what is already on the canvas (DESIGN.md §6.2).
    Mixer(MixerParams),
}

/// Parameters of the [`BrushDynamics::Mixer`] reservoir (DESIGN.md §6.2).
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MixerParams {
    /// Fraction of the wet canvas paint pulled into the reservoir per unit step
    /// (scaled by canvas wetness and how much paint is present). 0 = no pickup.
    pub pickup: f32,
    /// How much of the brush's own color is injected as it deposits: 0 = pure
    /// smudge (only carries picked-up paint), 1 = always lays its own color.
    pub color_inject: f32,
}

impl Default for MixerParams {
    fn default() -> Self {
        Self {
            pickup: 0.5,
            color_inject: 0.1,
        }
    }
}

/// Brush configuration. `color` is straight **sRGB** RGBA; it is converted to
/// the Oklab working space at stamp time (DESIGN.md §6.5).
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BrushParams {
    /// Straight (un-premultiplied) sRGB RGBA, components in [0, 1].
    pub color: [f32; 4],
    /// Stamp radius in canvas pixels at full pressure.
    pub radius: f32,
    /// Spacing between stamps as a fraction of `radius`.
    pub spacing: f32,
    /// Edge softness in [0, 1): 0 = very soft, ~1 = hard edge.
    pub hardness: f32,
    /// Per-stamp coverage in [0, 1].
    pub flow: f32,
    /// Impasto: paint thickness deposited per unit coverage (height channel).
    pub height: f32,
    /// Wetness deposited per unit coverage (wet channel) — drives gloss (§6.3).
    pub wetness: f32,
    /// Reservoir depletion per canvas pixel travelled: the stroke thins as paint
    /// runs out (DESIGN.md §6.2). 0 = inexhaustible.
    pub drain: f32,
    /// Brush tip shape (DESIGN.md §6.6).
    pub shape: BrushShape,
    /// Rotate each stamp to the stroke tangent (organic, directional strokes).
    pub follow_path: bool,
    /// Random per-stamp rotation in radians (seeded; 0 = none).
    pub angle_jitter: f32,
    /// Canvas-pickup behavior (DESIGN.md §6.2). `#[serde(default)]` so documents
    /// saved before this field load as `Dry`.
    #[serde(default)]
    pub dynamics: BrushDynamics,
    /// Canvas **tooth** in [0, 1]: how strongly the surface bump (DESIGN.md §6.4)
    /// gates deposition — dry/light strokes catch on the weave's peaks and skip
    /// its valleys, fading as coverage builds. Historized (it changes stored
    /// pixels) so replay stays deterministic; `#[serde(default)]` (0 = no tooth)
    /// preserves the look of documents saved before it existed.
    #[serde(default)]
    pub tooth: f32,
}

impl Default for BrushParams {
    fn default() -> Self {
        Self {
            color: [0.0, 0.0, 0.0, 1.0],
            radius: 16.0,
            spacing: 0.25,
            hardness: 0.5,
            flow: 1.0,
            height: 0.6,
            wetness: 0.7,
            drain: 0.0015,
            shape: BrushShape::Round,
            follow_path: true,
            angle_jitter: 0.0,
            dynamics: BrushDynamics::Dry,
            tooth: 0.5,
        }
    }
}

/// A fully-recorded stroke: enough to replay it bit-for-bit (DESIGN.md §4).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StrokeRecord {
    pub layer: LayerId,
    pub tool: Tool,
    pub brush: BrushParams,
    /// Resampled input path, full fidelity.
    pub path: Vec<InputSample>,
    /// Seed for any brush jitter, making replay reproducible. Unused by the MVP
    /// brush but recorded so the format is stable.
    pub seed: u64,
}

/// What an action does to the document.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ActionKind {
    CommitStroke(StrokeRecord),
    AddLayer { id: LayerId, above: Option<LayerId> },
    RemoveLayer(LayerId),
    SetLayerBlend(LayerId, BlendMode),
    SetLayerOpacity(LayerId, f32),
    SetLayerVisible(LayerId, bool),
    MoveLayer { id: LayerId, above: Option<LayerId> },
    // `Undo(ActionId)` (undo-as-an-action) arrives with the replicated timeline
    // in step 7 (DESIGN.md §5.4, §12); single-user undo uses timeline navigation.
}

/// A committed document mutation with its identity.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Action {
    pub id: ActionId,
    pub kind: ActionKind,
}

/// Side-channel passed to [`history::Action::apply`]: the GPU resources needed
/// to render a stroke (DESIGN.md §5). It owns cheap `Arc`-backed clones, so it
/// has no borrow lifetime — which is what lets it be the `Action::Context`.
#[derive(Clone)]
pub struct ApplyCtx {
    pub pool: TilePool,
    pub stroke: StrokeRenderer,
    pub assets: crate::assets::AssetStore,
}

impl history::Action for Action {
    type State = DocState;
    type Context = ApplyCtx;
    // GPU work reports failure via wgpu's device error callbacks, not return
    // values, and tile allocation never fails — so applying an action is
    // genuinely infallible here (DESIGN.md §5).
    type Error = std::convert::Infallible;

    fn apply(&self, state: DocState, ctx: &mut ApplyCtx) -> Result<DocState, Self::Error> {
        Ok(match &self.kind {
            ActionKind::CommitStroke(rec) => match state.layer_index(rec.layer) {
                Some(idx) => {
                    let layer = state.layer_at(idx);
                    let tiles = ctx.stroke.render(&ctx.pool, &ctx.assets, &layer.tiles, rec);
                    state.with_layer_at(idx, Layer { tiles, ..layer.clone() })
                }
                None => state,
            },
            ActionKind::AddLayer { id, above } => state.insert_layer(*id, *above),
            ActionKind::RemoveLayer(id) => state.remove_layer(*id),
            ActionKind::SetLayerBlend(id, blend) => state.set_layer_blend(*id, *blend),
            ActionKind::SetLayerOpacity(id, opacity) => state.set_layer_opacity(*id, *opacity),
            ActionKind::SetLayerVisible(id, visible) => state.set_layer_visible(*id, *visible),
            ActionKind::MoveLayer { id, above } => state.move_layer(*id, *above),
        })
    }
}

