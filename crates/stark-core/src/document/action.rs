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

/// What sets the brush shape's orientation as it sweeps along the stroke (DESIGN.md
/// §6.6). The swept-depth integral runs along the stroke's travel direction, so the
/// shape is looked up in a per-orientation prefix-τ texture indexed by the *relative*
/// angle between the shape's native axis and the travel direction.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OrientationSource {
    /// The shape's native axis tracks the stroke tangent — the relative angle is always
    /// 0, so the footprint always faces along the motion (the historical behaviour).
    #[default]
    FollowStroke,
    /// The shape stays pinned to the pen's orientation (the tilt azimuth) in canvas
    /// space; as the stroke curves under a fixed pen the footprint angle stays put,
    /// like a calligraphy nib.
    Pen,
}

/// How a brush interacts with paint already on the canvas (DESIGN.md §6.2,
/// "Wet mixing & brush dynamics"). The pluggable fidelity axis.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum BrushDynamics {
    /// The unified paint-manipulation brush, driven by an explicit **tool reservoir**:
    /// **lift** picks canvas paint up onto the tool, **deposit** lays tool paint back
    /// down, and **add** lays the brush's own paint. With `add=0` paint is conserved —
    /// it only *moves* between canvas and tool — so dragging from a painted region onto
    /// bare canvas asymptotically empties the tool there. Spans erasing
    /// (`lift=1, deposit=0`), smudging (`lift+deposit`), painting (`add`), and every
    /// blend; the everyday dry brush is just `add=1, lift=0, deposit=0` (the default)
    /// (DESIGN.md §6.2).
    Dry(DryParams),
    /// Wet paint that simultaneously **bleeds** (isotropic wet-on-wet diffusion) and
    /// **drags** (advection along the stroke's motion) — both run over the stroke
    /// region, independently configurable (DESIGN.md §6.2).
    Wet(WetParams),
}

impl Default for BrushDynamics {
    /// The everyday brush: add the brush's own paint, no smear/remove.
    fn default() -> Self {
        BrushDynamics::Dry(DryParams::default())
    }
}

/// Parameters of the unified [`BrushDynamics::Dry`] brush (DESIGN.md §6.2), driven by a
/// **tool reservoir** that fills (lift) and empties (deposit). The three axes compose
/// freely: `lift`-only = eraser, `lift`+`deposit` (`add=0`) = a conservative smudge that
/// moves paint without creating or destroying it, `add`-only = ordinary paint, and every
/// blend between.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DryParams {
    /// How much of the brush's own paint it **adds** directly to the canvas per step, in
    /// [0, 1]: 0 = lays none (pure lift/deposit), 1 = a full deposit (ordinary painting).
    /// Bypasses the tool reservoir.
    pub add: f32,
    /// How much canvas paint the brush **lifts** onto the tool per step, as a fraction of
    /// the paint present, in [0, 1]: 0 = none, 1 = lift it all (scrape clean). Tooth-gated
    /// by the canvas surface. `#[serde(default)]` so pre-rewrite documents load.
    #[serde(default)]
    pub lift: f32,
    /// How much tool paint the brush **deposits** back onto the canvas per step, as a
    /// fraction of the paint on the tool, in [0, 1]: 0 = hold it all (an eraser fills but
    /// never lays back), 1 = lay it all immediately. `#[serde(default)]`.
    #[serde(default)]
    pub deposit: f32,
    /// How strongly displaced paint **piles into ridges** at the edges of the stroke,
    /// in [0, 1] — the impasto lip (DESIGN.md §6.2, lateral pile-up). `#[serde(default)]`.
    #[serde(default)]
    pub ridge: f32,
}

impl Default for DryParams {
    fn default() -> Self {
        Self { add: 1.0, lift: 0.0, deposit: 0.0, ridge: 0.0 }
    }
}

/// Parameters of the [`BrushDynamics::Wet`] flow (DESIGN.md §6.2): wet paint that
/// **adds** the brush's own paint, then simultaneously bleeds and drags it. All three
/// are independent; 0 disables that behaviour (`add`-only = plain wet paint,
/// `bleed`-only = alla-prima diffusion of what's already there, `drag`-only = a fluid
/// rake). The iteration count and per-step distances are fixed (for deterministic
/// replay).
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WetParams {
    /// How much of the brush's own paint it **adds**, in [0, 1] — the same axis as
    /// [`DryParams::add`]: 0 lays nothing (the stroke only bleeds/drags paint already
    /// on the canvas), 1 = a full deposit. Defaults to 1 on load so documents saved
    /// before this field keep their always-deposit behaviour.
    #[serde(default = "default_wet_add")]
    pub add: f32,
    /// How strongly the wet paint bleeds and levels, in [0, 1]: 0 = none, 1 = maximum
    /// bleed. Scales the per-iteration diffusion rate.
    pub bleed: f32,
    /// How strongly the brush drags wet paint along its motion, in [0, 1]: 0 = none,
    /// 1 = maximum drag. Scales the injected velocity. `#[serde(default)]` so the
    /// pre-unification `Wet` strokes load (drag 0 = pure bleed).
    #[serde(default)]
    pub drag: f32,
}

fn default_wet_add() -> f32 {
    1.0
}

impl Default for WetParams {
    /// Mostly deposit with a gentle bleed and a whisper of drag. Normalized to sum
    /// to 1, matching the UI's barycentric control — the axes' common scale is
    /// redundant with the stroke's rate, so the mix is what matters (DESIGN.md §6.2).
    fn default() -> Self {
        Self { add: 0.6, bleed: 0.3, drag: 0.1 }
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
    /// What orients the shape as it sweeps (DESIGN.md §6.6) — the successor to the old
    /// `follow_path`/`angle_jitter` knobs: `FollowStroke` is the former `follow_path =
    /// true`. `#[serde(default)]` so documents saved before this field (which instead
    /// carried `follow_path`, now ignored on load) come in as `FollowStroke`.
    #[serde(default)]
    pub orientation: OrientationSource,
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
            orientation: OrientationSource::default(),
            dynamics: BrushDynamics::default(),
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

