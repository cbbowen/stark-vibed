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

/// How a brush interacts with paint already on the canvas (DESIGN.md §6.2). One
/// **unified tool**, not a mode switch: every axis is a flux on the single conserved
/// quantity — paint **height** (the amount; DESIGN §6.1) — and the axes compose freely.
/// `add` is the only *source* (the brush's own paint); the rest move paint that is
/// already on the canvas, so with `add = 0` the tool conserves height (it only moves
/// paint around). The everyday dry brush is just `add = 1` with the rest 0 (the default).
///
/// Two axes are **vertical** flux between the canvas and a transient per-stroke *tool*
/// reservoir — Lagrangian, giving crisp long-range *directed* transport:
/// - [`load`](Self::load)    — lift canvas paint up onto the tool,
/// - [`deposit`](Self::deposit) — lay tool paint back down.
///
/// Three are **horizontal** flux across the canvas — Eulerian, giving local
/// omnidirectional flow over a composited stroke region:
/// - [`drag`](Self::drag)  — advect paint along the brush's motion (conservative
///   finite-volume; the velocity is injected from the stroke and de-rippled),
/// - [`bleed`](Self::bleed) — isotropic wet-on-wet diffusion / leveling,
/// - [`ridge`](Self::ridge) — pile displaced paint into impasto lips at the edges.
///
/// `load`-only is an eraser; `load`+`deposit` (`add = 0`) a conservative smudge;
/// `add`-only ordinary paint; `drag`/`bleed` a rake / alla-prima blender. All flow runs
/// with fixed iteration counts, so replay stays deterministic (DESIGN §6.2).
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BrushDynamics {
    /// The brush's own paint laid directly, in [0, 1]: 0 = lays none (pure manipulation of
    /// existing paint), 1 = a full deposit (ordinary painting). The only source term.
    pub add: f32,
    /// Canvas paint **lifted** onto the tool per step, as a fraction of the paint present,
    /// in [0, 1]: 0 = none, 1 = lift it all (scrape clean). Vertical flux canvas → tool.
    #[serde(default)]
    pub lift: f32,
    /// Tool paint **deposited** back per step, as a fraction of the paint on the tool, in
    /// [0, 1]: 0 = hold it all (an eraser fills but never lays back), 1 = lay it all
    /// immediately. Vertical flux tool → canvas.
    #[serde(default)]
    pub deposit: f32,
    /// Strength of **drag** (advection of paint along the stroke's motion), in [0, 1]:
    /// 0 = none, 1 = maximum. Horizontal flux; scales the injected velocity.
    #[serde(default)]
    pub drag: f32,
    /// Strength of **bleed** (isotropic wet-on-wet diffusion), in [0, 1]: 0 = none, 1 =
    /// maximum. Horizontal flux; scales the per-stroke Gaussian radius.
    #[serde(default)]
    pub bleed: f32,
    /// How strongly displaced paint **piles into ridges** at the footprint edges, in
    /// [0, 1] — the impasto lip (DESIGN.md §6.2). A conservative lateral redistribution.
    #[serde(default)]
    pub ridge: f32,
    /// Initial paint **pre-loaded onto the tool** reservoir before the stroke starts, as a
    /// height (the "load a glob on the palette knife" param). 0 = the tool starts empty (the
    /// historical behaviour). It depletes as the tool [`deposit`](Self::deposit)s and refills
    /// as it [`load`](Self::load)s — a finite carried amount, unlike the inexhaustible
    /// [`add`](Self::add) source (DESIGN.md §6.2).
    #[serde(default)]
    pub charge: f32,
    /// How strongly **pen pressure** modulates the scrape ([`load`](Self::load)), in [0, 1]:
    /// 0 = `load` is constant across the stroke (the historical behaviour), 1 = `load` scales
    /// fully with per-sample pressure (a palette knife scrapes more the harder you press).
    #[serde(default)]
    pub load_pressure: f32,
    /// How strongly **pen tilt toward the direction of motion** modulates the
    /// [`deposit`](Self::deposit), in [0, 1]: 0 = `deposit` is constant (the historical
    /// behaviour, and the fallback with no pen tilt), 1 = `deposit` scales fully with the
    /// forward lean (tilting the knife into the stroke lays more paint down).
    #[serde(default)]
    pub deposit_tilt: f32,
}

impl Default for BrushDynamics {
    /// The everyday brush: lay the brush's own paint, manipulate nothing.
    fn default() -> Self {
        Self {
            add: 1.0,
            lift: 0.0,
            deposit: 0.0,
            drag: 0.0,
            bleed: 0.0,
            ridge: 0.0,
            charge: 0.0,
            load_pressure: 0.0,
            deposit_tilt: 0.0,
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
    /// What orients the shape as it sweeps (DESIGN.md §6.6) — the successor to the old
    /// `follow_path`/`angle_jitter` knobs: `FollowStroke` is the former `follow_path =
    /// true`. `#[serde(default)]` so documents saved before this field (which instead
    /// carried `follow_path`, now ignored on load) come in as `FollowStroke`.
    #[serde(default)]
    pub orientation: OrientationSource,
    /// How the brush manipulates paint already on the canvas (DESIGN.md §6.2) — the
    /// unified six-axis tool. `#[serde(default)]` so documents saved before this field
    /// load as the everyday `add = 1` brush.
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

