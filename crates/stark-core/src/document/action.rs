//! Actions: committed, deterministic, replayable document mutations (DESIGN.md §4).
//!
//! An [`Action`] is the unit the timeline stores/replays and (later) the unit
//! serialized to disk. Every action carries a globally-unique [`ActionId`] so
//! the same records work unchanged in a future replicated, multi-peer log
//! (DESIGN.md §4, §12) — we pay that tiny cost from the first commit.

use serde::{Deserialize, Serialize};

use super::layer::{BlendMode, Layer, LayerId};
use super::selection::SelectionOp;
use super::state::DocState;
use crate::gpu::SurfaceId;
use crate::gpu::selection::SelectionRenderer;
use crate::gpu::stroke::StrokeRenderer;
use crate::gpu::tile::TilePool;

/// Identifies the author of an action: one local user, or a peer (DESIGN.md §4).
/// Maps to an iroh `NodeId` when collaborating; a fixed value when solo.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ActorId(pub u64);

impl ActorId {
    /// The fixed author id used when not collaborating. When a document is
    /// first shared, its solo-authored actions are rewritten to the sharer's
    /// real actor id (so the sharer can still undo them, DESIGN.md §12.3);
    /// after that every action in a shared log carries a peer-derived id.
    pub const SOLO: ActorId = ActorId(0);
}

/// Globally-unique action id; also the total order key `(lamport, actor)`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ActionId {
    pub lamport: u64,
    pub actor: ActorId,
}

/// The tool a gesture drives. Tools become an open registry later (DESIGN.md §10).
///
/// Only [`Brush`](Self::Brush) ever reaches a [`StrokeRecord`]: the selection tools
/// produce a [`SelectionOp`] instead of a stroke (DESIGN.md §6.8). They share the
/// enum — and so the pointer-gesture plumbing — because from the frontend's point of
/// view they are the same interaction: press, drag, release.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tool {
    #[default]
    Brush,
    /// Rectangular marquee.
    SelectRect,
    /// Elliptical marquee.
    SelectEllipse,
    /// Freehand lasso.
    SelectLasso,
}

impl Tool {
    /// Whether this tool edits the selection rather than the paint.
    pub fn is_selection(self) -> bool {
        matches!(
            self,
            Tool::SelectRect | Tool::SelectEllipse | Tool::SelectLasso
        )
    }
}

/// The brush tip shape (DESIGN.md §6.6).
#[derive(Copy, Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
pub enum BrushShape {
    /// Procedural soft disc; `hardness` controls the falloff.
    #[default]
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
/// [`add`](Self::add) is the only *source* (the brush's own paint); the rest move paint
/// that is already on the canvas, so with `add = 0` the tool conserves height (it only
/// moves paint around). The everyday brush is just `add` with the rest 0 (the default).
///
/// The two remaining axes are **vertical** flux between the canvas and a transient
/// per-stroke *tool* reservoir — Lagrangian, giving crisp long-range *directed*
/// transport:
/// - [`lift`](Self::lift)       — lift canvas paint up onto the tool,
/// - [`deposit`](Self::deposit) — lay tool paint back down.
///
/// `lift`-only is an eraser; `lift`+`deposit` (`add = 0`) a conservative smudge;
/// `add`-only ordinary paint. All flow runs with fixed iteration counts, so replay
/// stays deterministic (DESIGN §6.2).
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BrushDynamics {
    /// The brush's own paint laid directly: the paint **height** deposited per unit of
    /// swept optical depth (DESIGN.md §6.1), and the tool's only source term. 0 = lays
    /// none (pure manipulation of existing paint), 1 = a heavy full-thickness deposit.
    ///
    /// A *rate*, not a quantity — this source never runs out on its own. For a stroke
    /// that runs dry as it travels see [`BrushParams::drain`]; for a finite carried
    /// glob that depletes as it is laid see [`charge`](Self::charge).
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
    /// Initial paint **pre-loaded onto the tool** reservoir before the stroke starts, as a
    /// height (the "load a glob on the palette knife" param). 0 = the tool starts empty (the
    /// historical behaviour). It depletes as the tool [`deposit`](Self::deposit)s and refills
    /// as it [`lift`](Self::lift)s — a finite carried amount, unlike the inexhaustible
    /// [`add`](Self::add) source (DESIGN.md §6.2).
    #[serde(default)]
    pub charge: f32,
}

impl Default for BrushDynamics {
    /// The everyday brush: lay the brush's own paint, manipulate nothing.
    fn default() -> Self {
        Self {
            add: 0.6,
            lift: 0.0,
            deposit: 0.0,
            charge: 0.0,
        }
    }
}

/// The kind of noise field driving [`ColorDynamics`] (DESIGN.md §6.2). Each kind
/// is baked once into a small tileable 3-D texture (`noise.rs`), so lookups are
/// cheap and deterministic across replay, peers, and builds.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum NoiseKind {
    /// Uncorrelated per-texel randomness — grainy speckle.
    White,
    /// Smooth organic gradient noise (a seamlessly tiling simplex-class noise) —
    /// soft, flowing variation.
    #[default]
    Simplex,
}

/// Colour dynamics (colour jitter): lets the applied colour vary **across the
/// brush and along the stroke** (DESIGN.md §6.2). A 3-channel tileable 3-D noise
/// field is sampled at `(canvas.x, canvas.y, arc length)` — the two canvas axes
/// give spatial variation across the footprint (and keep tile aprons consistent,
/// §6.4: the offset is a pure function of canvas position + the stroke), the
/// third evolves the colour along the stroke — and the three noise channels
/// offset the three colour channels *of the current colour space* (Oklab
/// `L, a, b`; Mixbox pigment concentrations). The per-stroke `seed` translates
/// the lookup so each stroke draws a fresh part of the field, deterministically.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ColorDynamics {
    /// Which noise field to sample.
    pub noise: NoiseKind,
    /// Frequency scale per lookup axis (canvas x, canvas y, arc length): 1 = one
    /// noise tile per [`crate::noise::NOISE_TILE_PX`] canvas px; higher = finer
    /// variation along that axis; 0 = constant along that axis.
    pub frequency: [f32; 3],
    /// Noise amplitude per colour channel, in the colour space's own units
    /// (noise is signed, so a channel wanders ±amplitude). All 0 = off — the
    /// exact historical constant-colour deposit.
    pub amplitude: [f32; 3],
}

impl Default for ColorDynamics {
    fn default() -> Self {
        Self {
            noise: NoiseKind::default(),
            frequency: [1.0; 3],
            amplitude: [0.0; 3],
        }
    }
}

impl ColorDynamics {
    /// Whether the jitter has any effect (any channel amplitude non-zero).
    pub fn is_active(&self) -> bool {
        self.amplitude.iter().any(|a| *a != 0.0)
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
    /// Edge softness in [0, 1): 0 = very soft, ~1 = hard edge.
    pub hardness: f32,
    /// Reservoir depletion per canvas pixel travelled: the stroke thins as paint
    /// runs out (DESIGN.md §6.2). 0 = inexhaustible — which is what a pen, a
    /// charcoal stick, or an ordinary digital brush wants; a physical loaded
    /// brush wants a small positive value.
    pub drain: f32,
    /// Brush tip shape (DESIGN.md §6.6).
    pub shape: BrushShape,
    /// What orients the shape as it sweeps (DESIGN.md §6.6) — the successor to the old
    /// `follow_path`/`angle_jitter` knobs: `FollowStroke` is the former `follow_path =
    /// true`. `#[serde(default)]` so documents saved before this field (which instead
    /// carried `follow_path`, now ignored on load) come in as `FollowStroke`.
    #[serde(default)]
    pub orientation: OrientationSource,
    /// How much of its own paint the brush lays, and how it manipulates paint already
    /// on the canvas (DESIGN.md §6.2) — the unified tool. `#[serde(default)]` so
    /// documents saved before this field load as the everyday `add`-only brush.
    #[serde(default)]
    pub dynamics: BrushDynamics,
    /// Canvas **tooth** in [0, 1]: how strongly the surface bump (DESIGN.md §6.4)
    /// gates deposition — dry/light strokes catch on the weave's peaks and skip
    /// its valleys, fading as coverage builds. Historized (it changes stored
    /// pixels) so replay stays deterministic; `#[serde(default)]` (0 = no tooth)
    /// preserves the look of documents saved before it existed.
    #[serde(default)]
    pub tooth: f32,
    /// Colour dynamics (colour jitter) — how the applied colour varies across the
    /// brush and along the stroke (DESIGN.md §6.2). Historized (it changes stored
    /// pixels); the default (amplitude 0) is the historical constant colour.
    #[serde(default)]
    pub color_dynamics: ColorDynamics,
}

impl Default for BrushParams {
    fn default() -> Self {
        Self {
            color: [0.0, 0.0, 0.0, 1.0],
            radius: 16.0,
            hardness: 0.5,
            drain: 0.0015,
            shape: BrushShape::default(),
            orientation: OrientationSource::default(),
            dynamics: BrushDynamics::default(),
            tooth: 0.5,
            color_dynamics: ColorDynamics::default(),
        }
    }
}

/// A fully-recorded stroke: enough to replay it bit-for-bit (DESIGN.md §4).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StrokeRecord {
    pub layer: LayerId,
    pub tool: Tool,
    pub brush: BrushParams,
    /// The fitted stroke curve: the control points the raw pointer samples were
    /// smoothed and simplified down to (DESIGN.md §6.2), an order of magnitude
    /// fewer points and all that is needed to reconstruct the stroke. The raw
    /// samples are never stored — not in the file, not in the action log, not
    /// on the wire.
    pub path: Vec<crate::path::ControlPoint>,
    /// Seed for any brush jitter, making replay reproducible. Unused by the MVP
    /// brush but recorded so the format is stable.
    pub seed: u64,
}

/// What an action does to the document.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ActionKind {
    CommitStroke(StrokeRecord),
    AddLayer {
        id: LayerId,
        above: Option<LayerId>,
    },
    RemoveLayer(LayerId),
    SetLayerBlend(LayerId, BlendMode),
    SetLayerOpacity(LayerId, f32),
    SetLayerVisible(LayerId, bool),
    MoveLayer {
        id: LayerId,
        above: Option<LayerId>,
    },
    /// Undo **as a logged action** (DESIGN.md §5.4, §12.3): a fact peers can see
    /// and order, meaning "derive the document as if `target` were absent".
    /// Redo is an `Undo` of an `Undo`. Emitted only in shared sessions; solo
    /// undo stays pure timeline navigation and never logs one.
    ///
    /// Deliberately **not interpreted by [`Action::apply`]** — undo needs the
    /// whole log, not just the prior state, so the timeline layer resolves
    /// which actions are *effective* (see [`super::timeline::effective_actions`])
    /// and only ever materializes those. Appended last so postcard decoding of
    /// older files is unaffected.
    Undo(ActionId),

    /// Switch the canvas surface (DESIGN.md §6.4).
    ///
    /// Logged rather than kept as a view setting because the surface feeds the
    /// deposition tooth gate: what a stroke lays down depends on the surface in
    /// force when it was drawn, so replay must reconstruct it. Appended last so
    /// postcard decoding of older files is unaffected; documents saved before this
    /// existed simply never contain one and keep the surface from `CanvasMeta`.
    SetSurface(SurfaceId),
    /// Edit the selection mask (DESIGN.md §6.8). Historized because a stroke's
    /// pixels depend on the mask in force when it was drawn — replaying the log has
    /// to put the same mask back. Only the **op** travels (a few floats, or a
    /// decimated polyline); every peer rasterizes it identically from the same
    /// shader, so the log stays compact and convergence is unaffected.
    Select(SelectionOp),
    /// Swap selected for unselected everywhere (DESIGN.md §6.8).
    InvertSelection,
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
    pub selection: SelectionRenderer,
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
                    // The selection in force *at this point in the log* gates the
                    // stroke (DESIGN.md §6.8) — it is read from the state being
                    // folded over, so replay reproduces it exactly.
                    let tiles = ctx.stroke.render(
                        crate::gpu::stroke::StrokeScene {
                            pool: &ctx.pool,
                            assets: &ctx.assets,
                            base: &layer.tiles,
                            selection: &state.selection,
                        },
                        rec,
                    );
                    state.with_layer_at(
                        idx,
                        Layer {
                            tiles,
                            ..layer.clone()
                        },
                    )
                }
                None => state,
            },
            ActionKind::AddLayer { id, above } => state.insert_layer(*id, *above),
            ActionKind::RemoveLayer(id) => state.remove_layer(*id),
            ActionKind::SetLayerBlend(id, blend) => state.set_layer_blend(*id, *blend),
            ActionKind::SetLayerOpacity(id, opacity) => state.set_layer_opacity(*id, *opacity),
            ActionKind::SetLayerVisible(id, visible) => state.set_layer_visible(*id, *visible),
            ActionKind::MoveLayer { id, above } => state.move_layer(*id, *above),
            // Resolved at the timeline layer (effective-sequence filtering); an
            // `Undo` should never be materialized through `apply`. Identity, so
            // a stray one is harmless rather than wrong.
            ActionKind::Undo(_) => state,
            // An op too large to rasterize (see `MAX_SELECTION_TILES`) leaves the
            // selection alone — deterministically, since the bound is a pure
            // function of the op, so peers and replays agree.
            ActionKind::Select(op) => match ctx.selection.apply(&ctx.pool, &state.selection, op) {
                Some(selection) => state.with_selection(selection),
                None => {
                    tracing::warn!("selection op too large to rasterize; ignored");
                    state
                }
            },
            ActionKind::InvertSelection => {
                let selection = ctx.selection.invert(&ctx.pool, &state.selection);
                state.with_selection(selection)
            }
            ActionKind::SetSurface(id) => state.with_surface(*id),
        })
    }
}
