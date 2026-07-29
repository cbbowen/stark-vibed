//! Actions: committed, deterministic, replayable document mutations (DESIGN.md §4).
//!
//! An [`Action`] is the unit the timeline stores/replays and (later) the unit
//! serialized to disk. Every action carries a globally-unique [`ActionId`] so
//! the same records work unchanged in a future replicated, multi-peer log
//! (DESIGN.md §4, §12) — we pay that tiny cost from the first commit.

use serde::{Deserialize, Serialize};

use super::layer::{BlendMode, LayerId, MatteRegion};
use super::selection::SelectionOp;
use super::state::DocState;
use crate::geom::Vec2;
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
/// is baked once into a small tileable 2-D texture (`noise.rs`), so lookups are
/// cheap and deterministic across replay, peers, and builds.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum NoiseKind {
    /// Uncorrelated per-texel randomness — grainy speckle.
    White,
    /// Smooth organic gradient noise (a seamlessly tiling simplex-class noise) —
    /// soft, flowing variation.
    #[default]
    Simplex,
    /// Cellular (Worley F1) noise on a seamlessly tiling jittered grid — mottled
    /// patches with creases where cells meet, like pigment settling in clumps.
    Voronoi,
    /// The discrete form of [`Self::Voronoi`]: each cell one flat colour offset,
    /// with a hard edge to its neighbours — crystalline facets rather than a
    /// gradient. All three channels share the same cells, so the facets are
    /// whole polygons of one colour.
    Mosaic,
}

/// Colour dynamics (colour jitter): lets the applied colour vary **across the
/// brush and along the stroke** (DESIGN.md §6.2). A 3-channel tileable 2-D noise
/// field is sampled in the stroke's **own** frame — `(lateral offset from the
/// centreline, arc length)`, both in canvas px — so the variation belongs to the
/// gesture rather than to the patch of canvas under it: one axis spreads the
/// colour across the footprint, the other evolves it along the stroke. The three
/// noise channels offset the three colour channels *of the current colour space*
/// (Oklab `L, a, b`; Mixbox pigment concentrations). The per-stroke `seed`
/// translates the lookup so each stroke draws a fresh part of the field,
/// deterministically.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ColorDynamics {
    /// Which noise field to sample.
    pub noise: NoiseKind,
    /// Frequency scale per lookup axis (across the stroke, along it): 1 = one
    /// noise tile per [`crate::noise::NOISE_TILE_PX`] px; higher = finer
    /// variation along that axis; 0 = constant along that axis.
    pub frequency: [f32; 2],
    /// Noise amplitude per colour channel, in the colour space's own units
    /// (noise is signed, so a channel wanders ±amplitude). All 0 = off — the
    /// exact historical constant-colour deposit.
    pub amplitude: [f32; 3],
}

impl Default for ColorDynamics {
    fn default() -> Self {
        Self {
            noise: NoiseKind::default(),
            frequency: [1.0; 2],
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
    /// Deliberately **not interpreted by [`Action`]'s `apply`** — undo needs the
    /// whole log, not just the prior state, so the timeline layer resolves
    /// which actions are *effective* (see [`super::timeline::effective_actions`])
    /// and only ever materializes those. Appended last so postcard decoding of
    /// older files is unaffected.
    Undo(ActionId),

    /// Switch the canvas surface (DESIGN.md §6.4).
    ///
    /// Logged rather than kept as a view setting because the surface feeds the
    /// document: which canvas a piece was painted on is part of what it is, and
    /// replay has to reconstruct it. Appended last so
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

    /// Add a **matte** layer — a region filled with a flat colour
    /// (FRAME_DESIGN.md §2). A frame is one of these on top of the stack; the
    /// same action serves comic gutters and opaque grounds once the region
    /// generalizes (P4). Appended last, like every variant before it, so postcard
    /// — which encodes an enum by variant *index* — keeps decoding older files.
    AddMatte {
        id: LayerId,
        above: Option<LayerId>,
        region: MatteRegion,
        /// Straight sRGB, like `BrushParams::color` — converted to working-space
        /// channels at composite time, so the log is colour-space independent.
        color: [f32; 3],
    },
    /// Move a matte's rect — the frame drag's commit. One action per drag, not
    /// per pointer move: the gesture accumulates in session state and commits on
    /// release, so fifty tweaks are fifty undo steps rather than five thousand.
    SetMatteRect(LayerId, Vec2, Vec2),
    /// Recolour a matte (straight sRGB).
    SetMatteColor(LayerId, [f32; 3]),
    /// Set the canvas substrate colour — the ground the paint sits on, straight
    /// sRGB (FRAME_DESIGN.md §5). Logged because the ground a piece was painted on
    /// is part of what it is; it was previously a view setting, so the paper colour
    /// of a painting was not saved at all.
    SetBackground([f32; 3]),

    /// Affine transform of the selected paint on `layer` (TRANSFORM_DESIGN.md):
    /// cut what the **author's** selection holds, resample it once under
    /// `affine`, stack it back over what remained — and carry the author's mask
    /// along with it, so the moved region stays selected. A universal selection
    /// moves the whole layer. Six floats in the log; every peer re-derives the
    /// same tiles from them. Appended last so postcard keeps decoding older
    /// files.
    ///
    /// Deterministically **rejected** (the document is left unchanged) when the
    /// affine is unusable or the rewrite exceeds the tile caps — see
    /// [`super::transform`].
    Transform {
        layer: LayerId,
        affine: crate::geom::Affine2,
    },

    /// Name a layer, or with `None` take its name away again so it falls back to
    /// being described by its place in the stack.
    ///
    /// Logged like every other layer property: a name is part of the document —
    /// it is saved, it is replicated, and taking one back is an undo step, which
    /// is what makes a mistyped rename recoverable the same way a mis-set opacity
    /// is. Carries a `String` rather than the `Arc<str>` the state holds, because
    /// this is the file and wire form, where a shared pointer means nothing.
    /// Appended last so postcard — which encodes an enum by variant *index* —
    /// keeps decoding older files.
    SetLayerName(LayerId, Option<String>),
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
    pub transform: crate::gpu::transform::TransformRenderer,
}

impl history::Action for Action {
    type State = DocState;
    type Context = ApplyCtx;
    // GPU work reports failure via wgpu's device error callbacks, not return
    // values, and tile allocation never fails — so applying an action is
    // genuinely infallible here (DESIGN.md §5).
    type Error = std::convert::Infallible;
    /// An action commutes with everything its [`Footprint`] is disjoint from
    /// (DESIGN.md §12.6) — which is what lets the history splice an undone
    /// action out past a peer's unrelated work instead of replaying it.
    ///
    /// [`Footprint`]: super::footprint::Footprint
    type Centralizer<'a> = super::footprint::Footprint;

    /// Remove this action's effect by restoring what it wrote from
    /// `previous_state` — the values under its footprint, nothing more, so the
    /// edits of commuting actions applied after it survive. Tiles come back as
    /// the same shared handles (copy-on-write means identity is equality), so
    /// this re-renders nothing.
    fn inverse(&self, previous_state: &DocState, state: &mut DocState) {
        *state = super::patch::unapply(self, previous_state, state);
    }

    fn apply(&self, state: DocState, ctx: &mut ApplyCtx) -> Result<DocState, Self::Error> {
        Ok(match &self.kind {
            // A matte has no tile map, so a stroke targeting one is refused
            // rather than swallowed or magically rasterized (FRAME_DESIGN.md §7).
            // Refusing here (not only in the frontend) is what keeps replay and
            // peers agreeing about a log that contains such a stroke.
            ActionKind::CommitStroke(rec) => {
                match state
                    .layer_index(rec.layer)
                    .and_then(|idx| state.layer_at(idx).tiles().map(|base| (idx, base)))
                {
                    Some((idx, base)) => {
                        // The **author's** selection, as it stood at this point in
                        // the log, gates the stroke (DESIGN.md §6.8,
                        // PEER_DESIGN.md §3). Read from the state being folded over,
                        // so replay reproduces it exactly; keyed by the author, so a
                        // collaborator's lasso never clips this stroke.
                        let selection = state.selection_of(self.id.actor);
                        let tiles = ctx.stroke.render(
                            crate::gpu::stroke::StrokeScene {
                                pool: &ctx.pool,
                                assets: &ctx.assets,
                                base,
                                selection: &selection,
                            },
                            rec,
                        );
                        let layer = state.layer_at(idx).with_tiles(tiles);
                        state.with_layer_at(idx, layer)
                    }
                    // Absent layer, or a matte — a matte has no tile map, so a
                    // stroke targeting one is refused rather than swallowed
                    // (FRAME_DESIGN.md §7). Refusing here and not only in the
                    // frontend is what keeps replay and peers agreeing about a
                    // log that happens to contain such a stroke.
                    None => state,
                }
            }
            ActionKind::AddLayer { id, above } => state.insert_layer(*id, *above),
            ActionKind::RemoveLayer(id) => state.remove_layer(*id),
            ActionKind::SetLayerBlend(id, blend) => state.set_layer_blend(*id, *blend),
            ActionKind::SetLayerOpacity(id, opacity) => state.set_layer_opacity(*id, *opacity),
            ActionKind::SetLayerVisible(id, visible) => state.set_layer_visible(*id, *visible),
            ActionKind::SetLayerName(id, name) => {
                state.set_layer_name(*id, name.as_deref().map(Into::into))
            }
            ActionKind::MoveLayer { id, above } => state.move_layer(*id, *above),
            // Resolved at the timeline layer (effective-sequence filtering); an
            // `Undo` should never be materialized through `apply`. Identity, so
            // a stray one is harmless rather than wrong.
            ActionKind::Undo(_) => state,
            // The author's own selection, and only ever the author's: the key is
            // taken from `self.id.actor`, never from the payload, so an action
            // cannot address anyone else's mask (PEER_DESIGN.md §3).
            //
            // An op too large to rasterize (see `MAX_SELECTION_TILES`) leaves the
            // selection alone — deterministically, since the bound is a pure
            // function of the op, so peers and replays agree.
            ActionKind::Select(op) => {
                let prev = state.selection_of(self.id.actor);
                match ctx.selection.apply(&ctx.pool, &prev, op) {
                    Some(selection) => state.with_selection(self.id.actor, selection),
                    None => {
                        tracing::warn!("selection op too large to rasterize; ignored");
                        state
                    }
                }
            }
            ActionKind::InvertSelection => {
                let prev = state.selection_of(self.id.actor);
                let selection = ctx.selection.invert(&ctx.pool, &prev);
                state.with_selection(self.id.actor, selection)
            }
            ActionKind::SetSurface(id) => state.with_surface(*id),
            ActionKind::AddMatte {
                id,
                above,
                region,
                color,
            } => state.insert_matte(*id, *above, *region, *color),
            ActionKind::SetMatteRect(id, min, max) => state.set_matte_rect(*id, *min, *max),
            ActionKind::SetMatteColor(id, color) => state.set_matte_color(*id, *color),
            ActionKind::SetBackground(rgb) => state.with_background(*rgb),
            // Cut the author's selected paint, restack it under the affine, and
            // carry the author's mask with it (TRANSFORM_DESIGN.md). Gated and
            // keyed exactly as a stroke is: the mask comes off the state being
            // folded over, the actor off the action's own id. A matte or absent
            // layer refuses it, like a stroke; an unusable or oversized transform
            // is rejected deterministically, so peers and replays agree.
            ActionKind::Transform { layer, affine } => {
                match state
                    .layer_index(*layer)
                    .and_then(|idx| state.layer_at(idx).tiles().map(|base| (idx, base)))
                {
                    Some((idx, base)) => {
                        let selection = state.selection_of(self.id.actor);
                        match ctx.transform.apply(&ctx.pool, base, &selection, *affine) {
                            Some((tiles, moved_selection)) => {
                                let layer = state.layer_at(idx).with_tiles(tiles);
                                state
                                    .with_layer_at(idx, layer)
                                    .with_selection(self.id.actor, moved_selection)
                            }
                            None => {
                                tracing::warn!(
                                    "transform rejected (unusable affine or too many tiles); ignored"
                                );
                                state
                            }
                        }
                    }
                    None => state,
                }
            }
        })
    }
}
