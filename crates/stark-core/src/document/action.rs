//! Actions: committed, deterministic, replayable document mutations (§4).
//!
//! An [`Action`] is the unit the timeline stores/replays and (later) the unit
//! serialized to disk. Every action carries a globally-unique [`ActionId`] so
//! the same records work unchanged in a future replicated, multi-peer log
//! (§4, §12) — we pay that tiny cost from the first commit.

use serde::{Deserialize, Serialize};

use super::brush::BrushParams;
use super::filter::Filter;
use super::layer::{BlendMode, Layer, LayerId, MatteRegion, Place};
use super::selection::SelectionOp;
use super::state::DocState;
use crate::geom::Vec2;
use crate::gpu::SurfaceId;
use crate::gpu::selection::SelectionRenderer;
use crate::gpu::stroke::StrokeRenderer;
use crate::gpu::tile::TilePool;

/// Identifies the author of an action: one local user, or a peer (§4).
/// Maps to an iroh `NodeId` when collaborating; a fixed value when solo.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ActorId(pub u64);

impl ActorId {
    /// The fixed author id used when not collaborating. When a document is
    /// first shared, its solo-authored actions are rewritten to the sharer's
    /// real actor id (so the sharer can still undo them, §12.3);
    /// after that every action in a shared log carries a peer-derived id.
    pub const SOLO: ActorId = ActorId(0);
}

/// Globally-unique action id; also the total order key `(lamport, actor)`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ActionId {
    pub lamport: u64,
    pub actor: ActorId,
}

/// The tool a gesture drives. Tools become an open registry later (§10).
///
/// **Session state, not document state.** Only [`Brush`](Self::Brush) ever reaches
/// a [`StrokeRecord`]: the selection tools produce a [`SelectionOp`] instead of a
/// stroke (§6.8). They share the enum — and so the pointer-gesture plumbing —
/// because from the frontend's point of view they are the same interaction: press,
/// drag, release. But which of them was in hand is not part of what a document
/// *is*; the stroke or the op it produced is, and that is what the log carries.
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

/// A fully-recorded stroke: enough to replay it bit-for-bit (§4).
///
/// Deliberately does **not** carry the [`Tool`]. Only `Tool::Brush` can reach a
/// stroke — the selection tools produce a [`SelectionOp`] instead — so the field
/// held one value for every stroke of every document and no reader ever asked it
/// (§8, wire version 5). A tool worth recording would be recorded by whatever
/// distinguishes it, which this enum does not.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StrokeRecord {
    pub layer: LayerId,
    pub brush: BrushParams,
    /// The fitted stroke curve: the control points the raw pointer samples were
    /// smoothed and simplified down to (§6.2), an order of magnitude
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
        /// Whose carried stack to add it to — `None` for the document's own
        /// (§14.8).
        carrier: Option<LayerId>,
        above: Option<LayerId>,
    },
    /// Remove a layer **and everything it carries**: the subtree is the group
    /// (§14.2). Promoting what it carried instead is a
    /// [`MoveLayer`](Self::MoveLayer), which is what "release" is spelled with.
    RemoveLayer(LayerId),
    SetLayerBlend(LayerId, BlendMode),
    SetLayerOpacity(LayerId, f32),
    SetLayerVisible(LayerId, bool),
    /// Move a layer — with everything it carries — into the stack carried by
    /// `carrier` (the document's own when `None`), at the place `at` names in it.
    ///
    /// The **only** structural move, covering all three gestures at once
    /// (§14.8): reorder is `carrier` unchanged, *carry* is `carrier`
    /// set, *release* is `carrier` cleared. Declined deterministically when it
    /// would make a layer carry its own ancestor — see
    /// [`DocState::move_layer`](super::state::DocState::move_layer) for why the
    /// log's total order is all the cycle protection this needs.
    MoveLayer {
        id: LayerId,
        carrier: Option<LayerId>,
        at: Place,
    },
    /// Undo **as a logged action** (§5.4, §12.3): a fact peers can see
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

    /// Switch the canvas surface (§6.4).
    ///
    /// Logged rather than kept as a view setting because the surface feeds the
    /// document: which canvas a piece was painted on is part of what it is, and
    /// replay has to reconstruct it. Appended last so
    /// postcard decoding of older files is unaffected; documents saved before this
    /// existed simply never contain one and keep the surface from `CanvasMeta`.
    SetSurface(SurfaceId),
    /// Edit the selection mask (§6.8). Historized because a stroke's
    /// pixels depend on the mask in force when it was drawn — replaying the log has
    /// to put the same mask back. Only the **op** travels (a few floats, or a
    /// decimated polyline); every peer rasterizes it identically from the same
    /// shader, so the log stays compact and convergence is unaffected.
    Select(SelectionOp),
    /// Swap selected for unselected everywhere (§6.8).
    InvertSelection,

    /// Add a **matte** layer — a region filled with a flat colour
    /// (§15.2). A frame is one of these on top of the stack; the
    /// same action serves comic gutters and opaque grounds once the region
    /// generalizes (P4). Appended last, like every variant before it, so postcard
    /// — which encodes an enum by variant *index* — keeps decoding older files.
    AddMatte {
        id: LayerId,
        carrier: Option<LayerId>,
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
    /// sRGB (§15.5). Logged because the ground a piece was painted on
    /// is part of what it is; it was previously a view setting, so the paper colour
    /// of a painting was not saved at all.
    SetBackground([f32; 3]),

    /// Affine transform of the selected paint on `layer` (§16):
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

    /// Fill a region of `layer` with paint (§18.0.4).
    ///
    /// The fifth thing a shape gesture can do, alongside the four ways it can
    /// combine into the selection — see [`ShapeAction`](super::fill::ShapeAction).
    /// Gated and keyed exactly as a stroke is: the **author's** mask, taken off the
    /// state being folded over, bounds the fill, and the actor comes from the
    /// action's own id. A matte or absent layer refuses it, like a stroke.
    ///
    /// Deterministically **rejected** (the document is left unchanged) when the fill
    /// would be unbounded — [`SelectionShape::All`](super::selection::SelectionShape::All)
    /// with nothing selected — or would
    /// exceed [`MAX_FILL_TILES`](super::fill::MAX_FILL_TILES). Appended last so
    /// postcard, which encodes an enum by variant *index*, keeps decoding older
    /// files.
    Fill {
        layer: LayerId,
        op: super::fill::FillOp,
    },

    /// Clip a layer to the paint beneath it, or stop (§14.4).
    ///
    /// A presentation property like the blend mode it is applied beside, and
    /// logged like one: it changes what the document *looks* like, so replay and
    /// peers both have to reproduce it. On the base of a group it clips the whole
    /// group to what lies under the group (§14.4.3) — the same outward reading its
    /// blend mode gets, which is why this needs no second action for groups.
    SetLayerClip(LayerId, bool),

    /// Perspective transform of the selected paint on `layer` within the map's
    /// source rect (§16.8): cut what the **author's** selection holds inside the
    /// rect, resample it once under the homography the map's corners define,
    /// stack it back over what remained — and carry the covered part of the
    /// author's mask along, unioned with what stayed outside the rect. Twelve
    /// floats in the log; every peer re-derives the same matrix from them.
    /// Appended last so postcard keeps decoding older files.
    ///
    /// Deterministically **rejected** (the document is left unchanged) when the
    /// map is unusable — a non-convex target quad, a degenerate rect — or the
    /// rewrite exceeds the tile caps (§16.1).
    TransformPerspective {
        layer: LayerId,
        map: super::transform::PerspectiveMap,
    },

    /// Warp of the selected paint on `layer` within the mesh's source rect
    /// (§16.9): the same cut/stack/carry as
    /// [`TransformPerspective`](Self::TransformPerspective), under the smooth
    /// surface through the map's control grid. The log carries only the grid —
    /// a few dozen floats — and every peer subdivides it identically. Appended
    /// last so postcard keeps decoding older files.
    ///
    /// Deterministically **rejected** when the mesh folds (any sub-cell's
    /// Jacobian runs non-positive), is malformed, or exceeds the tile caps.
    TransformWarp {
        layer: LayerId,
        map: super::warp::WarpMap,
    },

    /// Copy a layer — **and everything it carries** — into its own stack,
    /// directly above the layer it was copied from (§14.8). The subtree travels
    /// as one for the reason [`RemoveLayer`](Self::RemoveLayer)'s does: the
    /// subtree *is* the group (§14.2).
    ///
    /// `ids` pairs every layer of that subtree, in composite order, with the id
    /// its copy takes; the first pair's source is the layer being duplicated.
    /// The copies' ids are minted by the author and travel in the log for the
    /// reason [`AddLayer`](Self::AddLayer)'s does — a replay must mint what the
    /// run that recorded it minted, and two peers duplicating at once must not
    /// land on one id (§17.9).
    ///
    /// Naming the **sources** as well is what lets the footprint be honest: a
    /// copy is a function of every tile and every property of every layer it
    /// copies, so a duplicate does not commute with a stroke or a rename inside
    /// the group, and an action that named only the root could not say so
    /// (§12.6).
    ///
    /// Deterministically **rejected** (the document is left unchanged) when the
    /// subtree holds a layer `ids` does not name — which is what a concurrent add
    /// into the group looks like from here. Appended last so postcard, which
    /// encodes an enum by variant *index*, keeps decoding older files.
    DuplicateLayer {
        ids: Vec<(LayerId, LayerId)>,
    },

    /// Add a **filter** layer — a function of everything composited beneath it in
    /// the stack it lands in (§21.2).
    ///
    /// It arrives holding a filter at its neutral setting, so adding one changes
    /// nothing until it is dialled; the dialling is [`SetFilter`](Self::SetFilter).
    /// Placed by the same two anchors every other layer is (§14.8), which is the
    /// whole of how far a filter reaches — there is no scope to set, because *where
    /// it sits is its scope*. Appended last, like every variant before it, so
    /// postcard — which encodes an enum by variant *index* — keeps decoding older
    /// files.
    AddFilter {
        id: LayerId,
        carrier: Option<LayerId>,
        above: Option<LayerId>,
        filter: Filter,
    },
    /// Retune a filter layer (§21.5). One action per adjustment, not per pointer
    /// move: a slider drag previews in view state and commits on release, the
    /// bargain the frame drag and the opacity slider already make (§15.7, §14.6).
    ///
    /// Carries the **whole** filter rather than one parameter, so a filter that
    /// grows a knob — or a new kind of filter entirely — needs no new action and no
    /// wire-format break. A no-op on a layer that is not a filter, like
    /// [`SetMatteColor`](Self::SetMatteColor) on a paint layer.
    SetFilter(LayerId, Filter),

    /// Merge `source` **down** onto `dest`, the layer directly beneath it: `dest`
    /// keeps its identity and its properties and takes the paint of both, `source`
    /// ceases to exist (§14.11).
    ///
    /// The one action in this list whose promise is about *pixels that do not change*:
    /// a merge is offered exactly where the pair composites identically to the one
    /// layer, so the document looks the same before and afterwards. Which pairs those
    /// are is [`merge::plan`](super::merge::plan), a pure function of the state — so
    /// the log carries no reasoning, only the two ids, and every peer and every replay
    /// re-derives the same answer from the same document.
    ///
    /// `dest` is derived rather than chosen, and travels anyway for the reason
    /// [`DuplicateLayer`](Self::DuplicateLayer)'s ids do: a [`Footprint`] is built from
    /// the action alone and cannot search the tree for what "down" meant (§12.6).
    /// Naming it is also what makes the rejection honest — an action whose plan now
    /// points somewhere else is **deterministically declined**, leaving the document
    /// unchanged, which is what a concurrent reorder looks like from here.
    ///
    /// Appended last, like every variant before it, so postcard — which encodes an
    /// enum by variant *index* — keeps decoding older files.
    ///
    /// [`Footprint`]: super::footprint::Footprint
    MergeLayerDown {
        source: LayerId,
        dest: LayerId,
    },
}

impl ActionKind {
    /// What this action *is*, in two or three words — the caption a history
    /// scrubber puts on the step it is about to cross (§18.2.4).
    ///
    /// A `&'static str` and nothing more: a timeline showing a hundred steps needs
    /// them by the hundred, and the point of the caption is to tell a stroke from a
    /// layer change at a glance, not to describe either. Anything richer — which
    /// layer, what colour — is what the canvas beside it is for.
    pub fn label(&self) -> &'static str {
        match self {
            ActionKind::CommitStroke(_) => "Stroke",
            ActionKind::Fill { .. } => "Fill",
            ActionKind::Transform { .. } => "Transform",
            ActionKind::TransformPerspective { .. } => "Perspective",
            ActionKind::TransformWarp { .. } => "Warp",
            ActionKind::Select(_) => "Select",
            ActionKind::InvertSelection => "Invert selection",
            ActionKind::AddLayer { .. } => "Add layer",
            ActionKind::DuplicateLayer { .. } => "Duplicate layer",
            ActionKind::RemoveLayer(_) => "Remove layer",
            ActionKind::MergeLayerDown { .. } => "Merge down",
            ActionKind::MoveLayer { .. } => "Reorder layer",
            ActionKind::SetLayerBlend(..) => "Blend mode",
            ActionKind::SetLayerClip(..) => "Clip layer",
            ActionKind::SetLayerOpacity(..) => "Layer opacity",
            ActionKind::SetLayerVisible(..) => "Layer visibility",
            ActionKind::SetLayerName(..) => "Rename layer",
            ActionKind::AddMatte { .. } => "Add frame",
            ActionKind::AddFilter { .. } => "Add filter",
            ActionKind::SetFilter(..) => "Filter",
            ActionKind::SetMatteRect(..) => "Move frame",
            ActionKind::SetMatteColor(..) => "Frame colour",
            ActionKind::SetBackground(_) => "Canvas colour",
            ActionKind::SetSurface(_) => "Canvas surface",
            ActionKind::Undo(_) => "Undo",
        }
    }
}

/// A committed document mutation with its identity.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Action {
    pub id: ActionId,
    pub kind: ActionKind,
}

/// Side-channel passed to [`history::Action::apply`]: the GPU resources needed
/// to render a stroke (§5). It owns cheap `Arc`-backed clones, so it
/// has no borrow lifetime — which is what lets it be the `Action::Context`.
pub struct ApplyCtx {
    pub pool: TilePool,
    pub stroke: StrokeRenderer,
    pub assets: crate::assets::AssetStore,
    pub selection: SelectionRenderer,
    pub transform: crate::gpu::transform::TransformRenderer,
    pub fill: crate::gpu::fill::FillRenderer,
    pub merge: crate::gpu::merge::MergeRenderer,
    /// The device, so a canvas surface can be built here on demand.
    pub gpu: crate::gpu::context::GpuContext,
    /// The canvas surfaces and the bytes registered for them (§6.4).
    ///
    /// It lives here, rather than beside the compositor it also feeds, because the
    /// **deposit reads it**: the tooth gates the paint a stroke lays by the ground
    /// under it, so which surface a stroke sees is part of applying that stroke.
    /// Asked with [`DocState::surface`](super::state::DocState) *as the log stood at
    /// that action*, which is the whole reason `SetSurface` was made a logged action
    /// rather than a view setting — a stroke from before a mid-document switch
    /// deposits against the ground it was actually painted on, on replay and on a
    /// peer alike.
    pub surfaces: crate::gpu::registry::Registry<SurfaceId>,
}

impl ApplyCtx {
    /// The canvas surface `id` names, built on demand ([`Registry::get`]).
    ///
    /// Returns an owned handle rather than a borrow — `Surface` is a pair of
    /// reference-counted wgpu objects, so the clone is two atomic bumps — because
    /// the caller then borrows other fields of `self` to build the scene around it.
    ///
    /// [`Registry::get`]: crate::gpu::registry::Registry::get
    pub fn surface(&mut self, id: SurfaceId) -> crate::gpu::surface::Surface {
        let gpu = self.gpu.clone();
        self.surfaces.get(&gpu, id).clone()
    }
}

impl history::Action for Action {
    type State = DocState;
    type Context = ApplyCtx;
    // GPU work reports failure via wgpu's device error callbacks, not return
    // values, and tile allocation never fails — so applying an action is
    // genuinely infallible here (§5).
    type Error = std::convert::Infallible;
    /// An action commutes with everything its [`Footprint`] is disjoint from
    /// (§12.6) — which is what lets the history splice an undone
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
            ActionKind::CommitStroke(rec) => {
                let Some(base) = paint_base(&state, rec.layer) else {
                    return Ok(state);
                };
                // The **author's** selection, as it stood at this point in the
                // log, gates the stroke (§6.8, §17.3). Read from the state being
                // folded over, so replay reproduces it exactly; keyed by the
                // author, so a collaborator's lasso never clips this stroke.
                let selection = state.selection_of(self.id.actor);
                // The ground this stroke was painted on, as the log stood here —
                // not as it stands now (§6.4). The tooth gates the deposit by it,
                // so a mid-document `SetSurface` changes what comes *after* it and
                // nothing before, on replay exactly as it did live.
                let surface = ctx.surface(state.surface);
                let tiles = ctx.stroke.render(
                    crate::gpu::stroke::StrokeScene {
                        pool: &ctx.pool,
                        assets: &ctx.assets,
                        base: &base,
                        selection: &selection,
                        surface: &surface,
                    },
                    rec,
                );
                state.map_layer(rec.layer, |l| l.with_tiles(tiles))
            }
            ActionKind::AddLayer { id, carrier, above } => {
                state.insert_layer(*id, *carrier, *above)
            }
            // The copy's tiles are the shared handles the source already holds, so
            // duplicating a layer costs no GPU memory until one of the two is
            // painted on — copy-on-write is what makes this a cheap action rather
            // than a re-render of everything under it (§5.2).
            ActionKind::DuplicateLayer { ids } => state.duplicate_layer(ids),
            ActionKind::RemoveLayer(id) => state.remove_layer(*id),
            ActionKind::SetLayerBlend(id, blend) => state.set_layer_blend(*id, *blend),
            ActionKind::SetLayerClip(id, clip) => state.set_layer_clip(*id, *clip),
            ActionKind::SetLayerOpacity(id, opacity) => state.set_layer_opacity(*id, *opacity),
            ActionKind::SetLayerVisible(id, visible) => state.set_layer_visible(*id, *visible),
            ActionKind::SetLayerName(id, name) => {
                state.set_layer_name(*id, name.as_deref().map(Into::into))
            }
            ActionKind::MoveLayer { id, carrier, at } => state.move_layer(*id, *carrier, *at),
            // Resolved at the timeline layer (effective-sequence filtering); an
            // `Undo` should never be materialized through `apply`. Identity, so
            // a stray one is harmless rather than wrong.
            ActionKind::Undo(_) => state,
            // The author's own selection, and only ever the author's: the key is
            // taken from `self.id.actor`, never from the payload, so an action
            // cannot address anyone else's mask (§17.3).
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
                carrier,
                above,
                region,
                color,
            } => state.insert_matte(*id, *carrier, *above, *region, *color),
            // The payload is sanitized inside `insert_filter`/`set_filter` — the
            // funnel sits in state, where replayed files and remote peers land too,
            // not only where a local command is minted (§21.5).
            ActionKind::AddFilter {
                id,
                carrier,
                above,
                filter,
            } => state.insert_filter(*id, *carrier, *above, *filter),
            ActionKind::SetFilter(id, filter) => state.set_filter(*id, *filter),
            ActionKind::SetMatteRect(id, min, max) => state.set_matte_rect(*id, *min, *max),
            ActionKind::SetMatteColor(id, color) => state.set_matte_color(*id, *color),
            ActionKind::SetBackground(rgb) => state.with_background(*rgb),
            // Cut the author's selected paint, restack it under the affine, and
            // carry the author's mask with it (§16). Gated and
            // keyed exactly as a stroke is: the mask comes off the state being
            // folded over, the actor off the action's own id. A matte or absent
            // layer refuses it, like a stroke; an unusable or oversized transform
            // is rejected deterministically, so peers and replays agree.
            ActionKind::Transform { layer, affine } => transform_apply(
                state,
                ctx,
                self.id.actor,
                *layer,
                &crate::document::transform::TransformMap::Affine(*affine),
            ),
            // The rect-scoped siblings (§16.8, §16.9): identical shape — cut,
            // restack, carry the mask — differing only in the map handed to
            // the renderer.
            ActionKind::TransformPerspective { layer, map } => transform_apply(
                state,
                ctx,
                self.id.actor,
                *layer,
                &crate::document::transform::TransformMap::Perspective(*map),
            ),
            ActionKind::TransformWarp { layer, map } => transform_apply(
                state,
                ctx,
                self.id.actor,
                *layer,
                &crate::document::transform::TransformMap::Warp(map.clone()),
            ),
            // Lay a parcel of paint through the region's coverage, gated by the
            // author's selection — the same gate a stroke passes through, so a fill
            // is clipped by a selection exactly as a brush is
            // (§18.0.4). Refused on a matte or absent layer like a stroke; refused
            // deterministically when unbounded or oversized, so peers and replays
            // agree about a log that contains one.
            // Fold two layers into one without moving a pixel of the composite
            // (§14.11). The plan is re-derived from the state being folded over rather
            // than trusted from the log, so a replay and a peer decide from the same
            // document — and a plan that no longer names `dest` (a concurrent reorder,
            // a mode set on either layer since) declines the whole action, leaving the
            // document untouched. Deterministic, so everyone declines together.
            ActionKind::MergeLayerDown { source, dest } => {
                let Some(plan) = super::merge::plan(&state, *source) else {
                    return Ok(state);
                };
                if plan.dest != *dest {
                    tracing::warn!("merge down no longer names this destination; ignored");
                    return Ok(state);
                }
                // Both sides are paint that carries nothing — `plan` said so — so the
                // tile maps are there to be read. Cloned out before the rewrite for
                // the reason `paint_base` clones: a handful of `Arc` bumps, and it is
                // what keeps the borrow of the state off the tree being rebuilt.
                let (Some(lower), Some(upper)) =
                    (paint_base(&state, *dest), paint_base(&state, *source))
                else {
                    return Ok(state);
                };
                // Every number here is the plan's: what each side's tiles are worth on
                // their own, how the upper meets the lower, and what the survivor
                // carries afterwards. The two differ by where the destination's slider
                // belongs — folded into the tiles beside a sibling, left on the layer
                // when the destination is a carrier and the slider is the group's
                // (§14.7) — and that decision is made once, in `plan`, rather than
                // twice here and there.
                let tiles = ctx.merge.apply(
                    &ctx.pool,
                    crate::gpu::merge::MergeScene {
                        lower: crate::gpu::merge::MergeSide {
                            tiles: &lower,
                            opacity: plan.dest_opacity,
                        },
                        upper: crate::gpu::merge::MergeSide {
                            tiles: &upper,
                            opacity: plan.source_params.opacity,
                        },
                        blend: plan.source_params.blend,
                        clip: plan.source_params.clip,
                    },
                );
                state
                    .map_layer(*dest, |l| Layer {
                        composite: plan.keeps,
                        ..l.with_tiles(tiles)
                    })
                    .remove_layer(*source)
            }
            ActionKind::Fill { layer, op } => {
                let Some(base) = paint_base(&state, *layer) else {
                    return Ok(state);
                };
                let selection = state.selection_of(self.id.actor);
                match ctx.fill.apply(&ctx.pool, &base, &selection, op) {
                    Some(tiles) => state.map_layer(*layer, |l| l.with_tiles(tiles)),
                    None => {
                        tracing::warn!(
                            "fill rejected (unbounded region or too many tiles); ignored"
                        );
                        state
                    }
                }
            }
        })
    }
}

/// The shared body of the three transform actions (§16): cut the author's
/// selected paint, restack it under `map`, and carry the author's mask with it.
/// Gated and keyed exactly as a stroke is — the mask comes off the state being
/// folded over, the actor off the action's own id. A matte or absent layer
/// refuses it, like a stroke; an unusable or oversized map is rejected
/// deterministically, so peers and replays agree.
fn transform_apply(
    state: DocState,
    ctx: &ApplyCtx,
    actor: ActorId,
    layer: LayerId,
    map: &crate::document::transform::TransformMap,
) -> DocState {
    let Some(base) = paint_base(&state, layer) else {
        return state;
    };
    let selection = state.selection_of(actor);
    match ctx.transform.apply(&ctx.pool, &base, &selection, map) {
        Some((tiles, moved_selection)) => state
            .map_layer(layer, |l| l.with_tiles(tiles))
            .with_selection(actor, moved_selection),
        None => {
            tracing::warn!("transform rejected (unusable map or too many tiles); ignored");
            state
        }
    }
}

/// The tiles `layer` paints into, or `None` if it has none — the gate every
/// action that lays or moves paint passes through first.
///
/// **The refusal is the point.** A matte has no tile map, so a stroke, a fill or
/// a transform aimed at one is turned away rather than swallowed or magically
/// rasterized (§15.7) — and turned away *here*, in the engine, not only in the
/// frontend, which is what keeps replay and peers agreeing about a log that
/// happens to contain such an action. An absent layer reads the same way: there
/// is nothing there to paint on.
///
/// Cloned out of the tree before anything is rebuilt: the map is persistent, so
/// this is a handful of `Arc` bumps, and it is what keeps the borrow of the state
/// from outliving the rewrite that follows.
fn paint_base(state: &DocState, layer: LayerId) -> Option<crate::gpu::tile::TileMap> {
    state.layer(layer).and_then(|l| l.tiles()).cloned()
}
