//! Actions: committed, deterministic, replayable document mutations (§4).
//!
//! An [`Action`] is the unit the timeline stores/replays and (later) the unit
//! serialized to disk. Every action carries a globally-unique [`ActionId`] so
//! the same records work unchanged in a future replicated, multi-peer log
//! (§4, §12) — we pay that tiny cost from the first commit.

use serde::{Deserialize, Serialize};

use super::brush::BrushParams;
use super::filter::Filter;
use super::layer::{BlendMode, LayerId, MattePaint, MatteRegion, Place};
use super::selection::SelectionOp;
use crate::SurfaceId;
use crate::geom::Vec2;

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

    /// Add a **matte** layer — a region filled with a [`MattePaint`]
    /// (§15.2). A frame is one of these on top of the stack; a ground
    /// ([`MatteRegion::Everything`]) is one at the bottom, which is why the
    /// anchor is the full [`Place`] where `AddLayer`'s stays the two-state
    /// `Option` (§15.5). The same action serves comic gutters once the region
    /// algebra lands (P4).
    AddMatte {
        id: LayerId,
        carrier: Option<LayerId>,
        at: Place,
        region: MatteRegion,
        paint: MattePaint,
    },
    /// Move a matte's rect — the frame drag's commit. One action per drag, not
    /// per pointer move: the gesture accumulates in session state and commits on
    /// release, so fifty tweaks are fifty undo steps rather than five thousand.
    /// A no-op on a region with no rect to move ([`MatteRegion::with_rect`]).
    SetMatteRect(LayerId, Vec2, Vec2),
    /// Repaint a matte — a flat color or a gradient ramp (§15.4, §22.4).
    SetMattePaint(LayerId, MattePaint),
    /// Set the canvas substrate color — the ground the paint sits on, straight
    /// sRGB (§15.5). Logged rather than held as a view setting, because the ground a
    /// piece was painted on is part of what it is: unlogged, the paper color of a
    /// painting would not be saved at all.
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
    /// [`SetMattePaint`](Self::SetMattePaint) on a paint layer.
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
    /// Every layer id this action **mints** — the ids a client's counter has to
    /// resume past when it picks a log back up (§17.9).
    ///
    /// Lives here, beside the variants, because it is a fact about *them*: minting
    /// is what an `Add…` action does, and an engine that keeps its own list of which
    /// ones do has a list that a new variant does not appear in.
    /// The engine's `resync_counters` kept exactly such a list, `AddFilter` was
    /// added after it, and a document whose highest id came from a filter reloaded
    /// with a counter that would mint that id a second time — two layers with one
    /// id, which is the convergence failure §17.9 says per-client identity rules
    /// out.
    ///
    /// **Exhaustive, with no `_` arm, and that is the whole point of it.** A variant
    /// added to the enum stops this function compiling, three lines from the doc
    /// comment that says why — the device `slot` in `tests/footprint.rs` already
    /// uses, for the same reason and after the same variant escaped it.
    ///
    /// Note it reports what the action *names as minted*, not what applying it
    /// lands: a rejected `AddLayer` (unknown carrier) inserts nothing, and its
    /// ordinal is still spent — which is the answer the counter wants, since
    /// re-minting it would collide with a peer who accepted the same action.
    pub fn minted_layers(&self) -> impl Iterator<Item = LayerId> + '_ {
        // One id, or a map of them — the two shapes minting comes in. Named as a
        // pair so the match below decides nothing else.
        let (one, copies): (Option<LayerId>, &[(LayerId, LayerId)]) = match self {
            ActionKind::AddLayer { id, .. }
            | ActionKind::AddMatte { id, .. }
            | ActionKind::AddFilter { id, .. } => (Some(*id), &[]),
            // A duplicate mints one per layer of the subtree it copied, which is
            // why the map travels in the action (§14.8).
            ActionKind::DuplicateLayer { ids } => (None, ids),
            ActionKind::CommitStroke(_)
            | ActionKind::RemoveLayer(_)
            | ActionKind::SetLayerBlend(..)
            | ActionKind::SetLayerClip(..)
            | ActionKind::SetLayerOpacity(..)
            | ActionKind::SetLayerVisible(..)
            | ActionKind::SetLayerName(..)
            | ActionKind::MoveLayer { .. }
            | ActionKind::MergeLayerDown { .. }
            | ActionKind::Undo(_)
            | ActionKind::SetSurface(_)
            | ActionKind::Select(_)
            | ActionKind::InvertSelection
            | ActionKind::SetMatteRect(..)
            | ActionKind::SetMattePaint(..)
            | ActionKind::SetFilter(..)
            | ActionKind::SetBackground(_)
            | ActionKind::Transform { .. }
            | ActionKind::TransformPerspective { .. }
            | ActionKind::TransformWarp { .. }
            | ActionKind::Fill { .. } => (None, &[]),
        };
        one.into_iter().chain(copies.iter().map(|&(_, copy)| copy))
    }

    /// What this action *is*, in two or three words — the caption a history
    /// scrubber puts on the step it is about to cross (§18.2.4).
    ///
    /// A `&'static str` and nothing more: a timeline showing a hundred steps needs
    /// them by the hundred, and the point of the caption is to tell a stroke from a
    /// layer change at a glance, not to describe either. Anything richer — which
    /// layer, what color — is what the canvas beside it is for.
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
            ActionKind::AddMatte { .. } => "Add matte",
            ActionKind::AddFilter { .. } => "Add filter",
            ActionKind::SetFilter(..) => "Filter",
            ActionKind::SetMatteRect(..) => "Move frame",
            ActionKind::SetMattePaint(..) => "Matte paint",
            ActionKind::SetBackground(_) => "Canvas color",
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
