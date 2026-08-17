//! Actions: committed, deterministic, replayable document mutations (§4).
//!
//! An [`Action`] is the unit the timeline stores/replays and (later) the unit
//! serialized to disk. Every action carries a globally-unique [`ActionId`] so
//! the same records work unchanged in a future replicated, multi-peer log
//! (§4, §12) — we pay that tiny cost from the first commit.

use serde::{Deserialize, Serialize};

use super::brush::BrushParams;
use super::fill::FillOp;
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
    /// `stark-engine`'s `DocState::move_layer` for why the
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
    /// which actions are *effective* (see `stark-engine`'s `document::effective_actions`)
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
    /// `document::transform`.
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
    /// are is `stark-engine`'s `document::merge::plan`, a pure function of the state — so
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

    /// Bring an image in from **outside** the document — an image file, or the system
    /// clipboard — as a new layer holding it as paint (§23).
    ///
    /// One action rather than three, and that is the whole of why it mints its own
    /// layer instead of taking one. A paste lands on its own layer in every tool that
    /// has one, so spelling it as `AddLayer` then a placement then a rename would put
    /// the familiar gesture three undo steps deep, and leave two of those steps
    /// meaning nothing on their own. `AddMatte` already carries this shape: a layer
    /// arriving with content is one fact, not a layer and then its content.
    ///
    /// The pixels are the payload, behind an [`Arc`](std::sync::Arc) and PNG-encoded
    /// on the wire — see [`ImageRef`](super::ImageRef) for both, and for why this is
    /// the one action that carries content rather than naming it.
    ///
    /// **`at` is in whole canvas pixels, and that is a promise about resampling**: the
    /// image's texels land on canvas pixels one for one, so nothing is filtered and
    /// there is no sampling loss between the file and the tiles. Scaling and turning it
    /// afterwards is [`Transform`](Self::Transform), which is where resampling belongs
    /// and where its exactness is already pinned (§16.4) — expressing the placement as
    /// a float and then a scale would have spent one generation of blur on every import
    /// to reach the same picture. An integer vector is how that is said in a way the
    /// payload cannot express wrongly.
    ///
    /// Deterministically **rejected** (the document is left unchanged) when the anchors
    /// name a layer that is not there, exactly as an [`AddLayer`](Self::AddLayer) is, or
    /// when the placement falls off the tile grid an `i32` can address. Appended last,
    /// like every variant before it, so postcard — which encodes an enum by variant
    /// *index* — keeps decoding older files.
    PlaceImage {
        id: LayerId,
        /// Whose carried stack to add it to — `None` for the document's own (§14.8).
        carrier: Option<LayerId>,
        above: Option<LayerId>,
        /// Canvas position of the image's top-left texel, in whole canvas pixels.
        at: crate::geom::IVec2,
        /// What to call the layer — the file it came from, so the layers panel says
        /// "sunset.jpg" rather than a number. `None` leaves it described by its place
        /// in the stack, which is what a clipboard image with no name has.
        name: Option<String>,
        /// The picture, by **content id** — named here and carried beside the log,
        /// exactly as a stamp brush's shape is (§6.6, §23).
        image: crate::AssetId,
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
            | ActionKind::AddFilter { id, .. }
            | ActionKind::PlaceImage { id, .. } => (Some(*id), &[]),
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

    /// The same action with every payload finite and in range — **the one funnel
    /// an action passes through on its way into the document.**
    ///
    /// The pieces existed already and were called one at a time:
    /// `Filter::sanitized` at two sites, `BlendMode::sanitized` at two more, and
    /// nothing at all for the brush a stroke carries or the ops a fill and a
    /// selection do. That is a list a caller keeps, and §1 prefers ruling out a
    /// class to enumerating its instances — so the list is here, once, and the
    /// engine sanitizes an *action* rather than remembering which of its payloads
    /// have knobs.
    ///
    /// **Exhaustive, with no `_` arm**, which is the whole point of writing it this
    /// way: a variant added later stops this compiling until it says whether it
    /// carries a number, where a wildcard would answer "nothing to hold" on its
    /// behalf and be right until the day it was not. Same device as
    /// [`minted_layers`](Self::minted_layers) and
    /// [`action_content`](crate::content::action_content), after the same escape.
    ///
    /// **Idempotent** on anything this engine wrote, so applying it on the way in
    /// *and* on replay cannot make a load into a small edit.
    pub fn sanitized(self) -> Self {
        match self {
            ActionKind::CommitStroke(rec) => ActionKind::CommitStroke(StrokeRecord {
                brush: rec.brush.sanitized(),
                ..rec
            }),
            ActionKind::Fill { layer, op } => ActionKind::Fill {
                layer,
                // The op's own invariants are held by its deserialization gate
                // (`FillOp::with_paint`), so rebuilding through it is how they are
                // stated once rather than twice.
                op: FillOp::with_paint(op.shape, op.feather, op.paint, op.opacity),
            },
            ActionKind::Select(op) => {
                ActionKind::Select(SelectionOp::at(op.mode, op.shape, op.feather, op.opacity))
            }
            ActionKind::SetLayerBlend(id, mode) => ActionKind::SetLayerBlend(id, mode.sanitized()),
            ActionKind::SetFilter(id, f) => ActionKind::SetFilter(id, f.sanitized()),
            ActionKind::AddFilter {
                id,
                carrier,
                above,
                filter,
            } => ActionKind::AddFilter {
                id,
                carrier,
                above,
                filter: filter.sanitized(),
            },
            ActionKind::SetLayerOpacity(id, a) => {
                // A layer's opacity is a coverage weight the compositor multiplies
                // by; a NaN would take the whole layer with it.
                ActionKind::SetLayerOpacity(
                    id,
                    if a.is_finite() {
                        a.clamp(0.0, 1.0)
                    } else {
                        1.0
                    },
                )
            }
            // Nothing to hold: ids, flags, places, and the geometry whose own
            // `usable`/`affine_usable` gate rejects it at `apply` rather than
            // rounding it into something else (§16.1) — a transform that cannot be
            // clamped into a *different* transform without changing what the
            // author asked for.
            ActionKind::AddLayer { .. }
            | ActionKind::AddMatte { .. }
            // A placement carries pixels and an integer position: no float to be
            // non-finite, no knob to be out of range. What *could* be malformed about
            // an image — dimensions that disagree with the buffer, a size past the cap
            // — cannot be clamped into a different image, and is refused at
            // `ImageRef`'s constructor and at its decode instead (§23).
            | ActionKind::PlaceImage { .. }
            | ActionKind::DuplicateLayer { .. }
            | ActionKind::RemoveLayer(_)
            | ActionKind::MergeLayerDown { .. }
            | ActionKind::MoveLayer { .. }
            | ActionKind::SetLayerClip(..)
            | ActionKind::SetLayerVisible(..)
            | ActionKind::SetLayerName(..)
            | ActionKind::SetMatteRect(..)
            | ActionKind::SetMattePaint(..)
            | ActionKind::SetBackground(_)
            | ActionKind::SetSurface(_)
            | ActionKind::InvertSelection
            | ActionKind::Transform { .. }
            | ActionKind::TransformPerspective { .. }
            | ActionKind::TransformWarp { .. }
            | ActionKind::Undo(_) => self,
        }
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
            ActionKind::PlaceImage { .. } => "Place image",
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{Parcel, SelectionMode, SelectionShape};
    use crate::geom::Vec2;

    /// The funnel reaches the payloads that carry numbers — the three that had no
    /// gate of their own before it existed (a stroke's brush, a fill's op, a
    /// selection's op) alongside the two that did.
    ///
    /// A `NaN` in any of them reaches a shader: a brush's radius sizes a dispatch,
    /// a fill's opacity is inverted through the coverage law, a selection's opacity
    /// scales the mask every other tool acts through.
    #[test]
    fn the_funnel_reaches_every_payload_that_carries_a_number() {
        let bad = f32::NAN;

        let stroke = ActionKind::CommitStroke(StrokeRecord {
            layer: LayerId(0),
            brush: BrushParams {
                radius: bad,
                tooth: 9.0,
                ..BrushParams::default()
            },
            path: Vec::new(),
            seed: 0,
        })
        .sanitized();
        let ActionKind::CommitStroke(rec) = &stroke else {
            panic!("a stroke stays a stroke")
        };
        assert!(rec.brush.radius.is_finite());
        assert_eq!(rec.brush.tooth, 1.0);

        let fill = ActionKind::Fill {
            layer: LayerId(0),
            op: FillOp {
                shape: SelectionShape::All,
                feather: -1.0,
                paint: Parcel::Solid([bad; 3]),
                opacity: 5.0,
            },
        }
        .sanitized();
        let ActionKind::Fill { op, .. } = &fill else {
            panic!("a fill stays a fill")
        };
        assert_eq!((op.feather, op.opacity), (0.0, 1.0));

        let select = ActionKind::Select(SelectionOp {
            mode: SelectionMode::Replace,
            shape: SelectionShape::All,
            feather: bad,
            opacity: 0.5,
        })
        .sanitized();
        let ActionKind::Select(op) = &select else {
            panic!("a select stays a select")
        };
        assert_eq!((op.feather, op.opacity), (0.0, 1.0));

        // A layer's opacity is a coverage weight the compositor multiplies by.
        let ActionKind::SetLayerOpacity(_, a) =
            ActionKind::SetLayerOpacity(LayerId(0), bad).sanitized()
        else {
            panic!()
        };
        assert!(a.is_finite());
    }

    /// **Idempotent, on every variant.** The funnel runs where an action is minted
    /// *and* where it enters state, so a second pass that moved anything would make
    /// every load and every replay a small edit — and goldens are blessed against
    /// the first pass.
    ///
    /// Driven off the same one-of-each list the footprint's exhaustiveness device
    /// uses, so a variant added later is covered here as soon as it is added there.
    #[test]
    fn sanitizing_is_idempotent_on_every_kind() {
        let id = LayerId(3);
        let kinds = [
            ActionKind::CommitStroke(StrokeRecord {
                layer: id,
                brush: BrushParams::default(),
                path: vec![crate::path::ControlPoint::at(Vec2::splat(4.0))],
                seed: 1,
            }),
            ActionKind::AddLayer {
                id,
                carrier: None,
                above: None,
            },
            ActionKind::RemoveLayer(id),
            ActionKind::SetLayerBlend(id, BlendMode::Drago { k: 1e9 }),
            ActionKind::SetLayerOpacity(id, 0.5),
            ActionKind::SetLayerVisible(id, true),
            ActionKind::SetLayerClip(id, true),
            ActionKind::SetLayerName(id, Some("wash".into())),
            ActionKind::MoveLayer {
                id,
                carrier: None,
                at: Place::Top,
            },
            ActionKind::DuplicateLayer {
                ids: vec![(id, LayerId(9))],
            },
            ActionKind::MergeLayerDown {
                source: id,
                dest: LayerId(1),
            },
            ActionKind::Undo(ActionId {
                lamport: 1,
                actor: ActorId::SOLO,
            }),
            ActionKind::SetSurface(crate::SurfaceId::Flat),
            ActionKind::Select(SelectionOp::select_all()),
            ActionKind::InvertSelection,
            ActionKind::AddMatte {
                id,
                carrier: None,
                at: Place::Bottom,
                region: MatteRegion::Everything,
                paint: MattePaint::Solid([0.0; 3]),
            },
            ActionKind::PlaceImage {
                id,
                carrier: None,
                above: None,
                at: crate::geom::IVec2::new(-3, 9),
                name: Some("sunset.png".into()),
                image: crate::AssetId([4; 32]),
            },
            ActionKind::SetMatteRect(id, Vec2::ZERO, Vec2::splat(4.0)),
            ActionKind::SetMattePaint(id, MattePaint::Solid([1.0; 3])),
            ActionKind::SetBackground([0.5; 3]),
            ActionKind::Transform {
                layer: id,
                affine: crate::geom::Affine2::IDENTITY,
            },
            ActionKind::Fill {
                layer: id,
                op: FillOp::of_selection([0.2, 0.4, 0.6]),
            },
            ActionKind::AddFilter {
                id,
                carrier: None,
                above: None,
                filter: Filter::ALL[0].clone(),
            },
            ActionKind::SetFilter(id, Filter::ALL[1].clone()),
        ];
        for kind in kinds {
            let once = kind.sanitized();
            let twice = once.clone().sanitized();
            assert_eq!(
                format!("{once:?}"),
                format!("{twice:?}"),
                "sanitizing {} moved on the second pass",
                once.label(),
            );
        }
    }
}
