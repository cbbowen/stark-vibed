//! Actions: committed, deterministic, replayable document mutations (§4).
//!
//! An [`Action`] is the unit the timeline stores/replays and (later) the unit
//! serialized to disk. Every action carries a globally-unique [`ActionId`] so
//! the same records work unchanged in a future replicated, multi-peer log
//! (§4, §12) — we pay that tiny cost from the first commit.

use serde::{Deserialize, Serialize};

use super::brush::BrushParams;
use super::filter::Filter;
use super::guide::{GuideId, PerspectiveGuide};
use super::layer::{BlendMode, LayerId, MatteRegion, Place};
use super::paint::Parcel;
use super::selection::SelectionOp;
use crate::Srgb;
use crate::clamp01;
use crate::geom::Vec2;
use crate::{SubstrateId, SubstrateScale};

/// Identifies the author of an action: one local user, or a peer (§4).
/// Maps to an iroh `NodeId` when collaborating; a fixed value when solo.
#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    carbonite::Schema,
)]
pub struct ActorId(pub u64);

impl ActorId {
    /// The fixed author id used when not collaborating. When a document is
    /// first shared, its solo-authored actions are rewritten to the sharer's
    /// real actor id (so the sharer can still undo them, §12.3);
    /// after that every action in a shared log carries a peer-derived id.
    pub const SOLO: ActorId = ActorId(0);
}

/// Globally-unique action id; also the total order key `(lamport, actor)`.
#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    carbonite::Schema,
)]
pub struct ActionId {
    pub lamport: u64,
    pub actor: ActorId,
}

/// A fully-recorded stroke: enough to replay it bit-for-bit (§4).
///
/// Deliberately does **not** carry the tool. Only the brush tool can reach a
/// stroke — the selection tools produce a [`SelectionOp`] instead — so the field
/// held one value for every stroke of every document and no reader ever asked it
/// (§8, wire version 5). A tool worth recording would be recorded by whatever
/// distinguishes it, which this enum does not.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
pub struct StrokeRecord {
    pub layer: LayerId,
    pub brush: BrushParams,
    /// The fitted stroke curve: the control points the raw pointer samples were
    /// smoothed and simplified down to (§6.2), an order of magnitude
    /// fewer points and all that is needed to reconstruct the stroke. The raw
    /// samples are never stored — not in the file, not in the action log, not
    /// on the wire.
    pub path: Vec<crate::path::ControlPoint>,
    /// The seed every per-stroke randomness derives from (§6.2): the
    /// color-dynamics field baked for the stroke and the deposit jitter's gate
    /// are each their own draw off it, so replay reproduces both exactly. A
    /// fresh one per stroke — the document clock at the press — unless a caller
    /// pins it to re-render one stroke under the same jitter
    /// (`Engine::replay_stroke_seeded`).
    pub seed: u64,
    /// Where on [`path`](Self::path)'s curve the stroke itself begins — a curve
    /// parameter in span units, `0 ≤ start ≤` the path's span count (§6.2).
    ///
    /// The curve may extend *before* the press: the hover trail the engine was
    /// already watching becomes the stroke's **run-up**, fitted into the same
    /// curve so the entry's direction and curvature are measured from real
    /// motion rather than guessed from the first tolerance-quantized steps. This
    /// marker records where on that curve the press really happened, and the
    /// deposit begins exactly there — everything before it is evidence, never
    /// paint. Rendering honours it in one place (the flattening funnel,
    /// `stark-engine`'s `generate_segments_in`), so replay, live preview and
    /// peers cannot disagree about it.
    ///
    /// `0` — the curve's own head — for a stroke with no run-up, which is also
    /// what a file from before this field existed means by its absence: the
    /// whole curve is the stroke, and such files replay bit-identically.
    #[serde(default)]
    pub start: f32,
}

/// What an action does to the document.
///
/// # Changing this enum
///
/// This is the document's vocabulary, so it is also the thing every saved file is
/// read against — and the rules for changing it are **not** the ones that used to
/// hang off nearly every variant below (§8). A file carries the schema it was written
/// with, and loading reconciles it against this enum *by name*, so:
///
/// - A **new variant** goes wherever it reads best. Nothing has to be appended, and a
///   file written before it existed simply never contains one.
/// - A **new field** on an existing variant needs `#[serde(default)]`, which is what
///   an older file's missing column is filled from. Without it, that file fails to
///   load and says which field it wanted.
/// - A **renamed** field or variant needs `#[serde(alias = "…")]` to keep older files
///   readable; a renamed one without an alias is a break.
/// - A **removed** field is skipped by readers that no longer declare it.
/// - A variant's **shape** may change — every shape is a product of its fields in
///   order, so a unit may gain fields and a payload may be taken away. Moving to
///   *named* fields needs `#[serde(alias = "0")]` on the field taking each position.
///
/// ## Retiring an action: tombstone it, never delete it
///
/// **A variant removed from this enum makes every file that ever used it
/// unloadable** — and not just that action: the whole log is one value, so one
/// retired action in ten thousand refuses the document with `unknown variant`.
/// That is the one change the encoding cannot absorb, and it is the reverse of the
/// intuition the rules above build, which is why it is spelled out here.
///
/// §19's beta rung promises old files keep opening while promising nothing about what
/// they produce, so retiring an action is spelled as **keeping the variant and taking
/// away what it does**:
///
/// 1. Keep the variant, under its own name.
/// 2. Hollow the payload out to what is still read — often `{}`, since a reader
///    steps over columns no field claims. The types behind it can go.
/// 3. Make it a no-op in `DocState::fold`, and give it an empty [`Footprint`], so it
///    reads and writes nothing and commutes with everything.
///
/// **Keep any field that is load-bearing outside the fold**, which is the trap here
/// rather than the payload's size. An `Add…` variant's `id` is one: every later
/// reference to that layer — every stroke on it, every move of it, every merge into
/// it — is resolved against it, so a tombstone that dropped its `id` would leave the
/// rest of the log naming a layer nothing had introduced (§17.9). It is also what
/// [`minted_layers`](Self::minted_layers) reads, and so what the mint door checks
/// itself against.
///
/// **A tombstone is a wire change, not only a file change.** Two peers where one
/// still applies the action and the other ignores it diverge silently, and pixels
/// cannot show which path ran (§12.6). So retiring an action bumps the ALPN, exactly
/// as reshaping anything gossip touches does (`stark-net::codec`).
///
/// What has not changed is that the *meaning* of a live variant is fixed. Reusing a
/// name for something else, or narrowing what a field may hold, is a change no
/// encoding can absorb — replay would put back a different picture, which no file can
/// notice on its own.
///
/// [`Footprint`]: super::footprint::Footprint
#[derive(Clone, Debug, Serialize, Deserialize, carbonite::Schema)]
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
    ///
    /// **`carried` names the rest of the subtree**, for the reason
    /// [`DuplicateLayer`](Self::DuplicateLayer)'s `ids` do and
    /// [`MergeLayerDown`](Self::MergeLayerDown)'s `dest` does: a [`Footprint`] is
    /// built from the action alone and cannot walk the tree for what a group held
    /// (§12.6). Carried as a tuple variant it declared `id` and `StackOrder` and
    /// wrote the existence, the paint and every property of layers it never named —
    /// so a stroke inside the group was judged to *commute* with removing it, and
    /// the fast-path undo put the pre-stroke subtree back while a canonical replay
    /// kept the paint. Peers diverged, and §12.6's whole point is that no pixel can
    /// say which materialization ran.
    ///
    /// Root first, then depth-first in composite order — the order
    /// [`DocState::visit`] produces, so minting one is a walk and checking one is a
    /// comparison.
    ///
    /// **Deterministically declined when the subtree is not what it names**, which is
    /// what a concurrent add into the group looks like from here: `DuplicateLayer`
    /// declines the same way, for the same reason, and every peer declines the same
    /// action. A file written before this field existed carries an empty list, and a
    /// group removal in one is therefore declined rather than silently taking layers
    /// nothing declared — the honest reading, since the log genuinely does not say
    /// what came out. A *leaf* removal is unaffected, which is nearly all of them.
    ///
    /// [`Footprint`]: super::footprint::Footprint
    /// [`DocState::visit`]: crate::document::Materialize
    RemoveLayer {
        #[serde(alias = "0")]
        id: LayerId,
        /// Every layer the group carries, at any depth — empty for a leaf, and for
        /// a file older than this field.
        #[serde(default)]
        carried: Vec<LayerId>,
    },
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
    /// and only ever materializes those.
    Undo(ActionId),

    /// Switch the canvas substrate (§6.4).
    ///
    /// Logged rather than kept as a view setting because the substrate feeds the
    /// document: which canvas a piece was painted on is part of what it is, and
    /// replay has to reconstruct it. A document saved before this existed contains
    /// none, and keeps the substrate from `CanvasMeta`.
    #[serde(alias = "SetSurface")]
    SetSubstrate(SubstrateId),

    /// Lay the canvas substrate at a different scale (§6.4).
    ///
    /// Logged for exactly the reason [`SetSubstrate`](Self::SetSubstrate) is, and it is
    /// the same fact in two halves: the tooth reads the substrate's rise over a reach
    /// measured in **canvas px**, so how large the substrate is laid decides what a tip
    /// bites as surely as which substrate it is. A stroke replayed from before this
    /// deposits at the scale it was painted at. A document saved before this existed
    /// contains none and stands at [`SubstrateScale::NATURAL`].
    #[serde(alias = "SetSurfaceScale")]
    SetSubstrateScale(SubstrateScale),

    /// Edit the selection mask (§6.8). Historized because a stroke's
    /// pixels depend on the mask in force when it was drawn — replaying the log has
    /// to put the same mask back. Only the **op** travels (a few floats, or a
    /// decimated polyline); every peer rasterizes it identically from the same
    /// shader, so the log stays compact and convergence is unaffected.
    Select(SelectionOp),
    /// Swap selected for unselected everywhere (§6.8).
    InvertSelection,
    /// Set the **whole** mask's opacity, on top of the shape arithmetic (§6.8) —
    /// the Select panel's Opacity slider.
    ///
    /// Historized for [`Select`](Self::Select)'s reason and no other: a stroke's
    /// pixels depend on how strongly the mask gated it, so replay has to put the
    /// same number back. It carries no shape and rasterizes nothing — the mask
    /// tiles are whatever the ops made them, and this is how strongly they are
    /// *read*, which is exactly what lets it apply to a region already drawn. The
    /// per-shape [`SelectionOp::opacity`] is the same question asked of one shape
    /// and still stands underneath this; the two multiply.
    SetSelectionOpacity(f32),

    /// Add a **matte** layer — a region filled with a [`Parcel`]
    /// (§15.2). A frame is one of these on top of the stack; a substrate
    /// ([`MatteRegion::Everything`]) is one at the bottom, which is why the
    /// anchor is the full [`Place`] where `AddLayer`'s stays the two-state
    /// `Option` (§15.5). The same action serves comic gutters once the region
    /// algebra lands (P4).
    AddMatte {
        id: LayerId,
        carrier: Option<LayerId>,
        at: Place,
        region: MatteRegion,
        paint: Parcel,
    },
    /// Move a matte's rect — the frame drag's commit. One action per drag, not
    /// per pointer move: the gesture accumulates in session state and commits on
    /// release, so fifty tweaks are fifty undo steps rather than five thousand.
    /// A no-op on a region with no rect to move ([`MatteRegion::with_rect`]).
    SetMatteRect(LayerId, Vec2, Vec2),
    /// Repaint a matte — a flat color or a gradient ramp (§15.4, §22.4).
    SetMattePaint(LayerId, Parcel),
    /// Set the canvas substrate color — the substrate the paint sits on, straight
    /// sRGB (§15.5). Logged rather than held as a view setting, because the substrate a
    /// piece was painted on is part of what it is: unlogged, the paper color of a
    /// painting would not be saved at all.
    #[serde(alias = "SetBackground")]
    SetSubstrateColor(Srgb),

    /// Affine transform of the selected paint on `layer` (§16):
    /// cut what the **author's** selection holds, resample it once under
    /// `affine`, stack it back over what remained — and carry the author's mask
    /// along with it, so the moved region stays selected. A universal selection
    /// moves the whole layer. Six floats in the log; every peer re-derives the
    /// same tiles from them.
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
    /// exceed [`MAX_FILL_TILES`](super::fill::MAX_FILL_TILES).
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
    /// a few dozen floats — and every peer subdivides it identically.
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
    /// into the group looks like from here.
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
    /// it sits is its scope*.
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
    /// **The pixels are not the payload.** The picture is named by content id, like
    /// a stamp brush's shape, and travels beside the log — bundled in
    /// `DocumentFile::content`, and over the wire on the blob ALPN (§23). It was
    /// briefly built the other way, with the image in the action behind an `Arc`;
    /// `docs/images.md` records why that was wrong in three places at once, of which
    /// the sharpest is that an action is *cloned constantly* — a commit clones one
    /// for the outbox, the history clones them while splicing an undo past what it
    /// commutes with (§12.6) — and every one of those copies is thirty-two bytes now.
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
    /// when the placement falls off the tile grid an `i32` can address.
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
    /// Add a **drawing guide** — a perspective grid to construct through
    /// (§20.5).
    ///
    /// Logged like a layer, and for the same kind of reason `SetSubstrate` and
    /// `SetSubstrateColor` are: a perspective set up over a drawing is part of the
    /// drawing's construction, not a preference about how it is being looked at.
    /// Unlogged, it would be lost on reload and invisible to collaborators, and
    /// a scaffold the work is built on is worth exactly as much care as a layer.
    ///
    /// **It mints no id.** A guide's identity is the id of *this* action
    /// ([`GuideId`]), so there is no counter to partition, nothing for
    /// `minted_layers` to report, and no way for two peers adding at once to
    /// land on one id. What makes that available here and not to `AddLayer` is
    /// that this mints exactly one thing — see [`GuideId`].
    ///
    /// `after` names the guide it lands directly after in the roster, or the
    /// **head** of it when `None`. A flat list of `n` guides has `n + 1` places
    /// to land in and this reaches every one of them, which is why guides need
    /// no [`Place`] where a layer stack does: `Place` exists because a stack's
    /// two-state anchor could not say "under the bottom layer", and `None`
    /// meaning the head is that same third state spelled without a type.
    ///
    /// The name arrives with the guide rather than in a following
    /// [`SetGuideName`](Self::SetGuideName) for
    /// [`PlaceImage`](Self::PlaceImage)'s reason: duplicating a guide copies the
    /// artist's own word for it, and spelling that as two actions would put one
    /// gesture two undo steps deep with a nameless guide in between.
    AddGuide {
        /// The id this guide gets — **the id of this very action**, minted through
        /// the same door every layer id is (`Engine::commit_minting`, §17.9).
        ///
        /// Carried rather than derived inside the fold, which is where it was. A
        /// derived id is not part of the action, so `start_collaboration`'s rewrite of
        /// solo-authored `ActionId`s moved it while every `RemoveGuide`,
        /// `SetGuide`, `SetGuideName` and `MoveGuide` in the same log went on naming
        /// the old one — and each of those no-ops on an id it cannot find. Sharing a
        /// document therefore reverted every guide edit made before it and brought
        /// back every deleted guide. A payload the rewrite does not touch cannot do
        /// that, which is what `LayerId` had all along.
        id: GuideId,
        guide: PerspectiveGuide,
        /// The guide this one lands directly after, or the head of the roster
        /// when `None`.
        after: Option<GuideId>,
        /// What to call it, or `None` to leave it described by its place in the
        /// roster ("Perspective 2"). A `String` rather than the `Arc<str>` the
        /// state holds, exactly as [`SetLayerName`](Self::SetLayerName) carries
        /// one: this is the file and wire form, where a shared pointer means
        /// nothing.
        name: Option<String>,
    },
    /// Remove a drawing guide (§20.5). A no-op on a guide that is not there,
    /// which is what a concurrent removal looks like from here.
    RemoveGuide(GuideId),
    /// Reshape a guide — the **whole camera** at once (§20.5).
    ///
    /// One action for the orbit drag, the lens drag, the crosshair, the cell
    /// slider, the opacity slider, the plane chips and the fisheye toggle,
    /// carrying the camera entire for the reason [`SetFilter`](Self::SetFilter)
    /// carries the whole filter: a guide that grows a knob then needs no new
    /// action and no wire-format break, and there is nothing finer than the
    /// camera that an artist can be said to have set.
    ///
    /// **One per settled gesture, not one per pointer move.** A drag previews in
    /// view state and commits on release — the bargain
    /// [`SetMatteRect`](Self::SetMatteRect) and
    /// [`SetFilter`](Self::SetFilter) already strike (§15.7, §21.5) — so
    /// shaping a perspective costs one undo step per adjustment rather than one
    /// per sample of the hand.
    SetGuide(GuideId, PerspectiveGuide),
    /// Name a guide, or with `None` take its name away so it falls back to being
    /// described by its place in the roster.
    ///
    /// Its own action rather than a field of [`SetGuide`](Self::SetGuide),
    /// exactly as [`SetLayerName`](Self::SetLayerName) is not a field of the
    /// camera: the name is held as an `Arc<str>` in state and a `String` here,
    /// and a rename must commute with a drag of the same guide the way a layer's
    /// rename commutes with its opacity.
    SetGuideName(GuideId, Option<String>),
    /// Move a guide within the roster — the panel's drag-to-reorder (§20.5).
    ///
    /// `after` is [`AddGuide`](Self::AddGuide)'s anchor, read the same way. The
    /// roster's order changes no pixel: every guide this client draws is drawn,
    /// whatever order they are listed in. It is logged anyway because it is the
    /// artist's own arrangement of their scaffolding, which is the same thing a
    /// layer's name is and is kept for the same reason.
    MoveGuide {
        id: GuideId,
        after: Option<GuideId>,
    },
}

impl ActionKind {
    /// Every layer id this action **mints** — the ids a client's counter has to
    /// mints, which the engine's mint door checks are its own (§17.9).
    ///
    /// **The check it exists for**: a `LayerId` is the id of the action that minted
    /// it, and `Engine::commit_minting` is where that is arranged — it draws the
    /// action id, hands it to the closure that builds the kind, and then asks this
    /// which layers the kind claims to mint and whether they all name that id. What
    /// the door can enforce, the shape need not be trusted about.
    ///
    /// Lives here, beside the variants, because it is a fact about *them*: minting
    /// is what an `Add…` action does, and a caller that keeps its own list of which
    /// ones do has a list a new variant does not appear in. The engine kept exactly
    /// such a list once, `AddFilter` was added after it, and a document whose highest
    /// id came from a filter reloaded with a counter that would mint that id a second
    /// time — two layers under one id, the convergence failure §17.9 is about.
    ///
    /// **Exhaustive, with no `_` arm, and that is the whole point of it.** A variant
    /// added to the enum stops this function compiling, three lines from the doc
    /// comment that says why — the device `slot` in `tests/footprint.rs` already
    /// uses, for the same reason and after the same variant escaped it.
    ///
    /// Note it reports what the action *names as minted*, not what applying it
    /// lands: a rejected `AddLayer` (unknown carrier) inserts nothing, and the id it
    /// named belongs to it regardless — which is the honest answer, since the same
    /// action accepted by a peer does insert under exactly that id.
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
            | ActionKind::RemoveLayer { .. }
            | ActionKind::SetLayerBlend(..)
            | ActionKind::SetLayerClip(..)
            | ActionKind::SetLayerOpacity(..)
            | ActionKind::SetLayerVisible(..)
            | ActionKind::SetLayerName(..)
            | ActionKind::MoveLayer { .. }
            | ActionKind::MergeLayerDown { .. }
            | ActionKind::Undo(_)
            | ActionKind::SetSubstrate(_)
            | ActionKind::SetSubstrateScale(_)
            | ActionKind::Select(_)
            | ActionKind::InvertSelection
            | ActionKind::SetSelectionOpacity(_)
            | ActionKind::SetMatteRect(..)
            | ActionKind::SetMattePaint(..)
            | ActionKind::SetFilter(..)
            | ActionKind::SetSubstrateColor(_)
            | ActionKind::Transform { .. }
            | ActionKind::TransformPerspective { .. }
            | ActionKind::TransformWarp { .. }
            | ActionKind::Fill { .. }
            // A guide mints an id of its own, and deliberately not from a counter
            // this reports for: it *is* the id of the action that added it, so
            // there is nothing to resume past (`GuideId`).
            | ActionKind::AddGuide { .. }
            | ActionKind::RemoveGuide(_)
            | ActionKind::SetGuide(..)
            | ActionKind::SetGuideName(..)
            | ActionKind::MoveGuide { .. } => (None, &[]),
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
                // The marker's ceiling is the path's span count, which is the
                // flattening's to know — it clamps the top end itself. What the
                // record can state alone is held here: finite, and not before
                // the curve it marks a point on.
                start: if rec.start.is_finite() {
                    rec.start.max(0.0)
                } else {
                    0.0
                },
                ..rec
            }),
            // A coverage weight every gating read multiplies by, so a NaN here would
            // take the whole mask with it. `clamp01` rather than `clamp`, for
            // `SelectionOp::at`'s reason: both of NaN's comparisons are false.
            ActionKind::SetSelectionOpacity(a) => ActionKind::SetSelectionOpacity(clamp01(a)),
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
            // A layer's opacity is a coverage weight the compositor multiplies by;
            // a NaN would take the whole layer with it. Not `clamp01`, whose NaN
            // lands on 0 — an opacity that says nothing should leave the layer
            // *visible*, which is the neutral setting rather than the near end.
            ActionKind::SetLayerOpacity(id, a) => {
                ActionKind::SetLayerOpacity(id, crate::finite_in(a, 1.0, (0.0, 1.0)))
            }
            // A guide's every number reaches the guide pass's uniform, and now
            // the saved log as well — `PerspectiveGuide::sanitized` is where what
            // each of them may hold is written down (§20.5).
            ActionKind::AddGuide {
                id,
                guide,
                after,
                name,
            } => ActionKind::AddGuide {
                id,
                guide: guide.sanitized(),
                after,
                name,
            },
            ActionKind::SetGuide(id, guide) => ActionKind::SetGuide(id, guide.sanitized()),
            // A matte's paint is a color or a ramp on an axis, and every one of
            // those numbers reaches `matte.wesl` — the same payload a `Fill` carries
            // and, until this arm existed, the one that carried it *past* the
            // funnel. The region beside it is geometry, gated at `apply` by
            // `MatteRegion::usable` rather than clamped, for the reason the
            // transforms are.
            ActionKind::SetMattePaint(id, paint) => {
                ActionKind::SetMattePaint(id, paint.sanitized())
            }
            ActionKind::AddMatte {
                id,
                carrier,
                at,
                region,
                paint,
            } => ActionKind::AddMatte {
                id,
                carrier,
                at,
                region,
                paint: paint.sanitized(),
            },
            // Nothing to hold: ids, flags, places, and the geometry whose own
            // `usable`/`affine_usable` gate rejects it at `apply` rather than
            // rounding it into something else (§16.1) — a transform that cannot be
            // clamped into a *different* transform without changing what the
            // author asked for, and a frame rect that cannot either
            // (`MatteRegion::usable`).
            //
            // **The list is what the compiler cannot check.** Every arm above says
            // "this payload holds its own invariant"; this one says "there is no
            // invariant to hold", and nothing but the reader tells the two apart. It
            // is where `SetSubstrateColor`, `SetMattePaint` and `AddMatte` sat while
            // carrying colors to a shader, under a comment that named gates only the
            // three transforms have. So: a variant belongs here when its payload is
            // ids, flags, places, `bool`s and `String`s — and if it carries a float,
            // it belongs above, or beside a `usable` this comment can name.
            // A fill and a selection carry numbers and are still here, which is the
            // shape worth noticing rather than an oversight. Their fields are
            // private and `FillOp::with_paint` / `SelectionOp::at` are the only
            // doors — deserialization included — so an op in hand already holds its
            // bounds and there is nothing for a second gate to do. These arms used
            // to rebuild each op through its own constructor to say that; the
            // constructor now says it, the way `Filter::sanitized`'s gradient arm
            // stopped having a body once `Srgb` held the cube.
            ActionKind::Fill { .. }
            | ActionKind::Select(_)
            | ActionKind::AddLayer { .. }
            // A placement carries pixels and an integer position: no float to be
            // non-finite, no knob to be out of range. What *could* be malformed about
            // an image — dimensions that disagree with the buffer, a size past the cap
            // — cannot be clamped into a different image, and is refused at
            // `ImageRef`'s constructor and at its decode instead (§23).
            | ActionKind::PlaceImage { .. }
            | ActionKind::DuplicateLayer { .. }
            | ActionKind::RemoveLayer { .. }
            | ActionKind::MergeLayerDown { .. }
            | ActionKind::MoveLayer { .. }
            | ActionKind::SetLayerClip(..)
            | ActionKind::SetLayerVisible(..)
            | ActionKind::SetLayerName(..)
            // The rect a frame is dragged to, gated by `MatteRegion::usable` at
            // `apply` beside the three transforms below rather than rounded into a
            // different rectangle. Its *paint* is sanitized above.
            | ActionKind::SetMatteRect(..)
            // The substrate color, which holds itself now: `Srgb` cannot be built
            // outside the cube, so this is back to being an arm with nothing in it
            // — the shape the comment above describes.
            | ActionKind::SetSubstrateColor(_)
            | ActionKind::SetSubstrate(_)
            // And the scale it is laid at, which holds itself the same way: a
            // `SubstrateScale` off the ladder or outside the range cannot be built, so
            // there is nothing here to hold it to.
            | ActionKind::SetSubstrateScale(_)
            | ActionKind::InvertSelection
            | ActionKind::Transform { .. }
            | ActionKind::TransformPerspective { .. }
            | ActionKind::TransformWarp { .. }
            // A guide named, moved or removed carries an id, an anchor and a
            // name; the camera is the only thing with a number in it.
            | ActionKind::RemoveGuide(_)
            | ActionKind::SetGuideName(..)
            | ActionKind::MoveGuide { .. }
            | ActionKind::Undo(_) => self,
        }
    }

    /// What this action *is*, in two or three words — see [`ActionTag::label`].
    ///
    /// Delegated rather than matched again: the caption belongs to the *kind* and not
    /// to any payload, so it lives on the roster with the rest of what a kind is.
    pub fn label(&self) -> &'static str {
        self.tag().label()
    }
}

/// The document's vocabulary **as a roster**: one tag per [`ActionKind`], with no
/// payload — what an action *is*, apart from what it says.
///
/// # Why this exists
///
/// `ActionKind` is matched exhaustively in a handful of places, and that is working
/// as intended: a new variant does not compile until it says what it mints (§17.9),
/// what it clamps (§21.5), what it reads and writes (§12.6) and what content it
/// names (§23). Those are four different questions and they deserve four answers.
///
/// What did *not* deserve four answers was the roster itself. The enum's list of
/// members was written out four times — `label`'s match here, and `LABELS`, `KINDS`
/// and `slot` in `stark-testdata::vocabulary`, across a crate boundary — and only
/// one of the four was compiler-checked. `vocabulary`'s own header records what that
/// cost: `SetSelectionOpacity` got the arm the compiler demanded in each `slot` and
/// was left out of every list those arms index, in two crates at once, and both
/// suites went on passing having never driven it.
///
/// So the roster is **one list**, below, and everything else is derived from it: the
/// enum, [`ALL`](ActionTag::ALL), and [`label`](ActionTag::label) all come out of the same
/// macro invocation, and [`ActionKind::tag`] is the one exhaustive match that binds
/// a variant to its tag. A new kind now fails to compile in exactly one place it did
/// not before, and cannot be missing from a list at all, because there is no second
/// list to be missing from.
macro_rules! roster {
    ($($variant:ident => $label:literal,)*) => {
        #[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub enum ActionTag { $($variant,)* }

        impl ActionTag {
            /// Every tag there is, in [`ActionKind`]'s declared order.
            ///
            /// Nothing rests on the order — a tag is identified by name, and the
            /// indices are private to whoever iterates — but a roster wants *an*
            /// order, and the enum's is the one already written down.
            pub const ALL: &'static [ActionTag] = &[$(ActionTag::$variant,)*];

            /// What this kind of action is, in two or three words — the caption a
            /// history scrubber puts on the step it is about to cross (§18.2.4).
            ///
            /// A `&'static str` and nothing more: a timeline showing a hundred steps
            /// needs them by the hundred, and the point is to tell a stroke from a
            /// layer change at a glance, not to describe either. Anything richer —
            /// which layer, what color — is what the canvas beside it is for.
            pub fn label(self) -> &'static str {
                match self { $(ActionTag::$variant => $label,)* }
            }
        }

        impl ActionKind {
            /// Which kind of action this is, without its payload.
            ///
            /// **Exhaustive, with no `_` arm** — the one place a new variant has to
            /// name itself, and the reason the roster above can be trusted to be
            /// complete.
            pub fn tag(&self) -> ActionTag {
                match self { $(ActionKind::$variant { .. } => ActionTag::$variant,)* }
            }
        }
    };
}

roster! {
    CommitStroke => "Stroke",
    AddLayer => "Add layer",
    RemoveLayer => "Remove layer",
    SetLayerBlend => "Blend mode",
    SetLayerOpacity => "Layer opacity",
    SetLayerVisible => "Layer visibility",
    MoveLayer => "Reorder layer",
    Undo => "Undo",
    SetSubstrate => "Canvas substrate",
    SetSubstrateScale => "Substrate scale",
    Select => "Select",
    InvertSelection => "Invert selection",
    SetSelectionOpacity => "Selection opacity",
    AddMatte => "Add matte",
    SetMatteRect => "Move frame",
    SetMattePaint => "Matte paint",
    SetSubstrateColor => "Canvas color",
    Transform => "Transform",
    SetLayerName => "Rename layer",
    Fill => "Fill",
    SetLayerClip => "Clip layer",
    TransformPerspective => "Perspective",
    TransformWarp => "Warp",
    DuplicateLayer => "Duplicate layer",
    AddFilter => "Add filter",
    SetFilter => "Filter",
    MergeLayerDown => "Merge down",
    PlaceImage => "Place image",
    AddGuide => "Add guide",
    RemoveGuide => "Remove guide",
    SetGuide => "Perspective guide",
    SetGuideName => "Rename guide",
    MoveGuide => "Reorder guide",
}

impl ActionTag {
    /// Where this tag sits in [`ALL`](Self::ALL) — a stable index for a suite that
    /// wants the vocabulary as an array.
    pub fn index(self) -> usize {
        Self::ALL
            .iter()
            .position(|t| *t == self)
            .expect("every tag is in ALL, which the macro builds from one list")
    }
}

/// A committed document mutation with its identity.
#[derive(Clone, Debug, Serialize, Deserialize, carbonite::Schema)]
pub struct Action {
    pub id: ActionId,
    pub kind: ActionKind,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{FillOp, Parcel, SelectionMode, SelectionShape};

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
            layer: LayerId::ROOT,
            brush: BrushParams {
                size: bad,
                tooth: crate::document::brush::ToothParams {
                    give: 9.0,
                    ..Default::default()
                },
                ..BrushParams::default()
            },
            path: Vec::new(),
            seed: 0,
            // A marker before the curve names no place; the funnel floors it.
            start: -2.0,
        })
        .sanitized();
        let ActionKind::CommitStroke(rec) = &stroke else {
            panic!("a stroke stays a stroke")
        };
        assert!(rec.brush.size.is_finite());
        assert_eq!(rec.brush.tooth.give, 1.0);
        assert_eq!(rec.start, 0.0);

        // A fill and a selection are **not** repaired by the funnel, and no longer
        // can be: their fields are private, so the hostile values below cannot get
        // past the constructor to reach it. The op is already in range on the way
        // in, and `sanitized` leaves it exactly as it is — which is the whole of
        // what an arm here used to be doing by rebuilding it.
        let op = FillOp::with_paint(
            SelectionShape::All,
            -1.0,
            Parcel::Solid(Srgb::new([bad; 3])),
            5.0,
        );
        assert_eq!((op.feather(), op.opacity()), (0.0, 1.0), "held at the door");
        let fill = ActionKind::Fill {
            layer: LayerId::ROOT,
            op: op.clone(),
        }
        .sanitized();
        let ActionKind::Fill { op: after, .. } = &fill else {
            panic!("a fill stays a fill")
        };
        assert_eq!(after, &op, "and the funnel has nothing left to move");

        let op = SelectionOp::at(SelectionMode::Replace, SelectionShape::All, bad, 0.5);
        assert_eq!((op.feather(), op.opacity()), (0.0, 1.0), "held at the door");
        let select = ActionKind::Select(op.clone()).sanitized();
        let ActionKind::Select(after) = &select else {
            panic!("a select stays a select")
        };
        assert_eq!(after, &op, "and the funnel has nothing left to move");

        // A layer's opacity is a coverage weight the compositor multiplies by.
        let ActionKind::SetLayerOpacity(_, a) =
            ActionKind::SetLayerOpacity(LayerId::ROOT, bad).sanitized()
        else {
            panic!()
        };
        assert!(a.is_finite());
    }
}
