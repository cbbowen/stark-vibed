//! Action footprints: which parts of the document an action reads and writes
//! (§12.6).
//!
//! Two actions **commute** — applying them in either order produces the same
//! state — when neither writes anything the other reads or writes. The
//! history uses that (via the [`Centralizer`](history::Centralizer) impl
//! below) to service an undo by shifting the undone action out of the
//! materialization instead of replaying everything after it: strokes on
//! different layers commute, strokes on the same layer commute when their
//! padded extents don't touch, a rename commutes with nearly everything.
//!
//! Footprints are **conservative**: a stroke claims the whole tile-aligned
//! bounding box of its path (padded past any reach of the tip), a transform
//! claims its entire layer, and every structural edit claims the shared stack
//! order. A false conflict only costs the fast path; a missed one would
//! silently diverge peers, so every `apply` in `action.rs` must read only what
//! its kind's footprint declares and write only what it declares — that
//! locality is what makes the splice sound (see `timeline.rs`).

use super::action::{Action, ActionKind, ActorId, StrokeRecord};
use super::brush::BrushParams;
use super::layer::LayerId;
use crate::geom::{TileRect, Vec2};

/// The tiles a pass may touch within the canvas box `[lo, hi]`, grown by `ring`
/// tiles — [`TileRect::covering`] with a footprint's answer to a box it cannot
/// measure.
///
/// That answer is always **everything**. A footprint may only ever claim too
/// much (§12.6): a false conflict costs the commutation fast path, while a
/// missed one silently diverges peers with no pixel able to show which
/// materialization ran. So an unquantizable box claims the whole layer, exactly
/// as an unusable warp's unknown image does in [`gated_rect`].
fn claim(lo: Vec2, hi: Vec2, ring: i32) -> TileRect {
    TileRect::covering(lo, hi, ring).unwrap_or(TileRect::ALL)
}

/// A per-layer property, at the granularity undo needs to restore it: each
/// variant is overwritten wholesale by the actions that write it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Prop {
    Blend,
    /// Whether the layer clips to the paint beneath it (§14.4).
    /// Its own resource beside `Blend` rather than folded into it: the two are
    /// applied at the same step but written by different actions, and a clip
    /// toggle has to commute with a blend change on the same layer.
    Clip,
    Opacity,
    Visible,
    Name,
    /// A matte's region *and* color — split no finer because no action writes
    /// one without the freedom to write the other.
    Matte,
    /// A filter layer's settings (§21). One resource for the whole filter, for the
    /// reason [`Matte`](Self::Matte) is one for a region and a fill: `SetFilter`
    /// carries the filter entire, so there is no finer thing an action can write.
    Filter,
}

impl Prop {
    /// Every property, which is what [`Resource::Layer`] stands for the paint and the
    /// existence of.
    ///
    /// Hand-written, and held honest by `every_prop_is_named_in_all` below: Rust
    /// cannot enumerate an enum's variants, so what is available is a match that
    /// stops compiling until a new variant is *visited* — the device
    /// `Modulations::all` uses for a struct's fields. A `Prop` missing here would
    /// make a coarse claim quietly finer than it says it is.
    pub const ALL: [Prop; 7] = [
        Prop::Blend,
        Prop::Clip,
        Prop::Opacity,
        Prop::Visible,
        Prop::Name,
        Prop::Matte,
        Prop::Filter,
    ];
}

/// One addressable piece of document state.
///
/// **The vocabulary is closed over the log, with one exception.** Every resource
/// here names something an action carries or something the fold built from actions
/// before it, so two peers holding the same log agree about all of it. `PlaceImage`
/// is the exception: its tiles are built from a picture held in an out-of-log store
/// under the id the action carries (§23.2), so a peer that has not received the
/// picture folds the same action into an empty layer. There is no resource to name
/// for it — the divergence is not between two orders of the same actions but between
/// two peers' stores, which is a transport contract (§12.4) — and it is said here
/// because this list is where a reader comes to ask what determinism rests on.
#[derive(Clone, Debug, PartialEq)]
pub enum Resource {
    /// A layer's painted tiles within a tile rect.
    Paint(LayerId, TileRect),
    /// A layer's presence in the document at all. Every action that targets a
    /// layer reads this (they all no-op on an absent layer, and that no-op is
    /// order-dependent against add/remove).
    ///
    /// **It stands for the layer's *minted* kind as well** — paint, matte, filter —
    /// which several arms read while declaring nothing else: `cannot_carry` refuses a
    /// filter as a carrier (§21.2), `set_layer_blend` refuses a filter. That is sound
    /// only because those kinds are fixed when the layer is minted and no action
    /// changes them, so reading one is reading the same fact this resource already
    /// covers. An action that *converted* a layer between them would break it and
    /// needs a resource of its own: it would be a write no reader of this one sees.
    ///
    /// **Whether a layer is a *group* is not one of them** and must not be read this
    /// way. Carrying is structure, not kind — a leaf becomes a group the moment
    /// `MoveLayer` puts something under it — so what covers it is
    /// [`StackOrder`](Self::StackOrder), which is what `MergeLayerDown` writes before
    /// it asks.
    Existence(LayerId),
    /// One presentation property of a layer.
    Prop(LayerId, Prop),
    /// The **shape of the whole layer tree** — every stack's order and who
    /// carries whom (§14.8). One coarse resource: two concurrent
    /// restructures genuinely don't commute, and structural edits are rare
    /// enough that finer granularity would buy nothing.
    ///
    /// Nesting rides on it unchanged, which is the point. It is also what makes
    /// the carry-your-own-ancestor case safe without tree-CRDT machinery: two
    /// halves of a cycle conflict here, so the log's total order serializes them
    /// and the second one to apply sees the first's result and declines.
    StackOrder,
    /// **Everything about one layer**: its existence, all of its paint, and every
    /// one of its properties.
    ///
    /// The coarse resource [`StackOrder`](Self::StackOrder) is for the tree, and it
    /// earns its place the same way: two actions genuinely read a whole layer, and
    /// spelling that out finely is nine resources that always travel together.
    /// `DuplicateLayer` copies every tile and every property of every layer in a
    /// subtree, and `MergeLayerDown` is a function of everything about both sides.
    /// Spelled finely that is nine and five entries a layer, which makes
    /// [`Footprint::conflicts`] — a nested scan — quadratic in a number that has no
    /// business being large: a twenty-layer duplicate would claim 180 read
    /// resources.
    ///
    /// It says the same thing. A coarse claim is *more* conservative than the fine
    /// ones it replaces, never less, and §12.6 permits a footprint to claim too much
    /// — a false conflict costs the commutation fast path, where a missed one
    /// silently diverges peers.
    Layer(LayerId),
    /// An actor's selection mask (§17.3).
    Selection(ActorId),
    /// The canvas substrate (§6.4): **which substrate, and the scale it is laid at.**
    ///
    /// One resource for the two, on [`StackOrder`](Self::StackOrder)'s argument. They
    /// are one fact about the substrate — the tooth reads the substrate's rise over a
    /// reach in canvas px, so the substrate and how large it is laid decide the deposit
    /// together — and both are chosen between passages rather than during one. What a
    /// finer split would buy is the commutation fast path between two collaborators
    /// who happened to pick a substrate and a scale at the same moment; §12.6 permits a
    /// footprint to claim too much, and this is what that permission is for.
    Substrate,
    /// The substrate color (§15.5).
    SubstrateColor,
    /// **The whole drawing-guide roster** (§20.5): every guide, everything about
    /// each of them, and the order they are arranged in.
    ///
    /// One coarse resource, on [`StackOrder`](Self::StackOrder)'s argument and
    /// [`Layer`](Self::Layer)'s. Spelling it finely — a resource per guide, plus
    /// one for the arrangement — would let two artists shape two different
    /// guides concurrently without a rebase, and that is the entire prize; a
    /// footprint may claim too much (§12.6), and what it costs here is the
    /// commutation fast path between two *guide* edits, which arrive one per
    /// settled gesture on a roster that holds a handful of entries. Nothing on
    /// the drawing path touches it: a stroke reads the guides to snap through
    /// them, but it reads them before it is an action at all, so the action it
    /// commits names this nowhere and paint and guides never contend.
    Guides,
}

impl Resource {
    /// Whether these two name any state in common — the relation
    /// [`Footprint::conflicts`] is built from.
    ///
    /// Public so a test can hold an `apply` to its declaration through the *same*
    /// predicate the timeline commutes by (`stark-engine/tests/footprint.rs`).
    /// Answering that question a second way in the test is how a coarse claim comes
    /// to look finer than it is.
    pub fn overlaps(&self, other: &Resource) -> bool {
        match (self, other) {
            (Resource::Paint(a, ra), Resource::Paint(b, rb)) => a == b && ra.intersects(rb),
            // The coarse claim meets every finer claim on the same layer — and, on
            // both sides at once, itself. A `Paint` rect is not consulted: `Layer`
            // claims all of it, so there is no box to miss.
            (Resource::Layer(id), other) | (other, Resource::Layer(id)) => {
                other.layer() == Some(*id)
            }
            _ => self == other,
        }
    }

    /// The layer this resource is about, or `None` for the ones that are about the
    /// document — the tree's shape, a mask, the canvas.
    fn layer(&self) -> Option<LayerId> {
        match self {
            Resource::Paint(id, _)
            | Resource::Existence(id)
            | Resource::Prop(id, _)
            | Resource::Layer(id) => Some(*id),
            Resource::StackOrder
            | Resource::Selection(_)
            | Resource::Substrate
            | Resource::SubstrateColor
            | Resource::Guides => None,
        }
    }
}

/// What an action touches: resources it only reads, and resources it writes
/// (which may also be read — a written resource conflicts with everything, so
/// listing it once under `writes` covers both).
#[derive(Clone, Debug, Default)]
pub struct Footprint {
    pub reads: Vec<Resource>,
    pub writes: Vec<Resource>,
}

impl Footprint {
    /// Whether the two actions may fail to commute: any write here overlapping any
    /// read *or* write there (and vice versa). Reads never conflict with reads.
    ///
    /// This is what the `history::Centralizer` impl on `&Footprint` (in the `fold`
    /// module) commutes by. Disjoint footprints satisfy that
    /// contract — neither action reads or writes anything the other writes, so
    /// applying them in either order produces the same state, and `inverse`
    /// restricted to this footprint removes exactly this action's effect. A false
    /// conflict only costs the fast path; the contract permits false negatives,
    /// never false positives.
    pub fn conflicts(&self, other: &Footprint) -> bool {
        let hits =
            |xs: &[Resource], ys: &[Resource]| xs.iter().any(|x| ys.iter().any(|y| x.overlaps(y)));
        hits(&self.writes, &other.writes)
            || hits(&self.writes, &other.reads)
            || hits(&self.reads, &other.writes)
    }
}

/// Padding around a stroke's control-point bounding box, in canvas px: the
/// farthest any of the tip's marks can land from the fitted centerline.
///
/// The B-spline stays inside its control points' convex hull, so the bbox
/// bounds the centerline exactly; the tip then reaches at most `radius`
/// scaled by √2 for a square stamp swept at an angle (1.5 covers it), and the
/// renderer refreshes a `TILE_APRON` of duplicated neighbor pixels past its
/// marks (the +4 covers that with slack).
///
/// **Times the elongation** (§6.6), because a tip drawn out along its facing axis
/// reaches that much further and the `√2` alone does not cover it. This is a footprint,
/// so under-reporting it is not a clipped stroke but a §12.6 break — two peers deciding
/// a pair of strokes commute when the paint says otherwise, and pixels cannot show
/// which order ran. The brush's own knob rather than any segment's, since a modulation
/// only ever scales it down; and `elongation` is bounded and NaN-safe, so a malformed
/// one lands on a real factor here rather than on the infinity that would quietly widen
/// this to `ALL`.
///
/// # Why the lateral flux needs no allowance here
///
/// [`BrushDynamics::bleed`](super::brush::BrushDynamics::bleed) diffuses paint
/// sideways, and the engine's longest tap reaches half the radius further
/// (`gpu::stroke::dynamics::bleed::BLEED_REACH_MAX`). Added to the `√2` above that
/// would overrun this pad outright at a large radius — 957 px against 754 at
/// `radius = 500` — so it is worth saying once why it does not, rather than leaving
/// the next reader to find the arithmetic and reach for the alarm.
///
/// **The flux cannot cross the edge of the sweep.** `dynamics.wesl`'s exchange
/// weighs every tap by `min(w_t, w_n)` — this texel's mobility and its neighbour's
/// — and `bleed_weight` writes `w = 0` for any texel outside the sweep. So a texel
/// outside is never written (its own `w_t` zeroes all of its fluxes) and a tap
/// reaching out of the sweep carries exactly nothing. The reach sets how *far
/// within* the footprint paint is carried, never how far the footprint extends: a
/// no-flux wall, at every scale, which is the same sentence the shader's own
/// comment uses about the clamped tap at the rect border.
///
/// That makes the `1.5` here cover `√2` and nothing else — the coincidence that
/// `1.0 + 0.5` is also `1.5` is exactly that, and a bleed reach raised past it
/// would still be contained. What *would* break this is the wall coming down, and
/// that is a shader invariant, checkable only where the shader is.
fn stroke_pad(brush: &BrushParams) -> f32 {
    brush.size * 1.5 * BrushParams::elongation(brush.stretch) + 4.0
}

/// The tile-aligned reach of a stroke: everything its render may read or write.
///
/// Shared with the live-preview fold (§17.6), which asks the same question of a
/// gesture that has not become an action yet: whether two people painting at once
/// are painting on the same tiles. Deliberately the *same* answer the commit's
/// footprint gives, so the fold cannot decide two strokes are independent where the
/// log would decide they conflict.
///
/// The box covers the **whole** recorded curve, run-up included: the deposit
/// begins at [`StrokeRecord::start`], so the pre-marker stretch only ever
/// *over*-declares — the safe direction (§12.6), and the price of keeping this a
/// fold over control points with no span arithmetic in the model. A run-up is
/// bounded by the hover window's own scale, so the slack is a few dozen tolerances.
pub fn stroke_rect(rec: &StrokeRecord) -> TileRect {
    let mut min = Vec2::splat(f32::INFINITY);
    let mut max = Vec2::splat(f32::NEG_INFINITY);
    for p in &rec.path {
        // Tested rather than folded in: `f32::min`/`max` return the *non*-NaN
        // operand, so a non-finite point would step straight over the bbox and
        // leave it looking tight. Records arrive from files and peers, and a
        // stroke that cannot be bounded has to claim the layer.
        if !p.pos.is_finite() {
            return TileRect::ALL;
        }
        min = min.min(p.pos);
        max = max.max(p.pos);
    }
    if min.x > max.x {
        // An empty path touches nothing.
        return TileRect::EMPTY;
    }
    // A non-finite radius makes `pad` non-finite and the box unquantizable, which
    // `covering` answers with `ALL` — the safe direction, and the one the old
    // `NaN as i32` did not take.
    let pad = Vec2::splat(stroke_pad(&rec.brush));
    claim(min - pad, max + pad, 0)
}

/// The conservative footprint of an action, mirroring exactly what its arm of
/// [`Action`]'s `apply` touches. `Undo` has an empty footprint because it is
/// never materialized — the timeline resolves it into the effectiveness of its
/// target instead.
pub fn compute_footprint(action: &Action) -> Footprint {
    let actor = action.id.actor;
    match &action.kind {
        // The **substrate** is read alongside the author's mask, and it is the one
        // read here that is not about the layer being painted on. The tooth gates
        // how much paint lands by the substrate's rise over a reach in canvas px
        // (§6.4), so `apply` takes both the substrate and the scale it is laid at
        // off the state being folded over — which makes a stroke a function of
        // them, and means it does not commute with either changing under it.
        //
        // Claiming it costs a false conflict between a stroke and a substrate
        // switch, which is a pair nobody performs concurrently. Omitting it cost
        // the §12.6 direction that has no alarm: an undo of a `SetSubstrate` took
        // the commuting splice past every stroke after it, leaving those tiles
        // toothed by a substrate the log no longer contained, and no pixel can say
        // which materialization ran.
        ActionKind::CommitStroke(rec) => Footprint {
            reads: vec![
                Resource::Existence(rec.layer),
                Resource::Selection(actor),
                Resource::Substrate,
            ],
            writes: vec![Resource::Paint(rec.layer, stroke_rect(rec))],
        },
        // Both anchors are read — the sibling to insert above and the layer whose
        // stack to insert into — because either being absent changes where the
        // layer lands (§14.8). A matte arrives the same way and claims the same
        // things, so the arms share a body: what a new layer *is* differs, where
        // it lands does not. The matte's anchor is a `Place` (§15.5),
        // whose `anchor()` is the same optional sibling the other two carry.
        ActionKind::AddLayer { id, carrier, above }
        | ActionKind::AddFilter {
            id, carrier, above, ..
        } => Footprint {
            reads: [*carrier, *above]
                .into_iter()
                .flatten()
                .map(Resource::Existence)
                .collect(),
            // [`Resource::Layer`] for the minted id, as every action that mints one
            // claims: what it writes is *everything about a layer that did not exist*,
            // and naming that as `Existence` plus whichever finer resources the arm
            // happens to touch would be the same claim said three ways.
            writes: vec![Resource::Layer(*id), Resource::StackOrder],
        },
        // A placed image is an `AddLayer` that arrives with paint and a name in it
        // (§23), so it claims what one claims plus those two — and claims the paint as
        // the **whole layer**, not as the box the image covers.
        //
        // That is not a conservative shrug, it is the one thing worth saying about this
        // footprint. Every other action that writes tiles has to derive its box twice —
        // once here and once where the tiles are planned — and keeping the two in step
        // is the §12.6 hazard that `fill_bounds` exists to remove. Here there is nothing
        // to keep in step: the layer did not exist before this action, so *all* of its
        // paint is this action's by construction, whatever box the image happens to
        // cover. `image_tiles` is then the only quantization of that box in the tree.
        ActionKind::PlaceImage {
            id, carrier, above, ..
        } => Footprint {
            reads: [*carrier, *above]
                .into_iter()
                .flatten()
                .map(Resource::Existence)
                .collect(),
            writes: vec![Resource::Layer(*id), Resource::StackOrder],
        },
        ActionKind::AddMatte {
            id, carrier, at, ..
        } => Footprint {
            reads: [*carrier, at.anchor()]
                .into_iter()
                .flatten()
                .map(Resource::Existence)
                .collect(),
            writes: vec![Resource::Layer(*id), Resource::StackOrder],
        },
        // A copy is a function of *everything it copies* — every tile and every
        // property of every layer in the subtree — which is the whole reason the
        // action names its sources rather than only its root (§14.8). Claiming
        // less would let a duplicate commute with a stroke inside the group it
        // copied, and the two orders give different paint.
        //
        // The tree's own shape is read too, since the subtree is walked and the
        // copy lands beside its source; `StackOrder` is written here, and a write
        // covers the read.
        ActionKind::DuplicateLayer { ids } => Footprint {
            reads: ids.iter().map(|(src, _)| Resource::Layer(*src)).collect(),
            writes: ids
                .iter()
                .map(|(_, copy)| Resource::Layer(*copy))
                .chain([Resource::StackOrder])
                .collect(),
        },
        // A removal takes the **whole subtree**, so it writes everything about every
        // layer in it — which is why the action names them (`ActionKind::RemoveLayer`)
        // and why each gets the coarse [`Resource::Layer`], the same claim
        // `DuplicateLayer` makes about what it copied.
        //
        // `StackOrder` alone does not cover it: it is about the tree's *shape*, so it
        // meets other structural edits and nothing else. A stroke on a carried layer
        // claims `Paint(child, rect)` and a slider claims `Prop(child, _)`, neither of
        // which overlaps `{Existence(id), StackOrder, Paint(id, ALL)}` — the pair would
        // be judged to commute, and the undo's splice and a canonical replay would then
        // disagree with no pixel able to report it (§12.6).
        ActionKind::RemoveLayer { id, carried } => Footprint {
            reads: Vec::new(),
            writes: std::iter::once(*id)
                .chain(carried.iter().copied())
                .map(Resource::Layer)
                .chain([Resource::StackOrder])
                .collect(),
        },
        ActionKind::MoveLayer { id, carrier, at } => Footprint {
            reads: [Some(*id), *carrier, at.anchor()]
                .into_iter()
                .flatten()
                .map(Resource::Existence)
                .collect(),
            writes: vec![Resource::StackOrder],
        },
        ActionKind::SetLayerBlend(id, _) => prop_write(*id, Prop::Blend),
        ActionKind::SetLayerClip(id, _) => prop_write(*id, Prop::Clip),
        ActionKind::SetLayerOpacity(id, _) => prop_write(*id, Prop::Opacity),
        ActionKind::SetLayerVisible(id, _) => prop_write(*id, Prop::Visible),
        ActionKind::SetLayerName(id, _) => prop_write(*id, Prop::Name),
        ActionKind::SetMatteRect(id, _, _) => prop_write(*id, Prop::Matte),
        ActionKind::SetMattePaint(id, _) => prop_write(*id, Prop::Matte),
        ActionKind::SetFilter(id, _) => prop_write(*id, Prop::Filter),
        // The strength is *of* the mask, so it claims the mask: two peers cannot
        // dim and redraw one selection at once and have both land (§12.6).
        ActionKind::Select(_)
        | ActionKind::InvertSelection
        | ActionKind::SetSelectionOpacity(_) => Footprint {
            reads: Vec::new(),
            writes: vec![Resource::Selection(actor)],
        },
        ActionKind::SetSubstrate(_) | ActionKind::SetSubstrateScale(_) => Footprint {
            reads: Vec::new(),
            writes: vec![Resource::Substrate],
        },
        ActionKind::SetSubstrateColor(_) => Footprint {
            reads: Vec::new(),
            writes: vec![Resource::SubstrateColor],
        },
        // Every guide edit claims the whole roster, which is what
        // [`Resource::Guides`] being one resource means. The anchors an add and a
        // move are stated against are guides in that same roster, so there is
        // nothing left to name as a separate read.
        ActionKind::AddGuide { .. }
        | ActionKind::RemoveGuide(_)
        | ActionKind::SetGuide(..)
        | ActionKind::SetGuideName(..)
        | ActionKind::MoveGuide { .. } => Footprint {
            reads: Vec::new(),
            writes: vec![Resource::Guides],
        },
        ActionKind::Transform { layer, .. } => Footprint {
            reads: vec![Resource::Existence(*layer)],
            writes: vec![
                Resource::Paint(*layer, TileRect::ALL),
                Resource::Selection(actor),
            ],
        },
        // The rect-scoped transforms (§16.8, §16.9) cut only inside their rect
        // and paste only inside the map's image, so unlike the whole-plane
        // affine they can claim an honest box: the union of the two, padded a
        // tile for the apron reach. An unusable warp (whose image is unknown)
        // falls back to the whole layer — it will be rejected by `apply`, and
        // a too-big footprint is the safe direction.
        ActionKind::TransformPerspective { layer, map } => Footprint {
            reads: vec![Resource::Existence(*layer)],
            writes: vec![
                Resource::Paint(*layer, gated_rect((map.min, map.max), map.image_aabb())),
                Resource::Selection(actor),
            ],
        },
        ActionKind::TransformWarp { layer, map } => Footprint {
            reads: vec![Resource::Existence(*layer)],
            writes: vec![
                Resource::Paint(*layer, gated_rect((map.min, map.max), map.image_aabb())),
                Resource::Selection(actor),
            ],
        },
        // A fill reads the mask that bounds it and writes the paint its region
        // reaches — the same shape of footprint a stroke has. A fill bounded only
        // by the selection has no analytic box, so it claims the whole layer, the
        // conservative answer a transform gives for the same reason.
        ActionKind::Fill { layer, op } => Footprint {
            reads: vec![Resource::Existence(*layer), Resource::Selection(actor)],
            writes: vec![Resource::Paint(*layer, fill_rect(op))],
        },
        // A merge is a function of **everything about both layers**, because that is
        // what its plan reads: the tiles it stacks, the blend, clip, opacity and
        // visibility that decide whether the merge is offered at all (§14.11), and the
        // filter, which for a filter source is both half the offer (a resampling kind
        // is declined, §14.11.7) and what the merge writes into the destination's
        // texels. It is also a function of the tree's shape, which decides what "down"
        // means — and `StackOrder` is written here, so the write covers that read.
        //
        // `DuplicateLayer`'s argument taken one step further: a duplicate reads its
        // subtree's properties, while a merge would silently *change its own answer* if
        // one of them moved past it. Claiming them is what keeps a concurrent
        // blend-mode change from commuting with a merge the mode would have refused —
        // and why the filter is claimed on both ids though only a source can carry one.
        ActionKind::MergeLayerDown { source, dest } => Footprint {
            // Everything about both layers, in the one resource that says so: the
            // source's paint is stacked and the destination's is rewritten.
            reads: [*source, *dest].into_iter().map(Resource::Layer).collect(),
            writes: vec![
                Resource::Existence(*source),
                Resource::Existence(*dest),
                Resource::Paint(*dest, TileRect::ALL),
                // **All three of the survivor's composite params**, because that is
                // what `apply` assigns: the plan's `keeps` is a whole
                // `CompositeParams` and it is written as one.
                //
                // Two of the three are the identity today — `merge::plan` refuses a
                // clipped destination and takes `keeps.blend` from the destination's
                // own — so claiming them costs a false conflict and nothing else,
                // which is the direction §12.6 says to err in. Claiming only the
                // opacity would be resting the footprint's honesty on refusals made
                // in *another file*, and on the two merges `merge`'s own header says
                // it means to build next, both of which touch `keeps`.
                Resource::Prop(*dest, Prop::Blend),
                Resource::Prop(*dest, Prop::Clip),
                // The survivor is left at full opacity, both sliders having been
                // folded into its tiles.
                Resource::Prop(*dest, Prop::Opacity),
                Resource::StackOrder,
            ],
        },
        ActionKind::Undo(_) => Footprint::default(),
    }
}

/// The tile-aligned reach of a rect-scoped transform: its source rect unioned
/// with its image bound, padded one tile so apron rewrites are covered.
/// `None` for the image (an unusable map) claims everything.
///
/// **Both halves are tested for finiteness here**, and neither can be left to
/// [`claim`]: `Vec2::min`/`max` return the non-NaN operand, so a non-finite
/// corner is *swallowed* by the union rather than carried into
/// [`TileRect::covering`]'s guard — the box arrives finite and tight-looking,
/// which is the one answer §12.6 does not permit. The same test, for the same
/// reason, as the one [`stroke_rect`] makes per control point.
///
/// The `image` half is an [`Option`] for exactly this reason (see
/// `PerspectiveMap::image_aabb`); testing the `rect` half here rather than leaving it
/// to `usable`/`shape_ok` at `apply` is what keeps this footprint's honesty from
/// resting on a refusal in another file.
fn gated_rect(rect: (Vec2, Vec2), image: Option<(Vec2, Vec2)>) -> TileRect {
    let Some(image) = image else {
        return TileRect::ALL;
    };
    if !(rect.0.is_finite() && rect.1.is_finite()) {
        return TileRect::ALL;
    }
    claim(rect.0.min(image.0), rect.1.max(image.1), 1)
}

/// The tile-aligned reach of a fill: everything its pass may read or write.
/// Shared with the live-preview fold for the same reason as [`stroke_rect`].
pub fn fill_rect(op: &super::fill::FillOp) -> TileRect {
    let Some((lo, hi)) = super::fill::fill_bounds(op) else {
        return TileRect::ALL;
    };
    claim(lo, hi, 0)
}

fn prop_write(id: LayerId, prop: Prop) -> Footprint {
    Footprint {
        reads: vec![Resource::Existence(id)],
        writes: vec![Resource::Prop(id, prop)],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::Place;
    use crate::document::action::ActionId;
    use crate::document::brush::BrushParams;
    use crate::geom::Vec2;
    use crate::path::ControlPoint;

    fn act(actor: u64, kind: ActionKind) -> Action {
        Action {
            id: ActionId {
                lamport: 1,
                actor: ActorId(actor),
            },
            kind,
        }
    }

    fn stroke(actor: u64, layer: LayerId, from: Vec2, to: Vec2, radius: f32) -> Action {
        let point = |pos| ControlPoint {
            pos,
            pressure: 1.0,
            tilt: Vec2::ZERO,
            time: 0.0,
        };
        act(
            actor,
            ActionKind::CommitStroke(StrokeRecord {
                layer,
                brush: BrushParams {
                    size: radius,
                    ..BrushParams::default()
                },
                path: vec![point(from), point(to)],
                seed: 0,
                start: 0.0,
            }),
        )
    }

    fn commutes(a: &Action, b: &Action) -> bool {
        !compute_footprint(a).conflicts(&compute_footprint(b))
    }

    /// [`Prop::ALL`] is what [`Resource::Layer`] expands to, so a property missing
    /// from it would make the coarse claim quietly finer than it says it is — and
    /// finer is the direction §12.6 cannot survive.
    ///
    /// The match is the guard: exhaustive with no `_` arm, so a new `Prop` does not
    /// compile until it is named here, next to the assertion that `ALL` has grown with
    /// it. It forces a visit rather than proving the correspondence, which is as far as
    /// Rust goes without a derive.
    #[test]
    fn every_prop_is_named_in_all() {
        for prop in Prop::ALL {
            match prop {
                Prop::Blend
                | Prop::Clip
                | Prop::Opacity
                | Prop::Visible
                | Prop::Name
                | Prop::Matte
                | Prop::Filter => {}
            }
        }
        assert_eq!(Prop::ALL.len(), 7, "a new Prop needs a place in ALL");
    }

    /// Every [`Resource`] variant is visited, and one of each meets **exactly** what
    /// it names: itself, plus whatever [`Resource::Layer`] coarsens over.
    ///
    /// `overlaps` is the only match in the crate over this enum that ends in a `_`
    /// arm — every other one (`layer`, and `ActionKind`'s five) is exhaustive so a
    /// new variant cannot slip past unanswered. It cannot be spelled exhaustively
    /// without a quadratic match, and it is also the predicate where a wrong answer
    /// is the expensive kind: a resource that wrongly reports "no overlap" silently
    /// diverges peers, where [`Prop`]'s equivalent mistake only makes a coarse claim
    /// finer than it says.
    ///
    /// So the visit is forced here instead, exactly as [`every_prop_is_named_in_all`]
    /// forces one: the array below does not compile until a new variant is named in
    /// it, and naming it means saying whether it is about a layer — which is the
    /// only thing `overlaps` needs to know about it.
    #[test]
    fn every_resource_is_visited_and_meets_exactly_what_it_names() {
        let layer = LayerId::solo(4);
        // One of every variant, each paired with whether it is about `layer` — the
        // question the coarse claim asks. Written out rather than read off
        // `Resource::layer`, which is the helper `overlaps` is built from: an
        // expectation computed the same way as the answer agrees with it by
        // construction, which is the trap `overlaps`' own doc names.
        let samples = [
            (Resource::Paint(layer, TileRect::ALL), true),
            (Resource::Existence(layer), true),
            (Resource::Prop(layer, Prop::Name), true),
            (Resource::Layer(layer), true),
            (Resource::StackOrder, false),
            (Resource::Selection(ActorId(1)), false),
            (Resource::Substrate, false),
            (Resource::SubstrateColor, false),
            (Resource::Guides, false),
        ];
        for (r, _) in &samples {
            match r {
                Resource::Paint(..)
                | Resource::Existence(_)
                | Resource::Prop(..)
                | Resource::Layer(_)
                | Resource::StackOrder
                | Resource::Selection(_)
                | Resource::Substrate
                | Resource::SubstrateColor
                | Resource::Guides => {}
            }
        }
        assert_eq!(
            samples.len(),
            9,
            "a new Resource needs a row in the overlap matrix",
        );

        for (a, a_is_layers) in &samples {
            for (b, b_is_layers) in &samples {
                let coarse = |r: &Resource| matches!(r, Resource::Layer(_));
                // The whole spec: a resource meets itself, and the coarse claim
                // additionally meets everything about the same layer.
                let want = a == b || (coarse(a) && *b_is_layers) || (coarse(b) && *a_is_layers);
                assert_eq!(a.overlaps(b), want, "{a:?} vs {b:?}");
            }
        }

        // And nothing about one layer reaches another — the coarse claim included,
        // which is the arm that has a layer id to get wrong.
        let elsewhere = LayerId::solo(9);
        for (r, is_layers) in &samples {
            if !is_layers {
                continue;
            }
            for other in [
                Resource::Layer(elsewhere),
                Resource::Existence(elsewhere),
                Resource::Paint(elsewhere, TileRect::ALL),
                Resource::Prop(elsewhere, Prop::Name),
            ] {
                assert!(!r.overlaps(&other), "{r:?} must not meet {other:?}");
                assert!(!other.overlaps(r), "…and from the other side");
            }
        }
    }

    /// The coarse resource claims **everything** about its layer, and claims it
    /// symmetrically: whichever side it appears on, it meets every finer resource of
    /// that layer and nothing of any other.
    #[test]
    fn a_whole_layer_claim_meets_every_finer_claim_on_it() {
        let (a, b) = (LayerId::solo(4), LayerId::solo(9));
        let whole = Resource::Layer(a);
        let finer = [
            Resource::Existence(a),
            Resource::Paint(a, TileRect::ALL),
            Resource::Paint(
                a,
                TileRect::covering(Vec2::ZERO, Vec2::splat(9.0), 0).unwrap(),
            ),
            Resource::Layer(a),
        ]
        .into_iter()
        .chain(Prop::ALL.map(|p| Resource::Prop(a, p)));
        for r in finer {
            assert!(whole.overlaps(&r), "{whole:?} must meet {r:?}");
            assert!(r.overlaps(&whole), "…and from the other side");
        }
        // A different layer, and the resources that are about the document rather
        // than about any layer, are untouched by it.
        for r in [
            Resource::Layer(b),
            Resource::Existence(b),
            Resource::Paint(b, TileRect::ALL),
            Resource::Prop(b, Prop::Name),
            Resource::StackOrder,
            Resource::Selection(ActorId(1)),
            Resource::Substrate,
            Resource::SubstrateColor,
            Resource::Guides,
        ] {
            assert!(!whole.overlaps(&r), "{whole:?} must not meet {r:?}");
            assert!(!r.overlaps(&whole), "…and from the other side");
        }
    }

    /// A duplicate reads *everything* about every layer it copies (§14.8), which is
    /// what keeps it from commuting with a stroke or a rename inside the group. Stated
    /// as one coarse resource a layer, so this pins the claim rather than the
    /// spelling: what must hold is that each finer edit still conflicts.
    #[test]
    fn a_duplicate_conflicts_with_every_edit_inside_what_it_copies() {
        let inner = LayerId::solo(3);
        let dup = act(
            1,
            ActionKind::DuplicateLayer {
                ids: vec![
                    (LayerId::solo(2), LayerId::solo(20)),
                    (inner, LayerId::solo(30)),
                ],
            },
        );
        let edits = [
            ActionKind::SetLayerName(inner, Some("wash".into())),
            ActionKind::SetLayerBlend(inner, crate::document::BlendMode::Multiply),
            ActionKind::SetLayerClip(inner, true),
            ActionKind::SetLayerOpacity(inner, 0.5),
            ActionKind::SetLayerVisible(inner, false),
            ActionKind::SetMattePaint(inner, crate::document::Parcel::Solid(crate::Srgb::BLACK)),
            ActionKind::RemoveLayer {
                id: inner,
                carried: Vec::new(),
            },
        ];
        for kind in edits {
            let other = act(2, kind);
            assert!(
                !commutes(&dup, &other),
                "a duplicate must not commute with {:?}",
                other.kind,
            );
        }
        // A stroke inside the copied subtree, too.
        let paint = stroke(2, inner, Vec2::ZERO, Vec2::splat(40.0), 8.0);
        assert!(!commutes(&dup, &paint));
        // …and a layer it does not copy is still free.
        let elsewhere = stroke(2, LayerId::solo(99), Vec2::ZERO, Vec2::splat(40.0), 8.0);
        assert!(commutes(&dup, &elsewhere));
    }

    #[test]
    fn strokes_on_different_layers_commute() {
        let a = stroke(1, LayerId::ROOT, Vec2::ZERO, Vec2::splat(100.0), 16.0);
        let b = stroke(2, LayerId::solo(1), Vec2::ZERO, Vec2::splat(100.0), 16.0);
        assert!(commutes(&a, &b));
    }

    #[test]
    fn distant_strokes_on_one_layer_commute_and_near_ones_conflict() {
        let a = stroke(1, LayerId::ROOT, Vec2::ZERO, Vec2::splat(60.0), 16.0);
        let far = stroke(
            2,
            LayerId::ROOT,
            Vec2::splat(2000.0),
            Vec2::splat(2100.0),
            16.0,
        );
        let near = stroke(
            2,
            LayerId::ROOT,
            Vec2::splat(80.0),
            Vec2::splat(300.0),
            16.0,
        );
        assert!(commutes(&a, &far));
        assert!(!commutes(&a, &near));
    }

    #[test]
    fn rename_commutes_with_strokes_but_not_with_removal() {
        let name = act(
            1,
            ActionKind::SetLayerName(LayerId::ROOT, Some("wash".into())),
        );
        let paint = stroke(2, LayerId::ROOT, Vec2::ZERO, Vec2::splat(50.0), 8.0);
        let remove = act(
            2,
            ActionKind::RemoveLayer {
                id: LayerId::ROOT,
                carried: Vec::new(),
            },
        );
        let other_name = act(2, ActionKind::SetLayerName(LayerId::ROOT, None));
        assert!(commutes(&name, &paint));
        assert!(!commutes(&name, &remove));
        assert!(!commutes(&name, &other_name));
    }

    #[test]
    fn selection_gates_only_its_author() {
        let select = act(1, ActionKind::InvertSelection);
        let own = stroke(1, LayerId::ROOT, Vec2::ZERO, Vec2::splat(50.0), 8.0);
        let other = stroke(
            2,
            LayerId::ROOT,
            Vec2::splat(500.0),
            Vec2::splat(600.0),
            8.0,
        );
        assert!(!commutes(&select, &own));
        assert!(commutes(&select, &other));
    }

    /// **A stroke does not commute with the substrate changing under it.**
    ///
    /// The tooth gates the deposit by the substrate and by the scale it is laid at
    /// (§6.4), and `apply` reads both off the state being folded over — so the
    /// pixels of a stroke depend on which `SetSubstrate` preceded it, and an undo
    /// that spliced one out past its strokes would leave them toothed by a
    /// substrate the log no longer contains.
    ///
    /// Whose stroke it is makes no difference, unlike the mask: a substrate is
    /// shared document state, so it gates every author's paint at once.
    #[test]
    fn a_stroke_does_not_commute_with_the_substrate_under_it() {
        let own = stroke(1, LayerId::ROOT, Vec2::ZERO, Vec2::splat(50.0), 8.0);
        let theirs = stroke(
            2,
            LayerId::solo(1),
            Vec2::splat(500.0),
            Vec2::splat(600.0),
            8.0,
        );
        for substrate in [
            act(1, ActionKind::SetSubstrate(crate::SubstrateId::Flat)),
            act(
                1,
                ActionKind::SetSubstrateScale(crate::SubstrateScale::new(200)),
            ),
        ] {
            assert!(!commutes(&substrate, &own));
            assert!(!commutes(&substrate, &theirs), "a substrate is shared");
        }

        // The substrate *color* is a different resource and one no stroke reads:
        // it sits under the paint at composite rather than gating the deposit.
        let color = act(1, ActionKind::SetSubstrateColor(crate::Srgb::WHITE));
        assert!(commutes(&color, &own));
    }

    #[test]
    fn structural_edits_conflict_with_each_other() {
        let add = act(
            1,
            ActionKind::AddLayer {
                id: LayerId::solo(7),
                carrier: None,
                above: None,
            },
        );
        let mv = act(
            2,
            ActionKind::MoveLayer {
                id: LayerId::solo(3),
                carrier: None,
                at: Place::Top,
            },
        );
        assert!(!commutes(&add, &mv));
    }

    /// Clipping and the blend mode are applied at the same step but are separate
    /// resources, so setting one commutes with setting the other
    /// (§14.8) — while two clip toggles on one layer do not.
    #[test]
    fn clip_commutes_with_blend_but_not_with_itself() {
        let clip = act(1, ActionKind::SetLayerClip(LayerId::ROOT, true));
        let blend = act(
            2,
            ActionKind::SetLayerBlend(LayerId::ROOT, crate::document::BlendMode::Multiply),
        );
        let unclip = act(2, ActionKind::SetLayerClip(LayerId::ROOT, false));
        let elsewhere = act(2, ActionKind::SetLayerClip(LayerId::solo(1), true));
        assert!(commutes(&clip, &blend));
        assert!(commutes(&clip, &elsewhere));
        assert!(!commutes(&clip, &unclip));
    }

    /// **A stretched tip's footprint has to grow with it** (§6.6, §12.6).
    ///
    /// A brush drawn out along its facing axis reaches `elongation` times as far as its
    /// radius names, and this is the bound the *renderer* is held to
    /// (`gpu::stroke::segments::Sweep::reach`, the tip's reach times the same factor).
    /// Left at the `√2` that covered a square stamp alone, a leaned pencil paints past
    /// the tiles its action claimed — and unlike a clipped stroke that is not a visible
    /// bug but a §12.6 one: two peers commute a pair of strokes the paint says overlap,
    /// and no pixel can show which order ran.
    ///
    /// Checked as the inequality rather than against a copy of the expression, so the
    /// two sides stay free to differ by slack and not by kind.
    #[test]
    fn a_stretched_stroke_claims_the_tiles_its_drawn_out_tip_can_reach() {
        for stretch in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let brush = BrushParams {
                size: 40.0,
                stretch,
                ..BrushParams::default()
            };
            // What the renderer will draw into: the tip's own reach — a stamp may fill
            // its mask's corners — times how far the stretch draws it out.
            let painted = brush.size * std::f32::consts::SQRT_2 * BrushParams::elongation(stretch);
            assert!(
                stroke_pad(&brush) >= painted,
                "stretch {stretch}: the footprint pads {} where the tip paints {painted}",
                stroke_pad(&brush),
            );
        }

        // And the claim actually widens on the log, rather than the pad growing inside
        // a box that was already tile-aligned to the same rect.
        let rect = |stretch: f32| {
            let point = |pos| ControlPoint {
                pos,
                pressure: 1.0,
                tilt: Vec2::ZERO,
                time: 0.0,
            };
            stroke_rect(&StrokeRecord {
                layer: LayerId::ROOT,
                brush: BrushParams {
                    size: 40.0,
                    stretch,
                    ..BrushParams::default()
                },
                path: vec![point(Vec2::splat(600.0)), point(Vec2::splat(700.0))],
                seed: 0,
                start: 0.0,
            })
        };
        let (plain, drawn_out) = (rect(0.0), rect(0.875));
        assert_ne!(plain, drawn_out, "a stretched tip claimed no more tiles");
        assert_eq!(
            plain.union(drawn_out),
            drawn_out,
            "the stretched claim must contain the plain one, not merely differ",
        );
    }

    /// A stroke whose box cannot be quantized claims the **whole layer**, never a tile
    /// at the origin — which is what a bare `NaN as i32` would give. A non-finite
    /// radius or path point producing a tight-looking footprint is the one direction
    /// §12.6 cannot survive: a distant stroke commutes past it, the fast path splices
    /// on a lie, and no pixel can show it.
    #[test]
    fn an_unboundable_stroke_claims_the_layer_rather_than_the_origin() {
        let elsewhere = stroke(
            2,
            LayerId::ROOT,
            Vec2::splat(9000.0),
            Vec2::splat(9100.0),
            8.0,
        );
        let claims_all = |a: &Action| match &compute_footprint(a).writes[..] {
            [Resource::Paint(_, rect)] => *rect == TileRect::ALL,
            _ => false,
        };

        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            // A radius that cannot be padded with.
            let mut a = stroke(1, LayerId::ROOT, Vec2::ZERO, Vec2::splat(50.0), bad);
            assert!(claims_all(&a), "radius {bad} must claim the layer");
            assert!(!commutes(&a, &elsewhere));

            // A path point that cannot be bounded. `f32::min` would step over it and
            // leave the other point's box looking exact.
            a = stroke(
                1,
                LayerId::ROOT,
                Vec2::new(bad, 0.0),
                Vec2::splat(50.0),
                8.0,
            );
            assert!(claims_all(&a), "path point {bad} must claim the layer");
            assert!(!commutes(&a, &elsewhere));
        }

        // Finite, but past what an `i32` tile index can address: clamping inward
        // would shrink the claim.
        let far = stroke(
            1,
            LayerId::ROOT,
            Vec2::splat(1.0e30),
            Vec2::splat(1.1e30),
            8.0,
        );
        assert!(claims_all(&far));
        // …and an ordinary stroke still claims an ordinary box.
        let ordinary = stroke(1, LayerId::ROOT, Vec2::ZERO, Vec2::splat(50.0), 16.0);
        assert!(!claims_all(&ordinary));
        assert!(commutes(&ordinary, &elsewhere));
    }

    /// A rect-scoped transform whose **source rect** cannot be measured claims the
    /// whole layer, exactly as one whose image cannot be does.
    ///
    /// The union in [`gated_rect`] is `Vec2::min`/`max`, which return the non-NaN
    /// operand — so a non-finite `min`/`max` against a finite image does not reach
    /// [`TileRect::covering`]'s guard at all: it is discarded, and the claim comes
    /// back tight and wrong. That is the under-claim §12.6 cannot survive, and it
    /// is invisible in a picture, since `apply` refuses such a map anyway and no
    /// pixel is ever written by the action that lied about its reach.
    #[test]
    fn a_transform_with_an_unmeasurable_rect_claims_the_layer() {
        use crate::document::transform::{PerspectiveMap, rect_corners};
        use crate::document::warp::WarpMap;

        let elsewhere = stroke(
            2,
            LayerId::ROOT,
            Vec2::splat(9000.0),
            Vec2::splat(9100.0),
            8.0,
        );
        let claims_all = |a: &Action| {
            compute_footprint(a)
                .writes
                .iter()
                .any(|r| matches!(r, Resource::Paint(_, rect) if *rect == TileRect::ALL))
        };

        let (lo, hi) = (Vec2::ZERO, Vec2::splat(100.0));
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for corner in [Vec2::new(bad, 0.0), Vec2::new(0.0, bad)] {
                // The corners stay finite, so the image *is* measurable and the
                // `None` arm never fires — the rect alone is what cannot be read.
                let perspective = act(
                    1,
                    ActionKind::TransformPerspective {
                        layer: LayerId::ROOT,
                        map: PerspectiveMap {
                            min: corner,
                            max: hi,
                            corners: rect_corners(lo, hi),
                        },
                    },
                );
                assert!(claims_all(&perspective), "rect min {corner:?}");
                assert!(!commutes(&perspective, &elsewhere));

                let mut mesh = WarpMap::identity(lo, hi, 3, 3);
                mesh.max = corner;
                let warp = act(
                    1,
                    ActionKind::TransformWarp {
                        layer: LayerId::ROOT,
                        map: mesh,
                    },
                );
                assert!(claims_all(&warp), "rect max {corner:?}");
                assert!(!commutes(&warp, &elsewhere));
            }
        }

        // …and a measurable one still claims an ordinary box on both arms.
        let ordinary = act(
            1,
            ActionKind::TransformPerspective {
                layer: LayerId::ROOT,
                map: PerspectiveMap {
                    min: lo,
                    max: hi,
                    corners: rect_corners(lo, hi),
                },
            },
        );
        assert!(!claims_all(&ordinary));
        assert!(commutes(&ordinary, &elsewhere));
    }

    /// Carrying a layer is a `MoveLayer`, so it conflicts with every other
    /// structural edit through `StackOrder` — which is what serializes the two
    /// halves of a would-be cycle (§14.8).
    #[test]
    fn carrying_conflicts_with_the_reverse_carry() {
        let a_onto_b = act(
            1,
            ActionKind::MoveLayer {
                id: LayerId::ROOT,
                carrier: Some(LayerId::solo(1)),
                at: Place::Top,
            },
        );
        let b_onto_a = act(
            2,
            ActionKind::MoveLayer {
                id: LayerId::solo(1),
                carrier: Some(LayerId::ROOT),
                at: Place::Top,
            },
        );
        assert!(!commutes(&a_onto_b, &b_onto_a));
    }
}
