//! Action footprints: which parts of the document an action reads and writes
//! (§12.6).
//!
//! Two actions **commute** — applying them in either order produces the same
//! state — when neither writes anything the other reads or writes. The history
//! uses that (via the [`Centralizer`](history::Centralizer) impl below) to splice
//! an undone action out of the materialization instead of replaying past it.
//!
//! Footprints are **conservative**. A false conflict only costs the fast path; a
//! missed one silently diverges peers, so every arm of the fold ([`Materialize`],
//! in `stark-engine`'s `document/apply.rs`) must read and write only what its
//! kind's footprint declares. That locality is what makes the splice sound (see
//! `timeline.rs`), and `stark-engine`'s `document/audit.rs` holds every debug fold
//! to it.
//!
//! [`Materialize`]: super::Materialize

use super::action::{Action, ActionKind, ActorId, StrokeRecord};
use super::brush::BrushParams;
use super::layer::LayerId;
use crate::geom::{TileRect, Vec2};

/// The tiles a pass may touch within the canvas box `[lo, hi]`, grown by `ring`
/// tiles.
///
/// An unquantizable box claims **everything**: a footprint may only ever claim too
/// much (§12.6), since a false conflict costs the commutation fast path while a
/// missed one silently diverges peers with no pixel able to show which
/// materialization ran.
fn claim(lo: Vec2, hi: Vec2, ring: i32) -> TileRect {
    TileRect::covering(lo, hi, ring).unwrap_or(TileRect::ALL)
}

/// The per-layer properties **as a roster**: the enum and [`Prop::ALL`] come out of
/// one list, so a property cannot be missing from `ALL`.
///
/// The omission would be silent in both directions: `ALL` is what
/// [`Resource::Layer`] expands to, so a property absent from it makes that coarse
/// claim quietly *finer* than it says (§12.6), and undo restores only what the
/// expansion captured.
macro_rules! props {
    ($($(#[$m:meta])* $variant:ident,)*) => {
        /// A per-layer property, at the granularity undo needs to restore it: each
        /// variant is overwritten wholesale by the actions that write it.
        #[derive(Copy, Clone, Debug, PartialEq, Eq)]
        pub enum Prop { $($(#[$m])* $variant,)* }

        impl Prop {
            /// Every property, which is what [`Resource::Layer`] stands for the paint
            /// and the existence of.
            pub const ALL: &'static [Prop] = &[$(Prop::$variant,)*];
        }
    };
}

props! {
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
    /// Where the layer's frame sits on the canvas (§14.12). Its own resource so a
    /// translate commutes with a stroke on the same layer: paint actions never read
    /// it — their geometry is in the layer's frame, and the offset they reconcile the
    /// canvas-anchored mask against travels in the action, not in the state.
    Translation,
}

/// One addressable piece of document state.
///
/// **The vocabulary is closed over the log, with one exception.** Every resource
/// here names something an action carries or something the fold built from earlier
/// actions, so two peers holding the same log agree about all of it. `PlaceImage` is
/// the exception: its tiles come from a picture held in an out-of-log store under the
/// id the action carries (§23.2), so a peer that has not received the picture folds
/// the same action into an empty layer. There is no resource for it — the divergence
/// is between two peers' stores rather than two orders of the same actions, which is
/// a transport contract (§12.4).
#[derive(Clone, Debug, PartialEq)]
pub enum Resource {
    /// A layer's painted tiles within a tile rect.
    Paint(LayerId, TileRect),
    /// A layer's presence in the document at all. Every action that targets a
    /// layer reads this (they all no-op on an absent layer, and that no-op is
    /// order-dependent against add/remove).
    ///
    /// **It stands for the layer's *minted* kind as well** — paint, matte, filter —
    /// which several arms read while declaring nothing else (`cannot_carry` refuses a
    /// filter as a carrier, §21.2). Sound only because those kinds are fixed at mint
    /// and no action changes them; an action that *converted* a layer between them
    /// would be a write no reader of this one sees, and needs a resource of its own.
    ///
    /// **Whether a layer is a *group* is not one of them** and must not be read this
    /// way. Carrying is structure, not kind — a leaf becomes a group the moment
    /// `MoveLayer` puts something under it — so what covers it is
    /// [`StackOrder`](Self::StackOrder).
    Existence(LayerId),
    /// One presentation property of a layer.
    Prop(LayerId, Prop),
    /// The **shape of the whole layer tree** — every stack's order and who
    /// carries whom (§14.8). One coarse resource: two concurrent
    /// restructures genuinely don't commute, and structural edits are rare
    /// enough that finer granularity would buy nothing.
    ///
    /// It is also what makes the carry-your-own-ancestor case safe without
    /// tree-CRDT machinery: two halves of a cycle conflict here, so the log's total
    /// order serializes them and the second to apply sees the first's result and
    /// declines.
    StackOrder,
    /// **Everything about one layer**: its existence, all of its paint, and every
    /// one of its properties.
    ///
    /// For the two actions that genuinely read a whole layer: `DuplicateLayer` copies
    /// every tile and property of every layer in a subtree, and `MergeLayerDown` is a
    /// function of everything about both sides. Spelled finely that is nine and five
    /// entries a layer, which makes [`Footprint::conflicts`] — a nested scan —
    /// quadratic. It says the same thing: a coarse claim is *more* conservative than
    /// the fine ones it replaces, and §12.6 permits a footprint to claim too much.
    Layer(LayerId),
    /// A layer's **liquify run** (§6.13): the pristine base, the accumulated
    /// displacement field and the reach a sequence of liquify strokes has built on
    /// it, through which the next liquify stroke composes rather than resampling what
    /// the last one left.
    ///
    /// Read and written by every liquify stroke, so two on one layer always conflict.
    /// A paint stroke leaves the run alone and commutes with a liquify stroke exactly
    /// when their tiles do — and a liquify stroke's tile claim reaches
    /// [`liquify_reads`] beyond its own mark, since the picture it resamples is the
    /// base under the whole composed displacement.
    LiquifyRun(LayerId),
    /// An actor's selection mask (§17.3).
    Selection(ActorId),
    /// The canvas substrate (§6.4): **which substrate, and the scale it is laid at.**
    ///
    /// One resource for the two: they are one fact — the tooth reads the substrate's
    /// rise over a reach in canvas px, so the substrate and how large it is laid
    /// decide the deposit together. A finer split would only buy the commutation fast
    /// path between two collaborators picking both at the same moment, and §12.6
    /// permits a footprint to claim too much.
    Substrate,
    /// The substrate color (§15.5).
    SubstrateColor,
    /// **The whole drawing-guide roster** (§20.5): every guide, everything about
    /// each of them, and the order they are arranged in.
    ///
    /// One coarse resource. Spelling it finely would only let two artists shape two
    /// different guides concurrently without a rebase, and a footprint may claim too
    /// much (§12.6). Nothing on the drawing path touches it: a stroke reads the guides
    /// to snap through them before it is an action at all, so the action it commits
    /// names this nowhere and paint and guides never contend.
    Guides,
}

impl Resource {
    /// Whether these two name any state in common — the relation
    /// [`Footprint::conflicts`] is built from.
    ///
    /// Public so a test can hold an `apply` to its declaration through the *same*
    /// predicate the timeline commutes by (`stark-engine/tests/footprint.rs`);
    /// answering that question a second way is how a coarse claim comes to look finer
    /// than it is.
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
            | Resource::Layer(id)
            | Resource::LiquifyRun(id) => Some(*id),
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
    /// What the `history::Centralizer` impl on `&Footprint` (in the `fold` module)
    /// commutes by. A false conflict only costs the fast path; a missed one silently
    /// diverges peers (§12.6).
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
/// The B-spline stays inside its control points' convex hull, so the bbox bounds
/// the centerline exactly; the tip then reaches at most `radius` scaled by √2 for a
/// square stamp swept at an angle (the 1.5 covers it), times the elongation (§6.6)
/// for a tip drawn out along its facing axis, plus 4 px for the `TILE_APRON` of
/// duplicated neighbor pixels the renderer refreshes past its marks.
///
/// Under-reporting this is not a clipped stroke but a §12.6 break: two peers decide
/// a pair of strokes commute when the paint says otherwise, and pixels cannot show
/// which order ran. The brush's own `stretch` rather than any segment's, since a
/// modulation only ever scales it down; `elongation` is bounded and NaN-safe, so a
/// malformed one lands on a real factor rather than widening this to `ALL`.
///
/// # Why the lateral flux needs no allowance here
///
/// [`BrushDynamics::bleed`](super::brush::BrushDynamics::bleed) diffuses paint
/// sideways over a reach (`gpu::stroke::dynamics::bleed::BLEED_REACH_MAX`) that
/// would overrun this pad if it were additive to the √2 above — 957 px against 754
/// at `radius = 500`.
///
/// It is not, because **the flux cannot cross the edge of the sweep**.
/// `dynamics.wesl`'s exchange weighs every tap by `min(w_t, w_n)` and `bleed_weight`
/// writes `w = 0` for any texel outside the sweep, so a texel outside is never
/// written and a tap reaching out of the sweep carries nothing. The reach sets how
/// far *within* the footprint paint is carried, never how far the footprint extends,
/// so a bleed reach raised past 1.5 would still be contained. What *would* break
/// this is that no-flux wall coming down, checkable only where the shader is.
fn stroke_pad(brush: &BrushParams) -> f32 {
    brush.size * 1.5 * BrushParams::elongation(brush.stretch) + 4.0
}

/// The tile-aligned reach of a stroke: everything its render may read or write.
///
/// Shared with the live-preview fold (§17.6), and deliberately the *same* answer the
/// commit's footprint gives, so the fold cannot decide two strokes are independent
/// where the log would decide they conflict.
///
/// The box covers the **whole** recorded curve, run-up included, though the deposit
/// begins at [`StrokeRecord::start`] — the pre-marker stretch only ever
/// *over*-declares, which is the safe direction (§12.6).
pub fn stroke_rect(rec: &StrokeRecord) -> TileRect {
    let mut min = Vec2::splat(f32::INFINITY);
    let mut max = Vec2::splat(f32::NEG_INFINITY);
    for p in &rec.path {
        // Tested rather than folded in: `f32::min`/`max` return the *non*-NaN
        // operand, so a non-finite point would step straight over the bbox and leave
        // it looking tight. Records arrive from files and peers.
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
    // `covering` answers with `ALL` — the safe direction.
    let pad = Vec2::splat(stroke_pad(&rec.brush));
    claim(min - pad, max + pad, 0)
}

/// The tile-aligned reach of a **liquify** stroke's reads (§6.13): [`stroke_rect`]
/// grown by [`LiquifyEffect::REACH_PX`](super::brush::LiquifyEffect::REACH_PX) on
/// every side.
///
/// A liquify stroke writes only its own mark, but resamples the run's pristine base
/// under the *composed* displacement, which may point that far outside it. The engine
/// holds the displacement under the constant, so this is the whole of what the
/// stroke's render may read of the layer's paint — and a paint stroke inside it must
/// be ordered against the liquify stroke (§12.6).
pub fn liquify_reads(rec: &StrokeRecord) -> TileRect {
    let mut min = Vec2::splat(f32::INFINITY);
    let mut max = Vec2::splat(f32::NEG_INFINITY);
    for p in &rec.path {
        if !p.pos.is_finite() {
            return TileRect::ALL;
        }
        min = min.min(p.pos);
        max = max.max(p.pos);
    }
    if min.x > max.x {
        return TileRect::EMPTY;
    }
    let pad = Vec2::splat(stroke_pad(&rec.brush) + super::brush::LiquifyEffect::REACH_PX);
    claim(min - pad, max + pad, 0)
}

/// The conservative footprint of an action, mirroring exactly what its arm of the
/// fold touches (`stark-engine`'s `document/apply.rs`, checked on every debug fold by
/// its `document/audit.rs`). `Undo`'s is empty: it is never materialized — the
/// timeline resolves it into the effectiveness of its target instead.
pub fn compute_footprint(action: &Action) -> Footprint {
    let actor = action.id.actor;
    match &action.kind {
        // The **substrate** is the one read here that is not about the layer being
        // painted on. The tooth gates how much paint lands by the substrate's rise
        // over a reach in canvas px (§6.4), and `apply` takes both the substrate and
        // the scale it is laid at off the state being folded over, so a stroke does
        // not commute with either changing under it. Omitting it would let an undo of
        // a `SetSubstrate` splice past every stroke after it, leaving those tiles
        // toothed by a substrate the log no longer contains — the §12.6 direction no
        // pixel can report.
        //
        // A **liquify** stroke (§6.13) reads further than it writes: the run it
        // composes through and the layer's paint out to `liquify_reads`, where the
        // base it resamples may lie; and it writes the run back beside its mark.
        // Declared by the brush's effect, which is what decides the render path
        // (§6.2), so the two cannot disagree.
        ActionKind::CommitStroke(rec) => {
            let mut reads = vec![
                Resource::Existence(rec.layer),
                Resource::Selection(actor),
                Resource::Substrate,
            ];
            let mut writes = vec![Resource::Paint(rec.layer, stroke_rect(rec))];
            if rec.brush.liquify().is_some() {
                reads.push(Resource::Paint(rec.layer, liquify_reads(rec)));
                reads.push(Resource::LiquifyRun(rec.layer));
                writes.push(Resource::LiquifyRun(rec.layer));
            }
            Footprint { reads, writes }
        }
        // A placed image is an `AddLayer` that arrives with paint and a name in it
        // (§23), so it joins them here — and claims the paint as the **whole layer**,
        // not as the box the image covers. Every other action that writes tiles has
        // to derive its box twice, here and where the tiles are planned, which is the
        // §12.6 hazard `fill_bounds` exists to remove. Here there is nothing to keep
        // in step: the layer did not exist before this action, so all of its paint is
        // this action's by construction, whatever box the image covers.
        ActionKind::AddLayer {
            id, carrier, above, ..
        }
        | ActionKind::AddFilter {
            id, carrier, above, ..
        }
        | ActionKind::PlaceImage {
            id, carrier, above, ..
        } => mint(*id, *carrier, *above),
        // The matte's anchor is a `Place` (§15.5), whose `anchor()` is the same
        // optional sibling the other three carry.
        ActionKind::AddMatte {
            id, carrier, at, ..
        } => mint(*id, *carrier, at.anchor()),
        // A copy is a function of *everything it copies* — every tile and every
        // property of every layer in the subtree — which is why the action names its
        // sources rather than only its root (§14.8). Claiming less would let a
        // duplicate commute with a stroke inside the group it copied, and the two
        // orders give different paint. The tree's shape is read too, and the
        // `StackOrder` write covers that read.
        ActionKind::DuplicateLayer { ids, .. } => Footprint {
            reads: ids.iter().map(|(src, _)| Resource::Layer(*src)).collect(),
            writes: ids
                .iter()
                .map(|(_, copy)| Resource::Layer(*copy))
                .chain([Resource::StackOrder])
                .collect(),
        },
        // A removal takes the **whole subtree**, so it writes everything about every
        // layer in it — which is why the action names them and why each gets the
        // coarse `Resource::Layer`.
        //
        // `StackOrder` alone does not cover it: it is about the tree's *shape*, so it
        // meets other structural edits and nothing else. A stroke on a carried layer
        // claims `Paint(child, rect)` and a slider claims `Prop(child, _)`, neither of
        // which overlaps `{Existence(id), StackOrder, Paint(id, ALL)}` — the pair
        // would be judged to commute, and the undo's splice and a canonical replay
        // would then disagree with no pixel able to report it (§12.6).
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
        // One write per layer moved, so a subtree translated as one gesture still
        // commutes with a rename beside it — and, because no paint action reads
        // `Prop::Translation`, with every stroke inside it (§14.12).
        ActionKind::TranslateLayers { moves } => Footprint {
            reads: moves
                .iter()
                .map(|(id, _)| Resource::Existence(*id))
                .collect(),
            writes: moves
                .iter()
                .map(|(id, _)| Resource::Prop(*id, Prop::Translation))
                .collect(),
        },
        // The cut is bounded by the author's mask, which the footprint cannot
        // measure, so the source's paint is claimed whole — `Transform`'s answer to
        // the same question. The mask is *consumed*, so it is a write.
        ActionKind::FloatSelection { layer, child, .. } => Footprint {
            reads: vec![Resource::Existence(*layer)],
            writes: vec![
                Resource::Paint(*layer, TileRect::ALL),
                Resource::Layer(*child),
                Resource::Selection(actor),
                Resource::StackOrder,
            ],
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
        // The rect-scoped transforms (§16.8, §16.9) cut only inside their rect and
        // paste only inside the map's image, so unlike the whole-plane affine they
        // can claim an honest box: the union of the two, padded a tile for the apron
        // reach. An unusable warp, whose image is unknown, falls back to the whole
        // layer — the safe direction. The map is stated on the canvas and the claim
        // is on the layer's tiles, so the box is brought into the layer's frame by
        // the action's own `frame` (§14.12) — the same shift `apply` makes, from the
        // same field, so the two cannot disagree about which tiles are meant.
        ActionKind::TransformPerspective {
            layer,
            map,
            translation: frame,
        } => mapped_write(*layer, actor, map.min, map.max, map.image_aabb(), *frame),
        ActionKind::TransformWarp {
            layer,
            map,
            translation: frame,
        } => mapped_write(*layer, actor, map.min, map.max, map.image_aabb(), *frame),
        // A fill reads the mask that bounds it and writes the paint its region
        // reaches — the same shape of footprint a stroke has. A fill bounded only
        // by the selection has no analytic box, so it claims the whole layer, the
        // conservative answer a transform gives for the same reason.
        ActionKind::Fill { layer, op, .. } => Footprint {
            reads: vec![Resource::Existence(*layer), Resource::Selection(actor)],
            writes: vec![Resource::Paint(*layer, fill_rect(op))],
        },
        // A merge is a function of **everything about both layers**, because that is
        // what its plan reads: the tiles it stacks, the blend, clip, opacity and
        // visibility that decide whether the merge is offered at all (§14.11), and
        // the filter, which for a filter source is both half the offer (a resampling
        // kind is declined, §14.11.7) and what the merge writes into the
        // destination's texels. It is a function of the tree's shape too, which
        // decides what "down" means, and the `StackOrder` write covers that read.
        //
        // A merge would silently *change its own answer* if one of those properties
        // moved past it, so claiming them is what keeps a concurrent blend-mode
        // change from commuting with a merge the mode would have refused.
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
                // `CompositeParams` and it is written as one. Two of the three are
                // the identity today, so claiming them costs a false conflict and
                // nothing else — the direction §12.6 says to err in. Claiming only
                // the opacity would rest this footprint's honesty on refusals made
                // in another file.
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
/// **Both halves are tested for finiteness here** and neither can be left to
/// [`claim`]: `Vec2::min`/`max` return the non-NaN operand, so a non-finite corner is
/// *swallowed* by the union rather than carried into [`TileRect::covering`]'s guard,
/// and the box arrives finite and tight-looking — the one answer §12.6 does not
/// permit. Testing the `rect` half here rather than leaving it to `usable`/`shape_ok`
/// at `apply` keeps this footprint's honesty out of another file.
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

/// What **every action that mints a layer** claims.
///
/// Both anchors are read — the sibling to insert above and the layer whose stack to
/// insert into — because either being absent changes where the layer lands (§14.8).
/// [`Resource::Layer`] for the minted id: what it writes is everything about a layer
/// that did not exist.
fn mint(id: LayerId, carrier: Option<LayerId>, anchor: Option<LayerId>) -> Footprint {
    Footprint {
        reads: [carrier, anchor]
            .into_iter()
            .flatten()
            .map(Resource::Existence)
            .collect(),
        writes: vec![Resource::Layer(id), Resource::StackOrder],
    }
}

/// A rect-scoped transform's claim: the map's source rect unioned with its image by
/// [`gated_rect`], both brought into the layer's frame first.
///
/// One body for both families: a correction made to the perspective arm and not to
/// the warp arm would be a §12.6 break showing up on exactly one of them.
fn mapped_write(
    layer: LayerId,
    actor: ActorId,
    min: Vec2,
    max: Vec2,
    image: Option<(Vec2, Vec2)>,
    frame: crate::geom::IVec2,
) -> Footprint {
    let frame = frame.as_vec2();
    Footprint {
        reads: vec![Resource::Existence(layer)],
        writes: vec![
            Resource::Paint(
                layer,
                gated_rect(
                    (min - frame, max - frame),
                    image.map(|(lo, hi)| (lo - frame, hi - frame)),
                ),
            ),
            Resource::Selection(actor),
        ],
    }
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
                translation: crate::geom::IVec2::ZERO,
            }),
        )
    }

    fn commutes(a: &Action, b: &Action) -> bool {
        !compute_footprint(a).conflicts(&compute_footprint(b))
    }

    /// Every [`Resource`] variant is visited, and one of each meets **exactly** what
    /// it names: itself, plus whatever [`Resource::Layer`] coarsens over.
    ///
    /// `overlaps` is the only match in the crate over this enum ending in a `_` arm —
    /// it cannot be spelled exhaustively without a quadratic match — and the one
    /// where a wrong answer silently diverges peers. So the visit is forced here
    /// instead: the match below does not compile until a new variant is named in it,
    /// and naming it means saying whether it is about a layer, the only thing
    /// `overlaps` needs to know. The length assertion is a reminder, not a guard.
    #[test]
    fn every_resource_is_visited_and_meets_exactly_what_it_names() {
        let layer = LayerId::solo(4);
        // One of every variant, each paired with whether it is about `layer` — the
        // question the coarse claim asks. Written out rather than read off
        // `Resource::layer`, the helper `overlaps` is built from: an expectation
        // computed the same way as the answer agrees with it by construction.
        let samples = [
            (Resource::Paint(layer, TileRect::ALL), true),
            (Resource::Existence(layer), true),
            (Resource::Prop(layer, Prop::Name), true),
            (Resource::Layer(layer), true),
            (Resource::LiquifyRun(layer), true),
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
                | Resource::LiquifyRun(_)
                | Resource::StackOrder
                | Resource::Selection(_)
                | Resource::Substrate
                | Resource::SubstrateColor
                | Resource::Guides => {}
            }
        }
        assert_eq!(
            samples.len(),
            10,
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
        .chain(Prop::ALL.iter().map(|p| Resource::Prop(a, *p)));
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
                number: Some(20),
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

    /// A liquify brush, for the strokes below.
    fn liquify(actor: u64, layer: LayerId, from: Vec2, to: Vec2, radius: f32) -> Action {
        let mut a = stroke(actor, layer, from, to, radius);
        if let ActionKind::CommitStroke(rec) = &mut a.kind {
            rec.brush.effect = super::super::brush::BrushEffect::Liquify(
                super::super::brush::LiquifyEffect::default(),
            );
        }
        a
    }

    /// The liquify stroke's declaration (§6.13): it reads the layer's paint out to
    /// its reach and the run, and writes the run beside its mark — so a paint
    /// stroke inside the reach conflicts, one beyond it commutes, and two liquify
    /// strokes on one layer never commute, however far apart.
    #[test]
    fn a_liquify_stroke_claims_its_reach_and_the_run() {
        let warp = liquify(1, LayerId::ROOT, Vec2::ZERO, Vec2::splat(60.0), 16.0);
        let reach = super::super::brush::LiquifyEffect::REACH_PX;
        // Well inside the reach, but outside the mark's own padded tiles.
        let within = stroke(
            2,
            LayerId::ROOT,
            Vec2::splat(reach * 0.5),
            Vec2::splat(reach * 0.5 + 40.0),
            8.0,
        );
        let beyond = stroke(
            2,
            LayerId::ROOT,
            Vec2::splat(reach * 3.0),
            Vec2::splat(reach * 3.0 + 40.0),
            8.0,
        );
        assert!(!commutes(&warp, &within), "a paint inside the reach");
        assert!(commutes(&warp, &beyond), "a paint beyond the reach");
        let far_warp = liquify(
            2,
            LayerId::ROOT,
            Vec2::splat(reach * 3.0),
            Vec2::splat(reach * 3.0 + 40.0),
            16.0,
        );
        assert!(
            !commutes(&warp, &far_warp),
            "two liquify strokes share the run"
        );
        let other_layer = liquify(2, LayerId::solo(1), Vec2::ZERO, Vec2::splat(60.0), 16.0);
        assert!(commutes(&warp, &other_layer), "runs are per layer");
        let f = compute_footprint(&warp);
        assert!(
            f.writes.contains(&Resource::LiquifyRun(LayerId::ROOT)),
            "the run is written"
        );
        assert!(
            f.reads.contains(&Resource::LiquifyRun(LayerId::ROOT)),
            "the run is read"
        );
        // A plain stroke says nothing about the run.
        let plain = compute_footprint(&stroke(1, LayerId::ROOT, Vec2::ZERO, Vec2::ONE, 8.0));
        assert!(
            !plain
                .reads
                .iter()
                .chain(&plain.writes)
                .any(|r| matches!(r, Resource::LiquifyRun(_))),
        );
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
    /// (§6.4), and `apply` reads both off the state being folded over, so an undo
    /// that spliced one out past its strokes would leave them toothed by a substrate
    /// the log no longer contains. Whose stroke it is makes no difference, unlike the
    /// mask: a substrate is shared document state.
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
                number: Some(7),
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

    /// **A translate commutes with a stroke on the same layer** — the claim §14.12's
    /// whole design exists to make true: a stroke's geometry is in the layer's frame
    /// and its mask offset travels in the action, so neither reads what the translate
    /// writes. Two translates of one layer still conflict, as two opacities do.
    #[test]
    fn a_translate_commutes_with_paint_but_not_with_itself() {
        use crate::geom::IVec2;
        let translate = act(
            1,
            ActionKind::TranslateLayers {
                moves: vec![(LayerId::ROOT, IVec2::new(40, -8))],
            },
        );
        let paint = stroke(2, LayerId::ROOT, Vec2::ZERO, Vec2::splat(50.0), 8.0);
        let again = act(
            2,
            ActionKind::TranslateLayers {
                moves: vec![(LayerId::ROOT, IVec2::ZERO)],
            },
        );
        let rename = act(2, ActionKind::SetLayerName(LayerId::ROOT, None));
        let elsewhere = act(
            2,
            ActionKind::TranslateLayers {
                moves: vec![(LayerId::solo(1), IVec2::ONE)],
            },
        );
        assert!(commutes(&translate, &paint));
        assert!(!commutes(&translate, &again));
        assert!(commutes(&translate, &rename));
        assert!(commutes(&translate, &elsewhere));
        // …and not with the layer being removed, through the coarse claim.
        let remove = act(
            2,
            ActionKind::RemoveLayer {
                id: LayerId::ROOT,
                carried: Vec::new(),
            },
        );
        assert!(!commutes(&translate, &remove));
    }

    /// A float claims the source's paint whole and the child entire, so it conflicts
    /// with paint on either side and with its author's selection edits — and still
    /// commutes with a stroke on an unrelated layer.
    #[test]
    fn a_float_conflicts_with_paint_on_both_of_its_layers() {
        use crate::geom::IVec2;
        let child = LayerId::solo(7);
        let float = act(
            1,
            ActionKind::FloatSelection {
                layer: LayerId::ROOT,
                child,
                translation: IVec2::ZERO,
                number: Some(2),
            },
        );
        let on_source = stroke(2, LayerId::ROOT, Vec2::ZERO, Vec2::splat(50.0), 8.0);
        let on_child = stroke(2, child, Vec2::ZERO, Vec2::splat(50.0), 8.0);
        let own_select = act(1, ActionKind::InvertSelection);
        let elsewhere = stroke(2, LayerId::solo(1), Vec2::ZERO, Vec2::splat(50.0), 8.0);
        assert!(!commutes(&float, &on_source));
        assert!(!commutes(&float, &on_child));
        assert!(!commutes(&float, &own_select));
        assert!(commutes(&float, &elsewhere));
    }

    /// **A stretched tip's footprint has to grow with it** (§6.6, §12.6).
    ///
    /// A brush drawn out along its facing axis reaches `elongation` times as far as
    /// its radius names, and that is the bound the *renderer* is held to
    /// (`gpu::stroke::segments::Sweep::reach`). A pad covering only the square stamp's
    /// √2 would let a leaned pencil paint past the tiles its action claimed: not a
    /// visible bug but a §12.6 one.
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
                translation: crate::geom::IVec2::ZERO,
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
    /// at the origin. A non-finite radius or path point producing a tight-looking
    /// footprint is the one direction §12.6 cannot survive: a distant stroke commutes
    /// past it, the fast path splices on a lie, and no pixel can show it.
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
    /// operand, so a non-finite corner against a finite image never reaches
    /// [`TileRect::covering`]'s guard: the claim comes back tight and wrong. That is
    /// the under-claim §12.6 cannot survive, and it is invisible in a picture, since
    /// `apply` refuses such a map anyway.
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
                        translation: crate::geom::IVec2::ZERO,
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
                        translation: crate::geom::IVec2::ZERO,
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
                translation: crate::geom::IVec2::ZERO,
            },
        );
        assert!(!claims_all(&ordinary));
        assert!(commutes(&ordinary, &elsewhere));

        // A frame moves the claim with it: the same gated map minted while the
        // layer's frame sat elsewhere names the layer's *own* tiles, so two maps
        // over the same canvas rect under different frames claim different boxes.
        let framed = act(
            1,
            ActionKind::TransformPerspective {
                layer: LayerId::ROOT,
                map: PerspectiveMap {
                    min: lo,
                    max: hi,
                    corners: rect_corners(lo, hi),
                },
                translation: crate::geom::IVec2::new(4000, 0),
            },
        );
        let box_of = |a: &Action| {
            compute_footprint(a)
                .writes
                .iter()
                .find_map(|r| match r {
                    Resource::Paint(_, rect) => Some(*rect),
                    _ => None,
                })
                .expect("a gated transform claims paint")
        };
        assert_ne!(box_of(&ordinary), box_of(&framed));
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
