//! Drawing a stroke in pieces, and resuming where the last one stopped (§6.2).
//!
//! A live stroke re-renders on every pointer move, so it must cost its *tail* rather
//! than its length. That works because a stroke can be cut into ranges of spans and
//! composited in order for the same result as one pass — the swept path because its
//! deposit is a definite integral that composes by summing optical depth, the stamp
//! loop because [`ToolState`] carries the only thing it threads between segments that
//! is not already on the canvas.
//!
//! The catch is anything measured against the **whole** stroke — the trailing
//! taper — which a stroke still under the pointer does not know yet.
//! [`safe_frozen`] is the one rule that holds it back until its answer can no
//! longer change, and it is what makes `preview == committed` hold here (§1.3).

use std::collections::BTreeMap;
use std::sync::Arc;

use stark_model::document::StrokeRecord;
use stark_model::geom::TileCoord;

use super::accum::ParcelCarry;
use crate::gpu::scratch::Kept;

/// A stroke's carried state at a cut point (§6.2) — what its path threads
/// between pieces that does not already live on the canvas. Which of the two kinds
/// it holds follows the path itself, and cannot disagree with it: the path is a
/// pure function of the brush ([`dynamics_setup`](super::dynamics::dynamics_setup)),
/// and a carry only ever resumes the stroke that captured it.
///
/// The textures ride in [`Kept`] leases rather than being created and destroyed
/// per pointer move: one of these is captured per fold, and the pool hands the
/// same textures back on drop — which is sound *because* a run only ever borrows a
/// `ToolState`, so the drop that returns a lease can only happen after the run's
/// own submit (see [`Kept`]).
pub(crate) struct ToolState(pub(super) Carried);

/// The kinds of cross-piece stroke state, one per path that has any — the
/// swept deposit at full opacity carries nothing at all.
pub(super) enum Carried {
    /// The stamp loop's state (§6.2, §6.12's pattern where it has a
    /// canvas-local half). Boxed for size alone: three pooled leases dwarf the
    /// erase map's header, and a `ToolState` moves through `Option::or` on
    /// every fold.
    Loop(Box<LoopCarry>),
    /// The erase pass's accumulated extent (§6.12).
    Erase(ParcelCarry),
    /// The swept deposit's accumulated parcel (§6.2), carried only below
    /// full opacity — the one setting under which that path stops being
    /// stateless, because a scaled parcel is not composable per piece.
    Sweep(ParcelCarry),
}

// The two above hold the *same* payload, and one variant each is still worth
// having: the lane count of a `ParcelCarry`'s parcels is the effect's — one
// transparency mass against the channel trio — so a carry crossing between them
// would bind the wrong number of textures. The accessors below are where that is
// said, and they are the only reason the distinction is kept.

impl ToolState {
    /// The loop's carried state, on a carry the loop captured.
    ///
    /// The other kinds are unreachable rather than an error to handle: which path a
    /// stroke takes is a pure function of its brush (§6.2), a carry resumes only
    /// the stroke that captured it, and the brush is snapshotted when the gesture
    /// starts — so the loop can only ever be handed its own kind back.
    pub(super) fn looped(&self) -> &LoopCarry {
        match &self.0 {
            Carried::Loop(l) => l,
            Carried::Erase(_) | Carried::Sweep(_) => {
                unreachable!(
                    "another path's carry resumed the stamp loop; the path is a pure function of the brush (§6.2)"
                )
            }
        }
    }

    /// The erase pass's accumulation, on a carry it captured — [`looped`](Self::looped)'s
    /// argument, from the other side.
    pub(super) fn erased(&self) -> &ParcelCarry {
        match &self.0 {
            Carried::Erase(e) => e,
            Carried::Loop(_) | Carried::Sweep(_) => {
                unreachable!(
                    "another path's carry resumed the erase pass; the path is a pure function of the brush (§6.2)"
                )
            }
        }
    }

    /// The scaled swept deposit's accumulated parcel, on a carry it captured —
    /// [`looped`](Self::looped)'s argument, from the third side. The
    /// opacity is part of the brush, so whether the swept path carries at all is
    /// as much a pure function of it as which path runs.
    pub(super) fn swept(&self) -> &ParcelCarry {
        match &self.0 {
            Carried::Sweep(s) => s,
            Carried::Loop(_) | Carried::Erase(_) => {
                unreachable!(
                    "another path's carry resumed the scaled swept deposit; the path is a pure function of the brush (§6.2)"
                )
            }
        }
    }
}

/// The stamp loop's carried state (§6.2): the tool reservoir, and — below full
/// opacity — the mint budget's raw totals.
pub(super) struct LoopCarry {
    /// The brush-local half: what the tip is carrying.
    pub(super) reservoir: Reservoir,
    /// The canvas-local half: per touched tile, the opacity ceiling's running
    /// **raw mint totals** — the region aux whose `.yz` lanes the capped mint
    /// budgets against (`dynamics.wesl::lay_parcel`), cut per tile exactly as
    /// the write-back cuts paint. Empty at full opacity, where the lanes are
    /// never read: the identity ceiling needs no budget, and the common case
    /// carries and copies nothing.
    ///
    /// [`ParcelTile::accum`](super::accum::ParcelTile)'s sharing contract: a piece
    /// never writes a tile it resumed from — it seeds its region by *copying*
    /// these in and extracts fresh leases out — so the tiles a piece does not
    /// touch ride forward as clones of the same lease, and the live tail
    /// re-renders from the same frozen totals every pointer move.
    ///
    /// That contract is all these share with the two parcel carries, which is why
    /// they are not one type: no pristine handle beside them (the loop does not
    /// re-derive from pristine paint — the budget is running state, exactly like
    /// the reservoir), no pass ever binds one, they are seeded into a shared
    /// region at a per-tile offset rather than into a working texture per tile,
    /// and the tiles they end up in come from the region write-back rather than
    /// from a landing pass.
    pub(super) fresh: BTreeMap<TileCoord, Arc<Kept>>,
}

/// The loop's tool reservoir (§6.2).
///
/// The sequential loop threads exactly two things from one segment to the next that
/// do not already live on the canvas: the **tool reservoir** — what paint the tip is
/// carrying, and where on the tip it sits — and how far the tip has travelled since
/// it last exchanged with the canvas. Remember those at a span boundary and the rest
/// of the stroke can be drawn later, over the already-composited head, for the same
/// result as one pass. That is what lets a `lift`/`deposit`/`charge` brush get the
/// same incremental repaint the swept path gets.
///
/// The reservoir is brush-*local*, which is why this works at all: it says nothing
/// about where the stroke is, so the region rectangle may change completely between
/// the piece that produced this state and the piece that resumes from it. (The mint
/// budget beside it in [`LoopCarry`] is the one canvas-local exception, and it is
/// addressed by tile for exactly that reason.)
pub(super) struct Reservoir {
    /// Reservoir color: per texel, the latent paint (rgb) and its per-unit opacity.
    pub(super) color: Kept,
    /// Reservoir aux: per texel, the carried amount (height).
    pub(super) aux: Kept,
    /// Reservoir residual (§6.7), in a space that has one: the rest of the
    /// color above. Carried across ranges for the same reason the color is — a tip
    /// that picked up black paint has to still be carrying black when the next
    /// pointer move resumes it, and the concentrations alone cannot say so.
    pub(super) resid: Option<Kept>,
}

// The other two kinds of carried state — the erase pass's accumulated extent
// (§6.12) and the swept deposit's accumulated parcel below full opacity (§6.2) —
// are one type, [`ParcelCarry`] in `accum`, because they are one *procedure*: an
// effect whose law is neither of §6.2's two composable forms has to keep the
// composable half summing across pieces and apply the law once per render from
// pristine paint, and there is no second way to do that. Where the reservoir above
// is brush-local, both of those are **canvas-local** — a field over the tiles the
// stroke has reached — which is why they are addressed by tile and it is not.

/// What a range render leaves behind for the range that resumes after it.
pub(crate) struct StrokeCarry {
    /// Arc length at the end of the range. Not derivable from the span index — it is
    /// measured along the flattened polyline — and both the `drain` falloff and the
    /// color-dynamics noise read it, so restarting it at zero would make the middle
    /// of a stroke look like the start of one.
    pub dist: f32,
    /// The brush state to resume with, for a stroke that runs the stamp loop. `None`
    /// means *nothing changed*: the swept path carries no state at all, a range that
    /// reaches the end of the stroke has nothing following it to hand off to, and a
    /// range with no geometry leaves the brush as it found it — so a caller holding
    /// earlier state should keep it rather than treat this as a reset.
    pub tool: Option<ToolState>,
    /// The tiles this range rewrote: every coordinate the returned map holds a fresh
    /// handle at.
    ///
    /// A **superset** of the tiles whose pixels changed, deliberately. What the
    /// renderer enumerates is where the stroke's geometry *reaches*
    /// (`region::cover`), and a tile at the very edge of that reach can
    /// receive a fresh copy-on-write tile whose every fragment differenced its prefix-τ
    /// taps to zero — bit-identical to the base, and still listed here. Narrowing it
    /// would mean comparing pixels, which is the whole cost this field exists to avoid.
    ///
    /// Reporting them is what lets several in-flight strokes be composited over one
    /// committed document without diffing whole tile maps (§17.6), and a superset costs
    /// that a redundant composite rather than a wrong picture.
    pub dirty: Vec<TileCoord>,
}

/// How many leading spans of a *live* stroke may be rendered once and kept, given
/// that the fitter has settled `frozen` of them (§6.2).
///
/// Freezing is what makes a long live stroke cost its tail rather than its length
/// ([`StrokeRenderer::render_range`](super::StrokeRenderer::render_range)), and it rests on a frozen span's pixels being
/// final. Everything the sweep measures against the **whole** stroke breaks that on
/// its own terms, because while the pointer is down the whole stroke has not happened
/// yet. Bake such a quantity into a span too early and the stroke carries something
/// the commit does not — the live == committed invariant (§1.3), failing in the one
/// place it cannot be repainted. The taper is that quantity, and this is the one
/// rule that holds it back until the answer can no longer change.
///
/// A span is held back unless both are already settled for it:
///
/// * it is at least the trailing taper's length before the stroke's end, so its
///   trailing factor is 1 — and stays 1, since the stroke only gets longer;
/// * it is at least the leading taper's length past the start, which together with
///   the first condition proves the stroke is already longer than the two zones
///   together, so the "scale both to fit" compression ([`Taper`](super::segments::Taper))
///   is 1 and likewise stays 1.
///
/// Both are tested on **chords**, which under-estimate arc length — so a span
/// this admits genuinely satisfies them, and a stroke that doubles back near its own
/// start or end merely re-renders a little more than it had to. Only the last span
/// in the candidate prefix is tested: arc length
/// increases monotonically along the stroke, so it is the hardest case, and once a
/// prefix is admitted it stays admissible however the stroke continues (which is what
/// lets a kept head survive this shrinking under it).
///
/// The stroke's start is its **marker** ([`StrokeRecord::start`]), not the
/// curve's head: the leading taper is measured from where the deposit begins
/// (§6.2). A cut behind the marker passes or fails these chord tests
/// meaninglessly and harmlessly — a prefix that ends behind the marker renders
/// nothing, and final pixels of nothing are final. Freezing a span past a
/// marker that could still move is ruled out structurally: the marker is placed
/// by the fitter's arc profile, whose prefix settles exactly as spans freeze,
/// so by the time a span beyond it may freeze the marker is already final
/// (`PathFitter::start_on`).
pub(crate) fn safe_frozen(rec: &StrokeRecord, frozen: usize) -> usize {
    let (start_px, end_px) = rec.brush.taper_px();
    let last = crate::path::span_count(rec.path.len());
    if last == 0 {
        return frozen;
    }
    let head = crate::path::point_at(&rec.path, rec.start);
    let tip = crate::path::span_end(&rec.path, last - 1);
    let mut spans = frozen.min(last);
    while spans > 0 {
        let cut = crate::path::span_end(&rec.path, spans - 1);
        if (tip - cut).length() >= end_px && (cut - head).length() >= start_px {
            break;
        }
        spans -= 1;
    }
    spans
}

/// Which part of a stroke to build segments for, and the arc length its first
/// sample carries.
///
/// `dist` is not derivable from `range` — it is the arc length accumulated along
/// everything *before* it — so an incremental caller has to carry it forward. It
/// matters because the `drain` falloff and the color-dynamics noise are both
/// parameterized by distance travelled: restarting it at zero would make the tail
/// of a stroke look like the head of one.
#[derive(Clone, Debug)]
pub(crate) struct StrokeSpans {
    pub(super) range: std::ops::Range<usize>,
    pub(super) dist: f32,
}

impl StrokeSpans {
    /// The whole stroke, from the beginning.
    pub(crate) fn whole(rec: &StrokeRecord) -> Self {
        StrokeSpans {
            range: 0..crate::path::span_count(rec.path.len()),
            dist: 0.0,
        }
    }

    /// A checked cut through `rec`, carrying the arc distance already consumed by
    /// its predecessor. Keeping construction here means a caller cannot resume a
    /// stroke from an inverted/out-of-bounds span range or a non-finite distance.
    pub(crate) fn from_parts(rec: &StrokeRecord, range: std::ops::Range<usize>, dist: f32) -> Self {
        let last = crate::path::span_count(rec.path.len());
        assert!(
            range.start <= range.end && range.end <= last,
            "stroke span range {range:?} is outside 0..{last}",
        );
        assert!(
            dist.is_finite() && dist >= 0.0,
            "stroke span distance must be finite and non-negative, got {dist}",
        );
        Self { range, dist }
    }

    pub(crate) fn dist(&self) -> f32 {
        self.dist
    }
}
