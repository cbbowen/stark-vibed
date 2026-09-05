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

/// The two halves of resuming a stroke across ranges: what the range before this one
/// left, and whether a range after it will want what this one leaves.
///
/// One value because they are one question asked from both ends, and because the
/// second half was being derived three times — once in each path that has cross-piece
/// state — from `spans` and `rec`, which every path had for its own reasons. Two of
/// the three then ignored the answer and captured unconditionally, which is the kind
/// of drift a shared derivation does not have.
#[derive(Clone, Copy)]
pub(super) struct Resume<'a> {
    /// What the previous range left, if this is not the first.
    pub(super) prior: Option<&'a ToolState>,
    /// Whether a later range will resume this stroke.
    ///
    /// **False is the common case**, and that is the point: the live tail is exactly
    /// the range that reaches the end of the stroke, and it re-renders on every
    /// pointer move. Capturing there builds state with no reader and holds the
    /// parcel's working lanes — or the loop's reservoir — out of the pool for the
    /// length of the fold that discards them.
    pub(super) capture: bool,
}

impl<'a> Resume<'a> {
    /// Read off the range in hand against the whole path.
    pub(super) fn of(
        rec: &stark_model::document::StrokeRecord,
        spans: &super::StrokeSpans,
        prior: Option<&'a ToolState>,
    ) -> Self {
        Self {
            prior,
            capture: spans.range.end < crate::path::span_count(rec.path.len()),
        }
    }
}

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
    /// full opacity or under a pen-driven ceiling — the settings under which
    /// that path stops being stateless, because a scaled parcel is not
    /// composable per piece.
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
    /// budgets against, and whose `.w` is the ceiling lane a pen-driven opacity
    /// claims coverage in (`dynamics.wesl::lay_parcel`) — cut per tile exactly
    /// as the write-back cuts paint. Empty at full opacity, where the lanes are
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
    /// The ceiling lane over the same tiles (§6.2): the gated mass and moment sums
    /// a pen-driven opacity claims coverage with, drawn into the region per
    /// segment and cut per tile exactly as `fresh` is. Empty unless the pen drives
    /// the ceiling.
    pub(super) levels: BTreeMap<TileCoord, Arc<Kept>>,
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
    pub progress: Progress,
}

/// Whether a range render happened at all. One value with what a finished range
/// hands on, so a range that drew nothing *yet* cannot also report a tool or dirty
/// tiles for a caller to freeze.
pub(crate) enum Progress {
    /// The range is finished — drawn, or empty of geometry, which is the ordinary
    /// empty return — and this is what it hands on.
    Finished {
        /// The brush state to resume with, for a stroke that runs the stamp loop.
        /// `None` means *nothing changed*: the swept path carries no state at all, a
        /// range that reaches the end of the stroke has nothing following it to hand
        /// off to, and a range with no geometry leaves the brush as it found it — so
        /// a caller holding earlier state should keep it rather than treat this as
        /// a reset.
        tool: Option<ToolState>,
        /// The tiles this range rewrote: every coordinate the returned map holds a
        /// fresh handle at.
        ///
        /// A **superset** of the tiles whose pixels changed, deliberately. What the
        /// renderer enumerates is where the stroke's geometry *reaches*
        /// (`region::cover`), and a tile at the very edge of that reach can receive
        /// a fresh copy-on-write tile whose every fragment differenced its prefix-τ
        /// taps to zero — bit-identical to the base, and still listed here.
        /// Narrowing it would mean comparing pixels, which is the whole cost this
        /// field exists to avoid.
        ///
        /// Reporting them is what lets several in-flight strokes be composited over
        /// one committed document without diffing whole tile maps (§17.6), and a
        /// superset costs that a redundant composite rather than a wrong picture.
        dirty: Vec<TileCoord>,
    },
    /// **This range drew nothing, and will have to be drawn again** — the brush's
    /// stamp asset has not arrived yet (`StrokeRenderer::render_range`), and `dist`
    /// stands where the range began. A caller that freezes ranges must not freeze
    /// this one: the commit renders the stroke once with the asset present, so a
    /// head that took it would measure every later `drain` falloff and
    /// colour-dynamics tap from an arc length the commit does not (§1.3) — and
    /// nothing bumps the preview's epoch to repair it, because no *document*
    /// changed when the asset landed.
    Deferred,
}

impl StrokeCarry {
    /// Nothing drawn because there was nothing to draw: the range is finished, the
    /// arc clock stands at `dist`, and the brush is as it was found.
    pub(crate) fn unchanged(dist: f32) -> Self {
        Self {
            dist,
            progress: Progress::Finished {
                tool: None,
                dirty: Vec::new(),
            },
        }
    }

    /// Nothing drawn *yet* — [`Progress::Deferred`]. `dist` is where the range
    /// began, because that is still where the stroke has got to.
    pub(crate) fn deferred(dist: f32) -> Self {
        Self {
            dist,
            progress: Progress::Deferred,
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The two ways a carry says "nothing drawn" are two different answers: a
    /// finished range hands on nothing, a deferred one hands on no answer at all.
    #[test]
    fn unchanged_is_finished_and_deferred_is_not() {
        let finished = StrokeCarry::unchanged(3.0);
        assert_eq!(finished.dist, 3.0);
        assert!(matches!(
            finished.progress,
            Progress::Finished { tool: None, ref dirty } if dirty.is_empty()
        ));
        let deferred = StrokeCarry::deferred(3.0);
        assert_eq!(deferred.dist, 3.0, "the arc clock did not move");
        assert!(matches!(deferred.progress, Progress::Deferred));
    }
}
