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

use stark_model::document::StrokeRecord;
use stark_model::geom::TileCoord;

use super::scratch::Kept;

/// The stamp loop's carried state at a cut point in a stroke (§6.2).
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
/// the piece that produced this state and the piece that resumes from it.
/// The textures ride in [`Kept`] leases rather than being created and destroyed
/// per pointer move: one of these is captured per fold, and the pool hands the
/// same textures back on drop — which is sound *because* a run only ever borrows a
/// `ToolState`, so the drop that returns a lease can only happen after the run's
/// own submit (see [`Kept`]).
pub struct ToolState {
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

/// What a range render leaves behind for the range that resumes after it.
pub struct StrokeCarry {
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
    /// (`segments::affected_tiles`), and a tile at the very edge of that reach can
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
pub struct StrokeSpans {
    pub range: std::ops::Range<usize>,
    pub dist: f32,
}

impl StrokeSpans {
    /// The whole stroke, from the beginning.
    pub fn whole(rec: &StrokeRecord) -> Self {
        StrokeSpans {
            range: 0..crate::path::span_count(rec.path.len()),
            dist: 0.0,
        }
    }
}
