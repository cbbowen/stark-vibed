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

use super::scratch::Kept;
use crate::gpu::tile::TilePairHandle;

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
pub struct ToolState(pub(super) Carried);

/// The kinds of cross-piece stroke state, one per path that has any — the
/// swept deposit at full opacity carries nothing at all.
pub(super) enum Carried {
    /// The stamp loop's tool reservoir (§6.2). Boxed for size alone: three
    /// pooled leases dwarf the erase map's header, and a `ToolState` moves
    /// through `Option::or` on every fold.
    Loop(Box<Reservoir>),
    /// The erase pass's accumulated extent (§6.12).
    Erase(EraseCarry),
    /// The swept deposit's accumulated parcel (§6.2), carried only below
    /// full opacity — the one setting under which that path stops being
    /// stateless, because a scaled parcel is not composable per piece
    /// ([`SweepCarry`]).
    Sweep(SweepCarry),
}

impl ToolState {
    /// The loop's reservoir, on a carry the loop captured.
    ///
    /// The other kind is unreachable rather than an error to handle: which path a
    /// stroke takes is a pure function of its brush (§6.2), a carry resumes only
    /// the stroke that captured it, and the brush is snapshotted when the gesture
    /// starts — so the loop can only ever be handed its own kind back.
    pub(super) fn reservoir(&self) -> &Reservoir {
        match &self.0 {
            Carried::Loop(r) => r,
            Carried::Erase(_) | Carried::Sweep(_) => {
                unreachable!(
                    "another path's carry resumed the stamp loop; the path is a pure function of the brush (§6.2)"
                )
            }
        }
    }

    /// The erase pass's accumulation, on a carry it captured — [`reservoir`](Self::reservoir)'s
    /// argument, from the other side.
    pub(super) fn erased(&self) -> &EraseCarry {
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
    /// [`reservoir`](Self::reservoir)'s argument, from the third side. The
    /// opacity is part of the brush, so whether the swept path carries at all is
    /// as much a pure function of it as which path runs.
    pub(super) fn swept(&self) -> &SweepCarry {
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

/// The stamp loop's carried state (§6.2).
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

/// The erase pass's carried state (§6.12): per touched tile, the stroke's
/// accumulated extent and the tile it is measured against.
///
/// Where the reservoir is brush-local, this is **canvas-local** — the accumulated
/// transparency mass is a field over the tiles the stroke has reached — and that
/// is what it exists to be: the pass turns the *whole* stroke's extent on the
/// pristine paint at once (`erase.wesl`), so a piece needs everything the pieces
/// before it accumulated, per texel, where the loop needs only what the tip
/// carries.
pub(super) struct EraseCarry {
    pub(super) tiles: BTreeMap<TileCoord, EraseTile>,
}

/// One tile's share of an [`EraseCarry`].
pub(super) struct EraseTile {
    /// The layer's tile as the stroke found it — the paint every piece's rewrite
    /// is derived from. A piece rendered later must not read the *output* of an
    /// earlier one (the base it is handed holds exactly that), or the erase would
    /// compound per piece instead of per stroke. An `Arc`'d pool handle, so
    /// keeping it is a refcount, not a copy.
    pub(super) pristine: TilePairHandle,
    /// The stroke's transparency mass over this tile, summed so far
    /// (`stamp.wesl::fs_erase`). Shared between successive carries rather than
    /// copied: a piece never writes the accumulator it resumed from — it copies
    /// into a fresh working texture and extends that — so the tiles a piece does
    /// not touch ride forward as clones of the same lease.
    pub(super) accum: Arc<Kept>,
}

/// The swept deposit's carried state below full opacity (§6.2): per touched
/// tile, the stroke's accumulated **parcel** and the tile it will land on.
///
/// [`EraseCarry`]'s design at the other end of the same theorem. The opacity
/// scales the parcel's finished coverage, and a scaled coverage is neither
/// additive in `τ` nor `1 − exp(−k·τ)` — so it cannot be applied per piece, and
/// the parcel has to keep accumulating across the pieces of a live stroke with
/// every piece re-deriving its tiles from pristine paint under the whole of it
/// (`integrate.wesl`). At opacity 1 the integrate is the identity on the parcel,
/// pieces compose by plain stacking, and the path carries nothing — which is why
/// this exists only below it.
pub(super) struct SweepCarry {
    pub(super) tiles: BTreeMap<TileCoord, SweepTile>,
}

/// One tile's share of a [`SweepCarry`].
pub(super) struct SweepTile {
    /// The layer's tile as the stroke found it — [`EraseTile::pristine`]'s
    /// contract exactly. `None` is bare canvas: unlike an erase, a deposit onto
    /// nothing mints a tile, so the pass keeps painting and binds the 1×1 zeroes
    /// as the base it re-derives from.
    pub(super) pristine: Option<TilePairHandle>,
    /// The stroke's parcel over this tile so far — the same scratch pair a
    /// single-piece sweep accumulates, kept alive across pieces. Shared, never
    /// rewritten: a resuming piece copies it into fresh working textures and
    /// extends those ([`EraseTile::accum`]'s contract).
    pub(super) accum: Arc<SweepAccum>,
}

/// The textures of one carried parcel tile: the over-blended premultiplied
/// color, the additive wide aux (height, optical mass), and the residual in a
/// space that has one — the same three targets `stamp.wesl`'s deposit writes,
/// because this *is* that scratch, surviving its piece.
pub(super) struct SweepAccum {
    pub(super) color: Kept,
    pub(super) aux: Kept,
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
