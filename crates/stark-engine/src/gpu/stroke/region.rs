//! The region a stroke piece needs: which tiles its sweeps touch, the box they span,
//! and where the stamp loop cuts one stroke into pieces (§6.2, §6.4).
//!
//! The invariant that lives here: **the rectangle the chunker measures a piece against
//! is the rectangle the render then allocates.** One [`Coverage`] answers both, and
//! [`Covered::rect`] takes its extent from the very [`dims`](Coverage::dims) the
//! chunker checked.
//!
//! Nothing here touches the GPU — float arithmetic over [`Sweep`]s and tile
//! coordinates, so all of it is testable without an adapter.

use std::collections::BTreeSet;
use std::ops::Range;

use stark_model::geom::{TILE_APRON, TILE_SIZE, TILE_TEX, TileCoord, Vec2};

use super::budget::{MAX_REGION_DIM, MAX_STAMPS, REGION_BUDGET_DIM};
use super::segments::{BleedFire, Segment, Sweep};

/// Call `f(segment index, tile)` for every tile whose *texture* (interior + apron) a
/// segment's swept capsule overlaps, in segment order, and return the box walked.
///
/// The apron is included in the reach so a stroke landing within a tile's interior
/// but inside a neighbour's apron band re-renders that neighbour too, keeping the
/// shared apron/interior overlap bit-identical (§6.4).
///
/// **A segment writes exactly zero outside the tiles this names**, which is what lets
/// [`tiles_with_segments`] hand each tile a subset rather than the whole stroke. The
/// rasterized geometry does reach further at the caps, but out there a fragment
/// differences two equal prefix-τ taps and writes nothing (see [`coverage_bounds`]) —
/// and zero is an exact identity through both blends, so which segments a tile is
/// handed cannot change what lands in it.
fn for_each_touched<'a>(
    sweeps: impl Iterator<Item = &'a Sweep>,
    mut f: impl FnMut(usize, TileCoord),
) -> Coverage {
    let tile = TILE_SIZE as f32;
    let mut bounds = Coverage::default();
    for (i, s) in sweeps.enumerate() {
        let (lo, hi) = segment_bounds(s);
        bounds.lo = bounds.lo.min(lo);
        bounds.hi = bounds.hi.max(hi);
        let (x0, x1) = ((lo.x / tile).floor() as i32, (hi.x / tile).floor() as i32);
        let (y0, y1) = ((lo.y / tile).floor() as i32, (hi.y / tile).floor() as i32);
        for y in y0..=y1 {
            for x in x0..=x1 {
                f(i, TileCoord::new(x, y));
            }
        }
    }
    bounds
}

/// Every sweep a piece will rasterize: its painting segments **and its bleed firings'
/// windows**, in the order the plan dispatches them.
///
/// The windows belong in every accounting because they write: a firing's sweep is
/// walked back along the crossing segment's own arc, up to one
/// `dynamics::bleed::BLEED_TRAVEL_QUANTUM` before the segment it fires after
/// (`plan::bleed_fires`), which for a piece's first segment lies behind every segment
/// box. Left out, that flux is silently clipped by the region's bounds check and a
/// rewritten tile's apron can diverge from an unrewritten neighbour's interior (§6.4).
///
/// One function, so no caller can enumerate the segments and forget the windows.
pub(super) fn piece_sweeps<'a>(
    segments: &'a [Segment],
    fires: &'a [BleedFire],
) -> impl Iterator<Item = &'a Sweep> + Clone {
    segments
        .iter()
        .map(|s| &s.sweep)
        .chain(fires.iter().map(|f| &f.window))
}

/// The canvas box a set of sweeps covers, accumulated one sweep at a time.
///
/// **The one definition of what a piece needs.** The chunker's promise — "this piece
/// fits a region" — and the region the loop actually allocates are the same rectangle
/// because they are the same arithmetic: [`Covered::rect`] takes its extent from
/// [`dims`](Self::dims), the very function the chunker checked.
///
/// The tile set is enumerated separately, because only the dynamics path wants it and a
/// set insert per tile per segment is the cost the incremental repaint exists to keep
/// off a long stroke.
#[derive(Clone, Copy)]
pub(super) struct Coverage {
    lo: Vec2,
    hi: Vec2,
}

impl Default for Coverage {
    /// The empty box, which absorbs into any other: `min`/`max` against infinities.
    fn default() -> Self {
        Self {
            lo: Vec2::splat(f32::INFINITY),
            hi: Vec2::splat(f32::NEG_INFINITY),
        }
    }
}

impl Coverage {
    /// Grow to hold one more sweep, apron included ([`segment_bounds`]).
    fn add(&mut self, s: &Sweep) {
        let (lo, hi) = segment_bounds(s);
        self.lo = self.lo.min(lo);
        self.hi = self.hi.max(hi);
    }

    /// This box grown by another, without disturbing either — what the chunker asks
    /// before it commits a segment to the run in hand.
    fn union(self, other: Self) -> Self {
        Self {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    fn is_empty(&self) -> bool {
        !self.lo.x.is_finite()
    }

    /// The region rectangle's extent in texels: the tile block the coverage spans,
    /// measured between tile origins, plus the apron either side that makes each block
    /// a whole [`TILE_TEX`].
    ///
    /// Measured by bounding box rather than by enumerating tiles, and the two agree
    /// exactly: `min` over segments of `floor(lo.x / tile)` *is* `floor(min lo.x /
    /// tile)`. That identity is why [`Covered::rect`] can take its size from here while
    /// taking its halo from the set.
    fn dims(&self) -> (u32, u32) {
        let tile = TILE_SIZE as f32;
        let span = |a: f32, b: f32| ((b / tile).floor() - (a / tile).floor()) * tile;
        (
            span(self.lo.x, self.hi.x) as u32 + TILE_TEX,
            span(self.lo.y, self.hi.y) as u32 + TILE_TEX,
        )
    }
}

/// What one piece's sweeps cover: the tiles they touch and the box they span, from a
/// single walk — the write-back and the dirty set need the tiles, the region allocation
/// needs the box, and computing them apart is what lets them disagree.
pub(super) struct Covered {
    pub(super) tiles: BTreeSet<TileCoord>,
    bounds: Coverage,
}

/// Walk a piece's sweeps once, collecting the tiles they touch and the box they span.
pub(super) fn cover(segments: &[Segment], fires: &[BleedFire]) -> Covered {
    let mut tiles = BTreeSet::new();
    let bounds = for_each_touched(piece_sweeps(segments, fires), |_, c| {
        tiles.insert(c);
    });
    Covered { tiles, bounds }
}

/// The same walk, keeping **which** segments reach each tile.
///
/// What the swept path draws from, so a tile shades the segments that reach it rather
/// than the whole stroke: the total is `Σ tiles-per-segment`, the segment count times a
/// small constant, since a segment is at most a tip wide.
///
/// One `(tile, segment)` pair per reach, sorted, so a tile's segments are a
/// contiguous run — `chunk_by` on the tile is the grouping — and within a run the
/// indices ascend, which matters, since the color target's blend is `over` and
/// therefore ordered. The pairs are unique, so the whole tuple is the sort key and the
/// within-tile order is the key's rather than a sort algorithm's stability.
pub(super) fn tiles_with_segments(segments: &[Segment]) -> Vec<(TileCoord, u32)> {
    let mut pairs: Vec<(TileCoord, u32)> = Vec::new();
    // The box is `cover`'s business; this caller wants only the assignment.
    let _ = for_each_touched(segments.iter().map(|s| &s.sweep), |i, c| {
        pairs.push((c, i as u32))
    });
    pairs.sort_unstable();
    pairs
}

/// Where a sweep's centreline ends — along the arc, not along the chord.
pub(super) fn segment_end(s: &Sweep) -> Vec2 {
    crate::path::arc_at(s.start, s.dir, s.curvature, s.length).0
}

/// The canvas box one sweep's coverage occupies — the arc, grown by the tip that rides
/// along it.
///
/// The rasterized geometry reaches further than this at the caps, but every fragment
/// out there differences two prefix taps to exactly zero and writes nothing. What a box
/// has to contain is where the deposit *lands*, which is within the tip's
/// [`reach`](Sweep::reach) of the arc.
///
/// **The tip's reach, not its radius.** The two agree only for a shape that stays
/// inside the disc inscribed in its mask; a stamp that fills the corners reaches `√2`
/// times as far. Under-reporting it here clips a stroke at a tile boundary and makes
/// the dynamics loop dispatch a rect too small for its own extent.
pub(super) fn coverage_bounds(s: &Sweep) -> (Vec2, Vec2) {
    let end = segment_end(s);
    let reach = Vec2::splat(s.reach + crate::path::arc_sagitta(s.curvature, s.length));
    (s.start.min(end) - reach, s.start.max(end) + reach)
}

/// [`coverage_bounds`] grown by the apron a rewritten tile's neighbours reach into
/// (§6.4). The one place that reach is defined — [`Coverage`] is the only consumer,
/// and every rectangle in this module comes out of it.
fn segment_bounds(s: &Sweep) -> (Vec2, Vec2) {
    let (lo, hi) = coverage_bounds(s);
    let apron = Vec2::splat(TILE_APRON as f32);
    (lo - apron, hi + apron)
}

/// Split a stroke's segments into consecutive runs, each of which the stamp loop can
/// evolve inside one [`MAX_REGION_DIM`]-bounded region (§6.2).
///
/// Cutting is sound because the loop is *sequential*: running one run and then the
/// next, each over its own region with the second compositing what the first wrote
/// back, is the same computation as running them all over one region. The only state
/// threading between segments is the reservoir, which is brush-local — the same
/// argument that lets a live tail resume a frozen head ([`ToolState`](super::ToolState)).
///
/// Greedy: extend the run until one more segment would push its region past
/// [`REGION_BUDGET_DIM`], or its dispatch batch past [`MAX_STAMPS`]. A run always holds
/// at least one segment, so **a piece may exceed the budget**, and only in that one
/// way: a brush whose single segment wants more gets a piece of exactly that segment.
/// What it may never exceed is [`MAX_REGION_DIM`], the size a texture can be, which
/// `budget::fit_len` establishes before the first segment is measured here.
///
/// A segment is measured **with its own bleed firings** ([`piece_sweeps`]'s reason): a
/// window can reach back a quantum before the segment it fires after, so a piece's
/// region must hold everything the piece will write, windows included.
pub(super) fn chunk_segments(segments: &[Segment], fires: &[BleedFire]) -> Vec<Range<usize>> {
    chunk_segments_within(segments, fires, REGION_BUDGET_DIM, &[])
}

/// [`chunk_segments`] against a region edge of the caller's, with **forced cuts**:
/// every index in `cuts` (ascending) starts a new piece whatever the budget says.
///
/// The liquify path's chunker (§6.13): its region budget is the loop's less the base
/// composite's growth ([`LIQUIFY_REGION_BUDGET_DIM`](super::budget::LIQUIFY_REGION_BUDGET_DIM)),
/// and a run re-bases *at a segment*, decided by the reach walk before any piece is
/// drawn — so the re-base is a fact about the stroke's segments and not about where
/// this render happened to cut it, which is what `preview == committed` asks of it
/// (§1.3). The re-base changes the base a piece composites, so the piece starts there.
pub(super) fn chunk_segments_within(
    segments: &[Segment],
    fires: &[BleedFire],
    budget_dim: u32,
    cuts: &[usize],
) -> Vec<Range<usize>> {
    debug_assert!(cuts.is_sorted(), "forced cuts arrive in segment order");
    let mut runs = Vec::new();
    let mut run = Coverage::default();
    let mut start = 0;
    let mut pending = fires.iter().peekable();
    let mut cuts = cuts.iter().copied().peekable();
    for (i, s) in segments.iter().enumerate() {
        // This segment and whatever fires after it, as one box: they are committed to a
        // piece together or not at all.
        let mut here = Coverage::default();
        here.add(&s.sweep);
        while let Some(f) = pending.next_if(|f| f.after == i) {
            here.add(&f.window);
        }
        let forced = cuts.next_if(|c| *c <= i).is_some_and(|c| c == i);
        let grown = run.union(here);
        let (w, h) = grown.dims();
        if i > start && (forced || w > budget_dim || h > budget_dim || i - start >= MAX_STAMPS) {
            runs.push(start..i);
            (start, run) = (i, here);
        } else {
            run = grown;
        }
    }
    if start < segments.len() {
        runs.push(start..segments.len());
    }
    runs
}

/// The tiles one sweep's coverage box reaches (apron included, as [`cover`] counts
/// them) — the per-segment half of the walk, for the liquify reach walk that has to
/// account tiles segment by segment (§6.13).
pub(super) fn sweep_tiles(s: &Sweep) -> Vec<TileCoord> {
    let mut out = Vec::new();
    let _ = for_each_touched(std::iter::once(s), |_, c| out.push(c));
    out
}

impl Covered {
    /// The region the stamp loop evolves for this piece: exactly the tile block its
    /// sweeps span, grown by one apron on each side so the write-back can slice whole
    /// `TILE_TEX` blocks out of it — plus the *list* of tiles to composite into it,
    /// which is the touched set and the one-tile ring around it (§6.4).
    ///
    /// **The extent comes from [`Coverage::dims`]**, the function [`chunk_segments`]
    /// checked this piece against, so the promise "this piece fits a region" and the
    /// allocation the promise is about are one arithmetic. Only the halo comes from the
    /// tile set, and only because a diagonal stroke touches fewer tiles than its
    /// bounding rectangle holds — compositing the rectangle would be correct and slower.
    ///
    /// The ring is in the tile list but deliberately **not** in the rectangle: its job
    /// is to give a rewritten tile's apron the neighbour interior it overlaps, and an
    /// apron is [`TILE_APRON`] texels, so ring tiles outside the rectangle simply clip
    /// when composited.
    ///
    /// Returns `None` if nothing was covered.
    pub(super) fn rect(&self) -> Option<RegionRect> {
        if self.bounds.is_empty() {
            return None;
        }
        let (w, h) = self.bounds.dims();
        // Debug-only: the failure is an oversized allocation rather than a wrong
        // picture, and a panic in the render path is its own defect (see
        // `plan::dispatch_rect`). Against the **ceiling**, not the chunker's budget — a
        // piece holding a single oversized segment is allowed to exceed the budget by
        // design (`chunk_segments`); what is never allowed is a region past the size a
        // texture can be, which `budget::fit_len` guarantees upstream.
        debug_assert!(
            w <= MAX_REGION_DIM && h <= MAX_REGION_DIM,
            "a {w}x{h} region overruns the {MAX_REGION_DIM} a texture can be",
        );
        // The top-left *tile* origin the box spans. Taken off the tile set rather than
        // re-floored off the box, which is the same point: `dims` is a span between
        // exactly these origins.
        let mut lo = Vec2::splat(f32::INFINITY);
        // Sorted and deduped rather than inserted into a `BTreeSet` nine times per
        // touched tile: the halo is consumed as a flat list, so the ordered set was
        // paying for a tree it never queried. Same order out, one allocation.
        let mut halo: Vec<TileCoord> = Vec::with_capacity(self.tiles.len() * 9);
        for c in &self.tiles {
            lo = lo.min(c.origin());
            for dy in -1..=1 {
                for dx in -1..=1 {
                    halo.push(TileCoord::new(c.x + dx, c.y + dy));
                }
            }
        }
        halo.sort_unstable();
        halo.dedup();
        Some(RegionRect {
            halo,
            lo,
            origin: lo - Vec2::splat(TILE_APRON as f32),
            w,
            h,
        })
    }
}

/// What [`Covered::rect`] measures for a piece: the region rectangle the stamp loop
/// evolves, and the tiles composited into it.
pub(super) struct RegionRect {
    /// The tiles to composite: the affected set plus the one-tile ring around it, so
    /// rewritten tiles' aprons read real neighbour content (§6.4).
    pub(super) halo: Vec<TileCoord>,
    /// The top-left affected tile's origin — the region's *interior* origin, which
    /// the write-back measures each tile's offset against.
    pub(super) lo: Vec2,
    /// The region rectangle's top-left in canvas px: [`lo`](Self::lo) less one
    /// apron — what every slot's coordinates are measured from.
    ///
    /// **Whole texels, by construction.** [`lo`](Self::lo) is a `min` over
    /// [`TileCoord::origin`] values, which are integral multiples of `TILE_SIZE`, and
    /// the apron subtracted from it is an integer. The cell grid's anchor rests on that
    /// — `plan::cell_geometry` takes `origin.rem_euclid(cell)` to place the grid, so a
    /// fractional origin would put every cell boundary off the canvas texel grid, which
    /// is a seam (§6.4).
    pub(super) origin: Vec2,
    /// The rectangle's extent in texels.
    pub(super) w: u32,
    pub(super) h: u32,
}

#[cfg(test)]
mod tests {
    use super::super::dynamics::BLEED_TRAVEL_QUANTUM;
    // The endpoint shape of the shared fixtures (`segments::testing`): these cases are
    // about how the measurements combine boxes, so a sweep is named by where the tip
    // went, and every other axis is held at the fixture's neutral.
    use super::super::segments::testing::{seg_between, sweep_between};
    use super::*;

    // --- region measurement ----------------------------------------------

    /// A firing after segment `after`, sweeping `window`.
    fn fire(after: usize, window: Sweep) -> BleedFire {
        BleedFire {
            after,
            window,
            bleed: 0.5,
        }
    }

    /// The region extent as [`Coverage`] measures it: a bounding box over every sweep
    /// the piece will rasterize.
    fn measured(segments: &[Segment], fires: &[BleedFire]) -> Option<(u32, u32)> {
        let mut c = Coverage::default();
        for s in piece_sweeps(segments, fires) {
            c.add(s);
        }
        (!c.is_empty()).then(|| c.dims())
    }

    /// The same extent reached the **other** way: the tile block the touched set spans,
    /// measured between its extreme tile origins.
    fn from_tiles(tiles: &BTreeSet<TileCoord>) -> Option<(u32, u32)> {
        let mut lo = Vec2::splat(f32::INFINITY);
        let mut hi = Vec2::splat(f32::NEG_INFINITY);
        for c in tiles {
            lo = lo.min(c.origin());
            hi = hi.max(c.origin());
        }
        lo.x.is_finite().then(|| {
            (
                (hi.x - lo.x) as u32 + TILE_TEX,
                (hi.y - lo.y) as u32 + TILE_TEX,
            )
        })
    }

    /// The per-tile segment lists cover exactly the tiles [`cover`] names, and a tile's
    /// list holds exactly the segments whose bounds reach it — in stroke order, which
    /// the `over` blend on the color target makes load-bearing. An omission here is
    /// missing paint and a re-ordering is a different picture.
    #[test]
    fn the_per_tile_lists_hold_exactly_the_segments_that_reach_each_tile() {
        let tile = TILE_SIZE as f32;
        let segments: Vec<Segment> = (0..40)
            .map(|i| {
                let t = i as f32;
                seg_between(
                    Vec2::new(t * 31.0 - 200.0, (t * 0.4).sin() * 300.0),
                    Vec2::new((t + 1.0) * 31.0 - 200.0, ((t + 1.0) * 0.4).sin() * 300.0),
                    4.0 + (i % 5) as f32 * 9.0,
                )
            })
            .collect();

        let pairs = tiles_with_segments(&segments);
        let runs: Vec<&[(TileCoord, u32)]> = pairs.chunk_by(|a, b| a.0 == b.0).collect();
        assert_eq!(
            runs.iter().map(|run| run[0].0).collect::<BTreeSet<_>>(),
            cover(&segments, &[]).tiles,
            "the two walks disagree on which tiles a stroke touches",
        );
        assert_eq!(
            runs.len(),
            cover(&segments, &[]).tiles.len(),
            "a tile's pairs are not one contiguous run",
        );
        assert!(runs.len() > 4, "not enough tiles to be an interesting case");

        for run in &runs {
            let coord = run[0].0;
            assert!(
                run.windows(2).all(|w| w[0].1 < w[1].1),
                "tile {coord:?}'s segments are not in stroke order",
            );
            // The list against the membership test itself, segment by segment: a tile
            // is in a segment's block exactly when the segment is in the tile's list.
            for (i, s) in segments.iter().enumerate() {
                let (lo, hi) = segment_bounds(&s.sweep);
                let inside = (lo.x / tile).floor() <= coord.x as f32
                    && coord.x as f32 <= (hi.x / tile).floor()
                    && (lo.y / tile).floor() <= coord.y as f32
                    && coord.y as f32 <= (hi.y / tile).floor();
                assert_eq!(
                    run.iter().any(|&(_, j)| j == i as u32),
                    inside,
                    "tile {coord:?} and segment {i} disagree about reaching one another",
                );
            }
        }

        // And the whole point: the listed pairs are far fewer than the full
        // tile × segment product the swept path shades.
        let listed = pairs.len();
        assert!(
            listed < runs.len() * segments.len() / 4,
            "{listed} listed pairs against a {} product — the grouping is not buying \
             anything on this case",
            runs.len() * segments.len(),
        );
    }

    /// **The identity [`Covered::rect`] rests on**: the extent of the bounding box a
    /// piece covers is the extent of the tile block its touched set spans.
    ///
    /// `rect` takes its **size** from the box and its **halo** from the set, which is
    /// only sound because `min` over segments of `floor(lo / tile)` is
    /// `floor(min lo / tile)`. Checked on shapes where a disagreement would show — a
    /// fat tip reaching past its own endpoints, negative tiles, extremes contributed by
    /// different segments. Under-reporting allocates past [`MAX_REGION_DIM`];
    /// over-reporting cuts strokes into more pieces than they need.
    #[test]
    fn the_box_and_the_tile_set_measure_the_same_rectangle() {
        let tile = TILE_SIZE as f32;
        let cases: Vec<(&str, Vec<Segment>)> = vec![
            (
                "a dot",
                vec![seg_between(
                    Vec2::new(10.0, 10.0),
                    Vec2::new(10.5, 10.0),
                    4.0,
                )],
            ),
            (
                "one tile-aligned span",
                vec![seg_between(Vec2::ZERO, Vec2::new(tile, 0.0), 1.0)],
            ),
            (
                "across the origin, into negative tiles",
                vec![seg_between(
                    Vec2::new(-300.0, -140.0),
                    Vec2::new(220.0, 90.0),
                    12.0,
                )],
            ),
            (
                "a fat tip, whose radius reaches past its endpoints",
                vec![seg_between(
                    Vec2::new(500.0, 500.0),
                    Vec2::new(505.0, 500.0),
                    90.0,
                )],
            ),
            (
                "several segments, extremes in different ones",
                vec![
                    seg_between(Vec2::new(0.0, 0.0), Vec2::new(120.0, 30.0), 3.0),
                    seg_between(Vec2::new(120.0, 30.0), Vec2::new(-90.0, 400.0), 20.0),
                    seg_between(Vec2::new(-90.0, 400.0), Vec2::new(700.0, -60.0), 8.0),
                ],
            ),
        ];
        for (what, segments) in cases {
            let covered = cover(&segments, &[]);
            let want = from_tiles(&covered.tiles);
            assert_eq!(
                measured(&segments, &[]),
                want,
                "the box and the tile set disagree for {what}"
            );
            // And that is the size the region is actually allocated at.
            assert_eq!(
                covered.rect().map(|r| (r.w, r.h)),
                want,
                "the region built for {what} is not the rectangle measured"
            );
        }
        assert!(
            cover(&[], &[]).rect().is_none(),
            "no segments is not a region"
        );
        assert_eq!(measured(&[], &[]), None, "no segments is not a region");
    }

    /// **The accounting covers a firing window's reach back past the piece.** A window
    /// is walked back along its crossing segment's own arc and can start up to a
    /// [`BLEED_TRAVEL_QUANTUM`] before the piece's first segment (`plan::bleed_fires`),
    /// while the margin the segment boxes leave is one apron texel. Both halves must
    /// take the windows: the tile walk (a tile it misses is flux silently clipped and an
    /// apron/interior seam) and the chunker (a piece's region must hold everything the
    /// piece writes).
    #[test]
    fn a_windows_reach_back_is_in_the_tiles_and_the_region() {
        let tile = TILE_SIZE as f32;
        let radius = 40.0;
        let bq = BLEED_TRAVEL_QUANTUM * radius;
        // The piece's first segment, placed so its own coverage box starts 3 px
        // past a tile origin — inside the window's reach, outside the apron's.
        let x0 = 2.0 * tile + radius + TILE_APRON as f32 + 3.0;
        let s = seg_between(Vec2::new(x0, 8.0), Vec2::new(x0 + 50.0, 8.0), radius);
        // Its firing's window, one quantum of arc ending where the segment starts —
        // the shape `bleed_fires` emits for the first segment of a range.
        let w = sweep_between(Vec2::new(x0 - bq, 8.0), Vec2::new(x0, 8.0), radius);
        let fires = vec![fire(0, w)];

        let without = cover(&[s], &[]).tiles;
        let with = cover(&[s], &fires).tiles;
        let window_tiles = cover(&[], &fires).tiles;
        assert!(
            window_tiles.iter().any(|c| !without.contains(c)),
            "the window does not reach past the segment boxes — the case has gone \
             soft and pins nothing",
        );
        assert!(
            window_tiles.iter().all(|c| with.contains(c)),
            "a tile the window writes is missing from the walk",
        );
        // And the chunker measures the very region the render then builds from the
        // tiles — fires on both sides of the relation, like the segments always were.
        assert_eq!(
            measured(&[s], &fires),
            from_tiles(&with),
            "the box and the tile set disagree once the windows are counted",
        );
        assert_eq!(
            chunk_segments(&[s], &fires),
            vec![0..1],
            "one segment and its firing are one piece",
        );
    }

    /// A forced cut starts a piece exactly there, whatever the budget says, and an
    /// index that is already a piece start — or past the end — asks for nothing
    /// (§6.13's re-base rides this).
    #[test]
    fn a_forced_cut_starts_a_piece_where_it_says() {
        let segments: Vec<Segment> = (0..10)
            .map(|i| {
                let t = i as f32 * 8.0;
                seg_between(Vec2::new(t, 0.0), Vec2::new(t + 8.0, 0.0), 4.0)
            })
            .collect();
        assert_eq!(
            chunk_segments(&segments, &[]),
            vec![0..10],
            "fits one piece"
        );
        assert_eq!(
            chunk_segments_within(&segments, &[], REGION_BUDGET_DIM, &[3, 7]),
            vec![0..3, 3..7, 7..10],
        );
        assert_eq!(
            chunk_segments_within(&segments, &[], REGION_BUDGET_DIM, &[0, 3, 3, 20]),
            vec![0..3, 3..10],
            "a cut at the start, a repeat and one past the end are no cuts",
        );
    }

    /// What [`chunk_segments`] promises the loop: the pieces tile the stroke in order
    /// (so the sequence of segments the loop walks is unchanged — the reason cutting it
    /// is sound), and every piece fits the region bound the cut exists to respect.
    ///
    /// Measured against [`REGION_BUDGET_DIM`] rather than the ceiling: this tip is far
    /// inside what a single segment may demand, and against the ceiling the case would
    /// pass with the cut removed entirely.
    #[test]
    fn the_chunks_tile_the_stroke_and_each_one_fits() {
        // A stroke far longer than one region in both axes, and a fat tip whose own
        // extent eats a good part of the budget.
        let segments: Vec<Segment> = (0..600)
            .map(|i| {
                let t = i as f32;
                let a = Vec2::new(t * 9.0 - 400.0, (t * 0.05).sin() * 1500.0);
                let b = Vec2::new((t + 1.0) * 9.0 - 400.0, ((t + 1.0) * 0.05).sin() * 1500.0);
                seg_between(a, b, 60.0)
            })
            .collect();
        let runs = chunk_segments(&segments, &[]);
        assert!(runs.len() > 1, "an oversized stroke should be cut up");

        let mut next = 0;
        for run in &runs {
            assert_eq!(run.start, next, "the pieces leave a gap or overlap");
            next = run.end;
            let (w, h) = measured(&segments[run.clone()], &[]).expect("a piece is never empty");
            assert!(
                w <= REGION_BUDGET_DIM && h <= REGION_BUDGET_DIM,
                "piece {run:?} needs a {w}x{h} region",
            );
            assert!(run.len() <= MAX_STAMPS, "piece {run:?} overruns the batch");
        }
        assert_eq!(next, segments.len(), "the pieces do not cover the stroke");
    }
}
