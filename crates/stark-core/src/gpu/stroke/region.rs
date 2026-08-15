//! The region a stroke piece needs: which tiles its sweeps touch, the box they span,
//! and where the stamp loop cuts one stroke into pieces (§6.2, §6.4).
//!
//! Split out of [`segments`](super::segments) because this is where one invariant
//! lives and it deserves to be local: **the rectangle the chunker measures a piece
//! against is the rectangle the render then allocates.** Those used to be three
//! derivations — a tile set, a bounding box turned into region dimensions, and the set
//! turned back into a rectangle — with a comment asking them to agree. They are one
//! [`Coverage`] now, and [`Covered::rect`] takes its extent from the very
//! [`dims`](Coverage::dims) the chunker checked.
//!
//! Nothing here touches the GPU. It is float arithmetic over [`Sweep`]s and tile
//! coordinates, which is what lets the whole of it be pinned without an adapter
//! (`tests`).

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use crate::document::BrushParams;
use crate::geom::{TILE_APRON, TILE_SIZE, TILE_TEX, TileCoord, Vec2};

use super::budget::{MAX_REGION_DIM, MAX_STAMPS};
use super::dynamics::BLEED_TRAVEL_QUANTUM;
use super::segments::{BleedFire, DAB_TRAVEL, Segment, Sweep, tip_reach};

/// Call `f(segment index, tile)` for every tile whose *texture* (interior + apron) a
/// segment's swept capsule overlaps, in segment order.
///
/// The apron is included in the reach so a stroke landing within a tile's interior
/// but inside a neighbour's apron band re-renders that neighbour too, keeping the
/// shared apron/interior overlap bit-identical (§6.4).
///
/// **A segment writes exactly zero outside the tiles this names**, which is what lets
/// [`tiles_with_segments`] hand each tile a subset rather than the whole stroke. The
/// rasterized geometry does reach further — the shaders sweep a generous angular
/// margin so a round cap is never clipped — but out there a fragment differences two
/// prefix-τ taps that are equal and writes nothing at all (see [`coverage_bounds`]).
/// Zero through the `over` blend and zero through the additive one are both exact
/// identities, so which segments a tile is handed cannot change what lands in it.
fn for_each_touched<'a>(
    sweeps: impl Iterator<Item = &'a Sweep>,
    mut f: impl FnMut(usize, TileCoord),
) {
    let tile = TILE_SIZE as f32;
    for (i, s) in sweeps.enumerate() {
        let (lo, hi) = segment_bounds(s);
        let (x0, x1) = ((lo.x / tile).floor() as i32, (hi.x / tile).floor() as i32);
        let (y0, y1) = ((lo.y / tile).floor() as i32, (hi.y / tile).floor() as i32);
        for y in y0..=y1 {
            for x in x0..=x1 {
                f(i, TileCoord::new(x, y));
            }
        }
    }
}

/// Every sweep a piece will rasterize: its painting segments **and its bleed firings'
/// windows**, in the order the plan dispatches them.
///
/// The windows belong in every accounting because they write: a firing's sweep is
/// walked back along the crossing segment's own arc, up to one
/// [`BLEED_TRAVEL_QUANTUM`] before the segment it fires after (`plan::bleed_fires`) —
/// and for the first segment of a piece or a live-tail range that stretch lies behind
/// every segment box, with one apron texel of margin. Left out (as they were until
/// 2026-08-11, while the snapshot scratch's sizing *did* take them), the flux written
/// there was silently clipped by the region's bounds check, and a rewritten tile's
/// apron could diverge from an unrewritten neighbour's interior — a §6.4 break in
/// exactly the configuration `tests/seam.rs` does not draw.
///
/// One function, so no caller can enumerate one and forget the other. That is the
/// mistake above, stated as a thing the code no longer lets you make.
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
/// **The one definition of what a piece needs**, and the reason it is a type. Three
/// answers used to be derived three ways and were required to agree: `chunk_segments`
/// measured a bounding box and turned it into region dimensions, `affected_tiles`
/// enumerated a tile set, and `region_rect` turned that set back into a rectangle. The
/// chunker's promise — "this piece fits [`MAX_REGION_DIM`]" — was about the first, and
/// the region the loop actually allocates came from the third; a comment asked them to
/// be the same rectangle and a test checked that they were.
///
/// They are now the same rectangle because they are the same arithmetic:
/// [`Covered::rect`] takes its extent from [`dims`](Self::dims), which is the very
/// function the chunker checked. The tile set is still enumerated separately, because
/// only the dynamics path wants it and a set insert per tile per segment is exactly the
/// cost the incremental repaint exists to keep off a long stroke.
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
    /// tile)`, so the extreme tile origins of the set are the extreme tile origins of
    /// the box. That identity is why [`Covered::rect`] can take its size from here
    /// while taking its halo from the set.
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
/// single walk.
///
/// Both in one pass because both callers want both — the write-back and the dirty set
/// need the tiles, the region allocation needs the box — and because computing them
/// apart is what let them disagree.
pub(super) struct Covered {
    pub(super) tiles: BTreeSet<TileCoord>,
    bounds: Coverage,
}

/// Walk a piece's sweeps once, collecting the tiles they touch and the box they span.
pub(super) fn cover(segments: &[Segment], fires: &[BleedFire]) -> Covered {
    let mut tiles = BTreeSet::new();
    let mut bounds = Coverage::default();
    let sweeps = piece_sweeps(segments, fires);
    for_each_touched(sweeps.clone(), |_, c| {
        tiles.insert(c);
    });
    for s in sweeps {
        bounds.add(s);
    }
    Covered { tiles, bounds }
}

/// The same walk, keeping **which** segments reach each tile.
///
/// This is what the swept path draws from. Drawing every segment into every tile made
/// a stroke cost `segments × tiles` vertex invocations, nearly all of them on quads
/// that fall outside the tile being rendered and are discarded after being shaded —
/// and a tapered brush spends ~211 segments on a straight line, so a long stroke
/// crossing a document's worth of tiles paid for the product of two large numbers. Per
/// tile the cost is now the segments that actually reach it, and over a stroke the
/// total is `Σ tiles-per-segment`: the segment count times a small constant, since a
/// segment is at most a tip wide.
///
/// The indices come out ascending, because the walk is in segment order — which
/// matters, since the color target's blend is `over` and therefore ordered. Each tile
/// sees the stroke's own order over the subset that reaches it.
pub(super) fn tiles_with_segments(segments: &[Segment]) -> BTreeMap<TileCoord, Vec<u32>> {
    let mut map: BTreeMap<TileCoord, Vec<u32>> = BTreeMap::new();
    for_each_touched(segments.iter().map(|s| &s.sweep), |i, c| {
        map.entry(c).or_default().push(i as u32)
    });
    map
}

/// Where a sweep's centreline ends — along the arc, not along the chord.
pub(super) fn segment_end(s: &Sweep) -> Vec2 {
    crate::path::arc_at(s.start, s.dir, s.curvature, s.length).0
}

/// The canvas box one sweep's coverage occupies — the arc, grown by the tip that rides
/// along it.
///
/// The rasterized geometry reaches further than this at the caps (the shaders sweep a
/// generous angular margin so the round end is never clipped), but every fragment out
/// there differences two prefix taps to exactly zero and writes nothing. What a box
/// has to contain is where the deposit *lands*, which is within the tip's
/// [`reach`](Sweep::reach) of the arc.
///
/// **The tip's reach, not its radius.** The two are the same number only for a shape
/// that stays inside the disc inscribed in its mask; a stamp that fills the corners
/// reaches `√2` times as far, and swept along a diagonal that difference is a whole
/// corner of the footprint. Under-reporting it here is a stroke clipped at a tile
/// boundary — `for_each_touched` leaves the tile out of the render (or leaves this
/// segment out of a tile another segment brought in), and the dynamics loop dispatches
/// a rect too small for its own footprint.
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
/// The loop works on a 1:1 copy of the canvas under the stroke, so a stroke that
/// crosses the document would want a region the size of the document. It does not
/// have to have one: the loop is *sequential*, so running the first run of segments
/// and then the second — each over its own region, the second compositing what the
/// first wrote back — is the same computation as running them all over one region.
/// The same segments in the same order, and the state that threads between them is
/// the reservoir, which is brush-local and says nothing about where the stroke is.
/// That is the identical argument that lets a live tail resume a frozen head
/// ([`ToolState`](super::ToolState)); a piece is just a cut the renderer makes for
/// itself rather than one the fitter made for it.
///
/// Greedy: extend the run until one more segment would push its region past
/// [`MAX_REGION_DIM`], or its dispatch batch past [`MAX_STAMPS`]. A run always holds
/// at least one segment — one tip's own footprint is the floor no subdivision gets
/// under, which is what [`segment_fits_region`] gates on instead.
///
/// A segment is measured **with its own bleed firings** ([`piece_sweeps`]'s reason): a
/// window can reach back a quantum before the segment it fires after, so a piece's
/// region must hold everything the piece will write, windows included — the same
/// rectangle [`Covered::rect`] then builds, through the same [`Coverage::dims`] this
/// checks against.
pub(super) fn chunk_segments(segments: &[Segment], fires: &[BleedFire]) -> Vec<Range<usize>> {
    let mut runs = Vec::new();
    let mut run = Coverage::default();
    let mut start = 0;
    let mut pending = fires.iter().peekable();
    for (i, s) in segments.iter().enumerate() {
        // This segment and whatever fires after it, as one box: they are committed to a
        // piece together or not at all.
        let mut here = Coverage::default();
        here.add(&s.sweep);
        while let Some(f) = pending.next_if(|f| f.after == i) {
            here.add(&f.window);
        }
        let grown = run.union(here);
        let (w, h) = grown.dims();
        if i > start && (w > MAX_REGION_DIM || h > MAX_REGION_DIM || i - start >= MAX_STAMPS) {
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

/// Whether one segment of `b`'s swept footprint fits a region.
///
/// [`chunk_segments`] can cut a stroke as fine as a single segment, but no finer: the
/// reservoir pickup reduces over the whole tip at once, so the region can never be
/// smaller than one footprint. A brush too fat for that is the one thing left that
/// sends a dynamics stroke to the plain swept deposit — and, unlike the whole-stroke
/// measurement this replaced, it is decided from the brush alone, so it costs nothing
/// to re-ask on every pointer move and cannot answer differently for a piece than for
/// the stroke it belongs to.
///
/// Bounded rather than measured, since it has to hold for segments that do not exist
/// yet: radius peaks at the brush's own (pressure only scales it down), travel at the
/// flattening cap — or at the [`DAB_TRAVEL`] radii a touch-down dab sweeps, which
/// ignores the cap — and a coverage box of a given extent spans at most one tile more
/// than it covers, whichever tile boundary it happens to straddle.
pub(super) fn segment_fits_region(b: &BrushParams, tol: crate::path::FlattenTolerance) -> bool {
    let radius = b.radius.max(0.5);
    // The chord is what `path::within` caps; the arc over it is longer, and bows a
    // sagitta out of its own box. Both are bounded by the turn a segment may bend
    // through (`MAX_HALF_TURN_SIN`) — under 2% and under 5% of the chord — so a
    // single margin covers the pair with room to spare.
    //
    // A bleeding brush's segment also carries its firings, whose windows reach up
    // to one quantum back past its start ([`chunk_segments`]) — so the floor the
    // chunker cannot get under is that much longer for it.
    let bleed = if b.dynamics.bleed > 0.0 {
        BLEED_TRAVEL_QUANTUM * radius
    } else {
        0.0
    };
    let length = tol.max_len.max(DAB_TRAVEL * radius) * 1.1 + bleed;
    // The tip's reach rather than its radius, for [`coverage_bounds`]' reason: a
    // stamp that fills its mask's corners occupies a `√2`-wider box, and this is the
    // bound that decides whether the loop may draw the brush at all.
    let extent = length + 2.0 * (radius * tip_reach(&b.shape) + TILE_APRON as f32);
    let worst = (extent / TILE_SIZE as f32).ceil().max(0.0) as u32 * TILE_SIZE + TILE_TEX;
    worst <= MAX_REGION_DIM
}

impl Covered {
    /// The region the stamp loop evolves for this piece: exactly the tile block its
    /// sweeps span, grown by one apron on each side so the write-back can slice whole
    /// `TILE_TEX` blocks out of it — plus the *list* of tiles to composite into it,
    /// which is the touched set and the one-tile ring around it (§6.4).
    ///
    /// **The extent comes from [`Coverage::dims`]**, which is the function
    /// [`chunk_segments`] checked this piece against. That is the whole point of the
    /// type: the promise "this piece fits `MAX_REGION_DIM`" and the allocation the
    /// promise is about are now one arithmetic rather than two that a comment asked to
    /// agree. Only the halo comes from the tile set, and only because a diagonal stroke
    /// touches fewer tiles than its bounding rectangle holds — compositing the
    /// rectangle would be correct and slower.
    ///
    /// The ring is in the tile list but deliberately **not** in the rectangle. Its
    /// whole job is to give a rewritten tile's apron the neighbour interior it
    /// overlaps, and an apron is [`TILE_APRON`] texels — so extending the rectangle by
    /// a whole *tile* on every side, as it once did, paid for roughly 4× the region to
    /// fill a one-texel band. Ring tiles that fall outside the rectangle simply clip
    /// when composited. On a live tail, which covers a handful of tiles and is redrawn
    /// on every pointer move, that difference is most of the cost of the whole path.
    ///
    /// Returns `None` if nothing was covered.
    pub(super) fn rect(&self) -> Option<RegionRect> {
        if self.bounds.is_empty() {
            return None;
        }
        let (w, h) = self.bounds.dims();
        // Stated where it is relied on, which is what it was missing: `chunk_segments`
        // hands over pieces that fit by construction, and until this line nothing said
        // so at the point the region is actually allocated. Debug-only because the
        // failure is an oversized allocation rather than a wrong picture, and because a
        // panic in the render path is its own defect (see `plan::dispatch_rect`).
        debug_assert!(
            w <= MAX_REGION_DIM && h <= MAX_REGION_DIM,
            "a {w}x{h} region overruns the {MAX_REGION_DIM} the chunker promised",
        );
        // The top-left *tile* origin the box spans. Taken off the tile set rather than
        // re-floored off the box, which is the same point: `dims` is a span between
        // exactly these origins.
        let mut lo = Vec2::splat(f32::INFINITY);
        let mut halo: BTreeSet<TileCoord> = BTreeSet::new();
        for c in &self.tiles {
            lo = lo.min(c.origin());
            for dy in -1..=1 {
                for dx in -1..=1 {
                    halo.insert(TileCoord::new(c.x + dx, c.y + dy));
                }
            }
        }
        Some(RegionRect {
            halo: halo.into_iter().collect(),
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
    pub(super) origin: Vec2,
    /// The rectangle's extent in texels.
    pub(super) w: u32,
    pub(super) h: u32,
}

#[cfg(test)]
mod tests {
    use super::super::budget::flatten_tolerance;
    use super::super::segments::Paint;
    use super::*;

    // --- region measurement ----------------------------------------------

    /// A sweep carrying only what the region measurements read.
    ///
    /// The paint rates are absent rather than zeroed: the region measurements are
    /// geometry, they take a [`Sweep`], and there is no longer anywhere to put a rate
    /// that would imply one had been consulted.
    fn sweep(start: Vec2, end: Vec2, radius: f32) -> Sweep {
        let v = end - start;
        let length = v.length();
        Sweep {
            start,
            dir: if length > 0.0 {
                v / length
            } else {
                Vec2::new(1.0, 0.0)
            },
            curvature: 0.0,
            radius,
            // A tip that holds still: these cases are about how the measurements
            // combine boxes, and a ramp would put a second variable in every box.
            ramp: 0.0,
            // A round tip's frame and reach, both the radius: these cases are about how
            // the measurements combine boxes, not about how wide one shape is.
            frame: radius,
            reach: radius,
            length,
            orient: 0.0,
            dist: 0.0,
        }
    }

    /// The same, as a whole segment — what the chunker and the tile walks take.
    fn seg(start: Vec2, end: Vec2, radius: f32) -> Segment {
        Segment {
            sweep: sweep(start, end, radius),
            paint: Paint::default(),
        }
    }

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

    /// The per-tile segment lists cover exactly the tiles [`affected_tiles`] names, and
    /// a tile's list holds exactly the segments whose bounds reach it — in stroke
    /// order, which the `over` blend on the color target makes load-bearing.
    ///
    /// The swept path draws from these lists instead of drawing every segment into
    /// every tile, so an omission here is missing paint and a re-ordering is a
    /// different picture. Both are the kind of thing a golden would show as "the stroke
    /// looks a bit wrong" without saying why.
    #[test]
    fn the_per_tile_lists_hold_exactly_the_segments_that_reach_each_tile() {
        let tile = TILE_SIZE as f32;
        let segments: Vec<Segment> = (0..40)
            .map(|i| {
                let t = i as f32;
                seg(
                    Vec2::new(t * 31.0 - 200.0, (t * 0.4).sin() * 300.0),
                    Vec2::new((t + 1.0) * 31.0 - 200.0, ((t + 1.0) * 0.4).sin() * 300.0),
                    4.0 + (i % 5) as f32 * 9.0,
                )
            })
            .collect();

        let map = tiles_with_segments(&segments);
        assert_eq!(
            map.keys().copied().collect::<BTreeSet<_>>(),
            cover(&segments, &[]).tiles,
            "the two walks disagree on which tiles a stroke touches",
        );
        assert!(map.len() > 4, "not enough tiles to be an interesting case");

        for (coord, idx) in &map {
            assert!(
                idx.windows(2).all(|w| w[0] < w[1]),
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
                    idx.contains(&(i as u32)),
                    inside,
                    "tile {coord:?} and segment {i} disagree about reaching one another",
                );
            }
        }

        // And the whole point: the listed pairs are far fewer than the product the
        // swept path used to shade.
        let listed: usize = map.values().map(Vec::len).sum();
        assert!(
            listed < map.len() * segments.len() / 4,
            "{listed} listed pairs against a {} product — the grouping is not buying \
             anything on this case",
            map.len() * segments.len(),
        );
    }

    /// **The identity [`Covered::rect`] rests on**: the extent of the bounding box a
    /// piece covers is the extent of the tile block its touched set spans.
    ///
    /// The chunker decides where to cut by measuring a box ([`Coverage::dims`]) and the
    /// render allocates the region the cut was about; those are one function now, so
    /// that half is structural rather than tested. What is left is why it is allowed to
    /// be: `rect` takes its **size** from the box and its **halo** from the set, and
    /// that is only sound because `min` over segments of `floor(lo / tile)` is
    /// `floor(min lo / tile)`. This checks the two against each other on shapes where a
    /// disagreement would show — a fat tip reaching past its own endpoints, negative
    /// tiles, extremes contributed by different segments.
    ///
    /// If the box ever under-reported, a piece would allocate past [`MAX_REGION_DIM`];
    /// if it over-reported, strokes would be cut into more pieces than they need, each
    /// paying for its own region composite.
    #[test]
    fn the_box_and_the_tile_set_measure_the_same_rectangle() {
        let tile = TILE_SIZE as f32;
        let cases: Vec<(&str, Vec<Segment>)> = vec![
            (
                "a dot",
                vec![seg(Vec2::new(10.0, 10.0), Vec2::new(10.5, 10.0), 4.0)],
            ),
            (
                "one tile-aligned span",
                vec![seg(Vec2::ZERO, Vec2::new(tile, 0.0), 1.0)],
            ),
            (
                "across the origin, into negative tiles",
                vec![seg(Vec2::new(-300.0, -140.0), Vec2::new(220.0, 90.0), 12.0)],
            ),
            (
                "a fat tip, whose radius reaches past its endpoints",
                vec![seg(Vec2::new(500.0, 500.0), Vec2::new(505.0, 500.0), 90.0)],
            ),
            (
                "several segments, extremes in different ones",
                vec![
                    seg(Vec2::new(0.0, 0.0), Vec2::new(120.0, 30.0), 3.0),
                    seg(Vec2::new(120.0, 30.0), Vec2::new(-90.0, 400.0), 20.0),
                    seg(Vec2::new(-90.0, 400.0), Vec2::new(700.0, -60.0), 8.0),
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

    /// **The accounting covers a firing window's reach back past the piece** — the
    /// 2026-08-11 regression, pinned where it is exact. A window is walked back
    /// along its crossing segment's own arc and can start up to a
    /// [`BLEED_TRAVEL_QUANTUM`] before the piece's first segment
    /// (`plan::bleed_fires`); the margin the segment boxes leave is one apron
    /// texel, so a bleeding tip wider than a few px reaches ground no segment box
    /// names whenever its box falls within a quantum of a tile origin. Both halves
    /// must take the windows: the tile walk (the region rectangle and the
    /// write-back follow it — a tile it misses is flux silently clipped and an
    /// apron/interior seam), and the chunker (a piece's region must hold everything
    /// the piece writes).
    #[test]
    fn a_windows_reach_back_is_in_the_tiles_and_the_region() {
        let tile = TILE_SIZE as f32;
        let radius = 40.0;
        let bq = BLEED_TRAVEL_QUANTUM * radius;
        // The piece's first segment, placed so its own coverage box starts 3 px
        // past a tile origin — inside the window's reach, outside the apron's.
        let x0 = 2.0 * tile + radius + TILE_APRON as f32 + 3.0;
        let s = seg(Vec2::new(x0, 8.0), Vec2::new(x0 + 50.0, 8.0), radius);
        // Its firing's window, one quantum of arc ending where the segment starts —
        // the shape `bleed_fires` emits for the first segment of a range.
        let w = sweep(Vec2::new(x0 - bq, 8.0), Vec2::new(x0, 8.0), radius);
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

    /// What [`chunk_segments`] promises the loop: the pieces tile the stroke in order
    /// (so the sequence of segments the loop walks is unchanged — the whole reason
    /// cutting it is sound), and every piece actually fits the region bound the cut
    /// exists to respect.
    #[test]
    fn the_chunks_tile_the_stroke_and_each_one_fits() {
        // A stroke far longer than one region in both axes, and a fat tip whose own
        // footprint eats a good part of the budget.
        let segments: Vec<Segment> = (0..600)
            .map(|i| {
                let t = i as f32;
                let a = Vec2::new(t * 9.0 - 400.0, (t * 0.05).sin() * 1500.0);
                let b = Vec2::new((t + 1.0) * 9.0 - 400.0, ((t + 1.0) * 0.05).sin() * 1500.0);
                seg(a, b, 60.0)
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
                w <= MAX_REGION_DIM && h <= MAX_REGION_DIM,
                "piece {run:?} needs a {w}x{h} region",
            );
            assert!(run.len() <= MAX_STAMPS, "piece {run:?} overruns the batch");
        }
        assert_eq!(next, segments.len(), "the pieces do not cover the stroke");
    }

    /// The floor the chunker cannot get under: one segment's own footprint. A brush
    /// whose tip fits is drawn by the loop however long the stroke gets, which is the
    /// whole point of cutting it into pieces; one whose tip does not is the only case
    /// left that degrades to the swept deposit.
    #[test]
    fn the_gate_admits_any_brush_whose_own_tip_fits() {
        let fits = |radius: f32| {
            let mut b = BrushParams {
                radius,
                ..BrushParams::default()
            };
            b.dynamics.lift = 0.5;
            segment_fits_region(&b, flatten_tolerance(&b))
        };
        assert!(fits(1.0), "a hairline tip fits");
        assert!(fits(120.0), "the largest tip the UI offers fits");
        assert!(
            !fits(MAX_REGION_DIM as f32),
            "a tip wider than the whole region cannot fit"
        );
    }
}
