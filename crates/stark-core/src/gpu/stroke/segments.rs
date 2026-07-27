//! Turning a fitted path into swept segments, and the region measurements that decide
//! where the stamp loop cuts one into pieces (DESIGN.md §6.2).
//!
//! Both render paths flatten through here, so both see the same segments for the
//! same record — which is what lets a live tail and the commit that replaces it
//! agree pixel for pixel.

use std::collections::BTreeSet;
use std::ops::Range;

use bytemuck::{Pod, Zeroable};

use crate::document::{BrushParams, OrientationSource, StrokeRecord};
use crate::geom::{TILE_APRON, TILE_SIZE, TILE_TEX, TileCoord, Vec2};
use crate::noise::NOISE_TILE_PX;

use super::{MAX_REGION_DIM, MAX_STAMPS, StrokeSpans};

/// One swept segment of the stroke.
#[derive(Copy, Clone)]
pub(super) struct Segment {
    pub(super) start: Vec2,
    pub(super) dir: Vec2,
    pub(super) radius: f32,
    pub(super) length: f32,
    /// Paint **height** laid per unit swept optical depth: the brush's `add` source
    /// faded by the remaining load (`drain`). The single amount knob — the amount of
    /// paint and its per-unit opacity are independent (DESIGN.md §6.1), which is why
    /// `opacity` below is not derived from it.
    pub(super) amount: f32,
    /// Paint opacity laid by this segment (the brush's opacity × remaining load).
    /// Drives the color/opacity channel; the amount laid is independent
    /// (DESIGN.md §6.1, normalized representation).
    pub(super) opacity: f32,
    /// Shape orientation for this segment as a fraction of a full turn ∈ [0, 1): the
    /// relative angle between the shape's native axis and the travel direction, used to
    /// pick the prefix-τ orientation layer. 0 for follow-stroke (DESIGN.md §6.6).
    pub(super) orient: f32,
    /// Arc length from the stroke start to this segment's start (canvas px) — the
    /// third axis of the colour-dynamics noise lookup (DESIGN.md §6.2).
    pub(super) dist: f32,
}

/// Per-segment instance data for the sweep shader.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(super) struct SegmentInstance {
    pub(super) start: [f32; 2],
    pub(super) dir: [f32; 2],   // unit tangent
    pub(super) geom: [f32; 4],  // radius, length, amount (height per unit τ), opacity
    pub(super) extra: [f32; 4], // orientation (turns ∈ [0,1)), arc length at segment start, _, _
}

/// Generate the round tip's coverage: a soft disc with `hardness` falloff.
pub(super) fn round_coverage(hardness: f32, res: u32) -> Vec<f32> {
    let h = hardness.clamp(0.0, 0.99);
    let mut cov = vec![0.0f32; (res * res) as usize];
    for y in 0..res {
        for x in 0..res {
            let fx = (x as f32 + 0.5) / res as f32 * 2.0 - 1.0;
            let fy = (y as f32 + 0.5) / res as f32 * 2.0 - 1.0;
            let r = (fx * fx + fy * fy).sqrt();
            cov[(y * res + x) as usize] = 1.0 - smoothstep(h, 1.0, r);
        }
    }
    cov
}

pub(super) fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Build swept segments from the fitted control points (DESIGN.md §6.2): flatten
/// the curve adaptively, then make each polyline edge a segment. The one-way load
/// reservoir (`drain`) depletes with arc distance; radius follows pressure.
///
/// Returns the range's segments plus the arc length at its end — measured on the
/// emitted polyline rather than recomputed, so the range that resumes from it starts
/// on the exact accumulator these segments were built with.
pub(super) fn generate_segments_in(
    rec: &StrokeRecord,
    tol: crate::path::FlattenTolerance,
    spans: StrokeSpans,
) -> (Vec<Segment>, f32) {
    let b = &rec.brush;
    let dist0 = spans.dist;
    let pts = crate::path::flatten_spans(&rec.path, spans.range, dist0, tol);
    let end_dist = pts.last().map_or(dist0, |p| p.dist);
    let mut segs = Vec::new();
    if pts.is_empty() {
        return (segs, end_dist);
    }

    // Attributes are constant across a swept segment, so they are taken at its
    // *midpoint* rather than its start — with adaptive flattening a segment can be
    // long, and start-sampling would lag every ramp by half a segment. `dist` is
    // the exception: it is the segment start's arc length because the shader adds
    // the fragment's own offset along the travel to it (stamp_common.wesl).
    let make = |pos: Vec2, pressure: f32, tilt: Vec2, dir: Vec2, len: f32, dist: f32| {
        let drain = (1.0 - b.drain * (dist + len * 0.5)).max(0.0);
        Segment {
            start: pos,
            dir,
            radius: (b.radius * pressure).max(0.5),
            length: len,
            // The `add` source is the one amount knob; the brush's opacity (color[3])
            // rides the separate opacity channel (DESIGN.md §6.1).
            amount: b.dynamics.add * drain,
            opacity: b.color[3] * drain,
            orient: orientation_turns(b.orientation, dir, tilt),
            dist,
        }
    };

    for w in pts.windows(2) {
        let (a, c) = (w[0], w[1]);
        let v = c.pos - a.pos;
        let len = v.length();
        if len < 1e-5 {
            continue;
        }
        let pressure = (a.pressure + c.pressure) * 0.5;
        let tilt = (a.tilt + c.tilt) * 0.5;
        segs.push(make(a.pos, pressure, tilt, v / len, len, a.dist));
    }

    if segs.is_empty() {
        // A click: sweep a fraction of a radius so it deposits a soft blob.
        let p = pts[0];
        let r = (b.radius * p.pressure).max(0.5);
        segs.push(make(
            p.pos,
            p.pressure,
            p.tilt,
            Vec2::new(1.0, 0.0),
            r * 0.6,
            0.0,
        ));
    }
    (segs, end_dist)
}

/// The stroke's colour-dynamics uniform triplet — (frequency + 1/NOISE_TILE_PX,
/// per-channel amplitude, per-stroke lookup translation) — shared by the sweep's
/// `TileXform` and the dynamics loop's `Stamp` slots so both paths jitter
/// identically. Inactive jitter zeroes frequency *and* amplitude, so with the
/// zero volume bound the shader's early-out keeps the deposit bit-identical.
pub(super) fn noise_uniform(rec: &StrokeRecord) -> ([f32; 4], [f32; 4], [f32; 4]) {
    let cd = rec.brush.color_dynamics;
    let (freq, amp) = if cd.is_active() {
        (cd.frequency, cd.amplitude)
    } else {
        ([0.0; 3], [0.0; 3])
    };
    let off = noise_offset(rec.seed);
    (
        [freq[0], freq[1], freq[2], 1.0 / NOISE_TILE_PX],
        [amp[0], amp[1], amp[2], 0.0],
        [off[0], off[1], off[2], 0.0],
    )
}

/// The per-stroke noise lookup translation in [0, 1)³, derived from the stroke
/// seed via splitmix64 — each stroke samples a fresh part of the tileable field,
/// deterministically (replay and live == committed hold, DESIGN.md §6.2).
pub(super) fn noise_offset(seed: u64) -> [f32; 3] {
    let mut state = seed;
    [(); 3].map(|_| {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        // Top 24 bits → [0, 1): exact in f32, uniform.
        (z >> 40) as f32 / (1u64 << 24) as f32
    })
}

/// The shape's orientation for a segment, as a fraction of a full turn ∈ [0, 1): the
/// relative angle between the shape's native axis and the travel direction `dir`, which
/// picks the prefix-τ orientation layer (DESIGN.md §6.6).
///
/// - [`OrientationSource::FollowStroke`]: the shape tracks the tangent, so the relative
///   angle is always 0 (the historical behaviour; for a round tip it is moot anyway).
/// - [`OrientationSource::Pen`]: the shape is pinned to the pen's azimuth (the tilt
///   direction) in canvas space, so relative to the travel direction it is `α − φ` — as
///   the stroke curves the footprint angle stays fixed in the world, like a nib.
pub(super) fn orientation_turns(source: OrientationSource, dir: Vec2, tilt: Vec2) -> f32 {
    match source {
        OrientationSource::FollowStroke => 0.0,
        OrientationSource::Pen => {
            let alpha = tilt.y.atan2(tilt.x); // pen azimuth (0 when the pen is upright / mouse)
            let phi = dir.y.atan2(dir.x); // travel direction
            ((alpha - phi) / std::f32::consts::TAU).rem_euclid(1.0)
        }
    }
}

/// Tiles whose *texture* (interior + apron) any segment's swept capsule overlaps.
/// The apron is included in `reach` so a stroke landing within a tile's interior
/// but inside a neighbor's apron band re-renders that neighbor too, keeping the
/// shared apron/interior overlap bit-identical (DESIGN.md §6.4).
pub(super) fn affected_tiles(segments: &[Segment]) -> BTreeSet<TileCoord> {
    let tile = TILE_SIZE as f32;
    let mut coords = BTreeSet::new();
    for s in segments {
        let (lo, hi) = segment_bounds(s);
        let (x0, x1) = ((lo.x / tile).floor() as i32, (hi.x / tile).floor() as i32);
        let (y0, y1) = ((lo.y / tile).floor() as i32, (hi.y / tile).floor() as i32);
        for y in y0..=y1 {
            for x in x0..=x1 {
                coords.insert(TileCoord::new(x, y));
            }
        }
    }
    coords
}

/// The canvas box one segment's swept coverage occupies, grown by the apron a
/// rewritten tile's neighbours reach into (§6.4). The one place that reach is
/// defined: [`affected_tiles`] enumerates the tiles it touches, [`chunk_segments`]
/// accumulates it into the region a run of segments needs, and those two answers have
/// to be the same rectangle.
fn segment_bounds(s: &Segment) -> (Vec2, Vec2) {
    let end = s.start + s.dir * s.length;
    let reach = Vec2::splat(s.radius + TILE_APRON as f32);
    (s.start.min(end) - reach, s.start.max(end) + reach)
}

/// The size of the region [`region_rect`] would build for a coverage box, without
/// building the tile set.
///
/// Same rectangle, reached by bounding box rather than by enumerating tiles:
/// [`chunk_segments`] asks this question once per segment while it walks a stroke,
/// and `affected_tiles` costs a set insert per tile per segment — on a long stroke,
/// the very cost the incremental repaint exists to avoid.
fn region_of(lo: Vec2, hi: Vec2) -> (u32, u32) {
    let tile = TILE_SIZE as f32;
    // The tile block the coverage spans, measured between tile origins.
    let span = |a: f32, b: f32| ((b / tile).floor() - (a / tile).floor()) * tile;
    (
        span(lo.x, hi.x) as u32 + TILE_TEX,
        span(lo.y, hi.y) as u32 + TILE_TEX,
    )
}

/// Split a stroke's segments into consecutive runs, each of which the stamp loop can
/// evolve inside one [`MAX_REGION_DIM`]-bounded region (DESIGN.md §6.2).
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
pub(super) fn chunk_segments(segments: &[Segment]) -> Vec<Range<usize>> {
    let mut runs = Vec::new();
    let (mut lo, mut hi) = (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY));
    let mut start = 0;
    for (i, s) in segments.iter().enumerate() {
        let (slo, shi) = segment_bounds(s);
        let (glo, ghi) = (lo.min(slo), hi.max(shi));
        let (w, h) = region_of(glo, ghi);
        if i > start && (w > MAX_REGION_DIM || h > MAX_REGION_DIM || i - start >= MAX_STAMPS) {
            runs.push(start..i);
            (start, lo, hi) = (i, slo, shi);
        } else {
            (lo, hi) = (glo, ghi);
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
/// flattening cap — or at the 0.6 radii a click sweeps, which ignores the cap — and a
/// coverage box of a given extent spans at most one tile more than it covers,
/// whichever tile boundary it happens to straddle.
pub(super) fn segment_fits_region(b: &BrushParams, tol: crate::path::FlattenTolerance) -> bool {
    let radius = b.radius.max(0.5);
    let length = tol.max_len.max(0.6 * radius);
    let extent = length + 2.0 * (radius + TILE_APRON as f32);
    let worst = (extent / TILE_SIZE as f32).ceil().max(0.0) as u32 * TILE_SIZE + TILE_TEX;
    worst <= MAX_REGION_DIM
}

/// The region the stamp loop evolves for a stroke piece's affected `coords`: exactly
/// the tile block they span, grown by one apron on each side so the write-back can
/// slice whole `TILE_TEX` blocks out of it — plus the *list* of tiles to composite
/// into it, which is those tiles and the one-tile ring around them (§6.4).
///
/// The ring is in the tile list but deliberately **not** in the rectangle. Its whole
/// job is to give a rewritten tile's apron the neighbour interior it overlaps, and an
/// apron is [`TILE_APRON`] texels — so extending the rectangle by a whole *tile* on
/// every side, as it once did, paid for roughly 4× the region to fill a one-texel
/// band. Ring tiles that fall outside the rectangle simply clip when composited. On a
/// live tail, which covers a handful of tiles and is redrawn on every pointer move,
/// that difference is most of the cost of the whole path.
///
/// Returns `(tiles to composite, lo origin, region origin, w, h)`, or `None` if
/// `coords` is empty. The size is [`chunk_segments`]'s business, not this one's — it
/// hands over pieces that fit by construction.
pub(super) fn region_rect(
    coords: &BTreeSet<TileCoord>,
) -> Option<(Vec<TileCoord>, Vec2, Vec2, u32, u32)> {
    let mut lo = Vec2::splat(f32::INFINITY);
    let mut hi = Vec2::splat(f32::NEG_INFINITY);
    for c in coords {
        lo = lo.min(c.origin());
        hi = hi.max(c.origin());
    }
    if !lo.x.is_finite() {
        return None;
    }
    let w = (hi.x - lo.x) as u32 + TILE_TEX;
    let h = (hi.y - lo.y) as u32 + TILE_TEX;
    let mut halo: BTreeSet<TileCoord> = BTreeSet::new();
    for c in coords {
        for dy in -1..=1 {
            for dx in -1..=1 {
                halo.insert(TileCoord::new(c.x + dx, c.y + dy));
            }
        }
    }
    let region_origin = lo - Vec2::splat(TILE_APRON as f32);
    Some((halo.into_iter().collect(), lo, region_origin, w, h))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A segment carrying only what the region measurements read.
    fn seg(start: Vec2, end: Vec2, radius: f32) -> Segment {
        let v = end - start;
        let length = v.length();
        Segment {
            start,
            dir: if length > 0.0 {
                v / length
            } else {
                Vec2::new(1.0, 0.0)
            },
            radius,
            length,
            amount: 0.0,
            opacity: 1.0,
            orient: 0.0,
            dist: 0.0,
        }
    }

    /// The union of every segment's [`segment_bounds`], as [`chunk_segments`]
    /// accumulates it.
    fn measured(segments: &[Segment]) -> Option<(u32, u32)> {
        let (mut lo, mut hi) = (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY));
        for s in segments {
            let (slo, shi) = segment_bounds(s);
            (lo, hi) = (lo.min(slo), hi.max(shi));
        }
        lo.x.is_finite().then(|| region_of(lo, hi))
    }

    /// [`chunk_segments`] decides where to cut a stroke by measuring the region a run
    /// of segments would need with [`region_of`], but the render that follows sizes
    /// the actual textures from [`region_rect`]. They are two ways of measuring one
    /// rectangle — bounding box versus enumerated tiles — so they have to agree
    /// exactly. If the bounding box ever under-reported, a piece would allocate past
    /// [`MAX_REGION_DIM`]; if it over-reported, strokes would be cut into more pieces
    /// than they need, each paying for its own region composite.
    #[test]
    fn the_chunker_measures_the_region_the_render_builds() {
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
            let want = region_rect(&affected_tiles(&segments)).map(|(_, _, _, w, h)| (w, h));
            assert_eq!(
                measured(&segments),
                want,
                "region size disagrees for {what}"
            );
        }
        assert_eq!(measured(&[]), None, "no segments is not a region");
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
        let runs = chunk_segments(&segments);
        assert!(runs.len() > 1, "an oversized stroke should be cut up");

        let mut next = 0;
        for run in &runs {
            assert_eq!(run.start, next, "the pieces leave a gap or overlap");
            next = run.end;
            let (w, h) = measured(&segments[run.clone()]).expect("a piece is never empty");
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
            segment_fits_region(&b, super::super::flatten_tolerance(&b))
        };
        assert!(fits(1.0), "a hairline tip fits");
        assert!(fits(120.0), "the largest tip the UI offers fits");
        assert!(
            !fits(MAX_REGION_DIM as f32),
            "a tip wider than the whole region cannot fit"
        );
    }
}
