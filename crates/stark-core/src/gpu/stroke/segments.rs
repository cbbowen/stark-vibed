//! Turning a fitted path into swept segments, and the region measurements that decide
//! where the stamp loop cuts one into pieces (§6.2).
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
///
/// The centreline is a **circular arc**, not a chord: `start` and `dir` give the
/// frame it leaves in, `curvature` bends it, and `length` measures along it. A
/// straight sweep is `curvature == 0` and is what every quantity below reduces to,
/// exactly — see [`crate::path::fit_arc`].
#[derive(Copy, Clone)]
pub(super) struct Segment {
    pub(super) start: Vec2,
    /// Unit travel tangent **at the segment's start** — the x axis of the frame the
    /// sweep is integrated in. On a curved segment the tangent turns as the tip
    /// travels; this is where it begins.
    pub(super) dir: Vec2,
    /// Signed curvature of the centreline (1/canvas px), positive turning towards
    /// the left of `dir`. Exactly 0 for a straight sweep, which both render paths
    /// branch on — so a stroke the arc fit declines to bend is bit-identical to one
    /// drawn before arcs existed (§6.2).
    pub(super) curvature: f32,
    pub(super) radius: f32,
    /// Arc length of the centreline (canvas px) — the tip's own travel, which is the
    /// measure every rate in both paths is denominated in.
    pub(super) length: f32,
    /// Shape orientation for this segment as a fraction of a full turn ∈ [0, 1): the
    /// relative angle between the shape's native axis and the travel direction, used to
    /// pick the prefix-τ orientation layer. 0 for follow-stroke (§6.6).
    pub(super) orient: f32,
    /// Arc length from the stroke start to this segment's start (canvas px) — the
    /// third axis of the colour-dynamics noise lookup (§6.2).
    pub(super) dist: f32,
}

/// Per-segment instance data for the sweep shader. Carries only what actually varies
/// from segment to segment — the paint rates are stroke constants and ride the
/// `TileXform` uniform instead (see [`generate_segments_in`]).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(super) struct SegmentInstance {
    pub(super) start: [f32; 2],
    pub(super) dir: [f32; 2],  // unit tangent at the segment start
    pub(super) geom: [f32; 2], // radius, arc length
    // orientation (turns ∈ [0,1)), arc length at segment start, signed curvature, _
    pub(super) extra: [f32; 4],
}

/// Generate the round tip's coverage: a soft disc with `hardness` falloff.
pub(super) fn round_coverage(hardness: f32, res: u32) -> Vec<f32> {
    let h = 1.0 / (1.0 - hardness).max(0.01);
    let mut cov = vec![0.0f32; (res * res) as usize];
    for y in 0..res {
        for x in 0..res {
            let fx = (x as f32 + 0.5) / res as f32 * 2.0 - 1.0;
            let fy = (y as f32 + 0.5) / res as f32 * 2.0 - 1.0;
            let r2 = fx * fx + fy * fy;
            let k = 1.0 / (1.0 - fy * fy).max(1e-5).sqrt();
            cov[(y * res + x) as usize] = k * (1.0 - r2.min(1.0).powf(0.5 * h)).max(0.0);
        }
    }
    cov
}

// --- swept arcs ----------------------------------------------------------------
//
// The arc a flattened edge stands for is [`crate::path::fit_arc`]'s, called here with
// the very cap the flattener called it with (`FlattenTolerance::max_arc_curvature`,
// set by [`flatten_tolerance`](super::flatten_tolerance) from
// [`MAX_TIP_TURN`](super::MAX_TIP_TURN)). One function, one rule, so the geometry the
// flattener priced is the geometry that gets swept — and neither can spend the
// positional budget on a primitive the other does not use.

/// The taper's radius profile: the fraction of the brush's radius in force `t` of
/// the way through a taper (§6.2).
///
/// `f(t) = t(3 − t²)/2` — the cubic pinned by `f(0) = 0`, `f(1) = 1`, `f'(1) = 0`,
/// monotone on `[0, 1]`, and within 2% of `sin(πt/2)` everywhere. Both end
/// conditions are the point:
///
/// * `f'(1) = 0` is what makes the taper *smooth*. The taper meets the stroke's
///   full-width body there, and any profile with a slope left at the join (`√t`,
///   plain `t`) puts a visible crease across the stroke where the two meet — the
///   one artifact that would give the trick away.
/// * `f'(0) = 3/2` is what makes it a **point** rather than a blunt cap or a
///   hairline. The outline leaves the tip as a straight wedge, which is what an
///   inked entry stroke looks like; `smoothstep`'s `f'(0) = 0` instead holds the
///   width near zero for a tenth of the taper and reads as a whisker with a bulge
///   behind it.
///
/// A polynomial rather than the sine it approximates because it has to be
/// bit-identical across platforms: the taper decides stored pixels, so replay,
/// goldens and peers all have to agree on it (§12.1), and `sin` is not
/// specified to the last bit.
fn taper_profile(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    0.5 * t * (3.0 - t * t)
}

/// The largest `d/dt` [`taper_profile`] reaches, at `t = 0`. Used to bound how far
/// the radius can move across one swept segment.
const TAPER_MAX_SLOPE: f32 = 1.5;

/// The travel a stroke deposits at its very shortest, in radii of the tip in force
/// (§6.2) — the **touch-down dab**.
///
/// A swept deposit is a definite integral over travel, so a press that has not moved
/// integrates over nothing and lays nothing, and the first pixel of a drag lays a
/// twentieth of one radius' worth. Taken literally that is a tool that draws nothing
/// until the hand has moved a tip's width, which is not what pressing a loaded brush
/// to paper does.
///
/// So a stroke travels at least this far: whatever it is short by is made up by a
/// *dwell* segment swept about the stroke's own midpoint. A click is the limiting
/// case — the whole dab, centred on the one point that was pressed — and the dwell
/// shrinks to nothing by the time the tip has travelled this far under its own steam,
/// so the mark grows continuously from a dot into a stroke instead of jumping between
/// the two.
///
/// The value is what the mark *looks* like rather than a bound on anything: 0.6
/// radii of dwell is a little over a third of the optical depth a full pass lays,
/// which the slab law (§6.1) renders as a dot around 90% as opaque as the stroke the
/// same brush draws — a dab, and a round-looking one, since it stretches the tip's
/// own footprint by less than a third.
pub(super) const DAB_TRAVEL: f32 = 0.6;

/// Max change in the taper's radius factor across one swept segment.
///
/// The same 2% step [`crate::path::FLATTEN_TOLERANCE`] holds a *pen attribute* to,
/// and for exactly the same reason: pressure and the taper both scale the radius,
/// and a segment sweeps at a constant one, so a coarser step draws a taper as a
/// staircase of shrinking dabs rather than a smooth point.
const TAPER_STEP: f32 = 0.02;

/// Cap on the pieces one flattened edge is cut into for the taper. Well above what
/// the step above ever asks for (a whole taper costs ~`TAPER_MAX_SLOPE / TAPER_STEP`
/// = 75 pieces, however long it is), so this is a backstop on a pathological brush
/// rather than a quality knob.
const TAPER_MAX_PIECES: usize = 128;

/// A stroke's taper, resolved for one span range (§6.2).
///
/// Both lengths are in canvas px here, already scaled out of
/// [`BrushParams::taper_px`] and — crucially — already **fitted to the stroke**: if
/// the two zones together are longer than the stroke, both are scaled down in
/// proportion so they exactly meet. The stroke then reaches full width at one point
/// instead of never reaching it, which is what keeps a quick flick a small pointed
/// mark rather than a sliver, continuously as the stroke grows.
#[derive(Copy, Clone, Debug)]
struct Taper {
    /// Leading taper length (canvas px); 0 = none.
    start: f32,
    /// Trailing taper length (canvas px); 0 = none.
    end: f32,
    /// Arc length of the whole stroke, for measuring back from its end. Only read
    /// when `end > 0`.
    total: f32,
}

impl Taper {
    /// The taper in force for a range, given the stroke's total arc length — or
    /// `None` if this range stops short of the stroke's end and so cannot know it.
    ///
    /// A range that does not reach the end gets the **leading taper alone,
    /// uncompressed**. That is not a guess: the engine refuses to freeze any span
    /// that is within the trailing taper's reach of the live end, or that could
    /// still be compressed ([`safe_frozen`](super::safe_frozen)), so a
    /// partial range is one where both of those factors are exactly 1 — and the
    /// commit, which sees the whole stroke, computes the same 1 for it.
    fn resolve(b: &BrushParams, total: Option<f32>) -> Self {
        let (start, end) = b.taper_px();
        match total {
            Some(total) if start + end > total => {
                // Scaled in proportion, so the two zones meet at one point.
                let k = total / (start + end);
                Self {
                    start: start * k,
                    end: end * k,
                    total,
                }
            }
            Some(total) => Self { start, end, total },
            None => Self {
                start,
                end: 0.0,
                total: f32::INFINITY,
            },
        }
    }

    /// The fraction of the brush's radius in force at arc length `dist`.
    fn factor(&self, dist: f32) -> f32 {
        let mut f = 1.0;
        if self.start > 0.0 {
            f *= taper_profile(dist / self.start);
        }
        if self.end > 0.0 {
            f *= taper_profile((self.total - dist) / self.end);
        }
        f
    }

    /// A bound on `|d factor / d dist|` anywhere in `[dist, dist + len]`.
    ///
    /// Each zone contributes at most `TAPER_MAX_SLOPE / length`, and the product
    /// rule bounds the two together by their sum (both factors are ≤ 1). Zones the
    /// interval cannot reach contribute nothing, which is what keeps the extra
    /// subdivision below paid only near the ends of the stroke.
    fn slope_bound(&self, dist: f32, len: f32) -> f32 {
        let mut slope = 0.0;
        if self.start > 0.0 && dist < self.start {
            slope += TAPER_MAX_SLOPE / self.start;
        }
        if self.end > 0.0 && self.total - (dist + len) < self.end {
            slope += TAPER_MAX_SLOPE / self.end;
        }
        slope
    }

    /// How many swept segments a flattened edge of length `len` starting at `dist`
    /// has to be cut into to keep the radius stepping smoothly (see [`TAPER_STEP`]).
    /// 1 — no cut at all — wherever the taper is flat, which is everywhere on an
    /// untapered brush, so this path is bit-identical to having no taper code.
    fn pieces(&self, dist: f32, len: f32) -> usize {
        let slope = self.slope_bound(dist, len);
        if slope <= 0.0 {
            return 1;
        }
        // Float → int casts saturate in Rust, so a nonsense length cannot wrap here.
        ((slope * len / TAPER_STEP).ceil() as usize).clamp(1, TAPER_MAX_PIECES)
    }
}

/// Build swept segments from the fitted control points (§6.2): flatten
/// the curve adaptively, then make each polyline edge a segment. Radius follows
/// pressure and the stroke's start/end tapers.
///
/// **The `drain` falloff is deliberately not here.** It is a function of arc length
/// alone, and every shader that reads a segment already knows the arc length of the
/// fragment it is shading (`dist` plus the fragment's own offset along the travel), so
/// it is evaluated there instead of being baked in per segment. That is not a
/// micro-optimization: a per-segment factor makes the paint laid depend on where the
/// segment boundaries happened to fall, which is the one thing §6.2 works to keep out
/// of the deposit. Evaluated per fragment it drops out of the sum entirely — the
/// stroke lays `a(arc) · Στ`, and `Στ` is already independent of the cut — so the
/// flattener no longer has to buy accuracy for it with segments
/// (see [`flatten_tolerance`](super::flatten_tolerance)).
///
/// Returns the range's segments plus the arc length at its end — measured on the
/// emitted polyline rather than recomputed, so the range that resumes from it starts
/// on the exact accumulator these segments were built with.
///
/// Two things here are measured against the stroke's **whole** length, which only a
/// range that reaches its final span knows: the trailing taper (a range that stops
/// short takes the leading taper alone, [`Taper::resolve`]) and the touch-down dab
/// ([`DAB_TRAVEL`], which a partial range never has). Both are sound rather than
/// approximate, and [`safe_frozen`](super::safe_frozen) is the one rule that makes
/// them so.
pub(super) fn generate_segments_in(
    rec: &StrokeRecord,
    tol: crate::path::FlattenTolerance,
    spans: StrokeSpans,
) -> (Vec<Segment>, f32) {
    let b = &rec.brush;
    let dist0 = spans.dist;
    let reaches_end = spans.range.end >= crate::path::span_count(rec.path.len());
    let from_start = spans.range.start == 0;
    let pts = crate::path::flatten_spans(&rec.path, spans.range, dist0, tol);
    let end_dist = pts.last().map_or(dist0, |p| p.dist);
    let mut segs = Vec::new();
    if pts.is_empty() {
        return (segs, end_dist);
    }
    let taper = Taper::resolve(b, reaches_end.then_some(end_dist));

    // Attributes are constant across a swept segment, so they are taken at its
    // *midpoint* rather than its start — with adaptive flattening a segment can be
    // long, and start-sampling would lag every ramp by half a segment. `dist` is
    // the exception: it is the segment start's arc length because the shader adds
    // the fragment's own offset along the travel to it (stamp_common.wesl).
    //
    // `dir` is the tangent the sweep *starts* along (the frame's x axis) while
    // `mid_dir` is the one at the midpoint — the same midpoint-sampling argument,
    // applied to the one attribute that reads a direction. They are the same vector
    // on a straight segment.
    #[allow(clippy::too_many_arguments)]
    let make = |pos: Vec2,
                pressure: f32,
                tilt: Vec2,
                dir: Vec2,
                mid_dir: Vec2,
                kappa: f32,
                len: f32,
                dist: f32,
                tap: f32| {
        Segment {
            start: pos,
            dir,
            curvature: kappa,
            // Pressure and the taper both scale the tip; the floor keeps a tapered
            // tip a hairline at its very point rather than a degenerate zero-width
            // sweep (which would also divide by zero in the dynamics loop's
            // reservoir cadence).
            radius: (b.radius * pressure * tap).max(0.5),
            length: len,
            orient: orientation_turns(b.orientation, mid_dir, tilt),
            dist,
        }
    };

    for w in pts.windows(2) {
        let (a, c) = (w[0], w[1]);
        let v = c.pos - a.pos;
        let chord = v.length();
        if chord < 1e-5 {
            continue;
        }
        // The edge as an arc rather than a chord (see [`segment_arc`]): same
        // endpoints, but leaving along the curve's own tangent, so the swept outline
        // no longer breaks its curvature at every joint. Curvature 0 comes back for a
        // straight or barely-curved edge and everything below reduces to what it was.
        let crate::path::Arc {
            dir,
            curvature: kappa,
            length: len,
        } = crate::path::fit_arc(a.vel, v, tol.max_arc_curvature);
        // One flattened edge is one segment wherever the taper is flat — which is
        // everywhere on an untapered brush, so nothing below changes those strokes
        // by a bit. Inside a taper it is cut into pieces fine enough that the radius
        // steps smoothly, the same length bound `drain` and the reservoir cadence
        // ask of the *fitter* (`flatten_tolerance`), except paid only near the ends
        // instead of over the whole stroke. The pieces are sub-*arcs*: they inherit
        // the edge's curvature and are stepped along it, so cutting an edge up still
        // traces exactly the same centreline.
        let n = taper.pieces(a.dist, len);
        let step = len / n as f32;
        for k in 0..n {
            let mid = (k as f32 + 0.5) / n as f32;
            let pressure = a.pressure + (c.pressure - a.pressure) * mid;
            let tilt = a.tilt + (c.tilt - a.tilt) * mid;
            let along = step * k as f32;
            let dist = a.dist + along;
            let (pos, tan) = crate::path::arc_at(a.pos, dir, kappa, along);
            let (_, mid_tan) = crate::path::arc_at(a.pos, dir, kappa, along + step * 0.5);
            segs.push(make(
                pos,
                pressure,
                tilt,
                tan,
                mid_tan,
                kappa,
                step,
                dist,
                taper.factor(dist + step * 0.5),
            ));
        }
    }

    // The touch-down dab ([`DAB_TRAVEL`]): a stroke that has not yet travelled a dab's
    // worth sweeps the difference about its own **midpoint**, so a click leaves the
    // whole dab centred where it was pressed and every longer stroke grows out of that
    // one continuously. Centred rather than led from the start point because a dab has
    // no direction to lead in: a click has no tangent at all (the fitter gives a lone
    // knot none), and swept from the point it would read as a short dash in whatever
    // direction the fallback happened to name.
    //
    // Only a range that is the **whole** stroke may add it, for the trailing taper's
    // reason exactly — the length it is measured against is the stroke's, which a
    // partial range does not know. And for the same reason it is sound rather than
    // approximate: [`safe_frozen`](super::safe_frozen) refuses to freeze anything
    // until the stroke has travelled a whole dab, so a partial range is always one
    // whose dab is zero, and the commit computes zero for it too.
    //
    // `dab_bound` is the longest dwell any stroke of this brush could owe — pressure
    // and the taper only ever scale the tip *down*, and the fitter clamps both to the
    // curve as well as to its control points. So a stroke past it is past every dab,
    // and an ordinary stroke leaves here without so much as walking its own polyline.
    let dab_bound = DAB_TRAVEL * b.radius.max(0.5);
    if reaches_end && from_start && end_dist < dab_bound {
        // `range.start == 0` is what makes `end_dist` the whole stroke's arc length:
        // `dist` is the arc *before* the range, and before the first span there is none.
        let mid = end_dist * 0.5;
        let (pos, dir, pressure, tilt) = sample_at(&pts, mid);
        // The tip in force at the midpoint — where a stroke short enough to want a dab
        // is at its widest, its two compressed taper zones meeting there (`Taper`). A
        // click is the limit of that: zero length compresses both zones to nothing, so
        // `factor` is exactly 1 and a tapered brush dots at full size rather than
        // leaving the invisible speck a taper read literally would give.
        let tap = taper.factor(mid);
        let dwell = DAB_TRAVEL * (b.radius * pressure * tap).max(0.5) - end_dist;
        if dwell > 0.0 {
            segs.insert(
                0,
                make(
                    pos - dir * (dwell * 0.5),
                    pressure,
                    tilt,
                    dir,
                    dir,
                    0.0,
                    dwell,
                    // The dwell is *at* the stroke, not before it: it must not run the
                    // arc-length clock — which `drain` and the colour noise are
                    // measured on — backwards past the stroke's own start.
                    (mid - dwell * 0.5).max(0.0),
                    tap,
                ),
            );
        }
    }
    (segs, end_dist)
}

/// The stroke's state at arc length `arc` along a flattened polyline: position, unit
/// travel direction, pressure and tilt. Only the touch-down dab asks
/// ([`DAB_TRAVEL`]), and only ever about the midpoint of a stroke that is at most a
/// dab long.
///
/// `+x` where the stroke has no direction to give — a click, whose one knot the
/// fitter leaves with no tangent, and a press that reported the same position twice.
/// Which direction that is cannot matter, because the dab is swept symmetrically
/// about the point it is asked for.
fn sample_at(pts: &[crate::path::IntermediateSample], arc: f32) -> (Vec2, Vec2, f32, Vec2) {
    for w in pts.windows(2) {
        let (a, c) = (w[0], w[1]);
        let span = c.dist - a.dist;
        if span > 0.0 && arc <= c.dist {
            let t = ((arc - a.dist) / span).clamp(0.0, 1.0);
            let v = c.pos - a.pos;
            let len = v.length();
            return (
                a.pos + v * t,
                if len > 1e-5 {
                    v / len
                } else {
                    Vec2::new(1.0, 0.0)
                },
                a.pressure + (c.pressure - a.pressure) * t,
                a.tilt + (c.tilt - a.tilt) * t,
            );
        }
    }
    let p = pts.last().expect("a flattened range is never empty here");
    (p.pos, Vec2::new(1.0, 0.0), p.pressure, p.tilt)
}

/// The stroke's colour-dynamics uniform triplet — (per-axis frequency
/// (across the stroke, along it) + 1/NOISE_TILE_PX,
/// per-channel amplitude, per-stroke lookup translation) — shared by the sweep's
/// `TileXform` and the dynamics loop's `Stamp` slots so both paths jitter
/// identically. Inactive jitter zeroes frequency *and* amplitude, so with the
/// zero volume bound the shader's early-out keeps the deposit bit-identical.
pub(super) fn noise_uniform(rec: &StrokeRecord) -> ([f32; 4], [f32; 4], [f32; 4]) {
    let cd = rec.brush.color_dynamics;
    let (freq, amp) = if cd.is_active() {
        (cd.frequency, cd.amplitude)
    } else {
        ([0.0; 2], [0.0; 3])
    };
    let off = noise_offset(rec.seed);
    (
        [freq[0], freq[1], 1.0 / NOISE_TILE_PX, 0.0],
        [amp[0], amp[1], amp[2], 0.0],
        [off[0], off[1], 0.0, 0.0],
    )
}

/// The per-stroke noise lookup translation in [0, 1)², derived from the stroke
/// seed via splitmix64 — each stroke samples a fresh part of the tileable field,
/// deterministically (replay and live == committed hold, §6.2).
pub(super) fn noise_offset(seed: u64) -> [f32; 2] {
    let mut state = seed;
    [(); 2].map(|_| {
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
/// picks the prefix-τ orientation layer (§6.6).
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
/// shared apron/interior overlap bit-identical (§6.4).
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

/// Where a segment's centreline ends — along the arc, not along the chord.
pub(super) fn segment_end(s: &Segment) -> Vec2 {
    crate::path::arc_at(s.start, s.dir, s.curvature, s.length).0
}

/// The canvas box one segment's swept coverage occupies — the arc, grown by the tip
/// that rides along it.
///
/// The rasterized geometry reaches further than this at the caps (the shaders sweep a
/// generous angular margin so the round end is never clipped), but every fragment out
/// there differences two prefix taps to exactly zero and writes nothing. What a box
/// has to contain is where the deposit *lands*, which is within one radius of the arc.
pub(super) fn coverage_bounds(s: &Segment) -> (Vec2, Vec2) {
    let end = segment_end(s);
    let reach = Vec2::splat(s.radius + crate::path::arc_sagitta(s.curvature, s.length));
    (s.start.min(end) - reach, s.start.max(end) + reach)
}

/// [`coverage_bounds`] grown by the apron a rewritten tile's neighbours reach into
/// (§6.4). The one place that reach is defined: [`affected_tiles`] enumerates the
/// tiles it touches, [`chunk_segments`] accumulates it into the region a run of
/// segments needs, and those two answers have to be the same rectangle.
fn segment_bounds(s: &Segment) -> (Vec2, Vec2) {
    let (lo, hi) = coverage_bounds(s);
    let apron = Vec2::splat(TILE_APRON as f32);
    (lo - apron, hi + apron)
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
/// flattening cap — or at the [`DAB_TRAVEL`] radii a touch-down dab sweeps, which
/// ignores the cap — and a coverage box of a given extent spans at most one tile more
/// than it covers, whichever tile boundary it happens to straddle.
pub(super) fn segment_fits_region(b: &BrushParams, tol: crate::path::FlattenTolerance) -> bool {
    let radius = b.radius.max(0.5);
    // The chord is what `path::within` caps; the arc over it is longer, and bows a
    // sagitta out of its own box. Both are bounded by the turn a segment may bend
    // through (`MAX_HALF_TURN_SIN`) — under 2% and under 5% of the chord — so a
    // single margin covers the pair with room to spare.
    let length = tol.max_len.max(DAB_TRAVEL * radius) * 1.1;
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

    // --- tapers ----------------------------------------------------------

    /// A straight stroke `len` px long with a tapered brush of `radius`.
    fn tapered_record(radius: f32, start: f32, end: f32, len: f32) -> StrokeRecord {
        // Enough control points that the curve has spans to freeze part of, and
        // straight so arc length is the chord and the taper zones are easy to reason
        // about.
        let path: Vec<crate::path::ControlPoint> = (0..=12)
            .map(|i| crate::path::ControlPoint::at(Vec2::new(i as f32 / 12.0 * len, 0.0)))
            .collect();
        StrokeRecord {
            layer: crate::document::LayerId(0),
            tool: crate::document::Tool::Brush,
            brush: BrushParams {
                radius,
                drain: 0.0,
                start_taper_length: start,
                end_taper_length: end,
                ..BrushParams::default()
            },
            path,
            seed: 0,
        }
    }

    fn whole(rec: &StrokeRecord) -> Vec<Segment> {
        generate_segments_in(
            rec,
            super::super::flatten_tolerance(&rec.brush),
            StrokeSpans::whole(rec),
        )
        .0
    }

    /// The profile's two end conditions are the whole design (see [`taper_profile`]),
    /// so they are asserted rather than left to the formula: pinned at both ends,
    /// monotone in between, and *flat* where it meets the stroke's full-width body —
    /// which is what makes the join invisible.
    #[test]
    fn the_taper_profile_is_pinned_flat_at_the_join_and_monotone() {
        assert_eq!(taper_profile(0.0), 0.0, "the tip is a point");
        assert_eq!(taper_profile(1.0), 1.0, "the join is full width");
        assert_eq!(taper_profile(2.0), 1.0, "past the join it stays full width");

        let mut prev = 0.0;
        for i in 1..=200 {
            let f = taper_profile(i as f32 / 200.0);
            assert!(f > prev, "not monotone at t = {}", i as f32 / 200.0);
            prev = f;
        }
        // Numerical slope over the last 1% of the taper, as a multiple of the
        // average: ~0 means the curve arrives flat. (Exactly `f'(1) = 0`.)
        let slope = (taper_profile(1.0) - taper_profile(0.99)) / 0.01;
        assert!(slope < 0.05, "the taper meets the body with slope {slope}");
        // And leaves the tip as a wedge, not a whisker: a `smoothstep`-shaped profile
        // would be under 0.03 here.
        let tip_slope = (taper_profile(0.01) - taper_profile(0.0)) / 0.01;
        assert!(tip_slope > 1.0, "the tip is blunt, slope {tip_slope}");
    }

    /// What the taper does to a stroke: pointed at both ends, full width in between,
    /// and — because a segment sweeps at one radius — stepping finely enough through
    /// the taper that it reads as a point rather than a staircase of dabs.
    #[test]
    fn a_tapered_stroke_narrows_at_both_ends() {
        let radius = 20.0;
        let rec = tapered_record(radius, 4.0, 6.0, 900.0);
        let segs = whole(&rec);
        let first = segs.first().expect("segments");
        let last = segs.last().expect("segments");
        let widest = segs.iter().fold(0.0f32, |m, s| m.max(s.radius));

        assert!(
            first.radius < 0.1 * radius,
            "the start is not a point: {}",
            first.radius
        );
        assert!(
            last.radius < 0.1 * radius,
            "the end is not a point: {}",
            last.radius
        );
        assert!(
            (widest - radius).abs() < 1e-3,
            "the body should reach full radius, got {widest}"
        );
        // Every consecutive pair steps by at most the tolerance the subdivision is
        // sized for (with slack for the pressure/flattening interaction).
        for w in segs.windows(2) {
            let step = (w[1].radius - w[0].radius).abs();
            assert!(
                step <= TAPER_STEP * radius * 2.0,
                "radius jumps {step} between segments — the taper is a staircase"
            );
        }
    }

    /// A stroke shorter than its own two tapers still reaches full width, at one
    /// point in the middle: the zones are scaled down in proportion rather than
    /// clamped, so a quick flick is a small pointed mark and not an invisible sliver.
    /// And the behaviour is continuous in length — the whole reason to compress
    /// rather than clamp.
    #[test]
    fn short_strokes_compress_their_tapers_instead_of_vanishing() {
        let radius = 16.0;
        for len in [4.0f32, 20.0, 60.0, 160.0, 400.0] {
            let rec = tapered_record(radius, 6.0, 6.0, len);
            let widest = whole(&rec).iter().fold(0.0f32, |m, s| m.max(s.radius));
            assert!(
                widest > 0.9 * radius,
                "a {len}px stroke only reached radius {widest} of {radius}"
            );
        }
        // A click has no length for a taper to run along at all, and still dots at
        // full size — the limit of the compression above rather than a special case:
        // zero length scales both zones to nothing, so the profile is 1 at the dab.
        let mut dot = tapered_record(radius, 6.0, 6.0, 0.0);
        dot.path.truncate(1);
        let segs = whole(&dot);
        assert_eq!(segs.len(), 1, "a click is one swept dab");
        assert_eq!(segs[0].radius, radius, "a tapered brush should still dot");
    }

    // --- the touch-down dab ------------------------------------------------

    /// A click leaves a dab **centred on the point that was pressed**, of the travel
    /// [`DAB_TRAVEL`] names.
    ///
    /// The centring is the whole of it. A click has no tangent — the fitter leaves a
    /// lone knot without one — so a dab swept *from* the point goes off in whatever
    /// direction the fallback happens to name, and reads as a short dash rather than a
    /// dot: a full tip's width of travel, all of it on one side, on a mark only two
    /// radii across. Swept about the point, the same travel is a dot a little wider
    /// than it is tall, and the arbitrary direction stops being visible at all.
    #[test]
    fn a_click_dabs_symmetrically_about_the_point() {
        let radius = 20.0;
        let at = Vec2::new(37.0, -11.0);
        let rec = record(
            BrushParams {
                radius,
                ..BrushParams::default()
            },
            &[at],
        );
        let segs = whole(&rec);
        assert_eq!(segs.len(), 1, "a click is one swept dab");
        let dab = segs[0];
        assert_eq!(dab.curvature, 0.0, "a dab does not bend");
        assert!(
            (dab.length - DAB_TRAVEL * radius).abs() < 1e-4,
            "the dab swept {} of the {} it owes",
            dab.length,
            DAB_TRAVEL * radius
        );
        let centre = dab.start + dab.dir * (dab.length * 0.5);
        assert!(
            (centre - at).length() < 1e-4,
            "the dab is centred at {centre:?}, not on the {at:?} that was pressed"
        );
    }

    /// The dab and the stroke are one continuum, not two cases: a stroke sweeps
    /// `max(travel, dab)`, so the mark grows out of the dot instead of replacing it.
    ///
    /// The jump this rules out was the visible one. A press deposited a dab; the first
    /// pixel of movement made the stroke "long enough" to stand on its own and
    /// deposited a twentieth of one, so the dot vanished the instant the hand moved
    /// and came back only once the stroke had travelled a tip's width.
    #[test]
    fn a_short_stroke_is_topped_up_to_a_whole_dab() {
        let radius = 20.0;
        let dab = DAB_TRAVEL * radius;
        for len in [0.0f32, 0.5, 2.0, 6.0, dab - 0.5, dab + 0.5, 40.0] {
            let rec = tapered_record(radius, 0.0, 0.0, len);
            let segs = whole(&rec);
            let travel: f32 = segs.iter().map(|s| s.length).sum();
            assert!(
                (travel - len.max(dab)).abs() < 0.05,
                "a {len}px stroke swept {travel}, not the {} it owes",
                len.max(dab)
            );
            // And the dwell is swept about the stroke's own midpoint, so it can only
            // fatten the mark symmetrically — never lead it off in one direction.
            if len < dab {
                let d = segs[0];
                let centre = d.start + d.dir * (d.length * 0.5);
                assert!(
                    (centre - Vec2::new(len * 0.5, 0.0)).length() < 0.05,
                    "a {len}px stroke's dab sits at {centre:?}, not on its midpoint"
                );
            }
        }
    }

    /// Nothing may freeze while the dab is still in play, and this is what says so.
    ///
    /// The dab is measured against the *whole* stroke's travel, exactly as the
    /// trailing taper is against its length, so a span frozen before the stroke has
    /// outrun its dab would keep a dab the commit does not draw — live == committed
    /// (§1.3) failing where it cannot be repainted. Held back until the frozen prefix
    /// alone is a dab long, which proves the whole stroke is.
    #[test]
    fn nothing_freezes_until_the_stroke_has_outrun_its_dab() {
        let radius = 60.0; // dab = 36px, so a stroke can be many spans and still owe one
        let untapered = |len: f32| {
            let rec = tapered_record(radius, 0.0, 0.0, len);
            let all = crate::path::span_count(rec.path.len());
            (super::super::safe_frozen(&rec, all), rec)
        };
        let (frozen, _) = untapered(20.0);
        assert_eq!(frozen, 0, "a stroke inside its own dab froze a span");
        let (frozen, rec) = untapered(600.0);
        assert!(frozen > 0, "a long stroke never froze anything");

        // And what it admits really is dab-free: the head it hands over renders the
        // same segments the commit does, which is the property the whole rule is for.
        let tol = super::super::flatten_tolerance(&rec.brush);
        let all = crate::path::span_count(rec.path.len());
        let (head, dist) = generate_segments_in(
            &rec,
            tol,
            StrokeSpans {
                range: 0..frozen,
                dist: 0.0,
            },
        );
        let (tail, _) = generate_segments_in(
            &rec,
            tol,
            StrokeSpans {
                range: frozen..all,
                dist,
            },
        );
        let one_pass = whole(&rec);
        assert_eq!(
            head.len() + tail.len(),
            one_pass.len(),
            "the split re-cut it"
        );
        for (i, (a, b)) in head.iter().chain(&tail).zip(&one_pass).enumerate() {
            assert_eq!(a.start, b.start, "segment {i}: start differs");
            assert_eq!(a.length, b.length, "segment {i}: length differs");
        }
    }

    /// The load-bearing claim behind [`super::safe_frozen`]: for any prefix it
    /// admits, rendering the stroke as *head + tail* produces the very same swept
    /// segments as rendering it in one pass.
    ///
    /// That is what the live == committed invariant (§1.3) reduces to here. A frozen
    /// head is never redrawn, so if the head's segments differed from the commit's by
    /// even a radius the stroke would visibly change under the pointer at release —
    /// and the taper is exactly the kind of parameter that invites it, being measured
    /// from an end of the stroke that has not been drawn yet.
    #[test]
    fn a_taper_safe_head_plus_tail_is_the_single_pass_stroke() {
        let rec = tapered_record(18.0, 5.0, 9.0, 1200.0);
        let tol = super::super::flatten_tolerance(&rec.brush);
        let all = crate::path::span_count(rec.path.len());
        let frozen = super::super::safe_frozen(&rec, all);
        assert!(frozen > 0, "nothing could be frozen at all");
        assert!(
            frozen < all,
            "the trailing taper must hold the last spans back"
        );

        let (head, dist) = generate_segments_in(
            &rec,
            tol,
            StrokeSpans {
                range: 0..frozen,
                dist: 0.0,
            },
        );
        let (tail, _) = generate_segments_in(
            &rec,
            tol,
            StrokeSpans {
                range: frozen..all,
                dist,
            },
        );
        let split: Vec<Segment> = head.into_iter().chain(tail).collect();
        let one_pass = whole(&rec);

        assert_eq!(
            split.len(),
            one_pass.len(),
            "the split stroke has a different number of segments"
        );
        for (i, (a, b)) in split.iter().zip(&one_pass).enumerate() {
            assert_eq!(a.radius, b.radius, "segment {i}: radius differs (taper)");
            assert_eq!(a.dist, b.dist, "segment {i}: arc length differs");
            assert_eq!(a.length, b.length, "segment {i}: length differs");
            assert_eq!(a.start, b.start, "segment {i}: start differs");
        }
    }

    /// An untapered brush is untouched by any of the above, to the bit: the taper's
    /// subdivision has to be a no-op where the taper is flat, or it would re-cut
    /// every stroke ever drawn and invalidate every golden.
    #[test]
    fn an_untapered_stroke_is_not_subdivided() {
        let rec = tapered_record(18.0, 0.0, 0.0, 900.0);
        let segs = whole(&rec);
        let pts = crate::path::flatten_spans(
            &rec.path,
            0..crate::path::span_count(rec.path.len()),
            0.0,
            super::super::flatten_tolerance(&rec.brush),
        );
        assert_eq!(
            segs.len(),
            pts.len() - 1,
            "one segment per flattened edge, with no taper subdivision"
        );
        assert!(
            segs.iter().all(|s| s.radius == 18.0),
            "an untapered stroke is full width throughout"
        );
        assert_eq!(
            super::super::safe_frozen(&rec, 7),
            7,
            "an untapered brush holds nothing back from freezing"
        );
    }

    // --- swept arcs -------------------------------------------------------

    /// A stroke bending through `sweep` radians of a circle of radius `curve_radius`.
    fn curved_record(radius: f32, curve_radius: f32, sweep: f32) -> StrokeRecord {
        let path: Vec<crate::path::ControlPoint> = (0..=12)
            .map(|i| {
                let t = i as f32 / 12.0 * sweep;
                crate::path::ControlPoint::at(Vec2::new(
                    curve_radius * t.sin(),
                    curve_radius * (1.0 - t.cos()),
                ))
            })
            .collect();
        StrokeRecord {
            layer: crate::document::LayerId(0),
            tool: crate::document::Tool::Brush,
            brush: BrushParams {
                radius,
                drain: 0.0,
                ..BrushParams::default()
            },
            path,
            seed: 0,
        }
    }

    /// Densely sampled points of the true curve — the ground truth the two stand-ins
    /// below are measured against. Fifty times tighter than the render budget, so its
    /// own flattening error is nowhere near what is being compared.
    fn dense(rec: &StrokeRecord) -> Vec<Vec2> {
        let tol = super::super::flatten_tolerance(&rec.brush).relaxed(0.02);
        crate::path::flatten(&rec.path, tol)
            .into_iter()
            .map(|s| s.pos)
            .collect()
    }

    /// The largest distance from any point of the true curve to `poly`.
    fn deviation(curve: &[Vec2], poly: &[Vec2]) -> f32 {
        let to_seg = |p: Vec2, a: Vec2, b: Vec2| {
            let ab = b - a;
            let len2 = ab.length_squared();
            let t = if len2 < 1e-12 {
                0.0
            } else {
                ((p - a).dot(ab) / len2).clamp(0.0, 1.0)
            };
            (p - (a + ab * t)).length()
        };
        curve
            .iter()
            .map(|&p| {
                poly.windows(2)
                    .map(|w| to_seg(p, w[0], w[1]))
                    .fold(f32::INFINITY, f32::min)
            })
            .fold(0.0, f32::max)
    }

    /// `fit_arc` on a genuine circular arc recovers the circle it came from — the
    /// case the whole construction is pinned to, since a flattened edge of a smooth
    /// stroke is one to second order.
    #[test]
    fn the_arc_fit_recovers_a_circle() {
        for (r, theta) in [(50.0f64, 0.08f64), (300.0, 0.05), (900.0, 0.02)] {
            // An arc of radius `r` turning `theta`, starting at the origin heading +x.
            // Built in f64: the far endpoint's lateral offset is `r(1 − cos θ)`, and
            // forming that in f32 would cancel away enough digits to swamp the
            // tolerances below — an artifact of the test's own construction, not of
            // the fit it is checking.
            let start_dir = Vec2::new(1.0, 0.0);
            let end = Vec2::new((r * theta.sin()) as f32, (r * (1.0 - theta.cos())) as f32);
            let (r, theta) = (r as f32, theta as f32);
            let crate::path::Arc {
                dir,
                curvature: kappa,
                length: len,
            } = crate::path::fit_arc(start_dir, end, f32::INFINITY);
            assert!(kappa != 0.0, "r={r} θ={theta}: fitted straight");
            assert!(
                (kappa - 1.0 / r).abs() < 1e-4 / r,
                "r={r}: curvature {kappa} is not 1/{r}"
            );
            assert!(
                (len - r * theta).abs() < 1e-3 * len,
                "r={r}: arc length {len} is not {}",
                r * theta
            );
            // And walking it lands exactly on the far end, which is what makes
            // consecutive segments meet.
            let (landed, _) = crate::path::arc_at(Vec2::ZERO, dir, kappa, len);
            assert!(
                (landed - end).length() < 1e-3,
                "r={r}: the arc ends at {landed:?}, not {end:?}"
            );
        }
    }

    /// The claim the change exists for: swept **arcs** track the fitted curve far
    /// more closely than the chords they replace, at the same segment count.
    ///
    /// The chord's error is the flattener's positional budget by construction — that
    /// is what the budget *is* — so this is really a statement about what a segment
    /// can be asked to do without being made shorter. Measured on the curves below,
    /// the arcs land ~4× closer; the residual is the fitted spline's own curvature
    /// *variation* across a segment, which a single arc cannot follow and which the
    /// flattener's `angle` bound is what actually limits.
    ///
    /// The ratio understates the visible gain, because the amplitude is not what the
    /// eye is picking up: a chord sweep breaks the outline's curvature at every joint
    /// and creases it on the inside of a turn, and an arc sweep does neither. That is
    /// what facets are, and it is not something a distance metric sees.
    #[test]
    fn arcs_track_the_curve_far_closer_than_chords() {
        for curve_radius in [200.0f32, 600.0, 2000.0] {
            let rec = curved_record(8.0, curve_radius, 1.2);
            let segs = whole(&rec);
            let curve = dense(&rec);
            assert!(
                segs.iter().any(|s| s.curvature != 0.0),
                "r={curve_radius}: nothing was bent at all"
            );

            let chords: Vec<Vec2> = segs
                .iter()
                .map(|s| s.start)
                .chain(segs.last().map(segment_end))
                .collect();
            // Each arc sampled finely, so a point-to-polyline distance measures the
            // arc itself rather than its own chord.
            let mut arcs = Vec::new();
            for s in &segs {
                for i in 0..16 {
                    arcs.push(
                        crate::path::arc_at(
                            s.start,
                            s.dir,
                            s.curvature,
                            s.length * i as f32 / 16.0,
                        )
                        .0,
                    );
                }
            }
            arcs.extend(segs.last().map(segment_end));

            let chord_err = deviation(&curve, &chords);
            let arc_err = deviation(&curve, &arcs);
            assert!(
                arc_err < 0.35 * chord_err,
                "r={curve_radius}: arcs are off by {arc_err}, chords by {chord_err}"
            );
        }
    }

    /// Consecutive segments meet: each one's arc *ends* where the next one starts.
    /// Nothing in the deposit re-derives a segment's end — the shaders sweep from
    /// `start` along the arc for `length` — so a gap here would be a seam of missing
    /// paint at every joint, and an overlap a double deposit.
    #[test]
    fn segments_meet_end_to_start_along_their_arcs() {
        let rec = curved_record(12.0, 400.0, 1.5);
        let segs = whole(&rec);
        assert!(segs.len() > 4, "not enough segments to join up");
        for (i, w) in segs.windows(2).enumerate() {
            let gap = (segment_end(&w[0]) - w[1].start).length();
            assert!(
                gap < 1e-2,
                "segment {i} ends {gap}px from where {} starts",
                i + 1
            );
        }
    }

    /// A straight stroke is not bent, and the arc machinery leaves it on exactly the
    /// floats it was on before — the same no-op guarantee the taper's subdivision has.
    #[test]
    fn a_straight_stroke_is_never_bent() {
        let rec = tapered_record(18.0, 0.0, 0.0, 900.0);
        let segs = whole(&rec);
        assert!(
            segs.iter().all(|s| s.curvature == 0.0),
            "a straight stroke picked up curvature"
        );
        for s in &segs {
            assert_eq!(
                segment_end(s),
                s.start + s.dir * s.length,
                "a straight segment's end moved off its chord"
            );
        }
    }

    /// A tip too fat for the turn it is sweeping falls back to a straight segment.
    ///
    /// The bound exists because both shaders sweep a curved segment by unrolling the
    /// annulus about its centre of curvature, and that approximation degrades as the
    /// tip grows against the curve's own radius — see
    /// [`MAX_TIP_TURN`](super::super::MAX_TIP_TURN).
    #[test]
    fn a_fat_tip_on_a_tight_turn_sweeps_straight() {
        let curve_radius = 60.0;
        let fat = 50.0;
        for s in whole(&curved_record(fat, curve_radius, 1.5)) {
            assert!(
                s.radius * s.curvature.abs() <= super::super::MAX_TIP_TURN,
                "a segment sweeps an arc of radius {} under a {} tip",
                1.0 / s.curvature.abs(),
                s.radius
            );
        }
        // And the curve really is tight enough for that to have bitten: under a fine
        // tip the same path keeps curvature the fat one had to give up. Without this
        // the assertion above would pass on any straight line.
        let fine = whole(&curved_record(2.0, curve_radius, 1.5));
        assert!(
            fine.iter()
                .any(|s| fat * s.curvature.abs() > super::super::MAX_TIP_TURN),
            "the test curve is too gentle to exercise the guard"
        );
    }

    /// The flattener and the segment generator agree, edge for edge, on whether a
    /// piece of curve is swept as an arc or as a chord.
    ///
    /// This is what makes the positional budget mean anything. `path::within` prices
    /// an edge against whatever `fit_arc` returns for it, and the sweep is built from
    /// whatever `fit_arc` returns for it — so if the two ever called it with different
    /// caps, an edge could be *measured* as a well-tracked arc and then *drawn* as a
    /// chord that misses the curve by several times the allowance. Routing both through
    /// one function with one cap is what rules that out; this pins that they do.
    #[test]
    fn the_flattener_and_the_sweep_agree_on_which_edges_bend() {
        for radius in [2.0f32, 18.0, 50.0, 120.0] {
            for curve_radius in [80.0f32, 300.0, 1200.0, 5000.0] {
                let rec = curved_record(radius, curve_radius, 1.4);
                let tol = super::super::flatten_tolerance(&rec.brush);
                let pts = crate::path::flatten(&rec.path, tol);
                let segs = whole(&rec);
                // Untapered, so it is one segment per flattened edge — except the
                // degenerate ones the generator drops, which the clamped end condition
                // always produces a few of (its outermost spans are squashed to nearly
                // nothing).
                let edges: Vec<_> = pts
                    .windows(2)
                    .filter(|w| (w[1].pos - w[0].pos).length() >= 1e-5)
                    .collect();
                assert_eq!(segs.len(), edges.len(), "r={radius} R={curve_radius}");
                for (i, (w, s)) in edges.iter().zip(&segs).enumerate() {
                    let want =
                        crate::path::fit_arc(w[0].vel, w[1].pos - w[0].pos, tol.max_arc_curvature);
                    assert_eq!(
                        want.curvature, s.curvature,
                        "r={radius} R={curve_radius} edge {i}: the flattener priced \
                         curvature {} and the sweep drew {}",
                        want.curvature, s.curvature
                    );
                }
                // And the cap really is enforced, not merely never reached.
                assert!(
                    segs.iter()
                        .all(|s| s.curvature.abs() <= tol.max_arc_curvature + 1e-9),
                    "r={radius} R={curve_radius}: a segment bends past the sweepable cap"
                );
            }
        }
    }

    /// Every box a segment is measured by contains its whole arc, not just its two
    /// ends. Under-reporting here is a clipped stroke: `affected_tiles` would leave a
    /// tile out of the render, and the dynamics loop would dispatch a rect too small
    /// for its own footprint.
    #[test]
    fn the_coverage_box_contains_the_whole_arc() {
        let rec = curved_record(10.0, 150.0, 2.4);
        let segs = whole(&rec);
        assert!(segs.iter().any(|s| s.curvature != 0.0), "nothing bent");
        for (i, s) in segs.iter().enumerate() {
            let (lo, hi) = coverage_bounds(s);
            for k in 0..=32 {
                let (p, _) =
                    crate::path::arc_at(s.start, s.dir, s.curvature, s.length * k as f32 / 32.0);
                // Every point of the arc, plus the tip riding along it.
                let r = Vec2::splat(s.radius);
                assert!(
                    (p - r).x >= lo.x
                        && (p - r).y >= lo.y
                        && (p + r).x <= hi.x
                        && (p + r).y <= hi.y,
                    "segment {i}: the arc escapes its own coverage box at {p:?}"
                );
            }
        }
    }

    // --- segment budget ----------------------------------------------------

    /// A stroke through `pts` with `brush`, as a path of plain full-pressure knots.
    fn record(brush: BrushParams, pts: &[Vec2]) -> StrokeRecord {
        StrokeRecord {
            layer: crate::document::LayerId(0),
            tool: crate::document::Tool::Brush,
            brush,
            path: pts
                .iter()
                .map(|p| crate::path::ControlPoint::at(*p))
                .collect(),
            seed: 0,
        }
    }

    /// `sin` and `cos` from their Maclaurin series in plain f64 arithmetic.
    ///
    /// Not for accuracy — the curves below only have to be representative shapes. The
    /// library versions are not specified to the last bit and may differ between
    /// platforms, and these decide *control points*, so a knot differing by an ulp
    /// could flip a subdivision decision and fail this test on someone else's machine.
    /// Basic IEEE arithmetic is exactly specified, which rules that out rather than
    /// hoping — the same argument that makes [`taper_profile`] a polynomial (§12.1).
    fn sin_series(x: f64) -> f64 {
        let (x2, mut term, mut acc) = (x * x, x, x);
        for k in 1..10 {
            term *= -x2 / (((2 * k) * (2 * k + 1)) as f64);
            acc += term;
        }
        acc
    }

    fn cos_series(x: f64) -> f64 {
        let (x2, mut term, mut acc) = (x * x, 1.0, 1.0);
        for k in 1..10 {
            term *= -x2 / (((2 * k - 1) * (2 * k)) as f64);
            acc += term;
        }
        acc
    }

    /// `n + 1` knots along a curve given by its **heading** — the tangent angle as a
    /// function of arc length, stepped into positions. Curvature is that function's
    /// derivative, which is what lets the curved cases below state their curvature
    /// directly instead of implying it through a parameterization.
    fn by_heading(n: usize, length: f64, theta: impl Fn(f64) -> f64) -> Vec<Vec2> {
        const STEPS_PER_KNOT: usize = 16;
        let ds = length / (n * STEPS_PER_KNOT) as f64;
        let (mut x, mut y) = (0.0f64, 0.0f64);
        let mut pts = vec![Vec2::new(0.0, 0.0)];
        for i in 0..n * STEPS_PER_KNOT {
            let t = theta((i as f64 + 0.5) * ds); // midpoint: symmetric about an inflection
            x += cos_series(t) * ds;
            y += sin_series(t) * ds;
            if (i + 1) % STEPS_PER_KNOT == 0 {
                pts.push(Vec2::new(x as f32, y as f32));
            }
        }
        pts
    }

    /// A brush that manipulates paint, so the stroke takes the dynamics loop.
    fn smearing(radius: f32) -> BrushParams {
        use crate::document::BrushDynamics;
        BrushParams {
            radius,
            dynamics: BrushDynamics {
                lift: 0.8,
                deposit: 0.8,
                ..BrushDynamics::default()
            },
            ..BrushParams::default()
        }
    }

    /// **How many segments the flattener spends on a stroke — pinned, on purpose.**
    ///
    /// This is a change-detector test and it is meant to be one. **Updating these
    /// numbers is a normal thing to do:** if a change moves them and you have decided
    /// the new geometry is right, paste in the new counts and say why in the commit.
    /// The test is not asserting that any particular number is correct — it is making
    /// sure a number cannot move *silently*, because nothing else here would notice.
    ///
    /// Segment count is the loop's unit of cost. Every dispatch in the dynamics path
    /// is charged per segment (`dynamics.wesl`), so the budgets below are the dial
    /// between quality and time, and they are set from five different quantities that
    /// have nothing to do with one another. A change to any one of them moves a stroke
    /// nobody was thinking about: the cases are chosen so that each is dominated by a
    /// *different* budget, and the one that moves tells you which.
    ///
    /// Every count is reported in one pass rather than failing at the first, so a
    /// deliberate retuning gives you the whole new table to paste in from one run.
    ///
    /// These are CPU-side and float-deterministic (§12.1) — the same reason replay and
    /// peers agree on geometry — so a count that differs *per machine* is a bug in that
    /// determinism, not a tolerance to loosen.
    /// The exchange budget means the same thing to every brush
    /// (`super::super::flatten_tolerance`). These are properties of the rule, not
    /// measured counts — unlike the table below, a failure here is a bug rather than a
    /// retuning.
    #[test]
    fn the_exchange_budget_scales_with_the_transfer_rate() {
        use crate::document::BrushDynamics;
        let at = |lift: f32, deposit: f32, charge: f32| {
            super::super::flatten_tolerance(&BrushParams {
                radius: 100.0,
                dynamics: BrushDynamics {
                    lift,
                    deposit,
                    charge,
                    ..BrushDynamics::default()
                },
                ..BrushParams::default()
            })
            .max_len
        };

        // The calibration point: `lift = deposit = 0.95` is quoted at exactly
        // `RESERVOIR_EXCHANGE_STEP`, which is what leaves the goldens that use it alone.
        assert!(
            (at(0.95, 0.95, 0.0) - 12.5).abs() < 0.05,
            "calibration moved: {}",
            at(0.95, 0.95, 0.0)
        );

        // Halving the rate doubles the travel: `−ln((1−a)(1−b))` is what the step is
        // inversely proportional to, so squaring the retained fractions halves it.
        // (0.95 → 0.9975 has half the rate of 0.95 → 0.95 per axis.)
        let slow = at(0.7775, 0.7775, 0.0); // (1−a)² = 0.05 ⇒ half the rate of 0.95
        assert!(
            (slow - 25.0).abs() < 0.2,
            "the step is not inverse in the rate: {slow}"
        );

        // Monotone in each axis on its own — more trading, shorter segments.
        assert!(at(0.9, 0.0, 0.0) < at(0.5, 0.0, 0.0));
        assert!(at(0.0, 0.9, 0.0) < at(0.0, 0.5, 0.0));

        // `charge` is a starting load, not a rate. A brush that only charges never
        // enters the exchange at all (`exchange_at`'s no-trading branch), so it is
        // bounded by the structural ceiling and nothing else.
        assert_eq!(at(0.0, 0.0, 1.0), 100.0);
        // …and it does not tighten a brush that *does* trade.
        assert!((at(0.95, 0.95, 1.0) - at(0.95, 0.95, 0.0)).abs() < f32::EPSILON);

        // A brush with no dynamics at all is not capped by this at all.
        assert!(at(0.0, 0.0, 0.0) > 100.0);

        // Never a tightening: a brush that trades *faster* than the reference is left at
        // the reference step, so no setting pays more than it did before the scaling.
        assert!((at(1.0, 1.0, 0.0) - at(0.95, 0.95, 0.0)).abs() < 0.05);
        assert!(at(0.99, 0.99, 0.0) >= at(0.95, 0.95, 0.0) - 0.05);
    }

    #[test]
    fn the_segment_budget_is_what_it_was() {
        // Three curves, shared across brushes so that a difference between two rows on
        // the same path is the brush's doing and nothing else. Each is 400px of arc.
        //
        // The tip radii below are 20 and 80, so `max_arc_curvature` (MAX_TIP_TURN /
        // radius) sits at 0.005 and 0.00125 respectively — the curvatures are picked
        // around those two thresholds.
        let straight = vec![Vec2::new(0.0, 0.0), Vec2::new(400.0, 0.0)];
        // Constant curvature 0.004: inside what a radius-20 tip may sweep as an arc,
        // outside what a radius-80 tip may, so the same curve is priced both ways.
        let arc = by_heading(24, 400.0, |s| 0.004 * s);
        // An Euler spiral **through its inflection**: curvature linear in arc length,
        // running −0.006 → +0.006 with the zero at the middle. It is the one shape that
        // exercises the whole of `fit_arc` in a single stroke — a sign change, the
        // degenerate straight case exactly at the inflection, and the
        // `max_arc_curvature` threshold crossed once on each side (at |κ| = 0.005 for a
        // radius-20 tip), so the fitter alternates between arcs and chords along it.
        // Heading is the integral of curvature: ∫(a·s + b) ds with the constant chosen
        // to put the inflection at the halfway point.
        let spiral = by_heading(24, 400.0, |s| 0.5 * 0.00003 * s * s - 0.006 * s);

        let cases: &[(&str, usize, StrokeRecord)] = &[
            // `position` and `angle` alone: a straight line satisfies both everywhere,
            // so this is the floor — one segment per flattened span, and the number to
            // compare every other row against.
            (
                "straight, plain tip",
                3,
                record(
                    BrushParams {
                        radius: 20.0,
                        ..BrushParams::default()
                    },
                    &straight,
                ),
            ),
            // `max_len` from the exchange budget. `smearing()` trades at `lift = deposit = 0.8`,
            // which the rate scaling prices at 0.233 · radius = 4.7px over 400px — not the
            // 2.5px the 0.95 calibration point would cost.
            // **This is the row a reservoir-cadence retuning moves**, and the reason
            // the dynamics path costs what it does. Subdivision is by bisection, so a
            // count sits at or above the length bound's own `400/4.7 = 86` rather than
            // exactly on it.
            (
                "straight, smearing tip",
                118,
                record(smearing(20.0), &straight),
            ),
            // The same cadence on a tip four times as fat. The cap is a fraction of the
            // radius, so this row and the one above stand in the radius ratio — which
            // is what identifies the cadence, rather than something else, as what sets
            // them both.
            (
                "straight, fat smearing tip",
                30,
                record(smearing(80.0), &straight),
            ),
            // `drain` costs **nothing**, which is the point of this row: the falloff is
            // evaluated per fragment from its own arc length, so it asks the flattener
            // for no segments at all and this comes out identical to the smearing row
            // above. It used to bind at `0.02 / drain` = 4px, and now costs nothing at all —
            // for a quantity that is exact rather than merely finely sampled.
            (
                "straight, draining tip",
                118,
                record(
                    BrushParams {
                        drain: 0.005,
                        ..smearing(20.0)
                    },
                    &straight,
                ),
            ),
            // TAPER_STEP, and by a wide margin the most expensive row in the table: a
            // taper costs ~`TAPER_MAX_SLOPE / TAPER_STEP` ≈ 75 pieces however long it
            // is, and this brush has two of them. Nothing about the curve is driving
            // this one — it is the same straight line as the 3-segment row above.
            (
                "straight, tapered tip",
                211,
                record(
                    BrushParams {
                        radius: 20.0,
                        start_taper_length: 2.0,
                        end_taper_length: 3.0,
                        ..BrushParams::default()
                    },
                    &straight,
                ),
            ),
            // `angle` (0.1 rad): 0.004 × 400 = 1.6 radians of turning, so ≥ 16 segments
            // however large the curve is drawn.
            (
                "arc, plain tip",
                31,
                record(
                    BrushParams {
                        radius: 20.0,
                        ..BrushParams::default()
                    },
                    &arc,
                ),
            ),
            // The same curve under a tip too fat to sweep it as an arc, so `fit_arc`
            // hands back chords instead. **It costs exactly the same**, and that is the
            // point of keeping both rows: at this curvature `angle` binds first, so the
            // arc/chord choice changes what a segment *is* without changing how many
            // there are. If a change to `MAX_TIP_TURN` or to how a too-tight edge is
            // priced ever makes these two diverge, that is worth knowing about.
            (
                "arc, fat tip",
                31,
                record(
                    BrushParams {
                        radius: 80.0,
                        ..BrushParams::default()
                    },
                    &arc,
                ),
            ),
            ("arc, smearing tip", 103, record(smearing(20.0), &arc)),
            // The Euler spiral: `angle` again over 1.2 radians of total turning, but
            // with the fitter crossing the arc/chord threshold on each side of a
            // genuine inflection. Cheaper than the arc because it turns one way and
            // then back, rather than accumulating.
            (
                "euler spiral, plain tip",
                26,
                record(
                    BrushParams {
                        radius: 20.0,
                        ..BrushParams::default()
                    },
                    &spiral,
                ),
            ),
            (
                "euler spiral, fat tip",
                26,
                record(
                    BrushParams {
                        radius: 80.0,
                        ..BrushParams::default()
                    },
                    &spiral,
                ),
            ),
            // Back to `max_len`: the cadence asks for more than the spiral's own shape
            // does, so a smearing tip pays the same price on a curve as on a line.
            (
                "euler spiral, smearing tip",
                98,
                record(smearing(20.0), &spiral),
            ),
        ];

        let mut moved = Vec::new();
        for (name, expected, rec) in cases {
            let got = whole(rec).len();
            if got != *expected {
                moved.push(format!("  {name}: {expected} -> {got}"));
            }
        }
        assert!(
            moved.is_empty(),
            "the segment budget moved (update the counts if this was deliberate):\n{}",
            moved.join("\n")
        );
    }

    // --- region measurement ----------------------------------------------

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
            curvature: 0.0,
            radius,
            length,
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
