//! Stroke path fitting and interpolation (DESIGN.md §6.2).
//!
//! A stroke is captured as raw pointer samples, then stored as a compact set of
//! **control points** via Ramer–Douglas–Peucker simplification ([`simplify`]).
//! At render time those control points are expanded through a **centripetal
//! Catmull–Rom spline** into a fine polyline ([`flatten`]), which the stamp
//! generator walks at even arc length. This removes stair-step aliasing, makes
//! discrete stamps read as a continuous stroke, and shrinks the save file.
//!
//! All math here is deterministic, preserving golden / replay / save-load
//! equivalence.

use crate::command::InputSample;
use crate::geom::Vec2;

/// Default RDP tolerance in canvas pixels. A 1-px-stepped diagonal sits ~0.7px
/// off its ideal line, so ~1px clears that staircase while keeping any feature
/// larger than a pixel — imperceptible smoothing at normal brush sizes.
pub const SIMPLIFY_TOLERANCE: f32 = 1.0;

/// Default Catmull–Rom flattening step in canvas pixels.
pub const FLATTEN_STEP: f32 = 2.0;

/// Simplify raw samples to control points via Ramer–Douglas–Peucker, keeping the
/// endpoints. Each kept control point is an original sample (so it retains that
/// sample's pressure/tilt/time).
pub fn simplify(samples: &[InputSample], tolerance: f32) -> Vec<InputSample> {
    let n = samples.len();
    if n <= 2 {
        return samples.to_vec();
    }
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;
    rdp(samples, 0, n - 1, tolerance, &mut keep);
    samples
        .iter()
        .zip(keep)
        .filter_map(|(s, k)| k.then_some(*s))
        .collect()
}

fn rdp(samples: &[InputSample], first: usize, last: usize, tol: f32, keep: &mut [bool]) {
    if last <= first + 1 {
        return;
    }
    let a = samples[first].pos;
    let b = samples[last].pos;
    let mut max_d = 0.0;
    let mut idx = first;
    for (i, s) in samples.iter().enumerate().take(last).skip(first + 1) {
        let d = point_segment_distance(s.pos, a, b);
        if d > max_d {
            max_d = d;
            idx = i;
        }
    }
    if max_d > tol {
        keep[idx] = true;
        rdp(samples, first, idx, tol, keep);
        rdp(samples, idx, last, tol, keep);
    }
}

fn point_segment_distance(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let len2 = ab.length_squared();
    let t = if len2 < 1e-12 {
        0.0
    } else {
        ((p - a).dot(ab) / len2).clamp(0.0, 1.0)
    };
    (p - (a + ab * t)).length()
}

/// Expand control points through a centripetal Catmull–Rom spline into a fine
/// polyline whose points are spaced ~`step` canvas pixels apart **by arc length**
/// (not by the curve parameter), so the stamp generator and the wet-mixing
/// reservoir are parameterized by distance travelled (DESIGN.md §6.2). Position is
/// interpolated cubically; pressure/tilt/time linearly. Endpoints are preserved.
pub fn flatten(knots: &[InputSample], step: f32) -> Vec<InputSample> {
    match knots.len() {
        0 => return Vec::new(),
        1 => return vec![knots[0]],
        _ => {}
    }
    let step = step.max(0.5);
    // Sample the spline densely in its own (centripetal) parameter — fine enough
    // (¼ `step`) that the piecewise-linear result closely tracks the true curve, so
    // the arc-length pass below measures real distance — then resample by distance.
    let dense = dense_spline(knots, step * 0.25);
    resample_by_arc_length(&dense, step)
}

/// Densely subdivide each control span in the centripetal curve parameter (a
/// chord-proportional point count), giving a polyline that closely tracks the
/// curve. The spacing is *not* uniform in distance — [`resample_by_arc_length`]
/// fixes that.
fn dense_spline(knots: &[InputSample], step: f32) -> Vec<InputSample> {
    let n = knots.len();
    let mut out = Vec::with_capacity(n * 8);
    out.push(knots[0]);
    for i in 0..n - 1 {
        let p1 = knots[i];
        let p2 = knots[i + 1];
        let p0 = if i == 0 { knots[0] } else { knots[i - 1] };
        let p3 = if i + 2 < n { knots[i + 2] } else { knots[n - 1] };

        let chord = (p2.pos - p1.pos).length();
        let steps = ((chord / step).ceil() as usize).clamp(1, 512);
        for s in 1..=steps {
            let u = s as f32 / steps as f32;
            out.push(InputSample {
                pos: catmull_rom(p0.pos, p1.pos, p2.pos, p3.pos, u),
                pressure: p1.pressure + (p2.pressure - p1.pressure) * u,
                tilt: p1.tilt.lerp(p2.tilt, u),
                time: p1.time + (p2.time - p1.time) * u as f64,
            });
        }
    }
    out
}

/// Walk a fine polyline and emit a sample every `step` of arc length (both
/// endpoints included), linearly interpolating attributes between dense points.
/// The result is evenly spaced by distance regardless of the curve's local speed;
/// the final span is whatever remainder is left (≤ `step`).
fn resample_by_arc_length(dense: &[InputSample], step: f32) -> Vec<InputSample> {
    let mut out = vec![dense[0]];
    if dense.len() < 2 {
        return out;
    }
    let mut acc = 0.0f32; // arc length accumulated since the last emitted sample
    let mut prev = dense[0];
    for &cur in &dense[1..] {
        let seg = (cur.pos - prev.pos).length();
        if seg >= 1e-9 {
            let mut t = 0.0f32; // distance consumed within [prev, cur]
            while acc + (seg - t) >= step {
                t += step - acc;
                out.push(lerp_sample(prev, cur, (t / seg).clamp(0.0, 1.0)));
                acc = 0.0;
            }
            acc += seg - t;
        }
        prev = cur;
    }
    // Finish exactly on the stroke's end (unless we just emitted it).
    let last = *dense.last().unwrap();
    if (out.last().unwrap().pos - last.pos).length_squared() > 1e-6 {
        out.push(last);
    }
    out
}

/// Linear blend of two samples (position, pressure, tilt, time) at `f ∈ [0,1]`.
fn lerp_sample(a: InputSample, b: InputSample, f: f32) -> InputSample {
    InputSample {
        pos: a.pos.lerp(b.pos, f),
        pressure: a.pressure + (b.pressure - a.pressure) * f,
        tilt: a.tilt.lerp(b.tilt, f),
        time: a.time + (b.time - a.time) * f as f64,
    }
}

/// Centripetal Catmull–Rom (Barry–Goldman form) evaluated at `u ∈ [0, 1]` along
/// the segment `p1 → p2`, with neighbors `p0`, `p3`.
fn catmull_rom(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, u: f32) -> Vec2 {
    const ALPHA: f32 = 0.5;
    let dt = |a: Vec2, b: Vec2| (b - a).length().powf(ALPHA).max(1e-4);
    let t0 = 0.0;
    let t1 = t0 + dt(p0, p1);
    let t2 = t1 + dt(p1, p2);
    let t3 = t2 + dt(p2, p3);

    let t = t1 + (t2 - t1) * u;
    let a1 = p0.lerp(p1, (t - t0) / (t1 - t0));
    let a2 = p1.lerp(p2, (t - t1) / (t2 - t1));
    let a3 = p2.lerp(p3, (t - t2) / (t3 - t2));
    let b1 = a1.lerp(a2, (t - t0) / (t2 - t0));
    let b2 = a2.lerp(a3, (t - t1) / (t3 - t1));
    b1.lerp(b2, (t - t1) / (t2 - t1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(x: f32, y: f32) -> InputSample {
        InputSample::at(Vec2::new(x, y))
    }

    #[test]
    fn simplify_collapses_pixel_staircase() {
        // A diagonal drawn as 1-px right / 1-px up steps.
        let mut stair = Vec::new();
        for i in 0..10 {
            stair.push(sample(i as f32, i as f32));
            stair.push(sample(i as f32 + 1.0, i as f32));
        }
        let simplified = simplify(&stair, SIMPLIFY_TOLERANCE);
        // The staircase hugs the diagonal within ~1px, so it collapses sharply.
        assert!(
            simplified.len() <= 4,
            "staircase should collapse, got {} points",
            simplified.len()
        );
        // Endpoints preserved.
        assert_eq!(simplified.first().unwrap().pos, stair.first().unwrap().pos);
        assert_eq!(simplified.last().unwrap().pos, stair.last().unwrap().pos);
    }

    #[test]
    fn simplify_keeps_real_corners() {
        // An L-shape: the corner must survive.
        let pts = [
            sample(0.0, 0.0),
            sample(10.0, 0.0),
            sample(20.0, 0.0),
            sample(20.0, 10.0),
            sample(20.0, 20.0),
        ];
        let simplified = simplify(&pts, SIMPLIFY_TOLERANCE);
        assert!(simplified.iter().any(|s| s.pos == Vec2::new(20.0, 0.0)));
    }

    #[test]
    fn flatten_straight_line_stays_collinear() {
        let knots = [sample(0.0, 0.0), sample(30.0, 0.0)];
        let dense = flatten(&knots, FLATTEN_STEP);
        assert!(dense.len() > 2);
        assert!(dense.iter().all(|s| s.pos.y.abs() < 1e-3));
        assert_eq!(dense.first().unwrap().pos, Vec2::new(0.0, 0.0));
        assert!((dense.last().unwrap().pos - Vec2::new(30.0, 0.0)).length() < 1e-3);
    }

    #[test]
    fn flatten_passes_through_control_points() {
        // Catmull–Rom interpolates its control points. Arc-length resampling won't
        // land a sample exactly on the apex, but the polyline still passes within a
        // step of it.
        let knots = [
            sample(0.0, 0.0),
            sample(10.0, 10.0),
            sample(20.0, 0.0),
        ];
        let dense = flatten(&knots, FLATTEN_STEP);
        let nearest = dense
            .iter()
            .map(|s| (s.pos - Vec2::new(10.0, 10.0)).length())
            .fold(f32::INFINITY, f32::min);
        assert!(nearest < FLATTEN_STEP, "nearest sample to apex was {nearest}px");
    }

    #[test]
    fn flatten_spaces_samples_by_arc_length() {
        // A curved stroke: consecutive output points are ~`step` apart by distance
        // (the prep for distance-parameterized strokes, DESIGN.md §6.2).
        let knots = [
            sample(0.0, 0.0),
            sample(20.0, 30.0),
            sample(60.0, 30.0),
            sample(90.0, 0.0),
        ];
        let dense = flatten(&knots, FLATTEN_STEP);
        // Every interior gap is close to FLATTEN_STEP (the last may be a shorter
        // remainder, so it is excluded from the upper-bound check).
        for w in dense.windows(2) {
            let d = (w[1].pos - w[0].pos).length();
            assert!(d > 0.0, "no duplicate points");
        }
        let gaps: Vec<f32> = dense.windows(2).map(|w| (w[1].pos - w[0].pos).length()).collect();
        for &d in &gaps[..gaps.len() - 1] {
            assert!(
                (d - FLATTEN_STEP).abs() < 0.25 * FLATTEN_STEP,
                "interior gap {d}px should be ~{FLATTEN_STEP}px",
            );
        }
    }
}
