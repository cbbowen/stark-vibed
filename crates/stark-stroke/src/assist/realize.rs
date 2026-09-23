//! **Realization**: the ideal shape as a fitted path, carrying the pen channels the
//! stroke was drawn with (§6.9).
//!
//! This is what keeps a snapped stroke *painted* rather than vector art with a brush
//! texture on it: the geometry is replaced wholesale, and the pressure, tilt and time
//! that were put into the drawn stroke are carried onto the ideal shape at the same
//! fraction of the way along.

use super::{AssistShape, Frame};
use crate::path::{
    CHANNELS, FLATTEN_TOLERANCE, arc_profile, control_points, cumulative_lengths,
    flatten_spans_from, locate, param_at, pen_row, span_count,
};
use crate::spline::{CubicBSpline, Observations, SplineIndex};
use stark_model::geom::Vec2;
use stark_model::path::ControlPoint;
use std::f32::consts::TAU;

/// How far a snapped ellipse may sit off the true one, in canvas px — what fixes its
/// leg count.
///
/// The binding error is *not* the interior ripple (`r·Δ⁴/384` for a leg of `Δ` radians,
/// microscopic at any leg count worth using) but the **clamped end**: the first leg of
/// the path is deformed by the end condition, and its chord bows `r·Δ²/8` off the arc it
/// stands for. Placing the end control points by least squares ([`realize`]) spreads
/// that, leaving a little under a quarter — measured at 1.69px for a 200px circle on 30°
/// legs, against `r·Δ²/8 = 6.85`. So the leg count solves `r·Δ²/24 ≤` this, the divisor
/// rounded *down* from the measured ratio so the bound is a bound.
const ELLIPSE_ERROR: f32 = 0.4;

/// Legs a snapped ellipse is built from, whatever [`ELLIPSE_ERROR`] asks for. The floor
/// keeps a thumbnail ellipse from being a polygon; the ceiling keeps a canvas-wide one
/// bounded — and it is not a real cost, since the fitter itself would spend *more*
/// control points than this on a stroke that long (`path::fit::KNOT_SPACING`).
const MIN_LEGS: usize = 12;

const MAX_LEGS: usize = 96;

/// Extra legs drawn past a full turn, at each end.
///
/// A clamped cubic B-spline's **first span is a straight chord** — the clamp collapses
/// three of its four Bézier points onto the first control point — so an ellipse cannot
/// be exact at the two ends of the open path that draws it, however many control points
/// it gets. Overlapping the seam puts that flat sixth-of-a-leg *underneath* the far end's
/// correctly-curved interior, which is also what makes a closed loop join without a notch.
const SEAM_OVERLAP: usize = 2;

/// Ceiling on a snapped **line**'s control points. An ellipse is bounded by its legs
/// instead, at `MAX_LEGS + 2·SEAM_OVERLAP + 1` = 101. Neither needs to outgrow what the
/// fitter itself would spend on the same stroke.
const MAX_KNOTS: usize = 96;

/// Ideal-shape samples per control point that the realization fits against.
const TARGETS_PER_KNOT: usize = 4;

/// The pen channels of the stroke as drawn, against distance along it.
///
/// Read off the *fitted* path rather than off the raw reports: that is already the
/// smoothed, de-jittered version of the same signal, and it is what the stroke would
/// have been drawn with had nothing snapped.
pub struct PenProfile {
    /// Fraction along the stroke of each row, in order.
    at: Vec<f32>,
    rows: Vec<[f32; CHANNELS]>,
}

impl PenProfile {
    /// The profile of a fitted path from its marker `start` on
    /// ([`StrokeRecord::start`](stark_model::document::StrokeRecord::start)). The
    /// shape is recognized from the stroke after the press, so a run-up ahead of the
    /// marker is left out here too, or it would shift the pen along the shape.
    pub fn of(path: &[ControlPoint], start: f32) -> Self {
        let spans = 0..span_count(path.len());
        let poly = flatten_spans_from(path, start, spans, 0.0, FLATTEN_TOLERANCE);
        let total = poly.last().map_or(0.0, |s| s.dist);
        let inv = if total > 1e-6 { 1.0 / total } else { 0.0 };
        let (at, rows) = poly
            .iter()
            .map(|s| (s.dist * inv, pen_row(s.pressure, s.tilt, s.time)))
            .unzip();
        Self { at, rows }
    }

    /// The channels `f` of the way along the drawn stroke, `f ∈ [0, 1]`.
    fn sample(&self, f: f32) -> [f32; CHANNELS] {
        let (Some(&first), Some(&last)) = (self.rows.first(), self.rows.last()) else {
            // Nothing was drawn: a pen at rest.
            return pen_row(1.0, Vec2::ZERO, 0.0);
        };
        let f = f.clamp(0.0, 1.0);
        // The ends are held, not extrapolated — a NaN reads the first.
        if f.is_nan() || f <= self.at[0] {
            return first;
        }
        if f > self.at[self.at.len() - 1] {
            return last;
        }
        let (i, u) = locate(&self.at, f);
        let (a, b) = (self.rows[i], self.rows[i + 1]);
        std::array::from_fn(|d| a[d] + (b[d] - a[d]) * u)
    }
}

impl AssistShape {
    /// The ideal shape as a fitted path, carrying `pen`'s channels along it.
    ///
    /// `knots` is what the drawn stroke itself was fitted to: the shape never gets
    /// *fewer* control points than the stroke it replaces, because the pen channels
    /// ride the same polygon as the geometry and a pressure profile needs somewhere to
    /// live. Geometry decides the floor and the pen decides nothing else — the same
    /// split `SplineIndex::fit_into` draws.
    ///
    /// The geometry of a **line** is placed in closed form: any collinear control polygon
    /// draws exactly that line. An **ellipse** is *fitted*, because a clamped B-spline's
    /// ends are pinned to their own control points, so points placed analytically on the
    /// ellipse leave an `O(Δ²)` bulge exactly at the seam and the solve is what places the
    /// end rows to cancel it. The pen channels are fitted either way.
    pub fn to_path(self, pen: &PenProfile, knots: usize) -> Vec<ControlPoint> {
        let (seed, targets, fit_geometry) = match self {
            Self::Line { a, b, .. } => {
                let m = knots.clamp(2, MAX_KNOTS);
                let along = |f: f32| a.lerp(b, f);
                let seed = spread(m, along);
                let targets = spread(m * TARGETS_PER_KNOT, along);
                (seed, targets, false)
            }
            Self::Ellipse {
                shape,
                phase,
                winding,
                ..
            } => {
                // One leg per `delta` radians, from the ripple a leg leaves against
                // the true ellipse — then widened, if need be, to carry the pen.
                let want = knots.saturating_sub(2 * SEAM_OVERLAP + 1);
                let legs = leg_count(shape.radii.max_element()).max(want).min(MAX_LEGS);
                let m = legs + 2 * SEAM_OVERLAP + 1;
                let delta = TAU / legs as f32;
                let span = (m - 1) as f32 * delta;
                // Control points sit on a slightly *larger* ellipse: a uniform cubic
                // B-spline runs inside its own control polygon by `(1 - cos Δ)/3`, and
                // undoing that is what makes the drawn curve the ellipse asked for.
                // Exact for an ellipse as well as a circle — the construction is an
                // affine image of the circle case, and B-splines commute with affine maps.
                let bulge = 3.0 / (2.0 + delta.cos());
                let start = -(SEAM_OVERLAP as f32) * delta;
                let frame = Frame::of(shape);
                let at = |f: f32, scale: f32| {
                    frame.point_scaled(winding.advance(phase, start + f * span), scale)
                };
                let seed = spread(m, |f| at(f, bulge));
                let targets = spread(m * TARGETS_PER_KNOT, |f| at(f, 1.0));
                (seed, targets, true)
            }
        };
        realize(&seed, &targets, pen, fit_geometry)
    }
}

/// `n` values of `f` spread over `[0, 1]` inclusive.
///
/// [`spread_scalar`] is the fractions themselves, and this is those mapped — written
/// out separately, the two had the same off-by-one guard twice.
fn spread(n: usize, f: impl Fn(f32) -> Vec2) -> Vec<Vec2> {
    spread_scalar(n).into_iter().map(f).collect()
}

/// Fit `seed`'s control polygon to `targets`, and the pen channels alongside it.
///
/// The parameters the targets are fitted at come from the seed curve's own **arc
/// profile**, the same correction [`PathFitter`](crate::path::PathFitter) makes: a
/// clamped B-spline is not parameterized by distance, and assuming otherwise leaves a
/// residual on data the curve could match exactly.
fn realize(
    seed: &[Vec2],
    targets: &[Vec2],
    pen: &PenProfile,
    fit_geometry: bool,
) -> Vec<ControlPoint> {
    let m = seed.len();
    let mut geom: Vec<[f32; 2]> = seed.iter().map(|p| p.to_array()).collect();
    let Ok(index) = SplineIndex::new(m) else {
        // Fewer than two control points is not a curve; the caller's clamps rule it
        // out, and there is nothing to draw if they ever did not.
        return Vec::new();
    };
    let spans = index.num_spans() as f32;
    // The curve is read only to measure its arc profile, so the borrow ends here and
    // `geom` is free to be fitted in place below.
    let profile = arc_profile(
        &CubicBSpline::new(&geom).expect("the index agreed there are enough"),
        &[],
    );
    let fractions = arc_fractions(targets);
    let ts: Vec<f32> = fractions
        .iter()
        .map(|&f| param_at(&profile, spans, f))
        .collect();

    if fit_geometry {
        let values: Vec<[f32; 2]> = targets.iter().map(|p| p.to_array()).collect();
        index.fit_into(Observations::even(&ts, &values), &mut geom, 0, 0, 0.0);
    }

    // Seeded from the profile at each control point's own share of the polygon, not
    // from zero: the solve's proximal ridge pulls towards its prior, and a prior of
    // zero biases every channel low by about a percent (see `spline`'s tests).
    let mut attr: Vec<[f32; CHANNELS]> = (0..m)
        .map(|j| pen.sample(j as f32 / (m - 1).max(1) as f32))
        .collect();
    let values: Vec<[f32; CHANNELS]> = fractions.iter().map(|&f| pen.sample(f)).collect();
    index.fit_into(Observations::even(&ts, &values), &mut attr, 0, 0, 0.0);

    // Through the fitter's own door, which is `clamped`: this is a least-squares solve
    // too, and overshoots the same way.
    control_points(&geom, &attr)
}

/// Cumulative chord length along `pts`, normalized to `[0, 1]`.
fn arc_fractions(pts: &[Vec2]) -> Vec<f32> {
    let cum = cumulative_lengths(pts);
    match cum.last() {
        Some(&total) if total > 1e-6 => cum.iter().map(|c| c / total).collect(),
        _ => spread_scalar(pts.len()),
    }
}

/// `n` fractions spread over `[0, 1]` — the fallback when a polyline has no length.
fn spread_scalar(n: usize) -> Vec<f32> {
    let last = n.saturating_sub(1).max(1) as f32;
    (0..n).map(|i| i as f32 / last).collect()
}

/// Legs a full turn of an ellipse of this radius is drawn with — `r·Δ²/24 ≤`
/// [`ELLIPSE_ERROR`], which is what makes the count follow the shape's size instead of
/// being a number somebody picked.
fn leg_count(radius: f32) -> usize {
    if !(radius.is_finite() && radius > 0.0) {
        return MIN_LEGS;
    }
    let delta = (24.0 * ELLIPSE_ERROR / radius).sqrt();
    let legs = (TAU / delta).ceil();
    if legs.is_finite() {
        (legs as usize).clamp(MIN_LEGS, MAX_LEGS)
    } else {
        MAX_LEGS
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assist::Winding;
    use crate::path::flatten;
    use stark_model::geom::Ellipse;

    /// The realized path has to *be* the shape, not merely resemble it — including at
    /// the clamped ends, which is what the ellipse's least-squares placement buys.
    #[test]
    fn a_snapped_line_is_straight() {
        let (a, b) = (Vec2::new(-50.0, 30.0), Vec2::new(250.0, 130.0));
        let pen = PenProfile::of(&[ControlPoint::at(a), ControlPoint::at(b)], 0.0);
        let path = AssistShape::Line {
            a,
            b,
            on_axis: false,
        }
        .to_path(&pen, 12);
        let poly = flatten(&path, FLATTEN_TOLERANCE);
        let dir = (b - a).normalize();
        let worst = poly
            .iter()
            .map(|s| (s.pos - a).perp_dot(dir).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 0.02, "drifted {worst}px off the line");
    }

    #[test]
    fn a_snapped_ellipse_is_round() {
        for radius in [40.0f32, 200.0, 1200.0] {
            let shape = AssistShape::Ellipse {
                shape: Ellipse::new(Vec2::new(7.0, -13.0), Vec2::splat(radius), 0.0),
                plane: None,
                phase: 0.3,
                winding: Winding::Forward,
            };
            let pen = PenProfile::of(
                &[ControlPoint::at(Vec2::ZERO), ControlPoint::at(Vec2::X)],
                0.0,
            );
            let path = shape.to_path(&pen, 8);
            let poly = flatten(&path, FLATTEN_TOLERANCE);
            let worst = poly
                .iter()
                .map(|s| (s.pos.distance(Vec2::new(7.0, -13.0)) - radius).abs())
                .fold(0.0f32, f32::max);
            assert!(worst <= ELLIPSE_ERROR, "radius {radius}: off by {worst}px");
        }
    }

    /// The pen channels ride the ideal shape rather than being flattened out of it.
    #[test]
    fn a_snapped_stroke_keeps_its_pressure() {
        let drawn: Vec<ControlPoint> = (0..9)
            .map(|i| {
                let t = i as f32 / 8.0;
                ControlPoint {
                    pos: Vec2::new(t * 300.0, 0.0),
                    // A swell: light at both ends, full in the middle.
                    pressure: 0.15 + 0.85 * (t * std::f32::consts::PI).sin(),
                    tilt: Vec2::ZERO,
                    time: t,
                }
            })
            .collect();
        let pen = PenProfile::of(&drawn, 0.0);
        let path = AssistShape::Line {
            a: Vec2::ZERO,
            b: Vec2::new(300.0, 0.0),
            on_axis: false,
        }
        .to_path(&pen, drawn.len());
        let mid = path[path.len() / 2].pressure;
        let ends = path[0].pressure.max(path[path.len() - 1].pressure);
        assert!(mid > 0.8, "the swell was flattened out (mid {mid})");
        assert!(ends < 0.4, "the light ends were filled in (ends {ends})");
    }

    /// The profile starts at the marker: a run-up ahead of it is entry geometry, not
    /// pen the shape carries (§6.2). The shape starts at the press, so its first
    /// control point reads the press's channels.
    #[test]
    fn the_profile_starts_at_the_marker() {
        // Control points 0..4 are the run-up: light, leaning, and before the press.
        let drawn: Vec<ControlPoint> = (0..13)
            .map(|i| {
                let run_up = i < 4;
                ControlPoint {
                    pos: Vec2::new(i as f32 * 25.0, 0.0),
                    pressure: if run_up { 0.1 } else { 0.9 },
                    tilt: if run_up {
                        Vec2::new(0.6, 0.0)
                    } else {
                        Vec2::ZERO
                    },
                    time: (i as f32 - 4.0) * 0.1,
                }
            })
            .collect();
        // Span 6 reads control points 4..=7, all past the run-up.
        let start = 6.0;
        let press = flatten_spans_from(
            &drawn,
            start,
            0..span_count(drawn.len()),
            0.0,
            FLATTEN_TOLERANCE,
        )[0];
        let path = AssistShape::Line {
            a: press.pos,
            b: drawn[drawn.len() - 1].pos,
            on_axis: false,
        }
        .to_path(&PenProfile::of(&drawn, start), drawn.len());

        let first = path[0];
        assert!(
            (first.pressure - press.pressure).abs() < 0.05,
            "pressure {} at the shape's start, {} at the press",
            first.pressure,
            press.pressure
        );
        assert!(
            first.tilt.distance(press.tilt) < 0.05,
            "tilt {} at the shape's start, {} at the press",
            first.tilt,
            press.tilt
        );
        assert!(
            (first.time - press.time).abs() < 0.05,
            "time {} at the shape's start, {} at the press",
            first.time,
            press.time
        );
    }
}
