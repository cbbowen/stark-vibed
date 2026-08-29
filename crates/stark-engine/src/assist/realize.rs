//! **Realization**: the ideal shape as a fitted path, carrying the pen channels the
//! stroke was drawn with (§6.9).
//!
//! This is what keeps a snapped stroke *painted* rather than turning it into vector
//! art with a brush texture on it: the geometry is replaced wholesale, and the
//! pressure, tilt and time that were put into the drawn stroke are carried onto the
//! ideal shape at the same fraction of the way along.

use super::AssistShape;
use crate::path::{FLATTEN_TOLERANCE, arc_profile, flatten, param_at};
use crate::spline::{CubicBSpline, Observations, SplineIndex};
use nalgebra::{Const, Dyn, OMatrix};
use stark_model::geom::Vec2;
use stark_model::path::ControlPoint;
use std::f32::consts::TAU;

/// How far a snapped ellipse may sit off the true one, in canvas px — what fixes its
/// leg count.
///
/// The binding error is *not* the interior ripple (`r·Δ⁴/384` for a leg of `Δ`
/// radians, which is microscopic at any leg count worth using) but the **clamped end**:
/// the first leg of the path is deformed by the end condition, and its chord bows
/// `r·Δ²/8` off the arc it stands for. Placing the end control points by least squares
/// ([`realize`]) spreads that, leaving a little under a quarter of it — measured at
/// 1.69px for a 200px circle on 30° legs, against `r·Δ²/8 = 6.85`. So the leg count
/// solves `r·Δ²/24 ≤` this: the divisor is rounded *down* from the measured ratio, so
/// the number below is a bound rather than an average of one.
pub(super) const ELLIPSE_ERROR: f32 = 0.4;

/// Legs a snapped ellipse is built from, whatever [`ELLIPSE_ERROR`] asks for. The floor
/// keeps a thumbnail ellipse from being a polygon; the ceiling keeps a canvas-wide one
/// bounded — and it is not a real cost, since the fitter itself would spend *more*
/// control points than this on a stroke that long ([`KNOT_SPACING`](crate::path::KNOT_SPACING)).
const MIN_LEGS: usize = 12;

const MAX_LEGS: usize = 96;

/// Extra legs drawn past a full turn, at each end.
///
/// A clamped cubic B-spline's **first span is a straight chord** — the clamp collapses
/// three of its four Bézier points onto the first control point — so an ellipse cannot
/// be exact at the two ends of the open path that draws it, however many control
/// points it gets. Overlapping the seam puts that flat sixth-of-a-leg *underneath* the
/// far end's correctly-curved interior instead of beside it, which is also what makes
/// a closed loop join without a notch.
const SEAM_OVERLAP: usize = 2;

/// Ceiling on a snapped path's control points. Nothing here needs to outgrow what the
/// fitter itself would spend on the same stroke.
const MAX_KNOTS: usize = 96;

/// Ideal-shape samples per control point that the realization fits against.
const TARGETS_PER_KNOT: usize = 4;

/// The pen channels of the stroke as drawn, against distance along it.
///
/// This is what keeps a snapped stroke *painted*. The geometry is replaced wholesale;
/// the pressure, tilt and time that were put into it are carried onto the ideal shape
/// at the same fraction of the way along, so a line snapped out of a stroke that swelled
/// in the middle still swells in the middle. Without it the feature would produce
/// vector art with a brush texture on it.
///
/// Read off the *fitted* path rather than off the raw reports, because that is already
/// the smoothed, de-jittered version of the same signal and it is what the stroke would
/// have been drawn with had nothing snapped.
pub struct PenProfile {
    /// `(fraction along, [pressure, tilt x, tilt y, time])`, in order.
    at: Vec<(f32, [f32; CHANNELS])>,
}

/// Pen channels riding the control polygon — the same four [`PathFitter`](crate::path::PathFitter)
/// solves for.
const CHANNELS: usize = 4;

impl PenProfile {
    /// The profile of a fitted path.
    pub fn of(path: &[ControlPoint]) -> Self {
        let poly = flatten(path, FLATTEN_TOLERANCE);
        let total = poly.last().map_or(0.0, |s| s.dist);
        let inv = if total > 1e-6 { 1.0 / total } else { 0.0 };
        Self {
            at: poly
                .iter()
                .map(|s| (s.dist * inv, [s.pressure, s.tilt.x, s.tilt.y, s.time]))
                .collect(),
        }
    }

    /// The channels `f` of the way along the drawn stroke, `f ∈ [0, 1]`.
    fn sample(&self, f: f32) -> [f32; CHANNELS] {
        let Some(&(_, first)) = self.at.first() else {
            return [1.0, 0.0, 0.0, 0.0];
        };
        let f = f.clamp(0.0, 1.0);
        let i = self.at.partition_point(|&(x, _)| x < f);
        if i == 0 {
            return first;
        }
        let Some(&(hi, b)) = self.at.get(i) else {
            return self.at[self.at.len() - 1].1;
        };
        let (lo, a) = self.at[i - 1];
        let u = if hi > lo { (f - lo) / (hi - lo) } else { 0.0 };
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
    /// split [`CubicBSpline::fit_channels`] draws.
    ///
    /// The geometry of a **line** is placed in closed form (any collinear control
    /// polygon draws exactly that line, so there is nothing to solve and nothing to
    /// round off). An **ellipse** is *fitted*: a clamped B-spline's ends are pinned to
    /// their own control points, so control points placed analytically on the ellipse
    /// leave an `O(Δ²)` bulge exactly at the seam, and a solve is what places the end
    /// rows to cancel it. The pen channels are fitted either way.
    pub fn to_path(self, pen: &PenProfile, knots: usize) -> Vec<ControlPoint> {
        let (seed, targets, fit_geometry) = match self {
            Self::Line { a, b, .. } => {
                let m = knots.clamp(2, MAX_KNOTS);
                let along = |f: f32| a.lerp(b, f);
                let seed = spread(m, along);
                let targets = spread(m * TARGETS_PER_KNOT, along);
                (seed, targets, false)
            }
            Self::Ellipse { radii, .. } => {
                // One leg per `delta` radians, from the ripple a leg leaves against
                // the true ellipse — then widened, if need be, to carry the pen.
                let want = knots.saturating_sub(2 * SEAM_OVERLAP + 1);
                let legs = leg_count(radii.max_element()).max(want).min(MAX_LEGS);
                let m = legs + 2 * SEAM_OVERLAP + 1;
                let delta = TAU / legs as f32;
                let span = (m - 1) as f32 * delta;
                // Control points sit on a slightly *larger* ellipse: a uniform cubic
                // B-spline runs inside its own control polygon by `(1 - cos Δ)/3`, and
                // undoing that is what makes the drawn curve the ellipse asked for
                // rather than a shrunken one. Exact for an ellipse as well as a circle
                // — the construction is an affine image of the circle case, and
                // B-splines commute with affine maps.
                let bulge = 3.0 / (2.0 + delta.cos());
                let start = -(SEAM_OVERLAP as f32) * delta;
                let seed = spread(m, |f| self.at(start + f * span, bulge));
                let targets = spread(m * TARGETS_PER_KNOT, |f| self.at(start + f * span, 1.0));
                (seed, targets, true)
            }
        };
        realize(&seed, &targets, pen, fit_geometry)
    }

    /// A point of the shape's own parameterization, scaled about the centre by `bulge`
    /// (1.0 for the shape itself). For a line the parameter is the fraction from `a`
    /// to `b`; for an ellipse it is radians past the seam, signed by the winding.
    pub(super) fn at(&self, t: f32, bulge: f32) -> Vec2 {
        match *self {
            Self::Line { a, b, .. } => a.lerp(b, t),
            Self::Ellipse {
                center,
                radii,
                angle,
                phase,
                winding,
                ..
            } => {
                let u = phase + winding * t;
                let local = Vec2::new(radii.x * u.cos(), radii.y * u.sin()) * bulge;
                center + Vec2::from_angle(angle).rotate(local)
            }
        }
    }
}

/// `n` values of `f` spread over `[0, 1]` inclusive.
fn spread(n: usize, f: impl Fn(f32) -> Vec2) -> Vec<Vec2> {
    let last = n.saturating_sub(1).max(1) as f32;
    (0..n).map(|i| f(i as f32 / last)).collect()
}

/// Fit `seed`'s control polygon to `targets`, and the pen channels alongside it.
///
/// The parameters the targets are fitted at come from the seed curve's own **arc
/// profile**, which is the correction [`PathFitter`](crate::path::PathFitter) makes
/// for the same reason: a clamped B-spline is not parameterized by distance, and
/// assuming otherwise leaves a residual on data the curve could match exactly.
fn realize(
    seed: &[Vec2],
    targets: &[Vec2],
    pen: &PenProfile,
    fit_geometry: bool,
) -> Vec<ControlPoint> {
    let m = seed.len();
    let geom_seed = OMatrix::<f32, Dyn, Const<2>>::from_fn_generic(Dyn(m), Const::<2>, |j, d| {
        if d == 0 { seed[j].x } else { seed[j].y }
    });
    let Ok(index) = SplineIndex::new(m) else {
        // Fewer than two control points is not a curve; the caller's clamps rule it
        // out, and there is nothing to draw if they ever did not.
        return Vec::new();
    };
    let spans = index.num_spans() as f32;
    // The curve is read only to measure its arc profile, so the borrow ends here and
    // `geom_seed` is free to be handed to the fit — or moved out of — below.
    let profile = arc_profile(
        &CubicBSpline::new(&geom_seed).expect("the index agreed there are enough"),
        &[],
    );
    let fractions = arc_fractions(targets);
    let ts: Vec<f32> = fractions
        .iter()
        .map(|&f| param_at(&profile, spans, f))
        .collect();

    let geom = if fit_geometry {
        let values: Vec<[f32; 2]> = targets.iter().map(|p| [p.x, p.y]).collect();
        index.fit_channels(Observations::even(&ts, &values), 0, 0, &geom_seed, 0.0)
    } else {
        geom_seed
    };

    // Seeded from the profile at each control point's own share of the polygon, not
    // from zero: the solve's proximal ridge pulls towards its prior, and a prior of
    // zero biases every channel low by about a percent (see `spline`'s tests).
    let attr_seed =
        OMatrix::<f32, Dyn, Const<CHANNELS>>::from_fn_generic(Dyn(m), Const::<CHANNELS>, |j, d| {
            pen.sample(j as f32 / (m - 1).max(1) as f32)[d]
        });
    let values: Vec<[f32; CHANNELS]> = fractions.iter().map(|&f| pen.sample(f)).collect();
    let attr = index.fit_channels(Observations::even(&ts, &values), 0, 0, &attr_seed, 0.0);

    // Through `clamped`, the same door the streaming fitter's own solved channels go
    // through — this is a least-squares solve too, and overshoots the same way.
    (0..m)
        .map(|j| {
            ControlPoint::clamped(
                Vec2::new(geom[(j, 0)], geom[(j, 1)]),
                attr[(j, 0)],
                Vec2::new(attr[(j, 1)], attr[(j, 2)]),
                attr[(j, 3)],
            )
        })
        .collect()
}

/// Cumulative chord length along `pts`, normalized to `[0, 1]`.
fn arc_fractions(pts: &[Vec2]) -> Vec<f32> {
    let mut cum = Vec::with_capacity(pts.len());
    let mut acc = 0.0;
    let mut prev = pts.first().copied().unwrap_or(Vec2::ZERO);
    for p in pts {
        acc += prev.distance(*p);
        cum.push(acc);
        prev = *p;
    }
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
