//! **Arc length ↔ curve parameter**: the map a fit and a realized shape both place
//! their samples through (§6.2).
//!
//! Its own file because it has two independent consumers and belongs to neither.
//! [`PathFitter`](super::PathFitter) builds it to assign each pointer report a place
//! on the curve; `assist::realize` builds it to spread a recognized shape's targets
//! along one. It lived inside `fit`, so `path` re-exported two of its items
//! `pub(crate)` purely so `assist` could reach through the fitter's private module for
//! them — which is the shape `path.rs` already argues against: what belongs to none of
//! the three sits beside them, not inside one.
//!
//! Nothing here knows what is being fitted. It is a curve and a list of settled
//! lengths.

use crate::spline::CubicBSpline;

/// Samples per span used to measure a curve's own arc length (see `arc_profile`).
pub(super) const ARC_SAMPLES_PER_SPAN: usize = 4;

/// Cumulative arc length along `curve`, sampled evenly in *parameter*.
///
/// A clamped B-spline is not parameterized by distance: the triple knots at each end
/// squash the first and last spans into a fraction of the leg they cover, so
/// parameter `t` is well short of `t / num_spans` of the way along. Assuming
/// otherwise leaves a residual on input the curve could fit exactly — a straight
/// stroke read as several px of error — and the growth rule then buys control points
/// to explain it away. Sparse input has nothing between samples to hold those extra
/// control points, so they oscillate.
/// `settled` is a profile of the same curve's *frozen* spans, carried over from an
/// earlier update: those spans' control points are held, so their geometry — and so
/// their length — cannot change, and re-walking them every update is the last piece
/// of per-update work that scaled with the whole stroke rather than the window.
pub(crate) fn arc_profile(curve: &CubicBSpline<'_, 2>, settled: &[f32]) -> Vec<f32> {
    let spans = curve.num_spans();
    let n = spans * ARC_SAMPLES_PER_SPAN;
    let keep = settled.len().saturating_sub(1).min(n);
    let mut cum = Vec::with_capacity(n + 1);
    if keep == 0 {
        cum.push(0.0);
    } else {
        cum.extend_from_slice(&settled[..=keep]);
    }
    let step = |i: usize| i as f32 / ARC_SAMPLES_PER_SPAN as f32;
    let mut prev = curve.evaluate(step(keep));
    for i in keep + 1..=n {
        let c = curve.evaluate(step(i));
        let d = ((c[0] - prev[0]).powi(2) + (c[1] - prev[1]).powi(2)).sqrt();
        cum.push(cum[i - 1] + d);
        prev = c;
    }
    cum
}

/// The parameter at which `profile`'s curve is `f` of the way along its own length.
///
/// This is a *global, monotone* reparameterization: one function, applied to every
/// sample alike. That is what separates it from projecting each sample onto the
/// curve independently — samples keep their order and their relative spacing, so
/// they cannot slide past one another or bunch up on the input's jitter, which is
/// how per-sample correction blew strokes up to 45px with loops.
pub(crate) fn param_at(profile: &[f32], spans: f32, f: f32) -> f32 {
    let total = *profile.last().expect("profile is never empty");
    if total <= 1e-6 || profile.len() < 2 {
        return f.clamp(0.0, 1.0) * spans;
    }
    let want = f.clamp(0.0, 1.0) * total;
    let i = profile
        .partition_point(|&c| c < want)
        .clamp(1, profile.len() - 1);
    let (a, b) = (profile[i - 1], profile[i]);
    let u = if b > a { (want - a) / (b - a) } else { 0.0 };
    (((i - 1) as f32 + u) / ARC_SAMPLES_PER_SPAN as f32).min(spans)
}
