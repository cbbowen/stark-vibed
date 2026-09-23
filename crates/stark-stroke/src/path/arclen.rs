//! **Arc length ↔ curve parameter**: the map a fit and a realized shape both place
//! their samples through (§6.2), and the cumulative table and lookup under it.
//!
//! Two independent consumers, and it belongs to neither:
//! [`PathFitter`](super::PathFitter) builds it to assign each pointer report a place
//! on the curve; `assist::realize` builds it to spread a recognized shape's targets
//! along one. Recognition and realization also walk plain polylines by the same table
//! ([`cumulative_lengths`], [`locate`]).
//!
//! Nothing here knows what is being fitted. It is a curve and a list of settled
//! lengths.

use crate::spline::CubicBSpline;
use stark_model::geom::Vec2;

/// Samples per span used to measure a curve's own arc length (see `arc_profile`).
pub(super) const ARC_SAMPLES_PER_SPAN: usize = 4;

/// Cumulative arc length along `curve`, sampled evenly in *parameter*.
///
/// A clamped B-spline is not parameterized by distance: the triple knots at each end
/// squash the first and last spans into a fraction of the leg they cover, so
/// parameter `t` is well short of `t / num_spans` of the way along. Assuming
/// otherwise leaves a residual on input the curve could fit exactly — a straight
/// stroke read as several px of error — and the growth rule then buys control points
/// to explain it away, which sparse input has nothing to hold down.
///
/// `settled` is a profile of the same curve's *frozen* spans, carried over from an
/// earlier update: those spans' control points are held, so their length cannot
/// change, and re-walking them would be the one piece of per-update work that scales
/// with the whole stroke rather than with the window.
pub(crate) fn arc_profile(curve: &CubicBSpline<'_, 2>, settled: &[f32]) -> Vec<f32> {
    let mut cum = Vec::new();
    arc_profile_into(curve, settled, &mut cum);
    cum
}

/// [`arc_profile`] written into `out`, emptied first — the fitter's form, which
/// builds one per candidate per report and keeps the buffer between them.
pub(super) fn arc_profile_into(curve: &CubicBSpline<'_, 2>, settled: &[f32], out: &mut Vec<f32>) {
    let spans = curve.num_spans();
    let n = spans * ARC_SAMPLES_PER_SPAN;
    let keep = settled.len().saturating_sub(1).min(n);
    out.clear();
    out.reserve(n + 1);
    if keep == 0 {
        out.push(0.0);
    } else {
        out.extend_from_slice(&settled[..=keep]);
    }
    let step = |i: usize| i as f32 / ARC_SAMPLES_PER_SPAN as f32;
    let mut prev = curve.evaluate(step(keep));
    for i in keep + 1..=n {
        let c = curve.evaluate(step(i));
        let d = ((c[0] - prev[0]).powi(2) + (c[1] - prev[1]).powi(2)).sqrt();
        out.push(out[i - 1] + d);
        prev = c;
    }
}

/// The parameter at which `profile`'s curve is `f` of the way along its own length.
///
/// A *global, monotone* reparameterization: one function applied to every sample
/// alike, unlike projecting each sample onto the curve independently. Samples keep
/// their order and their relative spacing, so they cannot slide past one another or
/// bunch up on the input's jitter — per-sample correction blew looped strokes up to
/// 45px.
pub(crate) fn param_at(profile: &[f32], spans: f32, f: f32) -> f32 {
    let total = *profile.last().expect("profile is never empty");
    if total <= 1e-6 || profile.len() < 2 {
        return f.clamp(0.0, 1.0) * spans;
    }
    let (i, u) = locate(profile, f.clamp(0.0, 1.0) * total);
    ((i as f32 + u) / ARC_SAMPLES_PER_SPAN as f32).min(spans)
}

/// The distance along the polyline `pts` to each of its points: `0` at the first, and
/// one entry per point — empty for none.
///
/// `pub` for `benches/path.rs`, which resamples recorded strokes by it.
pub fn cumulative_lengths(pts: &[Vec2]) -> Vec<f32> {
    let mut out = Vec::with_capacity(pts.len());
    if !pts.is_empty() {
        out.push(0.0);
    }
    let mut acc = 0.0f32;
    out.extend(pts.windows(2).map(|w| {
        acc += w[0].distance(w[1]);
        acc
    }));
    out
}

/// The length of the polyline `pts` — [`cumulative_lengths`]' last entry, summed in the
/// same order.
pub(crate) fn polyline_length(pts: &[Vec2]) -> f32 {
    pts.windows(2).fold(0.0, |acc, w| acc + w[0].distance(w[1]))
}

/// Where `want` falls along the non-decreasing table `cum`: `(i, u)`, `u` of the way
/// from entry `i` to entry `i + 1`, and 0 across a step of no length.
///
/// Before the first entry or past the last it reads the end interval, extrapolated;
/// a caller that must hold the ends instead tests them first.
///
/// # Panics
///
/// If `cum` has fewer than two entries: there is no interval to fall in.
///
/// `pub` for `benches/path.rs`, which resamples recorded strokes by it.
pub fn locate(cum: &[f32], want: f32) -> (usize, f32) {
    let i = cum.partition_point(|&c| c < want).clamp(1, cum.len() - 1);
    let (a, b) = (cum[i - 1], cum[i]);
    (i - 1, if b > a { (want - a) / (b - a) } else { 0.0 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// An arc profile of 1 to 23 spans: non-decreasing from 0, with flats — the stretches
    /// a stationary curve leaves — included.
    fn profile() -> impl Strategy<Value = (Vec<f32>, f32)> {
        (1usize..24).prop_flat_map(|spans| {
            let step = prop_oneof![1 => Just(0.0f32), 4 => 0.0f32..50.0];
            prop::collection::vec(step, spans * ARC_SAMPLES_PER_SPAN).prop_map(move |steps| {
                let cum = std::iter::once(0.0)
                    .chain(steps.into_iter().scan(0.0f32, |acc, d| {
                        *acc += d;
                        Some(*acc)
                    }))
                    .collect();
                (cum, spans as f32)
            })
        })
    }

    /// One entry per point from 0, the total as the last, and a lookup that lands where
    /// the table says.
    #[test]
    fn a_polyline_table_reads_its_own_lengths() {
        let pts = [
            Vec2::ZERO,
            Vec2::new(3.0, 4.0),
            Vec2::new(3.0, 4.0),
            Vec2::new(3.0, 14.0),
        ];
        let cum = cumulative_lengths(&pts);
        assert_eq!(cum, [0.0, 5.0, 5.0, 15.0]);
        assert_eq!(polyline_length(&pts), 15.0);
        assert!(cumulative_lengths(&[]).is_empty());
        assert_eq!(cumulative_lengths(&pts[..1]), [0.0]);
        assert_eq!(locate(&cum, 2.5), (0, 0.5));
        // The flat step is skipped: 5.0 is the end of the first leg, not the dwell.
        assert_eq!(locate(&cum, 5.0), (0, 1.0));
        assert_eq!(locate(&cum, 10.0), (2, 0.5));
        // Past the last entry, the last interval extrapolated.
        assert_eq!(locate(&cum, 20.0), (2, 1.5));
    }

    proptest! {
        #![proptest_config(crate::path::test_support::cases())]

        /// The map is monotone and lands inside `[0, spans]` for any fraction, in range
        /// or not — what lets samples keep their order along the curve.
        #[test]
        fn param_at_is_monotone_and_inside_the_domain(
            (profile, spans) in profile(),
            mut fs in prop::collection::vec(-0.5f32..1.5, 2..64),
        ) {
            fs.sort_by(f32::total_cmp);
            let ts: Vec<f32> = fs.iter().map(|&f| param_at(&profile, spans, f)).collect();
            prop_assert!(
                ts.iter().all(|t| (0.0..=spans).contains(t)),
                "outside [0, {}]: {:?}",
                spans,
                ts
            );
            prop_assert!(ts.windows(2).all(|w| w[0] <= w[1]), "not monotone: {:?}", ts);
        }
    }
}
