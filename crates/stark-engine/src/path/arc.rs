//! **Arcs**: what a flattened edge actually stands for (§6.2).
//!
//! [`flatten`] replaces a piece of curve with one edge, and the renderer sweeps that
//! edge as a **circular arc** rather than a chord (§6.2). So this is where
//! the arc is defined, and the flattening budget below is measured against it — the
//! two have to be the same primitive or the budget is describing geometry nobody draws.
//!
//! Which matters twice over. A chord's error is first order in the turn and its outline
//! carries a curvature *impulse* at every joint — a round join of radius `r` spliced
//! between two straight runs — and periodic curvature impulses along a silhouette are
//! what the eye reads as facets. An arc's error is second order and its joints only
//! step the curvature slightly, so the artifact is gone whatever the segment length.
//! Measuring against the arc is therefore what lets segments grow: the same declared
//! error now buys a much longer edge, and buys it without the facets back.

use super::fit::point_segment_distance;
use stark_model::geom::Vec2;

/// The least sagitta (canvas px) an arc has to buy before an edge is bent at all.
/// Under this the arc is indistinguishable from its chord, so it is reported straight
/// — which keeps a straight or barely-curved stroke on exactly the floats it had
/// before arcs existed.
const MIN_SAGITTA: f32 = 0.01;

/// Cap on `sin(θ/2)` for the turn `θ` one edge may bend through — ~23°, far past the
/// ~5.7° [`FLATTEN_TOLERANCE`]'s `angle` bound admits. A backstop on a pathological
/// edge (a cusp, or a span that bottomed out [`MAX_SUBDIVISION_DEPTH`]) rather than a
/// quality knob: past it the series below stop being the right approximation, and so
/// does the annular sector the shaders rasterize a curved sweep as.
const MAX_HALF_TURN_SIN: f32 = 0.2;

/// One flattened edge, as it will actually be swept.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Arc {
    /// Unit tangent the arc leaves its start along.
    pub dir: Vec2,
    /// Signed curvature (1/canvas px), positive turning left of `dir`. Exactly 0 for
    /// a straight edge, and then `dir` is the chord's own direction — every consumer
    /// branches on that zero to keep the straight case on its original arithmetic.
    pub curvature: f32,
    /// Length along the arc.
    pub length: f32,
}

/// `sin(x)`, `1 − cos(x)` and `asin(u)/u` as **polynomials**, for the same reason
/// [`taper_profile`](crate::gpu::stroke) is one: these decide where segments are cut
/// and so which pixels are stored, and replay, goldens and peers all have to agree on
/// them to the last bit — which the transcendental library functions are not specified
/// to. Each is a truncated Maclaurin series, accurate to better than 1e-7 relative
/// over the range its caller is guarded to ([`MAX_HALF_TURN_SIN`]), well inside the
/// f32 the result lands in.
///
/// `versin` is `1 − cos(x)` evaluated *directly* rather than by subtraction: it is
/// used where `x` is small and the difference would cancel away most of its digits.
fn sin_small(x: f32) -> f32 {
    let x2 = x * x;
    x * (1.0 - x2 * (1.0 / 6.0 - x2 * (1.0 / 120.0 - x2 / 5040.0)))
}

fn versin_small(x: f32) -> f32 {
    let x2 = x * x;
    x2 * (0.5 - x2 * (1.0 / 24.0 - x2 / 720.0))
}

fn asin_over_x(u: f32) -> f32 {
    let u2 = u * u;
    1.0 + u2 * (1.0 / 6.0 + u2 * (3.0 / 40.0 + u2 * (15.0 / 336.0)))
}

/// The arc standing in for the edge from a curve point with derivative `vel` along
/// chord `v`: the arc that leaves along the **curve's own tangent** and passes through
/// the far end.
///
/// `max_curvature` is the tightest arc the *caller* can actually sweep
/// ([`FlattenTolerance::max_arc_curvature`](super::FlattenTolerance::max_arc_curvature)); anything tighter comes back straight.
/// That cap is a parameter rather than a constant because it depends on the brush, and
/// it is what keeps `flatten::within` honest: the flattener prices an edge as whatever this
/// returns, and the renderer sweeps whatever this returns, so the two cannot disagree
/// about which primitive the budget was spent on.
///
/// Fitting from the start tangent rather than from both is what makes it cheap: with
/// `t̂` the unit tangent and `n̂` its left normal, an arc leaving along `t̂` reaches
/// chord `v` iff `κ = 2 (v·n̂)/|v|²`, and `sin(θ/2) = κ|v|/2` falls straight out of the
/// same identity. No angle is ever formed, so nothing here needs a transcendental
/// beyond the series above.
///
/// The end tangent is then *not* pinned to the curve's tangent there, so consecutive
/// arcs still meet with a small kink — but it is second order in the turn where the
/// chord's was first order, which is the whole point.
pub fn fit_arc(vel: Vec2, v: Vec2, max_curvature: f32) -> Arc {
    let chord = v.length();
    if chord < 1e-5 {
        // No chord names no direction. Length 0 makes the edge a point, which is what
        // the caller will do with it either way.
        return Arc {
            dir: Vec2::new(1.0, 0.0),
            curvature: 0.0,
            length: chord,
        };
    }
    let straight = Arc {
        dir: v / chord,
        curvature: 0.0,
        length: chord,
    };
    let speed = vel.length();
    if speed < 1e-12 {
        // A stationary derivative names no tangent (`turn` declines the same case):
        // there is nothing to bend towards.
        return straight;
    }
    let t = vel / speed;
    let n = Vec2::new(-t.y, t.x);
    let curvature = 2.0 * v.dot(n) / (chord * chord);
    if !curvature.is_finite() || curvature == 0.0 {
        return straight;
    }
    // Half the turn, as its sine — the arc is `chord = 2 sin(θ/2)/κ` by construction.
    let u = 0.5 * curvature * chord;
    if u.abs() > MAX_HALF_TURN_SIN || v.dot(t) <= 0.0 || curvature.abs() > max_curvature {
        return straight;
    }
    if arc_sagitta(curvature, chord * asin_over_x(u)) < MIN_SAGITTA {
        return straight;
    }
    Arc {
        dir: t,
        curvature,
        length: chord * asin_over_x(u),
    }
}

/// How far an arc bows away from its own chord: `|R|(1 − cos(θ/2))`, and 0 for a
/// straight edge. What a bounding box has to add, since the two endpoints no longer
/// bound the edge between them.
pub fn arc_sagitta(curvature: f32, length: f32) -> f32 {
    if curvature == 0.0 {
        return 0.0;
    }
    versin_small(0.5 * curvature * length) / curvature.abs()
}

/// Where an arc leaving `start` along `dir` with signed `curvature` has got to after
/// `s` of arc length, and the unit tangent it is travelling along there. The straight
/// case is the exact limit and is taken as one, so a `curvature == 0` edge is stepped
/// by plain addition.
pub fn arc_at(start: Vec2, dir: Vec2, curvature: f32, s: f32) -> (Vec2, Vec2) {
    if curvature == 0.0 {
        return (start + dir * s, dir);
    }
    let turn = curvature * s;
    let sn = sin_small(turn);
    let vs = versin_small(turn);
    let perp = Vec2::new(-dir.y, dir.x);
    (
        start + dir * (sn / curvature) + perp * (vs / curvature),
        dir * (1.0 - vs) + perp * sn,
    )
}

/// Distance from `p` to the arc leaving `start` — radially, which is the true distance
/// for a point near the arc and all [`within`](super::flatten::within) needs it to be.
pub(super) fn point_arc_distance(p: Vec2, start: Vec2, arc: &Arc) -> f32 {
    if arc.curvature == 0.0 {
        return point_segment_distance(p, start, start + arc.dir * arc.length);
    }
    let perp = Vec2::new(-arc.dir.y, arc.dir.x);
    let r = 1.0 / arc.curvature;
    let centre = start + perp * r;
    ((p - centre).length() - r.abs()).abs()
}
