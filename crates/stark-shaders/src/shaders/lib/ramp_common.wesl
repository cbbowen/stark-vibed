// The gradient ramp's axis math (§22.4), shared by the fill parcel and the
// matte paint — one definition, so where `t = 0.5` falls cannot drift between
// the two consumers of the same `GradientAxis`.
//
// A binding-free leaf, like every module under `lib/` (§2).

// The two axis kinds a ramp's `axis_kind` lane holds. Declared here rather than
// only described, so the host writes the lane from the generated mirror instead of
// a literal it would have to keep level with this file (§6.10).
const AXIS_LINEAR: f32 = 0.0;
const AXIS_RADIAL: f32 = 1.0;

// Ramp position of canvas point `p` on `axis`, `kind` AXIS_LINEAR (axis =
// from.xy, to.xy) or AXIS_RADIAL (axis = center.xy, radius, unused). Clamped to
// [0, 1] both ways: beyond either end the ramp holds its end stop — a gradient
// covers its whole region, the axis only places the transition. The guards
// keep a degenerate axis (zero length, zero radius) at t = 0 rather than
// dividing by it.
fn ramp_position(p: vec2<f32>, axis: vec4<f32>, kind: f32) -> f32 {
    if kind != AXIS_RADIAL {
        // Linear: project onto the axis; constant on perpendiculars.
        let d = axis.zw - axis.xy;
        let len2 = max(dot(d, d), 1e-6);
        return clamp(dot(p - axis.xy, d) / len2, 0.0, 1.0);
    }
    // Radial: distance from the centre, 1 at the radius.
    return clamp(distance(p, axis.xy) / max(axis.z, 1e-6), 0.0, 1.0);
}
