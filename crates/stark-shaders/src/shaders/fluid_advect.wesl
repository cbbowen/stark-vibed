// Conservative paint advection (DESIGN.md §6.2, "Fluid" — the **drag** axis). One
// ping-pong step over the composited stroke region: the (static, injected) velocity
// field transports the paint, dragging it in the brush's direction.
//
// Why finite-volume, not the old semi-Lagrangian back-trace: every other transport in
// the brush conserves **height** (the amount of paint; DESIGN §6.1), but a semi-
// Lagrangian back-trace does not — it is dissipative and its mass integral drifts, so a
// dragged region slowly gains or loses paint (the canvas "breathes"). This step instead
// updates each cell by the **net flux through its four faces** (a donor-cell / upwind
// finite-volume update). Because adjacent cells compute the *same* shared face from the
// same two cells, the fluxes telescope: Σ height over the region is invariant up to the
// boundary flux, and the injected field is 0 at the footprint edge / in the halo, so the
// boundary flux is 0 → paint is conserved **exactly and locally** (no global rescale,
// which would smear a local loss across the whole region).
//
// Crisp, not mushy: a plain first-order upwind flux is conservative but diffusive (it
// would blur the dragged colour over the iterations). The face value is therefore MUSCL-
// reconstructed with a **minmod** slope limiter — second-order where the field is smooth,
// first-order (TVD, no overshoot/ringing) at edges. Explicit FV needs CFL ≤ 1, so the
// per-step face velocity is clamped to ±CFL texels; high `drag` then moves slightly less
// than its nominal px/iter, which is invisible (drag is an artistic rate, not a calibrated
// distance) and the fixed iteration count keeps replay deterministic.
//
// Each scalar channel is advected as its own conserved density: height & wet (aux.xy) and
// the premultiplied colour (latent·opacity + opacity). Under pure transport that keeps the
// per-unit latent colour ≈ fixed while the *amount* moves; where the field converges paint
// physically piles up (conservative compression), where it diverges it thins.

struct Params {
    knobs: vec4<f32>, // dt (iterations' time step; canvas px per unit velocity), _, _, _
}

// Max texels a face may transport per step. An unsplit 2-D MUSCL update (x- and y-fluxes
// summed from the same pre-step field) is only TVD well below the 1-D limit of 1; pushing
// it toward 1 rings as trailing oscillations behind the moving front. ≤ ~0.5 keeps it
// monotone; the fixed iteration count still carries paint a brush-scale distance.
const CFL: f32 = 0.4;

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var velocity: texture_2d<f32>;
@group(0) @binding(2) var src_color: texture_2d<f32>;
@group(0) @binding(3) var src_aux: texture_2d<f32>;
// Retained so the bind-group layout is unchanged; the finite-volume update uses
// `textureLoad` (integer taps), so no sampler is needed.
@group(0) @binding(4) var samp: sampler;

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) aux: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

// Clamp-to-edge integer load (zero-gradient boundary; the halo's velocity is 0 there so
// no flux crosses the edge regardless).
fn load_color(c: vec2<i32>, dims: vec2<i32>) -> vec4<f32> {
    return textureLoad(src_color, clamp(c, vec2<i32>(0), dims - vec2<i32>(1)), 0);
}
fn load_aux(c: vec2<i32>, dims: vec2<i32>) -> vec4<f32> {
    return textureLoad(src_aux, clamp(c, vec2<i32>(0), dims - vec2<i32>(1)), 0);
}
fn load_vel(c: vec2<i32>, dims: vec2<i32>) -> vec2<f32> {
    return textureLoad(velocity, clamp(c, vec2<i32>(0), dims - vec2<i32>(1)), 0).xy;
}

// Per-component minmod: the smaller-magnitude slope when both agree in sign, else 0.
fn minmod4(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return (sign(a) + sign(b)) * 0.5 * min(abs(a), abs(b));
}

// Donor-cell flux through one face with face velocity `u` (already CFL-clamped). The face
// sits between cells `m` (minus side) and `p` (plus side); `mm`/`pp` are the next cells out
// on each side, for the MUSCL slope. Upwind picks the donor by the sign of `u`.
fn face_flux(u: f32, mm: vec4<f32>, m: vec4<f32>, pp: vec4<f32>, p: vec4<f32>) -> vec4<f32> {
    let left = m + 0.5 * minmod4(m - mm, p - m);   // extrapolated from the minus cell
    let right = p - 0.5 * minmod4(p - m, pp - p);   // extrapolated from the plus cell
    return u * select(right, left, u >= 0.0);
}

// Net flux divergence for one channel set sampled along x (the four x-stencil cells around
// C) — returned as the x contribution `-(F_right - F_left)` to add to C.
fn advect_axis(
    uL: f32, uR: f32,
    ll: vec4<f32>, l: vec4<f32>, c: vec4<f32>, r: vec4<f32>, rr: vec4<f32>,
) -> vec4<f32> {
    let f_right = face_flux(uR, l, c, rr, r);  // face between C(minus) and R(plus)
    let f_left = face_flux(uL, ll, l, r, c);   // face between L(minus) and C(plus)
    return -(f_right - f_left);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> FsOut {
    let dims = vec2<i32>(textureDimensions(src_color));
    let ci = vec2<i32>(floor(frag.xy));
    let dt = p.knobs.x;

    // CFL-clamped face velocities (averaged from the two cells the face divides).
    let vC = load_vel(ci, dims);
    let vL = load_vel(ci + vec2<i32>(-1, 0), dims);
    let vR = load_vel(ci + vec2<i32>(1, 0), dims);
    let vD = load_vel(ci + vec2<i32>(0, -1), dims);
    let vU = load_vel(ci + vec2<i32>(0, 1), dims);
    let uL = clamp(0.5 * (vL.x + vC.x) * dt, -CFL, CFL);
    let uR = clamp(0.5 * (vC.x + vR.x) * dt, -CFL, CFL);
    let uD = clamp(0.5 * (vD.y + vC.y) * dt, -CFL, CFL);
    let uU = clamp(0.5 * (vC.y + vU.y) * dt, -CFL, CFL);

    // Colour: x then y flux divergence (unsplit; both read the pre-step field).
    let c_ll = load_color(ci + vec2<i32>(-2, 0), dims);
    let c_l = load_color(ci + vec2<i32>(-1, 0), dims);
    let c_c = load_color(ci, dims);
    let c_r = load_color(ci + vec2<i32>(1, 0), dims);
    let c_rr = load_color(ci + vec2<i32>(2, 0), dims);
    let c_dd = load_color(ci + vec2<i32>(0, -2), dims);
    let c_d = load_color(ci + vec2<i32>(0, -1), dims);
    let c_u = load_color(ci + vec2<i32>(0, 1), dims);
    let c_uu = load_color(ci + vec2<i32>(0, 2), dims);
    let color = c_c
        + advect_axis(uL, uR, c_ll, c_l, c_c, c_r, c_rr)
        + advect_axis(uD, uU, c_dd, c_d, c_c, c_u, c_uu);

    // Aux (height, wet): same update. Only .xy are stored (Rg16Float); .zw are ignored.
    let a_ll = load_aux(ci + vec2<i32>(-2, 0), dims);
    let a_l = load_aux(ci + vec2<i32>(-1, 0), dims);
    let a_c = load_aux(ci, dims);
    let a_r = load_aux(ci + vec2<i32>(1, 0), dims);
    let a_rr = load_aux(ci + vec2<i32>(2, 0), dims);
    let a_dd = load_aux(ci + vec2<i32>(0, -2), dims);
    let a_d = load_aux(ci + vec2<i32>(0, -1), dims);
    let a_u = load_aux(ci + vec2<i32>(0, 1), dims);
    let a_uu = load_aux(ci + vec2<i32>(0, 2), dims);
    let aux = a_c
        + advect_axis(uL, uR, a_ll, a_l, a_c, a_r, a_rr)
        + advect_axis(uD, uU, a_dd, a_d, a_c, a_u, a_uu);

    var out: FsOut;
    out.color = color;
    out.aux = vec4<f32>(aux.xy, 0.0, 0.0);
    return out;
}
