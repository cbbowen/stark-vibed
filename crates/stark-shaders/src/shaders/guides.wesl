// Drawing guides — the perspective-grid overlay, pass D (§20.4).
//
// One fullscreen pass over the finished, lit image. Every element — the three
// fans, the vanishing lines, the 45° circle, the markers — is an analytic
// distance field evaluated in *canvas* space and converted to *screen* pixels
// for anti-aliasing, so the overlay is crisp at any zoom, rotation or mirror
// and costs no geometry.
//
// The fan for axis `a` is the pencil of planes through the eye containing `a`,
// stepped by equal *visual angle* (§20.3). A texel's pencil coordinate is
// θ = atan2(n·b1, n·b0) with n = a × r (r = the eye's ray through the texel)
// and (b0, b1) the other two axes — which lands every pair plane (every
// vanishing trace) on a quarter turn of the fan. The gradient of θ is taken
// analytically from the atan2 quotient rather than with fwidth(θ), because θ
// has a branch cut and its true gradient does not.
//
// The lens (§20.8) enters exactly once: the map from the texel to its eye ray.
// Under the stereographic fisheye that map is the inverse stereographic
// projection, and the same fan arithmetic then draws the guide lines as the
// circles they truly image to — nothing else in the pass knows the lens
// changed. The CPU side hands the pair traces (lines or circles), both poles'
// vanishing points, and the dressed rings already derived.

struct Guide {
    // Screen px → canvas px: `canvas = m * frag + org.xy` (m column-major).
    inv: vec4<f32>,
    // org.xy, zoom (screen px per canvas px), focal (canvas px).
    org: vec4<f32>,
    // Center of view .xy (canvas px), fan step (radians), master opacity.
    cov: vec4<f32>,
    // Lens flag (0 rectilinear, 1 fisheye), 45° ring radius, 90° ring radius
    // (≤ 0 when the lens has none), unused.
    proj: vec4<f32>,
    // World axes in camera space .xyz; .w = that axis's fan opacity (0 = off).
    dirs: array<vec4<f32>, 3>,
    // Vanishing trace of pair (k, k+1), by .w: 0 = none; 1 = line (unit normal
    // .xy, offset .z — evaluating gives signed canvas px); 2 = circle (center
    // .xy, radius .z).
    lines: array<vec4<f32>, 3>,
    // Vanishing points: .xy canvas px, .z = valid. Slots 0–2 are the axes'
    // forward poles, 3–5 the backward — which only the fisheye separates.
    vps: array<vec4<f32>, 6>,
    // Station points of pair (k, k+1): .xy canvas px, .z = valid.
    sps: array<vec4<f32>, 3>,
}

@group(0) @binding(0) var<uniform> guide: Guide;

// Axis hues (display sRGB — this pass draws over the already-encoded image):
// the X/Y/Z semantics every 3D tool taught, softened to sit over paint.
const AXIS_X: vec3<f32> = vec3<f32>(0.91, 0.34, 0.36);
const AXIS_Y: vec3<f32> = vec3<f32>(0.33, 0.71, 0.40);
const AXIS_Z: vec3<f32> = vec3<f32>(0.38, 0.55, 0.94);
// Neutral chrome (center of view, 45° circle) and the dark halo that keeps it
// legible over paint of any value.
const CHROME: vec3<f32> = vec3<f32>(0.96, 0.96, 0.96);
const HALO: vec3<f32> = vec3<f32>(0.09, 0.09, 0.11);

// Line half-widths, screen px.
const FAN_HW: f32 = 0.6;
const PAIR_HW: f32 = 0.9;
const CIRCLE_HW: f32 = 0.8;
// Fan lines fade out as their spacing closes on the pixel grid (moiré guard),
// fully gone below SPACING_LO screen px, fully present above SPACING_HI.
const SPACING_LO: f32 = 3.0;
const SPACING_HI: f32 = 9.0;
// Dash period of the 45° circle, screen px of arc.
const DASH: f32 = 10.0;

fn axis_color(i: u32) -> vec3<f32> {
    if i == 0u { return AXIS_X; }
    if i == 1u { return AXIS_Y; }
    return AXIS_Z;
}

/// Coverage of a stroke of half-width `hw` at `d` screen px from its spine.
fn stroke_cov(d: f32, hw: f32) -> f32 {
    return 1.0 - smoothstep(hw - 0.5, hw + 0.5, d);
}

/// Coverage of a filled disc of radius `r` at `d` screen px from its center.
fn disc_cov(d: f32, r: f32) -> f32 {
    return 1.0 - smoothstep(r - 0.5, r + 0.5, d);
}

/// `rgb` at `a`, drawn OVER the premultiplied accumulator.
fn over(acc: vec4<f32>, rgb: vec3<f32>, a: f32) -> vec4<f32> {
    let aa = clamp(a, 0.0, 1.0);
    return vec4<f32>(rgb * aa + acc.rgb * (1.0 - aa), aa + acc.a * (1.0 - aa));
}

/// One dashed chrome ring about the center of view — the 45° and 90° cones'
/// images (§20.1, §20.8). A non-positive radius is a ring the lens does not
/// have, and draws nothing.
fn ring(acc: vec4<f32>, p: vec2<f32>, cov: vec2<f32>, radius: f32, zoom: f32, master: f32) -> vec4<f32> {
    if radius <= 0.0 { return acc; }
    let dp = p - cov;
    let rad = length(dp);
    let d_px = abs(rad - radius) * zoom;
    // Dash by arc length at the ring's own radius, screen px. The dash dims
    // rather than vanishes, so the ring never falls apart at a
    // glance-and-back. The `select` keeps atan2's indeterminate 0/0 (the
    // exact center texel, nowhere near the ring) from minting a NaN that
    // `0 × NaN` would carry into the accumulator.
    let arc = select(atan2(dp.y, dp.x), 0.0, rad < 0.5) * radius * zoom;
    let duty = mix(0.35, 1.0, step(0.5, fract(arc / DASH)));
    let haloed = over(acc, HALO, stroke_cov(d_px, CIRCLE_HW + 0.8) * duty * 0.4 * master);
    return over(haloed, CHROME, stroke_cov(d_px, CIRCLE_HW) * duty * 0.8 * master);
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    // Fullscreen triangle: (-1,-1), (3,-1), (-1,3).
    let corner = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    return vec4<f32>(corner * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let m = mat2x2<f32>(guide.inv.xy, guide.inv.zw);
    let p = m * frag.xy + guide.org.xy; // canvas px
    let zoom = guide.org.z;
    let f = guide.org.w;
    let cov = guide.cov.xy;
    let step_r = guide.cov.z;
    let master = guide.cov.w;

    // The eye's ray through this texel, camera space — the one place the lens
    // enters (§20.8). Unnormalized both ways: every use below is projective,
    // so scale cancels (the fisheye ray's true length is 1 + s). The branch is
    // on a uniform value, so the derivative calls below remain in uniform
    // control flow.
    var ray: vec3<f32>;
    if guide.proj.x > 0.5 {
        // Inverse stereographic: past the 90° ring (s > 1) the z component
        // goes negative and the texel sees *behind* the camera.
        let u = (p - cov) / (2.0 * f);
        let s = dot(u, u);
        ray = vec3<f32>(2.0 * u, 1.0 - s);
    } else {
        ray = vec3<f32>((p - cov) / f, 1.0);
    }

    var acc = vec4<f32>(0.0);

    // --- The three fans (§20.3), faintest layer first. ---------------------
    //
    // No varying-dependent branch may precede the derivative calls: WebGPU's
    // uniformity analysis requires them in uniform control flow, and only the
    // loop bounds and uniform-buffer reads are uniform here. Degenerate texels
    // (the axis's own vanishing point, where the pencil coordinate collapses)
    // are handled by value instead — the spacing fade reaches zero there, and
    // the `select` keeps atan2's indeterminate 0/0 out of the accumulator.
    for (var i = 0u; i < 3u; i++) {
        let a = guide.dirs[i].xyz;
        let n = cross(a, ray);
        let u = dot(n, guide.dirs[(i + 2u) % 3u].xyz);
        let v = dot(n, guide.dirs[(i + 1u) % 3u].xyz);
        // |∇θ| per screen px, from the atan2 quotient — continuous where θ
        // itself wraps, which fwidth(θ) is not.
        let gx = u * dpdx(v) - v * dpdx(u);
        let gy = u * dpdy(v) - v * dpdy(u);
        let den = max(u * u + v * v, 1e-10);
        let grad = max(length(vec2<f32>(gx, gy)) / den, 1e-8);
        let theta = select(atan2(v, u), 0.0, u * u + v * v < 1e-10);
        let t = theta / step_r;
        let d_px = abs(t - round(t)) * step_r / grad;
        let spacing = step_r / grad;
        let fade = smoothstep(SPACING_LO, SPACING_HI, spacing);
        let alpha = stroke_cov(d_px, FAN_HW) * fade * guide.dirs[i].w * master;
        acc = over(acc, axis_color(i), alpha);
    }

    // --- Vanishing traces: each pair's horizon, straight or bowed (§20.2,
    // §20.8) — a line or a circle by the slot's kind, one distance either way.
    for (var k = 0u; k < 3u; k++) {
        let kind = guide.lines[k].w;
        if kind < 0.5 { continue; }
        let pair_a = min(guide.dirs[k].w, guide.dirs[(k + 1u) % 3u].w);
        if pair_a <= 0.0 { continue; }
        var d_px: f32;
        if kind > 1.5 {
            d_px = abs(length(p - guide.lines[k].xy) - guide.lines[k].z) * zoom;
        } else {
            d_px = abs(dot(guide.lines[k].xy, p) + guide.lines[k].z) * zoom;
        }
        let rgb = mix(axis_color(k), axis_color((k + 1u) % 3u), 0.5);
        acc = over(acc, rgb, stroke_cov(d_px, PAIR_HW) * 0.9 * pair_a * master);
    }

    // --- The view-cone rings: 45° always, 90° where the lens has one
    // (§20.1, §20.8).
    acc = ring(acc, p, cov, guide.proj.y, zoom, master);
    acc = ring(acc, p, cov, guide.proj.z, zoom, master);

    // --- Station points: rings on the Thales circle (§20.2). ---------------
    for (var k = 0u; k < 3u; k++) {
        if guide.sps[k].z <= 0.0 { continue; }
        let pair_a = min(guide.dirs[k].w, guide.dirs[(k + 1u) % 3u].w);
        if pair_a <= 0.0 { continue; }
        let d = length(p - guide.sps[k].xy) * zoom;
        let rgb = mix(axis_color(k), axis_color((k + 1u) % 3u), 0.5);
        acc = over(acc, HALO, stroke_cov(abs(d - 3.2), 2.0) * 0.6 * pair_a * master);
        acc = over(acc, rgb, stroke_cov(abs(d - 3.2), 1.1) * pair_a * master);
        acc = over(acc, rgb, disc_cov(d, 0.9) * pair_a * master);
    }

    // --- Vanishing points: filled discs in their axis's color — both poles,
    // where the lens separates them (§20.8).
    for (var i = 0u; i < 6u; i++) {
        let axis = i % 3u;
        if guide.vps[i].z <= 0.0 || guide.dirs[axis].w <= 0.0 { continue; }
        let d = length(p - guide.vps[i].xy) * zoom;
        acc = over(acc, HALO, disc_cov(d, 5.2) * 0.6 * master);
        acc = over(acc, axis_color(axis), disc_cov(d, 4.0) * master);
        acc = over(acc, CHROME, stroke_cov(abs(d - 4.0), 0.7) * 0.9 * master);
    }

    // --- Center of view: a canvas-aligned crosshair, topmost. --------------
    {
        let q = abs((p - cov) * zoom);
        let arm = 6.0;
        let cross_d = min(max(q.x - 1.6, q.y - arm), max(q.y - 1.6, q.x - arm));
        acc = over(acc, HALO, (1.0 - smoothstep(-0.5, 0.5, cross_d)) * 0.6 * master);
        let core = min(max(q.x - 0.7, q.y - arm + 0.9), max(q.y - 0.7, q.x - arm + 0.9));
        acc = over(acc, CHROME, (1.0 - smoothstep(-0.5, 0.5, core)) * master);
    }

    // Premultiplied out; the pipeline blends `src + dst·(1 − src.a)`.
    return acc;
}
