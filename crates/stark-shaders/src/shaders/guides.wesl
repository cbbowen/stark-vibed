// Drawing guides — the perspective-grid overlay, pass D (§20.4).
//
// One fullscreen pass over the finished, lit image. Every element — the three
// fans, the vanishing lines, the 45° circle, the markers — is an analytic
// distance field evaluated in *canvas* space and converted to *screen* pixels
// for anti-aliasing, so the overlay is crisp at any zoom, rotation or mirror
// and costs no geometry.
//
// The fans draw a **lattice** of cubic cells (§20.3), so that the cells they cut
// on a pair plane are squares and the grid can be measured on. Each axis carries
// two families — one for each pair plane it lies in — and a family is the planes
// through the eye containing that plane's lines parallel to the axis, counted in
// whole cells **from the eye**. With (u, v) the ray resolved on the other two
// axes and p the plane's own distance from the eye in cells, the family is
// `p·u = −i·v` for integer i: one linear level set per guide line, whose gradient
// is exact and whose spacing is |v|/|∇|. Nothing here computes a vanishing point,
// branches on whether one is finite, or depends on where the corner fell.
//
// The lens (§20.8) enters exactly once: the map from the texel to its eye ray.
// Under the stereographic fisheye that map is the inverse stereographic
// projection, and the same fan arithmetic then draws the guide lines as the
// circles they truly image to — nothing else in the pass knows the lens
// changed. The CPU side hands the pair traces (lines or circles), both poles'
// vanishing points, and the dressed rings already derived.

import package::lib::quad::{fullscreen_clip};

struct Guide {
    // Screen px → canvas px: `canvas = m * frag + org.xy` (m column-major).
    inv: vec4<f32>,
    // org.xy, zoom (screen px per canvas px), focal (canvas px).
    org: vec4<f32>,
    // Center of view .xy (canvas px), master opacity, unused.
    cov: vec4<f32>,
    // Lens flag (0 rectilinear, 1 fisheye), 45° ring radius, 90° ring radius
    // (≤ 0 when the lens has none), unused.
    proj: vec4<f32>,
    // The lattice the fans measure out (§20.3): .xyz the corner where its three
    // squared planes meet, in camera space, in cells; .w = 1 when it names a
    // grid at all, 0 when the corner has come to rest at the eye and there is
    // no fan to draw.
    grid: vec4<f32>,
    // World axes in camera space .xyz; .w = that axis's opacity (0 = the axis
    // is nowhere on the canvas, both of its planes being off) — what the
    // markers belonging to an axis alone, its vanishing points, follow.
    dirs: array<vec4<f32>, 3>,
    // .xyz = pair plane k's opacity (0 = off), k spanning axes (k, k+1); .w
    // unused. What everything living on a plane follows: the two fans that rule
    // it, its vanishing trace, its station point. Handed over already resolved,
    // as everything here is — the pass draws numbers, it does not apply rules.
    pairs: vec4<f32>,
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
//
// Declared in Oklab and converted here, which is why they are not round:
// `stark.css`'s `--axis-x/y/z` are the statement of them, all three at one
// lightness (L = 0.667) so no axis reads heavier than the others, and the bar's
// chips wear those variables directly. This pass cannot read a stylesheet, so
// it carries the conversion — held to it by the frontend's
// `the_chips_are_painted_in_the_shader_s_own_axis_hues`, which parses the CSS.
// Four decimals rather than two because the agreement is to within a 1/255
// step, and two would not make it.
const AXIS_X: vec3<f32> = vec3<f32>(0.9349, 0.3629, 0.3803);
const AXIS_Y: vec3<f32> = vec3<f32>(0.2932, 0.6746, 0.3667);
const AXIS_Z: vec3<f32> = vec3<f32>(0.3922, 0.5631, 0.9544);
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
// Every plane recedes in the periphery, and the more squarely it faces the eye
// the sooner (`plane_fade`). Distance is in units of the 45° ring: a plane whose
// normal *is* the view axis holds inside PLANE_LO of it and is gone by PLANE_HI,
// and one the view axis lies in recedes at PLANE_BASE of that rate — far enough
// out that the substrate under a drawing is never the thing that goes.
const PLANE_LO: f32 = 0.18;
const PLANE_HI: f32 = 0.50;
const PLANE_BASE: f32 = 0.25;
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

/// How much of a plane's grid survives at `periph` — the texel's distance from
/// the center of view in units of the 45° ring (§20.3).
///
/// A plane the view axis is **normal** to repeats across the whole canvas
/// without ever foreshortening: out in the periphery it is saying nothing the
/// middle did not already say, and it says it over everything else. A plane the
/// view axis **lies in** is doing the opposite out there — running away toward
/// its own vanishing line, where every cell is a fresh statement — so it recedes
/// at a fraction of the rate and only well past the useful cone.
///
/// `face` is what sets the rate: the plane normal's component along the view
/// axis, which for the plane two axes span is just the third axis's own `z`. One
/// pose reads the whole scale — look straight down an axis and the picture-plane
/// wall thins to a patch around the center while the floor and side wall, whose
/// lines all converge, stay.
///
/// A legibility rule rather than a theorem, and the second the overlay spends on
/// the fans after the moiré guard — which it deliberately does not duplicate:
/// that one answers "these lines are too close to resolve", this one answers
/// "these lines are too far out to mean anything". Stated in units of the 45°
/// ring so it names the same visual angle at any focal length and under either
/// lens, and so the bound it leans on is one the guide already draws.
fn plane_fade(face: f32, periph: f32) -> f32 {
    let rate = mix(PLANE_BASE, 1.0, abs(face));
    return 1.0 - smoothstep(PLANE_LO, PLANE_HI, rate * periph);
}

/// Coverage of one family of guide lines: the lattice planes `w = i·den` for
/// integer `i` (§20.3), at a texel where `w` and `den` change by `dw` and
/// `dden` per screen px.
///
/// Taken as the level set of `w − i·den` rather than as `|w/den − i|`, so the
/// distance stays exact where the quotient runs away, and the two are computed
/// from the *same* gradient — which is also the family's spacing, and so drives
/// the moiré fade for free. Both sides are homogeneous in the ray's length, so
/// the fisheye's unnormalized ray cancels to first order, which is all an
/// antialiasing width needs.
///
/// Nothing where `den` vanishes: that is the plane's own vanishing line, where
/// the whole plane images onto one line and no texel is *between* cells. And
/// nothing where the level set has no **gradient**: a constant function has no
/// curve in it, so a family whose plane contains the eye — a zero in that
/// component of the lattice — draws no line here rather than a line everywhere.
/// Both tests reject a NaN rather than admitting one, so no indeterminate
/// reaches the accumulator.
fn fan_cov(w: f32, den: f32, dw: vec2<f32>, dden: vec2<f32>) -> f32 {
    if !(abs(den) > 1e-9) { return 0.0; }
    let i = round(w / den);
    let grad = length(dw - i * dden);
    let spacing = abs(den) / grad;
    if !(spacing < 1e6) { return 0.0; }
    return stroke_cov(abs(w - i * den) / grad, FAN_HW)
        * smoothstep(SPACING_LO, SPACING_HI, spacing);
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
    return fullscreen_clip(vi);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let m = mat2x2<f32>(guide.inv.xy, guide.inv.zw);
    let p = m * frag.xy + guide.org.xy; // canvas px
    let zoom = guide.org.z;
    let f = guide.org.w;
    let cov = guide.cov.xy;
    let master = guide.cov.z;

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

    // How far out this texel is, in units of the 45° ring — the guide's own
    // statement of how far out is far out (§20.1), and the one radius that names
    // the same visual angle at any focal length and under either lens.
    let periph = length(p - cov) / max(guide.proj.y, 1e-6);

    var acc = vec4<f32>(0.0);

    // --- The six fans (§20.3), faintest layer first. -----------------------
    //
    // Two families per axis, one for each pair plane the axis lies in — which
    // is what it costs to have every plane's cells be squares: a family that is
    // evenly spaced on one of an axis's two planes is a reciprocal run on the
    // other, so three families can square one plane and six square all three.
    //
    // No varying-dependent branch may precede the derivative calls: WebGPU's
    // uniformity analysis requires them in uniform control flow, and only the
    // loop bounds and uniform-buffer reads are uniform here. The degenerate
    // texels — a plane's own vanishing line — are handled by value, inside
    // `fan_cov`, after the derivatives are already in hand.
    for (var i = 0u; i < 3u; i++) {
        let j = (i + 1u) % 3u;
        let k = (i + 2u) % 3u;
        // The eye ray resolved on the *other* two axes. Identically `n·a_k` and
        // `n·a_j` for n = a × r, one cross product cheaper: the pencil of planes
        // through the eye containing `a` is named by the ray's component across
        // it, and `(u, v)` is that direction projectively.
        let u = dot(ray, guide.dirs[j].xyz);
        let v = -dot(ray, guide.dirs[k].xyz);
        let du = vec2<f32>(dpdx(u), dpdy(u));
        let dv = vec2<f32>(dpdx(v), dpdy(v));
        // How far each of this axis's two planes lies from the eye, in cells —
        // all the lattice contributes, because the grid's *phase* is the eye
        // rather than the corner (§20.3). The line `m` cells from the viewer on
        // the plane at `p` satisfies `p·u = −m·v`, so the whole family is that
        // level set and the corner never enters it. (`dot` with the axis rather
        // than a component read, since the lattice arrives turned into camera
        // space with everything else; a rotation makes the two the same number.)
        let pj = dot(guide.grid.xyz, guide.dirs[j].xyz);
        let pk = dot(guide.grid.xyz, guide.dirs[k].xyz);
        // A family is a set of lines *in a plane*, so it follows that plane's
        // own flag, exactly as the plane's vanishing trace and station point do
        // below. Switching a plane off takes both of the families ruling it —
        // one from each of its two axes — and touches nothing else those axes
        // do elsewhere. The pair spanned with the successor is pair `i`, the one
        // spanned with the predecessor is pair `k`.
        //
        // The plane's obliquity fade rides in the same product, keyed on the one
        // axis *not* in the plane, which is its normal (`plane_fade`).
        let lattice = guide.grid.w * master;
        let a_ij = guide.pairs[i] * plane_fade(guide.dirs[k].z, periph) * lattice;
        let a_ik = guide.pairs[k] * plane_fade(guide.dirs[j].z, periph) * lattice;
        // Stepping across the plane spanned with the successor axis (whose own
        // distance from the eye is `pk`, its normal being the predecessor), then
        // across the other: the two planes this axis is a side of.
        acc = over(acc, axis_color(i), fan_cov(-pk * u, v, -pk * du, dv) * a_ij);
        acc = over(acc, axis_color(i), fan_cov(-pj * v, u, -pj * dv, du) * a_ik);
    }

    // --- Vanishing traces: each pair's horizon, straight or bowed (§20.2,
    // §20.8) — a line or a circle by the slot's kind, one distance either way.
    for (var k = 0u; k < 3u; k++) {
        let kind = guide.lines[k].w;
        if kind < 0.5 { continue; }
        let pair_a = guide.pairs[k];
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
        let pair_a = guide.pairs[k];
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
