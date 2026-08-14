// Matte layers in compositing pass A (§15.4): a procedural region
// filled with a flat color, drawn at its own place in the layer stack.
//
// A matte is physically a flat, opaque *coat of paint*, which is what lets it be
// an ordinary layer rather than a special presentation mode:
//
//   · It writes the aux (thickness) target, not just color. The media pass
//     derives visible alpha from the translucent-slab law
//     `1 − exp(−OPACITY_K · color.a · thickness)`, so a matte that wrote only
//     color would be perfectly invisible (§15.4.1).
//   · Its interior is at *constant* height, so its gradient is zero and it lights
//     flat — no weave, and the paint film's uniform sheen (media_common.wesl) reads
//     as an even wash rather than a glint, like a mat board. Its boundary is a
//     height cliff and catches light the way every stroke edge already does.
//   · Both targets blend as premultiplied `over` (declared by the pipeline, NOT
//     the color space's additive aux blend), so an opaque matte *erases* the
//     relief beneath it instead of letting underlying impasto emboss ghost ridges
//     through it (§15.4.2).
//
// Coverage is evaluated analytically from a signed distance at canvas position —
// the same technique as `selection.wesl` — so the matte is exact at any zoom,
// costs no tiles, and is a pure function of canvas position (the §6.4 seam
// invariant, for free).

import package::lib::paint_common::{OPACITY_K};
import package::lib::ramp_common::{ramp_position};

struct View {
    // Canvas px -> NDC, column-major: `st.xy` is the first column, `st.zw` the
    // second. A 2x2 because the view can be turned and mirrored; the matte is a
    // canvas-space rect, so it turns with the canvas.
    st: vec4<f32>,
    xlate: vec4<f32>, // translate.xy, unused zw
    misc: vec4<f32>, // tile_size, interior uv scale, interior uv bias, zoom
}

@group(0) @binding(0) var<uniform> view: View;

// A gradient matte's ramp (§22.4) — per-matte where the view group is
// per-pass, so it rides its own bind group. A solid matte binds a zeroed one:
// `p.x == 0` says "the instance's own channels", so one pipeline serves both.
struct Ramp {
    // .x = stop count (0 = solid; a gradient always has ≥ 2, §22.1);
    // .y = axis kind (0 linear, 1 radial); .zw unused.
    p: vec4<f32>,
    // The axis, canvas px: linear = from.xy, to.xy; radial = center.xy, radius.
    axis: vec4<f32>,
    // Stops: latent .xyz, ramp position t in .w — ascending, endpoints at 0 and
    // 1 (`Gradient`'s invariants). Length is `fill.wesl`'s MAX_GRADIENT_STOPS
    // written out — the mirror generator (§6.10) cannot resolve a named const
    // in a type position.
    stop_c: array<vec4<f32>, 16>,
    // And their residuals (.xyz; .w unused), index-aligned with `stop_c`.
    stop_r: array<vec4<f32>, 16>,
}

@group(1) @binding(0) var<uniform> ramp: Ramp;

// Thickness of the matte slab, in the same units as paint height. With
// `OPACITY_K = 1` a thickness of 8 leaves the slab's visible alpha > 0.999 even
// after the weave-sink discount (media_common.wesl), so a matte at full opacity
// reads solid.
const MATTE_THICKNESS: f32 = 8.0;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) canvas: vec2<f32>,
    @location(1) @interpolate(flat) rect: vec4<f32>,
    @location(2) @interpolate(flat) channels: vec4<f32>,
    @location(3) @interpolate(flat) opacity: f32,
    // One screen pixel in canvas px — the edge antialiasing width. Derived here
    // rather than read from the uniform in the fragment stage, because pass A's
    // view bind group is declared VERTEX-only and this is the matte's only use of
    // it downstream. It is constant across the quad, so a flat varying is exact.
    @location(4) @interpolate(flat) aa: f32,
    @location(5) @interpolate(flat) flags: f32,
    @if(resid) @location(6) @interpolate(flat) resid: vec4<f32>,
}

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) aux: vec4<f32>,
    @if(resid) @location(2) resid: vec4<f32>,
}

// A fullscreen quad — the region is unbounded, so the draw covers the viewport
// and the fragment stage decides coverage. The canvas position is recovered by
// inverting the view's canvas->NDC map; that map is affine, so interpolating the
// corners is exact.
@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    // Per-matte instance: min.xy, max.xy in canvas px.
    @location(0) rect: vec4<f32>,
    // The fill, in the working color space.
    @location(1) channels: vec4<f32>,
    @location(2) opacity: f32,
    // The fill color's **residual** (.xyz), the half of it the space's three channels
    // cannot express (§6.7); .w unused. Zero for every space but Mixbox.
    //
    // An instance attribute declared unconditionally, where the varying and the target
    // below are `@if(resid)`: this parameter list is mirrored into Rust from the
    // unlinked source (§6.10), which has no feature set to evaluate, so a
    // parameter that came and went with the variant would give `MatteInstance` two
    // layouts. Sixteen bytes per matte, against a per-texel channel Oklab still does
    // not allocate.
    @location(3) resid: vec4<f32>,
    // Region kind: 0 = outside `rect` (the frame), 1 = the whole plane
    // (§15.5's ground) — `rect` then carries zeros and is never read.
    @location(4) flags: f32,
) -> VsOut {
    let corner = vec2<f32>(f32(vi & 1u), f32((vi >> 1u) & 1u));
    let ndc = corner * 2.0 - 1.0;

    var out: VsOut;
    out.clip = vec4<f32>(ndc, 0.0, 1.0);
    // Back through the view: the map is affine, so inverting it at the corners and
    // interpolating is exact. The 2x2 inverse is four vertices' worth of arithmetic
    // per matte, against carrying a second matrix in the uniform.
    let m = mat2x2<f32>(view.st.xy, view.st.zw);
    let det = m[0][0] * m[1][1] - m[1][0] * m[0][1];
    let inv = mat2x2<f32>(
        vec2<f32>(m[1][1], -m[0][1]),
        vec2<f32>(-m[1][0], m[0][0]),
    ) * (1.0 / det);
    out.canvas = inv * (ndc - view.xlate.xy);
    out.rect = rect;
    out.channels = channels;
    out.opacity = opacity;
    // `misc.w` is the zoom (screen px per canvas px), so the edge keeps a constant
    // on-screen softness at any zoom.
    out.aa = 1.0 / max(view.misc.w, 1e-6);
    out.flags = flags;
    @if(resid) out.resid = resid;
    return out;
}

// Coverage of the region *outside* `rect` at canvas position `p`, antialiased
// over one screen pixel. `sd` is the signed distance to the rect's boundary,
// negative inside; the ramp is centred on the boundary so the edge lands exactly
// on the rect rather than half a pixel inside or outside it.
fn outside_coverage(p: vec2<f32>, rect: vec4<f32>, aa: f32) -> f32 {
    let d = max(rect.xy - p, p - rect.zw);
    let outer = length(max(d, vec2<f32>(0.0)));
    let inner = min(max(d.x, d.y), 0.0);
    let sd = outer + inner;
    return clamp(sd / aa + 0.5, 0.0, 1.0);
}

@fragment
fn fs_main(in: VsOut) -> FsOut {
    // The whole plane needs no signed distance: its coverage is the statement.
    var region = 1.0;
    if in.flags < 0.5 {
        region = outside_coverage(in.canvas, in.rect, in.aa);
    }
    let a = region * in.opacity;

    // The paint at this fragment: the instance's flat channels, or the ramp
    // read from canvas position (§22.4). Interpolated in the working
    // space between stops already converted on the CPU — an Oklab matte grades
    // the way the library strip previews, a Mixbox one the way pigment mixes.
    // The *thickness* never varies: a graded wash is a transition in color,
    // not in relief, so the slab stays the flat coat §15.4 requires.
    var channels = in.channels.rgb;
    @if(resid) var resd = in.resid.rgb;
    let stops = u32(ramp.p.x);
    if stops >= 2 {
        let t = ramp_position(in.canvas, ramp.axis, ramp.p.y);
        // The segment t falls in: the last stop at or before it — the same
        // walk as `fill.wesl`'s, over the same invariants.
        var j = 0u;
        for (var i = 1u; i + 1u < stops; i++) {
            if ramp.stop_c[i].w <= t {
                j = i;
            }
        }
        let ca = ramp.stop_c[j];
        let cb = ramp.stop_c[j + 1u];
        let span = max(cb.w - ca.w, 1e-6);
        let s = clamp((t - ca.w) / span, 0.0, 1.0);
        channels = mix(ca.xyz, cb.xyz, s);
        @if(resid) resd = mix(ramp.stop_r[j].xyz, ramp.stop_r[j + 1u].xyz, s);
    }

    // Premultiplied, for the `over` blend the pipeline declares on both targets.
    // Color alpha is *coverage*, like every element of pass A (composite.wesl):
    // the matte is fully opaque paint (per-unit opacity 1) of the constant
    // thickness above, so its slab visible alpha is `1 − exp(−K·thickness)`,
    // scaled by how much of the fragment the matte occupies. The aux carries the
    // *amount* (§6.1) with `a` in its own alpha, so an opaque matte
    // still erases the relief beneath it.
    var out: FsOut;
    let w = a * (1.0 - exp(-OPACITY_K * MATTE_THICKNESS));
    out.color = vec4<f32>(channels * w, w);
    out.aux = vec4<f32>(MATTE_THICKNESS * a, 0.0, 0.0, a);
    // The residual rides the same premultiply and the same `over` (§6.7):
    // a matte is a flat coat of one color, and a color is both halves.
    @if(resid) out.resid = vec4<f32>(resd * w, w);
    return out;
}
