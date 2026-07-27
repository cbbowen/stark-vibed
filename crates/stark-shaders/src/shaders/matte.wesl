// Matte layers in compositing pass A (FRAME_DESIGN.md §4): a procedural region
// filled with a flat colour, drawn at its own place in the layer stack.
//
// A matte is physically a flat, opaque *coat of paint*, which is what lets it be
// an ordinary layer rather than a special presentation mode:
//
//   · It writes the aux (thickness) target, not just colour. The media pass
//     derives visible alpha from the translucent-slab law
//     `1 − exp(−OPACITY_K · color.a · thickness)`, so a matte that wrote only
//     colour would be perfectly invisible (FRAME_DESIGN.md §4.1).
//   · Its interior is at *constant* height, so its gradient is zero and it lights
//     flat and matte — no weave, no gloss, like a mat board. Its boundary is a
//     height cliff and catches light the way every stroke edge already does.
//   · Both targets blend as premultiplied `over` (declared by the pipeline, NOT
//     the colour space's additive aux blend), so an opaque matte *erases* the
//     relief beneath it instead of letting underlying impasto emboss ghost ridges
//     through it (FRAME_DESIGN.md §4.2).
//
// Coverage is evaluated analytically from a signed distance at canvas position —
// the same technique as `selection.wesl` — so the matte is exact at any zoom,
// costs no tiles, and is a pure function of canvas position (the §6.4 seam
// invariant, for free).

struct View {
    st: vec4<f32>,   // scale.xy, translate.xy  (canvas px -> NDC)
    misc: vec4<f32>, // tile_size, interior uv scale, interior uv bias, zoom
}

@group(0) @binding(0) var<uniform> view: View;

// Thickness of the matte slab, in the same units as paint height. With
// `OPACITY_K = 1` (media_common.wesl) a thickness of 8 leaves `vis > 0.999` even
// after the canvas surface height (<= ~0.6) is subtracted, so a matte at full
// opacity reads solid.
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
}

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) aux: vec4<f32>,
}

// A fullscreen quad — the region is unbounded, so the draw covers the viewport
// and the fragment stage decides coverage. The canvas position is recovered by
// inverting the view's canvas->NDC map; that map is affine, so interpolating the
// corners is exact.
@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @location(0) rect: vec4<f32>,
    @location(1) channels: vec4<f32>,
    @location(2) opacity: f32,
) -> VsOut {
    let corner = vec2<f32>(f32(vi & 1u), f32((vi >> 1u) & 1u));
    let ndc = corner * 2.0 - 1.0;

    var out: VsOut;
    out.clip = vec4<f32>(ndc, 0.0, 1.0);
    out.canvas = (ndc - view.st.zw) / view.st.xy;
    out.rect = rect;
    out.channels = channels;
    out.opacity = opacity;
    // `misc.w` is the zoom (screen px per canvas px), so the edge keeps a constant
    // on-screen softness at any zoom.
    out.aa = 1.0 / max(view.misc.w, 1e-6);
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
    let a = outside_coverage(in.canvas, in.rect, in.aa) * in.opacity;

    // Premultiplied, for the `over` blend the pipeline declares on both targets.
    // Colour alpha is the paint's *per-unit* opacity and the aux carries the
    // *amount* (DESIGN.md §6.1); the matte is fully opaque paint, so its per-unit
    // opacity is 1 and its thickness is the constant above.
    var out: FsOut;
    out.color = vec4<f32>(in.channels.rgb * a, a);
    out.aux = vec4<f32>(MATTE_THICKNESS * a, 0.0, 0.0, a);
    return out;
}
