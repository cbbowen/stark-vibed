// Shared swept-segment scaffolding for the stamp pass (DESIGN.md §6.2), imported
// by the per-color-space stamp shaders. Everything here is color-space-agnostic:
// the tile transform + prefix-τ bindings, the oriented-quad vertex, and the
// swept optical-depth lookup. Each variant differs only in how the fragment
// turns that depth into channel deposits (see stamp_oklab.wesl; the Mixbox space
// reuses it, since its deposit is the same premultiplied-"over").

struct TileXform {
    // params.xy = tile *texture* top-left (interior origin − apron) in canvas px;
    // params.z = 2 / TILE_TEX (full interior+apron maps to NDC);
    // params.w unused (the reservoir u-advance is per-instance — `SegmentInstance.aux.w`).
    params: vec4<f32>,
    // surf.x = inv surface-tile (canvas px → bump uv); surf.y = tooth;
    // surf.z = add fraction (how much of the brush's own height to lay); surf.w unused.
    surf: vec4<f32>,
}

@group(0) @binding(0) var<uniform> xf: TileXform;
// Prefix-τ volume (DESIGN.md §6.2, §6.6): x = travel, y = lateral, array layer = shape
// orientation (one slice per relative angle, sampled with wrapping). Round tips are
// rotation-invariant and have a single slice; image stamps carry up to
// MAX_ORIENTATION_LAYERS. A 2D array (not a 3D texture) so the mask keeps its full size.
@group(1) @binding(0) var prefix_tex: texture_2d_array<f32>;
// The canvas surface (tileable bump/height), sampled in canvas space so the
// weave gates deposition consistently regardless of which tile a fragment lands
// in (DESIGN.md §6.4).
@group(2) @binding(0) var surface_tex: texture_2d<f32>;
@group(2) @binding(1) var surface_samp: sampler;
// The stroke's deposit reservoir: one column per swept segment (the paint the brush lays
// as it evolves along the stroke) × one row per lateral band across the tip, so each part
// of the brush carries the colour it picked up (DESIGN.md §6.2). For a plain brush it is a
// 1×1 texel of the brush colour. Linear filtering interpolates both axes for free — across
// the tip (bands) and along the stroke (segments) — so this shader is agnostic to the band
// count; the mixer compute pass populates it (deposit·tool + add·brush).
@group(3) @binding(0) var reservoir: texture_2d<f32>;
@group(3) @binding(1) var reservoir_samp: sampler;
// Companion reservoir carrying the tool's **deposited height** in `.x` (≥ 0, a
// per-flatten-step rate; `.yzw` unused) — what the integrate adds to the base. The lift's
// matching removal does not ride this banded texture: the integrate computes it per-pixel
// (`lift × contact` of the exact base height), so an eraser leaves no banded residue
// (DESIGN §6.1/§6.2). Zero for a plain brush, whose flat reservoir lays only the brush's
// own paint.
@group(3) @binding(2) var aux_reservoir: texture_2d<f32>;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) local: vec2<f32>, // brush-local (x in radius units from start, y in [-1,1])
    @location(1) misc: vec4<f32>,  // length/radius, flow, height(thickness rate), wet
    // Reservoir column mapping into the padded texture: u.x = this segment's start
    // column-center; u.y = u-advance per radius travelled, so local.x (radii from the
    // segment start) advances u to the next column at the segment's end — by segment
    // *index*, not arc distance, so non-uniform segment lengths stay aligned.
    @location(2) u: vec2<f32>,
    @location(3) op: f32, // brush opacity at this segment (drives the opacity channel)
    // Shape orientation in turns ∈ [0,1): the relative angle between the shape's native
    // axis and the travel direction, picking the prefix-τ orientation slice (§6.6).
    @location(4) orient: f32,
}

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) aux: vec4<f32>,
}

// Build the oriented quad for one swept segment. The quad spans brush-local
// x ∈ [-1, lr+1], y ∈ [-1, 1] (a radius of margin each side) and is placed onto
// the canvas along the segment's tangent, mapped into the tile's apron'd target.
fn sweep_vertex(
    vi: u32,
    start: vec2<f32>,
    dir: vec2<f32>,   // unit tangent
    geom: vec4<f32>,  // radius, length, flow, reservoir start column (u)
    aux: vec4<f32>,   // height(thickness rate), wet, opacity, reservoir u-advance per radius
    extra: vec4<f32>, // orientation (turns ∈ [0,1)), _, _, _
) -> VsOut {
    let radius = geom.x;
    let lr = geom.y / radius; // segment length in radius units

    let cx = select(-1.0, lr + 1.0, (vi & 1u) != 0u);
    let cy = select(-1.0, 1.0, (vi >> 1u) != 0u);

    let perp = vec2<f32>(-dir.y, dir.x);
    let canvas = start + dir * (cx * radius) + perp * (cy * radius);
    let s = xf.params.z;
    let ndc = (canvas - xf.params.xy) * vec2<f32>(s, -s) + vec2<f32>(-1.0, 1.0);

    var out: VsOut;
    out.clip = vec4<f32>(ndc, 0.0, 1.0);
    out.local = vec2<f32>(cx, cy);
    out.misc = vec4<f32>(lr, geom.z, aux.x, aux.y);
    // u.x = segment-start column-center; u.y = per-segment u-advance per radius (aux.w),
    // so traversing the segment moves exactly one reservoir column.
    out.u = vec2<f32>(geom.w, aux.w);
    out.op = aux.z;
    out.orient = extra.x;
    return out;
}

// The deposited channels for a fragment at brush-local `local` (x = travel in
// radius units from the segment start, y across the tip ∈ [-1,1]). The reservoir
// is parameterized by *segment index* along the stroke (one column per segment —
// where the mixer writes), not by arc distance: `u.x + local.x·u.y` advances from
// this segment's start column-center to the next column over the segment's length
// (`u.y` is the per-segment advance, `aux.w`), so non-uniform segment lengths — the
// short flatten remainder at the stroke end, a curve's varying steps — stay aligned
// with their columns, the colour flows continuously across segments, and overlapping
// segment quads resolve to the same spot (no per-segment clamp). The reservoir's
// walk is extended half a brush past each end (the mixer keeps picking up canvas
// there), so the round caps continue it smoothly and the sampler's edge clamp only
// ever hits the outermost pad column. Linear filtering does the band (y) blend; a
// dry brush's 1×1 reservoir collapses to the brush color.
fn reservoir_color(u: vec2<f32>, local: vec2<f32>) -> vec4<f32> {
    let along = u.x + local.x * u.y;
    return textureSampleLevel(reservoir, reservoir_samp, vec2<f32>(along, local.y * 0.5 + 0.5), 0.0);
}

// The tool's **deposited height** at the reservoir position (per-column rate, ≥ 0; the
// deposit shader spreads it over the footprint and the integrate coverage-normalizes it).
// Zero for a plain brush, whose own height rides the per-segment attribute (DESIGN §6.1).
fn reservoir_dep_height(u: vec2<f32>, local: vec2<f32>) -> f32 {
    let along = u.x + local.x * u.y;
    return textureSampleLevel(aux_reservoir, reservoir_samp, vec2<f32>(along, local.y * 0.5 + 0.5), 0.0).x;
}

// How much of the brush's *own* paint (the per-segment height/wet attributes) the deposit
// lays — the `add` axis, in [0, 1]. The tool deposit (lift/deposit) rides the reservoir
// instead, so this gates only the own-paint contribution.
fn add_frac() -> f32 {
    return xf.surf.z;
}

// Bilinear lookup of the prefix-τ (fp32) at brush-local (xl, yl) ∈ [-1,1]: 0
// before the leading edge, clamp-to-row-total past the trailing edge.
//
// Why this is hand-rolled instead of a ClampToEdge sampler with linear filtering:
//   1. Format. prefix_tex is R32Float (we need fp32 because each fragment takes
//      the *difference* of two prefix sums — see swept_depth — which is sensitive
//      to catastrophic cancellation, and a row's total κ can be large for opaque
//      brushes). Hardware linear filtering of a 32-bit-float texture requires the
//      FLOAT32_FILTERABLE feature, which is optional on WebGPU (our primary
//      target) and not enabled by this device. A filterable 16-bit format would
//      band exactly where the differencing is most sensitive.
//   2. Boundary semantics. The travel (x) axis is asymmetric: 0 before the
//      leading edge, clamp-to-row-total after the trailing edge. A single sampler
//      address mode can't express that — ClampToEdge would read the column-0
//      value across the whole start margin instead of 0 — so the `xl <= -1` guard
//      below would be required regardless.
//   3. Precision. Fixed-function bilinear uses reduced-precision fractional
//      weights on many GPUs; the fp32 `mix` below keeps full precision for the
//      cancellation-sensitive difference.
// Bilinear lookup of one orientation slice `zi` at brush-local (xl, yl). Factored out so
// `prefix_at` can blend two adjacent slices (the orientation axis).
fn prefix_slice(xl: f32, yl: f32, zi: i32) -> f32 {
    let dim = vec2<f32>(textureDimensions(prefix_tex));
    let ux = clamp(xl * 0.5 + 0.5, 0.0, 1.0) * dim.x - 0.5;
    let uy = clamp(yl * 0.5 + 0.5, 0.0, 1.0) * dim.y - 0.5;
    let x0 = floor(ux);
    let y0 = floor(uy);
    let fx = ux - x0;
    let fy = uy - y0;
    let xi0 = clamp(i32(x0), 0, i32(dim.x) - 1);
    let xi1 = clamp(i32(x0) + 1, 0, i32(dim.x) - 1);
    let yi0 = clamp(i32(y0), 0, i32(dim.y) - 1);
    let yi1 = clamp(i32(y0) + 1, 0, i32(dim.y) - 1);
    let p00 = textureLoad(prefix_tex, vec2<i32>(xi0, yi0), zi, 0).r;
    let p10 = textureLoad(prefix_tex, vec2<i32>(xi1, yi0), zi, 0).r;
    let p01 = textureLoad(prefix_tex, vec2<i32>(xi0, yi1), zi, 0).r;
    let p11 = textureLoad(prefix_tex, vec2<i32>(xi1, yi1), zi, 0).r;
    return mix(mix(p00, p10, fx), mix(p01, p11, fx), fy);
}

// Trilinear lookup of the prefix-τ volume at brush-local (xl, yl) ∈ [-1,1] and shape
// orientation `w` (turns ∈ [0,1)). The orientation axis wraps (a full turn is periodic),
// so the two bracketing slices are chosen modulo the slice count. The x,y bilinear is
// hand-rolled for the same reasons (R32Float isn't filterable, the asymmetric travel-edge
// boundary, fp32 precision); 0 before the leading edge, clamp-to-row-total past the
// trailing edge. A single-slice (round) volume collapses to that one slice for every `w`.
fn prefix_at(xl: f32, yl: f32, w: f32) -> f32 {
    if xl <= -1.0 {
        return 0.0;
    }
    let nz = i32(textureNumLayers(prefix_tex));
    let fl = fract(w) * f32(nz);
    let l0f = floor(fl);
    let fw = fl - l0f;
    let z0 = (i32(l0f) % nz + nz) % nz;
    let z1 = (z0 + 1) % nz;
    let a = prefix_slice(xl, yl, z0);
    let b = prefix_slice(xl, yl, z1);
    return mix(a, b, fw);
}

// Swept geometric optical depth over the fragment's segment: the brush footprint
// integrated as it travels, as a difference of prefix sums on the segment's orientation
// slice. Sums exactly across a stroke's overlapping segment quads (under "over" for τ, or
// additively for amounts), reconstructing the continuous path integral.
fn swept_depth(in: VsOut) -> f32 {
    let depth = prefix_at(in.local.x, in.local.y, in.orient) - prefix_at(in.local.x - in.misc.x, in.local.y, in.orient);
    return max(depth, 0.0);
}

// Gate `coverage` by the canvas tooth at the fragment's canvas position (DESIGN.md
// §6.4): paint adheres to the weave's peaks and skips its valleys when light, and
// the effect fades as coverage builds (valleys fill). `frag_xy` is the fragment's
// framebuffer pixel; the tile texture is 1:1 with canvas, so canvas = tex_origin +
// frag_xy.
fn surface_tooth(height: f32, coverage: f32, frag_xy: vec2<f32>) -> f32 {
    // TODO: Find a more natural and physically-accurate way to implement this.
    // While this is a pass-through stub (and with the integrate's tooth gate removed),
    // NO shader reads the canvas surface at stamp/integrate time — the group(2)
    // `surface_tex`/`surface_samp` bindings above are placeholders for this TODO, and
    // the weave currently shows only through the media pass's
    // `thickness = height − surface` and relief normals (DESIGN §6.3/§6.4).
    return coverage;
}
