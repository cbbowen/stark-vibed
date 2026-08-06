// Texture-reading patterns shared across passes that bind different things.
//
// Under `lib/`, so **nothing here declares a binding**: every function takes the
// texture it reads as an argument. That is what lets `fill`, `integrate`,
// `transform`, `selection` and `dynamics` — five passes with five unrelated bind
// groups — call the same code.

// ---- the 1×1-or-real-tile load (§6.8) -------------------------------------------
//
// A load clamped to the bound texture's own extent, so a real tile and a 1×1
// constant read identically.
//
// The pattern is everywhere because the engine leans on it deliberately: rather
// than branch on "is there a mask / a base tile / a parcel here", the host binds a
// 1×1 texture holding the value that reigns where there is no tile
// (`SelectionRenderer::mask_for`, and the zero base a stroke onto bare canvas
// gets), and the shader reads both through one path. A virgin destination and a
// fully-masked one are then the same program, which is one fewer way for the two
// to disagree.
//
// It was written out nine times before this — `load1` in `fill` and `integrate`,
// open-coded in `transform` (five times), `dynamics` (twice) and `selection` —
// with four separate comments calling it "§6.8's pattern". This is that pattern.
//
// A no-op for a real tile bound 1:1 with its destination, where `c` is always in
// range; the clamp exists for the 1×1 case and costs a `min`.
fn load1(t: texture_2d<f32>, c: vec2<i32>) -> vec4<f32> {
    let dim = vec2<i32>(textureDimensions(t));
    return textureLoad(t, min(c, dim - vec2<i32>(1, 1)), 0);
}

// ---- hand-rolled bilinear ------------------------------------------------------
//
// The four texels and two weights of a bilinear tap, computed once so the two
// places that hand-roll one (`stamp_common::prefix_slice` over the prefix-τ
// volume, `dynamics::bake_at` over the reservoir's swept prefix) share the
// arithmetic rather than the comment. They cannot share the *load* — one reads a
// `texture_2d_array` slice and the other a `texture_2d`, and WESL has no generic
// over that — so the split is: this module works out *where* to tap, the caller
// taps, and `bilerp_mix` combines.
//
// Why hand-rolled at all, rather than a linear-filtering sampler. Both callers
// difference two of these against each other, which is sensitive to catastrophic
// cancellation, and both read fp32 targets for exactly that reason. Three things
// follow:
//
//   1. **Format.** Hardware linear filtering of a 32-bit float texture requires
//      the FLOAT32_FILTERABLE feature, which is optional on WebGPU — our primary
//      target — and not enabled by this device. A filterable 16-bit format would
//      band precisely where the differencing is most sensitive.
//   2. **Precision.** Fixed-function bilinear uses reduced-precision fractional
//      weights on many GPUs. The fp32 `mix` below keeps full precision for the
//      difference, and — because the weights are then a pure function of the
//      inputs rather than of the driver — keeps the tap bit-reproducible across
//      adapters, which replay and the goldens rest on (§12.1).
//   3. **Boundary semantics.** Both callers want an *asymmetric* travel axis: 0
//      before the leading edge, clamp-to-row-total after the trailing one. No
//      single sampler address mode expresses that, so each caller guards its own
//      leading edge before calling here and relies on the clamp below for the
//      trailing one.

struct Bilerp {
    // The lower and upper texel of each axis, clamped into the texture.
    i0: vec2<i32>,
    i1: vec2<i32>,
    // The fractional weight along each axis.
    f: vec2<f32>,
}

// Where a bilinear tap at `local` ∈ [-1,1]² lands in a texture of size `dim`.
// Out-of-range coordinates clamp to the edge texel on both axes, which is what
// gives the trailing edge its clamp-to-row-total.
fn bilerp_taps(dim: vec2<f32>, local: vec2<f32>) -> Bilerp {
    let u = clamp(local * 0.5 + vec2<f32>(0.5, 0.5), vec2<f32>(0.0), vec2<f32>(1.0)) * dim
        - vec2<f32>(0.5, 0.5);
    let base = floor(u);
    let hi = vec2<i32>(dim) - vec2<i32>(1, 1);
    var out: Bilerp;
    out.i0 = clamp(vec2<i32>(base), vec2<i32>(0, 0), hi);
    out.i1 = clamp(vec2<i32>(base) + vec2<i32>(1, 1), vec2<i32>(0, 0), hi);
    out.f = u - base;
    return out;
}

// The four taps combined, in the caller's own component count. Both are the plain
// separable form — mix along x at each row, then between the rows.
fn bilerp_mix(b: Bilerp, p00: vec4<f32>, p10: vec4<f32>, p01: vec4<f32>, p11: vec4<f32>) -> vec4<f32> {
    return mix(mix(p00, p10, b.f.x), mix(p01, p11, b.f.x), b.f.y);
}

fn bilerp_mix1(b: Bilerp, p00: f32, p10: f32, p01: f32, p11: f32) -> f32 {
    return mix(mix(p00, p10, b.f.x), mix(p01, p11, b.f.x), b.f.y);
}

// ---- analytic coverage ----------------------------------------------------------

// Coverage of an axis-aligned box with a half-pixel antialiasing ramp: exactly 1
// deeper than half a pixel inside, exactly 0 outside, a 1-px ramp on the boundary.
//
// A pure function of canvas position, which is what lets the two sides of a
// rect-scoped transform agree: `transform.wesl`'s parcel pass scales what it *cuts*
// by this gate at the source, and its combine pass scales what it *keeps* by the
// same gate at the destination, so an identity map recombines exactly (§16.4, §16.8).
// Two evaluations of one function rather than two spellings of one idea.
fn box_coverage(p: vec2<f32>, lo: vec2<f32>, hi: vec2<f32>) -> f32 {
    let d = min(p - lo, hi - p);
    return clamp(min(d.x, d.y) + 0.5, 0.0, 1.0);
}
