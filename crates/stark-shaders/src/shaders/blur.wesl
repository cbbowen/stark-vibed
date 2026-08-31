// The focal blur's **FFT convolution** (§21.12): the machinery that turns the
// accumulator, already decoded into XYZ light by the space's own `fs_blur_decode`
// (`filter_oklab.wesl` / `filter_mixbox.wesl`), into that light spread through the
// aperture's circle of confusion. Space-independent, like `resolve.wesl`: by the
// time anything here runs, a texel is light, and light is the one thing every
// space agrees about.
//
//
// WHY AN FFT AT ALL
//
// A blur is out(x) = Σ_d k(d) · f(x − d), and a lens's kernel is a filled disc —
// not separable, and at a painterly radius it is tens of thousands of taps per
// texel. The convolution theorem does the whole thing in O(N log N) whatever the
// kernel's size *or shape*: transform the image, multiply by the kernel's own
// transform, transform back. The shape being free is the point — the disc below
// is one `make_kernel` away from a user-drawn aperture, and nothing else here
// would change (§21.12).
//
//
// WHAT RIDES THE TRANSFORM
//
// Six real channels, packed as three complex planes across two textures:
//
//     plane0 = X·a + i·(Y·a)     `t0.xy`   premultiplied XYZ light —
//     plane1 = Z·a + i·(a)       `t0.zw`   the sum a blur takes is a sum of light,
//                                          weighed by the coverage it arrived with
//     plane2 = h  + i·(w)        `t1.xy`   the height (paint travels with the light
//                                          that shows it, §21.10's transport rule)
//                                          and the border weight `w` (see below)
//
// The packing is free, not a two-for-one trick to untangle: convolution is linear
// and the kernel is real, so (f + i·g) ⊛ k = f⊛k + i·(g⊛k) exactly — each packed
// pair comes back as its two blurred halves in place.
//
// `w` is 1 where the accumulator is and 0 in the padding, and the resolve divides
// every other channel by the blurred `w`. That is the chromatic filter's partition
// of unity generalized (§21.10): wherever the whole kernel lands on real image,
// w̄ = Σk and the division makes the kernel exactly normalized — for any kernel,
// with no constant to tune — and at the viewport rim it renormalizes over the taps
// that exist, which is the same honesty the chromatic gather buys by clamping to
// the rim. It is also why `make_kernel` never normalizes the disc: the scale
// cancels in the division.
//
//
// THE TRANSFORM
//
// Stockham radix-2, out-of-place, autosorting — no bit-reversal pass, and every
// pass is the same dispatch shape, which is what lets the host drive the whole
// plan as a list of (span, axis, direction) uniforms (`gpu/composite/blur.rs`).
// One pass: butterfly `t` of a line pairs in[t] with in[t + n/2],
//
//     out[(t / ns)·2ns + (t % ns)]      = a + wᵗ·b
//     out[(t / ns)·2ns + (t % ns) + ns] = a − wᵗ·b,    wᵗ = exp(dir·2πi·(t % ns)/2ns)
//
// with the span `ns` doubling 1, 2, …, n/2. `dir` is −1 forward and +1 inverse;
// the inverse's 1/(W·H) is folded into `apply_kernel` so no pass scales.
//
// Sizes are powers of two, padded past the image by the kernel radius on each
// side so the circular convolution's wrap-around lands in zeroed padding rather
// than carrying one edge of the picture into the other (§21.12).

// One FFT dispatch's worth of plan — which pass of which transform this is.
//
// The transform's extent rides in the uniform rather than being read off the
// textures, because it is **this layer's** extent, not the planes': a decimated
// layer (§21.12) transforms a power-of-two subregion at the planes' origin, and
// the planes themselves are sized for the largest layer in the frame.
struct Fft {
    // The transform's extent: this layer's padded convolution size. Texels past
    // it are never read or written by any pass of its chain.
    nx: u32,
    ny: u32,
    // This pass's Stockham span: 1, 2, …, n/2 along `axis`.
    ns: u32,
    // Which axis the butterflies run along: 0 rows, 1 columns.
    axis: u32,
    // −1 forward, +1 inverse — the sign of the twiddle's exponent.
    dir: f32,
    // `make_kernel` only: the disc's radius in convolution texels. The one knob
    // the kernel has, until it is an aperture image (§21.12).
    radius: f32,
    // `apply_kernel` only: 1/(nx·ny), the inverse transform's scale, folded into
    // the one pass that touches every plane anyway.
    inv_n: f32,
}

@group(0) @binding(0) var<uniform> f: Fft;
// The half of the ping-pong this pass reads…
@group(0) @binding(1) var src_light: texture_2d<f32>;
@group(0) @binding(2) var src_aux: texture_2d<f32>;
// …the kernel's transform, read by `apply_kernel` alone (a dummy texture rides
// here on the kernel-FFT passes whose destination *is* the kernel — one slot may
// not be bound as both texture and storage in one dispatch)…
@group(0) @binding(3) var kernel: texture_2d<f32>;
// …and the half it writes.
@group(0) @binding(4) var dst_light: texture_storage_2d<rgba32float, write>;
@group(0) @binding(5) var dst_aux: texture_storage_2d<rg32float, write>;

// The workgroup edge every entry point dispatches at, mirrored to the host
// (§6.10) so the dispatch arithmetic and the attribute below cannot disagree.
const BLUR_WG: u32 = 8u;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Texel `i` of line `line`, with `i` running along the transform's axis.
fn at(i: u32, line: u32) -> vec2<i32> {
    if f.axis == 0u {
        return vec2<i32>(i32(i), i32(line));
    }
    return vec2<i32>(i32(line), i32(i));
}

// One butterfly's twiddled pair: `a ± w·b` for each complex lane of a texel pair,
// written to the Stockham output positions. Factored so the two entry points are
// each one gather and one call.
fn twiddle(t: u32) -> vec2<f32> {
    let angle = f.dir * 6.283185307179586 * f32(t % f.ns) / f32(2u * f.ns);
    return vec2<f32>(cos(angle), sin(angle));
}

fn out_base(t: u32) -> u32 {
    return (t / f.ns) * 2u * f.ns + (t % f.ns);
}

// This pass's transform length, and how many lines it runs over.
fn fft_n() -> u32 {
    return select(f.ny, f.nx, f.axis == 0u);
}

fn fft_lines() -> u32 {
    return select(f.nx, f.ny, f.axis == 0u);
}

// One Stockham pass over **both** textures — the image's three planes.
@compute @workgroup_size(BLUR_WG, BLUR_WG, 1)
fn fft_both(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = fft_n();
    let t = gid.x;
    let line = gid.y;
    if t >= n / 2u || line >= fft_lines() {
        return;
    }
    let w = twiddle(t);
    let j = out_base(t);

    let a0 = textureLoad(src_light, at(t, line), 0);
    let b0 = textureLoad(src_light, at(t + n / 2u, line), 0);
    let bw0 = vec4<f32>(cmul(b0.xy, w), cmul(b0.zw, w));
    textureStore(dst_light, at(j, line), a0 + bw0);
    textureStore(dst_light, at(j + f.ns, line), a0 - bw0);

    let a1 = textureLoad(src_aux, at(t, line), 0).xy;
    let b1 = textureLoad(src_aux, at(t + n / 2u, line), 0).xy;
    let bw1 = cmul(b1, w);
    textureStore(dst_aux, at(j, line), vec4<f32>(a1 + bw1, 0.0, 0.0));
    textureStore(dst_aux, at(j + f.ns, line), vec4<f32>(a1 - bw1, 0.0, 0.0));
}

// One Stockham pass over the **aux** texture alone — the kernel's transform, which
// is one complex plane and should not pay three. The light pair is untouched.
@compute @workgroup_size(BLUR_WG, BLUR_WG, 1)
fn fft_one(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = fft_n();
    let t = gid.x;
    let line = gid.y;
    if t >= n / 2u || line >= fft_lines() {
        return;
    }
    let w = twiddle(t);
    let j = out_base(t);
    let a = textureLoad(src_aux, at(t, line), 0).xy;
    let b = textureLoad(src_aux, at(t + n / 2u, line), 0).xy;
    let bw = cmul(b, w);
    textureStore(dst_aux, at(j, line), vec4<f32>(a + bw, 0.0, 0.0));
    textureStore(dst_aux, at(j + f.ns, line), vec4<f32>(a - bw, 0.0, 0.0));
}

// The aperture, in the spatial domain: a filled disc of radius `f.radius`, centred
// on texel (0, 0) in **wrap-around order** — the far edges of the texture are the
// kernel's negative offsets, which is what makes `out[n] = Σ k[d]·f[n−d]` read the
// product of transforms directly.
//
// The edge is antialiased over one texel, which does two jobs: the disc's rim does
// not stairstep into the bokeh, and the radius fades continuously to a single-texel
// impulse — the identity — as it approaches zero, so the knob's bottom end is
// smooth rather than a pop. Never normalized; the border weight's division absorbs
// the kernel's scale exactly (see the header).
@compute @workgroup_size(BLUR_WG, BLUR_WG, 1)
fn make_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= f.nx || gid.y >= f.ny {
        return;
    }
    let d = vec2<f32>(
        f32(min(gid.x, f.nx - gid.x)),
        f32(min(gid.y, f.ny - gid.y)),
    );
    let cover = clamp(f.radius + 0.5 - length(d), 0.0, 1.0);
    textureStore(dst_aux, vec2<i32>(gid.xy), vec4<f32>(cover, 0.0, 0.0, 0.0));
}

// The convolution itself: every plane of the transformed image multiplied by the
// kernel's transform at the same frequency, with the inverse pass's 1/(W·H)
// folded in.
@compute @workgroup_size(BLUR_WG, BLUR_WG, 1)
fn apply_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= f.nx || gid.y >= f.ny {
        return;
    }
    let p = vec2<i32>(gid.xy);
    let k = textureLoad(kernel, p, 0).xy * f.inv_n;
    let light = textureLoad(src_light, p, 0);
    textureStore(
        dst_light,
        p,
        vec4<f32>(cmul(light.xy, k), cmul(light.zw, k)),
    );
    let aux = textureLoad(src_aux, p, 0).xy;
    textureStore(dst_aux, p, vec4<f32>(cmul(aux, k), 0.0, 0.0));
}
