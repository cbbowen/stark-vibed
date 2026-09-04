// Oklab <-> display color conversions (§6.5), shared by shader stages
// via WESL imports. Mirrors the Rust constants in `color.rs` so CPU ingest and
// GPU present agree exactly. (WGSL number literals may not contain underscores.)

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow(max((c + 0.055) / 1.055, vec3<f32>(0.0)), vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

// The inverse of `oklab_to_linear_srgb`, mirroring `color::linear_srgb_to_oklab`.
// The cube root is taken *signed* — `sign(x)·|x|^(1/3)`, matching Rust's `cbrt` —
// because a color that has been through a blend in another space can land slightly
// outside the sRGB gamut, and `pow` of a negative is NaN. Signed keeps the round
// trip exact where it is in gamut and merely approximate where it is not, instead
// of clamping (which shifts hue) or poisoning the frame with NaN.
fn linear_srgb_to_oklab(rgb: vec3<f32>) -> vec3<f32> {
    let l = 0.4122214708 * rgb.r + 0.5363325363 * rgb.g + 0.0514459929 * rgb.b;
    let m = 0.2119034982 * rgb.r + 0.6806995451 * rgb.g + 0.1073969566 * rgb.b;
    let s = 0.0883024619 * rgb.r + 0.2817188376 * rgb.g + 0.6299787005 * rgb.b;

    let lms = vec3<f32>(l, m, s);
    let root = sign(lms) * pow(abs(lms), vec3<f32>(1.0 / 3.0));

    return vec3<f32>(
        0.2104542553 * root.x + 0.7936177850 * root.y - 0.0040720468 * root.z,
        1.9779984951 * root.x - 2.4285922050 * root.y + 0.4505937099 * root.z,
        0.0259040371 * root.x + 0.7827717662 * root.y - 0.8086757660 * root.z,
    );
}

// Linear sRGB → XYZ normalized to D65 white. These are the sRGB primaries' rows,
// each divided by its own sum (= the white's tristimulus value), so `(1,1,1)` maps
// to `(1,1,1)`.
//
// **This is the space light is combined and scaled in** — see `blend_common.wesl`
// for the argument, which is one paragraph long and worth reading before touching
// anything here: XYZ is linear in light, non-negative for every real color, and
// primary-independent, and the row normalization is what makes "1" mean the same
// thing on all three axes. Here in `lib` rather than beside that argument because
// two passes now need the pair — the blend (§18.0.4) and the filter (§21) — and a
// binding-free leaf is where a conversion two pipelines share belongs.
fn linear_to_light(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(c, vec3<f32>(0.4339395, 0.3762078, 0.1898528)),
        dot(c, vec3<f32>(0.2126729, 0.7151522, 0.0721750)),
        dot(c, vec3<f32>(0.0177566, 0.1094671, 0.8727763)),
    );
}

// The exact inverse: the XYZ→linear-sRGB matrix with column *j* scaled by the white
// point's *j*th component, undoing the row normalization above.
fn light_to_linear(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(c, vec3<f32>( 3.0799545, -1.5371385, -0.5428160)),
        dot(c, vec3<f32>(-0.9212583,  1.8760108,  0.0452477)),
        dot(c, vec3<f32>( 0.0528873, -0.2040259,  1.1511385)),
    );
}

// Linear sRGB floored **in light** (§6.5): XYZ is non-negative for every real color
// where a wide-gamut color's linear sRGB is not, so a floor there keeps the chroma a
// per-channel floor would clip, and still catches an overshoot below zero light. The
// identity — bit for bit, not to rounding — for a color in the cube.
fn floor_light(lin: vec3<f32>) -> vec3<f32> {
    if all(lin >= vec3<f32>(0.0)) {
        return lin;
    }
    return light_to_linear(max(linear_to_light(lin), vec3<f32>(0.0)));
}

fn oklab_to_linear_srgb(lab: vec3<f32>) -> vec3<f32> {
    let l_ = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
    let m_ = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
    let s_ = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;

    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;

    return vec3<f32>(
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    );
}
