// Selection-mask rasterization (§6.8).
//
// One draw per affected tile: a fullscreen pass over a `TILE_TEX` R8 target that
// evaluates the op's shape **analytically at canvas position** and combines it with
// the previous mask. Because the coverage is a pure function of canvas position, a
// tile's apron is rasterized identically to its neighbour's interior — the same seam
// invariant the stroke passes hold (§6.4) — and the mask can be resampled at any
// zoom without ever having been stored at a particular one.
//
// Coverage comes from a signed distance (negative inside), so antialiasing and
// feathering are the same knob: the 0.5-contour is the selection boundary and the
// ramp around it spans `feather` canvas px (never less than one, which is the
// antialiased hard edge). A soft mask is what makes the brush fade out of the
// selection instead of stopping at a staircase.
//
// `prev` is bound either as the previous mask tile or as a 1×1 texture holding the
// constant coverage that reigns where the previous selection had no tile at all; the
// load is clamped to the texture's own extent, so the same code reads both.

struct Params {
    // .xy = this tile texture's top-left in canvas px, .z = 2/TILE_TEX (canvas px →
    // NDC), .w = feather width in canvas px.
    a: vec4<f32>,
    // Shape parameters: rect (min.xy, max.xy); ellipse (center.xy, radii.xy).
    b: vec4<f32>,
    // .x = shape kind, .y = combine mode, .z = lasso edge count, .w unused.
    c: vec4<f32>,
}

@group(0) @binding(0) var<uniform> p: Params;
// The previous mask over this tile (or a 1×1 constant — see above).
@group(0) @binding(1) var prev: texture_2d<f32>;
// Lasso edges, one texel each: (a.xy, b.xy) in canvas px. A 1×1 dummy for the
// analytic shapes, which never read it.
@group(0) @binding(2) var edges: texture_2d<f32>;

const KIND_ALL: f32 = 0.0;
const KIND_RECT: f32 = 1.0;
const KIND_ELLIPSE: f32 = 2.0;
const KIND_LASSO: f32 = 3.0;

const MODE_REPLACE: f32 = 0.0;
const MODE_UNION: f32 = 1.0;
const MODE_SUBTRACT: f32 = 2.0;
const MODE_INTERSECT: f32 = 3.0;
const MODE_INVERT: f32 = 4.0;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

// Signed distance to an axis-aligned box, negative inside.
fn sd_rect(q: vec2<f32>, lo: vec2<f32>, hi: vec2<f32>) -> f32 {
    let d = max(lo - q, q - hi);
    return length(max(d, vec2<f32>(0.0, 0.0))) + min(max(d.x, d.y), 0.0);
}

// Signed distance to an axis-aligned ellipse, negative inside. The exact distance
// needs a quartic root; this is the standard scaled-unit-circle approximation, which
// is exact on the axes and errs by well under the antialiasing ramp elsewhere.
fn sd_ellipse(q: vec2<f32>, c: vec2<f32>, r: vec2<f32>) -> f32 {
    let rr = max(r, vec2<f32>(1e-4, 1e-4));
    let n = (q - c) / rr;
    return (length(n) - 1.0) * min(rr.x, rr.y);
}

// Signed distance to a closed polygon, negative inside: the unsigned distance to the
// nearest edge, signed by an even-odd crossing test on a ray to +x. O(edges) per
// texel, which is why the lasso path is decimated before it gets here.
fn sd_lasso(q: vec2<f32>, count: u32) -> f32 {
    var d2 = 1e30;
    var inside = false;
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let e = textureLoad(edges, vec2<i32>(i32(i), 0), 0);
        let a = e.xy;
        let b = e.zw;
        let ab = b - a;
        let aq = q - a;
        let t = clamp(dot(aq, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
        let w = aq - ab * t;
        d2 = min(d2, dot(w, w));
        // Half-open in y so a vertex is counted exactly once.
        if (a.y > q.y) != (b.y > q.y) {
            let x = a.x + ab.x * (q.y - a.y) / ab.y;
            if q.x < x {
                inside = !inside;
            }
        }
    }
    let d = sqrt(d2);
    if inside {
        return -d;
    }
    return d;
}

// The op's own coverage at canvas position `q`, in [0, 1].
fn shape_coverage(q: vec2<f32>) -> f32 {
    let kind = p.c.x;
    if kind == KIND_ALL {
        return 1.0;
    }
    var sd = 1e30;
    if kind == KIND_RECT {
        sd = sd_rect(q, p.b.xy, p.b.zw);
    } else if kind == KIND_ELLIPSE {
        sd = sd_ellipse(q, p.b.xy, p.b.zw);
    } else if kind == KIND_LASSO {
        sd = sd_lasso(q, u32(p.c.z));
    }
    // One canvas pixel is the floor: a "hard" edge is still antialiased, and the
    // 0.5-contour stays exactly on the geometric boundary either way.
    let w = max(p.a.w, 1.0);
    return clamp(0.5 - sd / w, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    // The previous coverage here: a real tile 1:1 with this one, or the 1×1 constant
    // standing in for "the previous selection had no tile here".
    let dim = vec2<i32>(textureDimensions(prev));
    let c = min(vec2<i32>(floor(frag.xy)), dim - vec2<i32>(1, 1));
    let old = textureLoad(prev, c, 0).r;

    let mode = p.c.y;
    if mode == MODE_INVERT {
        return vec4<f32>(1.0 - old, 0.0, 0.0, 0.0);
    }

    let s = shape_coverage(p.a.xy + frag.xy);
    var out = s; // MODE_REPLACE
    if mode == MODE_UNION {
        out = max(old, s);
    } else if mode == MODE_SUBTRACT {
        out = old * (1.0 - s);
    } else if mode == MODE_INTERSECT {
        out = old * s;
    }
    return vec4<f32>(out, 0.0, 0.0, 0.0);
}
