// The fullscreen vertex, shared by every pass that covers its whole target.
//
// It lives here — under `lib/`, where **a module may not declare a binding** —
// for the reason the whole directory exists. This function was written out nine
// times (blend_common, fill, guides, integrate, media_common, resolve, selection,
// slice, transform), and `blend_common`'s copy carried the explanation on its
// face: *"a local copy rather than `media_common`'s: importing that module would
// drag its whole `@group(0)` in, and this pass binds something else entirely."*
// A helper that cannot be shared without inheriting a bind group is not a shared
// helper, and nine copies of six lines is what that costs. Binding-free modules
// have nothing to inherit, so the question does not arise.
//
// (`matte.wesl` is deliberately not a caller. It draws a four-vertex quad and
// needs the corner itself, to invert the view map at the vertices and interpolate
// canvas position across the fragment — a different shape, not a variant of this
// one.)

// Clip position of a fullscreen **triangle** from a vertex index — three
// vertices at (-1,-1), (3,-1), (-1,3), whose interior covers the viewport.
//
// One triangle rather than two: a quad's diagonal is a seam where the two halves
// meet, and fragments straddling it are shaded in separate quads, which both
// costs a little and — for anything reading derivatives, as `guides.wesl` does —
// changes the answer along a line down the middle of the screen.
//
// The host's draw call must ask for exactly 3 vertices.
fn fullscreen_clip(vi: u32) -> vec4<f32> {
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}
