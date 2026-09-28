// The canvas → NDC frame every pass that rasterizes canvas geometry draws in (§6.3).
//
// One declaration of `View`, the uniform that carries it and the tile sampler beside
// it, for the three passes that share them: pass A's tiles (`composite.wesl`), the
// mattes drawn inside it (`matte.wesl`) and the selection outline over it
// (`overlay.wesl`).
//
// It owns bindings, so it is **not** under `lib/`: it sits beside `blend_common` and
// `media_common`, the other shared modules a pipeline takes a group's lower slots
// from. An importer takes both slots with it, whichever it names — `matte.wesl` reads
// only the uniform, and the linker strips the sampler out of its artifact.

struct View {
    // The canvas px → NDC linear map, column-major: `st.xy` is the first column and
    // `st.zw` the second. A full 2x2 because the view can be turned and mirrored
    // (§18.1.2); upright and unmirrored it is diagonal, and this is the scale pair it
    // always was.
    st: vec4<f32>,
    // The map's translation.
    xlate: vec2<f32>,
    // The display transfer the frame's target is encoded in (`lib/display.wesl`'s
    // selector, as a float), which `overlay.wesl` alone reads.
    transfer: f32,
    // The tile interior, canvas px.
    //
    // `@align(16)` starts a new 16-byte row, which is what these four were written as
    // one `vec4` for: the lane above `transfer` is spare, and naming them must not
    // slide the row up into it (§6.10).
    @align(16) tile_size: f32,
    // A tile texture carries an apron around its interior, so a pass samples the
    // interior sub-rect alone and lets the bilinear filter reach into the apron at the
    // edges, which is what kills tile seams:
    //
    //     uv = corner * uv_scale + uv_bias,  TILE_SIZE/TILE_TEX and APRON/TILE_TEX
    uv_scale: f32,
    uv_bias: f32,
    // Screen px per canvas px — what `overlay.wesl` and `matte.wesl` measure a
    // constant on-screen width with.
    zoom: f32,
}

@group(0) @binding(0) var<uniform> view: View;
// The sampler a tile is read through, at the filter mode the host chose (`View::new`).
@group(0) @binding(1) var samp: sampler;

// The canvas px → NDC 2x2, unpacked from `st`'s two columns. The one statement of
// that packing; `matte.wesl` inverts what this returns.
fn view_map() -> mat2x2<f32> {
    return mat2x2<f32>(view.st.xy, view.st.zw);
}

// Clip position of a canvas-space point.
fn canvas_to_clip(p: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(view_map() * p + view.xlate, 0.0, 1.0);
}

// A quad corner as a uv in the interior sub-rect of an apron'd tile texture — see
// `View.uv_scale`.
fn interior_uv(corner: vec2<f32>) -> vec2<f32> {
    return corner * view.uv_scale + view.uv_bias;
}
