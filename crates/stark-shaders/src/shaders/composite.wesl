// Compositing pass A (§6.3): draw each visible tile's channels into
// viewport-sized offscreen targets. Color composites with premultiplied "over",
// the aux (paint height) accumulates additively. The media pass then lights the
// result. Vertex setup matches the present pass: one instanced quad per tile.
//
// Two fragment entry points, one law each:
//
//   · `fs_main` — the screen path. A layer's "over" weight is its **visible
//     alpha**: per-unit opacity and thickness combined by the translucent-slab law
//     `1 − exp(−K·opacity·height)` — the same law `paint_common.wesl` uses to
//     stack parcels *within* a layer, so a parcel covers the layer below exactly
//     as much as it covers resident paint. The accumulated alpha is then the
//     stack's coverage and the media pass reads it directly. (Weighting by
//     opacity alone let a film with opacity 1 and no thickness — every soft
//     brush's fringe — replace the colour beneath while drawing as nothing over
//     bare canvas.)
//
//   · `fs_raw` — the passthrough. The dynamics loop composites base tiles into
//     its working region with this same module, and that region must keep the
//     tile representation itself (per-unit opacity in alpha, §6.1):
//     the loop's `exchange` reads it and the slice writes it back to
//     persistent tiles. Running the slab law here would store *coverage* as
//     opacity — corrupting every smeared texel, differently on each side of a
//     piece or freeze cut, which is how the preview stops matching the commit.

import package::lib::paint_common::{OPACITY_K, PAINT_EPS};

struct View {
    // The canvas px -> NDC linear map, column-major: `st.xy` is the first column and
    // `st.zw` the second. A full 2x2 because the view can be turned and mirrored
    // (§18.1.2); upright and unmirrored it is diagonal, and this is the
    // scale pair it always was.
    st: vec4<f32>,
    xlate: vec4<f32>, // translate.xy, unused zw
    // misc.x = tile_size (interior, canvas px). The tile texture carries an apron
    // around the interior, so we sample only the interior sub-rect and let the
    // bilinear filter reach into the apron at the edges (kills tile seams):
    //   uv = corner * misc.y + misc.z,  misc.y = TILE_SIZE/TILE_TEX, misc.z = APRON/TILE_TEX.
    misc: vec4<f32>,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var samp: sampler;
@group(1) @binding(0) var tile_color: texture_2d<f32>;
@group(1) @binding(1) var tile_aux: texture_2d<f32>;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) opacity: f32,
}

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) aux: vec4<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    // Per-tile instance: the tile interior's top-left in canvas px.
    @location(0) origin: vec2<f32>,
    // The layer's opacity.
    @location(1) opacity: f32,
) -> VsOut {
    let corner = vec2<f32>(f32(vi & 1u), f32((vi >> 1u) & 1u));
    let m = mat2x2<f32>(view.st.xy, view.st.zw);
    let canvas_pos = origin + corner * view.misc.x;
    let ndc = m * canvas_pos + view.xlate.xy;

    var out: VsOut;
    out.clip = vec4<f32>(ndc, 0.0, 1.0);
    out.uv = corner * view.misc.y + view.misc.z; // interior sub-rect of the apron'd texture
    out.opacity = opacity;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> FsOut {
    let c = textureSample(tile_color, samp, in.uv);
    let h = textureSample(tile_aux, samp, in.uv);
    // The layer's visible alpha, scaled by layer opacity: coverage fades
    // linearly with the layer slider, and the latent colour is untouched.
    let w = (1.0 - exp(-OPACITY_K * c.a * h.x)) * in.opacity;
    var out: FsOut;
    // Re-premultiplied by coverage, so the fixed-function "over" both weighs the
    // colour and accumulates the stack's total coverage in alpha.
    out.color = vec4<f32>(c.rgb / max(c.a, PAINT_EPS) * w, w);
    out.aux = h * in.opacity;
    return out;
}

@fragment
fn fs_raw(in: VsOut) -> FsOut {
    // Straight passthrough of the tile representation (× layer opacity), for the
    // dynamics region composite — see the header.
    var out: FsOut;
    out.color = textureSample(tile_color, samp, in.uv) * in.opacity;
    out.aux = textureSample(tile_aux, samp, in.uv) * in.opacity;
    return out;
}
