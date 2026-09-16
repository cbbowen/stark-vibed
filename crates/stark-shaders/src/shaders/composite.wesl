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
//     brush's fringe — replace the color beneath while drawing as nothing over
//     bare canvas.)
//
//   · `fs_raw` — the passthrough. The dynamics loop composites base tiles into
//     its working region with this same module, and that region must keep the
//     tile representation itself (per-unit opacity in alpha, §6.1):
//     the loop's `exchange` reads it and the slice writes it back to
//     persistent tiles. Running the slab law here would store *coverage* as
//     opacity — corrupting every smeared texel, differently on each side of a
//     piece or freeze cut, which is how the preview stops matching the commit.

import package::lib::latent::{Latent, latent_of, latent_scale, latent_unpremult};
import package::lib::paint_common::{OPACITY_K, PAINT_EPS};
import package::lib::quad::{quad_corner};
// The canvas → NDC frame and the tile sampler, group 0.
import package::view::{view, samp, canvas_to_clip, interior_uv};

@group(1) @binding(0) var tile_color: texture_2d<f32>;
@group(1) @binding(1) var tile_aux: texture_2d<f32>;
// The tile's residual channel (§6.7) — bound only by a space that has one.
@if(resid) @group(1) @binding(2) var tile_resid: texture_2d<f32>;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) opacity: f32,
}

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) aux: vec4<f32>,
    // Re-premultiplied by the same coverage as `color` and over-blended by the same
    // rule, so the accumulator's residual composites through the stack exactly as its
    // latent does — the two are one color (§6.7).
    @if(resid) @location(2) resid: vec4<f32>,
}

// One tile of the layer being drawn — by pass A, and by the dynamics loop's own
// composite of its working region (see the header).
struct Instance {
    // Per-tile instance: the tile interior's top-left in canvas px.
    @location(0) origin: vec2<f32>,
    // The layer's opacity.
    @location(1) opacity: f32,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, inst: Instance) -> VsOut {
    let corner = quad_corner(vi);
    var out: VsOut;
    out.clip = canvas_to_clip(inst.origin + corner * view.misc.x);
    out.uv = interior_uv(corner);
    out.opacity = inst.opacity;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> FsOut {
    let c = textureSample(tile_color, samp, in.uv);
    let h = textureSample(tile_aux, samp, in.uv);
    // The layer's visible alpha, scaled by layer opacity: coverage fades
    // linearly with the layer slider, and the latent color is untouched.
    let w = (1.0 - exp(-OPACITY_K * c.a * h.x)) * in.opacity;
    // Un-premultiplied by the *color's* alpha, not the residual target's own: that
    // target's alpha is a duplicate of the same per-unit opacity, carried only because
    // the fixed-function "over" reads each target's own alpha, and one of the two has
    // to be the source of truth. `latent_unpremult` takes that alpha as its argument,
    // so the choice is made once for both halves rather than restated (`lib::latent`).
    var tile = latent_of(c.rgb, vec3<f32>(0.0));
    @if(resid) tile.res = textureSample(tile_resid, samp, in.uv).rgb;
    // Re-premultiplied by coverage, so the fixed-function "over" both weighs the
    // color and accumulates the stack's total coverage in alpha.
    let lit = latent_scale(latent_unpremult(tile, c.a, PAINT_EPS), w);

    var out: FsOut;
    out.color = vec4<f32>(lit.lat, w);
    out.aux = h * in.opacity;
    @if(resid) out.resid = vec4<f32>(lit.res, w);
    return out;
}

@fragment
fn fs_raw(in: VsOut) -> FsOut {
    // Straight passthrough of the tile representation (× layer opacity), for the
    // dynamics region composite — see the header.
    var out: FsOut;
    out.color = textureSample(tile_color, samp, in.uv) * in.opacity;
    out.aux = textureSample(tile_aux, samp, in.uv) * in.opacity;
    @if(resid) out.resid = textureSample(tile_resid, samp, in.uv) * in.opacity;
    return out;
}
