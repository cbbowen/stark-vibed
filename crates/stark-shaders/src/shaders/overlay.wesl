// Selection outline — the "marching ants" (§6.8).
//
// Drawn over the finished image, one instanced quad per mask tile, in the same
// canvas→NDC frame as the compositor. The outline is recovered from the mask itself
// rather than from the shape that produced it, so it costs nothing to keep correct
// under union/subtract/intersect: the boundary is the mask's half-contour, and the
// distance to it is the classic first-order estimate `(m − h) / |∇m|`. Taking the
// gradient at exactly one canvas pixel and converting the distance to *screen* pixels
// with the zoom gives a line of constant on-screen width at any zoom, and one that
// stays thin over a feathered edge (where |∇m| is small, so points a little way off
// the contour are already far in these units) instead of smearing across the ramp.
//
// **Half of what, is per selection.** A partial selection (§6.8) scales its whole
// mask, so a region selected at 0.4 has no 0.5-contour anywhere and the ants would
// simply not be drawn. Each instance therefore carries its selection's `level` — the
// strongest coverage in that mask — and the contour is at `level/2`. The recovered
// distance is unchanged by the scaling, since `m` and `|∇m|` scale together, so the
// line lands in the same place at the same width whatever strength the selection is
// drawn at. This is the *only* thing `level` does here: it says where to draw a
// boundary, never how strongly anything is selected.

import package::lib::display::{encode_from_srgb};
import package::lib::quad::{quad_corner};
// The canvas → NDC frame and the tile sampler, group 0 — the outline follows the
// canvas, so it turns and mirrors with it.
import package::view::{view, samp, canvas_to_clip, interior_uv};

@group(1) @binding(0) var mask: texture_2d<f32>;

/// Half-width of the drawn outline, in screen px.
const HALF_WIDTH: f32 = 1.0;
/// Dash period along the screen diagonal, in px.
const DASH: f32 = 8.0;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    // Flat: one value for the whole instance, not interpolated across the quad.
    @location(1) @interpolate(flat) tint: vec4<f32>,
    @location(2) @interpolate(flat) level: f32,
}

// One mask tile of the outline being traced.
struct OverlayInstance {
    // Per-tile instance: the mask tile interior's top-left in canvas px.
    @location(0) origin: vec2<f32>,
    // The outline's color — the marching ants, or a peer's hue (§17.3).
    @location(1) tint: vec4<f32>,
    // The mask's peak coverage (`Selection::level`) — what "fully selected" means
    // for this instance's selection, and so twice the contour to trace.
    @location(2) level: f32,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, inst: OverlayInstance) -> VsOut {
    let corner = quad_corner(vi);
    var out: VsOut;
    out.clip = canvas_to_clip(inst.origin + corner * view.misc.x);
    // The apron'd mask texture mapped exactly as the compositor maps a color tile —
    // so the taps below reach into the apron (the neighbour's real coverage) at the
    // tile edge rather than clamping.
    out.uv = interior_uv(corner);
    out.tint = inst.tint;
    out.level = inst.level;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // uv step of one canvas px: (TILE_SIZE/TILE_TEX) / TILE_SIZE = 1/TILE_TEX.
    let d = view.misc.y / view.misc.x;
    let m = textureSample(mask, samp, in.uv).r;
    let mx0 = textureSample(mask, samp, in.uv - vec2<f32>(d, 0.0)).r;
    let mx1 = textureSample(mask, samp, in.uv + vec2<f32>(d, 0.0)).r;
    let my0 = textureSample(mask, samp, in.uv - vec2<f32>(0.0, d)).r;
    let my1 = textureSample(mask, samp, in.uv + vec2<f32>(0.0, d)).r;

    // Coverage gradient per canvas px. Flat mask (interior or exterior) ⇒ no contour
    // anywhere near, so nothing to draw. The floor scales with the mask, because the
    // gradient does: a real edge in a selection drawn at 0.1 is a tenth as steep as
    // the same edge at full strength, and a fixed floor would read it as flat.
    let g = vec2<f32>(mx1 - mx0, my1 - my0) * 0.5;
    let gl = length(g);
    if gl < 1e-4 * in.level {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let dist_px = abs((m - 0.5 * in.level) / gl) * view.misc.w;
    let alpha = 1.0 - smoothstep(HALF_WIDTH - 0.5, HALF_WIDTH + 0.5, dist_px);
    if alpha <= 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    // A peer's selection (tint.a > 0) is a flat line in that peer's color, so it
    // reads as "someone else's region" at a glance and never as your own
    // (§17.3). The hue is a display-sRGB code (`peer_color`), and the target need
    // not be sRGB — so it is carried onto whatever this frame's is (§6.5).
    if in.tint.a > 0.0 {
        let rgb = encode_from_srgb(u32(view.xlate.z), in.tint.rgb);
        return vec4<f32>(rgb, alpha * in.tint.a);
    }
    // Your own: alternating black/white along the screen diagonal — legible over
    // paint of any value, which a single-color outline is not. Black and white are
    // the two colours every transfer agrees on, so they are stored as they are.
    let dash = step(0.5, fract((in.clip.x + in.clip.y) / DASH));
    return vec4<f32>(vec3<f32>(dash), alpha);
}
