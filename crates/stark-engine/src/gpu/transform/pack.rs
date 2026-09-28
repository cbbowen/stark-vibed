//! The transform's group-0 uniforms, packed on the CPU (§16): one per draw, built in
//! draw order from the plan before anything is encoded.

use bytemuck::Zeroable;
use stark_model::document::Homography;
use stark_model::geom::{Affine2, Mat2, TILE_SIZE, TILE_TEX, TileCoord, Vec2};
use stark_shaders::mirror::transform::{
    Combine as CombineUniform, Gated as GatedUniform, Quad as QuadUniform,
};

use super::plan::{FragMap, GatedKind, GatedMap, SourceUnit};
use crate::gpu::tile::mask_tex_origin;

/// The rect-scoped map a gated pass draws **through** (§16.8, §16.9): the pieces, the
/// shared inverse homography a perspective carries (a warp's cells carry their own
/// fragment map instead), and the source rect that scopes the whole thing.
pub(super) struct Gated<'a> {
    pub(super) units: &'a [SourceUnit],
    inv: Option<&'a Homography>,
    rect: (Vec2, Vec2),
}

impl<'a> Gated<'a> {
    pub(super) fn new(units: &'a [SourceUnit], rect: (Vec2, Vec2), geo: &'a GatedMap) -> Self {
        Self {
            units,
            inv: shared_inverse(geo),
            rect,
        }
    }
}

/// The inverse homography every piece of a perspective shares; `None` for a warp.
fn shared_inverse(geo: &GatedMap) -> Option<&Homography> {
    match &geo.kind {
        GatedKind::Persp { inv, .. } => Some(inv),
        GatedKind::Warp { .. } => None,
    }
}

/// How many draws `rewrites` plans — one per source reaching each destination — and
/// so exactly how many slots to stage, read off the plan before anything is encoded.
pub(super) fn slot_count<S>(rewrites: &[(TileCoord, Vec<S>)]) -> usize {
    rewrites.iter().map(|(_, sources)| sources.len()).sum()
}

/// The affine's quad uniforms in draw order: destination-major, sources in the plan's
/// order — what the affine parcel and `render_mask` walk.
pub(super) fn stage_quads(
    affine: Affine2,
    rewrites: &[(TileCoord, Vec<TileCoord>)],
    opacity: f32,
    into: &mut Vec<QuadUniform>,
) {
    for (dest, sources) in rewrites {
        for src in sources {
            into.push(quad_uniform(affine, *src, *dest, opacity));
        }
    }
}

/// The gated parcel's piece uniforms in draw order.
pub(super) fn stage_gated_parcels(
    g: &Gated<'_>,
    rewrites: &[(TileCoord, Vec<usize>)],
    opacity: f32,
    into: &mut Vec<GatedUniform>,
) {
    for (dest, idxs) in rewrites {
        for idx in idxs {
            into.push(gated_uniform(&g.units[*idx], g.inv, g.rect, *dest, opacity));
        }
    }
}

/// The gated mask's uniforms in draw order (`render_gated_mask`): per destination
/// the residue first, then its pieces at 1.0 — the mask pass carries coverage rather
/// than gating by it, so the opacity rides on the moved `Selection` instead (§6.8).
pub(super) fn stage_gated_masks(
    g: &Gated<'_>,
    rewrites: &[(TileCoord, Vec<usize>)],
    into: &mut Vec<GatedUniform>,
) {
    for (dest, idxs) in rewrites {
        into.push(gated_base(g.rect, *dest));
        for idx in idxs {
            into.push(gated_uniform(&g.units[*idx], g.inv, g.rect, *dest, 1.0));
        }
    }
}

/// One combine per destination `rewrites` names, in its order.
pub(super) fn stage_combines<S>(
    rewrites: &[(TileCoord, Vec<S>)],
    gate: Option<(Vec2, Vec2)>,
    opacity: f32,
) -> Vec<CombineUniform> {
    rewrites
        .iter()
        .map(|(dest, _)| combine_uniform(*dest, gate, opacity))
        .collect()
}

/// Whether (and where) the cut is gated by a source rect. The affine's whole-plane cut
/// binds the zero gate.
pub(super) fn combine_uniform(
    dest: TileCoord,
    gate: Option<(Vec2, Vec2)>,
    opacity: f32,
) -> CombineUniform {
    let dest_origin = mask_tex_origin(dest);
    match gate {
        Some(rect) => CombineUniform {
            dst_origin: dest_origin.to_array(),
            gated: 1.0,
            opacity,
            rect_min: rect.0.to_array(),
            rect_max: rect.1.to_array(),
        },
        None => CombineUniform {
            dst_origin: dest_origin.to_array(),
            gated: 0.0,
            opacity,
            ..Default::default()
        },
    }
}

/// One source tile's interior quad, drawn into `dest`'s texture (paint and mask
/// tiles share the `TILE_TEX` geometry).
fn quad_uniform(affine: Affine2, src: TileCoord, dest: TileCoord, opacity: f32) -> QuadUniform {
    let dest_origin = mask_tex_origin(dest);
    let src_origin = src.origin();
    let src_tex_origin = mask_tex_origin(src);
    let m = affine.matrix2;
    let t = affine.translation;
    // The fragment stage maps back through the inverse (see `src_uv` in the
    // shader). Exact for the exactness-invariant affines: the identity's
    // inverse is the identity, a translation's is its negation, an axis
    // flip's is itself.
    let inv = affine.inverse();
    let (im, it) = (inv.matrix2, inv.translation);
    QuadUniform {
        dst_origin: dest_origin.to_array(),
        tile_tex: TILE_TEX as f32,
        // Shader rows: c.x = m.x·p.x + m.y·p.y; glam's Mat2 is column-major.
        m: [m.x_axis.x, m.y_axis.x, m.x_axis.y, m.y_axis.y],
        xlate: t.to_array(),
        src_origin: src_tex_origin.to_array(),
        src_interior: src_origin.to_array(),
        tile_size: TILE_SIZE as f32,
        opacity,
        i: [im.x_axis.x, im.y_axis.x, im.x_axis.y, im.y_axis.y],
        inv_xlate: it.to_array(),
        ..Default::default()
    }
}

/// One [`SourceUnit`] drawn into `dest`'s texture — one piece of a rect-scoped map
/// (§16.8, §16.9). `inv` is the map's shared inverse homography (perspective); warp
/// cells carry their own fragment map in the unit.
fn gated_uniform(
    unit: &SourceUnit,
    inv: Option<&Homography>,
    rect: (Vec2, Vec2),
    dest: TileCoord,
    opacity: f32,
) -> GatedUniform {
    let mut u = gated_base(rect, dest);
    let c = &unit.corners;
    u.c0 = [c[0].x, c[0].y, c[1].x, c[1].y];
    u.c1 = [c[2].x, c[2].y, c[3].x, c[3].y];
    u.src_origin = mask_tex_origin(unit.src).to_array();
    u.opacity = opacity;
    match &unit.frag {
        FragMap::Persp => {
            let h = inv.expect("perspective units carry a shared inverse");
            set_rows(&mut u, &h.rows);
        }
        FragMap::Cell { g, min, size } => {
            let d = g[3] - g[1] - g[2] + g[0];
            if d == Vec2::ZERO {
                // The cell is a parallelogram — an affine, inverted through
                // the same arithmetic the affine action trusts, so an
                // untouched cell (whose map is exactly the identity) keeps
                // §16.4's tap exactness.
                let m = Mat2::from_cols((g[1] - g[0]) / size.x, (g[2] - g[0]) / size.y);
                let fwd = Affine2::from_mat2_translation(m, g[0] - m * *min);
                set_rows(&mut u, &Homography::from_affine(fwd.inverse()).rows);
            } else {
                u.mode = 1.0; // inverse-bilinear
                u.g0 = [g[0].x, g[0].y, g[1].x, g[1].y];
                u.g1 = [g[2].x, g[2].y, g[3].x, g[3].y];
                u.cell_min = min.to_array();
                u.cell_size = size.to_array();
            }
        }
    }
    u
}

/// The uniform for `fs_mask_base`: only the destination origin and the gate rect
/// matter — the residue pass has no piece.
fn gated_base(rect: (Vec2, Vec2), dest: TileCoord) -> GatedUniform {
    let dest_origin = mask_tex_origin(dest);
    let mut u = GatedUniform::zeroed();
    u.dst_origin = dest_origin.to_array();
    u.tile_tex = TILE_TEX as f32;
    u.rect_min = rect.0.to_array();
    u.rect_max = rect.1.to_array();
    u
}

fn set_rows(u: &mut GatedUniform, rows: &[[f32; 3]; 3]) {
    u.i0 = [rows[0][0], rows[0][1], rows[0][2], 0.0];
    u.i1 = [rows[1][0], rows[1][1], rows[1][2], 0.0];
    u.i2 = [rows[2][0], rows[2][1], rows[2][2], 0.0];
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::transform::plan::TransformPlan;

    /// A transform's draw count is a pure function of its plan — the sum over the
    /// destinations of the sources reaching each — so the slot buffer is sized exactly
    /// before a command is encoded, and [`stage_quads`] stages one uniform per draw in
    /// the order the draws take their slots back: destination-major, sources as the
    /// plan lists them. Adapter-free; the GPU half is the `spent` assertions.
    #[test]
    fn a_plans_draws_are_counted_and_staged_in_draw_order() {
        let c = |x, y| TileCoord::new(x, y);
        let plan = TransformPlan {
            rewrites: vec![
                (c(0, 0), vec![c(0, 0), c(1, 0)]),
                (c(1, 0), vec![]), // a cut with no incoming paint draws nothing
                (c(2, 1), vec![c(1, 1)]),
            ],
            drops: vec![],
        };
        assert_eq!(slot_count(&plan.rewrites), 3);

        let mut staged = Vec::new();
        stage_quads(Affine2::IDENTITY, &plan.rewrites, 0.5, &mut staged);
        assert_eq!(staged.len(), slot_count(&plan.rewrites));
        let expect = [(c(0, 0), c(0, 0)), (c(0, 0), c(1, 0)), (c(2, 1), c(1, 1))];
        for (u, (dest, src)) in staged.iter().zip(expect) {
            let d = mask_tex_origin(dest);
            let s = src.origin();
            assert_eq!(u.dst_origin, [d.x, d.y], "destination origin");
            assert_eq!(u.src_interior, [s.x, s.y], "source origin");
            assert_eq!(u.opacity, 0.5, "the gate's opacity rides in the quad");
        }
    }
}
