//! GPU execution of the transforms (§16): the whole-plane affine, and the
//! rect-scoped perspective (§16.8) and warp (§16.9).
//!
//! [`TransformRenderer::apply`] takes a layer's tile map, the author's selection,
//! and a [`TransformMap`], and produces the transformed tile map plus the carried
//! selection — copy-on-write like every stroke, so old history versions keep
//! their tiles and the pool reclaims what falls out of reach.
//!
//! The plan (which tiles, which quads) is CPU-side and pure
//! ([`crate::document::transform`]); this module owns only the passes of
//! `transform.wesl`:
//!
//! - **parcel** — the selected source interiors forward-rasterized as transformed
//!   quads into a scratch pair, one destination tile at a time. Source interiors
//!   tile the plane, so their images are disjoint and the pass needs no blending
//!   and no order. The gated families draw *pieces* (`tile ∩ rect`, further split
//!   per warp sub-cell) with CPU-precomputed corners, watertight the same way.
//! - **combine** — cut the destination's own base by its mask (the lift law) and
//!   stack the parcel by the shared parcel-deposit law (`paint_common.wesl`);
//!   for the gated families the cut is additionally scoped by the source rect's
//!   coverage.
//! - **mask** — the selection mask resampled under the same map: pure Replace
//!   for the affine, residue-plus-max-blended-union for the gated families
//!   (§16.8).
//!
//! Like [`StrokeRenderer`](super::stroke::StrokeRenderer) this holds only
//! immutable GPU objects, so it is cheap to `Clone` and rides in the
//! `Action::Context` (§5).

use std::collections::BTreeMap;

use bytemuck::Zeroable;
use rpds::HashTrieMap;
use wgpu::util::DeviceExt;

use crate::colorspace::ColorSpace;
use crate::document::selection::Selection;
use crate::document::transform::{
    FragMap, GatedKind, SourceUnit, gated_geometry, plan_gated_mask, plan_gated_paint, plan_mask,
    plan_paint,
};
use crate::gpu::channels::{ChannelFormats, Channels};
use crate::gpu::context::GpuContext;
use crate::gpu::desc::Slot;
use crate::gpu::desc::{self, Zeroes};
use stark_model::document::{Homography, TransformMap};
use stark_model::geom::{Affine2, Mat2, TILE_APRON, TILE_SIZE, TILE_TEX, TileCoord, Vec2};
use stark_shaders::mirror::transform::binding as t;
use stark_shaders::mirror::transform::decl as td;

/// The affine's group: its map, and the bilinear clamp sampler.
///
/// **Two lists where there was one layout**, and that is the migration paying for
/// itself. `transform.wesl` puts `Quad` (the affine) and `Gated` (perspective/warp) at
/// `@group(0) @binding(0)` because no entry point reaches both — its header explains
/// why that is sound. The host had described the slot once, with `min_binding_size:
/// None`, which is the only size that serves two structs and therefore checks neither.
///
/// Naming the declaration means naming *which* struct, and a layout then bounds the
/// uniform its own pipelines actually read. `wgpu` requires the bound size to be at
/// least what the shader needs, so one layout genuinely could not have covered both:
/// `Quad`'s bound rejects the gated pipelines at creation, and `Gated`'s rejects the
/// affine's buffer at bind time. Two layouts is what the shader was saying all along.
const QUAD_SLOTS: &[Slot] = &[
    // The vertex stage places the quad through the forward map; the fragment stage
    // taps the source through the inverse.
    Slot::at(td::Q).in_stages(wgpu::ShaderStages::VERTEX_FRAGMENT),
    Slot::at(td::SAMP),
];

/// The rect-scoped maps' group — the same two things [`QUAD_SLOTS`] holds, against
/// `Gated` rather than `Quad`.
const GATED_SLOTS: &[Slot] = &[
    Slot::at(td::QG).in_stages(wgpu::ShaderStages::VERTEX_FRAGMENT),
    Slot::at(td::SAMP),
];

/// The source tile being carried, at group 1 — its channels and the mask over it, all
/// sampled, since the parcel resamples under the map (§16).
const SRC_SLOTS: &[Slot] = &[
    Slot::sampled(td::SRC_COLOR),
    Slot::sampled(td::SRC_AUX),
    Slot::sampled(td::SRC_MASK),
    // The source tile's residual (§6.7): carried under the same map and sampled at the
    // same uv as the color it belongs to.
    Slot::sampled(td::SRC_RESID),
];

/// The mask pass reads only the source mask — the same group 1, one slot of it.
const MASK_SRC_SLOTS: &[Slot] = &[Slot::sampled(td::SRC_MASK)];

/// The combine's inputs, which claim group 0 from 2 up — the quad passes never see
/// them, and no entry point reaches both sets (`transform.wesl`'s header).
const COMBINE_SLOTS: &[Slot] = &[
    Slot::at(td::BASE_COLOR),
    Slot::at(td::BASE_AUX),
    Slot::at(td::BASE_MASK),
    Slot::at(td::PARCEL_COLOR),
    Slot::at(td::PARCEL_AUX),
    // The gate rect: zeroed for the affine's whole-plane cut, the source rect for
    // perspective/warp.
    Slot::at(td::QC),
    // The base's and the parcel's residuals — past the gate rect because 7 was already
    // taken when they were added, and the shader says the same.
    Slot::at(td::BASE_RESID),
    Slot::at(td::PARCEL_RESID),
];

/// A texture view as the resource a bind-group entry takes.
fn tex(v: &wgpu::TextureView) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::TextureView(v)
}
use crate::gpu::selection::{SelectionRenderer, outside_clear};
use crate::gpu::submit::TileScope;
use crate::gpu::tile::{AllocSource, MASK_FORMAT, TileMap, TilePool};

// Generated from `transform.wesl`'s own declarations (§6.7). The three constructors
// below are free functions rather than inherent impls: the types live in
// `stark-shaders` now, and an inherent impl on another crate's type is not allowed.
// Each is still the only way one is built.
use stark_shaders::mirror::transform::{
    Combine as CombineUniform, Gated as GatedUniform, Quad as QuadUniform,
};

/// One source tile's interior quad, drawn into `dest`'s texture (paint and mask
/// tiles share the `TILE_TEX` geometry).
fn quad_uniform(affine: Affine2, src: TileCoord, dest: TileCoord, opacity: f32) -> QuadUniform {
    let dest_origin = dest.origin() - Vec2::splat(TILE_APRON as f32);
    let src_origin = src.origin();
    let src_tex_origin = src_origin - Vec2::splat(TILE_APRON as f32);
    let m = affine.matrix2;
    let t = affine.translation;
    // The fragment stage maps back through the inverse (see `src_uv` in the
    // shader). Exact for the exactness-invariant affines: the identity's
    // inverse is the identity, a translation's is its negation, an axis
    // flip's is itself.
    let inv = affine.inverse();
    let (im, it) = (inv.matrix2, inv.translation);
    QuadUniform {
        a: [dest_origin.x, dest_origin.y, TILE_TEX as f32, 0.0],
        // Shader rows: c.x = m.x·p.x + m.y·p.y; glam's Mat2 is column-major.
        m: [m.x_axis.x, m.y_axis.x, m.x_axis.y, m.y_axis.y],
        t: [t.x, t.y, src_tex_origin.x, src_tex_origin.y],
        u: [src_origin.x, src_origin.y, TILE_SIZE as f32, opacity],
        i: [im.x_axis.x, im.y_axis.x, im.x_axis.y, im.y_axis.y],
        j: [it.x, it.y, 0.0, 0.0],
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
    let src_tex_origin = unit.src.origin() - Vec2::splat(TILE_APRON as f32);
    u.t = [src_tex_origin.x, src_tex_origin.y, opacity, 0.0];
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
                u.a[3] = 1.0; // inverse-bilinear mode
                u.g0 = [g[0].x, g[0].y, g[1].x, g[1].y];
                u.g1 = [g[2].x, g[2].y, g[3].x, g[3].y];
                u.s = [min.x, min.y, size.x, size.y];
            }
        }
    }
    u
}

/// The uniform for `fs_mask_base`: only the destination origin and the gate rect
/// matter — the residue pass has no piece.
fn gated_base(rect: (Vec2, Vec2), dest: TileCoord) -> GatedUniform {
    let dest_origin = dest.origin() - Vec2::splat(TILE_APRON as f32);
    let mut u = GatedUniform::zeroed();
    u.a = [dest_origin.x, dest_origin.y, TILE_TEX as f32, 0.0];
    u.r = [rect.0.x, rect.0.y, rect.1.x, rect.1.y];
    u
}

fn set_rows(u: &mut GatedUniform, rows: &[[f32; 3]; 3]) {
    u.i0 = [rows[0][0], rows[0][1], rows[0][2], 0.0];
    u.i1 = [rows[1][0], rows[1][1], rows[1][2], 0.0];
    u.i2 = [rows[2][0], rows[2][1], rows[2][2], 0.0];
}

/// Whether (and where) the cut is gated by a source rect. The affine path binds the
/// zero gate, whose arithmetic is untouched from before the gate existed.
fn combine_uniform(dest: TileCoord, gate: Option<(Vec2, Vec2)>, opacity: f32) -> CombineUniform {
    let dest_origin = dest.origin() - Vec2::splat(TILE_APRON as f32);
    match gate {
        Some(rect) => CombineUniform {
            a: [dest_origin.x, dest_origin.y, 1.0, opacity],
            r: [rect.0.x, rect.0.y, rect.1.x, rect.1.y],
        },
        None => CombineUniform {
            a: [dest_origin.x, dest_origin.y, 0.0, opacity],
            r: [0.0; 4],
        },
    }
}

/// What a transform's passes draw **from**, as against which piece they are drawing.
///
/// Assembled once per [`TransformRenderer::apply`] and threaded through every pass —
/// the shape `stroke::dynamics`'s `PlanCtx` and `StrokeScene` already use, and for
/// the same reason: these four travel together through every hop, so they are one
/// parameter rather than four repeated at each.
///
/// `src_bgs` rides along because it is scoped to exactly this: a source tile's bind
/// group is shared across every destination its image reaches, and there is no
/// destination outside one `apply` for it to be shared with.
struct Source<'a> {
    pool: &'a TilePool,
    base: &'a TileMap,
    selection: &'a Selection,
    src_bgs: BTreeMap<TileCoord, wgpu::BindGroup>,
}

impl<'a> Source<'a> {
    fn new(pool: &'a TilePool, base: &'a TileMap, selection: &'a Selection) -> Self {
        Self {
            pool,
            base,
            selection,
            src_bgs: BTreeMap::new(),
        }
    }
}

/// The rect-scoped map a gated pass draws **through** (§16.8, §16.9): the pieces, the
/// shared inverse homography a perspective carries (a warp's cells carry their own
/// fragment map instead), and the source rect that scopes the whole thing.
struct Gated<'a> {
    units: &'a [SourceUnit],
    inv: Option<&'a Homography>,
    rect: (Vec2, Vec2),
}

/// A tile's paint channels as this module passes them around: the moved parcel a
/// quad pass writes, and the destination a combine writes.
type Parcel = Channels;

#[derive(Clone)]
pub struct TransformRenderer {
    ctx: GpuContext,
    /// The channel formats this transform's tiles carry — the color space's,
    /// resolved once (§6.7).
    formats: ChannelFormats,
    parcel_pipeline: wgpu::RenderPipeline,
    mask_pipeline: wgpu::RenderPipeline,
    combine_pipeline: wgpu::RenderPipeline,
    /// The rect-scoped maps' passes (§16.8, §16.9): pieces through `vs_gated`,
    /// and the mask's residue + max-blended moved coverage.
    parcel_gated_pipeline: wgpu::RenderPipeline,
    mask_gated_pipeline: wgpu::RenderPipeline,
    mask_base_pipeline: wgpu::RenderPipeline,
    quad_bgl: wgpu::BindGroupLayout,
    /// The rect-scoped maps' own group. Separate from [`Self::quad_bgl`] because their
    /// uniform is a different, larger struct — see [`GATED_SLOTS`].
    gated_bgl: wgpu::BindGroupLayout,
    src_bgl: wgpu::BindGroupLayout,
    mask_src_bgl: wgpu::BindGroupLayout,
    combine_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// The base of a virgin destination and the parcel of a cut-only tile, so the
    /// combine is one shader whatever exists.
    zeroes: Zeroes,
    /// For the selection constants (0/1 coverage) bound where a mask has no tile.
    selection: SelectionRenderer,
}

impl TransformRenderer {
    pub(crate) fn new(
        ctx: &GpuContext,
        color_space: &dyn ColorSpace,
        selection: SelectionRenderer,
        zeroes: Zeroes,
    ) -> Self {
        let device = &ctx.device;
        let formats = ChannelFormats::of(color_space);
        let resid = formats.has_resid();

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark transform"),
            source: wgpu::ShaderSource::Wgsl(stark_shaders::transform(resid).into()),
        });

        let frag = wgpu::ShaderStages::FRAGMENT;
        let quad_bgl =
            desc::layout_for(device, "stark transform quad bgl", QUAD_SLOTS, frag, resid);
        let gated_bgl = desc::layout_for(
            device,
            "stark transform gated bgl",
            GATED_SLOTS,
            frag,
            resid,
        );
        let src_bgl = desc::layout_for(device, "stark transform src bgl", SRC_SLOTS, frag, resid);
        let mask_src_bgl = desc::layout_for(
            device,
            "stark transform mask src bgl",
            MASK_SRC_SLOTS,
            frag,
            resid,
        );
        let combine_bgl = desc::layout_for(
            device,
            "stark transform combine bgl",
            COMBINE_SLOTS,
            frag,
            resid,
        );

        let gated_layout = desc::pipeline_layout(
            device,
            "stark transform gated layout",
            &[Some(&gated_bgl), Some(&src_bgl)],
        );
        let gated_mask_layout = desc::pipeline_layout(
            device,
            "stark transform gated mask layout",
            &[Some(&gated_bgl), Some(&mask_src_bgl)],
        );
        let quad_layout = desc::pipeline_layout(
            device,
            "stark transform quad layout",
            &[Some(&quad_bgl), Some(&src_bgl)],
        );
        let mask_layout = desc::pipeline_layout(
            device,
            "stark transform mask layout",
            &[Some(&quad_bgl), Some(&mask_src_bgl)],
        );
        let combine_layout = desc::pipeline_layout(
            device,
            "stark transform combine layout",
            &[Some(&combine_bgl)],
        );

        // A negative-determinant affine (a flip) reverses winding, so both faces must
        // draw — which `QUAD_STRIP` already leaves unculled.
        let strip = desc::QUAD_STRIP;
        // Parcels are disjoint; the combine computes rather than blends.
        let target = desc::target;

        // The paint channels, and the mask alone. Both paint pipelines — the moved
        // parcel and the combine — write the residual as a third target where the
        // space has one (§6.7).
        let paint = formats.targets();
        let mask = [target(MASK_FORMAT)];
        // Moved mask coverage lands with **max** blending over the residue: the soft
        // union of what stayed and what arrived (§16.8), and — unlike the paint
        // parcels — safe under any draw order.
        let max_blend = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Max,
        };
        let mask_union = [desc::blended_target(
            MASK_FORMAT,
            Some(wgpu::BlendState {
                color: max_blend,
                alpha: max_blend,
            }),
        )];
        // One quad per source unit, forward-rasterized; and the two fullscreen passes
        // that read a whole destination tile back (`vs_fill`).
        let quad = |label, layout, fs, targets: &[Option<wgpu::ColorTargetState>], vs| {
            desc::render_pipeline(
                device,
                desc::RenderPipe {
                    label,
                    layout,
                    module: &shader,
                    vs,
                    fs,
                    primitive: strip,
                    buffers: &[],
                    targets,
                },
            )
        };
        let parcel_pipeline = quad(
            "stark transform parcel",
            &quad_layout,
            "fs_parcel",
            &paint,
            "vs_quad",
        );
        let mask_pipeline = quad(
            "stark transform mask",
            &mask_layout,
            "fs_mask",
            &mask,
            "vs_quad",
        );
        let parcel_gated_pipeline = quad(
            "stark transform parcel gated",
            &gated_layout,
            "fs_parcel_gated",
            &paint,
            "vs_gated",
        );
        let mask_gated_pipeline = quad(
            "stark transform mask gated",
            &gated_mask_layout,
            "fs_mask_gated",
            &mask_union,
            "vs_gated",
        );
        let combine_pipeline = desc::fullscreen_pipeline(
            device,
            "stark transform combine",
            &combine_layout,
            &shader,
            ("vs_fill", "fs_combine"),
            &paint,
        );
        // The residue is a *gated* pass — it reads `qg`'s source rect (`fs_mask_base`)
        // — so it takes the gated group, like the two pipelines it is drawn beside.
        let mask_base_pipeline = desc::fullscreen_pipeline(
            device,
            "stark transform mask base",
            &gated_mask_layout,
            &shader,
            ("vs_fill", "fs_mask_base"),
            &mask,
        );

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("stark transform sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            ctx: ctx.clone(),
            formats,
            parcel_pipeline,
            mask_pipeline,
            combine_pipeline,
            parcel_gated_pipeline,
            mask_gated_pipeline,
            mask_base_pipeline,
            quad_bgl,
            gated_bgl,
            src_bgl,
            mask_src_bgl,
            combine_bgl,
            sampler,
            zeroes,
            selection,
        }
    }

    /// Transform `base` (one layer's tiles) and `selection` (the author's mask)
    /// under `map`. `None` rejects the whole action — an unusable map, or more
    /// tiles than the caps allow — deterministically, so peers and replays agree
    /// (§16.1).
    pub fn apply(
        &self,
        pool: &TilePool,
        base: &TileMap,
        selection: &Selection,
        map: &TransformMap,
    ) -> Option<(TileMap, Selection)> {
        match map {
            TransformMap::Affine(affine) => self.apply_affine(pool, base, selection, *affine),
            TransformMap::Perspective(_) | TransformMap::Warp(_) => {
                self.apply_gated(pool, base, selection, map)
            }
        }
    }

    /// The whole-plane affine (§16), untouched: one quad per selected source
    /// tile, pure Replace on the mask.
    fn apply_affine(
        &self,
        pool: &TilePool,
        base: &TileMap,
        selection: &Selection,
        affine: Affine2,
    ) -> Option<(TileMap, Selection)> {
        let plan = plan_paint(base, selection, affine)?;
        let mask_plan = plan_mask(selection, affine)?;

        let mut scope = TileScope::new(&self.ctx, "stark transform");

        // Source-tile bind groups are shared across every destination they reach.
        let mut from = Source::new(pool, base, selection);

        let mut tiles = base.clone();
        for (dest, sources) in &plan.rewrites {
            let parcel = self.render_parcel(&mut scope, &mut from, affine, *dest, sources);
            let dst = Channels::acquire(pool, self.formats, AllocSource::TransformDestination);
            self.combine(&mut scope, &from, *dest, parcel.as_ref(), &dst, None);
            tiles = tiles.insert(*dest, dst.into_tile());
            scope.tile_done();
        }
        for coord in &plan.drops {
            tiles = tiles.remove(coord);
        }

        // The mask, carried under the same affine (pure Replace — §16.1).
        let mut mask_tiles: HashTrieMap<TileCoord, crate::gpu::tile::MaskHandle> =
            HashTrieMap::new();
        for (dest, sources) in &mask_plan.rewrites {
            let dst = pool.acquire_mask(AllocSource::TransformMask);
            self.render_mask(&mut scope, selection, affine, *dest, sources, &dst);
            mask_tiles = mask_tiles.insert(*dest, dst);
            scope.tile_done();
        }
        // The hull rides along: the AABB of the affine image of its corners.
        let hull = selection.hull().map(|(lo, hi)| {
            let corners = [
                affine.transform_point2(lo),
                affine.transform_point2(Vec2::new(hi.x, lo.y)),
                affine.transform_point2(Vec2::new(lo.x, hi.y)),
                affine.transform_point2(hi),
            ];
            let min = corners.iter().fold(corners[0], |a, p| a.min(*p));
            let max = corners.iter().fold(corners[0], |a, p| a.max(*p));
            (min, max)
        });
        // A transform *moves* coverage; it does not restrike it, so the opacity
        // the selection is read at rides through unchanged (§6.8).
        let moved_selection = Selection::from_parts(
            mask_tiles,
            selection.outside(),
            selection.level(),
            selection.opacity(),
            hull,
        );

        scope.finish();
        Some((tiles, moved_selection))
    }

    /// The rect-scoped maps (§16.8, §16.9): pieces of `tile ∩ rect` (further
    /// split per warp sub-cell) forward-rasterized through `vs_gated`, the cut
    /// gated by the rect's coverage, and the mask carried as
    /// `max(old · (1 − box), moved)` — the residue unioned with what landed.
    fn apply_gated(
        &self,
        pool: &TilePool,
        base: &TileMap,
        selection: &Selection,
        map: &TransformMap,
    ) -> Option<(TileMap, Selection)> {
        let (rect, geo) = gated_geometry(map)?;
        let plan = plan_gated_paint(base, selection, rect, &geo)?;
        let mask_plan = plan_gated_mask(selection, rect, &geo)?;
        let inv = match &geo.kind {
            GatedKind::Persp { inv, .. } => Some(*inv),
            GatedKind::Warp { .. } => None,
        };

        let mut scope = TileScope::new(&self.ctx, "stark transform gated");

        let mut from = Source::new(pool, base, selection);
        let paint = Gated {
            units: &plan.units,
            inv: inv.as_ref(),
            rect,
        };

        let mut tiles = base.clone();
        for (dest, unit_idxs) in &plan.rewrites {
            let parcel = self.render_gated_parcel(&mut scope, &mut from, &paint, unit_idxs, *dest);
            let dst = Channels::acquire(pool, self.formats, AllocSource::TransformDestination);
            self.combine(&mut scope, &from, *dest, parcel.as_ref(), &dst, Some(rect));
            tiles = tiles.insert(*dest, dst.into_tile());
            scope.tile_done();
        }
        for coord in &plan.drops {
            tiles = tiles.remove(coord);
        }

        // The mask: tiles outside the rect keep their handles; touched and
        // receiving tiles are rebuilt as residue + moved coverage. A universal
        // selection plans no mask work at all and stays universal.
        let moved_selection = if selection.is_universal() {
            selection.clone()
        } else {
            let mut mask_tiles = selection.tile_map().clone();
            for coord in &mask_plan.drops {
                mask_tiles = mask_tiles.remove(coord);
            }
            for (dest, unit_idxs) in &mask_plan.rewrites {
                let dst = pool.acquire_mask(AllocSource::TransformMask);
                let mask = Gated {
                    units: &mask_plan.units,
                    inv: inv.as_ref(),
                    rect,
                };
                self.render_gated_mask(&mut scope, selection, &mask, unit_idxs, *dest, &dst);
                mask_tiles = mask_tiles.insert(*dest, dst);
                scope.tile_done();
            }
            // The hull rides along: what stayed plus wherever the map can have
            // carried coverage, conservatively.
            let hull = selection
                .hull()
                .map(|(lo, hi)| (lo.min(geo.image_aabb.0), hi.max(geo.image_aabb.1)));
            Selection::from_parts(
                mask_tiles,
                selection.outside(),
                selection.level(),
                selection.opacity(),
                hull,
            )
        };

        scope.finish();
        Some((tiles, moved_selection))
    }

    /// Rasterize the pieces reaching `dest` into a fresh scratch pair —
    /// [`render_parcel`](Self::render_parcel)'s shape, with quads generalized
    /// to [`SourceUnit`]s and the deposit gated by the source rect.
    fn render_gated_parcel(
        &self,
        scope: &mut TileScope,
        from: &mut Source<'_>,
        g: &Gated<'_>,
        unit_idxs: &[usize],
        dest: TileCoord,
    ) -> Option<Parcel> {
        if unit_idxs.is_empty() {
            return None;
        }
        let device = &self.ctx.device;
        // The parcel carries the residual it was cut with: the lift scales height
        // alone, so the color — both halves of it — rides through unscaled (§16.2).
        let parcel = Channels::scratch(
            scope,
            from.pool,
            self.formats,
            AllocSource::TransformScratch,
        );

        let mut draws: Vec<(wgpu::BindGroup, wgpu::BindGroup)> = Vec::new();
        for idx in unit_idxs {
            let unit = &g.units[*idx];
            let Some(tile) = from.base.get(&unit.src) else {
                continue;
            };
            let src_bg = from
                .src_bgs
                .entry(unit.src)
                .or_insert_with(|| {
                    // A gating read — the opacity reaches `fs_parcel_gated`
                    // through the piece's uniform (`gated_uniform`).
                    let mask = self.selection.gate_for(from.selection, unit.src);
                    desc::bind_group_for(
                        device,
                        "stark transform src bg",
                        &self.src_bgl,
                        SRC_SLOTS,
                        tile.resid_view().is_some(),
                        |i| {
                            tex(match i {
                                t::SRC_COLOR => tile.color_view(),
                                t::SRC_AUX => tile.aux_view(),
                                t::SRC_MASK => mask.view(),
                                t::SRC_RESID => {
                                    tile.resid_view().expect("a residual space's tile has one")
                                }
                                other => unreachable!("`SRC_SLOTS` lists no binding {other}"),
                            })
                        },
                    )
                })
                .clone();
            draws.push((
                self.gated_bg(scope, unit, g.inv, g.rect, dest, from.selection.opacity()),
                src_bg,
            ));
        }

        let targets = parcel.targets();
        let parcel_att = targets.attachments(desc::CLEAR);
        let mut pass = scope
            .encoder()
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stark transform parcel gated"),
                color_attachments: &parcel_att[..targets.count()],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        pass.set_pipeline(&self.parcel_gated_pipeline);
        for (quad_bg, src_bg) in &draws {
            pass.set_bind_group(0, quad_bg, &[]);
            pass.set_bind_group(1, src_bg, &[]);
            pass.draw(0..4, 0..1);
        }
        drop(pass);
        Some(parcel)
    }

    /// One destination mask tile under a rect-scoped map: the residue
    /// `old · (1 − box)` laid down fullscreen, then the moved coverage pieces
    /// drawn over with max blending — the soft union (§16.8).
    fn render_gated_mask(
        &self,
        scope: &mut TileScope,
        selection: &Selection,
        g: &Gated<'_>,
        unit_idxs: &[usize],
        dest: TileCoord,
        dst: &crate::gpu::tile::MaskHandle,
    ) {
        let device = &self.ctx.device;
        let mask_bg = |view: &wgpu::TextureView| {
            desc::bind_group_for(
                device,
                "stark transform mask src bg",
                &self.mask_src_bgl,
                MASK_SRC_SLOTS,
                false,
                |_| tex(view),
            )
        };
        // The residue reads the destination's *old* coverage — a real tile or
        // the outside constant, through the same clamped-read pattern.
        let old = self.selection.mask_for(selection, dest);
        let base_draw = (self.gated_base_bg(scope, g.rect, dest), mask_bg(&old));
        let mut draws: Vec<(wgpu::BindGroup, wgpu::BindGroup)> = Vec::new();
        for idx in unit_idxs {
            let unit = &g.units[*idx];
            let src = self.selection.mask_for(selection, unit.src);
            draws.push((
                // The mask pass **carries** coverage rather than gating by it, so
                // the opacity rides on the moved `Selection` instead — passing it
                // here would apply it twice, once to the tiles and once to the
                // reading of them (§6.8).
                self.gated_bg(scope, unit, g.inv, g.rect, dest, 1.0),
                mask_bg(&src),
            ));
        }

        let mut pass = scope
            .encoder()
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stark transform mask gated"),
                color_attachments: &[Some(desc::attach(dst.view(), desc::CLEAR))],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        pass.set_pipeline(&self.mask_base_pipeline);
        pass.set_bind_group(0, &base_draw.0, &[]);
        pass.set_bind_group(1, &base_draw.1, &[]);
        pass.draw(0..3, 0..1);
        pass.set_pipeline(&self.mask_gated_pipeline);
        for (quad_bg, src_bg) in &draws {
            pass.set_bind_group(0, quad_bg, &[]);
            pass.set_bind_group(1, src_bg, &[]);
            pass.draw(0..4, 0..1);
        }
    }

    /// The group-0 bind for one gated piece draw.
    /// `opacity` is the author's mask opacity (§6.8); the mask-carrying pass
    /// binds the same uniform and does not read it (`transform.wesl`).
    fn gated_bg(
        &self,
        scope: &mut TileScope,
        unit: &SourceUnit,
        inv: Option<&Homography>,
        rect: (Vec2, Vec2),
        dest: TileCoord,
        opacity: f32,
    ) -> wgpu::BindGroup {
        self.gated_uniform_bg(scope, gated_uniform(unit, inv, rect, dest, opacity))
    }

    /// The group-0 bind for the mask residue pass.
    fn gated_base_bg(
        &self,
        scope: &mut TileScope,
        rect: (Vec2, Vec2),
        dest: TileCoord,
    ) -> wgpu::BindGroup {
        self.gated_uniform_bg(scope, gated_base(rect, dest))
    }

    fn gated_uniform_bg(&self, scope: &mut TileScope, uniform: GatedUniform) -> wgpu::BindGroup {
        let device = &self.ctx.device;
        // Registered with the scope, so it is destroyed at the submit that reads it
        // rather than waiting on the GC (`ScopedResources`). Still a buffer per
        // draw — see the note on `quad_bg`.
        let ubuf = scope.buffer(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stark transform gated uniform"),
                contents: bytemuck::bytes_of(&uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            }),
        );
        desc::bind_group_for(
            device,
            "stark transform gated bg",
            &self.gated_bgl,
            GATED_SLOTS,
            false,
            |i| match i {
                t::QG => ubuf.as_entire_binding(),
                t::SAMP => wgpu::BindingResource::Sampler(&self.sampler),
                other => unreachable!("`GATED_SLOTS` lists no binding {other}"),
            },
        )
    }

    /// Rasterize the transformed source quads reaching `dest` into a fresh
    /// scratch pair: `(premult color as-is, height·mask)` — the moved parcel.
    /// `None` when nothing reaches this tile (a cut with no incoming paint).
    fn render_parcel(
        &self,
        scope: &mut TileScope,
        from: &mut Source<'_>,
        affine: Affine2,
        dest: TileCoord,
        sources: &[TileCoord],
    ) -> Option<Parcel> {
        if sources.is_empty() {
            return None;
        }
        let device = &self.ctx.device;
        // The parcel carries the residual it was cut with: the lift scales height
        // alone, so the color — both halves of it — rides through unscaled (§16.2).
        let parcel = Channels::scratch(
            scope,
            from.pool,
            self.formats,
            AllocSource::TransformScratch,
        );

        let mut draws: Vec<(wgpu::BindGroup, wgpu::BindGroup)> = Vec::new();
        for src in sources {
            let Some(tile) = from.base.get(src) else {
                continue;
            };
            let src_bg = from
                .src_bgs
                .entry(*src)
                .or_insert_with(|| {
                    // A gating read — the opacity reaches `fs_parcel` through
                    // the quad's uniform (`quad_uniform`).
                    let mask = self.selection.gate_for(from.selection, *src);
                    desc::bind_group_for(
                        device,
                        "stark transform src bg",
                        &self.src_bgl,
                        SRC_SLOTS,
                        tile.resid_view().is_some(),
                        |i| {
                            tex(match i {
                                t::SRC_COLOR => tile.color_view(),
                                t::SRC_AUX => tile.aux_view(),
                                t::SRC_MASK => mask.view(),
                                t::SRC_RESID => {
                                    tile.resid_view().expect("a residual space's tile has one")
                                }
                                other => unreachable!("`SRC_SLOTS` lists no binding {other}"),
                            })
                        },
                    )
                })
                .clone();
            draws.push((
                self.quad_bg(scope, affine, *src, dest, from.selection.opacity()),
                src_bg,
            ));
        }

        let targets = parcel.targets();
        let parcel_att = targets.attachments(desc::CLEAR);
        let mut pass = scope
            .encoder()
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stark transform parcel"),
                color_attachments: &parcel_att[..targets.count()],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        pass.set_pipeline(&self.parcel_pipeline);
        for (quad_bg, src_bg) in &draws {
            pass.set_bind_group(0, quad_bg, &[]);
            pass.set_bind_group(1, src_bg, &[]);
            pass.draw(0..4, 0..1);
        }
        drop(pass);
        Some(parcel)
    }

    /// Cut `dest`'s base by its own mask and stack the parcel over it, into the
    /// fresh CoW `(color, aux)` pair `dst`. `gate` scopes the cut to a source
    /// rect (§16.8); `None` is the affine's whole-plane cut, arithmetically
    /// untouched.
    fn combine(
        &self,
        scope: &mut TileScope,
        from: &Source<'_>,
        dest: TileCoord,
        parcel: Option<&Parcel>,
        dst: &Parcel,
        gate: Option<(Vec2, Vec2)>,
    ) {
        let device = &self.ctx.device;
        let ubuf = scope.buffer(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stark transform combine uniform"),
                contents: bytemuck::bytes_of(&combine_uniform(
                    dest,
                    gate,
                    from.selection.opacity(),
                )),
                usage: wgpu::BufferUsages::UNIFORM,
            }),
        );
        let (base_color, base_aux) = match from.base.get(&dest) {
            Some(tile) => (tile.color_view().clone(), tile.aux_view().clone()),
            None => (self.zeroes.color.clone(), self.zeroes.aux.clone()),
        };
        // A gating read: the base is cut by `coverage · opacity`, the identical
        // factor the parcel side took at the source (`combine_uniform` above).
        let base_mask = self.selection.gate_for(from.selection, dest);
        let (parcel_color, parcel_aux) = match parcel {
            Some(p) => (p.color.view().clone(), p.aux.view().clone()),
            None => (self.zeroes.color.clone(), self.zeroes.aux.clone()),
        };
        // A virgin destination and a cut-only tile read the 1×1 zero for the residual
        // exactly as they do for the color — the combine is one shader whatever
        // exists (§6.8's pattern).
        let (base_resid, parcel_resid) = match &self.zeroes.resid {
            Some(zero) => (
                Some(
                    from.base
                        .get(&dest)
                        .and_then(|t| t.resid_view())
                        .unwrap_or(zero)
                        .clone(),
                ),
                Some(
                    parcel
                        .and_then(|p| p.resid.as_ref())
                        .map_or_else(|| zero.clone(), |r| r.view().clone()),
                ),
            ),
            None => (None, None),
        };
        let bg = desc::bind_group_for(
            device,
            "stark transform combine bg",
            &self.combine_bgl,
            COMBINE_SLOTS,
            base_resid.is_some() && parcel_resid.is_some(),
            |i| match i {
                t::BASE_COLOR => tex(&base_color),
                t::BASE_AUX => tex(&base_aux),
                t::BASE_MASK => tex(base_mask.view()),
                t::PARCEL_COLOR => tex(&parcel_color),
                t::PARCEL_AUX => tex(&parcel_aux),
                t::QC => ubuf.as_entire_binding(),
                t::BASE_RESID => tex(base_resid.as_ref().expect("a residual build has one")),
                t::PARCEL_RESID => tex(parcel_resid.as_ref().expect("a residual build has one")),
                other => unreachable!("`COMBINE_SLOTS` lists no binding {other}"),
            },
        );
        scope.fullscreen_pass(
            "stark transform combine",
            &self.combine_pipeline,
            &bg,
            &[],
            dst.targets(),
            desc::CLEAR,
        );
    }

    /// One destination mask tile: cleared to the coverage that reigns outside the
    /// mask's tiles, with the transformed source mask quads drawn over.
    fn render_mask(
        &self,
        scope: &mut TileScope,
        selection: &Selection,
        affine: Affine2,
        dest: TileCoord,
        sources: &[TileCoord],
        dst: &crate::gpu::tile::MaskHandle,
    ) {
        let device = &self.ctx.device;
        let mut draws: Vec<(wgpu::BindGroup, wgpu::BindGroup)> = Vec::new();
        for src in sources {
            let Some(handle) = selection.tile(*src) else {
                continue;
            };
            let src_bg = desc::bind_group_for(
                device,
                "stark transform mask src bg",
                &self.mask_src_bgl,
                MASK_SRC_SLOTS,
                false,
                |_| tex(handle.view()),
            );
            // 1.0: this pass carries the mask, it does not gate by it — the
            // opacity rides on the moved `Selection` (§6.8).
            draws.push((self.quad_bg(scope, affine, *src, dest, 1.0), src_bg));
        }

        let mut pass = scope
            .encoder()
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stark transform mask"),
                color_attachments: &[Some(desc::attach(dst.view(), outside_clear(selection)))],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        pass.set_pipeline(&self.mask_pipeline);
        for (quad_bg, src_bg) in &draws {
            pass.set_bind_group(0, quad_bg, &[]);
            pass.set_bind_group(1, src_bg, &[]);
            pass.draw(0..4, 0..1);
        }
    }

    /// The group-0 bind for one quad draw: its uniform plus the shared sampler.
    ///
    /// Still a buffer and a bind group **per draw**, where the fill and the selection
    /// now take a dynamic-offset slot apiece (`UniformSlots`). The difference is that
    /// those know their tile count before they encode anything, so one buffer can be
    /// sized up front; a transform's draw count is the sum over its plan of the
    /// sources reaching each destination, and growing a slot buffer mid-encode would
    /// reallocate under the bind groups already recorded against it. The buffers are
    /// at least destroyed at their submit now rather than left for the GC; slotting
    /// them properly means pre-counting the plan, which is a change to the plan's
    /// shape rather than to this function.
    fn quad_bg(
        &self,
        scope: &mut TileScope,
        affine: Affine2,
        src: TileCoord,
        dest: TileCoord,
        opacity: f32,
    ) -> wgpu::BindGroup {
        let device = &self.ctx.device;
        let ubuf = scope.buffer(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stark transform quad uniform"),
                contents: bytemuck::bytes_of(&quad_uniform(affine, src, dest, opacity)),
                usage: wgpu::BufferUsages::UNIFORM,
            }),
        );
        desc::bind_group_for(
            device,
            "stark transform quad bg",
            &self.quad_bgl,
            QUAD_SLOTS,
            false,
            |i| match i {
                t::Q => ubuf.as_entire_binding(),
                t::SAMP => wgpu::BindingResource::Sampler(&self.sampler),
                other => unreachable!("`QUAD_SLOTS` lists no binding {other}"),
            },
        )
    }
}
