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
    FragMap, GatedKind, Homography, SourceUnit, TransformMap, gated_geometry, plan_gated_mask,
    plan_gated_paint, plan_mask, plan_paint,
};
use crate::geom::{Affine2, Mat2, TILE_APRON, TILE_SIZE, TILE_TEX, TileCoord, Vec2};
use crate::gpu::context::GpuContext;
use crate::gpu::desc::{self, Zeroes};
use crate::gpu::selection::{SelectionRenderer, outside_clear};
use crate::gpu::tile::{AllocSource, MASK_FORMAT, TexHandle, TileMap, TilePairHandle, TilePool};

// Generated from `transform.wesl`'s own declarations (§6.7). The three constructors
// below are free functions rather than inherent impls: the types live in
// `stark-shaders` now, and an inherent impl on another crate's type is not allowed.
// Each is still the only way one is built.
use stark_shaders::mirror::transform::{
    Combine as CombineUniform, Gated as GatedUniform, Quad as QuadUniform,
};

/// One source tile's interior quad, drawn into `dest`'s texture (paint and mask
/// tiles share the `TILE_TEX` geometry).
fn quad_uniform(affine: Affine2, src: TileCoord, dest: TileCoord) -> QuadUniform {
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
        u: [src_origin.x, src_origin.y, TILE_SIZE as f32, 0.0],
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
) -> GatedUniform {
    let mut u = gated_base(rect, dest);
    let c = &unit.corners;
    u.c0 = [c[0].x, c[0].y, c[1].x, c[1].y];
    u.c1 = [c[2].x, c[2].y, c[3].x, c[3].y];
    let src_tex_origin = unit.src.origin() - Vec2::splat(TILE_APRON as f32);
    u.t = [src_tex_origin.x, src_tex_origin.y, 0.0, 0.0];
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
fn combine_uniform(dest: TileCoord, gate: Option<(Vec2, Vec2)>) -> CombineUniform {
    let dest_origin = dest.origin() - Vec2::splat(TILE_APRON as f32);
    match gate {
        Some(rect) => CombineUniform {
            a: [dest_origin.x, dest_origin.y, 1.0, 0.0],
            r: [rect.0.x, rect.0.y, rect.1.x, rect.1.y],
        },
        None => CombineUniform {
            a: [dest_origin.x, dest_origin.y, 0.0, 0.0],
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

/// A transform's command encoder together with the scratch parcels its passes will
/// read — and the reason they are one type.
///
/// Nothing in a recorded encoder has *run*. A parcel is written by one pass and read
/// by the `combine` after it, but both are only recorded, so releasing the parcel's
/// handle early hands its texture straight back to the pool, which gives it to the
/// next destination tile in this very encoder — whose parcel pass overwrites it
/// before the earlier tile's combine ever reads it. The corruption would be a
/// transform that smears one tile's paint into another's, on large selections only,
/// and no test would name it.
///
/// That rule used to live in a comment and a `drop(scratch)` placed after the submit
/// by hand, twice. Here [`Self::submit`] takes `self`, so the parcels cannot be
/// released except by submitting first; and dropping the recording *without*
/// submitting is safe too, because then no pass ran at all.
struct Recording {
    encoder: wgpu::CommandEncoder,
    scratch: Vec<TexHandle>,
}

/// A tile's paint channels as this module passes them around: the moved parcel a
/// quad pass writes, and the destination a combine writes. `None` in the third slot
/// is a space with no residual (§6.7) — the same `Option` `TilePairHandle` carries,
/// before it becomes one.
type Parcel = (TexHandle, TexHandle, Option<TexHandle>);

impl Recording {
    fn new(device: &wgpu::Device, label: &str) -> Self {
        Self {
            encoder: device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) }),
            scratch: Vec::new(),
        }
    }

    /// Hold a parcel until this recording is submitted.
    fn keep(&mut self, parcel: Option<Parcel>) {
        if let Some((color, aux, resid)) = parcel {
            self.scratch.push(color);
            self.scratch.push(aux);
            self.scratch.extend(resid);
        }
    }

    /// Submit, then release the parcels — in that order, which is the whole point.
    fn submit(self, queue: &wgpu::Queue) {
        queue.submit([self.encoder.finish()]);
    }
}

#[derive(Clone)]
pub struct TransformRenderer {
    ctx: GpuContext,
    color_format: wgpu::TextureFormat,
    aux_format: wgpu::TextureFormat,
    /// The residual channel's format, or `None` in a space that has none (§6.7).
    resid_format: Option<wgpu::TextureFormat>,
    parcel_pipeline: wgpu::RenderPipeline,
    mask_pipeline: wgpu::RenderPipeline,
    combine_pipeline: wgpu::RenderPipeline,
    /// The rect-scoped maps' passes (§16.8, §16.9): pieces through `vs_gated`,
    /// and the mask's residue + max-blended moved coverage.
    parcel_gated_pipeline: wgpu::RenderPipeline,
    mask_gated_pipeline: wgpu::RenderPipeline,
    mask_base_pipeline: wgpu::RenderPipeline,
    quad_bgl: wgpu::BindGroupLayout,
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
        let color_format = color_space.color_format();
        let aux_format = color_space.aux_format();
        let resid_format = color_space.resid_format();
        let resid = resid_format.is_some();

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark transform"),
            source: wgpu::ShaderSource::Wgsl(stark_shaders::transform(resid).into()),
        });

        let frag = wgpu::ShaderStages::FRAGMENT;
        let quad_bgl = desc::bind_group_layout(
            device,
            "stark transform quad bgl",
            &[
                // The vertex stage places the quad through the forward affine; the
                // fragment stage taps the source through the inverse.
                desc::uniform(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
                desc::sampler(1, frag),
            ],
        );
        let mut src_entries = vec![
            desc::sample_tex(0, frag),
            desc::sample_tex(1, frag),
            desc::sample_tex(2, frag),
        ];
        // 3, the source tile's residual (§6.7): carried under the same map and
        // sampled at the same uv as the colour it belongs to.
        if resid {
            src_entries.push(desc::sample_tex(3, frag));
        }
        let src_bgl = desc::bind_group_layout(device, "stark transform src bgl", &src_entries);
        // The mask pass reads only the source mask (binding 2).
        let mask_src_bgl = desc::bind_group_layout(
            device,
            "stark transform mask src bgl",
            &[desc::sample_tex(2, frag)],
        );
        let mut combine_entries = vec![
            desc::load_tex(2, frag),
            desc::load_tex(3, frag),
            desc::load_tex(4, frag),
            desc::load_tex(5, frag),
            desc::load_tex(6, frag),
            // The gate rect (binding 7): zeroed for the affine's whole-plane cut,
            // the source rect for perspective/warp.
            desc::uniform(7, frag),
        ];
        // 8 and 9, the base's and the parcel's residuals — past the gate rect because
        // 7 was already taken when they were added, and the shader says the same.
        if resid {
            combine_entries.push(desc::load_tex(8, frag));
            combine_entries.push(desc::load_tex(9, frag));
        }
        let combine_bgl =
            desc::bind_group_layout(device, "stark transform combine bgl", &combine_entries);

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
        let mut paint = vec![target(color_format), target(aux_format)];
        if let Some(f) = resid_format {
            paint.push(target(f));
        }
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
            &quad_layout,
            "fs_parcel_gated",
            &paint,
            "vs_gated",
        );
        let mask_gated_pipeline = quad(
            "stark transform mask gated",
            &mask_layout,
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
        let mask_base_pipeline = desc::fullscreen_pipeline(
            device,
            "stark transform mask base",
            &mask_layout,
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
            color_format,
            aux_format,
            resid_format,
            parcel_pipeline,
            mask_pipeline,
            combine_pipeline,
            parcel_gated_pipeline,
            mask_gated_pipeline,
            mask_base_pipeline,
            quad_bgl,
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

        let device = &self.ctx.device;
        let mut rec = Recording::new(device, "stark transform");

        // Source-tile bind groups are shared across every destination they reach.
        let mut from = Source::new(pool, base, selection);

        let mut tiles = base.clone();
        for (dest, sources) in &plan.rewrites {
            let parcel = self.render_parcel(&mut rec.encoder, &mut from, affine, *dest, sources);
            let dst = (
                pool.acquire_tex(self.color_format, AllocSource::TransformDestination),
                pool.acquire_tex(self.aux_format, AllocSource::TransformDestination),
                self.resid_format
                    .map(|f| pool.acquire_tex(f, AllocSource::TransformDestination)),
            );
            self.combine(&mut rec.encoder, &from, *dest, parcel.as_ref(), &dst, None);
            rec.keep(parcel);
            tiles = tiles.insert(*dest, TilePairHandle::new(dst.0, dst.1, dst.2));
        }
        for coord in &plan.drops {
            tiles = tiles.remove(coord);
        }

        // The mask, carried under the same affine (pure Replace — §16.1).
        let mut mask_tiles: HashTrieMap<TileCoord, crate::gpu::tile::MaskHandle> =
            HashTrieMap::new();
        for (dest, sources) in &mask_plan.rewrites {
            let dst = pool.acquire_mask(AllocSource::TransformMask);
            self.render_mask(&mut rec.encoder, selection, affine, *dest, sources, &dst);
            mask_tiles = mask_tiles.insert(*dest, dst);
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
        // A transform *moves* coverage; it does not restrike it, so the strength
        // the selection was drawn at rides through unchanged (§6.8).
        let moved_selection =
            Selection::from_parts(mask_tiles, selection.outside(), selection.level(), hull);

        rec.submit(&self.ctx.queue);
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

        let device = &self.ctx.device;
        let mut rec = Recording::new(device, "stark transform gated");

        let mut from = Source::new(pool, base, selection);
        let paint = Gated {
            units: &plan.units,
            inv: inv.as_ref(),
            rect,
        };

        let mut tiles = base.clone();
        for (dest, unit_idxs) in &plan.rewrites {
            let parcel =
                self.render_gated_parcel(&mut rec.encoder, &mut from, &paint, unit_idxs, *dest);
            let dst = (
                pool.acquire_tex(self.color_format, AllocSource::TransformDestination),
                pool.acquire_tex(self.aux_format, AllocSource::TransformDestination),
                self.resid_format
                    .map(|f| pool.acquire_tex(f, AllocSource::TransformDestination)),
            );
            self.combine(
                &mut rec.encoder,
                &from,
                *dest,
                parcel.as_ref(),
                &dst,
                Some(rect),
            );
            rec.keep(parcel);
            tiles = tiles.insert(*dest, TilePairHandle::new(dst.0, dst.1, dst.2));
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
                self.render_gated_mask(&mut rec.encoder, selection, &mask, unit_idxs, *dest, &dst);
                mask_tiles = mask_tiles.insert(*dest, dst);
            }
            // The hull rides along: what stayed plus wherever the map can have
            // carried coverage, conservatively.
            let hull = selection
                .hull()
                .map(|(lo, hi)| (lo.min(geo.image_aabb.0), hi.max(geo.image_aabb.1)));
            Selection::from_parts(mask_tiles, selection.outside(), selection.level(), hull)
        };

        rec.submit(&self.ctx.queue);
        Some((tiles, moved_selection))
    }

    /// Rasterize the pieces reaching `dest` into a fresh scratch pair —
    /// [`render_parcel`](Self::render_parcel)'s shape, with quads generalized
    /// to [`SourceUnit`]s and the deposit gated by the source rect.
    fn render_gated_parcel(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        from: &mut Source<'_>,
        g: &Gated<'_>,
        unit_idxs: &[usize],
        dest: TileCoord,
    ) -> Option<Parcel> {
        if unit_idxs.is_empty() {
            return None;
        }
        let device = &self.ctx.device;
        let color = from
            .pool
            .acquire_tex(self.color_format, AllocSource::TransformScratch);
        let aux = from
            .pool
            .acquire_tex(self.aux_format, AllocSource::TransformScratch);
        // The parcel carries the residual it was cut with: the lift scales height
        // alone, so the colour — both halves of it — rides through unscaled (§16.2).
        let resid = self
            .resid_format
            .map(|f| from.pool.acquire_tex(f, AllocSource::TransformScratch));

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
                    let mask = self.selection.mask_for(from.selection, unit.src);
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("stark transform src bg"),
                        layout: &self.src_bgl,
                        entries: &{
                            let mut e = vec![
                                desc::tex(0, tile.color_view()),
                                desc::tex(1, tile.aux_view()),
                                desc::tex(2, &mask),
                            ];
                            if let Some(view) = tile.resid_view() {
                                e.push(desc::tex(3, view));
                            }
                            e
                        },
                    })
                })
                .clone();
            draws.push((self.gated_bg(unit, g.inv, g.rect, dest), src_bg));
        }

        let (parcel_att, parcel_n) = desc::tile_attachments(
            color.view(),
            aux.view(),
            resid.as_ref().map(TexHandle::view),
            desc::CLEAR,
        );
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("stark transform parcel gated"),
            color_attachments: &parcel_att[..parcel_n],
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
        Some((color, aux, resid))
    }

    /// One destination mask tile under a rect-scoped map: the residue
    /// `old · (1 − box)` laid down fullscreen, then the moved coverage pieces
    /// drawn over with max blending — the soft union (§16.8).
    fn render_gated_mask(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        selection: &Selection,
        g: &Gated<'_>,
        unit_idxs: &[usize],
        dest: TileCoord,
        dst: &crate::gpu::tile::MaskHandle,
    ) {
        let device = &self.ctx.device;
        let mask_bg = |view: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("stark transform mask src bg"),
                layout: &self.mask_src_bgl,
                entries: &[desc::tex(2, view)],
            })
        };
        // The residue reads the destination's *old* coverage — a real tile or
        // the outside constant, through the same clamped-read pattern.
        let old = self.selection.mask_for(selection, dest);
        let base_draw = (self.gated_base_bg(g.rect, dest), mask_bg(&old));
        let mut draws: Vec<(wgpu::BindGroup, wgpu::BindGroup)> = Vec::new();
        for idx in unit_idxs {
            let unit = &g.units[*idx];
            let src = self.selection.mask_for(selection, unit.src);
            draws.push((self.gated_bg(unit, g.inv, g.rect, dest), mask_bg(&src)));
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
    fn gated_bg(
        &self,
        unit: &SourceUnit,
        inv: Option<&Homography>,
        rect: (Vec2, Vec2),
        dest: TileCoord,
    ) -> wgpu::BindGroup {
        self.gated_uniform_bg(gated_uniform(unit, inv, rect, dest))
    }

    /// The group-0 bind for the mask residue pass.
    fn gated_base_bg(&self, rect: (Vec2, Vec2), dest: TileCoord) -> wgpu::BindGroup {
        self.gated_uniform_bg(gated_base(rect, dest))
    }

    fn gated_uniform_bg(&self, uniform: GatedUniform) -> wgpu::BindGroup {
        let device = &self.ctx.device;
        let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stark transform gated uniform"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark transform gated bg"),
            layout: &self.quad_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ubuf.as_entire_binding(),
                },
                desc::samp(1, &self.sampler),
            ],
        })
    }

    /// Rasterize the transformed source quads reaching `dest` into a fresh
    /// scratch pair: `(premult color as-is, height·mask)` — the moved parcel.
    /// `None` when nothing reaches this tile (a cut with no incoming paint).
    fn render_parcel(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        from: &mut Source<'_>,
        affine: Affine2,
        dest: TileCoord,
        sources: &[TileCoord],
    ) -> Option<Parcel> {
        if sources.is_empty() {
            return None;
        }
        let device = &self.ctx.device;
        let color = from
            .pool
            .acquire_tex(self.color_format, AllocSource::TransformScratch);
        let aux = from
            .pool
            .acquire_tex(self.aux_format, AllocSource::TransformScratch);
        // The parcel carries the residual it was cut with: the lift scales height
        // alone, so the colour — both halves of it — rides through unscaled (§16.2).
        let resid = self
            .resid_format
            .map(|f| from.pool.acquire_tex(f, AllocSource::TransformScratch));

        let mut draws: Vec<(wgpu::BindGroup, wgpu::BindGroup)> = Vec::new();
        for src in sources {
            let Some(tile) = from.base.get(src) else {
                continue;
            };
            let src_bg = from
                .src_bgs
                .entry(*src)
                .or_insert_with(|| {
                    let mask = self.selection.mask_for(from.selection, *src);
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("stark transform src bg"),
                        layout: &self.src_bgl,
                        entries: &{
                            let mut e = vec![
                                desc::tex(0, tile.color_view()),
                                desc::tex(1, tile.aux_view()),
                                desc::tex(2, &mask),
                            ];
                            if let Some(view) = tile.resid_view() {
                                e.push(desc::tex(3, view));
                            }
                            e
                        },
                    })
                })
                .clone();
            draws.push((self.quad_bg(affine, *src, dest), src_bg));
        }

        let (parcel_att, parcel_n) = desc::tile_attachments(
            color.view(),
            aux.view(),
            resid.as_ref().map(TexHandle::view),
            desc::CLEAR,
        );
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("stark transform parcel"),
            color_attachments: &parcel_att[..parcel_n],
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
        Some((color, aux, resid))
    }

    /// Cut `dest`'s base by its own mask and stack the parcel over it, into the
    /// fresh CoW `(color, aux)` pair `dst`. `gate` scopes the cut to a source
    /// rect (§16.8); `None` is the affine's whole-plane cut, arithmetically
    /// untouched.
    fn combine(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        from: &Source<'_>,
        dest: TileCoord,
        parcel: Option<&Parcel>,
        dst: &Parcel,
        gate: Option<(Vec2, Vec2)>,
    ) {
        let device = &self.ctx.device;
        let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stark transform combine uniform"),
            contents: bytemuck::bytes_of(&combine_uniform(dest, gate)),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let (base_color, base_aux) = match from.base.get(&dest) {
            Some(tile) => (tile.color_view().clone(), tile.aux_view().clone()),
            None => (self.zeroes.color.clone(), self.zeroes.aux.clone()),
        };
        let base_mask = self.selection.mask_for(from.selection, dest);
        let (parcel_color, parcel_aux) = match parcel {
            Some((c, a, _)) => (c.view().clone(), a.view().clone()),
            None => (self.zeroes.color.clone(), self.zeroes.aux.clone()),
        };
        // A virgin destination and a cut-only tile read the 1×1 zero for the residual
        // exactly as they do for the colour — the combine is one shader whatever
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
                        .and_then(|(_, _, r)| r.as_ref())
                        .map_or_else(|| zero.clone(), |r| r.view().clone()),
                ),
            ),
            None => (None, None),
        };
        let mut entries = vec![
            desc::tex(2, &base_color),
            desc::tex(3, &base_aux),
            desc::tex(4, &base_mask),
            desc::tex(5, &parcel_color),
            desc::tex(6, &parcel_aux),
            wgpu::BindGroupEntry {
                binding: 7,
                resource: ubuf.as_entire_binding(),
            },
        ];
        if let (Some(b), Some(p)) = (&base_resid, &parcel_resid) {
            entries.push(desc::tex(8, b));
            entries.push(desc::tex(9, p));
        }
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark transform combine bg"),
            layout: &self.combine_bgl,
            entries: &entries,
        });
        let (dst_att, dst_n) = desc::tile_attachments(
            dst.0.view(),
            dst.1.view(),
            dst.2.as_ref().map(TexHandle::view),
            desc::CLEAR,
        );
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("stark transform combine"),
            color_attachments: &dst_att[..dst_n],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.combine_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }

    /// One destination mask tile: cleared to the coverage that reigns outside the
    /// mask's tiles, with the transformed source mask quads drawn over.
    fn render_mask(
        &self,
        encoder: &mut wgpu::CommandEncoder,
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
            let src_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("stark transform mask src bg"),
                layout: &self.mask_src_bgl,
                entries: &[desc::tex(2, handle.view())],
            });
            draws.push((self.quad_bg(affine, *src, dest), src_bg));
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
    fn quad_bg(&self, affine: Affine2, src: TileCoord, dest: TileCoord) -> wgpu::BindGroup {
        let device = &self.ctx.device;
        let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stark transform quad uniform"),
            contents: bytemuck::bytes_of(&quad_uniform(affine, src, dest)),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark transform quad bg"),
            layout: &self.quad_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ubuf.as_entire_binding(),
                },
                desc::samp(1, &self.sampler),
            ],
        })
    }
}
