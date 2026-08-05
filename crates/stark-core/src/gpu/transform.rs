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

use bytemuck::{Pod, Zeroable};
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
use crate::gpu::selection::SelectionRenderer;
use crate::gpu::tile::{AllocSource, MASK_FORMAT, TexHandle, TileMap, TilePairHandle, TilePool};

/// Mirrors `Quad` in `transform.wesl`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct QuadUniform {
    a: [f32; 4], // dest tex origin .xy, TILE_TEX, _
    m: [f32; 4], // forward affine linear part, rows (vertex: coverage)
    t: [f32; 4], // forward translation .xy, src *texture* origin .zw
    u: [f32; 4], // src interior origin .xy, TILE_SIZE, _
    i: [f32; 4], // inverse affine linear part, rows (fragment: the source tap)
    j: [f32; 4], // inverse translation .xy, _, _
}

impl QuadUniform {
    /// One source tile's interior quad, drawn into `dest`'s texture (paint and
    /// mask tiles share the `TILE_TEX` geometry).
    fn new(affine: Affine2, src: TileCoord, dest: TileCoord) -> Self {
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
        Self {
            a: [dest_origin.x, dest_origin.y, TILE_TEX as f32, 0.0],
            // Shader rows: c.x = m.x·p.x + m.y·p.y; glam's Mat2 is column-major.
            m: [m.x_axis.x, m.y_axis.x, m.x_axis.y, m.y_axis.y],
            t: [t.x, t.y, src_tex_origin.x, src_tex_origin.y],
            u: [src_origin.x, src_origin.y, TILE_SIZE as f32, 0.0],
            i: [im.x_axis.x, im.y_axis.x, im.x_axis.y, im.y_axis.y],
            j: [it.x, it.y, 0.0, 0.0],
        }
    }
}

/// Mirrors `Gated` in `transform.wesl` — one drawn piece of a rect-scoped map
/// (§16.8, §16.9).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GatedUniform {
    a: [f32; 4],  // dest tex origin .xy, TILE_TEX, fragment mode
    c0: [f32; 4], // piece corners (00, 10)
    c1: [f32; 4], // piece corners (01, 11)
    i0: [f32; 4], // inverse homography rows (mode 0)
    i1: [f32; 4],
    i2: [f32; 4],
    g0: [f32; 4], // warp cell corner images (00, 10) (mode 1)
    g1: [f32; 4], // (01, 11)
    s: [f32; 4],  // warp cell source sub-rect: min .xy, size .zw
    r: [f32; 4],  // gate rect: min .xy, max .zw
    t: [f32; 4],  // src tile texture origin .xy
}

impl GatedUniform {
    /// One [`SourceUnit`] drawn into `dest`'s texture. `inv` is the map's
    /// shared inverse homography (perspective); warp cells carry their own
    /// fragment map in the unit.
    fn new(
        unit: &SourceUnit,
        inv: Option<&Homography>,
        rect: (Vec2, Vec2),
        dest: TileCoord,
    ) -> Self {
        let mut u = Self::base(rect, dest);
        let c = &unit.corners;
        u.c0 = [c[0].x, c[0].y, c[1].x, c[1].y];
        u.c1 = [c[2].x, c[2].y, c[3].x, c[3].y];
        let src_tex_origin = unit.src.origin() - Vec2::splat(TILE_APRON as f32);
        u.t = [src_tex_origin.x, src_tex_origin.y, 0.0, 0.0];
        match &unit.frag {
            FragMap::Persp => {
                let h = inv.expect("perspective units carry a shared inverse");
                u.set_rows(&h.rows);
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
                    u.set_rows(&Homography::from_affine(fwd.inverse()).rows);
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

    /// The uniform for `fs_mask_base`: only the destination origin and the
    /// gate rect matter — the residue pass has no piece.
    fn base(rect: (Vec2, Vec2), dest: TileCoord) -> Self {
        let dest_origin = dest.origin() - Vec2::splat(TILE_APRON as f32);
        let mut u = Self::zeroed();
        u.a = [dest_origin.x, dest_origin.y, TILE_TEX as f32, 0.0];
        u.r = [rect.0.x, rect.0.y, rect.1.x, rect.1.y];
        u
    }

    fn set_rows(&mut self, rows: &[[f32; 3]; 3]) {
        self.i0 = [rows[0][0], rows[0][1], rows[0][2], 0.0];
        self.i1 = [rows[1][0], rows[1][1], rows[1][2], 0.0];
        self.i2 = [rows[2][0], rows[2][1], rows[2][2], 0.0];
    }
}

/// Mirrors `Combine` in `transform.wesl`: whether (and where) the cut is gated
/// by a source rect. The affine path binds the zero gate, whose arithmetic is
/// untouched from before the gate existed.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CombineUniform {
    a: [f32; 4],
    r: [f32; 4],
}

impl CombineUniform {
    fn new(dest: TileCoord, gate: Option<(Vec2, Vec2)>) -> Self {
        let dest_origin = dest.origin() - Vec2::splat(TILE_APRON as f32);
        match gate {
            Some(rect) => Self {
                a: [dest_origin.x, dest_origin.y, 1.0, 0.0],
                r: [rect.0.x, rect.0.y, rect.1.x, rect.1.y],
            },
            None => Self {
                a: [dest_origin.x, dest_origin.y, 0.0, 0.0],
                r: [0.0; 4],
            },
        }
    }
}

#[derive(Clone)]
pub struct TransformRenderer {
    ctx: GpuContext,
    color_format: wgpu::TextureFormat,
    aux_format: wgpu::TextureFormat,
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
    /// 1×1 zero color/aux — the base of a virgin destination and the parcel of a
    /// cut-only tile, so the combine is one shader whatever exists.
    zero_color: wgpu::TextureView,
    zero_aux: wgpu::TextureView,
    /// For the selection constants (0/1 coverage) bound where a mask has no tile.
    selection: SelectionRenderer,
}

impl TransformRenderer {
    pub fn new(
        ctx: &GpuContext,
        color_space: &dyn ColorSpace,
        selection: SelectionRenderer,
    ) -> Self {
        let device = &ctx.device;
        let color_format = color_space.color_format();
        let aux_format = color_space.aux_format();

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark transform"),
            source: wgpu::ShaderSource::Wgsl(stark_shaders::transform().into()),
        });

        let sample_tex = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let load_tex = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                // Read with `textureLoad` only, clamped to the bound extent.
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };

        let quad_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stark transform quad bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    // The vertex stage places the quad through the forward affine;
                    // the fragment stage taps the source through the inverse.
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let src_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stark transform src bgl"),
            entries: &[sample_tex(0), sample_tex(1), sample_tex(2)],
        });
        // The mask pass reads only the source mask (binding 2).
        let mask_src_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stark transform mask src bgl"),
            entries: &[sample_tex(2)],
        });
        let combine_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stark transform combine bgl"),
            entries: &[
                load_tex(2),
                load_tex(3),
                load_tex(4),
                load_tex(5),
                load_tex(6),
                // The gate rect (binding 7): zeroed for the affine's
                // whole-plane cut, the source rect for perspective/warp.
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let quad_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("stark transform quad layout"),
            bind_group_layouts: &[Some(&quad_bgl), Some(&src_bgl)],
            immediate_size: 0,
        });
        let mask_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("stark transform mask layout"),
            bind_group_layouts: &[Some(&quad_bgl), Some(&mask_src_bgl)],
            immediate_size: 0,
        });
        let combine_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("stark transform combine layout"),
            bind_group_layouts: &[Some(&combine_bgl)],
            immediate_size: 0,
        });

        let strip = wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            // A negative-determinant affine (a flip) reverses winding; both faces
            // must draw.
            cull_mode: None,
            ..Default::default()
        };
        let target = |format| {
            Some(wgpu::ColorTargetState {
                format,
                blend: None, // parcels are disjoint; the combine computes, not blends
                write_mask: wgpu::ColorWrites::ALL,
            })
        };

        let parcel_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("stark transform parcel"),
            layout: Some(&quad_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_quad"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: strip,
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_parcel"),
                compilation_options: Default::default(),
                targets: &[target(color_format), target(aux_format)],
            }),
            multiview_mask: None,
            cache: None,
        });
        let mask_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("stark transform mask"),
            layout: Some(&mask_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_quad"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: strip,
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_mask"),
                compilation_options: Default::default(),
                targets: &[target(MASK_FORMAT)],
            }),
            multiview_mask: None,
            cache: None,
        });
        let combine_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("stark transform combine"),
            layout: Some(&combine_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_fill"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_combine"),
                compilation_options: Default::default(),
                targets: &[target(color_format), target(aux_format)],
            }),
            multiview_mask: None,
            cache: None,
        });

        let parcel_gated_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("stark transform parcel gated"),
                layout: Some(&quad_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_gated"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: strip,
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_parcel_gated"),
                    compilation_options: Default::default(),
                    targets: &[target(color_format), target(aux_format)],
                }),
                multiview_mask: None,
                cache: None,
            });
        // Moved mask coverage lands with **max** blending over the residue:
        // the soft union of what stayed and what arrived (§16.8), and — unlike
        // the paint parcels — safe under any draw order.
        let mask_union = Some(wgpu::ColorTargetState {
            format: MASK_FORMAT,
            blend: Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Max,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Max,
                },
            }),
            write_mask: wgpu::ColorWrites::ALL,
        });
        let mask_gated_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("stark transform mask gated"),
            layout: Some(&mask_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_gated"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: strip,
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_mask_gated"),
                compilation_options: Default::default(),
                targets: std::slice::from_ref(&mask_union),
            }),
            multiview_mask: None,
            cache: None,
        });
        let mask_base_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("stark transform mask base"),
            layout: Some(&mask_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_fill"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_mask_base"),
                compilation_options: Default::default(),
                targets: &[target(MASK_FORMAT)],
            }),
            multiview_mask: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("stark transform sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let zero_color = zero_texture(ctx, color_format);
        let zero_aux = zero_texture(ctx, aux_format);

        Self {
            ctx: ctx.clone(),
            color_format,
            aux_format,
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
            zero_color,
            zero_aux,
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
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stark transform"),
        });

        // Source-tile bind groups are shared across every destination they reach.
        let mut src_bgs: BTreeMap<TileCoord, wgpu::BindGroup> = BTreeMap::new();
        // Scratch parcels must outlive their reads: the pool would otherwise hand
        // the same texture to a later tile inside this same encoder.
        let mut scratch: Vec<TexHandle> = Vec::new();

        let mut tiles = base.clone();
        for (dest, sources) in &plan.rewrites {
            let parcel = self.render_parcel(
                &mut encoder,
                pool,
                base,
                selection,
                affine,
                *dest,
                sources,
                &mut src_bgs,
            );
            let dst = (
                pool.acquire_tex(self.color_format, AllocSource::TransformDestination),
                pool.acquire_tex(self.aux_format, AllocSource::TransformDestination),
            );
            self.combine(
                &mut encoder,
                base,
                selection,
                *dest,
                parcel.as_ref(),
                &dst,
                None,
            );
            if let Some((c, a)) = parcel {
                scratch.push(c);
                scratch.push(a);
            }
            tiles = tiles.insert(*dest, TilePairHandle::new(dst.0, dst.1));
        }
        for coord in &plan.drops {
            tiles = tiles.remove(coord);
        }

        // The mask, carried under the same affine (pure Replace — §16.1).
        let mut mask_tiles: HashTrieMap<TileCoord, crate::gpu::tile::MaskHandle> =
            HashTrieMap::new();
        for (dest, sources) in &mask_plan.rewrites {
            let dst = pool.acquire_mask(AllocSource::TransformMask);
            self.render_mask(&mut encoder, selection, affine, *dest, sources, &dst);
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
        let moved_selection = Selection::from_parts(mask_tiles, selection.outside() > 0.5, hull);

        self.ctx.queue.submit([encoder.finish()]);
        drop(scratch); // now safe to recycle
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
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stark transform gated"),
        });

        let mut src_bgs: BTreeMap<TileCoord, wgpu::BindGroup> = BTreeMap::new();
        let mut scratch: Vec<TexHandle> = Vec::new();

        let mut tiles = base.clone();
        for (dest, unit_idxs) in &plan.rewrites {
            let parcel = self.render_gated_parcel(
                &mut encoder,
                pool,
                base,
                selection,
                &plan.units,
                unit_idxs,
                inv.as_ref(),
                rect,
                *dest,
                &mut src_bgs,
            );
            let dst = (
                pool.acquire_tex(self.color_format, AllocSource::TransformDestination),
                pool.acquire_tex(self.aux_format, AllocSource::TransformDestination),
            );
            self.combine(
                &mut encoder,
                base,
                selection,
                *dest,
                parcel.as_ref(),
                &dst,
                Some(rect),
            );
            if let Some((c, a)) = parcel {
                scratch.push(c);
                scratch.push(a);
            }
            tiles = tiles.insert(*dest, TilePairHandle::new(dst.0, dst.1));
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
                self.render_gated_mask(
                    &mut encoder,
                    selection,
                    &mask_plan.units,
                    unit_idxs,
                    inv.as_ref(),
                    rect,
                    *dest,
                    &dst,
                );
                mask_tiles = mask_tiles.insert(*dest, dst);
            }
            // The hull rides along: what stayed plus wherever the map can have
            // carried coverage, conservatively.
            let hull = selection
                .hull()
                .map(|(lo, hi)| (lo.min(geo.image_aabb.0), hi.max(geo.image_aabb.1)));
            Selection::from_parts(mask_tiles, selection.outside() > 0.5, hull)
        };

        self.ctx.queue.submit([encoder.finish()]);
        drop(scratch); // now safe to recycle
        Some((tiles, moved_selection))
    }

    /// Rasterize the pieces reaching `dest` into a fresh scratch pair —
    /// [`render_parcel`](Self::render_parcel)'s shape, with quads generalized
    /// to [`SourceUnit`]s and the deposit gated by the source rect.
    #[allow(clippy::too_many_arguments)]
    fn render_gated_parcel(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &TilePool,
        base: &TileMap,
        selection: &Selection,
        units: &[SourceUnit],
        unit_idxs: &[usize],
        inv: Option<&Homography>,
        rect: (Vec2, Vec2),
        dest: TileCoord,
        src_bgs: &mut BTreeMap<TileCoord, wgpu::BindGroup>,
    ) -> Option<(TexHandle, TexHandle)> {
        if unit_idxs.is_empty() {
            return None;
        }
        let device = &self.ctx.device;
        let color = pool.acquire_tex(self.color_format, AllocSource::TransformScratch);
        let aux = pool.acquire_tex(self.aux_format, AllocSource::TransformScratch);

        let mut draws: Vec<(wgpu::BindGroup, wgpu::BindGroup)> = Vec::new();
        for idx in unit_idxs {
            let unit = &units[*idx];
            let Some(tile) = base.get(&unit.src) else {
                continue;
            };
            let src_bg = src_bgs
                .entry(unit.src)
                .or_insert_with(|| {
                    let mask = self.selection.mask_for(selection, unit.src);
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("stark transform src bg"),
                        layout: &self.src_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(tile.color_view()),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(tile.aux_view()),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(&mask),
                            },
                        ],
                    })
                })
                .clone();
            draws.push((self.gated_bg(unit, inv, rect, dest), src_bg));
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("stark transform parcel gated"),
            color_attachments: &[
                Some(clear_attachment(color.view())),
                Some(clear_attachment(aux.view())),
            ],
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
        Some((color, aux))
    }

    /// One destination mask tile under a rect-scoped map: the residue
    /// `old · (1 − box)` laid down fullscreen, then the moved coverage pieces
    /// drawn over with max blending — the soft union (§16.8).
    #[allow(clippy::too_many_arguments)]
    fn render_gated_mask(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        selection: &Selection,
        units: &[SourceUnit],
        unit_idxs: &[usize],
        inv: Option<&Homography>,
        rect: (Vec2, Vec2),
        dest: TileCoord,
        dst: &crate::gpu::tile::MaskHandle,
    ) {
        let device = &self.ctx.device;
        let mask_bg = |view: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("stark transform mask src bg"),
                layout: &self.mask_src_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(view),
                }],
            })
        };
        // The residue reads the destination's *old* coverage — a real tile or
        // the outside constant, through the same clamped-read pattern.
        let old = self.selection.mask_for(selection, dest);
        let base_draw = (self.gated_base_bg(rect, dest), mask_bg(&old));
        let mut draws: Vec<(wgpu::BindGroup, wgpu::BindGroup)> = Vec::new();
        for idx in unit_idxs {
            let unit = &units[*idx];
            let src = self.selection.mask_for(selection, unit.src);
            draws.push((self.gated_bg(unit, inv, rect, dest), mask_bg(&src)));
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("stark transform mask gated"),
            color_attachments: &[Some(clear_attachment(dst.view()))],
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
        self.gated_uniform_bg(GatedUniform::new(unit, inv, rect, dest))
    }

    /// The group-0 bind for the mask residue pass.
    fn gated_base_bg(&self, rect: (Vec2, Vec2), dest: TileCoord) -> wgpu::BindGroup {
        self.gated_uniform_bg(GatedUniform::base(rect, dest))
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
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }

    /// Rasterize the transformed source quads reaching `dest` into a fresh
    /// scratch pair: `(premult color as-is, height·mask)` — the moved parcel.
    /// `None` when nothing reaches this tile (a cut with no incoming paint).
    #[allow(clippy::too_many_arguments)]
    fn render_parcel(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &TilePool,
        base: &TileMap,
        selection: &Selection,
        affine: Affine2,
        dest: TileCoord,
        sources: &[TileCoord],
        src_bgs: &mut BTreeMap<TileCoord, wgpu::BindGroup>,
    ) -> Option<(TexHandle, TexHandle)> {
        if sources.is_empty() {
            return None;
        }
        let device = &self.ctx.device;
        let color = pool.acquire_tex(self.color_format, AllocSource::TransformScratch);
        let aux = pool.acquire_tex(self.aux_format, AllocSource::TransformScratch);

        let mut draws: Vec<(wgpu::BindGroup, wgpu::BindGroup)> = Vec::new();
        for src in sources {
            let Some(tile) = base.get(src) else { continue };
            let src_bg = src_bgs
                .entry(*src)
                .or_insert_with(|| {
                    let mask = self.selection.mask_for(selection, *src);
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("stark transform src bg"),
                        layout: &self.src_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(tile.color_view()),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(tile.aux_view()),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(&mask),
                            },
                        ],
                    })
                })
                .clone();
            draws.push((self.quad_bg(affine, *src, dest), src_bg));
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("stark transform parcel"),
            color_attachments: &[
                Some(clear_attachment(color.view())),
                Some(clear_attachment(aux.view())),
            ],
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
        Some((color, aux))
    }

    /// Cut `dest`'s base by its own mask and stack the parcel over it, into the
    /// fresh CoW `(color, aux)` pair `dst`. `gate` scopes the cut to a source
    /// rect (§16.8); `None` is the affine's whole-plane cut, arithmetically
    /// untouched.
    #[allow(clippy::too_many_arguments)]
    fn combine(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        base: &TileMap,
        selection: &Selection,
        dest: TileCoord,
        parcel: Option<&(TexHandle, TexHandle)>,
        dst: &(TexHandle, TexHandle),
        gate: Option<(Vec2, Vec2)>,
    ) {
        let device = &self.ctx.device;
        let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stark transform combine uniform"),
            contents: bytemuck::bytes_of(&CombineUniform::new(dest, gate)),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let (base_color, base_aux) = match base.get(&dest) {
            Some(tile) => (tile.color_view().clone(), tile.aux_view().clone()),
            None => (self.zero_color.clone(), self.zero_aux.clone()),
        };
        let base_mask = self.selection.mask_for(selection, dest);
        let (parcel_color, parcel_aux) = match parcel {
            Some((c, a)) => (c.view().clone(), a.view().clone()),
            None => (self.zero_color.clone(), self.zero_aux.clone()),
        };
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark transform combine bg"),
            layout: &self.combine_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&base_color),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&base_aux),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&base_mask),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&parcel_color),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&parcel_aux),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: ubuf.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("stark transform combine"),
            color_attachments: &[
                Some(clear_attachment(dst.0.view())),
                Some(clear_attachment(dst.1.view())),
            ],
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
                entries: &[wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(handle.view()),
                }],
            });
            draws.push((self.quad_bg(affine, *src, dest), src_bg));
        }

        let outside = f64::from(selection.outside());
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("stark transform mask"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dst.view(),
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: outside,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
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
            contents: bytemuck::bytes_of(&QuadUniform::new(affine, src, dest)),
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
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }
}

fn clear_attachment(view: &wgpu::TextureView) -> wgpu::RenderPassColorAttachment<'_> {
    wgpu::RenderPassColorAttachment {
        view,
        resolve_target: None,
        depth_slice: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            store: wgpu::StoreOp::Store,
        },
    }
}

/// A 1×1 texture of `format` holding zeros — "no paint here", readable through
/// the clamped loads and samples every pass here uses.
fn zero_texture(ctx: &GpuContext, format: wgpu::TextureFormat) -> wgpu::TextureView {
    let extent = wgpu::Extent3d {
        width: 1,
        height: 1,
        depth_or_array_layers: 1,
    };
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("stark transform zero"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let bytes = format
        .block_copy_size(None)
        .expect("uncompressed tile format") as usize;
    ctx.queue.write_texture(
        texture.as_image_copy(),
        &vec![0u8; bytes],
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(bytes as u32),
            rows_per_image: Some(1),
        },
        extent,
    );
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}
