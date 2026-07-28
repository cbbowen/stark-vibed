//! GPU execution of the affine transform (TRANSFORM_DESIGN.md).
//!
//! [`TransformRenderer::apply`] takes a layer's tile map, the author's selection,
//! and six floats, and produces the transformed tile map plus the carried
//! selection — copy-on-write like every stroke, so old history versions keep
//! their tiles and the pool reclaims what falls out of reach.
//!
//! The plan (which tiles, which quads) is CPU-side and pure
//! ([`crate::document::transform`]); this module owns only the three passes of
//! `transform.wesl`:
//!
//! - **parcel** — the selected source interiors forward-rasterized as transformed
//!   quads into a scratch pair, one destination tile at a time. Source interiors
//!   tile the plane, so their images are disjoint and the pass needs no blending
//!   and no order.
//! - **combine** — cut the destination's own base by its mask (the lift law) and
//!   stack the parcel by the shared parcel-deposit law (`paint_common.wesl`).
//! - **mask** — the selection mask itself resampled under the same affine.
//!
//! Like [`StrokeRenderer`](super::stroke::StrokeRenderer) this holds only
//! immutable GPU objects, so it is cheap to `Clone` and rides in the
//! `Action::Context` (DESIGN.md §5).

use std::collections::BTreeMap;

use bytemuck::{Pod, Zeroable};
use rpds::HashTrieMap;
use wgpu::util::DeviceExt;

use crate::colorspace::ColorSpace;
use crate::document::selection::Selection;
use crate::document::transform::{plan_mask, plan_paint};
use crate::geom::{Affine2, TILE_APRON, TILE_SIZE, TILE_TEX, TileCoord, Vec2};
use crate::gpu::context::GpuContext;
use crate::gpu::selection::SelectionRenderer;
use crate::gpu::tile::{AllocSource, MASK_FORMAT, TexHandle, TilePairHandle, TilePool};

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

#[derive(Clone)]
pub struct TransformRenderer {
    ctx: GpuContext,
    color_format: wgpu::TextureFormat,
    aux_format: wgpu::TextureFormat,
    parcel_pipeline: wgpu::RenderPipeline,
    mask_pipeline: wgpu::RenderPipeline,
    combine_pipeline: wgpu::RenderPipeline,
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
            entries: &[load_tex(2), load_tex(3), load_tex(4), load_tex(5), load_tex(6)],
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
    /// under `affine`. `None` rejects the whole action — unusable affine, or more
    /// tiles than the caps allow — deterministically, so peers and replays agree
    /// (TRANSFORM_DESIGN.md §1).
    pub fn apply(
        &self,
        pool: &TilePool,
        base: &HashTrieMap<TileCoord, TilePairHandle>,
        selection: &Selection,
        affine: Affine2,
    ) -> Option<(HashTrieMap<TileCoord, TilePairHandle>, Selection)> {
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
            self.combine(&mut encoder, base, selection, *dest, parcel.as_ref(), &dst);
            if let Some((c, a)) = parcel {
                scratch.push(c);
                scratch.push(a);
            }
            tiles = tiles.insert(*dest, TilePairHandle::new(dst.0, dst.1));
        }
        for coord in &plan.drops {
            tiles = tiles.remove(coord);
        }

        // The mask, carried under the same affine (pure Replace — §1).
        let mut mask_tiles: HashTrieMap<TileCoord, crate::gpu::tile::MaskHandle> =
            HashTrieMap::new();
        for (dest, sources) in &mask_plan.rewrites {
            let dst = pool.acquire_mask(AllocSource::TransformMask);
            self.render_mask(&mut encoder, selection, affine, *dest, sources, &dst);
            mask_tiles = mask_tiles.insert(*dest, dst);
        }
        let moved_selection = Selection::from_parts(mask_tiles, selection.outside() > 0.5);

        self.ctx.queue.submit([encoder.finish()]);
        drop(scratch); // now safe to recycle
        Some((tiles, moved_selection))
    }

    /// Rasterize the transformed source quads reaching `dest` into a fresh
    /// scratch pair: `(premult color as-is, height·mask)` — the moved parcel.
    /// `None` when nothing reaches this tile (a cut with no incoming paint).
    #[allow(clippy::too_many_arguments)]
    fn render_parcel(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &TilePool,
        base: &HashTrieMap<TileCoord, TilePairHandle>,
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
    /// fresh CoW `(color, aux)` pair `dst`.
    fn combine(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        base: &HashTrieMap<TileCoord, TilePairHandle>,
        selection: &Selection,
        dest: TileCoord,
        parcel: Option<&(TexHandle, TexHandle)>,
        dst: &(TexHandle, TexHandle),
    ) {
        let device = &self.ctx.device;
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
