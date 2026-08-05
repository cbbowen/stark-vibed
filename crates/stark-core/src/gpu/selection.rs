//! Selection-mask rasterization and gathering (§6.8).
//!
//! Three small pieces, all colour-space independent (a mask is one coverage channel,
//! whatever the paint is made of), so this survives a colour-space rebuild untouched:
//!
//! - [`SelectionRenderer::apply`] — rasterize one [`SelectionOp`] into fresh mask
//!   tiles, combining with the previous mask. Copy-on-write, exactly like paint: the
//!   old tiles stay valid in older history versions and return to the pool when the
//!   last one drops.
//! - [`SelectionRenderer::region_mask`] — gather the mask into the 1:1 canvas region
//!   the brush-dynamics stamp loop works over (§6.2).
//! - [`SelectionRenderer::constant`] — the 1×1 textures standing in for "there is no
//!   tile here". Every consumer clamps its load to the bound texture's own extent, so
//!   the same shader code reads a real tile and a constant.
//!
//! Like [`StrokeRenderer`](super::stroke::StrokeRenderer) this holds only immutable
//! GPU objects, so it is cheap to `Clone` and can ride in the `Action::Context` (§5).

use bytemuck::{Pod, Zeroable};
use rpds::HashTrieMap;
use wgpu::util::DeviceExt;

use crate::document::selection::{
    MASK_TEX, Selection, SelectionOp, SelectionShape, lasso_edges, mask_tex_origin,
};
use crate::geom::{TileCoord, Vec2};
use crate::gpu::context::GpuContext;
use crate::gpu::tile::{AllocSource, MASK_FORMAT, TilePool};
use crate::gpu::wesl::mirrors_wesl;

/// Mirrors `Params` in `selection.wesl`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MaskUniform {
    a: [f32; 4], // tex_origin.xy (canvas px), 2/TILE_TEX, feather
    b: [f32; 4], // shape parameters
    c: [f32; 4], // kind, mode, edge count, _
}
mirrors_wesl!(MaskUniform, 48);

/// Mirrors `Region` in `mask_region.wesl`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct RegionUniform {
    a: [f32; 4], // region size .xy (px), TILE_TEX, _
}
mirrors_wesl!(RegionUniform, 16);

/// Per-tile instance for the region gather: the tile texture's top-left in region px.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MaskInstance {
    origin: [f32; 2],
}

/// The mode code `selection.wesl` reads for an inversion. Not a [`SelectionMode`] —
/// inverting is not a way of combining a shape, it is its own edit.
const MODE_INVERT: f32 = 4.0;

#[derive(Clone)]
pub struct SelectionRenderer {
    ctx: GpuContext,
    rasterize_pipeline: wgpu::RenderPipeline,
    rasterize_bgl: wgpu::BindGroupLayout,
    region_pipeline: wgpu::RenderPipeline,
    region_view_bgl: wgpu::BindGroupLayout,
    region_tile_bgl: wgpu::BindGroupLayout,
    /// 1×1 masks holding exactly 0 and 1, indexed by the `outside` flag.
    constants: [wgpu::TextureView; 2],
    /// 1×1 stand-in for the lasso edge list, bound by the analytic shapes.
    dummy_edges: wgpu::TextureView,
}

/// The shape half of a selection rasterize: what to draw, as opposed to where.
///
/// `b` and `c` are the shape's two uniform vectors — their meaning depends on the
/// [`SelectionShape`] being rasterized (rect corners, ellipse centre + radii, lasso
/// bounds) — and `edges` is the lasso's edge buffer, unused by the analytic shapes.
struct RasterShape<'a> {
    /// Coverage outside the rasterized tiles (§6.8).
    outside: bool,
    /// The result's analytic hull, as the plan computed it ([`Selection::hull`]).
    hull: Option<(Vec2, Vec2)>,
    b: [f32; 4],
    c: [f32; 4],
    feather: f32,
    edges: &'a wgpu::TextureView,
}

impl SelectionRenderer {
    pub fn new(ctx: &GpuContext) -> Self {
        let device = &ctx.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark selection"),
            source: wgpu::ShaderSource::Wgsl(stark_shaders::selection().into()),
        });
        let load_tex = |binding: u32, visibility| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Texture {
                // Read with `textureLoad` only, 1:1 with the destination.
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let rasterize_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stark selection bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                load_tex(1, wgpu::ShaderStages::FRAGMENT), // previous mask
                load_tex(2, wgpu::ShaderStages::FRAGMENT), // lasso edges
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("stark selection layout"),
            bind_group_layouts: &[Some(&rasterize_bgl)],
            immediate_size: 0,
        });
        let rasterize_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("stark selection pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: MASK_FORMAT,
                    blend: None, // the shader does the combine; write straight through
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        // ---- Region gather (for the brush-dynamics stamp loop, §6.2).
        let region_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark selection region"),
            source: wgpu::ShaderSource::Wgsl(stark_shaders::mask_region().into()),
        });
        let region_view_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stark selection region view bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let region_tile_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stark selection region tile bgl"),
            entries: &[load_tex(0, wgpu::ShaderStages::FRAGMENT)],
        });
        let region_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("stark selection region layout"),
            bind_group_layouts: &[Some(&region_view_bgl), Some(&region_tile_bgl)],
            immediate_size: 0,
        });
        let region_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("stark selection region pipeline"),
            layout: Some(&region_layout),
            vertex: wgpu::VertexState {
                module: &region_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<MaskInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                })],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &region_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: MASK_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let constants = [constant_mask(ctx, 0), constant_mask(ctx, 255)];
        let dummy_edges = dummy_edge_texture(ctx);

        Self {
            ctx: ctx.clone(),
            rasterize_pipeline,
            rasterize_bgl,
            region_pipeline,
            region_view_bgl,
            region_tile_bgl,
            constants,
            dummy_edges,
        }
    }

    /// The 1×1 mask holding `coverage` (0 or 1) — what consumers bind wherever the
    /// selection has no tile. Their clamped `textureLoad` then reads the constant for
    /// every texel, so nothing branches on whether a mask exists.
    pub fn constant(&self, coverage: f32) -> &wgpu::TextureView {
        &self.constants[usize::from(coverage >= 0.5)]
    }

    /// The mask bound for `coord`: the selection's own tile, or the constant that
    /// reigns outside its tile set.
    pub fn mask_for(&self, selection: &Selection, coord: TileCoord) -> wgpu::TextureView {
        match selection.tile(coord) {
            Some(handle) => handle.view().clone(),
            None => self.constant(selection.outside()).clone(),
        }
    }

    /// Apply `op` to `prev`, returning the new selection. `None` when the op's shape
    /// would need more than [`MAX_SELECTION_TILES`](crate::document::selection::MAX_SELECTION_TILES)
    /// mask tiles — the caller leaves the selection alone rather than clipping it.
    pub fn apply(&self, pool: &TilePool, prev: &Selection, op: &SelectionOp) -> Option<Selection> {
        let plan = prev.plan(op)?;
        let edges = match &op.shape {
            SelectionShape::Lasso(points) => lasso_edges(points),
            _ => Vec::new(),
        };
        // A lasso that encloses nothing has no boundary; treat it as a no-op rather
        // than rasterizing an empty edge list (which would read as "all outside").
        if matches!(op.shape, SelectionShape::Lasso(_)) && edges.is_empty() {
            return Some(prev.clone());
        }
        let (_edge_tex, edge_view) = match edges.is_empty() {
            true => (None, self.dummy_edges.clone()),
            false => {
                let (t, v) = self.edge_texture(&edges);
                (Some(t), v)
            }
        };
        let (b, c) = op.shader_params(edges.len());

        // Union and Subtract build on the previous mask (it survives where the shape
        // does not reach); Replace and Intersect start from nothing.
        let base = if plan.keep_prev {
            prev.tile_map().clone()
        } else {
            HashTrieMap::new()
        };
        Some(self.rasterize(
            pool,
            prev,
            base,
            &plan.rasterize,
            RasterShape {
                outside: plan.outside,
                hull: plan.hull,
                b,
                c,
                feather: op.feather,
                edges: &edge_view,
            },
        ))
    }

    /// Invert the selection: every mask tile flips, and so does the coverage outside
    /// them. Constant cost on an unbounded canvas — the whole point of carrying
    /// `outside` as a flag (§6.8).
    pub fn invert(&self, pool: &TilePool, prev: &Selection) -> Selection {
        let plan = prev.plan_invert();
        let edges = self.dummy_edges.clone();
        self.rasterize(
            pool,
            prev,
            HashTrieMap::new(),
            &plan.rasterize,
            RasterShape {
                outside: plan.outside,
                hull: plan.hull,
                b: [0.0; 4],
                c: [0.0, MODE_INVERT, 0.0, 0.0],
                feather: 0.0,
                edges: &edges,
            },
        )
    }

    /// Gather `selection` into a region-sized mask for the stamp loop, matching the
    /// region `stroke.rs` composited the paint into. Tiles the selection has no mask
    /// for are left at the clear value, so the pass draws only what actually exists.
    /// Returns the texture too, so the caller can register it for prompt destruction.
    pub fn region_mask(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        selection: &Selection,
        tiles: &[TileCoord],
        region_origin: Vec2,
        w: u32,
        h: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let device = &self.ctx.device;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("stark selection region mask"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: MASK_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stark selection region uniform"),
            contents: bytemuck::bytes_of(&RegionUniform {
                a: [w as f32, h as f32, MASK_TEX as f32, 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let view_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark selection region view bg"),
            layout: &self.region_view_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ubuf.as_entire_binding(),
            }],
        });

        let mut origins: Vec<MaskInstance> = Vec::new();
        let mut tile_bgs: Vec<wgpu::BindGroup> = Vec::new();
        for coord in tiles {
            let Some(handle) = selection.tile(*coord) else {
                continue;
            };
            let off = mask_tex_origin(*coord) - region_origin;
            origins.push(MaskInstance {
                origin: off.to_array(),
            });
            tile_bgs.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("stark selection region tile bg"),
                layout: &self.region_tile_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(handle.view()),
                }],
            }));
        }
        let instances = (!origins.is_empty()).then(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stark selection region instances"),
                contents: bytemuck::cast_slice(&origins),
                usage: wgpu::BufferUsages::VERTEX,
            })
        });

        {
            let outside = selection.outside() as f64;
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stark selection region gather"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        // Everything the selection has no tile for takes the constant
                        // coverage that reigns there.
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
            if let Some(inst) = &instances {
                pass.set_pipeline(&self.region_pipeline);
                pass.set_bind_group(0, &view_bg, &[]);
                pass.set_vertex_buffer(0, inst.slice(..));
                for (i, bg) in tile_bgs.iter().enumerate() {
                    let idx = i as u32;
                    pass.set_bind_group(1, bg, &[]);
                    pass.draw(0..4, idx..idx + 1);
                }
            }
        }
        (texture, view)
    }

    /// Rasterize `coords` into fresh mask tiles on top of `base`, reading `prev` for
    /// the combine. One draw per tile — they are independent, so this is a single
    /// encoder with no barriers between them.
    fn rasterize(
        &self,
        pool: &TilePool,
        prev: &Selection,
        base: HashTrieMap<TileCoord, crate::gpu::tile::MaskHandle>,
        coords: &[TileCoord],
        shape: RasterShape<'_>,
    ) -> Selection {
        let RasterShape {
            outside,
            hull,
            b,
            c,
            feather,
            edges,
        } = shape;
        if coords.is_empty() {
            return Selection::from_parts(base, outside, hull);
        }
        let device = &self.ctx.device;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stark selection edit"),
        });
        let mut tiles = base;

        for coord in coords {
            let dst = pool.acquire_mask(AllocSource::SelectionMask);
            let origin = mask_tex_origin(*coord);
            let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stark selection params"),
                contents: bytemuck::bytes_of(&MaskUniform {
                    a: [origin.x, origin.y, 2.0 / MASK_TEX as f32, feather],
                    b,
                    c,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let prev_view = self.mask_for(prev, *coord);
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("stark selection bg"),
                layout: &self.rasterize_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ubuf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&prev_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(edges),
                    },
                ],
            });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("stark selection rasterize"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: dst.view(),
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&self.rasterize_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.draw(0..3, 0..1);
            }
            tiles = tiles.insert(*coord, dst);
        }
        self.ctx.queue.submit([encoder.finish()]);
        Selection::from_parts(tiles, outside, hull)
    }

    /// Upload the lasso's edge list as an `N×1` texture (see `selection.wesl`).
    fn edge_texture(&self, edges: &[[f32; 4]]) -> (wgpu::Texture, wgpu::TextureView) {
        let extent = wgpu::Extent3d {
            width: edges.len() as u32,
            height: 1,
            depth_or_array_layers: 1,
        };
        let texture = self.ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("stark selection lasso edges"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.ctx.queue.write_texture(
            texture.as_image_copy(),
            bytemuck::cast_slice(edges),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(edges.len() as u32 * 16),
                rows_per_image: Some(1),
            },
            extent,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }
}

/// A 1×1 `R8Unorm` mask holding a single byte of coverage.
fn constant_mask(ctx: &GpuContext, value: u8) -> wgpu::TextureView {
    let extent = wgpu::Extent3d {
        width: 1,
        height: 1,
        depth_or_array_layers: 1,
    };
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("stark selection constant mask"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: MASK_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    ctx.queue.write_texture(
        texture.as_image_copy(),
        &[value],
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(1),
            rows_per_image: Some(1),
        },
        extent,
    );
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// A 1×1 stand-in for the lasso edge list — bound, never read, by every other shape.
fn dummy_edge_texture(ctx: &GpuContext) -> wgpu::TextureView {
    let extent = wgpu::Extent3d {
        width: 1,
        height: 1,
        depth_or_array_layers: 1,
    };
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("stark selection dummy edges"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    ctx.queue.write_texture(
        texture.as_image_copy(),
        bytemuck::cast_slice(&[0.0f32; 4]),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(16),
            rows_per_image: Some(1),
        },
        extent,
    );
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}
