//! Presentation: composite tiles onto a target surface under a pan/zoom
//! transform (DESIGN.md §6.4).
//!
//! Step 1 draws each provided tile as one screen-aligned quad. Visible-tile
//! culling, LOD/mip sampling, and the Oklab → display conversion are layered in
//! later (DESIGN.md §6.4, §6.5); the surface contract established here does not
//! change when they are.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::geom::{TileCoord, ViewTransform, INTERIOR_UV_BIAS, INTERIOR_UV_SCALE, TILE_SIZE};
use crate::gpu::context::GpuContext;
use crate::gpu::tile::TileHandle;

/// Mirrors the `View` uniform in `present.wesl` (std140, 32 bytes). Packed into
/// vec4s so the layout is unambiguous on both the Rust and WGSL sides.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ViewUniform {
    /// scale.xy, translate.xy
    st: [f32; 4],
    /// tile_size, (unused)
    misc: [f32; 4],
}

/// Per-tile instance attribute: canvas-space top-left origin in pixels.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Instance {
    origin: [f32; 2],
}

/// Renders canvas tiles to a target texture view.
pub struct Presenter {
    ctx: GpuContext,
    pipeline: wgpu::RenderPipeline,
    view_buf: wgpu::Buffer,
    view_bg: wgpu::BindGroup,
    tile_bgl: wgpu::BindGroupLayout,
    instances: wgpu::Buffer,
    instance_cap: usize,
}

impl Presenter {
    /// Build the present pipeline for a given target format (DESIGN.md §6.4).
    pub fn new(ctx: &GpuContext, target_format: wgpu::TextureFormat) -> Self {
        let device = &ctx.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark present"),
            source: wgpu::ShaderSource::Wgsl(stark_shaders::present().into()),
        });

        let view_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stark present view bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
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

        let tile_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stark present tile bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("stark present layout"),
            bind_group_layouts: &[Some(&view_bgl), Some(&tile_bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("stark present pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Instance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                }],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    // Tiles store premultiplied color (DESIGN.md §6.1), so the
                    // present pass composites with premultiplied "over".
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let view_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stark present view"),
            size: std::mem::size_of::<ViewUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("stark present sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let view_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark present view bg"),
            layout: &view_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: view_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let instances = Self::alloc_instances(device, 1);

        Self {
            ctx: ctx.clone(),
            pipeline,
            view_buf,
            view_bg,
            tile_bgl,
            instances,
            instance_cap: 1,
        }
    }

    /// Draw `tiles` into `target` under `view`, clearing to `background` first.
    pub fn render(
        &mut self,
        target: &wgpu::TextureView,
        view: ViewTransform,
        background: wgpu::Color,
        tiles: &[(TileCoord, TileHandle)],
    ) {
        let device = &self.ctx.device;
        let (scale, translate) = view.canvas_to_ndc();
        self.ctx.queue.write_buffer(
            &self.view_buf,
            0,
            bytemuck::bytes_of(&ViewUniform {
                st: [scale.x, scale.y, translate.x, translate.y],
                misc: [TILE_SIZE as f32, INTERIOR_UV_SCALE, INTERIOR_UV_BIAS, 0.0],
            }),
        );

        let instances: Vec<Instance> = tiles
            .iter()
            .map(|(coord, _)| Instance {
                origin: coord.origin().to_array(),
            })
            .collect();
        if !instances.is_empty() {
            if instances.len() > self.instance_cap {
                self.instances = Self::alloc_instances(device, instances.len());
                self.instance_cap = instances.len();
            }
            self.ctx
                .queue
                .write_buffer(&self.instances, 0, bytemuck::cast_slice(&instances));
        }

        // One bind group per tile (its own color texture). Cheap to build per
        // frame for the skeleton; cached alongside tiles later if needed.
        let tile_bgs: Vec<wgpu::BindGroup> = tiles
            .iter()
            .map(|(_, tile)| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("stark present tile bg"),
                    layout: &self.tile_bgl,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(tile.color_view()),
                    }],
                })
            })
            .collect();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stark present encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stark present pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(background),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.view_bg, &[]);
            pass.set_vertex_buffer(0, self.instances.slice(..));
            for (i, tile_bg) in tile_bgs.iter().enumerate() {
                let idx = i as u32;
                pass.set_bind_group(1, tile_bg, &[]);
                pass.draw(0..4, idx..idx + 1);
            }
        }
        self.ctx.queue.submit([encoder.finish()]);
    }

    fn alloc_instances(device: &wgpu::Device, count: usize) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stark present instances"),
            contents: bytemuck::cast_slice(&vec![Instance { origin: [0.0; 2] }; count.max(1)]),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        })
    }
}
