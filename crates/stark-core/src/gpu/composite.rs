//! Compositing and the media/lighting pass (DESIGN.md §6.3, §6.4).
//!
//! Two passes:
//!   A. Composite every visible tile's channels into viewport-sized offscreen
//!      targets — Oklab color (premultiplied "over") and `(height, wet)` aux
//!      (additive).
//!   B. A fullscreen media pass that derives normals from the height field,
//!      lights the impasto, adds wet gloss, converts Oklab → display, and
//!      composites over the background into the final target.
//!
//! This replaces the step-1 `Presenter` for engine rendering; the height/normal
//! lighting is the "old masters" payoff.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::geom::{Extent2, TileCoord, ViewTransform, TILE_SIZE};
use crate::gpu::context::GpuContext;
use crate::gpu::tile::{TileHandle, AUX_FORMAT, COLOR_FORMAT};

/// Mirrors `View` in `composite.wesl` (32 bytes).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ViewUniform {
    st: [f32; 4],   // scale.xy, translate.xy
    misc: [f32; 4], // tile_size, unused
}

/// Per-tile instance: canvas-space origin.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Instance {
    origin: [f32; 2],
}

/// Mirrors `Media` in `media.wesl` (48 bytes).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MediaUniform {
    light: [f32; 4], // dir.xyz, height_strength
    bg: [f32; 4],    // background linear RGB, unused
    shade: [f32; 4], // ambient, diffuse_k, spec_strength, shininess
}

/// Lighting parameters for the media pass (DESIGN.md §6.3). A single place to
/// tune the painterly look.
#[derive(Copy, Clone, Debug)]
pub struct MediaParams {
    pub light_dir: [f32; 3],
    pub height_strength: f32,
    pub ambient: f32,
    pub diffuse: f32,
    pub specular: f32,
    pub shininess: f32,
}

impl Default for MediaParams {
    fn default() -> Self {
        Self {
            light_dir: [-0.5, -0.6, 0.85],
            height_strength: 0.4,
            ambient: 0.55,
            diffuse: 0.55,
            specular: 0.2,
            shininess: 32.0,
        }
    }
}

pub struct Compositor {
    ctx: GpuContext,

    // Pass A: composite tiles into offscreen targets.
    composite_pipeline: wgpu::RenderPipeline,
    view_buf: wgpu::Buffer,
    view_bg: wgpu::BindGroup,
    tile_bgl: wgpu::BindGroupLayout,
    instances: wgpu::Buffer,
    instance_cap: usize,

    // Pass B: media/lighting → final target.
    media_pipeline: wgpu::RenderPipeline,
    media_buf: wgpu::Buffer,
    media_bgl: wgpu::BindGroupLayout,
    media: MediaParams,

    // Viewport-sized offscreen targets (recreated on resize).
    size: Extent2,
    comp_color_view: wgpu::TextureView,
    comp_aux_view: wgpu::TextureView,
    media_bg: wgpu::BindGroup,
}

impl Compositor {
    pub fn new(ctx: &GpuContext, target_format: wgpu::TextureFormat, size: Extent2) -> Self {
        let device = &ctx.device;

        // ---- Pass A: composite ----
        let comp_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark composite"),
            source: wgpu::ShaderSource::Wgsl(stark_shaders::composite().into()),
        });

        let view_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stark composite view bgl"),
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
            label: Some("stark composite tile bgl"),
            entries: &[
                tex_entry(0),
                tex_entry(1),
            ],
        });

        let comp_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("stark composite layout"),
            bind_group_layouts: &[Some(&view_bgl), Some(&tile_bgl)],
            immediate_size: 0,
        });

        let add = wgpu::BlendState {
            color: additive(),
            alpha: additive(),
        };
        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("stark composite pipeline"),
            layout: Some(&comp_layout),
            vertex: wgpu::VertexState {
                module: &comp_shader,
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
                module: &comp_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: COLOR_FORMAT,
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: AUX_FORMAT,
                        blend: Some(add),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
            }),
            multiview_mask: None,
            cache: None,
        });

        let view_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stark composite view"),
            size: std::mem::size_of::<ViewUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("stark composite sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let view_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark composite view bg"),
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

        // ---- Pass B: media ----
        let media_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark media"),
            source: wgpu::ShaderSource::Wgsl(stark_shaders::media().into()),
        });
        let media_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stark media bgl"),
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
                load_tex_entry(1),
                load_tex_entry(2),
            ],
        });
        let media_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("stark media layout"),
            bind_group_layouts: &[Some(&media_bgl)],
            immediate_size: 0,
        });
        let media_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("stark media pipeline"),
            layout: Some(&media_layout),
            vertex: wgpu::VertexState {
                module: &media_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &media_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let media_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stark media uniform"),
            size: std::mem::size_of::<MediaUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let instances = alloc_instances(device, 1);
        let (comp_color_view, comp_aux_view, media_bg) =
            make_offscreen(device, size, &media_bgl, &media_buf);

        Self {
            ctx: ctx.clone(),
            composite_pipeline,
            view_buf,
            view_bg,
            tile_bgl,
            instances,
            instance_cap: 1,
            media_pipeline,
            media_buf,
            media_bgl,
            media: MediaParams::default(),
            size,
            comp_color_view,
            comp_aux_view,
            media_bg,
        }
    }

    /// Adjust the media/lighting parameters (DESIGN.md §6.3).
    pub fn set_media(&mut self, media: MediaParams) {
        self.media = media;
    }

    /// Composite `tiles` and render the lit result into `target` under `view`.
    pub fn render(
        &mut self,
        target: &wgpu::TextureView,
        view: ViewTransform,
        background: wgpu::Color,
        tiles: &[(TileCoord, TileHandle)],
    ) {
        let device = &self.ctx.device;
        if view.viewport != self.size {
            self.size = view.viewport;
            let (c, a, bg) = make_offscreen(device, self.size, &self.media_bgl, &self.media_buf);
            self.comp_color_view = c;
            self.comp_aux_view = a;
            self.media_bg = bg;
        }

        // View uniform (canvas px -> NDC).
        let (scale, translate) = view.canvas_to_ndc();
        self.ctx.queue.write_buffer(
            &self.view_buf,
            0,
            bytemuck::bytes_of(&ViewUniform {
                st: [scale.x, scale.y, translate.x, translate.y],
                misc: [TILE_SIZE as f32, 0.0, 0.0, 0.0],
            }),
        );

        // Media uniform.
        self.ctx.queue.write_buffer(
            &self.media_buf,
            0,
            bytemuck::bytes_of(&MediaUniform {
                light: [
                    self.media.light_dir[0],
                    self.media.light_dir[1],
                    self.media.light_dir[2],
                    self.media.height_strength,
                ],
                bg: [
                    background.r as f32,
                    background.g as f32,
                    background.b as f32,
                    0.0,
                ],
                shade: [
                    self.media.ambient,
                    self.media.diffuse,
                    self.media.specular,
                    self.media.shininess,
                ],
            }),
        );

        // Instances (tile origins).
        let instances: Vec<Instance> = tiles
            .iter()
            .map(|(c, _)| Instance {
                origin: c.origin().to_array(),
            })
            .collect();
        if !instances.is_empty() {
            if instances.len() > self.instance_cap {
                self.instances = alloc_instances(device, instances.len());
                self.instance_cap = instances.len();
            }
            self.ctx
                .queue
                .write_buffer(&self.instances, 0, bytemuck::cast_slice(&instances));
        }

        let tile_bgs: Vec<wgpu::BindGroup> = tiles
            .iter()
            .map(|(_, t)| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("stark composite tile bg"),
                    layout: &self.tile_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(t.color_view()),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(t.aux_view()),
                        },
                    ],
                })
            })
            .collect();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stark composite encoder"),
        });

        // Pass A: composite tiles into offscreen color + aux.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stark composite pass"),
                color_attachments: &[
                    Some(clear_attachment(&self.comp_color_view, wgpu::Color::TRANSPARENT)),
                    Some(clear_attachment(&self.comp_aux_view, wgpu::Color::TRANSPARENT)),
                ],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, &self.view_bg, &[]);
            pass.set_vertex_buffer(0, self.instances.slice(..));
            for (i, bg) in tile_bgs.iter().enumerate() {
                let idx = i as u32;
                pass.set_bind_group(1, bg, &[]);
                pass.draw(0..4, idx..idx + 1);
            }
        }

        // Pass B: media/lighting → target.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stark media pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.media_pipeline);
            pass.set_bind_group(0, &self.media_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        self.ctx.queue.submit([encoder.finish()]);
    }
}

fn tex_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn load_tex_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            // Sampled only via textureLoad, so no filtering required.
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn additive() -> wgpu::BlendComponent {
    wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    }
}

fn clear_attachment(
    view: &wgpu::TextureView,
    color: wgpu::Color,
) -> wgpu::RenderPassColorAttachment<'_> {
    wgpu::RenderPassColorAttachment {
        view,
        resolve_target: None,
        depth_slice: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(color),
            store: wgpu::StoreOp::Store,
        },
    }
}

fn alloc_instances(device: &wgpu::Device, count: usize) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("stark composite instances"),
        contents: bytemuck::cast_slice(&vec![Instance { origin: [0.0; 2] }; count.max(1)]),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    })
}

/// (Re)create the offscreen composite targets and the media bind group.
fn make_offscreen(
    device: &wgpu::Device,
    size: Extent2,
    media_bgl: &wgpu::BindGroupLayout,
    media_buf: &wgpu::Buffer,
) -> (wgpu::TextureView, wgpu::TextureView, wgpu::BindGroup) {
    let extent = wgpu::Extent3d {
        width: size.width.max(1),
        height: size.height.max(1),
        depth_or_array_layers: 1,
    };
    let usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
    let make = |format, label| {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default())
    };
    let comp_color_view = make(COLOR_FORMAT, "stark comp color");
    let comp_aux_view = make(AUX_FORMAT, "stark comp aux");

    let media_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("stark media bg"),
        layout: media_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: media_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&comp_color_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&comp_aux_view),
            },
        ],
    });
    (comp_color_view, comp_aux_view, media_bg)
}
