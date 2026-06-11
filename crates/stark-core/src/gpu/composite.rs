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

use crate::colorspace::ColorSpace;
use crate::geom::{
    Extent2, TileCoord, ViewTransform, INTERIOR_UV_BIAS, INTERIOR_UV_SCALE, TILE_SIZE,
};
use crate::gpu::context::GpuContext;
use crate::gpu::environment::Environment;
use crate::gpu::surface::{Surface, SURFACE_TILE_PX};
use crate::gpu::tile::TilePairHandle;

/// Mirrors `View` in `composite.wesl` (32 bytes).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ViewUniform {
    st: [f32; 4],   // scale.xy, translate.xy
    misc: [f32; 4], // tile_size, unused
}

/// Per-tile instance: canvas-space origin + the layer's opacity.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Instance {
    origin: [f32; 2],
    opacity: f32,
}

/// Mirrors `Media` in `media_common.wesl` (80 bytes).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MediaUniform {
    light: [f32; 4], // _, _, _, height_strength (relief slope; xyz unused under IBL)
    bg: [f32; 4],    // background (substrate) in latent channels (xyz), unused w
    shade: [f32; 4], // exposure, diffuse_lod, wet_gloss, max_lod
    // Screen→canvas mapping + surface (bump) sampling for the canvas relief:
    surf_a: [f32; 4], // canvas_origin.xy (canvas px at pixel 0), canvas_per_px, inv_tile
    surf_b: [f32; 4], // surface_strength, normal_dither, _, _
}

/// Lighting parameters for the media pass (DESIGN.md §6.3). The painting is lit by
/// image-based lighting from an [`Environment`]; this is a single place to tune the
/// look. A view setting — never historized (it changes how the canvas looks, not
/// its pixels).
#[derive(Copy, Clone, Debug)]
pub struct MediaParams {
    /// Relief slope: how strongly the height field tilts normals (impasto/weave).
    pub height_strength: f32,
    /// Overall exposure applied to the lit result before the sRGB encode.
    pub exposure: f32,
    /// Wet glossiness in [0,1]: how smooth (low-roughness) fully-wet paint becomes,
    /// driving the Cook–Torrance specular. 0 = stays matte even when wet; 1 = near
    /// mirror-smooth. Dry paint and bare canvas are always rough → matte.
    pub specular: f32,
    /// How strongly the canvas surface relief shows (its weave amplitude).
    pub surface_strength: f32,
    /// Normal-dither amplitude in [0, 1]: canvas-anchored noise added to the relief
    /// heights before the normal gradient, breaking up banding in the lit result.
    /// Like the weave, the noise is seeded by canvas position, so it is *not*
    /// translation invariant — the seam tests set it to 0 (as they do
    /// `surface_strength`).
    pub normal_dither: f32,
}

impl Default for MediaParams {
    fn default() -> Self {
        Self {
            height_strength: 0.15,
            exposure: 0.8,
            specular: 0.20,
            surface_strength: 0.6,
            normal_dither: 1.0,
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

    // Offscreen channel formats (from the color space), for resize.
    color_format: wgpu::TextureFormat,
    aux_format: wgpu::TextureFormat,

    // The canvas surface (bump) sampled by the media pass for relief.
    surface: Surface,
    // The HDR lighting environment sampled by the media pass (DESIGN.md §6.3).
    environment: Environment,

    // Viewport-sized offscreen targets (recreated on resize).
    size: Extent2,
    comp_color_view: wgpu::TextureView,
    comp_aux_view: wgpu::TextureView,
    media_bg: wgpu::BindGroup,
}

impl Compositor {
    pub fn new(
        ctx: &GpuContext,
        target_format: wgpu::TextureFormat,
        size: Extent2,
        color_space: &dyn ColorSpace,
        surface: Surface,
        environment: Environment,
    ) -> Self {
        let device = &ctx.device;
        let color_format = color_space.color_format();
        let aux_format = color_space.aux_format();

        // ---- Pass A: composite (generic passthrough; blends from color space) ----
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
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32],
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
                        format: color_format,
                        blend: Some(color_space.color_blend()),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: aux_format,
                        blend: Some(color_space.aux_blend()),
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
            source: wgpu::ShaderSource::Wgsl(color_space.media_shader().into()),
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
                load_tex_entry(1), // comp_color (textureLoad)
                load_tex_entry(2), // comp_aux   (textureLoad)
                tex_entry(3),      // surface bump (filtered)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                tex_entry(5), // environment (filtered, mipped)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
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
        let (comp_color_view, comp_aux_view, media_bg) = make_offscreen(
            device, size, color_format, aux_format, &media_bgl, &media_buf, &surface, &environment,
        );

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
            color_format,
            aux_format,
            surface,
            environment,
            size,
            comp_color_view,
            comp_aux_view,
            media_bg,
        }
    }

    /// The current media/lighting parameters (DESIGN.md §6.3).
    pub fn media(&self) -> MediaParams {
        self.media
    }

    /// Adjust the media/lighting parameters (DESIGN.md §6.3).
    pub fn set_media(&mut self, media: MediaParams) {
        self.media = media;
    }

    /// Swap the canvas surface (bump), rebuilding the media bind group so the next
    /// render shades against it (DESIGN.md §6.4). A view-time swap — the composited
    /// tiles are untouched.
    pub fn set_surface(&mut self, surface: Surface) {
        self.surface = surface;
        let (c, a, bg) = make_offscreen(
            &self.ctx.device,
            self.size,
            self.color_format,
            self.aux_format,
            &self.media_bgl,
            &self.media_buf,
            &self.surface,
            &self.environment,
        );
        self.comp_color_view = c;
        self.comp_aux_view = a;
        self.media_bg = bg;
    }

    /// Swap the HDR lighting environment, rebuilding the media bind group so the
    /// next render samples it (DESIGN.md §6.3).
    pub fn set_environment(&mut self, environment: Environment) {
        self.environment = environment;
        let (c, a, bg) = make_offscreen(
            &self.ctx.device,
            self.size,
            self.color_format,
            self.aux_format,
            &self.media_bgl,
            &self.media_buf,
            &self.surface,
            &self.environment,
        );
        self.comp_color_view = c;
        self.comp_aux_view = a;
        self.media_bg = bg;
    }

    /// Composite `tiles` and render the lit result into `target` under `view`.
    pub fn render(
        &mut self,
        target: &wgpu::TextureView,
        view: ViewTransform,
        bg_channels: [f32; 4],
        tiles: &[(TileCoord, TilePairHandle, f32)],
    ) {
        let device = &self.ctx.device;
        if view.viewport != self.size {
            self.size = view.viewport;
            let (c, a, bg) = make_offscreen(
                device,
                self.size,
                self.color_format,
                self.aux_format,
                &self.media_bgl,
                &self.media_buf,
                &self.surface,
                &self.environment,
            );
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
                misc: [TILE_SIZE as f32, INTERIOR_UV_SCALE, INTERIOR_UV_BIAS, 0.0],
            }),
        );

        // Screen→canvas mapping for sampling the surface bump in canvas space, so
        // the weave stays attached to the canvas as it pans/zooms (DESIGN.md §6.4).
        let inv_zoom = 1.0 / view.zoom;
        let canvas_origin = view.center
            - crate::geom::Vec2::new(view.viewport.width as f32, view.viewport.height as f32)
                * (0.5 * inv_zoom);

        // Diffuse samples a heavily-blurred high mip ≈ hemispherical irradiance.
        // The Cook–Torrance specular picks its own mip from roughness, spanning the
        // whole chain (roughness 0 → mip 0 sharp; roughness 1 → `max_lod` blurred).
        let diffuse_lod = (self.environment.mip_count as f32 - 3.0).max(0.0);
        let max_lod = (self.environment.mip_count as f32 - 1.0).max(0.0);
        // Normalize by the environment's mean luminance so exposure means the same
        // thing for any environment (a flat surface reads ~its albedo).
        let exposure = self.media.exposure / self.environment.mean_luminance;

        // Media uniform.
        self.ctx.queue.write_buffer(
            &self.media_buf,
            0,
            bytemuck::bytes_of(&MediaUniform {
                light: [0.0, 0.0, 0.0, self.media.height_strength],
                bg: bg_channels,
                shade: [exposure, diffuse_lod, self.media.specular, max_lod],
                surf_a: [
                    canvas_origin.x,
                    canvas_origin.y,
                    inv_zoom,
                    1.0 / SURFACE_TILE_PX,
                ],
                surf_b: [self.media.surface_strength, self.media.normal_dither, 0.0, 0.0],
            }),
        );

        // Instances (tile origins).
        let instances: Vec<Instance> = tiles
            .iter()
            .map(|(c, _, opacity)| Instance {
                origin: c.origin().to_array(),
                opacity: *opacity,
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
            .map(|(_, t, _)| {
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
        contents: bytemuck::cast_slice(&vec![
            Instance { origin: [0.0; 2], opacity: 1.0 };
            count.max(1)
        ]),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    })
}

/// (Re)create the offscreen composite targets and the media bind group.
#[allow(clippy::too_many_arguments)]
fn make_offscreen(
    device: &wgpu::Device,
    size: Extent2,
    color_format: wgpu::TextureFormat,
    aux_format: wgpu::TextureFormat,
    media_bgl: &wgpu::BindGroupLayout,
    media_buf: &wgpu::Buffer,
    surface: &Surface,
    environment: &Environment,
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
    let comp_color_view = make(color_format, "stark comp color");
    let comp_aux_view = make(aux_format, "stark comp aux");

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
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&surface.view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(&surface.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(&environment.view),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::Sampler(&environment.sampler),
            },
        ],
    });
    (comp_color_view, comp_aux_view, media_bg)
}
