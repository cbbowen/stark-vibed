//! Compositing and the media/lighting pass (DESIGN.md §6.3, §6.4).
//!
//! Two passes:
//!   A. Composite every visible tile's channels into viewport-sized offscreen
//!      targets — Oklab color (premultiplied "over") and the `(height)` aux
//!      (additive).
//!   B. A fullscreen media pass that derives normals from the height field,
//!      lights the impasto, adds the paint film's gloss, converts Oklab →
//!      display, and composites over the background into the final target.
//!
//! This replaces the step-1 `Presenter` for engine rendering; the height/normal
//! lighting is the "old masters" payoff.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::colorspace::ColorSpace;
use crate::document::selection::Selection;
use crate::geom::{
    Extent2, INTERIOR_UV_BIAS, INTERIOR_UV_SCALE, TILE_SIZE, TileCoord, ViewTransform,
};
use crate::gpu::context::GpuContext;
use crate::gpu::environment::Environment;
use crate::gpu::surface::{SURFACE_TILE_PX, Surface};
use crate::gpu::tile::TilePairHandle;

/// Mirrors `View` in `composite.wesl` and `overlay.wesl` (32 bytes).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ViewUniform {
    st: [f32; 4],   // scale.xy, translate.xy
    misc: [f32; 4], // tile_size, interior uv scale, interior uv bias, zoom
}

/// Per-tile instance: canvas-space origin + the layer's opacity.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Instance {
    origin: [f32; 2],
    opacity: f32,
}

/// Per-mask-tile instance of the outline pass: where the tile is, and how to draw
/// its contour. `tint.a == 0` selects the local actor's black/white marching ants;
/// anything else draws a flat line in `tint.rgb` at that alpha — which is how
/// another collaborator's selection is distinguished from your own
/// (PEER_DESIGN.md §3).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct OverlayInstance {
    origin: [f32; 2],
    tint: [f32; 4],
}

/// One selection to outline, and whose it is (PEER_DESIGN.md §3).
#[derive(Copy, Clone)]
pub struct SelectionOutline<'a> {
    pub selection: &'a Selection,
    /// `None` for the local actor — the marching ants. `Some(rgb)` for a peer's,
    /// drawn as a flat line in their colour so the two never read as the same thing.
    pub tint: Option<[f32; 3]>,
}

/// Per-matte instance, mirroring `matte.wesl`'s vertex attributes.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MatteInstance {
    rect: [f32; 4],     // min.xy, max.xy in canvas px
    channels: [f32; 4], // fill, in the working color space
    opacity: f32,
    _pad: [f32; 3],
}

/// A matte layer's draw parameters (FRAME_DESIGN.md §4).
#[derive(Copy, Clone, Debug)]
pub struct MatteDraw {
    /// The region's rect in canvas px: `min.xy, max.xy`. For a frame this is the
    /// *hole* — the fill covers everything outside it.
    pub rect: [f32; 4],
    /// Fill color in the document's working color space.
    pub channels: [f32; 4],
    /// The layer's opacity.
    pub opacity: f32,
}

/// One item of compositing pass A, in bottom-to-top stack order.
///
/// An ordered list rather than a flat tile array because a matte composites at
/// its own place in the stack — a frame over the painting, a ground under it
/// (FRAME_DESIGN.md §4.4). Tiles already cost one draw each (each needs its own
/// bind group), so interleaving mattes adds no per-tile overhead.
#[derive(Clone)]
pub enum CompositeItem {
    Tile {
        coord: TileCoord,
        handle: TilePairHandle,
        opacity: f32,
    },
    Matte(MatteDraw),
}

/// Mirrors `Media` in `media_common.wesl` (80 bytes).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MediaUniform {
    light: [f32; 4], // _, _, _, height_strength (relief slope; xyz unused under IBL)
    bg: [f32; 4],    // background (substrate) in latent channels (xyz), unused w
    shade: [f32; 4], // exposure, diffuse_lod, gloss, _
    // Screen→canvas mapping + surface (bump) sampling for the canvas relief:
    surf_a: [f32; 4], // canvas_origin.xy (canvas px at pixel 0), canvas_per_px, inv_tile
    surf_b: [f32; 4], // surface_strength, _, _, _
}

/// Lighting parameters for the media pass (DESIGN.md §6.3). The painting is lit by
/// image-based lighting from an [`Environment`]; this is a single place to tune the
/// look. A view setting — never historized (it changes how the canvas looks, not
/// its pixels).
#[derive(Copy, Clone, Debug)]
pub struct MediaParams {
    /// Relief slope: how strongly the height field tilts normals (impasto/weave).
    pub height_strength: f32,
    /// Paint glossiness in `[0,1]`: how smooth (low-roughness) the paint film is,
    /// driving the Cook–Torrance specular. 0 = matte; 1 = near mirror-smooth. It is
    /// a uniform property of paint — every texel with paint on it is equally glossy,
    /// ramped only by how much of the fragment *is* paint (its visible alpha), so
    /// the bare canvas behind it stays rough → matte.
    pub specular: f32,
    /// How strongly the canvas surface relief shows (its weave amplitude).
    pub surface_strength: f32,
}

impl Default for MediaParams {
    fn default() -> Self {
        Self {
            height_strength: 0.15,
            specular: 0.20,
            // The weave is off until asked for: the default canvas is linen, and its
            // relief is there to be *painted into* (§6.2) whether or not the light is
            // made to show it. Raising this embosses it into the lit result.
            surface_strength: 0.0,
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

    // Matte layers, drawn inside pass A at their place in the stack
    // (FRAME_DESIGN.md §4). Its own pipeline because its blend state differs from
    // the color space's: `over` on *both* targets, so an opaque matte erases the
    // relief beneath it rather than letting underlying impasto emboss through.
    matte_pipeline: wgpu::RenderPipeline,
    matte_instances: wgpu::Buffer,
    matte_cap: usize,

    // Pass C: the selection outline, drawn over the lit result (DESIGN.md §6.8).
    // One instanced quad per mask tile, in the same canvas→NDC frame as pass A.
    overlay_pipeline: wgpu::RenderPipeline,
    overlay_view_bg: wgpu::BindGroup,
    overlay_tile_bgl: wgpu::BindGroupLayout,
    overlay_instances: wgpu::Buffer,
    overlay_cap: usize,

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
            entries: &[tex_entry(0), tex_entry(1)],
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
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Instance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32],
                })],
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

        // ---- Matte layers, inside pass A (FRAME_DESIGN.md §4) ----
        //
        // Reuses pass A's view bind group (vertex-only uniform: the fragment stage
        // gets canvas position as a varying, and the zoom rides through `misc.w`
        // for the edge antialiasing width).
        let matte_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark matte"),
            source: wgpu::ShaderSource::Wgsl(stark_shaders::matte().into()),
        });
        let matte_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("stark matte layout"),
            bind_group_layouts: &[Some(&view_bgl)],
            immediate_size: 0,
        });
        let matte_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("stark matte pipeline"),
            layout: Some(&matte_layout),
            vertex: wgpu::VertexState {
                module: &matte_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<MatteInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Float32],
                })],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &matte_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                // Premultiplied `over` on BOTH targets. The aux one is the load-
                // bearing difference from pass A's additive aux: additive would
                // keep the height of paint *underneath* the matte, and the media
                // pass would emboss that paint's impasto as ghost ridges through
                // an opaque mat board (FRAME_DESIGN.md §4.2). `OneMinusSrcAlpha`
                // is valid on the alpha-less R16Float aux: the factor reads the
                // *source* alpha from the shader's output vec4.
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: aux_format,
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
            }),
            multiview_mask: None,
            cache: None,
        });
        let matte_instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stark matte instances"),
            size: std::mem::size_of::<MatteInstance>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---- Pass C: selection outline (DESIGN.md §6.8) ----
        //
        // Its own view bind group rather than pass A's: the fragment stage needs the
        // uniform too (it converts a canvas-space distance to screen px with the
        // zoom), and pass A declares it vertex-only.
        let overlay_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark selection overlay"),
            source: wgpu::ShaderSource::Wgsl(stark_shaders::overlay().into()),
        });
        let overlay_view_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stark overlay view bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
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
        let overlay_tile_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stark overlay tile bgl"),
            entries: &[tex_entry(0)],
        });
        let overlay_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("stark overlay layout"),
            bind_group_layouts: &[Some(&overlay_view_bgl), Some(&overlay_tile_bgl)],
            immediate_size: 0,
        });
        let overlay_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("stark overlay pipeline"),
            layout: Some(&overlay_layout),
            vertex: wgpu::VertexState {
                module: &overlay_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<OverlayInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4],
                })],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &overlay_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    // The outline is drawn *over* the finished image, so it is the one
                    // pass that blends in straight (non-premultiplied) alpha.
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let overlay_view_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark overlay view bg"),
            layout: &overlay_view_bgl,
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
        let (comp_color_view, comp_aux_view, media_bg) = make_offscreen(OffscreenDesc {
            device,
            size,
            color_format,
            aux_format,
            media_bgl: &media_bgl,
            media_buf: &media_buf,
            surface: &surface,
            environment: &environment,
        });

        Self {
            ctx: ctx.clone(),
            composite_pipeline,
            view_buf,
            view_bg,
            tile_bgl,
            instances,
            instance_cap: 1,
            matte_pipeline,
            matte_instances,
            matte_cap: 1,
            overlay_pipeline,
            overlay_view_bg,
            overlay_tile_bgl,
            overlay_instances: alloc_overlay(device, 1),
            overlay_cap: 1,
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

    /// Rebuild the offscreen composite targets and the media bind group from the
    /// compositor's current state. Every caller previously spelled out the same eight
    /// fields and the same three assignments.
    fn rebuild_offscreen(&mut self) {
        let (c, a, bg) = make_offscreen(OffscreenDesc {
            device: &self.ctx.device,
            size: self.size,
            color_format: self.color_format,
            aux_format: self.aux_format,
            media_bgl: &self.media_bgl,
            media_buf: &self.media_buf,
            surface: &self.surface,
            environment: &self.environment,
        });
        self.comp_color_view = c;
        self.comp_aux_view = a;
        self.media_bg = bg;
    }

    /// Swap the canvas surface (bump), rebuilding the media bind group so the next
    /// render shades against it (DESIGN.md §6.4). A view-time swap — the composited
    /// tiles are untouched.
    pub fn set_surface(&mut self, surface: Surface) {
        self.surface = surface;
        self.rebuild_offscreen();
    }

    /// Swap the HDR lighting environment, rebuilding the media bind group so the
    /// next render samples it (DESIGN.md §6.3).
    pub fn set_environment(&mut self, environment: Environment) {
        self.environment = environment;
        self.rebuild_offscreen();
    }

    /// The raw channel formats pass A writes: `(color, aux)`. A caller supplying its
    /// own targets to [`Self::composite_channels`] has to match them.
    pub fn channel_formats(&self) -> (wgpu::TextureFormat, wgpu::TextureFormat) {
        (self.color_format, self.aux_format)
    }

    /// Write the view uniform and upload pass A's instance streams for `items`,
    /// returning the per-tile bind groups that pass draws with.
    ///
    /// Split out of [`Self::render`] so [`Self::composite_channels`] runs the *same*
    /// pass A rather than a second copy of it: what the eyedropper reports and what
    /// the screen shows then cannot drift, which is the whole reason for sampling
    /// through the compositor at all.
    fn prepare_composite(
        &mut self,
        view: ViewTransform,
        items: &[CompositeItem],
    ) -> Vec<wgpu::BindGroup> {
        let device = &self.ctx.device;

        // View uniform (canvas px -> NDC).
        let (scale, translate) = view.canvas_to_ndc();
        self.ctx.queue.write_buffer(
            &self.view_buf,
            0,
            bytemuck::bytes_of(&ViewUniform {
                st: [scale.x, scale.y, translate.x, translate.y],
                // `zoom` rides in `.w` for the outline pass, which measures its width
                // in screen px from a canvas-space distance (§6.8).
                misc: [
                    TILE_SIZE as f32,
                    INTERIOR_UV_SCALE,
                    INTERIOR_UV_BIAS,
                    view.zoom,
                ],
            }),
        );

        // Split the ordered item list into the two instance streams, remembering
        // for each item which stream slot it draws from. The *order* of `items` is
        // what has to survive — a matte must composite over the tiles below it and
        // under the tiles above — so the draw loop in `encode_composite` walks
        // `items`, not these.
        let mut instances: Vec<Instance> = Vec::new();
        let mut tile_bgs: Vec<wgpu::BindGroup> = Vec::new();
        let mut mattes: Vec<MatteInstance> = Vec::new();
        for item in items {
            match item {
                CompositeItem::Tile {
                    coord,
                    handle,
                    opacity,
                } => {
                    instances.push(Instance {
                        origin: coord.origin().to_array(),
                        opacity: *opacity,
                    });
                    tile_bgs.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("stark composite tile bg"),
                        layout: &self.tile_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(handle.color_view()),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(handle.aux_view()),
                            },
                        ],
                    }));
                }
                CompositeItem::Matte(m) => mattes.push(MatteInstance {
                    rect: m.rect,
                    channels: m.channels,
                    opacity: m.opacity,
                    _pad: [0.0; 3],
                }),
            }
        }
        if !instances.is_empty() {
            if instances.len() > self.instance_cap {
                self.instances = alloc_instances(device, instances.len());
                self.instance_cap = instances.len();
            }
            self.ctx
                .queue
                .write_buffer(&self.instances, 0, bytemuck::cast_slice(&instances));
        }
        if !mattes.is_empty() {
            if mattes.len() > self.matte_cap {
                self.matte_instances = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("stark matte instances"),
                    size: (std::mem::size_of::<MatteInstance>() * mattes.len()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.matte_cap = mattes.len();
            }
            self.ctx
                .queue
                .write_buffer(&self.matte_instances, 0, bytemuck::cast_slice(&mattes));
        }
        tile_bgs
    }

    /// Encode pass A: every item composited into `color` + `aux`, in stack order.
    /// Requires a preceding [`Self::prepare_composite`] for the same `items`.
    fn encode_composite(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        color: &wgpu::TextureView,
        aux: &wgpu::TextureView,
        items: &[CompositeItem],
        tile_bgs: &[wgpu::BindGroup],
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("stark composite pass"),
            color_attachments: &[
                Some(clear_attachment(color, wgpu::Color::TRANSPARENT)),
                Some(clear_attachment(aux, wgpu::Color::TRANSPARENT)),
            ],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        // Walk the stack in order, switching pipelines where a matte sits
        // between runs of tiles. Both pipelines share group 0 (the view
        // uniform), so only the vertex buffer and pipeline change.
        pass.set_bind_group(0, &self.view_bg, &[]);
        let (mut tile_i, mut matte_i) = (0u32, 0u32);
        let mut pipeline_is_matte = None;
        for item in items {
            match item {
                CompositeItem::Tile { .. } => {
                    if pipeline_is_matte != Some(false) {
                        pass.set_pipeline(&self.composite_pipeline);
                        pass.set_vertex_buffer(0, self.instances.slice(..));
                        pipeline_is_matte = Some(false);
                    }
                    pass.set_bind_group(1, &tile_bgs[tile_i as usize], &[]);
                    pass.draw(0..4, tile_i..tile_i + 1);
                    tile_i += 1;
                }
                CompositeItem::Matte(_) => {
                    if pipeline_is_matte != Some(true) {
                        pass.set_pipeline(&self.matte_pipeline);
                        pass.set_vertex_buffer(0, self.matte_instances.slice(..));
                        pipeline_is_matte = Some(true);
                    }
                    pass.draw(0..4, matte_i..matte_i + 1);
                    matte_i += 1;
                }
            }
        }
    }

    /// Composite `items` into caller-supplied targets and **stop there** — pass A
    /// alone, with no media pass over it.
    ///
    /// This is the eyedropper's sampling path (MISSING_FEATURES §0.2). What lands in
    /// `color` is the paint's own channels in the document's working space, which is
    /// what a picker has to read: the lit result has been through image-based
    /// lighting, a tonemap and an sRGB encode, so picking *that* would hand back a
    /// colour the palette never mixed — and in a Mixbox document (DESIGN.md §6.7) a
    /// pigment mixture that cannot be picked back up, which is the point of mixing
    /// in pigment space at all.
    ///
    /// `color` and `aux` must carry the formats [`Self::channel_formats`] reports,
    /// and be `view.viewport` in size. The compositor's own offscreen targets are
    /// left alone, unlike in [`Self::render`], which resizes them to whatever view
    /// it is given — so a sample never disturbs what is on screen.
    pub fn composite_channels(
        &mut self,
        color: &wgpu::TextureView,
        aux: &wgpu::TextureView,
        view: ViewTransform,
        items: &[CompositeItem],
    ) {
        let tile_bgs = self.prepare_composite(view, items);
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("stark pick encoder"),
            });
        self.encode_composite(&mut encoder, color, aux, items, &tile_bgs);
        self.ctx.queue.submit([encoder.finish()]);
    }

    /// Composite `tiles`, light the result into `target` under `view`, and outline
    /// each of `outlines` over it (DESIGN.md §6.8 — a universal selection draws
    /// nothing, so an unmasked document costs one skipped iteration).
    pub fn render(
        &mut self,
        target: &wgpu::TextureView,
        view: ViewTransform,
        bg_channels: [f32; 4],
        items: &[CompositeItem],
        outlines: &[SelectionOutline<'_>],
        transparent: bool,
    ) {
        if view.viewport != self.size {
            self.size = view.viewport;
            self.rebuild_offscreen();
        }
        let tile_bgs = self.prepare_composite(view, items);
        // Bound after everything that needs `&mut self`.
        let device = &self.ctx.device;

        // Screen→canvas mapping for sampling the surface bump in canvas space, so
        // the weave stays attached to the canvas as it pans/zooms (DESIGN.md §6.4).
        let inv_zoom = 1.0 / view.zoom;
        let canvas_origin = view.center
            - crate::geom::Vec2::new(view.viewport.width as f32, view.viewport.height as f32)
                * (0.5 * inv_zoom);

        // Diffuse samples a heavily-blurred high mip ≈ hemispherical irradiance; the
        // level is the environment's own, so the CPU-side normalization below is
        // reading exactly the texels the shader will. The Cook–Torrance specular picks
        // its own mip from roughness, spanning the whole chain (roughness 0 → mip 0
        // sharp; roughness 1 → the diffuse level, the hemispherical average).
        let diffuse_lod = self.environment.diffuse_lod as f32;
        // Exposure belongs to the light, not to a knob beside it: each environment is
        // shown at the value it was judged at (DESIGN.md §6.3). Normalized by the
        // irradiance a *flat* canvas receives, so `1.0` means the same thing in every
        // environment — an unrelieved patch of paint comes back out its own colour.
        let exposure = self.environment.exposure / self.environment.flat_irradiance;

        // Media uniform.
        self.ctx.queue.write_buffer(
            &self.media_buf,
            0,
            bytemuck::bytes_of(&MediaUniform {
                light: [0.0, 0.0, 0.0, self.media.height_strength],
                bg: bg_channels,
                shade: [exposure, diffuse_lod, self.media.specular, 0.0],
                surf_a: [
                    canvas_origin.x,
                    canvas_origin.y,
                    inv_zoom,
                    1.0 / SURFACE_TILE_PX,
                ],
                surf_b: [
                    self.media.surface_strength,
                    // Transparent export: the media pass skips the substrate and
                    // carries the paint's visible alpha out (FRAME_DESIGN.md §6).
                    if transparent { 1.0 } else { 0.0 },
                    0.0,
                    0.0,
                ],
            }),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stark composite encoder"),
        });

        // Pass A: composite tiles into offscreen color + aux.
        self.encode_composite(
            &mut encoder,
            &self.comp_color_view,
            &self.comp_aux_view,
            items,
            &tile_bgs,
        );

        // Pass B: media/lighting → target.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stark media pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        // The media pass covers every texel, so the clear only
                        // matters for what alpha an untouched texel would keep —
                        // transparent, on an export that wants a cut-out.
                        load: wgpu::LoadOp::Clear(if transparent {
                            wgpu::Color::TRANSPARENT
                        } else {
                            wgpu::Color::BLACK
                        }),
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

        // Pass C: the selection outlines, over the lit image — the local actor's and
        // every present peer's, one instanced quad per mask tile of each
        // (PEER_DESIGN.md §3). Flattened into one instance stream so N collaborators
        // still cost one pass.
        let mut overlay_instances: Vec<OverlayInstance> = Vec::new();
        let mut mask_tiles: Vec<wgpu::BindGroup> = Vec::new();
        for outline in outlines {
            if outline.selection.is_universal() {
                continue;
            }
            let tint = match outline.tint {
                Some([r, g, b]) => [r, g, b, PEER_OUTLINE_ALPHA],
                None => [0.0; 4],
            };
            for (coord, handle) in outline.selection.tiles() {
                overlay_instances.push(OverlayInstance {
                    origin: coord.origin().to_array(),
                    tint,
                });
                mask_tiles.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("stark overlay tile bg"),
                    layout: &self.overlay_tile_bgl,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(handle.view()),
                    }],
                }));
            }
        }
        if !mask_tiles.is_empty() {
            if overlay_instances.len() > self.overlay_cap {
                self.overlay_instances = alloc_overlay(device, overlay_instances.len());
                self.overlay_cap = overlay_instances.len();
            }
            self.ctx.queue.write_buffer(
                &self.overlay_instances,
                0,
                bytemuck::cast_slice(&overlay_instances),
            );

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stark selection overlay pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.overlay_pipeline);
            pass.set_bind_group(0, &self.overlay_view_bg, &[]);
            pass.set_vertex_buffer(0, self.overlay_instances.slice(..));
            for (i, bg) in mask_tiles.iter().enumerate() {
                let idx = i as u32;
                pass.set_bind_group(1, bg, &[]);
                pass.draw(0..4, idx..idx + 1);
            }
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

/// How strongly another actor's selection outline reads against the artwork. Well
/// below the local one, which is a full-strength dashed line: yours is a thing you
/// act through, theirs is a thing you need only be aware of.
const PEER_OUTLINE_ALPHA: f32 = 0.55;

fn alloc_overlay(device: &wgpu::Device, count: usize) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("stark overlay instances"),
        contents: bytemuck::cast_slice(&vec![
            OverlayInstance {
                origin: [0.0; 2],
                tint: [0.0; 4],
            };
            count.max(1)
        ]),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    })
}

fn alloc_instances(device: &wgpu::Device, count: usize) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("stark composite instances"),
        contents: bytemuck::cast_slice(&vec![
            Instance {
                origin: [0.0; 2],
                opacity: 1.0
            };
            count.max(1)
        ]),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    })
}

/// The inputs to [`make_offscreen`]. Every field is a `Compositor` field, which is
/// why [`Compositor::rebuild_offscreen`] exists — only the constructor, which has no
/// `self` yet, fills one in by hand.
struct OffscreenDesc<'a> {
    device: &'a wgpu::Device,
    size: Extent2,
    color_format: wgpu::TextureFormat,
    aux_format: wgpu::TextureFormat,
    media_bgl: &'a wgpu::BindGroupLayout,
    media_buf: &'a wgpu::Buffer,
    surface: &'a Surface,
    environment: &'a Environment,
}

/// (Re)create the offscreen composite targets and the media bind group.
fn make_offscreen(d: OffscreenDesc<'_>) -> (wgpu::TextureView, wgpu::TextureView, wgpu::BindGroup) {
    let OffscreenDesc {
        device,
        size,
        color_format,
        aux_format,
        media_bgl,
        media_buf,
        surface,
        environment,
    } = d;
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
