//! The brush engine: **swept-segment** stroke rasterization with copy-on-write
//! tiles (DESIGN.md §6.2, §5.2, §6.6, §6.7).
//!
//! Rather than stamping discrete dabs, each short segment of the flattened curve
//! is drawn as one oriented quad whose coverage is the brush *swept* along it —
//! the path integral of the footprint. Because alpha-"over" is additive in
//! optical depth `τ = −ln(1−α)`, the swept depth of a segment is a difference of
//! the brush's precomputed prefix-τ texture (`prefix(u) − prefix(u−d)`), and the
//! existing premultiplied-over blend across overlapping segment quads sums those
//! depths *exactly* — reconstructing the continuous stroke with no banding, no
//! scratch buffer, and no second pass.
//!
//! The renderer is parameterized by a [`ColorSpace`] (formats, blends, channel
//! mapping, shader). It holds only immutable GPU objects plus `Arc`-backed
//! handles, so it is cheap to `Clone` and can live in the `Action::Context` (§5).

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};
use rpds::HashTrieMap;
use wgpu::util::DeviceExt;

use crate::assets::{build_prefix_tau, AssetStore};
use crate::colorspace::ColorSpace;
use crate::command::InputSample;
use crate::document::{BrushShape, OrientationSource, StrokeRecord};
use crate::geom::{
    TileCoord, Vec2, INTERIOR_UV_BIAS, INTERIOR_UV_SCALE, TILE_APRON, TILE_SIZE, TILE_TEX,
};
use crate::gpu::surface::{Surface, SURFACE_TILE_PX};
use crate::gpu::context::GpuContext;
use crate::gpu::tile::{AllocSource, SCRATCH_AUX_FORMAT, TilePairHandle, TilePool};

/// Global tuning so a default brush (`flow = 1`) reads as a solid stroke;
/// `flow` is an optical-depth-per-length rate (DESIGN.md §6.2).
const SWEEP_FLOW_SCALE: f32 = 1.0;
/// Resolution of the generated round-tip prefix texture.
const ROUND_RES: u32 = 256;

/// Format of the per-tile **scrape** target (the third sweep MRT on the pressure-modulated
/// scrape path): one channel carries the coverage-weighted per-pixel load Σ(load·√cov) the
/// integrate uses for the exact removal (DESIGN.md §6.2). Transient (sweep → integrate),
/// never persisted.
const SCRAPE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;

/// Lateral reservoir bands across the brush tip: each picks up canvas color at its
/// own offset, so one side of the brush can carry a different color than the other
/// (DESIGN.md §6.2). This is the *height* of the reservoir texture; the mixer's
/// `BANDS` constant must match. The stamp shader samples the texture and is
/// otherwise agnostic to this count.
const LATERAL_BANDS: u32 = 64;

/// Largest base-region edge (canvas px) composited for wet-mixing pickup. Strokes
/// whose bounding box exceeds this skip pickup for that stroke — rare, and it
/// bounds the transient GPU memory of the region texture (DESIGN.md §6.2).
const MAX_REGION_DIM: u32 = 2048;

/// Fixed advection iterations per stroke for the drag axis (DESIGN.md §6.2). Constant so
/// replay is deterministic. The conservative finite-volume advection is explicit, so each
/// step may move at most ~CFL (≈0.4) texels (`fluid_advect.wesl`); unlike the old
/// unconditionally-stable semi-Lagrangian back-trace it cannot take one big jump, so a
/// satisfying drag throw needs many small stable steps. At full `drag` the injected
/// `FLOW_DRAG_PX` saturates the CFL clamp across the footprint core (a smooth top-hat →
/// near-zero internal divergence → no piling), so the throw is ≈ `FLOW_ITERS · CFL` px.
const FLOW_ITERS: u32 = 24;
const FLOW_DRAG_PX: f32 = 2.5;

/// Standard deviation (canvas px) of the separable-Gaussian **bleed** at full
/// `BrushDynamics::bleed`. The bleed runs in 2 passes (vs the old explicit diffusion's
/// many) and reaches ~`3·MAX_SIGMA` px — kept in sync with `MAX_RADIUS` in `blur.wesl`,
/// and well under the region's one-tile halo so the bleed reads only composited data.
const MAX_SIGMA: f32 = 14.0;

/// One swept segment of the stroke.
#[derive(Copy, Clone)]
struct Segment {
    start: Vec2,
    dir: Vec2,
    radius: f32,
    length: f32,
    flow: f32,
    height: f32,
    wet: f32,
    /// Paint opacity laid by this segment (the brush's opacity × remaining load).
    /// Drives the color/opacity channel; thickness (`height`) is independent now
    /// (DESIGN.md §6.1, normalized representation).
    opacity: f32,
    /// Per-segment **scrape** rate (the `load` axis modulated by pen pressure, DESIGN.md
    /// §6.2): the fraction of canvas height this segment lifts onto the tool. The mixer reads
    /// it to drive the per-band intake; the scrape sweep variant routes it to the integrate
    /// for the exact per-pixel removal.
    load: f32,
    /// Per-segment **deposit** rate (the `deposit` axis modulated by pen tilt toward motion,
    /// DESIGN.md §6.2): the fraction of tool height this segment lays back down.
    deposit: f32,
    /// Shape orientation for this segment as a fraction of a full turn ∈ [0, 1): the
    /// relative angle between the shape's native axis and the travel direction, used to
    /// pick the prefix-τ orientation layer. 0 for follow-stroke (DESIGN.md §6.6).
    orient: f32,
}

/// Per-segment instance data for the sweep shader. Padded to 48 bytes so the same
/// buffer is a valid `std430 array<Instance>` for the wet-mixing compute pass,
/// which reads it to drive the reservoir scan (see `mixer.wesl`).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SegmentInstance {
    start: [f32; 2],
    dir: [f32; 2],    // unit tangent
    geom: [f32; 4],   // radius, length, flow, reservoir column u ∈ [0,1]
    aux: [f32; 4],    // height (thickness rate), wet, opacity, reservoir u-advance per radius
    extra: [f32; 4],  // orientation (turns ∈ [0,1)), load (scrape rate), deposit rate, _ (→ 64 B std430)
}

/// Per-tile uniform: the tile *texture's* top-left in canvas px + canvas→NDC
/// scale. The texture origin is the interior origin minus the apron, so the
/// stroke rasterizes into the apron too (keeping it consistent with the
/// neighbor's interior — see [`crate::geom::TILE_APRON`]).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TileXform {
    params: [f32; 4], // tex_origin.x, tex_origin.y, 2/TILE_TEX, reservoir u per px travelled
    surf: [f32; 4],   // inv surface-tile (canvas px → bump uv), tooth, _, _
}

/// Mirrors `View` in `composite.wesl`: canvas→region NDC + tile/apron uv mapping.
/// Used to composite the base into a 1:1 region texture for wet-mixing pickup.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ViewUniform {
    st: [f32; 4],   // scale.xy, translate.xy
    misc: [f32; 4], // tile_size, uv_scale, uv_bias, _
}

/// Per-tile instance for the region composite: canvas origin + layer opacity.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TileInstance {
    origin: [f32; 2],
    opacity: f32,
}

/// Mirrors `Params` in `mixer.wesl` (64 bytes).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MixerUniform {
    brush_ch: [f32; 4],    // brush colour channels (.xyz); .w = `add` axis
    origin_dims: [f32; 4], // region origin.xy (canvas px), dims.xy (px)
    knobs: [f32; 4],       // charge (tool pre-load height), brush opacity, flatten_step, _
    counts: [u32; 4],      // segment_count, lead-in columns, reservoir width, _
}

#[derive(Clone)]
pub struct StrokeRenderer {
    ctx: GpuContext,
    color_space: Arc<dyn ColorSpace>,
    pipeline: wgpu::RenderPipeline,
    /// Sweep variant with a third MRT target (the [`SCRAPE_FORMAT`] per-pixel load) for the
    /// pressure-modulated scrape path. Structurally identical bind groups (the `fs_scrape`
    /// entry only adds an output), so it shares `uniform_bgl`/`prefix_bgl`/etc.
    scrape_pipeline: wgpu::RenderPipeline,
    /// 1×1 zero [`SCRAPE_FORMAT`] view bound to the integrate's scrape slot on the
    /// non-scrape path (the shader never samples it there — `mode.w = 0`).
    dummy_scrape: wgpu::TextureView,
    uniform_bgl: wgpu::BindGroupLayout,
    prefix_bgl: wgpu::BindGroupLayout,
    /// Cached round-tip prefix-τ, keyed by `hardness.to_bits()`.
    round_prefix: Arc<Mutex<Option<(u32, wgpu::TextureView)>>>,
    /// Canvas surface (group 2 of the sweep pipeline): bump + sampler for the tooth
    /// gate — currently a pass-through stub (`surface_tooth` TODO in stamp_common.wesl);
    /// no stamp/integrate shader reads the surface today, the weave shows through the
    /// media pass instead.
    surface_bg: wgpu::BindGroup,
    /// Color reservoir (group 3 of the sweep pipeline): the per-segment × per-band
    /// color the deposit samples. `reservoir_sampler` is linear+clamp so it blends
    /// bands (and segments) for free.
    reservoir_bgl: wgpu::BindGroupLayout,
    reservoir_sampler: wgpu::Sampler,

    // Wet-mixing pickup (Mixer dynamics): composite the base into a region, then a
    // serial compute scan writes the per-segment × per-band reservoir texture — all
    // on the GPU, no readback (DESIGN.md §6.2). Built once; cheap to clone (wgpu
    // handles are Arc-backed).
    composite_pipeline: wgpu::RenderPipeline,
    /// Same composite, but with the wide [`SCRATCH_AUX_FORMAT`] aux target — for
    /// compositing *scratch* tiles into a region without dropping their extra
    /// channels (the flow's footprint-coverage mask rides scratch aux `.z`).
    composite_wide_pipeline: wgpu::RenderPipeline,
    composite_sampler: wgpu::Sampler,
    composite_view_bgl: wgpu::BindGroupLayout,
    composite_tile_bgl: wgpu::BindGroupLayout,
    mixer_pipeline: wgpu::ComputePipeline,
    mixer_bgl: wgpu::BindGroupLayout,

    // Stroke integrate (DESIGN.md §6.2/§6.1): a fullscreen pass reads the base tile +
    // the stroke's footprint scratch and writes `new = f(base, scratch)` into a fresh
    // CoW tile's color+aux MRT. Normal mode = premultiplied-over + additive thickness.
    integrate_pipeline: wgpu::RenderPipeline,
    integrate_bgl: wgpu::BindGroupLayout,

    // Wet bleed (Wet dynamics, DESIGN.md §6.2): a separable Gaussian blur (two
    // fullscreen passes) over a composited stroke region; built once.
    blur_pipeline: wgpu::RenderPipeline,
    blur_bgl: wgpu::BindGroupLayout,

    // Fluid advect+inject micro-sim (Fluid dynamics, DESIGN.md §6.2): a velocity
    // injection pass (segments → velocity region) + a semi-Lagrangian advection pass
    // (ping-pong over the region); built once.
    fluid_inject_pipeline: wgpu::RenderPipeline,
    fluid_inject_bgl: wgpu::BindGroupLayout,
    fluid_advect_pipeline: wgpu::RenderPipeline,
    fluid_advect_bgl: wgpu::BindGroupLayout,
}

/// GPU resources scoped to one `render()` call: the wet-flow / smear-pickup region
/// (and reservoir) textures and the instance buffer. They're sized per-stroke, so —
/// unlike the fixed-`TILE_TEX` tile pool — they can't be recycled, and a *live* wet
/// stroke re-renders on every pointer move. Left to drop they'd only release the JS
/// handle and wait on GC, which can't keep up → the tab OOMs. So they're collected here
/// (cheap `Arc` clones) and **`destroy()`d on drop**, which `render` arranges to happen
/// right after the submit — safe, because WebGPU defers the real free until the in-flight
/// work referencing them completes.
#[derive(Default)]
struct ScopedResources {
    textures: Vec<wgpu::Texture>,
    buffers: Vec<wgpu::Buffer>,
}

impl ScopedResources {
    /// Register a per-stroke texture; returns it unchanged (the clone keeps the GPU
    /// resource alive until this `ScopedResources` drops).
    fn texture(&mut self, tex: wgpu::Texture) -> wgpu::Texture {
        self.textures.push(tex.clone());
        tex
    }

    /// Register a per-stroke buffer; returns it unchanged.
    fn buffer(&mut self, buf: wgpu::Buffer) -> wgpu::Buffer {
        self.buffers.push(buf.clone());
        buf
    }
}

impl Drop for ScopedResources {
    fn drop(&mut self) {
        if !self.textures.is_empty() || !self.buffers.is_empty() {
            tracing::trace!(
                textures = self.textures.len(),
                buffers = self.buffers.len(),
                "destroying scoped stroke resources",
            );
        }
        for tex in self.textures.drain(..) {
            tex.destroy();
        }
        for buf in self.buffers.drain(..) {
            buf.destroy();
        }
    }
}

/// Params for the integrate pass (`integrate.wesl` `Params`).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct IntegrateUniform {
    mode: [f32; 4], // mode.x = load (lift fraction); mode.y = ridge strength; .zw unused
}

/// Params for the separable Gaussian bleed pass (`blur.wesl` `Params`).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BlurUniform {
    knobs: [f32; 4], // dir.x, dir.y, sigma, combine
}

/// Mirrors `View` in `fluid_inject.wesl`: canvas→region NDC + velocity magnitude.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct InjectUniform {
    st: [f32; 4],    // scale.xy, translate.xy
    knobs: [f32; 4], // velocity magnitude (px/iter), _, _, _
}

/// Params for the fluid advection pass (`fluid_advect.wesl` `Params`).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FluidUniform {
    knobs: [f32; 4], // dt, _, _, _
}

impl StrokeRenderer {
    pub fn new(ctx: &GpuContext, color_space: Arc<dyn ColorSpace>, surface: Surface) -> Self {
        let device = &ctx.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark sweep"),
            source: wgpu::ShaderSource::Wgsl(color_space.stamp_shader().into()),
        });

        let uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stark sweep uniform bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // The prefix-τ texture is a R32Float 2D-array (x, y, + orientation layers), sampled
        // via textureLoad (not filterable), so the shader does its own trilinear lookup.
        let prefix_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stark sweep prefix bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            }],
        });

        // Group 2: the canvas surface (bump + sampler) for deposition tooth.
        let surface_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stark sweep surface bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
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
        let surface_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark sweep surface bg"),
            layout: &surface_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&surface.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&surface.sampler),
                },
            ],
        });

        // Group 3: the stroke's color reservoir (per-segment × per-band) + a
        // linear/clamp sampler, so the deposit blends bands and segments for free, plus a
        // companion height reservoir (a smear's carried paint-height rate; a dummy for
        // plain brushes, which lay the brush's own height).
        let res_tex = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let reservoir_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stark sweep reservoir bgl"),
            entries: &[
                res_tex(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                res_tex(2),
            ],
        });
        let reservoir_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("stark sweep reservoir sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("stark sweep layout"),
            bind_group_layouts: &[
                Some(&uniform_bgl),
                Some(&prefix_bgl),
                Some(&surface_bgl),
                Some(&reservoir_bgl),
            ],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("stark sweep pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<SegmentInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x2, 1 => Float32x2, 2 => Float32x4, 3 => Float32x4, 4 => Float32x4
                    ],
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
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: color_space.color_format(),
                        blend: Some(color_space.color_blend()),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // The stamp renders into a *scratch* tile, whose aux is the wide
                    // SCRATCH_AUX_FORMAT (thickness, wet, smear-lifted height) — not the
                    // compact persistent aux. Additive blend across overlapping segments.
                    Some(wgpu::ColorTargetState {
                        format: SCRATCH_AUX_FORMAT,
                        blend: Some(color_space.aux_blend()),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
            }),
            multiview_mask: None,
            cache: None,
        });

        // Scrape variant: identical to the sweep pipeline but with a third MRT target (the
        // per-pixel load, additively blended like the aux) and the `fs_scrape` fragment.
        // Used only when pen pressure modulates the scrape (`load_pressure > 0`); every
        // other stroke uses the 2-target `pipeline` above (DESIGN.md §6.2).
        let scrape_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("stark sweep scrape pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<SegmentInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x2, 1 => Float32x2, 2 => Float32x4, 3 => Float32x4, 4 => Float32x4
                    ],
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
                entry_point: Some("fs_scrape"),
                compilation_options: Default::default(),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: color_space.color_format(),
                        blend: Some(color_space.color_blend()),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: SCRATCH_AUX_FORMAT,
                        blend: Some(color_space.aux_blend()),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // Per-pixel load Σ(load·√cov): additive across overlapping segments.
                    Some(wgpu::ColorTargetState {
                        format: SCRAPE_FORMAT,
                        blend: Some(color_space.aux_blend()),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
            }),
            multiview_mask: None,
            cache: None,
        });

        // 1×1 zero scrape texture for the integrate's scrape slot on the non-scrape path.
        let dummy_scrape = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("stark dummy scrape"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: SCRAPE_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());

        // ---- Wet-mixing pickup: base-region composite + reservoir compute ----
        let (composite_pipeline, composite_view_bgl, composite_tile_bgl) =
            build_composite_pipeline(device, color_space.as_ref(), color_space.aux_format());
        // The wide variant's (structurally identical) bind group layouts are discarded:
        // WebGPU bind-group compatibility is structural, so groups made with the layouts
        // above bind to both pipelines.
        let (composite_wide_pipeline, _, _) =
            build_composite_pipeline(device, color_space.as_ref(), SCRATCH_AUX_FORMAT);
        let composite_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("stark mixer composite sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let (mixer_pipeline, mixer_bgl) = build_mixer_pipeline(device);
        let (integrate_pipeline, integrate_bgl) = build_integrate_pipeline(device, color_space.as_ref());
        let (blur_pipeline, blur_bgl) = build_blur_pipeline(device, color_space.as_ref());
        let (fluid_inject_pipeline, fluid_inject_bgl) =
            build_fluid_inject_pipeline(device, color_space.as_ref());
        let (fluid_advect_pipeline, fluid_advect_bgl) =
            build_fluid_advect_pipeline(device, color_space.as_ref());

        Self {
            ctx: ctx.clone(),
            color_space,
            pipeline,
            scrape_pipeline,
            dummy_scrape,
            uniform_bgl,
            prefix_bgl,
            round_prefix: Arc::new(Mutex::new(None)),
            surface_bg,
            reservoir_bgl,
            reservoir_sampler,
            composite_pipeline,
            composite_wide_pipeline,
            composite_sampler,
            composite_view_bgl,
            composite_tile_bgl,
            mixer_pipeline,
            mixer_bgl,
            integrate_pipeline,
            integrate_bgl,
            blur_pipeline,
            blur_bgl,
            fluid_inject_pipeline,
            fluid_inject_bgl,
            fluid_advect_pipeline,
            fluid_advect_bgl,
        }
    }

    /// Swap the canvas surface bound to the sweep's tooth gate (group 2), without
    /// touching pipelines or pools. The gate is currently a pass-through stub
    /// (`surface_tooth` TODO), but keeping the binding current means tooth reads the
    /// right weave the moment it returns (DESIGN.md §6.4).
    pub fn set_surface(&mut self, surface: &Surface) {
        self.surface_bg = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark sweep surface bg"),
            layout: &self.pipeline.get_bind_group_layout(2),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&surface.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&surface.sampler),
                },
            ],
        });
    }

    /// Render `rec` over `base`, returning a copy-on-write tile map.
    pub fn render(
        &self,
        pool: &TilePool,
        assets: &AssetStore,
        base: &HashTrieMap<TileCoord, TilePairHandle>,
        rec: &StrokeRecord,
    ) -> HashTrieMap<TileCoord, TilePairHandle> {
        let rgb = [rec.brush.color[0], rec.brush.color[1], rec.brush.color[2]];
        let channels = self.color_space.rgb_to_channels(rgb);
        let segments = generate_segments(rec);
        if segments.is_empty() {
            return base.clone();
        }

        // Per-stroke region/reservoir textures + the instance buffer register here and are
        // `destroy()`d when this drops (at the end of `render`, after the submit below) —
        // freeing them deterministically instead of leaking to JS GC (DESIGN.md §6.2).
        let mut scoped = ScopedResources::default();

        // Resolve the brush's prefix-τ texture: image brushes from the asset
        // store; the round tip generated (and cached) from its hardness.
        let prefix_view = match rec.brush.shape {
            BrushShape::Stamp(id) => assets
                .prefix_view(id)
                .unwrap_or_else(|| self.round_prefix(rec.brush.hardness)),
            BrushShape::Round => self.round_prefix(rec.brush.hardness),
        };

        let device = &self.ctx.device;
        let prefix_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark sweep prefix bg"),
            layout: &self.prefix_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&prefix_view),
            }],
        });

        // The reservoir has one column **per segment** (the mixer writes net paint at the
        // integer column `lead + i`), plus a half-brush of *lead-in*/*lead-out* padding so
        // the round end caps — which the sweep extends a radius beyond each endpoint — are
        // covered by the mixer's walk into those pads rather than clamping to a flat slab.
        // The deposit must sample **by that same column index**, not by arc distance: with
        // non-uniform segments (the short flatten remainder at the end, or a curve's varying
        // lengths) `dist/step` drifts away from `i`, so an arc-based mapping would read the
        // wrong column — a discontinuity ≈ one segment long at the stroke's end (the bug this
        // fixes). `geom.w` is segment `i`'s start column-center; `aux.w` is its `u`-advance
        // per radius travelled (`radius/(length·width)`), so traversing the segment advances
        // exactly one column to the next — independent of how long the segment is.
        let count = segments.len();
        let total = segments.iter().map(|s| s.length).sum::<f32>().max(1e-3);
        let step = total / count as f32; // ≈ FLATTEN_STEP, used only for the lead/tail pad
        let lead = (segments.first().unwrap().radius / step).ceil() as u32;
        let tail = (segments.last().unwrap().radius / step).ceil() as u32;
        let width = lead + count as u32 + tail;
        let instances: Vec<SegmentInstance> = segments
            .iter()
            .enumerate()
            .map(|(i, s)| SegmentInstance {
                start: s.start.to_array(),
                dir: s.dir.to_array(),
                geom: [
                    s.radius,
                    s.length,
                    s.flow,
                    (lead as f32 + i as f32 + 0.5) / width as f32,
                ],
                // aux.w = u-advance per radius travelled, so local.x ∈ [0, length/radius]
                // sweeps exactly one column (geom.w → next column) regardless of segment len.
                aux: [s.height, s.wet, s.opacity, s.radius / (s.length.max(1e-3) * width as f32)],
                extra: [s.orient, s.load, s.deposit, 0.0],
            })
            .collect();
        // The mixer compute pass reads this buffer to drive its reservoir scan, so
        // it is a storage source as well as vertex data. Written via `write_buffer`
        // (not `create_buffer_init`, which maps-at-creation): a long stroke makes this
        // buffer large, and Chrome/Dawn caps map-at-creation buffers well below the
        // normal `maxBufferSize`, so a long stroke would panic in `createBuffer`.
        let instance_bytes = bytemuck::cast_slice(&instances);
        let instance_buf = scoped.buffer(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stark sweep instances"),
            size: instance_bytes.len() as u64,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.ctx.queue.write_buffer(&instance_buf, 0, instance_bytes);

        let coords = affected_tiles(&segments);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stark stroke commit"),
        });

        // The unified tool (DESIGN §6.2): one set of axes, each gated independently. The
        // reservoir scan runs whenever the tool lifts or deposits — a serial walk that lifts
        // canvas paint onto a per-band tool, deposits a fraction back, and folds in the
        // brush's own `add` colour, emitting the deposit payload plus a subtractive
        // lift-mass (so the integrate scrapes exactly what was lifted). A pure-`add` brush
        // (no load/deposit) takes the cheap flat-reservoir path: a 1×1 texel of the brush
        // colour (scaled by `add`) so the deposit stays uniform.
        let d = rec.brush.dynamics;
        // `add` gates how much of the brush's own paint the deposit lays (with `add = 0` the
        // brush only works paint already on the canvas). Folded into the reservoir colour
        // (mixer/flat) and used to gate the brush's own height/wet in the stamp.
        let add_frac = d.add;
        // The mixer's per-band tool now reads lift/deposit **per-segment** (the instance
        // `extra.yz`), so the scan uniform carries the stroke-constant rest: the tool's
        // initial pre-`charge` and the brush opacity (for the pre-charge's per-unit colour),
        // plus the flatten step. Runs whenever the tool lifts or deposits (DESIGN §6.2).
        let scan = (d.load > 0.0 || d.deposit > 0.0)
            .then_some([d.charge, rec.brush.color[3], crate::path::FLATTEN_STEP, 0.0]);
        // The scan runs only when it actually fired; an oversized stroke skips the region
        // composite and degrades to a plain (flat-reservoir) deposit.
        let mut reservoir_tex = |label| {
            scoped.texture(device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height: LATERAL_BANDS,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            }))
        };
        // The colour reservoir and the companion height reservoir (the smear-lifted
        // height; zero for a non-smear brush, whose own height is laid from the per-
        // segment attribute). Both are always bound, so the deposit reads them uniformly.
        let (reservoir_view, aux_view) = match scan.zip(stroke_bbox(&segments)) {
            Some((knobs, bbox)) => {
                let color = reservoir_tex("stark reservoir");
                let aux = reservoir_tex("stark aux reservoir");
                let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
                let aux_view = aux.create_view(&wgpu::TextureViewDescriptor::default());
                self.encode_mixer(
                    &mut scoped, &mut encoder, base, &coords, &segments, &instance_buf, &color_view,
                    &aux_view, bbox, (lead, width), channels, add_frac, knobs,
                );
                (color_view, aux_view)
            }
            None => self.flat_reservoir(&mut encoder, channels, add_frac),
        };
        let reservoir_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark sweep reservoir bg"),
            layout: &self.reservoir_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&reservoir_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.reservoir_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&aux_view),
                },
            ],
        });

        // Every brush rasterizes its footprint into a *cleared scratch* tile, then the
        // integrate pass merges it over the base into a fresh CoW tile (DESIGN.md
        // §6.2/§6.1). `empty` (cleared) stands in as the base wherever the stroke
        // touches bare canvas — acquired tiles are undefined, so clear it once here.
        let empty = pool.acquire(AllocSource::IntegrateEmptyBase);
        {
            let clear = wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            };
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stark integrate empty clear"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment { view: empty.color_view(), resolve_target: None, depth_slice: None, ops: clear }),
                    Some(wgpu::RenderPassColorAttachment { view: empty.aux_view(), resolve_target: None, depth_slice: None, ops: clear }),
                ],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }

        // Scrape path: when pen pressure modulates the scrape (`load_pressure > 0`), the
        // removal is no longer a stroke-constant `load` — it varies per segment. The sweep
        // then writes a coverage-weighted per-pixel load into a third scratch target and the
        // integrate reads it (`mode.w = 1`). With `load_pressure = 0` (every existing brush)
        // we keep the exact constant-`mode.x` path → bit-identical output (DESIGN §6.2).
        let scrape = d.load > 0.0 && d.load_pressure > 0.0;

        // Integrate params: `mode.x` is the constant `load` axis — the integrate removes
        // `load × contact` of the base height per-pixel (exact, no banded residue), while
        // the tool's deposit rides the mixer's banded reservoir. With `load = 0` and a flat
        // reservoir it reduces exactly to the old additive "over" deposit (DESIGN §6.2).
        // `mode.y` is the conservative-ridge strength; `mode.w` selects the per-pixel scrape.
        let integrate_mode = [d.load, d.ridge, 0.0, if scrape { 1.0 } else { 0.0 }];

        // Flow (drag + bleed): after the deposit, work the freshly-laid region (DESIGN.md
        // §6.2). Collect this stroke's scratch footprints so the flow localizes to where the
        // brush bore down. Runs only when an axis is non-zero.
        let flow = d.drag > 0.0 || d.bleed > 0.0;
        let mut wet_scratch: Vec<(TileCoord, TilePairHandle)> = Vec::new();

        let mut new_map = base.clone();
        for coord in &coords {
            // Per-tile sweep transform: texture top-left = interior origin shifted
            // out by the apron, so the full TILE_TEX target maps to NDC [-1, 1].
            let apron = TILE_APRON as f32;
            let origin = coord.origin();
            let xform = TileXform {
                // params.w unused: the reservoir u-advance is now per-segment (instance aux.w).
                params: [origin.x - apron, origin.y - apron, 2.0 / TILE_TEX as f32, 0.0],
                // surf.z = the `add` fraction: the deposit lays `add·height_rate` of the
                // brush's own height plus the reservoir's smear-lifted height (one
                // continuous sum, no branch). Opacity always comes from the reservoir alpha.
                surf: [1.0 / SURFACE_TILE_PX, rec.brush.tooth, add_frac, 0.0],
            };
            let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stark sweep xform"),
                contents: bytemuck::bytes_of(&xform),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("stark sweep bg"),
                layout: &self.uniform_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ubuf.as_entire_binding(),
                }],
            });

            // Footprint → cleared scratch tile: within-stroke accumulation (the color
            // target over-blends opacity-premultiplied colour, the aux accumulates
            // thickness/wet/smear-lifted additively). The scratch aux is the wide format.
            let scratch = pool.acquire_scratch(AllocSource::MixerScratch);
            // On the scrape path, a third (transient) target collects the coverage-weighted
            // per-pixel load the integrate reads (DESIGN.md §6.2); scoped → freed after submit.
            let scrape_view = scrape.then(|| {
                scoped
                    .texture(device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("stark sweep scrape"),
                        size: wgpu::Extent3d { width: TILE_TEX, height: TILE_TEX, depth_or_array_layers: 1 },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: SCRAPE_FORMAT,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    }))
                    .create_view(&wgpu::TextureViewDescriptor::default())
            });
            {
                let clear = wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                };
                let mut color_attachments = vec![
                    Some(wgpu::RenderPassColorAttachment { view: scratch.color_view(), resolve_target: None, depth_slice: None, ops: clear }),
                    Some(wgpu::RenderPassColorAttachment { view: scratch.aux_view(), resolve_target: None, depth_slice: None, ops: clear }),
                ];
                if let Some(sv) = &scrape_view {
                    color_attachments.push(Some(wgpu::RenderPassColorAttachment { view: sv, resolve_target: None, depth_slice: None, ops: clear }));
                }
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("stark sweep pass"),
                    color_attachments: &color_attachments,
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(if scrape { &self.scrape_pipeline } else { &self.pipeline });
                pass.set_bind_group(0, &bind_group, &[]);
                pass.set_bind_group(1, &prefix_bg, &[]);
                pass.set_bind_group(2, &self.surface_bg, &[]);
                pass.set_bind_group(3, &reservoir_bg, &[]);
                pass.set_vertex_buffer(0, instance_buf.slice(..));
                pass.draw(0..4, 0..instances.len() as u32);
            }

            // Integrate the scratch slab over the base into a fresh CoW tile.
            let integrate_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stark integrate params"),
                contents: bytemuck::bytes_of(&IntegrateUniform {
                    mode: integrate_mode,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let dst = pool.acquire(AllocSource::IntegrateDestination);
            let base_tile = base.get(coord).unwrap_or(&empty);
            let integrate_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("stark integrate bg"),
                layout: &self.integrate_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: integrate_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(base_tile.color_view()) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(base_tile.aux_view()) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(scratch.color_view()) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(scratch.aux_view()) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(scrape_view.as_ref().unwrap_or(&self.dummy_scrape)) },
                ],
            });
            {
                let clear = wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                };
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("stark integrate"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment { view: dst.color_view(), resolve_target: None, depth_slice: None, ops: clear }),
                        Some(wgpu::RenderPassColorAttachment { view: dst.aux_view(), resolve_target: None, depth_slice: None, ops: clear }),
                    ],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&self.integrate_pipeline);
                pass.set_bind_group(0, &integrate_bg, &[]);
                pass.draw(0..3, 0..1);
            }
            if flow {
                wet_scratch.push((*coord, scratch.clone()));
            }
            new_map = new_map.insert(*coord, dst);
        }

        // Flow: drag (conservative advection) + bleed (diffusion) over the freshly-deposited
        // region (DESIGN.md §6.2). No-op when both axes are 0.
        if flow {
            self.encode_flow(
                &mut scoped, &mut encoder, pool, &mut new_map, &coords, &wet_scratch, &instance_buf,
                instances.len() as u32, d.bleed, d.drag,
            );
        }

        self.ctx.queue.submit([encoder.finish()]);

        // `scoped` drops here, *after* the submit — destroying this stroke's per-stroke
        // region/reservoir textures + instance buffer. They aren't pooled (sized per
        // stroke) and a live wet stroke re-renders every pointer move, so left to JS GC
        // they pile up and OOM the tab; `destroy()` after submit reclaims them at once
        // (WebGPU keeps the memory until the in-flight work that uses them completes).
        new_map
    }

    /// Debugging: run *only* the mixer reservoir scan for `rec` over `base` and read the
    /// two reservoir textures back to the CPU as `f32` RGBA (DESIGN.md §6.2). Returns
    /// `(reservoir_color, aux_reservoir, width, bands)` where `aux_reservoir.x` is the net
    /// height transfer per column. `None` if the stroke is empty or doesn't run the mixer
    /// (`lift = deposit = 0`). Used by the reservoir-visualization golden to inspect the
    /// per-column lift/deposit the deposit pass samples.
    pub fn debug_reservoir(
        &self,
        base: &HashTrieMap<TileCoord, TilePairHandle>,
        rec: &StrokeRecord,
    ) -> Option<(Vec<f32>, Vec<f32>, u32, u32)> {
        let d = rec.brush.dynamics;
        if d.load <= 0.0 && d.deposit <= 0.0 {
            return None;
        }
        let rgb = [rec.brush.color[0], rec.brush.color[1], rec.brush.color[2]];
        let channels = self.color_space.rgb_to_channels(rgb);
        let segments = generate_segments(rec);
        if segments.is_empty() {
            return None;
        }
        let bbox = stroke_bbox(&segments)?;

        let count = segments.len();
        let total = segments.iter().map(|s| s.length).sum::<f32>().max(1e-3);
        let step = total / count as f32;
        let lead = (segments.first().unwrap().radius / step).ceil() as u32;
        let tail = (segments.last().unwrap().radius / step).ceil() as u32;
        let width = lead + count as u32 + tail;
        let instances: Vec<SegmentInstance> = segments
            .iter()
            .enumerate()
            .map(|(i, s)| SegmentInstance {
                start: s.start.to_array(),
                dir: s.dir.to_array(),
                geom: [s.radius, s.length, s.flow, (lead as f32 + i as f32 + 0.5) / width as f32],
                aux: [s.height, s.wet, s.opacity, s.radius / (s.length.max(1e-3) * width as f32)],
                extra: [s.orient, s.load, s.deposit, 0.0],
            })
            .collect();

        let device = &self.ctx.device;
        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stark dbg instances"),
            size: std::mem::size_of_val(&instances[..]) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.ctx.queue.write_buffer(&instance_buf, 0, bytemuck::cast_slice(&instances));

        let coords = affected_tiles(&segments);
        // Lift/deposit are per-segment (instance `extra.yz`); the uniform carries charge +
        // brush opacity + flatten step (see `render`).
        let knobs = [d.charge, rec.brush.color[3], crate::path::FLATTEN_STEP, 0.0];
        // Reservoir textures sized to the padded width × bands, with COPY_SRC for readback.
        let make = |label| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width, height: LATERAL_BANDS, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let color = make("stark dbg reservoir");
        let aux = make("stark dbg aux reservoir");
        let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
        let aux_view = aux.create_view(&wgpu::TextureViewDescriptor::default());

        let mut scoped = ScopedResources::default();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stark dbg reservoir"),
        });
        self.encode_mixer(
            &mut scoped, &mut encoder, base, &coords, &segments, &instance_buf, &color_view,
            &aux_view, bbox, (lead, width), channels, d.add, knobs,
        );
        self.ctx.queue.submit([encoder.finish()]);

        let size = crate::geom::Extent2 { width, height: LATERAL_BANDS };
        let color_data = crate::gpu::readback::read_rgba16f(&self.ctx, &color, size);
        let aux_data = crate::gpu::readback::read_rgba16f(&self.ctx, &aux, size);
        Some((color_data, aux_data, width, LATERAL_BANDS))
    }

    /// Composite the given tiles' channels into a fresh 1:1 region (color, aux),
    /// reusing the region pipeline. `tiles` = (canvas origin, color view, aux view);
    /// `origin`/`w`/`h` define the region rect in canvas px. With `wide_aux` the aux
    /// region is the wide [`SCRATCH_AUX_FORMAT`] (for *scratch* tiles, whose extra
    /// channels — e.g. the footprint coverage in `.z` — must survive the composite);
    /// otherwise the colour-space's compact aux. The region textures also carry
    /// `COPY_SRC` so they can be sliced back into tiles.
    #[allow(clippy::too_many_arguments)]
    fn composite_region(
        &self,
        scoped: &mut ScopedResources,
        encoder: &mut wgpu::CommandEncoder,
        tiles: &[(Vec2, wgpu::TextureView, wgpu::TextureView)],
        origin: Vec2,
        w: u32,
        h: u32,
        wide_aux: bool,
    ) -> (wgpu::Texture, wgpu::Texture) {
        let (aux_format, pipeline) = if wide_aux {
            (SCRATCH_AUX_FORMAT, &self.composite_wide_pipeline)
        } else {
            (self.color_space.aux_format(), &self.composite_pipeline)
        };
        let device = &self.ctx.device;
        let extent = wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 };
        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC;
        let mut make = move |format, label| {
            scoped.texture(device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            }))
        };
        let color_tex = make(self.color_space.color_format(), "stark region color");
        let aux_tex = make(aux_format, "stark region aux");
        let color_view = color_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let aux_view = aux_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let (sx, sy) = (2.0 / w as f32, -2.0 / h as f32);
        let view = ViewUniform {
            st: [sx, sy, -origin.x * sx - 1.0, -origin.y * sy + 1.0],
            misc: [TILE_SIZE as f32, INTERIOR_UV_SCALE, INTERIOR_UV_BIAS, 0.0],
        };
        let view_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stark region view"),
            contents: bytemuck::bytes_of(&view),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let view_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark region view bg"),
            layout: &self.composite_view_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: view_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
            ],
        });
        let mut inst: Vec<TileInstance> = Vec::new();
        let mut bgs: Vec<wgpu::BindGroup> = Vec::new();
        for (o, cv, av) in tiles {
            inst.push(TileInstance { origin: o.to_array(), opacity: 1.0 });
            bgs.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("stark region tile bg"),
                layout: &self.composite_tile_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(cv) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(av) },
                ],
            }));
        }
        let inst_buf = (!inst.is_empty()).then(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stark region instances"),
                contents: bytemuck::cast_slice(&inst),
                usage: wgpu::BufferUsages::VERTEX,
            })
        });
        {
            let clear = wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stark region composite"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment { view: &color_view, resolve_target: None, depth_slice: None, ops: clear }),
                    Some(wgpu::RenderPassColorAttachment { view: &aux_view, resolve_target: None, depth_slice: None, ops: clear }),
                ],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if let Some(ib) = &inst_buf {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &view_bg, &[]);
                pass.set_vertex_buffer(0, ib.slice(..));
                for (i, bg) in bgs.iter().enumerate() {
                    let idx = i as u32;
                    pass.set_bind_group(1, bg, &[]);
                    pass.draw(0..4, idx..idx + 1);
                }
            }
        }
        (color_tex, aux_tex)
    }

    /// Wet flow over the stroke's region (DESIGN.md §6.2): composite the freshly-
    /// deposited tiles (+ a halo) into a region, **drag** the paint along an injected
    /// velocity field (semi-Lagrangian advection, `drag`) and **bleed** it with a
    /// separable Gaussian (`bleed`), then slice the region back into fresh CoW tiles.
    /// Either effect is skipped when its param is 0. Both are localized to the stroke
    /// footprint (identity in the halo) → seam-free. Pure function of the deposited
    /// tiles + path → replay-deterministic; velocity is transient (never stored).
    #[allow(clippy::too_many_arguments)]
    fn encode_flow(
        &self,
        scoped: &mut ScopedResources,
        encoder: &mut wgpu::CommandEncoder,
        pool: &TilePool,
        map: &mut HashTrieMap<TileCoord, TilePairHandle>,
        coords: &BTreeSet<TileCoord>,
        scratch: &[(TileCoord, TilePairHandle)],
        instance_buf: &wgpu::Buffer,
        instances: u32,
        bleed: f32,
        drag: f32,
    ) {
        if bleed <= 0.0 && drag <= 0.0 {
            return; // nothing to flow — the plain deposit stands
        }
        // Haloed region so rewritten tiles' aprons read real neighbour interiors →
        // seam-free; composite the halo but write back only the affected tiles.
        let Some((halo, lo, region_origin, w, h)) = region_rect(coords) else {
            return; // empty or larger than MAX_REGION_DIM — leave the plain deposit
        };
        let device = &self.ctx.device;
        let (a_color, a_aux) = self.composite_halo(scoped, encoder, map, &halo, region_origin, w, h);
        // B is the advection ping-pong scratch and the bleed's horizontal-blur target.
        let b_color = self.region_tex(scoped, w, h, self.color_space.color_format(), "stark flow color b");
        let b_aux = self.region_tex(scoped, w, h, self.color_space.aux_format(), "stark flow aux b");
        let av_c = a_color.create_view(&wgpu::TextureViewDescriptor::default());
        let av_a = a_aux.create_view(&wgpu::TextureViewDescriptor::default());
        let bv_c = b_color.create_view(&wgpu::TextureViewDescriptor::default());
        let bv_a = b_aux.create_view(&wgpu::TextureViewDescriptor::default());
        let store = wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            store: wgpu::StoreOp::Store,
        };

        // Footprint coverage region — the brush's **shaped** Σ√cov (wide scratch aux `.z`,
        // bristle streaks / stamp mask and all), composited from the stroke's scratch
        // footprints. Shared by the drag (which gates its velocity by this, so the rake
        // follows the brush shape instead of a round capsule) and the bleed (mix mask). It
        // is 0 outside the footprint / in the halo, so gating leaves the halo untouched →
        // the write-back aprons stay seam-free.
        let foot: Vec<_> = scratch
            .iter()
            .map(|(c, t)| (c.origin(), t.color_view().clone(), t.aux_view().clone()))
            .collect();
        let (_rate_color, rate_aux) =
            self.composite_region(scoped, encoder, &foot, region_origin, w, h, true);
        let rate_view = rate_aux.create_view(&wgpu::TextureViewDescriptor::default());

        // --- Drag: inject the stroke's velocity (dir · drag), gate it by the footprint
        //     coverage (shape), then conservatively advect FLOW_ITERS times, ping-ponging
        //     A↔B (even count → result back in A).
        if drag > 0.0 {
            let vel_tex = self.region_tex(scoped, w, h, self.color_space.aux_format(), "stark flow velocity");
            let vel_view = vel_tex.create_view(&wgpu::TextureViewDescriptor::default());
            let (sx, sy) = (2.0 / w as f32, -2.0 / h as f32);
            let inject = InjectUniform {
                st: [sx, sy, -region_origin.x * sx - 1.0, -region_origin.y * sy + 1.0],
                knobs: [drag * FLOW_DRAG_PX, 0.0, 0.0, 0.0],
            };
            let inject_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stark flow inject params"),
                contents: bytemuck::bytes_of(&inject),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let inject_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("stark flow inject bg"),
                layout: &self.fluid_inject_bgl,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: inject_buf.as_entire_binding() }],
            });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("stark flow inject"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &vel_view, resolve_target: None, depth_slice: None, ops: store,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&self.fluid_inject_pipeline);
                pass.set_bind_group(0, &inject_bg, &[]);
                pass.set_vertex_buffer(0, instance_buf.slice(..));
                pass.draw(0..4, 0..instances);
            }

            // The injected velocity already ramps to 0 at the footprint edge (the inject's
            // capsule falloff) and is 0 in the halo, so a zero-velocity cell's finite-volume
            // update is exact identity — the advection leaves the halo untouched and the
            // region write-back stays seam-free. (No velocity smoothing: it spread nonzero
            // velocity into the halo, breaking that property; the conservative scheme's
            // crispness is governed by its CFL limit instead — see fluid_advect.wesl.)
            let advect_uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stark flow advect params"),
                contents: bytemuck::bytes_of(&FluidUniform { knobs: [1.0, 0.0, 0.0, 0.0] }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            for i in 0..FLOW_ITERS {
                let (sc, sa, dc, da) = if i % 2 == 0 {
                    (&av_c, &av_a, &bv_c, &bv_a)
                } else {
                    (&bv_c, &bv_a, &av_c, &av_a)
                };
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("stark flow advect bg"),
                    layout: &self.fluid_advect_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: advect_uni.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&vel_view) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(sc) },
                        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(sa) },
                        wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                        wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&rate_view) },
                    ],
                });
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("stark flow advect"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment { view: dc, resolve_target: None, depth_slice: None, ops: store }),
                        Some(wgpu::RenderPassColorAttachment { view: da, resolve_target: None, depth_slice: None, ops: store }),
                    ],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&self.fluid_advect_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.draw(0..3, 0..1);
            }
        }

        // --- Bleed: a separable Gaussian (σ ∝ bleed) gated by the footprint rate.
        //     H pass blurs A → B (pure); V pass blurs B → C and mixes the full Gaussian
        //     back over the original A by rate·wet (identity where rate is 0). Result in
        //     C; with no bleed the (advected) A is written back directly.
        let bleed_result = if bleed > 0.0 {
            // The bleed's mask is the shared footprint coverage region (`rate_view`, wide
            // scratch aux `.z`) — laid regardless of the deposit, so an `add = 0` blender
            // still bleeds where it bore down.
            let c_color = self.region_tex(scoped, w, h, self.color_space.color_format(), "stark flow color c");
            let c_aux = self.region_tex(scoped, w, h, self.color_space.aux_format(), "stark flow aux c");
            let cv_c = c_color.create_view(&wgpu::TextureViewDescriptor::default());
            let cv_a = c_aux.create_view(&wgpu::TextureViewDescriptor::default());
            let sigma = bleed * MAX_SIGMA;

            // The two blur passes share a bind-group shape; only the uniform + targets
            // differ. `orig`/`rate` are bound for both but only used by the V pass.
            let blur_pass = |encoder: &mut wgpu::CommandEncoder,
                             uni: &wgpu::Buffer,
                             src_c: &wgpu::TextureView,
                             src_a: &wgpu::TextureView,
                             dst_c: &wgpu::TextureView,
                             dst_a: &wgpu::TextureView| {
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("stark flow blur bg"),
                    layout: &self.blur_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: uni.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(src_c) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(src_a) },
                        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&av_c) },
                        wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&av_a) },
                        wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&rate_view) },
                    ],
                });
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("stark flow blur"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment { view: dst_c, resolve_target: None, depth_slice: None, ops: store }),
                        Some(wgpu::RenderPassColorAttachment { view: dst_a, resolve_target: None, depth_slice: None, ops: store }),
                    ],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&self.blur_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.draw(0..3, 0..1);
            };

            let h_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stark flow blur-h params"),
                contents: bytemuck::bytes_of(&BlurUniform { knobs: [1.0, 0.0, sigma, 0.0] }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let v_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stark flow blur-v params"),
                contents: bytemuck::bytes_of(&BlurUniform { knobs: [0.0, 1.0, sigma, 1.0] }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            blur_pass(encoder, &h_buf, &av_c, &av_a, &bv_c, &bv_a); // horizontal: A → B
            blur_pass(encoder, &v_buf, &bv_c, &bv_a, &cv_c, &cv_a); // vertical + combine → C
            Some((c_color, c_aux))
        } else {
            None
        };

        let (rc, ra) = match &bleed_result {
            Some((c, a)) => (c, a),
            None => (&a_color, &a_aux),
        };
        self.writeback_region(encoder, pool, map, coords, rc, ra, lo);
    }

    /// A transient region texture (`RENDER_ATTACHMENT | TEXTURE_BINDING | COPY_SRC`).
    fn region_tex(
        &self,
        scoped: &mut ScopedResources,
        w: u32,
        h: u32,
        format: wgpu::TextureFormat,
        label: &str,
    ) -> wgpu::Texture {
        scoped.texture(self.ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        }))
    }

    /// Composite the post-deposit tiles for `halo` into a fresh region (color, aux).
    #[allow(clippy::too_many_arguments)]
    fn composite_halo(
        &self,
        scoped: &mut ScopedResources,
        encoder: &mut wgpu::CommandEncoder,
        map: &HashTrieMap<TileCoord, TilePairHandle>,
        halo: &[TileCoord],
        region_origin: Vec2,
        w: u32,
        h: u32,
    ) -> (wgpu::Texture, wgpu::Texture) {
        let post: Vec<_> = halo
            .iter()
            .filter_map(|c| map.get(c).map(|t| (c.origin(), t.color_view().clone(), t.aux_view().clone())))
            .collect();
        self.composite_region(scoped, encoder, &post, region_origin, w, h, false)
    }

    /// Slice a processed region back into fresh CoW tiles for the affected `coords`.
    /// Each tile's `TILE_TEX` block starts at `origin − lo` in the region; copying the
    /// full block (interior + apron) from the shared region keeps aprons bit-identical
    /// to neighbours' interiors → seamless (§6.4).
    #[allow(clippy::too_many_arguments)]
    fn writeback_region(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &TilePool,
        map: &mut HashTrieMap<TileCoord, TilePairHandle>,
        coords: &BTreeSet<TileCoord>,
        color_tex: &wgpu::Texture,
        aux_tex: &wgpu::Texture,
        lo: Vec2,
    ) {
        let block = wgpu::Extent3d { width: TILE_TEX, height: TILE_TEX, depth_or_array_layers: 1 };
        for c in coords {
            let dst = pool.acquire(AllocSource::FlowWritebackRegion);
            let src_origin = wgpu::Origin3d {
                x: (c.origin().x - lo.x) as u32,
                y: (c.origin().y - lo.y) as u32,
                z: 0,
            };
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo { texture: color_tex, mip_level: 0, origin: src_origin, aspect: wgpu::TextureAspect::All },
                dst.color().as_image_copy(),
                block,
            );
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo { texture: aux_tex, mip_level: 0, origin: src_origin, aspect: wgpu::TextureAspect::All },
                dst.aux().as_image_copy(),
                block,
            );
            *map = map.insert(*c, dst);
        }
    }

    /// The round tip's prefix-τ texture for a given `hardness`, cached so live
    /// preview (which re-renders per pointer move) doesn't rebuild it each frame.
    fn round_prefix(&self, hardness: f32) -> wgpu::TextureView {
        let key = hardness.to_bits();
        let mut cache = self.round_prefix.lock().expect("round prefix poisoned");
        if let Some((k, view)) = cache.as_ref()
            && *k == key
        {
            return view.clone();
        }
        // The round tip is rotation-invariant, so a single orientation layer suffices —
        // the shader's wrapping lookup reads it for every orientation (DESIGN.md §6.6).
        let coverage = round_coverage(hardness, ROUND_RES);
        let (_tex, view) = build_prefix_tau(&self.ctx, ROUND_RES, ROUND_RES, 1, &coverage);
        *cache = Some((key, view.clone()));
        view
    }

    /// A 1×1 reservoir holding the brush's own color premultiplied by `add` (so the
    /// deposit lays `add·brush`), plus a 1×1 aux cleared to 0 (no tool deposit, no lift),
    /// for brushes without a scan. Cleared (not CPU-uploaded) so the driver does the f16
    /// encode; the deposit samples one uniform color across the whole tip.
    fn flat_reservoir(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        channels: [f32; 4],
        add: f32,
    ) -> (wgpu::TextureView, wgpu::TextureView) {
        let make = |label| {
            self.ctx
                .device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let color = make("stark flat reservoir");
        let aux = make("stark flat aux reservoir");
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("stark flat reservoir clear"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &color,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        // Premultiplied by `add`: colour = channels·add, opacity = add, so
                        // the deposit lays `add·brush` (the stamp scales by op = in.op·√cov).
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: (channels[0] * add) as f64,
                            g: (channels[1] * add) as f64,
                            b: (channels[2] * add) as f64,
                            a: add as f64,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &aux,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        (color, aux)
    }

    /// Wet-mixing pickup (DESIGN.md §6.2), fully on the GPU. Pass A composites the
    /// base under the stroke into a 1:1 region; Pass B is a serial compute scan
    /// (one invocation per lateral band) that lifts wet paint into per-band
    /// reservoirs and writes the `reservoir` texture (column × band). No CPU
    /// readback — works on WebGPU. `bbox` is the stroke's region (origin, w, h);
    /// `pad` is `(lead, width)`: the lead-in column count and total texture width
    /// (lead-in + one per segment + lead-out, see `render`).
    #[allow(clippy::too_many_arguments)]
    fn encode_mixer(
        &self,
        scoped: &mut ScopedResources,
        encoder: &mut wgpu::CommandEncoder,
        base: &HashTrieMap<TileCoord, TilePairHandle>,
        coords: &BTreeSet<TileCoord>,
        segments: &[Segment],
        instance_buf: &wgpu::Buffer,
        reservoir: &wgpu::TextureView,
        aux_reservoir: &wgpu::TextureView,
        bbox: (Vec2, u32, u32),
        pad: (u32, u32),
        channels: [f32; 4],
        add: f32,
        knobs: [f32; 4],
    ) {
        let (origin, w, h) = bbox;
        let (lead, width) = pad;
        let device = &self.ctx.device;

        // Region targets: the base canvas under the stroke, 1:1 with canvas px.
        let extent = wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        };
        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let mut make_tex = move |format, label| {
            scoped.texture(device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            }))
            .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let region_color = make_tex(self.color_space.color_format(), "stark mixer region color");
        let region_aux = make_tex(self.color_space.aux_format(), "stark mixer region aux");

        // ---- Pass A: composite the base tiles under the stroke into the region.
        let (sx, sy) = (2.0 / w as f32, -2.0 / h as f32);
        let view = ViewUniform {
            st: [sx, sy, -origin.x * sx - 1.0, -origin.y * sy + 1.0],
            misc: [TILE_SIZE as f32, INTERIOR_UV_SCALE, INTERIOR_UV_BIAS, 0.0],
        };
        let view_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stark mixer view"),
            contents: bytemuck::bytes_of(&view),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let view_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark mixer view bg"),
            layout: &self.composite_view_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: view_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
                },
            ],
        });

        let mut tile_origins: Vec<TileInstance> = Vec::new();
        let mut tile_bgs: Vec<wgpu::BindGroup> = Vec::new();
        for coord in coords {
            if let Some(tile) = base.get(coord) {
                tile_origins.push(TileInstance {
                    origin: coord.origin().to_array(),
                    opacity: 1.0,
                });
                tile_bgs.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("stark mixer tile bg"),
                    layout: &self.composite_tile_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(tile.color_view()),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(tile.aux_view()),
                        },
                    ],
                }));
            }
        }
        // Created before the pass so it outlives the borrow; only used if non-empty.
        let tile_inst = (!tile_origins.is_empty()).then(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stark mixer tile instances"),
                contents: bytemuck::cast_slice(&tile_origins),
                usage: wgpu::BufferUsages::VERTEX,
            })
        });
        {
            let clear = wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stark mixer region composite"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &region_color,
                        resolve_target: None,
                        depth_slice: None,
                        ops: clear,
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &region_aux,
                        resolve_target: None,
                        depth_slice: None,
                        ops: clear,
                    }),
                ],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            // Empty region (no base tiles) just stays cleared → "no paint".
            if let Some(inst) = &tile_inst {
                pass.set_pipeline(&self.composite_pipeline);
                pass.set_bind_group(0, &view_bg, &[]);
                pass.set_vertex_buffer(0, inst.slice(..));
                for (i, bg) in tile_bgs.iter().enumerate() {
                    let idx = i as u32;
                    pass.set_bind_group(1, bg, &[]);
                    pass.draw(0..4, idx..idx + 1);
                }
            }
        }

        // ---- Pass B: the serial reservoir scan, writing the reservoir texture.
        let uni = MixerUniform {
            // The brush's own colour (.xyz) and the `add` axis (.w) the walk folds in.
            brush_ch: [channels[0], channels[1], channels[2], add],
            origin_dims: [origin.x, origin.y, w as f32, h as f32],
            knobs,
            counts: [segments.len() as u32, lead, width, 0],
        };
        let uni_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stark mixer params"),
            contents: bytemuck::bytes_of(&uni),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let mixer_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark mixer bg"),
            layout: &self.mixer_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uni_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&region_color),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&region_aux),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: instance_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(reservoir),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(aux_reservoir),
                },
            ],
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stark mixer scan"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.mixer_pipeline);
            cpass.set_bind_group(0, &mixer_bg, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }
    }
}

/// Generate the round tip's coverage: a soft disc with `hardness` falloff.
fn round_coverage(hardness: f32, res: u32) -> Vec<f32> {
    let h = hardness.clamp(0.0, 0.99);
    let mut cov = vec![0.0f32; (res * res) as usize];
    for y in 0..res {
        for x in 0..res {
            let fx = (x as f32 + 0.5) / res as f32 * 2.0 - 1.0;
            let fy = (y as f32 + 0.5) / res as f32 * 2.0 - 1.0;
            let r = (fx * fx + fy * fy).sqrt();
            cov[(y * res + x) as usize] = 1.0 - smoothstep(h, 1.0, r);
        }
    }
    cov
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Build swept segments from the fitted control points (DESIGN.md §6.2): flatten
/// the spline, then make each polyline edge a segment. The one-way load reservoir
/// (`drain`) depletes with arc distance; radius follows pressure. The deposited
/// color comes from the reservoir texture (the brush's own color, or per-segment ×
/// per-band smear when the tool lifts/deposits — see `encode_mixer`).
fn generate_segments(rec: &StrokeRecord) -> Vec<Segment> {
    let b = &rec.brush;
    let pts = crate::path::flatten(&rec.path, crate::path::FLATTEN_STEP);
    let mut segs = Vec::new();
    if pts.is_empty() {
        return segs;
    }

    // `dist` (arc length from the stroke start) drives only the drain here; the
    // reservoir is parameterized by segment *index*, not distance (see `render`).
    let dyn_ = b.dynamics;
    let make = |sample: &InputSample, dir: Vec2, len: f32, dist: f32| -> Segment {
        let drain = (1.0 - b.drain * dist).max(0.0);
        // Palette-knife dynamics (DESIGN.md §6.2): pen **pressure** modulates the scrape
        // (`load`) and pen **tilt toward the direction of motion** modulates the `deposit`.
        // Each `*_response` knob in [0,1] cross-fades from the constant axis value (response
        // = 0, the historical behaviour and the no-pen fallback) to fully input-driven
        // (response = 1). `lerp(a, b, t)` here is `a + (b − a)·t`.
        let press = sample.pressure.clamp(0.0, 1.0);
        let load = dyn_.load * (1.0 - dyn_.load_pressure + dyn_.load_pressure * press);
        // Deposit, modulated by pen tilt **relative to the fallback** so the response is
        // continuous through vertical (DESIGN.md §6.2). `forward` is the signed lean along the
        // travel direction — the **un-normalized** tilt projected onto the unit `dir`, so a
        // bigger tilt leans harder: > 0 leans into the motion, < 0 leans back, and **0 = an
        // upright pen OR a mouse**. The deposit is the constant fallback `dyn_.deposit` at
        // `forward = 0`, scaled up toward 2× as the pen leans forward and down toward 0 as it
        // leans back — a smooth swing about the fallback (`deposit_tilt` sets its size; 0 = no
        // tilt response). No magnitude threshold, so passing through vertical never jumps.
        let forward = sample.tilt.dot(dir).clamp(-1.0, 1.0);
        let deposit = dyn_.deposit * (1.0 + dyn_.deposit_tilt * forward);
        Segment {
            start: sample.pos,
            dir,
            radius: (b.radius * sample.pressure).max(0.5),
            length: len,
            // `flow` now drives only the footprint build-up; the brush's opacity
            // (color[3]) rides the separate opacity channel (DESIGN.md §6.1).
            flow: b.flow * drain * SWEEP_FLOW_SCALE,
            height: b.height * drain,
            wet: b.wetness * drain,
            opacity: b.color[3] * drain,
            load,
            deposit,
            orient: orientation_turns(b.orientation, dir, sample.tilt),
        }
    };

    let mut dist = 0.0f32;
    for w in pts.windows(2) {
        let (a, c) = (&w[0], &w[1]);
        let v = c.pos - a.pos;
        let len = v.length();
        if len < 1e-5 {
            continue;
        }
        segs.push(make(a, v / len, len, dist));
        dist += len;
    }

    if segs.is_empty() {
        // A click: sweep a fraction of a radius so it deposits a soft blob.
        let p = &pts[0];
        let r = (b.radius * p.pressure).max(0.5);
        segs.push(make(p, Vec2::new(1.0, 0.0), r * 0.6, 0.0));
    }
    segs
}

/// The shape's orientation for a segment, as a fraction of a full turn ∈ [0, 1): the
/// relative angle between the shape's native axis and the travel direction `dir`, which
/// picks the prefix-τ orientation layer (DESIGN.md §6.6).
///
/// - [`OrientationSource::FollowStroke`]: the shape tracks the tangent, so the relative
///   angle is always 0 (the historical behaviour; for a round tip it is moot anyway).
/// - [`OrientationSource::Pen`]: the shape is pinned to the pen's azimuth (the tilt
///   direction) in canvas space, so relative to the travel direction it is `α − φ` — as
///   the stroke curves the footprint angle stays fixed in the world, like a nib.
fn orientation_turns(source: OrientationSource, dir: Vec2, tilt: Vec2) -> f32 {
    match source {
        OrientationSource::FollowStroke => 0.0,
        OrientationSource::Pen => {
            let alpha = tilt.y.atan2(tilt.x); // pen azimuth (0 when the pen is upright / mouse)
            let phi = dir.y.atan2(dir.x); // travel direction
            ((alpha - phi) / std::f32::consts::TAU).rem_euclid(1.0)
        }
    }
}

/// Integer-pixel bounding box of a stroke (segments ± radius), or `None` if the
/// stroke is empty or larger than [`MAX_REGION_DIM`] in either axis (wet-mixing
/// pickup is skipped for it). Returns `(canvas origin, width, height)`.
fn stroke_bbox(segments: &[Segment]) -> Option<(Vec2, u32, u32)> {
    let mut lo = Vec2::splat(f32::INFINITY);
    let mut hi = Vec2::splat(f32::NEG_INFINITY);
    for s in segments {
        let end = s.start + s.dir * s.length;
        let r = Vec2::splat(s.radius);
        lo = lo.min(s.start.min(end) - r);
        hi = hi.max(s.start.max(end) + r);
    }
    if !lo.x.is_finite() {
        return None;
    }
    let origin = Vec2::new(lo.x.floor(), lo.y.floor());
    let w = (hi.x.ceil() - origin.x).max(1.0) as u32;
    let h = (hi.y.ceil() - origin.y).max(1.0) as u32;
    if w > MAX_REGION_DIM || h > MAX_REGION_DIM {
        return None;
    }
    Some((origin, w, h))
}

/// Tiles whose *texture* (interior + apron) any segment's swept capsule overlaps.
/// The apron is included in `reach` so a stroke landing within a tile's interior
/// but inside a neighbor's apron band re-renders that neighbor too, keeping the
/// shared apron/interior overlap bit-identical (DESIGN.md §6.4).
/// The haloed, tile-aligned region for a stroke's affected `coords`: those tiles plus
/// a one-tile ring, so a rewritten tile's apron reads its neighbour's real interior
/// from the region (the seam fix — region write-backs overwrite the whole `TILE_TEX`
/// block). Returns `(halo tiles, lo origin, region_origin, w, h)`, or `None` if empty
/// or larger than `MAX_REGION_DIM`. Shared by the Wet and Fluid region passes.
fn region_rect(coords: &BTreeSet<TileCoord>) -> Option<(Vec<TileCoord>, Vec2, Vec2, u32, u32)> {
    let mut halo: BTreeSet<TileCoord> = BTreeSet::new();
    for c in coords {
        for dy in -1..=1 {
            for dx in -1..=1 {
                halo.insert(TileCoord::new(c.x + dx, c.y + dy));
            }
        }
    }
    let mut lo = Vec2::splat(f32::INFINITY);
    let mut hi = Vec2::splat(f32::NEG_INFINITY);
    for c in &halo {
        lo = lo.min(c.origin());
        hi = hi.max(c.origin());
    }
    if !lo.x.is_finite() {
        return None;
    }
    let region_origin = lo - Vec2::splat(TILE_APRON as f32);
    let w = (hi.x - lo.x) as u32 + TILE_TEX;
    let h = (hi.y - lo.y) as u32 + TILE_TEX;
    if w > MAX_REGION_DIM || h > MAX_REGION_DIM {
        return None;
    }
    Some((halo.into_iter().collect(), lo, region_origin, w, h))
}

fn affected_tiles(segments: &[Segment]) -> BTreeSet<TileCoord> {
    let tile = TILE_SIZE as f32;
    let mut coords = BTreeSet::new();
    for s in segments {
        let end = s.start + s.dir * s.length;
        let reach = Vec2::splat(s.radius + TILE_APRON as f32);
        let lo = s.start.min(end) - reach;
        let hi = s.start.max(end) + reach;
        let (x0, x1) = ((lo.x / tile).floor() as i32, (hi.x / tile).floor() as i32);
        let (y0, y1) = ((lo.y / tile).floor() as i32, (hi.y / tile).floor() as i32);
        for y in y0..=y1 {
            for x in x0..=x1 {
                coords.insert(TileCoord::new(x, y));
            }
        }
    }
    coords
}

/// Build the base-region composite pipeline for wet mixing — a 1:1 re-composite
/// of the active layer under the stroke, reusing the `composite` shader (DESIGN
/// §6.2, §6.3). `aux_format` is the aux target: the colour-space's compact format
/// for canvas tiles, or the wide [`SCRATCH_AUX_FORMAT`] when compositing scratch
/// tiles whose extra channels must survive. Returns `(pipeline, view bgl, tile bgl)`.
fn build_composite_pipeline(
    device: &wgpu::Device,
    color_space: &dyn ColorSpace,
    aux_format: wgpu::TextureFormat,
) -> (
    wgpu::RenderPipeline,
    wgpu::BindGroupLayout,
    wgpu::BindGroupLayout,
) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("stark mixer composite"),
        source: wgpu::ShaderSource::Wgsl(stark_shaders::composite().into()),
    });
    let view_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("stark mixer composite view bgl"),
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
    let tex = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let tile_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("stark mixer composite tile bgl"),
        entries: &[tex(0), tex(1)],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("stark mixer composite layout"),
        bind_group_layouts: &[Some(&view_bgl), Some(&tile_bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("stark mixer composite pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<TileInstance>() as u64,
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
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: color_space.color_format(),
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
    (pipeline, view_bgl, tile_bgl)
}

/// Build the wet-mixing reservoir compute pipeline (`mixer` shader) — DESIGN §6.2.
fn build_mixer_pipeline(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("stark mixer"),
        source: wgpu::ShaderSource::Wgsl(stark_shaders::mixer().into()),
    });
    let region_tex = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            // Sampled via textureLoad only, so filtering isn't required.
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("stark mixer bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            region_tex(1),
            region_tex(2),
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    // Read-only: both modes drive off the instances and write only the
                    // reservoir texture (binding 4); the knife's per-band film amount
                    // rides the reservoir alpha.
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            // The companion height reservoir (carried paint-height rate along the stroke).
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("stark mixer layout"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("stark mixer pipeline"),
        layout: Some(&layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    (pipeline, bgl)
}

/// Build the stroke integrate pipeline (`integrate` shader) — DESIGN §6.2/§6.1. A
/// fullscreen pass with a `Params` uniform + four sampled tiles (base/scratch
/// color/aux), writing the color+aux MRT of a fresh tile.
fn build_integrate_pipeline(
    device: &wgpu::Device,
    color_space: &dyn ColorSpace,
) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("stark integrate"),
        source: wgpu::ShaderSource::Wgsl(stark_shaders::integrate().into()),
    });
    let load_tex = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            // Sampled via textureLoad only (1:1 with the destination).
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("stark integrate bgl"),
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
            load_tex(1), // base color
            load_tex(2), // base aux
            load_tex(3), // scratch color
            load_tex(4), // scratch aux
            load_tex(5), // per-pixel scrape load (or a 1×1 dummy on the non-scrape path)
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("stark integrate layout"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("stark integrate pipeline"),
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
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: color_space.color_format(),
                    blend: None, // the shader does the combine; write straight through
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: color_space.aux_format(),
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
        }),
        multiview_mask: None,
        cache: None,
    });
    (pipeline, bgl)
}

/// Build the Wet-brush bleed pipeline (`blur` shader) — DESIGN §6.2. A fullscreen
/// separable-Gaussian pass with a `Params` uniform + five sampled region textures
/// (src color/aux to blur, original color/aux + footprint rate for the combine),
/// writing the color+aux MRT of the ping-pong target.
fn build_blur_pipeline(
    device: &wgpu::Device,
    color_space: &dyn ColorSpace,
) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("stark blur"),
        source: wgpu::ShaderSource::Wgsl(stark_shaders::blur().into()),
    });
    let load_tex = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("stark blur bgl"),
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
            load_tex(1), // src color (to blur)
            load_tex(2), // src aux
            load_tex(3), // original color (for the combine)
            load_tex(4), // original aux
            load_tex(5), // footprint rate
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("stark blur layout"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("stark blur pipeline"),
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
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: color_space.color_format(),
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: color_space.aux_format(),
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
        }),
        multiview_mask: None,
        cache: None,
    });
    (pipeline, bgl)
}

/// Build the fluid velocity-injection pipeline (`fluid_inject` shader) — DESIGN §6.2.
/// Rasterizes the stroke's segments (oriented quads, reusing the `SegmentInstance`
/// buffer's start/dir/geom) into the velocity region with additive blending.
fn build_fluid_inject_pipeline(
    device: &wgpu::Device,
    color_space: &dyn ColorSpace,
) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("stark fluid inject"),
        source: wgpu::ShaderSource::Wgsl(stark_shaders::fluid_inject().into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("stark fluid inject bgl"),
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
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("stark fluid inject layout"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    // Additive blend so overlapping segments accumulate velocity.
    let additive = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    };
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("stark fluid inject pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<SegmentInstance>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                // Reads start, dir, geom from the SegmentInstance buffer (aux unused).
                attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x4],
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
                format: color_space.aux_format(), // Rg16Float velocity (vx, vy)
                blend: Some(wgpu::BlendState { color: additive, alpha: additive }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });
    (pipeline, bgl)
}

/// Build the fluid advection pipeline (`fluid_advect` shader) — DESIGN §6.2. A
/// fullscreen pass that bilinearly back-traces along the velocity field and samples
/// the paint (color, aux) it came from, writing the ping-pong target's MRT.
fn build_fluid_advect_pipeline(
    device: &wgpu::Device,
    color_space: &dyn ColorSpace,
) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("stark fluid advect"),
        source: wgpu::ShaderSource::Wgsl(stark_shaders::fluid_advect().into()),
    });
    // 16-bit-float region textures are filterable, so the back-trace can bilerp them.
    let filter_tex = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("stark fluid advect bgl"),
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
            filter_tex(1), // velocity
            filter_tex(2), // src color
            filter_tex(3), // src aux
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            filter_tex(5), // footprint coverage (shape gate for the velocity)
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("stark fluid advect layout"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("stark fluid advect pipeline"),
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
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: color_space.color_format(),
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: color_space.aux_format(),
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
        }),
        multiview_mask: None,
        cache: None,
    });
    (pipeline, bgl)
}

