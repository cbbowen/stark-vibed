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
use crate::document::{BrushDynamics, BrushShape, StrokeRecord};
use crate::geom::{
    TileCoord, Vec2, INTERIOR_UV_BIAS, INTERIOR_UV_SCALE, TILE_APRON, TILE_SIZE, TILE_TEX,
};
use crate::gpu::surface::{Surface, SURFACE_TILE_PX};
use crate::gpu::context::GpuContext;
use crate::gpu::tile::{TileHandle, TilePool};

/// Global tuning so a default brush (`flow = 1`) reads as a solid stroke;
/// `flow` is an optical-depth-per-length rate (DESIGN.md §6.2).
const SWEEP_FLOW_SCALE: f32 = 1.0;
/// Resolution of the generated round-tip prefix texture.
const ROUND_RES: u32 = 256;

/// Cap on the wet-mixing reservoir load so pickup saturates (mirrors `capacity`
/// in `mixer.wesl`).
const RESERVOIR_CAPACITY: f32 = 1.0;

/// Lateral reservoir bands across the brush tip: each picks up canvas color at its
/// own offset, so one side of the brush can carry a different color than the other
/// (DESIGN.md §6.2). This is the *height* of the reservoir texture; the mixer's
/// `BANDS` constant must match. The stamp shader samples the texture and is
/// otherwise agnostic to this count.
const LATERAL_BANDS: u32 = 16;

/// Largest base-region edge (canvas px) composited for wet-mixing pickup. Strokes
/// whose bounding box exceeds this skip pickup for that stroke — rare, and it
/// bounds the transient GPU memory of the region texture (DESIGN.md §6.2).
const MAX_REGION_DIM: u32 = 2048;

/// Fixed flow iterations per stroke for the Wet brush — each runs one advect + one
/// diffuse pass (DESIGN.md §6.2). Constant so replay is deterministic; `WetParams`
/// `bleed`/`drag` scale the per-iteration diffusion rate / injected velocity. Even so
/// the advect-then-diffuse ping-pong lands the result back in region A. `FLOW_DRAG_PX`
/// is the per-iteration drag distance (canvas px) at full `drag`.
const FLOW_ITERS: u32 = 12;
const FLOW_DRAG_PX: f32 = 2.5;

/// Standard deviation (canvas px) of the Wet brush's separable-Gaussian **bleed** at
/// full `WetParams::bleed`. The bleed runs in 2 passes (vs the old explicit diffusion's
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
    /// Longitudinal offset: arc length from the stroke start to this segment's
    /// start. Parameterizes the reservoir by distance (DESIGN.md §6.2).
    dist: f32,
}

/// Per-segment instance data for the sweep shader. Padded to 48 bytes so the same
/// buffer is a valid `std430 array<Instance>` for the wet-mixing compute pass,
/// which reads it to drive the reservoir scan (see `mixer.wesl`).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SegmentInstance {
    start: [f32; 2],
    dir: [f32; 2],   // unit tangent
    geom: [f32; 4],  // radius, length, flow, reservoir column u ∈ [0,1]
    aux: [f32; 4],   // height (thickness rate), wet, opacity, _ (→ 48 B std430)
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
    brush_ch: [f32; 4],
    origin_dims: [f32; 4], // region origin.xy (canvas px), dims.xy (px)
    knobs: [f32; 4],       // smear, add, flatten_step, reservoir capacity
    counts: [u32; 4],      // segment_count, lead-in columns, reservoir width, _
}

#[derive(Clone)]
pub struct StrokeRenderer {
    ctx: GpuContext,
    color_space: Arc<dyn ColorSpace>,
    pipeline: wgpu::RenderPipeline,
    uniform_bgl: wgpu::BindGroupLayout,
    prefix_bgl: wgpu::BindGroupLayout,
    /// Cached round-tip prefix-τ, keyed by `hardness.to_bits()`.
    round_prefix: Arc<Mutex<Option<(u32, wgpu::TextureView)>>>,
    /// Canvas surface (group 2 of the sweep pipeline): bump + sampler for tooth.
    surface_bg: wgpu::BindGroup,
    /// The surface itself, also bound to the integrate pass for the knife's
    /// tooth-gated scrape (its `relief` makes the gate a no-op on `Flat`).
    surface: Surface,
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

/// Mode + params for the integrate pass (`integrate.wesl` `Params`).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct IntegrateUniform {
    mode: [f32; 4], // mode.x (0=Normal, 1=Knife), bite, load, tooth·relief
    surf: [f32; 4], // tile_tex_origin.x, .y, inv_surface_tile, ridge (knife tooth gate)
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

        // The prefix-τ texture is R32Float, sampled via textureLoad (not
        // filterable), so the shader does its own bilinear lookup.
        let prefix_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stark sweep prefix bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
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
        // linear/clamp sampler, so the deposit blends bands and segments for free.
        let reservoir_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stark sweep reservoir bgl"),
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
                        0 => Float32x2, 1 => Float32x2, 2 => Float32x4, 3 => Float32x4
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
                    Some(wgpu::ColorTargetState {
                        format: color_space.aux_format(),
                        blend: Some(color_space.aux_blend()),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
            }),
            multiview_mask: None,
            cache: None,
        });

        // ---- Wet-mixing pickup: base-region composite + reservoir compute ----
        let (composite_pipeline, composite_view_bgl, composite_tile_bgl) =
            build_composite_pipeline(device, color_space.as_ref());
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
            uniform_bgl,
            prefix_bgl,
            round_prefix: Arc::new(Mutex::new(None)),
            surface_bg,
            surface,
            reservoir_bgl,
            reservoir_sampler,
            composite_pipeline,
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

    /// Render `rec` over `base`, returning a copy-on-write tile map.
    pub fn render(
        &self,
        pool: &TilePool,
        assets: &AssetStore,
        base: &HashTrieMap<TileCoord, TileHandle>,
        rec: &StrokeRecord,
    ) -> HashTrieMap<TileCoord, TileHandle> {
        let rgb = [rec.brush.color[0], rec.brush.color[1], rec.brush.color[2]];
        let channels = self.color_space.rgb_to_channels(rgb);
        let segments = generate_segments(rec);
        if segments.is_empty() {
            return base.clone();
        }

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

        // The reservoir is parameterized by distance, not segment index: the columns
        // run along the stroke's arc length, one per segment (≈ `step` px apart) plus
        // a half-brush of *lead-in* and *lead-out* padding so the round end caps —
        // which the sweep extends a radius beyond each endpoint — are covered by the
        // mixer's walk extended into those pads, rather than clamping to a flat slab.
        // `geom.w` is a segment start's column-center texel-u in that padded texture;
        // the deposit advances it by the fragment's own travel to the true arc
        // position (`local.x·radius·u_per_px`, see mixer.wesl).
        let count = segments.len();
        let total = segments.iter().map(|s| s.length).sum::<f32>().max(1e-3);
        let step = total / count as f32; // ≈ FLATTEN_STEP, the column spacing
        let lead = (segments.first().unwrap().radius / step).ceil() as u32;
        let tail = (segments.last().unwrap().radius / step).ceil() as u32;
        let width = lead + count as u32 + tail;
        let u_per_px = 1.0 / (step * width as f32); // normalized-u per px travelled
        let instances: Vec<SegmentInstance> = segments
            .iter()
            .map(|s| SegmentInstance {
                start: s.start.to_array(),
                dir: s.dir.to_array(),
                geom: [
                    s.radius,
                    s.length,
                    s.flow,
                    (lead as f32 + 0.5 + s.dist / step) / width as f32,
                ],
                aux: [s.height, s.wet, s.opacity, 0.0],
            })
            .collect();
        // The mixer compute pass reads this buffer to drive its reservoir scan, so
        // it is a storage source as well as vertex data. Written via `write_buffer`
        // (not `create_buffer_init`, which maps-at-creation): a long stroke makes this
        // buffer large, and Chrome/Dawn caps map-at-creation buffers well below the
        // normal `maxBufferSize`, so a long stroke would panic in `createBuffer`.
        let instance_bytes = bytemuck::cast_slice(&instances);
        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stark sweep instances"),
            size: instance_bytes.len() as u64,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.ctx.queue.write_buffer(&instance_buf, 0, instance_bytes);

        let coords = affected_tiles(&segments);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stark stroke commit"),
        });

        // The Dry brush's `smear` runs the reservoir scan: a serial walk that lifts
        // canvas paint (pickup) + injects the brush's own colour (add) into a per-band
        // reservoir, storing premultiplied carried paint that the deposit lays — fading
        // as the load runs out. Only runs when smearing; every other case (plain paint,
        // erase, Wet) gets a 1×1 texel of the brush colour so the deposit stays uniform.
        let dry = match rec.brush.dynamics {
            BrushDynamics::Dry(mp) => Some(mp),
            _ => None,
        };
        let scan = dry
            .filter(|mp| mp.smear > 0.0)
            .map(|mp| [mp.smear, mp.add, crate::path::FLATTEN_STEP, RESERVOIR_CAPACITY]);
        // The smear runs only when its scan actually ran; an oversized stroke skips the
        // region composite and degrades to a plain (scrape/paint) deposit.
        let mut smearing = false;
        let reservoir_view = match scan.zip(stroke_bbox(&segments)) {
            Some((knobs, bbox)) => {
                let tex = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("stark reservoir"),
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
                });
                let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
                self.encode_mixer(
                    &mut encoder, base, &coords, &segments, &instance_buf, &view, bbox,
                    (lead, width), channels, knobs,
                );
                smearing = true;
                view
            }
            None => self.flat_reservoir(&mut encoder, channels),
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
            ],
        });

        // Every brush rasterizes its footprint into a *cleared scratch* tile, then the
        // integrate pass merges it over the base into a fresh CoW tile (DESIGN.md
        // §6.2/§6.1). `empty` (cleared) stands in as the base wherever the stroke
        // touches bare canvas — acquired tiles are undefined, so clear it once here.
        let empty = pool.acquire();
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

        // Integrate mode: the Dry brush scrapes (remove) + films (add) + ridges; Wet
        // uses Normal (mode 0) for its deposit. The scrape is tooth-gated by the canvas
        // surface (`relief` makes it a no-op on Flat). When smearing, the film fraction
        // is forced to 1 — the carried amount already rides the scratch's opacity;
        // otherwise `add` is the own-paint film fraction (`remove`=`add`=0 → plain paint).
        let integrate_mode = match dry {
            Some(mp) => [
                1.0,
                mp.remove,
                if smearing { 1.0 } else { mp.add },
                rec.brush.tooth * self.surface.relief,
            ],
            None => [0.0; 4],
        };
        let ridge = dry.map_or(0.0, |mp| mp.ridge);

        // Wet brush: after the Normal deposit, diffuse the stroke region (DESIGN.md
        // §6.2). Collect this stroke's scratch footprints so the diffusion localizes
        // to where the brush bore down.
        let wet = match rec.brush.dynamics {
            BrushDynamics::Wet(wp) => Some(wp),
            _ => None,
        };
        let mut wet_scratch: Vec<(TileCoord, TileHandle)> = Vec::new();

        let mut new_map = base.clone();
        for coord in &coords {
            // Per-tile sweep transform: texture top-left = interior origin shifted
            // out by the apron, so the full TILE_TEX target maps to NDC [-1, 1].
            let apron = TILE_APRON as f32;
            let origin = coord.origin();
            let xform = TileXform {
                params: [origin.x - apron, origin.y - apron, 2.0 / TILE_TEX as f32, u_per_px],
                // The deposit reads its opacity from the reservoir alpha uniformly (the
                // smear's per-band load, or 1 for the flat reservoir), so no mode flag.
                surf: [1.0 / SURFACE_TILE_PX, rec.brush.tooth, 0.0, 0.0],
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
            // thickness/wet additively).
            let scratch = pool.acquire();
            {
                let clear = wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                };
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("stark sweep pass"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment { view: scratch.color_view(), resolve_target: None, depth_slice: None, ops: clear }),
                        Some(wgpu::RenderPassColorAttachment { view: scratch.aux_view(), resolve_target: None, depth_slice: None, ops: clear }),
                    ],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.set_bind_group(1, &prefix_bg, &[]);
                pass.set_bind_group(2, &self.surface_bg, &[]);
                pass.set_bind_group(3, &reservoir_bg, &[]);
                pass.set_vertex_buffer(0, instance_buf.slice(..));
                pass.draw(0..4, 0..instances.len() as u32);
            }

            // Integrate the scratch slab over the base into a fresh CoW tile. The
            // params carry this tile's origin so the knife's tooth gate samples the
            // canvas surface in canvas space.
            let integrate_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stark integrate params"),
                contents: bytemuck::bytes_of(&IntegrateUniform {
                    mode: integrate_mode,
                    surf: [origin.x - apron, origin.y - apron, 1.0 / SURFACE_TILE_PX, ridge],
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let dst = pool.acquire();
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
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.surface.view) },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&self.surface.sampler) },
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
            if wet.is_some() {
                wet_scratch.push((*coord, scratch.clone()));
            }
            new_map = new_map.insert(*coord, dst);
        }

        // Wet flow: drag (advect) + bleed (diffuse) the freshly-deposited region in one
        // pass over it (DESIGN.md §6.2). No-op when both params are 0.
        if let Some(wp) = wet {
            self.encode_flow(
                &mut encoder, pool, &mut new_map, &coords, &wet_scratch, &instance_buf,
                instances.len() as u32, wp.bleed, wp.drag,
            );
        }

        self.ctx.queue.submit([encoder.finish()]);
        new_map
    }

    /// Composite the given tiles' channels into a fresh 1:1 region (color, aux),
    /// reusing the region pipeline. `tiles` = (canvas origin, color view, aux view);
    /// `origin`/`w`/`h` define the region rect in canvas px. The region textures also
    /// carry `COPY_SRC` so they can be sliced back into tiles.
    fn composite_region(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        tiles: &[(Vec2, wgpu::TextureView, wgpu::TextureView)],
        origin: Vec2,
        w: u32,
        h: u32,
    ) -> (wgpu::Texture, wgpu::Texture) {
        let device = &self.ctx.device;
        let extent = wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 };
        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC;
        let make = |format, label| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let color_tex = make(self.color_space.color_format(), "stark region color");
        let aux_tex = make(self.color_space.aux_format(), "stark region aux");
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
                pass.set_pipeline(&self.composite_pipeline);
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
        encoder: &mut wgpu::CommandEncoder,
        pool: &TilePool,
        map: &mut HashTrieMap<TileCoord, TileHandle>,
        coords: &BTreeSet<TileCoord>,
        scratch: &[(TileCoord, TileHandle)],
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
        let (a_color, a_aux) = self.composite_halo(encoder, map, &halo, region_origin, w, h);
        // B is the advection ping-pong scratch and the bleed's horizontal-blur target.
        let b_color = self.region_tex(w, h, self.color_space.color_format(), "stark flow color b");
        let b_aux = self.region_tex(w, h, self.color_space.aux_format(), "stark flow aux b");
        let av_c = a_color.create_view(&wgpu::TextureViewDescriptor::default());
        let av_a = a_aux.create_view(&wgpu::TextureViewDescriptor::default());
        let bv_c = b_color.create_view(&wgpu::TextureViewDescriptor::default());
        let bv_a = b_aux.create_view(&wgpu::TextureViewDescriptor::default());
        let store = wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            store: wgpu::StoreOp::Store,
        };

        // --- Drag: inject the stroke's velocity (dir · drag), then semi-Lagrangian-
        //     advect FLOW_ITERS times, ping-ponging A↔B (even count → result back in A).
        if drag > 0.0 {
            let vel_tex = self.region_tex(w, h, self.color_space.aux_format(), "stark flow velocity");
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
            let foot: Vec<_> = scratch
                .iter()
                .map(|(c, t)| (c.origin(), t.color_view().clone(), t.aux_view().clone()))
                .collect();
            let (rate_color, _rate_aux) = self.composite_region(encoder, &foot, region_origin, w, h);
            let rate_view = rate_color.create_view(&wgpu::TextureViewDescriptor::default());
            let c_color = self.region_tex(w, h, self.color_space.color_format(), "stark flow color c");
            let c_aux = self.region_tex(w, h, self.color_space.aux_format(), "stark flow aux c");
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
    fn region_tex(&self, w: u32, h: u32, format: wgpu::TextureFormat, label: &str) -> wgpu::Texture {
        self.ctx.device.create_texture(&wgpu::TextureDescriptor {
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
        })
    }

    /// Composite the post-deposit tiles for `halo` into a fresh region (color, aux).
    fn composite_halo(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        map: &HashTrieMap<TileCoord, TileHandle>,
        halo: &[TileCoord],
        region_origin: Vec2,
        w: u32,
        h: u32,
    ) -> (wgpu::Texture, wgpu::Texture) {
        let post: Vec<_> = halo
            .iter()
            .filter_map(|c| map.get(c).map(|t| (c.origin(), t.color_view().clone(), t.aux_view().clone())))
            .collect();
        self.composite_region(encoder, &post, region_origin, w, h)
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
        map: &mut HashTrieMap<TileCoord, TileHandle>,
        coords: &BTreeSet<TileCoord>,
        color_tex: &wgpu::Texture,
        aux_tex: &wgpu::Texture,
        lo: Vec2,
    ) {
        let block = wgpu::Extent3d { width: TILE_TEX, height: TILE_TEX, depth_or_array_layers: 1 };
        for c in coords {
            let dst = pool.acquire();
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
        let coverage = round_coverage(hardness, ROUND_RES);
        let (_tex, view) = build_prefix_tau(&self.ctx, ROUND_RES, ROUND_RES, &coverage);
        *cache = Some((key, view.clone()));
        view
    }

    /// A 1×1 reservoir holding the brush's own color, for brushes without pickup.
    /// Cleared (not CPU-uploaded) so the driver does the f16 encode; the deposit
    /// then samples one uniform color across the whole tip.
    fn flat_reservoir(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        channels: [f32; 4],
    ) -> wgpu::TextureView {
        let tex = self.ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("stark flat reservoir"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("stark flat reservoir clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: channels[0] as f64,
                        g: channels[1] as f64,
                        b: channels[2] as f64,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        view
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
        encoder: &mut wgpu::CommandEncoder,
        base: &HashTrieMap<TileCoord, TileHandle>,
        coords: &BTreeSet<TileCoord>,
        segments: &[Segment],
        instance_buf: &wgpu::Buffer,
        reservoir: &wgpu::TextureView,
        bbox: (Vec2, u32, u32),
        pad: (u32, u32),
        channels: [f32; 4],
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
        let make_tex = |format, label| {
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
            brush_ch: channels,
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
/// per-band smear for a [`BrushDynamics::Dry`] brush — see `encode_mixer`).
fn generate_segments(rec: &StrokeRecord) -> Vec<Segment> {
    let b = &rec.brush;
    let pts = crate::path::flatten(&rec.path, crate::path::FLATTEN_STEP);
    let mut segs = Vec::new();
    if pts.is_empty() {
        return segs;
    }

    let make = |sample: &InputSample, dir: Vec2, len: f32, dist: f32| -> Segment {
        let load = (1.0 - b.drain * dist).max(0.0);
        Segment {
            start: sample.pos,
            dir,
            radius: (b.radius * sample.pressure).max(0.5),
            length: len,
            // `flow` now drives only the footprint build-up; the brush's opacity
            // (color[3]) rides the separate opacity channel (DESIGN.md §6.1).
            flow: b.flow * load * SWEEP_FLOW_SCALE,
            height: b.height * load,
            wet: b.wetness * load,
            opacity: b.color[3] * load,
            dist,
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
/// §6.2, §6.3). Returns `(pipeline, view bgl, tile bgl)`.
fn build_composite_pipeline(
    device: &wgpu::Device,
    color_space: &dyn ColorSpace,
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
                    format: color_space.aux_format(),
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
            // Canvas surface (filterable + sampler) for the knife's tooth gate.
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
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
