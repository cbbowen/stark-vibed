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
//! This is the plain **add** path: footprint → cleared scratch tile → integrate
//! over the base into a fresh CoW tile. Brush dynamics (smear, drag, bleed —
//! DESIGN §6.2) are being rebuilt on a sequential stamp loop and will layer on
//! top of this core.
//!
//! The renderer is parameterized by a [`ColorSpace`] (formats, blends, channel
//! mapping, shader). It holds only immutable GPU objects plus `Arc`-backed
//! handles, so it is cheap to `Clone` and can live in the `Action::Context` (§5).

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};
use rpds::HashTrieMap;
use wgpu::util::DeviceExt;

use crate::assets::{build_coverage_r8, build_prefix_tau, AssetStore};
use crate::colorspace::ColorSpace;
use crate::command::InputSample;
use crate::document::{BrushShape, OrientationSource, StrokeRecord};
use crate::geom::{
    TileCoord, Vec2, INTERIOR_UV_BIAS, INTERIOR_UV_SCALE, TILE_APRON, TILE_SIZE, TILE_TEX,
};
use crate::gpu::context::GpuContext;
use crate::gpu::surface::{Surface, SURFACE_TILE_PX};
use crate::gpu::tile::{AllocSource, SCRATCH_AUX_FORMAT, TilePairHandle, TilePool};

/// Global tuning so a default brush (`flow = 1`) reads as a solid stroke;
/// `flow` is an optical-depth-per-length rate (DESIGN.md §6.2).
const SWEEP_FLOW_SCALE: f32 = 1.0;
/// Resolution of the generated round-tip prefix texture.
const ROUND_RES: u32 = 256;

/// Resolution (texels per side) of the stamp loop's tool reservoir (DESIGN.md
/// §6.2). Brush-local, so carried colour detail is ~radius/32 canvas px — plenty
/// for smeared paint, and small enough that the per-stamp reservoir update is
/// nearly free.
const BRUSH_RES: u32 = 64;
/// Largest stroke-region edge (canvas px) the stamp loop composites. Oversized
/// strokes degrade to the plain swept deposit — rare, and it bounds the transient
/// GPU memory (DESIGN.md §6.2).
const MAX_REGION_DIM: u32 = 2048;
/// Hard cap on stamps per stroke: beyond it the spacing stretches, trading
/// per-radius fidelity on extremely long strokes for bounded cost.
const MAX_STAMPS: usize = 4096;
/// Gain on the `add` axis in the stamp loop, tuned so `add = 1` at default
/// spacing lays roughly the brush's `height` per pass of the tip.
const ADD_GAIN: f32 = 2.0;

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
    /// Drives the color/opacity channel; thickness (`height`) is independent
    /// (DESIGN.md §6.1, normalized representation).
    opacity: f32,
    /// Shape orientation for this segment as a fraction of a full turn ∈ [0, 1): the
    /// relative angle between the shape's native axis and the travel direction, used to
    /// pick the prefix-τ orientation layer. 0 for follow-stroke (DESIGN.md §6.6).
    orient: f32,
}

/// Per-segment instance data for the sweep shader.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SegmentInstance {
    start: [f32; 2],
    dir: [f32; 2],   // unit tangent
    geom: [f32; 4],  // radius, length, flow, _
    aux: [f32; 4],   // height (thickness rate), wet, opacity, _
    extra: [f32; 4], // orientation (turns ∈ [0,1)), _, _, _
}

/// Per-tile uniform: the tile *texture's* top-left in canvas px + canvas→NDC
/// scale, plus the brush's stroke-constant colour channels. The texture origin is
/// the interior origin minus the apron, so the stroke rasterizes into the apron
/// too (keeping it consistent with the neighbor's interior — see
/// [`crate::geom::TILE_APRON`]).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TileXform {
    params: [f32; 4], // tex_origin.x, tex_origin.y, 2/TILE_TEX, _
    color: [f32; 4],  // brush channels (.xyz), _
    surf: [f32; 4],   // inv surface-tile (canvas px → bump uv), tooth, _, _
}

/// Mirrors `View` in `composite.wesl`: canvas→region NDC + tile/apron uv mapping.
/// Used to composite the base into a 1:1 region texture for the stamp loop.
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

/// Mirrors `Params` in `slice.wesl`: the tile texture's top-left in region texels.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SliceUniform {
    offset: [f32; 4],
}

/// One stamp of the sequential loop (DESIGN.md §6.2): where the tip lands, its
/// frame, and this stamp's exchange rates. All precomputed CPU-side as pure
/// functions of the `StrokeRecord`, so replay is deterministic.
#[derive(Copy, Clone)]
struct StampPoint {
    pos: Vec2,
    /// The brush frame's x-axis in canvas space (unit): the travel tangent, or the
    /// pen azimuth for a pinned nib (§6.6).
    rot: Vec2,
    radius: f32,
    /// Fraction of the canvas paint under the tip lifted onto the tool this stamp.
    lift: f32,
    /// Fraction of the tool's carried paint laid back down this stamp.
    dep: f32,
    /// The brush's own paint (`add` axis): height laid this stamp at full coverage.
    add_h: f32,
    add_wet: f32,
}

/// GPU objects for the brush-dynamics stamp loop (DESIGN.md §6.2), built once.
/// All handles are `Arc`-backed, so the kit is cheap to clone with its renderer.
#[derive(Clone)]
struct DynamicsKit {
    // Region composite: base tiles → one 1:1 canvas region (colour + wide aux).
    composite_pipeline: wgpu::RenderPipeline,
    composite_view_bgl: wgpu::BindGroupLayout,
    composite_tile_bgl: wgpu::BindGroupLayout,
    composite_sampler: wgpu::Sampler,
    // The three stamp-loop dispatches (one compute shader, three entry points).
    snapshot_pipeline: wgpu::ComputePipeline,
    snapshot_bgl: wgpu::BindGroupLayout,
    pickup_pipeline: wgpu::ComputePipeline,
    pickup_bgl: wgpu::BindGroupLayout,
    deposit_pipeline: wgpu::ComputePipeline,
    deposit_bgl: wgpu::BindGroupLayout,
    /// Bilinear clamp sampler for the region / reservoir / coverage lookups.
    exchange_sampler: wgpu::Sampler,
    // Region → CoW tile write-back.
    slice_pipeline: wgpu::RenderPipeline,
    slice_bgl: wgpu::BindGroupLayout,
    /// Cached round-tip coverage texture, keyed by `hardness.to_bits()`.
    round_cov: Arc<Mutex<Option<(u32, wgpu::TextureView)>>>,
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
    /// Canvas surface (group 2 of the sweep pipeline): bump + sampler for the tooth
    /// gate — currently a pass-through stub (`surface_tooth` TODO in stamp_common.wesl);
    /// no stamp/integrate shader reads the surface today, the weave shows through the
    /// media pass instead.
    surface_bg: wgpu::BindGroup,

    // Stroke integrate (DESIGN.md §6.2/§6.1): a fullscreen pass reads the base tile +
    // the stroke's footprint scratch and writes `new = f(base, scratch)` into a fresh
    // CoW tile's color+aux MRT — premultiplied-over + additive height/wet.
    integrate_pipeline: wgpu::RenderPipeline,
    integrate_bgl: wgpu::BindGroupLayout,

    // Brush dynamics: the sequential stamp loop (DESIGN.md §6.2), used when the
    // brush manipulates existing paint (`load` / `deposit` / `charge`).
    dynamics: DynamicsKit,
}

/// GPU resources scoped to one `render()` call (currently the instance buffer;
/// per-stroke region textures register here too as dynamics return). They're sized
/// per-stroke, so — unlike the fixed-`TILE_TEX` tile pool — they can't be recycled,
/// and a *live* stroke re-renders on every pointer move. Left to drop they'd only
/// release the JS handle and wait on GC, which can't keep up → the tab OOMs. So
/// they're collected here (cheap `Arc` clones) and **`destroy()`d on drop**, which
/// `render` arranges to happen right after the submit — safe, because WebGPU defers
/// the real free until the in-flight work referencing them completes.
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

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("stark sweep layout"),
            bind_group_layouts: &[Some(&uniform_bgl), Some(&prefix_bgl), Some(&surface_bgl)],
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
                    // SCRATCH_AUX_FORMAT — not the compact persistent aux. Additive
                    // blend across overlapping segments.
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

        let (integrate_pipeline, integrate_bgl) =
            build_integrate_pipeline(device, color_space.as_ref());
        let dynamics = build_dynamics_kit(device, color_space.as_ref());

        Self {
            ctx: ctx.clone(),
            color_space,
            pipeline,
            uniform_bgl,
            prefix_bgl,
            round_prefix: Arc::new(Mutex::new(None)),
            surface_bg,
            integrate_pipeline,
            integrate_bgl,
            dynamics,
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

        // Brushes that manipulate existing paint run the sequential stamp loop
        // (DESIGN.md §6.2); pure-`add` brushes keep the swept fast path below.
        // `None` (an oversized stroke region) degrades to the fast path too.
        let d = rec.brush.dynamics;
        if (d.load > 0.0 || d.deposit > 0.0 || d.charge > 0.0)
            && let Some(map) = self.render_dynamic(pool, assets, base, rec, channels)
        {
            return map;
        }

        let segments = generate_segments(rec);
        if segments.is_empty() {
            return base.clone();
        }

        // The per-stroke instance buffer registers here and is `destroy()`d when this
        // drops (at the end of `render`, after the submit below) — freeing it
        // deterministically instead of leaking to JS GC (DESIGN.md §6.2).
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

        let instances: Vec<SegmentInstance> = segments
            .iter()
            .map(|s| SegmentInstance {
                start: s.start.to_array(),
                dir: s.dir.to_array(),
                geom: [s.radius, s.length, s.flow, 0.0],
                aux: [s.height, s.wet, s.opacity, 0.0],
                extra: [s.orient, 0.0, 0.0, 0.0],
            })
            .collect();
        // Written via `write_buffer` (not `create_buffer_init`, which maps-at-creation):
        // a long stroke makes this buffer large, and Chrome/Dawn caps map-at-creation
        // buffers well below the normal `maxBufferSize`, so a long stroke would panic
        // in `createBuffer`.
        let instance_bytes: &[u8] = bytemuck::cast_slice(&instances);
        let instance_buf = scoped.buffer(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stark sweep instances"),
            size: instance_bytes.len() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.ctx.queue.write_buffer(&instance_buf, 0, instance_bytes);

        let coords = affected_tiles(&segments);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stark stroke commit"),
        });

        // Every brush rasterizes its footprint into a *cleared scratch* tile, then the
        // integrate pass merges it over the base into a fresh CoW tile (DESIGN.md
        // §6.2/§6.1). `empty` (cleared) stands in as the base wherever the stroke
        // touches bare canvas — acquired tiles are undefined, so clear it once here.
        let clear = wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            store: wgpu::StoreOp::Store,
        };
        let empty = pool.acquire(AllocSource::IntegrateEmptyBase);
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("stark integrate empty clear"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: empty.color_view(),
                    resolve_target: None,
                    depth_slice: None,
                    ops: clear,
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: empty.aux_view(),
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

        let mut new_map = base.clone();
        for coord in &coords {
            // Per-tile sweep transform: texture top-left = interior origin shifted
            // out by the apron, so the full TILE_TEX target maps to NDC [-1, 1].
            let apron = TILE_APRON as f32;
            let origin = coord.origin();
            let xform = TileXform {
                params: [origin.x - apron, origin.y - apron, 2.0 / TILE_TEX as f32, 0.0],
                color: channels,
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
            // height/wet/coverage additively). The scratch aux is the wide format.
            let scratch = pool.acquire_scratch(AllocSource::StrokeScratch);
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("stark sweep pass"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: scratch.color_view(),
                            resolve_target: None,
                            depth_slice: None,
                            ops: clear,
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: scratch.aux_view(),
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
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.set_bind_group(1, &prefix_bg, &[]);
                pass.set_bind_group(2, &self.surface_bg, &[]);
                pass.set_vertex_buffer(0, instance_buf.slice(..));
                pass.draw(0..4, 0..instances.len() as u32);
            }

            // Integrate the scratch slab over the base into a fresh CoW tile.
            let dst = pool.acquire(AllocSource::IntegrateDestination);
            let base_tile = base.get(coord).unwrap_or(&empty);
            let integrate_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("stark integrate bg"),
                layout: &self.integrate_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(base_tile.color_view()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(base_tile.aux_view()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(scratch.color_view()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(scratch.aux_view()),
                    },
                ],
            });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("stark integrate"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: dst.color_view(),
                            resolve_target: None,
                            depth_slice: None,
                            ops: clear,
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: dst.aux_view(),
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
                pass.set_pipeline(&self.integrate_pipeline);
                pass.set_bind_group(0, &integrate_bg, &[]);
                pass.draw(0..3, 0..1);
            }
            new_map = new_map.insert(*coord, dst);
        }

        self.ctx.queue.submit([encoder.finish()]);

        // `scoped` drops here, *after* the submit — destroying this stroke's instance
        // buffer. It isn't pooled (sized per stroke) and a live stroke re-renders every
        // pointer move, so left to JS GC they pile up and OOM the tab; `destroy()`
        // after submit reclaims them at once (WebGPU keeps the memory until the
        // in-flight work that uses them completes).
        drop(scoped);
        new_map
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

    /// The round tip's coverage texture for `hardness`, cached like the prefix.
    fn round_coverage_view(&self, hardness: f32) -> wgpu::TextureView {
        let key = hardness.to_bits();
        let mut cache = self
            .dynamics
            .round_cov
            .lock()
            .expect("round coverage poisoned");
        if let Some((k, view)) = cache.as_ref()
            && *k == key
        {
            return view.clone();
        }
        let cov = round_coverage(hardness, ROUND_RES);
        let bytes: Vec<u8> = cov.iter().map(|c| (c * 255.0).round() as u8).collect();
        let (_tex, view) = build_coverage_r8(&self.ctx, ROUND_RES, ROUND_RES, &bytes);
        *cache = Some((key, view.clone()));
        view
    }

    /// Render a paint-manipulating stroke via the **sequential stamp loop**
    /// (DESIGN.md §6.2): composite the base under the stroke into a 1:1 region,
    /// walk the stamps *in order* on the GPU — each stamp exchanging paint between
    /// the evolving region and a 2-D tool reservoir — then slice the evolved region
    /// back into fresh CoW tiles. Returns `None` when the stroke's region exceeds
    /// [`MAX_REGION_DIM`]; the caller degrades to the plain swept deposit.
    fn render_dynamic(
        &self,
        pool: &TilePool,
        assets: &AssetStore,
        base: &HashTrieMap<TileCoord, TilePairHandle>,
        rec: &StrokeRecord,
        channels: [f32; 4],
    ) -> Option<HashTrieMap<TileCoord, TilePairHandle>> {
        let stamps = generate_stamps(rec);
        if stamps.is_empty() {
            return Some(base.clone());
        }
        let coords = stamp_tiles(&stamps);
        let (halo, lo, region_origin, w, h) = region_rect(&coords)?;

        let kit = &self.dynamics;
        let device = &self.ctx.device;
        let mut scoped = ScopedResources::default();

        // Footprint coverage mask: image brushes from the store; the round tip
        // generated (and cached) from its hardness.
        let cov_view = match rec.brush.shape {
            BrushShape::Stamp(id) => assets
                .coverage_view(id)
                .unwrap_or_else(|| self.round_coverage_view(rec.brush.hardness)),
            BrushShape::Round => self.round_coverage_view(rec.brush.hardness),
        };

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stark dynamics stroke"),
        });
        let clear = wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            store: wgpu::StoreOp::Store,
        };

        // ---- The stroke's canvas region (colour + wide aux), composited from the
        // base tiles of the affected set plus a one-tile ring, so rewritten tiles'
        // aprons read real neighbour content (§6.4). Rgba16Float throughout: it is
        // both filterable and a core storage format, and matches the tile colour
        // format of both color spaces (asserted in `build_dynamics_kit`).
        let make_tex = |scoped: &mut ScopedResources,
                        size: (u32, u32),
                        usage: wgpu::TextureUsages,
                        label: &'static str| {
            scoped
                .texture(device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: size.0,
                        height: size.1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage,
                    view_formats: &[],
                }))
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let region_usage = wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING;
        let region_color = make_tex(&mut scoped, (w, h), region_usage, "stark dynamics region color");
        let region_aux = make_tex(&mut scoped, (w, h), region_usage, "stark dynamics region aux");

        // Composite pass: base tiles → region, 1:1 with canvas px.
        let (sx, sy) = (2.0 / w as f32, -2.0 / h as f32);
        let view = ViewUniform {
            st: [sx, sy, -region_origin.x * sx - 1.0, -region_origin.y * sy + 1.0],
            misc: [TILE_SIZE as f32, INTERIOR_UV_SCALE, INTERIOR_UV_BIAS, 0.0],
        };
        let view_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stark dynamics region view"),
            contents: bytemuck::bytes_of(&view),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let view_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark dynamics region view bg"),
            layout: &kit.composite_view_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: view_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&kit.composite_sampler),
                },
            ],
        });
        let mut tile_origins: Vec<TileInstance> = Vec::new();
        let mut tile_bgs: Vec<wgpu::BindGroup> = Vec::new();
        for coord in &halo {
            if let Some(tile) = base.get(coord) {
                tile_origins.push(TileInstance {
                    origin: coord.origin().to_array(),
                    opacity: 1.0,
                });
                tile_bgs.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("stark dynamics region tile bg"),
                    layout: &kit.composite_tile_bgl,
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
        let tile_inst = (!tile_origins.is_empty()).then(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stark dynamics region tile instances"),
                contents: bytemuck::cast_slice(&tile_origins),
                usage: wgpu::BufferUsages::VERTEX,
            })
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stark dynamics region composite"),
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
            // An empty region (no base tiles) just stays cleared → "no paint".
            if let Some(inst) = &tile_inst {
                pass.set_pipeline(&kit.composite_pipeline);
                pass.set_bind_group(0, &view_bg, &[]);
                pass.set_vertex_buffer(0, inst.slice(..));
                for (i, bg) in tile_bgs.iter().enumerate() {
                    let idx = i as u32;
                    pass.set_bind_group(1, bg, &[]);
                    pass.draw(0..4, idx..idx + 1);
                }
            }
        }

        // ---- Tool reservoir (ping-pong) + footprint snapshot textures. The
        // footprint quad reaches radius·√2 at rotated corners, plus filter margin.
        let loop_usage =
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING;
        let rmax = stamps.iter().fold(0.5f32, |m, s| m.max(s.radius));
        let dsize = (2.0 * (rmax * std::f32::consts::SQRT_2 + 2.0)).ceil() as u32;
        let under_color = make_tex(&mut scoped, (dsize, dsize), loop_usage, "stark dynamics under color");
        let under_aux = make_tex(&mut scoped, (dsize, dsize), loop_usage, "stark dynamics under aux");
        // The first reservoir is initialized by a render clear (the driver does the
        // f16 encode), hence the extra RENDER_ATTACHMENT.
        let brush_usage = loop_usage | wgpu::TextureUsages::RENDER_ATTACHMENT;
        let bres = (BRUSH_RES, BRUSH_RES);
        let brush_color = [
            make_tex(&mut scoped, bres, brush_usage, "stark dynamics brush color a"),
            make_tex(&mut scoped, bres, brush_usage, "stark dynamics brush color b"),
        ];
        let brush_aux = [
            make_tex(&mut scoped, bres, brush_usage, "stark dynamics brush aux a"),
            make_tex(&mut scoped, bres, brush_usage, "stark dynamics brush aux b"),
        ];
        {
            // Init: latent = the brush's own colour, per-unit opacity = its alpha;
            // the carried amount starts at the pre-`charge` glob (0 = empty tool).
            let d = rec.brush.dynamics;
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stark dynamics brush init"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &brush_color[0],
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: channels[0] as f64,
                                g: channels[1] as f64,
                                b: channels[2] as f64,
                                a: rec.brush.color[3] as f64,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &brush_aux[0],
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: d.charge as f64,
                                g: (d.charge * rec.brush.wetness) as f64,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }

        // ---- Per-stamp params, one 256-byte slot each (dynamic uniform offsets —
        // the standard way to vary a uniform across dispatches within one pass).
        const STRIDE: usize = 256;
        let mut data = vec![0u8; stamps.len() * STRIDE];
        let half = (dsize / 2) as f32;
        for (i, s) in stamps.iter().enumerate() {
            let p = s.pos - region_origin;
            let vals: [f32; 16] = [
                p.x,
                p.y,
                s.rot.x,
                s.rot.y,
                s.radius,
                s.lift,
                s.dep,
                s.add_h,
                channels[0],
                channels[1],
                channels[2],
                rec.brush.color[3],
                (p.x - half).floor(),
                (p.y - half).floor(),
                s.add_wet,
                0.0,
            ];
            data[i * STRIDE..i * STRIDE + 64].copy_from_slice(bytemuck::cast_slice(&vals));
        }
        let stamp_buf = scoped.buffer(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stark dynamics stamps"),
            size: data.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.ctx.queue.write_buffer(&stamp_buf, 0, &data);

        // ---- Bind groups. `params` binds a single 64-byte window whose dynamic
        // offset selects the stamp; pickup/deposit come in two flavours for the
        // reservoir ping-pong (src = stamp index % 2).
        let params = || wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &stamp_buf,
                offset: 0,
                size: wgpu::BufferSize::new(64),
            }),
        };
        fn tex(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: wgpu::BindingResource::TextureView(view),
            }
        }
        let samp = || wgpu::BindGroupEntry {
            binding: 5,
            resource: wgpu::BindingResource::Sampler(&kit.exchange_sampler),
        };
        let snapshot_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark dynamics snapshot bg"),
            layout: &kit.snapshot_bgl,
            entries: &[
                params(),
                tex(1, &region_color),
                tex(2, &region_aux),
                tex(3, &under_color),
                tex(4, &under_aux),
            ],
        });
        let pickup_bgs: Vec<wgpu::BindGroup> = (0..2)
            .map(|i| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("stark dynamics pickup bg"),
                    layout: &kit.pickup_bgl,
                    entries: &[
                        params(),
                        tex(1, &region_color),
                        tex(2, &region_aux),
                        samp(),
                        tex(6, &cov_view),
                        tex(7, &brush_color[i]),
                        tex(8, &brush_aux[i]),
                        tex(9, &brush_color[1 - i]),
                        tex(10, &brush_aux[1 - i]),
                    ],
                })
            })
            .collect();
        let deposit_bgs: Vec<wgpu::BindGroup> = (0..2)
            .map(|i| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("stark dynamics deposit bg"),
                    layout: &kit.deposit_bgl,
                    entries: &[
                        params(),
                        samp(),
                        tex(6, &cov_view),
                        tex(7, &brush_color[i]),
                        tex(8, &brush_aux[i]),
                        tex(11, &under_color),
                        tex(12, &under_aux),
                        tex(13, &region_color),
                        tex(14, &region_aux),
                    ],
                })
            })
            .collect();

        // ---- The loop: snapshot → pickup → deposit per stamp, in stroke order.
        // One compute pass; the implicit barriers between dispatches give the
        // sequential semantics, and usage scopes are per-dispatch, so the region
        // may be sampled by one dispatch and storage-written by the next.
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stark dynamics stamp loop"),
                timestamp_writes: None,
            });
            let du = dsize.div_ceil(8);
            let bu = BRUSH_RES.div_ceil(8);
            for i in 0..stamps.len() {
                let off = (i * STRIDE) as u32;
                let pp = i % 2;
                cpass.set_pipeline(&kit.snapshot_pipeline);
                cpass.set_bind_group(0, &snapshot_bg, &[off]);
                cpass.dispatch_workgroups(du, du, 1);
                cpass.set_pipeline(&kit.pickup_pipeline);
                cpass.set_bind_group(0, &pickup_bgs[pp], &[off]);
                cpass.dispatch_workgroups(bu, bu, 1);
                cpass.set_pipeline(&kit.deposit_pipeline);
                cpass.set_bind_group(0, &deposit_bgs[pp], &[off]);
                cpass.dispatch_workgroups(du, du, 1);
            }
        }

        // ---- Write-back: slice each affected tile's full TILE_TEX block out of
        // the shared region → aprons stay bit-identical to neighbour interiors
        // (§6.4), and the wide region aux narrows to the persistent (height, wet).
        let mut new_map = base.clone();
        for coord in &coords {
            let dst = pool.acquire(AllocSource::DynamicsWriteback);
            let off = coord.origin() - lo;
            let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stark dynamics slice params"),
                contents: bytemuck::bytes_of(&SliceUniform {
                    offset: [off.x, off.y, 0.0, 0.0],
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("stark dynamics slice bg"),
                layout: &kit.slice_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ubuf.as_entire_binding(),
                    },
                    tex(1, &region_color),
                    tex(2, &region_aux),
                ],
            });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("stark dynamics slice"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: dst.color_view(),
                            resolve_target: None,
                            depth_slice: None,
                            ops: clear,
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: dst.aux_view(),
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
                pass.set_pipeline(&kit.slice_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.draw(0..3, 0..1);
            }
            new_map = new_map.insert(*coord, dst);
        }

        self.ctx.queue.submit([encoder.finish()]);
        // Destroy the per-stroke region/reservoir textures + buffers now (safe:
        // WebGPU defers the real free past the submitted work) — see the
        // `ScopedResources` docs for why waiting on JS GC OOMs the tab.
        drop(scoped);
        Some(new_map)
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
/// (`drain`) depletes with arc distance; radius follows pressure.
fn generate_segments(rec: &StrokeRecord) -> Vec<Segment> {
    let b = &rec.brush;
    let pts = crate::path::flatten(&rec.path, crate::path::FLATTEN_STEP);
    let mut segs = Vec::new();
    if pts.is_empty() {
        return segs;
    }

    // `dist` (arc length from the stroke start) drives the drain.
    let make = |sample: &InputSample, dir: Vec2, len: f32, dist: f32| -> Segment {
        let drain = (1.0 - b.drain * dist).max(0.0);
        Segment {
            start: sample.pos,
            dir,
            radius: (b.radius * sample.pressure).max(0.5),
            length: len,
            // `flow` drives only the footprint build-up; the brush's opacity
            // (color[3]) rides the separate opacity channel (DESIGN.md §6.1).
            flow: b.flow * drain * SWEEP_FLOW_SCALE,
            height: b.height * drain,
            wet: b.wetness * drain,
            opacity: b.color[3] * drain,
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

/// Tiles whose *texture* (interior + apron) any segment's swept capsule overlaps.
/// The apron is included in `reach` so a stroke landing within a tile's interior
/// but inside a neighbor's apron band re-renders that neighbor too, keeping the
/// shared apron/interior overlap bit-identical (DESIGN.md §6.4).
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

/// Build swept stamps for the sequential loop (DESIGN.md §6.2): flatten the
/// spline, then walk it at `spacing · radius` arc-length steps. Every per-stamp
/// rate is normalized by the travel since the last stamp, so the exchange over
/// one radius of travel applies each axis' full fraction — independent of the
/// spacing setting. Pure CPU float math → replay-deterministic.
fn generate_stamps(rec: &StrokeRecord) -> Vec<StampPoint> {
    let b = &rec.brush;
    let d = b.dynamics;
    let pts = crate::path::flatten(&rec.path, crate::path::FLATTEN_STEP);
    if pts.is_empty() {
        return Vec::new();
    }
    let spacing = b.spacing.clamp(0.05, 2.0);
    let total: f32 = pts.windows(2).map(|w| (w[1].pos - w[0].pos).length()).sum();
    // Cap the stamp count: an extremely long stroke stretches its spacing instead.
    let min_step = (total / MAX_STAMPS as f32).max(0.25);

    let load = d.load.clamp(0.0, 1.0);
    let deposit = d.deposit.clamp(0.0, 1.0);
    let make_stamp = |sample: InputSample, dir: Vec2, dist: f32, step: f32| -> StampPoint {
        let radius = (b.radius * sample.pressure).max(0.5);
        let drain = (1.0 - b.drain * dist).max(0.0);
        // Normalized travel since the last stamp: one radius applies the full axis.
        let ds = step / radius;
        let turns = orientation_turns(b.orientation, dir, sample.tilt);
        let (s, c) = (turns * std::f32::consts::TAU).sin_cos();
        let rot = Vec2::new(dir.x * c - dir.y * s, dir.x * s + dir.y * c);
        StampPoint {
            pos: sample.pos,
            rot,
            radius,
            lift: 1.0 - (1.0 - load).powf(ds),
            dep: 1.0 - (1.0 - deposit).powf(ds),
            add_h: b.height * d.add * drain * ds * ADD_GAIN,
            add_wet: b.wetness * d.add * drain * ds * ADD_GAIN,
        }
    };

    // First stamp at the start; direction from the first non-degenerate edge (a
    // click gets an arbitrary frame — its round footprint doesn't care).
    let first_dir = pts
        .windows(2)
        .map(|w| w[1].pos - w[0].pos)
        .find(|v| v.length() > 1e-5)
        .map(|v| v.normalize())
        .unwrap_or(Vec2::new(1.0, 0.0));
    let first_r = (b.radius * pts[0].pressure).max(0.5);
    let mut out: Vec<StampPoint> = Vec::new();
    out.push(make_stamp(pts[0], first_dir, 0.0, (spacing * first_r).max(min_step)));
    let mut last_r = first_r;

    let mut dist = 0.0f32;
    let mut since = 0.0f32; // arc distance since the last stamp
    for w in pts.windows(2) {
        let (p0, p1) = (&w[0], &w[1]);
        let v = p1.pos - p0.pos;
        let len = v.length();
        if len < 1e-6 {
            continue;
        }
        let dir = v / len;
        let mut t = 0.0f32;
        loop {
            let step = (spacing * last_r).max(min_step);
            let need = step - since;
            if t + need > len {
                since += len - t;
                dist += len - t;
                break;
            }
            t += need;
            dist += need;
            let f = t / len;
            let sample = InputSample {
                pos: p0.pos + v * f,
                pressure: p0.pressure + (p1.pressure - p0.pressure) * f,
                tilt: p0.tilt + (p1.tilt - p0.tilt) * f,
                time: p0.time + (p1.time - p0.time) * f as f64,
            };
            let stamp = make_stamp(sample, dir, dist, step);
            last_r = stamp.radius;
            out.push(stamp);
            since = 0.0;
        }
    }
    out
}

/// Tiles whose texture (interior + apron) any stamp's rotated footprint square
/// can overlap (reach = radius·√2 at the corners) — the stamp loop's write-back
/// set, mirroring [`affected_tiles`] for the swept path.
fn stamp_tiles(stamps: &[StampPoint]) -> BTreeSet<TileCoord> {
    let tile = TILE_SIZE as f32;
    let mut coords = BTreeSet::new();
    for s in stamps {
        let reach = Vec2::splat(s.radius * std::f32::consts::SQRT_2 + TILE_APRON as f32);
        let lo = s.pos - reach;
        let hi = s.pos + reach;
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

/// The haloed, tile-aligned region for a stroke's affected `coords`: those tiles
/// plus a one-tile ring, so a rewritten tile's apron reads its neighbour's real
/// interior from the region (the write-back overwrites whole `TILE_TEX` blocks —
/// §6.4). Returns `(halo tiles, lo origin, region origin, w, h)`, or `None` if
/// empty or larger than [`MAX_REGION_DIM`].
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

/// Build the brush-dynamics stamp-loop kit (DESIGN.md §6.2): the region
/// composite, the three loop compute pipelines, and the region→tile slice.
fn build_dynamics_kit(device: &wgpu::Device, color_space: &dyn ColorSpace) -> DynamicsKit {
    // The loop's storage-texture declarations are `rgba16float`; both color
    // spaces use that tile colour format (§6.7), so the region can hold either.
    debug_assert_eq!(color_space.color_format(), wgpu::TextureFormat::Rgba16Float);

    // ---- Region composite: the `composite` shader over region-sized targets
    // (colour + the wide aux, so nothing is narrowed until the write-back).
    let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("stark dynamics composite"),
        source: wgpu::ShaderSource::Wgsl(stark_shaders::composite().into()),
    });
    let composite_view_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("stark dynamics composite view bgl"),
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
    let filter_tex = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let composite_tile_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("stark dynamics composite tile bgl"),
        entries: &[filter_tex(0), filter_tex(1)],
    });
    let composite_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("stark dynamics composite layout"),
        bind_group_layouts: &[Some(&composite_view_bgl), Some(&composite_tile_bgl)],
        immediate_size: 0,
    });
    let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("stark dynamics composite pipeline"),
        layout: Some(&composite_layout),
        vertex: wgpu::VertexState {
            module: &composite_shader,
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
            module: &composite_shader,
            entry_point: Some("fs_main"),
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
            ],
        }),
        multiview_mask: None,
        cache: None,
    });
    let composite_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("stark dynamics composite sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    // ---- The stamp loop: one module, three entry points, one bind group each
    // (all include the dynamic-offset stamp uniform at binding 0; the binding
    // numbers partition the module's group(0) — see dynamics.wesl).
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("stark dynamics loop"),
        source: wgpu::ShaderSource::Wgsl(stark_shaders::dynamics().into()),
    });
    let params_entry = wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: wgpu::BufferSize::new(64),
        },
        count: None,
    };
    let ctex = |binding: u32, filterable: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let stor = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::Rgba16Float,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    };
    let csamp = wgpu::BindGroupLayoutEntry {
        binding: 5,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    };
    let snapshot_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("stark dynamics snapshot bgl"),
        entries: &[params_entry, ctex(1, false), ctex(2, false), stor(3), stor(4)],
    });
    let pickup_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("stark dynamics pickup bgl"),
        entries: &[
            params_entry,
            ctex(1, true),
            ctex(2, true),
            csamp,
            ctex(6, true),
            ctex(7, false),
            ctex(8, false),
            stor(9),
            stor(10),
        ],
    });
    let deposit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("stark dynamics deposit bgl"),
        entries: &[
            params_entry,
            csamp,
            ctex(6, true),
            ctex(7, true),
            ctex(8, true),
            ctex(11, false),
            ctex(12, false),
            stor(13),
            stor(14),
        ],
    });
    let cpipe = |label: &str, entry: &str, bgl: &wgpu::BindGroupLayout| {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[Some(bgl)],
            immediate_size: 0,
        });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let snapshot_pipeline = cpipe("stark dynamics snapshot", "snapshot", &snapshot_bgl);
    let pickup_pipeline = cpipe("stark dynamics pickup", "pickup", &pickup_bgl);
    let deposit_pipeline = cpipe("stark dynamics deposit", "deposit", &deposit_bgl);
    let exchange_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("stark dynamics exchange sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    // ---- Region → tile slice (write-back).
    let slice_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("stark dynamics slice"),
        source: wgpu::ShaderSource::Wgsl(stark_shaders::slice().into()),
    });
    let load_tex = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let slice_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("stark dynamics slice bgl"),
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
            load_tex(1),
            load_tex(2),
        ],
    });
    let slice_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("stark dynamics slice layout"),
        bind_group_layouts: &[Some(&slice_bgl)],
        immediate_size: 0,
    });
    let slice_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("stark dynamics slice pipeline"),
        layout: Some(&slice_layout),
        vertex: wgpu::VertexState {
            module: &slice_shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &slice_shader,
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

    DynamicsKit {
        composite_pipeline,
        composite_view_bgl,
        composite_tile_bgl,
        composite_sampler,
        snapshot_pipeline,
        snapshot_bgl,
        pickup_pipeline,
        pickup_bgl,
        deposit_pipeline,
        deposit_bgl,
        exchange_sampler,
        slice_pipeline,
        slice_bgl,
        round_cov: Arc::new(Mutex::new(None)),
    }
}

/// Build the stroke integrate pipeline (`integrate` shader) — DESIGN §6.2/§6.1. A
/// fullscreen pass with four sampled tiles (base/scratch color/aux), writing the
/// color+aux MRT of a fresh tile.
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
            load_tex(0), // base color
            load_tex(1), // base aux
            load_tex(2), // scratch color
            load_tex(3), // scratch aux
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
