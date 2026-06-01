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
use crate::document::{BrushDynamics, BrushShape, MixerParams, StrokeRecord};
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

/// Largest base-region edge (canvas px) composited for wet-mixing pickup. Strokes
/// whose bounding box exceeds this skip pickup for that stroke — rare, and it
/// bounds the transient GPU memory of the region texture (DESIGN.md §6.2).
const MAX_REGION_DIM: u32 = 2048;

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
    ch: [f32; 4],
}

/// Per-segment instance data for the sweep shader. Padded to 64 bytes so the
/// same buffer is a valid `std430 array<Instance>` for the wet-mixing compute
/// pass (which patches `ch` in place — see `mixer.wesl`).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SegmentInstance {
    start: [f32; 2],
    dir: [f32; 2],    // unit tangent
    geom: [f32; 4],   // radius, length, flow, unused
    ch: [f32; 4],     // color-space channels
    aux: [f32; 2],    // height, wet
    _pad: [f32; 2],   // → 64 B, vec4 std430 alignment
}

/// Per-tile uniform: the tile *texture's* top-left in canvas px + canvas→NDC
/// scale. The texture origin is the interior origin minus the apron, so the
/// stroke rasterizes into the apron too (keeping it consistent with the
/// neighbor's interior — see [`crate::geom::TILE_APRON`]).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TileXform {
    params: [f32; 4], // tex_origin.x, tex_origin.y, 2/TILE_TEX, unused
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
    knobs: [f32; 4],       // pickup, color_inject, flatten_step, capacity
    counts: [u32; 4],      // segment_count, color_space, _, _
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

    // Wet-mixing pickup (Mixer dynamics): composite the base into a region, then a
    // serial compute scan patches per-segment color — all on the GPU, no readback
    // (DESIGN.md §6.2). Built once; cheap to clone (wgpu handles are Arc-backed).
    composite_pipeline: wgpu::RenderPipeline,
    composite_sampler: wgpu::Sampler,
    composite_view_bgl: wgpu::BindGroupLayout,
    composite_tile_bgl: wgpu::BindGroupLayout,
    mixer_pipeline: wgpu::ComputePipeline,
    mixer_bgl: wgpu::BindGroupLayout,
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
                        0 => Float32x2, 1 => Float32x2, 2 => Float32x4, 3 => Float32x4, 4 => Float32x2
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

        Self {
            ctx: ctx.clone(),
            color_space,
            pipeline,
            uniform_bgl,
            prefix_bgl,
            round_prefix: Arc::new(Mutex::new(None)),
            surface_bg,
            composite_pipeline,
            composite_sampler,
            composite_view_bgl,
            composite_tile_bgl,
            mixer_pipeline,
            mixer_bgl,
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
        let segments = generate_segments(rec, channels);
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

        let instances: Vec<SegmentInstance> = segments
            .iter()
            .map(|s| SegmentInstance {
                start: s.start.to_array(),
                dir: s.dir.to_array(),
                geom: [s.radius, s.length, s.flow, 0.0],
                ch: s.ch,
                aux: [s.height, s.wet],
                _pad: [0.0; 2],
            })
            .collect();
        // The mixer compute pass patches `ch` in place, so the buffer is also a
        // storage target — not just vertex data.
        let instance_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stark sweep instances"),
            contents: bytemuck::cast_slice(&instances),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
        });

        let coords = affected_tiles(&segments);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stark stroke commit"),
        });

        // Wet mixing (DESIGN.md §6.2): composite the base under the stroke, then a
        // serial compute scan rewrites each segment's `ch` to the smeared color.
        // Encoded before the deposit so the patched colors are what gets stamped.
        if let BrushDynamics::Mixer(mp) = rec.brush.dynamics {
            self.encode_mixer(&mut encoder, base, &coords, &segments, &instance_buf, channels, mp);
        }

        let mut new_map = base.clone();
        for coord in coords {
            let dst = pool.acquire();

            let (color_load, aux_load) = match base.get(&coord) {
                Some(src) => {
                    let extent = wgpu::Extent3d {
                        width: TILE_TEX,
                        height: TILE_TEX,
                        depth_or_array_layers: 1,
                    };
                    encoder.copy_texture_to_texture(
                        src.color().as_image_copy(),
                        dst.color().as_image_copy(),
                        extent,
                    );
                    encoder.copy_texture_to_texture(
                        src.aux().as_image_copy(),
                        dst.aux().as_image_copy(),
                        extent,
                    );
                    (wgpu::LoadOp::Load, wgpu::LoadOp::Load)
                }
                None => (
                    wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                ),
            };

            // Texture top-left = interior origin shifted out by the apron, so the
            // full TILE_TEX target (interior + apron) maps to NDC [-1, 1].
            let apron = TILE_APRON as f32;
            let origin = coord.origin();
            let xform = TileXform {
                params: [origin.x - apron, origin.y - apron, 2.0 / TILE_TEX as f32, 0.0],
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

            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("stark sweep pass"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: dst.color_view(),
                            resolve_target: None,
                            depth_slice: None,
                            ops: wgpu::Operations {
                                load: color_load,
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: dst.aux_view(),
                            resolve_target: None,
                            depth_slice: None,
                            ops: wgpu::Operations {
                                load: aux_load,
                                store: wgpu::StoreOp::Store,
                            },
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

            new_map = new_map.insert(coord, dst);
        }

        self.ctx.queue.submit([encoder.finish()]);
        new_map
    }

    /// The round tip's prefix-τ texture for a given `hardness`, cached so live
    /// preview (which re-renders per pointer move) doesn't rebuild it each frame.
    fn round_prefix(&self, hardness: f32) -> wgpu::TextureView {
        let key = hardness.to_bits();
        let mut cache = self.round_prefix.lock().expect("round prefix poisoned");
        if let Some((k, view)) = cache.as_ref() {
            if *k == key {
                return view.clone();
            }
        }
        let coverage = round_coverage(hardness, ROUND_RES);
        let (_tex, view) = build_prefix_tau(&self.ctx, ROUND_RES, ROUND_RES, &coverage);
        *cache = Some((key, view.clone()));
        view
    }

    /// Wet-mixing pickup (DESIGN.md §6.2), fully on the GPU. Pass A composites the
    /// base under the stroke into a 1:1 region; Pass B is a single serial compute
    /// scan that lifts wet paint into a reservoir and patches each segment's `ch`
    /// in `instance_buf`. No CPU readback — works on WebGPU. No-op if the stroke's
    /// bbox exceeds [`MAX_REGION_DIM`].
    #[allow(clippy::too_many_arguments)]
    fn encode_mixer(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        base: &HashTrieMap<TileCoord, TileHandle>,
        coords: &BTreeSet<TileCoord>,
        segments: &[Segment],
        instance_buf: &wgpu::Buffer,
        channels: [f32; 4],
        mp: MixerParams,
    ) {
        let Some((origin, w, h)) = stroke_bbox(segments) else {
            return; // empty or too large — skip pickup for this stroke
        };
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

        // ---- Pass B: the serial reservoir scan, patching per-segment `ch`.
        let uni = MixerUniform {
            brush_ch: channels,
            origin_dims: [origin.x, origin.y, w as f32, h as f32],
            knobs: [
                mp.pickup,
                mp.color_inject,
                crate::path::FLATTEN_STEP,
                RESERVOIR_CAPACITY,
            ],
            counts: [segments.len() as u32, 0, 0, 0],
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
/// (`drain`) depletes with arc distance; radius follows pressure. Every segment
/// gets the brush's own `channels`; a [`BrushDynamics::Mixer`] brush overwrites
/// these per-segment on the GPU afterwards (`StrokeRenderer::encode_mixer`).
fn generate_segments(rec: &StrokeRecord, channels: [f32; 4]) -> Vec<Segment> {
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
            flow: b.flow * b.color[3] * load * SWEEP_FLOW_SCALE,
            height: b.height * load,
            wet: b.wetness * load,
            ch: channels,
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
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
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
