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
use crate::document::{BrushShape, StrokeRecord};
use crate::geom::{TileCoord, Vec2, TILE_APRON, TILE_SIZE, TILE_TEX};
use crate::gpu::context::GpuContext;
use crate::gpu::tile::{TileHandle, TilePool};

/// Global tuning so a default brush (`flow = 1`) reads as a solid stroke;
/// `flow` is an optical-depth-per-length rate (DESIGN.md §6.2).
const SWEEP_FLOW_SCALE: f32 = 1.0;
/// Resolution of the generated round-tip prefix texture.
const ROUND_RES: u32 = 256;

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

/// Per-segment instance data for the sweep shader.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SegmentInstance {
    start: [f32; 2],
    dir: [f32; 2],    // unit tangent
    geom: [f32; 4],   // radius, length, flow, unused
    ch: [f32; 4],     // color-space channels
    aux: [f32; 2],    // height, wet
}

/// Per-tile uniform: the tile *texture's* top-left in canvas px + canvas→NDC
/// scale. The texture origin is the interior origin minus the apron, so the
/// stroke rasterizes into the apron too (keeping it consistent with the
/// neighbor's interior — see [`crate::geom::TILE_APRON`]).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TileXform {
    params: [f32; 4], // tex_origin.x, tex_origin.y, 2/TILE_TEX, unused
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
}

impl StrokeRenderer {
    pub fn new(ctx: &GpuContext, color_space: Arc<dyn ColorSpace>) -> Self {
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

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("stark sweep layout"),
            bind_group_layouts: &[Some(&uniform_bgl), Some(&prefix_bgl)],
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

        Self {
            ctx: ctx.clone(),
            color_space,
            pipeline,
            uniform_bgl,
            prefix_bgl,
            round_prefix: Arc::new(Mutex::new(None)),
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
            })
            .collect();
        let instance_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stark sweep instances"),
            contents: bytemuck::cast_slice(&instances),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let coords = affected_tiles(&segments);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stark stroke commit"),
        });

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
/// the spline, then make each polyline edge a segment. Load reservoir depletes
/// with arc distance; radius follows pressure.
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
