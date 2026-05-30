//! The brush engine: stamp-based stroke rasterization with copy-on-write tiles
//! (DESIGN.md §6.2, §5.2, §6.6, §6.7).
//!
//! [`StrokeRenderer::render`] takes a layer's current tile map and a recorded
//! stroke, and returns a *new* tile map in which only the touched tiles are
//! replaced. The same path serves live painting, history replay, and golden
//! tests.
//!
//! The renderer is parameterized by a [`ColorSpace`]: the color/aux texture
//! formats, the deposit blends, the brush color → channel mapping, and the stamp
//! shader all come from it. Each stamp writes two render targets in one draw
//! (color channels + `aux`), is rotated to the stroke tangent, and uses either a
//! procedural disc or a sampled coverage mask (§6.6).
//!
//! It holds only immutable GPU objects plus `Arc`-backed handles, so it is cheap
//! to `Clone` and can live inside the `Action::Context` (§5).

use std::collections::BTreeSet;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use rpds::HashTrieMap;
use wgpu::util::DeviceExt;

use crate::assets::AssetStore;
use crate::colorspace::ColorSpace;
use crate::command::InputSample;
use crate::document::action::lerp_sample;
use crate::document::{BrushParams, BrushShape, StrokeRecord};
use crate::geom::{TileCoord, Vec2, TILE_SIZE};
use crate::gpu::context::GpuContext;
use crate::gpu::tile::{TileHandle, TilePool};

/// One brush dab placed along the stroke path.
#[derive(Copy, Clone)]
struct Stamp {
    center: Vec2,
    radius: f32,
    hardness: f32,
    flow: f32,
    height: f32,
    wet: f32,
    /// Color-space channels (e.g. Oklab `L,a,b,_`, or four pigments).
    ch: [f32; 4],
    /// Orientation as (cos θ, sin θ).
    rot: [f32; 2],
}

/// Per-stamp instance data for the stamp shader (color-space-agnostic layout).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct StampInstance {
    center: [f32; 2],
    shape: [f32; 4], // radius, hardness, flow, unused
    ch: [f32; 4],    // color-space channels
    aux: [f32; 2],   // height, wet
    rot: [f32; 2],   // cos, sin
}

/// Per-tile uniform: tile origin, canvas→NDC scale, and the shape mode.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TileXform {
    params: [f32; 4], // origin.x, origin.y, 2/TILE_SIZE, mode (0=round, 1=mask)
}

#[derive(Clone)]
pub struct StrokeRenderer {
    ctx: GpuContext,
    color_space: Arc<dyn ColorSpace>,
    pipeline: wgpu::RenderPipeline,
    uniform_bgl: wgpu::BindGroupLayout,
    mask_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// 1×1 white mask bound for `Round` (procedural; the texture is unused).
    dummy_mask: wgpu::TextureView,
}

impl StrokeRenderer {
    pub fn new(ctx: &GpuContext, color_space: Arc<dyn ColorSpace>) -> Self {
        let device = &ctx.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark stamp"),
            source: wgpu::ShaderSource::Wgsl(color_space.stamp_shader().into()),
        });

        let uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stark stamp uniform bgl"),
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

        let mask_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stark stamp mask bgl"),
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

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("stark stamp layout"),
            bind_group_layouts: &[Some(&uniform_bgl), Some(&mask_bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("stark stamp pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<StampInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x2, 1 => Float32x4, 2 => Float32x4, 3 => Float32x2, 4 => Float32x2
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

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("stark stamp sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let dummy = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("stark dummy mask"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        ctx.queue.write_texture(
            dummy.as_image_copy(),
            &[255u8],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(1),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let dummy_mask = dummy.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            ctx: ctx.clone(),
            color_space,
            pipeline,
            uniform_bgl,
            mask_bgl,
            sampler,
            dummy_mask,
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
        let stamps = generate_stamps(rec, channels);
        if stamps.is_empty() {
            return base.clone();
        }

        let coords = affected_tiles(&stamps);
        let device = &self.ctx.device;

        let (mask_view, mode) = match rec.brush.shape {
            BrushShape::Round => (self.dummy_mask.clone(), 0.0_f32),
            BrushShape::Stamp(id) => match assets.mask_view(id) {
                Some(view) => (view, 1.0),
                None => (self.dummy_mask.clone(), 0.0),
            },
        };
        let mask_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark stamp mask bg"),
            layout: &self.mask_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&mask_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        let instances: Vec<StampInstance> = stamps
            .iter()
            .map(|s| StampInstance {
                center: s.center.to_array(),
                shape: [s.radius, s.hardness, s.flow, 0.0],
                ch: s.ch,
                aux: [s.height, s.wet],
                rot: s.rot,
            })
            .collect();
        let instance_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stark stamp instances"),
            contents: bytemuck::cast_slice(&instances),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stark stroke commit"),
        });

        let mut new_map = base.clone();
        for coord in coords {
            let dst = pool.acquire();

            let (color_load, aux_load) = match base.get(&coord) {
                Some(src) => {
                    let extent = wgpu::Extent3d {
                        width: TILE_SIZE,
                        height: TILE_SIZE,
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

            let origin = coord.origin();
            let xform = TileXform {
                params: [origin.x, origin.y, 2.0 / TILE_SIZE as f32, mode],
            };
            let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stark stamp xform"),
                contents: bytemuck::bytes_of(&xform),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("stark stamp bg"),
                layout: &self.uniform_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ubuf.as_entire_binding(),
                }],
            });

            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("stark stamp pass"),
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
                pass.set_bind_group(1, &mask_bg, &[]);
                pass.set_vertex_buffer(0, instance_buf.slice(..));
                pass.draw(0..4, 0..instances.len() as u32);
            }

            new_map = new_map.insert(coord, dst);
        }

        self.ctx.queue.submit([encoder.finish()]);
        new_map
    }
}

/// Place stamps along the resampled path at even arc-length spacing, depleting
/// the load reservoir with distance and orienting each stamp (DESIGN.md §6.2).
fn generate_stamps(rec: &StrokeRecord, channels: [f32; 4]) -> Vec<Stamp> {
    let b = &rec.brush;
    let spacing = (b.radius * b.spacing).max(0.5);

    // Expand the fitted control points into a smooth fine polyline (§6.2), then
    // stamp along it. Smooth path + smooth tangents = no stair-step, continuous.
    let pts = crate::path::flatten(&rec.path, crate::path::FLATTEN_STEP);

    let mut out = Vec::new();
    let Some(first) = pts.first() else {
        return out;
    };

    let dir_of = |a: Vec2, c: Vec2| {
        let d = c - a;
        if d.length() > 1e-4 {
            d.normalize()
        } else {
            Vec2::ZERO
        }
    };

    let first_dir = pts.get(1).map_or(Vec2::ZERO, |s| dir_of(first.pos, s.pos));
    let mut idx = 0u32;
    out.push(make_stamp(rec.seed, b, channels, first, 0.0, first_dir, idx));
    idx += 1;

    let mut seg_start = 0.0f32;
    let mut next_at = spacing;
    for w in pts.windows(2) {
        let (a, c) = (&w[0], &w[1]);
        let len = (c.pos - a.pos).length();
        if len < 1e-4 {
            continue;
        }
        let dir = dir_of(a.pos, c.pos);
        let seg_end = seg_start + len;
        while next_at <= seg_end {
            let t = (next_at - seg_start) / len;
            out.push(make_stamp(rec.seed, b, channels, &lerp_sample(a, c, t), next_at, dir, idx));
            idx += 1;
            next_at += spacing;
        }
        seg_start = seg_end;
    }
    out
}

fn make_stamp(
    seed: u64,
    b: &BrushParams,
    channels: [f32; 4],
    s: &InputSample,
    dist: f32,
    dir: Vec2,
    idx: u32,
) -> Stamp {
    let load = (1.0 - b.drain * dist).max(0.0);
    let base_angle = if b.follow_path && dir != Vec2::ZERO {
        dir.y.atan2(dir.x)
    } else {
        0.0
    };
    let angle = base_angle + jitter_unit(seed, idx) * b.angle_jitter;
    Stamp {
        center: s.pos,
        radius: (b.radius * s.pressure).max(0.5),
        hardness: b.hardness,
        // Fold brush opacity (color alpha) into per-stamp coverage.
        flow: b.flow * load * b.color[3],
        height: b.height * load,
        wet: b.wetness * load,
        ch: channels,
        rot: [angle.cos(), angle.sin()],
    }
}

/// Deterministic per-stamp jitter in [-1, 1] (splitmix64 of seed + index).
fn jitter_unit(seed: u64, i: u32) -> f32 {
    let mut z = seed.wrapping_add((i as u64).wrapping_mul(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    ((z >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
}

/// The set of tiles any stamp footprint overlaps. A rotated stamp's extent is
/// bounded by its diagonal, so expand by `radius·√2`.
fn affected_tiles(stamps: &[Stamp]) -> BTreeSet<TileCoord> {
    let tile = TILE_SIZE as f32;
    let mut coords = BTreeSet::new();
    for s in stamps {
        let reach = s.radius * std::f32::consts::SQRT_2;
        let min = s.center - Vec2::splat(reach);
        let max = s.center + Vec2::splat(reach);
        let (x0, x1) = ((min.x / tile).floor() as i32, (max.x / tile).floor() as i32);
        let (y0, y1) = ((min.y / tile).floor() as i32, (max.y / tile).floor() as i32);
        for y in y0..=y1 {
            for x in x0..=x1 {
                coords.insert(TileCoord::new(x, y));
            }
        }
    }
    coords
}
