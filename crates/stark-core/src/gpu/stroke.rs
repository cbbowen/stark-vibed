//! The brush engine: stamp-based stroke rasterization with copy-on-write tiles
//! (DESIGN.md §6.2, §5.2).
//!
//! [`StrokeRenderer::render`] takes a layer's current tile map and a recorded
//! stroke, and returns a *new* tile map in which only the touched tiles are
//! replaced by freshly painted tiles — every untouched tile is shared with the
//! input. The same path serves live painting, history replay, and golden tests,
//! so they can never diverge (DESIGN.md §1).
//!
//! Each stamp writes all tile channels in a single multiple-render-target draw:
//! premultiplied Oklab color (blended "over") and `(height, wet)` aux (blended
//! additively). A CPU-side **load reservoir** depletes along the path so paint
//! thins as it runs out. (True bidirectional canvas pickup — the brush lifting
//! color it passes over — needs per-stamp canvas sampling and is a later
//! refinement; DESIGN.md §6.2.)
//!
//! The renderer holds only immutable GPU objects plus `Arc`-backed handles, so
//! it is cheap to `Clone` and can live inside the `Action::Context` (§5).

use std::collections::BTreeSet;

use bytemuck::{Pod, Zeroable};
use rpds::HashTrieMap;
use wgpu::util::DeviceExt;

use crate::color;
use crate::command::InputSample;
use crate::document::action::lerp_sample;
use crate::document::StrokeRecord;
use crate::geom::{TileCoord, Vec2, TILE_SIZE};
use crate::gpu::context::GpuContext;
use crate::gpu::tile::{TileHandle, TilePool, AUX_FORMAT, COLOR_FORMAT};

/// One brush dab placed along the stroke path.
#[derive(Copy, Clone)]
struct Stamp {
    center: Vec2,
    radius: f32,
    hardness: f32,
    flow: f32,
    height: f32,
    wet: f32,
    oklab: [f32; 3],
}

/// Per-stamp instance data for `stamp.wesl`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct StampInstance {
    center: [f32; 2],
    shape: [f32; 4], // radius, hardness, flow, height
    color: [f32; 4], // okL, oka, okb, wet
}

/// Per-tile uniform for the stamp shader: tile origin + canvas→NDC scale.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TileXform {
    params: [f32; 4], // origin.x, origin.y, 2/TILE_SIZE, unused
}

#[derive(Clone)]
pub struct StrokeRenderer {
    ctx: GpuContext,
    pipeline: wgpu::RenderPipeline,
    uniform_bgl: wgpu::BindGroupLayout,
}

impl StrokeRenderer {
    pub fn new(ctx: &GpuContext) -> Self {
        let device = &ctx.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark stamp"),
            source: wgpu::ShaderSource::Wgsl(stark_shaders::stamp().into()),
        });

        let uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stark stamp uniform bgl"),
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
            label: Some("stark stamp layout"),
            bind_group_layouts: &[Some(&uniform_bgl)],
            immediate_size: 0,
        });

        // Premultiplied "over" for color; additive for the (height, wet) aux.
        let over = wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING;
        let add = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };

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
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4, 2 => Float32x4],
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
                        format: COLOR_FORMAT,
                        blend: Some(over),
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

        Self {
            ctx: ctx.clone(),
            pipeline,
            uniform_bgl,
        }
    }

    /// Render `rec` over `base`, returning a copy-on-write tile map: only tiles
    /// the stroke touches are new; the rest are shared `Arc`s with `base`.
    pub fn render(
        &self,
        pool: &TilePool,
        base: &HashTrieMap<TileCoord, TileHandle>,
        rec: &StrokeRecord,
    ) -> HashTrieMap<TileCoord, TileHandle> {
        let stamps = generate_stamps(rec);
        if stamps.is_empty() {
            return base.clone();
        }

        let coords = affected_tiles(&stamps);
        let device = &self.ctx.device;

        let instances: Vec<StampInstance> = stamps
            .iter()
            .map(|s| StampInstance {
                center: s.center.to_array(),
                shape: [s.radius, s.hardness, s.flow, s.height],
                color: [s.oklab[0], s.oklab[1], s.oklab[2], s.wet],
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

            // Copy-on-write: start from the existing tile if there is one,
            // otherwise from a cleared one. Both channels are handled.
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
                params: [origin.x, origin.y, 2.0 / TILE_SIZE as f32, 0.0],
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
                pass.set_vertex_buffer(0, instance_buf.slice(..));
                // MVP: draw every stamp into each touched tile; off-tile stamps
                // are clipped. Per-tile stamp culling is a later optimization.
                pass.draw(0..4, 0..instances.len() as u32);
            }

            new_map = new_map.insert(coord, dst);
        }

        self.ctx.queue.submit([encoder.finish()]);
        new_map
    }
}

/// Place stamps along the resampled path at even arc-length spacing, depleting
/// the load reservoir with distance travelled (DESIGN.md §6.2).
fn generate_stamps(rec: &StrokeRecord) -> Vec<Stamp> {
    let b = &rec.brush;
    let oklab = color::srgb_to_oklab(b.color);
    let oklab = [oklab[0], oklab[1], oklab[2]];
    let spacing = (b.radius * b.spacing).max(0.5);

    let mut out = Vec::new();
    let Some(first) = rec.path.first() else {
        return out;
    };

    let make = |s: &InputSample, dist: f32| -> Stamp {
        let load = (1.0 - b.drain * dist).max(0.0);
        Stamp {
            center: s.pos,
            radius: (b.radius * s.pressure).max(0.5),
            hardness: b.hardness,
            flow: b.flow * load,
            height: b.height * load,
            wet: b.wetness * load,
            oklab,
        }
    };

    out.push(make(first, 0.0));
    let mut seg_start = 0.0f32; // arc length at the start of the current segment
    let mut next_at = spacing; // arc length of the next stamp
    for w in rec.path.windows(2) {
        let (a, c) = (&w[0], &w[1]);
        let len = (c.pos - a.pos).length();
        if len < 1e-4 {
            continue;
        }
        let seg_end = seg_start + len;
        while next_at <= seg_end {
            let t = (next_at - seg_start) / len;
            out.push(make(&lerp_sample(a, c, t), next_at));
            next_at += spacing;
        }
        seg_start = seg_end;
    }
    out
}

/// The set of tiles any stamp footprint overlaps.
fn affected_tiles(stamps: &[Stamp]) -> BTreeSet<TileCoord> {
    let tile = TILE_SIZE as f32;
    let mut coords = BTreeSet::new();
    for s in stamps {
        let min = s.center - Vec2::splat(s.radius);
        let max = s.center + Vec2::splat(s.radius);
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
