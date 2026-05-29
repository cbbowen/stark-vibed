//! The brush engine: stamp-based stroke rasterization with copy-on-write tiles
//! (DESIGN.md §6.2, §5.2).
//!
//! [`StrokeRenderer::render`] takes a layer's current tile map and a recorded
//! stroke, and returns a *new* tile map in which only the touched tiles are
//! replaced by freshly painted tiles — every untouched tile is shared with the
//! input. This is the same path used for live painting, history replay, and
//! golden tests, so the three can never diverge (DESIGN.md §1).
//!
//! The renderer holds only immutable GPU objects (pipeline, layouts, sampler)
//! plus `Arc`-backed handles, so it is cheap to `Clone` — which lets it live
//! inside the `Action::Context` (DESIGN.md §5). Per-stroke buffers are
//! allocated transiently; commits are far rarer than frames.

use std::collections::BTreeSet;

use bytemuck::{Pod, Zeroable};
use rpds::HashTrieMap;
use wgpu::util::DeviceExt;

use crate::document::action::lerp_sample;
use crate::document::StrokeRecord;
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
    color: [f32; 4],
}

/// Per-stamp instance data for the stamp shader (`stamp.wesl`).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct StampInstance {
    center: [f32; 2],
    shape: [f32; 4], // radius, hardness, flow, unused
    color: [f32; 4],
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
                targets: &[Some(wgpu::ColorTargetState {
                    format: crate::gpu::tile::COLOR_FORMAT,
                    // Premultiplied "over": stamps accumulate into the tile.
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
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
                shape: [s.radius, s.hardness, s.flow, 0.0],
                color: s.color,
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
            // otherwise from transparent.
            let load = match base.get(&coord) {
                Some(src) => {
                    encoder.copy_texture_to_texture(
                        src.color().as_image_copy(),
                        dst.color().as_image_copy(),
                        wgpu::Extent3d {
                            width: TILE_SIZE,
                            height: TILE_SIZE,
                            depth_or_array_layers: 1,
                        },
                    );
                    wgpu::LoadOp::Load
                }
                None => wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
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
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: dst.color_view(),
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
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

/// Place stamps along the resampled path at even arc-length spacing (DESIGN.md §6.2).
fn generate_stamps(rec: &StrokeRecord) -> Vec<Stamp> {
    let b = &rec.brush;
    let spacing = (b.radius * b.spacing).max(0.5);
    let mut out = Vec::new();
    let Some(first) = rec.path.first() else {
        return out;
    };

    let mut push = |sample: &crate::command::InputSample| {
        out.push(Stamp {
            center: sample.pos,
            radius: (b.radius * sample.pressure).max(0.5),
            hardness: b.hardness,
            flow: b.flow,
            color: b.color,
        });
    };

    push(first);
    let mut carry = 0.0f32; // distance already covered toward the next stamp
    for w in rec.path.windows(2) {
        let (a, c) = (&w[0], &w[1]);
        let seg = c.pos - a.pos;
        let len = seg.length();
        if len < 1e-4 {
            continue;
        }
        let mut dist = spacing - carry;
        while dist <= len {
            let sample = lerp_sample(a, c, dist / len);
            push(&sample);
            dist += spacing;
        }
        carry = len - (dist - spacing);
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
