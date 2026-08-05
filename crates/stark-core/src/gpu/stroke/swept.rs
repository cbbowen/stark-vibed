//! The swept fast path (§6.2): one quad per segment, coverage integrated
//! along the sweep through a precomputed prefix-τ texture, over-blended so optical
//! depth sums exactly.
//!
//! Carries no brush state between segments, which is what makes it the fast path —
//! a range needs nothing from its predecessor but the arc length.

use bytemuck::{Pod, Zeroable};
use rpds::HashTrieMap;

use crate::document::StrokeRecord;
use crate::geom::{TILE_APRON, TILE_TEX, TileCoord};
use crate::gpu::tile::{AllocSource, TilePairHandle};

use super::segments::{SegmentInstance, affected_tiles, generate_segments_in};
use super::{
    ScopedResources, StrokeCarry, StrokeRenderer, StrokeScene, StrokeSpans, UNIFORM_STRIDE,
};

/// Vertices in one segment's swept geometry: a triangle strip of two rims across
/// `SWEEP_SLICES` steps along the travel, since a segment's centreline is an arc
/// rather than a chord (§6.2). Must match `SWEEP_VERTS` in
/// `stamp_common.wesl`, which is where the strip is actually built — asking for fewer
/// would clip the sweep short, more would fold the strip back over itself.
const SWEEP_VERTS: u32 = 18;

/// Per-tile uniform: the tile *texture's* top-left in canvas px + canvas→NDC
/// scale, plus the brush's stroke-constant colour channels. The texture origin is
/// the interior origin minus the apron, so the stroke rasterizes into the apron
/// too (keeping it consistent with the neighbor's interior — see
/// [`crate::geom::TILE_APRON`]).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TileXform {
    params: [f32; 4], // tex_origin.x, tex_origin.y, 2/TILE_TEX, _
    color: [f32; 4],  // brush channels (.xyz), per-unit opacity (.w)
    /// The `drain` falloff per canvas px in `.x`, and the canvas → surface-tile uv
    /// scale in `.y` (0 on a ground with no relief, which makes the tooth exactly 1);
    /// `.zw` unused.
    ///
    /// These are here, and the `add` source is not, because of what each is a
    /// function of. `drain` depends on arc length and the tooth on canvas position —
    /// both things the fragment already knows — so each is recovered there and is a
    /// stroke constant in the only sense that matters. `add` is a *pen*-driven
    /// quantity now (§6.2) and varies segment to segment like the radius does, so it
    /// rides the instance, as does the brush's `tooth`.
    paint: [f32; 4],
    noise_freq: [f32; 4], // colour-dynamics frequency (across, along), 1/NOISE_TILE_PX, _
    noise_amp: [f32; 4],  // per colour-channel noise amplitude, _
    noise_off: [f32; 4],  // per-stroke noise lookup translation (2), _, _
}

/// One tile's window into the stroke's transform buffer — the `min_binding_size` the
/// sweep's layout declares, taken from the struct rather than written down.
pub(super) const XFORM_SLOT: u64 = std::mem::size_of::<TileXform>() as u64;

impl StrokeRenderer {
    /// [`Self::render_range`] through the plain swept fast path: no carried brush
    /// state at all, so a range needs nothing from its predecessor but the arc length.
    /// `tol` comes from [`dynamics_setup`](super::dynamics::dynamics_setup), which has
    /// already decided — from the brush — that this stroke takes the fast path, or
    /// that the loop cannot draw it. Handed over rather than recomputed, so one place
    /// answers what a stroke's segments are.
    pub(super) fn render_swept(
        &self,
        scene: StrokeScene<'_>,
        rec: &StrokeRecord,
        spans: StrokeSpans,
        tol: crate::path::FlattenTolerance,
    ) -> (HashTrieMap<TileCoord, TilePairHandle>, StrokeCarry) {
        let StrokeScene {
            pool,
            assets,
            base,
            selection,
            surface,
        } = scene;
        // Everything both paths share, resolved once (see [`StrokeConstants`]).
        let k = self.stroke_constants(rec, surface);
        let (segments, end_dist) = generate_segments_in(rec, tol, spans);
        if segments.is_empty() {
            return (
                base.clone(),
                StrokeCarry {
                    dist: end_dist,
                    tool: None,
                    dirty: Vec::new(),
                },
            );
        }

        // The per-stroke instance buffer registers here and is `destroy()`d when this
        // drops (at the end of `render`, after the submit below) — freeing it
        // deterministically instead of leaking to JS GC (§6.2).
        let mut scoped = ScopedResources::default();

        // Resolve the brush's prefix-τ texture: image brushes from the asset
        // store; the round tip generated (and cached) from its hardness.
        let prefix_view = self.prefix_view(assets, &rec.brush);

        let device = &self.ctx.device;
        let prefix_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark sweep prefix bg"),
            layout: &self.prefix_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&prefix_view),
            }],
        });

        // Colour dynamics (§6.2): the noise tile for this brush and
        // the stroke's lookup parameters. An inactive brush binds the zero
        // tile with zero amplitudes — the deposit is exactly the constant
        // colour.
        let noise_view = self.noise_view(&rec.brush.color_dynamics);
        // The canvas ground beside it (§6.4): the deposition tooth's height and the
        // rise ahead of it, in the same group because it is the same kind of thing —
        // a field the deposit samples per fragment.
        let noise_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark sweep noise bg"),
            layout: &self.noise_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&noise_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.noise_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&surface.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&surface.sampler),
                },
            ],
        });
        let instances: Vec<SegmentInstance> = segments
            .iter()
            .map(|s| SegmentInstance {
                start: s.start.to_array(),
                dir: s.dir.to_array(),
                geom: [s.radius, s.length],
                extra: [s.orient, s.dist, s.curvature, s.add],
                tooth: s.tooth,
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
        self.ctx
            .queue
            .write_buffer(&instance_buf, 0, instance_bytes);

        let coords = affected_tiles(&segments);
        let carry = StrokeCarry {
            dist: end_dist,
            tool: None,
            dirty: coords.iter().copied().collect(),
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stark stroke commit"),
        });

        // Every brush rasterizes its footprint into a *cleared scratch* tile, then the
        // integrate pass merges it over the base into a fresh CoW tile
        // (§6.2/§6.1). `empty` (cleared) stands in as the base wherever the stroke
        // touches bare canvas — acquired tiles are undefined, so clear it once here.
        let clear = wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            store: wgpu::StoreOp::Store,
        };
        let empty = self.acquire_tile(pool, AllocSource::IntegrateEmptyBase);
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

        // Per-tile sweep transforms, one [`UNIFORM_STRIDE`] slot each in a single
        // buffer the draws below select with a dynamic offset. The texture top-left is
        // the interior origin shifted out by the apron, so the full TILE_TEX target
        // maps to NDC [-1, 1]; everything else is a stroke constant, repeated per slot
        // because the slot is what the shader reads.
        //
        // One buffer and one bind group for the stroke, not one of each per tile: this
        // path redraws on every pointer move, and the allocation *rate* is what OOMs
        // the tab (see [`ScopedResources`] and [`UNIFORM_STRIDE`]).
        let apron = TILE_APRON as f32;
        let mut xform_data = vec![0u8; coords.len() * UNIFORM_STRIDE];
        for (i, coord) in coords.iter().enumerate() {
            let origin = coord.origin();
            let xform = TileXform {
                params: [
                    origin.x - apron,
                    origin.y - apron,
                    2.0 / TILE_TEX as f32,
                    0.0,
                ],
                color: k.channels,
                paint: [rec.brush.drain, k.grain_uv, 0.0, 0.0],
                noise_freq: k.nfreq,
                noise_amp: k.namp,
                noise_off: k.noff,
            };
            let at = i * UNIFORM_STRIDE;
            xform_data[at..at + XFORM_SLOT as usize].copy_from_slice(bytemuck::bytes_of(&xform));
        }
        let xform_buf = scoped.buffer(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stark sweep xforms"),
            size: xform_data.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.ctx.queue.write_buffer(&xform_buf, 0, &xform_data);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark sweep bg"),
            layout: &self.uniform_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &xform_buf,
                    offset: 0,
                    size: wgpu::BufferSize::new(XFORM_SLOT),
                }),
            }],
        });

        let mut new_map = base.clone();
        for (i, coord) in coords.iter().enumerate() {
            let xform_off = (i * UNIFORM_STRIDE) as u32;

            // Footprint → cleared scratch tile: within-stroke accumulation of the
            // parcel this stroke lays (the color target over-blends the parcel's
            // visible alpha with the latent premultiplied by it, the aux accumulates
            // its height and optical mass additively). The scratch aux is the wide
            // format.
            let scratch = self.acquire_scratch(pool, AllocSource::StrokeScratch);
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
                pass.set_bind_group(0, &bind_group, &[xform_off]);
                pass.set_bind_group(1, &prefix_bg, &[]);
                pass.set_bind_group(2, &noise_bg, &[]);
                pass.set_vertex_buffer(0, instance_buf.slice(..));
                pass.draw(0..SWEEP_VERTS, 0..instances.len() as u32);
            }

            // Integrate the scratch slab over the base into a fresh CoW tile, gated
            // by this tile's selection coverage — its own mask if it has one, or the
            // 1×1 constant standing in for the rest of the canvas (§6.8).
            let dst = self.acquire_tile(pool, AllocSource::IntegrateDestination);
            let base_tile = base.get(coord).unwrap_or(&empty);
            let mask_view = self.selection.mask_for(selection, *coord);
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
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&mask_view),
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
        (new_map, carry)
    }
}
