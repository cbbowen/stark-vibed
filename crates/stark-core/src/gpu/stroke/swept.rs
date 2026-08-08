//! The swept fast path (§6.2): one quad per segment, coverage integrated
//! along the sweep through a precomputed prefix-τ texture, over-blended so optical
//! depth sums exactly.
//!
//! Carries no brush state between segments, which is what makes it the fast path —
//! a range needs nothing from its predecessor but the arc length.

use crate::document::StrokeRecord;
use crate::geom::{TILE_APRON, TILE_TEX};
use crate::gpu::desc;
use crate::gpu::tile::{AllocSource, TileMap};

use super::segments::{SegmentInstance, affected_tiles, generate_segments_in};
use super::{
    ScopedResources, StrokeCarry, StrokeRenderer, StrokeScene, StrokeSpans, UNIFORM_STRIDE,
};

// Vertices in one segment's swept geometry: a triangle strip of two rims across
// `SWEEP_SLICES` steps along the travel, since a segment's centreline is an arc rather
// than a chord (§6.2). Generated from `stamp_common.wesl`, which is where the strip is
// actually built — asking for fewer would clip the sweep short, more would fold the
// strip back over itself.
use stark_shaders::mirror::stamp_common::{SWEEP_SLICES, SWEEP_VERTS};

/// The draw call and the strip agree on the vertex count.
///
/// Both numbers are the shader's now, so this is the shader's own invariant rather
/// than a boundary check — and it holds at compile time, where the runtime test that
/// scraped `SWEEP_SLICES` out of the linked source used to.
const _: () = assert!(
    SWEEP_VERTS == 2 * (SWEEP_SLICES + 1),
    "the sweep strip's slice count and its vertex count have diverged",
);

// The per-tile uniform, generated from `stamp_common.wesl`'s own declaration
// (§6.7): the tile *texture's* top-left in canvas px + canvas→NDC scale, plus the
// brush's stroke-constant colour channels.
use stark_shaders::mirror::stamp_common::TileXform;

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
    ) -> (TileMap, StrokeCarry) {
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
            entries: &[desc::tex(0, &prefix_view)],
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
                desc::tex(0, &noise_view),
                desc::samp(1, &self.noise_sampler),
                desc::tex(2, &surface.view),
                desc::samp(3, &surface.sampler),
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

        // Footprint → cleared scratch tile: within-stroke accumulation of the parcel
        // this stroke lays (the color target over-blends the parcel's visible alpha
        // with the latent premultiplied by it, the aux accumulates its height and
        // optical mass additively). The scratch aux is the wide format.
        //
        // **One pair for the whole stroke, released only after the submit.** Sharing
        // it across tiles is sound because every sweep pass below clears both targets,
        // so no tile can see what the tile before it left; what makes it *necessary*
        // is that nothing in a recorded encoder has run yet. A pair acquired per tile
        // and dropped at the end of its iteration goes back on the pool's free list
        // while the passes naming it are still only recorded — and the free list is
        // where `TilePool::trim` takes from, tail first, on any `acquire_tex` that
        // happens to end an epoch. Destroying a texture this command buffer names
        // fails the submit, so every destination tile in it keeps whatever paint the
        // pool last had there: one frame of other tiles' work, gone on the next
        // render. Same rule as `transform::Recording`, and for the same reason.
        let scratch = self.acquire_scratch(pool, AllocSource::StrokeScratch);

        let mut new_map = base.clone();
        for (i, coord) in coords.iter().enumerate() {
            let xform_off = (i * UNIFORM_STRIDE) as u32;

            // This tile's segments into the shared scratch, cleared as it goes.
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("stark sweep pass"),
                    color_attachments: &[
                        Some(desc::attach(scratch.color_view(), desc::CLEAR)),
                        Some(desc::attach(scratch.aux_view(), desc::CLEAR)),
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
            // The layer's resident paint here, or the 1×1 zero where it has none —
            // the integrate clamps its loads, so bare canvas costs no tile at all
            // (§6.8's pattern). This used to acquire a whole pooled pair and clear
            // it on every pointer move, whether or not the stroke reached anything
            // unpainted.
            let (base_color, base_aux) = match base.get(coord) {
                Some(tile) => (tile.color_view(), tile.aux_view()),
                None => (&self.zeroes.color, &self.zeroes.aux),
            };
            let mask_view = self.selection.mask_for(selection, *coord);
            let integrate_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("stark integrate bg"),
                layout: &self.integrate_bgl,
                entries: &[
                    desc::tex(0, base_color),
                    desc::tex(1, base_aux),
                    desc::tex(2, scratch.color_view()),
                    desc::tex(3, scratch.aux_view()),
                    desc::tex(4, &mask_view),
                ],
            });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("stark integrate"),
                    color_attachments: &[
                        Some(desc::attach(dst.color_view(), desc::CLEAR)),
                        Some(desc::attach(dst.aux_view(), desc::CLEAR)),
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
        // And the scratch pair after it, for the stronger reason given where it is
        // acquired: released any earlier it is a *pooled* texture this command buffer
        // still names, free to be handed out — or destroyed — before the submit.
        drop(scratch);
        (new_map, carry)
    }
}

// `the_draw_call_and_the_strip_agree_on_the_vertex_count` stood here. It had to check
// through `SWEEP_SLICES` rather than the shader's own `SWEEP_VERTS`, because the
// shader states that one for the host's benefit and never computes with it — so the
// linker stripped it and the check could not see it. Reading the *unlinked* source
// retires that limitation, and the assertion above holds at compile time.
