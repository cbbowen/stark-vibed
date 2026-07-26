//! The brush-dynamics path (DESIGN.md §6.2): a serial swept-exchange loop that lets
//! a stroke pick paint up off the canvas and put it back down.
//!
//! Where the swept path composes by summing optical depth — and so can draw its
//! segments in any order — this one is *sequential* by nature: what the tip carries
//! into a segment is what the previous segment left on it. The loop runs on the GPU
//! (no CPU readback, so it works on WebGPU) with a per-segment x per-lateral-band
//! reservoir texture standing in for the tip's load.

use bytemuck::{Pod, Zeroable};
use rpds::HashTrieMap;
use wgpu::util::DeviceExt;

use crate::colorspace::ColorSpace;
use crate::document::{BrushShape, StrokeRecord};
use std::sync::{Arc, Mutex};

use crate::geom::{INTERIOR_UV_BIAS, INTERIOR_UV_SCALE, TILE_SIZE, TileCoord, Vec2};
use crate::gpu::tile::{AllocSource, SCRATCH_AUX_FORMAT, TilePairHandle};

use super::segments::{
    Segment, affected_tiles, generate_segments_in, generate_segments_tol, noise_uniform,
    region_dim, region_rect,
};
use super::swept::{TileInstance, ViewUniform};
use super::{
    ADD_GAIN, BAKE_FORMAT, BAKE_RES, BRUSH_RES, MAX_REGION_DIM, MAX_STAMPS, RESERVOIR_CADENCE,
    ScopedResources, StrokeCarry, StrokeRenderer, StrokeScene, StrokeSpans, TAU_PER_PASS,
    ToolState, flatten_tolerance,
};

/// Mirrors `Params` in `slice.wesl`: the tile texture's top-left in region texels.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SliceUniform {
    offset: [f32; 4],
}

/// One dispatch step of the sequential swept-exchange loop (DESIGN.md §6.2):
/// either a reservoir `pickup` or a segment's `snapshot`+`deposit` pair. `slot`
/// is the 144-byte `Stamp` uniform (see dynamics.wesl), precomputed CPU-side as
/// a pure function of the `StrokeRecord`, so replay is deterministic.
struct LoopDispatch {
    pickup: bool,
    slot: [f32; 36],
}

/// GPU objects for the brush-dynamics stamp loop (DESIGN.md §6.2), built once.
/// All handles are `Arc`-backed, so the kit is cheap to clone with its renderer.
#[derive(Clone)]
pub(super) struct DynamicsKit {
    // Region composite: base tiles → one 1:1 canvas region (colour + wide aux).
    pub(super) composite_pipeline: wgpu::RenderPipeline,
    pub(super) composite_view_bgl: wgpu::BindGroupLayout,
    pub(super) composite_tile_bgl: wgpu::BindGroupLayout,
    pub(super) composite_sampler: wgpu::Sampler,
    // The stamp-loop dispatches (one compute shader, four entry points).
    pub(super) snapshot_pipeline: wgpu::ComputePipeline,
    pub(super) snapshot_bgl: wgpu::BindGroupLayout,
    pub(super) pickup_pipeline: wgpu::ComputePipeline,
    pub(super) pickup_bgl: wgpu::BindGroupLayout,
    /// Drains the tool by what each segment takes, so its state advances with travel
    /// rather than in pickup-sized steps (`dynamics.wesl::deplete`). Shares
    /// `pickup_bgl`.
    pub(super) deplete_pipeline: wgpu::ComputePipeline,
    /// Integrates the reservoir along the segment's travel axis so the deposit can
    /// read the whole pass instead of one mid-pass sample (`dynamics.wesl::bake`).
    pub(super) bake_pipeline: wgpu::ComputePipeline,
    pub(super) bake_bgl: wgpu::BindGroupLayout,
    pub(super) deposit_pipeline: wgpu::ComputePipeline,
    pub(super) deposit_bgl: wgpu::BindGroupLayout,
    /// The deposit's prefix-τ volume binding (group 1) — the same texture the
    /// swept fast path samples, so the exchange footprint *is* the definite
    /// integral of the brush along the travel (compute-visible variant).
    pub(super) prefix_bgl: wgpu::BindGroupLayout,
    /// Bilinear clamp sampler for the region / reservoir / coverage lookups.
    pub(super) exchange_sampler: wgpu::Sampler,
    // Region → CoW tile write-back.
    pub(super) slice_pipeline: wgpu::RenderPipeline,
    pub(super) slice_bgl: wgpu::BindGroupLayout,
    /// Cached round-tip coverage texture, keyed by `hardness.to_bits()`.
    pub(super) round_cov: Arc<Mutex<Option<(u32, wgpu::TextureView)>>>,
}

impl StrokeRenderer {
    /// Render `spans` of a paint-manipulating stroke via the **sequential
    /// swept-exchange loop** (DESIGN.md §6.2): composite the base under that piece
    /// into a 1:1 region, then walk it *in order* on the GPU — the canvas-side
    /// exchange swept per flattened segment through the prefix-τ integral (the
    /// same definite-integral footprint as the plain deposit), the 2-D tool
    /// reservoir updated at `spacing · radius` cadence — and slice the evolved
    /// region back into fresh CoW tiles.
    ///
    /// The loop starts from `tool` rather than from a fresh tip when one is given, and
    /// hands back the state it ends in whenever a further range remains to be drawn,
    /// so a live stroke redraws only its tail (see [`ToolState`]). `tol` comes from
    /// [`dynamics_setup`], which has already decided — from the whole record — that
    /// this stroke runs the loop at all.
    pub(super) fn render_dynamic(
        &self,
        scene: StrokeScene<'_>,
        rec: &StrokeRecord,
        spans: StrokeSpans,
        tool: Option<&ToolState>,
        tol: crate::path::FlattenTolerance,
    ) -> (HashTrieMap<TileCoord, TilePairHandle>, StrokeCarry) {
        let StrokeScene {
            pool,
            assets,
            base,
            selection,
        } = scene;
        let rgb = [rec.brush.color[0], rec.brush.color[1], rec.brush.color[2]];
        let channels = self.color_space.rgb_to_channels(rgb);
        // Nothing follows the range that reaches the end of the stroke, so there is no
        // reason to snapshot a reservoir for it — which is the common case, since the
        // live tail is exactly that range and it re-renders every pointer move.
        let capture = spans.range.end < crate::path::span_count(rec.path.len());
        let (segments, end_dist) = generate_segments_in(rec, tol, spans);
        let since0 = tool.map_or(f32::INFINITY, |t| t.since);
        // A range with no geometry runs no dispatches, so it leaves the brush exactly
        // as it found it. Handing back `None` says "unchanged" — the caller keeps the
        // state it passed in rather than paying for a copy of it.
        if segments.is_empty() {
            return (
                base.clone(),
                StrokeCarry {
                    dist: end_dist,
                    tool: None,
                },
            );
        }
        let coords = affected_tiles(&segments);
        // The range's region is a subset of the whole stroke's, which `dynamics_setup`
        // has already bounded, so this cannot be the oversized case — only the empty
        // one, and `segments` is non-empty.
        let Some((halo, lo, region_origin, w, h)) = region_rect(&coords) else {
            return (
                base.clone(),
                StrokeCarry {
                    dist: end_dist,
                    tool: None,
                },
            );
        };

        let kit = &self.dynamics;
        let device = &self.ctx.device;
        let mut scoped = ScopedResources::default();

        // The brush's swept-footprint prefix-τ (shared with the fast path) and its
        // plain coverage mask (the reservoir texels' own footprint weights).
        let prefix_view = self.prefix_view(assets, &rec.brush);
        let cov_view = match rec.brush.shape {
            BrushShape::Stamp(id) => assets
                .coverage_view(id)
                .unwrap_or_else(|| self.round_coverage_view(rec.brush.hardness)),
            BrushShape::Round => self.round_coverage_view(rec.brush.hardness),
        };
        // Colour dynamics for the brush's own `add` paint — the same field and
        // lookup parameters as the fast path (see `deposit` in dynamics.wesl).
        let noise_view = self.noise_view(&rec.brush.color_dynamics);

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
        let region_color = make_tex(
            &mut scoped,
            (w, h),
            region_usage,
            "stark dynamics region color",
        );
        let region_aux = make_tex(
            &mut scoped,
            (w, h),
            region_usage,
            "stark dynamics region aux",
        );

        // Composite pass: base tiles → region, 1:1 with canvas px.
        let (sx, sy) = (2.0 / w as f32, -2.0 / h as f32);
        let view = ViewUniform {
            st: [
                sx,
                sy,
                -region_origin.x * sx - 1.0,
                -region_origin.y * sy + 1.0,
            ],
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

        // ---- The selection over this region (DESIGN.md §6.8), gathered from the same
        // halo tiles the paint came from, so it is 1:1 with the colour/aux regions.
        // An unrestricted selection binds the 1×1 constant instead — the loop's masked
        // reads then return 1 everywhere and the stroke behaves exactly as before.
        let sel_mask = if selection.is_universal() {
            self.selection.constant(1.0).clone()
        } else {
            let (tex, view) =
                self.selection
                    .region_mask(&mut encoder, selection, &halo, region_origin, w, h);
            scoped.texture(tex);
            view
        };

        // ---- Tool reservoir (ping-pong) + footprint snapshot textures. The
        // snapshot rect must cover a segment quad's AABB at any rotation: half
        // extents (radius + len/2 + margin, radius + margin), bounded by √2 × the
        // half-diagonal.
        let loop_usage =
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING;
        let rmax = segments.iter().fold(0.5f32, |m, s| m.max(s.radius));
        let lmax = segments.iter().fold(0.0f32, |m, s| m.max(s.length));
        let dsize = (2.0 * std::f32::consts::SQRT_2 * (rmax + lmax * 0.5 + 1.5)).ceil() as u32;
        let under_color = make_tex(
            &mut scoped,
            (dsize, dsize),
            loop_usage,
            "stark dynamics under color",
        );
        let under_aux = make_tex(
            &mut scoped,
            (dsize, dsize),
            loop_usage,
            "stark dynamics under aux",
        );
        // A stroke that starts fresh initializes its first reservoir by a render clear
        // (the driver does the f16 encode), hence RENDER_ATTACHMENT; one resuming from
        // a [`ToolState`] copies into it instead, hence the COPY pair — which also
        // carries the end state back out.
        let brush_usage = loop_usage
            | wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST;
        let brush_tex = |scoped: &mut ScopedResources, label: &'static str| {
            scoped.texture(device.create_texture(&reservoir_desc(label, brush_usage)))
        };
        let brush_color_tex = [
            brush_tex(&mut scoped, "stark dynamics brush color a"),
            brush_tex(&mut scoped, "stark dynamics brush color b"),
        ];
        let brush_aux_tex = [
            brush_tex(&mut scoped, "stark dynamics brush aux a"),
            brush_tex(&mut scoped, "stark dynamics brush aux b"),
        ];
        let view_of = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());
        let brush_color = [view_of(&brush_color_tex[0]), view_of(&brush_color_tex[1])];
        let brush_aux = [view_of(&brush_aux_tex[0]), view_of(&brush_aux_tex[1])];
        // The segment's swept reservoir prefix (fp32, so the per-fragment difference
        // keeps its precision — see [`BAKE_FORMAT`]). Rebuilt per segment, so a
        // single buffer serves: nothing reads last segment's bake.
        let mut make_bake = |label: &'static str| {
            scoped
                .texture(device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: BAKE_RES,
                        height: BAKE_RES,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: BAKE_FORMAT,
                    usage: loop_usage,
                    view_formats: &[],
                }))
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let bake_load = make_bake("stark dynamics bake load");
        let bake_latm = make_bake("stark dynamics bake latm");
        if let Some(t) = tool {
            // Resume: the tip arrives at this piece carrying exactly what it carried
            // when the previous piece stopped.
            encoder.copy_texture_to_texture(
                t.color.as_image_copy(),
                brush_color_tex[0].as_image_copy(),
                RESERVOIR_EXTENT,
            );
            encoder.copy_texture_to_texture(
                t.aux.as_image_copy(),
                brush_aux_tex[0].as_image_copy(),
                RESERVOIR_EXTENT,
            );
        } else {
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
                                // Carried height = the pre-`charge` glob; carried wet
                                // is 0 (the brush has no wetness knob, so the tool
                                // never picks up or lays gloss of its own).
                                r: d.charge as f64,
                                g: 0.0,
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

        // ---- The dispatch plan (segments + interleaved pickups), one 256-byte
        // slot each (dynamic uniform offsets — the standard way to vary a uniform
        // across dispatches within one pass).
        let (plan, since_end) =
            dynamics_plan(rec, &segments, region_origin, dsize, channels, since0);
        const STRIDE: usize = 256;
        const SLOT: usize = 144; // sizeof the `Stamp` uniform (9 × vec4)
        let mut data = vec![0u8; plan.len() * STRIDE];
        for (i, d) in plan.iter().enumerate() {
            data[i * STRIDE..i * STRIDE + SLOT].copy_from_slice(bytemuck::cast_slice(&d.slot));
        }
        let stamp_buf = scoped.buffer(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stark dynamics stamps"),
            size: data.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.ctx.queue.write_buffer(&stamp_buf, 0, &data);

        // ---- Bind groups. `params` binds a single slot-sized window whose dynamic
        // offset selects the dispatch; pickup/deposit come in two flavours for the
        // reservoir ping-pong.
        let params = || wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &stamp_buf,
                offset: 0,
                size: wgpu::BufferSize::new(SLOT as u64),
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
                        tex(21, &sel_mask),
                    ],
                })
            })
            .collect();
        // One bake bind group per reservoir phase; the deposit reads only the baked
        // result, so it no longer needs the ping-pong at all.
        let bake_bgs: Vec<wgpu::BindGroup> = (0..2)
            .map(|i| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("stark dynamics bake bg"),
                    layout: &kit.bake_bgl,
                    entries: &[
                        params(),
                        samp(),
                        tex(7, &brush_color[i]),
                        tex(8, &brush_aux[i]),
                        tex(17, &bake_load),
                        tex(18, &bake_latm),
                    ],
                })
            })
            .collect();
        let deposit_bgs: Vec<wgpu::BindGroup> = (0..1)
            .map(|_| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("stark dynamics deposit bg"),
                    layout: &kit.deposit_bgl,
                    entries: &[
                        params(),
                        samp(),
                        tex(19, &bake_load),
                        tex(20, &bake_latm),
                        tex(11, &under_color),
                        tex(12, &under_aux),
                        tex(13, &region_color),
                        tex(14, &region_aux),
                        tex(15, &noise_view),
                        wgpu::BindGroupEntry {
                            binding: 16,
                            resource: wgpu::BindingResource::Sampler(&self.noise_sampler),
                        },
                        tex(21, &sel_mask),
                    ],
                })
            })
            .collect();
        // The deposit's prefix-τ volume (group 1) — the same view the fast path
        // binds, so the exchange footprint is the identical definite integral.
        let prefix_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark dynamics prefix bg"),
            layout: &kit.prefix_bgl,
            entries: &[tex(0, &prefix_view)],
        });

        // ---- The loop: snapshot → pickup → deposit per stamp, in stroke order.
        // One compute pass; the implicit barriers between dispatches give the
        // sequential semantics, and usage scopes are per-dispatch, so the region
        // may be sampled by one dispatch and storage-written by the next.
        //
        // `cur` outlives the pass: it names the reservoir texture holding the tool's
        // state, so after the last dispatch it names the state this piece ends in —
        // which is what the next piece has to resume from.
        let mut cur = 0usize;
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stark dynamics stamp loop"),
                timestamp_writes: None,
            });
            let du = dsize.div_ceil(8);
            let bu = BRUSH_RES.div_ceil(8);
            // The bake is one thread per row of its own texture.
            let ku = BAKE_RES.div_ceil(64);
            // The prefix-τ rides at group 1 for `bake` and `deposit`. Re-bound after
            // every pipeline switch: changing to a pipeline whose group-0 layout
            // differs invalidates the groups above it, and both consumers are
            // reached only across such a switch.
            // Each pickup reads `cur` and writes the other, then flips; the segment
            // bakes in between read `cur` (the post-pickup state).
            for (i, d) in plan.iter().enumerate() {
                let off = (i * STRIDE) as u32;
                if d.pickup {
                    cpass.set_pipeline(&kit.pickup_pipeline);
                    cpass.set_bind_group(0, &pickup_bgs[cur], &[off]);
                    cpass.dispatch_workgroups(bu, bu, 1);
                    cur = 1 - cur;
                } else {
                    // Bake this segment's swept reservoir prefix first — it folds in
                    // the tip's current orientation as well as the reservoir state,
                    // so it is per segment, not per pickup.
                    cpass.set_pipeline(&kit.bake_pipeline);
                    cpass.set_bind_group(0, &bake_bgs[cur], &[off]);
                    cpass.set_bind_group(1, &prefix_bg, &[]);
                    cpass.dispatch_workgroups(ku, 1, 1);
                    cpass.set_pipeline(&kit.snapshot_pipeline);
                    cpass.set_bind_group(0, &snapshot_bg, &[off]);
                    cpass.dispatch_workgroups(du, du, 1);
                    cpass.set_pipeline(&kit.deposit_pipeline);
                    cpass.set_bind_group(0, &deposit_bgs[0], &[off]);
                    cpass.set_bind_group(1, &prefix_bg, &[]);
                    cpass.dispatch_workgroups(du, du, 1);
                    // Drain the tool by what this segment just took, so the next one
                    // reads a tool that has actually travelled.
                    cpass.set_pipeline(&kit.deplete_pipeline);
                    cpass.set_bind_group(0, &pickup_bgs[cur], &[off]);
                    cpass.dispatch_workgroups(bu, bu, 1);
                    cur = 1 - cur;
                }
            }
        }

        // ---- Remember the tool. Copied rather than aliased: the loop's own reservoir
        // textures are scoped to this call and destroyed at the end of it, and the
        // range that resumes will write its first pickup straight into whatever it is
        // handed. 64² rgba16f, so the copy is ~32 KB — nothing beside the region work
        // it saves the next pointer move.
        let tool_out = capture.then(|| {
            let mut copy_out = |src: &wgpu::Texture, label: &'static str| {
                let usage = wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST;
                let dst = device.create_texture(&reservoir_desc(label, usage));
                encoder.copy_texture_to_texture(
                    src.as_image_copy(),
                    dst.as_image_copy(),
                    RESERVOIR_EXTENT,
                );
                dst
            };
            ToolState {
                color: copy_out(&brush_color_tex[cur], "stark tool state color"),
                aux: copy_out(&brush_aux_tex[cur], "stark tool state aux"),
                since: since_end,
            }
        });

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
        // `ScopedResources` docs for why waiting on JS GC OOMs the tab. `tool_out` is
        // deliberately *not* among them: it outlives this call by design.
        drop(scoped);
        (
            new_map,
            StrokeCarry {
                dist: end_dist,
                tool: tool_out,
            },
        )
    }
}

/// The reservoir textures' shape — [`BRUSH_RES`]² of the tile colour format, which
/// is what makes a [`ToolState`] copyable into and out of the loop's ping-pong.
const RESERVOIR_EXTENT: wgpu::Extent3d = wgpu::Extent3d {
    width: BRUSH_RES,
    height: BRUSH_RES,
    depth_or_array_layers: 1,
};

fn reservoir_desc(
    label: &'static str,
    usage: wgpu::TextureUsages,
) -> wgpu::TextureDescriptor<'static> {
    wgpu::TextureDescriptor {
        label: Some(label),
        size: RESERVOIR_EXTENT,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage,
        view_formats: &[],
    }
}

/// Build the swept-exchange dispatch plan (DESIGN.md §6.2): one `snapshot` +
/// `deposit` pair per flattened segment (the canvas-side exchange, swept through
/// the prefix-τ integral), interleaved with reservoir `pickup` steps every
/// `spacing · radius` of travel. λ = ln(1 − axis) makes every rate exponential in
/// exposure, so the exchange composes exactly across overlapping segment quads —
/// the continuous path integral, independent of any spacing. Pure CPU float math
/// → replay-deterministic.
///
/// `since0` is the travel already accumulated toward the next pickup — `INFINITY` at
/// a stroke start, which forces one immediately, or whatever the preceding range
/// ended on. Returned alongside the plan so the next range can continue it: the
/// pickup *cadence* is the one piece of loop state that is neither on the canvas nor
/// in the reservoir, and restarting it per range would reload the tool at every cut.
fn dynamics_plan(
    rec: &StrokeRecord,
    segments: &[Segment],
    region_origin: Vec2,
    dsize: u32,
    channels: [f32; 4],
    since0: f32,
) -> (Vec<LoopDispatch>, f32) {
    let b = &rec.brush;
    let d = b.dynamics;
    // λ = ln(1 − axis), clamped away from −∞ (axis = 1 ⇒ e^{−20} ≈ scraped clean),
    // per [`TAU_PER_PASS`] — so an axis reads as a fraction *per pass of the tip*,
    // which is what a 0..1 knob should mean, rather than per unit optical depth.
    let lambda = |axis: f32| (1.0 - axis.clamp(0.0, 1.0)).max(1e-9).ln().max(-20.0) / TAU_PER_PASS;
    let l_lift = lambda(d.lift);
    let l_dep = lambda(d.deposit);
    let half = (dsize / 2) as f32;
    // Colour dynamics for the `add` paint — the same uniform triplet as the fast
    // path, so both paths sample the identical field (DESIGN.md §6.2).
    let (nfreq, namp, noff) = noise_uniform(rec);

    let mut plan = Vec::new();
    let mut since = since0;
    for s in segments {
        let step = (RESERVOIR_CADENCE * s.radius).max(0.5);
        if since >= step {
            // Reservoir update: the tool exchanges for the travel just covered
            // (the first pickup uses one nominal step — a fresh tip arriving).
            let ds = if since.is_finite() { since } else { step };
            let (sn, cs) = (s.orient * std::f32::consts::TAU).sin_cos();
            let rot = Vec2::new(s.dir.x * cs - s.dir.y * sn, s.dir.x * sn + s.dir.y * cs);
            let p = s.start - region_origin;
            plan.push(LoopDispatch {
                pickup: true,
                slot: [
                    p.x,
                    p.y,
                    rot.x,
                    rot.y,
                    s.radius,
                    0.0,
                    l_lift,
                    l_dep,
                    channels[0],
                    channels[1],
                    channels[2],
                    s.opacity,
                    0.0,
                    0.0,
                    s.orient,
                    ds / s.radius,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    // f–i (colour dynamics) — unused by `pickup`.
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            });
            since = 0.0;
        }
        // The segment's swept exchange: quad frame = start + travel tangent; the
        // snapshot rect is centred on the segment midpoint.
        let p = s.start - region_origin;
        let mid = p + s.dir * (s.length * 0.5);
        plan.push(LoopDispatch {
            pickup: false,
            slot: [
                p.x,
                p.y,
                s.dir.x,
                s.dir.y,
                s.radius,
                s.length / s.radius,
                l_lift,
                l_dep,
                channels[0],
                channels[1],
                channels[2],
                s.opacity,
                (mid.x - half).floor(),
                (mid.y - half).floor(),
                s.orient,
                1.0,
                // e: the `add` source rate — height per unit exposure. The wet rate
                // (.y) is 0: paint carries no wetness now that the brush has no
                // wetness knob, so nothing ever adds to the gloss channel.
                s.amount * ADD_GAIN,
                0.0,
                0.0,
                0.0,
                // f–i: the colour-dynamics lookup (see `Stamp` in dynamics.wesl).
                nfreq[0],
                nfreq[1],
                nfreq[2],
                nfreq[3],
                namp[0],
                namp[1],
                namp[2],
                s.dist,
                noff[0],
                noff[1],
                noff[2],
                0.0,
                region_origin.x,
                region_origin.y,
                0.0,
                0.0,
            ],
        });
        since += s.length;
    }
    (plan, since)
}

/// Whether `rec` runs the sequential stamp loop, and the flattening budget it runs
/// at — or `None` for the plain swept fast path.
///
/// **A pure function of the whole record.** Both gates below could answer differently
/// for a short piece of a stroke than for the stroke it belongs to, and every render
/// of every piece has to agree with the commit that eventually replaces it: a live
/// tail that took the stamp loop while the commit degraded to the swept deposit would
/// redraw the stroke the moment the pointer came up. So neither gate is allowed to
/// look at the piece in hand — see [`StrokeRenderer::render_range`].
pub(super) fn dynamics_setup(rec: &StrokeRecord) -> Option<crate::path::FlattenTolerance> {
    let d = rec.brush.dynamics;
    if d.lift <= 0.0 && d.deposit <= 0.0 && d.charge <= 0.0 {
        return None;
    }
    // The same flattened segments as the fast path; extremely long strokes re-flatten
    // coarser so the dispatch count stays bounded. First stretch the length cap to the
    // spacing that would hit the cap exactly, then relax the error bounds — each
    // doubling roughly halves the count on a curved stroke.
    let mut tol = flatten_tolerance(&rec.brush);
    let mut segments = generate_segments_tol(rec, tol);
    if segments.len() > MAX_STAMPS {
        let total: f32 = segments.iter().map(|s| s.length).sum();
        tol.max_len = tol.max_len.max(total / MAX_STAMPS as f32);
        for _ in 0..8 {
            segments = generate_segments_tol(rec, tol);
            if segments.len() <= MAX_STAMPS {
                break;
            }
            tol = tol.relaxed(2.0);
        }
    }
    // An oversized stroke degrades to the swept deposit, bounding the transient GPU
    // memory the loop's 1:1 region costs.
    let (w, h) = region_dim(&segments)?;
    (w <= MAX_REGION_DIM && h <= MAX_REGION_DIM).then_some(tol)
}

/// Build the brush-dynamics stamp-loop kit (DESIGN.md §6.2): the region
/// composite, the three loop compute pipelines, and the region→tile slice.
pub(super) fn build_dynamics_kit(
    device: &wgpu::Device,
    color_space: &dyn ColorSpace,
) -> DynamicsKit {
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
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<TileInstance>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32],
            })],
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
            min_binding_size: wgpu::BufferSize::new(144), // sizeof `Stamp` (9 × vec4)
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
    // The baked swept prefix is fp32 — it is differenced per fragment, like the
    // prefix-τ volume, so f16 would band exactly where the difference is smallest.
    let stor32 = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: BAKE_FORMAT,
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
        entries: &[
            params_entry,
            ctex(1, false),
            ctex(2, false),
            stor(3),
            stor(4),
        ],
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
            // The selection mask over the region (§6.8) — sampled bilinearly here,
            // since a reservoir texel sits over an arbitrary sub-pixel spot.
            ctex(21, true),
        ],
    });
    // `bake` integrates the reservoir along the travel axis for one segment; the
    // deposit then reads the result instead of point-sampling the reservoir.
    let bake_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("stark dynamics bake bgl"),
        entries: &[
            params_entry,
            csamp,
            ctex(7, true),
            ctex(8, true),
            stor32(17),
            stor32(18),
        ],
    });
    let deposit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("stark dynamics deposit bgl"),
        entries: &[
            params_entry,
            csamp,
            ctex(19, false),
            ctex(20, false),
            ctex(11, false),
            ctex(12, false),
            stor(13),
            stor(14),
            // The colour-dynamics noise volume + its repeat sampler (§6.2).
            wgpu::BindGroupLayoutEntry {
                binding: 15,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D3,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 16,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            // The selection mask over the region (§6.8) — read 1:1 with the region
            // here, so `textureLoad` suffices.
            ctex(21, false),
        ],
    });
    // The deposit's prefix-τ volume (group 1) — same shape as the fast path's
    // prefix binding, but compute-visible.
    let prefix_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("stark dynamics prefix bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        }],
    });
    let cpipe = |label: &str, entry: &str, bgls: &[Option<&wgpu::BindGroupLayout>]| {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: bgls,
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
    let snapshot_pipeline = cpipe(
        "stark dynamics snapshot",
        "snapshot",
        &[Some(&snapshot_bgl)],
    );
    let pickup_pipeline = cpipe("stark dynamics pickup", "pickup", &[Some(&pickup_bgl)]);
    // `deplete` touches a subset of what `pickup` binds (no region), so it can share
    // the layout and its bind groups — unused entries are legal.
    let deplete_pipeline = cpipe("stark dynamics deplete", "deplete", &[Some(&pickup_bgl)]);
    // The bake reads the prefix-τ volume too (group 1) — the exposure weights in
    // its integral are that volume's own differences.
    let bake_pipeline = cpipe(
        "stark dynamics bake",
        "bake",
        &[Some(&bake_bgl), Some(&prefix_bgl)],
    );
    let deposit_pipeline = cpipe(
        "stark dynamics deposit",
        "deposit",
        &[Some(&deposit_bgl), Some(&prefix_bgl)],
    );
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
        deplete_pipeline,
        bake_pipeline,
        bake_bgl,
        deposit_pipeline,
        deposit_bgl,
        exchange_sampler,
        slice_pipeline,
        slice_bgl,
        prefix_bgl,
        round_cov: Arc::new(Mutex::new(None)),
    }
}

/// Build the stroke integrate pipeline (`integrate` shader) — DESIGN §6.2/§6.1. A
/// fullscreen pass with four sampled tiles (base/scratch color/aux), writing the
/// color+aux MRT of a fresh tile.
pub(super) fn build_integrate_pipeline(
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
            load_tex(4), // selection mask (§6.8) — this tile's, or a 1×1 constant
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
