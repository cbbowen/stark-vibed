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
    Segment, affected_tiles, chunk_segments, coverage_bounds, generate_segments_in, noise_uniform,
    region_rect, segment_fits_region,
};
use super::swept::{TileInstance, ViewUniform};
use super::{
    ADD_GAIN, BAKE_FORMAT, BAKE_RES, BRUSH_RES, ScopedResources, StrokeCarry, StrokeRenderer,
    StrokeScene, StrokeSpans, TAU_PER_PASS, ToolState, flatten_tolerance,
};

/// Mirrors `Params` in `slice.wesl`: the tile texture's top-left in region texels.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SliceUniform {
    offset: [f32; 4],
}

/// One segment of the sequential swept-exchange loop (DESIGN.md §6.2): its
/// `bake`, `snapshot`, `deposit` and `exchange` dispatches. `slot` is the 128-byte
/// `Stamp` uniform (see dynamics.wesl), precomputed CPU-side as a pure function of
/// the `StrokeRecord`, so replay is deterministic.
struct LoopDispatch {
    slot: [f32; 32],
    /// Workgroup counts for the segment's `snapshot`/`deposit` dispatches: the
    /// segment's own coverage box rather than the piece-wide worst-case square, so an
    /// axis-aligned sweep pays for the ~4·r² texels its footprint can reach instead
    /// of the ~10·r² a diagonal one might have needed.
    groups: (u32, u32),
    /// Workgroup counts for `exchange`, which covers the reservoir instead.
    exchange_groups: (u32, u32),
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
    /// The tool's own side of one segment's transfer: drain by what it just laid,
    /// then lift what is now under it (`dynamics.wesl::exchange`).
    pub(super) exchange_pipeline: wgpu::ComputePipeline,
    pub(super) exchange_bgl: wgpu::BindGroupLayout,
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
    /// swept-exchange loop** (DESIGN.md §6.2): composite the base under it into a 1:1
    /// region, then walk it *in order* on the GPU — the canvas-side exchange swept per
    /// flattened segment through the prefix-τ integral (the same definite-integral
    /// footprint as the plain deposit), the 2-D tool reservoir updated at
    /// `spacing · radius` cadence — and slice the evolved region back into fresh CoW
    /// tiles.
    ///
    /// A region is a 1:1 copy of the canvas under the stroke, so the range is drawn in
    /// as many region-sized **pieces** as it takes ([`chunk_segments`]) rather than in
    /// one: the loop is sequential, so pieces run back to back over the same segments
    /// in the same order, each compositing what the last wrote back. Length therefore
    /// costs the stroke extra pieces, not correctness — where it used to degrade past
    /// [`MAX_REGION_DIM`](super::MAX_REGION_DIM) to the plain swept deposit, which
    /// cannot manipulate paint at all.
    ///
    /// The loop starts from `tool` rather than from a fresh tip when one is given, and
    /// hands back the state it ends in whenever a further range remains to be drawn,
    /// so a live stroke redraws only its tail (see [`ToolState`]). `tol` comes from
    /// [`dynamics_setup`], which has already decided — from the brush — that this
    /// stroke runs the loop at all.
    pub(super) fn render_dynamic(
        &self,
        scene: StrokeScene<'_>,
        rec: &StrokeRecord,
        spans: StrokeSpans,
        tool: Option<&ToolState>,
        tol: crate::path::FlattenTolerance,
    ) -> (HashTrieMap<TileCoord, TilePairHandle>, StrokeCarry) {
        // Nothing follows the range that reaches the end of the stroke, so there is no
        // reason to snapshot a reservoir for it — which is the common case, since the
        // live tail is exactly that range and it re-renders every pointer move.
        let capture = spans.range.end < crate::path::span_count(rec.path.len());
        let (segments, end_dist) = generate_segments_in(rec, tol, spans);
        // A range with no geometry runs no dispatches, so it leaves the brush exactly
        // as it found it. Handing back `None` says "unchanged" — the caller keeps the
        // state it passed in rather than paying for a copy of it.
        if segments.is_empty() {
            return (
                scene.base.clone(),
                StrokeCarry {
                    dist: end_dist,
                    tool: None,
                    dirty: Vec::new(),
                },
            );
        }

        // The union over the pieces below, which each enumerate their own subset.
        let dirty: Vec<TileCoord> = affected_tiles(&segments).into_iter().collect();
        let mut run = DynamicsRun::new(self, scene, rec, tool);
        let mut map = scene.base.clone();
        for piece in chunk_segments(&segments) {
            map = run.draw(&map, &segments[piece]);
        }
        let tool_out = capture.then(|| run.capture_tool());
        run.submit();
        (
            map,
            StrokeCarry {
                dist: end_dist,
                tool: tool_out,
                dirty,
            },
        )
    }
}

/// One [`StrokeRenderer::render_dynamic`] call in progress: the brush-local state the
/// loop threads along the stroke, and the GPU objects that outlive any one region.
///
/// What survives from piece to piece is exactly what survives from one *range* to the
/// next — the tool reservoir — because that is all the loop carries between segments
/// that is not already on the canvas. It lives
/// here rather than being copied out into a [`ToolState`] and back in at every cut:
/// the pieces are recorded one after another against the same reservoir textures, so
/// the ping-pong simply keeps going.
struct DynamicsRun<'a> {
    r: &'a StrokeRenderer,
    rec: &'a StrokeRecord,
    scene: StrokeScene<'a>,
    encoder: wgpu::CommandEncoder,
    /// GPU objects scoped to the whole run — the reservoir and the bake pair, which
    /// carry the tool from one piece to the next.
    scoped: ScopedResources,
    /// GPU objects scoped to the piece being recorded: the region, its selection
    /// mask, the snapshot scratch, the stamp buffer. Destroyed as soon as the piece
    /// is submitted (see [`Self::flush`]), so a stroke of any length costs one
    /// region's worth of transient memory rather than one per piece.
    piece: ScopedResources,
    /// The brush's own colour in the working space, plus its per-unit opacity.
    channels: [f32; 4],
    /// Functions of the brush alone, so shared by every piece: the swept-footprint
    /// prefix-τ bind group (group 1 of `bake`/`deposit` — the same texture the swept
    /// fast path samples), the plain coverage mask the reservoir texels weight by,
    /// and the colour-dynamics field the `add` paint is jittered against.
    prefix_bg: wgpu::BindGroup,
    cov: wgpu::TextureView,
    noise: wgpu::TextureView,
    /// The tool reservoir ping-pong, and which half currently holds the tool.
    brush_color_tex: [wgpu::Texture; 2],
    brush_aux_tex: [wgpu::Texture; 2],
    brush_color: [wgpu::TextureView; 2],
    brush_aux: [wgpu::TextureView; 2],
    cur: usize,
    /// The segment's swept reservoir prefixes (fp32, so the per-fragment difference
    /// keeps its precision — see [`BAKE_FORMAT`]). Rebuilt per segment, so a single
    /// pair serves the whole stroke: nothing reads the last segment's bake.
    bake_load: wgpu::TextureView,
    bake_latm: wgpu::TextureView,
}

impl<'a> DynamicsRun<'a> {
    /// Open the run: resolve the brush's textures, and put the tool in the state the
    /// stroke arrives at this range with — resumed from `tool`, or freshly charged.
    fn new(
        r: &'a StrokeRenderer,
        scene: StrokeScene<'a>,
        rec: &'a StrokeRecord,
        tool: Option<&ToolState>,
    ) -> Self {
        let device = &r.ctx.device;
        let mut scoped = ScopedResources::default();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stark dynamics stroke"),
        });
        let rgb = [rec.brush.color[0], rec.brush.color[1], rec.brush.color[2]];
        let channels = r.color_space.rgb_to_channels(rgb);

        // The brush's swept-footprint prefix-τ (shared with the fast path) and its
        // plain coverage mask (the reservoir texels' own footprint weights).
        let prefix_view = r.prefix_view(scene.assets, &rec.brush);
        let prefix_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark dynamics prefix bg"),
            layout: &r.dynamics.prefix_bgl,
            entries: &[tex(0, &prefix_view)],
        });
        let cov = match rec.brush.shape {
            BrushShape::Stamp(id) => scene
                .assets
                .coverage_view(id)
                .unwrap_or_else(|| r.round_coverage_view(BrushShape::DEFAULT_HARDNESS)),
            BrushShape::Round { hardness } => r.round_coverage_view(hardness),
        };
        // Colour dynamics for the brush's own `add` paint — the same field and
        // lookup parameters as the fast path (see `deposit` in dynamics.wesl).
        let noise = r.noise_view(&rec.brush.color_dynamics);

        // A stroke that starts fresh initializes its first reservoir by a render clear
        // (the driver does the f16 encode), hence RENDER_ATTACHMENT; one resuming from
        // a [`ToolState`] copies into it instead, hence the COPY pair — which also
        // carries the end state back out.
        let brush_usage = LOOP_USAGE
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
        if let Some(t) = tool {
            // Resume: the tip arrives at this range carrying exactly what it carried
            // when the previous range stopped.
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
                                // Carried height = the pre-`charge` glob; the rest of
                                // the reservoir aux is unused (height is the only
                                // thing the tool carries, DESIGN.md §6.1).
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
        let mut bake = |label: &'static str| {
            scoped_view(
                device,
                &mut scoped,
                (BAKE_RES, BAKE_RES),
                BAKE_FORMAT,
                LOOP_USAGE,
                label,
            )
        };
        let bake_load = bake("stark dynamics bake load");
        let bake_latm = bake("stark dynamics bake latm");
        Self {
            r,
            rec,
            scene,
            encoder,
            scoped,
            piece: ScopedResources::default(),
            channels,
            prefix_bg,
            cov,
            noise,
            brush_color_tex,
            brush_aux_tex,
            brush_color,
            brush_aux,
            cur: 0,
            bake_load,
            bake_latm,
        }
    }

    /// Evolve one region-sized piece of the stroke over `base`: composite the tiles
    /// under `segments` into a region, walk them through the loop, and slice the
    /// result back into fresh CoW tiles. The tool carries on from where the previous
    /// piece left it, and the canvas side needs no carrying — it is in `base`, which
    /// for a later piece is what the earlier ones wrote back.
    fn draw(
        &mut self,
        base: &HashTrieMap<TileCoord, TilePairHandle>,
        segments: &[Segment],
    ) -> HashTrieMap<TileCoord, TilePairHandle> {
        self.flush();
        let r = self.r;
        let rec = self.rec;
        let kit = &r.dynamics;
        let device = &r.ctx.device;
        let channels = self.channels;

        let coords = affected_tiles(segments);
        // A piece holds at least one segment, and a segment covers at least one tile,
        // so the empty case cannot arise here — but it costs nothing to leave the
        // canvas alone if it ever did.
        let Some((halo, lo, region_origin, w, h)) = region_rect(&coords) else {
            return base.clone();
        };

        let clear = wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            store: wgpu::StoreOp::Store,
        };

        // ---- The piece's canvas region (colour + wide aux), composited from the
        // base tiles of the affected set plus a one-tile ring, so rewritten tiles'
        // aprons read real neighbour content (§6.4). Rgba16Float throughout: it is
        // both filterable and a core storage format, and matches the tile colour
        // format of both color spaces (asserted in `build_dynamics_kit`).
        let region_usage = wgpu::TextureUsages::RENDER_ATTACHMENT | LOOP_USAGE;
        let mut region_tex = |label: &'static str| {
            scoped_view(
                device,
                &mut self.piece,
                (w, h),
                wgpu::TextureFormat::Rgba16Float,
                region_usage,
                label,
            )
        };
        let region_color = region_tex("stark dynamics region color");
        let region_aux = region_tex("stark dynamics region aux");

        // Composite pass: base tiles → region, 1:1 with canvas px.
        let (sx, sy) = (2.0 / w as f32, -2.0 / h as f32);
        let view = ViewUniform {
            // Diagonal: the region is axis-aligned with the canvas whatever angle the
            // *screen* view happens to be at.
            st: [sx, 0.0, 0.0, sy],
            xlate: [
                -region_origin.x * sx - 1.0,
                -region_origin.y * sy + 1.0,
                0.0,
                0.0,
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
            let mut pass = self.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
        let sel_mask = if self.scene.selection.is_universal() {
            r.selection.constant(1.0).clone()
        } else {
            let (tex, view) = r.selection.region_mask(
                &mut self.encoder,
                self.scene.selection,
                &halo,
                region_origin,
                w,
                h,
            );
            self.piece.texture(tex);
            view
        };

        // ---- Footprint snapshot textures. The snapshot rect must cover any one
        // segment's coverage box — its swept arc plus the tip riding along it —
        // which is measured rather than bounded analytically, since `coverage_bounds`
        // is already the exact box and a curved sweep has no closed-form "worst
        // rotation" to fall back on. Sized from *this* piece's segments, so a piece
        // drawn with a fine tip pays for a fine tip.
        //
        // +3 for the sampling margin `dynamics_plan` adds each side, +2 because a
        // per-segment rect then rounds outward by a texel each side.
        let dmax = segments.iter().fold(1.0f32, |m, s| {
            let (lo, hi) = coverage_bounds(s);
            m.max(hi.x - lo.x).max(hi.y - lo.y)
        });
        let dsize = (dmax + 3.0).ceil() as u32 + 2;
        let mut under_tex = |label: &'static str| {
            scoped_view(
                device,
                &mut self.piece,
                (dsize, dsize),
                wgpu::TextureFormat::Rgba16Float,
                LOOP_USAGE,
                label,
            )
        };
        let under_color = under_tex("stark dynamics under color");
        let under_aux = under_tex("stark dynamics under aux");

        // ---- The dispatch plan, one segment each, one 256-byte
        // slot each (dynamic uniform offsets — the standard way to vary a uniform
        // across dispatches within one pass).
        let plan = dynamics_plan(rec, segments, region_origin, dsize, channels);
        const STRIDE: usize = 256;
        const SLOT: usize = 128; // sizeof the `Stamp` uniform (8 × vec4)
        let mut data = vec![0u8; plan.len() * STRIDE];
        for (i, d) in plan.iter().enumerate() {
            data[i * STRIDE..i * STRIDE + SLOT].copy_from_slice(bytemuck::cast_slice(&d.slot));
        }
        let stamp_buf = self
            .scoped
            .buffer(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("stark dynamics stamps"),
                size: data.len() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        r.ctx.queue.write_buffer(&stamp_buf, 0, &data);

        // ---- Bind groups. `params` binds a single slot-sized window whose dynamic
        // offset selects the dispatch; `exchange` comes in two flavours for the
        // reservoir ping-pong.
        let params = || wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &stamp_buf,
                offset: 0,
                size: wgpu::BufferSize::new(SLOT as u64),
            }),
        };
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
        let exchange_bgs: Vec<wgpu::BindGroup> = (0..2)
            .map(|i| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("stark dynamics exchange bg"),
                    layout: &kit.exchange_bgl,
                    entries: &[
                        params(),
                        tex(1, &region_color),
                        tex(2, &region_aux),
                        samp(),
                        tex(6, &self.cov),
                        tex(7, &self.brush_color[i]),
                        tex(8, &self.brush_aux[i]),
                        tex(9, &self.brush_color[1 - i]),
                        tex(10, &self.brush_aux[1 - i]),
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
                        tex(7, &self.brush_color[i]),
                        tex(8, &self.brush_aux[i]),
                        tex(17, &self.bake_load),
                        tex(18, &self.bake_latm),
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
                        tex(19, &self.bake_load),
                        tex(20, &self.bake_latm),
                        tex(11, &under_color),
                        tex(12, &under_aux),
                        tex(13, &region_color),
                        tex(14, &region_aux),
                        tex(15, &self.noise),
                        wgpu::BindGroupEntry {
                            binding: 16,
                            resource: wgpu::BindingResource::Sampler(&r.noise_sampler),
                        },
                        tex(21, &sel_mask),
                    ],
                })
            })
            .collect();

        // ---- The loop: bake → snapshot → deposit → exchange per segment, in stroke
        // order. One compute pass; the implicit barriers between dispatches give the
        // sequential semantics, and usage scopes are per-dispatch, so the region
        // may be sampled by one dispatch and storage-written by the next.
        //
        // `self.cur` outlives the pass: it names the reservoir texture holding the
        // tool's state, so after the last dispatch it names the state this piece ends
        // in — which is what the next piece, or the next range, resumes from.
        {
            let mut cur = self.cur;
            let prefix_bg = &self.prefix_bg;
            let mut cpass = self
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("stark dynamics stamp loop"),
                    timestamp_writes: None,
                });
            // The prefix-τ rides at group 1 for `bake` and `deposit`. Re-bound after
            // every pipeline switch: changing to a pipeline whose group-0 layout
            // differs invalidates the groups above it, and both consumers are
            // reached only across such a switch.
            for (i, d) in plan.iter().enumerate() {
                let off = (i * STRIDE) as u32;
                // Bake this segment's swept reservoir prefix first — it folds in the
                // tip's current orientation as well as the reservoir state.
                cpass.set_pipeline(&kit.bake_pipeline);
                cpass.set_bind_group(0, &bake_bgs[cur], &[off]);
                cpass.set_bind_group(1, prefix_bg, &[]);
                // One BAKE_RES-wide workgroup per row: the shader's scan width is a
                // constant, so the two must agree.
                cpass.dispatch_workgroups(1, BAKE_RES, 1);
                cpass.set_pipeline(&kit.snapshot_pipeline);
                cpass.set_bind_group(0, &snapshot_bg, &[off]);
                cpass.dispatch_workgroups(d.groups.0, d.groups.1, 1);
                cpass.set_pipeline(&kit.deposit_pipeline);
                cpass.set_bind_group(0, &deposit_bgs[0], &[off]);
                cpass.set_bind_group(1, prefix_bg, &[]);
                cpass.dispatch_workgroups(d.groups.0, d.groups.1, 1);
                // Then the tool's own side, for the very same segment: drain by what
                // it just laid, lift what is now under it. Reads `cur` and writes the
                // other half, so the next segment's bake sees a tool that has actually
                // travelled and reloaded.
                cpass.set_pipeline(&kit.exchange_pipeline);
                cpass.set_bind_group(0, &exchange_bgs[cur], &[off]);
                cpass.dispatch_workgroups(d.exchange_groups.0, d.exchange_groups.1, 1);
                cur = 1 - cur;
            }
            self.cur = cur;
        }

        // ---- Write-back: slice each affected tile's full TILE_TEX block out of
        // the shared region → aprons stay bit-identical to neighbour interiors
        // (§6.4), and the wide region aux narrows to the persistent (height).
        let mut new_map = base.clone();
        for coord in &coords {
            let dst = r.acquire_tile(self.scene.pool, AllocSource::DynamicsWriteback);
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
                let mut pass = self.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
        new_map
    }

    /// Close out the piece already recorded, if any: submit its encoder, then destroy
    /// the region it worked on. Peak transient memory is then one region however long
    /// the stroke is — which is what [`MAX_REGION_DIM`](super::MAX_REGION_DIM) is for,
    /// and it would not hold if every piece's region had to live until the last one
    /// finished. Submissions run in order, so the next piece still composites what
    /// this one wrote back.
    ///
    /// A stroke that fits one region never reaches the second call, so the everyday
    /// case still records and submits exactly once.
    fn flush(&mut self) {
        if self.piece.is_empty() {
            return;
        }
        let fresh = self
            .r
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("stark dynamics stroke"),
            });
        let done = std::mem::replace(&mut self.encoder, fresh);
        self.r.ctx.queue.submit([done.finish()]);
        drop(std::mem::take(&mut self.piece));
    }

    /// Remember the tool for the range that resumes after this one. Copied rather
    /// than aliased: the loop's own reservoir textures are scoped to this call and
    /// destroyed at the end of it, and the range that resumes will write its first
    /// exchange straight into whatever it is handed. 64² rgba16f ×2, so the copy is
    /// ~64 KB — nothing beside the region work it saves the next pointer move.
    fn capture_tool(&mut self) -> ToolState {
        let device = &self.r.ctx.device;
        let copy_out = |encoder: &mut wgpu::CommandEncoder, src: &wgpu::Texture, label| {
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
            color: copy_out(
                &mut self.encoder,
                &self.brush_color_tex[self.cur],
                "stark tool state color",
            ),
            aux: copy_out(
                &mut self.encoder,
                &self.brush_aux_tex[self.cur],
                "stark tool state aux",
            ),
        }
    }

    /// Close the run: submit what is still recorded, then destroy the per-stroke
    /// region/reservoir textures + buffers (safe: WebGPU defers the real free past the
    /// submitted work) — see the [`ScopedResources`] docs for why waiting on JS GC
    /// OOMs the tab. What [`Self::capture_tool`] handed back is deliberately *not*
    /// among them: it outlives this call by design.
    fn submit(self) {
        self.r.ctx.queue.submit([self.encoder.finish()]);
        drop(self.piece);
        drop(self.scoped);
    }
}

/// Shorthand for a texture-view bind-group entry.
fn tex(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

/// What every texture the loop reads and writes needs to be: sampled by one
/// dispatch, storage-written by the next.
const LOOP_USAGE: wgpu::TextureUsages =
    wgpu::TextureUsages::TEXTURE_BINDING.union(wgpu::TextureUsages::STORAGE_BINDING);

/// A texture scoped to one `render_dynamic` call, as a view: registered with `scoped`
/// so it is destroyed right after the submit.
fn scoped_view(
    device: &wgpu::Device,
    scoped: &mut ScopedResources,
    size: (u32, u32),
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
    label: &'static str,
) -> wgpu::TextureView {
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
            format,
            usage,
            view_formats: &[],
        }))
        .create_view(&wgpu::TextureViewDescriptor::default())
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
/// the prefix-τ integral), each followed by the tool's own `exchange`.
/// λ = ln(1 − axis) makes every rate exponential in
/// exposure, so the exchange composes exactly across overlapping segment quads —
/// the continuous path integral, independent of any spacing. Pure CPU float math
/// → replay-deterministic.
///
/// Every dispatch is a segment: the tool exchanges once per segment rather than on a
/// cadence of its own, so there is no interval state to carry between ranges.
fn dynamics_plan(
    rec: &StrokeRecord,
    segments: &[Segment],
    region_origin: Vec2,
    dsize: u32,
    channels: [f32; 4],
) -> Vec<LoopDispatch> {
    let b = &rec.brush;
    let d = b.dynamics;
    // λ = ln(1 − axis), clamped away from −∞ (axis = 1 ⇒ e^{−20} ≈ scraped clean),
    // per [`TAU_PER_PASS`] — so an axis reads as a fraction *per pass of the tip*,
    // which is what a 0..1 knob should mean, rather than per unit optical depth.
    let lambda = |axis: f32| (1.0 - axis.clamp(0.0, 1.0)).max(1e-9).ln().max(-20.0) / TAU_PER_PASS;
    let l_lift = lambda(d.lift);
    let l_dep = lambda(d.deposit);
    // Colour dynamics for the `add` paint — the same uniform triplet as the fast
    // path, so both paths sample the identical field (DESIGN.md §6.2).
    let (nfreq, namp, noff) = noise_uniform(rec);

    let mut plan = Vec::new();
    for s in segments {
        // The segment's swept exchange: the frame is (start, travel tangent at the
        // start, curvature), and the dispatch rect is the segment's own coverage box
        // plus a 1.5 px sampling margin — so an axis-aligned sweep dispatches ~4·r²
        // threads where a piece-wide square would spend ~10·r². Texels the dispatch
        // rounds in beyond the box read zero exposure and fall out of `deposit`
        // untouched, and every rect fits the `under` scratch because `dsize` was
        // measured from these same boxes.
        let p = s.start - region_origin;
        let (clo, chi) = coverage_bounds(s);
        let lo = clo - region_origin - Vec2::splat(1.5);
        let hi = chi - region_origin + Vec2::splat(1.5);
        let (ox, oy) = (lo.x.floor(), lo.y.floor());
        let (w, h) = (
            (((hi.x - ox).ceil() as u32) + 1).min(dsize),
            (((hi.y - oy).ceil() as u32) + 1).min(dsize),
        );
        // Where `exchange` samples the canvas: the segment's midpoint, along the arc
        // rather than the chord. The midpoint rule for a lift that is really swept
        // over the segment — second order, where either endpoint would be first.
        let mid =
            crate::path::arc_at(s.start, s.dir, s.curvature, s.length * 0.5).0 - region_origin;
        plan.push(LoopDispatch {
            groups: (w.div_ceil(8), h.div_ceil(8)),
            exchange_groups: (BRUSH_RES.div_ceil(8), BRUSH_RES.div_ceil(8)),
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
                ox,
                oy,
                s.orient,
                0.0,
                // e: the `add` source rate — height per unit exposure — the segment's
                // signed curvature, which bends the travel frame every dispatch of
                // this loop measures its exchange in, and the midpoint `exchange`
                // lifts from (dynamics.wesl).
                s.amount * ADD_GAIN,
                s.curvature,
                mid.x,
                mid.y,
                // f–h: the colour-dynamics lookup (see `Stamp` in dynamics.wesl).
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
            ],
        });
    }
    plan
}

/// Which path a stroke takes, as [`dynamics_setup`] decides it.
///
/// The two swept answers are kept apart because they are not the same event: one is
/// the fast path doing its job, the other is the renderer failing to draw the brush
/// it was given. Only the caller knows how loudly to say so, so the distinction is
/// carried out rather than resolved here.
pub(super) enum StrokePath {
    /// Run the sequential stamp loop, flattening at this budget.
    Loop(crate::path::FlattenTolerance),
    /// The brush manipulates no paint already on the canvas, so the swept deposit
    /// *is* the whole stroke — one pass, no region, nothing given up.
    Swept,
    /// The brush manipulates paint, but its tip alone wants more than one region, and
    /// the region is the one thing pieces cannot subdivide. The swept deposit draws
    /// what it can, which is the brush's own `add` paint and none of the manipulation.
    TipTooLarge,
}

/// Which path `rec` takes, and the flattening budget if it is the stamp loop.
///
/// **A pure function of the record, and of the brush alone.** This answer has to
/// agree across every render of every piece of the stroke and with the commit that
/// eventually replaces them: a live tail that took the stamp loop while the commit
/// degraded to the swept deposit would redraw the stroke the moment the pointer came
/// up. Asking only about the brush is the strongest form of that guarantee — there is
/// nothing about the piece in hand, or the stroke's length, for it to disagree over —
/// and it is what lets `render_range` re-ask on every pointer move for free.
///
/// It reads that way because the stroke's *size* no longer decides anything: an
/// oversized stroke is drawn one region-sized piece at a time ([`chunk_segments`])
/// rather than degraded. All that is left is the floor no subdivision gets under —
/// one segment's own footprint — which is [`segment_fits_region`]'s question.
pub(super) fn dynamics_setup(rec: &StrokeRecord) -> StrokePath {
    let d = rec.brush.dynamics;
    if d.lift <= 0.0 && d.deposit <= 0.0 && d.charge <= 0.0 {
        return StrokePath::Swept;
    }
    // The same flattened segments as the fast path, at the same budget: a long stroke
    // costs more pieces, not coarser geometry.
    let tol = flatten_tolerance(&rec.brush);
    if segment_fits_region(&rec.brush, tol) {
        StrokePath::Loop(tol)
    } else {
        StrokePath::TipTooLarge
    }
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
            // `fs_raw`, NOT the screen path's `fs_main`: the loop's region must
            // hold the tile representation itself (opacity in alpha), not the
            // coverage-weighted channels pass A shows — the exchange reads this
            // region and the slice writes it back to persistent tiles.
            entry_point: Some("fs_raw"),
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
            min_binding_size: wgpu::BufferSize::new(128), // sizeof `Stamp` (8 × vec4)
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
    let exchange_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("stark dynamics exchange bgl"),
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
            // The colour-dynamics noise tile + its repeat sampler (§6.2).
            wgpu::BindGroupLayoutEntry {
                binding: 15,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
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
    let exchange_pipeline = cpipe(
        "stark dynamics exchange",
        "exchange",
        &[Some(&exchange_bgl)],
    );
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
        exchange_pipeline,
        exchange_bgl,
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
