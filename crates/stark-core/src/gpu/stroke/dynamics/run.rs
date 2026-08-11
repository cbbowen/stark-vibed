//! Recording one dynamics render: regions, the reservoir ping-pong, the compute pass
//! the plan's slots are dispatched through, and the write-back (§6.2).
//!
//! This is the half that owns GPU state. What it is *asked* to record comes from
//! [`plan`](super::plan); what it records *with* comes from [`kit`](super::kit).

use wgpu::util::DeviceExt;

use std::collections::BTreeSet;

use crate::document::StrokeRecord;
use crate::geom::{TileCoord, Vec2};
use crate::gpu::composite::view_uniform;
use crate::gpu::desc;
use crate::gpu::tile::{AllocSource, TileMap};

use super::super::segments::{
    RegionRect, Segment, affected_tiles, chunk_segments, generate_segments_in, region_rect,
};
use super::super::{
    ScopedResources, StrokeCarry, StrokeRenderer, StrokeScene, StrokeSpans, ToolState,
    UNIFORM_STRIDE,
};
use super::BAKE_FORMAT;
use super::plan::{
    LoopDispatch, PlanCtx, SLOT, SlotKind, bleed_fires, cell_scratch_size, dynamics_plan,
    snapshot_size,
};
use super::scratch::{Key, Lease, ScratchPool};
/// Resolution (texels per side) of the stamp loop's tool reservoir
/// (§6.2). Brush-local, so carried colour detail is ~radius/32 canvas px — plenty
/// for smeared paint, and small enough that the per-stamp reservoir update is
/// nearly free.
const BRUSH_RES: u32 = 64;
// Resolution of the per-segment **swept prefix** of the reservoir
// (`dynamics.wesl::bake`), and the workgroup width of the scan that builds it —
// generated from the shader, which is the side that decides it (§6.10). A mismatch
// scanned the wrong width and rendered subtly wrong without crashing.
use stark_shaders::mirror::dynamics::BAKE_RES;
// The `@binding` indices, generated from `dynamics.wesl`'s own declarations (§6.10):
// the bind groups here and the layouts in [`kit`](super::kit) both name the
// shader's numbering rather than keeping copies of it.
use stark_shaders::mirror::dynamics::binding as b;

/// The snapshot square's pool quantum: [`DynamicsRun::snapshot_scratch`] rounds the
/// measured size up to a multiple of this, so the handful of sizes a stroke's folds
/// actually take recur — a [`ScratchPool`] hit — instead of drifting a few texels
/// per pointer move and missing every time. See the round-up's comment for why the
/// rounding is invisible to every pass.
const SNAPSHOT_QUANTUM: u32 = 64;

// The region composite runs `composite.wesl`, so it draws that shader's own
// per-instance record (§6.10) — which this module used to declare a second
// `#[repr(C)]` copy of, under a different name.
use stark_shaders::mirror::composite::Instance as TileInstance;

/// Workgroup counts for the reservoir half of `exchange`, which is dispatched over
/// these *plus* the slot's footprint groups on x, since the snapshot shares its
/// grid.
///
/// A constant, not per-dispatch data: the reservoir is [`BRUSH_RES`]² whatever the
/// segment does, and the two slot kinds that do not run an exchange never read it.
const RESERVOIR_GROUPS: (u32, u32) = (BRUSH_RES.div_ceil(8), BRUSH_RES.div_ceil(8));

impl StrokeRenderer {
    /// Render `spans` of a paint-manipulating stroke via the **sequential
    /// swept-exchange loop** (§6.2): composite the base under it into a 1:1
    /// region, then walk it *in order* on the GPU — the canvas-side exchange swept per
    /// flattened segment through the prefix-τ integral (the same definite-integral
    /// footprint as the plain deposit), the 2-D tool reservoir taking the complement
    /// of it over the same segment — and slice the evolved region back into fresh CoW
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
    pub(in crate::gpu::stroke) fn render_dynamic(
        &self,
        scene: StrokeScene<'_>,
        rec: &StrokeRecord,
        spans: StrokeSpans,
        tool: Option<&ToolState>,
        tol: crate::path::FlattenTolerance,
    ) -> (TileMap, StrokeCarry) {
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

        let mut run = DynamicsRun::new(self, scene, rec, tol, tool);
        let mut map = scene.base.clone();
        // The pen-up settle (§6.2) belongs to the range that reaches the *stroke's* end,
        // and within it to the last piece — which is the same condition that says there
        // is no reservoir worth keeping. A range that stops short hands its tool on
        // instead, so nothing is stranded for the settle to hand back, and a live tail
        // computes the same settle its commit will.
        let pieces = chunk_segments(&segments);
        let last = pieces.len() - 1;
        for (i, piece) in pieces.into_iter().enumerate() {
            map = run.draw(&map, &segments[piece], !capture && i == last);
        }
        let tool_out = capture.then(|| run.capture_tool());
        // The pieces partition the segments, so the union of what each one enumerated
        // for itself *is* what the whole range touched — accumulated as they went
        // rather than walked a second time over every (segment, tile) pair, which is
        // the very cost `region_of` exists to keep off a long stroke.
        let dirty = std::mem::take(&mut run.dirty).into_iter().collect();
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
    /// The budget the range was flattened at ([`dynamics_setup`]), carried so a plan
    /// can re-cut a piece of the record the same way — see [`PlanCtx::tol`].
    tol: crate::path::FlattenTolerance,
    scene: StrokeScene<'a>,
    encoder: wgpu::CommandEncoder,
    /// Pooled scratch checked out for the whole run — the reservoir ping-pong and
    /// the bake pair, which carry the tool from one piece to the next. Handed back
    /// to the [`ScratchPool`] at [`Self::submit`], *after* that submit — the
    /// ordering the pool's whole reuse argument rests on.
    run_leases: Vec<Lease>,
    /// GPU objects scoped to the piece being recorded and **not** pooled: the
    /// selection mask, the uniform and instance buffers. Destroyed as soon as the
    /// piece is submitted (see [`Self::flush`]), so a stroke of any length costs one
    /// region's worth of transient memory rather than one per piece.
    piece: ScopedResources,
    /// Pooled scratch checked out for the piece being recorded — the region, the
    /// narrow aux, the snapshot and the cell scratch. Handed back at the piece's own
    /// submit ([`Self::flush`] / [`Self::submit`]), like [`Self::run_leases`].
    piece_leases: Vec<Lease>,
    /// Whether [`Self::draw`] has recorded a piece that [`Self::flush`] still owes a
    /// submit. Said outright rather than inferred from `piece` being non-empty, which
    /// is a coincidence of the two rather than the question being asked.
    piece_open: bool,
    /// Every tile the run has rewritten, accumulated as each piece enumerates its own.
    /// The pieces partition the range's segments, so this ends up the set a second
    /// walk over the whole stroke would have built — for none of the cost.
    dirty: BTreeSet<TileCoord>,
    /// Everything both render paths read off the record and the scene, resolved once
    /// (see [`StrokeConstants`](super::super::StrokeConstants)).
    consts: super::super::StrokeConstants,
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
    /// The reservoir's **residual** half (§6.7), ping-ponged in step with the colour
    /// above — the tip carries the rest of the colour it picked up, or a pigment
    /// space's black would come back off the tool as the grey its concentrations
    /// alone describe. `None` in a space with no residual, which allocates nothing.
    brush_resid_tex: Option<[wgpu::Texture; 2]>,
    brush_resid: Option<[wgpu::TextureView; 2]>,
    cur: usize,
    /// The segment's swept reservoir prefixes (fp32, so the per-fragment difference
    /// keeps its precision — see [`BAKE_FORMAT`]). Rebuilt per segment, so a single
    /// pair serves the whole stroke: nothing reads the last segment's bake.
    bake_load: wgpu::TextureView,
    bake_latm: wgpu::TextureView,
    /// The residual's own swept prefix, `∫res·m·dτ` — same format, same differencing,
    /// and recovered against the very denominator `bake_latm` is.
    bake_rlm: Option<wgpu::TextureView>,
}

impl<'a> DynamicsRun<'a> {
    /// Open the run: resolve the brush's textures, and put the tool in the state the
    /// stroke arrives at this range with — resumed from `tool`, or freshly charged.
    fn new(
        r: &'a StrokeRenderer,
        scene: StrokeScene<'a>,
        rec: &'a StrokeRecord,
        tol: crate::path::FlattenTolerance,
        tool: Option<&ToolState>,
    ) -> Self {
        let device = &r.ctx.device;
        let mut run_leases = Vec::new();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stark dynamics stroke"),
        });
        let consts = r.stroke_constants(rec, scene.surface);

        // The brush's swept-footprint prefix-τ (shared with the fast path) and its
        // plain coverage mask (the reservoir texels' own footprint weights).
        let prefix_view = r.tips.prefix_view(scene.assets, &rec.brush);
        let prefix_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark dynamics prefix bg"),
            layout: &r.dynamics.prefix_bgl,
            entries: &[desc::tex(0, &prefix_view)],
        });
        let cov = r.tips.coverage_view(scene.assets, &rec.brush);
        // Colour dynamics for the brush's own `add` paint — the same field and
        // lookup parameters as the fast path (see `deposit` in dynamics.wesl).
        let noise = r.tips.noise_view(&rec.brush.color_dynamics);

        // A stroke that starts fresh initializes its first reservoir by a render clear
        // (the driver does the f16 encode), hence RENDER_ATTACHMENT; one resuming from
        // a [`ToolState`] copies into it instead, hence the COPY pair — which also
        // carries the end state back out.
        let brush_usage = LOOP_USAGE
            | wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST;
        // A ping-pong pair out of the pool: fully written (cleared, copied into, or
        // stored texel-for-texel by the reservoir passes) before anything reads it,
        // which is the pool's no-zero-init contract (`scratch`).
        let brush_pair = |leases: &mut Vec<Lease>, labels: [&'static str; 2]| {
            let pair = labels.map(|label| {
                pooled(
                    &r.scratch,
                    device,
                    leases,
                    Key {
                        size: (BRUSH_RES, BRUSH_RES),
                        format: wgpu::TextureFormat::Rgba16Float,
                        usage: brush_usage,
                        label,
                    },
                )
            });
            let [(tex_a, view_a), (tex_b, view_b)] = pair;
            ([tex_a, tex_b], [view_a, view_b])
        };
        let (brush_color_tex, brush_color) = brush_pair(
            &mut run_leases,
            [
                "stark dynamics brush color a",
                "stark dynamics brush color b",
            ],
        );
        let (brush_aux_tex, brush_aux) = brush_pair(
            &mut run_leases,
            ["stark dynamics brush aux a", "stark dynamics brush aux b"],
        );
        // The residual's half of the ping-pong, allocated only where there is one.
        let (brush_resid_tex, brush_resid) = match r.color_space.has_resid().then(|| {
            brush_pair(
                &mut run_leases,
                [
                    "stark dynamics brush resid a",
                    "stark dynamics brush resid b",
                ],
            )
        }) {
            Some((t, v)) => (Some(t), Some(v)),
            None => (None, None),
        };
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
            if let (Some(src), Some(dst)) = (&t.resid, &brush_resid_tex) {
                encoder.copy_texture_to_texture(
                    src.as_image_copy(),
                    dst[0].as_image_copy(),
                    RESERVOIR_EXTENT,
                );
            }
        } else {
            // Init: latent = the brush's own colour, per-unit opacity = its alpha;
            // the carried amount starts at the pre-`charge` glob (0 = empty tool).
            let d = rec.brush.dynamics;
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stark dynamics brush init"),
                color_attachments: &[
                    Some(desc::attach(
                        &brush_color[0],
                        desc::clear_to(wgpu::Color {
                            r: consts.channels[0] as f64,
                            g: consts.channels[1] as f64,
                            b: consts.channels[2] as f64,
                            a: consts.channels[3] as f64,
                        }),
                    )),
                    Some(desc::attach(
                        &brush_aux[0],
                        desc::clear_to(wgpu::Color {
                            // Carried height = the pre-`charge` glob; the rest of
                            // the reservoir aux is unused (height is the only
                            // thing the tool carries, §6.1).
                            r: d.charge as f64,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                    )),
                    // A freshly charged tip holds the brush's own colour, and that
                    // colour's residual with it — the same clear, off the same
                    // constants (§6.7).
                    brush_resid.as_ref().map(|v| {
                        desc::attach(
                            &v[0],
                            desc::clear_to(wgpu::Color {
                                r: consts.resid[0] as f64,
                                g: consts.resid[1] as f64,
                                b: consts.resid[2] as f64,
                                a: 0.0,
                            }),
                        )
                    }),
                ][..2 + usize::from(brush_resid.is_some())],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        let mut bake = |label: &'static str| {
            pooled(
                &r.scratch,
                device,
                &mut run_leases,
                Key {
                    size: (BAKE_RES, BAKE_RES),
                    format: BAKE_FORMAT,
                    usage: LOOP_USAGE,
                    label,
                },
            )
            .1
        };
        let bake_load = bake("stark dynamics bake load");
        let bake_latm = bake("stark dynamics bake latm");
        let bake_rlm = r
            .color_space
            .has_resid()
            .then(|| bake("stark dynamics bake rlm"));
        Self {
            r,
            rec,
            tol,
            scene,
            encoder,
            run_leases,
            piece: ScopedResources::default(),
            piece_leases: Vec::new(),
            piece_open: false,
            dirty: BTreeSet::new(),
            consts,
            prefix_bg,
            cov,
            noise,
            brush_color_tex,
            brush_aux_tex,
            brush_color,
            brush_aux,
            brush_resid_tex,
            brush_resid,
            cur: 0,
            bake_load,
            bake_latm,
            bake_rlm,
        }
    }

    /// Evolve one region-sized piece of the stroke over `base`: composite the tiles
    /// under `segments` into a region, walk them through the loop, and slice the
    /// result back into fresh CoW tiles. The tool carries on from where the previous
    /// piece left it, and the canvas side needs no carrying — it is in `base`, which
    /// for a later piece is what the earlier ones wrote back.
    /// `settle` is set only for the piece that ends the stroke: see
    /// [`StrokeRenderer::render_dynamic`] and `dynamics.wesl::settle`.
    fn draw(&mut self, base: &TileMap, segments: &[Segment], settle: bool) -> TileMap {
        self.flush();
        let coords = affected_tiles(segments);
        // A piece holds at least one segment, and a segment covers at least one tile,
        // so the empty case cannot arise here — but it costs nothing to leave the
        // canvas alone if it ever did.
        let Some(RegionRect {
            halo,
            lo,
            origin: region_origin,
            w,
            h,
        }) = region_rect(&coords)
        else {
            return base.clone();
        };
        // The run's dirty set is the union of the pieces', which is why no caller has
        // to enumerate the whole range's tiles a second time.
        self.dirty.extend(coords.iter().copied());
        self.piece_open = true;
        let region = self.composite_region(base, &halo, region_origin, w, h);

        // ---- The bleed cadence's fire slots for this piece (§6.2), built before
        // the snapshot scratch is sized: a firing's window sweeps up to
        // [`BLEED_TRAVEL_QUANTUM`] radii where the piece's own segments may be
        // sub-pixel, so its coverage box can be the largest in the piece.
        let fires = bleed_fires(self.rec.brush.dynamics.bleed, segments);
        let under = self.snapshot_scratch(segments, &fires);

        // ---- The dispatch plan, one slot per dispatch, uploaded as one buffer the
        // loop reads through dynamic offsets.
        let ctx = PlanCtx {
            rec: self.rec,
            tol: self.tol,
            region_origin,
            dsize: under.size,
            consts: &self.consts,
            surface: self.scene.surface,
        };
        let plan = dynamics_plan(&ctx, segments, &fires, settle);
        // The cell scratch (§6.2), only when some slot actually takes the coarse
        // path — a small or hard tip allocates nothing and binds nothing.
        let cells = plan
            .iter()
            .any(|d| d.cell_groups.is_some())
            .then(|| self.cell_scratch(under.size));
        let stamp_buf = self.upload_plan(&plan);
        let bind = self.bind_piece(&region, &under, &stamp_buf, cells.as_ref());

        self.record_loop(&plan, &bind);
        self.write_back(base, &coords, lo, &region)
    }

    /// The piece's canvas region — a 1:1 copy of what lies under its segments — and
    /// the selection over it.
    ///
    /// Composited from the base tiles of the affected set plus a one-tile ring, so
    /// rewritten tiles' aprons read real neighbour content (§6.4). Rgba16Float
    /// throughout: it is both filterable and a core storage format, and matches the
    /// tile colour format of both colour spaces (asserted in `build_dynamics_kit`).
    fn composite_region(
        &mut self,
        base: &TileMap,
        halo: &[TileCoord],
        region_origin: Vec2,
        w: u32,
        h: u32,
    ) -> Region {
        let r = self.r;
        let kit = &r.dynamics;
        let device = &r.ctx.device;

        // COPY_SRC because the write-back cuts the tiles straight out of these by
        // texture copy ([`Self::write_back`]). Pooled: the composite's clearing load
        // op rewrites every texel, so stale contents never show (`scratch`).
        let region_usage =
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC | LOOP_USAGE;
        let mut region_tex = |label: &'static str| {
            pooled(
                &r.scratch,
                device,
                &mut self.piece_leases,
                Key {
                    size: (w, h),
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: region_usage,
                    label,
                },
            )
        };
        let (color_tex, color) = region_tex("stark dynamics region color");
        let (_, aux) = region_tex("stark dynamics region aux");
        // The region's residual (§6.7), in the same format for the same reason: it is
        // the rest of the colour beside it, and the loop reads and writes the two at
        // exactly the same points.
        let (resid_tex, resid) = match r
            .color_space
            .has_resid()
            .then(|| region_tex("stark dynamics region resid"))
        {
            Some((t, v)) => (Some(t), Some(v)),
            None => (None, None),
        };

        // Composite pass: base tiles → region, 1:1 with canvas px. The compositor's
        // own `ViewUniform` — this path binds its own buffer to the very same
        // `composite.wesl`, so it wants that struct rather than a second declaration
        // of it that a comment asks to be kept in step (§6.2).
        let (sx, sy) = (2.0 / w as f32, -2.0 / h as f32);
        let view = view_uniform(
            // Diagonal: the region is axis-aligned with the canvas whatever angle the
            // *screen* view happens to be at.
            [sx, 0.0, 0.0, sy],
            Vec2::new(-region_origin.x * sx - 1.0, -region_origin.y * sy + 1.0),
            // Zoom reaches only the selection outline, and no chrome is drawn into a
            // working region — this is a buffer the loop evolves, not a picture.
            0.0,
        );
        let view_buf = self.piece.buffer(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("stark dynamics region view"),
                contents: bytemuck::bytes_of(&view),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        ));
        let view_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark dynamics region view bg"),
            layout: &kit.composite_view_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: view_buf.as_entire_binding(),
                },
                desc::samp(1, &kit.composite_sampler),
            ],
        });
        let mut tile_origins: Vec<TileInstance> = Vec::new();
        let mut tile_bgs: Vec<wgpu::BindGroup> = Vec::new();
        for coord in halo {
            if let Some(tile) = base.get(coord) {
                tile_origins.push(TileInstance {
                    origin: coord.origin().to_array(),
                    opacity: 1.0,
                });
                let mut entries = vec![
                    desc::tex(0, tile.color_view()),
                    desc::tex(1, tile.aux_view()),
                ];
                // Binding 2 is the tile's own residual. A resident tile in a pigment
                // space always has one, so `Zeroes` never stands in here.
                if let Some(v) = tile.resid_view() {
                    entries.push(desc::tex(2, v));
                }
                tile_bgs.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("stark dynamics region tile bg"),
                    layout: &kit.composite_tile_bgl,
                    entries: &entries,
                }));
            }
        }
        let tile_inst = (!tile_origins.is_empty()).then(|| {
            self.piece.buffer(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("stark dynamics region tile instances"),
                    contents: bytemuck::cast_slice(&tile_origins),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
            )
        });
        {
            let (att, att_n) = desc::tile_attachments(&color, &aux, resid.as_ref(), desc::CLEAR);
            let mut pass = self.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stark dynamics region composite"),
                color_attachments: &att[..att_n],
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

        // The selection over this region (§6.8), gathered from the same halo tiles the
        // paint came from, so it is 1:1 with the colour/aux regions. An unrestricted
        // selection binds the 1×1 constant instead — the loop's masked reads then
        // return 1 everywhere and the stroke behaves exactly as before.
        let sel_mask = if self.scene.selection.is_universal() {
            r.selection.constant(1.0).clone()
        } else {
            let (tex, view) = r.selection.region_mask(
                &mut self.encoder,
                self.scene.selection,
                halo,
                region_origin,
                w,
                h,
            );
            self.piece.texture(tex);
            view
        };

        Region {
            color_tex,
            color,
            aux,
            resid_tex,
            resid,
            sel_mask,
        }
    }

    /// The footprint snapshot scratch: the copy that gives `deposit` and `settle`
    /// something to read while they storage-write the region.
    ///
    /// The rect must cover any one segment's coverage box — its swept arc plus the tip
    /// riding along it — which is measured rather than bounded analytically, since
    /// `coverage_bounds` is already the exact box and a curved sweep has no
    /// closed-form "worst rotation" to fall back on. Sized from *this* piece's
    /// segments, so a piece drawn with a fine tip pays for a fine tip.
    ///
    /// The fit against every dispatch rect is structural, not arithmetic kept in
    /// step: both sides go through `plan::rect_extent` — [`snapshot_size`] takes its
    /// maximum over the piece's own coverage boxes, and `plan::dispatch_rect`
    /// asserts each rect against it — so no rect a piece can build overruns the
    /// scratch that piece sized.
    fn snapshot_scratch(&mut self, segments: &[Segment], fires: &[(usize, Segment)]) -> Snapshot {
        let r = self.r;
        let device = &r.ctx.device;
        // Rounded up to a coarse quantum for the pool's sake alone: the measured
        // maximum drifts a few texels per fold as the tail's geometry evolves, which
        // would make nearly every checkout a pool miss. Nothing the shaders compute
        // changes — stores and reads are gated by the slot rects and the sweep test,
        // and the `textureDimensions` bounds only widen onto texels those gates
        // reject — so the round-up moves no pixel; `dsize` downstream is this
        // rounded square, on both sides of every fit assertion.
        let size = snapshot_size(segments, fires).next_multiple_of(SNAPSHOT_QUANTUM);
        let mut under_tex = |label: &'static str| {
            pooled(
                &r.scratch,
                device,
                &mut self.piece_leases,
                Key {
                    size: (size, size),
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: LOOP_USAGE,
                    label,
                },
            )
            .1
        };
        Snapshot {
            size,
            color: under_tex("stark dynamics under color"),
            aux: under_tex("stark dynamics under aux"),
            resid: r
                .color_space
                .has_resid()
                .then(|| under_tex("stark dynamics under resid")),
        }
    }

    /// The footprint-cell scratch (§6.2): where `cell_hoist` leaves the per-cell
    /// means for `deposit_coarse` to read back. Sized by the same structural-fit
    /// relation the snapshot scratch uses — [`cell_scratch_size`] of the piece's own
    /// `dsize`, which `cell_geometry` asserted every slot's hoist grid against — and
    /// leased on the piece like the region, for the same lifetime argument.
    fn cell_scratch(&mut self, dsize: u32) -> Cells {
        let r = self.r;
        let device = &r.ctx.device;
        let size = cell_scratch_size(dsize);
        let mut tex = |label: &'static str| {
            pooled(
                &r.scratch,
                device,
                &mut self.piece_leases,
                Key {
                    size: (size, size),
                    format: BAKE_FORMAT,
                    usage: LOOP_USAGE,
                    label,
                },
            )
            .1
        };
        Cells {
            tool: tex("stark dynamics cell tool"),
            lat: tex("stark dynamics cell lat"),
            res: r
                .color_space
                .has_resid()
                .then(|| tex("stark dynamics cell res")),
        }
    }

    /// The plan's uniform slots, one [`UNIFORM_STRIDE`]-aligned window each — dynamic
    /// uniform offsets being the standard way to vary a uniform across dispatches
    /// within one pass.
    ///
    /// Registered on the *piece*, like the region it works on and for the same reason:
    /// nothing past this piece's own submission reads it, so holding it for the whole
    /// run would make a long stroke's peak cost scale with the number of pieces —
    /// which is what [`MAX_REGION_DIM`](super::MAX_REGION_DIM) exists to prevent.
    /// Destroying it at the next [`Self::flush`] is safe on exactly the argument the
    /// region already rests on: `flush` submits before it destroys, and WebGPU defers
    /// the real free until that submission retires.
    fn upload_plan(&mut self, plan: &[LoopDispatch]) -> wgpu::Buffer {
        let r = self.r;
        let mut data = vec![0u8; plan.len() * UNIFORM_STRIDE];
        for (i, d) in plan.iter().enumerate() {
            let at = i * UNIFORM_STRIDE;
            data[at..at + SLOT].copy_from_slice(bytemuck::bytes_of(&d.slot));
        }
        let buf = self
            .piece
            .buffer(r.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("stark dynamics stamps"),
                size: data.len() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        r.ctx.queue.write_buffer(&buf, 0, &data);
        buf
    }

    /// Every bind group the loop switches between while recording one piece.
    ///
    /// `params` binds a single slot-sized window of `stamp_buf` whose dynamic offset
    /// selects the dispatch, so all of these are built once per piece and the loop
    /// varies only the offset.
    fn bind_piece(
        &self,
        region: &Region,
        under: &Snapshot,
        stamp_buf: &wgpu::Buffer,
        cells: Option<&Cells>,
    ) -> PieceBindings {
        let r = self.r;
        let kit = &r.dynamics;
        let device = &r.ctx.device;
        let params = || wgpu::BindGroupEntry {
            binding: b::ST,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: stamp_buf,
                offset: 0,
                size: wgpu::BufferSize::new(SLOT as u64),
            }),
        };
        let samp = || desc::samp(b::SAMP, &kit.exchange_sampler);
        let mut snapshot_entries = vec![
            params(),
            desc::tex(b::REGION_COLOR, &region.color),
            desc::tex(b::REGION_AUX, &region.aux),
            desc::tex(b::UNDER_COLOR_W, &under.color),
            desc::tex(b::UNDER_AUX_W, &under.aux),
        ];
        push_resid(
            &mut snapshot_entries,
            &[
                (b::REGION_RESID, region.resid.as_ref()),
                (b::UNDER_RESID_W, under.resid.as_ref()),
            ],
        );
        let snapshot = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark dynamics snapshot bg"),
            layout: &kit.snapshot_bgl,
            entries: &snapshot_entries,
        });
        // `exchange` comes in two flavours for the reservoir ping-pong: each reads one
        // half and writes the other.
        let exchange = std::array::from_fn(|i| {
            let mut entries = vec![
                params(),
                desc::tex(b::REGION_COLOR, &region.color),
                desc::tex(b::REGION_AUX, &region.aux),
                desc::tex(b::UNDER_COLOR_W, &under.color),
                desc::tex(b::UNDER_AUX_W, &under.aux),
                samp(),
                desc::tex(b::COV_TEX, &self.cov),
                desc::tex(b::BRUSH_SRC_COLOR, &self.brush_color[i]),
                desc::tex(b::BRUSH_SRC_AUX, &self.brush_aux[i]),
                desc::tex(b::BRUSH_DST_COLOR_W, &self.brush_color[1 - i]),
                desc::tex(b::BRUSH_DST_AUX_W, &self.brush_aux[1 - i]),
                desc::tex(b::SEL_MASK, &region.sel_mask),
            ];
            // The residual ping-pongs on the same phase as the colour: read `i`,
            // write `1 - i`, or the tool's two halves would drift apart.
            push_resid(
                &mut entries,
                &[
                    (b::REGION_RESID, region.resid.as_ref()),
                    (b::UNDER_RESID_W, under.resid.as_ref()),
                    (b::BRUSH_SRC_RESID, self.brush_resid.as_ref().map(|v| &v[i])),
                    (
                        b::BRUSH_DST_RESID_W,
                        self.brush_resid.as_ref().map(|v| &v[1 - i]),
                    ),
                ],
            );
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("stark dynamics exchange bg"),
                layout: &kit.exchange_bgl,
                entries: &entries,
            })
        });
        // One bake bind group per reservoir phase; the deposit reads only the baked
        // result, so it no longer needs the ping-pong at all.
        let bake = std::array::from_fn(|i| {
            let mut entries = vec![
                params(),
                samp(),
                desc::tex(b::BRUSH_SRC_COLOR, &self.brush_color[i]),
                desc::tex(b::BRUSH_SRC_AUX, &self.brush_aux[i]),
                desc::tex(b::BAKE_LOAD_W, &self.bake_load),
                desc::tex(b::BAKE_LATM_W, &self.bake_latm),
            ];
            push_resid(
                &mut entries,
                &[
                    (b::BRUSH_SRC_RESID, self.brush_resid.as_ref().map(|v| &v[i])),
                    (b::BAKE_RLM_W, self.bake_rlm.as_ref()),
                ],
            );
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("stark dynamics bake bg"),
                layout: &kit.bake_bgl,
                entries: &entries,
            })
        });
        // The two passes that lay paint read the residual's baked prefix and its
        // snapshot, and write the region's residual back — the same three roles the
        // colour's 19/11/13 play beside them.
        let resid_write = [
            (b::BAKE_RLM, self.bake_rlm.as_ref()),
            (b::UNDER_RESID, under.resid.as_ref()),
            (b::REGION_RESID_W, region.resid.as_ref()),
        ];
        let mut deposit_entries = vec![
            params(),
            samp(),
            desc::tex(b::BAKE_LOAD, &self.bake_load),
            desc::tex(b::BAKE_LATM, &self.bake_latm),
            desc::tex(b::UNDER_COLOR, &under.color),
            desc::tex(b::UNDER_AUX, &under.aux),
            desc::tex(b::REGION_COLOR_W, &region.color),
            desc::tex(b::REGION_AUX_W, &region.aux),
            desc::tex(b::DYN_NOISE_TEX, &self.noise),
            desc::samp(b::DYN_NOISE_SAMP, &r.tips.noise_sampler),
            desc::tex(b::SEL_MASK, &region.sel_mask),
            desc::tex(b::SURFACE_TEX, &self.scene.surface.view),
        ];
        push_resid(&mut deposit_entries, &resid_write);
        let deposit = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark dynamics deposit bg"),
            layout: &kit.deposit_bgl,
            entries: &deposit_entries,
        });
        // The pen-up, which reads the reservoir only through its own `bake` — so unlike
        // `exchange` it needs no bind group per ping-pong half; the bake's does that.
        let mut settle_entries = vec![
            params(),
            desc::tex(b::BAKE_LOAD, &self.bake_load),
            desc::tex(b::BAKE_LATM, &self.bake_latm),
            desc::tex(b::UNDER_COLOR, &under.color),
            desc::tex(b::UNDER_AUX, &under.aux),
            desc::tex(b::REGION_COLOR_W, &region.color),
            desc::tex(b::REGION_AUX_W, &region.aux),
            desc::tex(b::SEL_MASK, &region.sel_mask),
            // The ground: the pen-up delivery is a deposit like any other, and is
            // gated by the same tooth (§6.4).
            desc::tex(b::SURFACE_TEX, &self.scene.surface.view),
        ];
        push_resid(&mut settle_entries, &resid_write);
        let settle = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark dynamics settle bg"),
            layout: &kit.settle_bgl,
            entries: &settle_entries,
        });
        // The coarse pair's groups (§6.2), only for a piece whose plan hoists at all.
        // The hoist reads the same baked prefixes the exact deposit's front half did;
        // the coarse deposit swaps those two bindings for the cell means and keeps
        // everything else the exact one binds.
        let coarse = cells.map(|cl| {
            let mut hoist_entries = vec![
                params(),
                desc::tex(b::BAKE_LOAD, &self.bake_load),
                desc::tex(b::BAKE_LATM, &self.bake_latm),
                desc::tex(b::CELL_TOOL_W, &cl.tool),
                desc::tex(b::CELL_LAT_W, &cl.lat),
            ];
            push_resid(
                &mut hoist_entries,
                &[
                    (b::BAKE_RLM, self.bake_rlm.as_ref()),
                    (b::CELL_RES_W, cl.res.as_ref()),
                ],
            );
            let hoist = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("stark dynamics cell hoist bg"),
                layout: &kit.hoist_bgl,
                entries: &hoist_entries,
            });
            let mut coarse_entries = vec![
                params(),
                desc::tex(b::UNDER_COLOR, &under.color),
                desc::tex(b::UNDER_AUX, &under.aux),
                desc::tex(b::REGION_COLOR_W, &region.color),
                desc::tex(b::REGION_AUX_W, &region.aux),
                desc::tex(b::DYN_NOISE_TEX, &self.noise),
                desc::samp(b::DYN_NOISE_SAMP, &r.tips.noise_sampler),
                desc::tex(b::SEL_MASK, &region.sel_mask),
                desc::tex(b::SURFACE_TEX, &self.scene.surface.view),
                desc::tex(b::CELL_TOOL, &cl.tool),
                desc::tex(b::CELL_LAT, &cl.lat),
            ];
            push_resid(
                &mut coarse_entries,
                &[
                    (b::UNDER_RESID, under.resid.as_ref()),
                    (b::REGION_RESID_W, region.resid.as_ref()),
                    (b::CELL_RES, cl.res.as_ref()),
                ],
            );
            let deposit = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("stark dynamics deposit coarse bg"),
                layout: &kit.deposit_coarse_bgl,
                entries: &coarse_entries,
            });
            CoarseBindings { hoist, deposit }
        });
        PieceBindings {
            snapshot,
            exchange,
            bake,
            deposit,
            coarse,
            settle,
        }
    }

    /// Record the loop: bake → exchange (+ snapshot) → deposit per segment, in
    /// stroke order. One compute pass; the implicit barriers between dispatches give
    /// the sequential semantics, and usage scopes are per-dispatch, so the region may
    /// be sampled by one dispatch and storage-written by the next.
    ///
    /// `exchange` comes *before* `deposit` and not after: the two are the two halves
    /// of one transfer, and they only add up if both read the canvas and the
    /// reservoir as this segment found them (`dynamics.wesl::exchange_at`).
    ///
    /// `self.cur` outlives the pass: it names the reservoir texture holding the
    /// tool's state, so after the last dispatch it names the state this piece ends
    /// in — which is what the next piece, or the next range, resumes from.
    fn record_loop(&mut self, plan: &[LoopDispatch], bind: &PieceBindings) {
        let kit = &self.r.dynamics;
        let mut cur = self.cur;
        let prefix_bg = &self.prefix_bg;
        let mut cpass = self
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stark dynamics stamp loop"),
                timestamp_writes: None,
            });
        // The prefix-τ rides at group 1 for `bake`, `deposit` and `settle`. Re-bound
        // after every pipeline switch: changing to a pipeline whose group-0 layout
        // differs invalidates the groups above it, and every consumer is reached only
        // across such a switch.
        for (i, d) in plan.iter().enumerate() {
            let off = (i * UNIFORM_STRIDE) as u32;
            match d.kind {
                // The tool plays no part in the lateral flux, so `cur` — the reservoir
                // ping-pong — stays exactly where the previous segment left it. The
                // standalone snapshot pipeline rather than the exchange's tail,
                // because there is no exchange dispatch to ride in.
                SlotKind::Bleed => {
                    cpass.set_pipeline(&kit.snapshot_pipeline);
                    cpass.set_bind_group(0, &bind.snapshot, &[off]);
                    cpass.dispatch_workgroups(d.groups.0, d.groups.1, 1);
                    cpass.set_pipeline(&kit.deposit_pipeline);
                    cpass.set_bind_group(0, &bind.deposit, &[off]);
                    cpass.set_bind_group(1, prefix_bg, &[]);
                    cpass.dispatch_workgroups(d.groups.0, d.groups.1, 1);
                }
                SlotKind::Segment => {
                    // Bake this segment's swept reservoir prefix first — it folds in the
                    // tip's current orientation as well as the reservoir state.
                    cpass.set_pipeline(&kit.bake_pipeline);
                    cpass.set_bind_group(0, &bind.bake[cur], &[off]);
                    cpass.set_bind_group(1, prefix_bg, &[]);
                    // One BAKE_RES-wide workgroup per row: the shader's scan width is a
                    // constant, so the two must agree.
                    cpass.dispatch_workgroups(1, BAKE_RES, 1);
                    // Then the tool's own side of this segment's transfer, off the
                    // region as the segment found it. Reads `cur` and writes the other
                    // half, so the next segment's bake sees a tool that has actually
                    // travelled and reloaded.
                    //
                    // The footprint `snapshot` rides in the tail of this same grid: it
                    // depends on nothing the exchange writes and the deposit needs
                    // both, so the barrier that used to sit between them bought no
                    // ordering. Hence the widened x — reservoir groups first, footprint
                    // groups after — and a y tall enough for the taller of the two
                    // (`dynamics.wesl::exchange`).
                    cpass.set_pipeline(&kit.exchange_pipeline);
                    cpass.set_bind_group(0, &bind.exchange[cur], &[off]);
                    cpass.dispatch_workgroups(
                        RESERVOIR_GROUPS.0 + d.groups.0,
                        RESERVOIR_GROUPS.1.max(d.groups.1),
                        1,
                    );
                    // The canvas's half: exact per texel, or — where the tip's
                    // shoulder allows (`footprint_cell`) — hoisted once per cell and
                    // applied over the same texel grid. The hoist reads the bake this
                    // segment just wrote and nothing the exchange writes, so its
                    // place in the chain costs one more serialized dispatch only on
                    // the wide tips that are texel-bound rather than dispatch-bound.
                    match d
                        .cell_groups
                        .map(|cg| (cg, bind.coarse.as_ref().expect("a coarse slot binds cells")))
                    {
                        Some((cg, cb)) => {
                            cpass.set_pipeline(&kit.hoist_pipeline);
                            cpass.set_bind_group(0, &cb.hoist, &[off]);
                            cpass.set_bind_group(1, prefix_bg, &[]);
                            cpass.dispatch_workgroups(cg.0, cg.1, 1);
                            cpass.set_pipeline(&kit.deposit_coarse_pipeline);
                            cpass.set_bind_group(0, &cb.deposit, &[off]);
                            cpass.dispatch_workgroups(d.groups.0, d.groups.1, 1);
                        }
                        None => {
                            cpass.set_pipeline(&kit.deposit_pipeline);
                            cpass.set_bind_group(0, &bind.deposit, &[off]);
                            cpass.set_bind_group(1, prefix_bg, &[]);
                            cpass.dispatch_workgroups(d.groups.0, d.groups.1, 1);
                        }
                    }
                    cur = 1 - cur;
                }
                // The pen-up: snapshot the final footprint, bake the standing tip's
                // remaining-pass delivery off the reservoir the last segment left
                // (`cur` still names it — the slot's zero travel switches the bake onto
                // the settle's weighted integral), then settle the transfer the stroke
                // stopped in the middle of. The tool is not written back, so `cur` is
                // left alone here too.
                SlotKind::Settle => {
                    cpass.set_pipeline(&kit.snapshot_pipeline);
                    cpass.set_bind_group(0, &bind.snapshot, &[off]);
                    cpass.dispatch_workgroups(d.groups.0, d.groups.1, 1);
                    cpass.set_pipeline(&kit.bake_pipeline);
                    cpass.set_bind_group(0, &bind.bake[cur], &[off]);
                    cpass.set_bind_group(1, prefix_bg, &[]);
                    cpass.dispatch_workgroups(1, BAKE_RES, 1);
                    cpass.set_pipeline(&kit.settle_pipeline);
                    cpass.set_bind_group(0, &bind.settle, &[off]);
                    cpass.set_bind_group(1, prefix_bg, &[]);
                    cpass.dispatch_workgroups(d.groups.0, d.groups.1, 1);
                }
            }
        }
        self.cur = cur;
    }

    /// Slice each affected tile's full `TILE_TEX` block out of the shared region into
    /// a fresh CoW tile → aprons stay bit-identical to neighbour interiors (§6.4), and
    /// the wide region aux narrows to the persistent one (height).
    ///
    /// One region-sized narrow pass, then plain texture copies: the colour and
    /// residual tiles are the region's own formats, so a copy is exact, and the aux
    /// copies out of the narrowed texture the single pass wrote — where this used to
    /// record a render pass per tile, ~30 of them per fold under a wide tip, whose
    /// fixed pass cost was most of the write-back's 15% share of a live gesture.
    /// Rounding is untouched: a copy is bit-exact, and the narrow pass render-writes
    /// a loaded f16 value back to its own lattice point (see `slice.wesl`).
    ///
    /// `lo` is the region's *interior* origin — the top-left tile origin, an apron in
    /// from the region rectangle — so a tile's offset into the region is measured
    /// against it. Offsets are integral and non-negative by [`region_rect`]'s
    /// construction, and the far edge of the last tile's block is exactly the region's
    /// extent, so every copy is in bounds (a violation is a loud validation error,
    /// where the draws this replaced would have silently read out of bounds).
    fn write_back(
        &mut self,
        base: &TileMap,
        coords: &BTreeSet<TileCoord>,
        lo: Vec2,
        region: &Region,
    ) -> TileMap {
        let r = self.r;
        let kit = &r.dynamics;
        let device = &r.ctx.device;

        // The one channel that changes representation, narrowed once over the whole
        // region (§6.7's aux stays a single height channel on the persistent tile).
        // Pooled; the pass's clearing load op rewrites every texel.
        let (narrow, narrow_view) = pooled(
            &r.scratch,
            device,
            &mut self.piece_leases,
            Key {
                size: (region.color_tex.width(), region.color_tex.height()),
                format: r.color_space.aux_format(),
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                label: "stark dynamics narrow aux",
            },
        );
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark dynamics slice bg"),
            layout: &kit.slice_bgl,
            entries: &[desc::tex(0, &region.aux)],
        });
        {
            let mut pass = self.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stark dynamics slice"),
                color_attachments: &[Some(desc::attach(&narrow_view, desc::CLEAR))],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&kit.slice_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }

        let mut new_map = base.clone();
        for coord in coords {
            let dst = r.acquire_tile(self.scene.pool, AllocSource::DynamicsWriteback);
            let off = coord.origin() - lo;
            dst.copy_from_region(
                &mut self.encoder,
                &region.color_tex,
                &narrow,
                region.resid_tex.as_ref(),
                wgpu::Origin3d {
                    x: off.x as u32,
                    y: off.y as u32,
                    z: 0,
                },
            );
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
        if !self.piece_open {
            return;
        }
        self.piece_open = false;
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
        // Leases go back only now, behind the submit — the pool's reuse argument.
        for lease in self.piece_leases.drain(..) {
            self.r.scratch.give(lease);
        }
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
            // Carried across ranges with the colour, and for the same reason: a tip
            // that lifted black paint has to still be carrying black when the next
            // pointer move resumes it (§6.7).
            resid: self
                .brush_resid_tex
                .as_ref()
                .map(|t| copy_out(&mut self.encoder, &t[self.cur], "stark tool state resid")),
        }
    }

    /// Close the run: submit what is still recorded, then destroy the piece's
    /// unpooled resources (safe: WebGPU defers the real free past the submitted
    /// work — see the [`ScopedResources`] docs for why waiting on JS GC OOMs the
    /// tab) and hand every scratch lease back to the pool, behind that same submit.
    /// What [`Self::capture_tool`] handed back is deliberately *not* among them: it
    /// outlives this call by design.
    fn submit(mut self) {
        self.r.ctx.queue.submit([self.encoder.finish()]);
        drop(self.piece);
        for lease in self.piece_leases.drain(..).chain(self.run_leases.drain(..)) {
            self.r.scratch.give(lease);
        }
    }
}

/// One piece's canvas region: a 1:1 copy of the canvas under its segments, which the
/// loop evolves in place before [`DynamicsRun::write_back`] slices it into tiles.
///
/// The colour and residual carry their textures beside the views: the write-back
/// cuts tiles out of them by texture copy, and a copy is addressed by texture where
/// every pass binding is addressed by view. Both handles are `Arc`-backed clones of
/// the one piece-scoped allocation.
struct Region {
    color_tex: wgpu::Texture,
    color: wgpu::TextureView,
    aux: wgpu::TextureView,
    /// The region's residual (§6.7), in a space that has one — evolved in place by the
    /// same dispatches that evolve the colour, and sliced back with it.
    resid_tex: Option<wgpu::Texture>,
    resid: Option<wgpu::TextureView>,
    /// The selection over the region (§6.8) — its own gathered mask, or the 1×1
    /// constant that stands in for an unrestricted selection.
    sel_mask: wgpu::TextureView,
}

/// The footprint snapshot scratch, and the square it was sized to. Every dispatch
/// rect in the piece's plan is clamped to `size`, so the two travel together.
struct Snapshot {
    size: u32,
    color: wgpu::TextureView,
    aux: wgpu::TextureView,
    /// The snapshot's residual half, copied on exactly the texels the colour is.
    resid: Option<wgpu::TextureView>,
}

/// Append the residual bindings a layout declares (§6.7) to a bind group's entries.
///
/// A tail rather than an interleave: entries are keyed by binding number, so their
/// order in the slice is free, and keeping the residual out of the plain list is what
/// leaves that list readable as the thing it still is. Every `Option` handed here is
/// `Some` exactly when the colour space has a residual, so a group takes its whole
/// tail or none of it — the same all-or-nothing the `[..n + k]` layout slices in
/// [`build_dynamics_kit`](super::kit::build_dynamics_kit) express on the other side.
fn push_resid<'v>(
    entries: &mut Vec<wgpu::BindGroupEntry<'v>>,
    want: &[(u32, Option<&'v wgpu::TextureView>)],
) {
    for (binding, view) in want {
        if let Some(v) = view {
            entries.push(desc::tex(*binding, v));
        }
    }
}

/// The footprint-cell scratch (§6.2): the coarse deposit's per-cell means, sized by
/// [`cell_scratch_size`] of the piece's snapshot square so every hoist grid a slot
/// can ask for fits by construction.
struct Cells {
    tool: wgpu::TextureView,
    lat: wgpu::TextureView,
    /// The residual's cell mean, allocated exactly when the colour space has one.
    res: Option<wgpu::TextureView>,
}

/// The bind groups one piece's dispatches switch between. Built once per piece
/// because every dispatch varies only the stamp uniform's dynamic offset; the
/// reservoir ping-pong is the one thing that needs a pair.
struct PieceBindings {
    snapshot: wgpu::BindGroup,
    exchange: [wgpu::BindGroup; 2],
    bake: [wgpu::BindGroup; 2],
    deposit: wgpu::BindGroup,
    /// The coarse pair's groups — `Some` exactly when the piece allocated a cell
    /// scratch, i.e. when some slot's [`footprint_cell`](super::plan) beat 1.
    coarse: Option<CoarseBindings>,
    settle: wgpu::BindGroup,
}

/// The two bind groups a coarse slot dispatches with (§6.2).
struct CoarseBindings {
    hoist: wgpu::BindGroup,
    deposit: wgpu::BindGroup,
}

/// What every texture the loop reads and writes needs to be: sampled by one
/// dispatch, storage-written by the next.
const LOOP_USAGE: wgpu::TextureUsages =
    wgpu::TextureUsages::TEXTURE_BINDING.union(wgpu::TextureUsages::STORAGE_BINDING);

/// Check `key` out of the scratch pool onto `leases`, handing back `Arc` clones of
/// the leased texture and its view. The lease itself rides in `leases` until the
/// GPU work recorded against it is submitted, at which point the run gives it back
/// ([`DynamicsRun::flush`] / [`DynamicsRun::submit`]).
fn pooled(
    pool: &ScratchPool,
    device: &wgpu::Device,
    leases: &mut Vec<Lease>,
    key: Key,
) -> (wgpu::Texture, wgpu::TextureView) {
    let lease = pool.take(device, key);
    let out = (lease.tex.clone(), lease.view.clone());
    leases.push(lease);
    out
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
