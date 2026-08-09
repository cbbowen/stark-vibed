//! The brush-dynamics path (§6.2): a serial swept-exchange loop that lets
//! a stroke pick paint up off the canvas and put it back down.
//!
//! Where the swept path composes by summing optical depth — and so can draw its
//! segments in any order — this one is *sequential* by nature: what the tip carries
//! into a segment is what the previous segment left on it. The loop runs on the GPU
//! (no CPU readback, so it works on WebGPU) with a per-segment x per-lateral-band
//! reservoir texture standing in for the tip's load.

use wgpu::util::DeviceExt;

use crate::colorspace::ColorSpace;
use crate::document::StrokeRecord;
use std::collections::BTreeSet;

use crate::geom::{TileCoord, Vec2};
use crate::gpu::composite::view_uniform;
use crate::gpu::desc;
use crate::gpu::tile::{AllocSource, SCRATCH_AUX_FORMAT, TileMap};

use super::budget::{
    BLEED_TRAVEL_QUANTUM, MAX_BLEED_FIRES_PER_SEGMENT, TAU_PER_PASS, WICK_TRAVEL_QUANTUM,
    bleed_stencil, flatten_tolerance,
};
use super::segments::{
    Segment, affected_tiles, chunk_segments, coverage_bounds, generate_segments_in, region_rect,
    segment_fits_region,
};
use super::{
    ScopedResources, StrokeCarry, StrokeRenderer, StrokeScene, StrokeSpans, ToolState,
    UNIFORM_STRIDE,
};

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
/// fp32, for the same reason the prefix-τ volume is: every fragment reads it as a
/// *difference* of two prefix sums (§6.2).
const BAKE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

// The region composite runs `composite.wesl`, so it draws that shader's own
// per-instance record (§6.10) — which this module used to declare a second
// `#[repr(C)]` copy of, under a different name.
use stark_shaders::mirror::composite::Instance as TileInstance;

// The write-back's uniform, generated from `slice.wesl`'s own declaration (§6.7):
// the tile texture's top-left in region texels.
use stark_shaders::mirror::slice::Params as SliceUniform;

/// One tile's window into the write-back's offset buffer — the `min_binding_size` the
/// slice layout declares, taken from the struct rather than written down.
const SLICE_SLOT: u64 = std::mem::size_of::<SliceUniform>() as u64;

// The `Stamp` uniform, generated from `dynamics.wesl`'s own declaration at build
// time (`stark-shaders/build/mirror.rs`) — lanes, offsets, and the documentation of
// what each lane holds, which is now on the generated fields.
//
// It used to be written out here as well, nine `[f32; 4]` fields against the shader's
// nine `vec4`s, with a second copy of the lane map in the doc comments and nothing
// checking either half. Both halves drifted: this one still described `e.zw` as the
// midpoint `exchange` samples the canvas at, some time after the shader had stopped
// reading the lane at all. The shader decides how the lanes are *read*, so it is now
// the only place they are written down.
//
// Every slot is still a pure function of the `StrokeRecord` and the piece's own
// geometry, computed in plain CPU float math, so replay is deterministic (§12.1).
use stark_shaders::mirror::dynamics::Stamp;

/// One slot's window into the stamp buffer, and the `min_binding_size` its layout
/// declares — both of which have to be `Stamp`'s own size, so they are taken from it
/// rather than written down.
const SLOT: usize = std::mem::size_of::<Stamp>();

/// One slot of the sequential swept-exchange loop (§6.2), and the dispatches it
/// stands for.
struct LoopDispatch {
    slot: Stamp,
    /// Workgroup counts for the slot's footprint work — the `deposit`, and the
    /// `snapshot` that rides in `exchange`'s grid. The slot's own coverage box
    /// rather than the piece-wide worst-case square, so an axis-aligned sweep pays for
    /// the ~4·r² texels its footprint can reach instead of the ~10·r² a diagonal one
    /// might have needed.
    groups: (u32, u32),
    kind: SlotKind,
}

/// Which of the loop's three dispatch shapes a slot takes (§6.2).
///
/// A tag rather than a pair of flags because the three are genuinely different
/// sequences over the same uniform, and only one of them touches the tool: the
/// reservoir ping-pong advances on a [`Segment`](SlotKind::Segment) and on nothing
/// else, which is easier to see as an arm than as an early `continue` plus a tail
/// block indexed past the end of the loop.
enum SlotKind {
    /// A stretch of painting: `wick` → `bake` → `exchange` (+ `snapshot`) →
    /// `deposit`.
    ///
    /// `wick_steps` is how many `wick` firings fall during it
    /// ([`WICK_TRAVEL_QUANTUM`](super::WICK_TRAVEL_QUANTUM)). Usually 0 or 1 — the
    /// wick keeps its own cadence, so a segment shorter than the quantum often skips
    /// it entirely and a long one pays for several.
    Segment { wick_steps: u32 },
    /// A dedicated **bleed slot**: a quad whose sweep is one firing of the bleed
    /// cadence's travel window, with every vertical rate and the source zeroed.
    /// Dispatched as `snapshot` + `deposit` alone — the tool plays no part, so there
    /// is nothing to wick, bake or exchange, and the reservoir ping-pong is left
    /// exactly where the previous segment put it.
    Bleed,
    /// The pen-up: `snapshot` → `bake` → `settle`. At most one per plan, and always
    /// its last slot — the transfer the tip was still in the middle of when the
    /// stroke stopped (`dynamics.wesl::settle`).
    Settle,
}

/// Workgroup counts for the reservoir passes (`wick`, and `exchange`'s own half).
/// `exchange` is dispatched over these *plus* the slot's footprint groups on x, since
/// the snapshot shares its grid.
///
/// A constant, not per-dispatch data: the reservoir is [`BRUSH_RES`]² whatever the
/// segment does, and the two slot kinds that do not run an exchange never read it.
const RESERVOIR_GROUPS: (u32, u32) = (BRUSH_RES.div_ceil(8), BRUSH_RES.div_ceil(8));

/// GPU objects for the brush-dynamics stamp loop (§6.2), built once.
/// All handles are `Arc`-backed, so the kit is cheap to clone with its renderer.
///
/// Immutable throughout, which the type now says rather than merely intends: it
/// used to carry the round tip's coverage cache, the one mutable thing in a struct
/// documented as built-once. That lives with its sibling on the renderer
/// ([`StrokeRenderer::round_tip`](super::StrokeRenderer)), where the rest of the
/// lazily-baked brush textures already were.
#[derive(Clone)]
pub(super) struct DynamicsKit {
    // Region composite: base tiles → one 1:1 canvas region (colour + wide aux).
    pub(super) composite_pipeline: wgpu::RenderPipeline,
    pub(super) composite_view_bgl: wgpu::BindGroupLayout,
    pub(super) composite_tile_bgl: wgpu::BindGroupLayout,
    pub(super) composite_sampler: wgpu::Sampler,
    // The stamp-loop dispatches (one compute shader, several entry points).
    /// The footprint copy that gives the `deposit`/`settle` something to read while
    /// they storage-write the region.
    ///
    /// A painting segment does not dispatch it: its snapshot rides in the tail of its
    /// own `exchange` grid, since it depends on nothing that pass writes
    /// (`dynamics.wesl::exchange`). The two slot kinds with no exchange to ride in —
    /// [`SlotKind::Bleed`] and [`SlotKind::Settle`] — dispatch it standalone. (The
    /// settle could not have shared a grid in any case: it *reads* the snapshot,
    /// rather than merely sharing a consumer with it.)
    pub(super) snapshot_pipeline: wgpu::ComputePipeline,
    pub(super) snapshot_bgl: wgpu::BindGroupLayout,
    /// The tool's own side of one segment's transfer — the complement of every share
    /// the `deposit` after it hands the canvas (`dynamics.wesl::exchange`).
    pub(super) exchange_pipeline: wgpu::ComputePipeline,
    pub(super) exchange_bgl: wgpu::BindGroupLayout,
    /// Paint migrating *within* the tip, ahead of both halves of the transfer
    /// (`dynamics.wesl::wick_axis`). Shares `exchange`'s bind group: it reads and writes
    /// the same reservoir ping-pong and needs a strict subset of the same bindings.
    ///
    /// Two pipelines because the stencil is **separable** and runs as one 1-D pass per
    /// axis: `.0` along the tip's travel, `.1` across it. Splitting them is what lets
    /// the kernel be four times as wide for the same work per unit travel — and it is
    /// the shape a stroke-space march needs, where the along-travel pass lives inside a
    /// lateral row's own workgroup and only the across-row pass is a barrier.
    pub(super) wick_pipelines: (wgpu::ComputePipeline, wgpu::ComputePipeline),
    /// Integrates the reservoir along the segment's travel axis so the deposit can
    /// read the whole pass instead of one mid-pass sample (`dynamics.wesl::bake`).
    pub(super) bake_pipeline: wgpu::ComputePipeline,
    pub(super) bake_bgl: wgpu::BindGroupLayout,
    pub(super) deposit_pipeline: wgpu::ComputePipeline,
    pub(super) deposit_bgl: wgpu::BindGroupLayout,
    /// The pen-up: settles the transfer the tip was still in the middle of when the
    /// stroke stopped (`dynamics.wesl::settle`). Reads the reservoir through its own
    /// `bake` dispatch — the zero-travel slot bakes the *remaining pass's* delivery
    /// integral, not a per-segment window — never the cell that sits overhead.
    pub(super) settle_pipeline: wgpu::ComputePipeline,
    pub(super) settle_bgl: wgpu::BindGroupLayout,
    /// The deposit's prefix-τ volume binding (group 1) — the same texture the
    /// swept fast path samples, so the exchange footprint *is* the definite
    /// integral of the brush along the travel (compute-visible variant).
    pub(super) prefix_bgl: wgpu::BindGroupLayout,
    /// Bilinear clamp sampler for the region / reservoir / coverage lookups.
    pub(super) exchange_sampler: wgpu::Sampler,
    // Region → CoW tile write-back.
    pub(super) slice_pipeline: wgpu::RenderPipeline,
    pub(super) slice_bgl: wgpu::BindGroupLayout,
}

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
    pub(super) fn render_dynamic(
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

        // The union over the pieces below, which each enumerate their own subset.
        let dirty: Vec<TileCoord> = affected_tiles(&segments).into_iter().collect();
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
    /// GPU objects scoped to the whole run — the reservoir and the bake pair, which
    /// carry the tool from one piece to the next.
    scoped: ScopedResources,
    /// GPU objects scoped to the piece being recorded: the region, its selection
    /// mask, the snapshot scratch, the stamp buffer. Destroyed as soon as the piece
    /// is submitted (see [`Self::flush`]), so a stroke of any length costs one
    /// region's worth of transient memory rather than one per piece.
    piece: ScopedResources,
    /// Whether [`Self::draw`] has recorded a piece that [`Self::flush`] still owes a
    /// submit. Said outright rather than inferred from `piece` being non-empty, which
    /// is a coincidence of the two rather than the question being asked.
    piece_open: bool,
    /// Everything both render paths read off the record and the scene, resolved once
    /// (see [`StrokeConstants`](super::StrokeConstants)).
    consts: super::StrokeConstants,
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
        tol: crate::path::FlattenTolerance,
        tool: Option<&ToolState>,
    ) -> Self {
        let device = &r.ctx.device;
        let mut scoped = ScopedResources::default();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stark dynamics stroke"),
        });
        let consts = r.stroke_constants(rec, scene.surface);

        // The brush's swept-footprint prefix-τ (shared with the fast path) and its
        // plain coverage mask (the reservoir texels' own footprint weights).
        let prefix_view = r.prefix_view(scene.assets, &rec.brush);
        let prefix_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark dynamics prefix bg"),
            layout: &r.dynamics.prefix_bgl,
            entries: &[desc::tex(0, &prefix_view)],
        });
        let cov = r.coverage_view(scene.assets, &rec.brush);
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
            tol,
            scene,
            encoder,
            scoped,
            piece: ScopedResources::default(),
            piece_open: false,
            consts,
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
    /// `settle` is set only for the piece that ends the stroke: see
    /// [`StrokeRenderer::render_dynamic`] and `dynamics.wesl::settle`.
    fn draw(&mut self, base: &TileMap, segments: &[Segment], settle: bool) -> TileMap {
        self.flush();
        let coords = affected_tiles(segments);
        // A piece holds at least one segment, and a segment covers at least one tile,
        // so the empty case cannot arise here — but it costs nothing to leave the
        // canvas alone if it ever did.
        let Some((halo, lo, region_origin, w, h)) = region_rect(&coords) else {
            return base.clone();
        };
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
        let stamp_buf = self.upload_plan(&plan);
        let bind = self.bind_piece(&region, &under, &stamp_buf);

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
        let color = region_tex("stark dynamics region color");
        let aux = region_tex("stark dynamics region aux");

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
                tile_bgs.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("stark dynamics region tile bg"),
                    layout: &kit.composite_tile_bgl,
                    entries: &[
                        desc::tex(0, tile.color_view()),
                        desc::tex(1, tile.aux_view()),
                    ],
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
            let mut pass = self.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stark dynamics region composite"),
                color_attachments: &[
                    Some(desc::attach(&color, desc::CLEAR)),
                    Some(desc::attach(&aux, desc::CLEAR)),
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
            color,
            aux,
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
    /// +3 for the sampling margin `dynamics_plan` adds each side, +2 because a
    /// per-segment rect then rounds outward by a texel each side.
    fn snapshot_scratch(&mut self, segments: &[Segment], fires: &[(usize, Segment)]) -> Snapshot {
        let device = &self.r.ctx.device;
        let size = snapshot_size(segments, fires);
        let mut under_tex = |label: &'static str| {
            scoped_view(
                device,
                &mut self.piece,
                (size, size),
                wgpu::TextureFormat::Rgba16Float,
                LOOP_USAGE,
                label,
            )
        };
        Snapshot {
            size,
            color: under_tex("stark dynamics under color"),
            aux: under_tex("stark dynamics under aux"),
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
    ) -> PieceBindings {
        let r = self.r;
        let kit = &r.dynamics;
        let device = &r.ctx.device;
        let params = || wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: stamp_buf,
                offset: 0,
                size: wgpu::BufferSize::new(SLOT as u64),
            }),
        };
        let samp = || desc::samp(5, &kit.exchange_sampler);
        let snapshot = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark dynamics snapshot bg"),
            layout: &kit.snapshot_bgl,
            entries: &[
                params(),
                desc::tex(1, &region.color),
                desc::tex(2, &region.aux),
                desc::tex(3, &under.color),
                desc::tex(4, &under.aux),
            ],
        });
        // `exchange` comes in two flavours for the reservoir ping-pong: each reads one
        // half and writes the other.
        let exchange = std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("stark dynamics exchange bg"),
                layout: &kit.exchange_bgl,
                entries: &[
                    params(),
                    desc::tex(1, &region.color),
                    desc::tex(2, &region.aux),
                    desc::tex(3, &under.color),
                    desc::tex(4, &under.aux),
                    samp(),
                    desc::tex(6, &self.cov),
                    desc::tex(7, &self.brush_color[i]),
                    desc::tex(8, &self.brush_aux[i]),
                    desc::tex(9, &self.brush_color[1 - i]),
                    desc::tex(10, &self.brush_aux[1 - i]),
                    desc::tex(21, &region.sel_mask),
                ],
            })
        });
        // One bake bind group per reservoir phase; the deposit reads only the baked
        // result, so it no longer needs the ping-pong at all.
        let bake = std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("stark dynamics bake bg"),
                layout: &kit.bake_bgl,
                entries: &[
                    params(),
                    samp(),
                    desc::tex(7, &self.brush_color[i]),
                    desc::tex(8, &self.brush_aux[i]),
                    desc::tex(17, &self.bake_load),
                    desc::tex(18, &self.bake_latm),
                ],
            })
        });
        let deposit = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark dynamics deposit bg"),
            layout: &kit.deposit_bgl,
            entries: &[
                params(),
                samp(),
                desc::tex(19, &self.bake_load),
                desc::tex(20, &self.bake_latm),
                desc::tex(11, &under.color),
                desc::tex(12, &under.aux),
                desc::tex(13, &region.color),
                desc::tex(14, &region.aux),
                desc::tex(15, &self.noise),
                desc::samp(16, &r.noise_sampler),
                desc::tex(21, &region.sel_mask),
                desc::tex(22, &self.scene.surface.view),
            ],
        });
        // The pen-up, which reads the reservoir only through its own `bake` — so unlike
        // `exchange` it needs no bind group per ping-pong half; the bake's does that.
        let settle = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark dynamics settle bg"),
            layout: &kit.settle_bgl,
            entries: &[
                params(),
                desc::tex(19, &self.bake_load),
                desc::tex(20, &self.bake_latm),
                desc::tex(11, &under.color),
                desc::tex(12, &under.aux),
                desc::tex(13, &region.color),
                desc::tex(14, &region.aux),
                desc::tex(21, &region.sel_mask),
                // The ground: the pen-up delivery is a deposit like any other, and is
                // gated by the same tooth (§6.4).
                desc::tex(22, &self.scene.surface.view),
            ],
        });
        PieceBindings {
            snapshot,
            exchange,
            bake,
            deposit,
            settle,
        }
    }

    /// Record the loop: wick → bake → exchange (+ snapshot) → deposit per segment, in
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
                SlotKind::Segment { wick_steps } => {
                    // Let the tool's own paint migrate across the tip before anything
                    // reads it. Ahead of *both* halves of the transfer, so `bake` and
                    // `exchange` still see one another's entry state and their shares
                    // still add up (`dynamics.wesl::wick_axis`). Each pass reads `cur`
                    // and writes the other half, like every reservoir pass, so it
                    // cycles the ping-pong once.
                    //
                    // A firing is **two** passes, because the stencil is separable: one
                    // along the tip's travel, one across it.
                    for _ in 0..wick_steps {
                        for pipe in [&kit.wick_pipelines.0, &kit.wick_pipelines.1] {
                            cpass.set_pipeline(pipe);
                            cpass.set_bind_group(0, &bind.exchange[cur], &[off]);
                            cpass.dispatch_workgroups(RESERVOIR_GROUPS.0, RESERVOIR_GROUPS.1, 1);
                            cur = 1 - cur;
                        }
                    }
                    // Bake this segment's swept reservoir prefix next — it folds in the
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
                    cpass.set_pipeline(&kit.deposit_pipeline);
                    cpass.set_bind_group(0, &bind.deposit, &[off]);
                    cpass.set_bind_group(1, prefix_bg, &[]);
                    cpass.dispatch_workgroups(d.groups.0, d.groups.1, 1);
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
    /// `lo` is the region's *interior* origin — the top-left tile origin, an apron in
    /// from the region rectangle — so a tile's offset into the region is measured
    /// against it.
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

        // Every tile slices out of the *same* region, so the only thing that varies
        // across these draws is the offset uniform — one [`UNIFORM_STRIDE`] slot each
        // in one buffer, under one bind group, rather than a buffer and a bind group
        // per tile on every pointer move ([`UNIFORM_STRIDE`]).
        let mut data = vec![0u8; coords.len() * UNIFORM_STRIDE];
        for (i, coord) in coords.iter().enumerate() {
            let off = coord.origin() - lo;
            let slice = SliceUniform {
                offset: [off.x, off.y, 0.0, 0.0],
            };
            let at = i * UNIFORM_STRIDE;
            data[at..at + SLICE_SLOT as usize].copy_from_slice(bytemuck::bytes_of(&slice));
        }
        let ubuf = self
            .piece
            .buffer(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("stark dynamics slice params"),
                size: data.len() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        r.ctx.queue.write_buffer(&ubuf, 0, &data);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark dynamics slice bg"),
            layout: &kit.slice_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &ubuf,
                        offset: 0,
                        size: wgpu::BufferSize::new(SLICE_SLOT),
                    }),
                },
                desc::tex(1, &region.color),
                desc::tex(2, &region.aux),
            ],
        });

        let mut new_map = base.clone();
        for (i, coord) in coords.iter().enumerate() {
            let dst = r.acquire_tile(self.scene.pool, AllocSource::DynamicsWriteback);
            {
                let mut pass = self.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("stark dynamics slice"),
                    color_attachments: &[
                        Some(desc::attach(dst.color_view(), desc::CLEAR)),
                        Some(desc::attach(dst.aux_view(), desc::CLEAR)),
                    ],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&kit.slice_pipeline);
                pass.set_bind_group(0, &bg, &[(i * UNIFORM_STRIDE) as u32]);
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

/// One piece's canvas region: a 1:1 copy of the canvas under its segments, which the
/// loop evolves in place before [`DynamicsRun::write_back`] slices it into tiles.
struct Region {
    color: wgpu::TextureView,
    aux: wgpu::TextureView,
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
}

/// The bind groups one piece's dispatches switch between. Built once per piece
/// because every dispatch varies only the stamp uniform's dynamic offset; the
/// reservoir ping-pong is the one thing that needs a pair.
struct PieceBindings {
    snapshot: wgpu::BindGroup,
    exchange: [wgpu::BindGroup; 2],
    bake: [wgpu::BindGroup; 2],
    deposit: wgpu::BindGroup,
    settle: wgpu::BindGroup,
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

/// The [`Stamp`] lanes every slot in a plan fills the same way, resolved once — so
/// the three slot kinds below list only what actually differs between them, which
/// is the whole of what makes a bleed slot or a settle slot readable against a
/// painting segment.
struct SlotCommon<'a> {
    /// The stroke's own constants: `c` outright, and the colour-dynamics lookup that
    /// fills `f`, `g.xyz` and `h.xy`. Borrowed rather than copied out, so a slot and
    /// the swept path's `TileXform` are demonstrably reading one resolution of them.
    k: &'a super::StrokeConstants,
    /// `i.yzw`: the region texel → weave map, with the piece's origin already
    /// folded into the bias. Only `i.x` — how deep this slot's tip bites — varies.
    ///
    /// The one lane here that is not a stroke constant: the bias is where the *piece*
    /// sits, which `k` cannot know.
    weave: [f32; 3],
}

impl SlotCommon<'_> {
    /// The lanes every slot fills the same way — the stroke's colour and the weave
    /// map — over the neutral value of everything a slot kind may leave alone.
    ///
    /// A slot kind then names only what it actually differs by, which is the whole of
    /// what makes a bleed or settle slot readable against a painting segment.
    fn slot(&self) -> Slot {
        Slot {
            channels: self.k.channels,
            weave_scale: self.weave[0],
            weave_bias: Vec2::new(self.weave[1], self.weave[2]),
            ..Slot::default()
        }
    }

    /// [`Self::slot`] plus the colour-dynamics jitter, for a slot that lays the
    /// brush's own `add` paint: the shared field, this slot's arc length, and the
    /// bearing fraction it books the tool's half of the transfer against.
    ///
    /// `lambda_bleed` stays 0, which is what every such slot wants: the lateral flux
    /// runs only on the dedicated bleed slots, so between firings the canvas takes the
    /// no-bleed path bit-for-bit (§6.2).
    fn painting(&self, dist: f32, bearing: f32) -> Slot {
        let (namp, noff) = (self.k.namp, self.k.noff);
        Slot {
            noise_freq: self.k.nfreq,
            noise_amp: [namp[0], namp[1], namp[2]],
            noise_off: [noff[0], noff[1]],
            dist,
            bearing,
            ..self.slot()
        }
    }
}

/// One dispatch's uniform in **named fields**, packed into [`Stamp`]'s nine `vec4`
/// lanes by [`Slot::pack`] — the one place on this side of the boundary that knows
/// which lane is which.
///
/// The lanes are `vec4`s because that is what a uniform wants, and `dynamics.wesl`
/// long ago stopped reading them as such: every consumer there goes through a named
/// accessor (`radius()`, `travel_px()`, `lift_rate()`), so the shader's lane map lives
/// beside the declaration that decides it. This is the same move on the host, and it
/// is overdue for the same reason. Three slot kinds filled nine lanes at three sites
/// with wholly different meanings per component — 108 positional floats, and nothing
/// checking one of them. The generated `offset_of` assertions pin where a *lane*
/// starts, not what lives inside it, so `lambda(lift)` and `lambda(deposit)` written
/// the wrong way round was a silent wrong picture.
///
/// That it drifts is on the record: the note above the `Stamp` import remembers a
/// host-side copy that "still described `e.zw` as the midpoint `exchange` samples the
/// canvas at, some time after the shader had stopped reading the lane at all".
///
/// [`Default`] is the neutral slot — every rate off, and `bearing` at the 1 that
/// leaves an exchange alone — so each kind below lists only what it differs by.
#[derive(Clone, Copy)]
struct Slot {
    /// The sweep's start in region px, and the unit travel tangent it leaves along.
    start: Vec2,
    dir: Vec2,
    /// The tip's radius in region px, and its travel as a multiple of that radius —
    /// 0 on a settle, which is a break of contact rather than a stretch of it.
    radius: f32,
    travel_radii: f32,
    /// `λ = ln(1 − axis) ≤ 0`, clamped away from −∞. Zero is "no transfer".
    lambda_lift: f32,
    lambda_deposit: f32,
    /// The brush's own colour channels + per-unit opacity. **Undrained**.
    channels: [f32; 4],
    /// The dispatch rect's top-left in region texels, integral.
    rect_origin: Vec2,
    /// Shape orientation in turns ∈ [0, 1) — picks the prefix-τ slice (§6.6).
    orient: f32,
    /// The `drain` falloff per canvas px.
    drain: f32,
    /// The `add` source rate per unit exposure, **undrained** like the opacity.
    add: f32,
    /// Signed curvature of the sweep (1/region px); 0 is a straight one.
    curvature: f32,
    /// The bleed stencil's longest tap in texels — nonzero **only** on a bleed slot.
    bleed_reach: f32,
    /// The colour-dynamics lookup (§6.2): frequency per axis + 1/NOISE_TILE_PX,
    /// per-channel amplitude, and the per-stroke translation. All zero = no jitter.
    noise_freq: [f32; 4],
    noise_amp: [f32; 3],
    noise_off: [f32; 2],
    /// Arc length at the slot's start (canvas px) — the noise's third axis.
    dist: f32,
    /// The tooth's bearing fraction: the share of the ground a tip with this `tooth`,
    /// going this way, stands on (§6.4). What the *tool* books its half of the
    /// transfer against, having no ground of its own. 1 where there is nothing to bite.
    bearing: f32,
    /// The lateral canvas diffusion rate (≤ 0) — nonzero **only** on a bleed slot.
    lambda_bleed: f32,
    /// How little give this slot's tip has (0 = the ground gates nothing), over the
    /// region texel → weave map `uv = rt · weave_scale + weave_bias` (§6.4).
    tooth: f32,
    weave_scale: f32,
    weave_bias: Vec2,
}

impl Default for Slot {
    /// Zero everywhere except `bearing`, whose neutral value is **1** — the share of
    /// the ground a tip stands on where there is nothing to bite, and what leaves an
    /// exchange alone if one runs. A zeroed bearing would silently book the tool's
    /// half of every transfer against no ground at all, which is not "no tooth" but
    /// "infinite tooth", so it is the one field a derived `Default` would get wrong.
    fn default() -> Self {
        Self {
            start: Vec2::ZERO,
            dir: Vec2::ZERO,
            radius: 0.0,
            travel_radii: 0.0,
            lambda_lift: 0.0,
            lambda_deposit: 0.0,
            channels: [0.0; 4],
            rect_origin: Vec2::ZERO,
            orient: 0.0,
            drain: 0.0,
            add: 0.0,
            curvature: 0.0,
            bleed_reach: 0.0,
            noise_freq: [0.0; 4],
            noise_amp: [0.0; 3],
            noise_off: [0.0; 2],
            dist: 0.0,
            bearing: 1.0,
            lambda_bleed: 0.0,
            tooth: 0.0,
            weave_scale: 0.0,
            weave_bias: Vec2::ZERO,
        }
    }
}

impl Slot {
    /// The lane packing — `dynamics.wesl`'s `struct Stamp`, read the other way round.
    /// Every field above lands here exactly once.
    fn pack(&self) -> Stamp {
        Stamp {
            a: [self.start.x, self.start.y, self.dir.x, self.dir.y],
            b: [
                self.radius,
                self.travel_radii,
                self.lambda_lift,
                self.lambda_deposit,
            ],
            c: self.channels,
            d: [
                self.rect_origin.x,
                self.rect_origin.y,
                self.orient,
                self.drain,
            ],
            // `.w` is the last of the lane that carried the midpoint `exchange` used
            // to sample the canvas at; it walks the texel's own track now.
            e: [self.add, self.curvature, self.bleed_reach, 0.0],
            f: self.noise_freq,
            g: [
                self.noise_amp[0],
                self.noise_amp[1],
                self.noise_amp[2],
                self.dist,
            ],
            h: [
                self.noise_off[0],
                self.noise_off[1],
                self.bearing,
                self.lambda_bleed,
            ],
            i: [
                self.tooth,
                self.weave_scale,
                self.weave_bias.x,
                self.weave_bias.y,
            ],
        }
    }
}

/// What a plan is built *against*, as opposed to the segments it is built *from*:
/// where the piece's region sits, how large its snapshot scratch is, and the stroke
/// constants every slot is filled from.
///
/// Bundled because these five travel together through the plan and its rect
/// arithmetic, and because a slot's geometry is only meaningful relative to them.
struct PlanCtx<'a> {
    rec: &'a StrokeRecord,
    /// The budget `rec` was flattened at, handed down from [`dynamics_setup`] rather
    /// than recomputed — one place answers what a stroke's segments are. Only the
    /// pen-up frame reads it ([`settle_tangent`]), which re-flattens a footprint's
    /// worth of the record and must cut it exactly as the segments in hand were cut.
    tol: crate::path::FlattenTolerance,
    /// The region rectangle's top-left in canvas px — what every slot's coordinates
    /// are measured from, since the shader never learns where the piece sits.
    region_origin: Vec2,
    /// The snapshot scratch's square ([`Snapshot::size`]), which every dispatch rect
    /// is clamped to.
    dsize: u32,
    /// Everything both render paths read off the record and the scene
    /// ([`StrokeConstants`](super::StrokeConstants)) — the colour a slot's `c` is, the
    /// weave map its `i` carries, and the colour-dynamics lookup for `f`–`h`.
    consts: &'a super::StrokeConstants,
    surface: &'a crate::gpu::surface::Surface,
}

/// The margin, in canvas px, a dispatch rect is grown by each side so a fragment
/// sampling just outside its own texel still lands inside the rect.
const RECT_MARGIN: f32 = 1.5;

/// The snapshot scratch's square for a piece: large enough for any one slot's
/// [`dispatch_rect`], measured from the coverage boxes rather than bounded
/// analytically, since `coverage_bounds` is already the exact box and a curved sweep
/// has no closed-form "worst rotation" to fall back on.
///
/// `+3` for the [`RECT_MARGIN`] a rect adds each side, `+2` because a rect then floors
/// its origin and rounds its far edge outward, a texel each way. Split out from
/// [`DynamicsRun::snapshot_scratch`] so the fit against `dispatch_rect` — the argument
/// those two numbers *are* — can be checked without a GPU (`tests`).
fn snapshot_size(segments: &[Segment], fires: &[(usize, Segment)]) -> u32 {
    let dmax = segments
        .iter()
        .chain(fires.iter().map(|(_, f)| f))
        .fold(1.0f32, |m, s| {
            let (lo, hi) = coverage_bounds(s);
            m.max(hi.x - lo.x).max(hi.y - lo.y)
        });
    (dmax + 3.0).ceil() as u32 + 2
}

/// One slot's dispatch rectangle over a canvas-space coverage box: its integral origin
/// in region texels, and the workgroup counts covering it.
///
/// The slot's own box rather than the piece-wide worst case, so an axis-aligned sweep
/// dispatches ~4·r² threads where a square would spend ~10·r². Texels the rounding adds
/// beyond the box read zero exposure and fall out of `deposit` untouched.
///
/// Every rect fits `dsize` by construction, because [`snapshot_size`] was measured from
/// these same boxes with room for this margin and this rounding — asserted rather than
/// left implied, since the `min` below would otherwise clip a too-large rect into a
/// silently truncated footprint.
fn dispatch_rect(lo: Vec2, hi: Vec2, region_origin: Vec2, dsize: u32) -> Rect {
    let lo = lo - region_origin - Vec2::splat(RECT_MARGIN);
    let hi = hi - region_origin + Vec2::splat(RECT_MARGIN);
    let origin = Vec2::new(lo.x.floor(), lo.y.floor());
    let (w, h) = (
        ((hi.x - origin.x).ceil() as u32) + 1,
        ((hi.y - origin.y).ceil() as u32) + 1,
    );
    debug_assert!(
        w <= dsize && h <= dsize,
        "a {w}x{h} dispatch rect overruns the {dsize} snapshot scratch",
    );
    let (w, h) = (w.min(dsize), h.min(dsize));
    Rect {
        origin,
        groups: (w.div_ceil(8), h.div_ceil(8)),
    }
}

/// One slot's dispatch rectangle.
struct Rect {
    /// The rect's top-left in region texels, integral — the `d.xy` a slot carries.
    origin: Vec2,
    /// Workgroup counts covering it, at the shaders' 8×8.
    groups: (u32, u32),
}

impl PlanCtx<'_> {
    /// [`dispatch_rect`] against this piece's origin and snapshot square.
    fn rect(&self, lo: Vec2, hi: Vec2) -> Rect {
        dispatch_rect(lo, hi, self.region_origin, self.dsize)
    }
}

/// Build the swept-exchange dispatch plan (§6.2): one `snapshot` +
/// `deposit` pair per flattened segment (the canvas-side exchange, swept through
/// the prefix-τ integral), each followed by the tool's own `exchange`.
/// λ = ln(1 − axis) makes every rate exponential in
/// exposure, so the exchange composes exactly across overlapping segment quads —
/// the continuous path integral, independent of any spacing. Pure CPU float math
/// → replay-deterministic.
///
/// Every painting dispatch is a segment: the tool exchanges once per segment rather
/// than on a cadence of its own, so there is no interval state to carry between
/// ranges. The bleed cadence and the pen-up ride as their own [`SlotKind`]s.
fn dynamics_plan(
    ctx: &PlanCtx<'_>,
    segments: &[Segment],
    fires: &[(usize, Segment)],
    settle: bool,
) -> Vec<LoopDispatch> {
    let &PlanCtx {
        rec,
        region_origin,
        consts,
        surface,
        ..
    } = ctx;
    let b = &rec.brush;
    // The canvas → weave map, folded so the shader can go straight from its *region*
    // texel to the ground under it: `uv = rt · grain_uv + grain_bias` (§6.4). Only the
    // bias belongs to the piece — the shader never learns where the piece sits, only
    // where the weave does; the scale is a stroke constant and comes off `consts`,
    // which is what keeps it the same number the swept path writes.
    let grain_bias = region_origin * consts.grain_uv;
    // What share of the ground a tip with this tooth, going this way, stands on — per
    // segment because the tooth is modulated per segment (§6.2) and because the
    // direction is the segment's own. The canvas side of the exchange asks the ground
    // ahead of each texel; the tool has none of its own and books against this mean,
    // which is what makes a toothed smear conserve (`Surface::bearing`).
    //
    // At the segment's **midpoint** tangent, the same second-order choice `mid` is
    // sampled at below: a curved segment's canvas side reads a tangent that turns
    // across the sweep, and the midpoint is the representative of that whose error is
    // second order where either endpoint's would be first.
    let bearing = |tooth: f32, dir: Vec2| surface.bearing(tooth, dir.to_array());
    // λ = ln(1 − axis), clamped away from −∞ (axis = 1 ⇒ e^{−20} ≈ scraped clean),
    // per [`TAU_PER_PASS`] — so an axis reads as a fraction *per pass of the tip*,
    // which is what a 0..1 knob should mean, rather than per unit optical depth.
    //
    // Taken **per segment**, off the rates the segment generator resolved from the
    // pen (§6.2), rather than once for the stroke. Nothing else about the loop
    // changes: every dispatch already carried its own λs in its slot, because a
    // segment is where the exchange happens.
    let lambda = |axis: f32| (1.0 - axis.clamp(0.0, 1.0)).max(1e-9).ln().max(-20.0) / TAU_PER_PASS;
    let common = SlotCommon {
        k: consts,
        weave: [consts.grain_uv, grain_bias.x, grain_bias.y],
    };

    let mut plan = Vec::new();
    // Drained in step with the walk below, which is only correct because `bleed_fires`
    // emits them in segment order. Cheap to state, and the alternative — a firing
    // silently landing in the wrong piece of the plan — is not something a pixel would
    // show.
    debug_assert!(
        fires.is_sorted_by_key(|(after, _)| *after),
        "bleed firings must arrive in segment order",
    );
    let mut pending = fires.iter().peekable();
    for (si, s) in segments.iter().enumerate() {
        // The segment's swept exchange: the frame is (start, travel tangent at the
        // start, curvature), over the segment's own coverage box.
        let p = s.start - region_origin;
        let (clo, chi) = coverage_bounds(s);
        let rect = ctx.rect(clo, chi);
        // The tangent at the segment's **midpoint**, along the arc rather than the
        // chord: what the bearing below is read along, since a curved segment's canvas
        // side sees a heading that turns across the sweep and the midpoint is the
        // representative of that whose error is second order where either endpoint's
        // would be first.
        let (_, mid_dir) = crate::path::arc_at(s.start, s.dir, s.curvature, s.length * 0.5);
        // The `wick` passes falling during this segment. Counted off the segment's own
        // **absolute** arc length rather than by accumulating a debt across the loop,
        // and that is the whole reason it is written this way: a live tail re-renders
        // from a span boundary, so anything carried between segments would have to be
        // threaded through [`ToolState`] and would still have to agree with the commit
        // that eventually replaces it. Keyed on the arc, the count is a pure function of
        // the segment, so every render of a stretch of stroke wicks it identically
        // (§6.2, live == committed).
        //
        // A segment's own radius sets its quantum, so a stroke whose radius varies can
        // land a boundary twice or skip one where the pitch changes. That is a ±1 on a
        // *smoothing* cadence, and it stays deterministic, which is the property that
        // matters.
        let quantum = WICK_TRAVEL_QUANTUM * s.radius;
        let wick_steps = ((s.dist + s.length) / quantum).floor() - (s.dist / quantum).floor();
        plan.push(LoopDispatch {
            groups: rect.groups,
            kind: SlotKind::Segment {
                wick_steps: wick_steps.max(0.0) as u32,
            },
            slot: Slot {
                start: p,
                dir: s.dir,
                radius: s.radius,
                travel_radii: s.length / s.radius,
                lambda_lift: lambda(s.lift),
                lambda_deposit: lambda(s.deposit),
                rect_origin: rect.origin,
                orient: s.orient,
                drain: b.drain,
                // The `add` source rate is passed through **unscaled**, exactly as
                // `stamp.wesl` takes it. It used to carry a gain of 2 ("tuned so
                // `add = 1` lays roughly a full-thickness deposit per pass"), which made
                // the same slider mean two different amounts of paint depending on
                // whether some *other* axis happened to be non-zero — nudging `deposit`
                // off zero doubled the flow. The tuning it claimed is already met without
                // it: a pass of the tip is `TAU_PER_PASS ≈ 6.9` of exposure, so
                // `add = 1` lays 6.9 of height, which the slab law reads as 0.999
                // coverage.
                //
                // Off the segment, since the pen can drive it (§6.2) — the same number
                // the swept path now reads off its instance.
                add: s.add,
                curvature: s.curvature,
                tooth: s.tooth,
                // No `bleed_reach` and no `lambda_bleed`: the lateral flux runs only on
                // the dedicated firings, so a painting segment takes the no-bleed path
                // bit-for-bit (§6.2). Both are `Slot::default`'s zero.
                ..common.painting(s.dist, bearing(s.tooth, mid_dir))
            }
            .pack(),
        });

        // The bleed slots that fire at this segment's end (§6.2, `bleed_fires`):
        // a quad whose sweep is the firing's travel window, with every vertical rate
        // and the source zeroed — the dispatch is the identity everywhere except the
        // lateral flux. The noise lanes are zeroed too, so the deposit skips its
        // colour-jitter taps.
        while let Some((_, fire)) = pending.next_if(|(after, _)| *after == si) {
            let p = fire.start - region_origin;
            let (clo, chi) = coverage_bounds(fire);
            let rect = ctx.rect(clo, chi);
            // The stencil this firing diffuses with: how far it reaches, and how hard
            // it relaxes to get there. Both come out of the diffusivity the axis asks
            // for — see [`bleed_stencil`], which is where the axis's whole meaning is.
            let (reach, lambda_bleed) = bleed_stencil(fire.bleed, fire.radius, fire.length);
            plan.push(LoopDispatch {
                groups: rect.groups,
                kind: SlotKind::Bleed,
                // Everything a painting segment carries and this does not is
                // `Slot::default`'s zero, which is what the slot *means*: λ_lift = 0 so
                // the canvas keeps everything, λ_deposit = 0 so the (uninvolved) tool
                // lays nothing, no drain because nothing is laid, no `add` because this
                // is not a stretch of painting, no tooth because there is no `add` for
                // the ground to gate, and no colour jitter — which is zeroed rather
                // than shared, so the deposit skips its noise taps entirely.
                slot: Slot {
                    start: p,
                    dir: fire.dir,
                    radius: fire.radius,
                    travel_radii: fire.length / fire.radius,
                    rect_origin: rect.origin,
                    orient: fire.orient,
                    // The window's own curvature, so the relaxed band follows the paint
                    // rather than cutting the corner off it (`bleed_fires`).
                    curvature: fire.curvature,
                    // The stencil's longest tap — the only slot that carries one.
                    bleed_reach: reach,
                    dist: fire.dist,
                    // The rate that lands this window's exposure on the blend its reach
                    // needs — not `lambda(axis)`, which is the vertical rates' mapping
                    // and would make the axis a rate rather than a diffusivity. A firing
                    // whose modulated axis has fallen to zero still dispatches: λ = 0
                    // makes it the identity, and keeping the plan a pure function of the
                    // segmentation is worth more than the dispatch it would save.
                    lambda_bleed,
                    ..common.slot()
                }
                .pack(),
            });
        }
    }

    // The pen-up (`dynamics.wesl::settle`), as one more slot on the same uniform: the
    // tip standing at the stroke's last point with **zero travel**, which is what makes
    // the shared `segment_frame`/`outside_sweep` reduce to the tip's own footprint and
    // `snapshot` copy exactly the texels the settle will write. Everything the settle
    // reads is already here — the frame, the radius, the two λs and the orientation —
    // so it costs a slot rather than a second uniform.
    if let Some(s) = settle.then(|| segments.last()).flatten() {
        let (end, _) = crate::path::arc_at(s.start, s.dir, s.curvature, s.length);
        // The frame comes off the *record* — see [`settle_tangent`]. `segments` is one
        // piece of one range, and a lookback that walked it would stop wherever that
        // cut fell, so a live tail and its commit would settle at different angles.
        let tan = settle_tangent(rec, ctx.tol, segments);
        let p = end - region_origin;
        // The tip's own square rather than a swept box — a pen-up is a standing tip.
        // It cannot overrun the snapshot scratch: that was sized from coverage boxes,
        // and a segment's box is this square grown by its travel.
        let rect = ctx.rect(end - Vec2::splat(s.radius), end + Vec2::splat(s.radius));
        plan.push(LoopDispatch {
            groups: rect.groups,
            kind: SlotKind::Settle,
            slot: Slot {
                start: p,
                dir: tan,
                // No travel: a pen-up is a break of contact, not a stretch of it. The
                // rates are the *last* segment's, which is where the pen was when it
                // left the page — the same segment this slot takes its radius and
                // orientation from. (`travel_radii` stays at its default 0.)
                radius: s.radius,
                lambda_lift: lambda(s.lift),
                lambda_deposit: lambda(s.deposit),
                rect_origin: rect.origin,
                orient: s.orient,
                drain: b.drain,
                // The last segment's tooth: the settle delivers what the pass still
                // owed, and it owes it through the same ground the pass was laying
                // through. What the valleys do not take stays on the tool, which is
                // discarded — a knife lifted off a canvas keeps what it did not
                // reach (§6.4).
                tooth: s.tooth,
                // No `add`: the source is a rate per unit of travel, and there is none.
                // No curvature, for the same reason — the frame is a standing tip. No
                // bleed reach: a settle is not a firing. And no λ_bleed either — that
                // axis carries no reservoir, every firing having applied its window as
                // the tip passed, so a break of contact strands nothing for a settle to
                // finish, unlike the vertical transfer whose in-flight half lives on the
                // tool. All four are `Slot::default`'s zero.
                //
                // The bearing is the neutral 1: the tool is not written back at pen-up,
                // so nothing reads it — the settle's own gate is per texel, from the
                // weave. The colour channels are filled consistently with a segment slot
                // rather than left as junk, though the settle lays the tool's *carried*
                // paint and so reads none of them.
                ..common.painting(s.dist + s.length, 1.0)
            }
            .pack(),
        });
    }
    plan
}

/// The bleed cadence (§6.2): one dedicated **bleed slot** per crossing of
/// [`BLEED_TRAVEL_QUANTUM`] of absolute arc, as `(after, window)` pairs — the index
/// of the piece segment the firing follows, and a synthetic segment whose sweep is
/// the firing's travel window: **exactly one quantum** of path, bending the way the
/// crossing segment bends. A segment that crosses the cadence twice fires twice, and
/// the two windows tile back from its end rather than merging into one.
///
/// **One quantum per firing is what makes the axis a diffusivity** rather than a
/// number that means less the faster the hand moves. A window asks the stencil for
/// `σ² ∝ its own travel`, and what one firing can carry is `2·Σ(share·d²)` — a
/// property of the stencil, flat in the travel. So a merged N-quantum window asks for
/// N times what a firing can give and is clamped back to roughly `1/N` of it
/// ([`bleed_stencil`]). That is not the exotic case: a segment at the travel cap
/// crosses a half-radius cadence twice, so an ordinary fast stroke was already
/// diffusing a tenth short before this fired per crossing. Variance adds linearly in
/// travel across firings, so N of them deliver N quanta's worth exactly — more steps,
/// not bigger ones, as in any explicit diffusion solver.
///
/// Counted off the **absolute** arc exactly as the wick's crossings are, and for
/// the same reason: the firings, and the windows they sweep, are then a pure
/// function of the record, independent of how the path was cut (§6.2, live ==
/// committed). Why the lateral flux cannot simply ride the painting segments is a
/// numeric story told at the shader (`dynamics.wesl`, the bleed-slot note): on real
/// slow input the fitter emits sub-pixel segments, whose per-texel exposure is
/// prefix-cancellation noise and whose per-segment fluxes sit under the f16 ULP of
/// the heights they edit — measured as a 20-level directional ghost on a 177-knot
/// repro. A half-radius window has neither problem.
///
/// Each window is an **arc**, not the chord across one. At this cadence the two are
/// a fraction of a texel apart, so this is not a correction — it is that a window
/// *is* a stretch of the path, and a representation that says so cannot be wrong at
/// whatever cadence some later tuning picks. Its start is walked **back along the
/// crossing segment's own arc** rather than looked up among the segments in hand, so a
/// window is never truncated by where the range being drawn happens to begin — see the
/// note at the walk itself for what that truncation cost.
fn bleed_fires(bleed: f32, segments: &[Segment]) -> Vec<(usize, Segment)> {
    let mut fires = Vec::new();
    // The brush's own axis, so *which* windows fire stays a function of the geometry
    // and the brush alone; how hard each one relaxes is the pen's business, and comes
    // off the crossing segment below. A brush at zero bleed can be modulated nowhere
    // above zero, so this early-out is exact (`document::Modulation`).
    if bleed <= 0.0 {
        return fires;
    }
    for (i, s) in segments.iter().enumerate() {
        let bq = BLEED_TRAVEL_QUANTUM * s.radius;
        let crossings = ((s.dist + s.length) / bq).floor() - (s.dist / bq).floor();
        if crossings < 1.0 {
            continue;
        }
        if bq <= 1e-3 {
            continue; // a tip with no width: nothing swept, nothing to relax
        }
        // Capped so a plan stays bounded. `crossings` is the segment's travel over its
        // *own* radius' quantum, and those two are priced apart: the flattener buys
        // segment length off the brush's nominal radius while the cadence is the
        // modulated one, so a pen thinning the tip drives the count up without
        // shortening anything. Eight covers a tip down to a quarter of the brush;
        // under that the axis under-delivers, on a tip carrying almost no paint to
        // spread. Without a cap this is a memory blow-up on a degenerate stroke, which
        // is a worse failure than a gentle one.
        let crossings = (crossings as usize).min(MAX_BLEED_FIRES_PER_SEGMENT);
        let (end, end_dir) = crate::path::arc_at(s.start, s.dir, s.curvature, s.length);
        // Walked **back along the crossing segment's own arc**, rather than looked up
        // in the segments this piece happens to hold. Reversing an arc is negating
        // both its direction and its curvature, so this is the same circle traced the
        // other way and is exact for any path the segment itself describes.
        //
        // It is history-free, and that is the point. Looking the position up meant
        // clamping to the first segment in hand, so a window reaching further back
        // than the range being drawn came out short — and a live tail always starts at
        // a span boundary while the commit renders the whole stroke from zero, so the
        // two relaxed different amounts of paint at exactly that seam. That is a
        // `preview == committed` break (§1.3), in the one place it cannot be
        // repainted, and it was visible: a bleeding stroke lightened when the pointer
        // came up.
        //
        // What it costs is extrapolating one segment's curvature over the window,
        // where the old form used the true path — the same bend for the whole span
        // rather than each segment's own. Bounded by
        // [`MAX_TIP_TURN`](super::budget::MAX_TIP_TURN), which caps how far the tip's
        // curvature may move at all, and the window is the arc that extrapolation
        // describes rather than a chord across it, so nothing else is given up on top.
        //
        // Emitted oldest first: the firings tile back from the segment's end, but they
        // edit the canvas in sequence and paint laid earlier should relax first.
        for n in (0..crossings).rev() {
            let back = (n + 1) as f32 * bq;
            let (start, back_dir) = crate::path::arc_at(end, end_dir * -1.0, -s.curvature, back);
            fires.push((
                i,
                Segment {
                    start,
                    // The reversed walk arrives pointing back the way it came, so the
                    // window's own heading is its negation — the tangent the path had
                    // at `start`, which is where the arc below is measured from.
                    dir: back_dir * -1.0,
                    // **The window bends with the path it stands for.** Its two
                    // endpoints were always on the arc; carrying the curvature is what
                    // puts the sweep between them there too. At this cadence a chord
                    // would sit `span²·κ/8` off the paint, which the tip covers many
                    // times over — so this is not a correction, it is that a window
                    // *is* a stretch of the path and nothing is gained by representing
                    // it as something else. Nothing downstream needs telling:
                    // `coverage_bounds` already grows a box by the sagitta, and
                    // `deposit` sweeps an arc for every painting segment by unrolling
                    // the annulus (`stamp_common::sweep_at`) — a bleed slot just takes
                    // the same path. The unroll's own error is `radius·|curvature|/2`,
                    // which the window inherits from the crossing segment and the
                    // flattener has capped
                    // ([`MAX_TIP_TURN`](super::budget::MAX_TIP_TURN)).
                    curvature: s.curvature,
                    radius: s.radius,
                    // One quantum of arc length, which is what `sweep_at` measures
                    // travel in — and what `bleed_stencil` is calibrated against.
                    length: bq,
                    orient: s.orient,
                    dist: s.dist + s.length - back,
                    // The window inherits the crossing segment's rates: it is that
                    // segment's own firing, and `bleed` is the only one it will use —
                    // every other axis is zeroed in the slot it becomes. Reading them
                    // from one point of the window is the same approximation the
                    // wick's cadence already makes about the radius it fires at.
                    add: s.add,
                    lift: s.lift,
                    deposit: s.deposit,
                    bleed: s.bleed,
                    tooth: s.tooth,
                },
            ));
        }
    }
    fires
}

/// The travel direction the pen-up settle measures `owed` and `received` along: the
/// chord over the **last footprint's worth of path**, rather than the last segment's
/// own tangent.
///
/// The last segment's tangent cannot be trusted, and the reason is a property of real
/// input rather than a rare edge case. A hand pauses before it lifts, so a pen-up
/// arrives as a cluster of samples at almost one point; the fitter turns that into
/// spans of no length, and the flattener into edges whose chord is a rounding error
/// and whose direction is therefore arbitrary — measured on a straight drag down, the
/// final edges came out at 0°, −90°, 90° and 180° against a stroke running at 90°.
///
/// Nothing else in the loop notices: a segment of no length deposits nothing, so its
/// direction never reaches a pixel. The settle is the exception, because it takes a
/// whole tip's worth of exchange from that one frame — and its `min(owed, received)`
/// lens is elongated *along* it, so a wrong direction lands a tip-shaped disc across
/// the stroke instead of along it, at a different angle every time the hand pauses
/// differently. That is exactly what it looked like: a fade-out cap whose orientation
/// wandered from stroke to stroke, and worse the higher `lift` and `deposit` were.
///
/// One radius is the natural window because it is the extent of the thing being
/// settled — the tip's own footprint — so this is the direction the tip was travelling
/// over precisely the stretch of canvas the settle acts on, and no new constant.
///
/// **Measured on the record, not on the segments in hand**, and that is the whole of
/// why it takes a `rec`. The slice a plan is built from is one *piece* of one *range*
/// — `chunk_segments`'s cut of what `render_range` was asked for — so a lookback that
/// walked it stopped at whichever boundary came first. A live tail always starts at a
/// span boundary while the commit renders the whole stroke from zero, so a tail
/// carrying less than a radius of travel measured its frame over a shorter window than
/// the commit measured the same frame over, and on a curving stroke the two came out
/// pointing different ways: the fade-out cap turned as the pointer came up, which is a
/// `preview == committed` break (§1.3) in the one place it cannot be repainted.
///
/// This is the same defect [`bleed_fires`] was fixed for, and the cure is the same in
/// spirit — ask the record rather than the range — but not in mechanism. Walking back
/// along the last segment's own arc, as a firing's window does, is exactly what this
/// function exists to avoid: the last segments *are* the degenerate ones. So it walks
/// the curve's own polyline instead, flattening only the trailing spans a radius
/// reaches back over (`span_end` prices a span boundary without subdividing anything,
/// so finding them costs no polyline).
///
/// `segments` is still read, for the two things only it knows: the radius of the tip
/// being settled, and a click's frame — a lone control point is not a curve, and
/// `generate_segments_in` gives its dab a real direction where the path has none.
fn settle_tangent(
    rec: &StrokeRecord,
    tol: crate::path::FlattenTolerance,
    segments: &[Segment],
) -> Vec2 {
    let radius = segments.last().map_or(1.0, |s| s.radius);
    // A click's frame: the dab's own direction, which is deliberate rather than
    // fitted. `generate_segments_in` sweeps it symmetrically about the point pressed,
    // so which direction it is cannot matter — but it has to be *a* direction.
    let fallback = || segments.last().map_or(Vec2::new(1.0, 0.0), |s| s.dir);
    let last = crate::path::span_count(rec.path.len());
    if last == 0 {
        return fallback();
    }
    let tip = crate::path::span_end(&rec.path, last - 1);
    // The first span boundary a radius or more back from the tip, measured on chords
    // — which under-estimate arc length, so the span this admits genuinely holds a
    // footprint's worth of path behind it. Walking boundaries rather than the polyline
    // is what keeps the flatten below proportional to the *radius* instead of to the
    // length of the stroke.
    let mut from = 0;
    for k in (0..last).rev() {
        let cut = if k == 0 {
            rec.path[0].pos
        } else {
            crate::path::span_end(&rec.path, k - 1)
        };
        if (tip - cut).length() >= radius {
            from = k;
            break;
        }
    }
    let pts = crate::path::flatten_spans(&rec.path, from..last, 0.0, tol);
    let Some(tail) = pts.last() else {
        return fallback();
    };
    // Back one radius of **travel**, not of displacement — the window is a footprint's
    // worth of path, and a stroke that curls back on itself still spent that path. The
    // polyline carries its own arc-length accumulator, so this is a comparison rather
    // than a second summation.
    let back = pts
        .iter()
        .rev()
        .find(|p| tail.dist - p.dist >= radius)
        .unwrap_or(&pts[0]);
    let v = tail.pos - back.pos;
    let len = v.length();
    if len > 1e-4 {
        v / len
    } else {
        // A stroke with no travel at all — every knot on one spot. There is no
        // direction in the path to find, so the dab's stands.
        fallback()
    }
}

/// Which path a stroke takes, as [`dynamics_setup`] decides it.
///
/// The two swept answers are kept apart because they are not the same event: one is
/// the fast path doing its job, the other is the renderer failing to draw the brush
/// it was given. Only the caller knows how loudly to say so, so the distinction is
/// carried out rather than resolved here.
pub(super) enum StrokePath {
    /// Run the sequential stamp loop.
    Loop,
    /// The brush manipulates no paint already on the canvas, so the swept deposit
    /// *is* the whole stroke — one pass, no region, nothing given up.
    Swept,
    /// The brush manipulates paint, but its tip alone wants more than one region, and
    /// the region is the one thing pieces cannot subdivide. The swept deposit draws
    /// what it can, which is the brush's own `add` paint and none of the manipulation.
    TipTooLarge,
}

/// Which path a stroke takes and the budget it flattens at — both decided together,
/// because both are answers about the brush alone and every path needs the second.
pub(super) struct StrokePlan {
    pub(super) path: StrokePath,
    pub(super) tol: crate::path::FlattenTolerance,
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
pub(super) fn dynamics_setup(rec: &StrokeRecord) -> StrokePlan {
    let d = rec.brush.dynamics;
    // The brush's **own** rates, not the modulated ones — and that is sound rather
    // than an oversight the pen could catch out. A modulation is a factor in [0, 1]
    // (`document::Modulation`), so an axis the brush leaves at zero is zero at every
    // point of every stroke it could ever draw, and one it leaves positive is
    // positive *somewhere*. There is no segment this test could be asked about that
    // would answer differently — which is exactly the property the function's
    // contract above needs, and the reason a modulation was built as a multiplier.
    // The same flattened segments whichever path runs, at the same budget: a long
    // stroke costs more pieces, not coarser geometry — and the swept fallback below
    // draws the very segments the loop would have.
    let tol = flatten_tolerance(&rec.brush);
    let path = if d.lift <= 0.0 && d.deposit <= 0.0 && d.charge <= 0.0 && d.bleed <= 0.0 {
        StrokePath::Swept
    } else if segment_fits_region(&rec.brush, tol) {
        StrokePath::Loop
    } else {
        StrokePath::TipTooLarge
    };
    StrokePlan { path, tol }
}

/// Build the brush-dynamics stamp-loop kit (§6.2): the region
/// composite, the loop's seven compute pipelines, and the region→tile slice.
pub(super) fn build_dynamics_kit(
    device: &wgpu::Device,
    color_space: &dyn ColorSpace,
) -> DynamicsKit {
    // The loop's storage-texture declarations are `rgba16float`; both color
    // spaces use that tile colour format (§6.7), so the region can hold either.
    debug_assert_eq!(color_space.color_format(), wgpu::TextureFormat::Rgba16Float);
    let frag = wgpu::ShaderStages::FRAGMENT;

    // ---- Region composite: the `composite` shader over region-sized targets
    // (colour + the wide aux, so nothing is narrowed until the write-back).
    let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("stark dynamics composite"),
        source: wgpu::ShaderSource::Wgsl(stark_shaders::composite().into()),
    });
    let composite_view_bgl = desc::bind_group_layout(
        device,
        "stark dynamics composite view bgl",
        &[
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
    );
    let composite_tile_bgl = desc::bind_group_layout(
        device,
        "stark dynamics composite tile bgl",
        &[desc::sample_tex(0, frag), desc::sample_tex(1, frag)],
    );
    let composite_layout = desc::pipeline_layout(
        device,
        "stark dynamics composite layout",
        &[Some(&composite_view_bgl), Some(&composite_tile_bgl)],
    );
    let composite_pipeline = desc::render_pipeline(
        device,
        desc::RenderPipe {
            label: "stark dynamics composite pipeline",
            layout: &composite_layout,
            module: &composite_shader,
            vs: "vs_main",
            // `fs_raw`, NOT the screen path's `fs_main`: the loop's region must hold
            // the tile representation itself (opacity in alpha), not the
            // coverage-weighted channels pass A shows — the exchange reads this
            // region and the slice writes it back to persistent tiles.
            fs: "fs_raw",
            primitive: desc::QUAD_STRIP,
            buffers: &[Some(stark_shaders::mirror::composite::instance_layout(
                wgpu::VertexStepMode::Instance,
            ))],
            targets: &[
                desc::blended_target(color_space.color_format(), Some(color_space.color_blend())),
                desc::blended_target(SCRATCH_AUX_FORMAT, Some(color_space.aux_blend())),
            ],
        },
    );
    let composite_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("stark dynamics composite sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    // ---- The stamp loop: one module, seven entry points — `snapshot`, `exchange`,
    // `wick_x`, `wick_y`, `bake`, `deposit`, `settle` — over five bind group layouts,
    // since the two wick axes share `exchange`'s (they need a strict subset of it).
    // Every layout includes the dynamic-offset stamp uniform at binding 0; the binding
    // numbers partition the module's group(0) — see dynamics.wesl.
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("stark dynamics loop"),
        source: wgpu::ShaderSource::Wgsl(stark_shaders::dynamics().into()),
    });
    // Every layout below is compute-visible and opens with the dynamic-offset stamp
    // slot; the binding numbers partition the module's group(0), so a layout lists
    // only the bindings its own entry point reads.
    let comp = wgpu::ShaderStages::COMPUTE;
    let params_entry = desc::uniform_slot(0, comp, SLOT as u64);
    let ctex = |binding: u32, filterable: bool| {
        if filterable {
            desc::sample_tex(binding, comp)
        } else {
            desc::load_tex(binding, comp)
        }
    };
    let stor = |binding: u32| desc::storage_tex(binding, comp, wgpu::TextureFormat::Rgba16Float);
    // The baked swept prefix is fp32 — it is differenced per fragment, like the
    // prefix-τ volume, so f16 would band exactly where the difference is smallest.
    let stor32 = |binding: u32| desc::storage_tex(binding, comp, BAKE_FORMAT);
    let csamp = desc::sampler(5, comp);
    let snapshot_bgl = desc::bind_group_layout(
        device,
        "stark dynamics snapshot bgl",
        &[
            params_entry,
            ctex(1, false),
            ctex(2, false),
            stor(3),
            stor(4),
        ],
    );
    let exchange_bgl = desc::bind_group_layout(
        device,
        "stark dynamics exchange bgl",
        &[
            params_entry,
            ctex(1, true),
            ctex(2, true),
            // The footprint snapshot's targets: the segment's `snapshot` runs from the
            // tail of the `exchange` grid rather than from a dispatch of its own
            // (`dynamics.wesl::exchange`), so its writes belong to this layout. The
            // `wick`, which shares this layout, leaves them alone.
            stor(3),
            stor(4),
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
    );
    // `bake` integrates the reservoir along the travel axis for one segment; the
    // deposit then reads the result instead of point-sampling the reservoir.
    let bake_bgl = desc::bind_group_layout(
        device,
        "stark dynamics bake bgl",
        &[
            params_entry,
            csamp,
            ctex(7, true),
            ctex(8, true),
            stor32(17),
            stor32(18),
        ],
    );
    // The pen-up settle: the deposit's targets and snapshot, and the deposit's *baked*
    // reservoir reads too — its parcel is the delivery integral of the remaining pass,
    // which the settle slot's own `bake` dispatch stores (`dynamics.wesl::settle`),
    // not the cell that happens to sit overhead.
    let settle_bgl = desc::bind_group_layout(
        device,
        "stark dynamics settle bgl",
        &[
            params_entry,
            ctex(19, false),
            ctex(20, false),
            ctex(11, false),
            ctex(12, false),
            stor(13),
            stor(14),
            ctex(21, false),
            // The ground (§6.4): the settle lays paint, so it reads the tooth too.
            ctex(22, false),
        ],
    );
    let deposit_bgl = desc::bind_group_layout(
        device,
        "stark dynamics deposit bgl",
        &[
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
            // The canvas surface's height map — the deposition tooth (§6.4). Read
            // nearest, so it needs no sampler and is not filterable.
            ctex(22, false),
        ],
    );
    // The deposit's prefix-τ volume (group 1) — same shape as the fast path's
    // prefix binding, but compute-visible.
    let prefix_bgl = desc::bind_group_layout(
        device,
        "stark dynamics prefix bgl",
        &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        }],
    );
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
    // The wick reads the reservoir and writes the other half of the ping-pong, which is
    // exactly what `exchange_bgl` already describes — the region and selection bindings
    // it also carries simply go unused, which a pipeline layout is free to do.
    //
    // One pipeline per axis: the stencil is separable, so a firing is two 1-D passes
    // (`dynamics.wesl::wick_axis`). The axis rides in the entry point rather than the
    // uniform because both passes of a firing share the segment's slot.
    let wick_pipelines = (
        cpipe("stark dynamics wick x", "wick_x", &[Some(&exchange_bgl)]),
        cpipe("stark dynamics wick y", "wick_y", &[Some(&exchange_bgl)]),
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
    // The settle reads the prefix-τ volume too (group 1): its exposure is a pair of
    // readings of it, which is what makes the pen-up fade over the whole tip rather
    // than over the few pixels of its coverage knee.
    let settle_pipeline = cpipe(
        "stark dynamics settle",
        "settle",
        &[Some(&settle_bgl), Some(&prefix_bgl)],
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
    // One slot per tile the piece writes back, selected by a dynamic offset: the
    // region bindings beside it are the same for every tile, so the whole group is
    // built once per piece ([`UNIFORM_STRIDE`]).
    let slice_bgl = desc::bind_group_layout(
        device,
        "stark dynamics slice bgl",
        &[
            desc::uniform_slot(0, frag, SLICE_SLOT),
            desc::load_tex(1, frag),
            desc::load_tex(2, frag),
        ],
    );
    let slice_layout =
        desc::pipeline_layout(device, "stark dynamics slice layout", &[Some(&slice_bgl)]);
    let slice_pipeline = desc::fullscreen_pipeline(
        device,
        "stark dynamics slice pipeline",
        &slice_layout,
        &slice_shader,
        ("vs_main", "fs_main"),
        &[
            desc::target(color_space.color_format()),
            desc::target(color_space.aux_format()),
        ],
    );

    DynamicsKit {
        composite_pipeline,
        composite_view_bgl,
        composite_tile_bgl,
        composite_sampler,
        snapshot_pipeline,
        snapshot_bgl,
        exchange_pipeline,
        exchange_bgl,
        wick_pipelines,
        bake_pipeline,
        bake_bgl,
        deposit_pipeline,
        deposit_bgl,
        settle_pipeline,
        settle_bgl,
        exchange_sampler,
        slice_pipeline,
        slice_bgl,
        prefix_bgl,
    }
}

/// Build the stroke integrate pipeline (`integrate` shader) — §6.2/§6.1. A
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
    let frag = wgpu::ShaderStages::FRAGMENT;
    let bgl = desc::bind_group_layout(
        device,
        "stark integrate bgl",
        &[
            desc::load_tex(0, frag), // base color
            desc::load_tex(1, frag), // base aux
            desc::load_tex(2, frag), // scratch color
            desc::load_tex(3, frag), // scratch aux
            desc::load_tex(4, frag), // selection mask (§6.8) — this tile's, or a 1×1 constant
        ],
    );
    let layout = desc::pipeline_layout(device, "stark integrate layout", &[Some(&bgl)]);
    // No blend on either target: the shader does the combine and writes straight
    // through.
    let pipeline = desc::fullscreen_pipeline(
        device,
        "stark integrate pipeline",
        &layout,
        &shader,
        ("vs_main", "fs_main"),
        &[
            desc::target(color_space.color_format()),
            desc::target(color_space.aux_format()),
        ],
    );
    (pipeline, bgl)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::Vec2;

    /// A straight segment of `length` from `start` along `dir`, at arc length `dist`.
    /// The plan builders read the frame, the radius and the arc clock; the paint rates
    /// are left at zero except where a test sets one, so a value that mattered would
    /// have to be given deliberately.
    fn seg(start: Vec2, dir: Vec2, length: f32, radius: f32, dist: f32) -> Segment {
        Segment {
            start,
            dir,
            curvature: 0.0,
            radius,
            length,
            orient: 0.0,
            dist,
            add: 0.0,
            lift: 0.0,
            deposit: 0.0,
            bleed: 0.0,
            tooth: 0.0,
        }
    }

    /// `n` straight segments of `len` each, running +x from the origin — a stroke cut
    /// the way the flattener would cut a steady drag.
    fn run(n: usize, len: f32, radius: f32) -> Vec<Segment> {
        (0..n)
            .map(|i| {
                let d = i as f32 * len;
                seg(Vec2::new(d, 0.0), Vec2::new(1.0, 0.0), len, radius, d)
            })
            .collect()
    }

    // --- the slot's lane packing -------------------------------------------

    /// [`Slot::pack`] is an **ABI**, not a convenience: the lanes it fills are read by
    /// `dynamics.wesl`'s accessors, which name them from the other side. So every field
    /// is pinned to the component it lands in, with a value that appears nowhere else —
    /// a swap of two same-typed neighbours (`lambda_lift` for `lambda_deposit`, `orient`
    /// for `drain`) is otherwise a silent wrong picture that no golden would attribute
    /// to the right cause.
    ///
    /// The generated `offset_of` assertions pin where a *lane* starts. This is what
    /// pins what lives inside one.
    #[test]
    fn every_slot_field_lands_in_the_lane_the_shader_reads_it_from() {
        let packed = Slot {
            start: Vec2::new(1.0, 2.0),
            dir: Vec2::new(3.0, 4.0),
            radius: 5.0,
            travel_radii: 6.0,
            lambda_lift: 7.0,
            lambda_deposit: 8.0,
            channels: [9.0, 10.0, 11.0, 12.0],
            rect_origin: Vec2::new(13.0, 14.0),
            orient: 15.0,
            drain: 16.0,
            add: 17.0,
            curvature: 18.0,
            bleed_reach: 19.0,
            noise_freq: [20.0, 21.0, 22.0, 23.0],
            noise_amp: [24.0, 25.0, 26.0],
            noise_off: [27.0, 28.0],
            dist: 29.0,
            bearing: 30.0,
            lambda_bleed: 31.0,
            tooth: 32.0,
            weave_scale: 33.0,
            weave_bias: Vec2::new(34.0, 35.0),
        }
        .pack();

        assert_eq!(packed.a, [1.0, 2.0, 3.0, 4.0], "a: start.xy, dir.zw");
        assert_eq!(
            packed.b,
            [5.0, 6.0, 7.0, 8.0],
            "b: radius, travel, λ_lift, λ_dep"
        );
        assert_eq!(packed.c, [9.0, 10.0, 11.0, 12.0], "c: colour + opacity");
        assert_eq!(
            packed.d,
            [13.0, 14.0, 15.0, 16.0],
            "d: rect.xy, orient, drain"
        );
        assert_eq!(
            packed.e,
            [17.0, 18.0, 19.0, 0.0],
            "e: add, curvature, reach, —"
        );
        assert_eq!(packed.f, [20.0, 21.0, 22.0, 23.0], "f: noise frequency");
        assert_eq!(
            packed.g,
            [24.0, 25.0, 26.0, 29.0],
            "g: noise amplitude, dist"
        );
        assert_eq!(
            packed.h,
            [27.0, 28.0, 30.0, 31.0],
            "h: noise off, bearing, λ_bleed"
        );
        assert_eq!(
            packed.i,
            [32.0, 33.0, 34.0, 35.0],
            "i: tooth, weave scale + bias"
        );
    }

    /// The neutral slot is neutral *in the shader's terms*, which for one field is not
    /// zero: a `bearing` of 0 books the tool's half of every transfer against no ground
    /// at all — infinite tooth, not absent tooth — so a derived `Default` would make
    /// the two slot kinds that leave it alone quietly wrong.
    #[test]
    fn the_default_slot_is_neutral_rather_than_zeroed() {
        let d = Slot::default().pack();
        assert_eq!(d.h[2], 1.0, "the default bearing must be 1, not 0");
        for (lane, name) in [
            (d.a, "a"),
            (d.b, "b"),
            (d.c, "c"),
            (d.d, "d"),
            (d.e, "e"),
            (d.f, "f"),
            (d.g, "g"),
            (d.i, "i"),
        ] {
            assert_eq!(lane, [0.0; 4], "lane {name} is not neutral by default");
        }
        assert_eq!(
            [d.h[0], d.h[1], d.h[3]],
            [0.0; 3],
            "lane h beyond the bearing"
        );
    }

    // --- the bleed cadence -------------------------------------------------

    /// **The claim `bleed_fires` is built on**: which windows fire, and what each one
    /// sweeps, is a pure function of the record — not of where the renderer happened to
    /// cut the stroke into pieces or ranges.
    ///
    /// This is a `preview == committed` property (§1.3) in the one place it cannot be
    /// repainted. A live tail always starts at a span boundary while the commit renders
    /// the whole stroke from zero, so if a window came out shorter for one than the
    /// other, a bleeding stroke would visibly lighten the moment the pointer came up —
    /// which it did, before the window learned to walk back along the crossing
    /// segment's own arc instead of looking its start up among the segments in hand.
    ///
    /// Checked at every cut point rather than one, since the interesting cuts are
    /// exactly the ones that land mid-window.
    #[test]
    fn bleed_firings_do_not_depend_on_where_the_stroke_was_cut() {
        // Segments well under the quantum (0.5 · radius = 5px), so windows routinely
        // reach back over several of them and a cut can land inside one.
        let all = run(40, 1.5, 10.0);
        let whole: Vec<_> = bleed_fires(0.4, &all)
            .into_iter()
            .map(|(i, f)| (i, f.start, f.length, f.dist))
            .collect();
        assert!(
            whole.len() > 3,
            "the case does not fire often enough to be interesting: {}",
            whole.len()
        );

        for cut in 1..all.len() {
            let mut split: Vec<_> = bleed_fires(0.4, &all[..cut])
                .into_iter()
                .map(|(i, f)| (i, f.start, f.length, f.dist))
                .collect();
            split.extend(
                bleed_fires(0.4, &all[cut..])
                    .into_iter()
                    .map(|(i, f)| (i + cut, f.start, f.length, f.dist)),
            );
            assert_eq!(
                split, whole,
                "cutting after segment {cut} changed the firings"
            );
        }
    }

    /// The firings of a curved segment **tile it**, a quantum each, and every one of
    /// them lies on the path rather than on a chord across it.
    ///
    /// Two properties that hold each other up. Tiling is what makes the axis a
    /// diffusivity: a firing carries a fixed variance, so N quanta of travel have to
    /// arrive as N firings or the axis is quietly scaled by `1/N` (`bleed_stencil`).
    /// And a window on the arc is what makes each of those tiles a stretch of the path
    /// rather than an approximation to one — at this cadence the bow a chord would sit
    /// off it is under a thousandth of a texel, so this is not a correction the picture
    /// needs today; it is that the representation cannot go wrong if the cadence is
    /// ever coarsened, which is the lever `BLEED_REACH_MAX` names as the way to
    /// diffuse further.
    #[test]
    fn a_segments_firings_tile_it_along_its_own_arc() {
        use super::super::budget::MAX_TIP_TURN;
        // A 40 px brush at the tightest arc the flattener will sweep it along, its
        // size modulated down to a 3 px tip — and segments at the travel cap, which is
        // priced off the 40 rather than the 3, so each crosses the cadence many times.
        let (nominal, tip, len) = (40.0f32, 3.0f32, 40.0f32);
        let kappa = MAX_TIP_TURN / nominal;
        let r = 1.0 / kappa;
        let centre = Vec2::new(0.0, r);
        let quantum = BLEED_TRAVEL_QUANTUM * tip;

        let mut segs = Vec::new();
        let (mut p, mut d, mut dist) = (Vec2::ZERO, Vec2::new(1.0, 0.0), 0.0);
        for _ in 0..20 {
            let mut s = seg(p, d, len, tip, dist);
            s.curvature = kappa;
            segs.push(s);
            (p, d) = crate::path::arc_at(p, d, kappa, len);
            dist += len;
        }

        let fires = bleed_fires(0.4, &segs);
        // The cap is what stops this being `len / quantum` = 53 per segment.
        assert_eq!(fires.len(), 20 * MAX_BLEED_FIRES_PER_SEGMENT);
        for (i, f) in &fires {
            assert_eq!(
                f.length, quantum,
                "a firing swept more than its own quantum"
            );
            // Every point of the window sits on the circle the path traced — not just
            // its two ends, which the walk back along the crossing segment's own arc
            // already put there.
            for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
                let on = crate::path::arc_at(f.start, f.dir, f.curvature, f.length * t).0;
                assert!(
                    ((on - centre).length() - r).abs() < 1e-2,
                    "the window left the path {} px at t = {t}",
                    (on - centre).length() - r,
                );
            }
            // Butted end to end, back from the segment's end: no quantum of travel is
            // diffused twice, and none is skipped between the cap and that end.
            let s = &segs[*i];
            let from_end = s.dist + s.length - f.dist;
            let quanta = from_end / quantum;
            assert!(
                (quanta - quanta.round()).abs() < 1e-3
                    && quanta <= MAX_BLEED_FIRES_PER_SEGMENT as f32 + 1e-3,
                "a firing sits {quanta} quanta back from its segment's end",
            );
        }
    }

    /// A brush that does not bleed fires nothing at all — the early-out is exact, and
    /// is what lets every non-bleeding stroke keep the no-bleed path bit-for-bit.
    #[test]
    fn a_brush_that_does_not_bleed_fires_nothing() {
        assert!(bleed_fires(0.0, &run(40, 1.5, 10.0)).is_empty());
    }

    /// **Why the cadence exists at all**: a firing's window is a quarter-radius of travel
    /// however finely the path was cut, so its exposure is a well-conditioned prefix
    /// difference rather than the f16 noise a per-segment flux would be.
    ///
    /// A hand that draws slowly is fitted at a control point per pointer sample — the
    /// repro that prompted this carried 177 knots over 68 px — and at that cut a texel's
    /// per-segment flux lands under the f16 ULP of the height it is editing, so every
    /// store either snaps it away or ratchets a whole ULP. One firing moves what those
    /// micro-segments would each have tried to move, in a step far above the floor.
    #[test]
    fn a_firing_sweeps_its_own_quantum_however_finely_the_path_was_cut() {
        let radius = 20.0;
        let quantum = BLEED_TRAVEL_QUANTUM * radius;
        // 0.39 px a segment — the repro's mean span.
        let fine = run(400, 0.39, radius);
        let fires = bleed_fires(0.4, &fine);
        assert!(fires.len() > 5, "only {} firings", fires.len());
        for (_, f) in &fires {
            assert!(
                (f.length - quantum).abs() < 0.5,
                "a firing swept {} of the {quantum} its cadence carries",
                f.length,
            );
            assert!(
                f.length > 10.0 * 0.39,
                "the window is segment-sized, which is the regime the cadence exists \
                 to leave",
            );
        }
    }

    // --- the pen-up frame --------------------------------------------------

    /// A brush that manipulates paint, so a stroke of it settles at all.
    fn smearing(radius: f32) -> crate::document::BrushParams {
        crate::document::BrushParams {
            radius,
            dynamics: crate::document::BrushDynamics {
                lift: 0.8,
                deposit: 0.8,
                ..crate::document::BrushDynamics::default()
            },
            ..crate::document::BrushParams::default()
        }
    }

    /// A stroke through `pts` with `brush`, as plain full-pressure knots.
    fn record(brush: crate::document::BrushParams, pts: &[Vec2]) -> StrokeRecord {
        StrokeRecord {
            layer: crate::document::LayerId(0),
            brush,
            path: pts
                .iter()
                .map(|p| crate::path::ControlPoint::at(*p))
                .collect(),
            seed: 0,
        }
    }

    /// The segments of `range` of `rec`, at the budget the loop would flatten it with.
    fn segments_of(rec: &StrokeRecord, range: std::ops::Range<usize>) -> Vec<Segment> {
        generate_segments_in(
            rec,
            flatten_tolerance(&rec.brush),
            StrokeSpans { range, dist: 0.0 },
        )
        .0
    }

    /// What [`settle_tangent`] used to do: walk back a radius through **the segments in
    /// hand**. Kept here as the thing the test below measures against — it is the
    /// behaviour that made a stroke's fade-out cap turn as the pointer came up.
    fn piece_local_tangent(segments: &[Segment], end: Vec2) -> Vec2 {
        let radius = segments.last().map_or(1.0, |s| s.radius);
        let mut back = end;
        let mut acc = 0.0;
        for s in segments.iter().rev() {
            back = s.start;
            acc += s.length;
            if acc >= radius {
                break;
            }
        }
        let v = end - back;
        let len = v.length();
        if len > 1e-4 {
            v / len
        } else {
            segments.last().map_or(Vec2::new(1.0, 0.0), |s| s.dir)
        }
    }

    /// **The claim [`settle_tangent`] is built on**: the pen-up frame is a function of
    /// the record, not of where the renderer happened to cut the stroke into ranges and
    /// pieces.
    ///
    /// A live tail starts at a span boundary while the commit renders the whole stroke
    /// from zero, and both run the settle — the tail's range reaches the stroke's end,
    /// which is exactly the condition that asks for one. So a frame measured over "the
    /// segments in hand" came out over a shorter window for the tail than for the
    /// commit, and on a curving stroke that is a different direction: the settle's
    /// `min(owed, received)` lens is elongated along it, so the fade-out cap visibly
    /// turned at pen-up. That is a `preview == committed` break (§1.3) in the one place
    /// it cannot be repainted — the same one `bleed_fires` was fixed for.
    ///
    /// Checked at every cut point, since the interesting ones are those that leave the
    /// tail shorter than the tip being settled.
    #[test]
    fn the_settle_frame_does_not_depend_on_where_the_stroke_was_cut() {
        // A circle of radius 200 under a 60 px tip: a footprint's worth of path turns
        // through 60/200 ≈ 17°, so a lookback that comes up short points somewhere
        // measurably different.
        let curve: Vec<Vec2> = (0..=16)
            .map(|i| {
                let t = i as f32 / 16.0 * 1.2;
                Vec2::new(200.0 * t.sin(), 200.0 * (1.0 - t.cos()))
            })
            .collect();
        let rec = record(smearing(60.0), &curve);
        let all = crate::path::span_count(rec.path.len());
        let whole = segments_of(&rec, 0..all);
        let want = settle_tangent(&rec, flatten_tolerance(&rec.brush), &whole);

        let mut ever_differed = false;
        for cut in 1..all {
            let tail = segments_of(&rec, cut..all);
            if tail.is_empty() {
                continue;
            }
            let got = settle_tangent(&rec, flatten_tolerance(&rec.brush), &tail);
            assert!(
                (got - want).length() < 1e-4,
                "cutting at span {cut} moved the settle frame from {want:?} to {got:?}",
            );
            // And the old piece-local walk really does move on these cuts, so the
            // assertion above is not passing because the case is uninteresting.
            let last = whole.last().expect("segments");
            let end = crate::path::arc_at(last.start, last.dir, last.curvature, last.length).0;
            let stale = piece_local_tangent(&tail, end);
            ever_differed |= (stale - piece_local_tangent(&whole, end)).length() > 1e-2;
        }
        assert!(
            ever_differed,
            "no cut left a tail short enough to move the old frame — the test proves \
             nothing",
        );
    }

    /// [`settle_tangent`] must survive the way a real pen-up arrives: a hand pauses
    /// before it lifts, so the last samples cluster at one point and the flattener
    /// turns them into edges whose chord is a rounding error and whose direction is
    /// therefore arbitrary.
    ///
    /// Nothing else in the loop notices — a segment of no length deposits nothing, so
    /// its direction never reaches a pixel. The settle is the exception: it takes a
    /// whole tip's worth of exchange from that one frame, and its `min(owed, received)`
    /// lens is elongated *along* it, so a wrong direction lays a tip-shaped disc across
    /// the stroke instead of along it. That is what a wandering fade-out cap was.
    ///
    /// Reading the record's own polyline is what makes this structural rather than a
    /// rule to remember: knots piled on one spot contribute nothing to the chord over
    /// the last radius, so they cannot steer it whatever direction the fitter gave the
    /// edges between them.
    #[test]
    fn the_settle_frame_ignores_a_paused_hands_arbitrary_last_edges() {
        // A straight drag along +y, then the pause: four knots on the stopping point.
        let mut pts: Vec<Vec2> = (0..20).map(|i| Vec2::new(0.0, i as f32 * 2.0)).collect();
        let stop = Vec2::new(0.0, 38.0);
        pts.extend([stop; 4]);
        let rec = record(smearing(12.0), &pts);
        let all = crate::path::span_count(rec.path.len());
        let segs = segments_of(&rec, 0..all);

        let tan = settle_tangent(&rec, flatten_tolerance(&rec.brush), &segs);
        assert!(
            (tan - Vec2::new(0.0, 1.0)).length() < 1e-2,
            "the settle frame followed the paused hand: {tan:?}"
        );
    }

    /// A click has no travel to measure a direction over — a lone control point is not
    /// a curve, so there is no polyline to walk — and still gets a real frame: the
    /// dab's own, which `generate_segments_in` gave it deliberately.
    #[test]
    fn a_click_settles_along_its_dab() {
        let rec = record(smearing(10.0), &[Vec2::new(4.0, -7.0)]);
        let segs = segments_of(&rec, 0..crate::path::span_count(rec.path.len()));
        assert_eq!(segs.len(), 1, "a click is one swept dab");
        assert_eq!(
            settle_tangent(&rec, flatten_tolerance(&rec.brush), &segs),
            segs[0].dir,
        );
    }

    // --- the dispatch rects fit the scratch --------------------------------

    /// Every dispatch rect a piece builds fits the snapshot scratch that piece sized.
    ///
    /// The two are related by an argument spread over four numbers — [`snapshot_size`]'s
    /// `+3` and `+2` against [`dispatch_rect`]'s margin and its rounding — and the
    /// `min` in `dispatch_rect` means getting it wrong clips a footprint rather than
    /// failing. `dispatch_rect` asserts the fit; this is what runs the assert over
    /// shapes chosen to stress it, on the CPU, with no adapter needed.
    #[test]
    fn every_dispatch_rect_fits_the_scratch_its_piece_sized() {
        // Sub-pixel origins are the point: the rect floors its origin and rounds its
        // far edge out, so the worst case is a box straddling texel boundaries at both
        // ends. Curvature is swept too, since a bent sweep bows a sagitta out of its box.
        for &radius in &[0.5f32, 1.0, 7.3, 40.0, 120.0] {
            for &length in &[0.0f32, 0.37, 4.0, 60.0] {
                for &kappa in &[0.0f32, 0.004, -0.02] {
                    for &frac in &[0.0f32, 0.499, 0.5, 0.999] {
                        let start = Vec2::new(frac, -frac);
                        let mut s = seg(start, Vec2::new(1.0, 0.0), length, radius, 0.0);
                        s.curvature = kappa;
                        let segments = [s];
                        let dsize = snapshot_size(&segments, &[]);

                        let (lo, hi) = coverage_bounds(&s);
                        // Region origins that put the box at both ends of the region.
                        for &origin in &[Vec2::ZERO, lo.floor(), Vec2::new(-13.7, 91.2)] {
                            dispatch_rect(lo, hi, origin, dsize);
                            // The pen-up's square, which is sized from the same scratch
                            // but built from the tip alone rather than the swept box.
                            let end = crate::path::arc_at(s.start, s.dir, s.curvature, s.length).0;
                            dispatch_rect(
                                end - Vec2::splat(radius),
                                end + Vec2::splat(radius),
                                origin,
                                dsize,
                            );
                        }
                    }
                }
            }
        }
    }

    /// A bleed window can be the largest footprint in its piece — it sweeps up to a
    /// half-radius where the piece's own segments may be sub-pixel — so the scratch has
    /// to be sized with the firings in it, not just the segments.
    #[test]
    fn the_scratch_is_sized_with_the_bleed_windows_in_it() {
        // Long enough to cross the 0.5 · 30 = 15 px cadence, cut far finer than it.
        let segments = run(200, 0.2, 30.0);
        let fires = bleed_fires(0.5, &segments);
        assert!(!fires.is_empty(), "no firing to size against");
        let with = snapshot_size(&segments, &fires);
        let without = snapshot_size(&segments, &[]);
        assert!(
            with > without,
            "a firing's window did not widen the scratch ({without} -> {with})"
        );
        for (_, f) in &fires {
            let (lo, hi) = coverage_bounds(f);
            dispatch_rect(lo, hi, Vec2::ZERO, with);
        }
    }

    // `the_host_and_the_shader_agree_on_the_loops_constants` stood here, reading
    // `BAKE_RES` and `WICK_HALF` out of the linked shader. Both are generated now, so
    // there is one declaration of each; the wick's cadence relation moved to a
    // compile-time assertion beside `WICK_TRAVEL_QUANTUM` (`budget.rs`), which is
    // where it constrains something. `WICK_RATE` — which this test noted it could not
    // read, the linker having stripped a constant that survives only in prose — is
    // read from the unlinked source and is part of that assertion.

    // `the_stamp_struct_has_the_same_nine_lanes_on_both_sides` stood here, counting
    // `vec4<f32>` in the shader source and comparing against [`SLOT`]. There is no
    // longer a second declaration for it to disagree with: `Stamp` is generated from
    // the WESL, and the generator emits `offset_of` assertions per lane, so a tenth
    // lane moves both sides at once and a mistake in the layout is a build failure.
}
