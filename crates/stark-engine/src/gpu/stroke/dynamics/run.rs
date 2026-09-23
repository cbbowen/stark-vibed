//! Recording one dynamics render: the reservoir ping-pong, the carried lanes, and the
//! compute pass the plan's slots are dispatched through (§6.2).
//!
//! The region around them — the piece prologue, the composite, the selection gather,
//! the stamp upload, the slice and the epilogue — is [`region`](super::region)'s, which
//! the liquify path runs around a different kernel (§6.13, `liquify.rs`).
//!
//! The half that owns GPU state: what it is *asked* to record comes from
//! [`plan`](super::plan), what it records *with* from [`kit`](super::kit).

use std::sync::Arc;

use crate::gpu::desc;
use crate::gpu::tile::TileMap;

use super::super::incremental::{Carried, LoopCarry, LoopTiles};
use super::super::region::chunk_segments;
use super::super::segments::{BleedFire, Segment};
use super::super::swept::{SweepPass, segment_instance, tile_xform, xform_group};
use super::super::volume::ResolvedTip;
use super::super::{Complaint, ResolvedRange, StrokeCarry, StrokeRenderer, StrokeScene, ToolState};
use super::bleed::bleed_fires;
use super::plan::{LoopDispatch, WetStep, cell_scratch_size, dynamics_plan};
use super::region::{
    LOOP_USAGE, Piece, REGION_WRITES, RegionBox, RegionChannels, RegionRun, composite_tiles,
    region_selection, slice_region,
};
use super::tool::{BRUSH_RES, ToolPingPong};
use super::trio::{Life, Trio};
use crate::gpu::scratch::{BufKey, Key, LeasedSlots, LeasedTex, SubmitScope};
use stark_shaders::mirror::stamp::SegmentInstance;
use stark_shaders::mirror::stamp_common::SWEEP_VERTS;
/// The label on the loop's compute pass, which the ceiling lane's render pass cuts in
/// two and which resumes under the same name after it.
const LOOP_PASS: &str = "stark dynamics stamp loop";
// Resolution of the per-segment **swept prefix** of the reservoir
// (`dynamics.wesl::bake`), and the workgroup width of the scan that builds it —
// generated from the shader, which is the side that decides it (§6.10). A mismatch
// scanned the wrong width and rendered subtly wrong without crashing.
use stark_shaders::mirror::dynamics::BAKE_RES;
// The `@binding` indices, generated from `dynamics.wesl`'s own declarations (§6.10):
// the bind groups here and the layouts in [`kit`](super::kit) both name the
// shader's numbering rather than keeping copies of it.
use stark_shaders::mirror::dynamics::binding as b;
// The storage-texture *formats*, taken from the same declarations the indices come
// from (§6.10): a slot names its own format, so the host asks rather than repeats.
use stark_shaders::mirror::dynamics::decl as d;
use stark_shaders::mirror::dynamics_common::Stamp;
use stark_shaders::mirror::dynamics_common::binding as sb;
use stark_shaders::mirror::dynamics_common::decl as sd;

use stark_model::document::{BrushDynamics, StrokeRecord};

/// The mint-budget tile's pool key (`LoopCarry::fresh`): copy usages only — no pass
/// binds one. They ferry the region aux's `.yz` lanes across pieces, and the whole aux
/// texel rides along because a copy cannot address two lanes of one; seeding the `.x`
/// height back is bit-identical, being the value the write-back sliced out.
///
/// A whole `TILE_TEX` block — interior plus apron — cut from and seeded into the
/// region aux exactly as the write-back cuts paint tiles, so the two address the
/// region with one rule. Adjacent blocks overlap by the apron and are cut from one
/// region, so seeding them back writes identical texels twice. The copies take their
/// extent from this key ([`Key::extent`]).
const fn fresh_key() -> Key {
    Key::tile(
        // The region aux's own format: these blocks are cut from and seeded into it,
        // and a copy between two formats is not a copy (§6.10).
        sd::REGION_AUX_W.storage_format(),
        wgpu::TextureUsages::COPY_SRC.union(wgpu::TextureUsages::COPY_DST),
        "stark dynamics fresh",
    )
}

/// The ceiling lane's carried tile ([`LoopCarry::levels`], §6.2): [`fresh_key`]'s
/// twin, for the lane's own sums, cut from and seeded into the region's lane by
/// the same copies.
const fn levels_key() -> Key {
    Key::tile(
        super::super::swept::CEILING_FORMAT,
        wgpu::TextureUsages::COPY_SRC.union(wgpu::TextureUsages::COPY_DST),
        "stark dynamics levels",
    )
}

/// One lane of the region, carried per tile across pieces and ranges (§6.2): the
/// mint budget's raw totals ([`fresh_key`]) or the ceiling lane's sums
/// ([`levels_key`]), under the one contract [`LoopCarry::fresh`] states — a piece
/// never writes a tile it resumed from. It [`seed`](Self::seed)s its region by
/// copying each carried block in and [`extract`](Self::extract)s a fresh lease back
/// out, so the tiles it does not touch ride forward as clones of the same lease.
struct CarriedLanes {
    tiles: LoopTiles,
    /// The pool key every tile here is cut with; its extent is every copy's.
    key: Key,
}

impl CarriedLanes {
    fn new(key: Key, tiles: LoopTiles) -> Self {
        Self { tiles, key }
    }

    /// Copy each carried tile of `piece` over its block of `region`.
    fn seed(&self, scope: &mut SubmitScope, piece: &Piece, region: &wgpu::Texture) {
        for coord in &piece.tiles {
            let Some(kept) = self.tiles.get(coord) else {
                continue;
            };
            scope.encoder().copy_texture_to_texture(
                kept.tex().as_image_copy(),
                block(region, piece.block_origin(*coord)),
                self.key.extent(),
            );
        }
    }

    /// Cut a fresh lease out of `region` for every tile of `piece`, and carry it in
    /// place of whatever was carried there.
    fn extract(
        &mut self,
        r: &StrokeRenderer,
        scope: &mut SubmitScope,
        piece: &Piece,
        region: &wgpu::Texture,
    ) {
        for coord in &piece.tiles {
            let kept = r.scratch.keep(&r.ctx.device, self.key);
            scope.encoder().copy_texture_to_texture(
                block(region, piece.block_origin(*coord)),
                kept.tex().as_image_copy(),
                self.key.extent(),
            );
            // Held, not dropped: the lease this displaces is the previous piece's,
            // still named by a seed copy recorded into an encoder this call does not
            // submit. Dropping it would return the texture to the free list while
            // pending commands read it. Consecutive pieces share the tiles around
            // their cut, so this is the ordinary case, not a corner.
            if let Some(displaced) = self.tiles.get(coord) {
                scope.hold(Arc::clone(displaced));
            }
            self.tiles.insert_mut(*coord, Arc::new(kept));
        }
    }
}

/// The tile-sized block of `region` at `origin`, as a copy addresses it.
fn block(region: &wgpu::Texture, origin: wgpu::Origin3d) -> wgpu::TexelCopyTextureInfo<'_> {
    wgpu::TexelCopyTextureInfo {
        texture: region,
        mip_level: 0,
        origin,
        aspect: wgpu::TextureAspect::All,
    }
}

/// **The one value that says the pen drives this stroke's ceiling** (§6.2): the brush
/// as the *swept* sweep binds it — prefix-τ and noise over the swept layouts, so the
/// lane's sums are the fast path's — and the lane's own carried tiles. Absent on every
/// other stroke, which is what `ceiling_lane = 0` says on the shader's side, and
/// implies `consts.capped`.
struct CeilingLane {
    prefix: wgpu::BindGroup,
    noise: wgpu::BindGroup,
    /// The lane's tiles ([`LoopCarry::levels`]), evolving with the run exactly as the
    /// mint budget beside them does.
    carried: CarriedLanes,
}

impl CeilingLane {
    /// The lane of a stroke whose ceiling the pen drives, resuming from `carried`.
    fn of(
        region: &mut RegionRun<'_>,
        tip: &ResolvedTip,
        rec: &StrokeRecord,
        carried: LoopTiles,
    ) -> Option<Self> {
        let (r, consts, substrate) = (region.r, region.consts, region.scene.substrate);
        consts.ceiling_lane.then(|| {
            let (prefix, noise) = super::super::swept::sweep_brush_groups(
                r,
                &mut region.scope,
                tip,
                rec,
                substrate,
                consts,
            );
            Self {
                prefix,
                noise,
                carried: CarriedLanes::new(levels_key(), carried),
            }
        })
    }

    /// Open the lane over one piece: the region-sized lane it draws into — **drawn**
    /// into per segment, so a render attachment rather than storage, and read by the
    /// deposits between draws — cleared (the pool hands back stale texels), seeded from
    /// the carry, and given the piece's own draw state.
    fn open<'p>(
        &'p self,
        region: &mut RegionRun<'_>,
        piece: &Piece,
        segments: &[Segment],
        bbox: RegionBox,
    ) -> LanePiece<'p> {
        let (r, consts) = (region.r, region.consts);
        let RegionBox { origin, w, h } = bbox;
        let lane = region.scope.take_piece(Key {
            size: (w, h),
            format: super::super::swept::CEILING_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            label: "stark dynamics region levels",
        });
        region
            .scope
            .encoder()
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stark dynamics region levels clear"),
                color_attachments: &[Some(desc::attach(&lane.view, desc::CLEAR))],
                ..Default::default()
            });
        self.carried.seed(&mut region.scope, piece, &lane.tex);
        let instances: Vec<SegmentInstance> = segments.iter().map(segment_instance).collect();
        let bytes: &[u8] = bytemuck::cast_slice(&instances);
        let buf = region.scope.take_piece_buffer(BufKey {
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            label: "stark dynamics levels instances",
        });
        region.scope.write_lease(&buf, bytes);
        let xform = tile_xform(consts, origin, (w as f32, h as f32), 1);
        LanePiece {
            lane,
            xform: xform_group(r, &mut region.scope, &xform),
            instances: buf,
            prefix: &self.prefix,
            noise: &self.noise,
        }
    }
}

/// One piece's ceiling lane (§6.2): the region-sized lane, the piece's segments
/// instanced as the sweep draws them, and the three groups `stamp.wesl::fs_levels` is
/// drawn with — two of them the run's own, borrowed rather than copied per piece.
struct LanePiece<'a> {
    lane: LeasedTex,
    xform: wgpu::BindGroup,
    instances: wgpu::Buffer,
    prefix: &'a wgpu::BindGroup,
    noise: &'a wgpu::BindGroup,
}

impl LanePiece<'_> {
    /// What a dispatch binds where a stroke without a lane binds the kit's 1×1 zero.
    fn view(&self) -> &wgpu::TextureView {
        &self.lane.view
    }

    /// Draw segment `index`'s sums over the lane — a render pass, because the lane
    /// accumulates by blending, the one read-modify-write of a region texel the loop
    /// makes without a snapshot.
    fn draw(&self, region: &mut RegionRun<'_>, index: u32) {
        let pipeline = region.r.swept.pipeline(SweepPass::Levels);
        let mut pass = region
            .scope
            .encoder()
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stark dynamics ceiling lane"),
                color_attachments: &[Some(desc::attach(&self.lane.view, desc::LOAD))],
                ..Default::default()
            });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.xform, &[0]);
        pass.set_bind_group(1, self.prefix, &[]);
        pass.set_bind_group(2, self.noise, &[]);
        pass.set_vertex_buffer(0, self.instances.slice(..));
        pass.draw(0..SWEEP_VERTS, index..index + 1);
    }

    /// The lane itself, the piece done with it — taking it out is what hands the run's
    /// carry back for the extract.
    fn into_region(self) -> LeasedTex {
        self.lane
    }
}

impl StrokeRenderer {
    /// Render a range of a paint-manipulating stroke via the **sequential
    /// swept-exchange loop** (§6.2): composite the base under it into a 1:1 region,
    /// walk it *in order* on the GPU — the canvas-side exchange swept per flattened
    /// segment through the prefix-τ integral (the same definite-integral extent as the
    /// plain deposit), the 2-D tool reservoir taking the complement of it over the same
    /// segment — then slice the evolved region back into fresh CoW tiles.
    ///
    /// The range is drawn in as many region-sized **pieces** as it takes
    /// ([`chunk_segments`]): the loop is sequential, so pieces run back to back over
    /// the same segments in the same order, each compositing what the last wrote back.
    /// Length costs extra pieces, not correctness — where degrading past
    /// [`MAX_REGION_DIM`](super::super::budget::MAX_REGION_DIM) to the plain swept
    /// deposit would cost correctness, that path being unable to manipulate paint.
    ///
    /// Resumes from the tool it is handed rather than a fresh tip when one is given,
    /// and hands back the state it ends in whenever a further range remains, so a live
    /// stroke redraws only its tail (see [`ToolState`]). The segments and constants
    /// arrive resolved ([`ResolvedRange`]): `render_range` has already decided from the
    /// brush that this stroke runs the loop, and flattened at that decision's budget.
    pub(in crate::gpu::stroke) fn render_dynamic(
        &self,
        scene: StrokeScene<'_>,
        range: ResolvedRange<'_>,
        dynamics: BrushDynamics,
    ) -> (TileMap, StrokeCarry) {
        // The sequential stamp loop end to end. Everything below it — `stroke.piece`
        // and its phases, then `stroke.submit` — partitions this row, so the shares
        // have a denominator and a phase missing from the sum is one nobody has
        // instrumented. Flattening is not below it: `stroke.segments` is cut once for
        // every path, so it sits beside this row under `stroke.range`.
        crate::timing::span!("stroke.dynamics");
        let ResolvedRange {
            rec,
            segments,
            end_dist,
            resume,
            ..
        } = range;
        let capture = resume.capture;

        let mut run = DynamicsRun::new(self, scene, range, dynamics);
        let mut map = scene.base.clone();
        // The bleed cadence's firings for the whole range (§6.2), computed once and
        // sliced per piece: the chunker must measure every piece **with its windows**,
        // since a window can reach back a quantum before the segment it fires after —
        // for a piece's first segment, onto canvas no segment box covers
        // ([`chunk_segments`]).
        let (fires, bleed_capped) = bleed_fires(dynamics.bleed, segments);
        // The third budget, reported like the other two: past
        // `MAX_BLEED_FIRES_PER_SEGMENT` the bleed knob stops meaning what it says at
        // small modulated radii, and nothing else would say so.
        if bleed_capped && self.complained.say(rec.seed, Complaint::BleedCapped) {
            tracing::warn!(
                radius = rec.brush.size,
                cap = super::bleed::MAX_BLEED_FIRES_PER_SEGMENT,
                "a segment wants more bleed firings than it may have: the bleed axis under-delivers on this stroke",
            );
        }
        // The pen-up settle (§6.2) belongs to the range that reaches the *stroke's*
        // end, and within it to the last piece — the same condition that says there is
        // no reservoir worth keeping. A range that stops short hands its tool on
        // instead, so nothing is stranded for the settle to hand back, and a live tail
        // computes the same settle its commit will.
        let mut pieces = chunk_segments(segments, &fires).into_iter().peekable();
        while let Some(piece) = pieces.next() {
            let is_last = pieces.peek().is_none();
            // This piece's firings: the range's, narrowed. `after` still indexes the
            // range, so the piece's own first segment goes with them — the walk pairs
            // them by `first + si` rather than by a shifted copy of the list
            // (`slot_sources`).
            let first = piece.start;
            let lo = fires.partition_point(|f| f.after < first);
            let hi = fires.partition_point(|f| f.after < piece.end);
            map = run.draw(
                &map,
                &segments[piece],
                first,
                &fires[lo..hi],
                !capture && is_last,
            );
        }
        (map, run.finish(end_dist, capture))
    }
}

/// One [`StrokeRenderer::render_dynamic`] call in progress: the brush-local state the
/// loop threads along the stroke, and the GPU objects that outlive any one region.
///
/// What survives from piece to piece is what survives from one *range* to the next —
/// the tool reservoir — that being all the loop carries between segments which is not
/// already on the canvas. It lives here rather than being copied out into a
/// [`ToolState`] and back in at every cut, so the ping-pong simply keeps going.
struct DynamicsRun<'a> {
    /// The region machinery this path shares with liquify (§6.13): the scene, the
    /// scope, the dirty set, the piece prologue and the epilogue.
    region: RegionRun<'a>,
    /// Functions of the brush alone, so shared by every piece: the unintegrated depth
    /// the reservoir texels weigh their exposure by, and the color-dynamics field the
    /// `add` paint is jittered against. The prefix-τ at group 1 is
    /// [`RegionRun::prefix_bg`].
    depth: wgpu::TextureView,
    noise: super::super::tips::NoiseLease,
    /// The tool reservoir ping-pong (§6.2) — what the tip is carrying, and which half
    /// holds it.
    brush: ToolPingPong,
    /// The segment's swept reservoir prefixes (fp32, so the per-fragment difference
    /// keeps its precision — each is created at its own `bake_*_w` declaration's
    /// format). Rebuilt per segment, so a single set serves the whole stroke: nothing
    /// reads the last segment's bake.
    bake_load: wgpu::TextureView,
    bake_latm: wgpu::TextureView,
    /// The residual's own swept prefix, `∫res·m·dτ` — same differencing, and recovered
    /// against the very denominator `bake_latm` is.
    bake_rlm: Option<wgpu::TextureView>,
    /// The opacity ceiling's mint-budget tiles ([`LoopCarry::fresh`]), evolving
    /// with the run: resumed entries first, each piece replacing the tiles it
    /// touched with the leases it extracted. Untouched — and empty — at the
    /// identity opacity.
    fresh: CarriedLanes,
    /// The ceiling lane (§6.2), on a stroke whose ceiling the pen drives.
    ceiling: Option<CeilingLane>,
}

impl<'a> DynamicsRun<'a> {
    /// Open the run: resolve the brush's textures, and put the tool in the state the
    /// stroke arrives at this range with — resumed from `tool`, or freshly charged.
    fn new(
        r: &'a StrokeRenderer,
        scene: StrokeScene<'a>,
        range: ResolvedRange<'a>,
        dynamics: BrushDynamics,
    ) -> Self {
        let ResolvedRange {
            rec,
            tip,
            consts,
            resume,
            ..
        } = range;
        let prior = resume.prior.map(ToolState::looped);
        // The brush's swept-extent prefix-τ, shared with the fast path, at the group
        // every kernel with a tip binds one.
        let mut region = RegionRun::new(r, scene, &range, &tip.prefix, "stark dynamics stroke");
        let resid = r.color_space.has_resid();
        let scope = &mut region.scope;
        // The unintegrated field under the prefix (the reservoir texels' own exposure).
        let depth = tip.depth.clone();
        // Color dynamics for the brush's own `add` paint — the same field and
        // lookup parameters as the fast path (see `deposit` in dynamics.wesl).
        let noise = r.tips.noise(&rec.brush.color_dynamics(), consts.noise_seed);
        scope.hold(noise.clone());

        let brush = ToolPingPong::take(scope, resid);
        // Cloned, which for a persistent map is a refcount: the run replaces only the
        // tiles its own pieces draw, and the rest ride forward shared with the carry
        // this range resumed from.
        let levels = prior.map_or_else(LoopTiles::new, |t| t.levels.clone());
        let ceiling = CeilingLane::of(&mut region, tip, rec, levels);
        let scope = &mut region.scope;
        let fresh = match prior {
            Some(t) => {
                brush.resume(scope, &t.reservoir);
                t.fresh.clone()
            }
            None => {
                brush.charge(scope, consts, dynamics.charge);
                LoopTiles::new()
            }
        };
        // Each at the format its own `bake_*_w` declaration names (§6.10).
        let mut bake = |at: stark_shaders::Binding, label: &'static str| {
            scope
                .take_run(Key {
                    size: (BAKE_RES, BAKE_RES),
                    format: at.storage_format(),
                    usage: LOOP_USAGE,
                    label,
                })
                .view
        };
        let bake_load = bake(d::BAKE_LOAD_W, "stark dynamics bake load");
        let bake_latm = bake(d::BAKE_LATM_W, "stark dynamics bake latm");
        let bake_rlm = resid.then(|| bake(d::BAKE_RLM_W, "stark dynamics bake rlm"));
        Self {
            region,
            depth,
            noise,
            brush,
            bake_load,
            bake_latm,
            bake_rlm,
            fresh: CarriedLanes::new(fresh_key(), fresh),
            ceiling,
        }
    }

    /// Evolve one region-sized piece of the stroke over `base`: composite the tiles
    /// under `segments` into a region, walk them through the loop, and slice the
    /// result back into fresh CoW tiles. The tool carries on from where the previous
    /// piece left it; the canvas side needs no carrying — it is in `base`, which for a
    /// later piece is what the earlier ones wrote back.
    ///
    /// `fires` is this piece's slice of the range's bleed firings and `first` the
    /// piece's own first segment among the range's, which is what their `after` is
    /// keyed to; `settle` is set only for the piece that ends the stroke (see
    /// `dynamics.wesl::settle`).
    fn draw(
        &mut self,
        base: &TileMap,
        segments: &[Segment],
        first: usize,
        fires: &[BleedFire],
        settle: bool,
    ) -> TileMap {
        // One region-sized piece of the range. Its count against
        // `stroke.dynamics`'s says how often a range needed chunking at all, which
        // is a fact about brush radius and stroke length that nothing else reports.
        crate::timing::span!("stroke.piece");
        let Some(piece) = self.region.open(base, segments, fires) else {
            return base.clone();
        };
        let region_origin = piece.bbox().origin;
        // Composite the canvas under the piece into a 1:1 region — the read half of
        // the loop's cost, and the one that scales with the *area* the stroke covers
        // rather than with its segment count.
        let region = {
            crate::timing::span!("stroke.region");
            self.composite_region(base, &piece)
        };
        // Seed the opacity ceiling's mint budget (§6.2): the composite above laid
        // zeros in the aux's `.yz` lanes, and a resumed tile's running totals are
        // copied back over its block — the `.x` height riding along bit-identical,
        // being the value the write-back sliced into the tile the composite just
        // re-laid. Nothing to seed at the identity ceiling, where the shader never
        // reads the lanes.
        if self.region.consts.capped {
            self.fresh
                .seed(&mut self.region.scope, &piece, &region.channels.aux.tex);
        }
        // The ceiling lane over this piece (§6.2), seeded from its own carry by the
        // same cut, on a stroke whose ceiling the pen drives.
        let lane = self
            .ceiling
            .as_ref()
            .map(|c| c.open(&mut self.region, &piece, segments, piece.bbox()));

        // ---- The dispatch plan, one slot per dispatch, uploaded as one buffer the
        // loop reads through dynamic offsets.
        //
        // It comes **before the scratch it sizes**: the plan measures every rect it
        // will dispatch and the snapshot square is their maximum, so no rect has to be
        // asserted against a bound. The plan, that scratch and the bind groups naming
        // it share one timing row — they are the fixed per-piece setup, and three
        // sub-quantum numbers is more than the browser's clock can resolve.
        //
        // **The block ends the scratch handles' scope, and that is sound** — worth
        // stating in this file, where an early release is the standing hazard
        // (`scratch::SubmitScope`). `take_piece` and `take_piece_buffer` register the
        // *lease* with the scope and hand back refcounted `wgpu` clones, so what drops
        // here is a handle and never a claim on the pool: the leases are released by
        // the flush that submits the commands naming them. The bind groups outlive the
        // block because the loop below is recorded against them, and a `BindGroup`
        // holds its own references to every view in it.
        let (plan, stamps, bind) = {
            crate::timing::span!("stroke.plan");
            let ctx = self.region.plan_ctx(region_origin);
            let plan = dynamics_plan(&ctx, segments, fires, first, settle);
            let under = snapshot_scratch(&mut self.region, plan.dsize);
            // The cell scratch (§6.2), only when some slot actually takes the coarse
            // path — a small or hard tip allocates nothing and binds nothing.
            let cells = plan
                .slots
                .iter()
                .any(|d| matches!(d.step, WetStep::Segment { cells: Some(_), .. }))
                .then(|| cell_scratch(&mut self.region, plan.dsize));
            // The bleed pair's mobility scratch (§6.2), only when some slot is a firing —
            // a brush that does not bleed allocates nothing and binds the 1×1 stand-in.
            let bleed = plan
                .slots
                .iter()
                .any(|d| matches!(d.step, WetStep::Bleed))
                .then(|| bleed_scratch(&mut self.region, plan.dsize));
            let stamps = self.region.upload(&plan.slots);
            let bind = self.bind_piece(
                PieceScratch {
                    region: &region,
                    under: &under,
                    cells: cells.as_ref(),
                    bleed: bleed.as_ref(),
                    lane: lane.as_ref().map(LanePiece::view),
                },
                &stamps,
                Needs::of(&plan.slots),
            );
            (plan, stamps, bind)
        };

        {
            // The stamp loop itself: bake → exchange → deposit per segment, plus the
            // bleed firings and the pen-up settle. **The row to watch** — this is
            // where a per-segment cost lands, and where a change to the dispatch
            // count shows up before it shows up in a total.
            crate::timing::span!("stroke.loop");
            record_loop(
                &mut self.region,
                &mut self.brush,
                &plan.slots,
                plan.dsize,
                &stamps,
                &bind,
                lane.as_ref(),
            );
        }
        // Extract the budget's new totals for the carry (§6.2): a fresh lease
        // per touched tile, cut exactly as the write-back cuts paint — the
        // carried tiles themselves are never written, which is what lets the
        // live tail resume the same frozen totals every pointer move.
        let r = self.region.r;
        if self.region.consts.capped {
            self.fresh
                .extract(r, &mut self.region.scope, &piece, &region.channels.aux.tex);
        }
        // The ceiling lane's new sums, cut the same way and held the same way.
        let lane = lane.map(LanePiece::into_region);
        if let (Some(ceiling), Some(lane)) = (&mut self.ceiling, &lane) {
            ceiling
                .carried
                .extract(r, &mut self.region.scope, &piece, &lane.tex);
        }

        // Slicing the evolved region back into fresh CoW tiles — the write half,
        // paired with `stroke.region`'s read.
        crate::timing::span!("stroke.writeback");
        self.write_back(base, &piece, &region)
    }

    /// The piece's canvas region — a 1:1 copy of what lies under its segments — and
    /// the selection over it.
    ///
    /// Composited from the base tiles of the affected set plus a one-tile ring, so
    /// rewritten tiles' aprons read real neighbour content (§6.4). Each channel at the
    /// format the loop storage-writes it through ([`REGION_WRITES`]), which is also the
    /// format the composite fills it at and the tiles are cut from it in (`kit`).
    fn composite_region(&mut self, base: &TileMap, piece: &Piece) -> Region {
        let r = self.region.r;
        let rect = piece.bbox();
        let (w, h) = (rect.w, rect.h);

        // COPY_SRC because the write-back cuts the tiles straight out of these by
        // texture copy ([`Self::write_back`]); COPY_DST because the opacity
        // ceiling's mint budget is seeded back *into* the aux the same way
        // (§6.2, `LoopCarry::fresh`). Pooled: the composite's clearing load
        // op rewrites every texel, so stale contents never show (`scratch`).
        let region_usage = wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST
            | LOOP_USAGE;
        let channels = Trio::take(
            &mut self.region.scope,
            Life::Piece,
            (w, h),
            region_usage,
            REGION_WRITES,
            [
                "stark dynamics region color",
                "stark dynamics region aux",
                "stark dynamics region resid",
            ],
            r.color_space.has_resid(),
        );
        let halo = piece.halo();
        composite_tiles(
            r,
            &mut self.region.scope,
            base,
            halo,
            rect,
            channels.targets(),
        );
        let selection = self.region.scene.selection;
        let sel_mask = region_selection(r, &mut self.region.scope, selection, halo, rect);

        Region { channels, sel_mask }
    }

    /// Every bind group the loop switches between while recording one piece.
    ///
    /// Each is built from the very entries its layout was ([`desc::Bindings`]), so a
    /// group and its layout cannot disagree about which bindings are present or in what
    /// order. What a group supplies is only the **resources**, and that is one function
    /// of the slot for all of them — see `res` below.
    ///
    /// `ST` binds a single slot-sized window of `stamps` whose dynamic offset
    /// selects the dispatch, so all of these are built once per piece and the loop
    /// varies only the offset. Only the groups `needs` names are built: a group is a
    /// WebGPU object per piece per pointer move.
    fn bind_piece<'p>(
        &'p self,
        over: PieceScratch<'p>,
        stamps: &'p LeasedSlots<Stamp>,
        needs: Needs,
    ) -> PieceBindings {
        let PieceScratch {
            region,
            under,
            cells,
            bleed,
            lane,
        } = over;
        let r = self.region.r;
        let kit = &r.dynamics.wet;
        let device = &r.ctx.device;

        let params = stamps.window();
        let view = |v: &'p wgpu::TextureView| wgpu::BindingResource::TextureView(v);
        let samp = |s: &'p wgpu::Sampler| wgpu::BindingResource::Sampler(s);
        // A residual view is `Some` exactly when the color space has one, and `res` is
        // only ever asked for a residual slot in that case — the shader's own
        // `@if(resid)` gate, which is in the layout. So this states that gate rather
        // than guarding a case the host can reach.
        let opt = |v: Option<&'p wgpu::TextureView>, what: &'static str| {
            wgpu::BindingResource::TextureView(
                v.unwrap_or_else(|| panic!("a residual build must have a {what} residual")),
            )
        };
        // The two scratches a piece takes only when its plan asks for one. The groups
        // whose layouts declare their slots are built on that same condition, and no
        // other group declares one, so a resolver reached for one of them has it.
        let cell = || cells.expect("only a hoisting piece builds a group with a cell slot");
        let mobility = || bleed.expect("only a firing's piece builds the mobility group");

        // **One resolver for the whole loop.** Every `@binding` the two modules declare
        // at group 0 has an index of its own, so which resource fills a slot is a
        // function of the slot — plus `phase`, the reservoir half being read, since the
        // ping-pong's `brush_src_*` is that half and its `brush_dst_*` the other. Each
        // group is then its kernel's layout applied to this, which is why none of them
        // says which bindings it has: the layout already does (§6.10).
        let res = |s: u32, phase: usize| -> wgpu::BindingResource<'p> {
            match s {
                sb::ST => params.clone(),
                sb::SEL_MASK => view(&region.sel_mask),
                sb::REGION_COLOR_W => view(&region.channels.color.view),
                sb::REGION_AUX_W => view(&region.channels.aux.view),
                sb::REGION_RESID_W => opt(region.channels.resid_view(), "region"),
                b::REGION_COLOR => view(&region.channels.color.view),
                b::REGION_AUX => view(&region.channels.aux.view),
                b::REGION_RESID => opt(region.channels.resid_view(), "region"),
                b::UNDER_COLOR => view(&under.color.view),
                b::UNDER_AUX => view(&under.aux.view),
                b::UNDER_RESID => opt(under.resid_view(), "snapshot"),
                b::UNDER_COLOR_W => view(&under.color.view),
                b::UNDER_AUX_W => view(&under.aux.view),
                b::UNDER_RESID_W => opt(under.resid_view(), "snapshot"),
                b::BAKE_LOAD => view(&self.bake_load),
                b::BAKE_LATM => view(&self.bake_latm),
                b::BAKE_RLM => opt(self.bake_rlm.as_ref(), "bake"),
                b::BAKE_LOAD_W => view(&self.bake_load),
                b::BAKE_LATM_W => view(&self.bake_latm),
                b::BAKE_RLM_W => opt(self.bake_rlm.as_ref(), "bake"),
                // The reservoir ping-pong: read `phase`, write the other half. The
                // residual turns on the same phase as the color, or the tool's two
                // halves would drift apart.
                b::BRUSH_SRC_COLOR => view(&self.brush.src(phase).color.view),
                b::BRUSH_SRC_AUX => view(&self.brush.src(phase).aux.view),
                b::BRUSH_SRC_RESID => opt(self.brush.src(phase).resid_view(), "reservoir"),
                b::BRUSH_DST_COLOR_W => view(&self.brush.dst(phase).color.view),
                b::BRUSH_DST_AUX_W => view(&self.brush.dst(phase).aux.view),
                b::BRUSH_DST_RESID_W => opt(self.brush.dst(phase).resid_view(), "reservoir"),
                b::CELL_TOOL => view(&cell().tool),
                b::CELL_LAT => view(&cell().lat),
                b::CELL_RES => opt(cell().res.as_ref(), "cell"),
                b::CELL_TOOL_W => view(&cell().tool),
                b::CELL_LAT_W => view(&cell().lat),
                b::CELL_RES_W => opt(cell().res.as_ref(), "cell"),
                // The real scratch on a piece that fires, the kit's 1×1 otherwise — a
                // painting segment carries `lambda_bleed = 0` and never reads it. The
                // mobility pass writes it, so its own slot takes no stand-in.
                b::BLEED_W => view(bleed.unwrap_or(&kit.bleed_placeholder)),
                b::BLEED_W_W => view(mobility()),
                // The ceiling lane (§6.2), by the stand-in argument again: a stroke
                // whose ceiling the pen does not drive carries `ceiling_lane = 0`.
                b::REGION_LEVELS => view(lane.unwrap_or(&kit.levels_placeholder)),
                b::SUBSTRATE_TEX => view(&self.region.scene.substrate.view),
                b::DEPTH_TEX => view(&self.depth),
                b::SAMP => samp(&kit.sampler),
                b::DYN_NOISE_TEX => view(self.noise.view()),
                b::DYN_NOISE_SAMP => samp(&r.tips.noise_sampler),
                other => unreachable!("the stamp loop declares no group-0 binding {other}"),
            }
        };
        // The phase a group that never names the reservoir is built at: either, since
        // its layout declares no `brush_*` slot for `res` to be asked about.
        const EITHER: usize = 0;

        let snapshot = needs.snapshot.then(|| {
            kit.snapshot
                .group(device, "stark dynamics snapshot bg", |s| res(s, EITHER))
        });
        // `exchange` comes in two flavours for the reservoir ping-pong: each reads one
        // half and writes the other.
        let exchange = needs.exchange.then(|| {
            std::array::from_fn(|i| {
                kit.exchange
                    .group(device, "stark dynamics exchange bg", |s| res(s, i))
            })
        });
        // One bake bind group per reservoir phase; the deposit reads only the baked
        // result, so it never touches the ping-pong.
        let bake = needs.bake.then(|| {
            std::array::from_fn(|i| {
                kit.bake
                    .group(device, "stark dynamics bake bg", |s| res(s, i))
            })
        });
        // The mobility pass, on a piece that has one to run.
        let bleed_weight = bleed.map(|_| {
            kit.bleed_weight
                .group(device, "stark dynamics bleed weight bg", |s| res(s, EITHER))
        });
        let deposit = needs.deposit.then(|| {
            kit.deposit
                .group(device, "stark dynamics deposit bg", |s| res(s, EITHER))
        });
        // The pen-up, which reads the reservoir only through its own `bake` — so unlike
        // `exchange` it needs no bind group per ping-pong half; the bake's does that.
        let settle = needs.settle.then(|| {
            kit.settle
                .group(device, "stark dynamics settle bg", |s| res(s, EITHER))
        });
        // The coarse pair's groups (§6.2), only for a piece whose plan hoists at all.
        let coarse = cells.map(|_| CoarseBindings {
            hoist: kit
                .hoist
                .group(device, "stark dynamics cell hoist bg", |s| res(s, EITHER)),
            deposit: kit
                .deposit_coarse
                .group(device, "stark dynamics deposit coarse bg", |s| {
                    res(s, EITHER)
                }),
        });
        PieceBindings {
            snapshot,
            exchange,
            bake,
            deposit,
            bleed_weight,
            coarse,
            settle,
        }
    }

    /// Slice each affected tile's full `TILE_TEX` block out of the shared region into
    /// a fresh CoW tile → aprons stay bit-identical to neighbour interiors (§6.4), and
    /// the wide region aux narrows to the persistent one (height). See
    /// [`slice_region`], which does the work.
    fn write_back(&mut self, base: &TileMap, piece: &Piece, region: &Region) -> TileMap {
        slice_region(
            self.region.r,
            &mut self.region.scope,
            self.region.scene.pool,
            base,
            piece,
            RegionChannels {
                color_tex: &region.channels.color.tex,
                aux: &region.channels.aux.view,
                resid_tex: region.channels.resid_tex(),
            },
        )
    }

    /// Remember the tool for the range that resumes after this one
    /// ([`ToolPingPong::capture`]).
    fn capture_tool(&mut self) -> ToolState {
        let reservoir = self.brush.capture(self.region.r, &mut self.region.scope);
        ToolState(Carried::Loop(Box::new(LoopCarry {
            reservoir,
            // The mint budget rides as the leases the last piece extracted —
            // already copies of the run's own state, so nothing more to copy
            // out here. Empty at the identity opacity.
            fresh: std::mem::take(&mut self.fresh.tiles),
            levels: self
                .ceiling
                .as_mut()
                .map_or_else(LoopTiles::new, |c| std::mem::take(&mut c.carried.tiles)),
        })))
    }

    /// Close the run: capture the tool for the range that resumes after this one when
    /// one will, then hand the region machinery what is still owed a submit.
    ///
    /// **The mint budget and the ceiling lane go to the scope rather than drop**:
    /// whatever `capture_tool` did not take is still named by the last piece's extract
    /// copies, which the submit inside [`RegionRun::finish`] is about to make.
    fn finish(mut self, end_dist: f32, capture: bool) -> StrokeCarry {
        let tool = capture.then(|| self.capture_tool());
        let held = (
            std::mem::take(&mut self.fresh.tiles),
            self.ceiling
                .as_mut()
                .map(|c| std::mem::take(&mut c.carried.tiles)),
        );
        self.region.finish(end_dist, tool, held)
    }
}

/// The extent snapshot scratch: the copy that gives `deposit` and `settle`
/// something to read while they storage-write the region.
///
/// `size` is the plan's own [`dsize`](super::plan::Plan::dsize) — the
/// largest rect the piece will actually dispatch, rounded to the pool's quantum —
/// so it is a maximum over the rects rather than a bound they must respect, and
/// nothing needs asserting or clamping.
fn snapshot_scratch(region: &mut RegionRun<'_>, size: u32) -> Trio {
    let resid = region.r.color_space.has_resid();
    Trio::take(
        &mut region.scope,
        Life::Piece,
        (size, size),
        LOOP_USAGE,
        [d::UNDER_COLOR_W, d::UNDER_AUX_W, d::UNDER_RESID_W],
        [
            "stark dynamics under color",
            "stark dynamics under aux",
            "stark dynamics under resid",
        ],
        resid,
    )
}

/// The bleed pair's mobility scratch (§6.2): where `bleed_weight` leaves the weight
/// the ladder reads back thirty-six times a texel.
///
/// Sized to the **snapshot square**, not a firing's rect, and that is what makes
/// the read safe: the ladder clamps a tap into the scratch's own extent, so a
/// texel outside the dispatched rect can still be read, and one this pass had not
/// written would hold the previous firing's answer.
fn bleed_scratch(region: &mut RegionRun<'_>, dsize: u32) -> wgpu::TextureView {
    region
        .scope
        .take_piece(Key {
            size: (dsize, dsize),
            format: d::BLEED_W_W.storage_format(),
            usage: LOOP_USAGE,
            label: "stark dynamics bleed w",
        })
        .view
}

/// The extent-cell scratch (§6.2): where `cell_hoist` leaves the per-cell means
/// for `deposit_coarse` to read back. [`cell_scratch_size`] of the piece's own
/// `dsize`, which `cell_geometry` asserted every slot's hoist grid against, and
/// leased on the piece like the region.
fn cell_scratch(region: &mut RegionRun<'_>, dsize: u32) -> Cells {
    let resid = region.r.color_space.has_resid();
    let size = cell_scratch_size(dsize);
    // Each at the format its own `cell_*_w` declaration names (§6.10).
    let mut tex = |at: stark_shaders::Binding, label: &'static str| {
        region
            .scope
            .take_piece(Key {
                size: (size, size),
                format: at.storage_format(),
                usage: LOOP_USAGE,
                label,
            })
            .view
    };
    Cells {
        tool: tex(d::CELL_TOOL_W, "stark dynamics cell tool"),
        lat: tex(d::CELL_LAT_W, "stark dynamics cell lat"),
        res: resid.then(|| tex(d::CELL_RES_W, "stark dynamics cell res")),
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
/// The run's region and ping-pong are taken apart rather than passed whole, because
/// `lane` is the run's own ceiling lane borrowed for the piece. The ping-pong outlives
/// the pass: it names the half holding the tool's state, so after the last dispatch it
/// names the state this piece ends in — which is what the next piece, or the next
/// range, resumes from.
fn record_loop(
    region: &mut RegionRun<'_>,
    brush: &mut ToolPingPong,
    plan: &[LoopDispatch<WetStep>],
    dsize: u32,
    stamps: &LeasedSlots<Stamp>,
    bind: &PieceBindings,
    lane: Option<&LanePiece<'_>>,
) {
    let kit = &region.r.dynamics.wet;
    // The reservoir half of `exchange`, which the extent half is laid beside on x:
    // [`BRUSH_RES`]² whatever the segment does.
    let reservoir = kit.exchange.groups((BRUSH_RES, BRUSH_RES));
    let mut pass = region.loop_pass(LOOP_PASS);
    for (i, d) in plan.iter().enumerate() {
        // Asked of the lease the slots were staged into, so the stride they were
        // *written* at and the stride they are *bound* at are one expression rather
        // than two that happen to agree.
        let off = stamps.offset(i);
        match d.step {
            // The tool plays no part in the lateral flux, so `cur` — the reservoir
            // ping-pong — stays exactly where the previous segment left it. The
            // standalone snapshot kernel rather than the exchange's tail, because
            // there is no exchange dispatch to ride in.
            WetStep::Bleed => {
                kit.snapshot
                    .dispatch(&mut pass, bind.snapshot(), off, d.extent);
                // The pair's mobility, once per texel, for the ladder to read back
                // (§6.2). Over the **whole snapshot square** rather than the
                // firing's rect: the ladder clamps a tap into the scratch's extent,
                // so every texel it can reach has to have been written this firing.
                kit.bleed_weight
                    .dispatch(&mut pass, bind.bleed_weight(), off, (dsize, dsize));
                kit.deposit
                    .dispatch(&mut pass, bind.deposit(), off, d.extent);
            }
            WetStep::Segment { index, cells } => {
                let cur = brush.cur();
                // Bake this segment's swept reservoir prefix first — it folds in the
                // tip's current orientation as well as the reservoir state.
                kit.bake
                    .dispatch(&mut pass, bind.bake(cur), off, (BAKE_RES, BAKE_RES));
                // Then the tool's own side of this segment's transfer, off the
                // region as the segment found it. Reads `cur` and writes the other
                // half, so the next segment's bake sees a tool that has travelled
                // and reloaded.
                //
                // The extent `snapshot` rides in the tail of this same grid: it
                // depends on nothing the exchange writes, so a barrier between them
                // would buy no ordering. Hence the widened x — reservoir groups
                // first, extent groups after — and a y tall enough for the taller
                // of the two (`dynamics.wesl::exchange`).
                let tail = kit.exchange.groups(d.extent);
                kit.exchange.dispatch_groups(
                    &mut pass,
                    bind.exchange(cur),
                    off,
                    (reservoir.0 + tail.0, reservoir.1.max(tail.1)),
                );
                // The canvas's half: exact per texel, or — where the tip's
                // shoulder allows (`extent_cell`) — hoisted once per cell and
                // applied over the same texel grid. The hoist reads the bake this
                // segment just wrote and nothing the exchange writes, so the extra
                // serialized dispatch lands only on the wide tips that are
                // texel-bound rather than dispatch-bound.
                match cells.map(|cl| (cl, bind.coarse())) {
                    Some((cl, cb)) => {
                        kit.hoist.dispatch(&mut pass, &cb.hoist, off, cl);
                        kit.deposit_coarse
                            .dispatch(&mut pass, &cb.deposit, off, d.extent);
                    }
                    None => kit
                        .deposit
                        .dispatch(&mut pass, bind.deposit(), off, d.extent),
                }
                // The ceiling lane (§6.2): this segment's sums drawn over the
                // region *after* its deposit, so the deposit read the lane as
                // the segments before it left it and the next reads it with
                // this one in. The draw is a render pass, so the compute pass
                // ends around it and resumes after.
                if let Some(l) = lane {
                    drop(pass);
                    l.draw(region, index);
                    pass = region.loop_pass(LOOP_PASS);
                }
                brush.advance();
            }
            // The pen-up: snapshot the final extent, bake the standing tip's
            // remaining-pass delivery off the reservoir the last segment left
            // (`cur` still names it — the slot's zero travel switches the bake onto
            // the settle's weighted integral), then settle the transfer the stroke
            // stopped in the middle of. The tool is not written back, so the
            // ping-pong is left alone here too.
            WetStep::Settle => {
                kit.snapshot
                    .dispatch(&mut pass, bind.snapshot(), off, d.extent);
                kit.bake
                    .dispatch(&mut pass, bind.bake(brush.cur()), off, (BAKE_RES, BAKE_RES));
                kit.settle.dispatch(&mut pass, bind.settle(), off, d.extent);
            }
        }
    }
}

/// One piece's canvas region: a 1:1 copy of the canvas under its segments, which the
/// loop evolves in place before [`DynamicsRun::write_back`] slices it into tiles.
struct Region {
    /// The evolving paint itself (§6.7). Each channel carries its texture beside its
    /// view because the loop does both to it: a pass binds the view, while the
    /// write-back cuts tiles out of the colour and residual and the mint budget is
    /// seeded into and extracted out of the aux — and a copy is addressed by texture.
    channels: Trio,
    /// The selection over the region (§6.8) — its own gathered mask, or the 1×1
    /// constant that stands in for an unrestricted selection.
    sel_mask: wgpu::TextureView,
}

/// What one piece's bind groups are built over: the region and its selection, the
/// extent snapshot, and the three a piece takes only when its plan asks for one — the
/// cells, the bleed pair's mobility and the ceiling lane.
#[derive(Clone, Copy)]
struct PieceScratch<'p> {
    region: &'p Region,
    under: &'p Trio,
    cells: Option<&'p Cells>,
    bleed: Option<&'p wgpu::TextureView>,
    lane: Option<&'p wgpu::TextureView>,
}

/// The extent-cell scratch (§6.2): the coarse deposit's per-cell means, sized by
/// [`cell_scratch_size`] of the piece's snapshot square so every hoist grid a slot
/// can ask for fits by construction.
struct Cells {
    tool: wgpu::TextureView,
    lat: wgpu::TextureView,
    /// The residual's cell mean, allocated exactly when the color space has one.
    res: Option<wgpu::TextureView>,
}

/// Which of a piece's bind groups its plan will bind — one flag per group
/// [`record_loop`] reaches for, read off the plan's steps exactly as its
/// arms consume them. `bleed_weight` and `coarse` are gated the same way, by the
/// scratch they need.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Needs {
    snapshot: bool,
    exchange: bool,
    bake: bool,
    deposit: bool,
    settle: bool,
}

impl Needs {
    const NONE: Self = Needs {
        snapshot: false,
        exchange: false,
        bake: false,
        deposit: false,
        settle: false,
    };

    fn of(plan: &[LoopDispatch<WetStep>]) -> Self {
        let mut n = Self::NONE;
        for d in plan {
            match d.step {
                WetStep::Segment { cells, .. } => {
                    n.bake = true;
                    n.exchange = true;
                    // A coarse segment deposits through its own cell group.
                    n.deposit |= cells.is_none();
                }
                WetStep::Bleed => {
                    n.snapshot = true;
                    n.deposit = true;
                }
                WetStep::Settle => {
                    n.snapshot = true;
                    n.bake = true;
                    n.settle = true;
                }
            }
        }
        n
    }
}

/// The bind groups one piece's dispatches switch between. Built once per piece
/// because every dispatch varies only the stamp uniform's dynamic offset; the
/// reservoir ping-pong is the one thing that needs a pair.
///
/// The five without a doc of their own are `Some` exactly when [`Needs::of`] the
/// piece's plan said so, and the accessors state that contract: the arm that reads
/// a group is the arm that asked for it.
struct PieceBindings {
    snapshot: Option<wgpu::BindGroup>,
    exchange: Option<[wgpu::BindGroup; 2]>,
    bake: Option<[wgpu::BindGroup; 2]>,
    deposit: Option<wgpu::BindGroup>,
    /// The mobility pass's group — `Some` exactly when the piece allocated a bleed
    /// scratch, i.e. when some slot is a firing.
    bleed_weight: Option<wgpu::BindGroup>,
    /// The coarse pair's groups — `Some` exactly when the piece allocated a cell
    /// scratch, i.e. when some slot's [`extent_cell`](super::plan) beat 1.
    coarse: Option<CoarseBindings>,
    settle: Option<wgpu::BindGroup>,
}

impl PieceBindings {
    fn snapshot(&self) -> &wgpu::BindGroup {
        self.snapshot
            .as_ref()
            .expect("a firing or a settle asked for a snapshot group")
    }

    fn exchange(&self, cur: usize) -> &wgpu::BindGroup {
        let pair = self.exchange.as_ref();
        &pair.expect("a segment asked for the exchange pair")[cur]
    }

    fn bake(&self, cur: usize) -> &wgpu::BindGroup {
        let pair = self.bake.as_ref();
        &pair.expect("a segment or a settle asked for the bake pair")[cur]
    }

    fn deposit(&self) -> &wgpu::BindGroup {
        self.deposit
            .as_ref()
            .expect("a firing or an exact segment asked for a deposit group")
    }

    fn bleed_weight(&self) -> &wgpu::BindGroup {
        self.bleed_weight
            .as_ref()
            .expect("a firing's piece allocated a bleed scratch")
    }

    fn coarse(&self) -> &CoarseBindings {
        self.coarse.as_ref().expect("a coarse slot binds cells")
    }

    fn settle(&self) -> &wgpu::BindGroup {
        self.settle
            .as_ref()
            .expect("a settle slot asked for a settle group")
    }
}

/// The two bind groups a coarse slot dispatches with (§6.2).
struct CoarseBindings {
    hoist: wgpu::BindGroup,
    deposit: wgpu::BindGroup,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each wet step asks for exactly the groups its arm of
    /// [`record_loop`] binds — read the arms, then this.
    #[test]
    fn each_wet_step_asks_for_the_groups_its_arm_binds() {
        let of = |step: WetStep| {
            Needs::of(&[LoopDispatch {
                slot: bytemuck::Zeroable::zeroed(),
                extent: (1, 1),
                step,
            }])
        };
        let segment = |cells| WetStep::Segment { index: 0, cells };
        assert_eq!(
            of(segment(None)),
            Needs {
                exchange: true,
                bake: true,
                deposit: true,
                ..Needs::NONE
            }
        );
        // A coarse segment deposits through its cell group, not the exact one.
        assert_eq!(
            of(segment(Some((1, 1)))),
            Needs {
                exchange: true,
                bake: true,
                ..Needs::NONE
            }
        );
        assert_eq!(
            of(WetStep::Bleed),
            Needs {
                snapshot: true,
                deposit: true,
                ..Needs::NONE
            }
        );
        assert_eq!(
            of(WetStep::Settle),
            Needs {
                snapshot: true,
                bake: true,
                settle: true,
                ..Needs::NONE
            }
        );
    }
}
