//! Recording one dynamics render: regions, the reservoir ping-pong, the compute pass
//! the plan's slots are dispatched through, and the write-back (§6.2).
//!
//! This is the half that owns GPU state. What it is *asked* to record comes from
//! [`plan`](super::plan); what it records *with* comes from [`kit`](super::kit).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::gpu::channels::Targets;
use crate::gpu::composite::view_uniform;
use crate::gpu::desc;
use crate::gpu::tile::{AllocSource, TileMap};
use stark_model::document::StrokeRecord;
use stark_model::geom::{TileCoord, Vec2};

/// The paint effect the loop is drawing — its own by construction:
use stark_shaders::mirror::composite::binding as cb;

use super::super::incremental::{Carried, LoopCarry, Reservoir, Resume};
use super::super::region::{RegionRect, chunk_segments, cover};
use super::super::segments::{BleedFire, Segment, generate_segments_in};
use super::super::swept::{segment_instance, tile_xform, xform_group};
use super::super::{StrokeCarry, StrokeRenderer, StrokeScene, StrokeSpans, ToolState};
use super::bleed::bleed_fires;
use super::plan::{
    LoopDispatch, PlanCtx, SLOT, STAMP_STRIDE, SlotKind, cell_scratch_size, dynamics_plan,
    liquify_plan,
};
use super::slots;
use crate::gpu::scratch::{BufKey, Kept, Key, SubmitScope};
use stark_shaders::mirror::stamp::SegmentInstance;
use stark_shaders::mirror::stamp_common::SWEEP_VERTS;
/// Resolution (texels per side) of the stamp loop's tool reservoir
/// (§6.2). Brush-local, so carried color detail is ~radius/32 canvas px — plenty
/// for smeared paint, and small enough that the per-stamp reservoir update is
/// nearly free.
const BRUSH_RES: u32 = 64;
// Resolution of the per-segment **swept prefix** of the reservoir
// (`dynamics.wesl::bake`), and the workgroup width of the scan that builds it —
// generated from the shader, which is the side that decides it (§6.10). A mismatch
// scanned the wrong width and rendered subtly wrong without crashing.
use stark_shaders::mirror::dynamics::{BAKE_RES, TILE_WG};
// The `@binding` indices, generated from `dynamics.wesl`'s own declarations (§6.10):
// the bind groups here and the layouts in [`kit`](super::kit) both name the
// shader's numbering rather than keeping copies of it.
use stark_shaders::mirror::dynamics::binding as b;
// The storage-texture *formats*, taken from the same declarations the indices come
// from (§6.10): a slot names its own format, so the host asks rather than repeats.
use stark_shaders::mirror::dynamics::decl as d;

// The region composite runs `composite.wesl`, so it draws that shader's own
// per-instance record (§6.10) rather than a second `#[repr(C)]` copy of it declared
// here under a different name.
use crate::gpu::stroke::tips::ResolvedTip;

/// The brush as [`StrokeRenderer::render_range`](crate::gpu::stroke::StrokeRenderer)'s
/// gate resolved it, for the paths that need both halves.
///
/// The tip every path binds, and the tool that chose *this* path. Both were
/// re-derived inside the run behind an `expect` naming the gate that had already
/// produced them — one re-resolving the tip through the LRU under its lock, the other
/// unwrapping the effect the plan had matched on to get here. Carried together
/// because they arrive together and are exactly what "the gate said the loop runs"
/// means.
#[derive(Clone, Copy)]
pub(in crate::gpu::stroke) struct LoopBrush<'a> {
    pub(in crate::gpu::stroke) tip: &'a ResolvedTip,
    pub(in crate::gpu::stroke) tool: LoopTool,
}

/// Which tool the region loop is running — the wet exchange with the axes that
/// chose it (§6.2), or the liquify warp (§6.13), which shares the region, the
/// piece chunking and the write-back and dispatches a plan of warp slots instead.
/// A tag on [`LoopBrush`] rather than a second entry point, because everything
/// that differs between the two is a per-slot question the plan already answers
/// ([`SlotKind`]); what is decided here is only which plan gets built and whether
/// there is a tool to seed, settle and carry at all.
#[derive(Clone, Copy)]
pub(in crate::gpu::stroke) enum LoopTool {
    Wet(stark_model::document::BrushDynamics),
    Liquify,
}
use crate::gpu::uniforms::UniformSlots;
use stark_shaders::mirror::composite::Instance as TileInstance;

/// The mint-budget tile's pool key (`LoopCarry::fresh`): copies both ways, nothing
/// else — no pass ever binds one, they exist purely to ferry the region aux's `.yz`
/// lanes across pieces. The whole aux texel rides along (the `.x` height is the very
/// value the write-back sliced into the base tile, so seeding it back over the freshly
/// composited region is bit-identical), because a copy cannot address two lanes of a
/// texel and a pass that could would cost more than the lanes.
///
/// A whole `TILE_TEX` block — interior plus apron — cut from and seeded into the region
/// aux exactly as the write-back cuts paint tiles, so the two address the region with
/// one rule. Adjacent blocks overlap by the apron; the overlaps are cut from one
/// region, so seeding them back writes identical texels twice. The copies take their
/// extent from this key ([`Key::extent`]).
const fn fresh_key() -> Key {
    Key::tile(
        wgpu::TextureFormat::Rgba16Float,
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

/// The brush as the swept sweep binds it (§6.2) — for the per-segment draw of
/// the ceiling lane over the region, which runs the sweep's own pipeline over
/// the sweep's own layouts.
struct LaneBinds {
    prefix: wgpu::BindGroup,
    noise: wgpu::BindGroup,
}

/// One piece's draw state for the ceiling lane (§6.2): the region-sized lane the
/// deposits read, the piece's segments instanced as the sweep draws them, and
/// the three groups `stamp.wesl::fs_levels` is drawn with.
struct LevelsPass {
    view: wgpu::TextureView,
    xform: wgpu::BindGroup,
    prefix: wgpu::BindGroup,
    noise: wgpu::BindGroup,
    instances: wgpu::Buffer,
}

/// Workgroup counts for the reservoir half of `exchange`, which is dispatched over
/// these *plus* the slot's extent groups on x, since the snapshot shares its
/// grid.
///
/// A constant, not per-dispatch data: the reservoir is [`BRUSH_RES`]² whatever the
/// segment does, and the two slot kinds that do not run an exchange never read it.
const RESERVOIR_GROUPS: (u32, u32) = (BRUSH_RES.div_ceil(TILE_WG), BRUSH_RES.div_ceil(TILE_WG));

impl StrokeRenderer {
    /// Render `spans` of a paint-manipulating stroke via the **sequential
    /// swept-exchange loop** (§6.2): composite the base under it into a 1:1
    /// region, then walk it *in order* on the GPU — the canvas-side exchange swept per
    /// flattened segment through the prefix-τ integral (the same definite-integral
    /// extent as the plain deposit), the 2-D tool reservoir taking the complement
    /// of it over the same segment — and slice the evolved region back into fresh CoW
    /// tiles.
    ///
    /// A region is a 1:1 copy of the canvas under the stroke, so the range is drawn in
    /// as many region-sized **pieces** as it takes ([`chunk_segments`]) rather than in
    /// one: the loop is sequential, so pieces run back to back over the same segments
    /// in the same order, each compositing what the last wrote back. Length therefore
    /// costs the stroke extra pieces, not correctness. Degrading past
    /// [`MAX_REGION_DIM`](super::super::budget::MAX_REGION_DIM) to the plain swept deposit instead
    /// would cost correctness, since that path cannot manipulate paint at all.
    ///
    /// The loop starts from `tool` rather than from a fresh tip when one is given, and
    /// hands back the state it ends in whenever a further range remains to be drawn,
    /// so a live stroke redraws only its tail (see [`ToolState`]). `tol` comes from
    /// [`dynamics_setup`](super::dynamics_setup), which has already decided — from the brush — that this
    /// stroke runs the loop at all.
    pub(in crate::gpu::stroke) fn render_dynamic(
        &self,
        scene: StrokeScene<'_>,
        rec: &StrokeRecord,
        spans: StrokeSpans,
        resume: Resume<'_>,
        tol: crate::path::FlattenTolerance,
        brush: LoopBrush<'_>,
    ) -> (TileMap, StrokeCarry) {
        // The sequential stamp loop end to end. Everything below it — `stroke.piece`
        // and the four phases inside that — partitions this row, which is what makes
        // the table a phase split rather than a pile of unrelated timings: the shares
        // have a denominator, and a phase that is missing from the sum is a phase
        // nobody has instrumented yet.
        crate::timing::span!("stroke.dynamics");
        let capture = resume.capture;
        // Fitting outweighs flattening by ~350:1 on the CPU side, which is a claim
        // about *this* call against `input.fit` — and one that has to stay checkable,
        // because the flattener is the thing that looks worth optimizing and is not.
        let (segments, end_dist) = {
            crate::timing::span!("stroke.segments");
            generate_segments_in(rec, tol, spans)
        };
        // A range with no geometry runs no dispatches, so it leaves the brush exactly
        // as it found it. Handing back `None` says "unchanged" — the caller keeps the
        // state it passed in rather than paying for a copy of it.
        if segments.is_empty() {
            return (scene.base.clone(), StrokeCarry::unchanged(end_dist));
        }

        let mut run = DynamicsRun::new(self, scene, rec, tol, resume.prior, brush);
        let mut map = scene.base.clone();
        // The bleed cadence's firings for the whole range (§6.2), computed once and
        // sliced per piece rather than re-derived inside each: the chunker must
        // measure every piece **with its windows** — a window can reach back a
        // quantum before the segment it fires after, which for a piece's first
        // segment is substrate no segment box covers ([`chunk_segments`]).
        // A liquify stroke has no such axis, so it fires nothing (§6.13).
        let (fires, bleed_capped) = match brush.tool {
            LoopTool::Wet(d) => bleed_fires(d.bleed, &segments),
            LoopTool::Liquify => (Vec::new(), false),
        };
        // The third budget, said out loud like the other two. A segment wanting more
        // than `MAX_BLEED_FIRES_PER_SEGMENT` firings gets the cap, and past it the axis
        // "quietly under-delivers" — `bleed.rs`'s own words. The artist sees a bleed
        // knob that stops meaning what it says at small modulated radii, with nothing
        // in the log; the segment-shortening cap next door has been reported since it
        // existed, on the same once-per-stroke gate and for the same reason.
        if bleed_capped && self.complain_once(rec.seed) {
            tracing::warn!(
                radius = rec.brush.size,
                cap = super::bleed::MAX_BLEED_FIRES_PER_SEGMENT,
                "a segment wants more bleed firings than it may have: the bleed axis                  under-delivers on this stroke",
            );
        }
        // The pen-up settle (§6.2) belongs to the range that reaches the *stroke's* end,
        // and within it to the last piece — which is the same condition that says there
        // is no reservoir worth keeping. A range that stops short hands its tool on
        // instead, so nothing is stranded for the settle to hand back, and a live tail
        // computes the same settle its commit will.
        // Asked of the iterator rather than counted against `len() - 1`, which
        // underflowed on an empty cut. That cut cannot be empty — `chunk_segments`
        // pushes a run whenever it is given a segment, and the empty range returned
        // above — but the proof was two functions away from the subtraction relying on
        // it, and "is there another piece after this one" is the question anyway.
        // Only the wet tool has a pen-up to settle: the warp applies each
        // segment's follow whole as the tip passes, so a break of contact strands
        // nothing — the bleed axis's argument, and §6.13's statement of it.
        let settles = matches!(brush.tool, LoopTool::Wet(_));
        let mut pieces = chunk_segments(&segments, &fires).into_iter().peekable();
        while let Some(piece) = pieces.next() {
            let is_last = pieces.peek().is_none();
            // This piece's firings, re-keyed to its slice: `after` indexes `segments`,
            // and the piece's plan walks its own slice. The order — what
            // `dynamics_plan` asserts — survives a slice-and-shift untouched.
            let lo = fires.partition_point(|f| f.after < piece.start);
            let hi = fires.partition_point(|f| f.after < piece.end);
            let piece_fires: Vec<BleedFire> = fires[lo..hi]
                .iter()
                .map(|f| BleedFire {
                    after: f.after - piece.start,
                    ..*f
                })
                .collect();
            map = run.draw(
                &map,
                &segments[piece],
                &piece_fires,
                settles && !capture && is_last,
            );
        }
        // A liquify range hands nothing on: the canvas *is* the state (§6.13),
        // and a later range recomposites it from the tiles this one wrote back.
        let tool_out = match brush.tool {
            LoopTool::Wet(_) => capture.then(|| run.capture_tool()),
            LoopTool::Liquify => None,
        };
        // The pieces partition the segments, so the union of what each one enumerated
        // for itself *is* what the whole range touched — accumulated as they went
        // rather than walked a second time over every (segment, tile) pair, which is
        // the very cost cutting a stroke into region-sized pieces exists to keep off a
        // long one (`region::chunk_segments`).
        let dirty = std::mem::take(&mut run.dirty).into_iter().collect();
        {
            // `queue.submit` plus the scratch releases that may only follow it
            // (`SubmitScope`). Its own row because it is the one phase here that is
            // not encoding: on the web `submit` hands a command buffer across the
            // wasm boundary, and if that ever became the wall it would be invisible
            // folded into the piece that recorded it.
            crate::timing::span!("stroke.submit");
            run.submit();
        }
        (
            map,
            StrokeCarry {
                dist: end_dist,
                tool: tool_out,
                dirty,
                deferred: false,
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
    /// The budget the range was flattened at ([`dynamics_setup`](super::dynamics_setup)),
    /// carried so a plan
    /// can re-cut a piece of the record the same way — see [`PlanCtx::tol`].
    tol: crate::path::FlattenTolerance,
    scene: StrokeScene<'a>,
    /// The encoder this run records into, and every resource whose release must
    /// trail a submit — the run-scoped leases (reservoir ping-pong, bake pair) and the
    /// piece-scoped ones (region, snapshot, cells, the selection mask, and every
    /// buffer). The scope releases each tier only in the call that submits the
    /// commands naming it ([`SubmitScope`]), so the ordering is the type's rather than
    /// three fields and a flag holding it by convention.
    scope: SubmitScope,
    /// Every tile the run has rewritten, accumulated as each piece enumerates its own.
    /// The pieces partition the range's segments, so this ends up the set a second
    /// walk over the whole stroke would have built — for none of the cost.
    dirty: BTreeSet<TileCoord>,
    /// Everything both render paths read off the record and the scene, resolved once
    /// (see [`StrokeConstants`](super::super::StrokeConstants)).
    consts: super::super::StrokeConstants,
    /// Functions of the brush alone, so shared by every piece: the swept-extent
    /// prefix-τ bind group (group 1 of `bake`/`deposit` — the same texture the swept
    /// fast path samples), the plain coverage mask the reservoir texels weight by,
    /// and the color-dynamics field the `add` paint is jittered against.
    prefix_bg: wgpu::BindGroup,
    cov: wgpu::TextureView,
    noise: super::super::tips::NoiseLease,
    /// The tool reservoir ping-pong, and which half currently holds the tool.
    brush_color_tex: [wgpu::Texture; 2],
    brush_aux_tex: [wgpu::Texture; 2],
    brush_color: [wgpu::TextureView; 2],
    brush_aux: [wgpu::TextureView; 2],
    /// The reservoir's **residual** half (§6.7), ping-ponged in step with the color
    /// above — the tip carries the rest of the color it picked up, or a pigment
    /// space's black would come back off the tool as the grey its concentrations
    /// alone describe. `None` in a space with no residual, which allocates nothing.
    brush_resid_tex: Option<[wgpu::Texture; 2]>,
    brush_resid: Option<[wgpu::TextureView; 2]>,
    cur: usize,
    /// The segment's swept reservoir prefixes (fp32, so the per-fragment difference
    /// keeps its precision — `bake_load_w` is declared `rgba32float`). Rebuilt per
    /// segment, so a single
    /// pair serves the whole stroke: nothing reads the last segment's bake.
    bake_load: wgpu::TextureView,
    bake_latm: wgpu::TextureView,
    /// The residual's own swept prefix, `∫res·m·dτ` — same format, same differencing,
    /// and recovered against the very denominator `bake_latm` is.
    bake_rlm: Option<wgpu::TextureView>,
    /// The opacity ceiling's mint-budget tiles ([`LoopCarry::fresh`]), evolving
    /// with the run: resumed entries first, each piece replacing the tiles it
    /// touched with the leases it extracted. Untouched — and empty — at the
    /// identity opacity.
    fresh: BTreeMap<TileCoord, Arc<Kept>>,
    /// The ceiling lane's carried tiles ([`LoopCarry::levels`]), evolving with
    /// the run exactly as `fresh` does. Empty unless the pen drives the ceiling.
    levels: BTreeMap<TileCoord, Arc<Kept>>,
    /// The brush as the **swept** sweep binds it — its prefix-τ and noise groups
    /// over the swept layouts — for the per-segment draw of the ceiling lane
    /// over the region (§6.2). `None` unless the pen drives the ceiling.
    lane: Option<LaneBinds>,
    /// Which tool this run is drawing (§6.2, §6.13) — what picks the plan a
    /// piece dispatches, and gates everything only the wet tool has: the mint
    /// budget, the settle, the carry.
    tool: LoopTool,
    /// The liquify follow's normalizer, `1 / peak_tau` of the resolved tip
    /// (`assets::peak_tau`) — read only by [`liquify_plan`]'s slots.
    inv_peak_tau: f32,
}

impl<'a> DynamicsRun<'a> {
    /// Whether the mint runs under a ceiling anywhere in this stroke, and so
    /// whether its budget lanes have to be carried across pieces (§6.2): the
    /// dial below 1, the pen driving it — the ceiling lane in the aux's `.w`
    /// then rides the same carry — *or* a mask in force, whose rim is at least
    /// a pixel soft, and a texel under it caps its mint exactly as a dial below
    /// 1 does (§6.8). The mask's overall opacity is already inside
    /// `consts.opacity` (`stroke_constants`); this is about its coverage.
    ///
    /// Never for the liquify tool: it mints nothing for a ceiling to budget —
    /// its opacity *is* 1 (`BrushEffect::opacity`) and the mask scales its
    /// follow directly in the warp kernel (§6.13) — so even under a selection
    /// there are no lanes to seed or carry.
    fn capped(&self) -> bool {
        matches!(self.tool, LoopTool::Wet(_))
            && (self.consts.opacity < 1.0
                || self.consts.ceiling_lane
                || self.scene.selection.is_active())
    }

    /// Whether the pen drives this stroke's ceiling (§6.2) — the run then draws
    /// the ceiling lane over its region per segment and carries it per tile
    /// beside the mint budget. Implies [`capped`](Self::capped).
    fn leveled(&self) -> bool {
        self.lane.is_some()
    }

    /// Open the run: resolve the brush's textures, and put the tool in the state the
    /// stroke arrives at this range with — resumed from `tool`, or freshly charged.
    fn new(
        r: &'a StrokeRenderer,
        scene: StrokeScene<'a>,
        rec: &'a StrokeRecord,
        tol: crate::path::FlattenTolerance,
        tool: Option<&ToolState>,
        brush: LoopBrush<'_>,
    ) -> Self {
        let device = &r.ctx.device;
        let mut scope = r.scratch.scope(&r.ctx, "stark dynamics stroke");
        let consts = r.stroke_constants(rec, scene.substrate, scene.selection);

        // The brush's swept-extent prefix-τ (shared with the fast path) and its
        // plain coverage mask (the reservoir texels' own extent weights).
        let prefix_view = brush.tip.prefix.clone();
        let prefix_bg = desc::bind_group_for(
            device,
            "stark dynamics prefix bg",
            &r.dynamics.prefix_bgl,
            super::kit::PREFIX_SLOTS,
            false,
            |_| wgpu::BindingResource::TextureView(&prefix_view),
        );
        let cov = brush.tip.coverage.clone();
        // Color dynamics for the brush's own `add` paint — the same field and
        // lookup parameters as the fast path (see `deposit` in dynamics.wesl).
        let noise = r.tips.noise(&rec.brush.color_dynamics(), consts.noise_seed);
        scope.hold(noise.clone());

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
        let brush_pair = |scope: &mut SubmitScope, labels: [&'static str; 2]| {
            let pair = labels.map(|label| {
                scope.take_run(Key {
                    size: (BRUSH_RES, BRUSH_RES),
                    format: d::BRUSH_DST_COLOR_W.storage_format(),
                    usage: brush_usage,
                    label,
                })
            });
            let [(tex_a, view_a), (tex_b, view_b)] = pair;
            ([tex_a, tex_b], [view_a, view_b])
        };
        let (brush_color_tex, brush_color) = brush_pair(
            &mut scope,
            [
                "stark dynamics brush color a",
                "stark dynamics brush color b",
            ],
        );
        let (brush_aux_tex, brush_aux) = brush_pair(
            &mut scope,
            ["stark dynamics brush aux a", "stark dynamics brush aux b"],
        );
        // The residual's half of the ping-pong, allocated only where there is one.
        let (brush_resid_tex, brush_resid) = r
            .color_space
            .has_resid()
            .then(|| {
                brush_pair(
                    &mut scope,
                    [
                        "stark dynamics brush resid a",
                        "stark dynamics brush resid b",
                    ],
                )
            })
            .unzip();
        // The carried mint budget (§6.2): shared forward — a piece only ever
        // reads the tiles it resumed, seeding its region by copy and extracting
        // fresh leases — so this clone is a refcount per tile. Empty at the
        // identity opacity, and on a fresh tip.
        // The ceiling lane (§6.2), on a wet stroke whose ceiling the pen drives:
        // the carried tiles, and the swept sweep's own bind groups for drawing the
        // lane over the region — the brush bound exactly as the fast path binds
        // it, so the lane's sums are the fast path's.
        let leveled = matches!(brush.tool, LoopTool::Wet(_)) && consts.ceiling_lane;
        let levels: BTreeMap<TileCoord, Arc<Kept>> =
            tool.map(ToolState::looped).map_or_else(BTreeMap::new, |t| {
                t.levels.iter().map(|(c, k)| (*c, Arc::clone(k))).collect()
            });
        let lane = leveled.then(|| {
            let (prefix, noise) = super::super::swept::sweep_binds(
                r,
                &mut scope,
                brush.tip,
                rec,
                scene.substrate,
                &consts,
            );
            LaneBinds { prefix, noise }
        });
        let fresh: BTreeMap<TileCoord, Arc<Kept>> = if let Some(t) = tool.map(ToolState::looped) {
            // Resume: the tip arrives at this range carrying exactly what it carried
            // when the previous range stopped.
            scope.encoder().copy_texture_to_texture(
                t.reservoir.color.tex().as_image_copy(),
                brush_color_tex[0].as_image_copy(),
                RESERVOIR_EXTENT,
            );
            scope.encoder().copy_texture_to_texture(
                t.reservoir.aux.tex().as_image_copy(),
                brush_aux_tex[0].as_image_copy(),
                RESERVOIR_EXTENT,
            );
            if let (Some(src), Some(dst)) = (&t.reservoir.resid, &brush_resid_tex) {
                scope.encoder().copy_texture_to_texture(
                    src.tex().as_image_copy(),
                    dst[0].as_image_copy(),
                    RESERVOIR_EXTENT,
                );
            }
            t.fresh.iter().map(|(c, k)| (*c, Arc::clone(k))).collect()
        } else {
            // Init: latent = the brush's own color, per-unit opacity = exactly 1
            // (minted paint is opaque per unit, §6.2); the carried amount starts
            // at the pre-`charge` glob (0 = empty tool). A liquify brush has no
            // glob — and no reservoir reader either: its plan dispatches nothing
            // that touches the tool, so this clear only keeps the pool's
            // no-stale-reads contract the cheap way.
            let charge = match brush.tool {
                LoopTool::Wet(d) => d.charge,
                LoopTool::Liquify => 0.0,
            };
            scope
                .encoder()
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("stark dynamics brush init"),
                    color_attachments: &[
                        Some(desc::attach(
                            &brush_color[0],
                            desc::clear_to(wgpu::Color {
                                r: consts.channels[0] as f64,
                                g: consts.channels[1] as f64,
                                b: consts.channels[2] as f64,
                                a: 1.0,
                            }),
                        )),
                        Some(desc::attach(
                            &brush_aux[0],
                            desc::clear_to(wgpu::Color {
                                // Carried height = the pre-`charge` glob; the rest of
                                // the reservoir aux is unused (height is the only
                                // thing the tool carries, §6.1). The effect's opacity
                                // scales the glob outright, and for a finite source
                                // that *is* the ceiling: everything the glob can ever
                                // deliver is the scaled glob, so no budget lane is
                                // needed where the `add` rate's unbounded mint takes
                                // one (§6.2, `dynamics.wesl::lay_parcel`).
                                r: (charge * consts.opacity) as f64,
                                g: 0.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                        )),
                        // A freshly charged tip holds the brush's own color, and that
                        // color's residual with it — the same clear, off the same
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
                    ..Default::default()
                });
            BTreeMap::new()
        };
        let mut bake = |label: &'static str| {
            scope
                .take_run(Key {
                    size: (BAKE_RES, BAKE_RES),
                    format: d::BAKE_LOAD_W.storage_format(),
                    usage: LOOP_USAGE,
                    label,
                })
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
            scope,
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
            fresh,
            levels,
            lane,
            tool: brush.tool,
            // Floored where it is inverted, though `assets::peak_tau` already
            // floors what it hands out: a divisor is the one reading of the
            // number that cannot shrug off a zero.
            inv_peak_tau: 1.0 / brush.tip.peak_tau.max(1e-6),
        }
    }

    /// Evolve one region-sized piece of the stroke over `base`: composite the tiles
    /// under `segments` into a region, walk them through the loop, and slice the
    /// result back into fresh CoW tiles. The tool carries on from where the previous
    /// piece left it, and the canvas side needs no carrying — it is in `base`, which
    /// for a later piece is what the earlier ones wrote back.
    /// `fires` is this piece's slice of the range's bleed firings, `after` re-keyed
    /// to `segments`; `settle` is set only for the piece that ends the stroke: see
    /// [`StrokeRenderer::render_dynamic`] and `dynamics.wesl::settle`.
    fn draw(
        &mut self,
        base: &TileMap,
        segments: &[Segment],
        fires: &[BleedFire],
        settle: bool,
    ) -> TileMap {
        // One region-sized piece of the range. Its count against
        // `stroke.dynamics`'s says how often a range needed chunking at all, which
        // is a fact about brush radius and stroke length that nothing else reports.
        crate::timing::span!("stroke.piece");
        self.scope.flush();
        // Segments and firings both, in one walk: a window writes up to a quantum
        // behind the piece's first segment, and a tile (or region) the walk misses is
        // flux silently clipped at the boundary (`segments::piece_sweeps`).
        let covered = cover(segments, fires);
        // A piece holds at least one segment, and a segment covers at least one tile,
        // so the empty case cannot arise here — but it costs nothing to leave the
        // canvas alone if it ever did.
        let Some(RegionRect {
            halo,
            lo,
            origin: region_origin,
            w,
            h,
        }) = covered.rect()
        else {
            return base.clone();
        };
        let coords = &covered.tiles;
        // The run's dirty set is the union of the pieces', which is why no caller has
        // to enumerate the whole range's tiles a second time.
        self.dirty.extend(coords.iter().copied());
        // **Hold the map this piece is about to read.** The composite below samples
        // `base`'s tiles into the piece's encoder, and the caller replaces its `map`
        // with what this call returns — dropping, at that assignment, every tile handle
        // the new map superseded. A `TilePairHandle`'s drop *is* its release to
        // `TilePool`'s free list (`gpu::tile::GpuTex::drop`), so those textures become
        // available to the next `acquire_tex` while the commands reading them are still
        // only recorded. `PoolInner::trim` guards the irreversible half of that — it
        // will not *destroy* a slot returned during the current epoch — but reuse is
        // unguarded by construction, and reuse is enough: the next acquire renders over
        // a texture this encoder is still compositing from.
        //
        // Consecutive pieces share the tiles around their cut, because `region::cover`
        // grows every segment's box by an apron — so this is the ordinary case for any
        // stroke long enough to be chunked, not a corner.
        //
        // It has never fired, and the reason was not a rule: the next statement the run
        // executes is the following piece's `flush`, and nothing acquires in between.
        // Holding the map moves that from an accident of statement order to the same
        // footing everything else here stands on — the scope releases it in the call
        // that submits the commands naming it, exactly as `render_swept` holds its
        // scratch pair across the identical boundary. The clone is an `rpds` map of
        // `Arc` handles: a refcount per tile, no pixels (§5.2).
        self.scope.hold(base.clone());
        // Composite the canvas under the piece into a 1:1 region — the read half of
        // the loop's cost, and the one that scales with the *area* the stroke covers
        // rather than with its segment count.
        let region = {
            crate::timing::span!("stroke.region");
            self.composite_region(base, &halo, region_origin, w, h)
        };
        // Seed the opacity ceiling's mint budget (§6.2): the composite above
        // laid zeros in the aux's `.yz` lanes, and a resumed tile's running
        // totals are copied back over its block — the `.x` height rides along
        // bit-identical, being the very value the write-back sliced into the
        // tile the composite just re-laid. Nothing to seed at the identity
        // ceiling, where the shader never reads the lanes.
        if self.capped() {
            for coord in coords {
                let Some(kept) = self.fresh.get(coord) else {
                    continue;
                };
                let off = coord.origin() - lo;
                self.scope.encoder().copy_texture_to_texture(
                    kept.tex().as_image_copy(),
                    wgpu::TexelCopyTextureInfo {
                        texture: &region.aux_tex,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: off.x as u32,
                            y: off.y as u32,
                            z: 0,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    fresh_key().extent(),
                );
            }
        }
        // …and the ceiling lane's carried tiles over the cleared lane (§6.2), the
        // same cut.
        if let Some(levels_tex) = &region.levels_tex {
            for coord in coords {
                let Some(kept) = self.levels.get(coord) else {
                    continue;
                };
                let off = coord.origin() - lo;
                self.scope.encoder().copy_texture_to_texture(
                    kept.tex().as_image_copy(),
                    wgpu::TexelCopyTextureInfo {
                        texture: levels_tex,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: off.x as u32,
                            y: off.y as u32,
                            z: 0,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    levels_key().extent(),
                );
            }
        }

        // ---- The dispatch plan, one slot per dispatch, uploaded as one buffer the
        // loop reads through dynamic offsets.
        //
        // **Before the scratch it sizes**, which is the order the two are actually in:
        // the plan measures every rect it will dispatch and the snapshot square is
        // their maximum. Run the other way, the scratch would size itself from a
        // position-independent bound over the same coverage boxes and the plan would
        // have to assert each real rect came in under it.
        //
        // The plan, the scratch it sizes and the bind groups that name it are one
        // row: they are the fixed per-piece setup, they move together (a slot count
        // drives all three), and splitting them would put three sub-quantum numbers
        // where the browser's clock can resolve one.
        //
        // **The block ends the scratch handles' scope, and that is sound** — worth
        // stating outright in this file, where an early release is the standing
        // hazard (`scratch::SubmitScope`). `take_piece` and `take_piece_buffer`
        // register the *lease* with the scope and hand back refcounted `wgpu` clones,
        // so what drops here is a handle and never a claim on the pool: the leases are
        // released by the flush that submits the commands naming them, which is the
        // next piece's or `finish`'s. The bind groups outlive the block because they
        // are what the loop below is recorded against, and a `BindGroup` holds its
        // own references to every view in it.
        let (plan, bind, levels_pass) = {
            crate::timing::span!("stroke.plan");
            let ctx = PlanCtx {
                rec: self.rec,
                tol: self.tol,
                region_origin,
                consts: &self.consts,
                substrate: self.scene.substrate,
            };
            let plan = match self.tool {
                LoopTool::Wet(_) => dynamics_plan(&ctx, segments, fires, settle),
                LoopTool::Liquify => liquify_plan(&ctx, segments, self.inv_peak_tau),
            };
            let under = self.snapshot_scratch(plan.dsize);
            // The cell scratch (§6.2), only when some slot actually takes the coarse
            // path — a small or hard tip allocates nothing and binds nothing.
            let cells = plan
                .slots
                .iter()
                .any(|d| d.cell_groups.is_some())
                .then(|| self.cell_scratch(plan.dsize));
            // The bleed pair's mobility scratch (§6.2), only when some slot is a firing —
            // a brush that does not bleed allocates nothing and binds the 1×1 stand-in.
            let bleed = plan
                .slots
                .iter()
                .any(|d| matches!(d.kind, SlotKind::Bleed))
                .then(|| self.bleed_scratch(plan.dsize));
            let stamp_buf = self.upload_plan(&plan.slots);
            let bind = self.bind_piece(&region, &under, &stamp_buf, cells.as_ref(), bleed.as_ref());
            // The ceiling lane's draw state (§6.2): the segments as the sweep
            // instances them, and the region as one sweep target.
            let levels_pass = region.levels.as_ref().map(|view| {
                let lane = self
                    .lane
                    .as_ref()
                    .expect("a leveled region comes of a lane");
                let instances: Vec<SegmentInstance> =
                    segments.iter().map(segment_instance).collect();
                let bytes: &[u8] = bytemuck::cast_slice(&instances);
                let buf = self.scope.take_piece_buffer(BufKey {
                    size: bytes.len() as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    label: "stark dynamics levels instances",
                });
                self.scope.write_lease(&buf, bytes);
                let xform = tile_xform(
                    self.rec,
                    &self.consts,
                    region_origin,
                    (w as f32, h as f32),
                    1,
                );
                LevelsPass {
                    view: view.clone(),
                    xform: xform_group(self.r, &mut self.scope, &xform),
                    prefix: lane.prefix.clone(),
                    noise: lane.noise.clone(),
                    instances: buf,
                }
            });
            (plan, bind, levels_pass)
        };

        {
            // The stamp loop itself: bake → exchange → deposit per segment, plus the
            // bleed firings and the pen-up settle. **The row to watch** — this is
            // where a per-segment cost lands, and where a change to the dispatch
            // count shows up before it shows up in a total.
            crate::timing::span!("stroke.loop");
            self.record_loop(&plan.slots, plan.dsize, &bind, levels_pass.as_ref());
        }
        // Extract the budget's new totals for the carry (§6.2): a fresh lease
        // per touched tile, cut exactly as the write-back cuts paint — the
        // carried tiles themselves are never written, which is what lets the
        // live tail resume the same frozen totals every pointer move.
        if self.capped() {
            let r = self.r;
            for coord in coords {
                let kept = r.scratch.keep(&r.ctx.device, fresh_key());
                let off = coord.origin() - lo;
                self.scope.encoder().copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &region.aux_tex,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: off.x as u32,
                            y: off.y as u32,
                            z: 0,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    kept.tex().as_image_copy(),
                    fresh_key().extent(),
                );
                // Held, not dropped. The lease this displaces is the *previous*
                // piece's, and the seed copy that read it (above) is recorded into
                // an encoder this call does not submit — so dropping it here returns
                // a texture to the free list while pending commands still name it,
                // which is `scope.hold(base.clone())`'s hazard one level down and
                // the ordinary case for the same reason: consecutive pieces share
                // the tiles around their cut. Reuse happens to be harmless (commands
                // in one encoder run in recorded order, and the seed was recorded
                // first), but `trim`'s `destroy` is not, and neither is an argument
                // that rests on statement order.
                if let Some(displaced) = self.fresh.insert(*coord, Arc::new(kept)) {
                    self.scope.hold(displaced);
                }
            }
        }
        // The ceiling lane's new sums, cut the same way and held the same way.
        if let Some(levels_tex) = &region.levels_tex {
            let r = self.r;
            for coord in coords {
                let kept = r.scratch.keep(&r.ctx.device, levels_key());
                let off = coord.origin() - lo;
                self.scope.encoder().copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: levels_tex,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: off.x as u32,
                            y: off.y as u32,
                            z: 0,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    kept.tex().as_image_copy(),
                    levels_key().extent(),
                );
                if let Some(displaced) = self.levels.insert(*coord, Arc::new(kept)) {
                    self.scope.hold(displaced);
                }
            }
        }

        // Slicing the evolved region back into fresh CoW tiles — the write half,
        // paired with `stroke.region`'s read.
        crate::timing::span!("stroke.writeback");
        self.write_back(base, coords, lo, &region)
    }

    /// The piece's canvas region — a 1:1 copy of what lies under its segments — and
    /// the selection over it.
    ///
    /// Composited from the base tiles of the affected set plus a one-tile ring, so
    /// rewritten tiles' aprons read real neighbour content (§6.4). Rgba16Float
    /// throughout: it is both filterable and a core storage format, and matches the
    /// tile color format of both color spaces (asserted in `build_dynamics_kit`).
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
        // texture copy ([`Self::write_back`]); COPY_DST because the opacity
        // ceiling's mint budget is seeded back *into* the aux the same way
        // (§6.2, `LoopCarry::fresh`). Pooled: the composite's clearing load
        // op rewrites every texel, so stale contents never show (`scratch`).
        let region_usage = wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST
            | LOOP_USAGE;
        let mut region_tex = |label: &'static str| {
            self.scope.take_piece(Key {
                size: (w, h),
                format: d::REGION_COLOR_W.storage_format(),
                usage: region_usage,
                label,
            })
        };
        let (color_tex, color) = region_tex("stark dynamics region color");
        let (aux_tex, aux) = region_tex("stark dynamics region aux");
        // The region's residual (§6.7), in the same format for the same reason: it is
        // the rest of the color beside it, and the loop reads and writes the two at
        // exactly the same points.
        let (resid_tex, resid) = r
            .color_space
            .has_resid()
            .then(|| region_tex("stark dynamics region resid"))
            .unzip();
        // The ceiling lane over the region (§6.2), on a stroke whose ceiling the
        // pen drives: **drawn** into per segment, so a render attachment rather
        // than storage, and read by the deposits between draws. Cleared here —
        // the pool hands back stale texels — and seeded from the carry below.
        let (levels_tex, levels) = self
            .leveled()
            .then(|| {
                let (tex, view) = self.scope.take_piece(Key {
                    size: (w, h),
                    format: super::super::swept::CEILING_FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::COPY_DST,
                    label: "stark dynamics region levels",
                });
                self.scope
                    .encoder()
                    .begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("stark dynamics region levels clear"),
                        color_attachments: &[Some(desc::attach(&view, desc::CLEAR))],
                        ..Default::default()
                    });
                (tex, view)
            })
            .unzip();

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
        let view_buf = self.scope.take_piece_buffer(BufKey {
            size: std::mem::size_of_val(&view) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            label: "stark dynamics region view",
        });
        self.scope.write_lease(&view_buf, bytemuck::bytes_of(&view));
        let view_bg = desc::bind_group_for(
            device,
            "stark dynamics region view bg",
            &kit.composite_view_bgl,
            crate::gpu::composite::COMPOSITE_VIEW_SLOTS,
            false,
            |i| match i {
                cb::VIEW => view_buf.as_entire_binding(),
                cb::SAMP => wgpu::BindingResource::Sampler(&kit.composite_sampler),
                other => unreachable!("the composite view group lists no binding {other}"),
            },
        );
        let mut tile_origins: Vec<TileInstance> = Vec::new();
        let mut tile_bgs: Vec<&wgpu::BindGroup> = Vec::new();
        for coord in halo {
            if let Some(tile) = base.get(coord) {
                tile_origins.push(TileInstance {
                    origin: coord.origin().to_array(),
                    opacity: 1.0,
                });
                // **The group the tile itself caches**, which is why this loop is
                // handed pass A's layout rather than building one from the same slot
                // list (`kit.rs`). A live stroke re-composites its halo on every
                // pointer move, and the halo of a wide tip is tens of tiles per piece
                // — so a group built here was tens of WebGPU objects a move, which is
                // the allocation *rate* `TilePairHandle::composite_bg` was introduced
                // to stop pass A paying and this path went on paying.
                //
                // A resident tile in a pigment space always has a residual, so
                // `Zeroes` never stands in here.
                tile_bgs.push(crate::gpu::composite::tile_bind_group(
                    device,
                    &kit.composite_tile_bgl,
                    tile,
                ));
            }
        }
        let tile_inst = (!tile_origins.is_empty()).then(|| {
            let bytes: &[u8] = bytemuck::cast_slice(&tile_origins);
            let buf = self.scope.take_piece_buffer(BufKey {
                size: bytes.len() as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                label: "stark dynamics region tile instances",
            });
            self.scope.write_lease(&buf, bytes);
            buf
        });
        {
            let targets = Targets {
                color: &color,
                aux: &aux,
                resid: resid.as_ref(),
            };
            let att = targets.attachments(desc::CLEAR);
            let mut pass = self
                .scope
                .encoder()
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("stark dynamics region composite"),
                    color_attachments: &att[..targets.count()],
                    ..Default::default()
                });
            // An empty region (no base tiles) just stays cleared → "no paint".
            if let Some(inst) = &tile_inst {
                pass.set_pipeline(&kit.composite_pipeline);
                // Offset 0: the slot list this group answers to is pass A's, whose
                // view is dynamic for the eyedropper's sake (`composite::ViewBindings`).
                // This loop composites one region through one view and has one slot.
                pass.set_bind_group(0, &view_bg, &[0]);
                pass.set_vertex_buffer(0, inst.slice(..));
                for (i, bg) in tile_bgs.iter().enumerate() {
                    let idx = i as u32;
                    pass.set_bind_group(1, *bg, &[]);
                    pass.draw(0..4, idx..idx + 1);
                }
            }
        }

        // The selection over this region (§6.8), gathered from the same halo tiles the
        // paint came from, so it is 1:1 with the color/aux regions. An unrestricted
        // selection binds the 1×1 constant instead — the loop's masked reads then
        // return 1 everywhere and the stroke behaves exactly as before.
        let sel_mask = if self.scene.selection.is_universal() {
            r.selection.constant(1.0)
        } else {
            // Leased, not created: this is up to `MAX_REGION_DIM`² of `R8Unorm` and a
            // live stroke re-gathers it per piece per pointer move, so creating one
            // was megabytes of allocation and teardown a move for a texture of the
            // very same shape each time (`scratch`).
            let (_, view) = self.scope.take_piece(Key {
                size: (w, h),
                format: crate::gpu::tile::MASK_FORMAT,
                usage: crate::gpu::SelectionRenderer::REGION_MASK_USAGE,
                label: "stark selection region mask",
            });
            r.selection.region_mask(
                &mut self.scope,
                &view,
                self.scene.selection,
                halo,
                (region_origin, stark_model::geom::Extent2::new(w, h)),
            );
            view
        };

        Region {
            color_tex,
            color,
            aux_tex,
            aux,
            resid_tex,
            resid,
            levels_tex,
            levels,
            sel_mask,
        }
    }

    /// The extent snapshot scratch: the copy that gives `deposit` and `settle`
    /// something to read while they storage-write the region.
    ///
    /// `size` is the plan's own [`dsize`](super::plan::DynamicsPlan::dsize) — the
    /// largest rect the piece will actually dispatch, rounded to the pool's quantum.
    /// Nothing is asserted or clamped here because there is nothing left to check: the
    /// square is a maximum over the rects, not a bound the rects have to respect.
    fn snapshot_scratch(&mut self, size: u32) -> Snapshot {
        let r = self.r;
        let mut under_tex = |label: &'static str| {
            self.scope
                .take_piece(Key {
                    size: (size, size),
                    format: d::REGION_COLOR_W.storage_format(),
                    usage: LOOP_USAGE,
                    label,
                })
                .1
        };
        Snapshot {
            color: under_tex("stark dynamics under color"),
            aux: under_tex("stark dynamics under aux"),
            resid: r
                .color_space
                .has_resid()
                .then(|| under_tex("stark dynamics under resid")),
        }
    }

    /// The bleed pair's mobility scratch (§6.2): where `bleed_weight` leaves the weight
    /// the ladder reads back thirty-six times a texel.
    ///
    /// The **snapshot square**, not a firing's rect, and that is what makes the read
    /// safe: the ladder clamps a tap into the scratch's own extent, so a texel outside
    /// the dispatched rect can still be read, and one this pass had not written would
    /// hold the previous firing's answer. Covering the square is the same
    /// structural-fit argument the snapshot and the cells make, with the same `dsize`.
    fn bleed_scratch(&mut self, dsize: u32) -> wgpu::TextureView {
        self.scope
            .take_piece(Key {
                size: (dsize, dsize),
                format: d::BLEED_W_W.storage_format(),
                usage: LOOP_USAGE,
                label: "stark dynamics bleed w",
            })
            .1
    }

    /// The extent-cell scratch (§6.2): where `cell_hoist` leaves the per-cell
    /// means for `deposit_coarse` to read back. Sized by the same structural-fit
    /// relation the snapshot scratch uses — [`cell_scratch_size`] of the piece's own
    /// `dsize`, which `cell_geometry` asserted every slot's hoist grid against — and
    /// leased on the piece like the region, for the same lifetime argument.
    fn cell_scratch(&mut self, dsize: u32) -> Cells {
        let r = self.r;
        let size = cell_scratch_size(dsize);
        let mut tex = |label: &'static str| {
            self.scope
                .take_piece(Key {
                    size: (size, size),
                    format: d::BAKE_LOAD_W.storage_format(),
                    usage: LOOP_USAGE,
                    label,
                })
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

    /// The plan's uniform slots, one [`STAMP_STRIDE`]-aligned window each — dynamic
    /// uniform offsets being the standard way to vary a uniform across dispatches
    /// within one pass.
    ///
    /// Leased on the *piece*, like the region it works on and for the same reason:
    /// nothing past this piece's own submission reads it, so holding it for the whole
    /// run would make a long stroke's peak cost scale with the number of pieces —
    /// which is what [`MAX_REGION_DIM`](super::super::budget::MAX_REGION_DIM) exists to prevent.
    /// The scope returns it to the pool behind the piece's own submit
    /// ([`SubmitScope::flush`]), so the next piece's plan is written into the buffer
    /// this one used.
    fn upload_plan(&mut self, plan: &[LoopDispatch]) -> wgpu::Buffer {
        let mut data = vec![0u8; plan.len() * STAMP_STRIDE as usize];
        for (i, d) in plan.iter().enumerate() {
            let at = i * STAMP_STRIDE as usize;
            data[at..at + SLOT].copy_from_slice(bytemuck::bytes_of(&d.slot));
        }
        let buf = self.scope.take_piece_buffer(BufKey {
            size: data.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            label: "stark dynamics stamps",
        });
        self.scope.write_lease(&buf, &data);
        buf
    }

    /// Every bind group the loop switches between while recording one piece.
    ///
    /// Each is built from the very slot list its layout was ([`slots`]),
    /// so a group and its layout cannot disagree about which bindings are present or in
    /// what order. Written as two hand-kept arrays per entry point — one here, one in
    /// [`kit`](super::kit) — they would be aligned by nothing but the order they were
    /// written in and a per-layout element count.
    ///
    /// What each arm below supplies is therefore only the **resources**: given a slot,
    /// which view or buffer goes in it. A slot the residual gate excludes is never
    /// asked for, which is what retired `push_resid` and the `Option` juggling around
    /// it — a group takes its whole residual tail or none of it because the shader's
    /// `@if(resid)` says so, not because the host counted correctly.
    ///
    /// `ST` binds a single slot-sized window of `stamp_buf` whose dynamic offset
    /// selects the dispatch, so all of these are built once per piece and the loop
    /// varies only the offset.
    fn bind_piece<'p>(
        &'p self,
        region: &'p Region,
        under: &'p Snapshot,
        stamp_buf: &'p wgpu::Buffer,
        cells: Option<&'p Cells>,
        bleed: Option<&'p wgpu::TextureView>,
    ) -> PieceBindings {
        let r = self.r;
        let kit = &r.dynamics;
        let device = &r.ctx.device;
        let resid = r.color_space.has_resid();

        let params = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: stamp_buf,
            offset: 0,
            size: wgpu::BufferSize::new(SLOT as u64),
        });
        let view = |v: &'p wgpu::TextureView| wgpu::BindingResource::TextureView(v);
        let samp = |s: &'p wgpu::Sampler| wgpu::BindingResource::Sampler(s);
        // A residual view is `Some` exactly when the color space has one, and the
        // resolvers below are only ever asked for a residual slot in that case — the
        // shader's own `@if(resid)` gate, applied by `bind_group_for`. So this states
        // that gate rather than guarding a case the host can reach.
        let opt = |v: Option<&'p wgpu::TextureView>, what: &'static str| {
            wgpu::BindingResource::TextureView(
                v.unwrap_or_else(|| panic!("a residual build must have a {what} residual")),
            )
        };
        // The slots the two paint-laying entry points share, plus the ones every arm
        // reads. Returns `None` for anything outside that common set, so each arm's
        // `match` states only what makes it different from its neighbours.
        let common = |s: u32| -> Option<wgpu::BindingResource<'p>> {
            Some(match s {
                b::ST => params.clone(),
                b::BAKE_LOAD => view(&self.bake_load),
                b::BAKE_LATM => view(&self.bake_latm),
                b::BAKE_RLM => opt(self.bake_rlm.as_ref(), "bake"),
                b::UNDER_COLOR => view(&under.color),
                b::UNDER_AUX => view(&under.aux),
                b::UNDER_RESID => opt(under.resid.as_ref(), "snapshot"),
                b::REGION_COLOR_W => view(&region.color),
                b::REGION_AUX_W => view(&region.aux),
                b::REGION_RESID_W => opt(region.resid.as_ref(), "region"),
                b::SEL_MASK => view(&region.sel_mask),
                b::SUBSTRATE_TEX => view(&self.scene.substrate.view),
                // The real scratch on a piece that fires, the kit's 1×1 otherwise — a
                // painting segment carries `lambda_bleed = 0` and never reads it.
                b::BLEED_W => view(bleed.unwrap_or(&kit.bleed_placeholder)),
                // The ceiling lane (§6.2), by the same argument: a stroke whose
                // ceiling the pen does not drive carries `ceiling_lane = 0`.
                b::REGION_LEVELS => view(region.levels.as_ref().unwrap_or(&kit.levels_placeholder)),
                b::DYN_NOISE_TEX => view(self.noise.view()),
                b::DYN_NOISE_SAMP => samp(&r.tips.noise_sampler),
                _ => return None,
            })
        };

        let snapshot = desc::bind_group_for(
            device,
            "stark dynamics snapshot bg",
            &kit.snapshot_bgl,
            slots::SNAPSHOT,
            resid,
            |s| match s {
                b::REGION_COLOR => view(&region.color),
                b::REGION_AUX => view(&region.aux),
                b::REGION_RESID => opt(region.resid.as_ref(), "region"),
                b::UNDER_COLOR_W => view(&under.color),
                b::UNDER_AUX_W => view(&under.aux),
                b::UNDER_RESID_W => opt(under.resid.as_ref(), "snapshot"),
                other => common(other).expect("snapshot lists no other binding"),
            },
        );
        // `exchange` comes in two flavours for the reservoir ping-pong: each reads one
        // half and writes the other. The residual ping-pongs on the same phase as the
        // color — read `i`, write `1 - i` — or the tool's two halves would drift apart.
        let exchange = std::array::from_fn(|i| {
            desc::bind_group_for(
                device,
                "stark dynamics exchange bg",
                &kit.exchange_bgl,
                slots::EXCHANGE,
                resid,
                |s| match s {
                    b::REGION_COLOR => view(&region.color),
                    b::REGION_AUX => view(&region.aux),
                    b::REGION_RESID => opt(region.resid.as_ref(), "region"),
                    b::UNDER_COLOR_W => view(&under.color),
                    b::UNDER_AUX_W => view(&under.aux),
                    b::UNDER_RESID_W => opt(under.resid.as_ref(), "snapshot"),
                    b::SAMP => samp(&kit.exchange_sampler),
                    b::COV_TEX => view(&self.cov),
                    b::BRUSH_SRC_COLOR => view(&self.brush_color[i]),
                    b::BRUSH_SRC_AUX => view(&self.brush_aux[i]),
                    b::BRUSH_SRC_RESID => {
                        opt(self.brush_resid.as_ref().map(|v| &v[i]), "reservoir")
                    }
                    b::BRUSH_DST_COLOR_W => view(&self.brush_color[1 - i]),
                    b::BRUSH_DST_AUX_W => view(&self.brush_aux[1 - i]),
                    b::BRUSH_DST_RESID_W => {
                        opt(self.brush_resid.as_ref().map(|v| &v[1 - i]), "reservoir")
                    }
                    other => common(other).expect("exchange lists no other binding"),
                },
            )
        });
        // One bake bind group per reservoir phase; the deposit reads only the baked
        // result, so it never touches the ping-pong.
        let bake = std::array::from_fn(|i| {
            desc::bind_group_for(
                device,
                "stark dynamics bake bg",
                &kit.bake_bgl,
                slots::BAKE,
                resid,
                |s| match s {
                    b::SAMP => samp(&kit.exchange_sampler),
                    b::BRUSH_SRC_COLOR => view(&self.brush_color[i]),
                    b::BRUSH_SRC_AUX => view(&self.brush_aux[i]),
                    b::BRUSH_SRC_RESID => {
                        opt(self.brush_resid.as_ref().map(|v| &v[i]), "reservoir")
                    }
                    b::BAKE_LOAD_W => view(&self.bake_load),
                    b::BAKE_LATM_W => view(&self.bake_latm),
                    b::BAKE_RLM_W => opt(self.bake_rlm.as_ref(), "bake"),
                    other => common(other).expect("bake lists no other binding"),
                },
            )
        });
        // The mobility pass, on a piece that has one to run.
        let bleed_weight = bleed.map(|w| {
            desc::bind_group_for(
                device,
                "stark dynamics bleed weight bg",
                &kit.bleed_weight_bgl,
                slots::BLEED_WEIGHT,
                resid,
                |sl| match sl {
                    b::BLEED_W_W => view(w),
                    other => common(other).expect("bleed_weight lists no other binding"),
                },
            )
        });
        let deposit = desc::bind_group_for(
            device,
            "stark dynamics deposit bg",
            &kit.deposit_bgl,
            slots::DEPOSIT,
            resid,
            |s| match s {
                b::SAMP => samp(&kit.exchange_sampler),
                other => common(other).expect("deposit lists no other binding"),
            },
        );
        // The pen-up, which reads the reservoir only through its own `bake` — so unlike
        // `exchange` it needs no bind group per ping-pong half; the bake's does that.
        let settle = desc::bind_group_for(
            device,
            "stark dynamics settle bg",
            &kit.settle_bgl,
            slots::SETTLE,
            resid,
            |s| common(s).expect("settle lists no other binding"),
        );
        // The warp's group (§6.13), only on the run whose plan can dispatch one —
        // every slot it lists is in `common`, the tool having nothing of its own
        // to bind.
        let warp = matches!(self.tool, LoopTool::Liquify).then(|| {
            desc::bind_group_for(
                device,
                "stark dynamics warp bg",
                &kit.warp_bgl,
                slots::WARP,
                resid,
                |s| common(s).expect("warp lists no other binding"),
            )
        });
        // The coarse pair's groups (§6.2), only for a piece whose plan hoists at all.
        // The hoist reads the same baked prefixes the exact deposit's front half did;
        // the coarse deposit swaps those two bindings for the cell means and keeps
        // everything else the exact one binds.
        let coarse = cells.map(|cl| {
            let hoist = desc::bind_group_for(
                device,
                "stark dynamics cell hoist bg",
                &kit.hoist_bgl,
                slots::HOIST,
                resid,
                |s| match s {
                    b::CELL_TOOL_W => view(&cl.tool),
                    b::CELL_LAT_W => view(&cl.lat),
                    b::CELL_RES_W => opt(cl.res.as_ref(), "cell"),
                    other => common(other).expect("cell hoist lists no other binding"),
                },
            );
            let deposit = desc::bind_group_for(
                device,
                "stark dynamics deposit coarse bg",
                &kit.deposit_coarse_bgl,
                slots::DEPOSIT_COARSE,
                resid,
                |s| match s {
                    b::CELL_TOOL => view(&cl.tool),
                    b::CELL_LAT => view(&cl.lat),
                    b::CELL_RES => opt(cl.res.as_ref(), "cell"),
                    other => common(other).expect("deposit_coarse lists no other binding"),
                },
            );
            CoarseBindings { hoist, deposit }
        });
        PieceBindings {
            snapshot,
            exchange,
            bake,
            deposit,
            bleed_weight,
            coarse,
            settle,
            warp,
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
    fn record_loop(
        &mut self,
        plan: &[LoopDispatch],
        dsize: u32,
        bind: &PieceBindings,
        levels: Option<&LevelsPass>,
    ) {
        let kit = &self.r.dynamics;
        let mut cur = self.cur;
        let prefix_bg = &self.prefix_bg;
        // Which of the piece's segments the next painting slot is: the plan emits
        // one such slot per segment, in order, with the firings and the settle
        // between them.
        let mut seg: u32 = 0;
        // The mobility pass covers the snapshot square, where every other dispatch
        // covers its own slot's rect — see the note at its `set_pipeline` below.
        let square_groups = super::plan::groups_for(dsize);
        let mut cpass = self
            .scope
            .encoder()
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stark dynamics stamp loop"),
                timestamp_writes: None,
            });
        // The prefix-τ rides at group 1 for `bake`, `deposit` and `settle`. Re-bound
        // after every pipeline switch: changing to a pipeline whose group-0 layout
        // differs invalidates the groups above it, and every consumer is reached only
        // across such a switch.
        for (i, d) in plan.iter().enumerate() {
            // Asked of `UniformSlots`, like the swept path's `xform_offset`, so the
            // stride the slots were *written* at and the stride they are *bound* at
            // are one expression rather than two that agree. This was
            // `(i as u64 * STAMP_STRIDE) as u32` — the same number, arrived at
            // independently.
            let off = UniformSlots::<stark_shaders::mirror::dynamics::Stamp>::offset(i as u32);
            match d.kind {
                // The tool plays no part in the lateral flux, so `cur` — the reservoir
                // ping-pong — stays exactly where the previous segment left it. The
                // standalone snapshot pipeline rather than the exchange's tail,
                // because there is no exchange dispatch to ride in.
                SlotKind::Bleed => {
                    cpass.set_pipeline(&kit.snapshot_pipeline);
                    cpass.set_bind_group(0, &bind.snapshot, &[off]);
                    cpass.dispatch_workgroups(d.groups.0, d.groups.1, 1);
                    // The pair's mobility, once per texel, for the ladder to read back
                    // (§6.2). Over the **whole snapshot square** rather than the
                    // firing's rect: the ladder clamps a tap into the scratch's extent,
                    // so every texel it can reach has to have been written this firing.
                    cpass.set_pipeline(&kit.bleed_weight_pipeline);
                    cpass.set_bind_group(
                        0,
                        bind.bleed_weight
                            .as_ref()
                            .expect("a firing's piece allocated a bleed scratch"),
                        &[off],
                    );
                    cpass.set_bind_group(1, prefix_bg, &[]);
                    cpass.dispatch_workgroups(square_groups, square_groups, 1);
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
                    // The extent `snapshot` rides in the tail of this same grid: it
                    // depends on nothing the exchange writes and the deposit needs
                    // both, so a barrier between them would buy no ordering at all.
                    // Hence the widened x — reservoir groups first, extent
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
                    // shoulder allows (`extent_cell`) — hoisted once per cell and
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
                    // The ceiling lane (§6.2): this segment's sums drawn over the
                    // region *after* its deposit, so the deposit read the lane as
                    // the segments before it left it and the next reads it with
                    // this one in. A render pass, because the lane accumulates by
                    // blending — the one read-modify-write of a region texel the
                    // loop makes without a snapshot — so the compute pass ends
                    // around it and resumes after.
                    if let Some(l) = levels {
                        drop(cpass);
                        {
                            let mut rpass = self.scope.encoder().begin_render_pass(
                                &wgpu::RenderPassDescriptor {
                                    label: Some("stark dynamics ceiling lane"),
                                    color_attachments: &[Some(desc::attach(&l.view, desc::LOAD))],
                                    ..Default::default()
                                },
                            );
                            rpass.set_pipeline(&self.r.swept.levels_pipeline);
                            rpass.set_bind_group(0, &l.xform, &[0]);
                            rpass.set_bind_group(1, &l.prefix, &[]);
                            rpass.set_bind_group(2, &l.noise, &[]);
                            rpass.set_vertex_buffer(0, l.instances.slice(..));
                            rpass.draw(0..SWEEP_VERTS, seg..seg + 1);
                        }
                        cpass =
                            self.scope
                                .encoder()
                                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                                    label: Some("stark dynamics stamp loop"),
                                    timestamp_writes: None,
                                });
                    }
                    seg += 1;
                    cur = 1 - cur;
                }
                // The pen-up: snapshot the final extent, bake the standing tip's
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
                // One liquify segment (§6.13): snapshot, then the gather. The
                // tool plays no part, so `cur` stays put — as it does for a
                // bleed firing, and for the same reason.
                SlotKind::Warp => {
                    // Over the **whole snapshot square**, the mobility pass's
                    // argument one arm up: the gather pulls from upstream of any
                    // one texel's sweep test, and a clamped tap must land on a
                    // texel this slot's snapshot wrote (`dynamics.wesl`'s
                    // `snapshot_at` reads `st.drag` to copy the square whole).
                    cpass.set_pipeline(&kit.snapshot_pipeline);
                    cpass.set_bind_group(0, &bind.snapshot, &[off]);
                    cpass.dispatch_workgroups(square_groups, square_groups, 1);
                    cpass.set_pipeline(&kit.warp_pipeline);
                    cpass.set_bind_group(
                        0,
                        bind.warp
                            .as_ref()
                            .expect("a warp slot comes only from a liquify plan"),
                        &[off],
                    );
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
    /// One region-sized narrow pass, then plain texture copies: the color and
    /// residual tiles are the region's own formats, so a copy is exact, and the aux
    /// copies out of the narrowed texture the single pass wrote — where this used to
    /// record a render pass per tile, ~30 of them per fold under a wide tip, whose
    /// fixed pass cost was most of the write-back's 15% share of a live gesture.
    /// Rounding is untouched: a copy is bit-exact, and the narrow pass render-writes
    /// a loaded f16 value back to its own lattice point (see `slice.wesl`).
    ///
    /// `lo` is the region's *interior* origin — the top-left tile origin, an apron in
    /// from the region rectangle — so a tile's offset into the region is measured
    /// against it. Offsets are integral and non-negative by [`Covered::rect`](super::super::region::Covered::rect)'s
    /// construction, and the far edge of the last tile's block is exactly the region's
    /// extent, so every copy is in bounds — and a violation is a loud validation error
    /// here, where a draw would silently read out of bounds instead.
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
        let (narrow, narrow_view) = self.scope.take_piece(Key {
            size: (region.color_tex.width(), region.color_tex.height()),
            format: r.color_space.aux_format(),
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            label: "stark dynamics narrow aux",
        });
        let bg = desc::bind_group_for(
            device,
            "stark dynamics slice bg",
            &kit.slice_bgl,
            super::kit::SLICE_SLOTS,
            false,
            |_| wgpu::BindingResource::TextureView(&region.aux),
        );
        {
            let mut pass = self
                .scope
                .encoder()
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("stark dynamics slice"),
                    color_attachments: &[Some(desc::attach(&narrow_view, desc::CLEAR))],
                    ..Default::default()
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
                self.scope.encoder(),
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

    /// Remember the tool for the range that resumes after this one. Copied rather
    /// than aliased: the loop's own reservoir textures go back to the pool when the
    /// run's scope closes, and the range that resumes will write its first exchange
    /// straight into whatever it is handed. 64² rgba16f ×2, so the copy is ~64 KB —
    /// nothing beside the region work it saves the next pointer move.
    ///
    /// The copies are pooled [`Kept`] leases rather
    /// than fresh textures: one of these is captured per pointer move, and the pool
    /// hands the same textures back — the drop that returns one is provably behind
    /// the resuming run's submit (see `Kept`).
    fn capture_tool(&mut self) -> ToolState {
        let r = self.r;
        let cur = self.cur;
        let (color_src, aux_src) = (
            self.brush_color_tex[cur].clone(),
            self.brush_aux_tex[cur].clone(),
        );
        let resid_src = self.brush_resid_tex.as_ref().map(|t| t[cur].clone());
        let scope = &mut self.scope;
        let mut copy_out = |src: &wgpu::Texture, label: &'static str| {
            let dst = r.scratch.keep(
                &r.ctx.device,
                Key {
                    size: (BRUSH_RES, BRUSH_RES),
                    format: d::REGION_COLOR_W.storage_format(),
                    usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
                    label,
                },
            );
            scope.encoder().copy_texture_to_texture(
                src.as_image_copy(),
                dst.tex().as_image_copy(),
                RESERVOIR_EXTENT,
            );
            dst
        };
        ToolState(Carried::Loop(Box::new(LoopCarry {
            reservoir: Reservoir {
                color: copy_out(&color_src, "stark tool state color"),
                aux: copy_out(&aux_src, "stark tool state aux"),
                // Carried across ranges with the color, and for the same reason: a tip
                // that lifted black paint has to still be carrying black when the next
                // pointer move resumes it (§6.7).
                resid: resid_src
                    .as_ref()
                    .map(|t| copy_out(t, "stark tool state resid")),
            },
            // The mint budget rides as the leases the last piece extracted —
            // already copies of the run's own state, so nothing more to copy
            // out here. Empty at the identity opacity.
            fresh: std::mem::take(&mut self.fresh),
            levels: std::mem::take(&mut self.levels),
        })))
    }

    /// Close the run: the scope submits what is still recorded, then releases
    /// everything behind that submit ([`SubmitScope::finish`]). What
    /// [`Self::capture_tool`] handed back is deliberately not among it: a `Kept`
    /// lease outlives this call by design.
    ///
    /// **The mint budget is handed to the scope rather than left to drop.** Whatever
    /// `capture_tool` did not take is still named by the last piece's extract copies,
    /// which this call is about to submit. Dropping it would be sound — un-bound
    /// fields of a destructured value drop at the end of the function, after
    /// `finish` — but only by a rule about drop timing, where every other release
    /// here stands on the scope. One footing is better than two.
    fn submit(mut self) {
        let fresh = std::mem::take(&mut self.fresh);
        let levels = std::mem::take(&mut self.levels);
        let Self { mut scope, .. } = self;
        scope.hold(fresh);
        scope.hold(levels);
        scope.finish();
    }
}

/// One piece's canvas region: a 1:1 copy of the canvas under its segments, which the
/// loop evolves in place before [`DynamicsRun::write_back`] slices it into tiles.
///
/// The color and residual carry their textures beside the views: the write-back
/// cuts tiles out of them by texture copy, and a copy is addressed by texture where
/// every pass binding is addressed by view. Both handles are `Arc`-backed clones of
/// the one piece-scoped allocation.
struct Region {
    color_tex: wgpu::Texture,
    color: wgpu::TextureView,
    /// The aux's texture beside its view, the color's reason: the opacity
    /// ceiling's mint budget is seeded into and extracted out of it by copy
    /// (§6.2, `LoopCarry::fresh`).
    aux_tex: wgpu::Texture,
    aux: wgpu::TextureView,
    /// The region's residual (§6.7), in a space that has one — evolved in place by the
    /// same dispatches that evolve the color, and sliced back with it.
    resid_tex: Option<wgpu::Texture>,
    resid: Option<wgpu::TextureView>,
    /// The ceiling lane over the region (§6.2), on a stroke whose ceiling the pen
    /// drives: drawn per segment, read by the deposits, seeded from and extracted
    /// into the carry by copy like the mint budget — never written back, being
    /// no part of a tile.
    levels_tex: Option<wgpu::Texture>,
    levels: Option<wgpu::TextureView>,
    /// The selection over the region (§6.8) — its own gathered mask, or the 1×1
    /// constant that stands in for an unrestricted selection.
    sel_mask: wgpu::TextureView,
}

/// The extent snapshot scratch.
///
/// It does not carry the square it was sized to: that number is the plan's
/// ([`DynamicsPlan::dsize`](super::plan::DynamicsPlan)), derived from the rects the
/// plan will dispatch, and the scratch is allocated *from* it rather than the plan
/// being checked against the scratch.
struct Snapshot {
    color: wgpu::TextureView,
    aux: wgpu::TextureView,
    /// The snapshot's residual half, copied on exactly the texels the color is.
    resid: Option<wgpu::TextureView>,
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

/// The bind groups one piece's dispatches switch between. Built once per piece
/// because every dispatch varies only the stamp uniform's dynamic offset; the
/// reservoir ping-pong is the one thing that needs a pair.
struct PieceBindings {
    snapshot: wgpu::BindGroup,
    exchange: [wgpu::BindGroup; 2],
    bake: [wgpu::BindGroup; 2],
    deposit: wgpu::BindGroup,
    /// The mobility pass's group — `Some` exactly when the piece allocated a bleed
    /// scratch, i.e. when some slot is a firing.
    bleed_weight: Option<wgpu::BindGroup>,
    /// The coarse pair's groups — `Some` exactly when the piece allocated a cell
    /// scratch, i.e. when some slot's [`extent_cell`](super::plan) beat 1.
    coarse: Option<CoarseBindings>,
    settle: wgpu::BindGroup,
    /// The warp's group (§6.13) — `Some` exactly when the run's tool is
    /// [`LoopTool::Liquify`], whose plan is the only source of warp slots.
    warp: Option<wgpu::BindGroup>,
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

/// The reservoir textures' shape — [`BRUSH_RES`]² of the tile color format, which
/// is what makes a [`ToolState`] copyable into and out of the loop's ping-pong.
const RESERVOIR_EXTENT: wgpu::Extent3d = wgpu::Extent3d {
    width: BRUSH_RES,
    height: BRUSH_RES,
    depth_or_array_layers: 1,
};
