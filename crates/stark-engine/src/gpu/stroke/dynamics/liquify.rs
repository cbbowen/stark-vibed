//! The liquify path (§6.13): a stroke as a **homeomorphism**, kept as a displacement
//! field and applied to the picture once.
//!
//! The region machinery is the wet loop's — the same pieces, the same composite,
//! the same slice into copy-on-write tiles ([`run`](super::run)) — around three
//! kernels of its own (`dynamics.wesl`): a snapshot of the field under each
//! segment's square, the composition of that segment's step into the field, and
//! one resample of the whole piece through the composed field from the run's
//! pristine base. What the picture pays for a stroke of any length, and for any
//! number of strokes in a run, is one clamped Catmull-Rom generation.
//!
//! Three things are decided on the CPU before a piece is drawn, and all three are
//! pure functions of the record and the run the stroke found — which is what lets a
//! live tail and the commit that replaces it agree to the bit (§1.3):
//!
//! - **whether the run is fresh** ([`LiquifyRun::is_fresh`]): the tiles inside the
//!   stroke's declared reach are what the run expects, or the stroke starts a run
//!   afresh from the picture;
//! - **the reach walk** ([`ReachWalk`]): per segment, a bound on how far the field
//!   under it may point after the step, from the run's per-tile bounds and the
//!   travel since; and, where that bound would pass the cap, the segment before
//!   which the run **re-bases** — decided at a segment, never at a piece cut;
//! - **the pieces** ([`chunk_segments_within`]): the loop's chunker at the liquify
//!   budget, cut wherever the walk re-bases.

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use stark_model::document::{StrokeRecord, liquify_reads};
use stark_model::geom::{TILE_APRON, TILE_SIZE, TILE_TEX, TileCoord, TileRect, Vec2};

use super::super::budget::{
    LIQUIFY_BASE_SLACK, LIQUIFY_REACH_CAP, LIQUIFY_REGION_BUDGET_DIM, MAX_REGION_DIM,
    liquify_plateau,
};
use super::super::region::{RegionRect, chunk_segments_within, cover, sweep_tiles};
use super::super::segments::Segment;
use super::super::{Painted, Progress, ResolvedRange, StrokeCarry, StrokeRenderer, StrokeScene};
use super::plan::{PlanCtx, SlotKind, groups_for, liquify_plan};
use super::run::{
    LOOP_USAGE, RegionBox, RegionChannels, composite_tiles, region_selection, slice_region,
    upload_stamps,
};
use super::slots;
use crate::document::liquify::LiquifyRun;
use crate::gpu::channels::Targets;
use crate::gpu::desc;
use crate::gpu::scratch::{Key, SubmitScope};
use crate::gpu::tile::{AllocSource, FIELD_FORMAT, TileMap};
use crate::gpu::uniforms::UniformSlots;
use stark_shaders::mirror::dynamics::binding as b;
use stark_shaders::mirror::dynamics::decl as d;

/// The most tiles a stroke's declared reach may span for its run to be kept
/// (§6.13). Past it the stroke re-bases every time — the picture is resampled per
/// stroke, as every other tool resamples — rather than recording a base for a rect
/// no run could compose over. A stroke that long is many thousands of px across.
const MAX_RUN_TILES: u64 = 4096;

impl StrokeRenderer {
    /// Render a range of a liquify stroke (§6.13): compose its segments into the
    /// layer's run and resample the picture through the composed field, one
    /// region-sized piece at a time, and hand back the tiles with the run they
    /// are composed through.
    ///
    /// The range machinery is [`render_dynamic`](Self::render_dynamic)'s, and it
    /// carries nothing across ranges of its own: the field *is* the state, and it
    /// rides the layer's run rather than a [`ToolState`](super::super::ToolState).
    /// A frozen head leaves its run on the state it hands the tail, and the tail
    /// composes into it — which is the same run a whole-stroke render composes
    /// into, because every decision here is a function of the record and the run
    /// the range found.
    pub(in crate::gpu::stroke) fn render_liquify(
        &self,
        scene: StrokeScene<'_>,
        range: ResolvedRange<'_>,
    ) -> (Painted, StrokeCarry) {
        crate::timing::span!("stroke.dynamics");
        let ResolvedRange {
            rec,
            segments,
            end_dist,
            tol,
            consts,
            ..
        } = range;
        // The run the stroke composes through: the layer's, where the tiles inside
        // the stroke's declared reach are still what it expects, and a fresh one from
        // the picture otherwise (`document::liquify`).
        let within = liquify_reads(rec);
        let kept = within.count() <= MAX_RUN_TILES;
        let mut run = match scene.liquify {
            Some(run) if kept && run.is_fresh(scene.base, within) => (**run).clone(),
            _ => LiquifyRun::fresh(),
        };
        // What the run reads, recorded for **every** tile of the declared reach
        // rather than for the tiles this range's pieces happen to composite: the
        // domain of the base is then a function of the record alone, so a head and
        // its tail, a whole-stroke render and a peer all build the same run — and a
        // later paint inside the reach is caught by the same identities on every
        // one of them (§12.6).
        if kept {
            record_base(&mut run, scene.base, within);
        }
        let walk = ReachWalk::over(&run, segments, rec);
        let mut draw = LiquifyDraw::new(self, scene, rec, tol, consts, run);
        let mut map = scene.base.clone();
        let pieces = chunk_segments_within(segments, &[], LIQUIFY_REGION_BUDGET_DIM, &walk.cuts);
        for piece in pieces {
            // A piece the walk re-bases at starts a run afresh from the picture the
            // pieces before it left — the base under the walk's post-reset bounds.
            if walk.cuts.contains(&piece.start) {
                draw.run = LiquifyRun::fresh();
                if kept {
                    record_base(&mut draw.run, &map, within);
                }
            }
            let margin = walk.after[piece.clone()]
                .iter()
                .copied()
                .fold(0.0f32, f32::max);
            map = draw.piece(
                &map,
                &segments[piece.clone()],
                margin,
                &walk.reach_by(piece.end),
            );
        }
        let dirty = std::mem::take(&mut draw.dirty).into_iter().collect();
        let run = draw.finish();
        // A run past the tile cap is not kept: the tiles ride out under the identity,
        // and the next stroke resamples the picture as this one did.
        let run = if kept { run } else { LiquifyRun::fresh() };
        (
            Painted {
                tiles: map,
                liquify: Some(Rc::new(run)),
            },
            StrokeCarry {
                dist: end_dist,
                progress: Progress::Finished { tool: None, dirty },
            },
        )
    }
}

/// Record what the run reads at every tile of `within` it has not read before: the
/// layer's tile there now, absent included.
fn record_base(run: &mut LiquifyRun, tiles: &TileMap, within: TileRect) {
    for c in within.coords() {
        run.record_base(c, tiles.get(&c).cloned());
    }
}

/// The reach walk (§6.13): per segment, a bound on the displacement under it after
/// its step, and the segments before which the run re-bases.
///
/// The step at a texel pulls from at most one pass upstream, and the field there
/// was bounded by the tiles' own bounds — so after the segment the field under its
/// tiles is bounded by the largest bound over the tiles within a pass of them, plus
/// the pass. Coarse, and deliberately so: it is a pure function of the record and
/// the run's bounds, evaluated segment by segment, so a head, its tail and a
/// whole-stroke render walk the same numbers and re-base at the same segment.
struct ReachWalk {
    /// Segment indices at which the run re-bases, ascending.
    cuts: Vec<usize>,
    /// Per segment, the bound after it over the tiles it reaches, canvas px.
    after: Vec<f32>,
    /// Per segment, the tiles it reaches ([`sweep_tiles`]).
    tiles: Vec<Vec<TileCoord>>,
}

impl ReachWalk {
    fn over(run: &LiquifyRun, segments: &[Segment], rec: &StrokeRecord) -> Self {
        let strength = rec
            .brush
            .liquify()
            .map_or(0.0, |l| l.strength.clamp(0.0, 1.0));
        let mut cuts = Vec::new();
        let mut after = Vec::with_capacity(segments.len());
        let mut tiles = Vec::with_capacity(segments.len());
        // The bounds this stroke has set so far, over the run's own; cleared at a
        // re-base, after which the run's bounds no longer apply either.
        let mut overlay: BTreeMap<TileCoord, f32> = BTreeMap::new();
        let mut since_reset = false;
        for (k, s) in segments.iter().enumerate() {
            let own = sweep_tiles(&s.sweep);
            // A modulation only scales the strength down, so the brush's own bounds
            // the segment's; the pass it can pull from is its length.
            let step = s.sweep.length * strength;
            let rings = (step / TILE_SIZE as f32).ceil() as i32 + 1;
            let mut r_in = 0.0f32;
            for c in &own {
                for dy in -rings..=rings {
                    for dx in -rings..=rings {
                        let n = TileCoord::new(c.x + dx, c.y + dy);
                        let was = if since_reset { 0.0 } else { run.reach_at(n) };
                        r_in = r_in.max(was).max(overlay.get(&n).copied().unwrap_or(0.0));
                    }
                }
            }
            // Past the cap, the run re-bases before this segment — at the first
            // segment too, when the run a stroke found already stands at the cap
            // under it: the stroke then starts afresh from the picture.
            if r_in + step > LIQUIFY_REACH_CAP {
                cuts.push(k);
                overlay.clear();
                since_reset = true;
                r_in = 0.0;
            }
            let bound = r_in + step;
            for c in &own {
                let e = overlay.entry(*c).or_insert(0.0);
                *e = e.max(bound);
            }
            after.push(bound);
            tiles.push(own);
        }
        Self { cuts, after, tiles }
    }

    /// The bound to record on each tile after the segments before `end` — the
    /// largest bound of any segment before it that reaches the tile, since the last
    /// re-base before `end`.
    fn reach_by(&self, end: usize) -> BTreeMap<TileCoord, f32> {
        let from = self
            .cuts
            .iter()
            .rev()
            .find(|c| **c < end)
            .copied()
            .unwrap_or(0);
        let mut out = BTreeMap::new();
        for k in from..end {
            for c in &self.tiles[k] {
                let e = out.entry(*c).or_insert(0.0f32);
                *e = e.max(self.after[k]);
            }
        }
        out
    }
}

/// One [`render_liquify`](StrokeRenderer::render_liquify) call in progress: the run
/// as the pieces evolve it, and the GPU objects that outlive any one piece.
struct LiquifyDraw<'a> {
    r: &'a StrokeRenderer,
    scene: StrokeScene<'a>,
    rec: &'a StrokeRecord,
    tol: crate::path::FlattenTolerance,
    consts: &'a super::super::StrokeConstants,
    scope: SubmitScope,
    run: LiquifyRun,
    dirty: BTreeSet<TileCoord>,
    /// The plan's staging bytes, kept across pieces (`DynamicsRun::stamps`).
    stamps: Vec<u8>,
    plateau: f32,
}

impl<'a> LiquifyDraw<'a> {
    fn new(
        r: &'a StrokeRenderer,
        scene: StrokeScene<'a>,
        rec: &'a StrokeRecord,
        tol: crate::path::FlattenTolerance,
        consts: &'a super::super::StrokeConstants,
        run: LiquifyRun,
    ) -> Self {
        Self {
            r,
            scene,
            rec,
            tol,
            consts,
            scope: r.scratch.scope(&r.ctx, "stark liquify stroke"),
            run,
            dirty: BTreeSet::new(),
            stamps: Vec::new(),
            plateau: liquify_plateau(&rec.brush.shape),
        }
    }

    /// Draw one region-sized piece over `base`: composite the run's base under it
    /// (grown by `margin` px, the walk's bound over the piece), lay the run's field
    /// over the region, compose the piece's segments into the field, resample the
    /// region through it, and slice both the picture and the field back into fresh
    /// tiles — recording each in the run with the bound `reach` gives it.
    fn piece(
        &mut self,
        base: &TileMap,
        segments: &[Segment],
        margin: f32,
        reach: &BTreeMap<TileCoord, f32>,
    ) -> TileMap {
        crate::timing::span!("stroke.piece");
        self.scope.flush();
        let covered = cover(segments, &[]);
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
        self.dirty.extend(coords.iter().copied());
        self.scope.hold(base.clone());
        let r = self.r;
        let kit = &r.dynamics;
        let device = &r.ctx.device;
        let resid = r.color_space.has_resid();

        // ---- The base composite: the run's base under the region, grown by the
        // margin so every `x + d(x)` lands inside it. The margin is whole texels, and
        // clamped so the composite fits a texture — which a brush this app can build
        // never reaches (`max_tip_reach`), but a record from elsewhere might.
        let margin = (margin.ceil() as u32 + LIQUIFY_BASE_SLACK)
            .min((MAX_REGION_DIM.saturating_sub(w.max(h))) / 2);
        let (bw, bh) = (w + 2 * margin, h + 2 * margin);
        let base_origin = region_origin - Vec2::splat(margin as f32);
        let base_tiles = tiles_over(base_origin, bw, bh);
        // What the composite draws: the run's base where the run has read the tile,
        // the layer's own tile where it has not (a run past the tile cap records
        // nothing, and reads the picture).
        let mut base_map = TileMap::new();
        for c in &base_tiles {
            let tile = match self.run.base_at(*c) {
                Some(t) => t.cloned(),
                None => base.get(c).cloned(),
            };
            if let Some(t) = tile {
                base_map = base_map.insert(*c, t);
            }
        }
        let base_usage =
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let mut base_tex = |label: &'static str| {
            self.scope
                .take_piece(Key {
                    size: (bw, bh),
                    format: d::REGION_COLOR_W.storage_format(),
                    usage: base_usage,
                    label,
                })
                .1
        };
        let base_color = base_tex("stark liquify base color");
        let base_aux = base_tex("stark liquify base aux");
        let base_resid = resid.then(|| base_tex("stark liquify base resid"));
        {
            crate::timing::span!("stroke.region");
            composite_tiles(
                r,
                &mut self.scope,
                &base_map,
                &base_tiles,
                RegionBox {
                    origin: base_origin,
                    w: bw,
                    h: bh,
                },
                Targets {
                    color: &base_color,
                    aux: &base_aux,
                    resid: base_resid.as_ref(),
                },
            );
        }

        // ---- The region: the resample's targets, fully written by `warp_apply`, and
        // the field over it — cleared to the identity, then the run's field tiles laid
        // in, clipped to the region.
        let region_usage = LOOP_USAGE | wgpu::TextureUsages::COPY_SRC;
        let mut region_tex = |label: &'static str| {
            self.scope.take_piece(Key {
                size: (w, h),
                format: d::REGION_COLOR_W.storage_format(),
                usage: region_usage,
                label,
            })
        };
        let (color_tex, color) = region_tex("stark liquify region color");
        let (_aux_tex, aux) = region_tex("stark liquify region aux");
        let (resid_tex, resid_view) = resid
            .then(|| region_tex("stark liquify region resid"))
            .unzip();
        let (field_tex, field) = self.scope.take_piece(Key {
            size: (w, h),
            format: FIELD_FORMAT,
            usage: LOOP_USAGE
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            label: "stark liquify region field",
        });
        self.scope
            .encoder()
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stark liquify field clear"),
                color_attachments: &[Some(desc::attach(&field, desc::CLEAR))],
                ..Default::default()
            });
        for c in &halo {
            let Some(tile) = self.run.field_at(*c) else {
                continue;
            };
            // The tile's block on the canvas, clipped to the region rect.
            let block_lo = c.origin() - Vec2::splat(TILE_APRON as f32);
            let block_hi = block_lo + Vec2::splat(TILE_TEX as f32);
            let region_hi = region_origin + Vec2::new(w as f32, h as f32);
            let lo = block_lo.max(region_origin);
            let hi = block_hi.min(region_hi);
            if lo.x >= hi.x || lo.y >= hi.y {
                continue;
            }
            let src = lo - block_lo;
            let dst = lo - region_origin;
            let extent = hi - lo;
            tile.copy_block_out(
                self.scope.encoder(),
                wgpu::Origin3d {
                    x: src.x as u32,
                    y: src.y as u32,
                    z: 0,
                },
                &field_tex,
                wgpu::Origin3d {
                    x: dst.x as u32,
                    y: dst.y as u32,
                    z: 0,
                },
                wgpu::Extent3d {
                    width: extent.x as u32,
                    height: extent.y as u32,
                    depth_or_array_layers: 1,
                },
            );
        }
        let sel_mask = region_selection(
            r,
            &mut self.scope,
            self.scene.selection,
            &halo,
            RegionBox {
                origin: region_origin,
                w,
                h,
            },
        );

        // ---- The plan, the scratch it sizes, and the groups that name it.
        let (plan, bind) = {
            crate::timing::span!("stroke.plan");
            let ctx = PlanCtx {
                rec: self.rec,
                tol: self.tol,
                region_origin,
                consts: self.consts,
                substrate: self.scene.substrate,
            };
            let plan = liquify_plan(
                &ctx,
                segments,
                self.plateau,
                [margin as i32, margin as i32],
                (w, h),
            );
            let under_field = self
                .scope
                .take_piece(Key {
                    size: (plan.dsize, plan.dsize),
                    format: FIELD_FORMAT,
                    usage: LOOP_USAGE,
                    label: "stark liquify under field",
                })
                .1;
            let stamp_buf = upload_stamps(&mut self.scope, &mut self.stamps, &plan.slots);
            let params = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &stamp_buf,
                offset: 0,
                size: wgpu::BufferSize::new(super::plan::SLOT as u64),
            });
            fn view(v: &wgpu::TextureView) -> wgpu::BindingResource<'_> {
                wgpu::BindingResource::TextureView(v)
            }
            let snapshot = desc::bind_group_for(
                device,
                "stark liquify snapshot field bg",
                &kit.snapshot_field_bgl,
                slots::SNAPSHOT_FIELD,
                resid,
                |s| match s {
                    b::ST => params.clone(),
                    b::FIELD => view(&field),
                    b::UNDER_FIELD_W => view(&under_field),
                    other => unreachable!("snapshot_field lists no binding {other}"),
                },
            );
            let warp = desc::bind_group_for(
                device,
                "stark liquify warp bg",
                &kit.warp_bgl,
                slots::WARP,
                resid,
                |s| match s {
                    b::ST => params.clone(),
                    b::UNDER_FIELD => view(&under_field),
                    b::FIELD_W => view(&field),
                    b::SEL_MASK => view(&sel_mask),
                    other => unreachable!("warp lists no binding {other}"),
                },
            );
            let apply = desc::bind_group_for(
                device,
                "stark liquify warp apply bg",
                &kit.warp_apply_bgl,
                slots::WARP_APPLY,
                resid,
                |s| match s {
                    b::ST => params.clone(),
                    b::FIELD => view(&field),
                    b::BASE_COLOR => view(&base_color),
                    b::BASE_AUX => view(&base_aux),
                    b::BASE_RESID => view(base_resid.as_ref().expect("a residual build")),
                    b::REGION_COLOR_W => view(&color),
                    b::REGION_AUX_W => view(&aux),
                    b::REGION_RESID_W => view(resid_view.as_ref().expect("a residual build")),
                    other => unreachable!("warp_apply lists no binding {other}"),
                },
            );
            (
                plan,
                Bindings {
                    snapshot,
                    warp,
                    apply,
                },
            )
        };

        // ---- The kernels, in order: per segment, the field's snapshot then the
        // composition; then the one resample.
        {
            crate::timing::span!("stroke.loop");
            let square = groups_for(plan.dsize);
            let mut cpass = self
                .scope
                .encoder()
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("stark liquify field"),
                    timestamp_writes: None,
                });
            for (i, dsp) in plan.slots.iter().enumerate() {
                let off = UniformSlots::<stark_shaders::mirror::dynamics::Stamp>::offset(i as u32);
                match dsp.kind {
                    SlotKind::Warp => {
                        cpass.set_pipeline(&kit.snapshot_field_pipeline);
                        cpass.set_bind_group(0, &bind.snapshot, &[off]);
                        cpass.dispatch_workgroups(square, square, 1);
                        cpass.set_pipeline(&kit.warp_pipeline);
                        cpass.set_bind_group(0, &bind.warp, &[off]);
                        cpass.dispatch_workgroups(dsp.groups.0, dsp.groups.1, 1);
                    }
                    SlotKind::WarpApply => {
                        cpass.set_pipeline(&kit.warp_apply_pipeline);
                        cpass.set_bind_group(0, &bind.apply, &[off]);
                        cpass.dispatch_workgroups(dsp.groups.0, dsp.groups.1, 1);
                    }
                    SlotKind::Segment | SlotKind::Bleed | SlotKind::Settle => {
                        unreachable!(
                            "a wet slot in a liquify plan; the plan is the effect's (§6.2)"
                        )
                    }
                }
            }
        }
        // ---- Slice the picture and the field back into tiles, and record both.
        crate::timing::span!("stroke.writeback");
        let new_map = slice_region(
            r,
            &mut self.scope,
            self.scene.pool,
            base,
            coords,
            lo,
            RegionChannels {
                color_tex: &color_tex,
                aux: &aux,
                resid_tex: resid_tex.as_ref(),
            },
        );
        for coord in coords {
            let field_tile = self
                .scene
                .pool
                .acquire_tex(FIELD_FORMAT, AllocSource::LiquifyField);
            let off = coord.origin() - lo;
            field_tile.copy_into(
                self.scope.encoder(),
                &field_tex,
                wgpu::Origin3d {
                    x: off.x as u32,
                    y: off.y as u32,
                    z: 0,
                },
            );
            let picture = new_map
                .get(coord)
                .cloned()
                .expect("the slice wrote every covered tile");
            let bound = reach.get(coord).copied().unwrap_or(0.0);
            self.run.record_written(*coord, picture, field_tile, bound);
        }
        new_map
    }

    /// Close the draw: submit what is recorded, release what the submit frees, and
    /// hand back the run the pieces built.
    fn finish(self) -> LiquifyRun {
        let Self { scope, run, .. } = self;
        {
            crate::timing::span!("stroke.submit");
            scope.finish();
        }
        run
    }
}

/// The three bind groups a liquify piece dispatches with (§6.13).
struct Bindings {
    snapshot: wgpu::BindGroup,
    warp: wgpu::BindGroup,
    apply: wgpu::BindGroup,
}

/// Every tile whose texture block overlaps the box at `origin` of extent `w × h`,
/// in canvas px — the tiles a base composite over that box draws.
fn tiles_over(origin: Vec2, w: u32, h: u32) -> Vec<TileCoord> {
    let hi = origin + Vec2::new(w as f32, h as f32);
    // A block reaches an apron past its interior on every side, so a tile whose
    // interior ends an apron short of the box still overlaps it.
    let apron = TILE_APRON as f32;
    let tile = TILE_SIZE as f32;
    let (x0, x1) = (
        ((origin.x - apron) / tile).floor() as i32,
        ((hi.x + apron - 1.0) / tile).floor() as i32,
    );
    let (y0, y1) = (
        ((origin.y - apron) / tile).floor() as i32,
        ((hi.y + apron - 1.0) / tile).floor() as i32,
    );
    let mut out = Vec::with_capacity(((x1 - x0 + 1) * (y1 - y0 + 1)).max(0) as usize);
    for y in y0..=y1 {
        for x in x0..=x1 {
            out.push(TileCoord::new(x, y));
        }
    }
    out
}
