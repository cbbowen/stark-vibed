//! The liquify path (§6.13): a stroke as a **homeomorphism**, kept as a displacement
//! field and applied to the picture once.
//!
//! The region machinery is the wet loop's — the same pieces, the same composite,
//! the same slice into copy-on-write tiles ([`run`](super::run)) — around three
//! kernels of its own (`liquify.wesl`): a snapshot of the field under each
//! segment's square, the composition of that segment's step into the field, and
//! one resample of the whole piece through the composed field from the run's
//! pristine base. A stroke of any length, and any number of strokes in a run, cost
//! the picture one clamped Catmull-Rom generation. The step reads the tip's own
//! coverage, through the **coverage prefix** the tip resolves for a liquify brush
//! ([`ResolvedTip::warp`](super::super::volume::ResolvedTip)) bound where every other
//! kernel binds its prefix-τ.
//!
//! Three things are decided on the CPU before a piece is drawn, and all three are
//! pure functions of the record and the run the stroke found — which is what lets a
//! live tail and the commit that replaces it agree to the bit (§1.3):
//!
//! - **whether the run is fresh** ([`LiquifyRun::is_fresh`]): the tiles inside the
//!   stroke's declared reach are what the run expects, or the stroke starts a run
//!   afresh from the picture;
//! - **the reach walk** ([`ReachWalk`]): per segment, a bound on how far the field
//!   under it may point after the step, and — where that bound would pass the cap —
//!   the segment before which the run **re-bases**, decided at a segment, never at
//!   a piece cut;
//! - **the pieces** ([`chunk_segments_within`]): the loop's chunker at the liquify
//!   budget, cut wherever the walk re-bases.

use std::collections::BTreeMap;
use std::rc::Rc;

use stark_model::document::{StrokeRecord, liquify_reads};
use stark_model::geom::{TILE_APRON, TILE_SIZE, TILE_TEX, TileCoord, TileRect, Vec2};

use super::super::budget::{
    LIQUIFY_BASE_SLACK, LIQUIFY_REACH_CAP, LIQUIFY_REGION_BUDGET_DIM, MAX_REGION_DIM,
};
use super::super::region::{chunk_segments_within, sweep_tiles};
use super::super::segments::Segment;
use super::super::{ResolvedRange, StrokeCarry, StrokeRenderer, StrokeScene};
use super::plan::{Plan, WarpStep, liquify_plan};
use super::region::{
    LOOP_USAGE, Piece, REGION_WRITES, RegionBox, RegionChannels, RegionRun, composite_tiles,
    region_selection, slice_region,
};
use super::trio::{Life, Trio};
use crate::document::PaintTiles;
use crate::document::liquify::LiquifyRun;
use crate::gpu::desc;
use crate::gpu::scratch::{Key, LeasedSlots, LeasedTex};
use crate::gpu::tile::{AllocSource, FIELD_FORMAT, TileMap};
use stark_shaders::mirror::dynamics_common::Stamp;
use stark_shaders::mirror::dynamics_common::binding as sb;
use stark_shaders::mirror::liquify::binding as lb;
use stark_shaders::mirror::liquify::decl as ld;

/// The most tiles a stroke's declared reach may span for its run to be kept
/// (§6.13). Past it the stroke re-bases every time — the picture is resampled per
/// stroke, as every other tool resamples — rather than recording a base for a rect
/// no run could compose over. A stroke that long is many thousands of px across.
const MAX_RUN_TILES: u64 = 4096;

/// **One texture, two declarations**, the region trio's case again (`kit`): the field a
/// piece composes is storage-written through `field_w` and copied to and from
/// [`FIELD_FORMAT`] tiles, and a copy between two formats is not a copy.
const _: () = assert!(
    matches!(FIELD_FORMAT, wgpu::TextureFormat::Rg32Float)
        && matches!(ld::FIELD_W.storage_format(), wgpu::TextureFormat::Rg32Float),
    "the liquify field is cut into `FIELD_FORMAT` tiles and storage-written through \
     `field_w`",
);

impl StrokeRenderer {
    /// Render a range of a liquify stroke (§6.13): compose its segments into the
    /// layer's run and resample the picture through the composed field, one
    /// region-sized piece at a time, and hand back the tiles with the run they
    /// are composed through.
    ///
    /// The range machinery is [`render_dynamic`](Self::render_dynamic)'s, but nothing
    /// is carried across ranges: the field *is* the state, and it rides the layer's
    /// run rather than a [`ToolState`](super::super::ToolState). A frozen head leaves
    /// its run on the state it hands the tail, and the tail composes into the same run
    /// a whole-stroke render would, because every decision here is a function of the
    /// record and the run the range found.
    pub(in crate::gpu::stroke) fn render_liquify(
        &self,
        scene: StrokeScene<'_>,
        range: ResolvedRange<'_>,
    ) -> (PaintTiles, StrokeCarry) {
        crate::timing::span!("stroke.dynamics");
        let ResolvedRange {
            rec,
            tip,
            segments,
            end_dist,
            ..
        } = range;
        // The run the stroke composes through: the layer's, where the tiles inside
        // the stroke's declared reach are still what it expects, and a fresh one from
        // the picture otherwise (`document::liquify`).
        let within = liquify_reads(rec);
        let kept = within.count() <= MAX_RUN_TILES;
        let mut run = match scene.liquify() {
            Some(run) if kept && run.is_fresh(scene.base(), within) => (**run).clone(),
            _ => LiquifyRun::default(),
        };
        // Recorded for **every** tile of the declared reach rather than for the tiles
        // this range's pieces happen to composite: the base's domain is then a function
        // of the record alone, so a head and its tail, a whole-stroke render and a peer
        // build the same run, and a later paint inside the reach is caught by the same
        // identities on all of them (§12.6).
        if kept {
            record_base(&mut run, scene.base(), within);
        }
        let walk = ReachWalk::over(&run, segments, rec);
        // The coverage prefix the follow reads (§6.13), resolved beside the tip's
        // other textures for exactly this brush — so its absence is a resolver
        // bug, not a brush a stroke could name.
        let warp_prefix = tip
            .warp
            .as_ref()
            .expect("a liquify brush resolves its coverage prefix (tips::resolve)");
        let mut draw = LiquifyDraw::new(self, scene, &range, run, warp_prefix);
        let mut map = scene.base().clone();
        let pieces = chunk_segments_within(segments, &[], LIQUIFY_REGION_BUDGET_DIM, &walk.cuts);
        for piece in pieces {
            // A piece the walk re-bases at starts a run afresh from the picture the
            // pieces before it left — the base under the walk's post-reset bounds.
            // The old run may be the only owner of tiles the previous piece's
            // recorded, unsubmitted commands read and write, so the scope holds it
            // until that submit rather than handing them back to the pool now.
            if walk.cuts.contains(&piece.start) {
                let old = std::mem::take(&mut draw.run);
                draw.region.scope.hold(old);
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
        let (run, carry) = draw.finish(end_dist);
        // A run past the tile cap is not kept: the tiles ride out under the identity,
        // and the next stroke resamples the picture as this one did.
        let run = if kept { run } else { LiquifyRun::default() };
        // The run rides out with the tiles it composed: the paint a layer holds
        // *is* that pair (§6.13).
        (PaintTiles::new(map, Some(Rc::new(run))), carry)
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
        // In [0, 1] already: the model holds it at every door a record comes through
        // (`LiquifyEffect::strength`), and the drag itself reads it unclamped.
        let strength = rec.brush.liquify().map_or(0.0, |l| l.strength);
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
    /// The region machinery this path shares with the wet loop (§6.2): the scene, the
    /// scope, the dirty set, the piece prologue and the epilogue. Its `prefix_bg` holds
    /// the **coverage** prefix, which the liquify path binds where every deposit binds
    /// its prefix-τ (§6.13).
    region: RegionRun<'a>,
    run: LiquifyRun,
}

impl<'a> LiquifyDraw<'a> {
    fn new(
        r: &'a StrokeRenderer,
        scene: StrokeScene<'a>,
        range: &ResolvedRange<'a>,
        run: LiquifyRun,
        warp_prefix: &wgpu::TextureView,
    ) -> Self {
        Self {
            region: RegionRun::new(r, scene, range, warp_prefix, "stark liquify stroke"),
            run,
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
        let Some(piece) = self.region.open(base, segments, &[]) else {
            return base.clone();
        };
        let bbox = piece.bbox();
        let r = self.region.r;
        // The base composite's margin, in whole texels, and clamped so the composite
        // fits a texture — which a brush this app can build never reaches
        // (`max_tip_reach`), but a record from elsewhere might.
        let margin = (margin.ceil() as u32 + LIQUIFY_BASE_SLACK)
            .min((MAX_REGION_DIM.saturating_sub(bbox.w.max(bbox.h))) / 2);
        let under = self.composite_base(base, bbox, margin);
        // The resample's targets, fully written by `warp_apply`.
        let region = Trio::take(
            &mut self.region.scope,
            Life::Piece,
            (bbox.w, bbox.h),
            LOOP_USAGE | wgpu::TextureUsages::COPY_SRC,
            REGION_WRITES,
            [
                "stark liquify region color",
                "stark liquify region aux",
                "stark liquify region resid",
            ],
            r.color_space.has_resid(),
        );
        let field = self.lay_field(&piece, bbox);
        let selection = self.region.scene.selection;
        let sel_mask = region_selection(r, &mut self.region.scope, selection, piece.halo(), bbox);

        let (plan, stamps, bind) = self.bind(
            segments,
            bbox,
            margin,
            PieceTargets {
                field: &field.view,
                under: &under,
                region: &region,
                sel_mask: &sel_mask,
            },
        );

        // ---- The kernels, in order: per segment, the field's snapshot then the
        // composition; then the one resample.
        {
            crate::timing::span!("stroke.loop");
            let kernels = &r.dynamics.warp;
            let mut pass = self.region.loop_pass("stark liquify field");
            for (i, dsp) in plan.slots.iter().enumerate() {
                let off = stamps.offset(i);
                match dsp.step {
                    WarpStep::Warp => {
                        // The snapshot covers the whole square, since the gather pulls
                        // from upstream of any one texel.
                        let square = (plan.dsize, plan.dsize);
                        kernels
                            .snapshot_field
                            .dispatch(&mut pass, &bind.snapshot, off, square);
                        kernels
                            .warp
                            .dispatch(&mut pass, &bind.warp, off, dsp.extent);
                    }
                    WarpStep::Apply => {
                        kernels
                            .apply
                            .dispatch(&mut pass, &bind.apply, off, dsp.extent);
                    }
                }
            }
        }
        // ---- Slice the picture and the field back into tiles, and record both.
        crate::timing::span!("stroke.writeback");
        let new_map = slice_region(
            r,
            &mut self.region.scope,
            self.region.scene.pool,
            base,
            &piece,
            RegionChannels {
                color_tex: &region.color.tex,
                aux: &region.aux.view,
                resid_tex: region.resid_tex(),
            },
        );
        self.write_field(&piece, &field.tex, &new_map, reach);
        new_map
    }

    /// The base composite (§6.13): the run's base under the region, grown by `margin`
    /// px on every side so every `x + d(x)` the resample follows lands inside it.
    ///
    /// What it draws is the run's base where the run has read the tile and the layer's
    /// own tile where it has not — a run past the tile cap records nothing, and reads
    /// the picture.
    fn composite_base(&mut self, base: &TileMap, bbox: RegionBox, margin: u32) -> Trio {
        let r = self.region.r;
        let grown = RegionBox {
            origin: bbox.origin - Vec2::splat(margin as f32),
            w: bbox.w + 2 * margin,
            h: bbox.h + 2 * margin,
        };
        let tiles = tiles_over(grown.origin, grown.w, grown.h);
        let mut map = TileMap::new();
        for c in &tiles {
            let tile = match self.run.base_at(*c) {
                Some(t) => t.cloned(),
                None => base.get(c).cloned(),
            };
            if let Some(t) = tile {
                map = map.insert(*c, t);
            }
        }
        // A region composite's destination, so the region's own formats
        // ([`REGION_WRITES`]) — it is only ever sampled, and a sampled declaration
        // names no format for it to take instead.
        let into = Trio::take(
            &mut self.region.scope,
            Life::Piece,
            (grown.w, grown.h),
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            REGION_WRITES,
            [
                "stark liquify base color",
                "stark liquify base aux",
                "stark liquify base resid",
            ],
            r.color_space.has_resid(),
        );
        crate::timing::span!("stroke.region");
        composite_tiles(
            r,
            &mut self.region.scope,
            &map,
            &tiles,
            grown,
            into.targets(),
        );
        into
    }

    /// The field over the region: cleared to the identity, then the run's own field
    /// tiles laid in, each clipped to the region rect (§6.13).
    fn lay_field(&mut self, piece: &Piece, bbox: RegionBox) -> LeasedTex {
        let RegionBox { origin, w, h } = bbox;
        let field = self.region.scope.take_piece(Key {
            size: (w, h),
            // The tiles the field is cut into are `FIELD_FORMAT`, and a copy between
            // two formats is not a copy; `field_w` is what the composition writes it
            // through.
            format: FIELD_FORMAT,
            usage: LOOP_USAGE
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            label: "stark liquify region field",
        });
        self.region
            .scope
            .encoder()
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stark liquify field clear"),
                color_attachments: &[Some(desc::attach(&field.view, desc::CLEAR))],
                ..Default::default()
            });
        let region_hi = origin + Vec2::new(w as f32, h as f32);
        for c in piece.halo() {
            let Some(tile) = self.run.field_at(*c) else {
                continue;
            };
            // The tile's block on the canvas, clipped to the region rect.
            let block_lo = c.origin() - Vec2::splat(TILE_APRON as f32);
            let block_hi = block_lo + Vec2::splat(TILE_TEX as f32);
            let lo = block_lo.max(origin);
            let hi = block_hi.min(region_hi);
            if lo.x >= hi.x || lo.y >= hi.y {
                continue;
            }
            tile.copy_block_out(
                self.region.scope.encoder(),
                texel_origin(lo - block_lo),
                &field.tex,
                texel_origin(lo - origin),
                texel_extent(hi - lo),
            );
        }
        field
    }

    /// The piece's plan, its uniform slots, and the three groups its kernels dispatch
    /// with — the field snapshot's scratch, which the plan sizes, among them.
    fn bind(
        &mut self,
        segments: &[Segment],
        bbox: RegionBox,
        margin: u32,
        over: PieceTargets<'_>,
    ) -> (Plan<WarpStep>, LeasedSlots<Stamp>, Bindings) {
        crate::timing::span!("stroke.plan");
        let PieceTargets {
            field,
            under,
            region,
            sel_mask,
        } = over;
        let r = self.region.r;
        let device = &r.ctx.device;
        let kernels = &r.dynamics.warp;
        let ctx = self.region.plan_ctx(bbox.origin);
        let plan = liquify_plan(
            &ctx,
            segments,
            [margin as i32, margin as i32],
            (bbox.w, bbox.h),
        );
        let under_field = self
            .region
            .scope
            .take_piece(Key {
                size: (plan.dsize, plan.dsize),
                format: ld::UNDER_FIELD_W.storage_format(),
                usage: LOOP_USAGE,
                label: "stark liquify under field",
            })
            .view;
        let stamps = self.region.upload(&plan.slots);
        let params = stamps.window();
        fn view(v: &wgpu::TextureView) -> wgpu::BindingResource<'_> {
            wgpu::BindingResource::TextureView(v)
        }
        let snapshot =
            kernels
                .snapshot_field
                .group(device, "stark liquify snapshot field bg", |s| match s {
                    sb::ST => params.clone(),
                    lb::FIELD => view(field),
                    lb::UNDER_FIELD_W => view(&under_field),
                    other => unreachable!("snapshot_field reads no binding {other}"),
                });
        let warp = kernels
            .warp
            .group(device, "stark liquify warp bg", |s| match s {
                sb::ST => params.clone(),
                lb::UNDER_FIELD => view(&under_field),
                lb::FIELD_W => view(field),
                sb::SEL_MASK => view(sel_mask),
                other => unreachable!("warp reads no binding {other}"),
            });
        let apply = kernels
            .apply
            .group(device, "stark liquify warp apply bg", |s| match s {
                sb::ST => params.clone(),
                lb::FIELD => view(field),
                lb::BASE_COLOR => view(&under.color.view),
                lb::BASE_AUX => view(&under.aux.view),
                lb::BASE_RESID => view(under.resid_view().expect("a residual build")),
                sb::REGION_COLOR_W => view(&region.color.view),
                sb::REGION_AUX_W => view(&region.aux.view),
                sb::REGION_RESID_W => view(region.resid_view().expect("a residual build")),
                other => unreachable!("warp_apply reads no binding {other}"),
            });
        (
            plan,
            stamps,
            Bindings {
                snapshot,
                warp,
                apply,
            },
        )
    }

    /// Cut the composed field into tiles and record each of them in the run beside the
    /// picture `painted` sliced out, under the bound `reach` gives it (§6.13).
    fn write_field(
        &mut self,
        piece: &Piece,
        field: &wgpu::Texture,
        painted: &TileMap,
        reach: &BTreeMap<TileCoord, f32>,
    ) {
        let pool = self.region.scene.pool;
        let mut displaced = Vec::with_capacity(piece.tiles.len());
        for coord in &piece.tiles {
            let tile = pool.acquire_tex(FIELD_FORMAT, AllocSource::LiquifyField);
            tile.copy_into(
                self.region.scope.encoder(),
                field,
                piece.block_origin(*coord),
            );
            let picture = painted
                .get(coord)
                .cloned()
                .expect("the slice wrote every covered tile");
            let bound = reach.get(coord).copied().unwrap_or(0.0);
            displaced.push(self.run.record_written(*coord, picture, tile, bound));
        }
        // The halo copy above read the field tiles these replace.
        self.region.scope.hold(displaced);
    }

    /// Close the draw: hand the region machinery what is owed a submit, and hand back
    /// the run the pieces built beside what the range carries on. Nothing is carried
    /// across ranges here — the field *is* the state, and it rides the run (§6.13).
    fn finish(self, end_dist: f32) -> (LiquifyRun, StrokeCarry) {
        let Self { region, run } = self;
        (run, region.finish(end_dist, None, ()))
    }
}

/// The three bind groups a liquify piece dispatches with (§6.13).
struct Bindings {
    snapshot: wgpu::BindGroup,
    warp: wgpu::BindGroup,
    apply: wgpu::BindGroup,
}

/// What one liquify piece's kernels read and write (§6.13): the field they compose,
/// the pristine base they resample *from*, the region they resample *into*, and the
/// selection the composition is gated by.
struct PieceTargets<'p> {
    field: &'p wgpu::TextureView,
    under: &'p Trio,
    region: &'p Trio,
    sel_mask: &'p wgpu::TextureView,
}

/// A canvas-px offset as a copy's texel origin, and a canvas-px size as its extent.
/// Integral by construction: every corner measured here is a tile block's corner or a
/// region rect's, and both sit on the texel grid.
fn texel_origin(v: Vec2) -> wgpu::Origin3d {
    wgpu::Origin3d {
        x: v.x as u32,
        y: v.y as u32,
        z: 0,
    }
}

fn texel_extent(v: Vec2) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: v.x as u32,
        height: v.y as u32,
        depth_or_array_layers: 1,
    }
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
