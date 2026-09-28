//! The **region machinery** both loop paths run: one piece of a stroke opened over a
//! 1:1 canvas region, evolved there, and sliced back into copy-on-write tiles (§6.2,
//! §6.13).
//!
//! [`RegionRun`] is everything a wet run and a liquify draw hold in common — the
//! renderer, the scene, the record, the flatten budget, the stroke constants, the
//! submit scope, the dirty set and the group-1 prefix — plus the piece prologue and the
//! epilogue that read them. Each path embeds one and adds
//! only what is its own: the reservoir ping-pong and the carried lanes on one side
//! ([`run`](super::run)), the displacement field and its run on the other
//! ([`liquify`](super::liquify)).
//!
//! **Ordering lives here.** Two hazards sit in the prologue and the epilogue, and both
//! are the same shape: a handle whose drop *is* a release to a pool, dropped before the
//! submit of commands that still name it. [`RegionRun::open`] holds the base map the
//! piece is about to composite from; [`RegionRun::finish`] holds whatever the caller's
//! last piece left named by unsubmitted copies. A path that wrote its own prologue
//! could forget either, and nothing but a corrupted picture would say so.
//!
//! Which tiles a piece covers and where the cuts fall is
//! [`region`](super::super::region)'s — float arithmetic over sweeps, no GPU. This is
//! the half that records.

use std::collections::BTreeSet;

use stark_model::document::StrokeRecord;
use stark_model::geom::{TileCoord, Vec2};

use super::super::region::{RegionRect, cover};
use super::super::segments::{BleedFire, Segment};
use super::super::{ResolvedRange, StrokeCarry, StrokeRenderer, StrokeScene, ToolState};
use super::kit::{LoopPass, Prefix};
use super::plan::{LoopDispatch, PlanCtx};
use crate::gpu::channels::{TILE_AUX_FORMAT, Targets};
use crate::gpu::composite::view_uniform;
use crate::gpu::desc;
use crate::gpu::scratch::{BufKey, Key, LeasedSlots, LeasedTex, SubmitScope};
use crate::gpu::tile::{AllocSource, TileMap};
// The region composite runs `composite.wesl`, so it draws that shader's own
// per-instance record (§6.10) rather than a second `#[repr(C)]` copy of it declared
// here under a different name.
use stark_shaders::mirror::composite::Instance as TileInstance;
use stark_shaders::mirror::dynamics_common::Stamp;
use stark_shaders::mirror::dynamics_common::decl as sd;
use stark_shaders::mirror::view::binding as vb;

/// What every texture the loop reads and writes needs to be: sampled by one
/// dispatch, storage-written by the next.
pub(super) const LOOP_USAGE: wgpu::TextureUsages =
    wgpu::TextureUsages::TEXTURE_BINDING.union(wgpu::TextureUsages::STORAGE_BINDING);

/// The three declarations the loop storage-writes a region's channels through — the
/// formats a region trio, and anything a region is composited into, is created at
/// (§6.10, and the `kit` assertion that ties them to the composite's own targets).
pub(super) const REGION_WRITES: [stark_shaders::Binding; 3] =
    [sd::REGION_COLOR_W, sd::REGION_AUX_W, sd::REGION_RESID_W];

/// Where a region sits and how large it is: its top-left in canvas px and its extent
/// in texels — the three numbers every region helper below takes together.
#[derive(Clone, Copy)]
pub(super) struct RegionBox {
    pub(super) origin: Vec2,
    pub(super) w: u32,
    pub(super) h: u32,
}

/// The channel textures a region's tiles are sliced out of ([`slice_region`]): the
/// color and residual by texture, since a copy is addressed by texture, and the wide
/// aux by view, since the narrow pass samples it.
pub(super) struct RegionChannels<'a> {
    pub(super) color_tex: &'a wgpu::Texture,
    pub(super) aux: &'a wgpu::TextureView,
    pub(super) resid_tex: Option<&'a wgpu::Texture>,
}

/// One piece of a stroke, measured: the region rectangle its sweeps need, and the
/// tiles they touch.
///
/// The two come from one walk ([`cover`]) and are used together everywhere — the
/// composite takes the rectangle and the halo, the seeds and the write-back take the
/// tiles and measure each one's block against the rectangle's interior origin.
pub(super) struct Piece {
    pub(super) rect: RegionRect,
    /// The tiles the piece's sweeps touch — what it rewrites, and what the carried
    /// lanes are seeded from and extracted into.
    pub(super) tiles: BTreeSet<TileCoord>,
}

impl Piece {
    /// Where `coord`'s whole `TILE_TEX` block sits in the region, as a copy addresses
    /// it — measured against the region's **interior** origin, an apron in from the
    /// rectangle.
    ///
    /// Integral and non-negative by [`Covered::rect`](super::super::region::Covered)'s
    /// construction, and the far edge of the last tile's block is exactly the region's
    /// extent, so every copy built on this is in bounds.
    pub(super) fn block_origin(&self, coord: TileCoord) -> wgpu::Origin3d {
        let off = coord.origin() - self.rect.lo;
        wgpu::Origin3d {
            x: off.x as u32,
            y: off.y as u32,
            z: 0,
        }
    }

    /// The tiles to composite under the piece: the touched set plus the one-tile ring
    /// around it, so rewritten tiles' aprons read real neighbour content (§6.4).
    pub(super) fn halo(&self) -> &[TileCoord] {
        &self.rect.halo
    }

    /// The region rectangle as the helpers here take it.
    pub(super) fn bbox(&self) -> RegionBox {
        RegionBox {
            origin: self.rect.origin,
            w: self.rect.w,
            h: self.rect.h,
        }
    }
}

/// The half of a loop render that does not depend on which kernels run: what a piece
/// is drawn against, the scope it is recorded into, and what the range hands back.
///
/// Held by [`DynamicsRun`](super::run) and [`LiquifyDraw`](super::liquify) alike,
/// which is the point — the two paths differ in their kernels and their carried state,
/// and differed in nine fields and two prologues besides.
pub(super) struct RegionRun<'a> {
    pub(super) r: &'a StrokeRenderer,
    pub(super) scene: StrokeScene<'a>,
    rec: &'a StrokeRecord,
    /// The budget the range was flattened at
    /// ([`StrokePlan::of`](crate::gpu::stroke::path::StrokePlan::of)), carried so a
    /// plan can re-cut a piece of the record the same way — see [`PlanCtx::tol`].
    tol: crate::path::FlattenTolerance,
    /// Everything both render paths read off the record and the scene, resolved once
    /// (see [`StrokeConstants`](super::super::StrokeConstants)).
    pub(super) consts: &'a super::super::StrokeConstants,
    /// The encoder this run records into, and every resource whose release must trail
    /// a submit — the run-scoped leases and the piece-scoped ones alike. The scope
    /// releases each tier only in the call that submits the commands naming it
    /// ([`SubmitScope`]).
    pub(super) scope: SubmitScope,
    /// Every tile the run has rewritten, accumulated as each piece enumerates its own.
    /// The pieces partition the range's segments, so this ends up the set a second
    /// walk over the whole stroke would have built — for none of the cost.
    dirty: BTreeSet<TileCoord>,
    /// The prefix volume at group 1 of every kernel that reads a tip — the swept
    /// prefix-τ for the wet loop, the coverage prefix for the liquify composition
    /// (§6.13). A function of the brush alone, so bound once for the whole run.
    pub(super) prefix_bg: wgpu::BindGroup,
}

impl<'a> RegionRun<'a> {
    /// Open a run: a scope on the scratch pool labelled `label`, and `prefix_view`
    /// bound at the group every kernel with a tip reads it from.
    pub(super) fn new(
        r: &'a StrokeRenderer,
        scene: StrokeScene<'a>,
        range: &ResolvedRange<'a>,
        prefix_view: &wgpu::TextureView,
        label: &'static str,
    ) -> Self {
        let prefix_bg =
            r.dynamics
                .prefix_bgl
                .group(&r.ctx.device, &format!("{label} prefix bg"), |_| {
                    wgpu::BindingResource::TextureView(prefix_view)
                });
        Self {
            r,
            scene,
            rec: range.rec,
            tol: range.tol,
            consts: range.consts,
            scope: r.scratch.scope(&r.ctx, label),
            dirty: BTreeSet::new(),
            prefix_bg,
        }
    }

    /// Open one piece over `base`: submit what the previous piece recorded, measure
    /// what these segments and firings cover, and take the region that holds it.
    ///
    /// `None` where nothing is covered — a piece holds at least one segment and a
    /// segment covers at least one tile, so it cannot arise, but it costs nothing to
    /// leave the canvas alone if it ever did.
    ///
    /// **Holds the map the piece is about to read.** The composite that follows samples
    /// `base`'s tiles into this encoder, and the caller replaces its map with what the
    /// piece returns — dropping every tile handle the new map supersedes. A
    /// `TilePairHandle`'s drop *is* its release to `TilePool`'s free list
    /// (`gpu::tile::GpuTex::drop`), so without this the next `acquire_tex` could render
    /// over a texture this encoder is still compositing from. Consecutive pieces share
    /// the tiles around their cut, `region::cover` growing every segment's box by an
    /// apron, so that is the ordinary case rather than a corner. The clone is an `rpds`
    /// map of `Arc` handles: a refcount per tile, no pixels (§5.2).
    pub(super) fn open(
        &mut self,
        base: &TileMap,
        segments: &[Segment],
        fires: &[BleedFire],
    ) -> Option<Piece> {
        self.scope.flush();
        // Segments and firings both, in one walk: a window writes up to a quantum
        // behind the piece's first segment, and a tile (or region) the walk misses is
        // flux silently clipped at the boundary (`segments::piece_sweeps`).
        let covered = cover(segments, fires);
        let rect = covered.rect()?;
        // The run's dirty set is the union of the pieces', which is why no caller has
        // to enumerate the whole range's tiles a second time.
        self.dirty.extend(covered.tiles.iter().copied());
        self.scope.hold(base.clone());
        Some(Piece {
            rect,
            tiles: covered.tiles,
        })
    }

    /// Begin a compute pass on this run's encoder, carrying the prefix volume at the
    /// group its own declaration names — so every [`Kernel`](super::kit::Kernel) that
    /// reads one binds it and no dispatch site states which do (§6.10).
    pub(super) fn loop_pass(&mut self, label: &'static str) -> LoopPass<'_> {
        let Self {
            r,
            scope,
            prefix_bg,
            ..
        } = self;
        let prefix = Prefix {
            bgl: &r.dynamics.prefix_bgl,
            group: prefix_bg,
        };
        LoopPass::open(scope.encoder(), label, prefix)
    }

    /// What a plan for a piece at `region_origin` is built against.
    pub(super) fn plan_ctx(&self, region_origin: Vec2) -> PlanCtx<'_> {
        PlanCtx {
            rec: self.rec,
            tol: self.tol,
            region_origin,
            consts: self.consts,
            substrate: self.scene.substrate,
        }
    }

    /// The plan's uniforms, one dynamic-offset slot each — slots being how a uniform
    /// varies across the dispatches of one pass ([`crate::gpu::uniforms`]).
    ///
    /// Leased on the *piece*, like the region it works on: nothing past this piece's
    /// own submission reads it, so holding it for the whole run would make a long
    /// stroke's peak cost scale with the number of pieces. The scope returns it to the
    /// pool behind the piece's own submit ([`SubmitScope::flush`]), so the next piece's
    /// plan is written into the buffer this one used.
    ///
    /// # Panics
    /// On an empty plan, which no piece has: a wet piece holds at least one segment,
    /// and a liquify plan always ends with its resample.
    pub(super) fn upload<K>(&mut self, plan: &[LoopDispatch<K>]) -> LeasedSlots<Stamp> {
        self.scope
            .piece_slots("stark dynamics stamps", plan.iter().map(|d| &d.slot))
            .expect("a piece dispatches at least one slot")
    }

    /// Close the run: submit what is still recorded, release everything behind that
    /// submit ([`SubmitScope::finish`]), and hand back what the range carries on.
    ///
    /// **`hold` is handed to the scope rather than left to drop.** Whatever a path's
    /// last piece did not hand on is still named by copies this call is about to
    /// submit — the wet loop's un-captured mint budget and ceiling lane, for instance.
    /// A `Kept` lease the caller *did* hand on outlives this call by design.
    pub(super) fn finish(
        mut self,
        end_dist: f32,
        tool: Option<ToolState>,
        hold: impl std::any::Any,
    ) -> StrokeCarry {
        let dirty = std::mem::take(&mut self.dirty).into_iter().collect();
        self.scope.hold(hold);
        // `queue.submit` plus the scratch releases that may only follow it
        // (`SubmitScope`). Its own row because it is the one phase here that is not
        // encoding — on the web `submit` hands a command buffer across the wasm
        // boundary.
        crate::timing::span!("stroke.submit");
        self.scope.finish();
        StrokeCarry::finished(end_dist, tool, dirty)
    }
}

/// Composite `base`'s tiles among `halo` into `targets`, 1:1 with canvas px, the
/// region's top-left at `origin` — the read half of a region piece (§6.2), shared
/// with the liquify path (§6.13), which composites its run's base the same way over
/// a larger rect.
///
/// The targets are cleared first, so an empty halo — no base tiles — leaves them at
/// exactly zero: "no paint". The compositor's own `View` and pass-A pipeline, so a
/// composited texel is the tile's texel bit for bit (`over` onto a cleared target
/// at opacity 1 is the source), which is what lets every kernel downstream treat
/// the region as the canvas.
pub(super) fn composite_tiles(
    r: &StrokeRenderer,
    scope: &mut SubmitScope,
    base: &TileMap,
    halo: &[TileCoord],
    rect: RegionBox,
    targets: Targets<'_>,
) {
    let kit = &r.dynamics.region;
    let device = &r.ctx.device;
    let RegionBox {
        origin: region_origin,
        w,
        h,
    } = rect;
    // Composite pass: base tiles → region, 1:1 with canvas px. The compositor's own
    // `ViewUniform` — this path binds its own buffer to the very same
    // `composite.wesl` (§6.2).
    let (sx, sy) = (2.0 / w as f32, -2.0 / h as f32);
    let view = view_uniform(
        // Diagonal: the region is axis-aligned with the canvas whatever angle the
        // *screen* view happens to be at.
        [sx, 0.0, 0.0, sy],
        Vec2::new(-region_origin.x * sx - 1.0, -region_origin.y * sy + 1.0),
        // Zoom and the display transfer reach only the selection outline, and no
        // chrome is drawn into a working region — this is a buffer the loop evolves,
        // not a picture.
        0.0,
        crate::Transfer::Srgb,
    );
    let view_slot = scope.piece_uniform("stark dynamics region view", &view);
    let view_bg =
        kit.composite_view_bgl
            .group(device, "stark dynamics region view bg", |i| match i {
                vb::VIEW => view_slot.window(),
                vb::SAMP => wgpu::BindingResource::Sampler(&kit.composite_sampler),
                other => unreachable!("the composite view group has no binding {other}"),
            });
    let mut tile_origins: Vec<TileInstance> = Vec::with_capacity(halo.len());
    let mut tile_bgs: Vec<&wgpu::BindGroup> = Vec::with_capacity(halo.len());
    for coord in halo {
        if let Some(tile) = base.get(coord) {
            tile_origins.push(TileInstance {
                origin: coord.origin().to_array(),
                opacity: 1.0,
            });
            // **The group the tile itself caches**, which is why this loop is
            // handed pass A's layout rather than building one from the same slot
            // list (`kit.rs`): a live stroke re-composites its halo on every pointer
            // move, and a wide tip's halo is tens of tiles per piece, so a group
            // built here is tens of WebGPU objects a move.
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
        let buf = scope.take_piece_buffer(BufKey {
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            label: "stark dynamics region tile instances",
        });
        scope.write_lease(&buf, bytes);
        buf
    });
    {
        let mut pass =
            desc::begin_pass(scope.encoder(), "stark dynamics region composite", targets);
        // An empty region (no base tiles) just stays cleared → "no paint".
        if let Some(inst) = &tile_inst {
            pass.set_pipeline(&kit.composite_pipeline);
            // Offset 0: `kit.composite_view_bgl` declares the view at a dynamic
            // offset, and this loop composites one region through one view.
            kit.composite_view_bgl
                .set(&mut pass, &view_bg, &[view_slot.offset(0)]);
            pass.set_vertex_buffer(0, inst.slice(..));
            for (i, bg) in tile_bgs.iter().enumerate() {
                let idx = i as u32;
                kit.composite_tile_bgl.set(&mut pass, bg, &[]);
                pass.draw(desc::QUAD_VERTICES, idx..idx + 1);
            }
        }
    }
}

/// The selection over a region (§6.8), gathered from the same halo tiles the paint
/// came from so it is 1:1 with the region — or the 1×1 constant that stands in for
/// an unrestricted selection, under which every masked read returns 1.
pub(super) fn region_selection(
    r: &StrokeRenderer,
    scope: &mut SubmitScope,
    selection: &crate::document::selection::Selection,
    halo: &[TileCoord],
    rect: RegionBox,
) -> wgpu::TextureView {
    let RegionBox {
        origin: region_origin,
        w,
        h,
    } = rect;
    // An unrestricted selection binds the 1×1 constant — the loop's masked reads then
    // return 1 everywhere and the stroke behaves exactly as before.
    if selection.is_universal() {
        return r.selection.constant(1.0);
    }
    // Leased, not created: this is up to `MAX_REGION_DIM`² of `R8Unorm` and a live
    // stroke re-gathers it per piece per pointer move, so creating one was megabytes
    // of allocation and teardown a move for a texture of the very same shape each
    // time (`scratch`).
    let view = scope
        .take_piece(Key {
            size: (w, h),
            format: crate::gpu::tile::MASK_FORMAT,
            usage: crate::gpu::selection::SelectionRenderer::REGION_MASK_USAGE,
            label: "stark selection region mask",
        })
        .view;
    r.selection.region_mask(
        scope,
        &view,
        selection,
        halo,
        (region_origin, crate::view::Extent2::new(w, h)),
    );
    view
}

/// Slice each affected tile's full `TILE_TEX` block out of a region into a fresh CoW
/// tile → aprons stay bit-identical to neighbour interiors (§6.4), and the wide region
/// aux narrows to the persistent one (height). The write half of every region piece,
/// wet or liquify (§6.13).
///
/// One region-sized narrow pass, then plain texture copies: the color and residual
/// tiles are the region's own formats, so a copy is exact. Rounding is untouched — a
/// copy is bit-exact, and the narrow pass render-writes a loaded f16 value back to its
/// own lattice point (see `slice.wesl`).
pub(super) fn slice_region(
    r: &StrokeRenderer,
    scope: &mut SubmitScope,
    pool: &crate::gpu::tile::TilePool,
    base: &TileMap,
    piece: &Piece,
    from: RegionChannels<'_>,
) -> TileMap {
    let RegionChannels {
        color_tex,
        aux,
        resid_tex,
    } = from;
    let kit = &r.dynamics.region;
    let device = &r.ctx.device;

    // The one channel that changes representation, narrowed once over the whole
    // region (§6.7's aux stays a single height channel on the persistent tile).
    // Pooled; the pass's clearing load op rewrites every texel.
    let LeasedTex {
        tex: narrow,
        view: narrow_view,
    } = scope.take_piece(Key {
        size: (color_tex.width(), color_tex.height()),
        format: TILE_AUX_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        label: "stark dynamics narrow aux",
    });
    let bg = kit.slice_bgl.group(device, "stark dynamics slice bg", |_| {
        wgpu::BindingResource::TextureView(aux)
    });
    {
        let mut pass = desc::begin_pass(scope.encoder(), "stark dynamics slice", &narrow_view);
        pass.set_pipeline(&kit.slice_pipeline);
        kit.slice_bgl.set(&mut pass, &bg, &[]);
        pass.draw(desc::FULLSCREEN_VERTICES, 0..1);
    }

    let mut new_map = base.clone();
    for coord in &piece.tiles {
        let dst = r.acquire_tile(pool, AllocSource::DynamicsWriteback);
        dst.copy_from_region(
            scope.encoder(),
            color_tex,
            &narrow,
            resid_tex,
            piece.block_origin(*coord),
        );
        new_map = new_map.insert(*coord, dst);
    }
    new_map
}
