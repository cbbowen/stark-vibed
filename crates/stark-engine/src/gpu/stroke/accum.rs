//! The per-tile landing procedure every swept render shares (§6.2, §6.12): sweep a
//! tile's segments into a working **parcel**, then land the parcel on the paint under
//! the tile with one fullscreen pass into a fresh copy-on-write tile.
//!
//! The plain deposit lands each parcel the moment it is swept. Two effects have a law
//! that is neither of §6.2's composable forms — the erase's `1 − opacity·w` (§6.12) and
//! the deposit's scaled coverage below full opacity (§6.2) — so applying it per piece
//! would compound at every cut a live stroke makes. Those keep the *composable* half
//! summing across pieces, a parcel per touched tile, and land the law once per tile per
//! render against **pristine** paint: the tile as the stroke first found it, never an
//! earlier piece's output. The parcels and the pristine handles ride the stroke's carry
//! ([`ParcelCarry`]), and a resuming piece *copies* the parcel it resumed rather than
//! writing it, which is what lets the live tail re-render from the same frozen head
//! every frame.
//!
//! Only a render a later piece will resume keeps its parcels past itself. Every other
//! — the live tail, a commit, a replay — lands each tile through a [`Ring`] of working
//! parcels, so its scratch does not grow with the stroke.
//!
//! The effects differ in which fragment rasterizes the extent (`stamp.wesl`'s `fs_main`
//! or `fs_erase`), the parcel's lanes, the landing shader (`integrate.wesl` or
//! `erase.wesl`) and what bare canvas means ([`ParcelEffect`]). So this holds the
//! orchestration and not the rendering: each effect hands in its own pipelines and binds
//! its own landing group, next to the shader it mirrors (§6.10).
//!
//! **The stamp loop's carried tiles are deliberately not here.** `LoopCarry::fresh`
//! (§6.2) obeys the same share-never-write contract and nothing else of this: its state
//! is seeded into a shared region at a per-tile offset, no pass ever binds it, and its
//! copy-on-write tiles come from the region write-back rather than a landing pass.

use std::sync::Arc;

use stark_model::geom::TileCoord;

use crate::device::channels::Targets;
use crate::device::desc;
use crate::device::scratch::{Kept, Key, LeasedTex, SubmitScope};
use crate::device::tile::{AllocSource, TileMap, TilePairHandle};

use super::incremental::{Carried, Resume};
use super::swept::SweepDraws;
use super::{StrokeCarry, StrokeRenderer, StrokeScene, ToolState};

/// The most lanes a parcel can have: the channel trio (§6.7) plus the ceiling lane a
/// pen-driven opacity adds (§6.2), which is the widest thing a swept pass writes. The
/// erase's single transparency mass is the other end of the same range.
///
/// A lane list may have a **hole**: the sweep's targets are the shader's `@location`s,
/// and the ceiling sits at 3 whether or not the space has a residual at 2 — so a
/// colorimetric space under a pen-driven ceiling attaches `[color, aux, none, ceiling]`,
/// the hole riding the list as a `None` rather than shifting the lane after it
/// ([`desc::Attachments`] attaches through the last lane there is).
const MAX_LANES: usize = 4;

/// How many working parcels a render that keeps none rotates through ([`Ring`]).
///
/// At 1, tile `n+1`'s sweep waits on tile `n`'s landing pass and the `2N` render passes
/// cannot overlap at all. Three keeps a sweep, a landing and a spare in flight, which
/// is where the win flattens. One parcel is ~1.5 MB in a pigment space at 1×, four
/// times that supersampled.
const SCRATCH_RING: usize = 3;

/// An effect's parcel lanes by index — the sweep shader's `@location`s — with `None` at
/// a lane this stroke does not carry. Fixed-size so a hole is a `None` at a *named*
/// index rather than a push order the lane constants have to agree with.
pub(super) type LaneKeys = [Option<Key>; MAX_LANES];

/// One parcel lane's pool key: a full tile texture (interior + apron), renderable (the
/// sweep accumulates into it), bindable (the landing pass reads it), and copyable both
/// ways (a resuming piece copies the carried total into its working lane). Only the
/// format and the label are the effect's.
///
/// The whole texture, apron included, because the apron is rasterized with the interior
/// and a piece that resumed only the interior would seam (§6.4). The resume copy takes
/// its extent from this key ([`Key::extent`]), so the two cannot name different blocks.
pub(super) fn lane_key(format: wgpu::TextureFormat, label: &'static str) -> Key {
    Key::tile(
        format,
        wgpu::TextureUsages::RENDER_ATTACHMENT
            .union(wgpu::TextureUsages::TEXTURE_BINDING)
            .union(wgpu::TextureUsages::COPY_SRC)
            .union(wgpu::TextureUsages::COPY_DST),
        label,
    )
}

/// A stroke's accumulated parcel over one tile, kept past the render that swept it.
pub(super) struct Parcel {
    /// `None` is a hole (see [`MAX_LANES`]).
    lanes: [Option<Kept>; MAX_LANES],
}

impl Parcel {
    /// Check out a parcel of `keys`' shape, holes included.
    fn take(r: &StrokeRenderer, keys: LaneKeys) -> Self {
        Parcel {
            lanes: keys.map(|k| k.map(|k| r.scratch.keep(&r.ctx.device, k))),
        }
    }

    fn lanes(&self) -> ParcelLanes<'_> {
        ParcelLanes(
            self.lanes
                .each_ref()
                .map(|lane| lane.as_ref().map(Kept::leased)),
        )
    }
}

/// One tile's working parcel as its passes name it, in the lane order the effect's keys
/// named them — which is the order the sweep pipeline's own target list declares, so a
/// parcel cannot attach in one order and bind in another.
#[derive(Clone, Copy)]
pub(super) struct ParcelLanes<'a>([Option<&'a LeasedTex>; MAX_LANES]);

impl<'a> ParcelLanes<'a> {
    /// Lane `i`, as the landing pass binds it. Panics on a hole: a lane is asked for
    /// only where the effect's landing shader declares a slot for it, and both that and
    /// the key list follow the same predicates (§6.7: a space has a residual or it does
    /// not; §6.2: the pen drives the ceiling or it does not).
    pub(super) fn lane(&self, i: usize) -> &'a wgpu::TextureView {
        let lane = self.0.get(i).copied().flatten().unwrap_or_else(|| {
            panic!(
                "lane {i} of the parcel is a hole: an effect's lane names and its \
                 keys disagree about what its parcel holds",
            )
        });
        &lane.view
    }

    /// The lanes in key order, as the sweep attaches them: a hole attaches nothing.
    fn views(self) -> impl Iterator<Item = Option<&'a wgpu::TextureView>> {
        self.0.into_iter().map(|lane| lane.map(|l| &l.view))
    }
}

/// The working parcels of a render that keeps none: up to [`SCRATCH_RING`], reused
/// round-robin. Each tile's sweep clears or copies into its slot before that tile's
/// landing pass reads it, and one encoder orders the reuse behind the earlier read.
///
/// Leased on the scope's **run** tier: a flush hands back the piece tier, and a slot
/// handed back mid-render could be leased out again while a later tile still draws
/// into it.
#[derive(Default)]
struct Ring {
    slots: Vec<[Option<LeasedTex>; MAX_LANES]>,
    next: usize,
}

impl Ring {
    /// The next slot, leased on first use, so a render reaching fewer tiles than the
    /// ring takes only as many.
    fn next(&mut self, scope: &mut SubmitScope, keys: LaneKeys) -> ParcelLanes<'_> {
        if self.slots.len() < SCRATCH_RING {
            self.slots.push(keys.map(|k| k.map(|k| scope.take_run(k))));
        }
        let slot = &self.slots[self.next % self.slots.len()];
        self.next += 1;
        ParcelLanes(slot.each_ref().map(Option::as_ref))
    }
}

/// An effect's carried parcels (§6.2, §6.12): per touched tile, the paint the
/// stroke found there and everything it has accumulated over it so far.
///
/// One type for both effects, carried tagged with the one that captured it
/// ([`Carried::Parcel`]): the payloads are the same shape, but the lane count is the
/// effect's, so which one a stroke resumes is still asserted.
pub(super) struct ParcelCarry {
    pub(super) tiles: ParcelTiles,
}

/// A carry's tiles, **persistent** rather than a `BTreeMap`.
///
/// The map holds every tile the stroke has ever touched, and it has to: a stroke can
/// cross back over any of them, and a tile whose parcel had been dropped would be
/// resumed from the *head's own output* instead of from pristine paint — the
/// compounding [`ParcelTile::pristine`] exists to rule out. So the entries cannot be
/// pruned, and what is left to fix is the *copying*: a capturing render shares what
/// the pieces before it accumulated and replaces only the tiles it drew, where
/// cloning a `BTreeMap` walked the whole of it. That was the one term in a live
/// stroke's per-move cost that followed its length rather than its tail, bounded
/// only by [`MAX_STROKE_TILES`](stark_model::document::MAX_STROKE_TILES) — a million.
///
/// Nothing iterates it, so the hash map's order costs no determinism (§12.6): the
/// carry is read by `get` at a coordinate the sweep named.
pub(super) type ParcelTiles = rpds::HashTrieMap<TileCoord, ParcelTile>;

/// One tile's share of a [`ParcelCarry`].
#[derive(Clone)]
pub(super) struct ParcelTile {
    /// The layer's tile as the stroke found it — the paint every piece's rewrite is
    /// derived from. A piece rendered later must not read the *output* of an earlier one
    /// (the base it is handed holds exactly that), or the effect would compound per
    /// piece instead of per stroke. An `Arc`'d pool handle, so keeping it is a refcount.
    ///
    /// `None` is bare canvas, reachable only for [`ParcelEffect::Deposit`]: a deposit
    /// onto nothing mints a tile over the 1×1 zeroes, where an erase never records the
    /// tile.
    pub(super) pristine: Option<TilePairHandle>,
    /// The stroke's parcel over this tile, summed so far. Shared between successive
    /// carries rather than copied: a piece never writes the parcel it resumed — it
    /// copies into fresh working lanes and extends those.
    pub(super) accum: Arc<Parcel>,
}

/// The two effects this procedure serves — what a carry was captured by, and what a
/// tile the layer does not have means to it: the one place the two diverge that is not
/// a shader.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(super) enum ParcelEffect {
    /// The swept deposit (§6.2). Bare canvas is painted anyway, over the 1×1 zeroes
    /// (§6.8's pattern): a stroke onto nothing mints a tile, and the landing pass's
    /// clamped loads mean bare canvas costs no tile of its own to read.
    Deposit,
    /// The erase pass (§6.12). Bare canvas is nothing to work on: no output tile, no
    /// parcel, no entry in the carry — there is no transparency to add to absent
    /// paint, so an eraser over bare canvas mints nothing.
    Erase,
}

/// How a render's parcels are resumed and kept.
pub(super) enum Lanes<'p> {
    /// Nothing to resume and nothing to keep: the deposit at full opacity, whose law
    /// composes across pieces on its own (§6.2).
    Fresh,
    /// The carried shape (§6.2, §6.12): resume what the previous piece captured, and
    /// keep this render's parcels if a later piece will resume them.
    Resume(Resume<'p>),
}

/// The swept-extent rasterization half of one tile, which the effects differ in by
/// **pipeline and lanes alone**: the same [`SweepDraws`] — `stamp.wesl`'s `fs_main`
/// laying paint, or its `fs_erase` accumulating transparency.
pub(super) struct SweepStep<'a> {
    /// This pass's label, so a debug capture still names the effect that recorded it
    /// rather than the procedure they share.
    pub(super) label: &'static str,
    pub(super) pipeline: &'a wgpu::RenderPipeline,
    pub(super) draws: &'a SweepDraws,
    /// The pool keys of the lanes `pipeline` writes, at its `@location`s.
    pub(super) keys: LaneKeys,
}

/// The landing half: the fullscreen pass that applies the effect's law once,
/// writing `new = f(pristine, parcel)` into a fresh copy-on-write tile's MRT.
pub(super) struct LandStep<'a> {
    pub(super) label: &'static str,
    pub(super) pipeline: &'a wgpu::RenderPipeline,
    /// What bare canvas means to this landing, and what a carry it keeps is tagged by.
    pub(super) effect: ParcelEffect,
}

/// What one tile's landing pass is bound against, handed to the effect so it can name
/// its own shader's slots.
pub(super) struct Landing<'a> {
    /// The paint the stroke found under this tile, as the landing pass reads it: the
    /// pristine tile's channels, or the renderer's 1×1 zeroes where the layer has
    /// none — which for [`ParcelEffect::Erase`] never happens.
    pub(super) base: Targets<'a>,
    /// The stroke's parcel over this tile so far, in the effect's lane order.
    pub(super) parcel: ParcelLanes<'a>,
    /// The selection coverage here (§6.8) — the coverage *alone*: the mask's opacity
    /// is already folded into `StrokeConstants::opacity`, which every landing shader
    /// multiplies this by.
    pub(super) mask: &'a wgpu::TextureView,
}

/// What a render leaves behind: the copy-on-write map, the carry for the piece that
/// resumes after it, and the tiles it rewrote.
pub(super) struct Landed {
    map: TileMap,
    tool: Option<ToolState>,
    dirty: Vec<TileCoord>,
}

impl Landed {
    /// The finished range, its arc clock at `dist`.
    pub(super) fn finished(self, dist: f32) -> (TileMap, StrokeCarry) {
        (self.map, StrokeCarry::finished(dist, self.tool, self.dirty))
    }
}

/// Draw a piece: per tile the sweep reaches, extend the tile's parcel and land it on
/// the paint the stroke found there.
///
/// Takes the scope *after* the caller has built its per-piece buffers and bind groups
/// against it, so the effect's setup and this loop land in one submit — and consumes it,
/// submitting before the carry leaves, so no [`Kept`] lease can reach the pool ahead of
/// the commands naming it (`scratch`).
///
/// `bind` builds the landing pass's group — the one thing the effects cannot share,
/// their shaders declaring different slots (§6.10).
pub(super) fn land_tiles(
    r: &StrokeRenderer,
    scene: StrokeScene<'_>,
    mut scope: SubmitScope,
    lanes: Lanes<'_>,
    sweep: &SweepStep<'_>,
    land: &LandStep<'_>,
    mut bind: impl FnMut(&Landing<'_>) -> wgpu::BindGroup,
) -> Landed {
    let (prior, capture) = match lanes {
        Lanes::Fresh => (None, false),
        Lanes::Resume(resume) => (resume.prior.map(|t| t.parcels(land.effect)), resume.capture),
    };
    // A carry shares every tile the pieces before it accumulated, this piece's own
    // replacing theirs below — structurally, so resuming costs the tiles this piece
    // draws and not the stroke's length (see `ParcelTiles`). Built only when a later
    // piece will read it.
    let mut carry = capture.then(|| prior.map_or_else(ParcelTiles::new, |p| p.tiles.clone()));
    let mut ring = Ring::default();
    // The copy-on-write map being built. The pristine fallback is read out of
    // `scene.base()` rather than out of this: they start equal, but this one accumulates
    // the piece's own output, and a tile re-derived from that is exactly the
    // compounding the pristine handle rules out.
    let mut map = scene.base().clone();
    let mut dirty = Vec::with_capacity(sweep.draws.coords().len());

    for (i, coord) in sweep.draws.coords().iter().enumerate() {
        // The paint the stroke found under this tile: what an earlier piece recorded,
        // or — for a tile this stroke reaches for the first time — the base itself,
        // which no earlier piece can have rewritten. A carried entry answers for itself,
        // `None` included: for the deposit a tile first touched on bare canvas has this
        // stroke's own minted tile in the base by now, and falling back to it would
        // re-derive from that.
        let resumed = prior.and_then(|p| p.tiles.get(coord));
        let pristine = match resumed {
            Some(t) => t.pristine.clone(),
            None => scene.base().get(coord).cloned(),
        };
        if pristine.is_none() && land.effect == ParcelEffect::Erase {
            continue;
        }

        // This tile's working parcel: its own where the carry keeps it, the ring's next
        // slot otherwise. Either way every texel is written before the landing pass
        // reads it — the carried total copied in, or a clear for a first touch — which
        // is the pool's no-zero-init contract (`scratch`).
        let parcel = carry.is_some().then(|| Parcel::take(r, sweep.keys));
        let work = match &parcel {
            Some(p) => p.lanes(),
            None => ring.next(&mut scope, sweep.keys),
        };
        if let Some(old) = resumed.map(|t| &t.accum) {
            // A hole in one list is a hole in all three: the keys made both parcels. A
            // carry only ever resumes the stroke that captured it, and this is where
            // that promise would show if it broke.
            debug_assert!(
                old.lanes
                    .iter()
                    .zip(&sweep.keys)
                    .all(|(lane, key)| lane.is_some() == key.is_some()),
                "a resumed parcel and this piece's keys disagree about where the holes are",
            );
            for ((src, dst), key) in old.lanes.iter().zip(work.0).zip(&sweep.keys) {
                if let (Some(src), Some(dst), Some(key)) = (src, dst, key) {
                    scope.encoder().copy_texture_to_texture(
                        src.tex().as_image_copy(),
                        dst.tex.as_image_copy(),
                        key.extent(),
                    );
                }
            }
        }

        // This tile's segments into the working parcel, over the carried total.
        {
            let into = if resumed.is_some() {
                desc::Attachments::loaded(work.views())
            } else {
                desc::Attachments::cleared(work.views())
            };
            let mut pass = desc::begin_pass(scope.encoder(), sweep.label, into);
            sweep.draws.draw(&mut pass, sweep.pipeline, i);
        }

        // The whole stroke's parcel so far, landed on the pristine paint.
        let dst = r.acquire_tile(scene.pool, AllocSource::IntegrateDestination);
        let mask = r.selection.mask_for(scene.selection, *coord);
        // The group holds its own references to every view in it, so it outlives the
        // borrow of `pristine` that described it.
        let bg = bind(&Landing {
            base: r.zeroes.or(pristine.as_ref().map(TilePairHandle::targets)),
            parcel: work,
            mask: mask.view(),
        });
        scope.fullscreen_pass(
            land.label,
            dst.targets(),
            land.pipeline,
            desc::Bound::one(&bg),
        );

        map = map.insert(*coord, dst);
        dirty.push(*coord);
        if let (Some(tiles), Some(parcel)) = (&mut carry, parcel) {
            tiles.insert_mut(
                *coord,
                ParcelTile {
                    pristine,
                    accum: Arc::new(parcel),
                },
            );
        }
    }

    // Submit before the carry leaves this call: a `Kept` may reach the pool's free list
    // only behind the submit of the commands naming it.
    //
    // `finish` submits only when the scope has something recorded, so that rests on a
    // lease worth protecting implying a recording: every `Kept` this call puts into the
    // carry is named by a sweep pass and a landing pass. One with nothing recorded
    // behind it holds only `Arc` clones of the *previous* carry's parcels, whose
    // commands an earlier render already submitted.
    scope.finish();
    let tool = carry.map(|tiles| ToolState(Carried::Parcel(land.effect, ParcelCarry { tiles })));
    Landed { map, tool, dirty }
}
