//! The incremental tile accumulator (§6.2, §6.12): the bookkeeping the two
//! **non-composable** swept effects share, written once.
//!
//! A stroke is drawn in pieces and re-drawn on every pointer move, which works
//! because the deposit composes under re-cutting: its per-segment law is additive
//! in `τ` or of the form `1 − exp(−k·τ)`, and either shape lets a piece stack on
//! what the pieces before it left. Two effects have a law of neither shape — the
//! erase's `1 − opacity·w` (§6.12) and the swept deposit's scaled coverage below
//! full opacity (§6.2) — so applying it per piece would compound at every cut a
//! live stroke makes.
//!
//! Both answer it the same way, and it is the only answer: keep the *composable*
//! half accumulating across the pieces — a **parcel** per touched tile, summed by
//! the sweep — and apply the non-composable law exactly once per tile per render,
//! always against **pristine** paint, the tile as the stroke first found it rather
//! than an earlier piece's output. The parcels and the pristine handles ride the
//! stroke's carry ([`ParcelCarry`]) beside the stamp loop's reservoir, and a
//! resuming piece *copies* the parcel it resumed rather than writing it, which is
//! what lets the live tail re-render from the same frozen head every frame.
//!
//! **What the two effects actually differ in** is short, and none of it is
//! bookkeeping: which fragment rasterizes the extent (`stamp.wesl`'s `fs_erase`
//! against its `fs_main`), how many lanes the parcel has (one transparency mass
//! against the channel trio), what the landing pass computes (`erase.wesl` against
//! `integrate.wesl`), and what bare canvas means ([`BareCanvas`]). Everything
//! else — resolving the pristine handle, taking the working leases, the resume
//! copy, load-versus-clear, the swept draw, the copy-on-write destination, the
//! dirty list, the carry, and the submit that has to precede the carry leaving the
//! call — is otherwise written twice, in two files, with the same comments on it.
//!
//! So this holds the orchestration and **not** the rendering:
//! [`IncrementalTileAccumulator::run`] is handed the effect's own pipelines and
//! asks the effect to bind its own landing group, so a reader still finds
//! `erase.wesl`'s slots in `erase.rs` and `integrate.wesl`'s in `swept.rs`, next to
//! the shader each mirrors (§6.10). The two are one *procedure*, not one pass.
//!
//! **The stamp loop's carried tiles are deliberately not here.** `LoopCarry::fresh`
//! (§6.2) obeys the same share-never-write contract, but nothing else of this: its
//! state is seeded into a shared region at a per-tile offset rather than into a
//! working texture per tile, no pass ever binds it, it has no pristine handle at
//! all — the mint budget is running state, like the reservoir — and its
//! copy-on-write tiles come from the region write-back rather than from a landing
//! pass. Folding it in would mean an abstraction over what a pass *is*, which is
//! the thing worth not having.

use std::collections::BTreeMap;
use std::sync::Arc;

use stark_model::geom::TileCoord;
use stark_shaders::mirror::stamp_common::SWEEP_VERTS;

use crate::gpu::channels::Targets;
use crate::gpu::desc;
use crate::gpu::tile::{AllocSource, TileMap, TilePairHandle};

use super::swept::SweepDraws;
use super::{StrokeRenderer, StrokeScene};
use crate::gpu::scratch::{Kept, Key, SubmitScope};

/// The most lanes a parcel can have: the channel trio (§6.7) plus the ceiling lane
/// a pen-driven opacity adds (§6.2), which is the widest thing a swept pass writes.
/// The erase's single transparency mass is the other end of the same range.
///
/// A lane list may have a **hole**: the sweep's targets are the shader's
/// `@location`s, and the ceiling sits at 3 whether or not the space has a residual
/// at 2 — so a colorimetric space under a pen-driven ceiling attaches
/// `[color, aux, none, ceiling]`, and the hole rides the list as a `None` rather
/// than shifting the lane after it.
const MAX_LANES: usize = 4;

/// One parcel lane's pool key: a full tile texture (interior + apron), renderable
/// (the sweep accumulates into it), bindable (the landing pass reads it), and
/// copyable both ways (a resuming piece copies the carried total into its working
/// lane). Only the format and the label are the effect's — what a missing usage
/// would cost is the same on both paths, so it is stated once.
///
/// The whole texture, apron included, because the apron is rasterized with the
/// interior and a piece that resumed only the interior would seam (§6.4). The resume
/// copy takes its extent from this key ([`Key::extent`]) rather than from a constant
/// beside it, so the two cannot name different blocks.
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

/// The stroke's accumulated parcel over one tile: the working textures the sweep
/// sums into, in the lane order the effect's keys named them.
///
/// Ordered rather than named, because the order is what the sweep pipeline's own
/// target list declares — the effect names its lanes with index constants beside
/// the keys it builds them from, so the two cannot drift into a parcel that
/// attaches in a different order than it binds.
pub(super) struct Parcel {
    /// `None` is a hole (see [`MAX_LANES`]): no lease, and no attachment at that
    /// index of the sweep.
    lanes: Vec<Option<Kept>>,
}

impl Parcel {
    /// Check out a working parcel of `keys`' shape, holes included.
    fn take(r: &StrokeRenderer, keys: &[Option<Key>]) -> Self {
        assert!(
            keys.len() <= MAX_LANES,
            "a parcel is at most the channel trio and the ceiling lane, got {} lanes",
            keys.len(),
        );
        Parcel {
            lanes: keys
                .iter()
                .map(|k| k.map(|k| r.scratch.keep(&r.ctx.device, k)))
                .collect(),
        }
    }

    /// Lane `i`, as the landing pass binds it.
    ///
    /// The panic says what the bare index would not. A lane is asked for only where
    /// the effect's landing shader declares a slot for it, and how many lanes exist
    /// is what the effect's key list said — both derived from the same predicates
    /// (§6.7: a space has a residual or it does not; §6.2: the pen drives the
    /// ceiling or it does not). So this is unreachable, and unreachable through an
    /// agreement between two lists in the *effect's* file, which is exactly the
    /// kind that goes stale quietly.
    pub(super) fn lane(&self, i: usize) -> &wgpu::TextureView {
        self.lanes
            .get(i)
            .and_then(Option::as_ref)
            .unwrap_or_else(|| {
                panic!(
                    "lane {i} of a {}-lane parcel: an effect's lane names and its keys \
                     disagree about what its parcel holds",
                    self.lanes.len(),
                )
            })
            .view()
    }

    /// The lanes as the sweep attaches them, in key order — a fixed array plus
    /// [`Self::len`] rather than a `Vec`, [`Targets::attachments`]' reason: one of
    /// these is built per tile per piece, and that rate is what a per-pass
    /// allocation costs (§6.2). A hole attaches nothing, at the index the sweep
    /// pipeline's own target list has nothing.
    fn attachments(
        &self,
        ops: wgpu::Operations<wgpu::Color>,
    ) -> [Option<wgpu::RenderPassColorAttachment<'_>>; MAX_LANES] {
        let mut att = [None, None, None, None];
        for (slot, lease) in att.iter_mut().zip(&self.lanes) {
            *slot = lease.as_ref().map(|l| desc::attach(l.view(), ops));
        }
        att
    }

    /// How many of [`Self::attachments`] are real.
    fn len(&self) -> usize {
        self.lanes.len()
    }
}

/// An effect's carried parcels (§6.2, §6.12): per touched tile, the paint the
/// stroke found there and everything it has accumulated over it so far.
///
/// One type for both effects, and one variant each in
/// [`Carried`](super::incremental::Carried) — the payloads are the same shape, but
/// which one a stroke resumes is still worth asserting, since the lane count is the
/// effect's and a carry only ever resumes the stroke that captured it.
pub(super) struct ParcelCarry {
    pub(super) tiles: BTreeMap<TileCoord, ParcelTile>,
}

/// One tile's share of a [`ParcelCarry`].
pub(super) struct ParcelTile {
    /// The layer's tile as the stroke found it — the paint every piece's rewrite is
    /// derived from. A piece rendered later must not read the *output* of an earlier
    /// one (the base it is handed holds exactly that), or the effect would compound
    /// per piece instead of per stroke. An `Arc`'d pool handle, so keeping it is a
    /// refcount, not a copy.
    ///
    /// `None` is bare canvas, reachable only under [`BareCanvas::Mint`]: a deposit
    /// onto nothing mints a tile over the 1×1 zeroes, where an erase has nothing to
    /// erase and never records the tile at all.
    pub(super) pristine: Option<TilePairHandle>,
    /// The stroke's parcel over this tile, summed so far. Shared between successive
    /// carries rather than copied: a piece never writes the parcel it resumed — it
    /// copies into fresh working lanes and extends those — so the tiles a piece does
    /// not touch ride forward as clones of the same lease.
    pub(super) accum: Arc<Parcel>,
}

/// What a tile the layer does not have means to the effect — the one place the two
/// diverge that is not a shader.
#[derive(Copy, Clone)]
pub(super) enum BareCanvas {
    /// Nothing to work on: no output tile, no parcel, no entry in the carry. The
    /// erase pass (§6.12) — there is no transparency to add to absent paint, so an
    /// eraser over bare canvas mints nothing.
    Skip,
    /// Paint anyway, over the 1×1 zeroes (§6.8's pattern). The deposit — a stroke
    /// onto nothing mints a tile, and the landing pass's clamped loads mean bare
    /// canvas costs no tile of its own to read.
    Mint,
}

/// The swept-extent rasterization half of one tile, which the two effects differ in
/// by **pipeline alone**: the same segments in the same order, the same per-tile
/// instance runs and transform slots ([`sweep_draws`](super::swept::sweep_draws)),
/// the same brush-resolved bind groups ([`sweep_binds`](super::swept::sweep_binds))
/// — `stamp.wesl`'s `fs_main` laying paint, or its `fs_erase` accumulating
/// transparency.
pub(super) struct Sweep<'a> {
    /// This pass's label, so a debug capture still names the effect that recorded it
    /// rather than the procedure the two share.
    pub(super) label: &'static str,
    pub(super) pipeline: &'a wgpu::RenderPipeline,
    pub(super) draws: &'a SweepDraws,
    /// The prefix-τ volume at group 1, and the noise + substrate fields at group 2.
    pub(super) prefix: &'a wgpu::BindGroup,
    pub(super) noise: &'a wgpu::BindGroup,
}

/// The landing half: the fullscreen pass that applies the non-composable law once,
/// writing `new = f(pristine, parcel)` into a fresh copy-on-write tile's MRT.
pub(super) struct Land<'a> {
    pub(super) label: &'static str,
    pub(super) pipeline: &'a wgpu::RenderPipeline,
}

/// What one tile's landing pass is bound against — everything the accumulator
/// resolved, handed to the effect so it can name its own shader's slots.
pub(super) struct Landing<'a> {
    /// The paint the stroke found under this tile, as the landing pass reads it: the
    /// pristine tile's channels, or the renderer's 1×1 zeroes where the layer has
    /// none — which under [`BareCanvas::Skip`] never happens.
    pub(super) base: Targets<'a>,
    /// The stroke's parcel over this tile so far, in the effect's lane order.
    pub(super) parcel: &'a Parcel,
    /// The selection coverage here (§6.8) — the coverage *alone*: the mask's opacity
    /// is already folded into `StrokeConstants::opacity`, which every landing shader
    /// multiplies this by.
    pub(super) mask: &'a wgpu::TextureView,
}

/// What a run leaves behind: the copy-on-write map, the carry for the piece that
/// resumes after it, and the tiles this piece rewrote.
pub(super) struct Landed {
    pub(super) map: TileMap,
    pub(super) carry: ParcelCarry,
    pub(super) dirty: Vec<TileCoord>,
}

/// The shared procedure (§6.2, §6.12): resume a stroke's parcels, extend them over
/// this piece's tiles, and land each one on the paint the stroke found there.
///
/// Owns the [`SubmitScope`] for the whole of it, which is what makes the ordering
/// structural rather than a comment: [`run`](Self::run) consumes the accumulator and
/// submits before it returns the carry, so a caller cannot hand a [`Kept`] lease out
/// ahead of the submit of the commands naming it (`scratch`).
pub(super) struct IncrementalTileAccumulator<'a> {
    r: &'a StrokeRenderer,
    scene: StrokeScene<'a>,
    scope: SubmitScope,
    keys: &'a [Option<Key>],
    bare: BareCanvas,
    /// The copy-on-write map being built. Read the pristine fallback out of
    /// `scene.base` rather than out of this: they start equal, but this one is
    /// already accumulating the piece's own output, and a tile re-derived from
    /// that is exactly the compounding the pristine handle rules out.
    map: TileMap,
    tiles: BTreeMap<TileCoord, ParcelTile>,
    dirty: Vec<TileCoord>,
}

impl<'a> IncrementalTileAccumulator<'a> {
    /// Open a piece: everything the pieces before it accumulated — shared, never
    /// rewritten — with this piece's tiles replacing theirs in [`run`](Self::run).
    ///
    /// Takes the scope by value *after* the caller has built its per-piece buffers
    /// and bind groups against it, so the effect's setup and the shared loop record
    /// into one encoder and land in one submit.
    pub(super) fn resume(
        r: &'a StrokeRenderer,
        scene: StrokeScene<'a>,
        scope: SubmitScope,
        keys: &'a [Option<Key>],
        bare: BareCanvas,
        prior: Option<&ParcelCarry>,
    ) -> Self {
        let tiles = prior.map_or_else(BTreeMap::new, |p| {
            p.tiles
                .iter()
                .map(|(c, t)| {
                    (
                        *c,
                        ParcelTile {
                            pristine: t.pristine.clone(),
                            accum: Arc::clone(&t.accum),
                        },
                    )
                })
                .collect()
        });
        Self {
            r,
            scene,
            scope,
            keys,
            bare,
            map: scene.base.clone(),
            tiles,
            dirty: Vec::new(),
        }
    }

    /// Draw the piece: per tile the sweep reaches, extend the parcel and land it.
    ///
    /// `bind` builds the landing pass's group from what the accumulator resolved —
    /// the one thing here that cannot be shared, since the two effects' shaders
    /// declare different slots and each names its own (§6.10).
    pub(super) fn run(
        mut self,
        sweep: &Sweep<'_>,
        land: &Land<'_>,
        mut bind: impl FnMut(&Landing<'_>) -> wgpu::BindGroup,
    ) -> Landed {
        for (i, coord) in sweep.draws.coords.iter().enumerate() {
            // The paint the stroke found under this tile: what an earlier piece
            // recorded, or — for a tile this stroke reaches for the first time — the
            // base itself, which no earlier piece can have rewritten.
            //
            // A carried entry answers for itself, `None` included: under
            // [`BareCanvas::Mint`] a tile first touched on bare canvas has this
            // stroke's own minted tile in the base by now, and falling back to it
            // would re-derive from that — exactly the compounding the pristine
            // handle exists to rule out.
            let pristine = match self.tiles.get(coord) {
                Some(t) => t.pristine.clone(),
                None => self.scene.base.get(coord).cloned(),
            };
            if pristine.is_none() && matches!(self.bare, BareCanvas::Skip) {
                continue;
            }

            // This piece's working parcel: the carried total copied in, or a clear
            // for a first touch — either way every texel is written before the
            // landing pass reads it, the pool's no-zero-init contract (`scratch`).
            // The carried lanes themselves are only ever read: the live tail resumes
            // the same frozen carry on every pointer move.
            let work = Parcel::take(self.r, self.keys);
            let resumed = self.tiles.get(coord).map(|t| Arc::clone(&t.accum));
            if let Some(old) = &resumed {
                for ((src, dst), key) in old.lanes.iter().zip(&work.lanes).zip(self.keys) {
                    // A hole in one list is a hole in all three: the keys made both
                    // parcels, so the `None`s line up and nothing is copied there.
                    if let (Some(src), Some(dst), Some(key)) = (src, dst, key) {
                        self.scope.encoder().copy_texture_to_texture(
                            src.tex().as_image_copy(),
                            dst.tex().as_image_copy(),
                            key.extent(),
                        );
                    }
                }
            }

            // This tile's segments into the working parcel. Just this tile's: every
            // other segment differences its prefix-τ taps to zero here anyway, so
            // what the instance range skips is the shading, not a contribution.
            {
                let ops = if resumed.is_some() {
                    desc::LOAD
                } else {
                    desc::CLEAR
                };
                let att = work.attachments(ops);
                let mut pass =
                    self.scope
                        .encoder()
                        .begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some(sweep.label),
                            color_attachments: &att[..work.len()],
                            ..Default::default()
                        });
                pass.set_pipeline(sweep.pipeline);
                pass.set_bind_group(0, &sweep.draws.xforms, &[sweep.draws.xform_offset(i)]);
                pass.set_bind_group(1, sweep.prefix, &[]);
                pass.set_bind_group(2, sweep.noise, &[]);
                pass.set_vertex_buffer(0, sweep.draws.instances.slice(..));
                pass.draw(0..SWEEP_VERTS, sweep.draws.runs[i].clone());
            }

            // The whole stroke's parcel so far, landed on the pristine paint — never
            // on the base in hand, which for a resumed tile is an earlier piece's
            // output.
            let dst = self
                .r
                .acquire_tile(self.scene.pool, AllocSource::IntegrateDestination);
            let mask = self.r.selection.mask_for(self.scene.selection, *coord);
            // Scoped so the landing's borrow of `pristine` ends before it moves into
            // the carry below. The group holds its own references to every view in
            // it, so it outlives what described it.
            let bg = {
                let landing = Landing {
                    base: base_targets(self.r, pristine.as_ref()),
                    parcel: &work,
                    mask: mask.view(),
                };
                bind(&landing)
            };
            {
                let targets = dst.targets();
                let att = targets.attachments(desc::CLEAR);
                let mut pass =
                    self.scope
                        .encoder()
                        .begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some(land.label),
                            color_attachments: &att[..targets.count()],
                            ..Default::default()
                        });
                pass.set_pipeline(land.pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.draw(0..3, 0..1);
            }

            self.map = self.map.insert(*coord, dst);
            self.dirty.push(*coord);
            self.tiles.insert(
                *coord,
                ParcelTile {
                    pristine,
                    accum: Arc::new(work),
                },
            );
        }

        // Submit before the carry leaves this call: a `Kept` may reach the pool's
        // free list only behind the submit of the commands naming it, and handing the
        // carry out first would let a caller drop it ahead of one. Consuming `self`
        // is what says so — there is no accumulator left to hand anything out of
        // until this has happened.
        //
        // `finish` submits only when the scope has something recorded, so what the
        // sentence above rests on is that a lease worth protecting implies a recording:
        // every `Kept` this call can put into the carry came from a `Parcel::take` past
        // the `BareCanvas::Skip` arm, and every one of those is then named by a copy, a
        // sweep pass and a landing pass — all through `scope.encoder()`, which opens the
        // piece. A carry with nothing recorded behind it holds only `Arc` clones of the
        // *previous* carry's parcels, whose commands an earlier run already submitted.
        self.scope.finish();
        Landed {
            map: self.map,
            carry: ParcelCarry { tiles: self.tiles },
            dirty: self.dirty,
        }
    }
}

/// The pristine tile's channels as a landing pass reads them, with the renderer's
/// 1×1 zeroes standing in where the layer has nothing (§6.8's pattern) — the landing
/// shaders clamp their loads, so bare canvas costs no tile at all, where acquiring a
/// real pooled trio would mean allocating and clearing one on every pointer move
/// whether or not the stroke reached anything unpainted.
pub(super) fn base_targets<'a>(
    r: &'a StrokeRenderer,
    pristine: Option<&'a TilePairHandle>,
) -> Targets<'a> {
    Targets {
        color: pristine.map_or(&r.zeroes.color, TilePairHandle::color_view),
        aux: pristine.map_or(&r.zeroes.aux, TilePairHandle::aux_view),
        // The resident residual, or the 1×1 zero on bare canvas — the same pairing
        // the color above makes, since the two are one color (§6.7).
        resid: r.zeroes.resid.as_ref().map(|zero| {
            pristine
                .and_then(TilePairHandle::resid_view)
                .unwrap_or(zero)
        }),
    }
}
