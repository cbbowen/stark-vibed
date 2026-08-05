//! Tiles and the recycling tile pool (§5.1, §5.2, §6.1).
//!
//! A tile's channels are independent GPU textures, each held through a
//! [`TexHandle`] (`Arc`); when the last handle drops, the texture returns to the
//! [`TilePool`]'s free list for its format — so history retention drives GPU memory
//! reclamation with no manual GC. A [`TilePairHandle`] bundles a tile's `color` + `aux`
//! textures; cloning one is two `Arc` bumps, which is what makes persistent
//! `DocState` snapshots cheap.
//!
//! The pool keys its free lists by **format**, and hands out one texture at a time,
//! so different consumers can mix formats freely. In particular a brush-dynamics
//! *scratch* tile takes a wider `Rgba16Float` aux (an extra channel the deposit and
//! integrate use internally) while persistent tiles keep the compact color-space
//! `aux` format — the two never need to match (§6.2).
//!
//! Channels (§6.1, normalized representation):
//! - `color`: `Rgba16Float`, latent colour premultiplied by **opacity**
//!   (`L·op, a·op, b·op, op`) — opacity, *not* coverage.
//! - `aux`: `R16Float`, `(height)` — the amount of paint, from which the media pass
//!   gets impasto thickness (height − surface height) and combines opacity ×
//!   thickness into the visible alpha. Height is the *only* persistent auxiliary
//!   channel: gloss is a uniform property of paint (§6.3), not something a stroke
//!   stores per texel.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

use crate::geom::TILE_TEX;
use crate::gpu::context::GpuContext;

const CHANNEL_USAGE: wgpu::TextureUsages = wgpu::TextureUsages::TEXTURE_BINDING
    .union(wgpu::TextureUsages::RENDER_ATTACHMENT)
    .union(wgpu::TextureUsages::COPY_SRC)
    .union(wgpu::TextureUsages::COPY_DST);

/// The aux format a brush-dynamics *scratch* tile uses: wider than the persistent
/// `aux` so the deposit can stash an extra channel (the smear-lifted height) for the
/// integrate to read, without disturbing the compact persistent layout (§6.2).
pub const SCRATCH_AUX_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The format of a **selection mask** tile (§6.8): one unsigned-normalized
/// coverage channel. Same `TILE_TEX` geometry (apron included) as a paint tile, so a
/// mask texel is 1:1 with the tile texel it gates and the mask is pooled and recycled
/// exactly like paint.
pub const MASK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AllocSource {
    #[default]
    Unknown,
    IntegrateEmptyBase,
    IntegrateDestination,
    StrokeScratch,
    DynamicsWriteback,
    SelectionMask,
    /// The transform's moved-parcel scratch pair (§16.5).
    TransformScratch,
    /// A tile rewritten by a transform's combine pass.
    TransformDestination,
    /// A selection mask tile carried under a transform's affine.
    TransformMask,
    /// A tile rewritten by a region fill (§18.0.4).
    FillDestination,
}

/// A recycled texture **and the view onto it**, as the free list holds them.
///
/// The view is pooled with the texture rather than made afresh at each acquire, and
/// that is worth a word because it is the one thing a reader might expect to be
/// per-consumer. It is not: every acquire built the same view — the whole texture,
/// through the default descriptor — so making a new one bought nothing but an object.
///
/// The rate is what makes it matter. A stroke acquires ~4 of these per affected tile
/// (a scratch pair, a destination pair) on every pointer move, so a stroke crossing
/// twenty tiles at pen rate was creating thousands of views a second. Natively that
/// is a small allocation and some validation; on the web it is a JS object per
/// acquire, which is precisely the allocation *rate* `ScopedResources` and
/// `UNIFORM_STRIDE` exist to keep down (§6.2) — the pool was quietly the largest
/// remaining source of it.
struct Pooled {
    tex: wgpu::Texture,
    view: wgpu::TextureView,
}

/// One pooled GPU texture (`TILE_TEX` square) checked out of the pool.
///
/// `tex` is an `Option` only so [`Drop`] can move it back to the free list. `view`
/// is not: it is `Clone` (an `Arc` handle), so the return path clones it and the
/// read path — which runs once per bind group, per tile, per frame — stays a plain
/// borrow with nothing to unwrap.
struct GpuTex {
    tex: Option<wgpu::Texture>,
    view: wgpu::TextureView,
    pool: Weak<Mutex<PoolInner>>,
    source: AllocSource,
}

impl Drop for GpuTex {
    /// Return the texture to its format's free list. **Nothing in here may panic.**
    /// A tile handle is as likely to be dropped by an unwind as by an ordinary
    /// scope exit — a stroke's handles unwind with it — and a panic that starts in
    /// a `Drop` during an unwind is an abort, not a caught failure.
    fn drop(&mut self) {
        let Some(pool) = self.pool.upgrade() else {
            return; // the pool is gone; the texture goes with it
        };
        // Recovered from rather than propagated, because the alternative is a leak.
        // A poisoned lock means some other thread panicked holding it; the state it
        // guards is a free list and a counter, and neither can be left saying
        // something a return would violate. Taking `Err`'s inner guard hands the
        // texture back; the `if let Ok(..)` this replaced dropped it on the floor —
        // never recycled, and `capacity` never told, so the pool quietly grew a
        // replacement for a texture it still owned.
        let mut inner = pool
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(tex) = self.tex.take() else { return };
        // The view rides back with its texture, so the next acquire of this slot
        // needs no `create_view` (see [`Pooled`]). Cloning it is an `Arc` bump.
        inner.free.push(Pooled {
            tex,
            view: self.view.clone(),
        });
        // Saturating rather than asserted. Every acquire records its source, so a
        // missing entry is unreachable — and an unreachable branch is not worth the
        // abort that reaching it from a `Drop` would cost.
        if let Some(live) = inner.sources.get_mut(&self.source) {
            *live = live.saturating_sub(1);
        }
    }
}

/// A handle to one pooled texture; cloning is an `Arc` bump, and the texture returns
/// to its format's free list when the last handle drops.
///
/// This is the unit the pool deals in. Pairing two of them into a tile is the
/// *caller's* job, because which two formats make a tile is the colour space's
/// business (§6.7) and the pool has no view of that.
///
/// **A handle hands out a view and never the texture**, and that is what keeps a
/// recycled [`Pooled`] slot's view valid. The view outlives any one checkout, so a
/// consumer that could reach the `wgpu::Texture` could `destroy()` it and leave the
/// free list holding a view onto nothing — which the next acquire would hand to a
/// bind group, and which no test would catch until a driver complained. Nothing
/// needs the texture today (the accessors that offered one had no callers at all),
/// so the way to rule that out is not to offer it. Re-adding one means reading this
/// paragraph first, which is the point.
#[derive(Clone)]
pub struct TexHandle(Arc<GpuTex>);

impl TexHandle {
    pub fn view(&self) -> &wgpu::TextureView {
        &self.0.view
    }
}

/// A tile's two channels (`color` + `aux`), each a separately pooled texture.
struct TilePair {
    color: TexHandle,
    aux: TexHandle,
}

/// A layer's painted tiles: sparse, so only populated ones exist, and persistent,
/// so a `DocState` snapshot of one costs a handful of `Arc` bumps (§5.1, §6.1).
/// The sparsity *is* the infinite canvas.
pub type TileMap = rpds::HashTrieMap<crate::geom::TileCoord, TilePairHandle>;

/// A selection's coverage tiles, in the very same sparse map the paint lives in —
/// which is what lets a mask be feathered, unbounded, and free to snapshot (§6.8).
pub type MaskMap = rpds::HashTrieMap<crate::geom::TileCoord, MaskHandle>;

/// A handle to a tile. Cloning is cheap (Arc bumps), which is what makes persistent
/// `DocState` snapshots cheap (§5.1).
///
/// Built from two [`TexHandle`]s rather than acquired from the pool: see
/// [`TexHandle`] for why the pool does not know what a tile is.
#[derive(Clone)]
pub struct TilePairHandle(Arc<TilePair>);

impl TilePairHandle {
    pub fn new(color: TexHandle, aux: TexHandle) -> Self {
        TilePairHandle(Arc::new(TilePair { color, aux }))
    }
    pub fn color_view(&self) -> &wgpu::TextureView {
        self.0.color.view()
    }
    pub fn aux_view(&self) -> &wgpu::TextureView {
        self.0.aux.view()
    }

    /// Whether two handles are the same allocation. A tile's texels are never
    /// rewritten once a commit lands — copy-on-write hands out a fresh tile
    /// instead (§5.2) — so identity doubles as "unchanged", which is
    /// what the timeline's patch capture diffs by (§12.6).
    pub fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

/// A handle to one pooled **selection mask** tile (§6.8). Cloning is an
/// `Arc` bump, so a `Selection` snapshot is as cheap as a `DocState` one — and the
/// texture returns to the pool when the last history version referencing it drops.
#[derive(Clone)]
pub struct MaskHandle(TexHandle);

impl MaskHandle {
    pub fn view(&self) -> &wgpu::TextureView {
        self.0.view()
    }

    /// Whether two handles are the same allocation — [`TilePairHandle::same`]
    /// for masks, and true for the same reason: a mask tile is rasterized
    /// afresh rather than rewritten, so identity doubles as "unchanged".
    pub fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0.0, &other.0.0)
    }
}

/// How many acquires make one **epoch**: the window the pool measures its own peak
/// demand over, and the cadence it releases surplus textures on.
///
/// Counted in acquires rather than frames or seconds, because that is the only clock
/// the pool honestly has. The engine renders on demand — an idle app paints no
/// frames — so a frame counter would tick fastest exactly when there is nothing to
/// reclaim and stop when there is; and a wall clock has no business in a crate that
/// compiles to wasm and whose whole point is being a deterministic function of an
/// action log (§1).
///
/// **The consequence, stated rather than hidden: an idle pool does not shrink.** A
/// session that does something enormous and then sits still keeps that memory until
/// the next spell of work, which is when the epoch advances and the surplus is
/// measured against what that work actually needs. What this rules out is the thing
/// §5.1 promises and did not deliver — that a peak reached once is resident for the
/// rest of the session.
///
/// 4096 is about fifty stroke renders (a stroke over twenty tiles acquires ~82), so
/// an epoch is roughly a second of painting: long enough that the peak is a real
/// working set rather than one gesture, short enough that a big transform's surplus
/// is gone within seconds of ordinary work.
const TRIM_INTERVAL: u32 = 4096;

struct PoolInner {
    /// Recycled textures and their views, one free list per format ([`Pooled`]).
    free: Vec<Pooled>,
    /// How many textures this pool **owns** — created, less those released back to
    /// the driver by [`Self::trim`]. `capacity - free.len()` is therefore what its
    /// consumers are holding.
    capacity: usize,
    /// The most this pool has had checked out at once during the current epoch.
    peak: usize,
    /// Acquires left before the epoch ends ([`TRIM_INTERVAL`]).
    countdown: u32,
    /// Current allocation sources.
    sources: HashMap<AllocSource, usize>,
}

impl Default for PoolInner {
    /// A full epoch to begin with, so a fresh pool measures a whole window before it
    /// first trims — `countdown: 0` from a derived `Default` would fire on the very
    /// first acquire, against a peak of one.
    fn default() -> Self {
        Self {
            free: Vec::new(),
            capacity: 0,
            peak: 0,
            countdown: TRIM_INTERVAL,
            sources: HashMap::new(),
        }
    }
}

impl PoolInner {
    /// Textures checked out right now. Derived rather than counted: `capacity` is
    /// what the pool owns and `free` is what it is sitting on, so the difference is
    /// what its consumers have.
    fn in_use(&self) -> usize {
        self.capacity - self.free.len()
    }

    fn increase_capacity(&mut self, format: wgpu::TextureFormat) {
        self.capacity += 1;
        tracing::debug!(format = ?format, capacity = self.capacity, sources = ?self.sources, "increased texture pool capacity");
    }

    /// Note this acquire against the epoch, and release surplus at its end.
    ///
    /// **The whole policy is `capacity > peak`.** The most the pool needed at once
    /// during the epoch was `peak`, so anything it owns beyond that was not needed by
    /// *any* moment of it — and since `in_use ≤ peak`, that surplus is provably all
    /// sitting in `free` rather than checked out. Nothing a consumer holds, and
    /// nothing the epoch's busiest instant wanted, can be dropped by it.
    ///
    /// Half the surplus rather than all of it, for hysteresis: demand that alternates
    /// between epochs — a transform, then a stroke, then a transform — would otherwise
    /// hand every texture back and build it again. Halving converges within a few
    /// epochs and costs one epoch's patience.
    fn tick(&mut self, format: wgpu::TextureFormat) {
        self.peak = self.peak.max(self.in_use());
        self.countdown = self.countdown.saturating_sub(1);
        if self.countdown > 0 {
            return;
        }
        let drop = surplus_to_release(self.capacity, self.peak, self.free.len());
        if drop > 0 {
            // Explicitly, not merely by dropping the handle. On the web a dropped
            // texture only releases its JS object and waits for GC, which is the
            // opposite of what a trim is for; `destroy()` hands the memory back as
            // soon as the in-flight work referencing it retires. The view beside it
            // goes with it and is never read again ([`Pooled`]).
            for slot in self.free.drain(self.free.len() - drop..) {
                slot.tex.destroy();
            }
            self.capacity -= drop;
            tracing::debug!(
                format = ?format,
                released = drop,
                capacity = self.capacity,
                peak = self.peak,
                "released surplus texture pool capacity",
            );
        }
        self.peak = self.in_use();
        self.countdown = TRIM_INTERVAL;
    }
}

/// How many free slots an epoch ending in this state releases — the whole of the
/// trim policy, as arithmetic with no GPU in it.
///
/// `capacity − peak` is what the pool owns beyond anything the epoch ever needed at
/// once. Half of that goes back, and never more than is actually idle.
///
/// The `min` is not defensive padding: `free = capacity − in_use` and `in_use ≤ peak`
/// together already prove `free ≥ surplus`, so the clamp is unreachable today. It is
/// there because the alternative to it being unreachable is truncating a `Vec` past
/// its length, and the proof lives in two other functions.
fn surplus_to_release(capacity: usize, peak: usize, free: usize) -> usize {
    let surplus = capacity.saturating_sub(peak);
    (surplus / 2).min(free).min(MAX_RELEASE_PER_EPOCH)
}

/// Most textures one epoch boundary may hand back.
///
/// The trim runs inside `acquire_tex`, holding the pool's lock, so its cost lands on
/// one unlucky acquire — and a transform over a few thousand tiles can leave a
/// surplus in the thousands, half of which is a lot of `destroy()` calls to make
/// while a stroke is waiting for a scratch tile.
///
/// This is a **bound on a cost that has not been measured**, not a tuned figure: it
/// says the spike stays within a few hundred driver calls whatever the surplus, and
/// costs only that a very large one takes more epochs to drain. If profiling ever
/// shows the release is cheap, the honest change is to raise this and say so, not to
/// discover the cap by wondering why reclamation is slow.
const MAX_RELEASE_PER_EPOCH: usize = 256;

/// Recycling allocator for tile textures (§6.1). Hands out one texture at a
/// time, keyed by format, so `Rgba16Float` textures are shared across every consumer
/// that needs one (persistent colour, scratch colour, the wide scratch aux).
#[derive(Clone)]
pub struct TilePool {
    ctx: GpuContext,
    format_pools: HashMap<wgpu::TextureFormat, Arc<Mutex<PoolInner>>>,
}

impl TilePool {
    /// A pool serving `formats` — the colour space's `color` and `aux` (§6.7), which
    /// are the only formats a caller knows — **plus the two the pool defines itself**.
    ///
    /// [`MASK_FORMAT`] and [`SCRATCH_AUX_FORMAT`] are unioned in here rather than
    /// asked of the caller, because they are this module's constants and a call site
    /// that had to remember them could forget one. That is not hypothetical: the
    /// scratch aux was omitted, and the omission was invisible only because
    /// `SCRATCH_AUX_FORMAT` happens to equal both colour spaces' `color_format` —
    /// the very coincidence [`StrokeRenderer::acquire_tile`] warns about one level
    /// down. The first space to choose otherwise would have met `acquire_tex`'s
    /// "unsupported format" panic on its first stroke.
    ///
    /// [`StrokeRenderer::acquire_tile`]: crate::gpu::StrokeRenderer
    pub fn new(ctx: GpuContext, formats: impl IntoIterator<Item = wgpu::TextureFormat>) -> Self {
        let format_pools = formats
            .into_iter()
            .chain([MASK_FORMAT, SCRATCH_AUX_FORMAT])
            .map(|f| (f, Arc::default()))
            .collect();
        Self { ctx, format_pools }
    }

    /// Acquire a selection mask tile ([`MASK_FORMAT`], §6.8). Contents are
    /// undefined until rasterized; the selection renderer always writes the whole
    /// target, aprons included.
    pub fn acquire_mask(&self, source: AllocSource) -> MaskHandle {
        MaskHandle(self.acquire_tex(MASK_FORMAT, source))
    }

    /// Acquire one pooled texture of `format`, reusing a recycled one when available.
    /// Contents are undefined until painted or cleared.
    ///
    /// A recycled slot brings its **view** with it ([`Pooled`]), so the common path
    /// creates no wgpu objects at all — it is a `Vec::pop` and an `Arc::new`.
    ///
    /// # Panics
    ///
    /// Panics if `format` was not among those the pool was built with.
    pub fn acquire_tex(&self, format: wgpu::TextureFormat, source: AllocSource) -> TexHandle {
        let pool = self.format_pools.get(&format).expect("unsupported format");
        let Pooled { tex, view } = {
            let mut pool = pool.lock().expect("tile pool poisoned");
            *pool.sources.entry(source).or_default() += 1;
            let slot = match pool.free.pop() {
                Some(slot) => slot,
                None => {
                    pool.increase_capacity(format);
                    self.create_pooled(format)
                }
            };
            // After the slot is out, so `in_use` counts it: the epoch's peak is what
            // the pool had checked out at its busiest, this acquire included.
            pool.tick(format);
            slot
        };
        TexHandle(Arc::new(GpuTex {
            tex: Some(tex),
            view,
            pool: Arc::downgrade(pool),
            source,
        }))
    }

    /// Number of recycled color-format textures available (for tests).
    pub fn free_count(&self) -> usize {
        let format = wgpu::TextureFormat::Rgba16Float;
        self.format_pools
            .get(&format)
            .expect("unsupported format")
            .lock()
            .expect("tile pool poisoned")
            .free
            .len()
    }

    /// A fresh texture and the view onto it — the only path that talks to the
    /// device, reached once per texture the pool ever owns rather than once per
    /// acquire.
    fn create_pooled(&self, format: wgpu::TextureFormat) -> Pooled {
        let tex = self.ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("stark tile channel"),
            // Interior + apron on every side (§6.4).
            size: wgpu::Extent3d {
                width: TILE_TEX,
                height: TILE_TEX,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: CHANNEL_USAGE,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        Pooled { tex, view }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A trim may never take a texture the epoch needed**, which is the one way
    /// this policy could hurt: it runs on the acquire path, so getting it wrong
    /// trades a memory win for recreating textures inside the work that is using
    /// them.
    ///
    /// The invariant is that the pool never falls below the epoch's peak concurrent
    /// demand, so every acquire the epoch actually made would still have been served
    /// from the free list.
    #[test]
    fn a_trim_never_drops_below_the_epochs_peak_demand() {
        for capacity in [0usize, 1, 2, 7, 64, 1000] {
            for peak in 0..=capacity {
                // Every idle count the state can actually be in: `free` is
                // `capacity − in_use` and `in_use ≤ peak`, so it runs from
                // `capacity − peak` (the epoch's busiest instant, still checked out)
                // up to `capacity` (everything handed back).
                for free in (capacity - peak)..=capacity {
                    let released = surplus_to_release(capacity, peak, free);
                    assert!(
                        released <= free,
                        "released {released} of {free} idle (cap {capacity}, peak {peak})",
                    );
                    assert!(
                        capacity - released >= peak,
                        "cap {capacity} → {} is under the epoch's peak {peak}",
                        capacity - released,
                    );
                }
            }
        }
    }

    /// A pool that is not holding more than it needed releases nothing — the case
    /// that must stay free, since it is every epoch of ordinary painting.
    #[test]
    fn a_pool_at_its_working_set_releases_nothing() {
        for n in [0usize, 1, 50, 4096] {
            assert_eq!(surplus_to_release(n, n, 0), 0, "at exactly the peak");
            // And a pool whose peak exceeded what it owns (impossible, but the
            // saturation must not wrap into an enormous release).
            assert_eq!(surplus_to_release(n, n + 10, n), 0, "peak above capacity");
        }
    }

    /// Surplus decays geometrically rather than all at once, so demand that
    /// alternates between epochs keeps a cushion instead of rebuilding it each time.
    /// Halving also has to actually *converge* — a policy that rounded down to zero
    /// early would strand most of the surplus forever.
    #[test]
    fn surplus_halves_away_over_a_few_epochs() {
        let peak = 4;
        let mut capacity = 1000;
        let mut epochs = 0;
        while capacity > peak {
            let released = surplus_to_release(capacity, peak, capacity - peak);
            if released == 0 {
                break;
            }
            capacity -= released;
            epochs += 1;
            assert!(
                epochs < 40,
                "not converging: still {capacity} after {epochs}"
            );
        }
        // Halving from 1000 to 4 is ~8 epochs; the tail rounds down and stops one
        // slot above the peak, which is a slot and not a leak.
        assert!(
            capacity <= peak + 1,
            "settled at {capacity} for a peak of {peak}"
        );
        assert!(epochs >= 8, "converged implausibly fast ({epochs} epochs)");
    }

    /// No single epoch boundary may hand back an unbounded number of textures: the
    /// trim holds the pool's lock, so its cost falls on one acquire in the middle of
    /// whatever work comes next.
    #[test]
    fn one_epoch_releases_a_bounded_number_of_textures() {
        for capacity in [1_000usize, 100_000, usize::MAX / 2] {
            let released = surplus_to_release(capacity, 0, capacity);
            assert!(
                released <= MAX_RELEASE_PER_EPOCH,
                "a surplus of {capacity} released {released} in one go",
            );
        }
        // And the cap does not stop it converging — a capped release still drains,
        // it just takes more boundaries.
        let mut capacity = 100_000usize;
        let mut epochs = 0;
        while capacity > 4 {
            let released = surplus_to_release(capacity, 4, capacity - 4);
            if released == 0 {
                break;
            }
            capacity -= released;
            epochs += 1;
            assert!(epochs < 1_000, "capped release stalled at {capacity}");
        }
        assert!(capacity <= 5, "settled at {capacity}");
    }
}
