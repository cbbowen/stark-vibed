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

#[derive(Default)]
struct PoolInner {
    /// Recycled textures and their views, one free list per format ([`Pooled`]).
    free: Vec<Pooled>,
    /// How many textures this pool has ever created — its high-water mark, since
    /// nothing here is ever handed back to the driver.
    capacity: usize,
    /// Current allocation sources.
    sources: HashMap<AllocSource, usize>,
}

impl PoolInner {
    fn increase_capacity(&mut self, format: wgpu::TextureFormat) {
        self.capacity += 1;
        tracing::debug!(format = ?format, capacity = self.capacity, sources = ?self.sources, "increased texture pool capacity");
    }
}

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
            match pool.free.pop() {
                Some(slot) => slot,
                None => {
                    pool.increase_capacity(format);
                    self.create_pooled(format)
                }
            }
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
