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
//! - `color`: `Rgba16Float`, latent color premultiplied by **opacity**
//!   (`L·op, a·op, b·op, op`) — opacity, *not* coverage.
//! - `aux`: `R16Float`, `(height)` — the amount of paint, from which the media pass
//!   gets impasto thickness (height − substrate height) and combines opacity ×
//!   thickness into the visible alpha. Height is the *only* persistent auxiliary
//!   channel: gloss is a uniform property of paint (§6.3), not something a stroke
//!   stores per texel.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, Weak};

use crate::gpu::context::GpuContext;
use crate::unpoisoned;
use stark_model::geom::{TILE_APRON, TILE_SIZE, TILE_TEX, TileCoord, Vec2};

// —— the tile texture's geometry, as the shaders address it ——————————————————
//
// These are the engine's, not the document's, and they moved here from
// `stark_model::geom` for the reason `io.rs` gives for not recording `TILE_SIZE` in
// a save file: *an implementation detail is not a fact about a painting*. Nothing in
// the model reads any of them — a footprint quantizes against `TILE_SIZE` and pads
// by `TILE_APRON`, which is the whole of what a *log* is addressed in — while a
// UV bias, a mask tile's edge length and where its texture starts are all questions
// about how a pass samples a texture, which is this crate's business and nobody
// else's.

/// Maps a tile's interior quad corner (`∈ [0, 1]`) to a UV coordinate in the
/// apron'd texture: `uv = corner * INTERIOR_UV_SCALE + INTERIOR_UV_BIAS`. The
/// compositor and presenter sample only the interior sub-rect; bilinear taps at
/// the interior edge then fall into the apron (neighbor content), not a clamp.
pub const INTERIOR_UV_SCALE: f32 = TILE_SIZE as f32 / TILE_TEX as f32;
pub const INTERIOR_UV_BIAS: f32 = TILE_APRON as f32 / TILE_TEX as f32;

/// The mask tile's edge length, for the shaders that place it in a region.
pub const MASK_TEX: u32 = TILE_TEX;

/// The tile geometry a mask tile is rasterized over: its texture's top-left in canvas
/// px (the interior origin, shifted out by the apron — §6.4).
pub fn mask_tex_origin(coord: TileCoord) -> Vec2 {
    coord.origin() - Vec2::splat(TILE_APRON as f32)
}

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

/// What a texture was taken out of the pool *for*.
///
/// It earns its place twice over, which is worth saying because it is otherwise the
/// shape of plumbing added for a log. At the 26 call sites it is documentation the
/// compiler keeps honest — `AllocSource::TransformScratch` says what the acquire is,
/// where a bare `acquire_tex(format)` would say only that one happened. And it is
/// the only way to answer the question a large pool actually raises: not *how many*
/// textures are out, which `capacity` already reports, but **who is holding them**.
///
/// What it must not be is a cost. The census it feeds is an array indexed by
/// discriminant ([`Census`]), not a map: incrementing it is an add, on a path that
/// runs thousands of times a second and that §6.2's whole allocation-rate argument
/// is about.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum AllocSource {
    #[default]
    Unknown,
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
    /// A tile rewritten by a layer merge-down (§14.11).
    MergeDestination,
    /// A merge's composited scratch: the two layers expanded into what they
    /// composite to, and the blend between them (§14.11).
    MergeScratch,
    /// A tile of an image brought in from outside the document (§23).
    PlacedImage,
}

impl AllocSource {
    /// Every variant, in discriminant order — what [`Census`] indexes by.
    ///
    /// [`Self::name`] below has no wildcard, so adding a variant is a compile error
    /// there; this array is what that error exists to remind you to extend, and
    /// `a_census_slot_belongs_to_the_source_that_indexes_it` checks the two agree.
    /// A variant that slipped past both would go uncounted rather than out of
    /// bounds — telemetry degrading is the right failure for telemetry.
    const ALL: [Self; 12] = [
        Self::Unknown,
        Self::IntegrateDestination,
        Self::StrokeScratch,
        Self::DynamicsWriteback,
        Self::SelectionMask,
        Self::TransformScratch,
        Self::TransformDestination,
        Self::TransformMask,
        Self::FillDestination,
        Self::MergeDestination,
        Self::MergeScratch,
        Self::PlacedImage,
    ];

    const fn name(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::IntegrateDestination => "integrate destination",
            Self::StrokeScratch => "stroke scratch",
            Self::DynamicsWriteback => "dynamics writeback",
            Self::SelectionMask => "selection mask",
            Self::TransformScratch => "transform scratch",
            Self::TransformDestination => "transform destination",
            Self::TransformMask => "transform mask",
            Self::FillDestination => "fill destination",
            Self::MergeDestination => "merge destination",
            Self::MergeScratch => "merge scratch",
            Self::PlacedImage => "placed image",
        }
    }
}

/// How many of the pool's textures each [`AllocSource`] is holding.
///
/// An array rather than a `HashMap<AllocSource, usize>`, which is what this was: the
/// map cost a hash on every acquire *and* every release, both under the pool's lock,
/// to serve one `tracing::debug!`. Indexing by discriminant makes the same census an
/// increment.
#[derive(Default)]
struct Census([usize; AllocSource::ALL.len()]);

impl Census {
    /// The slot for `source`, or `None` for a variant missing from
    /// [`AllocSource::ALL`] — see there for why that degrades rather than panics.
    fn slot(&mut self, source: AllocSource) -> Option<&mut usize> {
        self.0.get_mut(source as usize)
    }

    fn add(&mut self, source: AllocSource) {
        if let Some(live) = self.slot(source) {
            *live += 1;
        }
    }

    fn remove(&mut self, source: AllocSource) {
        if let Some(live) = self.slot(source) {
            // Saturating rather than asserted: this runs from `Drop`, where a panic
            // during an unwind is an abort, and a miscount is a wrong log line.
            *live = live.saturating_sub(1);
        }
    }
}

impl std::fmt::Debug for Census {
    /// Only the sources actually holding something, by name — which is both shorter
    /// and more useful than the map's output, since that printed in whatever order
    /// the hashing gave and kept every source it had ever seen, at zero.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map()
            .entries(
                AllocSource::ALL
                    .iter()
                    .zip(self.0)
                    .filter(|(_, live)| *live > 0)
                    .map(|(source, live)| (source.name(), live)),
            )
            .finish()
    }
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
    /// The pool stamp this slot was handed back on — what the trim's quarantine
    /// orders it against ([`PoolInner::epoch_start`]).
    returned: u64,
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
        // something a return would violate. [`unpoisoned`] hands the texture back;
        // an `if let Ok(..)` would drop it on the floor — never recycled, and
        // `capacity` never told, so the pool would quietly grow a replacement for a
        // texture it still owned.
        let mut inner = unpoisoned(pool.lock());
        let Some(tex) = self.tex.take() else { return };
        inner.stamp += 1;
        let returned = inner.stamp;
        // The view rides back with its texture, so the next acquire of this slot
        // needs no `create_view` (see [`Pooled`]). Cloning it is an `Arc` bump.
        inner.free.push(Pooled {
            tex,
            view: self.view.clone(),
            returned,
        });
        inner.sources.remove(self.source);
    }
}

/// A handle to one pooled texture; cloning is an `Arc` bump, and the texture returns
/// to its format's free list when the last handle drops.
///
/// This is the unit the pool deals in. Pairing two of them into a tile is the
/// *caller's* job, because which two formats make a tile is the color space's
/// business (§6.7) and the pool has no view of that.
///
/// **A handle hands out a view and never the texture**, and that is what keeps a
/// recycled [`Pooled`] slot's view valid. The view outlives any one checkout, so a
/// consumer that could reach the `wgpu::Texture` could `destroy()` it and leave the
/// free list holding a view onto nothing — which the next acquire would hand to a
/// bind group, and which no test would catch until a driver complained. Nothing
/// needs the texture today (the accessors that offered one had no callers at all),
/// so the way to rule that out is not to offer it. Re-adding one means reading this
/// paragraph first, which is the point. The one consumer that needs the texture *as
/// a copy destination* — the dynamics write-back — gets [`Self::copy_into`], which
/// encodes the command in here and hands nothing out, so the class stays ruled out
/// rather than re-opened with a caveat.
#[derive(Clone)]
pub struct TexHandle(Arc<GpuTex>);

impl TexHandle {
    pub fn view(&self) -> &wgpu::TextureView {
        &self.0.view
    }

    /// Encode a copy of one full `TILE_TEX` block out of `src` at `origin` into this
    /// texture — the write-back's slice, as a bit-exact copy (§6.2/§6.4).
    ///
    /// The formats must match (a copy's are required to), which is the write-back's
    /// own guarantee: it copies each tile channel from a region-sized texture of that
    /// channel's format.
    pub fn copy_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Texture,
        origin: wgpu::Origin3d,
    ) {
        let dst = self
            .0
            .tex
            .as_ref()
            .expect("a live handle holds its texture");
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: src,
                mip_level: 0,
                origin,
                aspect: wgpu::TextureAspect::All,
            },
            dst.as_image_copy(),
            wgpu::Extent3d {
                width: TILE_TEX,
                height: TILE_TEX,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Fill this texture with one full `TILE_TEX` block computed on the **CPU** —
    /// what a placed image's tiles are built by (§23).
    ///
    /// The only writer here that is not a render pass, and the only one that needs to
    /// be: an imported image's texels are already the answer, so there is nothing for a
    /// shader to compute from them and a pass would be a round trip through the GPU to
    /// copy a buffer. It is also what makes those tiles adapter-independent to the
    /// byte, which no pass in this engine is.
    ///
    /// `bytes` is the whole block, row-major, `TILE_TEX` rows of this format's texel
    /// size — the caller builds it, because what a texel *means* is the channel's
    /// business and not the pool's. A queue write is ordered against everything
    /// submitted after it, so a tile written here is safe to read in the next pass
    /// without a fence.
    ///
    /// # Panics
    ///
    /// Panics unless `bytes` is exactly one block, which is a caller arithmetic error
    /// rather than a state to handle.
    pub fn write_block(&self, queue: &wgpu::Queue, bytes: &[u8]) {
        let dst = self
            .0
            .tex
            .as_ref()
            .expect("a live handle holds its texture");
        let texel = dst
            .format()
            .block_copy_size(None)
            .expect("uncompressed tile format");
        let row = texel * TILE_TEX;
        assert_eq!(
            bytes.len() as u32,
            row * TILE_TEX,
            "a block write is exactly one {TILE_TEX}\u{00D7}{TILE_TEX} tile",
        );
        queue.write_texture(
            dst.as_image_copy(),
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(TILE_TEX),
            },
            wgpu::Extent3d {
                width: TILE_TEX,
                height: TILE_TEX,
                depth_or_array_layers: 1,
            },
        );
    }
}

/// A tile's channels (`color` + `aux`, and a pigment space's `resid`), each a
/// separately pooled texture.
struct TilePair {
    color: TexHandle,
    aux: TexHandle,
    /// The **residual** channel (§6.7) — present exactly when the document's color
    /// space declares a `resid_format`, which is a property of the space and so is
    /// the same for every tile of a document.
    ///
    /// `Option` rather than a texture every space allocates: Oklab has no residual,
    /// and giving it one would be eight bytes a texel of zeroes on the default
    /// space's tiles, plus a third attachment through every pass that writes one.
    resid: Option<TexHandle>,
    /// Pass A's bind group over the three channels above, built on first composite
    /// and kept for the rest of this tile's life ([`TilePairHandle::composite_bg`]).
    composite_bg: OnceLock<wgpu::BindGroup>,
}

/// A layer's painted tiles: sparse, so only populated ones exist, and persistent,
/// so a `DocState` snapshot of one costs a handful of `Arc` bumps (§5.1, §6.1).
/// The sparsity *is* the infinite canvas.
pub type TileMap = rpds::HashTrieMap<stark_model::geom::TileCoord, TilePairHandle>;

/// A selection's coverage tiles, in the very same sparse map the paint lives in —
/// which is what lets a mask be feathered, unbounded, and free to snapshot (§6.8).
pub type MaskMap = rpds::HashTrieMap<stark_model::geom::TileCoord, MaskHandle>;

/// A handle to a tile. Cloning is cheap (Arc bumps), which is what makes persistent
/// `DocState` snapshots cheap (§5.1).
///
/// Built from two [`TexHandle`]s rather than acquired from the pool: see
/// [`TexHandle`] for why the pool does not know what a tile is.
#[derive(Clone)]
pub struct TilePairHandle(Arc<TilePair>);

impl TilePairHandle {
    pub fn new(color: TexHandle, aux: TexHandle, resid: Option<TexHandle>) -> Self {
        TilePairHandle(Arc::new(TilePair {
            color,
            aux,
            resid,
            composite_bg: OnceLock::new(),
        }))
    }

    /// Pass A's bind group over this tile's channels, built by `make` the first
    /// time it is asked for and kept thereafter.
    ///
    /// **The cache is sound because a tile is immutable.** Its texels are never
    /// rewritten once a commit lands — copy-on-write hands out a fresh tile instead
    /// (§5.2), which is the same property [`Self::same`] rests on — so a bind group
    /// naming this tile's three views describes it correctly for as long as it
    /// exists. It is dropped with the tile, so the pool reclaims the textures and
    /// the group naming them together, and no eviction policy is needed.
    ///
    /// **And the layout cannot change under it.** A bind group answers to one
    /// `BindGroupLayout`, which here is the compositor's `tile_bgl` — a function of
    /// the color space alone (§6.7). Every consumer that composites a given tile
    /// shares one `CompositorPasses`: a sibling engine is handed the very same `Arc`
    /// ([`Engine::new_sharing`]), and the one thing that builds a *different* one is
    /// a color-space rebuild, which replaces the tile pool and requires an empty
    /// document (`rebuild_gpu_for`) — so no tile survives it to be asked twice.
    ///
    /// What this replaces is a bind group per tile, per layer, **per frame**. The
    /// visible tile count scales as 1/zoom², so a zoomed-out multi-layer document
    /// was creating ~10⁵ of them a frame — on the web, a JS object apiece, which is
    /// the allocation *rate* `ScopedResources` and the pool's own [`Pooled`] exist
    /// to keep down (§6.2). Now only a newly painted or newly loaded tile pays.
    ///
    /// [`Engine::new_sharing`]: crate::Engine::new_sharing
    pub(crate) fn composite_bg(&self, make: impl FnOnce() -> wgpu::BindGroup) -> &wgpu::BindGroup {
        self.0.composite_bg.get_or_init(make)
    }

    pub fn color_view(&self) -> &wgpu::TextureView {
        self.0.color.view()
    }
    pub fn aux_view(&self) -> &wgpu::TextureView {
        self.0.aux.view()
    }
    /// The residual channel's view, or `None` in a space that has no residual
    /// (§6.7). A caller in a pigment document may `expect` it: the space decides,
    /// once, for every tile it ever makes.
    pub fn resid_view(&self) -> Option<&wgpu::TextureView> {
        self.0.resid.as_ref().map(TexHandle::view)
    }

    /// Encode the write-back's slice of this tile out of region-sized channel
    /// textures (§6.2): one `TILE_TEX` block from each at `origin` — the color, the
    /// narrowed aux, and the residual where the color space has one. Copies, so the
    /// tile is bit-identical to the region block it was cut from, and every
    /// rewritten tile's apron to its neighbour's interior (§6.4).
    pub fn copy_from_region(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        color: &wgpu::Texture,
        aux: &wgpu::Texture,
        resid: Option<&wgpu::Texture>,
        origin: wgpu::Origin3d,
    ) {
        // The pairing is the color space's, decided once for tile and region alike
        // (§6.7) — a mismatch here means the two were built against different spaces.
        debug_assert_eq!(
            self.0.resid.is_some(),
            resid.is_some(),
            "tile and region were built against different color spaces"
        );
        self.0.color.copy_into(encoder, color, origin);
        self.0.aux.copy_into(encoder, aux, origin);
        if let (Some(t), Some(src)) = (&self.0.resid, resid) {
            t.copy_into(encoder, src, origin);
        }
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
pub struct MaskHandle(TexHandle, Arc<OnceLock<wgpu::BindGroup>>);

impl MaskHandle {
    pub fn view(&self) -> &wgpu::TextureView {
        self.0.view()
    }

    /// Pass C's bind group over this mask tile, built by `make` on first use and
    /// kept for the tile's life — [`TilePairHandle::composite_bg`] for masks, sound
    /// for the same reason: a mask tile is rasterized afresh rather than rewritten,
    /// so identity doubles as "unchanged" ([`Self::same`]).
    ///
    /// **Named for its one consumer**, unlike the paint tile's, because a mask is
    /// bound through three different layouts — the overlay's here, the transform's
    /// `mask_src_bgl`, the stamp loop's `region_tile_bgl` — and one cache slot can
    /// only answer for one of them. The other two are per *action* rather than per
    /// frame, so they have nothing to gain and no slot here to take by mistake.
    pub(crate) fn overlay_bg(&self, make: impl FnOnce() -> wgpu::BindGroup) -> &wgpu::BindGroup {
        self.1.get_or_init(make)
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
    /// Who is holding this pool's textures right now.
    sources: Census,
    /// Monotonic count of returns, stamped onto each [`Pooled`] as it comes back.
    stamp: u64,
    /// The [`Self::stamp`] the current epoch opened at. A slot returned at or after
    /// it came back during *this* epoch and is too young to destroy — see
    /// [`Self::tick`].
    epoch_start: u64,
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
            sources: Census::default(),
            stamp: 0,
            epoch_start: 0,
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
    ///
    /// # The quarantine
    ///
    /// **Only slots returned before this epoch opened may be destroyed**, which is a
    /// second rule on top of the policy above and guards something else entirely.
    ///
    /// "No live handle" is not "no pending GPU work". A texture whose last handle
    /// drops while an unsubmitted encoder still names its view reaches this free list
    /// early — and reuse alone makes that wrong pixels, which is bad but recoverable
    /// and is what the consumers' submit scopes exist to prevent
    /// ([`TileScope`](crate::gpu::submit::TileScope)). `destroy()` makes the same
    /// mistake a *dangling view*, handed to the next bind group: a device error, from
    /// a pool that cannot see which of its consumers was careful.
    ///
    /// So the irreversible half waits. An epoch is [`TRIM_INTERVAL`] acquires and an
    /// encoder spans one operation, so a slot that has survived a whole epoch on the
    /// free list is long past any encoder that could still name it. Reuse is
    /// unaffected and stays immediate: this delays only the `destroy`, and costs a
    /// burst one extra epoch before its surplus starts to drain.
    fn tick(&mut self, format: wgpu::TextureFormat) {
        self.peak = self.peak.max(self.in_use());
        self.countdown = self.countdown.saturating_sub(1);
        if self.countdown > 0 {
            return;
        }
        let returns: Vec<u64> = self.free.iter().map(|slot| slot.returned).collect();
        let eligible = quarantine_passed(&returns, self.epoch_start);
        let drop = surplus_to_release(self.capacity, self.peak, eligible.len());
        if drop > 0 {
            // Removed by descending index, so each `swap_remove` cannot move a slot
            // this loop has yet to take.
            let mut taken: Vec<usize> = eligible[..drop].to_vec();
            taken.sort_unstable_by(|a, b| b.cmp(a));
            for i in taken {
                // Explicitly, not merely by dropping the handle. On the web a dropped
                // texture only releases its JS object and waits for GC, which is the
                // opposite of what a trim is for; `destroy()` hands the memory back as
                // soon as the in-flight work referencing it retires. The view beside it
                // goes with it and is never read again ([`Pooled`]).
                self.free.swap_remove(i).tex.destroy();
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
        // Everything on the list from here on has survived a full epoch by the time
        // the next boundary asks.
        self.epoch_start = self.stamp;
    }
}

/// How many free slots an epoch ending in this state releases — the whole of the
/// trim policy, as arithmetic with no GPU in it.
///
/// `capacity − peak` is what the pool owns beyond anything the epoch ever needed at
/// once. Half of that goes back, and never more than is actually idle.
///
/// `free` is the count the trim may actually take: idle **and** past the quarantine
/// ([`PoolInner::tick`]). Passing the eligible count rather than the whole free list
/// is what keeps this function the only place the arithmetic lives — a young slot is
/// simply not offered, rather than being subtracted somewhere else afterwards.
///
/// The `min` is not defensive padding: `free = capacity − in_use` and `in_use ≤ peak`
/// together already prove `free ≥ surplus`, so the clamp is unreachable for the whole
/// list. With the quarantine it is genuinely reachable — a pool whose surplus all
/// came back this epoch offers nothing — and it is what turns that into "release
/// nothing yet" instead of truncating a `Vec` past its length.
/// Which free-list slots have served the trim's quarantine, oldest first: those
/// returned before the current epoch opened ([`PoolInner::tick`]).
///
/// Oldest first so a repeated trim drains the quarantine in the order slots entered
/// it, rather than stranding the same young ones at every boundary.
///
/// Taken as the slots' return stamps rather than the slots, so the rule is decidable
/// without a GPU — the whole of it is an ordering on `u64`s, and a texture would only
/// stop it being tested.
fn quarantine_passed(returns: &[u64], epoch_start: u64) -> Vec<usize> {
    let mut passed: Vec<usize> = (0..returns.len())
        .filter(|&i| returns[i] < epoch_start)
        .collect();
    passed.sort_unstable_by_key(|&i| returns[i]);
    passed
}

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

/// One `TILE_TEX` square of `format`, in bytes.
///
/// Asked of the format rather than tabulated, for [`bytes_per_texel`]'s reason one
/// level up: a table would be a second statement of something wgpu already knows, and
/// a wrong entry would under-report memory rather than fail. A format that cannot
/// answer is counted at zero — it is telemetry, and telemetry degrading is the right
/// failure for telemetry.
///
/// [`bytes_per_texel`]: crate::gpu::readback
fn texture_bytes(format: wgpu::TextureFormat) -> u64 {
    let texel = format.block_copy_size(None).unwrap_or(0) as u64;
    texel * TILE_TEX as u64 * TILE_TEX as u64
}

/// Recycling allocator for tile textures (§6.1). Hands out one texture at a
/// time, keyed by format, so `Rgba16Float` textures are shared across every consumer
/// that needs one (persistent color, scratch color, the wide scratch aux).
#[derive(Clone)]
pub struct TilePool {
    ctx: GpuContext,
    format_pools: HashMap<wgpu::TextureFormat, Arc<Mutex<PoolInner>>>,
}

impl TilePool {
    /// A pool serving `formats` — the color space's `color` and `aux` (§6.7), which
    /// are the only formats a caller knows — **plus the two the pool defines itself**.
    ///
    /// [`MASK_FORMAT`] and [`SCRATCH_AUX_FORMAT`] are unioned in here rather than
    /// asked of the caller, because they are this module's constants and a call site
    /// that had to remember them could forget one. That is not hypothetical: the
    /// scratch aux was omitted, and the omission was invisible only because
    /// `SCRATCH_AUX_FORMAT` happens to equal both color spaces' `color_format` —
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
        MaskHandle(
            self.acquire_tex(MASK_FORMAT, source),
            Arc::new(OnceLock::new()),
        )
    }

    /// Acquire one pooled texture of `format`, reusing a recycled one when available.
    /// Contents are undefined until painted or cleared.
    ///
    /// A recycled slot brings its **view** with it ([`Pooled`]), so the common path
    /// creates no wgpu objects at all — it is a `Vec::pop` and an `Arc::new`.
    ///
    /// **The device call is made outside the lock**, which is the whole reason this
    /// is two phases rather than one. `create_texture` is the single slowest thing
    /// the pool can do and it used to run *inside* the critical section, so a miss on
    /// one thread stalled every other thread's `Vec::pop` behind a driver call — and
    /// `Drop for GpuTex`, which returns a texture from whichever thread happened to
    /// drop it, is exactly such a thread. The accounting still happens under the lock
    /// and in the same order: a miss books its capacity there, so a second acquirer
    /// arriving in the gap sees the texture as already owned and does not double-count
    /// the peak. What that trades is a transient over-report from
    /// [`resident_bytes`](Self::resident_bytes) — bounded by the number of acquires
    /// in flight, and it is telemetry.
    ///
    /// # Panics
    ///
    /// Panics if `format` was not among those the pool was built with.
    pub fn acquire_tex(&self, format: wgpu::TextureFormat, source: AllocSource) -> TexHandle {
        let pool = self.format_pools.get(&format).expect("unsupported format");
        // Phase one, under the lock: take a recycled slot if there is one, and book
        // the acquire against the epoch either way.
        let recycled = {
            // Poison recovered from, not propagated ([`unpoisoned`]). This is the
            // hottest path in the crate — a stroke acquires ~4 per affected tile per
            // pointer move — so a panic here is a renderer that never draws again,
            // which is precisely the outcome that helper exists to rule out.
            let mut inner = unpoisoned(pool.lock());
            inner.sources.add(source);
            let slot = inner.free.pop();
            if slot.is_none() {
                // Booked before the texture exists, so the count is monotonic from
                // every other thread's point of view — see the note on this function.
                inner.increase_capacity(format);
            }
            // After the slot is out (or the capacity booked), so `in_use` counts it:
            // the epoch's peak is what the pool had checked out at its busiest, this
            // acquire included.
            inner.tick(format);
            slot
        };
        // Phase two, unlocked: build the texture a miss asked for.
        let Pooled { tex, view, .. } = match recycled {
            Some(slot) => slot,
            None => self.create_pooled(format),
        };
        TexHandle(Arc::new(GpuTex {
            tex: Some(tex),
            view,
            pool: Arc::downgrade(pool),
            source,
        }))
    }

    /// How many bytes of tile texture this pool **owns**, across every format:
    /// what its consumers are holding plus what is idle on its free lists.
    ///
    /// The number a history-retention policy is measured against (§5). It is what
    /// the pool owns rather than what is in use, deliberately: a texture on the free
    /// list has been paid for and is not given back to the driver until an epoch
    /// boundary decides the pool no longer needs it ([`PoolInner::tick`]), so from
    /// the process's point of view it is resident either way.
    ///
    /// Derived rather than tracked, so it cannot drift from the capacity it is a
    /// function of. `O(formats)` — four of them — under each format's lock in turn,
    /// which is why it is safe to ask on a commit but not per tile.
    pub fn resident_bytes(&self) -> u64 {
        self.format_pools
            .iter()
            .map(|(format, pool)| {
                let capacity = unpoisoned(pool.lock()).capacity as u64;
                capacity * texture_bytes(*format)
            })
            .sum()
    }

    /// How many recycled textures of `format` are idle — what the pool would serve
    /// the next acquires from without touching the device.
    ///
    /// Takes the format rather than assuming one: the pool's whole design is that free
    /// lists are *per format*, so an answer that assumed `Rgba16Float` would quietly
    /// tell a caller asking about the aux or the mask list about the color one.
    pub fn free_count(&self, format: wgpu::TextureFormat) -> usize {
        unpoisoned(
            self.format_pools
                .get(&format)
                .expect("unsupported format")
                .lock(),
        )
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
        // Stamped when it is handed *back*; a texture on its way out has not
        // served any quarantine and never will be asked to (see `Drop for GpuTex`).
        Pooled {
            tex,
            view,
            returned: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`Census`] indexes by discriminant, so a slot only means anything if
    /// [`AllocSource::ALL`] lists the variants in that order. Reordering the enum
    /// without reordering the array would silently attribute every acquire to the
    /// wrong subsystem — a wrong answer to the one question the census exists for,
    /// and one nothing else would contradict.
    #[test]
    fn a_census_slot_belongs_to_the_source_that_indexes_it() {
        for (i, source) in AllocSource::ALL.iter().enumerate() {
            assert_eq!(
                *source as usize, i,
                "{source:?} is listed at {i} but indexes {}",
                *source as usize,
            );
        }
        // And every variant has a slot to land in: a new one missing from `ALL`
        // would index past the end and go uncounted.
        let mut census = Census::default();
        for source in AllocSource::ALL {
            assert!(
                census.slot(source).is_some(),
                "{source:?} has no census slot",
            );
        }
    }

    /// The census reports who is holding textures, and stops reporting a source once
    /// it has handed everything back — the map it replaced kept every source it had
    /// ever seen, at zero.
    #[test]
    fn the_census_names_only_live_sources() {
        let mut census = Census::default();
        census.add(AllocSource::StrokeScratch);
        census.add(AllocSource::StrokeScratch);
        census.add(AllocSource::FillDestination);
        let live = format!("{census:?}");
        assert!(live.contains("stroke scratch"), "{live}");
        assert!(live.contains("fill destination"), "{live}");
        assert!(
            !live.contains("unknown"),
            "reported a source holding nothing: {live}"
        );

        census.remove(AllocSource::FillDestination);
        let live = format!("{census:?}");
        assert!(!live.contains("fill destination"), "{live}");
        assert!(live.contains("stroke scratch"), "{live}");
    }

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

    /// **A slot returned during this epoch is never destroyed at its end**, however
    /// large the surplus looks — the quarantine ([`PoolInner::tick`]).
    ///
    /// The stake is the difference between a wrong pixel and a dangling view: a
    /// texture handed back while an unsubmitted encoder still names it survives being
    /// *reused*, but not being `destroy()`ed.
    #[test]
    fn a_trim_never_destroys_a_slot_returned_this_epoch() {
        // Twenty slots the pool owns and is sitting on, all returned just now — the
        // shape a burst leaves behind. The epoch has not turned over since, so none
        // of them has served its quarantine.
        let capacity = 20;
        let returns: Vec<u64> = (1..=20).collect();
        let eligible = quarantine_passed(&returns, 0);
        assert!(
            eligible.is_empty(),
            "a slot returned this epoch is not eligible",
        );
        assert_eq!(
            surplus_to_release(capacity, 0, eligible.len()),
            0,
            "with nothing eligible the trim must release nothing",
        );

        // One boundary later the same slots are old enough, and the ordinary policy
        // takes over: half the surplus.
        let eligible = quarantine_passed(&returns, 21);
        assert_eq!(
            eligible.len(),
            20,
            "after an epoch the whole burst is eligible"
        );
        assert_eq!(surplus_to_release(capacity, 0, eligible.len()), 10);
    }

    /// The quarantine splits a mixed list rather than shifting the whole thing:
    /// slots from before the boundary are taken, this epoch's are held back — and the
    /// old ones come out **oldest first**, so a repeated trim drains them in order
    /// instead of stranding the same slots at every boundary.
    #[test]
    fn the_quarantine_takes_the_old_slots_oldest_first() {
        // Interleaved on purpose: the free list is a stack, so age and position are
        // not the same order and the rule must not assume they are.
        let returns = [7u64, 2, 9, 1, 5];
        let eligible = quarantine_passed(&returns, 6);
        assert_eq!(
            eligible,
            vec![3, 1, 4],
            "expected the pre-boundary slots (1, 2, 5) by age",
        );
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
