//! Tiles and the recycling tile pool (DESIGN.md §5.1, §5.2, §6.1).
//!
//! A tile's channels are independent GPU textures, each held through a
//! [`TexHandle`] (`Arc`); when the last handle drops, the texture returns to the
//! [`TilePool`]'s free list for its format — so history retention drives GPU memory
//! reclamation with no manual GC. A [`TileHandle`] bundles a tile's `color` + `aux`
//! textures; cloning one is two `Arc` bumps, which is what makes persistent
//! `DocState` snapshots cheap.
//!
//! The pool keys its free lists by **format**, and hands out one texture at a time,
//! so different consumers can mix formats freely. In particular a brush-dynamics
//! *scratch* tile takes a wider `Rgba16Float` aux (an extra channel the deposit and
//! integrate use internally) while persistent tiles keep the compact color-space
//! `aux` format — the two never need to match (DESIGN.md §6.2).
//!
//! Channels (DESIGN.md §6.1, normalized representation):
//! - `color`: `Rgba16Float`, latent colour premultiplied by **opacity**
//!   (`L·op, a·op, b·op, op`) — opacity, *not* coverage.
//! - `aux`: `Rg16Float`, `(thickness, wet)` — impasto thickness and wetness. The
//!   media pass combines opacity × thickness into the visible alpha.

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
/// integrate to read, without disturbing the compact persistent layout (DESIGN §6.2).
pub const SCRATCH_AUX_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AllocSource {
    #[default]
    Unknown,
    IntegrateEmptyBase,
    IntegrateDestination,
    StrokeScratch,
    DynamicsWriteback,
}

/// One pooled GPU texture (`TILE_TEX` square). `Option` only so [`Drop`] can move it
/// back to the pool's free list for its format.
struct GpuTex {
    tex: Option<wgpu::Texture>,
    view: wgpu::TextureView,
    pool: Weak<Mutex<PoolInner>>,
    source: AllocSource,
}

impl Drop for GpuTex {
    fn drop(&mut self) {
        if let Some(pool) = self.pool.upgrade()
            && let Ok(mut inner) = pool.lock()
            && let Some(t) = self.tex.take()
        {
            inner.free.push(t);
            *inner.sources.get_mut(&self.source).expect("source not recorded") -= 1;
        }
    }
}

/// A handle to one pooled texture; cloning is an `Arc` bump.
#[derive(Clone)]
struct TexHandle(Arc<GpuTex>);

impl TexHandle {
    fn texture(&self) -> &wgpu::Texture {
        self.0.tex.as_ref().expect("texture present until drop")
    }
    fn view(&self) -> &wgpu::TextureView {
        &self.0.view
    }
}

// TODO: Remove this. Callers should just get two handles.
/// A tile's two channels (`color` + `aux`), each a pooled texture.
struct TilePair {
    color: TexHandle,
    aux: TexHandle,
}

// TODO: Remove this. Callers should just get two handles.
/// A handle to a tile. Cloning is cheap (Arc bumps), which is what makes persistent
/// `DocState` snapshots cheap (DESIGN.md §5.1).
#[derive(Clone)]
pub struct TilePairHandle(Arc<TilePair>);

impl TilePairHandle {
    pub fn color(&self) -> &wgpu::Texture {
        self.0.color.texture()
    }
    pub fn aux(&self) -> &wgpu::Texture {
        self.0.aux.texture()
    }
    pub fn color_view(&self) -> &wgpu::TextureView {
        self.0.color.view()
    }
    pub fn aux_view(&self) -> &wgpu::TextureView {
        self.0.aux.view()
    }
}

#[derive(Default)]
struct PoolInner {
    /// Recycled textures, one free list per format.
    free: Vec<wgpu::Texture>,
    /// The total number of textures available to this pool.
    capacity: usize,
    /// Current allocation sources.
    sources: HashMap<AllocSource, usize>,
}

impl PoolInner {
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    fn increase_capacity(&mut self, format: wgpu::TextureFormat) {
        self.capacity += 1;
        tracing::debug!(format = ?format, capacity = self.capacity(), sources = ?self.sources, "increased texture pool capacity");
    }
}

/// Recycling allocator for tile textures (DESIGN.md §6.1). Hands out one texture at a
/// time, keyed by format, so `Rgba16Float` textures are shared across every consumer
/// that needs one (persistent colour, scratch colour, the wide scratch aux).
#[derive(Clone)]
pub struct TilePool {
    ctx: GpuContext,
    format_pools: HashMap<wgpu::TextureFormat, Arc<Mutex<PoolInner>>>,
}

impl TilePool {
    pub fn new(
        ctx: GpuContext,
        formats: impl IntoIterator<Item=wgpu::TextureFormat>,
    ) -> Self {
        let format_pools = formats.into_iter().map(|f| (f, Arc::default())).collect();
        Self {
            ctx,
            format_pools,
        }
    }

    // TODO: Replace this with `acquire_tex` calls.
    /// Acquire a persistent tile (color-space `color` + `aux` formats), reusing
    /// recycled textures when available. Contents are undefined until painted or cleared.
    pub fn acquire(&self, source: AllocSource) -> TilePairHandle {
        self.tile(wgpu::TextureFormat::Rg16Float, source)
    }

    // TODO: Replace this with `acquire_tex` calls.
    /// Acquire a brush-dynamics *scratch* tile: the same colour channel, but a wider
    /// [`SCRATCH_AUX_FORMAT`] aux (an extra channel the deposit/integrate use internally).
    pub fn acquire_scratch(&self, source: AllocSource) -> TilePairHandle {
        self.tile(SCRATCH_AUX_FORMAT, source)
    }

    // TODO: Remove this once the two callers above have been removed.
    fn tile(&self, aux_format: wgpu::TextureFormat, source: AllocSource) -> TilePairHandle {
        let color = self.acquire_tex(wgpu::TextureFormat::Rgba16Float, source);
        let aux = self.acquire_tex(aux_format, source);
        TilePairHandle(Arc::new(TilePair { color, aux }))
    }

    /// Acquire one pooled texture of `format`, reusing a recycled one when available.
    fn acquire_tex(&self, format: wgpu::TextureFormat, source: AllocSource) -> TexHandle {
        let pool = self.format_pools.get(&format).expect("unsupported format");
        let tex = {
            let mut pool = pool.lock().expect("tile pool poisoned");
            *pool.sources.entry(source).or_default() += 1;
            if let Some(tex) = pool.free.pop() {
                tex
            } else {
                pool.increase_capacity(format);
                self.create_texture(format)
            }
        };
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        TexHandle(Arc::new(GpuTex {
            tex: Some(tex),
            view,
            pool: Arc::downgrade(&pool),
            source,
        }))
    }

    /// Number of recycled color-format textures available (for tests).
    pub fn free_count(&self) -> usize {
        let format = wgpu::TextureFormat::Rgba16Float;
        self.format_pools.get(&format).expect("unsupported format")
            .lock()
            .expect("tile pool poisoned")
            .free.len()
    }

    fn create_texture(&self, format: wgpu::TextureFormat) -> wgpu::Texture {
        self.ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("stark tile channel"),
            // Interior + apron on every side (DESIGN.md §6.4).
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
        })
    }
}
