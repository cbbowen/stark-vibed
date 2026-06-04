//! Tiles and the recycling tile pool (DESIGN.md §5.1, §5.2, §6.1).
//!
//! A [`GpuTile`] owns the GPU textures for one tile's channels. Tiles are held
//! through [`TileHandle`] (`Arc<GpuTile>`); cloning a handle is an `Arc` bump,
//! which is what makes persistent `DocState` snapshots cheap. When the last
//! handle to a tile drops, its textures return to the [`TilePool`] free list —
//! so history retention drives GPU memory reclamation with no manual GC.
//!
//! Channels (DESIGN.md §6.1, normalized representation):
//! - `color`: `Rgba16Float`, latent colour premultiplied by **opacity**
//!   (`L·op, a·op, b·op, op`) — opacity, *not* coverage.
//! - `aux`: `Rg16Float`, `(thickness, wet)` — impasto thickness and wetness. The
//!   media pass combines opacity × thickness into the visible alpha.

use std::sync::{Arc, Mutex, Weak};

use crate::geom::TILE_TEX;
use crate::gpu::context::GpuContext;

const CHANNEL_USAGE: wgpu::TextureUsages = wgpu::TextureUsages::TEXTURE_BINDING
    .union(wgpu::TextureUsages::RENDER_ATTACHMENT)
    .union(wgpu::TextureUsages::COPY_SRC)
    .union(wgpu::TextureUsages::COPY_DST);

/// One tile's GPU-resident channels.
pub struct GpuTile {
    // `Option` only so [`Drop`] can move the textures back to the pool.
    color: Option<wgpu::Texture>,
    aux: Option<wgpu::Texture>,
    color_view: wgpu::TextureView,
    aux_view: wgpu::TextureView,
    pool: Weak<Mutex<PoolInner>>,
}

impl GpuTile {
    pub fn color(&self) -> &wgpu::Texture {
        self.color.as_ref().expect("color present until drop")
    }
    pub fn aux(&self) -> &wgpu::Texture {
        self.aux.as_ref().expect("aux present until drop")
    }
    pub fn color_view(&self) -> &wgpu::TextureView {
        &self.color_view
    }
    pub fn aux_view(&self) -> &wgpu::TextureView {
        &self.aux_view
    }
}

impl Drop for GpuTile {
    fn drop(&mut self) {
        if let Some(pool) = self.pool.upgrade()
            && let Ok(mut inner) = pool.lock()
        {
            if let Some(t) = self.color.take() {
                inner.free_color.push(t);
            }
            if let Some(t) = self.aux.take() {
                inner.free_aux.push(t);
            }
        }
    }
}

/// A handle to a tile. Cloning is an `Arc` bump (DESIGN.md §5.1).
#[derive(Clone)]
pub struct TileHandle(Arc<GpuTile>);

impl TileHandle {
    pub fn color(&self) -> &wgpu::Texture {
        self.0.color()
    }
    pub fn aux(&self) -> &wgpu::Texture {
        self.0.aux()
    }
    pub fn color_view(&self) -> &wgpu::TextureView {
        self.0.color_view()
    }
    pub fn aux_view(&self) -> &wgpu::TextureView {
        self.0.aux_view()
    }
}

struct PoolInner {
    free_color: Vec<wgpu::Texture>,
    free_aux: Vec<wgpu::Texture>,
}

/// Recycling allocator for tile textures (DESIGN.md §6.1).
#[derive(Clone)]
pub struct TilePool {
    ctx: GpuContext,
    /// Channel texture formats, chosen by the document's color space (§6.7).
    color_format: wgpu::TextureFormat,
    aux_format: wgpu::TextureFormat,
    inner: Arc<Mutex<PoolInner>>,
}

impl TilePool {
    pub fn new(
        ctx: GpuContext,
        color_format: wgpu::TextureFormat,
        aux_format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            ctx,
            color_format,
            aux_format,
            inner: Arc::new(Mutex::new(PoolInner {
                free_color: Vec::new(),
                free_aux: Vec::new(),
            })),
        }
    }

    /// Acquire a tile, reusing recycled textures when available. Contents are
    /// undefined until painted or cleared.
    pub fn acquire(&self) -> TileHandle {
        let (color, aux) = {
            let mut inner = self.inner.lock().expect("tile pool poisoned");
            (inner.free_color.pop(), inner.free_aux.pop())
        };
        let color = color.unwrap_or_else(|| self.create_texture(self.color_format));
        let aux = aux.unwrap_or_else(|| self.create_texture(self.aux_format));

        let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
        let aux_view = aux.create_view(&wgpu::TextureViewDescriptor::default());
        TileHandle(Arc::new(GpuTile {
            color: Some(color),
            aux: Some(aux),
            color_view,
            aux_view,
            pool: Arc::downgrade(&self.inner),
        }))
    }

    /// Number of recycled color textures available (for tests).
    pub fn free_count(&self) -> usize {
        self.inner.lock().expect("tile pool poisoned").free_color.len()
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
