//! Tiles and the recycling tile pool (DESIGN.md §5.1, §5.2, §6.1).
//!
//! A [`GpuTile`] owns the GPU textures for one tile's channels. Tiles are held
//! through [`TileHandle`] (`Arc<GpuTile>`); cloning a handle is an `Arc` bump,
//! which is what makes persistent `DocState` snapshots cheap. When the last
//! handle to a tile drops, its textures return to the [`TilePool`] free list —
//! so history retention drives GPU memory reclamation with no manual GC.
//!
//! Step 1 implements only the color channel; height/wet and the data-driven
//! `ChannelSet` (DESIGN.md §6.1) are additive and slot in here later.

use std::sync::{Arc, Mutex, Weak};

use crate::geom::TILE_SIZE;
use crate::gpu::context::GpuContext;

/// Texture format of the color channel. Linear (not sRGB) so blending is done
/// in a linear/perceptual space; in step 4 this carries Oklab (DESIGN.md §6.5).
pub const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Usages every color tile needs: sampled when presenting, cleared/painted as a
/// render target, and copyable for readback in tests (DESIGN.md §9).
const COLOR_USAGE: wgpu::TextureUsages = wgpu::TextureUsages::TEXTURE_BINDING
    .union(wgpu::TextureUsages::RENDER_ATTACHMENT)
    .union(wgpu::TextureUsages::COPY_SRC);

/// One tile's GPU-resident channels.
pub struct GpuTile {
    /// Color channel; `Option` only so [`Drop`] can move it back to the pool.
    color: Option<wgpu::Texture>,
    color_view: wgpu::TextureView,
    /// Pool to recycle into; `Weak` so a dropped pool doesn't keep tiles alive.
    pool: Weak<Mutex<PoolInner>>,
}

impl GpuTile {
    /// View of the color channel, for sampling or as a render attachment.
    pub fn color_view(&self) -> &wgpu::TextureView {
        &self.color_view
    }

    /// The color channel texture.
    pub fn color(&self) -> &wgpu::Texture {
        self.color.as_ref().expect("color present until drop")
    }
}

impl Drop for GpuTile {
    fn drop(&mut self) {
        if let (Some(texture), Some(pool)) = (self.color.take(), self.pool.upgrade()) {
            if let Ok(mut inner) = pool.lock() {
                inner.free_color.push(texture);
            }
        }
    }
}

/// A handle to a tile. Cloning is an `Arc` bump (DESIGN.md §5.1).
#[derive(Clone)]
pub struct TileHandle(Arc<GpuTile>);

impl TileHandle {
    pub fn color_view(&self) -> &wgpu::TextureView {
        self.0.color_view()
    }

    pub fn color(&self) -> &wgpu::Texture {
        self.0.color()
    }
}

struct PoolInner {
    free_color: Vec<wgpu::Texture>,
}

/// Recycling allocator for tile textures (DESIGN.md §6.1).
#[derive(Clone)]
pub struct TilePool {
    ctx: GpuContext,
    inner: Arc<Mutex<PoolInner>>,
}

impl TilePool {
    pub fn new(ctx: GpuContext) -> Self {
        Self {
            ctx,
            inner: Arc::new(Mutex::new(PoolInner {
                free_color: Vec::new(),
            })),
        }
    }

    /// Acquire a tile, reusing a recycled texture when available. The contents
    /// are undefined until painted or cleared (see [`TilePool::acquire_filled`]).
    pub fn acquire(&self) -> TileHandle {
        let color = self
            .inner
            .lock()
            .expect("tile pool poisoned")
            .free_color
            .pop()
            .unwrap_or_else(|| self.create_color_texture());

        let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
        TileHandle(Arc::new(GpuTile {
            color: Some(color),
            color_view,
            pool: Arc::downgrade(&self.inner),
        }))
    }

    /// Acquire a tile cleared to a solid linear-RGBA color. Used by the step-1
    /// skeleton to prove the present path (DESIGN.md §12 build order, step 1).
    pub fn acquire_filled(&self, color: wgpu::Color) -> TileHandle {
        let tile = self.acquire();
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("stark fill tile"),
            });
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("stark fill tile pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: tile.color_view(),
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(color),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        self.ctx.queue.submit([encoder.finish()]);
        tile
    }

    /// Number of recycled textures currently available (for tests).
    pub fn free_count(&self) -> usize {
        self.inner.lock().expect("tile pool poisoned").free_color.len()
    }

    fn create_color_texture(&self) -> wgpu::Texture {
        self.ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("stark tile color"),
            size: wgpu::Extent3d {
                width: TILE_SIZE,
                height: TILE_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COLOR_FORMAT,
            usage: COLOR_USAGE,
            view_formats: &[],
        })
    }
}
