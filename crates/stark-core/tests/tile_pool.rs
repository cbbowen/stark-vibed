//! Tile-pool test: the recycling allocator returns dropped tiles to its free
//! list, so history retention drives GPU memory reclamation (DESIGN.md §5.1,
//! §6.1). The render path itself is covered end-to-end by the golden tests.
//!
//! Needs a GPU adapter; skips (rather than fails) where none is available.

use stark_core::gpu::{GpuContext, TilePool};

/// Acquire a context or skip the test if the machine has no usable adapter.
fn context_or_skip() -> Option<GpuContext> {
    match pollster::block_on(GpuContext::headless()) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("skipping GPU test: {e}");
            None
        }
    }
}

#[test]
fn pool_recycles_dropped_tiles() {
    let Some(ctx) = context_or_skip() else { return };
    let pool = TilePool::new(
        ctx,
        wgpu::TextureFormat::Rgba16Float,
        wgpu::TextureFormat::Rg16Float,
    );

    assert_eq!(pool.free_count(), 0, "fresh pool has no recycled tiles");

    let a = pool.acquire();
    let b = pool.acquire();
    assert_eq!(pool.free_count(), 0, "live tiles are not in the free list");

    drop(a);
    assert_eq!(pool.free_count(), 1, "dropping the last handle recycles the tile");
    drop(b);
    assert_eq!(pool.free_count(), 2);

    // A subsequent acquire reuses a recycled texture rather than allocating.
    let _c = pool.acquire();
    assert_eq!(pool.free_count(), 1, "acquire reuses a recycled tile");
}
