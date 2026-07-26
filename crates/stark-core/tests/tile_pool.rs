//! Tile-pool test: the recycling allocator returns dropped tiles to its free
//! list, so history retention drives GPU memory reclamation (DESIGN.md §5.1,
//! §6.1). The render path itself is covered end-to-end by the golden tests.
//!
//! Needs a GPU adapter; skips (rather than fails) where none is available.

use stark_core::gpu::{GpuContext, TilePool, tile::AllocSource};

/// The pool's channel format, and the one [`TilePool::free_count`] reports on.
const COLOR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Acquire a context, or `None` if the machine has no usable adapter *and*
/// `STARK_ALLOW_NO_GPU=1` permits the skip. Panics otherwise — a skipped GPU test
/// still reports `ok`, so a missing adapter must not pass silently (see
/// `tests/common/mod.rs`, which does the same for the engine harness).
fn context_or_skip() -> Option<GpuContext> {
    match pollster::block_on(GpuContext::headless()) {
        Ok(ctx) => Some(ctx),
        Err(e) if std::env::var("STARK_ALLOW_NO_GPU").is_ok_and(|v| v == "1") => {
            eprintln!("skipping GPU test (STARK_ALLOW_NO_GPU=1): {e}");
            None
        }
        Err(e) => panic!("no usable GPU adapter: {e}\nset STARK_ALLOW_NO_GPU=1 to skip GPU tests"),
    }
}

#[test]
fn pool_recycles_dropped_tiles() {
    let Some(ctx) = context_or_skip() else { return };
    let pool = TilePool::new(ctx, [COLOR, wgpu::TextureFormat::Rg16Float]);

    assert_eq!(pool.free_count(), 0, "fresh pool has no recycled tiles");

    let a = pool.acquire_tex(COLOR, AllocSource::Unknown);
    let b = pool.acquire_tex(COLOR, AllocSource::Unknown);
    assert_eq!(pool.free_count(), 0, "live tiles are not in the free list");

    drop(a);
    assert_eq!(
        pool.free_count(),
        1,
        "dropping the last handle recycles the tile"
    );
    drop(b);
    assert_eq!(pool.free_count(), 2);

    // A subsequent acquire reuses a recycled texture rather than allocating.
    let _c = pool.acquire_tex(COLOR, AllocSource::Unknown);
    assert_eq!(pool.free_count(), 1, "acquire reuses a recycled tile");
}

/// Free lists are keyed by format: recycling a colour texture must not satisfy a
/// request for an aux one. This is what lets a scratch tile take a wider aux
/// (`SCRATCH_AUX_FORMAT`) from the same pool as a persistent tile (DESIGN.md §6.1).
#[test]
fn free_lists_do_not_cross_formats() {
    let Some(ctx) = context_or_skip() else { return };
    let aux = wgpu::TextureFormat::Rg16Float;
    let pool = TilePool::new(ctx, [COLOR, aux]);

    let c = pool.acquire_tex(COLOR, AllocSource::Unknown);
    drop(c);
    assert_eq!(pool.free_count(), 1, "the colour texture was recycled");

    // Taking an aux texture must allocate, not steal the recycled colour one.
    let _a = pool.acquire_tex(aux, AllocSource::Unknown);
    assert_eq!(
        pool.free_count(),
        1,
        "an aux acquire must not consume the colour free list"
    );
}
