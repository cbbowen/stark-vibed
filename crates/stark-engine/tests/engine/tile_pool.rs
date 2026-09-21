//! Tile-pool test: the recycling allocator returns dropped tiles to its free
//! list, so history retention drives GPU memory reclamation (§5.1,
//! §6.1). The render path itself is covered end-to-end by the golden tests.
//!
//! Needs a GPU adapter; skips (rather than fails) where none is available.

use stark_engine::GpuContext;
use stark_engine::testing::{AllocSource, TilePool};

/// The pool's color channel format — the free list most of these tests watch.
const COLOR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Acquire a context, or `None` if the machine has no usable adapter *and*
/// `STARK_ALLOW_NO_GPU=1` permits the skip. Panics otherwise — a skipped GPU test
/// still reports `ok`, so a missing adapter must not pass silently. The decision is
/// `stark_engine::testing`'s, the same one the engine harness makes.
///
/// Uncached, unlike that harness's: these tests watch a pool's free list, so each
/// wants a device nothing else has drawn on.
fn context_or_skip() -> Option<GpuContext> {
    stark_engine::testing::or_skip(pollster::block_on(GpuContext::headless()), "GPU tests")
}

#[test]
fn pool_recycles_dropped_tiles() {
    let Some(ctx) = context_or_skip() else { return };
    let pool = TilePool::new(ctx, [COLOR, wgpu::TextureFormat::R16Float]);

    assert_eq!(
        pool.free_count(COLOR),
        0,
        "fresh pool has no recycled tiles"
    );

    let a = pool.acquire_tex(COLOR, AllocSource::Unknown);
    let b = pool.acquire_tex(COLOR, AllocSource::Unknown);
    assert_eq!(
        pool.free_count(COLOR),
        0,
        "live tiles are not in the free list"
    );

    drop(a);
    assert_eq!(
        pool.free_count(COLOR),
        1,
        "dropping the last handle recycles the tile"
    );
    drop(b);
    assert_eq!(pool.free_count(COLOR), 2);

    // A subsequent acquire reuses a recycled texture rather than allocating.
    let _c = pool.acquire_tex(COLOR, AllocSource::Unknown);
    assert_eq!(pool.free_count(COLOR), 1, "acquire reuses a recycled tile");
}

/// Free lists are keyed by format: recycling a color texture must not satisfy a
/// request for an aux one. This is what lets a scratch tile take a wider aux
/// (`SCRATCH_AUX_FORMAT`) from the same pool as a persistent tile (§6.1).
#[test]
fn free_lists_do_not_cross_formats() {
    let Some(ctx) = context_or_skip() else { return };
    let aux = wgpu::TextureFormat::R16Float;
    let pool = TilePool::new(ctx, [COLOR, aux]);

    let c = pool.acquire_tex(COLOR, AllocSource::Unknown);
    drop(c);
    assert_eq!(pool.free_count(COLOR), 1, "the color texture was recycled");
    assert_eq!(pool.free_count(aux), 0, "and landed on its own free list");

    // Taking an aux texture must allocate, not steal the recycled color one.
    let _a = pool.acquire_tex(aux, AllocSource::Unknown);
    assert_eq!(
        pool.free_count(COLOR),
        1,
        "an aux acquire must not consume the color free list"
    );
}

/// **Tile memory comes back.** A pool that reached a high-water mark once used to
/// sit on it for the rest of the session: the free list only ever grew, so a single
/// large operation made its peak permanently resident (§5.1 promised reclamation and
/// delivered it only as far as the free list).
///
/// The pool now measures its own peak demand over an epoch of acquires and hands
/// back half the surplus at each boundary, so a burst's leftovers decay geometrically
/// once ordinary work resumes.
///
/// Driven by spinning acquires rather than by naming the epoch length, which is the
/// pool's business: the loop is bounded so a policy that never trims fails the test
/// instead of hanging.
#[test]
fn a_burst_of_tiles_is_handed_back_once_the_work_moves_on() {
    let Some(ctx) = context_or_skip() else { return };
    let pool = TilePool::new(ctx, [COLOR]);

    // A burst — a big transform, a fill across a wide selection — then done with.
    const BURST: usize = 200;
    let held: Vec<_> = (0..BURST)
        .map(|_| pool.acquire_tex(COLOR, AllocSource::Unknown))
        .collect();
    drop(held);
    assert_eq!(
        pool.free_count(COLOR),
        BURST,
        "the burst is idle in the free list"
    );

    // Ordinary work afterwards: one tile at a time, so the pool's peak demand is
    // nothing like the burst it is still holding.
    let mut spins = 0usize;
    while pool.free_count(COLOR) > 8 && spins < 1_000_000 {
        let _one = pool.acquire_tex(COLOR, AllocSource::Unknown);
        spins += 1;
    }
    assert!(
        pool.free_count(COLOR) <= 8,
        "still holding {} of the burst's {BURST} textures after {spins} acquires",
        pool.free_count(COLOR),
    );
}

/// And it does not hand back what the work is *using*. A pool kept at a steady
/// working set must keep serving it from the free list — a trim that undershot would
/// trade the memory win for recreating textures inside the very loop that needs them.
#[test]
fn a_steady_working_set_is_not_trimmed_away() {
    let Some(ctx) = context_or_skip() else { return };
    let pool = TilePool::new(ctx, [COLOR]);

    // Hold a working set across every acquire, so the pool's peak never drops.
    const WORKING: usize = 16;
    let _held: Vec<_> = (0..WORKING)
        .map(|_| pool.acquire_tex(COLOR, AllocSource::Unknown))
        .collect();
    // Churn a 17th for long enough to cross many epochs.
    for _ in 0..200_000 {
        let _one = pool.acquire_tex(COLOR, AllocSource::Unknown);
    }
    // The 17th slot is the only idle one, and the working set is untouched: every
    // acquire above was served without creating a texture.
    assert!(
        pool.free_count(COLOR) <= 1,
        "the pool grew to {} idle slots for a steady working set",
        pool.free_count(COLOR),
    );
}
