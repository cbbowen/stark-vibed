//! Recording GPU work, and the two rules about *when* what it names may be
//! released (§6.2, §5.2).
//!
//! Nothing in a recorded encoder has run. Both rules follow from that:
//!
//! * **A pooled resource handed back before its commands are submitted is not free — it
//!   is the next consumer's.** The pool gives it straight out again, to a pass in the
//!   very same encoder, which overwrites it before the earlier pass ever reads it. The
//!   corruption is one tile's paint smeared into another's, on large operations only,
//!   and no test names it; since `TilePool`'s trim can `destroy()` a free texture, the
//!   same mistake reaches a dangling view rather than merely wrong pixels.
//! * **An unpooled resource merely dropped is not freed either** — on the web that
//!   releases the JS handle and waits for GC, which cannot keep up with a rate.
//!   [`ScopedResources`] destroys instead, once the submit that reads them lands.
//!
//! **Both rules live in one type**, [`SubmitScope`](crate::gpu::scratch::SubmitScope);
//! this module is the destructor that answers the second.

/// GPU resources scoped to one recording: sized per call, so — unlike the
/// fixed-`TILE_TEX` tile pool — they cannot be recycled, and left to drop they
/// would only release the JS handle and wait on GC, which cannot keep up → the tab
/// OOMs. So they are collected here (cheap `Arc` clones) and **`destroy()`d on
/// drop**, which the scopes arrange to happen right after their submit — safe,
/// because WebGPU defers the real free until the in-flight work referencing them
/// completes.
///
/// **Buffers only.** Every scratch *texture* has a shape some later recording wants
/// again, so those lease from a pool instead (`gpu::scratch`, `gpu::tile`). Buffer sizes
/// follow a stroke's segment count and a piece's tile count, and so genuinely differ
/// call to call.
#[derive(Default)]
pub(crate) struct ScopedResources {
    buffers: Vec<wgpu::Buffer>,
}

impl ScopedResources {
    /// Register a buffer; returns it unchanged (the clone keeps the GPU resource
    /// alive until this `ScopedResources` drops).
    pub(crate) fn buffer(&mut self, buf: wgpu::Buffer) -> wgpu::Buffer {
        self.buffers.push(buf.clone());
        buf
    }
}

impl Drop for ScopedResources {
    fn drop(&mut self) {
        if !self.buffers.is_empty() {
            tracing::trace!(buffers = self.buffers.len(), "destroying scoped resources");
        }
        for buf in self.buffers.drain(..) {
            buf.destroy();
        }
    }
}
