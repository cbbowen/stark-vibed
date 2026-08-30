//! Recording GPU work, and the two rules about *when* what it names may be
//! released (§6.2, §5.2).
//!
//! Nothing in a recorded encoder has run. That one fact is behind both types
//! here, and behind every bug they exist to rule out:
//!
//! * **A pooled resource handed back before its commands are submitted is not
//!   free — it is the next consumer's.** The pool gives it straight out again, to
//!   a pass in the very same encoder, which overwrites it before the earlier pass
//!   ever reads it. The corruption is one tile's paint smeared into another's, on
//!   large operations only, and no test names it. Worse now than when it was
//!   first written up: `TilePool`'s trim can `destroy()` a free texture, so the
//!   same mistake reaches a dangling view rather than merely wrong pixels.
//! * **An unpooled resource merely dropped is not freed either** — on the web that
//!   releases the JS handle and waits for GC, which cannot keep up with a rate.
//!   [`ScopedResources`] destroys instead, once the submit that reads them lands.
//!
//! **Both rules live in one type**, [`SubmitScope`](crate::gpu::scratch::SubmitScope),
//! and this module is what is left over: the destructor that answers the second of
//! them. There were two scopes for as long as `ScratchPool`'s release was private to
//! the stroke path and the tile writers had no pool to lease from — a split that cost
//! two copies of one flush cadence, two arguments for one ordering rule, and one of
//! them saying `encoder()` opened a piece where the other did not.

/// GPU resources scoped to one recording: sized per call, so — unlike the
/// fixed-`TILE_TEX` tile pool — they cannot be recycled, and left to drop they
/// would only release the JS handle and wait on GC, which cannot keep up → the tab
/// OOMs. So they are collected here (cheap `Arc` clones) and **`destroy()`d on
/// drop**, which the scopes arrange to happen right after their submit — safe,
/// because WebGPU defers the real free until the in-flight work referencing them
/// completes.
///
/// **Buffers only, now.** Every scratch *texture* a recording wants has a shape some
/// later recording wants again, so all of them lease from a pool instead
/// (`gpu::scratch`, `gpu::tile`) — which is strictly better than destroying one
/// promptly, since a reused texture is not allocated at all. What is left here is the
/// buffers, whose sizes follow a stroke's segment count and a piece's tile count and
/// so genuinely differ call to call (`ENGINE_CLEANUP.md`, item H).
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
