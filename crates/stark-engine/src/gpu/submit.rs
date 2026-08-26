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
//! [`TileScope`] carries both rules for the renderers that rewrite whole tiles
//! out of the [`TilePool`](crate::gpu::TilePool) — the transform, the merge, the
//! fill, the selection. It is the sibling of `gpu::stroke::scratch::SubmitScope`,
//! which carries the identical rules for the *stroke* path; the two are separate
//! only because they hold different things. The stroke's holds `ScratchPool`
//! leases, whose release must be unforgeable and is therefore private to that
//! module; this one holds ordinary pooled handles, whose release is their `Drop`.
//! Neither may be collapsed into the other without weakening one of them, and a
//! change to the ordering rule belongs in both.

use std::any::Any;

use crate::gpu::channels::Targets;
use crate::gpu::context::GpuContext;

/// GPU resources scoped to one recording: sized per call, so — unlike the
/// fixed-`TILE_TEX` tile pool — they cannot be recycled, and left to drop they
/// would only release the JS handle and wait on GC, which cannot keep up → the tab
/// OOMs. So they are collected here (cheap `Arc` clones) and **`destroy()`d on
/// drop**, which the scopes arrange to happen right after their submit — safe,
/// because WebGPU defers the real free until the in-flight work referencing them
/// completes.
#[derive(Default)]
pub(crate) struct ScopedResources {
    textures: Vec<wgpu::Texture>,
    buffers: Vec<wgpu::Buffer>,
}

impl ScopedResources {
    /// Register a texture; returns it unchanged (the clone keeps the GPU resource
    /// alive until this `ScopedResources` drops).
    pub(crate) fn texture(&mut self, tex: wgpu::Texture) -> wgpu::Texture {
        self.textures.push(tex.clone());
        tex
    }

    /// Register a buffer; returns it unchanged.
    pub(crate) fn buffer(&mut self, buf: wgpu::Buffer) -> wgpu::Buffer {
        self.buffers.push(buf.clone());
        buf
    }
}

impl Drop for ScopedResources {
    fn drop(&mut self) {
        if !self.textures.is_empty() || !self.buffers.is_empty() {
            tracing::trace!(
                textures = self.textures.len(),
                buffers = self.buffers.len(),
                "destroying scoped resources",
            );
        }
        for tex in self.textures.drain(..) {
            tex.destroy();
        }
        for buf in self.buffers.drain(..) {
            buf.destroy();
        }
    }
}

/// How many tiles one [`TileScope`] records before it submits what it has and
/// releases the scratch behind it.
///
/// **This is what stops peak GPU memory scaling with the operation.** A blended
/// merge takes three scratch trios per tile on top of the destination it keeps
/// (`merge::encode_blended`), and a merge has no cap — its tile count is the union
/// of two layers the document already holds, which on a full canvas is tens of
/// thousands. Recorded into one encoder that is submitted once, that is every one
/// of those trios live at the same moment: ~15 GB at 10k tiles, and ~40,000 render
/// passes in a single command buffer, which is a Windows TDR as much as it is an
/// allocation failure.
///
/// A cadence rather than a cap on the operation, because the operation is not the
/// problem — holding all of it at once is. Tiles are independent: destinations are
/// disjoint, each pass reads only its own tile's inputs and per-operation uniforms
/// that outlive the whole recording, and submits on one queue execute in order. So
/// cutting the recording anywhere is invisible in the result.
///
/// 256 bounds the blended merge's transient scratch at roughly 150 MB and its
/// command buffers at about a thousand render passes. Like
/// [`MAX_RELEASE_PER_EPOCH`](crate::gpu::tile) it is **a bound on a cost that has
/// not been measured**, not a tuned figure: raising it costs memory and lowers the
/// submit count, and the honest way to change it is to measure and say so.
const FLUSH_TILES: usize = 256;

/// GPU work being recorded, and everything whose release must trail its submit:
/// the pooled handles the commands name, and the unpooled buffers and textures
/// they read.
///
/// **A held resource can reach its pool only through a submit**, and the type is what
/// says so rather than a `drop` each call site has to place after the submit by hand.
/// [`finish`](Self::finish) takes `self` and [`flush`](Self::flush) submits before it
/// releases, so a call site cannot get the order wrong: it never holds anything loose.
/// A scope dropped on an unwind releases nothing to any pool, which is sound too — its
/// commands were never submitted, so nothing pending names them.
pub(crate) struct TileScope {
    ctx: GpuContext,
    label: &'static str,
    encoder: wgpu::CommandEncoder,
    /// Unpooled per-tile resources — the uniform buffers a pass binds — destroyed
    /// at the submit that reads them.
    scoped: ScopedResources,
    /// Resources whose *drop* is their release to some pool: the scratch tile
    /// handles a pass writes and a later pass in the same encoder reads.
    held: Vec<Box<dyn Any>>,
    /// Whether anything has been recorded since the last submit — what makes a
    /// flush free when there is nothing to flush, and what keeps an operation that
    /// touched no tile from submitting an empty command buffer.
    recorded: bool,
    /// Tiles recorded since the last submit, against [`FLUSH_TILES`].
    since_flush: usize,
}

impl TileScope {
    pub(crate) fn new(ctx: &GpuContext, label: &'static str) -> Self {
        Self {
            ctx: ctx.clone(),
            label,
            encoder: fresh_encoder(ctx, label),
            scoped: ScopedResources::default(),
            held: Vec::new(),
            recorded: false,
            since_flush: 0,
        }
    }

    /// The encoder this scope's commands are recorded into.
    pub(crate) fn encoder(&mut self) -> &mut wgpu::CommandEncoder {
        self.recorded = true;
        &mut self.encoder
    }

    /// Register a per-tile buffer; returns it unchanged, destroyed at the submit
    /// ([`ScopedResources`]).
    pub(crate) fn buffer(&mut self, buf: wgpu::Buffer) -> wgpu::Buffer {
        self.scoped.buffer(buf)
    }

    /// Keep `thing` alive past the submit, dropping it just after — for a handle
    /// whose drop *is* its release to a pool.
    pub(crate) fn hold(&mut self, thing: impl Any) {
        self.held.push(Box::new(thing));
    }

    /// Record one **fullscreen pass over a tile's channels**: the shape every
    /// renderer here writes a destination tile with.
    ///
    /// The merge's four passes, the fill's one and the transform's combine were each
    /// spelling out the same fifteen lines — a render pass over two-or-three
    /// attachments, a pipeline, a bind group, `draw(0..3, 0..1)` — and the attachment
    /// count was the residual's `Option` decided a fourth and fifth time (§6.7). With
    /// [`Targets`] carrying that, what is left is one call.
    ///
    /// `ops` because the callers do differ there, if only just: everything writes
    /// every texel and clears, but a pass that reads its own target would not, and a
    /// helper that hid the choice would be the wrong kind of shared.
    pub(crate) fn fullscreen_pass(
        &mut self,
        label: &str,
        pipeline: &wgpu::RenderPipeline,
        bg: &wgpu::BindGroup,
        offsets: &[u32],
        into: Targets<'_>,
        ops: wgpu::Operations<wgpu::Color>,
    ) {
        let attachments = into.attachments(ops);
        let mut pass = self
            .encoder()
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
                color_attachments: &attachments[..into.count()],
                ..Default::default()
            });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bg, offsets);
        pass.draw(0..3, 0..1);
    }

    /// Note that one tile has been recorded, submitting and releasing if that
    /// reaches [`FLUSH_TILES`].
    ///
    /// Called once per destination tile, at the point where everything that tile
    /// needs has been recorded — never in the middle of one, since the scratch a
    /// half-recorded tile is holding is exactly what a flush would hand away.
    pub(crate) fn tile_done(&mut self) {
        self.since_flush += 1;
        if self.since_flush >= FLUSH_TILES {
            self.flush();
        }
    }

    /// Submit what is recorded, then release everything behind it — in that order,
    /// which is the type's whole reason to exist.
    fn flush(&mut self) {
        if !self.recorded {
            self.since_flush = 0;
            return;
        }
        let done = std::mem::replace(&mut self.encoder, fresh_encoder(&self.ctx, self.label));
        self.ctx.queue.submit([done.finish()]);
        drop(std::mem::take(&mut self.scoped));
        self.held.clear();
        self.recorded = false;
        self.since_flush = 0;
    }

    /// Close the scope: submit what is still recorded, then release everything.
    ///
    /// Takes `self`, so the held resources cannot be released except by submitting
    /// first. An operation whose every tile passed through by handle recorded
    /// nothing and submits nothing, rather than an empty command buffer.
    pub(crate) fn finish(mut self) {
        self.flush();
    }
}

fn fresh_encoder(ctx: &GpuContext, label: &'static str) -> wgpu::CommandEncoder {
    ctx.device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::tile::{AllocSource, TilePool};

    const COLOR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

    /// A context, or `None` where the machine has no adapter *and*
    /// `STARK_ALLOW_NO_GPU=1` permits the skip — `tests/tile_pool.rs`'s guard, for
    /// the same reason: a skipped GPU test still reports `ok`.
    fn context_or_skip() -> Option<GpuContext> {
        match pollster::block_on(GpuContext::headless()) {
            Ok(ctx) => Some(ctx),
            Err(e) if std::env::var("STARK_ALLOW_NO_GPU").is_ok_and(|v| v == "1") => {
                eprintln!("skipping GPU test (STARK_ALLOW_NO_GPU=1): {e}");
                None
            }
            Err(e) => {
                panic!("no usable GPU adapter: {e}\nset STARK_ALLOW_NO_GPU=1 to skip GPU tests")
            }
        }
    }

    /// **The cadence is what bounds peak memory**, which is the whole reason
    /// [`FLUSH_TILES`] exists: without it a merge holds every tile's scratch at once,
    /// and a merge has no cap on its tile count.
    ///
    /// Asked of the pool rather than of the scope, because the pool is where the cost
    /// actually lands. A scope that never flushed would leave every texture it ever
    /// took on the free list at `finish` — one per tile — where a flushing one keeps
    /// reusing the same working set and ends with about a cadence's worth. The gap
    /// between `3 · FLUSH_TILES` and `FLUSH_TILES` is the finding.
    ///
    /// The tile count stays under `TRIM_INTERVAL` acquires so the pool's own trim
    /// cannot fire and make the numbers a matter of two policies rather than one.
    #[test]
    fn a_scope_hands_its_scratch_back_as_it_goes() {
        let Some(ctx) = context_or_skip() else { return };
        let pool = TilePool::new(ctx.clone(), [COLOR]);
        let mut scope = TileScope::new(&ctx, "stark submit test");

        const TILES: usize = FLUSH_TILES * 3;
        for _ in 0..TILES {
            let scratch = pool.acquire_tex(COLOR, AllocSource::MergeScratch);
            // Something has to be *recorded* against it, or the flush is a no-op by
            // design and this would be testing nothing.
            scope
                .encoder()
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("stark submit test pass"),
                    color_attachments: &[Some(crate::gpu::desc::attach(
                        scratch.view(),
                        crate::gpu::desc::CLEAR,
                    ))],
                    ..Default::default()
                });
            scope.hold(scratch);
            scope.tile_done();
        }
        scope.finish();

        let idle = pool.free_count(COLOR);
        assert!(
            idle <= FLUSH_TILES + 1,
            "the scope held {idle} textures across {TILES} tiles; \
             a cadence of {FLUSH_TILES} should have kept it to about that",
        );
        // And it really did serve them all — a scope that somehow released nothing
        // would also fail the bound above by holding every one.
        assert!(idle > 0, "nothing was ever handed back to the pool");
    }

    /// A scope that recorded nothing submits nothing — an operation whose every tile
    /// passed through by handle (the lopsided merge, §14.11) should not cost an empty
    /// command buffer per flush boundary, nor one at `finish`.
    #[test]
    fn an_empty_scope_submits_nothing() {
        let Some(ctx) = context_or_skip() else { return };
        let mut scope = TileScope::new(&ctx, "stark submit test empty");
        for _ in 0..FLUSH_TILES * 2 {
            scope.tile_done();
        }
        assert!(!scope.recorded, "nothing was recorded");
        scope.finish();
    }
}
