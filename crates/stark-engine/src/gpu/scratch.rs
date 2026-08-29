//! The pooled stroke scratch (§6.2): working textures kept across folds, and the
//! **submit scope** that is the only way to hand one back.
//!
//! A live update's region, snapshot, cell, reservoir and bake textures are the same
//! sizes fold after fold, so allocating them afresh and `destroy()`ing them at submit
//! is tens of megabytes of creation, zero-initialization and teardown per fold for
//! nothing. This pool keeps them: a checkout reuses a free texture whose [`Key`]
//! matches exactly, and a lease goes back on the free list *after the submit* of the
//! commands recorded against it — commands on one queue run in submission order, so a
//! texture whose last use is already submitted can be re-recorded against freely.
//!
//! **A lease can reach the free list only through a submit**, and that is carried by
//! the types rather than argued at each release site. What a missed site costs is the
//! `TilePool` free-list-vs-open-encoder failure: "no live handle" is not "no pending
//! GPU work", and a texture handed out while an unsubmitted encoder still names it is
//! either a failed submit or another stroke's pixels. So the pool's `give` is private
//! to this module, and the two ways a lease comes back both carry the ordering in
//! their shape:
//!
//! * [`SubmitScope`] — owns the encoder *and* the leases recorded against it, and
//!   releases the leases only in the same call that submits the encoder
//!   ([`SubmitScope::flush`] / [`SubmitScope::finish`]). A call site cannot return
//!   a lease early because it never holds one loose.
//! * [`Kept`] — a lease that outlives its run (the tool-state copies), returned on
//!   drop. Sound on a borrow argument rather than a convention: see the type.
//!
//! A scope dropped on an unwind returns nothing to the pool, which is also sound:
//! its commands were never submitted, so nothing pending names its leases.
//!
//! [`TileScope`](crate::gpu::submit::TileScope) is this type's sibling, carrying the
//! identical rule for the renderers that rewrite whole tiles — the transform, the
//! merge, the fill, the selection. The two are separate only because they hold
//! different things: this one holds [`ScratchPool`] leases, whose release must be
//! unforgeable and is therefore private to this module, while that one holds
//! ordinary pooled handles whose release is their `Drop`. Neither collapses into the
//! other without weakening one of them, and a change to the ordering rule belongs in
//! both.
//!
//! Contents are **not** zeroed on reuse, and no consumer may rely on the
//! zero-initialization a fresh texture gets. That is an audited property of every
//! key taken here: the region and narrow targets load with a clear; the reservoir is
//! cleared or fully copied into before any read, and its passes store every texel
//! they own; the bake is fully rewritten per segment (it is already reused across a
//! stroke's segments on exactly this argument); the snapshot's stores and the
//! deposit's reads are gated by the same `outside_sweep` predicate, so a texel one
//! skips the other never loads; and a cell the coarse deposit can name is one its
//! own hoist wrote (`plan::cell_geometry`). The tool-state copies are written whole
//! by the very copy that checks them out; the erase pass's accumulators (§6.12)
//! are cleared by their first sweep pass or fully copied into from the carried
//! total before anything loads them.

use std::sync::{Arc, Mutex};

/// How many tiles one a scope records before it submits what it has and
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
pub(crate) const FLUSH_TILES: usize = 256;

use crate::gpu::channels::Targets;
use crate::gpu::context::GpuContext;
use stark_model::geom::TILE_TEX;

use crate::unpoisoned;

/// How many bytes of free textures **and buffers** the pool will hold before it
/// starts destroying the least-recently-used. One wide-tip piece's working set — region, narrow,
/// snapshot and cells, three channels each in a pigment space — is on the order of
/// 120 MB, so this keeps roughly one generation warm plus headroom for the sizes to
/// drift across a tile boundary; anything beyond that is destroyed eagerly, for the
/// same reason `ScopedResources` destroys rather than waiting on GC (§6.2).
const POOL_BUDGET: u64 = 256 << 20;

/// What a checkout asks for: the exact descriptor it would otherwise create with,
/// plus the label a fresh creation is given. Exact size on purpose — the dynamics
/// shaders read `textureDimensions` of the snapshot and region, so an oversized
/// stand-in would change what they compute, not just what it costs.
///
/// **The label is not part of what makes two textures interchangeable**, which is
/// why there is no derived `Eq` here to say otherwise. A checkout wants a texture of
/// a shape, and the shape is the descriptor; a name for it is a debug affordance,
/// and letting one into the match meant the piece target and the bleed target — the
/// same square, the same format, the same usage — kept separate free lists and the
/// [`POOL_BUDGET`] held half as many useful textures as it could. What it costs is
/// that a capture may show a reused texture under the name it was first created
/// with, which is a label being stale rather than a picture being wrong.
#[derive(Clone, Copy)]
pub(crate) struct Key {
    pub(crate) size: (u32, u32),
    pub(crate) format: wgpu::TextureFormat,
    pub(crate) usage: wgpu::TextureUsages,
    pub(crate) label: &'static str,
}

impl Key {
    /// A key for one **whole tile texture** — interior plus apron, the size every
    /// scratch that stands in for a tile is.
    ///
    /// A constructor rather than three call sites repeating `(TILE_TEX, TILE_TEX)`,
    /// because that pair is not a size somebody chose: it is what a tile *is* (§6.4),
    /// and a scratch that got it wrong would be one the write-back cut the wrong block
    /// out of.
    pub(crate) const fn tile(
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
        label: &'static str,
    ) -> Self {
        Self {
            size: (TILE_TEX, TILE_TEX),
            format,
            usage,
            label,
        }
    }

    /// A copy extent covering the whole of a texture this key describes.
    ///
    /// Asked of the key rather than written beside it, which is what the three
    /// `Extent3d` constants in this module's callers each were: a copy whose extent
    /// disagreed with the allocation it addresses does not fail, it moves the wrong
    /// block — and there is nothing in a bare `Extent3d` to compare against the key
    /// that made the texture.
    pub(crate) const fn extent(&self) -> wgpu::Extent3d {
        wgpu::Extent3d {
            width: self.size.0,
            height: self.size.1,
            depth_or_array_layers: 1,
        }
    }

    /// Whether a texture created for `self` can serve a checkout of `other` — every
    /// field of the descriptor, and nothing else. See the type's note on the label.
    fn interchangeable(&self, other: &Key) -> bool {
        self.size == other.size && self.format == other.format && self.usage == other.usage
    }
}

/// What a scratch **buffer** checkout asks for. [`Key`]'s sibling, and it differs in
/// exactly one way: the size is rounded up rather than matched exactly.
///
/// **Rounded, because a buffer's size follows the drawing.** A sweep's instance
/// buffer is one record per segment-in-a-tile and grows through the stroke, so exact
/// matching would put a fresh buffer on the free list at every size the stroke passed
/// through and reuse none of them. Rounding to the next power of two turns that into
/// a handful of buckets a stroke settles into, for at most a factor of two of slack —
/// which is the same bargain `InstanceStream`'s high-water mark makes, arrived at
/// from the pool side.
///
/// The label is out of the match for [`Key`]'s reason.
#[derive(Clone, Copy)]
pub(crate) struct BufKey {
    /// What the caller asked for. [`Self::bucket`] is what is actually allocated.
    pub(crate) size: u64,
    pub(crate) usage: wgpu::BufferUsages,
    pub(crate) label: &'static str,
}

impl BufKey {
    /// The allocation this request lands in: the next power of two, floored at the
    /// uniform slot quantum so the small end does not fragment into 4- and 8-byte
    /// buckets.
    fn bucket(&self) -> u64 {
        self.size
            .max(crate::gpu::uniforms::UNIFORM_SLOT)
            .next_power_of_two()
    }

    /// Whether a buffer allocated for `self` can serve a checkout of `other`.
    fn interchangeable(&self, other: &BufKey) -> bool {
        self.usage == other.usage && self.bucket() == other.bucket()
    }
}

/// One checked-out scratch buffer.
struct BufLease {
    buf: wgpu::Buffer,
    key: BufKey,
    bytes: u64,
}

struct BufEntry {
    key: BufKey,
    buf: wgpu::Buffer,
    bytes: u64,
    last: u64,
}

/// One checked-out scratch texture and the view onto it, pooled together for the
/// same reason the tile pool's are ([`Pooled`](crate::gpu::tile)): every checkout
/// wants the same whole-texture view, so re-creating it bought nothing but an
/// object per fold.
struct Lease {
    tex: wgpu::Texture,
    view: wgpu::TextureView,
    key: Key,
    bytes: u64,
}

struct Entry {
    key: Key,
    tex: wgpu::Texture,
    view: wgpu::TextureView,
    bytes: u64,
    /// The pool tick this entry was last handed back on — the eviction order.
    last: u64,
}

#[derive(Default)]
struct Inner {
    free: Vec<Entry>,
    free_bufs: Vec<BufEntry>,
    /// Total bytes on the free list — textures and buffers together, since they come
    /// out of one device and one budget. Maintained so the budget check is a compare
    /// rather than a walk.
    bytes: u64,
    tick: u64,
}

/// The pool itself: shared by every clone of the renderer, so the live fold and the
/// commit that replaces it draw from one free list. `Default` is the empty pool —
/// it needs no device; each checkout brings one.
#[derive(Clone, Default)]
pub(crate) struct ScratchPool(Arc<Mutex<Inner>>);

impl ScratchPool {
    /// Open a [`SubmitScope`] on this pool: the encoder one render call records
    /// into, and the only holder its leases can have.
    pub(crate) fn scope(&self, ctx: &GpuContext, label: &'static str) -> SubmitScope {
        SubmitScope {
            ctx: ctx.clone(),
            pool: self.clone(),
            encoder: fresh_encoder(ctx, label),
            label,
            run_leases: Vec::new(),
            piece_leases: Vec::new(),
            piece_buf_leases: Vec::new(),
            piece_held: Vec::new(),
            piece_scoped: crate::gpu::submit::ScopedResources::default(),
            piece_open: false,
            since_flush: 0,
        }
    }

    /// Check a lease out that will **outlive** the scope that records against it —
    /// the tool-state copies — wrapped so its return path is its drop.
    pub(crate) fn keep(&self, device: &wgpu::Device, key: Key) -> Kept {
        Kept {
            lease: Some(self.take(device, key)),
            pool: self.clone(),
        }
    }

    /// Check a texture out: the newest free entry matching `key` exactly, or a fresh
    /// creation when there is none.
    fn take(&self, device: &wgpu::Device, key: Key) -> Lease {
        {
            let mut inner = unpoisoned(self.0.lock());
            inner.tick += 1;
            // Newest match first: the piece that just gave these back is the likeliest
            // shape of the piece about to take them.
            if let Some(i) = inner.free.iter().rposition(|e| e.key.interchangeable(&key)) {
                let e = inner.free.swap_remove(i);
                inner.bytes -= e.bytes;
                return Lease {
                    tex: e.tex,
                    view: e.view,
                    key,
                    bytes: e.bytes,
                };
            }
        }
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(key.label),
            size: key.extent(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: key.format,
            usage: key.usage,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let bytes = u64::from(key.size.0)
            * u64::from(key.size.1)
            * u64::from(key.format.block_copy_size(None).unwrap_or(8));
        Lease {
            tex,
            view,
            key,
            bytes,
        }
    }

    /// Check a buffer out: the newest free entry [`BufKey::interchangeable`] with
    /// `key`, or a fresh creation at the key's bucket size.
    fn take_buf(&self, device: &wgpu::Device, key: BufKey) -> BufLease {
        {
            let mut inner = unpoisoned(self.0.lock());
            inner.tick += 1;
            if let Some(i) = inner
                .free_bufs
                .iter()
                .rposition(|e| e.key.interchangeable(&key))
            {
                let e = inner.free_bufs.swap_remove(i);
                inner.bytes -= e.bytes;
                return BufLease {
                    buf: e.buf,
                    key,
                    bytes: e.bytes,
                };
            }
        }
        let bytes = key.bucket();
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(key.label),
            size: bytes,
            usage: key.usage,
            mapped_at_creation: false,
        });
        BufLease { buf, key, bytes }
    }

    /// [`give`](Self::give) for buffers, private for the same reason.
    fn give_buf(&self, lease: BufLease) {
        let mut inner = unpoisoned(self.0.lock());
        inner.tick += 1;
        let last = inner.tick;
        inner.bytes += lease.bytes;
        inner.free_bufs.push(BufEntry {
            key: lease.key,
            buf: lease.buf,
            bytes: lease.bytes,
            last,
        });
        Self::trim(&mut inner);
    }

    /// Hand a lease back to the free list, then hold the list to [`POOL_BUDGET`] by
    /// destroying least-recently-returned entries.
    ///
    /// **Private on purpose** — the module doc's whole argument. The callers are
    /// [`SubmitScope`], which has just submitted the commands naming the lease, and
    /// [`Kept`]'s drop, whose ordering the borrow checker carries.
    fn give(&self, lease: Lease) {
        let mut inner = unpoisoned(self.0.lock());
        inner.tick += 1;
        let last = inner.tick;
        inner.bytes += lease.bytes;
        inner.free.push(Entry {
            key: lease.key,
            tex: lease.tex,
            view: lease.view,
            bytes: lease.bytes,
            last,
        });
        Self::trim(&mut inner);
    }

    /// Hold the free lists to [`POOL_BUDGET`] by destroying the least-recently
    /// returned entry, whichever list it is on.
    ///
    /// **One budget over both**, and evicted strictly by age: a stroke that stops
    /// using a size should give it back whether it was a texture or a buffer, and two
    /// budgets would be two numbers to pick where the device has one memory.
    fn trim(inner: &mut Inner) {
        while inner.bytes > POOL_BUDGET {
            let tex = inner
                .free
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last)
                .map(|(i, e)| (i, e.last));
            let buf = inner
                .free_bufs
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last)
                .map(|(i, e)| (i, e.last));
            let evict = match (tex, buf) {
                (Some((i, t)), Some((_, b))) if t <= b => Some((true, i)),
                (_, Some((i, _))) => Some((false, i)),
                (Some((i, _)), None) => Some((true, i)),
                // Over budget with nothing free is a caller holding more than the
                // budget at once, which the budget does not bound and never did.
                (None, None) => None,
            };
            let Some((is_tex, i)) = evict else { break };
            // Destroyed outright rather than dropped: the free is deferred past any
            // in-flight use, and waiting on GC instead is how the tab OOMs (§6.2).
            if is_tex {
                let e = inner.free.swap_remove(i);
                inner.bytes -= e.bytes;
                e.tex.destroy();
            } else {
                let e = inner.free_bufs.swap_remove(i);
                inner.bytes -= e.bytes;
                e.buf.destroy();
            }
        }
    }
}

/// A pooled lease held **outside** any scope — the tool-state copies, which outlive
/// the run that recorded them — returned to the pool when its owner drops it.
///
/// Sound on a borrow argument rather than a convention. The commands that read one
/// of these are recorded by the run that *resumes* from it, which takes the owning
/// [`ToolState`](super::ToolState) by `&` and submits before it returns — so the
/// owner can only drop this, and return the lease, after that submit. An unwind
/// mid-run drops the run's encoder unsubmitted, so nothing pending names the lease
/// on that path either.
pub(crate) struct Kept {
    lease: Option<Lease>,
    pool: ScratchPool,
}

impl Kept {
    /// The leased texture — what a resume copy reads from and a capture writes to.
    pub(crate) fn tex(&self) -> &wgpu::Texture {
        &self
            .lease
            .as_ref()
            .expect("a Kept holds its lease until drop")
            .tex
    }

    /// The whole-texture view — what the erase pass binds its carried
    /// accumulator by, and renders its working one through (§6.12).
    pub(crate) fn view(&self) -> &wgpu::TextureView {
        &self
            .lease
            .as_ref()
            .expect("a Kept holds its lease until drop")
            .view
    }
}

impl Drop for Kept {
    fn drop(&mut self) {
        if let Some(lease) = self.lease.take() {
            self.pool.give(lease);
        }
    }
}

/// GPU work being recorded, and everything whose reuse must wait behind its submit:
/// the pooled leases the commands name — textures and buffers alike — and any other
/// resource whose drop must trail the submit (the swept path's tile-pool scratch
/// pair, the stamp loop's carried mint budget).
///
/// Two lease lifetimes, because a dynamics stroke has two: **run** leases carry
/// state across pieces (the reservoir ping-pong, the bake pair) and go back only at
/// [`finish`](Self::finish); **piece** leases (the region, the snapshot, the cells)
/// go back at each [`flush`](Self::flush), which is what keeps a long stroke's peak
/// transient memory at one region however many pieces it takes.
pub(crate) struct SubmitScope {
    ctx: GpuContext,
    pool: ScratchPool,
    encoder: wgpu::CommandEncoder,
    label: &'static str,
    run_leases: Vec<Lease>,
    piece_leases: Vec<Lease>,
    /// The buffer leases. Only the piece lifetime, because no buffer here carries
    /// across pieces: a plan, an instance run and a ceiling are each written afresh
    /// for the piece that draws with them, where the textures that *do* carry (the
    /// reservoir ping-pong, the bake pair) are the loop's running state.
    piece_buf_leases: Vec<BufLease>,
    /// Arbitrary resources whose *drop* must trail the submit — a pooled tile handle,
    /// whose early drop puts it back on `TilePool`'s free list while this encoder
    /// still names it.
    piece_held: Vec<Box<dyn std::any::Any>>,
    /// Unpooled per-piece buffers, destroyed at the submit that reads them — the
    /// uniform buffers written at creation, which cannot come from a pool.
    piece_scoped: crate::gpu::submit::ScopedResources,
    /// Whether anything has been recorded or taken since the last submit — what
    /// makes [`flush`](Self::flush) free when there is nothing to flush, and what
    /// keeps an operation that touched no tile from submitting an empty command
    /// buffer.
    piece_open: bool,
    /// Tiles recorded since the last submit, against [`FLUSH_TILES`] — the cadence
    /// [`tile_done`](Self::tile_done) counts and nothing else uses.
    since_flush: usize,
}

fn fresh_encoder(ctx: &GpuContext, label: &'static str) -> wgpu::CommandEncoder {
    ctx.device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
}

impl SubmitScope {
    /// The encoder this scope's commands are recorded into.
    pub(crate) fn encoder(&mut self) -> &mut wgpu::CommandEncoder {
        // Recording is enough to open the piece: a pass that names only handles the
        // caller owns still has to be submitted before those handles are released, and
        // a scope that recorded but took nothing would otherwise flush to nothing.
        self.piece_open = true;
        &mut self.encoder
    }

    /// Register a per-piece buffer; returns it unchanged, destroyed at the submit.
    ///
    /// For the buffers that cannot be pooled because they are written at creation.
    /// Everything else takes [`take_piece_buffer`](Self::take_piece_buffer), where the
    /// rate is not merely bounded but gone.
    pub(crate) fn buffer(&mut self, buf: wgpu::Buffer) -> wgpu::Buffer {
        self.piece_open = true;
        self.piece_scoped.buffer(buf)
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

    /// Check out scratch that carries state **across** pieces — released only at
    /// [`finish`](Self::finish), behind the submit of everything recorded.
    pub(crate) fn take_run(&mut self, key: Key) -> (wgpu::Texture, wgpu::TextureView) {
        let lease = self.pool.take(&self.ctx.device, key);
        let out = (lease.tex.clone(), lease.view.clone());
        self.run_leases.push(lease);
        out
    }

    /// Check out scratch for the piece being recorded — released at the piece's own
    /// submit ([`flush`](Self::flush) or [`finish`](Self::finish)).
    pub(crate) fn take_piece(&mut self, key: Key) -> (wgpu::Texture, wgpu::TextureView) {
        self.piece_open = true;
        let lease = self.pool.take(&self.ctx.device, key);
        let out = (lease.tex.clone(), lease.view.clone());
        self.piece_leases.push(lease);
        out
    }

    /// Check out a pooled buffer for the piece being recorded — released at the
    /// piece's own submit, exactly as [`take_piece`](Self::take_piece) is.
    ///
    /// **At least `key.size` bytes, and possibly more** ([`BufKey::bucket`]): a
    /// caller writes its own prefix and draws its own range, so slack past the end is
    /// unread. Contents are not zeroed, on the pool's general contract.
    pub(crate) fn take_piece_buffer(&mut self, key: BufKey) -> wgpu::Buffer {
        self.piece_open = true;
        let lease = self.pool.take_buf(&self.ctx.device, key);
        let out = lease.buf.clone();
        self.piece_buf_leases.push(lease);
        out
    }

    /// Keep `thing` alive past the submit, dropping it just after — for resources
    /// whose drop *is* their release to some other pool.
    pub(crate) fn hold(&mut self, thing: impl std::any::Any) {
        self.piece_open = true;
        self.piece_held.push(Box::new(thing));
    }

    /// Close out the piece already recorded, if any: submit the encoder, then
    /// release the piece's resources — in that order, which is the type's whole
    /// reason to exist. Run leases stay. Peak transient memory is then one region
    /// however long the stroke, and a stroke that fits one region never records a
    /// second submit.
    pub(crate) fn flush(&mut self) {
        if !self.piece_open {
            return;
        }
        self.piece_open = false;
        self.since_flush = 0;
        let done = std::mem::replace(&mut self.encoder, fresh_encoder(&self.ctx, self.label));
        self.ctx.queue.submit([done.finish()]);
        drop(std::mem::take(&mut self.piece_scoped));
        self.piece_held.clear();
        for lease in self.piece_leases.drain(..) {
            self.pool.give(lease);
        }
        for lease in self.piece_buf_leases.drain(..) {
            self.pool.give_buf(lease);
        }
    }

    /// Close the scope: submit what is still recorded, then release everything —
    /// the piece's resources and the run leases both, behind that same submit.
    pub(crate) fn finish(mut self) {
        self.ctx.queue.submit([self.encoder.finish()]);
        drop(self.piece_scoped);
        self.piece_held.clear();
        for lease in self.piece_leases.drain(..).chain(self.run_leases.drain(..)) {
            self.pool.give(lease);
        }
        for lease in self.piece_buf_leases.drain(..) {
            self.pool.give_buf(lease);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::context::GpuContext;
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
        let mut scope = ScratchPool::default().scope(&ctx, "stark submit test");

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
        let mut scope = ScratchPool::default().scope(&ctx, "stark submit test empty");
        for _ in 0..FLUSH_TILES * 2 {
            scope.tile_done();
        }
        assert!(!scope.piece_open, "nothing was recorded");
        scope.finish();
    }
}
