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
pub(super) struct Key {
    pub(super) size: (u32, u32),
    pub(super) format: wgpu::TextureFormat,
    pub(super) usage: wgpu::TextureUsages,
    pub(super) label: &'static str,
}

impl Key {
    /// A key for one **whole tile texture** — interior plus apron, the size every
    /// scratch that stands in for a tile is.
    ///
    /// A constructor rather than three call sites repeating `(TILE_TEX, TILE_TEX)`,
    /// because that pair is not a size somebody chose: it is what a tile *is* (§6.4),
    /// and a scratch that got it wrong would be one the write-back cut the wrong block
    /// out of.
    pub(super) const fn tile(
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
    pub(super) const fn extent(&self) -> wgpu::Extent3d {
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
pub(super) struct BufKey {
    /// What the caller asked for. [`Self::bucket`] is what is actually allocated.
    pub(super) size: u64,
    pub(super) usage: wgpu::BufferUsages,
    pub(super) label: &'static str,
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
pub(super) struct ScratchPool(Arc<Mutex<Inner>>);

impl ScratchPool {
    /// Open a [`SubmitScope`] on this pool: the encoder one render call records
    /// into, and the only holder its leases can have.
    pub(super) fn scope(&self, ctx: &GpuContext, label: &'static str) -> SubmitScope {
        SubmitScope {
            ctx: ctx.clone(),
            pool: self.clone(),
            encoder: fresh_encoder(ctx, label),
            label,
            run_leases: Vec::new(),
            piece_leases: Vec::new(),
            piece_buf_leases: Vec::new(),
            piece_held: Vec::new(),
            piece_open: false,
        }
    }

    /// Check a lease out that will **outlive** the scope that records against it —
    /// the tool-state copies — wrapped so its return path is its drop.
    pub(super) fn keep(&self, device: &wgpu::Device, key: Key) -> Kept {
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
            size: wgpu::Extent3d {
                width: key.size.0,
                height: key.size.1,
                depth_or_array_layers: 1,
            },
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
pub(super) struct Kept {
    lease: Option<Lease>,
    pool: ScratchPool,
}

impl Kept {
    /// The leased texture — what a resume copy reads from and a capture writes to.
    pub(super) fn tex(&self) -> &wgpu::Texture {
        &self
            .lease
            .as_ref()
            .expect("a Kept holds its lease until drop")
            .tex
    }

    /// The whole-texture view — what the erase pass binds its carried
    /// accumulator by, and renders its working one through (§6.12).
    pub(super) fn view(&self) -> &wgpu::TextureView {
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
pub(super) struct SubmitScope {
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
    /// Arbitrary resources whose *drop* must trail the submit — the swept path's
    /// pooled tile pair, whose early drop puts it back on the tile pool's free list
    /// while this encoder still names it.
    piece_held: Vec<Box<dyn std::any::Any>>,
    /// Whether anything piece-scoped has been taken since the last submit — what
    /// makes [`flush`](Self::flush) free when there is nothing to flush.
    piece_open: bool,
}

fn fresh_encoder(ctx: &GpuContext, label: &'static str) -> wgpu::CommandEncoder {
    ctx.device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
}

impl SubmitScope {
    /// The encoder this scope's commands are recorded into.
    pub(super) fn encoder(&mut self) -> &mut wgpu::CommandEncoder {
        &mut self.encoder
    }

    /// Check out scratch that carries state **across** pieces — released only at
    /// [`finish`](Self::finish), behind the submit of everything recorded.
    pub(super) fn take_run(&mut self, key: Key) -> (wgpu::Texture, wgpu::TextureView) {
        let lease = self.pool.take(&self.ctx.device, key);
        let out = (lease.tex.clone(), lease.view.clone());
        self.run_leases.push(lease);
        out
    }

    /// Check out scratch for the piece being recorded — released at the piece's own
    /// submit ([`flush`](Self::flush) or [`finish`](Self::finish)).
    pub(super) fn take_piece(&mut self, key: Key) -> (wgpu::Texture, wgpu::TextureView) {
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
    pub(super) fn take_piece_buffer(&mut self, key: BufKey) -> wgpu::Buffer {
        self.piece_open = true;
        let lease = self.pool.take_buf(&self.ctx.device, key);
        let out = lease.buf.clone();
        self.piece_buf_leases.push(lease);
        out
    }

    /// Keep `thing` alive past the submit, dropping it just after — for resources
    /// whose drop *is* their release to some other pool.
    pub(super) fn hold(&mut self, thing: impl std::any::Any) {
        self.piece_open = true;
        self.piece_held.push(Box::new(thing));
    }

    /// Close out the piece already recorded, if any: submit the encoder, then
    /// release the piece's resources — in that order, which is the type's whole
    /// reason to exist. Run leases stay. Peak transient memory is then one region
    /// however long the stroke, and a stroke that fits one region never records a
    /// second submit.
    pub(super) fn flush(&mut self) {
        if !self.piece_open {
            return;
        }
        self.piece_open = false;
        let done = std::mem::replace(&mut self.encoder, fresh_encoder(&self.ctx, self.label));
        self.ctx.queue.submit([done.finish()]);
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
    pub(super) fn finish(mut self) {
        self.ctx.queue.submit([self.encoder.finish()]);
        self.piece_held.clear();
        for lease in self.piece_leases.drain(..).chain(self.run_leases.drain(..)) {
            self.pool.give(lease);
        }
        for lease in self.piece_buf_leases.drain(..) {
            self.pool.give_buf(lease);
        }
    }
}
