//! The pooled stroke scratch (§6.2): working textures kept across folds, and the
//! **submit scope** that is the only way to hand one back.
//!
//! Every live update used to allocate its region, snapshot, cell, reservoir and
//! bake textures afresh and `destroy()` them at submit — tens of megabytes of
//! creation, zero-initialization and teardown per fold, all of it for textures the
//! next fold would ask for again at the same sizes. This pool keeps them: a
//! checkout reuses a free texture whose [`Key`] matches exactly, and a lease goes
//! back on the free list *after the submit* of the commands recorded against it —
//! commands on one queue run in submission order, so a texture whose last use is
//! already submitted can be re-recorded against freely.
//!
//! **A lease can reach the free list only through a submit.** That used to be a
//! convention argued in a comment at every release site — and the `TilePool`
//! free-list-vs-open-encoder incident is what a missed site costs: "no live
//! handle" is not "no pending GPU work", and a texture handed out while an
//! unsubmitted encoder still names it is either a failed submit or another
//! stroke's pixels. So the pool's `give` is private to this module, and the two
//! ways a lease comes back both carry the ordering in their shape:
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
//! Contents are **not** zeroed on reuse, and no consumer may rely on the
//! zero-initialization a fresh texture gets. That is an audited property of every
//! key taken here: the region and narrow targets load with a clear; the reservoir is
//! cleared or fully copied into before any read, and its passes store every texel
//! they own; the bake is fully rewritten per segment (it is already reused across a
//! stroke's segments on exactly this argument); the snapshot's stores and the
//! deposit's reads are gated by the same `outside_sweep` predicate, so a texel one
//! skips the other never loads; and a cell the coarse deposit can name is one its
//! own hoist wrote (`plan::cell_geometry`). The tool-state copies are written whole
//! by the very copy that checks them out.

use std::sync::{Arc, Mutex};

use crate::gpu::context::GpuContext;

use super::{ScopedResources, unpoisoned};

/// How many bytes of free textures the pool will hold before it starts destroying
/// the least-recently-used. One wide-tip piece's working set — region, narrow,
/// snapshot and cells, three channels each in a pigment space — is on the order of
/// 120 MB, so this keeps roughly one generation warm plus headroom for the sizes to
/// drift across a tile boundary; anything beyond that is destroyed eagerly, for the
/// same reason `ScopedResources` destroys rather than waiting on GC (§6.2).
const POOL_BUDGET: u64 = 256 << 20;

/// What makes two scratch textures interchangeable: the exact descriptor a checkout
/// would otherwise create with. Exact size on purpose — the dynamics shaders read
/// `textureDimensions` of the snapshot and region, so an oversized stand-in would
/// change what they compute, not just what it costs. The label is part of the key,
/// which both keeps a debug capture truthful and gives each purpose its own line in
/// the free list.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Key {
    pub(super) size: (u32, u32),
    pub(super) format: wgpu::TextureFormat,
    pub(super) usage: wgpu::TextureUsages,
    pub(super) label: &'static str,
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
    /// Total bytes on the free list, maintained so the budget check is a compare
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
            piece_scoped: ScopedResources::default(),
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
            if let Some(i) = inner.free.iter().rposition(|e| e.key == key) {
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
        while inner.bytes > POOL_BUDGET {
            let (i, _) = inner
                .free
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last)
                .expect("bytes > 0 means the list is non-empty");
            let e = inner.free.swap_remove(i);
            inner.bytes -= e.bytes;
            // Destroyed outright rather than dropped: the free is deferred past any
            // in-flight use, and waiting on GC instead is how the tab OOMs (§6.2).
            e.tex.destroy();
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
}

impl Drop for Kept {
    fn drop(&mut self) {
        if let Some(lease) = self.lease.take() {
            self.pool.give(lease);
        }
    }
}

/// GPU work being recorded, and everything whose reuse or destruction must wait
/// behind its submit: the pooled leases the commands name, the per-stroke
/// buffers/textures destroyed once they land, and any other resource whose drop
/// must trail the submit (the swept path's tile-pool scratch pair).
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
    /// Per-piece buffers and textures that are not pooled (uniforms, instance
    /// buffers, the gathered selection mask) — destroyed at the piece's submit.
    piece_scoped: ScopedResources,
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

    /// Register a per-piece buffer; returns it unchanged, destroyed at the piece's
    /// submit ([`ScopedResources`]'s argument: the allocation *rate* is what JS GC
    /// cannot keep up with).
    pub(super) fn buffer(&mut self, buf: wgpu::Buffer) -> wgpu::Buffer {
        self.piece_open = true;
        self.piece_scoped.buffer(buf)
    }

    /// Register a per-piece texture; returns it unchanged.
    pub(super) fn texture(&mut self, tex: wgpu::Texture) -> wgpu::Texture {
        self.piece_open = true;
        self.piece_scoped.texture(tex)
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
        drop(std::mem::take(&mut self.piece_scoped));
        self.piece_held.clear();
        for lease in self.piece_leases.drain(..) {
            self.pool.give(lease);
        }
    }

    /// Close the scope: submit what is still recorded, then release everything —
    /// the piece's resources and the run leases both, behind that same submit.
    pub(super) fn finish(mut self) {
        self.ctx.queue.submit([self.encoder.finish()]);
        drop(self.piece_scoped);
        self.piece_held.clear();
        for lease in self.piece_leases.drain(..).chain(self.run_leases.drain(..)) {
            self.pool.give(lease);
        }
    }
}
