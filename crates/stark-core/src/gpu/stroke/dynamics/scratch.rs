//! The pooled dynamics scratch (§6.2): the loop's working textures, kept across
//! folds instead of created and destroyed per pointer move.
//!
//! Every live update used to allocate its region, snapshot, cell, reservoir and
//! bake textures afresh and `destroy()` them at submit — tens of megabytes of
//! creation, zero-initialization and teardown per fold, all of it for textures the
//! next fold would ask for again at the same sizes. This pool keeps them: a checkout
//! ([`ScratchPool::take`]) reuses a free texture whose [`Key`] matches exactly, and
//! the run hands its leases back ([`ScratchPool::give`]) *after the submit*, which
//! is the whole reuse argument — commands on one queue run in submission order, so
//! a texture whose last use is already submitted can be re-recorded against freely.
//! A lease dropped on an unwind returns nothing to the pool, which is also sound:
//! its commands were never submitted.
//!
//! Contents are **not** zeroed on reuse, and no consumer may rely on the
//! zero-initialization a fresh texture gets. That is an audited property of every
//! key taken here: the region and narrow targets load with a clear; the reservoir is
//! cleared or fully copied into before any read, and its passes store every texel
//! they own; the bake is fully rewritten per segment (it is already reused across a
//! stroke's segments on exactly this argument); the snapshot's stores and the
//! deposit's reads are gated by the same `outside_sweep` predicate, so a texel one
//! skips the other never loads; and a cell the coarse deposit can name is one its
//! own hoist wrote (`plan::cell_geometry`).

use std::sync::{Arc, Mutex};

use super::super::unpoisoned;

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
pub(super) struct Lease {
    pub(super) tex: wgpu::Texture,
    pub(super) view: wgpu::TextureView,
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
pub(in crate::gpu::stroke) struct ScratchPool(Arc<Mutex<Inner>>);

impl ScratchPool {
    /// Check a texture out: the newest free entry matching `key` exactly, or a fresh
    /// creation when there is none.
    pub(super) fn take(&self, device: &wgpu::Device, key: Key) -> Lease {
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
    /// destroying least-recently-returned entries. **Only after the GPU work that
    /// references the lease is submitted** — which is when the run's own
    /// `flush`/`submit` call this, and what makes the next checkout free to record
    /// against it.
    pub(super) fn give(&self, lease: Lease) {
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
