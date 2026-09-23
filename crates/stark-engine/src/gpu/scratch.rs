//! The pooled stroke scratch (§6.2): working textures kept across folds, and the
//! **submit scope** that is the only way to hand one back.
//!
//! A checkout reuses a free texture whose [`Key`] matches exactly, and a lease goes
//! back on the free list *after the submit* of the commands recorded against it.
//!
//! **A lease can reach the free list only through a submit.** Commands on one queue
//! run in submission order, so a texture whose last use is already submitted is safe
//! to re-record against, while one an unsubmitted encoder still names is either a
//! failed submit or another stroke's pixels. The pool's `give` is therefore private
//! to this module, and the two ways a lease comes back both carry the ordering in
//! their shape:
//!
//! * [`SubmitScope`] — owns the encoder *and* the leases recorded against it, and
//!   releases the leases only in the same call that submits the encoder
//!   ([`SubmitScope::flush`] / [`SubmitScope::finish`]), so a call site never holds
//!   one loose.
//! * [`Kept`] — a lease that outlives its run (the tool-state copies), returned on
//!   drop. Sound on a borrow argument rather than a convention: see the type.
//!
//! A scope dropped on an unwind returns nothing to the pool, which is also sound:
//! its commands were never submitted, so nothing pending names its leases.
//!
//! **An unpooled resource merely dropped is not freed either.** On the web a drop
//! releases the JS handle and waits for GC, which cannot keep up with a rate, so a
//! resource replaced at one is `destroy()`ed instead — safe once the work naming it
//! is submitted, since WebGPU defers the real free until that work completes.
//!
//! [`SubmitScope`] is also what the renderers that rewrite whole tiles record
//! through — the transform, the merge, the fill, the selection. Those four take
//! their working tiles from [`TilePool`](crate::gpu::tile::TilePool) rather than from
//! here; what they get from this type is the ordering, and the leased buffers their
//! per-operation uniforms are staged into ([`SubmitScope::run_slots`]).
//!
//! Contents are **not** zeroed on reuse, and no consumer may rely on the
//! zero-initialization a fresh texture gets. Every key taken here is audited against
//! that: each target either loads with a clear, is fully copied into, or is fully
//! rewritten before anything reads it.

use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

/// How many tiles a scope records before it submits what it has and releases the
/// scratch behind it.
///
/// **This is what stops peak GPU memory scaling with the operation.** A blended
/// merge takes three scratch trios per tile (`merge::encode_blended`) and has no cap
/// on its tile count, so one encoder submitted once would hold every trio at the
/// same moment: ~15 GB and ~40,000 render passes at 10k tiles, a Windows TDR as much
/// as an allocation failure.
///
/// A cadence rather than a cap on the operation, because tiles are independent:
/// destinations are disjoint, each pass reads only its own tile's inputs and
/// per-operation uniforms that outlive the whole recording, and submits on one queue
/// execute in order. So cutting the recording anywhere is invisible in the result.
///
/// 256 bounds the blended merge's transient scratch at roughly 150 MB and its
/// command buffers at about a thousand render passes. Like
/// [`MAX_RELEASE_PER_EPOCH`](crate::gpu::tile) it is **a bound on a cost that has
/// not been measured**: raising it costs memory and lowers the submit count, and the
/// honest way to change it is to measure and say so.
pub(crate) const FLUSH_TILES: usize = 256;

use crate::gpu::context::GpuContext;
use crate::gpu::desc;
use crate::gpu::uniforms::{Slot, stage_slots};
use stark_model::geom::TILE_TEX;

use crate::unpoisoned;

/// How many bytes of free textures **and buffers** the pool will hold before it
/// starts destroying the least-recently-used. One wide-tip piece's working set —
/// region, narrow, snapshot and cells, three channels each in a pigment space — is
/// on the order of 120 MB, so this keeps roughly one generation warm plus headroom
/// for the sizes to drift across a tile boundary. Anything beyond is destroyed
/// eagerly rather than left to a collector (§6.2).
const POOL_BUDGET: u64 = 256 << 20;

/// What a checkout asks for: the exact descriptor it would otherwise create with,
/// plus the label a fresh creation is given. Exact size on purpose — the dynamics
/// shaders read `textureDimensions` of the snapshot and region, so an oversized
/// stand-in would change what they compute, not just what it costs.
///
/// **The label is not part of what makes two textures interchangeable**, which is
/// why there is no derived `Eq` here to say otherwise: two targets of the same
/// square, format and usage share one free list whatever they are called. The cost
/// is that a capture may show a reused texture under the name it was first created
/// with.
#[derive(Clone, Copy)]
pub(crate) struct Key {
    pub(crate) size: (u32, u32),
    pub(crate) format: wgpu::TextureFormat,
    pub(crate) usage: wgpu::TextureUsages,
    pub(crate) label: &'static str,
}

impl Key {
    /// A key for one **whole tile texture** — interior plus apron, the size every
    /// scratch that stands in for a tile is (§6.4). A scratch that got it wrong is
    /// one the write-back cuts the wrong block out of.
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

    /// The same key at `n×` the linear resolution — the supersampled sweep's parcel
    /// lanes (§6.2), which cover the same canvas extent at `n` texels per px so the
    /// integrate can box-resolve the slab law's output rather than its input.
    pub(crate) const fn scaled(self, n: u32) -> Self {
        Self {
            size: (self.size.0 * n, self.size.1 * n),
            ..self
        }
    }

    /// A copy extent covering the whole of a texture this key describes.
    ///
    /// Asked of the key rather than written beside it: a copy whose extent disagrees
    /// with the allocation it addresses does not fail, it moves the wrong block.
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

    /// What a texture this key describes costs the budget.
    fn bytes(&self) -> u64 {
        u64::from(self.size.0) * u64::from(self.size.1) * u64::from(desc::texel_bytes(self.format))
    }
}

/// What a scratch **buffer** checkout asks for. [`Key`]'s sibling, and it differs in
/// exactly one way: the size is rounded up rather than matched exactly.
///
/// **Rounded, because a buffer's size follows the drawing.** A sweep's instance
/// buffer is one record per segment-in-a-tile and grows through the stroke, so exact
/// matching would put a fresh buffer on the free list at every size the stroke passed
/// through and reuse none of them. Rounding to the next power of two gives a handful
/// of buckets a stroke settles into, for at most a factor of two of slack.
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

    /// A uniform buffer of `size` bytes, written through [`SubmitScope::write_lease`].
    fn uniform(size: u64, label: &'static str) -> Self {
        Self {
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            label,
        }
    }
}

/// A texture checkout: the texture and the whole-texture view onto it, pooled together
/// for the same reason the tile pool's are ([`Pooled`](crate::gpu::tile)): every
/// checkout wants that same view.
#[derive(Clone)]
pub(crate) struct LeasedTex {
    pub(crate) tex: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
}

/// Uniforms staged into a buffer a scope leased, laid out by [`Slot`]: bound through
/// [`window`](Self::window), each draw selecting its own by [`offset`](Self::offset).
///
/// What [`SubmitScope::run_uniform`], [`piece_uniform`](SubmitScope::piece_uniform),
/// [`run_slots`](SubmitScope::run_slots) and
/// [`piece_slots`](SubmitScope::piece_slots) hand back. Never empty: staging nothing
/// is `None` at either of the last two.
pub(crate) struct LeasedSlots<T> {
    buf: wgpu::Buffer,
    len: u32,
    _uniform: PhantomData<T>,
}

impl<T: bytemuck::Pod> LeasedSlots<T> {
    fn new(buf: wgpu::Buffer, len: u32) -> Self {
        Self {
            buf,
            len,
            _uniform: PhantomData,
        }
    }

    /// How many slots were staged.
    pub(crate) fn len(&self) -> usize {
        self.len as usize
    }

    /// The dynamic offset slot `i` binds at ([`Slot::offset`]).
    ///
    /// # Panics
    /// If no slot `i` was staged. A lease is not zeroed, so a draw past the last slot
    /// would read another operation's bytes rather than fail.
    pub(crate) fn offset(&self, i: usize) -> u32 {
        assert!(i < self.len(), "slot {i} of the {} staged", self.len);
        // Lossless: below `len`, which is a `u32`.
        Slot::<T>::offset(i as u32)
    }

    /// Slot 0 as a binding, one uniform wide ([`Slot::window`]).
    pub(crate) fn window(&self) -> wgpu::BindingResource<'_> {
        Slot::<T>::window(&self.buf)
    }
}

/// One checked-out resource and the key it was created for, which is also what it
/// costs the budget.
enum Lease {
    Tex { tex: LeasedTex, key: Key },
    Buf { buf: wgpu::Buffer, key: BufKey },
}

impl Lease {
    fn bytes(&self) -> u64 {
        match self {
            Lease::Tex { key, .. } => key.bytes(),
            Lease::Buf { key, .. } => key.bucket(),
        }
    }

    /// Destroyed outright rather than dropped: the free is deferred past any in-flight
    /// use, and waiting on GC instead is how the tab OOMs (§6.2).
    fn destroy(self) {
        match self {
            Lease::Tex { tex, .. } => tex.tex.destroy(),
            Lease::Buf { buf, .. } => buf.destroy(),
        }
    }
}

/// A lease on the free list.
struct Entry {
    lease: Lease,
    /// The pool tick this entry was handed back on — the eviction order.
    last: u64,
}

#[derive(Default)]
struct Inner {
    /// Textures and buffers on one list: they come out of one device and one budget.
    free: Vec<Entry>,
    /// Total bytes on the free list, maintained so the budget check is a compare
    /// rather than a walk.
    bytes: u64,
    tick: u64,
    /// Bytes checked out and not yet handed back.
    leased: u64,
    /// The most `leased` has reached since [`ScratchPool::take_peak`] last asked.
    peak: u64,
}

/// The pool itself: shared by every clone of the renderer, so the live fold and the
/// commit that replaces it draw from one free list. `Default` is the empty pool —
/// it needs no device; each checkout brings one.
///
/// **The stroke path and the whole-tile renderers beside it hold the same one.** The
/// latter take only the scope — their working textures come from `TilePool` through
/// `Channels::scratch` — so what the pool buys them is the submit-then-release
/// ordering (§7), and their uniforms' buffers.
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
            run: Vec::new(),
            piece: Vec::new(),
            piece_held: Vec::new(),
            piece_open: false,
            since_flush: 0,
            staging: Vec::new(),
            #[cfg(debug_assertions)]
            _open: open::Open::new(),
        }
    }

    /// Check a lease out that will **outlive** the scope that records against it —
    /// the tool-state copies — wrapped so its return path is its drop.
    pub(crate) fn keep(&self, device: &wgpu::Device, key: Key) -> Kept {
        let (tex, lease) = self.take_tex(device, key);
        Kept {
            tex,
            lease: Some(lease),
            pool: self.clone(),
        }
    }

    /// Check a texture out: a free one matching `key` exactly, or a fresh creation.
    fn take_tex(&self, device: &wgpu::Device, key: Key) -> (LeasedTex, Lease) {
        let reused = self.reuse(|held| match held {
            Lease::Tex { tex, key: had } if had.interchangeable(&key) => Some(tex.clone()),
            _ => None,
        });
        let taken = reused.unwrap_or_else(|| {
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
            let tex = LeasedTex { tex, view };
            (tex.clone(), Lease::Tex { tex, key })
        });
        self.lent(&taken.1);
        taken
    }

    /// Check a buffer out: a free one [`BufKey::interchangeable`] with `key`, or a
    /// fresh creation at the key's bucket size.
    fn take_buf(&self, device: &wgpu::Device, key: BufKey) -> (wgpu::Buffer, Lease) {
        let reused = self.reuse(|held| match held {
            Lease::Buf { buf, key: had } if had.interchangeable(&key) => Some(buf.clone()),
            _ => None,
        });
        let taken = reused.unwrap_or_else(|| {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(key.label),
                size: key.bucket(),
                usage: key.usage,
                mapped_at_creation: false,
            });
            (buf.clone(), Lease::Buf { buf, key })
        });
        self.lent(&taken.1);
        taken
    }

    /// Count `lease` as checked out.
    fn lent(&self, lease: &Lease) {
        let mut inner = unpoisoned(self.0.lock());
        inner.leased += lease.bytes();
        inner.peak = inner.peak.max(inner.leased);
    }

    /// The most bytes checked out at once since the last call, restarting the count
    /// from what is checked out now.
    pub(crate) fn take_peak(&self) -> u64 {
        let mut inner = unpoisoned(self.0.lock());
        let now = inner.leased;
        std::mem::replace(&mut inner.peak, now)
    }

    /// Take the newest free entry `claim` accepts off the list, with the handles it
    /// projected out of it.
    ///
    /// Newest first: the piece that just gave these back is the likeliest shape of the
    /// piece about to take them.
    fn reuse<T>(&self, claim: impl Fn(&Lease) -> Option<T>) -> Option<(T, Lease)> {
        let mut inner = unpoisoned(self.0.lock());
        inner.tick += 1;
        let (i, handles) = inner
            .free
            .iter()
            .enumerate()
            .rev()
            .find_map(|(i, e)| Some((i, claim(&e.lease)?)))?;
        let e = inner.free.swap_remove(i);
        inner.bytes -= e.lease.bytes();
        Some((handles, e.lease))
    }

    /// Hand leases back to the free list, then hold it to [`POOL_BUDGET`].
    ///
    /// **Private on purpose** — the module doc's whole argument. The callers are
    /// [`SubmitScope`], which has just submitted the commands naming the leases, and
    /// [`Kept`]'s drop.
    fn give(&self, leases: impl IntoIterator<Item = Lease>) {
        let mut inner = unpoisoned(self.0.lock());
        for lease in leases {
            inner.tick += 1;
            let last = inner.tick;
            inner.leased -= lease.bytes();
            inner.bytes += lease.bytes();
            inner.free.push(Entry { lease, last });
        }
        trim(&mut inner, POOL_BUDGET);
    }
}

/// Destroy free entries, least recently returned first, until `inner` holds at most
/// `budget` bytes.
///
/// **One budget over textures and buffers, evicted strictly by age**: the device has
/// one memory, so two budgets would be two numbers to pick.
fn trim(inner: &mut Inner, budget: u64) {
    while inner.bytes > budget {
        let Some(oldest) = inner
            .free
            .iter()
            .enumerate()
            .min_by_key(|(_, e)| e.last)
            .map(|(i, _)| i)
        else {
            break;
        };
        let e = inner.free.swap_remove(oldest);
        inner.bytes -= e.lease.bytes();
        e.lease.destroy();
    }
}

/// A pooled lease held **outside** any scope — the tool-state copies, which outlive
/// the run that recorded them — returned to the pool when its owner drops it.
///
/// Sound on a borrow argument rather than a convention: the commands that read one
/// are recorded by the run that *resumes* from it, which takes the owning
/// `stroke::incremental`'s `ToolState` by `&` and submits before it returns, so the
/// owner can only drop this after that submit. An unwind mid-run drops the run's
/// encoder unsubmitted, so nothing pending names the lease on that path either.
pub(crate) struct Kept {
    tex: LeasedTex,
    /// `Some` until the drop hands it back.
    lease: Option<Lease>,
    pool: ScratchPool,
}

impl Kept {
    /// The leased texture — what a resume copy reads from and a capture writes to.
    pub(crate) fn tex(&self) -> &wgpu::Texture {
        &self.tex.tex
    }

    /// The texture with its whole-texture view, as a scope's own checkouts hand them
    /// out — so one pass records against a kept lease and a run lease alike.
    pub(crate) fn leased(&self) -> &LeasedTex {
        &self.tex
    }
}

impl Drop for Kept {
    fn drop(&mut self) {
        if let Some(lease) = self.lease.take() {
            self.pool.give([lease]);
        }
    }
}

/// GPU work being recorded, and everything whose reuse must wait behind its submit:
/// the pooled leases the commands name — textures and buffers alike — and any other
/// resource whose drop must trail the submit (the swept path's tile-pool scratch
/// pair, the stamp loop's carried mint budget).
///
/// Two lease lifetimes, because a dynamics stroke has two: **run** leases carry
/// state across pieces (the reservoir ping-pong, the bake pair, the sweep's instance
/// run and per-tile slots) and go back only at [`finish`](Self::finish); **piece**
/// leases (the region, the snapshot, the cells, a plan's staging) go back at each
/// [`flush`](Self::flush), which keeps a long stroke's peak transient memory at one
/// region however many pieces it takes.
pub(crate) struct SubmitScope {
    ctx: GpuContext,
    pool: ScratchPool,
    encoder: wgpu::CommandEncoder,
    label: &'static str,
    run: Vec<Lease>,
    piece: Vec<Lease>,
    /// Arbitrary resources whose *drop* must trail the submit — a pooled tile handle,
    /// whose early drop puts it back on `TilePool`'s free list while this encoder
    /// still names it.
    piece_held: Vec<Box<dyn std::any::Any>>,
    /// Whether anything has been recorded or taken since the last submit — what
    /// keeps an operation that touched no tile from submitting an empty command
    /// buffer.
    piece_open: bool,
    /// Tiles recorded since the last submit, against [`FLUSH_TILES`] — the cadence
    /// [`tile_done`](Self::tile_done) counts and nothing else uses.
    since_flush: usize,
    /// [`run_slots`](Self::run_slots)/[`piece_slots`](Self::piece_slots) staging,
    /// kept across the scope's lifetime: a dynamics piece stages a whole plan, so a
    /// fresh `Vec` per call was that allocation per pointer move.
    staging: Vec<u8>,
    #[cfg(debug_assertions)]
    _open: open::Open,
}

/// The scopes open on this thread, counted in a debug build — what
/// [`TilePool::release_idle`](crate::gpu::tile::TilePool::release_idle) asserts is
/// zero, since a free-list texture may be named by an open scope's unsubmitted
/// encoder.
///
/// Per thread, not per pool: a scope is `!Send`, and every engine sharing a pool runs
/// on the thread that owns it, so a scope that could name this pool's textures is one
/// on this thread. A count per device would also see unrelated engines, which one
/// `cargo test` process runs side by side on one shared device.
#[cfg(debug_assertions)]
mod open {
    use std::cell::Cell;

    std::thread_local! {
        static OPEN: Cell<usize> = const { Cell::new(0) };
    }

    /// One scope, counted open for as long as it lives — through a flush, and until
    /// `finish` or an unwind drops it.
    pub(super) struct Open(());

    impl Open {
        pub(super) fn new() -> Self {
            OPEN.with(|n| n.set(n.get() + 1));
            Self(())
        }
    }

    impl Drop for Open {
        fn drop(&mut self) {
            // Cannot fail: a `const`-initialized `Cell` registers no destructor, so
            // it is readable for the thread's whole life.
            OPEN.with(|n| n.set(n.get() - 1));
        }
    }

    pub(super) fn scopes() -> usize {
        OPEN.with(Cell::get)
    }
}

/// [`SubmitScope`]s open on this thread (debug builds only).
#[cfg(debug_assertions)]
pub(crate) fn open_scopes() -> usize {
    open::scopes()
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

    /// Record one [fullscreen pass](desc::fullscreen_pass) — the shape every renderer
    /// here writes a destination tile or mask with. `into` is a channel trio, whose
    /// residual `Option` carries the attachment count (§6.7), or a single target.
    pub(crate) fn fullscreen_pass<'a>(
        &mut self,
        label: &str,
        into: impl Into<desc::Attachments<'a>>,
        pipeline: &wgpu::RenderPipeline,
        bound: desc::Bound<'_>,
    ) {
        desc::fullscreen_pass(self.encoder(), label, into, pipeline, bound);
    }

    /// Note that one tile has been recorded, submitting and releasing if that
    /// reaches [`FLUSH_TILES`].
    ///
    /// Call once per destination tile, at the point where everything that tile needs
    /// has been recorded — never in the middle of one, since the scratch a
    /// half-recorded tile is holding is exactly what a flush would hand away.
    pub(crate) fn tile_done(&mut self) {
        self.since_flush += 1;
        if self.since_flush >= FLUSH_TILES {
            self.flush();
        }
    }

    /// Check out scratch that carries state **across** pieces — released only at
    /// [`finish`](Self::finish), behind the submit of everything recorded.
    pub(crate) fn take_run(&mut self, key: Key) -> LeasedTex {
        let (tex, lease) = self.pool.take_tex(&self.ctx.device, key);
        self.run.push(lease);
        tex
    }

    /// Check out scratch for the piece being recorded — released at the piece's own
    /// submit ([`flush`](Self::flush) or [`finish`](Self::finish)).
    ///
    /// **Taking opens the piece**, where [`take_piece_buffer`](Self::take_piece_buffer)
    /// leaves that to the write. The asymmetry is about what is being bounded: a
    /// [`flush`](Self::flush) with the piece closed returns early and hands nothing
    /// back, so a texture taken here would ride out the whole scope — and these are
    /// region-sized, which is the peak [`FLUSH_TILES`] exists to cap. A buffer lease is
    /// bucketed and small, and the thing that must not outrun a submit is its pending
    /// [`write_lease`](Self::write_lease), which opens the piece exactly when it
    /// happens. The price here is one empty command buffer for a texture nothing was
    /// recorded against.
    pub(crate) fn take_piece(&mut self, key: Key) -> LeasedTex {
        self.piece_open = true;
        let (tex, lease) = self.pool.take_tex(&self.ctx.device, key);
        self.piece.push(lease);
        tex
    }

    /// Check out a pooled buffer that carries **across** pieces — released only at
    /// [`finish`](Self::finish), exactly as [`take_run`](Self::take_run) is.
    ///
    /// Taking does **not** open the piece; [`write_lease`](Self::write_lease) does. A
    /// lease nobody wrote and nobody recorded against has nothing pending, and
    /// submitting for it is the empty command buffer [`finish`](Self::finish) skips.
    pub(crate) fn take_run_buffer(&mut self, key: BufKey) -> wgpu::Buffer {
        let (buf, lease) = self.pool.take_buf(&self.ctx.device, key);
        self.run.push(lease);
        buf
    }

    /// Check out a pooled buffer for the piece being recorded — released at the
    /// piece's own submit, exactly as [`take_piece`](Self::take_piece) is. Taking does
    /// not open the piece; [`write_lease`](Self::write_lease) does.
    ///
    /// **At least `key.size` bytes, and possibly more** ([`BufKey::bucket`]): a
    /// caller writes its own prefix and draws its own range, so slack past the end is
    /// unread. Contents are not zeroed, on the pool's general contract.
    pub(crate) fn take_piece_buffer(&mut self, key: BufKey) -> wgpu::Buffer {
        let (buf, lease) = self.pool.take_buf(&self.ctx.device, key);
        self.piece.push(lease);
        buf
    }

    /// Stage `value` into a leased uniform buffer for the whole run — a per-operation
    /// uniform every tile's pass binds, so it outlives each [`flush`](Self::flush).
    pub(crate) fn run_uniform<T: bytemuck::Pod>(
        &mut self,
        label: &'static str,
        value: &T,
    ) -> LeasedSlots<T> {
        let buf = self.take_run_buffer(BufKey::uniform(size_of::<T>() as u64, label));
        self.write_lease(&buf, bytemuck::bytes_of(value));
        LeasedSlots::new(buf, 1)
    }

    /// [`run_uniform`](Self::run_uniform) for the piece being recorded, released at
    /// its submit like everything else the piece takes.
    pub(crate) fn piece_uniform<T: bytemuck::Pod>(
        &mut self,
        label: &'static str,
        value: &T,
    ) -> LeasedSlots<T> {
        let buf = self.take_piece_buffer(BufKey::uniform(size_of::<T>() as u64, label));
        self.write_lease(&buf, bytemuck::bytes_of(value));
        LeasedSlots::new(buf, 1)
    }

    /// Stage `uniforms` one per [`Slot`] into a leased buffer for the whole run — an
    /// operation's per-tile or per-draw uniforms.
    ///
    /// **All of them, before anything is recorded against them.** A slot has to be
    /// written before the submit that reads it and the scope may flush at any
    /// [`tile_done`](Self::tile_done), while a buffer grown mid-recording would move
    /// out from under the offsets already recorded.
    ///
    /// `None` for none: no draw will bind it, and staging opens the piece, so an
    /// operation that records nothing would otherwise submit for nothing.
    pub(crate) fn run_slots<'a, T: bytemuck::Pod>(
        &mut self,
        label: &'static str,
        uniforms: impl IntoIterator<Item = &'a T, IntoIter: ExactSizeIterator>,
    ) -> Option<LeasedSlots<T>> {
        self.slots(label, uniforms, Self::take_run_buffer)
    }

    /// [`run_slots`](Self::run_slots) for the piece being recorded, released at its own
    /// submit like everything else the piece takes — a dynamics plan, whose slots
    /// nothing past that piece's dispatches reads.
    pub(crate) fn piece_slots<'a, T: bytemuck::Pod>(
        &mut self,
        label: &'static str,
        uniforms: impl IntoIterator<Item = &'a T, IntoIter: ExactSizeIterator>,
    ) -> Option<LeasedSlots<T>> {
        self.slots(label, uniforms, Self::take_piece_buffer)
    }

    /// The two above, over the lease tier `take` checks out of.
    fn slots<'a, T: bytemuck::Pod>(
        &mut self,
        label: &'static str,
        uniforms: impl IntoIterator<Item = &'a T, IntoIter: ExactSizeIterator>,
        take: fn(&mut Self, BufKey) -> wgpu::Buffer,
    ) -> Option<LeasedSlots<T>> {
        let uniforms = uniforms.into_iter();
        let len = u32::try_from(uniforms.len()).expect("an operation's draws are capped");
        if len == 0 {
            return None;
        }
        let mut staged = std::mem::take(&mut self.staging);
        stage_slots(uniforms, &mut staged);
        let buf = take(self, BufKey::uniform(staged.len() as u64, label));
        self.write_lease(&buf, &staged);
        self.staging = staged;
        Some(LeasedSlots::new(buf, len))
    }

    /// Stage `bytes` into a **leased** buffer, opening the piece.
    ///
    /// **The one way to write a lease.** `queue.write_buffer` does not write anything:
    /// it stages the bytes and hands them over at the next submit. So a leased buffer
    /// with a pending write must not reach the free list before one — and the moment
    /// that becomes true is the *write*, not the checkout. Routing it through the scope
    /// makes the write and the flag one operation; a caller reaching past this to
    /// `ctx.queue` reopens the hazard.
    pub(crate) fn write_lease(&mut self, buf: &wgpu::Buffer, bytes: &[u8]) {
        self.piece_open = true;
        self.ctx.queue.write_buffer(buf, 0, bytes);
    }

    /// [`write_lease`](Self::write_lease) for a leased **texture**: stage `bytes` as
    /// `layout` into `extent` at its origin, opening the piece for the same reason.
    /// `extent` may be narrower than the lease — a texture pooled at a bucketed size
    /// is written to the width its reader was told (the lasso's edge list, §6.8).
    pub(crate) fn write_texture_lease(
        &mut self,
        tex: &wgpu::Texture,
        bytes: &[u8],
        layout: wgpu::TexelCopyBufferLayout,
        extent: wgpu::Extent3d,
    ) {
        self.piece_open = true;
        self.ctx
            .queue
            .write_texture(tex.as_image_copy(), bytes, layout, extent);
    }

    /// Keep `thing` alive past the submit, dropping it just after — for resources
    /// whose drop *is* their release to some other pool.
    ///
    /// Opens the piece conservatively rather than necessarily: `thing` is `dyn Any`,
    /// so this call cannot know what its drop releases and assumes the worst for the
    /// price of one empty command buffer.
    pub(crate) fn hold(&mut self, thing: impl std::any::Any) {
        self.piece_open = true;
        self.piece_held.push(Box::new(thing));
    }

    /// Close out the piece already recorded, if any: submit the encoder, then
    /// release the piece's resources — in that order, which is the type's whole
    /// reason to exist. Run leases stay, so peak transient memory is one region
    /// however long the stroke.
    pub(crate) fn flush(&mut self) {
        // Above the early return, not below it: `tile_done` counts unconditionally, so
        // a tile that recorded nothing still counts, and leaving the tally standing
        // through a no-op flush would make the *next* tile — the first with anything in
        // it — submit on its own instead of at the cadence.
        self.since_flush = 0;
        if !self.piece_open {
            return;
        }
        self.piece_open = false;
        let done = std::mem::replace(&mut self.encoder, fresh_encoder(&self.ctx, self.label));
        self.ctx.queue.submit([done.finish()]);
        self.piece_held.clear();
        self.pool.give(self.piece.drain(..));
    }

    /// Close the scope: submit what is still recorded, then release everything —
    /// the piece's resources and the run leases both, behind that same submit.
    pub(crate) fn finish(mut self) {
        // Only if there is something to submit. An empty command buffer is a wasm
        // boundary crossing for nothing, and the operations that reach here having
        // recorded nothing are ordinary: a lopsided merge passes every tile by handle,
        // `apply_filter` may find an empty map, a transform may have empty plans.
        if self.piece_open {
            self.ctx.queue.submit([self.encoder.finish()]);
        }
        self.piece_held.clear();
        self.pool
            .give(self.piece.drain(..).chain(self.run.drain(..)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::context::GpuContext;
    use crate::gpu::tile::{AllocSource, TilePool};

    const COLOR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

    /// A context, or `None` where the machine has no adapter *and*
    /// [`ALLOW_NO_GPU`](crate::testing::ALLOW_NO_GPU) permits the skip.
    fn context_or_skip() -> Option<GpuContext> {
        crate::testing::or_skip(pollster::block_on(GpuContext::headless()), "GPU tests")
    }

    /// **The cadence is what bounds peak memory**, which is the whole reason
    /// [`FLUSH_TILES`] exists: without it a merge holds every tile's scratch at once,
    /// and a merge has no cap on its tile count.
    ///
    /// Asked of the pool, where the cost lands: a scope that never flushed would leave
    /// one texture per tile on the free list at `finish`, where a flushing one reuses
    /// the same working set and ends with about a cadence's worth. The tile count stays
    /// under `TRIM_INTERVAL` acquires so the pool's own trim cannot fire and make the
    /// numbers a matter of two policies rather than one.
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
            desc::begin_pass(scope.encoder(), "stark submit test pass", scratch.view());
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

    /// **The tile pool's idle textures go back to the driver only between
    /// operations**: a debug build refuses `release_idle` while a scope on this thread
    /// is open, since its unsubmitted encoder may name a texture on the free list —
    /// and the refusal comes before anything is taken off it.
    #[cfg(debug_assertions)]
    #[test]
    fn idle_tiles_are_not_released_while_a_scope_is_open() {
        let Some(ctx) = context_or_skip() else { return };
        let pool = TilePool::new(ctx.clone(), [COLOR]);
        drop(pool.acquire_tex(COLOR, AllocSource::Unknown));
        assert_eq!(pool.free_count(COLOR), 1);

        let scope = ScratchPool::default().scope(&ctx, "stark release test");
        let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.release_idle();
        }));
        assert!(refused.is_err(), "released idle textures with a scope open");
        assert_eq!(
            pool.free_count(COLOR),
            1,
            "the refused release took a texture"
        );

        scope.finish();
        pool.release_idle();
        assert_eq!(
            pool.free_count(COLOR),
            0,
            "a closed scope still blocks the release"
        );
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

    /// **Staging nothing opens nothing**: an operation with no draws leases no uniform
    /// buffer, so it still submits nothing. Staging anything opens the piece, since
    /// the write is handed over only at the next submit and the lease must not reach
    /// the free list before it.
    #[test]
    fn staging_opens_the_piece_and_staging_nothing_does_not() {
        type U = [f32; 4];
        let Some(ctx) = context_or_skip() else { return };
        let mut scope = ScratchPool::default().scope(&ctx, "stark submit test staging");
        assert!(scope.run_slots::<U>("stark submit test", &[]).is_none());
        assert!(!scope.piece_open, "nothing was staged");
        let slots = scope
            .run_slots::<U>("stark submit test", &[[1.0; 4]; 3])
            .expect("three were staged");
        assert!(scope.piece_open, "a staged write opens the piece");
        assert_eq!(slots.len(), 3);
        assert_eq!(slots.offset(2), Slot::<U>::offset(2));
        scope.finish();
    }

    /// **Eviction is by age alone**, not by kind or size: the oldest return goes first
    /// whatever it is — here a small buffer ahead of a larger texture returned after
    /// it, and then that texture though a buffer is also free.
    #[test]
    fn the_budget_evicts_the_oldest_return_whatever_it_is() {
        let Some(ctx) = context_or_skip() else { return };
        let (pool, device) = (ScratchPool::default(), &ctx.device);
        let buf = BufKey {
            size: 1,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            label: "stark scratch test buffer",
        };
        let tex = Key::tile(
            COLOR,
            wgpu::TextureUsages::TEXTURE_BINDING,
            "stark scratch test texture",
        );
        // All three out at once, so each is its own allocation, then back oldest first.
        let (_, old_buf) = pool.take_buf(device, buf);
        let (_, texture) = pool.take_tex(device, tex);
        let (_, new_buf) = pool.take_buf(device, buf);
        pool.give([old_buf, texture, new_buf]);

        let mut inner = unpoisoned(pool.0.lock());
        let by_age = |inner: &Inner| {
            let mut free: Vec<_> = inner
                .free
                .iter()
                .map(|e| (e.last, matches!(e.lease, Lease::Tex { .. })))
                .collect();
            free.sort_unstable();
            free.into_iter()
                .map(|(_, is_tex)| is_tex)
                .collect::<Vec<_>>()
        };
        assert_eq!(by_age(&inner), [false, true, false]);
        let budget = inner.bytes - 1;
        trim(&mut inner, budget);
        assert_eq!(by_age(&inner), [true, false], "the first buffer goes first");
        let budget = inner.bytes - 1;
        trim(&mut inner, budget);
        assert_eq!(by_age(&inner), [false], "then the texture");
    }
}
