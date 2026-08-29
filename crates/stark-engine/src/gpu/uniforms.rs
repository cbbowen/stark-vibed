//! Buffers a pass grows to **what this frame holds**, in the two shapes a draw reads
//! them: [`UniformSlots`] for what varies across the draws of one submit, and
//! [`InstanceStream`] for the per-instance records beside it (§6.2).
//!
//! # The rule the slots exist for
//!
//! `write_buffer` is a *queue* operation, so N rewrites of one buffer before a single
//! submit leave every pass reading the last value written. Anything that varies per
//! draw therefore needs either a buffer per draw — a rate of small WebGPU
//! allocations, which is what JS GC cannot keep up with — or one buffer of
//! **dynamic-offset slots**, which is [`UniformSlots`].
//!
//! It lived in `composite::blend` and was used only there, while three other call
//! sites wrote a buffer *and* a bind group per draw for want of it: the transform's
//! per-quad uniform, the fill's per-tile origin, the selection's per-tile params.
//!
//! **`gpu::stroke` takes the stride and not the buffer**, which is the one place a
//! consumer departs from these types on purpose. Its two dynamic-offset uniforms —
//! the sweep's per-tile `TileXform` and the stamp loop's per-dispatch `Stamp` — take
//! their stride from [`UniformSlots::STRIDE`] like everyone else (`XFORM_STRIDE`,
//! `STAMP_STRIDE`), so the law is stated once; what they do not take is the grow-only
//! buffer, because that path has something stronger. A stroke's buffers are *leased*
//! from its scratch pool (`stroke::scratch`), which recycles them across strokes
//! rather than across the frames of one, and releases them only behind the submit of
//! the commands that named them.
//!
//! # Why the vertex side lives here too
//!
//! An [`InstanceStream`] obeys no slot law — a vertex buffer is indexed by the draw's
//! own instance range, not by an aligned offset — and shares the other half of what
//! `UniformSlots` is: *allocate to the high-water mark, never shrink within a
//! session, write the whole of this frame before the submit*. That policy was written
//! out four times in `composite` (pass A's tiles and mattes, pass C's outlines, pass
//! D's guides) as a `buf`/`cap` field pair and a grow-then-write block apiece. One
//! policy, one place; the two types differ in stride and usage and in nothing else.

/// The dynamic-offset alignment every backend accepts
/// (`min_uniform_buffer_offset_alignment` is 256 on the strictest) — the quantum a
/// [`UniformSlots`] stride is rounded up to. Exported for the merge renderer, whose
/// single blend uniform buffer is one such slot wide.
pub(crate) const UNIFORM_SLOT: u64 = 256;

/// A grow-on-demand buffer of uniform slots, one per pass — the one mechanism
/// behind the blend pass's per-merge uniforms and the filter pass's per-layer
/// ones, so the slot law lives in one place rather than once per pass that
/// needs it.
///
/// A slot per pass rather than one buffer rewritten between passes: `write_buffer`
/// is a *queue* operation, so N rewrites before a single submit would leave every
/// pass reading the last value written. Two blend groups — or two filters — in one
/// document is not an edge case, so a buffer holds them all and each pass binds its
/// own offset.
///
/// **Typed**, and the stride is the type's: [`Self::STRIDE`] is the uniform's own
/// size rounded up to [`UNIFORM_SLOT`], so a uniform that outgrows one alignment
/// quantum (the filter's did, when the gradient map's stop table landed — §21.11)
/// widens its own buffer's slots and nobody else's, and a buffer can no more be
/// written with the wrong shape than offset by the wrong stride.
///
/// The buffers themselves stay separate per pass (the two uniforms are different
/// shapes, and a document with three filters and no blend modes should not have to
/// reason about which slots the other pass skipped); what is shared is the
/// allocation, the growth policy, and the write-every-slot-before-the-submit rule.
pub(crate) struct UniformSlots<T> {
    buf: wgpu::Buffer,
    slots: usize,
    label: &'static str,
    _uniform: std::marker::PhantomData<T>,
}

impl<T: bytemuck::Pod> UniformSlots<T> {
    /// One slot's width for this uniform: its size, padded to the alignment.
    pub(crate) const STRIDE: u64 =
        (std::mem::size_of::<T>() as u64).div_ceil(UNIFORM_SLOT) * UNIFORM_SLOT;

    pub(crate) fn new(device: &wgpu::Device, label: &'static str, count: usize) -> Self {
        Self {
            buf: Self::alloc(device, label, count),
            slots: count.max(1),
            label,
            _uniform: std::marker::PhantomData,
        }
    }

    /// Write one uniform per slot, growing the buffer first if this frame has more
    /// of them than any before it. Every slot is written before the frame's single
    /// submit, which is the whole reason slots exist.
    ///
    /// **Returns whether the buffer moved.** Growing does not resize a buffer, it
    /// *replaces* one — so any bind group built over the old one is now naming a
    /// buffer too small for the offsets it is about to be given, which is a
    /// validation error rather than a wrong pixel. A caller that keeps such a bind
    /// group has to drop it when this says `true`; one that rebuilds per frame can
    /// ignore the answer, which is why this is a plain `bool` and not `#[must_use]`.
    pub(crate) fn write(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        uniforms: &[T],
    ) -> bool {
        if uniforms.is_empty() {
            return false;
        }
        let moved = uniforms.len() > self.slots;
        if moved {
            self.buf = Self::alloc(device, self.label, uniforms.len());
            self.slots = uniforms.len();
        }
        for (i, uniform) in uniforms.iter().enumerate() {
            queue.write_buffer(
                &self.buf,
                i as u64 * Self::STRIDE,
                bytemuck::bytes_of(uniform),
            );
        }
        moved
    }

    /// The dynamic offset slot `slot` binds at.
    pub(crate) fn offset(slot: u32) -> u32 {
        slot * Self::STRIDE as u32
    }

    /// This buffer as a dynamic-offset bind-group entry — one slot, at offset 0,
    /// which each draw then displaces by its own [`Self::offset`].
    pub(crate) fn binding(&self, binding: u32) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry {
            binding,
            resource: self.resource(),
        }
    }

    /// The same slot as a bare resource, for a group built from a shader-declared slot
    /// list (`desc::bind_group_for`), which supplies the binding index itself.
    pub(crate) fn resource(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.buf,
            offset: 0,
            size: wgpu::BufferSize::new(std::mem::size_of::<T>() as u64),
        })
    }

    fn alloc(device: &wgpu::Device, label: &'static str, count: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: Self::STRIDE * count.max(1) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
}

/// A grow-on-demand **vertex** buffer of per-instance records — [`UniformSlots`]'s
/// sibling, with the slot law removed and the growth policy kept.
///
/// Packed rather than padded: a vertex buffer is walked by the draw's own instance
/// range against the stride the pipeline's `VertexBufferLayout` declares, so there is
/// no alignment quantum to round up to and no offset for a caller to get wrong. What
/// it shares with `UniformSlots` is everything else — allocate to the high-water
/// mark, keep it, and write the whole of this frame's records in one go before the
/// submit that draws them.
///
/// The records past `items.len()` are left as whatever the last frame wrote. That is
/// sound because a draw names its own instance range and never reaches them, and it
/// is why this allocates with a bare `create_buffer`: two of the four hand-rolled
/// buffers this replaces used `create_buffer_init` with a `vec![Default; count]`,
/// building and uploading a CPU-side vector of placeholders that the very next
/// `write_buffer` overwrote in full.
pub(crate) struct InstanceStream<T> {
    buf: wgpu::Buffer,
    cap: usize,
    label: &'static str,
    _instance: std::marker::PhantomData<T>,
}

impl<T: bytemuck::Pod> InstanceStream<T> {
    pub(crate) fn new(device: &wgpu::Device, label: &'static str) -> Self {
        Self {
            buf: Self::alloc(device, label, 1),
            cap: 1,
            label,
            _instance: std::marker::PhantomData,
        }
    }

    /// Upload this frame's records, growing the buffer first if there are more of
    /// them than any frame before it. Empty is a no-op: nothing is drawn from a
    /// stream with no records, so there is nothing to overwrite.
    pub(crate) fn write(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, items: &[T]) {
        if items.is_empty() {
            return;
        }
        if items.len() > self.cap {
            self.buf = Self::alloc(device, self.label, items.len());
            self.cap = items.len();
        }
        queue.write_buffer(&self.buf, 0, bytemuck::cast_slice(items));
    }

    /// The whole buffer, as `set_vertex_buffer` takes it.
    pub(crate) fn slice(&self) -> wgpu::BufferSlice<'_> {
        self.buf.slice(..)
    }

    fn alloc(device: &wgpu::Device, label: &'static str, count: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (std::mem::size_of::<T>() * count.max(1)) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
}
