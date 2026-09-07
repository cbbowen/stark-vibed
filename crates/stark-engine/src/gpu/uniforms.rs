//! Buffers a pass grows to **what this frame holds**, in the two shapes a draw reads
//! them: [`UniformSlots`] for what varies across the draws of one submit, and
//! [`InstanceStream`] for the per-instance records beside it (§6.2).
//!
//! # The rule the slots exist for
//!
//! `write_buffer` is a *queue* operation, so N rewrites of one buffer before a single
//! submit leave every pass reading the last value written. Anything that varies per
//! draw therefore needs either a buffer per draw — a rate of small WebGPU allocations
//! JS GC cannot keep up with — or one buffer of **dynamic-offset slots**, which is
//! [`UniformSlots`].
//!
//! **`gpu::stroke` takes the stride and not the buffer**, the one deliberate departure.
//! Its two dynamic-offset uniforms take their stride from [`UniformSlots::STRIDE`]
//! (`XFORM_STRIDE`, `STAMP_STRIDE`) so the law is stated once, but their buffers are
//! *leased* from the stroke scratch pool (`gpu::scratch`), which recycles across strokes
//! and releases only behind the submit of the commands that named them.
//!
//! # Why the vertex side lives here too
//!
//! An [`InstanceStream`] obeys no slot law — a vertex buffer is indexed by the draw's
//! own instance range, not by an aligned offset — and shares the other half of what
//! `UniformSlots` is: *allocate to the high-water mark, never shrink within a session,
//! write the whole of this frame before the submit*.

/// The dynamic-offset alignment every backend accepts
/// (`min_uniform_buffer_offset_alignment` is 256 on the strictest) — the quantum a
/// [`UniformSlots`] stride is rounded up to. Exported for the merge renderer, whose
/// single blend uniform buffer is one such slot wide.
pub(crate) const UNIFORM_SLOT: u64 = 256;

/// A grow-on-demand buffer of uniform slots, one per pass — the blend pass's per-merge
/// uniforms and the filter pass's per-layer ones.
///
/// A slot per pass rather than one buffer rewritten between passes: `write_buffer` is a
/// *queue* operation, so N rewrites before a single submit would leave every pass
/// reading the last value written. Two blend groups — or two filters — in one document
/// is not an edge case, so a buffer holds them all and each pass binds its own offset.
///
/// **Typed**, and the stride is the type's: [`Self::STRIDE`] is the uniform's own size
/// rounded up to [`UNIFORM_SLOT`] (§21.11), so a uniform that outgrows one alignment
/// quantum widens its own buffer's slots and nobody else's, and a buffer can no more be
/// written with the wrong shape than offset by the wrong stride.
///
/// A buffer per pass, since the uniforms are different shapes; what is shared is the
/// allocation, the growth policy, and the write-every-slot-before-the-submit rule.
pub(crate) struct UniformSlots<T> {
    buf: wgpu::Buffer,
    slots: usize,
    label: &'static str,
    /// A bind group over [`buf`](Self::buf), built on first ask and **dropped by the
    /// write that replaces the buffer** ([`Self::write`]).
    ///
    /// Here rather than beside each consumer because the invalidation is the whole of
    /// the difficulty: growing does not resize a buffer, it replaces one, and a group
    /// over the old allocation names a buffer too small for the offsets it is about to
    /// be given. The type that *does* the replacing is the one that clears the cache.
    ///
    /// `None` for the consumers that keep no group: a bind group answers to a layout,
    /// and several of these are bound as part of a larger group somebody else builds.
    group: Option<wgpu::BindGroup>,
    /// The frame's slots laid out at [`Self::STRIDE`], kept so [`Self::write`] is
    /// one `write_buffer` and no allocation after the first.
    staging: Vec<u8>,
    _uniform: std::marker::PhantomData<T>,
}

/// Lay `uniforms` out one per [`UniformSlots::STRIDE`] into `into` — the bytes a single
/// `write_buffer` uploads for a whole pass. Padding is zeroed; nothing reads it.
pub(crate) fn stage_slots<T: bytemuck::Pod>(uniforms: &[T], into: &mut Vec<u8>) {
    let stride = UniformSlots::<T>::STRIDE as usize;
    into.clear();
    into.resize(uniforms.len() * stride, 0);
    for (slot, uniform) in into.chunks_exact_mut(stride).zip(uniforms) {
        slot[..std::mem::size_of::<T>()].copy_from_slice(bytemuck::bytes_of(uniform));
    }
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
            group: None,
            staging: Vec::new(),
            _uniform: std::marker::PhantomData,
        }
    }

    /// A bind group over these slots, built by `make` on first ask after each growth.
    ///
    /// `make` is handed the slot as a [`BindingResource`](wgpu::BindingResource) —
    /// offset 0, one uniform wide — which each draw then displaces by its own
    /// [`offset`](Self::offset). It is not handed `self`: what it may name is the thing
    /// whose replacement invalidates the group.
    pub(crate) fn group(
        &mut self,
        make: impl FnOnce(wgpu::BindingResource<'_>) -> wgpu::BindGroup,
    ) -> &wgpu::BindGroup {
        let Self { buf, group, .. } = self;
        group.get_or_insert_with(|| {
            make(wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: buf,
                offset: 0,
                size: wgpu::BufferSize::new(std::mem::size_of::<T>() as u64),
            }))
        })
    }

    /// The group [`group`](Self::group) built, for a reader that cannot take `&mut`.
    ///
    /// `None` before the first `group` call after a growth — which is why the two are
    /// separate: a caller ensures the group while it has the buffer to hand, and reads
    /// it back later while it does not.
    pub(crate) fn built_group(&self) -> Option<&wgpu::BindGroup> {
        self.group.as_ref()
    }

    /// Write one uniform per slot, growing the buffer first if this frame has more
    /// of them than any before it. Every slot is written before the frame's single
    /// submit, which is the whole reason slots exist.
    ///
    /// **Returns whether the buffer moved.** Growing *replaces* the buffer, so any bind
    /// group built over the old one now names one too small for the offsets it is about
    /// to be given — a validation error rather than a wrong pixel. The group this type
    /// keeps ([`Self::group`]) is dropped here; the answer is for a caller holding one
    /// of its *own*, over a layout this type knows nothing about.
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
            // The group named the buffer that is now gone.
            self.group = None;
        }
        stage_slots(uniforms, &mut self.staging);
        queue.write_buffer(&self.buf, 0, &self.staging);
        moved
    }

    /// The dynamic offset slot `slot` binds at.
    pub(crate) fn offset(slot: u32) -> u32 {
        slot * Self::STRIDE as u32
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
/// range against the stride the pipeline's `VertexBufferLayout` declares, so there is no
/// alignment quantum to round up to and no offset for a caller to get wrong. The rest is
/// `UniformSlots`' policy — allocate to the high-water mark, keep it, and write the
/// whole of this frame's records before the submit that draws them.
///
/// The records past `items.len()` are left as whatever the last frame wrote, which is
/// sound because a draw names its own instance range and never reaches them.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// [`stage_slots`] is the slot law as bytes: uniform `i` starts at `i · STRIDE`,
    /// the staging spans exactly `len · STRIDE`, and the padding is zero. Adapter-free,
    /// so the one layout every dynamic-offset draw in the crate reads through can be
    /// checked where no device exists.
    #[test]
    fn slots_are_staged_at_stride_apart() {
        type U = [f32; 4];
        const STRIDE: usize = UniformSlots::<U>::STRIDE as usize;
        let uniforms: [U; 3] = [[1.0, 2.0, 3.0, 4.0], [5.0; 4], [-1.0, 0.0, 0.5, 9.0]];
        let mut staged = vec![0xAAu8; 7];
        stage_slots(&uniforms, &mut staged);
        assert_eq!(
            staged.len(),
            3 * STRIDE,
            "one stride per slot, nothing else"
        );
        for (i, u) in uniforms.iter().enumerate() {
            let at = i * STRIDE;
            assert_eq!(
                &staged[at..at + std::mem::size_of::<U>()],
                bytemuck::bytes_of(u),
                "slot {i}"
            );
            assert!(
                staged[at + std::mem::size_of::<U>()..at + STRIDE]
                    .iter()
                    .all(|b| *b == 0),
                "slot {i}'s padding is not zeroed"
            );
        }
        stage_slots::<U>(&[], &mut staged);
        assert!(staged.is_empty(), "nothing staged for nothing");
    }
}
