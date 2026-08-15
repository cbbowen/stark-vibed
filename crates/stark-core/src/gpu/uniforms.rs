//! Uniforms a pass varies **across the draws of one submit**, and the one rule that
//! makes that work (§6.2).
//!
//! `write_buffer` is a *queue* operation, so N rewrites of one buffer before a single
//! submit leave every pass reading the last value written. Anything that varies per
//! draw therefore needs either a buffer per draw — a rate of small WebGPU
//! allocations, which is what JS GC cannot keep up with — or one buffer of
//! **dynamic-offset slots**, which is this.
//!
//! It lived in `composite::blend` and was used only there, while three other call
//! sites wrote a buffer *and* a bind group per draw for want of it: the transform's
//! per-quad uniform, the fill's per-tile origin, the selection's per-tile params.
//! `gpu::stroke` had a third copy of the law as a bare `UNIFORM_STRIDE` constant.

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
    pub(crate) fn write(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, uniforms: &[T]) {
        if uniforms.is_empty() {
            return;
        }
        if uniforms.len() > self.slots {
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
    }

    /// The dynamic offset slot `slot` binds at.
    pub(crate) fn offset(slot: u32) -> u32 {
        slot * Self::STRIDE as u32
    }

    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        &self.buf
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
