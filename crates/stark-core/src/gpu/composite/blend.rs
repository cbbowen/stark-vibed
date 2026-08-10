//! Per-layer blend modes and the isolation they are defined against (§18.0.4,
//! §14.7).
//!
//! A group with a mode of its own composites *alone* into a scratch pair, and one
//! fullscreen pass then merges that into the accumulator. Because the merge reads
//! the accumulator and writes the result, and a texture cannot be both, the
//! accumulator ping-pongs between the caller's pair and this module's `swap`.

use crate::colorspace::ColorSpace;
use crate::document::BlendMode;
use crate::geom::Extent2;
use crate::gpu::context::GpuContext;
use crate::gpu::desc;
use crate::gpu::pigment::PigmentLut;

/// The channel targets pass A hands around: colour, aux, and — in a space that has
/// one — the residual (§6.7).
///
/// A struct rather than the tuple this was, because the third is an `Option` and
/// `target.2` would have said nothing about why. `resid` is `Some` for every target
/// of a pigment document and `None` for every target of a colorimetric one; it is
/// decided by the colour space once, never per call site.
#[derive(Copy, Clone)]
pub(super) struct Targets<'a> {
    pub(super) color: &'a wgpu::TextureView,
    pub(super) aux: &'a wgpu::TextureView,
    pub(super) resid: Option<&'a wgpu::TextureView>,
}

impl<'a> Targets<'a> {
    /// The colour attachments in target order, with `resid` at location 2 when the
    /// space has one — the order every pass A pipeline declares.
    ///
    /// Returned as an array-plus-length rather than a `Vec`: this is called once per
    /// render pass encoded, and a render pass per tile run is the common case.
    pub(super) fn attachments(
        &self,
        ops: wgpu::Operations<wgpu::Color>,
    ) -> [Option<wgpu::RenderPassColorAttachment<'a>>; 3] {
        [
            Some(desc::attach(self.color, ops)),
            Some(desc::attach(self.aux, ops)),
            self.resid.map(|v| desc::attach(v, ops)),
        ]
    }

    /// How many of [`Self::attachments`] are real — 2 without a residual, 3 with.
    pub(super) fn count(&self) -> usize {
        2 + usize::from(self.resid.is_some())
    }
}

// Generated from `blend_common.wesl`'s own declaration (§6.7).
pub(super) use stark_shaders::mirror::blend_common::Blend as BlendUniform;

/// The shader ABI for [`BlendMode`], kept here rather than on the enum: which `u32`
/// a mode is numbered is a fact about `blend_common.wesl`, not about the document.
///
/// `Normal` reaches the pass only when the group is **clipped** or carries an
/// opacity of its own (§14.4); an ordinary normal layer is the
/// absence of a pass.
pub(super) fn blend_code(mode: BlendMode) -> u32 {
    match mode {
        BlendMode::Normal => 0,
        BlendMode::Reinhard => 1,
        BlendMode::Drago => 2,
        BlendMode::Multiply => 3,
    }
}

/// One dynamic-offset uniform slot, padded to the alignment every backend accepts
/// (`min_uniform_buffer_offset_alignment` is 256 on the strictest). The blend and
/// filter passes share the number because they share the law behind it.
///
/// A slot per pass rather than one buffer rewritten between passes: `write_buffer`
/// is a *queue* operation, so N rewrites before a single submit would leave every
/// pass reading the last value written. Two blend groups — or two filters — in one
/// document is not an edge case, so a buffer holds them all and each pass binds its
/// own offset.
pub(super) const UNIFORM_SLOT: u64 = 256;

/// A grow-on-demand buffer of [`UNIFORM_SLOT`]-sized uniform slots — the one
/// mechanism behind the blend pass's per-merge uniforms and the filter pass's
/// per-layer ones, so the slot law above lives in one place rather than once per
/// pass that needs it.
///
/// The buffers themselves stay separate per pass (the two uniforms are different
/// shapes, and a document with three filters and no blend modes should not have to
/// reason about which slots the other pass skipped); what is shared is the
/// allocation, the growth policy, and the write-every-slot-before-the-submit rule.
pub(super) struct UniformSlots {
    buf: wgpu::Buffer,
    slots: usize,
    label: &'static str,
}

impl UniformSlots {
    pub(super) fn new(device: &wgpu::Device, label: &'static str, count: usize) -> Self {
        Self {
            buf: alloc_slots(device, label, count),
            slots: count.max(1),
            label,
        }
    }

    /// Write one uniform per slot, growing the buffer first if this frame has more
    /// of them than any before it. Every slot is written before the frame's single
    /// submit, which is the whole reason slots exist — see [`UNIFORM_SLOT`].
    pub(super) fn write<T: bytemuck::Pod>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        uniforms: &[T],
    ) {
        debug_assert!(
            std::mem::size_of::<T>() as u64 <= UNIFORM_SLOT,
            "a pass uniform must fit its slot",
        );
        if uniforms.is_empty() {
            return;
        }
        if uniforms.len() > self.slots {
            self.buf = alloc_slots(device, self.label, uniforms.len());
            self.slots = uniforms.len();
        }
        for (i, uniform) in uniforms.iter().enumerate() {
            queue.write_buffer(
                &self.buf,
                i as u64 * UNIFORM_SLOT,
                bytemuck::bytes_of(uniform),
            );
        }
    }

    pub(super) fn buffer(&self) -> &wgpu::Buffer {
        &self.buf
    }
}

fn alloc_slots(device: &wgpu::Device, label: &'static str, count: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: UNIFORM_SLOT * count.max(1) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// The blend pass: one fullscreen draw merging an isolated group into the
/// accumulator.
pub(super) struct BlendPass {
    pub(super) pipeline: wgpu::RenderPipeline,
    pub(super) bgl: wgpu::BindGroupLayout,
    pub(super) pigment: PigmentLut,
}

impl BlendPass {
    pub(super) fn new(
        ctx: &GpuContext,
        color_space: &dyn ColorSpace,
        color_format: wgpu::TextureFormat,
        aux_format: wgpu::TextureFormat,
    ) -> Self {
        let device = &ctx.device;
        let frag = wgpu::ShaderStages::FRAGMENT;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark blend"),
            source: wgpu::ShaderSource::Wgsl(color_space.blend_shader().into()),
        });
        // Its own bind group layout: every texture here is read with `textureLoad`
        // at the fragment's own coordinate, so nothing needs filtering — except the
        // pigment LUT, which is a table Mixbox interpolates in hardware
        // (`mixbox_lut.wesl`).
        let resid_format = color_space.resid_format();
        let mut entries = vec![
            // One slot per blend group in the frame; see [`BLEND_SLOT`].
            desc::uniform_slot(0, frag, std::mem::size_of::<BlendUniform>() as u64),
            desc::load_tex(1, frag),   // accumulator color
            desc::load_tex(2, frag),   // accumulator aux
            desc::load_tex(3, frag),   // isolated layer color
            desc::load_tex(4, frag),   // isolated layer aux
            desc::sample_tex(5, frag), // pigment LUT (filtered)
            desc::sampler(6, frag),
        ];
        // 7 and 8, the two residuals — `blend_mixbox.wesl` declares them itself, past
        // where `blend_common` (0–4) and `mixbox_lut` (5–6) stop. Unlike the LUT above
        // these get no placeholder: the shader that reads them is reached only by the
        // space that has them, so there is a layout per space rather than one layout
        // and a texture Oklab would bind and never sample.
        if let Some(f) = resid_format {
            debug_assert_eq!(
                f, color_format,
                "the blend pass loads both residual targets with the colour's decode",
            );
            entries.push(desc::load_tex(7, frag)); // accumulator residual
            entries.push(desc::load_tex(8, frag)); // isolated layer residual
        }
        let bgl = desc::bind_group_layout(device, "stark blend bgl", &entries);
        let layout = desc::pipeline_layout(device, "stark blend layout", &[Some(&bgl)]);
        // No fixed-function blend on either target: the pass computes the whole
        // merge — backdrop included — and *replaces* what it writes. That is the
        // point of the ping-pong.
        let mut targets = vec![desc::target(color_format), desc::target(aux_format)];
        if let Some(f) = resid_format {
            targets.push(desc::target(f));
        }
        let pipeline = desc::fullscreen_pipeline(
            device,
            "stark blend pipeline",
            &layout,
            &shader,
            ("vs_main", "fs_main"),
            &targets,
        );
        // Decoded only where it is read from: an Oklab document gets a 1×1 stand-in
        // so the one bind group layout still has something to bind. Without the
        // `mixbox` feature no space asks for the real table, and there is none to
        // decode — `needs_pigment_lut` is then false for every space in the build, so
        // this is the stand-in unconditionally.
        #[cfg(feature = "mixbox")]
        let pigment = if color_space.needs_pigment_lut() {
            PigmentLut::load(ctx)
        } else {
            PigmentLut::placeholder(ctx)
        };
        #[cfg(not(feature = "mixbox"))]
        let pigment = {
            debug_assert!(
                !color_space.needs_pigment_lut(),
                "no colour space in this build has a pigment LUT to bind",
            );
            PigmentLut::placeholder(ctx)
        };
        Self {
            pipeline,
            bgl,
            pigment,
        }
    }
}

/// One set of channel targets — colour, aux, and (in a space that has one) the
/// residual — owned rather than borrowed, as [`Targets`] is the borrowed view of.
struct Trio {
    color: wgpu::TextureView,
    aux: wgpu::TextureView,
    resid: Option<wgpu::TextureView>,
}

impl Trio {
    fn new(
        device: &wgpu::Device,
        size: Extent2,
        labels: (&str, &str, &str),
        color_format: wgpu::TextureFormat,
        aux_format: wgpu::TextureFormat,
        resid_format: Option<wgpu::TextureFormat>,
    ) -> Self {
        let make = |format, label| super::offscreen_view(device, size, format, label);
        Self {
            color: make(color_format, labels.0),
            aux: make(aux_format, labels.1),
            // A pigment document isolates its residual alongside its concentrations:
            // the blend reads both to work out what light the layer carried
            // (§6.7), so a level that isolated only the colour would hand the pass
            // a mixture and none of the correction that makes it a colour.
            resid: resid_format.map(|f| make(f, labels.2)),
        }
    }

    fn targets(&self) -> Targets<'_> {
        Targets {
            color: &self.color,
            aux: &self.aux,
            resid: self.resid.as_ref(),
        }
    }
}

/// The extra viewport-sized targets **one level** of isolation needs
/// (§18.0.4).
///
/// `swap` is the other half of a ping-pong, because a merge — and a filter pass —
/// reads the accumulator and writes the result and a texture cannot be both; every
/// level has one. `iso`, where a group composites alone, exists only on a level
/// whose stack actually isolates something: a level that only ping-pongs (a stack
/// whose sole non-direct members are filters, §21.3) never allocates the trio it
/// provably cannot bind.
pub(super) struct ScratchLevel {
    swap: Trio,
    iso: Option<Trio>,
}

impl ScratchLevel {
    fn new(
        device: &wgpu::Device,
        size: Extent2,
        iso: bool,
        color_format: wgpu::TextureFormat,
        aux_format: wgpu::TextureFormat,
        resid_format: Option<wgpu::TextureFormat>,
    ) -> Self {
        let trio = |labels| Trio::new(device, size, labels, color_format, aux_format, resid_format);
        Self {
            swap: trio((
                "stark blend swap color",
                "stark blend swap aux",
                "stark blend swap resid",
            )),
            iso: iso.then(|| {
                trio((
                    "stark blend iso color",
                    "stark blend iso aux",
                    "stark blend iso resid",
                ))
            }),
        }
    }

    pub(super) fn swap(&self) -> Targets<'_> {
        self.swap.targets()
    }

    /// Whether this level was allocated with an isolation trio — what
    /// `ensure_scratch` checks a cached level against the frame's needs with.
    pub(super) fn has_iso(&self) -> bool {
        self.iso.is_some()
    }

    pub(super) fn iso(&self) -> Targets<'_> {
        self.iso
            .as_ref()
            .expect("a merge at a level allocated without iso scratch (scratch_needs)")
            .targets()
    }
}

/// One [`ScratchLevel`] per level of group nesting the document actually reaches
/// (§14.7).
///
/// A group's members isolate into *its* level's `iso`, which is the target the
/// next level down composites into — so nesting costs one of these per level and
/// not one per group. Allocated only when a document contains something that has
/// to be isolated at all — an ordinary painting never pays the ~40 MB — and each
/// level allocates only the half its stack uses (`needs`, from
/// [`scratch_needs`](super::group::scratch_needs)): a document whose only
/// non-`Normal` thing is a filter pays for the ping-pong pair alone.
pub(super) struct ScratchTargets {
    pub(super) size: Extent2,
    pub(super) levels: Vec<ScratchLevel>,
}

impl ScratchTargets {
    pub(super) fn new(
        device: &wgpu::Device,
        size: Extent2,
        needs: &[bool],
        color_format: wgpu::TextureFormat,
        aux_format: wgpu::TextureFormat,
        resid_format: Option<wgpu::TextureFormat>,
    ) -> Self {
        Self {
            size,
            levels: needs
                .iter()
                .map(|&iso| {
                    ScratchLevel::new(device, size, iso, color_format, aux_format, resid_format)
                })
                .collect(),
        }
    }
}

/// A render pass that only clears. Encoded when the bottom of the stack is a blend
/// group: that pass *reads* the accumulator, so unlike a run of tiles it cannot
/// fold the clear into its own load op.
pub(super) fn clear_targets(encoder: &mut wgpu::CommandEncoder, into: Targets<'_>) {
    let attachments = into.attachments(desc::CLEAR);
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("stark composite clear"),
        color_attachments: &attachments[..into.count()],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
}
