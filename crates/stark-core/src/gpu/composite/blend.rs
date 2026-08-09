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

/// One dynamic-offset slot of the blend uniform, padded to the alignment every
/// backend accepts (`min_uniform_buffer_offset_alignment` is 256 on the strictest).
///
/// A slot per blend group rather than one buffer rewritten per pass: `write_buffer`
/// is a *queue* operation, so N rewrites before a single submit would leave every
/// pass reading the last mode written. Two blend layers in one document is not an
/// edge case, so the buffer holds them all and each pass binds its own offset.
pub(super) const BLEND_SLOT: u64 = 256;

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
        // so the one bind group layout still has something to bind.
        let pigment = if color_space.needs_pigment_lut() {
            PigmentLut::load(ctx)
        } else {
            PigmentLut::placeholder(ctx)
        };
        Self {
            pipeline,
            bgl,
            pigment,
        }
    }
}

pub(super) fn alloc_blend(device: &wgpu::Device, count: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("stark blend uniform"),
        size: BLEND_SLOT * count.max(1) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// The extra viewport-sized targets **one level** of isolation needs
/// (§18.0.4).
///
/// Two pairs, not one. `iso` is where a group composites alone; `swap` is the other
/// half of a ping-pong, because the blend pass reads the accumulator and writes the
/// merged result and a texture cannot be both.
pub(super) struct ScratchLevel {
    swap_color: wgpu::TextureView,
    swap_aux: wgpu::TextureView,
    swap_resid: Option<wgpu::TextureView>,
    iso_color: wgpu::TextureView,
    iso_aux: wgpu::TextureView,
    iso_resid: Option<wgpu::TextureView>,
}

impl ScratchLevel {
    fn new(
        device: &wgpu::Device,
        size: Extent2,
        color_format: wgpu::TextureFormat,
        aux_format: wgpu::TextureFormat,
        resid_format: Option<wgpu::TextureFormat>,
    ) -> Self {
        let make = |format, label| super::offscreen_view(device, size, format, label);
        Self {
            swap_color: make(color_format, "stark blend swap color"),
            swap_aux: make(aux_format, "stark blend swap aux"),
            // A pigment document isolates its residual alongside its concentrations:
            // the blend reads both to work out what light the layer carried
            // (§6.7), so a level that isolated only the colour would hand the pass
            // a mixture and none of the correction that makes it a colour.
            swap_resid: resid_format.map(|f| make(f, "stark blend swap resid")),
            iso_color: make(color_format, "stark blend iso color"),
            iso_aux: make(aux_format, "stark blend iso aux"),
            iso_resid: resid_format.map(|f| make(f, "stark blend iso resid")),
        }
    }

    pub(super) fn swap(&self) -> Targets<'_> {
        Targets {
            color: &self.swap_color,
            aux: &self.swap_aux,
            resid: self.swap_resid.as_ref(),
        }
    }

    pub(super) fn iso(&self) -> Targets<'_> {
        Targets {
            color: &self.iso_color,
            aux: &self.iso_aux,
            resid: self.iso_resid.as_ref(),
        }
    }
}

/// One [`ScratchLevel`] per level of group nesting the document actually reaches
/// (§14.7).
///
/// A group's members isolate into *its* level's `iso`, which is the target the
/// next level down composites into — so nesting costs one of these per level and
/// not one per group. Allocated only when a document contains something that has
/// to be isolated at all: an ordinary painting never pays the ~40 MB, and one
/// that uses blend modes without groups pays for exactly the one level it uses.
pub(super) struct ScratchTargets {
    pub(super) size: Extent2,
    pub(super) levels: Vec<ScratchLevel>,
}

impl ScratchTargets {
    pub(super) fn new(
        device: &wgpu::Device,
        size: Extent2,
        levels: usize,
        color_format: wgpu::TextureFormat,
        aux_format: wgpu::TextureFormat,
        resid_format: Option<wgpu::TextureFormat>,
    ) -> Self {
        Self {
            size,
            levels: (0..levels)
                .map(|_| ScratchLevel::new(device, size, color_format, aux_format, resid_format))
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
