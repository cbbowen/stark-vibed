//! Per-layer blend modes and the isolation they are defined against (§18.0.4,
//! §14.7).
//!
//! A group with a mode of its own composites *alone* into a scratch pair, and one
//! fullscreen pass then merges that into the accumulator. Because the merge reads
//! the accumulator and writes the result, and a texture cannot be both, the
//! accumulator ping-pongs between the caller's pair and this module's `swap`.

use std::sync::OnceLock;

use crate::colorspace::ColorSpace;
use crate::document::BlendMode;
use crate::geom::Extent2;
use crate::gpu::channels::{ChannelFormats, Targets};
use crate::gpu::context::GpuContext;
use crate::gpu::desc;
use crate::gpu::pigment::PigmentLut;
use crate::gpu::uniforms::UniformSlots;

// Generated from `blend_common.wesl`'s own declaration (§6.7).
pub(crate) use stark_shaders::mirror::blend_common::Blend as BlendUniform;

/// The shader ABI for [`BlendMode`], kept here rather than on the enum: which `u32`
/// a mode is numbered is a fact about `blend_common.wesl`, not about the document.
///
/// `Normal` reaches the pass only when the group is **clipped** or carries an
/// opacity of its own (§14.4); an ordinary normal layer is the
/// absence of a pass.
pub(crate) fn blend_code(mode: BlendMode) -> u32 {
    match mode {
        BlendMode::Normal => 0,
        BlendMode::Reinhard => 1,
        BlendMode::Drago { .. } => 2,
        BlendMode::Multiply => 3,
    }
}

/// The blend pass: one fullscreen draw merging an isolated group into the
/// accumulator.
///
/// **Shared, not owned.** A merge-down through a blend mode runs this very pipeline on
/// tile-sized targets (§14.11, `gpu::merge`), so a merged layer cannot drift from the
/// stack it replaced — the same argument the eyedropper makes for sampling through the
/// compositor rather than beside it. It is behind an `Arc` for a blunter reason too:
/// building one decodes the Mixbox LUT, which is not a thing to do twice per
/// document.
pub(crate) struct BlendPass {
    pub(crate) pipeline: wgpu::RenderPipeline,
    pub(crate) bgl: wgpu::BindGroupLayout,
    pub(crate) pigment: PigmentLut,
}

impl BlendPass {
    pub(crate) fn new(ctx: &GpuContext, color_space: &dyn ColorSpace) -> Self {
        let device = &ctx.device;
        let color_format = ChannelFormats::of(color_space).color;
        let frag = wgpu::ShaderStages::FRAGMENT;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark blend"),
            source: wgpu::ShaderSource::Wgsl(color_space.blend_shader().into()),
        });
        // Its own bind group layout: every texture here is read with `textureLoad`
        // at the fragment's own coordinate, so nothing needs filtering — except the
        // pigment LUT, which is a table Mixbox interpolates in hardware
        // (`mixbox_lut.wesl`).
        let formats = ChannelFormats::of(color_space);
        let resid_format = formats.resid;
        let mut entries = vec![
            // One slot per blend group in the frame; see [`UniformSlots`].
            UniformSlots::<BlendUniform>::layout(0, frag),
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
                "the blend pass loads both residual targets with the color's decode",
            );
            entries.push(desc::load_tex(7, frag)); // accumulator residual
            entries.push(desc::load_tex(8, frag)); // isolated layer residual
        }
        let bgl = desc::bind_group_layout(device, "stark blend bgl", &entries);
        let layout = desc::pipeline_layout(device, "stark blend layout", &[Some(&bgl)]);
        // No fixed-function blend on either target: the pass computes the whole
        // merge — backdrop included — and *replaces* what it writes. That is the
        // point of the ping-pong.
        let targets = formats.targets();
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
                "no color space in this build has a pigment LUT to bind",
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

/// One set of channel targets — color, aux, and (in a space that has one) the
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
        formats: ChannelFormats,
    ) -> Self {
        let make = |format, label| super::offscreen_view(device, size, format, label);
        Self {
            color: make(formats.color, labels.0),
            aux: make(formats.aux, labels.1),
            // A pigment document isolates its residual alongside its concentrations:
            // the blend reads both to work out what light the layer carried
            // (§6.7), so a level that isolated only the color would hand the pass
            // a mixture and none of the correction that makes it a color.
            resid: formats.resid.map(|f| make(f, labels.2)),
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
    /// The blend and filter bind groups this level's passes read through, one per
    /// **phase** of its ping-pong.
    ///
    /// A bind group over an accumulator is fully determined by which way round the
    /// ping-pong currently is: a pass at level `l` reads either this level's `swap` or
    /// the stack's own target (the caller's at level 0, level `l−1`'s `iso` below
    /// that), and a merge's source is always this level's `iso`. Two phases, so two of
    /// each — however many merges the document has, and however many frames it is
    /// drawn for.
    ///
    /// Filled on first use and kept, `TilePairHandle::composite_bg`'s bargain with a
    /// shorter life: everything named here is either this level's own or the
    /// accumulator, and `ensure_targets` drops the whole scratch whenever the
    /// accumulator is rebuilt — so the lifetime is exactly the views'. That is what
    /// makes it a `OnceLock` rather than a cache with a key and an eviction policy.
    blend_bg: [OnceLock<wgpu::BindGroup>; 2],
    filter_bg: [OnceLock<wgpu::BindGroup>; 2],
}

impl ScratchLevel {
    fn new(device: &wgpu::Device, size: Extent2, iso: bool, formats: ChannelFormats) -> Self {
        let trio = |labels| Trio::new(device, size, labels, formats);
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
            blend_bg: Default::default(),
            filter_bg: Default::default(),
        }
    }

    /// The blend bind group for the phase in which the backdrop is (or is not) this
    /// level's `swap`, built on first use.
    pub(super) fn blend_bg(
        &self,
        back_is_swap: bool,
        make: impl FnOnce() -> wgpu::BindGroup,
    ) -> &wgpu::BindGroup {
        self.blend_bg[usize::from(back_is_swap)].get_or_init(make)
    }

    /// [`Self::blend_bg`] for the filter pass, which reads the same accumulator
    /// through a layout of its own (§21.3).
    pub(super) fn filter_bg(
        &self,
        back_is_swap: bool,
        make: impl FnOnce() -> wgpu::BindGroup,
    ) -> &wgpu::BindGroup {
        self.filter_bg[usize::from(back_is_swap)].get_or_init(make)
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
        formats: ChannelFormats,
    ) -> Self {
        Self {
            size,
            levels: needs
                .iter()
                .map(|&iso| ScratchLevel::new(device, size, iso, formats))
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
