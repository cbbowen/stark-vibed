//! Per-layer blend modes and the isolation they are defined against (§18.0.4,
//! §14.7).
//!
//! A group with a mode of its own composites *alone* into a scratch pair, and one
//! fullscreen pass then merges that into the accumulator. Because the merge reads
//! the accumulator and writes the result, and a texture cannot be both, the
//! accumulator ping-pongs between the caller's pair and this module's `swap`.

use std::sync::OnceLock;

use crate::colorspace::ColorSpace;
use crate::gpu::channels::{ChannelFormats, Targets};
use crate::gpu::context::GpuContext;
use crate::gpu::desc::{self, Slot};
use stark_model::document::BlendMode;
use stark_model::geom::Extent2;
use stark_shaders::mirror::blend_common::binding as bc;
use stark_shaders::mirror::blend_common::decl as bcd;
use stark_shaders::mirror::blend_mixbox::binding as bm;
use stark_shaders::mirror::blend_mixbox::decl as bmd;
use stark_shaders::mirror::mixbox_lut::binding as ml;
use stark_shaders::mirror::mixbox_lut::decl as mld;

/// Which bindings the blend pass reads, in layout order (§6.10).
///
/// **Three modules share this group**, and the list is where that shows: 0–4 are
/// `blend_common.wesl`'s, 5–6 `mixbox_lut.wesl`'s, and 7–8 `blend_mixbox.wesl`'s. The
/// partition used to be held by a comment in each file; the slot list names the
/// declarations, so the host cannot disagree with any of the three about an index, and
/// `build.rs` checks the linked artifact for a collision between them.
pub(crate) const BLEND_SLOTS: &[Slot] = &[
    // One slot per blend group in the frame; see [`UniformSlots`].
    Slot::dynamic(bcd::B),
    Slot::at(bcd::BACK_COLOR),
    Slot::at(bcd::BACK_AUX),
    Slot::at(bcd::SRC_COLOR),
    Slot::at(bcd::SRC_AUX),
    // The pigment LUT is a table Mixbox interpolates in hardware (`mixbox_lut.wesl`);
    // every other texture here is `textureLoad`ed at the fragment's own coordinate.
    Slot::sampled(mld::PIGMENT_LUT),
    Slot::at(mld::PIGMENT_SAMP),
    // The two residuals get no placeholder, unlike the LUT above: the shader that reads
    // them is reached only by the space that has them, so there is a layout per space
    // rather than one layout and a texture Oklab would bind and never sample.
    Slot::at(bmd::BACK_RESID).only_with_resid(),
    Slot::at(bmd::SRC_RESID).only_with_resid(),
];

/// A texture view as the resource a bind-group entry takes.
fn tex(v: &wgpu::TextureView) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::TextureView(v)
}
use crate::gpu::pigment::PigmentLut;
use crate::gpu::uniforms::UniformSlots;

use super::plan::Phase;

// Generated from `blend_common.wesl`'s own declaration (§6.7).
pub(crate) use stark_shaders::mirror::blend_common::Blend as BlendUniform;

/// The shader ABI for [`BlendMode`], kept here rather than on the enum: which `u32`
/// a mode is numbered is a fact about `blend_common.wesl`, not about the document.
///
/// And it is that shader's own number, generated from its declaration (§6.10). The
/// four literals that stood here were the thing the mirror exists to prevent — a
/// second declaration of the ABI, three files from the first, with a comment in
/// `blend_common.wesl` claiming they were "mirrored" when they were transcribed.
///
/// `Normal` reaches the pass only when the group is **clipped** or carries an
/// opacity of its own (§14.4); an ordinary normal layer is the
/// absence of a pass.
pub(crate) fn blend_code(mode: BlendMode) -> u32 {
    use stark_shaders::mirror::blend_common as bc;
    match mode {
        BlendMode::Normal => bc::MODE_NORMAL,
        BlendMode::Reinhard => bc::MODE_REINHARD,
        BlendMode::Drago { .. } => bc::MODE_DRAGO,
        BlendMode::Multiply => bc::MODE_MULTIPLY,
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
        let resid = formats.has_resid();
        if let Some(f) = formats.resid {
            debug_assert_eq!(
                f, color_format,
                "the blend pass loads both residual targets with the color's decode",
            );
        }
        let bgl = desc::layout_for(device, "stark blend bgl", BLEND_SLOTS, frag, resid);
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

    /// Encode one merge: the isolated layer `src` into the accumulator `b.back`,
    /// through blend slot `b.slot`, writing `b.out` (§18.0.4).
    pub(super) fn encode(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        b: Bounce<'_>,
        src: Targets<'_>,
        slots: &UniformSlots<BlendUniform>,
    ) {
        // Two per level rather than one per merge per frame: the views this names are
        // fixed by the phase, and the whole scratch is dropped when they change (see
        // [`ScratchLevel::blend_bg`]).
        let bg = b.here.blend_bg(b.phase.back_is_swap, || {
            // Both residuals or neither: `back`, `src` and `out` are all targets of
            // the same document, so the space that gave one a residual gave all three
            // one — which is why one `resid` answers for the pair.
            let resid = b.back.resid.is_some() && src.resid.is_some();
            desc::bind_group_for(
                &ctx.device,
                "stark blend bg",
                &self.bgl,
                BLEND_SLOTS,
                resid,
                |i| match i {
                    bc::B => slots.resource(),
                    bc::BACK_COLOR => tex(b.back.color),
                    bc::BACK_AUX => tex(b.back.aux),
                    bc::SRC_COLOR => tex(src.color),
                    bc::SRC_AUX => tex(src.aux),
                    ml::PIGMENT_LUT => tex(&self.pigment.view),
                    ml::PIGMENT_SAMP => wgpu::BindingResource::Sampler(&self.pigment.sampler),
                    bm::BACK_RESID => tex(b.back.resid.expect("a residual build has one")),
                    bm::SRC_RESID => tex(src.resid.expect("a residual build has one")),
                    other => unreachable!("`BLEND_SLOTS` lists no binding {other}"),
                },
            )
        });
        b.pass(
            encoder,
            "stark blend pass",
            &self.pipeline,
            bg,
            UniformSlots::<BlendUniform>::offset(b.slot),
        );
    }
}

/// One **bouncing** pass: what a merge and a filter both need beyond the pipeline kit
/// (§18.0.4, §21.3).
///
/// The two are the same shape — read the accumulator, write the other half of the
/// ping-pong, off a uniform slot of their own — and differ only in that a merge also
/// binds an isolated source. Here rather than in `filter.rs` because the level it
/// names is this module's, which is also why the filter pass borrows the scratch:
/// they bounce through the same one.
pub(super) struct Bounce<'a> {
    /// The accumulator this pass reads.
    pub(super) back: Targets<'a>,
    /// The other half of the ping-pong, which it wholly replaces.
    pub(super) out: Targets<'a>,
    /// Which dynamic-offset slot holds this pass's uniform.
    pub(super) slot: u32,
    /// The level it bounces at, which owns the bind groups it reads through.
    pub(super) here: &'a ScratchLevel,
    pub(super) phase: Phase,
}

impl Bounce<'_> {
    /// The render pass both bouncing passes encode: one fullscreen triangle that
    /// replaces every texel of `out`. Shared with [`FilterPass::encode`], which is
    /// this pass with the source removed (§21.3).
    ///
    /// [`FilterPass::encode`]: super::filter::FilterPass::encode
    pub(super) fn pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
        pipeline: &wgpu::RenderPipeline,
        bg: &wgpu::BindGroup,
        offset: u32,
    ) {
        let attachments = self.out.attachments(desc::CLEAR);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            // Covers every texel and reads nothing from `out` — including the aux,
            // which a filter copies across from `back` rather than leaving to a load
            // op, since `out` is the other half of a ping-pong and holds a stale
            // bounce. So the load is a don't-care, and clearing states that rather
            // than implying the previous contents matter.
            color_attachments: &attachments[..self.out.count()],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bg, &[offset]);
        pass.draw(0..3, 0..1);
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
    /// **Two things can invalidate one, and the second is easy to miss.** The
    /// *textures* are this level's own or the accumulator's, and `ensure_targets`
    /// drops the whole scratch whenever the accumulator is rebuilt — so those are
    /// covered by the scratch's own lifetime. But the group also names the pass's
    /// **uniform buffer**, and a frame with more merges than any before it does not
    /// resize that buffer, it *replaces* it ([`UniformSlots::write`]) — leaving a kept
    /// bind group pointing at one too small for the offset it is about to be given.
    /// That is a validation error, not a wrong pixel, and no single-render test can
    /// reach it: a fresh compositor sizes its buffer before it builds anything over
    /// it. [`Compositor::upload`] calls [`ScratchTargets::invalidate_bind_groups`]
    /// when the buffer moves, which is the whole of the second half.
    ///
    /// [`Compositor::upload`]: super::Compositor
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

    /// Drop both caches, for the one thing they name that the scratch's own lifetime
    /// does not cover — see [`Self::blend_bg`].
    fn invalidate_bind_groups(&mut self) {
        self.blend_bg = Default::default();
        self.filter_bg = Default::default();
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
    /// Drop every level's cached bind groups, because the uniform buffer they name
    /// has been replaced — see [`ScratchLevel::blend_bg`]. Cheap: at most two of each
    /// per level, and only on the frame that grew the buffer.
    pub(super) fn invalidate_bind_groups(&mut self) {
        for level in &mut self.levels {
            level.invalidate_bind_groups();
        }
    }

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
