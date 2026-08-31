//! Filter layers: the pass that reads the accumulator and writes it back adjusted
//! (§21).
//!
//! **The blend pass with the source removed.** A filter layer has no content to
//! isolate — it is a function of what its stack has already composited — so where
//! [`blend`](super::blend) binds a backdrop *and* an isolated layer, this binds only
//! the backdrop. Everything else is shared: the same ping-pong (a texture cannot be
//! both read and written), the same [`ScratchLevel`] to bounce through, the same
//! [`UniformSlots`] mechanism for a slot per pass, and the same "no fixed-function
//! blend, the pass computes the whole result" pipeline.
//!
//! [`ScratchLevel`]: super::blend::ScratchLevel
//! [`UniformSlots`]: crate::gpu::uniforms::UniformSlots

use crate::colorspace::ColorSpace;
use crate::gpu::channels::Targets;
use crate::gpu::context::GpuContext;
use crate::gpu::desc::{self, Slot};
use crate::gpu::uniforms::UniformSlots;
use stark_shaders::mirror::filter_common::binding as fc;
use stark_shaders::mirror::filter_common::decl as fcd;
use stark_shaders::mirror::filter_mixbox::binding as fm;
use stark_shaders::mirror::filter_mixbox::decl as fmd;
use stark_shaders::mirror::mixbox_lut::binding as ml;
use stark_shaders::mirror::mixbox_lut::decl as mld;

use super::blend::Bounce;

// Generated from `filter_common.wesl`'s own declaration (§6.10).
pub(crate) use stark_shaders::mirror::filter_common::Filter as FilterUniform;

/// Which bindings the filter pass reads, in layout order (§6.10).
///
/// **The gap at 4 is the point of the list, not a gap in it.** This pass is the blend's
/// shape with one input instead of two, so binding 3 — one of the two the source layer
/// would have occupied — carries the chromatic gather's sampler instead and 4 stays
/// undeclared, while everything after keeps the number it has in the pass this one is a
/// narrowing of. A slot list says that by naming six declarations; an array indexed by
/// position could only say it in a comment.
///
/// The accumulator textures are declared **sampled** rather than loaded, because the
/// chromatic filter (§21.10) reads them through `back_samp` at fractional positions.
/// That asks their formats to be filterable, which `Rgba16Float`/`R16Float` are
/// everywhere this runs — including WebGPU's core feature set — and costs the point
/// filters nothing: a sampled declaration still serves their exact `textureLoad`s.
const FILTER_SLOTS: &[Slot] = &[
    Slot::dynamic(fcd::F),
    Slot::sampled(fcd::BACK_COLOR),
    Slot::sampled(fcd::BACK_AUX),
    Slot::at(fcd::BACK_SAMP),
    // The focal blur's convolved planes (§21.12), at the blend's other source
    // slot and past the space partition — `filter_common.wesl` says why those
    // numbers. Loaded exactly, never sampled: their `f32` formats are not
    // filterable everywhere this runs, and the resolve wants its own texel.
    // A 1×1 zero stands in whenever the frame has no blur (§6.8's pattern).
    Slot::at(fcd::BLUR_LIGHT),
    Slot::sampled(mld::PIGMENT_LUT),
    Slot::at(mld::PIGMENT_SAMP),
    // Sampled, unlike the blend's two: the gather reads the residual through the same
    // taps as the color it belongs to.
    Slot::sampled(fmd::BACK_RESID).only_with_resid(),
    Slot::at(fcd::BLUR_AUX),
];

/// The filter pass: one fullscreen draw rewriting the accumulator.
///
/// `pub(crate)` and shared behind an `Arc` for [`BlendPass`]'s reason: `gpu::merge`
/// runs this very module's other entry point on tile-sized targets to merge a filter
/// layer into the paint beneath it (§14.11.7), and a second copy would decode the
/// Mixbox LUT twice.
///
/// [`BlendPass`]: super::blend::BlendPass
pub(crate) struct FilterPass {
    pub(crate) pipeline: wgpu::RenderPipeline,
    /// The **tile-space** entry point of the same module, on the same bind group
    /// layout: `fs_tile` reads a tile's stored channels where `fs_main` reads the
    /// accumulator's, and writes them back adjusted (§14.11.7).
    ///
    /// One layout for both because they bind the same shapes in the same slots — a
    /// tile's three channel textures answer to `back_color` / `back_aux` /
    /// `back_resid` exactly as the accumulator's do, and the pigment LUT is the same
    /// LUT. What differs is what the alpha lane *means* (per-unit opacity rather than
    /// coverage), which is a fact about the caller rather than about the binding.
    pub(crate) tile: wgpu::RenderPipeline,
    /// The focal blur's decode (§21.12): the same module's `fs_blur_decode`, on the
    /// same layout, into `blur.wesl`'s two spatial-domain planes — the accumulator
    /// as premultiplied XYZ light plus coverage, height and the border weight.
    /// Here rather than in [`BlurPass`](super::blur::BlurPass) because the decode
    /// is the per-space half of the blur: the FFT is arithmetic on light, this is
    /// what makes a texel light at all.
    ///
    /// [`BlurPass`]: super::blur::BlurPass
    pub(super) blur_decode: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    /// How the chromatic gather (§21.10) reads the accumulator *between* texels:
    /// bilinear, clamped to the edge — a tap displaced past the viewport reads the
    /// rim rather than wrapping the far side of the picture into a fringe. The
    /// point filters keep their exact `textureLoad`s and never touch it.
    ///
    /// Bound by the tile pass too, which never reads it: a bind group has to satisfy
    /// the whole layout, and one layout for two pipelines is the trade that buys.
    sampler: wgpu::Sampler,
    /// What stands at the blur-plane slots when the frame has no blur — and in
    /// every merge, which refuses a resampling filter and so never reads them
    /// (§14.11.7). The §6.8 stand-in pattern; a bind group answers to the whole
    /// layout.
    blur_zero: (wgpu::TextureView, wgpu::TextureView),
}

impl FilterPass {
    pub(crate) fn new(ctx: &GpuContext, color_space: &dyn ColorSpace) -> Self {
        let device = &ctx.device;
        let formats = crate::gpu::channels::ChannelFormats::of(color_space);
        let frag = wgpu::ShaderStages::FRAGMENT;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark filter"),
            source: wgpu::ShaderSource::Wgsl(color_space.filter_shader().into()),
        });
        let resid_format = formats.resid;
        // The binding numbers are the blend pass's, gaps and all: `filter_common`
        // owns 0–3 where `blend_common` owns 0–4, and `mixbox_lut.wesl` hard-codes
        // the LUT at 5–6 for whoever imports it (see the note in that file). Slot 3
        // — one of the two the source layer would have occupied — carries the
        // chromatic gather's sampler instead, and 4 stays undeclared; everything
        // after keeps the number it has in the pass this one is a narrowing of.
        //
        // The accumulator textures are declared *sampled* rather than loaded,
        // because the chromatic filter (§21.10) reads them through `sampler` at
        // fractional positions. That asks their formats to be filterable, which
        // `Rgba16Float`/`R16Float` are everywhere this runs — including WebGPU's
        // core feature set — and costs the point filters nothing: a sampled
        // declaration still serves their exact `textureLoad`s.
        let bgl = desc::layout_for(
            device,
            "stark filter bgl",
            FILTER_SLOTS,
            frag,
            resid_format.is_some(),
        );
        let layout = desc::pipeline_layout(device, "stark filter layout", &[Some(&bgl)]);
        // No fixed-function blend: the pass computes the whole texel — including the
        // height it copies straight across — and *replaces* what it writes. That is
        // what the ping-pong is for.
        let targets = formats.targets();
        let pipeline = desc::fullscreen_pipeline(
            device,
            "stark filter pipeline",
            &layout,
            &shader,
            ("vs_main", "fs_main"),
            &targets,
        );
        // The same module, the same layout, the same targets — a tile's channel
        // textures carry the color space's own formats, which is what makes the merge
        // able to borrow this pass rather than restate its algebra (§14.11.7).
        let tile = desc::fullscreen_pipeline(
            device,
            "stark filter tile pipeline",
            &layout,
            &shader,
            ("vs_main", "fs_tile"),
            &targets,
        );
        // The blur decode's targets are the FFT planes' own formats, read off the
        // shader that will transform them (§6.10) rather than restated.
        use stark_shaders::mirror::blur::decl as bld;
        let blur_targets = [
            desc::target(bld::DST_LIGHT.storage_format()),
            desc::target(bld::DST_AUX.storage_format()),
        ];
        let blur_decode = desc::fullscreen_pipeline(
            device,
            "stark filter blur decode pipeline",
            &layout,
            &shader,
            ("vs_main", "fs_blur_decode"),
            &blur_targets,
        );
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("stark filter sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let blur_zero = (
            desc::zero_texture(ctx, bld::DST_LIGHT.storage_format(), "stark blur light 1x1"),
            desc::zero_texture(ctx, bld::DST_AUX.storage_format(), "stark blur aux 1x1"),
        );
        Self {
            pipeline,
            tile,
            blur_decode,
            bgl,
            sampler,
            blur_zero,
        }
    }

    /// Encode one filter layer: the accumulator `b.back` read and written back
    /// adjusted into `b.out`, through filter slot `b.slot` (§21.3).
    ///
    /// `pigment` is the **blend pass's** LUT, passed in rather than owned. Both passes
    /// ask it the same question and an Oklab document binds the same 1×1 stand-in, so
    /// there is one table per space rather than one per pass — a coupling that now
    /// shows in this signature instead of only in a comment.
    /// **The one description of `filter_common.wesl`'s group** — the screen's and the
    /// merge's alike, on [`BlendPass::bind_group`](super::blend::BlendPass::bind_group)'s
    /// argument: merging a filter layer into the paint beneath it runs this module's
    /// tile-space entry point (§14.11.7), so the merged tile comes out of the shader
    /// the screen runs and must bind the very group rather than a second description
    /// of it.
    ///
    /// The sampler at `BACK_SAMP` is bound by the tile pass and never read — `fs_tile`
    /// takes no taps — because a bind group answers to the whole layout.
    /// `blur` is the frame's convolved planes when it has a focal blur (§21.12),
    /// and absent otherwise — the 1×1 zeroes stand in, so the layout is answered
    /// either way and a document without a blur pays two stand-in bindings rather
    /// than a variant of this group.
    pub(crate) fn bind_group(
        &self,
        device: &wgpu::Device,
        uniform: wgpu::BindingResource<'_>,
        back: Targets<'_>,
        pigment: &crate::gpu::pigment::PigmentLut,
        blur: Option<(&wgpu::TextureView, &wgpu::TextureView)>,
    ) -> wgpu::BindGroup {
        let (blur_light, blur_aux) = blur.unwrap_or((&self.blur_zero.0, &self.blur_zero.1));
        desc::bind_group_for(
            device,
            "stark filter bg",
            &self.bgl,
            FILTER_SLOTS,
            back.resid.is_some(),
            |i| match i {
                fc::F => uniform.clone(),
                fc::BACK_COLOR => wgpu::BindingResource::TextureView(back.color),
                fc::BACK_AUX => wgpu::BindingResource::TextureView(back.aux),
                fc::BACK_SAMP => wgpu::BindingResource::Sampler(&self.sampler),
                fc::BLUR_LIGHT => wgpu::BindingResource::TextureView(blur_light),
                fc::BLUR_AUX => wgpu::BindingResource::TextureView(blur_aux),
                ml::PIGMENT_LUT => wgpu::BindingResource::TextureView(&pigment.view),
                ml::PIGMENT_SAMP => wgpu::BindingResource::Sampler(&pigment.sampler),
                fm::BACK_RESID => wgpu::BindingResource::TextureView(
                    back.resid.expect("a residual build has one"),
                ),
                other => unreachable!("`FILTER_SLOTS` lists no binding {other}"),
            },
        )
    }

    /// `blur` is this consumer's blur scratch when the frame has a focal blur
    /// anywhere — the planes it lands in ride the bind group either way, so a
    /// point filter beside a blur reads the same group (§21.12). `convolve` is
    /// `Some` exactly when **this** layer is the blur: the frame's kernel,
    /// decode and FFT round trip are encoded ahead of the fullscreen pass, whose
    /// `FILTER_FOCAL_BLUR` arm then reads what they landed.
    #[expect(
        clippy::too_many_arguments,
        reason = "every argument is a distinct piece of what one filter pass names"
    )]
    pub(super) fn encode(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        b: Bounce<'_>,
        slots: &UniformSlots<FilterUniform>,
        pigment: &crate::gpu::pigment::PigmentLut,
        blur: Option<&super::blur::BlurFrame>,
        convolve: Option<&super::blur::BlurPass>,
    ) {
        let bg = b.here.filter_bg(b.phase.back_is_swap, || {
            self.bind_group(
                &ctx.device,
                slots.resource(),
                b.back,
                pigment,
                blur.map(|f| f.planes()),
            )
        });
        let offset = UniformSlots::<FilterUniform>::offset(b.slot);
        if let Some(pass) = convolve {
            // `b.slot` is this layer's dense filter slot (§14.7's one walk), which
            // is also how the frame's jobs are keyed — one index, two lists.
            blur.expect("a focal blur's frame is prepared before anything encodes")
                .encode(pass, self, encoder, bg, offset, b.slot);
        }
        b.pass(encoder, "stark filter pass", &self.pipeline, bg, offset);
    }
}
