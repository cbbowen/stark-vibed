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
use crate::gpu::context::GpuContext;
use crate::gpu::desc;
use crate::gpu::uniforms::UniformSlots;

use super::blend::Bounce;

// Generated from `filter_common.wesl`'s own declaration (§6.10).
pub(crate) use stark_shaders::mirror::filter_common::Filter as FilterUniform;

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
    pub(crate) bgl: wgpu::BindGroupLayout,
    /// How the chromatic gather (§21.10) reads the accumulator *between* texels:
    /// bilinear, clamped to the edge — a tap displaced past the viewport reads the
    /// rim rather than wrapping the far side of the picture into a fringe. The
    /// point filters keep their exact `textureLoad`s and never touch it.
    ///
    /// Bound by the tile pass too, which never reads it: a bind group has to satisfy
    /// the whole layout, and one layout for two pipelines is the trade that buys.
    pub(crate) sampler: wgpu::Sampler,
}

impl FilterPass {
    pub(crate) fn new(ctx: &GpuContext, color_space: &dyn ColorSpace) -> Self {
        let device = &ctx.device;
        let formats = crate::gpu::channels::ChannelFormats::of(color_space);
        let (color_format, aux_format) = (formats.color, formats.aux);
        let frag = wgpu::ShaderStages::FRAGMENT;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark filter"),
            source: wgpu::ShaderSource::Wgsl(color_space.filter_shader().into()),
        });
        let resid_format = color_space.resid_format();
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
        let mut entries = vec![
            UniformSlots::<FilterUniform>::layout(0, frag),
            desc::sample_tex(1, frag), // accumulator color
            desc::sample_tex(2, frag), // accumulator aux
            desc::sampler(3, frag),    // the chromatic gather's bilinear taps
            desc::sample_tex(5, frag), // pigment LUT (filtered)
            desc::sampler(6, frag),
        ];
        if let Some(f) = resid_format {
            debug_assert_eq!(
                f, color_format,
                "the filter pass loads the residual target with the color's decode",
            );
            entries.push(desc::sample_tex(7, frag)); // accumulator residual
        }
        let bgl = desc::bind_group_layout(device, "stark filter bgl", &entries);
        let layout = desc::pipeline_layout(device, "stark filter layout", &[Some(&bgl)]);
        // No fixed-function blend: the pass computes the whole texel — including the
        // height it copies straight across — and *replaces* what it writes. That is
        // what the ping-pong is for.
        let mut targets = vec![desc::target(color_format), desc::target(aux_format)];
        if let Some(f) = resid_format {
            targets.push(desc::target(f));
        }
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
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("stark filter sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            pipeline,
            tile,
            bgl,
            sampler,
        }
    }

    /// Encode one filter layer: the accumulator `b.back` read and written back
    /// adjusted into `b.out`, through filter slot `b.slot` (§21.3).
    ///
    /// `pigment` is the **blend pass's** LUT, passed in rather than owned. Both passes
    /// ask it the same question and an Oklab document binds the same 1×1 stand-in, so
    /// there is one table per space rather than one per pass — a coupling that now
    /// shows in this signature instead of only in a comment.
    pub(super) fn encode(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        b: Bounce<'_>,
        slots: &UniformSlots<FilterUniform>,
        pigment: &crate::gpu::pigment::PigmentLut,
    ) {
        let bg = b.here.filter_bg(b.phase.back_is_swap, || {
            let mut entries = vec![
                slots.binding(0),
                desc::tex(1, b.back.color),
                desc::tex(2, b.back.aux),
                desc::samp(3, &self.sampler),
                desc::tex(5, &pigment.view),
                desc::samp(6, &pigment.sampler),
            ];
            if let Some(r) = b.back.resid {
                entries.push(desc::tex(7, r));
            }
            desc::bind_group(&ctx.device, "stark filter bg", &self.bgl, &entries)
        });
        b.pass(
            encoder,
            "stark filter pass",
            &self.pipeline,
            bg,
            UniformSlots::<FilterUniform>::offset(b.slot),
        );
    }
}
