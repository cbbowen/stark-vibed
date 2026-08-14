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
//! [`UniformSlots`]: super::blend::UniformSlots

use crate::colorspace::ColorSpace;
use crate::gpu::context::GpuContext;
use crate::gpu::desc;

// Generated from `filter_common.wesl`'s own declaration (§6.10).
pub(super) use stark_shaders::mirror::filter_common::Filter as FilterUniform;

/// The filter pass: one fullscreen draw rewriting the accumulator.
pub(super) struct FilterPass {
    pub(super) pipeline: wgpu::RenderPipeline,
    pub(super) bgl: wgpu::BindGroupLayout,
    /// How the chromatic gather (§21.10) reads the accumulator *between* texels:
    /// bilinear, clamped to the edge — a tap displaced past the viewport reads the
    /// rim rather than wrapping the far side of the picture into a fringe. The
    /// point filters keep their exact `textureLoad`s and never touch it.
    pub(super) sampler: wgpu::Sampler,
}

impl FilterPass {
    pub(super) fn new(
        ctx: &GpuContext,
        color_space: &dyn ColorSpace,
        color_format: wgpu::TextureFormat,
        aux_format: wgpu::TextureFormat,
    ) -> Self {
        let device = &ctx.device;
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
            desc::uniform_slot(0, frag, std::mem::size_of::<FilterUniform>() as u64),
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
            bgl,
            sampler,
        }
    }
}
