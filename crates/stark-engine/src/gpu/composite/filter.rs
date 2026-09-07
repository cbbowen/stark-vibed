//! Filter layers: the pass that reads the accumulator and writes it back adjusted
//! (§21).
//!
//! **The blend pass with the source removed.** A filter layer has no content to
//! isolate — it is a function of what its stack has already composited — so where
//! [`blend`](super::blend) binds a backdrop *and* an isolated layer, this binds only
//! the backdrop. Everything else is shared: the ping-pong (a texture cannot be both
//! read and written), the [`ScratchLevel`] to bounce through, [`UniformSlots`] for a
//! slot per pass, and the "the pass computes the whole result" pipeline.
//!
//! [`ScratchLevel`]: super::blend::ScratchLevel
//! [`UniformSlots`]: crate::gpu::uniforms::UniformSlots

use crate::colorspace::ColorSpace;
use crate::gpu::channels::Targets;
use crate::gpu::context::GpuContext;
use crate::gpu::desc::{self, Slot};
use crate::gpu::uniforms::UniformSlots;
use crate::view::ViewTransform;
use stark_shaders::mirror::filter_common::binding as fc;
use stark_shaders::mirror::filter_common::decl as fcd;
use stark_shaders::mirror::filter_mixbox::binding as fm;
use stark_shaders::mirror::filter_mixbox::decl as fmd;
use stark_shaders::mirror::mixbox_lut::binding as ml;
use stark_shaders::mirror::mixbox_lut::decl as mld;

use super::blend::Bounce;
use super::group::FilterDraw;

// Generated from `filter_common.wesl`'s own declaration (§6.10).
pub(crate) use stark_shaders::mirror::filter_common::Filter as FilterUniform;

/// Which bindings the filter pass reads, in layout order (§6.10).
///
/// **The gap at 4 is deliberate.** The numbers are the blend pass's: this pass is
/// that shape with one input instead of two, so binding 3 carries the chromatic
/// gather's sampler, 4 stays undeclared, and everything after keeps the number it
/// has there.
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
    // The focal blur's convolved planes (§21.12). Loaded exactly, never sampled:
    // their `f32` formats are not filterable everywhere this runs, and the resolve
    // wants its own texel. A 1×1 zero stands in when the frame has no blur (§6.8).
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
/// runs this module's tile-space entry point to merge a filter layer into the paint
/// beneath it (§14.11.7), and a second copy would decode the Mixbox LUT twice.
///
/// [`BlendPass`]: super::blend::BlendPass
pub(crate) struct FilterPass {
    pub(crate) pipeline: wgpu::RenderPipeline,
    /// The **tile-space** entry point of the same module, on the same bind group
    /// layout: `fs_tile` reads a tile's stored channels where `fs_main` reads the
    /// accumulator's, and writes them back adjusted (§14.11.7).
    ///
    /// One layout serves both — a tile's three channel textures answer to
    /// `back_color` / `back_aux` / `back_resid` exactly as the accumulator's do. What
    /// differs is what the alpha lane *means* (per-unit opacity rather than
    /// coverage), which is the caller's fact rather than the binding's.
    pub(crate) tile: wgpu::RenderPipeline,
    /// The focal blur's decode (§21.12): the same module's `fs_blur_decode`, on the
    /// same layout, into `blur.wesl`'s two spatial-domain planes — the accumulator
    /// as premultiplied XYZ light plus coverage, height and the border weight.
    ///
    /// Here rather than in [`BlurPass`] because the decode is the per-color-space
    /// half of the blur; the FFT itself is arithmetic on light.
    ///
    /// [`BlurPass`]: super::blur::BlurPass
    pub(super) blur_decode: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    /// How the chromatic gather (§21.10) reads the accumulator *between* texels:
    /// bilinear, clamped to the edge — a tap displaced past the viewport reads the
    /// rim rather than wrapping the far side of the picture into a fringe. The point
    /// filters keep their exact `textureLoad`s and never touch it, and the tile pass
    /// binds it without reading it, a bind group having to satisfy the whole layout.
    sampler: wgpu::Sampler,
    /// What stands at the blur-plane slots when the frame has no blur — and in every
    /// merge, which refuses a resampling filter and so never reads them (§14.11.7).
    /// The §6.8 stand-in pattern.
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

    /// **The one description of `filter_common.wesl`'s group**, the screen's and the
    /// merge's alike: merging a filter layer into the paint beneath it runs this
    /// module's tile-space entry point (§14.11.7), so it must bind this very group
    /// rather than a second description of it.
    ///
    /// `pigment` is the **blend pass's** LUT, passed in rather than owned: both
    /// passes ask it the same question, so there is one table per color space rather
    /// than one per pass. `blur` is the frame's convolved planes when it has a focal
    /// blur (§21.12); the 1×1 zeroes stand in otherwise, so the layout is answered
    /// either way.
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

    /// Encode one filter layer: the accumulator `b.back` read and written back
    /// adjusted into `b.out`, through filter slot `b.slot` (§21.3).
    ///
    /// `blur` is this consumer's blur scratch whenever the frame has a focal blur
    /// anywhere, since the planes ride the bind group either way (§21.12).
    /// `convolve` is `Some` exactly when **this** layer is the blur; its kernel,
    /// decode and FFT round trip are then encoded ahead of the fullscreen pass.
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

/// The filter pass's uniform for `f`, under this frame's `view` (§21).
///
/// Here rather than on [`FilterDraw`] for the reason [`Plan::filters`](super::plan::Plan::filters) gives: one of
/// its lanes is a fact about the view, which the draw deliberately has none of.
pub(crate) fn filter_uniform(f: &FilterDraw, view: ViewTransform) -> FilterUniform {
    FilterUniform {
        kind: f.kind,
        strength: f.strength,
        clip: u32::from(f.clip),
        disp: view_lanes(f, view),
        params: f.params,
        params2: f.params2,
        // The gradient map's ramp, zeroed for every other kind — `disp`'s convention:
        // the true value, since no other kind has stops (§21.11).
        stops: f.stops.as_deref().copied().unwrap_or([[0.0; 4]; 16]),
        // The padding WGSL's alignment leaves around `clip`, which the generator
        // names and nothing reads (§6.10).
        ..Default::default()
    }
}

/// The uniform's view-derived lanes for this frame — `disp`, whose meaning is
/// the kind's to say (`filter_common.wesl`).
///
/// For the **chromatic** filter: the red-end → blue-end displacement, carried from
/// the canvas terms the document states (`params` = spread in canvas px, angle in
/// canvas radians) into the **accumulator texels** the pass samples in, through the
/// view's full linear map — zoom, rotation and mirror alike, so the fringes stay
/// attached to the artwork as the canvas substrate does (§21.10, §6.4).
///
/// For the **focal blur**: `.x` is the decimation scale the convolution runs at
/// (§21.12), by the same rule [`BlurPass::prepare`] plans the transform with, so
/// the decode, the transform and the resolve cannot disagree about it.
///
/// Zero for every other kind, which is the true value rather than a stand-in.
///
/// [`BlurPass::prepare`]: super::blur::BlurPass::prepare
/// [`Plan::filters`]: super::plan::Plan::filters
fn view_lanes(f: &FilterDraw, view: ViewTransform) -> [f32; 2] {
    if f.kind == stark_shaders::mirror::filter_common::FILTER_CHROMATIC {
        let (spread, angle) = (f.params[0], f.params[1]);
        let d = view.linear() * stark_model::geom::Vec2::new(angle.cos(), angle.sin()) * spread;
        return [d.x, d.y];
    }
    if f.kind == stark_shaders::mirror::filter_common::FILTER_FOCAL_BLUR {
        let s = super::blur::scale(super::blur::texel_radius(f, view));
        return [s as f32, 0.0];
    }
    [0.0; 2]
}
