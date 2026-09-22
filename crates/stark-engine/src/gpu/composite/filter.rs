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
use crate::gpu::desc::{self, Bindings};
use crate::gpu::pigment::PigmentLut;
use crate::gpu::uniforms::UniformSlots;
use crate::view::ViewTransform;
use stark_shaders::Stages;
use stark_shaders::mirror::filter::binding as fm;
use stark_shaders::mirror::filter::decl as fmd;
use stark_shaders::mirror::filter_common::binding as fc;
use stark_shaders::mirror::filter_common::decl as fcd;

use super::blend::Bounce;
use super::blur::{self, Aperture};
use super::group::{FilterDraw, FilterOp};
use stark_model::geom::Vec2;

// Generated from `filter_common.wesl`'s own declaration (§6.10).
pub(crate) use stark_shaders::mirror::filter_common::Filter as FilterUniform;

/// The filter pass: one fullscreen draw rewriting the accumulator.
///
/// `pub(crate)` and shared behind an `Arc` for [`BlendPass`]'s reason: `gpu::merge`
/// runs this module's tile-space entry point to merge a filter layer into the paint
/// beneath it (§14.11.7), so a merged tile comes out of the pipeline the screen runs.
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
    /// `filter_common`'s whole `@group(0)`, then the space's own `@group(1)` — the
    /// LUT and the backdrop's residual, or nothing at all in a colorimetric space
    /// (§6.7).
    bgls: [Bindings; 2],
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
    /// Mixbox's LUT, present exactly where this pass's layout holds a slot for it
    /// (§6.7) — the blend pass's table, since both passes ask it the same question.
    pigment: Option<PigmentLut>,
}

impl FilterPass {
    /// The pass for `color_space`, binding `pigment` — the blend pass's LUT
    /// ([`BlendPass::pigment`](super::blend::BlendPass::pigment)), so a space decodes
    /// one table rather than one per pass.
    ///
    /// # Panics
    /// If `pigment` is present where this pass's shader reads no LUT, or absent where
    /// it reads one: a space links both Mixbox shaders or neither (§6.7), and a build
    /// that broke that would leave a layout slot with nothing to fill it.
    pub(crate) fn new(
        ctx: &GpuContext,
        color_space: &dyn ColorSpace,
        pigment: Option<PigmentLut>,
    ) -> Self {
        let device = &ctx.device;
        let formats = crate::gpu::channels::ChannelFormats::of(color_space);
        let filter = color_space.filter_shader();
        let shader = desc::Module::new(device, "stark filter", filter);
        // One layout for all three fragment entry points below, so the accumulator's
        // textures come out **filterable**: the chromatic gather (§21.10) reads them
        // through `back_samp` at fractional positions where the point filters
        // `textureLoad` them, and one filterable declaration serves both. The blur's
        // convolved planes stay unfilterable — their `f32` formats are not filterable
        // everywhere this runs, and nothing samples them (§21.12).
        let stages = [
            Stages::Render(filter.vs_main, filter.fs_main),
            Stages::Render(filter.vs_main, filter.fs_tile),
            Stages::Render(filter.vs_main, filter.fs_blur_decode),
        ];
        // A group per module (§6.10): `filter_common`'s, then the space's own, which is
        // empty in a colorimetric one.
        let bgls = [
            Bindings::shared_by(device, "stark filter bgl", &stages, fcd::F, &[fcd::F]),
            Bindings::if_reached(device, "stark filter space bgl", &stages, fmd::PIGMENT_LUT),
        ];
        // Every entry point the layout was built over, not `fs_main` alone: the bind
        // group answers for the whole layout, so a LUT only one of them reached is
        // still a slot to fill.
        let reads_lut = stages.iter().any(|s| match *s {
            Stages::Render(vs, fs) => vs.reads(fmd::PIGMENT_LUT) || fs.reads(fmd::PIGMENT_LUT),
            Stages::Compute(k) => k.reads(fmd::PIGMENT_LUT),
        });
        assert_eq!(
            pigment.is_some(),
            reads_lut,
            "the filter pass is given a pigment LUT exactly where its shader reads one",
        );
        let layout = desc::pipeline_layout_of(device, "stark filter layout", &[&bgls[0], &bgls[1]]);
        // No fixed-function blend: the pass computes the whole texel — including the
        // height it copies straight across — and *replaces* what it writes. That is
        // what the ping-pong is for.
        let targets = formats.targets();
        let pipeline = desc::fullscreen_pipeline(
            device,
            "stark filter pipeline",
            &layout,
            &shader,
            (filter.vs_main, filter.fs_main),
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
            (filter.vs_main, filter.fs_tile),
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
            (filter.vs_main, filter.fs_blur_decode),
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
            bgls,
            sampler,
            blur_zero,
            pigment,
        }
    }

    /// **The one description of `filter_common.wesl`'s group**, the screen's and the
    /// merge's alike: merging a filter layer into the paint beneath it runs this
    /// module's tile-space entry point (§14.11.7), so it must bind this very group
    /// rather than a second description of it.
    ///
    /// `blur` is the frame's convolved planes when it has a focal blur (§21.12); the
    /// 1×1 zeroes stand in otherwise, so the layout is answered either way.
    pub(crate) fn bind_group(
        &self,
        device: &wgpu::Device,
        uniform: wgpu::BindingResource<'_>,
        back: Targets<'_>,
        blur: Option<(&wgpu::TextureView, &wgpu::TextureView)>,
    ) -> [wgpu::BindGroup; 2] {
        let (blur_light, blur_aux) = blur.unwrap_or((&self.blur_zero.0, &self.blur_zero.1));
        let lut = || {
            self.pigment
                .as_ref()
                .expect("the layout holds the LUT only where the shader reads it")
        };
        [
            self.bgls[0].group(device, "stark filter bg", |i| match i {
                fc::F => uniform.clone(),
                fc::BACK_COLOR => wgpu::BindingResource::TextureView(back.color),
                fc::BACK_AUX => wgpu::BindingResource::TextureView(back.aux),
                fc::BACK_SAMP => wgpu::BindingResource::Sampler(&self.sampler),
                fc::BLUR_LIGHT => wgpu::BindingResource::TextureView(blur_light),
                fc::BLUR_AUX => wgpu::BindingResource::TextureView(blur_aux),
                other => unreachable!("`filter_common`'s group has no binding {other}"),
            }),
            // Empty in a colorimetric space, where the closure is asked for nothing.
            self.bgls[1].group(device, "stark filter space bg", |i| match i {
                fm::PIGMENT_LUT => wgpu::BindingResource::TextureView(&lut().view),
                fm::PIGMENT_SAMP => wgpu::BindingResource::Sampler(&lut().sampler),
                fm::BACK_RESID => wgpu::BindingResource::TextureView(
                    back.resid.expect("a residual build has one"),
                ),
                other => unreachable!("`filter.wesl`'s own group has no binding {other}"),
            }),
        ]
    }

    /// Encode one filter layer: the accumulator `b.back` read and written back
    /// adjusted into `b.out`, through filter slot `b.slot` (§21.3).
    ///
    /// `blur` is this consumer's blur scratch whenever the frame has a focal blur
    /// anywhere, since the planes ride the bind group either way (§21.12).
    /// `convolve` is `Some` exactly when **this** layer is the blur; its kernel,
    /// decode and FFT round trip are then encoded ahead of the fullscreen pass.
    pub(super) fn encode(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        b: Bounce<'_>,
        slots: &UniformSlots<FilterUniform>,
        blur: Option<&super::blur::BlurFrame>,
        convolve: Option<&super::blur::BlurPass>,
    ) {
        let bg = b.here.filter_bg(b.phase.back_is_swap, || {
            self.bind_group(
                &ctx.device,
                slots.resource(),
                b.back,
                blur.map(|f| f.planes()),
            )
        });
        let offsets = [UniformSlots::<FilterUniform>::offset(b.slot)];
        let bound = desc::Bound::new(bg, &offsets);
        if let Some(pass) = convolve {
            // `b.slot` is this layer's dense filter slot (§14.7's one walk), which
            // is also how the frame's jobs are keyed — one index, two lists.
            blur.expect("a focal blur's frame is prepared before anything encodes")
                .encode(pass, self, encoder, bound, b.slot);
        }
        // A clear, like every bounce: the aux too is written whole, copied across
        // from `b.back` by the shader rather than left to a load op.
        desc::fullscreen_pass(encoder, "stark filter pass", b.out, &self.pipeline, bound);
    }
}

/// The filter pass's uniform for `f`, under this frame's `view` (§21) — **the one
/// place a [`FilterOp`] becomes numbers**: its `FILTER_*` code, mirrored from
/// `filter_common.wesl` (§6.10) rather than written as a literal, and the lanes that
/// code reads there. Every lane an arm leaves out is zero, which is the true value
/// rather than a stand-in.
///
/// Here rather than on [`FilterDraw`] for the reason
/// [`Plan::filters`](super::plan::Plan::filters) gives: `disp` is a fact about the
/// view, which the draw deliberately has none of.
pub(crate) fn filter_uniform(f: &FilterDraw, view: ViewTransform) -> FilterUniform {
    use stark_shaders::mirror::filter_common as fc;
    let base = FilterUniform {
        strength: f.strength,
        clip: u32::from(f.clip),
        // The padding WGSL's alignment leaves around `clip`, which the generator
        // names and nothing reads (§6.10).
        ..Default::default()
    };
    match &f.op {
        FilterOp::Color(c) => FilterUniform {
            kind: fc::FILTER_COLOR,
            params: [c.exposure, c.contrast, c.saturation, c.hue],
            params2: [c.tint[0], c.tint[1], 0.0, 0.0],
            ..base
        },
        // `disp` is the red-end → blue-end displacement, carried from the canvas terms
        // the document states into the **accumulator texels** the pass samples in,
        // through the view's full linear map — zoom, rotation and mirror alike, so the
        // fringes stay attached to the artwork as the canvas substrate does (§21.10,
        // §6.4).
        FilterOp::Chromatic(c) => {
            let d = view.linear() * Vec2::new(c.angle.cos(), c.angle.sin()) * c.spread;
            FilterUniform {
                kind: fc::FILTER_CHROMATIC,
                disp: [d.x, d.y],
                params: [c.spread, c.angle, 0.0, 0.0],
                ..base
            }
        }
        // `disp.x` is the decimation scale the convolution runs at (§21.12), by the
        // rule `BlurPass::prepare` plans the transform with, so the decode, the
        // transform and the resolve cannot disagree about it. The aperture's lanes
        // are read by no fullscreen arm — the convolution has run by then — and are
        // carried so the uniform states the document's numbers (`filter_common.wesl`).
        FilterOp::FocalBlur(b) => {
            let [shape, param, angle] = Aperture::of(b.aperture).lanes();
            FilterUniform {
                kind: fc::FILTER_FOCAL_BLUR,
                disp: [blur::scale(blur::texel_radius(b, view)) as f32, 0.0],
                params: [b.radius, shape, param, angle],
                ..base
            }
        }
        FilterOp::GradientMap { stops, count } => FilterUniform {
            kind: fc::FILTER_GRADIENT_MAP,
            params: [*count as f32, 0.0, 0.0, 0.0],
            stops: **stops,
            ..base
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::CompositeParams;
    use crate::view::Extent2;
    use stark_model::document::Filter;

    /// Every filter the document can hold reaches the shader as **its own** code — a
    /// code copied from the arm above renders one filter as another, which no
    /// assertion about a single arm can see.
    #[test]
    fn every_filter_is_its_own_code() {
        let view = ViewTransform::identity(Extent2::new(64, 64));
        let mut codes: Vec<u32> = Filter::ALL
            .into_iter()
            .map(|f| filter_uniform(&FilterDraw::new(f, CompositeParams::IDENTITY), view).kind)
            .collect();
        let offered = codes.len();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), offered, "two filters share one code");
    }
}
