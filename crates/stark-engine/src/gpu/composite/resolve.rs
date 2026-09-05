//! Pass E: the presentation resolve — the supersampled render box-averaged down to
//! the caller's target (§6.4).
//!
//! Encoded only when the view is zoomed out. At 1:1 the passes above write the
//! target directly and this is never bound, so painting at 100% costs exactly
//! what it always did.

use super::display::Transfer;
use crate::gpu::context::GpuContext;
use crate::gpu::desc;
use crate::gpu::desc::Slot;
use crate::view::Extent2;
use stark_shaders::mirror::resolve::binding as rb;
use stark_shaders::mirror::resolve::decl as rd;

/// The presentation resolve's bindings (§6.4): the box filter's extent, and the
/// supersampled render it averages down.
const RESOLVE_SLOTS: &[Slot] = &[Slot::at(rd::R), Slot::at(rd::SRC)];

// Generated from `resolve.wesl`'s own declaration (§6.7).
pub(super) use stark_shaders::mirror::resolve::Resolve as ResolveUniform;

/// Most samples per axis presentation will take (§6.4). A 4×4 box is 16
/// taps per output pixel; past that the box filter's remaining error is dominated by
/// its own shape rather than by the sample count, and a view zoomed further out than
/// 1:4 is a thumbnail whose next improvement is a better filter, not more of this one.
const MAX_SUPERSAMPLE: u32 = 4;

/// Most GPU memory a render may hold in viewport-sized attachments, whatever the
/// zoom asks for.
///
/// The ceiling exists because *every* offscreen attachment scales with the sample
/// count: pass A's color, aux and — in a pigment space — residual, the target the
/// resolve reads, and **the blend scratch, two trios per level of nesting**, if the
/// document has a mode or a filter in it. Only taken while the view is actually
/// zoomed out, since [`supersample`] returns 1 at 1:1 and the attachments shrink
/// back. Crossing a threshold reallocates all of them, so this is also a bound on the
/// hitch a wheel-zoom can cost.
///
/// **It is a byte budget, not a pixel count.** No pixel count can enumerate the
/// scratch, whose size is a fact about the *document*, nor the residual, which is a
/// fact about the color space. At 16 Mpx the real figures run 234 MB for a flat Oklab
/// document, 571 MB with one blend group, 973 MB for the same in pigment, and ~1.6 GB
/// at two levels of nesting — a spread of seven that a megapixel ceiling cannot see.
/// So the number is bytes, and what varies is how many pixels of *this* document's
/// attachments fit inside it.
///
/// The flat Oklab case is the calibration point: 14 bytes a texel (8 + 2 accumulator,
/// 4 resolve target) into 224 MiB is 16 Mpx exactly.
///
/// It binds on the window and not on a miniature, which is the right way round: the
/// navigator renders a whole piece into ~250 px, is the worst-aliased view in the
/// application, and reaches [`MAX_SUPERSAMPLE`] for a few megapixels. What it costs
/// is the common zoom-outs on a large window, where the picture is already most of
/// the way back and the fourth sample buys least.
const MAX_SUPERSAMPLED_BYTES: u64 = 224 << 20;

/// How many samples per axis a render of `size` at `zoom` takes (§6.4).
///
/// `1` at 1:1 and closer — a view that magnifies is already oversampling, so
/// painting at 100% costs exactly what it always did and a golden blessed at
/// `zoom = 1.0` never sees this pass. Below that it is the minification ratio, so each
/// output pixel gets back roughly one sample per canvas pixel it covers, capped by
/// [`MAX_SUPERSAMPLE`], by [`MAX_SUPERSAMPLED_BYTES`] and by what the device will
/// allocate.
///
/// `bytes_per_px` is what *one supersampled texel of this frame* costs across every
/// attachment that scales with the sample count — see [`attachment_bytes`]. It is the
/// caller's to compute because only the caller knows the document: the color space
/// decides whether there is a residual, and the group tree decides how many scratch
/// levels there are.
pub(super) fn supersample(
    size: Extent2,
    zoom: f32,
    limits: &wgpu::Limits,
    bytes_per_px: u64,
) -> u32 {
    if !(zoom.is_finite() && zoom > 0.0) {
        return 1;
    }
    let (w, h) = (size.width.max(1), size.height.max(1));
    let want = (1.0 / zoom).ceil();
    // `as` saturates, so a zoom small enough to overflow lands on the cap rather
    // than wrapping to something tiny.
    let want = (want as u32).clamp(1, MAX_SUPERSAMPLE);
    (1..=want)
        .rev()
        .find(|n| {
            let (sw, sh) = (w.saturating_mul(*n), h.saturating_mul(*n));
            let bytes = u64::from(sw)
                .saturating_mul(u64::from(sh))
                .saturating_mul(bytes_per_px);
            sw <= limits.max_texture_dimension_2d
                && sh <= limits.max_texture_dimension_2d
                && bytes <= MAX_SUPERSAMPLED_BYTES
        })
        .unwrap_or(1)
}

/// What one supersampled texel costs this frame, across every attachment that scales
/// with the sample count (§6.4) — the input [`supersample`] measures its budget in.
///
/// - the **accumulator** trio, always;
/// - the **resolve target**, which only exists above 1× but is what the whole
///   question is about, so it is counted unconditionally;
/// - **two trios per scratch level** that isolates and one per level that only
///   ping-pongs, which is [`Plan::scratch`](super::plan::Plan::scratch) read as
///   memory (§18.0.4, §21.3);
/// - the **focal blur's FFT planes** when the frame has one (§21.12), which dwarf
///   everything above.
///
/// The last two are the terms a pixel count cannot express, and they dominate: a
/// single blend group more than doubles the frame's memory footprint, and a focal
/// blur is [`blur_bytes_per_px`] on its own — which is the mechanism that keeps a
/// blurred frame from also being a heavily supersampled one.
pub(super) fn attachment_bytes(
    formats: crate::gpu::channels::ChannelFormats,
    target_format: wgpu::TextureFormat,
    scratch: &[bool],
    blur: bool,
) -> u64 {
    let trio = formats.bytes_per_px();
    let resolve = u64::from(target_format.block_copy_size(None).unwrap_or(0));
    let scratch: u64 = scratch.iter().map(|&iso| trio * (1 + u64::from(iso))).sum();
    trio + resolve + scratch + if blur { blur_bytes_per_px() } else { 0 }
}

/// What one supersampled texel costs the focal blur (§21.12), as this budget
/// counts it: the five `f32` planes — two ping-pong sets of the light and aux
/// planes, and the kernel, which `blur.rs` allocates in the aux plane's format —
/// per **padded** texel, and the power-of-two padding runs up to about double the
/// accumulator's area before the guard band. An estimate rather than the exact
/// figure, because the padding depends on the radius and the radius on the zoom
/// this call is deciding — the pessimistic side of the round-up is the safe side
/// of a memory budget.
///
/// Read off the shader's own storage formats rather than summed by hand (§6.10):
/// `block_copy_size` is not `const`, and a plane whose format moved in
/// `blur.wesl` would otherwise leave this budget counting the old one.
fn blur_bytes_per_px() -> u64 {
    use stark_shaders::mirror::blur::decl as bd;
    let of = |f: wgpu::TextureFormat| u64::from(f.block_copy_size(None).unwrap_or(0));
    let (light, aux) = (
        of(bd::DST_LIGHT.storage_format()),
        of(bd::DST_AUX.storage_format()),
    );
    2 * (2 * (light + aux) + aux)
}

/// The resolve pass's bind group layout, shared by the pipeline compiled for each
/// target format ([`TargetPasses`](super::TargetPasses)) so a [`Supersampled`]
/// set's bind group is valid against either.
pub(super) fn resolve_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    desc::layout_for(
        device,
        "stark resolve bgl",
        RESOLVE_SLOTS,
        wgpu::ShaderStages::FRAGMENT,
        false,
    )
}

/// The resolve pass — the pipeline, compiled for one target format. The uniform it
/// reads (`n`, the sample count) is the *rendering* consumer's: `n` is a function of
/// that target's zoom, so the substrate and a miniature beside it disagree about it
/// by construction. It rides in the [`Supersampled`] set, which is exactly the state
/// that exists only while a view is zoomed out.
pub(super) struct ResolvePass {
    pipeline: wgpu::RenderPipeline,
}

impl ResolvePass {
    pub(super) fn new(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        target: &[Option<wgpu::ColorTargetState>],
    ) -> Self {
        // A fullscreen pass reading the supersampled target with `textureLoad` at an
        // integer block of its own choosing, so nothing here needs a sampler.
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark resolve"),
            source: wgpu::ShaderSource::Wgsl(stark_shaders::resolve().into()),
        });
        let layout = desc::pipeline_layout(device, "stark resolve layout", &[Some(bgl)]);
        // The pass covers every texel and carries the alpha it averaged, so there is
        // nothing for a fixed-function blend to do.
        let pipeline = desc::fullscreen_pipeline(
            device,
            "stark resolve pipeline",
            &layout,
            &shader,
            ("vs_main", "fs_main"),
            target,
        );
        Self { pipeline }
    }

    /// Encode pass E: everything the passes above drew, box-averaged in light down to
    /// the caller's `target` (§6.4).
    ///
    /// The caller has already skipped this at 1:1, where there is no [`Supersampled`]
    /// set at all and the passes above wrote `target` directly.
    pub(super) fn encode(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        ss: &Supersampled,
        scene: ResolveScene,
        target: &wgpu::TextureView,
    ) {
        ctx.queue.write_buffer(
            &ss.buf,
            0,
            bytemuck::bytes_of(&ResolveUniform {
                n: [
                    scene.n as f32,
                    scene.dither_step,
                    scene.transfer.lane(),
                    0.0,
                ],
            }),
        );
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("stark resolve pass"),
            // Covers every texel and reads nothing back, so the load is a don't-care;
            // clearing says so rather than implying otherwise.
            color_attachments: &[Some(desc::attach(target, desc::CLEAR))],
            ..Default::default()
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &ss.bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// What one resolve needs beyond the supersampled set (§6.4).
pub(super) struct ResolveScene {
    /// Samples per axis.
    pub(super) n: u32,
    /// One code of the target's encoding, or 0 (`media::dither_step`, §6.5).
    pub(super) dither_step: f32,
    /// The transfer the media pass wrote the supersampled texels in, so the decode
    /// and the encode around the average are a pair (§6.5).
    pub(super) transfer: Transfer,
}

/// Where passes B–D write when the view is zoomed out, and what pass E reads it back
/// through (§6.4).
///
/// One value rather than three because they are built and dropped together and none
/// of them means anything without the others: the bind group names the view, and the
/// uniform holds the very `n` that decided the view's size. Absent entirely at 1:1,
/// which is what returns the memory the moment the artist zooms back in to paint.
pub(super) struct Supersampled {
    /// What passes B–D draw into, `n` times the caller's target on each axis.
    target: super::Attachment,
    bg: wgpu::BindGroup,
    /// `n` itself. This consumer's, because the sample count is a function of *this*
    /// target's zoom — see [`ResolvePass`].
    buf: wgpu::Buffer,
}

impl Supersampled {
    pub(super) fn new(
        device: &wgpu::Device,
        size: Extent2,
        format: wgpu::TextureFormat,
        bgl: &wgpu::BindGroupLayout,
    ) -> Self {
        let target = super::Attachment::new(device, size, format, "stark supersampled");
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stark resolve uniform"),
            size: std::mem::size_of::<ResolveUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg =
            desc::bind_group_for(
                device,
                "stark resolve bg",
                bgl,
                RESOLVE_SLOTS,
                false,
                |i| match i {
                    rb::R => buf.as_entire_binding(),
                    rb::SRC => wgpu::BindingResource::TextureView(target.view()),
                    other => unreachable!("`RESOLVE_SLOTS` lists no binding {other}"),
                },
            );
        Self { target, bg, buf }
    }

    /// What passes B–D attach — the caller's target, supersampled.
    pub(super) fn view(&self) -> &wgpu::TextureView {
        self.target.view()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::channels::ChannelFormats;

    /// The device this policy is written against: the `downlevel_webgl2` floor's
    /// 8192-px limit, which is what the wasm build can count on.
    fn limits() -> wgpu::Limits {
        wgpu::Limits {
            max_texture_dimension_2d: 8192,
            ..wgpu::Limits::default()
        }
    }

    const TARGET: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;

    fn oklab() -> ChannelFormats {
        ChannelFormats {
            color: wgpu::TextureFormat::Rgba16Float,
            aux: wgpu::TextureFormat::R16Float,
            resid: None,
        }
    }

    fn pigment() -> ChannelFormats {
        ChannelFormats {
            resid: Some(wgpu::TextureFormat::Rgba16Float),
            ..oklab()
        }
    }

    /// A flat document: the accumulator and the resolve target, no scratch.
    fn flat() -> u64 {
        attachment_bytes(oklab(), TARGET, &[], false)
    }

    #[test]
    fn magnification_costs_nothing() {
        let window = Extent2::new(1280, 720);
        for zoom in [1.0, 1.5, 4.0, 64.0] {
            assert_eq!(
                supersample(window, zoom, &limits(), flat()),
                1,
                "at {zoom}×"
            );
        }
        // A degenerate zoom asks for nothing rather than for a division by it.
        for zoom in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert_eq!(
                supersample(window, zoom, &limits(), flat()),
                1,
                "at {zoom}×"
            );
        }
    }

    #[test]
    fn minification_is_matched_sample_for_canvas_pixel() {
        // Small enough that no cap binds, so this is the rule itself.
        let thumb = Extent2::new(252, 176);
        assert_eq!(supersample(thumb, 0.5, &limits(), flat()), 2);
        assert_eq!(supersample(thumb, 0.25, &limits(), flat()), 4);
        // Past the cap the ratio keeps growing and the answer does not.
        assert_eq!(supersample(thumb, 0.01, &limits(), flat()), MAX_SUPERSAMPLE);
    }

    #[test]
    fn a_large_target_gives_up_samples_rather_than_the_render() {
        // The memory budget binds first at window sizes: 4× of this is 176 Mpx.
        let window = Extent2::new(2560, 1440);
        let n = supersample(window, 0.1, &limits(), flat());
        assert!(n > 1, "a zoomed-out window should still supersample");
        let px = u64::from(window.width * n) * u64::from(window.height * n);
        assert!(
            px * flat() <= MAX_SUPERSAMPLED_BYTES,
            "{n}× of {window:?} is over the memory budget",
        );
        // And the device limit binds before either of the others on a wide one.
        let wide = Extent2::new(7000, 400);
        assert_eq!(supersample(wide, 0.1, &limits(), flat()), 1);
    }

    /// The blur term joins the scratch as a byte cost a pixel count cannot see
    /// (§21.12), and it is the largest single one: a frame with a focal blur in
    /// it gives up supersampling long before a flat frame does, which is the
    /// budget doing exactly what it is for — the FFT planes and a heavy sample
    /// count are not affordable together.
    #[test]
    fn a_focal_blur_gives_up_samples_before_the_planes_overrun() {
        assert_eq!(
            attachment_bytes(oklab(), TARGET, &[], true),
            14 + 112,
            "the five f32 planes at the padding estimate",
        );
        let window = Extent2::new(2560, 1440);
        let at = |blur| {
            supersample(
                window,
                0.1,
                &limits(),
                attachment_bytes(oklab(), TARGET, &[], blur),
            )
        };
        assert!(at(false) > at(true), "a blurred frame must give up samples");
        assert_eq!(at(true), 1);
    }

    /// The byte budget's calibration point: a flat Oklab document sees a 16-Mpx
    /// ceiling, which is what the common zoom-out is sized against. 14 bytes a texel
    /// into 224 MiB is 16 Mpx on the nose.
    #[test]
    fn a_flat_oklab_frame_still_stops_at_sixteen_megapixels() {
        assert_eq!(flat(), 14, "8 + 2 accumulator, 4 resolve target");
        assert_eq!(MAX_SUPERSAMPLED_BYTES / flat(), 16 << 20);
    }

    /// The term a pixel count cannot express, and the reason the budget is in bytes:
    /// **the scratch dominates**. One blend group more than doubles the frame — a
    /// `swap` trio and an `iso` trio, each the size of the accumulator — and a pigment
    /// document pays 18 bytes a texel for every one of them.
    ///
    /// A flat 16-Mpx ceiling would let all four of these render at that size, which is
    /// 973 MB for the third and about 1.6 GB for the fourth.
    #[test]
    fn the_blend_scratch_is_most_of_what_a_nested_frame_costs() {
        let cost = |f, s: &[bool]| attachment_bytes(f, TARGET, s, false);
        assert_eq!(cost(oklab(), &[]), 14);
        assert_eq!(cost(oklab(), &[true]), 14 + 20, "swap + iso, 10 bytes each");
        // A level whose only non-direct members are filters ping-pongs and never
        // isolates, so it pays for `swap` alone (§21.3).
        assert_eq!(cost(oklab(), &[false]), 14 + 10, "swap alone");
        assert_eq!(cost(pigment(), &[]), 22, "18 accumulator + 4 target");
        assert_eq!(cost(pigment(), &[true, true]), 22 + 4 * 18);

        // And the budget turns that into samples: the same window that supersamples
        // when flat gives them up as the document nests.
        let window = Extent2::new(2560, 1440);
        let at = |f, s: &[bool]| supersample(window, 0.1, &limits(), cost(f, s));
        assert!(at(oklab(), &[]) >= at(oklab(), &[true]));
        assert!(at(oklab(), &[true]) >= at(pigment(), &[true]));
        assert_eq!(
            at(pigment(), &[true, true]),
            1,
            "two nested levels in pigment cannot afford a second sample on this window",
        );
    }
}
