//! Pass E: the presentation resolve — the supersampled render box-averaged down to
//! the caller's target (§6.4).
//!
//! Encoded only when the view is zoomed out. At 1:1 the passes above write the
//! target directly, this is never bound, and every golden blessed at `zoom = 1.0`
//! is bit-identical to before supersampling existed.

use crate::geom::Extent2;
use crate::gpu::context::GpuContext;
use crate::gpu::desc;

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
/// **It is a byte budget because it was already meant to be one.** It was written as
/// 16 Mpx with the note that this came to "~210 MB in the worst case", enumerating
/// the scratch — but no pixel count can enumerate the scratch, whose size is a fact
/// about the *document*, nor the residual, which is a fact about the color space. The
/// real figures at 16 Mpx: 234 MB for a flat Oklab document, 571 MB with one blend
/// group, 973 MB for the same in pigment, and ~1.6 GB at two levels of nesting. So
/// the number here is the ~224 MiB that was always intended, and what varies is how
/// many pixels of *this* document's attachments fit inside it.
///
/// A flat Oklab document is unchanged by the restatement, deliberately: 14 bytes a
/// texel (8 + 2 accumulator, 4 resolve target) into 224 MiB is exactly the 16 Mpx
/// this used to be. What moves is the case the old arithmetic was wrong about.
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
/// painting at 100% costs exactly what it always did and every golden blessed at
/// `zoom = 1.0` is bit-identical. Below that it is the minification ratio, so each
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
///   ping-pongs, which is `scratch_needs`' answer read as memory (§18.0.4, §21.3).
///
/// The last is the term the old pixel budget could not express, and it is the one
/// that dominates: a single blend group more than doubles the frame's footprint, and
/// in a pigment document it more than triples it.
pub(super) fn attachment_bytes(
    formats: crate::gpu::channels::ChannelFormats,
    target_format: wgpu::TextureFormat,
    scratch: &[bool],
) -> u64 {
    let trio = formats.bytes_per_px();
    let resolve = u64::from(target_format.block_copy_size(None).unwrap_or(0));
    let scratch: u64 = scratch.iter().map(|&iso| trio * (1 + u64::from(iso))).sum();
    trio + resolve + scratch
}

/// The resolve pass — the pipeline and its layout. The uniform it reads (`n`, the
/// sample count) is the *rendering* consumer's: `n` is a function of that target's
/// zoom, so the surface and a miniature beside it disagree about it by construction.
/// It rides in the [`Supersampled`] set, which is exactly the state that exists only
/// while a view is zoomed out.
pub(super) struct ResolvePass {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
}

impl ResolvePass {
    pub(super) fn new(device: &wgpu::Device, target: &[Option<wgpu::ColorTargetState>]) -> Self {
        let frag = wgpu::ShaderStages::FRAGMENT;
        // A fullscreen pass reading the supersampled target with `textureLoad` at an
        // integer block of its own choosing, so nothing here needs a sampler.
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark resolve"),
            source: wgpu::ShaderSource::Wgsl(stark_shaders::resolve().into()),
        });
        let bgl = desc::bind_group_layout(
            device,
            "stark resolve bgl",
            &[
                desc::uniform(0, frag),
                desc::load_tex(1, frag), // the supersampled render
            ],
        );
        let layout = desc::pipeline_layout(device, "stark resolve layout", &[Some(&bgl)]);
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
        Self { pipeline, bgl }
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
        n: u32,
        target: &wgpu::TextureView,
    ) {
        ctx.queue.write_buffer(
            &ss.buf,
            0,
            bytemuck::bytes_of(&ResolveUniform {
                n: [n as f32, 0.0, 0.0, 0.0],
            }),
        );
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("stark resolve pass"),
            // Covers every texel and reads nothing back, so the load is a don't-care;
            // clearing says so rather than implying otherwise.
            color_attachments: &[Some(desc::attach(target, desc::CLEAR))],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &ss.bg, &[]);
        pass.draw(0..3, 0..1);
    }
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
    pub(super) view: wgpu::TextureView,
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
        pass: &ResolvePass,
    ) -> Self {
        let view = super::offscreen_view(device, size, format, "stark supersampled");
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stark resolve uniform"),
            size: std::mem::size_of::<ResolveUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark resolve bg"),
            layout: &pass.bgl,
            entries: &[desc::uniform_entry(0, &buf), desc::tex(1, &view)],
        });
        Self { view, bg, buf }
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
        attachment_bytes(oklab(), TARGET, &[])
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

    /// The restatement's compatibility claim: a flat Oklab document sees exactly the
    /// 16-Mpx ceiling the pixel budget used to impose, so nothing about the common
    /// zoom-out moved. 14 bytes a texel into 224 MiB is 16 Mpx on the nose.
    #[test]
    fn a_flat_oklab_frame_still_stops_at_sixteen_megapixels() {
        assert_eq!(flat(), 14, "8 + 2 accumulator, 4 resolve target");
        assert_eq!(MAX_SUPERSAMPLED_BYTES / flat(), 16 << 20);
    }

    /// The term the pixel budget could not express, and the reason this is worth
    /// changing: **the scratch dominates**. One blend group more than doubles the
    /// frame — a `swap` trio and an `iso` trio, each the size of the accumulator —
    /// and a pigment document pays 18 bytes a texel for every one of them.
    ///
    /// The old ceiling would have let all four of these render at 16 Mpx, which is
    /// 973 MB for the third and about 1.6 GB for the fourth.
    #[test]
    fn the_blend_scratch_is_most_of_what_a_nested_frame_costs() {
        let cost = |f, s: &[bool]| attachment_bytes(f, TARGET, s);
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
