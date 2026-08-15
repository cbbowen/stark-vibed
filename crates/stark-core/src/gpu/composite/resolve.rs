//! Pass E: the presentation resolve — the supersampled render box-averaged down to
//! the caller's target (§6.4).
//!
//! Encoded only when the view is zoomed out. At 1:1 the passes above write the
//! target directly, this is never bound, and every golden blessed at `zoom = 1.0`
//! is bit-identical to before supersampling existed.

use crate::geom::Extent2;
use crate::gpu::desc;

// Generated from `resolve.wesl`'s own declaration (§6.7).
pub(super) use stark_shaders::mirror::resolve::Resolve as ResolveUniform;

/// Most samples per axis presentation will take (§6.4). A 4×4 box is 16
/// taps per output pixel; past that the box filter's remaining error is dominated by
/// its own shape rather than by the sample count, and a view zoomed further out than
/// 1:4 is a thumbnail whose next improvement is a better filter, not more of this one.
const MAX_SUPERSAMPLE: u32 = 4;

/// Most supersampled pixels a render may cover, whatever the zoom asks for.
///
/// The ceiling exists because *every* offscreen attachment scales with it: pass A's
/// `Rgba16Float` color and `R16Float` aux, the blend scratch if the document has a
/// mode in it, and the target the resolve reads. At 16 Mpx that is ~210 MB in the
/// worst case, which is the most a painting canvas may quietly take to stop
/// sparkling — and it is only taken while the view is actually zoomed out, since
/// [`supersample`] returns 1 at 1:1 and the attachments shrink back. Crossing a
/// threshold reallocates all of them, so the ceiling is also a bound on the hitch a
/// wheel-zoom can cost.
///
/// It binds on the window and not on a miniature, which is the right way round: the
/// navigator renders a whole piece into ~250 px, is the worst-aliased view in the
/// application, and reaches [`MAX_SUPERSAMPLE`] for a few megapixels. What it costs
/// is the common zoom-outs on a large window — 2× rather than 4× past 1:2 — where
/// the picture is already most of the way back and the fourth sample buys least.
const MAX_SUPERSAMPLED_PX: u32 = 16 << 20;

/// How many samples per axis a render of `size` at `zoom` takes (§6.4).
///
/// `1` at 1:1 and closer — a view that magnifies is already oversampling, so
/// painting at 100% costs exactly what it always did and every golden blessed at
/// `zoom = 1.0` is bit-identical. Below that it is the minification ratio, so each
/// output pixel gets back roughly one sample per canvas pixel it covers, capped by
/// [`MAX_SUPERSAMPLE`], by [`MAX_SUPERSAMPLED_PX`] and by what the device will
/// allocate.
pub(super) fn supersample(size: Extent2, zoom: f32, limits: &wgpu::Limits) -> u32 {
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
            sw <= limits.max_texture_dimension_2d
                && sh <= limits.max_texture_dimension_2d
                && sw.saturating_mul(sh) <= MAX_SUPERSAMPLED_PX
        })
        .unwrap_or(1)
}

/// The resolve pass — the pipeline and its layout. The uniform it reads (`n`, the
/// sample count) is the *rendering* consumer's: `ss` is a function of that target's
/// zoom, so the surface and a miniature beside it disagree about it by construction.
/// It rides in the [`Supersampled`](super::Supersampled) set, which is exactly the
/// state that exists only while a view is zoomed out.
pub(super) struct ResolvePass {
    pub(super) pipeline: wgpu::RenderPipeline,
    pub(super) bgl: wgpu::BindGroupLayout,
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
}

/// The uniform buffer one supersampled render writes its sample count into — see
/// [`ResolvePass`] for why it is not the pass's.
pub(super) fn uniform_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("stark resolve uniform"),
        size: std::mem::size_of::<ResolveUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The device this policy is written against: the `downlevel_webgl2` floor's
    /// 8192-px limit, which is what the wasm build can count on.
    fn limits() -> wgpu::Limits {
        wgpu::Limits {
            max_texture_dimension_2d: 8192,
            ..wgpu::Limits::default()
        }
    }

    #[test]
    fn magnification_costs_nothing() {
        let window = Extent2::new(1280, 720);
        for zoom in [1.0, 1.5, 4.0, 64.0] {
            assert_eq!(supersample(window, zoom, &limits()), 1, "at {zoom}×");
        }
        // A degenerate zoom asks for nothing rather than for a division by it.
        for zoom in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert_eq!(supersample(window, zoom, &limits()), 1, "at {zoom}×");
        }
    }

    #[test]
    fn minification_is_matched_sample_for_canvas_pixel() {
        // Small enough that neither cap binds, so this is the rule itself.
        let thumb = Extent2::new(252, 176);
        assert_eq!(supersample(thumb, 0.5, &limits()), 2);
        assert_eq!(supersample(thumb, 0.25, &limits()), 4);
        // Past the cap the ratio keeps growing and the answer does not.
        assert_eq!(supersample(thumb, 0.01, &limits()), MAX_SUPERSAMPLE);
    }

    #[test]
    fn a_large_target_gives_up_samples_rather_than_the_render() {
        // The pixel budget binds first at window sizes: 4× of this is 176 Mpx.
        let window = Extent2::new(2560, 1440);
        let n = supersample(window, 0.1, &limits());
        assert!(n > 1, "a zoomed-out window should still supersample");
        assert!(
            (window.width * n) * (window.height * n) <= MAX_SUPERSAMPLED_PX,
            "{n}× of {window:?} is over the pixel budget"
        );
        // And the device limit binds before either of the others on a wide one.
        let wide = Extent2::new(7000, 400);
        assert_eq!(supersample(wide, 0.1, &limits()), 1);
    }
}
