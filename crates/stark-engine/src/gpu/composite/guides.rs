//! Pass D: the drawing guides, over everything (§20.4).
//!
//! The perspective grid is chrome the whole canvas is read *through*, so it is the
//! topmost thing drawn. One fullscreen triangle per visible guide, each off its own
//! dynamic-offset slot; the shader branches on data rather than on pipeline
//! variants, so an absent element is a zeroed slot rather than a second pipeline.

use super::display::Transfer;
use crate::gpu::context::GpuContext;
use crate::gpu::desc::{self, Bindings};
use crate::view::ViewTransform;
use stark_model::document::GuideScene;
use stark_shaders::Stages;
use stark_shaders::mirror::guides::binding as gb;
use stark_shaders::mirror::guides::decl as gd;

use crate::gpu::uniforms::UniformSlots;

// Generated from `guides.wesl`'s own declaration — pass D, the drawing guides
// (§20.4, §6.7).
pub(super) use stark_shaders::mirror::guides::Guide as GuideUniform;

/// Pack the derived guide scene plus this render's view mapping (§20.4).
/// Absent elements become a zeroed slot with `valid = 0` (`TRACE_NONE`, for a
/// trace), so the shader branches on data rather than on pipeline variants.
///
/// A free function rather than an inherent `pack`: the type is generated into
/// `stark-shaders`, and an inherent impl on another crate's type is not allowed.
fn pack_guides(scene: &GuideScene, view: ViewTransform, transfer: Transfer) -> GuideUniform {
    use stark_model::document::{Lens, PlaneTrace};
    use stark_shaders::mirror::guides::{
        LENS_FISHEYE, LENS_RECTILINEAR, TRACE_CIRCLE, TRACE_LINE, TRACE_NONE,
    };
    let inv = view.inverse_linear();
    let org = view.screen_to_canvas(stark_model::geom::Vec2::ZERO);
    let point = |v: Option<stark_model::geom::Vec2>| match v {
        Some(p) => [p.x, p.y, 1.0, 0.0],
        None => [0.0; 4],
    };
    // A trace's four numbers and its kind — the packing `lines` and `rays` share
    // because the curve is the same construction either way (`PlaneTrace`). The
    // kind is the shader's own code (§6.10).
    let trace = |t: Option<PlaneTrace>| match t {
        Some(PlaneTrace::Line { normal, offset }) => [normal.x, normal.y, offset, TRACE_LINE],
        Some(PlaneTrace::Circle { center, radius }) => [center.x, center.y, radius, TRACE_CIRCLE],
        None => [0.0, 0.0, 0.0, TRACE_NONE],
    };
    let (r45, r90) = scene.rings;
    GuideUniform {
        inv: inv.to_cols_array(),
        org: [org.x, org.y, view.zoom, scene.focal],
        cov: [
            scene.center.x,
            scene.center.y,
            scene.opacity,
            transfer.lane(),
        ],
        proj: [
            match scene.lens {
                Lens::Rectilinear => LENS_RECTILINEAR,
                Lens::Fisheye => LENS_FISHEYE,
            },
            r45,
            r90.unwrap_or(0.0),
            0.0,
        ],
        // A guide whose lattice names no grid is a zeroed slot like any other
        // absent element, and its `.w = 0` takes all six fans out (§20.3).
        grid: match scene.lattice {
            Some(g) => [g.x, g.y, g.z, 1.0],
            None => [0.0; 4],
        },
        dirs: std::array::from_fn(|i| {
            let d = scene.dirs[i];
            [d.x, d.y, d.z, scene.axis_alpha[i]]
        }),
        pairs: [
            scene.pair_alpha[0],
            scene.pair_alpha[1],
            scene.pair_alpha[2],
            0.0,
        ],
        lines: std::array::from_fn(|i| trace(scene.lines[i])),
        // Forward poles in the first three slots, backward in the last —
        // the shader colors slot `i` by axis `i % 3`.
        vps: std::array::from_fn(|i| {
            point(if i < 3 {
                scene.vps[i]
            } else {
                scene.anti_vps[i - 3]
            })
        }),
        sps: std::array::from_fn(|i| point(scene.stations[i])),
        // The rays through the hand, and the cuts that make them rays (§20.9).
        // A pointer off the canvas is the absent element it looks like: three
        // zeroed slots, and three more beside them.
        rays: std::array::from_fn(|i| trace(scene.rays[i].map(|r| r.trace))),
        cuts: std::array::from_fn(|i| match scene.rays[i].and_then(|r| r.cut) {
            Some(c) => [c.normal.x, c.normal.y, c.offset, 1.0],
            None => [0.0; 4],
        }),
    }
}

/// The guide pass's bind group layout, shared by the pipeline compiled for each
/// target format ([`TargetPasses`](super::TargetPasses)).
///
/// One slot per visible guide in the frame (§20.4), so the uniform is bound at a
/// dynamic offset; the stride is [`GuideUniform`]'s own, never named here. It comes
/// out visible to **both** stages, because the vertex stage reads the display
/// transfer out of it to convert the pass's sRGB-coded hues once per triangle rather
/// than once per texel (`guides.wesl`'s `VsOut`).
pub(super) fn guide_layout(device: &wgpu::Device) -> Bindings {
    let gu = stark_shaders::guides();
    Bindings::of(
        device,
        "stark guides bgl",
        Stages::Render(gu.vs_main, gu.fs_main),
        gd::GUIDE,
        &[gd::GUIDE],
    )
}

pub(super) struct GuidePass {
    pub(super) pipeline: wgpu::RenderPipeline,
    pub(super) bgl: Bindings,
}

impl GuidePass {
    pub(super) fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        bgl: &Bindings,
    ) -> Self {
        let guides = stark_shaders::guides();
        let shader = desc::Module::new(device, "stark guides", guides);
        let layout = desc::pipeline_layout_of(device, "stark guides layout", &[bgl]);
        let pipeline = desc::fullscreen_pipeline(
            device,
            "stark guides pipeline",
            &layout,
            &shader,
            (guides.vs_main, guides.fs_main),
            // The shader accumulates its elements premultiplied, so the pass
            // composites `src + dst·(1 − src.a)`.
            &[desc::blended_target(
                target_format,
                Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
            )],
        );
        Self {
            pipeline,
            bgl: bgl.clone(),
        }
    }

    /// Encode pass D: one fullscreen triangle per visible guide, over everything
    /// (§20.4). No pass at all when nothing is visible.
    ///
    /// Each guide draws off its own dynamic-offset slot — see [`UniformSlots`] for
    /// why they cannot share one rewritten buffer.
    pub(super) fn encode(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        slots: &mut UniformSlots<GuideUniform>,
        draw: GuideDraw<'_>,
    ) {
        let GuideDraw {
            scenes,
            view,
            transfer,
            target,
        } = draw;
        if scenes.is_empty() {
            return;
        }
        let packed: Vec<GuideUniform> = scenes
            .iter()
            .map(|s| pack_guides(s, view, transfer))
            .collect();
        slots.write(&ctx.device, &ctx.queue, &packed);
        // Cached, and dropped by whatever write reallocates the buffer under it, which
        // `UniformSlots::group` handles.
        let bindings = &self.bgl;
        let bg = slots.group(|slot| {
            bindings.group(&ctx.device, "stark guides bg", |i| match i {
                gb::GUIDE => slot.clone(),
                other => unreachable!("the guide group has no binding {other}"),
            })
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("stark guides pass"),
            color_attachments: &[Some(desc::attach(target, desc::LOAD))],
            ..Default::default()
        });
        pass.set_pipeline(&self.pipeline);
        for i in 0..scenes.len() as u32 {
            pass.set_bind_group(0, bg, &[UniformSlots::<GuideUniform>::offset(i)]);
            pass.draw(0..3, 0..1);
        }
    }
}

/// What pass D draws, and where (§20.4).
pub(super) struct GuideDraw<'a> {
    /// The guides this client has an eye on, one fullscreen triangle each.
    pub(super) scenes: &'a [GuideScene],
    /// The **supersampled** view when there is one, as every pass above the resolve
    /// reads it (§6.4).
    pub(super) view: ViewTransform,
    /// What the frame's target encodes light in (§6.5). The pass's hues are
    /// display-sRGB codes, so without this they would be texel values on a surface
    /// that is not sRGB — washed out on scRGB, oversaturated on Display P3.
    pub(super) transfer: Transfer,
    /// The lit image to draw over — the supersampled target when there is one, so the
    /// grid goes through the same resolve as the paint and comes out antialiased.
    pub(super) target: &'a wgpu::TextureView,
}
