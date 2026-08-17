//! Pass D: the drawing guides, over everything (§20.4).
//!
//! The perspective grid is chrome the whole canvas is read *through*, so it is the
//! topmost thing drawn. One fullscreen triangle per visible guide, each off its own
//! dynamic-offset slot; the shader branches on data rather than on pipeline
//! variants, so an absent element is a zeroed slot rather than a second pipeline.

use crate::geom::ViewTransform;
use crate::gpu::context::GpuContext;
use crate::gpu::desc;
use crate::gpu::desc::Slot;
use stark_shaders::mirror::guides::decl as gd;

/// The guide overlay's one binding (§20.4).
///
/// One slot per visible guide in the frame, the stride derived from
/// [`GuideUniform`] itself. It was a hand-written `GUIDE_SLOT = 512` for
/// as long as this pass allocated its own buffer — and 512 is what
/// `UniformSlots` computes, so the constant was *right* and would have
/// stayed right only until the next time the uniform grew. It had
/// already been widened once, when the fisheye brought the second set of
/// poles (§20.8); a second such growth past 512 would have under-strided
/// every slot, and two visible guides would have read each other's
/// lanes with nothing to say so.
const GUIDE_SLOTS: &[Slot] = &[Slot::dynamic(gd::GUIDE)];
use crate::gpu::uniforms::UniformSlots;

// Generated from `guides.wesl`'s own declaration — pass D, the drawing guides
// (§20.4, §6.7).
pub(super) use stark_shaders::mirror::guides::Guide as GuideUniform;

/// Pack the derived guide scene plus this render's view mapping (§20.4).
/// Absent elements become a zeroed slot with `valid = 0` (a trace's kind), so the
/// shader branches on data rather than on pipeline variants.
///
/// A free function rather than the `GuideUniform::pack` it replaced: the type is
/// generated into `stark-shaders` now, and an inherent impl on another crate's type
/// is not allowed.
fn pack_guides(scene: &crate::guides::GuideScene, view: ViewTransform) -> GuideUniform {
    use crate::guides::{Lens, PairTrace};
    let inv = view.inverse_linear();
    let org = view.screen_to_canvas(crate::geom::Vec2::ZERO);
    let point = |v: Option<crate::geom::Vec2>| match v {
        Some(p) => [p.x, p.y, 1.0, 0.0],
        None => [0.0; 4],
    };
    let (r45, r90) = scene.rings;
    GuideUniform {
        inv: inv.to_cols_array(),
        org: [org.x, org.y, view.zoom, scene.focal],
        cov: [scene.center.x, scene.center.y, scene.opacity, 0.0],
        proj: [
            match scene.lens {
                Lens::Rectilinear => 0.0,
                Lens::Fisheye => 1.0,
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
        lines: std::array::from_fn(|i| match scene.lines[i] {
            Some(PairTrace::Line { normal, offset }) => [normal.x, normal.y, offset, 1.0],
            Some(PairTrace::Circle { center, radius }) => [center.x, center.y, radius, 2.0],
            None => [0.0; 4],
        }),
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
    }
}

pub(super) struct GuidePass {
    pub(super) pipeline: wgpu::RenderPipeline,
    pub(super) bgl: wgpu::BindGroupLayout,
}

impl GuidePass {
    pub(super) fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let frag = wgpu::ShaderStages::FRAGMENT;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark guides"),
            source: wgpu::ShaderSource::Wgsl(stark_shaders::guides().into()),
        });
        let bgl = desc::layout_for(device, "stark guides bgl", GUIDE_SLOTS, frag, false);
        let layout = desc::pipeline_layout(device, "stark guides layout", &[Some(&bgl)]);
        let pipeline = desc::fullscreen_pipeline(
            device,
            "stark guides pipeline",
            &layout,
            &shader,
            ("vs_main", "fs_main"),
            // The shader accumulates its elements premultiplied, so the pass
            // composites `src + dst·(1 − src.a)`.
            &[desc::blended_target(
                target_format,
                Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
            )],
        );
        Self { pipeline, bgl }
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
        scenes: &[crate::guides::GuideScene],
        view: ViewTransform,
        target: &wgpu::TextureView,
    ) {
        if scenes.is_empty() {
            return;
        }
        let packed: Vec<GuideUniform> = scenes.iter().map(|s| pack_guides(s, view)).collect();
        slots.write(&ctx.device, &ctx.queue, &packed);
        // Per render rather than kept: it has to follow the buffer through
        // reallocation, and a bind group over one small uniform is cheap beside
        // everything else in the frame.
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark guides bg"),
            layout: &self.bgl,
            entries: &[slots.binding(0)],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("stark guides pass"),
            color_attachments: &[Some(desc::attach(target, desc::LOAD))],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        for i in 0..scenes.len() as u32 {
            pass.set_bind_group(0, &bg, &[UniformSlots::<GuideUniform>::offset(i)]);
            pass.draw(0..3, 0..1);
        }
    }
}
