//! Pass D: the drawing guides, over everything (§20.4).
//!
//! The perspective grid is chrome the whole canvas is read *through*, so it is the
//! topmost thing drawn. One fullscreen triangle per visible guide, each off its own
//! dynamic-offset slot; the shader branches on data rather than on pipeline
//! variants, so an absent element is a zeroed slot rather than a second pipeline.

use crate::geom::ViewTransform;
use crate::gpu::desc;

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
pub(super) fn pack_guides(scene: &crate::guides::GuideScene, view: ViewTransform) -> GuideUniform {
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

/// One dynamic-offset slot of the guide uniform, padded to a multiple of the
/// alignment like [`BLEND_SLOT`](super::blend::BLEND_SLOT) and for the same reason:
/// every visible guide's slot is written before the single submit, and each draw
/// binds its own offset. Two alignment units, because [`GuideUniform`] outgrew one
/// when the fisheye brought the second set of poles (§20.8).
pub(super) const GUIDE_SLOT: u64 = 512;

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
        let bgl = desc::bind_group_layout(
            device,
            "stark guides bgl",
            // One slot per visible guide in the frame; see [`GUIDE_SLOT`].
            &[desc::uniform_slot(
                0,
                frag,
                std::mem::size_of::<GuideUniform>() as u64,
            )],
        );
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
}

pub(super) fn alloc_guides(device: &wgpu::Device, count: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("stark guides uniform"),
        size: GUIDE_SLOT * count.max(1) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
