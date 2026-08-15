//! Pass C: the selection outline, drawn over the lit result (§6.8, §17.3).
//!
//! One instanced quad per mask tile, in the same canvas → NDC frame as pass A. The
//! local actor's selection draws as marching ants; a peer's draws as a flat line in
//! their own color, so the two never read as the same thing.

use crate::document::selection::Selection;
use crate::gpu::desc::{self, RenderPipe};

/// Per-mask-tile instance of the outline pass: where the tile is, and how to draw
/// its contour. `tint.a == 0` selects the local actor's black/white marching ants;
/// anything else draws a flat line in `tint.rgb` at that alpha — which is how
/// another collaborator's selection is distinguished from your own
/// (§17.3).
// Generated from `overlay.wesl`'s vertex parameters (§6.10).
pub(super) use stark_shaders::mirror::overlay::OverlayInstance;

/// One selection to outline, and whose it is (§17.3).
#[derive(Copy, Clone)]
pub struct SelectionOutline<'a> {
    pub selection: &'a Selection,
    /// `None` for the local actor — the marching ants. `Some(rgb)` for a peer's,
    /// drawn as a flat line in their color so the two never read as the same thing.
    pub tint: Option<[f32; 3]>,
}

/// How strongly another actor's selection outline reads against the artwork. Well
/// below the local one, which is a full-strength dashed line: yours is a thing you
/// act through, theirs is a thing you need only be aware of.
pub(super) const PEER_OUTLINE_ALPHA: f32 = 0.55;

pub(super) struct OverlayPass {
    pub(super) pipeline: wgpu::RenderPipeline,
    /// Group 0's layout, for the consumer that owns the buffer behind it
    /// ([`ViewBindings`](super::view::ViewBindings)).
    pub(super) view_bgl: wgpu::BindGroupLayout,
    pub(super) tile_bgl: wgpu::BindGroupLayout,
}

impl OverlayPass {
    pub(super) fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let frag = wgpu::ShaderStages::FRAGMENT;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark selection overlay"),
            source: wgpu::ShaderSource::Wgsl(stark_shaders::overlay().into()),
        });
        // Its own view bind group rather than pass A's: the fragment stage needs the
        // uniform too (it converts a canvas-space distance to screen px with the
        // zoom), and pass A declares it vertex-only.
        let view_bgl = desc::bind_group_layout(
            device,
            "stark overlay view bgl",
            &[
                desc::uniform(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
                desc::sampler(1, frag),
            ],
        );
        let tile_bgl = desc::bind_group_layout(
            device,
            "stark overlay tile bgl",
            &[desc::sample_tex(0, frag)],
        );
        let layout = desc::pipeline_layout(
            device,
            "stark overlay layout",
            &[Some(&view_bgl), Some(&tile_bgl)],
        );
        let pipeline = desc::render_pipeline(
            device,
            RenderPipe {
                label: "stark overlay pipeline",
                layout: &layout,
                module: &shader,
                vs: "vs_main",
                fs: "fs_main",
                primitive: desc::QUAD_STRIP,
                buffers: &[Some(
                    stark_shaders::mirror::overlay::overlay_instance_layout(
                        wgpu::VertexStepMode::Instance,
                    ),
                )],
                // The outline is drawn *over* the finished image, so it is the one
                // pass that blends in straight (non-premultiplied) alpha.
                targets: &[desc::blended_target(
                    target_format,
                    Some(wgpu::BlendState::ALPHA_BLENDING),
                )],
            },
        );
        Self {
            pipeline,
            view_bgl,
            tile_bgl,
        }
    }
}
