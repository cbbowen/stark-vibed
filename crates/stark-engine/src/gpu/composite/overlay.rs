//! Pass C: the selection outline, drawn over the lit result (§6.8, §17.3).
//!
//! One instanced quad per mask tile, in the same canvas → NDC frame as pass A. The
//! local actor's selection draws as marching ants; a peer's draws as a flat line in
//! their own color, so the two never read as the same thing.

use crate::document::selection::Selection;
use crate::gpu::context::GpuContext;
use crate::gpu::desc::Slot;
use crate::gpu::desc::{self, RenderPipe};
use stark_shaders::mirror::overlay::decl as od;

/// The overlay's **view** group — the same two things pass A's view group holds, from
/// `overlay.wesl`'s own declarations (§6.10).
///
/// Its uniform is `VERTEX_FRAGMENT` where pass A's is vertex-only: the fragment stage
/// reads the view here too, to convert a canvas-space dash length into screen px with
/// the zoom.
pub(super) const VIEW_SLOTS: &[Slot] = &[
    Slot::dynamic(od::VIEW).in_stages(wgpu::ShaderStages::VERTEX_FRAGMENT),
    Slot::at(od::SAMP),
];

/// One selection mask tile, sampled to find the contour.
const MASK_SLOTS: &[Slot] = &[Slot::sampled(od::MASK)];
use crate::gpu::uniforms::InstanceStream;

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

/// The overlay pass's two layouts, shared by the pipeline compiled for each target
/// format ([`TargetPasses`](super::TargetPasses)) so a consumer's view bind group
/// ([`ViewBindings`](super::view::ViewBindings)) is valid against either.
pub(super) struct OverlayLayouts {
    /// Group 0's layout, for the consumer that owns the buffer behind it.
    pub(super) view: wgpu::BindGroupLayout,
    pub(super) tile: wgpu::BindGroupLayout,
}

impl OverlayLayouts {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let frag = wgpu::ShaderStages::FRAGMENT;
        // Its own view bind group rather than pass A's: the fragment stage needs the
        // uniform too (it converts a canvas-space distance to screen px with the
        // zoom), and pass A declares it vertex-only.
        Self {
            view: desc::layout_for(device, "stark overlay view bgl", VIEW_SLOTS, frag, false),
            tile: desc::layout_for(device, "stark overlay tile bgl", MASK_SLOTS, frag, false),
        }
    }
}

pub(super) struct OverlayPass {
    pub(super) pipeline: wgpu::RenderPipeline,
    pub(super) tile_bgl: wgpu::BindGroupLayout,
}

impl OverlayPass {
    pub(super) fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        layouts: &OverlayLayouts,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark selection overlay"),
            source: wgpu::ShaderSource::Wgsl(stark_shaders::overlay().into()),
        });
        let layout = desc::pipeline_layout(
            device,
            "stark overlay layout",
            &[Some(&layouts.view), Some(&layouts.tile)],
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
            tile_bgl: layouts.tile.clone(),
        }
    }

    /// Encode pass C: every selection's contour over the lit image, one instanced
    /// quad per mask tile (§6.8, §17.3).
    ///
    /// The local actor's and every present peer's are flattened into **one** instance
    /// stream, so N collaborators still cost one render pass. A universal selection
    /// draws nothing, so an unmasked document costs one skipped iteration; a frame
    /// with no outlined tile at all encodes no pass.
    pub(super) fn encode(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        instances: &mut InstanceStream<OverlayInstance>,
        scene: OverlayScene<'_>,
    ) {
        let mut records: Vec<OverlayInstance> = Vec::new();
        let mut mask_tiles: Vec<&wgpu::BindGroup> = Vec::new();
        for outline in scene.outlines {
            if outline.selection.is_universal() {
                continue;
            }
            let tint = match outline.tint {
                Some([r, g, b]) => [r, g, b, PEER_OUTLINE_ALPHA],
                None => [0.0; 4],
            };
            // Each selection's own level, so the ants trace *its* half-contour: a
            // partial selection has no 0.5 in it at all (§6.8), and every peer's
            // may differ from yours.
            let level = outline.selection.level();
            for (coord, handle) in outline.selection.tiles() {
                if scene.visible.is_some_and(|r| !r.contains(*coord)) {
                    continue;
                }
                records.push(OverlayInstance {
                    origin: coord.origin().to_array(),
                    tint,
                    level,
                });
                // Kept on the mask tile, like pass A's on the paint tile: the ants
                // redraw every frame a selection is live, and the mask is immutable.
                mask_tiles.push(handle.overlay_bg(|| {
                    desc::bind_group_for(
                        &ctx.device,
                        "stark selection outline tile bg",
                        &self.tile_bgl,
                        MASK_SLOTS,
                        false,
                        |_| wgpu::BindingResource::TextureView(handle.view()),
                    )
                }));
            }
        }
        if mask_tiles.is_empty() {
            return;
        }
        instances.write(&ctx.device, &ctx.queue, &records);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("stark selection overlay pass"),
            color_attachments: &[Some(desc::attach(scene.target, desc::LOAD))],
            ..Default::default()
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, scene.view_bg, &[scene.view_offset]);
        pass.set_vertex_buffer(0, instances.slice());
        for i in 0..mask_tiles.len() as u32 {
            pass.set_bind_group(1, mask_tiles[i as usize], &[]);
            pass.draw(0..4, i..i + 1);
        }
    }
}

/// What pass C draws, and where.
pub(super) struct OverlayScene<'a> {
    pub(super) outlines: &'a [SelectionOutline<'a>],
    /// The renderer's group 0 — the canvas → NDC mapping, bound to both stages here
    /// (§6.8).
    pub(super) view_bg: &'a wgpu::BindGroup,
    /// Which view slot to draw through. The outline only ever runs on the screen
    /// path, which has one view, so this is 0 — stated rather than assumed, since the
    /// group it binds is slotted for pass A's sake.
    pub(super) view_offset: u32,
    /// The lit image to draw over: the supersampled target when there is one, so the
    /// ants go through the same resolve as the paint and come out antialiased.
    pub(super) target: &'a wgpu::TextureView,
    /// The tiles this view can reach, or `None` to claim all of them — pass A's cull
    /// (§6.3), applied to the ants for its reason.
    ///
    /// A selection may hold up to `MAX_SELECTION_TILES` of them and the outline is
    /// redrawn every frame it is live, so a zoomed-in view was building an instance
    /// and looking up a bind group for a thousand tiles to draw the two on screen.
    /// Same rect, same conservatism, same "cannot measure it, so cull nothing".
    pub(super) visible: Option<stark_model::geom::TileRect>,
}
