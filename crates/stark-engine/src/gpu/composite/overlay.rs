//! Pass C: the selection outline, drawn over the lit result (§6.8, §17.3).
//!
//! One instanced quad per mask tile, in the same canvas → NDC frame as pass A. The
//! local actor's selection draws as marching ants; a peer's draws as a flat line in
//! their own color, so the two never read as the same thing.

use crate::document::selection::Selection;
use crate::gpu::context::GpuContext;
use crate::gpu::desc::{self, Bindings, RenderPipe};
use stark_shaders::Stages;
use stark_shaders::mirror::overlay::binding as ob;
use stark_shaders::mirror::overlay::decl as od;
use stark_shaders::mirror::view::decl as vd;

use crate::gpu::uniforms::InstanceStream;

/// Per-mask-tile instance of the outline pass: where the tile is, and how to draw
/// its contour. `tint.a == 0` selects the local actor's black/white marching ants;
/// anything else draws a flat line in `tint.rgb` at that alpha, which is how a
/// collaborator's selection is distinguished from your own (§17.3).
// Generated from `overlay.wesl`'s vertex parameters (§6.10).
pub(super) use stark_shaders::mirror::overlay::OverlayInstance;

/// One selection to outline, and whose it is (§17.3).
#[derive(Copy, Clone)]
pub(crate) struct SelectionOutline<'a> {
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
    ///
    /// Its uniform comes out `VERTEX_FRAGMENT` where pass A's is vertex-only, from
    /// `overlay.wesl`'s own reads: the fragment stage measures a canvas-space dash
    /// length in screen px with the zoom (§6.8).
    pub(super) view: Bindings,
    pub(super) tile: Bindings,
}

impl OverlayLayouts {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let ov = stark_shaders::overlay();
        let stages = Stages::Render(ov.vs_main, ov.fs_main);
        Self {
            view: Bindings::of(
                device,
                "stark overlay view bgl",
                stages,
                vd::VIEW,
                &[vd::VIEW],
            ),
            tile: Bindings::of(device, "stark overlay tile bgl", stages, od::MASK, &[]),
        }
    }
}

pub(super) struct OverlayPass {
    pub(super) pipeline: wgpu::RenderPipeline,
    /// Group 0's layout, kept for the `@group` it stands at — the consumer owns the
    /// buffer and hands the group in, and this is what says where it goes.
    view_bgl: Bindings,
    pub(super) tile_bgl: Bindings,
}

impl OverlayPass {
    /// The pipeline over `module`, `overlay.wesl` compiled, for `target_format`.
    pub(super) fn new(
        device: &wgpu::Device,
        module: &desc::Module,
        layouts: &OverlayLayouts,
        target_format: wgpu::TextureFormat,
    ) -> Self {
        let overlay = stark_shaders::overlay();
        let layout = desc::pipeline_layout_of(
            device,
            "stark overlay layout",
            &[&layouts.view, &layouts.tile],
        );
        let pipeline = desc::render_pipeline(
            device,
            RenderPipe {
                label: "stark overlay pipeline",
                layout: &layout,
                module,
                vs: overlay.vs_main,
                fs: overlay.fs_main,
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
            view_bgl: layouts.view.clone(),
            tile_bgl: layouts.tile.clone(),
        }
    }

    /// Encode pass C: every selection's contour over the lit image, one instanced
    /// quad per mask tile (§6.8, §17.3).
    ///
    /// The local actor's and every peer's flatten into **one** instance stream, so N
    /// collaborators still cost one render pass. A universal selection draws nothing,
    /// and a frame with no outlined tile encodes no pass at all.
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
                    self.tile_bgl
                        .group(
                            &ctx.device,
                            "stark selection outline tile bg",
                            |i| match i {
                                ob::MASK => wgpu::BindingResource::TextureView(handle.view()),
                                other => {
                                    unreachable!("the outline tile group has no binding {other}")
                                }
                            },
                        )
                }));
            }
        }
        if mask_tiles.is_empty() {
            return;
        }
        instances.write(&ctx.device, &ctx.queue, &records);
        let mut pass = desc::begin_pass(
            encoder,
            "stark selection overlay pass",
            desc::Attachments::loaded([Some(scene.target)]),
        );
        pass.set_pipeline(&self.pipeline);
        self.view_bgl
            .set(&mut pass, scene.view_bg, &[scene.view_offset]);
        pass.set_vertex_buffer(0, instances.slice());
        for (i, tile) in (0u32..).zip(mask_tiles) {
            self.tile_bgl.set(&mut pass, tile, &[]);
            pass.draw(desc::QUAD_VERTICES, i..i + 1);
        }
    }
}

/// What pass C draws, and where.
pub(super) struct OverlayScene<'a> {
    pub(super) outlines: &'a [SelectionOutline<'a>],
    /// The renderer's group 0 — the canvas → NDC mapping, bound to both stages here
    /// (§6.8).
    pub(super) view_bg: &'a wgpu::BindGroup,
    /// Which view slot to draw through — 0, the outline only ever running on the
    /// screen path, whose group is slotted for pass A's sake.
    pub(super) view_offset: u32,
    /// The lit image to draw over: the supersampled target when there is one, so the
    /// ants go through the same resolve as the paint and come out antialiased.
    pub(super) target: &'a wgpu::TextureView,
    /// The tiles this view can reach, or `None` to claim all of them — pass A's cull
    /// (§6.3), applied to the ants. A selection may hold up to `MAX_SELECTION_TILES`
    /// and the outline redraws every frame it is live, so without the cull a zoomed-in
    /// view builds an instance and a bind group per off-screen tile.
    pub(super) visible: Option<stark_model::geom::TileRect>,
}
