//! Pass A: every visible tile's channels composited into the offscreen accumulator,
//! with matte layers drawn at their own place in the stack (§6.3, §15.4).
//!
//! One instanced quad per tile, one per matte, both in the canvas → NDC frame
//! [`ViewUniform`](super::view::ViewUniform) holds. The two pipelines share group 0,
//! so a run that interleaves them changes only the pipeline and the vertex buffer.

use crate::colorspace::ColorSpace;
use crate::gpu::channels::{ChannelFormats, Targets};
use crate::gpu::desc::{self, Bindings, RenderPipe};
use stark_shaders::Stages;
use stark_shaders::mirror::matte::decl as md;
use stark_shaders::mirror::view::decl as vd;

use crate::gpu::uniforms::{InstanceStream, UniformSlots};

use super::plan::Draw;

// The two per-instance records pass A draws with, generated from the `@location`
// parameters of the vertex entry points that read them (§6.10) — the struct, the
// attribute formats and their offsets all from the one declaration.
pub(super) use stark_shaders::mirror::composite::Instance;
pub(super) use stark_shaders::mirror::matte::{MatteInstance, Ramp};

/// Pass A's pipelines and the bind groups they draw through.
pub(super) struct TilePass {
    pub(super) pipeline: wgpu::RenderPipeline,
    /// Matte layers, drawn inside pass A at their place in the stack (§15.4). Its
    /// own pipeline because its blend state differs from the color space's: `over`
    /// on *both* targets, so an opaque matte erases the relief beneath it rather
    /// than letting underlying impasto emboss through.
    pub(super) matte_pipeline: wgpu::RenderPipeline,
    /// Group 0's layout, for the consumer that owns the buffer behind it
    /// ([`ViewBindings`](super::view::ViewBindings)).
    pub(super) view_bgl: Bindings,
    pub(super) tile_bgl: Bindings,
    /// The matte pipeline's group 1: the per-matte gradient ramp (§22.4), read
    /// through a **dynamic offset** so one buffer and one bind group serve every
    /// matte in the frame. A solid matte's slot is zeroed, its stop count then
    /// saying "use the instance's own channels".
    pub(super) ramp_bgl: Bindings,
}

impl TilePass {
    /// `tile_bgl` must be the one layout every tile group answers to
    /// ([`tile_bind_group_layout`](super::tile_bind_group_layout)): a tile caches the
    /// group over its own channels ([`TilePairHandle::composite_bg`]), and the stamp
    /// loop binds that same group.
    ///
    /// [`TilePairHandle::composite_bg`]: crate::gpu::tile::TilePairHandle::composite_bg
    pub(super) fn new(
        device: &wgpu::Device,
        color_space: &dyn ColorSpace,
        formats: ChannelFormats,
        tile_bgl: Bindings,
    ) -> Self {
        let composite = stark_shaders::composite(color_space.resid());
        let matte = stark_shaders::matte(color_space.resid());
        let shader = desc::Module::new(device, "stark composite", composite);

        // Both pipelines below bind this, so both shaders' stages are folded in.
        // `view.wesl`'s uniform comes out vertex-only — the fragment stage gets canvas
        // position as a varying and the zoom through `misc.w` — and its sampler
        // fragment-only, which no list has to say.
        let view_bgl = Bindings::shared_by(
            device,
            "stark composite view bgl",
            &[
                composite.vs_main,
                composite.fs_main,
                matte.vs_main,
                matte.fs_main,
            ],
            vd::VIEW,
            &[vd::VIEW],
        );
        let layout =
            desc::pipeline_layout_of(device, "stark composite layout", &[&view_bgl, &tile_bgl]);
        // Pass A is the one pipeline whose targets do *not* share a blend, so this is
        // spelled out rather than `formats.blended(..)`: premultiplied `over` on the
        // color, additive on the height aux, and the residual through the *color's*
        // blend, being the rest of the same color (§6.7).
        let mut space_targets = vec![
            desc::blended_target(formats.color, Some(color_space.color_blend())),
            desc::blended_target(formats.aux, Some(color_space.aux_blend())),
        ];
        if let Some(f) = formats.resid {
            space_targets.push(desc::blended_target(f, Some(color_space.color_blend())));
        }
        let pipeline = desc::render_pipeline(
            device,
            RenderPipe {
                label: "stark composite pipeline",
                layout: &layout,
                module: &shader,
                vs: composite.vs_main,
                fs: composite.fs_main,
                primitive: desc::QUAD_STRIP,
                buffers: &[Some(stark_shaders::mirror::composite::instance_layout(
                    wgpu::VertexStepMode::Instance,
                ))],
                targets: &space_targets,
            },
        );

        // ---- Matte layers, inside pass A (§15.4), on pass A's own view group.
        let matte_shader = desc::Module::new(device, "stark matte", matte);
        // The ramp is per matte where the view is per pass (§22.4), so it is bound at a
        // dynamic offset — the one thing `var<uniform> ramp` does not say.
        let ramp_bgl = Bindings::of(
            device,
            "stark matte ramp bgl",
            Stages::Render(matte.vs_main, matte.fs_main),
            md::RAMP,
            &[md::RAMP],
        );
        let matte_layout =
            desc::pipeline_layout_of(device, "stark matte layout", &[&view_bgl, &ramp_bgl]);
        // Premultiplied `over` on BOTH targets. On the aux that is the load-bearing
        // difference from pass A's additive blend: additive would keep the height of
        // paint *underneath* the matte, and the media pass would emboss it as ghost
        // ridges through an opaque mat board (§15.4.2). `OneMinusSrcAlpha` is valid on
        // the alpha-less R16Float aux — the factor reads the *source* alpha, from the
        // shader's output vec4.
        let over = Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING);
        let matte_targets = formats.blended(over);
        let matte_pipeline = desc::render_pipeline(
            device,
            RenderPipe {
                label: "stark matte pipeline",
                layout: &matte_layout,
                module: &matte_shader,
                vs: matte.vs_main,
                fs: matte.fs_main,
                primitive: desc::QUAD_STRIP,
                buffers: &[Some(stark_shaders::mirror::matte::matte_instance_layout(
                    wgpu::VertexStepMode::Instance,
                ))],
                targets: &matte_targets,
            },
        );

        Self {
            pipeline,
            matte_pipeline,
            view_bgl,
            tile_bgl,
            ramp_bgl,
        }
    }

    /// Encode one run of pass A: `draws` into `into`, in stack order, switching
    /// pipelines where a matte sits between runs of tiles.
    ///
    /// `draws` must already be in stack order and each [`Draw`] must carry its own
    /// index into the streams of `s` — the plan's job (`composite::plan`), since this
    /// walks no cursors of its own.
    pub(super) fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        into: Targets<'_>,
        draws: &[Draw],
        clear: bool,
        s: &TileStreams<'_>,
    ) {
        let ops = if clear { desc::CLEAR } else { desc::LOAD };
        let attachments = into.attachments(ops);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("stark composite pass"),
            color_attachments: &attachments[..into.count()],
            ..Default::default()
        });
        pass.set_bind_group(0, s.view_bg, &[s.view_offset]);
        let mut pipeline_is_matte = None;
        for draw in draws {
            match *draw {
                Draw::Tile(i) => {
                    if pipeline_is_matte != Some(false) {
                        pass.set_pipeline(&self.pipeline);
                        pass.set_vertex_buffer(0, s.instances.slice());
                        pipeline_is_matte = Some(false);
                    }
                    pass.set_bind_group(1, s.tile_bgs[i as usize], &[]);
                    pass.draw(0..4, i..i + 1);
                }
                Draw::Matte(i) => {
                    if pipeline_is_matte != Some(true) {
                        pass.set_pipeline(&self.matte_pipeline);
                        pass.set_vertex_buffer(0, s.mattes.slice());
                        pipeline_is_matte = Some(true);
                    }
                    // Group 1 is the ramp, at this matte's own slot (§22.4) — re-set
                    // per matte because the offset changes.
                    let ramp = s.ramp_bg.expect("a matte draw without its ramp slots");
                    pass.set_bind_group(1, ramp, &[UniformSlots::<Ramp>::offset(i)]);
                    pass.draw(0..4, i..i + 1);
                }
            }
        }
    }
}

/// The buffers and bind groups one run of pass A draws through — everything the
/// renderer prepared, as against what the step itself says.
pub(super) struct TileStreams<'a> {
    /// Group 0: the canvas → NDC mapping, vertex-only here.
    pub(super) view_bg: &'a wgpu::BindGroup,
    /// Which view slot this run draws through
    /// ([`ViewBindings::offset`](super::view::ViewBindings::offset)). A frame has one
    /// view and passes 0; the eyedropper's trace has one per sampled point, all in
    /// the same submit.
    pub(super) view_offset: u32,
    pub(super) instances: &'a InstanceStream<Instance>,
    pub(super) mattes: &'a InstanceStream<MatteInstance>,
    /// One per tile instance, at the same index — each borrowed from the tile itself.
    pub(super) tile_bgs: &'a [&'a wgpu::BindGroup],
    /// The slotted ramp buffer's bind group, `None` when the frame has no matte.
    pub(super) ramp_bg: Option<&'a wgpu::BindGroup>,
}
