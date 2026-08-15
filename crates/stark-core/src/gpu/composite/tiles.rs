//! Pass A: every visible tile's channels composited into the offscreen accumulator,
//! with matte layers drawn at their own place in the stack (§6.3, §15.4).
//!
//! One instanced quad per tile, one per matte, both in the canvas → NDC frame
//! [`View`] holds. The two pipelines share group 0, so a run that interleaves them
//! changes only the pipeline and the vertex buffer.

use crate::colorspace::ColorSpace;
use crate::gpu::channels::{ChannelFormats, Targets};
use crate::gpu::desc::{self, RenderPipe};
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
    pub(super) view_bgl: wgpu::BindGroupLayout,
    pub(super) tile_bgl: wgpu::BindGroupLayout,
    /// The matte pipeline's group 1: the per-matte gradient ramp (§22.4), read
    /// through a **dynamic offset** so one buffer and one bind group serve every
    /// matte in the frame. A solid matte's slot is simply zeroed, its stop count
    /// then saying "use the instance's own channels" — so the absent-ramp case is 544
    /// zero bytes in the buffer everything else reads, not a bind group of its own.
    pub(super) ramp_bgl: wgpu::BindGroupLayout,
}

impl TilePass {
    pub(super) fn new(
        device: &wgpu::Device,
        color_space: &dyn ColorSpace,
        formats: ChannelFormats,
    ) -> Self {
        let frag = wgpu::ShaderStages::FRAGMENT;
        // The residual channel a pigment space carries (§6.7): a third sampled tile
        // texture and a third target, on the shader variant built for it.
        let resid = formats.has_resid();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark composite"),
            source: wgpu::ShaderSource::Wgsl(stark_shaders::composite(resid).into()),
        });

        // Vertex-only: the fragment stage gets canvas position as a varying, and the
        // zoom rides through `misc.w` for the matte's edge antialiasing width.
        let view_bgl = desc::bind_group_layout(
            device,
            "stark composite view bgl",
            &[
                desc::uniform(0, wgpu::ShaderStages::VERTEX),
                desc::sampler(1, frag),
            ],
        );
        let mut tile_entries = vec![desc::sample_tex(0, frag), desc::sample_tex(1, frag)];
        if resid {
            tile_entries.push(desc::sample_tex(2, frag));
        }
        let tile_bgl = desc::bind_group_layout(device, "stark composite tile bgl", &tile_entries);
        let layout = desc::pipeline_layout(
            device,
            "stark composite layout",
            &[Some(&view_bgl), Some(&tile_bgl)],
        );
        // Pass A's blends come from the color space (§6.7): premultiplied `over` on
        // color, additive on the height aux.
        // The residual composites through the *color's* blend, never the aux's: it is
        // premultiplied by the same coverage and covers by the same rule, being the
        // rest of the same color (§6.7).
        // Spelled out rather than `formats.blended(..)`: pass A is the one pipeline
        // whose three targets do *not* share a blend — the height aux is additive
        // where the color and its residual composite `over` (§6.7).
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
                vs: "vs_main",
                fs: "fs_main",
                primitive: desc::QUAD_STRIP,
                buffers: &[Some(stark_shaders::mirror::composite::instance_layout(
                    wgpu::VertexStepMode::Instance,
                ))],
                targets: &space_targets,
            },
        );

        // ---- Matte layers, inside pass A (§15.4), on pass A's own view group.
        let matte_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark matte"),
            source: wgpu::ShaderSource::Wgsl(stark_shaders::matte(resid).into()),
        });
        // Group 1: the gradient ramp, per matte where the view is per pass
        // (§22.4). Fragment-only — the vertex stage has no use for it.
        let ramp_bgl = desc::bind_group_layout(
            device,
            "stark matte ramp bgl",
            &[UniformSlots::<Ramp>::layout(0, frag)],
        );
        let matte_layout = desc::pipeline_layout(
            device,
            "stark matte layout",
            &[Some(&view_bgl), Some(&ramp_bgl)],
        );
        // Premultiplied `over` on BOTH targets. The aux one is the load-bearing
        // difference from pass A's additive aux: additive would keep the height of
        // paint *underneath* the matte, and the media pass would emboss that paint's
        // impasto as ghost ridges through an opaque mat board (§15.4.2).
        // `OneMinusSrcAlpha` is valid on the alpha-less R16Float aux: the factor
        // reads the *source* alpha from the shader's output vec4.
        let over = Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING);
        let matte_targets = formats.blended(over);
        let matte_pipeline = desc::render_pipeline(
            device,
            RenderPipe {
                label: "stark matte pipeline",
                layout: &matte_layout,
                module: &matte_shader,
                vs: "vs_main",
                fs: "fs_main",
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
    /// Both pipelines share group 0 (the view uniform), so only the pipeline and the
    /// vertex buffer change at the boundary. Each [`Draw`] carries its own index into
    /// the flat streams, so the encoder walks no cursors of its own — that is the
    /// plan's job (`composite::plan`).
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
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_bind_group(0, s.view_bg, &[]);
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
                    // Group 1 is the ramp, at this matte's own slot (§22.4). Re-set
                    // per matte either way: the offset changes, and a pipeline switch
                    // has already invalidated whatever was bound, the two pipelines'
                    // group-1 layouts differing.
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
    pub(super) instances: &'a InstanceStream<Instance>,
    pub(super) mattes: &'a InstanceStream<MatteInstance>,
    /// One per tile instance, at the same index — each borrowed from the tile itself.
    pub(super) tile_bgs: &'a [&'a wgpu::BindGroup],
    /// The slotted ramp buffer's bind group, `None` when the frame has no matte.
    pub(super) ramp_bg: Option<&'a wgpu::BindGroup>,
}
