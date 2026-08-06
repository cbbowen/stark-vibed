//! Pass A: every visible tile's channels composited into the offscreen accumulator,
//! with matte layers drawn at their own place in the stack (§6.3, §15.4).
//!
//! One instanced quad per tile, one per matte, both in the canvas → NDC frame
//! [`View`] holds. The two pipelines share group 0, so a run that interleaves them
//! changes only the pipeline and the vertex buffer.

use wgpu::util::DeviceExt;

use crate::colorspace::ColorSpace;
use crate::gpu::desc::{self, RenderPipe};

use super::view::View;

// The two per-instance records pass A draws with, generated from the `@location`
// parameters of the vertex entry points that read them (§6.10) — the struct, the
// attribute formats and their offsets all from the one declaration.
pub(super) use stark_shaders::mirror::composite::Instance;
pub(super) use stark_shaders::mirror::matte::MatteInstance;

/// Pass A's pipelines and the bind groups they draw through.
pub(super) struct TilePass {
    pub(super) pipeline: wgpu::RenderPipeline,
    /// Matte layers, drawn inside pass A at their place in the stack (§15.4). Its
    /// own pipeline because its blend state differs from the colour space's: `over`
    /// on *both* targets, so an opaque matte erases the relief beneath it rather
    /// than letting underlying impasto emboss through.
    pub(super) matte_pipeline: wgpu::RenderPipeline,
    pub(super) view_bg: wgpu::BindGroup,
    pub(super) tile_bgl: wgpu::BindGroupLayout,
}

impl TilePass {
    pub(super) fn new(
        device: &wgpu::Device,
        view: &View,
        color_space: &dyn ColorSpace,
        color_format: wgpu::TextureFormat,
        aux_format: wgpu::TextureFormat,
    ) -> Self {
        let frag = wgpu::ShaderStages::FRAGMENT;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark composite"),
            source: wgpu::ShaderSource::Wgsl(stark_shaders::composite().into()),
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
        let tile_bgl = desc::bind_group_layout(
            device,
            "stark composite tile bgl",
            &[desc::sample_tex(0, frag), desc::sample_tex(1, frag)],
        );
        let layout = desc::pipeline_layout(
            device,
            "stark composite layout",
            &[Some(&view_bgl), Some(&tile_bgl)],
        );
        // Pass A's blends come from the colour space (§6.7): premultiplied `over` on
        // colour, additive on the height aux.
        let space_targets = [
            desc::blended_target(color_format, Some(color_space.color_blend())),
            desc::blended_target(aux_format, Some(color_space.aux_blend())),
        ];
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
            source: wgpu::ShaderSource::Wgsl(stark_shaders::matte().into()),
        });
        let matte_layout = desc::pipeline_layout(device, "stark matte layout", &[Some(&view_bgl)]);
        // Premultiplied `over` on BOTH targets. The aux one is the load-bearing
        // difference from pass A's additive aux: additive would keep the height of
        // paint *underneath* the matte, and the media pass would emboss that paint's
        // impasto as ghost ridges through an opaque mat board (§15.4.2).
        // `OneMinusSrcAlpha` is valid on the alpha-less R16Float aux: the factor
        // reads the *source* alpha from the shader's output vec4.
        let over = Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING);
        let matte_targets = [
            desc::blended_target(color_format, over),
            desc::blended_target(aux_format, over),
        ];
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

        let view_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark composite view bg"),
            layout: &view_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: view.buf.as_entire_binding(),
                },
                desc::samp(1, &view.sampler),
            ],
        });

        Self {
            pipeline,
            matte_pipeline,
            view_bg,
            tile_bgl,
        }
    }
}

pub(super) fn alloc_instances(device: &wgpu::Device, count: usize) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("stark composite instances"),
        contents: bytemuck::cast_slice(&vec![
            Instance {
                origin: [0.0; 2],
                opacity: 1.0
            };
            count.max(1)
        ]),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    })
}

pub(super) fn alloc_mattes(device: &wgpu::Device, count: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("stark matte instances"),
        size: (std::mem::size_of::<MatteInstance>() * count.max(1)) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
