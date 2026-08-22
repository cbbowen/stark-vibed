//! Pass B: the media/lighting pass (§6.3).
//!
//! One fullscreen draw that derives normals from the composited height field,
//! lights the impasto with image-based lighting from an [`Environment`], adds the
//! paint film's gloss, converts the working channels to display, and composites
//! over the substrate into the target. This is the "old masters" payoff.

use crate::colorspace::ColorSpace;
use crate::gpu::context::GpuContext;
use crate::gpu::desc::{self, Slot};
use crate::view::ViewTransform;
use stark_model::geom::Extent2;
use stark_shaders::mirror::media_common::binding as mc;
use stark_shaders::mirror::media_common::decl as mcd;
use stark_shaders::mirror::media_mixbox::binding as mm;
use stark_shaders::mirror::media_mixbox::decl as mmd;

/// Which bindings the media pass reads, in layout order (§6.10).
const MEDIA_SLOTS: &[Slot] = &[
    Slot::at(mcd::M),
    Slot::at(mcd::COMP_COLOR),
    Slot::at(mcd::COMP_AUX),
    // The weave and the environment are read through samplers — the first filtered at
    // canvas scale, the second mipped for the roughness lobe (§6.3).
    Slot::sampled(mcd::SURFACE),
    Slot::at(mcd::SURFACE_SAMP),
    Slot::sampled(mcd::ENV),
    Slot::at(mcd::ENV_SAMP),
    // The composited residual, declared by `media_mixbox.wesl` itself rather than by
    // the shared `media_common` — so a space with no residual gets a layout without it
    // instead of a placeholder to bind (§6.7).
    Slot::at(mmd::COMP_RESID).only_with_resid(),
];

/// A texture view as the resource a bind-group entry takes.
fn tex(v: &wgpu::TextureView) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::TextureView(v)
}
use crate::gpu::environment::Environment;
use crate::gpu::surface::Surface;

// Generated from `media_common.wesl`'s own declaration (§6.7).
pub(super) use stark_shaders::mirror::media_common::Media as MediaUniform;

/// Lighting parameters for the media pass (§6.3). The painting is lit by
/// image-based lighting from an [`Environment`]; this is a single place to tune the
/// look. A view setting — never historized (it changes how the canvas looks, not
/// its pixels).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MediaParams {
    /// Relief slope: how strongly the height field tilts normals (impasto/weave).
    pub height_strength: f32,
    /// Paint glossiness in `[0,1]`: how smooth (low-roughness) the paint film is,
    /// driving the Cook–Torrance specular. 0 = matte; 1 = near mirror-smooth. It is
    /// a uniform property of paint — every texel with paint on it is equally glossy,
    /// ramped only by how much of the fragment *is* paint (its visible alpha), so
    /// the bare canvas behind it stays rough → matte.
    pub specular: f32,
    /// How strongly the canvas surface relief shows (its weave amplitude).
    pub surface_strength: f32,
}

impl Default for MediaParams {
    fn default() -> Self {
        Self {
            height_strength: 0.15,
            specular: 0.20,
            // The weave is off until asked for: the default canvas is linen, and its
            // relief is there to be *painted into* (§6.2) whether or not the light is
            // made to show it. Raising this embosses it into the lit result.
            surface_strength: 0.0,
        }
    }
}

/// The media pass — the pipeline and its layout, which is the whole of what is
/// shareable about it.
///
/// The parameters it is tuned to live on the
/// [`CompositorPipeline`](super::CompositorPipeline) beside the other view settings,
/// and the **uniform buffer** they are written into lives on the
/// [`Compositor`](super::Compositor) that is rendering. Both for the same reason:
/// two engines sharing one set of pipelines light their canvases differently (the
/// brush editor's preview mirrors the canvas, a preset thumbnail deliberately does
/// not), so the values are per-consumer and so is the buffer holding them. The bind
/// group over it was per-`Compositor` already, which is what made moving the buffer
/// beside it cost nothing.
pub(super) struct MediaPass {
    pub(super) pipeline: wgpu::RenderPipeline,
    pub(super) bgl: wgpu::BindGroupLayout,
}

impl MediaPass {
    pub(super) fn new(
        device: &wgpu::Device,
        color_space: &dyn ColorSpace,
        target: &[Option<wgpu::ColorTargetState>],
    ) -> Self {
        let frag = wgpu::ShaderStages::FRAGMENT;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark media"),
            source: wgpu::ShaderSource::Wgsl(color_space.media_shader().into()),
        });
        let bgl = desc::layout_for(
            device,
            "stark media bgl",
            MEDIA_SLOTS,
            frag,
            color_space.has_resid(),
        );
        let layout = desc::pipeline_layout(device, "stark media layout", &[Some(&bgl)]);
        let pipeline = desc::fullscreen_pipeline(
            device,
            "stark media pipeline",
            &layout,
            &shader,
            ("vs_main", "fs_main"),
            target,
        );
        Self { pipeline, bgl }
    }

    /// Encode pass B: normals off the composited height field, lit by the
    /// environment, tonemapped, over the substrate and into `target` (§6.3).
    ///
    /// Writes the consumer's own uniform first — `buf`, whose bind group is the
    /// accumulator's `bg` — so the values a render reads are the ones it wrote.
    pub(super) fn encode(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        buf: &wgpu::Buffer,
        accum_bg: &wgpu::BindGroup,
        target: &wgpu::TextureView,
        scene: MediaScene<'_>,
    ) {
        ctx.queue
            .write_buffer(buf, 0, bytemuck::bytes_of(&scene.uniform()));
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("stark media pass"),
            // The pass covers every texel, so the clear only matters for what alpha an
            // untouched texel would keep — transparent, on an export that wants a
            // cut-out.
            color_attachments: &[Some(desc::attach(
                target,
                desc::clear_to(if scene.transparent {
                    wgpu::Color::TRANSPARENT
                } else {
                    wgpu::Color::BLACK
                }),
            ))],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, accum_bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// What pass B needs beyond the accumulator: how the frame is lit, what is under the
/// paint, and where the canvas is (§6.3).
pub(super) struct MediaScene<'a> {
    pub(super) params: MediaParams,
    pub(super) environment: &'a Environment,
    /// Canvas px → surface-tile uv for the ground **the document is on**
    /// ([`Ground::uv_scale`](crate::gpu::Ground::uv_scale)) — how large the weave is
    /// laid, as this frame is showing it.
    ///
    /// Off the document rather than off the bound map, and that is not a shortcut:
    /// the map's *height* channel — the only one this pass reads — is what the ground
    /// is, and is the same field however large it is laid. All the scale-dependence
    /// of a bake is in the rise channels, which only the deposit reads. So the light
    /// can follow a scale the instant it changes, while the tooth waits for the bake
    /// the commit triggers — which is exactly what makes the slider previewable
    /// (`ViewCommand::PreviewSurfaceScale`) instead of costing a whole-image filter
    /// per pointer sample.
    pub(super) grain_uv: f32,
    /// The **supersampled** view, so the weave and the relief are measured in the
    /// texels this pass is actually shading (§6.4).
    pub(super) view: ViewTransform,
    pub(super) background: [f32; 4],
    pub(super) background_resid: [f32; 4],
    /// Skip the substrate and carry the paint's visible alpha out, for a cut-out
    /// export (§15.6).
    pub(super) transparent: bool,
}

impl MediaScene<'_> {
    /// The scene as the shader reads it — the one place these numbers become an ABI.
    fn uniform(&self) -> MediaUniform {
        let m = self.params;
        // Screen→canvas mapping for sampling the surface bump in canvas space, so the
        // weave stays attached to the canvas as it pans, zooms, turns and mirrors
        // (§6.4, §18.1.2).
        let canvas_origin = self.view.screen_to_canvas(stark_model::geom::Vec2::ZERO);
        // Diffuse samples a heavily-blurred high mip ≈ hemispherical irradiance; the
        // level is the environment's own, so this CPU-side normalization is reading
        // exactly the texels the shader will. The Cook–Torrance specular picks its own
        // mip from roughness, spanning the whole chain (roughness 0 → mip 0 sharp;
        // roughness 1 → the diffuse level, the hemispherical average).
        let diffuse_lod = self.environment.diffuse_lod as f32;
        // Exposure belongs to the light, not to a knob beside it: each environment is
        // shown at the value it was judged at (§6.3). Normalized by the irradiance a
        // *flat* canvas receives, so `1.0` means the same thing in every environment —
        // an unrelieved patch of paint comes back out its own color.
        let exposure = self.environment.exposure / self.environment.flat_irradiance;
        MediaUniform {
            bg: self.background,
            bg_resid: self.background_resid,
            shade: [exposure, diffuse_lod, m.specular, m.height_strength],
            surf_a: [
                canvas_origin.x,
                canvas_origin.y,
                1.0 / self.view.zoom,
                self.grain_uv,
            ],
            surf_b: [
                m.surface_strength,
                if self.transparent { 1.0 } else { 0.0 },
                0.0,
                0.0,
            ],
            surf_m: self.view.inverse_linear().to_cols_array(),
        }
    }
}

/// The uniform buffer one consumer writes its lighting into — see [`MediaPass`] for
/// why it is not the pass's.
pub(super) fn uniform_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("stark media uniform"),
        size: std::mem::size_of::<MediaUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// The inputs to [`offscreen`]: one field from the [`Compositor`](super::Compositor)
/// being built or rebuilt (`size`) and the rest read off the shared
/// [`CompositorPipeline`](super::CompositorPipeline). Grouped because the two callers
/// would otherwise each spell out the same eight fields.
pub(super) struct OffscreenDesc<'a> {
    pub(super) device: &'a wgpu::Device,
    pub(super) size: Extent2,
    /// The channel formats the accumulator carries (§6.7).
    pub(super) formats: crate::gpu::channels::ChannelFormats,
    pub(super) media: &'a MediaPass,
    /// The consumer's own uniform buffer, which this bind group names.
    pub(super) media_buf: &'a wgpu::Buffer,
    pub(super) surface: &'a Surface,
    pub(super) environment: &'a Environment,
}

/// The offscreen targets pass A writes and the media bind group that reads them.
pub(super) struct Offscreen {
    pub(super) color: wgpu::TextureView,
    pub(super) aux: wgpu::TextureView,
    /// The residual accumulator, present exactly when the space has a residual — and
    /// then it is pass A's third attachment as well as the media pass's binding 7.
    pub(super) resid: Option<wgpu::TextureView>,
    pub(super) bg: wgpu::BindGroup,
}

impl Offscreen {
    /// The trio as pass A attaches it (§6.7) — the same three views the media bind
    /// group above reads, which is the invariant this type exists to hold together.
    pub(super) fn targets(&self) -> crate::gpu::channels::Targets<'_> {
        crate::gpu::channels::Targets {
            color: &self.color,
            aux: &self.aux,
            resid: self.resid.as_ref(),
        }
    }
}

/// (Re)create the offscreen composite targets and the media bind group over them.
pub(super) fn offscreen(d: OffscreenDesc<'_>) -> Offscreen {
    let OffscreenDesc {
        device,
        size,
        formats,
        media,
        media_buf,
        surface,
        environment,
    } = d;
    let color = super::offscreen_view(device, size, formats.color, "stark comp color");
    let aux = super::offscreen_view(device, size, formats.aux, "stark comp aux");
    let resid = formats
        .resid
        .map(|f| super::offscreen_view(device, size, f, "stark comp resid"));

    let bg = desc::bind_group_for(
        device,
        "stark media bg",
        &media.bgl,
        MEDIA_SLOTS,
        resid.is_some(),
        |i| match i {
            mc::M => media_buf.as_entire_binding(),
            mc::COMP_COLOR => tex(&color),
            mc::COMP_AUX => tex(&aux),
            mc::SURFACE => tex(&surface.view),
            mc::SURFACE_SAMP => wgpu::BindingResource::Sampler(&surface.sampler),
            mc::ENV => tex(&environment.view),
            mc::ENV_SAMP => wgpu::BindingResource::Sampler(&environment.sampler),
            mm::COMP_RESID => tex(resid
                .as_ref()
                .expect("a residual build has a composited residual")),
            other => unreachable!("`MEDIA_SLOTS` lists no binding {other}"),
        },
    );
    Offscreen {
        color,
        aux,
        resid,
        bg,
    }
}
