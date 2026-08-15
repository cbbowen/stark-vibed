//! Pass B: the media/lighting pass (§6.3).
//!
//! One fullscreen draw that derives normals from the composited height field,
//! lights the impasto with image-based lighting from an [`Environment`], adds the
//! paint film's gloss, converts the working channels to display, and composites
//! over the substrate into the target. This is the "old masters" payoff.

use crate::colorspace::ColorSpace;
use crate::geom::Extent2;
use crate::gpu::desc;
use crate::gpu::environment::Environment;
use crate::gpu::surface::Surface;

// Generated from `media_common.wesl`'s own declaration (§6.7).
pub(super) use stark_shaders::mirror::media_common::Media as MediaUniform;

/// Lighting parameters for the media pass (§6.3). The painting is lit by
/// image-based lighting from an [`Environment`]; this is a single place to tune the
/// look. A view setting — never historized (it changes how the canvas looks, not
/// its pixels).
#[derive(Copy, Clone, Debug)]
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
        let mut entries = vec![
            desc::uniform(0, frag),
            desc::load_tex(1, frag),   // comp_color (textureLoad)
            desc::load_tex(2, frag),   // comp_aux   (textureLoad)
            desc::sample_tex(3, frag), // surface bump (filtered)
            desc::sampler(4, frag),
            desc::sample_tex(5, frag), // environment (filtered, mipped)
            desc::sampler(6, frag),
        ];
        // 7, the composited residual — declared by `media_mixbox.wesl` itself rather
        // than by the shared `media_common`, so a space with no residual gets a layout
        // without it instead of a placeholder to bind (§6.7).
        if color_space.has_resid() {
            entries.push(desc::load_tex(7, frag));
        }
        let bgl = desc::bind_group_layout(device, "stark media bgl", &entries);
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

    let mut entries = vec![
        desc::uniform_entry(0, media_buf),
        desc::tex(1, &color),
        desc::tex(2, &aux),
        desc::tex(3, &surface.view),
        desc::samp(4, &surface.sampler),
        desc::tex(5, &environment.view),
        desc::samp(6, &environment.sampler),
    ];
    if let Some(view) = &resid {
        entries.push(desc::tex(7, view));
    }
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("stark media bg"),
        layout: &media.bgl,
        entries: &entries,
    });
    Offscreen {
        color,
        aux,
        resid,
        bg,
    }
}
