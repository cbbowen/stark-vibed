//! Pass B: the media/lighting pass (§6.3).
//!
//! One fullscreen draw that derives normals from the composited height field,
//! lights the impasto with image-based lighting from an [`Environment`], adds the
//! paint film's gloss, converts the working channels to display, and composites
//! over the substrate into the target. This is the "old masters" payoff.

use bytemuck::{Pod, Zeroable};

use crate::colorspace::ColorSpace;
use crate::geom::Extent2;
use crate::gpu::desc;
use crate::gpu::environment::Environment;
use crate::gpu::surface::Surface;
use crate::gpu::wesl::mirrors_wesl;

/// Mirrors `Media` in `media_common.wesl`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(super) struct MediaUniform {
    pub(super) light: [f32; 4], // _, _, _, height_strength (relief slope; xyz unused under IBL)
    pub(super) bg: [f32; 4],    // background (substrate) in latent channels (xyz), unused w
    pub(super) shade: [f32; 4], // exposure, diffuse_lod, gloss, _
    // Screen→canvas mapping + surface (bump) sampling for the canvas relief:
    pub(super) surf_a: [f32; 4], // canvas_origin.xy (canvas px at pixel 0), canvas_per_px, inv_tile
    pub(super) surf_b: [f32; 4], // surface_strength, transparent (0/1), _, _
    // The screen→canvas linear map, column-major: what carries a fragment's position
    // into canvas space so the weave stays attached to the canvas however the view is
    // turned or mirrored. `surf_a.z` is the same map's *length* scale, which rotation
    // and mirroring leave alone, and which the relief slope still wants as a scalar.
    pub(super) surf_m: [f32; 4],
}
mirrors_wesl!(MediaUniform, 96);

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

/// The media pass, plus the parameters it is currently tuned to.
pub(super) struct MediaPass {
    pub(super) pipeline: wgpu::RenderPipeline,
    pub(super) bgl: wgpu::BindGroupLayout,
    pub(super) buf: wgpu::Buffer,
    pub(super) params: MediaParams,
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
        let bgl = desc::bind_group_layout(
            device,
            "stark media bgl",
            &[
                desc::uniform(0, frag),
                desc::load_tex(1, frag),   // comp_color (textureLoad)
                desc::load_tex(2, frag),   // comp_aux   (textureLoad)
                desc::sample_tex(3, frag), // surface bump (filtered)
                desc::sampler(4, frag),
                desc::sample_tex(5, frag), // environment (filtered, mipped)
                desc::sampler(6, frag),
            ],
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
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stark media uniform"),
            size: std::mem::size_of::<MediaUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            bgl,
            buf,
            params: MediaParams::default(),
        }
    }
}

/// The inputs to [`offscreen`]: one field from the [`Compositor`](super::Compositor)
/// being built or rebuilt (`size`) and the rest read off the shared
/// [`CompositorPipeline`](super::CompositorPipeline). Grouped because the two callers
/// would otherwise each spell out the same eight fields.
pub(super) struct OffscreenDesc<'a> {
    pub(super) device: &'a wgpu::Device,
    pub(super) size: Extent2,
    pub(super) color_format: wgpu::TextureFormat,
    pub(super) aux_format: wgpu::TextureFormat,
    pub(super) media: &'a MediaPass,
    pub(super) surface: &'a Surface,
    pub(super) environment: &'a Environment,
}

/// (Re)create the offscreen composite targets and the media bind group over them.
pub(super) fn offscreen(
    d: OffscreenDesc<'_>,
) -> (wgpu::TextureView, wgpu::TextureView, wgpu::BindGroup) {
    let OffscreenDesc {
        device,
        size,
        color_format,
        aux_format,
        media,
        surface,
        environment,
    } = d;
    let color = super::offscreen_view(device, size, color_format, "stark comp color");
    let aux = super::offscreen_view(device, size, aux_format, "stark comp aux");

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("stark media bg"),
        layout: &media.bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: media.buf.as_entire_binding(),
            },
            desc::tex(1, &color),
            desc::tex(2, &aux),
            desc::tex(3, &surface.view),
            desc::samp(4, &surface.sampler),
            desc::tex(5, &environment.view),
            desc::samp(6, &environment.sampler),
        ],
    });
    (color, aux, bg)
}
