//! Pass B: the media/lighting pass (§6.3).
//!
//! One fullscreen draw that derives normals from the composited height field,
//! lights the impasto with image-based lighting from an [`Environment`], adds the
//! paint film's gloss, converts the working channels to display, and composites
//! over the substrate into the target. This is the "old masters" payoff.

use super::attachment::Trio;
use super::display::Output;
use crate::colorspace::ColorSpace;
use crate::gpu::context::GpuContext;
use crate::gpu::desc::{self, Slot};
use crate::gpu::environment::Environment;
use crate::gpu::substrate::SubstrateMap;
use crate::view::{Extent2, ViewTransform};
use stark_shaders::mirror::media_common::binding as mc;
use stark_shaders::mirror::media_common::decl as mcd;
use stark_shaders::mirror::media_mixbox::binding as mm;
use stark_shaders::mirror::media_mixbox::decl as mmd;

// Generated from `media_common.wesl`'s own declaration (§6.7).
pub(super) use stark_shaders::mirror::media_common::Media as MediaUniform;

/// Which bindings the media pass reads, in layout order (§6.10).
const MEDIA_SLOTS: &[Slot] = &[
    Slot::at(mcd::M),
    Slot::at(mcd::COMP_COLOR),
    Slot::at(mcd::COMP_AUX),
    // The substrate and the environment are read through samplers — the first filtered at
    // canvas scale, the second mipped for the roughness lobe (§6.3).
    Slot::sampled(mcd::SUBSTRATE),
    Slot::at(mcd::SUBSTRATE_SAMP),
    Slot::sampled(mcd::ENV),
    Slot::at(mcd::ENV_SAMP),
    // The composited residual, declared by `media_mixbox.wesl` itself rather than by
    // the shared `media_common` — so a space with no residual gets a layout without it
    // instead of a placeholder to bind (§6.7).
    Slot::at(mmd::COMP_RESID).only_with_resid(),
];

/// Lighting parameters for the media pass (§6.3). The painting is lit by
/// image-based lighting from an [`Environment`]; this is a single place to tune the
/// look. A view setting — never historized (it changes how the canvas looks, not
/// its pixels).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MediaParams {
    /// Relief slope: how strongly the height field tilts normals (impasto/substrate).
    pub height_strength: f32,
    /// Paint glossiness in `[0,1]`: how smooth (low-roughness) the paint film is,
    /// driving the Cook–Torrance specular. 0 = matte; 1 = near mirror-smooth. It is
    /// a uniform property of paint — every texel with paint on it is equally glossy,
    /// ramped only by how much of the fragment *is* paint (its visible alpha), so
    /// the bare canvas behind it stays rough → matte.
    pub specular: f32,
    /// How strongly the canvas substrate relief shows (its substrate amplitude).
    pub substrate_strength: f32,
    /// Whether the display encode dithers its rounding (§6.5). On for every canvas
    /// a person looks at; a consumer measuring *paint* through the render — the
    /// suite's reference-identity configurations — turns it off, by the same
    /// argument that flattens the light there: the half-code of noise is
    /// deliberate on screen and a contaminant in a measurement.
    pub dither: bool,
}

impl Default for MediaParams {
    fn default() -> Self {
        Self {
            height_strength: 0.15,
            specular: 0.20,
            // The substrate is off until asked for: the default canvas is linen, and its
            // relief is there to be *painted into* (§6.2) whether or not the light is
            // made to show it. Raising this embosses it into the lit result.
            substrate_strength: 0.0,
            dither: true,
        }
    }
}

/// One code step of `format`'s encoding — the amplitude the media pass and the
/// resolve scale their quantization dither by (§6.5) — or 0 for a format deep
/// enough not to band, which turns the dither off.
///
/// Half a code either way (uniform), not the audio-style full-code triangle: a
/// value the encode lands exactly on a code — black, white, anything the tonemap
/// clamped — must render that code at every pixel rather than speckle its
/// neighbours, and ±half a step is the widest swing with that property
/// (`lib/noise.wesl::dither2` keeps it *strictly* inside the half, so the
/// rounding tie between two codes is unreachable). A deep target — an HDR surface,
/// or the f16 targets picks render into — gets 0 until it demonstrably bands,
/// because dither a target does not need is only noise.
pub(super) fn dither_step(format: wgpu::TextureFormat) -> f32 {
    if super::display::is_eight_bit(format) {
        1.0 / 255.0
    } else {
        0.0
    }
}

/// The media pass's bind group layout, shared by the pipeline compiled for each
/// target format ([`TargetPasses`](super::TargetPasses)) so a consumer's bind group
/// is valid against either.
pub(super) fn media_layout(
    device: &wgpu::Device,
    color_space: &dyn ColorSpace,
) -> wgpu::BindGroupLayout {
    desc::layout_for(
        device,
        "stark media bgl",
        MEDIA_SLOTS,
        wgpu::ShaderStages::FRAGMENT,
        color_space.has_resid(),
    )
}

/// The media pass — the pipeline, compiled for one target format.
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
}

impl MediaPass {
    pub(super) fn new(
        device: &wgpu::Device,
        color_space: &dyn ColorSpace,
        bgl: &wgpu::BindGroupLayout,
        target: &[Option<wgpu::ColorTargetState>],
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark media"),
            source: wgpu::ShaderSource::Wgsl(color_space.media_shader().into()),
        });
        let layout = desc::pipeline_layout(device, "stark media layout", &[Some(bgl)]);
        let pipeline = desc::fullscreen_pipeline(
            device,
            "stark media pipeline",
            &layout,
            &shader,
            ("vs_main", "fs_main"),
            target,
        );
        Self { pipeline }
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
            ..Default::default()
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
    /// Canvas px → substrate-tile uv for the substrate **the document is on**
    /// ([`Substrate::uv_scale`](crate::gpu::Substrate::uv_scale)) — how large the substrate is
    /// laid, as this frame is showing it.
    ///
    /// Off the document rather than off the bound map, and that is not a shortcut:
    /// the map's *height* channel — the only one this pass reads — is what the substrate
    /// is, and is the same field however large it is laid. All the scale-dependence
    /// of a bake is in the rise channels, which only the deposit reads. So the light
    /// can follow a scale the instant it changes, while the tooth waits for the bake
    /// the commit triggers — which is exactly what makes the slider previewable
    /// (`ViewCommand::PreviewSubstrateScale`) instead of costing a whole-image filter
    /// per pointer sample.
    pub(super) substrate_uv_scale: f32,
    /// The **supersampled** view, so the substrate and the relief are measured in the
    /// texels this pass is actually shading (§6.4).
    pub(super) view: ViewTransform,
    pub(super) substrate_color: [f32; 4],
    pub(super) substrate_resid: [f32; 4],
    /// Skip the substrate and carry the paint's visible alpha out, for a cut-out
    /// export (§15.6).
    pub(super) transparent: bool,
    /// One code of the target's encoding — [`dither_step`]'s answer for the format
    /// this pass stores into, whether that is the caller's target or the
    /// supersampled intermediate of the same format (§6.4) — already 0 when this
    /// consumer's [`MediaParams::dither`] asks for the undithered reference. The
    /// shader scales its store's quantization dither by it (§6.5).
    pub(super) dither_step: f32,
    /// The display this render is presented on (§6.5) — [`Output::SDR`] for every
    /// 8-bit target (`Compositor::render`).
    pub(super) output: Output,
}

impl MediaScene<'_> {
    /// The scene as the shader reads it — the one place these numbers become an ABI.
    fn uniform(&self) -> MediaUniform {
        let m = self.params;
        // Screen→canvas mapping for sampling the substrate bump in canvas space, so the
        // substrate stays attached to the canvas as it pans, zooms, turns and mirrors
        // (§6.4, §18.1.2).
        let canvas_origin = self.view.screen_to_canvas(stark_model::geom::Vec2::ZERO);
        // Diffuse samples a heavily-blurred high mip ≈ hemispherical irradiance; the
        // level is the environment's own, so this CPU-side normalization is reading
        // exactly the texels the shader will. The Cook–Torrance specular picks its own
        // mip from roughness, spanning the whole chain (roughness 0 → mip 0 sharp;
        // roughness 1 → the diffuse level, the hemispherical average).
        let diffuse_lod = self.environment.diffuse_lod as f32;
        // The one normalization the light gets: the reciprocal of the irradiance a
        // *flat* canvas receives, so an unrelieved patch of paint comes back out its
        // own color in every environment, procedural or HDR (§6.3). No light carries
        // an exposure of its own — every one was 1.0, and a knob whose every setting
        // is the same value is a knob that says nothing.
        let exposure = self.environment.flat_irradiance.recip();
        MediaUniform {
            bg: self.substrate_color,
            bg_resid: self.substrate_resid,
            shade: [exposure, diffuse_lod, m.specular, m.height_strength],
            view_a: [
                canvas_origin.x,
                canvas_origin.y,
                1.0 / self.view.zoom,
                self.substrate_uv_scale,
            ],
            sub_b: [
                m.substrate_strength,
                if self.transparent { 1.0 } else { 0.0 },
                self.dither_step,
                0.0,
            ],
            view_m: self.view.inverse_linear().to_cols_array(),
            out: [
                self.output.transfer().lane(),
                self.output.headroom(),
                0.0,
                0.0,
            ],
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
    /// The layout the media bind group is built against ([`media_layout`]).
    pub(super) media_bgl: &'a wgpu::BindGroupLayout,
    /// The consumer's own uniform buffer, which this bind group names.
    pub(super) media_buf: &'a wgpu::Buffer,
    pub(super) substrate: &'a SubstrateMap,
    pub(super) environment: &'a Environment,
}

/// The offscreen targets pass A writes and the media bind group that reads them.
pub(super) struct Offscreen {
    /// The accumulator itself — the same owned trio a scratch level holds
    /// ([`Trio`]), including its residual, which is present
    /// exactly when the space has one and is then pass A's third attachment as well as
    /// the media pass's binding 7.
    pub(super) channels: Trio,
    pub(super) bg: wgpu::BindGroup,
}

impl Offscreen {
    /// Point the media bind group at a different canvas substrate or light, keeping
    /// the attachments (§6.3, §6.4).
    ///
    /// **The whole cost of a swap.** Both are *bound into* this group and nothing
    /// else here names them, so a swapped substrate, a swapped light, a dragged
    /// substrate scale committing, or an undo across any of those costs one bind
    /// group — where rebuilding the `Offscreen` costs the accumulator trio, and with
    /// it (in the caller) the supersampled target and the blend scratch, up to
    /// `MAX_SUPERSAMPLED_BYTES` of allocation to say that a texture view moved.
    pub(super) fn rebind(
        &mut self,
        device: &wgpu::Device,
        media_bgl: &wgpu::BindGroupLayout,
        media_buf: &wgpu::Buffer,
        substrate: &SubstrateMap,
        environment: &Environment,
    ) {
        self.bg = media_bind_group(
            device,
            media_bgl,
            media_buf,
            substrate,
            environment,
            &self.channels,
        );
    }

    /// The trio as pass A attaches it (§6.7) — the same three views the media bind
    /// group above reads, which is the invariant this type exists to hold together.
    pub(super) fn targets(&self) -> crate::gpu::channels::Targets<'_> {
        self.channels.targets()
    }
}

/// (Re)create the offscreen composite targets and the media bind group over them.
pub(super) fn offscreen(d: OffscreenDesc<'_>) -> Offscreen {
    let OffscreenDesc {
        device,
        size,
        formats,
        media_bgl,
        media_buf,
        substrate,
        environment,
    } = d;
    let channels = Trio::new(
        device,
        size,
        ("stark comp color", "stark comp aux", "stark comp resid"),
        formats,
    );
    let bg = media_bind_group(
        device,
        media_bgl,
        media_buf,
        substrate,
        environment,
        &channels,
    );
    Offscreen { channels, bg }
}

/// The media pass's bind group over one accumulator, its uniform, the canvas
/// substrate and the light.
///
/// Split out because it is the **only** thing here that a substrate or a light swap
/// invalidates: the attachments are sized by the target and carry the color space's
/// formats, neither of which a swap touches. `Offscreen::rebind` is what that split
/// buys — see [`Compositor::ensure_targets`](super::Compositor), where rebuilding the
/// trio for a swap would cost a viewport of memory.
fn media_bind_group(
    device: &wgpu::Device,
    media_bgl: &wgpu::BindGroupLayout,
    media_buf: &wgpu::Buffer,
    substrate: &SubstrateMap,
    environment: &Environment,
    channels: &Trio,
) -> wgpu::BindGroup {
    let (color, aux, resid) = (&channels.color, &channels.aux, channels.resid.as_ref());
    desc::bind_group_for(
        device,
        "stark media bg",
        media_bgl,
        MEDIA_SLOTS,
        resid.is_some(),
        |i| match i {
            mc::M => media_buf.as_entire_binding(),
            mc::COMP_COLOR => wgpu::BindingResource::TextureView(color.view()),
            mc::COMP_AUX => wgpu::BindingResource::TextureView(aux.view()),
            mc::SUBSTRATE => wgpu::BindingResource::TextureView(&substrate.view),
            mc::SUBSTRATE_SAMP => wgpu::BindingResource::Sampler(&substrate.sampler),
            mc::ENV => wgpu::BindingResource::TextureView(&environment.view),
            mc::ENV_SAMP => wgpu::BindingResource::Sampler(&environment.sampler),
            mm::COMP_RESID => wgpu::BindingResource::TextureView(
                resid
                    .expect("a residual build has a composited residual")
                    .view(),
            ),
            other => unreachable!("`MEDIA_SLOTS` lists no binding {other}"),
        },
    )
}
