//! Pass B: the media/lighting pass (§6.3).
//!
//! One fullscreen draw: normals off the composited height field, image-based
//! lighting from an [`Environment`], the paint film's gloss, the working channels
//! converted to display, all composited over the substrate into the target.

use super::attachment::Trio;
use super::display::Output;
use crate::colorspace::{ColorSpace, Latent};
use crate::device::channels::{ChannelFormats, Targets};
use crate::device::context::GpuContext;
use crate::device::desc::{self, Bindings};
use crate::gpu::environment::Environment;
use crate::gpu::substrate::SubstrateMap;
use crate::view::{Extent2, ViewTransform};
use stark_shaders::Stages;
use stark_shaders::mirror::media::binding as mm;
use stark_shaders::mirror::media::decl as mmd;
use stark_shaders::mirror::media_common::binding as mc;
use stark_shaders::mirror::media_common::decl as mcd;

// Generated from `media_common.wesl`'s own declaration (§6.7).
pub(super) use stark_shaders::mirror::media_common::Media as MediaUniform;

/// Lighting parameters for the media pass (§6.3). A view setting — never
/// historized: it changes how the canvas looks, not its pixels.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MediaParams {
    /// Relief slope: how strongly the height field tilts normals (impasto/substrate).
    pub height_strength: f32,
    /// Paint glossiness in `[0,1]` driving the Cook–Torrance specular: 0 = matte,
    /// 1 = near mirror-smooth. A uniform property of paint, ramped only by how much
    /// of the fragment *is* paint (its visible alpha), so bare canvas stays matte.
    pub specular: f32,
    /// How strongly the canvas substrate relief shows (its substrate amplitude).
    pub substrate_strength: f32,
    /// Whether the display encode dithers its rounding (§6.5). On for anything a
    /// person looks at; off for a consumer measuring *paint* through the render,
    /// where the half-code of noise is a contaminant.
    pub dither: bool,
}

impl Default for MediaParams {
    fn default() -> Self {
        Self {
            height_strength: 0.15,
            specular: 0.20,
            // Off until asked for: the substrate's relief is there to be *painted
            // into* (§6.2) whether or not the light is made to show it. Raising this
            // embosses it into the lit result.
            substrate_strength: 0.0,
            dither: true,
        }
    }
}

/// One code step of `format`'s encoding — the amplitude the media pass and the
/// resolve scale their quantization dither by (§6.5) — or 0 for a format deep
/// enough not to band, which turns the dither off.
///
/// The dither is ±half a code (uniform), not a full-code triangle: a value landing
/// exactly on a code — black, white, anything the tonemap clamped — must render
/// that code at every pixel rather than speckle its neighbours, and half a step is
/// the widest swing with that property.
pub(super) fn dither_step(format: wgpu::TextureFormat) -> f32 {
    if super::display::is_eight_bit(format) {
        1.0 / 255.0
    } else {
        0.0
    }
}

/// The media pass's bind group layouts, shared by the pipeline compiled for each
/// target format ([`TargetPasses`](super::TargetPasses)) so a consumer's bind groups
/// are valid against either.
///
/// A group per module (§6.10): `media_common`'s whole `@group(0)`, then
/// `media.wesl`'s own `@group(1)` — the residual alone, and empty in a space that has
/// none (§6.7), so an Oklab document binds nothing there rather than a placeholder.
pub(super) fn media_layout(device: &wgpu::Device, color_space: &dyn ColorSpace) -> [Bindings; 2] {
    let media = color_space.media_shader();
    let stages = Stages::Render(media.vs_main, media.fs_main);
    [
        Bindings::of(device, "stark media bgl", stages, mcd::M, &[]),
        Bindings::if_reached(device, "stark media space bgl", &[stages], mmd::COMP_RESID),
    ]
}

/// The media pass — the pipeline, compiled for one target format.
///
/// The **uniform buffer** the lighting is written into is per consumer, not per
/// pipeline: two engines sharing one set of pipelines light their canvases differently
/// (a preset thumbnail deliberately does not mirror the canvas), so it lives on each
/// consumer's [`Accumulator`].
pub(super) struct MediaPass {
    pub(super) pipeline: wgpu::RenderPipeline,
}

impl MediaPass {
    /// The pipeline over `module`, `color_space`'s media shader compiled.
    pub(super) fn new(
        device: &wgpu::Device,
        color_space: &dyn ColorSpace,
        module: &desc::Module,
        bgls: &[Bindings; 2],
        target: &[Option<wgpu::ColorTargetState>],
    ) -> Self {
        let media = color_space.media_shader();
        let layout = desc::pipeline_layout_of(device, "stark media layout", &[&bgls[0], &bgls[1]]);
        let pipeline = desc::fullscreen_pipeline(
            device,
            "stark media pipeline",
            &layout,
            module,
            (media.vs_main, media.fs_main),
            target,
        );
        Self { pipeline }
    }

    /// Encode pass B: `accum`'s channels — normals off the composited height field,
    /// lit by the environment, tonemapped — over the substrate and into `target`
    /// (§6.3).
    pub(super) fn encode(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        accum: &Accumulator,
        target: &wgpu::TextureView,
        scene: MediaScene<'_>,
    ) {
        ctx.queue
            .write_buffer(&accum.uniform, 0, bytemuck::bytes_of(&scene.uniform()));
        // The pass covers every texel, so the clear only decides the alpha an
        // untouched texel keeps — transparent, for a cut-out export.
        let clear = if scene.transparent {
            wgpu::Color::TRANSPARENT
        } else {
            wgpu::Color::BLACK
        };
        desc::fullscreen_pass(
            encoder,
            "stark media pass",
            desc::Attachments::cleared_to(target, clear),
            &self.pipeline,
            desc::Bound::new(&accum.bg, &[]),
        );
    }
}

/// The substrate as the media pass's `bg` and `bg_resid` lanes (§15.5, §6.7) — and the
/// eyedropper's, which composites a patch over the same numbers.
///
/// Three channels and no alpha in either: the substrate is opaque paint under
/// everything, so both sides write full coverage as a literal rather than reading it
/// from a lane.
pub(crate) fn substrate_lanes(substrate: Latent) -> [[f32; 3]; 2] {
    let Latent { lat, res } = substrate;
    [lat, res]
}

/// What pass B needs beyond the accumulator: how the frame is lit, what is under the
/// paint, and where the canvas is (§6.3).
pub(super) struct MediaScene<'a> {
    pub(super) params: MediaParams,
    pub(super) environment: &'a Environment,
    /// Canvas px → substrate-tile uv for the substrate **the document is on**
    /// ([`Substrate::uv_scale`](crate::document::Substrate::uv_scale)), as this frame is
    /// showing it.
    ///
    /// Taken off the document rather than the bound map: the map's *height* channel —
    /// the only one this pass reads — is the same field however large it is laid, so
    /// the light can follow a scale change immediately while the tooth waits for the
    /// bake a commit triggers. That is what makes the scale slider previewable.
    pub(super) substrate_uv_scale: f32,
    /// The **supersampled** view, so the substrate and the relief are measured in the
    /// texels this pass is actually shading (§6.4).
    pub(super) view: ViewTransform,
    /// The substrate color in the document's working channels.
    pub(super) substrate: Latent,
    /// Skip the substrate and carry the paint's visible alpha out, for a cut-out
    /// export (§15.6).
    pub(super) transparent: bool,
    /// One code of the target's encoding — [`dither_step`] for the format this pass
    /// stores into, which is the supersampled intermediate when there is one (§6.4),
    /// and 0 when [`MediaParams::dither`] asks for the undithered reference. The
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
        // Diffuse samples a heavily-blurred high mip ≈ hemispherical irradiance. The
        // Cook–Torrance specular picks its own mip from roughness, spanning the whole
        // chain (roughness 0 → mip 0 sharp; roughness 1 → this level).
        let diffuse_lod = self.environment.diffuse_lod as f32;
        // The one normalization the light gets: the reciprocal of the irradiance a
        // *flat* canvas receives, so an unrelieved patch of paint comes back out its
        // own color in every environment, procedural or HDR (§6.3).
        let exposure = self.environment.flat_irradiance.recip();
        let [bg, bg_resid] = substrate_lanes(self.substrate);
        MediaUniform {
            bg,
            bg_resid,
            exposure,
            diffuse_lod,
            gloss: m.specular,
            height_strength: m.height_strength,
            canvas_origin: canvas_origin.to_array(),
            canvas_per_px: 1.0 / self.view.zoom,
            inv_tile: self.substrate_uv_scale,
            substrate_strength: m.substrate_strength,
            transparent: f32::from(u8::from(self.transparent)),
            dither_step: self.dither_step,
            view_m: self.view.inverse_linear().to_cols_array(),
            transfer: self.output.transfer().lane(),
            headroom: self.output.headroom(),
            ..Default::default()
        }
    }
}

/// Pass A's accumulator and what the media pass reads it through: the channels, this
/// consumer's own uniform ([`MediaPass`] says why it is not the pass's), and the bind
/// group naming both.
///
/// One value because the three are meaningless apart — the group names the other two,
/// and [`MediaPass::encode`] must write the very buffer it binds. [`Supersampled`] is
/// the same arrangement for the resolve.
///
/// [`Supersampled`]: super::resolve::Supersampled
pub(super) struct Accumulator {
    /// The residual is present exactly when the space has one, and is then pass A's
    /// third attachment as well as a binding here.
    channels: Trio,
    uniform: wgpu::Buffer,
    /// One per layout: `media_common`'s group, then the space's, which is empty where
    /// the space has no residual (§6.7).
    bg: [wgpu::BindGroup; 2],
}

impl Accumulator {
    /// Channels of `formats` at `size`, and the media bind group over them, the
    /// canvas substrate and the light.
    pub(super) fn new(
        device: &wgpu::Device,
        size: Extent2,
        formats: ChannelFormats,
        bgl: &[Bindings; 2],
        substrate: &SubstrateMap,
        environment: &Environment,
    ) -> Self {
        let channels = Trio::new(
            device,
            size,
            ("stark comp color", "stark comp aux", "stark comp resid"),
            formats,
        );
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stark media uniform"),
            size: std::mem::size_of::<MediaUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg = bind_group(device, bgl, &channels, &uniform, substrate, environment);
        Self {
            channels,
            uniform,
            bg,
        }
    }

    /// Point the media bind group at a different canvas substrate or light, keeping
    /// the channels (§6.3, §6.4).
    ///
    /// **The whole cost of a swap**: the channels are sized by the target and carry the
    /// color space's formats, neither of which a swap touches, so rebuilding the
    /// accumulator instead would reallocate them — and, in the caller, the supersampled
    /// target and blend scratch beside them — for nothing.
    pub(super) fn rebind(
        &mut self,
        device: &wgpu::Device,
        bgl: &[Bindings; 2],
        substrate: &SubstrateMap,
        environment: &Environment,
    ) {
        self.bg = bind_group(
            device,
            bgl,
            &self.channels,
            &self.uniform,
            substrate,
            environment,
        );
    }

    /// The channels as pass A attaches them (§6.7) — the views the media bind group
    /// reads.
    pub(super) fn targets(&self) -> Targets<'_> {
        self.channels.targets()
    }
}

/// The media pass's bind groups over one accumulator's `channels`, its `uniform`, the
/// canvas substrate and the light.
fn bind_group(
    device: &wgpu::Device,
    bgl: &[Bindings; 2],
    channels: &Trio,
    uniform: &wgpu::Buffer,
    substrate: &SubstrateMap,
    environment: &Environment,
) -> [wgpu::BindGroup; 2] {
    let (color, aux, resid) = (&channels.color, &channels.aux, channels.resid.as_ref());
    [
        bgl[0].group(device, "stark media bg", |i| match i {
            mc::M => uniform.as_entire_binding(),
            mc::COMP_COLOR => wgpu::BindingResource::TextureView(color.view()),
            mc::COMP_AUX => wgpu::BindingResource::TextureView(aux.view()),
            mc::SUBSTRATE => wgpu::BindingResource::TextureView(&substrate.view),
            mc::SUBSTRATE_SAMP => wgpu::BindingResource::Sampler(&substrate.sampler),
            mc::ENV => wgpu::BindingResource::TextureView(&environment.view),
            mc::ENV_SAMP => wgpu::BindingResource::Sampler(&environment.sampler),
            other => unreachable!("`media_common`'s group has no binding {other}"),
        }),
        // Empty in a space with no residual, where the closure is asked for nothing.
        bgl[1].group(device, "stark media space bg", |i| match i {
            mm::COMP_RESID => wgpu::BindingResource::TextureView(
                resid
                    .expect("a residual build has a composited residual")
                    .view(),
            ),
            other => unreachable!("`media.wesl`'s own group has no binding {other}"),
        }),
    ]
}
