//! The focal blur's convolution, driven (§21.12).
//!
//! `blur.wesl` holds the arithmetic — the Stockham passes, the aperture, the
//! frequency-domain multiply — and this module holds everything the arithmetic
//! cannot say about itself: how large the padded transform is, which pass reads
//! which half of the ping-pong, when the kernel's transform is stale, and where
//! the round trip must land. All of that is decided in [`BlurPass::prepare`]
//! as **plain data** ([`plan`]), for the compositor's own reason (§14.7's one
//! walk): the encoder then replays a list, and the property everything rests on —
//! the inverse transform lands in the pair the filter bind group names — is a
//! fact about the list, testable without an adapter.
//!
//! # The shape of one frame's work
//!
//! Per focal-blur layer, in encoder order (which is execution order):
//!
//! 1. **The kernel**, only when what the kernel texture holds is not this
//!    layer's aperture at this layer's radius and extent: `make_kernel`
//!    rasterizes the shape into a scratch plane and `fft_one` walks it into the
//!    dedicated kernel texture. Cached across frames — a settled bokeh costs
//!    nothing here — and rebuilt mid-frame when two blur layers disagree.
//! 2. **The decode**: the space's own `fs_blur_decode` (`FilterPass::blur_decode`)
//!    lays the accumulator into set **B** as premultiplied XYZ light, coverage,
//!    height and the border weight, zeros in the padding. It binds the very
//!    filter bind group the resolve pass will — which is what lets it read the
//!    right accumulator half without machinery of its own — and set B is the
//!    half that group does *not* name, so nothing is both bound and attached.
//! 3. **The round trip**: forward FFT, multiply, inverse FFT — `2·(log₂W+log₂H)+1`
//!    dispatches, each reading one half of the ping-pong and writing the other.
//!    The count is odd *whatever the sizes*, so a chain that starts in B always
//!    lands in **A** — the pair the filter bind group reads — with no copy and
//!    no parity to get wrong.
//!
//! The filter's own fullscreen pass then runs as every filter's does (§21.3),
//! with its `FILTER_FOCAL_BLUR` arm reading set A.
//!
//! # Memory, and what bounds it
//!
//! The planes are `f32` complex — three planes across an `rgba32float` and an
//! `rg32float`, two sets plus the kernel — at the padded power-of-two size, which
//! is the largest scratch the application makes and the term
//! [`resolve::attachment_bytes`](super::resolve::attachment_bytes) charges the
//! supersampling budget for. `f16` is not an option the precision argument loses
//! narrowly: a transform's DC term is the *sum* of the image, which overflows
//! half floats at any real size.
//!
//! A radius is a view-mapped quantity — the same document blurs by the same
//! **canvas** distance at every zoom (§6.4) — which is exactly what once made
//! zooming in fatal: the guard band grew with the zoom, and the planes grew
//! past the device's memory and buffer limits with it. What bounds them now is
//! [`scale`]: past [`MAX_CONV_RADIUS`] on-screen texels a layer **decimates**
//! rather than growing, its transform running at a power-of-two fraction of the
//! accumulator's resolution — the chromatic tap cap's own trade (§21.10),
//! degrade the sampling and never the picture's geometry, and invisible where
//! it is legal because an aperture that wide carries nothing near texel frequency.
//! The decode averages down and the resolve interpolates back up
//! (`filter_common.wesl`'s `blur_src` / `blur_read`); the transform between
//! them is scale-blind. The device's texture limit stays as the last-resort
//! clamp, and only there does the radius itself give way ([`layers`]).

use std::ops::Range;

use crate::gpu::context::GpuContext;
use crate::gpu::desc::{self, Slot};
use crate::gpu::uniforms::UniformSlots;
use crate::view::{Extent2, ViewTransform};
use stark_shaders::mirror::blur::binding as bb;
use stark_shaders::mirror::blur::decl as bd;
use stark_shaders::mirror::blur::{BLUR_WG, Fft as FftUniform};

use super::filter::FilterPass;
use super::group::FilterDraw;

/// The FFT group's bindings, in layout order (§6.10): the per-dispatch plan, the
/// half of the ping-pong being read, the kernel's transform, and the half being
/// written.
const BLUR_SLOTS: &[Slot] = &[
    Slot::dynamic(bd::F),
    Slot::at(bd::SRC_LIGHT),
    Slot::at(bd::SRC_AUX),
    Slot::at(bd::KERNEL),
    Slot::at(bd::DST_LIGHT),
    Slot::at(bd::DST_AUX),
];

/// The blur's compute pipelines — one module, four entry points — built once per
/// pipeline kit and shared by every consumer's [`BlurFrame`].
pub(crate) struct BlurPass {
    fft_both: wgpu::ComputePipeline,
    fft_one: wgpu::ComputePipeline,
    make_kernel: wgpu::ComputePipeline,
    apply_kernel: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl BlurPass {
    pub(crate) fn new(ctx: &GpuContext) -> Self {
        let device = &ctx.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark blur"),
            source: wgpu::ShaderSource::Wgsl(stark_shaders::blur().into()),
        });
        let bgl = desc::layout_for(
            device,
            "stark blur bgl",
            BLUR_SLOTS,
            wgpu::ShaderStages::COMPUTE,
            false,
        );
        let layout = desc::pipeline_layout(device, "stark blur layout", &[Some(&bgl)]);
        let cpipe = |label: &str, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            fft_both: cpipe("stark blur fft", "fft_both"),
            fft_one: cpipe("stark blur kernel fft", "fft_one"),
            make_kernel: cpipe("stark blur make kernel", "make_kernel"),
            apply_kernel: cpipe("stark blur apply kernel", "apply_kernel"),
            bgl,
        }
    }

    /// Bring `frame` in line with what this render is about to blur: the planes
    /// for an accumulator of `accum`, the dispatch plan for `kernels` (one
    /// `(filter slot, radius in accumulator texels, aperture)` per focal-blur
    /// layer, in slot order), and every per-dispatch uniform written. Empty
    /// `kernels` drops the frame whole — the planes are the largest scratch there
    /// is, and a document that stops blurring should stop paying for it.
    ///
    /// **Returns whether the planes the filter bind group names changed** —
    /// created, resized, or dropped — because those groups are cached per scratch
    /// level (`ScratchLevel::filter_bg`) and the caller must invalidate them.
    /// A moved uniform buffer is *not* part of that answer: the only groups
    /// naming it are this module's own, rebuilt here.
    pub(crate) fn prepare(
        &self,
        ctx: &GpuContext,
        frame: &mut Option<BlurFrame>,
        accum: Extent2,
        kernels: &[(u32, f32, Aperture)],
        max_dim: u32,
    ) -> bool {
        if kernels.is_empty() {
            return frame.take().is_some();
        }
        let layers = layers(accum, kernels, max_dim);
        // The planes hold the largest layer's transform; a smaller layer runs on
        // a subregion at their origin, its extent riding its own uniforms.
        let planes = layers.iter().fold(Extent2::new(2, 2), |m, l| {
            Extent2::new(m.width.max(l.conv.width), m.height.max(l.conv.height))
        });
        let rebuilt = !frame
            .as_ref()
            .is_some_and(|f| f.planes == planes && f.accum == accum);
        if rebuilt {
            // Dropped before the replacement is built, for `ensure_targets`'
            // reason: these are the largest allocations the application makes,
            // and holding both sets across the build doubles the peak.
            *frame = None;
            *frame = Some(BlurFrame::new(ctx, planes, accum));
        }
        let frame = frame.as_mut().expect("just ensured");

        let (uniforms, dispatches, jobs, kernel_key) = plan(&layers, frame.kernel_key);
        let moved = frame.uniforms.write(&ctx.device, &ctx.queue, &uniforms);
        frame.dispatches = dispatches;
        frame.jobs = jobs;
        // What the kernel texture will hold once this frame's encoder has run —
        // recorded now, because the encode path is `&self` (§14.7's plan/encode
        // split, made here for the same reason).
        frame.kernel_key = kernel_key;
        if rebuilt || moved {
            frame.build_binds(&ctx.device, &self.bgl);
        }
        rebuilt
    }
}

/// A focal-blur layer's canvas radius as accumulator texels: the view's linear
/// map applied — the chromatic dispersion's own trip (§21.10), for the same
/// reason. The bokeh belongs to the artwork, so it scales with the zoom and
/// holds its size in an export.
pub(super) fn texel_radius(f: &FilterDraw, view: ViewTransform) -> f32 {
    (view.linear() * stark_model::geom::Vec2::new(f.params[0], 0.0)).length()
}

/// The aperture's shape as `make_kernel` reads it (§21.12) — the shader's own
/// `APERTURE_*` code, the shape's one number, and its turn.
///
/// Three lanes rather than the document's [`Aperture`] enum for
/// [`FilterDraw`]'s reason: by the time a filter reaches the compositor it is
/// numbers, and which number means which shape is a fact about `blur.wesl`
/// rather than about the document.
///
/// `shape` and `param` are **scale-free** — a code, a blade count, a fraction of
/// the radius, a ratio — which is what lets a decimated layer (§21.12) rescale
/// its radius and carry them through untouched. `angle` is not: it is the one
/// lane that has already made a trip, from the canvas frame the document states
/// it in into this one ([`aperture`]).
///
/// [`Aperture`]: stark_model::document::Aperture
#[derive(Copy, Clone, PartialEq, Debug)]
pub(super) struct Aperture {
    shape: u32,
    param: f32,
    /// The aperture's turn **in the accumulator's frame**, not the canvas's — see
    /// [`aperture`]. Zero for a shape that has no direction, and held there, since
    /// this keys the kernel cache.
    angle: f32,
}

/// The aperture a focal-blur draw carries, off the lanes `group::aperture_lanes`
/// wrote — with its turn carried into the frame the convolution runs in.
///
/// **The turn makes the same trip the radius does** (§21.10, §6.4), and for the
/// same reason the chromatic filter's angle does (`plan::view_lanes`): the bokeh
/// belongs to the artwork, so a six-bladed iris turns when the canvas is turned
/// and flips when it is mirrored, and an export at a rotated view shows the
/// picture the screen showed. Without it the polygon stays welded to the screen
/// while the painting rotates under it.
///
/// [`ViewTransform::orientation`] rather than the full `linear()` because the
/// zoom is the *radius*' half of the map and cancels out of a direction anyway;
/// the mirror is not optional, or a flipped view would leave a turned iris
/// unflipped.
fn aperture(f: &FilterDraw, view: ViewTransform) -> Aperture {
    let shape = f.params[1] as u32;
    let canvas = f.params[3];
    let turned = view.orientation() * stark_model::geom::Vec2::new(canvas.cos(), canvas.sin());
    let angle = turned.y.atan2(turned.x);
    Aperture {
        shape,
        param: f.params[2],
        // Only a shape that *has* a direction takes the trip. A disc's lane is zero
        // and has to stay zero — turning it would key a new kernel on every frame of
        // a canvas rotation, for a shape that cannot tell one turn from another.
        // Non-finite lands on zero for `layers`' reason: the view multiplies this,
        // and the arithmetic has to hold whatever a hostile one does.
        angle: if shape == stark_shaders::mirror::blur::APERTURE_DISC || !angle.is_finite() {
            0.0
        } else {
            angle
        },
    }
}

/// What each focal-blur layer asks the convolution for this frame, in filter-slot
/// order: its slot, its radius in accumulator texels, and its aperture.
pub(super) fn blur_kernels(
    filters: &[&FilterDraw],
    view: ViewTransform,
) -> Vec<(u32, f32, Aperture)> {
    filters
        .iter()
        .enumerate()
        .filter(|(_, f)| f.kind == stark_shaders::mirror::filter_common::FILTER_FOCAL_BLUR)
        .map(|(slot, f)| (slot as u32, texel_radius(f, view), aperture(f, view)))
        .collect()
}

/// The widest radius the convolution runs at full resolution, in convolution
/// texels — past it the layer decimates instead (§21.12, [`scale`]).
///
/// The number is the document knob's own ceiling ([`FocalBlur::RADIUS`]), which
/// makes the statement exact: **at any zoom up to 1:1 the convolution is never
/// decimated**, and past 1:1 it decimates only once the on-screen radius has
/// outgrown the widest blur the knob can ask for. An aperture this many texels
/// wide carries nothing near texel frequency, so halving the resolution under it
/// is invisible where it is legal — which is the same trade the chromatic
/// filter's tap cap makes, chosen the same way round: degrade the *sampling*,
/// never the picture's own geometry.
///
/// "This many texels wide" is a claim about the *thinnest* part of a shape, not
/// about its span, which is why [`Aperture::OBSTRUCTION`] has a ceiling: a ring's
/// rim is a tenth of its radius at the widest obstruction the document will hold,
/// so even that shape is a dozen texels thick where the bound bites.
///
/// [`Aperture::OBSTRUCTION`]: stark_model::document::Aperture::OBSTRUCTION
/// [`FocalBlur::RADIUS`]: stark_model::document::FocalBlur::RADIUS
const MAX_CONV_RADIUS: f32 = 128.0;

/// The decimation scale for an on-screen radius of `r_texels`: the smallest
/// power of two that brings the convolution-space radius inside
/// [`MAX_CONV_RADIUS`]. 1 — no decimation — at every working zoom.
///
/// This is what bounds the FFT (§21.12): without it, zooming into a blur grows
/// the guard band with the zoom, and the padded planes balloon past the device's
/// buffer and memory limits — the transform would be spending gigabytes to
/// resolve texel-scale detail that an aperture hundreds of texels wide provably
/// cannot contain. The floor keeps a hostile radius (a non-finite zoom, a log
/// this engine did not write) from looping; past it the extent clamp in
/// [`layers`] absorbs what is left.
pub(super) fn scale(r_texels: f32) -> u32 {
    let mut s = 1u32;
    while s < (1 << 20) && r_texels / s as f32 > MAX_CONV_RADIUS {
        s *= 2;
    }
    s
}

/// Whether any of this plan's filters is a focal blur — what decides both the
/// scratch charge ([`resolve::attachment_bytes`](super::resolve::attachment_bytes))
/// and whether a frame is prepared at all.
pub(super) fn has_blur(filters: &[&FilterDraw]) -> bool {
    filters
        .iter()
        .any(|f| f.kind == stark_shaders::mirror::filter_common::FILTER_FOCAL_BLUR)
}

/// One texture of one plane pair — [`Attachment`](super::attachment::Attachment)
/// with the storage usage the FFT writes through, and the same destroy-on-drop
/// (these are bigger than anything that type owns).
struct Plane {
    tex: wgpu::Texture,
    view: wgpu::TextureView,
}

impl Plane {
    fn new(
        device: &wgpu::Device,
        size: Extent2,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
        label: &str,
    ) -> Self {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        Self { tex, view }
    }
}

impl Drop for Plane {
    fn drop(&mut self) {
        self.tex.destroy();
    }
}

/// Which compute pipeline one dispatch runs.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Pipe {
    FftBoth,
    FftOne,
    MakeKernel,
    ApplyKernel,
}

/// Which cached bind group one dispatch reads through: the source set, and
/// whether the destination is the other set or the kernel texture. Four groups
/// per frame, however many dispatches — the ping-pong has two phases and the
/// kernel chain two endings, exactly as the blend scratch keeps two per level
/// (`ScratchLevel::blend_bg`).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Binds {
    AToB,
    BToA,
    /// The kernel chain's last pass: reads a set, writes the kernel texture.
    /// The unused destination and kernel-texture slots are filled with planes
    /// that are safely elsewhere this dispatch — a slot may not be bound as both
    /// texture and storage in one usage scope.
    AToKernel,
    BToKernel,
}

/// One compute dispatch, planned: everything `encode` replays.
#[derive(Copy, Clone, Debug)]
struct Dispatch {
    pipe: Pipe,
    binds: Binds,
    /// The dynamic-offset slot of this dispatch's [`FftUniform`] — its own index,
    /// since the two lists are built in one loop.
    slot: u32,
    groups: (u32, u32),
}

/// One focal-blur layer's share of the frame: which filter slot it serves, and
/// its two runs of [`Dispatch`]es — the kernel work it owes (often empty), and
/// the image's round trip. Two ranges rather than one because the decode render
/// pass has to be encoded between them.
struct Job {
    slot: u32,
    kernel: Range<usize>,
    image: Range<usize>,
}

/// One consumer's blur scratch and plan: the padded plane sets, the kernel, the
/// per-dispatch uniforms, and the dispatch list [`BlurPass::prepare`] decided.
/// On the [`Compositor`](super::Compositor) for the screen and in a
/// [`PreparedPick`](super::PreparedPick) for the eyedropper, sized by each one's
/// own accumulator — the blend scratch's split, for the blend scratch's reason.
pub(crate) struct BlurFrame {
    /// The plane textures' extent: the largest layer's convolution size this
    /// frame. A smaller layer transforms a subregion at the origin.
    planes: Extent2,
    /// The accumulator extent the plan was built against — part of the reuse
    /// check because the decode's border weight is a function of it: the same
    /// plane size over a different accumulator is a different `w` field.
    accum: Extent2,
    /// Set **A**: where the inverse transform lands, and what the filter bind
    /// group reads (`filter_common.wesl`'s `blur_light` / `blur_aux`).
    a_light: Plane,
    a_aux: Plane,
    /// Set **B**: the other half of the ping-pong, and the decode's target.
    b_light: Plane,
    b_aux: Plane,
    /// The kernel's transform, kept across frames while the bokeh holds still.
    kern: Plane,
    /// What `kern` holds (after this frame's encoded work runs) — the aperture
    /// and radius, **and the extent it was transformed at**, since one shape at two
    /// convolution sizes is two different spectra — or `None` while it holds
    /// nothing.
    kernel_key: Option<KernelKey>,
    uniforms: UniformSlots<FftUniform>,
    /// The four bind groups every dispatch selects among — rebuilt when the
    /// planes or the uniform buffer are replaced, never per dispatch.
    binds: Option<[wgpu::BindGroup; 4]>,
    dispatches: Vec<Dispatch>,
    jobs: Vec<Job>,
}

impl BlurFrame {
    fn new(ctx: &GpuContext, planes: Extent2, accum: Extent2) -> Self {
        let device = &ctx.device;
        let light_format = bd::DST_LIGHT.storage_format();
        let aux_format = bd::DST_AUX.storage_format();
        // The sets are decoded into (render), transformed through (storage), and
        // resolved from (texture). The kernel is never rendered to, but it wears
        // `RENDER_ATTACHMENT` anyway, and the flag is load-bearing: a texture
        // without it can only be zero-initialized through a staging **buffer**
        // of its full size, and Dawn refuses one past `maxBufferSize` — a
        // validation failure at submit, on the frame's whole command buffer.
        // Renderable, the lazy clear is a render pass and costs no buffer.
        let usage = wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::RENDER_ATTACHMENT;
        Self {
            planes,
            accum,
            a_light: Plane::new(device, planes, light_format, usage, "stark blur a light"),
            a_aux: Plane::new(device, planes, aux_format, usage, "stark blur a aux"),
            b_light: Plane::new(device, planes, light_format, usage, "stark blur b light"),
            b_aux: Plane::new(device, planes, aux_format, usage, "stark blur b aux"),
            kern: Plane::new(device, planes, aux_format, usage, "stark blur kernel"),
            kernel_key: None,
            uniforms: UniformSlots::new(&ctx.device, "stark blur fft uniform", 1),
            binds: None,
            dispatches: Vec::new(),
            jobs: Vec::new(),
        }
    }

    /// The planes the filter pass reads — what
    /// [`FilterPass::bind_group`](super::filter::FilterPass::bind_group) binds in
    /// place of its 1×1 stand-ins while a frame exists.
    pub(super) fn planes(&self) -> (&wgpu::TextureView, &wgpu::TextureView) {
        (&self.a_light.view, &self.a_aux.view)
    }

    /// The four bind groups of [`Binds`], over the current planes and uniforms.
    fn build_binds(&mut self, device: &wgpu::Device, bgl: &wgpu::BindGroupLayout) {
        let make = |label: &str,
                    src: (&wgpu::TextureView, &wgpu::TextureView),
                    kern: &wgpu::TextureView,
                    dst: (&wgpu::TextureView, &wgpu::TextureView)| {
            desc::bind_group_for(device, label, bgl, BLUR_SLOTS, false, |i| match i {
                bb::F => self.uniforms.resource(),
                bb::SRC_LIGHT => wgpu::BindingResource::TextureView(src.0),
                bb::SRC_AUX => wgpu::BindingResource::TextureView(src.1),
                bb::KERNEL => wgpu::BindingResource::TextureView(kern),
                bb::DST_LIGHT => wgpu::BindingResource::TextureView(dst.0),
                bb::DST_AUX => wgpu::BindingResource::TextureView(dst.1),
                other => unreachable!("`BLUR_SLOTS` lists no binding {other}"),
            })
        };
        let a = (&self.a_light.view, &self.a_aux.view);
        let b = (&self.b_light.view, &self.b_aux.view);
        // The kernel-destination groups stand in for the slots their dispatches
        // never touch with planes that are only *read* in them: the kernel slot
        // takes the source's own light plane (bound twice as a texture, which is
        // two reads of one thing), and the light destination takes the other
        // set's — written by nothing in a `fft_one` dispatch, and a storage
        // binding an entry point does not store to is merely a binding.
        self.binds = Some([
            make("stark blur a→b", a, &self.kern.view, b),
            make("stark blur b→a", b, &self.kern.view, a),
            make("stark blur a→kern", a, a.0, (b.0, &self.kern.view)),
            make("stark blur b→kern", b, b.0, (a.0, &self.kern.view)),
        ]);
    }

    /// Encode the convolution for the focal blur at filter `slot`: the kernel
    /// passes it still owes, the decode of the accumulator `bg` names, and the
    /// FFT round trip — everything but the filter's own fullscreen pass, which
    /// the caller encodes next and which reads set A.
    ///
    /// `bg` is the **filter bind group** of the bounce this blur is part of, and
    /// `offset` its layer's dynamic-offset slot: the decode pass binds the very
    /// group the resolve pass will, which is how it reads the right half of the
    /// accumulator's ping-pong without a description of its own.
    pub(super) fn encode(
        &self,
        pass: &BlurPass,
        filter: &FilterPass,
        encoder: &mut wgpu::CommandEncoder,
        bg: &wgpu::BindGroup,
        offset: u32,
        slot: u32,
    ) {
        let job = self
            .jobs
            .iter()
            .find(|j| j.slot == slot)
            .expect("a focal blur the frame was not prepared for (BlurPass::prepare)");
        if !job.kernel.is_empty() {
            self.dispatch(pass, encoder, "stark blur kernel", job.kernel.clone());
        }
        // The decode: the accumulator into set B, zeros in the padding. Set A —
        // which `bg` binds — is not attached, so nothing is both read and
        // written; the pass covers the whole padded extent, so the clear is a
        // don't-care stated as one.
        {
            let attachments = [
                Some(desc::attach(&self.b_light.view, desc::CLEAR)),
                Some(desc::attach(&self.b_aux.view, desc::CLEAR)),
            ];
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stark blur decode"),
                color_attachments: &attachments,
                ..Default::default()
            });
            rp.set_pipeline(&filter.blur_decode);
            rp.set_bind_group(0, bg, &[offset]);
            rp.draw(0..3, 0..1);
        }
        self.dispatch(pass, encoder, "stark blur fft", job.image.clone());
    }

    /// Replay one run of the plan as a compute pass. Dispatches in one pass are
    /// each their own usage scope, so the ping-pong's read-after-write hazards
    /// are the driver's to fence — the plan only has to alternate.
    fn dispatch(
        &self,
        pass: &BlurPass,
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
        range: Range<usize>,
    ) {
        let binds = self
            .binds
            .as_ref()
            .expect("prepare builds the bind groups before anything encodes");
        let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        for d in &self.dispatches[range] {
            cp.set_pipeline(match d.pipe {
                Pipe::FftBoth => &pass.fft_both,
                Pipe::FftOne => &pass.fft_one,
                Pipe::MakeKernel => &pass.make_kernel,
                Pipe::ApplyKernel => &pass.apply_kernel,
            });
            let bind = &binds[match d.binds {
                Binds::AToB => 0,
                Binds::BToA => 1,
                Binds::AToKernel => 2,
                Binds::BToKernel => 3,
            }];
            cp.set_bind_group(0, bind, &[UniformSlots::<FftUniform>::offset(d.slot)]);
            cp.dispatch_workgroups(d.groups.0, d.groups.1, 1);
        }
    }
}

/// What the kernel texture holds: an aperture at a radius, at the convolution
/// extent it was transformed at — all three, since the same shape at two radii or
/// two sizes is two different spectra.
type KernelKey = (Aperture, f32, Extent2);

/// One focal-blur layer's transform, decided: its filter slot, its decimation
/// [`scale`], its radius in **convolution texels**, the aperture rasterized at
/// that radius, and the power-of-two extent its chain runs on.
#[derive(Copy, Clone, Debug)]
struct Layer {
    slot: u32,
    conv: Extent2,
    /// The radius in convolution texels, after the extent clamp below — what
    /// `make_kernel` rasterizes, and half of what keys the kernel cache.
    radius: f32,
    /// The shape rasterized at that radius — carried through the decimation
    /// untouched, since every lane of it is scale-free ([`Aperture`]).
    aperture: Aperture,
}

/// Decide every layer's transform: per layer, decimate by [`scale`], then take
/// each axis to the next power of two past the decimated image plus a guard
/// band of the radius on both sides — what keeps the circular convolution's
/// wrap-around in zeroed padding.
///
/// The scale is what bounds this: the guard is at most [`MAX_CONV_RADIUS`]
/// convolution texels however hard the view zooms, so an extent never outgrows
/// the accumulator's own power of two by more than a fixed band. The device's
/// texture limit stays as the fallback of last resort — if it still binds
/// (an accumulator near the limit on a device whose cap is not a power of two),
/// the **radius** gives way to the guard the clamp left, a smaller blur rather
/// than a wrapped one.
fn layers(accum: Extent2, kernels: &[(u32, f32, Aperture)], max_dim: u32) -> Vec<Layer> {
    // The largest power of two the device can hold. Every real adapter's limit
    // is itself one; the floor is for a hypothetical that is not.
    let cap = if max_dim.is_power_of_two() {
        max_dim
    } else {
        (max_dim / 2).max(2).next_power_of_two()
    };
    kernels
        .iter()
        .map(|&(slot, r_texels, aperture)| {
            // The document's radius is sanitized, but the *view* multiplies it
            // (`texel_radius`), and this arithmetic must hold whatever a zoom
            // does: a non-finite product decimates to a point rather than
            // saturating the extent sums below.
            let r_texels = if r_texels.is_finite() {
                r_texels.max(0.0)
            } else {
                0.0
            };
            let s = scale(r_texels);
            let r = r_texels / s as f32;
            let guard = r.ceil() as u32;
            let axis = |d: u32| {
                let scaled = d.div_ceil(s).max(2);
                let base = scaled.next_power_of_two();
                (scaled + 2 * guard)
                    .next_power_of_two()
                    .clamp(base, cap.max(base))
            };
            let conv = Extent2::new(axis(accum.width), axis(accum.height));
            // The guard the clamp actually left, per axis; the radius yields to
            // the smaller one.
            let left = |c: u32, d: u32| (c - d.div_ceil(s).max(2).min(c)) / 2;
            let radius = r
                .min(left(conv.width, accum.width) as f32)
                .min(left(conv.height, accum.height) as f32);
            Layer {
                slot,
                conv,
                radius,
                aperture,
            }
        })
        .collect()
}

/// Decide one frame's whole dispatch plan, as data: the per-dispatch uniforms
/// (index `i` is slot `i`), the dispatches, the per-layer [`Job`]s, and what the
/// kernel texture holds when the plan has run.
///
/// `kern_holds` is what it holds *now* — a layer whose aperture, convolution
/// radius and extent all match owes no kernel work, which is every settled frame;
/// two layers that disagree in any of the three rebuild it between them, which is
/// the cost of one kernel texture rather than one per layer. The extent is part of
/// the key because it is part of the spectrum: one shape transformed at two sizes
/// is two different tables of frequencies.
fn plan(
    layers: &[Layer],
    kern_holds: Option<KernelKey>,
) -> (Vec<FftUniform>, Vec<Dispatch>, Vec<Job>, Option<KernelKey>) {
    let mut b = Builder {
        uniforms: Vec::new(),
        dispatches: Vec::new(),
    };
    let mut jobs = Vec::new();
    let mut kern_holds = kern_holds;

    for &Layer {
        slot,
        conv,
        radius,
        aperture,
    } in layers
    {
        let kernel_start = b.dispatches.len();
        if kern_holds != Some((aperture, radius, conv)) {
            // The aperture into set A's aux plane — nothing meaningful lives there
            // between layers — then its transform, landing in the kernel.
            b.push(
                FftUniform {
                    radius,
                    shape: aperture.shape,
                    param: aperture.param,
                    angle: aperture.angle,
                    ..for_conv(conv)
                },
                Pipe::MakeKernel,
                Binds::BToA,
                full_groups(conv),
            );
            b.sweep(conv, Pipe::FftOne, -1.0, true, true);
            kern_holds = Some((aperture, radius, conv));
        }
        let kernel = kernel_start..b.dispatches.len();

        // The image: decoded into B by the render pass the encoder interposes,
        // then forward, multiply, inverse — an odd count, so the chain that
        // starts reading B ends writing A, where the filter bind group looks.
        let image_start = b.dispatches.len();
        let from_a = b.sweep(conv, Pipe::FftBoth, -1.0, false, false);
        b.push(
            FftUniform {
                inv_n: 1.0 / (conv.width as f32 * conv.height as f32),
                ..for_conv(conv)
            },
            Pipe::ApplyKernel,
            if from_a { Binds::AToB } else { Binds::BToA },
            full_groups(conv),
        );
        b.sweep(conv, Pipe::FftBoth, 1.0, !from_a, false);
        jobs.push(Job {
            slot,
            kernel,
            image: image_start..b.dispatches.len(),
        });
    }
    (b.uniforms, b.dispatches, jobs, kern_holds)
}

/// A uniform naming `conv` as its transform extent and nothing else — every
/// dispatch of a layer's chain starts from this, so no pass can run at another
/// layer's size.
fn for_conv(conv: Extent2) -> FftUniform {
    FftUniform {
        nx: conv.width,
        ny: conv.height,
        ..Default::default()
    }
}

/// The workgroup grid covering every texel of `conv` — `make_kernel` and
/// `apply_kernel`'s shape.
fn full_groups(conv: Extent2) -> (u32, u32) {
    (conv.width.div_ceil(BLUR_WG), conv.height.div_ceil(BLUR_WG))
}

/// [`plan`]'s pen: the two lists grown in step, so slot `i` is dispatch `i`'s by
/// construction rather than by two counters agreeing.
struct Builder {
    uniforms: Vec<FftUniform>,
    dispatches: Vec<Dispatch>,
}

impl Builder {
    fn push(&mut self, u: FftUniform, pipe: Pipe, binds: Binds, groups: (u32, u32)) {
        let slot = self.uniforms.len() as u32;
        self.uniforms.push(u);
        self.dispatches.push(Dispatch {
            pipe,
            binds,
            slot,
            groups,
        });
    }

    /// One whole 2-D transform over `conv`: every Stockham span of every axis,
    /// sources alternating from `from_a`, the last pass redirected into the
    /// kernel texture when asked. Returns where the *next* pass would read.
    fn sweep(
        &mut self,
        conv: Extent2,
        pipe: Pipe,
        dir: f32,
        mut from_a: bool,
        into_kernel: bool,
    ) -> bool {
        let (w, h) = (conv.width, conv.height);
        let total = w.trailing_zeros() + h.trailing_zeros();
        let mut i = 0;
        for (axis, n) in [(0u32, w), (1u32, h)] {
            // The lines run along the other axis; each pass is n/2 butterflies
            // per line.
            let lines = if axis == 0 { h } else { w };
            let groups = ((n / 2).div_ceil(BLUR_WG), lines.div_ceil(BLUR_WG));
            for s in 0..n.trailing_zeros() {
                let last = i + 1 == total;
                let binds = match (from_a, into_kernel && last) {
                    (true, false) => Binds::AToB,
                    (false, false) => Binds::BToA,
                    (true, true) => Binds::AToKernel,
                    (false, true) => Binds::BToKernel,
                };
                self.push(
                    FftUniform {
                        ns: 1 << s,
                        axis,
                        dir,
                        ..for_conv(conv)
                    },
                    pipe,
                    binds,
                    groups,
                );
                from_a = !from_a;
                i += 1;
            }
        }
        from_a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Where a run of dispatches leaves the data, given where it started.
    fn lands_in_a(dispatches: &[Dispatch], range: &Range<usize>) -> bool {
        match dispatches[range.clone()].last().expect("a run runs").binds {
            Binds::BToA => true,
            Binds::AToB => false,
            other => panic!("an image run cannot end in the kernel: {other:?}"),
        }
    }

    fn layer(slot: u32, w: u32, h: u32, radius: f32) -> Layer {
        shaped(slot, w, h, radius, DISC)
    }

    fn shaped(slot: u32, w: u32, h: u32, radius: f32, aperture: Aperture) -> Layer {
        Layer {
            slot,
            conv: Extent2::new(w, h),
            radius,
            aperture,
        }
    }

    /// The engine's reading of a focal-blur draw's lanes, through the encoder that
    /// wrote them.
    /// The lanes a plain, unobstructed disc decodes to — what the assertions below
    /// compare against, and the value the shader's own `APERTURE_DISC` names.
    const DISC: Aperture = Aperture {
        shape: stark_shaders::mirror::blur::APERTURE_DISC,
        param: 0.0,
        angle: 0.0,
    };

    /// The engine's reading of a focal-blur draw's lanes, through the encoder that
    /// wrote them, under a view that is not turning the canvas.
    fn lanes_of(a: stark_model::document::Aperture) -> Aperture {
        lanes_under(a, ViewTransform::identity(Extent2::new(256, 256)))
    }

    /// The same, under a chosen view — the half of the trip `aperture` adds.
    fn lanes_under(a: stark_model::document::Aperture, view: ViewTransform) -> Aperture {
        aperture(
            &FilterDraw::new(
                stark_model::document::Filter::FocalBlur(stark_model::document::FocalBlur {
                    radius: 4.0,
                    aperture: a,
                }),
                crate::document::CompositeParams::IDENTITY,
            ),
            view,
        )
    }

    /// **The inverse transform lands in set A**, whatever the extent — the
    /// parity fact the filter bind group's whole design rests on, checked as a
    /// property of the plan rather than trusted to the comment that derives it.
    /// And the chain alternates strictly: every dispatch reads what the one
    /// before it wrote.
    #[test]
    fn every_image_chain_lands_in_the_planes_the_filter_reads() {
        for (w, h) in [(2u32, 2u32), (4, 2), (8, 8), (256, 64), (4096, 2048)] {
            let (uniforms, dispatches, jobs, _) = plan(&[layer(0, w, h, 1.0)], None);
            assert_eq!(uniforms.len(), dispatches.len(), "one uniform per dispatch");
            let job = &jobs[0];
            assert!(
                lands_in_a(&dispatches, &job.image),
                "{w}×{h}: the resolve would read a stale set",
            );
            // Strict alternation, decode (which fills B) included — and every
            // dispatch states this layer's own extent.
            let mut from_a = false;
            for d in &dispatches[job.image.clone()] {
                let expect = if from_a { Binds::AToB } else { Binds::BToA };
                assert_eq!(d.binds, expect, "{w}×{h}: a dispatch read a stale set");
                let u = &uniforms[d.slot as usize];
                assert_eq!((u.nx, u.ny), (w, h), "a dispatch at another layer's size");
                from_a = !from_a;
            }
            // The round trip is forward + multiply + inverse, one pass per span
            // per axis each way.
            let log = (w.trailing_zeros() + h.trailing_zeros()) as usize;
            assert_eq!(job.image.len(), 2 * log + 1);
        }
    }

    /// The kernel is owed exactly when the texture does not already hold this
    /// aperture, at this radius, **at this extent**: a settled frame plans no
    /// kernel work; a change in any of the three plans one build plus one
    /// transform whose last pass writes the kernel texture.
    #[test]
    fn the_kernel_is_cached_by_aperture_radius_and_extent() {
        let conv = Extent2::new(64, 64);
        let disc = (DISC, 3.0, conv);
        let (_, dispatches, jobs, holds) = plan(&[layer(0, 64, 64, 3.0)], None);
        let log = 12; // 6 + 6
        assert_eq!(jobs[0].kernel.len(), 1 + log, "build + one transform");
        assert!(matches!(
            dispatches[jobs[0].kernel.clone()].last().unwrap().binds,
            Binds::AToKernel | Binds::BToKernel
        ));
        assert_eq!(holds, Some(disc));

        let (_, _, jobs, holds) = plan(&[layer(0, 64, 64, 3.0)], Some(disc));
        assert!(jobs[0].kernel.is_empty(), "a settled radius owes nothing");
        assert_eq!(holds, Some(disc));

        // Two layers, two radii: the second rebuilds, and the cache ends on it.
        let (_, _, jobs, holds) = plan(&[layer(0, 64, 64, 3.0), layer(2, 64, 64, 7.0)], Some(disc));
        assert!(jobs[0].kernel.is_empty());
        assert!(!jobs[1].kernel.is_empty());
        assert_eq!(holds, Some((DISC, 7.0, conv)));
        assert_eq!(jobs[1].slot, 2, "a job answers for its filter slot");

        // The part of the key that is easy to forget: one radius, two extents —
        // a decimated layer beside an undecimated one — is two spectra.
        let (_, _, jobs, _) = plan(&[layer(0, 64, 64, 3.0), layer(1, 32, 32, 3.0)], Some(disc));
        assert!(jobs[0].kernel.is_empty());
        assert!(
            !jobs[1].kernel.is_empty(),
            "one disc at two transform sizes is two different tables of frequencies",
        );

        // And the part added with the shapes: the same radius at the same extent
        // through a *different aperture* is a different spectrum too. Missing this
        // is the failure with no symptom in the plan — the picture would simply go
        // on wearing the shape the layer before it asked for.
        let hex = Aperture {
            shape: stark_shaders::mirror::blur::APERTURE_BLADES,
            param: 6.0,
            angle: 0.0,
        };
        let (uniforms, dispatches, jobs, holds) = plan(
            &[layer(0, 64, 64, 3.0), shaped(1, 64, 64, 3.0, hex)],
            Some(disc),
        );
        assert!(jobs[0].kernel.is_empty());
        assert!(
            !jobs[1].kernel.is_empty(),
            "a shape change owes a kernel: the disc's spectrum is not the hexagon's",
        );
        assert_eq!(holds, Some((hex, 3.0, conv)));
        // …and the shape reaches `make_kernel`'s own uniform rather than stopping
        // at the cache key.
        let build = dispatches[jobs[1].kernel.start];
        assert_eq!(build.pipe, Pipe::MakeKernel);
        let u = &uniforms[build.slot as usize];
        assert_eq!((u.shape, u.param, u.radius), (hex.shape, 6.0, 3.0));

        // One shape at two turns is likewise two kernels — an angle that keyed
        // nothing would leave a rotated iris showing the previous one's.
        let turned = Aperture { angle: 0.4, ..hex };
        let (_, _, jobs, _) = plan(&[shaped(0, 64, 64, 3.0, turned)], Some((hex, 3.0, conv)));
        assert!(!jobs[0].kernel.is_empty());
    }

    /// **Every aperture the document can hold arrives here as its own lanes**, and
    /// the shape codes are the shader's own — the round trip through
    /// `group::aperture_lanes`, which is the only place a `Filter` becomes numbers.
    ///
    /// Pinned as a set rather than arm by arm because what would actually break is
    /// two shapes agreeing: a code copied from the arm above it renders one
    /// aperture as another, and no assertion about a single arm can see that.
    #[test]
    fn every_aperture_survives_the_trip_through_the_draw() {
        use stark_model::document::Aperture as Doc;
        use stark_shaders::mirror::blur as code;

        assert_eq!(lanes_of(Doc::Disc { obstruction: 0.0 }), DISC);
        // The obstruction is the disc's own lane, not a shape of its own — the
        // merge, checked where it would show if the two ever came apart again.
        assert_eq!(
            lanes_of(Doc::Disc { obstruction: 0.5 }),
            Aperture { param: 0.5, ..DISC },
        );
        assert_eq!(
            lanes_of(Doc::Blades {
                count: 5,
                angle: 0.25
            }),
            Aperture {
                shape: code::APERTURE_BLADES,
                param: 5.0,
                angle: 0.25,
            },
        );
        assert_eq!(
            lanes_of(Doc::Oval {
                squeeze: 2.0,
                angle: -0.5
            }),
            Aperture {
                shape: code::APERTURE_OVAL,
                param: 2.0,
                angle: -0.5,
            },
        );

        let codes: Vec<u32> = Doc::ALL.iter().map(|a| lanes_of(*a).shape).collect();
        let mut sorted = codes.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), codes.len(), "two apertures share one code");
    }

    /// **The aperture turns with the canvas, not with the screen** (§21.10, §6.4) —
    /// the trip the radius has always made, made by the angle too.
    ///
    /// The bug this pins had no symptom a still frame could show: the polygon was
    /// correct at every setting, and only *rotating the canvas* revealed that it had
    /// stayed welded to the screen while the painting turned under it — with an
    /// export at a rotated view then disagreeing with the display, which is a
    /// document whose pixels depend on view state.
    ///
    /// Also pinned here: the disc's lane stays zero however the view turns. It is
    /// not that a turned disc would look wrong — it cannot — but that the angle keys
    /// the kernel cache, so a disc that took the trip would rebuild its kernel on
    /// every frame of a canvas rotation for no visible difference.
    #[test]
    fn the_apertures_turn_is_carried_from_the_canvas_frame() {
        use stark_model::document::Aperture as Doc;

        let flat = ViewTransform::identity(Extent2::new(256, 256));
        let turned = ViewTransform {
            rotation: 0.5,
            ..flat
        };
        let blades = Doc::Blades {
            count: 6,
            angle: 0.25,
        };
        assert!((lanes_under(blades, flat).angle - 0.25).abs() < 1e-5);
        assert!(
            (lanes_under(blades, turned).angle - 0.75).abs() < 1e-5,
            "a canvas turned by 0.5 rad must turn the iris with it",
        );

        // The mirror is part of the trip too. `flip_h` negates x, so a canvas
        // direction θ is seen at π − θ; an iris that ignored the mirror would point
        // the wrong way in the one view whose whole purpose is to show the drawing
        // as it is rather than as it is expected.
        let flipped = ViewTransform {
            flip_h: true,
            ..flat
        };
        assert!(
            (lanes_under(blades, flipped).angle - (std::f32::consts::PI - 0.25)).abs() < 1e-5,
            "a mirrored view must mirror the iris's turn",
        );

        // An oval is a direction too, and takes the same trip.
        let oval = Doc::Oval {
            squeeze: 2.0,
            angle: 0.0,
        };
        assert!((lanes_under(oval, turned).angle - 0.5).abs() < 1e-5);

        // The disc has no direction to carry, at any view.
        for view in [flat, turned, flipped] {
            assert_eq!(lanes_under(Doc::Disc { obstruction: 0.5 }, view).angle, 0.0);
        }
    }

    /// A decimated layer keeps its aperture and rescales only its radius — the
    /// shape's own numbers are ratios and counts, which a change of resolution
    /// does not touch (§21.12).
    #[test]
    fn decimation_rescales_the_radius_and_leaves_the_shape_alone() {
        let ring = Aperture { param: 0.5, ..DISC };
        let l = layers(Extent2::new(2560, 1440), &[(0, 2800.0, ring)], 8192);
        assert_eq!(l[0].radius, 2800.0 / 32.0, "the radius survives, rescaled");
        assert_eq!(l[0].aperture, ring, "a fraction of the radius is still it");
    }

    /// The decimation rule: full resolution through the whole of the knob's own
    /// range on screen, then the smallest power of two that brings the
    /// convolution radius back inside it.
    #[test]
    fn the_scale_holds_full_resolution_through_the_knobs_range() {
        assert_eq!(scale(0.0), 1);
        assert_eq!(scale(MAX_CONV_RADIUS), 1, "the bound itself is undecimated");
        assert_eq!(scale(MAX_CONV_RADIUS + 0.5), 2);
        assert_eq!(scale(2048.0), 16);
        assert_eq!(scale(2049.0), 32);
        // The loop's own floor: a hostile value decimates rather than spinning.
        assert!(scale(f32::INFINITY) <= 1 << 20);
    }

    /// **Zooming into a blur cannot balloon the transform** — the regression the
    /// decimation exists for. An on-screen radius in the thousands once pushed
    /// the padded planes to the device's limits (gigabytes of `f32`, and a
    /// kernel too large to zero-initialize); decimated, the same ask comes out
    /// a few hundred texels square.
    #[test]
    fn zooming_in_cannot_balloon_the_transform() {
        let accum = Extent2::new(2560, 1440);
        // Radius 44 canvas px at 64× zoom, say: ~2800 accumulator texels.
        let l = layers(accum, &[(0, 2800.0, DISC)], 8192);
        assert_eq!(l[0].conv, Extent2::new(256, 256), "decimated to affordable");
        assert_eq!(l[0].radius, 2800.0 / 32.0, "the radius survives, rescaled");

        // Absurd and hostile radii stay bounded rather than saturating.
        let l = layers(accum, &[(0, 1e6, DISC)], 8192);
        assert!(l[0].conv.width <= 512 && l[0].conv.height <= 512);
        let l = layers(accum, &[(0, f32::NAN, DISC)], 8192);
        assert!(
            l[0].conv.width <= 4096,
            "a non-finite radius decimates to a point"
        );

        // And at working zooms nothing changed: the guard band pads at full
        // resolution, exactly as before.
        let l = layers(Extent2::new(300, 200), &[(0, 10.0, DISC)], 8192);
        assert_eq!(l[0].conv, Extent2::new(512, 256));
        assert_eq!(l[0].radius, 10.0);
    }

    /// The device's texture limit is the clamp of last resort, and only there
    /// does the radius itself give way — a smaller blur, never a wrapped one.
    #[test]
    fn the_device_cap_binds_last_and_costs_radius() {
        let l = layers(Extent2::new(200, 200), &[(0, 60.0, DISC)], 256);
        assert_eq!(l[0].conv, Extent2::new(256, 256), "capped by the device");
        assert_eq!(l[0].radius, 28.0, "the guard the cap left: (256 − 200) / 2");
    }
}
