//! Compositing and the media/lighting pass (§6.3, §6.4).
//!
//! Five passes, one per module, each owning its uniform, its pipeline and the
//! constants only it reads:
//!
//! | Pass | Module | What it does |
//! |---|---|---|
//! | A | [`tiles`] | every visible tile's channels into the offscreen accumulator, mattes interleaved (§15.4) |
//! | — | [`blend`] | a group with a mode of its own, isolated and merged back (§18.0.4) |
//! | — | [`filter`] | a filter layer: the accumulator beneath it, read and rewritten (§21) |
//! | B | [`media`] | normals off the height field, lit, tonemapped, over the substrate (§6.3) |
//! | C | [`overlay`] | selection outlines over the lit result (§6.8, §17.3) |
//! | D | [`guides`] | the perspective grid over everything (§20.4) |
//! | E | [`resolve`] | the supersampled render boxed down to the target (§6.4) |
//!
//! What is left here is the part no single pass owns: the two structs the passes
//! hang off, and the walk that orders them.
//!
//! # Two types, split along one line: does it depend on the target?
//!
//! [`CompositorPipeline`] is everything that does not — the pipelines, the layouts,
//! the pigment LUT, and the view settings the media pass reads. A [`Compositor`] is
//! one target's worth of what does: pass A's offscreen attachments, the media bind
//! group over them, the blend scratch, the instance streams, and every uniform a
//! render writes (what it is looking at, how it is lit, how many samples it took).
//!
//! The split exists because more than one thing gets drawn from the same document,
//! at different sizes: the surface every frame, and beside it an export or the
//! navigator's miniature. One `Compositor` shared between them spends a rebuild of
//! *both* sizes' attachments per alternation — which is affordable for a file export
//! and not for a miniature refreshed on every edit. So each keeps its own, and they
//! share the expensive half by reference. What they must not each keep a copy of is
//! the view settings: two consumers disagreeing about the canvas weave or the
//! lighting would be a bug visible only in the smaller picture, so those live in the
//! pipeline behind a generation counter that each `Compositor` notices.

mod blend;
mod filter;
mod group;
mod guides;
mod media;
mod overlay;
mod plan;
mod resolve;
mod tiles;
mod view;

use std::sync::Arc;

use crate::colorspace::ColorSpace;
use crate::geom::{Extent2, ViewTransform};
use crate::gpu::channels::{ChannelFormats, Targets};
use crate::gpu::context::GpuContext;
use crate::gpu::desc;
use crate::gpu::environment::Environment;
use crate::gpu::surface::{SURFACE_TILE_PX, Surface};

use crate::gpu::uniforms::{InstanceStream, UniformSlots};
pub(crate) use blend::{BlendPass, BlendUniform, blend_code};
use blend::{ScratchLevel, ScratchTargets};
pub(crate) use filter::{FilterPass, FilterUniform};
// `pack_guides` is imported by name rather than qualified, because `render`'s own
// `guides` binding (the scene's list) would otherwise shadow the module in a reader's
// eye — `guides::pack_guides(scene, view)` resolves fine and reads badly.
use guides::{GuidePass, GuideUniform, pack_guides};
use media::{MediaPass, MediaUniform};
use overlay::{OverlayInstance, OverlayPass, PEER_OUTLINE_ALPHA};
use plan::{Draw, Plan, Slot, Step};
use resolve::{ResolvePass, ResolveUniform, supersample};
use tiles::{Instance, MatteInstance, Ramp, TilePass};
use view::{View, ViewBindings};

pub use group::{CompositeGroup, CompositeItem, FilterDraw, GroupContent, MatteDraw};
pub use media::MediaParams;
pub use overlay::SelectionOutline;
pub(crate) use view::view_uniform;

/// What stays the same for every step of pass A, as against what the step varies.
///
/// A step changes the targets and the slot; these two do not, so they travel as one
/// parameter rather than two repeated at every call — the shape `CompositeScene` and
/// `stroke::StrokeScene` already use.
struct Encode<'a> {
    p: &'a CompositorPipeline,
    /// The per-item bind groups [`Compositor::upload`] gathered, in the flat order
    /// the plan's [`Draw`]s index.
    streams: &'a PreparedStreams<'a>,
}

/// What [`Compositor::upload`] hands the encoder: the per-tile bind groups, and the
/// per-matte ramp bind group beside them. One value because they are one
/// preparation, gathered from one plan and indexed by its [`Draw`]s.
///
/// The tile groups are **borrowed from the tiles themselves** — each is built once
/// and kept for that tile's life ([`TilePairHandle::composite_bg`]) — so what this
/// collects per frame is a list of references rather than a list of new wgpu
/// objects. The lifetime is the draw list's, which outlives the render.
///
/// The ramps are one bind group over one slotted buffer, not one apiece: matte `i`
/// binds slot `i`, which is the instance index the draw already carries. `None` only
/// when the frame has no matte at all.
struct PreparedStreams<'a> {
    tile_bgs: Vec<&'a wgpu::BindGroup>,
    matte_ramp_bg: Option<wgpu::BindGroup>,
}

/// What one render draws, as against *where and how* it draws it (the target and the
/// view, which stay separate parameters).
///
/// These four travel together because they are one description of the document at an
/// instant, assembled in one place — [`Engine::render_view`](crate::Engine) — and
/// meaningless apart: the ground belongs under the stack, the outlines over it, and
/// `transparent` says whether the ground is drawn at all.
pub struct CompositeScene<'a> {
    /// The substrate color in the document's working channels — the ground under
    /// the paint (§15.5).
    pub background: [f32; 4],
    /// The substrate color's **residual** in `.xyz` (§6.7); `.w` unused. Zero in a
    /// space that has none — and the reason a *black* ground in a pigment document
    /// finally reads black, since Mixbox's polynomial renders those concentrations
    /// `#383838` on their own.
    pub background_resid: [f32; 4],
    /// The visible layers, bottom-to-top, cut into blend groups.
    pub groups: &'a [CompositeGroup],
    /// Selection outlines to draw over the lit result: the local actor's and each
    /// present peer's (§17.3). Empty for anything that is not the screen
    /// — chrome is a thing to draw *with* (§15.6).
    pub outlines: &'a [SelectionOutline<'a>],
    /// Leave the substrate out and carry the paint's own alpha to the target, for a
    /// cut-out export.
    pub transparent: bool,
    /// The visible drawing guides, drawn over everything (§20.4). Empty for
    /// anything that is not the screen — chrome, on the same argument as
    /// `outlines`.
    pub guides: &'a [crate::guides::GuideScene],
}

/// The pipelines themselves — the six passes, their layouts and samplers, and the
/// channel formats they were built for.
///
/// Split from [`CompositorPipeline`] along a second line: does it ever *change*?
/// Nothing here does, which is what lets a second engine on the same device share
/// one of these by `Arc` ([`Engine::new_sharing`]) instead of compiling seven
/// shaders and eight pipelines of its own — while each engine keeps its own view
/// settings, since the brush editor's preview mirrors the canvas's look and a
/// preset thumbnail deliberately does not.
///
/// **No uniform buffer is here**, and that is the whole content of "nothing here
/// changes". The view, media and resolve uniforms used to be, sound on the argument
/// that each render writes them through the queue immediately before the submit that
/// reads them and submits on one queue are ordered. That argument is about the
/// *sequence of calls*, not about these types: two `Compositor`s over one of these is
/// the documented arrangement, `&CompositorPipeline` is all either needs, and nothing
/// stopped a caller straddling a write and its submit with another render. Each of
/// the three holds per-target state anyway — what this render is looking at, how it
/// is lit, how many samples it took — so they belong to the [`Compositor`] and now
/// live there ([`ViewBindings`], [`Compositor::media_buf`], [`Supersampled::buf`]).
///
/// [`Engine::new_sharing`]: crate::Engine::new_sharing
pub struct CompositorPasses {
    ctx: GpuContext,

    /// The tile sampler passes A and C share — the whole of the view that is not
    /// per-target (see [`ViewBindings`] for the half that is).
    view: View,
    tiles: TilePass,
    /// Shared with `gpu::merge`, which runs the same pipeline on tile-sized
    /// targets to merge a layer down through its mode (§14.11).
    blend: Arc<BlendPass>,
    /// Filter layers (§21) — the blend pass with the isolated source removed, so
    /// close to it that the two share the scratch and the pigment LUT.
    ///
    /// Shared with `gpu::merge` on the same terms as `blend` above: merging a filter
    /// layer into the paint beneath it runs this module's tile-space entry point
    /// (§14.11.7), so the merged tile comes out of the shader the screen runs.
    filter: Arc<FilterPass>,
    overlay: OverlayPass,
    guides: GuidePass,
    /// `media_pass`, not `media`: [`CompositorPipeline`] reaches this whole struct
    /// through `Deref` *and* has a `media()` of its own returning the
    /// [`MediaParams`], so a bare `media` made `p.media` and `p.media()` two
    /// different things one character apart — and the field, being the `Deref`'d one,
    /// is the half a reader is least expecting.
    media_pass: MediaPass,
    resolve: ResolvePass,

    /// Offscreen channel formats, from the color space (§6.7) — including whether
    /// there is a residual at all, which decides the third attachment on every pass A
    /// target and is the space's answer rather than a per-target choice.
    formats: ChannelFormats,
    /// What passes B–D write, and therefore what the supersampled target carries.
    target_format: wgpu::TextureFormat,
}

/// Everything about compositing that does not depend on *what is being drawn into*:
/// the passes ([`CompositorPasses`], reached through `Deref`) and the view settings
/// the media pass reads.
///
/// Split from [`Compositor`] so several of them can share one of these. Each renders
/// into a target of its own size and therefore keeps its own attachments; what they
/// must *not* keep their own of is anything on this side of the line — the pipelines
/// because they are expensive (six passes, plus a decoded Mixbox LUT), and the view
/// settings because two consumers disagreeing about the canvas weave or the lighting
/// would be a bug that shows only in the smaller picture.
///
/// Not immutable: the view settings change, through `&mut self` on this type. It
/// holds no GPU-visible state that a render writes — every uniform a render fills in
/// belongs to the [`Compositor`] doing it — so a consumer's `&CompositorPipeline` is
/// a read of settled values and nothing else.
pub struct CompositorPipeline {
    /// The pipelines, shared: this engine's alone after [`Self::new`], a sibling
    /// engine's too after [`Self::sharing`].
    passes: Arc<CompositorPasses>,

    /// Lighting parameters for the media pass (§6.3) — a view setting like the two
    /// below, copied into each renderer's own media uniform on every render.
    media_params: MediaParams,
    // The canvas surface (bump) sampled by the media pass for relief.
    surface: Surface,
    // The HDR lighting environment sampled by the media pass (§6.3).
    environment: Environment,
    /// A stamp for "the state a media bind group would be built against". Moved
    /// whenever `surface` or `environment` is swapped: both are bound *into* each
    /// consumer's bind group, so each has to notice and rebuild — and a stamp is what
    /// makes noticing structural rather than a fan-out of notifications that a new
    /// consumer could be left out of.
    ///
    /// Drawn from a **process-wide** counter rather than counted per pipeline, so no
    /// two states anywhere ever share a value: "same stamp" then implies "same
    /// pipeline, same settings", and a consumer's decision to reuse cannot be wrong.
    ///
    /// The case that needs that is a color-space rebuild (§6.7), which does
    /// not mutate a pipeline but *replaces* it. A per-pipeline counter would start the
    /// replacement back at its initial value — the very value a consumer that had
    /// rendered against the old pipeline is holding — so a kept [`Compositor`] would
    /// see "no change" and keep attachments belonging to the pipeline that is gone.
    /// Today both color spaces happen to use the same channel formats, so that would
    /// come out *harmless*; but which formats a space wants is a decision the
    /// `ColorSpace` trait deliberately leaves open ([`ColorSpace::color_format`]), and
    /// "correct because two implementations coincide" is not a property to build on.
    generation: u64,
}

/// The next value for [`CompositorPipeline::generation`] — see there for why this is
/// process-wide.
fn next_generation() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// One consumer's worth of compositing state: the offscreen attachments pass A
/// writes, the media bind group over them, the scratch the blend passes bounce
/// through, the instance streams, and the uniforms a render fills in. Everything here
/// is sized by the target, by how much there is to draw, or by neither and simply
/// **written per render** — which is the same line, since a value a render writes is
/// a value two targets would disagree about.
///
/// One per thing being drawn into — the surface, and (with its own) anything that
/// renders beside it: an export, the navigator's miniature. Sharing one across
/// targets of different sizes means each render resizes the attachments the other
/// just built, so the cost is paid twice per alternation rather than once ever.
pub struct Compositor {
    /// The size **every** offscreen attachment here is at: the target's, times
    /// [`Self::ss`]. Not the target's own, which is only known where the target is.
    /// Meaningless while [`Self::accum`] is `None`.
    size: Extent2,
    /// The canvas → NDC mapping *this* render is drawing through, and the two group-0
    /// bind groups over it (§6.4). Per consumer, not per pipeline — see
    /// [`ViewBindings`].
    view: ViewBindings,
    /// Pass A's offscreen channels at [`Self::size`], and the media bind group over
    /// them. `None` until the first render, because only a render knows the zoom and
    /// therefore the supersampling factor these are sized by: building them at the
    /// caller's guess would have been an allocation of the whole viewport that
    /// [`Self::ensure_targets`] then threw away on its first call. It is also what
    /// leaves the sizing rule in one place instead of two that could disagree.
    accum: Option<media::Offscreen>,
    /// This consumer's lighting parameters as the media pass reads them, written at
    /// the top of every render from the pipeline's [`MediaParams`]. Per consumer for
    /// [`MediaPass`]'s reason: two engines sharing one pipeline kit light their
    /// canvases differently, and the bind group naming this was per-consumer already.
    media_buf: wgpu::Buffer,
    /// The [`CompositorPipeline::generation`] [`Self::accum`] was built against.
    generation: u64,

    /// Samples per axis the attachments are built for (§6.4). `1` — the
    /// zoomed-in and 1:1 case — means passes B–D write the caller's target directly
    /// and `ss` is `None`, so a view that never zooms out never allocates any of it.
    ss: u32,
    /// Everything that exists only while this view is zoomed out — see
    /// [`Supersampled`].
    ss_target: Option<Supersampled>,

    // Allocated on first use and kept: only a document with a non-`Normal` layer
    // ever pays for them.
    scratch: Option<ScratchTargets>,

    // Pass A's instance streams, grown to the frame's tile and matte counts.
    instances: InstanceStream<Instance>,
    matte_instances: InstanceStream<MatteInstance>,
    /// One dynamic-offset slot per matte in the frame, holding its gradient ramp —
    /// zeroed for a solid one (§22.4).
    matte_ramps: UniformSlots<Ramp>,
    // One dynamic-offset slot per blend group in the frame, and one per filter
    // layer (§21). Separate buffers, one mechanism — see [`UniformSlots`] for both
    // the sharing and the separation.
    blend_uniforms: UniformSlots<BlendUniform>,
    filter_uniforms: UniformSlots<FilterUniform>,
    // Pass C's, grown to the outlined mask-tile count.
    overlay_instances: InstanceStream<OverlayInstance>,
    /// Pass D's, one dynamic-offset slot per visible guide — the same mechanism as
    /// the two above, which this pass spent a hand-written stride and a hand-rolled
    /// grow loop reimplementing.
    guide_uniforms: UniformSlots<GuideUniform>,
}

/// Where passes B–D write when the view is zoomed out, and what pass E reads it back
/// through (§6.4).
///
/// One value rather than three because they are built and dropped together and none
/// of them means anything without the others: the bind group names the view, and the
/// uniform holds the very `ss` that decided the view's size.
struct Supersampled {
    view: wgpu::TextureView,
    bg: wgpu::BindGroup,
    /// `n`, the sample count pass E box-averages over. This consumer's, because `ss`
    /// is a function of *this* target's zoom.
    buf: wgpu::Buffer,
}

/// Somewhere for a render that is **not** the surface's to keep its attachments
/// between calls — a [`Compositor`], built on first use.
///
/// Whether they are worth keeping is the caller's to know, not the engine's, and it
/// turns entirely on whether the render repeats:
///
/// - A **one-shot** — writing a PNG — makes one of these locally and drops it, so an
///   8192-px export does not leave half a gigabyte of attachments parked for the rest
///   of the session waiting for a render that may never come.
/// - A **repeating** render at a steady size — the navigator's miniature, refreshed
///   on every edit — holds one for its lifetime and pays the allocation once.
///
/// Empty is the valid initial state, so nothing needs a size or a pipeline to make
/// one; the first render fills it in and later renders reuse or resize it.
#[derive(Default)]
pub struct Offscreen(Option<Compositor>);

impl Offscreen {
    /// The compositor, built against `p` if this is the first use.
    ///
    /// No size: the attachments are sized by the first *render*, which is the only
    /// thing that knows the zoom and therefore the supersampling factor. This used to
    /// take one and pass it to `Compositor::new`, where it was overwritten by the
    /// very next call — so a miniature allocated a whole 1:1 viewport and threw it
    /// away, and the parameter told a reader something that was not true.
    pub(crate) fn get(&mut self, p: &CompositorPipeline) -> &mut Compositor {
        self.0.get_or_insert_with(|| Compositor::new(p))
    }
}

// The passes are what almost every read wants — `p.tiles`, `p.ctx`, `p.resolve` —
// and they moved wholesale into [`CompositorPasses`]. `Deref` keeps those reads
// spelled as they were, so the split shows up only where it means something: the
// three view settings, which stayed behind.
impl std::ops::Deref for CompositorPipeline {
    type Target = CompositorPasses;

    fn deref(&self) -> &CompositorPasses {
        &self.passes
    }
}

impl CompositorPipeline {
    pub(crate) fn new(
        ctx: &GpuContext,
        target_format: wgpu::TextureFormat,
        color_space: &dyn ColorSpace,
        surface: Surface,
        environment: Environment,
        blend: Arc<BlendPass>,
        filter: Arc<FilterPass>,
    ) -> Self {
        let device = &ctx.device;
        let formats = ChannelFormats::of(color_space);
        // Passes B–E all write the one target the frame is presented from.
        let screen = [desc::target(target_format)];

        let passes = CompositorPasses {
            tiles: TilePass::new(device, color_space, formats),
            blend,
            filter,
            overlay: OverlayPass::new(device, target_format),
            guides: GuidePass::new(device, target_format),
            media_pass: MediaPass::new(device, color_space, &screen),
            resolve: ResolvePass::new(device, &screen),
            view: View::new(device),
            ctx: ctx.clone(),
            formats,
            target_format,
        };
        Self::sharing(
            Arc::new(passes),
            surface,
            environment,
            MediaParams::default(),
        )
    }

    /// A pipeline over **already-built** passes — the whole of what a sibling engine
    /// pays for its compositor ([`Engine::new_sharing`]): no shader compiles, no
    /// pipeline creation, just its own copy of the three view settings.
    ///
    /// The settings are the caller's to seed because the two consumers want opposite
    /// things: the brush editor's preview opens mirroring the canvas's current look,
    /// a preset thumbnail pins the fixed look its cache key assumes.
    ///
    /// [`Engine::new_sharing`]: crate::Engine::new_sharing
    pub(crate) fn sharing(
        passes: Arc<CompositorPasses>,
        surface: Surface,
        environment: Environment,
        media_params: MediaParams,
    ) -> Self {
        Self {
            passes,
            media_params,
            surface,
            environment,
            generation: next_generation(),
        }
    }

    /// The shared pipeline kit, for building a sibling engine's pipeline over it.
    pub(crate) fn passes(&self) -> Arc<CompositorPasses> {
        Arc::clone(&self.passes)
    }

    /// The current media/lighting parameters (§6.3).
    pub fn media(&self) -> MediaParams {
        self.media_params
    }

    /// Adjust the media/lighting parameters (§6.3).
    pub fn set_media(&mut self, media: MediaParams) {
        self.media_params = media;
    }

    /// Swap the canvas surface (bump) so the next render shades against it
    /// (§6.4). A view-time swap — the composited tiles are untouched.
    ///
    /// Each [`Compositor`] rebuilds its media bind group when it next notices the
    /// generation moved, rather than being told: a swap has to reach every consumer,
    /// and the one that would be forgotten is exactly the one nobody is looking at.
    pub fn set_surface(&mut self, surface: Surface) {
        self.surface = surface;
        self.generation = next_generation();
    }

    /// Swap the HDR lighting environment so the next render samples it (§6.3).
    pub fn set_environment(&mut self, environment: Environment) {
        self.environment = environment;
        self.generation = next_generation();
    }

    /// The channel formats pass A writes (§6.7) — what a caller supplying its own
    /// targets to [`Compositor::composite_channels`] has to allocate.
    ///
    /// The whole [`ChannelFormats`] rather than the `(color, aux, resid)` tuple it
    /// used to be taken apart into: the residual is not a channel a caller may decide
    /// to skip, and handing back the type that says so is cheaper than the paragraph
    /// that used to.
    pub(crate) fn channel_formats(&self) -> ChannelFormats {
        self.formats
    }

    /// The offscreen pair and the media bind group over it, at `size`. `media_buf`
    /// is the consumer's own uniform, which that bind group names.
    fn offscreen(&self, size: Extent2, media_buf: &wgpu::Buffer) -> media::Offscreen {
        media::offscreen(media::OffscreenDesc {
            device: &self.ctx.device,
            size,
            formats: self.formats,
            media: &self.media_pass,
            media_buf,
            surface: &self.surface,
            environment: &self.environment,
        })
    }
}

impl Compositor {
    /// The instance streams, uniforms and view bindings for one consumer, against the
    /// shared `pipeline`. Cheap — everything expensive (the six passes' pipelines,
    /// their layouts, the decoded pigment LUT) lives in the pipeline and is only
    /// borrowed, and everything *sized* (the accumulator, the supersampled target,
    /// the blend scratch) waits for the first render, which is the only thing that
    /// knows the zoom.
    pub fn new(pipeline: &CompositorPipeline) -> Self {
        let device = &pipeline.ctx.device;
        Self {
            size: Extent2::new(0, 0),
            view: ViewBindings::new(
                device,
                &pipeline.view,
                &pipeline.tiles.view_bgl,
                &pipeline.overlay.view_bgl,
            ),
            media_buf: media::uniform_buffer(device),
            accum: None,
            generation: pipeline.generation,
            // 1:1 until a render says otherwise — `ensure_targets` is what decides,
            // because only a render knows the zoom.
            ss: 1,
            ss_target: None,
            scratch: None,
            instances: InstanceStream::new(device, "stark composite instances"),
            matte_instances: InstanceStream::new(device, "stark matte instances"),
            matte_ramps: UniformSlots::new(device, "stark matte ramp", 1),
            blend_uniforms: UniformSlots::new(device, "stark blend uniform", 1),
            filter_uniforms: UniformSlots::new(device, "stark filter uniform", 1),
            overlay_instances: InstanceStream::new(device, "stark overlay instances"),
            guide_uniforms: UniformSlots::new(device, "stark guide uniform", 1),
        }
    }

    /// Bring the attachments in line with what is about to be drawn — a target of
    /// `target_size` rendered at `ss` samples per axis, against the pipeline's current
    /// surface/environment — and hand back the accumulator to draw into.
    ///
    /// Returning it rather than leaving the caller to unwrap [`Self::accum`] is what
    /// makes "the accumulator exists" a consequence of having called this, in the one
    /// place that can build it.
    ///
    /// Called at the top of every render, so a resized target, a zoom that crossed a
    /// supersampling threshold, a swapped canvas weave, a swapped light and a whole
    /// rebuilt pipeline (a color-space change, which changes the channel *formats*)
    /// all land without anyone having to be notified — see
    /// [`CompositorPipeline::generation`].
    ///
    /// The blend scratch is dropped rather than kept through any of it: it is sized
    /// like the attachments and carries their formats, so "everything that depends on
    /// the target or the pipeline is rebuilt together" holds by construction instead
    /// of by a second condition that could disagree with this one. It costs one
    /// reallocation on the next blended render, and only a document with a
    /// non-`Normal` layer has one at all.
    fn ensure_targets(
        &mut self,
        p: &CompositorPipeline,
        target_size: Extent2,
        ss: u32,
    ) -> &media::Offscreen {
        let size = Extent2::new(target_size.width * ss, target_size.height * ss);
        let current = self.accum.is_some()
            && size == self.size
            && ss == self.ss
            && self.generation == p.generation;
        if !current {
            self.size = size;
            self.ss = ss;
            self.generation = p.generation;
            self.scratch = None;
            self.accum = Some(p.offscreen(self.size, &self.media_buf));
            // Allocated only where it is written into. A 1:1 view drops it here, which
            // is what returns the memory the moment the artist zooms back in to paint.
            self.ss_target = (ss > 1).then(|| {
                let device = &p.ctx.device;
                let view = offscreen_view(device, size, p.target_format, "stark supersampled");
                let buf = resolve::uniform_buffer(device);
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("stark resolve bg"),
                    layout: &p.resolve.bgl,
                    entries: &[desc::uniform_entry(0, &buf), desc::tex(1, &view)],
                });
                Supersampled { view, bg, buf }
            });
        }
        self.accum
            .as_ref()
            .expect("the branch above builds it when it is absent")
    }

    /// Write the view uniform and upload everything `plan` decided, returning the
    /// per-tile bind groups pass A draws with — and, beside them, the per-matte ramp
    /// bind group (`None` when the frame has no matte at all; §22.4).
    ///
    /// Split out of [`Self::render`] so [`Self::composite_channels`] runs the *same*
    /// pass A rather than a second copy of it: what the eyedropper reports and what
    /// the screen shows then cannot drift, which is the whole reason for sampling
    /// through the compositor at all.
    ///
    /// No walk of its own. Everything here is a loop over what [`Plan::build`]
    /// already ordered, which is what makes "slot `n` is the `n`th merge the encoder
    /// reaches" true by construction rather than by two recursions agreeing.
    fn upload<'a>(
        &mut self,
        p: &CompositorPipeline,
        view: ViewTransform,
        plan: &Plan<'a>,
    ) -> PreparedStreams<'a> {
        let device = &p.ctx.device;
        let queue = &p.ctx.queue;
        self.view.write(queue, view);

        // Built once per tile and kept on it, not once per tile per frame — see
        // [`TilePairHandle::composite_bg`] for why a tile's immutability makes that
        // sound, and what it was costing.
        let tile_bgs = plan
            .tiles
            .iter()
            .map(|handle| {
                handle.composite_bg(|| {
                    let mut entries = vec![
                        desc::tex(0, handle.color_view()),
                        desc::tex(1, handle.aux_view()),
                    ];
                    // The layout carries slot 2 exactly when the space has a residual
                    // (§6.7), and every tile of such a document has one — the space
                    // decides once, at `acquire_tile`.
                    if let Some(view) = handle.resid_view() {
                        entries.push(desc::tex(2, view));
                    }
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("stark composite tile bg"),
                        layout: &p.tiles.tile_bgl,
                        entries: &entries,
                    })
                })
            })
            .collect();

        self.instances.write(device, queue, &plan.instances);
        let mut matte_ramp_bg = None;
        if !plan.mattes.is_empty() {
            self.matte_instances.write(device, queue, &plan.mattes);
            self.matte_ramps.write(device, queue, &plan.ramps);
            // Built after the write, so it names the buffer the write may have just
            // grown — the same reason the guide pass builds its own per render.
            matte_ramp_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("stark matte ramp bg"),
                layout: &p.tiles.ramp_bgl,
                entries: &[self.matte_ramps.binding(0)],
            }));
        }

        // One uniform slot per merge and one per filter layer, all written before the
        // single submit — see [`UniformSlots`] for why they cannot share one. The
        // *order* is the plan's, and each step carries the index it binds, so a
        // filter and a blend group side by side cannot count each other's slots.
        //
        // The filter uniforms are built here rather than in the plan for one lane's
        // sake: the chromatic dispersion is stated in canvas terms by the document
        // and sampled in accumulator texels by the pass, and this — with `view`
        // already supersampled — is the moment the two meet (§21.10).
        self.blend_uniforms.write(device, queue, &plan.blends);
        let filters: Vec<FilterUniform> = plan
            .filters
            .iter()
            .map(|f| plan::filter_uniform(f, view))
            .collect();
        self.filter_uniforms.write(device, queue, &filters);
        PreparedStreams {
            tile_bgs,
            matte_ramp_bg,
        }
    }

    /// Encode pass A: every step of `plan`, in order, into `target`.
    ///
    /// `scratch` is the extra target sets the frame's bounces need, sized to match
    /// `target`. It may be `None` only when `plan.scratch` is empty — which
    /// [`Self::ensure_scratch`] guarantees, having been given that very vector.
    ///
    /// **No recursion, no cursors, no parity.** All three were decided in
    /// [`Plan::build`] and are read back off the steps here; what is left is a
    /// `match` that resolves three slot names against real targets.
    fn encode_plan(
        &self,
        p: &CompositorPipeline,
        encoder: &mut wgpu::CommandEncoder,
        target: Targets<'_>,
        plan: &Plan<'_>,
        streams: &PreparedStreams<'_>,
        scratch: Option<&ScratchTargets>,
    ) {
        // The parity claim, checked where it is relied on: whatever the plan did, the
        // accumulator ends in the caller's own targets. That is what lets the media
        // pass keep one bind group across every document and the eyedropper read back
        // the buffers it supplied. `Plan`'s tests pin it for every shape they know;
        // this catches one they do not.
        debug_assert_eq!(
            plan.steps.last().map(Step::out),
            Some(Slot::Target),
            "the ping-pong must land the accumulator in the caller's targets (§14.7)",
        );
        let levels: &[ScratchLevel] = scratch.map_or(&[], |s| &s.levels);
        // The one place a `Slot` becomes a texture. Both `expect`s are the plan's own
        // invariant read back — `Plan`'s tests assert that no step names a level the
        // scratch was not told to allocate — rather than a condition this file could
        // get out of step with.
        let at = |slot: Slot| match slot {
            Slot::Target => target,
            Slot::Swap(l) => levels
                .get(l)
                .expect("a bounce at an unallocated level (Plan::scratch)")
                .swap(),
            Slot::Iso(l) => levels
                .get(l)
                .expect("an isolation at an unallocated level (Plan::scratch)")
                .iso(),
        };
        let e = Encode { p, streams };
        for step in &plan.steps {
            match step {
                Step::Draw { into, draws, clear } => {
                    self.encode_items(&e, encoder, at(*into), &plan.draws[draws.clone()], *clear)
                }
                Step::Clear { into } => blend::clear_targets(encoder, at(*into)),
                Step::Blend {
                    back,
                    src,
                    out,
                    slot,
                } => self.encode_blend(&e, encoder, at(*back), at(*src), at(*out), *slot),
                Step::Filter { back, out, slot } => {
                    self.encode_filter(&e, encoder, at(*back), at(*out), *slot)
                }
            }
        }
    }

    /// Draw one run's items into `into`, in stack order, switching pipelines where
    /// a matte sits between runs of tiles. Both pipelines share group 0 (the view
    /// uniform), so only the vertex buffer and pipeline change.
    ///
    /// Each [`Draw`] carries its own index into the flat streams, which is what the
    /// walking cursors used to reconstruct.
    fn encode_items(
        &self,
        e: &Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
        into: Targets<'_>,
        draws: &[Draw],
        clear: bool,
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
        pass.set_bind_group(0, &self.view.tiles, &[]);
        let mut pipeline_is_matte = None;
        for draw in draws {
            match *draw {
                Draw::Tile(i) => {
                    if pipeline_is_matte != Some(false) {
                        pass.set_pipeline(&e.p.tiles.pipeline);
                        pass.set_vertex_buffer(0, self.instances.slice());
                        pipeline_is_matte = Some(false);
                    }
                    pass.set_bind_group(1, e.streams.tile_bgs[i as usize], &[]);
                    pass.draw(0..4, i..i + 1);
                }
                Draw::Matte(i) => {
                    if pipeline_is_matte != Some(true) {
                        pass.set_pipeline(&e.p.tiles.matte_pipeline);
                        pass.set_vertex_buffer(0, self.matte_instances.slice());
                        pipeline_is_matte = Some(true);
                    }
                    // Group 1 is the ramp, at this matte's own slot (§22.4).
                    // Re-set per matte either way: the offset changes, and a
                    // pipeline switch has already invalidated whatever was
                    // bound, the two pipelines' group-1 layouts differing.
                    let ramp = e
                        .streams
                        .matte_ramp_bg
                        .as_ref()
                        .expect("a matte draw without its ramp slots");
                    pass.set_bind_group(1, ramp, &[UniformSlots::<Ramp>::offset(i)]);
                    pass.draw(0..4, i..i + 1);
                }
            }
        }
    }

    /// Merge the isolated layer `src` into the accumulator `back` through blend slot
    /// `slot`, writing the result to `out` (§18.0.4).
    fn encode_blend(
        &self,
        e: &Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
        back: Targets<'_>,
        src: Targets<'_>,
        out: Targets<'_>,
        slot: u32,
    ) {
        let mut entries = vec![
            self.blend_uniforms.binding(0),
            desc::tex(1, back.color),
            desc::tex(2, back.aux),
            desc::tex(3, src.color),
            desc::tex(4, src.aux),
            desc::tex(5, &e.p.blend.pigment.view),
            desc::samp(6, &e.p.blend.pigment.sampler),
        ];
        // Both residuals or neither: `back`, `src` and `out` are all targets of the
        // same document, so the space that gave one a residual gave all three one.
        if let (Some(b), Some(s)) = (back.resid, src.resid) {
            entries.push(desc::tex(7, b));
            entries.push(desc::tex(8, s));
        }
        let bg =
            e.p.ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("stark blend bg"),
                    layout: &e.p.blend.bgl,
                    entries: &entries,
                });
        let attachments = out.attachments(desc::CLEAR);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("stark blend pass"),
            // The pass covers every texel and reads nothing from `out`, so the load
            // is a don't-care; clearing states that rather than implying the previous
            // contents matter.
            color_attachments: &attachments[..out.count()],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&e.p.blend.pipeline);
        pass.set_bind_group(0, &bg, &[UniformSlots::<BlendUniform>::offset(slot)]);
        pass.draw(0..3, 0..1);
    }

    /// Run filter slot `slot` over the accumulator `back`, writing the adjusted
    /// result to `out` (§21.3).
    ///
    /// [`Self::encode_blend`] with the source dropped, down to the bind group's
    /// numbering: `filter_common.wesl` declares 0–3 where `blend_common.wesl`
    /// declares 0–4 (3 is the chromatic gather's sampler, in one of the slots the
    /// missing source vacates), and the pigment LUT keeps 5–6 because
    /// `mixbox_lut.wesl` hard-codes them for whoever imports it. The LUT itself is
    /// the blend pass's — both passes ask it the same question, and an Oklab
    /// document binds the same 1×1 stand-in so there is one layout per space
    /// rather than one per pass.
    fn encode_filter(
        &self,
        e: &Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
        back: Targets<'_>,
        out: Targets<'_>,
        slot: u32,
    ) {
        let mut entries = vec![
            self.filter_uniforms.binding(0),
            desc::tex(1, back.color),
            desc::tex(2, back.aux),
            desc::samp(3, &e.p.filter.sampler),
            desc::tex(5, &e.p.blend.pigment.view),
            desc::samp(6, &e.p.blend.pigment.sampler),
        ];
        if let Some(r) = back.resid {
            entries.push(desc::tex(7, r));
        }
        let bg =
            e.p.ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("stark filter bg"),
                    layout: &e.p.filter.bgl,
                    entries: &entries,
                });
        let attachments = out.attachments(desc::CLEAR);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("stark filter pass"),
            // Covers every texel and reads nothing from `out` — including the aux,
            // which it copies across from `back` rather than leaving to a load op,
            // since `out` is the other half of a ping-pong and holds a stale bounce.
            color_attachments: &attachments[..out.count()],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&e.p.filter.pipeline);
        pass.set_bind_group(0, &bg, &[UniformSlots::<FilterUniform>::offset(slot)]);
        pass.draw(0..3, 0..1);
    }

    /// Make sure the cached scratch targets match `size`, if `groups` needs any, and
    /// say whether they are wanted. `false` when every group is `Normal` — the common
    /// case, which never allocates.
    ///
    /// The cache belongs to the *render* path, whose size changes only when this
    /// compositor's target does. [`Self::composite_channels`] deliberately does not
    /// use it: a pick viewport is a handful of texels, and letting the two share one
    /// cache would reallocate a target-sized pair twice a frame for the whole of an
    /// Alt-drag.
    /// `needs` is [`scratch_needs`]' answer for this frame, computed by the caller
    /// because the supersampling decision above wanted it first: how many levels there
    /// are, and how many of them isolate, is most of what a zoomed-out frame costs in
    /// memory (`resolve::attachment_bytes`).
    fn ensure_scratch(
        &mut self,
        p: &CompositorPipeline,
        size: Extent2,
        mut needs: Vec<bool>,
    ) -> bool {
        if needs.is_empty() {
            return false;
        }
        // Grown to the most the document has *reached* — depth and per-level iso
        // alike — and never shrunk within a size: a group opened and closed again
        // over and over would otherwise reallocate viewport-sized targets each
        // time. The union with what the cache already holds is what makes "never
        // shrunk" one rule for both axes.
        if let Some(s) = self.scratch.as_ref().filter(|s| s.size == size) {
            if needs.len() < s.levels.len() {
                needs.resize(s.levels.len(), false);
            }
            for (need, level) in needs.iter_mut().zip(&s.levels) {
                *need |= level.has_iso();
            }
            let sufficient = s.levels.len() == needs.len()
                && s.levels.iter().zip(&needs).all(|(l, &n)| l.has_iso() || !n);
            if sufficient {
                return true;
            }
        }
        self.scratch = Some(ScratchTargets::new(&p.ctx.device, size, &needs, p.formats));
        true
    }

    /// Composite `items` into caller-supplied targets and **stop there** — pass A
    /// alone, with no media pass over it.
    ///
    /// This is the eyedropper's sampling path (§18.0.2). What lands in
    /// `color` is the paint's own channels in the document's working space, which is
    /// what a picker has to read: the lit result has been through image-based
    /// lighting, a tonemap and an sRGB encode, so picking *that* would hand back a
    /// color the palette never mixed — and in a Mixbox document (§6.7) a
    /// pigment mixture that cannot be picked back up, which is the point of mixing
    /// in pigment space at all.
    ///
    /// `into` must carry the formats [`CompositorPipeline::channel_formats`] reports
    /// and be `view.viewport` in size. It is the caller's, not this compositor's: a
    /// sample is taken through the compositor that belongs to the screen, so it must
    /// leave the screen's own attachments — a few hundred texels wide against the
    /// window's millions — exactly where they were. That is why this does not go
    /// through [`Self::ensure_targets`], and why the blend scratch below is its own
    /// too.
    ///
    /// A [`Targets`] rather than three views, which is how the residual stops being a
    /// caller's decision. A pigment document's pass A **writes three attachments**,
    /// so a caller offering two is missing one — a validation error the Oklab half of
    /// the suite cannot see, guarded here by a `debug_assert` that only debug builds
    /// ran. `ChannelFormats` exists to make "all three or none of them" unsayable
    /// (§6.7); taking the trio it produces is what lets this path inherit that
    /// instead of re-checking it.
    pub(crate) fn composite_channels(
        &mut self,
        p: &CompositorPipeline,
        into: Targets<'_>,
        view: ViewTransform,
        groups: &[CompositeGroup],
    ) {
        debug_assert_eq!(
            into.count(),
            p.formats.count(),
            "pass A's attachment count is the color space's, not the caller's",
        );
        let plan = Plan::build(groups);
        let streams = self.upload(p, view, &plan);
        // Its own scratch, thrown away with the call. A pick viewport is `2r+1`
        // square, so this is a few kilobytes; sharing the render path's cache would
        // trade that for reallocating the *window* twice a frame (see
        // [`Self::ensure_scratch`]). Blend modes have to be honoured here or an
        // eyedropper would report a color the screen never showed.
        let scratch = (!plan.scratch.is_empty())
            .then(|| ScratchTargets::new(&p.ctx.device, view.viewport, &plan.scratch, p.formats));
        let mut encoder = p
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("stark pick encoder"),
            });
        self.encode_plan(p, &mut encoder, into, &plan, &streams, scratch.as_ref());
        p.ctx.queue.submit([encoder.finish()]);
    }

    /// Composite `scene`'s layers, light the result into `target` under `view`, and
    /// outline each of its selections over it (§6.8 — a universal selection
    /// draws nothing, so an unmasked document costs one skipped iteration).
    ///
    /// A zoomed-out view runs the whole of that at `ss` samples per axis and boxes the
    /// result down at the end (§6.4). Everything between here and the resolve
    /// is written against `view` alone, so supersampling is one substitution at the
    /// top and one pass at the bottom rather than a parameter every pass has to carry.
    pub fn render(
        &mut self,
        p: &CompositorPipeline,
        target: &wgpu::TextureView,
        view: ViewTransform,
        scene: CompositeScene<'_>,
    ) {
        let CompositeScene {
            background: bg_channels,
            background_resid: bg_resid,
            groups,
            outlines,
            transparent,
            guides,
        } = scene;
        // How hard this view is minifying, and therefore how many samples per output
        // pixel it takes to stop the paint, the weave and the impasto relief aliasing
        // (§6.4). 1 at 1:1 and closer, where the rest of this is a no-op.
        // Everything this frame's pass A does, decided once (§14.7). It comes first
        // because the sample count below is chosen from what it costs: the scratch
        // is most of a zoomed-out frame's memory — two viewport-sized trios per
        // isolating level — so the group tree has to be walked *before* the
        // attachments are sized, not after (§6.4).
        //
        // Deliberately free of the view, which is why it can be built here at all:
        // the one view-dependent number in pass A is a filter's dispersion, and that
        // is filled in at `upload` once `ss` has settled (§21.10).
        let plan = Plan::build(groups);
        let ss = supersample(
            view.viewport,
            view.zoom,
            &p.ctx.device.limits(),
            resolve::attachment_bytes(p.formats, p.target_format, &plan.scratch),
        );
        // This compositor's attachments, brought in line with what is about to be
        // drawn. Nobody else's: a render into something other than this target — an
        // export, the navigator's miniature — goes through a `Compositor` of its own,
        // so the surface's attachments (and the frame already presented from them)
        // are never resized out from under it and rebuilt on the next frame.
        self.ensure_targets(p, view.viewport, ss);
        // From here down, `view` is the supersampled one and `target` is the only
        // thing that still knows the real size — which is exactly the split the
        // resolve at the bottom closes.
        let view = view.supersampled(ss);
        let streams = self.upload(p, view, &plan);
        let want_scratch = self.ensure_scratch(p, self.size, plan.scratch.clone());
        // Bound after everything that needs `&mut self`.
        let device = &p.ctx.device;
        let scratch = if want_scratch {
            self.scratch.as_ref()
        } else {
            None
        };
        let accum = self
            .accum
            .as_ref()
            .expect("ensure_targets builds the accumulator");
        // What the lit image, the outlines and the guides are drawn into: the
        // supersampled target when there is one, else the caller's directly. Chrome
        // goes through the same resolve as the paint, so the marching ants and the
        // perspective grid come out antialiased rather than as the stairs a
        // one-sample-per-pixel line draws at any angle but the axes.
        let draw_target = self.ss_target.as_ref().map_or(target, |s| &s.view);

        // Screen→canvas mapping for sampling the surface bump in canvas space, so the
        // weave stays attached to the canvas as it pans, zooms, turns and mirrors
        // (§6.4, §18.1.2).
        let inv_zoom = 1.0 / view.zoom;
        let inv_linear = view.inverse_linear();
        let canvas_origin = view.screen_to_canvas(crate::geom::Vec2::ZERO);

        // Diffuse samples a heavily-blurred high mip ≈ hemispherical irradiance; the
        // level is the environment's own, so the CPU-side normalization below is
        // reading exactly the texels the shader will. The Cook–Torrance specular picks
        // its own mip from roughness, spanning the whole chain (roughness 0 → mip 0
        // sharp; roughness 1 → the diffuse level, the hemispherical average).
        let diffuse_lod = p.environment.diffuse_lod as f32;
        // Exposure belongs to the light, not to a knob beside it: each environment is
        // shown at the value it was judged at (§6.3). Normalized by the
        // irradiance a *flat* canvas receives, so `1.0` means the same thing in every
        // environment — an unrelieved patch of paint comes back out its own color.
        let exposure = p.environment.exposure / p.environment.flat_irradiance;

        // Media uniform.
        let m = p.media_params;
        p.ctx.queue.write_buffer(
            &self.media_buf,
            0,
            bytemuck::bytes_of(&MediaUniform {
                bg: bg_channels,
                bg_resid,
                shade: [exposure, diffuse_lod, m.specular, m.height_strength],
                surf_a: [
                    canvas_origin.x,
                    canvas_origin.y,
                    inv_zoom,
                    1.0 / SURFACE_TILE_PX,
                ],
                surf_b: [
                    m.surface_strength,
                    // Transparent export: the media pass skips the substrate and
                    // carries the paint's visible alpha out (§15.6).
                    if transparent { 1.0 } else { 0.0 },
                    0.0,
                    0.0,
                ],
                surf_m: inv_linear.to_cols_array(),
            }),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stark composite encoder"),
        });

        // Pass A: composite tiles into offscreen color + aux. The parity trick in
        // `encode_composite` guarantees the result lands in these two views however
        // many blend passes ran, so the media bind group never has to be rebuilt.
        self.encode_plan(p, &mut encoder, accum.targets(), &plan, &streams, scratch);

        // Pass B: media/lighting → target.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stark media pass"),
                // The media pass covers every texel, so the clear only matters for
                // what alpha an untouched texel would keep — transparent, on an
                // export that wants a cut-out.
                color_attachments: &[Some(desc::attach(
                    draw_target,
                    desc::clear_to(if transparent {
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
            pass.set_pipeline(&p.media_pass.pipeline);
            pass.set_bind_group(0, &accum.bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // Pass C: the selection outlines, over the lit image — the local actor's and
        // every present peer's, one instanced quad per mask tile of each
        // (§17.3). Flattened into one instance stream so N collaborators
        // still cost one pass.
        let mut overlay_instances: Vec<OverlayInstance> = Vec::new();
        let mut mask_tiles: Vec<&wgpu::BindGroup> = Vec::new();
        for outline in outlines {
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
                overlay_instances.push(OverlayInstance {
                    origin: coord.origin().to_array(),
                    tint,
                    level,
                });
                // Kept on the mask tile, like pass A's on the paint tile: the ants
                // redraw every frame a selection is live, and the mask is immutable.
                mask_tiles.push(handle.overlay_bg(|| {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("stark overlay tile bg"),
                        layout: &p.overlay.tile_bgl,
                        entries: &[desc::tex(0, handle.view())],
                    })
                }));
            }
        }
        if !mask_tiles.is_empty() {
            self.overlay_instances
                .write(device, &p.ctx.queue, &overlay_instances);

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stark selection overlay pass"),
                color_attachments: &[Some(desc::attach(draw_target, desc::LOAD))],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&p.overlay.pipeline);
            pass.set_bind_group(0, &self.view.overlay, &[]);
            pass.set_vertex_buffer(0, self.overlay_instances.slice());
            for (i, bg) in mask_tiles.iter().enumerate() {
                let idx = i as u32;
                pass.set_bind_group(1, *bg, &[]);
                pass.draw(0..4, idx..idx + 1);
            }
        }

        // Pass D: the drawing guides, over the lit image and the outlines —
        // the perspective grid is chrome the whole canvas is read *through*,
        // so it is the topmost thing drawn (§20.4). One render pass; one
        // fullscreen triangle per visible guide, each off its own uniform
        // slot (see [`UniformSlots`] for why they cannot share one).
        if !guides.is_empty() {
            let packed: Vec<GuideUniform> = guides.iter().map(|s| pack_guides(s, view)).collect();
            self.guide_uniforms.write(device, &p.ctx.queue, &packed);
            // Per render rather than kept, like the blend bind group: it has
            // to follow the buffer through reallocation, and a bind group over
            // one small uniform is cheap beside everything else in the frame.
            let guide_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("stark guides bg"),
                layout: &p.guides.bgl,
                entries: &[self.guide_uniforms.binding(0)],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stark guides pass"),
                color_attachments: &[Some(desc::attach(draw_target, desc::LOAD))],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&p.guides.pipeline);
            for i in 0..guides.len() as u32 {
                pass.set_bind_group(0, &guide_bg, &[UniformSlots::<GuideUniform>::offset(i)]);
                pass.draw(0..3, 0..1);
            }
        }

        // Pass E: the resolve — everything above, box-averaged in light down to the
        // caller's target (§6.4). Absent at 1:1, where `draw_target` *is* the
        // caller's target and the picture is already the size it was asked for.
        if let Some(ss_target) = &self.ss_target {
            p.ctx.queue.write_buffer(
                &ss_target.buf,
                0,
                bytemuck::bytes_of(&ResolveUniform {
                    n: [ss as f32, 0.0, 0.0, 0.0],
                }),
            );
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stark resolve pass"),
                // Covers every texel and reads nothing back, so the load is a
                // don't-care; clearing says so rather than implying otherwise.
                color_attachments: &[Some(desc::attach(target, desc::CLEAR))],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&p.resolve.pipeline);
            pass.set_bind_group(0, &ss_target.bg, &[]);
            pass.draw(0..3, 0..1);
        }

        p.ctx.queue.submit([encoder.finish()]);
    }
}

/// A viewport-sized offscreen render target, as pass A and the blend pass use.
pub(super) fn offscreen_view(
    device: &wgpu::Device,
    size: Extent2,
    format: wgpu::TextureFormat,
    label: &str,
) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size.width.max(1),
                height: size.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}
