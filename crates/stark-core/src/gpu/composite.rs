//! Compositing and the media/lighting pass (§6.3, §6.4).
//!
//! Two passes:
//!   A. Composite every visible tile's channels into viewport-sized offscreen
//!      targets — Oklab color (premultiplied "over") and the `(height)` aux
//!      (additive). Layers whose [`BlendMode`] is not `Normal` are isolated and
//!      merged through a blend pass; see [`CompositeGroup`].
//!   B. A fullscreen media pass that derives normals from the height field,
//!      lights the impasto, adds the paint film's gloss, converts Oklab →
//!      display, and composites over the background into the final target.
//!
//! This replaces the step-1 `Presenter` for engine rendering; the height/normal
//! lighting is the "old masters" payoff.
//!
//! # Two types, split along one line: does it depend on the target?
//!
//! [`CompositorPipeline`] is everything that does not — the five pipelines, the
//! layouts, the pigment LUT, and the view settings the media pass reads. A
//! [`Compositor`] is one target's worth of what does: pass A's offscreen
//! attachments, the media bind group over them, the blend scratch, and the instance
//! streams.
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

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::colorspace::ColorSpace;
use crate::document::selection::Selection;
use crate::document::{BlendMode, DRAGO_K};
use crate::geom::{
    Extent2, INTERIOR_UV_BIAS, INTERIOR_UV_SCALE, TILE_SIZE, TileCoord, ViewTransform,
};
use crate::gpu::context::GpuContext;
use crate::gpu::environment::Environment;
use crate::gpu::pigment::PigmentLut;
use crate::gpu::surface::{SURFACE_TILE_PX, Surface};
use crate::gpu::tile::TilePairHandle;

/// Mirrors `View` in `composite.wesl` and `overlay.wesl` (32 bytes).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ViewUniform {
    // The canvas px -> NDC linear map, column-major (`mat2x2` in the shaders). A
    // full 2x2 rather than a scale pair because the view can be turned and mirrored
    // (§18.1.2); upright and unmirrored it is diagonal, and every
    // shader that reads it multiplies the same way either way.
    st: [f32; 4],
    // translate.xy, then padding to the 16-byte stride a uniform member takes
    // anyway.
    xlate: [f32; 4],
    misc: [f32; 4], // tile_size, interior uv scale, interior uv bias, zoom
}

/// Per-tile instance: canvas-space origin + the layer's opacity.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Instance {
    origin: [f32; 2],
    opacity: f32,
}

/// Per-mask-tile instance of the outline pass: where the tile is, and how to draw
/// its contour. `tint.a == 0` selects the local actor's black/white marching ants;
/// anything else draws a flat line in `tint.rgb` at that alpha — which is how
/// another collaborator's selection is distinguished from your own
/// (§17.3).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct OverlayInstance {
    origin: [f32; 2],
    tint: [f32; 4],
}

/// One selection to outline, and whose it is (§17.3).
#[derive(Copy, Clone)]
pub struct SelectionOutline<'a> {
    pub selection: &'a Selection,
    /// `None` for the local actor — the marching ants. `Some(rgb)` for a peer's,
    /// drawn as a flat line in their colour so the two never read as the same thing.
    pub tint: Option<[f32; 3]>,
}

/// Per-matte instance, mirroring `matte.wesl`'s vertex attributes.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MatteInstance {
    rect: [f32; 4],     // min.xy, max.xy in canvas px
    channels: [f32; 4], // fill, in the working color space
    opacity: f32,
    _pad: [f32; 3],
}

/// A matte layer's draw parameters (§15.4).
#[derive(Copy, Clone, Debug)]
pub struct MatteDraw {
    /// The region's rect in canvas px: `min.xy, max.xy`. For a frame this is the
    /// *hole* — the fill covers everything outside it.
    pub rect: [f32; 4],
    /// Fill color in the document's working color space.
    pub channels: [f32; 4],
    /// The layer's opacity.
    pub opacity: f32,
}

/// One item of compositing pass A, in bottom-to-top stack order.
///
/// An ordered list rather than a flat tile array because a matte composites at
/// its own place in the stack — a frame over the painting, a ground under it
/// (§15.4.4). Tiles already cost one draw each (each needs its own
/// bind group), so interleaving mattes adds no per-tile overhead.
#[derive(Clone)]
pub enum CompositeItem {
    Tile {
        coord: TileCoord,
        handle: TilePairHandle,
        opacity: f32,
    },
    Matte(MatteDraw),
}

/// One **blend group** of pass A: something that composites on its own, and how
/// its result merges into everything below it (§18.0.4,
/// §14.7).
///
/// A group is defined against *what is underneath it* — which means it has to be
/// composited alone, on nothing, before it can be merged. That is the per-layer
/// isolation §6.3 names as the prerequisite for richer modes, and layer
/// groups are the same investment recursed: [`GroupContent::Stack`] is a group
/// whose members are themselves groups.
#[derive(Clone)]
pub struct CompositeGroup {
    pub blend: BlendMode,
    /// Clip to the coverage of what this group composites onto
    /// (§14.4). Costs the same isolation a blend mode does, and for the same reason:
    /// the merge has to *read* the backdrop's alpha.
    pub clip: bool,
    /// Applied to this group's whole composited result at the merge, not to its
    /// members one at a time.
    ///
    /// That distinction is only visible on a [`Stack`](GroupContent::Stack),
    /// whose members overlap — a leaf layer's tiles do not, so its opacity rides
    /// on [`CompositeItem::Tile`] and this stays 1.0 for it. Two granularities of
    /// the same fact, and the cheaper one is used wherever it is equivalent.
    pub opacity: f32,
    pub content: GroupContent,
}

/// What a [`CompositeGroup`] is made of — and the fast path, expressed as a
/// shape rather than as a condition inside the encoder.
#[derive(Clone)]
pub enum GroupContent {
    /// A run of the stack that composites internally under plain premultiplied
    /// "over", with **no isolation**.
    ///
    /// Consecutive `Normal`, unclipped layers carrying nothing compose correctly
    /// against each other *and* against the accumulator, so a document that uses
    /// no modes, no clipping and no groups is a single `Run` and costs exactly
    /// what the flat tile list cost before any of this existed.
    Run(Vec<CompositeItem>),
    /// Members composited bottom-to-top, each merging into the one below through
    /// its own blend mode and clip — a **layer group** (§14.2).
    ///
    /// The builder collapses a `Stack` that could not tell itself apart from a
    /// `Run` into one, which is what makes "organization is free" structural
    /// rather than a promise (§14.7 rule 2).
    Stack(Vec<CompositeGroup>),
}

impl CompositeGroup {
    /// A run of drawables that merges outward through `blend` and `clip`.
    pub fn run(blend: BlendMode, clip: bool, items: Vec<CompositeItem>) -> Self {
        Self {
            blend,
            clip,
            opacity: 1.0,
            content: GroupContent::Run(items),
        }
    }

    /// A group of `members` — **collapsed into a plain [`Run`](GroupContent::Run)
    /// when nothing about it could tell itself apart from one**
    /// (§14.7 rule 2).
    ///
    /// This is where "organization is free" is made structural rather than
    /// promised. A group that merges normally, unclipped, at full opacity, and
    /// whose every member draws directly, changes no blending scope: its members
    /// were composing against everything below them under `over` already, and
    /// isolating them would produce the same pixels through two extra render
    /// passes per member. So it produces the identical draw list to no group at
    /// all — which is the property the golden test pins, and the answer to
    /// "grouping my layers changed my painting".
    ///
    /// The condition cannot be relaxed to "the group itself is normal": a member
    /// with a mode of its own *does* blend against a different backdrop once
    /// isolated, and that difference is the feature (§14.5).
    pub fn stack(blend: BlendMode, clip: bool, opacity: f32, members: Vec<Self>) -> Self {
        let free = blend.is_normal() && !clip && opacity >= 1.0;
        if free && members.iter().all(Self::is_direct) {
            let items = members
                .into_iter()
                .flat_map(|m| match m.content {
                    GroupContent::Run(items) => items,
                    GroupContent::Stack(_) => {
                        unreachable!("a direct group is a Run by construction")
                    }
                })
                .collect();
            return Self::run(blend, clip, items);
        }
        Self {
            blend,
            clip,
            opacity,
            content: GroupContent::Stack(members),
        }
    }

    /// Whether this group can be drawn straight into the accumulator with no
    /// isolation — the fast path. True only for an unclipped `Normal` `Run` at
    /// full opacity.
    pub fn is_direct(&self) -> bool {
        self.blend.is_normal()
            && !self.clip
            && self.opacity >= 1.0
            && matches!(self.content, GroupContent::Run(_))
    }

    /// How deep the isolation nests below this group: 0 for a `Run`, one more
    /// than its deepest member for a `Stack`. The scratch stack is sized by this
    /// (§14.7).
    fn depth(&self) -> usize {
        match &self.content {
            GroupContent::Run(_) => 0,
            GroupContent::Stack(members) => 1 + members.iter().map(Self::depth).max().unwrap_or(0),
        }
    }

    /// Every drawable in this group, in composite order — the flat streams pass A
    /// uploads (the draw loop walks the tree, but the instance buffers do not
    /// need to).
    fn items(&self) -> Vec<&CompositeItem> {
        let mut out = Vec::new();
        fn walk<'a>(g: &'a CompositeGroup, out: &mut Vec<&'a CompositeItem>) {
            match &g.content {
                GroupContent::Run(items) => out.extend(items.iter()),
                GroupContent::Stack(members) => members.iter().for_each(|m| walk(m, out)),
            }
        }
        walk(self, &mut out);
        out
    }
}

/// Mirrors `Blend` in `blend_common.wesl` (16 bytes).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BlendUniform {
    mode: u32,
    k: f32,
    clip: u32,
    /// The group's own opacity, applied to its composited result at the merge —
    /// which is the *only* place it can be applied for a group, since its members
    /// overlap (§14.7). Always 1.0 for a leaf layer, whose opacity
    /// rides on its tiles instead.
    opacity: f32,
}

/// The shader ABI for [`BlendMode`], kept here rather than on the enum: which `u32`
/// a mode is numbered is a fact about `blend_common.wesl`, not about the document.
///
/// `Normal` reaches the pass only when the group is **clipped** or carries an
/// opacity of its own (§14.4); an ordinary normal layer is the
/// absence of a pass.
fn blend_code(mode: BlendMode) -> u32 {
    match mode {
        BlendMode::Normal => 0,
        BlendMode::Reinhard => 1,
        BlendMode::Drago => 2,
        BlendMode::Multiply => 3,
    }
}

/// One dynamic-offset slot of the blend uniform, padded to the alignment every
/// backend accepts (`min_uniform_buffer_offset_alignment` is 256 on the strictest).
///
/// A slot per blend group rather than one buffer rewritten per pass: `write_buffer`
/// is a *queue* operation, so N rewrites before a single submit would leave every
/// pass reading the last mode written. Two blend layers in one document is not an
/// edge case, so the buffer holds them all and each pass binds its own offset.
const BLEND_SLOT: u64 = 256;

/// The extra viewport-sized targets **one level** of isolation needs
/// (§18.0.4).
///
/// Two pairs, not one. `iso` is where a group composites alone; `swap` is the other
/// half of a ping-pong, because the blend pass reads the accumulator and writes the
/// merged result and a texture cannot be both.
struct ScratchLevel {
    swap_color: wgpu::TextureView,
    swap_aux: wgpu::TextureView,
    iso_color: wgpu::TextureView,
    iso_aux: wgpu::TextureView,
}

impl ScratchLevel {
    fn new(
        device: &wgpu::Device,
        size: Extent2,
        color_format: wgpu::TextureFormat,
        aux_format: wgpu::TextureFormat,
    ) -> Self {
        let make = |format, label| offscreen_view(device, size, format, label);
        Self {
            swap_color: make(color_format, "stark blend swap color"),
            swap_aux: make(aux_format, "stark blend swap aux"),
            iso_color: make(color_format, "stark blend iso color"),
            iso_aux: make(aux_format, "stark blend iso aux"),
        }
    }

    fn swap(&self) -> Targets<'_> {
        (&self.swap_color, &self.swap_aux)
    }

    fn iso(&self) -> Targets<'_> {
        (&self.iso_color, &self.iso_aux)
    }
}

/// One [`ScratchLevel`] per level of group nesting the document actually reaches
/// (§14.7).
///
/// A group's members isolate into *its* level's `iso`, which is the target the
/// next level down composites into — so nesting costs one of these per level and
/// not one per group. Allocated only when a document contains something that has
/// to be isolated at all: an ordinary painting never pays the ~40 MB, and one
/// that uses blend modes without groups pays for exactly the one level it uses.
struct ScratchTargets {
    size: Extent2,
    levels: Vec<ScratchLevel>,
}

impl ScratchTargets {
    fn new(
        device: &wgpu::Device,
        size: Extent2,
        levels: usize,
        color_format: wgpu::TextureFormat,
        aux_format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            size,
            levels: (0..levels)
                .map(|_| ScratchLevel::new(device, size, color_format, aux_format))
                .collect(),
        }
    }
}

/// How many [`ScratchLevel`]s composting `members` as one stack takes: none if
/// every member draws straight into the accumulator, else one for this stack plus
/// however many the deepest nested group below it needs.
fn scratch_levels(members: &[CompositeGroup]) -> usize {
    if members.iter().all(CompositeGroup::is_direct) {
        return 0;
    }
    1 + members.iter().map(CompositeGroup::depth).max().unwrap_or(0)
}

/// A color + aux target pair, as pass A hands one around.
type Targets<'a> = (&'a wgpu::TextureView, &'a wgpu::TextureView);

/// How far through the frame's flat streams the encoder has drawn.
///
/// The instance buffers and the blend uniform's slots are flat across the whole
/// group tree, while the drawing walks it — so the walk carries one of these and
/// each run, each merge, takes the next entry. They are `u32` because that is what
/// `draw` and the dynamic bind-group offset take.
#[derive(Default)]
struct Cursors {
    tile: u32,
    matte: u32,
    blend: u32,
}

/// What one render draws, as against *where and how* it draws it (the target and the
/// view, which stay separate parameters).
///
/// These four travel together because they are one description of the document at an
/// instant, assembled in one place — [`Engine::render_view`](crate::Engine) — and
/// meaningless apart: the ground belongs under the stack, the outlines over it, and
/// `transparent` says whether the ground is drawn at all.
pub struct CompositeScene<'a> {
    /// The substrate colour in the document's working channels — the ground under
    /// the paint (§15.5).
    pub background: [f32; 4],
    /// The visible layers, bottom-to-top, cut into blend groups.
    pub groups: &'a [CompositeGroup],
    /// Selection outlines to draw over the lit result: the local actor's and each
    /// present peer's (§17.3). Empty for anything that is not the screen
    /// — chrome is a thing to draw *with* (§15.6).
    pub outlines: &'a [SelectionOutline<'a>],
    /// Leave the substrate out and carry the paint's own alpha to the target, for a
    /// cut-out export.
    pub transparent: bool,
}

/// Mirrors `Media` in `media_common.wesl` (80 bytes).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MediaUniform {
    light: [f32; 4], // _, _, _, height_strength (relief slope; xyz unused under IBL)
    bg: [f32; 4],    // background (substrate) in latent channels (xyz), unused w
    shade: [f32; 4], // exposure, diffuse_lod, gloss, _
    // Screen→canvas mapping + surface (bump) sampling for the canvas relief:
    surf_a: [f32; 4], // canvas_origin.xy (canvas px at pixel 0), canvas_per_px, inv_tile
    surf_b: [f32; 4], // surface_strength, transparent (0/1), _, _
    // The screen→canvas linear map, column-major: what carries a fragment's position
    // into canvas space so the weave stays attached to the canvas however the view is
    // turned or mirrored. `surf_a.z` is the same map's *length* scale, which rotation
    // and mirroring leave alone, and which the relief slope still wants as a scalar.
    surf_m: [f32; 4],
}

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

/// Everything about compositing that does not depend on *what is being drawn into*:
/// the pipelines and layouts, the uniform buffers whose identity never changes, the
/// pigment LUT, and the view settings the media pass reads.
///
/// Split from [`Compositor`] so several of them can share one of these. Each renders
/// into a target of its own size and therefore keeps its own attachments; what they
/// must *not* keep their own of is anything on this side of the line — the pipelines
/// because they are expensive (five of them, plus a decoded Mixbox LUT), and the view
/// settings because two consumers disagreeing about the canvas weave or the lighting
/// would be a bug that shows only in the smaller picture.
///
/// Not immutable: the view settings change. It is immutable *during a render*, which
/// is what lets every consumer hold it by shared reference. The uniform buffers are
/// written through the queue rather than through `&mut`, and renders are sequential
/// on one queue, so those writes stay ordered with the submits that read them.
pub struct CompositorPipeline {
    ctx: GpuContext,

    // Pass A: composite tiles into offscreen targets.
    composite_pipeline: wgpu::RenderPipeline,
    view_buf: wgpu::Buffer,
    view_bg: wgpu::BindGroup,
    tile_bgl: wgpu::BindGroupLayout,

    // Matte layers, drawn inside pass A at their place in the stack
    // (§15.4). Its own pipeline because its blend state differs from
    // the color space's: `over` on *both* targets, so an opaque matte erases the
    // relief beneath it rather than letting underlying impasto emboss through.
    matte_pipeline: wgpu::RenderPipeline,

    // Per-layer blend modes, inside pass A (§18.0.4). One
    // fullscreen draw merging an isolated layer into the accumulator.
    blend_pipeline: wgpu::RenderPipeline,
    blend_bgl: wgpu::BindGroupLayout,
    pigment: PigmentLut,

    // Pass C: the selection outline, drawn over the lit result (§6.8).
    // One instanced quad per mask tile, in the same canvas→NDC frame as pass A.
    overlay_pipeline: wgpu::RenderPipeline,
    overlay_view_bg: wgpu::BindGroup,
    overlay_tile_bgl: wgpu::BindGroupLayout,

    // Pass B: media/lighting → final target.
    media_pipeline: wgpu::RenderPipeline,
    media_buf: wgpu::Buffer,
    media_bgl: wgpu::BindGroupLayout,
    media: MediaParams,

    // Offscreen channel formats (from the color space).
    color_format: wgpu::TextureFormat,
    aux_format: wgpu::TextureFormat,

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
    /// The case that needs that is a colour-space rebuild (§6.7), which does
    /// not mutate a pipeline but *replaces* it. A per-pipeline counter would start the
    /// replacement back at its initial value — the very value a consumer that had
    /// rendered against the old pipeline is holding — so a kept [`Compositor`] would
    /// see "no change" and keep attachments belonging to the pipeline that is gone.
    /// Today both colour spaces happen to use the same channel formats, so that would
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
/// through, and the instance streams. Everything here is sized either by the target
/// or by how much there is to draw.
///
/// One per thing being drawn into — the surface, and (with its own) anything that
/// renders beside it: an export, the navigator's miniature. Sharing one across
/// targets of different sizes means each render resizes the attachments the other
/// just built, so the cost is paid twice per alternation rather than once ever.
pub struct Compositor {
    // Target-sized offscreen attachments, and the media bind group over them.
    size: Extent2,
    comp_color_view: wgpu::TextureView,
    comp_aux_view: wgpu::TextureView,
    media_bg: wgpu::BindGroup,
    /// The [`CompositorPipeline::generation`] `media_bg` was built against.
    generation: u64,

    // Allocated on first use and kept: only a document with a non-`Normal` layer
    // ever pays for them.
    scratch: Option<ScratchTargets>,

    // Pass A's instance streams, grown to the frame's tile and matte counts.
    instances: wgpu::Buffer,
    instance_cap: usize,
    matte_instances: wgpu::Buffer,
    matte_cap: usize,
    // One dynamic-offset slot per blend group in the frame.
    blend_buf: wgpu::Buffer,
    blend_slots: usize,
    // Pass C's, grown to the outlined mask-tile count.
    overlay_instances: wgpu::Buffer,
    overlay_cap: usize,
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
    /// The compositor, built against `p` at `size` if this is the first use.
    pub(crate) fn get(&mut self, p: &CompositorPipeline, size: Extent2) -> &mut Compositor {
        self.0.get_or_insert_with(|| Compositor::new(p, size))
    }
}

impl CompositorPipeline {
    pub fn new(
        ctx: &GpuContext,
        target_format: wgpu::TextureFormat,
        color_space: &dyn ColorSpace,
        surface: Surface,
        environment: Environment,
    ) -> Self {
        let device = &ctx.device;
        let color_format = color_space.color_format();
        let aux_format = color_space.aux_format();

        // ---- Pass A: composite (generic passthrough; blends from color space) ----
        let comp_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark composite"),
            source: wgpu::ShaderSource::Wgsl(stark_shaders::composite().into()),
        });

        let view_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stark composite view bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let tile_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stark composite tile bgl"),
            entries: &[tex_entry(0), tex_entry(1)],
        });

        let comp_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("stark composite layout"),
            bind_group_layouts: &[Some(&view_bgl), Some(&tile_bgl)],
            immediate_size: 0,
        });

        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("stark composite pipeline"),
            layout: Some(&comp_layout),
            vertex: wgpu::VertexState {
                module: &comp_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Instance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32],
                })],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &comp_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(color_space.color_blend()),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: aux_format,
                        blend: Some(color_space.aux_blend()),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
            }),
            multiview_mask: None,
            cache: None,
        });

        let view_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stark composite view"),
            size: std::mem::size_of::<ViewUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("stark composite sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let view_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark composite view bg"),
            layout: &view_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: view_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // ---- Matte layers, inside pass A (§15.4) ----
        //
        // Reuses pass A's view bind group (vertex-only uniform: the fragment stage
        // gets canvas position as a varying, and the zoom rides through `misc.w`
        // for the edge antialiasing width).
        let matte_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark matte"),
            source: wgpu::ShaderSource::Wgsl(stark_shaders::matte().into()),
        });
        let matte_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("stark matte layout"),
            bind_group_layouts: &[Some(&view_bgl)],
            immediate_size: 0,
        });
        let matte_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("stark matte pipeline"),
            layout: Some(&matte_layout),
            vertex: wgpu::VertexState {
                module: &matte_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<MatteInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Float32],
                })],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &matte_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                // Premultiplied `over` on BOTH targets. The aux one is the load-
                // bearing difference from pass A's additive aux: additive would
                // keep the height of paint *underneath* the matte, and the media
                // pass would emboss that paint's impasto as ghost ridges through
                // an opaque mat board (§15.4.2). `OneMinusSrcAlpha`
                // is valid on the alpha-less R16Float aux: the factor reads the
                // *source* alpha from the shader's output vec4.
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: aux_format,
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
            }),
            multiview_mask: None,
            cache: None,
        });
        // ---- Per-layer blend, inside pass A (§18.0.4) ----
        //
        // A fullscreen pass reading the accumulator and one isolated layer, writing
        // the merge to the *other* accumulator. Its own bind group layout: every
        // texture here is read with `textureLoad` at the fragment's own coordinate,
        // so nothing needs filtering — except the pigment LUT, which is a table
        // Mixbox interpolates in hardware (`mixbox_lut.wesl`).
        let blend_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark blend"),
            source: wgpu::ShaderSource::Wgsl(color_space.blend_shader().into()),
        });
        let blend_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stark blend bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        // One slot per blend group in the frame; see `BLEND_SLOT`.
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<BlendUniform>() as u64,
                        ),
                    },
                    count: None,
                },
                load_tex_entry(1), // accumulator color
                load_tex_entry(2), // accumulator aux
                load_tex_entry(3), // isolated layer color
                load_tex_entry(4), // isolated layer aux
                tex_entry(5),      // pigment LUT (filtered)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let blend_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("stark blend layout"),
            bind_group_layouts: &[Some(&blend_bgl)],
            immediate_size: 0,
        });
        let blend_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("stark blend pipeline"),
            layout: Some(&blend_layout),
            vertex: wgpu::VertexState {
                module: &blend_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &blend_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                // No fixed-function blend on either target: the pass computes the
                // whole merge — backdrop included — and *replaces* what it writes.
                // That is the point of the ping-pong.
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: aux_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
            }),
            multiview_mask: None,
            cache: None,
        });
        // Decoded only where it is read from: an Oklab document gets a 1×1 stand-in
        // so the one bind group layout still has something to bind.
        let pigment = if color_space.needs_pigment_lut() {
            PigmentLut::load(ctx)
        } else {
            PigmentLut::placeholder(ctx)
        };

        // ---- Pass C: selection outline (§6.8) ----
        //
        // Its own view bind group rather than pass A's: the fragment stage needs the
        // uniform too (it converts a canvas-space distance to screen px with the
        // zoom), and pass A declares it vertex-only.
        let overlay_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark selection overlay"),
            source: wgpu::ShaderSource::Wgsl(stark_shaders::overlay().into()),
        });
        let overlay_view_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stark overlay view bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let overlay_tile_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stark overlay tile bgl"),
            entries: &[tex_entry(0)],
        });
        let overlay_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("stark overlay layout"),
            bind_group_layouts: &[Some(&overlay_view_bgl), Some(&overlay_tile_bgl)],
            immediate_size: 0,
        });
        let overlay_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("stark overlay pipeline"),
            layout: Some(&overlay_layout),
            vertex: wgpu::VertexState {
                module: &overlay_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<OverlayInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4],
                })],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &overlay_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    // The outline is drawn *over* the finished image, so it is the one
                    // pass that blends in straight (non-premultiplied) alpha.
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let overlay_view_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark overlay view bg"),
            layout: &overlay_view_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: view_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // ---- Pass B: media ----
        let media_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stark media"),
            source: wgpu::ShaderSource::Wgsl(color_space.media_shader().into()),
        });
        let media_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stark media bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                load_tex_entry(1), // comp_color (textureLoad)
                load_tex_entry(2), // comp_aux   (textureLoad)
                tex_entry(3),      // surface bump (filtered)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                tex_entry(5), // environment (filtered, mipped)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let media_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("stark media layout"),
            bind_group_layouts: &[Some(&media_bgl)],
            immediate_size: 0,
        });
        let media_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("stark media pipeline"),
            layout: Some(&media_layout),
            vertex: wgpu::VertexState {
                module: &media_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &media_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let media_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stark media uniform"),
            size: std::mem::size_of::<MediaUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            ctx: ctx.clone(),
            composite_pipeline,
            view_buf,
            view_bg,
            tile_bgl,
            matte_pipeline,
            blend_pipeline,
            blend_bgl,
            pigment,
            overlay_pipeline,
            overlay_view_bg,
            overlay_tile_bgl,
            media_pipeline,
            media_buf,
            media_bgl,
            media: MediaParams::default(),
            color_format,
            aux_format,
            surface,
            environment,
            generation: next_generation(),
        }
    }

    /// The current media/lighting parameters (§6.3).
    pub fn media(&self) -> MediaParams {
        self.media
    }

    /// Adjust the media/lighting parameters (§6.3).
    pub fn set_media(&mut self, media: MediaParams) {
        self.media = media;
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

    /// The raw channel formats pass A writes: `(color, aux)`. A caller supplying its
    /// own targets to [`Compositor::composite_channels`] has to match them.
    pub fn channel_formats(&self) -> (wgpu::TextureFormat, wgpu::TextureFormat) {
        (self.color_format, self.aux_format)
    }
}

impl Compositor {
    /// Attachments and instance streams for one target of `size`, against the shared
    /// `pipeline`. Cheap — everything expensive (five pipelines, the layouts, the
    /// decoded pigment LUT) lives in the pipeline and is only borrowed.
    pub fn new(pipeline: &CompositorPipeline, size: Extent2) -> Self {
        let device = &pipeline.ctx.device;
        let (comp_color_view, comp_aux_view, media_bg) = make_offscreen(OffscreenDesc {
            device,
            size,
            color_format: pipeline.color_format,
            aux_format: pipeline.aux_format,
            media_bgl: &pipeline.media_bgl,
            media_buf: &pipeline.media_buf,
            surface: &pipeline.surface,
            environment: &pipeline.environment,
        });
        Self {
            size,
            comp_color_view,
            comp_aux_view,
            media_bg,
            generation: pipeline.generation,
            scratch: None,
            instances: alloc_instances(device, 1),
            instance_cap: 1,
            matte_instances: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("stark matte instances"),
                size: std::mem::size_of::<MatteInstance>() as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            matte_cap: 1,
            blend_buf: alloc_blend(device, 1),
            blend_slots: 1,
            overlay_instances: alloc_overlay(device, 1),
            overlay_cap: 1,
        }
    }

    /// Bring the attachments in line with what is about to be drawn: `size`, and the
    /// pipeline's current surface/environment.
    ///
    /// Called at the top of every render, so a resized target, a swapped canvas
    /// weave, a swapped light and a whole rebuilt pipeline (a colour-space change,
    /// which changes the channel *formats*) all land without anyone having to be
    /// notified — see [`CompositorPipeline::generation`].
    ///
    /// The blend scratch is dropped rather than kept through any of it: it is sized
    /// like the attachments and carries their formats, so "everything that depends on
    /// the target or the pipeline is rebuilt together" holds by construction instead
    /// of by a second condition that could disagree with this one. It costs one
    /// reallocation on the next blended render, and only a document with a
    /// non-`Normal` layer has one at all.
    fn ensure_targets(&mut self, p: &CompositorPipeline, size: Extent2) {
        if size == self.size && self.generation == p.generation {
            return;
        }
        self.size = size;
        self.generation = p.generation;
        self.scratch = None;
        let (c, a, bg) = make_offscreen(OffscreenDesc {
            device: &p.ctx.device,
            size: self.size,
            color_format: p.color_format,
            aux_format: p.aux_format,
            media_bgl: &p.media_bgl,
            media_buf: &p.media_buf,
            surface: &p.surface,
            environment: &p.environment,
        });
        self.comp_color_view = c;
        self.comp_aux_view = a;
        self.media_bg = bg;
    }

    /// Write the view uniform and upload pass A's instance streams for `groups`,
    /// returning the per-tile bind groups that pass draws with.
    ///
    /// Split out of [`Self::render`] so [`Self::composite_channels`] runs the *same*
    /// pass A rather than a second copy of it: what the eyedropper reports and what
    /// the screen shows then cannot drift, which is the whole reason for sampling
    /// through the compositor at all.
    fn prepare_composite(
        &mut self,
        p: &CompositorPipeline,
        view: ViewTransform,
        groups: &[CompositeGroup],
    ) -> Vec<wgpu::BindGroup> {
        let device = &p.ctx.device;

        // View uniform (canvas px -> NDC).
        let (m, translate) = view.canvas_to_ndc();
        p.ctx.queue.write_buffer(
            &p.view_buf,
            0,
            bytemuck::bytes_of(&ViewUniform {
                st: m.to_cols_array(),
                xlate: [translate.x, translate.y, 0.0, 0.0],
                // `zoom` rides in `.w` for the outline pass, which measures its width
                // in screen px from a canvas-space distance (§6.8).
                misc: [
                    TILE_SIZE as f32,
                    INTERIOR_UV_SCALE,
                    INTERIOR_UV_BIAS,
                    view.zoom,
                ],
            }),
        );

        // Split the ordered item list into the two instance streams, remembering
        // for each item which stream slot it draws from. The *order* of the items is
        // what has to survive — a matte must composite over the tiles below it and
        // under the tiles above — so the draw loop in `encode_composite` walks the
        // groups, not these. The streams are flat across every group, so a blend
        // group costs no extra buffer and the instance index keeps running.
        let mut instances: Vec<Instance> = Vec::new();
        let mut tile_bgs: Vec<wgpu::BindGroup> = Vec::new();
        let mut mattes: Vec<MatteInstance> = Vec::new();
        for item in groups.iter().flat_map(CompositeGroup::items) {
            match item {
                CompositeItem::Tile {
                    coord,
                    handle,
                    opacity,
                } => {
                    instances.push(Instance {
                        origin: coord.origin().to_array(),
                        opacity: *opacity,
                    });
                    tile_bgs.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("stark composite tile bg"),
                        layout: &p.tile_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(handle.color_view()),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(handle.aux_view()),
                            },
                        ],
                    }));
                }
                CompositeItem::Matte(m) => mattes.push(MatteInstance {
                    rect: m.rect,
                    channels: m.channels,
                    opacity: m.opacity,
                    _pad: [0.0; 3],
                }),
            }
        }
        if !instances.is_empty() {
            if instances.len() > self.instance_cap {
                self.instances = alloc_instances(device, instances.len());
                self.instance_cap = instances.len();
            }
            p.ctx
                .queue
                .write_buffer(&self.instances, 0, bytemuck::cast_slice(&instances));
        }
        if !mattes.is_empty() {
            if mattes.len() > self.matte_cap {
                self.matte_instances = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("stark matte instances"),
                    size: (std::mem::size_of::<MatteInstance>() * mattes.len()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.matte_cap = mattes.len();
            }
            p.ctx
                .queue
                .write_buffer(&self.matte_instances, 0, bytemuck::cast_slice(&mattes));
        }

        // One uniform slot per merge, all written before the single submit — see
        // `BLEND_SLOT` for why they cannot share one.
        //
        // Collected by the **same recursion the encoder consumes them with**, so
        // slot `n` is the `n`th merge either walk reaches. That is a post-order
        // DFS: a group's members merge before the group itself does, because the
        // group cannot be merged until it has been composited.
        fn collect(members: &[CompositeGroup], out: &mut Vec<BlendUniform>) {
            for m in members {
                if m.is_direct() {
                    continue;
                }
                if let GroupContent::Stack(inner) = &m.content {
                    collect(inner, out);
                }
                out.push(BlendUniform {
                    mode: blend_code(m.blend),
                    k: DRAGO_K,
                    clip: u32::from(m.clip),
                    opacity: m.opacity,
                });
            }
        }
        let mut blends = Vec::new();
        collect(groups, &mut blends);
        if !blends.is_empty() {
            if blends.len() > self.blend_slots {
                self.blend_buf = alloc_blend(device, blends.len());
                self.blend_slots = blends.len();
            }
            for (i, uniform) in blends.iter().enumerate() {
                p.ctx.queue.write_buffer(
                    &self.blend_buf,
                    i as u64 * BLEND_SLOT,
                    bytemuck::bytes_of(uniform),
                );
            }
        }
        tile_bgs
    }

    /// Encode pass A: every group composited into `color` + `aux`, in stack order.
    /// Requires a preceding [`Self::prepare_composite`] for the same `groups`.
    ///
    /// `scratch` is the extra target pair set a non-`Normal` group needs, sized to
    /// match `color`/`aux`. It may be `None` only when every group is `Normal`.
    ///
    /// **The ping-pong, and why the caller's targets always win.** A blend pass reads
    /// the accumulator and writes the merge, so it needs somewhere else to write; the
    /// accumulator therefore alternates between the caller's pair and `scratch.swap`.
    /// Rather than copy at the end, the *start* is chosen by parity: with an odd
    /// number of blend groups the stack begins in `swap`, and every flip lands the
    /// final result exactly where the caller asked for it. That is what lets the
    /// media pass keep one bind group and the eyedropper keep its own targets.
    fn encode_composite(
        &self,
        p: &CompositorPipeline,
        encoder: &mut wgpu::CommandEncoder,
        target: Targets<'_>,
        groups: &[CompositeGroup],
        tile_bgs: &[wgpu::BindGroup],
        scratch: Option<&ScratchTargets>,
    ) {
        let levels: &[ScratchLevel] = scratch.map_or(&[], |s| &s.levels);
        self.encode_stack(
            p,
            encoder,
            target,
            groups,
            tile_bgs,
            &mut Cursors::default(),
            levels,
            0,
        );
    }

    /// Composite one stack's members into `target`, bottom-to-top — the recursion
    /// (§14.7).
    ///
    /// Called on the document's root stack, and again on each group's members one
    /// level deeper. `level` selects this stack's ping-pong pair and the `iso` its
    /// members composite alone into; a member that is itself a group recurses into
    /// that `iso` at `level + 1`, which is why nesting costs a pair-set per level
    /// rather than per group.
    #[allow(clippy::too_many_arguments)]
    fn encode_stack(
        &self,
        p: &CompositorPipeline,
        encoder: &mut wgpu::CommandEncoder,
        target: Targets<'_>,
        members: &[CompositeGroup],
        tile_bgs: &[wgpu::BindGroup],
        cursors: &mut Cursors,
        levels: &[ScratchLevel],
        level: usize,
    ) {
        let merges = members.iter().filter(|m| !m.is_direct()).count();
        let here = levels.get(level);
        let swap = here.map_or(target, ScratchLevel::swap);
        let (mut cur, mut alt) = if merges % 2 == 1 {
            (swap, target)
        } else {
            (target, swap)
        };

        // Whether `cur` holds a real accumulator yet. A direct member's draw clears
        // as it goes; a merge cannot, because the pass *reads* what is under it, so
        // a stack that opens with one needs the clear encoded on its own.
        let mut written = false;

        for member in members {
            if member.is_direct() {
                let GroupContent::Run(items) = &member.content else {
                    unreachable!("a direct group is a Run by construction");
                };
                self.encode_items(p, encoder, cur, items, tile_bgs, cursors, !written);
                written = true;
                continue;
            }
            let scratch = here.expect("a merge without scratch targets");
            if !written {
                clear_targets(encoder, cur);
                written = true;
            }
            // The group, alone on nothing — the isolation its mode and its clip are
            // both defined against.
            let iso = scratch.iso();
            match &member.content {
                GroupContent::Run(items) => {
                    self.encode_items(p, encoder, iso, items, tile_bgs, cursors, true)
                }
                GroupContent::Stack(inner) => {
                    self.encode_stack(p, encoder, iso, inner, tile_bgs, cursors, levels, level + 1)
                }
            }
            self.encode_blend(p, encoder, cur, iso, alt, cursors.blend);
            cursors.blend += 1;
            // `alt` now holds the merged stack and becomes the accumulator; what was
            // `cur` is stale, and the next blend pass overwrites all of it.
            std::mem::swap(&mut cur, &mut alt);
        }
        // An empty stack still has to leave the caller a cleared accumulator.
        if !written {
            clear_targets(encoder, cur);
        }
    }

    /// Draw one run's items into `into`, in stack order, switching pipelines where
    /// a matte sits between runs of tiles. Both pipelines share group 0 (the view
    /// uniform), so only the vertex buffer and pipeline change.
    ///
    /// The cursors are `&mut` because the streams are flat across the whole tree:
    /// a run draws the next stretch of them and hands the cursor on.
    #[allow(clippy::too_many_arguments)]
    fn encode_items(
        &self,
        p: &CompositorPipeline,
        encoder: &mut wgpu::CommandEncoder,
        into: Targets<'_>,
        items: &[CompositeItem],
        tile_bgs: &[wgpu::BindGroup],
        cursors: &mut Cursors,
        clear: bool,
    ) {
        let Cursors {
            tile: tile_i,
            matte: matte_i,
            ..
        } = cursors;
        let load = if clear {
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
        } else {
            wgpu::LoadOp::Load
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("stark composite pass"),
            color_attachments: &[
                Some(load_attachment(into.0, load)),
                Some(load_attachment(into.1, load)),
            ],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_bind_group(0, &p.view_bg, &[]);
        let mut pipeline_is_matte = None;
        for item in items {
            match item {
                CompositeItem::Tile { .. } => {
                    if pipeline_is_matte != Some(false) {
                        pass.set_pipeline(&p.composite_pipeline);
                        pass.set_vertex_buffer(0, self.instances.slice(..));
                        pipeline_is_matte = Some(false);
                    }
                    pass.set_bind_group(1, &tile_bgs[*tile_i as usize], &[]);
                    pass.draw(0..4, *tile_i..*tile_i + 1);
                    *tile_i += 1;
                }
                CompositeItem::Matte(_) => {
                    if pipeline_is_matte != Some(true) {
                        pass.set_pipeline(&p.matte_pipeline);
                        pass.set_vertex_buffer(0, self.matte_instances.slice(..));
                        pipeline_is_matte = Some(true);
                    }
                    pass.draw(0..4, *matte_i..*matte_i + 1);
                    *matte_i += 1;
                }
            }
        }
    }

    /// Merge the isolated layer `src` into the accumulator `back` through blend slot
    /// `slot`, writing the result to `out` (§18.0.4).
    fn encode_blend(
        &self,
        p: &CompositorPipeline,
        encoder: &mut wgpu::CommandEncoder,
        back: Targets<'_>,
        src: Targets<'_>,
        out: Targets<'_>,
        slot: u32,
    ) {
        let bg = p.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stark blend bg"),
            layout: &p.blend_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.blend_buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(std::mem::size_of::<BlendUniform>() as u64),
                    }),
                },
                view_entry(1, back.0),
                view_entry(2, back.1),
                view_entry(3, src.0),
                view_entry(4, src.1),
                view_entry(5, &p.pigment.view),
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&p.pigment.sampler),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("stark blend pass"),
            // The pass covers every texel and reads nothing from `out`, so the load
            // is a don't-care; clearing states that rather than implying the previous
            // contents matter.
            color_attachments: &[
                Some(clear_attachment(out.0, wgpu::Color::TRANSPARENT)),
                Some(clear_attachment(out.1, wgpu::Color::TRANSPARENT)),
            ],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&p.blend_pipeline);
        pass.set_bind_group(0, &bg, &[slot * BLEND_SLOT as u32]);
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
    fn ensure_scratch(
        &mut self,
        p: &CompositorPipeline,
        size: Extent2,
        groups: &[CompositeGroup],
    ) -> bool {
        let levels = scratch_levels(groups);
        if levels == 0 {
            return false;
        }
        // Grown to the deepest nesting the document has *reached*, and never
        // shrunk within a size: a group opened and closed again over and over
        // would otherwise reallocate two viewport-sized pairs each time.
        if self
            .scratch
            .as_ref()
            .is_none_or(|s| s.size != size || s.levels.len() < levels)
        {
            self.scratch = Some(ScratchTargets::new(
                &p.ctx.device,
                size,
                levels,
                p.color_format,
                p.aux_format,
            ));
        }
        true
    }

    /// Composite `items` into caller-supplied targets and **stop there** — pass A
    /// alone, with no media pass over it.
    ///
    /// This is the eyedropper's sampling path (§18.0.2). What lands in
    /// `color` is the paint's own channels in the document's working space, which is
    /// what a picker has to read: the lit result has been through image-based
    /// lighting, a tonemap and an sRGB encode, so picking *that* would hand back a
    /// colour the palette never mixed — and in a Mixbox document (§6.7) a
    /// pigment mixture that cannot be picked back up, which is the point of mixing
    /// in pigment space at all.
    ///
    /// `color` and `aux` must carry the formats
    /// [`CompositorPipeline::channel_formats`] reports, and be `view.viewport` in
    /// size. They are the caller's, not this compositor's: a sample is taken through
    /// the compositor that belongs to the screen, so it must leave the screen's own
    /// attachments — a few hundred texels wide against the window's millions —
    /// exactly where they were. That is why this does not go through
    /// [`Self::ensure_targets`], and why the blend scratch below is its own too.
    pub fn composite_channels(
        &mut self,
        p: &CompositorPipeline,
        color: &wgpu::TextureView,
        aux: &wgpu::TextureView,
        view: ViewTransform,
        groups: &[CompositeGroup],
    ) {
        let tile_bgs = self.prepare_composite(p, view, groups);
        // Its own scratch, thrown away with the call. A pick viewport is `2r+1`
        // square, so this is a few kilobytes; sharing the render path's cache would
        // trade that for reallocating the *window* twice a frame (see
        // [`Self::ensure_scratch`]). Blend modes have to be honoured here or an
        // eyedropper would report a colour the screen never showed.
        let levels = scratch_levels(groups);
        let scratch = (levels > 0).then(|| {
            ScratchTargets::new(
                &p.ctx.device,
                view.viewport,
                levels,
                p.color_format,
                p.aux_format,
            )
        });
        let mut encoder = p
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("stark pick encoder"),
            });
        self.encode_composite(
            p,
            &mut encoder,
            (color, aux),
            groups,
            &tile_bgs,
            scratch.as_ref(),
        );
        p.ctx.queue.submit([encoder.finish()]);
    }

    /// Composite `scene`'s layers, light the result into `target` under `view`, and
    /// outline each of its selections over it (§6.8 — a universal selection
    /// draws nothing, so an unmasked document costs one skipped iteration).
    pub fn render(
        &mut self,
        p: &CompositorPipeline,
        target: &wgpu::TextureView,
        view: ViewTransform,
        scene: CompositeScene<'_>,
    ) {
        let CompositeScene {
            background: bg_channels,
            groups,
            outlines,
            transparent,
        } = scene;
        // This compositor's attachments, brought in line with what is about to be
        // drawn. Nobody else's: a render into something other than this target — an
        // export, the navigator's miniature — goes through a `Compositor` of its own,
        // so the surface's attachments (and the frame already presented from them)
        // are never resized out from under it and rebuilt on the next frame.
        self.ensure_targets(p, view.viewport);
        let tile_bgs = self.prepare_composite(p, view, groups);
        let want_scratch = self.ensure_scratch(p, self.size, groups);
        // Bound after everything that needs `&mut self`.
        let device = &p.ctx.device;
        let scratch = if want_scratch {
            self.scratch.as_ref()
        } else {
            None
        };
        let (comp_color_view, comp_aux_view, media_bg) =
            (&self.comp_color_view, &self.comp_aux_view, &self.media_bg);

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
        // environment — an unrelieved patch of paint comes back out its own colour.
        let exposure = p.environment.exposure / p.environment.flat_irradiance;

        // Media uniform.
        p.ctx.queue.write_buffer(
            &p.media_buf,
            0,
            bytemuck::bytes_of(&MediaUniform {
                light: [0.0, 0.0, 0.0, p.media.height_strength],
                bg: bg_channels,
                shade: [exposure, diffuse_lod, p.media.specular, 0.0],
                surf_a: [
                    canvas_origin.x,
                    canvas_origin.y,
                    inv_zoom,
                    1.0 / SURFACE_TILE_PX,
                ],
                surf_m: inv_linear.to_cols_array(),
                surf_b: [
                    p.media.surface_strength,
                    // Transparent export: the media pass skips the substrate and
                    // carries the paint's visible alpha out (§15.6).
                    if transparent { 1.0 } else { 0.0 },
                    0.0,
                    0.0,
                ],
            }),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stark composite encoder"),
        });

        // Pass A: composite tiles into offscreen color + aux. The parity trick in
        // `encode_composite` guarantees the result lands in these two views however
        // many blend passes ran, so the media bind group never has to be rebuilt.
        self.encode_composite(
            p,
            &mut encoder,
            (comp_color_view, comp_aux_view),
            groups,
            &tile_bgs,
            scratch,
        );

        // Pass B: media/lighting → target.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stark media pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        // The media pass covers every texel, so the clear only
                        // matters for what alpha an untouched texel would keep —
                        // transparent, on an export that wants a cut-out.
                        load: wgpu::LoadOp::Clear(if transparent {
                            wgpu::Color::TRANSPARENT
                        } else {
                            wgpu::Color::BLACK
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&p.media_pipeline);
            pass.set_bind_group(0, media_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // Pass C: the selection outlines, over the lit image — the local actor's and
        // every present peer's, one instanced quad per mask tile of each
        // (§17.3). Flattened into one instance stream so N collaborators
        // still cost one pass.
        let mut overlay_instances: Vec<OverlayInstance> = Vec::new();
        let mut mask_tiles: Vec<wgpu::BindGroup> = Vec::new();
        for outline in outlines {
            if outline.selection.is_universal() {
                continue;
            }
            let tint = match outline.tint {
                Some([r, g, b]) => [r, g, b, PEER_OUTLINE_ALPHA],
                None => [0.0; 4],
            };
            for (coord, handle) in outline.selection.tiles() {
                overlay_instances.push(OverlayInstance {
                    origin: coord.origin().to_array(),
                    tint,
                });
                mask_tiles.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("stark overlay tile bg"),
                    layout: &p.overlay_tile_bgl,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(handle.view()),
                    }],
                }));
            }
        }
        if !mask_tiles.is_empty() {
            if overlay_instances.len() > self.overlay_cap {
                self.overlay_instances = alloc_overlay(device, overlay_instances.len());
                self.overlay_cap = overlay_instances.len();
            }
            p.ctx.queue.write_buffer(
                &self.overlay_instances,
                0,
                bytemuck::cast_slice(&overlay_instances),
            );

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stark selection overlay pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&p.overlay_pipeline);
            pass.set_bind_group(0, &p.overlay_view_bg, &[]);
            pass.set_vertex_buffer(0, self.overlay_instances.slice(..));
            for (i, bg) in mask_tiles.iter().enumerate() {
                let idx = i as u32;
                pass.set_bind_group(1, bg, &[]);
                pass.draw(0..4, idx..idx + 1);
            }
        }

        p.ctx.queue.submit([encoder.finish()]);
    }
}

fn tex_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn load_tex_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            // Sampled only via textureLoad, so no filtering required.
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn clear_attachment(
    view: &wgpu::TextureView,
    color: wgpu::Color,
) -> wgpu::RenderPassColorAttachment<'_> {
    load_attachment(view, wgpu::LoadOp::Clear(color))
}

fn load_attachment(
    view: &wgpu::TextureView,
    load: wgpu::LoadOp<wgpu::Color>,
) -> wgpu::RenderPassColorAttachment<'_> {
    wgpu::RenderPassColorAttachment {
        view,
        resolve_target: None,
        depth_slice: None,
        ops: wgpu::Operations {
            load,
            store: wgpu::StoreOp::Store,
        },
    }
}

/// A render pass that only clears. Encoded when the bottom of the stack is a blend
/// group: that pass *reads* the accumulator, so unlike a run of tiles it cannot
/// fold the clear into its own load op.
fn clear_targets(encoder: &mut wgpu::CommandEncoder, into: Targets<'_>) {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("stark composite clear"),
        color_attachments: &[
            Some(clear_attachment(into.0, wgpu::Color::TRANSPARENT)),
            Some(clear_attachment(into.1, wgpu::Color::TRANSPARENT)),
        ],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
}

fn view_entry(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

/// A viewport-sized offscreen render target, as pass A and the blend pass use.
fn offscreen_view(
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

/// The blend uniform buffer: `count` dynamic-offset slots (see [`BLEND_SLOT`]).
fn alloc_blend(device: &wgpu::Device, count: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("stark blend uniform"),
        size: BLEND_SLOT * count.max(1) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// How strongly another actor's selection outline reads against the artwork. Well
/// below the local one, which is a full-strength dashed line: yours is a thing you
/// act through, theirs is a thing you need only be aware of.
const PEER_OUTLINE_ALPHA: f32 = 0.55;

fn alloc_overlay(device: &wgpu::Device, count: usize) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("stark overlay instances"),
        contents: bytemuck::cast_slice(&vec![
            OverlayInstance {
                origin: [0.0; 2],
                tint: [0.0; 4],
            };
            count.max(1)
        ]),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    })
}

fn alloc_instances(device: &wgpu::Device, count: usize) -> wgpu::Buffer {
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

/// The inputs to [`make_offscreen`]: one field from the [`Compositor`] being built or
/// rebuilt (`size`) and the rest read off the shared [`CompositorPipeline`]. Grouped
/// because the two callers — [`Compositor::new`] and [`Compositor::ensure_targets`] —
/// would otherwise each spell out the same eight fields.
struct OffscreenDesc<'a> {
    device: &'a wgpu::Device,
    size: Extent2,
    color_format: wgpu::TextureFormat,
    aux_format: wgpu::TextureFormat,
    media_bgl: &'a wgpu::BindGroupLayout,
    media_buf: &'a wgpu::Buffer,
    surface: &'a Surface,
    environment: &'a Environment,
}

/// (Re)create the offscreen composite targets and the media bind group.
fn make_offscreen(d: OffscreenDesc<'_>) -> (wgpu::TextureView, wgpu::TextureView, wgpu::BindGroup) {
    let OffscreenDesc {
        device,
        size,
        color_format,
        aux_format,
        media_bgl,
        media_buf,
        surface,
        environment,
    } = d;
    let comp_color_view = offscreen_view(device, size, color_format, "stark comp color");
    let comp_aux_view = offscreen_view(device, size, aux_format, "stark comp aux");

    let media_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("stark media bg"),
        layout: media_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: media_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&comp_color_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&comp_aux_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&surface.view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(&surface.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(&environment.view),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::Sampler(&environment.sampler),
            },
        ],
    });
    (comp_color_view, comp_aux_view, media_bg)
}
