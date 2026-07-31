//! The engine: owns the GPU, session, and timeline; turns commands into state
//! and renders the canvas (DESIGN.md §7).
//!
//! For the MVP this exposes a synchronous [`Engine::process`]. The asynchronous
//! actor loop and reactive `ObservableState` channel (DESIGN.md §7) wrap this
//! same core in a later step.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::assets::{AssetId, AssetStore};
use crate::colorspace::{ColorSpace, ColorSpaceId};
use crate::command::{DocCommand, GestureCommand, InputCommand, PeerCommand, ViewCommand};
use crate::document::{
    Action, ActionId, ActionKind, ActorId, ApplyCtx, BlendMode, BrushParams, BrushShape,
    CanvasBounds, DocState, Layer, LayerContent, LayerId, LinearTimeline, ReplicatedTimeline,
    ShapeAction, StrokeRecord, Timeline, TimelineStats, Tool, effective_actions,
};
use crate::geom::{Extent2, TileCoord, ViewTransform};
use crate::gpu::tile::MASK_FORMAT;
use crate::gpu::{
    CompositeGroup, CompositeItem, CompositeScene, Compositor, CompositorPipeline, Environment,
    EnvironmentId, FillRenderer, GpuContext, GroupContent, MatteDraw, Offscreen, Registry,
    SelectionOutline, SelectionRenderer, StrokeRenderer, StrokeSpans, Surface, SurfaceId, TilePool,
    TransformRenderer,
};
use crate::image::RgbaImage;
use crate::io::DocumentFile;
use crate::peer::{GestureView, Identity, LiveGesture, Peer, PeerFrame, Peers};
use crate::session::ShapeResult;
use crate::{EngineError, Result};

/// The starting layer present in every new document.
const ROOT_LAYER: LayerId = LayerId(0);

/// What sits under the paint when rendering (FRAME_DESIGN.md §6).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Background {
    /// The document's substrate colour, lit and textured by the canvas weave —
    /// what the screen shows.
    #[default]
    Substrate,
    /// Nothing: the paint's own visible alpha becomes the image's alpha, for a
    /// cut-out PNG. A real branch in the media pass rather than an alpha tweak —
    /// the substrate composite is skipped entirely, so bare canvas is genuinely
    /// absent rather than white-and-invisible.
    Transparent,
}

/// Whether on-canvas affordances (the selection outline) are drawn. Screen: yes.
/// Export: never — chrome is a thing to draw *with*, not a thing to ship
/// (FRAME_DESIGN.md §6).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Chrome {
    Shown,
    Hidden,
}

/// Which [`Compositor`] a render's offscreen attachments come from.
///
/// Compositing runs through pass-A attachments the size of the target, so *whose*
/// they are decides who pays for a resize. The surface's are kept from frame to
/// frame; anything rendered beside them is a different size and brings its own, so
/// the screen's are never resized out from under it — and never rebuilt on the next
/// frame to recover. That mattered as soon as something rendered off-screen
/// *repeatedly*: the navigator's miniature is one render per edit, and sharing the
/// surface's attachments made it two rebuilds of window-sized textures and a full
/// recomposite per edit.
enum Attachments<'a> {
    /// The surface's own, cached across frames ([`Engine::compositor`]).
    Surface,
    /// The caller's, so whether they outlive the call is decided by whoever knows
    /// whether the render repeats — see [`Offscreen`].
    Offscreen(&'a mut Offscreen),
}

/// Which document a render draws: the one being *shown*, or the committed one
/// alone.
///
/// The distinction only exists because a render can be asked for while a gesture
/// is in flight. The screen wants [`Rendered::Live`] — that is what makes a stroke
/// visible as it is drawn. A render that stands in for the *state of the work*
/// wants [`Rendered::Committed`]: it is refreshed when the document changes, so
/// following the in-flight stroke would mean re-rendering at pointer rate to show
/// something that is already on screen at full size.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Rendered {
    /// The committed document with every in-flight gesture — this client's and
    /// each peer's — and any unlogged drag preview drawn over it
    /// (PEER_DESIGN.md §6). What the screen shows.
    #[default]
    Live,
    /// The committed document alone: no in-flight stroke, no drag preview.
    Committed,
}

/// How large an exported image is, relative to the frame's canvas-space size
/// (FRAME_DESIGN.md §6). Resolution is a property of the *output*, not of the
/// artwork, which is why the frame stores only a canvas-space rect.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ExportScale {
    /// A multiple of the frame's canvas size — 1× is one canvas px per image px.
    Factor(f32),
    /// An exact width in image px; the height follows the frame's aspect.
    Width(u32),
}

/// What an export will produce, before producing it — so a dialog can show the
/// pixel size the user is about to get.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ExportPlan {
    /// The canvas-space rect being exported.
    pub min: crate::geom::Vec2,
    pub max: crate::geom::Vec2,
    /// Output size in image px.
    pub size: Extent2,
    /// Image px per canvas px.
    pub zoom: f32,
}

impl ExportPlan {
    /// The view this plan renders through: centred on the rect, at `zoom` = its scale,
    /// with the plan's pixel size as the viewport.
    ///
    /// The plan *is* the view, in other words, which is why both things that render a
    /// planned rect — writing a file ([`Engine::export`]) and drawing the navigator's
    /// miniature ([`Engine::render_into`]) — derive it here rather than each spelling
    /// out the same three lines and drifting.
    pub fn view(&self) -> ViewTransform {
        ViewTransform {
            center: (self.min + self.max) * 0.5,
            zoom: self.zoom,
            // Upright and unmirrored, whatever angle the artist has the canvas at:
            // turning the easel is a way of *looking* at the piece, and a file — or
            // the navigator's overview, which frames itself the same way — shows the
            // piece rather than the easel (MISSING_FEATURES §1.2).
            rotation: 0.0,
            flip_h: false,
            viewport: self.size,
        }
    }
}

/// Largest exported edge, in px. Guards against a stray zero-ish frame or a huge
/// scale asking for a texture the device will refuse — reported as an error rather
/// than surfacing as a wgpu validation panic.
const MAX_EXPORT_DIM: u32 = 8192;

/// Which layers an eyedropper sample is taken from (MISSING_FEATURES §0.2).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum PickSource {
    /// Every visible layer, composited — the colour the canvas shows.
    #[default]
    Composite,
    /// One layer alone: the colour that layer would have if it were the only one in
    /// the document. What "sample the current layer" has to mean, since a glaze on
    /// top of somebody else's underpainting is not the same paint as the two mixed.
    Layer(LayerId),
}

/// How an eyedropper sample is taken (MISSING_FEATURES §0.2).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub struct PickOptions {
    pub source: PickSource,
    /// Half-width of the averaged square, in **canvas** px: 0 samples the single
    /// canvas pixel under the point, 1 averages 3×3, 2 averages 5×5 — the prior
    /// art's point / N×N sampler.
    ///
    /// Canvas px rather than screen px, so what a sample covers is a property of the
    /// painting and not of how far the artist happens to be zoomed in.
    pub radius: u32,
}

/// Largest eyedropper radius, in canvas px — a 65×65 average. Not a taste limit but
/// a bound on what one sample may cost: the sampled square is rendered and read
/// back, so an unbounded radius is an unbounded render.
const MAX_PICK_RADIUS: u32 = 32;

/// Below this summed opacity a sampled patch holds no paint worth calling a colour,
/// and dividing by it would amplify float noise into an arbitrary hue.
const PICK_MIN_OPACITY: f32 = 1e-3;

/// Longest layer name that will be recorded, in `char`s. Not a taste limit but a
/// bound on the log: a name is replicated to every peer and saved with the
/// document, and nothing about a text field stops a paste from being a megabyte.
/// Truncated by `char` rather than by byte so the cut can never land inside one.
const MAX_LAYER_NAME: usize = 64;

/// The name to record for a layer, given what a frontend collected: surrounding
/// whitespace trimmed, length capped, and anything that comes out empty treated as
/// *no name* rather than as a name that happens to be blank.
///
/// One funnel for every source — the panel's field, a script, a peer's command —
/// so "a layer's name is either absent or something you can read" is a property of
/// the model rather than a habit of the UI. The logged action carries the result,
/// so replay reproduces it without re-running these rules.
fn normalize_layer_name(name: Option<String>) -> Option<String> {
    let name = name?;
    let trimmed = name.trim();
    let capped: String = trimmed.chars().take(MAX_LAYER_NAME).collect();
    (!capped.is_empty()).then_some(capped)
}

/// The mean **unpremultiplied** channels of a sampled patch, or `None` where there
/// is no paint in it.
///
/// The composite is premultiplied by opacity (DESIGN.md §6.1), so summing and
/// dividing by the summed opacity *is* the opacity-weighted mean: a texel carrying
/// more paint counts for more and a bare one counts for nothing. That is what lets a
/// radius wider than the stroke still report the stroke's colour rather than a wash
/// of it fading into empty canvas.
fn mean_channels(texels: &[f32]) -> Option<[f32; 4]> {
    let mut sum = [0.0f32; 4];
    for t in texels.as_chunks::<4>().0 {
        for (s, v) in sum.iter_mut().zip(t) {
            *s += v;
        }
    }
    if !sum[3].is_finite() || sum[3] <= PICK_MIN_OPACITY {
        return None;
    }
    Some([sum[0] / sum[3], sum[1] / sum[3], sum[2] / sum[3], 1.0])
}

/// A layer's presentation properties, for the UI's layer panel (DESIGN.md §11).
///
/// `Clone` but not `Copy` since it carries the name — an `Arc<str>` bump, so
/// cloning one is still a handful of instructions and `observe()` stays cheap.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerInfo {
    pub id: LayerId,
    pub blend: crate::document::BlendMode,
    /// Whether this layer clips to the paint beneath it (GROUP_DESIGN.md §4).
    pub clip: bool,
    pub opacity: f32,
    pub visible: bool,
    /// The layer carrying this one — i.e. the group it is in — or `None` for one
    /// in the document's own stack (GROUP_DESIGN.md §2).
    pub carrier: Option<LayerId>,
    /// How deeply nested it is: 0 in the document's own stack, one more per level
    /// of carrying. What a panel indents by.
    pub depth: usize,
    /// Whether this layer carries others, i.e. whether it is a **group**. A panel
    /// gives one of these a disclosure triangle; nothing else distinguishes it.
    pub is_group: bool,
    /// Whether anything composites beneath it, so its blend mode and its clip do
    /// anything at all (GROUP_DESIGN.md §4.3). False on exactly one row — the
    /// bottom of the document — where a mode is the identity and a clip would
    /// erase the layer, and where a panel therefore shows both controls inert.
    pub has_backdrop: bool,
    /// What the author called this layer, or `None` for one that has never been
    /// named — in which case it is for the frontend to describe it, since only the
    /// frontend knows how it presents a stack (see [`Layer::name`]).
    ///
    /// [`Layer::name`]: Layer::name
    pub name: Option<std::sync::Arc<str>>,
    /// Set when this layer is a **matte** (FRAME_DESIGN.md §2) — a frame rather
    /// than paint. `None` for an ordinary paint layer.
    ///
    /// Projected so the frontend can label it, draw its handles, and show that the
    /// brush has nowhere to go while it is selected — all without reaching past
    /// `observe()` into `DocState`.
    pub matte: Option<MatteInfo>,
}

impl LayerInfo {
    /// Whether a stroke aimed at this layer would draw anything. A matte has no
    /// tile map, so selecting one is legal but painting on it does nothing
    /// (FRAME_DESIGN.md §7).
    pub fn is_paintable(&self) -> bool {
        self.matte.is_none()
    }
}

/// A matte layer's geometry and fill, for the frame chrome (FRAME_DESIGN.md §7).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MatteInfo {
    /// The rect the region is defined against, in canvas px. For a frame this is
    /// the *hole* — the piece — which is what the handles resize and what export
    /// frames against (FRAME_DESIGN.md §6).
    pub min: crate::geom::Vec2,
    pub max: crate::geom::Vec2,
    /// Fill colour, straight sRGB.
    pub color: [f32; 3],
}

impl MatteInfo {
    pub fn width(&self) -> f32 {
        self.max.x - self.min.x
    }
    pub fn height(&self) -> f32 {
        self.max.y - self.min.y
    }
}

/// A cheap, UI-facing projection of engine state (DESIGN.md §7). Published to
/// the frontend so it can render chrome reactively without touching pixels.
#[derive(Clone, Debug)]
pub struct ObservableState {
    pub can_undo: bool,
    pub can_redo: bool,
    pub is_stroking: bool,
    pub tool: Tool,
    pub brush: BrushParams,
    pub view: ViewTransform,
    pub bounds: CanvasBounds,
    /// A counter that changes whenever the **committed** document does — a commit,
    /// an undo, a merged remote action, a load.
    ///
    /// For a frontend that keeps a *rendered* stand-in for the document and has to
    /// know when it went stale: the navigator's miniature (a small `export`) is one
    /// GPU render plus a readback, so it cannot be redone per observation. Nothing
    /// else in this projection answers the question — a second stroke over the same
    /// tiles leaves the bounds, the layer list and the undo flags all exactly as
    /// they were.
    ///
    /// Deliberately *not* bumped by an in-flight gesture or an unlogged drag
    /// preview, both of which change what the canvas shows at pointer rate. Those
    /// are already on screen at full size; a watcher keyed on them would re-render
    /// the miniature per pointer sample to say something the canvas is saying
    /// better. Compare `is_stroking`, which is exactly the in-flight question.
    pub doc_revision: u64,
    pub active_layer: LayerId,
    /// Layers bottom-to-top.
    pub layers: Vec<LayerInfo>,
    /// Whether a selection is masking the canvas (DESIGN.md §6.8) — drives the
    /// "Deselect"/"Invert" affordances and the selection indicator.
    pub has_selection: bool,
    /// A conservative canvas-space bounding box of this client's selected
    /// coverage, or `None` when the selection is unbounded or unknown
    /// ([`Selection::hull`](crate::document::Selection::hull)). What the
    /// transform chrome hangs its handles on; committed-only, like
    /// `has_selection`.
    pub selection_hull: Option<(crate::geom::Vec2, crate::geom::Vec2)>,
    /// What the next shape gesture will do with the region it encloses — combine
    /// it into the selection one of four ways, or fill it (MISSING_FEATURES §0.4).
    pub shape_action: ShapeAction,
    /// Edge softness (canvas px) the next shape gesture will apply.
    pub selection_feather: f32,
    /// Whether collaborators' selection outlines are drawn (PEER_DESIGN.md §3).
    pub show_peer_selections: bool,

    // --- view settings (per-client, never historized) ---------------------
    //
    // Projected here for the same reason as `tool` and `brush`: a frontend that
    // has to read these back off the engine ends up keeping its own copy, and a
    // copy seeded from `Default` rather than from the engine goes stale the
    // moment anything else changes them (DESIGN.md §4).
    /// Media/lighting parameters of the painterly pass (DESIGN.md §6.3).
    pub media: crate::gpu::MediaParams,
    /// The HDR lighting environment in use (DESIGN.md §6.3).
    pub environment: EnvironmentId,

    // --- document properties fixed at creation ----------------------------
    /// The document's colour space. Immutable for the document's life — changing
    /// it means starting a new document ([`Engine::new_with_color_space`]).
    pub color_space: ColorSpaceId,
    /// The physical canvas surface (DESIGN.md §6.4).
    pub surface: SurfaceId,
    /// The canvas substrate colour, straight sRGB (FRAME_DESIGN.md §5). Document
    /// state, not a view setting — projected here so the frontend shows what the
    /// document says rather than a copy of its own that goes stale.
    pub background: [f32; 3],
}

/// Colour the live tail is drawn in under the `debug-unfrozen` feature.
/// Full-opacity magenta: it has to read against paint of any hue, and against the
/// stroke's own colour in particular.
#[cfg(feature = "debug-unfrozen")]
const DEBUG_UNFROZEN_COLOR: [f32; 4] = [1.0, 0.0, 1.0, 1.0];

/// The part of the in-flight stroke that has stopped changing, already composited
/// onto the committed document.
///
/// A live stroke is re-rendered on every pointer move, and rendering it costs
/// (segments × tiles it covers) — both of which grow with its length, so a long
/// stroke gets quadratically expensive to keep drawing. But the fitter freezes
/// control points behind the pointer and never revises them
/// ([`PathFitter::frozen_spans`](crate::path::PathFitter::frozen_spans)), so the
/// spans they determine are final: render them once, keep the result, and each move
/// only has to draw the short live tail over it. Work per move then follows the tail
/// rather than the stroke.
struct FrozenHead {
    /// How many leading spans `state` already has drawn on it.
    spans: usize,
    /// Arc length at the end of those spans — where the tail's `dist` resumes.
    /// Not recoverable from `spans` alone, and the `drain` falloff and colour
    /// dynamics both read it (see `gpu::stroke::StrokeSpans`).
    dist: f32,
    /// The brush state the tail must resume from, for a stroke that runs the
    /// sequential stamp loop (`lift`/`deposit`/`charge`). `None` for the swept path,
    /// which carries nothing between segments. See
    /// [`ToolState`](crate::gpu::stroke::ToolState).
    tool: Option<crate::gpu::stroke::ToolState>,
    state: DocState,
    /// Which gesture this is the head of — its author's ordinal. A head is only ever
    /// legitimately reused across consecutive moves of the *same* gesture, and the
    /// span count alone cannot tell a new stroke from a continued one when the new
    /// one has already grown past where the old was frozen.
    gesture: u64,
    /// The base this head was composited onto ([`Engine::doc_epoch`]). Anything that
    /// replaces the base — a commit, an undo, a remote merge, a load — bumps the
    /// epoch, and a head from an earlier one is discarded rather than drawn over a
    /// canvas that no longer exists.
    epoch: u64,
    /// Every tile the head has rewritten so far, so the fold knows what to overlay
    /// (PEER_DESIGN.md §6). Accumulated because a head grows across many advances.
    dirty: BTreeSet<TileCoord>,
}

/// What one presence-pump tick moved ([`Engine::take_presence`]).
///
/// The two halves reach different places: `frame` is owed to the wire, `repaint`
/// to the compositor. They travel together because the pump is the engine's only
/// clock, so the expiry that produces `repaint` can only run on its cadence.
#[derive(Debug)]
pub struct PresenceTick {
    /// This client's presence, if anything a peer would care about has changed
    /// since the last drain — `None` when solo or when nothing moved.
    pub frame: Option<PeerFrame>,
    /// Expiry took something off the canvas — a stalled gesture or a departed
    /// peer — so a repaint is owed. Without it the last composite, stale stroke
    /// and all, stays on screen until something else forces a paint.
    pub repaint: bool,
}

pub struct Engine {
    gpu: GpuContext,
    target_format: wgpu::TextureFormat,
    color_space: Arc<dyn ColorSpace>,
    /// The GPU subsystems an action needs in order to apply itself — the tile
    /// pool, the stroke renderer, the asset store and the selection rasterizer —
    /// held as the `history::Action::Context` (DESIGN.md §5).
    ///
    /// Stored rather than built per call. `history`'s `Context` is an owned
    /// associated type, so there is nothing to hand it a borrow of; this used to be
    /// rebuilt by cloning all four on *every* commit, undo, redo and remote merge —
    /// tens of `Arc` bumps plus a `HashMap` allocation each time, for a value that
    /// only changes when the colour space is rebuilt.
    ///
    /// They live only here: the engine reaches them through `self.apply` too, so
    /// there is one copy rather than the engine's plus the context's.
    /// `selection` is colour-space independent (a mask is one coverage channel
    /// whatever the paint is), so unlike the pool and the stroke renderer it
    /// survives a rebuild.
    apply: ApplyCtx,
    /// Compositing state for the **surface**: the attachments a screen frame is
    /// built through, kept from frame to frame (`gpu::composite`). Anything drawn
    /// beside the screen — an export, the navigator's miniature — gets a
    /// [`Compositor`] of its own for the call, so it never resizes these.
    compositor: Compositor,
    /// The pipelines, layouts and view settings every `Compositor` shares. Held
    /// beside the one above rather than inside it because a second one borrows it:
    /// the expensive half of compositing is built once, and the view settings the
    /// media pass reads have one owner, so two consumers cannot disagree about the
    /// canvas weave or the lighting.
    compositor_pipeline: CompositorPipeline,
    /// The physical canvas surface (relief) and the bytes registered for
    /// it. Colour-space-independent, so it survives colour-space rebuilds. Which
    /// surface is in use is a *cache* of `document().surface`, kept in step by
    /// [`Engine::apply_document_surface`] — the document is the source of truth
    /// (DESIGN.md §6.4).
    surface: Registry<SurfaceId>,
    /// The surface the action log starts from, written to `CanvasMeta` and used to
    /// seed the document. Plays the same role as `CanvasMeta::color_space`: it
    /// describes the empty document that the log is replayed onto, and is not
    /// itself a logged change.
    initial_surface: SurfaceId,
    /// The HDR lighting environment (image-based lighting) and its registered
    /// bytes. A *view* setting — not historized, colour-space-independent — so it
    /// survives rebuilds and switching it never touches the document (§6.3).
    environment: Registry<EnvironmentId>,
    timeline: Box<dyn Timeline>,
    session: crate::session::Session,
    /// Everyone else in the session (PEER_DESIGN.md §4). Empty when solo.
    peers: Peers,
    /// The presence clock: the newest instant a caller has handed in, in seconds on
    /// a monotonic scale.
    ///
    /// `stark-core` deliberately owns no clock *source* — that is what lets it run on
    /// wasm and native alike — but it does own the *value*. Sampling one per tick and
    /// reading it everywhere else means expiry, publishing and the timestamping of
    /// arriving frames all see the same instant, instead of each call site being free
    /// to hand in its own. `max` because a clock that steps backwards must not
    /// un-expire a peer, whatever the frontend hands in.
    now: f64,
    /// The unlogged document edit in flight: a whole document that stands in for
    /// the committed one, because what these edits change — a matte's rect
    /// (FRAME_DESIGN.md §7), the substrate colour (§5) — is document state rather
    /// than a tile edit, so there is nothing to draw *over* the document the way a
    /// stroke preview does. One slot, not one per kind: only one such drag can be
    /// in flight at a time (they all belong to a single held pointer), and the
    /// stand-in is built from the committed state each time, so a second kind
    /// starting mid-drag supersedes the first rather than compounding with it.
    /// `None` when nothing is being dragged.
    doc_preview: Option<DocState>,
    /// The **presented** document: the committed state (or `doc_preview`) with
    /// every in-flight gesture — this client's and every peer's — drawn over it
    /// (PEER_DESIGN.md §6). `None` when nobody is mid-gesture.
    live: Option<DocState>,
    /// The settled head of each in-flight stroke, keyed by its author (see
    /// [`FrozenHead`]). Every head is rooted at the *committed* document rather than
    /// at the previous peer's preview: chaining would be marginally more faithful
    /// for two strokes overlapping in the same instant, and would invalidate peer
    /// *k*'s cache on every move by peers before it — collapsing the incremental
    /// repaint exactly when two people are painting at once.
    heads: BTreeMap<ActorId, FrozenHead>,
    /// Bumped whenever the document the previews are composited onto changes. A
    /// [`FrozenHead`] stamped with an older epoch is stale and discarded — which
    /// rules out the whole class of "drawn over a canvas that has since moved"
    /// rather than enumerating the ways it arises.
    doc_epoch: u64,
    /// Bumped whenever the **committed** document changes — a commit, an undo, a
    /// merged remote action, a load. Projected as
    /// [`ObservableState::doc_revision`], where it is what a frontend showing a
    /// rendered stand-in for the document (the navigator's miniature) watches to
    /// know when that render is out of date.
    ///
    /// Strictly narrower than `doc_epoch`, which an unlogged drag preview also
    /// bumps: a preview moves at pointer rate and is *not* a change to the
    /// document, so a watcher keyed on the epoch would re-render for every sample
    /// of a drag. The two advance together through [`Engine::committed_changed`].
    doc_revision: u64,
    /// Raw pointer reports of the in-flight stroke, dumped on release under the
    /// `debug-unfrozen` feature so a misfit stroke can be replayed as a test.
    debug_samples: Vec<crate::command::InputSample>,
    actor: ActorId,
    clock: u64,
    next_layer: u64,
    /// Locally-committed actions awaiting broadcast to peers (DESIGN.md §12.4).
    /// Only populated in a shared session (`outbox_enabled`), and drained by the
    /// transport via [`Engine::take_outbox`]; solo mode never accumulates.
    outbox: Vec<Action>,
    outbox_enabled: bool,
}

impl Engine {
    /// Build an engine that presents to `target_format` (a surface format, or a
    /// test target), in the default Oklab color space. Takes wgpu handles per
    /// GOALS §Inputs.
    pub fn new(gpu: GpuContext, target_format: wgpu::TextureFormat, viewport: Extent2) -> Self {
        Self::new_with_color_space(gpu, target_format, viewport, ColorSpaceId::Oklab)
    }

    /// Build an engine in a chosen color space (DESIGN.md §6.7).
    pub fn new_with_color_space(
        gpu: GpuContext,
        target_format: wgpu::TextureFormat,
        viewport: Extent2,
        color_space: ColorSpaceId,
    ) -> Self {
        let color_space = color_space.make();
        // The registry starts on the builtin flat surface — it is all that can be
        // built before any bytes exist. A fresh document is on `DEFAULT_SURFACE`
        // (linen), so the two are reconciled at the end of this function; until the
        // frontend registers the height map, linen renders as flat (DESIGN.md §6.4).
        let surface = Registry::<SurfaceId>::new(&gpu, SurfaceId::default());
        // Lighting starts on the procedural neutral environment; image HDRs are
        // registered later by the frontend (DESIGN.md §6.3).
        let _environment_id = EnvironmentId::default();
        let environment = Registry::<EnvironmentId>::new(&gpu, EnvironmentId::default());
        let selection = SelectionRenderer::new(&gpu);
        let (pool, stroke, compositor_pipeline, compositor, transform, fill) =
            build_gpu(GpuBuild {
                gpu: &gpu,
                target_format,
                viewport,
                cs: &color_space,
                surface: surface.current(),
                environment: environment.current(),
                selection: &selection,
            });
        let assets = AssetStore::new(gpu.clone());

        let initial = DocState::with_layer(ROOT_LAYER);
        let initial_surface = initial.surface;
        let timeline: Box<dyn Timeline> = Box::new(LinearTimeline::new(initial));
        let session = crate::session::Session::new(ViewTransform::identity(viewport), ROOT_LAYER);

        let mut engine = Self {
            gpu,
            target_format,
            color_space,
            apply: ApplyCtx {
                pool,
                stroke,
                assets,
                selection,
                transform,
                fill,
            },
            compositor,
            compositor_pipeline,
            initial_surface,
            surface,
            environment,
            timeline,
            session,
            peers: Peers::new(),
            now: 0.0,
            doc_preview: None,
            live: None,
            heads: BTreeMap::new(),
            doc_epoch: 0,
            doc_revision: 0,
            debug_samples: Vec::new(),
            actor: ActorId::SOLO,
            clock: 0,
            next_layer: 1,
            outbox: Vec::new(),
            outbox_enabled: false,
        };
        // Point the surface registry at the document's surface. A no-op when that is
        // the builtin; otherwise it parks on the id so the frontend's later
        // `register_surface` is recognised as "the one in use" and takes effect.
        engine.apply_document_surface();
        engine
    }

    /// Apply one input command (DESIGN.md §4).
    ///
    /// One-way by construction: nothing comes back. Reads go through
    /// [`Engine::observe`]; anything that must answer is a request (see
    /// [`command`](crate::command)).
    pub fn process(&mut self, command: impl Into<InputCommand>) {
        match command.into() {
            InputCommand::Gesture(c) => self.process_gesture(c),
            InputCommand::Doc(c) => self.process_doc(c),
            InputCommand::View(c) => self.process_view(c),
            InputCommand::Peer(c) => self.process_peer(c),
        }
    }

    /// The press-drag-release lifecycle. One path for both kinds of tool
    /// (DESIGN.md §6.8): the selection tools build an op where the brush builds a
    /// stroke, and both preview through the same `preview` DocState.
    fn process_gesture(&mut self, command: GestureCommand) {
        match command {
            GestureCommand::Start {
                tool,
                sample,
                tolerance,
            } => {
                if tool.is_selection() {
                    // A marquee or lasso fits no curve, so it has no use for the
                    // tolerance; its own decimation is a mask-cost knob (§6.8).
                    self.session.start_selection(tool, sample.pos);
                } else {
                    let seed = self.clock;
                    self.session.start_stroke(tool, sample, seed, tolerance);
                    self.debug_samples.clear();
                    self.debug_samples.push(sample);
                }
                self.refresh_live();
            }
            GestureCommand::To { sample } => {
                if self.session.is_selecting() {
                    self.session.selection_to(sample.pos);
                } else {
                    self.session.stroke_to(sample);
                    if cfg!(feature = "debug-unfrozen") {
                        self.debug_samples.push(sample);
                    }
                }
                self.refresh_live();
            }
            // The one edge that produces document state.
            GestureCommand::End => {
                if self.session.is_selecting() {
                    // One gesture, two things it can commit — which one was decided
                    // when the drag started (MISSING_FEATURES §0.4).
                    match self.session.end_shape() {
                        Some(ShapeResult::Select(op)) => self.commit(ActionKind::Select(op)),
                        Some(ShapeResult::Fill(op)) => self.commit(ActionKind::Fill {
                            layer: self.session.active_layer,
                            op,
                        }),
                        None => {}
                    }
                } else if let Some(rec) = self.session.end_stroke() {
                    self.log_debug_samples();
                    self.commit(ActionKind::CommitStroke(rec));
                }
                self.refresh_live();
            }
            GestureCommand::Cancel => {
                self.session.cancel_stroke();
                self.refresh_live();
            }
        }
    }

    /// Per-client state that is published rather than logged (PEER_DESIGN.md §7).
    /// Nothing here enters the history or the save file; it rides the presence
    /// channel so collaborators can see where this client is working.
    fn process_peer(&mut self, command: PeerCommand) {
        match command {
            // Any existing layer, including a matte. `active_layer` is *the
            // selected layer*, not "a paint target" — a frame is selected the same
            // way a paint layer is, which is what lets the frontend have one
            // selection concept instead of two (FRAME_DESIGN.md §7). A stroke aimed
            // at a matte then simply draws nothing, refused identically by `apply`
            // and by the preview path.
            PeerCommand::SetActiveLayer(id) => {
                if self.document().contains_layer(id) {
                    self.session.active_layer = id;
                }
            }
            PeerCommand::SetCursor(pos) => self.session.cursor = pos,
            PeerCommand::SetName(name) => self.session.set_name(name),
        }
    }

    /// Document-state mutations: every arm here either commits an action or
    /// navigates the history that holds them.
    fn process_doc(&mut self, command: DocCommand) {
        self.process_doc_inner(command);
        // Every arm changes the document the in-flight previews are drawn over, so
        // the fold is rebuilt once, here, rather than at each of a dozen call sites.
        // Cheap when nothing is in flight (there is nothing to fold) and correct when
        // a peer is mid-stroke while this client edits.
        self.refresh_live();
    }

    fn process_doc_inner(&mut self, command: DocCommand) {
        match command {
            DocCommand::Undo => {
                self.doc_preview = None;
                // Shared sessions log undo as an action peers can order
                // (DESIGN.md §5.4, §12.3); solo falls back to navigation.
                if let Some(target) = self.timeline.undo_as_action() {
                    self.commit(ActionKind::Undo(target));
                } else {
                    self.timeline.undo(&mut self.apply);
                    self.committed_changed();
                }
                self.apply_document_surface();
            }
            DocCommand::Redo => {
                self.doc_preview = None;
                // Redo is an `Undo` of an `Undo` in a shared session.
                if let Some(target) = self.timeline.redo_as_action() {
                    self.commit(ActionKind::Undo(target));
                } else {
                    self.timeline.redo(&mut self.apply);
                    self.committed_changed();
                }
                self.apply_document_surface();
            }
            DocCommand::Seek(to) => {
                self.doc_preview = None;
                if self.timeline.seek(to, &mut self.apply) {
                    self.committed_changed();
                    self.apply_document_surface();
                    // A scrub crosses layer additions wholesale — dragging to the
                    // start of the log withdraws every one of them — so the
                    // selected layer routinely stops existing here, where an undo
                    // has to be aimed at exactly the right step to manage it. A
                    // playhead left somewhere the brush has nowhere to go is a
                    // canvas that silently swallows the next stroke.
                    self.repoint_active_layer();
                }
            }
            DocCommand::Select(op) => self.commit(ActionKind::Select(op)),
            DocCommand::InvertSelection => self.commit(ActionKind::InvertSelection),
            DocCommand::Fill { layer, op } => self.commit(ActionKind::Fill { layer, op }),
            DocCommand::Transform { layer, affine } => {
                // The commit supersedes whatever the gesture was previewing, for
                // the same reason `SetMatteRect` drops its preview.
                self.doc_preview = None;
                // A degenerate or non-finite affine would be rejected by `apply`
                // anyway (deterministically — TRANSFORM_DESIGN.md §1); refusing it
                // here as well keeps a knowably-dead action out of the log.
                if crate::document::affine_usable(affine) {
                    self.commit(ActionKind::Transform { layer, affine });
                }
            }
            DocCommand::SetSurface(id) => {
                if id != self.document().surface {
                    self.commit(ActionKind::SetSurface(id));
                    self.apply_document_surface();
                }
            }
            DocCommand::AddLayer { carrier, above } => {
                let id = self.mint_layer();
                self.commit(ActionKind::AddLayer { id, carrier, above });
                // A freshly added layer becomes the active painting target — but
                // only if it landed. An unknown carrier adds nothing
                // (GROUP_DESIGN.md §8), and arming an id no layer has would leave
                // the next stroke with nowhere to go.
                if self.document().contains_layer(id) {
                    self.session.active_layer = id;
                }
            }
            DocCommand::AddMatte {
                carrier,
                above,
                region,
                color,
            } => {
                let id = self.mint_layer();
                self.commit(ActionKind::AddMatte {
                    id,
                    carrier,
                    above,
                    region,
                    color,
                });
                // Deliberately *not* made the active layer, unlike `AddLayer`: a
                // matte has no tile map, so painting on it is refused
                // (FRAME_DESIGN.md §7) and arming it as the target would just
                // swallow the user's next stroke.
            }
            DocCommand::SetMatteRect(id, min, max) => {
                // The committed rect supersedes whatever the drag was previewing;
                // leaving the preview up would pin the canvas to the last dragged
                // value and shadow every later edit.
                self.doc_preview = None;
                self.commit(ActionKind::SetMatteRect(id, min, max));
            }
            DocCommand::SetMatteColor(id, color) => {
                self.commit(ActionKind::SetMatteColor(id, color))
            }
            DocCommand::SetBackground(rgb) => {
                // The committed colour supersedes whatever the drag was previewing,
                // for the same reason `SetMatteRect` drops it above.
                self.doc_preview = None;
                self.commit(ActionKind::SetBackground(rgb));
            }
            DocCommand::RemoveLayer(id) => {
                self.commit(ActionKind::RemoveLayer(id));
                self.repoint_active_layer();
            }
            DocCommand::SetLayerBlend(id, blend) => {
                self.commit(ActionKind::SetLayerBlend(id, blend))
            }
            DocCommand::SetLayerClip(id, clip) => self.commit(ActionKind::SetLayerClip(id, clip)),
            DocCommand::SetLayerOpacity(id, opacity) => {
                self.commit(ActionKind::SetLayerOpacity(id, opacity))
            }
            DocCommand::SetLayerVisible(id, visible) => {
                self.commit(ActionKind::SetLayerVisible(id, visible))
            }
            DocCommand::SetLayerName(id, name) => {
                let name = normalize_layer_name(name);
                // A rename to the name it already has is not an edit, and logging it
                // would spend an undo step that appears to do nothing when reached.
                // Commit-on-blur makes this the common case: leaving a field you only
                // looked at must cost nothing.
                if self.timeline.current().layer_name(id) != name.as_deref() {
                    self.commit(ActionKind::SetLayerName(id, name));
                }
            }
            DocCommand::MoveLayer { id, carrier, above } => {
                self.commit(ActionKind::MoveLayer { id, carrier, above })
            }
        }
    }

    /// View-state mutations: nothing here is logged, replicated, or reachable by
    /// undo.
    fn process_view(&mut self, command: ViewCommand) {
        match command {
            ViewCommand::SetTool(tool) => {
                // Switching away mid-gesture abandons it rather than committing a
                // half-dragged marquee.
                self.session.cancel_stroke();
                self.session.tool = tool;
                self.refresh_live();
            }
            ViewCommand::SetBrush(brush) => {
                self.session.brush = brush;
                self.refresh_live();
            }
            ViewCommand::Pan { delta } => {
                // Grab-and-drag: content follows the cursor, so the view center moves
                // opposite by the drag delta, carried into canvas units — through the
                // whole map, since a turned or mirrored canvas sends a screen-space
                // drag somewhere else entirely.
                let delta = self.session.view.canvas_delta(delta);
                self.session.view.center -= delta;
            }
            ViewCommand::SetRotation(radians) => self.session.view.set_rotation(radians),
            ViewCommand::MirrorH => self.session.view.mirror_screen_h(),
            ViewCommand::CenterOn(point) => {
                self.session.view.center = point;
            }
            ViewCommand::Zoom { anchor, factor } => {
                self.session.view.zoom_about(anchor, factor);
            }
            ViewCommand::Resize(viewport) => {
                self.session.view.viewport = viewport;
            }
            ViewCommand::SetShapeAction(action) => self.session.shape_action = action,
            ViewCommand::SetSelectionFeather(feather) => {
                self.session.selection_feather = feather.max(0.0)
            }
            ViewCommand::SetShowPeerSelections(show) => self.session.show_peer_selections = show,
            ViewCommand::PreviewMatteRect(drag) => {
                let preview =
                    drag.map(|(id, min, max)| self.timeline.current().set_matte_rect(id, min, max));
                self.set_doc_preview(preview);
            }
            ViewCommand::PreviewBackground(rgb) => {
                let preview = rgb.map(|rgb| self.timeline.current().with_background(rgb));
                self.set_doc_preview(preview);
            }
            ViewCommand::PreviewTransform(t) => {
                let preview = t.and_then(|(layer, affine)| self.preview_transform(layer, affine));
                self.set_doc_preview(preview);
            }
            ViewCommand::SetMediaParams(params) => self.compositor_pipeline.set_media(params),
            ViewCommand::SetEnvironment(id) => self.set_environment(id),
        }
    }

    /// Replay a whole recorded stroke as a single commit: start → samples →
    /// end, skipping the per-sample live-preview refresh. `refresh_preview`
    /// re-renders the entire in-flight stroke after every sample — right for
    /// interactive drawing (the user must see each move), but O(n²) across a
    /// replay where nothing is presented in between. This renders the stroke
    /// exactly once, at commit. Used by the brush editor's test-stroke replay.
    pub fn replay_stroke(&mut self, tool: Tool, samples: &[crate::command::InputSample]) {
        self.replay_stroke_seeded(tool, samples, self.clock);
    }

    /// [`Engine::replay_stroke`] with an explicit jitter `seed` instead of the
    /// Lamport clock. Replaying the same samples repeatedly advances the clock
    /// (each replay commits), so the seed — and with it the colour dynamics and
    /// dither — changes on every replay. A caller re-rendering *one* stroke to
    /// show the effect of a brush change (the brush editor's preview) wants the
    /// jitter held fixed, so only the edited parameter moves.
    pub fn replay_stroke_seeded(
        &mut self,
        tool: Tool,
        samples: &[crate::command::InputSample],
        seed: u64,
    ) {
        let mut it = samples.iter();
        let Some(first) = it.next() else { return };
        // Replayed samples are already in canvas space and came from a fit or from a
        // generator, not from a device, so there is no device grain to declare.
        self.session
            .start_stroke(tool, *first, seed, crate::path::DEFAULT_TOLERANCE);
        for s in it {
            self.session.stroke_to(*s);
        }
        if let Some(rec) = self.session.end_stroke() {
            self.commit(ActionKind::CommitStroke(rec));
        }
        self.refresh_live();
    }

    /// Render the current canvas (preview if stroking, else committed) into
    /// `target`, through the session's own pan/zoom (DESIGN.md §6.4).
    pub fn render(&mut self, target: &wgpu::TextureView) {
        self.render_view(
            target,
            self.session.view,
            Background::Substrate,
            Chrome::Shown,
            Rendered::Live,
            Attachments::Surface,
        );
    }

    /// Render the document through `view` into a target that is **not** the engine's
    /// own surface — a second surface showing the same document (DESIGN.md §11).
    ///
    /// The navigator's miniature is the consumer: an overview of the whole piece is a
    /// second view of the canvas, and once it has somewhere to draw there is no reason
    /// for it to travel through the CPU. It used to be an [`export`](Self::export) —
    /// render, copy back, hand the browser a `<canvas>` full of bytes — and this is
    /// the same render with the copy deleted, which also deletes the frame of latency
    /// the copy cost and the megabyte the pixels occupied on the way through.
    ///
    /// `into` holds the pass-A attachments (see [`Offscreen`]); a consumer drawing
    /// repeatedly keeps them, so a refresh allocates nothing at all. `target` must
    /// carry the format [`target_format`](Self::target_format) reports and be
    /// `view.viewport` in size — a surface texture configured to match.
    ///
    /// No chrome: a selection outline belongs to the surface you are painting on, not
    /// to a thumbnail of the piece.
    pub fn render_into(
        &mut self,
        into: &mut Offscreen,
        target: &wgpu::TextureView,
        view: ViewTransform,
        background: Background,
        content: Rendered,
    ) {
        self.render_view(
            target,
            view,
            background,
            Chrome::Hidden,
            content,
            Attachments::Offscreen(into),
        );
    }

    /// The texture format this engine's pipelines render to. A frontend configuring a
    /// second surface for [`render_into`](Self::render_into) has to match it.
    pub fn target_format(&self) -> wgpu::TextureFormat {
        self.target_format
    }

    /// Render through an **explicit** view rather than the session's, choosing what
    /// sits under the paint and whether on-canvas chrome is drawn (DESIGN.md §6.4,
    /// FRAME_DESIGN.md §6).
    ///
    /// This is the seam export needs: exporting a frame is rendering at
    /// `frame.rect × scale`, centred on the frame, at `zoom = scale` — the same
    /// path the screen takes, so what is written is what was seen. `render` reading
    /// `session.view` instead of taking one is exactly what made "export" a
    /// screenshot of the viewport.
    ///
    /// Private, with [`Engine::export`] and [`Engine::render_into`] as the two
    /// consumers: what a caller may choose is a view, a ground and where the
    /// attachments live, never whether chrome is drawn (it is, for the screen alone)
    /// nor how the two are wired together.
    fn render_view(
        &mut self,
        target: &wgpu::TextureView,
        view: ViewTransform,
        background: Background,
        chrome: Chrome,
        content: Rendered,
        attachments: Attachments,
    ) {
        let doc = match content {
            Rendered::Live => self.presented(),
            Rendered::Committed => self.timeline.current(),
        };
        let groups = self.composite_groups(doc, None);

        // The substrate is document state now (FRAME_DESIGN.md §5), so the ground a
        // piece was painted on travels with it instead of living in whichever
        // frontend happened to render it.
        let bg_channels = self.color_space.rgb_to_channels(doc.background);
        // Chrome never reaches a file: an exported image gets no selection outline
        // (FRAME_DESIGN.md §6). Keyed on `chrome`, deliberately *not* on the
        // background — a substrate export is still an export, and tying the two
        // together silently leaked the outline into every opaque PNG.
        //
        // Own the masks (a handful of `Arc` bumps) so the borrow of `doc` — and with
        // it of `self` — ends before the compositor is borrowed mutably below.
        let outlines: Vec<(crate::document::Selection, Option<[f32; 3]>)> = match chrome {
            Chrome::Hidden => Vec::new(),
            Chrome::Shown => self.visible_selections(),
        };
        let outlines: Vec<SelectionOutline<'_>> = outlines
            .iter()
            .map(|(selection, tint)| SelectionOutline {
                selection,
                tint: *tint,
            })
            .collect();
        let scene = CompositeScene {
            background: bg_channels,
            groups: &groups,
            outlines: &outlines,
            transparent: background == Background::Transparent,
        };
        match attachments {
            Attachments::Surface => {
                self.compositor
                    .render(&self.compositor_pipeline, target, view, scene)
            }
            Attachments::Offscreen(into) => into
                .get(&self.compositor_pipeline, view.viewport)
                .render(&self.compositor_pipeline, target, view, scene),
        }
    }

    /// Render the current canvas to a CPU-side image at the viewport size
    /// (DESIGN.md §9). The backbone of golden tests. The target uses the engine's
    /// configured format, so it matches on-screen rendering.
    /// Blocking, and therefore **native-only**: WebGPU has no blocking poll, so
    /// this shape cannot work on the web (see `gpu::readback`). The frontend uses
    /// [`export`](Self::export), which awaits the map.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn render_to_image(&mut self) -> RgbaImage {
        // One render per call, so nothing is kept: the attachments are allocated here
        // and dropped with this `Offscreen`.
        let (target, size) = self.render_offscreen(
            &mut Offscreen::default(),
            self.session.view,
            Background::Substrate,
            Chrome::Shown,
            Rendered::Live,
        );
        let pixels = crate::gpu::readback::read_rgba8_blocking(&self.gpu, &target, size);
        RgbaImage::from_target_bytes(size.width, size.height, pixels, self.target_format)
    }

    /// Render through an explicit view into an offscreen texture, ready to be read
    /// back. Split out so the blocking and async readbacks share every step but
    /// the wait.
    fn render_offscreen(
        &mut self,
        into: &mut Offscreen,
        view: ViewTransform,
        background: Background,
        chrome: Chrome,
        content: Rendered,
    ) -> (wgpu::Texture, Extent2) {
        let size = view.viewport;
        let target = self.gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("stark export target"),
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
        // The caller's attachments, not the surface's — see [`Attachments`].
        self.render_view(
            &target_view,
            view,
            background,
            chrome,
            content,
            Attachments::Offscreen(into),
        );
        (target, size)
    }

    /// What exporting `frame` at `scale` would produce, without producing it —
    /// so a dialog can show the pixel size before committing to the render.
    ///
    /// `frame` names a **matte layer** whose rect is the piece (FRAME_DESIGN.md
    /// §6). With no frame it falls back to the painted bounds, and on an empty
    /// canvas to the current viewport, so export always has *something* to mean.
    pub fn export_plan(&self, frame: Option<LayerId>, scale: ExportScale) -> Result<ExportPlan> {
        let (min, max) = self.export_rect(frame);
        let (w, h) = (max.x - min.x, max.y - min.y);
        if !(w.is_finite() && h.is_finite()) || w < 1.0 || h < 1.0 {
            return Err(EngineError::Export(format!(
                "frame is too small to export ({w:.0} × {h:.0} canvas px)"
            )));
        }
        let zoom = match scale {
            ExportScale::Factor(f) => f,
            ExportScale::Width(px) => px as f32 / w,
        };
        if !(zoom.is_finite() && zoom > 0.0) {
            return Err(EngineError::Export("export scale must be positive".into()));
        }
        // Round rather than truncate, so a 1× export of a 100.5-px frame is 101
        // rather than silently dropping most of a pixel off two edges.
        let size = Extent2::new(
            (w * zoom).round().max(1.0) as u32,
            (h * zoom).round().max(1.0) as u32,
        );
        if size.width > MAX_EXPORT_DIM || size.height > MAX_EXPORT_DIM {
            return Err(EngineError::Export(format!(
                "export is {} × {} px; the limit is {MAX_EXPORT_DIM}",
                size.width, size.height
            )));
        }
        Ok(ExportPlan {
            min,
            max,
            size,
            zoom,
        })
    }

    /// Render a frame to a CPU-side image (FRAME_DESIGN.md §6).
    ///
    /// This is the same path the screen takes — every visible layer composited
    /// through the media pass — just with the view centred on the frame at
    /// `zoom = scale`. Nothing is special-cased: a frame matte covers only
    /// *outside* its rect, which is clipped away here, so it contributes nothing
    /// to its own export, while a ground matte is inside and contributes exactly
    /// what it should.
    /// Renders immediately and returns a future for the **readback**, which is the
    /// only asynchronous part (DESIGN.md §7 — on WebGPU `mapAsync` settles only
    /// when the browser's event loop runs, so there is no way to block on it).
    ///
    /// Deliberately *not* an `async fn`. An `async fn` would hold `&mut self` for
    /// the whole readback, and a frontend must take that borrow from a shared cell
    /// — so the engine would stay locked across an await during which the UI
    /// re-renders and tries to read it, panicking with `AlreadyBorrowedMut`. This
    /// shape ends the borrow when `export` returns: the returned future owns a
    /// cloned [`GpuContext`] (cheap — the handles are reference-counted) and the
    /// target texture, and touches the engine not at all.
    ///
    /// `content` chooses whether the in-flight gesture is in the picture: a file
    /// export takes [`Rendered::Live`], since that is what the artist is looking at.
    /// (Anything refreshed per *committed* change wants [`Rendered::Committed`]
    /// instead — see [`render_into`](Self::render_into), which is the shape that
    /// suits a render repeated on a cadence rather than written to a file.)
    ///
    /// `into` is where the render's attachments live. It renders **beside** the
    /// surface rather than into it, so it never touches the screen's; whether its own
    /// outlive the call is the caller's call, and the caller is the only one who knows
    /// (see [`Offscreen`]) — a `&mut Offscreen::default()` for a one-shot, a held one
    /// for a render that repeats.
    ///
    /// ```ignore
    /// let readback = { engine.write().export(&mut own, frame, scale, bg, content)? }; // borrow ends
    /// let image = readback.await;
    /// ```
    pub fn export(
        &mut self,
        into: &mut Offscreen,
        frame: Option<LayerId>,
        scale: ExportScale,
        background: Background,
        content: Rendered,
    ) -> Result<impl std::future::Future<Output = RgbaImage> + use<>> {
        let plan = self.export_plan(frame, scale)?;
        let view = plan.view();
        // No chrome: a selection outline or any other on-canvas affordance is a
        // thing to draw *with*, never a thing to ship.
        let (target, size) = self.render_offscreen(into, view, background, Chrome::Hidden, content);
        let gpu = self.gpu.clone();
        // Captured, not read through `self`: the future deliberately does not
        // borrow the engine.
        let format = self.target_format;
        Ok(async move {
            let pixels = crate::gpu::readback::read_rgba8(&gpu, &target, size).await;
            RgbaImage::from_target_bytes(size.width, size.height, pixels, format)
        })
    }

    /// The compositor's draw list for `doc`, bottom-to-top: every visible layer's
    /// tiles and mattes, each tagged with its layer opacity, cut into blend groups
    /// (MISSING_FEATURES §0.4, GROUP_DESIGN.md §7).
    ///
    /// Consecutive layers that need no isolation share one `Run` — they compose
    /// correctly against each other and against everything below under
    /// premultiplied "over", so a document that uses no blend modes, no clipping
    /// and no groups produces exactly one `Run` and the compositor's work is
    /// unchanged. Anything else becomes a group of its own, because its mode and
    /// its clip are both defined against *what is underneath it*, which means it
    /// has to be composited in isolation first.
    ///
    /// A layer that **carries** others is a group, and composites as a `Stack`:
    /// its own content at the bottom, then each carried layer merging into what is
    /// beneath it *within the group* (GROUP_DESIGN.md §2). The group as a whole
    /// then merges outward through its own — that is, its base's — blend mode,
    /// clip and opacity.
    ///
    /// Within a run this is an *ordered* item list rather than a flat tile list
    /// because a matte has to composite at its own place in the stack — a frame over
    /// the painting, a ground under it (FRAME_DESIGN.md §4.4). The compositor
    /// re-batches consecutive tiles into one instanced draw, so an all-paint document
    /// costs nothing for it.
    ///
    /// `only` restricts the list to a single layer — the eyedropper's
    /// sample-one-layer option (MISSING_FEATURES §0.2). It means that layer's *own*
    /// paint: what it carries is left out, and its mode, clip and opacity go with
    /// it, since a sample is of the paint that is there rather than of the part of
    /// it that survives its surroundings. Sharing this with rendering is what makes
    /// a sample come off the same stack the screen draws.
    fn composite_groups(&self, doc: &DocState, only: Option<LayerId>) -> Vec<CompositeGroup> {
        if let Some(id) = only {
            let Some(layer) = doc.layer(id).filter(|l| l.visible && l.opacity > 0.0) else {
                return Vec::new();
            };
            let items = self.layer_items(layer);
            return if items.is_empty() {
                Vec::new()
            } else {
                vec![CompositeGroup::run(BlendMode::Normal, false, items)]
            };
        }
        self.composite_stack(doc.layers.iter())
    }

    /// One stack's worth of groups — the root's, or a layer's carried stack.
    fn composite_stack<'a>(&self, layers: impl Iterator<Item = &'a Layer>) -> Vec<CompositeGroup> {
        let mut groups: Vec<CompositeGroup> = Vec::new();
        for layer in layers {
            // Hiding a layer hides what it carries: the group is the layer
            // (GROUP_DESIGN.md §3), so its visibility is the group's.
            if !layer.visible || layer.opacity <= 0.0 {
                continue;
            }
            let own = self.layer_items(layer);
            let carried = self.composite_stack(layer.carries.iter());
            // An empty layer is dropped rather than given a group. For `Normal`
            // that only saves a loop; for a blend mode or a clip it saves two
            // render passes that provably compute the identity, which is what
            // keeps a stack of empty glow layers free. A layer that carries
            // something visible is not empty, whatever its own content.
            if own.is_empty() && carried.is_empty() {
                continue;
            }
            let mut group = if carried.is_empty() {
                // A leaf: its opacity is already folded into its tiles, which is
                // equivalent because tiles within a layer do not overlap.
                CompositeGroup::run(layer.blend, layer.clip, own)
            } else {
                // A group: the base's own paint at the bottom of it, then what it
                // carries. Its opacity applies to the composite, not to the
                // members — they overlap.
                let mut members = Vec::with_capacity(carried.len() + 1);
                if !own.is_empty() {
                    members.push(CompositeGroup::run(BlendMode::Normal, false, own));
                }
                members.extend(carried);
                CompositeGroup::stack(layer.blend, layer.clip, layer.opacity, members)
            };
            // Merge into the run below when neither side needs isolating — the
            // fast path, and the reason an ordinary document is one group.
            let merged = match groups.last_mut() {
                Some(last) if last.is_direct() && group.is_direct() => {
                    if let (GroupContent::Run(items), GroupContent::Run(more)) =
                        (&mut last.content, &mut group.content)
                    {
                        items.append(more);
                    }
                    true
                }
                _ => false,
            };
            if !merged {
                groups.push(group);
            }
        }
        groups
    }

    /// What one layer's own content draws, without what it carries.
    fn layer_items(&self, layer: &Layer) -> Vec<CompositeItem> {
        match &layer.content {
            LayerContent::Paint(tiles) => tiles
                .iter()
                .map(|(coord, handle)| CompositeItem::Tile {
                    coord: *coord,
                    handle: handle.clone(),
                    opacity: layer.opacity,
                })
                .collect(),
            LayerContent::Matte { region, color } => {
                let (min, max) = region.rect();
                vec![CompositeItem::Matte(MatteDraw {
                    rect: [min.x, min.y, max.x, max.y],
                    // sRGB in the log, working-space channels on the GPU — the
                    // same conversion the brush colour gets, so a matte means
                    // the same colour in an Oklab and a Mixbox document.
                    channels: self.color_space.rgb_to_channels(*color),
                    opacity: layer.opacity,
                })]
            }
        }
    }

    /// Sample the canvas colour at `at` — the eyedropper (MISSING_FEATURES §0.2).
    ///
    /// A **request**, not a command: it has to answer, so it stays a direct method
    /// beside `save_bytes` rather than joining [`InputCommand`](crate::InputCommand),
    /// whose whole property is that nothing comes back (DESIGN.md §4).
    ///
    /// What it samples is the **raw layer channels**, not the composited, *lit*
    /// result the screen shows, and that is the decision the feature turns on. The
    /// media pass lights the paint, tonemaps it and encodes sRGB, so picking its
    /// output would load the brush with a colour the palette never mixed — and in a
    /// Mixbox document (DESIGN.md §6.7) with a display colour rather than the pigment
    /// mixture, which would make picking the mix back up impossible. That is the
    /// entire reason pigment mixing is worth having.
    ///
    /// `None` where the sampled patch holds no paint: the substrate is the ground,
    /// not something a brush picks up, so bare canvas answers "nothing here" rather
    /// than quietly loading the brush with the paper colour.
    ///
    /// Renders immediately and returns a future for the **readback**, the only
    /// asynchronous part — the same shape as [`Engine::export`], and for the same
    /// reason: an `async fn` would hold `&mut self` across an await during which the
    /// frontend re-renders and tries to read the engine.
    pub fn pick_color(
        &mut self,
        at: crate::geom::Vec2,
        options: PickOptions,
    ) -> impl std::future::Future<Output = Option<[f32; 3]>> + use<> {
        let radius = options.radius.min(MAX_PICK_RADIUS);
        let size = Extent2::new(2 * radius + 1, 2 * radius + 1);
        // Centred on the canvas *pixel* the point falls in rather than on the point
        // itself: pass A samples tile textures bilinearly, so a fractional offset
        // would blend neighbouring texels and a "point sample" would answer with a
        // colour that is at neither of them. Snapping puts every fragment on a texel
        // centre, so radius 0 reports exactly the texel under the cursor.
        let view = ViewTransform {
            center: crate::geom::Vec2::new(at.x.floor() + 0.5, at.y.floor() + 0.5),
            zoom: 1.0,
            // Axis-aligned with the *canvas*: the sampled square is a patch of the
            // painting, so which way the easel is turned cannot change which texels
            // fall in it.
            rotation: 0.0,
            flip_h: false,
            viewport: size,
        };
        // The *presented* document, so a sample agrees with what is on screen —
        // including a collaborator's stroke that has not committed yet.
        let groups = {
            let doc = self.presented();
            let only = match options.source {
                PickSource::Composite => None,
                PickSource::Layer(id) => Some(id),
            };
            self.composite_groups(doc, only)
        };

        let (color_format, aux_format) = self.compositor_pipeline.channel_formats();
        // `read_rgba16f` decodes four halves per texel. Both colour spaces store the
        // colour channels that way (DESIGN.md §6.1); a new one that did not would
        // have to say so here rather than silently mis-decoding.
        debug_assert_eq!(color_format, wgpu::TextureFormat::Rgba16Float);
        let color = self.pick_target(
            "stark pick color",
            color_format,
            size,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        // Written by pass A and never read: the composite pipeline has two targets,
        // and the height it accumulates says how *much* paint is there, not what
        // colour it is.
        let aux = self.pick_target(
            "stark pick aux",
            aux_format,
            size,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        self.compositor.composite_channels(
            &self.compositor_pipeline,
            &color.create_view(&wgpu::TextureViewDescriptor::default()),
            &aux.create_view(&wgpu::TextureViewDescriptor::default()),
            view,
            &groups,
        );

        // Captured, not read through `self`: the future deliberately does not borrow
        // the engine (see `export`). The colour space is an `Arc`, so carrying the
        // channels→RGB conversion into it costs a refcount bump.
        let gpu = self.gpu.clone();
        let color_space = self.color_space.clone();
        async move {
            let texels = crate::gpu::readback::read_rgba16f(&gpu, &color, size).await;
            mean_channels(&texels).map(|c| color_space.channels_to_rgb(c))
        }
    }

    /// A small offscreen target for one eyedropper sample.
    fn pick_target(
        &self,
        label: &str,
        format: wgpu::TextureFormat,
        size: Extent2,
        usage: wgpu::TextureUsages,
    ) -> wgpu::Texture {
        self.gpu.device.create_texture(&wgpu::TextureDescriptor {
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
        })
    }

    /// The canvas-space rect an export covers: the named frame, else the painted
    /// bounds, else the viewport.
    fn export_rect(&self, frame: Option<LayerId>) -> (crate::geom::Vec2, crate::geom::Vec2) {
        let doc = self.timeline.current();
        if let Some(id) = frame
            && let Some(region) = doc.layer(id).and_then(|l| l.matte_region())
        {
            return region.rect();
        }
        if let Some((min, max)) = doc.bounds.tile_range() {
            let t = crate::geom::TILE_SIZE as f32;
            return (
                crate::geom::Vec2::new(min.x as f32 * t, min.y as f32 * t),
                crate::geom::Vec2::new((max.x + 1) as f32 * t, (max.y + 1) as f32 * t),
            );
        }
        // Everything the viewport shows — a *bound* under rotation, which is the
        // safe direction: an export with nothing painted and no frame should not
        // crop tighter than what the artist is looking at.
        self.session.view.visible_bounds()
    }

    /// Snapshot the document as a saveable [`DocumentFile`] (DESIGN.md §8),
    /// bundling the brush-shape assets that strokes actually reference (§6.6).
    pub fn document_file(&self) -> DocumentFile {
        let actions = self.timeline.clone_actions();
        let mut referenced = std::collections::HashSet::new();
        for action in &actions {
            if let ActionKind::CommitStroke(rec) = &action.kind
                && let BrushShape::Stamp(id) = rec.brush.shape
            {
                referenced.insert(id);
            }
        }
        let assets = self
            .apply
            .assets
            .all_bytes()
            .into_iter()
            .filter(|(id, _)| referenced.contains(id))
            .collect();
        let mut file = DocumentFile::new(actions);
        file.canvas.color_space = self.color_space.id();
        file.canvas.surface = self.initial_surface;
        file.assets = assets;
        file
    }

    /// Serialize the document to the compact on-disk container (DESIGN.md §8).
    pub fn save_bytes(&self) -> Result<Vec<u8>> {
        self.document_file().to_bytes()
    }

    /// Replace the document by replaying a loaded file's action log. The full
    /// undo timeline is available afterwards — undo-after-load (DESIGN.md §8).
    pub fn load_document(&mut self, file: &DocumentFile) {
        // The surface the log starts from, before `reset_document` seeds with it.
        // Replayed `SetSurface` actions move it from there (DESIGN.md §6.4).
        self.initial_surface = file.canvas.surface;
        self.reset_document();
        // Match the document's color space before replaying (DESIGN.md §6.7).
        if file.canvas.color_space != self.color_space.id() {
            self.rebuild_gpu_for(file.canvas.color_space);
        }
        // Brush assets must be available before replaying strokes that use them.
        for (_, bytes) in &file.assets {
            if let Err(e) = self.apply.assets.insert_bytes(bytes) {
                eprintln!("skipping unreadable brush asset: {e}");
            }
        }
        // Replay only the *effective* sequence: a file saved from a shared
        // session is the full log, including `Undo` actions and the actions
        // they suppress (DESIGN.md §12.3). A solo load flattens those away.
        for action in effective_actions(&file.actions) {
            self.replay_one(action);
        }
        self.resync_counters(&file.actions);
        // Whatever the replayed log left the document on.
        self.apply_document_surface();
    }

    /// Decode and load a container produced by [`Engine::save_bytes`].
    pub fn load_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        let file = DocumentFile::from_bytes(bytes)?;
        self.load_document(&file);
        Ok(())
    }

    /// Replay a document, invoking `on_frame` with the rendered image after each
    /// action — a timelapse (DESIGN.md §8). Ends with the document fully loaded.
    ///
    /// Native-only, because it reads each frame back with the blocking path. Making
    /// it web-capable means awaiting the readback per frame — a change to this
    /// signature, not to the replay.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn replay_timelapse(&mut self, file: &DocumentFile, mut on_frame: impl FnMut(RgbaImage)) {
        self.reset_document();
        for (_, bytes) in &file.assets {
            let _ = self.apply.assets.insert_bytes(bytes);
        }
        for action in effective_actions(&file.actions) {
            self.replay_one(action);
            on_frame(self.render_to_image());
        }
        self.resync_counters(&file.actions);
    }

    /// A snapshot of UI-facing state (DESIGN.md §7).
    pub fn observe(&self) -> ObservableState {
        let doc = self.timeline.current();
        // The layers and the substrate colour are read from the *previewed*
        // document when one is in flight, so the frame's handles track a drag and
        // the colour swatch tracks the picker (both live in the preview,
        // FRAME_DESIGN.md §7, §5) instead of lagging on the committed value — which
        // for the colour would leave the panel disagreeing with the canvas it
        // controls, since rendering reads `presented`.
        //
        // Deliberately only those two. `has_selection` must stay committed-only —
        // a marquee drag would otherwise flash the selection bar in and out before
        // anything is selected — and that is asserted by
        // `a_selection_gesture_commits_the_same_op_it_previewed`. A stroke preview
        // changes no presentation property, so it is not consulted here at all.
        let shown = self.doc_preview.as_ref().unwrap_or(doc);
        // Flattened in **composite order** — each stack bottom-to-top, a group's
        // base before what it carries — with the tree carried alongside as `depth`
        // and `carrier` (GROUP_DESIGN.md §6). Flat rather than nested because that
        // is the order a panel draws in and the order the compositor draws in, and
        // one list that means both is one thing to keep in agreement.
        let mut layers: Vec<LayerInfo> = Vec::new();
        let mut carriers: Vec<LayerId> = Vec::new();
        shown.visit(&mut |l, depth| {
            carriers.truncate(depth);
            layers.push(LayerInfo {
                id: l.id,
                blend: l.blend,
                clip: l.clip,
                opacity: l.opacity,
                visible: l.visible,
                carrier: carriers.last().copied(),
                depth,
                is_group: l.is_group(),
                has_backdrop: shown.has_backdrop(l.id),
                name: l.name.clone(),
                matte: match &l.content {
                    LayerContent::Matte { region, color } => {
                        let (min, max) = region.rect();
                        Some(MatteInfo {
                            min,
                            max,
                            color: *color,
                        })
                    }
                    LayerContent::Paint(_) => None,
                },
            });
            carriers.push(l.id);
        });
        ObservableState {
            can_undo: self.timeline.can_undo(),
            can_redo: self.timeline.can_redo(),
            is_stroking: self.session.is_stroking(),
            tool: self.session.tool,
            brush: self.session.brush,
            view: self.session.view,
            bounds: doc.bounds,
            doc_revision: self.doc_revision,
            active_layer: self.session.active_layer,
            layers,
            has_selection: doc.has_selection(self.actor),
            selection_hull: doc.selection_of(self.actor).hull(),
            shape_action: self.session.shape_action,
            selection_feather: self.session.selection_feather,
            show_peer_selections: self.session.show_peer_selections,
            media: self.compositor_pipeline.media(),
            environment: self.environment.id(),
            color_space: self.color_space.id(),
            surface: doc.surface,
            background: shown.background,
        }
    }

    /// The current committed document state.
    pub fn document(&self) -> &DocState {
        self.timeline.current()
    }

    /// Where the history playhead stands and how far it can travel, in actions —
    /// or `None` for a document whose history is not this client's alone to walk
    /// (a shared session). See
    /// [`Timeline::scrub_range`](crate::document::Timeline::scrub_range).
    ///
    /// A **request** rather than a field of [`ObservableState`]: it is asked for
    /// only while a scrubber is on screen, and putting it in the projection would
    /// have every command — every pointer sample of every stroke — pay for a
    /// number nothing else reads.
    pub fn scrub_range(&self) -> Option<(usize, usize)> {
        self.timeline.scrub_range()
    }

    /// A caption per action across the whole scrub range, oldest first — what a
    /// scrubber labels its ticks with.
    pub fn scrub_labels(&self) -> Vec<&'static str> {
        self.timeline.scrub_labels()
    }

    /// The GPU context this engine renders with (for surface/readback setup).
    pub fn gpu(&self) -> &GpuContext {
        &self.gpu
    }

    /// The current pan/zoom view (for mapping pointer input to canvas space).
    pub fn view(&self) -> ViewTransform {
        self.session.view
    }

    /// The current media/lighting parameters (DESIGN.md §6.3).
    pub fn media_params(&self) -> crate::gpu::MediaParams {
        self.compositor_pipeline.media()
    }

    /// Import a brush-shape image (PNG bytes), returning its content id for use
    /// in `BrushParams::shape = BrushShape::Stamp(id)` (DESIGN.md §6.6).
    pub fn import_brush(&self, png_bytes: &[u8]) -> Result<AssetId> {
        self.apply.assets.import(png_bytes)
    }

    // --- collaboration (DESIGN.md §12) -----------------------------------
    //
    // The engine stays network-agnostic: it owns the merge semantics (the
    // `ReplicatedTimeline`) and these hooks; `stark-net` owns the wire.

    /// Whether this engine is in a shared session (replicated timeline active).
    pub fn is_shared(&self) -> bool {
        self.outbox_enabled
    }

    /// This engine's author id for new actions.
    pub fn actor(&self) -> ActorId {
        self.actor
    }

    /// Start sharing the **current** document as `actor` (the host side).
    /// Converts the linear history into a [`ReplicatedTimeline`] over the same
    /// log. Solo-authored actions ([`ActorId::SOLO`]) are rewritten to `actor`
    /// — done once, before any peer has seen them — so the sharer can still
    /// undo their pre-share strokes (undo targets *my* actions, §12.3).
    pub fn start_collaboration(&mut self, identity: impl Into<Identity>) {
        if self.is_shared() {
            return;
        }
        let identity = identity.into();
        let actor = identity.actor;
        let mut log = self.timeline.clone_actions();
        for a in &mut log {
            if a.id.actor == ActorId::SOLO {
                a.id.actor = actor;
            }
        }
        // Replay from the surface this document's log *starts* from, not from the
        // default — same base state `reset_document` builds, so re-hosting a document
        // that was created on a non-default canvas doesn't silently move it.
        let initial = DocState::with_layer(ROOT_LAYER).with_surface(self.initial_surface);
        let ctx = &mut self.apply;
        self.timeline = Box::new(ReplicatedTimeline::from_log(actor, initial, log, ctx));
        self.actor = actor;
        self.session.adopt_identity(identity);
        self.outbox_enabled = true;
        self.doc_preview = None;
        self.committed_changed();
        // New actor, new layer-id space: this client's counter restarts, and the
        // pre-share layers keep the `SOLO` ids they were minted with.
        self.next_layer = 1;
        self.refresh_live();
    }

    /// Join a shared session (the peer side): replace the document with the
    /// session's **full** log — including `Undo` actions, which the replicated
    /// timeline resolves — and author future actions as `actor`.
    pub fn join_collaboration(&mut self, file: &DocumentFile, identity: impl Into<Identity>) {
        let identity = identity.into();
        let actor = identity.actor;
        // The surface the shared log starts from; replayed `SetSurface` actions
        // move it from there, exactly as in `load_document` (DESIGN.md §6.4).
        self.initial_surface = file.canvas.surface;
        self.reset_document();
        if file.canvas.color_space != self.color_space.id() {
            self.rebuild_gpu_for(file.canvas.color_space);
        }
        for (_, bytes) in &file.assets {
            if let Err(e) = self.apply.assets.insert_bytes(bytes) {
                tracing::warn!("skipping unreadable brush asset: {e}");
            }
        }
        let ctx = &mut self.apply;
        let initial = DocState::with_layer(ROOT_LAYER).with_surface(self.initial_surface);
        self.timeline = Box::new(ReplicatedTimeline::from_log(
            actor,
            initial,
            file.actions.clone(),
            ctx,
        ));
        self.actor = actor;
        self.session.adopt_identity(identity);
        self.resync_counters(&file.actions);
        self.outbox_enabled = true;
        // Whatever the replayed log left the document on.
        self.apply_document_surface();
        self.committed_changed();
        self.refresh_live();
    }

    /// Leave a shared session: stop queueing broadcasts and forget everyone who was
    /// in it. The replicated timeline (and the shared log) stays — editing continues
    /// solo on the same canvas, and a later [`Self::start_collaboration`] re-shares
    /// it.
    ///
    /// The peers' *selections* stay in the document, because replay still needs them
    /// to reproduce their strokes; they simply stop being drawn, since the roster is
    /// what decides that (PEER_DESIGN.md §3).
    pub fn end_collaboration(&mut self) {
        self.outbox.clear();
        self.outbox_enabled = false;
        self.peers.clear();
        self.refresh_live();
    }

    /// Integrate an action authored by a peer (DESIGN.md §12.1). Idempotent —
    /// duplicates are rejected by id. Advances the Lamport clock past the
    /// remote action so future local ids order after everything seen.
    pub fn merge_remote(&mut self, action: Action) -> bool {
        self.clock = self.clock.max(action.id.lamport + 1);
        let author = action.id.actor;
        let removed = match &action.kind {
            ActionKind::RemoveLayer(id) => Some(*id),
            _ => None,
        };
        let ctx = &mut self.apply;
        let merged = self.timeline.merge(action, ctx);
        if merged {
            // Replaces the document every frozen head was composited onto.
            self.committed_changed();
            // A gesture is a thing that becomes an action, so the action's arrival is
            // the end-of-gesture signal — no id to correlate, and no window in which
            // both the live copy and the committed one are drawn.
            self.peers.clear_gesture(author);
            if removed.is_some_and(|id| self.session.active_layer == id) {
                // A peer deleting the layer this client is painting on used to leave
                // it pointed at a layer that no longer exists, after which every
                // stroke was silently refused by `apply` with nothing on screen to
                // explain it (PEER_DESIGN.md §9).
                self.repoint_active_layer();
            }
            self.refresh_live();
        }
        // A peer may have switched the surface (DESIGN.md §6.4).
        self.apply_document_surface();
        merged
    }

    /// Drain locally-committed actions awaiting broadcast (empty when solo).
    pub fn take_outbox(&mut self) -> Vec<Action> {
        std::mem::take(&mut self.outbox)
    }

    /// How the timeline has serviced materializations (DESIGN.md §12.6): the
    /// commutation fast paths versus rewind-and-replay. Zeros when solo. For
    /// tests and diagnostics — pixels can't tell the paths apart, by design.
    pub fn timeline_stats(&self) -> TimelineStats {
        self.timeline.stats()
    }

    // --- presence (PEER_DESIGN.md §4) -------------------------------------
    //
    // Symmetric with the action hooks above, and deliberately a separate channel:
    // nothing in the action log ever references presence, which is what lets the
    // transport drop, coalesce or delay these frames without touching convergence.

    /// Whether [`take_presence`](Self::take_presence) would do anything at `now` —
    /// a `&self` test a pump can run without borrowing the engine mutably.
    ///
    /// This is what keeps an idle shared session free. The pump has to wake on a
    /// fixed cadence (that is what makes the latch coalesce, §5.1, and it is the
    /// engine's only clock), but *waking* need not mean working: a tick where
    /// nothing has moved and no peer is due to expire should cost this comparison
    /// and nothing else — no mutable borrow, no roster rebuild, and above all no
    /// write to the signal the engine lives in, which would mark it dirty and
    /// re-render every component that reads it.
    ///
    /// Conservative in the same direction as [`Session::publish_due`]: it may say
    /// yes where the drain then finds nothing, never the reverse.
    pub fn presence_due(&self, now: f64) -> bool {
        self.peers.expiry_due(now) || (self.outbox_enabled && self.session.publish_due(now))
    }

    /// A counter that changes whenever the peer roster does, so a frontend can tell
    /// that its projection is stale without rebuilding it (PEER_DESIGN.md §4).
    pub fn peers_revision(&self) -> u64 {
        self.peers.revision()
    }

    /// This client's presence, if anything a peer would care about has changed since
    /// the last call (PEER_DESIGN.md §5.1). Also expires peers that have gone quiet,
    /// since this is called on the frontend's publish cadence — the only clock
    /// `stark-core` has, because it deliberately owns none.
    ///
    /// `frame` is `None` when solo: presence with nobody to read it is pure cost.
    /// `repaint` reports whether the expiry changed the canvas — a stalled gesture
    /// or a departed peer takes paint off it, and a caller that drops that bit
    /// leaves the stale stroke on screen until something else forces a paint.
    pub fn take_presence(&mut self, now: f64) -> PresenceTick {
        self.now = now.max(self.now);
        let repaint = self.peers.tick(self.now).canvas;
        if repaint {
            self.refresh_live();
        }
        let frame = self
            .outbox_enabled
            .then(|| self.session.publish(self.now))
            .flatten();
        PresenceTick { frame, repaint }
    }

    /// The farewell frame, so peers drop this client at once instead of waiting out
    /// [`PEER_TIMEOUT`](crate::peer::PEER_TIMEOUT). Send it before tearing the
    /// transport down.
    pub fn leaving_presence(&mut self) -> PeerFrame {
        self.session.publish_leaving()
    }

    /// Integrate presence published by `actor`, whose identity comes from the
    /// transport's authenticated origin and never from the frame body — a peer can
    /// publish its own presence and nobody else's (PEER_DESIGN.md §7).
    ///
    /// Returns whether the **canvas** changed, i.e. whether a repaint is owed. A
    /// frame that only moved a cursor or a selected layer returns `false`: those are
    /// chrome, drawn from the roster projection, which a caller notices moved through
    /// [`peers_revision`](Self::peers_revision) instead. Presence arrives at pointer
    /// rate from every peer at once, so the difference between the two questions is
    /// the difference between a compositor pass per remote pointer move and none.
    ///
    /// Dated by `now`, the **caller's** clock — the same one it hands
    /// [`take_presence`](Self::take_presence) — folded into [`Self::now`] so the
    /// engine's clock stays monotonic. It used to be dated by `self.now` alone, on
    /// the assumption the pump advanced it every tick; but the pump skips
    /// `take_presence` on a tick with nothing to publish, so on a client that was
    /// only *watching* the clock advanced per [`HEARTBEAT`](crate::peer::HEARTBEAT)
    /// — and every frame merged in between aged a whole heartbeat at once when the
    /// expiry finally ran, which took down live gestures whose frames were arriving
    /// thirty times a second.
    pub fn merge_presence(&mut self, actor: ActorId, frame: PeerFrame, now: f64) -> bool {
        self.now = now.max(self.now);
        let now = self.now;
        if actor == self.actor {
            // Our own frame, echoed back by a flood transport. The local session is
            // the authority on this client; taking it back off the wire would fight
            // with it.
            return false;
        }
        let change = self.peers.merge(actor, frame, now);
        if change.canvas {
            self.refresh_live();
        }
        change.canvas
    }

    /// Everyone else in the session, in ascending [`ActorId`] order (empty solo).
    pub fn peers(&self) -> impl Iterator<Item = &Peer> {
        self.peers.iter()
    }

    /// This client's display name, as peers see it.
    pub fn name(&self) -> &str {
        self.session.name()
    }

    /// Every imported brush asset (id + canonical PNG bytes) — used to seed a
    /// transport session's asset mirror so peers can fetch any brush a future
    /// stroke references (DESIGN.md §12.4).
    pub fn all_asset_bytes(&self) -> Vec<(AssetId, Vec<u8>)> {
        self.apply.assets.all_bytes()
    }

    /// The canonical PNG bytes of one imported brush asset, if loaded — for
    /// seeding a live session's mirror or a second (preview) engine.
    pub fn asset_bytes(&self, id: AssetId) -> Option<Vec<u8>> {
        self.apply.assets.bytes(id)
    }

    /// Whether a brush asset is loaded in this engine.
    pub fn has_asset(&self, id: AssetId) -> bool {
        self.apply.assets.contains(id)
    }

    /// Note that the **committed** document has been replaced: every cached
    /// [`FrozenHead`] built against the old one is stale, and anything the frontend
    /// derived from it is out of date.
    ///
    /// One call rather than two counters bumped side by side at each of seven sites
    /// — a commit, either half of undo/redo, a merge, a share, a join, a reset —
    /// because "these advance together" is the property that has to hold, and a
    /// site that remembered one and forgot the other would be silent. The
    /// preview path deliberately does *not* come through here: it moves what is
    /// drawn without changing the document (see
    /// [`ObservableState::doc_revision`]).
    fn committed_changed(&mut self) {
        self.doc_epoch += 1;
        self.doc_revision += 1;
    }

    fn commit(&mut self, kind: ActionKind) {
        let action = Action {
            id: self.next_action_id(),
            kind,
        };
        let ctx = &mut self.apply;
        self.timeline.push(action.clone(), ctx);
        // The committed document is what every in-flight preview is drawn over, so
        // every cached head built against the old one is now stale.
        self.committed_changed();
        if self.outbox_enabled {
            self.outbox.push(action);
        }
    }

    /// Mint the next layer id for this client (PEER_DESIGN.md §9).
    fn mint_layer(&mut self) -> LayerId {
        let id = LayerId::mint(self.actor, self.next_layer);
        self.next_layer += 1;
        id
    }

    /// Point the active layer at something that exists, preferring a paintable one:
    /// a matte may legitimately be selected, but someone who just lost the layer they
    /// were painting on wants to keep painting, not to land on the frame.
    /// Searches the whole tree, not just the root stack: removing a group takes
    /// carried layers with it, and the replacement may itself be carried by
    /// something (GROUP_DESIGN.md §2).
    fn repoint_active_layer(&mut self) {
        if self.document().contains_layer(self.session.active_layer) {
            return;
        }
        let (mut paintable, mut any) = (None, None);
        self.document().visit(&mut |l, _| {
            any = any.or(Some(l.id));
            if paintable.is_none() && l.is_paintable() {
                paintable = Some(l.id);
            }
        });
        if let Some(id) = paintable.or(any) {
            self.session.active_layer = id;
        }
    }

    /// The document's color space id (DESIGN.md §6.7).
    pub fn color_space(&self) -> ColorSpaceId {
        self.color_space.id()
    }

    /// Start a fresh, empty document in `color_space`, on `surface`.
    ///
    /// The **only** way to choose a colour space, and deliberately so: the channel
    /// layouts differ between spaces, so existing tiles cannot be reinterpreted and
    /// changing it can never preserve a document. Modelling it as a setter hid that
    /// — every caller was really asking for a new document (DESIGN.md §6.7).
    ///
    /// Takes `&mut self` rather than being an associated function because
    /// frontend-provided *resources* survive: imported brush assets, and the
    /// registered surface and environment bytes. Those belong to the app, not to
    /// the document, and re-fetching them on every New would be gratuitous.
    pub fn new_document(&mut self, color_space: ColorSpaceId, surface: SurfaceId) {
        self.initial_surface = surface;
        self.reset_document();
        self.rebuild_gpu_for(color_space);
        self.apply_document_surface();
    }

    /// The document's current surface (DESIGN.md §6.4). Change it with
    /// [`crate::command::DocCommand::SetSurface`].
    pub fn surface(&self) -> SurfaceId {
        self.document().surface
    }

    /// Whether `id` is ready to use — `Flat` always is; an image-backed surface
    /// is ready once its bytes have been [`register_surface`](Self::register_surface)ed.
    pub fn surface_loaded(&self, id: SurfaceId) -> bool {
        self.surface.is_loaded(id)
    }

    /// Provide (frontend-fetched) image bytes for a surface. If it's the one in
    /// use, the surface is rebuilt so the bytes take effect immediately.
    pub fn register_surface(&mut self, id: SurfaceId, png_bytes: Vec<u8>) {
        if self.surface.register(&self.gpu, id, png_bytes) {
            self.apply_surface();
        }
    }

    /// Bring the GPU-side surface in line with the document's, rebuilding it if the
    /// document moved to a different one — after a commit, an undo, a load, or a
    /// remote merge. A no-op when unchanged, which is the common case.
    ///
    /// There is deliberately no public `set_surface`: the surface is document state
    /// (DESIGN.md §6.4), so it changes by logging an action like anything else.
    fn apply_document_surface(&mut self) {
        let id = self.document().surface;
        if self.surface.set(&self.gpu, id) {
            self.apply_surface();
        }
    }

    /// Rebind the current surface in the media pass — the only thing that samples
    /// it. No pipeline or pool rebuild, no document reset.
    fn apply_surface(&mut self) {
        self.compositor_pipeline
            .set_surface(self.surface.current().clone());
    }

    /// The current lighting environment (DESIGN.md §6.3).
    pub fn environment(&self) -> EnvironmentId {
        self.environment.id()
    }

    /// Whether `id` is ready — `Neutral` always is; an HDR environment is ready once
    /// its bytes have been [`register_environment`](Self::register_environment)ed.
    pub fn environment_loaded(&self, id: EnvironmentId) -> bool {
        self.environment.is_loaded(id)
    }

    /// Provide (frontend-fetched) HDR bytes for an environment. If it's the one in
    /// use, it's rebuilt so the bytes take effect immediately.
    pub fn register_environment(&mut self, id: EnvironmentId, hdr_bytes: Vec<u8>) {
        if self.environment.register(&self.gpu, id, hdr_bytes) {
            self.apply_environment();
        }
    }

    /// Switch the lighting environment. A view setting, so this never touches the
    /// document — it just re-lights the canvas on the next render. Image
    /// environments fall back to the procedural neutral one until their bytes arrive.
    ///
    /// Private: reached through
    /// [`crate::command::ViewCommand::SetEnvironment`].
    fn set_environment(&mut self, id: EnvironmentId) {
        if self.environment.set(&self.gpu, id) {
            self.apply_environment();
        }
    }

    /// Rebind the current environment in the media pass.
    fn apply_environment(&mut self) {
        self.compositor_pipeline
            .set_environment(self.environment.current().clone());
    }

    /// Rebuild the GPU subsystems (pool/stroke/compositor) for `id`. Assumes the
    /// document is already empty (no tiles of the old format are referenced).
    fn rebuild_gpu_for(&mut self, id: ColorSpaceId) {
        let cs = id.make();
        let (pool, stroke, compositor_pipeline, compositor, transform, fill) =
            build_gpu(GpuBuild {
                gpu: &self.gpu,
                target_format: self.target_format,
                viewport: self.session.view.viewport,
                cs: &cs,
                surface: self.surface.current(),
                environment: self.environment.current(),
                selection: &self.apply.selection,
            });
        self.color_space = cs;
        self.apply.pool = pool;
        self.apply.stroke = stroke;
        self.apply.transform = transform;
        self.apply.fill = fill;
        self.compositor = compositor;
        self.compositor_pipeline = compositor_pipeline;
    }

    /// Reset to an empty document (one root layer) before a load/replay. Also
    /// leaves any shared session: the caller (UI/transport) tears down the
    /// network side; `join_collaboration` re-enables after its reset.
    fn reset_document(&mut self) {
        self.timeline = Box::new(LinearTimeline::new(
            DocState::with_layer(ROOT_LAYER).with_surface(self.initial_surface),
        ));
        self.doc_preview = None;
        self.live = None;
        self.heads.clear();
        self.peers.clear();
        self.committed_changed();
        self.clock = 0;
        self.next_layer = 1;
        self.session.cancel_stroke();
        self.session.active_layer = ROOT_LAYER;
        self.actor = ActorId::SOLO;
        self.outbox.clear();
        self.outbox_enabled = false;
    }

    /// Commit one already-built action onto the timeline (replays its GPU work).
    fn replay_one(&mut self, action: Action) {
        let ctx = &mut self.apply;
        self.timeline.push(action, ctx);
    }

    /// After loading, advance the id counters past everything in the log so new
    /// edits get fresh, monotonic ids.
    fn resync_counters(&mut self, actions: &[Action]) {
        let mut max_lamport = None;
        // Only *this* client's layer ids matter: the id space is partitioned by
        // author (PEER_DESIGN.md §9), so resuming past someone else's counter would
        // skip ids for no reason and, worse, hide the fact that they cannot collide.
        let mut max_ordinal = 0u64;
        for a in actions {
            max_lamport = Some(max_lamport.map_or(a.id.lamport, |m: u64| m.max(a.id.lamport)));
            let id = match &a.kind {
                ActionKind::AddLayer { id, .. } | ActionKind::AddMatte { id, .. } => Some(*id),
                _ => None,
            };
            if let Some(id) = id.filter(|id| id.minted_by(self.actor)) {
                max_ordinal = max_ordinal.max(id.ordinal());
            }
        }
        self.clock = max_lamport.map_or(0, |m| m + 1);
        self.next_layer = max_ordinal + 1;
    }

    fn next_action_id(&mut self) -> ActionId {
        let id = ActionId {
            lamport: self.clock,
            actor: self.actor,
        };
        self.clock += 1;
        id
    }

    /// Install (or, with `None`, drop) the stand-in document for an unlogged edit
    /// in flight — the shared tail of every `Preview*` command.
    /// The document as a `Transform { layer, affine }` commit would leave it,
    /// built through the **same renderer** the commit uses — which is what makes
    /// the preview lossless and exact: what is shown is what "Done" will produce
    /// (TRANSFORM_DESIGN.md §6). `None` when the layer cannot be transformed (a
    /// matte, absent) or the transform is rejected — the preview then simply
    /// shows the committed document, matching the commit's refusal.
    fn preview_transform(&self, layer: LayerId, affine: crate::geom::Affine2) -> Option<DocState> {
        let doc = self.timeline.current();
        let base = doc.layer(layer)?.tiles()?;
        let selection = doc.selection_of(self.actor);
        let (tiles, moved) =
            self.apply
                .transform
                .apply(&self.apply.pool, base, &selection, affine)?;
        Some(
            doc.map_layer(layer, |l| l.with_tiles(tiles))
                .with_selection(self.actor, moved),
        )
    }

    fn set_doc_preview(&mut self, preview: Option<DocState>) {
        self.doc_preview = preview;
        // The gesture previews are composited onto this, so moving it invalidates
        // every cached head exactly as a commit would.
        self.doc_epoch += 1;
        self.refresh_live();
    }

    /// The document as it should be *shown*: the committed state, or the unlogged
    /// edit in flight standing in for it, with every in-flight gesture drawn over
    /// the top.
    fn presented(&self) -> &DocState {
        self.live
            .as_ref()
            .or(self.doc_preview.as_ref())
            .unwrap_or_else(|| self.timeline.current())
    }

    /// The selection masks to outline, and whose each is (PEER_DESIGN.md §3).
    ///
    /// `DocState` holds a selection for every actor that ever made one, because
    /// replay needs them all; only the actors actually *here* are candidates. The log
    /// decides what exists, presence decides what could be shown — and
    /// `show_peer_selections` decides whether it is, since a second contour over the
    /// artwork is a preference rather than a fact about the drawing.
    fn visible_selections(&self) -> Vec<(crate::document::Selection, Option<[f32; 3]>)> {
        let doc = self.presented();
        let mut out = Vec::new();
        let mine = doc.selection_of(self.actor);
        if mine.is_active() {
            out.push((mine, None));
        }
        if self.session.show_peer_selections {
            for peer in self.peers.iter() {
                let theirs = doc.selection_of(peer.actor);
                if theirs.is_active() {
                    out.push((theirs, Some(peer.color)));
                }
            }
        }
        out
    }

    /// Rebuild [`Self::live`]: the committed document (or the frame drag standing in
    /// for it) with every in-flight gesture composited over it, this client's and
    /// every peer's, in ascending [`ActorId`] order (PEER_DESIGN.md §6).
    ///
    /// The order is fixed and derivable, so every client folds the same picture. Each
    /// stroke is rendered against the *committed* base and then overlaid tile-wise,
    /// rather than chained peer-over-peer: chaining would invalidate one peer's
    /// cached head on every move of the peers before it, which is precisely when two
    /// people are painting at once and the cache matters most. Where two live strokes
    /// touch the same tile in the same instant, the higher `ActorId` wins it — and a
    /// preview of concurrent strokes is provisional in any case, because the true
    /// result depends on the total order, which is not known until both commit.
    fn refresh_live(&mut self) {
        let gestures = self.live_gestures();
        if gestures.is_empty() {
            self.live = None;
            self.heads.clear();
            return;
        }
        let base = self
            .doc_preview
            .clone()
            .unwrap_or_else(|| self.timeline.current().clone());
        let mut out = base.clone();
        let mut heads = std::mem::take(&mut self.heads);
        for GestureView {
            actor,
            gesture,
            ordinal,
            frozen_spans: frozen,
        } in gestures
        {
            match gesture {
                LiveGesture::Selection(op) => {
                    // A marquee previews as the mask it will commit — the very same
                    // call `Select` makes (§6.8), so what is previewed is what lands.
                    let prev = base.selection_of(actor);
                    if let Some(selection) =
                        self.apply.selection.apply(&self.apply.pool, &prev, &op)
                    {
                        out = out.with_selection(actor, selection);
                    }
                    heads.remove(&actor);
                }
                // A fill previews as the paint it will lay, not as an outline of
                // where it would go — the same `FillRenderer::apply` the commit
                // makes, over the same base, so what is on screen mid-drag is
                // literally the result. Losslessly, and thrown away and redone on
                // each move rather than accumulated, which is what keeps dragging a
                // rectangle out from stacking a hundred glazes.
                LiveGesture::Fill { layer, op } => {
                    if let Some(tiles) = out.layer(layer).and_then(|l| l.tiles()).cloned() {
                        let gate = base.selection_of(actor);
                        if let Some(filled) =
                            self.apply.fill.apply(&self.apply.pool, &tiles, &gate, &op)
                        {
                            out = out.map_layer(layer, |l| l.with_tiles(filled));
                        }
                    }
                    heads.remove(&actor);
                }
                LiveGesture::Stroke(rec) => {
                    let head = heads.remove(&actor).filter(|h| {
                        h.epoch == self.doc_epoch && h.gesture == ordinal && h.spans <= frozen
                    });
                    let (head, tail_state) =
                        self.render_live_stroke(actor, &base, &rec, frozen, head, ordinal);
                    out = overlay_tiles(&out, rec.layer, &tail_state, &head.dirty);
                    heads.insert(actor, head);
                }
            }
        }
        self.heads = heads;
        self.live = Some(out);
    }

    /// Every gesture in flight, in ascending [`ActorId`] order.
    ///
    /// The local client's is *derived* from the session's fitter rather than kept in
    /// the roster: copying it there would make two sources of truth for the one thing
    /// the `preview == committed` invariant rests on. Merging the two here is what
    /// gives the uniform ordering without the duplication (PEER_DESIGN.md §4.1).
    fn live_gestures(&self) -> Vec<GestureView> {
        let mut out: Vec<GestureView> = Vec::new();
        out.extend(self.session.gesture_view(self.actor));
        out.extend(self.peers.iter().filter_map(Peer::gesture_view));
        out.sort_by_key(|g| g.actor);
        out
    }

    /// Advance one stroke's frozen head and render its live tail, returning the head
    /// to keep and the state the tail left behind.
    ///
    /// Uses the same entry point a commit does (`StrokeRenderer::render_range`), so
    /// the live preview and the `Action::apply` that replaces it draw the same pixels.
    fn render_live_stroke(
        &self,
        author: ActorId,
        base: &DocState,
        rec: &StrokeRecord,
        frozen: usize,
        head: Option<FrozenHead>,
        ordinal: u64,
    ) -> (FrozenHead, DocState) {
        // A tapered brush cannot freeze a span the taper has not finished with: the
        // taper is measured from the *ends* of the whole stroke, and the far end is
        // still under the pointer. Held back here rather than in the fitter because
        // it is a fact about the brush, not about the curve — the same control points
        // freeze at the same place for every other brush (see `taper_safe_frozen`).
        // An already-kept head is unaffected: a prefix this admitted once stays
        // admissible, so the clamp can only slow the head down, never invalidate it.
        let frozen = crate::gpu::stroke::taper_safe_frozen(rec, frozen);
        // Nothing cached, or the fit went backwards (a new stroke): start over from
        // the committed document, with a fresh (uncharged) brush.
        let mut head = head.unwrap_or_else(|| FrozenHead {
            spans: 0,
            dist: 0.0,
            tool: None,
            state: base.clone(),
            gesture: ordinal,
            epoch: self.doc_epoch,
            dirty: BTreeSet::new(),
        });
        if frozen > head.spans {
            head = self.advance_head(author, head, rec, frozen);
        }

        let all = crate::path::span_count(rec.path.len());
        let tail = StrokeSpans {
            range: head.spans..all,
            dist: head.dist,
        };
        // The diagnostic recolours only what this move actually redrew, so the seam
        // between tinted and untinted paint *is* the freezing boundary. Build-time
        // only (the `debug-unfrozen` feature): a shipping build has no code path that
        // paints the tail in anything but the stroke's own colour.
        #[cfg(feature = "debug-unfrozen")]
        let tinted = {
            let mut r = rec.clone();
            r.brush.color = DEBUG_UNFROZEN_COLOR;
            r
        };
        #[cfg(feature = "debug-unfrozen")]
        let tail_rec = &tinted;
        #[cfg(not(feature = "debug-unfrozen"))]
        let tail_rec = rec;
        // The tail reaches the end of the stroke, so the state it leaves the brush in
        // is handed to nobody — it is thrown away and rebuilt from the head on the next
        // move, which is exactly what makes the tail re-renderable. Its dirty tiles,
        // though, are part of what the fold has to overlay, so they join the head's.
        let (state, carry) =
            self.render_span_range(author, &head.state, tail_rec, tail, head.tool.as_ref());
        head.dirty.extend(carry.dirty);
        (head, state)
    }

    /// Dump the finished stroke's raw input as a pasteable Rust literal.
    ///
    /// A misfit seen in the app is otherwise unreproducible: the fit depends on the
    /// exact sequence of pointer reports — their spacing carries the pen's speed,
    /// which is what the density policy and the freezing both key off — and no
    /// synthetic curve stands in for a real hand. This turns one into a test case.
    fn log_debug_samples(&mut self) {
        if !cfg!(feature = "debug-unfrozen") || self.debug_samples.is_empty() {
            return;
        }
        // Positions *and* the pen channels. Position alone is not the input: pressure
        // sizes the brush and tilt steers it, both are fitted as their own least-squares
        // channels, and a capture without them cannot reproduce a fault in either.
        let mut lit = String::from("&[");
        for (i, s) in self.debug_samples.iter().enumerate() {
            if i > 0 {
                lit.push(',');
            }
            lit.push_str(&format!(
                "[{:.2},{:.2},{:.3},{:.3},{:.3}]",
                s.pos.x, s.pos.y, s.pressure, s.tilt.x, s.tilt.y
            ));
        }
        lit.push(']');
        tracing::info!(
            samples = self.debug_samples.len(),
            "raw stroke [x,y,pressure,tiltx,tilty]: {lit}"
        );
        self.debug_samples.clear();
    }

    /// Draw spans `head.spans..frozen` onto the frozen head, so the next move need
    /// not draw them again.
    fn advance_head(
        &self,
        author: ActorId,
        head: FrozenHead,
        rec: &StrokeRecord,
        frozen: usize,
    ) -> FrozenHead {
        let spans = StrokeSpans {
            range: head.spans..frozen,
            dist: head.dist,
        };
        // The renderer reports where it stopped rather than the caller recomputing it:
        // arc length is accumulated along the *emitted* polyline, and only the renderer
        // knows the budget it flattened at (a dynamics brush may have coarsened it), so
        // a second measurement here could hand the tail a distance the head never
        // reached — which `drain` and the colour-dynamics noise would both show.
        let (state, carry) =
            self.render_span_range(author, &head.state, rec, spans, head.tool.as_ref());
        let mut dirty = head.dirty;
        dirty.extend(carry.dirty);
        FrozenHead {
            spans: frozen,
            dist: carry.dist,
            // `None` from a range means "unchanged", not "reset" — a range with no
            // geometry runs nothing and leaves the brush as it found it.
            tool: carry.tool.or(head.tool),
            state,
            gesture: head.gesture,
            epoch: head.epoch,
            dirty,
        }
    }

    /// Render one span range of the in-flight stroke over an arbitrary base, resuming
    /// the brush from `tool` and reporting what the next range must resume from.
    ///
    /// Uses the same entry point a commit does (`StrokeRenderer::render_range`), so
    /// the live preview and the `Action::apply` that replaces it draw the same pixels.
    fn render_span_range(
        &self,
        author: ActorId,
        base: &DocState,
        rec: &StrokeRecord,
        spans: StrokeSpans,
        tool: Option<&crate::gpu::stroke::ToolState>,
    ) -> (DocState, crate::gpu::stroke::StrokeCarry) {
        let carry_only = |dist| crate::gpu::stroke::StrokeCarry {
            dist,
            tool: None,
            dirty: Vec::new(),
        };
        // A matte has no tile map, so it previews as nothing — matching the commit,
        // which refuses the stroke outright (FRAME_DESIGN.md §7). Preview and
        // commit agreeing is the §1.3 invariant, so the two refusals must line up.
        let Some(tiles_base) = base.layer(rec.layer).and_then(|l| l.tiles()) else {
            return (base.clone(), carry_only(spans.dist));
        };
        // The **author's** mask, exactly as the commit will read it — which is what
        // lets one client's live stroke be reproduced faithfully on another's screen
        // while their selections differ (PEER_DESIGN.md §3).
        let selection = base.selection_of(author);
        let (tiles, carry) = self.apply.stroke.render_range(
            crate::gpu::stroke::StrokeScene {
                pool: &self.apply.pool,
                assets: &self.apply.assets,
                base: tiles_base,
                selection: &selection,
            },
            rec,
            spans,
            tool,
        );
        (base.map_layer(rec.layer, |l| l.with_tiles(tiles)), carry)
    }
}

/// Copy `dirty`'s tiles from `src`'s `layer` into `out` — the overlay step of the
/// preview fold (PEER_DESIGN.md §6).
///
/// Only the named tiles move, which is what keeps two peers painting on one layer
/// from erasing each other's work back to the committed state: each contributes
/// exactly the tiles its own stroke touched.
fn overlay_tiles(
    out: &DocState,
    layer: LayerId,
    src: &DocState,
    dirty: &BTreeSet<TileCoord>,
) -> DocState {
    if dirty.is_empty() {
        return out.clone();
    }
    let Some(src_tiles) = src.layer(layer).and_then(|l| l.tiles()) else {
        return out.clone();
    };
    let Some(tiles) = out.layer(layer).and_then(|l| l.tiles()) else {
        return out.clone();
    };
    let mut tiles = tiles.clone();
    for coord in dirty {
        match src_tiles.get(coord) {
            Some(handle) => tiles = tiles.insert(*coord, handle.clone()),
            None => tiles = tiles.remove(coord),
        }
    }
    out.map_layer(layer, |l| l.with_tiles(tiles))
}

/// Build the GPU subsystems whose layout/shaders depend on the color space.
/// What the colour-space-dependent GPU subsystems are built from.
///
/// Grouped because they are always supplied together: the pool, stroke renderer and
/// compositor are torn down and rebuilt as a set whenever the colour space changes
/// (DESIGN.md §6.7), and `surface` / `environment` / `selection` are precisely the
/// pieces that *survive* that rebuild and have to be handed back in.
struct GpuBuild<'a> {
    gpu: &'a GpuContext,
    target_format: wgpu::TextureFormat,
    viewport: Extent2,
    cs: &'a Arc<dyn ColorSpace>,
    surface: &'a Surface,
    environment: &'a Environment,
    selection: &'a SelectionRenderer,
}

fn build_gpu(
    b: GpuBuild<'_>,
) -> (
    TilePool,
    StrokeRenderer,
    CompositorPipeline,
    Compositor,
    TransformRenderer,
    FillRenderer,
) {
    let GpuBuild {
        gpu,
        target_format,
        viewport,
        cs,
        surface,
        environment,
        selection,
    } = b;
    // Selection masks are pooled and recycled like paint (DESIGN.md §6.8), so their
    // format joins the pool's free lists.
    let pool = TilePool::new(
        gpu.clone(),
        [cs.color_format(), cs.aux_format(), MASK_FORMAT],
    );
    let stroke = StrokeRenderer::new(gpu, cs.clone(), selection.clone());
    let compositor_pipeline = CompositorPipeline::new(
        gpu,
        target_format,
        cs.as_ref(),
        surface.clone(),
        environment.clone(),
    );
    let compositor = Compositor::new(&compositor_pipeline, viewport);
    let transform = TransformRenderer::new(gpu, cs.as_ref(), selection.clone());
    let fill = FillRenderer::new(gpu, cs.clone(), selection.clone());
    (
        pool,
        stroke,
        compositor_pipeline,
        compositor,
        transform,
        fill,
    )
}

/// Convenience for tests/tools: build an engine on a headless device.
pub async fn headless_engine(
    target_format: wgpu::TextureFormat,
    viewport: Extent2,
) -> Result<Engine> {
    headless_engine_with(target_format, viewport, ColorSpaceId::Oklab).await
}

/// Headless engine in a chosen color space (DESIGN.md §6.7).
pub async fn headless_engine_with(
    target_format: wgpu::TextureFormat,
    viewport: Extent2,
    color_space: ColorSpaceId,
) -> Result<Engine> {
    let gpu = GpuContext::headless().await?;
    Ok(Engine::new_with_color_space(
        gpu,
        target_format,
        viewport,
        color_space,
    ))
}
