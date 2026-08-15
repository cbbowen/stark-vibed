//! The engine: owns the GPU, session, and timeline; turns commands into state
//! and renders the canvas (§7).
//!
//! For the MVP this exposes a synchronous [`Engine::process`]. The asynchronous
//! actor loop and reactive `ObservableState` channel (§7) wrap this
//! same core in a later step.
//!
//! # What is where
//!
//! [`Engine`] owns everything and is one type, but it is not one subject, so its
//! `impl` is split across this module's children along the seams it already had
//! section comments for. A child module can reach a private field of a struct its
//! parent defines, so this is a division of the *file* and not of the type: no field
//! moved, and nothing became `pub(crate)` to allow it.
//!
//! - here — the state itself, how a command reaches it ([`Engine::process`]), and
//!   the projection that comes back out ([`Engine::observe`]);
//! - `render` — the compositor's draw list, the screen frame and export (§6.3,
//!   §15.6);
//! - `live` — the preview fold and its per-stroke cache (§17.6, §6.2);
//! - `pick` — the eyedropper (§18.0.2);
//! - `collab` — the action and presence channels of a shared session (§12, §17);
//! - `file` — saving, opening, replay, and the resources a replay is run against
//!   (§8, §6.4, §6.6).

mod collab;
mod file;
mod live;
mod pick;
mod render;

use std::sync::Arc;

pub use collab::PresenceTick;
pub use pick::{PickOptions, PickSource};
pub use render::{Background, ExportPlan, ExportScale, Rendered};

use crate::Result;
use crate::assets::{AssetId, AssetStore};
use crate::colorspace::{ColorSpace, ColorSpaceId};
use crate::command::{DocCommand, GestureCommand, InputCommand, PeerCommand, ViewCommand};
use crate::document::{
    Action, ActionId, ActionKind, ActorId, ApplyCtx, BrushParams, CanvasBounds, DocState, Layer,
    LayerContent, LayerId, LinearTimeline, ShapeAction, Timeline, Tool,
};
use crate::error::EngineError;
use crate::geom::{Extent2, ViewTransform};
use crate::gpu::desc::Zeroes;
use crate::gpu::{
    BlendPass, Compositor, CompositorPipeline, Environment, EnvironmentId, FillRenderer,
    GpuContext, MergeRenderer, Registry, SelectionRenderer, StrokeRenderer, Surface, SurfaceId,
    TilePool, TransformRenderer,
};
use crate::peer::Peers;
use crate::session::ShapeResult;

/// The starting layer present in every new document.
const ROOT_LAYER: LayerId = LayerId(0);

/// Longest name that will be recorded, in `char`s. Not a taste limit but a bound
/// on the log: a layer's name is replicated to every peer and saved with the
/// document, and nothing about a text field stops a paste from being a megabyte.
/// Truncated by `char` rather than by byte so the cut can never land inside one.
const MAX_NAME: usize = 64;

/// The name to record, given what a frontend collected: surrounding whitespace
/// trimmed, length capped, and anything that comes out empty treated as *no name*
/// rather than as a name that happens to be blank.
///
/// One funnel for every source — the panel's field, a script, a peer's command —
/// so "a name is either absent or something you can read" is a property of the
/// model rather than a habit of the UI. The logged action carries the result, so
/// replay reproduces it without re-running these rules.
///
/// Shared by layers and drawing guides, which is the whole reason it is not called
/// `normalize_layer_name` any more: the two are named through different commands —
/// one logged, one view state — and the rule for what a name *is* should not be a
/// property of which command carried it.
///
/// Generic at both ends for that reason too, and only for that reason: the two
/// callers hold their names differently — a logged action carries a `String`,
/// because that is what goes on the wire, while a guide holds an `Arc<str>`, because
/// its list is re-projected at pointer rate — and neither difference is about what a
/// name is. `String: From<String>` is the identity, so the logged path still moves
/// its bytes rather than copying them.
fn normalize_name<T: From<String>>(name: Option<impl AsRef<str>>) -> Option<T> {
    let trimmed = name?;
    let capped: String = trimmed.as_ref().trim().chars().take(MAX_NAME).collect();
    (!capped.is_empty()).then(|| T::from(capped))
}

/// A layer's presentation properties, for the UI's layer panel (§11).
///
/// `Clone` but not `Copy` since it carries the name — an `Arc<str>` bump, so
/// cloning one is still a handful of instructions and `observe()` stays cheap.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerInfo {
    pub id: LayerId,
    pub blend: crate::document::BlendMode,
    /// Whether this layer clips to the paint beneath it (§14.4).
    pub clip: bool,
    pub opacity: f32,
    pub visible: bool,
    /// The layer carrying this one — i.e. the group it is in — or `None` for one
    /// in the document's own stack (§14.2).
    pub carrier: Option<LayerId>,
    /// How deeply nested it is: 0 in the document's own stack, one more per level
    /// of carrying. What a panel indents by.
    pub depth: usize,
    /// Whether this layer carries others, i.e. whether it is a **group**. A panel
    /// gives one of these a disclosure triangle; nothing else distinguishes it.
    pub is_group: bool,
    /// Whether anything composites beneath it, so its blend mode and its clip do
    /// anything at all (§14.4.3). False on exactly one row — the
    /// bottom of the document — where a mode is the identity and a clip would
    /// erase the layer, and where a panel therefore shows both controls inert.
    pub has_backdrop: bool,
    /// What the author called this layer, or `None` for one that has never been
    /// named — in which case it is for the frontend to describe it, since only the
    /// frontend knows how it presents a stack (see [`Layer::name`]).
    ///
    /// [`Layer::name`]: crate::document::Layer::name
    pub name: Option<std::sync::Arc<str>>,
    /// Set when this layer is a **matte** (§15.2) — a frame rather
    /// than paint. `None` for an ordinary paint layer.
    ///
    /// Projected so the frontend can label it, draw its handles, and show that the
    /// brush has nowhere to go while it is selected — all without reaching past
    /// `observe()` into `DocState`.
    pub matte: Option<MatteInfo>,
    /// Set when this layer is a **filter** (§21) — a function of what is composited
    /// beneath it rather than content of its own. `None` for anything else.
    ///
    /// The whole filter, not a name for it: the filter bar's sliders read their
    /// current values off this and send the adjusted filter straight back, so a
    /// filter that grows a knob costs the projection nothing (§21.6).
    pub filter: Option<crate::document::Filter>,
    /// Whether the compositor would draw anything beneath this layer **within its
    /// own stack** — which is exactly what a filter layer rewrites (§21.2).
    ///
    /// Not the same question as [`has_backdrop`](Self::has_backdrop), in two ways
    /// that both matter to the one consumer (the filter bar's "nothing below it"
    /// notice). It counts the **carrier's own content**: a group's base composites
    /// at the bottom of the group, so a filter carried onto a painted layer — the
    /// "filter just this layer" gesture — has that paint beneath it even as the
    /// first carried row. And it counts only what would actually **draw**: a
    /// hidden, fully transparent or never-painted layer is culled from the draw
    /// list (`render.rs`), so a filter above nothing but those reaches nothing,
    /// whatever the row order says. `has_backdrop` stays positional because blend
    /// and clip are defined against position (§14.4.3); this one follows the
    /// renderer because a filter's reach *is* the renderer's accumulator.
    pub has_underlay: bool,
    /// The layer this one would **merge down** onto, or `None` when there is no merge
    /// here that leaves the document looking the same (§14.11).
    ///
    /// Projected as the destination rather than as a `bool` because the panel says
    /// what the click will do — the row it folds into — and asking the engine twice
    /// for one answer is how a tooltip ends up describing a different merge from the
    /// one the button performs.
    ///
    /// Unlike every other field here this is a statement about a *pair* of layers, and
    /// it is the only control in the panel that is absent rather than merely inert
    /// when the answer is no: a merge that would change the picture is not a weaker
    /// merge, it is a different edit, and offering it greyed out would suggest the
    /// document is what stands in the way.
    pub merge_down: Option<LayerId>,
}

impl LayerInfo {
    /// Whether a stroke aimed at this layer would draw anything. Neither a matte nor
    /// a filter has a tile map, so selecting one is legal but painting on it does
    /// nothing (§15.7, §21.4).
    pub fn is_paintable(&self) -> bool {
        self.matte.is_none() && self.filter.is_none()
    }
}

/// A matte layer's geometry and fill, for the frame chrome (§15.7).
#[derive(Clone, Debug, PartialEq)]
pub struct MatteInfo {
    /// The rect the region is defined against, in canvas px. For a frame this is
    /// the *hole* — the piece — which is what the handles resize and what export
    /// frames against (§15.6). `None` for a region defined against no rect
    /// ([`MatteRegion::Everything`]): the handle box, the aspect readout and the
    /// export frame all stand down rather than invent one.
    pub rect: Option<(crate::geom::Vec2, crate::geom::Vec2)>,
    /// The paint the region wears — flat, or a ramp (§15.4, §22.4).
    pub paint: crate::document::MattePaint,
}

impl MatteInfo {
    /// Width × height of the rect, where there is one.
    pub fn dims(&self) -> Option<(f32, f32)> {
        let (min, max) = self.rect?;
        Some((max.x - min.x, max.y - min.y))
    }
}

/// A cheap, UI-facing projection of engine state (§7). Published to
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
    /// Whether a selection is masking the canvas (§6.8) — drives the
    /// "Deselect"/"Invert" affordances and the selection indicator.
    pub has_selection: bool,
    /// A conservative canvas-space bounding box of this client's selected
    /// coverage, or `None` when the selection is unbounded or unknown
    /// ([`Selection::hull`](crate::document::Selection::hull)). What the
    /// transform chrome hangs its handles on; committed-only, like
    /// `has_selection`.
    pub selection_hull: Option<(crate::geom::Vec2, crate::geom::Vec2)>,
    /// What the next shape gesture will do with the region it encloses — combine
    /// it into the selection one of four ways, or fill it (§18.0.4).
    pub shape_action: ShapeAction,
    /// Edge softness (canvas px) the next shape gesture will apply.
    pub selection_feather: f32,
    /// How strongly the next shape gesture's coverage will land, `0..=1` (§6.8).
    pub shape_opacity: f32,
    /// Whether collaborators' selection outlines are drawn (§17.3).
    pub show_peer_selections: bool,
    /// The drawing guides (§20.5) — projected so the Drawing Guides panel and
    /// the edit bar read the engine's list rather than a shadow of their own.
    pub guides: Vec<crate::guides::PerspectiveGuide>,

    // --- view settings (per-client, never historized) ---------------------
    //
    // Projected here for the same reason as `tool` and `brush`: a frontend that
    // has to read these back off the engine ends up keeping its own copy, and a
    // copy seeded from `Default` rather than from the engine goes stale the
    // moment anything else changes them (§4).
    /// Media/lighting parameters of the painterly pass (§6.3).
    pub media: crate::gpu::MediaParams,
    /// The HDR lighting environment in use (§6.3).
    pub environment: EnvironmentId,

    // --- document properties fixed at creation ----------------------------
    /// The document's color space. Immutable for the document's life — changing
    /// it means starting a new document ([`Engine::new_with_color_space`]).
    pub color_space: ColorSpaceId,
    /// The physical canvas surface (§6.4).
    pub surface: SurfaceId,
    /// The canvas substrate color, straight sRGB (§15.5). Document
    /// state, not a view setting — projected here so the frontend shows what the
    /// document says rather than a copy of its own that goes stale.
    pub background: [f32; 3],
}

pub struct Engine {
    gpu: GpuContext,
    target_format: wgpu::TextureFormat,
    color_space: Arc<dyn ColorSpace>,
    /// The GPU subsystems an action needs in order to apply itself — the tile
    /// pool, the stroke renderer, the asset store and the selection rasterizer —
    /// held as the `history::Action::Context` (§5).
    ///
    /// Stored rather than built per call. `history`'s `Context` is an owned
    /// associated type, so there is nothing to hand it a borrow of; this used to be
    /// rebuilt by cloning all four on *every* commit, undo, redo and remote merge —
    /// tens of `Arc` bumps plus a `HashMap` allocation each time, for a value that
    /// only changes when the color space is rebuilt.
    ///
    /// They live only here: the engine reaches them through `self.apply` too, so
    /// there is one copy rather than the engine's plus the context's.
    /// `selection` is color-space independent (a mask is one coverage channel
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
    /// The surface the action log starts from, written to `CanvasMeta` and used to
    /// seed the document. Plays the same role as `CanvasMeta::color_space`: it
    /// describes the empty document that the log is replayed onto, and is not
    /// itself a logged change.
    initial_surface: SurfaceId,
    /// The HDR lighting environment (image-based lighting) and its registered
    /// bytes. A *view* setting — not historized, color-space-independent — so it
    /// survives rebuilds and switching it never touches the document (§6.3).
    environment: Registry<EnvironmentId>,
    timeline: Box<dyn Timeline>,
    session: crate::session::Session,
    /// Everyone else in the session (§17.4). Empty when solo.
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
    /// What is being *shown* over the committed document, and the caches that make
    /// showing it affordable: the unlogged drag in flight, the fold of every
    /// in-flight gesture, the settled head of each live stroke, and the epoch that
    /// says when a head has gone stale (§17.6).
    ///
    /// One field rather than four, because they are one thing with one invariant.
    /// As four they could be — and were — moved independently: three of the epoch's
    /// five bumps were written out at their call sites, and `DocCommand::Seek`
    /// dropped the drag preview without bumping it at all when the seek turned out
    /// to be a no-op. Now the slot cannot move without the epoch moving with it.
    preview: live::Preview,
    /// Bumped whenever the **committed** document changes — a commit, an undo, a
    /// merged remote action, a load. Projected as
    /// [`ObservableState::doc_revision`], where it is what a frontend showing a
    /// rendered stand-in for the document (the navigator's miniature) watches to
    /// know when that render is out of date.
    ///
    /// Strictly narrower than the preview's epoch, which an unlogged drag also
    /// bumps: a drag moves at pointer rate and is *not* a change to the document,
    /// so a watcher keyed on the epoch would re-render for every sample of one. The
    /// two advance together through [`Engine::committed_changed`].
    doc_revision: u64,
    /// Raw pointer reports of the in-flight stroke, dumped on release under the
    /// `debug-unfrozen` feature so a misfit stroke can be replayed as a test.
    debug_samples: Vec<crate::command::InputSample>,
    actor: ActorId,
    clock: u64,
    next_layer: u64,
    /// Locally-committed actions awaiting broadcast to peers (§12.4).
    /// Only populated in a shared session (`outbox_enabled`), and drained by the
    /// transport via [`Engine::take_outbox`]; solo mode never accumulates.
    outbox: Vec<Action>,
    outbox_enabled: bool,
}

impl Engine {
    /// Build an engine that presents to `target_format` (a surface format, or a
    /// test target), in the default Oklab color space. Takes wgpu handles from
    /// the frontend (CLAUDE.md).
    pub fn new(gpu: GpuContext, target_format: wgpu::TextureFormat, viewport: Extent2) -> Self {
        // Oklab is in every build by construction — it is the space with no optional
        // dependency behind it — so the only fallible case cannot arise here.
        Self::new_with_color_space(gpu, target_format, viewport, ColorSpaceId::Oklab)
            .expect("Oklab is unconditional")
    }

    /// Build an engine in a chosen color space (§6.7).
    ///
    /// Fails with [`EngineError::UnsupportedColorSpace`] if this build does not carry
    /// the space — today only `Mixbox` without the `mixbox` feature. A frontend that
    /// builds its picker from
    /// [`ColorSpaceId::all_available`](crate::colorspace::ColorSpaceId::all_available)
    /// never sees it.
    pub fn new_with_color_space(
        gpu: GpuContext,
        target_format: wgpu::TextureFormat,
        viewport: Extent2,
        color_space: ColorSpaceId,
    ) -> Result<Self> {
        let color_space = color_space
            .make()
            .ok_or(EngineError::UnsupportedColorSpace(color_space))?;
        // The registry starts on the builtin flat ground — it is all that can be
        // built before any bytes exist, and it is also what a fresh document is on
        // (`DEFAULT_SURFACE`). The two agree now, where they used to have to be
        // reconciled: a ground is named by the hash of its height map (§6.4), so an
        // engine with no bytes has exactly one ground it can truthfully name, and a
        // frontend that wants another opens a document on it.
        let surface = Registry::<SurfaceId>::new(&gpu, SurfaceId::default());
        // Lighting starts on the procedural neutral environment; image HDRs are
        // registered later by the frontend (§6.3).
        let environment = Registry::<EnvironmentId>::new(&gpu, EnvironmentId::default());
        let selection = SelectionRenderer::new(&gpu);
        let gpu_for_ctx = gpu.clone();
        let (pool, stroke, compositor_pipeline, compositor, transform, fill, merge) =
            build_gpu(GpuBuild {
                gpu: &gpu,
                target_format,
                viewport,
                cs: &color_space,
                surface: &surface.current(),
                environment: &environment.current(),
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
                merge,
                gpu: gpu_for_ctx,
                surfaces: surface,
            },
            compositor,
            compositor_pipeline,
            initial_surface,
            environment,
            timeline,
            session,
            peers: Peers::new(),
            now: 0.0,
            preview: Default::default(),
            doc_revision: 0,
            debug_samples: Vec::new(),
            actor: ActorId::SOLO,
            clock: 0,
            next_layer: 1,
            outbox: Vec::new(),
            outbox_enabled: false,
        };
        // Point the ground registry at the document's ground. A no-op for a fresh
        // document (both are `Flat`), and not for one seeded by `new_document`,
        // where it parks the registry on the id so the ground actually renders.
        engine.apply_document_surface();
        Ok(engine)
    }

    /// A second engine on `donor`'s device, **sharing** everything expensive and
    /// immutable — the compiled pipelines (stroke, compositing, selection,
    /// transform, fill, merge, the blend pass and its pigment LUT), the tile
    /// allocator, the content-addressed brush assets, and the decoded ground and
    /// environment caches — around a fresh document of its own.
    ///
    /// This is what a *preview* engine is (§11): the brush editor's test canvas and
    /// a preset thumbnail both paint strokes that must render exactly as the main
    /// canvas would, which is an argument for sharing the machinery, not just an
    /// economy. Building one used to mean recompiling ~19 shaders and ~30 pipelines
    /// and re-decoding every image the app had already decoded once; now it is a
    /// document, a compositor's attachments, and a fistful of `Arc` bumps.
    ///
    /// What is shared is exactly what cannot disagree: the shared pieces are either
    /// immutable (pipelines), content-addressed (assets, the ground/environment
    /// byte-and-build caches), or an allocator (the tile pool). Everything an engine
    /// can *set* stays per-engine — the document, the session view, and the three
    /// compositor view settings, which start mirroring the donor's current look
    /// (ground, lighting, media parameters) and move independently from there.
    ///
    /// The document opens on the donor's current ground, so a preview needs no
    /// `SetSurface` step — and no ground bytes handed across, which is the point.
    ///
    /// Divergence after construction is safe but not tracked: a
    /// [`new_document`](Self::new_document) that changes *this* engine's color
    /// space rebuilds it an unshared set (`rebuild_gpu_for`), and the donor doing
    /// the same simply stops feeding the shared caches this engine keeps using.
    pub fn new_sharing(donor: &Engine, viewport: Extent2) -> Self {
        let gpu = donor.gpu.clone();
        let initial_surface = donor.document().surface;
        let timeline: Box<dyn Timeline> = Box::new(LinearTimeline::new(
            DocState::with_layer(ROOT_LAYER).with_surface(initial_surface),
        ));
        let session = crate::session::Session::new(ViewTransform::identity(viewport), ROOT_LAYER);
        let compositor_pipeline = CompositorPipeline::sharing(
            donor.compositor_pipeline.passes(),
            donor.apply.surfaces.current(),
            donor.environment.current(),
            donor.compositor_pipeline.media(),
        );
        let compositor = Compositor::new(&compositor_pipeline, viewport);
        let mut engine = Self {
            target_format: donor.target_format,
            color_space: donor.color_space.clone(),
            apply: ApplyCtx {
                pool: donor.apply.pool.clone(),
                stroke: donor.apply.stroke.clone(),
                assets: donor.apply.assets.clone(),
                selection: donor.apply.selection.clone(),
                transform: donor.apply.transform.clone(),
                fill: donor.apply.fill.clone(),
                merge: donor.apply.merge.clone(),
                gpu: gpu.clone(),
                surfaces: donor.apply.surfaces.clone(),
            },
            gpu,
            compositor,
            compositor_pipeline,
            initial_surface,
            environment: donor.environment.clone(),
            timeline,
            session,
            peers: Peers::new(),
            now: 0.0,
            preview: Default::default(),
            doc_revision: 0,
            debug_samples: Vec::new(),
            actor: ActorId::SOLO,
            clock: 0,
            next_layer: 1,
            outbox: Vec::new(),
            outbox_enabled: false,
        };
        // Park the sibling registry on the document's ground — a no-op here, since
        // both were just seeded from the donor's current one, but stated so this
        // constructor upholds the invariant the same way `new_with_color_space` does.
        engine.apply_document_surface();
        engine
    }

    /// Apply one input command (§4).
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
    /// (§6.8): the selection tools build an op where the brush builds a
    /// stroke, and both preview through the same `preview` DocState.
    fn process_gesture(&mut self, command: GestureCommand) {
        match command {
            GestureCommand::Start {
                tool,
                sample,
                tolerance,
                rope,
            } => {
                if tool.is_selection() {
                    // A marquee or lasso fits no curve, so it has no use for the
                    // tolerance (or the rope); its own decimation is a mask-cost
                    // knob (§6.8).
                    self.session.start_selection(tool, sample.pos);
                } else {
                    let seed = self.clock;
                    self.session
                        .start_stroke(tool, sample, seed, tolerance, rope);
                    // Gated like the `To` arm below, which it was not: the capture is
                    // a diagnostic, so a shipping build must not keep a sample it has
                    // no path to ever print.
                    if cfg!(feature = "debug-unfrozen") {
                        self.debug_samples.clear();
                        self.debug_samples.push(sample);
                    }
                }
                self.mark_live_stale();
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
                self.mark_live_stale();
            }
            // A held pointer: snap the stroke to the shape it resembles (§6.9). Nothing
            // is committed and nothing is decided about the gesture's end — a snap
            // changes what the *same* drag builds, and the release still commits one
            // stroke either way.
            GestureCommand::Hold => {
                if self.session.assist_stroke() {
                    self.mark_live_stale();
                }
            }
            // The one edge that produces document state.
            GestureCommand::End => {
                if self.session.is_selecting() {
                    // One gesture, two things it can commit — which one was decided
                    // when the drag started (§18.0.4).
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
                self.mark_live_stale();
            }
            GestureCommand::Cancel => {
                self.session.cancel_stroke();
                self.mark_live_stale();
            }
        }
    }

    /// Per-client state that is published rather than logged (§17.7).
    /// Nothing here enters the history or the save file; it rides the presence
    /// channel so collaborators can see where this client is working.
    fn process_peer(&mut self, command: PeerCommand) {
        match command {
            // Any existing layer, including a matte. `active_layer` is *the
            // selected layer*, not "a paint target" — a frame is selected the same
            // way a paint layer is, which is what lets the frontend have one
            // selection concept instead of two (§15.7). A stroke aimed
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
        self.mark_live_stale();
    }

    fn process_doc_inner(&mut self, command: DocCommand) {
        match command {
            // Shared sessions log undo as an action peers can order (§5.4, §12.3);
            // solo falls back to navigation. Redo is an `Undo` of an `Undo`, which
            // is why the two differ only in which pair of timeline methods they
            // name — see [`Self::navigate`].
            DocCommand::Undo => {
                self.navigate(|t| t.undo_as_action(), |t, ctx| t.undo(ctx));
            }
            DocCommand::Redo => {
                self.navigate(|t| t.redo_as_action(), |t, ctx| t.redo(ctx));
            }
            DocCommand::Seek(to) => {
                self.preview.set_doc(None);
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
            DocCommand::Fill { layer, op } => {
                // The commit supersedes whatever the gradient-fill mode was
                // previewing (§22.4) — `Transform`'s reason, one arm up.
                self.preview.set_doc(None);
                self.commit(ActionKind::Fill { layer, op });
            }
            DocCommand::Transform { layer, map } => {
                // The commit supersedes whatever the gesture was previewing, for
                // the same reason `SetMatteRect` drops its preview.
                self.preview.set_doc(None);
                // A degenerate or non-finite map would be rejected by `apply`
                // anyway (deterministically — §16.1); refusing it
                // here as well keeps a knowably-dead action out of the log.
                // Each family goes to its own action kind — the wire format
                // never carries the routing enum, only the map it named.
                if map.usable() {
                    use crate::document::TransformMap;
                    self.commit(match map {
                        TransformMap::Affine(affine) => ActionKind::Transform { layer, affine },
                        TransformMap::Perspective(map) => {
                            ActionKind::TransformPerspective { layer, map }
                        }
                        TransformMap::Warp(map) => ActionKind::TransformWarp { layer, map },
                    });
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
                // (§14.8), and arming an id no layer has would leave
                // the next stroke with nowhere to go.
                if self.document().contains_layer(id) {
                    self.session.active_layer = id;
                }
            }
            DocCommand::AddMatte {
                carrier,
                at,
                region,
                paint,
            } => {
                let id = self.mint_layer();
                self.commit(ActionKind::AddMatte {
                    id,
                    carrier,
                    at,
                    region,
                    paint,
                });
                // Deliberately *not* made the active layer, unlike `AddLayer`: a
                // matte has no tile map, so painting on it is refused
                // (§15.7) and arming it as the target would just
                // swallow the user's next stroke.
            }
            DocCommand::AddFilter {
                carrier,
                above,
                filter,
            } => {
                let id = self.mint_layer();
                self.commit(ActionKind::AddFilter {
                    id,
                    carrier,
                    above,
                    filter: filter.sanitized(),
                });
                // Deliberately *not* made the active layer, for the reason
                // `AddMatte` is not: a filter has no tile map, so arming it as the
                // paint target would swallow the next stroke (§21.4). The frontend
                // selects it, which is what raises its bar.
            }
            DocCommand::SetFilter(id, filter) => {
                // Drops the preview whether or not the commit below happens, for the
                // reason `SetMatteColor` drops it: a drag that settles on the value
                // it opened at must still supersede what it was showing.
                self.preview.set_doc(None);
                // Normalized here, where the action is minted, so replay puts back
                // what was applied rather than re-deriving it from rules that may
                // have moved (§21.5) — the same funnel `SetLayerName` goes through.
                let filter = filter.sanitized();
                // Refused when it would change nothing, as `SetLayerOpacity` and
                // `SetMatteColor` are, and asked of the layer's *content* for the
                // same reason: only a filter layer has a filter to compare. A
                // non-filter layer still commits an action `apply` will no-op,
                // rather than growing a second rule about what a filter is somewhere
                // `apply` cannot see.
                let unchanged = self
                    .document()
                    .layer(id)
                    .and_then(|l| l.filter())
                    .is_some_and(|current| current == filter);
                if !unchanged {
                    self.commit(ActionKind::SetFilter(id, filter));
                }
            }
            DocCommand::SetMatteRect(id, min, max) => {
                // The committed rect supersedes whatever the drag was previewing;
                // leaving the preview up would pin the canvas to the last dragged
                // value and shadow every later edit.
                self.preview.set_doc(None);
                self.commit(ActionKind::SetMatteRect(id, min, max));
            }
            DocCommand::SetMattePaint(id, paint) => {
                // Drops the preview whether or not the commit below happens, for the
                // reason `SetMatteRect` drops it above: a pick that settles on the
                // paint it opened on must still supersede what it was showing.
                self.preview.set_doc(None);
                // Refused when it would change nothing, as `SetLayerOpacity` and
                // `SetLayerName` are — and asked of the layer's *content*, since a
                // matte is the only thing that has paint to compare (§15.2). A
                // paint layer still commits, which is what it did before: that action
                // is inert rather than duplicated, and refusing it here would be a
                // second rule about what a matte is, kept somewhere `apply` cannot see.
                let unchanged = matches!(
                    self.document().layer(id).map(|l| &l.content),
                    Some(LayerContent::Matte { paint: current, .. }) if *current == paint
                );
                if !unchanged {
                    self.commit(ActionKind::SetMattePaint(id, paint));
                }
            }
            DocCommand::SetBackground(rgb) => {
                // The committed color supersedes whatever the drag was previewing,
                // for the same reason `SetMatteRect` drops it above.
                self.preview.set_doc(None);
                self.commit(ActionKind::SetBackground(rgb));
            }
            DocCommand::DuplicateLayer(source) => {
                // One minted id per layer of the subtree, paired with the layer it
                // copies, in composite order — the map the action carries
                // (§14.8). Collected off the document before any minting, because
                // `mint_layer` needs `&mut self` and the walk is borrowing it.
                let mut sources = Vec::new();
                if let Some(l) = self.document().layer(source) {
                    l.visit(0, &mut |l, _| sources.push(l.id));
                }
                if !sources.is_empty() {
                    let ids: Vec<_> = sources
                        .into_iter()
                        .map(|src| (src, self.mint_layer()))
                        .collect();
                    let copy = ids[0].1;
                    self.commit(ActionKind::DuplicateLayer { ids });
                    // The copy is what you go on to work on — but only if it
                    // landed and can take a stroke. A matte cannot (§15.7), and
                    // arming one would swallow the next stroke, which is the same
                    // reason `AddMatte` arms nothing.
                    if self
                        .document()
                        .layer(copy)
                        .is_some_and(|l| l.is_paintable())
                    {
                        self.session.active_layer = copy;
                    }
                }
            }
            DocCommand::RemoveLayer(id) => {
                self.commit(ActionKind::RemoveLayer(id));
                self.repoint_active_layer();
            }
            DocCommand::MergeLayerDown(id) => {
                // Asked here rather than only inside `apply`, so a merge that cannot
                // preserve the document's appearance never reaches the log at all —
                // the same argument `Transform` makes about a degenerate map. `apply`
                // asks again anyway, because a peer's action arrives without passing
                // through here (§14.11).
                if let Some(plan) = crate::document::merge::plan(self.document(), id) {
                    self.commit(ActionKind::MergeLayerDown {
                        source: plan.source,
                        dest: plan.dest,
                    });
                    // The merged layer is where the work now is, so the brush follows
                    // it — and the layer that was selected may be the one that has just
                    // stopped existing, which `repoint_active_layer` alone would answer
                    // by picking whatever is nearest rather than by picking the paint.
                    if self.session.active_layer == id
                        && self
                            .document()
                            .layer(plan.dest)
                            .is_some_and(|l| l.is_paintable())
                    {
                        self.session.active_layer = plan.dest;
                    }
                    self.repoint_active_layer();
                }
            }
            DocCommand::SetLayerBlend(id, blend) => {
                // Drops the preview whether or not the commit below happens, for
                // `SetFilter`'s reason: a parameter drag that settles on the value it
                // opened at must still supersede what it was showing.
                self.preview.set_doc(None);
                // Normalized here, where the action is minted, so replay puts back
                // what was applied rather than re-deriving it from rules that may
                // have moved — the same funnel `SetFilter` goes through, and the
                // reason the comparison below is against the sanitized value rather
                // than the raw one.
                let blend = blend.sanitized();
                // Refused when it would change nothing, as `SetLayerOpacity` is: a
                // Bend drag that travelled out and came back must log nothing, and
                // without this a slider reporting 0.6000001 would spend an undo step
                // on a document that already reads that way.
                let unchanged = self
                    .document()
                    .layer(id)
                    .is_some_and(|l| l.composite.blend == blend);
                if !unchanged {
                    self.commit(ActionKind::SetLayerBlend(id, blend));
                }
            }
            DocCommand::SetLayerClip(id, clip) => self.commit(ActionKind::SetLayerClip(id, clip)),
            DocCommand::SetLayerOpacity(id, opacity) => {
                // The committed opacity supersedes whatever the drag was previewing,
                // for the same reason `SetMatteRect` drops its preview above — and
                // it is dropped whether or not the commit below happens, so a drag
                // that ends where it started leaves nothing pinned.
                self.preview.set_doc(None);
                // Clamped here rather than compared raw, because that is what the
                // action would store (`DocState::set_layer_opacity`): without it a
                // slider that reports 1.0000001 would log a step that changes
                // nothing when reached. The same argument as `SetLayerName`'s, and
                // the same case makes it — a drag that returns to the value it
                // started on is not an edit.
                let opacity = opacity.clamp(0.0, 1.0);
                if self
                    .document()
                    .layer(id)
                    .is_none_or(|l| l.composite.opacity != opacity)
                {
                    self.commit(ActionKind::SetLayerOpacity(id, opacity));
                }
            }
            DocCommand::SetLayerVisible(id, visible) => {
                self.commit(ActionKind::SetLayerVisible(id, visible))
            }
            DocCommand::SetLayerName(id, name) => {
                let name = normalize_name(name);
                // A rename to the name it already has is not an edit, and logging it
                // would spend an undo step that appears to do nothing when reached.
                // Commit-on-blur makes this the common case: leaving a field you only
                // looked at must cost nothing.
                if self.timeline.current().layer_name(id) != name.as_deref() {
                    self.commit(ActionKind::SetLayerName(id, name));
                }
            }
            DocCommand::MoveLayer { id, carrier, at } => {
                self.commit(ActionKind::MoveLayer { id, carrier, at })
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
                self.mark_live_stale();
            }
            ViewCommand::SetBrush(brush) => {
                self.session.brush = brush;
                self.mark_live_stale();
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
            ViewCommand::Pinch {
                anchor,
                to,
                scale,
                turn,
            } => self.session.view.pinch(anchor, to, scale, turn),
            ViewCommand::Resize(viewport) => {
                self.session.view.viewport = viewport;
            }
            ViewCommand::SetShapeAction(action) => self.session.shape_action = action,
            ViewCommand::SetSelectionFeather(feather) => {
                self.session.selection_feather = feather.max(0.0)
            }
            ViewCommand::SetShapeOpacity(opacity) => {
                self.session.shape_opacity = opacity.clamp(0.0, 1.0)
            }
            ViewCommand::SetShowPeerSelections(show) => self.session.show_peer_selections = show,
            ViewCommand::SetGuides(mut guides) => {
                // The whole list arrives on every edit (§20.5), so the names are
                // normalized here rather than in a rename command of their own —
                // there is no path to a guide's name that does not come through
                // this arm, which is what makes the guarantee structural.
                for g in &mut guides {
                    g.name = normalize_name(g.name.take());
                }
                self.session.guides = guides;
            }
            ViewCommand::PreviewMatteRect(drag) => {
                let preview =
                    drag.map(|(id, min, max)| self.timeline.current().set_matte_rect(id, min, max));
                self.set_doc_preview(preview);
            }
            ViewCommand::PreviewBackground(rgb) => {
                let preview = rgb.map(|rgb| self.timeline.current().with_background(rgb));
                self.set_doc_preview(preview);
            }
            ViewCommand::PreviewMattePaint(pick) => {
                let preview =
                    pick.map(|(id, paint)| self.timeline.current().set_matte_paint(id, paint));
                self.set_doc_preview(preview);
            }
            ViewCommand::PreviewLayerOpacity(set) => {
                let preview =
                    set.map(|(id, opacity)| self.timeline.current().set_layer_opacity(id, opacity));
                self.set_doc_preview(preview);
            }
            ViewCommand::PreviewFilter(set) => {
                // `set_filter` sanitizes on the way into state (§21.5), so the
                // preview cannot show numbers the commit would then refuse.
                let preview =
                    set.map(|(id, filter)| self.timeline.current().set_filter(id, filter));
                self.set_doc_preview(preview);
            }
            ViewCommand::PreviewLayerBlend(set) => {
                // `set_layer_blend` sanitizes on the way into state too, for the
                // reason above: what the drag shows is what the commit would store.
                let preview =
                    set.map(|(id, blend)| self.timeline.current().set_layer_blend(id, blend));
                self.set_doc_preview(preview);
            }
            ViewCommand::PreviewTransform(t) => {
                let preview = t.and_then(|(layer, map)| self.preview_transform(layer, &map));
                self.set_doc_preview(preview);
            }
            ViewCommand::PreviewFill(f) => {
                let preview = f.and_then(|(layer, op)| self.preview_fill(layer, &op));
                self.set_doc_preview(preview);
            }
            ViewCommand::SetMediaParams(params) => self.compositor_pipeline.set_media(params),
            ViewCommand::SetEnvironment(id) => self.set_environment(id),
        }
    }

    /// Replay a whole recorded stroke as a single commit: start → samples →
    /// end, without the per-sample staleness marks. Interactive samples go
    /// through `GestureCommand::To`, whose marks a frame's `flush_live` services
    /// by rendering the in-flight tail — right for drawing (the user must see
    /// each frame's moves), pointless across a replay where nothing is presented
    /// in between. This renders the stroke exactly once, at commit. Used by the
    /// brush editor's test-stroke replay.
    pub fn replay_stroke(&mut self, tool: Tool, samples: &[crate::command::InputSample]) {
        self.replay_stroke_seeded(tool, samples, self.clock, 0.0);
    }

    /// [`Engine::replay_stroke`] with an explicit jitter `seed` instead of the
    /// Lamport clock. Replaying the same samples repeatedly advances the clock
    /// (each replay commits), so the seed — and with it the color dynamics and
    /// dither — changes on every replay. A caller re-rendering *one* stroke to
    /// show the effect of a brush change (the brush editor's preview) wants the
    /// jitter held fixed, so only the edited parameter moves.
    /// `rope` is the §6.11 smoothing string, and it is a parameter here — where
    /// [`Engine::replay_stroke`] pins it to zero — because the brush editor's
    /// preview replays a *recorded hand* (the user's own test stroke) and has to
    /// show what the smoothing slider beside it would do to that hand.
    pub fn replay_stroke_seeded(
        &mut self,
        tool: Tool,
        samples: &[crate::command::InputSample],
        seed: u64,
        rope: f32,
    ) {
        let mut it = samples.iter();
        let Some(first) = it.next() else { return };
        // Replayed samples are already in canvas space and came from a fit or from a
        // generator, not from a device, so there is no device grain to declare.
        self.session
            .start_stroke(tool, *first, seed, crate::path::DEFAULT_TOLERANCE, rope);
        for s in it {
            self.session.stroke_to(*s);
        }
        if let Some(rec) = self.session.end_stroke() {
            self.commit(ActionKind::CommitStroke(rec));
        }
        self.mark_live_stale();
    }

    /// The in-flight tow, for the frontend's string overlay (§6.11) — a named
    /// read like [`view`](Self::view), at pointer rate while a smoothing brush
    /// draws. `None` whenever there is no string to show (no stroke, no rope,
    /// or the gesture has snapped to a shape).
    pub fn tow_string(&self) -> Option<crate::tow::TowString> {
        self.session.tow_string()
    }

    /// A snapshot of UI-facing state (§7).
    pub fn observe(&self) -> ObservableState {
        /// Whether `l`'s **own content** puts anything into the accumulator: a
        /// matte always covers, paint only once painted, a filter never — it
        /// rewrites what is there and adds nothing (§21.3).
        fn draws_content(l: &Layer) -> bool {
            match &l.content {
                LayerContent::Matte { .. } => true,
                LayerContent::Paint(_) => l.tiles().is_some_and(|t| !t.is_empty()),
                LayerContent::Filter(_) => false,
            }
        }
        /// Whether the compositor would draw anything at all for `l` — the same
        /// culls `render.rs` applies (visibility, opacity, emptiness), asked of the
        /// document rather than of a viewport, so the answer does not change when
        /// the artist scrolls.
        fn contributes(l: &Layer) -> bool {
            l.visible
                && l.composite.opacity > 0.0
                && (draws_content(l) || l.carries.iter().any(contributes))
        }
        let doc = self.timeline.current();
        // The layers and the substrate color are read from the *previewed*
        // document when one is in flight, so the frame's handles track a drag and
        // the color swatch tracks the picker (both live in the preview,
        // §15.7, §15.5) instead of lagging on the committed value — which
        // for the color would leave the panel disagreeing with the canvas it
        // controls, since rendering reads `presented`.
        //
        // Deliberately only those two. `has_selection` must stay committed-only —
        // a marquee drag would otherwise flash the selection bar in and out before
        // anything is selected — and that is asserted by
        // `a_selection_gesture_commits_the_same_op_it_previewed`. A stroke preview
        // changes no presentation property, so it is not consulted here at all.
        let shown = self.preview.doc().unwrap_or(doc);
        // Flattened in **composite order** — each stack bottom-to-top, a group's
        // base before what it carries — with the tree carried alongside as `depth`
        // and `carrier` (§14.6). Flat rather than nested because that
        // is the order a panel draws in and the order the compositor draws in, and
        // one list that means both is one thing to keep in agreement.
        let mut layers: Vec<LayerInfo> = Vec::new();
        // The carrier chain down to the current row: the id (what `LayerInfo::carrier`
        // reports) and whether that carrier's own content draws anything — the seed
        // for the stack its carries open, since a base composites at the bottom of
        // its group (§14.1).
        let mut carriers: Vec<(LayerId, bool)> = Vec::new();
        // Whether each depth's *current* stack has anything drawable beneath the row
        // being visited — the walk's own answer to "would a filter here reach
        // something" (§21.2), kept in agreement with the draw list's culls
        // (`render.rs`) rather than with row order: a hidden, transparent or empty
        // sibling fills nothing. Truncating on the way back up is what makes it
        // per-stack rather than per-depth: re-entering depth `d` from deeper is the
        // same stack and keeps its flag, while descending to a new `d` starts it at
        // the carrier's own content.
        let mut filled: Vec<bool> = Vec::new();
        shown.visit(&mut |l, depth| {
            carriers.truncate(depth);
            if filled.len() > depth {
                filled.truncate(depth + 1);
            } else {
                let seed = carriers.last().is_some_and(|&(_, draws)| draws);
                filled.resize(depth + 1, seed);
            }
            let has_underlay = filled[depth];
            filled[depth] = filled[depth] || contributes(l);
            layers.push(LayerInfo {
                id: l.id,
                blend: l.composite.blend,
                clip: l.composite.clip,
                opacity: l.composite.opacity,
                visible: l.visible,
                carrier: carriers.last().map(|&(id, _)| id),
                depth,
                is_group: l.is_group(),
                // Read straight off the traversal: composite order visits the
                // bottom of the root stack first, and that is the *only* layer
                // with nothing beneath it (§14.4.3) — every other one has either
                // a lower sibling or the content of the layer carrying it. So
                // "has a backdrop" is "is not the first row", and asking the tree
                // per layer was a search for an answer the walk already gave.
                has_backdrop: !layers.is_empty(),
                name: l.name.clone(),
                matte: match &l.content {
                    LayerContent::Matte { region, paint } => Some(MatteInfo {
                        rect: region.rect(),
                        paint: paint.clone(),
                    }),
                    LayerContent::Paint(_) | LayerContent::Filter(_) => None,
                },
                filter: l.filter(),
                has_underlay,
                // Asked of the *shown* document, like everything else on this row, so
                // the control tracks a drag preview rather than the value behind it.
                merge_down: crate::document::merge::plan(shown, l.id).map(|p| p.dest),
            });
            carriers.push((l.id, draws_content(l)));
        });
        ObservableState {
            can_undo: self.timeline.can_undo(),
            can_redo: self.timeline.can_redo(),
            is_stroking: self.session.is_stroking(),
            tool: self.session.tool,
            brush: self.session.brush,
            view: self.session.view,
            bounds: doc.bounds(),
            doc_revision: self.doc_revision,
            active_layer: self.session.active_layer,
            layers,
            has_selection: doc.has_selection(self.actor),
            selection_hull: doc.selection_of(self.actor).hull(),
            shape_action: self.session.shape_action,
            selection_feather: self.session.selection_feather,
            shape_opacity: self.session.shape_opacity,
            show_peer_selections: self.session.show_peer_selections,
            guides: self.session.guides.clone(),
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

    /// The current media/lighting parameters (§6.3).
    pub fn media_params(&self) -> crate::gpu::MediaParams {
        self.compositor_pipeline.media()
    }

    /// Import a brush-shape image (PNG bytes), returning its content id for use
    /// in `BrushParams::shape = BrushShape::Stamp(id)` (§6.6).
    pub fn import_brush(&self, png_bytes: &[u8]) -> Result<AssetId> {
        self.apply.assets.import(png_bytes)
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
        self.preview.invalidate();
        self.doc_revision += 1;
    }

    /// Move the history playhead one step, the way [`DocCommand::Undo`] and
    /// [`DocCommand::Redo`] each do it.
    ///
    /// The two are one operation named twice: a shared session logs the step as an
    /// `Undo` action peers can order (§5.4, §12.3) and a solo one navigates, and redo
    /// is an `Undo` of an `Undo` — so the *only* thing that differs is which pair of
    /// timeline methods is asked. Passing the pair rather than writing the body out
    /// twice is what stops the two drifting: dropping the preview, bumping the
    /// revision on the navigating branch and re-reading the document's ground
    /// afterwards are all things one arm could have grown and the other not.
    fn navigate(
        &mut self,
        as_action: impl Fn(&dyn Timeline) -> Option<ActionId>,
        step: impl Fn(&mut dyn Timeline, &mut ApplyCtx) -> bool,
    ) {
        self.preview.set_doc(None);
        if let Some(target) = as_action(self.timeline.as_ref()) {
            self.commit(ActionKind::Undo(target));
        } else {
            step(self.timeline.as_mut(), &mut self.apply);
            self.committed_changed();
        }
        // A step across a `SetSurface` moves the document's ground (§6.4).
        self.apply_document_surface();
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

    /// Mint the next layer id for this client (§17.9).
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
    /// something (§14.2).
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

    /// The document's color space id (§6.7).
    pub fn color_space(&self) -> ColorSpaceId {
        self.color_space.id()
    }

    fn next_action_id(&mut self) -> ActionId {
        let id = ActionId {
            lamport: self.clock,
            actor: self.actor,
        };
        self.clock += 1;
        id
    }
}

/// Build the GPU subsystems whose layout/shaders depend on the color space.
/// What the color-space-dependent GPU subsystems are built from.
///
/// Grouped because they are always supplied together: the pool, stroke renderer and
/// compositor are torn down and rebuilt as a set whenever the color space changes
/// (§6.7), and `surface` / `environment` / `selection` are precisely the
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
    MergeRenderer,
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
    // The color space's formats — the only ones this call site knows. The pool
    // unions in its own (the selection mask, the wide scratch aux), so none can be
    // forgotten here (`TilePool::new`). The residual's is `Rgba16Float`, which every
    // space's color already is, but it is passed rather than assumed for the same
    // reason the aux is: the first space to choose otherwise would meet
    // `acquire_tex`'s "unsupported format" panic on its first stroke.
    let pool = TilePool::new(
        gpu.clone(),
        [cs.color_format(), cs.aux_format()]
            .into_iter()
            .chain(cs.resid_format()),
    );
    let zeroes = Zeroes::new(gpu, cs.color_format(), cs.aux_format(), cs.resid_format());
    let stroke = StrokeRenderer::new(gpu, cs.clone(), selection.clone(), zeroes.clone());
    // Built once and shared: `gpu::merge` runs this very pipeline on tile-sized
    // targets to merge a layer down through its mode (§14.11), and building a second
    // one would decode the Mixbox LUT twice.
    let blend = Arc::new(BlendPass::new(
        gpu,
        cs.as_ref(),
        cs.color_format(),
        cs.aux_format(),
    ));
    let compositor_pipeline = CompositorPipeline::new(
        gpu,
        target_format,
        cs.as_ref(),
        surface.clone(),
        environment.clone(),
        blend.clone(),
    );
    let compositor = Compositor::new(&compositor_pipeline, viewport);
    let transform = TransformRenderer::new(gpu, cs.as_ref(), selection.clone(), zeroes.clone());
    let fill = FillRenderer::new(gpu, cs.clone(), selection.clone(), zeroes.clone());
    let merge = MergeRenderer::new(gpu, cs.as_ref(), zeroes, blend);
    (
        pool,
        stroke,
        compositor_pipeline,
        compositor,
        transform,
        fill,
        merge,
    )
}

/// Convenience for tests/tools: build an engine on a headless device.
pub async fn headless_engine(
    target_format: wgpu::TextureFormat,
    viewport: Extent2,
) -> Result<Engine> {
    headless_engine_with(target_format, viewport, ColorSpaceId::Oklab).await
}

/// Headless engine in a chosen color space (§6.7).
pub async fn headless_engine_with(
    target_format: wgpu::TextureFormat,
    viewport: Extent2,
    color_space: ColorSpaceId,
) -> Result<Engine> {
    let gpu = GpuContext::headless().await?;
    Engine::new_with_color_space(gpu, target_format, viewport, color_space)
}
