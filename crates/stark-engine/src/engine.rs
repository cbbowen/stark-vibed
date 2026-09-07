//! The engine: owns the GPU, session, and timeline; turns commands into state
//! and renders the canvas (§7).
//!
//! # What is where
//!
//! [`Engine`] is one type whose `impl` is split across this module's children by
//! subject; the three structs are defined here and nowhere else.
//!
//! - here — the state: [`Engine`], [`EngineShared`] and [`Authoring`], the one door
//!   a command comes in by ([`Engine::process`]), and the small named reads a
//!   frontend asks beside a projection;
//! - `build` — the GPU half, the constructors, the color-space rebuild, the
//!   headless engine tests use (§6.7, §11);
//! - `input` — commands into state: the arms of `process`, the setters, the stroke
//!   replay (§4);
//! - `commit` — the doors into the log, history navigation and retention (§5,
//!   §12.4);
//! - `observe` — the projection that comes back out ([`Engine::observe`]);
//! - `render` — the compositor's draw list, the screen frame and export (§6.3,
//!   §15.6);
//! - `live` — the preview fold and its per-stroke cache (§17.6, §6.2);
//! - `pick` — the eyedropper (§18.0.2);
//! - `collab` — the action and presence channels of a shared session (§12, §17);
//! - `file` — saving, opening, replay, and the resources a replay is run against
//!   (§8, §6.4, §6.6).
//!
//! The data structures the read side is built from — [`Projected`], `Memo` and the
//! `Revision` that keys one — name no engine type and live at the crate root
//! (`projection`).

mod build;
mod collab;
mod commit;
mod file;
mod input;
mod live;
mod observe;
mod pick;
pub(crate) mod render;

use std::sync::Arc;

pub use crate::projection::Projected;
pub use build::headless_engine;
pub use collab::PresenceTick;
pub use commit::{DEFAULT_FAST_COMMIT, DEFAULT_HISTORY_BUDGET};
pub use observe::{GuideInfo, Guides, LayerInfo, Layers, MatteInfo, ObservableState};
pub use pick::{PickOptions, PickSource};
pub use render::{Background, ExportPlan, ExportScale, Rendered};

use crate::Result;
use crate::colorspace::ColorSpace;
use crate::command::InputCommand;
use crate::document::{ApplyCtx, DocState, Timeline};
use crate::gpu::scratch::ScratchPool;
use crate::gpu::{
    Compositor, CompositorPipeline, EnvironmentId, GpuContext, MediaParams, Output, Registry,
};
use crate::peer::Peers;
use crate::projection::{Memo, Revision};
use crate::view::ViewTransform;
use observe::{GuideKey, ShownKey};
use stark_model::document::{Action, ActorId, LayerId, Scaffold};
use stark_model::{AssetId, ColorSpaceId, SubstrateId};

/// The starting layer present in every new document.
const ROOT_LAYER: LayerId = LayerId::ROOT;

/// The expensive half of an engine: everything a second engine on the same device
/// reuses rather than rebuilding (§11).
///
/// **The store is shared; the choice is not** — [`Registry`]'s own rule. The
/// registered bytes, decoded substrates and environments, tile pool, compiled
/// pipelines and brush assets are genuinely one copy behind `Arc`s. The choices that
/// ride along — substrate, environment, media parameters — are per-engine values a
/// clone merely **seeds** from the donor, so a sibling opens mirroring the canvas it
/// came from and is free to move from there.
///
/// One type rather than a constructor's argument list, so that anything added here
/// is shared on every path — the new engine, the sibling, and the color-space
/// rebuild. It is also cheap to clone and outlives whoever it came from, so a
/// consumer wanting only the device and the pipelines (a preset thumbnail) need not
/// borrow a live engine to reach them.
#[derive(Clone)]
pub struct EngineShared {
    gpu: GpuContext,
    /// The format every pipeline in `passes` was compiled against. A sibling must
    /// present to the same one — a second substrate that chose differently would fail
    /// validation rather than merely look wrong.
    target_format: wgpu::TextureFormat,
    /// The document's color space (§6.7). Shared because the pipelines below were
    /// built for it: a sibling in a *different* space is not a sibling at all, it is
    /// a rebuild (`rebuild_gpu_for`), which replaces this whole value.
    color_space: Arc<dyn ColorSpace>,
    /// The GPU subsystems an action needs in order to apply itself — the tile pool,
    /// the stroke renderer, the asset store, the selection rasterizer, and the canvas
    /// substrates — held as the `history::Action::Context` (§5).
    ///
    /// Stored rather than built per call: it only changes when the color space is
    /// rebuilt, and `Context` is an owned associated type, so building it per call
    /// would clone all of it on every commit, undo, redo and remote merge.
    ///
    /// `selection` is color-space independent (a mask is one coverage channel whatever
    /// the paint is), so unlike the pool and the stroke renderer it survives a rebuild.
    apply: ApplyCtx,
    /// The working textures and buffers every recording leases (`gpu::scratch`), one
    /// pool for the whole stack. Held here as well as inside the renderers that lease
    /// from it, because a rebuild has to carry it across and those renderers do not
    /// survive one (`GpuKeep`).
    scratch: ScratchPool,
    /// The compiled compositing passes — the ~19 shaders and ~30 pipelines that make
    /// building an engine expensive. A sibling's [`CompositorPipeline`] is built over
    /// these ([`CompositorPipeline::sharing`]), so it pays for its own three view
    /// settings and nothing else.
    passes: Arc<crate::gpu::composite::CompositorPasses>,
    /// The HDR lighting environment and its registered bytes (§6.3). A view setting,
    /// so it is the *store* that is shared and the current id that is seeded.
    environment: Registry<EnvironmentId>,
    /// The media/lighting parameters a sibling opens with (§6.3) — a seed, not a
    /// shared value; see the note on the type.
    media: MediaParams,
    /// The display a sibling opens presenting to (§6.5) — a seed on `media`'s terms,
    /// since a sibling's surface is the same screen's.
    output: Output,
}

pub struct Engine {
    /// Everything a sibling engine reuses (§11). Held as one value so that a thing
    /// added to it is shared on every path rather than on the paths somebody
    /// remembered — see [`EngineShared`].
    shared: EngineShared,
    /// Compositing state for the **substrate**: the attachments a screen frame is
    /// built through, kept from frame to frame (`gpu::composite`). Anything drawn
    /// beside the screen — an export, the navigator's miniature — gets a
    /// [`Compositor`] of its own for the call, so it never resizes these.
    compositor: Compositor,
    /// The pipelines, layouts and view settings every `Compositor` shares. Beside the
    /// one above rather than inside it because a second `Compositor` borrows it: the
    /// view settings the media pass reads have one owner, so two consumers cannot
    /// disagree about the canvas substrate or the lighting.
    compositor_pipeline: CompositorPipeline,
    /// The substrate the action log starts from, written to `CanvasMeta` and used to
    /// seed the document. Plays the same role as `CanvasMeta::color_space`: it
    /// describes the empty document that the log is replayed onto, and is not
    /// itself a logged change.
    initial_substrate: SubstrateId,
    timeline: Timeline,
    session: crate::session::Session,
    /// Everyone else in the session (§17.4). Empty when solo.
    peers: Peers,
    /// The presence clock: the newest instant a caller has handed in, in seconds on
    /// a monotonic scale.
    ///
    /// The engine owns no clock *source* (so it runs on wasm and native alike) but it
    /// owns the *value*, so expiry, publishing and the timestamping of arriving frames
    /// all see one instant. Advanced by `max`: a clock that steps backwards must not
    /// un-expire a peer.
    now: f64,
    /// What is being *shown* over the committed document, and the caches that make
    /// showing it affordable: the unlogged drag in flight, the fold of every
    /// in-flight gesture, the settled head of each live stroke, and the epoch that
    /// says when a head has gone stale (§17.6).
    ///
    /// One field rather than four because they carry one invariant: the slot cannot
    /// move without the epoch moving with it.
    preview: live::Preview,
    /// How much resident tile memory history retention may hold before undo depth
    /// is given up (§5) —
    /// [`ViewCommand::SetHistoryBudget`](crate::command::ViewCommand::SetHistoryBudget),
    /// defaulting to
    /// [`DEFAULT_HISTORY_BUDGET`].
    ///
    /// Per-client and never logged: how much history a machine can afford is a fact
    /// about the machine.
    history_budget: u64,
    /// Whether a stroke's commit takes the tiles its live preview already drew
    /// (§6.2) — [`ViewCommand::SetFastCommit`](crate::command::ViewCommand::SetFastCommit),
    /// defaulting to
    /// [`DEFAULT_FAST_COMMIT`].
    ///
    /// Per-client and never logged, like the budget above: it changes how *this*
    /// client spends the moment the pointer comes up, not the document. A peer
    /// receives the stroke as an action either way.
    fast_commit: bool,
    /// The compositor's draw list and the key it was built from — the largest of the
    /// three memos, and the only one whose value is not a projection
    /// ([`Engine::draw_list`]).
    draw_cache: Memo<render::DrawKey, std::sync::Arc<[crate::gpu::CompositeGroup]>>,
    /// The layer roster and the key it was built from ([`Engine::projected_layers`]).
    layer_cache: Memo<ShownKey, Layers>,
    /// The guide roster this client sees, on the roster above's terms plus one
    /// ([`Engine::projected_guides`]).
    guide_cache: Memo<GuideKey, Guides>,
    /// Bumped whenever this client opens or shuts a **guide's eye** (§20.5).
    ///
    /// Its own counter beside `doc_revision` and the exact complement of one: the eye
    /// is the one thing about a guide that is not in the document, so the document's
    /// revision does not move when it does, and the roster this client sees is a
    /// function of both.
    guide_epoch: Revision,
    /// Bumped whenever the **committed** document changes — a commit, an undo, a
    /// merged remote action, a load. Projected as
    /// [`ObservableState::doc_revision`], which is what a frontend showing a
    /// rendered stand-in for the document (the navigator's miniature) watches.
    ///
    /// Strictly narrower than the preview's epoch, which an unlogged drag also bumps
    /// at pointer rate without changing the document. The two advance together
    /// through [`Engine::committed_changed`].
    doc_revision: u64,
    /// What [`doc_revision`](Self::doc_revision) read when the document now open
    /// *arrived* — the reset that made a new one, or the last action of a load.
    /// Projected as [`ObservableState::edited`], which is the comparison.
    ///
    /// The engine's rather than a frontend's, because every way a document can be
    /// replaced — `new_document`, `load_document`, a collaboration join — reaches
    /// [`reset_document`](Self::reset_document). The other half of "unsaved" —
    /// which revision was last written to a file — is the frontend's alone.
    doc_origin: u64,
    /// How many of this client's stroke commits took the preview's tiles instead of
    /// rendering the stroke again (`PreparedStroke`, §6.2). For tests and
    /// diagnostics: the two paths are the same pixels by design, so only a count can
    /// say which one ran.
    strokes_reused: u64,
    /// Raw pointer reports of the in-flight stroke, dumped on release under the
    /// `debug-unfrozen` feature so a misfit stroke can be replayed as a test.
    #[cfg(feature = "debug-unfrozen")]
    debug_samples: Vec<crate::command::InputSample>,
    /// Who this client is when it writes to the log, and the counters that keep
    /// its writes unique.
    authoring: Authoring,
}

/// Who this client is when it writes to the log (§17.9), and what it owes the
/// wire (§12.4).
///
/// One struct because every field moves at exactly the moments that identity does —
/// sharing, joining, and the reset that precedes a load — so "a fresh solo session"
/// is one value rather than a shape each call site fills in for itself.
struct Authoring {
    actor: ActorId,
    /// This client's Lamport counter: the `lamport` half of every
    /// [`ActionId`](stark_model::document::ActionId) it
    /// mints, advanced past everything it has seen (§12.1).
    clock: u64,
    /// Locally-committed actions awaiting broadcast to peers (§12.4), drained by
    /// the transport through [`Engine::take_outbox`].
    ///
    /// `None` when solo, rather than an empty `Vec` beside a flag: the presence of
    /// the queue *is* the answer to [`is_shared`](Engine::is_shared), and it decides
    /// whether a commit pays to clone its action at all — a stroke's control-point
    /// list is the largest thing in the log.
    outbox: Option<Vec<Action>>,
}

impl Authoring {
    /// A fresh, unshared session: the solo actor, the clock at its origin, nothing
    /// owed to anybody.
    ///
    /// One counter, because a `LayerId` is the id of the action that minted it, so
    /// the clock is the only thing a fresh session starts and a loaded one resumes
    /// (§17.9).
    const fn solo() -> Self {
        Self {
            actor: ActorId::SOLO,
            clock: 0,
            outbox: None,
        }
    }
}

impl Engine {
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

    /// The in-flight tow, for the frontend's string overlay (§6.11) — a named
    /// read like [`view`](Self::view), at pointer rate while a smoothing brush
    /// draws. `None` whenever there is no string to show (no stroke, no rope,
    /// or the gesture has snapped to a shape).
    pub fn tow_string(&self) -> Option<crate::tow::TowString> {
        self.session.tow_string()
    }

    /// Whether a hover mark is folded into the shown canvas (§18.1.10) — what a
    /// frontend peeks before spending a
    /// [`ViewCommand::PreviewHover`](crate::command::ViewCommand::PreviewHover)`(None)`,
    /// so taking the mark down costs a command and a repaint only when there is
    /// one to take down.
    pub fn hover_held(&self) -> bool {
        self.session.hover_held()
    }

    /// What the stroke in flight has snapped to (§6.9), or `None` where there is no
    /// stroke or the hold found nothing.
    ///
    /// A named read like [`view`](Self::view) and [`tow_string`](Self::tow_string).
    /// The frontend owns the *dwell* (§6.9) and the engine owns what a hold means, so
    /// whether one found anything is knowable only here.
    ///
    /// Read **before** the gesture's `End`, which is the only moment it answers: what
    /// is committed is the path the shape produced, not the shape, and the assist goes
    /// with the gesture (`assist::AssistShape`).
    pub fn assisted(&self) -> Option<crate::assist::Assisted> {
        self.session.assisted()
    }

    /// What the guide overlay draws and what a snapped stroke is held to, for the
    /// document `doc` — this client's shown guides (§20.5), gathered.
    ///
    /// Where the two halves of the roster meet on the *rendering* side, as
    /// `GuideInfo` is on the panel's: the document holds the guides,
    /// [`Session::shown_guides`](crate::session::Session::shown_guides) drops the
    /// ones this client has hidden, and everything past here sees only geometry.
    pub(crate) fn scaffold(&self, doc: &DocState) -> Scaffold {
        Scaffold::of(self.session.shown_guides(doc).map(|g| &g.camera))
    }

    /// Whether the GPU is still usable, and what went wrong if not (§5) — the same
    /// fact [`ObservableState::gpu_failure`] projects, as a **request** for a caller
    /// that holds the engine and has no projection to hand.
    ///
    /// The collaboration pump is the caller: it services peer traffic without taking
    /// an observation each time (§17.5), and must stop applying anything once the
    /// device has died.
    pub fn gpu_failure(&self) -> Option<crate::gpu::DeviceFailure> {
        self.shared.gpu.health().failure()
    }

    /// The current committed document state. `pub` for the suite and nothing else,
    /// and hidden to say so (`testing`).
    #[doc(hidden)]
    pub fn document(&self) -> &DocState {
        self.timeline.current()
    }

    /// Where the history playhead stands and how far it can travel, in actions —
    /// or `None` for a document whose history is not this client's alone to walk
    /// (a shared session). See
    /// [`Timeline::scrub_range`](crate::document::Timeline::scrub_range).
    ///
    /// A **request** rather than a field of [`ObservableState`]: it is asked for only
    /// while a scrubber is on screen, and in the projection every pointer sample of
    /// every stroke would pay for it.
    pub fn scrub_range(&self) -> Option<(usize, usize)> {
        self.timeline.scrub_range()
    }

    /// A caption per action across the whole scrub range, oldest first — what a
    /// scrubber labels its ticks with.
    pub fn scrub_labels(&self) -> Vec<&'static str> {
        self.timeline.scrub_labels()
    }

    /// The GPU context this engine renders with (for substrate/readback setup).
    pub fn gpu(&self) -> &GpuContext {
        self.shared.gpu()
    }

    /// The current pan/zoom view (for mapping pointer input to canvas space).
    pub fn view(&self) -> ViewTransform {
        self.session.view
    }

    /// Import a brush-shape image (PNG bytes), returning its content id for use
    /// in `BrushParams::shape = BrushShape::Stamp(id)` (§6.6).
    pub fn import_brush(&self, png_bytes: &[u8]) -> Result<AssetId> {
        self.shared.apply.assets.import(png_bytes)
    }

    /// The document's color space id (§6.7).
    pub fn color_space(&self) -> ColorSpaceId {
        self.shared.color_space()
    }
}
