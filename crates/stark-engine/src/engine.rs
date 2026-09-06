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
//! `impl` is split across this module's children by subject. A child module can
//! reach a private field of a struct its parent defines, so this is a division of
//! the *file* and not of the type: no field moved, and nothing became `pub(crate)`
//! to allow it — which is why the three structs are defined here and nowhere else.
//!
//! - here — **the state**: [`Engine`], [`EngineShared`] and [`Authoring`] with every
//!   field's doc, the one door a command comes in by ([`Engine::process`]), and the
//!   small named reads a frontend asks beside a projection;
//! - `build` — what an engine is made of: the GPU half, the constructors, the
//!   color-space rebuild, the headless one tests use (§6.7, §11);
//! - `input` — commands into state: the four arms of `process`, the setters, the
//!   stroke replay (§4);
//! - `commit` — what logging an action costs: the four doors into the log, history
//!   navigation and retention (§5, §12.4);
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
/// **What "shared" means here is exactly what [`Registry`] means by it** — *the store
/// is shared; the choice is not*. The maps of registered bytes, the decoded substrates
/// and environments, the tile pool, the compiled pipelines and the content-addressed
/// brush assets all sit behind `Arc`s and are genuinely one copy. The *choices* that
/// ride along — which substrate is in use, which environment, the media parameters —
/// are per-engine values that a clone merely **seeds** from the donor, so a sibling
/// opens mirroring the canvas it came from and is free to move from there. That
/// seeding is deliberate and is what lets the brush editor's preview open on the
/// document's own substrate with nothing re-fetched (`CompositorPipeline::sharing`).
///
/// **Why it is a type rather than a constructor's argument list.** Assembled field by
/// field, a renderer added to [`ApplyCtx`] is shared by every engine except the one
/// that constructor builds, and nothing says so until a preview canvas uses it.
/// `ApplyCtx` closes that one level down by being cloned whole; this is the same move
/// one level up, so a thing added here is shared on every path — the new engine, the
/// sibling, and the color-space rebuild.
///
/// It is also what a consumer can *hold*. A preset thumbnail wants the device and the
/// pipelines and nothing else, and this clones for a handful of refcount bumps and
/// outlives whoever it came from — where borrowing a live engine to reach them means
/// the thumbnail rig cannot exist until one does.
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
    /// Stored rather than built per call. `history`'s `Context` is an owned associated
    /// type, so there is nothing to hand it a borrow of — and building it per call
    /// means cloning all of it on *every* commit, undo, redo and remote merge: tens of
    /// `Arc` bumps plus a `HashMap` allocation each time, for a value that only changes
    /// when the color space is rebuilt.
    ///
    /// `selection` is color-space independent (a mask is one coverage channel whatever
    /// the paint is), so unlike the pool and the stroke renderer it survives a rebuild.
    apply: ApplyCtx,
    /// The working textures and buffers every recording leases (`gpu::scratch`), one
    /// pool for the whole stack. Here as well as inside the renderers that lease from
    /// it, because a rebuild has to carry it across and the renderers it would ask do
    /// not survive one (`GpuKeep`).
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
    /// The pipelines, layouts and view settings every `Compositor` shares. Held
    /// beside the one above rather than inside it because a second one borrows it:
    /// the expensive half of compositing is built once, and the view settings the
    /// media pass reads have one owner, so two consumers cannot disagree about the
    /// canvas substrate or the lighting.
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
    /// `stark-engine` deliberately owns no clock *source* — that is what lets it run on
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
    /// One field rather than four, because they are one thing with one invariant: as
    /// four they can be moved independently, and a call site that drops the drag
    /// preview without bumping the epoch shows a document nothing knows is stale. The
    /// slot cannot move without the epoch moving with it.
    preview: live::Preview,
    /// How much resident tile memory history retention may hold before undo depth
    /// is given up (§5) —
    /// [`ViewCommand::SetHistoryBudget`](crate::command::ViewCommand::SetHistoryBudget),
    /// defaulting to
    /// [`DEFAULT_HISTORY_BUDGET`].
    ///
    /// Per-client and never logged, for the reason the command's doc gives: how much
    /// history a machine can afford is a fact about the machine.
    history_budget: u64,
    /// Whether a stroke's commit takes the tiles its live preview already drew
    /// (§6.2) — [`ViewCommand::SetFastCommit`](crate::command::ViewCommand::SetFastCommit),
    /// defaulting to
    /// [`DEFAULT_FAST_COMMIT`].
    ///
    /// Per-client and never logged, like the budget above and for a related reason:
    /// what it changes is not the document but how *this* client spends the moment
    /// the pointer comes up. A peer receives the stroke as an action either way and
    /// renders it whole, so nothing here reaches anybody else's picture.
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
    /// Its own counter beside `doc_revision` because it is the exact complement of
    /// one: the eye is the one thing about a guide that is *not* in the document,
    /// so nothing in the document's revision moves when it does — and the roster
    /// this client sees is a function of both.
    guide_epoch: Revision,
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
    /// What [`doc_revision`](Self::doc_revision) read when the document now open
    /// *arrived* — the reset that made a new one, or the last action of a load.
    /// Projected as [`ObservableState::edited`], which is the comparison.
    ///
    /// A baseline the engine keeps rather than one a frontend keeps for it, because
    /// the engine is where the list of ways a document can be replaced is complete:
    /// `new_document`, `load_document` and a collaboration join all reach
    /// [`reset_document`](Self::reset_document), and a frontend tracking the same
    /// thing would be enumerating those call sites and missing the next one. What
    /// a frontend does own is the other half of "unsaved" — which revision it last
    /// wrote to a file — and that is a question no engine can answer.
    doc_origin: u64,
    /// How many of this client's stroke commits took the preview's tiles instead of
    /// rendering the stroke again (`PreparedStroke`, §6.2). For tests and
    /// diagnostics, on `live_head_count`'s terms: the two paths are the same pixels
    /// by design, so only a count can say which one ran.
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
/// One struct because they are one thing and they *move* as one thing: an engine is
/// authoring solo, or as some actor in a session, and every field here changes at
/// exactly the moments that identity does — sharing, joining, and the reset that
/// precedes a load. Flat, "a fresh solo session" is stated once in `reset_document`
/// and again in each constructor, and the compiler can hold those to the same *shape*
/// but never to the same values.
struct Authoring {
    actor: ActorId,
    /// This client's Lamport counter: the `lamport` half of every
    /// [`ActionId`](stark_model::document::ActionId) it
    /// mints, advanced past everything it has seen (§12.1).
    clock: u64,
    /// Locally-committed actions awaiting broadcast to peers (§12.4), drained by
    /// the transport through [`Engine::take_outbox`].
    ///
    /// `None` when solo, rather than an empty `Vec` beside a flag: "queued actions
    /// that will never be sent" was a state the pair could express and this cannot,
    /// and the presence of the queue *is* the answer to
    /// [`is_shared`](Engine::is_shared). It also decides whether a commit pays to
    /// clone its action at all — a stroke's control-point list is the largest thing
    /// in the log, and a solo session has nowhere to put the copy.
    outbox: Option<Vec<Action>>,
}

impl Authoring {
    /// A fresh, unshared session: the solo actor, the clock at its origin, nothing
    /// owed to anybody.
    ///
    /// One counter, because a `LayerId` is the id of the action that minted it — so
    /// the clock is the only thing a fresh session has to start and a loaded one has
    /// to resume (§17.9).
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
    /// A named read like [`view`](Self::view) and [`tow_string`](Self::tow_string),
    /// and it exists for the reason the split in §6.9 exists: the frontend owns the
    /// *dwell* — how long a pause has to be is a fact about a hand and there is no
    /// clock here — while the engine owns what a hold **means**. So whether a hold
    /// found anything is knowable only on this side, and a frontend that wants to
    /// know has to ask.
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
    /// The one place the two halves of the roster meet on the *rendering* side, as
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
    /// The collaboration pump is the caller that wants it: it services peer traffic
    /// without taking an observation each time (§17.5), and a device that has died is
    /// exactly the thing that should stop it applying anything further.
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
