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
//! - `build` — the GPU half, the constructors, the color-space rebuild (§6.7, §11);
//! - `input` — commands into state: the arms of `process`, the setters, the stroke
//!   replay (§4);
//! - `commit` — the doors into the log, history navigation and retention (§5,
//!   §12.4);
//! - `observe` — the projection that comes back out ([`Engine::observe`]);
//! - `draw_list` — what a document draws, as a function of the document (§6.3,
//!   §14.7);
//! - `presenter` — the compositing state a frame is built through, and the memo the
//!   draw list is held in (§6.3);
//! - `render` — gathering a frame and handing it over, and export (§6.3, §15.6);
//! - `live` — the preview fold and its per-stroke cache (§17.6, §6.2);
//! - `pick` — the eyedropper (§18.0.2);
//! - `collab` — the action and presence channels of a shared session (§12, §17);
//! - `file` — saving, opening, replay, and the resources a replay is run against
//!   (§8, §6.4, §6.6);
//! - `capture` — the raw pointer reports a `debug-unfrozen` build keeps (§6.2);
//! - `probe` — the suite's [`Probe`](crate::testing::Probe) reads (§9).
//!
//! The data structures the read side is built from — [`Projected`], `Memo` and the
//! `Revision` that keys one — live at the crate root (`projection`).

mod build;
mod capture;
mod collab;
mod commit;
mod draw_list;
mod file;
mod input;
mod live;
mod observe;
mod pick;
mod presenter;
mod probe;
mod render;

use std::sync::Arc;

pub use crate::projection::Projected;
pub use collab::PresenceTick;
pub use commit::{DEFAULT_FAST_COMMIT, DEFAULT_HISTORY_BUDGET};
pub use observe::{GuideInfo, Guides, LayerInfo, Layers, MatteInfo, ObservableState};
pub use pick::{PickOptions, PickSource};
pub use render::{Background, ExportPlan, ExportScale, Rendered};

use crate::Result;
use crate::colorspace::ColorSpace;
use crate::command::InputCommand;
use crate::device::context::GpuContext;
use crate::device::scratch::ScratchPool;
use crate::document::DocState;
use crate::document::Substrate;
use crate::fold::ApplyCtx;
use crate::gpu::composite::ViewSettings;
use crate::gpu::environment::EnvironmentId;
use crate::gpu::registry::Registry;
use crate::peer::Peers;
use crate::presence::PresenceTx;
use crate::projection::{Memo, Revision};
use crate::view::ViewTransform;
use capture::DebugCapture;
use commit::ClientSettings;
use observe::{GuideKey, ShownKey};
use presenter::Presenter;
use stark_model::document::{Action, ActorId, LayerId, Scaffold};
use stark_model::{AssetId, ColorSpaceId};

/// The starting layer present in every new document.
const ROOT_LAYER: LayerId = LayerId::ROOT;

/// The expensive half of an engine: everything a second engine on the same device
/// reuses rather than rebuilding (§11).
///
/// **The store is shared; the choice is not** — [`Registry`]'s own rule. Registered
/// bytes, decoded substrates and environments, tile pool, compiled pipelines and
/// brush assets are one copy behind `Arc`s; the choices riding along — substrate,
/// environment, media parameters — are per-engine values a sibling only **seeds** from
/// the donor, so it opens mirroring the canvas it came from.
///
/// **Pictures are not shared** (§23): they are the donor's document's, so the value
/// [`Engine::shared`] hands out holds none (`ApplyCtx::for_sibling`).
///
/// Outlives its donor, so a consumer wanting only the device and the pipelines (a
/// preset thumbnail) need not borrow a live engine.
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
    /// A color-space rebuild replaces the pool and the stroke renderer; `selection`
    /// survives one, a mask being one coverage channel whatever the paint is.
    apply: ApplyCtx,
    /// The working textures and buffers every recording leases (`gpu::scratch`), one
    /// pool for the whole stack. Held here as well as inside the renderers that lease
    /// from it, because a rebuild has to carry it across and those renderers do not
    /// survive one (`GpuKeep`).
    scratch: ScratchPool,
    /// The compiled compositing passes — the ~19 shaders and ~30 pipelines that make
    /// building an engine expensive. A sibling's [`Presenter`] is built over these
    /// (`CompositorPipeline::sharing`), so it pays for its own view settings and
    /// nothing else.
    passes: Arc<crate::gpu::composite::CompositorPasses>,
    /// The HDR lighting environment and its registered bytes (§6.3). A view setting,
    /// so it is the *store* that is shared and the current id that is seeded.
    environment: Registry<EnvironmentId>,
    /// The lighting and the display a sibling opens with (§6.3, §6.5) — a seed. The
    /// display too, since a sibling's surface is the same screen's.
    settings: ViewSettings,
}

pub struct Engine {
    /// Everything a sibling engine reuses (§11). Held as one value so that a thing
    /// added to it is shared on every path rather than on the paths somebody
    /// remembered — see [`EngineShared`].
    shared: EngineShared,
    /// How a frame is composited (§6.3) — see [`Presenter`], which also says why the
    /// three it holds are apart from the rest of this struct.
    presenter: Presenter,
    /// The substrate the action log starts from, and the size it is laid at — the
    /// initial condition [`origin`](Self::origin) builds the replay base on, as
    /// `CanvasMeta::color_space` is the other. Not itself a logged change.
    initial: Substrate,
    /// The action log and the committed document it folds to — changed only through
    /// [`Engine::change_committed`] (see [`commit::Committed`]).
    timeline: commit::Committed,
    session: crate::session::Session,
    /// How the canvas is being looked at (§6.4, §18.1.2) — pan, zoom, rotation and
    /// the mirror. Per-client, never logged and never sent.
    ///
    /// The engine's rather than the session's because nothing about a *gesture*
    /// depends on it: a stroke is fitted in canvas space, and the view is what the
    /// renderer and the projection read.
    view: ViewTransform,
    /// Whether collaborators' selection outlines are drawn (§17.3). Off by default: a
    /// second contour over the artwork is a cost paid on every frame you look at it,
    /// for a fact that is occasionally useful.
    show_peer_selections: bool,
    /// The half of this client published to collaborators (§17.4): the name, the
    /// cursor, and the latch that decides when a frame goes out.
    ///
    /// Beside [`peers`](Self::peers) rather than inside the session, because the two
    /// are the ends of one channel and sat at different depths: what crosses from the
    /// session is the active layer, the gesture ordinal and
    /// [`Session::live`](crate::session::Session::live), handed over per publish.
    presence: PresenceTx,
    /// Everyone else in the session (§17.4), and the presence clock they are expired
    /// against ([`Peers::advance`](crate::peer::Peers::advance)). Empty when solo.
    peers: Peers,
    /// What is being *shown* over the committed document, and the caches that make
    /// showing it affordable: the unlogged drag in flight, the fold of every
    /// in-flight gesture, the settled head of each live stroke, and the epoch that
    /// says when a head has gone stale (§17.6). One field rather than four, since the
    /// slot cannot move without the epoch moving with it.
    preview: live::Preview,
    /// What this client is willing to spend, as a frontend has set it — see
    /// [`ClientSettings`].
    client: ClientSettings,
    /// The layer roster and the key it was built from ([`Engine::projected_layers`]).
    layer_cache: Memo<ShownKey, Layers>,
    /// The guide roster this client sees, on the roster above's terms plus one
    /// ([`Engine::projected_guides`]).
    guide_cache: Memo<GuideKey, Guides>,
    /// Bumped whenever this client opens or shuts a **guide's eye** (§20.5). Its own
    /// counter beside `doc_revision`, the eye being the one thing about a guide that
    /// is not in the document; the roster this client sees is a function of both.
    guide_epoch: Revision,
    /// Bumped whenever the **committed** document changes — a commit, an undo, a
    /// merged remote action, a load. Projected as
    /// [`ObservableState::doc_revision`], which is what a frontend showing a
    /// rendered stand-in for the document (the navigator's miniature) watches.
    ///
    /// Strictly narrower than the preview's epoch, which an unlogged drag also bumps
    /// at pointer rate without changing the document.
    doc_revision: u64,
    /// What [`doc_revision`](Self::doc_revision) read when the document now open
    /// *arrived* — the reset that made a new one, or the last action of a load.
    /// Projected as [`ObservableState::edited`], which is the comparison.
    ///
    /// The other half of "unsaved" — which revision was last written to a file — is
    /// the frontend's alone.
    doc_origin: u64,
    /// How many of this client's stroke commits took the preview's tiles instead of
    /// rendering the stroke again (`PreparedStroke`, §6.2). The two paths are the same
    /// pixels by design, so only a count can say which one ran.
    strokes_reused: u64,
    /// The raw pointer reports of the stroke in hand, for a dump — see
    /// [`DebugCapture`], which is nothing at all without its feature.
    debug: DebugCapture,
    /// Who this client is when it writes to the log, and the counters that keep
    /// its writes unique.
    authoring: Authoring,
}

/// Who this client is when it writes to the log (§17.9), and what it owes the
/// wire (§12.4). One struct because every field moves at the moments identity does:
/// sharing, joining, and the reset that precedes a load.
struct Authoring {
    actor: ActorId,
    /// This client's Lamport counter: the `lamport` half of every
    /// [`ActionId`](stark_model::document::ActionId) it
    /// mints, advanced past everything it has seen (§12.1).
    clock: u64,
    /// Locally-committed actions awaiting broadcast to peers (§12.4), drained by
    /// the transport through [`Engine::take_outbox`].
    ///
    /// `None` when solo: the presence of the queue *is* the answer to
    /// [`is_shared`](Engine::is_shared), and it decides whether a commit pays to
    /// clone its action at all.
    outbox: Option<Vec<Action>>,
}

impl Authoring {
    /// A fresh, unshared session: the solo actor, the clock at its origin, nothing
    /// owed to anybody. The clock is the only thing a loaded session resumes, a
    /// `LayerId` being the id of the action that minted it (§17.9).
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
    /// Read **before** the gesture's `End`, the only moment it answers: the assist
    /// goes with the gesture, and what is committed is the path the shape produced,
    /// not the shape.
    pub fn assisted(&self) -> Option<crate::assist::Assisted> {
        self.session.assisted()
    }

    /// What the guide overlay draws and what a snapped stroke is held to, for the
    /// document `doc` — this client's shown guides (§20.5), gathered.
    ///
    /// The eye is applied here — the document holds the guides,
    /// [`Session::shown_guides`](crate::session::Session::shown_guides) drops the
    /// ones this client has hidden — so everything past here sees only geometry.
    fn scaffold(&self, doc: &DocState) -> Scaffold {
        Scaffold::of(self.session.shown_guides(doc).map(|g| &g.camera))
    }

    /// Whether the GPU is still usable, and what went wrong if not (§5) — the same
    /// fact [`ObservableState::gpu_failure`] projects, as a **request** for a caller
    /// that holds the engine and has no projection to hand. The collaboration pump is
    /// that caller: it services peer traffic without taking an observation each time
    /// (§17.5), and must stop applying anything once the device has died.
    pub fn gpu_failure(&self) -> Option<crate::error::DeviceFailure> {
        self.shared.gpu.health().failure()
    }

    /// The current committed document state.
    pub(crate) fn document(&self) -> &DocState {
        self.timeline.current()
    }

    /// Where the history playhead stands and how far it can travel, in actions —
    /// or `None` for a document whose history is not this client's alone to walk
    /// (a shared session). See
    /// [`Timeline::scrub_range`](crate::fold::Timeline::scrub_range).
    ///
    /// A **request** rather than a field of [`ObservableState`]: only a scrubber on
    /// screen asks for it, where in the projection every pointer sample of every
    /// stroke would pay for it.
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
        self.view
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
