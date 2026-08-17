//! The app's shared state handle and the command path everything funnels through.
//!
//! [`dispatch`] is the single seam between the UI and the engine: every mutation
//! goes through it, so repaint, observable refresh and collaboration broadcast
//! happen in one place rather than at each call site (§4).
//!
//! "Goes through it" is enforced by the types rather than asked for in prose. The
//! engine and its projection are held in [`ReadOnly`] handles, which have `read` and
//! `peek` and no `write`, so `&mut Renderer` cannot be obtained outside this module
//! at all — the only doors are [`dispatch`] and [`with_engine`], which publish, and
//! [`with_engine_quiet`], which is named for what it declines to do. The failure this
//! rules out: a panel reaches the engine through the signal, changes something the
//! chrome reads back, and leaves the chrome showing the previous value until an
//! unrelated command refreshes it (§4, §7). Both the canvas ground and the lighting
//! environment are reachable that way, and neither spelling compiles.

use dioxus::dioxus_core::{Subscribers, Task};
use dioxus::prelude::*;

use crate::collab;
use crate::gesture::TransformUi;
use crate::prefs::Prefs;
use crate::render::Renderer;
use stark_engine::command::ViewCommand;
use stark_engine::document::{BrushParams, LayerId};
use stark_engine::{InputCommand, ObservableState};
use stark_model::geom::Vec2;

/// Create one of [`AppState`]'s signals, owned by the **root** scope rather than by
/// the component that declares it.
///
/// A `use_signal` belongs to the scope that ran the hook, and reading it from a scope
/// that is not a descendant is what Dioxus warns about — the value could be dropped
/// while still held. Every signal in `AppState` is reached from detached
/// `spawn_forever` tasks (the renderer init, the collaboration pumps), and those live
/// in [`ScopeId::ROOT`], which is *not* a descendant of `app`. Owning them at the root
/// is the fix the warning itself prescribes: move the value to the common ancestor of
/// everything that uses it.
///
/// Nothing about their lifetime actually changes — `app` is the root component and is
/// never unmounted — so this only makes the ownership Dioxus checks match the
/// ownership the design always assumed.
fn root_signal<T: 'static>(init: impl FnOnce() -> T) -> Signal<T> {
    use_hook(|| Signal::new_in_scope(init(), ScopeId::ROOT))
}

/// A signal handed out with its `write` half kept back: `read` and `peek` work
/// exactly as they do on a [`Signal`], and there is no way through it to the value
/// as `&mut`.
///
/// The inner signal is a private field of a type declared in this module, so *this
/// module* is the whole set of code that can mutate what one of these wraps. That
/// is the point: [`AppState::renderer`] and [`AppState::obs`] are the two pieces of
/// state whose mutation has to be paired with a publish, and the pairing is done
/// once here ([`with_engine`]) rather than remembered at two dozen call sites.
///
/// Not `ReadSignal` (dioxus's own read-only wrapper), for one reason: `ReadSignal`
/// boxes the readable into a `CopyValue` owned by whatever scope constructs it, and
/// these are built in `app`'s scope while being read from the `spawn_forever` tasks
/// that live in `ScopeId::ROOT` — the ownership mismatch `root_signal` exists to
/// avoid. Wrapping the root-owned `Signal` keeps its ownership and adds nothing to
/// the read path.
pub struct ReadOnly<T: 'static>(Signal<T>);

// `Signal` is `Copy` whatever it holds, so these are hand-written: a derive would
// demand `T: Copy` and none of what this wraps is.
impl<T: 'static> Clone for ReadOnly<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: 'static> Copy for ReadOnly<T> {}

impl<T: 'static> Readable for ReadOnly<T> {
    type Target = T;
    type Storage = UnsyncStorage;

    fn try_read_unchecked(&self) -> Result<ReadableRef<'static, Self>, BorrowError> {
        self.0.try_read_unchecked()
    }

    fn try_peek_unchecked(&self) -> Result<ReadableRef<'static, Self>, BorrowError> {
        self.0.try_peek_unchecked()
    }

    fn subscribers(&self) -> Subscribers {
        self.0.subscribers()
    }
}

/// Shared `Copy` handle to the app's signals. Provided once by `app` and read back
/// through `use_context` wherever a component needs to reach the engine.
///
/// Built by [`AppState::new`] rather than field-by-field at the call site, so every
/// signal here goes through [`root_signal`] — the one property that has to hold for
/// all of them, and the kind of thing a hand-written literal drifts out of the moment
/// a field is added.
#[derive(Clone, Copy)]
pub struct AppState {
    /// Surface + engine, built asynchronously once the canvas mounts. `None`
    /// until WebGPU init completes. Not `Send` — lives in unsync storage.
    ///
    /// Read freely; to *move* engine state, call [`dispatch`] or [`with_engine`]
    /// (see the module note on [`ReadOnly`]).
    pub renderer: ReadOnly<Option<Renderer>>,
    /// UI-facing engine projection, refreshed after each command — by
    /// [`with_engine`], which is the only thing that can write it.
    pub obs: ReadOnly<Option<ObservableState>>,
    /// Whether WebGPU init has finished and [`publish_renderer`] has handed the
    /// engine over. Set once, never cleared.
    ///
    /// **A separate signal because `renderer.read().is_some()` is not the same
    /// question.** Reading the renderer subscribes to *every write of it*, and
    /// every door into the engine takes it as `&mut` — so an effect that asked the
    /// `Option` re-ran on every command, every pointer sample of a stroke included,
    /// to learn a `bool` that changes once in the life of the page. That is what
    /// the root's thumbnail effect was doing (U2). A signal that only ever moves
    /// false → true wakes its readers exactly when they asked to be woken, and it
    /// says what it means at the site that reads it.
    pub renderer_ready: Signal<bool>,
    /// Whether the user is holding space.
    pub space_down: Signal<bool>,
    /// Whether a canvas gesture is in flight (a stroke, a selection drag, a pan,
    /// or a run of wheel zooming). The floating chrome fades out while it is set,
    /// handing the screen back to the painting — see
    /// [`chrome_class`](crate::layout::chrome_class).
    pub canvas_active: Signal<bool>,
    /// Whether the brush editor dialog is open (rendered at the app root so its
    /// backdrop escapes the panels' `backdrop-filter` containing blocks).
    pub brush_editor_open: Signal<bool>,
    /// Whether the "Save preset" dialog is open. At the app root for the same reason
    /// as the editor above: a panel is a `backdrop-filter` containing block, so a
    /// dialog rendered inside one would be trapped in its box rather than covering
    /// the window.
    pub preset_save_open: Signal<bool>,
    /// Bumped whenever the brush color is set from **outside** the color picker —
    /// today only by the eyedropper.
    ///
    /// The Color panel's picker holds its own Oklab state (out-of-gamut `a`/`b`
    /// survive there but not in the clamped sRGB it reports), so it is *seeded*
    /// rather than driven, and this is how it is told to re-seed. Deliberately not
    /// "the color changed": a pick inside the field would then drag its own marker
    /// back onto the gamut boundary under the user's cursor.
    pub color_epoch: Signal<u64>,
    /// The eyedropper (§18.0.2).
    pub pick: PickState,
    /// The size ring a brush-tuning drag draws over the canvas (§18.1.9), or `None`
    /// when no size drag is in flight.
    ///
    /// Shared state rather than the gesture's own, for [`PickState::dragging`]'s
    /// reason: the drag is the canvas's, but the ring is a sibling overlay of it — a
    /// `<div>` cannot be drawn from inside a `<canvas>`'s handler without somewhere
    /// for both to read.
    pub brush_ring: Signal<Option<BrushRing>>,
    /// The live brush's stroke-smoothing amount, 0..=1 (§6.11). Frontend state
    /// beside the engine's own `BrushParams`, never inside them: the stored
    /// path already embodies the smoothing, so a field there would be one that
    /// replay reads and ignores. It travels with every whole-brush snapshot
    /// through [`presets::Wearable`](crate::presets::Wearable).
    pub smoothing: Signal<f32>,
    /// The tow string on screen while a smoothing brush draws (§6.11), in the
    /// canvas element's own px — `None` when there is nothing to show. Its own
    /// signal for the reason [`brush_ring`](Self::brush_ring) is: only the
    /// little overlay re-renders at pointer rate, never the chrome.
    pub tow: Signal<Option<TowUi>>,
    /// The drag-and-hold drawing assist (§6.9; `crate::input`).
    pub assist: AssistState,
    /// Minimal mode: the chrome over the canvas drops its words and keeps its marks
    /// (§11).
    ///
    /// A client preference like [`AssistState::enabled`], not document state — it is a
    /// way of *looking* at the application, so it never enters the document and is never
    /// replicated, and a collaborator's copy is theirs to set. It does follow *this*
    /// browser between visits (`crate::prefs`).
    ///
    /// Nothing reads this but the app root, which turns it into a class on `.app-root`;
    /// what the class hides is decided in the stylesheet against the `.label` spans the
    /// components emit. Deliberately not a prop threaded through the tree: every
    /// control would then have to be handed a boolean it does nothing with but pass on,
    /// and a control that forgot to would be the one that kept its word.
    pub minimal: Signal<bool>,
    /// The transform gesture in flight (§16.6, §16.8, §16.9): `Some` while the
    /// user is composing a move/scale/flip, a perspective or a warp of the
    /// selected paint. View state — the engine sees only the previews it
    /// produces and the one commit on "Done".
    pub transform: Signal<Option<TransformUi>>,
    /// The gradient bar's gesture in flight (§22.4): `Some` while a ramp's
    /// axis is being composed — over the selection for a fill, over a matte
    /// for its paint. View state on [`transform`]'s model — the engine sees
    /// only previews and the one commit on "Done".
    pub gradient_bar: Signal<Option<GradientUi>>,
    /// The drawing-guide edit mode (§20.5): `Some` while a guide is selected
    /// for composing — the canvas drags the camera, and the Perspective Guide
    /// bar stands in at the bottom. View state through and through: the engine
    /// sees only the `SetGuides` lists the gestures produce.
    pub guide_edit: Signal<Option<GuideEdit>>,
    /// Whether a [`request_paint`] is already waiting on the next animation frame.
    /// The latch that turns any number of paint requests into one paint per frame.
    /// Read and written only from non-component code (`peek`/`set`), so no
    /// component ever subscribes to it.
    pub paint_queued: Signal<bool>,
    /// Everything to do with a shared drawing (§12).
    pub collab: CollabState,
    /// Timeline mode: scrubbing and playing back the history
    /// (§18.2.4; `crate::panels::timeline`).
    pub timeline: TimelineState,
    /// The custom brush-shape library (§6.6; `crate::shapes`).
    pub shapes: ShapesState,
    /// The brush preset library (`crate::presets`), loaded from `localStorage`
    /// at startup like the shape library.
    pub presets: Signal<Vec<crate::presets::PresetEntry>>,
    /// Rendered preset thumbnails and the offscreen rig that generates them
    /// (`crate::thumbs`).
    pub thumbs: crate::thumbs::ThumbState,
    /// The gradient library and its trace mode (§22; `crate::gradients`),
    /// loaded from `localStorage` at startup like the libraries above.
    pub gradients: crate::gradients::GradientsState,
    /// The ten brushes under the hand (§18.1.8; `crate::slots`).
    pub slots: SlotState,
}

/// The quick-brush rack's signals (§18.1.8), grouped because they are one
/// feature's worth of state and because the second is meaningless without the
/// first: a hold names a slot in the rack.
///
/// Root-owned like everything here, and it has to be — the rack is read from the
/// window's own key handlers (`crate::input::bind_shortcuts`), which are bound
/// once for the life of the page and belong to no component's scope.
#[derive(Clone, Copy)]
pub struct SlotState {
    /// What each digit holds; `None` for a slot nobody has filled. Loaded from
    /// `localStorage` at startup like the shape and preset libraries.
    pub brushes: Signal<crate::slots::Rack>,
    /// The hold in flight — `Some` for exactly as long as a number key is down
    /// or the pen's eraser end is on the glass. The rack's overlay is mounted on
    /// this (`slots::SlotOverlay`), and the release reads the brushes it has to
    /// restore from it.
    pub held: Signal<Option<crate::slots::Held>>,
    /// Whether the rack is kept open with no key held — the Panels menu's
    /// "Quick brushes" (§18.1.8). What it buys is a rack that can be *clicked*,
    /// which is the only way to a slot for a hand with no keyboard under it.
    ///
    /// Not persisted, like the panel stack's own visibility (`layout::PanelLayout`
    /// — which this is not part of, the rack being chrome of its own down the
    /// left rather than a panel in the stack): the menu says what is on screen
    /// now, and none of its entries claims to say what will be next time.
    pub pinned: Signal<bool>,
}

/// A drawing guide selected for composing (§20.5): which entry of the
/// engine's guide list the mode edits, and the per-axis locks constraining
/// the canvas drag.
///
/// The locks live here rather than on the guide because they are *gesture*
/// state — a constraint on the hand for the duration of the mode, not a fact
/// about the guide worth keeping (or, later, saving). Leaving the mode
/// releases them.
#[derive(Clone, Copy, PartialEq)]
pub struct GuideEdit {
    /// Index into the engine's guide list ([`ObservableState::guides`]).
    pub index: usize,
    /// World axes held fixed under the orbit drag: one lock constrains the
    /// drag to turning about that axis, two pin the frame entirely
    /// ([`PerspectiveGuide::dragged`](stark_engine::PerspectiveGuide::dragged)).
    pub locked: [bool; 3],
}

/// The drag-and-hold drawing assist's signals (§6.9).
///
/// The engine owns what a hold *means* ([`GestureCommand::Hold`](stark_engine::command::GestureCommand));
/// what lives here is the other half — noticing that the pointer has stopped, which is
/// gesture feel and needs a clock the engine deliberately does not have (§7).
#[derive(Clone, Copy)]
pub struct AssistState {
    /// Whether a held pointer snaps the stroke at all. On by default, and off is a
    /// real answer: assist changes what an ordinary stroke does, so somebody who wants
    /// their line left crooked has to be able to say so (`crate::settings`).
    pub enabled: Signal<bool>,
    /// The dwell being watched — `Some` for exactly as long as a paint gesture is in
    /// flight, and clearing it is what stops the watcher below.
    pub dwell: Signal<Option<Dwell>>,
    /// The watcher counting it down. Cancelled and replaced per gesture, like the
    /// collaboration pumps.
    pub task: Signal<Option<Task>>,
}

/// A pointer that has stopped moving, as the assist watcher reads it.
#[derive(Clone, Copy, PartialEq)]
pub struct Dwell {
    /// Where it stopped, in **element (CSS) pixels**.
    ///
    /// Screen space and not canvas space, because holding still is a fact about the
    /// hand: measured on the canvas, the same tremor would count as a hold at one zoom
    /// level and as movement at another, and the threshold would mean something
    /// different every time the view changed.
    pub at: Vec2,
    /// When it got there, on the monotonic clock (`crate::collab::now_seconds`).
    pub since: f64,
    /// Whether this dwell has already been reported.
    ///
    /// A latch rather than an exit, because a gesture may earn several holds: the
    /// pause a pen makes on touching down before the hand starts moving is one, and
    /// giving up there would disarm the assist for the stroke that followed it. So the
    /// watcher latches, and movement past the slop is what clears it.
    pub fired: bool,
}

/// What a brush-tuning drag's size indicator draws (§18.1.9): the size being asked
/// for, and the size the drag started from, about the point it pressed on.
///
/// **Screen px, not canvas px** — deliberately a drawing instruction rather than a
/// statement about the brush. The gesture holds the one zoom it measures against
/// (`input::TuneDrag::zoom`), so converting there means the ring and the radius it
/// reports cannot be scaled by two different numbers; and it leaves the overlay pure
/// layout, with no view to read and nothing to re-render it when the engine writes.
///
/// A circle in canvas space is still a circle on screen at any angle or handedness, so
/// a radius through the zoom is the whole of the transform: this needs no matrix, which
/// is the one thing that makes a `<div>` a fair way to draw it.
#[derive(Copy, Clone, PartialEq)]
pub struct BrushRing {
    /// The press position in page px — where the ring is centred, and the one point a
    /// gesture agrees on however far it has wandered since.
    pub at: Vec2,
    /// The radius the brush had when the drag began, screen px. The reference: without
    /// it the ring says how big the brush is about to be and nothing about whether that
    /// is bigger or smaller than what the last stroke was made with.
    pub was: f32,
    /// The radius being asked for now, screen px.
    pub now: f32,
}

/// The tow string as the overlay draws it (§6.11): the towed tip, the pointer,
/// and the rope, all in the canvas element's own px.
///
/// **Screen px, not canvas px**, on [`BrushRing`]'s own argument: a drawing
/// instruction rather than a statement about the stroke. The conversion happens
/// where the engine is read, against the one view the gesture holds — a stroke
/// cannot outlive its view, since a pinch cancels it — and the overlay stays
/// pure layout.
#[derive(Copy, Clone, PartialEq)]
pub struct TowUi {
    /// Where the mark is being laid — the towed tip.
    pub tip: Vec2,
    /// Where the hand is — the pointer the string runs to.
    pub target: Vec2,
    /// The string's length. `|target − tip|` short of it is slack, and the
    /// overlay shows the difference as sag.
    pub rope: f32,
}

/// The custom brush-shape library's signals (`crate::shapes`). Root-owned:
/// imports are started from the brush editor's modal scope but must survive
/// its close.
#[derive(Clone, Copy)]
pub struct ShapesState {
    /// Library entries, loaded from `localStorage` at startup.
    pub entries: Signal<Vec<crate::shapes::ShapeEntry>>,
    /// A transient line under the shape gallery: import errors, or the
    /// "inverted a dark-on-light image" explanation. `None` when quiet.
    pub notice: Signal<Option<String>>,
}

impl AppState {
    /// Build the app's state. Call once, from the root component.
    pub fn new() -> Self {
        Self {
            renderer: ReadOnly(root_signal(|| None)),
            obs: ReadOnly(root_signal(|| None)),
            renderer_ready: root_signal(|| false),
            space_down: root_signal(|| false),
            canvas_active: root_signal(|| false),
            brush_editor_open: root_signal(|| false),
            preset_save_open: root_signal(|| false),
            color_epoch: root_signal(|| 0),
            pick: PickState {
                scope: root_signal(PickScope::default),
                radius: root_signal(|| 0),
                busy: root_signal(|| false),
                alt_down: root_signal(|| false),
                dragging: root_signal(|| false),
            },
            brush_ring: root_signal(|| None),
            smoothing: root_signal(|| 0.0),
            tow: root_signal(|| None),
            assist: AssistState {
                enabled: root_signal(|| Prefs::default().assist),
                dwell: root_signal(|| None),
                task: root_signal(|| None),
            },
            // Seeded from the preference defaults rather than written out again here:
            // `prefs::load` overwrites both of these at app start, so a default stated
            // in this file would be the one that never applies (`crate::prefs`).
            minimal: root_signal(|| Prefs::default().minimal),
            transform: root_signal(|| None),
            gradient_bar: root_signal(|| None),
            guide_edit: root_signal(|| None),
            paint_queued: root_signal(|| false),
            collab: CollabState {
                session: root_signal(|| None),
                ticket: root_signal(|| None),
                phase: root_signal(collab::CollabPhase::default),
                error: root_signal(|| None),
                peers: root_signal(Vec::new),
                links: root_signal(Vec::new),
                pump: root_signal(|| None),
                presence: root_signal(|| None),
            },
            timeline: TimelineState {
                open: root_signal(|| false),
                playing: root_signal(|| false),
                speed: root_signal(|| 1.0),
                task: root_signal(|| None),
            },
            shapes: ShapesState {
                entries: root_signal(Vec::new),
                notice: root_signal(|| None),
            },
            presets: root_signal(Vec::new),
            thumbs: crate::thumbs::ThumbState {
                cache: root_signal(Vec::new),
                rig: root_signal(|| None),
                shared: root_signal(|| None),
                busy: root_signal(|| false),
            },
            gradients: crate::gradients::GradientsState {
                entries: root_signal(Vec::new),
                selected: root_signal(|| None),
                armed: root_signal(|| false),
                busy: root_signal(|| false),
                notice: root_signal(|| None),
            },
            slots: SlotState {
                brushes: root_signal(|| [None; crate::slots::COUNT]),
                held: root_signal(|| None),
                pinned: root_signal(|| false),
            },
        }
    }
}

/// Subscribe to **one slice** of the projection instead of to all of it.
///
/// [`AppState::obs`] is a single signal carrying everything from the view — which
/// moves at pointer rate — to the history budget, which moves about once a session.
/// A `Signal` write marks every subscriber dirty with no equality check, so a
/// component that calls `state.obs.read()` in its body re-renders on **every**
/// command: a pan, a brush-size drag, a transform preview, each sample of an
/// eyedropper drag. Most of the chrome does not depend on any of those.
///
/// A memo is the seam. It recomputes when `obs` moves, but only *propagates* when
/// the slice it selected actually changed (`Memo::recompute` compares before it
/// writes), so a panel reading its layer list through one is woken by commits and
/// slept through pans. `panels::navigator` and `panels::timeline` already did this
/// by hand; this is the same move, named, so a panel gets it by reading the
/// projection the ordinary way rather than by remembering to.
///
/// A hook: call unconditionally, like any `use_*`. `slice` runs against the whole
/// projection, so a component wanting several fields should take **one** memo of a
/// tuple rather than one memo each — the fields it reads change together far more
/// often than not, and one memo is one comparison.
///
/// `None` before WebGPU init has published a projection, which every caller may
/// simply let fall through — the same shape they already handle.
pub fn use_obs<T>(
    state: AppState,
    slice: impl Fn(&ObservableState) -> T + 'static,
) -> Memo<Option<T>>
where
    T: Clone + PartialEq + 'static,
{
    use_memo(move || state.obs.read().as_ref().map(&slice))
}

/// The gradient gesture being composed on the shared gradient bar (§22.4):
/// what the ramp lands on, how the composing drag is read, and the drag itself.
///
/// The drag is kept as its two raw points and the axis **derived** per kind —
/// linear reads them as from→to, radial as centre and reach — so switching
/// kinds on the bar reinterprets the drag the hand already made instead of
/// throwing it away.
#[derive(Clone, PartialEq)]
pub struct GradientUi {
    /// What "Done" commits — a fill of the selection, or a matte's paint. One
    /// bar, one catcher, two targets: the interface for laying a ramp is the
    /// same wherever the ramp lands.
    pub target: GradientTarget,
    /// How the drag becomes an axis — the bar's Linear/Radial chips.
    pub kind: GradientAxisKind,
    /// The composing drag, canvas space: anchor and current end. `None` until
    /// the first drag — there is no axis to preview yet.
    pub drag: Option<(Vec2, Vec2)>,
}

/// What the gradient bar's ramp lands on (§22.4).
#[derive(Clone, PartialEq)]
pub enum GradientTarget {
    /// Fill the selection on `layer`. The ramp is read live from the library
    /// (the panel's highlighted row), so a click there re-previews.
    ///
    /// It lays opaque paint through the mask, so how strongly it lands is the
    /// selection's business and there is nothing here to capture (§6.8).
    Fill { layer: LayerId },
    /// Repaint the matte `layer` (§15.4). The ramp rides the target — seeded
    /// from the matte's own paint, so re-composing an old gradient's axis does
    /// not silently swap its colors for the library's current row; a library
    /// click replaces it deliberately.
    Matte {
        layer: LayerId,
        gradient: stark_model::Gradient,
    },
}

/// How a gradient drag is read as an axis (§22.4).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GradientAxisKind {
    Linear,
    Radial,
}

impl GradientUi {
    /// The axis the current drag composes, or `None` before the first drag.
    pub fn axis(&self) -> Option<stark_engine::document::GradientAxis> {
        let (from, to) = self.drag?;
        Some(match self.kind {
            GradientAxisKind::Linear => stark_engine::document::GradientAxis::Linear { from, to },
            GradientAxisKind::Radial => stark_engine::document::GradientAxis::Radial {
                center: from,
                radius: from.distance(to),
            },
        })
    }
}

/// Timeline mode's signals (§18.2.4), grouped because they are one
/// mode's worth of view state: the mode itself, whether it is playing, and how fast.
///
/// None of this is the *playhead* — that lives in the engine's timeline, where undo
/// and redo already move it, and is read back through
/// [`Engine::scrub_range`](stark_engine::Engine::scrub_range). A copy here is exactly
/// the copy that would go stale the moment a stroke, an undo or a load moved the
/// history underneath it.
#[derive(Clone, Copy)]
pub struct TimelineState {
    /// Whether the mode is on — which is to say whether the bar is mounted.
    pub open: Signal<bool>,
    /// Whether playback is running. The source of truth for the transport button;
    /// the loop below reads it every tick and stops when it goes false, so anything
    /// that wants playback to end only has to clear this.
    pub playing: Signal<bool>,
    /// Playback rate, as a multiple of
    /// [`BASE_RATE`](crate::panels::timeline::BASE_RATE).
    pub speed: Signal<f32>,
    /// The playback loop, so a second Play cannot start a second one and closing
    /// the mode can cancel it. Root-owned like everything here: the loop outlives
    /// the bar that started it (the bar unmounts when the mode closes), which is
    /// the case `spawn_forever` exists for.
    pub task: Signal<Option<Task>>,
}

/// The eyedropper's signals (§18.0.2), grouped because they are one
/// feature's worth of view state: the two options a sample is taken with, the latch
/// that keeps Alt+drag from asking for samples faster than the GPU answers them, and
/// the two flags that say where in the gesture we are.
///
/// The options live here rather than in the engine because nothing in the engine
/// reads them between calls — [`Engine::pick_color`](stark_engine::Engine::pick_color)
/// is a request and they are its arguments, so a copy projected back through
/// `observe()` would be state with no owner.
#[derive(Clone, Copy)]
pub struct PickState {
    /// Which layers a sample comes off, and whether the canvas stands behind them.
    pub scope: Signal<PickScope>,
    /// Half-width of the averaged square, in canvas px (0 = point sample).
    pub radius: Signal<u32>,
    /// Whether a sample is in flight — see [`crate::input::pick_color`].
    pub busy: Signal<bool>,
    /// Whether Alt is held. Only ever *shown*: the canvas wears the eyedropper
    /// cursor while it is and the options bar comes up, so the modifier announces
    /// itself before it is used — which is the whole reason a modifier binding is
    /// discoverable at all.
    pub alt_down: Signal<bool>,
    /// Whether an Alt+drag is actually sampling. Shared rather than local to the
    /// canvas, unlike `drawing`/`panning`, because the options bar is mounted on
    /// *armed but not yet dragging* and so has to be able to tell the two apart.
    pub dragging: Signal<bool>,
}

/// Which of the eyedropper's three sources the bar has selected (§18.0.2).
///
/// The *choice*, not [`PickSource`](stark_engine::PickSource): which layer "this layer"
/// means is resolved against the selected layer at the moment of the sample
/// ([`crate::input::pick_color`]), so the bar cannot be left holding a layer id whose
/// layer has since been deleted — the same reason the radius is a number here and a
/// clamped one in the engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PickScope {
    /// Every visible layer — the paint the canvas shows.
    #[default]
    AllLayers,
    /// Every visible layer over the substrate color, so bare canvas and thin paint
    /// answer with what the eye sees rather than with nothing and with the glaze.
    AllLayersAndCanvas,
    /// The selected layer alone, ignoring anything over or under it.
    ThisLayer,
}

/// The shared-session signals, grouped because they share one lifecycle: they are
/// set together when a session starts and cleared together when it ends, and
/// nothing outside [`crate::collab`] should be writing them piecemeal.
#[derive(Clone, Copy)]
pub struct CollabState {
    /// The live session, if any. `!Send` iroh handles live in unsync storage
    /// beside the renderer.
    pub session: Signal<Option<stark_net::CollabSession>>,
    /// The shareable ticket string, while hosting/joined.
    pub ticket: Signal<Option<String>>,
    /// Where the session lifecycle stands (drives the dialog + rail badge).
    pub phase: Signal<collab::CollabPhase>,
    /// The last share/join failure, surfaced in the dialog.
    pub error: Signal<Option<String>>,
    /// Who else is in the session, refreshed by the presence pump
    /// (§17.4). Its own signal rather than a field of `obs`: it changes on every remote
    /// pointer move, and re-running the whole component tree at that rate to move a
    /// cursor would be absurd.
    pub peers: Signal<Vec<crate::render::PeerInfo>>,
    /// How each directly-connected peer is reached — WebRTC, hole-punched UDP,
    /// or an iroh relay — polled off the mesh by the presence pump on a slow
    /// cadence (links change on the order of seconds, not frames). Peers in the
    /// roster but absent here have no direct connection; the mesh forwards
    /// their traffic. Read by the session dialog.
    pub links: Signal<Vec<stark_net::PeerLink>>,
    /// The incoming-event pump for `session`. Its lifetime is tied to the
    /// session's: `collab::install` replaces it, [`crate::collab::leave`]
    /// cancels it.
    pub pump: Signal<Option<Task>>,
    /// The outgoing presence pump — a fixed-cadence loop that drains the engine's
    /// presence latch onto the mesh (§17.5). Separate from `pump`
    /// because it is a *pull* on a timer rather than a reaction to arriving events,
    /// but shares the same lifecycle.
    pub presence: Signal<Option<Task>>,
}

impl CollabState {
    /// Whether a live session exists — i.e. whether anyone is on the other end
    /// of presence-only commands. `peek`: asked from event handlers at pointer
    /// rate, and nothing there should subscribe.
    pub fn active(&self) -> bool {
        self.session.peek().is_some()
    }
}

/// Repaint the canvas surface on the **next animation frame**, coalescing however
/// many requests land before it into one paint.
///
/// This is deliberately not `Renderer::paint` called inline. Paint requests arrive
/// per local pointer sample ([`dispatch`]) *and* per peer gesture frame (the collab
/// pump, ~30 Hz × peers) — with several clients drawing, that sums to hundreds of
/// full-frame composites a second submitted to a display that shows ~60. Nothing
/// bounds the GPU queue, so it grows for as long as the strokes last and input
/// latency climbs with it. The engine's state still integrates per event
/// (coalescing there would drop samples); it is only the *presentation* that is a
/// snapshot, and painting the latest state once per frame shows exactly what
/// painting it per event would have — minus the queue.
pub fn request_paint(state: AppState) {
    if gpu_lost(state) {
        return;
    }
    let mut queued = state.paint_queued;
    if *queued.peek() {
        return;
    }
    queued.set(true);
    schedule_paint(state);
}

/// Whether the GPU has died and the app has therefore stopped painting and
/// dispatching (§5).
///
/// **Asked of the projection, not of the engine**, though both can answer
/// ([`Renderer::gpu_healthy`]). The projection is what the report is mounted on
/// (`crate::failure`), so keying the shutdown on the same field makes "the app has
/// stopped" and "the artist has been told" one fact rather than two that could
/// disagree — a stop with no dialog is a frozen canvas, and a dialog over a running
/// app is a lie.
///
/// `peek`: called from the doors below, which are not renders, and from event
/// handlers. Nothing here should widen what a pointer move wakes.
pub fn gpu_lost(state: AppState) -> bool {
    state
        .obs
        .peek()
        .as_ref()
        .is_some_and(|o| o.gpu_failure.is_some())
}

/// One animation-frame hop of [`request_paint`]: paint, unless the GPU has
/// fallen behind — then hold the latch and try again next frame.
///
/// Directly in the rAF callback, not a task the rAF wakes. A woken task
/// resumes in the microtask drain, two scheduler hops after the callback —
/// and a *dioxus* task further waits out a VDOM render of whatever scopes
/// are dirty by then, since the scheduler polls tasks only once no scope is.
/// Any of that slipping past the frame's rendering steps shows the previous
/// frame's canvas for a whole display interval. The callback itself runs in
/// the animation phase, ahead of the rendering steps by definition, so the
/// frame that fires it is the frame that shows it. The closure is a plain
/// move of root-owned signals, so no component scope is involved and there
/// is nothing here to die with one (see the module note on `root_signal`).
fn schedule_paint(state: AppState) {
    crate::platform::on_animation_frame(move || {
        let mut renderer = state.renderer.0;
        let mut guard = renderer.write();
        // The GPU still owes more than a full pipeline of painted frames
        // (`Renderer::gpu_behind`): submitting another would deepen the queue,
        // not the picture. Skip the frame — with the latch *held*, so the next
        // rAF re-checks — and let the work drain. Nothing is lost: mutations
        // keep integrating per event and only mark the fold stale, so the
        // first paint after the drain folds everything accrued at once and
        // shows exactly what the skipped frames would have.
        if guard.as_ref().is_some_and(Renderer::gpu_behind) {
            {
                // **The count is the datum here, not the duration** — this row is
                // how often the back-pressure above actually fired, which is the one
                // thing about it no benchmark can see (a bench drains the device by
                // hand and never queues a second frame). Read against `frame`'s
                // count: a session where the two are comparable is a session
                // spending half its animation frames waiting for the GPU.
                //
                // A block holding nothing but the span, because there is nothing
                // here to time — it closes on the brace, and its duration is noise.
                stark_engine::timing::span!("frame.skipped");
            }
            drop(guard);
            schedule_paint(state);
            return;
        }
        // Everything a shown frame costs this client, from inside the animation-frame
        // callback: the engine's own `render.view` plus the surface acquire and the
        // present around it. Its *count* over the window is the frame rate the app
        // actually achieved, which is the top line of the end-to-end story and the
        // number every phase row underneath is read against.
        stark_engine::timing::span!("frame");
        let mut queued = state.paint_queued;
        queued.set(false);
        // A device that has died renders nothing, and **this** is where that has to
        // be noticed (§5): painting is the one thing that goes on happening when no
        // command is being dispatched, so a failure suffered between commands — a
        // driver reset while the artist is looking at their work — would otherwise
        // sit undiscovered until they next touched something. Publishing is what
        // raises the report (`crate::failure`); after it [`gpu_lost`] is true, and
        // nothing asks for another frame or dispatches another command.
        let failed = match guard.as_mut() {
            Some(r) if !r.gpu_healthy() => true,
            Some(r) => {
                r.paint();
                false
            }
            None => false,
        };
        // Before the publish, which reads the renderer to take its snapshot while
        // this closure is holding it as `&mut`.
        drop(guard);
        if failed {
            publish_observation(state);
        }
    });
}

/// Reach the engine as `&mut`, publish what it now looks like, and repaint —
/// `None` when WebGPU init has yet to hand one over, which every caller may
/// simply let fall through.
///
/// The publish is attached to the *door* rather than to the caller's memory, and
/// that is the whole design: `observe` is what the chrome renders from, so an
/// engine mutation that skips it leaves the chrome asserting something the engine
/// stopped believing — visible as a control that snaps back to its old value and
/// stays there (§4, §7). Two shipped bugs of exactly that shape are why this is a
/// function and `renderer` is a [`ReadOnly`].
///
/// For engine entry points that are not commands — opening a document, joining a
/// session. A command goes through [`dispatch`], which is this plus the broadcast.
///
/// **`None` also once the GPU has died** ([`gpu_lost`], §5), which is the same
/// answer callers already handle for "the renderer is not up yet" — deliberately,
/// because it is the same fact from the caller's side: there is no engine to move.
/// Both doors are guarded rather than the call sites, so "stop dispatching after a
/// device failure" is a property of the seam and not a rule 82 of them could
/// forget. The command that *causes* the failure still runs and still publishes,
/// since the health is only set while `f` is executing — which is what makes the
/// report appear at all.
pub fn with_engine<R>(state: AppState, f: impl FnOnce(&mut Renderer) -> R) -> Option<R> {
    if gpu_lost(state) {
        return None;
    }
    let (mut renderer, mut obs) = (state.renderer.0, state.obs.0);
    let out = {
        let mut guard = renderer.write();
        let r = guard.as_mut()?;
        let out = f(r);
        // Inside the guard, as `dispatch` has always done it: `obs.set` marks
        // subscribers dirty but renders nothing synchronously, so no reader can
        // observe the renderer mid-borrow.
        //
        // **Published only when it moved.** A `Signal` write dirties every
        // subscriber whether or not the value changed, so a command the engine
        // declined — a slider dragged back to the value the document already
        // holds, a rename to the name a layer already has, a selection command
        // with nothing to select — would otherwise wake every memo in the chrome
        // to report the answer they are already showing. `ObservableState` is
        // `PartialEq` for exactly this, and the comparison is cheap: the layer
        // list is shared, so the expensive field answers by pointer
        // (`stark_engine::Layers`).
        //
        // The comparison is bound to a local before the `if` rather than written
        // inline in its condition. A guard held in a condition outliving the body
        // is the shape that has borrow-panicked in this crate before — the reason
        // `modes::leave` takes its `write().take()` into a local — and inline here
        // it would be a read of `obs` live across the write below.
        let snapshot = r.observe();
        let moved = obs.peek().as_ref() != Some(&snapshot);
        if moved {
            obs.set(Some(snapshot));
        }
        out
    };
    request_paint(state);
    Some(out)
}

/// Reach the engine as `&mut` **without** publishing anything or asking for a
/// frame — for work that cannot change what `observe` projects.
///
/// That is a real and large category, which is why this exists rather than a
/// `dispatch` everywhere: rendering, the readbacks (export, the eyedropper, the
/// navigator's miniature), draining the outbox and the presence tick, and
/// installing asset bytes an action will later name. None of it moves a value the
/// chrome shows, and several run at pointer or frame rate where the `observe` walk
/// and the VDOM diff behind a publish would be pure cost (see [`dispatch_sample`]).
///
/// If a closure here calls `Renderer::process`, it is in the wrong door — that is a
/// command, and commands publish. Callers wanting the frame but not the publish ask
/// for it with [`request_paint`], which several below do.
///
/// Guarded by [`gpu_lost`] exactly as [`with_engine`] is, and for its reason: the
/// readbacks, the collaboration pumps and the presence tick all come through here,
/// and none of them can produce anything on a device that has stopped answering.
/// **Saving is unaffected**, because saving is not a GPU operation and does not use
/// this door — `files::save_document` reads the renderer directly and serializes an
/// action log that lives in ordinary memory (§8). That is the whole reason the
/// report is worth raising rather than merely logging.
pub fn with_engine_quiet<R>(state: AppState, f: impl FnOnce(&mut Renderer) -> R) -> Option<R> {
    if gpu_lost(state) {
        return None;
    }
    let mut renderer = state.renderer.0;
    let mut guard = renderer.write();
    let r = guard.as_mut()?;
    Some(f(r))
}

/// Publish the engine's current projection without having mutated anything.
///
/// One caller, and it earns it: the collaboration pump takes the snapshot for the
/// events that commit and skips it for the ones that arrive at presence rate
/// (§17.5), so *which* event happened decides, not the fact of holding the engine.
pub fn publish_observation(state: AppState) {
    let mut obs = state.obs;
    let Some(snapshot) = state.renderer.peek().as_ref().map(Renderer::observe) else {
        return;
    };
    // Only when it moved, on [`with_engine`]'s argument and by the same
    // comparison — one rule for publishing rather than two.
    let moved = obs.peek().as_ref() != Some(&snapshot);
    if moved {
        obs.0.set(Some(snapshot));
    }
}

/// Hand the freshly-built engine to the app and publish its opening projection.
///
/// The order is the point: the projection is taken from `r` *before* it is moved
/// into the signal, because everything that reads the signal is written on the
/// assumption that a renderer being there means the chrome already describes it.
pub fn publish_renderer(state: AppState, r: Renderer) {
    let mut renderer = state.renderer;
    let mut obs = state.obs;
    obs.0.set(Some(r.observe()));
    // The thumbnail generator's half of the engine, published here rather than
    // fetched from the renderer signal on demand: it needs the device and the
    // pipelines, not the canvas, and holding those directly is what keeps it from
    // borrowing a live renderer to make a picture of a brush (`crate::thumbs`).
    let mut shared = state.thumbs.shared;
    shared.set(Some(r.engine_shared()));
    renderer.0.set(Some(r));
    // Last, and the announcement everything else waits on: the readiness flag is
    // set only once the renderer is in the signal and the chrome can already
    // describe it, so a reader woken by it finds the whole arrangement in place
    // (see [`AppState::renderer_ready`]).
    let mut ready = state.renderer_ready;
    ready.set(true);
}

/// Apply a command, request a repaint, and refresh the observable snapshot.
/// In a shared session, whatever the command committed is then broadcast.
pub fn dispatch(state: AppState, command: impl Into<InputCommand>) {
    with_engine(state, |r| r.process(command));
    collab::flush_outbox(state);
}

/// Apply a pointer-rate sample of an in-flight gesture: integrate and repaint,
/// skipping the observable refresh and the outbox flush [`dispatch`] does.
///
/// A mid-gesture sample changes what the canvas shows but nothing the chrome
/// reads: the committed document, the layer list and the undo flags all stand
/// until the gesture ends, so `Engine::observe` would walk the layer tree to
/// produce the same answer — and `obs.set` marks every subscriber dirty
/// regardless, which puts a full VDOM diff of the chrome *ahead* of the rAF
/// registration on every sample (the scheduler polls tasks only once no scope
/// is dirty). Nothing commits mid-gesture either, so there is no outbox to
/// flush; the gesture's End goes through [`dispatch`], which refreshes the
/// observable and broadcasts whatever the commit banked.
pub fn dispatch_sample(state: AppState, command: impl Into<InputCommand>) {
    // The other end-to-end row, and the partner to `frame`: its count over the
    // window is how many pointer reports a second actually reached the engine. That
    // is a *measurement*, not a setting — the browser coalesces pointer moves to
    // roughly one a frame and `input::samples` un-coalesces them, so whether a
    // 240 Hz pen is being heard at 240 Hz or at 60 is a question only this row
    // answers. It contains `input.fit`, so the difference between the two is the
    // engine door and the paint request around the work.
    stark_engine::timing::span!("input.sample");
    with_engine_quiet(state, |r| r.process(command));
    request_paint(state);
}

/// Apply a command **without** repainting, refreshing the observable, or
/// broadcasting — for the ones that change nothing this client can see.
///
/// [`PeerCommand::SetCursor`](stark_engine::command::PeerCommand::SetCursor) is the
/// case that needs it: it arrives at pointer rate, and drawing our own cursor is the
/// browser's job, so the full `dispatch` would repaint the canvas hundreds of times a
/// second to show nothing. The value still reaches peers, because the presence pump
/// reads it off the engine on its own cadence (§17.5) rather than being
/// pushed from here.
pub fn dispatch_quiet(state: AppState, command: impl Into<InputCommand>) {
    with_engine_quiet(state, |r| r.process(command));
}

/// Resize the surface/engine, then repaint — inline, not [`request_paint`]: the
/// surface was just reconfigured, and a frame of the old size shown until the next
/// rAF is a visible flash. Resize arrives at layout rate, which cannot flood.
///
/// A report that arrives before the renderer exists is *dropped*, and the app start
/// is full of them — the canvas is laid out long before WebGPU init has finished
/// fetching its assets. That is safe only because the last thing init does before
/// publishing the renderer is re-read the element itself
/// ([`Renderer::sync_to_canvas`](crate::render::Renderer::sync_to_canvas)): the size
/// is recovered from the DOM rather than replayed from here, so nothing needs to be
/// queued. Move that call and the viewport goes stale again.
///
/// The paint stays inline *inside* the closure for the reason above; the frame
/// [`with_engine`] then queues is a redundant repaint of the same state, which
/// layout rate can well afford and which keeps the publish where every other
/// mutation has it.
pub fn resize(state: AppState, width: u32, height: u32) {
    with_engine(state, |r| {
        r.resize(width, height);
        r.paint();
    });
}

/// Read the current brush, mutate a copy, and commit it (releasing the `obs`
/// read guard before `dispatch` writes — avoids an AlreadyBorrowed panic).
///
/// **The loud door, deliberately, even though this runs at pointer rate** — the
/// brush-tuning drag and the eyedropper both come through here on every move.
/// [`dispatch_sample`] would be wrong: a tune drag's whole answer is read off the
/// Brush panel's sliders (`input::Tune`), and the eyedropper's off the Color
/// panel's swatch, so the publish is not overhead here but the point of the
/// gesture. What made it expensive was that *every* panel woke for it; that is
/// [`use_obs`]'s business, and the panels that do not show a brush now sleep
/// through this. What is left is `observe`'s own walk of the layer tree, which
/// only splitting the projection by cadence can remove (U1).
pub fn update_brush(state: AppState, f: impl FnOnce(&mut BrushParams)) {
    let brush = state.obs.read().as_ref().map(|o| o.brush);
    if let Some(mut brush) = brush {
        f(&mut brush);
        dispatch(state, ViewCommand::SetBrush(brush));
    }
}
