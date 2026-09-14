//! The app's shared state handle and the command path everything funnels through.
//!
//! The engine and its projection sit in [`ReadOnly`] handles, so `&mut Renderer` is
//! reachable only through this module's doors: [`dispatch`] (the single seam, §4),
//! [`with_engine`] and [`replace_document`], which publish, and [`with_engine_quiet`],
//! which does not. A mutation that skips the publish leaves the chrome stale (§4, §7).
//!
//! A signal guard held across a write of that signal, including a door's publish of the
//! projection, is a borrow panic: read into a local that drops first.

use dioxus::dioxus_core::{Runtime, Subscribers};
use dioxus::prelude::*;

use crate::collab;
use crate::render::Renderer;
use stark_engine::ObservableState;
use stark_engine::command::InputCommand;
use stark_engine::command::{GestureCommand, InputSample, Tool, ViewCommand};
use stark_model::AssetNeed;
use stark_model::geom::Vec2;
use stark_ui::commands::VisibilityToggle;
use stark_ui::desk::{Replaced, Replacement};
use stark_ui::prefs::{ChromeHiding, Prefs};

/// Create one of [`AppState`]'s signals, owned by [`ScopeId::ROOT`] rather than by the
/// component that declares it.
///
/// They are read from detached `spawn_forever` tasks (the renderer init, the collab
/// pumps), which live in `ScopeId::ROOT`, not a descendant of `app`, and Dioxus warns when
/// a signal is read outside its owner's subtree. `app` is never unmounted, so no lifetime
/// changes. Crate-visible for each group's own `new` and for a gesture that holds its own
/// state (§25.3) but is written from a detached task, such as `input::PickMove`. A hook:
/// call it unconditionally.
pub(crate) fn root_signal<T: 'static>(init: impl FnOnce() -> T) -> Signal<T> {
    use_hook(|| Signal::new_in_scope(init(), ScopeId::ROOT))
}

/// A [`Signal`] with its `write` half withheld, so only this module can mutate
/// [`Signals::renderer`] and [`Signals::obs`], and each of its doors decides whether to
/// publish.
///
/// Not dioxus's `ReadSignal`, which boxes the readable into a `CopyValue` owned by the
/// constructing scope (`app`), while these are read from tasks in `ScopeId::ROOT`.
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

/// Shared `Copy` handle to the app's signals, provided once by `app` and read back
/// through `use_context`.
///
/// One pointer wide, because [`Signals`] is kilobytes and this is passed by value to
/// hundreds of functions. `Signals` is leaked once, on the root's first render; the root
/// is never unmounted. [`Deref`](std::ops::Deref) but not `DerefMut`, so
/// `state.foo.set(…)` does not compile: copy the signal into a local first.
#[derive(Clone, Copy)]
pub struct AppState(&'static Signals);

impl std::ops::Deref for AppState {
    type Target = Signals;

    fn deref(&self) -> &Signals {
        self.0
    }
}

/// Every signal the app holds; what [`AppState`] is a handle to.
///
/// Built by [`AppState::new`], so that every signal here goes through [`root_signal`].
pub struct Signals {
    /// Surface + engine, built asynchronously once the canvas mounts; `None` until WebGPU
    /// init completes. Not `Send`, so unsync storage. Mutate only through this module's
    /// doors (see the module doc).
    pub renderer: ReadOnly<Option<Renderer>>,
    /// UI-facing engine projection, refreshed after each command. Written only by this
    /// module ([`publish`] and [`publish_renderer`]).
    pub obs: ReadOnly<Option<ObservableState>>,
    /// Why the app could not start, if it could not (§5, `crate::failure`). Set once by
    /// the startup task, never cleared.
    ///
    /// Not [`obs`](Self::obs)'s `gpu_failure`, a device that died with a document behind
    /// it (whose report offers to save): this device never arrived, so there is no engine
    /// to publish a projection.
    pub startup_failure: Signal<Option<crate::render::StartupFailure>>,
    /// Whether [`publish_renderer`] has handed the engine over. Set once, never cleared.
    /// Its own signal because reading [`renderer`](Self::renderer) subscribes to every
    /// write of it, and every door into the engine writes it, stroke samples included.
    pub renderer_ready: Signal<bool>,
    /// How many shipped assets (substrate maps fetched on first use) have landed in the
    /// main engine since it was published (`crate::shipped::landed`). Wakes a picker
    /// listing them, for [`renderer_ready`](Self::renderer_ready)'s reason.
    pub shipped_landed: Signal<u32>,
    /// The `doc_revision` this document was last written out at, by Save or Export; zero
    /// until then (`crate::files`). With the projection's `edited`, answers the unload
    /// guard. The engine cannot hold it: a file leaving the browser is not an event the
    /// document can notice. Not reset by a replacement, which is safe because `doc_revision`
    /// counts up for the page's life, past zero before the opening document exists.
    pub written_revision: Signal<u64>,
    /// Whether the user is holding space.
    pub space_down: Signal<bool>,
    /// The modifiers currently held, tracked off key events and corrected off every
    /// event's modifier set, so a change made while unfocused is caught up on the next
    /// keystroke. Only ever shown: the resting cursor and the eyedropper's options bar ask
    /// the drag table what a press under these would open (`stark_ui::drags::armed`).
    pub held_mods: Signal<stark_ui::keys::Mods>,
    /// Whether a canvas gesture is in flight (a stroke, a selection drag, a pan,
    /// or a run of wheel zooming). The floating chrome fades out while it is set —
    /// see [`chrome_dimmed`](crate::layout::chrome_dimmed).
    pub canvas_active: Signal<bool>,
    /// Whether the panel stack is asleep: still faded after the gesture ended, until the
    /// pointer reaches into its column (§11).
    ///
    /// Set by [`end_interaction`](crate::input::end_interaction) only where the fade was
    /// in force, so a gesture that keeps the chrome up (the eyedropper, brush tuning) does
    /// not put it to sleep. Cleared by the pointer arriving (`layout::PanelStack`) and by
    /// opening a panel (`layout::open_panel`).
    ///
    /// Only the stack sleeps: the rest of what [`chrome_dimmed`](crate::layout::chrome_dimmed)
    /// dims (a mode's bar, transform handles, the rail) lives outside this column, so a
    /// sleep woken only here would leave it faded while the pointer reaches for it.
    pub panels_asleep: Signal<bool>,
    /// This browser's preferences (§25.6), seeded from the stored record. Written only by
    /// [`prefs::set`](crate::prefs::set). For the three the engine holds
    /// (`show_peer_selections`, `history_budget`, `fast_commit`) the live value is the
    /// projection's.
    pub prefs: Signal<Prefs>,
    /// `prefs.chrome_hiding`, as a root memo (see [`use_pref`]), asked by every chrome
    /// container through [`chrome_dimmed`](crate::layout::chrome_dimmed). Read it, never
    /// peek it: a memo's `peek` does not recompute a stale value. A handler that must not
    /// subscribe reads `prefs.peek()` instead.
    pub chrome_hiding: Memo<ChromeHiding>,
    /// `prefs.hdr.on`, as a root memo: every menu row for `ToggleHdr` asks it inside
    /// its own memo (`stark_ui::commands::Lit`). Read it, never peek it.
    pub hdr_on: Memo<bool>,
    /// Bumped when the brush color is set from outside the color picker (today only the
    /// eyedropper). The picker keeps its own Oklab state, where out-of-gamut `a`/`b`
    /// survive, so it is seeded rather than driven; bumping for its own changes would drag
    /// its marker back onto the gamut boundary under the cursor.
    pub color_epoch: Signal<u64>,
    /// The eyedropper (§18.0.2).
    pub pick: crate::panels::pick::PickState,
    /// What a brush-tuning drag shows over the canvas (§18.1.9), the size ring or the
    /// flow bar; `None` when none is in flight. Shared because the readout is a sibling
    /// overlay of the canvas that owns the drag. The canvas hides its crosshair while this
    /// is up, asking through a memo so it does not re-render per move.
    pub tune_readout: Signal<Option<crate::input::TuneReadout>>,
    /// Where the pointer hovers over the canvas, in the canvas element's px; `None` while
    /// it is elsewhere or the gesture is not paint. The brush cursor (`BrushCursor`,
    /// §18.1.10) rides it, as its own signal so only that overlay re-renders at pointer
    /// rate. Position only: the size is the projection's, read through a memo, so a
    /// bracket tap resizes the circle without a pointer move.
    pub brush_cursor: Signal<Option<Vec2>>,
    /// The live brush's durable half ([`BrushConfig`](stark_ui::brush_config::BrushConfig)):
    /// tip knobs, every effect and the switch between them (so Paint ↔ Erase forgets
    /// nothing), and the smoothing feel (§6.11). The source of truth: the engine gets only
    /// the projection, through [`update_brush`], and nothing reads a brush back off it.
    pub brush: Signal<stark_ui::brush_config::BrushConfig>,
    /// The live brush's transient half: size, flow and color (`brush_config::Transient`,
    /// §18.1.9). Its own signal so a tuning drag wakes only what shows a number, and so
    /// "the same tool" is plain equality on [`brush`](Self::brush)
    /// (`stark_ui::presets::same_tool`). Written through [`update_brush`].
    pub transient: Signal<stark_ui::brush_config::Transient>,
    /// The tow string on screen while a smoothing brush draws (§6.11), in the canvas
    /// element's own px; `None` when there is nothing to show. Its own signal so only
    /// the overlay re-renders at pointer rate.
    pub tow: Signal<Option<crate::overlays::TowUi>>,
    /// Which shape tool the Select panel's action row reaches for (§6.8): the last armed,
    /// rect until then. Needed because a selecting gesture hands the canvas back to the
    /// brush, after which the engine's `tool` no longer says which marquee drew. Written
    /// by [`commands::arm_tool`](crate::commands); read by
    /// [`commands::arm_shape_tool`](crate::commands::arm_shape_tool).
    pub shape_tool: Signal<Tool>,
    /// The whole-canvas composing mode in flight (`crate::modes`): the transform widget
    /// (§16.6, §16.8, §16.9), a perspective guide being shaped (§20.5), the gradient
    /// library's trace (§22.2), or the gradient fill's axis (§22.4).
    ///
    /// One signal, so two modes at once cannot be expressed; written only by
    /// `crate::modes`. View state: previews per sample and one commit on "Done"
    /// (`crate::preview`). A guide's camera is in the log; this holds only which guide is
    /// in hand and this sitting's locks on it.
    pub mode: Signal<Option<stark_ui::modes::Composing>>,
    /// The gradient-bar gesture a trace set aside (§22.2), handed back when the trace
    /// ends; `None` while no trace is armed. Parked rather than dropped because the trace's
    /// catcher cannot share the canvas with the axis catcher
    /// (`panels::gradient_bar::suspend`). Not a second live mode: nothing previews off it
    /// and [`mode`](Self::mode) never holds it.
    pub gradient_resume: Signal<Option<stark_ui::modes::GradientUi>>,
    /// Whether a [`request_paint`] is already waiting on the next animation frame: the
    /// latch that coalesces requests to one paint per frame. Only `peek`ed and `set`, from
    /// non-component code, so no component subscribes to it.
    pub paint_queued: Signal<bool>,
    /// Everything to do with a shared drawing (§12).
    pub collab: collab::CollabState,
    /// Timeline mode: scrubbing and playing back the history
    /// (§18.2.4; `crate::panels::timeline`).
    pub timeline: crate::panels::timeline::TimelineState,
    /// The brush stamps this browser has imported (§6.6; `crate::library`).
    pub shapes: crate::library::LibraryState,
    /// The canvas substrates this browser has imported (§6.4; `crate::library`).
    pub substrates: crate::library::LibraryState,
    /// The brush preset library (`crate::presets`), loaded from `localStorage`
    /// at startup like the shape library.
    pub presets: Signal<Vec<stark_ui::presets::PresetEntry>>,
    /// The name of the preset the brush in hand descends from, kept through every edit;
    /// `None` when it came from no preset the library still has. Unlike the Brush panel's
    /// highlighted row (`stark_ui::presets::same_tool`), it survives edits: "Overwrite
    /// preset" asks it, as does a held number's release (`stark_ui::slots::Held::settle`),
    /// a slot being a preset's name. Written only by `presets::wear`,
    /// `presets::save_current` and `presets::remove`.
    pub preset_in_hand: Signal<Option<String>>,
    /// Rendered preset thumbnails and the offscreen rig that generates them
    /// (`crate::thumbs`).
    pub thumbs: crate::thumbs::ThumbState,
    /// Rendered thumbnails of the document's own layers (§14.6; `crate::layer_thumbs`).
    /// A separate cache from [`thumbs`](Self::thumbs): those picture a brush, keyed on it
    /// and fixed, while these are keyed on tiles that move as the document is painted.
    pub layer_thumbs: crate::layer_thumbs::LayerThumbState,
    /// The gradient library and its trace mode (§22; `crate::gradients`),
    /// loaded from `localStorage` at startup like the libraries above.
    pub gradients: crate::gradients::GradientsState,
    /// The ten brushes under the hand (§18.1.8; `crate::slots`).
    pub slots: crate::slots::SlotState,
    /// Whether the Navigator's miniature shows in the bottom-left corner (§11;
    /// `crate::navigator`). A signal rather than an entry in `PanelLayout::hidden` because
    /// it is not a panel in the stack. Written only by
    /// [`navigator::set_open`](crate::navigator::set_open); stored in the visibility record
    /// (`crate::visibility`, §25.6).
    pub navigator: Signal<bool>,
    /// The guided tour (§24; `crate::tutor`): what the user has done often enough to be
    /// told about, and the lesson on screen. Here, not in its own context, because
    /// [`dispatch`] writes it from outside any component's scope.
    pub tutor: crate::tutor::TutorState,
    /// The pop-out a bar or a panel has flown open, if any (`widgets::PopoutId`, §25.7).
    /// One signal, so two cannot be open at once; app state so Escape can see it. Not on
    /// the dialog stack, which stands `FinishMode` down: the gradient library opens from a
    /// bar while a fill is composing and must not take Enter's "Done" away.
    pub popout: Signal<Option<crate::widgets::PopoutId>>,
    /// The root-mounted dialogs, in the order they were opened (`crate::dialogs`,
    /// the only thing that writes it).
    pub dialogs: Signal<Vec<crate::dialogs::DialogId>>,
    /// The floating panel stack: order, which are open, and the in-flight gestures
    /// (`crate::layout::PanelLayout`). Here, not in its own context, so a command
    /// (`stark_ui::commands::Command::TogglePanel`) reaches it through this handle.
    pub panels: crate::layout::PanelLayout,
    /// This browser's chord table (`stark_ui::commands::Bindings`): shipped defaults with
    /// the user's rebindings over them. Seeded at app start (`commands::load`); a signal
    /// so a shortcut column re-renders when a rebind lands.
    pub bindings: Signal<stark_ui::commands::Bindings>,
    /// This browser's drag table (`stark_ui::drags::DragBindings`): the user's rows over
    /// the shipped ones (§25.8). A signal so the resting cursor, which shows what a press
    /// under the held modifiers would open, changes in the same frame as a rebind.
    pub drags: Signal<stark_ui::drags::DragBindings>,
    /// Whether this browser has been offered drag presets, and whether the offer is
    /// waiting for the hand to leave the canvas (`stark_ui::drags::Offer`, §25.8). Not on
    /// the dialog stack, which says whether the dialog is up; this decides whether it ever
    /// will be.
    pub drag_offer: Signal<stark_ui::drags::Offer>,
}

impl AppState {
    /// Build the app's state. A hook: call it unconditionally, from the root component.
    /// Each group is built, and seeded, by the module that declares it.
    pub fn new() -> Self {
        let prefs = root_signal(crate::prefs::stored);
        // Every render re-reads the same hooks; only the first leaks a box to hold them.
        let signals = Signals {
            renderer: ReadOnly(root_signal(|| None)),
            obs: ReadOnly(root_signal(|| None)),
            startup_failure: root_signal(|| None),
            renderer_ready: root_signal(|| false),
            shipped_landed: root_signal(|| 0),
            written_revision: root_signal(|| 0),
            space_down: root_signal(|| false),
            held_mods: root_signal(Default::default),
            canvas_active: root_signal(|| false),
            panels_asleep: root_signal(|| false),
            prefs,
            chrome_hiding: root_pref(prefs, |p| p.chrome_hiding),
            hdr_on: root_pref(prefs, |p| p.hdr.on),
            color_epoch: root_signal(|| 0),
            pick: crate::panels::pick::PickState::new(),
            tune_readout: root_signal(|| None),
            brush_cursor: root_signal(|| None),
            brush: root_signal(stark_ui::brush_config::BrushConfig::default),
            // The Color panel's opening color rather than the default: the picker seeds
            // off this at mount, before any engine exists, and the first stroke has to
            // lay the color the marker shows (`crate::app` pushes it to the engine later).
            transient: root_signal(|| stark_ui::brush_config::Transient {
                color: stark_ui::color::INITIAL_COLOR,
                ..Default::default()
            }),
            tow: root_signal(|| None),
            shape_tool: root_signal(|| Tool::SelectRect),
            mode: root_signal(|| None),
            gradient_resume: root_signal(|| None),
            paint_queued: root_signal(|| false),
            collab: collab::CollabState::new(),
            timeline: crate::panels::timeline::TimelineState::new(),
            shapes: crate::library::LibraryState::new(),
            substrates: crate::library::LibraryState::new(),
            presets: root_signal(Vec::new),
            preset_in_hand: root_signal(|| None),
            thumbs: crate::thumbs::ThumbState::new(),
            layer_thumbs: crate::layer_thumbs::LayerThumbState::new(),
            gradients: crate::gradients::GradientsState::new(),
            slots: crate::slots::SlotState::new(),
            // What this browser last had on screen (`crate::visibility`, §25.6), here
            // rather than in a load hook so the first render is already that screen.
            navigator: root_signal(|| {
                stark_ui::visibility::stored_showing(VisibilityToggle::Navigator)
            }),
            tutor: crate::tutor::TutorState::new(),
            popout: root_signal(|| None),
            dialogs: root_signal(Vec::new),
            panels: crate::layout::PanelLayout::new(),
            bindings: root_signal(Default::default),
            drags: root_signal(Default::default),
            drag_offer: root_signal(Default::default),
        };
        use_hook(move || AppState(Box::leak(Box::new(signals))))
    }
}

/// Subscribe to one slice of the projection instead of to all of it.
///
/// [`Signals::obs`] carries everything from the view (pointer rate) to the history budget,
/// so reading it in a body re-renders whenever any of it moves. A memo propagates only
/// when its slice changed. A hook; take one memo of a tuple rather than one per field.
/// `None` before WebGPU init has published a projection.
pub fn use_obs<T>(
    state: AppState,
    slice: impl Fn(&ObservableState) -> T + 'static,
) -> Memo<Option<T>>
where
    T: Clone + PartialEq + 'static,
{
    use_memo(move || state.obs.read().as_ref().map(&slice))
}

/// [`use_obs`] with `slice` handed the `Option`, for a reader with its own answer to "no
/// engine yet": a fallback such as `Command::enabled(None)`, a slice that is itself an
/// `Option`, or one that wants `?`. Spares a `Memo<Option<Option<T>>>`. A hook.
pub fn use_obs_opt<T>(
    state: AppState,
    slice: impl Fn(Option<&ObservableState>) -> T + 'static,
) -> Memo<T>
where
    T: PartialEq + 'static,
{
    use_memo(move || slice(state.obs.read().as_ref()))
}

/// Subscribe to one slice of this browser's preferences, for [`use_obs`]'s reason. A
/// hook; a reader that cannot call one (a plain function, another memo's closure) reads a
/// slice kept on [`Signals`] instead ([`Signals::chrome_hiding`], [`Signals::hdr_on`]).
pub fn use_pref<T>(state: AppState, slice: impl Fn(&Prefs) -> T + 'static) -> Memo<T>
where
    T: PartialEq + 'static,
{
    use_memo(move || slice(&state.prefs.read()))
}

/// [`use_pref`] for a slice kept on [`Signals`]: owned by the root scope, for
/// [`root_signal`]'s reason.
fn root_pref<T>(prefs: Signal<Prefs>, slice: impl Fn(&Prefs) -> T + 'static) -> Memo<T>
where
    T: PartialEq + 'static,
{
    root_memo(move || slice(&prefs.read()))
}

/// A memo owned by the root scope, for [`root_signal`]'s reason. A hook, like it.
pub(crate) fn root_memo<T>(f: impl FnMut() -> T + 'static) -> Memo<T>
where
    T: PartialEq + 'static,
{
    use_hook(|| Runtime::current().in_scope(ScopeId::ROOT, || Memo::new(f)))
}

/// Repaint the canvas on the next animation frame, coalescing every request before it.
///
/// Requests arrive per local pointer sample and per peer gesture frame (~30 Hz × peers),
/// far above the display rate, and nothing bounds the GPU queue, so painting each inline
/// would grow latency for as long as strokes last. The engine still integrates every
/// event; only presentation is coalesced.
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

/// Whether the canvas can no longer take a mark, so the app has stopped painting
/// and dispatching (§5).
///
/// Asks the fields `crate::failure` mounts its report on, not [`Renderer::gpu_healthy`],
/// so "stopped" and "the artist has been told" cannot disagree;
/// [`Signals::startup_failure`] covers a device that never arrived. `peek`: callers are
/// doors and event handlers, not renders, and this must not widen what a pointer move
/// wakes.
pub fn gpu_lost(state: AppState) -> bool {
    state.startup_failure.peek().is_some()
        || state
            .obs
            .peek()
            .as_ref()
            .is_some_and(|o| o.gpu_failure.is_some())
}

/// One animation-frame hop of [`request_paint`]: paint, unless the GPU has
/// fallen behind — then hold the latch and try again next frame.
///
/// Runs in the rAF callback itself, not a task it wakes: a woken task resumes in the
/// microtask drain and a dioxus task also waits out a render of dirty scopes, either of
/// which can slip past the frame's rendering steps and show the previous canvas for a
/// display interval.
fn schedule_paint(state: AppState) {
    crate::platform::on_animation_frame(move || {
        let mut renderer = state.renderer.0;
        let mut guard = renderer.write();
        // The GPU owes more than a full pipeline of frames: skip with the latch held so
        // the next rAF re-checks. Nothing is lost; the next paint folds everything accrued.
        if guard.as_ref().is_some_and(Renderer::gpu_behind) {
            {
                // The count is the datum: how often this back-pressure fired, which no
                // bench sees. The block closes the span at once.
                stark_engine::timing::span!("frame.skipped");
            }
            drop(guard);
            schedule_paint(state);
            return;
        }
        // Everything a shown frame costs; its count over the window is the frame rate
        // actually achieved, which every phase row is read against.
        stark_engine::timing::span!(stark_engine::timing::FRAME);
        let mut queued = state.paint_queued;
        queued.set(false);
        // A dead device has to be noticed here (§5): painting goes on between commands.
        // The publish raises the report (`crate::failure`).
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

/// Reach the engine as `&mut`, publish what it now looks like, and repaint. `None` when
/// there is no engine yet, or once the GPU has died ([`gpu_lost`], §5).
///
/// The publish is on the door so no mutation leaves the chrome showing a value the engine
/// no longer holds (§4, §7). For entry points that are not commands, such as leaving a
/// session; commands go through [`dispatch`], replacements through [`replace_document`].
/// The GPU check precedes `f`, so the command that kills the device still publishes,
/// which is what raises the report.
pub fn with_engine<R>(state: AppState, f: impl FnOnce(&mut Renderer) -> R) -> Option<R> {
    if gpu_lost(state) {
        return None;
    }
    let mut renderer = state.renderer.0;
    let out = {
        let mut guard = renderer.write();
        let r = guard.as_mut()?;
        let out = f(r);
        // Inside the guard: `obs.set` marks subscribers dirty but renders nothing
        // synchronously, so no reader can observe the renderer mid-borrow.
        publish(state, r.desk.engine().observe());
        out
    };
    request_paint(state);
    Some(out)
}

/// Hand `snapshot` to the chrome, only when it moved: a `Signal` write wakes every
/// subscriber even when the value is unchanged. Cheap, since the layer list compares by
/// pointer (`stark_engine::Layers`).
fn publish(state: AppState, snapshot: ObservableState) {
    let mut obs = state.obs.0;
    // The `peek` guard drops with this statement, before the `set` below.
    let moved = obs.peek().as_ref() != Some(&snapshot);
    if moved {
        obs.set(Some(snapshot));
    }
}

/// Replace the document (open a file, join a session, start afresh), then paint inline
/// and publish. The frontend's half of [`Desk::replace`].
///
/// Not a [`dispatch`]: it commits and broadcasts nothing and is no deed the tour counts
/// (§4, §24), yet moves everything the chrome shows. `None` is [`with_engine`]'s;
/// `Some(Err)` is a refusal with the open document untouched. What a join skipped is
/// logged here: a peer fetch supplies it, and the pump has nobody to tell.
///
/// Leaves any live session first, even if the engine then refuses: a replacement does not
/// stop the pump, the phase or the ticket in the URL, which would go on merging peers'
/// actions into the new document (§12.4). Outside the engine hold because
/// [`collab::leave`] takes the engine. A join leaves nothing: [`collab::join`] installs
/// its session only after this returns. The inline paint makes the first frame shown the
/// framed new document.
///
/// [`Desk::replace`]: stark_ui::desk::Desk::replace
#[must_use = "a refusal leaves the old document on screen, which the caller has to say"]
pub fn replace_document(
    state: AppState,
    replacement: Replacement<'_>,
    owed: &[(AssetNeed, &[u8])],
) -> Option<stark_engine::Result<()>> {
    collab::leave(state);
    let replaced = with_engine_quiet(state, |r| {
        let replaced = r.desk.replace(replacement, owed);
        if replaced.is_ok() {
            r.paint();
        }
        replaced
    })?;
    let out = match replaced {
        // The replacement's own projection, rather than a second one built here.
        Ok(Replaced { seen, skipped }) => {
            if !skipped.is_empty() {
                tracing::warn!(?skipped, "owed content did not install; a peer supplies it");
            }
            publish(state, seen);
            Ok(())
        }
        // Nothing the chrome shows moved — unless the device failed on the way, which
        // only a publish reports (§5).
        Err(e) => {
            publish_observation(state);
            Err(e)
        }
    };
    request_paint(state);
    Some(out)
}

/// Reach the engine as `&mut` without publishing or asking for a frame, for work that
/// changes nothing `observe` projects (rendering, readbacks, draining the outbox, the
/// presence tick, installing asset bytes) or whose caller publishes afterwards
/// ([`publish_observation`]). Several run at pointer or frame rate.
///
/// Outside this module, a closure here that calls `Renderer::process` is in the wrong
/// door: commands publish. Guarded by [`gpu_lost`] like [`with_engine`]; saving does not
/// use this door (`files::save_document`, §8), so it still works after a device failure.
pub fn with_engine_quiet<R>(state: AppState, f: impl FnOnce(&mut Renderer) -> R) -> Option<R> {
    if gpu_lost(state) {
        return None;
    }
    let mut renderer = state.renderer.0;
    let mut guard = renderer.write();
    let r = guard.as_mut()?;
    Some(f(r))
}

/// Publish the engine's current projection without mutating anything, for a caller that
/// held the engine quietly and learns afterwards whether the chrome is stale: the collab
/// pump, which publishes for events that commit and not at presence rate (§17.5), and the
/// frame loop and [`replace_document`], where only a device failure moved anything (§5).
pub fn publish_observation(state: AppState) {
    let Some(snapshot) = state
        .renderer
        .peek()
        .as_ref()
        .map(|r| r.desk.engine().observe())
    else {
        return;
    };
    publish(state, snapshot);
}

/// Hand the freshly-built engine to the app and publish its opening projection, taken
/// before `r` moves into the signal: whatever reads the renderer assumes the chrome
/// already describes it.
pub fn publish_renderer(state: AppState, r: Renderer) {
    let mut renderer = state.renderer;
    let mut obs = state.obs;
    obs.0.set(Some(r.desk.engine().observe()));
    // The thumbnail generator holds the device and pipelines directly, so it never
    // borrows a live renderer to draw a brush (`crate::thumbs`).
    let mut shared = state.thumbs.shared;
    shared.set(Some(r.desk.engine().shared()));
    renderer.0.set(Some(r));
    // Last, so a reader woken by readiness finds the renderer in place and the chrome
    // already describing it (see [`Signals::renderer_ready`]).
    let mut ready = state.renderer_ready;
    ready.set(true);
}

/// Apply a command, request a repaint, and refresh the observable snapshot.
/// In a shared session, whatever the command committed is then broadcast.
pub fn dispatch(state: AppState, command: impl Into<InputCommand>) {
    let command = command.into();
    // Before the engine sees it, so the tour can ask what a command changes: the shape
    // assist answers only before the gesture's `End` lands (§24.2). Here, at the one seam,
    // so the tour also counts ways of doing things added after it was written.
    crate::tutor::observe(state, &command);
    with_engine(state, |r| r.process(command));
    collab::flush_outbox(state);
}

/// Apply the pointer-rate samples of an in-flight gesture, oldest first, and repaint,
/// without [`dispatch`]'s publish and outbox flush.
///
/// Mid-gesture nothing the chrome reads moves (the committed document, the layer list and
/// the undo flags stand until the gesture ends), so `Engine::observe` would walk the layer
/// tree for the same answer, and nothing commits to flush; the End goes through
/// [`dispatch`]. A batch because one `pointermove` carries every report the browser
/// coalesced into it.
pub fn dispatch_samples(state: AppState, samples: &[InputSample]) {
    if samples.is_empty() {
        return;
    }
    let processed = with_engine_quiet(state, |r| {
        for &sample in samples {
            // One span per report reaching the engine: its count over the window is the
            // input rate the engine heard, e.g. whether a 240 Hz pen is heard at 240 Hz.
            stark_engine::timing::span!(stark_engine::timing::INPUT_SAMPLE);
            r.process(GestureCommand::To { sample });
        }
    });
    if processed.is_some() {
        request_paint(state);
    }
}

/// Apply a pointer-rate report of where the hand is and repaint, without the publish and
/// flush for [`dispatch_samples`]' reason: the hover mark (§18.1.10) and the guide rays
/// through the cursor (§20.9) change only what the canvas shows. Its own door so the
/// `input.sample` timing row stays stroke input; hover is priced on `input.hover` (§7.1).
pub fn dispatch_hover(state: AppState, command: impl Into<InputCommand>) {
    with_engine_quiet(state, |r| r.process(command));
    request_paint(state);
}

/// Apply a command without repainting, publishing or broadcasting, for
/// [`PeerCommand::SetCursor`](stark_engine::command::PeerCommand::SetCursor): it arrives
/// at pointer rate and the browser draws our own cursor. Peers still get it, as the
/// presence pump reads it off the engine on its own cadence (§17.5).
pub fn dispatch_quiet(state: AppState, command: impl Into<InputCommand>) {
    with_engine_quiet(state, |r| r.process(command));
}

/// Resize the surface/engine, then repaint inline rather than through [`request_paint`]:
/// a frame of the old size shown until the next rAF is a visible flash.
///
/// Inline only while the GPU keeps up. An edge drag resizes every frame, this paint skips
/// [`schedule_paint`]'s back-pressure, and the first paint at a new size rebuilds every
/// compositing attachment (`composite::TargetSized::ensure_targets`): on a large window,
/// enough fresh render targets against a lagging queue to lose the device. While the GPU
/// is behind, the frame [`with_engine`] queues shows the new size on a later frame.
///
/// A report before the renderer exists is dropped, safe only because init re-reads the
/// element just before publishing the renderer
/// ([`Renderer::sync_to_canvas`](crate::render::Renderer::sync_to_canvas)).
pub fn resize(state: AppState, width: u32, height: u32) {
    with_engine(state, |r| {
        r.resize(width, height);
        if !r.gpu_behind() {
            r.paint();
        }
    });
}

/// Mutate a copy of the live [`BrushConfig`](stark_ui::brush_config::BrushConfig) and
/// `Transient`, and commit them: the signals take the new values and the engine their
/// projection (`ViewCommand::SetBrush`). The only door, so the signals and the engine hold
/// the same brush.
///
/// The dispatch goes first: the tour compares a `SetBrush` with the brush still held in
/// both signals (§24), so they are set after it. Loud even at pointer rate (tuning drags, the
/// eyedropper): their readouts are the Brush panel's sliders and the Color panel's swatch,
/// so the publish is the point; panels that show no brush sleep through it via
/// [`use_obs`]. Returns nothing because most callers are `oninput` handlers, where a
/// returned value stops the closure coercing to a `Callback`; a caller wanting a value
/// takes it out through `f`.
pub fn update_brush(
    state: AppState,
    f: impl FnOnce(&mut stark_ui::brush_config::BrushConfig, &mut stark_ui::brush_config::Transient),
) {
    let mut config = *state.brush.peek();
    let mut tune = *state.transient.peek();
    f(&mut config, &mut tune);
    // Held here to a brush the renderer will draw (`BrushConfig::settle`, §6.2): the
    // stretch gives where the size cannot carry it, and the size has writers that a clamp
    // on the stretch control would not see.
    config.settle(tune);
    dispatch(
        state,
        ViewCommand::SetBrush {
            brush: config.params(tune),
            color: tune.color,
        },
    );
    let mut sig = state.brush;
    sig.set(config);
    let mut sig = state.transient;
    sig.set(tune);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`AppState`] is one pointer, whatever [`Signals`] grows to hold: this catches a
    /// field added to `AppState` rather than `Signals`.
    #[test]
    fn the_handle_is_one_pointer() {
        assert_eq!(
            size_of::<AppState>(),
            size_of::<&'static Signals>(),
            "AppState carries something beside the pointer to its signals"
        );
        // And the pointee is large, so the check above is not passing vacuously.
        assert!(
            size_of::<Signals>() > 64 * size_of::<usize>(),
            "Signals is unexpectedly small: {} bytes; is it still the whole state?",
            size_of::<Signals>()
        );
    }
}
