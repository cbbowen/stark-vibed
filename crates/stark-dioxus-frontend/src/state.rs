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
//! unrelated command refreshes it (§4, §7). Both the canvas substrate and the lighting
//! environment are reachable that way, and neither spelling compiles.

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
use stark_ui::prefs::{ChromeHiding, Prefs};
use stark_ui::session::Replacement;

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
///
/// Crate-visible because most of these signals are built beside the module that owns
/// them (each group's `new`, such as `thumbs::ThumbState::new`), and because a
/// gesture that holds its own state (§25.3) but is written to from a detached task,
/// such as `input::PickMove`, needs the same ownership. A hook either way: call it
/// unconditionally, in a component body.
pub(crate) fn root_signal<T: 'static>(init: impl FnOnce() -> T) -> Signal<T> {
    use_hook(|| Signal::new_in_scope(init(), ScopeId::ROOT))
}

/// A signal handed out with its `write` half kept back: `read` and `peek` work
/// exactly as they do on a [`Signal`], and there is no way through it to the value
/// as `&mut`.
///
/// The inner signal is a private field of a type declared in this module, so *this
/// module* is the whole set of code that can mutate what one of these wraps. That
/// is the point: [`Signals::renderer`](crate::state::Signals::renderer) and [`Signals::obs`](crate::state::Signals::obs) are the two pieces of
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
/// **One word wide**, and that is the whole of why it is a newtype rather than the
/// struct itself. [`Signals`] holds eighty-odd `Signal`s, each three words on
/// wasm32 — call it two kilobytes — and this handle is a by-value parameter on
/// some two hundred and seventy functions and is copied out of `use_context` by
/// every component on every render. Passing the struct meant passing all of it.
///
/// Leaking one [`Signals`] costs nothing that was not already true: it is built
/// once, in the root component, which is never unmounted — the same fact
/// [`root_signal`] is built on, and the reason every signal in it is owned by
/// `ScopeId::ROOT` rather than by the scope that declared it.
///
/// [`Deref`](std::ops::Deref) and deliberately **not** `DerefMut`, so `state.foo`
/// reads exactly as it did and `state.foo.set(…)` does not compile. Every signal
/// here is `Copy`, so the way to write one is to take it into a local first —
/// `let mut foo = state.foo; foo.set(…)` — which is the form the crate already
/// prefers everywhere, for a reason of its own: a `set` through a place expression
/// is one keystroke from a read guard held across its own write, which is the
/// borrow panic this module warns about twice.
#[derive(Clone, Copy)]
pub struct AppState(&'static Signals);

impl std::ops::Deref for AppState {
    type Target = Signals;

    fn deref(&self) -> &Signals {
        self.0
    }
}

/// Every signal the app holds — what [`AppState`] is a handle to.
///
/// Built by [`AppState::new`] rather than field-by-field at the call site, so every
/// signal here goes through [`root_signal`] — the one property that has to hold for
/// all of them, and the kind of thing a hand-written literal drifts out of the moment
/// a field is added.
pub struct Signals {
    /// Surface + engine, built asynchronously once the canvas mounts. `None`
    /// until WebGPU init completes. Not `Send` — lives in unsync storage.
    ///
    /// Read freely; to *move* engine state, call [`dispatch`] or [`with_engine`]
    /// (see the module note on [`ReadOnly`]).
    pub renderer: ReadOnly<Option<Renderer>>,
    /// UI-facing engine projection, refreshed after each command — by
    /// [`with_engine`], which is the only thing that can write it.
    pub obs: ReadOnly<Option<ObservableState>>,
    /// Why the app could not start, if it could not (§5, `crate::failure`).
    ///
    /// **The other half of [`obs`](Self::obs)'s `gpu_failure`, and deliberately
    /// not the same field.** That one is a device that died with a document
    /// behind it, so the report offers to save; this is a device that never
    /// arrived, so there is no engine, no document, and nothing to offer but the
    /// explanation. Keeping them apart is what lets each report say the true
    /// thing — and it has to be a signal of its own besides, because a
    /// projection is something only an engine can publish.
    ///
    /// Set once by the startup task and never cleared: the browser does not grow
    /// a GPU while the page is open.
    pub startup_failure: Signal<Option<crate::render::StartupFailure>>,
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
    /// How many shipped assets have landed in the main engine since it was published —
    /// the substrate maps fetched on first use (`crate::shipped::landed`).
    ///
    /// For [`renderer_ready`](Self::renderer_ready)'s reason: a picker listing shipped
    /// assets has to redraw when one lands, and reading the renderer to learn that
    /// redrew it on every command. A count, because more than one lands.
    pub shipped_landed: Signal<u32>,
    /// The `doc_revision` this document was last **written out** at — by Save, which
    /// wrote the log, or by Export, which drew the picture. Zero until one of them
    /// has happened (`crate::files`).
    ///
    /// The frontend's half of "is there work only this tab holds", which is what the
    /// unload guard asks. The engine holds the other half and cannot hold this one:
    /// what it knows is whether the document has moved since it arrived
    /// (`ObservableState::edited`), and a file leaving the browser is not an event a
    /// document has any way to notice.
    ///
    /// Zero is safe as "never", not merely conventional: `doc_revision` counts up
    /// across the whole life of the page and is past zero before the opening document
    /// exists, so no live revision can collide with the initial value — and `edited`
    /// gates the comparison in any case.
    pub written_revision: Signal<u64>,
    /// Whether the user is holding space.
    pub space_down: Signal<bool>,
    /// The modifiers currently held, tracked off the window's key events
    /// (`input`'s keydown and keyup) and self-correcting off every event's
    /// modifier set, so a press or release that happened while the window was
    /// not focused is caught up on the next keystroke. Only ever *shown*: the
    /// resting cursor and the eyedropper's options bar ask the drag table what
    /// a press under these would open (`stark_ui::drags::armed`), which is what makes a
    /// modifier binding discoverable before it is used.
    pub held_mods: Signal<stark_ui::keys::Mods>,
    /// Whether a canvas gesture is in flight (a stroke, a selection drag, a pan,
    /// or a run of wheel zooming). The floating chrome fades out while it is set,
    /// handing the screen back to the painting — see
    /// [`chrome_dimmed`](crate::layout::chrome_dimmed).
    pub canvas_active: Signal<bool>,
    /// Whether the panel stack is **asleep**: still out of the way after the gesture
    /// that faded it, until the pointer reaches into the column it lives in
    /// (§11).
    ///
    /// [`canvas_active`](Self::canvas_active) says the canvas is in hand *now*; this
    /// says it was, and nothing has asked for the panels back yet. Set by
    /// [`end_interaction`](crate::input::end_interaction) — only where the fade was
    /// actually in force, so a gesture that deliberately keeps the chrome up (the
    /// eyedropper, brush tuning) does not put it to sleep on the way out. Cleared by
    /// the slice that hears the pointer arrive (`layout::PanelStack`) and by opening
    /// a panel (`layout::open_panel`), which would otherwise be a command with
    /// nothing to show for itself.
    ///
    /// **The stack alone sleeps**, not everything [`chrome_dimmed`] dims. The rest of
    /// the floating chrome is either a mode's — a bar with the Done button on it, the
    /// handles a transform is dragged by — or the rail every menu is behind, and all
    /// of it lives somewhere other than this column: latched off until the pointer
    /// visited the *right edge*, those would be controls you cannot reach by reaching
    /// for them. The stack is what there is a lot of, and it is the thing whose
    /// coming back over the painting was worth waiting for.
    ///
    /// [`chrome_dimmed`]: crate::layout::chrome_dimmed
    pub panels_asleep: Signal<bool>,
    /// This browser's preferences (§25.6), seeded from the stored record. Written
    /// only by [`prefs::set`](crate::prefs::set).
    ///
    /// The frontend's fields are live here. The three the engine holds —
    /// `show_peer_selections`, `history_budget`, `fast_commit` — are what was stored
    /// or last asked for; what the engine holds is the projection's.
    pub prefs: Signal<Prefs>,
    /// `prefs.chrome_hiding`, as a root memo ([`use_pref`]'s note): every chrome
    /// container asks it through [`chrome_dimmed`](crate::layout::chrome_dimmed).
    ///
    /// Read it, never peek it: a memo's `peek` does not recompute a stale value.
    /// A handler that must not subscribe reads `prefs.peek()` instead.
    pub chrome_hiding: Memo<ChromeHiding>,
    /// `prefs.hdr.on`, as a root memo: every menu row for `ToggleHdr` asks it inside
    /// its own memo (`commands::Lit`). Read it, never peek it.
    pub hdr_on: Memo<bool>,
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
    pub pick: crate::panels::pick::PickState,
    /// What a brush-tuning drag is showing over the canvas (§18.1.9) — the size ring
    /// or the flow bar — or `None` when no tuning drag is in flight.
    ///
    /// Shared state rather than the gesture's own, for [`PickState::dragging`](crate::panels::pick::PickState::dragging)'s
    /// reason: the drag is the canvas's, but the readout is a sibling overlay of it — a
    /// `<div>` cannot be drawn from inside a `<canvas>`'s handler without somewhere for
    /// both to read.
    ///
    /// The canvas reads it back for one thing of its own: while this is up the
    /// crosshair goes ([`TuneReadout`](crate::input::TuneReadout)), and *that* a drag is in flight is the whole of
    /// what it asks — through a memo, so the surface a stroke is made on does not
    /// re-render per move to find the answer unchanged.
    pub tune_readout: Signal<Option<crate::input::TuneReadout>>,
    /// Where the pointer hovers over the canvas, in the canvas element's own px —
    /// `None` while it is elsewhere, or while the gesture in hand is not paint (a
    /// pinch, a pan, a tuning drag). The brush cursor rides it (`BrushCursor`,
    /// §18.1.10): a circle of the live brush's size under the resting pointer.
    ///
    /// Its own signal for [`tune_readout`](Self::tune_readout)'s reason: it moves per
    /// pointer report, and only the little overlay may re-render at that rate.
    /// **Position only** — the size half of the picture is the projection's
    /// (`brush.radius × view.zoom`), read in the overlay through a memo, so a
    /// bracket tap or a wheel notch resizes the circle where it stands, with no
    /// pointer move needed to notice.
    pub brush_cursor: Signal<Option<Vec2>>,
    /// The live brush's **durable** half, as this frontend configures it
    /// ([`BrushConfig`](stark_ui::brush_config::BrushConfig)): the shared tip
    /// knobs, **every** effect with the switch between them — so toggling
    /// Paint ↔ Erase forgets nothing, the hand's color above all — and the
    /// stroke-smoothing feel (§6.11). The size and flow it is being worked
    /// at are [`transient`](Self::transient), the other half of the pair.
    ///
    /// **The source of truth.** The engine holds only the projection, sent
    /// through [`update_brush`] — the one door — so nothing here reads a brush
    /// back off the observable, and what the engine cannot represent (the
    /// inactive effect, the feel) never has to round-trip through it.
    pub brush: Signal<stark_ui::brush_config::BrushConfig>,
    /// The live brush's **transient** half — the size and flow the hand is
    /// working the tool at (`brush_config::Transient`, §18.1.9). Its own
    /// signal beside [`brush`](Self::brush) rather than a pair of fields on
    /// it, so a tuning drag at pointer rate wakes only what shows a number,
    /// never the chrome that shows the tool — and so "the same tool" is plain
    /// equality on the durable half (`stark_ui::presets::same_tool`). Written through
    /// [`update_brush`], the same one door.
    pub transient: Signal<stark_ui::brush_config::Transient>,
    /// The tow string on screen while a smoothing brush draws (§6.11), in the
    /// canvas element's own px — `None` when there is nothing to show. Its own
    /// signal for the reason [`tune_readout`](Self::tune_readout) is: only the
    /// little overlay re-renders at pointer rate, never the chrome.
    pub tow: Signal<Option<crate::overlays::TowUi>>,
    /// Which of the three shape tools the Select panel's action row reaches for
    /// (§6.8) — the last one armed, rect until one has been.
    ///
    /// The counterpart of a rule the session owns: a selecting gesture hands the
    /// canvas straight back to the brush, which is exactly what makes the
    /// *identity* of the tool it used worth keeping — the engine's `tool` says
    /// `Brush` a moment later and no longer remembers which marquee drew.
    ///
    /// Written by the one act that arms a tool at all
    /// ([`commands::arm_tool`](crate::commands)), so a chip, a chord and the
    /// search palette all leave the same answer behind, and read by the one act
    /// that hands a tool back without naming one
    /// ([`commands::arm_shape_tool`](crate::commands::arm_shape_tool)).
    pub shape_tool: Signal<Tool>,
    /// The whole-canvas composing mode in flight, and what it is composing
    /// (`crate::modes`): the transform widget (§16.6, §16.8, §16.9), a
    /// perspective guide being shaped (§20.5), the gradient library's trace
    /// (§22.2), or the gradient fill's axis (§22.4).
    ///
    /// **One signal, and that is the guarantee.** These were four, with "at most
    /// one is live" held true by every entry point remembering to put down
    /// whatever it found; two modes at once is now a state the app cannot
    /// express rather than one it declines to reach. `modes::enter` is the only
    /// way in and `modes::leave` the only way out — do not write this from
    /// anywhere else, which is the one rule that replaces the seven.
    ///
    /// View state throughout: what each mode produces is previews per pointer
    /// sample and one commit on "Done" (`crate::preview`). A guide's *camera* is
    /// document state and lives in the log; what is here is only which guide is
    /// in hand, and the locks held on it for this sitting.
    pub mode: Signal<Option<stark_ui::modes::Composing>>,
    /// The gradient-bar gesture a **trace** set aside (§22.2), to be handed back
    /// when the trace ends. `None` whenever no trace is armed.
    ///
    /// Arming a trace from the bar's own well is not abandoning the composition
    /// — it is reaching into the library the bar is holding — but the trace's
    /// catcher cannot share the canvas with the axis catcher, so the gesture is
    /// parked here for the mode's life rather than dropped
    /// (`panels::gradient_bar::suspend`). Deliberately *not* a second live
    /// gradient mode, which is why it stays a field of its own now that the
    /// modes share one: nothing previews off this, and [`mode`](Self::mode)
    /// never holds it, so "one mode composing at a time" stays true while it is
    /// held.
    pub gradient_resume: Signal<Option<stark_ui::modes::GradientUi>>,
    /// Whether a [`request_paint`] is already waiting on the next animation frame.
    /// The latch that turns any number of paint requests into one paint per frame.
    /// Read and written only from non-component code (`peek`/`set`), so no
    /// component ever subscribes to it.
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
    /// The preset the brush in hand was taken from, by name — and still, after
    /// every edit since. That is what tells it apart from the Brush panel's
    /// highlighted row (`stark_ui::presets::same_tool`): the row says the brush still *is*
    /// a preset, this says which one it *descends from*, and the brush editor's
    /// "Overwrite preset" is a question about the second — as is what a held
    /// number binds itself to at the release (`slots::Held::settle`), since a
    /// slot is a preset's name and not a brush. `None` when the tool in hand
    /// came from no preset the library still has.
    ///
    /// Written only where a whole tool arrives (`presets::wear`), where a
    /// name is given to the brush in hand (`presets::save_current`) and where
    /// the named preset goes (`presets::remove`) — never by an edit, which is
    /// the point of it.
    pub preset_in_hand: Signal<Option<String>>,
    /// Rendered preset thumbnails and the offscreen rig that generates them
    /// (`crate::thumbs`).
    pub thumbs: crate::thumbs::ThumbState,
    /// Rendered thumbnails of the document's own layers (§14.6;
    /// `crate::layer_thumbs`). A separate cache from `thumbs` above, and not
    /// merely a second use of it: those are pictures of a brush, keyed on the
    /// brush and pinned to one look forever, while these are pictures of the
    /// *document*, keyed on tiles that move as it is painted.
    pub layer_thumbs: crate::layer_thumbs::LayerThumbState,
    /// The gradient library and its trace mode (§22; `crate::gradients`),
    /// loaded from `localStorage` at startup like the libraries above.
    pub gradients: crate::gradients::GradientsState,
    /// The ten brushes under the hand (§18.1.8; `crate::slots`).
    pub slots: crate::slots::SlotState,
    /// Whether the Navigator's miniature is showing in the bottom-left corner
    /// (§11; `crate::navigator`) — the visibility menu's "Navigator".
    ///
    /// A signal here rather than an entry in `PanelLayout::hidden`, because the
    /// overview is no longer a panel in the stack: it wears no title bar, its box
    /// is the artwork's aspect rather than a column's width, and it stands in a
    /// corner of the window. What it keeps from the stack is durability — written
    /// only by [`navigator::set_open`](crate::navigator::set_open), and stored in the
    /// one record the stack's own visibility is in (`crate::visibility`, §25.6).
    pub navigator: Signal<bool>,
    /// The guided tour (§24; `crate::tutor`): what the user has done often enough
    /// to be told about, and the lesson on screen.
    ///
    /// Here rather than in a context of its own because it is written from
    /// [`dispatch`] — the seam every command passes through — which is free
    /// function code belonging to no component's scope.
    pub tutor: crate::tutor::TutorState,
    /// The pop-out a bar or a panel has flown open, if any (`widgets::PopoutId`,
    /// §25.7).
    ///
    /// **One signal for all of them**, on `stark_ui::modes::Composing`'s argument: two open
    /// at once is a state nothing wants and nothing should have to prevent. And
    /// app state rather than the locals these were, so Escape can see them —
    /// deliberately *not* on the dialog stack, which is also what stands
    /// `FinishMode` down, because the gradient library is opened from a bar while
    /// a fill is composing and must not take Enter's "Done" away.
    pub popout: Signal<Option<crate::widgets::PopoutId>>,
    /// The root-mounted dialogs, in the order they were opened (`crate::dialogs`,
    /// the only thing that writes it).
    pub dialogs: Signal<Vec<crate::dialogs::DialogId>>,
    /// The floating panel stack: order, which are open, and the in-flight
    /// gestures (`crate::layout::PanelLayout`). Here rather than provided as
    /// its own context because the panels' commands live in the registry now
    /// (`stark_ui::commands::Command::TogglePanel`), and a command reaches everything it
    /// acts on through this one handle.
    pub panels: crate::layout::PanelLayout,
    /// This browser's chord table (`stark_ui::commands::Bindings`): the shipped defaults
    /// with the user's rebindings laid over them. A signal so a shortcut column
    /// re-renders the moment a rebind lands; seeded from storage at app start
    /// (`commands::load`) and written back on every rebind.
    pub bindings: Signal<stark_ui::commands::Bindings>,
    /// This browser's drag table (`stark_ui::drags::DragBindings`): the shipped rows with
    /// the user's own laid over them (§25.8). A signal for the chord table's
    /// reason and one more of its own — the resting cursor asks it what a press
    /// under the held modifiers would open, so a rebind has to move the promise
    /// in the same frame it moves the press.
    pub drags: Signal<stark_ui::drags::DragBindings>,
    /// Whether this browser has been offered a table of drag presets, and
    /// whether one is waiting for the hand to come off the canvas
    /// (`stark_ui::drags::Offer`, §25.8).
    ///
    /// Not on the dialog stack though it raises a dialog: the stack says *whether
    /// the dialog is up*, and this is the durable fact that decides whether it ever
    /// will be — seeded from storage at app start alongside the table above.
    pub drag_offer: Signal<stark_ui::drags::Offer>,
}

impl AppState {
    /// Build the app's state. Call once, from the root component.
    ///
    /// **Every group is built by the module that declares it.** This function
    /// used to write out each one field by field, which made `state.rs` — the
    /// module everything else depends on — the place that had to know what
    /// `ThumbState`, `PanelLayout` and `TutorState` are *made of*, and made a
    /// field added to any of them an edit here as well as there. The seeding
    /// rules travelled with them: which panels this browser left open, which
    /// preference a signal opens on, how wide the quick-brush rack is. What is
    /// left is the list, which is what this was always for.
    pub fn new() -> Self {
        let prefs = root_signal(crate::prefs::stored);
        AppState(Box::leak(Box::new(Signals {
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
            // Seeded with the Color panel's opening color rather than the model
            // default's black: the panel reads its picker's seed off this signal
            // at mount, before any engine exists, and the first stroke has to
            // lay the color the marker shows (`crate::app` pushes the same
            // configuration to the engine once one is up).
            brush: root_signal(stark_ui::brush_config::BrushConfig::default),
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
            // Seeded from what this browser last had on screen, like the other
            // three entries of that menu (`crate::visibility`, §25.6) — here
            // rather than in a load hook, so the first render is already the
            // screen the artist left.
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
        })))
    }
}

/// Subscribe to **one slice** of the projection instead of to all of it.
///
/// [`Signals::obs`](crate::state::Signals::obs) is a single signal carrying everything from the view — which
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

/// [`use_obs`] for a reader that has its **own** answer to "there is no engine
/// yet": `slice` is handed the `Option` rather than being run inside it.
///
/// The same memo and the same bargain; what differs is only the shape at the
/// call site, and the difference earns a second hook because the alternative is
/// a `Memo<Option<Option<T>>>` that every caller then has to flatten. Three
/// kinds of reader want this one:
///
/// - the answer is a **fallback**, not an absence — a menu row's greyed state is
///   `Command::enabled(None)` before the engine is up, which is a real answer
///   about the act rather than a missing one;
/// - the slice is itself an `Option` — "the selected frame, if the selected
///   layer is one" — where the outer `Option` says nothing the inner one does
///   not;
/// - the slice wants `?` over the projection, which a `&ObservableState`
///   closure has nowhere to spend.
///
/// A hook: call unconditionally, like [`use_obs`].
pub fn use_obs_opt<T>(
    state: AppState,
    slice: impl Fn(Option<&ObservableState>) -> T + 'static,
) -> Memo<T>
where
    T: PartialEq + 'static,
{
    use_memo(move || slice(state.obs.read().as_ref()))
}

/// Subscribe to **one slice** of this browser's preferences, for [`use_obs`]'s
/// reason: a write of [`Signals::prefs`] wakes every reader of the whole record.
///
/// A hook: call unconditionally. A reader with no hook to call — a plain function
/// asked from many renders, or another memo's closure — reads a slice kept on
/// [`Signals`] instead ([`Signals::chrome_hiding`], [`Signals::hdr_on`]).
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

/// Whether the canvas can no longer take a mark, so the app has stopped painting
/// and dispatching (§5).
///
/// **Asked of exactly what the report is mounted on**, which is the whole design:
/// keying the shutdown on the same fields `crate::failure` reads makes "the app
/// has stopped" and "the artist has been told" one fact rather than two that
/// could disagree — a stop with no dialog is a frozen canvas, and a dialog over a
/// running app is a lie. So this asks the *projection* rather than the engine,
/// though both can answer ([`Renderer::gpu_healthy`]).
///
/// Two fields, because there are two ways to get here and only one of them has a
/// projection to carry it: a device that died mid-session, and a device that
/// never arrived ([`Signals::startup_failure`](crate::state::Signals::startup_failure)). The second changes nothing on
/// its own — with no renderer the doors below already answer `None` — and is
/// asked anyway, so that the sentence above stays true of both.
///
/// `peek`: called from the doors below, which are not renders, and from event
/// handlers. Nothing here should widen what a pointer move wakes.
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
        stark_engine::timing::span!(stark_engine::timing::FRAME);
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
/// For engine entry points that are not commands, such as leaving a session. A command
/// goes through [`dispatch`], which is this plus the broadcast, and replacing the
/// document through [`replace_document`]; outside this module, a closure here that calls
/// `Renderer::process` belongs in one of those.
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
    let mut renderer = state.renderer.0;
    let out = {
        let mut guard = renderer.write();
        let r = guard.as_mut()?;
        let out = f(r);
        // Inside the guard, as `dispatch` has always done it: `obs.set` marks
        // subscribers dirty but renders nothing synchronously, so no reader can
        // observe the renderer mid-borrow.
        publish(state, r.session.engine().observe());
        out
    };
    request_paint(state);
    Some(out)
}

/// Hand `snapshot` to the chrome — **only when it moved**.
///
/// A `Signal` write dirties every subscriber whether or not the value changed, so a
/// command the engine declined — a slider dragged back to the value the document
/// already holds, a rename to the name a layer already has — would otherwise wake every
/// memo in the chrome to report the answer it is already showing. `ObservableState` is
/// `PartialEq` for exactly this, and the comparison is cheap: the layer list is shared,
/// so the expensive field answers by pointer (`stark_engine::Layers`).
fn publish(state: AppState, snapshot: ObservableState) {
    let mut obs = state.obs.0;
    // Bound to a local rather than inline in the `if`: a guard held in a condition
    // outlives the body, and here it would be a read of `obs` live across the write.
    let moved = obs.peek().as_ref() != Some(&snapshot);
    if moved {
        obs.set(Some(snapshot));
    }
}

/// Replace the document — open a file, join a session, start afresh — then paint inline
/// and publish. The frontend's half of [`Session::replace`], which installs what the new
/// document owes and frames the piece.
///
/// A door rather than a [`dispatch`] because a replacement is not a command: it commits
/// nothing to the log, broadcasts nothing and is no deed the tour counts (§4, §24), yet
/// it moves everything the chrome shows.
///
/// `None` is [`with_engine`]'s "no engine to move"; `Some(Err)` is a refusal with the
/// open document untouched.
///
/// **Leaves any live session first**, even when the engine then refuses. A replacement
/// drops the engine to solo authoring but not the pump, the phase or the ticket in the
/// URL, which would go on merging peers' actions into the new document (§12.4). Outside
/// the engine hold because [`collab::leave`] takes the engine for its farewell. A join
/// leaves nothing: [`collab::join`] installs its session only after this returns.
///
/// The paint is inline so the first frame shown is already the framed new document
/// rather than the old view over it.
///
/// [`Session::replace`]: stark_ui::session::Session::replace
#[must_use = "a refusal leaves the old document on screen, which the caller has to say"]
pub fn replace_document(
    state: AppState,
    replacement: Replacement<'_>,
    owed: &[(AssetNeed, &[u8])],
) -> Option<stark_engine::Result<()>> {
    collab::leave(state);
    let replaced = with_engine_quiet(state, |r| {
        let seen = r.session.replace(replacement, owed);
        if seen.is_ok() {
            r.paint();
        }
        seen
    })?;
    let out = match replaced {
        // The replacement's own projection, rather than a second one built here.
        Ok(seen) => {
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

/// Reach the engine as `&mut` **without** publishing anything or asking for a
/// frame — for work that cannot change what `observe` projects.
///
/// That is a real and large category, which is why this exists rather than a
/// `dispatch` everywhere: rendering, the readbacks (export, the eyedropper, the
/// navigator's miniature), draining the outbox and the presence tick, and
/// installing asset bytes an action will later name. None of it moves a value the
/// chrome shows, and several run at pointer or frame rate where the `observe` walk
/// and the VDOM diff behind a publish would be pure cost (see [`dispatch_samples`]).
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
    let Some(snapshot) = state
        .renderer
        .peek()
        .as_ref()
        .map(|r| r.session.engine().observe())
    else {
        return;
    };
    publish(state, snapshot);
}

/// Hand the freshly-built engine to the app and publish its opening projection.
///
/// The order is the point: the projection is taken from `r` *before* it is moved
/// into the signal, because everything that reads the signal is written on the
/// assumption that a renderer being there means the chrome already describes it.
pub fn publish_renderer(state: AppState, r: Renderer) {
    let mut renderer = state.renderer;
    let mut obs = state.obs;
    obs.0.set(Some(r.session.engine().observe()));
    // The thumbnail generator's half of the engine, published here rather than
    // fetched from the renderer signal on demand: it needs the device and the
    // pipelines, not the canvas, and holding those directly is what keeps it from
    // borrowing a live renderer to make a picture of a brush (`crate::thumbs`).
    let mut shared = state.thumbs.shared;
    shared.set(Some(r.session.engine().shared()));
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
    let command = command.into();
    // Before the engine sees it, which is what lets the tour ask what a command
    // *changes* — `SetBrush` means one thing beside the brush the engine is still
    // holding and nothing at all beside the one it is about to hold (§24).
    //
    // Here rather than at the handlers that produce these commands, on this
    // function's own argument: it is the one seam, so a tour hung off it counts a
    // way of doing something that was added after it was written.
    crate::tutor::observe(state, &command);
    with_engine(state, |r| r.process(command));
    collab::flush_outbox(state);
}

/// Apply the pointer-rate samples of an in-flight gesture, oldest first: integrate
/// and repaint, skipping the observable refresh and the outbox flush [`dispatch`]
/// does.
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
///
/// A batch because one delivered `pointermove` carries every report the browser
/// coalesced into it: the engine door is taken and the frame requested once per
/// event, not once per report.
pub fn dispatch_samples(state: AppState, samples: &[InputSample]) {
    if samples.is_empty() {
        return;
    }
    let processed = with_engine_quiet(state, |r| {
        for &sample in samples {
            // The partner row to `frame`: one span per report that reaches the
            // engine, so its count over the window is the input rate the engine
            // heard — whether a 240 Hz pen is heard at 240 Hz or at 60 is a
            // question only this row answers.
            stark_engine::timing::span!(stark_engine::timing::INPUT_SAMPLE);
            r.process(GestureCommand::To { sample });
        }
    });
    if processed.is_some() {
        request_paint(state);
    }
}

/// Apply a pointer-rate report of where the hand is: integrate and repaint,
/// skipping the observable refresh and outbox flush [`dispatch`] does —
/// [`dispatch_samples`]' bargain, for its reason. The hover mark (§18.1.10) and
/// the guide rays through the cursor (§20.9) both change what the canvas shows
/// and nothing the chrome reads.
///
/// Its own door rather than [`dispatch_samples`] so the timing table's
/// `input.sample` row keeps meaning "stroke input reaching the engine" — hover
/// work is priced on the engine's own `input.hover` row (§7.1).
pub fn dispatch_hover(state: AppState, command: impl Into<InputCommand>) {
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
/// rAF is a visible flash.
///
/// **Inline, but not unconditional**, and that is the whole of what a resize may
/// help itself to. A drag on the window edge reports a new size every animation
/// frame for as long as the hand is moving, and this is the one paint in the app
/// that does not pass [`schedule_paint`]'s back-pressure. Painting anyway submits a
/// frame the GPU has not caught up with, once per frame, for the length of the drag
/// — and each of those frames also rebuilds every compositing attachment, because
/// they are sized by the target and the target just changed
/// (`composite::Compositor::ensure_targets`). On a large window over a complex
/// document that is hundreds of megabytes of fresh render targets per frame against
/// a queue that is already behind: the device is lost, and the report reads as a GPU
/// failure rather than as the flood that caused it. So the inline paint is offered
/// only while the GPU is keeping up; when it is not, the frame queued below shows
/// the new size an animation frame later, which is what a moment of stretch costs.
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
        if !r.gpu_behind() {
            r.paint();
        }
    });
}

/// Read the current [`BrushConfig`](stark_ui::brush_config::BrushConfig), mutate a
/// copy, and commit it: the signal takes the new configuration and the engine
/// takes its projection (`ViewCommand::SetBrush`), in that door and no other —
/// which is what keeps the two views of the brush the same brush.
///
/// The dispatch goes **first**: the tour reads a `SetBrush` beside the brush
/// still held (§24), and the held one is this signal's.
///
/// **The loud door, deliberately, even though this runs at pointer rate** — the
/// brush-tuning drag and the eyedropper both come through here on every move.
/// [`dispatch_samples`] would be wrong: a tune drag's whole answer is read off the
/// Brush panel's sliders (`input::Tune`), and the eyedropper's off the Color
/// panel's swatch, so the publish is not overhead here but the point of the
/// gesture. What made it expensive was that *every* panel woke for it; that is
/// [`use_obs`]'s business, and the panels that do not show a brush now sleep
/// through this. What is left is `observe`'s own walk of the layer tree, which
/// only splitting the projection by cadence can remove (U1).
/// Returns nothing, though one caller wants what it set — the flow half of the tuning
/// drag, whose readout has to show a number its own knob computes inside `f`. It takes
/// it out through the closure rather than from a return value here, and that is the
/// cheaper trade by a wide margin: most of the thirty callers are `oninput` handlers
/// where a returned value stops the closure coercing to the `Callback` the prop wants,
/// so answering would cost a discarded value at every one of them.
pub fn update_brush(
    state: AppState,
    f: impl FnOnce(&mut stark_ui::brush_config::BrushConfig, &mut stark_ui::brush_config::Transient),
) {
    let mut config = *state.brush.peek();
    let mut tune = *state.transient.peek();
    f(&mut config, &mut tune);
    // The one door the live brush goes through, so this is where it is held to a brush
    // the renderer will actually draw (`BrushConfig::settle`, §6.2): the stretch has to
    // give where the size the drag just moved cannot carry it, and the size has four
    // writers a clamp on the stretch control would be bypassed by.
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

    /// The claim [`AppState`] is a newtype for: the handle is **one pointer**,
    /// whatever [`Signals`] grows to hold.
    ///
    /// Worth pinning rather than trusting, because the cost it replaced was
    /// invisible: `Signals` is eighty-odd `Signal`s at three words apiece, and
    /// the struct was a by-value parameter on some two hundred and seventy
    /// functions. A field added to `Signals` cannot make this fail; a field added
    /// to `AppState` — which is what somebody reaching for "just one more flag"
    /// would do — is exactly what it catches.
    #[test]
    fn the_handle_is_one_pointer() {
        assert_eq!(
            size_of::<AppState>(),
            size_of::<&'static Signals>(),
            "AppState carries something beside the pointer to its signals"
        );
        // And the thing it points at is the large one, so the test above is not
        // passing because there is nothing to hold.
        assert!(
            size_of::<Signals>() > 64 * size_of::<usize>(),
            "Signals is unexpectedly small: {} bytes; is it still the whole state?",
            size_of::<Signals>()
        );
    }
}
