//! Stark's Dioxus **web** frontend (§11).
//!
//! The backend runs in WASM and paints through a WebGPU surface bound to the
//! page's `<canvas>` — the engine renders straight into the surface texture
//! after each command, with no GPU→CPU readback. The canvas fills the window;
//! unobtrusive floating panels (color, brush, layers) sit on top.
//!
//! Run with `dx serve --web -p stark-dioxus-frontend` in a WebGPU-capable browser.
//!
//! # What is in this file
//!
//! The root, and only the root: `main`, the tracing install, and [`app`] — which
//! is the list of what is on screen and the order it is stacked in, plus the
//! startup task that builds the engine and everything that has to wait for one.
//!
//! The three things it used to also be are beside it now. [`crate::canvas`] is
//! the painting surface and the press ladder that decides what a pointer on it
//! means (§25.4); [`crate::overlays`] is the chrome that rides over it — peer
//! cursors, the brush ring, the tow string; [`crate::rail`] is the command rail,
//! its menus and the search palette. Each of those is one subject with one long
//! argument, and reading `app`'s stacking order should not mean scrolling past
//! all three.

// `rsx!` lowers every interpolated attribute and text node — `id: "{CANVAS_ID}"`,
// `"{title}"` — through `format!`, so clippy sees a `format!` with nothing to
// format and suggests `.to_string()`. The suggestion applies to the expansion,
// not to anything writable in the source: plain interpolation *is* the idiom
// here. Suppressed crate-wide because it fires wherever rsx! does.
#![expect(
    clippy::useless_format,
    reason = "rsx! expands plain interpolation through format!; the suggestion applies to the expansion"
)]

mod anchor;
mod brush_editor;
mod builtin_ids;
mod builtins;
mod canvas;
mod collab;
mod commands;
mod credits;
mod drags;
mod failure;
mod files;
mod gradients;
mod icons;
mod identity;
mod images;
mod input;
mod layer_thumbs;
mod layout;
mod modes;
mod navigator;
mod overlays;
mod panels;
mod platform;
mod prefs;
mod presets;
mod preview;
mod rail;
mod records;
mod render;
mod settings;
mod shapes;
mod slots;
mod state;
mod substrates;
mod thumbs;
mod timings;
mod tutor;
mod visibility;
mod widgets;

use dioxus::prelude::*;

use brush_editor::BrushEditorModal;
use canvas::Canvas;
use input::{bind_context_menu, bind_pen, bind_shortcuts};
use layout::PanelStack;
use layout::{resize_end, resize_move};
use navigator::NavigatorOverlay;
use overlays::{BrushCursor, PeerCursors, PickLoupe, TowStringOverlay, TuneReadoutOverlay};
use panels::brush::PresetSaveModal;
use panels::lighting::{DEFAULT_ENVIRONMENT, environment_asset};
use panels::{
    FilterBar, FrameBar, FrameOverlay, GradientBar, GradientBarOverlay, GradientTraceOverlay,
    GuideEditOverlay, PerspectiveGuideBar, PickBar, SelectionBar, StackPopouts, TimelineBar,
    TraceBar, TransformBar, TransformOverlay,
};
use platform::canvas_by_id;
use rail::CommandRail;
use render::CANVAS_ID;
use slots::SlotOverlay;
use stark_engine::command::ViewCommand;
use state::{AppState, update_brush};

/// The UI's global stylesheet — panel chrome (shared CSS custom properties) plus
/// every component class referenced below. Linked once by [`app`] so the rsx!
/// blocks carry class names, not inline styles.
static STARK_CSS: Asset = asset!("/assets/stark.css");

fn main() {
    #[cfg(target_arch = "wasm32")]
    {
        console_error_panic_hook::set_once();
        install_tracing();
    }
    dioxus::launch(app);
}

/// Point `tracing` at both of the places this app reads it: the browser console, and
/// the timing histograms behind `timings::TimingModal` (`stark_engine::timing`,
/// §7.1).
///
/// Two layers over one registry rather than `tracing_wasm::set_as_global_default`,
/// and the **filter is the whole reason** it is written out here.
/// `TimingFilter::timing` (inside `timing::layer`) and `TimingFilter::logging` are
/// exact complements by construction, so no event is dropped and none is handled
/// twice — and neither this file nor the engine restates what a timing span is.
///
/// The negative half is not tidiness. `tracing_wasm`'s layer calls
/// `performance.mark` on every span it is shown and `performance.measure` on every
/// exit, formatting a `String` for each: unfiltered, the dozen-odd phases a painted
/// frame opens would spend two JS calls apiece reporting Stark's own instrumentation
/// into the devtools timeline, sixty times a second, for nobody.
#[cfg(target_arch = "wasm32")]
fn install_tracing() {
    use tracing_subscriber::Layer;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::registry::Registry;

    let console = tracing_wasm::WASMLayer::new(tracing_wasm::WASMLayerConfig::default())
        .with_filter(stark_engine::timing::TimingFilter::<false>);
    let subscriber = Registry::default()
        .with(stark_engine::timing::layer())
        .with(console);
    // A second install would be a second set of histograms with the first still
    // collecting, so it is refused rather than papered over — but `main` runs once
    // per page, so the only way to reach the error is a change to this file.
    if tracing::subscriber::set_global_default(subscriber).is_err() {
        web_sys::console::warn_1(&"stark: a tracing subscriber was already installed".into());
    }
}

fn app() -> Element {
    // Root-owned, because the collaboration pumps and the renderer's async init are
    // detached tasks living in `ScopeId::ROOT` — see `state::root_signal`.
    let state = AppState::new();
    use_context_provider(|| state);

    // The keyboard shortcuts live on the window, not on the root element below, so
    // they answer whatever has focus — including `document.body`, where the browser
    // leaves it after a clicked button unmounts itself (see `platform::on_window_key`).
    use_hook(|| bind_shortcuts(state));
    // And the pen's other end, on the window for the same reason: it is a hold
    // like a number key's, so it belongs to no one surface (§18.1.8).
    use_hook(|| bind_pen(state));
    // And the browser's context menu, refused on the same grounds: a pen held
    // still is this app's own gesture (§6.9), and Windows reads the same hold as
    // a right-click.
    use_hook(bind_context_menu);
    // And the way out: the browser asks before the tab goes, if there is committed
    // work here that neither Save nor Export has taken out (`files::guard_unload`).
    // In the root's body rather than at the end of the startup task, unlike the
    // launch queue and the paste hook below — this one binds a predicate over the
    // signals, so it wants no engine and cannot be left unbound by a start that
    // fails before there is one.
    use_hook(|| files::guard_unload(state));

    // Before anything reads the browser's store, say what the store *is*: the format
    // is `stark_chrome`'s and shared with the native frontend, and where the bytes go
    // is this frontend's alone (§11.2). Nothing here fails without it — every read
    // would answer "nothing stored" — which is why it is an install rather than a
    // constructor everything else threads.
    use_hook(|| stark_chrome::storage::install(platform::LocalStore));
    // Then drop the keys the old formats were kept under, so their bytes are not
    // still spending this origin's quota (`storage::drop_retired`, which says when to
    // delete this line).
    use_hook(stark_chrome::storage::drop_retired);

    // The brush presets follow the browser rather than the document (seeded with
    // the built-ins on a browser that has never stored any). The shape library
    // follows it too, but its bytes are in the blob store now, so reading it is a
    // fetch — it is loaded in the startup task below instead of here (§25.6).
    use_hook(|| presets::load(state));
    // The gradient library follows the browser the same way (§22.3) — and has no
    // built-ins to install later: every entry is something this user traced.
    use_hook(|| gradients::load(state));
    // And so does the quick-brush rack (§18.1.8) — a browser that has never set
    // a slot is seeded below, once there is a preset library to seed it from.
    use_hook(|| slots::load(state));
    // The ⚙ dialog's settings follow the browser the same way. Applied here, in the
    // root's own body, so the very first render is already in the mode the user left
    // the app in — the engine-owned half of them waits for the renderer below
    // (`crate::prefs`).
    use_hook(|| prefs::load(state));
    // And this browser's rebound shortcuts (`stark_chrome::commands::Bindings`) — before the
    // first keystroke could ask the table, like everything above it.
    use_hook(|| commands::load(state));
    // And its rebound canvas drags (§25.8), which the very first press asks and
    // which also carry whether the preset offer has already been made — so a
    // browser that has seen it must not be shown it again on this visit.
    use_hook(|| drags::load(state));

    // Every brush with a picture to show wants a rendered stroke (`crate::thumbs`,
    // §11): the preset library's rows and the quick-brush rack's overlay
    // (§18.1.8). In the **root** rather than in either viewer, because neither is
    // always mounted — the Brush panel closes, and the rack's overlay exists only
    // while a key is held, which is far too late to start rendering the thing it
    // is there to show. Generation needs the main renderer, so this watches the
    // renderer signal alongside the two libraries: whichever lands last kicks it
    // off, and a slot tuned under a hold re-runs it on the release that stores it.
    //
    // The renderer is watched through `renderer_ready` rather than by asking the
    // renderer signal whether it holds one: reading that signal subscribes to every
    // *write* of it, and every door into the engine takes it as `&mut` — so this
    // effect used to re-run on every command and every pointer sample of a stroke,
    // rescanning the whole library each time, to learn a boolean that moves once
    // (U2, `state::renderer_ready`).
    use_effect(move || {
        let _ = state.presets.read().len();
        let _ = state.slots.brushes.read().len();
        let _ = (state.renderer_ready)();
        thumbs::refresh(state);
    });

    use_hook(|| {
        spawn(async move {
            // The browser's three answers about WebGPU, all of which used to be
            // `expect` (`render::StartupFailure`). A browser that has none is the
            // most likely way this app fails a first-time visitor, and it failed
            // it by killing this task and leaving the chrome standing over a
            // canvas that would never take a mark. Now it says so, on the same
            // surface a device that dies mid-session says it on
            // (`crate::failure`) — and nothing below runs, which is what it
            // already did.
            let mut r = match render::init(canvas_by_id(CANVAS_ID)).await {
                Ok(r) => r,
                Err(why) => {
                    tracing::error!(%why, "the canvas cannot be drawn on");
                    let mut failed = state.startup_failure;
                    failed.set(Some(why));
                    return;
                }
            };
            // Fetch the bundled brush shapes at runtime (kept out of the wasm
            // binary) and import them once, so the gallery's built-in cards are
            // ready — and so the default presets have ids to name
            // (§6.6, `crate::builtins`).
            builtins::import_all(&mut r).await;
            // Fetch the default substrate's height map and open the document on it
            // (§6.4, §6.6). A substrate is named by the hash of its image, so this
            // cannot be done the other way round any more: the engine boots on
            // `Flat` — the one substrate it can name without bytes — and the id it
            // moves to is only knowable once those bytes are in hand.
            //
            // `new_document` rather than a `SetSubstrate`, so no bogus first step
            // lands in the undo history of every fresh document. It replaces a
            // document nobody has touched: the renderer signal is not published
            // until this whole block finishes, so nothing can have been painted yet.
            let color_space = r.color_space();
            substrates::open_default(&mut r, color_space).await;
            // Fetch the default environment's HDR and light the canvas with it
            // (§6.3); until it arrives the procedural neutral one is used,
            // and the Lighting panel can switch back to it at any time. A no-op while
            // the default *is* the procedural one, which has no bytes to fetch.
            if let Some(asset) = environment_asset(DEFAULT_ENVIRONMENT)
                && let Ok(bytes) = dioxus::asset_resolver::read_asset_bytes(asset).await
            {
                // Switch only once the bytes are known to decode: the light this
                // build ships should never fail here, and if it does the canvas
                // stays on the procedural one rather than on a light that is not
                // there.
                match r.register_environment(DEFAULT_ENVIRONMENT, bytes) {
                    Ok(()) => r.process(ViewCommand::SetEnvironment(DEFAULT_ENVIRONMENT)),
                    Err(e) => {
                        tracing::warn!("the bundled environment will not decode: {e}");
                    }
                }
            }
            // Every fetch above is a window in which the canvas can be laid out —
            // and any resize reported during it was dropped, because the signal
            // `state::resize` needs is the one being set two lines down. Re-read
            // the element here, where a size can no longer go missing, so the
            // first frame is painted through the viewport the canvas actually has
            // (`Renderer::sync_to_canvas`). No `await` between this and the set.
            r.sync_to_canvas();
            r.paint();
            // Projection first, then the engine — `publish_renderer` is that order,
            // so no reader ever sees a renderer the chrome cannot yet describe.
            state::publish_renderer(state, r);

            // The custom shape library, from the browser's two stores (§25.6). Here
            // rather than in a `use_hook` above because its bytes are a fetch now —
            // and *before* `apply_first` below, which is the first thing that turns a
            // preset's stamp id back into bytes: a library that had not arrived yet
            // would put those brushes silently on the round tip.
            shapes::load(state).await;

            // And the custom surface library, on the same footing and for a sharper
            // version of the same reason: a document opened from a file may name a
            // substrate this browser holds, and a library that had not arrived yet would
            // leave the gallery unable to say so (§6.4).
            substrates::load(state).await;

            // The app's own presets join the library now rather than at
            // `presets::load`: they name bundled brush shapes, and a stamp is
            // named by content id — which the imports just above are what
            // produce. Every start, not only a first one, so an improved default
            // reaches a browser that has been running Stark for months.
            presets::install_builtins(state);

            // And the rack under the number keys, from that library — after it,
            // because it takes each preset to the digit that preset declares
            // rather than restating the list here (§18.1.8). A browser that has
            // already set a slot keeps what it set; only an untouched rack is
            // filled in.
            slots::seed_defaults(state);

            // The brush this app start begins on: the library's first preset (an
            // empty library leaves the app's default brush), and then the color
            // the Color panel is already showing — the brush signal is seeded
            // with `INITIAL_COLOR` (`state::Signals::new`) so the picker the
            // panel mounted before the engine existed shows it, and the
            // identity write here is what pushes the same configuration to the
            // engine, which opened on black. Both go through
            // `ViewCommand::SetBrush`, which is session state, so neither
            // leaves a step in the undo history. Once per app start, not per
            // document: a new document keeps the brush the user is holding.
            presets::apply_first(state);
            update_brush(state, |_, t| t.color = panels::color::INITIAL_COLOR);

            // The settings that live in the engine rather than in a signal — there is
            // one, and it is read by the session this block may be about to join, so it
            // goes in before the join rather than after (`crate::prefs`).
            prefs::load_engine(state);

            // A `#stark…` fragment in the page URL is a session invitation:
            // join it now that the engine is up (§12.4).
            if let Some(ticket) = collab::url_ticket() {
                tracing::info!("joining shared session from URL fragment");
                collab::join(state, ticket);
            }

            // And the other thing a launch can carry: a `.stark` the OS was asked
            // to open in this app (§11). Bound *here*, at the end of startup,
            // rather than in the root's body — setting the consumer is what
            // delivers a queued launch, and a document has nowhere to load until
            // the renderer above exists.
            files::bind_file_launch(state);
            // …and the third way a picture can arrive: pasted (§23). Bound here
            // beside the launch queue and for a related reason — a paste has an
            // engine to place into only once the renderer above exists.
            images::bind_paste(state);

            // And last of all, the guided tour starts listening (§24). Last is the
            // whole of why it is here: every line above dispatches commands on the
            // user's behalf — the opening preset, the opening color, the stored
            // preferences — and a tour armed before them would count the app
            // starting up as the artist at work.
            tutor::begin(state);
        });
    });

    rsx! {
        document::Stylesheet { href: STARK_CSS }

        div {
            class: "app-root",
            // Two modes that the whole tree is inside of, so both ride the root element
            // and the stylesheet does the rest — one rule each, rather than every
            // control learning about a state it does not own.
            //
            // Timeline mode reserves the foot of the window for its bar, and the
            // stylesheet lifts the bottom-bars column clear of it. Minimal mode hides
            // the `.label` spans the chrome marks its words with (`icons::label`); it
            // is a class here rather than a prop for the same reason — a boolean
            // threaded through the tree would have to reach every control, and the one
            // that failed to pass it on would be the one that kept its word.
            class: if (state.timeline.open)() { "timeline-mode" },
            class: if (state.minimal)() { "minimal" },
            // A panel resize by the bottom-edge grip is driven here — events bubble up even
            // over the canvas, so it keeps tracking wherever the pointer goes, and leaving
            // the window ends it so it cannot get stuck. A no-op unless armed.
            //
            // The title-bar *reorder* drag is not here: it captures the pointer and handles
            // its own move and release (`layout::Panel`), which is the only way to be sure
            // of getting the release.
            onpointermove: move |e| resize_move(state.panels, &e),
            onpointerup: move |_| resize_end(state.panels),
            onpointerleave: move |_| resize_end(state.panels),

            // Dropping a file anywhere on the app (§23.4). At the **root** rather
            // than on the canvas, and that is not for convenience: an unclaimed
            // drop is one the browser handles, and what the browser does with a
            // dropped file is navigate to it — discarding an unsaved painting. So
            // every drop over the window has to be claimed, including the ones that
            // land on a panel, and `preventDefault` on `dragover` is what makes the
            // window a drop target at all.
            //
            // A descendant that wants a drop for itself takes it by stopping
            // propagation — the brush editor's shape library is the one that does
            // (`brush_editor`), because dropping a stamp into the library and
            // placing a photograph on the canvas are different acts.
            ondragover: move |e| e.prevent_default(),
            ondrop: move |e| {
                e.prevent_default();
                images::drop_files(state, &e);
            },

            Canvas {}

            // The frame's edges and handles, over the canvas but *under* all the
            // floating chrome. Mounted only while a frame is selected for composing
            // (§15.7); its interior passes pointer events through, so
            // painting inside the frame is unaffected.
            FrameOverlay {}

            // The transform gesture's box and handles, over the canvas while the
            // selected paint is being composed (§16.6). Its
            // catcher blocks canvas painting for the mode's duration.
            TransformOverlay {}

            // The drawing-guide edit mode's catcher, while a perspective grid
            // is being composed (§20.5): dragging orbits the camera, the 45°
            // circle drags the lens, the crosshair moves the construction.
            GuideEditOverlay {}

            // The gradient trace's catcher and rubber line, while the library
            // pop-out's Trace is armed (§22.2).
            GradientTraceOverlay {}

            // The gradient fill's catcher and axis chrome, while the Selection
            // bar's Gradient is composing (§22.4).
            GradientBarOverlay {}

            // Order among those four says nothing, because at most one of them
            // is ever mounted: entering any mode leaves whichever was live
            // (`modes::leave`). Do not read it as a priority — the four catchers
            // all sit at the same z-index, where the *last* sibling takes the
            // pointer rather than the first. A rule that cannot be got wrong is
            // better than one stated correctly in four places.

            // Collaborators' pointers, over the canvas and under the chrome
            // (§17.4). Empty and free when solo.
            PeerCursors {}

            // The live brush under the resting pointer (§18.1.10), same layer:
            // the size the next stroke would land at, riding the hover. Empty
            // and free while the pointer is off the canvas.
            BrushCursor {}

            // The brush-tuning drag's readout — the size ring or the flow bar — in
            // the same layer and mounted for the same reason (§18.1.9). Empty and
            // free unless a tuning drag is in flight.
            TuneReadoutOverlay {}

            // The tow string while a smoothing brush draws (§6.11), same layer
            // again. Empty and free unless a rope is in flight.
            TowStringOverlay {}

            // The held touch pick's swatch (§18.1.11), same layer once more.
            // Empty and free unless a finger is holding one.
            PickLoupe {}

            // Left command rail: rarely-used document commands, tucked away.
            CommandRail {}

            // The chrome down the left, under the rail: the quick-brush rack while
            // a number key is held (§18.1.8), and the navigator's miniature in the
            // corner (§11). Neither is a panel — one is summoned by a finger already
            // on the keyboard, the other is a picture read at a glance — and both
            // stand on this edge, which is the whole reason they share a box.
            //
            // A column rather than two independently placed overlays, because the
            // two are sized by things that have nothing to do with each other: the
            // rack by how many digits are filled, the miniature by the artwork's
            // aspect. Any pair of fixed offsets that kept them apart on this window
            // would put one over the other on a shorter one. Here the navigator
            // takes what it needs from the bottom and the rack centres its rows in
            // whatever is left, so "they do not overlap" is a fact about the layout
            // rather than an arithmetic nobody re-checks.
            //
            // The box itself takes no pointer events; its children decide for
            // themselves (the rack only while it is pinned, the miniature always).
            div { class: "left-chrome",
                SlotOverlay {}
                NavigatorOverlay {}
            }

            // Floating tool panels, stacked top-right — order + visibility are data-driven.
            PanelStack {}

            // Whatever a panel has flown out beside itself: the canvas colour's
            // picker, the surface gallery (§25.7). Here rather than in the panel
            // because the column clips, so a pop-out has to be placed against its
            // row's measured box instead of drawn in it (`panels::popout`). After
            // the stack in the DOM for the tour card's reason — it stands *over*
            // the panel it flew out of — and empty whenever none is open.
            StackPopouts {}

            // The guided tour's card, beside whichever panel or bar the lesson on
            // screen is about (§24). After the stack in the DOM, which is what puts
            // it over the panel it points at rather than under it — it is positioned
            // against that panel's own measured box, so an overlap of a pixel or two
            // where the shadows meet is expected and must land the right way round.
            // Empty and free whenever no lesson is showing.
            tutor::TutorCard {}

            // Bottom-centre: the bars that are mounted only while the thing they act
            // on exists. Stacked in one column so a frame and a selection in force at
            // the same time sit above one another instead of on top of each other.
            //
            // Two registers share the column (MODAL_DESIGN.md): standing-state
            // bars describe a fact and dissolve with it, while a composing
            // mode's bar (`mode-bar`) fronts a catcher that has taken the
            // pointer, wears the armed accent, and stands the others down —
            // *recessed* rather than unmounted, so the place its Done and Esc
            // return to stays on screen.
            // The column is anchored at its foot, so an earlier child stands
            // *higher* — and the order here is the stack's: deepest first, so
            // each bar lands **above** the one it covers, the way a card lands
            // on a pile. A trace stands over the gradient bar it parked, a
            // mode over the standing bars it recessed.
            //
            // The covered bars leave the flow entirely (`.recessed` is
            // absolute), which is what holds the live bar still: it is the
            // column's one in-flow child, at the baseline whatever the stack
            // holds, and the bars it covers peek out beneath it — lower and
            // fainter with depth, priced by a sibling chain that this order
            // is load-bearing for.
            div { class: "bottom-bars",
                // The eyedropper's options, present only while Alt arms it
                // (§18.0.2). First, deliberately: in a foot-anchored column a
                // child mounting *above* the others moves none of them, so the
                // rack coming and going with the Alt key never bumps the bar
                // somebody is working in. It is no part of the stack below —
                // it coexists with painting rather than covering anything — so
                // it floats above the pile.
                PickBar {}
                // The trace's name and its Cancel while one is armed (§22.2) —
                // the deepest the stack goes, since a trace is the one mode
                // entered from inside another mode's bar. Before this bar the
                // mode had no standing indicator at all, only a lit chip inside
                // a pop-out that closes the moment the mode starts.
                TraceBar {}
                // The transform gesture's flips and "Done", standing over the
                // selection bar while one is composing (§16.6).
                TransformBar {}
                // The gradient fill's axis kinds and "Done", standing the same
                // way while a ramp is being composed (§22.4) — and staying,
                // recessed under the trace's bar, while one has it parked
                // (`state::gradient_resume`).
                GradientBar {}
                // The drawing-guide edit mode's controls — locks, axis
                // visibility, cell count, opacity — while a perspective grid is
                // being composed (§20.5).
                PerspectiveGuideBar {}
                // The whole selection's opacity and commands, present while
                // there is a selection or a shape tool is armed to make one — so
                // it doubles as the "canvas is masked" indicator.
                SelectionBar {}
                // The frame's composition controls, present only while a frame is
                // selected for composing (§15.7).
                FrameBar {}
                // The selected filter layer's own numbers, present only while one is
                // selected (§21.6) — the same bargain as the frame bar, and mutually
                // exclusive with it by construction, since both key off the single
                // selected layer being of their kind.
                FilterBar {}
            }

            // Timeline mode's own bar, below the bottom-bars column and spanning
            // the window rather than hugging its contents: the others are commands
            // that happen to apply right now, while this is a *mode* the whole
            // canvas is in, and a scrubber wants every pixel of width it can get
            // (§18.2.4). Mounted from here rather than gated inside
            // itself, because it owns hooks — a component may not gain or lose
            // those between renders.
            if (state.timeline.open)() {
                TimelineBar {}
            }

            // The brush editor dialog (mounted only while open, so each open
            // re-inits its preview against the current canvas look).
            if (state.brush_editor_open)() {
                BrushEditorModal {
                    on_close: move |_| {
                        let mut open = state.brush_editor_open;
                        open.set(false);
                    }
                }
            }

            // The name for a new preset, asked for by the brush editor's "Save new
            // preset" (and the command of that name) — after the editor, so it stacks
            // over it (`AppState::root_dialogs`). Mounted only while open, so each
            // open proposes a fresh name for the library as it stands now.
            if (state.preset_save_open)() {
                PresetSaveModal {
                    on_close: move |_| {
                        let mut open = state.preset_save_open;
                        open.set(false);
                    }
                }
            }

            // Last, and over everything: the GPU has died and the canvas is showing
            // its final frame (§5, `crate::failure`). It gates on its own read of
            // the projection rather than on a condition here, because the same read
            // is what stops the engine doors — one field, so the app cannot be
            // stopped without saying so or say so while still running. Empty and
            // free on a healthy device, which is every ordinary session.
            failure::GpuFailureModal {}
        }
    }
}
