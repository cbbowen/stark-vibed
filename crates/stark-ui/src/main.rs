//! Stark's Dioxus **web** frontend (§11).
//!
//! The backend runs in WASM and paints through a WebGPU surface bound to the
//! page's `<canvas>` — the engine renders straight into the surface texture
//! after each command, with no GPU→CPU readback. The canvas fills the window;
//! unobtrusive floating panels (color, brush, layers) sit on top.
//!
//! Run with `dx serve --web -p stark-ui` in a WebGPU-capable browser.

// `rsx!` lowers every interpolated attribute and text node — `id: "{CANVAS_ID}"`,
// `"{title}"` — through `format!`, so clippy sees a `format!` with nothing to
// format and suggests `.to_string()`. The suggestion applies to the expansion,
// not to anything writable in the source: plain interpolation *is* the idiom
// here. Suppressed crate-wide because it fires wherever rsx! does.
#![allow(clippy::useless_format)]

mod brush_editor;
mod builtin_ids;
mod builtins;
mod collab;
mod components;
mod credits;
mod failure;
mod files;
mod gesture;
mod gradients;
mod grounds;
mod icons;
mod identity;
mod images;
mod input;
mod layout;
mod modes;
mod panels;
mod platform;
mod prefs;
mod presets;
mod preview;
mod render;
mod settings;
mod shapes;
mod slots;
mod state;
mod storage;
mod thumbs;
mod timings;
mod widgets;

use std::collections::HashSet;

use dioxus::dioxus_core::spawn_forever;
use dioxus::html::Modifiers;
use dioxus::prelude::*;

use brush_editor::BrushEditorModal;
use components::menubar::{Menubar, MenubarContent, MenubarItem, MenubarMenu, MenubarTrigger};
use credits::CreditsModal;
use icons::{icon, icon_large};
use input::{
    Nav, Paint, Tune, bind_context_menu, bind_pen, bind_shortcuts, end_interaction, pick_color,
    sample,
};
use layout::{PanelId, PanelLayout, PanelStack, chrome_class, resize_end, resize_move};
use panels::brush::PresetSaveModal;
use panels::lighting::{DEFAULT_ENVIRONMENT, environment_asset};
use panels::select::current_tool;
use panels::{
    FilterBar, FrameBar, FrameOverlay, GradientBar, GradientBarOverlay, GradientTraceOverlay,
    GuideEditOverlay, PerspectiveGuideBar, PickBar, SelectionBar, TimelineBar, TransformBar,
    TransformOverlay,
};
use platform::{canvas_by_id, capture_pointer};
use render::CANVAS_ID;
use settings::SettingsModal;
use slots::SlotOverlay;
use stark_engine::command::{DocCommand, PeerCommand, ViewCommand};
use stark_model::ColorSpaceId;
use stark_model::document::SelectionOp;
use state::{AppState, dispatch, dispatch_quiet, resize, update_brush, use_obs};

/// The UI's global stylesheet — panel chrome (shared CSS custom properties) plus
/// every component class referenced below. Linked once by [`app`] so the rsx!
/// blocks carry class names, not inline styles. Custom properties are global, so
/// the css_module menubar styles pick up `--panel-shadow` / `--panel-background` too.
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
/// the timing histograms behind [`TimingModal`] (`stark_engine::timing`, §7.1).
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

    // Floating-panel layout: order + which are open. Provided so the panel chrome and
    // the "Panels" menu can reorder/close/restore them. Panels in `CLOSED_BY_DEFAULT`
    // start hidden but keep their slot, so the menu reopens them where they belong.
    let panels = PanelLayout {
        order: use_signal(|| PanelId::ALL.to_vec()),
        hidden: use_signal(|| HashSet::from(PanelId::CLOSED_BY_DEFAULT)),
        drag: use_signal(|| None),
        heights: use_signal(PanelLayout::default_heights),
        resize: use_signal(|| None),
    };
    use_context_provider(|| panels);

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

    // The shape library follows the browser, not the document — load it before
    // the renderer exists so the gallery is populated on first open. The brush
    // presets follow the browser the same way (seeded with the built-ins on a
    // browser that has never stored any).
    use_hook(|| shapes::load(state));
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
            let mut r = render::init(canvas_by_id(CANVAS_ID)).await;
            // Fetch the bundled brush shapes at runtime (kept out of the wasm
            // binary) and import them once, so the gallery's built-in cards are
            // ready — and so the default presets have ids to name
            // (§6.6, `crate::builtins`).
            builtins::import_all(&mut r).await;
            // Fetch the default ground's height map and open the document on it
            // (§6.4, §6.6). A ground is named by the hash of its image, so this
            // cannot be done the other way round any more: the engine boots on
            // `Flat` — the one ground it can name without bytes — and the id it
            // moves to is only knowable once those bytes are in hand.
            //
            // `new_document` rather than a `SetSurface`, so no bogus first step
            // lands in the undo history of every fresh document. It replaces a
            // document nobody has touched: the renderer signal is not published
            // until this whole block finishes, so nothing can have been painted yet.
            let color_space = r.color_space();
            grounds::open_default(&mut r, color_space).await;
            // Fetch the default environment's HDR and light the canvas with it
            // (§6.3); until it arrives the procedural neutral one is used,
            // and the Lighting panel can switch back to it at any time. A no-op while
            // the default *is* the procedural one, which has no bytes to fetch.
            if let Some(asset) = environment_asset(DEFAULT_ENVIRONMENT)
                && let Ok(bytes) = dioxus::asset_resolver::read_asset_bytes(asset).await
            {
                r.register_environment(DEFAULT_ENVIRONMENT, bytes);
                r.process(ViewCommand::SetEnvironment(DEFAULT_ENVIRONMENT));
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
            // empty library leaves the engine's default brush), and then the
            // color the Color panel is already showing — the panel mounted
            // before the engine existed, so it seeded its picker from
            // `INITIAL_COLOR` alone, and pushing the same color here is what
            // keeps the engine from painting black under a red marker. Both go
            // through `ViewCommand::SetBrush`, which is session state, so
            // neither leaves a step in the undo history. Once per app start, not
            // per document: a new document keeps the brush the user is holding.
            presets::apply_first(state);
            update_brush(state, |b| {
                b.color[..3].copy_from_slice(&panels::color::INITIAL_COLOR)
            });

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
        });
    });

    // Two modes that the whole tree is inside of, so both ride the root element and
    // the stylesheet does the rest — one rule each, rather than every control learning
    // about a state it does not own.
    //
    // Timeline mode reserves the foot of the window for its bar, and the stylesheet
    // lifts the bottom-bars column clear of it. Minimal mode hides the `.label` spans
    // the chrome marks its words with (`icons::label`); it is a class here rather than
    // a prop for the same reason — a boolean threaded through the tree would have to
    // reach every control, and the one that failed to pass it on would be the one that
    // kept its word.
    let mut root_class = String::from("app-root");
    if (state.timeline.open)() {
        root_class.push_str(" timeline-mode");
    }
    if (state.minimal)() {
        root_class.push_str(" minimal");
    }

    rsx! {
        document::Stylesheet { href: STARK_CSS }

        div {
            class: root_class,
            // A panel resize by the bottom-edge grip is driven here — events bubble up even
            // over the canvas, so it keeps tracking wherever the pointer goes, and leaving
            // the window ends it so it cannot get stuck. A no-op unless armed.
            //
            // The title-bar *reorder* drag is not here: it captures the pointer and handles
            // its own move and release (`layout::Panel`), which is the only way to be sure
            // of getting the release.
            onpointermove: move |e| resize_move(panels, &e),
            onpointerup: move |_| resize_end(panels),
            onpointerleave: move |_| resize_end(panels),

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

            // The gradient trace's catcher and rubber line, while the Gradients
            // panel's Trace is armed (§22.2).
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

            // The brush-tuning drag's size ring, in the same layer and mounted for
            // the same reason (§18.1.9). Empty and free unless one is in flight.
            BrushSizeRing {}

            // The tow string while a smoothing brush draws (§6.11), same layer
            // again. Empty and free unless a rope is in flight.
            TowStringOverlay {}

            // Left command rail: rarely-used document commands, tucked away.
            CommandRail {}

            // The quick-brush rack, down the left under the rail, while a number
            // key is held (§18.1.8). Empty and free — and reading nothing — the
            // rest of the time. It takes no pointer events, because the gesture
            // it belongs to is hold-*and-draw*.
            SlotOverlay {}

            // Floating tool panels, stacked top-right — order + visibility are data-driven.
            PanelStack {}

            // Bottom-centre: the bars that are mounted only while the thing they act
            // on exists. Stacked in one column so a frame and a selection in force at
            // the same time sit above one another instead of on top of each other.
            div { class: "bottom-bars",
                // The whole-selection commands, present only while there is a
                // selection — so it doubles as the "canvas is masked" indicator.
                SelectionBar {}
                // The transform gesture's flips and "Done", standing in for the
                // selection bar while one is composing (§16.6).
                TransformBar {}
                // The gradient fill's axis kinds and "Done", standing in the
                // same way while a ramp is being composed (§22.4).
                GradientBar {}
                // The drawing-guide edit mode's controls — locks, axis
                // visibility, cell count, opacity — while a perspective grid is
                // being composed (§20.5).
                PerspectiveGuideBar {}
                // The frame's composition controls, present only while a frame is
                // selected for composing (§15.7).
                FrameBar {}
                // The selected filter layer's own numbers, present only while one is
                // selected (§21.6) — the same bargain as the frame bar, and mutually
                // exclusive with it by construction, since both key off the single
                // selected layer being of their kind.
                FilterBar {}
                // The eyedropper's options, present only while Alt arms it
                // (§18.0.2). Last in the column, so it comes up
                // nearest the canvas — it is the most transient of the three.
                PickBar {}
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

            // "Save preset", asked for by the Brush panel. Mounted only while open, so
            // each open proposes a fresh name for the library as it stands now.
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

/// The full-window painting surface (a WebGPU canvas the engine draws into).
#[component]
fn Canvas() -> Element {
    let state = use_context::<AppState>();
    // The paint gesture itself — the stroke or marquee, from the press that opens
    // one to the release that commits it (`input::Paint`). It owns whether one is
    // in flight and the shape action a modifier overrode, so this component is
    // left with the part that is genuinely its own: deciding, on a press, which of
    // the four bindings it is.
    let paint = Paint::use_paint(state);
    // The shared pan/zoom bindings (`input::Nav`) — the same instance the
    // transform overlay makes for itself, so navigation means one thing.
    let nav = Nav::use_nav(state);
    // Accelerator+drag tunes Size and Flow instead of painting (`input::Tune`,
    // §18.1.9). The canvas's own, unlike `nav`: it moves the brush, and the
    // overlays that navigate have no brush.
    let tune = Tune::use_tune(state);
    // Whether an Alt+drag is sampling color off the canvas rather than painting on
    // it (§18.0.2). Shared rather than local, unlike the two above,
    // because the options bar is mounted on *armed but not dragging*.
    let mut picking = state.pick.dragging;
    // Set for as long as the canvas is the thing being used, which fades the floating
    // chrome out of the way. Pointer gestures clear it on release (`end_interaction`).
    let mut canvas_active = state.canvas_active;

    // Everything this component reads off the projection, in **one** memo — so the
    // canvas is re-rendered when its cursor would change and not when the engine is
    // merely touched (`state::use_obs`). It is the component that can least afford
    // the difference: it is the surface a stroke is being made on, and every sample
    // of that stroke writes the engine.
    //
    // The two facts are:
    //
    // - **Whether the selected layer takes paint.** A frame does not (§15.7).
    //   Rather than block the gesture, say so in the cursor: the brush crosshair
    //   becomes "not-allowed", so the canvas explains itself before the user draws a
    //   stroke that would go nowhere. Panning still works, so the pan cursor wins
    //   while space is held.
    // - **The tool**, for the eyedropper cursor below. It has to be *read* here
    //   rather than peeked as the handlers do (`current_tool`): a peek would leave
    //   the canvas wearing the wrong cursor until some other change happened to
    //   re-render it, which is precisely what subscribing to the whole projection
    //   was accidentally covering up.
    let look = use_obs(state, |o| {
        let paintable = o
            .layers
            .iter()
            .any(|l| l.id == o.active_layer && l.is_paintable());
        (paintable, o.tool)
    });
    let (paintable, tool) = look().unwrap_or((false, stark_model::document::Tool::Brush));
    // Alt arms the eyedropper over the brush, and the cursor says so before it is
    // used — the only thing that makes a modifier binding discoverable. Not over a
    // selection tool, where alt already means "subtract from the selection"
    // (§6.8), so the cursor promises the pick exactly where a press would
    // take one. It beats `no-paint`, because a layer that takes no paint can still
    // be sampled.
    let sampling = (state.pick.alt_down)() && !(state.space_down)() && !tool.is_selection();
    let canvas_class = if sampling {
        "paint-canvas picking"
    } else if paintable || (state.space_down)() {
        "paint-canvas"
    } else {
        "paint-canvas no-paint"
    };

    rsx! {
        canvas {
            id: "{CANVAS_ID}",
            class: canvas_class,
            onresize: move |e| {
                if let Ok(size) = e.get_content_box_size() {
                    resize(state, size.width as u32, size.height as u32);
                }
            },
            // Strokes and pans capture the pointer (like the pads/pickers): leaving the
            // window mid-stroke keeps painting — the infinite canvas extends past the
            // viewport anyway — and the interaction ends on release/cancel, never by
            // crossing the canvas edge.
            onpointerdown: move |e| {
                // Navigation first: a second finger on the glass, middle-drag, or
                // space + the primary button (`input::Nav` — the one definition of
                // the navigation bindings, shared with the transform overlay).
                // Taking it here is also what keeps space+Alt panning rather than
                // sampling.
                if nav.begin(&e) {
                    // Whatever was being drawn was never meant to be paint — it was
                    // the opening half of a pinch (§18.1.7). Cancelled rather
                    // than committed, so reaching for the canvas leaves no mark.
                    paint.abandon();
                    canvas_active.set(true);
                    return;
                }
                // Ctrl+drag tunes the brush rather than painting with it — Size
                // sideways, Flow up and down (§18.1.9). Below `nav`, which is
                // what leaves Ctrl+space a zoom; above the playback guard, because
                // the brush is view state and tuning it commits nothing.
                //
                // Deliberately *not* `canvas_active`, for the eyedropper's reason
                // below: the Brush panel is where this gesture's answer is read,
                // so fading the chrome would hide the one thing it is for.
                if tune.begin(&e) {
                    // A stroke was in flight only if some *other* pointer opened
                    // one; it can no longer be finished by this press, and a
                    // gesture the hand has walked away from must leave no mark.
                    paint.abandon();
                    return;
                }
                // Nothing may be *committed* while the playhead is moving: a
                // commit clears the withheld half of the timeline, so a stroke
                // laid under a running playback would delete the rest of the
                // piece (`panels::timeline`). Panning is taken above and stays
                // available — looking around during playback costs the document
                // nothing.
                if panels::timeline::is_playing(state) {
                    return;
                }
                // The pen's other end draws too — it is a contact like the tip,
                // differing only in the brush it arrives holding (§18.1.8).
                if input::is_contact(&e) {
                    capture_pointer(&e);
                    // Painting and selecting are the same gesture from here — the
                    // tool decides what the engine builds (§6.8).
                    let tool = current_tool(state);
                    // Alt+press samples the canvas instead of painting on it, and
                    // Alt+drag keeps sampling — the binding Clip Studio Paint and
                    // Rebelle both use, so a color is picked up without putting
                    // the brush down (§18.0.2). Alt over a selection
                    // tool is left alone: there it already means subtract.
                    // (Space+Alt never reaches here — `nav` took it as a pan.)
                    let alt_pick =
                        e.modifiers().contains(Modifiers::ALT) && !tool.is_selection();
                    if alt_pick {
                        // Deliberately *not* `canvas_active`: the chrome fade
                        // exists to hand the screen back to the painting
                        // mid-stroke, but the Color panel is where a pick's answer
                        // shows up, so fading it out would hide the one thing this
                        // gesture is for.
                        picking.set(true);
                        if let Some(s) = sample(state, &e) {
                            pick_color(state, s.pos);
                        }
                        return;
                    }
                    canvas_active.set(true);
                    // From here the press is paint, and what it does with itself is
                    // the gesture's business rather than this handler's — including
                    // the case where there is no view to land in yet, which opens
                    // nothing and leaves the moves after it inert (`input::Paint`).
                    paint.begin(&e, tool);
                }
            },
            onpointermove: move |e| {
                // Navigation is asked first, and **unconditionally** — including
                // while a stroke is in flight. A lone finger's moves say nothing to
                // the view (`Nav::advance` answers false and the stroke below sees
                // them), but they still have to be *recorded*, because a second
                // finger landing pairs with where the first one has got to rather
                // than with where it pressed (§18.1.7).
                if nav.advance(&e) {
                    // The view moved, so nothing below applies: a sample taken here
                    // would be mapped through the view as it was *before* the move,
                    // and with two fingers down there is no single pointer to report
                    // as a cursor anyway.
                    return;
                }
                // The brush moved rather than the pointer's meaning on the canvas
                // (§18.1.9): nothing below applies, since this press was never
                // painting and a peer has no use for a cursor being used as a knob.
                if tune.advance(&e) {
                    return;
                }
                // A composing mode opened under the hand (`crate::modes`). Its
                // catcher covers the canvas, so no *new* press can reach here —
                // but this pointer was captured by the canvas before the catcher
                // existed, and a captured pointer's moves are delivered to the
                // element that took them whatever has been stacked over it since.
                // A pen drawing while the other hand reaches for Transform is
                // exactly that, and without this the stroke would go on feeding
                // the fitter underneath the widget.
                //
                // Cancelled rather than left to commit, for the same reason a
                // pinch cancels the stroke it interrupts: the canvas stopped
                // taking paint the moment the mode took it, so the gesture must
                // leave no mark.
                if modes::is_composing(state) {
                    paint.abandon();
                    // And the canvas is no longer what is in hand — the mode is.
                    // Unlike the pinch, which goes on using it, so `nav` sets
                    // this the other way. Left dimmed, the mode's own bar would
                    // be faded and taking no clicks (§11) until the pen lifted,
                    // which is the one control the artist now needs.
                    canvas_active.set(false);
                    return;
                }
                // The canvas takes pointer events from the first frame, while the
                // engine is still being built asynchronously — so there may be no
                // view to map through yet, and a move with nowhere to land simply
                // does nothing.
                if let Some(s) = sample(state, &e) {
                    if picking() {
                        // Alt+drag keeps sampling; `pick_color` drops a move that
                        // arrives while the last sample is still settling.
                        pick_color(state, s.pos);
                    } else {
                        // The paint gesture takes it if it has one in flight, and
                        // says so — a move with no gesture behind it is only a
                        // cursor to report below (`input::Paint::advance`).
                        paint.advance(&e);
                    }
                    // Where collaborators see this client's pointer
                    // (§17.4). Quiet: it changes nothing *this* client renders — the
                    // browser draws our own cursor — so repainting the canvas at
                    // pointer rate to show ourselves nothing would be pure waste.
                    // The presence pump reads it off the engine on its own cadence.
                    // Solo, not even that: with no session the value has no reader
                    // at all, so the engine borrow it costs is skipped entirely.
                    if state.collab.active() {
                        dispatch_quiet(state, PeerCommand::SetCursor(Some(s.pos)));
                    }
                }
            },
            onpointerleave: move |_| {
                if state.collab.active() {
                    dispatch_quiet(state, PeerCommand::SetCursor(None));
                }
            },
            // One finger of several lifting ends nothing — the rest are still
            // navigating, and tearing down here would end the gesture on whichever
            // finger the hand happened to raise first (§18.1.7).
            onpointerup: move |e| {
                if !nav.release(&e) {
                    end_interaction(state, paint, nav, tune);
                }
            },
            onpointercancel: move |e| {
                if !nav.release(&e) {
                    end_interaction(state, paint, nav, tune);
                }
            },
            onwheel: move |e| nav.wheel(e),
        }
    }
}

/// Collaborators' pointers, drawn in each peer's own color (§17.4).
///
/// DOM rather than a compositor pass, on purpose: a cursor is chrome, not artwork —
/// it must never reach an export, and a label beside it is a `<div>` the browser
/// already knows how to lay out. The positions are canvas-space, so they follow the
/// painting under pan and zoom exactly as the paint does.
#[component]
fn PeerCursors() -> Element {
    let state = use_context::<AppState>();
    let peers = (state.collab.peers)();
    if peers.is_empty() {
        return rsx! {};
    }
    // Read the view once per render rather than per peer, and `peek` rather than
    // `read`: this component is driven by `peers`, and subscribing to the renderer
    // as well would re-render it on every engine write — every stroke sample, every
    // pan — to redraw cursors that had not moved.
    let Some(view) = state.renderer.peek().as_ref().map(|r| r.view()) else {
        return rsx! {};
    };
    rsx! {
        div { class: "peer-cursors",
            for peer in peers {
                if let Some(canvas) = peer.cursor {
                    {
                        let p = view.canvas_to_screen(canvas);
                        rsx! {
                            div {
                                key: "{peer.actor.0}",
                                class: "peer-cursor",
                                style: "left:{p.x}px; top:{p.y}px; --peer:{peer.css_color()}",
                                div { class: "peer-cursor-dot" }
                                div { class: "peer-cursor-name", "{peer.name}" }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The brush-tuning drag's size indicator (§18.1.9): a ring at the radius being asked
/// for, with the radius it started from behind it.
///
/// DOM rather than a compositor pass, for [`PeerCursors`]'s reason — it is chrome, and
/// the one thing it must never do is reach an export. It is also pure layout: the
/// gesture converts to screen px on its way in (`state::BrushRing`), so this reads no
/// view and re-renders on nothing but the ring itself.
///
/// A circle, though the brush may be any shape (§6.6). Deliberately, for now: what the
/// drag sets is one number, and a ring is the honest picture of one number — an outline
/// of the actual tip would be a picture of the *shape*, which this gesture cannot
/// change, and would say the mark is that crisp when a soft brush's is not.
#[component]
fn BrushSizeRing() -> Element {
    let state = use_context::<AppState>();
    let Some(ring) = (state.brush_ring)() else {
        return rsx! {};
    };
    // Both circles are laid out the same way: a box of the diameter, pulled back onto
    // the centre. `left`/`top` rather than a transform, like the peer cursors.
    let circle = |class: &'static str, r: f32| {
        rsx! {
            div {
                class: class,
                style: "left:{ring.at.x - r}px; top:{ring.at.y - r}px; width:{2.0 * r}px; height:{2.0 * r}px",
            }
        }
    };
    rsx! {
        div { class: "brush-ring",
            // The old size first, so the one being asked for draws over it.
            {circle("brush-ring-circle was", ring.was)}
            {circle("brush-ring-circle", ring.now)}
        }
    }
}

/// The tow string while a smoothing brush draws (§6.11): a hairline from the
/// towed tip — where paint is landing — to the pointer, with a dot under the
/// hand. It **sags while slack and straightens as it tows**, so the state of
/// the mechanism is legible at a glance: the one thing that makes deliberate
/// lag feel like latency is being unexplained, and the sag is the explanation.
///
/// DOM (an SVG overlay) rather than a compositor pass, for [`PeerCursors`]'s
/// reason: it is chrome, and it must never reach an export. Pure layout, too —
/// the gesture converts to screen px on its way in (`input::refresh_tow`,
/// `state::TowUi`), so this reads no view and re-renders on nothing but the
/// string itself.
#[component]
fn TowStringOverlay() -> Element {
    let state = use_context::<AppState>();
    let Some(t) = (state.tow)() else {
        return rsx! {};
    };
    let slack = (t.rope - (t.target - t.tip).length()).max(0.0);
    // The sag is the slack made visible: a quadratic dip whose depth is a share
    // of the string the tow is not using — capped, so a fresh stroke's fully
    // slack string drapes rather than plunges. Screen-down whatever the canvas
    // rotation, because the string hangs from the hand, not from the painting.
    let sag = (slack * 0.4).min(t.rope * 0.2);
    let mid = (t.tip + t.target) * 0.5 + stark_model::Vec2::new(0.0, sag);
    let d = format!(
        "M {:.1} {:.1} Q {:.1} {:.1} {:.1} {:.1}",
        t.tip.x, t.tip.y, mid.x, mid.y, t.target.x, t.target.y
    );
    rsx! {
        svg { class: "tow-string",
            path { d }
            circle { cx: "{t.target.x}", cy: "{t.target.y}", r: "2.5" }
        }
    }
}

/// A vertical menu rail on the far left for uncommon or keyboard-driven commands
/// (§11). Built on the vendored `menubar` component; the dropdown flies
/// out to the right. Undo/Redo live here purely to advertise their Ctrl+Z / Ctrl+Y
/// shortcuts (the everyday way to invoke them); "New document…" opens a modal.
///
/// The rail ends in a ⚙ that opens [`SettingsModal`] directly rather than dropping
/// a menu: settings are a *destination*, not a list of commands to pick one from,
/// and the one thing that menu would ever contain is the dialog itself.
#[component]
fn CommandRail() -> Element {
    let state = use_context::<AppState>();
    let layout = use_context::<PanelLayout>();
    let mut show_new_doc = use_signal(|| false);
    let mut show_session = use_signal(|| false);
    let mut show_export = use_signal(|| false);
    let mut show_settings = use_signal(|| false);
    let mut show_timing = use_signal(|| false);
    let mut show_credits = use_signal(|| false);
    let live = (state.collab.phase)() == collab::CollabPhase::Shared;
    // Through a memo: all three move on a commit and the rail is a menu of
    // commands, so re-rendering it per pointer sample of a stroke would be to
    // re-diff two dropdowns to enable the Undo entry that was already enabled
    // (`state::use_obs`).
    let flags = use_obs(state, |o| (o.can_undo, o.can_redo, o.has_selection));
    let (can_undo, can_redo, has_selection) = flags().unwrap_or((false, false, false));
    let hidden = (layout.hidden)();
    let timeline_open = (state.timeline.open)();
    let rack_pinned = (state.slots.pinned)();

    rsx! {
        div { class: chrome_class(state, "command-rail"),
            Menubar {
                MenubarMenu { index: 0usize,
                    // The catch-all menu for infrequent commands.
                    MenubarTrigger { {icon_large(icons::MENU)} }
                    MenubarContent {
                        MenubarItem {
                            index: 2usize,
                            value: "new-document".to_string(),
                            on_select: move |_| show_new_doc.set(true),
                            span { class: "menu-item", {icon(icons::ADD)} "New document…" }
                        }
                        MenubarItem {
                            index: 3usize,
                            value: "open-document".to_string(),
                            on_select: move |_| files::open_document(state),
                            span { class: "menu-item", {icon(icons::OPEN_DOC)} "Open\u{2026}" }
                        }
                        MenubarItem {
                            index: 4usize,
                            value: "save-document".to_string(),
                            on_select: move |_| files::save_document(state),
                            span { class: "menu-item", {icon(icons::SAVE)} "Save" }
                        }
                        // Directly above Export, and named as its pair: one picture
                        // coming *in* and one going out, neither of which is the
                        // painting itself (`crate::files`, `crate::images`). The
                        // action it commits is a *placement* (§23) — what the
                        // document does with the picture — where the menu says where
                        // it came from, which is the question a menu answers.
                        MenubarItem {
                            index: 5usize,
                            value: "import-image".to_string(),
                            on_select: move |_| images::import_image(state),
                            span { class: "menu-item", {icon(icons::IMPORT_IMAGE)} "Import image\u{2026}" }
                            span { class: "menu-shortcut", "Ctrl+V" }
                        }
                        MenubarItem {
                            index: 6usize,
                            value: "export-image".to_string(),
                            on_select: move |_| show_export.set(true),
                            span { class: "menu-item", {icon(icons::EXPORT)} "Export image\u{2026}" }
                        }
                        MenubarItem {
                            index: 7usize,
                            value: "share".to_string(),
                            // Sharing starts on the click, not on a second button
                            // inside the dialog: the dialog exists to hand over the
                            // link. A no-op once the session is live.
                            on_select: move |_| {
                                collab::share(state);
                                show_session.set(true);
                            },
                            span { class: "menu-item",
                                {icon(icons::SHARE)}
                                if live { "Share \u{25CF}" } else { "Share…" }
                            }
                        }
                        MenubarItem {
                            index: 0usize,
                            value: "undo".to_string(),
                            disabled: !can_undo,
                            on_select: move |_| dispatch(state, DocCommand::Undo),
                            span { class: "menu-item", {icon(icons::UNDO)} "Undo" }
                            span { class: "menu-shortcut", "Ctrl+Z" }
                        }
                        MenubarItem {
                            index: 1usize,
                            value: "redo".to_string(),
                            disabled: !can_redo,
                            on_select: move |_| dispatch(state, DocCommand::Redo),
                            span { class: "menu-item", {icon(icons::REDO)} "Redo" }
                            span { class: "menu-shortcut", "Ctrl+Y" }
                        }
                        // The same two glyphs the selection bar's chips wear. One
                        // command reached two ways has to look like one command,
                        // and these are the only entries in the menu that have a
                        // second home elsewhere in the chrome.
                        MenubarItem {
                            index: 8usize,
                            value: "deselect".to_string(),
                            disabled: !has_selection,
                            on_select: move |_| {
                                dispatch(state, DocCommand::Select(SelectionOp::select_all()))
                            },
                            span { class: "menu-item", {icon(icons::SELECTION_NONE)} "Deselect" }
                            span { class: "menu-shortcut", "Ctrl+D" }
                        }
                        MenubarItem {
                            index: 9usize,
                            value: "invert-selection".to_string(),
                            disabled: !has_selection,
                            on_select: move |_| dispatch(state, DocCommand::InvertSelection),
                            span { class: "menu-item",
                                {icon(icons::SELECTION_INVERT)}
                                "Invert selection"
                            }
                            span { class: "menu-shortcut", "Ctrl+Shift+I" }
                        }
                        // A mode rather than a command, so it carries a check like
                        // the Panels menu's entries do — the menu says whether you
                        // are in it, not only how to get there
                        // (§18.2.4).
                        MenubarItem {
                            index: 10usize,
                            value: "timeline".to_string(),
                            on_select: move |_| {
                                panels::timeline::set_open(state, !timeline_open)
                            },
                            span { class: "menu-item", {icon(icons::TIMELINE)} "Timeline" }
                            span { class: "menu-check",
                                if timeline_open { {icon(icons::CHECK)} }
                            }
                        }
                        // The last two, and last for a reason: neither is a thing to
                        // *do* to the drawing. Timing Stats above Credits because it
                        // is the one somebody might open twice — and it is a dialog
                        // rather than a panel because a live frame-rate readout beside
                        // the canvas is a thing to watch instead of painting
                        // (§7.1, `crate::timings`).
                        MenubarItem {
                            index: 11usize,
                            value: "timing".to_string(),
                            on_select: move |_| show_timing.set(true),
                            span { class: "menu-item", {icon(icons::TIMING)} "Timing stats\u{2026}" }
                        }
                        MenubarItem {
                            index: 12usize,
                            value: "credits".to_string(),
                            on_select: move |_| show_credits.set(true),
                            span { class: "menu-item", {icon(icons::CREDITS)} "Credits\u{2026}" }
                        }
                    }
                }
                MenubarMenu { index: 1usize,
                    // Toggle which floating panels are shown. Each entry wears the
                    // panel's own mark, which is the same one its title bar wears —
                    // so the menu is a picture of the stack rather than a list of
                    // its names (`PanelId::glyph`).
                    MenubarTrigger { {icon_large(icons::PANELS)} }
                    MenubarContent {
                        for (i, id) in PanelId::ALL.into_iter().enumerate() {
                            MenubarItem {
                                index: i,
                                value: format!("panel-{id:?}"),
                                on_select: move |_| {
                                    let mut hidden = layout.hidden;
                                    let mut h = hidden.write();
                                    if !h.remove(&id) { h.insert(id); }
                                },
                                span { class: "menu-item", {icon(id.glyph())} "{id.title()}" }
                                span { class: "menu-check",
                                    if !hidden.contains(&id) { {icon(icons::CHECK)} }
                                }
                            }
                        }
                        // The quick-brush rack (§18.1.8), last and slightly apart:
                        // it is the one thing this menu shows and hides that is not
                        // a panel in the stack — it stands down the left, it has no
                        // title bar to close itself from, and while a number is held
                        // it appears whatever this entry says.
                        //
                        // It belongs here all the same, because this menu is the map
                        // of what is on screen, and because kept open the rack is
                        // *clickable*: it is the mouse-only way to a slot, which a
                        // hand with a pen and no keyboard has no other route to.
                        MenubarItem {
                            index: PanelId::ALL.len(),
                            value: "quick-brushes".to_string(),
                            on_select: move |_| {
                                let mut pin = state.slots.pinned;
                                pin.set(!rack_pinned);
                            },
                            span { class: "menu-item",
                                {icon(icons::QUICK_BRUSHES)}
                                "Quick brushes"
                            }
                            span { class: "menu-check",
                                if rack_pinned { {icon(icons::CHECK)} }
                            }
                        }
                    }
                }
                // This client's preferences. A plain button inside the rail rather
                // than a third `MenubarMenu`: it opens a dialog on the click, so
                // there is no dropdown for the menubar to manage.
                button {
                    class: "rail-button",
                    // The menubar's own triggers carry `role="menuitem"`; matching
                    // it keeps the rail a well-formed menubar rather than a
                    // menubar with a stray button in it.
                    role: "menuitem",
                    r#type: "button",
                    title: "Settings",
                    onclick: move |_| show_settings.set(true),
                    {icon_large(icons::SETTINGS)}
                }
            }
        }
        if show_new_doc() {
            NewDocumentModal { on_close: move |_| show_new_doc.set(false) }
        }
        if show_session() {
            collab::SessionModal { on_close: move |_| show_session.set(false) }
        }
        if show_export() {
            files::ExportModal { on_close: move |_| show_export.set(false) }
        }
        if show_settings() {
            SettingsModal { on_close: move |_| show_settings.set(false) }
        }
        if show_timing() {
            timings::TimingModal { on_close: move |_| show_timing.set(false) }
        }
        if show_credits() {
            CreditsModal { on_close: move |_| show_credits.set(false) }
        }
    }
}

/// Modal for starting a fresh document. Today it carries the color-space choice
/// (§6.7); it's a dialog so more document settings can join it later.
#[component]
fn NewDocumentModal(on_close: EventHandler<()>) -> Element {
    let state = use_context::<AppState>();
    // Off the projection, not off the renderer. Both facts are in `obs`, and
    // reading the renderer signal in a render body subscribes the dialog to every
    // engine write — so it re-rendered on every command for the whole time it was
    // open, to re-seed two `use_signal`s that are seeded once (U9, and `PeerCursors`
    // carries the same warning).
    let document = use_obs(state, |o| (o.color_space, o.surface));
    let (current, current_surface) = match document() {
        Some((space, surface)) => (space, Some(surface)),
        None => (ColorSpaceId::Oklab, None),
    };
    let choice = use_signal(|| current);

    // The ground is chosen by catalog *name*, not by id: an id is the hash of a
    // height map, so it is not knowable until that map has been fetched — and this
    // dialog runs before any of them have (§6.4, `crate::grounds`). The name is
    // resolved to an id at Create, once the bytes are in hand.
    let current_ground = grounds::resolved(state)
        .into_iter()
        .find(|(_, id)| *id == current_surface)
        .map(|(g, _)| g.name)
        .unwrap_or(grounds::DEFAULT_GROUND);
    let surf_choice = use_signal(|| current_ground);

    // One selectable color-space card; `selected` toggles the highlight.
    let card = |id: ColorSpaceId, title: &str, desc: &str| {
        let class = if choice() == id {
            "space-card selected"
        } else {
            "space-card"
        };
        rsx! {
            div {
                class,
                onclick: move |_| { let mut choice = choice; choice.set(id); },
                div { class: "space-card-title", "{title}" }
                div { class: "space-card-desc", "{desc}" }
            }
        }
    };

    // Same card, for the canvas ground choice — one row per catalog entry, so
    // adding a ground is still a file plus a row in `grounds::GROUNDS`.
    let scard = |g: &'static grounds::BuiltinGround| {
        let class = if surf_choice() == g.name {
            "space-card selected"
        } else {
            "space-card"
        };
        rsx! {
            div {
                class,
                onclick: move |_| { let mut c = surf_choice; c.set(g.name); },
                div { class: "space-card-title", "{g.name}" }
                div { class: "space-card-desc", "{g.blurb}" }
            }
        }
    };

    rsx! {
        // Dimmed backdrop; click outside the dialog to dismiss.
        div {
            class: "modal-backdrop",
            onclick: move |_| on_close.call(()),
            div {
                class: "modal-dialog",
                onclick: move |e| e.stop_propagation(),

                div { class: "modal-title", "New Document" }
                div { class: "modal-subtitle", "Starting a new document replaces the current canvas." }

                div { class: "modal-section-label", "COLOR SPACE" }
                {card(ColorSpaceId::Oklab, "Oklab", "Perceptual color with smooth, predictable blending. The standard choice for digital painting.")}
                // Offered only where the engine carries it. `ColorSpaceId::Mixbox` is
                // a variant in every build — the save format's enum indices cannot
                // depend on a feature (§8) — so the id below still compiles; what a
                // build without the `mixbox` feature lacks is the space behind it, and
                // `ColorSpaceId::available` is the same question this asks.
                {cfg!(feature = "mixbox").then(|| card(ColorSpaceId::Mixbox, "Mixbox", "Realistic pigment mixing (Mixbox): blue + yellow makes green, like real paint. For natural media."))}

                div { class: "modal-section-label", "SURFACE" }
                for g in grounds::GROUNDS.iter() {
                    {scard(g)}
                }

                div { class: "modal-actions",
                    button {
                        class: "btn btn-secondary",
                        onclick: move |_| on_close.call(()),
                        "Cancel"
                    }
                    button {
                        class: "btn btn-primary",
                        onclick: move |_| new_document(state, choice(), surf_choice(), on_close),
                        "Create"
                    }
                }
            }
        }
    }
}

/// Replace the document with a fresh one in the chosen color space, on the chosen
/// ground, then repaint. The ground's height map is fetched on first use (the large
/// bump maps stay out of the wasm binary — §6.6), so this runs async: `ground` is a
/// catalog name and the id `new_document` needs is the hash of the image behind it.
///
/// It owns closing the modal (`on_close`), calling it only once the work is done.
/// `spawn_forever`, not `spawn`: a plain spawn would tie the task to the
/// modal's scope, and the backdrop/Cancel still work during the fetch — a
/// dismissal would cancel it mid-flight *after* `collab::leave` already ran
/// (session gone, document never replaced). The task must outlive the modal;
/// calling `on_close` after it unmounted is harmless (the callback lives in
/// CommandRail's scope, which persists).
fn new_document(
    state: AppState,
    color: ColorSpaceId,
    ground: &'static str,
    on_close: EventHandler<()>,
) {
    // Replacing the document abandons any shared session (and clears the
    // ticket from the URL) — the fresh canvas is private until re-shared.
    collab::leave(state);
    spawn_forever(async move {
        // A ground that will not fetch opens the document smooth rather than
        // refusing to open it — and the document then honestly *says* it is smooth
        // instead of claiming a weave it hasn't got.
        let surface = grounds::resolve_signal(state, ground).await;
        state::with_engine(state, |r| {
            r.new_document(color, surface);
            r.paint();
        });
        tracing::info!(?color, ground, ?surface, "new document ready");
        on_close.call(());
    });
}

// --- reusable chrome ---
