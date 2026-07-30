//! Stark's Dioxus **web** frontend (DESIGN.md §11).
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
mod collab;
mod components;
mod files;
mod identity;
mod input;
mod layout;
mod panels;
mod platform;
mod presets;
mod render;
mod shapes;
mod state;
mod widgets;

use std::collections::{HashMap, HashSet};

use dioxus::dioxus_core::spawn_forever;
use dioxus::html::Modifiers;
use dioxus::html::input_data::MouseButton;
use dioxus::prelude::*;

use brush_editor::BrushEditorModal;
use components::menubar::{Menubar, MenubarContent, MenubarItem, MenubarMenu, MenubarTrigger};
use input::{Nav, bind_shortcuts, end_interaction, input_tolerance, pick_color, sample};
use layout::{PanelId, PanelLayout, PanelStack, chrome_class, drag_end, drag_move};
use panels::brush::BRISTLE_BRUSH;
use panels::lighting::{DEFAULT_ENVIRONMENT, environment_asset, surface_asset};
use panels::select::{current_mode, current_tool, modifier_mode};
use panels::{FrameBar, FrameOverlay, PickBar, SelectionBar, TransformBar, TransformOverlay};
use platform::capture_pointer;
use render::CANVAS_ID;
use stark_core::command::{DocCommand, GestureCommand, PeerCommand, ViewCommand};
use stark_core::document::{DEFAULT_SURFACE, SelectionMode, SelectionOp};
use stark_core::{ColorSpaceId, SurfaceId};
use state::{AppState, dispatch, dispatch_quiet, resize, update_brush};

/// The UI's global stylesheet — panel chrome (shared CSS custom properties) plus
/// every component class referenced below. Linked once by [`app`] so the rsx!
/// blocks carry class names, not inline styles. Custom properties are global, so
/// the css_module menubar styles pick up `--panel-shadow` / `--panel-background` too.
static STARK_CSS: Asset = asset!("/assets/stark.css");

fn main() {
    #[cfg(target_arch = "wasm32")]
    {
        console_error_panic_hook::set_once();
        // Route `tracing` events (engine + UI) to the browser console.
        tracing_wasm::set_as_global_default();
    }
    dioxus::launch(app);
}

fn app() -> Element {
    // Root-owned, because the collaboration pumps and the renderer's async init are
    // detached tasks living in `ScopeId::ROOT` — see `state::root_signal`.
    let state = AppState::new();
    let (renderer, obs) = (state.renderer, state.obs);
    use_context_provider(|| state);

    // Floating-panel layout: order + which are open. Provided so the panel chrome and
    // the "Panels" menu can reorder/close/restore them. Panels in `CLOSED_BY_DEFAULT`
    // start hidden but keep their slot, so the menu reopens them where they belong.
    let panels = PanelLayout {
        order: use_signal(|| PanelId::ALL.to_vec()),
        hidden: use_signal(|| HashSet::from(PanelId::CLOSED_BY_DEFAULT)),
        drag: use_signal(|| None),
        refs: use_signal(HashMap::new),
    };
    use_context_provider(|| panels);

    // The keyboard shortcuts live on the window, not on the root element below, so
    // they answer whatever has focus — including `document.body`, where the browser
    // leaves it after a clicked button unmounts itself (see `platform::on_window_key`).
    use_hook(|| bind_shortcuts(state));

    // The shape library follows the browser, not the document — load it before
    // the renderer exists so the gallery is populated on first open. The brush
    // presets follow the browser the same way (seeded with the built-ins on a
    // browser that has never stored any).
    use_hook(|| shapes::load(state));
    use_hook(|| presets::load(state));

    use_hook(|| {
        let mut renderer = renderer;
        let mut obs = obs;
        spawn(async move {
            let mut r = render::init(render::canvas_element(CANVAS_ID)).await;
            // Fetch the built-in brush at runtime (kept out of the wasm binary)
            // and import it once, so the Bristles chip is ready (DESIGN.md §6.6).
            if let Ok(bytes) = dioxus::asset_resolver::read_asset_bytes(BRISTLE_BRUSH).await {
                r.load_bristle(&bytes);
            }
            // Fetch the default canvas surface's height map (DESIGN.md §6.4, §6.6).
            // The document already starts on it, so registering the bytes is all it
            // takes for the engine to swap the flat stand-in for the real weave —
            // no `SetSurface` action, which would put a bogus first step in the undo
            // history of every fresh document.
            if let Some(asset) = surface_asset(DEFAULT_SURFACE)
                && let Ok(bytes) = dioxus::asset_resolver::read_asset_bytes(asset).await
            {
                r.register_surface(DEFAULT_SURFACE, bytes);
            }
            // Fetch the default environment's HDR and light the canvas with it
            // (DESIGN.md §6.3); until it arrives the procedural neutral one is used,
            // and the Lighting panel can switch back to it at any time. A no-op while
            // the default *is* the procedural one, which has no bytes to fetch.
            if let Some(asset) = environment_asset(DEFAULT_ENVIRONMENT)
                && let Ok(bytes) = dioxus::asset_resolver::read_asset_bytes(asset).await
            {
                r.register_environment(DEFAULT_ENVIRONMENT, bytes);
                r.set_environment(DEFAULT_ENVIRONMENT);
            }
            r.paint();
            obs.set(Some(r.observe()));
            renderer.set(Some(r));

            // The brush this app start begins on: the library's first preset (an
            // empty library leaves the engine's default brush), and then the
            // colour the Color panel is already showing — the panel mounted
            // before the engine existed, so it seeded its picker from
            // `INITIAL_COLOR` alone, and pushing the same colour here is what
            // keeps the engine from painting black under a red marker. Both go
            // through `ViewCommand::SetBrush`, which is session state, so
            // neither leaves a step in the undo history. Once per app start, not
            // per document: a new document keeps the brush the user is holding.
            presets::apply_first(state);
            update_brush(state, |b| {
                b.color[..3].copy_from_slice(&panels::color::INITIAL_COLOR)
            });

            // A `#stark…` fragment in the page URL is a session invitation:
            // join it now that the engine is up (DESIGN.md §12.4).
            if let Some(ticket) = collab::url_ticket() {
                tracing::info!("joining shared session from URL fragment");
                collab::join(state, ticket);
            }
        });
    });

    rsx! {
        document::Stylesheet { href: STARK_CSS }

        div {
            class: "app-root",
            // A panel drag is driven here (events bubble up even over the canvas), so it
            // keeps tracking wherever the pointer goes. No-op unless a drag is active;
            // leaving the window commits it so it can't get stuck.
            onpointermove: move |e| drag_move(panels, &e),
            onpointerup: move |_| drag_end(panels),
            onpointerleave: move |_| drag_end(panels),

            Canvas {}

            // The frame's edges and handles, over the canvas but *under* all the
            // floating chrome. Mounted only while a frame is selected for composing
            // (FRAME_DESIGN.md §7); its interior passes pointer events through, so
            // painting inside the frame is unaffected.
            FrameOverlay {}

            // The transform gesture's box and handles, over the canvas while the
            // selected paint is being composed (TRANSFORM_DESIGN.md §6). Its
            // catcher blocks canvas painting for the mode's duration.
            TransformOverlay {}

            // Collaborators' pointers, over the canvas and under the chrome
            // (PEER_DESIGN.md §4). Empty and free when solo.
            PeerCursors {}

            // Left command rail: rarely-used document commands, tucked away.
            CommandRail {}

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
                // selection bar while one is composing (TRANSFORM_DESIGN.md §6).
                TransformBar {}
                // The frame's composition controls, present only while a frame is
                // selected for composing (FRAME_DESIGN.md §7).
                FrameBar {}
                // The eyedropper's options, present only while Alt arms it
                // (MISSING_FEATURES §0.2). Last in the column, so it comes up
                // nearest the canvas — it is the most transient of the three.
                PickBar {}
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
        }
    }
}

/// The full-window painting surface (a WebGPU canvas the engine draws into).
#[component]
fn Canvas() -> Element {
    let state = use_context::<AppState>();
    let mut drawing = use_signal(|| false);
    // The shared pan/zoom bindings (`input::Nav`) — the same instance the
    // transform overlay makes for itself, so navigation means one thing.
    let nav = Nav::use_nav(state);
    // Whether an Alt+drag is sampling colour off the canvas rather than painting on
    // it (MISSING_FEATURES §0.2). Shared rather than local, unlike the two above,
    // because the options bar is mounted on *armed but not dragging*.
    let mut picking = state.pick.dragging;
    // The panel's selection mode, stashed while a gesture's modifier keys override it
    // (DESIGN.md §6.8) and restored when the gesture ends.
    let mut mode_restore = use_signal(|| None::<SelectionMode>);
    // Set for as long as the canvas is the thing being used, which fades the floating
    // chrome out of the way. Pointer gestures clear it on release (`end_interaction`).
    let mut canvas_active = state.canvas_active;

    // The selected layer may be a frame, which takes no paint (FRAME_DESIGN.md §7).
    // Rather than block the gesture, say so in the cursor: the brush crosshair
    // becomes "not-allowed", so the canvas explains itself before the user draws a
    // stroke that would go nowhere. Panning still works, so the pan cursor wins
    // while space is held.
    let paintable = state.obs.read().as_ref().is_some_and(|o| {
        o.layers
            .iter()
            .any(|l| l.id == o.active_layer && l.is_paintable())
    });
    // Alt arms the eyedropper over the brush, and the cursor says so before it is
    // used — the only thing that makes a modifier binding discoverable. Not over a
    // selection tool, where alt already means "subtract from the selection"
    // (DESIGN.md §6.8), so the cursor promises the pick exactly where a press would
    // take one. It beats `no-paint`, because a layer that takes no paint can still
    // be sampled.
    let sampling =
        (state.pick.alt_down)() && !(state.space_down)() && !current_tool(state).is_selection();
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
                // Navigation first: middle-drag, or space + the primary button
                // (`input::Nav` — the one definition of the pan bindings, shared
                // with the transform overlay). Taking it here is also what keeps
                // space+Alt panning rather than sampling.
                if nav.start_pan(&e) {
                    canvas_active.set(true);
                    return;
                }
                if e.trigger_button() == Some(MouseButton::Primary) {
                    capture_pointer(&e);
                    // Painting and selecting are the same gesture from here — the
                    // tool decides what the engine builds (DESIGN.md §6.8).
                    let tool = current_tool(state);
                    // Alt+press samples the canvas instead of painting on it, and
                    // Alt+drag keeps sampling — the binding Clip Studio Paint and
                    // Rebelle both use, so a colour is picked up without putting
                    // the brush down (MISSING_FEATURES §0.2). Alt over a selection
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
                    // A press before WebGPU init has finished has no canvas
                    // space to land in, so it starts no gesture (and `drawing`
                    // stays false, which is what keeps the moves after it inert
                    // too).
                    if let Some(sample) = sample(state, &e)
                        && let Some(tolerance) = input_tolerance(state, &e)
                    {
                        if tool.is_selection()
                            && let Some(m) = modifier_mode(e.modifiers())
                        {
                            mode_restore.set(Some(current_mode(state)));
                            dispatch(state, ViewCommand::SetSelectionMode(m));
                        }
                        dispatch(state, GestureCommand::Start {
                            tool,
                            sample,
                            // What this device and this zoom level actually
                            // resolve to, which is what the fit prices against.
                            tolerance,
                        });
                        drawing.set(true);
                    }
                }
            },
            onpointermove: move |e| {
                // The canvas takes pointer events from the first frame, while the
                // engine is still being built asynchronously — so there may be no
                // view to map through yet, and a move with nowhere to land simply
                // does nothing.
                if let Some(s) = sample(state, &e) {
                    if picking() {
                        // Alt+drag keeps sampling; `pick_color` drops a move that
                        // arrives while the last sample is still settling.
                        pick_color(state, s.pos);
                    } else if drawing() {
                        dispatch(state, GestureCommand::To { sample: s });
                    } else {
                        nav.pan_move(&e);
                    }
                    // Where collaborators see this client's pointer (PEER_DESIGN.md
                    // §4). Quiet: it changes nothing *this* client renders — the
                    // browser draws our own cursor — so repainting the canvas at
                    // pointer rate to show ourselves nothing would be pure waste.
                    // The presence pump reads it off the engine on its own cadence.
                    dispatch_quiet(state, PeerCommand::SetCursor(Some(s.pos)));
                }
            },
            onpointerleave: move |_| dispatch_quiet(state, PeerCommand::SetCursor(None)),
            onpointerup: move |_| end_interaction(state, &mut drawing, nav, &mut mode_restore),
            onpointercancel: move |_| end_interaction(state, &mut drawing, nav, &mut mode_restore),
            onwheel: move |e| nav.wheel(e),
        }
    }
}

/// Collaborators' pointers, drawn in each peer's own colour (PEER_DESIGN.md §4).
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

/// A vertical menu rail on the far left for uncommon or keyboard-driven commands
/// (DESIGN.md §11). Built on the vendored `menubar` component; the dropdown flies
/// out to the right. Undo/Redo live here purely to advertise their Ctrl+Z / Ctrl+Y
/// shortcuts (the everyday way to invoke them); "New document…" opens a modal.
#[component]
fn CommandRail() -> Element {
    let state = use_context::<AppState>();
    let layout = use_context::<PanelLayout>();
    let mut show_new_doc = use_signal(|| false);
    let mut show_session = use_signal(|| false);
    let mut show_export = use_signal(|| false);
    let live = (state.collab.phase)() == collab::CollabPhase::Shared;
    let (can_undo, can_redo, has_selection) = state
        .obs
        .read()
        .as_ref()
        .map(|o| (o.can_undo, o.can_redo, o.has_selection))
        .unwrap_or((false, false, false));
    let hidden = (layout.hidden)();

    rsx! {
        div { class: chrome_class(state, "command-rail"),
            Menubar {
                MenubarMenu { index: 0usize,
                    // ☰ — the catch-all menu for infrequent commands.
                    MenubarTrigger { "\u{2630}" }
                    MenubarContent {
                        MenubarItem {
                            index: 2usize,
                            value: "new-document".to_string(),
                            on_select: move |_| show_new_doc.set(true),
                            span { "New document…" }
                        }
                        MenubarItem {
                            index: 3usize,
                            value: "open-document".to_string(),
                            on_select: move |_| files::open_document(state),
                            span { "Open\u{2026}" }
                        }
                        MenubarItem {
                            index: 4usize,
                            value: "save-document".to_string(),
                            on_select: move |_| files::save_document(state),
                            span { "Save" }
                        }
                        MenubarItem {
                            index: 5usize,
                            value: "export-image".to_string(),
                            on_select: move |_| show_export.set(true),
                            span { "Export image\u{2026}" }
                        }
                        MenubarItem {
                            index: 6usize,
                            value: "share".to_string(),
                            // Sharing starts on the click, not on a second button
                            // inside the dialog: the dialog exists to hand over the
                            // link. A no-op once the session is live.
                            on_select: move |_| {
                                collab::share(state);
                                show_session.set(true);
                            },
                            span { if live { "Share \u{25CF}" } else { "Share…" } }
                        }
                        MenubarItem {
                            index: 0usize,
                            value: "undo".to_string(),
                            disabled: !can_undo,
                            on_select: move |_| dispatch(state, DocCommand::Undo),
                            span { "Undo" }
                            span { class: "menu-shortcut", "Ctrl+Z" }
                        }
                        MenubarItem {
                            index: 1usize,
                            value: "redo".to_string(),
                            disabled: !can_redo,
                            on_select: move |_| dispatch(state, DocCommand::Redo),
                            span { "Redo" }
                            span { class: "menu-shortcut", "Ctrl+Y" }
                        }
                        MenubarItem {
                            index: 7usize,
                            value: "deselect".to_string(),
                            disabled: !has_selection,
                            on_select: move |_| {
                                dispatch(state, DocCommand::Select(SelectionOp::select_all()))
                            },
                            span { "Deselect" }
                            span { class: "menu-shortcut", "Ctrl+D" }
                        }
                        MenubarItem {
                            index: 8usize,
                            value: "invert-selection".to_string(),
                            disabled: !has_selection,
                            on_select: move |_| dispatch(state, DocCommand::InvertSelection),
                            span { "Invert selection" }
                            span { class: "menu-shortcut", "Ctrl+Shift+I" }
                        }
                    }
                }
                MenubarMenu { index: 1usize,
                    // ▤ — toggle which floating panels are shown.
                    MenubarTrigger { "\u{25A4}" }
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
                                span { "{id.title()}" }
                                span { class: "menu-check",
                                    if hidden.contains(&id) { "" } else { "\u{2713}" }
                                }
                            }
                        }
                    }
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
    }
}

/// Modal for starting a fresh document. Today it carries the color-space choice
/// (DESIGN.md §6.7); it's a dialog so more document settings can join it later.
#[component]
fn NewDocumentModal(on_close: EventHandler<()>) -> Element {
    let state = use_context::<AppState>();
    let current = state
        .renderer
        .read()
        .as_ref()
        .map(|r| r.color_space())
        .unwrap_or(ColorSpaceId::Oklab);
    let choice = use_signal(|| current);

    let current_surface = state
        .renderer
        .read()
        .as_ref()
        .map(|r| r.surface())
        .unwrap_or_default();
    let surf_choice = use_signal(|| current_surface);

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

    // Same card, for the canvas surface choice.
    let scard = |id: SurfaceId, title: &str, desc: &str| {
        let class = if surf_choice() == id {
            "space-card selected"
        } else {
            "space-card"
        };
        rsx! {
            div {
                class,
                onclick: move |_| { let mut c = surf_choice; c.set(id); },
                div { class: "space-card-title", "{title}" }
                div { class: "space-card-desc", "{desc}" }
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
                {card(ColorSpaceId::Mixbox, "Mixbox", "Realistic pigment mixing (Mixbox): blue + yellow makes green, like real paint. For natural media.")}

                div { class: "modal-section-label", "SURFACE" }
                {scard(SurfaceId::Flat, "Smooth", "A perfectly smooth surface — paint lies flat, no canvas texture.")}
                {scard(SurfaceId::Linen, "Canvas", "Linen weave: the canvas texture catches the light.")}

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

/// Replace the document with a fresh one in the chosen color space and surface,
/// then repaint. Image-backed surfaces are fetched on first use (the large bump
/// maps stay out of the wasm binary — DESIGN.md §6.6), so this runs async.
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
    surface: SurfaceId,
    on_close: EventHandler<()>,
) {
    let mut renderer = state.renderer;
    let mut obs = state.obs;
    // Replacing the document abandons any shared session (and clears the
    // ticket from the URL) — the fresh canvas is private until re-shared.
    collab::leave(state);
    spawn_forever(async move {
        // Fetch + register the surface bytes the first time it's chosen
        // (procedural surfaces have no asset — see `surface_asset`).
        let needs_bytes = renderer
            .read()
            .as_ref()
            .is_some_and(|r| !r.surface_loaded(surface));
        if needs_bytes && let Some(asset) = surface_asset(surface) {
            tracing::info!(?surface, url = %asset, "fetching surface asset");
            match dioxus::asset_resolver::read_asset_bytes(asset).await {
                Ok(bytes) => {
                    tracing::info!(
                        ?surface,
                        bytes = bytes.len(),
                        "surface fetched; registering"
                    );
                    if let Some(r) = renderer.write().as_mut() {
                        r.register_surface(surface, bytes);
                    }
                }
                Err(e) => {
                    tracing::warn!("surface fetch failed: {e}");
                    on_close.call(());
                    return;
                }
            }
        }

        if let Some(r) = renderer.write().as_mut() {
            r.new_document(color, surface);
            r.paint();
            obs.set(Some(r.observe()));
        }
        tracing::info!(?color, ?surface, "new document ready");
        on_close.call(());
    });
}

// --- reusable chrome ---
