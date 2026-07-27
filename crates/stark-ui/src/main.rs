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
mod input;
mod layout;
mod panels;
mod platform;
mod render;
mod state;
mod widgets;

use std::collections::{HashMap, HashSet};

use dioxus::dioxus_core::{Task, spawn_forever};
use dioxus::html::input_data::MouseButton;
use dioxus::prelude::*;

use brush_editor::BrushEditorModal;
use components::menubar::{Menubar, MenubarContent, MenubarItem, MenubarMenu, MenubarTrigger};
use input::{elem_xy, end_interaction, handle_keydown, handle_keyup, sample};
use layout::{PanelId, PanelLayout, PanelStack, drag_end, drag_move};
use panels::brush::BRISTLE_BRUSH;
use panels::lighting::{ENV_FERNDALE, surface_asset};
use panels::SelectionBar;
use panels::select::{current_mode, current_tool, modifier_mode};
use platform::capture_pointer;
use render::{CANVAS_ID, Renderer};
use stark_core::command::{DocCommand, GestureCommand, ViewCommand};
use stark_core::document::{SelectionMode, SelectionOp};
use stark_core::geom::Vec2;
use stark_core::{ColorSpaceId, EnvironmentId, ObservableState, SurfaceId};
use state::{AppState, CollabState, dispatch, resize};

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
    let renderer = use_signal(|| None::<Renderer>);
    let obs = use_signal(|| None::<ObservableState>);
    let space_down = use_signal(|| false);
    let brush_editor_open = use_signal(|| false);
    let collab = CollabState {
        session: use_signal(|| None::<stark_net::CollabSession>),
        ticket: use_signal(|| None::<String>),
        phase: use_signal(collab::CollabPhase::default),
        error: use_signal(|| None::<String>),
        pump: use_signal(|| None::<Task>),
    };
    let state = AppState {
        renderer,
        obs,
        space_down,
        brush_editor_open,
        collab,
    };
    use_context_provider(|| state);

    // Floating-panel layout: order + which are open. Provided so the panel chrome and
    // the "Panels" menu can reorder/close/restore them.
    let panels = PanelLayout {
        order: use_signal(|| PanelId::ALL.to_vec()),
        hidden: use_signal(HashSet::new),
        drag: use_signal(|| None),
        refs: use_signal(HashMap::new),
    };
    use_context_provider(|| panels);

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
            // Fetch the studio HDR and light the canvas with it (DESIGN.md §6.3);
            // until then the procedural studio environment is used.
            if let Ok(bytes) = dioxus::asset_resolver::read_asset_bytes(ENV_FERNDALE).await {
                r.register_environment(EnvironmentId::Ferndale, bytes);
                r.set_environment(EnvironmentId::Ferndale);
            }
            r.paint();
            obs.set(Some(r.observe()));
            renderer.set(Some(r));

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
            tabindex: "0",
            autofocus: true,
            onkeydown: move |e| handle_keydown(state, &e),
            onkeyup: move |e| handle_keyup(state, &e),
            // A panel drag is driven here (events bubble up even over the canvas), so it
            // keeps tracking wherever the pointer goes. No-op unless a drag is active;
            // leaving the window commits it so it can't get stuck.
            onpointermove: move |e| drag_move(panels, &e),
            onpointerup: move |_| drag_end(panels),
            onpointerleave: move |_| drag_end(panels),

            Canvas {}

            // Left command rail: rarely-used document commands, tucked away.
            CommandRail {}

            // Floating tool panels, stacked top-right — order + visibility are data-driven.
            PanelStack {}

            // Bottom-centre: the whole-selection commands, present only while there is
            // a selection to act on — so it doubles as the "canvas is masked" indicator.
            SelectionBar {}

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
    let mut panning = use_signal(|| false);
    let mut last_position = use_signal(|| None::<Vec2>);
    // The panel's selection mode, stashed while a gesture's modifier keys override it
    // (DESIGN.md §6.8) and restored when the gesture ends.
    let mut mode_restore = use_signal(|| None::<SelectionMode>);

    rsx! {
        canvas {
            id: "{CANVAS_ID}",
            class: "paint-canvas",
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
                match e.trigger_button() {
                    Some(MouseButton::Primary) => {
                        capture_pointer(&e);
                        if (state.space_down)() {
                            panning.set(true);
                        } else {
                            // Painting and selecting are the same gesture from here —
                            // the tool decides what the engine builds (DESIGN.md §6.8).
                            let tool = current_tool(state);
                            if tool.is_selection()
                                && let Some(m) = modifier_mode(e.modifiers())
                            {
                                mode_restore.set(Some(current_mode(state)));
                                dispatch(state, ViewCommand::SetSelectionMode(m));
                            }
                            dispatch(state, GestureCommand::Start { tool, sample: sample(state, &e) });
                            drawing.set(true);
                        }
                    }
                    Some(MouseButton::Auxiliary) => {
                        e.prevent_default(); // suppress middle-click autoscroll
                        capture_pointer(&e);
                        panning.set(true);
                    }
                    _ => {}
                }
            },
            onpointermove: move |e| {
                if drawing() {
                    dispatch(state, GestureCommand::To { sample: sample(state, &e) });
                } else if panning() && let Some(l) = last_position() {
                    dispatch(state, ViewCommand::Pan { delta: elem_xy(&e) - l });
                }
                last_position.set(Some(elem_xy(&e)));
            },
            onpointerup: move |_| end_interaction(state, &mut drawing, &mut panning, &mut mode_restore),
            onpointercancel: move |_| {
                end_interaction(state, &mut drawing, &mut panning, &mut mode_restore);
                last_position.set(None);
            },
            onwheel: move |e| {
                e.prevent_default();
                let dy = e.delta().strip_units().y;
                if dy != 0.0 {
                    let factor = if dy < 0.0 { 1.15 } else { 1.0 / 1.15 };
                    let c = e.element_coordinates();
                    dispatch(state, ViewCommand::Zoom { anchor: Vec2::new(c.x as f32, c.y as f32), factor });
                }
            },
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
    let live = (state.collab.phase)() == collab::CollabPhase::Shared;
    let (can_undo, can_redo, has_selection) = state
        .obs
        .read()
        .as_ref()
        .map(|o| (o.can_undo, o.can_redo, o.has_selection))
        .unwrap_or((false, false, false));
    let hidden = (layout.hidden)();

    rsx! {
        div { class: "command-rail",
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
                            value: "shared-drawing".to_string(),
                            on_select: move |_| show_session.set(true),
                            span { if live { "Shared drawing \u{25CF}" } else { "Shared drawing…" } }
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
                            index: 4usize,
                            value: "deselect".to_string(),
                            disabled: !has_selection,
                            on_select: move |_| {
                                dispatch(state, DocCommand::Select(SelectionOp::select_all()))
                            },
                            span { "Deselect" }
                            span { class: "menu-shortcut", "Ctrl+D" }
                        }
                        MenubarItem {
                            index: 5usize,
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
