//! The Navigator panel: a miniature of the whole piece, the viewport marked on it,
//! and a click to go there (DESIGN.md §11).
//!
//! # What "the whole piece" means
//!
//! Exactly what an export would write (FRAME_DESIGN.md §6): the topmost frame's
//! rect, or — with no frame — the painted bounds, or on an empty canvas the
//! viewport. That is not a coincidence to be maintained but the *same call*:
//! [`Engine::export_plan`] answers the rect and [`Engine::export`] renders it, so
//! the miniature cannot come to disagree with the picture the file would hold. It
//! is also the only sensible answer on an unbounded canvas, which has no extent of
//! its own to show.
//!
//! [`Engine::export_plan`]: stark_core::Engine::export_plan
//! [`Engine::export`]: stark_core::Engine::export
//!
//! # Why it does not simply track the canvas
//!
//! One miniature is a GPU render plus a readback, and the render resizes the
//! compositor's viewport-sized offscreen targets to the thumbnail and back
//! (`gpu::composite`), so it costs a repaint of the real canvas too. That is
//! nothing on an edit and ruinous per pointer sample — a navigator that redrew with
//! the canvas would tax every stroke to show, in 250 px, what the canvas is already
//! showing full size.
//!
//! So the miniature is a picture of the **committed document**, refreshed when that
//! changes and not otherwise. `ObservableState::doc_revision` is the whole
//! subscription: it moves on a commit, an undo, a merged remote action or a load,
//! and deliberately not on the in-flight stroke or the unlogged drag preview. A
//! short settle delay then collapses a burst of edits — a held Ctrl+Z, a peer's
//! stream of arriving actions — into one render, and a render that would land
//! mid-gesture waits for the hand to lift rather than stealing frames from it.
//!
//! The viewport rectangle over the top is *not* rendered: it is a positioned
//! `<div>` read from the live view, so panning and zooming move it at no cost.

use dioxus::prelude::*;

use crate::input::elem_xy;
use crate::platform::{capture_pointer, draw_rgba, sleep_ms};
use crate::state::{AppState, dispatch};
use stark_core::command::ViewCommand;
use stark_core::geom::Vec2;
use stark_core::{Background, ExportScale, LayerId, ObservableState, Rendered};

/// The largest miniature, in CSS px. The width is the panel's inner width (see
/// `.panel-stack` / `.panel` in `stark.css`), so a landscape piece reaches both
/// edges; the height is a cap on how much of the panel stack one overview may
/// take, which a tall piece runs into instead.
const MAX_WIDTH: u32 = 252;
const MAX_HEIGHT: u32 = 176;

/// How long a change has to stop arriving before the miniature is re-rendered.
/// Long enough to collapse a burst — a held undo, a peer's actions landing, the
/// several commits a Fill-then-recolour makes — short enough that a single stroke's
/// overview appears while the artist is still looking at where it landed.
const SETTLE_MS: i32 = 180;

/// The DOM id of the miniature's canvas. One panel, so one id: the pixels arrive
/// from an async readback and are drawn imperatively (see
/// [`crate::platform::draw_rgba`]), which needs something to look the element up
/// by.
const THUMB_ID: &str = "stark-navigator-thumb";

/// A rendered miniature: the canvas-space rect it covers, and the pixels covering
/// it. The rect travels *with* the pixels rather than being read from the engine at
/// draw time, so the viewport marker cannot be placed against a crop the image on
/// screen was not rendered at.
///
/// Deliberately not `Clone` and not `PartialEq`: it is written once per refresh and
/// read by reference, and either impl would put an O(pixels) operation within easy
/// reach of a component that re-renders on every engine write.
struct Miniature {
    /// Canvas-space rect, as the export plan reported it.
    min: Vec2,
    max: Vec2,
    /// Image size in px, which is also its CSS size — the canvas is presented 1:1
    /// (like the painting canvas itself, which ignores `devicePixelRatio` too).
    width: u32,
    height: u32,
    /// Straight-alpha RGBA8, `width * height * 4`.
    pixels: Vec<u8>,
}

/// The frame the overview is taken against: the **topmost** matte layer, or `None`
/// when the document has none, which is what asks the engine for the painted
/// bounds instead (see the module docs).
///
/// Topmost rather than *selected*, unlike the export dialog: the export dialog is
/// framing one picture and the selected frame is the one being composed, while this
/// is a permanent readout of where you are in the piece — and "the piece" is what
/// the frame on top says it is.
///
/// Only the id is taken from the projection, which is why reading it from a
/// possibly-previewed layer list is safe: a frame handle drag moves a matte's rect,
/// never its identity.
fn overview_frame(o: &ObservableState) -> Option<LayerId> {
    o.layers
        .iter()
        .rev()
        .find(|l| l.matte.is_some())
        .map(|l| l.id)
}

/// Scale for a miniature of a `(width, height)` canvas rect: the largest that fits
/// the panel's box. Small pieces are scaled *up* as happily as large ones down —
/// the overview's job is to show the whole of it at a glance, and a 60 px sketch
/// shown at 60 px would say less than the empty panel around it.
fn fit_scale(width: u32, height: u32) -> f32 {
    let by_width = MAX_WIDTH as f32 / width.max(1) as f32;
    let by_height = MAX_HEIGHT as f32 / height.max(1) as f32;
    by_width.min(by_height)
}

/// Render the miniature: one export of the committed document, scaled to the
/// panel's box. `None` before the engine exists, or when the overview rect has no
/// area to render (a frame dragged to nothing — [`ExportPlan`] refuses it, and the
/// panel keeps whatever it last had rather than blinking).
///
/// [`ExportPlan`]: stark_core::ExportPlan
async fn render_miniature(state: AppState, frame: Option<LayerId>) -> Option<Miniature> {
    // Render, then **drop the guard before awaiting** — the readback future owns
    // everything it needs, and the UI re-renders (reading the renderer) while the
    // browser's event loop runs the copy. The same bargain `files::export_png` and
    // the eyedropper make.
    let (plan, readback) = {
        let mut renderer = state.renderer;
        let mut guard = renderer.write();
        let r = guard.as_mut()?;
        // The rect at 1:1 first, because the scale that fits the panel is a
        // property of the rect; the plan is then asked again *at that scale* rather
        // than scaled here, so the size the pixels come back at and the size the
        // canvas is given are one number from one source.
        let rect = r.export_plan(frame, ExportScale::Factor(1.0)).ok()?;
        let factor = ExportScale::Factor(fit_scale(rect.size.width, rect.size.height));
        let plan = r.export_plan(frame, factor).ok()?;
        let readback = r
            .export(frame, factor, Background::Substrate, Rendered::Committed)
            .ok()?;
        // Export renders through its own view into the compositor's offscreen
        // buffers, resizing them to the export size — so the on-screen surface has
        // to be repainted before anyone sees it again.
        r.paint();
        (plan, readback)
    };
    let image = readback.await;
    Some(Miniature {
        min: plan.min,
        max: plan.max,
        width: image.width,
        height: image.height,
        pixels: image.pixels,
    })
}

/// A CSS `left/top/width/height` for the part of `mini` the viewport covers, in
/// percentages of the miniature.
///
/// Not clamped: the rect is placed where it truly falls and the miniature's box
/// clips it, so panning off the piece shows the marker sliding out of the frame
/// rather than sticking to an edge and claiming you are still on the painting. What
/// the stylesheet contributes is a minimum size, so a viewport that is a fraction
/// of a percent of a large canvas is still something you can see.
fn viewport_style(mini: &Miniature, view: stark_core::ViewTransform) -> String {
    let half =
        Vec2::new(view.viewport.width as f32, view.viewport.height as f32) * (0.5 / view.zoom);
    let span = (mini.max - mini.min).max(Vec2::splat(1e-3));
    let lo = (view.center - half - mini.min) / span * 100.0;
    let size = (2.0 * half) / span * 100.0;
    format!(
        "left: {:.3}%; top: {:.3}%; width: {:.3}%; height: {:.3}%;",
        lo.x, lo.y, size.x, size.y
    )
}

/// The Navigator panel (see the module docs).
#[component]
pub fn NavigatorPanel() -> Element {
    let state = use_context::<AppState>();
    // The miniature on screen. Component-owned: it is worth nothing once the panel
    // is closed, and a fresh one is rendered on reopening.
    let mut mini = use_signal(|| None::<Miniature>);
    // Which refresh is the current one. A burst of edits arms several, and each
    // checks this after its settle delay so all but the last stand down — the
    // debounce, in one integer.
    let mut ticket = use_signal(|| 0u64);
    // Whether a press in the miniature is still held. Declared here, above every
    // early return, because hooks are positional.
    let mut dragging = use_signal(|| false);

    // What the miniature is a picture *of*: the committed document's revision and
    // the frame that crops it, or `None` when there is nothing to overview.
    //
    // A memo, so this notifies only when the answer changes — `obs` is rewritten on
    // every engine command, including every pointer sample of a stroke, and none of
    // that moves the committed document.
    let subject = use_memo(move || {
        let obs = state.obs.read();
        let o = obs.as_ref()?;
        let frame = overview_frame(o);
        // Nothing painted and no frame: the rect the engine would fall back to is
        // the *viewport* (FRAME_DESIGN.md §6), which for an overview would be a
        // picture of the window presented as the piece — and, since panning is not a
        // change to the document, one that then froze where it was rendered. An
        // unbounded canvas with nothing on it has no overview, and saying so is the
        // honest answer.
        let has_content = frame.is_some() || o.bounds.tile_range().is_some();
        has_content.then_some((o.doc_revision, frame))
    });

    use_effect(move || {
        let Some((_, frame)) = subject() else { return };
        let mine = *ticket.peek() + 1;
        ticket.set(mine);
        spawn(async move {
            // Wait out the burst, then wait out the gesture: a render that lands
            // mid-stroke would spend its cost exactly where it is least affordable.
            // `canvas_active` is the frontend's own "the canvas is in hand" flag, so
            // this covers strokes, marquees, pans and runs of wheel zoom alike.
            loop {
                sleep_ms(SETTLE_MS).await;
                if *ticket.peek() != mine {
                    return; // superseded by a later change
                }
                if !*state.canvas_active.peek() {
                    break;
                }
            }
            if let Some(next) = render_miniature(state, frame).await {
                mini.set(Some(next));
            }
        });
    });

    // Draw the pixels when they land. Keyed on `mini` alone, so an ordinary
    // re-render — the marker moving as the canvas pans — costs nothing, and the
    // canvas element keeps what was last drawn on it because nothing in the virtual
    // tree touches its size (see `platform::draw_rgba`).
    use_effect(move || {
        let mini = mini.read();
        if let Some(m) = mini.as_ref() {
            draw_rgba(THUMB_ID, m.width, m.height, &m.pixels);
        }
    });
    // The other half: draw when the *element* appears rather than when the pixels
    // do. A fresh `<canvas>` starts blank, and it can appear with a miniature
    // already in hand — undoing back to an empty document swaps in the empty state
    // and redoing swaps the canvas back, with no write to `mini` in between to wake
    // the effect above. Between the two, "there is a canvas and there are pixels"
    // always ends with the pixels on it.
    let draw_on_mount = move |_: Event<MountedData>| {
        let mini = mini.peek();
        if let Some(m) = mini.as_ref() {
            draw_rgba(THUMB_ID, m.width, m.height, &m.pixels);
        }
    };

    if subject().is_none() {
        return rsx! {
            div { class: "nav-empty", "Paint something, or add a frame" }
        };
    }

    // The marker and the miniature's size, without cloning the pixels: this
    // component re-renders on every engine write, and what it needs from a
    // ~150 KB image is six floats.
    let view = state.obs.read().as_ref().map(|o| o.view);
    let placed = mini
        .read()
        .as_ref()
        .zip(view)
        .map(|(m, v)| (viewport_style(m, v), m.width, m.height));
    let Some((marker, width, height)) = placed else {
        return rsx! {
            div { class: "nav-empty", "Rendering the overview\u{2026}" }
        };
    };

    // Where a press in the miniature points, in canvas space. The pixels are shown
    // 1:1, so the element's own coordinates are the image's.
    let target = move |e: &Event<PointerData>| {
        let f = elem_xy(e) / Vec2::new(width.max(1) as f32, height.max(1) as f32);
        mini.peek().as_ref().map(|m| m.min + (m.max - m.min) * f)
    };

    rsx! {
        div { class: "nav-thumb-row",
            div {
                class: "nav-frame",
                title: "Click to go there \u{2014} or drag to move the view around the piece",
                // Deliberately *not* `canvas_active`: the chrome fade exists to hand
                // the screen back to the painting mid-gesture, and fading this out
                // would take away the very thing being dragged.
                onpointerdown: move |e| {
                    capture_pointer(&e);
                    dragging.set(true);
                    if let Some(p) = target(&e) {
                        dispatch(state, ViewCommand::CenterOn(p));
                    }
                },
                // Held-and-dragged is one continuous "show me here", which is also
                // what makes the marker follow the pointer instead of jumping to it.
                onpointermove: move |e| {
                    if dragging()
                        && let Some(p) = target(&e)
                    {
                        dispatch(state, ViewCommand::CenterOn(p));
                    }
                },
                onpointerup: move |_| dragging.set(false),
                onpointercancel: move |_| dragging.set(false),

                canvas { id: "{THUMB_ID}", class: "nav-thumb", onmounted: draw_on_mount }
                div { class: "nav-view", style: "{marker}" }
            }
        }
    }
}
