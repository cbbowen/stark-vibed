//! The Navigator panel: a miniature of the whole piece, the viewport marked on it,
//! and a click to go there (DESIGN.md §11).
//!
//! # What "the whole piece" means
//!
//! Exactly what an export would write (FRAME_DESIGN.md §6): the topmost frame's
//! rect, or — with no frame — the painted bounds, or on an empty canvas nothing at
//! all. That is not a coincidence to be maintained but the *same call*:
//! [`Engine::export_plan`] answers the rect, and the plan it returns *is* the view the
//! miniature renders through ([`ExportPlan::view`]), so the overview cannot come to
//! disagree with the picture a file would hold.
//!
//! [`Engine::export_plan`]: stark_core::Engine::export_plan
//! [`ExportPlan::view`]: stark_core::ExportPlan::view
//!
//! # It is a surface, not an image
//!
//! The miniature is a second WebGPU surface bound to the panel's own `<canvas>`, and
//! the engine renders the document straight into it
//! ([`Renderer::paint_overview`](crate::render::Renderer::paint_overview)) — the same
//! arrangement as the painting canvas, one document seen twice. So this module holds
//! **no pixels**: a refresh is one render and a present, synchronously, and what the
//! component keeps is four numbers describing where the picture sits in canvas space.
//!
//! # Why it does not simply track the canvas
//!
//! One refresh composites every tile in the document. That is nothing on an edit and
//! ruinous per pointer sample — a navigator that redrew with the canvas would tax
//! every stroke to show, in 250 px, what the canvas is already showing full size.
//!
//! So the miniature is a picture of the **committed document**, refreshed when that
//! changes and not otherwise. `ObservableState::doc_revision` is the whole
//! subscription: it moves on a commit, an undo, a merged remote action or a load, and
//! deliberately not on the in-flight stroke or the unlogged drag preview. A short
//! settle delay then collapses a burst of edits — a held Ctrl+Z, a peer's stream of
//! arriving actions — into one render, and a render that would land mid-gesture waits
//! for the hand to lift rather than stealing frames from it.
//!
//! The viewport rectangle over the top is not rendered either: it is a positioned
//! `<div>` read from the live view, so panning and zooming move it at no cost.

use dioxus::prelude::*;

use crate::input::elem_xy;
use crate::platform::{capture_pointer, sleep_ms};
use crate::state::{AppState, dispatch};
use stark_core::command::ViewCommand;
use stark_core::geom::Vec2;
use stark_core::{ExportScale, LayerId, ObservableState};

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

/// Where the miniature sits in canvas space, and how large it is drawn.
///
/// All the panel keeps: the picture itself lives on the GPU, in the surface bound to
/// the panel's canvas. `Copy`, and four numbers wide, so the component that
/// re-renders on every engine write can read it freely — where the old readback path
/// kept a ~150 KB pixel buffer here and had to be careful never to clone it.
#[derive(Clone, Copy, PartialEq)]
struct Overview {
    /// The canvas-space rect the miniature covers.
    min: Vec2,
    max: Vec2,
    /// Its size in px, which is also its CSS size — the surface is presented 1:1,
    /// like the painting canvas, which ignores `devicePixelRatio` too.
    width: u32,
    height: u32,
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

/// Draw the miniature: one render of the committed document into the panel's own
/// surface, scaled to fit the panel's box.
///
/// `None` before the engine exists, before the panel's canvas has been attached to
/// it, or when the overview rect has no area to render — a frame dragged to nothing,
/// which [`export_plan`](stark_core::Engine::export_plan) refuses; the panel then
/// keeps showing whatever it last drew rather than blinking.
///
/// Synchronous throughout: there is no readback, so nothing here awaits and nothing
/// has to survive an await.
fn draw_overview(state: AppState, frame: Option<LayerId>) -> Option<Overview> {
    let mut renderer = state.renderer;
    let mut guard = renderer.write();
    let r = guard.as_mut()?;
    // The rect at 1:1 first, because the scale that fits the panel is a property of
    // the rect; the plan is then asked again *at that scale* rather than scaled here,
    // so the size the surface is configured to and the size the picture is rendered at
    // are one number from one source.
    let rect = r.export_plan(frame, ExportScale::Factor(1.0)).ok()?;
    let factor = ExportScale::Factor(fit_scale(rect.size.width, rect.size.height));
    let plan = r.export_plan(frame, factor).ok()?;
    r.paint_overview(&plan).then_some(Overview {
        min: plan.min,
        max: plan.max,
        width: plan.size.width,
        height: plan.size.height,
    })
}

/// A CSS `left/top/width/height` for the part of `over` the viewport covers, in
/// percentages of the miniature.
///
/// Not clamped: the rect is placed where it truly falls and the miniature's box
/// clips it, so panning off the piece shows the marker sliding out of the frame
/// rather than sticking to an edge and claiming you are still on the painting. What
/// the stylesheet contributes is a minimum size, so a viewport that is a fraction
/// of a percent of a large canvas is still something you can see.
fn viewport_style(over: Overview, view: stark_core::ViewTransform) -> String {
    let half =
        Vec2::new(view.viewport.width as f32, view.viewport.height as f32) * (0.5 / view.zoom);
    let span = (over.max - over.min).max(Vec2::splat(1e-3));
    let lo = (view.center - half - over.min) / span * 100.0;
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
    // Where the miniature currently sits in canvas space. Component-owned, and
    // meaningless once the panel closes — the surface it describes goes with it.
    let mut over = use_signal(|| None::<Overview>);
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
            if let Some(next) = draw_overview(state, frame) {
                over.set(Some(next));
            }
        });
    });

    if subject().is_none() {
        return rsx! {
            div { class: "nav-empty", "Paint something, or add a frame" }
        };
    }

    // The canvas is mounted whatever state the picture is in, because it *is* the
    // picture — there is nothing to show it with before it exists. Until the first
    // render lands the marker is simply absent.
    let placed = over()
        .zip(state.obs.read().as_ref().map(|o| o.view))
        .map(|(o, v)| (viewport_style(o, v), o.width, o.height));

    // Where a press in the miniature points, in canvas space. The surface is
    // presented 1:1, so the element's own coordinates are the picture's.
    let target = move |e: &Event<PointerData>| {
        let o = over.peek().as_ref().copied()?;
        let f = elem_xy(e) / Vec2::new(o.width.max(1) as f32, o.height.max(1) as f32);
        Some(o.min + (o.max - o.min) * f)
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

                canvas {
                    class: "nav-thumb",
                    // The panel's canvas *is* the render target, so mounting it is
                    // what gives the engine somewhere to draw — and every remount
                    // needs a fresh surface, since the element the old one was bound
                    // to went with the panel. Drawing in the same handler is what
                    // fills it before anyone sees it: the element is in the DOM by
                    // now, and nothing here measures layout.
                    onmounted: move |e: Event<MountedData>| {
                        if let Some(canvas) = crate::platform::canvas_of(&e) {
                            let mut renderer = state.renderer;
                            if let Some(r) = renderer.write().as_mut() {
                                r.attach_overview(canvas);
                            }
                        }
                        if let Some(next) = subject().and_then(|(_, f)| draw_overview(state, f)) {
                            over.set(Some(next));
                        }
                    },
                }
                if let Some((marker, _, _)) = placed {
                    div { class: "nav-view", style: "{marker}" }
                }
            }
        }
    }
}
