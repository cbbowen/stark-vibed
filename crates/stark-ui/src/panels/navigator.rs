//! The Navigator panel: a miniature of the whole piece, the viewport marked on it,
//! and a click to go there (§11).
//!
//! # What "the whole piece" means
//!
//! Exactly what an export would write (§15.6): the topmost frame's
//! rect, or — with no frame — the painted bounds, or on an empty canvas nothing at
//! all. That is not a coincidence to be maintained but the *same call*:
//! [`Engine::export_plan`] answers the rect, and the plan it returns *is* the view the
//! miniature renders through ([`ExportPlan::view`]), so the overview cannot come to
//! disagree with the picture a file would hold.
//!
//! [`Engine::export_plan`]: stark_engine::Engine::export_plan
//! [`ExportPlan::view`]: stark_engine::ExportPlan::view
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

use dioxus::html::input_data::MouseButton;
use dioxus::prelude::*;

// The snap and the shortest way round are shared with the two-finger gesture
// (§18.1.7): "how square is square enough" has to mean one thing however the canvas
// is being turned.
use crate::input::{elem_xy, shortest_turn, snap_quarter};
use crate::panels::frame::piece_frame;
use crate::platform::{capture_pointer, sleep_ms};
use crate::state::{AppState, dispatch};
use stark_engine::ExportScale;
use stark_engine::command::ViewCommand;
use stark_model::document::LayerId;
use stark_model::geom::{Extent2, Vec2};

/// The largest miniature, in CSS px. The width is the panel's inner width (see
/// `.panel-stack` / `.panel` in `stark.css`), so a landscape piece reaches both
/// edges; the height is a cap on how much of the panel stack one overview may
/// take, which a tall piece runs into instead.
const MAX_WIDTH: u32 = 252;
const MAX_HEIGHT: u32 = 176;

/// How long a change has to stop arriving before the miniature is re-rendered.
/// Long enough to collapse a burst — a held undo, a peer's actions landing, the
/// several commits a Fill-then-recolor makes — short enough that a single stroke's
/// overview appears while the artist is still looking at where it landed.
const SETTLE_MS: i32 = 180;

/// Where the miniature sits in canvas space, and how large it is drawn.
///
/// All the panel keeps: the picture itself lives on the GPU, in the surface bound to
/// the panel's canvas. `Copy`, and four numbers wide, so the component that
/// re-renders on every engine write can read it freely — where a readback path would
/// keep a ~150 KB pixel buffer here and have to be careful never to clone it.
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

/// Draw the miniature: one render of the committed document into the panel's own
/// surface, scaled to fit the panel's box.
///
/// `None` before the engine exists, before the panel's canvas has been attached to
/// it, or when the overview rect has no area to render — a frame dragged to nothing,
/// which [`export_plan`](stark_engine::Engine::export_plan) refuses; the panel then
/// keeps showing whatever it last drew rather than blinking.
///
/// **One plan, asked for what is actually wanted.** Asking for a 1× plan first, purely
/// to learn the rect's size, and working the fitting scale out here puts a whole extra
/// question in the way of the answer — one with a stricter precondition than the render
/// it stands in for, since a 1× plan of a piece past the device's texture limit is
/// refused as a texture it could not allocate. Past that much painting or frame the
/// first call fails, `draw_overview` returns `None`, and the panel silently goes on
/// showing a stale miniature at exactly the size where an overview earns its place.
/// [`ExportScale::Fit`] asks the engine the question the panel has — "the largest that
/// fits this box" — and nothing about the size the picture *isn't* being rendered at
/// can refuse it.
///
/// Synchronous throughout: there is no readback, so nothing here awaits and nothing
/// has to survive an await.
fn draw_overview(state: AppState, frame: Option<LayerId>) -> Option<Overview> {
    // Quiet: a miniature is a second render of state this panel is *reading*. It
    // runs from a render and from the mount handler, either of which publishing
    // would be a component asking to be re-rendered while rendering.
    crate::state::with_engine_quiet(state, |r| {
        let fit = ExportScale::Fit(Extent2::new(MAX_WIDTH, MAX_HEIGHT));
        let plan = r.export_plan(frame, fit).ok()?;
        r.paint_overview(&plan).then_some(Overview {
            min: plan.min,
            max: plan.max,
            width: plan.size.width,
            height: plan.size.height,
        })
    })
    .flatten()
}

/// A CSS box for the part of `over` the viewport covers, in percentages of the
/// miniature, turned to match the view.
///
/// The miniature itself is always upright — it is a picture of the *piece*, and an
/// overview that turned with the easel would answer "where am I?" with a moving
/// frame of reference. So the turn shows in the marker instead: the viewport is a
/// screen-aligned rectangle, which in canvas space is a rectangle rotated the other
/// way, and handing CSS the inverse orientation draws exactly that. A mirrored view
/// mirrors the marker too, which for a rectangle is invisible — as it should be,
/// since the region really is the same region.
///
/// Not clamped: the rect is placed where it truly falls and the miniature's box
/// clips it, so panning off the piece shows the marker sliding out of the frame
/// rather than sticking to an edge and claiming you are still on the painting. What
/// the stylesheet contributes is a minimum size, so a viewport that is a fraction
/// of a percent of a large canvas is still something you can see.
fn viewport_style(over: Overview, view: stark_model::ViewTransform) -> String {
    let span = (over.max - over.min).max(Vec2::splat(1e-3));
    // The viewport's own size in canvas px — the rect *before* it is turned, so this
    // is the screen rectangle over the zoom rather than the bound it sweeps.
    let size = Vec2::new(view.viewport.width as f32, view.viewport.height as f32)
        / view.zoom.max(1e-6)
        / span
        * 100.0;
    let at = (view.center - over.min) / span * 100.0;
    let o = view.orientation().transpose();
    format!(
        "left: {:.3}%; top: {:.3}%; width: {:.3}%; height: {:.3}%; \
         transform: translate(-50%, -50%) matrix({}, {}, {}, {}, 0, 0);",
        at.x, at.y, size.x, size.y, o.x_axis.x, o.x_axis.y, o.y_axis.x, o.y_axis.y,
    )
}

/// What a press in the miniature started. The two buttons do different things, and
/// which one is held has to survive until the release.
#[derive(Clone, Copy, PartialEq)]
enum Drag {
    /// Left: the view follows the pointer around the piece.
    Center,
    /// Right: the drag is a vector, and it says which way is up. Carries where the
    /// press landed (miniature px) and the angle the canvas was at when it did —
    /// both, because the turn is measured *from* the press rather than accumulated
    /// move by move, so the same pointer position always means the same angle however
    /// it was arrived at.
    Turn { from: Vec2, was: f32 },
}

/// How far the turn-drag has to be pulled before the canvas follows the pointer
/// exactly, in miniature px.
///
/// Short of it the canvas eases toward where the drag points, in proportion to how
/// far it has been pulled — because near the press the *direction* of a two-pixel
/// vector is almost pure noise, and following it exactly makes the canvas snap to a
/// wild angle the instant the button goes down. Easing in means the first few pixels
/// barely move it, and by the time the pointer is a thumb's width out it is doing
/// exactly what it is told.
const TURN_FOLLOW_PX: f32 = 64.0;

/// The angle a turn-drag of `v` (miniature px, measured from the press) asks the
/// canvas to be at, having started the drag at `was`.
///
/// `None` when the drag has gone nowhere, which asks for nothing. The snap is applied
/// to the *target* rather than to the eased result, so a long pull lands exactly
/// square while a short one still eases smoothly toward that.
fn turn_to(view: stark_model::ViewTransform, v: Vec2, was: f32) -> Option<f32> {
    // The miniature is an upright, uniformly scaled picture of canvas space, so a
    // direction in its pixels *is* a direction in canvas px — no mapping needed, and
    // the pull length stays in the screen units the feel constants are written in.
    let target = snap_quarter(view.rotation_for_up(v)?);
    let ease = (v.length() / TURN_FOLLOW_PX).clamp(0.0, 1.0);
    Some(was + ease * shortest_turn(was, target))
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
    // The press in flight, if any. Declared here, above every early return, because
    // hooks are positional.
    let mut dragging = use_signal(|| None::<Drag>);

    // What the miniature is a picture *of*: the committed document's revision and
    // the frame that crops it, or `None` when there is nothing to overview.
    //
    // A memo, so this notifies only when the answer changes — `obs` is rewritten on
    // every engine command, including every pointer sample of a stroke, and none of
    // that moves the committed document.
    let subject = use_memo(move || {
        let obs = state.obs.read();
        let o = obs.as_ref()?;
        // The **topmost** frame rather than the *selected* one, unlike the export
        // dialog: that dialog is framing one picture and the selected frame is the
        // one being composed, while this is a permanent readout of where you are in
        // the piece — and "the piece" is what the frame on top says it is. Only the
        // id is taken, which is what makes reading it from a possibly-previewed
        // layer list safe: a handle drag moves a matte's rect, never its identity.
        let frame = piece_frame(o);
        // Nothing painted and no frame: the rect the engine would fall back to is
        // the *viewport* (§15.6), which for an overview would be a
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

    // The live view: where the marker goes, and what a turn-drag measures from.
    let Some(view) = state.obs.read().as_ref().map(|o| o.view) else {
        return rsx! {
            div { class: "nav-empty", "Rendering the overview\u{2026}" }
        };
    };
    // The canvas is mounted whatever state the picture is in, because it *is* the
    // picture — there is nothing to show it with before it exists. Until the first
    // render lands the marker is simply absent.
    let placed = over().map(|o| (viewport_style(o, view), o.width, o.height));

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
                title: "Click to go there, or drag to move the view around the piece. \
                        Right-drag to turn the canvas: the direction you drag becomes up. \
                        H mirrors it.",
                // Deliberately *not* `canvas_active`: the chrome fade exists to hand
                // the screen back to the painting mid-gesture, and fading this out
                // would take away the very thing being dragged.
                onpointerdown: move |e| {
                    capture_pointer(&e);
                    match e.trigger_button() {
                        // The right button turns the canvas. Nothing happens on the
                        // press itself: a turn is a *direction*, and one point does
                        // not have one — so the gesture says nothing until it has
                        // been dragged somewhere.
                        Some(MouseButton::Secondary) => dragging.set(Some(Drag::Turn {
                            from: elem_xy(&e),
                            was: view.rotation,
                        })),
                        Some(MouseButton::Primary) => {
                            dragging.set(Some(Drag::Center));
                            if let Some(p) = target(&e) {
                                dispatch(state, ViewCommand::CenterOn(p));
                            }
                        }
                        _ => {}
                    }
                },
                // Held-and-dragged is one continuous request in both cases — "show me
                // here", or "this way up" — which is what makes the view follow the
                // pointer instead of jumping to wherever it is let go.
                onpointermove: move |e| {
                    let Some(d) = dragging() else { return };
                    match d {
                        Drag::Center => {
                            if let Some(p) = target(&e) {
                                dispatch(state, ViewCommand::CenterOn(p));
                            }
                        }
                        Drag::Turn { from, was } => {
                            if let Some(to) = turn_to(view, elem_xy(&e) - from, was) {
                                dispatch(state, ViewCommand::SetRotation(to));
                            }
                        }
                    }
                },
                onpointerup: move |_| dragging.set(None),
                onpointercancel: move |_| dragging.set(None),
                // The right button is a tool here, so the browser's menu would be in
                // the way of it — and would arrive mid-drag, which is worse than
                // useless. Refused for the whole page now (`input::bind_context_menu`),
                // this being the surface that made the case first.

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
                            crate::state::with_engine_quiet(state, |r| r.attach_overview(canvas));
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
