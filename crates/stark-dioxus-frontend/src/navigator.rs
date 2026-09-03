//! The Navigator: a miniature of the whole piece in the bottom-left corner, the
//! viewport marked on it, and a click to go there (§11).
//!
//! # Why it is not a panel
//!
//! It was one, and the three things wrong with that are the three things this
//! module now is:
//!
//! - **It has no title to wear.** A picture of the piece is the one piece of
//!   chrome that says what it is by being looked at, so a bar naming it was a
//!   row of pixels spent restating the obvious — and a ✕ on a thing the Panels
//!   menu already toggles.
//! - **Its aspect is the artwork's**, not the stack's. A panel is a fixed-width
//!   column, so a portrait piece left two empty gutters and a landscape one a
//!   band under the picture; free of the column the box is exactly the miniature
//!   and the corner of the window keeps the difference.
//! - **It is read, not operated.** It earns its place by being glanceable, which
//!   is an argument for a corner rather than for a slot in the queue of things
//!   you reach for between strokes.
//!
//! So it is chrome of its own, like the quick-brush rack it shares a column with
//! (`crate::slots`): no background, no header, a shadow to lift it off the paint,
//! and the visibility menu to show and hide it — the only way to it, having no
//! title bar of its own to close from. And, like the rack and like the panel stack,
//! it is **remembered**: an artist who wants the overview up wants it up next time.
//! That is one record for all four now, and this module's share of it is
//! [`set_open`] (`crate::visibility`, §25.6).
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
//! The miniature is a second WebGPU surface bound to this component's own `<canvas>`,
//! and the engine renders the document straight into it
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
use crate::layout::chrome_dimmed;
use crate::panels::frame::piece_frame;
use crate::platform::{capture_pointer, sleep_ms};
use crate::state::{AppState, dispatch, use_obs, use_obs_opt};
use stark_engine::ExportScale;
use stark_engine::command::ViewCommand;
use stark_model::document::LayerId;
use stark_model::geom::{Extent2, Vec2};

/// The box the miniature is fitted into, in CSS px — the largest it is ever drawn,
/// on whichever axis the piece runs out of first.
///
/// A box rather than a width, and both numbers are caps in their own right: the
/// overlay shrink-wraps whatever comes back, so a landscape piece spends the width,
/// a portrait one spends the height, and neither pays for the axis it does not use.
/// That is what leaving the panel stack bought, and it is why this is nearly square
/// where the panel's box was a wide letterbox — a column's width was the constraint
/// there, and the corner of a window is not a column.
///
/// The size itself is a bargain with the painting: every pixel of it is canvas the
/// artist cannot see, and an overview too small to find the viewport marker in is
/// not worth the ones it does spend.
const MAX_WIDTH: u32 = 260;
const MAX_HEIGHT: u32 = 200;

/// How long a change has to stop arriving before the miniature is re-rendered.
/// Long enough to collapse a burst — a held undo, a peer's actions landing, the
/// several commits a Fill-then-recolor makes — short enough that a single stroke's
/// overview appears while the artist is still looking at where it landed.
const SETTLE_MS: i32 = 180;

/// Show the overview or put it away, and remember it — **the only thing that writes
/// [`Signals::navigator`](crate::state::Signals::navigator)(crate::state::Signals::navigator)**, which is what makes
/// durability structural rather than a line every call site has to remember (the move
/// `layout::set_open` makes for the panel stack, and `settings::SettingToggle` for the
/// preferences).
///
/// Guarded on the value actually moving, since the tour calls it for an overview that
/// is very often already up (§24.3) — and a `Signal` write dirties every subscriber
/// whether or not the value changed.
pub fn set_open(state: AppState, open: bool) {
    let mut showing = state.navigator;
    // Into a `bool` before the write: a read guard held across one is the shape that
    // has borrow-panicked in this crate before.
    let was = *showing.peek();
    if was == open {
        return;
    }
    showing.set(open);
    crate::visibility::persist(state);
}

/// Where the miniature sits in canvas space, and how large it is drawn.
///
/// All the overlay keeps: the picture itself lives on the GPU, in the surface bound to
/// its canvas. `Copy`, and four numbers wide, so the component that
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

/// Draw the miniature: one render of the committed document into the overlay's own
/// surface, scaled to fit [`MAX_WIDTH`] × [`MAX_HEIGHT`].
///
/// `None` before the engine exists, before the overlay's canvas has been attached to
/// it, or when the overview rect has no area to render — a frame dragged to nothing,
/// which [`export_plan`](stark_engine::Engine::export_plan) refuses; the overlay then
/// keeps showing whatever it last drew rather than blinking.
///
/// **One plan, asked for what is actually wanted.** Asking for a 1× plan first, purely
/// to learn the rect's size, and working the fitting scale out here puts a whole extra
/// question in the way of the answer — one with a stricter precondition than the render
/// it stands in for, since a 1× plan of a piece past the device's texture limit is
/// refused as a texture it could not allocate. Past that much painting or frame the
/// first call fails, `draw_overview` returns `None`, and the overlay silently goes on
/// showing a stale miniature at exactly the size where an overview earns its place.
/// [`ExportScale::Fit`] asks the engine the question the overlay has — "the largest that
/// fits this box" — and nothing about the size the picture *isn't* being rendered at
/// can refuse it.
///
/// Synchronous throughout: there is no readback, so nothing here awaits and nothing
/// has to survive an await.
fn draw_overview(state: AppState, frame: Option<LayerId>) -> Option<Overview> {
    // Quiet: a miniature is a second render of state this overlay is *reading*. It
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
fn viewport_style(over: Overview, view: stark_engine::ViewTransform) -> String {
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
fn turn_to(view: stark_engine::ViewTransform, v: Vec2, was: f32) -> Option<f32> {
    // The miniature is an upright, uniformly scaled picture of canvas space, so a
    // direction in its pixels *is* a direction in canvas px — no mapping needed, and
    // the pull length stays in the screen units the feel constants are written in.
    let target = snap_quarter(view.rotation_for_up(v)?);
    let ease = (v.length() / TURN_FOLLOW_PX).clamp(0.0, 1.0);
    Some(was + ease * shortest_turn(was, target))
}

/// The Navigator's miniature, down in the bottom-left corner (see the module docs).
#[component]
pub fn NavigatorOverlay() -> Element {
    let state = use_context::<AppState>();
    // Where the miniature currently sits in canvas space. Component-owned, and
    // meaningless once the overview is put away — the surface it describes goes with
    // it.
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
    let subject = use_obs_opt(state, |o| {
        let o = o?;
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

    // Where the marker goes — the other half of what this overlay draws, and the
    // half that moves at a different cadence from `subject` above: a pan changes
    // the view and not the document, a stroke changes both. Two memos rather than
    // one tuple for exactly that reason (`state::use_obs` asks for one where the
    // fields move together, and these do not). Declared here, above the early
    // returns, because hooks are positional.
    let live_view = use_obs(state, |o| o.view);

    use_effect(move || {
        // Subscribed to rather than peeked, which is what makes *showing* the
        // overview schedule its first refresh. Put away there is no canvas mounted
        // and so no surface to draw into, and a render would composite every tile in
        // the document into nothing at all.
        if !(state.navigator)() {
            return;
        }
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

    if !(state.navigator)() || subject().is_none() {
        return rsx! {};
    }

    // The live view: where the marker goes, and what a turn-drag measures from.
    // Through a memo like `subject` above, and for the identical reason it gives:
    // reading the projection straight here woke this overlay on every engine
    // write — every brush-tuning drag, every eyedropper sample — to redraw a
    // marker that only a pan or a zoom can move.
    let Some(view) = live_view() else {
        return rsx! {};
    };
    // The canvas is mounted whatever state the picture is in, because it *is* the
    // picture — there is nothing to show it with before it exists. Until the first
    // render lands the marker is simply absent.
    let placed = over().map(|o| viewport_style(o, view));

    // Where a press in the miniature points, in canvas space. The surface is
    // presented 1:1, so the element's own coordinates are the picture's.
    let target = move |e: &Event<PointerData>| {
        let o = over.peek().as_ref().copied()?;
        let f = elem_xy(e) / Vec2::new(o.width.max(1) as f32, o.height.max(1) as f32);
        Some(o.min + (o.max - o.min) * f)
    };

    rsx! {
        // Fades with the rest of the floating chrome while a canvas gesture is in
        // flight, exactly as the panel that used to hold it did: mid-stroke the
        // screen goes back to being the painting. Its own drag is deliberately not a
        // canvas gesture (see the press handler), so this never fades what is being
        // dragged.
        //
        // A wrapper around the frame rather than the frame itself, because the fade
        // has to out-specify the pointer-events the corner hands back — see
        // `.navigator-overlay` in the stylesheet, which is where that argument is.
        div {
            class: "navigator-overlay chrome",
            class: if chrome_dimmed(state) { "dimmed" },
            div {
                class: "nav-frame",
                // The mirror chord is printed from its own binding
                // (`Command::shortcut`), so this sentence cannot outlive a
                // rebind — and it is not said at all for a browser whose
                // rebinds left the mirror with no key to press.
                title: {
                    let mirror = crate::commands::Command::MirrorView
                        .shortcut(&state.bindings.read());
                    let mut title = "Click to go there, or drag to move the view around \
                                     the piece. Right-drag to turn the canvas: the \
                                     direction you drag becomes up."
                        .to_string();
                    if let Some(chord) = mirror {
                        title.push_str(&format!(" {chord} mirrors it."));
                    }
                    title
                },
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
                    // This canvas *is* the render target, so mounting it is what gives
                    // the engine somewhere to draw — and every remount needs a fresh
                    // surface, since the element the old one was bound to went with the
                    // overlay. Drawing in the same handler is what fills it before
                    // anyone sees it: the element is in the DOM by now, and nothing
                    // here measures layout.
                    onmounted: move |e: Event<MountedData>| {
                        if let Some(canvas) = crate::platform::canvas_of(&e) {
                            crate::state::with_engine_quiet(state, |r| r.attach_overview(canvas));
                        }
                        if let Some(next) = subject().and_then(|(_, f)| draw_overview(state, f)) {
                            over.set(Some(next));
                        }
                    },
                }
                if let Some(marker) = placed {
                    div { class: "nav-view", style: "{marker}" }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stark_model::geom::Extent2;

    /// A 400×200 piece under a miniature, and a viewport looking at it.
    fn piece() -> Overview {
        Overview {
            min: Vec2::new(-200.0, -100.0),
            max: Vec2::new(200.0, 100.0),
            width: 200,
            height: 100,
        }
    }

    /// One declaration out of the style string, as a number.
    fn css(style: &str, name: &str) -> f32 {
        style
            .split(';')
            .filter_map(|d| d.split_once(':'))
            .find(|(k, _)| k.trim() == name)
            .and_then(|(_, v)| v.trim().trim_end_matches('%').parse().ok())
            .unwrap_or_else(|| panic!("no {name} in {style}"))
    }

    /// The marker is placed by the view's **centre**, in percentages of the
    /// piece — so a view looking at the middle of the piece puts it in the
    /// middle of the miniature.
    ///
    /// Percentages rather than px because the miniature is laid out by the
    /// artwork's aspect, and pinning that here is what keeps the marker from
    /// needing to know how large the browser drew it.
    #[test]
    fn the_marker_sits_where_the_view_is_centred() {
        let view = stark_engine::ViewTransform::identity(Extent2::new(200, 100));
        let style = viewport_style(piece(), view);
        assert!((css(&style, "left") - 50.0).abs() < 1e-2, "{style}");
        assert!((css(&style, "top") - 50.0).abs() < 1e-2, "{style}");
        // 200 screen px at zoom 1 over a 400 px piece is half of it.
        assert!((css(&style, "width") - 50.0).abs() < 1e-2, "{style}");
        assert!((css(&style, "height") - 50.0).abs() < 1e-2, "{style}");
    }

    /// Zooming in shrinks the marker, because the marker is how much of the
    /// piece the window covers — the one thing an overview is for.
    #[test]
    fn zooming_in_shrinks_the_marker() {
        let mut view = stark_engine::ViewTransform::identity(Extent2::new(200, 100));
        let wide = css(&viewport_style(piece(), view), "width");
        view.zoom_about(Vec2::ZERO, 2.0);
        let close = css(&viewport_style(piece(), view), "width");
        assert!(close < wide, "{close} should be less than {wide}");
        assert!((close - wide * 0.5).abs() < 1e-2, "2x should halve it");
    }

    /// Panning off the piece slides the marker **out of the frame** rather than
    /// pinning it to an edge (§11). Pinned because clamping is the obvious-looking
    /// edit and it would have the overview claim you are still on the painting.
    #[test]
    fn panning_off_the_piece_takes_the_marker_with_it() {
        let mut view = stark_engine::ViewTransform::identity(Extent2::new(200, 100));
        view.center_on(Vec2::new(4_000.0, 0.0));
        let style = viewport_style(piece(), view);
        assert!(css(&style, "left") > 100.0, "{style}");
    }

    /// A degenerate overview — a frame dragged to nothing — divides by a floor
    /// rather than by zero, so the style is still a style and not a run of NaNs
    /// the browser silently drops.
    #[test]
    fn a_collapsed_piece_still_yields_numbers() {
        let flat = Overview {
            min: Vec2::ZERO,
            max: Vec2::ZERO,
            width: 200,
            height: 100,
        };
        let view = stark_engine::ViewTransform::identity(Extent2::new(200, 100));
        let style = viewport_style(flat, view);
        assert!(!style.contains("NaN"), "{style}");
        assert!(css(&style, "width").is_finite(), "{style}");
    }
}
