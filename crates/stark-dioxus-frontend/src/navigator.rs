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

use crate::input::elem_xy;
use crate::layout::chrome_dimmed;
use crate::panels::frame::piece_frame;
use crate::platform::{capture_pointer, now_seconds, sleep_ms};
use crate::state::{AppState, dispatch, use_obs, use_obs_opt};
use stark_engine::ExportScale;
use stark_engine::Extent2;
use stark_engine::command::ViewCommand;
use stark_model::document::LayerId;
use stark_model::geom::Vec2;
use stark_ui::bounds::{Overview, Refresh};

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

/// [`stark_ui::bounds::SETTLE`] in the milliseconds this frontend's timer takes.
/// How long the wait is is the overview's business; what a `sleep` is denominated in
/// is this frontend's.
const SETTLE_MS: i32 = (stark_ui::bounds::SETTLE * 1000.0) as i32;

/// Show the overview or put it away, and remember it — **the only thing that writes
/// [`Signals::navigator`](crate::state::Signals::navigator)(crate::state::Signals::navigator)**, which is what makes
/// durability structural rather than a line every call site has to remember (the move
/// `layout::set_open` makes for the panel stack, and `prefs::set` for the
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
        // Scale 1.0: this surface is presented 1:1, like the painting canvas, which
        // ignores `devicePixelRatio` too.
        r.paint_overview(&plan).then(|| Overview::of(&plan, 1.0))
    })
    .flatten()
}

/// A CSS box for the part of `over` the viewport covers, in the miniature's px, turned
/// to match the view.
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
    // Off the shared map (`stark_ui::bounds::marker`) rather than an inverse worked
    // out here: the zoom, the turn and the mirror are all in it, and a second
    // spelling would put the marker somewhere the pointer does not agree with. What
    // stays this frontend's is the *encoding* — CSS wants a placed box and a matrix
    // where the native overlay wants four points.
    //
    // Into px before anything is measured: on a miniature that is not square a fraction
    // of its width and one of its height are different lengths, so a length taken
    // across both is wrong and, at an angle, the two axes shear. And px out, because a
    // CSS `%` width is of the frame's width whichever way the box is turned.
    let px = Vec2::new(over.width, over.height);
    let c = stark_ui::bounds::marker(over, view);
    let pt = |i: usize| Vec2::new(c[i].0, c[i].1) * px;
    let (tl, tr, bl) = (pt(0), pt(1), pt(3));
    // The half-axes of the rectangle those corners describe. Their lengths are the
    // unturned box; their directions are the turn, which is exactly the split
    // `width`/`height` and `matrix()` want.
    let x = (tr - tl) * 0.5;
    let y = (bl - tl) * 0.5;
    let at = tl + x + y;
    let size = Vec2::new(x.length(), y.length()) * 2.0;
    // A degenerate axis has no direction to state; the identity is the honest
    // fallback, and the box it turns is zero-sized anyway.
    let unit = |v: Vec2, fallback: Vec2| {
        let n = v.length();
        if n > 1e-9 { v / n } else { fallback }
    };
    let (ux, uy) = (unit(x, Vec2::X), unit(y, Vec2::Y));
    format!(
        "left: {:.3}px; top: {:.3}px; width: {:.3}px; height: {:.3}px; \
         transform: translate(-50%, -50%) matrix({}, {}, {}, {}, 0, 0);",
        at.x, at.y, size.x, size.y, ux.x, ux.y, uy.x, uy.y,
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
    // What the surface last drew, and when — the refresh policy the native navigator
    // asks too (`stark_ui::bounds::Refresh`).
    let mut refresh = use_signal(Refresh::default);
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
        let Some((revision, frame)) = subject() else {
            return;
        };
        let mine = *ticket.peek() + 1;
        ticket.set(mine);
        spawn(async move {
            // Wait out the burst, then ask the policy — which waits out a gesture:
            // `canvas_active` is the frontend's own "the canvas is in hand" flag, so
            // that covers strokes, marquees, pans and runs of wheel zoom alike.
            loop {
                sleep_ms(SETTLE_MS).await;
                if *ticket.peek() != mine {
                    return; // superseded by a later change
                }
                let drawn = *refresh.peek();
                if drawn.due(revision, now_seconds(), *state.canvas_active.peek()) {
                    break;
                }
                if !drawn.stale(revision) {
                    return; // already a picture of this revision
                }
            }
            if let Some(next) = draw_overview(state, frame) {
                over.set(Some(next));
                refresh.write().record(revision, now_seconds());
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
    // presented 1:1, so the element's own coordinates are the picture's — and a
    // drag that leaves the box is held to the piece's edge, as it is natively
    // (`Overview::target`).
    let target = move |e: &Event<PointerData>| {
        let o = over.peek().as_ref().copied()?;
        let f = elem_xy(e) / Vec2::new(o.width.max(1.0), o.height.max(1.0));
        Some(o.target(f.x, f.y))
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
                    let mirror = stark_ui::commands::Command::MirrorView
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
                            if let Some(to) = stark_ui::nav::turn_to(view, elem_xy(&e) - from, was) {
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
                        // A fresh surface holds no picture, whatever the last one held.
                        refresh.set(Refresh::default());
                        if let Some((revision, frame)) = subject()
                            && let Some(next) = draw_overview(state, frame)
                        {
                            over.set(Some(next));
                            refresh.write().record(revision, now_seconds());
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
    use stark_engine::{Extent2, ViewTransform};
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};

    /// A 400×200 piece under a 200×100 miniature.
    fn piece() -> Overview {
        Overview {
            min: Vec2::new(-200.0, -100.0),
            max: Vec2::new(200.0, 100.0),
            width: 200.0,
            height: 100.0,
        }
    }

    /// A 200×100 viewport turned to `angle`.
    fn turned(angle: f32) -> ViewTransform {
        let mut view = ViewTransform::identity(Extent2::new(200, 100));
        view.set_rotation(angle);
        view
    }

    /// One declaration out of the style string, which must be in px.
    fn css(style: &str, name: &str) -> f32 {
        style
            .split(';')
            .filter_map(|d| d.split_once(':'))
            .find(|(k, _)| k.trim() == name)
            .and_then(|(_, v)| v.trim().strip_suffix("px")?.parse().ok())
            .unwrap_or_else(|| panic!("no {name} in px in {style}"))
    }

    /// The four numbers of the style's `matrix()`.
    fn matrix(style: &str) -> [f32; 4] {
        let inner = style
            .split_once("matrix(")
            .and_then(|(_, rest)| rest.split_once(')'))
            .map(|(inner, _)| inner)
            .unwrap_or_else(|| panic!("no matrix in {style}"));
        let n: Vec<f32> = inner
            .split(',')
            .map(|v| {
                v.trim()
                    .parse()
                    .unwrap_or_else(|_| panic!("{v:?} in {style}"))
            })
            .collect();
        [n[0], n[1], n[2], n[3]]
    }

    /// The corners the style draws, in miniature px and in `marker`'s order: the box's
    /// own corners through its `matrix()`, about the centre `translate(-50%, -50%)`
    /// puts at `left`/`top`.
    fn drawn(style: &str) -> [Vec2; 4] {
        let at = Vec2::new(css(style, "left"), css(style, "top"));
        let half = Vec2::new(css(style, "width"), css(style, "height")) * 0.5;
        let [a, b, c, d] = matrix(style);
        let (ux, uy) = (Vec2::new(a, b), Vec2::new(c, d));
        [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
            .map(|(sx, sy)| at + ux * (sx * half.x) + uy * (sy * half.y))
    }

    /// `stark_ui::bounds::marker`'s corners, in miniature px.
    fn marked(over: Overview, view: ViewTransform) -> [Vec2; 4] {
        stark_ui::bounds::marker(over, view)
            .map(|(fx, fy)| Vec2::new(fx * over.width, fy * over.height))
    }

    #[track_caller]
    fn assert_draws(style: &str, want: [Vec2; 4]) {
        for (got, want) in drawn(style).into_iter().zip(want) {
            assert!(got.distance(want) < 1e-2, "drew {got} for {want}: {style}");
        }
    }

    /// Where the corners fall is `marker`'s; the CSS box is this frontend's. A quarter
    /// turn on a miniature that is not square is the case a box measured in fractions of
    /// it got wrong: the screen's 200 px width runs down all 100 px of the miniature's
    /// height, and its 100 px height across a quarter of the width, 50 px.
    #[test]
    fn a_quarter_turn_draws_the_markers_box() {
        let view = turned(FRAC_PI_2);
        let style = viewport_style(piece(), view);
        let want = marked(piece(), view);
        assert_draws(&style, want);
        let (lo, hi) = stark_ui::bounds::aabb(drawn(&style)).expect("four corners");
        assert!((hi - lo).distance(Vec2::new(50.0, 100.0)) < 1e-2, "{style}");
        // Signed, so a turn is told from its inverse: the box's x axis runs along
        // `tr - tl`, and its y axis a quarter turn on from that.
        let [_, b, c, _] = matrix(&style);
        let edge = want[1] - want[0];
        assert_eq!(b.signum(), edge.y.signum(), "{style}");
        assert_eq!(c.signum(), -edge.y.signum(), "{style}");
    }

    /// At an angle, fractions of a miniature that is not square shear the box. In px its
    /// edges are perpendicular, and its corners are the marker's.
    #[test]
    fn an_angled_box_is_not_skewed() {
        let view = turned(FRAC_PI_4);
        let style = viewport_style(piece(), view);
        let [a, b, c, d] = matrix(&style);
        assert!((a * c + b * d).abs() < 1e-4, "{style}");
        assert_draws(&style, marked(piece(), view));
    }

    /// A degenerate marker — a frame dragged to nothing, a viewport with no width —
    /// is still a style: numbers and an honest matrix, never a run of NaNs the
    /// browser silently drops.
    #[test]
    fn a_degenerate_marker_never_emits_nan() {
        let flat = Overview {
            min: Vec2::ZERO,
            max: Vec2::ZERO,
            width: 200.0,
            height: 100.0,
        };
        let cases = [
            (flat, ViewTransform::identity(Extent2::new(200, 100))),
            (piece(), ViewTransform::identity(Extent2::new(0, 100))),
        ];
        for (over, view) in cases {
            let style = viewport_style(over, view);
            assert!(!style.contains("NaN"), "{style}");
            for name in ["left", "top", "width", "height"] {
                assert!(css(&style, name).is_finite(), "{style}");
            }
            assert!(matrix(&style).iter().all(|v| v.is_finite()), "{style}");
        }
    }
}
