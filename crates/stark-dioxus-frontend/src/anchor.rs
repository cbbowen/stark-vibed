//! Standing a floating surface beside a box the stylesheet cannot see (§11).
//!
//! Two surfaces in the chrome are placed this way and both for one reason: what each
//! stands beside lives inside `.panel-stack`, which is a scroll container that clips
//! its overflow — and every panel in it carries a `backdrop-filter`, which makes a
//! containing block, so not even `position: fixed` gets out of one. A surface that
//! must fly *out* of that column cannot be drawn inside it at all. It is mounted at
//! the app root instead and told where to stand, and being told means measuring.
//!
//! The two are the guided tour's card (§24.3) and a panel's pop-out
//! (`widgets::PopoutId`, §25.7). What they have in common is here: how far off the
//! anchor to stand, how close to the window's edge to come, the declarations that
//! stop a surface hung from one edge running off the opposite one, and how to wait
//! for an anchor the browser has not laid out yet.
//!
//! What is deliberately *not* here is which side to stand on. The two answer that
//! differently enough that a shared vocabulary would be a third thing to keep true:
//! a card's side is stated on the lesson (`tutor::Side`), while a pop-out's is the
//! panel column's own geometry, which the stylesheet knows without being told
//! (`.stack-popout`).

use dioxus::prelude::*;

use crate::platform::{self, ElementBox};

/// How far a floating surface sits from the thing it was placed against, in CSS px.
///
/// The tour's card keeps its arrow in this gap, which is what sets the figure; a
/// pop-out has no arrow and takes the same distance anyway, because two surfaces
/// standing different distances off the same column is a difference the eye reads as
/// an accident rather than as a meaning.
pub const GAP: f32 = 14.0;

/// How close to the window's edge a floating surface may come, in CSS px. The panel
/// column's own inset, so a surface beside the stack stops where the stack does.
pub const EDGE: f32 = 14.0;

/// How wide a surface whose **right** edge is pinned at `x` may be, as a declaration.
///
/// A surface is placed from its anchor's edge and given no width to work with, so on
/// a narrow window it would otherwise run off the side. This is the answer, and it is
/// a *narrowing* rather than a nudge: a surface that shifted to stay on screen would
/// leave its arrow pointing at nothing, and the arrow is the half that says which
/// thing is being talked about.
///
/// Written as a declaration and not measured, because it does not have to be: the
/// stylesheet already gives the surface a width, this is a `max-width` over the top
/// of it, and `calc` knows the viewport where Rust would have to ask for it.
pub fn room_left(x: f32) -> String {
    format!("max-width: {:.1}px;", (x - EDGE).max(0.0))
}

/// [`room_left`] for a surface whose **left** edge is pinned at `x`.
pub fn room_right(x: f32) -> String {
    format!("max-width: calc(100vw - {x:.1}px - {EDGE}px);")
}

/// [`room_left`] for a surface **centred** on `x`, which is constrained by whichever
/// side of it has less room — hence the `min`, and the doubling: it grows both ways
/// from the middle, so it may only be twice the narrower half.
pub fn room_about(x: f32) -> String {
    format!(
        "max-width: calc(min({x:.1}px, 100vw - {x:.1}px) * 2 - {:.1}px);",
        2.0 * EDGE
    )
}

/// How tall a surface **hanging from** `y` may be, as a declaration — the room left
/// between it and the foot of the window.
///
/// [`room_left`]'s argument stood on end, with one difference that decides which
/// callers can use it: narrowing a surface makes it *taller*, so a width cap is
/// something any surface can take, while a height cap only works for one that can
/// scroll what does not fit. The tour's card cannot and does not ask; a pop-out is a
/// list or a picker and does (`.stack-popout`).
pub fn room_below(y: f32) -> String {
    format!("max-height: calc(100vh - {y:.1}px - {EDGE}px);")
}

/// [`room_below`] for a surface **rising from** `y`: the room above it.
pub fn room_above(y: f32) -> String {
    format!("max-height: {:.1}px;", (y - EDGE).max(0.0))
}

/// How many animation frames [`measure`] will wait for its anchor before giving up.
///
/// Eight is about an eighth of a second, which is far longer than a render and a
/// layout and far shorter than anybody notices a surface arriving. It is a *bound*
/// rather than a duration: the ordinary case answers on the first or second frame,
/// and this only decides how long an anchor that is never coming keeps being asked
/// for.
pub const TRIES: u32 = 8;

/// Measure whatever `selector` finds into `at`, asking again on the next few frames
/// while it finds nothing.
///
/// **The retry is not defensive, it is the ordinary path.** A surface is often placed
/// by the very effect that *revealed* what it points at — the panel a lesson opened,
/// the rack it pinned — and a reveal is a signal write whose render has not happened,
/// let alone been laid out. So the first look routinely finds nothing, and a single
/// animation frame is a race against Dioxus's own patch rather than a guarantee.
///
/// Losing that race used to be silent and strange: the tour's card was armed and
/// correct and simply never drew, until something *else* its effect follows moved and
/// measured it again. Since `canvas_active` is one of those, the symptom was a tip
/// that appeared when the artist next painted a stroke — nowhere near the click that
/// earned it.
///
/// `still` is what the chain was started for, asked again on every frame: a chain in
/// flight when the surface changes — a second lesson promoted, a different pop-out
/// opened — is answering a question nobody asked any more, and must not write the new
/// surface's box away.
///
/// Setting `at` to `None` on the last try is not a failure path either. It is what
/// takes a surface down that is pointing at something gone; the caller decides
/// whether the thing itself survives that (`tutor::abandon`).
pub fn measure(
    selector: String,
    mut at: Signal<Option<ElementBox>>,
    still: impl Fn() -> bool + Copy + 'static,
    tries: u32,
) {
    if !still() {
        return;
    }
    if let Some(found) = platform::anchor_box(&selector) {
        at.set(Some(found));
        return;
    }
    if tries == 0 {
        at.set(None);
        return;
    }
    platform::on_animation_frame(move || measure(selector, at, still, tries - 1));
}

/// Keep `at` on whatever `measure` answers, every frame, for as long as `still` says
/// the surface is wanted.
///
/// [`measure`]'s question the other way up, and the difference is what each is
/// waiting for. A card is placed **once**, by the very effect that revealed its
/// anchor, so all it is waiting for is the browser to lay that anchor out — hence a
/// few frames and then give up. A pop-out's anchor is a row that was already on
/// screen when it was pressed, and the question is whether it *stays* there. It does
/// not: the column it lives in is a scroller, the window resizes, a panel above it
/// opens or folds or is dragged, and every one of those moves the row without
/// touching anything about the pop-out.
///
/// **A frame loop rather than a list of listeners**, and that is the whole argument
/// for it. Each of those causes has an event, the events are not the same shape — a
/// scroll does not bubble, a fold is not an event at all — and a listener per cause is
/// a list that is wrong again the next time somebody adds a way to move a panel.
/// Asking every frame rules the class out instead, for one `getBoundingClientRect` per
/// frame over the few seconds a pop-out is open.
///
/// It writes only where the answer *changed*, which is what keeps that from being a
/// render per frame: the ordinary frame measures the same box and does nothing. And
/// `measure` is the caller's rather than a selector, because "where it is" is not
/// always "what the browser says": a row scrolled out of the column it lives in still
/// has a box, and standing a surface beside it would put that surface off the screen
/// (`panels::popout`).
///
/// Two loops for one surface is possible — `still` can go false and true again before
/// a frame lands — and costs a second measurement of the same box and nothing else,
/// which is why the guard is a predicate rather than a token to be held.
pub fn follow(
    mut at: Signal<Option<ElementBox>>,
    measure: impl Fn() -> Option<ElementBox> + Copy + 'static,
    still: impl Fn() -> bool + Copy + 'static,
) {
    if !still() {
        return;
    }
    let found = measure();
    if *at.peek() != found {
        at.set(found);
    }
    platform::on_animation_frame(move || follow(at, measure, still));
}
