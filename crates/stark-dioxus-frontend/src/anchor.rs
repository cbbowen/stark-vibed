//! Standing a floating surface beside a box the stylesheet cannot see (§11).
//!
//! `.panel-stack` clips its overflow, and every panel in it carries a `backdrop-filter`,
//! which traps even `position: fixed`. A surface that must fly out of the column is
//! mounted at the app root and placed by measuring: the tour's card (§24.3) and a panel's
//! pop-out (`widgets::PopoutId`, §25.7). Which side each stands on is its own
//! (`tutor::Side`, `.stack-popout`).

use dioxus::prelude::*;

use crate::platform::{self, ElementBox};

/// How far a floating surface sits from the thing it was placed against, in CSS px. Set
/// by the tour card's arrow; a pop-out takes the same so surfaces off one column agree.
pub const GAP: f32 = 14.0;

/// How close to the window's edge a floating surface may come, in CSS px. The panel
/// column's own inset, so a surface beside the stack stops where the stack does.
pub const EDGE: f32 = 14.0;

/// How wide a surface whose **right** edge is pinned at `x` may be, as a declaration.
///
/// A narrowing rather than a shift, so a card's arrow still points at its anchor.
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

/// How tall a surface **hanging from** `y` may be, as a declaration. Only for a surface
/// that scrolls what does not fit (`.stack-popout`); the tour's card cannot.
pub fn room_below(y: f32) -> String {
    format!("max-height: calc(100vh - {y:.1}px - {EDGE}px);")
}

/// [`room_below`] for a surface **rising from** `y`: the room above it.
pub fn room_above(y: f32) -> String {
    format!("max-height: {:.1}px;", (y - EDGE).max(0.0))
}

/// How many animation frames [`measure`] waits for its anchor before giving up — about an
/// eighth of a second. A bound: the ordinary case answers on the first or second frame.
pub const TRIES: u32 = 8;

/// Measure whatever `selector` finds into `at`, retrying on the next few frames while it
/// finds nothing.
///
/// The retry is the ordinary path: a surface is often placed by the effect that revealed
/// its anchor, whose render and layout have not happened yet.
///
/// `still` is asked on every frame, so a chain whose surface has since changed does not
/// write the new surface's box away. `None` on the last try takes down a surface pointing
/// at something gone; whether the thing itself survives is the caller's
/// (`Tour::abandon`).
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

/// Keep `at` on whatever `measure` answers, every frame, while `still` holds.
///
/// Unlike [`measure`]'s one-shot wait, a pop-out's row keeps moving — the column scrolls,
/// the window resizes, a panel above folds — and those causes share no event, so this
/// polls every frame and writes only when the box changed; a duplicate loop costs one
/// extra measurement, so the guard is a predicate. `measure` is the caller's because a
/// row scrolled out of its column still has a box (`panels::popout`).
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
