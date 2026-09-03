//! The one piece of the list-drag gesture that cannot leave a frontend: swallowing
//! the click a finished drag leaves behind.
//!
//! Everything else — [`Grab`](stark_chrome::reorder::Grab), the column arithmetic and
//! what a displaced row does about it — is `stark_chrome::reorder`, because it is
//! arithmetic over boxes and the hand. This is not: it reads and clears a `Signal`,
//! which is the definition of chrome (§11.2).

use dioxus::prelude::*;

use stark_chrome::reorder::Grab;

/// Whether the `click` the browser has just sent belongs to a drag that already
/// landed — and clears the grab either way, so the next press starts clean.
///
/// A row that both clicks and drags gets a click after *every* release, drag or not,
/// and on a list that has just reordered itself that click is aimed at a row which is
/// no longer the one that was pressed. Swallowing it is what keeps a move from also
/// being a selection of whatever took the moved row's place.
///
/// A press always overwrites the grab, so a finished one cannot outlive the gesture
/// that left it even if no click ever arrives — and while it waits it is inert, which
/// is [`Grab::spend`]'s job rather than this one's. It was not always: a finished grab
/// that a hover could re-arm turned "no click arrived" from a thing that costs nothing
/// into a row that follows the pointer until the panel is closed (`2026-08-09`).
pub fn claimed(grab: &mut Signal<Option<Grab>>) -> bool {
    let over = grab.peek().as_ref().is_some_and(Grab::over);
    if over {
        grab.set(None);
    }
    over
}
