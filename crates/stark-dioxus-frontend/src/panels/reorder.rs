//! The two pieces of the list-drag gesture that cannot leave a frontend: swallowing
//! the click a finished drag leaves behind, and writing a row's motion as CSS.
//!
//! Everything else — [`Grab`](stark_ui::reorder::Grab), the column arithmetic and
//! what a displaced row does about it — is `stark_ui::reorder`, because it is
//! arithmetic over boxes and the hand. Neither of these is: one reads and clears a
//! `Signal`, and the other emits a stylesheet dialect. Both are the definition of
//! chrome (§11.2).

use dioxus::prelude::*;

use stark_ui::reorder::{Grab, Motion};

/// How long a displaced row takes to reach its new place. Long enough to be followed
/// by eye, short enough that the list has settled by the time the hand arrives.
const SLIDE_MS: u32 = 180;

/// A row's own two declarations — **both of them, every render, including the ones
/// that are "off"**.
///
/// Inline styles are applied property by property rather than by replacing the
/// attribute, so a declaration left out of a render is not cleared: it keeps whatever
/// the last render that *did* mention it gave it. That is how a dropped row was once
/// left wearing a preview's transform over a list that had since reordered
/// (`layout::Panel` carries the same scar). Writing the pair from one place is one
/// place for the rule to hold rather than one per panel.
///
/// A free function rather than a method on [`Motion`], because the orphan rule puts
/// one out of reach here — and because the shift, the lift and the ease are what a
/// native list would want too, while the declarations are this frontend's alone.
pub fn css(motion: Motion) -> String {
    let (dx, dy) = motion.shift;
    let ease = if motion.live && !motion.lifted {
        format!("transform {SLIDE_MS}ms ease")
    } else {
        "none".to_string()
    };
    format!("transform: translate({dx}px, {dy}px); transition: {ease};")
}

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Every row's transform is written on every render, including at rest — the
    /// declaration a render leaves out is the one that goes stale.
    #[test]
    fn a_resting_row_still_states_its_transform() {
        let css = css(Motion::default());
        assert!(css.contains("translate(0px, 0px)"), "{css}");
        assert!(css.contains("transition: none"), "{css}");
    }
}
