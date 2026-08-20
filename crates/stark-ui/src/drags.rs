//! The drag-binding table (§25): which chord+button opens which canvas drag
//! gesture — the pointer half of the command registry (`crate::commands`).
//! §25.3 is the checklist for adding an action.
//!
//! A [`Command`](crate::commands::Command) is an act asked for *whole*; a drag
//! action is a **gesture**, with a press that opens it, moves that feed it and a
//! release that ends it. The gestures themselves stay where they live — the
//! brush-tuning drag is `input::Tune` (§18.1.9), the eyedropper is
//! `input::pick_color` and the `PickState` flags (§18.0.2) — because each owns
//! its own lifecycle and state. What this module owns is the one question the
//! canvas used to answer with a hand-written ladder of modifier tests: **which
//! of them does this press open?** One table, one reader on the press path
//! ([`find`]), and one on the advertisement path ([`armed`]) — so what a press
//! does and what the cursor promises cannot drift apart, which is the same
//! bargain the chord table makes between `find` and `shortcut`.
//!
//! Chords are **exact** about their modifiers, exactly as the keyboard's are
//! (`commands::Chord`): Ctrl+Alt+drag is not the Ctrl row with a bystanding
//! Alt, it is an unbound chord, and an unbound chord's press falls through to
//! what an unmodified press does — painting. Unlike the keyboard table, **Alt
//! is nameable here**: the chord table refuses it because AltGr arrives as
//! Ctrl+Alt and a layout types *through* it, but a drag types nothing, so the
//! trap has nothing to spring on.
//!
//! What is deliberately *not* a row:
//!
//! - **Navigation.** Space-drag, middle-drag and the two-finger gestures are
//!   `input::Nav`'s, asked before this table and shared with surfaces that have
//!   no brush (the transform overlay, the gradient bar). Space is a *hold*
//!   owning both edges of its key, the middle button is the pan whatever is
//!   held with it, and a second finger is hardware, not a chord — and none of
//!   the three reads its modifiers exactly, on purpose: space+Alt must stay a
//!   pan, not fall through an exact table to paint. Nothing stops `Pan` or
//!   `Zoom` becoming *actions* here one day, bound to ordinary chords (a bare
//!   right-drag pan is a row this table could hold); it is the holds
//!   themselves that are not rows.
//! - **The marquee's combine modifiers.** Shift and Alt over a selection tool
//!   modulate the paint gesture (`panels::select::modifier_mode`, §6.8) rather
//!   than replacing it with another, so they are the gesture's business — and
//!   the reason [`DragAction::PickColor`] declines there ([`claims`]).
//! - **The plain press.** Painting is not a bound action, it is what an
//!   unclaimed press *is* — the resting meaning every unbound chord falls
//!   through to, as an unclaimed keystroke falls through to the browser.
//!
//! The variant, not the chord, is an action's identity — two rows could name
//! one act, and a rebinding would move the chord while the name held still.
//! There is no user-facing rebinding *yet*: the override-table-over-defaults
//! shape is `commands::Bindings`' to copy the day a surface exists to edit it,
//! and storing a table nothing can edit would be scaffolding. Until then
//! [`defaults`] simply *is* the table, and [`lookup`] its one reader.

use dioxus::html::Modifiers;
use dioxus::html::input_data::MouseButton;
use dioxus::prelude::{Event, ModifiersInteraction, PointerData, PointerInteraction};

use crate::input::{accel, is_contact};
use crate::state::AppState;

/// The modifier half of a drag chord — and, held in
/// [`AppState::held_mods`](crate::state::AppState::held_mods), the modifiers
/// currently down, which is what the resting cursor asks the table with
/// ([`armed`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Mods {
    /// The accelerator: Ctrl, or Command on a Mac (`input::accel`).
    pub ctrl: bool,
    pub shift: bool,
    pub alt: bool,
}

impl Mods {
    /// The triple as an event reports it — the one translation from
    /// [`Modifiers`], shared by the press path and the key tracker so the two
    /// cannot read the same keystroke differently.
    pub fn of(m: Modifiers) -> Self {
        Self {
            ctrl: accel(m),
            shift: m.contains(Modifiers::SHIFT),
            alt: m.contains(Modifiers::ALT),
        }
    }
}

/// Which button a drag binding means, named the way the hand knows it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DragButton {
    /// A **contact** (`input::is_contact`): the primary button, or the pen's
    /// eraser end against the glass. The eraser is deliberately in — it tunes
    /// the eraser's own brush for the reason it erases (§18.1.8), and a "left"
    /// that meant only the mouse would make every bound drag work one way up
    /// the stylus and not the other.
    Left,
    /// The secondary button. Free for the taking on the canvas: the browser's
    /// context menu is already refused everywhere but text fields
    /// (`input::bind_context_menu`), and the right button is a tool only in
    /// the navigator's miniature.
    Right,
}

/// The press half of a drag binding: which modifier tier, on which button.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DragChord {
    pub mods: Mods,
    pub button: DragButton,
}

/// One gesture a bound press can open. The closed set the table maps into —
/// each variant is a gesture the canvas already knows how to drive, so adding
/// one is a variant, a row in [`defaults`], and an arm in the canvas's press
/// handler; the routing itself never grows another case.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DragAction {
    /// The brush-tuning drag (`input::Tune`, §18.1.9): Size sideways, Flow up
    /// and down, under the hand that is already on the painting.
    TuneBrush,
    /// The eyedropper (`input::pick_color`, §18.0.2): the press samples, and
    /// the drag keeps sampling, so a color is picked up without putting the
    /// brush down.
    PickColor,
}

/// The drag table Stark ships with. Rows are disjoint by construction — the
/// chords are exact, so no row can shadow another
/// (`tests::default_rows_are_disjoint`) — and there is no order to carry
/// meaning.
fn defaults() -> Vec<(DragChord, DragAction)> {
    fn on(ctrl: bool, shift: bool, alt: bool, button: DragButton) -> DragChord {
        DragChord {
            mods: Mods { ctrl, shift, alt },
            button,
        }
    }
    vec![
        (
            on(true, false, false, DragButton::Left),
            DragAction::TuneBrush,
        ),
        (
            on(false, false, true, DragButton::Left),
            DragAction::PickColor,
        ),
    ]
}

/// The action a chord asks for, if any — the policy half, taken apart from the
/// DOM event the way `Bindings::lookup` is so a test can reach it.
fn lookup(mods: Mods, button: DragButton) -> Option<DragAction> {
    defaults()
        .into_iter()
        .find(|(chord, _)| chord.mods == mods && chord.button == button)
        .map(|(_, action)| action)
}

/// The action a **left press** under `held` would open — what the resting
/// cursor and the options bar advertise ([`armed`]'s callers), asked of the
/// same table the press will ask, so the promise and the press cannot
/// disagree. Takes the held triple as a value rather than reading it, because
/// whether a call site subscribes or peeks is the call site's own discipline.
pub fn armed(held: Mods) -> Option<DragAction> {
    lookup(held, DragButton::Left)
}

/// The action `e` asks for, if any — the one reader on the canvas's press
/// path. A row that matches may still decline the *press* ([`claims`]), and a
/// declined press falls through to the paint path exactly as an unbound chord
/// does: over a selection tool, Alt+drag *is* the subtract marquee.
pub fn find(state: AppState, e: &Event<PointerData>) -> Option<DragAction> {
    let button = if is_contact(e) {
        DragButton::Left
    } else if e.trigger_button() == Some(MouseButton::Secondary) {
        DragButton::Right
    } else {
        return None;
    };
    lookup(Mods::of(e.modifiers()), button).filter(|action| action.claims(state))
}

impl DragAction {
    /// Whether this action claims a press right now — the act's own gate,
    /// asked by [`find`] where the canvas's ladder used to encode it in
    /// ordering, for `Command::run`'s reason: which question an act must ask
    /// is a fact about the act, not about the call site.
    fn claims(self, state: AppState) -> bool {
        match self {
            // Tuning edits no document — the brush is view state, and the
            // sliders this drag shadows are not refused mid-playback either
            // (`commands::step_radius` makes the same argument).
            DragAction::TuneBrush => true,
            // Two stand-downs. Over a selection tool Alt already means
            // subtract (§6.8), and whichever chord this action wears, a
            // marquee's combine modifiers outrank a sample — the selection
            // gesture is what the press is *for* there. And during playback a
            // sample would read the replay mid-flight: the picture under the
            // pointer is the playhead's, not the painting's, so the press
            // falls through to the guard that refuses paint for the same
            // reason.
            DragAction::PickColor => {
                !crate::panels::select::current_tool(state).is_selection()
                    && !crate::panels::timeline::is_playing(state)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_rows_are_disjoint() {
        let rows = defaults();
        for (i, (a, _)) in rows.iter().enumerate() {
            for (b, _) in &rows[i + 1..] {
                assert_ne!(a, b, "two rows on one chord: the table is not a function");
            }
        }
    }

    #[test]
    fn the_shipped_bindings() {
        let m = |ctrl, shift, alt| Mods { ctrl, shift, alt };
        assert_eq!(
            lookup(m(true, false, false), DragButton::Left),
            Some(DragAction::TuneBrush)
        );
        assert_eq!(
            lookup(m(false, false, true), DragButton::Left),
            Some(DragAction::PickColor)
        );
        // A bare press is not a row: painting is what an unbound press is,
        // not an act the table names.
        assert_eq!(lookup(m(false, false, false), DragButton::Left), None);
    }

    #[test]
    fn chords_are_exact() {
        let m = |ctrl, shift, alt| Mods { ctrl, shift, alt };
        // Ctrl+Alt is nobody's — not the Ctrl row with a bystander, exactly
        // as the keyboard table reads its modifiers.
        assert_eq!(lookup(m(true, false, true), DragButton::Left), None);
        assert_eq!(lookup(m(true, true, false), DragButton::Left), None);
        assert_eq!(lookup(m(false, true, true), DragButton::Left), None);
    }

    #[test]
    fn the_button_is_part_of_the_chord() {
        let m = |ctrl, shift, alt| Mods { ctrl, shift, alt };
        // Both shipped rows are left-drags; their chords on the right button
        // ask for nothing.
        assert_eq!(lookup(m(true, false, false), DragButton::Right), None);
        assert_eq!(lookup(m(false, false, true), DragButton::Right), None);
    }
}
