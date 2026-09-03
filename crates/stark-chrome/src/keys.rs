//! One keystroke, as the two binding tables read it (§25).
//!
//! The chord table names keys ([`crate::commands`]) and the drag table names presses
//! ([`crate::drags`]), and both start from the same three modifiers — which is why
//! [`Mods`] is here rather than in either. "The two tables read one keystroke the
//! same way" was a comment in the chord table; this is the type that makes it so.
//!
//! # What a frontend still owes
//!
//! Everything below is a *description* of a keystroke, and neither frontend's event
//! is one. A Dioxus keyboard event and a wgpui key-down event each answer the same
//! four questions in their own vocabulary, and translating is the frontend's one job
//! here — deliberately, because the answers are not the same:
//!
//! - **`accel`** is Ctrl on Windows and Linux and Command on a Mac, and only a
//!   frontend knows which machine it is on.
//! - **`typed`** is the character the key produces *under the current layout*, which
//!   is a fact the platform reports and nothing here can derive.
//! - **`code`** is the physical position, W3C-named. Both toolkits speak it.
//! - **`role`** is the handful of keys a capture spends on itself rather than
//!   binding, and each toolkit spells them differently.

use serde::{Deserialize, Serialize};

/// The three modifiers a binding may name, as an event reports them.
///
/// Both tables' chords carry this triple, and both compare it **exactly**: Ctrl+Alt
/// is not the Ctrl row with a bystanding Alt, it is a different chord. That is what
/// keeps a modifier from silently widening every binding it is added to.
///
/// `#[serde(default)]` for the store's reason (§25.6): a fourth modifier added one
/// day has to read as absent out of every chord written before it existed.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Mods {
    /// The accelerator: Ctrl, or Command on a Mac.
    pub ctrl: bool,
    pub shift: bool,
    pub alt: bool,
}

impl Mods {
    /// Whether nothing is held.
    ///
    /// A press with no modifier on the primary button is never a chord the drag
    /// table can name — that press *is* painting — which is what its capture spends
    /// it on.
    pub fn bare(self) -> bool {
        self == Self::default()
    }
}

/// Which of the few keys a **capture** spends on itself rather than binding.
///
/// Not an alphabet: the three below are the only non-character keys either table
/// needs to tell apart, and everything else is [`Ordinary`](Self::Ordinary) and
/// named by its [`Keystroke::code`].
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Role {
    /// Anything a binding could name.
    #[default]
    Ordinary,
    /// Calls a capture off, binding unchanged.
    Escape,
    /// Erases a binding rather than replacing it.
    Backspace,
    /// A bare modifier — a chord still being formed.
    Modifier,
}

/// One keydown, described in the terms a binding is written in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Keystroke<'a> {
    pub mods: Mods,
    /// The character this key types under the current layout, if it types exactly
    /// one. `None` for an F-key, an arrow, a modifier — and for a key held through
    /// Alt on some layouts, where what it types is not the key's own.
    pub typed: Option<char>,
    /// The physical position, by W3C `code` value. Empty where the platform will not
    /// say, which a capture treats as "nothing left to name this by".
    pub code: &'a str,
    pub role: Role,
}

impl<'a> Keystroke<'a> {
    /// A plain keystroke with no modifiers — the shape most tests want.
    pub fn new(typed: Option<char>, code: &'a str) -> Self {
        Self {
            mods: Mods::default(),
            typed,
            code,
            role: Role::Ordinary,
        }
    }

    /// The same keystroke with `mods` held.
    #[must_use]
    pub fn with(mut self, mods: Mods) -> Self {
        self.mods = mods;
        self
    }

    /// The same keystroke, named as one of the keys a capture spends.
    #[must_use]
    pub fn as_role(mut self, role: Role) -> Self {
        self.role = role;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Nothing held is bare, and any one modifier is not — the whole of what the
    /// drag table's "this press is painting" rests on.
    #[test]
    fn only_an_empty_triple_is_bare() {
        assert!(Mods::default().bare());
        for m in [
            Mods {
                ctrl: true,
                ..Default::default()
            },
            Mods {
                shift: true,
                ..Default::default()
            },
            Mods {
                alt: true,
                ..Default::default()
            },
        ] {
            assert!(!m.bare(), "{m:?} is a chord");
        }
    }

    /// A modifier added to a stored chord reads as absent, rather than making the
    /// record unreadable — §25.6's rule, which is why the struct defaults.
    #[test]
    fn a_chord_stored_before_a_modifier_existed_still_reads() {
        let older: Mods = serde_json::from_str(r#"{"ctrl":true,"shift":false}"#)
            .expect("a triple written before `alt` existed");
        assert!(older.ctrl && !older.alt);
    }
}
