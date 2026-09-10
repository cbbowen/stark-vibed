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
//! - **`ctrl`** is the accelerator, which is Ctrl *or* Command on every machine
//!   ([`accel`]); what the frontend owes is which of its toolkit's modifiers those two
//!   are.
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
    /// The accelerator: Ctrl or the platform key, either one on every OS ([`accel`]).
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

/// Whether this keystroke is the **space bar**, by either of its names.
///
/// Both, because Alt takes the character away: a layout reports space as the typed
/// `' '` ordinarily and as the positional `"Space"` under a modifier that consumes
/// it, and a reader that knew only one would lose the key exactly when a modifier is
/// held — which is when the scrubby zoom wants it (§18.1.9).
///
/// Here rather than in either frontend because three places ask: the chord table,
/// which claims space before any binding can have it, and each app's keydown, which
/// arms the pan off the key itself. One authority, so a frontend cannot come to think
/// space is down while the table thinks it is a chord.
pub fn is_space(stroke: &Keystroke<'_>) -> bool {
    stroke.code == "Space" || stroke.typed == Some(' ')
}

/// Whether the **accelerator** is held, from the two keys that can be it: `control`,
/// and `platform` — Command, or the Windows key.
///
/// Either, on every OS, rather than asking which machine this is: a binding that insisted
/// on Ctrl would be unreachable on the one platform where Ctrl+click is how a secondary
/// click is reported, and one that insisted on Command would be unreachable on keyboards
/// without one.
pub fn accel(control: bool, platform: bool) -> bool {
    control || platform
}

/// `s` as a `char` when it is exactly one, else `None`.
///
/// What a key *types* has to be a single character to be a chord: a dead key or an IME
/// composition reports a longer string, and neither is one.
pub fn one_char(s: &str) -> Option<char> {
    let mut chars = s.chars();
    match (chars.next(), chars.next()) {
        (Some(c), None) => Some(c),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Either key is the accelerator, and only neither is not.
    #[test]
    fn ctrl_or_command_is_the_accelerator_everywhere() {
        assert!(accel(true, false));
        assert!(accel(false, true));
        assert!(accel(true, true));
        assert!(!accel(false, false));
    }

    /// Exactly one `char` types that char; nothing, a composed sequence or two
    /// characters type nothing a chord can name.
    #[test]
    fn only_a_single_char_is_typed() {
        assert_eq!(one_char("z"), Some('z'));
        assert_eq!(one_char("ß"), Some('ß'));
        assert_eq!(one_char(""), None);
        assert_eq!(one_char("e\u{301}"), None);
        assert_eq!(one_char("ab"), None);
    }

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
