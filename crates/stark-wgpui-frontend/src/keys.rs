//! This window's keystrokes, as the shared chord table reads them (§25).
//!
//! The table is `stark_chrome::commands`; what is here is the one translation this
//! frontend owes it — wgpui's `Keystroke` into a `stark_chrome::keys::Keystroke`.
//!
//! # The one thing wgpui cannot say
//!
//! A chord names its key one of two ways (`stark_chrome::commands::ChordKey`): by the
//! **character** it types, which survives a change of layout, or by its **position**,
//! which survives a change of *character* — `[` and `]` step the brush down and up
//! precisely because they sit side by side.
//!
//! The web gets position free: `KeyboardEvent.code` is a W3C name for the physical
//! key. wgpui reports no such thing — its `Keystroke::key` is a name like `"escape"`
//! or `"f5"`, which is neither a character nor a position — so the mapping below
//! reconstructs a code from that name for the keys the shipped table actually binds
//! spatially, and answers an empty code for everything else.
//!
//! Empty is the honest answer rather than a guess: a chord table already treats it as
//! "nothing left to name this by", so an unmapped key falls through to its character
//! or to nothing at all. What it costs is a *rebinding* to a key not on the list
//! below, which stores as `Code("")` and would never fire. The list is where that
//! stops being true, one row at a time — and a wgpui that grows a `code` deletes it.

use stark_chrome::keys::{Keystroke, Mods, Role};

/// The accelerator: Ctrl on Windows and Linux, Command on a Mac.
///
/// wgpui reports both — `control` and `platform` — and which one a chord means is
/// exactly the fact only a frontend holds (`stark_chrome::keys`).
fn accel(m: &wgpui::Modifiers) -> bool {
    if cfg!(target_os = "macos") {
        m.platform
    } else {
        m.control
    }
}

/// A W3C `code` for the keys the shipped chord table binds by position, and `""` for
/// everything else — see the module note.
///
/// wgpui lower-cases its key names, so the match is over those rather than over
/// anything a keycap says.
fn code_of(key: &str) -> &'static str {
    match key {
        "[" => "BracketLeft",
        "]" => "BracketRight",
        "escape" => "Escape",
        "enter" => "Enter",
        "tab" => "Tab",
        "space" => "Space",
        "backspace" => "Backspace",
        "delete" => "Delete",
        "left" => "ArrowLeft",
        "right" => "ArrowRight",
        "up" => "ArrowUp",
        "down" => "ArrowDown",
        _ => "",
    }
}

/// One wgpui keydown, described the way a binding is written.
pub fn stroke(k: &wgpui::Keystroke) -> Keystroke<'static> {
    let role = match k.key.as_str() {
        "escape" => Role::Escape,
        "backspace" => Role::Backspace,
        // wgpui delivers no keydown for a bare modifier, so `Modifier` is
        // unreachable here — kept as an arm rather than a comment because the
        // capture's contract is over the whole enum.
        "control" | "shift" | "alt" | "platform" | "function" => Role::Modifier,
        _ => Role::Ordinary,
    };
    // `key_char` is what was *typed* — `ß` for option-s — and `key` is what the keycap
    // says. The chord table wants the keycap's letter, so that Ctrl+Z is Z on a layout
    // where Ctrl+Z types nothing at all.
    let typed = one_char(&k.key).or_else(|| k.key_char.as_deref().and_then(one_char));
    Keystroke {
        mods: Mods {
            ctrl: accel(&k.modifiers),
            shift: k.modifiers.shift,
            alt: k.modifiers.alt,
        },
        typed,
        code: code_of(&k.key),
        role,
    }
}

/// A string that is exactly one `char`, as that char.
///
/// A dead key or an IME composition reports a longer one, and neither is a chord.
fn one_char(s: &str) -> Option<char> {
    let mut chars = s.chars();
    match (chars.next(), chars.next()) {
        (Some(c), None) => Some(c),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stark_chrome::commands::{Bindings, Command};

    fn press(key: &str, ctrl: bool, shift: bool) -> wgpui::Keystroke {
        wgpui::Keystroke {
            modifiers: wgpui::Modifiers {
                control: ctrl && !cfg!(target_os = "macos"),
                platform: ctrl && cfg!(target_os = "macos"),
                shift,
                alt: false,
                function: false,
            },
            key: key.to_string(),
            key_char: None,
        }
    }

    /// The exit criterion of §11.2's N3, as a test: the shipped table answers this
    /// frontend's keystrokes, so Ctrl+Z is undo here because it is undo there.
    #[test]
    fn the_shipped_table_answers_a_native_keystroke() {
        let table = Bindings::default();
        assert_eq!(
            table.lookup(&stroke(&press("z", true, false))),
            Some(Command::Undo)
        );
        assert_eq!(
            table.lookup(&stroke(&press("z", true, true))),
            Some(Command::Redo)
        );
        // A mnemonic chord is caseless: Shift is the chord's business, not the
        // letter's.
        assert_eq!(
            table.lookup(&stroke(&press("Z", true, false))),
            Some(Command::Undo)
        );
    }

    /// A spatial row answers too, over the reconstructed code — the half of the
    /// binding vocabulary wgpui does not report.
    #[test]
    fn a_spatial_chord_answers_over_the_reconstructed_code() {
        let table = Bindings::default();
        assert_eq!(
            table.lookup(&stroke(&press("[", false, false))),
            Some(Command::BrushSmaller)
        );
        assert_eq!(
            table.lookup(&stroke(&press("]", false, false))),
            Some(Command::BrushLarger)
        );
    }

    /// An unmapped key has no position to be named by, and says so rather than
    /// guessing at one.
    #[test]
    fn an_unmapped_key_reports_no_code() {
        assert_eq!(stroke(&press("f13", false, false)).code, "");
        assert_eq!(stroke(&press("escape", false, false)).code, "Escape");
    }

    /// The two keys a capture spends on itself are named by role, not by chord.
    #[test]
    fn a_capture_key_is_named_by_its_role() {
        assert_eq!(stroke(&press("escape", false, false)).role, Role::Escape);
        assert_eq!(
            stroke(&press("backspace", false, false)).role,
            Role::Backspace
        );
        assert_eq!(stroke(&press("a", false, false)).role, Role::Ordinary);
    }
}
