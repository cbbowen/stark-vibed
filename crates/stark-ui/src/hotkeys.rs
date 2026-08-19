//! The hotkey table (§11): every simple keyboard shortcut, declared as one row
//! of data.
//!
//! A *simple* shortcut is a chord that fires on its keydown and is finished —
//! Ctrl+Z, or a bare `]`. Each is a `(Chord, Hotkey)` row in [`BINDINGS`], and
//! the row is the entire declaration: the dispatch path asks [`find`] once
//! (`input`'s keydown handler), and the chrome prints its advertisements — the
//! menu's shortcut column, a tooltip's parenthesis — from the same rows through
//! [`label`]. One authority, so what the keyboard answers and what the menu
//! claims cannot drift apart. It is also the shape rebinding needs: the day
//! chords become user state, [`Hotkey`] is the stable name a stored binding
//! keys on, the chord is the half that turns into data, and `find` and `label`
//! are already the only two readers.
//!
//! What is deliberately *not* a row:
//!
//! - **Anything that owns both edges of its key.** A held digit is a grip on a
//!   brush slot with a release (§18.1.8), space is a pan for as long as it is
//!   down, Alt wears the eyedropper's cursor until it lifts. Those stay in
//!   `input`, which owns keyup.
//! - **Ctrl+V.** A paste is data arriving, not a command: the browser delivers
//!   the clipboard *with* the event, and must be left to (`crate::images`).
//!
//! The guard that precedes any binding at all — a keystroke in a text field is
//! the field's (`platform::KeyEvent::on_text_entry`) — is `input`'s too: by the
//! time this table is consulted, the keystroke is known to be the canvas's.

use dioxus::html::{Key, Modifiers};

use crate::input::accel;
use crate::panels::brush::{MAX_RADIUS, MIN_RADIUS};
use crate::platform;
use crate::state::{AppState, dispatch, update_brush};
use stark_engine::command::{DocCommand, ViewCommand};
use stark_model::document::SelectionOp;

/// The key half of a binding: which modifier tier, and which key.
///
/// Chords are **exact** about their modifiers. Ctrl+Shift+Z is not Ctrl+Z with
/// a bystanding Shift, it is another row; and a modifier no chord can even name
/// (Alt) matches nothing at all, because on Windows AltGr arrives as Ctrl+Alt,
/// and a table that shrugged at Alt would fire its Ctrl rows under a layout's
/// ordinary typing. (The digit rack tolerates Shift, and may: it is a hold in
/// `input`, not a row here, and its reasons are its own — §18.1.8.)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Chord {
    /// Ctrl, or Command on a Mac.
    pub ctrl: bool,
    pub shift: bool,
    pub key: ChordKey,
}

/// Which key, named the way the binding *means* it.
///
/// A **mnemonic** binding is about the letter — Z undoes wherever the layout
/// puts the Z — so it names the character the key types (`KeyboardEvent.key`),
/// compared caseless: Shift is the chord's business and CapsLock nobody's. A
/// **spatial** binding is about where the key sits — `[` and `]` step the brush
/// down and up precisely because they are side by side — so it names the
/// position (`KeyboardEvent.code`) and survives the layouts that type something
/// else there, which is the rack's own argument (`slots::of_code`, §18.1.8).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ChordKey {
    /// The character the key types, caseless.
    Char(&'static str),
    /// The physical position, by W3C `code` value.
    Code(&'static str),
}

/// What a chord is bound to: one nameable thing the keyboard can ask for.
///
/// The variant, not the chord, is a hotkey's identity — two chords may name the
/// same act (Ctrl+Y and Ctrl+Shift+Z), and when rebinding arrives the chords
/// move while these names hold still. What running one *means* — including the
/// gate it must ask first — lives on [`run`](Self::run), so a new binding
/// cannot forget its act's own rules.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Hotkey {
    Undo,
    Redo,
    /// "Select all" and "Deselect" are the same edit — a selection covering the
    /// whole canvas *is* no selection (§6.8) — so both spellings land here.
    SelectAll,
    InvertSelection,
    /// Mirror the view about the screen's vertical midline, whatever angle the
    /// canvas is at (`ViewTransform::mirror_screen_h`).
    MirrorH,
    BrushSmaller,
    BrushLarger,
}

/// The table. **A hotkey's first row is the one the chrome advertises**
/// ([`label`]): Ctrl+Y above Ctrl+Shift+Z because the menu has always said
/// Ctrl+Y, and Ctrl+D above Ctrl+A because the entry doing the advertising is
/// named "Deselect". Order carries no other meaning — chords are disjoint
/// (`tests::chords_are_disjoint`), so no row can shadow another.
pub const BINDINGS: &[(Chord, Hotkey)] = &[
    (
        Chord {
            ctrl: true,
            shift: false,
            key: ChordKey::Char("z"),
        },
        Hotkey::Undo,
    ),
    (
        Chord {
            ctrl: true,
            shift: false,
            key: ChordKey::Char("y"),
        },
        Hotkey::Redo,
    ),
    (
        Chord {
            ctrl: true,
            shift: true,
            key: ChordKey::Char("z"),
        },
        Hotkey::Redo,
    ),
    (
        Chord {
            ctrl: true,
            shift: false,
            key: ChordKey::Char("d"),
        },
        Hotkey::SelectAll,
    ),
    (
        Chord {
            ctrl: true,
            shift: false,
            key: ChordKey::Char("a"),
        },
        Hotkey::SelectAll,
    ),
    (
        Chord {
            ctrl: true,
            shift: true,
            key: ChordKey::Char("i"),
        },
        Hotkey::InvertSelection,
    ),
    (
        Chord {
            ctrl: false,
            shift: false,
            key: ChordKey::Char("h"),
        },
        Hotkey::MirrorH,
    ),
    (
        Chord {
            ctrl: false,
            shift: false,
            key: ChordKey::Code("BracketLeft"),
        },
        Hotkey::BrushSmaller,
    ),
    (
        Chord {
            ctrl: false,
            shift: false,
            key: ChordKey::Code("BracketRight"),
        },
        Hotkey::BrushLarger,
    ),
];

/// The hotkey `e` asks for, if any — the one reader on the dispatch path.
pub fn find(e: &platform::KeyEvent) -> Option<Hotkey> {
    let m = e.modifiers();
    lookup(
        accel(m),
        m.contains(Modifiers::SHIFT),
        m.contains(Modifiers::ALT),
        &e.key(),
        &e.code(),
    )
}

/// [`find`] with the event already taken apart — the half a test can reach,
/// there being no way to construct a DOM keystroke off the page.
fn lookup(accel: bool, shift: bool, alt: bool, key: &Key, code: &str) -> Option<Hotkey> {
    if alt {
        return None;
    }
    BINDINGS
        .iter()
        .find(|(chord, _)| {
            chord.ctrl == accel
                && chord.shift == shift
                && match chord.key {
                    ChordKey::Char(want) => {
                        matches!(key, Key::Character(c) if c.eq_ignore_ascii_case(want))
                    }
                    ChordKey::Code(want) => code == want,
                }
        })
        .map(|&(_, hotkey)| hotkey)
}

impl Hotkey {
    /// Do the thing. Each act's gate lives here rather than at the call site,
    /// because which question it must ask is a fact about the act: a document
    /// edit asks [`may_edit`], undo and redo instead *resolve* what is in
    /// flight ([`edit_history`]), and the view and the brush ask nothing. The
    /// keystroke is claimed by the caller either way — a declined command must
    /// not be answered with the browser's default.
    pub fn run(self, state: AppState) {
        match self {
            Hotkey::Undo => edit_history(state, DocCommand::Undo),
            Hotkey::Redo => edit_history(state, DocCommand::Redo),
            Hotkey::SelectAll => {
                if may_edit(state) {
                    dispatch(state, DocCommand::Select(SelectionOp::select_all()));
                }
            }
            Hotkey::InvertSelection => {
                if may_edit(state) {
                    dispatch(state, DocCommand::InvertSelection);
                }
            }
            Hotkey::MirrorH => dispatch(state, ViewCommand::MirrorH),
            Hotkey::BrushSmaller => step_radius(state, 1.0 / SIZE_STEP),
            Hotkey::BrushLarger => step_radius(state, SIZE_STEP),
        }
    }
}

/// One tap of `[` or `]`, as a ratio.
///
/// Equal *ratios* rather than equal pixels, because the hand feels radius
/// proportionally: the +1px that is a visible jump on a 5px liner is nothing on
/// a 300px wash. A tenth is about the smallest change a mark reliably shows,
/// and it compounds across the whole range quickly under the key's own
/// auto-repeat — 1 → 500 is ~65 repeats, a couple of seconds of holding `]`.
/// Up and down are exact inverses (multiply by it, divide by it), so a tap too
/// far is a tap back rather than a slowly drifting number.
const SIZE_STEP: f32 = 1.1;

/// Step the live brush's radius by `factor` — the keyboard sibling of the Size
/// slider and the accelerator drag (§18.1.9), writing through the same
/// [`update_brush`] and clamped to the same bounds, so a tap cannot put the
/// brush anywhere the panel could not show or take back.
///
/// Ungated by [`may_edit`] on purpose: tuning the brush edits no document, and
/// the slider this shadows is not refused mid-playback either — the keyboard
/// says what the panel says.
fn step_radius(state: AppState, factor: f32) {
    update_brush(state, move |b| {
        b.radius = (b.radius * factor).clamp(MIN_RADIUS, MAX_RADIUS);
    });
}

/// Whether a **document edit** may be accepted from the keyboard right now.
///
/// The two questions the canvas already asks of a press, asked of the shortcuts —
/// which were the one door into the document that asked neither:
///
/// - **The playhead is moving.** A commit clears the withheld half of the
///   timeline, so an edit laid under a running playback deletes the rest of the
///   piece (`crate::panels::timeline`). The canvas refuses a press for this;
///   Ctrl+A went through and truncated the history from the keyboard.
/// - **A mode is composing.** Its preview is computed against the committed
///   document (`crate::modes`), and the bar that carries these very commands has
///   stood down for the mode's own — deselecting mid-transform would move the
///   wrong region on "Done" (`crate::panels::select::SelectionBar`). The keyboard
///   says what the screen says.
fn may_edit(state: AppState) -> bool {
    !crate::panels::timeline::is_playing(state) && !crate::modes::is_composing(state)
}

/// Undo or redo, having first put down whatever was in hand.
///
/// Not [`may_edit`]'s flat refusal, because these two are not refusable in the
/// same sense. Nothing on screen says undo is unavailable — no bar stood down to
/// carry the message — so a shortcut that silently did nothing would read as a
/// broken keyboard rather than as a rule. Editing the history is instead an
/// unambiguous statement that the composition in flight is over, so it ends the
/// way scrubbing ends one: the preview dropped, nothing committed. Playback
/// stops for the same reason it stops when the transport is touched — the hand
/// has taken the playhead back off the loop that was moving it.
fn edit_history(state: AppState, command: DocCommand) {
    crate::panels::timeline::stop(state);
    crate::modes::leave(state);
    dispatch(state, command);
}

/// The chord the chrome should print for `hotkey` — its first row in
/// [`BINDINGS`].
///
/// Total, because every hotkey has a row (`tests::every_hotkey_bound`); when
/// rebinding makes an unbound hotkey expressible, this is where the `Option`
/// appears, with the feature. "Ctrl" names the accelerator on every platform
/// for now, as the menu always has — a ⌘ on the one platform that draws it is a
/// presentation question, not a binding one.
pub fn label(hotkey: Hotkey) -> String {
    let (chord, _) = BINDINGS
        .iter()
        .find(|&&(_, h)| h == hotkey)
        .expect("every Hotkey has a row in BINDINGS (tests::every_hotkey_bound)");
    let mut s = String::new();
    if chord.ctrl {
        s.push_str("Ctrl+");
    }
    if chord.shift {
        s.push_str("Shift+");
    }
    match chord.key {
        ChordKey::Char(c) => s.push_str(&c.to_ascii_uppercase()),
        // A position's label is its US engraving: what the keycap most often
        // says, and the only honest answer short of asking the layout — which
        // the web cannot portably do. Rebinding will let the label be wrong
        // less often than the binding already is.
        ChordKey::Code("BracketLeft") => s.push('['),
        ChordKey::Code("BracketRight") => s.push(']'),
        ChordKey::Code(code) => s.push_str(code),
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every variant, by hand — kept in step in review, since nothing here can
    /// make the compiler demand a new variant be listed. The rows a forgotten
    /// entry would weaken are `every_hotkey_bound`'s.
    const ALL: &[Hotkey] = &[
        Hotkey::Undo,
        Hotkey::Redo,
        Hotkey::SelectAll,
        Hotkey::InvertSelection,
        Hotkey::MirrorH,
        Hotkey::BrushSmaller,
        Hotkey::BrushLarger,
    ];

    fn ch(s: &str) -> Key {
        Key::Character(s.into())
    }

    #[test]
    fn chords_are_disjoint() {
        for (i, (a, _)) in BINDINGS.iter().enumerate() {
            for (b, _) in &BINDINGS[i + 1..] {
                assert_ne!(a, b, "two rows on one chord: the table is not a function");
            }
        }
    }

    #[test]
    fn every_hotkey_bound() {
        for &h in ALL {
            assert!(
                BINDINGS.iter().any(|&(_, k)| k == h),
                "{h:?} has no chord, so label() would panic in the chrome"
            );
        }
    }

    #[test]
    fn shift_is_part_of_the_chord() {
        assert_eq!(
            lookup(true, false, false, &ch("z"), "KeyZ"),
            Some(Hotkey::Undo)
        );
        assert_eq!(
            lookup(true, true, false, &ch("Z"), "KeyZ"),
            Some(Hotkey::Redo)
        );
        // Ctrl+Shift+Y is nobody's: an unclaimed chord falls through to the
        // browser rather than being Ctrl+Y plus a bystander.
        assert_eq!(lookup(true, true, false, &ch("Y"), "KeyY"), None);
    }

    #[test]
    fn caps_lock_is_nobodys() {
        // CapsLock types "Z" with no Shift held; the chord reads the modifier,
        // never the letter's case.
        assert_eq!(
            lookup(true, false, false, &ch("Z"), "KeyZ"),
            Some(Hotkey::Undo)
        );
    }

    #[test]
    fn alt_matches_nothing() {
        // AltGr arrives as Ctrl+Alt on Windows: a layout typing *through* Alt
        // must not find Ctrl rows under its ordinary characters.
        assert_eq!(lookup(true, false, true, &ch("z"), "KeyZ"), None);
        assert_eq!(lookup(false, false, true, &ch("h"), "KeyH"), None);
    }

    #[test]
    fn brackets_are_positions() {
        // Whatever the layout types on them, the pair beside P steps the brush.
        assert_eq!(
            lookup(false, false, false, &ch("ü"), "BracketLeft"),
            Some(Hotkey::BrushSmaller)
        );
        assert_eq!(
            lookup(false, false, false, &ch("]"), "BracketRight"),
            Some(Hotkey::BrushLarger)
        );
        assert_eq!(lookup(true, false, false, &ch("["), "BracketLeft"), None);
    }

    #[test]
    fn advertised_chords() {
        // A hotkey's first row is what the chrome prints; these strings are the
        // menu's shortcut column.
        assert_eq!(label(Hotkey::Undo), "Ctrl+Z");
        assert_eq!(label(Hotkey::Redo), "Ctrl+Y");
        assert_eq!(label(Hotkey::SelectAll), "Ctrl+D");
        assert_eq!(label(Hotkey::InvertSelection), "Ctrl+Shift+I");
        assert_eq!(label(Hotkey::BrushLarger), "]");
    }
}
