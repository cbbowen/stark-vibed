//! The command registry (§11): every simple act the chrome can ask for,
//! declared as one variant of [`Command`] carrying its whole description.
//!
//! A *simple* command is one that takes no argument at its call site — click
//! "Undo", press Ctrl+Z, pick Undo from the rail's menu, and it is the same act
//! reached three ways. [`Command`] is that act's one name, and everything the
//! chrome says about it hangs off the variant: the word a control wears
//! ([`name`](Command::name), [`word`](Command::word)), the mark beside it
//! ([`icon`](Command::icon)), the tooltip ([`tooltip`](Command::tooltip)),
//! whether a menu row is greyed ([`enabled`](Command::enabled)), and what
//! running it *means* — including the gate it must ask first — on
//! [`run`](Command::run). A control renders a command rather than restating it
//! (`crate::widgets::CommandButton`, the rail's `CmdItem`), so a place that
//! reaches an act cannot describe it differently from the next place, and
//! cannot forget its rules: the rail's menu did exactly that, dispatching
//! Deselect with no gate while the keyboard asked [`may_edit`] first.
//!
//! The keyboard is one column of the registry. Each simple chord is a
//! `(Chord, Command)` row in [`BINDINGS`], the dispatch path asks [`find`] once
//! (`input`'s keydown handler), and the chrome prints its advertisements — the
//! menu's shortcut column, a tooltip's parenthesis — from the same rows through
//! [`label`]. One authority, so what the keyboard answers and what the menu
//! claims cannot drift apart. It is also the shape rebinding needs: the day
//! chords become user state, [`Command`] is the stable name a stored binding
//! keys on, the chord is the half that turns into data, and `find` and `label`
//! are already the only two readers.
//!
//! What is deliberately *not* a chord row:
//!
//! - **Anything that owns both edges of its key.** A held digit is a grip on a
//!   brush slot with a release (§18.1.8), space is a pan for as long as it is
//!   down, Alt wears the eyedropper's cursor until it lifts. Those stay in
//!   `input`, which owns keyup.
//! - **Ctrl+V.** A paste is data arriving, not a command: the browser delivers
//!   the clipboard *with* the event, and must be left to (`crate::images`).
//!
//! And what is not a *variant*: anything parameterized. A control that acts on
//! a particular row — this layer's eye, that guide's trash — names its target,
//! and a registry of every (act, target) pair would be a second copy of the
//! panels. The registry holds the acts a search bar could list or a chord could
//! carry whole.
//!
//! The guard that precedes any binding at all — a keystroke in a text field is
//! the field's (`platform::KeyEvent::on_text_entry`) — is `input`'s too: by the
//! time this table is consulted, the keystroke is known to be the canvas's.

use dioxus::html::{Key, Modifiers};
use dioxus::prelude::{ReadableExt, Signal, WritableExt};

use crate::icons;
use crate::input::accel;
use crate::panels::brush::{MAX_RADIUS, MIN_RADIUS};
use crate::platform;
use crate::state::{AppState, dispatch, update_brush};
use stark_engine::ObservableState;
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

/// One nameable thing the chrome can ask for whole: no argument at the call
/// site, no gesture to compose. The variant, not any control or chord, is the
/// act's identity — two chords may name the same act (Ctrl+Y and Ctrl+Shift+Z),
/// three surfaces may carry it (the menu, a bar chip, the keyboard), and when
/// rebinding arrives the chords move while these names hold still.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Command {
    Undo,
    Redo,
    /// "Select all" and "Deselect" are the same edit — a selection covering the
    /// whole canvas *is* no selection (§6.8) — so both spellings (Ctrl+A and
    /// Ctrl+D) land here, named for what the chrome calls it.
    Deselect,
    InvertSelection,
    /// Mirror the view about the screen's vertical midline, whatever angle the
    /// canvas is at (`ViewTransform::mirror_screen_h`).
    MirrorView,
    BrushSmaller,
    BrushLarger,
    /// Open the "New document…" dialog (`main::NewDocumentModal`).
    NewDocument,
    OpenDocument,
    SaveDocument,
    /// Import a picture into the painting (§23). The act it commits is a
    /// *placement* — what the document does with the picture — where the name
    /// says where it came from, which is the question a menu answers.
    ImportImage,
    /// Open the "Export image…" dialog (`files::ExportModal`).
    ExportImage,
    /// Start sharing and open the session dialog. Sharing starts on the
    /// command, not on a second button inside the dialog: the dialog exists to
    /// hand over the link. A no-op once the session is live.
    Share,
    /// Enter or leave Timeline mode (§18.2.4) — a mode rather than a dialog, so
    /// its menu row carries a check.
    ToggleTimeline,
    /// Open the Timing Stats dialog (§7.1, `timings::TimingModal`).
    TimingStats,
    Credits,
    /// Show or hide the navigator's miniature (§11).
    ToggleNavigator,
    /// Pin or unpin the quick-brush rack (§18.1.8) — the mouse-only way to a
    /// slot, which a hand with a pen and no keyboard has no other route to.
    ToggleQuickBrushes,
    /// Open the ⚙ preferences dialog (`settings::SettingsModal`).
    Settings,
    /// Open the full brush editor off the Brush panel.
    EditBrush,
    /// Open the "Save preset" dialog for the brush in hand.
    SavePreset,
    /// Pick the selected paint up into the transform widget (§16.6).
    Transform,
    /// Fill the selection with the brush's paint (§18.0.4). The color comes off
    /// the brush — a fill lays the paint you have in hand, so the Color panel
    /// is already its setting — and the selection's own coverage bounds it
    /// (`FillOp::of_selection`).
    FillSelection,
    /// The same act with the parcel varying along a dragged axis (§22.4) — a
    /// mode rather than a click, because the axis is composed by hand and
    /// judged by eye.
    GradientFill,
    /// Add a paint layer into the selected layer's own stack, above it — adding
    /// while working inside a group should land in that group, which is where
    /// you are looking.
    AddLayer,
    /// Add a frame at the top of the stack and select it (§15.7).
    AddFrame,
    /// Add a perspective guide where the artist is looking and pick it up
    /// (§20.5): adding *is* asking to shape it.
    AddPerspective,
}

/// The chord table. **A command's first row is the one the chrome advertises**
/// ([`label`]): Ctrl+Y above Ctrl+Shift+Z because the menu has always said
/// Ctrl+Y, and Ctrl+D above Ctrl+A because the entry doing the advertising is
/// named "Deselect". Order carries no other meaning — chords are disjoint
/// (`tests::chords_are_disjoint`), so no row can shadow another. Most commands
/// have no row at all: a chord is one way to reach an act, not part of being
/// one.
pub const BINDINGS: &[(Chord, Command)] = &[
    (
        Chord {
            ctrl: true,
            shift: false,
            key: ChordKey::Char("z"),
        },
        Command::Undo,
    ),
    (
        Chord {
            ctrl: true,
            shift: false,
            key: ChordKey::Char("y"),
        },
        Command::Redo,
    ),
    (
        Chord {
            ctrl: true,
            shift: true,
            key: ChordKey::Char("z"),
        },
        Command::Redo,
    ),
    (
        Chord {
            ctrl: true,
            shift: false,
            key: ChordKey::Char("d"),
        },
        Command::Deselect,
    ),
    (
        Chord {
            ctrl: true,
            shift: false,
            key: ChordKey::Char("a"),
        },
        Command::Deselect,
    ),
    (
        Chord {
            ctrl: true,
            shift: true,
            key: ChordKey::Char("i"),
        },
        Command::InvertSelection,
    ),
    (
        Chord {
            ctrl: false,
            shift: false,
            key: ChordKey::Char("h"),
        },
        Command::MirrorView,
    ),
    (
        Chord {
            ctrl: false,
            shift: false,
            key: ChordKey::Code("BracketLeft"),
        },
        Command::BrushSmaller,
    ),
    (
        Chord {
            ctrl: false,
            shift: false,
            key: ChordKey::Code("BracketRight"),
        },
        Command::BrushLarger,
    ),
];

/// The command `e` asks for, if any — the one reader on the dispatch path.
pub fn find(e: &platform::KeyEvent) -> Option<Command> {
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
fn lookup(accel: bool, shift: bool, alt: bool, key: &Key, code: &str) -> Option<Command> {
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
        .map(|&(_, command)| command)
}

impl Command {
    /// The command's display name: what a menu row or a search result would
    /// print. A trailing `…` is part of the name — it says the act opens a
    /// dialog that will ask more before anything happens.
    pub fn name(self) -> &'static str {
        match self {
            Command::Undo => "Undo",
            Command::Redo => "Redo",
            Command::Deselect => "Deselect",
            Command::InvertSelection => "Invert selection",
            Command::MirrorView => "Mirror view",
            Command::BrushSmaller => "Shrink brush",
            Command::BrushLarger => "Enlarge brush",
            Command::NewDocument => "New document\u{2026}",
            Command::OpenDocument => "Open\u{2026}",
            Command::SaveDocument => "Save",
            Command::ImportImage => "Import image\u{2026}",
            Command::ExportImage => "Export image\u{2026}",
            Command::Share => "Share\u{2026}",
            Command::ToggleTimeline => "Timeline",
            Command::TimingStats => "Timing stats\u{2026}",
            Command::Credits => "Credits\u{2026}",
            Command::ToggleNavigator => "Navigator",
            Command::ToggleQuickBrushes => "Quick brushes",
            Command::Settings => "Settings",
            Command::EditBrush => "Edit brush\u{2026}",
            Command::SavePreset => "Save preset\u{2026}",
            Command::Transform => "Transform",
            Command::FillSelection => "Fill selection",
            Command::GradientFill => "Gradient fill",
            Command::AddLayer => "Add layer",
            Command::AddFrame => "Add frame",
            Command::AddPerspective => "Add perspective grid",
        }
    }

    /// The terse word a chip wears where its bar or header already names the
    /// subject: "Invert" on the selection bar, "Layer" beside the stack's other
    /// two adds. Everything else keeps its [`name`](Self::name) — the split
    /// exists so both spellings are the registry's, not a call site's.
    pub fn word(self) -> &'static str {
        match self {
            Command::InvertSelection => "Invert",
            Command::FillSelection => "Fill",
            Command::GradientFill => "Gradient",
            Command::AddLayer => "Layer",
            Command::AddFrame => "Frame",
            Command::AddPerspective => "Perspective",
            _ => self.name(),
        }
    }

    /// The command's mark (`crate::icons`). Total, because a control rendering
    /// a command has nothing else to wear; the three keyboard-only commands
    /// wear the mark of the knob they step or the subject they turn, on the
    /// sharing argument `icons` already makes — the bracket keys are the Size
    /// slider's own knob (§18.1.9), so they wear its ruler.
    pub fn icon(self) -> &'static str {
        match self {
            Command::Undo => icons::UNDO,
            Command::Redo => icons::REDO,
            Command::Deselect => icons::SELECTION_NONE,
            Command::InvertSelection => icons::SELECTION_INVERT,
            Command::MirrorView => icons::MIRROR_VIEW,
            Command::BrushSmaller | Command::BrushLarger => icons::SIZE,
            Command::NewDocument => icons::ADD,
            Command::OpenDocument => icons::OPEN_DOC,
            Command::SaveDocument => icons::SAVE,
            Command::ImportImage => icons::IMPORT_IMAGE,
            Command::ExportImage => icons::EXPORT,
            Command::Share => icons::SHARE,
            Command::ToggleTimeline => icons::TIMELINE,
            Command::TimingStats => icons::TIMING,
            Command::Credits => icons::CREDITS,
            Command::ToggleNavigator => icons::NAVIGATOR,
            Command::ToggleQuickBrushes => icons::QUICK_BRUSHES,
            Command::Settings => icons::SETTINGS,
            Command::EditBrush => icons::EDIT_BRUSH,
            Command::SavePreset => icons::SAVE,
            Command::Transform => icons::TRANSFORM,
            Command::FillSelection => icons::PAINT_BUCKET,
            Command::GradientFill => icons::GRADIENT,
            Command::AddLayer => icons::ADD_LAYER,
            Command::AddFrame => icons::ADD_FRAME,
            Command::AddPerspective => icons::ADD_LAYER,
        }
    }

    /// The sentence a tooltip leads with — what the act will do, not how it is
    /// reached ([`tooltip`](Self::tooltip) appends that half). Falls back to
    /// the name where the name already is the sentence.
    pub fn hint(self) -> &'static str {
        match self {
            Command::MirrorView => "Mirror the view left-to-right",
            Command::BrushSmaller => "Step the brush size down",
            Command::BrushLarger => "Step the brush size up",
            Command::EditBrush => "Open the full brush editor",
            Command::SavePreset => "Save the current brush as a preset",
            Command::Transform => "Move, scale, flip, perspective or warp the selected paint (§16)",
            Command::FillSelection => "Fill the selection with the brush's paint",
            Command::GradientFill => {
                "Fill the selection with a gradient \u{2014} pick or \
                 trace the ramp on the bar, drag the axis, then Done"
            }
            Command::AddLayer => "Add a paint layer above the selected one",
            Command::AddFrame => {
                "Add a frame: it marks what the piece is, and clips nothing \u{2014} \
                 paint past it and re-crop whenever you like"
            }
            Command::AddPerspective => "Add a perspective grid where you are looking",
            _ => self.name(),
        }
    }

    /// The `title` a control rendering this command carries: the hint, plus the
    /// advertised chord for the commands the keyboard can reach.
    pub fn tooltip(self) -> String {
        match label(self) {
            Some(chord) => format!("{} ({chord})", self.hint()),
            None => self.hint().to_string(),
        }
    }

    /// Whether the chrome should offer this command right now — the menu's
    /// greyed rows, read off the projection so a disabled entry is a fact about
    /// the document ("nothing to undo") rather than a mood.
    ///
    /// **Presentation only.** The act's own gate lives on [`run`](Self::run)
    /// and asks different questions, deliberately: undo during playback is
    /// *enabled* — nothing on screen says otherwise — and resolves what is in
    /// flight rather than refusing (see [`edit_history`]). A caller must not
    /// skip `run`'s gate because this said yes.
    ///
    /// `None` is startup — no engine yet, so no document: the commands that ask
    /// the projection answer no, and everything else (a dialog, a file pick)
    /// needs nothing from it.
    pub fn enabled(self, o: Option<&ObservableState>) -> bool {
        match self {
            Command::Undo => o.is_some_and(|o| o.can_undo),
            Command::Redo => o.is_some_and(|o| o.can_redo),
            Command::Deselect
            | Command::InvertSelection
            | Command::Transform
            | Command::FillSelection
            | Command::GradientFill => o.is_some_and(|o| o.has_selection),
            _ => true,
        }
    }

    /// Do the thing. Each act's gate lives here rather than at the call site,
    /// because which question it must ask is a fact about the act: a document
    /// edit asks [`may_edit`], undo and redo instead *resolve* what is in
    /// flight ([`edit_history`]), and the view, the brush, the dialogs and the
    /// toggles ask nothing. A keystroke is claimed by the caller either way — a
    /// declined command must not be answered with the browser's default.
    pub fn run(self, state: AppState) {
        match self {
            Command::Undo => edit_history(state, DocCommand::Undo),
            Command::Redo => edit_history(state, DocCommand::Redo),
            Command::Deselect => {
                if may_edit(state) {
                    dispatch(state, DocCommand::Select(SelectionOp::select_all()));
                }
            }
            Command::InvertSelection => {
                if may_edit(state) {
                    dispatch(state, DocCommand::InvertSelection);
                }
            }
            Command::MirrorView => dispatch(state, ViewCommand::MirrorH),
            Command::BrushSmaller => step_radius(state, 1.0 / SIZE_STEP),
            Command::BrushLarger => step_radius(state, SIZE_STEP),
            Command::NewDocument => open_dialog(state.dialogs.new_document),
            Command::OpenDocument => crate::files::open_document(state),
            Command::SaveDocument => crate::files::save_document(state),
            Command::ImportImage => crate::images::import_image(state),
            Command::ExportImage => open_dialog(state.dialogs.export),
            Command::Share => {
                crate::collab::share(state);
                open_dialog(state.dialogs.session);
            }
            Command::ToggleTimeline => {
                let open = *state.timeline.open.peek();
                crate::panels::timeline::set_open(state, !open);
            }
            Command::TimingStats => open_dialog(state.dialogs.timing),
            Command::Credits => open_dialog(state.dialogs.credits),
            Command::ToggleNavigator => {
                let open = *state.navigator.peek();
                crate::navigator::set_open(state, !open);
            }
            Command::ToggleQuickBrushes => {
                let mut pinned = state.slots.pinned;
                let now = !*pinned.peek();
                pinned.set(now);
            }
            Command::Settings => open_dialog(state.dialogs.settings),
            Command::EditBrush => {
                open_dialog(state.brush_editor_open);
                // The dialog is frontend state and reaches no engine, so there
                // is no command for the tour to read (§24.2). Its series of
                // cards is the one thing this click owes anybody.
                crate::tutor::did(state, crate::tutor::Deed::OpenedBrushEditor);
            }
            Command::SavePreset => open_dialog(state.preset_save_open),
            Command::Transform => {
                if may_edit(state) {
                    crate::panels::transform::begin_transform(state);
                }
            }
            Command::FillSelection => {
                if may_edit(state) {
                    crate::panels::select::fill_selection(state);
                }
            }
            Command::GradientFill => {
                if may_edit(state) {
                    crate::panels::gradient_bar::begin_fill(state);
                }
            }
            Command::AddLayer => {
                if may_edit(state) {
                    crate::panels::layer::add_layer(state);
                }
            }
            Command::AddFrame => {
                if may_edit(state) {
                    crate::panels::frame::add_frame(state);
                }
            }
            // Ungated: guides are view state (`ViewCommand::SetGuides`), not an
            // edit of the document, and entering the shaping mode already puts
            // down whatever was composing (`modes::leave`).
            Command::AddPerspective => crate::panels::guides::add_perspective(state),
        }
    }
}

/// Raise a root-mounted dialog's flag; the dialog's own `on_close` lowers it.
fn open_dialog(mut flag: Signal<bool>) {
    flag.set(true);
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

/// Whether a **document edit** may be accepted right now.
///
/// The two questions the canvas already asks of a press, asked of every other
/// door into the document — the keyboard shortcuts and the chrome's own rows,
/// which between them were the doors that asked neither:
///
/// - **The playhead is moving.** A commit clears the withheld half of the
///   timeline, so an edit laid under a running playback deletes the rest of the
///   piece (`crate::panels::timeline`). The canvas refuses a press for this;
///   Ctrl+A went through and truncated the history from the keyboard — and the
///   menu's Deselect kept doing it after the keyboard was fixed, which is what
///   putting the gate on the act rather than the call site is for.
/// - **A mode is composing.** Its preview is computed against the committed
///   document (`crate::modes`), and the bar that carries these very commands has
///   stood down for the mode's own — deselecting mid-transform would move the
///   wrong region on "Done" (`crate::panels::select::SelectionBar`). The chrome
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

/// The chord the chrome should print for `command` — its first row in
/// [`BINDINGS`], or `None` for the many commands the keyboard cannot reach yet.
/// (When rebinding arrives, `None` is simply what an unbound command stays.)
/// "Ctrl" names the accelerator on every platform for now, as the menu always
/// has — a ⌘ on the one platform that draws it is a presentation question, not
/// a binding one.
pub fn label(command: Command) -> Option<String> {
    let (chord, _) = BINDINGS.iter().find(|&&(_, c)| c == command)?;
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
    Some(s)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn shift_is_part_of_the_chord() {
        assert_eq!(
            lookup(true, false, false, &ch("z"), "KeyZ"),
            Some(Command::Undo)
        );
        assert_eq!(
            lookup(true, true, false, &ch("Z"), "KeyZ"),
            Some(Command::Redo)
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
            Some(Command::Undo)
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
            Some(Command::BrushSmaller)
        );
        assert_eq!(
            lookup(false, false, false, &ch("]"), "BracketRight"),
            Some(Command::BrushLarger)
        );
        assert_eq!(lookup(true, false, false, &ch("["), "BracketLeft"), None);
    }

    #[test]
    fn advertised_chords() {
        // A command's first row is what the chrome prints; these strings are
        // the menu's shortcut column.
        assert_eq!(label(Command::Undo).as_deref(), Some("Ctrl+Z"));
        assert_eq!(label(Command::Redo).as_deref(), Some("Ctrl+Y"));
        assert_eq!(label(Command::Deselect).as_deref(), Some("Ctrl+D"));
        assert_eq!(
            label(Command::InvertSelection).as_deref(),
            Some("Ctrl+Shift+I")
        );
        assert_eq!(label(Command::BrushLarger).as_deref(), Some("]"));
        // And an unbound command advertises nothing rather than panicking in
        // the chrome — the shortcut column simply is not rendered.
        assert_eq!(label(Command::SaveDocument), None);
    }
}
