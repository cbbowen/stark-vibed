//! The command registry (§11, §25): every simple act the chrome can ask for,
//! declared as one variant of [`Command`] carrying its whole description.
//! §25.2 is the checklist for adding one.
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
//! The keyboard is one column of the registry, and it is the user's. The
//! shipped chords are `(Chord, Command)` rows in [`defaults`]; this browser's
//! rebindings lie over them as [`Bindings`] — a signal on the app state, stored
//! like the preset library, edited from the palette's own rows
//! (`rail::CommandSearch`: click a row's chord, press the new one). The
//! dispatch path asks [`find`] once (`input`'s keydown handler), and the chrome
//! prints its advertisements — a row's shortcut column, a tooltip's parenthesis
//! — from the same table through [`shortcut`](Command::shortcut). One
//! authority, so what the keyboard answers and what a row claims cannot drift
//! apart, however the table has been rearranged: [`Command`] is the stable name
//! a stored binding keys on, the chord is the half that is data, and `find` and
//! `shortcut` are the only two readers.
//!
//! What is deliberately *not* a chord row:
//!
//! - **Anything that owns both edges of its key.** A held digit is a grip on a
//!   brush slot with a release (§18.1.8), space is a pan for as long as it is
//!   down, Alt wears the eyedropper's cursor until it lifts. Those stay in
//!   `input`, which owns keyup.
//! - **A chord that opens a drag.** Ctrl+drag tunes the brush and Alt+drag
//!   samples — gestures rather than acts, with a press, moves and a release —
//!   so they are rows of the pointer's own table (`crate::drags`), not this
//!   one. Both tables name their modifiers by the same triple — this one's
//!   is [`Chord`]'s three flags, the pointer's is `drags::Mods` — so one
//!   keystroke cannot be read two ways by the two of them.
//! - **Ctrl+V.** A paste is data arriving, not a command: the browser delivers
//!   the clipboard *with* the event, and must be left to (`crate::images`).
//!
//! And what is not a *variant*: anything aimed at the document's own rows. A
//! control that acts on a particular layer or guide — this layer's eye, that
//! guide's trash — names a target only the document knows, and a registry of
//! every (act, target) pair would be a second copy of the panels. A payload
//! drawn from the *chrome's* own closed set is different:
//! [`TogglePanel`](Command::TogglePanel) carries a `PanelId`, and each of the
//! six is still one act [`ALL`] lists by name. The registry holds the acts the
//! search palette can offer (`rail::CommandSearch`) and a chord can carry
//! whole.
//!
//! The guard that precedes any binding at all — a keystroke in a text field is
//! the field's (`platform::KeyEvent::on_text_entry`) — is `input`'s too: by the
//! time this table is consulted, the keystroke is known to be the canvas's.

use dioxus::html::{Key, Modifiers};
use dioxus::prelude::{ReadableExt, Signal, WritableExt};
use serde::{Deserialize, Serialize};
use stark_engine::command::Tool;

use crate::icons;
use crate::input::accel;
use crate::layout::PanelId;
use crate::panels::brush::{MAX_RADIUS, MIN_RADIUS};
use crate::platform;
use crate::state::{AppState, PickScope, dispatch, update_brush};
use crate::storage::{Entry, Store};
use stark_engine::ObservableState;
use stark_engine::command::{DocCommand, ViewCommand};
use stark_model::document::SelectionOp;

/// The key half of a binding: which modifiers, and which key.
///
/// Chords are **exact** about their modifiers. Ctrl+Shift+Z is not Ctrl+Z with
/// a bystanding Shift, it is another row; Alt+H is not the bare `h` row either.
/// (The digit rack tolerates Shift, and may: it is a hold in `input`, not a row
/// here, and its reasons are its own — §18.1.8.)
///
/// **Ctrl+Alt is bindable and never shipped**, which is a deliberate pair of
/// facts rather than an oversight. On Windows AltGr *is* Ctrl+Alt: the OS
/// synthesizes the pair for the right-hand Alt, and it synthesizes the same
/// pair for a deliberate Ctrl+Alt, so on a layout that has an AltGr the two
/// keystrokes cannot be told apart. A shipped Ctrl+Alt row would therefore fire
/// under a German layout's `@` or a Polish one's `ą` — ordinary typing, on the
/// one path where a matched chord `prevent_default`s the character dead, and a
/// bug invisible to everyone whose layout has no AltGr (the plain US one has
/// none; US-International has a full set). A row the **user** put there is a
/// different thing entirely: they chose the keystroke, on their own keyboard,
/// through a capture that showed them what it was. So the rule lives on
/// [`defaults`] — `tests::no_default_chord_is_ctrl_alt` — and not on the type,
/// where it would also have cost Command+Option on a Mac, which is idiomatic
/// there and has no AltGr in it at all.
///
/// If that ever has to be tightened, the honest test is
/// `KeyboardEvent.getModifierState("AltGraph")`: plumb it through
/// [`platform::KeyEvent`] and decline the keystroke when it is set. That leaves
/// Ctrl+Alt working on the plain US layout, on Linux (where AltGr is
/// `ISO_Level3_Shift` and sets neither flag) and on a Mac, and yields to AltGr
/// where a layout has one — at the cost of a modifier state not every browser
/// need report the same way, which is why it is a note and not the code.
///
/// Serde, because a chord the user set is stored with this browser's
/// preferences and libraries — see [`Bindings`]. `alt` defaults, so a row
/// written before there was an Alt column reads as the chord it meant instead
/// of being dropped (§8's rule, for the browser's store rather than the file's:
/// a field an older record lacks has to say what its absence meant).
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct Chord {
    /// Ctrl, or Command on a Mac (`input::accel`).
    pub ctrl: bool,
    pub shift: bool,
    /// Alt, or Option on a Mac. Last of the three and defaulted for the store's
    /// sake; the same triple the drag table names its presses by
    /// (`drags::Mods`), so the two tables read one keystroke the same way.
    #[serde(default)]
    pub alt: bool,
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
///
/// A capture makes the same choice ([`capture`]): a key that types a character
/// is taken as the character, and a key that types nothing — an F-key, an
/// arrow — has only its position to be named by. **A key held through Alt is
/// in that second family**, whatever it would type unmodified: Option+G types
/// `©` on a Mac, so the character on offer is not the key's own and the
/// position is all that is left to mean.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum ChordKey {
    /// The character the key types, caseless (stored lowercase).
    Char(char),
    /// The physical position, by W3C `code` value.
    Code(String),
}

impl ChordKey {
    /// Whether a keystroke's `key`/`code` pair is this key.
    fn hit(&self, key: &Key, code: &str) -> bool {
        match self {
            ChordKey::Char(want) => matches!(key, Key::Character(c) if {
                let mut chars = c.chars();
                matches!((chars.next(), chars.next()),
                    (Some(k), None) if k.eq_ignore_ascii_case(want))
            }),
            ChordKey::Code(want) => code == want,
        }
    }
}

/// One nameable thing the chrome can ask for whole: no argument at the call
/// site, no gesture to compose. The variant, not any control or chord, is the
/// act's identity — two chords may name the same act (Ctrl+Y and Ctrl+Shift+Z),
/// three surfaces may carry it (the menu, a bar chip, the keyboard), and a
/// rebinding moves the chords while these names hold still: the stored table
/// keys on the variant's name and nothing else ([`Bindings`]).
///
/// Serde, because that name **is** the stored key — the derive spells a variant the
/// way `Debug` does, so a rebinding is stored under `"Undo"` or
/// `{"TogglePanel":"Layers"}` without anything here writing a name down twice.
/// Renaming a variant orphans its stored binding, which is dropped on load
/// (`storage::load_list`) rather than being an error.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum Command {
    Undo,
    Redo,
    /// "Select all" and "Deselect" are the same edit — a selection covering the
    /// whole canvas *is* no selection (§6.8) — so both spellings (Ctrl+A and
    /// Ctrl+D) land here, named for what the chrome calls it.
    Deselect,
    InvertSelection,
    /// Arm the rectangular marquee for the next canvas gesture (§6.8) — and
    /// **disarm it if it is already in hand**, which is the Select panel chip's
    /// rule made the act's own: the control that armed a tool is the one you
    /// reach for to take it back, and there is no "Paint" tool to switch to
    /// because no tool armed *is* the brush.
    ///
    /// Arming is the half of a shape tool that is a **simple act**, which is
    /// what puts it here (§25.1): no argument, no drag, the same thing however
    /// it is reached. The drag is a gesture and belongs to the canvas, and what
    /// the region it encloses *does* — select four ways, or fill — is the
    /// Select panel's action row: a question with five answers, not an act.
    SelectRect,
    /// Arm the elliptical marquee, on [`SelectRect`](Self::SelectRect)'s terms.
    SelectEllipse,
    /// Arm the freehand lasso, on [`SelectRect`](Self::SelectRect)'s terms.
    SelectLasso,
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
    /// its row in the visibility menu carries a check
    /// ([`VisibilityToggle::Timeline`]) rather than a trailing `…`.
    ToggleTimeline,
    /// Open the Timing Stats dialog (§7.1, `timings::TimingModal`).
    TimingStats,
    Credits,
    /// Show or hide the navigator's miniature (§11). Reachable from the
    /// visibility menu and nowhere else — the miniature has no title bar to
    /// close itself from ([`VisibilityToggle::Navigator`]).
    ToggleNavigator,
    /// Pin or unpin the quick-brush rack (§18.1.8) — the mouse-only way to a
    /// slot, which a hand with a pen and no keyboard has no other route to
    /// ([`VisibilityToggle::QuickBrushes`]).
    ToggleQuickBrushes,
    /// Show or hide one of the floating tool panels (§11). The one variant
    /// with a payload, and it keeps the module's rule because the target is
    /// the chrome's, not the document's: `PanelId` is a closed set the build
    /// enumerates, so each of the six is still a single nameable act —
    /// listed in [`ALL`], searchable, bindable to a chord of its own. The
    /// visibility menu draws its first six rows from these
    /// ([`VisibilityToggle::Panel`]).
    TogglePanel(PanelId),
    /// Open the ⚙ preferences dialog (`settings::SettingsModal`).
    Settings,
    /// Open the full brush editor off the Brush panel.
    EditBrush,
    /// Open the "Save preset" dialog for the brush in hand.
    SavePreset,
    /// How far the eyedropper's next sample sees (§18.0.2): the selected layer,
    /// it and everything beneath it, or every layer.
    ///
    /// The registry's second payload, and it keeps the module's rule for
    /// [`TogglePanel`](Self::TogglePanel)'s reason — `PickScope` is a closed set
    /// the *chrome* enumerates (`state::PickScope::ALL`), not a row of the
    /// document — so each of the three is one nameable act: listed in [`ALL`],
    /// searchable, and bound to a chord of its own. One variant rather than
    /// three because the three are one question with three answers, which is
    /// the same sentence the bar's segmented row is drawn to say.
    ///
    /// **Setting, never cycling.** A reach reached by name is a reach a chip can
    /// be lit for ([`active`](Self::active)) and a chord can land on directly,
    /// where one "next scope" key would answer differently depending on a state
    /// the hand cannot see — and the bar this lights is up for as long as Alt is
    /// held, so the hand is looking straight at the answer it would have to
    /// count.
    SetPickScope(PickScope),
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
    /// Put down what is up — one rung per press (MODAL_DESIGN.md): the open
    /// dialog, else the composing mode (committing nothing, `crate::modes`),
    /// else the frame or filter selected for composing, else Timeline. Esc's
    /// home, and the ✕ chip every mode bar wears.
    ///
    /// Deliberately **not** a rung: the selection. A selection is committed,
    /// undoable document state, not a preview — Ctrl+D already names that act,
    /// and an Esc that could destroy standing work is how a hand learns to
    /// fear the key.
    CancelMode,
    /// Commit the composing mode and leave — the mode bar's own "Done" under
    /// the one name a chord can carry (Enter, `crate::modes::finish`).
    FinishMode,
}

/// The chord table Stark ships with. **A command's first row is the one the
/// chrome advertises** ([`Bindings::of`]): Ctrl+Y above Ctrl+Shift+Z because
/// the menu has always said Ctrl+Y, and Ctrl+D above Ctrl+A because the entry
/// doing the advertising is named "Deselect". Order carries no other meaning —
/// chords are disjoint (`tests::default_chords_are_disjoint`), so no row can
/// shadow another. Most commands have no row at all: a chord is one way to
/// reach an act, not part of being one.
///
/// A function rather than a const because [`ChordKey`] owns its strings now
/// that chords are user state; what holds still is the *data*, which the user's
/// own table is laid over rather than written into ([`Bindings`]).
fn defaults() -> Vec<(Chord, Command)> {
    // No helper can say Ctrl *and* Alt, which is the one combination a shipped
    // row must never be — that pair is AltGr on half the world's layouts
    // ([`Chord`]) — so it stays unreachable from the three ways a row is
    // written. `tests::no_default_chord_is_ctrl_alt` is what holds that once a
    // fourth way exists.
    fn ch(ctrl: bool, shift: bool, key: char) -> Chord {
        Chord {
            ctrl,
            shift,
            alt: false,
            key: ChordKey::Char(key),
        }
    }
    fn code(key: &str) -> Chord {
        Chord {
            ctrl: false,
            shift: false,
            alt: false,
            key: ChordKey::Code(key.to_string()),
        }
    }
    // Alt *alone* is a column the table ships in, and spends in one place: the
    // eyedropper, whose own binding is the Alt press (`drags::defaults`), so the
    // keys that answer the bar Alt raises are held under the same modifier that
    // raised it. Spatial like every Alt chord ([`capture`]) — under Alt a key
    // does not type its own character, so the position is the only honest name.
    fn alt(key: &str) -> Chord {
        Chord {
            ctrl: false,
            shift: false,
            alt: true,
            key: ChordKey::Code(key.to_string()),
        }
    }
    vec![
        (ch(true, false, 'z'), Command::Undo),
        (ch(true, false, 'y'), Command::Redo),
        (ch(true, true, 'z'), Command::Redo),
        (ch(true, false, 'd'), Command::Deselect),
        (ch(true, false, 'a'), Command::Deselect),
        (ch(true, true, 'i'), Command::InvertSelection),
        // The three shape tools, each on the initial of its own name — Rect,
        // Ellipse, Lasso, which is the word its chip wears as well as the word
        // its palette row starts with ([`Command::word`]). A trio picked as a
        // *set* rather than borrowed letter by letter from elsewhere: R/E/L is
        // one rule a hand learns once and can then re-derive, where
        // M-for-marquee beside L-for-lasso is two facts and no rule.
        //
        // Mnemonic (`Char`), like every letter row: R arms the rectangle
        // wherever the layout puts the R. Bare, like `h` — and bare letters
        // are the table's to spend precisely because they are rebindable
        // ([`Bindings`]), which is the answer to a hand that wants them back.
        (ch(false, false, 'r'), Command::SelectRect),
        (ch(false, false, 'e'), Command::SelectEllipse),
        (ch(false, false, 'l'), Command::SelectLasso),
        (ch(false, false, 'h'), Command::MirrorView),
        (code("BracketLeft"), Command::BrushSmaller),
        (code("BracketRight"), Command::BrushLarger),
        // The eyedropper's three reaches (§18.0.2), on the left hand's own
        // column: Q over A over Z, which read *upward* is the bar's row read
        // rightward — one more layer let in at every step. One rule a hand
        // learns once rather than three letters, which is R/E/L's bargain
        // above, and it is a column the same hand can walk without leaving the
        // Alt it is already holding down.
        (alt("KeyQ"), Command::SetPickScope(PickScope::AllLayers)),
        (alt("KeyA"), Command::SetPickScope(PickScope::AndBelow)),
        (alt("KeyZ"), Command::SetPickScope(PickScope::ThisLayer)),
        // Escape cannot be *re*captured (`capture` spends it on calling a
        // capture off), so unlike every other row this one is one-way for a
        // user who rebinds it: the chord can be moved off Escape, never back
        // on. The right trade — an escape key a capture could take would be an
        // escape key that could not end one.
        (code("Escape"), Command::CancelMode),
        // Claimed only while a mode is composing ([`Command::claims`]): bare
        // Enter is the keyboard's activation of whatever control has focus,
        // and a row that ate every Enter would press no button again.
        (code("Enter"), Command::FinishMode),
    ]
}

/// The chord table as this browser has it: [`defaults`] with the user's
/// rebindings laid over them. Lives on [`AppState`](crate::state::AppState) as
/// a signal, so a row's shortcut column re-renders the moment a rebind lands;
/// stored under its own key like the preset library, since it is a table the
/// palette edits rather than a ⚙ setting (`crate::prefs`'s remit).
///
/// Only the *overrides* are state. A command the user never touched keeps its
/// default rows, minus any chord an override has claimed — so stealing Ctrl+Y
/// leaves Redo advertising Ctrl+Shift+Z with nothing stored about Redo at all,
/// and a default added in a later build shows up for a browser that stored
/// this table before it existed.
#[derive(Clone, Default, PartialEq, Debug)]
pub struct Bindings {
    /// The commands the user has taken over, each with the chord that is now
    /// its **whole** binding — a rebind replaces every default row the command
    /// had, secondary spellings included — or `None` for a command whose chord
    /// was stolen by a later rebind and has nothing left.
    overrides: Vec<(Command, Option<Chord>)>,
}

impl Bindings {
    /// Whether an override has claimed `chord` — the question that kills a
    /// default row: the user's table always wins over the shipped one.
    fn taken(&self, chord: &Chord) -> bool {
        self.overrides
            .iter()
            .any(|(_, c)| c.as_ref() == Some(chord))
    }

    /// The chord the chrome should advertise for `command`: its override, or
    /// its first default row still standing. `None` is a command the keyboard
    /// cannot reach — never bound, or rebound away.
    pub fn of(&self, command: Command) -> Option<Chord> {
        if let Some((_, chord)) = self.overrides.iter().find(|(c, _)| *c == command) {
            return chord.clone();
        }
        defaults()
            .into_iter()
            .find(|(chord, c)| *c == command && !self.taken(chord))
            .map(|(chord, _)| chord)
    }

    /// The command a keystroke asks for, if any. Overrides answer first —
    /// though they can never disagree with each other ([`rebind`](Self::rebind)
    /// steals a colliding chord) — and a default row answers only while its
    /// command is untouched and its chord unclaimed.
    fn lookup(
        &self,
        accel: bool,
        shift: bool,
        alt: bool,
        key: &Key,
        code: &str,
    ) -> Option<Command> {
        // All three modifiers, exactly — Alt included, which the shipped table
        // holds a column of (§18.0.2) and any rebinding may reach for
        // ([`Chord`]).
        let hit = |chord: &Chord| {
            chord.ctrl == accel
                && chord.shift == shift
                && chord.alt == alt
                && chord.key.hit(key, code)
        };
        if let Some((command, _)) = self
            .overrides
            .iter()
            .find(|(_, chord)| chord.as_ref().is_some_and(&hit))
        {
            return Some(*command);
        }
        defaults()
            .into_iter()
            .find(|(chord, command)| !self.overridden(*command) && !self.taken(chord) && hit(chord))
            .map(|(_, command)| command)
    }

    /// Whether the user has taken this command's binding over (even to nothing).
    fn overridden(&self, command: Command) -> bool {
        self.overrides.iter().any(|(c, _)| *c == command)
    }

    /// Give `command` the chord as its whole binding. If another rebinding
    /// held that chord it is stolen — the loser keeps an override saying so,
    /// which is what its row shows — and a default elsewhere dies by
    /// [`taken`](Self::taken) with nothing stored about it.
    pub fn rebind(&mut self, command: Command, chord: Chord) {
        for (other, held) in &mut self.overrides {
            if *other != command && held.as_ref() == Some(&chord) {
                *held = None;
            }
        }
        self.set(command, Some(chord));
    }

    /// Take `command`'s binding away — deliberately, which is why it is an
    /// override to nothing rather than a removed entry: the defaults must not
    /// resurrect a chord the user just erased.
    pub fn unbind(&mut self, command: Command) {
        self.set(command, None);
    }

    fn set(&mut self, command: Command, chord: Option<Chord>) {
        match self.overrides.iter_mut().find(|(c, _)| *c == command) {
            Some((_, held)) => *held = chord,
            None => self.overrides.push((command, chord)),
        }
    }
}

/// One stored override: the command, and the chord that is now its whole binding —
/// or none, for a command whose chord a later rebind stole.
///
/// A list of these rather than one record, so a command today's build does not know
/// costs its own row and not the table (`storage::load_list`): a binding for a retired
/// command is a binding for nothing, and the rest of the user's chords still load.
#[derive(Serialize, Deserialize)]
pub(crate) struct StoredBinding {
    command: Command,
    #[serde(default)]
    chord: Option<Chord>,
}

impl Entry for StoredBinding {
    const STORE: Store = Store::Bindings;
}

/// Seed [`AppState::bindings`](crate::state::AppState::bindings) from this
/// browser's stored rebindings. Called once at app start, beside `prefs::load`.
pub fn load(state: AppState) {
    let Some(stored) = crate::storage::load_list::<StoredBinding>() else {
        return;
    };
    let overrides = stored
        .into_iter()
        .map(|row| (row.command, row.chord))
        .collect();
    let mut bindings = state.bindings;
    bindings.set(Bindings { overrides });
}

/// Give `command` the captured chord, and persist the table — the palette's
/// commit (`rail::CommandSearch`), written through [`Bindings::rebind`] so a
/// stolen chord and its victim's row change in the same write.
pub fn rebind(state: AppState, command: Command, chord: Chord) {
    edit(state, |b| b.rebind(command, chord));
}

/// Take `command`'s binding away and persist the table — the palette's other
/// commit, Backspace where a chord would be.
pub fn unbind(state: AppState, command: Command) {
    edit(state, |b| b.unbind(command));
}

/// One change to the table, written to the signal and to storage as one act —
/// so what the rows show, what the keyboard answers, and what the next visit
/// loads cannot be three states.
fn edit(state: AppState, change: impl FnOnce(&mut Bindings)) {
    let mut bindings = state.bindings;
    let mut next = bindings.peek().clone();
    change(&mut next);
    bindings.set(next);
    let stored: Vec<StoredBinding> = bindings
        .peek()
        .overrides
        .iter()
        .map(|(command, chord)| StoredBinding {
            command: *command,
            chord: chord.clone(),
        })
        .collect();
    crate::storage::save_list(&stored);
}

/// What a keydown means to a rebinding capture (`rail::CommandSearch`): the
/// chord to commit, the capture called off, or nothing yet.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Capture {
    /// A bindable chord — commit it.
    Chord(Chord),
    /// Backspace: the binding is erased rather than replaced. The one key a
    /// capture spends on itself instead of taking as a chord — which also
    /// means no command's shortcut can *be* Backspace, a bargain worth it for
    /// an unbind that is the same gesture as clearing a field.
    Clear,
    /// Escape: the capture is called off, binding unchanged.
    Cancel,
    /// Not a chord — keep waiting. A bare modifier is a chord still being
    /// formed, and the rest are keys a binding could never answer on, refused
    /// here so the user cannot store a shortcut that silently never fires:
    ///
    /// - **space** and the **bare digits**, which are holds owning both edges
    ///   of their key (`input`, §18.1.8) and are claimed before the table;
    /// - **Ctrl+V**, which is not a binding of ours to give away — the paste
    ///   import rides the browser's own event (§23), and a chord row would
    ///   `prevent_default` it dead.
    Pending,
}

/// Read one keydown as a capture — the policy half, taken apart from the DOM
/// event the way [`Bindings::lookup`] is so a test can reach it.
pub fn capture(accel: bool, shift: bool, alt: bool, key: &Key, code: &str) -> Capture {
    if *key == Key::Escape {
        return Capture::Cancel;
    }
    if *key == Key::Backspace {
        return Capture::Clear;
    }
    if matches!(
        key,
        Key::Control | Key::Shift | Key::Alt | Key::AltGraph | Key::Meta
    ) {
        return Capture::Pending;
    }
    // Space is claimed before the table whatever is held with it — `input`'s
    // keydown arms the pan off the key itself — so it is nobody's chord, and
    // it is asked for by both names because Alt is about to take the character
    // away.
    if code == "Space" || matches!(key, Key::Character(c) if c == " ") {
        return Capture::Pending;
    }
    // The rack's own reading of the digit row, asked rather than restated —
    // and asked under the same modifiers the rack claims under, which is
    // neither: it tolerates Shift and refuses Alt (`input`'s keydown), so
    // Alt+1 is a chord going spare rather than a hold being stolen.
    if !accel && !alt && crate::slots::of_code(code).is_some() {
        return Capture::Pending;
    }
    // **Under Alt a key does not type its own character**, so there is no
    // mnemonic to name it by and only the position is left — the same sentence
    // [`ChordKey`] already writes for the F-keys and the arrows, reaching one
    // key further. Option+G types `©` on a Mac, and AltGr+A types `ą` on a
    // Polish one: naming the character would store a chord that means another
    // key on the next machine and print a label no keycap says, where the
    // position is the same key and the same engraving on both.
    if alt {
        if code.is_empty() {
            return Capture::Pending;
        }
        return Capture::Chord(Chord {
            ctrl: accel,
            shift,
            alt,
            key: ChordKey::Code(code.to_string()),
        });
    }
    match key {
        Key::Character(c) => {
            let mut chars = c.chars();
            let (Some(k), None) = (chars.next(), chars.next()) else {
                return Capture::Pending;
            };
            if accel && !shift && k.eq_ignore_ascii_case(&'v') {
                return Capture::Pending;
            }
            Capture::Chord(Chord {
                ctrl: accel,
                shift,
                alt,
                key: ChordKey::Char(k.to_ascii_lowercase()),
            })
        }
        _ if !code.is_empty() => Capture::Chord(Chord {
            ctrl: accel,
            shift,
            alt,
            key: ChordKey::Code(code.to_string()),
        }),
        _ => Capture::Pending,
    }
}

/// Every command, in the order the search palette lists ties: the file family
/// first (the resting offer, [`BASIC`]), then the acts on the document and the
/// view, the openers, the toggles, and the keyboard-only steps last.
///
/// **By hand, and checked**. The order is presentational and belongs to a human;
/// the *completeness* does not, and used to be left to review — a variant left
/// out compiled clean and was simply unfindable in the palette.
/// `tests::all_lists_every_command` is what ended that: it counts the enum's
/// variants at compile time, so a command added without a row here fails the
/// build with the row it is missing. `tests::names_are_unique` runs over the
/// list too, so a duplicate display name cannot hide in it either.
///
/// What is deliberately still by hand is everything *about* a command — its
/// word, mark, hint and aliases are `match` arms rather than columns of a table
/// here, because half those arms carry a sentence saying why that one is spelled
/// the way it is ("a search result stands alone", "the menu's panel half is a
/// picture of the stack"), and a table would have nowhere to put them. The list
/// is data; the descriptions are an argument.
pub const ALL: &[Command] = &[
    Command::NewDocument,
    Command::OpenDocument,
    Command::SaveDocument,
    Command::ImportImage,
    Command::ExportImage,
    Command::Share,
    Command::Undo,
    Command::Redo,
    // Drawing a region, then the acts on the region you drew.
    Command::SelectRect,
    Command::SelectEllipse,
    Command::SelectLasso,
    Command::Deselect,
    Command::InvertSelection,
    Command::Transform,
    Command::FillSelection,
    Command::GradientFill,
    Command::AddLayer,
    Command::AddFrame,
    Command::AddPerspective,
    Command::EditBrush,
    Command::SavePreset,
    // The eyedropper's three reaches, beside the brush's own two: what a sample
    // does is load the brush, so this is the same family reached from the canvas
    // instead of from the panel (§18.0.2).
    Command::SetPickScope(PickScope::ThisLayer),
    Command::SetPickScope(PickScope::AndBelow),
    Command::SetPickScope(PickScope::AllLayers),
    Command::ToggleTimeline,
    Command::TogglePanel(PanelId::Color),
    Command::TogglePanel(PanelId::Brush),
    Command::TogglePanel(PanelId::Select),
    Command::TogglePanel(PanelId::Layers),
    Command::TogglePanel(PanelId::Guides),
    Command::TogglePanel(PanelId::Lighting),
    Command::ToggleNavigator,
    Command::ToggleQuickBrushes,
    Command::TimingStats,
    Command::Credits,
    Command::Settings,
    Command::MirrorView,
    Command::BrushSmaller,
    Command::BrushLarger,
    Command::CancelMode,
    Command::FinishMode,
];

/// The search palette's resting offer, shown before a first keystroke: the
/// file family — the commands about the document as a thing in the world.
/// They are the ones reached for rarely enough to have no muscle-memory home
/// (a brush has a panel, undo has a chord; "Export image…" has only this), so
/// they are the ones worth showing to a hand that arrived without a word in
/// mind.
pub const BASIC: &[Command] = &[
    Command::NewDocument,
    Command::OpenDocument,
    Command::SaveDocument,
    Command::ImportImage,
    Command::ExportImage,
    Command::Share,
];

/// One entry of the rail's **visibility menu** (§11): a thing the window shows
/// or hides, whether or not it is a panel.
///
/// The menu began as the panel stack's own — one row per [`PanelId`], each
/// wearing the mark its title bar wears, so the list reads as a picture of the
/// stack rather than a column of its names. Then it took in the chrome that
/// stands *outside* the stack: the navigator's miniature, the quick-brush rack,
/// and Timeline mode. None of those has a title bar to close itself from, so
/// for each of them this menu is the only way there and back — and the menu was
/// never really a list of the panels anyway. It is the map of what is on screen.
///
/// Which is what this enum is for. They arrive one at a time, and the first two
/// came in as rows appended by hand, each with its focus index counted off
/// `PanelId::ALL.len()`: bookkeeping the loop beside them was already doing,
/// restated where nothing would catch it going wrong, and a third addition would
/// have been a third copy of the arithmetic. [`ALL`](Self::ALL) is the map
/// written down instead, and the menu is one loop over it
/// (`rail::VisibilityMenu`).
///
/// Deliberately thin. An entry's word, mark, lit state, greyed state and act are
/// all the registry's — it knows only *which* command it is
/// ([`command`](Self::command)) — so this is a **view** of the registry rather
/// than a second one, and nothing reachable here is a thing a search for its
/// name would miss.
/// Serde, because an entry is named in the record of what this browser last had on
/// screen (`crate::visibility`) — and the derive spells a variant exactly as `Debug`
/// does, so the stored word and the enum's are one by construction.
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub enum VisibilityToggle {
    /// One of the floating tool panels (§11).
    Panel(PanelId),
    /// The navigator's miniature (§11). First of the three that are not panels,
    /// and the one most like them: a standing readout, kept between visits.
    Navigator,
    /// The quick-brush rack (§18.1.8). While a number is held the rack appears
    /// whatever this entry says; what the entry buys is a rack that is
    /// *clickable* — the mouse-only way to a slot, which a hand with a pen and
    /// no keyboard has no other route to.
    QuickBrushes,
    /// Timeline mode (§18.2.4): the scrubber and transport across the foot of
    /// the window.
    ///
    /// The one entry that is a **mode** rather than a piece of furniture, and it
    /// belongs here all the same, for the reason the other two do: this menu
    /// answers *what is on screen*, and the timeline is the largest thing that
    /// can be. Until it had a row the mode was reachable by name in the palette
    /// and nowhere else — no chord, no chip, nothing in the chrome that says the
    /// history can be walked at all.
    ///
    /// Being a mode costs it nothing here and gains it one thing no other entry
    /// has: Esc leaves it, as the last rung of the ladder.
    Timeline,
}

impl VisibilityToggle {
    /// Every entry, in menu order: the panel stack first, in the order it stacks
    /// ([`PanelId::ALL`]), then the three that are not panels — last, and in
    /// increasing order of how much of the window they take.
    ///
    /// The panel half is written out rather than folded in from `PanelId::ALL`,
    /// so this list is the one a reader has to trust and it says what it holds;
    /// `tests::the_visibility_menu_is_the_stack_plus_three` is what keeps the two
    /// in step when a panel is added.
    pub const ALL: [VisibilityToggle; 9] = [
        VisibilityToggle::Panel(PanelId::Color),
        VisibilityToggle::Panel(PanelId::Brush),
        VisibilityToggle::Panel(PanelId::Select),
        VisibilityToggle::Panel(PanelId::Layers),
        VisibilityToggle::Panel(PanelId::Guides),
        VisibilityToggle::Panel(PanelId::Lighting),
        VisibilityToggle::Navigator,
        VisibilityToggle::QuickBrushes,
        VisibilityToggle::Timeline,
    ];

    /// The act the entry's row runs — the whole of what the row *is*, since the
    /// menu draws every other part of it from here too (`rail::CmdItem`).
    pub fn command(self) -> Command {
        match self {
            VisibilityToggle::Panel(id) => Command::TogglePanel(id),
            VisibilityToggle::Navigator => Command::ToggleNavigator,
            VisibilityToggle::QuickBrushes => Command::ToggleQuickBrushes,
            VisibilityToggle::Timeline => Command::ToggleTimeline,
        }
    }
}

/// The commands `query` asks for, in the order the palette shows them: what
/// the query begins before what it merely appears in — both caseless — and
/// [`BASIC`] for no query at all, so an opened palette is never an empty box.
/// Within each tier the display name answers before an alias
/// ([`aliases`](Command::aliases)) does: the name is the match the row can
/// show, the alias the one it deliberately cannot.
pub fn search(query: &str) -> Vec<Command> {
    let q = query.trim().to_lowercase();
    if q.is_empty() {
        return BASIC.to_vec();
    }
    // Four tiers: name-prefix, alias-prefix, name-substring, alias-substring.
    let mut tiers: [Vec<Command>; 4] = Default::default();
    for &command in ALL {
        let name = command.name().to_lowercase();
        let aliases: Vec<String> = command.aliases().iter().map(|a| a.to_lowercase()).collect();
        let tier = if name.starts_with(&q) {
            0
        } else if aliases.iter().any(|a| a.starts_with(&q)) {
            1
        } else if name.contains(&q) {
            2
        } else if aliases.iter().any(|a| a.contains(&q)) {
            3
        } else {
            continue;
        };
        tiers[tier].push(command);
    }
    tiers.concat()
}

/// The command `e` asks for, if any — the one reader on the dispatch path,
/// asking this browser's own table ([`Bindings`]). `peek`: a keydown is no
/// reason for anything to re-render.
pub fn find(state: AppState, e: &platform::KeyEvent) -> Option<Command> {
    let m = e.modifiers();
    state
        .bindings
        .peek()
        .lookup(
            accel(m),
            m.contains(Modifiers::SHIFT),
            m.contains(Modifiers::ALT),
            &e.key(),
            &e.code(),
        )
        // A row that matches may still decline the *keystroke* — today only
        // FinishMode's bare Enter ([`Command::claims`]). Filtered here rather
        // than in `run`, because the caller `prevent_default`s whatever this
        // answers, and a claim is the one thing a declined Enter must not make.
        .filter(|command| command.claims(state))
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
            // Named for the shape and then the act, so the three sort together
            // wherever names are listed and a query for "select" lists the
            // family. The panel says the shape alone ([`word`](Self::word)) —
            // its header is the word "Selection" already.
            Command::SelectRect => "Rectangle select",
            Command::SelectEllipse => "Ellipse select",
            Command::SelectLasso => "Lasso select",
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
            Command::CancelMode => "Cancel mode",
            Command::FinishMode => "Finish mode",
            // Named for the act and then the reach, so the three sort
            // together wherever names are listed and a query for "pick" lists
            // the family — the shape tools' rule, applied to the other trio
            // that answers one question. The bar says the reach alone
            // ([`word`](Self::word)); it is headed "Eyedropper" already.
            Command::SetPickScope(scope) => match scope {
                PickScope::ThisLayer => "Pick from this layer",
                PickScope::AndBelow => "Pick from layers below",
                PickScope::AllLayers => "Pick from all layers",
            },
            // Named after the panel with "panel" said out loud, because a
            // search result stands alone: a row reading just "Color" would
            // claim the subject rather than the box that holds its controls.
            // The menu under the Panels trigger says the terse half instead
            // ([`word`](Self::word)), and a test pins each name to its
            // panel's own title (`tests::every_panel_has_a_toggle_row`).
            Command::TogglePanel(id) => match id {
                PanelId::Color => "Color panel",
                PanelId::Brush => "Brush panel",
                PanelId::Select => "Select panel",
                PanelId::Layers => "Layers panel",
                PanelId::Guides => "Drawing guides panel",
                PanelId::Lighting => "Lighting panel",
            },
        }
    }

    /// The terse word a chip wears where its bar or header already names the
    /// subject: "Invert" on the selection bar, "Rect" in the Select panel's
    /// tool row, "Layer" beside the stack's other two adds. Everything else
    /// keeps its [`name`](Self::name) — the split exists so both spellings are
    /// the registry's, not a call site's, and the abbreviation a control needs
    /// is never a reason for it to stop rendering the command (§25.2's step 7).
    ///
    /// The width argument is the same one every run of chips makes (§25.9): a
    /// panel column is 280px, and "Rectangle select" over a glyph in a third of
    /// it is not a chip. What is *not* a reason to abbreviate is a control
    /// wanting a word of its own — a place that would say something the name
    /// does not say is a place naming a different act.
    pub fn word(self) -> &'static str {
        match self {
            Command::InvertSelection => "Invert",
            // The shape alone: the panel these three sit in is headed
            // "Selection", and the bar their gesture raises is too.
            Command::SelectRect => "Rect",
            Command::SelectEllipse => "Ellipse",
            Command::SelectLasso => "Lasso",
            Command::FillSelection => "Fill",
            Command::GradientFill => "Gradient",
            Command::AddLayer => "Layer",
            Command::AddFrame => "Frame",
            Command::AddPerspective => "Perspective",
            // On a mode bar the mode is already named by the label; the chip
            // says the act alone, as its Done neighbour always has.
            Command::CancelMode => "Cancel",
            Command::FinishMode => "Done",
            // The reach alone: the bar these three sit on is headed with the
            // eyedropper's mark and name, so the verb is already said — and a
            // 3-chip row beside a Group toggle and four patch sizes has no
            // width for it twice. The plus is where "and below" survives the
            // shortening: it is the one thing the long name leaves out, that
            // the selected layer is in the sample too.
            Command::SetPickScope(scope) => match scope {
                PickScope::ThisLayer => "This layer",
                PickScope::AndBelow => "+ Below",
                PickScope::AllLayers => "All layers",
            },
            // The visibility menu's panel half is a picture of the stack, so
            // its rows wear the panels' own title-bar labels.
            Command::TogglePanel(id) => id.title(),
            _ => self.name(),
        }
    }

    /// The command's other names: what different software calls the same act,
    /// searched by [`search`] but never printed. A hand that arrives typing
    /// "flip" finds a row named "Mirror view" — the alias does the finding,
    /// the name does the teaching — so this list can borrow another tool's
    /// vocabulary freely without two rows ever reading alike
    /// (`tests::no_alias_shadows_a_name` holds it to that).
    pub fn aliases(self) -> &'static [&'static str] {
        match self {
            // Ctrl+A lands on Deselect because covering everything *is*
            // selecting nothing (§6.8) — both of that key's names belong here.
            Command::Deselect => &["Select all", "Select none"],
            Command::InvertSelection => &["Select inverse"],
            // Bare "marquee" lands on the rectangle, which is what it means in
            // the software that spells it that way.
            Command::SelectRect => &["Marquee", "Rectangular marquee", "Box select"],
            Command::SelectEllipse => &["Elliptical marquee", "Circle select", "Oval select"],
            // Not "Polygon select": that is a different tool in the software
            // that has it, and an alias is for another word for *this* act.
            Command::SelectLasso => &["Freehand select", "Free select"],
            Command::MirrorView => &["Flip horizontal"],
            Command::BrushSmaller => &["Decrease brush size"],
            Command::BrushLarger => &["Increase brush size"],
            Command::NewDocument => &["New canvas", "New file"],
            // A paste is one of the import's own doors (§23).
            Command::ImportImage => &["Place image", "Paste"],
            Command::ExportImage => &["Save as"],
            Command::Share => &["Collaborate"],
            // The timeline is the undo history made scrubbable (§18.2.4), and
            // its replay is what other tools sell as a timelapse.
            Command::ToggleTimeline => &["History", "Timelapse"],
            Command::ToggleNavigator => &["Minimap"],
            Command::Settings => &["Preferences", "Options"],
            Command::Credits => &["About"],
            Command::EditBrush => &["Brush settings", "Brush studio"],
            Command::Transform => &["Free transform"],
            Command::FillSelection => &["Paint bucket"],
            Command::AddLayer => &["New layer"],
            // The frame is where cropping went (§15.7): it marks the piece's
            // edge without deleting what lies past it.
            Command::AddFrame => &["Crop", "Canvas size"],
            Command::AddPerspective => &["Drawing guide", "Vanishing point"],
            // The words the same three reaches wear elsewhere — Photoshop's
            // Sample menu, and the "merged" that means all-layers in most of
            // the software that has it. "Eyedropper" is on all three because
            // the tool itself has no row to find: it is a held modifier rather
            // than a command (§18.0.2), so a hand typing its name should still
            // be shown what there is of it here.
            Command::SetPickScope(PickScope::ThisLayer) => {
                &["Eyedropper", "Sample current layer", "Sample this layer"]
            }
            Command::SetPickScope(PickScope::AndBelow) => {
                &["Eyedropper", "Sample current and below", "Sample below"]
            }
            Command::SetPickScope(PickScope::AllLayers) => {
                &["Eyedropper", "Sample merged", "Sample all layers"]
            }
            Command::CancelMode => &["Escape", "Abandon", "Leave mode", "Close dialog"],
            Command::FinishMode => &["Done", "Apply", "Commit"],
            _ => &[],
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
            // The one family where the glyph *is* the meaning rather than the
            // control's (`icons`): a tool that draws a rectangle is marked
            // with a rectangle.
            Command::SelectRect => icons::RECTANGLE,
            Command::SelectEllipse => icons::CIRCLE,
            Command::SelectLasso => icons::LASSO,
            Command::MirrorView => icons::MIRROR_VIEW,
            Command::BrushSmaller | Command::BrushLarger => icons::SIZE,
            Command::NewDocument => icons::NEW_DOCUMENT,
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
            // The dismissal mark every panel header wears, and the tick every
            // Done chip does — the two acts these commands are the names of.
            Command::CancelMode => icons::CLOSE,
            Command::FinishMode => icons::DONE,
            // The bar's own three marks, which are a picture of the question:
            // one sheet, a sheet over what is under it, a stack.
            Command::SetPickScope(scope) => match scope {
                PickScope::ThisLayer => icons::ONE_LAYER,
                PickScope::AndBelow => icons::AND_BELOW,
                PickScope::AllLayers => icons::ALL_LAYERS,
            },
            // The mark its own title bar wears, so the menu and the palette
            // both stay a picture of the stack.
            Command::TogglePanel(id) => id.glyph(),
        }
    }

    /// The sentence a tooltip leads with — what the act will do, not how it is
    /// reached ([`tooltip`](Self::tooltip) appends that half). Falls back to
    /// the name where the name already is the sentence.
    pub fn hint(self) -> &'static str {
        match self {
            // What the gesture encloses is a *region*, not yet a selection:
            // the panel's action row decides where that coverage lands, and it
            // may be paint (§6.8). The clause each of these ends on is the
            // app's least guessable rule, and it is stated where it is true —
            // a fill leaves the tool armed, so only the selecting case is
            // claimed here (`Session::end_shape`).
            Command::SelectRect => {
                "Drag out a rectangular region \u{2014} selecting it hands the brush straight back"
            }
            Command::SelectEllipse => {
                "Drag out an elliptical region \u{2014} selecting it hands the brush straight back"
            }
            Command::SelectLasso => {
                "Draw a freehand region \u{2014} selecting it hands the brush straight back"
            }
            // What each reach *sees*, said as the sample sees it — and the
            // sentence a hand holding Alt reads off the bar, with the chord
            // that reaches it in the parenthesis ([`tooltip`](Self::tooltip)),
            // which is the whole of how a modifier binding is learned (§24).
            Command::SetPickScope(scope) => match scope {
                PickScope::ThisLayer => {
                    "Sample the selected layer alone, ignoring anything over or under it"
                }
                PickScope::AndBelow => {
                    "Sample the selected layer and everything beneath it \u{2014} what the \
                     canvas would show with the layers above switched off"
                }
                PickScope::AllLayers => {
                    "Sample every visible layer \u{2014} the color the canvas shows"
                }
            },
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
            Command::CancelMode => {
                "Put down what's in progress, keeping nothing \u{2014} close the \
                 dialog, or cancel the composing mode, or stop composing the \
                 frame or filter, or leave Timeline"
            }
            Command::FinishMode => {
                "Commit what the composing mode has made \u{2014} its bar's own Done"
            }
            _ => self.name(),
        }
    }

    /// The `title` a control rendering this command carries: the hint, plus the
    /// advertised shortcut for the commands the keyboard can reach.
    pub fn tooltip(self, bindings: &Bindings) -> String {
        match self.shortcut(bindings) {
            Some(chord) => format!("{} ({chord})", self.hint()),
            None => self.hint().to_string(),
        }
    }

    /// The shortcut the chrome prints beside this command's name — its
    /// advertised chord in this browser's table ([`Bindings::of`]), or `None`
    /// for a command the keyboard cannot reach: never bound, or rebound away.
    ///
    /// One command's advertisement is written by hand instead: Import's Ctrl+V
    /// is a paste, not a binding of ours — the browser delivers the clipboard
    /// *with* the event (`crate::images`) — so there is no row for it, and yet
    /// the shortcut is true and worth saying wherever the command is shown.
    pub fn shortcut(self, bindings: &Bindings) -> Option<String> {
        match self {
            Command::ImportImage => Some("Ctrl+V".to_string()),
            _ => bindings.of(self).map(|chord| chord_label(&chord)),
        }
    }

    /// Whether the palette offers to change this command's shortcut. `false`
    /// for exactly the advertisement that is not ours to move: Import's Ctrl+V
    /// is the browser's paste, true whatever this table says, so a chip that
    /// offered to change it would be offering a lie.
    pub fn rebindable(self) -> bool {
        !matches!(self, Command::ImportImage)
    }

    /// Whether this command's act is live right now.
    ///
    /// Three surfaces read it, and they are two pictures of one fact: the select
    /// blue on the mark, which a menu row and a palette row draw alike, and a
    /// [`CommandButton`](crate::widgets::CommandButton)'s lit chip. That last
    /// one is why the shape tools answer here rather than the Select panel
    /// computing `tool == t` for its own chips — a chip lit by the registry
    /// cannot disagree with the chord that lit it.
    pub fn active(self, state: AppState) -> Option<bool> {
        match self {
            Command::SelectRect => Some(armed(state, Tool::SelectRect)),
            Command::SelectEllipse => Some(armed(state, Tool::SelectEllipse)),
            Command::SelectLasso => Some(armed(state, Tool::SelectLasso)),
            Command::ToggleTimeline => Some(*state.timeline.open.read()),
            Command::ToggleNavigator => Some(*state.navigator.read()),
            Command::ToggleQuickBrushes => Some(*state.slots.pinned.read()),
            // Exactly one of the three is lit, always, which is the claim that
            // the row is one question rather than three switches — and it is
            // read here so a chord pressed under the bar moves the light the
            // chip would have moved.
            Command::SetPickScope(scope) => Some(*state.pick.scope.read() == scope),
            Command::TogglePanel(id) => Some(!state.panels.hidden.read().contains(&id)),
            Command::Share => {
                Some(*state.collab.phase.read() == crate::collab::CollabPhase::Shared)
            }
            _ => None,
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
            Command::SelectRect => arm_tool(state, Tool::SelectRect),
            Command::SelectEllipse => arm_tool(state, Tool::SelectEllipse),
            Command::SelectLasso => arm_tool(state, Tool::SelectLasso),
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
            // Through `slots::set_pinned` rather than writing the signal, for the
            // reason the two above go through theirs: the rack's visibility is
            // remembered, and the one writer is what makes that structural
            // (`crate::visibility`).
            Command::ToggleQuickBrushes => {
                let pinned = *state.slots.pinned.peek();
                crate::slots::set_pinned(state, !pinned);
            }
            // Ungated like the other toggles: which panels are up is chrome,
            // not document. The two halves an entry must not forget — waking a
            // sleeping stack on open, telling the tour on close — live in
            // `layout`'s own functions, which is why this goes through
            // `toggle_panel` rather than writing `hidden`.
            Command::TogglePanel(id) => {
                crate::layout::toggle_panel(state, state.panels, id);
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
            // Ungated, with the view and brush acts: how far a sample reaches is
            // an argument to a *request* (`Engine::pick_color`), read at the
            // moment of the sample and committing nothing — and the bar's own
            // chips have never been refused mid-playback either. The gate that
            // matters is the sample's, and it is the drag table's
            // (`DragAction::claims`).
            Command::SetPickScope(scope) => {
                let mut want = state.pick.scope;
                want.set(scope);
            }
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
            // Half of [`may_edit`], and the halves are asked separately on purpose.
            // Adding a guide *is* a document edit now (§20.5), so it is refused
            // while the timeline is playing back, like every other one: what is on
            // screen then is a historical state, and editing it would be editing
            // the wrong document. The composing half is deliberately not asked —
            // this command puts down whatever was composing itself
            // (`modes::leave`), so it replaces a mode rather than being refused by
            // one, which is the behaviour it has always had.
            Command::AddPerspective => {
                if !crate::panels::timeline::is_playing(state) {
                    crate::panels::guides::add_perspective(state);
                }
            }
            Command::CancelMode => escape(state),
            // Gated on the dialogs where CancelMode ladders through them:
            // Enter under a dialog belongs to the dialog's form, and a commit
            // it could not see landing beneath it would be the worse surprise.
            Command::FinishMode => {
                if !dialog_open(state) {
                    crate::modes::finish(state);
                }
            }
        }
    }

    /// Whether this command claims its keystroke right now — asked by [`find`]
    /// **before** the caller's `prevent_default`, where [`run`](Self::run)'s
    /// own gates decide only what happens after the claim.
    ///
    /// `true` for almost everything: a declined act still claims its chord,
    /// because the browser's default would answer it with something worse (see
    /// `input`'s keydown handler). The exception is bare **Enter**, which is
    /// the keyboard's activation of whatever control has focus — a Done that
    /// claimed it unconditionally would eat every focused button and dialog
    /// form in the app, so FinishMode claims it only while there is a mode for
    /// it to finish and no dialog over that mode. Esc has no such double life:
    /// outside text entry (already carved out before the table is consulted),
    /// the browser does nothing with it worth keeping.
    fn claims(self, state: AppState) -> bool {
        match self {
            Command::FinishMode => crate::modes::is_composing(state) && !dialog_open(state),
            _ => true,
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
        b.size = (b.size * factor).clamp(MIN_RADIUS, MAX_RADIUS);
    });
}

/// Arm `tool` for the next canvas gesture — or hand the brush back, if it is
/// the tool already in hand (§6.8). One act rather than two, because the chip
/// and the chord must mean the same thing on a second press: the control that
/// armed a tool is the one that takes it back, and R pressed twice is the
/// keyboard reaching that same control twice.
///
/// **Ungated**, with the view and brush acts: arming commits nothing to the
/// document — `SetTool` is a `ViewCommand` — and the panel's chips have never
/// been refused mid-playback or under a composing mode either. The *gesture*
/// an armed tool then makes is a different act with a gate of its own, which
/// the canvas has always asked (`crate::input`).
fn arm_tool(state: AppState, tool: Tool) {
    let already = crate::panels::select::current_tool(state) == tool;
    let next = if already { Tool::Brush } else { tool };
    // Which of the three was last in hand, kept where the session cannot keep it:
    // a selecting gesture disarms to `Tool::Brush` (§6.8), so the engine's own
    // `tool` has forgotten which marquee drew by the time anything asks. Recorded
    // here because this is the one door into arming — the chip, the chord and the
    // palette are all this call — and read back by [`arm_shape_tool`].
    if next.is_selection() {
        let mut last = state.shape_tool;
        last.set(next);
    }
    dispatch(state, ViewCommand::SetTool(next));
}

/// Hand back a shape tool without naming one — the last one armed
/// ([`AppState::shape_tool`]), and nothing at all if one is already in hand.
///
/// The Select panel's action row is what asks (`crate::panels::select`): picking
/// what a shape *does* is a statement about a gesture that has not been made yet,
/// and with the brush in hand there is nothing for it to be a statement about.
/// Which of the three would draw it is a question the row does not answer, so it
/// takes the answer the chips above left behind.
///
/// Leaving an armed tool alone is the same rule read the other way: the row says
/// nothing about which of the three, so a lasso stays a lasso.
pub fn arm_shape_tool(state: AppState) {
    if !crate::panels::select::current_tool(state).is_selection() {
        // Read out first, never `*state.shape_tool.peek()` in the argument: a
        // signal's read guard lives to the end of the *statement*, and the call it
        // would be an argument to writes that same signal — which is a panic, taken
        // in a handler that has already dispatched half of what it came to do.
        let last = *state.shape_tool.peek();
        arm_tool(state, last);
    }
}

/// Whether `tool` is the one the next gesture would use — reactively (`read`,
/// not `peek`), because this is the answer a lit chip is mounted on.
fn armed(state: AppState, tool: Tool) -> bool {
    state.obs.read().as_ref().is_some_and(|o| o.tool == tool)
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
///   document (`crate::modes`), and the bar that carries these very commands
///   stands recessed and inert behind the mode's own — deselecting
///   mid-transform would move the wrong region on "Done"
///   (`crate::panels::select::SelectionBar`). The chrome says what the screen
///   says — and this gate is also what lets a recessed bar keep its chips
///   mounted: a click that somehow reached one would be refused here.
fn may_edit(state: AppState) -> bool {
    !crate::panels::timeline::is_playing(state) && !crate::modes::is_composing(state)
}

/// Esc's ladder (MODAL_DESIGN.md), one rung per press: the open dialog, else
/// the composing mode, else the frame or filter layer selected for composing,
/// else Timeline. Ordered outermost-in — a dialog stands over a mode, a mode
/// over the bar that raised it, the bars over the timeline's — so each press
/// peels the layer the eye reads as topmost, and never two at once: Esc from
/// a gradient matte drops the axis first and leaves the frame second.
///
/// The dialogs are closed *here*, not declined in deference to their own Esc
/// handlers, because outside a text field they have none: every element-level
/// Escape in the app lives on an input (the palette's field, the rename and
/// name drafts), where the window's keydown binding is already withheld
/// (`platform::KeyEvent::on_text_entry`). One actor per keystroke, so the
/// dioxus-vs-window handler ordering that a second actor would hang on never
/// gets asked.
fn escape(state: AppState) {
    // A pop-out flown out of a bar stands over everything below, so it is the
    // first thing Escape puts down (`widgets::PopoutId`, §25.7). Above the
    // dialogs because it is the innermost surface, and kept off the dialog list
    // for a reason of its own: that list is what stands `FinishMode` down, and a
    // library opened from a composing bar must not take Enter's "Done" away.
    if crate::widgets::close_popout(state) {
        return;
    }
    if close_dialogs(state) {
        return;
    }
    if crate::modes::is_composing(state) {
        crate::modes::cancel(state);
        return;
    }
    // The two layer kinds that are composed rather than painted (§15.7,
    // §21.6): Esc is their bars' own Done — the topmost paint layer selected
    // instead, the only way a frame or filter is ever "deselected". The guide
    // bar's bargain: nothing is uncommitted, so leaving is the whole act.
    // Enter deliberately does *not* reach here — these are standing states,
    // and an Enter claimed through one would eat every focused button's
    // Enter for as long as the layer stayed selected ([`Command::claims`]).
    if composing_layer_selected(state) {
        crate::panels::frame::done_composing(state);
        return;
    }
    if *state.timeline.open.peek() {
        crate::panels::timeline::set_open(state, false);
    }
}

/// Whether the selected layer is a frame or a filter — the kinds whose bar is
/// up for as long as they are selected, asked handler-time (`peek`).
fn composing_layer_selected(state: AppState) -> bool {
    state
        .obs
        .peek()
        .as_ref()
        .is_some_and(|o| crate::panels::frame::selected_frame_of(o).is_some())
        || crate::panels::filter::selected_filter(state).is_some()
}

/// Whether any root-mounted dialog is up — Esc's first rung, and the fact that
/// stands FinishMode down ([`Command::claims`]).
fn dialog_open(state: AppState) -> bool {
    state.root_dialogs().iter().any(|flag| *flag.peek())
}

/// Lower whichever root dialogs are up; `true` if any was. Lowering the flag
/// *is* the dialog's own close — every `on_close` in `main` does nothing else
/// (`AppState::root_dialogs`).
fn close_dialogs(state: AppState) -> bool {
    let mut any = false;
    for mut flag in state.root_dialogs() {
        if *flag.peek() {
            flag.set(false);
            any = true;
        }
    }
    any
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

/// `hint` with `command`'s advertised chord appended, the way
/// [`Command::tooltip`] spells its own — for a control whose words are its
/// mode's (a bar's Done says what *this* Done lays) but whose key is the
/// registry's (Enter, via [`Command::FinishMode`]).
pub fn advertised(hint: &str, command: Command, bindings: &Bindings) -> String {
    match command.shortcut(bindings) {
        Some(chord) => format!("{hint} ({chord})"),
        None => hint.to_string(),
    }
}

/// A chord spelled out for the chrome. Private: the chrome asks
/// [`Command::shortcut`], which is this over [`Bindings::of`], plus the one
/// advertisement that has no chord.
/// "Ctrl" names the accelerator on every platform for now, as the menu always
/// has — a ⌘ on the one platform that draws it is a presentation question, not
/// a binding one.
fn chord_label(chord: &Chord) -> String {
    // Ctrl, Shift, Alt — the order `drags::chord_label` spells its own presses
    // in, so one keystroke does not read two ways in one app depending on
    // which table is describing it.
    let mut s = String::new();
    for (held, name) in [
        (chord.ctrl, "Ctrl+"),
        (chord.shift, "Shift+"),
        (chord.alt, "Alt+"),
    ] {
        if held {
            s.push_str(name);
        }
    }
    match &chord.key {
        ChordKey::Char(c) => s.push(c.to_ascii_uppercase()),
        // A position's label is its US engraving: what the keycap most often
        // says, and the only honest answer short of asking the layout — which
        // the web cannot portably do. Rebinding lets the label be wrong less
        // often than a fixed binding already was.
        ChordKey::Code(code) => match code.as_str() {
            // The punctuation ring, which every Alt chord on those keys now
            // lands in: a captured Alt binding is spatial whatever key it is
            // on ([`capture`]), so a row reading "Alt+Comma" stopped being a
            // corner case the day the tier opened. Named where the W3C code is
            // a word for a mark rather than the mark.
            "BracketLeft" => s.push('['),
            "BracketRight" => s.push(']'),
            "Comma" => s.push(','),
            "Period" => s.push('.'),
            "Slash" => s.push('/'),
            "Backslash" => s.push('\\'),
            "Semicolon" => s.push(';'),
            "Quote" => s.push('\''),
            "Backquote" => s.push('`'),
            "Minus" => s.push('-'),
            "Equal" => s.push('='),
            // "KeyQ" is engraved "Q", and the digit row's caps are their digits
            // — the two families a captured spatial chord usually lands in.
            code => match code
                .strip_prefix("Key")
                .or_else(|| code.strip_prefix("Digit"))
            {
                Some(cap) => s.push_str(cap),
                None => s.push_str(code),
            },
        },
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ch(s: &str) -> Key {
        Key::Character(s.into())
    }

    /// The stock table — what every browser has before its first rebind.
    fn stock() -> Bindings {
        Bindings::default()
    }

    #[test]
    fn default_chords_are_disjoint() {
        let rows = defaults();
        for (i, (a, _)) in rows.iter().enumerate() {
            for (b, _) in &rows[i + 1..] {
                assert_ne!(a, b, "two rows on one chord: the table is not a function");
            }
        }
    }

    #[test]
    fn shift_is_part_of_the_chord() {
        assert_eq!(
            stock().lookup(true, false, false, &ch("z"), "KeyZ"),
            Some(Command::Undo)
        );
        assert_eq!(
            stock().lookup(true, true, false, &ch("Z"), "KeyZ"),
            Some(Command::Redo)
        );
        // Ctrl+Shift+Y is nobody's: an unclaimed chord falls through to the
        // browser rather than being Ctrl+Y plus a bystander.
        assert_eq!(stock().lookup(true, true, false, &ch("Y"), "KeyY"), None);
    }

    #[test]
    fn escape_and_enter_are_rows() {
        // The two bare-key rows (MODAL_DESIGN.md). Spatial (`code`), so a
        // layout that types something exotic on them changes nothing.
        assert_eq!(
            stock().lookup(false, false, false, &Key::Escape, "Escape"),
            Some(Command::CancelMode)
        );
        assert_eq!(
            stock().lookup(false, false, false, &Key::Enter, "Enter"),
            Some(Command::FinishMode)
        );
        // With a modifier held they are nobody's: Ctrl+Enter is not Enter plus
        // a bystander, exactly as every chord row reads its modifiers.
        assert_eq!(
            stock().lookup(true, false, false, &Key::Enter, "Enter"),
            None
        );
    }

    #[test]
    fn caps_lock_is_nobodys() {
        // CapsLock types "Z" with no Shift held; the chord reads the modifier,
        // never the letter's case.
        assert_eq!(
            stock().lookup(true, false, false, &ch("Z"), "KeyZ"),
            Some(Command::Undo)
        );
    }

    #[test]
    fn alt_is_a_column_a_user_can_take() {
        // What the Alt column buys a user: a key the shipped table has no row
        // for, answering exactly its own keystroke.
        let mut b = stock();
        b.rebind(Command::GradientFill, alt_chord(false, false, "KeyG"));
        assert_eq!(
            b.lookup(false, false, true, &ch("g"), "KeyG"),
            Some(Command::GradientFill)
        );
        // Exact in all three modifiers, as every row is: the same key bare,
        // under the accelerator, or under both is not this row.
        assert_eq!(b.lookup(false, false, false, &ch("g"), "KeyG"), None);
        assert_eq!(b.lookup(true, false, false, &ch("g"), "KeyG"), None);
        assert_eq!(b.lookup(true, false, true, &ch("g"), "KeyG"), None);
        // Advertised by the engraving the key wears, not by what a Mac's
        // Option would have typed on it.
        assert_eq!(Command::GradientFill.shortcut(&b).as_deref(), Some("Alt+G"));
    }

    #[test]
    fn alt_is_never_a_bystander() {
        // Alt held is not a bystander to a bare row, which is the same
        // exactness Shift gets: `h` mirrors the view and Alt+H does nothing at
        // all until somebody binds it — and that holds now that the shipped
        // table has an Alt column of its own (§18.0.2), which is exactly when
        // a bystanding read would start answering the wrong act.
        assert_eq!(stock().lookup(false, false, true, &ch("h"), "KeyH"), None);
        assert_eq!(stock().lookup(true, false, true, &ch("z"), "KeyZ"), None);
    }

    #[test]
    fn every_pick_scope_has_a_row() {
        // The eyedropper's three reaches are one question with three answers
        // (§18.0.2), so they are checked as a set rather than as three rows: a
        // reach added to the bar must arrive in the palette with it, named for
        // the act and then the reach, and bound under the modifier its own bar
        // comes up on.
        let b = stock();
        for scope in PickScope::ALL {
            let command = Command::SetPickScope(scope);
            assert!(ALL.contains(&command), "{scope:?} has no row in ALL");
            assert!(
                command.name().starts_with("Pick from "),
                "{scope:?} is not named for the act and then the reach"
            );
            let chord = command.shortcut(&b).expect("a reach with no chord");
            assert!(
                chord.starts_with("Alt+"),
                "{scope:?} is on {chord}, not on the modifier that raises its bar"
            );
            // Armed, never committed: no state of the document makes "sample
            // this far" unavailable, so nothing greys these.
            assert!(command.enabled(None));
        }
        assert_eq!(search("pick from").len(), PickScope::ALL.len());
    }

    #[test]
    fn pick_scopes_answer_the_keys_they_advertise() {
        // The left hand's own column, read upward as the bar reads rightward:
        // one more layer let in at every step.
        let b = stock();
        for (code, letter, scope) in [
            ("KeyZ", "z", PickScope::ThisLayer),
            ("KeyA", "a", PickScope::AndBelow),
            ("KeyQ", "q", PickScope::AllLayers),
        ] {
            assert_eq!(
                b.lookup(false, false, true, &ch(letter), code),
                Some(Command::SetPickScope(scope))
            );
            // Exact in all three modifiers, as every row is — and the bare
            // letter matters here: `a` under the accelerator is Deselect, and
            // Alt is what tells the two apart.
            assert_eq!(b.lookup(false, false, false, &ch(letter), code), None);
            assert_eq!(b.lookup(true, false, true, &ch(letter), code), None);
        }
        assert_eq!(
            b.lookup(true, false, false, &ch("a"), "KeyA"),
            Some(Command::Deselect)
        );
        // Spatial, so the reach is the same key on every layout: a Mac's
        // Option+Q types `œ` and still means all layers.
        assert_eq!(
            b.lookup(false, false, true, &ch("\u{153}"), "KeyQ"),
            Some(Command::SetPickScope(PickScope::AllLayers))
        );
    }

    /// **The one rule about Ctrl+Alt** ([`Chord`]): a user may bind it, and we
    /// may not ship it. On a layout with an AltGr the pair is that key, and a
    /// shipped row would eat a character somebody was typing — where a row the
    /// user captured is a keystroke they chose on their own keyboard.
    #[test]
    fn no_default_chord_is_ctrl_alt() {
        for (chord, command) in defaults() {
            assert!(
                !(chord.ctrl && chord.alt),
                "{command:?} ships on Ctrl+Alt, which is AltGr on a German,                  Polish or US-International layout: {}",
                chord_label(&chord)
            );
        }
        // Not vacuous by accident — the table can hold the pair, and a lookup
        // answers it. It is `defaults` that must not carry one.
        let mut b = stock();
        b.rebind(Command::Settings, alt_chord(true, false, "KeyQ"));
        assert_eq!(
            b.lookup(true, false, true, &ch("q"), "KeyQ"),
            Some(Command::Settings)
        );
    }

    #[test]
    fn a_label_spells_every_modifier_it_holds() {
        // Ctrl, Shift, Alt — `drags::chord_label`'s order, and a code that is
        // a word for a mark prints the mark, which is how an Alt binding on
        // the punctuation ring reads.
        assert_eq!(chord_label(&alt_chord(false, true, "Comma")), "Shift+Alt+,");
        assert_eq!(chord_label(&alt_chord(true, false, "KeyQ")), "Ctrl+Alt+Q");
        assert_eq!(chord_label(&chord(true, true, 'z')), "Ctrl+Shift+Z");
        assert_eq!(chord_label(&chord(false, false, 'h')), "H");
    }

    #[test]
    fn brackets_are_positions() {
        // Whatever the layout types on them, the pair beside P steps the brush.
        assert_eq!(
            stock().lookup(false, false, false, &ch("ü"), "BracketLeft"),
            Some(Command::BrushSmaller)
        );
        assert_eq!(
            stock().lookup(false, false, false, &ch("]"), "BracketRight"),
            Some(Command::BrushLarger)
        );
        assert_eq!(
            stock().lookup(true, false, false, &ch("["), "BracketLeft"),
            None
        );
    }

    #[test]
    fn advertised_chords() {
        // A command's first row is what the chrome prints; these strings are
        // the rows' shortcut column.
        let b = stock();
        assert_eq!(Command::Undo.shortcut(&b).as_deref(), Some("Ctrl+Z"));
        assert_eq!(Command::Redo.shortcut(&b).as_deref(), Some("Ctrl+Y"));
        assert_eq!(Command::Deselect.shortcut(&b).as_deref(), Some("Ctrl+D"));
        assert_eq!(
            Command::InvertSelection.shortcut(&b).as_deref(),
            Some("Ctrl+Shift+I")
        );
        assert_eq!(Command::BrushLarger.shortcut(&b).as_deref(), Some("]"));
        // The one advertisement with no chord row: a paste is the browser's
        // binding, not ours, and the registry still says so — and refuses to
        // move it.
        assert_eq!(Command::ImportImage.shortcut(&b).as_deref(), Some("Ctrl+V"));
        assert!(!Command::ImportImage.rebindable());
        // And an unbound command advertises nothing rather than panicking in
        // the chrome — the shortcut column simply is not rendered.
        assert_eq!(Command::SaveDocument.shortcut(&b), None);
    }

    /// A chord as a capture would build it.
    fn chord(ctrl: bool, shift: bool, key: char) -> Chord {
        Chord {
            ctrl,
            shift,
            alt: false,
            key: ChordKey::Char(key),
        }
    }

    /// A spatial chord under Alt, which is the only shape a captured Alt
    /// binding has ([`capture`]).
    fn alt_chord(ctrl: bool, shift: bool, code: &str) -> Chord {
        Chord {
            ctrl,
            shift,
            alt: true,
            key: ChordKey::Code(code.to_string()),
        }
    }

    #[test]
    fn rebind_moves_the_chord_whole() {
        let mut b = stock();
        b.rebind(Command::SaveDocument, chord(true, false, 's'));
        assert_eq!(
            Command::SaveDocument.shortcut(&b).as_deref(),
            Some("Ctrl+S")
        );
        assert_eq!(
            b.lookup(true, false, false, &ch("s"), "KeyS"),
            Some(Command::SaveDocument)
        );
        // The rest of the table stands exactly as shipped.
        assert_eq!(
            b.lookup(true, false, false, &ch("z"), "KeyZ"),
            Some(Command::Undo)
        );
    }

    #[test]
    fn rebind_steals_from_a_default() {
        // Ctrl+Z now saves; Undo's only chord is dead with nothing stored
        // about Undo, so its row shows unbound and the keystroke saves.
        let mut b = stock();
        b.rebind(Command::SaveDocument, chord(true, false, 'z'));
        assert_eq!(
            b.lookup(true, false, false, &ch("z"), "KeyZ"),
            Some(Command::SaveDocument)
        );
        assert_eq!(Command::Undo.shortcut(&b), None);
        // Redo loses only the stolen spelling: Ctrl+Y was untouched, so a
        // command with a second default falls back to it in the same way.
        b.rebind(Command::Settings, chord(true, false, 'y'));
        assert_eq!(Command::Redo.shortcut(&b).as_deref(), Some("Ctrl+Shift+Z"));
        assert_eq!(
            b.lookup(true, true, false, &ch("Z"), "KeyZ"),
            Some(Command::Redo)
        );
    }

    #[test]
    fn rebind_steals_from_an_override() {
        // Two rebinds contest one chord: the later wins, and the earlier is
        // left explicitly unbound — not fallen back to a default it replaced.
        let mut b = stock();
        b.rebind(Command::Undo, chord(true, false, 'q'));
        b.rebind(Command::Redo, chord(true, false, 'q'));
        assert_eq!(
            b.lookup(true, false, false, &ch("q"), "KeyQ"),
            Some(Command::Redo)
        );
        assert_eq!(Command::Undo.shortcut(&b), None);
        // And the chords the overrides replaced are dead, not inherited: an
        // override is the command's whole binding.
        assert_eq!(b.lookup(true, false, false, &ch("z"), "KeyZ"), None);
        assert_eq!(b.lookup(true, false, false, &ch("y"), "KeyY"), None);
    }

    /// A rebinding survives the store, and a row for a command this build does not
    /// have costs its own row.
    ///
    /// The reading half of `load` without the browser: `storage::load_list` is what
    /// drops the unreadable row, and it is tested there — what is this module's own is
    /// that a `Command` is spelled by its variant, payload included.
    #[test]
    fn stored_bindings_round_trip() {
        let mut b = stock();
        b.rebind(Command::MirrorView, chord(false, true, 'm'));
        b.rebind(Command::Undo, chord(false, true, 'm')); // steals; MirrorView None
        b.rebind(
            Command::TogglePanel(PanelId::Layers),
            chord(true, true, 'l'),
        );
        b.rebind(Command::Transform, alt_chord(false, false, "KeyT"));
        let stored: Vec<StoredBinding> = b
            .overrides
            .iter()
            .map(|(command, chord)| StoredBinding {
                command: *command,
                chord: chord.clone(),
            })
            .collect();
        let json = serde_json::to_string(&stored).unwrap();
        // The payload rides inside the stored name — so renaming a `PanelId` variant
        // orphans its stored binding exactly the way renaming a command does.
        assert!(
            json.contains(r#"{"TogglePanel":"Layers"}"#),
            "a command is its variant's name: {json}"
        );
        let restored = Bindings {
            overrides: serde_json::from_str::<Vec<StoredBinding>>(&json)
                .unwrap()
                .into_iter()
                .map(|row| (row.command, row.chord))
                .collect(),
        };
        assert_eq!(restored, b);
        // A name from a build that knew commands this one does not is unreadable as a
        // row, which is what puts it in reach of the drop `load_list` does.
        assert!(serde_json::from_str::<StoredBinding>(r#"{"command":"NoSuchCommand"}"#).is_err());
    }

    /// The tier is stored as the flags it was before there was a tier, so a
    /// browser that saved its table under an earlier build still loads it.
    #[test]
    fn a_chord_stored_before_alt_existed_still_reads() {
        // Exactly what the store holds today: no `alt` key at all.
        let old = r#"{"ctrl":true,"shift":false,"key":{"Char":"s"}}"#;
        assert_eq!(
            serde_json::from_str::<Chord>(old).unwrap(),
            chord(true, false, 's'),
            "a row missing `alt` means the accelerator it meant, not a refusal"
        );
        let bare = r#"{"ctrl":false,"shift":true,"key":{"Char":"m"}}"#;
        assert_eq!(
            serde_json::from_str::<Chord>(bare).unwrap(),
            chord(false, true, 'm')
        );
        // And the flags are what it writes back, so a downgrade reads its own
        // rows and an Alt one simply looks like a chord it does not know.
        let alt = alt_chord(false, false, "KeyG");
        let json = serde_json::to_string(&alt).unwrap();
        assert!(
            json.contains(r#""ctrl":false"#) && json.contains(r#""alt":true"#),
            "a tier is stored as the flags an event reports: {json}"
        );
        assert_eq!(serde_json::from_str::<Chord>(&json).unwrap(), alt);
        // And Ctrl+Alt loads, because a stored row is by definition one the
        // user asked for — the rule it must not break is about what *we* ship
        // (`no_default_chord_is_ctrl_alt`), not about what they may keep.
        let both = r#"{"ctrl":true,"alt":true,"shift":false,"key":{"Code":"KeyQ"}}"#;
        assert_eq!(
            serde_json::from_str::<Chord>(both).unwrap(),
            alt_chord(true, false, "KeyQ")
        );
    }

    #[test]
    fn capture_reads_a_chord() {
        assert_eq!(
            capture(true, false, false, &ch("s"), "KeyS"),
            Capture::Chord(chord(true, false, 's'))
        );
        // Shift's uppercase is the chord's Shift bit, not the letter's case.
        assert_eq!(
            capture(false, true, false, &ch("M"), "KeyM"),
            Capture::Chord(chord(false, true, 'm'))
        );
        // A key that types nothing is named by its position.
        assert_eq!(
            capture(false, false, false, &Key::F5, "F5"),
            Capture::Chord(Chord {
                ctrl: false,
                shift: false,
                alt: false,
                key: ChordKey::Code("F5".to_string()),
            })
        );
        // And so is every key held through Alt, which types something that is
        // not itself on at least one platform: the capture names the position
        // rather than storing a `©` nobody's keycap says.
        assert_eq!(
            capture(false, false, true, &ch("g"), "KeyG"),
            Capture::Chord(alt_chord(false, false, "KeyG"))
        );
        assert_eq!(
            capture(false, false, true, &ch("\u{a9}"), "KeyG"),
            Capture::Chord(alt_chord(false, false, "KeyG"))
        );
        assert_eq!(
            capture(false, false, false, &Key::Escape, "Escape"),
            Capture::Cancel
        );
        // Backspace is the capture's own eraser, so it can never be a chord.
        assert_eq!(
            capture(false, false, false, &Key::Backspace, "Backspace"),
            Capture::Clear
        );
    }

    #[test]
    fn unbind_erases_rather_than_resets() {
        let mut b = stock();
        b.unbind(Command::Undo);
        assert_eq!(Command::Undo.shortcut(&b), None);
        assert_eq!(b.lookup(true, false, false, &ch("z"), "KeyZ"), None);
        // The chord is anyone's to take now…
        b.rebind(Command::SaveDocument, chord(true, false, 'z'));
        assert_eq!(
            b.lookup(true, false, false, &ch("z"), "KeyZ"),
            Some(Command::SaveDocument)
        );
        // …and the erased command can be given a fresh one.
        b.rebind(Command::Undo, chord(true, false, 'u'));
        assert_eq!(Command::Undo.shortcut(&b).as_deref(), Some("Ctrl+U"));
    }

    #[test]
    fn capture_refuses_what_could_never_fire() {
        // A modifier alone is a chord still being formed.
        assert_eq!(
            capture(true, false, false, &Key::Control, "ControlLeft"),
            Capture::Pending
        );
        // Ctrl+Alt *is* capturable — the user asking for it is the whole
        // difference from shipping it (`no_default_chord_is_ctrl_alt`) — and
        // it comes back spatial like every Alt chord, which is what keeps a
        // Polish layout's AltGr+X from being stored as `Char('x')`… or worse,
        // as the `ń` it actually typed.
        assert_eq!(
            capture(true, false, true, &ch("x"), "KeyX"),
            Capture::Chord(alt_chord(true, false, "KeyX"))
        );
        // Space is a hold that owns both edges of its key, and `input` claims
        // it off the key itself — so it is refused under every modifier, not
        // only where the character survives to be read.
        assert_eq!(
            capture(false, false, false, &ch(" "), "Space"),
            Capture::Pending
        );
        assert_eq!(
            capture(false, false, true, &ch(" "), "Space"),
            Capture::Pending
        );
        // The bare digit row is the rack's hold (§18.1.8).
        assert_eq!(
            capture(false, false, false, &ch("1"), "Digit1"),
            Capture::Pending
        );
        // …but a digit under the accelerator is nobody's hold — nor under Alt,
        // which the rack refuses too, so capture may not be stricter than the
        // hold it is deferring to.
        assert_eq!(
            capture(true, false, false, &ch("1"), "Digit1"),
            Capture::Chord(chord(true, false, '1'))
        );
        assert_eq!(
            capture(false, false, true, &ch("1"), "Digit1"),
            Capture::Chord(alt_chord(false, false, "Digit1"))
        );
        // Ctrl+V is the paste's, and a row would prevent_default it dead.
        assert_eq!(
            capture(true, false, false, &ch("v"), "KeyV"),
            Capture::Pending
        );
    }

    #[test]
    fn shape_tools_are_one_family() {
        // Three commands that must read as three answers to one question
        // wherever they are shown — so they are checked as a set, not as three
        // independent rows: the same terse word in the panel's 280px column,
        // the same shape-then-act name in the palette, and a chord trio that
        // is one rule (each command's own initial) rather than three letters.
        let b = stock();
        for (command, word, name) in [
            (Command::SelectRect, "Rect", "Rectangle select"),
            (Command::SelectEllipse, "Ellipse", "Ellipse select"),
            (Command::SelectLasso, "Lasso", "Lasso select"),
        ] {
            // The abbreviation is a shortening of the name, never another word
            // for the act: a chip and a palette row must be recognisable as
            // the same thing by someone who has only seen the other.
            assert_eq!(command.word(), word);
            assert_eq!(command.name(), name);
            assert!(name.to_lowercase().starts_with(&word.to_lowercase()));
            assert!(name.ends_with(" select"));
            assert_eq!(
                command.shortcut(&b).as_deref(),
                Some(&word[..1]),
                "{command:?}'s chord is not its own initial"
            );
            // A tool is armed, never committed, so nothing greys it: there is
            // no document state that makes "draw a rectangle" unavailable.
            assert!(command.enabled(None));
        }
    }

    #[test]
    fn shape_tools_answer_the_letters_they_advertise() {
        // The other half of the trio: what the chips *say* is what the
        // keyboard *does*, which is one table read twice.
        let b = stock();
        assert_eq!(
            b.lookup(false, false, false, &ch("r"), "KeyR"),
            Some(Command::SelectRect)
        );
        assert_eq!(
            b.lookup(false, false, false, &ch("e"), "KeyE"),
            Some(Command::SelectEllipse)
        );
        assert_eq!(
            b.lookup(false, false, false, &ch("l"), "KeyL"),
            Some(Command::SelectLasso)
        );
        // Bare letters, so an accelerator over one is nobody's — Ctrl+R must
        // stay the browser's reload rather than arming a marquee.
        assert_eq!(b.lookup(true, false, false, &ch("r"), "KeyR"), None);
    }

    /// **Every variant of [`Command`] has a row in [`ALL`]**, which §25.2 step 5
    /// says nothing will remind you of:
    ///
    /// > List it in `ALL` — by hand, and nothing will remind you: a variant left
    /// > out compiles clean and is simply unfindable in the palette.
    ///
    /// This is the reminder. `variant_count` is a compile-time property of the
    /// enum, so adding a variant moves the expected total and fails here — and
    /// the only edit that fixes it is the row that was forgotten.
    ///
    /// The arithmetic is the one thing worth reading twice. Two variants carry a
    /// payload drawn from the chrome's own closed set, and each stands for as
    /// many acts as that set has members — so they count once as variants and
    /// once per member as rows. That the members really are all present is
    /// `every_panel_has_a_toggle_row` and `every_pick_scope_has_a_row`, which
    /// check the two families by name; this checks the total, and between them
    /// there is nowhere for a missing row to hide.
    #[test]
    fn all_lists_every_command() {
        let payload_families = 2; // TogglePanel, SetPickScope
        let expected = std::mem::variant_count::<Command>() - payload_families
            + PanelId::ALL.len()
            + PickScope::ALL.len();
        assert_eq!(
            ALL.len(),
            expected,
            "ALL has {} rows for {} acts — a command was added without one, and              it is unfindable in the palette until it has one (§25.2 step 5)",
            ALL.len(),
            expected
        );
    }

    #[test]
    fn names_are_unique() {
        // The palette prints nothing but the name, so two commands sharing one
        // would be indistinguishable rows. Runs over ALL, which also makes a
        // copy-pasted duplicate *entry* fail, not only a duplicate name.
        for (i, a) in ALL.iter().enumerate() {
            for b in &ALL[i + 1..] {
                assert_ne!(a, b, "one command listed twice in ALL");
                assert_ne!(
                    a.name(),
                    b.name(),
                    "{a:?} and {b:?} share a display name, so the palette \
                     could not tell their rows apart"
                );
            }
        }
    }

    #[test]
    fn every_panel_has_a_toggle_row() {
        // ALL is kept by hand, but this family need not be taken on faith: a
        // panel added to the stack must arrive in the palette with it, wearing
        // a name that still says which panel it is — and saying "panel", which
        // is how a query for the word lists the whole stack.
        for id in PanelId::ALL {
            let command = Command::TogglePanel(id);
            assert!(ALL.contains(&command), "{id:?} has no row in ALL");
            assert!(
                command
                    .name()
                    .to_lowercase()
                    .starts_with(&id.title().to_lowercase()),
                "{id:?}'s command is not named after its panel"
            );
        }
        assert_eq!(search("panel").len(), PanelId::ALL.len());
    }

    #[test]
    fn the_visibility_menu_is_the_stack_plus_three() {
        // The menu's panel half is `PanelId::ALL` in the order it stacks —
        // the claim that the list reads as a picture of the stack. It is
        // written out in `VisibilityToggle::ALL` rather than folded in from
        // there, so this is what holds the two together when a panel arrives.
        let panels: Vec<PanelId> = VisibilityToggle::ALL
            .into_iter()
            .filter_map(|entry| match entry {
                VisibilityToggle::Panel(id) => Some(id),
                _ => None,
            })
            .collect();
        assert_eq!(
            panels,
            PanelId::ALL.to_vec(),
            "the visibility menu's panel rows are not the stack, in order"
        );
        for (i, entry) in VisibilityToggle::ALL.into_iter().enumerate() {
            // Every row is a registry command, which is the whole of what an
            // entry is: reachable by search and bindable to a chord, with the
            // menu adding nothing of its own.
            assert!(
                ALL.contains(&entry.command()),
                "{entry:?} has no row in ALL, so the menu reaches an act search cannot"
            );
            // A doubled entry would draw its row twice and hand both the same
            // act; nothing else would notice.
            assert!(
                !VisibilityToggle::ALL[i + 1..].contains(&entry),
                "{entry:?} is listed twice in the visibility menu"
            );
        }
    }

    #[test]
    fn every_bound_command_is_findable() {
        // The chord table names acts; every one of them must also be a row the
        // palette can list, or a chord would reach a command search cannot.
        for (_, command) in defaults() {
            assert!(
                ALL.contains(&command),
                "{command:?} has a chord but is missing from ALL"
            );
        }
    }

    #[test]
    fn empty_query_offers_the_file_family() {
        assert_eq!(search(""), BASIC.to_vec());
        assert_eq!(search("   "), BASIC.to_vec());
        for command in BASIC {
            assert!(ALL.contains(command), "{command:?} in BASIC but not ALL");
        }
    }

    #[test]
    fn search_is_caseless_and_ranks_prefixes_first() {
        assert_eq!(search("UNDO"), vec![Command::Undo]);
        // "in" begins "Invert selection" and merely appears in "Timing stats…",
        // so the prefix match leads whatever ALL's order says.
        let hits = search("in");
        assert_eq!(hits.first(), Some(&Command::InvertSelection));
        assert!(hits.contains(&Command::TimingStats));
        // A word from the middle of a name still finds it.
        assert!(search("selection").contains(&Command::FillSelection));
        // And a miss is an empty list, not an error and not BASIC.
        assert!(search("qqq").is_empty());
    }

    #[test]
    fn search_speaks_other_softwares_words() {
        // Another tool's vocabulary finds our act by its alias…
        assert_eq!(search("flip"), vec![Command::MirrorView]);
        assert_eq!(search("increase brush size"), vec![Command::BrushLarger]);
        assert_eq!(search("decrease"), vec![Command::BrushSmaller]);
        assert_eq!(search("select all"), vec![Command::Deselect]);
        assert_eq!(search("preferences"), vec![Command::Settings]);
        assert_eq!(search("crop"), vec![Command::AddFrame]);
        // The shape tools answer to the marquee vocabulary, and bare "marquee"
        // lands on the rectangle first: it is that word's own default
        // elsewhere, and the alias tiers put the prefix match above the
        // ellipse's "Elliptical marquee".
        assert_eq!(
            search("marquee"),
            vec![Command::SelectRect, Command::SelectEllipse]
        );
        assert_eq!(search("freehand"), vec![Command::SelectLasso]);
        // …and a name match still leads an alias match: "save" begins two
        // display names and then Export's "Save as", in that order.
        assert_eq!(
            search("save"),
            vec![
                Command::SaveDocument,
                Command::SavePreset,
                Command::ExportImage
            ]
        );
    }

    #[test]
    fn no_alias_shadows_a_name() {
        // An alias equal to some command's display name would have a query
        // answered by two rows, only one of them wearing the matched word.
        // The borrowed names must stay borrowed — if a later command claims
        // one as its display name, the alias yields.
        for a in ALL {
            for alias in a.aliases() {
                for b in ALL {
                    assert!(
                        !alias.eq_ignore_ascii_case(b.name()),
                        "{a:?}'s alias {alias:?} is {b:?}'s display name"
                    );
                }
            }
        }
    }
}
