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
//! The keyboard is one column of the registry, and it is the user's. The
//! shipped chords are `(Chord, Command)` rows in [`defaults`]; this browser's
//! rebindings lie over them as [`Bindings`] — a signal on the app state, stored
//! like the preset library, edited from the palette's own rows
//! (`main::CommandSearch`: click a row's chord, press the new one). The
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
//! search palette can offer (`main::CommandSearch`) and a chord can carry
//! whole.
//!
//! The guard that precedes any binding at all — a keystroke in a text field is
//! the field's (`platform::KeyEvent::on_text_entry`) — is `input`'s too: by the
//! time this table is consulted, the keystroke is known to be the canvas's.

use dioxus::html::{Key, Modifiers};
use dioxus::prelude::{ReadableExt, Signal, WritableExt};
use serde::{Deserialize, Serialize};

use crate::icons;
use crate::input::accel;
use crate::layout::PanelId;
use crate::panels::brush::{MAX_RADIUS, MIN_RADIUS};
use crate::platform;
use crate::state::{AppState, dispatch, update_brush};
use stark_engine::ObservableState;
use stark_engine::command::{DocCommand, ViewCommand};
use stark_model::document::SelectionOp;

/// One key, namespaced and versioned like the preference and library keys'.
const KEY_BINDINGS: &str = "stark.bindings.v1";

/// The key half of a binding: which modifier tier, and which key.
///
/// Chords are **exact** about their modifiers. Ctrl+Shift+Z is not Ctrl+Z with
/// a bystanding Shift, it is another row; and a modifier no chord can even name
/// (Alt) matches nothing at all, because on Windows AltGr arrives as Ctrl+Alt,
/// and a table that shrugged at Alt would fire its Ctrl rows under a layout's
/// ordinary typing. (The digit rack tolerates Shift, and may: it is a hold in
/// `input`, not a row here, and its reasons are its own — §18.1.8.)
///
/// Serde, because a chord the user set is stored with this browser's
/// preferences and libraries — see [`Bindings`].
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
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
///
/// A capture makes the same choice ([`capture`]): a key that types a character
/// is taken as the character, and a key that types nothing — an F-key, an
/// arrow — has only its position to be named by.
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
    /// Show or hide one of the floating tool panels (§11). The one variant
    /// with a payload, and it keeps the module's rule because the target is
    /// the chrome's, not the document's: `PanelId` is a closed set the build
    /// enumerates, so each of the six is still a single nameable act —
    /// listed in [`ALL`], searchable, bindable to a chord of its own.
    TogglePanel(PanelId),
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
    fn ch(ctrl: bool, shift: bool, key: char) -> Chord {
        Chord {
            ctrl,
            shift,
            key: ChordKey::Char(key),
        }
    }
    fn code(key: &str) -> Chord {
        Chord {
            ctrl: false,
            shift: false,
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
        (ch(false, false, 'h'), Command::MirrorView),
        (code("BracketLeft"), Command::BrushSmaller),
        (code("BracketRight"), Command::BrushLarger),
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
        if alt {
            return None;
        }
        let hit =
            |chord: &Chord| chord.ctrl == accel && chord.shift == shift && chord.key.hit(key, code);
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

/// The stored form: overrides keyed by the variant's name, which is the stable
/// identity rebinding was always going to hang on. An id today's build does not
/// know is dropped on load — a binding for a retired command is a binding for
/// nothing.
#[derive(Serialize, Deserialize)]
struct StoredBindings {
    overrides: Vec<(String, Option<Chord>)>,
}

/// Seed [`AppState::bindings`](crate::state::AppState::bindings) from this
/// browser's stored rebindings. Called once at app start, beside `prefs::load`.
pub fn load(state: AppState) {
    let Some(stored) = crate::storage::get(KEY_BINDINGS)
        .and_then(|json| serde_json::from_str::<StoredBindings>(&json).ok())
    else {
        return;
    };
    let overrides = stored
        .overrides
        .into_iter()
        .filter_map(|(id, chord)| {
            ALL.iter()
                .find(|c| format!("{c:?}") == id)
                .map(|&c| (c, chord))
        })
        .collect();
    let mut bindings = state.bindings;
    bindings.set(Bindings { overrides });
}

/// Give `command` the captured chord, and persist the table — the palette's
/// commit (`main::CommandSearch`), written through [`Bindings::rebind`] so a
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
    let stored = StoredBindings {
        overrides: bindings
            .peek()
            .overrides
            .iter()
            .map(|(c, chord)| (format!("{c:?}"), chord.clone()))
            .collect(),
    };
    if let Ok(json) = serde_json::to_string(&stored) {
        crate::storage::set(KEY_BINDINGS, "the shortcuts", &json);
    }
}

/// What a keydown means to a rebinding capture (`main::CommandSearch`): the
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
    /// - anything with **Alt**, which the table matches nothing on (AltGr);
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
    if alt
        || matches!(
            key,
            Key::Control | Key::Shift | Key::Alt | Key::AltGraph | Key::Meta
        )
    {
        return Capture::Pending;
    }
    // The rack's own reading of the digit row, asked rather than restated.
    if !accel && crate::slots::of_code(code).is_some() {
        return Capture::Pending;
    }
    match key {
        Key::Character(c) => {
            let mut chars = c.chars();
            let (Some(k), None) = (chars.next(), chars.next()) else {
                return Capture::Pending;
            };
            if k == ' ' || (accel && !shift && k.eq_ignore_ascii_case(&'v')) {
                return Capture::Pending;
            }
            Capture::Chord(Chord {
                ctrl: accel,
                shift,
                key: ChordKey::Char(k.to_ascii_lowercase()),
            })
        }
        _ if !code.is_empty() => Capture::Chord(Chord {
            ctrl: accel,
            shift,
            key: ChordKey::Code(code.to_string()),
        }),
        _ => Capture::Pending,
    }
}

/// Every command, in the order the search palette lists ties: the file family
/// first (the resting offer, [`BASIC`]), then the acts on the document and the
/// view, the openers, the toggles, and the keyboard-only steps last.
///
/// By hand, kept in step in review — nothing here can make the compiler demand
/// a new variant be listed, and a variant left out is simply unfindable in the
/// palette. `tests::names_are_unique` runs over this list, so at least a
/// duplicate display name cannot hide in it.
pub const ALL: &[Command] = &[
    Command::NewDocument,
    Command::OpenDocument,
    Command::SaveDocument,
    Command::ImportImage,
    Command::ExportImage,
    Command::Share,
    Command::Undo,
    Command::Redo,
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
    state.bindings.peek().lookup(
        accel(m),
        m.contains(Modifiers::SHIFT),
        m.contains(Modifiers::ALT),
        &e.key(),
        &e.code(),
    )
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
            // The Panels menu is a picture of the stack, so its rows wear the
            // panels' own title-bar labels.
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
    pub fn active(self, state: AppState) -> Option<bool> {
        match self {
            Command::ToggleTimeline => Some(*state.timeline.open.read()),
            Command::ToggleNavigator => Some(*state.navigator.read()),
            Command::ToggleQuickBrushes => Some(*state.slots.pinned.read()),
            Command::TogglePanel(id) => Some(!state.panels.hidden.read().contains(&id)),
            Command::Share => Some(*state.collab.phase.read() == crate::collab::CollabPhase::Shared),
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

/// A chord spelled out for the chrome. Private: the chrome asks
/// [`Command::shortcut`], which is this over [`Bindings::of`], plus the one
/// advertisement that has no chord.
/// "Ctrl" names the accelerator on every platform for now, as the menu always
/// has — a ⌘ on the one platform that draws it is a presentation question, not
/// a binding one.
fn chord_label(chord: &Chord) -> String {
    let mut s = String::new();
    if chord.ctrl {
        s.push_str("Ctrl+");
    }
    if chord.shift {
        s.push_str("Shift+");
    }
    match &chord.key {
        ChordKey::Char(c) => s.push(c.to_ascii_uppercase()),
        // A position's label is its US engraving: what the keycap most often
        // says, and the only honest answer short of asking the layout — which
        // the web cannot portably do. Rebinding lets the label be wrong less
        // often than a fixed binding already was.
        ChordKey::Code(code) => match code.as_str() {
            "BracketLeft" => s.push('['),
            "BracketRight" => s.push(']'),
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
    fn caps_lock_is_nobodys() {
        // CapsLock types "Z" with no Shift held; the chord reads the modifier,
        // never the letter's case.
        assert_eq!(
            stock().lookup(true, false, false, &ch("Z"), "KeyZ"),
            Some(Command::Undo)
        );
    }

    #[test]
    fn alt_matches_nothing() {
        // AltGr arrives as Ctrl+Alt on Windows: a layout typing *through* Alt
        // must not find Ctrl rows under its ordinary characters.
        assert_eq!(stock().lookup(true, false, true, &ch("z"), "KeyZ"), None);
        assert_eq!(stock().lookup(false, false, true, &ch("h"), "KeyH"), None);
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
            key: ChordKey::Char(key),
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

    #[test]
    fn stored_bindings_round_trip() {
        let mut b = stock();
        b.rebind(Command::MirrorView, chord(false, true, 'm'));
        b.rebind(Command::Undo, chord(false, true, 'm')); // steals; MirrorView None
        b.rebind(
            Command::TogglePanel(PanelId::Layers),
            chord(true, true, 'l'),
        );
        let stored = StoredBindings {
            overrides: b
                .overrides
                .iter()
                .map(|(c, chord)| (format!("{c:?}"), chord.clone()))
                .collect(),
        };
        // A payload rides inside the stored id — so renaming a PanelId variant
        // orphans its stored binding exactly the way renaming a command does:
        // dropped on load, never an error.
        assert!(
            stored
                .overrides
                .iter()
                .any(|(id, _)| id == "TogglePanel(Layers)")
        );
        let json = serde_json::to_string(&stored).unwrap();
        let back: StoredBindings = serde_json::from_str(&json).unwrap();
        let restored = Bindings {
            overrides: back
                .overrides
                .into_iter()
                .filter_map(|(id, chord)| {
                    ALL.iter()
                        .find(|c| format!("{c:?}") == id)
                        .map(|&c| (c, chord))
                })
                .collect(),
        };
        assert_eq!(restored, b);
        // An id from a build that knew commands this one does not is dropped,
        // not an error.
        let stale: StoredBindings =
            serde_json::from_str(r#"{"overrides":[["NoSuchCommand",null]]}"#).unwrap();
        assert!(
            stale
                .overrides
                .iter()
                .all(|(id, _)| !ALL.iter().any(|c| format!("{c:?}") == *id))
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
                key: ChordKey::Code("F5".to_string()),
            })
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
        // Alt in any combination matches nothing in the table (AltGr).
        assert_eq!(
            capture(false, false, true, &ch("x"), "KeyX"),
            Capture::Pending
        );
        // Space and the bare digit row are holds, claimed before the table.
        assert_eq!(
            capture(false, false, false, &ch(" "), "Space"),
            Capture::Pending
        );
        assert_eq!(
            capture(false, false, false, &ch("1"), "Digit1"),
            Capture::Pending
        );
        // …but a digit under the accelerator is nobody's hold.
        assert_eq!(
            capture(true, false, false, &ch("1"), "Digit1"),
            Capture::Chord(chord(true, false, '1'))
        );
        // Ctrl+V is the paste's, and a row would prevent_default it dead.
        assert_eq!(
            capture(true, false, false, &ch("v"), "KeyV"),
            Capture::Pending
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
