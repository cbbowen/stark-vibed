//! The command registry (§11, §25): every simple act the chrome can ask for,
//! declared as one variant of [`Command`] carrying its whole description.
//! §25.2 is the checklist for adding one.
//!
//! A *simple* command is one that takes no argument at its call site — click "Undo",
//! press Ctrl+Z, pick Undo from a menu, and it is the same act reached three ways.
//! [`Command`] is that act's one name, and everything a chrome *says* about it hangs
//! off the variant: the word a control wears ([`Command::name`], [`Command::word`]),
//! the tooltip, whether a row is greyed ([`Command::enabled`]), and the chord that
//! reaches it from the keyboard ([`Bindings`]).
//!
//! # What is here and what is a frontend's
//!
//! Everything above is a description, and a description is not toolkit-shaped: it is
//! a word, a rule over [`ObservableState`], or a row in a table. So it is here, and
//! both frontends render *the same* registry rather than restating it — which is the
//! bargain §25 already made between a menu row and the keyboard, one crate up.
//!
//! Three things stay with each frontend, and each for its own reason:
//!
//! - **what a command *does*** — `run` dispatches, opens dialogs, writes signals, and
//!   asks gates that are the app's;
//! - **whether it is *live right now*** — `active` reads a frontend's own state
//!   rather than the engine's projection;
//! - **its mark** — the web app's icons are inline SVG, which is a DOM idiom; a
//!   native chrome draws a glyph another way entirely.
//!
//! The keyboard is one column of the registry, and it is the user's. The shipped
//! chords are `(Chord, Command)` rows in [`defaults`]; a client's rebindings lie over
//! them as [`Bindings`], stored like the preset library (§25.6). [`Bindings::lookup`]
//! and [`Command::shortcut`] are the only two readers, so what the keyboard answers
//! and what a row claims cannot drift apart.

use serde::{Deserialize, Serialize};
use stark_engine::ObservableState;

use crate::icons::Icon;
use crate::keys::{Keystroke, Mods, Role};
use crate::panels::PanelId;
use crate::slots;
use crate::storage::{Entry, Store};

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
/// a frontend's own key event and decline the keystroke when it is set. That leaves
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
/// else there, which is the rack's own argument (`stark_ui::slots::of_code`, §18.1.8).
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
    /// Whether a keystroke is this key.
    fn hit(&self, stroke: &Keystroke<'_>) -> bool {
        match self {
            ChordKey::Char(want) => stroke.typed.is_some_and(|k| k.eq_ignore_ascii_case(want)),
            ChordKey::Code(want) => stroke.code == want,
        }
    }
}

/// How far the eyedropper sees, before a frontend's `PickState::group_only` narrows it
/// (§18.0.2).
///
/// The *choice*, not [`PickSource`](stark_engine::PickSource): which layer "this layer"
/// means is resolved against the selected layer at the moment of the sample
/// (a frontend's `input::pick_color`), so the bar cannot be left holding a layer id whose
/// layer has since been deleted — the same reason the radius is a number here and a
/// clamped one in the engine.
///
/// Serde, because a scope is named in a stored record: the rebinding of the
/// `Command::SetPickScope` row that carries it (§25.2). The derive spells a
/// variant exactly as `Debug` does, so the stored name and this enum are one
/// word by construction, and a variant renamed costs that browser's binding
/// rather than quietly mis-matching it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum PickScope {
    /// The selected layer alone, ignoring anything over or under it.
    ThisLayer,
    /// The selected layer and everything beneath it — what the canvas would show
    /// with the layers above switched off.
    AndBelow,
    /// Every visible layer.
    #[default]
    AllLayers,
}

impl PickScope {
    /// Every reach, ordered by how much each one lets in — one layer, the layers
    /// beneath it too, then all of them. The ordering is the claim that the three
    /// are one question — *how far does this sample see* — rather than three
    /// unrelated buttons (§18.0.2), and the default sits where that puts it rather
    /// than at the head of the row, which is what an ordering worth having costs.
    ///
    /// Written once here because the bar's row is drawn from it and the registry
    /// must agree with the bar: `commands::ALL` is kept by hand, so
    /// `commands::tests::every_pick_scope_has_a_row` walks this array to say that a
    /// reach added to the bar arrives in the palette with a chord of its own.
    pub const ALL: [PickScope; 3] = [
        PickScope::ThisLayer,
        PickScope::AndBelow,
        PickScope::AllLayers,
    ];
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
    /// Show the canvas with the display's headroom, or as an export would look
    /// (§6.5). A view act; enabled only where the surface can show anything above
    /// white.
    ToggleHdr,
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
    /// be lit for (a frontend's `active`) and a chord can land on directly,
    /// where one "next scope" key would answer differently depending on a state
    /// the hand cannot see — and the bar this lights is up for as long as Alt is
    /// held, so the hand is looking straight at the answer it would have to
    /// count.
    SetPickScope(PickScope),
    /// Pick the selected paint up into the transform widget (§16.6).
    Transform,
    /// Cut what the selection holds on the active layer into a floating child
    /// layer (§16.12) — the act the pinned drag commits on its first travel
    /// (`input::carry`), reachable as a click. The selection is consumed: the
    /// mask has become a layer, the thing the outline stood in for.
    FloatSelection,
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
/// rebindings laid over them. Lives on a frontend's app state as
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
    pub overrides: Vec<(Command, Option<Chord>)>,
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
    pub fn lookup(&self, stroke: &Keystroke<'_>) -> Option<Command> {
        // All three modifiers, exactly — Alt included, which the shipped table
        // holds a column of (§18.0.2) and any rebinding may reach for
        // ([`Chord`]).
        let hit = |chord: &Chord| {
            chord.ctrl == stroke.mods.ctrl
                && chord.shift == stroke.mods.shift
                && chord.alt == stroke.mods.alt
                && chord.key.hit(stroke)
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
pub struct StoredBinding {
    pub command: Command,
    #[serde(default)]
    pub chord: Option<Chord>,
}

impl Entry for StoredBinding {
    const STORE: Store = Store::Bindings;
}

/// What a keydown means to a rebinding capture: the chord to commit, the capture
/// called off, or nothing yet.
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

/// Read one keydown as a capture — the policy half, taken apart from the platform
/// event the way [`Bindings::lookup`] is, so a test can reach it and both frontends
/// answer the same.
pub fn capture(stroke: &Keystroke<'_>) -> Capture {
    match stroke.role {
        Role::Escape => return Capture::Cancel,
        Role::Backspace => return Capture::Clear,
        Role::Modifier => return Capture::Pending,
        Role::Ordinary => {}
    }
    let Mods { ctrl, shift, alt } = stroke.mods;
    // Space is claimed before the table whatever is held with it — a frontend's
    // keydown arms the pan off the key itself — so it is nobody's chord, and it is
    // asked for by both names because Alt is about to take the character away.
    if crate::keys::is_space(stroke) {
        return Capture::Pending;
    }
    // The rack's own reading of the digit row, asked rather than restated — and
    // asked under the same modifiers the rack claims under, which is neither: it
    // tolerates Shift and refuses Alt, so Alt+1 is a chord going spare rather than a
    // hold being stolen.
    if !ctrl && !alt && slots::of_code(stroke.code).is_some() {
        return Capture::Pending;
    }
    // **Under Alt a key does not type its own character**, so there is no mnemonic to
    // name it by and only the position is left — the same sentence [`ChordKey`]
    // already writes for the F-keys and the arrows, reaching one key further.
    // Option+G types `©` on a Mac, and AltGr+A types `ą` on a Polish one: naming the
    // character would store a chord that means another key on the next machine and
    // print a label no keycap says, where the position is the same key and the same
    // engraving on both.
    if alt {
        if stroke.code.is_empty() {
            return Capture::Pending;
        }
        return Capture::Chord(Chord {
            ctrl,
            shift,
            alt,
            key: ChordKey::Code(stroke.code.to_string()),
        });
    }
    match stroke.typed {
        Some(k) => {
            // Not ours to give away: the paste import rides the platform's own event
            // (§23), and a chord row would take it dead.
            if ctrl && !shift && k.eq_ignore_ascii_case(&'v') {
                return Capture::Pending;
            }
            Capture::Chord(Chord {
                ctrl,
                shift,
                alt,
                key: ChordKey::Char(k.to_ascii_lowercase()),
            })
        }
        None if !stroke.code.is_empty() => Capture::Chord(Chord {
            ctrl,
            shift,
            alt,
            key: ChordKey::Code(stroke.code.to_string()),
        }),
        None => Capture::Pending,
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
    Command::FloatSelection,
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
    Command::ToggleHdr,
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
            Command::ToggleHdr => "HDR",
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
            Command::FloatSelection => "Float selection",
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

    /// The command's mark (`crate::icons`). Total, because a control rendering
    /// a command has nothing else to wear; the three keyboard-only commands
    /// wear the mark of the knob they step or the subject they turn, on the
    /// sharing argument `icons` already makes — the bracket keys are the Size
    /// slider's own knob (§18.1.9), so they wear its ruler.
    pub fn icon(self) -> Icon {
        match self {
            Command::Undo => crate::icons::UNDO,
            Command::Redo => crate::icons::REDO,
            Command::Deselect => crate::icons::SELECTION_NONE,
            Command::InvertSelection => crate::icons::SELECTION_INVERT,
            // The one family where the glyph *is* the meaning rather than the
            // control's (`icons`): a tool that draws a rectangle is marked
            // with a rectangle.
            Command::SelectRect => crate::icons::RECTANGLE,
            Command::SelectEllipse => crate::icons::CIRCLE,
            Command::SelectLasso => crate::icons::LASSO,
            Command::MirrorView => crate::icons::MIRROR_VIEW,
            Command::ToggleHdr => crate::icons::HDR,
            Command::BrushSmaller | Command::BrushLarger => crate::icons::SIZE,
            Command::NewDocument => crate::icons::NEW_DOCUMENT,
            Command::OpenDocument => crate::icons::OPEN_DOC,
            Command::SaveDocument => crate::icons::SAVE,
            Command::ImportImage => crate::icons::IMPORT_IMAGE,
            Command::ExportImage => crate::icons::EXPORT,
            Command::Share => crate::icons::SHARE,
            Command::ToggleTimeline => crate::icons::TIMELINE,
            Command::TimingStats => crate::icons::TIMING,
            Command::Credits => crate::icons::CREDITS,
            Command::ToggleNavigator => crate::icons::NAVIGATOR,
            Command::ToggleQuickBrushes => crate::icons::QUICK_BRUSHES,
            Command::Settings => crate::icons::SETTINGS,
            Command::EditBrush => crate::icons::EDIT_BRUSH,
            Command::SavePreset => crate::icons::SAVE,
            Command::Transform => crate::icons::TRANSFORM,
            Command::FloatSelection => crate::icons::FLOAT,
            Command::FillSelection => crate::icons::PAINT_BUCKET,
            Command::GradientFill => crate::icons::GRADIENT,
            Command::AddLayer => crate::icons::ADD_LAYER,
            Command::AddFrame => crate::icons::ADD_FRAME,
            Command::AddPerspective => crate::icons::ADD_LAYER,
            // The dismissal mark every panel header wears, and the tick every
            // Done chip does — the two acts these commands are the names of.
            Command::CancelMode => crate::icons::CLOSE,
            Command::FinishMode => crate::icons::DONE,
            // The bar's own three marks, which are a picture of the question:
            // one sheet, a sheet over what is under it, a stack.
            Command::SetPickScope(scope) => match scope {
                PickScope::ThisLayer => crate::icons::ONE_LAYER,
                PickScope::AndBelow => crate::icons::AND_BELOW,
                PickScope::AllLayers => crate::icons::ALL_LAYERS,
            },
            // The mark its own title bar wears, so the menu and the palette
            // both stay a picture of the stack.
            Command::TogglePanel(id) => id.glyph(),
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
            Command::FloatSelection => "Float",
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
            Command::ToggleHdr => &["High dynamic range", "SDR preview", "Preview export"],
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
            // Photoshop's spelling of the cut half, and the plainer phrase a
            // hand that has never seen "float" would try first.
            Command::FloatSelection => &["Layer via cut", "Cut to layer"],
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
            Command::ToggleHdr => {
                "Show the light above white, or the canvas as an export will look (§6.5)"
            }
            Command::BrushSmaller => "Step the brush size down",
            Command::BrushLarger => "Step the brush size up",
            Command::EditBrush => "Open the full brush editor",
            Command::SavePreset => "Save the current brush as a preset",
            Command::Transform => "Move, scale, flip, perspective or warp the selected paint (§16)",
            Command::FloatSelection => {
                "Cut the selected paint into a floating layer of its own \u{2014} \
                 move it, paint on it, merge it back down (§16.12)"
            }
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

    /// Whether a control for this command is **live**, read off the engine's own
    /// projection — so a greyed row and a refused act cannot disagree.
    ///
    /// A rule over [`ObservableState`] rather than over a frontend's state, which is
    /// what lets it travel: `active` is the one that could not (see the module note).
    pub fn enabled(self, o: Option<&ObservableState>) -> bool {
        match self {
            Command::Undo => o.is_some_and(|o| o.can_undo),
            // The transfer is the surface's whether or not the switch is on (a
            // frontend flips only the headroom), so a surface with no range above
            // white has nothing to switch — a wide *gamut* is not range (§6.5).
            Command::ToggleHdr => o.is_some_and(|o| o.output.transfer().is_hdr()),
            Command::Redo => o.is_some_and(|o| o.can_redo),
            Command::Deselect
            | Command::InvertSelection
            | Command::Transform
            | Command::FloatSelection
            | Command::FillSelection
            | Command::GradientFill => o.is_some_and(|o| o.has_selection),
            _ => true,
        }
    }
}

/// A hint with its chord in parentheses — what a tooltip reads.
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

    /// A keystroke that types `s`, at position `code`.
    fn key(s: &str, code: &'static str) -> Keystroke<'static> {
        Keystroke::new(s.chars().next(), code)
    }

    /// A keystroke that types nothing, named by its position alone.
    fn bare(code: &'static str) -> Keystroke<'static> {
        Keystroke::new(None, code)
    }

    fn mods(ctrl: bool, shift: bool, alt: bool) -> Mods {
        Mods { ctrl, shift, alt }
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
            stock().lookup(&key("z", "KeyZ").with(mods(true, false, false))),
            Some(Command::Undo)
        );
        assert_eq!(
            stock().lookup(&key("Z", "KeyZ").with(mods(true, true, false))),
            Some(Command::Redo)
        );
        // Ctrl+Shift+Y is nobody's: an unclaimed chord falls through to the
        // browser rather than being Ctrl+Y plus a bystander.
        assert_eq!(
            stock().lookup(&key("Y", "KeyY").with(mods(true, true, false))),
            None
        );
    }

    #[test]
    fn escape_and_enter_are_rows() {
        // The two bare-key rows (MODAL_DESIGN.md). Spatial (`code`), so a
        // layout that types something exotic on them changes nothing.
        assert_eq!(
            stock().lookup(&bare("Escape").with(mods(false, false, false))),
            Some(Command::CancelMode)
        );
        assert_eq!(
            stock().lookup(&bare("Enter").with(mods(false, false, false))),
            Some(Command::FinishMode)
        );
        // With a modifier held they are nobody's: Ctrl+Enter is not Enter plus
        // a bystander, exactly as every chord row reads its modifiers.
        assert_eq!(
            stock().lookup(&bare("Enter").with(mods(true, false, false))),
            None
        );
    }

    #[test]
    fn caps_lock_is_nobodys() {
        // CapsLock types "Z" with no Shift held; the chord reads the modifier,
        // never the letter's case.
        assert_eq!(
            stock().lookup(&key("Z", "KeyZ").with(mods(true, false, false))),
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
            b.lookup(&key("g", "KeyG").with(mods(false, false, true))),
            Some(Command::GradientFill)
        );
        // Exact in all three modifiers, as every row is: the same key bare,
        // under the accelerator, or under both is not this row.
        assert_eq!(
            b.lookup(&key("g", "KeyG").with(mods(false, false, false))),
            None
        );
        assert_eq!(
            b.lookup(&key("g", "KeyG").with(mods(true, false, false))),
            None
        );
        assert_eq!(
            b.lookup(&key("g", "KeyG").with(mods(true, false, true))),
            None
        );
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
        assert_eq!(
            stock().lookup(&key("h", "KeyH").with(mods(false, false, true))),
            None
        );
        assert_eq!(
            stock().lookup(&key("z", "KeyZ").with(mods(true, false, true))),
            None
        );
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
                b.lookup(&key(letter, code).with(mods(false, false, true))),
                Some(Command::SetPickScope(scope))
            );
            // Exact in all three modifiers, as every row is — and the bare
            // letter matters here: `a` under the accelerator is Deselect, and
            // Alt is what tells the two apart.
            assert_eq!(
                b.lookup(&key(letter, code).with(mods(false, false, false))),
                None
            );
            assert_eq!(
                b.lookup(&key(letter, code).with(mods(true, false, true))),
                None
            );
        }
        assert_eq!(
            b.lookup(&key("a", "KeyA").with(mods(true, false, false))),
            Some(Command::Deselect)
        );
        // Spatial, so the reach is the same key on every layout: a Mac's
        // Option+Q types `œ` and still means all layers.
        assert_eq!(
            b.lookup(&key("\u{153}", "KeyQ").with(mods(false, false, true))),
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
            b.lookup(&key("q", "KeyQ").with(mods(true, false, true))),
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
            stock().lookup(&key("ü", "BracketLeft").with(mods(false, false, false))),
            Some(Command::BrushSmaller)
        );
        assert_eq!(
            stock().lookup(&key("]", "BracketRight").with(mods(false, false, false))),
            Some(Command::BrushLarger)
        );
        assert_eq!(
            stock().lookup(&key("[", "BracketLeft").with(mods(true, false, false))),
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
            b.lookup(&key("s", "KeyS").with(mods(true, false, false))),
            Some(Command::SaveDocument)
        );
        // The rest of the table stands exactly as shipped.
        assert_eq!(
            b.lookup(&key("z", "KeyZ").with(mods(true, false, false))),
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
            b.lookup(&key("z", "KeyZ").with(mods(true, false, false))),
            Some(Command::SaveDocument)
        );
        assert_eq!(Command::Undo.shortcut(&b), None);
        // Redo loses only the stolen spelling: Ctrl+Y was untouched, so a
        // command with a second default falls back to it in the same way.
        b.rebind(Command::Settings, chord(true, false, 'y'));
        assert_eq!(Command::Redo.shortcut(&b).as_deref(), Some("Ctrl+Shift+Z"));
        assert_eq!(
            b.lookup(&key("Z", "KeyZ").with(mods(true, true, false))),
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
            b.lookup(&key("q", "KeyQ").with(mods(true, false, false))),
            Some(Command::Redo)
        );
        assert_eq!(Command::Undo.shortcut(&b), None);
        // And the chords the overrides replaced are dead, not inherited: an
        // override is the command's whole binding.
        assert_eq!(
            b.lookup(&key("z", "KeyZ").with(mods(true, false, false))),
            None
        );
        assert_eq!(
            b.lookup(&key("y", "KeyY").with(mods(true, false, false))),
            None
        );
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
            capture(&key("s", "KeyS").with(mods(true, false, false))),
            Capture::Chord(chord(true, false, 's'))
        );
        // Shift's uppercase is the chord's Shift bit, not the letter's case.
        assert_eq!(
            capture(&key("M", "KeyM").with(mods(false, true, false))),
            Capture::Chord(chord(false, true, 'm'))
        );
        // A key that types nothing is named by its position.
        assert_eq!(
            capture(&bare("F5").with(mods(false, false, false))),
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
            capture(&key("g", "KeyG").with(mods(false, false, true))),
            Capture::Chord(alt_chord(false, false, "KeyG"))
        );
        assert_eq!(
            capture(&key("\u{a9}", "KeyG").with(mods(false, false, true))),
            Capture::Chord(alt_chord(false, false, "KeyG"))
        );
        assert_eq!(
            capture(
                &bare("Escape")
                    .with(mods(false, false, false))
                    .as_role(Role::Escape)
            ),
            Capture::Cancel
        );
        // Backspace is the capture's own eraser, so it can never be a chord.
        assert_eq!(
            capture(
                &bare("Backspace")
                    .with(mods(false, false, false))
                    .as_role(Role::Backspace)
            ),
            Capture::Clear
        );
    }

    #[test]
    fn unbind_erases_rather_than_resets() {
        let mut b = stock();
        b.unbind(Command::Undo);
        assert_eq!(Command::Undo.shortcut(&b), None);
        assert_eq!(
            b.lookup(&key("z", "KeyZ").with(mods(true, false, false))),
            None
        );
        // The chord is anyone's to take now…
        b.rebind(Command::SaveDocument, chord(true, false, 'z'));
        assert_eq!(
            b.lookup(&key("z", "KeyZ").with(mods(true, false, false))),
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
            capture(
                &bare("ControlLeft")
                    .with(mods(true, false, false))
                    .as_role(Role::Modifier)
            ),
            Capture::Pending
        );
        // Ctrl+Alt *is* capturable — the user asking for it is the whole
        // difference from shipping it (`no_default_chord_is_ctrl_alt`) — and
        // it comes back spatial like every Alt chord, which is what keeps a
        // Polish layout's AltGr+X from being stored as `Char('x')`… or worse,
        // as the `ń` it actually typed.
        assert_eq!(
            capture(&key("x", "KeyX").with(mods(true, false, true))),
            Capture::Chord(alt_chord(true, false, "KeyX"))
        );
        // Space is a hold that owns both edges of its key, and `input` claims
        // it off the key itself — so it is refused under every modifier, not
        // only where the character survives to be read.
        assert_eq!(
            capture(&key(" ", "Space").with(mods(false, false, false))),
            Capture::Pending
        );
        assert_eq!(
            capture(&key(" ", "Space").with(mods(false, false, true))),
            Capture::Pending
        );
        // The bare digit row is the rack's hold (§18.1.8).
        assert_eq!(
            capture(&key("1", "Digit1").with(mods(false, false, false))),
            Capture::Pending
        );
        // …but a digit under the accelerator is nobody's hold — nor under Alt,
        // which the rack refuses too, so capture may not be stricter than the
        // hold it is deferring to.
        assert_eq!(
            capture(&key("1", "Digit1").with(mods(true, false, false))),
            Capture::Chord(chord(true, false, '1'))
        );
        assert_eq!(
            capture(&key("1", "Digit1").with(mods(false, false, true))),
            Capture::Chord(alt_chord(false, false, "Digit1"))
        );
        // Ctrl+V is the paste's, and a row would prevent_default it dead.
        assert_eq!(
            capture(&key("v", "KeyV").with(mods(true, false, false))),
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
            b.lookup(&key("r", "KeyR").with(mods(false, false, false))),
            Some(Command::SelectRect)
        );
        assert_eq!(
            b.lookup(&key("e", "KeyE").with(mods(false, false, false))),
            Some(Command::SelectEllipse)
        );
        assert_eq!(
            b.lookup(&key("l", "KeyL").with(mods(false, false, false))),
            Some(Command::SelectLasso)
        );
        // Bare letters, so an accelerator over one is nobody's — Ctrl+R must
        // stay the browser's reload rather than arming a marquee.
        assert_eq!(
            b.lookup(&key("r", "KeyR").with(mods(true, false, false))),
            None
        );
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
