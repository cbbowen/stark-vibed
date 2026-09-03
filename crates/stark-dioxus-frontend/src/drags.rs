//! The drag-binding table (§25): which chord+button opens which canvas drag
//! gesture — the pointer half of the command registry (`crate::commands`).
//! §25.3 is the checklist for adding an action, §25.8 for what the user may do
//! to the table once it is there.
//!
//! A [`Command`](crate::commands::Command) is an act asked for *whole*; a drag
//! action is a **gesture**, with a press that opens it, moves that feed it and a
//! release that ends it. The gestures themselves stay where they live — the
//! brush-tuning drag is `input::Tune` (§18.1.9), the eyedropper is
//! `input::pick_color` and the `PickState` flags (§18.0.2), the layer carry is
//! `input::PickMove` (§16.11) — because each owns its own lifecycle and state.
//! What this module owns is the one question the
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
//!   the reason [`DragAction::PickColor`] and
//!   [`DragAction::PickAndTranslate`] both decline there
//!   ([`claims`](DragAction::claims)). Between them those two chords *are* the
//!   marquee's Alt and Shift, which is not a collision to be resolved by
//!   picking other chords: they are the conventional bindings for both acts,
//!   and which one a press means is answered by the tool in hand.
//! - **The plain press.** Painting is not a bound action, it is what an
//!   unclaimed press *is* — the resting meaning every unbound chord falls
//!   through to, as an unclaimed keystroke falls through to the browser.
//!
//! # The user's table over the shipped one
//!
//! The variant, not the chord, is an action's identity — two rows may name one
//! act, and a rebinding moves the chord while the name holds still. [`defaults`]
//! is the table Stark ships; [`DragBindings`] is that table with this browser's
//! own rows laid over it, and it has the same two readers the shipped table had
//! ([`find`] on the press, [`armed`] on the advertisement), because those are
//! the only two questions anybody asks it.
//!
//! The shape is `commands::Bindings`' — overrides over defaults, keyed on the
//! variant's name, stored per browser — **copied rather than shared**, which
//! §25.3 anticipated. The two tables agree on nothing but that shape: a keyboard
//! chord is a key, a drag chord is a button, the defaults differ, and the
//! captures are a keystroke and a press. A trait generic over both would be two
//! associated types and a blanket impl to save forty lines of the plainest code
//! in either module.
//!
//! # Presets, and the one time they are offered
//!
//! Three modifier drags is three secrets, and every app an artist might arrive
//! from keeps them somewhere else. [`DragPreset`] is the whole answer: a named
//! table per app, taken in one click, listed in ⚙ Settings and offered
//! **once** — the first time a press asks this table for a chord nothing is
//! bound to ([`Offer`]). That press is somebody reaching for a binding they
//! already have in their hands, which is the one moment the offer is worth
//! making and the only evidence Stark will ever get that it should be made.

use dioxus::html::Modifiers;
use dioxus::html::input_data::MouseButton;
use dioxus::prelude::*;
use serde::{Deserialize, Serialize};

use crate::icons::{self, icon};
use crate::input::{accel, is_contact};
use crate::state::AppState;
use crate::widgets::Modal;
use stark_chrome::storage::{Entry, Store};

/// The modifier half of a drag chord — and, held in
/// [`Signals::held_mods`](crate::state::Signals::held_mods)(crate::state::Signals::held_mods), the modifiers
/// currently down, which is what the resting cursor asks the table with
/// ([`armed`]).
///
/// `#[serde(default)]` for the store's reason (§25.6): a fourth modifier added
/// one day has to read as absent out of every chord written before it existed.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
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

    /// Whether nothing is held. A press with no modifier on the primary button
    /// is never a chord this table can name — that press *is* painting — which
    /// is what [`capture`] spends it on.
    fn bare(self) -> bool {
        self == Self::default()
    }
}

/// Which button a drag binding means, named the way the hand knows it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
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

impl DragButton {
    /// How a chord spells this button ([`chord_label`]). The primary button is
    /// unnamed, because "drag" with nothing said about the button is what every
    /// app's own documentation means by it.
    fn word(self) -> &'static str {
        match self {
            DragButton::Left => "drag",
            DragButton::Right => "right-drag",
        }
    }
}

/// The press half of a drag binding: which modifier tier, on which button.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct DragChord {
    pub mods: Mods,
    pub button: DragButton,
}

/// A chord spelled out for the chrome — a settings row's chip, a preset card's
/// line.
///
/// "Ctrl" names the accelerator on every platform, as `commands::chord_label`
/// does and for its reason: which glyph a Mac draws is a presentation question,
/// not a binding one.
pub fn chord_label(chord: DragChord) -> String {
    let mut s = String::new();
    for (held, name) in [
        (chord.mods.ctrl, "Ctrl"),
        (chord.mods.shift, "Shift"),
        (chord.mods.alt, "Alt"),
    ] {
        if held {
            s.push_str(name);
            s.push_str(" + ");
        }
    }
    s.push_str(chord.button.word());
    s
}

/// One gesture a bound press can open. The closed set the table maps into —
/// each variant is a gesture the canvas already knows how to drive, so adding
/// one is a variant, a row in [`defaults`], and an arm in the canvas's press
/// handler; the routing itself never grows another case.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum DragAction {
    /// The brush-tuning drag (`input::Tune`, §18.1.9): Size sideways, Flow up
    /// and down, under the hand that is already on the painting.
    TuneBrush,
    /// The eyedropper (`input::pick_color`, §18.0.2): the press samples, and
    /// the drag keeps sampling, so a color is picked up without putting the
    /// brush down.
    PickColor,
    /// Pick the layer under the press and carry it (`input::PickMove`,
    /// §16.11): the press auto-selects the topmost layer showing paint where it
    /// landed, and the drag translates that layer's selected paint. A tap is
    /// the selection alone — which is the same gesture stopped early, not a
    /// second binding.
    ///
    /// The Move tool's auto-select, which is what a hand arriving from
    /// Photoshop or Clip Studio Paint already knows; what it does *not* borrow
    /// is their tool-shaped framing, since a mode you have to enter and leave
    /// is exactly what a chord is for.
    PickAndTranslate,
}

/// The drag table Stark ships with — [`DragPreset::Stark`]'s rows, read the way
/// the press path wants them.
///
/// The shipped table *is* a preset, rather than a list beside one: they would
/// otherwise be two statements of one fact, and the day they disagreed the ⚙
/// dialog would light "Stark" on a table Stark does not ship. Rows are disjoint
/// by construction — the chords are exact, so no row can shadow another
/// (`tests::default_rows_are_disjoint`) — and there is no order to carry
/// meaning.
fn defaults() -> impl Iterator<Item = (DragChord, DragAction)> {
    DragPreset::Stark
        .rows()
        .iter()
        .map(|&(action, chord)| (chord, action))
}

/// The drag table as this browser has it: [`defaults`] with the user's own rows
/// laid over them. Lives on [`Signals::drags`](crate::state::Signals::drags)(crate::state::Signals::drags)
/// as a signal, so a settings row and the resting cursor both move the moment a
/// rebind lands.
///
/// Only the *overrides* are state, exactly as `commands::Bindings` holds only
/// its own: an action the user never touched keeps its default row, minus any
/// chord an override has claimed — so a default moved in a later build reaches a
/// browser that stored this table before the move.
#[derive(Clone, Default, PartialEq, Debug)]
pub struct DragBindings {
    /// The actions the user has taken over, each with the chord that is now its
    /// **whole** binding, or `None` for an action whose chord a later rebind
    /// stole or whose binding was erased outright.
    overrides: Vec<(DragAction, Option<DragChord>)>,
}

impl DragBindings {
    /// Whether an override has claimed `chord` — the question that kills a
    /// default row: the user's table always wins over the shipped one.
    fn taken(&self, chord: DragChord) -> bool {
        self.overrides.iter().any(|(_, c)| *c == Some(chord))
    }

    /// Whether the user has taken this action's binding over (even to nothing).
    fn overridden(&self, action: DragAction) -> bool {
        self.overrides.iter().any(|(a, _)| *a == action)
    }

    /// The chord the chrome should advertise for `action`: its override, or its
    /// default row if that row is still standing. `None` is an act no press can
    /// reach — never bound, or rebound away.
    pub fn of(&self, action: DragAction) -> Option<DragChord> {
        if let Some((_, chord)) = self.overrides.iter().find(|(a, _)| *a == action) {
            return *chord;
        }
        defaults()
            .find(|(chord, a)| *a == action && !self.taken(*chord))
            .map(|(chord, _)| chord)
    }

    /// The action a chord asks for, if any — the policy half, taken apart from
    /// the DOM event the way `Bindings::lookup` is so a test can reach it.
    ///
    /// Overrides answer first, and can never disagree with each other
    /// ([`rebind`](Self::rebind) steals a colliding chord); a default row
    /// answers only while its action is untouched and its chord unclaimed.
    fn lookup(&self, mods: Mods, button: DragButton) -> Option<DragAction> {
        let chord = DragChord { mods, button };
        if let Some((action, _)) = self.overrides.iter().find(|(_, c)| *c == Some(chord)) {
            return Some(*action);
        }
        defaults()
            .find(|(c, action)| !self.overridden(*action) && !self.taken(*c) && *c == chord)
            .map(|(_, action)| action)
    }

    /// Give `action` the chord as its whole binding. If another row held that
    /// chord it is stolen — the loser keeps an override saying so, which is
    /// what its settings row then shows — and a default elsewhere dies by
    /// [`taken`](Self::taken) with nothing stored about it.
    pub fn rebind(&mut self, action: DragAction, chord: DragChord) {
        for (other, held) in &mut self.overrides {
            if *other != action && *held == Some(chord) {
                *held = None;
            }
        }
        self.set(action, Some(chord));
    }

    /// Take `action`'s binding away — deliberately, which is why it is an
    /// override to nothing rather than a removed entry: the defaults must not
    /// resurrect a chord the user just erased.
    pub fn unbind(&mut self, action: DragAction) {
        self.set(action, None);
    }

    fn set(&mut self, action: DragAction, chord: Option<DragChord>) {
        match self.overrides.iter_mut().find(|(a, _)| *a == action) {
            Some((_, held)) => *held = chord,
            None => self.overrides.push((action, chord)),
        }
    }

    /// Take `preset`'s table whole.
    ///
    /// The shipped preset is applied by **clearing** the overrides rather than
    /// by writing its three rows out: a browser storing three rows that say
    /// what Stark already does is a browser no later build could ever move
    /// (§25.8). Every other preset is stored in full, for the mirror-image
    /// reason — it is a claim about *another* app's table, and a default that
    /// moves must not move it.
    fn take(&mut self, preset: DragPreset) {
        self.overrides.clear();
        if preset == DragPreset::Stark {
            return;
        }
        self.overrides = DragAction::ALL
            .iter()
            .map(|&action| (action, preset.chord(action)))
            .collect();
    }
}

/// One stored row: an action and the chord that is now its whole binding — or
/// none, for an action whose chord a later rebind stole — or the mark saying
/// this browser has been offered the presets ([`Offer`]).
///
/// **Untagged, and two kinds of row in one record**, which is `tutor::Row`'s
/// arrangement and its argument: the table and whether its offer has been made
/// are one feature's worth of browser-local state, and splitting them would put
/// a second key in the registry to hold one bit. A list rather than one record,
/// so a row today's build cannot make sense of — a binding for a retired action
/// — costs its own row and not the table (`storage::load_list`).
#[derive(Serialize, Deserialize)]
#[serde(untagged)]
pub(crate) enum DragRow {
    Bound {
        action: DragAction,
        #[serde(default)]
        chord: Option<DragChord>,
    },
    Offered {
        offered: bool,
    },
}

impl Entry for DragRow {
    const STORE: Store = Store::Drags;
}

/// Whether this browser has been shown the preset offer, and whether one is
/// waiting to be shown (§25.8).
///
/// Three states rather than two bools, because the middle one is a *transition*
/// the other two exclude: an offer comes due on a press and is made on the
/// release after it, and neither "not yet" nor "already" can be true meanwhile.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Offer {
    /// Never made, and nothing is waiting.
    #[default]
    Unoffered,
    /// A press asked the table for a chord nothing binds; the dialog is waiting
    /// for the hand to come off the canvas ([`settle_offer`]).
    Due,
    /// Made. Once ever, per browser — the settings dialog is the way back.
    Offered,
}

/// The action a **left press** under `held` would open — what the resting
/// cursor and the options bar advertise ([`armed`]'s callers), asked of the same
/// table the press will ask, so the promise and the press cannot disagree.
///
/// Takes both the table and the held triple as values rather than reading
/// either, because whether a call site subscribes or peeks is the call site's
/// own discipline — which is `Command::shortcut(&bindings)`'s bargain too.
pub fn armed(bindings: &DragBindings, held: Mods) -> Option<DragAction> {
    bindings.lookup(held, DragButton::Left)
}

/// Which button `e` presses, as a chord names buttons — `None` for one no chord
/// can hold: the middle button is the pan's, and a hold is not a row (§25.3).
fn button_of(e: &Event<PointerData>) -> Option<DragButton> {
    if is_contact(e) {
        Some(DragButton::Left)
    } else if e.trigger_button() == Some(MouseButton::Secondary) {
        Some(DragButton::Right)
    } else {
        None
    }
}

/// The action `e` asks for, if any — the one reader on the canvas's press path.
/// A row that matches may still decline the *press*
/// ([`claims`](DragAction::claims)), and a declined press falls through to the
/// paint path exactly as an unbound chord does: over a selection tool, Alt+drag
/// *is* the subtract marquee.
///
/// It also **notices** a modified press the table has nothing for, which is what
/// brings the preset offer due ([`Offer`]). Here rather than in the canvas's
/// ladder because it is the table's own observation — "somebody reached for a
/// binding I do not have" — and because this is the one place that has already
/// asked the question.
pub fn find(state: AppState, e: &Event<PointerData>) -> Option<DragAction> {
    let button = button_of(e)?;
    let mods = Mods::of(e.modifiers());
    let found = state.drags.peek().lookup(mods, button);
    if found.is_none() {
        note_unbound(state, mods);
    }
    // Asked of the *action*, not folded into the line above: a bound chord that
    // declines is not an unbound one, and offering a table of presets to
    // somebody whose Shift is busy being the union marquee would be answering a
    // question they did not ask.
    found.filter(|action| action.claims(state))
}

/// A modified press landed on a chord this table does not bind: bring the preset
/// offer due, if this browser has never had it.
///
/// **Modified only.** A bare contact is painting, and a bare right press is a
/// chord nobody arrives with from anywhere, so neither is evidence of a hand
/// reaching for a binding it knows from another app.
fn note_unbound(state: AppState, mods: Mods) {
    let mut offer = state.drag_offer;
    if mods.bare() || *offer.peek() != Offer::Unoffered {
        return;
    }
    offer.set(Offer::Due);
}

/// Show an offer that has come due, now the canvas is out of the artist's hand —
/// called from [`end_interaction`](crate::input::end_interaction), which is
/// where every canvas gesture is put down.
///
/// Due and shown are two steps for `tutor`'s reason: the press that brings this
/// due goes on to paint a stroke, and a modal over a live stroke would take the
/// canvas away mid-mark. Marked offered on the way *up*, so it is one offer
/// whatever the answer is — including no answer at all, which is what dismissing
/// it is.
pub fn settle_offer(state: AppState) {
    let mut offer = state.drag_offer;
    if *offer.peek() != Offer::Due {
        return;
    }
    offer.set(Offer::Offered);
    save(state);
    let mut showing = state.dialogs.drag_presets;
    showing.set(true);
}

impl DragAction {
    /// Every action, which is what the settings rows and the presets are written
    /// against. By hand, and `tests::all_is_every_action` is what notices a
    /// variant left out of it.
    pub const ALL: &'static [DragAction] = &[
        DragAction::TuneBrush,
        DragAction::PickColor,
        DragAction::PickAndTranslate,
    ];

    /// What the act is called where it has a row to itself — the settings
    /// dialog's label.
    pub fn name(self) -> &'static str {
        match self {
            DragAction::TuneBrush => "Brush size and flow",
            DragAction::PickColor => "Pick up a color",
            DragAction::PickAndTranslate => "Move a layer",
        }
    }

    /// The terse word for a surface that has already said what it is listing — a
    /// preset card, whose three lines are three drags and nothing else. The
    /// split is `Command::word`'s, and exists so both spellings are the
    /// registry's rather than a call site's.
    pub fn word(self) -> &'static str {
        match self {
            DragAction::TuneBrush => "Brush size",
            DragAction::PickColor => "Color",
            DragAction::PickAndTranslate => "Move layer",
        }
    }

    /// The one sentence a settings row owes: what the drag does, in the terms
    /// the hand has (`crate::settings`' rule for a row).
    pub fn hint(self) -> &'static str {
        match self {
            DragAction::TuneBrush => {
                "Drag right and left for size, up and down for flow, without leaving the painting."
            }
            DragAction::PickColor => {
                "Sample the color under the pointer, and keep sampling as the drag moves."
            }
            DragAction::PickAndTranslate => {
                "Pick up whichever layer shows paint under the press, and carry it."
            }
        }
    }

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
            // The same two stand-downs, arrived at from the other side. Over a
            // selection tool **Shift** is the union marquee (§6.8) — the chord
            // this action wears is the marquee's own combine modifier there, and
            // a gesture that is what the press is *for* outranks one that
            // reaches past it. And this one *commits*: the ladder's playback
            // guard sits below the table (§25.4), so an action that would lay an
            // undo step down has to refuse the playhead itself.
            DragAction::PickAndTranslate => {
                !crate::panels::select::current_tool(state).is_selection()
                    && !crate::panels::timeline::is_playing(state)
            }
        }
    }

    /// Whether a press under this binding takes the canvas away from the brush
    /// — so the chrome that *promises* paint stands down while the chord is
    /// held: the brush circle, and the hover mark under it (§18.1.10).
    ///
    /// A property of the act rather than a list kept at each of those call
    /// sites, for [`claims`](Self::claims)' reason. The mark is the sharper half
    /// of the bill: it is a preview folded into the shown document, so a press
    /// that *reads* the canvas back would read the hypothesis as paint — which
    /// is a wrong color for the eyedropper and a wrong layer for the hit test.
    pub fn shadows_paint(self) -> bool {
        match self {
            // Tuning *is* about the brush, and draws its own picture of it (the
            // size ring at the press, §18.1.9): the circle and the mark are both
            // still telling the truth about what the brush will do.
            DragAction::TuneBrush => false,
            DragAction::PickColor | DragAction::PickAndTranslate => true,
        }
    }
}

/// A named table: what one app puts these three drags on (§25.8).
///
/// Indexed by **the app somebody is arriving from**, not by distinct tables —
/// which is why Clip Studio Paint and Corel Painter both have a row although
/// they agree on all three. A preset is picked by recognising a name, and a list
/// that had merged them would offer neither.
///
/// The tables are each app's defaults for these three gestures, on Windows, with
/// the accelerator standing in for Ctrl and Command alike (`input::accel`). They
/// are a starting point rather than a fidelity claim, and the surfaces say so:
/// every row stays separately rebindable the moment a preset lands, which is the
/// only honest shape for a table transcribed out of somebody else's manual.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DragPreset {
    /// What Stark ships — and therefore stored as no overrides at all
    /// ([`DragBindings::take`]).
    Stark,
    Photoshop,
    ClipStudio,
    Rebelle,
    Krita,
}

/// One chord, spelled as the preset tables want to read: `chord(ctrl, shift,
/// alt, button)`.
const fn chord(ctrl: bool, shift: bool, alt: bool, button: DragButton) -> DragChord {
    DragChord {
        mods: Mods { ctrl, shift, alt },
        button,
    }
}

impl DragPreset {
    /// Every preset, in the order both surfaces list them: Stark first, because
    /// it is the table the app starts on and the one a reader has to recognise
    /// as "what I have now".
    pub const ALL: &'static [DragPreset] = &[
        DragPreset::Stark,
        DragPreset::Photoshop,
        DragPreset::ClipStudio,
        DragPreset::Rebelle,
        DragPreset::Krita,
    ];

    /// The app's name, as its own users spell it.
    pub fn name(self) -> &'static str {
        match self {
            DragPreset::Stark => "Stark",
            DragPreset::Photoshop => "Photoshop",
            DragPreset::ClipStudio => "Clip Studio Paint",
            DragPreset::Rebelle => "Rebelle",
            DragPreset::Krita => "Krita",
        }
    }

    /// The sentence a chip carries as its tooltip and a card prints under its
    /// name: what is *distinctive* about this table, rather than a restatement
    /// of the three rows the card already lists beside it.
    pub fn blurb(self) -> &'static str {
        match self {
            DragPreset::Stark => {
                "One bare modifier each, with the accelerator on the drag that changes no pixels."
            }
            DragPreset::Photoshop => {
                "The brush is resized with the right button, and the accelerator reaches the \
                 Move tool without putting the brush down."
            }
            DragPreset::ClipStudio => "The brush is resized on the accelerator with Alt.",
            DragPreset::Rebelle => "The brush is resized with the right button alone.",
            DragPreset::Krita => {
                "Shift resizes and the accelerator samples. Krita moves a layer with a tool \
                 rather than with a drag, so that row is left unbound."
            }
        }
    }

    /// This preset's rows. An action absent from the list is **unbound** under
    /// it — a real answer, and Krita's: an app that reaches an act through a
    /// tool rather than a modifier should not have a modifier invented for it.
    pub fn rows(self) -> &'static [(DragAction, DragChord)] {
        use DragAction::{PickAndTranslate, PickColor, TuneBrush};
        use DragButton::{Left, Right};
        // Const tables rather than lists built per call: `defaults()` reads
        // Stark's row out of here, and it is on the press path.
        const STARK: &[(DragAction, DragChord)] = &[
            (TuneBrush, chord(true, false, false, Left)),
            (PickColor, chord(false, false, true, Left)),
            (PickAndTranslate, chord(false, true, false, Left)),
        ];
        const PHOTOSHOP: &[(DragAction, DragChord)] = &[
            (TuneBrush, chord(false, false, true, Right)),
            (PickColor, chord(false, false, true, Left)),
            (PickAndTranslate, chord(true, false, false, Left)),
        ];
        const CLIP_STUDIO: &[(DragAction, DragChord)] = &[
            (TuneBrush, chord(true, false, true, Left)),
            (PickColor, chord(false, false, true, Left)),
            (PickAndTranslate, chord(true, false, false, Left)),
        ];
        const REBELLE: &[(DragAction, DragChord)] = &[
            (TuneBrush, chord(false, false, false, Right)),
            (PickColor, chord(false, false, true, Left)),
            (PickAndTranslate, chord(true, false, false, Left)),
        ];
        const KRITA: &[(DragAction, DragChord)] = &[
            (TuneBrush, chord(false, true, false, Left)),
            (PickColor, chord(true, false, false, Left)),
        ];
        match self {
            DragPreset::Stark => STARK,
            DragPreset::Photoshop => PHOTOSHOP,
            DragPreset::ClipStudio => CLIP_STUDIO,
            DragPreset::Rebelle => REBELLE,
            DragPreset::Krita => KRITA,
        }
    }

    /// What this preset binds `action` to, or `None` where it leaves it unbound.
    pub fn chord(self, action: DragAction) -> Option<DragChord> {
        self.rows()
            .iter()
            .find(|(a, _)| *a == action)
            .map(|(_, chord)| *chord)
    }

    /// Whether `bindings` *is* this preset's table — asked per chip rather than
    /// answered once, so two presets that agree both light up. Which is the
    /// truth: the table is what both apps do, and lighting only the first would
    /// make clicking the second look like it had done nothing.
    pub fn matches(self, bindings: &DragBindings) -> bool {
        DragAction::ALL
            .iter()
            .all(|&action| bindings.of(action) == self.chord(action))
    }
}

/// What a press means to a rebinding capture ([`DragBindingRow`]).
///
/// The pointer's counterpart to `commands::Capture`, and shorter by one arm:
/// erasing a binding is a control of its own here, because a press cannot say
/// "nothing" the way Backspace can.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DragCapture {
    /// A bindable chord — commit it.
    Chord(DragChord),
    /// The capture is called off, the binding left as it was.
    ///
    /// This is what a **bare contact** is spent on, and it is the choice
    /// `commands::capture` makes with Escape: the one press that can never be a
    /// binding is the one that can be the way out. A bare left press *is*
    /// painting (§25.3), so it was never a chord this table could hold, and a
    /// user who clicks the chip and thinks better of it clicks it again.
    Cancel,
    /// Nothing to make of it yet — the middle button, which is the pan's and no
    /// chord's. The capture keeps listening.
    Pending,
}

/// What a press with `mods` on `button` asks a capture for — the policy half,
/// taken apart from the DOM event so a test can reach it.
///
/// It records **what the browser reported**, which is the only thing that keeps
/// a capture and a press agreeing. The one place that shows: on a Mac, Ctrl and
/// the trackpad arrive as the *secondary* button (`input::accel` says so from
/// the other side), so a chord captured that way is stored and labelled
/// `Ctrl + right-drag`. The binding still works — the canvas press reports the
/// same thing — and the alternative, guessing that a secondary press with Ctrl
/// "really" meant the primary one, would bind a chord the press then misses.
pub fn capture(mods: Mods, button: Option<DragButton>) -> DragCapture {
    match button {
        Some(DragButton::Left) if mods.bare() => DragCapture::Cancel,
        Some(button) => DragCapture::Chord(DragChord { mods, button }),
        None => DragCapture::Pending,
    }
}

/// Seed [`Signals::drags`](crate::state::Signals::drags)(crate::state::Signals::drags) and the offer's state
/// from this browser's stored rows. Called once at app start, beside
/// `commands::load`.
pub fn load(state: AppState) {
    let Some(rows) = stark_chrome::storage::load_list::<DragRow>() else {
        return;
    };
    let mut overrides = Vec::new();
    let mut offered = false;
    for row in rows {
        match row {
            DragRow::Bound { action, chord } => overrides.push((action, chord)),
            DragRow::Offered { offered: seen } => offered |= seen,
        }
    }
    let mut bindings = state.drags;
    bindings.set(DragBindings { overrides });
    if offered {
        let mut offer = state.drag_offer;
        offer.set(Offer::Offered);
    }
}

/// Give `action` the captured chord, and persist the table — written through
/// [`DragBindings::rebind`] so a stolen chord and its victim's row change in the
/// same write.
pub fn rebind(state: AppState, action: DragAction, chord: DragChord) {
    edit(state, |b| b.rebind(action, chord));
}

/// Take `action`'s binding away and persist the table.
pub fn unbind(state: AppState, action: DragAction) {
    edit(state, |b| b.unbind(action));
}

/// Take `preset`'s whole table and persist it — one write, so its rows cannot be
/// seen half-applied.
pub fn set_preset(state: AppState, preset: DragPreset) {
    edit(state, |b| b.take(preset));
}

/// One change to the table, written to the signal and to storage as one act — so
/// what the rows show, what a press answers, and what the next visit loads
/// cannot be three states.
fn edit(state: AppState, change: impl FnOnce(&mut DragBindings)) {
    let mut bindings = state.drags;
    let mut next = bindings.peek().clone();
    change(&mut next);
    bindings.set(next);
    save(state);
}

/// Write the whole record: the override rows, then the offer's mark.
///
/// One writer for both, called by everything that changes either (§25.6's fourth
/// step) — a rebind, a preset, and the offer being made.
fn save(state: AppState) {
    let rows: Vec<DragRow> = state
        .drags
        .peek()
        .overrides
        .iter()
        .map(|&(action, chord)| DragRow::Bound { action, chord })
        .chain(std::iter::once(DragRow::Offered {
            offered: *state.drag_offer.peek() == Offer::Offered,
        }))
        .collect();
    stark_chrome::storage::save_list(&rows);
}

/// The ⚙ dialog's drag section (§25.8): the presets as a run of chips, then a
/// row per action carrying its chord as the door to changing it.
///
/// Mounted by `settings::SettingsModal` rather than written there, because this
/// is one feature's surface over one feature's table and the settings module's
/// business is the *dialog* — the same split that leaves `collab::SessionModal`
/// and `files::ExportModal` with their own modules.
#[component]
pub fn DragBindingSection() -> Element {
    let state = use_context::<AppState>();
    // The action whose chord is being recaptured, if any — armed by its own
    // chip, spent by the next press that chip hears. One at a time, like the
    // palette's, and held here rather than per row so arming a second row
    // disarms the first.
    let capturing: Signal<Option<DragAction>> = use_signal(|| None);
    let table = state.drags.read().clone();
    let known = DragPreset::ALL.iter().any(|p| p.matches(&table));

    rsx! {
        div { class: "setting-row",
            // The checkbox's column, kept empty so this row's text starts where
            // every other row's does (`settings::SettingChoice`).
            div { class: "setting-check-spacer" }
            div { class: "setting-text",
                div { class: "setting-label", "Start from another app" }
                div { class: "setting-desc",
                    "Take the three drags below from the app you already know. Each row can \
                     still be changed afterwards."
                }
                // `segmented`, because the chips answer one question and
                // picking one un-picks the rest (§25.9) — butted into a single
                // control the shape carries that, where chips standing
                // apart would promise switches that could be held down
                // together. They fit one line at the dialog's width, which is
                // the condition the rule turns on; a seventh preset is the cue
                // to re-measure and, failing that, to become a `.select`.
                div { class: "drag-presets segmented",
                    for preset in DragPreset::ALL.iter().copied() {
                        button {
                            key: "{preset:?}",
                            class: if preset.matches(&table) { "chip active" } else { "chip" },
                            r#type: "button",
                            title: "{preset.blurb()}",
                            onclick: move |_| set_preset(state, preset),
                            "{preset.name()}"
                        }
                    }
                }
                if !known {
                    div { class: "setting-note",
                        "Your own table \u{2014} pick an app above to start over from its drags."
                    }
                }
            }
        }
        for action in DragAction::ALL.iter().copied() {
            DragBindingRow { key: "{action:?}", action, capturing }
        }
    }
}

/// One action's row: what the drag does, and the chord that opens it as a chip
/// that captures a new one.
///
/// Click, then press the chord you want *with the button you want* — the same
/// gesture as using it, which is `main::BindChip`'s bargain restated for
/// presses. The press is read by [`capture`], so a plain click calls the capture
/// off and the ✕ beside the chip is what erases the binding.
#[component]
fn DragBindingRow(action: DragAction, capturing: Signal<Option<DragAction>>) -> Element {
    let state = use_context::<AppState>();
    let mut capturing = capturing;
    let listening = capturing() == Some(action);
    let bound = state.drags.read().of(action);

    let press = move |e: Event<PointerData>| {
        // The chip's press is the chip's alone, and never the browser's: a
        // right press here is a chord being captured rather than a context
        // menu, and a left one must not move focus off whatever has it.
        e.stop_propagation();
        e.prevent_default();
        if !listening {
            capturing.set(Some(action));
            return;
        }
        match capture(Mods::of(e.modifiers()), button_of(&e)) {
            DragCapture::Chord(chord) => {
                rebind(state, action, chord);
                capturing.set(None);
            }
            DragCapture::Cancel => capturing.set(None),
            DragCapture::Pending => {}
        }
    };

    rsx! {
        div { class: "setting-row",
            div { class: "setting-check-spacer" }
            // Not a `<label>`, unlike the toggle rows: there is no one control
            // for the text to stand in front of, and a click on the sentence
            // must not arm the capture beside it.
            div { class: "setting-text",
                div { class: "setting-label",
                    "{action.name()}"
                    span { class: "drag-chord",
                        if listening {
                            span {
                                class: "menu-shortcut bind-chip capturing",
                                title: "Hold the keys you want and press here \u{2014} a plain click keeps what was there",
                                onpointerdown: press,
                                "hold keys, press\u{2026}"
                            }
                        } else if let Some(chord) = bound {
                            span {
                                class: "menu-shortcut bind-chip",
                                title: "Click, then hold the keys you want and press again",
                                onpointerdown: press,
                                {chord_label(chord)}
                            }
                            button {
                                class: "drag-clear",
                                r#type: "button",
                                title: "Leave this act with no drag at all",
                                onclick: move |_| unbind(state, action),
                                {icon(icons::CLOSE)}
                            }
                        } else {
                            span {
                                class: "menu-shortcut bind-chip bind-add",
                                title: "Bind a drag: click, then hold the keys you want and press again",
                                onpointerdown: press,
                                {icon(icons::ADD)}
                            }
                        }
                    }
                }
                div { class: "setting-desc", "{action.hint()}" }
            }
        }
    }
}

/// The offer, made once (§25.8): the presets as cards, each printing the three
/// drags it would set.
///
/// Raised by [`settle_offer`] off the release after a modified press this table
/// had nothing for — so it arrives to somebody who has just reached for a
/// binding they know from elsewhere, and to nobody else. Dismissing it is an
/// answer, which is why the mark is written when the dialog is *shown*.
#[component]
pub fn DragPresetModal(on_close: EventHandler<()>) -> Element {
    let state = use_context::<AppState>();

    rsx! {
        // Wide for the settings dialog's reason: rows of three
        // drags each, and at the standard width every line of them wraps.
        Modal { class: "modal-wide", on_close,
            div { class: "modal-title", "Drags on the canvas" }
            div { class: "modal-subtitle",
                "You held a modifier and pressed, and Stark has nothing bound to that. If \
                 you have come from another app, start from the drags you already know."
            }

            div { class: "drag-preset-list",
                for preset in DragPreset::ALL.iter().copied() {
                    button {
                        key: "{preset:?}",
                        class: "drag-preset",
                        r#type: "button",
                        onclick: move |_| {
                            set_preset(state, preset);
                            on_close.call(());
                        },
                        div { class: "drag-preset-head",
                            span { class: "drag-preset-name", "{preset.name()}" }
                            span { class: "drag-preset-blurb", "{preset.blurb()}" }
                        }
                        div { class: "drag-preset-rows",
                            for action in DragAction::ALL.iter().copied() {
                                div { key: "{action:?}", class: "drag-preset-row",
                                    span { class: "drag-preset-act", "{action.word()}" }
                                    span { class: "menu-shortcut",
                                        match preset.chord(action) {
                                            Some(chord) => chord_label(chord),
                                            // An act this app reaches some
                                            // other way. Said with a dash
                                            // rather than left blank, so a
                                            // short card reads as a choice
                                            // and not as a card cut off.
                                            None => "\u{2014}".to_string(),
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            div { class: "drag-offer-note",
                "Whatever you pick, \u{2699} Settings lists these three drags and can \
                 change any of them at any time."
            }

            div { class: "modal-actions",
                button {
                    class: "btn btn-primary",
                    onclick: move |_| on_close.call(()),
                    {icon(icons::DONE)}
                    "Not now"
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn m(ctrl: bool, shift: bool, alt: bool) -> Mods {
        Mods { ctrl, shift, alt }
    }

    fn left(ctrl: bool, shift: bool, alt: bool) -> DragChord {
        chord(ctrl, shift, alt, DragButton::Left)
    }

    /// The stock table — what every browser has before its first rebind.
    fn stock() -> DragBindings {
        DragBindings::default()
    }

    #[test]
    fn default_rows_are_disjoint() {
        let rows: Vec<_> = defaults().collect();
        for (i, (a, _)) in rows.iter().enumerate() {
            for (b, _) in &rows[i + 1..] {
                assert_ne!(a, b, "two rows on one chord: the table is not a function");
            }
        }
    }

    #[test]
    fn the_shipped_bindings() {
        let b = stock();
        assert_eq!(
            b.lookup(m(true, false, false), DragButton::Left),
            Some(DragAction::TuneBrush)
        );
        assert_eq!(
            b.lookup(m(false, false, true), DragButton::Left),
            Some(DragAction::PickColor)
        );
        assert_eq!(
            b.lookup(m(false, true, false), DragButton::Left),
            Some(DragAction::PickAndTranslate)
        );
        // A bare press is not a row: painting is what an unbound press is,
        // not an act the table names.
        assert_eq!(b.lookup(m(false, false, false), DragButton::Left), None);
    }

    #[test]
    fn chords_are_exact() {
        let b = stock();
        // Ctrl+Alt is nobody's — not the Ctrl row with a bystander, exactly
        // as the keyboard table reads its modifiers.
        assert_eq!(b.lookup(m(true, false, true), DragButton::Left), None);
        assert_eq!(b.lookup(m(true, true, false), DragButton::Left), None);
        // Shift+Alt neither: the layer carry is the *bare* Shift row, and
        // Alt beside it is no more a bystander here than anywhere else —
        // which is also what keeps the marquee's intersect chord (§6.8) from
        // being read as a carry the instant the tool changes.
        assert_eq!(b.lookup(m(false, true, true), DragButton::Left), None);
    }

    #[test]
    fn the_button_is_part_of_the_chord() {
        let b = stock();
        // Every shipped row is a left-drag; their chords on the right button
        // ask for nothing.
        assert_eq!(b.lookup(m(true, false, false), DragButton::Right), None);
        assert_eq!(b.lookup(m(false, false, true), DragButton::Right), None);
        assert_eq!(b.lookup(m(false, true, false), DragButton::Right), None);
    }

    /// The chrome that promises paint stands down for exactly the actions that
    /// take the press away from it — asked of the table rather than of a list
    /// kept beside the cursor and the hover mark (§25.5).
    #[test]
    fn the_acts_that_shadow_the_brush() {
        assert!(!DragAction::TuneBrush.shadows_paint());
        assert!(DragAction::PickColor.shadows_paint());
        assert!(DragAction::PickAndTranslate.shadows_paint());
    }

    /// `ALL` is the action set and nothing else is. The match is exhaustive, so
    /// a variant added without a line here does not compile; the count is what
    /// notices one added to the match and forgotten in `ALL`.
    #[test]
    fn all_is_every_action() {
        let mut seen = HashSet::new();
        for action in DragAction::ALL.iter().copied() {
            assert!(seen.insert(action), "{action:?} is listed twice");
            match action {
                DragAction::TuneBrush | DragAction::PickColor | DragAction::PickAndTranslate => {}
            }
        }
        assert_eq!(
            DragAction::ALL.len(),
            3,
            "a new action needs a line in the match above and a row in `ALL`",
        );
    }

    /// A rebind is the action's **whole** binding, and it steals the chord from
    /// whoever held it — the loser keeping an override that says so, rather than
    /// falling back onto a default the user has just given away.
    #[test]
    fn a_rebind_steals_the_chord() {
        let mut b = stock();
        b.rebind(DragAction::TuneBrush, left(false, false, true));
        assert_eq!(b.of(DragAction::TuneBrush), Some(left(false, false, true)));
        assert_eq!(
            b.lookup(m(false, false, true), DragButton::Left),
            Some(DragAction::TuneBrush),
            "the chord answers with its new owner",
        );
        assert_eq!(
            b.of(DragAction::PickColor),
            None,
            "the eyedropper's default was taken, and it does not get another",
        );
        // And the tuning drag's own default is free again, bound to nothing.
        assert_eq!(b.lookup(m(true, false, false), DragButton::Left), None);
    }

    /// Erasing a binding is an override to nothing, so the shipped row cannot
    /// come back and undo the erasure.
    #[test]
    fn unbinding_is_not_a_reset() {
        let mut b = stock();
        b.unbind(DragAction::PickColor);
        assert_eq!(b.of(DragAction::PickColor), None);
        assert_eq!(b.lookup(m(false, false, true), DragButton::Left), None);
    }

    /// No preset may bind two actions to one chord: a table that is not a
    /// function would make the press path's answer depend on row order.
    #[test]
    fn every_preset_is_a_function() {
        for preset in DragPreset::ALL.iter().copied() {
            let mut chords = HashSet::new();
            let mut actions = HashSet::new();
            for &(action, c) in preset.rows() {
                assert!(
                    chords.insert(c),
                    "{}: two actions on {}",
                    preset.name(),
                    chord_label(c),
                );
                assert!(
                    actions.insert(action),
                    "{}: {action:?} twice",
                    preset.name()
                );
            }
        }
    }

    /// A preset's rows are what a press finds after taking it — for every
    /// action, including the ones a preset deliberately leaves unbound.
    #[test]
    fn taking_a_preset_gives_its_table() {
        for preset in DragPreset::ALL.iter().copied() {
            let mut b = stock();
            b.take(preset);
            for &action in DragAction::ALL {
                let want = preset.chord(action);
                assert_eq!(b.of(action), want, "{}: {action:?}", preset.name());
                if let Some(c) = want {
                    assert_eq!(
                        b.lookup(c.mods, c.button),
                        Some(action),
                        "{}: {} does not open {action:?}",
                        preset.name(),
                        chord_label(c),
                    );
                }
            }
            assert!(
                preset.matches(&b),
                "{} does not recognise itself",
                preset.name()
            );
        }
    }

    /// Krita's is the row that proves a preset may leave an act unbound: taking
    /// it must *erase* the carry rather than leave the shipped Shift standing.
    ///
    /// Krita puts its resize on that very chord, which is what makes the case
    /// worth a test of its own: Shift+drag still opens something, and a preset
    /// that had merely *added* its rows would have two acts on it.
    #[test]
    fn a_preset_may_leave_an_act_unbound() {
        let mut b = stock();
        b.take(DragPreset::Krita);
        assert_eq!(b.of(DragAction::PickAndTranslate), None);
        assert_eq!(
            b.lookup(m(false, true, false), DragButton::Left),
            Some(DragAction::TuneBrush),
            "Shift is Krita's resize, and the carry it displaced does not share it",
        );
        // And no other chord answers with it either — the whole point of an
        // override to nothing is that the act is off the table, not moved.
        for bits in 0..8u8 {
            let mods = m(bits & 1 != 0, bits & 2 != 0, bits & 4 != 0);
            for button in [DragButton::Left, DragButton::Right] {
                assert_ne!(
                    b.lookup(mods, button),
                    Some(DragAction::PickAndTranslate),
                    "{} still carries a layer",
                    chord_label(DragChord { mods, button }),
                );
            }
        }
    }

    /// An untouched table is Stark's, and taking Stark's stores **nothing** —
    /// which is what leaves a later build free to move a default.
    #[test]
    fn the_shipped_preset_is_stored_as_no_override_at_all() {
        assert!(DragPreset::Stark.matches(&stock()));
        let mut b = stock();
        b.take(DragPreset::Photoshop);
        assert!(!b.overrides.is_empty());
        b.take(DragPreset::Stark);
        assert!(
            b.overrides.is_empty(),
            "the shipped table is the defaults, not three rows saying so",
        );
    }

    /// A capture reads a press the way the table reads one — with the one
    /// exception the type exists for.
    #[test]
    fn what_a_capture_makes_of_a_press() {
        assert_eq!(
            capture(m(true, false, false), Some(DragButton::Left)),
            DragCapture::Chord(left(true, false, false)),
        );
        // The right button needs no modifier: it is free on the canvas.
        assert_eq!(
            capture(m(false, false, false), Some(DragButton::Right)),
            DragCapture::Chord(chord(false, false, false, DragButton::Right)),
        );
        // The one press that can never be a binding is the way out.
        assert_eq!(
            capture(m(false, false, false), Some(DragButton::Left)),
            DragCapture::Cancel,
        );
        // A button no chord can name leaves the capture listening.
        assert_eq!(capture(m(false, false, true), None), DragCapture::Pending);
    }

    /// The chip's spelling, which is the only place a user reads a chord.
    #[test]
    fn a_chord_reads_as_the_hand_holds_it() {
        assert_eq!(chord_label(left(true, false, false)), "Ctrl + drag");
        assert_eq!(chord_label(left(false, true, true)), "Shift + Alt + drag");
        assert_eq!(
            chord_label(chord(false, false, true, DragButton::Right)),
            "Alt + right-drag",
        );
        assert_eq!(
            chord_label(chord(false, false, false, DragButton::Right)),
            "right-drag",
        );
    }

    /// The stored form, round-tripped through both row shapes the record holds —
    /// and the untagged decode telling them apart, which is the one thing that
    /// could quietly read a binding as an offer.
    #[test]
    fn the_record_holds_two_kinds_of_row() {
        let mut b = stock();
        b.take(DragPreset::Photoshop);
        let rows: Vec<DragRow> = b
            .overrides
            .iter()
            .map(|&(action, chord)| DragRow::Bound { action, chord })
            .chain(std::iter::once(DragRow::Offered { offered: true }))
            .collect();
        let json = serde_json::to_string(&rows).unwrap();
        let back: Vec<DragRow> = serde_json::from_str(&json).unwrap();
        let mut read = DragBindings::default();
        let mut offered = false;
        for row in back {
            match row {
                DragRow::Bound { action, chord } => read.overrides.push((action, chord)),
                DragRow::Offered { offered: seen } => offered |= seen,
            }
        }
        assert!(offered);
        assert_eq!(read, b);
    }

    /// A row for an action this build no longer has costs that row and not the
    /// table — the list format's whole promise (`storage::load_list`), leaned on
    /// here because a retired action is exactly what §25.3 says to expect.
    #[test]
    fn a_row_for_a_retired_action_costs_one_row() {
        let json = r#"[
            {"action":"TuneBrush","chord":{"mods":{"ctrl":true},"button":"Left"}},
            {"action":"WalkTheDog","chord":null},
            {"offered":true}
        ]"#;
        let values: Vec<serde_json::Value> = serde_json::from_str(json).unwrap();
        let rows = values
            .into_iter()
            .filter_map(|v| serde_json::from_value::<DragRow>(v).ok())
            .count();
        assert_eq!(rows, 2, "the two readable rows survive the one that is not");
    }

    /// A chord written before a fourth modifier existed reads as not holding it
    /// — `Mods`' `#[serde(default)]`, which is the store's rule (§25.6) and the
    /// only reason a stored table survives that struct gaining a field.
    #[test]
    fn an_older_chord_reads_as_holding_what_it_named() {
        let c: DragChord =
            serde_json::from_str(r#"{"mods":{"ctrl":true},"button":"Left"}"#).unwrap();
        assert_eq!(c, left(true, false, false));
    }
}
