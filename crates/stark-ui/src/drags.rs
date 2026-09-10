//! The drag-binding table (§25): which chord and button opens which canvas drag —
//! the pointer half of the command registry ([`crate::commands`]).
//!
//! A [`Command`](crate::commands::Command) is an act asked for *whole*; a drag action
//! is a **gesture**, with a press that opens it, moves that feed it and a release that
//! ends it. The gestures themselves stay in whichever frontend runs them, because each
//! owns its own lifecycle and state. What this module owns is the one question a
//! canvas used to answer with a hand-written ladder of modifier tests: **which of them
//! does this press open?** One table, one reader on the press path and one on the
//! advertisement path, so what a press does and what the cursor promises cannot drift
//! apart — the same bargain the chord table makes.
//!
//! Chords are **exact** about their modifiers, exactly as the keyboard's are
//! ([`Mods`]): Ctrl+Alt+drag is not the Ctrl row with a bystanding Alt, it is an
//! unbound chord, and an unbound chord's press falls through to what an unmodified
//! press does — painting. Unlike the keyboard table, **Alt is nameable here**: the
//! chord table refuses it because AltGr arrives as Ctrl+Alt and a layout types
//! *through* it, but a drag types nothing, so the trap has nothing to spring on.

use serde::{Deserialize, Serialize};
use strum::VariantArray;

use crate::keys::Mods;
use crate::storage::{self, Entry, Store};

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
///
/// `VariantArray::VARIANTS` is the action set the settings rows and the preset tables
/// are written against — derived, so it cannot be the place a fourth action is left
/// out of.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize, strum::VariantArray)]
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
/// laid over them. Lives on a frontend's drag signal
/// as a signal, so a settings row and the resting cursor both move the moment a
/// rebind lands.
///
/// Only the *overrides* are state, exactly as `stark_ui::commands::Bindings` holds only
/// its own: an action the user never touched keeps its default row, minus any
/// chord an override has claimed — so a default moved in a later build reaches a
/// browser that stored this table before the move.
#[derive(Clone, Default, PartialEq, Debug)]
pub struct DragBindings {
    /// The actions the user has taken over, each with the chord that is now its
    /// **whole** binding, or `None` for an action whose chord a later rebind
    /// stole or whose binding was erased outright.
    ///
    /// **Private for `commands::Bindings::overrides`' reason**: two rows on one chord
    /// would make [`lookup`](Self::lookup) and [`of`](Self::of) answer by vector
    /// position, and a record is where such a pair arrives from.
    /// [`from_stored`](Self::from_stored) is the only other way to fill it.
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
    pub fn lookup(&self, mods: Mods, button: DragButton) -> Option<DragAction> {
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

    /// The table a record's rows describe, laid down **one rebind at a time** —
    /// `commands::Bindings::from_stored`'s twin, and for its reason: a record carrying
    /// two rows on one chord resolves by the steal rule rather than by vector order.
    #[must_use]
    pub fn from_stored(rows: impl IntoIterator<Item = (DragAction, Option<DragChord>)>) -> Self {
        let mut bindings = Self::default();
        for (action, chord) in rows {
            match chord {
                Some(chord) => bindings.rebind(action, chord),
                None => bindings.unbind(action),
            }
        }
        bindings
    }

    /// Take `preset`'s table whole.
    ///
    /// The shipped preset is applied by **clearing** the overrides rather than
    /// by writing its three rows out: a browser storing three rows that say
    /// what Stark already does is a browser no later build could ever move
    /// (§25.8). Every other preset is stored in full, for the mirror-image
    /// reason — it is a claim about *another* app's table, and a default that
    /// moves must not move it.
    pub fn take(&mut self, preset: DragPreset) {
        self.overrides.clear();
        if preset == DragPreset::Stark {
            return;
        }
        self.overrides = DragAction::VARIANTS
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
pub enum DragRow {
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

/// This client's stored drag table and the mark saying whether it has been offered a
/// preset, or `None` where it has never kept the record.
///
/// Both halves through one door because they are one record ([`DragRow`]): a reader
/// that took only the bindings would leave the offer to be made again next launch.
pub fn stored_drags() -> Option<(DragBindings, Offer)> {
    let rows: Vec<DragRow> = storage::load_list()?;
    let mut bound = Vec::new();
    let mut offered = false;
    for row in rows {
        match row {
            DragRow::Bound { action, chord } => bound.push((action, chord)),
            DragRow::Offered { offered: seen } => offered |= seen,
        }
    }
    let offer = if offered {
        Offer::Offered
    } else {
        Offer::Unoffered
    };
    Some((DragBindings::from_stored(bound), offer))
}

/// Write the whole record: the override rows, then the offer's mark.
///
/// **One writer for both row kinds**, which is what the untagged shape is for — a
/// rebind, a preset and the offer being made all come through here, so neither half
/// can be written without the other beside it (§25.6's fourth step).
pub fn persist_drags(bindings: &DragBindings, offer: Offer) {
    let rows: Vec<DragRow> = bindings
        .overrides
        .iter()
        .map(|&(action, chord)| DragRow::Bound { action, chord })
        // `Due` is a press waiting for the hand to come off the canvas, not an offer
        // made: only `Offered` is the mark (§25.8).
        .chain(std::iter::once(DragRow::Offered {
            offered: offer == Offer::Offered,
        }))
        .collect();
    storage::save_list(&rows);
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
    /// for the hand to come off the canvas (a frontend's `settle_offer`).
    Due,
    /// Made. Once ever, per browser — the settings dialog is the way back.
    Offered,
}

impl Offer {
    /// What a canvas press makes of the offer (§25.8): `found` is what
    /// [`DragBindings::lookup`] answered, and `mods` what the hand was holding.
    ///
    /// The whole of the trigger and its three exclusions, which are a rule over this
    /// triple and nothing else — a frontend keeps only the signal write:
    ///
    /// - **Asked of `lookup`, not of `find`'s answer.** A bound chord that *declines*
    ///   — Shift over a selection tool, where it is the union marquee — is not an
    ///   unbound one, and offering a table of presets there answers a question nobody
    ///   asked. So `found` is the table's answer before [`DragAction::claims`].
    /// - **Modified presses only.** A bare contact is painting, and a bare right press
    ///   is a chord nobody arrives holding.
    /// - **Once ever**, which is what makes [`Offered`](Self::Offered) absorbing.
    #[must_use]
    pub fn on_press(self, mods: Mods, found: Option<DragAction>) -> Offer {
        if self != Offer::Unoffered || found.is_some() || mods.bare() {
            return self;
        }
        Offer::Due
    }

    /// The offer coming off the canvas: the state to keep, and whether to show the
    /// dialog now.
    ///
    /// Due and shown are two steps for the tour's reason (§24): the press that found
    /// nothing bound goes on to paint a stroke, so the dialog waits until
    /// `end_interaction` takes the hand off the canvas. The mark is written *here*,
    /// on the way up — dismissing the dialog is an answer, and one that came back
    /// until it got the one it wanted would be a dialog nobody forgives.
    #[must_use]
    pub fn settle(self) -> (Offer, bool) {
        match self {
            Offer::Due => (Offer::Offered, true),
            held => (held, false),
        }
    }
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

/// What the chrome knows about the hand that the drag table does not (§18.0.2).
///
/// A fact about **presses** rather than about sampling, which is why it sits with the
/// table that reads them: [`DragAction::claims`] is what asks it, and the eyedropper
/// is only one of the acts that stand down.
///
/// [`Hovering`](crate::input::Hovering)'s shape, and for its reason: the five are
/// spellings of one question — *is this press already promised* — and a caller handing
/// them over positionally could transpose two and arm the eyedropper for something
/// else's reason.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Hand {
    /// Space is down, so the press pans (§18.1.7) — which is what leaves space+Alt a
    /// pan rather than a sample.
    pub panning: bool,
    /// A shape tool is in hand, where Alt is already the subtract marquee (§6.8): the
    /// selection gesture is what the press is *for* there, so the sampler stands
    /// down whichever chord it happens to wear.
    pub selecting: bool,
    /// The timeline is playing. The picture under the pointer is the playhead's
    /// rather than the painting's, so a sample would read the replay mid-flight.
    pub playing: bool,
    /// The eyedropper is already down — a drag in hand, or a held touch that resolved
    /// into one (§18.1.11).
    pub sampling: bool,
    /// Some other gesture already has the canvas: a stroke, a pan, a knob.
    pub busy: bool,
}

impl Hand {
    /// Whether a press right now would **sample rather than paint** — what the cursor
    /// promises, and what the press path then answers.
    ///
    /// [`armed`] plus the act's own claim: the free function says which act the chord
    /// opens, and [`DragAction::claims`] says whether this hand may have it.
    pub fn armed(self, bindings: &DragBindings, held: Mods) -> bool {
        armed(bindings, held) == Some(DragAction::PickColor) && DragAction::PickColor.claims(self)
    }

    /// Whether the press is still going begging — nothing with a stronger claim on
    /// it already holds it. The whole of what [`DragAction::claims`] stands the two
    /// canvas-reading acts down for.
    pub fn free(self) -> bool {
        !self.panning && !self.selecting && !self.playing
    }

    /// Whether the eyedropper's options bar is up: armed, and **not yet in use**.
    ///
    /// It goes away again the moment the drag starts, because from then on the thing
    /// to look at is the canvas and the color coming off it — and a chord pressed
    /// mid-stroke must not pop a bar up over the painting.
    pub fn shows_options(self, bindings: &DragBindings, held: Mods) -> bool {
        self.armed(bindings, held) && !self.sampling && !self.busy
    }
}

impl DragAction {
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

    /// Whether this action claims a press from `hand` right now — the act's own gate
    /// (§25.3), asked by a frontend's `find` where the canvas's ladder used to encode
    /// it in ordering. `Command::gate`'s lesson restated for presses: which question
    /// an act must ask is a fact about the act, not about the call site.
    ///
    /// A declined press falls through to the paint path, which is usually exactly
    /// right — the modifiers then mean whatever the paint gesture says they mean.
    ///
    /// The two canvas-reading acts stand down for the same three, arriving from
    /// opposite sides: over a selection tool PickColor's Alt is the subtract marquee
    /// and PickAndTranslate's Shift is the union one (§6.8), and the selection
    /// gesture is what the press is *for* there; during playback a sample would read
    /// the replay rather than the painting, and a carry *commits*, which the playhead
    /// forbids. Tuning declines nothing: the brush is view state, and the sliders it
    /// shadows are not refused mid-playback either.
    pub fn claims(self, hand: Hand) -> bool {
        match self {
            DragAction::TuneBrush => true,
            DragAction::PickColor | DragAction::PickAndTranslate => hand.free(),
        }
    }

    /// Whether a press under this binding takes the canvas away from the brush — so
    /// the chrome that *promises* paint stands down while the chord is held: the brush
    /// circle, and the hover mark under it (§18.1.10).
    ///
    /// A property of the act rather than a list kept at each of those call sites, for
    /// [`claims`](Self::claims)' reason. The mark is the sharper half of the bill: it
    /// is a preview folded into the shown document, so a press that *reads* the canvas
    /// back would read the hypothesis as paint — the wrong color for the eyedropper,
    /// the wrong layer for the hit test.
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
/// Indexed by **the app somebody is arriving from**, not by distinct tables: two
/// apps that agreed on all three would still each keep a row. A preset is picked
/// by recognising a name, and a list that had merged them would offer neither.
///
/// The tables are each app's defaults for these three gestures, on Windows, with
/// the accelerator standing in for Ctrl and Command alike (`input::accel`). They
/// are a starting point rather than a fidelity claim, and the surfaces say so:
/// every row stays separately rebindable the moment a preset lands, which is the
/// only honest shape for a table transcribed out of somebody else's manual.
///
/// **Declared in the order both surfaces list them**, which is what
/// `VariantArray::VARIANTS` hands them: Stark first, because it is the table the app
/// starts on and the one a reader has to recognise as "what I have now"
/// (`tests::the_shipped_table_leads_the_list`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, strum::VariantArray)]
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
        DragAction::VARIANTS
            .iter()
            .all(|&action| bindings.of(action) == self.chord(action))
    }
}

/// What a press means to a rebinding capture (the settings dialog's row).
///
/// The pointer's counterpart to `stark_ui::commands::Capture`, and shorter by one arm:
/// erasing a binding is a control of its own here, because a press cannot say
/// "nothing" the way Backspace can.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DragCapture {
    /// A bindable chord — commit it.
    Chord(DragChord),
    /// The capture is called off, the binding left as it was.
    ///
    /// This is what a **bare contact** is spent on, and it is the choice
    /// `stark_ui::commands::capture` makes with Escape: the one press that can never be a
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

    /// The shipped table leads the list, which is what makes the row a reader
    /// recognises as "what I have now" the first one.
    ///
    /// Declaration order is now the list's order, so this is the one thing about it
    /// still worth stating: a preset inserted above `Stark` would compile clean and
    /// quietly demote it.
    #[test]
    fn the_shipped_table_leads_the_list() {
        assert_eq!(DragPreset::VARIANTS.first(), Some(&DragPreset::Stark));
    }

    /// The shipped table binds **every** action, which is the claim the deleted
    /// count was really making: [`defaults`] reads Stark's rows and nothing else, so
    /// an action missing from that one table is an act no press can reach, wearing a
    /// settings row that advertises nothing.
    #[test]
    fn the_shipped_table_binds_every_action() {
        for &action in DragAction::VARIANTS {
            assert!(
                DragPreset::Stark.chord(action).is_some(),
                "{action:?} has no shipped chord, so no press opens it",
            );
        }
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
        for preset in DragPreset::VARIANTS.iter().copied() {
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
        for preset in DragPreset::VARIANTS.iter().copied() {
            let mut b = stock();
            b.take(preset);
            for &action in DragAction::VARIANTS {
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
        let (read, offer) = read_record(&json);
        assert_eq!(offer, Offer::Offered);
        assert_eq!(read, b);
    }

    /// [`stored_drags`]' reading, without a store behind it — no backend is installed
    /// in this binary, so the real door answers `None` and this is the half worth
    /// testing.
    fn read_record(json: &str) -> (DragBindings, Offer) {
        let mut bound = Vec::new();
        let mut offered = false;
        for row in serde_json::from_str::<Vec<DragRow>>(json).unwrap() {
            match row {
                DragRow::Bound { action, chord } => bound.push((action, chord)),
                DragRow::Offered { offered: seen } => offered |= seen,
            }
        }
        let offer = if offered {
            Offer::Offered
        } else {
            Offer::Unoffered
        };
        (DragBindings::from_stored(bound), offer)
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

    /// Every action's **advertised** chord is the chord that opens it — the chord
    /// table's `what_a_row_advertises_is_what_the_keyboard_answers`, over this table.
    ///
    /// [`DragBindings::of`] and [`DragBindings::lookup`] are two independent walks of
    /// [`defaults`] under two different filters, and the whole module rests on their
    /// agreeing. The tables are every shape an override can leave the pair in, plus
    /// each shipped preset.
    #[test]
    fn what_a_row_advertises_is_what_a_press_opens() {
        let mut tables: Vec<(String, DragBindings)> = vec![
            ("stock".into(), stock()),
            ("one rebind".into(), {
                let mut b = stock();
                b.rebind(DragAction::TuneBrush, left(false, true, true));
                b
            }),
            ("a steal from a default".into(), {
                let mut b = stock();
                b.rebind(DragAction::TuneBrush, left(false, false, true));
                b
            }),
            ("an unbind".into(), {
                let mut b = stock();
                b.unbind(DragAction::PickColor);
                b
            }),
            ("two contested rebinds".into(), {
                let mut b = stock();
                b.rebind(DragAction::PickColor, left(true, true, false));
                b.rebind(DragAction::PickAndTranslate, left(true, true, false));
                b
            }),
        ];
        tables.extend(DragPreset::VARIANTS.iter().map(|&preset| {
            let mut b = stock();
            b.take(preset);
            (preset.name().to_string(), b)
        }));
        for (what, table) in &tables {
            for &action in DragAction::VARIANTS {
                let Some(chord) = table.of(action) else {
                    continue;
                };
                assert_eq!(
                    table.lookup(chord.mods, chord.button),
                    Some(action),
                    "{what}: {action:?} advertises {}, which opens something else",
                    chord_label(chord),
                );
            }
        }
    }

    /// A record whose rows put two actions on one chord still reads as a **function**
    /// — resolved by the steal rule, not by vector order (`DragBindings::from_stored`).
    #[test]
    fn a_record_with_two_rows_on_one_chord_still_reads_as_a_function() {
        let clash = left(true, true, false);
        let b = DragBindings::from_stored([
            (DragAction::PickColor, Some(clash)),
            (DragAction::PickAndTranslate, Some(clash)),
        ]);
        assert_eq!(
            b.lookup(clash.mods, clash.button),
            Some(DragAction::PickAndTranslate),
            "the later row wins, as a live rebind would",
        );
        assert_eq!(
            b.of(DragAction::PickColor),
            None,
            "and the loser is left unbound rather than back on its default",
        );
    }

    /// The offer comes due for a **modified** press the table has nothing bound to,
    /// and for nothing else (§25.8).
    ///
    /// A bare contact is painting, and a bare right press is a chord nobody arrives
    /// holding — neither is evidence of a hand reaching for a binding it knows from
    /// another app.
    #[test]
    fn a_bare_press_never_brings_the_offer_due() {
        let bare = m(false, false, false);
        assert_eq!(Offer::Unoffered.on_press(bare, None), Offer::Unoffered);
        // And the same press with a modifier is exactly what it is for.
        assert_eq!(
            Offer::Unoffered.on_press(m(false, true, true), None),
            Offer::Due,
        );
    }

    /// A chord this table **does** bind never brings the offer due, whether or not the
    /// act then takes the press.
    ///
    /// The exclusion that has to be asked of `lookup` rather than of a frontend's
    /// `find`: Shift over a selection tool is the union marquee, so the carry declines
    /// — and offering a table of presets to somebody whose Shift is busy would be
    /// answering a question they did not ask.
    #[test]
    fn a_bound_chord_that_declines_never_brings_the_offer_due() {
        let shift = m(false, true, false);
        let found = stock().lookup(shift, DragButton::Left);
        assert_eq!(found, Some(DragAction::PickAndTranslate));
        assert!(
            !DragAction::PickAndTranslate.claims(Hand {
                selecting: true,
                ..Hand::default()
            }),
            "the premise: over a selection tool this press is the marquee's",
        );
        assert_eq!(Offer::Unoffered.on_press(shift, found), Offer::Unoffered);
    }

    /// Made is made: the mark absorbs everything afterwards, which is what "once ever,
    /// per browser" means.
    #[test]
    fn the_offer_is_made_once() {
        let modified = m(false, false, true);
        assert_eq!(Offer::Offered.on_press(modified, None), Offer::Offered);
        assert_eq!(Offer::Offered.settle(), (Offer::Offered, false));
        // Due is the one state that shows a dialog, and showing it spends it.
        assert_eq!(Offer::Due.settle(), (Offer::Offered, true));
        assert_eq!(Offer::Unoffered.settle(), (Offer::Unoffered, false));
        // A second press while one is already waiting changes nothing either — the
        // dialog is owed once, not once per press.
        assert_eq!(Offer::Due.on_press(modified, None), Offer::Due);
    }

    /// Tuning claims every hand; the two acts that read the canvas back stand down for
    /// the same three (§25.3).
    ///
    /// One list rather than two byte-identical ones, which is what the frontends had:
    /// the drags differ in which chord reaches them, not in what outranks them.
    #[test]
    fn the_two_canvas_reading_acts_stand_down_together() {
        let promised = [
            Hand {
                panning: true,
                ..Hand::default()
            },
            Hand {
                selecting: true,
                ..Hand::default()
            },
            Hand {
                playing: true,
                ..Hand::default()
            },
        ];
        for hand in promised {
            assert!(
                DragAction::TuneBrush.claims(hand),
                "{hand:?} declined the tuning drag, which refuses nothing",
            );
            assert!(!DragAction::PickColor.claims(hand), "{hand:?} sampled");
            assert!(
                !DragAction::PickAndTranslate.claims(hand),
                "{hand:?} carried a layer",
            );
        }
        // A free hand takes all three — measured against the above, so none of them
        // can pass by the door being shut on everything.
        for &action in DragAction::VARIANTS {
            assert!(action.claims(Hand::default()), "{action:?}");
        }
    }

    /// A stored **unbind** survives the round trip rather than resurrecting the row it
    /// erased — Krita's, since it is the preset that writes a null chord (§25.8), and
    /// the row shape `from_stored`'s other arm is for.
    #[test]
    fn a_stored_unbind_is_not_a_reset() {
        let mut b = stock();
        b.take(DragPreset::Krita);
        let rows: Vec<DragRow> = b
            .overrides
            .iter()
            .map(|&(action, chord)| DragRow::Bound { action, chord })
            .collect();
        let json = serde_json::to_string(&rows).unwrap();
        assert!(
            json.contains(r#""chord":null"#),
            "Krita leaves the carry unbound, and the record has to say so: {json}",
        );
        let (read, offer) = read_record(&json);
        assert_eq!(read, b);
        assert_eq!(
            read.of(DragAction::PickAndTranslate),
            None,
            "an erased row must not come back on the next load",
        );
        assert_eq!(offer, Offer::Unoffered, "no mark in the record is no offer");
    }

    /// The chord arms the sampler, and a free hand takes the press.
    #[test]
    fn the_chord_arms_the_sampler() {
        let table = stock();
        assert!(Hand::default().armed(&table, m(false, false, true)));
        assert!(Hand::default().shows_options(&table, m(false, false, true)));
        assert!(!Hand::default().armed(&table, Mods::default()));
    }

    /// Each stand-down is a press already promised elsewhere, and each takes the
    /// arming with it — measured against the free hand above, so none of them can
    /// pass by the door being shut on everything.
    #[test]
    fn a_press_promised_elsewhere_samples_nothing() {
        let table = stock();
        let promised = [
            Hand {
                panning: true,
                ..Hand::default()
            },
            Hand {
                selecting: true,
                ..Hand::default()
            },
            Hand {
                playing: true,
                ..Hand::default()
            },
        ];
        for hand in promised {
            assert!(
                !hand.armed(&table, m(false, false, true)),
                "{hand:?} armed the sampler",
            );
            assert!(
                !hand.shows_options(&table, m(false, false, true)),
                "{hand:?} raised the bar",
            );
        }
    }

    /// A sampler in use is still armed — the cursor and the press path go on meaning
    /// the eyedropper — and its bar is down, because the answer is on the canvas now.
    #[test]
    fn the_bar_goes_down_the_moment_the_gesture_starts() {
        let table = stock();
        for hand in [
            Hand {
                sampling: true,
                ..Hand::default()
            },
            Hand {
                busy: true,
                ..Hand::default()
            },
        ] {
            assert!(
                hand.armed(&table, m(false, false, true)),
                "{hand:?} disarmed the sampler",
            );
            assert!(
                !hand.shows_options(&table, m(false, false, true)),
                "{hand:?} kept the bar up",
            );
        }
    }

    /// The arming follows a rebinding rather than the shipped chord: the bar comes up
    /// on whatever chord the pick actually wears.
    #[test]
    fn the_arming_follows_the_table() {
        let mut table = stock();
        table.rebind(DragAction::PickColor, left(true, false, false));
        assert!(Hand::default().armed(&table, m(true, false, false)));
        assert!(!Hand::default().armed(&table, m(false, false, true)));
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
