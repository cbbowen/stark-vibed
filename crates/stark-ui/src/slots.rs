//! The ten brushes under the hand: hold a number, paint with that brush (§18.1.8).
//!
//! Other painting applications bind keys to *tools* — B for brush, E for eraser, R
//! for blur. Stark has no such list to bind: an eraser is a brush whose effect is
//! `Erase` (§6.12), a blur is one with `bleed` up, and both are points in the same
//! parameter space as the brush that made them (§6.2). A key that selected a tool
//! would have to select a *brush*, and which brush is the artist's answer, not ours.
//!
//! So the numbers hold brushes — **a preset each, at a size and flow of the number's
//! own** ([`QuickBrush`]) — and there is exactly one rule:
//!
//! > **A held number is a temporary swap of the live brush.** Whatever you change
//! > while it is held stays with the number; the brush you were holding comes back
//! > when you let go.
//!
//! What a number can keep is what it holds: which preset, and the size and flow. The
//! tool itself is the preset's, looked up live ([`resolve`]), so a preset overwritten
//! in the brush editor is what every number bound to it holds from then on — the
//! binding is *durable*, which is what a binding is for — while the size and flow are
//! the number's own and survive the preset being edited under them
//! ([`Transient`], the **transient** half).
//!
//! # What is here and what a frontend still owes
//!
//! Everything above is arithmetic, so all of it is here: what a slot holds, what a
//! press and a release each decide ([`Held`]), how two presses become a pick
//! ([`Taps`]), which rows the rack draws ([`rows`]) and how the whole is stored.
//!
//! What a frontend owes is the three things only it can do — put a brush on and take
//! the live one back, keep these values wherever it keeps state, and draw. That split
//! is why [`Held::settle`] takes the live tune as an argument rather than reading it:
//! the rule is one function, and it is the same function in a browser and in a
//! window.

use serde::{Deserialize, Serialize};

use crate::brush_config::{BrushConfig, Transient};
use crate::presets::PresetEntry;
use crate::storage::{self, Store};

/// How many quick brushes there are — one per digit.
pub const COUNT: usize = 10;

/// The slot the pen's other end holds, and the one the `0` key holds: the digit past
/// the nine, which is where a tenth of anything goes on a keyboard.
pub const ERASER: usize = 0;

/// The slot a key **code** stands for, or `None` for every other key.
///
/// The physical key rather than the character it produces (`code`, not `key`),
/// which is what a hold binding wants: on a French layout the digit row types
/// `&é"'` unshifted, and a rack reachable only through Shift would be no rack at
/// all. The numeric keypad is the same ten slots — it is a digit row that happens
/// to be square.
pub fn of_code(code: &str) -> Option<usize> {
    let digit = code
        .strip_prefix("Digit")
        .or_else(|| code.strip_prefix("Numpad"))?;
    match digit.as_bytes() {
        [d @ b'0'..=b'9'] => Some((d - b'0') as usize),
        _ => None,
    }
}

/// What a digit holds: a **binding** to a preset, by name, and the transient half of
/// a brush — the size and flow the number was last used at.
///
/// A name and not a snapshot, which is the whole of what the rack is: the tool on a
/// number is whatever the preset is *today* ([`resolve`]), so the brush editor's
/// "Overwrite preset" reaches every number bound to the preset and a slot can never
/// be a stale copy of the tool it says it is. The size and flow are the number's own
/// because they are the two knobs a hand changes without changing its mind about the
/// tool — the pen on 2 at the width you draw with, the same pen on 7 at the width you
/// letter with.
///
/// Serde, since this is what the rack stores ([`StoredSlot`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QuickBrush {
    /// The preset's name ([`PresetEntry::name`]) — the library's key, and the one a
    /// preset keeps through every overwrite.
    pub preset: String,
    /// The size and flow the number keeps of its own. The color the tune also carries
    /// is dead weight here — a slot never changes the color you are painting with,
    /// and a color picked mid-hold does not rebind ([`Held::settle`]).
    pub transient: Transient,
}

/// The rack: one optional binding per digit, indexed **by the digit itself**, so
/// `rack[3]` is what the `3` key holds and there is no off-by-one to get wrong
/// between the keyboard, the panel and storage.
pub type Rack = [Option<QuickBrush>; COUNT];

/// An empty rack — `Default` spelled out, since `[None; COUNT]` wants a `Copy`
/// element and a binding holds a name.
pub fn empty_rack() -> Rack {
    std::array::from_fn(|_| None)
}

/// The brush a binding stands for **right now**: the named preset's tool at the
/// slot's own size and flow — or nothing, where the library no longer has the name.
///
/// The one place a slot becomes a brush, and it is asked at every use rather than
/// once at assignment — the hold, a row's click, the rack's picture and the thumbnail
/// scan all come here — which is what makes the binding live: there is no copy
/// anywhere for an overwritten preset to leave behind. A name the library cannot
/// answer to is treated as an empty slot wherever it is met, so a stale row (a preset
/// removed under a binding, or a stored rack naming something this build does not
/// ship) can only ever do nothing; [`unbind`] empties such slots outright so the case
/// stays a guard and not a state.
pub fn resolve(library: &[PresetEntry], slot: &QuickBrush) -> Option<(BrushConfig, Transient)> {
    crate::presets::find(library, &slot.preset).map(|e| (e.brush, slot.transient))
}

/// Fill a rack from the library: every preset that declares a home digit is bound to
/// it, at the size and flow it ships at ([`PresetEntry::slot`]).
///
/// Reading *from* the library rather than restating it — so a tool reaches the
/// keyboard under the name the panel lists it by, and adding a shipped preset puts it
/// on a digit by writing one field instead of by editing a second table here. This is
/// why the module defines no brush of its own: what a slot starts as is a question
/// about the app's tools, and those live in one place.
///
/// A slot past the rack is a definition to fix, not a panic to take: the preset is
/// still perfectly usable from the list.
pub fn seed(library: &[PresetEntry]) -> Rack {
    let mut rack = empty_rack();
    for entry in library {
        if let Some(slot) = entry.slot.filter(|s| *s < COUNT) {
            rack[slot] = Some(QuickBrush {
                preset: entry.name.clone(),
                transient: entry.transient,
            });
        }
    }
    rack
}

/// Bind `slot` to `brush`. `false` for a digit off the rack, which is the caller's
/// cue that there is nothing to persist.
pub fn assign(rack: &mut Rack, slot: usize, brush: QuickBrush) -> bool {
    let Some(row) = rack.get_mut(slot) else {
        return false;
    };
    *row = Some(brush);
    true
}

/// Empty `slot`, and say whether that changed anything.
///
/// The live brush is untouched, exactly as removing a preset leaves it: what goes is
/// the *binding*, not the tool. A slot cleared while it is being held is cleared for
/// good — the release then finds nothing changed and keeps nothing, so the emptying
/// stands and the displaced brush still comes back.
///
/// Clearing the last filled slot leaves a rack that is empty rather than unset, and
/// the two must not be confused: see [`read_storage`].
pub fn clear(rack: &mut Rack, slot: usize) -> bool {
    rack.get_mut(slot).is_some_and(|row| row.take().is_some())
}

/// Empty every slot bound to the preset called `name` — what removing a preset does
/// to the rack. A slot is a name and a tune, and a name the library no longer answers
/// to holds nothing; emptying it outright keeps the rack honest about what it holds
/// rather than leaving a row that looks empty and secretly is not.
///
/// `false` where nothing was bound, so an ordinary removal touches storage once, for
/// the library.
pub fn unbind(rack: &mut Rack, name: &str) -> bool {
    let bound_to = |b: &Option<QuickBrush>| b.as_ref().is_some_and(|b| b.preset == name);
    if !rack.iter().any(bound_to) {
        return false;
    }
    for slot in rack.iter_mut() {
        if bound_to(slot) {
            *slot = None;
        }
    }
    true
}

/// How close together two presses of one digit have to fall to read as a
/// **double-tap**, seconds — measured press to press, which is how the desktop
/// measures a double-click too.
///
/// Press to press rather than release to press because it then bounds the first press
/// as well: a hold that lasted longer than this can never be the first of a pair, so a
/// number held to draw under and pressed again for more is two holds and never a pick,
/// however quick the second press. Shorter than the desktop's half-second for the same
/// reason: a number under a painting hand is pressed in bursts, and the window has to
/// close between them.
pub const DOUBLE_TAP: f64 = 0.3;

/// The last press of a number key — which digit and when — so the next press can tell
/// whether it is the second of a pair ([`Taps::press`]).
///
/// Presses only, never releases: the window is press to press ([`DOUBLE_TAP`]), and a
/// key's own auto-repeat never reaches here, since a frontend declines a press while a
/// hold is in flight before it asks.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Taps {
    last: Option<(usize, f64)>,
}

impl Taps {
    /// Record a press of `slot` at `now` (seconds, on any monotonic clock), and say
    /// whether it is the second of a double-tap: the same digit, pressed again within
    /// [`DOUBLE_TAP`] of the last press.
    ///
    /// A pair is spent by being reported, so a third press in the same window starts
    /// over rather than pairing with the second: the count is exactly two.
    pub fn press(&mut self, slot: usize, now: f64) -> bool {
        let double = self
            .last
            .is_some_and(|(s, at)| s == slot && now - at < DOUBLE_TAP);
        self.last = (!double).then_some((slot, now));
        double
    }
}

/// What is holding a slot down.
///
/// Carried through the hold so a release can only be reported by the thing that made
/// it: a keyup must not end an eraser stroke, and lifting the pen must not release a
/// key the hand is still on. Without it either release would end the other's hold and
/// hand back a brush that was never displaced.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Grip {
    /// A number key.
    Key,
    /// The pen's eraser end facing the glass — hovering in range or in contact with
    /// it.
    Eraser,
}

impl Grip {
    /// Whether a press made with this grip takes the brush from a hold `held` already
    /// has — the difference between an **act** and a **posture**.
    ///
    /// The tail's hold says only which end of the pen faces the glass, so a number
    /// deliberately pressed under one takes the brush from it, and the tail takes it
    /// back on its next report. Nothing else displaces anything: a key cannot take a
    /// key, which is what keeps a hand rolling from 3 to 4 on the hold it has, and the
    /// tail cannot take a key, so a number held while the pen is flipped over goes on
    /// being the brush that draws.
    pub fn displaces(self, held: Self) -> bool {
        self == Self::Key && held == Self::Eraser
    }
}

/// A hold in flight: which slot, what is holding it, and what the release is decided
/// against.
#[derive(Clone, PartialEq, Debug)]
pub struct Held {
    slot: usize,
    grip: Grip,
    /// The brush the hold displaced — both halves, what comes back when it ends.
    base: (BrushConfig, Transient),
    /// The preset `base` was taken from, put back with it. Carried rather than
    /// re-derived at the release, because `base` may have been edited off its preset —
    /// and a hold is a loan of the hand, not the act that forgets where the brush in
    /// it came from.
    base_from: Option<String>,
    /// The size and flow the hold *began* on, once the swap had happened. What the
    /// release compares against to decide whether anything was changed, and so whether
    /// the number has something new to keep. The transient half alone, because it is
    /// all a number can keep of a value: the tool is the preset's, and a preset chosen
    /// says so for itself ([`Held::claim`]).
    entered: Transient,
    /// Whether a whole tool was deliberately put on during the hold — a preset row
    /// clicked, or another slot's row.
    claimed: bool,
    /// Whether the release keeps the slot's brush in hand rather than putting `base`
    /// back: the press that made this hold was the second of a double-tap ([`Taps`]),
    /// which *picks* the number where a single press borrows it.
    picked: bool,
}

impl Held {
    /// Open a hold on `slot`. `base` is the brush being displaced and `base_from` the
    /// preset it came from; `picked` says the press was the second of a double-tap.
    ///
    /// `entered` starts as the base's own tune, which is the right answer for an empty
    /// slot — nothing is swapped, so nothing moved. A hold that *did* swap says what it
    /// landed on with [`enter`](Self::enter).
    pub fn open(
        slot: usize,
        grip: Grip,
        base: (BrushConfig, Transient),
        base_from: Option<String>,
        picked: bool,
    ) -> Self {
        Self {
            slot,
            grip,
            base,
            base_from,
            entered: base.1,
            claimed: false,
            picked,
        }
    }

    /// Say what the swap actually left in hand.
    ///
    /// Read back rather than assumed: putting a brush on resolves its stamp and clamps
    /// what the renderer cannot draw, so what the app now holds is not necessarily what
    /// was handed to it — and it is what the release has to compare against.
    pub fn enter(&mut self, tune: Transient) {
        self.entered = tune;
    }

    /// The digit being held.
    pub fn slot(&self) -> usize {
        self.slot
    }

    /// What is holding it — what a release has to be reported by.
    pub fn grip(&self) -> Grip {
        self.grip
    }

    /// The preset the displaced brush came from, which goes back with it.
    pub fn base_from(&self) -> Option<String> {
        self.base_from.clone()
    }

    /// Whether a **key** is what is holding the slot down, rather than the pen's tail.
    /// What the rack is drawn on — the one place the two grips are told apart, since a
    /// tail is on the glass for every erase stroke and a rack flying in and out of the
    /// corner of the eye on each one is noise answering a question nobody asked.
    pub fn by_key(&self) -> bool {
        matches!(self.grip, Grip::Key)
    }

    /// Whether a release of `slot` by `grip` is this hold's own.
    ///
    /// Both halves matter. The **grip** keeps a keyup from ending an eraser stroke and
    /// a pen lift from releasing a key still under a finger; the **slot** keeps a
    /// second number pressed and released during a hold — a hand rolling from 3 to 4 —
    /// from ending the hold that 3 still has.
    pub fn ends_on(&self, slot: usize, grip: Grip) -> bool {
        self.slot == slot && self.grip == grip
    }

    /// Whether a press made with `grip` takes this hold's brush away
    /// ([`Grip::displaces`]), and with it the slot and grip whose release the caller
    /// owes first.
    pub fn displaced_by(&self, grip: Grip) -> Option<(usize, Grip)> {
        grip.displaces(self.grip).then_some((self.slot, self.grip))
    }

    /// Say that a whole tool was just put on **deliberately** — a preset row clicked,
    /// or another slot's row — so the release keeps what is live whether or not that
    /// moved anything. `false` where it had already been said.
    ///
    /// Raised by the two acts that mean *the artist chose a tool from a library*, and
    /// by nothing else. A slider drag needs no such word: it changed a value, and
    /// [`settle`](Self::settle)'s comparison sees that. Neither does the swap the hold
    /// makes itself, which is why a frontend must not raise it from inside the door
    /// both directions of the hold go through — a claim there would make every hold
    /// claim itself on the way in.
    pub fn claim(&mut self) -> bool {
        let fresh = !self.claimed;
        self.claimed = true;
        fresh
    }

    /// What the release does: the binding to keep in the slot (`None` when nothing was
    /// changed), and the brush to put back (`None` when the hold was a double-tap's,
    /// and the hand keeps what it has). `current` is the live brush's transient half at
    /// the release, and `from` the preset in hand — the tool the number would be bound
    /// to.
    ///
    /// A pure function because it is the whole rule, and the rule is the part worth
    /// being sure of. Five properties it has to have:
    ///
    /// - **An unused hold keeps nothing.** Holding 5 and drawing must not quietly make
    ///   5 the brush you happened to be holding — the numbers are assigned deliberately
    ///   or not at all, and an empty slot that filled itself the first time it was
    ///   pressed would be impossible to tell from one the user had set.
    /// - **Only the transient half is a change.** A slot keeps a preset's name and a
    ///   size and flow, so those two are what "did anything change?" is asked of. A
    ///   color picked mid-hold cannot reach the number, which is the one thing a slot
    ///   is defined not to carry; nor can a hardness moved in the editor, which is the
    ///   *preset's* to keep.
    /// - **A tool put on deliberately counts, even when it changes nothing.** The
    ///   comparison above is a *proxy* for "did the user set this brush?", and it is
    ///   wrong in exactly one place: clicking the preset that is already in hand.
    ///   Holding an empty 3 and clicking the preset you are painting with is the most
    ///   natural way there is to fill 3, and it moves no value at all — so a whole tool
    ///   arriving says so for itself ([`claim`](Self::claim)) rather than being
    ///   inferred from its effect.
    /// - **A number binds to a preset or to nothing.** What it keeps is the preset in
    ///   hand at the release, which a preset clicked under the hold set and an
    ///   untouched hold left as the slot's own (or, for an empty slot, the displaced
    ///   brush's). A brush that descends from no preset the library still has is a tool
    ///   the rack cannot name, and a tune with no tool under it is kept nowhere.
    /// - **A double-tap hands nothing back.** The second press of one is a hold in
    ///   every other respect — what the number keeps is decided exactly as above — but
    ///   its release leaves the slot's brush in hand, which is what makes tapping a
    ///   number twice the same act as clicking its row. Decided here rather than by the
    ///   caller because "what comes back" is the other half of the rule, and the two
    ///   halves are one function.
    pub fn settle(
        &self,
        current: Transient,
        from: Option<&str>,
    ) -> (Option<QuickBrush>, Option<(BrushConfig, Transient)>) {
        // `same_tune`, not `==`: the tune carries the hand's color now, and a color
        // picked mid-hold is the Color panel's act — the one thing a slot is defined
        // not to keep. Size and flow are what "did anything change?" is asked of.
        let changed = self.claimed || !current.same_tune(&self.entered);
        let kept = changed
            .then(|| {
                from.map(|preset| QuickBrush {
                    preset: preset.to_string(),
                    transient: current,
                })
            })
            .flatten();
        let back = (!self.picked).then_some(self.base);
        (kept, back)
    }

    /// What the number would keep if the hold ended **now** — the rule asked one moment
    /// early.
    ///
    /// The rack draws the held digit from this rather than from what is stored, so a
    /// preset clicked mid-hold shows up on the row it is about to land on instead of at
    /// the release that stores it — by which time a transient rack is already gone, and
    /// the answer arrives only on the next press of the same key. One function, so what
    /// the row promises and what the release does cannot be two answers.
    pub fn would_keep(&self, current: Transient, from: Option<&str>) -> Option<QuickBrush> {
        self.settle(current, from).0
    }
}

/// What the rack is drawn from: the ten bindings, the library that turns a name into a
/// brush, the hold in flight and the brush in hand.
///
/// `holding` is a **key** hold ([`Held::by_key`]) — the caller filters, because whether
/// a tail's hold draws the rack is a question about the surface rather than about the
/// rule.
pub struct View<'a> {
    pub rack: &'a Rack,
    pub library: &'a [PresetEntry],
    pub holding: Option<&'a Held>,
    /// The live brush, both halves — what a row is lit against.
    pub live: (BrushConfig, Transient),
    /// The preset in hand: the tool a held digit is about to be bound to.
    pub in_hand: Option<&'a str>,
}

/// One row of the rack.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    /// The digit this row is reached by.
    pub slot: usize,
    /// The brush the row is *about to* hold, resolved — the release's own rule asked a
    /// moment early for the held digit ([`Held::would_keep`]), and what is stored for
    /// every other row.
    pub brush: Option<(BrushConfig, Transient)>,
    /// What the digit holds now, resolved: what the trash would take, and the last true
    /// picture of the slot for a frontend that draws one.
    ///
    /// The two differ only under a hold, and the difference is what a rack showing only
    /// stored state gets wrong: a brush *tuned* under a hold is new with every pixel of
    /// the drag and has no thumbnail until the release stores it, so a row shows the
    /// binding it is about to keep and the picture it still has.
    pub stored: Option<(BrushConfig, Transient)>,
    /// The preset the row is bound to, where the library still answers to the name.
    pub name: Option<String>,
    /// Being held right now. At most one row is.
    pub held: bool,
    /// Its brush is the one in hand — the size and flow counted
    /// ([`crate::presets::same_brush`]), since a slot *is* a size and a flow where a
    /// preset row lights for the tool at any size.
    pub lit: bool,
}

impl Row {
    /// What the row is called: its binding's preset, or what an empty digit is for.
    pub fn label(&self) -> String {
        match &self.name {
            Some(name) => name.clone(),
            None => "Empty \u{2014} click a preset to fill it".to_string(),
        }
    }

    /// The sentence a hover owes. The eraser's names the pen rather than the key,
    /// because the thing that reaches it is already in the hand.
    pub fn tip(&self) -> String {
        match (self.slot, self.name.is_some()) {
            (ERASER, true) => {
                "Click or tap 0 twice to paint with this, or flip the pen over".to_string()
            }
            (ERASER, false) => "Empty. Hold 0 (or the pen's eraser end) and click a preset".into(),
            (n, true) => {
                format!("Click or tap {n} twice to paint with this; hold {n} to borrow it")
            }
            (n, false) => format!("Empty. Hold {n} and click a preset to fill it"),
        }
    }
}

/// The rows to draw, in the order the digits sit on the keyboard — the eraser's own
/// slot last, where the `0` key is.
///
/// **A digit with nothing in it is not a row**, with one exception: the digit being
/// held. Holding an empty number is not a mistake — it is how the number gets its
/// first brush — while a standing rack of empty rows would be a column of controls
/// that do nothing.
pub fn rows(view: View<'_>) -> Vec<Row> {
    let held = view.holding.map(Held::slot);
    (1..COUNT)
        .chain(std::iter::once(ERASER))
        .filter(|slot| view.rack[*slot].is_some() || Some(*slot) == held)
        .map(|slot| {
            let stored = view.rack[slot]
                .as_ref()
                .and_then(|b| resolve(view.library, b));
            let binding = view
                .holding
                .filter(|h| h.slot() == slot)
                .and_then(|h| h.would_keep(view.live.1, view.in_hand))
                .or_else(|| view.rack[slot].clone());
            let brush = binding.as_ref().and_then(|b| resolve(view.library, b));
            Row {
                slot,
                brush,
                stored,
                // The row's name is its binding's — the preset the digit holds,
                // whatever size it holds it at. A binding the library cannot answer is
                // shown for what it does, which is nothing.
                name: binding.filter(|_| brush.is_some()).map(|b| b.preset),
                held: Some(slot) == held,
                lit: brush.is_some_and(|b| crate::presets::same_brush(&view.live, &b)),
            }
        })
        .collect()
}

// --- persistence ----------------------------------------------------------
//
// One [`storage`] entry per **assigned** slot. The format and the skip-a-damaged-entry
// rule live there, so what is this module's own is that an entry names its digit rather
// than sitting at a position: a rack with holes stores as the few entries it has, and
// one whose digit is out of range is dropped instead of shifting its neighbours. What
// an entry holds is a binding ([`QuickBrush`]) — a preset's name and a size and flow —
// and never a brush: the tool is the preset's, and the preset library is the one store
// that holds one.

/// One assigned slot.
#[derive(Serialize, Deserialize)]
pub struct StoredSlot {
    digit: usize,
    #[serde(flatten)]
    brush: QuickBrush,
}

impl storage::Entry for StoredSlot {
    const STORE: Store = Store::Slots;
}

/// Write the rack down.
pub fn persist(rack: &Rack) {
    let stored: Vec<StoredSlot> = rack
        .iter()
        .enumerate()
        .filter_map(|(digit, brush)| {
            Some(StoredSlot {
                digit,
                brush: brush.clone()?,
            })
        })
        .collect();
    storage::save_list(&stored);
}

/// `None` when this client has never set a slot (or storage is unavailable) — the
/// caller seeds the rack from the library ([`seed`]). **`Some(empty)` is a different
/// answer**, and the difference is load-bearing now that a slot can be cleared
/// ([`clear`]): a rack the user has emptied to the last digit is left empty, where an
/// untouched one is seeded. A rack that re-seeded itself the moment it was emptied
/// would make the trash on the last row do nothing at all — the same emptied-versus-
/// absent case the preset library has, and `storage::load_list` keeps them apart for
/// both.
pub fn read_storage() -> Option<Rack> {
    let mut rack = empty_rack();
    for entry in storage::load_list::<StoredSlot>()? {
        // A digit past the rack is an entry a shorter build cannot place, and dropping
        // it is the only answer that does not move its neighbours.
        if entry.digit < COUNT {
            rack[entry.digit] = Some(entry.brush);
        }
    }
    Some(rack)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tune(size: f32, flow: f32) -> Transient {
        Transient {
            size,
            flow,
            ..Transient::default()
        }
    }

    fn held(entered: Transient, base: BrushConfig) -> Held {
        let mut h = Held::open(3, Grip::Key, (base, entered), None, false);
        h.enter(entered);
        h
    }

    fn bound(preset: &str, transient: Transient) -> Option<QuickBrush> {
        Some(QuickBrush {
            preset: preset.into(),
            transient,
        })
    }

    fn entry(name: &str, brush: BrushConfig, transient: Transient) -> PresetEntry {
        PresetEntry {
            name: name.into(),
            brush,
            transient,
            slot: None,
            builtin: false,
        }
    }

    /// The digit row and the keypad are the same ten slots, read by position.
    #[test]
    fn both_digit_rows_reach_the_same_slots() {
        for d in 0..10 {
            assert_eq!(of_code(&format!("Digit{d}")), Some(d));
            assert_eq!(of_code(&format!("Numpad{d}")), Some(d));
        }
        assert!(of_code("KeyA").is_none());
        assert!(of_code("Digit").is_none());
        assert!(of_code("DigitX").is_none());
    }

    /// Every slot a key can reach is a slot the rack has.
    #[test]
    fn no_key_names_a_slot_off_the_rack() {
        for d in 0..10 {
            assert!(of_code(&format!("Digit{d}")).is_some_and(|s| s < COUNT));
        }
    }

    #[test]
    fn a_hold_that_changed_nothing_keeps_nothing() {
        let brush = BrushConfig::default();
        let entered = Transient::default();
        let h = held(entered, brush);
        let (kept, back) = h.settle(entered, Some("Pen"));
        assert_eq!(kept, None, "an unused hold must not claim the slot");
        assert_eq!(back, Some((brush, entered)));
    }

    #[test]
    fn a_size_or_flow_dragged_stays_with_the_number() {
        // Bound to the preset in hand, at the size that was dragged — and the brush in
        // hand comes back untouched.
        let base = BrushConfig::default();
        let entered = tune(40.0, 1.0);
        let (kept, back) = held(entered, base).settle(tune(64.0, 1.0), Some("Pen"));
        assert_eq!(kept, bound("Pen", tune(64.0, 1.0)));
        assert_eq!(
            back,
            Some((base, entered)),
            "the pair in hand comes back untouched"
        );
        let (kept, _) = held(entered, base).settle(tune(40.0, 0.2), Some("Pen"));
        assert_eq!(
            kept,
            bound("Pen", tune(40.0, 0.2)),
            "flow is the other half"
        );
    }

    #[test]
    fn a_change_to_the_tool_alone_is_not_the_numbers_to_keep() {
        // The number keeps its binding and the two knobs of the tune: an opacity moved
        // under the hold or a hardness changed reach the config alone, and a **color
        // picked reaches the tune but is the Color panel's act** — so none of them is a
        // change the number can keep, and the release finds nothing to do.
        let base = BrushConfig::default();
        let entered = Transient::default();
        let recolored = Transient {
            color: [0.9, 0.1, 0.2],
            ..entered
        };
        let (kept, back) = held(entered, base).settle(recolored, Some("Pen"));
        assert_eq!(kept, None, "a color picked mid-hold binds nothing");
        assert_eq!(back, Some((base, entered)));
    }

    #[test]
    fn a_preset_clicked_under_the_hold_fills_the_slot_even_when_it_moves_nothing() {
        // The case the value comparison cannot see: hold an empty 3 and click the
        // preset you are already painting with. Nothing changes, and the slot has to
        // end up bound to it all the same.
        let brush = BrushConfig::default();
        let entered = Transient::default();
        let mut h = held(entered, brush);
        assert!(h.claim(), "the first claim of a hold is news");
        assert!(!h.claim(), "and the second is not");
        let (kept, back) = h.settle(entered, Some("Pen"));
        assert_eq!(
            kept,
            bound("Pen", entered),
            "the tool was chosen, not inferred"
        );
        assert_eq!(back, Some((brush, entered)));
    }

    #[test]
    fn a_number_binds_to_a_preset_or_to_nothing() {
        // A brush descending from no preset the library has is a tool the rack cannot
        // name: a tune dragged under the hold has nothing to be kept against, and even
        // a claim keeps nothing.
        let base = BrushConfig::default();
        let entered = tune(40.0, 1.0);
        let (kept, back) = held(entered, base).settle(tune(64.0, 1.0), None);
        assert_eq!(kept, None);
        assert_eq!(back, Some((base, entered)));
        let mut h = held(entered, base);
        h.claim();
        assert_eq!(h.settle(entered, None).0, None);
    }

    #[test]
    fn a_claim_still_hands_the_displaced_brush_back() {
        // The half of the rule the claim must not touch: what the number keeps is one
        // question, what comes back to the hand is the other.
        let base = BrushConfig::default();
        let entered = tune(40.0, 1.0);
        let mut h = held(entered, base);
        h.claim();
        let (kept, back) = h.settle(entered, Some("Pen"));
        assert_eq!(kept, bound("Pen", entered));
        assert_eq!(back, Some((base, entered)));
    }

    #[test]
    fn a_double_tapped_number_stays_in_hand() {
        // The click's outcome by way of the keyboard: the hold hands nothing back, so
        // what was swapped in at the press is what the hand keeps.
        let base = BrushConfig::default();
        let entered = tune(40.0, 1.0);
        let mut h = Held::open(3, Grip::Key, (base, entered), None, true);
        h.enter(entered);
        let (kept, back) = h.settle(entered, Some("Pen"));
        assert_eq!(
            kept, None,
            "untouched, the number keeps nothing, tapped twice or not"
        );
        assert_eq!(back, None, "the displaced brush is not put back");
    }

    #[test]
    fn tuning_under_a_double_tap_reaches_the_number_and_stays() {
        // The one rule is still in force for the second press: the change goes to the
        // number — and, this once, stays in hand as well.
        let base = BrushConfig::default();
        let entered = tune(40.0, 1.0);
        let mut h = Held::open(3, Grip::Key, (base, entered), None, true);
        h.enter(entered);
        let (kept, back) = h.settle(tune(64.0, 1.0), Some("Pen"));
        assert_eq!(kept, bound("Pen", tune(64.0, 1.0)));
        assert_eq!(back, None);
    }

    /// A key takes the brush from the pen's tail and from nothing else — an act over a
    /// posture, and no grip may take a key's own hold.
    #[test]
    fn only_a_key_displaces_and_only_the_tail_is_displaced() {
        assert!(Grip::Key.displaces(Grip::Eraser));
        assert!(!Grip::Key.displaces(Grip::Key));
        assert!(!Grip::Eraser.displaces(Grip::Key));
        assert!(!Grip::Eraser.displaces(Grip::Eraser));
        let h = held(Transient::default(), BrushConfig::default());
        assert_eq!(h.displaced_by(Grip::Key), None, "a key cannot take a key");
        assert!(h.ends_on(3, Grip::Key));
        assert!(!h.ends_on(4, Grip::Key), "a roll from 3 to 4 ends neither");
        assert!(!h.ends_on(3, Grip::Eraser), "nor does the wrong hand");
    }

    #[test]
    fn a_slot_is_its_preset_at_its_own_size_and_flow() {
        // The binding is live: the tool is whatever the library holds under the name
        // today, and only the size and flow are the slot's. A name the library does not
        // have is an empty slot.
        let pen = BrushConfig {
            smoothing: 0.5,
            ..BrushConfig::default()
        };
        let library = vec![entry("Pen", pen, tune(18.0, 1.0))];
        let slot = QuickBrush {
            preset: "Pen".into(),
            transient: tune(64.0, 0.2),
        };
        let (tool, tune) = resolve(&library, &slot).expect("the library has the name");
        assert_eq!(tune, self::tune(64.0, 0.2), "the slot's own tune");
        assert!(
            crate::presets::same_tool(&tool, &pen),
            "…on the preset's tool, feel included"
        );
        let gone = QuickBrush {
            preset: "Gone".into(),
            ..slot
        };
        assert_eq!(resolve(&library, &gone), None);
    }

    #[test]
    fn two_presses_of_one_digit_within_the_window_are_a_double_tap() {
        let mut taps = Taps::default();
        assert!(
            !taps.press(3, 10.0),
            "the first press of anything is a hold"
        );
        assert!(taps.press(3, 10.0 + DOUBLE_TAP / 2.0));
    }

    #[test]
    fn a_slow_second_press_is_another_hold() {
        let mut taps = Taps::default();
        assert!(!taps.press(3, 10.0));
        assert!(!taps.press(3, 10.0 + DOUBLE_TAP));
        // ...and opens a window of its own.
        assert!(taps.press(3, 10.0 + DOUBLE_TAP + 0.1));
    }

    #[test]
    fn a_different_digit_never_pairs() {
        // A hand rolling 3, 4 is two holds, however fast.
        let mut taps = Taps::default();
        assert!(!taps.press(3, 10.0));
        assert!(!taps.press(4, 10.1));
        // And the roll moved the window: 3 again is measured against the 4.
        assert!(!taps.press(3, 10.2));
    }

    #[test]
    fn a_pair_is_spent_by_being_reported() {
        let mut taps = Taps::default();
        assert!(!taps.press(3, 10.0));
        assert!(taps.press(3, 10.1));
        assert!(!taps.press(3, 10.2), "a third press starts over");
        assert!(taps.press(3, 10.3), "...and pairs with the fourth");
    }

    /// A rack is seeded from the digits the presets themselves declare, so the two
    /// orders — the list's and the keyboard's — stay free of each other.
    #[test]
    fn the_rack_is_seeded_from_the_librarys_own_digits() {
        let mut pen = entry("Pen", BrushConfig::default(), tune(18.0, 1.0));
        pen.slot = Some(2);
        let mut eraser = entry("Eraser", BrushConfig::default(), tune(80.0, 1.0));
        eraser.slot = Some(ERASER);
        // A digit off the rack is a definition to fix, not a row to place.
        let mut stray = entry("Stray", BrushConfig::default(), Transient::default());
        stray.slot = Some(COUNT + 3);
        let rack = seed(&[
            pen,
            eraser,
            stray,
            entry("Loose", BrushConfig::default(), tune(9.0, 1.0)),
        ]);
        assert_eq!(rack[2].as_ref().map(|b| b.preset.as_str()), Some("Pen"));
        assert_eq!(rack[2].as_ref().map(|b| b.transient), Some(tune(18.0, 1.0)));
        assert_eq!(
            rack[ERASER].as_ref().map(|b| b.preset.as_str()),
            Some("Eraser")
        );
        assert_eq!(rack.iter().flatten().count(), 2);
    }

    /// Removing a preset empties every number bound to it, and touches storage only
    /// when something was.
    #[test]
    fn removing_a_preset_empties_the_numbers_bound_to_it() {
        let mut rack = empty_rack();
        assert!(assign(
            &mut rack,
            1,
            QuickBrush {
                preset: "Pen".into(),
                transient: tune(10.0, 1.0)
            }
        ));
        assert!(assign(
            &mut rack,
            7,
            QuickBrush {
                preset: "Pen".into(),
                transient: tune(60.0, 1.0)
            }
        ));
        assert!(assign(
            &mut rack,
            4,
            QuickBrush {
                preset: "Ink".into(),
                transient: tune(30.0, 1.0)
            }
        ));
        assert!(!assign(
            &mut rack,
            COUNT,
            QuickBrush {
                preset: "Off".into(),
                transient: Transient::default()
            }
        ));
        assert!(unbind(&mut rack, "Pen"));
        assert!(rack[1].is_none() && rack[7].is_none());
        assert!(rack[4].is_some(), "a slot on another name is untouched");
        assert!(
            !unbind(&mut rack, "Pen"),
            "and a second removal writes nothing"
        );
        assert!(clear(&mut rack, 4));
        assert!(!clear(&mut rack, 4), "an empty slot is already clear");
        assert!(
            !clear(&mut rack, COUNT),
            "and a digit off the rack is not one"
        );
    }

    /// The rows are the digits in keyboard order with the eraser's last, empty ones
    /// left out — and the held one drawn from what it is *about to* keep.
    #[test]
    fn the_rack_draws_what_the_held_digit_is_about_to_hold() {
        let pen = BrushConfig::default();
        let ink = BrushConfig {
            smoothing: 0.75,
            ..BrushConfig::default()
        };
        let library = vec![
            entry("Pen", pen, tune(18.0, 1.0)),
            entry("Ink", ink, tune(24.0, 1.0)),
        ];
        let mut rack = empty_rack();
        assign(
            &mut rack,
            2,
            QuickBrush {
                preset: "Pen".into(),
                transient: tune(18.0, 1.0),
            },
        );
        assign(
            &mut rack,
            ERASER,
            QuickBrush {
                preset: "Ink".into(),
                transient: tune(24.0, 1.0),
            },
        );
        // Nothing held: two rows, the eraser's last.
        let listed = rows(View {
            rack: &rack,
            library: &library,
            holding: None,
            live: (pen, tune(18.0, 1.0)),
            in_hand: Some("Pen"),
        });
        assert_eq!(
            listed.iter().map(|r| r.slot).collect::<Vec<_>>(),
            vec![2, ERASER]
        );
        assert!(listed[0].lit, "the brush in hand lights its row");
        assert!(!listed[1].lit);
        assert_eq!(listed[0].name.as_deref(), Some("Pen"));

        // Holding an empty 5 and having clicked Ink under it: the row exists because it
        // is held, and it already shows what the release will write.
        let mut h = Held::open(
            5,
            Grip::Key,
            (pen, tune(18.0, 1.0)),
            Some("Pen".into()),
            false,
        );
        h.enter(tune(18.0, 1.0));
        h.claim();
        let listed = rows(View {
            rack: &rack,
            library: &library,
            holding: Some(&h),
            live: (ink, tune(24.0, 1.0)),
            in_hand: Some("Ink"),
        });
        assert_eq!(
            listed.iter().map(|r| r.slot).collect::<Vec<_>>(),
            vec![2, 5, ERASER]
        );
        let five = &listed[1];
        assert!(five.held);
        assert_eq!(five.name.as_deref(), Some("Ink"), "one keystroke early");
        assert_eq!(five.stored, None, "and the rack still holds nothing there");
        assert!(five.label().contains("Ink"));
        assert!(
            listed[2].tip().contains("pen"),
            "the eraser's names the pen"
        );
    }

    /// A binding the library cannot answer to is an empty row wherever it is met.
    #[test]
    fn a_stale_binding_can_only_ever_do_nothing() {
        let mut rack = empty_rack();
        assign(
            &mut rack,
            3,
            QuickBrush {
                preset: "Gone".into(),
                transient: tune(10.0, 1.0),
            },
        );
        let listed = rows(View {
            rack: &rack,
            library: &[],
            holding: None,
            live: (BrushConfig::default(), Transient::default()),
            in_hand: None,
        });
        assert_eq!(
            listed.len(),
            1,
            "the row is still drawn: something is stored"
        );
        assert_eq!(listed[0].name, None);
        assert!(listed[0].label().starts_with("Empty"));
        assert!(!listed[0].lit);
    }
}
