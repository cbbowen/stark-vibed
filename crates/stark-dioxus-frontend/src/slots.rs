//! The ten brushes under the hand: hold a number, paint with that brush (§18.1.8).
//!
//! Other painting applications bind keys to *tools* — B for brush, E for eraser,
//! R for blur. Stark has no such list to bind: an eraser is a brush whose
//! effect is `Erase` (§6.12), a blur is one with `bleed` up, and both
//! are points in the same parameter space as the brush that made them (§6.2). A
//! key that selected a tool would have to select a *brush*, and which brush is
//! the artist's answer, not ours.
//!
//! So the numbers hold brushes — **a preset each, at a size and flow of the
//! number's own** ([`QuickBrush`]) — and there is exactly one rule:
//!
//! > **A held number is a temporary swap of the live brush.** Whatever you change
//! > while it is held stays with the number; the brush you were holding comes
//! > back when you let go.
//!
//! What a number can keep is what it holds: which preset, and the size and flow.
//! The tool itself is the preset's, looked up live, so a preset overwritten in
//! the brush editor is what every number bound to it holds from then on — the
//! binding is *durable*, which is what a binding is for — while the size and
//! flow are the number's own and survive the preset being edited under them
//! (`brush_config::Transient`, the **transient** half). Everything the feature
//! does falls out of the rule rather than being wired up three times:
//!
//! - **Hold and draw** — the stroke is laid with the slot's brush, because the
//!   slot's brush *is* the live brush for the length of the hold, and a stroke
//!   snapshots the brush at `Start` (`Session::start_stroke`).
//! - **Hold and click a preset** — the preset lands on the live brush, so at
//!   release it is what the number is bound to, at the size and flow the preset
//!   brought with it. Including the preset already in hand, which is how an
//!   empty number is most often filled: a tool chosen from a library counts as
//!   the hold's change whether or not it moved a value ([`claim`]).
//! - **Hold and drag Size or Flow** — the panel's sliders write the live brush,
//!   as they always did, so the number keeps the number that was dragged. The
//!   panel needs to know nothing about any of this: it shows the live brush, and
//!   the live brush is the slot's.
//! - **Hold and edit the tool** — a hardness moved in the brush editor under a
//!   hold is an edit to the *preset's* tool, and the number has nowhere of its
//!   own to keep it: write it back with the editor's "Overwrite preset" and it
//!   reaches this number and every other bound to that preset; let the hold go
//!   without, and the swap-back takes it away with the borrowed brush. What the
//!   number keeps is its binding and its tune, never a third copy of a tool.
//! - **Flip the pen over** — the eraser end holds [`ERASER`] for as long as it
//!   *faces* the glass, hovering in range or pressed against anything at all
//!   (`input::tail_says`). It is the same hold, made by hardware instead of by a
//!   key, and being bound at the window rather than by any one surface
//!   (`input::bind_pen`) it earns the three lines above rather than only the
//!   first: erasing with the tail is one gesture, tuning the eraser's Size or
//!   Flow with it is another, and eraser-clicking a preset assigns that preset
//!   to the tail. A key and a hand do the same thing here — and because the
//!   hold begins in range rather than on contact, the brush cursor and the mark
//!   under it show the eraser *before* the press instead of after it
//!   (§18.1.10). A number pressed while the tail hovers outranks it ([`hold`]).
//!
//! What the rule deliberately does *not* carry is the painting **color**, which
//! is [`presets::wear`]'s rule and is applied in both directions: a slot never
//! changes the color you are painting with, and a color picked during a hold
//! survives the release. A slot is a tool; color belongs to the Color panel.
//!
//! One thing is said about the **release** rather than the hold: a number
//! **tapped twice** is picked rather than borrowed. The second press is a hold
//! like any other — the rack comes up, the rule above is in force — and the one
//! difference is that letting go keeps the slot's brush in hand instead of
//! handing the displaced one back ([`Held::settle`]). It is what clicking the
//! row does ([`pick`]), for a hand that has the keyboard and not the pin. Told
//! apart by *count* rather than by length, which is what keeps a hold from ever
//! being mistaken for it: the first tap is an ordinary hold, and an unused hold
//! does nothing ([`Taps`]).
//!
//! Like the shape and preset libraries the rack follows this browser rather than
//! the document (`localStorage`), and degrades to a per-session rack where
//! storage is unavailable.
//!
//! The rack draws itself while a number is held ([`SlotOverlay`]) — a column of
//! the brushes the digits carry, each as the rendered stroke the preset library
//! shows. The visibility menu can keep it up ("Quick brushes"), which is what makes
//! it clickable: the rule above needs a keyboard, and a rack that can be clicked
//! is the whole of what a hand without one gets ([`pick`]) — that, and the one
//! operation the rule cannot express, since a hold *assigns* and no length of
//! holding can mean *nothing*: a filled row wears a trash that empties the slot
//! when it is held down ([`clear`]) — held, because a tap on it is a tap on the
//! row, and a row's tap picks.

use dioxus::html::input_data::MouseButton;
use dioxus::prelude::*;

use crate::icons::icon;
use crate::layout::chrome_dimmed;
use crate::presets;
use crate::state::AppState;
use stark_ui::brush_config::{BrushConfig, Transient};
use stark_ui::presets::PresetEntry;
use stark_ui::storage::{self, Store};

/// How many quick brushes there are — one per digit.
pub const COUNT: usize = 10;

/// The slot the pen's other end holds, and the one the `0` key holds: the digit
/// past the nine, which is where a tenth of anything goes on a keyboard.
pub const ERASER: usize = 0;

/// What a digit holds: a **binding** to a preset, by name, and the transient half
/// of a brush — the size and flow the number was last used at.
///
/// A name and not a snapshot, which is the whole of what the rack is now: the
/// tool on a number is whatever the preset is *today* ([`resolve`]), so the
/// brush editor's "Overwrite preset" reaches every number bound to the preset
/// and a slot can never be a stale copy of the tool it says it is. The size
/// and flow are the number's own because they are the two knobs a hand changes
/// without changing its mind about the tool (`brush_config::Transient`) — the
/// pen on 2 at the width you draw with, the same pen on 7 at the width you
/// letter with.
///
/// Serde, since this is what the rack stores ([`StoredSlot`]).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct QuickBrush {
    /// The preset's name (`PresetEntry::name`) — the library's key, and the one
    /// a preset keeps through every overwrite.
    pub preset: String,
    /// The size and flow the number keeps of its own. The color the tune also
    /// carries is dead weight here — a slot never changes the color you are
    /// painting with (`presets::wear` keeps the hand's), and a color picked
    /// mid-hold does not rebind ([`Held::settle`], `Transient::same_tune`).
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
/// slot's own size and flow — or nothing, where the library no longer has the
/// name.
///
/// The one place a slot becomes a brush, and it is asked at every use rather
/// than once at assignment — the hold, the pinned row's click, the rack's
/// picture and the thumbnail scan all come here — which is what makes the
/// binding live: there is no copy anywhere for an overwritten preset to leave
/// behind. A name the library cannot answer to is treated as an empty slot
/// wherever it is met, so a stale row (a preset removed under a binding, or a
/// stored rack naming something this build does not ship) can only ever do
/// nothing; `presets::remove` empties such slots outright ([`unbind`]) so the
/// case stays a guard and not a state.
pub fn resolve(library: &[PresetEntry], slot: &QuickBrush) -> Option<(BrushConfig, Transient)> {
    stark_ui::presets::find(library, &slot.preset).map(|e| (e.brush, slot.transient))
}

/// [`resolve`] against the app's library, for the callers that have only the
/// state: its own statement so the library's read guard is dropped before the
/// caller goes on to dispatch (`presets::wear` rewrites signals under it).
fn resolve_in(state: AppState, slot: &QuickBrush) -> Option<(BrushConfig, Transient)> {
    let library = state.presets.peek();
    resolve(&library, slot)
}

/// How close together two presses of one digit have to fall to read as a
/// **double-tap**, seconds — measured press to press, which is how the desktop
/// measures a double-click too.
///
/// Press to press rather than release to press because it then bounds the first
/// press as well: a hold that lasted longer than this can never be the first of a
/// pair, so a number held to draw under and pressed again for more is two holds
/// and never a pick, however quick the second press. Shorter than the desktop's
/// half-second for the same reason: a number under a painting hand is pressed in
/// bursts, and the window has to close between them.
pub const DOUBLE_TAP: f64 = 0.3;

/// The last press of a number key — which digit and when — so the next press can
/// tell whether it is the second of a pair ([`Taps::press`]).
///
/// Presses only, never releases: the window is press to press ([`DOUBLE_TAP`]),
/// and a key's own auto-repeat never reaches here, since [`hold`] declines a
/// press while a hold is in flight before it asks.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Taps {
    last: Option<(usize, f64)>,
}

impl Taps {
    /// Record a press of `slot` at `now` (seconds, on any monotonic clock), and
    /// say whether it is the second of a double-tap: the same digit, pressed
    /// again within [`DOUBLE_TAP`] of the last press.
    ///
    /// A pair is spent by being reported, so a third press in the same window
    /// starts over rather than pairing with the second: the count is exactly two.
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
/// Carried through the hold so a release can only be reported by the thing that
/// made it: a keyup must not end an eraser stroke, and lifting the pen must not
/// release a key the hand is still on. Without it either release would end the
/// other's hold and hand back a brush that was never displaced.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Grip {
    /// A number key.
    Key,
    /// The pen's eraser end facing the glass — hovering in range or in contact
    /// with it (`input::tail_says`).
    Eraser,
}

/// A hold in flight: which slot, what is holding it, and what the release is
/// decided against.
#[derive(Clone, PartialEq, Debug)]
pub struct Held {
    /// The digit being held.
    pub slot: usize,
    grip: Grip,
    /// The brush the hold displaced — both halves, what comes back when it
    /// ends.
    base: (BrushConfig, Transient),
    /// The preset `base` was taken from (`AppState::preset_in_hand`), put back
    /// with it. Carried rather than re-derived at the release, because `base`
    /// may have been edited off its preset — and a hold is a loan of the hand,
    /// not the act that forgets where the brush in it came from.
    base_from: Option<String>,
    /// The size and flow the hold *began* on, once the swap had happened. What
    /// the release compares against to decide whether anything was changed, and
    /// so whether the number has something new to keep. The transient half
    /// alone, because it is all a number can keep of a value: the tool is the
    /// preset's, and a preset chosen says so for itself ([`claim`]).
    entered: Transient,
    /// Whether a whole tool was deliberately put on during the hold — a preset
    /// row clicked, or another slot's row ([`claim`]).
    claimed: bool,
    /// Whether the release keeps the slot's brush in hand rather than putting
    /// `base` back: the press that made this hold was the second of a
    /// double-tap ([`Taps`]), which *picks* the number ([`pick`]) where a single
    /// press borrows it.
    picked: bool,
}

impl Held {
    /// Whether a **key** is what is holding the slot down, rather than the pen's
    /// tail. What [`SlotOverlay`] is mounted on — see there for why the two
    /// grips are told apart for showing when they are alike for everything else.
    pub fn by_key(&self) -> bool {
        matches!(self.grip, Grip::Key)
    }

    /// What the release does: the binding to keep in the slot (`None` when
    /// nothing was changed), and the brush to put back (`None` when the hold was
    /// a double-tap's, and the hand keeps what it has). `current` is the live
    /// brush's transient half at the release, and `from` the preset in hand
    /// (`AppState::preset_in_hand`) — the tool the number would be bound to.
    ///
    /// Split out as a pure function because it is the whole rule, and the rule is
    /// the part worth being sure of. Five properties it has to have:
    ///
    /// - **An unused hold keeps nothing.** Holding 5 and drawing must not quietly
    ///   make 5 the brush you happened to be holding — the numbers are assigned
    ///   deliberately or not at all, and an empty slot that filled itself the
    ///   first time it was pressed would be impossible to tell from one the user
    ///   had set.
    /// - **Only the transient half is a change.** A slot keeps a preset's name
    ///   and a size and flow, so those two are what "did anything change?" is
    ///   asked of. A color picked mid-hold cannot reach the number, which is the
    ///   one thing a slot is defined not to carry; nor can a hardness moved in
    ///   the editor, which is the *preset's* to keep — the module doc says where
    ///   that edit goes.
    /// - **A tool put on deliberately counts, even when it changes nothing.**
    ///   The comparison above is a *proxy* for "did the user set this brush?",
    ///   and it is wrong in exactly one place: clicking the preset that is
    ///   already in hand. Holding an empty 3 and clicking the preset you are
    ///   painting with is the most natural way there is to fill 3, and it moves
    ///   no value at all — read through the proxy alone it is indistinguishable
    ///   from a hold nobody used, and the slot would stay empty. So a whole tool
    ///   arriving says so for itself ([`claim`]) rather than being inferred from
    ///   its effect.
    /// - **A number binds to a preset or to nothing.** What it keeps is the
    ///   preset in hand at the release, which a preset clicked under the hold
    ///   set and an untouched hold left as the slot's own (or, for an empty
    ///   slot, the displaced brush's). A brush that descends from no preset the
    ///   library still has is a tool the rack cannot name, and a tune with no
    ///   tool under it is kept nowhere.
    /// - **A double-tap hands nothing back.** The second press of one is a hold
    ///   in every other respect — what the number keeps is decided exactly as
    ///   above — but its release leaves the slot's brush in hand, which is what
    ///   makes tapping a number twice the same act as clicking its row
    ///   ([`pick`]). Decided here rather than in [`release`] because "what comes
    ///   back" is the other half of the rule, and the two halves are one
    ///   function.
    fn settle(
        &self,
        current: Transient,
        from: Option<&str>,
    ) -> (Option<QuickBrush>, Option<(BrushConfig, Transient)>) {
        // `same_tune`, not `==`: the tune carries the hand's color now, and a
        // color picked mid-hold is the Color panel's act — the one thing a
        // slot is defined not to keep (the module doc). Size and flow are what
        // "did anything change?" is asked of.
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

    /// What the number would keep if the hold ended **now** — the rule asked one
    /// moment early.
    ///
    /// The rack draws the held digit from this rather than from what is stored
    /// ([`SlotOverlay`]), so a preset clicked mid-hold shows up on the row it is
    /// about to land on instead of at the release that stores it — by which time
    /// a transient rack is already gone, and the answer arrives only on the next
    /// press of the same key. One function, so what the row promises and what the
    /// release does cannot be two answers.
    fn would_keep(&self, current: Transient, from: Option<&str>) -> Option<QuickBrush> {
        self.settle(current, from).0
    }
}

/// Begin holding `slot`. Ignored when a hold is already in flight — which is what
/// makes it safe to call on every keydown, since a held key repeats at the
/// system's repeat rate and each repeat is another keydown, and on every report
/// of a hovering pen, which is what the tail's hold is read off
/// (`input::tail_says`).
///
/// With one exception, and it is the difference between an **act** and a
/// **posture**: the tail's hold says only which end of the pen faces the glass,
/// so a number deliberately pressed under one takes the brush from it. The tail
/// takes it back on its next report. Nothing else displaces anything — a key
/// cannot take a key, which is what keeps a hand rolling from 3 to 4 on the hold
/// it has, and the tail cannot take a key, so a number held while the pen is
/// flipped over goes on being the brush that draws.
///
/// A slot with nothing in it still enters the hold rather than declining: the
/// hold *is* the arming, and holding an empty number while clicking a preset is
/// how the number gets its first brush.
///
/// A **key** pressed twice within [`DOUBLE_TAP`] enters its second hold *picked*
/// ([`Held::picked`]): the same hold, whose release keeps the slot's brush in
/// hand instead of handing the displaced one back — the click's outcome
/// ([`pick`]) by way of the keyboard. Counted below the guard on a hold in
/// flight, so a held key's repeats are never presses; and counted for keys
/// alone, since the pen's tail is on the glass or off it, and two dabs of it
/// are two erase strokes.
pub fn hold(state: AppState, slot: usize, grip: Grip) {
    if slot >= COUNT {
        return;
    }
    let mut held = state.slots.held;
    // Answered without cloning the hold, since this runs on every report of a
    // hovering pen and nearly all of them are already holding.
    let in_flight = held.peek().as_ref().map(|h| (h.slot, h.grip));
    if let Some((held_slot, held_grip)) = in_flight {
        if grip != Grip::Key || held_grip != Grip::Eraser {
            return;
        }
        release(state, held_slot, held_grip);
    }
    let mut taps = state.slots.taps;
    let picked = grip == Grip::Key && taps.write().press(slot, crate::platform::now_seconds());
    // Each read is its own statement, so no guard is alive when `wear` dispatches
    // and rewrites the brush signal underneath it (`state::update_brush`).
    let base = presets::worn(state);
    let base_from = state.preset_in_hand.peek().clone();
    let bound = state.slots.brushes.peek()[slot].clone();
    // The slot's brush as it is *now* — its preset looked up live, at the
    // slot's own size and flow. A binding the library cannot answer is an empty
    // slot, and an empty slot is held without a swap.
    if let Some(bound) = bound
        && let Some((brush, tune)) = resolve_in(state, &bound)
    {
        presets::wear(state, brush, tune, Some(bound.preset));
    }
    // Read back rather than assumed: `wear` resolves the stamp and clamps what the
    // renderer cannot draw, so what the app now holds is not necessarily what was
    // handed to it — and it is what the release has to compare against.
    let entered = presets::worn(state).1;
    held.set(Some(Held {
        slot,
        grip,
        base,
        base_from,
        entered,
        claimed: false,
        picked,
    }));
}

/// Say that a whole tool was just put on **deliberately** — a preset row clicked
/// ([`presets::apply`]) or another slot's row ([`pick`]) — so a hold in flight
/// keeps what is live when it ends whether or not that moved anything.
///
/// Called by the two acts that mean *the artist chose a tool from a library*,
/// and by nothing else. A slider drag needs no such word: it changed a value, and
/// [`Held::settle`]'s comparison sees that. Neither does the swap [`hold`] and
/// [`release`] make themselves, which is why this is not said inside
/// [`presets::wear`] — that door is used in both directions by the hold itself,
/// and a claim raised there would make every hold claim itself on the way in.
///
/// A no-op with no hold in flight, which is every other click on those same rows.
pub fn claim(state: AppState) {
    let mut held = state.slots.held;
    let Some(h) = held.peek().clone() else { return };
    if h.claimed {
        return;
    }
    held.set(Some(Held { claimed: true, ..h }));
}

/// End the hold on `slot`, if `grip` is what is holding it: keep whatever was
/// changed, and put the displaced brush back.
///
/// Both halves of the guard matter. The **grip** keeps a keyup from ending an
/// eraser stroke and a pen lift from releasing a key still under a finger; the
/// **slot** keeps a second number pressed and released during a hold — a hand
/// rolling from 3 to 4 — from ending the hold that 3 still has.
pub fn release(state: AppState, slot: usize, grip: Grip) {
    let mut held = state.slots.held;
    // The guard is answered before the hold is cloned: this runs on every report
    // of a pen whose tail is not facing the glass (`input::tail_says`), and the
    // overwhelming majority of them are holding nothing.
    let h = {
        let in_flight = held.peek();
        match in_flight.as_ref() {
            Some(h) if h.slot == slot && h.grip == grip => h.clone(),
            _ => return,
        }
    };
    held.set(None);
    let current = presets::worn(state).1;
    // The tool the number would be bound to: whatever preset is in hand at the
    // release — the one a click under the hold put on, or the slot's own.
    let from = state.preset_in_hand.peek().clone();
    let (kept, back) = h.settle(current, from.as_deref());
    if let Some(bound) = kept {
        assign(state, h.slot, bound);
    }
    // Back through the door it left by, with the name it had: the hold borrowed
    // the hand, and a preset chosen *during* it went to the slot, not to this.
    // Or not back at all, for a double-tap's hold, whose whole point is that the
    // swap stands.
    if let Some((back, back_tune)) = back {
        presets::wear(state, back, back_tune, h.base_from);
    }
}

/// End whatever hold is in flight, whoever made it — for the one event that can
/// take a key away without ever sending its keyup: the window losing focus.
///
/// Alt+Tab with a number held would otherwise leave the swap in force for the
/// rest of the session, with the key that would undo it now belonging to another
/// window. The same class of bug the modifier tracker (`input`'s `track_mods`)
/// rules out by re-reading the modifier set, ruled out here by the event that
/// says the keyboard has gone.
pub fn release_all(state: AppState) {
    let held = state.slots.held.peek().clone();
    if let Some(h) = held {
        release(state, h.slot, h.grip);
    }
}

/// The rack, drawn while a number is held (§18.1.8): a column down the left of
/// the window of the brushes the digits carry, each shown as the rendered test
/// stroke the preset library shows it by (`crate::thumbs`).
///
/// It replaced a permanent row of ten chips at the head of the Brush panel, and
/// the trade is the point. The chips spent the scarcest space in the app — panel
/// height — every second of every session to say a digit and three lit states,
/// and the digit is the one thing about a quick brush nobody needs told: the
/// question a rack of ten unlabelled numbers actually raises is *what is on 4*.
/// Being momentary is what pays for the answer — on screen only while a finger
/// is on the key that summons it, this can afford the width of a real preview
/// and costs the panel nothing.
///
/// Up while a number is held, and **kept** up by the visibility menu's "Quick
/// brushes" ([`Signals::slots`](crate::state::Signals::slots)`.pinned`) — which is the same picture with one
/// thing added: pinned, the rows take clicks, and clicking one is [`pick`].
/// That is the mouse-only way to a slot, and it is why the pin exists at all —
/// a pen in one hand and a tablet under it leaves no spare finger for the number
/// row, and the transient rack cannot be clicked (see below).
///
/// A filled row pinned also carries the **trash** every other roster in the app
/// carries ([`clear`]), hover-revealed like the preset rows'. It is the one
/// operation on a slot that the one rule cannot express: a hold assigns, and no
/// length of holding can mean *nothing*. It goes here rather than anywhere else
/// for the reason the pin exists — this is the only place the rack is a list of
/// controls rather than an answer to "what is on 4".
///
/// **It is held, not clicked** — the one trash in the app that is. Everywhere else
/// a trash stands beside the thing it removes; here it rides *on* a control whose
/// own tap is the commonest act on the rack, at the end of the row nearest the
/// painting, which is the end a pen coming off the canvas reaches first — and what
/// it takes is a binding the action log never held, so there is no undo behind it.
/// A tap that landed on it emptied the slot the hand was reaching for. So a tap on
/// the trash is a tap on the row ([`pick`]), and clearing takes the trash held
/// down: the disc fills red around the icon for as long as it is pressed, and the
/// slot goes when the fill closes. The fill is the clock — `onanimationend` is
/// what clears, so what the disc shows and what happens are one thing rather than
/// a stylesheet duration and a timer agreeing — and a tap shows the first sliver
/// of it, which is how a hand that expected a click finds out there is a hold to
/// be made. Releasing, or sliding off, before the fill closes is a tap.
///
/// Which is why the rows arm on their own press, as a dialog's backdrop does
/// (`widgets::Modal`, §25.7): the hold removes the row while the pen is still
/// down, and a pen's release is hit-tested afresh, so the click it makes lands on
/// whichever row has moved up under it — and would pick that.
///
/// Two things it deliberately does not do:
///
/// - **It never takes the pointer while it is transient** (`pointer-events` in
///   the stylesheet, granted to the rows only under `.pinned`). The gesture it
///   belongs to then is hold-*and-draw*, and the hand is very often painting
///   under it; a row that swallowed the stroke being drawn beneath it would be a
///   worse control than no control. Pinned is the user asking for that space to
///   be a control, and paying for it in canvas, exactly as a panel is. The
///   *container* declines the pointer either way, so the gaps and the empty
///   column below the last row are never anything but painting.
/// - **It shows no digit that has nothing in it**, pinned or not, with one
///   exception: the digit being held. Holding an empty number is not a mistake —
///   it is how the number gets its first brush — while a standing rack of empty
///   rows would be a column of controls that do nothing.
///
/// It **fades with the rest of the floating chrome** (`layout::chrome_dimmed`),
/// which goes to nothing while a canvas gesture is in flight: it stands over the
/// painting like the panels and the bars, and while a stroke is being laid the
/// screen goes back to being the painting. What makes that safe on something
/// this momentary is that **the hold outlives the stroke** — the key is still
/// down when the pen lifts, so the rack comes straight back and the answer is
/// there whenever the hand wants it rather than only before the first mark.
///
/// The **held** row is the only one at full strength: the rest of the rack is a
/// ghost of itself, while the held one is opaque and reaches further out from the
/// edge than any of them, so the digit under the finger is found rather than
/// looked for (`.slot-row.held`). The row whose brush the live brush *is* (color
/// aside) is separately lit, which is a different question and only looks like
/// the same one while a key is down: pinned and idle nothing is held, and the lit
/// row is then the only thing saying which slot is in hand. Lit on the stricter
/// of the library's two tests (`stark_ui::presets::same_brush`), the size and flow counted
/// — a slot *is* a size and a flow, where the preset rows light for the tool at
/// any size (`stark_ui::presets::same_tool`). Reading the live brush costs nothing per
/// stroke — a sample dispatches quietly and never refreshes the observable
/// (`state::dispatch_sample`), so this re-renders when the brush *changes*, not
/// while one is painting.
///
/// The hold that mounts it is a **key** hold alone, which is the one place the
/// two grips are told apart rather than being the same hold. The pen's tail holds
/// [`ERASER`] for as long as it faces the glass (`input::tail_says`) — that is
/// every erase stroke and every hover between them — and a rack standing in the
/// corner of the eye for as long as the pen is inverted is noise answering a
/// question nobody asked. Holding `0` shows the same row.
#[component]
pub fn SlotOverlay() -> Element {
    let state = use_context::<AppState>();
    let pinned = (state.slots.pinned)();
    // The hold itself, not only its digit: the held row is drawn from the rule it
    // is under (`Held::would_keep`) rather than from what is stored.
    let holding = (state.slots.held)().filter(Held::by_key);
    let held = holding.as_ref().map(|h| h.slot);
    // Neither held nor pinned: read nothing else at all, so a rack, a library or a
    // brush that changes during a stroke cannot re-render anything.
    //
    // **Which is why the rack itself is a component and not the rest of this
    // function.** What it draws is read through a memo now (`SlotRack`), and a
    // memo declared here would have to be declared above this line — hooks are
    // positional — leaving it recomputing against every engine write for the
    // whole time the rack is away, which is nearly all of the time. Mounted
    // behind the gate instead, it does not exist until there is something to
    // show, exactly as the timeline bar is mounted (`main`).
    if !pinned && held.is_none() {
        return rsx! {};
    }
    rsx! {
        SlotRack { pinned, holding }
    }
}

/// The rack proper, mounted only while there is one to draw ([`SlotOverlay`]).
///
/// `holding` rather than a bare slot number: the held row is drawn from the rule
/// the release is under, which needs the whole hold ([`Held::would_keep`]).
#[component]
fn SlotRack(pinned: bool, holding: Option<Held>) -> Element {
    let state = use_context::<AppState>();
    // Which row heard the press in flight, so a click applies a row only when it
    // closes a press that landed there (see [`SlotOverlay`] on the pen's release).
    let mut pressed: Signal<Option<usize>> = use_signal(|| None);
    // The trash being held down right now, if any: its row wears the fill for as
    // long as this says so, and nothing else keeps time.
    let mut arming: Signal<Option<usize>> = use_signal(|| None);
    let arming_now = arming();
    let held = holding.as_ref().map(|h| h.slot);
    let rack = (state.slots.brushes)();
    // The whole tool, feel and inactive effect included, off the frontend's own
    // signals — so the lit row tracks a smoothing drag exactly as it tracks
    // a radius one, and wakes for brush edits and nothing else.
    let live = (state.brush)();
    let live_tune = (state.transient)();
    // The preset in hand is the tool a held digit is about to be bound to
    // (`Held::would_keep`); it moves only when a whole tool arrives.
    let in_hand = (state.preset_in_hand)();
    // The rows are resolved against the library up front — a slot is a name, and
    // the library is where the name becomes a brush ([`resolve`]) — so no read
    // guard is alive while the rows below read the thumbnail cache one by one.
    // A subscribing read, since an overwritten preset is a changed row.
    #[expect(
        clippy::type_complexity,
        reason = "one local's shape, read two lines down"
    )]
    let rows: Vec<(
        usize,
        Option<(BrushConfig, Transient)>,
        Option<(BrushConfig, Transient)>,
        String,
    )> = {
        let library = state.presets.read();
        // The digits in the order they sit on the keyboard, with the eraser's own
        // slot last — where the `0` key is, and where a tenth of anything goes.
        (1..COUNT)
            .chain(std::iter::once(ERASER))
            .filter(|&slot| rack[slot].is_some() || Some(slot) == held)
            .map(|slot| {
                // What the digit holds, as its preset is today.
                let stored = rack[slot].as_ref().and_then(|b| resolve(&library, b));
                // And, for the one being held, what it is *about to* hold: the
                // release's own rule, asked a moment early (`Held::would_keep`).
                // A hold is where a slot changes, so a rack that showed only what
                // was stored would answer every question about the digit under
                // the finger one keystroke late — and for a transient rack, which
                // is gone by the release, not until the next press of that key.
                // Falls back to the stored binding, which is what an untouched
                // hold and every other row show.
                let binding = holding
                    .as_ref()
                    .filter(|h| h.slot == slot)
                    .and_then(|h| h.would_keep(live_tune, in_hand.as_deref()))
                    .or_else(|| rack[slot].clone());
                let brush = binding.as_ref().and_then(|b| resolve(&library, b));
                // The row's name is its binding's — the preset the digit holds,
                // whatever size it holds it at. A binding the library cannot
                // answer is shown for what it does, which is nothing.
                let label = match (&binding, brush) {
                    (Some(b), Some(_)) => b.preset.clone(),
                    _ => "Empty \u{2014} click a preset to fill it".to_string(),
                };
                (slot, brush, stored, label)
            })
            .collect()
    };

    rsx! {
        div {
            class: "slot-overlay chrome",
            class: if chrome_dimmed(state) { "dimmed" },
            class: if pinned { "pinned" },
            for (slot, brush, stored, label) in rows {
                {
                    // The brush as a stroke, filling the row as its background —
                    // the preset row's own recipe (`panels::brush`), down to
                    // writing `none` out rather than omitting the property: these
                    // nodes are reused as the rack changes under them, and a
                    // stranded declaration would leave one slot showing another's
                    // brush (inline style merges per property).
                    //
                    // The row it is about to hold where that has been rendered,
                    // and the one it still holds where it has not: a brush *tuned*
                    // under a hold is new with every pixel of the drag and has no
                    // thumbnail until the release stores it, and rendering one per
                    // frame of a Size drag is GPU spent on a picture nobody asked
                    // to keep. So the row shows the last true picture of the slot
                    // rather than blinking empty for the length of a drag — while
                    // a preset clicked mid-hold lands instantly, its thumbnail
                    // being the one the library is already showing.
                    let thumb = |b: Option<(BrushConfig, Transient)>| {
                        b.and_then(|(b, t)| crate::thumbs::url(state, &b, t))
                            .filter(|url| !url.is_empty())
                    };
                    let bg = match thumb(brush).or_else(|| thumb(stored)) {
                        Some(url) => format!("background-image: url({url});"),
                        None => "background-image: none;".to_string(),
                    };
                    // Says the binding a picture cannot, and only ever seen pinned
                    // — an element that takes no pointer is shown no tooltip. The
                    // eraser's names the pen rather than the key, because the thing
                    // that reaches it is already in the hand.
                    let title = match (slot, brush.is_some()) {
                        (ERASER, true) => "Click or tap 0 twice to paint with this, or flip the pen over".to_string(),
                        (ERASER, false) => "Empty. Hold 0 (or the pen's eraser end) and click a preset".to_string(),
                        (n, true) => format!("Click or tap {n} twice to paint with this; hold {n} to borrow it"),
                        (n, false) => format!("Empty. Hold {n} and click a preset to fill it"),
                    };
                    rsx! {
                        div {
                            key: "{slot}",
                            class: "slot-row",
                            // Lit where this is the brush in hand, color aside — the size
                            // and flow counted, since they are what a slot keeps of its
                            // own. Held wins in the stylesheet: it is what the user is
                            // doing right now rather than a state they are in.
                            class: if brush.is_none() { "empty" }
                                   else if brush.is_some_and(|b| stark_ui::presets::same_brush(&(live, live_tune), &b)) { "active" },
                            class: if Some(slot) == held { "held" },
                            style: "{bg}",
                            title,
                            // Reachable only while pinned, the stylesheet having
                            // given the rows the pointer there and nowhere else —
                            // the same way faded chrome stops taking clicks
                            // (`.chrome.dimmed`) rather than by every handler
                            // asking whether it is visible.
                            onpointerdown: move |_| pressed.set(Some(slot)),
                            // Terminal, as every disarm is: a press that was
                            // cancelled, or left the row, is not this row's click
                            // whatever lands here next.
                            onpointercancel: move |_| pressed.set(None),
                            onpointerleave: move |_| pressed.set(None),
                            onclick: move |_| {
                                // Bound, not read in the `if`: the read would be
                                // held through a body that writes the same signal.
                                let heard_the_press = pressed() == Some(slot);
                                pressed.set(None);
                                if heard_the_press {
                                    pick(state, slot);
                                }
                            },
                            span { class: "slot-row-digit",
                                if slot == ERASER {
                                    {icon(stark_ui::icons::ERASER)}
                                } else {
                                    "{slot}"
                                }
                            }
                            span { class: "slot-row-name", "{label}" }
                            if stored.is_some() {
                                // The stored binding, not the row's: the trash
                                // takes something *out of the rack*, and a held
                                // row previewing a binding the release has not
                                // written yet has nothing there for it to take.
                                //
                                // The trash every other roster in the app wears
                                // (`stark_ui::icons::REMOVE`) — presets, layers, guides,
                                // shapes, gradients — because emptying a row of
                                // this one is the same act, revealed on the same
                                // hover and answering it in the same red ink.
                                //
                                // Only where there is something to remove: an
                                // empty row is on screen because it is *held*, and
                                // a trash on it would offer to undo nothing. Only
                                // reachable pinned, for the row's own reason — the
                                // stylesheet grants the pointer there and nowhere
                                // else, so nothing here asks whether the rack is a
                                // control right now.
                                //
                                // Held rather than clicked, alone among them: see
                                // [`SlotOverlay`]. The press bubbles on to the row,
                                // so a tap here is the row's own tap and needs no
                                // click of its own. Nothing here keeps time — the
                                // fill's own end is what clears, so a hold that
                                // outlived its row could not fire, there being no
                                // element left for an animation to end on.
                                button {
                                    class: "slot-clear",
                                    class: if arming_now == Some(slot) { "arming" },
                                    title: "Hold to clear this slot",
                                    onpointerdown: move |e| {
                                        if e.trigger_button() == Some(MouseButton::Primary) {
                                            arming.set(Some(slot));
                                        }
                                    },
                                    // The release ends it, or the pointer leaving —
                                    // which is what sliding off a button has always
                                    // meant. Taking the class off cancels the
                                    // animation, and a cancelled animation never
                                    // ends: a tap cannot clear.
                                    onpointerup: move |_| arming.set(None),
                                    onpointercancel: move |_| arming.set(None),
                                    onpointerleave: move |_| arming.set(None),
                                    // The fill closing is the act. By name, so an
                                    // animation the icon inside might one day carry
                                    // cannot end a hold by bubbling up to here.
                                    onanimationend: move |e| {
                                        if e.animation_name() != "slot-clear-arm" {
                                            return;
                                        }
                                        arming.set(None);
                                        // The press is spent: the release that
                                        // follows is nobody's click.
                                        pressed.set(None);
                                        clear(state, slot);
                                    },
                                    {icon(stark_ui::icons::REMOVE)}
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Make `slot`'s brush the live one for good — what clicking a row of the pinned
/// rack does, and the only way to a slot for a hand with no keyboard under it (a
/// pen in one hand and a tablet under the other leaves no spare finger for the
/// number row).
///
/// Also where **tapping the number twice** arrives, by way of the hold itself:
/// the second press enters a hold whose release keeps the slot's brush rather
/// than putting the displaced one back ([`Held::picked`]), so this is not called
/// for it and there is no second path to the same place. Deliberately *not*
/// what a single tap does. A tap and a hold are the same keystroke told apart
/// only by how long it lasted, so binding them to different outcomes would make
/// every hold a race against the user's own reflexes; two presses are told apart
/// by *count*, and the first is an ordinary hold that does nothing unused. A
/// click is its own gesture and says what it means.
///
/// Clicking a row *during* a hold is left to mean what the one rule already says
/// it means: the click makes that slot's brush live, and the hold then keeps
/// whatever is live when it ends — so holding 3 and clicking 5 copies 5 onto 3,
/// binding and tune alike, exactly as holding 3 and clicking a preset assigns
/// that preset. One rule, not a special case (`Held::settle`), and it is a whole
/// tool arriving, so it says so ([`claim`]) — copying 5 onto 3 must work when 5
/// is already what is in hand.
pub fn pick(state: AppState, slot: usize) {
    let bound = state.slots.brushes.peek().get(slot).cloned().flatten();
    let Some(bound) = bound else { return };
    // A binding the library cannot answer is an empty row, and an empty row's
    // click puts on nothing.
    let Some((brush, tune)) = resolve_in(state, &bound) else {
        return;
    };
    claim(state);
    presets::wear(state, brush, tune, Some(bound.preset));
}

/// Pin the rack up or put it away, and remember it — **the only thing that writes
/// [`SlotState::pinned`](crate::state::SlotState::pinned)**, which is what makes
/// durability structural rather than a line the menu row has to remember (the move
/// `navigator::set_open` and `layout::set_open` both make).
///
/// Pinning is not the same question as the rack being *up*: while a number is held
/// the rack shows whatever this says, and what pinning buys is a rack that stays and
/// takes clicks — the only route to a slot for a hand with no keyboard under it
/// (§18.1.8). That standing choice is the one worth keeping, and it is kept with the
/// rest of what is on screen (`crate::visibility`, §25.6).
///
/// Guarded on the value actually moving, like the other two: a `Signal` write dirties
/// every subscriber whether or not the value changed, and the rack's subscribers
/// include the overlay that renders it.
pub fn set_pinned(state: AppState, pinned: bool) {
    let mut up = state.slots.pinned;
    // Into a `bool` before the write, as everything that toggles a signal here does.
    let was = *up.peek();
    if was == pinned {
        return;
    }
    up.set(pinned);
    crate::visibility::persist(state);
}

/// Bind `slot` to `brush` — a preset, at a size and flow — and persist the rack.
pub fn assign(state: AppState, slot: usize, brush: QuickBrush) {
    if slot >= COUNT {
        return;
    }
    let mut brushes = state.slots.brushes;
    brushes.write()[slot] = Some(brush);
    persist(&brushes.read());
}

/// Empty `slot` and persist the rack — the trash on a pinned row, held down until
/// its fill closes ([`SlotOverlay`]).
///
/// The live brush is untouched, exactly as removing a preset leaves it
/// (`presets::remove`): what goes is the *binding*, not the tool. A slot cleared
/// while it is being held is cleared for good — the release then finds nothing
/// changed and keeps nothing, so the emptying stands and the displaced brush
/// still comes back.
///
/// Clearing the last filled slot leaves a rack that is empty rather than unset,
/// and the two must not be confused: see [`read_storage`].
pub fn clear(state: AppState, slot: usize) {
    if slot >= COUNT {
        return;
    }
    let mut brushes = state.slots.brushes;
    // Its own statement, this module's rule: a `peek` in a condition stays
    // borrowed through the body, and the write below is of the very signal read.
    let already_empty = brushes.peek()[slot].is_none();
    if already_empty {
        return;
    }
    brushes.write()[slot] = None;
    persist(&brushes.read());
}

/// Empty every slot bound to the preset called `name` — what removing a preset
/// does to the rack (`presets::remove`). A slot is a name and a tune, and a name
/// the library no longer answers to holds nothing; emptying it outright keeps
/// the rack honest about what it holds rather than leaving a row that looks
/// empty and secretly is not.
///
/// A write, and so a persist, only where something was bound: an ordinary
/// removal touches storage once, for the library.
pub fn unbind(state: AppState, name: &str) {
    let mut brushes = state.slots.brushes;
    let bound_to = |b: &Option<QuickBrush>| b.as_ref().is_some_and(|b| b.preset == name);
    // Its own statement, for the module's rule on a `peek` in a condition.
    let any = brushes.peek().iter().any(bound_to);
    if !any {
        return;
    }
    for slot in brushes.write().iter_mut() {
        if bound_to(slot) {
            *slot = None;
        }
    }
    persist(&brushes.read());
}

/// Populate the rack from storage. Called once at app start, before the renderer
/// exists — nothing here needs an engine, or even the library: a slot is a name
/// until it is resolved, and it is resolved at every use ([`resolve`]).
pub fn load(state: AppState) {
    let mut brushes = state.slots.brushes;
    if let Some(rack) = read_storage() {
        brushes.set(rack);
    }
}

/// Fill a rack that has never been set from the library: every preset that
/// declares a home digit is bound to it, at the size and flow it ships at
/// ([`PresetEntry::slot`]).
///
/// Called after `presets::install_builtins`, and reading *from* the library
/// rather than restating it — so a tool reaches the keyboard under the name the
/// panel lists it by, and adding a shipped preset puts it on a digit by writing
/// one field instead of by editing a second table here. The rack is why this
/// module holds no brush of its own: what a slot starts as is a question about
/// the app's tools, and `crate::presets` is where those are defined.
///
/// **Seeded in memory and not persisted.** Storage is written only by the user's
/// own act ([`assign`]), which is what `read_storage().is_some()` then means: not
/// "this browser has run Stark before" but "this browser has set a slot". That
/// keeps the seed live — a shipped preset moved to another digit reaches the
/// rack on the next start, exactly as its edits reach the preset list — and it
/// costs nothing that a binding does not already buy, since the *tool* on a
/// seeded digit is looked up live either way.
pub fn seed_defaults(state: AppState) {
    if read_storage().is_some() {
        return;
    }
    let mut rack = empty_rack();
    for entry in state.presets.peek().iter() {
        // A slot past the rack is a definition to fix, not a panic to take: the
        // preset is still perfectly usable from the list.
        if let Some(slot) = entry.slot.filter(|s| *s < COUNT) {
            rack[slot] = Some(QuickBrush {
                preset: entry.name.clone(),
                transient: entry.transient,
            });
        }
    }
    let mut brushes = state.slots.brushes;
    brushes.set(rack);
}

// --- persistence ----------------------------------------------------------
//
// One `stark_ui::storage` entry per **assigned** slot. The format and the
// skip-a-damaged-entry rule live there, so what is this module's own is that an entry
// names its digit rather than sitting at a position: a rack with holes stores as the
// few entries it has, and one whose digit is out of range is dropped instead of
// shifting its neighbours. What an entry holds is a binding ([`QuickBrush`]) — a
// preset's name and a size and flow — and never a brush: the tool is the preset's,
// and the preset library is the one store that holds one.

/// One assigned slot.
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct StoredSlot {
    digit: usize,
    #[serde(flatten)]
    brush: QuickBrush,
}

impl storage::Entry for StoredSlot {
    const STORE: Store = Store::Slots;
}

fn persist(rack: &Rack) {
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

/// `None` when this browser has never set a slot (or storage is unavailable) —
/// the caller seeds the rack. **`Some(empty)` is a different answer**, and the
/// difference is load-bearing now that a slot can be cleared ([`clear`]): a rack
/// the user has emptied to the last digit is left empty, where an untouched one
/// is seeded from the library ([`seed_defaults`]). A rack that re-seeded itself
/// the moment it was emptied would make the trash on the last row do nothing at
/// all — the same emptied-versus-absent case the preset library has, and
/// `storage::load_list` keeps them apart for both.
fn read_storage() -> Option<Rack> {
    let mut rack = empty_rack();
    for entry in storage::load_list::<StoredSlot>()? {
        // A digit past the rack is an entry a shorter build cannot place, and
        // dropping it is the only answer that does not move its neighbours.
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
        Held {
            slot: 3,
            grip: Grip::Key,
            base: (base, entered),
            base_from: None,
            entered,
            claimed: false,
            picked: false,
        }
    }

    fn bound(preset: &str, transient: Transient) -> Option<QuickBrush> {
        Some(QuickBrush {
            preset: preset.into(),
            transient,
        })
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
        // Bound to the preset in hand, at the size that was dragged — and the
        // brush in hand comes back untouched.
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
        // The number keeps its binding and the two knobs of the tune: an
        // opacity moved under the hold or a hardness changed reach the config
        // alone, and a **color picked reaches the tune but is the Color
        // panel's act** (`Transient::same_tune`) — so none of them is a change
        // the number can keep, and the release finds nothing to do.
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
        // preset you are already painting with. Nothing changes, and the slot has
        // to end up bound to it all the same.
        let brush = BrushConfig::default();
        let entered = Transient::default();
        let h = Held {
            claimed: true,
            ..held(entered, brush)
        };
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
        // A brush descending from no preset the library has is a tool the rack
        // cannot name: a tune dragged under the hold has nothing to be kept
        // against, and even a claim keeps nothing.
        let base = BrushConfig::default();
        let entered = tune(40.0, 1.0);
        let (kept, back) = held(entered, base).settle(tune(64.0, 1.0), None);
        assert_eq!(kept, None);
        assert_eq!(back, Some((base, entered)));
        let h = Held {
            claimed: true,
            ..held(entered, base)
        };
        assert_eq!(h.settle(entered, None).0, None);
    }

    #[test]
    fn a_claim_still_hands_the_displaced_brush_back() {
        // The half of the rule the claim must not touch: what the number keeps is
        // one question, what comes back to the hand is the other.
        let base = BrushConfig::default();
        let entered = tune(40.0, 1.0);
        let h = Held {
            claimed: true,
            ..held(entered, base)
        };
        let (kept, back) = h.settle(entered, Some("Pen"));
        assert_eq!(kept, bound("Pen", entered));
        assert_eq!(back, Some((base, entered)));
    }

    #[test]
    fn a_double_tapped_number_stays_in_hand() {
        // The click's outcome by way of the keyboard (`pick`): the hold hands
        // nothing back, so what was swapped in at the press is what the hand
        // keeps.
        let base = BrushConfig::default();
        let entered = tune(40.0, 1.0);
        let h = Held {
            picked: true,
            ..held(entered, base)
        };
        let (kept, back) = h.settle(entered, Some("Pen"));
        assert_eq!(
            kept, None,
            "untouched, the number keeps nothing, tapped twice or not"
        );
        assert_eq!(back, None, "the displaced brush is not put back");
    }

    #[test]
    fn tuning_under_a_double_tap_reaches_the_number_and_stays() {
        // The one rule is still in force for the second press: the change goes to
        // the number — and, this once, stays in hand as well.
        let base = BrushConfig::default();
        let entered = tune(40.0, 1.0);
        let h = Held {
            picked: true,
            ..held(entered, base)
        };
        let (kept, back) = h.settle(tune(64.0, 1.0), Some("Pen"));
        assert_eq!(kept, bound("Pen", tune(64.0, 1.0)));
        assert_eq!(back, None);
    }

    #[test]
    fn a_slot_is_its_preset_at_its_own_size_and_flow() {
        // The binding is live: the tool is whatever the library holds under the
        // name today, and only the size and flow are the slot's. A name the
        // library does not have is an empty slot.
        let pen = BrushConfig {
            smoothing: 0.5,
            ..BrushConfig::default()
        };
        let library = vec![PresetEntry {
            name: "Pen".into(),
            brush: pen,
            transient: tune(18.0, 1.0),
            slot: None,
            builtin: false,
        }];
        let slot = QuickBrush {
            preset: "Pen".into(),
            transient: tune(64.0, 0.2),
        };
        let (tool, tune) = resolve(&library, &slot).expect("the library has the name");
        assert_eq!(tune, self::tune(64.0, 0.2), "the slot's own tune");
        assert!(
            stark_ui::presets::same_tool(&tool, &pen),
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
}
