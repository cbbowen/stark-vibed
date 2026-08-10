//! The ten brushes under the hand: hold a number, paint with that brush (§18.1.8).
//!
//! Other painting applications bind keys to *tools* — B for brush, E for eraser,
//! R for blur. Stark has no such list to bind: an eraser is a brush whose
//! `lift` is up and whose `add` is zero, a blur is one with `bleed` up, and both
//! are points in the same parameter space as the brush that made them (§6.2). A
//! key that selected a tool would have to select a *brush*, and which brush is
//! the artist's answer, not ours.
//!
//! So the numbers hold brushes, and there is exactly one rule:
//!
//! > **A held number is a temporary swap of the live brush.** Whatever you change
//! > while it is held stays with the number; the brush you were holding comes
//! > back when you let go.
//!
//! Everything the feature does falls out of that rather than being wired up
//! three times:
//!
//! - **Hold and draw** — the stroke is laid with the slot's brush, because the
//!   slot's brush *is* the live brush for the length of the hold, and a stroke
//!   snapshots the brush at `Start` (`Session::start_stroke`).
//! - **Hold and click a preset** — the preset lands on the live brush, so at
//!   release it is what the number keeps.
//! - **Hold and drag Size or Flow** — the panel's sliders write the live brush,
//!   as they always did, so the number keeps the number that was dragged. The
//!   panel needs to know nothing about any of this: it shows the live brush, and
//!   the live brush is the slot's.
//! - **Flip the pen over** — the eraser end holds [`ERASER`] for as long as it is
//!   on the glass, whatever it is pressed against. It is the same hold, made by
//!   hardware instead of by a key, and being bound at the window rather than by
//!   any one surface (`input::bind_pen`) it earns the three lines above rather
//!   than only the first: erasing with the tail is one gesture, tuning the
//!   eraser's Size or Flow with it is another, and eraser-clicking a preset
//!   assigns that preset to the tail. A key and a hand do the same thing here.
//!
//! What the rule deliberately does *not* carry is the painting **colour**, which
//! is [`presets::wear`]'s rule and is applied in both directions: a slot never
//! changes the colour you are painting with, and a colour picked during a hold
//! survives the release. A slot is a tool; colour belongs to the Color panel.
//!
//! Like the shape and preset libraries the rack follows this browser rather than
//! the document (`localStorage`), and degrades to a per-session rack where
//! storage is unavailable.

use dioxus::prelude::*;

use stark_core::document::BrushParams;

use crate::platform::{base64_decode, base64_encode};
use crate::presets;
use crate::state::AppState;

/// One key, namespaced like the shape and preset libraries'; versioned so a
/// future format change can migrate rather than mis-parse.
const KEY_SLOTS: &str = "stark.slots.v1";

/// How many quick brushes there are — one per digit.
pub const COUNT: usize = 10;

/// The slot the pen's other end holds, and the one the `0` key holds: the digit
/// past the nine, which is where a tenth of anything goes on a keyboard.
pub const ERASER: usize = 0;

/// The rack: one optional brush per digit, indexed **by the digit itself**, so
/// `rack[3]` is what the `3` key holds and there is no off-by-one to get wrong
/// between the keyboard, the panel and storage.
pub type Rack = [Option<BrushParams>; COUNT];

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
    /// The pen's eraser end, against the glass.
    Eraser,
}

/// A hold in flight: which slot, what is holding it, and the two brushes the
/// release is decided against.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Held {
    /// The digit being held.
    pub slot: usize,
    grip: Grip,
    /// The brush the hold displaced — what comes back when it ends.
    base: BrushParams,
    /// The brush the hold *began* on, once the swap had happened. What the
    /// release compares against to decide whether anything was changed, and so
    /// whether the number has something new to keep.
    entered: BrushParams,
}

impl Held {
    /// What the release does: the brush to keep in the slot (`None` when nothing
    /// was changed), and the brush to put back.
    ///
    /// Split out as a pure function because it is the whole rule, and the rule is
    /// the part worth being sure of. Two properties it has to have:
    ///
    /// - **An unused hold keeps nothing.** Holding 5 and drawing must not quietly
    ///   make 5 the brush you happened to be holding — the numbers are assigned
    ///   deliberately or not at all, and an empty slot that filled itself the
    ///   first time it was pressed would be impossible to tell from one the user
    ///   had set.
    /// - **A colour change is not a change.** Compared through
    ///   [`presets::matches`], which is exactly "the same brush, colour aside" —
    ///   the same test the preset rows highlight on. Picking a colour mid-hold
    ///   would otherwise write the whole brush into the slot, which is the one
    ///   thing a slot is defined not to carry.
    fn settle(&self, current: BrushParams) -> (Option<BrushParams>, BrushParams) {
        let kept = (!presets::matches(&current, &self.entered)).then_some(current);
        (kept, self.base)
    }
}

/// Begin holding `slot`. Ignored when a hold is already in flight — which is what
/// makes it safe to call on every keydown, since a held key repeats at the
/// system's repeat rate and each repeat is another keydown.
///
/// A slot with nothing in it still enters the hold rather than declining: the
/// hold *is* the arming, and holding an empty number while clicking a preset is
/// how the number gets its first brush.
pub fn hold(state: AppState, slot: usize, grip: Grip) {
    if slot >= COUNT {
        return;
    }
    let mut held = state.slots.held;
    if held.peek().is_some() {
        return;
    }
    // Each read is its own statement, so no guard is alive when `wear` dispatches
    // and rewrites the observable underneath it (`state::update_brush`).
    let base = state.obs.peek().as_ref().map(|o| o.brush);
    let Some(base) = base else { return };
    let assigned = state.slots.brushes.peek()[slot];
    if let Some(brush) = assigned {
        presets::wear(state, brush);
    }
    // Read back rather than assumed: `wear` keeps the live colour and resolves the
    // stamp, so what the engine now holds is not what was handed to it — and it is
    // what the release has to compare against.
    let entered = state.obs.peek().as_ref().map(|o| o.brush).unwrap_or(base);
    held.set(Some(Held {
        slot,
        grip,
        base,
        entered,
    }));
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
    let Some(h) = *held.peek() else { return };
    if h.slot != slot || h.grip != grip {
        return;
    }
    held.set(None);
    let current = state.obs.peek().as_ref().map(|o| o.brush);
    let Some(current) = current else { return };
    let (kept, back) = h.settle(current);
    if let Some(brush) = kept {
        assign(state, h.slot, brush);
    }
    presets::wear(state, back);
}

/// End whatever hold is in flight, whoever made it — for the one event that can
/// take a key away without ever sending its keyup: the window losing focus.
///
/// Alt+Tab with a number held would otherwise leave the swap in force for the
/// rest of the session, with the key that would undo it now belonging to another
/// window. The same class of bug the eyedropper's `track_alt` rules out by
/// re-reading the modifier set, ruled out here by the event that says the
/// keyboard has gone.
pub fn release_all(state: AppState) {
    let held = *state.slots.held.peek();
    if let Some(h) = held {
        release(state, h.slot, h.grip);
    }
}

/// Make `slot`'s brush the live one for good — what clicking a chip in the rack
/// does, and the mouse-only way to reach a slot at all (a hand on a tablet has no
/// spare finger for the number row).
///
/// Deliberately *not* what tapping the number key does. A tap and a hold are the
/// same keystroke told apart only by how long it lasted, so binding them to
/// different outcomes would make every hold a race against the user's own
/// reflexes. A click is its own gesture and says what it means.
pub fn pick(state: AppState, slot: usize) {
    let brush = state.slots.brushes.peek().get(slot).copied().flatten();
    if let Some(brush) = brush {
        presets::wear(state, brush);
    }
}

/// Put `brush` in `slot` and persist the rack.
pub fn assign(state: AppState, slot: usize, brush: BrushParams) {
    if slot >= COUNT {
        return;
    }
    let mut brushes = state.slots.brushes;
    brushes.write()[slot] = Some(brush);
    persist(&brushes.read());
}

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

/// Populate the rack from storage. Called once at app start, before the renderer
/// exists — nothing here needs an engine, since a slot is only a snapshot until
/// it is worn.
pub fn load(state: AppState) {
    let mut brushes = state.slots.brushes;
    if let Some(rack) = read_storage() {
        brushes.set(rack);
    }
}

/// Fill a rack that has never been set from the library: every preset that
/// declares a home digit goes to it ([`PresetEntry::slot`]).
///
/// Called after `presets::install_builtins`, and reading *from* the library
/// rather than restating it — so a tool reaches the keyboard under the same name
/// and with the same parameters the panel lists it by, and adding a shipped
/// preset puts it on a digit by writing one field instead of by editing a second
/// table here. The rack is why this module holds no brush of its own: what a
/// slot starts as is a question about the app's tools, and `crate::presets` is
/// where those are defined.
///
/// **Seeded in memory and not persisted.** Storage is written only by the user's
/// own act ([`assign`]), which is what `read_storage().is_some()` then means: not
/// "this browser has run Stark before" but "this browser has set a slot". That
/// keeps the seed live — an improved default reaches the rack on the next start,
/// exactly as it reaches the preset list — and it means a start whose bundled
/// shapes failed to fetch cannot freeze a degraded pencil into slot 3, because
/// it writes nothing at all.
pub fn seed_defaults(state: AppState) {
    if read_storage().is_some() {
        return;
    }
    let mut rack: Rack = [None; COUNT];
    for entry in state.presets.peek().iter() {
        // A slot past the rack is a definition to fix, not a panic to take: the
        // preset is still perfectly usable from the list.
        if let Some(slot) = entry.slot.filter(|s| *s < COUNT) {
            rack[slot] = Some(entry.brush);
        }
    }
    let mut brushes = state.slots.brushes;
    brushes.set(rack);
}

// --- persistence ----------------------------------------------------------
//
// One storage key holding one line per **assigned** slot: `digit|b64(json)`.
// Line-oriented and field-delimited like the shape and preset libraries, so one
// damaged entry costs one slot rather than the rack; keyed by digit rather than
// positional, so a rack with holes in it stores as the few lines it has and a
// line whose digit is out of range is skipped instead of shifting its
// neighbours. JSON for the reason the presets give: `localStorage` outlives app
// versions, and a `BrushParams` field added later still reads every stored slot.

fn persist(rack: &Rack) {
    let Some(store) = storage() else { return };
    let text: Vec<String> = rack
        .iter()
        .enumerate()
        .filter_map(|(slot, brush)| {
            let json = serde_json::to_string(&(*brush)?).ok()?;
            Some(format!("{slot}|{}", base64_encode(json.as_bytes())))
        })
        .collect();
    if store.set_item(KEY_SLOTS, &text.join("\n")).is_err() {
        // Quota, most likely. The rack still works for this session.
        tracing::warn!("could not persist the quick brushes (storage full or unavailable)");
    }
}

/// `None` when this browser has never set a slot (or storage is unavailable) —
/// the caller seeds the rack. A rack emptied down to nothing is unreachable (a
/// slot is overwritten, never cleared), so unlike the preset library there is no
/// emptied-versus-absent case to tell apart.
fn read_storage() -> Option<Rack> {
    let text = storage()?.get_item(KEY_SLOTS).ok()??;
    let mut rack: Rack = [None; COUNT];
    for (slot, brush) in text.lines().filter_map(parse_entry) {
        rack[slot] = Some(brush);
    }
    Some(rack)
}

fn parse_entry(line: &str) -> Option<(usize, BrushParams)> {
    let mut fields = line.split('|');
    let slot: usize = fields.next()?.parse().ok()?;
    if slot >= COUNT {
        return None;
    }
    let brush = serde_json::from_slice(&base64_decode(fields.next()?).ok()?).ok()?;
    Some((slot, brush))
}

fn storage() -> Option<web_sys::Storage> {
    web_sys::window()?.local_storage().ok().flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn held(entered: BrushParams, base: BrushParams) -> Held {
        Held {
            slot: 3,
            grip: Grip::Key,
            base,
            entered,
        }
    }

    #[test]
    fn a_hold_that_changed_nothing_keeps_nothing() {
        let brush = BrushParams::default();
        let h = held(brush, brush);
        let (kept, back) = h.settle(brush);
        assert_eq!(kept, None, "an unused hold must not claim the slot");
        assert_eq!(back, brush);
    }

    #[test]
    fn a_changed_brush_stays_with_the_number() {
        let base = BrushParams::default();
        let entered = BrushParams {
            radius: 40.0,
            ..base
        };
        let dragged = BrushParams {
            radius: 64.0,
            ..entered
        };
        let (kept, back) = held(entered, base).settle(dragged);
        assert_eq!(kept, Some(dragged));
        assert_eq!(back, base, "the brush in hand comes back untouched");
    }

    #[test]
    fn a_colour_picked_mid_hold_is_not_a_change() {
        // The one edit a slot is defined not to carry (see `presets::wear`).
        let base = BrushParams::default();
        let entered = BrushParams {
            radius: 40.0,
            ..base
        };
        let recoloured = BrushParams {
            color: [0.9, 0.1, 0.2, entered.color[3]],
            ..entered
        };
        let (kept, _) = held(entered, base).settle(recoloured);
        assert_eq!(kept, None, "colour is not part of a slot");
    }

    #[test]
    fn the_brush_opacity_is_part_of_a_slot_though() {
        // `color[3]` is the brush's own opacity — a material property, not the
        // colour choice (§6.1) — and the preset library carries it for the same
        // reason. Dragging it under a hold has to reach the number.
        let base = BrushParams::default();
        let entered = base;
        let thinned = BrushParams {
            color: [base.color[0], base.color[1], base.color[2], 0.4],
            ..base
        };
        let (kept, _) = held(entered, base).settle(thinned);
        assert_eq!(kept, Some(thinned));
    }

    #[test]
    fn codes_map_to_the_digit_they_are() {
        assert_eq!(of_code("Digit0"), Some(ERASER));
        assert_eq!(of_code("Digit3"), Some(3));
        assert_eq!(of_code("Digit9"), Some(9));
        // The keypad is the same rack.
        assert_eq!(of_code("Numpad7"), Some(7));
        // Everything else is somebody else's key.
        assert_eq!(of_code("KeyH"), None);
        assert_eq!(of_code("Digit"), None);
        assert_eq!(of_code("NumpadAdd"), None);
        assert_eq!(of_code("Numpad10"), None);
    }
}
