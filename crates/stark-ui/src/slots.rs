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
//! What the rule deliberately does *not* carry is the painting **color**, which
//! is [`presets::wear`]'s rule and is applied in both directions: a slot never
//! changes the color you are painting with, and a color picked during a hold
//! survives the release. A slot is a tool; color belongs to the Color panel.
//!
//! Like the shape and preset libraries the rack follows this browser rather than
//! the document (`localStorage`), and degrades to a per-session rack where
//! storage is unavailable.
//!
//! The rack draws itself while a number is held ([`SlotOverlay`]) — a column of
//! the brushes the digits carry, each as the rendered stroke the preset library
//! shows. The Panels menu can keep it up ("Quick brushes"), which is what makes
//! it clickable: the rule above needs a keyboard, and a rack that can be clicked
//! is the whole of what a hand without one gets ([`pick`]).

use dioxus::prelude::*;

use crate::icons::{self, icon};
use crate::layout::chrome_class;
use crate::presets::{self, Wearable};
use crate::state::AppState;
use crate::storage;

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
pub type Rack = [Option<Wearable>; COUNT];

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
    base: Wearable,
    /// The brush the hold *began* on, once the swap had happened. What the
    /// release compares against to decide whether anything was changed, and so
    /// whether the number has something new to keep.
    entered: Wearable,
}

impl Held {
    /// Whether a **key** is what is holding the slot down, rather than the pen's
    /// tail. What [`SlotOverlay`] is mounted on — see there for why the two
    /// grips are told apart for showing when they are alike for everything else.
    pub fn by_key(&self) -> bool {
        matches!(self.grip, Grip::Key)
    }

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
    /// - **A color change is not a change.** Compared through
    ///   [`presets::matches`], which is exactly "the same brush, color aside" —
    ///   the same test the preset rows highlight on. Picking a color mid-hold
    ///   would otherwise write the whole brush into the slot, which is the one
    ///   thing a slot is defined not to carry.
    fn settle(&self, current: Wearable) -> (Option<Wearable>, Wearable) {
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
    let Some(base) = presets::worn(state) else {
        return;
    };
    let assigned = state.slots.brushes.peek()[slot];
    if let Some(brush) = assigned {
        presets::wear(state, brush);
    }
    // Read back rather than assumed: `wear` keeps the live color and resolves the
    // stamp, so what the engine now holds is not what was handed to it — and it is
    // what the release has to compare against.
    let entered = presets::worn(state).unwrap_or(base);
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
    let Some(current) = presets::worn(state) else {
        return;
    };
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
/// Up while a number is held, and **kept** up by the Panels menu's "Quick
/// brushes" ([`AppState::slots`]`.pinned`) — which is the same picture with one
/// thing added: pinned, the rows take clicks, and clicking one is [`pick`].
/// That is the mouse-only way to a slot, and it is why the pin exists at all —
/// a pen in one hand and a tablet under it leaves no spare finger for the number
/// row, and the transient rack cannot be clicked (see below).
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
/// It **fades with the rest of the floating chrome** (`layout::chrome_class`),
/// which goes to nothing while a canvas gesture is in flight: it stands over the
/// painting like the panels and the bars, and while a stroke is being laid the
/// screen goes back to being the painting. What makes that safe on something
/// this momentary is that **the hold outlives the stroke** — the key is still
/// down when the pen lifts, so the rack comes straight back and the answer is
/// there whenever the hand wants it rather than only before the first mark.
///
/// The **held** row is ringed and the row the live brush *is* (color aside) is
/// lit, which are two different questions and only look like one while a key is
/// down: pinned and idle, the lit row is the only thing saying which slot is in
/// hand, and it is the same test and the same light the preset rows use. Reading
/// the live brush costs nothing per stroke — a sample dispatches quietly and
/// never refreshes the observable (`state::dispatch_sample`), so this re-renders
/// when the brush *changes*, not while one is painting.
///
/// The hold that mounts it is a **key** hold alone, which is the one place the
/// two grips are told apart rather than being the same hold. The pen's tail holds
/// [`ERASER`] for as long as it is on the glass (`input::bind_pen`) — that is
/// every erase stroke — and a rack flying in and out of the corner of the eye on
/// each one is noise answering a question nobody asked. Holding `0` shows the
/// same row.
#[component]
pub fn SlotOverlay() -> Element {
    let state = use_context::<AppState>();
    let pinned = (state.slots.pinned)();
    let held = (state.slots.held)().filter(Held::by_key).map(|h| h.slot);
    // Neither held nor pinned: read nothing else at all, so a rack, a library or a
    // brush that changes during a stroke cannot re-render anything.
    if !pinned && held.is_none() {
        return rsx! {};
    }
    let rack = (state.slots.brushes)();
    // The whole tool, feel included (§6.11) — subscribing reads, so the lit row
    // tracks a smoothing drag exactly as it tracks a radius one.
    let live = presets::Wearable {
        params: state
            .obs
            .read()
            .as_ref()
            .map(|o| o.brush)
            .unwrap_or_default(),
        smoothing: (state.smoothing)(),
    };
    // The rows are resolved against the library up front, so no read guard is
    // alive while the rows below read the thumbnail cache one by one.
    let rows: Vec<(usize, Option<Wearable>, String)> = {
        let library = state.presets.read();
        // The digits in the order they sit on the keyboard, with the eraser's own
        // slot last — where the `0` key is, and where a tenth of anything goes.
        (1..COUNT)
            .chain(std::iter::once(ERASER))
            .filter(|&slot| rack[slot].is_some() || Some(slot) == held)
            .map(|slot| {
                let brush = rack[slot];
                // The library's name for it, where the slot still *is* one of the
                // presets (color aside). A slot tuned away from the preset it came
                // from has no name to give and carries none — the preview is what
                // it is now, which is the honest answer.
                let label = match brush.map(|b| presets::name_of(&library, &b)) {
                    Some(Some(name)) => name,
                    Some(None) => String::new(),
                    None => "Empty \u{2014} click a preset to fill it".to_string(),
                };
                (slot, brush, label)
            })
            .collect()
    };

    let mut class = chrome_class(state, "slot-overlay");
    if pinned {
        class.push_str(" pinned");
    }
    rsx! {
        div { class,
            for (slot, brush, label) in rows {
                {
                    // The brush as a stroke, filling the row as its background —
                    // the preset row's own recipe (`panels::brush`), down to
                    // writing `none` out rather than omitting the property: these
                    // nodes are reused as the rack changes under them, and a
                    // stranded declaration would leave one slot showing another's
                    // brush (inline style merges per property).
                    let bg = match brush.and_then(|b| crate::thumbs::url(state, &b)) {
                        Some(url) if !url.is_empty() => format!("background-image: url({url});"),
                        _ => "background-image: none;".to_string(),
                    };
                    // Lit like a preset row and on the same test — this is the
                    // brush in hand, color aside, until any knob moves off it.
                    // Held wins in the stylesheet: it is what the user is doing
                    // right now rather than a state they are in.
                    let mut class = String::from("slot-row");
                    if brush.is_none() {
                        class.push_str(" empty");
                    } else if brush.is_some_and(|b| presets::matches(&live, &b)) {
                        class.push_str(" active");
                    }
                    if Some(slot) == held {
                        class.push_str(" held");
                    }
                    // Says the binding a picture cannot, and only ever seen pinned
                    // — an element that takes no pointer is shown no tooltip. The
                    // eraser's names the pen rather than the key, because the thing
                    // that reaches it is already in the hand.
                    let title = match (slot, brush.is_some()) {
                        (ERASER, true) => "Click to paint with this, or flip the pen over".to_string(),
                        (ERASER, false) => "Empty. Hold 0 (or the pen's eraser end) and click a preset".to_string(),
                        (n, true) => format!("Click to paint with this, or hold {n}"),
                        (n, false) => format!("Empty. Hold {n} and click a preset to fill it"),
                    };
                    rsx! {
                        div {
                            key: "{slot}",
                            class,
                            style: "{bg}",
                            title,
                            // Reachable only while pinned, the stylesheet having
                            // given the rows the pointer there and nowhere else —
                            // the same way faded chrome stops taking clicks
                            // (`.chrome.dimmed`) rather than by every handler
                            // asking whether it is visible.
                            onclick: move |_| pick(state, slot),
                            span { class: "slot-row-digit",
                                if slot == ERASER {
                                    {icon(icons::ERASER)}
                                } else {
                                    "{slot}"
                                }
                            }
                            span { class: "slot-row-name", "{label}" }
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
/// Deliberately *not* what tapping the number key does. A tap and a hold are the
/// same keystroke told apart only by how long it lasted, so binding them to
/// different outcomes would make every hold a race against the user's own
/// reflexes. A click is its own gesture and says what it means.
///
/// Clicking a row *during* a hold is left to mean what the one rule already says
/// it means: the click makes that slot's brush live, and the hold then keeps
/// whatever is live when it ends — so holding 3 and clicking 5 copies 5 onto 3,
/// exactly as holding 3 and clicking a preset assigns that preset. One rule, not
/// a special case (`Held::settle`).
pub fn pick(state: AppState, slot: usize) {
    let brush = state.slots.brushes.peek().get(slot).copied().flatten();
    if let Some(brush) = brush {
        presets::wear(state, brush);
    }
}

/// Put `brush` in `slot` and persist the rack.
pub fn assign(state: AppState, slot: usize, brush: Wearable) {
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
// One `crate::storage` table, a line per **assigned** slot: `digit|b64(json)`.
// The format and the skip-a-damaged-line rule live there, so what is this
// module's own is that the record is keyed by digit rather than positional: a
// rack with holes stores as the few lines it has, and a line whose digit is out
// of range is skipped instead of shifting its neighbours. JSON for the reason
// the presets give: `localStorage` outlives app versions, and a `BrushParams`
// field added later still reads every stored slot.
// The brush fields themselves are the preset library's own wire shape
// (`presets::encode_wearable`), so the two libraries cannot come to disagree
// about what a stored brush is — including the trailing feel field (§6.11),
// optional there and optional here.

fn persist(rack: &Rack) {
    storage::save_table(
        KEY_SLOTS,
        "the quick brushes",
        rack.iter().enumerate().filter_map(|(slot, brush)| {
            Some(storage::record([
                slot.to_string().as_str(),
                presets::encode_wearable(&(*brush)?)?.as_str(),
            ]))
        }),
    );
}

/// `None` when this browser has never set a slot (or storage is unavailable) —
/// the caller seeds the rack. A rack emptied down to nothing is unreachable (a
/// slot is overwritten, never cleared), so unlike the preset library there is no
/// emptied-versus-absent case to tell apart.
fn read_storage() -> Option<Rack> {
    let mut rack: Rack = [None; COUNT];
    for (slot, brush) in storage::load_table(KEY_SLOTS, parse_entry)? {
        rack[slot] = Some(brush);
    }
    Some(rack)
}

fn parse_entry(line: &str) -> Option<(usize, Wearable)> {
    let mut fields = line.split(storage::FIELD);
    let slot: usize = fields.next()?.parse().ok()?;
    if slot >= COUNT {
        return None;
    }
    Some((slot, presets::decode_wearable(&mut fields)?))
}

#[cfg(test)]
mod tests {
    use stark_engine::document::BrushParams;

    use super::*;

    /// A whole tool from bare params — the tests' brushes carry no smoothing
    /// unless the feel is the point of the test.
    fn w(params: BrushParams) -> Wearable {
        Wearable {
            params,
            smoothing: 0.0,
        }
    }

    fn held(entered: Wearable, base: Wearable) -> Held {
        Held {
            slot: 3,
            grip: Grip::Key,
            base,
            entered,
        }
    }

    #[test]
    fn a_hold_that_changed_nothing_keeps_nothing() {
        let brush = w(BrushParams::default());
        let h = held(brush, brush);
        let (kept, back) = h.settle(brush);
        assert_eq!(kept, None, "an unused hold must not claim the slot");
        assert_eq!(back, brush);
    }

    #[test]
    fn a_changed_brush_stays_with_the_number() {
        let base = w(BrushParams::default());
        let entered = w(BrushParams {
            radius: 40.0,
            ..base.params
        });
        let dragged = w(BrushParams {
            radius: 64.0,
            ..entered.params
        });
        let (kept, back) = held(entered, base).settle(dragged);
        assert_eq!(kept, Some(dragged));
        assert_eq!(back, base, "the brush in hand comes back untouched");
    }

    #[test]
    fn a_color_picked_mid_hold_is_not_a_change() {
        // The one edit a slot is defined not to carry (see `presets::wear`).
        let base = w(BrushParams::default());
        let entered = w(BrushParams {
            radius: 40.0,
            ..base.params
        });
        let recolored = w(BrushParams {
            color: [0.9, 0.1, 0.2, entered.params.color[3]],
            ..entered.params
        });
        let (kept, _) = held(entered, base).settle(recolored);
        assert_eq!(kept, None, "color is not part of a slot");
    }

    #[test]
    fn the_brush_opacity_is_part_of_a_slot_though() {
        // `color[3]` is the brush's own opacity — a material property, not the
        // color choice (§6.1) — and the preset library carries it for the same
        // reason. Dragging it under a hold has to reach the number.
        let base = w(BrushParams::default());
        let entered = base;
        let thinned = w(BrushParams {
            color: [
                base.params.color[0],
                base.params.color[1],
                base.params.color[2],
                0.4,
            ],
            ..base.params
        });
        let (kept, _) = held(entered, base).settle(thinned);
        assert_eq!(kept, Some(thinned));
    }

    #[test]
    fn a_smoothing_drag_mid_hold_stays_with_the_number() {
        // The feel is part of what a tool *is* (§6.11): smoothing tuned under a
        // hold reaches the number exactly as a radius drag does.
        let base = w(BrushParams::default());
        let entered = base;
        let smoothed = Wearable {
            smoothing: 0.6,
            ..entered
        };
        let (kept, _) = held(entered, base).settle(smoothed);
        assert_eq!(kept, Some(smoothed));
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
