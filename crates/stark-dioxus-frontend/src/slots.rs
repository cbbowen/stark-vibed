//! This browser's half of the quick-brush rack (§18.1.8): the signals it lives in,
//! the acts that write them, and the overlay that draws it.
//!
//! **The rule is not here.** What a slot holds, what a press and a release each
//! decide, how two presses become a pick and which rows the rack draws are
//! `stark_ui::slots` — one function each, shared with the native frontend (§11.2). So
//! is the record. What is left is what only a frontend can do: reach the live brush
//! through [`presets::wear`], keep the four values in signals, and draw.
//!
//! Everything the feature does still falls out of the one rule rather than being
//! wired up three times:
//!
//! - **Hold and draw** — the stroke is laid with the slot's brush, because the slot's
//!   brush *is* the live brush for the length of the hold, and a stroke snapshots the
//!   brush at `Start` (`Session::start_stroke`).
//! - **Hold and click a preset** — the preset lands on the live brush, so at release
//!   it is what the number is bound to, at the size and flow the preset brought.
//! - **Hold and drag Size or Flow** — the Brush panel writes the live brush, as it
//!   always did. The panel shows the live brush; while a number is held the live
//!   brush is that number's, so it shows and edits the slot without a line of code
//!   that knows about slots.
//! - **Flip the pen over** — the eraser end holds [`ERASER`] for as long as its tail
//!   faces the glass, bound at the window rather than by any one surface
//!   (`input::bind_pen`), so it earns all three of the lines above.
//!
//! Like the shape and preset libraries the rack follows this browser rather than the
//! document (`localStorage`), and degrades to a per-session rack where storage is
//! unavailable.

use dioxus::html::input_data::MouseButton;
use dioxus::prelude::*;

use crate::icons::icon;
use crate::layout::chrome_dimmed;
use crate::presets;
use crate::state::AppState;
use stark_ui::brush_config::{BrushConfig, Transient};
use stark_ui::slots::{self, Digit, ERASER, Grip, Held, QuickBrush, Row, View};

/// [`slots::resolve`] against the app's library, for the callers that have only the
/// state: its own statement so the library's read guard is dropped before the caller
/// goes on to dispatch (`presets::wear` rewrites signals under it).
fn resolve_in(state: AppState, slot: &QuickBrush) -> Option<(BrushConfig, Transient)> {
    let library = state.presets.peek();
    slots::resolve(&library, slot)
}

/// Begin holding `slot`. Ignored when a hold is already in flight — which is what
/// makes it safe to call on every keydown, since a held key repeats at the system's
/// repeat rate and each repeat is another keydown, and on every report of a hovering
/// pen, which is what the tail's hold is read off (`input::tail_says`).
///
/// The one exception is [`Grip::displaces`]'s: a number deliberately pressed under a
/// hovering tail takes the brush from it.
///
/// A slot with nothing in it still enters the hold rather than declining: the hold
/// *is* the arming, and holding an empty number while clicking a preset is how the
/// number gets its first brush.
///
/// A **key** pressed twice within `slots::DOUBLE_TAP` enters its second hold *picked*:
/// the same hold, whose release keeps the slot's brush in hand instead of handing the
/// displaced one back — the click's outcome ([`pick`]) by way of the keyboard. Counted
/// below the guard on a hold in flight, so a held key's repeats are never presses; and
/// counted for keys alone, since the pen's tail is on the glass or off it, and two
/// dabs of it are two erase strokes.
pub fn hold(state: AppState, slot: Digit, grip: Grip) {
    let mut held = state.slots.held;
    // Answered without cloning the hold, since this runs on every report of a hovering
    // pen and nearly all of them are already holding.
    let displaced = {
        let in_flight = held.peek();
        match in_flight.as_ref() {
            Some(h) => match h.displaced_by(grip) {
                Some(pair) => Some(pair),
                None => return,
            },
            None => None,
        }
    };
    if let Some((slot, grip)) = displaced {
        release(state, slot, grip);
    }
    let mut taps = state.slots.taps;
    let picked = grip == Grip::Key && taps.write().press(slot, crate::platform::now_seconds());
    // Each read is its own statement, so no guard is alive when `wear` dispatches and
    // rewrites the brush signal underneath it (`state::update_brush`).
    let base = presets::worn(state);
    let base_from = state.preset_in_hand.peek().clone();
    let bound = state.slots.brushes.peek()[slot.as_index()].clone();
    let mut hold = Held::open(slot, grip, base, base_from, picked);
    // The slot's brush as it is *now* — its preset looked up live, at the slot's own
    // size and flow. A binding the library cannot answer is an empty slot, and an
    // empty slot is held without a swap.
    if let Some(bound) = bound
        && let Some((brush, tune)) = resolve_in(state, &bound)
    {
        presets::wear(state, brush, tune, Some(bound.preset));
        // Read back rather than assumed: `wear` resolves the stamp and clamps what the
        // renderer cannot draw, so what the app now holds is not necessarily what was
        // handed to it — and it is what the release has to compare against.
        hold.enter(presets::worn(state).1);
    }
    held.set(Some(hold));
}

/// Say that a whole tool was just put on **deliberately** — a preset row clicked
/// ([`presets::apply`]) or another slot's row ([`pick`]) — so a hold in flight keeps
/// what is live when it ends whether or not that moved anything ([`Held::claim`]).
///
/// A no-op with no hold in flight, which is every other click on those same rows.
pub fn claim(state: AppState) {
    let mut held = state.slots.held;
    let Some(mut h) = held.peek().clone() else {
        return;
    };
    if h.claim() {
        held.set(Some(h));
    }
}

/// End the hold on `slot`, if `grip` is what is holding it ([`Held::ends_on`]): keep
/// whatever was changed, and put the displaced brush back.
pub fn release(state: AppState, slot: Digit, grip: Grip) {
    let mut held = state.slots.held;
    // The guard is answered before the hold is cloned: this runs on every report of a
    // pen whose tail is not facing the glass (`input::tail_says`), and the overwhelming
    // majority of them are holding nothing.
    let h = {
        let in_flight = held.peek();
        match in_flight.as_ref() {
            Some(h) if h.ends_on(slot, grip) => h.clone(),
            _ => return,
        }
    };
    held.set(None);
    let current = presets::worn(state).1;
    // The tool the number would be bound to: whatever preset is in hand at the release
    // — the one a click under the hold put on, or the slot's own.
    let from = state.preset_in_hand.peek().clone();
    let (kept, back) = h.settle(current, from.as_deref());
    if let Some(bound) = kept {
        assign(state, h.slot(), bound);
    }
    // Back through the door it left by, with the name it had: the hold borrowed the
    // hand, and a preset chosen *during* it went to the slot, not to this. Or not back
    // at all, for a double-tap's hold, whose whole point is that the swap stands.
    if let Some((back, back_tune)) = back {
        presets::wear(state, back, back_tune, h.base_from());
    }
}

/// End whatever hold is in flight, whoever made it — for the one event that can take a
/// key away without ever sending its keyup: the window losing focus.
///
/// Alt+Tab with a number held would otherwise leave the swap in force for the rest of
/// the session, with the key that would undo it now belonging to another window. The
/// same class of bug the modifier tracker (`input`'s `track_mods`) rules out by
/// re-reading the modifier set, ruled out here by the event that says the keyboard has
/// gone.
pub fn release_all(state: AppState) {
    let held = state.slots.held.peek().clone();
    if let Some(h) = held {
        release(state, h.slot(), h.grip());
    }
}

/// The rack, drawn while a number is held (§18.1.8): a column down the left of the
/// window of the brushes the digits carry, each shown as the rendered test stroke the
/// preset library shows it by (`crate::thumbs`).
///
/// It replaced a permanent row of ten chips at the head of the Brush panel, and the
/// trade is the point. The chips spent the scarcest space in the app — panel height —
/// every second of every session to say a digit and three lit states, and the digit is
/// the one thing about a quick brush nobody needs told: the question a rack of ten
/// unlabelled numbers actually raises is *what is on 4*. Being momentary is what pays
/// for the answer — on screen only while a finger is on the key that summons it, this
/// can afford the width of a real preview and costs the panel nothing.
///
/// Up while a number is held, and **kept** up by the visibility menu's "Quick brushes"
/// ([`Signals::slots`](crate::state::Signals::slots)`.pinned`) — which is the same
/// picture with one thing added: pinned, the rows take clicks, and clicking one is
/// [`pick`]. That is the mouse-only way to a slot, and it is why the pin exists at all
/// — a pen in one hand and a tablet under it leaves no spare finger for the number
/// row, and the transient rack cannot be clicked (see below).
///
/// A filled row pinned also carries the **trash** every other roster in the app
/// carries ([`clear`]), hover-revealed like the preset rows'. It is the one operation
/// on a slot that the one rule cannot express: a hold assigns, and no length of
/// holding can mean *nothing*. It goes here rather than anywhere else for the reason
/// the pin exists — this is the only place the rack is a list of controls rather than
/// an answer to "what is on 4".
///
/// **It is held, not clicked** — the one trash in the app that is. Everywhere else a
/// trash stands beside the thing it removes; here it rides *on* a control whose own tap
/// is the commonest act on the rack, at the end of the row nearest the painting, which
/// is the end a pen coming off the canvas reaches first — and what it takes is a
/// binding the action log never held, so there is no undo behind it. A tap that landed
/// on it emptied the slot the hand was reaching for. So a tap on the trash is a tap on
/// the row ([`pick`]), and clearing takes the trash held down: the disc fills red
/// around the icon for as long as it is pressed, and the slot goes when the fill
/// closes. The fill is the clock — `onanimationend` is what clears, so what the disc
/// shows and what happens are one thing rather than a stylesheet duration and a timer
/// agreeing — and a tap shows the first sliver of it, which is how a hand that expected
/// a click finds out there is a hold to be made. Releasing, or sliding off, before the
/// fill closes is a tap.
///
/// Which is why the rows arm on their own press, as a dialog's backdrop does
/// (`widgets::Modal`, §25.7): the hold removes the row while the pen is still down, and
/// a pen's release is hit-tested afresh, so the click it makes lands on whichever row
/// has moved up under it — and would pick that.
///
/// Two things it deliberately does not do:
///
/// - **It never takes the pointer while it is transient** (`pointer-events` in the
///   stylesheet, granted to the rows only under `.pinned`). The gesture it belongs to
///   then is hold-*and-draw*, and the hand is very often painting under it; a row that
///   swallowed the stroke being drawn beneath it would be a worse control than no
///   control. Pinned is the user asking for that space to be a control, and paying for
///   it in canvas, exactly as a panel is. The *container* declines the pointer either
///   way, so the gaps and the empty column below the last row are never anything but
///   painting.
/// - **It shows no digit that has nothing in it**, pinned or not, with one exception:
///   the digit being held (`slots::rows`).
///
/// It **fades with the rest of the floating chrome** (`layout::chrome_dimmed`), which
/// goes to nothing while a canvas gesture is in flight: it stands over the painting
/// like the panels and the bars, and while a stroke is being laid the screen goes back
/// to being the painting. What makes that safe on something this momentary is that
/// **the hold outlives the stroke** — the key is still down when the pen lifts, so the
/// rack returns and the answer is there whenever the hand wants it.
///
/// The hold that summons it is a **key** hold alone ([`Held::by_key`]), which is the
/// one place the two grips are told apart rather than being the same hold. The pen's
/// tail holds [`ERASER`] for as long as it faces the glass (`input::tail_says`) — that
/// is every erase stroke and every hover between them — and a rack standing in the
/// corner of the eye for as long as the pen is inverted is noise answering a question
/// nobody asked. Holding `0` shows the same row.
#[component]
pub fn SlotOverlay() -> Element {
    let state = use_context::<AppState>();
    let pinned = (state.slots.pinned)();
    // The hold itself, not only its digit: the held row is drawn from the rule it is
    // under (`Held::would_keep`) rather than from what is stored.
    let holding = (state.slots.held)().filter(Held::by_key);
    // Neither held nor pinned: read nothing else at all, so a rack, a library or a
    // brush that changes during a stroke cannot re-render anything.
    //
    // **Which is why the rack itself is a component and not the rest of this
    // function.** What it draws is read through a memo now (`SlotRack`), and a memo
    // declared here would have to be declared above this line — hooks are positional —
    // leaving it recomputing against every engine write for the whole time the rack is
    // away, which is nearly all of the time. Mounted behind the gate instead, it does
    // not exist until there is something to show, exactly as the timeline bar is
    // mounted (`main`).
    if !pinned && holding.is_none() {
        return rsx! {};
    }
    rsx! {
        SlotRack { pinned, holding }
    }
}

/// The rack proper, mounted only while there is one to draw ([`SlotOverlay`]).
///
/// `holding` rather than a bare slot number: the held row is drawn from the rule the
/// release is under, which needs the whole hold ([`Held::would_keep`]).
#[component]
fn SlotRack(pinned: bool, holding: Option<Held>) -> Element {
    let state = use_context::<AppState>();
    // Which row heard the press in flight, so a click applies a row only when it closes
    // a press that landed there (see [`SlotOverlay`] on the pen's release).
    let mut pressed: Signal<Option<Digit>> = use_signal(|| None);
    // The trash being held down right now, if any: its row wears the fill for as long
    // as this says so, and nothing else keeps time.
    let mut arming: Signal<Option<Digit>> = use_signal(|| None);
    let arming_now = arming();
    let rack = (state.slots.brushes)();
    // The whole tool, feel and inactive effect included, off the frontend's own signals
    // — so the lit row tracks a smoothing drag exactly as it tracks a radius one, and
    // wakes for brush edits and nothing else.
    let live = (state.brush)();
    let live_tune = (state.transient)();
    // The preset in hand is the tool a held digit is about to be bound to
    // (`Held::would_keep`); it moves only when a whole tool arrives.
    let in_hand = (state.preset_in_hand)();
    // The rows are resolved against the library up front — a slot is a name, and the
    // library is where the name becomes a brush (`slots::resolve`) — so no read guard
    // is alive while the rows below read the thumbnail cache one by one. A subscribing
    // read, since an overwritten preset is a changed row.
    let rows: Vec<Row> = {
        let library = state.presets.read();
        slots::rows(View {
            rack: &rack,
            library: &library,
            holding: holding.as_ref(),
            live: (live, live_tune),
            in_hand: in_hand.as_deref(),
        })
    };

    rsx! {
        div {
            class: "slot-overlay chrome",
            class: if chrome_dimmed(state) { "dimmed" },
            class: if pinned { "pinned" },
            for row in rows {
                {
                    let slot = row.slot;
                    // The brush as a stroke, filling the row as its background — the
                    // preset row's own recipe (`panels::brush`), down to writing `none`
                    // out rather than omitting the property: these nodes are reused as
                    // the rack changes under them, and a stranded declaration would
                    // leave one slot showing another's brush (inline style merges per
                    // property).
                    //
                    // The row it is about to hold where that has been rendered, and the
                    // one it still holds where it has not: a brush *tuned* under a hold
                    // is new with every pixel of the drag and has no thumbnail until the
                    // release stores it, and rendering one per frame of a Size drag is
                    // GPU spent on a picture nobody asked to keep. So the row shows the
                    // last true picture of the slot rather than blinking empty for the
                    // length of a drag — while a preset clicked mid-hold lands
                    // instantly, its thumbnail being the one the library is already
                    // showing.
                    let thumb = |b: Option<(BrushConfig, Transient)>| {
                        b.and_then(|(b, t)| crate::thumbs::url(state, &b, t))
                            .filter(|url| !url.is_empty())
                    };
                    let bg = match thumb(row.brush).or_else(|| thumb(row.stored)) {
                        Some(url) => format!("background-image: url({url});"),
                        None => "background-image: none;".to_string(),
                    };
                    // Says the binding a picture cannot, and only ever seen pinned — an
                    // element that takes no pointer is shown no tooltip.
                    let (label, title, lit, held) = (row.label(), row.tip(), row.lit, row.held);
                    let (empty, has_stored) = (row.name.is_none(), row.stored.is_some());
                    rsx! {
                        div {
                            key: "{slot}",
                            class: "slot-row",
                            // Lit where this is the brush in hand, color aside — the size
                            // and flow counted, since they are what a slot keeps of its
                            // own. Held wins in the stylesheet: it is what the user is
                            // doing right now rather than a state they are in.
                            class: if empty { "empty" } else if lit { "active" },
                            class: if held { "held" },
                            style: "{bg}",
                            title,
                            // Reachable only while pinned, the stylesheet having given
                            // the rows the pointer there and nowhere else — the same way
                            // faded chrome stops taking clicks (`.chrome.dimmed`) rather
                            // than by every handler asking whether it is visible.
                            onpointerdown: move |_| pressed.set(Some(slot)),
                            // Terminal, as every disarm is: a press that was cancelled,
                            // or left the row, is not this row's click whatever lands
                            // here next.
                            onpointercancel: move |_| pressed.set(None),
                            onpointerleave: move |_| pressed.set(None),
                            onclick: move |_| {
                                // Bound, not read in the `if`: the read would be held
                                // through a body that writes the same signal.
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
                            if has_stored {
                                // The stored binding, not the row's: the trash takes
                                // something *out of the rack*, and a held row previewing
                                // a binding the release has not written yet has nothing
                                // there for it to take.
                                //
                                // The trash every other roster in the app wears
                                // (`stark_ui::icons::REMOVE`) — presets, layers, guides,
                                // shapes, gradients — because emptying a row of this one
                                // is the same act, revealed on the same hover and
                                // answering it in the same red ink.
                                //
                                // Only where there is something to remove: an empty row
                                // is on screen because it is *held*, and a trash on it
                                // would offer to undo nothing. Only reachable pinned,
                                // for the row's own reason.
                                //
                                // Held rather than clicked, alone among them: see
                                // [`SlotOverlay`]. The press bubbles on to the row, so a
                                // tap here is the row's own tap and needs no click of its
                                // own. Nothing here keeps time — the fill's own end is
                                // what clears, so a hold that outlived its row could not
                                // fire, there being no element left for an animation to
                                // end on.
                                button {
                                    class: "slot-clear",
                                    class: if arming_now == Some(slot) { "arming" },
                                    title: "Hold to clear this slot",
                                    onpointerdown: move |e| {
                                        if e.trigger_button() == Some(MouseButton::Primary) {
                                            arming.set(Some(slot));
                                        }
                                    },
                                    // The release ends it, or the pointer leaving — which
                                    // is what sliding off a button has always meant.
                                    // Taking the class off cancels the animation, and a
                                    // cancelled animation never ends: a tap cannot clear.
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
                                        // The press is spent: the release that follows is
                                        // nobody's click.
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

/// Make `slot`'s brush the live one for good — what clicking a row of the pinned rack
/// does, and the only way to a slot for a hand with no keyboard under it (a pen in one
/// hand and a tablet under the other leaves no spare finger for the number row).
///
/// Also where **tapping the number twice** arrives, by way of the hold itself: the
/// second press enters a hold whose release keeps the slot's brush rather than putting
/// the displaced one back, so this is not called for it and there is no second path to
/// the same place. Deliberately *not* what a single tap does. A tap and a hold are the
/// same keystroke told apart only by how long it lasted, so binding them to different
/// outcomes would make every hold a race against the user's own reflexes; two presses
/// are told apart by *count*, and the first is an ordinary hold that does nothing
/// unused. A click is its own gesture and says what it means.
///
/// Clicking a row *during* a hold is left to mean what the one rule already says it
/// means: the click makes that slot's brush live, and the hold then keeps whatever is
/// live when it ends — so holding 3 and clicking 5 copies 5 onto 3, binding and tune
/// alike, exactly as holding 3 and clicking a preset assigns that preset. One rule, not
/// a special case (`Held::settle`), and it is a whole tool arriving, so it says so
/// ([`claim`]) — copying 5 onto 3 must work when 5 is already what is in hand.
pub fn pick(state: AppState, slot: Digit) {
    let bound = state.slots.brushes.peek()[slot.as_index()].clone();
    let Some(bound) = bound else { return };
    // A binding the library cannot answer is an empty row, and an empty row's click
    // puts on nothing.
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
/// Pinning is not the same question as the rack being *up*: while a number is held the
/// rack shows whatever this says, and what pinning buys is a rack that stays and takes
/// clicks — the only route to a slot for a hand with no keyboard under it (§18.1.8).
/// That standing choice is the one worth keeping, and it is kept with the rest of what
/// is on screen (`crate::visibility`, §25.6).
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
pub fn assign(state: AppState, slot: Digit, brush: QuickBrush) {
    let mut brushes = state.slots.brushes;
    // Its own statement, this module's rule: a write guard held across the call would
    // still be alive under the `read` beside it.
    slots::assign(&mut brushes.write(), slot, brush);
    slots::persist(&brushes.read());
}

/// Empty `slot` and persist the rack — the trash on a pinned row, held down until its
/// fill closes ([`SlotOverlay`]).
pub fn clear(state: AppState, slot: Digit) {
    let mut brushes = state.slots.brushes;
    // Its own statement, this module's rule: a write guard held across the `if` would
    // still be alive under the `read` in its body.
    let emptied = slots::clear(&mut brushes.write(), slot);
    if emptied {
        slots::persist(&brushes.read());
    }
}

/// Empty every slot bound to the preset called `name` — what removing a preset does to
/// the rack (`presets::remove`).
pub fn unbind(state: AppState, name: &str) {
    let mut brushes = state.slots.brushes;
    let any = slots::unbind(&mut brushes.write(), name);
    if any {
        slots::persist(&brushes.read());
    }
}

/// Populate the rack from storage. Called once at app start, before the renderer exists
/// — nothing here needs an engine, or even the library: a slot is a name until it is
/// resolved, and it is resolved at every use (`slots::resolve`).
pub fn load(state: AppState) {
    let mut brushes = state.slots.brushes;
    if let Some(rack) = slots::read_storage() {
        brushes.set(rack);
    }
}

/// Fill a rack that has never been set from the library (`slots::seed`).
///
/// Called after `presets::install_builtins`, and reading *from* the library rather than
/// restating it — so a tool reaches the keyboard under the name the panel lists it by.
///
/// **Seeded in memory and not persisted.** Storage is written only by the user's own
/// act ([`assign`]), which is what `read_storage().is_some()` then means: not "this
/// browser has run Stark before" but "this browser has set a slot". That keeps the seed
/// live — a shipped preset moved to another digit reaches the rack on the next start,
/// exactly as its edits reach the preset list — and it costs nothing that a binding does
/// not already buy, since the *tool* on a seeded digit is looked up live either way.
pub fn seed_defaults(state: AppState) {
    if slots::read_storage().is_some() {
        return;
    }
    let rack = slots::seed(&state.presets.peek());
    let mut brushes = state.slots.brushes;
    brushes.set(rack);
}
