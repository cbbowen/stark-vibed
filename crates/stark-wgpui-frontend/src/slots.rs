//! The quick-brush rack, as this window draws it (§18.1.8, §11.2).
//!
//! **Nothing about what a slot means is decided here.** What a digit holds, what a
//! press and a release each keep, how two presses become a pick and which rows there
//! are to draw are all `stark_ui::slots`, shared with the web frontend — the same
//! split `crate::pick` makes, for its reason. What is this module's is where the rack
//! stands, how it is measured, and the one clock the web app keeps in a stylesheet.
//!
//! # What is different here, and why
//!
//! **A row wears a name, not a picture.** The web rack shows each brush as the
//! rendered test stroke the preset library shows it by; this frontend's preset list is
//! words too (`crate::panel`), and ten thumbnails would be ten more test canvases on
//! the GPU for chrome that is on screen while a finger is on a key. The digit and the
//! preset's name are the whole of what a rack of ten unlabelled numbers is asked, so
//! that is what a row says.
//!
//! **The tail is the web's alone so far.** `stark-pen` reports contact and no hover
//! (§11.3), and §18.1.8's eraser end holds [`ERASER`] from the moment it is *in range*
//! — a hold armed on contact would preview paint under an inverted pen and then erase,
//! which is the one place that rule must not be approximated. So the pen's other end
//! goes on doing what it did here (`Canvas::open_canvas` swaps the effect for the
//! length of the stroke) and the rack's grip is `Grip::Key` alone; the vocabulary
//! for the other is already shared, and what it waits on is a hovering report.

use stark_ui::slots::{self, Digit, ERASER, Row};
use wgpui::{
    AnyElement, Bounds, IntoElement, Pixels, Point, SharedString, canvas, div, prelude::*, px,
    relative, rgb,
};

use crate::style;

/// How long the trash is held before the slot goes, seconds.
///
/// **The clock and the picture are one value**, which is the web app's bargain reached
/// from the other side: over there the fill *is* the clock and the duration is stated
/// once in the stylesheet, and here there is no stylesheet — so the frame reads this to
/// decide both how much red to draw and whether the hold has closed, and the two cannot
/// come apart. Not shortened for anything: it is not decoration, it is the hold.
pub const CLEAR_HOLD: f64 = 0.6;

/// A row's width, logical px — and what the held one grows to.
///
/// It **grows** rather than sliding out, the rack being anchored to this edge, so the
/// digits stay in one column and only the picture moves. The web slides its rows out
/// instead, which is the same answer for a strip whose rows carry a full-bleed image.
const WIDTH: f32 = 150.0;
const HELD_WIDTH: f32 = WIDTH + 26.0;

/// What the rest of the rack says its name in while a number is held.
///
/// The rack is asked one thing at a time — which digit is under the finger — so one row
/// answers at full strength and the others sink back. Deliberately not a badge on the
/// held row: a badge has to be found among nine rows shouting as loudly as it does,
/// while the one row still standing is found without being looked for.
///
/// **The ghost is in the ink, not in the ground**, which is where the web app puts it —
/// and the difference is what the rows are made of. Over there a row *is* a rendered
/// stroke, so fading the whole row leaves an image that still reads; here a row is a
/// word, and a word faded into the painting under it is a word nobody can read. The
/// grounds stay opaque for the same reason a panel's does.
const GHOST_INK: u32 = 0x7f8489;

/// The red a destructive control answers in, matching the web's `--danger`.
const DANGER: u32 = 0xe8757a;

/// Which of the rack's controls a press landed on.
///
/// Recorded **only while the rack is pinned**: transient, the gesture it belongs to is
/// hold-*and*-draw and the hand is very often painting directly under it, so a row that
/// swallowed the stroke would be a worse control than no control. Pinning is the user
/// asking for that strip to be a control and paying for it in canvas, exactly as
/// opening a panel is (§18.1.8).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Region {
    /// A row, by the digit it is reached by. Its press picks the slot.
    Row(Digit),
    /// The trash on a filled row, by the same digit.
    Trash(Digit),
}

/// Where the rack's controls were laid out — `crate::panel`'s device, for its reason.
pub type Regions = std::rc::Rc<std::cell::RefCell<Vec<(Region, Bounds<Pixels>)>>>;

fn probe(regions: &Regions, region: Region) -> impl IntoElement {
    let regions = regions.clone();
    canvas(
        move |bounds, _, _| regions.borrow_mut().push((region, bounds)),
        |_, (), _, _| {},
    )
    .absolute()
    .top_0()
    .left_0()
    .right_0()
    .bottom_0()
}

/// Which control a press landed on. Innermost first (`.rev()`), which is
/// `crate::pick`'s rule and its reason: a row is probed as its own first child and
/// contains the trash on it, so reading forwards would hand back the row under a trash
/// that was pressed.
pub fn hit(regions: &Regions, at: Point<Pixels>) -> Option<Region> {
    regions
        .borrow()
        .iter()
        .rev()
        .find(|(_, bounds)| bounds.contains(&at))
        .map(|(region, _)| *region)
}

/// This window's rack: what the digits hold, what is holding one down, and the two
/// pointer facts a pinned rack needs.
///
/// The first three are `stark_ui::slots`' own values, kept in fields where the web
/// frontend keeps them in signals — which is the whole of what a frontend owes the
/// shared rule.
pub struct Rack {
    /// What each digit holds: a preset's name and a size and flow, never a brush.
    pub brushes: slots::Rack,
    /// The hold in flight — `Some` for exactly as long as a number key is down.
    pub held: Option<slots::Held>,
    /// The last press of a number key, so the next can tell whether it is the second of
    /// a double-tap.
    pub taps: slots::Taps,
    /// Whether the rack is kept up with no key held — the Window menu's "Quick
    /// brushes". What it buys is a rack that can be *clicked*, which is the only way to
    /// a slot for a hand with no keyboard under it.
    pub pinned: bool,
    /// Which row heard the press in flight, so a click applies a row only when it
    /// closes a press that landed there — a dialog's rule (§25.7), for the same device:
    /// the trash's hold removes the row while the pen is still down, and a release is
    /// hit-tested afresh.
    pressed: Option<Digit>,
    /// The trash being held down: which row, and when the press landed. The row wears
    /// the fill for as long as this says so, and **nothing else keeps time**.
    arming: Option<(Digit, f64)>,
}

impl Default for Rack {
    fn default() -> Self {
        Self {
            brushes: slots::empty_rack(),
            held: None,
            taps: slots::Taps::default(),
            pinned: false,
            pressed: None,
            arming: None,
        }
    }
}

impl Rack {
    /// Read this client's rack at start: what it stored, or the library's own digits
    /// where it has never set one (`slots::seed`).
    ///
    /// **Seeded in memory and not persisted**, which is what "has never set one" then
    /// means: not "this machine has run Stark before" but "this machine has set a slot".
    /// That keeps the seed live — a shipped preset moved to another digit reaches the
    /// rack on the next start — and a rack emptied to the last digit stays empty rather
    /// than re-seeding itself (`slots::read_storage`).
    pub fn stored(library: &[stark_ui::presets::PresetEntry]) -> Self {
        Self {
            brushes: slots::read_storage().unwrap_or_else(|| slots::seed(library)),
            pinned: stark_ui::visibility::stored_showing(
                stark_ui::commands::VisibilityToggle::QuickBrushes,
            ),
            ..Self::default()
        }
    }

    /// Whether there is a rack on screen at all: one is held, or it is pinned up.
    pub fn up(&self) -> bool {
        self.pinned || self.held.as_ref().is_some_and(slots::Held::by_key)
    }

    /// Note a press on `region` at `now`, and say whether the frame changed.
    ///
    /// A press on the trash is **also** the row's own press, exactly as it is in the
    /// web app where it bubbles: what the trash adds is a clock, and a tap on it that
    /// let go before the clock closed is the row's tap.
    pub fn press(&mut self, region: Region, now: f64) -> bool {
        match region {
            Region::Row(slot) => self.pressed = Some(slot),
            Region::Trash(slot) => {
                self.pressed = Some(slot);
                self.arming = Some((slot, now));
            }
        }
        true
    }

    /// A release over `region` (or over nothing). `Some(slot)` where it closes a press
    /// this same row heard, which is the only click that picks a slot.
    ///
    /// Disarms whatever was armed either way: letting go is what a trash held down
    /// stops being.
    pub fn release(&mut self, region: Option<Region>) -> Option<Digit> {
        self.arming = None;
        let pressed = self.pressed.take()?;
        match region {
            Some(Region::Row(slot) | Region::Trash(slot)) if slot == pressed => Some(pressed),
            _ => None,
        }
    }

    /// The pointer moved to `region`. Sliding off the trash backs out of the hold —
    /// what sliding off a button has always meant.
    pub fn moved(&mut self, region: Option<Region>) -> bool {
        let Some((slot, _)) = self.arming else {
            return false;
        };
        if region == Some(Region::Trash(slot)) {
            return false;
        }
        self.arming = None;
        true
    }

    /// The slot the trash has finished emptying, if the hold that was in flight has
    /// closed by `now` — the frame's own question, asked once per frame.
    ///
    /// Spent by being answered: the press is over, and the release that follows is
    /// nobody's click, so the row moving up under a still-down pen cannot be picked.
    pub fn armed_out(&mut self, now: f64) -> Option<Digit> {
        let (slot, at) = self.arming?;
        if now - at < CLEAR_HOLD {
            return None;
        }
        self.arming = None;
        self.pressed = None;
        Some(slot)
    }

    /// Whether a trash is being held down right now — the frame's cue that it owes
    /// another, since the fill is drawn from a clock and nothing else here moves.
    pub fn clearing(&self) -> bool {
        self.arming.is_some()
    }

    /// How far the fill on `slot`'s trash has closed, `0..=1` — nothing for a row that
    /// is not being held down.
    fn arming_at(&self, slot: Digit, now: f64) -> Option<f32> {
        let (armed, at) = self.arming?;
        (armed == slot).then(|| (((now - at) / CLEAR_HOLD) as f32).clamp(0.0, 1.0))
    }
}

/// The rack: a column down the left of the canvas of the brushes the digits carry.
///
/// Vertically centred rather than hung from the top, which is where the web app puts
/// its own and what keeps it clear of the three bars that take that edge
/// (`crate::pick`, `crate::select`, `crate::transform`). The container takes no pointer
/// at all — it has no id and no handler — so the gaps between rows and the column above
/// and below them are never anything but painting.
///
/// **Two marks, two questions.** The held row is the only one at full strength, opaque
/// and reaching further out from the edge where the rest of the rack is a ghost; that
/// is the answer for an eye on the *painting*, which is where the eye is during
/// hold-and-draw. The second mark is the row whose brush the live brush still *is*
/// (color aside), lit in the panel's own blue — on the stricter of the library's two
/// tests, the size and flow counted (`presets::same_brush`), since a slot *is* a size
/// and a flow. Held wins where both apply: it is what the user is doing, not a state
/// they are in.
pub fn rack(rack: &Rack, rows: &[Row], now: f64, regions: &Regions) -> AnyElement {
    let pinned = rack.pinned;
    div()
        .absolute()
        .top_0()
        .bottom_0()
        .left_0()
        .flex()
        .flex_col()
        .justify_center()
        .items_start()
        .gap_1()
        .p_2()
        .children(
            rows.iter()
                .map(|row| self::row(rack, row, pinned, now, regions)),
        )
        .into_any_element()
}

/// One row: the key to press, what it holds, and — pinned and filled — the trash.
fn row(state: &Rack, row: &Row, pinned: bool, now: f64, regions: &Regions) -> AnyElement {
    let slot = row.slot;
    // Held wins over lit, and both over the ghost the rest of the rack rests at while
    // something is held. Pinned and idle nothing is held, so the whole rack stands at
    // full strength and the lit row is the only thing saying which of the ten is in
    // hand — which is what a permanent row of chips used to be for.
    let anything_held = state.held.as_ref().is_some_and(slots::Held::by_key);
    let (ground, ink) = match (row.held, row.lit, anything_held) {
        (true, _, _) => (style::CONTROL, style::INK_LIT),
        (_, true, _) => (style::LIT, style::INK_LIT),
        (_, _, true) => (style::PANEL, GHOST_INK),
        _ => (style::PANEL, style::INK),
    };
    let width = if row.held { HELD_WIDTH } else { WIDTH };
    let el = div()
        .id(SharedString::from(format!("slot-{slot}")))
        .relative()
        .flex()
        .items_center()
        .gap_2()
        .w(px(width))
        .px_2()
        .py_1p5()
        .rounded_sm()
        .text_xs()
        .bg(rgb(ground))
        .border_1()
        // The held row is keylined, as the chips it replaced were; an empty one is an
        // outline where the picture would be, since holding an empty number is not a
        // mistake but how the number gets its first brush.
        .border_color(rgb(if row.held {
            style::INK_LIT
        } else {
            style::EDGE
        }))
        .text_color(rgb(if row.name.is_some() {
            ink
        } else {
            style::INK_MARK
        }))
        // Reachable only while pinned: the regions are what a press is answered
        // against, and a transient rack records none.
        .when(pinned, |el| {
            el.cursor_pointer()
                .hover(|s| s.bg(rgb(style::HOVER)))
                .child(probe(regions, Region::Row(slot)))
        })
        // The key to press, on a ground sunk below the row's: a digit is an index
        // rather than a word, and it reads as one only if it sits in its own field.
        .child(
            div()
                .flex()
                .flex_none()
                .items_center()
                .justify_center()
                .size(px(18.0))
                .rounded_sm()
                .bg(rgb(style::WELL))
                .when(slot == ERASER, |el| {
                    el.child(crate::icons::icon(stark_ui::icons::ERASER, style::INK_LIT))
                })
                .when(slot != ERASER, |el| {
                    el.text_color(rgb(style::INK_LIT)).child(slot.to_string())
                }),
        )
        // The name is the binding's — the preset the digit holds, whatever size it
        // holds it at. A name the artist chose, so it stays a word (`crate::panel`).
        .child(div().flex_1().truncate().child(row.label()))
        // The one operation the one rule cannot express: a hold *assigns*, and no
        // length of holding can mean *nothing*. So it is a control rather than a
        // gesture, and it belongs in the one state where the rack is a list of controls
        // rather than an answer to "what is on 4".
        //
        // The **stored** binding, not the row's: the trash takes something out of the
        // rack, and a held row previewing a binding the release has not written yet has
        // nothing there for it to take.
        .children((pinned && row.stored.is_some()).then(|| {
            trash(
                state.arming_at(slot, now),
                probe(regions, Region::Trash(slot)),
            )
        }));
    style::tip(el, row.tip()).into_any_element()
}

/// The trash, and how far holding it has got.
///
/// **Held down, not clicked** — the only one in the app that is. Every other trash
/// stands beside what it removes; this one rides on a control whose own tap is the
/// commonest act on the rack, at the end of the row nearest the painting, and what it
/// takes is a binding the action log never held, so no undo stands behind it. A tap
/// that landed on it would empty the slot the hand was reaching for.
///
/// So the button says what holding it would do by doing it slowly: the red fills across
/// it for as long as it is pressed, and the slot goes when the fill closes
/// ([`CLEAR_HOLD`]). A tap shows the first sliver of it, which is how a hand that
/// expected a click learns there is a hold to make.
fn trash(arming: Option<f32>, probe: impl IntoElement) -> impl IntoElement {
    let filled = arming.unwrap_or(0.0);
    div()
        .relative()
        .flex()
        .flex_none()
        .items_center()
        .justify_center()
        .size(px(18.0))
        .rounded_sm()
        .overflow_hidden()
        .bg(rgb(style::WELL))
        // The fill, under the mark and over the ground. Zero-width until the press,
        // which is what makes one element cover both states.
        .child(
            div()
                .absolute()
                .top_0()
                .bottom_0()
                .left_0()
                .w(relative(filled))
                .bg(rgb(DANGER)),
        )
        .child(crate::icons::icon(
            stark_ui::icons::REMOVE,
            if arming.is_some() {
                style::INK_LIT
            } else {
                style::INK_MARK
            },
        ))
        .child(probe)
}

#[cfg(test)]
mod tests {
    use super::*;
    use stark_ui::brush_config::Transient;
    use stark_ui::slots::QuickBrush;

    /// The digit `n`, for a test that knows it named one.
    fn d(n: usize) -> Digit {
        Digit::new(n).expect("a digit the rack has")
    }

    fn filled() -> Rack {
        let mut rack = Rack::default();
        slots::assign(
            &mut rack.brushes,
            d(3),
            QuickBrush {
                preset: "Pen".into(),
                transient: Transient::default(),
            },
        );
        rack.pinned = true;
        rack
    }

    /// A click picks a slot only when it closes a press the same row heard — the rule a
    /// trash that removes its own row underneath a still-down pen makes necessary.
    #[test]
    fn a_row_is_picked_only_by_a_click_it_heard_the_press_of() {
        let mut rack = filled();
        rack.press(Region::Row(d(3)), 0.0);
        assert_eq!(rack.release(Some(Region::Row(d(3)))), Some(d(3)));
        assert_eq!(rack.release(Some(Region::Row(d(3)))), None, "a spent press");
        rack.press(Region::Row(d(3)), 0.0);
        assert_eq!(
            rack.release(Some(Region::Row(d(7)))),
            None,
            "a release over another row is nobody's click"
        );
        rack.press(Region::Row(d(3)), 0.0);
        assert_eq!(rack.release(None), None, "nor is one over the painting");
    }

    /// A tap on the trash is the row's own tap, and clearing takes it held down until
    /// the fill closes.
    #[test]
    fn the_trash_is_held_and_a_tap_on_it_picks_the_row() {
        let mut rack = filled();
        rack.press(Region::Trash(d(3)), 10.0);
        assert_eq!(rack.armed_out(10.0 + CLEAR_HOLD / 2.0), None);
        assert_eq!(
            rack.release(Some(Region::Trash(d(3)))),
            Some(d(3)),
            "letting go early is the row's tap"
        );
        assert_eq!(rack.armed_out(10.0 + CLEAR_HOLD * 2.0), None, "disarmed");

        rack.press(Region::Trash(d(3)), 20.0);
        assert_eq!(rack.arming_at(d(3), 20.0 + CLEAR_HOLD / 2.0), Some(0.5));
        assert_eq!(rack.arming_at(d(7), 20.0 + CLEAR_HOLD / 2.0), None);
        assert_eq!(rack.armed_out(20.0 + CLEAR_HOLD), Some(d(3)));
        assert_eq!(
            rack.release(Some(Region::Trash(d(3)))),
            None,
            "the press was spent by the hold closing"
        );
    }

    /// Sliding off the trash backs out of the hold, and the fill goes with it.
    #[test]
    fn sliding_off_the_trash_is_a_tap() {
        let mut rack = filled();
        rack.press(Region::Trash(d(3)), 30.0);
        assert!(rack.moved(Some(Region::Row(d(3)))));
        assert!(
            !rack.moved(Some(Region::Row(d(3)))),
            "nothing left to disarm"
        );
        assert_eq!(rack.armed_out(30.0 + CLEAR_HOLD * 2.0), None);
        assert_eq!(
            rack.release(Some(Region::Row(d(3)))),
            Some(d(3)),
            "and the row still heard the press"
        );
    }

    /// A rack is on screen while a **key** holds one or while it is pinned, and at no
    /// other time — the pen's tail holds a slot without summoning it (§18.1.8).
    #[test]
    fn a_rack_is_up_while_it_is_held_or_pinned() {
        let mut rack = Rack::default();
        assert!(!rack.up());
        rack.pinned = true;
        assert!(rack.up());
        rack.pinned = false;
        rack.held = Some(slots::Held::open(
            d(2),
            slots::Grip::Eraser,
            (Default::default(), Transient::default()),
            None,
            false,
        ));
        assert!(!rack.up(), "the tail's hold draws no rack");
        rack.held = Some(slots::Held::open(
            d(2),
            slots::Grip::Key,
            (Default::default(), Transient::default()),
            None,
            false,
        ));
        assert!(rack.up());
    }
}
