//! Moving a row of a list by dragging it — the gesture, with no opinion about what
//! the list is.
//!
//! Two panels are rosters of a stack the artist arranges: the layer tree and the
//! guide list. They already share their rows' shape down to the glyphs
//! (`panels::guides`), and this is the rest of that: the same press, the same lift,
//! the same slot opening under the hand, the same release. What differs is only what
//! a landing *means* — a layer's is a place in a tree (§14.8), a guide's is an index
//! — and that stays in the panels, which is the whole of the boundary here.
//!
//! Three pieces, in the order a gesture meets them:
//!
//! - [`Grab`] is the pointer bookkeeping: what was pressed, where the hand started
//!   and is now, and every row's box **as it stood at the press**. Nothing derived
//!   is kept, so a row that has slid out of the way cannot feed back into the
//!   decision that moved it — the same bargain the panel-stack drag makes
//!   ([`crate::layout::DragState`]).
//! - [`Slide`] is the column arithmetic: given the block of rows that travels and
//!   how far the hand has taken it, which rows yield and by how much.
//! - [`Motion`] is what one row does about it, written as the two CSS declarations
//!   that say so.
//!
//! A row that is *both* pressable and draggable has one problem this module also
//! owns: the browser sends a `click` after the release, and that click belongs to
//! the drag rather than to the row. See [`claimed`].

use dioxus::prelude::*;

/// How far the pointer must travel before a press on a row is a drag rather than a
/// click. Small, because a row's other gestures are a click and a double-click and
/// neither survives being mistaken for a drag; large enough that the hand's own
/// tremor on a pen does not move anything.
const GRAB_SLOP: f32 = 4.0;

/// How long a displaced row takes to reach its new place. Long enough to be followed
/// by eye, short enough that the list has settled by the time the hand arrives.
const SLIDE_MS: u32 = 180;

/// A press on a row, which may or may not have become a drag yet.
///
/// Boxes carry the identity string off the element they were measured from (the
/// `data-` attribute `platform::layer_boxes` and its neighbours read), so a panel
/// matches them to its rows **by identity**. Matched by position a row would be measured through its neighbour's
/// box in silence.
#[derive(Clone, PartialEq)]
pub struct Grab {
    key: String,
    boxes: Vec<(String, f32, f32)>,
    anchor: (f32, f32),
    pointer: (f32, f32),
    live: bool,
    over: bool,
}

impl Grab {
    /// Arm a press on the row wearing `key`, with `boxes` as the whole list stood at
    /// that moment and `at` the pointer in client px.
    ///
    /// Measured on the press rather than on the first move, so the list is described
    /// as it stood when the hand closed on it and no pointer travel is lost waiting
    /// for a measurement.
    pub fn begin(key: impl Into<String>, boxes: Vec<(String, f32, f32)>, at: (f32, f32)) -> Self {
        Self {
            key: key.into(),
            boxes,
            anchor: at,
            pointer: at,
            live: false,
            over: false,
        }
    }

    /// Follow the pointer, `held` being whether a button is still down.
    ///
    /// Once a press is a drag it stays one — bringing a row back near where it
    /// started would otherwise turn the gesture into a click again. What ends it is
    /// the press ending, and `held` is the third way to hear that, after the release
    /// and the cancel: **a move with nothing held is a hover**, and a hover is not the
    /// middle of a drag.
    ///
    /// That third way is not belt and braces. A release this panel never hears about
    /// — over the empty space past the end of the list, over anything in a row that is
    /// not the grip — would otherwise leave a grab that is still armed, and since a
    /// hover reaches the grip too, the row goes on following the pointer around a
    /// panel with no button down. The same applies once the drop *has* landed: the
    /// grab is kept a moment longer for the click behind it ([`claimed`]), and it must
    /// not be steerable while it waits. Both are ruled out here rather than by asking
    /// every caller to check, because a caller that forgets leaves a panel only a
    /// close and reopen can put right.
    pub fn track(&mut self, at: (f32, f32), held: bool) {
        if self.over {
            return;
        }
        if !held {
            self.spend();
            return;
        }
        self.pointer = at;
        let (dx, dy) = self.delta();
        self.live |= dx.abs().max(dy.abs()) > GRAB_SLOP;
    }

    /// Whether the pointer has said this is a drag. A press that has not travelled
    /// yet draws nothing and lands nothing.
    pub fn live(&self) -> bool {
        self.live
    }

    /// Whether the gesture is finished — landed, or ended by a press that is no longer
    /// down. Such a grab draws nothing and follows nothing; it is kept only until the
    /// click behind it arrives (see [`claimed`]), which is what a panel checks this
    /// for: there is no gesture here to feed.
    pub fn over(&self) -> bool {
        self.over
    }

    /// How far the hand has taken the row, in px.
    pub fn delta(&self) -> (f32, f32) {
        (
            self.pointer.0 - self.anchor.0,
            self.pointer.1 - self.anchor.1,
        )
    }

    /// Where the grabbed row sits among `keys` — the panel's rows in the order it
    /// draws them — and every row's `(top, height)`.
    ///
    /// `None` if any row was not measured, or if the grabbed row is no longer in the
    /// list. That is what a list changing under the pointer looks like, and the
    /// answer to it is to draw at rest and land nothing rather than to act on
    /// geometry describing a list that is gone.
    pub fn resolve(&self, keys: &[String]) -> Option<(usize, Vec<(f32, f32)>)> {
        let at = keys.iter().position(|k| *k == self.key)?;
        let boxes = keys
            .iter()
            .map(|k| {
                self.boxes
                    .iter()
                    .find(|(m, ..)| m == k)
                    .map(|&(_, top, h)| (top, h))
            })
            .collect::<Option<Vec<_>>>()?;
        Some((at, boxes))
    }

    /// The gesture is over and its drop has been made.
    ///
    /// **This is the disarm**, and it goes before the list is written: a row's shift
    /// is stated against the list as it stood at the press, so a frame carrying the
    /// new order while the transforms were still on would be the move applied twice.
    /// The grab is kept rather than dropped only so the click behind it can be
    /// recognized — see [`claimed`].
    ///
    /// **Terminal**: nothing brings a spent grab back to life. It is what [`track`]
    /// checks first, and the reason it is a state of the grab rather than the panel
    /// dropping it — a panel that has just committed a move must not be able to
    /// commit it again from the same press.
    ///
    /// [`track`]: Self::track
    pub fn spend(&mut self) {
        self.live = false;
        self.over = true;
    }
}

/// Whether the `click` the browser has just sent belongs to a drag that already
/// landed — and clears the grab either way, so the next press starts clean.
///
/// A row that both clicks and drags gets a click after *every* release, drag or not,
/// and on a list that has just reordered itself that click is aimed at a row which is
/// no longer the one that was pressed. Swallowing it is what keeps a move from also
/// being a selection of whatever took the moved row's place.
///
/// A press always overwrites the grab, so a finished one cannot outlive the gesture
/// that left it even if no click ever arrives — and while it waits it is inert, which
/// is [`Grab::spend`]'s job rather than this one's. It was not always: a finished grab
/// that a hover could re-arm turned "no click arrived" from a thing that costs nothing
/// into a row that follows the pointer until the panel is closed (`2026-08-09`).
pub fn claimed(grab: &mut Signal<Option<Grab>>) -> bool {
    let over = grab.peek().as_ref().is_some_and(Grab::over);
    if over {
        grab.set(None);
    }
    over
}

/// Where a block of rows would land, and what the rows it displaces do about it.
///
/// Everything here is derived from the boxes and the hand; nothing is stored between
/// frames.
#[derive(Clone, Copy, PartialEq)]
pub struct Slide {
    /// The rows that travel, as an inclusive range of display indices. One row for a
    /// flat list; a whole subtree where a row can carry others.
    pub block: (usize, usize),
    /// How far a displaced row slides: the block's whole extent, gap included, so
    /// the slot it opens is exactly the size of what is going into it.
    pub step: f32,
    /// Where the block lands, counted in the rows that stay put — `gap` of them end
    /// up above it. For a flat list that is also the index to insert at once the
    /// block has been taken out.
    pub gap: usize,
}

impl Slide {
    /// Resolve the hand's vertical travel against the rows' boxes, in display order.
    ///
    /// A row yields when the block's **leading** edge — its top going up, its bottom
    /// going down — crosses that row's centre, so the block can always be dragged
    /// clear to either end of the list.
    ///
    /// `None` if the block is not a range of `boxes`, which a caller that measured
    /// the list it is talking about cannot produce.
    pub fn resolve(boxes: &[(f32, f32)], block: (usize, usize), dy: f32) -> Option<Self> {
        let (start, end) = block;
        if start > end || end >= boxes.len() {
            return None;
        }
        let (block_top, block_bottom) = (boxes[start].0, boxes[end].0 + boxes[end].1);
        // The space between two rows, so a slide closes the slot exactly rather than
        // leaving a seam the width of a margin.
        let gap_px = if boxes.len() > 1 {
            (boxes[1].0 - boxes[0].0 - boxes[0].1).max(0.0)
        } else {
            0.0
        };
        let (top, bottom) = (block_top + dy, block_bottom + dy);
        let gap = (0..boxes.len())
            .filter(|i| *i < start || *i > end)
            .filter(|&k| {
                let center = boxes[k].0 + boxes[k].1 * 0.5;
                if k < start {
                    top >= center
                } else {
                    bottom > center
                }
            })
            .count();
        Some(Self {
            block,
            step: block_bottom - block_top + gap_px,
            gap,
        })
    }

    /// Whether the block would end up in the slot it came out of. A drag that goes
    /// nowhere must not be committed as a move.
    ///
    /// The block started with exactly `block.0` rows above it, since every row before
    /// it in the display is one that stays put.
    pub fn inert(&self) -> bool {
        self.gap == self.block.0
    }

    /// How to draw the row at display index `i`, given where the block itself is
    /// being drawn (which is the panel's to decide: it follows the hand, and may
    /// snap sideways as well).
    pub fn motion(&self, i: usize, block: (f32, f32)) -> Motion {
        let (start, end) = self.block;
        if (start..=end).contains(&i) {
            return Motion {
                shift: block,
                lifted: true,
                live: true,
            };
        }
        // Where this row sits among the ones that stay put, which is what `gap` is
        // counted in. Everything below the block closes up behind it.
        let k = if i < start { i } else { i - (end - start + 1) };
        let dy = if i > end && k < self.gap {
            -self.step
        } else if i < start && k >= self.gap {
            self.step
        } else {
            0.0
        };
        Motion {
            shift: (0.0, dy),
            lifted: false,
            live: true,
        }
    }
}

/// How one row is drawn while the list is in motion — resolved by the panel and
/// handed to the row, so the rows that do not move do not re-render as the pointer
/// travels.
///
/// The default is the resting state, which is what every row gets when nothing is
/// being dragged: no shift, no lift, no transition.
#[derive(Clone, Copy, Default, PartialEq)]
pub struct Motion {
    /// Where to draw this row relative to where it belongs, in px.
    pub shift: (f32, f32),
    /// In the travelling block: this row is one of the ones being carried by the
    /// pointer, so it tracks the hand 1:1 and slides for nobody.
    pub lifted: bool,
    /// A drag is in flight, so a row that is *not* lifted eases into its new place
    /// rather than jumping there.
    pub live: bool,
}

impl Motion {
    /// The row's own two declarations — **both of them, every render, including the
    /// ones that are "off"**.
    ///
    /// Inline styles are applied property by property rather than by replacing the
    /// attribute, so a declaration left out of a render is not cleared: it keeps
    /// whatever the last render that *did* mention it gave it. That is how a dropped
    /// row was once left wearing a preview's transform over a list that had since
    /// reordered (`layout::Panel` carries the same scar). Writing the pair from one
    /// place is one place for the rule to hold rather than one per panel.
    pub fn css(self) -> String {
        let (dx, dy) = self.shift;
        let ease = if self.live && !self.lifted {
            format!("transform {SLIDE_MS}ms ease")
        } else {
            "none".to_string()
        };
        format!("transform: translate({dx}px, {dy}px); transition: {ease};")
    }

    /// `" dragging"` for a row in flight, to be appended to its class list — the
    /// hook both panels' stylesheets hang the lift on.
    pub fn class(self) -> &'static str {
        if self.lifted { " dragging" } else { "" }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A row's height and the gap between two of them. The numbers do not matter —
    /// every answer here is a comparison against a row's centre — but they have to be
    /// consistent, which is what makes a drag of `STEP` "one row down".
    const H: f32 = 20.0;
    const GAP: f32 = 8.0;
    const STEP: f32 = H + GAP;
    const FIRST: f32 = 100.0;

    fn boxes(n: usize) -> Vec<(f32, f32)> {
        (0..n).map(|i| (FIRST + i as f32 * STEP, H)).collect()
    }

    /// One row moved down past its neighbours, a row at a time.
    #[test]
    fn a_row_yields_when_the_leading_edge_passes_its_centre() {
        let b = boxes(4);
        for (dy, gap) in [(0.0, 0), (STEP, 1), (2.0 * STEP, 2), (9.0 * STEP, 3)] {
            let s = Slide::resolve(&b, (0, 0), dy).expect("resolves");
            assert_eq!(s.gap, gap, "{dy}px down");
        }
    }

    /// And up, where the *top* edge is what leads. The asymmetry is the whole reason
    /// both are tested: one rule stated once for a block that can be dragged either
    /// way is easy to write as two rules that disagree near the ends.
    #[test]
    fn dragging_up_leads_with_the_top_edge() {
        let b = boxes(4);
        for (dy, gap) in [(0.0, 3), (-STEP, 2), (-2.0 * STEP, 1), (-9.0 * STEP, 0)] {
            let s = Slide::resolve(&b, (3, 3), dy).expect("resolves");
            assert_eq!(s.gap, gap, "{dy}px up");
        }
    }

    /// The slot a block opens is the size of the block, so the rows that close up
    /// behind it and the rows that open ahead of it move by the same amount and the
    /// column's total extent never changes.
    #[test]
    fn the_slot_is_the_size_of_what_goes_into_it() {
        let b = boxes(5);
        let one = Slide::resolve(&b, (0, 0), STEP).expect("resolves");
        let three = Slide::resolve(&b, (0, 2), 3.0 * STEP).expect("resolves");
        assert!((one.step - STEP).abs() < 0.01, "{}", one.step);
        assert!((three.step - 3.0 * STEP).abs() < 0.01, "{}", three.step);
    }

    /// Displaced rows move by exactly one slot, and only the ones between where the
    /// block was and where it is going move at all.
    #[test]
    fn only_the_rows_the_block_crossed_slide() {
        let b = boxes(4);
        let s = Slide::resolve(&b, (0, 0), 2.0 * STEP).expect("resolves");
        assert_eq!(s.gap, 2);
        assert_eq!(
            s.motion(0, (0.0, 55.0)).shift,
            (0.0, 55.0),
            "the block itself"
        );
        assert!(s.motion(0, (0.0, 55.0)).lifted);
        assert_eq!(s.motion(1, (0.0, 0.0)).shift, (0.0, -s.step), "closed up");
        assert_eq!(s.motion(2, (0.0, 0.0)).shift, (0.0, -s.step), "closed up");
        assert_eq!(s.motion(3, (0.0, 0.0)).shift, (0.0, 0.0), "never crossed");
    }

    /// A press that has not travelled is not a drag, and a drag that has stays one.
    #[test]
    fn a_press_becomes_a_drag_once_and_stays_one() {
        let mut g = Grab::begin("a", vec![], (0.0, 0.0));
        assert!(!g.live());
        g.track((0.0, GRAB_SLOP), true);
        assert!(!g.live(), "the slop itself is still a click");
        g.track((0.0, GRAB_SLOP + 1.0), true);
        assert!(g.live());
        g.track((0.0, 0.0), true);
        assert!(g.live(), "back where it started is still a drag");
    }

    /// **A spent grab is dead.** It outlives its gesture by a moment, for the click
    /// behind the release ([`claimed`]) — and a row's grip is also a thing the pointer
    /// merely passes over, so it hears hovers in that moment. One of them re-arming it
    /// left the dropped row following the pointer around the panel, over a list it had
    /// already been moved in, with no gesture left that could put it down.
    #[test]
    fn a_spent_grab_is_not_re_armed_by_the_hover_behind_it() {
        let mut g = Grab::begin("a", vec![], (0.0, 0.0));
        g.track((0.0, 8.0 * GRAB_SLOP), true);
        assert!(g.live());
        g.spend();
        // Far from the anchor, which is the whole point: the pointer is wherever the
        // drop left it, so every one of these is a travel of more than the slop.
        for at in [(0.0, 8.0 * GRAB_SLOP), (30.0, 200.0)] {
            g.track(at, false);
            assert!(!g.live(), "a hover at {at:?} woke a spent grab");
            g.track(at, true);
            assert!(!g.live(), "a press at {at:?} woke a spent grab");
        }
        assert!(g.over(), "and it is still the receipt the click needs");
    }

    /// The release this panel never heard. A drag that ends over anything without a
    /// handler — the empty space past the end of the list, which is where a row
    /// dragged to the bottom is let go — sends no `pointerup` here at all; the first
    /// move with nothing held is the panel finding out, and it ends the gesture rather
    /// than steering it, because a drop is committed by a release and this is not one.
    #[test]
    fn a_move_with_nothing_held_ends_the_gesture() {
        let mut g = Grab::begin("a", vec![], (0.0, 0.0));
        g.track((0.0, 8.0 * GRAB_SLOP), true);
        assert!(g.live());
        g.track((0.0, 9.0 * GRAB_SLOP), false);
        assert!(!g.live(), "the press is not down, so there is no drag");
        assert!(g.over());
    }

    /// A row the press never measured abandons the gesture rather than guessing.
    #[test]
    fn an_unmeasured_row_abandons_the_gesture() {
        let measured = vec![("a".into(), 0.0, H), ("b".into(), STEP, H)];
        let g = Grab::begin("a", measured, (0.0, 0.0));
        let keys = |ks: &[&str]| ks.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(g.resolve(&keys(&["a", "b"])).map(|(i, _)| i), Some(0));
        assert!(g.resolve(&keys(&["a", "b", "c"])).is_none(), "c has no box");
        assert!(
            g.resolve(&keys(&["b"])).is_none(),
            "the grabbed row is gone"
        );
    }

    /// Every row's transform is written on every render, including at rest — the
    /// declaration a render leaves out is the one that goes stale.
    #[test]
    fn a_resting_row_still_states_its_transform() {
        let css = Motion::default().css();
        assert!(css.contains("translate(0px, 0px)"), "{css}");
        assert!(css.contains("transition: none"), "{css}");
    }
}
