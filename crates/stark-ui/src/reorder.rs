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
//! - [`Grab`](crate::reorder::Grab) is the pointer bookkeeping: what was pressed, where the hand started
//!   and is now, and every row's box **as it stood at the press**. Nothing derived
//!   is kept, so a row that has slid out of the way cannot feed back into the
//!   decision that moved it — the same bargain the panel-stack drag makes
//!   (`stark-dioxus-frontend`'s `layout::ThumbGrab`).
//! - [`Slide`](crate::reorder::Slide) is the column arithmetic: given the block of rows that travels and
//!   how far the hand has taken it, which rows yield and by how much.
//! - [`Motion`](crate::reorder::Motion) is what one row does about it: where the row
//!   sits relative to where it belongs, whether it is the one being carried, and
//!   whether it should ease there. How that is *drawn* is each frontend's — the web
//!   one writes it as two CSS declarations (`panels::reorder::css`).
//!
//! A row that is *both* pressable and draggable has one problem this module also
//! owns: the browser sends a `click` after the release, and that click belongs to
//! the drag rather than to the row. Swallowing that click needs a signal, so it is the
//! one piece of this left in each frontend — `panels::reorder::claimed` in the web one.

/// How far the pointer must travel before a press on a row is a drag rather than a
/// click. Small, because a row's other gestures are a click and a double-click and
/// neither survives being mistaken for a drag; large enough that the hand's own
/// tremor on a pen does not move anything.
const GRAB_SLOP: f32 = 4.0;

/// One row's vertical box, with the identity the element it was measured from wears.
///
/// The identity is what a panel matches its rows by; matched by position a row would
/// be measured through its neighbour's box in silence.
#[derive(Clone, PartialEq, Debug)]
pub struct RowBox {
    /// The `data-` attribute `platform::layer_boxes` and its neighbours read.
    pub key: String,
    pub top: f32,
    pub height: f32,
}

impl From<(String, f32, f32)> for RowBox {
    /// The shape a DOM measurement arrives in — `(key, top, height)`, named here so
    /// that is the last place the order has to be remembered.
    fn from((key, top, height): (String, f32, f32)) -> Self {
        Self { key, top, height }
    }
}

/// A row's vertical box: where it starts and how tall it is.
///
/// **Never where it ends.** [`Slide::resolve`] adds the height to get the bottom, so
/// a caller handing it `(top, bottom)` got geometry off by a height and no type error
/// for it. The field name is the check two `f32`s in a tuple could not be.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Extent {
    pub top: f32,
    pub height: f32,
}

impl Extent {
    /// The row's lower edge.
    pub fn bottom(&self) -> f32 {
        self.top + self.height
    }

    /// The line a row yields at — the leading edge of the travelling block has to
    /// cross it (§14.8).
    fn center(&self) -> f32 {
        self.top + self.height * 0.5
    }
}

/// Where a press on a row has got to.
///
/// Three states rather than two flags, which could spell a fourth: a grab that is
/// both spent and live is a dropped row still following the pointer, and it was ruled
/// out only by `spend` happening to clear one of them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Phase {
    /// Pressed, not travelled: still a click.
    Pressed,
    /// Travelled: a drag, and it stays one.
    Dragging,
    /// Landed, or ended by a press that is no longer down. **Terminal.**
    Spent,
}

/// A press on a row, which may or may not have become a drag yet.
#[derive(Clone, PartialEq)]
pub struct Grab {
    key: String,
    boxes: Vec<RowBox>,
    anchor: (f32, f32),
    pointer: (f32, f32),
    phase: Phase,
}

impl Grab {
    /// Arm a press on the row wearing `key`, with `boxes` as the whole list stood at
    /// that moment and `at` the pointer in client px.
    ///
    /// Measured on the press rather than on the first move, so the list is described
    /// as it stood when the hand closed on it and no pointer travel is lost waiting
    /// for a measurement.
    pub fn begin(
        key: impl Into<String>,
        boxes: impl IntoIterator<Item = impl Into<RowBox>>,
        at: (f32, f32),
    ) -> Self {
        Self {
            key: key.into(),
            boxes: boxes.into_iter().map(Into::into).collect(),
            anchor: at,
            pointer: at,
            phase: Phase::Pressed,
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
    /// grab is kept a moment longer for the click behind it (the frontend's `claimed`), and it must
    /// not be steerable while it waits. Both are ruled out here rather than by asking
    /// every caller to check, because a caller that forgets leaves a panel only a
    /// close and reopen can put right.
    pub fn track(&mut self, at: (f32, f32), held: bool) {
        if self.phase == Phase::Spent {
            return;
        }
        if !held {
            self.spend();
            return;
        }
        self.pointer = at;
        let (dx, dy) = self.delta();
        if dx.abs().max(dy.abs()) > GRAB_SLOP {
            self.phase = Phase::Dragging;
        }
    }

    /// Whether the pointer has said this is a drag. A press that has not travelled
    /// yet draws nothing and lands nothing.
    pub fn live(&self) -> bool {
        self.phase == Phase::Dragging
    }

    /// Whether the gesture is finished — landed, or ended by a press that is no longer
    /// down. Such a grab draws nothing and follows nothing; it is kept only until the
    /// click behind it arrives (see the frontend's `claimed`), which is what a panel checks this
    /// for: there is no gesture here to feed.
    pub fn over(&self) -> bool {
        self.phase == Phase::Spent
    }

    /// How far the hand has taken the row, in px.
    pub fn delta(&self) -> (f32, f32) {
        (
            self.pointer.0 - self.anchor.0,
            self.pointer.1 - self.anchor.1,
        )
    }

    /// Where the grabbed row sits among `keys` — the panel's rows in the order it
    /// draws them — and every row's extent.
    ///
    /// `None` if any row was not measured, or if the grabbed row is no longer in the
    /// list. That is what a list changing under the pointer looks like, and the
    /// answer to it is to draw at rest and land nothing rather than to act on
    /// geometry describing a list that is gone.
    pub fn resolve(&self, keys: &[String]) -> Option<(usize, Vec<Extent>)> {
        let at = keys.iter().position(|k| *k == self.key)?;
        let boxes = keys
            .iter()
            .map(|k| {
                self.boxes.iter().find(|b| b.key == *k).map(|b| Extent {
                    top: b.top,
                    height: b.height,
                })
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
    /// recognized — see the frontend's `claimed`.
    ///
    /// **Terminal**: nothing brings a spent grab back to life. It is what [`track`]
    /// checks first, and the reason it is a state of the grab rather than the panel
    /// dropping it — a panel that has just committed a move must not be able to
    /// commit it again from the same press.
    ///
    /// [`track`]: Self::track
    pub fn spend(&mut self) {
        self.phase = Phase::Spent;
    }
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
    pub fn resolve(boxes: &[Extent], block: (usize, usize), dy: f32) -> Option<Self> {
        let (start, end) = block;
        if start > end || end >= boxes.len() {
            return None;
        }
        let (block_top, block_bottom) = (boxes[start].top, boxes[end].bottom());
        // The space between two rows, so a slide closes the slot exactly rather than
        // leaving a seam the width of a margin.
        let gap_px = if boxes.len() > 1 {
            (boxes[1].top - boxes[0].bottom()).max(0.0)
        } else {
            0.0
        };
        let (top, bottom) = (block_top + dy, block_bottom + dy);
        let gap = (0..boxes.len())
            .filter(|i| *i < start || *i > end)
            .filter(|&k| {
                let center = boxes[k].center();
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
    /// pointer, so it tracks the hand 1:1 and slides for nobody. `.dragging` in the
    /// markup — the hook every reorderable list's stylesheet hangs the lift on.
    pub lifted: bool,
    /// A drag is in flight, so a row that is *not* lifted eases into its new place
    /// rather than jumping there.
    pub live: bool,
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

    fn boxes(n: usize) -> Vec<Extent> {
        (0..n)
            .map(|i| Extent {
                top: FIRST + i as f32 * STEP,
                height: H,
            })
            .collect()
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
        let mut g = Grab::begin("a", Vec::<RowBox>::new(), (0.0, 0.0));
        assert!(!g.live());
        g.track((0.0, GRAB_SLOP), true);
        assert!(!g.live(), "the slop itself is still a click");
        g.track((0.0, GRAB_SLOP + 1.0), true);
        assert!(g.live());
        g.track((0.0, 0.0), true);
        assert!(g.live(), "back where it started is still a drag");
    }

    /// **A spent grab is dead.** It outlives its gesture by a moment, for the click
    /// behind the release (the frontend's `claimed`) — and a row's grip is also a thing the pointer
    /// merely passes over, so it hears hovers in that moment. One of them re-arming it
    /// left the dropped row following the pointer around the panel, over a list it had
    /// already been moved in, with no gesture left that could put it down.
    #[test]
    fn a_spent_grab_is_not_re_armed_by_the_hover_behind_it() {
        let mut g = Grab::begin("a", Vec::<RowBox>::new(), (0.0, 0.0));
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
        let mut g = Grab::begin("a", Vec::<RowBox>::new(), (0.0, 0.0));
        g.track((0.0, 8.0 * GRAB_SLOP), true);
        assert!(g.live());
        g.track((0.0, 9.0 * GRAB_SLOP), false);
        assert!(!g.live(), "the press is not down, so there is no drag");
        assert!(g.over());
    }

    /// A row the press never measured abandons the gesture rather than guessing.
    #[test]
    fn an_unmeasured_row_abandons_the_gesture() {
        let measured: Vec<RowBox> = vec![("a".into(), 0.0, H).into(), ("b".into(), STEP, H).into()];
        let g = Grab::begin("a", measured, (0.0, 0.0));
        let keys = |ks: &[&str]| ks.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(g.resolve(&keys(&["a", "b"])).map(|(i, _)| i), Some(0));
        assert!(g.resolve(&keys(&["a", "b", "c"])).is_none(), "c has no box");
        assert!(
            g.resolve(&keys(&["b"])).is_none(),
            "the grabbed row is gone"
        );
    }

    /// **A block dragged up**, which is where the leading-edge rule and the index
    /// remap meet: going up the block's *top* leads, and the rows it passes are the
    /// ones before it — the half of `motion`'s arithmetic the downward cases never
    /// reach.
    #[test]
    fn a_block_dragged_up_takes_the_rows_it_passed_with_it() {
        let b = boxes(5);
        // Rows 2 and 3 travel two rows up. `gap == 0` is the block landing above
        // every row that stayed put.
        let s = Slide::resolve(&b, (2, 3), -2.0 * STEP).expect("resolves");
        assert_eq!(s.gap, 0);
        assert!((s.step - 2.0 * STEP).abs() < 0.01, "{}", s.step);
        // The rows above it open up by exactly the slot, and the one below never
        // moved — nothing between the block and the end of the list was crossed.
        assert_eq!(s.motion(0, (0.0, 0.0)).shift, (0.0, s.step), "opened");
        assert_eq!(s.motion(1, (0.0, 0.0)).shift, (0.0, s.step), "opened");
        assert!(s.motion(2, (0.0, -40.0)).lifted);
        assert!(s.motion(3, (0.0, -40.0)).lifted);
        assert_eq!(s.motion(4, (0.0, 0.0)).shift, (0.0, 0.0), "never crossed");
    }

    /// **The two halves of a slide have to describe the same list.** `motion` draws
    /// the rows shifting and `gap` commits the order; nothing in the code makes them
    /// agree, and a disagreement is a row that animates into a slot it does not land
    /// in. Exhaustive over every list up to six rows, every contiguous block in it,
    /// and every travel that reaches past either end.
    #[test]
    fn what_the_shifts_draw_is_the_order_the_gap_commits() {
        for n in 1..=6usize {
            let b = boxes(n);
            for start in 0..n {
                for end in start..n {
                    for k in -(n as i32)..=(n as i32) {
                        let dy = k as f32 * STEP;
                        let s = Slide::resolve(&b, (start, end), dy).expect("resolves");

                        // Where each row is drawn: its own top plus whatever the
                        // block's shift or its own slide puts it at.
                        let mut drawn: Vec<(usize, f32)> = (0..n)
                            .map(|i| {
                                let m = s.motion(i, (0.0, dy));
                                (i, b[i].top + m.shift.1)
                            })
                            .collect();
                        drawn.sort_by(|a, c| a.1.total_cmp(&c.1));
                        let drawn: Vec<usize> = drawn.into_iter().map(|(i, _)| i).collect();

                        // And where `gap` says they land: the block taken out and put
                        // back after that many of the rows that stayed.
                        let mut rest: Vec<usize> =
                            (0..n).filter(|i| *i < start || *i > end).collect();
                        let block: Vec<usize> = (start..=end).collect();
                        let mut committed = rest.split_off(s.gap);
                        let mut landed = rest;
                        landed.extend(block);
                        landed.append(&mut committed);

                        assert_eq!(drawn, landed, "n={n} block=({start},{end}) dy={k} rows",);
                    }
                }
            }
        }
    }
}
