//! The canvas's four gestures as one value (§25.4): which of them holds the
//! pointer, which of them a press may open, the order a move is offered to them
//! in, and the release that puts them all down.
//!
//! The press ladder's *routing* — which binding a press is — stays at the canvas;
//! what is here is everything about the gestures that does not depend on it.

use super::*;
use stark_ui::nav::Lift;

/// The canvas's gestures — paint (behind the wait a finger's press is held in),
/// navigation, the brush-tuning drag and the layer carry — made once by
/// [`use_gestures`].
///
/// The overlays make their own [`Nav`] and none of the rest.
#[derive(Clone, Copy)]
pub(crate) struct Gestures {
    state: AppState,
    landing: Landing,
    nav: Nav,
    tune: Tune,
    carry: PickMove,
    /// The pointer whose press opened the gesture last begun. Read only beside
    /// [`holder`](Gestures::holder), which says whether that gesture still holds.
    held_by: Signal<Option<i32>>,
}

/// Which of the canvas's gestures holds the pointer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Holder {
    Nav,
    Tune,
    Carry,
    /// A stroke or a marquee, or a finger's press still held in front of one.
    Paint,
}

/// Whether a press may open `opens` while `holder` holds the pointer.
///
/// Paint yields: the press of every other family abandons it. Any other holder
/// refuses a second pointer's press unless it opens the same family — a further
/// finger joining a pinch — which is what keeps two gestures from being in flight at
/// once.
fn admits(holder: Option<Holder>, opens: Holder) -> bool {
    match holder {
        None | Some(Holder::Paint) => true,
        Some(held) => held == opens,
    }
}

/// The one gesture `holding` marks as holding the pointer, if any.
fn holder_of(holding: [(Holder, bool); 4]) -> Option<Holder> {
    let mut held = holding.into_iter().filter_map(|(h, on)| on.then_some(h));
    let holder = held.next();
    debug_assert!(
        held.next().is_none(),
        "two canvas gestures hold the pointer: {holding:?}"
    );
    holder
}

/// Whether a release of pointer `released` ends the interaction, `held_by` being the
/// pointer that holds the canvas. Another pointer's release — a palm under a stroke, a
/// pen under a carry — ends nothing, or it would commit the gesture mid-drag.
fn release_ends(held_by: Option<i32>, released: i32) -> bool {
    held_by.is_none_or(|id| id == released)
}

/// Whether a press of pointer `pressed` proves the release of the gesture `held_by`
/// holds was lost: a pointer cannot press while it is already down.
fn release_lost(held_by: Option<i32>, pressed: i32) -> bool {
    held_by == Some(pressed)
}

/// The canvas's gestures. A hook: call unconditionally, like any `use_*`.
pub(crate) fn use_gestures(state: AppState) -> Gestures {
    let paint = Paint::use_paint(state);
    Gestures {
        state,
        landing: Landing::use_landing(state, paint),
        nav: Nav::use_nav(state),
        tune: Tune::use_tune(state),
        carry: PickMove::use_pick_move(state),
        held_by: use_signal(|| None),
    }
}

impl Gestures {
    /// Which gesture holds the pointer, if any.
    pub(crate) fn holder(self) -> Option<Holder> {
        holder_of([
            (Holder::Nav, self.nav.holds_pointer()),
            (Holder::Tune, self.tune.holds_pointer()),
            (Holder::Carry, self.carry.holds_pointer()),
            (Holder::Paint, self.landing.holds_pointer()),
        ])
    }

    /// The pointer holding the canvas, if a gesture does.
    fn held_by(self) -> Option<i32> {
        self.holder().and(*self.held_by.peek())
    }

    /// Whether a press may open `opens` now ([`admits`]).
    fn admits(self, opens: Holder) -> bool {
        admits(self.holder(), opens)
    }

    /// Whether `e`'s press proves the holder's release was lost ([`release_lost`]), so the
    /// interaction has to end before the press is read.
    pub(crate) fn release_lost(self, e: &Event<PointerData>) -> bool {
        release_lost(self.held_by(), e.pointer_id())
    }

    /// Open navigation at `e` ([`Nav::begin`]). `true` means the press was taken.
    pub(crate) fn begin_nav(self, e: &Event<PointerData>) -> bool {
        // Recorded before `admits` is asked: a refused finger is still on the glass, and
        // the next one to land after the holder lets go has to pair with it.
        let Some(press) = self.nav.record(e) else {
            return false;
        };
        if !self.admits(Holder::Nav) {
            return false;
        }
        self.nav.claim(e, press);
        self.took(e);
        // Whatever was being drawn was the opening half of a pinch (§18.1.7), or
        // a pen stroke a middle-drag interrupted.
        self.abandon_paint();
        true
    }

    /// Open the brush-tuning drag at `e` ([`Tune::begin`]).
    pub(crate) fn begin_tune(self, e: &Event<PointerData>) -> bool {
        let taken = self.admits(Holder::Tune) && self.tune.begin(e);
        if taken {
            self.took(e);
            self.abandon_paint();
        }
        taken
    }

    /// Open the layer carry at `e` ([`PickMove::begin`]).
    pub(crate) fn begin_carry(self, e: &Event<PointerData>) -> bool {
        let taken = self.admits(Holder::Carry) && self.carry.begin(e);
        if taken {
            self.took(e);
            self.abandon_paint();
        }
        taken
    }

    /// Open paint at `e` with `tool` ([`Landing::begin`]). `false` when another pointer's
    /// gesture refuses it, or the gesture declined it — a palm, or no engine yet.
    pub(crate) fn begin_paint(self, e: &Event<PointerData>, tool: Tool) -> bool {
        let taken = self.admits(Holder::Paint) && self.landing.begin(e, tool);
        if taken {
            self.took(e);
        }
        taken
    }

    /// Record `e`'s pointer as the one holding the gesture just begun.
    fn took(self, e: &Event<PointerData>) {
        let mut held_by = self.held_by;
        held_by.set(Some(e.pointer_id()));
    }

    /// The press is not paint: a stroke another pointer had in flight can no longer
    /// be finished by it and must leave no mark, and the hover's promise of paint is
    /// withdrawn (§18.1.10).
    fn abandon_paint(self) {
        self.landing.abandon();
        hover_gone(self.state);
    }

    /// Offer a move to the gestures above paint. `true` means one took it, and nothing
    /// below may see it; `false` leaves it to paint or the hover.
    ///
    /// The order is §25.4's. Navigation first, because it records every finger's move
    /// whoever takes it (§18.1.7). The mode check above every gesture holding a document
    /// preview — the carry here, paint below — so a mode's arrival reaches that gesture's
    /// `abandon` before another move renews the preview.
    pub(crate) fn advance(self, e: &Event<PointerData>) -> bool {
        let taken = self.nav.advance(e)
            || self.tune.advance(e)
            || self.interrupted()
            || self.carry.advance(e);
        if taken {
            // Not a move that paints, so nothing may promise paint under it.
            hover_gone(self.state);
        }
        taken
    }

    /// Whether a composing mode has opened under the hand (`crate::modes`), and if
    /// so, cancel what the canvas was previewing.
    ///
    /// Its catcher covers the canvas, but a pointer the canvas captured before the
    /// catcher existed still delivers here. Cancelled rather than committed: the
    /// canvas stopped taking paint the moment the mode took it. And the chrome comes
    /// back, or the mode's own bar would stay faded until the pen lifted (§11).
    fn interrupted(self) -> bool {
        if !crate::modes::is_composing(self.state) {
            return false;
        }
        self.landing.abandon();
        self.carry.abandon();
        let mut canvas_active = self.state.canvas_active;
        canvas_active.set(false);
        true
    }

    /// Whether paint takes a move of `e`'s ([`Landing::claims`]).
    pub(crate) fn paints(self, e: &Event<PointerData>) -> bool {
        self.landing.claims(e)
    }

    /// Feed paint a move of `e`'s, `samples` being every report it carries
    /// ([`Landing::advance`]).
    pub(crate) fn advance_paint(self, e: &Event<PointerData>, samples: &[InputSample]) -> bool {
        self.landing.advance(e, samples)
    }

    /// Route a release or cancel of `e`'s pointer. [`Lift::Ended`] when the interaction
    /// is over, carrying the tap the hand made; [`Lift::Continuing`] when fingers are
    /// still down or another pointer's gesture holds the canvas ([`release_ends`]).
    pub(crate) fn release(self, e: &Event<PointerData>) -> Lift {
        if self.nav.release(e) {
            return Lift::Continuing;
        }
        // Spent whether or not this release ends anything: left for the holder's own
        // release, a refused pair's tap would undo on it.
        let tap = self.nav.take_tap();
        if release_ends(self.held_by(), e.pointer_id()) {
            Lift::Ended { tap }
        } else {
            Lift::Continuing
        }
    }

    /// The canvas's navigation, for the wheel.
    pub(crate) fn nav(self) -> Nav {
        self.nav
    }

    /// Put every gesture down.
    fn end(self) {
        self.landing.end();
        self.nav.stop();
        self.tune.stop();
        // The one whose ending the release does not finish: the layer it picked up
        // may still be a readback away, and the commit waits for it
        // (`PickMove::settle`).
        self.carry.stop();
        let mut held_by = self.held_by;
        if held_by.peek().is_some() {
            held_by.set(None);
        }
    }
}

/// End everything the canvas can have in hand — its gestures and the eyedropper —
/// and hand the canvas back, so the floating chrome fades in.
pub(crate) fn end_interaction(gestures: Gestures) {
    let state = gestures.state;
    gestures.end();
    // Not a gesture of the carrier's because the eyedropper's drag flag is shared
    // state — the options bar reads it (see `PickState`). Nothing to undo, either: a
    // sample already in flight is left to land, since it is the answer to a press the
    // user made.
    let mut dragging = state.pick.dragging;
    dragging.set(false);
    // And the swatch a held pick was showing goes with the finger that asked for it
    // (§18.1.11). Guarded like every other idle write here: this runs on every
    // release the canvas sees, and almost none of them had a loupe up.
    let mut loupe = state.pick.loupe;
    if loupe.peek().is_some() {
        loupe.set(None);
    }
    // The panel stack does not come straight back: it stays out of the way until the
    // pointer reaches into its column (`AppState::panels_asleep`, §11). The chrome
    // going *out* mid-stroke was never the distracting half — coming back the instant
    // the pen lifts is, because it happens at exactly the moment the artist is looking
    // at what they just drew.
    //
    // Gated on the fade having actually been in force. This runs on every release the
    // canvas sees, including the ones that deliberately keep the chrome up — an
    // eyedropper sample reads its answer off the Color panel, brush tuning off the
    // Brush panel — and putting the stack to sleep on the way out of those would hide
    // the panel the gesture was for. Read before the clear, since the clear is what
    // makes it false.
    //
    // Whether it sleeps at all is this browser's own choice, and that question is
    // asked inside `sleep_panels` rather than here — one door, so the setting reaches
    // every caller (`layout::ChromeHiding`, §11).
    let was_faded = *state.canvas_active.peek();
    let mut canvas_active = state.canvas_active;
    canvas_active.set(false);
    if was_faded {
        crate::layout::sleep_panels(state);
    }
    // And a drag-preset offer brought due by a press this release is the end of
    // (§25.8). Here rather than at the press for `tutor`'s reason: the press
    // that finds nothing bound goes on to paint, and a modal over a live stroke
    // would take the canvas away mid-mark. Last, because it is the one thing in
    // this function that puts something *up*.
    crate::drags::settle_offer(state);
}

#[cfg(test)]
mod tests {
    use super::*;

    const MOUSE: i32 = 1;
    const PEN: i32 = 2;
    const PALM: i32 = 7;

    fn holding([nav, tune, carry, paint]: [bool; 4]) -> [(Holder, bool); 4] {
        [
            (Holder::Nav, nav),
            (Holder::Tune, tune),
            (Holder::Carry, carry),
            (Holder::Paint, paint),
        ]
    }

    #[test]
    fn the_holder_is_the_one_gesture_holding() {
        assert_eq!(holder_of(holding([false; 4])), None);
        assert_eq!(
            holder_of(holding([false, true, false, false])),
            Some(Holder::Tune)
        );
        assert_eq!(
            holder_of(holding([false, false, false, true])),
            Some(Holder::Paint)
        );
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "two canvas gestures hold the pointer")]
    fn two_holders_are_a_bug() {
        holder_of(holding([true, false, false, true]));
    }

    #[test]
    fn only_the_holders_release_ends_its_gesture() {
        assert!(
            !release_ends(Some(PEN), PALM),
            "a palm lifting under a stroke"
        );
        assert!(
            !release_ends(Some(MOUSE), PEN),
            "a pen lifting under a carry"
        );
        assert!(release_ends(Some(PEN), PEN));
        assert!(
            release_ends(None, PALM),
            "nothing holds, so nothing to protect"
        );
    }

    #[test]
    fn a_press_from_the_holders_pointer_is_a_lost_release() {
        assert!(release_lost(Some(MOUSE), MOUSE));
        assert!(
            !release_lost(Some(MOUSE), PEN),
            "a second pointer, not a lost one"
        );
        assert!(!release_lost(None, MOUSE));
    }
}
