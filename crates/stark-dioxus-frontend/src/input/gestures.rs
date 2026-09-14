//! The canvas's four gestures as one value (§25.4): which of them holds the
//! pointer, which of them a press may open, the order a move is offered to them
//! in, and the release that puts them all down.
//!
//! The press ladder's *routing* — which binding a press is — stays at the canvas;
//! what is here is everything about the gestures that does not depend on it.

use super::*;

/// The canvas's gestures — paint (behind the wait a finger's press is held in),
/// navigation, the brush-tuning drag and the layer carry — made once by
/// [`use_gestures`].
///
/// The overlays make their own [`Nav`] and none of the rest.
#[derive(Clone, Copy)]
pub struct Gestures {
    state: AppState,
    landing: Landing,
    nav: Nav,
    tune: Tune,
    carry: PickMove,
}

/// Which of the canvas's gestures holds the pointer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Holder {
    Nav,
    Tune,
    Carry,
    /// A stroke or a marquee, or a finger's press still held in front of one.
    Paint,
}

/// A rung of the move ladder above paint — what took a move.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Rung {
    Nav,
    Tune,
    /// A composing mode opened under a captured pointer, and the move put down what
    /// the canvas was previewing.
    Mode,
    Carry,
}

/// The order a move is offered in. `Nav` first because it records every finger's
/// move whoever takes it (§18.1.7); `Mode` above every rung that holds a document
/// preview, so a mode's arrival reaches that gesture's `abandon` before another move
/// renews the preview (§25.4).
const MOVE_LADDER: [Rung; 4] = [Rung::Nav, Rung::Tune, Rung::Mode, Rung::Carry];

/// Whether a press may open `opens` while `holder` holds the pointer.
///
/// Paint yields: the press of every other family abandons it. Any other holder
/// refuses a second pointer's press unless it opens the same family — a further
/// finger joining a pinch, a press over its own lost release — which is what keeps
/// two gestures from being in flight at once.
fn admits(holder: Option<Holder>, opens: Holder) -> bool {
    match holder {
        None | Some(Holder::Paint) => true,
        Some(held) => held == opens,
    }
}

/// The canvas's gestures. A hook: call unconditionally, like any `use_*`.
pub fn use_gestures(state: AppState) -> Gestures {
    let paint = Paint::use_paint(state);
    Gestures {
        state,
        landing: Landing::use_landing(state, paint),
        nav: Nav::use_nav(state),
        tune: Tune::use_tune(state),
        carry: PickMove::use_pick_move(state),
    }
}

impl Gestures {
    /// Which gesture holds the pointer, if any.
    pub fn holder(self) -> Option<Holder> {
        let holding = [
            (Holder::Nav, self.nav.holds_pointer()),
            (Holder::Tune, self.tune.holds_pointer()),
            (Holder::Carry, self.carry.holds_pointer()),
            (Holder::Paint, self.landing.holds_pointer()),
        ];
        let mut held = holding.into_iter().filter_map(|(h, on)| on.then_some(h));
        let holder = held.next();
        debug_assert!(
            held.next().is_none(),
            "two canvas gestures hold the pointer: {holding:?}"
        );
        holder
    }

    /// Whether a press may open `opens` now ([`admits`]).
    pub fn admits(self, opens: Holder) -> bool {
        admits(self.holder(), opens)
    }

    /// Open navigation at `e` ([`Nav::begin`]). `true` means the press was taken.
    pub fn begin_nav(self, e: &Event<PointerData>) -> bool {
        let taken = self.admits(Holder::Nav) && self.nav.begin(e);
        if taken {
            // Whatever was being drawn was the opening half of a pinch (§18.1.7), or
            // a pen stroke a middle-drag interrupted.
            self.abandon_paint();
        }
        taken
    }

    /// Open the brush-tuning drag at `e` ([`Tune::begin`]).
    pub fn begin_tune(self, e: &Event<PointerData>) -> bool {
        let taken = self.admits(Holder::Tune) && self.tune.begin(e);
        if taken {
            self.abandon_paint();
        }
        taken
    }

    /// Open the layer carry at `e` ([`PickMove::begin`]).
    pub fn begin_carry(self, e: &Event<PointerData>) -> bool {
        let taken = self.admits(Holder::Carry) && self.carry.begin(e);
        if taken {
            self.abandon_paint();
        }
        taken
    }

    /// Open paint at `e` with `tool` ([`Landing::begin`]).
    pub fn begin_paint(self, e: &Event<PointerData>, tool: Tool) -> bool {
        self.admits(Holder::Paint) && self.landing.begin(e, tool)
    }

    /// The press is not paint: a stroke another pointer had in flight can no longer
    /// be finished by it and must leave no mark, and the hover's promise of paint is
    /// withdrawn (§18.1.10).
    fn abandon_paint(self) {
        self.landing.abandon();
        hover_gone(self.state);
    }

    /// Offer a move to the rungs above paint, in [`MOVE_LADDER`]'s order. The rung
    /// that took it, whose move nothing below may see; `None` leaves it to paint or
    /// the hover.
    pub fn advance(self, e: &Event<PointerData>) -> Option<Rung> {
        for rung in MOVE_LADDER {
            let taken = match rung {
                Rung::Nav => self.nav.advance(e),
                Rung::Tune => self.tune.advance(e),
                Rung::Mode => self.interrupted(),
                Rung::Carry => self.carry.advance(e),
            };
            if taken {
                // Not a move that paints, so nothing may promise paint under it.
                hover_gone(self.state);
                return Some(rung);
            }
        }
        None
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
    pub fn paints(self, e: &Event<PointerData>) -> bool {
        self.landing.claims(e)
    }

    /// Feed paint a move of `e`'s, `samples` being every report it carries
    /// ([`Landing::advance`]).
    pub fn advance_paint(self, e: &Event<PointerData>, samples: &[InputSample]) -> bool {
        self.landing.advance(e, samples)
    }

    /// The canvas's navigation, for the release, the tap and the wheel.
    pub fn nav(self) -> Nav {
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
    }
}

/// End everything the canvas can have in hand — its gestures and the eyedropper —
/// and hand the canvas back, so the floating chrome fades in.
pub fn end_interaction(gestures: Gestures) {
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

    const FAMILIES: [Holder; 4] = [Holder::Nav, Holder::Tune, Holder::Carry, Holder::Paint];

    #[test]
    fn a_free_canvas_or_a_stroke_admits_every_press() {
        for holder in [None, Some(Holder::Paint)] {
            for opens in FAMILIES {
                assert!(admits(holder, opens), "{holder:?} refused {opens:?}");
            }
        }
    }

    #[test]
    fn any_other_holder_admits_only_its_own_family() {
        for held in [Holder::Nav, Holder::Tune, Holder::Carry] {
            for opens in FAMILIES {
                assert_eq!(
                    admits(Some(held), opens),
                    held == opens,
                    "{held:?} holding, {opens:?} pressing"
                );
            }
        }
    }

    #[test]
    fn a_mode_reaches_the_carry_before_its_next_move() {
        let rank = |rung: Rung| {
            MOVE_LADDER
                .iter()
                .position(|&r| r == rung)
                .expect("every rung is on the ladder")
        };
        // Nav records every finger's move, so nothing may take one ahead of it.
        assert_eq!(rank(Rung::Nav), 0);
        // The carry holds a document preview; paint, the other, is below the ladder.
        assert!(rank(Rung::Mode) < rank(Rung::Carry));
    }
}
