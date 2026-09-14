//! The canvas's gestures as one value (§25.4): what holds the pointer, what a press may
//! open, the order a move is offered in, and the release that puts them all down.
//!
//! The press ladder's *routing* — which binding a press is — stays at the canvas. What a
//! second pointer's event may do is `stark_ui::route`'s, asked of what each gesture
//! records about its own pointer.

use super::*;
use stark_ui::nav::Lift;
use stark_ui::route::{self, Gesture, Holder, Moves, Opens};

/// The canvas's gestures — paint (behind the wait a finger's press is held in),
/// navigation, the brush-tuning drag and the layer carry — made once by
/// [`use_gestures`]. The eyedropper's holder is shared state (`PickState::holder`), and
/// is read here beside them.
///
/// The overlays make their own [`Nav`] and none of the rest.
#[derive(Clone, Copy)]
pub(crate) struct Gestures {
    state: AppState,
    landing: Landing,
    nav: Nav,
    tune: Tune,
    carry: PickMove,
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
    }
}

impl Gestures {
    /// What holds the canvas, each gesture answering for its own pointer.
    fn holder(self) -> Option<Holder> {
        let pick = (*self.state.pick.holder.peek()).map(|p| Holder::Gesture(Gesture::Pick, p));
        let holding = [
            self.nav.holder(),
            self.tune.holder(),
            self.carry.holder(),
            pick,
            self.landing.holder(),
        ];
        let mut held = holding.into_iter().flatten();
        let holder = held.next();
        debug_assert!(
            held.next().is_none(),
            "two canvas gestures hold the pointer: {holding:?}"
        );
        holder
    }

    /// Whether a press may open `gesture` now ([`route::admits`]).
    fn admits(self, gesture: Gesture) -> bool {
        route::admits(self.holder(), Opens::Gesture(gesture))
    }

    /// If `e`'s press proves the holder's release was lost ([`route::release_lost`]), put
    /// the canvas down before the press is read. Only the put-down: a release's tail can
    /// raise a dialog, and this press is about to open a stroke.
    pub(crate) fn recover_lost_release(self, e: &Event<PointerData>) {
        if route::release_lost(self.holder(), pointer_of(e), e.is_primary()) {
            self.put_down();
        }
    }

    /// Open navigation at `e` ([`Nav::begin`]). `true` means the press was taken.
    pub(crate) fn begin_nav(self, e: &Event<PointerData>) -> bool {
        // Recorded before admission is asked: a refused finger is still on the glass, and
        // the next one to land after the holder lets go has to pair with it.
        let Some(press) = self.nav.record(e) else {
            return false;
        };
        if !route::admits(self.holder(), press.opens()) {
            return false;
        }
        self.nav.claim(e, press);
        // Whatever was being drawn was the opening half of a pinch (§18.1.7), or
        // a pen stroke a middle-drag interrupted.
        self.abandon_paint();
        true
    }

    /// Open the brush-tuning drag at `e` ([`Tune::begin`]).
    pub(crate) fn begin_tune(self, e: &Event<PointerData>) -> bool {
        let taken = self.admits(Gesture::Tune) && self.tune.begin(e);
        if taken {
            self.abandon_paint();
        }
        taken
    }

    /// Open the layer carry at `e` ([`PickMove::begin`]).
    pub(crate) fn begin_carry(self, e: &Event<PointerData>) -> bool {
        let taken = self.admits(Gesture::Carry) && self.carry.begin(e);
        if taken {
            self.abandon_paint();
        }
        taken
    }

    /// Open the eyedropper's drag at `e`, sampling under the press now and under every
    /// move of its pointer after ([`picks`](Self::picks)).
    pub(crate) fn begin_pick(self, e: &Event<PointerData>) -> bool {
        if !self.admits(Gesture::Pick) {
            return false;
        }
        self.abandon_paint();
        capture_pointer(e);
        let mut holder = self.state.pick.holder;
        holder.set(Some(pointer_of(e)));
        if let Some(s) = sample(self.state, e) {
            pick_color(self.state, s.pos);
        }
        true
    }

    /// Open paint at `e` with `tool` ([`Landing::begin`]). `false` when another pointer's
    /// gesture refuses it, or the gesture declined it — a palm, or no engine yet.
    pub(crate) fn begin_paint(self, e: &Event<PointerData>, tool: Tool) -> bool {
        self.admits(Gesture::Paint) && self.landing.begin(e, tool)
    }

    /// The press is not paint: a stroke another pointer had in flight can no longer
    /// be finished by it and must leave no mark, and the hover's promise of paint is
    /// withdrawn (§18.1.10).
    fn abandon_paint(self) {
        self.landing.abandon();
        hover_gone(self.state);
    }

    /// Offer a move to the gestures above paint. `true` means the move is spoken for, and
    /// nothing below may see it; `false` leaves it to the eyedropper, paint or the hover.
    ///
    /// The order is §25.4's. Navigation first, because it records every finger's move
    /// whoever takes it (§18.1.7). Then another pointer's move under a gesture that does
    /// not yield, which is nothing's. The mode check above every gesture holding a document
    /// preview — the carry here, paint below — so a mode's arrival reaches that gesture's
    /// `abandon` before another move renews the preview.
    pub(crate) fn advance(self, e: &Event<PointerData>) -> bool {
        if self.nav.advance(e) {
            hover_gone(self.state);
            return true;
        }
        if self.ignores(e) {
            return true;
        }
        let taken = self.tune.advance(e) || self.interrupted() || self.carry.advance(e);
        if taken {
            // Not a move that paints, so nothing may promise paint under it.
            hover_gone(self.state);
        }
        taken
    }

    /// Whether `e` is another pointer's, under a gesture that does not yield
    /// ([`Moves::Ignored`]): not a hover, and not the cursor peers see.
    pub(crate) fn ignores(self, e: &Event<PointerData>) -> bool {
        route::moved(self.holder(), pointer_of(e)) == Moves::Ignored
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

    /// Whether a move of `e`'s is the eyedropper's.
    pub(crate) fn picks(self, e: &Event<PointerData>) -> bool {
        self.state
            .pick
            .holder
            .peek()
            .is_some_and(|p| p.id == e.pointer_id())
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

    /// Route a release or cancel of `e`'s pointer ([`route::released`]): [`Lift::Ended`]
    /// when the interaction is over, carrying a pinch's tap.
    pub(crate) fn release(self, e: &Event<PointerData>) -> Lift {
        // Asked before the lift is recorded, since the last finger's lift ends the pinch
        // whose rule decides it.
        let holder = self.holder();
        let fingers = self.nav.lift(e);
        route::released(holder, pointer_of(e), fingers)
    }

    /// The canvas's navigation, for the wheel.
    pub(crate) fn nav(self) -> Nav {
        self.nav
    }

    /// Put everything the canvas can have in hand down — its gestures, and the eyedropper
    /// with its loupe — and hand the canvas back.
    pub(crate) fn put_down(self) {
        self.landing.end();
        self.nav.stop();
        self.tune.stop();
        // The one whose ending the release does not finish: the layer it picked up
        // may still be a readback away, and the commit waits for it
        // (`PickMove::settle`).
        self.carry.stop();
        let state = self.state;
        // A sample already in flight is left to land: it answers a press the user made.
        let mut picker = state.pick.holder;
        if picker.peek().is_some() {
            picker.set(None);
        }
        // The swatch a held pick was showing goes with the finger that asked for it
        // (§18.1.11).
        let mut loupe = state.pick.loupe;
        if loupe.peek().is_some() {
            loupe.set(None);
        }
        let mut canvas_active = state.canvas_active;
        canvas_active.set(false);
    }
}

/// A release: put everything down ([`Gestures::put_down`]), then what only the end of an
/// interaction owes.
pub(crate) fn end_interaction(gestures: Gestures) {
    let state = gestures.state;
    // The panel stack does not come straight back: it stays out of the way until the
    // pointer reaches into its column (`AppState::panels_asleep`, §11), because coming
    // back the instant the pen lifts happens at the moment the artist looks at what they
    // just drew.
    //
    // Only where the fade was in force: an eyedropper sample or a tuning drag keeps the
    // chrome up, and sleeping the stack on the way out of those would hide the panel the
    // gesture was for. Read before the put-down clears it. Whether it sleeps at all is
    // asked inside `sleep_panels` (`layout::ChromeHiding`, §11).
    let was_faded = *state.canvas_active.peek();
    gestures.put_down();
    if was_faded {
        crate::layout::sleep_panels(state);
    }
    // A drag-preset offer brought due by a press this release is the end of (§25.8).
    // Here rather than at the press: the press that finds nothing bound goes on to
    // paint, and a modal over a live stroke would take the canvas away mid-mark. Last,
    // because it is the one thing here that puts something *up*.
    crate::drags::settle_offer(state);
}
