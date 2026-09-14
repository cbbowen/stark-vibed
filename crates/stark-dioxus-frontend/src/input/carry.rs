//! Shift-and-drag picks up whichever layer is showing paint under the press and
//! carries it (§16.11) — the Move tool's auto-select, without the tool.
//!
//! The canvas's own gesture, for the mirror of [`Tune`]'s reason: it
//! moves the *painting*, which only the surface the painting is on can be
//! pointing at.
//!
//! What the gesture decides is `stark_ui::carry`'s: when a press engages, what it
//! previews, and when both of its halves are in. What is here is the readback that
//! answers the press, the preview on screen, and the commands. The readback lands on a
//! detached task, so the record is owned where that write can reach it
//! (`state::root_signal`).

use super::*;
use stark_engine::command::DocCommand;
use stark_model::geom::IVec2;
use stark_ui::carry::{self, Carry, Hit, Settle};
use stark_ui::route::{Gesture, Holder};

/// A carry in flight, and the preview it has on screen.
#[derive(Copy, Clone)]
struct InFlight {
    carry: Carry,
    /// The pointer that pressed, the only one whose moves carry.
    pointer: Pointer,
    /// The frame the canvas is currently previewing, so a move that rounds to the same
    /// whole canvas pixel costs no dispatch — which at pointer rate is most of them.
    shown: Option<(LayerId, IVec2)>,
}

/// The layer carry — **pick and translate** (§16.11): the press selects the
/// topmost layer showing paint where it landed, and the drag moves that layer's
/// selected paint. Which chord opens it is the drag table's row
/// (`crate::drags`, Shift+drag by default).
///
/// A hook shaped like [`Nav`], [`Tune`] and [`Paint`] and driven the same way —
/// [`begin`](Self::begin) on press, [`advance`](Self::advance) on move,
/// [`stop`](Self::stop) on release or cancel — each answering *was this event
/// mine?*.
///
/// It holds no transform state of its own: the drag is a target frame for the
/// layer, previewed and committed through the one [`preview::TRANSLATE`] pair
/// (§14.12), so what the canvas shows mid-drag is what the release will lay down,
/// by construction.
///
/// A selection pins the press to the active layer ([`carry::pinned_layer`]), and the
/// first travel past the deadzone floats the selected paint into a child layer
/// (§16.12) — so a selection drag pays the cut once, and every move after it is the
/// same cheap property write an unmasked carry makes.
///
/// [`preview::TRANSLATE`]: crate::preview::TRANSLATE
#[derive(Clone, Copy)]
pub struct PickMove {
    state: AppState,
    /// The carry in flight, or `None`.
    ///
    /// Root-owned (`state::root_signal`) rather than this component's: the hit
    /// test's answer is written from a detached task, which lives in
    /// `ScopeId::ROOT` and may not write a signal owned by a scope it is not under.
    drag: Signal<Option<InFlight>>,
    /// How many presses this gesture has opened — what an answer names its press by.
    presses: Signal<u64>,
}

impl PickMove {
    /// A hook: call unconditionally, like any `use_*`.
    pub fn use_pick_move(state: AppState) -> Self {
        Self {
            state,
            drag: crate::state::root_signal(|| None),
            presses: crate::state::root_signal(|| 0),
        }
    }

    /// The carry in the hand and the pointer holding it, if any. A released carry still
    /// waiting on its readback holds nothing: the hand has let go, and only the answer is
    /// outstanding.
    pub fn holder(self) -> Option<Holder> {
        in_hand(*self.drag.peek()).map(|d| Holder::Gesture(Gesture::Carry, d.pointer))
    }

    /// Begin the carry at `e`: capture the pointer and ask what is under it.
    /// `true` means "this press picks up a layer, it does not paint".
    ///
    /// Declines before the engine exists, where there is no view to map the
    /// press through and nothing painted to pick up. The press then falls
    /// through to the paint path, which does nothing with it for the same
    /// reason (`Tune::begin` declines identically).
    pub fn begin(self, e: &Event<PointerData>) -> bool {
        let Some(s) = sample(self.state, e) else {
            return false;
        };
        e.prevent_default();
        e.stop_propagation();
        capture_pointer(e);
        // A press over a press — a double-click racing its own readback, or a
        // release the panel never heard — replaces the record below, so whatever
        // the old one was previewing has to come down first or it is shown by
        // nobody and cleared by nothing.
        if (*self.drag.peek()).is_some_and(|d| d.shown.is_some()) {
            crate::preview::TRANSLATE.clear(self.state);
        }
        let mut presses = self.presses;
        let press = *presses.peek() + 1;
        presses.set(press);
        let pinned = self.state.obs.peek().as_ref().and_then(carry::pinned_layer);
        let deadzone = carry::deadzone(pointer_kind(e));
        let flight = InFlight {
            carry: Carry::press(press, s.pos, page_xy(e), deadzone, pinned),
            pointer: pointer_of(e),
            shown: None,
        };
        let mut drag = self.drag;
        drag.set(Some(flight));
        if flight.carry.awaits(press) {
            self.ask(press, s.pos);
        }
        true
    }

    /// Ask the engine which layer is under `at`, and record the answer.
    ///
    /// Render now and **drop the guard before awaiting**, exactly as
    /// [`pick_color`] does: the readback future owns everything it needs, so
    /// nothing holds the renderer while the browser's event loop runs the copy —
    /// which it must be free to do, since this gesture is previewing through the
    /// same engine while the copy is in flight.
    ///
    /// Detached, because the answer to a press must land even though the release
    /// may already have happened ([`Carry::settled`]).
    fn ask(self, press: u64, at: Vec2) {
        let Some(readback) = crate::state::with_engine_quiet(self.state, |r| r.pick_layer(at))
        else {
            // No engine: nothing to answer with, and no gesture to leave
            // waiting for an answer that is not coming.
            let mut drag = self.drag;
            drag.set(None);
            return;
        };
        spawn_forever(async move {
            let answered = readback.await;
            let Some(mut flight) = *self.drag.peek() else {
                return;
            };
            if !flight.carry.awaits(press) {
                return;
            }
            // The press's own act, and the whole of what a tap does: the layer
            // under it becomes the selected one. Before the preview below, so
            // the panel highlight and the paint move together rather than a
            // frame apart — and before the base is read, so it is read off the
            // committed document rather than any preview's echo.
            let hit = match answered {
                Some(id) => {
                    dispatch(self.state, PeerCommand::SetActiveLayer(id));
                    Hit::Layer {
                        id,
                        base: layer_translation(self.state, id),
                    }
                }
                None => Hit::Nothing,
            };
            flight.carry.answered(hit);
            // Whatever travel the drag has already accumulated is owed a
            // preview now that there is a layer to show it on.
            let flight = self.refresh(flight);
            let mut drag = self.drag;
            drag.set(Some(flight));
            self.settle();
        });
    }

    /// Advance the carry in flight, if any. `true` means the move was this
    /// gesture's and the caller's own logic should not see it — including the
    /// moves before the hit test has answered, which are this gesture's even
    /// though they can show nothing yet.
    ///
    /// Only the pressing pointer's moves are, and none once released: a stroke begun
    /// while the answer is outstanding would otherwise lose its first moves.
    pub fn advance(self, e: &Event<PointerData>) -> bool {
        let Some(mut flight) = moved_by(*self.drag.peek(), e.pointer_id()) else {
            return false;
        };
        let Some(s) = sample(self.state, e) else {
            return true;
        };
        flight.carry.moved(s.pos, page_xy(e));
        let flight = self.refresh(flight);
        let mut drag = self.drag;
        drag.set(Some(flight));
        true
    }

    /// End the carry in flight. Harmless when there is none.
    ///
    /// The release is only half of an ending here: a flick that outruns the
    /// readback lands when the answer does ([`settle`](Self::settle)).
    pub fn stop(self) {
        let Some(mut flight) = *self.drag.peek() else {
            return;
        };
        flight.carry.released();
        let mut drag = self.drag;
        drag.set(Some(flight));
        self.settle();
    }

    /// Drop the carry without committing — what an interruption needs.
    ///
    /// A composing mode opening under a captured pointer is the case (`modes`),
    /// and the stance is the stroke's: the canvas stopped taking this gesture
    /// the moment the mode took it, so it must leave no mark. The layer
    /// selection goes with it — the readback will find no record to write into —
    /// because the pick and the carry are one press, and abandoning a press
    /// abandons all of it.
    pub fn abandon(self) {
        let Some(flight) = *self.drag.peek() else {
            return;
        };
        let mut drag = self.drag;
        drag.set(None);
        if flight.shown.is_some() {
            // The float, if one was made, stands: it is a committed action, and
            // withdrawing it here would be the chrome undoing document state on
            // its own authority. What is dropped is the unlogged translation.
            crate::preview::TRANSLATE.clear(self.state);
        }
    }

    /// Bring the canvas into line with the carry: make the float it is due, then
    /// show the frame it asks for or take a shown one down. Returns the record with
    /// what is shown brought up to date.
    ///
    /// The one place a preview is raised, so the canvas cannot lag the gesture —
    /// `panels::transform`'s `update` makes the same bargain for the widget.
    fn refresh(self, mut flight: InFlight) -> InFlight {
        if let Some(parent) = flight.carry.float_due() {
            // One committed cut, after which the drag carries the child — whose
            // frame is read off the committed document the float just produced.
            dispatch(self.state, DocCommand::FloatSelection { layer: parent });
            let child = self
                .state
                .obs
                .peek()
                .as_ref()
                .map_or(Hit::Nothing, |o| carry::float_child(o, parent));
            flight.carry.floated(child);
        }
        let want = flight.carry.wanted();
        if want == flight.shown {
            return flight;
        }
        match want {
            Some(frame) => crate::preview::TRANSLATE.show(self.state, frame),
            None => crate::preview::TRANSLATE.clear(self.state),
        }
        flight.shown = want;
        flight
    }

    /// Finish the gesture once **both** its halves have arrived
    /// ([`Carry::settled`]). Run from the release and from the readback, and a
    /// no-op from whichever gets there first.
    fn settle(self) {
        let Some(flight) = *self.drag.peek() else {
            return;
        };
        let Some(settle) = flight.carry.settled() else {
            return;
        };
        let mut drag = self.drag;
        drag.set(None);
        match settle {
            // One logged action for the whole drag, superseding the preview
            // engine-side — so there is no frame showing the layer back where it
            // started (`preview::Preview::commit`).
            Settle::Commit { layer, to } => {
                crate::preview::TRANSLATE.commit(self.state, (layer, to));
            }
            // A tap, whose whole act was selecting the layer, must not spend an
            // undo step saying so.
            Settle::Nothing => {
                if flight.shown.is_some() {
                    crate::preview::TRANSLATE.clear(self.state);
                }
            }
        }
    }
}

/// The carry still in the hand: `drag`, unless it is released and only its answer is
/// outstanding.
fn in_hand(drag: Option<InFlight>) -> Option<InFlight> {
    drag.filter(|d| !d.carry.is_released())
}

/// The carry a move of `pointer` advances: the one in the hand, if that pointer pressed it.
fn moved_by(drag: Option<InFlight>, pointer: i32) -> Option<InFlight> {
    in_hand(drag).filter(|d| d.pointer.id == pointer)
}

/// [`carry::layer_translation`] against the projection as it stands; zero before
/// there is one.
fn layer_translation(state: AppState, id: LayerId) -> IVec2 {
    state
        .obs
        .peek()
        .as_ref()
        .map_or(IVec2::ZERO, |o| carry::layer_translation(o, id))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MOUSE: i32 = 1;
    const PEN: i32 = 2;

    fn pressed(press: u64) -> InFlight {
        InFlight {
            carry: Carry::press(press, Vec2::ZERO, Vec2::ZERO, 4.0, None),
            pointer: Pointer {
                id: MOUSE,
                kind: PointerKind::Mouse,
            },
            shown: None,
        }
    }

    #[test]
    fn a_released_carry_awaiting_its_answer_claims_no_move() {
        let press = 1;
        let mut flight = pressed(press);
        assert!(in_hand(None).is_none());
        assert!(
            moved_by(Some(flight), MOUSE).is_some(),
            "pressed, not yet answered"
        );
        flight.carry.released();
        assert!(
            flight.carry.awaits(press),
            "the hit test is still outstanding"
        );
        assert!(in_hand(Some(flight)).is_none());
        assert!(
            moved_by(Some(flight), PEN).is_none(),
            "a stroke begun meanwhile"
        );
        assert!(moved_by(Some(flight), MOUSE).is_none());
    }

    #[test]
    fn only_the_pressing_pointer_carries() {
        let flight = pressed(1);
        assert!(moved_by(Some(flight), MOUSE).is_some());
        assert!(moved_by(Some(flight), PEN).is_none());
    }
}
