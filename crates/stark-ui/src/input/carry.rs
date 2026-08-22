//! Shift-and-drag picks up whichever layer is showing paint under the press and
//! carries it (§16.11) — the Move tool's auto-select, without the tool.
//!
//! The canvas's own gesture, for the mirror of [`Tune`](super::Tune)'s reason: it
//! moves the *painting*, which only the surface the painting is on can be
//! pointing at.
//!
//! The one gesture whose opening is not settled by the press that makes it: which
//! layer is under the pointer is a GPU readback, so the answer arrives on a
//! detached task and the drag's own signal has to be owned where that write can
//! reach it (`state::root_signal`).

use super::*;

/// How far the pointer has to travel before a layer carry **engages**, in page
/// px — the screen's own units (§16.11).
///
/// Screen px and not canvas px, because what this separates is a tap of the hand
/// from a drag of it, and neither becomes the other by zooming. That has a
/// consequence worth naming rather than discovering: in canvas terms the
/// threshold shrinks as you zoom in, so the escape hatch from a deadzone too
/// coarse for the nudge you want is the one artists already reach for to do fine
/// work — the same bargain [`ROPE_MAX_SCREEN_PX`] strikes for the smoothing
/// string.
///
/// Wider than the transform widget's 2 px jiggle snap (`panels::transform`'s
/// `SNAP_PX`), and much wider on a pen, because the two thresholds decide
/// different things. There, either answer is the same act — a shove of the
/// widget, possibly by nothing — and the snap is only there so touching it never
/// resamples. Here the two answers are *different acts*: below it the press
/// selected a layer and laid nothing down, above it the press moved the
/// painting.
///
/// **Graded by pointer type**, on [`input_resolution`]'s argument turned to
/// another purpose. A mouse rests on a desk and moves when it is pushed; a pen
/// tip flexes and a fingertip rolls, so both wander several px on their way off
/// the glass. One number for all three would be either a mouse that will not
/// nudge or a tap that moves the painting — and the tap is the side to protect,
/// because the two mistakes do not cost the same: a nudge that did not happen is
/// retried in a second, while a nudge that did leaves the painting changed and an
/// undo step to go and find.
fn carry_deadzone(e: &Event<PointerData>) -> f32 {
    match e.pointer_type().as_str() {
        "pen" | "touch" => 10.0,
        _ => 4.0,
    }
}

/// Where a layer-carry drag has got to, and what it is waiting for.
///
/// The last field is the shape of the whole gesture: its two halves arrive out
/// of order. The pointer's travel is known immediately and the layer under the
/// press is a GPU readback, so a flick can be over before the hit test answers —
/// which is why the record outlives the release, and why [`PickMove::settle`] is
/// run from both ends and does its work when the second one arrives.
#[derive(Copy, Clone)]
struct MoveDrag {
    /// Which press this is. Compared by the readback task against the record it
    /// finds when it wakes, so an answer to a press that has since been replaced
    /// is dropped rather than written into its successor — the double-click
    /// case, where the second press is made before the first one is answered.
    press: u64,
    /// Where the press landed, canvas px — what the translation is measured
    /// from, rather than from the last move: what this gesture asks for is a
    /// function of where the pointer *is*, so it cannot drift over a long drag
    /// (`Tune`'s size axis makes the same argument).
    from: Vec2,
    /// Where the pointer has got to, canvas px.
    to: Vec2,
    /// Where the press landed, page px — what [`deadzone`](Self::deadzone) is
    /// measured from.
    from_page: Vec2,
    /// How far this press has to travel to engage ([`carry_deadzone`]).
    ///
    /// Latched at the press rather than asked per move, for `TuneDrag::zoom`'s
    /// reason: which pointer opened the gesture is a fact
    /// about the whole of it, so it is read once instead of re-derived from
    /// every report — and a threshold that could move mid-drag would be one the
    /// hand could cross by standing still.
    deadzone: f32,
    /// Whether the pointer has been past the deadzone. **Latched**: a drag that
    /// wanders back to the press is still a drag, and what it then asks for is
    /// "put it back where it was", not "that was a tap after all".
    dragged: bool,
    /// What the hit test answered, or that it has not yet.
    hit: Hit,
    /// Set once the pointer is up. The gesture is over and all it is waiting for
    /// is the answer to its own press.
    released: bool,
    /// The translation the canvas is currently previewing, so a move that rounds
    /// to the same whole canvas pixel costs no dispatch — which at pointer rate
    /// is most of them, since the rounding is what makes the move exact.
    shown: Option<Vec2>,
}

/// Which layer a press landed on.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Hit {
    /// The readback has not come back yet.
    Pending,
    /// The press landed on this layer's paint.
    Layer(LayerId),
    /// Nothing the canvas is showing is under the press. The gesture stays in
    /// flight and does nothing: there is no layer to select and none to carry.
    Nothing,
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
/// It holds no transform state of its own, deliberately: it composes the same
/// `TransformMap::Affine` the widget composes and previews it through the same
/// [`preview::TRANSFORM`] pair (§16.6), so what the canvas shows mid-drag is what
/// the release will lay down, by construction.
///
/// **What the selection does to it** is worth stating here rather than only in
/// §16.11, because it is two separate facts. The *carry* respects the mask for
/// free — `ActionKind::Transform` cuts under the author's own selection already
/// (§16.1), so only the selected paint travels. The *pick* is what has to be
/// taught: a mask in force pins the press to the active layer
/// ([`pinned_layer`]), because a selection was drawn against paint the artist
/// had in mind, and a press that re-targeted would carry a different layer's
/// paint through their lasso.
///
/// [`preview::TRANSFORM`]: crate::preview::TRANSFORM
#[derive(Clone, Copy)]
pub struct PickMove {
    state: AppState,
    /// The drag in flight, or `None`.
    ///
    /// Root-owned (`state::root_signal`) rather than this component's, and that
    /// is not a lifetime nicety: the hit test's answer is written from a
    /// detached task, which lives in `ScopeId::ROOT` and may not write a signal
    /// owned by a scope it is not under.
    drag: Signal<Option<MoveDrag>>,
    /// How many presses this gesture has opened — [`MoveDrag::press`]'s source.
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
        let mut presses = self.presses;
        let press = *presses.peek() + 1;
        presses.set(press);
        // A mask in force answers the press without asking the canvas.
        let hit = pinned_layer(self.state).unwrap_or(Hit::Pending);
        let mut drag = self.drag;
        drag.set(Some(MoveDrag {
            press,
            from: s.pos,
            to: s.pos,
            from_page: page_xy(e),
            deadzone: carry_deadzone(e),
            dragged: false,
            hit,
            released: false,
            shown: None,
        }));
        if hit == Hit::Pending {
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
    /// Detached, and for [`pick_color`]'s reason turned up one notch: the answer
    /// to a press must land even though the release may already have happened.
    /// That is what [`settle`](Self::settle) is written to survive.
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
            let hit = readback.await.map_or(Hit::Nothing, Hit::Layer);
            let Some(mut in_flight) = *self.drag.peek() else {
                return;
            };
            if in_flight.press != press {
                return;
            }
            in_flight.hit = hit;
            // The press's own act, and the whole of what a tap does: the layer
            // under it becomes the selected one. Before the preview below, so
            // the panel highlight and the paint move together rather than a
            // frame apart.
            if let Hit::Layer(id) = hit {
                dispatch(self.state, PeerCommand::SetActiveLayer(id));
            }
            // Whatever travel the drag has already accumulated is owed a
            // preview now that there is a layer to show it on.
            let in_flight = self.refresh(in_flight);
            let mut drag = self.drag;
            drag.set(Some(in_flight));
            self.settle();
        });
    }

    /// Advance the carry in flight, if any. `true` means the move was this
    /// gesture's and the caller's own logic should not see it — including the
    /// moves before the hit test has answered, which are this gesture's even
    /// though they can show nothing yet.
    pub fn advance(self, e: &Event<PointerData>) -> bool {
        let Some(mut in_flight) = *self.drag.peek() else {
            return false;
        };
        // A captured pointer can deliver a move after its own release; the
        // gesture is over and only its answer is outstanding.
        if in_flight.released {
            return true;
        }
        let Some(s) = sample(self.state, e) else {
            return true;
        };
        in_flight.to = s.pos;
        in_flight.dragged |= page_xy(e).distance(in_flight.from_page) >= in_flight.deadzone;
        let in_flight = self.refresh(in_flight);
        let mut drag = self.drag;
        drag.set(Some(in_flight));
        true
    }

    /// End the carry in flight. Harmless when there is none.
    ///
    /// The release is only half of an ending here — see [`settle`](Self::settle)
    /// for the other half, and for why a flick that outruns the readback still
    /// lands.
    pub fn stop(self) {
        let Some(mut in_flight) = *self.drag.peek() else {
            return;
        };
        in_flight.released = true;
        let mut drag = self.drag;
        drag.set(Some(in_flight));
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
        let Some(in_flight) = *self.drag.peek() else {
            return;
        };
        let mut drag = self.drag;
        drag.set(None);
        if in_flight.shown.is_some() {
            crate::preview::TRANSFORM.clear(self.state);
        }
    }

    /// Bring the canvas into line with the record: show the translation it now
    /// asks for, or take a shown one down. Returns the record with
    /// [`MoveDrag::shown`] brought up to date.
    ///
    /// The one place a preview is raised, so the canvas cannot lag the gesture —
    /// `panels::transform`'s `update` makes the same bargain for the widget.
    fn refresh(self, mut in_flight: MoveDrag) -> MoveDrag {
        // Nothing to move, or nothing known to move yet. Either way the record
        // keeps accumulating travel: the answer may still be on its way, and it
        // is owed whatever the hand did while it was.
        let Hit::Layer(layer) = in_flight.hit else {
            return in_flight;
        };
        let delta = carry_delta(&in_flight);
        let want = (delta != Vec2::ZERO).then_some(delta);
        if want == in_flight.shown {
            return in_flight;
        }
        match want {
            Some(delta) => crate::preview::TRANSFORM.show(self.state, (layer, carry_map(delta))),
            None => crate::preview::TRANSFORM.clear(self.state),
        }
        in_flight.shown = want;
        in_flight
    }

    /// Finish the gesture, once **both** its halves have arrived: the pointer is
    /// up and the hit test has answered.
    ///
    /// Run from the release and from the readback, and a no-op from whichever
    /// gets there first. A short flick is over in a couple of frames and the
    /// answer to its press can easily land after it — so the release cannot be
    /// the thing that decides, and the gesture is not free to forget a press it
    /// has already captured the pointer for.
    ///
    /// A carry that asks for no translation lays nothing down. That is the tap:
    /// the same gesture stopped early, whose whole act was selecting the layer,
    /// and it must not spend an undo step saying so.
    fn settle(self) {
        let Some(in_flight) = *self.drag.peek() else {
            return;
        };
        if !in_flight.released || in_flight.hit == Hit::Pending {
            return;
        }
        let mut drag = self.drag;
        drag.set(None);
        let Hit::Layer(layer) = in_flight.hit else {
            return;
        };
        let delta = carry_delta(&in_flight);
        if delta == Vec2::ZERO {
            if in_flight.shown.is_some() {
                crate::preview::TRANSFORM.clear(self.state);
            }
            return;
        }
        // One logged action for the whole drag, superseding the preview
        // engine-side — so there is no frame showing the layer back where it
        // started (`preview::Preview::commit`).
        crate::preview::TRANSFORM.commit(self.state, (layer, carry_map(delta)));
    }
}

/// The map a carry of `delta` stands for.
///
/// One spelling, called from the preview and from the commit, so the two cannot
/// be different affines — `preview::Preview` makes exactly this argument one
/// level up, about the pair of commands, and it is worth making again about the
/// value they carry.
fn carry_map(delta: Vec2) -> TransformMap {
    TransformMap::Affine(Affine2::from_translation(delta))
}

/// The layer a press must carry because a **selection** says so, or `None` where
/// no mask is in force and the press is free to go looking (§16.11).
///
/// The carve-out is the whole of what "respect the selection" adds beyond what
/// the action already does. A mask is drawn against paint the artist is looking
/// at, on a layer they have in mind; a press that re-targeted would then cut a
/// *different* layer through their lasso — paint they never selected, moved by a
/// gesture they thought they understood. So the mask pins the layer, and picking
/// one up again is a Ctrl+D away.
///
/// A **universal** selection is not one: select-all and deselect are the same
/// state here (`DocState::with_selection` stores neither), which is the answer
/// that reads right — "everything is selected" is not a claim about a layer.
///
/// `Hit::Nothing` when the pinned layer cannot hold paint (a frame, a filter):
/// there is nothing on it to carry, and the mask still says the press may not go
/// looking elsewhere for something there is.
fn pinned_layer(state: AppState) -> Option<Hit> {
    let obs = state.obs.peek();
    let o = obs.as_ref()?;
    if !o.has_selection {
        return None;
    }
    Some(
        match o
            .layers
            .iter()
            .find(|l| l.id == o.active_layer && l.is_paintable())
        {
            Some(l) => Hit::Layer(l.id),
            None => Hit::Nothing,
        },
    )
}

/// The translation a carry asks for: **whole canvas pixels**.
///
/// Rounded, and that is the quality decision in this gesture. An affine
/// translation by a whole number of texels resamples nothing — sampling at texel
/// centres lands on texel centres, which §16.4 pins as an exactness property of
/// the transform — while a fractional one is honest bilinear and costs the
/// layer a generation of blur for a movement no eye asked for. Every other 2D
/// app moves in document pixels for the same reason, and it costs nothing in the
/// hand: at high zoom a step is many screen px, and at low zoom the pointer
/// crosses several per report.
///
/// A drag that has not passed the deadzone asks for nothing, which is what makes
/// the tap and the carry one gesture rather than two bindings.
fn carry_delta(in_flight: &MoveDrag) -> Vec2 {
    if !in_flight.dragged {
        return Vec2::ZERO;
    }
    (in_flight.to - in_flight.from).round()
}
