//! The two halves of laying a mark: the gesture itself, and the wait a finger's
//! press is held in before it is believed to be one.
//!
//! [`Paint`] is the stroke or the marquee, from the press that opens one to the
//! release that commits it — painting and selecting are the same gesture from
//! here, and the tool decides what the engine builds (§6.8).
//!
//! [`Landing`] stands in front of it for a **finger** alone (§18.1.11). A pen's
//! press is an answer and is handed straight on; a finger's is a question, held
//! until it says which of the three touch gestures it is, because a press
//! believed on landing is a press that has already painted by the time the second
//! finger of a pinch arrives. That is why the two live in one file: `Landing`
//! holds a `Paint` and pours what it held into it, and reading either alone
//! leaves out half of what a press means.

use super::*;

/// The canvas's **paint** gesture: a stroke or a marquee, from the press that
/// starts one to the release that commits it (§6.8, §6.9, §6.11).
///
/// A hook shaped like [`Nav`] and [`Tune`] and driven the same way, and it exists
/// for the reason those two do — *one thing owns one gesture*. This one was
/// spread over three places: the `drawing` flag and the stashed shape action were
/// the `Canvas` component's, the tow string and the assist watcher were
/// `AppState`'s, and the two teardown paths were free functions that took the
/// component's halves by `&mut`. [`end_interaction`] therefore had five
/// parameters and `abandon_gesture` had three, and between them they had to agree,
/// by hand, about what "in flight" means — across four call sites.
///
/// What is *not* here is deliberate. Whether this press is navigation, brush
/// tuning, an eyedropper sample or a stroke is **routing**, and routing stays at
/// the canvas: it is the one place that can see all four bindings at once and put
/// them in the order that makes space+Alt a pan and Ctrl+space a zoom. This owns
/// only what happens once the press has turned out to be paint.
///
/// The signals that stay in [`AppState`] stay there for stated reasons and are
/// unaffected: `pick.dragging` because the eyedropper's options bar reads it,
/// `tune_readout` and `tow` because sibling overlays draw them, `canvas_active`
/// because the whole chrome fades on it.
#[derive(Clone, Copy)]
pub struct Paint {
    state: AppState,
    /// Whether a gesture is in flight — the thing the three entry points below
    /// keep in step with the engine, and the whole of what used to be passed
    /// around as `&mut Signal<bool>`.
    drawing: Signal<bool>,
    /// The panel's shape action, stashed while a gesture's modifier keys override
    /// it (§6.8) and put back when the gesture ends, however it ends.
    restore: Signal<Option<ShapeAction>>,
}

impl Paint {
    /// A hook: call unconditionally, like any `use_*`.
    pub fn use_paint(state: AppState) -> Self {
        Self {
            state,
            drawing: use_signal(|| false),
            restore: use_signal(|| None),
        }
    }

    /// Open a gesture at `e` with `tool`. `false` when there is nothing to open it
    /// against — a press that arrives before WebGPU init has finished has no canvas
    /// space to land in, and leaving `drawing` clear is what keeps the moves after
    /// it inert too.
    ///
    /// A finger's press does **not** arrive here directly: [`Landing`] holds it
    /// until it has said whether it is paint at all (§18.1.11), and calls
    /// [`open`](Self::open) with everything it held when it turns out to be.
    pub fn begin(self, e: &Event<PointerData>, tool: Tool) -> bool {
        let state = self.state;
        let (Some(sample), Some(tolerance)) = (sample(state, e), input_tolerance(state, e)) else {
            return false;
        };
        // The eraser end's brush is already in force by now — it holds its slot
        // from a window-level binding that runs ahead of every handler in the tree
        // (`bind_pen`, §18.1.8). This press only has to *draw*, with whatever brush
        // the engine is holding.

        // The marquee modifiers override the *combine mode*, so they apply only
        // while the panel's action is a selecting one: under Fill there is nothing
        // to combine, and letting shift quietly turn a fill into a union-select
        // would be the worst kind of surprise (§18.0.4).
        let action = current_action(state);
        if tool.is_selection()
            && action.is_select()
            && let Some(m) = modifier_mode(e.modifiers())
        {
            let mut restore = self.restore;
            restore.set(Some(action));
            dispatch(state, ViewCommand::SetShapeAction(ShapeAction::Select(m)));
        }
        // The brush's smoothing as a canvas-px string (§6.11); zero for the
        // selection tools, which fit no curve.
        let rope = if tool.is_selection() {
            0.0
        } else {
            input_rope(state)
        };
        self.open(tool, &[sample], tolerance, rope, elem_xy(e))
    }

    /// Open the gesture on samples **already taken**: the press first, then every
    /// report since it, in the order the hand made them.
    ///
    /// Split out of [`begin`](Self::begin) rather than folded into it because a
    /// finger's press is not believed when it lands — [`Landing`] holds it, and
    /// pours what it held in here once the press turns out to be paint (§18.1.11).
    /// A pen's press comes through with a list of one, which is what it always was.
    ///
    /// `at` is where the pointer is *now*, in element (CSS) px — the frame the hold
    /// watcher measures in ([`Dwell::at`]).
    fn open(
        self,
        tool: Tool,
        samples: &[InputSample],
        tolerance: f32,
        rope: f32,
        at: Vec2,
    ) -> bool {
        let state = self.state;
        let Some((press, since)) = samples.split_first() else {
            return false;
        };
        dispatch(
            state,
            GestureCommand::Start {
                tool,
                sample: *press,
                // What this device and this zoom level actually resolve to, which
                // is what the fit prices against.
                tolerance,
                rope,
            },
        );
        let mut drawing = self.drawing;
        drawing.set(true);
        // Everything the hand did while the press was being held, oldest first —
        // so the wait cost the stroke a few milliseconds at its head and none of
        // its shape. Empty for every press that was believed as it landed.
        for s in since {
            crate::state::dispatch_sample(state, GestureCommand::To { sample: *s });
        }
        // Seed the string overlay; a ropeless gesture leaves it `None` and the
        // per-move refresh stays gated off.
        refresh_tow(state);
        // Watch for the pen being held still, which snaps the stroke to the shape
        // it resembles (§6.9). Painting only: a marquee is already an exact shape,
        // so there is nothing for a hold to improve.
        //
        // From where the pointer is rather than from where it pressed, which are
        // the same point for everything but a held press — and for that one the
        // hand has demonstrably just been moving, so the dwell starts here.
        if !tool.is_selection() {
            watch_for_hold(state, at);
        }
        true
    }

    /// Whether a gesture is in flight. What tells a finger landing on a canvas
    /// that is already being drawn on that it is a **palm** and not a gesture
    /// ([`Landing::begin`]).
    pub fn in_flight(self) -> bool {
        *self.drawing.peek()
    }

    /// Feed a move to the gesture in flight. `false` when there is none, which is
    /// what leaves the caller's cursor reporting to run.
    pub fn advance(self, e: &Event<PointerData>) -> bool {
        if !(self.drawing)() {
            return false;
        }
        let state = self.state;
        // In screen px, before the sample is mapped: whether the hand is holding
        // still is a fact about the hand (§6.9). Once the stroke has snapped this
        // stops watching and the same `To` steers the shape instead.
        pointer_moved(state, elem_xy(e));
        // Every report the browser folded into this event reaches the fitter
        // (`samples`), not just the one it chose to deliver. `dispatch_sample`, not
        // `dispatch`: a sample changes pixels, not chrome, and the full dispatch's
        // observable refresh re-diffs the chrome per pointer move. The preview fold
        // is rebuilt once per painted frame either way, so extra samples cost a fit
        // push each, not a render.
        for s in samples(state, e).unwrap_or_default() {
            crate::state::dispatch_sample(state, GestureCommand::To { sample: s });
        }
        // The string overlay tracks the tow (§6.11). Gated on its own signal so a
        // plain brush pays nothing here: only a gesture that started with a rope
        // ever reads the engine or dirties the overlay's scope per move.
        if state.tow.peek().is_some() {
            refresh_tow(state);
        }
        true
    }

    /// End the gesture, **committing** what it drew. Harmless when there is none.
    ///
    /// Unless a composing mode opened under the hand mid-gesture, in which case
    /// this release belongs to a canvas that stopped taking paint the moment the
    /// mode took it (`crate::modes`) — the canvas's own move handler abandons as
    /// soon as the pointer stirs, and this is the hand that opened a transform and
    /// then simply lifted, whose release would otherwise commit the press as a dot.
    pub fn end(self) {
        let composing = crate::modes::is_composing(self.state);
        self.close(if composing {
            GestureCommand::Cancel
        } else {
            GestureCommand::End
        });
    }

    /// End the gesture **committing nothing** — what a press that turns out to be
    /// navigation does to the stroke it interrupted (§18.1.7). Harmless when there
    /// is none.
    ///
    /// [`GestureCommand::Cancel`] rather than `End`, and that is the whole point: a
    /// second finger landing means the first was never drawing, it was opening a
    /// pinch, so navigating must leave no mark. The same applies to a middle-drag
    /// begun mid-stroke, which is the one other way [`Nav::begin`] can answer
    /// `true` with a gesture already running.
    pub fn abandon(self) {
        self.close(GestureCommand::Cancel);
    }

    /// The one teardown both ends share, so "what a finished gesture leaves behind"
    /// is stated once: the command, the shape action put back, the hold watcher
    /// stopped, the string taken down. It was two copies that had to agree.
    fn close(self, command: GestureCommand) {
        let state = self.state;
        let mut drawing = self.drawing;
        if drawing() {
            dispatch(state, command);
            drawing.set(false);
        }
        let mut restore = self.restore;
        if let Some(base) = restore.take() {
            dispatch(state, ViewCommand::SetShapeAction(base));
        }
        stop_watching(state);
        // The stroke the string belonged to is over, however it ended (§6.11).
        refresh_tow(state);
    }
}

/// A finger's press, **held until it says what it means** (§18.1.11).
///
/// A mouse or a pen says what it is the moment it lands: there is one of it, it is
/// aimed, and the only question left is which chord opened it. A finger says
/// nothing. The same contact is the opening half of a pinch, the start of a stroke
/// and the beginning of a hold, and which one it turns out to be is not knowable at
/// the press — it becomes known when a second finger lands (navigation, §18.1.7),
/// when this one travels ([`TOUCH_SLOP`]), or when it does neither for long enough
/// ([`DWELL`], the eyedropper).
///
/// So a finger's press is *held* rather than obeyed, and the three ways it resolves
/// are the three touch gestures. That is what fixes the oldest complaint about
/// painting by touch: reaching for the canvas with two fingers used to lay a stroke
/// and then take it back, because the first finger had already been believed and
/// fingers never land together. A held press has nothing to take back.
///
/// **Nothing is lost by waiting.** Every report the browser delivers while a press
/// is held is kept — coalesced list and all, so the full input rate survives — and
/// replayed into the stroke the instant it opens ([`Paint::open`]). The mark starts
/// where the finger touched down and carries every sample since; what the wait costs
/// is the first few milliseconds of *latency*, never a millimetre of the path.
///
/// A hook shaped like [`Nav`] and [`Tune`] and driven the same way, and the one
/// difference is where it sits: this stands **in front of** [`Paint`] rather than
/// beside it. Every press the canvas would have handed the paint gesture comes here
/// first, and a press that is not a finger is handed straight on — a pen is
/// unaffected by every line of this, which is the point. The pen is what serious
/// work is done with, and it has never needed to be second-guessed.
#[derive(Clone, Copy)]
pub struct Landing {
    state: AppState,
    /// The gesture a held press opens if it turns out to be paint, and the one this
    /// forwards every non-finger press to unchanged.
    paint: Paint,
    /// The press being held, or `None`.
    ///
    /// Root-owned (`state::root_signal`) rather than the component's, for
    /// [`PickMove::drag`]'s reason and not as a lifetime nicety: the hold that
    /// resolves this into the eyedropper fires from a detached task.
    held: Signal<Option<Held>>,
    /// That task, cancelled and replaced per press exactly as the assist watcher's
    /// is ([`watch_for_hold`]).
    watcher: Signal<Option<Task>>,
    /// How many presses have been held. A timer that outlives the press it was
    /// started for — the press resolved, the finger lifted, another landed — must
    /// not fire on its successor, and a counter is what rules that out rather than
    /// a pointer id, which the browser reuses.
    epoch: Signal<u64>,
}

/// A press being held, and everything the stroke it may become will need.
///
/// It carries the *fit's* parameters (`tolerance`, `rope`) as they were at the
/// press, not as they are when it opens, for the reason `tune::TuneDrag::zoom` is
/// latched: a gesture measures against the view it started in.
#[derive(Clone)]
struct Held {
    /// Which press this is ([`Landing::epoch`]).
    epoch: u64,
    /// The finger holding it. A pen or a second finger arriving is a different
    /// pointer, and its reports must not be fed into this one's stroke.
    id: i32,
    /// What the press would paint with. Read at the press like everything else
    /// here, so a tool changed by the other hand mid-hold does not retroactively
    /// change what this press was.
    tool: Tool,
    /// Where it landed and where it is now, element (CSS) px — the frame a stray is
    /// measured in, for [`Dwell::at`]'s reason: holding still is a fact about the
    /// hand, not about the canvas under it.
    from: Vec2,
    at: Vec2,
    /// The furthest it has been from `from`. Monotone, so a finger that wandered
    /// out and came back has still moved: it asked to paint, and the answer to that
    /// cannot be withdrawn by holding still afterwards.
    strayed: f32,
    /// Every canvas-space sample since the press, oldest first — the press itself,
    /// then the full coalesced list of every move ([`samples`]).
    samples: Vec<InputSample>,
    tolerance: f32,
    rope: f32,
}

impl Landing {
    /// A hook: call unconditionally, like any `use_*`.
    pub fn use_landing(state: AppState, paint: Paint) -> Self {
        Self {
            state,
            paint,
            held: crate::state::root_signal(|| None),
            watcher: crate::state::root_signal(|| None),
            epoch: crate::state::root_signal(|| 0),
        }
    }

    /// Take the press. `true` means it opened something — a stroke for a pen or a
    /// mouse, a held question for a finger.
    ///
    /// Two presses are refused outright, and both are the same hand:
    ///
    /// - **A finger arriving while something is already being drawn** is a palm.
    ///   The hand that rests on the glass beside a working pen puts one down every
    ///   time, and believing it used to end the pen's stroke and start a second one
    ///   under the heel of the hand. Refusing it is not a general palm rejection —
    ///   this file cannot see the contact's size — but it is the case that actually
    ///   happens, and the mark it used to make was the pen's.
    /// - **A pen arriving while a finger is held** is the other order of the same
    ///   hand: rest first, then draw. The pen is what the artist meant, so the held
    ///   press is dropped rather than left to fire an eyedropper under their palm
    ///   half a second into the stroke.
    pub fn begin(self, e: &Event<PointerData>, tool: Tool) -> bool {
        if !is_finger(e) {
            self.take();
            return self.paint.begin(e, tool);
        }
        // A *primary* touch is the first contact of its type, so a press still held
        // when one arrives is a finger whose release never came — `Nav::finger_down`
        // clears its own set on exactly this fact and for exactly this reason
        // (§18.1.7). One stale record here would refuse every press after it, and
        // painting by touch would stop working for the rest of the session.
        if e.is_primary() {
            self.take();
        }
        if self.paint.in_flight() || self.held.peek().is_some() {
            return false;
        }
        let state = self.state;
        let (Some(sample), Some(tolerance)) = (sample(state, e), input_tolerance(state, e)) else {
            return false;
        };
        let at = elem_xy(e);
        let mut epoch = self.epoch;
        let n = *epoch.peek() + 1;
        epoch.set(n);
        let mut held = self.held;
        held.set(Some(Held {
            epoch: n,
            id: e.pointer_id(),
            tool,
            from: at,
            at,
            strayed: 0.0,
            samples: vec![sample],
            tolerance,
            rope: if tool.is_selection() {
                0.0
            } else {
                input_rope(state)
            },
        }));
        self.watch(n);
        true
    }

    /// Feed a move to whatever this press has become. `true` means the move was
    /// taken — including the moves that are still only being *collected*, which are
    /// this gesture's even though they have changed nothing yet.
    ///
    /// A move from any other pointer falls through to the paint gesture, exactly as
    /// it would have without this in the way.
    pub fn advance(self, e: &Event<PointerData>) -> bool {
        let mine = self
            .held
            .peek()
            .as_ref()
            .is_some_and(|h| h.id == e.pointer_id());
        if !mine {
            return self.paint.advance(e);
        }
        let at = elem_xy(e);
        // Read before the record is locked, so nothing holds two signals at once.
        let more = samples(self.state, e).unwrap_or_default();
        let mut held = self.held;
        let travelled = {
            let mut w = held.write();
            let Some(h) = w.as_mut() else {
                return true;
            };
            h.at = at;
            h.strayed = h.strayed.max(at.distance(h.from));
            h.samples.extend(more);
            h.strayed > TOUCH_SLOP
        };
        // The press has asked to paint. Outside the borrow, because opening the
        // stroke re-enters the engine and rewrites the frontend's observable.
        if travelled {
            self.open();
        }
        true
    }

    /// The press turned out to be paint: open the stroke and pour everything held
    /// into it. Harmless when nothing is held.
    fn open(self) {
        let Some(h) = self.take() else { return };
        self.paint
            .open(h.tool, &h.samples, h.tolerance, h.rope, h.at);
    }

    /// Drop everything this press was going to be — what a second finger, a
    /// composing mode, or a chord that outranks paint does to it.
    ///
    /// **Nothing is dispatched for the held half.** A held press never opened a
    /// gesture, so there is no stroke to cancel and no frame to repaint, and that
    /// is the whole of why reaching for the canvas with two fingers no longer
    /// flashes a mark (§18.1.11). The paint gesture underneath is abandoned as it
    /// always was, for the press that *was* believed — a pen's, or a finger's that
    /// had already travelled.
    pub fn abandon(self) {
        self.take();
        self.paint.abandon();
    }

    /// The release, **committing** what the press drew.
    ///
    /// A press still held when the finger leaves was a **tap**, and a tap paints
    /// its dot: it opens the stroke it was going to open and ends it in the same
    /// breath, so touching the canvas leaves the mark the brush would have made.
    /// The two-finger tap never reaches here — [`Nav`] took the second finger, and
    /// the canvas abandoned this the moment it did.
    pub fn end(self) {
        self.open();
        self.paint.end();
    }

    /// Take the held press and stop the hold counting it down. `None` when there is
    /// none, which is every press that is not a finger's and every finger's press
    /// that has already resolved.
    fn take(self) -> Option<Held> {
        let mut watcher = self.watcher;
        if let Some(task) = watcher.write().take() {
            task.cancel();
        }
        let mut held = self.held;
        held.write().take()
    }

    /// Count this press down to the eyedropper (§18.1.11).
    ///
    /// One sleep rather than the assist watcher's poll ([`watch_for_hold`]), and
    /// the difference is what each is watching *for*. That one waits for the
    /// pointer to stop and has to keep looking, because it may stop at any moment
    /// and start again after. This one has exactly one moment worth checking: a
    /// press that travelled far enough to be a stroke has already opened one and
    /// left nothing here to find, so the only question is whether anything is still
    /// held when the wait is up.
    fn watch(self, epoch: u64) {
        // `spawn_forever` for `watch_for_hold`'s reason: this is started from a
        // component's event handler and must not be tied to that scope's lifetime.
        // Every signal it touches is root-owned (see `state::root_signal`).
        let task = spawn_forever(async move {
            sleep_ms((DWELL * 1000.0) as i32).await;
            self.hold(epoch);
        });
        let mut watcher = self.watcher;
        if let Some(old) = watcher.write().replace(task) {
            old.cancel();
        }
    }

    /// The wait is up. If the press is still held and still still, it was never
    /// paint at all — it was the eyedropper, asking for the color under the finger
    /// (§18.1.11).
    fn hold(self, epoch: u64) {
        // This timer is spent whatever it decides, and it must not be cancelled by
        // the take below: it is the task doing the cancelling.
        let mut watcher = self.watcher;
        watcher.set(None);
        // Still this press, and still a press the sampler is willing to take. Over
        // a selection tool it stands down for `DragAction::PickColor`'s reason —
        // there the press is *for* the marquee (§6.8) — and the press stays held
        // rather than being resolved, so it can still become one.
        let ready = self
            .held
            .peek()
            .as_ref()
            .is_some_and(|h| h.epoch == epoch && !h.tool.is_selection());
        if !ready {
            return;
        }
        let mut held = self.held;
        let Some(h) = held.write().take() else { return };
        let Some(last) = h.samples.last().copied() else {
            return;
        };
        let state = self.state;
        // From here the press is the eyedropper's, and the canvas's own move
        // handler routes it there on the flag alone — the same flag the chord
        // binding sets, so a hold and an Alt+drag are one gesture from this point
        // and `end_interaction` puts both down the same way (§18.0.2).
        let mut dragging = state.pick.dragging;
        dragging.set(true);
        // The chrome comes **back** rather than staying faded, which is the one
        // thing this differs from a stroke about. A sample's answer is read off the
        // Color panel, and a sample taken behind a hidden panel tells nobody
        // anything — `end_interaction` makes the same argument for the eyedropper's
        // chord, and `main.rs` for the tuning drag.
        let mut canvas_active = state.canvas_active;
        canvas_active.set(false);
        // Where the answer is shown, since a finger has neither a cursor nor a
        // clear view of the panel (`PickState::loupe`).
        let mut loupe = state.pick.loupe;
        loupe.set(Some(h.at));
        // The mark under the cursor is a promise of paint, and this press has just
        // stopped making one (§18.1.10). It cannot come back while the sampler is
        // down: `hover_stroke` is gated on exactly this flag.
        clear_hover_mark(state);
        pick_color(state, last.pos);
    }
}
