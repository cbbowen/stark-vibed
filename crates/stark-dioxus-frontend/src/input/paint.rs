//! Laying a mark: [`Paint`], the stroke or marquee from press to release (the tool
//! decides which, §6.8), and [`Landing`], which holds a **finger's** press in front
//! of it until the press says what it is (§18.1.11).

use super::*;
use stark_ui::input::{DWELL, Dwell, HeldPress};
use stark_ui::route::{Gesture, Holder};

/// How often the hold watcher looks, in ms — the most a snap can lag its hold.
const DWELL_POLL_MS: i32 = 60;

/// Whether a move of `pointer` feeds the stroke `drawing` holds the pointer of.
fn drawn_by(drawing: Option<Pointer>, pointer: i32) -> bool {
    drawing.is_some_and(|d| d.id == pointer)
}

/// The canvas's **paint** gesture: a stroke or a marquee, from press to release
/// (§6.8, §6.9, §6.11).
///
/// Owns everything in flight — the hold it watches for, the shape action a modifier
/// borrowed — but not routing: whether a press is paint at all is decided at the
/// canvas, which sees every binding.
#[derive(Clone, Copy)]
pub struct Paint {
    state: AppState,
    /// The pointer drawing the gesture in flight, kept in step with the engine.
    drawing: Signal<Option<Pointer>>,
    /// The panel's shape action while a gesture's modifiers override it (§6.8), put
    /// back however the gesture ends.
    restore: Signal<Option<ShapeAction>>,
    /// The hold being watched for (§6.9): `Some` exactly while a painting gesture is
    /// in flight, and clearing it stops the watcher. Root-owned, because the watcher
    /// is a detached task.
    dwell: Signal<Option<Dwell>>,
    /// Cancelled and replaced per gesture.
    watcher: Signal<Option<Task>>,
}

impl Paint {
    /// A hook: call unconditionally, like any `use_*`.
    pub fn use_paint(state: AppState) -> Self {
        Self {
            state,
            drawing: use_signal(|| None),
            restore: use_signal(|| None),
            dwell: crate::state::root_signal(|| None),
            watcher: crate::state::root_signal(|| None),
        }
    }

    /// Open a gesture at `e` with `tool`. `false` before WebGPU init, leaving
    /// `drawing` clear so the moves after it are inert too.
    ///
    /// A finger's press comes through [`Landing`] and [`open`](Self::open) instead.
    pub fn begin(self, e: &Event<PointerData>, tool: Tool) -> bool {
        let state = self.state;
        let (Some(sample), Some(tolerance)) = (sample(state, e), input_tolerance(state, e)) else {
            return false;
        };
        // The eraser end's brush is already in force: `bind_pen` runs ahead of every
        // handler in the tree (§18.1.8).

        // Marquee modifiers may override the combine mode for this gesture only.
        let action = current_action(state);
        if tool.is_selection()
            && let Some(next) = stark_ui::selection::override_for(action, mods_of(e.modifiers()))
        {
            let mut restore = self.restore;
            restore.set(Some(action));
            dispatch(state, ViewCommand::SetShapeAction(next));
        }
        // The smoothing string in canvas px (§6.11); selection tools fit no curve.
        let rope = if tool.is_selection() {
            0.0
        } else {
            input_rope(state)
        };
        self.open(pointer_of(e), tool, &[sample], tolerance, rope, page_xy(e))
    }

    /// Open the gesture on samples already taken, oldest first: the press, then every
    /// report [`Landing`] held since (§18.1.11). `at` is where the pointer is now, in
    /// page px.
    fn open(
        self,
        pointer: Pointer,
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
                tolerance,
                rope,
            },
        );
        let mut drawing = self.drawing;
        drawing.set(Some(pointer));
        // What the hand did while the press was held; empty for a press believed on
        // landing.
        crate::state::dispatch_samples(state, since);
        // Seed the string overlay; a ropeless gesture leaves it `None`, which gates
        // off the per-move refresh.
        refresh_tow(state);
        // A hold snaps the stroke to a shape (§6.9); a marquee already is one. The
        // dwell starts where the pointer is now, not where it pressed.
        if !tool.is_selection() {
            self.watch_for_hold(at);
        }
        true
    }

    pub fn in_flight(self) -> bool {
        self.drawing.peek().is_some()
    }

    /// Feed a move's reports ([`samples`]) to the gesture in flight. `false` when the
    /// move is not the stroke's: none is in flight, or another pointer drew it — a
    /// palm [`Landing::begin`] refused still reports its moves.
    pub fn advance(self, e: &Event<PointerData>, samples: &[InputSample]) -> bool {
        if !drawn_by(*self.drawing.peek(), e.pointer_id()) {
            return false;
        }
        let state = self.state;
        // In page px, not canvas px: holding still is a fact about the hand (§6.9).
        self.track_hold(page_xy(e));
        // `dispatch_samples`, not `dispatch`: a sample changes pixels, not chrome, and
        // the full dispatch re-diffs the chrome per pointer move.
        crate::state::dispatch_samples(state, samples);
        // The string overlay tracks the tow (§6.11), gated so a ropeless brush pays
        // nothing per move.
        if state.tow.peek().is_some() {
            refresh_tow(state);
        }
        true
    }

    /// End the gesture, **committing** what it drew. Harmless when there is none.
    ///
    /// If a composing mode opened mid-gesture (`crate::modes`) this cancels instead,
    /// or a press that opened a transform and lifted without moving would commit a dot.
    pub fn end(self) {
        let composing = crate::modes::is_composing(self.state);
        self.close(if composing {
            GestureCommand::Cancel
        } else {
            GestureCommand::End
        });
    }

    /// End the gesture **committing nothing**, as navigation does to the stroke it
    /// interrupts (§18.1.7): a second finger means the first was opening a pinch, not
    /// drawing. Harmless when there is none.
    pub fn abandon(self) {
        self.close(GestureCommand::Cancel);
    }

    /// The teardown both ends share.
    fn close(self, command: GestureCommand) {
        let state = self.state;
        let mut drawing = self.drawing;
        if self.in_flight() {
            dispatch(state, command);
            drawing.set(None);
        }
        let mut restore = self.restore;
        if let Some(base) = restore.take() {
            dispatch(state, ViewCommand::SetShapeAction(base));
        }
        self.stop_watching();
        // Takes the string down (§6.11).
        refresh_tow(state);
    }

    /// Begin watching for a hold at `at`, page px (§6.9). A no-op when the assist is
    /// off.
    fn watch_for_hold(self, at: Vec2) {
        let state = self.state;
        if !state.prefs.peek().assist {
            return;
        }
        let mut dwell = self.dwell;
        dwell.set(Some(Dwell::new(at, now_seconds())));
        // `spawn_forever`: started from an event handler, it must outlive that scope.
        let task = spawn_forever(async move {
            let mut dwell = self.dwell;
            loop {
                sleep_ms(DWELL_POLL_MS).await;
                // Only the gesture ending stops this: a declined hold may be followed
                // by another.
                let Some(mut held) = *dwell.peek() else {
                    return;
                };
                if !held.take_due(now_seconds()) {
                    continue;
                }
                // Latched before dispatching, so a still pointer reports one hold.
                dwell.set(Some(held));
                dispatch(state, GestureCommand::Hold);
            }
        });
        let mut watcher = self.watcher;
        if let Some(old) = watcher.write().replace(task) {
            old.cancel();
        }
    }

    /// Report a move at `at`, page px, against the hold being watched. Written only
    /// when [`Dwell::moved`] says so.
    fn track_hold(self, at: Vec2) {
        let mut dwell = self.dwell;
        let Some(mut held) = *dwell.peek() else {
            return;
        };
        if held.moved(at, now_seconds()) {
            dwell.set(Some(held));
        }
    }

    fn stop_watching(self) {
        let mut dwell = self.dwell;
        if dwell.peek().is_some() {
            dwell.set(None);
        }
        let mut watcher = self.watcher;
        if let Some(task) = watcher.write().take() {
            task.cancel();
        }
    }
}

/// A finger's press, **held until it says what it means** (§18.1.11).
///
/// A finger's contact may open a pinch, a stroke or a hold, which
/// `stark_ui::input::HeldPress` decides. Held rather than believed, so a two-finger
/// pinch lays no stroke; every report is replayed into the stroke if one opens
/// ([`Paint::open`]). Every other pointer's press passes straight to [`Paint`].
#[derive(Clone, Copy)]
pub struct Landing {
    state: AppState,
    paint: Paint,
    /// The press being held. Root-owned, because the hold that resolves it into the
    /// eyedropper fires from a detached task.
    held: Signal<Option<Held>>,
    /// Cancelled and replaced per press.
    watcher: Signal<Option<Task>>,
    /// How many presses have been held, so a timer that outlives its press cannot
    /// fire on the next one. A pointer id would not do: the browser reuses them.
    epoch: Signal<u64>,
}

/// A press being held, and everything the stroke it may become will need.
#[derive(Clone)]
struct Held {
    /// Which press this is ([`Landing::epoch`]).
    epoch: u64,
    /// The finger holding it; another pointer's reports must not feed this stroke.
    pointer: Pointer,
    press: HeldPress,
    /// Latched at the press: a gesture measures against the view it started in.
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

    /// Take the press. `true` means it opened a stroke, or for a finger a held press.
    ///
    /// - **A finger landing while something is drawn** is refused as a palm.
    /// - **A non-finger press drops a held finger**, which would otherwise fire the
    ///   eyedropper under a resting palm mid-stroke.
    pub fn begin(self, e: &Event<PointerData>, tool: Tool) -> bool {
        if !is_finger(e) {
            self.take();
            return self.paint.begin(e, tool);
        }
        // A primary touch proves a held press lost its release (§18.1.7, see
        // `route::release_lost`); a stale record would refuse every later press.
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
        let at = page_xy(e);
        let mut epoch = self.epoch;
        let n = *epoch.peek() + 1;
        epoch.set(n);
        let mut held = self.held;
        held.set(Some(Held {
            epoch: n,
            pointer: pointer_of(e),
            press: HeldPress::new(tool, at, sample),
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

    /// The held press or the stroke in flight, and the pointer holding it.
    pub fn holder(self) -> Option<Holder> {
        let held = self.held.peek().as_ref().map(|h| h.pointer);
        held.or(*self.paint.drawing.peek())
            .map(|pointer| Holder::Gesture(Gesture::Paint, pointer))
    }

    /// Whether [`advance`](Self::advance) would take a move of `e`'s, asked first so
    /// an unclaimed move costs no coalesced-report read.
    pub fn claims(self, e: &Event<PointerData>) -> bool {
        self.held_by(e) || self.paint.in_flight()
    }

    fn held_by(self, e: &Event<PointerData>) -> bool {
        self.held
            .peek()
            .as_ref()
            .is_some_and(|h| h.pointer.id == e.pointer_id())
    }

    /// Feed a move's reports ([`samples`]) to whatever this press has become. `true`
    /// means the move was taken, including one only collected while held. Another
    /// pointer's move falls through to [`Paint::advance`].
    pub fn advance(self, e: &Event<PointerData>, samples: &[InputSample]) -> bool {
        if !self.held_by(e) {
            return self.paint.advance(e, samples);
        }
        let at = page_xy(e);
        let mut held = self.held;
        let travelled = {
            let mut w = held.write();
            let Some(h) = w.as_mut() else {
                return true;
            };
            h.press.advance(at, samples)
        };
        // Outside the borrow: opening re-enters the engine and rewrites the observable.
        if travelled {
            self.open();
        }
        true
    }

    /// Open the stroke with everything held. Harmless when nothing is held.
    fn open(self) {
        let Some(h) = self.take() else { return };
        self.paint.open(
            h.pointer,
            h.press.tool(),
            h.press.samples(),
            h.tolerance,
            h.rope,
            h.press.at(),
        );
    }

    /// Drop the held press and abandon any stroke in flight. A held press dispatches
    /// nothing, having opened no gesture, so a pinch flashes no mark (§18.1.11).
    pub fn abandon(self) {
        self.take();
        self.paint.abandon();
    }

    /// The release, **committing** what the press drew. A press still held is a tap,
    /// which opens and ends its stroke at once to leave a dot.
    pub fn end(self) {
        self.open();
        self.paint.end();
    }

    /// Take the held press, if any, and cancel its timer.
    fn take(self) -> Option<Held> {
        let mut watcher = self.watcher;
        if let Some(task) = watcher.write().take() {
            task.cancel();
        }
        let mut held = self.held;
        held.write().take()
    }

    /// Count this press down to the eyedropper (§18.1.11). One sleep, not a poll: a
    /// press that travelled has already opened a stroke and left nothing held.
    fn watch(self, epoch: u64) {
        // `spawn_forever`, as in `Paint::watch_for_hold`; every signal it touches is
        // root-owned.
        let task = spawn_forever(async move {
            sleep_ms((DWELL * 1000.0) as i32).await;
            self.hold(epoch);
        });
        let mut watcher = self.watcher;
        if let Some(old) = watcher.write().replace(task) {
            old.cancel();
        }
    }

    /// The wait is up: a press still held is the eyedropper (§18.1.11).
    fn hold(self, epoch: u64) {
        // Cleared first, so the take below does not cancel the task running this.
        let mut watcher = self.watcher;
        watcher.set(None);
        // Over a selection tool the press stays held, to become a marquee.
        let ready = self
            .held
            .peek()
            .as_ref()
            .is_some_and(|h| h.epoch == epoch && h.press.resolves_to_pick());
        if !ready {
            return;
        }
        let mut held = self.held;
        let Some(h) = held.write().take() else { return };
        let Some(last) = h.press.samples().last().copied() else {
            return;
        };
        let state = self.state;
        // Held as the chord's press holds it, so `end_interaction` puts both down the
        // same way (§18.0.2).
        let mut picker = state.pick.holder;
        picker.set(Some(h.pointer));
        // Unfade the chrome: the sample's answer is read off the Color panel.
        let mut canvas_active = state.canvas_active;
        canvas_active.set(false);
        // A finger has no cursor, so the loupe shows the answer (`PickState::loupe`).
        let mut loupe = state.pick.loupe;
        loupe.set(Some(h.press.at()));
        // No paint is coming (§18.1.10); `hover_stroke` stays gated off while the
        // sampler is down.
        clear_hover_mark(state);
        pick_color(state, last.pos);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_pointer_that_began_a_stroke_feeds_it() {
        const PEN: Pointer = Pointer {
            id: 2,
            kind: PointerKind::Pen,
        };
        const PALM: i32 = 7;
        assert!(drawn_by(Some(PEN), PEN.id));
        assert!(!drawn_by(Some(PEN), PALM), "a refused palm's move");
        assert!(!drawn_by(None, PEN.id), "no stroke in flight");
    }
}
