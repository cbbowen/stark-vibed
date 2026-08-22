//! Which whole-canvas **composing mode** is live, and how to leave one.
//!
//! Four gestures take the canvas away from the brush for the length of a
//! composition: the transform widget (§16.6), the perspective-guide edit
//! (§20.5), the gradient trace (§22.2) and the gradient fill's axis (§22.4).
//! Each mounts a full-viewport catcher, so a press aimed at the canvas lands on
//! the mode instead of on the paint.
//!
//! That catcher is a claim about **hit testing**, and hit testing is not the
//! only way a pointer reaches the canvas. A gesture already in flight has
//! *captured* its pointer, and a captured pointer's moves and its release are
//! delivered to the element that took them whatever has been stacked over it
//! since. A pen drawing while the other hand opens a transform is exactly that:
//! the stroke keeps feeding the fitter under the widget, and commits on
//! release. So "is a mode composing?" has to be a question Rust can ask, not
//! only one the stylesheet answers — and this module is where it is asked.
//!
//! # One mode, in one signal
//!
//! [`Composing`] carries what the mode is composing, and [`AppState::mode`] is
//! the one place it lives. Two modes at once is therefore not a state the app
//! declines to reach — it is a state it cannot express.
//!
//! It was four signals, one per mode, with "at most one is live" kept true by
//! every entry point remembering to call [`leave`] first. Seven call sites did,
//! and nothing caught an eighth that forgot; this module said so itself, in a
//! comment on the ladder that read them:
//!
//! > At most one is live once every entry point goes through `leave`; the order
//! > here is only what an already-broken state would report.
//!
//! There is no such order now, and no such ladder. [`enter`] is the only way in
//! and puts down whatever it finds, so the standing preference for ruling a class
//! out over enumerating its instances is paid for by the type rather than by
//! seven call sites agreeing.
//!
//! [`advance`] is the other writer, and the one thing it may not do is change
//! *which* mode is live: a drag sample replaces what the mode in hand is
//! composing, and handing it a different mode would be the very swap the enum
//! exists to prevent, so it refuses.
//!
//! [`AppState::mode`]: crate::state::AppState

use dioxus::prelude::*;

use crate::gesture::TransformUi;
use crate::state::{AppState, GradientUi, GuideEdit};

/// The composing mode in flight, and what it is composing.
///
/// Named rather than a bare `bool` because [`leave`] has to know *what* to put
/// down: each of the four holds a preview of a different kind, and dropping the
/// wrong one leaves the canvas showing a composition nothing is composing.
///
/// It carries the payload rather than pointing at a signal that holds it, which
/// is what makes "the mode in hand" and "what it is composing" one value that
/// cannot come apart — a mode live with its gesture missing, or a gesture live
/// under a different mode's catcher, are both unspellable.
#[derive(Clone, PartialEq)]
pub enum Composing {
    /// The transform widget (§16.6).
    Transform(TransformUi),
    /// A perspective guide being shaped (§20.5).
    GuideEdit(GuideEdit),
    /// The gradient library's trace, armed from its pop-out (§22.2).
    ///
    /// The one mode with no payload: what a trace composes is a path held by
    /// its own overlay, because a trace that is abandoned leaves nothing and a
    /// fresh arm should start clean either way
    /// (`panels::gradients::GradientTraceOverlay`).
    GradientTrace,
    /// The gradient fill's axis, on the shared bar (§22.4).
    GradientFill(GradientUi),
}

impl Composing {
    /// The transform in hand, if that is the mode — for the chrome that wants
    /// one particular mode's gesture rather than the fact of any.
    ///
    /// Three extractors rather than a `match` at each of the thirty-odd call
    /// sites, and they read as what those sites are asking:
    /// `modes::composing(state).and_then(Composing::transform)`.
    pub fn transform(self) -> Option<TransformUi> {
        match self {
            Composing::Transform(ui) => Some(ui),
            _ => None,
        }
    }

    /// The guide being shaped, if that is the mode (§20.5).
    pub fn guide_edit(self) -> Option<GuideEdit> {
        match self {
            Composing::GuideEdit(edit) => Some(edit),
            _ => None,
        }
    }

    /// The gradient axis being composed, if that is the mode (§22.4).
    pub fn gradient_fill(self) -> Option<GradientUi> {
        match self {
            Composing::GradientFill(ui) => Some(ui),
            _ => None,
        }
    }

    /// Whether these are the same *mode*, whatever each is composing — what
    /// [`advance`] asks, and nothing else needs.
    fn same_mode(&self, other: &Composing) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }
}

/// The mode composing right now, for chrome that must stand down while one is.
///
/// `read`, so the caller re-renders when the answer changes — this is the
/// render-time half, and [`composing_now`] is the handler-time one.
///
/// One read where this was four, which is the second thing the single signal
/// bought: the old version had to read all four rather than stopping at the
/// first that answered, because a short-circuiting chain subscribes the caller
/// only to the modes it got as far as — so chrome that had stood down for the
/// transform would not wake when the transform closed straight into a trace.
pub fn composing(state: AppState) -> Option<Composing> {
    state.mode.read().clone()
}

/// The mode composing right now, asked from an event handler.
///
/// `peek`, like [`panels::timeline::is_playing`](crate::panels::timeline::is_playing)
/// and for its reason: a handler is not a render and has nothing to re-run, so
/// subscribing from one would only widen what a pointer move wakes.
/// [`composing`] is the render-time half.
pub fn composing_now(state: AppState) -> Option<Composing> {
    state.mode.peek().clone()
}

/// Whether any mode is composing, asked from an event handler —
/// [`composing_now`] where only the fact matters.
pub fn is_composing(state: AppState) -> bool {
    state.mode.peek().is_some()
}

/// Enter `mode`, putting down whatever was composing first.
///
/// **The only way in**, which is what makes the four mutually exclusive: there
/// is no path that sets the mode without this, and no path through this that
/// skips [`leave`].
pub fn enter(state: AppState, mode: Composing) {
    leave(state);
    let mut live = state.mode;
    live.set(Some(mode));
}

/// Replace what the mode in hand is composing — what a drag sample does.
///
/// **Not an entry.** It writes only when `mode` is the same mode as the one
/// live, so it cannot swap a transform for a gradient axis behind the catcher's
/// back, and it cannot open a mode from nothing. Both of those are [`enter`]'s,
/// which drops the previous mode's preview on the way; skipping that drop is
/// precisely the bug the single signal exists to rule out, so this refuses
/// rather than silently doing it.
///
/// Silent about the refusal, deliberately: the only way to reach it is a caller
/// that has already lost track of which mode it is in, and by then a log line is
/// no use to anybody. The `debug_assert` is what says so during development.
pub fn advance(state: AppState, mode: Composing) {
    let mut live = state.mode;
    let matches = live.peek().as_ref().is_some_and(|m| m.same_mode(&mode));
    debug_assert!(
        matches,
        "advance() may only replace what the live mode is composing; use enter()"
    );
    if matches {
        live.set(Some(mode));
    }
}

/// Put the mode down **without** dropping its preview — what a "Done" that has
/// already committed needs.
///
/// [`leave`] is the abandoning path and drops the preview, which is right for it
/// and wrong here: a commit supersedes the preview engine-side, so clearing after
/// one would show the document without the thing just laid for a frame
/// (`crate::preview`). The two mode finishers that commit —
/// `panels::transform::finish` and `panels::gradient_bar::finish` — are its only
/// callers, and it exists so that they are not raw writes to
/// [`AppState::mode`](crate::state::AppState) sitting outside this module.
///
/// The parked gradient bar goes with it, exactly as [`leave`] takes it: a
/// composition that has been laid has no more use for the bar a trace set aside
/// than an abandoned one has.
pub fn leave_settled(state: AppState) {
    let mut live = state.mode;
    live.set(None);
    let mut resume = state.gradient_resume;
    if resume.peek().is_some() {
        resume.set(None);
    }
}

/// Commit the composing mode and leave — every mode bar's "Done" under one
/// name, so a chord can carry it (`Command::FinishMode`, MODAL_DESIGN.md).
/// Each arm is the same function its bar's own chip calls, deliberately: the
/// key and the chip must be one act, not two spellings that could drift.
///
/// The trace has no commit of its own — the release *is* the capture (§22.2)
/// — so its "finish" is the disarm, which hands back whatever bar it parked
/// (`gradients::set_armed`).
pub fn finish(state: AppState) {
    match composing_now(state) {
        Some(Composing::Transform(_)) => crate::panels::transform::finish(state),
        Some(Composing::GuideEdit(_)) => crate::panels::guides::end_guide_edit(state),
        Some(Composing::GradientTrace) => crate::gradients::set_armed(state, false),
        Some(Composing::GradientFill(_)) => crate::panels::gradient_bar::finish(state),
        None => {}
    }
}

/// [`leave`], as Esc means it (`Command::CancelMode`, MODAL_DESIGN.md): put
/// the composing mode down, committing nothing — except that a trace pops back
/// to the gradient bar it parked rather than dropping it too. The distinction
/// is the trace's own: it was armed *from* that bar to fetch a ramp for it
/// (`gradients::set_armed`), and a stray click already ends it exactly this
/// way — Esc is the same "never mind", so it must not take more with it than
/// the click does. Every other mode holds nothing on another mode's behalf,
/// and cancelling it is leaving.
pub fn cancel(state: AppState) {
    if matches!(composing_now(state), Some(Composing::GradientTrace)) {
        crate::gradients::set_armed(state, false);
    } else {
        leave(state);
    }
}

/// Put down whatever is composing, dropping its preview and committing nothing.
///
/// Called by [`enter`], which is what makes the modes mutually exclusive, and by
/// the two things that pull the ground out from under one: entering Timeline
/// mode (the playhead moves the committed document a preview was computed
/// against) and an undo or a redo from the keyboard.
///
/// Nothing is committed, deliberately — abandoning a composition is not the same
/// act as its "Done", and a mode that quietly laid its half-composed preview
/// because the artist reached for another tool would be the worse surprise.
///
/// **One `match` over the mode taken**, where this was four guarded blocks in a
/// row: the signal is cleared first and the preview dropped after, so the
/// dispatch inside each arm cannot run with the signal still borrowed — the same
/// care the four blocks each took for themselves, taken once.
pub fn leave(state: AppState) {
    let mut live = state.mode;
    // `take` into a local before anything below: a write guard held across the
    // dispatch each arm makes is the borrow panic `crate::state` warns about.
    let dropped = live.write().take();
    match dropped {
        Some(Composing::Transform(_)) => crate::preview::TRANSFORM.clear(state),
        // A guide is document state (§20.5), so its drag previews and commits
        // like the transform above rather than being live — and leaving has to
        // drop what a drag still in the hand was showing, or the pose the artist
        // abandoned stays on the canvas with no commit ever coming to supersede
        // it.
        Some(Composing::GuideEdit(_)) => crate::preview::GUIDE.clear(state),
        // Nothing to drop: a trace previews nothing, and its path lives in its
        // own overlay, which unmounts with the mode.
        Some(Composing::GradientTrace) => {}
        // Which preview the fill was showing depends on what it was aimed at
        // (§22.4) — and the module that composes it is the one that answers,
        // rather than a second `match` here that had to be kept in step with the
        // two over there.
        Some(Composing::GradientFill(ui)) => {
            crate::panels::gradient_bar::clear_preview(state, &ui.target)
        }
        None => {}
    }

    // And the gradient bar a trace set aside to be handed back
    // (`state::gradient_resume`). Dropped rather than resumed: this is the
    // abandoning path, and a gesture parked for a trace has no better claim on
    // the canvas than one that was composing when the transform opened. Its
    // preview went down at `suspend`, so there is nothing to clear.
    //
    // **Outside the `match`**, and it has to be: the stash is not a mode, so it
    // outlives whichever one is being put down here — including the case where
    // there is no mode at all, which is exactly what `leave` after a
    // `suspend` sees.
    let mut resume = state.gradient_resume;
    if resume.peek().is_some() {
        resume.set(None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stark_model::document::{ActionId, ActorId, GuideId};

    fn a_guide() -> Composing {
        Composing::GuideEdit(GuideEdit {
            id: GuideId(ActionId {
                lamport: 1,
                actor: ActorId(1),
            }),
            locked: [false; 3],
        })
    }

    /// An extractor answers for **its own** mode and for no other, which is the
    /// whole of what the call sites lean on: `composing(state).and_then(
    /// Composing::guide_edit)` has to be `None` while a gradient axis is being
    /// composed, or a guide bar would mount over a gradient's catcher.
    #[test]
    fn an_extractor_answers_only_for_its_own_mode() {
        assert!(a_guide().guide_edit().is_some());
        assert!(a_guide().transform().is_none());
        assert!(a_guide().gradient_fill().is_none());

        assert!(Composing::GradientTrace.guide_edit().is_none());
        assert!(Composing::GradientTrace.transform().is_none());
        assert!(Composing::GradientTrace.gradient_fill().is_none());
    }

    /// [`advance`]'s gate: same mode with a different payload passes, a
    /// different mode does not.
    ///
    /// This is the check that keeps `advance` from being a second `enter` — the
    /// one that would swap a mode without dropping the preview the old one was
    /// showing, which is the failure the single signal exists to rule out.
    #[test]
    fn same_mode_ignores_the_payload_and_nothing_else() {
        let mut moved = match a_guide() {
            Composing::GuideEdit(mut e) => {
                e.locked[1] = true;
                Composing::GuideEdit(e)
            }
            _ => unreachable!(),
        };
        assert!(
            a_guide().same_mode(&moved),
            "a lock is not a different mode"
        );
        assert!(a_guide() != moved, "and it is a different value");

        moved = Composing::GradientTrace;
        assert!(!a_guide().same_mode(&moved));
        assert!(!moved.same_mode(&a_guide()));
    }
}
