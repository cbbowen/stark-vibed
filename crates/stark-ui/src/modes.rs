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
//! [`leave`] is the other half. Entering a mode leaves whichever was already
//! live, so "two modes composing at once" stops being a case the DOM order has
//! to arbitrate and becomes a state the app cannot reach — the standing
//! preference for ruling a class out over enumerating its instances.

use dioxus::prelude::*;

use crate::state::AppState;

/// The composing mode in flight.
///
/// Named rather than a bare `bool` because [`leave`] has to know *what* to put
/// down: each of the four holds a preview of a different kind, and dropping the
/// wrong one leaves the canvas showing a composition nothing is composing.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Composing {
    /// The transform widget (§16.6).
    Transform,
    /// A perspective guide being shaped (§20.5).
    GuideEdit,
    /// The gradient library's trace, armed from its pop-out (§22.2).
    GradientTrace,
    /// The gradient fill's axis, on the shared bar (§22.4).
    GradientFill,
}

/// The mode composing right now, for chrome that must stand down while one is.
///
/// `read`, so the caller re-renders when the answer changes — this is the
/// render-time half, and [`is_composing`] is the handler-time one.
pub fn composing(state: AppState) -> Option<Composing> {
    // All four read, rather than stopping at the first that answers: a
    // short-circuiting chain subscribes the caller only to the modes it got as
    // far as, so chrome that had stood down for the transform would not wake
    // when the transform closed straight into a trace.
    let transform = state.transform.read().is_some();
    let guide = state.guide_edit.read().is_some();
    let trace = (state.gradients.armed)();
    let fill = state.gradient_bar.read().is_some();
    // At most one is live once every entry point goes through `leave`; the
    // order here is only what an already-broken state would report.
    if transform {
        Some(Composing::Transform)
    } else if guide {
        Some(Composing::GuideEdit)
    } else if trace {
        Some(Composing::GradientTrace)
    } else if fill {
        Some(Composing::GradientFill)
    } else {
        None
    }
}

/// The mode composing right now, asked from an event handler.
///
/// `peek`, like [`panels::timeline::is_playing`](crate::panels::timeline::is_playing)
/// and for its reason: a handler is not a render and has nothing to re-run, so
/// subscribing to four signals from one would only widen what a pointer move
/// wakes. [`composing`] is the render-time half.
pub fn composing_now(state: AppState) -> Option<Composing> {
    if state.transform.peek().is_some() {
        Some(Composing::Transform)
    } else if state.guide_edit.peek().is_some() {
        Some(Composing::GuideEdit)
    } else if *state.gradients.armed.peek() {
        Some(Composing::GradientTrace)
    } else if state.gradient_bar.peek().is_some() {
        Some(Composing::GradientFill)
    } else {
        None
    }
}

/// Whether any mode is composing, asked from an event handler —
/// [`composing_now`] where only the fact matters.
pub fn is_composing(state: AppState) -> bool {
    composing_now(state).is_some()
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
        Some(Composing::Transform) => crate::panels::transform::finish(state),
        Some(Composing::GuideEdit) => crate::panels::guides::end_guide_edit(state),
        Some(Composing::GradientTrace) => crate::gradients::set_armed(state, false),
        Some(Composing::GradientFill) => crate::panels::gradient_bar::finish(state),
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
/// Called by every entry into a mode, which is what makes them mutually
/// exclusive, and by the two things that pull the ground out from under one:
/// entering Timeline mode (the playhead moves the committed document a preview
/// was computed against) and an undo or a redo from the keyboard.
///
/// Nothing is committed, deliberately — abandoning a composition is not the same
/// act as its "Done", and a mode that quietly laid its half-composed preview
/// because the artist reached for another tool would be the worse surprise.
pub fn leave(state: AppState) {
    // `take` into a local before the `if`: a write guard held in a condition
    // lives to the end of the statement, and the dispatch inside would then run
    // with this signal still borrowed.
    let mut transform = state.transform;
    let dropped = transform.write().take();
    if dropped.is_some() {
        crate::preview::TRANSFORM.clear(state);
    }

    // A guide is document state now (§20.5), so its drag previews and commits
    // like the transform above rather than being live — and leaving has to drop
    // what a drag still in the hand was showing, or the pose the artist abandoned
    // stays on the canvas with no commit ever coming to supersede it.
    let mut guide_edit = state.guide_edit;
    if guide_edit.peek().is_some() {
        crate::preview::GUIDE.clear(state);
        guide_edit.set(None);
    }

    // Written straight rather than through `gradients::set_armed`, which calls
    // *this* to make room for the arm it is performing.
    let mut armed = state.gradients.armed;
    if *armed.peek() {
        armed.set(false);
    }
    // And with it, the gradient bar a trace set aside to be handed back
    // (`state::gradient_resume`). Dropped rather than resumed: this is the
    // abandoning path, and a gesture parked for a trace has no better claim on
    // the canvas than one that was composing when the transform opened. Its
    // preview went down at `suspend`, so there is nothing to clear.
    let mut resume = state.gradient_resume;
    if resume.peek().is_some() {
        resume.set(None);
    }

    // Which preview the fill was showing depends on what it was aimed at (§22.4)
    // — and the module that composes it is the one that answers, rather than a
    // second `match` here that had to be kept in step with the two over there.
    let mut gradient_bar = state.gradient_bar;
    let abandoned = gradient_bar.write().take();
    if let Some(ui) = abandoned {
        crate::panels::gradient_bar::clear_preview(state, &ui.target);
    }
}
