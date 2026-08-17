//! Showing an edit on the canvas before laying it down — the bargain every
//! continuous control in the app makes, stated once.
//!
//! A slider, a color picker, a frame handle and a transform widget all want the
//! same thing: **the value is judged by looking**, so every value the pointer
//! crosses has to reach the canvas — and only the answer belongs in the log. So
//! each of them previews per sample through a [`ViewCommand`] that logs nothing,
//! and commits once through a [`DocCommand`] when the gesture settles. One undo
//! step per adjustment, and one replicated action, rather than one per pointer
//! move (§14.6, §15.7, §16.6, §21.6, §22.4).
//!
//! # The pair is data, not control flow
//!
//! Look at the eight commands in [`ViewCommand`] that do this: every one is
//! `PreviewX(Option<T>)`, and for every one there is a commit taking exactly that
//! `T`. The preview and the commit are not two decisions a control makes; they are
//! **one fact about the control**, and [`Preview<T>`] is that fact — a pair of
//! function pointers, named once in the table below.
//!
//! Written as control flow instead, it was written out repeatedly: the commit
//! spelled at three `settle` call sites per slider, the preview spelled again in
//! the `oninput` beside them, and the *drop* — the third command, the one a
//! half-composed gesture needs when it is abandoned — spelled somewhere else
//! again, in `crate::modes`, behind a `match` on which mode was in hand. Four
//! copies of one pairing, and nothing anywhere to say they had to agree.
//!
//! # What the type rules out
//!
//! A control that previews with one command and commits with another. That is not
//! a hypothetical: it is silent when it happens, because both halves work — the
//! canvas shows the right thing under the hand, and the release lays down
//! something else, or nothing at all. Holding the two in one value means a caller
//! can only name the pair.
//!
//! It does not rule out *forgetting to settle*. What the caller still has to do is
//! wire [`Preview::settle`] to all three of the events that can end a drag, and
//! that is why the method is idempotent — see its own note.

use dioxus::prelude::*;

use crate::state::{AppState, dispatch};
use stark_engine::LayerId;
use stark_engine::command::{DocCommand, ViewCommand};
use stark_engine::document::{BlendMode, FillOp, Filter, MattePaint, TransformMap};
use stark_engine::geom::Vec2;

/// One control's pair: the command that shows `T` on the canvas without logging
/// it, and the command that lays it down.
///
/// `Copy`, and a pair of function pointers wide, so the constants below are
/// `const` and handing one to an event closure costs nothing.
#[derive(Copy, Clone)]
pub struct Preview<T: 'static> {
    /// `Some(value)` shows it; `None` drops whatever is shown.
    show: fn(Option<T>) -> ViewCommand,
    lay: fn(T) -> DocCommand,
}

impl<T: 'static> Preview<T> {
    const fn new(show: fn(Option<T>) -> ViewCommand, lay: fn(T) -> DocCommand) -> Self {
        Self { show, lay }
    }

    /// Show `value` on the canvas, logging nothing.
    pub fn show(self, state: AppState, value: T) {
        dispatch(state, (self.show)(Some(value)));
    }

    /// Drop whatever is being shown, laying nothing.
    ///
    /// What an abandoned composition needs (`crate::modes`), and what a control
    /// whose gesture ended with nothing to lay needs. Deliberately not `commit`
    /// with a neutral value: dropping the preview restores the document, while
    /// committing a neutral would be an edit that says the artist chose it.
    pub fn clear(self, state: AppState) {
        dispatch(state, (self.show)(None));
    }

    /// Lay `value` down — one logged action, one undo step.
    ///
    /// The commit supersedes the preview engine-side, so nothing has to drop it
    /// first and there is no intermediate frame showing the document without it.
    pub fn commit(self, state: AppState, value: T) {
        dispatch(state, (self.lay)(value));
    }
}

impl<T: Clone + 'static> Preview<T> {
    /// One sample of a drag: show `value`, and remember it as what a
    /// [`settle`](Self::settle) would lay.
    ///
    /// The two halves are one call because they are one act, and splitting them
    /// is how a control ends up showing one value and committing another — the
    /// pending value **is** what the canvas is showing, by construction.
    ///
    /// The value is held rather than read back off the projection at commit time,
    /// which matters and is easy to get wrong: the projection is reporting the
    /// *preview* mid-drag (§14.6), so a commit that re-read it would be reading
    /// its own echo.
    pub fn during(self, state: AppState, mut pending: Signal<Option<T>>, value: T) {
        pending.set(Some(value.clone()));
        self.show(state, value);
    }

    /// End a drag: lay down what the previews have been showing, once, and
    /// disarm.
    ///
    /// **A no-op when nothing is pending**, which is what lets a control wire this
    /// to all three of the events that can end a drag — `change`, `pointerup`,
    /// `pointercancel` — without spending an undo step per event. `change` alone
    /// is not enough, because it is not sent when a drag ends on the value it
    /// started on, which would strand a preview with no commit to supersede it.
    /// A commit to the value the document already holds is refused engine-side, so
    /// a drag that travelled out and came back logs nothing.
    pub fn settle(self, state: AppState, mut pending: Signal<Option<T>>) {
        let settled = pending.write().take();
        if let Some(value) = settled {
            self.commit(state, value);
        }
    }
}

// --- the table -------------------------------------------------------------
//
// Every previewing control in the app, as its pair. Adding one is a row here and
// nothing else; what a row cannot express is a preview with no commit, or a
// commit whose preview shows something different.

/// A layer's opacity — the Layers panel's slider (§14.6).
pub const LAYER_OPACITY: Preview<(LayerId, f32)> =
    Preview::new(ViewCommand::PreviewLayerOpacity, |(layer, opacity)| {
        DocCommand::SetLayerOpacity(layer, opacity)
    });

/// A layer's blend mode — the Bend slider, which drags a parameter *of* the mode
/// and so carries the whole mode (§18.0.4).
pub const LAYER_BLEND: Preview<(LayerId, BlendMode)> =
    Preview::new(ViewCommand::PreviewLayerBlend, |(layer, mode)| {
        DocCommand::SetLayerBlend(layer, mode)
    });

/// A filter layer's settings — every knob on the filter bar (§21.6).
pub const FILTER: Preview<(LayerId, Filter)> =
    Preview::new(ViewCommand::PreviewFilter, |(layer, filter)| {
        DocCommand::SetFilter(layer, filter)
    });

/// A frame's rect — the on-canvas handle drag (§15.7).
pub const MATTE_RECT: Preview<(LayerId, Vec2, Vec2)> =
    Preview::new(ViewCommand::PreviewMatteRect, |(layer, min, max)| {
        DocCommand::SetMatteRect(layer, min, max)
    });

/// A matte's paint — the frame bar's color pop-out, and the gradient bar's axis
/// when it is aimed at a matte (§15.4, §22.4).
pub const MATTE_PAINT: Preview<(LayerId, MattePaint)> =
    Preview::new(ViewCommand::PreviewMattePaint, |(layer, paint)| {
        DocCommand::SetMattePaint(layer, paint)
    });

/// The canvas substrate color — the Lighting panel's pop-out (§15.5).
pub const BACKGROUND: Preview<[f32; 3]> =
    Preview::new(ViewCommand::PreviewBackground, DocCommand::SetBackground);

/// A gradient fill of the selection — the gradient bar's composing axis (§22.4).
pub const FILL: Preview<(LayerId, FillOp)> =
    Preview::new(ViewCommand::PreviewFill, |(layer, op)| DocCommand::Fill {
        layer,
        op,
    });

/// The transform gesture — the widget's whole composition, committed on "Done"
/// (§16.6, §16.8, §16.9).
pub const TRANSFORM: Preview<(LayerId, TransformMap)> =
    Preview::new(ViewCommand::PreviewTransform, |(layer, map)| {
        DocCommand::Transform { layer, map }
    });

#[cfg(test)]
mod tests {
    use super::*;

    /// The claim the type makes, asked of each pair: **the command that shows a
    /// value and the command that lays it down carry the same value**.
    ///
    /// This is the failure the pairing exists to rule out, and it is worth pinning
    /// because it is silent: both halves work on their own, so a mismatched pair
    /// shows the right thing under the hand and lays something else on release.
    /// Written per pair rather than generically because what has to be compared is
    /// the *payload inside two different enums*, which only a `match` can reach.
    #[test]
    fn a_pair_shows_and_lays_the_same_value() {
        let id = LayerId(7);

        match (
            (LAYER_OPACITY.show)(Some((id, 0.25))),
            (LAYER_OPACITY.lay)((id, 0.25)),
        ) {
            (
                ViewCommand::PreviewLayerOpacity(Some(shown)),
                DocCommand::SetLayerOpacity(layer, opacity),
            ) => assert_eq!(shown, (layer, opacity)),
            _ => panic!("LAYER_OPACITY is not a pair"),
        }

        let mode = BlendMode::Drago { k: 1.5 };
        match (
            (LAYER_BLEND.show)(Some((id, mode))),
            (LAYER_BLEND.lay)((id, mode)),
        ) {
            (
                ViewCommand::PreviewLayerBlend(Some(shown)),
                DocCommand::SetLayerBlend(layer, laid),
            ) => assert_eq!(shown, (layer, laid)),
            _ => panic!("LAYER_BLEND is not a pair"),
        }

        let (min, max) = (Vec2::new(-3.0, -4.0), Vec2::new(5.0, 6.0));
        match (
            (MATTE_RECT.show)(Some((id, min, max))),
            (MATTE_RECT.lay)((id, min, max)),
        ) {
            (
                ViewCommand::PreviewMatteRect(Some(shown)),
                DocCommand::SetMatteRect(layer, lo, hi),
            ) => assert_eq!(shown, (layer, lo, hi)),
            _ => panic!("MATTE_RECT is not a pair"),
        }

        let paint = MattePaint::Solid([0.1, 0.2, 0.3]);
        match (
            (MATTE_PAINT.show)(Some((id, paint.clone()))),
            (MATTE_PAINT.lay)((id, paint)),
        ) {
            (
                ViewCommand::PreviewMattePaint(Some(shown)),
                DocCommand::SetMattePaint(layer, laid),
            ) => assert_eq!(shown, (layer, laid)),
            _ => panic!("MATTE_PAINT is not a pair"),
        }

        let rgb = [0.93, 0.91, 0.86];
        match ((BACKGROUND.show)(Some(rgb)), (BACKGROUND.lay)(rgb)) {
            (ViewCommand::PreviewBackground(Some(shown)), DocCommand::SetBackground(laid)) => {
                assert_eq!(shown, laid)
            }
            _ => panic!("BACKGROUND is not a pair"),
        }
    }

    /// Dropping a preview is the **same command** that shows it, with nothing in
    /// it — not a separate command a pair could get wrong, and not a commit of
    /// some neutral value, which would be an edit the artist never made.
    #[test]
    fn clearing_is_the_showing_command_with_nothing_in_it() {
        assert!(matches!(
            (LAYER_OPACITY.show)(None),
            ViewCommand::PreviewLayerOpacity(None)
        ));
        assert!(matches!(
            (FILTER.show)(None),
            ViewCommand::PreviewFilter(None)
        ));
        assert!(matches!((FILL.show)(None), ViewCommand::PreviewFill(None)));
        assert!(matches!(
            (TRANSFORM.show)(None),
            ViewCommand::PreviewTransform(None)
        ));
        assert!(matches!(
            (MATTE_PAINT.show)(None),
            ViewCommand::PreviewMattePaint(None)
        ));
    }
}
