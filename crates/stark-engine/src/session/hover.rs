//! The hover mark (§18.1.10): the stroke a drag begun this instant would open,
//! previewed under the resting pointer — and the window of recent reports its
//! heading is read from, which a press then takes as its run-up (§6.2).

use super::{Session, path_in_frame};
use crate::command::InputSample;
use crate::path::PathFitter;
use crate::peer::{GestureView, LiveGesture};
use stark_model::document::{ActorId, StrokeRecord};
use stark_model::geom::Vec2;
use stark_model::path::ControlPoint;

/// The hover mark's state (§18.1.10): the trailing window of recent reports —
/// the heading's estimator — the probe laid from it, and the bookkeeping that
/// keeps its renders distinct.
pub(super) struct HoverStroke {
    /// The estimator, oldest first: every entry a report the hand really made.
    /// Never rendered — the mark is where a press would *go*, not where the
    /// pointer was — but it is what the prediction is honest about.
    window: Vec<InputSample>,
    /// The fitted probe — bit-for-bit the record a gesture of its two samples
    /// would commit, from the cursor along the extrapolated heading. Rebuilt
    /// per accepted report, kept so a fold costs a clone.
    path: Vec<ControlPoint>,
    /// This update's slot in the session's gesture-ordinal space (see
    /// [`Session::gesture_ordinal`]).
    ordinal: u64,
}

/// How much motion the estimator's window may span, in multiples of the input
/// **tolerance** — the report's own positional resolution ([`Session::hover_to`]).
///
/// Denominated on the tolerance because the tolerance is what the window exists to
/// average away: this many tolerances of recent motion hold the heading to roughly
/// `atan(1/this)` of noise, whatever the device and whatever the zoom — which
/// is also why it is the engine's to derive rather than the frontend's to
/// state, unlike the reach: nothing about it is a fact about the screen.
const WINDOW_ARC_TOLERANCES: f32 = 40.0;

/// Most reports the hover window keeps ([`Session::hover_to`]).
///
/// The tolerance-derived arc ([`WINDOW_ARC_TOLERANCES`]) is the extent; this is the
/// cost ceiling behind it. Every accepted report refits the whole window, so
/// this bounds what a report can cost — and it binds only where motion is
/// dense at the tolerance (a pen crawling pitch by pitch), where the window is
/// this many *tolerances* long and the heading has long since settled.
const HOVER_WINDOW: usize = 32;

/// Prune `window` from its old end until its arc fits `arc` and its count
/// fits [`HOVER_WINDOW`] — but never below the newest **pair**, which is the
/// least history a heading can be read from at all.
fn prune_window(window: &mut Vec<InputSample>, arc: f32) {
    let mut kept: f32 = window.windows(2).map(|w| w[0].pos.distance(w[1].pos)).sum();
    let mut from = 0;
    while window.len() - from > 2 && (window.len() - from > HOVER_WINDOW || kept > arc) {
        kept -= window[from].pos.distance(window[from + 1].pos);
        from += 1;
    }
    window.drain(..from);
}

/// Fit `samples` as a finished gesture — the path `End` would adopt, which is
/// what makes anything built from it the commit's own pixels.
fn fit_finished(
    tolerance: f32,
    samples: impl IntoIterator<Item = InputSample>,
) -> Vec<ControlPoint> {
    let mut fitter = PathFitter::with_tolerance(tolerance);
    for s in samples {
        fitter.push(s);
    }
    fitter.finish();
    fitter.path()
}

/// The heading at the cursor end of the fitted trace — the chord of its final
/// span, unit length — or `None` where the trace is too short to carry one (a
/// lone report) or the chord degenerates (a trace that folds back onto its own
/// end).
fn probe_heading(trace: &[ControlPoint]) -> Option<Vec2> {
    let spans = crate::path::span_count(trace.len());
    if spans < 2 {
        return None;
    }
    let d = crate::path::span_end(trace, spans - 1) - crate::path::span_end(trace, spans - 2);
    let len = d.length();
    (len.is_finite() && len > 1e-6).then(|| d / len)
}

impl Session {
    /// Whether the hand is free to hover: nothing in flight, and a tool that
    /// paints rather than drags a shape.
    ///
    /// **Both halves of the mark read it** — the door reports enter through
    /// ([`hover_to`](Self::hover_to)) and the fold that draws it
    /// ([`hover_view`](Self::hover_view)) — because the door is not free: an
    /// accepted report bumps the gesture ordinal, and a bump mid-stroke makes
    /// every peer discard its assembly and restart (§17.5) and drops the local
    /// renderer's cached head, for a mark the fold would decline to draw.
    fn hovering(&self) -> bool {
        self.in_flight.is_none() && self.selecting.is_none() && !self.tool.is_selection()
    }

    /// Feed the hover mark one report: append it to the trailing window of
    /// recent reports, refit the window, and lay the **probe** — the stroke a
    /// drag begun this instant would open, `reach` canvas px from the cursor
    /// along the hover's extrapolated heading. The engine keeps the window, so
    /// the frontend sends one sample per move exactly as it does for a stroke.
    ///
    /// **The window is the estimator, not the mark** (§6.2). Reports are
    /// quantized to the device tolerance, so the heading of two adjacent reports
    /// snaps between the eight compass points — jitter in the *input*, which
    /// the fitter is precisely the machinery to price against detail, given
    /// the redundancy a window carries and a bare pair never did. But what is
    /// rendered is not the window's own trace — that is where the pointer
    /// *was*, which the screen already shows — it is the trace's heading
    /// carried forward from the cursor: a press is being predicted, and a
    /// press starts where the pointer is.
    ///
    /// The probe is **straight**, deliberately: continuing the trace's
    /// curvature would double down on the very quantity the tolerance makes
    /// noisiest, and "from here, this way" is the whole of what a press this
    /// instant can honestly be said to do. It is built through the fitter from
    /// two synthesized samples wearing the newest report's own channels, so it
    /// is bit-for-bit the record a real gesture of those two samples would
    /// commit — the prediction is synthesized; the rendering of it is not.
    ///
    /// `tolerance` is the frontend's statement of its input tolerance, as
    /// [`start_stroke`](Self::start_stroke) takes it — the window's own extent
    /// derives from it ([`WINDOW_ARC_TOLERANCES`]), since how much history the
    /// estimator needs is a fact about the tolerance. `reach` is how far the probe
    /// extends, in **canvas px by nature rather than by conversion**: the mark
    /// is a hypothesis about paint, paint is denominated on the canvas, and a
    /// screen-fixed length grew in canvas terms as the view zoomed out —
    /// promising more painting the less closely you looked.
    ///
    /// A report arriving while the hand is not [`hovering`](Self::hovering) is
    /// refused outright, rather than kept for a fold that would decline to draw
    /// it — see there for what accepting one costs.
    ///
    /// A non-finite report is refused at the door for
    /// [`stroke_to`](Self::stroke_to)'s reason — the window *remembers* — and
    /// a report within a tolerance of the last is dropped: it carries nothing the
    /// fit could use, and a resting pen's sub-tolerance drift would otherwise buy
    /// a whole-window refit per report. Hence the answer: whether anything
    /// changed, so a caller can skip the refold for a report that changed
    /// nothing.
    pub fn hover_to(&mut self, sample: InputSample, tolerance: f32, reach: f32) -> bool {
        if !self.hovering() || !sample.is_admissible() || !reach.is_finite() {
            return false;
        }
        if let Some(h) = self.hover.as_ref()
            && let Some(last) = h.window.last()
            && (last.pos == sample.pos || last.pos.distance(sample.pos) < tolerance)
        {
            return false;
        }
        let mut window = self.hover.take().map(|h| h.window).unwrap_or_default();
        window.push(sample);
        prune_window(&mut window, WINDOW_ARC_TOLERANCES * tolerance);
        // The estimator: the window's smoothed trace, wanted only for where it
        // is going. Finished, so the heading is the one a pen-up would have
        // settled on rather than the mid-solve one.
        let trace = fit_finished(tolerance, window.iter().copied());
        let path = match probe_heading(&trace) {
            Some(dir) => fit_finished(
                tolerance,
                [
                    sample,
                    InputSample {
                        pos: sample.pos + dir * reach,
                        ..sample
                    },
                ],
            ),
            // One report, or a trace too tangled to carry a heading: the honest
            // prediction of a press with no motion behind it is a click's — a
            // lone knot, which deposits nothing, exactly as the click would.
            None => fit_finished(tolerance, [sample]),
        };
        self.gesture_ordinal += 1;
        self.hover = Some(HoverStroke {
            path,
            window,
            ordinal: self.gesture_ordinal,
        });
        true
    }

    /// Drop the hover mark. Answers whether there was one to drop, so a caller
    /// can skip the refold — and the repaint — when there was not.
    pub fn clear_hover(&mut self) -> bool {
        self.hover.take().is_some()
    }

    /// The hover window, surrendered as a stroke's run-up (§6.2) — empty when
    /// there is none, or when it does not lead up to `press`: a window whose
    /// newest report sits farther from the press than the window's own scale
    /// (the same tolerance-derived arc that bounds it, [`WINDOW_ARC_TOLERANCES`]) is
    /// history from somewhere else — a teleported pointer, a trail gone stale —
    /// and evidence about nothing this stroke does.
    pub(super) fn take_hover_context(&mut self, press: Vec2, tolerance: f32) -> Vec<InputSample> {
        let Some(h) = self.hover.take() else {
            return Vec::new();
        };
        let near = h
            .window
            .last()
            .is_some_and(|last| last.pos.distance(press) <= WINDOW_ARC_TOLERANCES * tolerance);
        if near { h.window } else { Vec::new() }
    }

    /// Whether a hover mark is held — what the frontend peeks before spending a
    /// command (and the frame behind it) taking one down.
    pub fn hover_held(&self) -> bool {
        self.hover.is_some()
    }

    /// The hover mark as the fold wants it, authored by `actor` with `seed` —
    /// the same [`GestureView`] shape a real gesture folds as, so the renderer
    /// cannot tell them apart, and the probe renders as exactly the pixels its
    /// gesture would commit: the prediction is synthesized, its rendering is
    /// inherited.
    ///
    /// `None` unless the hand is [`hovering`](Self::hovering): a fact outranks a
    /// hypothesis and the fold holds at most one gesture per actor, so neither a
    /// stroke nor a shape drag leaves room for the mark — and a selection tool
    /// drags a shape rather than the brush. An unpaintable active layer needs no
    /// test here: the stroke renderer refuses it exactly as a commit would.
    pub fn hover_view(
        &self,
        actor: ActorId,
        seed: u64,
        translation: stark_model::geom::IVec2,
    ) -> Option<GestureView> {
        if !self.hovering() {
            return None;
        }
        let h = self.hover.as_ref()?;
        Some(GestureView {
            actor,
            gesture: LiveGesture::Stroke(StrokeRecord {
                layer: self.active_layer,
                brush: self.brush,
                // In the layer's frame, like every record (§14.12) — `frame` is
                // read at the fold because a probe has no press to pin it at.
                path: path_in_frame(h.path.clone(), translation),
                seed,
                // The probe is the whole prediction: two synthesized samples,
                // no run-up ahead of them, so the mark starts where its curve
                // does — exactly as the gesture it predicts would record it.
                start: 0.0,
                translation,
            }),
            ordinal: h.ordinal,
            // Nothing settles: both samples move with every report, so the whole
            // mark is live tail — the same answer a marquee gives.
            frozen_spans: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::Tool;
    use crate::session::fixtures::{press, session};
    use stark_model::document::ShapeAction;

    /// The tolerance the hover tests report at; positions below are spaced well
    /// clear of it, so nothing is dropped as sub-tolerance drift.
    const TOLERANCE: f32 = 1.0;

    /// One hover report, far enough from the last to be accepted.
    fn hover(s: &mut Session, at: Vec2) -> bool {
        s.hover_to(InputSample::at(at), TOLERANCE, 50.0)
    }

    /// With the hand free the mark is laid, and each accepted report takes its
    /// own ordinal — the baseline the three refusals below are measured against,
    /// so none of them can pass by the door having been shut on everything.
    #[test]
    fn a_hover_with_the_hand_free_still_lays_the_mark() {
        let mut s = session(ShapeAction::default());
        let ordinal = s.gesture_ordinal();
        assert!(hover(&mut s, Vec2::ZERO));
        assert!(hover(&mut s, Vec2::new(20.0, 0.0)));
        assert_eq!(s.gesture_ordinal(), ordinal + 2);
        assert!(s.hover_held());
        let view = s.hover_view(ActorId::SOLO, 0, stark_model::geom::IVec2::ZERO);
        assert!(view.is_some(), "the free hand's mark did not fold");
    }

    /// A hover report arriving mid-stroke is refused, ordinal untouched. The
    /// fold already declined to draw one; what it could not undo was the door's
    /// side effects — an ordinal bump is a gesture *restart* on the wire (§17.5),
    /// so a frontend that forgot to gate its moves restarted every peer per
    /// pointer report of the stroke it was drawing.
    #[test]
    fn a_hover_during_a_stroke_does_not_advance_the_gesture_ordinal() {
        let mut s = session(ShapeAction::default());
        s.start_stroke(
            Tool::Brush,
            InputSample::at(Vec2::ZERO),
            0,
            TOLERANCE,
            0.0,
            stark_model::geom::IVec2::ZERO,
        );
        let ordinal = s.gesture_ordinal();
        assert!(!hover(&mut s, Vec2::new(20.0, 0.0)));
        assert_eq!(s.gesture_ordinal(), ordinal);
        assert!(!s.hover_held());
    }

    /// Same for a shape gesture being dragged out: the other in-flight slot.
    ///
    /// The tool is put back to the brush, or this would pass on the conjunct the
    /// test below is about — a press arms both at once.
    #[test]
    fn a_hover_during_a_shape_drag_does_not_advance_the_gesture_ordinal() {
        let mut s = session(ShapeAction::default());
        press(&mut s, Tool::SelectRect, Vec2::ZERO);
        s.tool = Tool::Brush;
        let ordinal = s.gesture_ordinal();
        assert!(!hover(&mut s, Vec2::new(20.0, 0.0)));
        assert_eq!(s.gesture_ordinal(), ordinal);
        assert!(!s.hover_held());
    }

    /// A selection tool refuses the report at the door, not merely at the fold.
    /// It is the condition that holds across a whole stretch of moves rather
    /// than a gesture, so it is where the waste piled up: two full solves and a
    /// spent ordinal per pointer move, building a window the fold declined every
    /// time.
    #[test]
    fn a_selection_tool_refuses_the_hover_report_not_just_the_fold() {
        let mut s = session(ShapeAction::default());
        s.tool = Tool::SelectRect;
        let ordinal = s.gesture_ordinal();
        assert!(!hover(&mut s, Vec2::ZERO));
        assert!(!hover(&mut s, Vec2::new(20.0, 0.0)));
        assert_eq!(s.gesture_ordinal(), ordinal);
        assert!(!s.hover_held());
    }
}
