//! The shape gesture (§6.8): the drag a selection tool makes, and what it
//! resolves to on release — an edit to the selection, or a fill.

use super::{Hand, Session};
use crate::command::Tool;
use crate::gesture::LiveGesture;
use stark_model::Srgb;
use stark_model::document::{
    FillOp, LayerId, SelectionMode, SelectionOp, SelectionShape, ShapeAction,
};
use stark_model::geom::Vec2;

/// Minimum spacing (canvas px) between lasso vertices. The mask shader costs one
/// segment test per texel per vertex, and pointer samples arrive far denser than a
/// mask boundary can resolve, so the polyline is thinned as it is collected —
/// bounding both the rasterization cost and the size of the logged op (§6.8).
const LASSO_MIN_STEP: f32 = 2.0;

/// The region a shape gesture drags out (§6.8) — the *geometry*, which is all the
/// drag has any use for, with the lasso's collected vertices inside the one arm that
/// has any.
///
/// A [`Tool`] in its place could be [`Tool::Brush`], which drags no region at all, so
/// the drag carried an arm no press could reach and a vertex list two thirds of it
/// never filled. What a tool answers is now `Tool::outline`, asked once at the press.
#[derive(Clone, Debug, PartialEq)]
pub enum Outline {
    Rect,
    Ellipse,
    /// The decimated outline so far, oldest first (`LASSO_MIN_STEP` apart).
    Lasso(Vec<Vec2>),
}

impl Tool {
    /// The region this tool drags out, or `None` for one that lays paint instead —
    /// **the press's own dispatch** (§6.8).
    ///
    /// Answering with the region rather than with a `bool` is what leaves [`ShapeDrag`]
    /// no way to hold a tool that drags none.
    pub(crate) fn outline(self) -> Option<Outline> {
        match self {
            Tool::Brush => None,
            Tool::SelectRect => Some(Outline::Rect),
            Tool::SelectEllipse => Some(Outline::Ellipse),
            // Seeded by the press, through `push` like every later vertex.
            Tool::SelectLasso => Some(Outline::Lasso(Vec::new())),
        }
    }
}

/// What a shape gesture's action *means* against the selection it is drawn over:
/// **Add, with nothing selected, is New** (§6.8).
///
/// The algebra says otherwise — `max(1, s) = 1`, so a union with the unrestricted
/// selection *is* the unrestricted selection — but a mask that comes back covering
/// everything reads as the tool having done nothing at all.
///
/// Resolved **here, where a gesture becomes an op**, and deliberately not in the mask
/// algebra: `SelectionMode::combine` stays the honest soft-set operation it documents,
/// `Selection::plan` keeps its four identities, and what reaches the log is the
/// `Replace` the user got. So replay, undo, a save file and a peer receiving the op
/// all reproduce the picture without knowing this rule exists, and no reordering of
/// the log can make one op mean two things (§12.6).
///
/// Only `Union` has anything to answer for: subtracting from everything is the
/// complement and intersecting with it is the shape, which is what each already reads
/// as.
fn against_selection(action: ShapeAction, has_selection: bool) -> ShapeAction {
    match action {
        ShapeAction::Select(SelectionMode::Union) if !has_selection => {
            ShapeAction::Select(SelectionMode::Replace)
        }
        action => action,
    }
}

/// The shape gesture currently being dragged out (§6.8). Like a stroke it is
/// ephemeral: only the [`LiveGesture`] it resolves to on release is committed, and
/// that is derived from the drag on demand, so a live preview, the wire and the
/// committed op come from exactly the same code.
pub(super) struct ShapeDrag {
    outline: Outline,
    /// What the release will do with the region — captured at the *start* of the
    /// drag, like the feather, so re-picking a chip mid-gesture cannot change what the
    /// gesture already looks like it is doing.
    ///
    /// The action as [`against_selection`] resolved it, not as the panel is set: what
    /// a gesture means depends on what it is drawn over, and only the press settles
    /// that.
    action: ShapeAction,
    feather: f32,
    /// The color a fill will lay, taken off the brush when the drag began. Unused
    /// by a selecting gesture, which has no paint.
    color: Srgb,
    /// How strongly a **fill** lands — captured with the rest, so moving the slider
    /// mid-drag cannot change what the drag already looks like it is doing.
    ///
    /// Unused by a selecting gesture, which mints its op at full strength: how
    /// strongly a selection gates is the *whole* mask's opacity, set after the region
    /// is drawn rather than baked into it (§6.8). The per-shape
    /// [`SelectionOp::opacity`] still stands underneath; nothing in the UI reaches it.
    opacity: f32,
    /// Where the drag started; for the marquees this is one corner of the box.
    start: Vec2,
    /// The newest sample, so the marquees can span `start`..`current`.
    current: Vec2,
    /// The active layer at the press — a **fill**'s target, pinned beside the frame
    /// below, so a retarget landing mid-drag (a peer removing the layer under the
    /// hand) cannot land an op converted into one layer's frame on another layer. A
    /// selecting gesture never reads either.
    layer: LayerId,
    /// The active layer's translation at the press (§14.12) — what a **fill** result is
    /// converted into. A selecting gesture never reads it: the mask lives on the
    /// canvas, whatever any layer's translation says.
    translation: stark_model::geom::IVec2,
}

impl ShapeDrag {
    pub(super) fn push(&mut self, pos: Vec2) {
        self.current = pos;
        if let Outline::Lasso(points) = &mut self.outline
            && points
                .last()
                .is_none_or(|q| q.distance(pos) >= LASSO_MIN_STEP)
        {
            points.push(pos);
        }
    }

    /// The region this drag currently encloses — `None` for a gesture that encloses
    /// nothing (a click with a marquee, a lasso too short to have an interior).
    fn to_shape(&self) -> Option<SelectionShape> {
        // The marquees differ only in which conic the corners name, degeneracy test
        // included.
        let spanned = |of: fn(Vec2, Vec2) -> SelectionShape| {
            let (min, max) = (self.start.min(self.current), self.start.max(self.current));
            ((max - min).min_element() > 0.0).then(|| of(self.start, self.current))
        };
        match &self.outline {
            Outline::Rect => spanned(SelectionShape::rect_from_corners),
            Outline::Ellipse => spanned(SelectionShape::ellipse_from_corners),
            Outline::Lasso(points) => {
                // Close the loop with the newest sample: the shape has to reach the
                // cursor mid-gesture, exactly as a stroke preview does. No second
                // decimation pass — `push` already holds `LASSO_MIN_STEP` between
                // kept points, and the trailing `current` is the one point a pass
                // would keep anyway.
                let mut points = points.clone();
                if points.last().is_none_or(|q| *q != self.current) {
                    points.push(self.current);
                }
                (points.len() >= 3).then_some(SelectionShape::Lasso(points))
            }
        }
    }

    /// The gesture this drag currently stands for — what a release right now would
    /// commit, what the preview fold draws, and what the wire carries (§18.0.4).
    ///
    /// One gesture with one preview: the [`ShapeAction`] chosen when the drag started
    /// decides which of the two it is, so nothing downstream has to ask what tool was
    /// in hand.
    pub(super) fn to_gesture(&self) -> Option<LiveGesture> {
        let shape = self.to_shape()?;
        Some(match self.action {
            // At full strength: the mask's own opacity is a separate, retroactive
            // number (`ActionKind::SetSelectionOpacity`, §6.8), so a shape says
            // *where* and never how strongly.
            ShapeAction::Select(mode) => {
                LiveGesture::Selection(SelectionOp::new(mode, shape, self.feather))
            }
            ShapeAction::Fill => LiveGesture::Fill {
                layer: self.layer,
                // The shape is dragged on the canvas and the paint lands in the
                // layer's frame — one conversion, here, where the gesture becomes
                // an op (§14.12).
                op: FillOp::new(
                    shape.translated(-self.translation.as_vec2()),
                    self.feather,
                    self.color,
                    self.opacity,
                ),
                translation: self.translation,
            },
        })
    }
}

impl Session {
    /// Whether a shape gesture is being dragged out (§6.8).
    pub fn is_selecting(&self) -> bool {
        matches!(self.hand, Hand::Dragging(_))
    }

    /// Begin a shape gesture with the session's current action, feather and brush.
    /// Whatever the hand was doing is abandoned.
    ///
    /// `outline` is what the tool drags (`Tool::outline`), asked at the engine's
    /// press dispatch because that is where the two kinds of gesture part. `tool` is
    /// carried alongside it because the hand still has to *say* which tool it holds —
    /// that is a view setting, not a fact about the drag.
    ///
    /// `has_selection` is the engine's to supply — whether a mask is already in force
    /// (`DocState::has_selection`) — because it decides what an Add gesture means; see
    /// [`against_selection`]. Off the *committed* document, which is the only
    /// selection this gesture can be adding to. `translation` is the active layer's
    /// frame at the press (§14.12), read only by a gesture that resolves to a fill.
    ///
    /// A non-finite `pos` starts **no drag**, matching
    /// [`set_cursor`](Self::set_cursor)'s filter at the other door a canvas position
    /// enters this session through. [`ShapeDrag::to_shape`]'s degeneracy tests do not
    /// close the class behind it: for the reason [`SelectionShape::bounds`] gives, an
    /// all-NaN drag passes them as a `Rect { min: NaN, max: NaN }` and an infinite
    /// corner as an unbounded one. The press's other effects stand, the ordinal bump
    /// included — it did abandon what was in flight.
    pub fn start_selection(
        &mut self,
        tool: Tool,
        outline: Outline,
        pos: Vec2,
        has_selection: bool,
        translation: stark_model::geom::IVec2,
    ) {
        self.tool = tool;
        // The press supersedes whatever it interrupted — a stroke abandoned, and the
        // hover window, which must not survive the gesture or the mark would reappear
        // at the pre-press position the moment the hand lifted (§18.1.10).
        self.hand = Hand::Idle;
        self.gesture_ordinal += 1;
        if !pos.is_finite() {
            return;
        }
        let [r, g, b] = self.color;
        let mut drag = ShapeDrag {
            outline,
            action: against_selection(self.shape_action, has_selection),
            feather: self.selection_feather,
            // The hand's color, not the brush's: a fill lays the paint you have in
            // hand even while an eraser is held. Its alpha is the brush's pigment
            // talking, so how strongly a fill lands is `shape_opacity` instead.
            color: Srgb::new([r, g, b]),
            opacity: self.shape_opacity,
            start: pos,
            current: pos,
            layer: self.active_layer,
            translation,
        };
        // The press is the lasso's first vertex; the marquees read `start` instead.
        drag.push(pos);
        self.hand = Hand::Dragging(drag);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::fixtures::{press, session};

    /// A pointer report at `p`, through the one door a gesture takes them at.
    fn to(s: &mut Session, p: Vec2) {
        s.gesture_to(crate::command::InputSample::at(p));
    }

    /// What the drag in flight currently stands for.
    fn drawn(s: &Session) -> Option<LiveGesture> {
        s.live().map(|(gesture, _)| gesture)
    }

    /// The mode a marquee dragged across `(0,0)..(10,10)` resolves to.
    fn dragged(action: ShapeAction, has_selection: bool) -> SelectionMode {
        let mut s = session(action);
        s.start_selection(
            Tool::SelectRect,
            Outline::Rect,
            Vec2::ZERO,
            has_selection,
            stark_model::geom::IVec2::ZERO,
        );
        to(&mut s, Vec2::splat(10.0));
        match drawn(&s).expect("the marquee encloses something") {
            LiveGesture::Selection(op) => op.mode(),
            other => panic!("a selecting action does not draw {other:?}"),
        }
    }

    /// **Add, with nothing selected, is New** ([`against_selection`]) — and the
    /// gesture is logged as the New it behaved as, so nothing downstream carries the
    /// rule. Read off the *preview*, which is the same call the release commits
    /// ([`Session::end_gesture`]).
    #[test]
    fn adding_to_nothing_selects_the_region() {
        let add = ShapeAction::Select(SelectionMode::Union);
        assert_eq!(dragged(add, false), SelectionMode::Replace);
        assert_eq!(
            dragged(add, true),
            SelectionMode::Union,
            "with a mask in force there is something to add to",
        );
    }

    /// The other three are already what the gesture reads as against an empty
    /// selection — the shape, its complement, and the shape again — so only Add is
    /// touched, and Fill is not a combining question at all.
    #[test]
    fn the_other_actions_are_unchanged_by_an_empty_selection() {
        for mode in [
            SelectionMode::Replace,
            SelectionMode::Subtract,
            SelectionMode::Intersect,
        ] {
            assert_eq!(dragged(ShapeAction::Select(mode), false), mode);
        }
        let mut s = session(ShapeAction::Fill);
        press(&mut s, Tool::SelectRect, Vec2::ZERO);
        to(&mut s, Vec2::splat(10.0));
        assert!(matches!(drawn(&s), Some(LiveGesture::Fill { .. })));
    }

    /// A non-finite press encloses nothing rather than a region of NaN, so the release
    /// has nothing to commit — [`ShapeDrag::to_shape`]'s degeneracy tests never have
    /// to catch it, because no drag was started to run them.
    ///
    /// Both halves matter: released with no move behind it, and dragged. A later
    /// report covers a NaN axis (`min`/`max` return the finite operand,
    /// [`SelectionShape::bounds`]), but an infinite corner passes the degeneracy test
    /// as an unbounded rect.
    #[test]
    fn a_non_finite_press_starts_no_shape_drag() {
        for pos in [
            Vec2::splat(f32::NAN),
            Vec2::new(f32::INFINITY, 0.0),
            Vec2::new(0.0, f32::NEG_INFINITY),
        ] {
            let mut s = session(ShapeAction::Select(SelectionMode::Replace));
            press(&mut s, Tool::SelectRect, pos);
            assert!(!s.is_selecting(), "{pos} opened a drag");
            assert!(s.end_gesture().is_none(), "{pos} alone committed an op");

            let mut s = session(ShapeAction::Select(SelectionMode::Replace));
            press(&mut s, Tool::SelectRect, pos);
            to(&mut s, Vec2::splat(10.0));
            assert_eq!(drawn(&s), None, "{pos} drew a region");
            assert!(s.end_gesture().is_none(), "{pos} dragged into an op");
        }
    }

    /// And a non-finite move leaves the drag exactly as the last usable report
    /// left it. The marquee would only have carried it until the next move; the
    /// lasso keeps every point it is given, so one such report would sit in the
    /// logged polyline for the rest of the gesture.
    #[test]
    fn a_non_finite_move_does_not_reach_the_shape_drag() {
        let junk = [Vec2::splat(f32::NAN), Vec2::new(0.0, f32::INFINITY)];

        let mut s = session(ShapeAction::Select(SelectionMode::Replace));
        press(&mut s, Tool::SelectRect, Vec2::ZERO);
        to(&mut s, Vec2::splat(10.0));
        let held = drawn(&s).expect("the marquee encloses a region");
        for p in junk {
            to(&mut s, p);
        }
        assert_eq!(drawn(&s), Some(held));

        let mut s = session(ShapeAction::Select(SelectionMode::Replace));
        press(&mut s, Tool::SelectLasso, Vec2::ZERO);
        for p in [
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0),
            Vec2::new(0.0, 10.0),
        ] {
            to(&mut s, p);
        }
        let held = drawn(&s).expect("the loop encloses a region");
        for p in junk {
            to(&mut s, p);
        }
        assert_eq!(drawn(&s), Some(held));
    }
}
