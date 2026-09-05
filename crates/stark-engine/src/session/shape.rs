//! The shape gesture (§6.8): the drag a selection tool makes, and what it
//! resolves to on release — an edit to the selection, or a fill.

use super::Session;
use crate::command::Tool;
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

/// What a finished shape gesture resolves to (§6.8,
/// §18.0.4): an edit to the selection, or a fill.
///
/// The two travel together because they are one gesture with one preview — the
/// [`ShapeAction`] chosen when the drag started decides which of them the release
/// commits, and nothing downstream has to ask what tool was in hand.
#[derive(Clone, Debug, PartialEq)]
pub enum ShapeResult {
    Select(SelectionOp),
    Fill {
        /// The layer the fill lands on — the active layer **at the press**,
        /// pinned beside the frame it decides. The stroke builder pins its
        /// target the same way and for the same reason: a retarget landing
        /// mid-drag (a peer removing the layer under the hand) must not land an
        /// op converted into one layer's frame on another layer.
        layer: LayerId,
        /// In that layer's frame (§14.12) — converted where the gesture becomes
        /// an op, so the preview, the wire and the commit take one value.
        op: FillOp,
        /// The translation of the layer, pinned at the press.
        translation: stark_model::geom::IVec2,
    },
}

/// What a shape gesture's action *means* against the selection it is drawn over:
/// **Add, with nothing selected, is New** (§6.8).
///
/// The algebra says otherwise and goes on saying it — `max(1, s) = 1`, so a union
/// with the unrestricted selection *is* the unrestricted selection, and that is the
/// answer every peer would rasterize. What it is not is what anyone means by the
/// gesture: "add this region to the selection" with no selection in hand asks for a
/// selection of that region, and a mask that comes back covering everything reads as
/// the tool having done nothing at all. The Select panel's action row is one question
/// with five answers — *what does this shape do?* — and one of them being inert on a
/// fresh document is the row failing to answer it.
///
/// Resolved **here, where a gesture becomes an op**, and deliberately not in the
/// mask algebra. `SelectionMode::combine` stays the honest soft-set operation it
/// documents, `Selection::plan` keeps its four identities, and what reaches the log
/// is `Replace` — which is what the user got. So replay, undo, a save file and a peer
/// receiving the op all reproduce the picture without knowing this rule exists, and
/// no reordering of the log can make one op mean two things (§12.6).
///
/// Only `Union` has anything to answer for. Subtracting from everything is the
/// complement and intersecting with it is the shape, both of which are already what
/// the gesture reads as.
fn against_selection(action: ShapeAction, has_selection: bool) -> ShapeAction {
    match action {
        ShapeAction::Select(SelectionMode::Union) if !has_selection => {
            ShapeAction::Select(SelectionMode::Replace)
        }
        action => action,
    }
}

/// The shape gesture currently being dragged out (§6.8). Like a stroke it
/// is ephemeral: only the [`ShapeResult`] it resolves to on release is committed,
/// and the shape is derived from the drag on demand so a live preview and the
/// committed op come from exactly the same code.
pub(super) struct ShapeDrag {
    tool: Tool,
    /// What the release will do with the region — captured at the *start* of the
    /// drag, like the feather, so re-picking a chip mid-gesture cannot change what
    /// the gesture already looks like it is doing.
    ///
    /// The action as [`against_selection`] resolved it, not as the panel is set:
    /// what a gesture means depends on what it is drawn over, and the only moment
    /// that is settled is the press.
    action: ShapeAction,
    feather: f32,
    /// The color a fill will lay, taken off the brush when the drag began. Unused
    /// by a selecting gesture, which has no paint.
    color: Srgb,
    /// How strongly a **fill** lands — captured with the rest, so moving the slider
    /// mid-drag cannot change what the drag already looks like it is doing.
    ///
    /// Unused by a selecting gesture, which mints its op at full strength: how
    /// strongly a selection gates is the *whole* mask's opacity now, set after the
    /// region is drawn rather than baked into it (§6.8). The per-shape
    /// [`SelectionOp::opacity`] still stands underneath — nothing in the UI reaches
    /// it.
    opacity: f32,
    /// Where the drag started; for the marquees this is one corner of the box.
    start: Vec2,
    /// The lasso's decimated outline (empty for the marquees).
    points: Vec<Vec2>,
    /// The newest sample, so the marquees can span `start`..`current`.
    current: Vec2,
    /// The active layer at the press — a **fill**'s target, pinned with the
    /// frame below (see [`ShapeResult::Fill`]). A selecting gesture never reads
    /// either.
    layer: LayerId,
    /// The active layer's translation at the press (§14.12) — what a **fill** result is
    /// converted into. A selecting gesture never reads it: the mask lives on the
    /// canvas, whatever any layer's translation says.
    translation: stark_model::geom::IVec2,
}

impl ShapeDrag {
    fn push(&mut self, pos: Vec2) {
        self.current = pos;
        if self.tool == Tool::SelectLasso
            && self
                .points
                .last()
                .is_none_or(|q| q.distance(pos) >= LASSO_MIN_STEP)
        {
            self.points.push(pos);
        }
    }

    /// The region this drag currently encloses — `None` for a gesture that encloses
    /// nothing (a click with a marquee, a lasso too short to have an interior).
    fn to_shape(&self) -> Option<SelectionShape> {
        let shape = match self.tool {
            Tool::SelectRect => {
                let (min, max) = (self.start.min(self.current), self.start.max(self.current));
                if (max - min).min_element() <= 0.0 {
                    return None;
                }
                SelectionShape::rect_from_corners(self.start, self.current)
            }
            Tool::SelectEllipse => {
                let (min, max) = (self.start.min(self.current), self.start.max(self.current));
                if (max - min).min_element() <= 0.0 {
                    return None;
                }
                SelectionShape::ellipse_from_corners(self.start, self.current)
            }
            Tool::SelectLasso => {
                // Close the loop with the newest sample: the shape has to reach the
                // cursor mid-gesture, exactly as a stroke preview does.
                //
                // No second decimation pass: `push` already enforces `LASSO_MIN_STEP`
                // between consecutive kept points as they arrive, and the one point
                // that could fail that test — the trailing `current` — is one
                // `decimate` would put straight back under its keep-the-last-sample
                // rule. Running it here was an O(n) scan and an allocation, per
                // preview frame *and* per publish tick, that could not change a thing.
                let mut points = self.points.clone();
                if points.last().is_none_or(|q| *q != self.current) {
                    points.push(self.current);
                }
                if points.len() < 3 {
                    return None;
                }
                SelectionShape::Lasso(points)
            }
            Tool::Brush => return None,
        };
        Some(shape)
    }

    /// The action this drag currently stands for — what a release right now would
    /// commit, and equally what the preview fold draws.
    fn to_result(&self) -> Option<ShapeResult> {
        let shape = self.to_shape()?;
        Some(match self.action {
            // At full strength: the mask's own opacity is a separate, retroactive
            // number (`ActionKind::SetSelectionOpacity`, §6.8), so a shape says
            // *where* and never how strongly.
            ShapeAction::Select(mode) => {
                ShapeResult::Select(SelectionOp::new(mode, shape, self.feather))
            }
            ShapeAction::Fill => ShapeResult::Fill {
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
        self.selecting.is_some()
    }

    /// Begin a shape gesture with the session's current action, feather and brush.
    /// Any in-flight stroke or earlier gesture is abandoned.
    ///
    /// `has_selection` is the engine's to supply — whether the author already has a
    /// mask in force (`DocState::has_selection`) — because it decides what an Add
    /// gesture means; see [`against_selection`]. Off the *committed* document, which
    /// is the only selection this gesture can be adding to: the one thing that could
    /// change it mid-drag is this drag.
    /// `frame` is the active layer's frame at the press (§14.12) — read by a
    /// gesture that resolves to a *fill*; a selecting one never consults it.
    ///
    /// A non-finite `pos` starts **no drag** — [`set_cursor`](Self::set_cursor)'s
    /// filter, at the other door a canvas position enters this session through.
    /// [`ShapeDrag::to_shape`]'s degeneracy tests do not close the class behind
    /// it: for the reason [`SelectionShape::bounds`] gives, an all-NaN drag went
    /// past them as a `Rect { min: NaN, max: NaN }` and an infinite corner as an
    /// unbounded one. The model refuses both downstream, so no peer diverged —
    /// but the release had already spent an undo step on an op that changes no
    /// pixel. The press's other effects stand, the ordinal bump included: it did
    /// abandon what was in flight, and that abandonment is what the bump records.
    pub fn start_selection(
        &mut self,
        tool: Tool,
        pos: Vec2,
        has_selection: bool,
        translation: stark_model::geom::IVec2,
    ) {
        self.tool = tool;
        self.in_flight = None;
        // The press supersedes the hover it interrupted — and the window must
        // not survive the gesture, or the mark would reappear at the pre-press
        // position the moment the hand lifted (§18.1.10).
        self.hover = None;
        self.gesture_ordinal += 1;
        let [r, g, b] = self.color;
        self.selecting = pos.is_finite().then(|| ShapeDrag {
            tool,
            action: against_selection(self.shape_action, has_selection),
            feather: self.selection_feather,
            // The color is the hand's ([`color`](Self::color)) — a fill lays the
            // paint you have in hand, even while the brush held is an eraser.
            // Its *alpha* is not: that is the brush's pigment talking, and how
            // strongly a fill lands is the panel's own question (see
            // [`shape_opacity`](Self::shape_opacity)).
            color: Srgb::new([r, g, b]),
            opacity: self.shape_opacity,
            start: pos,
            points: vec![pos],
            current: pos,
            layer: self.active_layer,
            translation,
        });
    }

    /// Extend the in-flight shape gesture. A non-finite `pos` is dropped at the
    /// door for [`start_selection`](Self::start_selection)'s reason — and the
    /// lasso keeps every point it is given, so one such report would sit in the
    /// logged polyline for the rest of the gesture.
    pub fn selection_to(&mut self, pos: Vec2) {
        if !pos.is_finite() {
            return;
        }
        if let Some(drag) = self.selecting.as_mut() {
            drag.push(pos);
        }
    }

    /// What the in-flight gesture currently stands for, for live preview — the very
    /// same call [`Self::end_shape`] commits, so preview == committed.
    pub fn preview_shape(&self) -> Option<ShapeResult> {
        self.selecting.as_ref().and_then(ShapeDrag::to_result)
    }

    /// Finish the gesture, returning what to commit (`None` if it encloses nothing).
    ///
    /// The **selecting** actions are momentary: having drawn a selection, the session
    /// hands the canvas straight back to the brush. Selecting is a step *towards*
    /// painting, essentially never something you do twice in a row, so making it
    /// modal costs a deliberate switch-back on the overwhelmingly common path — and
    /// leaves a brush gesture silently redefining the selection when the user forgets.
    ///
    /// **Fill** stays armed, because that argument is about a gesture that is a step
    /// towards painting, and a fill *is* painting. Blocking in is done many times in
    /// a row, and being handed the brush back after each one would be the same cost
    /// the momentary rule exists to avoid, paid in the other direction. So the rule
    /// is one sentence rather than a special case: the tool disarms when the gesture
    /// was a step towards painting, and stays armed when the gesture was painting.
    ///
    /// A gesture that enclosed nothing (a stray click) leaves the tool armed either
    /// way, so a mis-click doesn't disarm it.
    pub fn end_shape(&mut self) -> Option<ShapeResult> {
        let result = self.selecting.take().and_then(|d| d.to_result());
        if matches!(result, Some(ShapeResult::Select(_))) {
            self.tool = Tool::Brush;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::fixtures::{press, session};

    /// The mode a marquee dragged across `(0,0)..(10,10)` resolves to.
    fn dragged(action: ShapeAction, has_selection: bool) -> SelectionMode {
        let mut s = session(action);
        s.start_selection(
            Tool::SelectRect,
            Vec2::ZERO,
            has_selection,
            stark_model::geom::IVec2::ZERO,
        );
        s.selection_to(Vec2::splat(10.0));
        match s.preview_shape().expect("the marquee encloses something") {
            ShapeResult::Select(op) => op.mode(),
            ShapeResult::Fill { .. } => panic!("a selecting action does not fill"),
        }
    }

    /// **Add, with nothing selected, is New** ([`against_selection`]) — and the
    /// gesture is logged as the New it behaved as, so nothing downstream carries the
    /// rule.
    ///
    /// Read off the *preview*, which is the same call the release commits
    /// ([`Session::end_shape`]): the resolution happens once, at the press, so the
    /// two cannot disagree about what the drag is doing.
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
        s.start_selection(
            Tool::SelectRect,
            Vec2::ZERO,
            false,
            stark_model::geom::IVec2::ZERO,
        );
        s.selection_to(Vec2::splat(10.0));
        assert!(matches!(s.preview_shape(), Some(ShapeResult::Fill { .. })));
    }

    /// A non-finite press encloses nothing rather than a region of NaN, so the
    /// release has nothing to commit — [`ShapeDrag::to_shape`]'s degeneracy
    /// tests never have to catch it, because no drag was started to run them.
    ///
    /// **Released with no move behind it**, which is the case that reached the
    /// log: a later report covers the bad axis, `min`/`max` returning the finite
    /// operand ([`SelectionShape::bounds`]), so a NaN *drag* was already refused
    /// by the degeneracy test. The second half drags anyway, for the infinite
    /// corner, which passed that test as an unbounded rect.
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
            assert_eq!(s.end_shape(), None, "{pos} alone committed an op");

            let mut s = session(ShapeAction::Select(SelectionMode::Replace));
            press(&mut s, Tool::SelectRect, pos);
            s.selection_to(Vec2::splat(10.0));
            assert_eq!(s.preview_shape(), None, "{pos} drew a region");
            assert_eq!(s.end_shape(), None, "{pos} dragged into an op");
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
        s.selection_to(Vec2::splat(10.0));
        let drawn = s.preview_shape().expect("the marquee encloses a region");
        for p in junk {
            s.selection_to(p);
        }
        assert_eq!(s.preview_shape(), Some(drawn));

        let mut s = session(ShapeAction::Select(SelectionMode::Replace));
        press(&mut s, Tool::SelectLasso, Vec2::ZERO);
        for p in [
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0),
            Vec2::new(0.0, 10.0),
        ] {
            s.selection_to(p);
        }
        let drawn = s.preview_shape().expect("the loop encloses a region");
        for p in junk {
            s.selection_to(p);
        }
        assert_eq!(s.preview_shape(), Some(drawn));
    }
}
