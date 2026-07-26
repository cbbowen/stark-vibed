//! Session: ephemeral, non-historized state (DESIGN.md §3).
//!
//! The session holds the current tool/brush, the pan/zoom view, and the
//! in-flight stroke being dragged out. None of this is undoable — switching
//! tools or panning never creates a history step. The session interprets
//! pointer commands and, on `EndStroke`, hands the [`Engine`](crate::Engine) a
//! finished [`StrokeRecord`] to commit.

use crate::command::InputSample;
use crate::document::selection::decimate;
use crate::document::{
    BrushDynamics, BrushParams, ColorDynamics, LayerId, NoiseKind, SelectionMode, SelectionOp,
    SelectionShape, StrokeRecord, Tool,
};
use crate::geom::{Vec2, ViewTransform};
use crate::path::PathFitter;

/// Minimum spacing (canvas px) between lasso vertices. The mask shader costs one
/// segment test per texel per vertex, and pointer samples arrive far denser than a
/// mask boundary can resolve, so the polyline is thinned as it is collected —
/// bounding both the rasterization cost and the size of the logged op (DESIGN §6.8).
const LASSO_MIN_STEP: f32 = 2.0;

/// Accumulates the stroke currently being drawn.
///
/// Pointer samples are fitted to control points *as they arrive* rather than
/// buffered and re-fitted on every move (DESIGN.md §6.2): the builder holds only
/// the fitter's short working window, and each new sample costs work proportional
/// to that window instead of to the stroke so far.
struct StrokeBuilder {
    tool: Tool,
    brush: BrushParams,
    layer: LayerId,
    seed: u64,
    fitter: PathFitter,
}

/// The selection gesture currently being dragged out (DESIGN.md §6.8). Like a
/// stroke it is ephemeral: only the [`SelectionOp`] it resolves to on release is
/// committed, and the shape is derived from the drag on demand so a live preview and
/// the committed op come from exactly the same code.
struct SelectionDrag {
    tool: Tool,
    mode: SelectionMode,
    feather: f32,
    /// Where the drag started; for the marquees this is one corner of the box.
    start: Vec2,
    /// The lasso's decimated outline (empty for the marquees).
    points: Vec<Vec2>,
    /// The newest sample, so the marquees can span `start`..`current`.
    current: Vec2,
}

impl SelectionDrag {
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

    /// The op this drag currently stands for — `None` for a gesture that encloses
    /// nothing (a click with a marquee, a lasso too short to have an interior).
    fn to_op(&self) -> Option<SelectionOp> {
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
                let mut points = self.points.clone();
                if points.last().is_none_or(|q| *q != self.current) {
                    points.push(self.current);
                }
                let points = decimate(&points, LASSO_MIN_STEP);
                if points.len() < 3 {
                    return None;
                }
                SelectionShape::Lasso(points)
            }
            Tool::Brush => return None,
        };
        Some(SelectionOp::new(self.mode, shape, self.feather))
    }
}

pub struct Session {
    pub view: ViewTransform,
    pub tool: Tool,
    pub brush: BrushParams,
    pub active_layer: LayerId,
    /// How the next selection gesture combines with the selection in force (§6.8).
    pub selection_mode: SelectionMode,
    /// Edge softness (canvas px) applied by the next selection gesture.
    pub selection_feather: f32,
    in_flight: Option<StrokeBuilder>,
    selecting: Option<SelectionDrag>,
}

fn hard_round_brush_params() -> BrushParams {
    BrushParams {
        radius: 100.0,
        hardness: 0.95,
        dynamics: BrushDynamics {
            add: 0.4,
            lift: 0.6,
            deposit: 0.95,
            ..BrushDynamics::default()
        },
        color_dynamics: ColorDynamics {
            noise: NoiseKind::Simplex,
            frequency: [1.0, 1.0, 4.0],
            amplitude: [0.0, 0.1, 0.2],
        },
        ..BrushParams::default()
    }
}

impl Session {
    pub fn new(view: ViewTransform, active_layer: LayerId) -> Self {
        Self {
            view,
            tool: Tool::Brush,
            brush: hard_round_brush_params(),
            active_layer,
            selection_mode: SelectionMode::default(),
            selection_feather: 0.0,
            in_flight: None,
            selecting: None,
        }
    }

    pub fn is_stroking(&self) -> bool {
        self.in_flight.is_some()
    }

    /// Whether a selection gesture is being dragged out (DESIGN.md §6.8).
    pub fn is_selecting(&self) -> bool {
        self.selecting.is_some()
    }

    /// Begin a selection gesture with the session's current mode and feather. Any
    /// in-flight stroke or earlier gesture is abandoned.
    pub fn start_selection(&mut self, tool: Tool, pos: Vec2) {
        self.tool = tool;
        self.in_flight = None;
        self.selecting = Some(SelectionDrag {
            tool,
            mode: self.selection_mode,
            feather: self.selection_feather,
            start: pos,
            points: vec![pos],
            current: pos,
        });
    }

    /// Extend the in-flight selection gesture.
    pub fn selection_to(&mut self, pos: Vec2) {
        if let Some(drag) = self.selecting.as_mut() {
            drag.push(pos);
        }
    }

    /// The op the in-flight gesture currently stands for, for live preview — the very
    /// same call [`Self::end_selection`] commits, so preview == committed.
    pub fn preview_selection(&self) -> Option<SelectionOp> {
        self.selecting.as_ref().and_then(SelectionDrag::to_op)
    }

    /// Finish the gesture, returning the op to commit (`None` if it encloses nothing).
    pub fn end_selection(&mut self) -> Option<SelectionOp> {
        self.selecting.take().and_then(|d| d.to_op())
    }

    /// Discard the in-flight selection gesture without committing.
    pub fn cancel_selection(&mut self) {
        self.selecting = None;
    }

    /// Begin a stroke. `seed` is supplied by the engine so it can be derived
    /// deterministically (DESIGN.md §6.2). Replaces any abandoned in-flight one.
    pub fn start_stroke(&mut self, tool: Tool, sample: InputSample, seed: u64) {
        self.tool = tool;
        self.selecting = None;
        let mut fitter = PathFitter::new();
        fitter.push(sample);
        self.in_flight = Some(StrokeBuilder {
            tool,
            brush: self.brush,
            layer: self.active_layer,
            seed,
            fitter,
        });
    }

    /// Extend the in-flight stroke with another sample.
    pub fn stroke_to(&mut self, sample: InputSample) {
        if let Some(b) = self.in_flight.as_mut() {
            b.fitter.push(sample);
        }
    }

    /// Snapshot the in-flight stroke as a record without ending it, for live
    /// preview (DESIGN.md §6.2). `None` if no stroke is active.
    pub fn preview_record(&self) -> Option<StrokeRecord> {
        self.in_flight.as_ref().map(StrokeBuilder::to_record)
    }

    /// How many spans of the in-flight stroke are settled — the prefix a live
    /// preview could render once instead of repainting per pointer move
    /// (see [`PathFitter::frozen_spans`]). 0 when no stroke is active.
    pub fn frozen_spans(&self) -> usize {
        self.in_flight
            .as_ref()
            .map_or(0, |b| b.fitter.frozen_spans())
    }

    /// Finish the stroke, returning the record to commit (`None` if empty).
    pub fn end_stroke(&mut self) -> Option<StrokeRecord> {
        self.in_flight.take().map(|mut b| {
            b.fitter.finish();
            b.to_record()
        })
    }

    /// Discard the in-flight stroke without committing.
    pub fn cancel_stroke(&mut self) {
        self.in_flight = None;
        self.selecting = None;
    }
}

impl StrokeBuilder {
    fn to_record(&self) -> StrokeRecord {
        StrokeRecord {
            layer: self.layer,
            tool: self.tool,
            brush: self.brush,
            // The fitted control points (DESIGN.md §6.2). Mid-stroke this ends in
            // a provisional knot at the newest sample, so the preview reaches the
            // cursor; the same fitter produces the committed path, so live ==
            // committed.
            path: self.fitter.path(),
            seed: self.seed,
        }
    }
}
