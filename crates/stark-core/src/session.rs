//! Session: ephemeral, non-historized state (DESIGN.md §3).
//!
//! The session holds the current tool/brush, the pan/zoom view, and the
//! in-flight stroke being dragged out. None of this is undoable — switching
//! tools or panning never creates a history step. The session interprets
//! pointer commands and, on `EndStroke`, hands the [`Engine`](crate::Engine) a
//! finished [`StrokeRecord`] to commit.

use crate::command::InputSample;
use crate::document::{BrushParams, LayerId, StrokeRecord, Tool};
use crate::geom::ViewTransform;

/// Accumulates the stroke currently being drawn.
struct StrokeBuilder {
    tool: Tool,
    brush: BrushParams,
    layer: LayerId,
    seed: u64,
    path: Vec<InputSample>,
}

pub struct Session {
    pub view: ViewTransform,
    pub tool: Tool,
    pub brush: BrushParams,
    pub active_layer: LayerId,
    in_flight: Option<StrokeBuilder>,
}

impl Session {
    pub fn new(view: ViewTransform, active_layer: LayerId) -> Self {
        Self {
            view,
            tool: Tool::Brush,
            brush: BrushParams::default(),
            active_layer,
            in_flight: None,
        }
    }

    pub fn is_stroking(&self) -> bool {
        self.in_flight.is_some()
    }

    /// Begin a stroke. `seed` is supplied by the engine so it can be derived
    /// deterministically (DESIGN.md §6.2). Replaces any abandoned in-flight one.
    pub fn start_stroke(&mut self, tool: Tool, sample: InputSample, seed: u64) {
        self.tool = tool;
        self.in_flight = Some(StrokeBuilder {
            tool,
            brush: self.brush,
            layer: self.active_layer,
            seed,
            path: vec![sample],
        });
    }

    /// Extend the in-flight stroke with another sample.
    pub fn stroke_to(&mut self, sample: InputSample) {
        if let Some(b) = self.in_flight.as_mut() {
            b.path.push(sample);
        }
    }

    /// Snapshot the in-flight stroke as a record without ending it, for live
    /// preview (DESIGN.md §6.2). `None` if no stroke is active.
    pub fn preview_record(&self) -> Option<StrokeRecord> {
        self.in_flight.as_ref().map(StrokeBuilder::to_record)
    }

    /// Finish the stroke, returning the record to commit (`None` if empty).
    pub fn end_stroke(&mut self) -> Option<StrokeRecord> {
        self.in_flight.take().map(|b| b.to_record())
    }

    /// Discard the in-flight stroke without committing.
    pub fn cancel_stroke(&mut self) {
        self.in_flight = None;
    }
}

impl StrokeBuilder {
    fn to_record(&self) -> StrokeRecord {
        StrokeRecord {
            layer: self.layer,
            tool: self.tool,
            brush: self.brush,
            // Fit the raw samples to compact spline control points (DESIGN.md
            // §6.2). Done for both preview and commit, so live == committed.
            path: crate::path::simplify(&self.path, crate::path::SIMPLIFY_TOLERANCE),
            seed: self.seed,
        }
    }
}
