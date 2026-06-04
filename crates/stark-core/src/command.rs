//! `InputCommand`: raw, high-frequency user intent (DESIGN.md §4).
//!
//! Commands are deliberately distinct from [`Action`](crate::document::Action)s.
//! Many commands are ephemeral (pointer moves mid-stroke, pan/zoom, tool
//! changes) and never enter history; only committed mutations become actions.
//! The `Session` (DESIGN.md §3) interprets commands and decides what, if
//! anything, to commit.

use serde::{Deserialize, Serialize};

use crate::document::{BlendMode, BrushParams, LayerId, Tool};
use crate::geom::Vec2;

/// One pen/mouse sample in canvas space.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InputSample {
    pub pos: Vec2,
    pub pressure: f32,
    pub tilt: Vec2,
    /// Timestamp in seconds, for velocity and timelapse (DESIGN.md §8).
    pub time: f64,
}

impl InputSample {
    /// A simple full-pressure sample (e.g. mouse input or tests).
    pub fn at(pos: Vec2) -> Self {
        Self {
            pos,
            ..Default::default()
        }
    }
}

impl Default for InputSample {
    fn default() -> Self {
        Self {
            pos: Vec2::ZERO,
            pressure: 1.0,
            tilt: Vec2::ZERO,
            time: 0.0,
        }
    }
}

/// Every stateful interaction the backend accepts (GOALS §Inputs, DESIGN.md §4).
#[derive(Clone, Debug)]
pub enum InputCommand {
    // --- stroke lifecycle (high frequency) ---
    StartStroke { tool: Tool, sample: InputSample },
    StrokeTo { sample: InputSample },
    EndStroke,
    CancelStroke,

    // --- history navigation ---
    Undo,
    Redo,

    // --- session / view (never historized) ---
    SetTool(Tool),
    SetBrush(BrushParams),
    /// Pan the view by a screen-pixel drag delta.
    Pan { delta: Vec2 },
    /// Zoom by `factor`, keeping the canvas point under `anchor` (a screen-pixel
    /// position, e.g. the cursor) fixed on screen.
    Zoom { anchor: Vec2, factor: f32 },

    // --- active layer selection (session state, never historized) ---
    SetActiveLayer(LayerId),

    // --- document edits that ARE historized ---
    AddLayer { above: Option<LayerId> },
    RemoveLayer(LayerId),
    SetLayerBlend(LayerId, BlendMode),
    SetLayerOpacity(LayerId, f32),
    SetLayerVisible(LayerId, bool),
    MoveLayer { id: LayerId, above: Option<LayerId> },
}
