//! The stored form of a stroke's path (§6.2) — the knots, not the fitter.
//!
//! A stroke travels and is saved as a short list of [`ControlPoint`]s. Fitting
//! pointer reports down to them, and flattening them back into the segments the
//! brush sweeps along, is `stark-engine`'s `path` — live work on a gesture in hand,
//! and no part of what the document *is*.

use serde::{Deserialize, Serialize};

use crate::geom::Vec2;

/// A control point of the fitted stroke curve — the stored form of a path.
///
/// Distinct from `stark-engine`'s `InputSample` on purpose: an input sample is one *pointer
/// report* (raw, jittery, high frequency, discarded once fitted); a control point
/// is one coefficient of the fitted curve (stable, saved to the file and sent to
/// peers). It is a **cubic B-spline** control point, so the curve is pulled
/// towards it rather than through it — only the first and last are on the curve,
/// which the clamped end condition pins them to.
///
/// `time` is seconds since the stroke started rather than an absolute clock —
/// that is what velocity and timelapse want (§8), and it halves the
/// field.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ControlPoint {
    pub pos: Vec2,
    pub pressure: f32,
    pub tilt: Vec2,
    pub time: f32,
}

impl ControlPoint {
    /// A full-pressure knot at `pos` (mouse input, or tests).
    pub fn at(pos: Vec2) -> Self {
        Self {
            pos,
            pressure: 1.0,
            tilt: Vec2::ZERO,
            time: 0.0,
        }
    }
}
