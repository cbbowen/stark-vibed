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
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
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

    /// A control point with its channels **held to what a pen can report** —
    /// pressure in `[0, 1]`, tilt inside the unit disc.
    ///
    /// **Every fitted point is built this way**, and the reason is that a fit is a
    /// least-squares solve rather than an interpolation: a control point the data
    /// barely reaches is held only by the ridge, so it can overshoot the values it
    /// was fitted from. Pressure is a radius the renderer multiplies the brush by
    /// with no ceiling of its own, and tilt steers the footprint.
    ///
    /// Clamping the *control* values bounds the whole curve and not just the control
    /// polygon: B-spline bases are non-negative and sum to one, so every evaluated
    /// value is a convex combination of them.
    ///
    /// It lives here rather than beside any one fitter because there are three —
    /// the streaming fit, its finished form, and the shape assist's realization
    /// (§6.9) — and a fourth is what the next tool that produces a path will be.
    /// Three copies of a clamp is three places for one of them to be forgotten, and
    /// what a forgotten one costs is a stroke whose radius the log does not bound.
    pub fn clamped(pos: Vec2, pressure: f32, tilt: Vec2, time: f32) -> Self {
        // Scaled rather than component-clamped: the pen reports a direction and a
        // lean, and clipping the components alone would turn a diagonal overshoot
        // into a different direction.
        let len = tilt.length();
        Self {
            pos,
            pressure: pressure.clamp(0.0, 1.0),
            tilt: if len > 1.0 { tilt / len } else { tilt },
            time,
        }
    }
}
