//! One pointer report, as it arrived (§6.2).
//!
//! The geometry's own input type: everything here consumes it and nothing produces
//! it, so it sits below the engine's `command` module rather than in it — which is
//! also what lets the fitter be tested without a GPU in sight.

use serde::{Deserialize, Serialize};
use stark_model::geom::Vec2;

/// One pen/mouse sample in canvas space.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InputSample {
    pub pos: Vec2,
    pub pressure: f32,
    pub tilt: Vec2,
    /// Timestamp in seconds, for velocity and timelapse (§8).
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

    /// Whether this sample may be remembered — the gate a report has to pass
    /// before anything stateful is allowed to keep it.
    ///
    /// **A sample arrives from outside, and one NaN in it would freeze the stroke for
    /// good**, which is why this is a property of the sample rather than a check at
    /// each consumer: [`PathFitter`](crate::path::PathFitter) parameterizes its samples
    /// by arc length, a NaN spreads from `arc` into every curve parameter, the normal
    /// equations go NaN and no ridge factors them, so every later solve leaves the
    /// polygon where it was.
    ///
    /// Every channel, not just the position: `time` seeds the fitter's epoch, and
    /// `pressure` and `tilt` ride the same least-squares solve the geometry does.
    ///
    /// **Finite is not enough, which is why this is not `is_finite`.** Arc length is
    /// a *difference* of two positions, and two finite reports a whole `f32` range
    /// apart differ by infinity — from which `arc` divides `inf / inf` and hands the
    /// solve the NaN it was written to refuse. So the position is bounded as well as
    /// finite, by [`COORD_LIMIT`].
    pub fn is_admissible(&self) -> bool {
        self.pos.is_finite()
            && self.pos.abs().max_element() < COORD_LIMIT
            && self.pressure.is_finite()
            && self.tilt.is_finite()
            && self.time.is_finite()
    }
}

/// How far from the origin, in canvas px on either axis, a sample may report and
/// still be [admissible](InputSample::is_admissible) — **exclusive**, since this is
/// the first coordinate the tile grid cannot address rather than the last one it can.
///
/// **The canvas's own edge, not a number picked to be large.** Tiles are addressed
/// by an `i32` [`TileCoord`](stark_model::geom::TileCoord), and this is `2³¹` of them
/// (`i32::MAX as f32` rounds up to `2³¹`, and `2³¹ · 254` is exact in an `f32`), so
/// [`TileRect::covering`](stark_model::geom::TileRect::covering) refuses this value
/// and everything beyond it.
///
/// It is a **ceiling, not the exact reach**: every consumer pads before it quantizes,
/// so `covering` starts refusing somewhat inside this on both sides and then claims
/// [`TileRect::ALL`](stark_model::geom::TileRect::ALL), which is the safe direction
/// (§12.6). What this bound is *for* is the other end: two admissible samples are
/// under `2 · COORD_LIMIT` ≈ 10¹² apart, so no difference of two positions is
/// infinite, and the normal equations square that to ~10²⁴ where an `f32` still has
/// fourteen orders of magnitude in hand.
pub const COORD_LIMIT: f32 = i32::MAX as f32 * stark_model::geom::TILE_SIZE as f32;

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
