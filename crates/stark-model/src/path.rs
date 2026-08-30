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
///
/// **Deserialization funnels through [`ControlPoint::clamped`]** (§8), so a point
/// arriving from a file or a peer holds the same bounds a fitter's does. A `9.0`
/// pressure from a corrupt log is not a slightly-wider stroke but a stamp nine times
/// the size the footprint padded for, which is a §12.6 divergence rather than a
/// visible bug.
///
/// The fields stay `pub`, so this is a funnel and not a wall: the three fitters and
/// the tests that stage a bad point still build one directly. `pos` needs no gate —
/// `footprint::stroke_rect` tests every point for finiteness itself and claims the
/// whole layer rather than trusting the box.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
#[serde(from = "RawControlPoint", into = "RawControlPoint")]
#[carbonite(as = "RawControlPoint")]
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
    /// **Every fitted point is built this way, and so is every decoded one.** A fit
    /// is a least-squares solve rather than an interpolation, so a control point the
    /// data barely reaches is held only by the ridge and can overshoot the values it
    /// was fitted from. Pressure is a radius the renderer multiplies the brush by
    /// with no ceiling of its own, and tilt steers the footprint.
    ///
    /// Clamping the *control* values bounds the whole curve and not just the control
    /// polygon: B-spline bases are non-negative and sum to one, so every evaluated
    /// value is a convex combination of them.
    ///
    /// It lives here rather than beside any one fitter because there are three — the
    /// streaming fit, its finished form, and the shape assist's realization (§6.9) —
    /// and a fourth is whatever tool next produces a path.
    pub fn clamped(pos: Vec2, pressure: f32, tilt: Vec2, time: f32) -> Self {
        // Scaled rather than component-clamped: the pen reports a direction and a
        // lean, and clipping the components alone would turn a diagonal overshoot
        // into a different direction.
        let len = tilt.length();
        Self {
            pos,
            // `clamp01` rather than `clamp`, per the crate's NaN policy. The tilt
            // below is already safe on NaN — `len > 1.0` is false, and the vector
            // passes through to a fitter that tests it.
            pressure: crate::clamp01(pressure),
            tilt: if len > 1.0 { tilt / len } else { tilt },
            time,
        }
    }
}

/// The wire shape of a [`ControlPoint`], which is the same shape — its only job is
/// to be the type `#[serde(from)]` deserializes *before* the constructor runs.
///
/// Named in both directions because a schema describes reading and writing at once
/// (§8). The fields mirror the originals **in order**, so the encoding is
/// unchanged.
#[derive(Serialize, Deserialize, carbonite::Schema)]
#[serde(rename = "ControlPoint")]
struct RawControlPoint {
    pos: Vec2,
    pressure: f32,
    tilt: Vec2,
    time: f32,
}

impl From<RawControlPoint> for ControlPoint {
    fn from(raw: RawControlPoint) -> Self {
        Self::clamped(raw.pos, raw.pressure, raw.tilt, raw.time)
    }
}

impl From<ControlPoint> for RawControlPoint {
    /// A rename and nothing more: a point in hand already holds what
    /// [`ControlPoint::clamped`] promises, and writing it back unchanged keeps this
    /// from being a second gate that could disagree with the first.
    fn from(p: ControlPoint) -> Self {
        Self {
            pos: p.pos,
            pressure: p.pressure,
            tilt: p.tilt,
            time: p.time,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A control point decoded from a file or a peer holds what
    /// [`ControlPoint::clamped`] promises.
    ///
    /// Three lies a corrupt or hostile log could tell: a pressure past 1 (a radius
    /// the footprint did not pad for), a NaN pressure (the case `f32::clamp` lets
    /// through), and a tilt outside the unit disc, which steers the stamp
    /// footprint.
    #[test]
    fn a_control_point_from_the_wire_is_normalized() {
        let wire = |p: &ControlPoint| carbonite::to_vec_static(p).expect("encodes");
        let back = |b: &[u8]| carbonite::from_slice_static::<ControlPoint>(b).expect("decodes");

        // Built by hand in the states the constructor refuses, then round-tripped.
        let hot = ControlPoint {
            pos: Vec2::splat(3.0),
            pressure: 9.0,
            tilt: Vec2::splat(4.0),
            time: 0.25,
        };
        let landed = back(&wire(&hot));
        assert_eq!(landed.pressure, 1.0, "pressure is clamped into range");
        assert!(
            landed.tilt.length() <= 1.0 + 1e-6,
            "tilt is scaled into the unit disc, not clipped per component",
        );
        // Scaled rather than component-clamped, so the lean's *direction* survives.
        assert!(
            (landed.tilt.x - landed.tilt.y).abs() < 1e-6,
            "a diagonal lean must stay diagonal",
        );
        assert_eq!(landed.pos, hot.pos, "position is the footprint's to judge");

        assert_eq!(
            back(&wire(&ControlPoint {
                pressure: f32::NAN,
                ..hot
            }))
            .pressure,
            0.0,
            "a NaN pressure lands on zero, not NaN",
        );

        // …and an ordinary point comes through bit for bit: the funnel is
        // idempotent on anything a fitter wrote, or every load would be a small
        // edit to the picture.
        let clean = ControlPoint::clamped(Vec2::splat(7.0), 0.4, Vec2::new(0.3, -0.2), 1.5);
        assert_eq!(back(&wire(&clean)), clean);
        assert_eq!(
            back(&wire(&ControlPoint::at(Vec2::ZERO))),
            ControlPoint::at(Vec2::ZERO)
        );
    }
}
