//! The poses the guide tests are stated at (§20).
//!
//! Shared because the same camera has to answer to `camera`'s derivations and to
//! `conic`'s charts — a theorem checked through one and a chart checked through the
//! other are claims about one guide, not two.

use glam::{Quat, Vec2};

use super::{Lens, PerspectiveGuide};

/// A guide at the classical Euler pose — the tests state poses this way
/// because the theorems are stated this way; the *state* is one
/// quaternion.
pub(super) fn guide(yaw: f32, pitch: f32, roll: f32) -> PerspectiveGuide {
    PerspectiveGuide {
        center: Vec2::new(320.0, -140.0),
        focal: 800.0,
        rotation: Quat::from_rotation_z(roll)
            * Quat::from_rotation_x(pitch)
            * Quat::from_rotation_y(yaw),
        ..Default::default()
    }
}

/// [`guide`] seen through the curvilinear lens (§20.8).
pub(super) fn fisheye(yaw: f32, pitch: f32, roll: f32) -> PerspectiveGuide {
    PerspectiveGuide {
        lens: Lens::Fisheye,
        ..guide(yaw, pitch, roll)
    }
}
