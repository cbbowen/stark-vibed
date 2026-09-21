//! The two builders both the hover and the shape tests want. Anything one of
//! them wants alone sits beside that test.

use super::Session;
use crate::command::Tool;
use crate::view::{Extent2, ViewTransform};
use stark_model::document::{LayerId, ShapeAction};
use stark_model::geom::Vec2;

/// A session over a 256² identity view, with `action` armed for the next shape
/// gesture.
pub(super) fn session(action: ShapeAction) -> Session {
    let mut s = Session::new(
        ViewTransform::identity(Extent2::new(256, 256)),
        LayerId::ROOT,
    );
    s.shape_action = action;
    s
}

/// Input tolerances the fit would not use as given ([`crate::path::clamp_tolerance`]):
/// a pen past 16× zoom is under the floor, and the rest are one bad division away.
pub(super) const UNUSABLE_TOLERANCES: [f32; 5] = [f32::NAN, 0.0, 1e-6, -1.0, f32::INFINITY];

/// Press a shape gesture at `pos`, with nothing already selected.
pub(super) fn press(s: &mut Session, tool: Tool, pos: Vec2) {
    s.start_selection(tool, pos, false, stark_model::geom::IVec2::ZERO);
}
