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

/// Press a shape gesture at `pos`, with nothing already selected.
pub(super) fn press(s: &mut Session, tool: Tool, pos: Vec2) {
    s.start_selection(tool, pos, false, stark_model::geom::IVec2::ZERO);
}
