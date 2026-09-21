//! Two engines in one session with no wire between them (§12): the fixture every
//! collaboration test opens with, and the pump that stands in for the transport.

use stark_engine::{Engine, Extent2, RgbaImage};
use stark_model::document::ActorId;

use super::{SIZE, engine_or_skip_sized};

/// A session of two: `a` started it as actor 1 and `b` joined from `a`'s file as
/// actor 2, so both hold one log and nothing has been painted yet. `None` where there
/// is no adapter and `stark_engine::testing::ALLOW_NO_GPU` permits the skip.
///
/// A scenario whose second peer has to join *late* — after a substrate is set, after
/// paint is down — opens by hand, since when `b` joins is the scenario.
pub fn pair() -> Option<(Engine, Engine)> {
    pair_sized(SIZE)
}

/// [`pair`] on a chosen viewport, for the scenarios that need two tile columns.
pub fn pair_sized(size: Extent2) -> Option<(Engine, Engine)> {
    let mut a = engine_or_skip_sized(size)?;
    let mut b = engine_or_skip_sized(size)?;
    a.start_collaboration(ActorId(1));
    b.join_collaboration(&a.document_file(), ActorId(2))
        .expect("join a session this build can render");
    Some((a, b))
}

pub fn snap(e: &mut Engine) -> RgbaImage {
    e.render_to_image()
}

/// Pump every pending local action from `from` into `into`.
pub fn sync_into(from: &mut Engine, into: &mut Engine) {
    for action in from.take_outbox() {
        into.merge_remote(action);
    }
}

/// Exchange outboxes both ways, `a`'s first.
pub fn sync(a: &mut Engine, b: &mut Engine) {
    sync_into(a, b);
    sync_into(b, a);
}
