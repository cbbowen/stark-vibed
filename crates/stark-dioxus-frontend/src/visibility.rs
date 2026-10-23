//! Writing this browser's on-screen state back to the store (§25.6).
//!
//! The record and every read of it are `stark_ui::visibility`. What is here is
//! the one direction that needs signals: capturing what the app has on screen right
//! now.

use dioxus::prelude::*;

use crate::state::AppState;
use stark_ui::commands::VisibilityToggle;

/// Write what is on screen. **The one writer**, called by everything that shows or
/// hides anything — `layout::set_open` and `layout::toggle_collapse`,
/// `navigator::set_open`, `slots::set_pinned`, `panels::timeline::set_open`.
///
/// Whole-record rather than incremental, and that is the point: the four facts live
/// in four signals, and reading all of them here is what keeps the record from ever
/// holding a half-updated screen. It is also what makes the `match` in it the place
/// a new menu entry is *made* to declare itself.
///
/// `peek` throughout, never `read`: this runs from inside the writers, and
/// subscribing a click handler to the state it is in the middle of changing is how a
/// signal read ends up live across a write of itself.
pub fn persist(state: AppState) {
    let hidden = state.panels.hidden.peek().clone();
    let collapsed = state.panels.collapsed.peek().clone();
    let navigator = *state.navigator.peek();
    let quick_brushes = *state.slots.pinned.peek();
    let timeline = *state.timeline.open.peek();

    stark_ui::visibility::persist(
        |what| match what {
            // The exhaustive match durability hangs on. A tenth entry in the menu
            // stops the build here until somebody says where its bit is kept — which
            // is why the closure is this frontend's and the row shape is not.
            VisibilityToggle::Panel(id) => !hidden.contains(&id),
            VisibilityToggle::Navigator => navigator,
            VisibilityToggle::QuickBrushes => quick_brushes,
            VisibilityToggle::Timeline => timeline,
        },
        // A floating stack folds panels and nothing else: the three entries below
        // the panels have no title bar to fold to (`stark_ui::visibility::persist`).
        |what| matches!(what, VisibilityToggle::Panel(id) if collapsed.contains(&id)),
    );
}
