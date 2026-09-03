//! Writing this browser's on-screen state back to the store (§25.6).
//!
//! The record and every read of it are `stark_chrome::visibility`. What is here is
//! the one direction that needs signals: capturing what the app has on screen right
//! now.

use dioxus::prelude::*;

use crate::state::AppState;
use stark_chrome::commands::VisibilityToggle;
use stark_chrome::storage;
use stark_chrome::visibility::StoredVisible;

pub fn persist(state: AppState) {
    let hidden = state.panels.hidden.peek().clone();
    let collapsed = state.panels.collapsed.peek().clone();
    let navigator = *state.navigator.peek();
    let quick_brushes = *state.slots.pinned.peek();
    let timeline = *state.timeline.open.peek();

    let rows: Vec<StoredVisible> = VisibilityToggle::ALL
        .into_iter()
        .filter(|what| match what {
            // The exhaustive match durability now hangs on. A tenth entry in the
            // menu stops the build here until somebody says where its bit is kept.
            VisibilityToggle::Panel(id) => !hidden.contains(id),
            VisibilityToggle::Navigator => navigator,
            VisibilityToggle::QuickBrushes => quick_brushes,
            VisibilityToggle::Timeline => timeline,
        })
        .map(|what| StoredVisible {
            what,
            collapsed: matches!(what, VisibilityToggle::Panel(id) if collapsed.contains(&id)),
        })
        .collect();
    storage::save_list(&rows);
}
