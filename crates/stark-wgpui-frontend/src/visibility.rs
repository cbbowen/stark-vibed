//! What this window has on screen, and what it remembers of that (§11.2, §25.6).
//!
//! The record, its rows and its writer are `stark_ui::visibility`; what is here is
//! the part only this frontend can answer — **which shelves it has**, and what "not
//! stored" means to a window whose chrome is docked rather than floating.
//!
//! # A shelf, not a panel
//!
//! The columns are keyed by [`VisibilityToggle`] rather than by `PanelId`, and the
//! navigator is why: it is not a panel in either app — the web app's is a miniature
//! in the corner of the canvas with no title to wear (`navigator`) — but in a docked
//! chrome it is a box in a column beside the panels, foldable and hideable exactly as
//! they are. `VisibilityToggle` is already the vocabulary for "a thing the visibility
//! menu shows and hides, panel or not", so it is the key, and the record it writes is
//! keyed the same way.
//!
//! # Hidden and folded are different questions
//!
//! A folded shelf leaves its title bar behind, so the column still says it is there
//! (`crate::panel`). A hidden one is not built at all, and the Window menu is the only
//! way back — which is why the menu is a map of the whole set rather than a list of
//! what is up, and why every row wears its own state. Both are one row in the record:
//! a shelf that is not showing cannot also be folded, because there is nothing on
//! screen to fold.
//!
//! # The default is furnished
//!
//! The web app's stack floats over the painting and opens empty, so "never been here"
//! and "closed everything" are one screen there and `stored_hidden` collapses them.
//! This window's columns are its furniture: the canvas is what is *left* once they
//! have taken their room, and a first run showing a menu bar over an empty grey field
//! would read as a broken app. So [`stored`] asks for the distinction
//! (`stark_ui::visibility::stored_open`) and answers the absent case itself.

use std::collections::HashSet;

use stark_ui::commands::VisibilityToggle;
use stark_ui::panels::PanelId;

/// One shelf of a column, as this module names them.
const fn panel(id: PanelId) -> VisibilityToggle {
    VisibilityToggle::Panel(id)
}

/// The shelves the left-hand column stacks, top to bottom (`crate::panel`).
///
/// The tool column: what the hand is about to do, and what it is about to do it
/// through. The medium and the scaffolding sit under those because they are chosen
/// between passages rather than during one.
pub const LEFT: [VisibilityToggle; 4] = [
    panel(PanelId::Brush),
    panel(PanelId::Select),
    panel(PanelId::Lighting),
    panel(PanelId::Guides),
];

/// The shelves pinned to the right-hand edge.
///
/// The reading column: where you are in the piece, what the next stroke is made of,
/// and what it will land on. Color leads the panels here for the reason it leads
/// `PanelId::ALL` — it is reached for between nearly every pair of strokes — and the
/// overview leads the column because it is glanceable rather than operated.
pub const RIGHT: [VisibilityToggle; 3] = [
    VisibilityToggle::Navigator,
    panel(PanelId::Color),
    panel(PanelId::Layers),
];

/// Every shelf this frontend draws, in the order the Window menu lists them: the tool
/// column down one edge, then the reading column down the other.
///
/// **Seven of the nine**, and the absences are the same ones the menu bar's are:
/// there is no quick-brush rack here and no Timeline mode, and a Window menu offering
/// to show something that does not exist would be worse than a short menu
/// (`crate::menu`). Written out rather than folded from the two above so this list
/// says what it holds; a test keeps the three in step.
pub const SHELVES: [VisibilityToggle; 7] = [
    panel(PanelId::Brush),
    panel(PanelId::Select),
    panel(PanelId::Lighting),
    panel(PanelId::Guides),
    VisibilityToggle::Navigator,
    panel(PanelId::Color),
    panel(PanelId::Layers),
];

/// The shelves this client left hidden — **none**, for a client that has never said.
pub fn stored() -> HashSet<VisibilityToggle> {
    let Some(open) = stark_ui::visibility::stored_screen() else {
        return HashSet::new();
    };
    SHELVES.into_iter().filter(|w| !open.contains(w)).collect()
}

/// Which shelves this client left folded to their title bar.
pub fn stored_folded() -> HashSet<VisibilityToggle> {
    stark_ui::visibility::stored_folded()
        .into_iter()
        .filter(|what| SHELVES.contains(what))
        .collect()
}

/// Write what this window has on screen back.
///
/// The exhaustive match is what the shared writer asks for and what this frontend
/// owes it: a tenth entry in the vocabulary stops the build here until somebody says
/// where its bit is kept. Two of them have no bit at all — the quick-brush rack and
/// Timeline mode are surfaces this frontend has not got — and saying so is the
/// answer, not a gap.
pub fn persist(hidden: &HashSet<VisibilityToggle>, folded: &HashSet<VisibilityToggle>) {
    stark_ui::visibility::persist(
        |what| match what {
            VisibilityToggle::Panel(_) | VisibilityToggle::Navigator => {
                SHELVES.contains(&what) && !hidden.contains(&what)
            }
            VisibilityToggle::QuickBrushes | VisibilityToggle::Timeline => false,
        },
        |what| folded.contains(&what),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The menu's list is the two columns' lists, in that order — the one thing that
    /// could quietly stop being true, since all three are written out by hand and a
    /// shelf is added to one of them at a time.
    #[test]
    fn the_window_menu_is_the_two_columns() {
        let both: Vec<VisibilityToggle> = LEFT.into_iter().chain(RIGHT).collect();
        assert_eq!(SHELVES.to_vec(), both);
    }

    /// Every shelf this frontend claims is one the vocabulary knows, and none is
    /// claimed twice — the roster's own list is what the record is keyed by.
    #[test]
    fn every_shelf_named_is_one_that_exists() {
        for what in SHELVES {
            assert!(
                VisibilityToggle::ALL.contains(&what),
                "{what:?} is not a menu entry"
            );
        }
        let unique: HashSet<VisibilityToggle> = SHELVES.into_iter().collect();
        assert_eq!(unique.len(), SHELVES.len(), "a shelf is listed once");
    }

    /// A window that has never stored anything opens furnished — the whole of what
    /// this module adds to the shared reader, and the case a first run is in.
    #[test]
    fn a_first_run_opens_with_every_shelf_up() {
        assert!(
            stored().is_empty(),
            "nothing stored hides nothing: the columns are this window's furniture"
        );
        assert!(stored_folded().is_empty());
    }
}
