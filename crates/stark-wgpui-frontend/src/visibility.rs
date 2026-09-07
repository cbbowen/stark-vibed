//! What this window has on screen, and what it remembers of that (§11.2, §25.6).
//!
//! The record, its rows and its writer are `stark_ui::visibility`; what is here is
//! the part only this frontend can answer — **which panels it has**, and what "not
//! stored" means to a window whose panels are docked rather than floating.
//!
//! # Hidden and folded are different questions
//!
//! A folded panel leaves its title bar behind, so the column still says the panel is
//! there (`crate::panel`). A hidden one is gone, and the Window menu is the only way
//! back — which is why the menu is a map of the whole set rather than a list of what
//! is up, and why every row wears its own state. The two facts are one row in the
//! record: a panel that is not showing cannot also be folded, because there is
//! nothing on screen to fold.
//!
//! # The default is furnished
//!
//! The web app's stack floats over the painting and opens empty, so "never been here"
//! and "closed everything" are one screen there and `stored_hidden` collapses them.
//! This window's panels are its furniture: the canvas is what is *left* once they
//! have taken their columns, and a first run showing a menu bar over an empty grey
//! field would read as a broken app. So [`stored`] asks for the distinction
//! (`stark_ui::visibility::stored_open`) and answers the absent case itself.

use std::collections::HashSet;

use stark_ui::commands::VisibilityToggle;
use stark_ui::panels::PanelId;

/// The panels the left-hand column stacks, top to bottom (`crate::panel`).
pub const COLUMN: [PanelId; 3] = [PanelId::Color, PanelId::Brush, PanelId::Select];

/// The panels pinned to the right-hand edge (`crate::layers`).
pub const ROSTER: [PanelId; 1] = [PanelId::Layers];

/// Every panel this frontend draws, in the order the Window menu lists them: the
/// column down one edge, then the roster down the other.
///
/// **Four of the six**, and the absences are the same ones the menu bar's are: there
/// is no Guides panel here and no Lighting one, and a Window menu offering to show a
/// panel that does not exist would be worse than a short menu (`crate::menu`). Written
/// out rather than folded from the two above so this list says what it holds; a test
/// keeps the three in step.
pub const PANELS: [PanelId; 4] = [
    PanelId::Color,
    PanelId::Brush,
    PanelId::Select,
    PanelId::Layers,
];

/// The panels this client left hidden — **none**, for a client that has never said.
pub fn stored() -> HashSet<PanelId> {
    let Some(open) = stark_ui::visibility::stored_open() else {
        return HashSet::new();
    };
    PANELS.into_iter().filter(|id| !open.contains(id)).collect()
}

/// Write what this window has on screen back.
///
/// The exhaustive match is what the shared writer asks for and what this frontend
/// owes it: a tenth entry in the vocabulary stops the build here until somebody says
/// where its bit is kept. Today three of them have no bit at all — the navigator, the
/// quick-brush rack and Timeline mode are surfaces this frontend has not got — and
/// saying so is the answer, not a gap.
///
/// A panel this frontend does not draw gets no row either, for the same reason: this
/// window cannot report a Guides panel as being on screen when it has never had one
/// to show.
pub fn persist(hidden: &HashSet<PanelId>, folded: &HashSet<PanelId>) {
    stark_ui::visibility::persist(
        |what| match what {
            VisibilityToggle::Panel(id) => PANELS.contains(&id) && !hidden.contains(&id),
            VisibilityToggle::Navigator
            | VisibilityToggle::QuickBrushes
            | VisibilityToggle::Timeline => false,
        },
        folded,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The menu's list is the two columns' lists, in that order — the one thing that
    /// could quietly stop being true, since all three are written out by hand and a
    /// panel is added to one of them at a time.
    #[test]
    fn the_window_menu_is_the_two_columns() {
        let both: Vec<PanelId> = COLUMN.into_iter().chain(ROSTER).collect();
        assert_eq!(PANELS.to_vec(), both);
    }

    /// Every panel this frontend claims is one the vocabulary knows, and no panel is
    /// claimed twice — the roster's own list is what the record is keyed by.
    #[test]
    fn every_panel_named_is_a_panel_that_exists() {
        for id in PANELS {
            assert!(PanelId::ALL.contains(&id), "{id:?} is not a panel");
        }
        let unique: HashSet<PanelId> = PANELS.into_iter().collect();
        assert_eq!(unique.len(), PANELS.len(), "a panel is listed once");
    }

    /// A window that has never stored anything opens furnished — the whole of what
    /// this module adds to the shared reader, and the case a first run is in.
    #[test]
    fn a_first_run_opens_with_every_panel_up() {
        assert!(
            stored().is_empty(),
            "nothing stored hides nothing: the columns are this window's furniture"
        );
    }
}
