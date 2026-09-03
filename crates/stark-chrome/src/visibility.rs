//! What is on screen, as a client remembers it (§11, §25.6).
//!
//! One record, keyed by [`VisibilityToggle`] — the enum a visibility menu is already
//! a loop over. The menu calls itself "the map of what is on screen"; this is that map
//! written down, and the two cannot drift because they are the same list.
//!
//! # Why one record and not four
//!
//! There were two, and two things were forgotten entirely. The panel stack kept
//! `stark.panels` and the navigator kept `stark.navigator`, each with its own type,
//! its own reader and its own writer — an arrangement that made sense while the
//! navigator was the *only* thing outside the stack. Then the quick-brush rack and
//! Timeline mode joined the menu, and neither joined a store.
//!
//! **This is the record N1 could not move.** Its key is a `VisibilityToggle` and its
//! rows hold `PanelId`s, and both of those were the command registry's and the
//! layout's; a record cannot move before its vocabulary has. N3 brought both down, so
//! this follows.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::commands::VisibilityToggle;
use crate::panels::PanelId;
use crate::storage::{self, Entry, Store};

/// One thing a client last had on screen.
///
/// The `what` is the menu's own entry, so the stored word is the enum's word by
/// construction (`{"Panel":"Layers"}`, `"Navigator"`) and a variant renamed costs the
/// row rather than mis-matching it — `PanelId`'s bargain, one level out.
#[derive(Serialize, Deserialize)]
pub struct StoredVisible {
    pub what: VisibilityToggle,
    /// Folded to its title bar. Only a panel can be, and only while it is open.
    #[serde(default)]
    pub collapsed: bool,
}

impl Entry for StoredVisible {
    const STORE: Store = Store::Visible;
}

/// The rows this browser stored. A row today's build cannot make sense of — an entry
/// retired since it was written — costs that row and not the screen
/// (`storage::load_list`).
fn stored() -> Vec<StoredVisible> {
    storage::load_list().unwrap_or_default()
}

/// Whether `what` was on screen when this browser last looked. The seed for each of
/// the three entries that are not panels; the panels come through the two below,
/// which answer for the whole stack at once.
pub fn stored_showing(what: VisibilityToggle) -> bool {
    stored().iter().any(|row| row.what == what)
}

/// The panels this browser did **not** leave open, as `PanelLayout::hidden` — every
/// panel, for a browser that has never been here or whose record will not read.
pub fn stored_hidden() -> HashSet<PanelId> {
    let open: HashSet<PanelId> = panels(stored()).map(|(id, _)| id).collect();
    PanelId::ALL
        .into_iter()
        .filter(|id| !open.contains(id))
        .collect()
}

/// The panels this browser left **folded** (`PanelLayout::collapsed`).
///
/// Read from the same rows as [`stored_hidden`], since it is the same fact about the
/// same panel; a panel that is not open cannot appear here at all.
pub fn stored_collapsed() -> HashSet<PanelId> {
    panels(stored())
        .filter(|(_, collapsed)| *collapsed)
        .map(|(id, _)| id)
        .collect()
}

/// The panel rows of a stored screen, as `(id, folded)` — the other three entries
/// dropped, since neither reader above has anything to say about them.
fn panels(rows: Vec<StoredVisible>) -> impl Iterator<Item = (PanelId, bool)> {
    rows.into_iter().filter_map(|row| match row.what {
        VisibilityToggle::Panel(id) => Some((id, row.collapsed)),
        _ => None,
    })
}

/// Write what is on screen. **The one writer**, called by everything that shows or
/// hides anything — `layout::set_open` and `layout::toggle_collapse`,
/// `navigator::set_open`, `slots::set_pinned`, `panels::timeline::set_open`.
///
/// Whole-record rather than incremental, and that is the point: the four facts live
/// in four signals, and reading all of them here is what keeps the record from ever
/// holding a half-updated screen. It is also what makes the `match` below the place
/// a new menu entry is *made* to declare itself.
///
/// `peek` throughout, never `read`: this runs from inside the writers, and
/// subscribing a click handler to the state it is in the middle of changing is how a
/// signal read ends up live across a write of itself.
#[cfg(test)]
mod tests {
    use super::*;

    /// Every entry of the visibility menu round-trips through a row, and the stored
    /// word is the enum's own.
    ///
    /// The menu and the record are one list by construction — [`persist`] matches on
    /// `VisibilityToggle::ALL` — so what is worth pinning is the *spelling*: a row is
    /// only found again if what went in reads back equal, and the two entries that
    /// carry a payload are where that could quietly stop being true.
    #[test]
    fn every_menu_entry_survives_being_written_down() {
        for what in VisibilityToggle::ALL {
            let json = serde_json::to_string(&StoredVisible {
                what,
                collapsed: false,
            })
            .expect("a row encodes");
            let back: StoredVisible = serde_json::from_str(&json).expect("and reads back");
            assert_eq!(back.what, what, "{json}");
        }
        let json = serde_json::to_string(&StoredVisible {
            what: VisibilityToggle::Panel(PanelId::Layers),
            collapsed: true,
        })
        .unwrap();
        assert!(
            json.contains(r#"{"Panel":"Layers"}"#),
            "a panel is stored by its own name: {json}"
        );
    }

    /// A row for an entry this build no longer has costs that row and not the screen
    /// — the rule the list format exists for, leaned on here by every entry that is
    /// ever retired from the menu.
    #[test]
    fn an_entry_this_build_does_not_know_costs_its_own_row() {
        let json = r#"[
            {"what":{"Panel":"Layers"},"collapsed":true},
            {"what":"Holodeck"},
            {"what":"Navigator","collapsed":false},
            {"what":{"Panel":"Atlantis"}}
        ]"#;
        let rows: Vec<StoredVisible> = serde_json::from_str::<Vec<serde_json::Value>>(json)
            .unwrap()
            .into_iter()
            .filter_map(|v| serde_json::from_value(v).ok())
            .collect();
        assert_eq!(
            rows.len(),
            2,
            "the two readable rows survive the two that are not"
        );
        assert_eq!(
            stored_hidden_from(rows),
            PanelId::ALL
                .into_iter()
                .filter(|id| *id != PanelId::Layers)
                .collect::<HashSet<_>>(),
            "and the surviving panel row is still the only one open"
        );
    }

    /// A folded panel is a **field** on its own row, so the two states cross a
    /// version in both directions: a row written before folding existed reads as an
    /// open panel, and one written by this build reads as an open panel to a version
    /// that knows only the entry.
    #[test]
    fn a_stored_row_carries_whether_the_panel_was_folded() {
        let read = |json: &str| {
            serde_json::from_str::<StoredVisible>(json).map(|row| (row.what, row.collapsed))
        };
        assert_eq!(
            read(r#"{"what":{"Panel":"Brush"},"collapsed":true}"#).unwrap(),
            (VisibilityToggle::Panel(PanelId::Brush), true)
        );
        // A row from before folding existed, and one from a build that has since
        // dropped it: the entry alone is showing and unfolded.
        assert_eq!(
            read(r#"{"what":{"Panel":"Layers"}}"#).unwrap(),
            (VisibilityToggle::Panel(PanelId::Layers), false)
        );
        // And the field is meaningless on the entries that are not panels, which is
        // what makes it safe to leave off every one of their rows.
        assert_eq!(
            read(r#"{"what":"Navigator"}"#).unwrap(),
            (VisibilityToggle::Navigator, false)
        );
    }

    /// [`stored_hidden`]'s reading, without the store — the half worth testing.
    fn stored_hidden_from(rows: Vec<StoredVisible>) -> HashSet<PanelId> {
        let open: HashSet<PanelId> = panels(rows).map(|(id, _)| id).collect();
        PanelId::ALL
            .into_iter()
            .filter(|id| !open.contains(id))
            .collect()
    }
}
