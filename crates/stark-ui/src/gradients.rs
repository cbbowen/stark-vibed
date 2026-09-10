//! The gradient library (§22.3): named ramps captured off the canvas, as this client
//! keeps them.
//!
//! An entry follows this client across documents and never enters one — a gradient is
//! something the artist paints **with** — so whatever consumes one embeds the ramp by
//! value in the action it commits (§22.4). The trace that captures an entry and the
//! pop-out that lists them are each frontend's; the record, its names and the rule a
//! rename keeps are here.

use serde::{Deserialize, Serialize};
use stark_model::gradient::Gradient;

use crate::storage::{self, Store};

/// One named gradient — and, unchanged, one stored entry: both fields are durable, so a
/// second struct to map it onto would be a copy with nothing to say.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GradientEntry {
    /// Unique in the library, because it is the entry's identity: the selection, a
    /// removal and a row's key all speak it. [`next_name`] proposes a free one and
    /// [`rename`] refuses one already worn.
    pub name: String,
    pub gradient: Gradient,
}

impl storage::Entry for GradientEntry {
    const STORE: Store = Store::Gradients;
}

/// The entry a fill would use: the one `selected` names, or the first while nothing is
/// selected or the selection names an entry since removed — so a library with anything
/// in it always answers, and a list can highlight the row this resolves to.
pub fn current<'a>(
    entries: &'a [GradientEntry],
    selected: Option<&str>,
) -> Option<&'a GradientEntry> {
    selected
        .and_then(|name| entries.iter().find(|e| e.name == name))
        .or_else(|| entries.first())
}

/// Why [`rename`] left the library as it was.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenameRefused {
    /// The new name is empty once trimmed.
    Empty,
    /// The new name is the one the entry already wears.
    Unchanged,
    /// Another entry already wears it. The one refusal worth telling the artist about:
    /// a rename field whose text was silently ignored reads as a lost edit.
    Taken,
    /// No entry wears the old name.
    Missing,
}

/// Rename the entry called `from` to `to`, trimmed, in place.
///
/// # Errors
///
/// [`RenameRefused`] says why nothing changed; the library is untouched in every case.
pub fn rename(entries: &mut [GradientEntry], from: &str, to: &str) -> Result<(), RenameRefused> {
    let to = to.trim();
    if to.is_empty() {
        return Err(RenameRefused::Empty);
    }
    if to == from {
        return Err(RenameRefused::Unchanged);
    }
    if entries.iter().any(|e| e.name == to) {
        return Err(RenameRefused::Taken);
    }
    let entry = entries
        .iter_mut()
        .find(|e| e.name == from)
        .ok_or(RenameRefused::Missing)?;
    entry.name = to.to_string();
    Ok(())
}

/// The first free "Gradient N" name — a capture is named by the machinery, so the artist
/// traces twice without a dialog between.
pub fn next_name(entries: &[GradientEntry]) -> String {
    (1..)
        .map(|i| format!("Gradient {i}"))
        .find(|n| !entries.iter().any(|e| &e.name == n))
        .expect("an unbounded count always reaches a free name")
}

/// Store the library.
///
/// The format and its skip-a-damaged-entry rule are `storage`'s. What this library leans
/// on it for is `Gradient`'s own deserialization gate: what that cannot repair is refused
/// there and the entry dropped on the way back, rather than becoming an unsampleable ramp.
pub fn persist(entries: &[GradientEntry]) {
    storage::save_list(entries);
}

/// What this client has stored, or `None` where it has stored nothing.
pub fn read_storage() -> Option<Vec<GradientEntry>> {
    storage::load_list()
}

#[cfg(test)]
mod tests {
    use super::*;
    use stark_model::Srgb;
    use stark_model::gradient::GradientStop;

    fn gradient() -> Gradient {
        Gradient::new(vec![
            GradientStop {
                t: 0.0,
                color: Srgb::new([1.0, 0.0, 0.0]),
            },
            GradientStop {
                t: 1.0,
                color: Srgb::new([0.0, 0.0, 1.0]),
            },
        ])
        .expect("two stops are a ramp")
    }

    fn library(names: &[&str]) -> Vec<GradientEntry> {
        names
            .iter()
            .map(|name| GradientEntry {
                name: (*name).to_string(),
                gradient: gradient(),
            })
            .collect()
    }

    fn names(entries: &[GradientEntry]) -> Vec<&str> {
        entries.iter().map(|e| e.name.as_str()).collect()
    }

    #[test]
    fn names_count_past_the_holes() {
        let mut entries = library(&["Gradient 1", "Gradient 3"]);
        assert_eq!(next_name(&entries), "Gradient 2");
        entries.remove(0);
        assert_eq!(next_name(&entries), "Gradient 1");
    }

    #[test]
    fn a_stored_entry_survives_the_round_trip_and_a_bad_one_is_repaired() {
        let entry = GradientEntry {
            name: "Dusk".into(),
            gradient: gradient(),
        };
        let json = serde_json::to_string(&entry).expect("an entry encodes");
        let back: GradientEntry = serde_json::from_str(&json).expect("round trip");
        assert_eq!(back.name, "Dusk");
        assert_eq!(back.gradient, entry.gradient);

        // One stop names no ramp, and the load path repairs it into one rather than
        // refusing (§22.1) — so the row survives `storage::load_list` as a flat ramp of
        // its own color, under the name the artist gave it, rather than vanishing from a
        // list the artist can see.
        let bad = r#"{"name":"Bad","gradient":[{"t":0.5,"color":[0.25,0.5,0.75]}]}"#;
        let back: GradientEntry = serde_json::from_str(bad).expect("a row still reads");
        assert_eq!(back.name, "Bad");
        assert_eq!(back.gradient.sample(0.0), Srgb::new([0.25, 0.5, 0.75]));
        assert_eq!(back.gradient.sample(1.0), Srgb::new([0.25, 0.5, 0.75]));
    }

    /// A rename happens in place — the row keeps its seat in the list — and takes the
    /// name trimmed, so a stray space typed at either end is not a second name.
    #[test]
    fn a_rename_happens_in_place() {
        let mut entries = library(&["Dusk", "Dawn", "Noon"]);
        assert_eq!(rename(&mut entries, "Dawn", "  Morning "), Ok(()));
        assert_eq!(names(&entries), ["Dusk", "Morning", "Noon"]);
    }

    /// A name another entry wears is refused, and says so: names are the library's
    /// identity, so taking one would make two rows one.
    #[test]
    fn a_rename_refuses_a_name_already_worn() {
        let mut entries = library(&["Dusk", "Dawn"]);
        assert_eq!(
            rename(&mut entries, "Dawn", "Dusk"),
            Err(RenameRefused::Taken)
        );
        assert_eq!(
            names(&entries),
            ["Dusk", "Dawn"],
            "the library moved anyway"
        );
    }

    /// An empty name is refused, blank space included — and so is a name the entry
    /// already wears, or an entry that is not there, each without touching the library.
    #[test]
    fn a_rename_refuses_nothing_to_rename_to() {
        let mut entries = library(&["Dusk", "Dawn"]);
        assert_eq!(rename(&mut entries, "Dawn", ""), Err(RenameRefused::Empty));
        assert_eq!(
            rename(&mut entries, "Dawn", " \t "),
            Err(RenameRefused::Empty)
        );
        assert_eq!(
            rename(&mut entries, "Dawn", " Dawn "),
            Err(RenameRefused::Unchanged)
        );
        assert_eq!(
            rename(&mut entries, "Noon", "Evening"),
            Err(RenameRefused::Missing)
        );
        assert_eq!(
            names(&entries),
            ["Dusk", "Dawn"],
            "a refusal moved the library"
        );
    }

    /// The selection wins while it names an entry; otherwise the first stands in, so a
    /// library with anything in it always answers — and an empty one never does.
    #[test]
    fn the_current_entry_falls_back_to_the_first() {
        let entries = library(&["Dusk", "Dawn"]);
        let named = |selected| current(&entries, selected).map(|e| e.name.as_str());
        assert_eq!(named(Some("Dawn")), Some("Dawn"));
        assert_eq!(named(None), Some("Dusk"), "nothing selected");
        assert_eq!(
            named(Some("Removed")),
            Some("Dusk"),
            "a selection orphaned by a removal"
        );
        assert_eq!(current(&[], Some("Dawn")), None);
    }
}
