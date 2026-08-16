//! This browser's local store: the one door to `localStorage`, and the
//! line-oriented table the libraries are kept in.
//!
//! Six modules keep something here — the shape, preset, gradient and quick-brush
//! libraries, the ⚙ dialog's settings, and this client's identity — and each of
//! them used to open the door itself. `fn storage()` was written out six times,
//! and four of those modules carried a `persist`/`read_storage`/`parse_entry`
//! triple that differed in nothing but the record's fields: the same `\n`-separated
//! lines, the same "skip a damaged line rather than poison the library" rule stated
//! in four module comments, the same warning when the quota is full.
//!
//! # The table format
//!
//! One key holds one line per entry, and each line is its fields joined by
//! [`FIELD`]. Every field is base64 or hex, so neither the separator nor a newline
//! can occur inside one — which is the whole reason the format can be split before
//! it is understood, and therefore the reason a **single damaged line costs one
//! entry rather than the library**. That rule lives in [`load_table`] now, so it is
//! one thing to hold and one thing to test.
//!
//! # What is deliberately not here
//!
//! **The base64 codec stays in [`crate::platform`]**, though this is its main
//! caller. `platform` is the bottom layer — it is what this module opens the store
//! through — and it needs base64 itself, to read the data URL the browser hands
//! back when it re-encodes an imported brush image. A codec owned here would be a
//! dependency pointing the wrong way down the stack.
//!
//! # Failure is silence, on purpose
//!
//! A browser with no storage — a private window, storage disabled — reads as a
//! browser that has stored nothing, and a write that will not fit warns and carries
//! on. Both are the same bargain [`crate::identity`] makes and states: what is lost
//! is *durability*, and the session still works to the end. Nothing here returns an
//! error for a caller to handle, because there is no handling of it that is better
//! than carrying on.

/// The character separating a record's fields. Stated once because four modules
/// split on it and three build records with it — see [`record`].
pub const FIELD: char = '|';

/// The store, or `None` where the browser has none to offer.
fn store() -> Option<web_sys::Storage> {
    web_sys::window()?.local_storage().ok().flatten()
}

/// What is stored under `key`, or `None` if nothing is (or there is no store).
pub fn get(key: &str) -> Option<String> {
    store()?.get_item(key).ok().flatten()
}

/// Store `value` under `key`. `what` names the thing for the one warning — "the
/// gradient library", "the settings" — so the message says which of the six ran out
/// of room.
pub fn set(key: &str, what: &str, value: &str) {
    let Some(store) = store() else { return };
    if store.set_item(key, value).is_err() {
        // Quota, most likely. It still works for this session; only its durability
        // is lost.
        tracing::warn!("could not persist {what} (storage full or unavailable)");
    }
}

/// One stored record: `fields` joined by [`FIELD`].
pub fn record<'a>(fields: impl IntoIterator<Item = &'a str>) -> String {
    let mut out = String::new();
    for field in fields {
        if !out.is_empty() {
            out.push(FIELD);
        }
        out.push_str(field);
    }
    out
}

/// Store `rows` as the table under `key`, one row per line.
pub fn save_table(key: &str, what: &str, rows: impl IntoIterator<Item = String>) {
    let rows: Vec<String> = rows.into_iter().collect();
    set(key, what, &rows.join("\n"));
}

/// The table under `key`, each line through `parse`, or `None` where this browser
/// has stored nothing at all.
///
/// **A line `parse` rejects is dropped, and the rest of the table still loads** —
/// the property the format exists for. `None` and `Some(vec![])` are different
/// answers and callers rely on the difference: an untouched quick-brush rack is
/// seeded with defaults, while one the user has emptied is left empty.
pub fn load_table<T>(key: &str, parse: impl FnMut(&str) -> Option<T>) -> Option<Vec<T>> {
    let text = get(key)?;
    Some(text.lines().filter_map(parse).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_record_is_its_fields_between_separators() {
        assert_eq!(record(["a", "b", "c"]), "a|b|c");
        assert_eq!(record(["only"]), "only");
        assert_eq!(record(std::iter::empty()), "");
    }

    /// The rule the whole format is for, asked of the parser rather than of a
    /// store: a line nobody can read costs that line.
    #[test]
    fn a_damaged_line_costs_one_entry_and_not_the_table() {
        let parse = |line: &str| {
            let mut fields = line.split(FIELD);
            let name = fields.next()?.to_string();
            let n: u32 = fields.next()?.parse().ok()?;
            Some((name, n))
        };
        let text = "a|1\nbroken\nb|2\nc|not-a-number\nd|4";
        let kept: Vec<_> = text.lines().filter_map(parse).collect();
        assert_eq!(
            kept,
            vec![("a".into(), 1), ("b".into(), 2), ("d".into(), 4)],
            "the three readable entries survive the two that are not"
        );
    }
}
