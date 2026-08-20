//! This browser's local store: the one door to `localStorage`, and the one format
//! everything behind it is kept in.
//!
//! Ten records live here — the shape, preset, gradient and quick-brush libraries, the
//! ⚙ dialog's settings, the chord table, which panels are open, whether the navigator
//! is up, what the tour has seen, and this client's identity. There were six formats
//! between them once: JSON for two, a base64 field table for three, bare
//! space-separated panel names, a tagged `deed|key|count` row for the tour, `"1"`
//! versus `""` for the navigator's one bit, and hex for the identity. Each was
//! defensible where it was written and none of them was worth a reader having to learn
//! it, so they are one format now.
//!
//! # One format: JSON, through one typed door
//!
//! A record is a serde type its own module declares, and it goes in and out through
//! [`save`], [`load`] and [`load_list`]. **There is no untyped door** — [`get`] and
//! [`set`] are private, so nothing can hand this module a string it spelled itself,
//! and there is nowhere for a seventh format to come from.
//!
//! JSON for the reason `Prefs` gave first and the rest inherit: `localStorage`
//! outlives app versions, and a self-describing format reconciles a stored value
//! against today's types by *name*. A field added later reads as its default out of
//! every value stored before it existed (`#[serde(default)]`), and a field removed is
//! ignored — so the format is its own migration, which is why none of these keys
//! carries a version suffix. That is the save file's bargain (§8) at browser scale,
//! and the reason there is no schema here to keep.
//!
//! Bytes are the one thing JSON cannot hold: a shape's PNG goes through [`b64`], and
//! a content id or a secret key through [`hex`], because the store is text and 32
//! bytes read better as `3f9a…` than as a list of numbers.
//!
//! # The registry
//!
//! [`Store`] is the whole authority on where a record lives and what a warning calls
//! it. Both facts sit on one row, so a new record is one row and one serde type —
//! never a `const KEY` beside a matching string at each call site, which is what the
//! ten keys used to be (§25.6).
//!
//! # A damaged entry costs that entry
//!
//! The four libraries are lists, and [`load_list`] reads them **element by element**:
//! one entry today's build cannot make sense of is dropped and the rest of the library
//! still loads. That is what the old line table bought with its separator, kept here
//! without one, and it is what the lists lean on for every case a strict decoder would
//! call an error — a preset naming a brush field that has since gone, a binding for a
//! retired command, a panel this build no longer has.
//!
//! Records that are not lists ([`load`]) are all-or-nothing, and want to be: a
//! half-read `Prefs` is a worse answer than the defaults.
//!
//! # What is deliberately not here
//!
//! **The store itself, and the base64 codec**, both in [`crate::platform`]. That
//! module is the only one allowed to name a browser type — the compiler checks it off
//! wasm (U6) — so the `localStorage` calls behind [`get`] and [`set`] live there, and
//! this module is the *format* and the failure policy rather than the door. The codec
//! is there for a second reason on top: `platform` needs base64 itself, to read the
//! data URL the browser hands back when it re-encodes an imported brush image, so
//! owning it here would point a dependency up the stack.
//!
//! # Failure is silence, on purpose
//!
//! A browser with no storage — a private window, storage disabled — reads as a browser
//! that has stored nothing, and a write that will not fit warns and carries on. Both
//! are the same bargain [`crate::identity`] makes and states: what is lost is
//! *durability*, and the session still works to the end. Nothing here returns an error
//! for a caller to handle, because there is no handling of it that is better than
//! carrying on.

use serde::Serialize;
use serde::de::DeserializeOwned;

/// Every record this browser keeps, and the only place its key is written down.
///
/// A variant per record rather than a `const KEY` per module, because the key and the
/// name a warning calls the record by are two halves of one fact that used to be two
/// constants three lines apart — see [`Store::named`] and §25.6.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Store {
    /// The key this client's `ActorId` derives from, and its run counter
    /// (`crate::identity`).
    Identity,
    /// What the ⚙ dialog sets (`crate::prefs`).
    Prefs,
    /// The chords the user has taken over (`crate::commands`).
    Bindings,
    /// Which floating panels are open, and which are folded (`crate::layout`).
    Panels,
    /// Whether the navigator's miniature is up (§11, `crate::navigator`).
    Navigator,
    /// What the guided tour has counted and given (§24, `crate::tutor`).
    Tutor,
    /// The custom brush-shape library (§6.6, `crate::shapes`).
    Shapes,
    /// The user's own brush presets (`crate::presets`).
    Presets,
    /// The ten brushes under the hand (§18.1.8, `crate::slots`).
    Slots,
    /// The gradient library (§22.3, `crate::gradients`).
    Gradients,
}

impl Store {
    /// The `localStorage` key, and the name a warning calls this record by — "the
    /// gradient library", "the settings" — so a full quota says which of the ten ran
    /// out of room.
    ///
    /// One row per record with both facts on it: a key without a name, and a name that
    /// had drifted off its key, are exactly the two mistakes a second table three
    /// lines away made possible.
    ///
    /// The keys are namespaced because `localStorage` is shared per origin, and carry
    /// **no version suffix**: the format is self-describing and reconciles by name, so
    /// there is nothing for a suffix to gate — see the module comment.
    const fn named(self) -> (&'static str, &'static str) {
        match self {
            Store::Identity => ("stark.identity", "this browser's identity"),
            Store::Prefs => ("stark.prefs", "the settings"),
            Store::Bindings => ("stark.bindings", "the shortcuts"),
            Store::Panels => ("stark.panels", "which panels are open"),
            Store::Navigator => ("stark.navigator", "whether the navigator is showing"),
            Store::Tutor => ("stark.tutor", "the tips you have seen"),
            Store::Shapes => ("stark.shapes", "the shape library"),
            Store::Presets => ("stark.presets", "the brush presets"),
            Store::Slots => ("stark.slots", "the quick brushes"),
            Store::Gradients => ("stark.gradients", "the gradient library"),
        }
    }
}

/// What this browser has stored, or `None` where it has stored nothing — and where
/// what it stored is not readable as a `T`, which is the same answer for the same
/// reason: there is nothing here this build can act on.
pub fn load<T: DeserializeOwned>(store: Store) -> Option<T> {
    let text = get(store)?;
    match serde_json::from_str(&text) {
        Ok(value) => Some(value),
        Err(e) => {
            // Not a failure to handle — the caller's defaults are the answer — but
            // worth saying, because the visible symptom is a setting quietly back
            // where it started.
            tracing::warn!("could not read {} ({e})", store.named().1);
            None
        }
    }
}

/// A stored **list**, read entry by entry: one entry this build cannot make sense of
/// is dropped and the rest survive (see the module comment).
///
/// `None` and `Some(vec![])` are different answers and callers rely on the difference:
/// an untouched quick-brush rack is seeded from the preset library, while one the user
/// has emptied is left empty.
pub fn load_list<T: DeserializeOwned>(store: Store) -> Option<Vec<T>> {
    let list = entries(&get(store)?);
    if list.is_none() {
        tracing::warn!("could not read {}", store.named().1);
    }
    list
}

/// [`load_list`]'s reading, without the store — the half worth testing.
///
/// A value that is not a list at all is `None` rather than an empty one: an unreadable
/// record and an absent one are the same case, and the emptied-versus-never-set
/// distinction above must not be decided by damage.
fn entries<T: DeserializeOwned>(json: &str) -> Option<Vec<T>> {
    let values: Vec<serde_json::Value> = serde_json::from_str(json).ok()?;
    Some(
        values
            .into_iter()
            .filter_map(|v| serde_json::from_value(v).ok())
            .collect(),
    )
}

/// Store `value`. A store that will not take it warns and carries on ([`set`]).
pub fn save<T: Serialize + ?Sized>(store: Store, value: &T) {
    match serde_json::to_string(value) {
        Ok(json) => set(store, &json),
        Err(e) => tracing::warn!("could not encode {} ({e})", store.named().1),
    }
}

/// The untyped half, private so [`save`]/[`load`] are the only way in or out — which
/// is what makes "one format" a property of the module rather than a habit.
fn get(store: Store) -> Option<String> {
    crate::platform::local_get(store.named().0)
}

fn set(store: Store, value: &str) {
    if !crate::platform::local_set(store.named().0, value) {
        // Quota, most likely. It still works for this session; only its durability is
        // lost.
        tracing::warn!(
            "could not persist {} (storage full or unavailable)",
            store.named().1
        );
    }
}

/// The keys the six old formats were kept under, dropped once at start.
///
/// Not a migration — nothing is read out of them. It is the **quota**:
/// `stark.shapes.v1` holds a base64 PNG per imported shape, and megabytes stranded
/// under a name nothing will ever read again come out of the same few MB the new
/// library has to fit in. The symptom would be shape imports that silently stop
/// persisting, which is the hardest kind of storage bug to attribute.
///
/// **Delete this, and `platform::local_remove` with it, once no browser plausibly
/// carries the old keys.** It is dead weight the day that is true.
pub fn drop_retired() {
    for key in [
        "stark.prefs.v1",
        "stark.bindings.v1",
        "stark.panels.v1",
        "stark.navigator.v1",
        "stark.tutor.v1",
        "stark.shapes.v1",
        "stark.presets.v1",
        "stark.slots.v1",
        "stark.gradients.v1",
        "stark.identity.secret",
        "stark.identity.boot",
    ] {
        crate::platform::local_remove(key);
    }
}

/// A 32-byte value as lowercase hex, for `#[serde(with = "crate::storage::hex")]`.
///
/// How Stark spells a content id everywhere else (`AssetId::to_hex`, its `Display`,
/// the build script's literals) and the conventional rendering of a key — so the one
/// thing in a stored record somebody might read by eye reads the same here as in a log
/// line. The derived form would be a list of 32 numbers.
pub mod hex {
    use serde::{Deserialize, Deserializer, Serializer, de::Error};

    pub fn serialize<S, T>(value: &T, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        T: Copy + Into<[u8; 32]>,
    {
        let bytes: [u8; 32] = (*value).into();
        let text: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
        s.serialize_str(&text)
    }

    pub fn deserialize<'de, D, T>(d: D) -> Result<T, D::Error>
    where
        D: Deserializer<'de>,
        T: From<[u8; 32]>,
    {
        let text = String::deserialize(d)?;
        // Bytes, not chars: 64 of the first need not be 64 of the second, so the
        // slicing below is fallible even past this check.
        if text.len() != 64 {
            return Err(D::Error::custom("expected 64 hex characters"));
        }
        let mut bytes = [0u8; 32];
        for (i, byte) in bytes.iter_mut().enumerate() {
            let pair = text
                .get(i * 2..i * 2 + 2)
                .ok_or_else(|| D::Error::custom("expected hex"))?;
            *byte = u8::from_str_radix(pair, 16).map_err(D::Error::custom)?;
        }
        Ok(T::from(bytes))
    }
}

/// Bytes as base64, for `#[serde(with = "crate::storage::b64")]` — the store is text,
/// so a blob has to be spelled in it. The codec is [`crate::platform`]'s; see the
/// module comment for why it lives there.
pub mod b64 {
    use serde::{Deserialize, Deserializer, Serializer, de::Error};

    pub fn serialize<S: Serializer>(bytes: &[u8], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&crate::platform::base64_encode(bytes))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let text = String::deserialize(d)?;
        crate::platform::base64_decode(&text).map_err(D::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use std::collections::HashSet;

    const ALL: [Store; 10] = [
        Store::Identity,
        Store::Prefs,
        Store::Bindings,
        Store::Panels,
        Store::Navigator,
        Store::Tutor,
        Store::Shapes,
        Store::Presets,
        Store::Slots,
        Store::Gradients,
    ];

    #[derive(Debug, PartialEq, Deserialize, Serialize)]
    struct Entry {
        name: String,
        n: u32,
    }

    /// Every record has its own key and its own name, and every key is namespaced.
    /// The one thing a tenth row added in a hurry gets wrong is a key spelled twice,
    /// and the symptom is two records overwriting each other.
    #[test]
    fn every_store_has_its_own_key_and_its_own_name() {
        let keys: HashSet<&str> = ALL.iter().map(|s| s.named().0).collect();
        let names: HashSet<&str> = ALL.iter().map(|s| s.named().1).collect();
        assert_eq!(
            keys.len(),
            ALL.len(),
            "a key spelled twice is a record lost"
        );
        assert_eq!(names.len(), ALL.len());
        assert!(ALL.iter().all(|s| s.named().0.starts_with("stark.")));
    }

    /// A retired key that is still in use would delete a live record on every start —
    /// the one way [`drop_retired`] can do harm, and the one worth ruling out.
    #[test]
    fn no_retired_key_is_a_live_one() {
        let live: HashSet<&str> = ALL.iter().map(|s| s.named().0).collect();
        for key in ["stark.prefs.v1", "stark.shapes.v1", "stark.identity.secret"] {
            assert!(!live.contains(key));
        }
        assert!(ALL.iter().all(|s| !s.named().0.ends_with(".v1")));
    }

    /// The rule the list format exists for: an entry nobody can read costs that entry.
    #[test]
    fn a_damaged_entry_costs_one_entry_and_not_the_list() {
        let json = r#"[
            {"name":"a","n":1},
            "not an entry",
            {"name":"b","n":2},
            {"name":"c","n":"not a number"},
            {"n":4},
            {"name":"d","n":4}
        ]"#;
        assert_eq!(
            entries::<Entry>(json),
            Some(vec![
                Entry {
                    name: "a".into(),
                    n: 1
                },
                Entry {
                    name: "b".into(),
                    n: 2
                },
                Entry {
                    name: "d".into(),
                    n: 4
                },
            ]),
            "the three readable entries survive the three that are not"
        );
    }

    /// A record that is not a list at all is nothing stored, not an empty list — the
    /// distinction the quick-brush rack seeds itself on.
    #[test]
    fn damage_is_not_an_empty_list() {
        assert_eq!(entries::<Entry>("{}"), None);
        assert_eq!(entries::<Entry>("garbage"), None);
        assert_eq!(entries::<Entry>("[]"), Some(vec![]));
    }

    #[derive(Debug, PartialEq, Deserialize, Serialize)]
    struct Blobs {
        #[serde(with = "hex")]
        id: [u8; 32],
        #[serde(with = "b64")]
        bytes: Vec<u8>,
    }

    #[test]
    fn bytes_round_trip_through_their_spellings() {
        let value = Blobs {
            id: std::array::from_fn(|i| i as u8),
            bytes: vec![0, 1, 2, 250, 255],
        };
        let json = serde_json::to_string(&value).unwrap();
        assert!(json.contains("\"000102"), "the id reads as hex: {json}");
        assert_eq!(serde_json::from_str::<Blobs>(&json).unwrap(), value);
    }

    /// A hex field that is not 32 bytes is refused rather than padded — which, inside
    /// a list, costs its own entry and nothing else.
    #[test]
    fn a_short_id_is_refused() {
        let json = r#"{"id":"00ff","bytes":""}"#;
        assert!(serde_json::from_str::<Blobs>(json).is_err());
    }
}
