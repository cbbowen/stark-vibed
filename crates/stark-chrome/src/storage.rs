//! This browser's local store: the one door to `localStorage`, and the one format
//! everything behind it is kept in.
//!
//! Ten records live here — the shape, preset, gradient and quick-brush libraries,
//! the ⚙ dialog's settings, the chord table, the drag table, what is on screen, what
//! the tour has seen, and this client's identity.
//! There were six formats
//! between them once: JSON for two, a base64 field table for three, bare
//! space-separated panel names, a tagged `deed|key|count` row for the tour, `"1"`
//! versus `""` for the navigator's one bit, and hex for the identity. Each was
//! defensible where it was written and none of them was worth a reader having to learn
//! it, so they are one format now.
//!
//! # One format: JSON, through one typed door
//!
//! A record is a serde type its own module declares, and it goes in and out through
//! [`save`], [`load`], [`save_list`] and [`load_list`]. **There is no untyped door** —
//! [`get`] and [`set`] are private, so nothing can hand this module a string it spelled
//! itself, and there is nowhere for a seventh format to come from.
//!
//! Nor is there an untyped *key*. A type declares which record it is by implementing
//! [`Record`] or [`Entry`], and the four functions ask the type rather than taking a
//! [`Store`] argument — so the type and the key are one choice instead of two
//! agreeing ones, and `load::<Prefs>()` cannot be pointed at the chord table. Which
//! trait a type implements decides how it is read, too: see [`Entry`] for why that is
//! not one trait with a flag.
//!
//! JSON for the reason `Prefs` gave first and the rest inherit: `localStorage`
//! outlives app versions, and a self-describing format reconciles a stored value
//! against today's types by *name*. A field added later reads as its default out of
//! every value stored before it existed (`#[serde(default)]`), and a field removed is
//! ignored — so the format is its own migration, which is why none of these keys
//! carries a version suffix. That is the save file's bargain (§8) at browser scale,
//! and the reason there is no schema here to keep.
//!
//! A content id or a secret key goes through [`hex`], because the store is text and
//! 32 bytes read better as `3f9a…` than as a list of numbers.
//!
//! # Bytes are not kept here at all
//!
//! `localStorage` is text, and **~5 MB of it per origin shared across all ten
//! records**. A brush shape's PNG went in it once, base64'd inline in the shape
//! library's rows: two of the app's own stamps are 408 KB and 226 KB on disk, half as
//! much again as base64, and twice *that* against the quota in an engine that counts
//! a JS string's UTF-16. Five or ten imports filled the origin — and what a full
//! origin breaks is not the shape library, it is [`set`], for `Prefs` and the chord
//! table and the tour's ledger and `Identity` alike. Every standing choice this
//! browser has made stops persisting, silently, because somebody imported a brush.
//!
//! So bytes live in **IndexedDB, keyed by the content id that names them** — the
//! second door below ([`Blob`], [`blob_load_all`], [`blob_save`], [`blob_remove`]).
//! It is quota'd against the disk rather than against the settings, and it is
//! asynchronous, which is the other half of what was wrong: `save_list` re-encodes a
//! whole library per change, and it was doing that on the thread the canvas paints on.
//!
//! Content-addressing is what keeps that second door small. An id *names* its bytes
//! (§19), so a write is idempotent, a re-import is free, there is no invalidation to
//! get wrong and no schema to migrate — which is exactly why the argument for JSON
//! above does not reach it. There is nothing in a blob store to reconcile by name.
//! A record's rows and its blobs are the two halves of one library, and the writing
//! order is what holds them together: **blob first, then the row; row first, then the
//! blob.** A row that exists has its bytes, whatever a crash lands in the middle of.
//!
//! # The registry
//!
//! [`Store`] is the whole authority on where a record lives and what a warning calls
//! it. Both facts sit on one row, so a new record is one row, one serde type and the
//! one-line impl that pairs them — never a `const KEY` beside a matching string at each
//! call site, which is what the eleven keys used to be (§25.6).
//!
//! The impls name a variant rather than restating its strings, which is what keeps the
//! map readable in one place: eleven impls each spelling their own key would scatter the
//! answer to "what does this browser keep?" across ten modules, and nothing would
//! notice two of them colliding. `every_record_claims_one_store` does.
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
//! **Both stores themselves.** This module is the *format*, the key and the failure
//! policy; where the bytes actually go is a [`Backend`], which each frontend installs
//! once at startup ([`install`]). The web one is `localStorage` and IndexedDB behind
//! its `platform` module — the only module there allowed to name a browser type; the
//! native one is two directories.
//!
//! Six methods, because that is exactly how many doors the format needed: three over
//! text and three over bytes. Nothing was designed for this — the trait is the six
//! calls this file was already funnelling to, lifted verbatim.
//!
//! **The base64 codec**, in the web frontend's `platform` and not used here for
//! anything: it is what reads the data URL the browser hands back when it re-encodes
//! an imported brush image, so owning it here would point a dependency up the stack —
//! and now that a blob is bytes all the way down, this module has nothing to spell
//! in it.
//!
//! # Failure is silence, on purpose
//!
//! A browser with no storage — a private window, storage disabled — reads as a browser
//! that has stored nothing, and a write that will not fit warns and carries on. Both
//! are the same bargain [`identity`](crate::identity) makes and states: what is lost
//! is *durability*, and the session still works to the end. Nothing here returns an
//! error for a caller to handle, because there is no handling of it that is better
//! than carrying on.
//!
//! **A frontend that installs no backend is that same case**, which is why [`install`]
//! is not required and no call here fails without one. It is how a test runs, and how
//! a frontend that has not grown persistence yet behaves — every read answers "nothing
//! stored", every write warns.

use serde::Serialize;
use serde::de::DeserializeOwned;
use stark_model::AssetId;

/// Every record this browser keeps, and the only place its key is written down.
///
/// A variant per record rather than a `const KEY` per module, because the key and the
/// name a warning calls the record by are two halves of one fact that used to be two
/// constants three lines apart — see [`Store::named`] and §25.6.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Store {
    /// The key this client's `ActorId` derives from, and its run counter
    /// (`crate::identity`).
    Identity,
    /// What the ⚙ dialog sets (`crate::prefs`).
    Prefs,
    /// The chords the user has taken over (`crate::commands`).
    Bindings,
    /// The canvas drags the user has taken over, and whether this browser has been
    /// offered a preset table (§25.8, `crate::drags`).
    Drags,
    /// What is on screen: which panels are open and folded, whether the navigator's
    /// miniature is up, whether the quick-brush rack is pinned, whether Timeline mode
    /// is on (§11, `crate::visibility`).
    Visible,
    /// What the guided tour has counted and given (§24, `crate::tutor`).
    Tutor,
    /// The custom brush-shape library (§6.6, `crate::shapes`).
    Shapes,
    /// The custom canvas-substrate library (§6.4, `crate::substrates`) — [`Shapes`]'s
    /// sibling in every respect, rows here and height maps in the blob store beside
    /// them (`stark_chrome::library`).
    ///
    /// **Its key is still `stark.grounds`.** A key is browser-local data, not
    /// vocabulary: renaming one silently orphans every library already saved under
    /// it, and there is nothing to gain that is worth a user's imported substrates.
    /// The same holds for every other row here.
    ///
    /// [`Shapes`]: Self::Shapes
    Substrates,
    /// The user's own brush presets (`crate::presets`).
    Presets,
    /// The ten brushes under the hand (§18.1.8, `crate::slots`).
    ///
    /// **Its key is `stark.quick`, and was `stark.slots`** — the one renaming
    /// this registry has made, and made for the opposite of the usual reason. A
    /// slot used to be stored as a brush snapshot and is stored as a binding now
    /// (`slots::QuickBrush`): every old row is unreadable by this build, so
    /// nothing was orphaned that could have been read. What the new key buys is
    /// the *seed*. Under the old one a rack of ten unreadable rows loads as
    /// `Some(empty)` — "set to nothing", which is left alone — and every browser
    /// that had ever set a slot would open on an empty rack for good. Under a
    /// fresh key it loads as "never set" and is seeded from the library
    /// (`slots::seed_defaults`). The old key is dropped at start
    /// ([`drop_retired`]).
    Slots,
    /// The gradient library (§22.3, a frontend's `gradients`).
    Gradients,
    /// Where the window was and how big — **the native frontend's alone** (§11.1).
    ///
    /// A browser has no such thing to keep: a page is where the user put the tab. It
    /// is a row of this registry all the same, because the registry's job is that a
    /// key is written down once, and a second enum in the native frontend would be a
    /// second place for one to collide from.
    Window,
}

impl Store {
    /// Every row, and the only place the list is written down.
    ///
    /// Public because the check that every row is *claimed* cannot live here any
    /// more: most record types are a frontend's, so that test is too — see
    /// `stark-dioxus-frontend`'s `records`.
    pub const ALL: [Store; 12] = [
        Store::Identity,
        Store::Prefs,
        Store::Bindings,
        Store::Drags,
        Store::Visible,
        Store::Tutor,
        Store::Shapes,
        Store::Substrates,
        Store::Presets,
        Store::Slots,
        Store::Gradients,
        Store::Window,
    ];
}

impl Store {
    /// The key, and the name a warning calls this record by — "the gradient library",
    /// "the settings" — so a full quota says which of the eleven ran out of room.
    ///
    /// One key, both stores: a record that keeps bytes as well as rows spells its blob
    /// keys `stark.shapes/<hex>` (see [`Blob`]), so there is still exactly one place
    /// the answer to "where does this record live" is written down.
    ///
    /// One row per record with both facts on it: a key without a name, and a name that
    /// had drifted off its key, are exactly the two mistakes a second table three
    /// lines away made possible.
    ///
    /// Public because a [`Backend`] may need it: the native one turns a key into a
    /// path, and the name is what a warning about a full store would print.
    ///
    /// The keys are namespaced because `localStorage` is shared per origin, and carry
    /// **no version suffix**: the format is self-describing and reconciles by name, so
    /// there is nothing for a suffix to gate — see the module comment.
    pub const fn named(self) -> (&'static str, &'static str) {
        match self {
            Store::Identity => ("stark.identity", "this browser's identity"),
            Store::Prefs => ("stark.prefs", "the settings"),
            Store::Bindings => ("stark.bindings", "the shortcuts"),
            Store::Drags => ("stark.drags", "the drag bindings"),
            Store::Visible => ("stark.visible", "what is on screen"),
            Store::Tutor => ("stark.tutor", "the tips you have seen"),
            Store::Shapes => ("stark.shapes", "the shape library"),
            Store::Substrates => ("stark.grounds", "the substrate library"),
            Store::Presets => ("stark.presets", "the brush presets"),
            Store::Slots => ("stark.quick", "the quick brushes"),
            Store::Gradients => ("stark.gradients", "the gradient library"),
            Store::Window => ("stark.window", "the window's place"),
        }
    }
}

/// A type that is **the whole of** what one record holds — read and written in one
/// piece ([`load`], [`save`]).
///
/// The impl is what binds a type to its key, and it is the reason neither is a
/// parameter: `load` and `save` take no [`Store`], they ask the type which one it is.
/// So there is no call site at which the type and the key are two separate choices,
/// and reading `Prefs` out of the chord table is not a mistake that can be written
/// down.
pub trait Record {
    /// Which record this type is.
    const STORE: Store;
}

/// A type that is **one entry of** a record that is a list — read and written entry by
/// entry ([`load_list`], [`save_list`]).
///
/// A second trait rather than a flag on [`Record`], because the two are read
/// differently and the difference is not one a caller should be able to get wrong:
/// nine of the eleven records are lists, and `load::<StoredVisible>()` under one trait
/// would compile and quietly answer `None` — an array is not an object — leaving a
/// screen that silently forgot itself. A type is one or the other, and the
/// compiler says which functions it is for.
pub trait Entry {
    /// Which record this type is an entry of.
    const STORE: Store;
}

/// What this browser has stored, or `None` where it has stored nothing — and where
/// what it stored is not readable as a `T`, which is the same answer for the same
/// reason: there is nothing here this build can act on.
pub fn load<T: Record + DeserializeOwned>() -> Option<T> {
    let text = get(T::STORE)?;
    match serde_json::from_str(&text) {
        Ok(value) => Some(value),
        Err(e) => {
            // Not a failure to handle — the caller's defaults are the answer — but
            // worth saying, because the visible symptom is a setting quietly back
            // where it started.
            tracing::warn!("could not read {} ({e})", T::STORE.named().1);
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
pub fn load_list<T: Entry + DeserializeOwned>() -> Option<Vec<T>> {
    let list = entries(&get(T::STORE)?);
    if list.is_none() {
        tracing::warn!("could not read {}", T::STORE.named().1);
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

/// Store `value` as the whole of its record. A store that will not take it warns and
/// carries on ([`set`]).
pub fn save<T: Record + Serialize>(value: &T) {
    write(T::STORE, value);
}

/// Store `entries` as the whole of their record — [`load_list`]'s counterpart, and a
/// slice rather than a `Vec` because every caller is already holding one.
pub fn save_list<T: Entry + Serialize>(entries: &[T]) {
    write(T::STORE, entries);
}

fn write<T: Serialize + ?Sized>(store: Store, value: &T) {
    match serde_json::to_string(value) {
        Ok(json) => set(store, &json),
        Err(e) => tracing::warn!("could not encode {} ({e})", store.named().1),
    }
}

/// A future this crate can hold without knowing whose executor will poll it.
///
/// **Not `Send`**, deliberately: the web backend's futures are IndexedDB requests
/// bridged from JavaScript and cannot be, and requiring it here would make the one
/// backend that must exist impossible to write.
pub type Stored<'a, T> = std::pin::Pin<Box<dyn std::future::Future<Output = T> + 'a>>;

/// Where a record's bytes actually go — the one thing this module does not decide.
///
/// Six methods, three over text and three over bytes, which is exactly the surface
/// this file was already calling into the web frontend for. A frontend implements it
/// once and hands it to [`install`].
///
/// Every method's failure is the caller's silence, not an error: see the module's
/// "Failure is silence" note. `set` and `blob_put` answer `false` where the store
/// refused them, which is all this module does anything with.
pub trait Backend: Send + Sync + 'static {
    /// The text stored under `key`, or `None` where there is none — and where what is
    /// there cannot be read, which is the same answer.
    fn get(&self, key: &str) -> Option<String>;
    /// Store `value` under `key`; `false` if the store would not take it.
    fn set(&self, key: &str, value: &str) -> bool;
    /// Forget `key`. A key that was never stored is not an error.
    fn remove(&self, key: &str);
    /// The bytes for each of `keys`, in that order — see [`blob_load_all`] for why
    /// this is plural.
    fn blob_get_many<'a>(&'a self, keys: &'a [String]) -> Stored<'a, Vec<Option<Vec<u8>>>>;
    /// Store `bytes` under `key`; `false` if the store would not take them.
    fn blob_put<'a>(&'a self, key: &'a str, bytes: &'a [u8]) -> Stored<'a, bool>;
    /// Drop the bytes under `key`.
    fn blob_delete<'a>(&'a self, key: &'a str) -> Stored<'a, ()>;
}

/// The installed backend, or `None` where a frontend has not given one.
static BACKEND: std::sync::OnceLock<Box<dyn Backend>> = std::sync::OnceLock::new();

/// Give this process its store. Call once, before anything reads a record.
///
/// A second call is ignored rather than a panic: the loser is a store nothing has
/// read through yet, and taking the app down over it would trade a bug that costs
/// nothing for one that costs the session. It answers whether this call was the one
/// that installed, so a caller that cares can say so.
pub fn install(backend: impl Backend) -> bool {
    BACKEND.set(Box::new(backend)).is_ok()
}

fn backend() -> Option<&'static dyn Backend> {
    BACKEND.get().map(AsRef::as_ref)
}

/// The untyped half, private so [`save`]/[`load`] are the only way in or out — which
/// is what makes "one format" a property of the module rather than a habit.
fn get(store: Store) -> Option<String> {
    backend()?.get(store.named().0)
}

fn set(store: Store, value: &str) {
    if !backend().is_some_and(|b| b.set(store.named().0, value)) {
        // Quota, most likely — or no backend, which is the same case from here. It
        // still works for this session; only its durability is lost.
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
        // The two records `Store::Visible` replaced. What is on screen used to be
        // kept in two, which is how the quick-brush rack and Timeline mode came to
        // be kept in none (`crate::visibility`).
        "stark.panels",
        "stark.navigator",
        // The rack as brush snapshots, before a slot became a binding to a preset
        // — see `Store::Slots` for why the key moved rather than the rows.
        "stark.slots",
    ] {
        if let Some(b) = backend() {
            b.remove(key);
        }
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

// --- the blob store --------------------------------------------------------

/// A type whose **bytes** this browser keeps beside its rows, each blob under the
/// content id that names it.
///
/// A third trait rather than a fourth pair of functions taking a [`Store`], for the
/// reason [`Record`] and [`Entry`] are traits: the type declares which record its
/// bytes belong to, so the key is not a second choice a call site could get wrong.
/// It is implemented *alongside* one of the other two — `ShapeEntry` is an [`Entry`]
/// for its row and a [`Blob`] for its PNG, which is what says the two halves are one
/// record and not two.
pub trait Blob {
    /// Which record these bytes belong to.
    const STORE: Store;
}

/// Where one blob lives: the record's key, then the id, so everything a record owns
/// sorts together and a second blob record is a second prefix rather than a schema
/// change (`platform::BLOB_DB`).
fn blob_key<T: Blob>(id: AssetId) -> String {
    format!("{}/{}", T::STORE.named().0, id.to_hex())
}

/// The bytes for each of `ids`, in that order — `None` where this browser has none.
///
/// Plural because it is one exchange with the store: the whole library is read at
/// start, and a door taking one id would make that N opens and N transactions. The
/// positional answer is what lets a caller zip it back onto the rows it read them from.
///
/// A missing blob is not an error here for the same reason a damaged row is not: it
/// costs that entry, and the caller is the one that says so. IndexedDB is evictable
/// under storage pressure, so "the row is here and the bytes are gone" is a state
/// that really happens rather than one that only follows a crash — `shapes::load`
/// drops such a row and writes the library back without it.
pub async fn blob_load_all<T: Blob>(ids: &[AssetId]) -> Vec<Option<Vec<u8>>> {
    let keys: Vec<String> = ids.iter().map(|&id| blob_key::<T>(id)).collect();
    match backend() {
        Some(b) => b.blob_get_many(&keys).await,
        // No store is the same answer as an empty one: the caller drops the rows
        // whose bytes did not come back, which is what it already does for a blob
        // the store evicted.
        None => vec![None; ids.len()],
    }
}

/// Store `bytes` under `id`. A store that will not take them warns and carries on,
/// exactly as [`set`] does — and for the same reason: what is lost is durability, and
/// the session still works to the end.
///
/// Write this **before** the row that names it. A crash between the two then leaves a
/// blob nothing points at, which costs some bytes; the other order leaves a row whose
/// shape has no picture and cannot be painted with.
pub async fn blob_save<T: Blob>(id: AssetId, bytes: &[u8]) {
    let stored = match backend() {
        Some(b) => b.blob_put(&blob_key::<T>(id), bytes).await,
        None => false,
    };
    if !stored {
        tracing::warn!(
            "could not persist an entry of {} (storage full or unavailable)",
            T::STORE.named().1
        );
    }
}

/// Drop the bytes stored under `id` — **after** the row that named them, per
/// [`blob_save`].
pub async fn blob_remove<T: Blob>(id: AssetId) {
    if let Some(b) = backend() {
        b.blob_delete(&blob_key::<T>(id)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use std::collections::HashSet;

    const ALL: [Store; 12] = Store::ALL;

    #[derive(Debug, PartialEq, Deserialize, Serialize)]
    struct Item {
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
        for key in [
            "stark.prefs.v1",
            "stark.shapes.v1",
            "stark.identity.secret",
            "stark.panels",
            "stark.navigator",
            "stark.slots",
        ] {
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
            entries::<Item>(json),
            Some(vec![
                Item {
                    name: "a".into(),
                    n: 1
                },
                Item {
                    name: "b".into(),
                    n: 2
                },
                Item {
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
        assert_eq!(entries::<Item>("{}"), None);
        assert_eq!(entries::<Item>("garbage"), None);
        assert_eq!(entries::<Item>("[]"), Some(vec![]));
    }

    #[derive(Debug, PartialEq, Deserialize, Serialize)]
    struct Named {
        #[serde(with = "hex")]
        id: [u8; 32],
    }

    #[test]
    fn an_id_round_trips_through_its_spelling() {
        let value = Named {
            id: std::array::from_fn(|i| i as u8),
        };
        let json = serde_json::to_string(&value).unwrap();
        assert!(json.contains("\"000102"), "the id reads as hex: {json}");
        assert_eq!(serde_json::from_str::<Named>(&json).unwrap(), value);
    }

    /// A hex field that is not 32 bytes is refused rather than padded — which, inside
    /// a list, costs its own entry and nothing else.
    #[test]
    fn a_short_id_is_refused() {
        let json = r#"{"id":"00ff"}"#;
        assert!(serde_json::from_str::<Named>(json).is_err());
    }

    /// A blob's key is its record's key and then the id — so the two halves of one
    /// library sort together, and no record can reach into another's bytes.
    #[test]
    fn a_blob_is_keyed_under_its_own_record() {
        let id = AssetId::from([0xabu8; 32]);
        // A stand-in for the shape library's entry: what is under test is the key's
        // shape, which is this module's, and the real type is a frontend's.
        struct Stamps;
        impl Blob for Stamps {
            const STORE: Store = Store::Shapes;
        }
        let key = blob_key::<Stamps>(id);
        assert_eq!(key, format!("stark.shapes/{}", id.to_hex()));
        assert!(
            ALL.iter()
                .filter(|s| **s != Store::Shapes)
                .all(|s| !key.starts_with(s.named().0)),
            "one record's blobs are not in reach of another's key"
        );
    }
}
