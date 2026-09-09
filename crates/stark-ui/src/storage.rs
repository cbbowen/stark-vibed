//! This browser's local store: the one door to `localStorage`, and the one format
//! everything behind it is kept in.
//!
//! [`Store`] is the whole list of records — the shape, substrate, preset, gradient and
//! quick-brush libraries, the ⚙ dialog's settings, the chord table, the drag table,
//! what is on screen, what the tour has seen, this client's identity, and where the
//! native frontend's window was. There were six formats between them once: JSON for
//! two, a base64 field table for three, bare space-separated panel names, a tagged
//! `deed|key|count` row for the tour, `"1"` versus `""` for the navigator's one bit,
//! and hex for the identity. Each was defensible where it was written and none of them
//! was worth a reader having to learn it, so they are one format now.
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
//! `localStorage` is text, and **~5 MB of it per origin shared across every record
//! here**. A brush shape's PNG went in it once, base64'd inline in the shape
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
//! call site, which is what every one of these keys used to be (§25.6).
//!
//! The impls name a variant rather than restating its strings, which is what keeps the
//! map readable in one place: an impl per record spelling its own key would scatter the
//! answer to "what does this browser keep?" across as many modules, and nothing would
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
//! **The drop is counted and named.** It is the visible symptom of every mistake a
//! record can make here, and while it was silent the symptom pointed at nothing: the
//! user's library opened empty and the log said so nowhere. What rules the mistake
//! out rather than reporting it is [`every_field_may_be_absent`], which each frontend
//! runs over the records it keeps.
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
//!
//! It is *also* what a frontend that installs one too late looks like from in here,
//! and those two are worth telling apart: the first is a decision and the second is a
//! bug that costs the user a setting. Neither can be told from the answer, so the
//! first read with no backend says so once ([`warn_no_store`]).

use serde::Serialize;
use serde::de::DeserializeOwned;
use stark_model::AssetId;

/// Every record this browser keeps, and the only place its key is written down.
///
/// A variant per record rather than a `const KEY` per module, because the key and the
/// name a warning calls the record by are two halves of one fact that used to be two
/// constants three lines apart — see [`Store::named`] and §25.6.
///
/// The list is `VariantArray::VARIANTS`, derived rather than kept beside the enum —
/// which is what the check that every row is actually *claimed* walks. That check
/// cannot live here: most record types are a frontend's, so it is
/// `stark-dioxus-frontend`'s `records`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, strum::VariantArray)]
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
    /// them (`stark_ui::library`).
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
    /// The key, and the name a warning calls this record by — "the gradient library",
    /// "the settings" — so a full quota says which record ran out of room.
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
/// most records are lists, and `load::<StoredVisible>()` under one trait
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
    let Some((list, dropped)) = entries(&get(T::STORE)?) else {
        tracing::warn!("could not read {}", T::STORE.named().1);
        return None;
    };
    if let Some(first) = dropped.first {
        // Named and counted, because the drop is the visible symptom of every mistake
        // this format can make (§25.6) and it used to be silent — a library that
        // opens empty with nothing in the log to attribute it to. `assets::load`
        // reports its own dropped rows this way for the same reason.
        tracing::warn!(
            "{} of {} could not be read and were dropped ({first})",
            match dropped.count {
                1 => "one entry".to_string(),
                n => format!("{n} entries"),
            },
            T::STORE.named().1,
        );
    }
    Some(list)
}

/// What [`entries`] could not read on its way through a list: how many, and why the
/// first of them would not.
///
/// The first reason rather than all of them: a list damaged in one way is damaged in
/// it throughout — a field added without a default drops every row with the same
/// error — so the second message onwards is the first repeated per entry.
#[derive(Default, Debug, PartialEq)]
struct Dropped {
    count: usize,
    first: Option<String>,
}

/// [`load_list`]'s reading, without the store — the half worth testing.
///
/// A value that is not a list at all is `None` rather than an empty one: an unreadable
/// record and an absent one are the same case, and the emptied-versus-never-set
/// distinction above must not be decided by damage.
fn entries<T: DeserializeOwned>(json: &str) -> Option<(Vec<T>, Dropped)> {
    let values: Vec<serde_json::Value> = serde_json::from_str(json).ok()?;
    let mut kept = Vec::with_capacity(values.len());
    let mut dropped = Dropped::default();
    for value in values {
        match serde_json::from_value(value) {
            Ok(entry) => kept.push(entry),
            Err(e) => {
                dropped.count += 1;
                dropped.first.get_or_insert_with(|| e.to_string());
            }
        }
    }
    Some((kept, dropped))
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

// --- the compatibility gate ------------------------------------------------

/// Every field of a record may be absent — the property `#[serde(default)]` buys.
/// `exempt` names the fields whose absence is deliberately fatal, and is the one
/// place such an exception is written down.
///
/// # What it is for
///
/// The module's bargain is that a field added later reads as its default out of every
/// value stored before it existed. Nothing enforced that but a reviewer remembering,
/// and it has been forgotten four times. The symptom is always the same and never
/// points at the cause: [`load_list`] drops the entry, [`load`] drops the record, and
/// the user opens Stark with an empty library.
///
/// **Every field at every depth**, named by path (`modulation.size.floor`), because
/// the field that gets added is as likely to be on a type a record *holds* as on the
/// record — and a check of the top level alone would have to keep a hand-written list
/// of what to recurse into, which is the reviewer's memory again under a new name.
/// Array elements are walked too (`stops[0].t`), so a list a record carries is not a
/// hole either.
///
/// `exempt` cuts a whole subtree: exempting `what` covers `what.Panel` under it. So an
/// exemption is also a **hole where the type below it is not walked**, which is the
/// cost of granting one and is worth saying at the call site.
///
/// The walk sees what the sample holds and nothing else — an `Option` left `None` is a
/// type that goes unchecked, and a `Vec` left empty likewise. **Fill a sample**: it is
/// what makes the difference between covering a record and covering the shape of one.
///
/// # Panics
///
/// Listing **every** path whose absence is fatal, not the first — a record is usually
/// wrong in one way throughout, and one round of the test should say so. Also when
/// `T` is not stored as a JSON object, when an `exempt` path is not in `sample` (so an
/// exception cannot outlive a rename), and when an `exempt` path turns out to be
/// readable after all — an exemption that has stopped being necessary is coverage the
/// record silently lost.
///
/// # Why it is compiled rather than `#[cfg(test)]`
///
/// Three of the records live in a frontend (`tutor::Row`, `gradients::GradientEntry`,
/// `window::Placement`), and a `#[cfg(test)]` item is not visible across a crate
/// boundary — a test-only spelling here would leave exactly the records this crate
/// cannot see unchecked. It is generic, so a build that never calls it never
/// instantiates it, and the wasm bundle carries nothing.
#[track_caller]
pub fn every_field_may_be_absent<T: Serialize + DeserializeOwned>(sample: &T, exempt: &[&str]) {
    let of = std::any::type_name::<T>();
    let value =
        serde_json::to_value(sample).unwrap_or_else(|e| panic!("{of} does not encode: {e}"));
    assert!(
        value.is_object(),
        "{of} is not stored as a JSON object, so there is no field here to drop",
    );
    let mut paths = Vec::new();
    walk(&value, &mut Vec::new(), &mut paths);

    for name in exempt {
        let steps = paths
            .iter()
            .find(|(path, _)| path == name)
            .map(|(_, steps)| steps.clone())
            .unwrap_or_else(|| {
                panic!(
                    "{of} has no field `{name}` for its exemption to be about — a renamed \
                     field leaves its exemption behind, still passing"
                )
            });
        assert!(
            serde_json::from_value::<T>(dropping(&value, &steps)).is_err(),
            "{of} reads without `{name}`, so exempting it costs the coverage and buys \
             nothing. Drop the exemption.",
        );
    }

    let fatal: Vec<String> = paths
        .iter()
        .filter(|(path, _)| !exempt.iter().any(|e| covers(e, path)))
        .filter_map(|(path, steps)| {
            let e = serde_json::from_value::<T>(dropping(&value, steps)).err()?;
            Some(format!("  {path} ({e})"))
        })
        .collect();
    assert!(
        fatal.is_empty(),
        "{of} is unreadable without these, so every row a build that predates one of \
         them wrote is dropped whole:\n{}\nGive each #[serde(default)] — or exempt it \
         at this call, if its absence is meant to be fatal.",
        fatal.join("\n"),
    );
}

/// One hop down a JSON value: an object's key, or an array's index.
#[derive(Clone, Debug)]
enum Step {
    Key(String),
    Index(usize),
}

/// Every removable field under `value`, deepest last, as a path and the hops to it.
///
/// Only object keys are removable — an array's *length* is not a compatibility
/// question, and a missing element is a shorter list rather than an older one.
fn walk(value: &serde_json::Value, at: &mut Vec<Step>, out: &mut Vec<(String, Vec<Step>)>) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, child) in map {
                at.push(Step::Key(key.clone()));
                out.push((spell(at), at.clone()));
                walk(child, at, out);
                at.pop();
            }
        }
        serde_json::Value::Array(items) => {
            for (i, child) in items.iter().enumerate() {
                at.push(Step::Index(i));
                walk(child, at, out);
                at.pop();
            }
        }
        _ => {}
    }
}

/// How a path reads in a failure and in an exemption: `modulation.size.floor`,
/// `stops[0].t`.
fn spell(steps: &[Step]) -> String {
    let mut out = String::new();
    for step in steps {
        match step {
            Step::Key(k) if out.is_empty() => out.push_str(k),
            Step::Key(k) => {
                out.push('.');
                out.push_str(k);
            }
            Step::Index(i) => out.push_str(&format!("[{i}]")),
        }
    }
    out
}

/// Whether exempting `exempt` covers `path` — the path itself, and everything under it.
fn covers(exempt: &str, path: &str) -> bool {
    path == exempt
        || path
            .strip_prefix(exempt)
            .is_some_and(|rest| rest.starts_with('.') || rest.starts_with('['))
}

/// `value` with the field at `steps` taken out. The last hop is always an object key
/// ([`walk`] pushes no other), so there is always a field to remove.
fn dropping(value: &serde_json::Value, steps: &[Step]) -> serde_json::Value {
    let mut out = value.clone();
    let mut at = &mut out;
    let (last, above) = steps.split_last().expect("a path has at least one hop");
    for step in above {
        at = match step {
            Step::Key(k) => at.get_mut(k),
            Step::Index(i) => at.get_mut(i),
        }
        .expect("the path was walked out of this very value");
    }
    let Step::Key(key) = last else {
        unreachable!("only an object key is removable")
    };
    at.as_object_mut()
        .expect("a key's parent is an object")
        .remove(key);
    out
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
/// nothing for one that costs the session. It is still a bug, so it is said here —
/// the `bool` this used to answer was one no `main` could act on, and both discarded
/// it.
pub fn install(backend: impl Backend) {
    if BACKEND.set(Box::new(backend)).is_err() {
        tracing::warn!(
            "a second store was installed and ignored; every record this process keeps \
             goes to the first one"
        );
    }
}

fn backend() -> Option<&'static dyn Backend> {
    BACKEND.get().map(AsRef::as_ref)
}

/// The untyped half, private so [`save`]/[`load`] are the only way in or out — which
/// is what makes "one format" a property of the module rather than a habit.
fn get(store: Store) -> Option<String> {
    let Some(backend) = backend() else {
        warn_no_store();
        return None;
    };
    backend.get(store.named().0)
}

/// Say — once — that a record was read with no store installed.
///
/// The caller cannot tell that from "this client has stored nothing": both are `None`
/// and both take the default, which is right for the second and a silently lost
/// setting for the first. The web frontend read four records that way for a release,
/// `install` having sat below `AppState::new` in the root's body.
///
/// Once, because the answer cannot change: a store is installed before the first read
/// or the process runs without one.
fn warn_no_store() {
    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
        tracing::warn!(
            "a record was read before a store was installed; every read here answers \"nothing stored\""
        );
    }
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

/// An `f32` that survives a value JSON cannot spell, for
/// `#[serde(with = "crate::storage::finite")]`.
///
/// JSON has no NaN and no infinity: `serde_json` writes one as `null`, and `null`
/// will not read back into an `f32`. In a list that costs the entry, which is the
/// bargain the format already makes — but a whole record ([`load`]) is
/// all-or-nothing, so one float that ever went non-finite makes **every field beside
/// it** unreadable on the next launch. `#[serde(default)]` does not reach it: that is
/// about a field that is *missing*, and this one is present and `null`.
///
/// The read is made total rather than the write made lossy: `null` arrives as the NaN
/// it was written from, so the stored bytes are what they always were and the record
/// is readable whatever is in it. **What a non-finite value means is then the
/// record's own to say**, at the one place that knows what the number is for —
/// `Hdr::clamped_headroom` for the settings, `Placement::usable` for the window,
/// which answer differently. A stand-in invented here would be a second opinion about
/// both.
///
/// The cost is that a loaded record may hold a NaN until something funnels it, which
/// makes a derived `PartialEq` non-reflexive. Both records that use this are read and
/// compared only through those funnels; a third should check.
pub mod finite {
    use serde::{Deserialize, Deserializer, Serializer};

    /// Unchanged from the derive — `serde_json` writes a non-finite float as `null`,
    /// and it is the *read* that has to cope. Here so the field can say `with`.
    pub fn serialize<S: Serializer>(value: &f32, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_f32(*value)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<f32, D::Error> {
        Ok(Option::<f32>::deserialize(d)?.unwrap_or(f32::NAN))
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
    use strum::VariantArray;

    const ALL: &[Store] = Store::VARIANTS;

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

    /// The rule the list format exists for: an entry nobody can read costs that entry
    /// — **and is counted**, since the drop is the symptom every stored-record mistake
    /// shows up as and a silent one attributes to nothing.
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
        let (kept, dropped) = entries::<Item>(json).expect("the list reads");
        assert_eq!(
            kept,
            vec![
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
            ],
            "the three readable entries survive the three that are not"
        );
        assert_eq!(dropped.count, 3, "and the three that are not are counted");
        assert!(
            dropped.first.is_some_and(|e| !e.is_empty()),
            "with a reason for the first, which is what a warning has to print"
        );
    }

    /// A list nothing is wrong with reports nothing — so the warning above means
    /// something when it appears.
    #[test]
    fn a_whole_list_drops_nothing() {
        let (kept, dropped) = entries::<Item>(r#"[{"name":"a","n":1}]"#).expect("the list reads");
        assert_eq!(kept.len(), 1);
        assert_eq!(dropped, Dropped::default());
    }

    /// A record that is not a list at all is nothing stored, not an empty list — the
    /// distinction the quick-brush rack seeds itself on.
    #[test]
    fn damage_is_not_an_empty_list() {
        assert!(entries::<Item>("{}").is_none());
        assert!(entries::<Item>("garbage").is_none());
        assert_eq!(entries::<Item>("[]"), Some((vec![], Dropped::default())));
    }

    /// What [`every_field_may_be_absent`] said when it refused `f` — and nothing on
    /// stderr, since a panic this test *wants* would otherwise read as a failure in
    /// the log beside the ones that are.
    fn refused(f: impl FnOnce() + std::panic::UnwindSafe) -> String {
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let out = std::panic::catch_unwind(f);
        std::panic::set_hook(hook);
        let e = out.expect_err("the gate let this through");
        e.downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string()))
            .expect("a panic message")
    }

    /// A tune, standing in for the half of a brush a preset stores: one field that
    /// falls back and one that does not.
    #[derive(Serialize, Deserialize)]
    struct Tune {
        #[serde(default)]
        flow: f32,
        size: f32,
    }

    /// A record holding one — and holding it by a field that *does* default, so the
    /// only way to reach `size` is to walk into it.
    #[derive(Serialize, Deserialize)]
    struct Held {
        name: String,
        #[serde(default)]
        tune: Tune,
    }

    impl Default for Tune {
        fn default() -> Self {
            Self {
                flow: 1.0,
                size: 8.0,
            }
        }
    }

    fn held() -> Held {
        Held {
            name: "Wet Oil".into(),
            tune: Tune::default(),
        }
    }

    /// The gate's own claim: a field that cannot be absent is named, **at whatever
    /// depth it sits**. `tune.size` is the case a top-level check passes green on,
    /// and it is the shape a stored preset really has — a `stark_model` type three
    /// levels below the record (§25.6).
    #[test]
    fn the_gate_names_a_field_that_cannot_be_absent_at_any_depth() {
        let said = refused(|| every_field_may_be_absent(&held(), &["name"]));
        assert!(said.contains("tune.size"), "{said}");
        assert!(
            !said.contains("tune.flow"),
            "a field that does fall back is not a finding: {said}",
        );
        // And with it exempted, the record passes: `name` and `tune.size` are the
        // whole of what its absence is fatal about.
        every_field_may_be_absent(&held(), &["name", "tune.size"]);
    }

    /// An exemption is a claim, and a claim is checked: one that has stopped being
    /// necessary is coverage the record lost without saying so.
    #[test]
    fn an_exemption_that_buys_nothing_is_refused() {
        let said = refused(|| every_field_may_be_absent(&held(), &["name", "tune.size", "tune"]));
        assert!(said.contains("`tune`"), "{said}");
    }

    /// And one that names no field at all — what a rename leaves behind, still
    /// passing.
    #[test]
    fn an_exemption_that_names_nothing_is_refused() {
        let said = refused(|| every_field_may_be_absent(&held(), &["title", "tune.size"]));
        assert!(said.contains("`title`"), "{said}");
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
