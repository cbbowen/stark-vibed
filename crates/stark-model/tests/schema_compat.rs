//! **Today's types still read every schema this crate shipped** (§8).
//!
//! What a stored `.stark` proves, `format_stability.rs` proves: those *bytes* still
//! open. What it cannot prove is anything about a shape it does not contain — a
//! variant no fixture ever wrote, a field on a type no fixture exercises. This walks
//! the released **schema** instead, so every variant of every enum under it is
//! visited, and asks whether today's reader takes each one.
//!
//! The answer lives in the reader's attributes rather than in either schema: whether
//! an added field has a `#[serde(default)]`, whether a renamed one has a
//! `#[serde(alias)]`. `carbonite::compat` settles it by *running the reader* over
//! synthetic values conforming to the stored schema. `check_static` rather than
//! `check`, because tracing cannot see through an alias: `CanvasMeta`'s
//! `alias = "surface"` makes it and `DocumentFile` untraceable, so `check` would
//! answer `Inconclusive` on a break in the two entries that matter most.
//!
//! `Inconclusive` is a failure here, not a pass. It means the probe could not answer,
//! and a gate that reads "could not answer" as "fine" reports `ok` having checked
//! nothing.
//!
//! # Never regenerate a snapshot — add a new one
//!
//! A snapshot is a schema a past build shipped. Rewriting one with today's derive
//! leaves a test that passes because both halves moved together. The writer is
//! [`writes_fresh_snapshots`], `#[ignore]`d and refusing to overwrite.
//!
//! One file per type per date (`<date>-<Type>.bin`), rather than one dated container:
//! adding a type to [`roster`] then adds a file at today's date instead of rewriting
//! a stored one.

use std::any::type_name;
use std::error::Error;
use std::path::{Path, PathBuf};

use carbonite::{Schema, StaticSchema, compat};
use serde::de::DeserializeOwned;
use stark_model::document::{
    BrushParams, Filter, PerspectiveGuide, PerspectiveMap, StrokeRecord, WarpMap,
};
use stark_model::{BuildId, CanvasMeta, DocumentFile, StrokeHead};

/// Reads a stored schema and probes today's type against it.
type Check = fn(&[u8]) -> Result<(), Box<dyn Error>>;

/// One type's snapshot, monomorphized: what to store, and what to ask of a stored one.
struct Snapshot {
    /// The type's bare name, which is also the tail of its snapshot's filename.
    name: &'static str,
    today: fn() -> Vec<u8>,
    check: Check,
}

fn snapshot<T: DeserializeOwned + StaticSchema>() -> Snapshot {
    Snapshot {
        name: type_name::<T>().rsplit("::").next().expect("a type name"),
        today: || T::schema().to_bytes(),
        check: |bytes| {
            let released = Schema::<T>::from_bytes(bytes)?;
            compat::check_static::<T>(&released)?;
            Ok(())
        },
    }
}

/// The document, and the types a break would most likely be *in*.
///
/// The root alone would fail without naming the type that caused it, and `BrushParams`
/// — 42 `#[serde(default)]`s — is the one most likely to gain a field. The rest are the
/// payloads a fixture reaches least deeply. [`StrokeHead`] is the presence wire rather
/// than the file (§8), where the ALPN would refuse a disagreement outright; it is here
/// because it declares an older sender's `translation` defaulted, and nothing else
/// holds it to that.
fn roster() -> Vec<Snapshot> {
    vec![
        snapshot::<DocumentFile>(),
        snapshot::<BrushParams>(),
        snapshot::<StrokeRecord>(),
        snapshot::<CanvasMeta>(),
        snapshot::<BuildId>(),
        snapshot::<StrokeHead>(),
        snapshot::<PerspectiveGuide>(),
        snapshot::<Filter>(),
        snapshot::<WarpMap>(),
        snapshot::<PerspectiveMap>(),
    ]
}

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/schemas")
}

/// Every `<date>-<Type>.bin` under `tests/schemas/`, as `(path, type name)`.
fn stored() -> Vec<(PathBuf, String)> {
    let dir = dir();
    let entries =
        std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("no schemas at {}: {e}", dir.display()));
    let mut found: Vec<(PathBuf, String)> = entries
        .map(|e| e.expect("a directory entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "bin"))
        .map(|p| {
            let stem = p.file_stem().expect("a file stem").to_string_lossy();
            let (_, name) = stem
                .rsplit_once('-')
                .unwrap_or_else(|| panic!("{} is not <date>-<Type>.bin", p.display()));
            let name = name.to_string();
            (p, name)
        })
        .collect();
    found.sort();
    found
}

/// Today's types read every schema stored beside this file.
#[test]
fn a_released_schema_is_still_readable() {
    let roster = roster();
    let stored = stored();

    for (path, name) in &stored {
        let probe = roster
            .iter()
            .find(|s| s.name == name)
            .unwrap_or_else(|| panic!("{} names no type in the roster", path.display()));
        let bytes = std::fs::read(path).expect("read the snapshot");
        // `{e}` says which of the three verdicts it is and which leaf it is about;
        // `Inconclusive` reaches here too, and is a failure by the header's rule.
        (probe.check)(&bytes)
            .unwrap_or_else(|e| panic!("{}: {name} no longer reads it: {e}", path.display()));
    }

    // A gate that checked nothing would still report `ok` (CLAUDE.md).
    for s in &roster {
        assert!(
            stored.iter().any(|(_, name)| name == s.name),
            "{} holds no snapshot for {}, so nothing was checked for it",
            dir().display(),
            s.name,
        );
    }
}

// ---------------------------------------------------------------------------
// The stored snapshots

/// The snapshot round this build would write, if asked. Bumped — never edited in
/// place — when a new one is added.
const SNAPSHOT: &str = "2026-09";

/// Writes `tests/schemas/<SNAPSHOT>-<Type>.bin` for every type in [`roster`], for
/// [`a_released_schema_is_still_readable`] to open on every later build (§8).
///
/// `#[ignore]`d because it is a **command**, not a test: run it deliberately, having
/// first changed [`SNAPSHOT`] to today's date, and commit what it wrote.
///
/// ```sh
/// cargo nextest run -p stark-model -E 'test(writes_fresh_snapshots)' --run-ignored all
/// ```
///
/// It refuses to overwrite, for the reason the header gives.
#[test]
#[ignore = "writes snapshots; run deliberately, under a new date"]
fn writes_fresh_snapshots() {
    let dir = dir();
    std::fs::create_dir_all(&dir).expect("the schemas directory");
    for s in roster() {
        let path = dir.join(format!("{SNAPSHOT}-{}.bin", s.name));
        assert!(
            !path.exists(),
            "{} exists: a snapshot is a schema a past build shipped, so add a dated \
             round rather than regenerating this",
            path.display(),
        );
        std::fs::write(&path, (s.today)()).expect("write the snapshot");
    }
}
