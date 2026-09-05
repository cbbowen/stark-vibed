//! **Files past builds wrote still open** (§8).
//!
//! The format reconciles by name against a schema the file carries, so a field
//! wants `#[serde(default)]`, a rename wants `#[serde(alias)]`, and a retired
//! action keeps its variant. Those rules were enforced by nothing: `io.rs`'s two
//! compat tests hand-transcribe two *past shapes* and so pin history, saying
//! nothing about a change made today. This says it — against bytes rather than
//! against a transcription.
//!
//! # Never regenerate a fixture — add a new one
//!
//! A fixture is a file a past build wrote, and that is the whole of what it is worth.
//! Rewriting one with today's encoder destroys exactly the evidence it exists to
//! carry, and leaves a test that passes because both halves moved together. The
//! writer is `action_kinds.rs::writes_a_fresh_fixture`, `#[ignore]`d and refusing to
//! overwrite; adding a fixture is changing its `FIXTURE` to today's date and running
//! it.
//!
//! # What the manifest holds, and what it does not
//!
//! Each action's [`ActionKind::label`] and lamport, the canvas, the bundle's needs
//! and one `LayerId` — identity, not contents. **Not a digest of the document**: a
//! field gaining a default is a legitimate change the format is built to absorb, and
//! a fixture that failed on it would be a tripwire for the opposite of what it
//! guards. What it fails on is an action that stopped loading or came back as
//! something else.

use std::path::{Path, PathBuf};

use stark_model::DocumentFile;
use stark_model::document::ActionKind;

/// Every `.stark` under `tests/fixtures/`, with the `.txt` beside it.
fn fixtures() -> Vec<(PathBuf, PathBuf)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let entries =
        std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("no fixtures at {}: {e}", dir.display()));
    let mut found: Vec<(PathBuf, PathBuf)> = entries
        .map(|e| e.expect("a directory entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "stark"))
        .map(|p| {
            let txt = p.with_extension("txt");
            (p, txt)
        })
        .collect();
    found.sort();
    // A gate that checked nothing would still report `ok` (CLAUDE.md).
    assert!(
        !found.is_empty(),
        "{} holds no .stark fixture, so this gate checked nothing",
        dir.display(),
    );
    found
}

/// The manifest as `(key, value)` pairs, comments and blanks dropped.
fn manifest(path: &Path) -> Vec<(String, String)> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{} has no manifest beside it: {e}", path.display()));
    text.lines()
        .map(str::trim_end)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            let (key, rest) = l.split_once(' ').unwrap_or((l, ""));
            (key.to_string(), rest.to_string())
        })
        .collect()
}

/// Every stored fixture loads, and every action in it is still the action it was.
#[test]
fn a_stored_document_still_opens_as_what_it_was() {
    for (stark, txt) in fixtures() {
        let name = stark.file_name().expect("a file name").to_string_lossy();
        let bytes = std::fs::read(&stark).expect("read the fixture");
        // `{e:?}` rather than `{e}`: the reconciliation failure — which name this
        // build no longer answers to — is in the `carbonite::Error` behind `source`.
        let doc = DocumentFile::from_bytes(&bytes)
            .unwrap_or_else(|e| panic!("{name} no longer loads: {e:?}"));
        // The stranger's door too: a fixture is nowhere near the cap, so the two
        // must agree, and this is where a bound that crept onto both would show.
        assert!(
            DocumentFile::from_untrusted_bytes(&bytes).is_ok(),
            "{name} loads for its owner but not from a peer",
        );

        let lines = manifest(&txt);
        let want = |key: &str| -> Vec<&str> {
            lines
                .iter()
                .filter(|(k, _)| k == key)
                .map(|(_, v)| v.as_str())
                .collect()
        };

        assert_eq!(
            want("canvas.color_space"),
            vec![format!("{:?}", doc.canvas.color_space)],
            "{name}: the canvas came back in a different color space",
        );
        assert_eq!(
            want("canvas.substrate"),
            vec![format!("{:?}", doc.canvas.substrate)],
            "{name}: the substrate the log starts from moved",
        );

        let mut needs: Vec<String> = doc
            .content
            .iter()
            .map(|(need, _)| format!("{need:?}"))
            .collect();
        needs.sort();
        assert_eq!(want("content"), needs, "{name}: the bundle changed shape");
        assert!(
            doc.unbundled_content().is_empty(),
            "{name}: the log now names content the file was written carrying",
        );

        let minted = doc.actions.iter().find_map(|a| match a.kind {
            ActionKind::AddLayer { id, .. } => Some(id),
            _ => None,
        });
        assert_eq!(
            want("add_layer"),
            vec![format!("{:?}", minted.expect("an AddLayer"))],
            "{name}: a layer id came back as a different layer (§17.9)",
        );

        let actions: Vec<String> = doc
            .actions
            .iter()
            .map(|a| format!("{} {}", a.id.lamport, a.kind.label()))
            .collect();
        assert_eq!(
            want("action"),
            actions,
            "{name}: the log no longer reads as the actions it was written from",
        );
    }
}
