//! What this crate's consumers claim about their source, checked by reading it.

use std::path::{Path, PathBuf};

/// The engine calls that replace the document. `Session::replace` is the one caller:
/// a frontend reaching one directly installs nothing first and leaves the view where
/// the last document had it (§18.1.2).
///
/// The engine keeps them `pub` — its own suite replays files through them — so what
/// holds a frontend to the door is this, not visibility.
const REPLACES: &[&str] = &[
    "load_document(",
    "load_bytes(",
    "join_collaboration(",
    "new_document(",
    "replay_timelapse(",
];

/// The file that is the door.
const DOOR: &str = "session.rs";

/// **No crate built on `stark-ui` replaces a document except through
/// `Session::replace`**, and nothing in this crate does but that method.
///
/// Consumers are found by their manifests rather than listed, so a third frontend is
/// held to it the day it depends on this crate. The native frontend is excluded from
/// CI's build; reading its source needs no build, so this covers it too.
#[test]
fn only_the_session_replaces_a_document() {
    let crates = manifest_dir().parent().expect("crates/ holds this crate");
    let mut consumers = Vec::new();
    for entry in std::fs::read_dir(crates).expect("crates/ reads") {
        let dir = entry.expect("a readable directory entry").path();
        let Ok(manifest) = std::fs::read_to_string(dir.join("Cargo.toml")) else {
            continue;
        };
        // `stark-ui = …` or `stark-ui.workspace = …`, and not `stark-uix`.
        let depends = manifest.lines().any(|line| {
            line.trim_start()
                .strip_prefix("stark-ui")
                .is_some_and(|rest| rest.starts_with([' ', '=', '.']))
        });
        if depends || dir == manifest_dir() {
            consumers.push(dir);
        }
    }
    assert!(
        consumers.len() >= 3,
        "expected this crate and both frontends, found {consumers:?}"
    );

    let mut found = Vec::new();
    for dir in &consumers {
        for path in rust_files(&dir.join("src")) {
            if dir == manifest_dir() && path.file_name().is_some_and(|n| n == DOOR) {
                continue;
            }
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("{} does not read: {e}", path.display()));
            for (n, line) in text.lines().enumerate() {
                if line.trim_start().starts_with("//") {
                    continue;
                }
                if let Some(call) = REPLACES
                    .iter()
                    .find(|call| line.contains(&format!(".{call}")))
                {
                    found.push(format!("{}:{}: {call}", path.display(), n + 1));
                }
            }
        }
    }
    assert!(
        found.is_empty(),
        "the document is replaced around `stark_ui::session::Session::replace`, which \
         installs what it owes and frames it:\n{}",
        found.join("\n")
    );
}

/// Every `.rs` file under `dir`, recursively. Asserted non-empty: a walk that finds
/// nothing is a green test that read nothing.
fn rust_files(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("a readable directory") {
            let path = entry.expect("a readable directory entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                found.push(path);
            }
        }
    }
    assert!(!found.is_empty(), "no source under {}", dir.display());
    found
}

fn manifest_dir() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}
