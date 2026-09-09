//! What this crate claims about its own source, checked by reading it.
//!
//! `no_toolkit_types` is here rather than in `src` because it is a list of strings
//! that must not appear in the tree, and a check that lived in the tree would have to
//! exempt itself — which is exactly what went wrong: `lib.rs` was skipped wholesale
//! for naming the banned prefixes in its prose, so every rule ever added to that test
//! was unenforced in the crate's largest and most-edited file.
//!
//! `every_module_is_in_the_index` has no such problem and is here for company: both
//! read this crate's source rather than run any of it.

use std::path::{Path, PathBuf};

/// Toolkit types, path-shaped so that *naming* a toolkit in prose stays legal while
/// naming one of its types does not: `stark-dioxus-frontend` is a neighbour worth
/// pointing at, `dioxus::Event` is the thing this crate must not know.
///
/// `gpui::` covers `wgpui::` too, and `gpui_` covers the widget layers either spelling
/// grows — `wgpui_component`, `wgpui_base` (§11.1). Two patterns for the family rather
/// than a row per crate, since the next one is not on any list.
const TOOLKIT: &[&str] = &["dioxus::", "gpui::", "gpui_", "web_sys::", "winit::"];

/// CSS declarations — [`TOOLKIT`]'s blind spot, and a real one: two functions here
/// emitted stylesheet text and passed every round, because CSS is not a Rust path. It
/// is a toolkit dialect all the same. A page is one frontend's idea of a surface, and
/// the native one draws the same motion by moving a quad.
///
/// Each pattern carries the punctuation a *declaration* has, so prose cannot trip it:
/// a doc comment says "a `background-image`", never "background-image:".
const CSS: &[&str] = &[
    "background-image:",
    "transform: translate",
    "transition:",
    "z-index:",
    "!important",
];

/// **No toolkit type, and no stylesheet, reaches this crate.**
///
/// Read off the source rather than the manifest, because the manifest is the weaker
/// claim: a `dioxus::` path can arrive through a dependency's re-export without ever
/// appearing in `[dependencies]`, and what would go wrong is not a build failure but a
/// module that quietly stops being movable.
///
/// The check a reviewer would do, made a thing that runs.
#[test]
fn no_toolkit_types() {
    let mut found = Vec::new();
    for path in sources() {
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{} does not read: {e}", path.display()));
        for (n, line) in text.lines().enumerate() {
            if let Some(name) = TOOLKIT.iter().chain(CSS).find(|b| line.contains(**b)) {
                found.push(format!("{}:{}: {name}", path.display(), n + 1));
            }
        }
    }
    assert!(
        found.is_empty(),
        "a toolkit type or a stylesheet reached stark-ui, which is what it exists not \
         to name (§11.2):\n{}",
        found.join("\n")
    );
}

/// **Every module is a row of the crate's index**, so the list at the top of `lib.rs`
/// is what is actually here rather than what was here when somebody last wrote it down.
///
/// It had drifted to twelve rows for twenty-six modules — everything stages N3, N7 and
/// N8 brought down was missing, the two binding registries included, which are the
/// crate's largest single subject (§25). Reviewers noticed it more than once; a
/// reviewer is not a gate, and the next omission would have been as quiet as those
/// fourteen.
///
/// One direction only. The other — a row for a module that has since gone — is already
/// a broken intra-doc link, and the workspace doc gate runs with
/// `-D rustdoc::broken_intra_doc_links`.
#[test]
fn every_module_is_in_the_index() {
    let lib = std::fs::read_to_string(manifest_dir().join("src/lib.rs")).expect("lib.rs reads");
    let mut missing = Vec::new();
    let mut modules = 0;
    for line in lib.lines() {
        let Some(name) = line
            .strip_prefix("pub mod ")
            .and_then(|rest| rest.strip_suffix(';'))
        else {
            continue;
        };
        modules += 1;
        if !lib.contains(&format!("//! - [`{name}`]")) {
            missing.push(name.to_string());
        }
    }
    assert!(modules > 0, "no `pub mod` line was found in lib.rs at all");
    assert!(
        missing.is_empty(),
        "declared but not in the crate's index — a module nobody can find from the top \
         of the crate (§11.2): {}",
        missing.join(", ")
    );
}

/// Every Rust file this crate is built from: `src`, **recursively**, plus `build.rs`.
///
/// Recursion is the whole point. The one-level `read_dir` this replaces filtered on the
/// extension, and a directory has no extension — so the day anything here became
/// `commands/defaults.rs` the check would have gone on reporting `ok` while reading
/// less of the crate every year. The count is asserted for the same reason: a walk that
/// finds nothing is a green test that read nothing.
fn sources() -> Vec<PathBuf> {
    let root = manifest_dir();
    let mut found = vec![root.join("build.rs")];
    let mut stack = vec![root.join("src")];
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
    assert!(
        found.len() > 1,
        "the walk found no source under `src`, which is a green test that read nothing"
    );
    found.sort();
    found
}

fn manifest_dir() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}
