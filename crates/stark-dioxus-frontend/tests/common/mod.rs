//! What the source-scanning tests share: reading a file, and walking `src` for every
//! `.rs` file in it. One copy, so the guard against a walk that found nothing — a green
//! test that read nothing — cannot be dropped from one of them.

use std::path::{Path, PathBuf};

pub fn read(path: &Path) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{} does not read: {e}", path.display()))
}

/// Every `.rs` file under `root`, recursively.
pub fn rust_files(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
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
