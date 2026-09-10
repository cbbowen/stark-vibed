//! Only `platform/web.rs` names the browser (§11).
//!
//! `web-sys`, `js-sys` and `wasm-bindgen` are wasm32-only dependencies, so the host build
//! refuses a path into them anywhere it compiles. It does not compile what sits under
//! `#[cfg(target_arch = "wasm32")]`, and that is where a browser call outside the platform
//! module would hide: checked only by the wasm build, which runs no tests. `dioxus::web`
//! is the other door to the same types — its `WebEventExt` hands back a `web_sys` event —
//! and it resolves on the host too.

use std::path::{Path, PathBuf};

const BROWSER: &[&str] = &["web_sys", "js_sys", "wasm_bindgen", "dioxus::web"];

#[test]
fn only_the_web_half_of_platform_names_the_browser() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let web = Path::new("platform").join("web.rs");

    // The check matches something, or it is a green test that matched nothing.
    assert!(
        !browser_names(&read(&src.join(&web))).is_empty(),
        "platform/web.rs names none of {BROWSER:?}, so this check cannot see a browser call"
    );

    let mut found = Vec::new();
    for path in rust_files(&src) {
        let relative = path.strip_prefix(&src).expect("the walk stays under src");
        if relative == web || relative.starts_with(Path::new("platform").join("web")) {
            continue;
        }
        for (line, name) in browser_names(&read(&path)) {
            found.push(format!("{}:{line}: {name}", path.display()));
        }
    }
    assert!(
        found.is_empty(),
        "a browser crate is named outside platform/web.rs, where the host build cannot \
         see it if it is cfg-gated (§11):\n{}",
        found.join("\n")
    );
}

/// Each line of `source` that names a browser crate, as `(line number, name)`. Comment
/// lines are prose and may name anything.
fn browser_names(source: &str) -> Vec<(usize, &'static str)> {
    source
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim_start().starts_with("//"))
        .filter_map(|(n, line)| {
            let name = BROWSER.iter().find(|name| line.contains(**name))?;
            Some((n + 1, *name))
        })
        .collect()
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{} does not read: {e}", path.display()))
}

/// Every `.rs` file under `root`, recursively.
fn rust_files(root: &Path) -> Vec<PathBuf> {
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
