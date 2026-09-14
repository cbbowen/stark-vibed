//! What this crate's consumers claim about their source, checked by reading it.

use std::path::{Path, PathBuf};

/// The engine methods that replace the document. `Desk::replace` is the one caller: a
/// frontend reaching one directly installs nothing first and leaves the view where the
/// last document had it (§18.1.2).
///
/// The engine keeps them `pub` — its own suite replays files through them — so what
/// holds a consumer to the door is its `clippy.toml`, which refuses each by path. CI
/// does not lint the native frontend, so the source is read here too.
const REPLACES: &[&str] = &[
    "load_document",
    "load_bytes",
    "join_collaboration",
    "new_document",
    "replay_timelapse",
];

/// The door itself, relative to this crate's manifest.
const DOOR: &str = "src/desk.rs";

/// **No crate built on `stark-ui` replaces a document except through
/// `Desk::replace`**, and nothing in this crate does but that method.
#[test]
fn only_the_desk_replaces_a_document() {
    let mut found = Vec::new();
    for dir in consumers() {
        for path in sources(&dir) {
            if dir == manifest_dir() && path.strip_prefix(&dir).is_ok_and(|p| p == Path::new(DOOR))
            {
                continue;
            }
            for (n, line) in read(&path).lines().enumerate() {
                if let Some(name) = replacement_call(line) {
                    found.push(format!("{}:{}: {name}", path.display(), n + 1));
                }
            }
        }
    }
    assert!(
        found.is_empty(),
        "the document is replaced around `stark_ui::desk::Desk::replace`, which installs \
         what it owes and frames it:\n{}",
        found.join("\n")
    );
}

/// **Each consumer's `clippy.toml` holds the root file's rows and refuses every
/// replacement method.** Clippy reads the nearest file alone, so a consumer's replaces
/// the root one: this is what keeps a row added there applying here, and the lint and
/// [`REPLACES`] one list.
#[test]
fn every_consumer_lints_the_root_rows_and_the_replacements() {
    let root = read(&manifest_dir().join("../../clippy.toml"));
    let rows: Vec<&str> = root
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("{ path ="))
        .collect();
    assert!(!rows.is_empty(), "the root clippy.toml declares no rows");
    for dir in consumers() {
        let path = dir.join("clippy.toml");
        let own = read(&path);
        let own: Vec<&str> = own.lines().map(str::trim).collect();
        for row in &rows {
            assert!(
                own.contains(row),
                "{} lacks the root row {row}",
                path.display()
            );
        }
        for name in REPLACES {
            let refused = format!("{{ path = \"stark_engine::Engine::{name}\"");
            assert!(
                own.iter().any(|l| l.starts_with(&refused)),
                "{} does not refuse `{name}`",
                path.display()
            );
        }
    }
}

/// The gate's reading recognizes a call by receiver and by path, and not a comment, a
/// free function of the same name or a longer name.
#[test]
fn a_replacement_is_recognized_however_it_is_spelled() {
    for &name in REPLACES {
        for line in [
            format!("    engine.{name}(&file)?;"),
            format!("        .{name}(&file)"),
            format!("    Engine::{name}(&mut engine, &file)?;"),
            format!("    stark_engine::Engine::{name}(&mut engine, space, substrate)"),
            format!("    files.iter().map(Engine::{name})"),
        ] {
            assert_eq!(replacement_call(&line), Some(name), "{line:?} replaces");
        }
        for line in [
            format!("    // engine.{name}(&file)"),
            format!("    /// [`Engine::{name}`]"),
            format!("    crate::substrates::{name}(state, color, pick, pending);"),
            format!("pub fn {name}(state: AppState) {{"),
            format!("    Engine::{name}_later(&mut engine)"),
            format!("    PreviewEngine::{name}(&mut engine)"),
        ] {
            assert_eq!(replacement_call(&line), None, "{line:?} does not replace");
        }
    }
}

/// Which replacement method `line` calls or names, if any: `.name(` on a receiver, or
/// the path `Engine::name`, which a call through the type and a method passed as a
/// function both spell. A comment line names nothing.
fn replacement_call(line: &str) -> Option<&'static str> {
    if line.trim_start().starts_with("//") {
        return None;
    }
    REPLACES.iter().copied().find(|name| {
        line.contains(&format!(".{name}(")) || names_path(line, &format!("Engine::{name}"))
    })
}

/// Whether `path` stands in `line` as a whole path, not inside a longer identifier at
/// either end.
fn names_path(line: &str, path: &str) -> bool {
    let ident = |c: char| c == '_' || c.is_alphanumeric();
    line.match_indices(path)
        .any(|(at, _)| !line[..at].ends_with(ident) && !line[at + path.len()..].starts_with(ident))
}

/// This crate and every crate whose manifest depends on it. Found rather than listed,
/// so a third frontend is held to these the day it depends on this crate; the native
/// one is excluded from CI's build, and reading its source needs none.
fn consumers() -> Vec<PathBuf> {
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
    consumers
}

/// Every Rust file a crate compiles: `src/`, `tests/`, `examples/` and its build
/// script. Asserted non-empty, since a walk that finds nothing is a green test that
/// read nothing.
fn sources(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack: Vec<PathBuf> = ["src", "tests", "examples"]
        .iter()
        .map(|sub| dir.join(sub))
        .filter(|sub| sub.is_dir())
        .collect();
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
    let build = dir.join("build.rs");
    if build.is_file() {
        found.push(build);
    }
    assert!(!found.is_empty(), "no source under {}", dir.display());
    found
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{} does not read: {e}", path.display()))
}

fn manifest_dir() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}
