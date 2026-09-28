//! The crate's module layering, read off its own source (§2).
//!
//! Sixty thousand lines is enough that a cycle between two top-level modules is
//! invisible: it is two `use` lines in two files, and nothing reports it. This module
//! is the report. [`TIERS`] is the order — lowest first — and the rule is one sentence:
//!
//! > **A module may name only what is in a tier below its own.**
//!
//! Which rules out cycles as a *class* rather than listing the ones we happen to know
//! about: a strict ordering cannot contain one. A new edge that would close a loop
//! fails [`every_edge_goes_down_a_tier`] the round it is written, instead of settling in
//! for a year.
//!
//! [`CROSS_TIER`] is the closed set of edges that do not obey it, each with the reason
//! and what retiring it would cost. It is asserted to be exactly used, so it cannot
//! quietly outlive its reason either.
//!
//! # What it cannot see
//!
//! This reads text, so it is a lower bound on the graph. It does not see:
//!
//! - **a `use` behind any `#[cfg]` but `test`** — a `#[cfg(feature = "…")] use
//!   crate::x` is counted as an ordinary edge, which is right, but a `#[cfg]` that
//!   removes one is not noticed;
//! - **a re-export chain** — `crate::a::B` where `a` re-exports `B` from `c` is an edge
//!   to `a` here, not to `c`. Only the crate root's re-exports are resolved, because
//!   `crate::Engine` would otherwise look like an edge to nothing;
//! - **an alias** — `use crate::gpu as g;` then `g::stroke`, or a trait method reaching
//!   a type never named;
//! - **a relative path** — `super::super::tile` is invisible. There were three such in
//!   the crate when this was written and the moves turned them all absolute; a new one
//!   would slip past. `crate::`-rooted paths are the house style, not a rule this
//!   checks;
//! - **a type parameter's own crate** — `stark_model` and `wgpu` edges are not the
//!   subject.
//!
//! What it deliberately ignores: `#[cfg(test)]` code, items and whole module files
//! alike. A test module legitimately reaches up — it builds a whole engine to exercise
//! one low module — and counting those would make the table a list of exceptions. The
//! stripper fails loudly rather than skipping silently: if it cannot find the closing
//! brace of a `#[cfg(test)]` item it panics naming the file, so a shape it does not
//! understand can never quietly take the rest of a file out of the scan.
//!
//! `lib.rs` is not scanned as a *source*, because a crate root necessarily names every
//! module it declares. It is read for the other purpose: the re-export table.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

/// The two items the crate root holds itself — `unpoisoned` and the timing macro — as a
/// tier. Below everything, because neither names a module.
const ROOT: &str = "(the crate root)";

/// The layering, **lowest tier first**. A module may name only what is in a tier below
/// its own; two modules in the same tier may not name each other.
///
/// The shape of it, from the bottom: the leaves that name nothing, the error vocabulary,
/// the device and what it hands out, the stores, the state a log folds into, the passes,
/// the fold, and then the session tier and the engine that owns all of it.
const TIERS: &[&[&str]] = &[
    // Named by everyone, naming no one.
    &[ROOT],
    // The leaves: no `crate::` path leaves any of these. `filters` and `guides` read the
    // generated shader mirror and are named by a frontend, not from in here (§6.10).
    &[
        "view",
        "colorspace",
        "timing",
        "gesture",
        "noise",
        "projection",
        "filters",
        "guides",
        "path",
        "stark_stroke",
    ],
    // The error vocabulary, `DeviceFailure` included: a device that answers `Result`
    // cannot also own the error it answers with.
    &["error"],
    // The wgpu context and what it allocates. No pass, no pipeline — which is what lets
    // `document` hold a tile handle.
    &["device"],
    &["image", "pictures"],
    // The brush-shape store. Below the passes that read it; see `CROSS_TIER`.
    &["assets"],
    // The state a log folds into. Names `device` and nothing above it.
    &["document"],
    // The passes.
    &["gpu"],
    // The folding: `ApplyCtx` holds every renderer, so it is above all of them.
    &["fold"],
    &["command"],
    &["presence"],
    &["peer"],
    &["session"],
    &["engine"],
];

/// Modules the layering does not rank, with the reason.
///
/// One entry, and it is not a layer at all: `testing` is the suite's and the benchmarks'
/// harness (§9). It reaches down into whatever a test needs to see, and `#[cfg(test)]`
/// code anywhere reaches *it* for a device or a skip. Ranking it would say something
/// false whichever rank it got.
const UNRANKED: &[(&str, &str)] = &[(
    "testing",
    "the suite's own harness: it reaches down to whatever a test must see, and test code \
     at every tier reaches it back",
)];

/// Every edge that does **not** go down a tier, with why it stands and what retiring it
/// would cost. Asserted to be exactly used: an entry whose edge is gone fails the suite
/// as surely as an edge with no entry.
const CROSS_TIER: &[(&str, &str, &str)] = &[(
    "assets",
    "gpu",
    "`assets.rs` bakes a brush shape's prefix volumes through `gpu::stroke::volume`, \
     whose rise measurement is priced against the liquify step budget it sits beside — \
     while `gpu::stroke` resolves a tip *through* the store. The store is below the \
     passes because two of them read it and it reads one module of one of them. \
     Retiring this means `StrokeScene` carrying a resolved tip instead of the store, so \
     that the resolution happens above: a signature change, not a move.",
)];

// ---------------------------------------------------------------------------
// the tests
// ---------------------------------------------------------------------------

#[test]
fn every_edge_goes_down_a_tier() {
    let graph = Graph::read();
    let rank = ranks();
    let unranked: BTreeSet<&str> = UNRANKED.iter().map(|&(m, _)| m).collect();
    let allowed: BTreeSet<(&str, &str)> = CROSS_TIER.iter().map(|&(a, b, _)| (a, b)).collect();

    let mut wrong = Vec::new();
    for edge in &graph.edges {
        let (from, to) = (edge.from.as_str(), edge.to.as_str());
        if from == to || unranked.contains(from) || unranked.contains(to) {
            continue;
        }
        if allowed.contains(&(from, to)) {
            continue;
        }
        match (rank.get(from), rank.get(to)) {
            (Some(a), Some(b)) if a > b => {}
            (Some(_), Some(_)) => wrong.push(format!(
                "{}:{} — {} names {}, which is not in a lower tier ({})",
                edge.file, edge.line, from, to, edge.path
            )),
            _ => wrong.push(format!(
                "{}:{} — {} names {}, and one of the two has no tier ({})",
                edge.file, edge.line, from, to, edge.path
            )),
        }
    }
    assert!(
        wrong.is_empty(),
        "these `crate::` paths climb the layering (§2). Either the edge is the bug, or \
         `TIERS` is wrong and this test is how it gets fixed:\n  {}",
        wrong.join("\n  ")
    );
}

#[test]
fn every_cross_tier_exception_is_still_taken() {
    let graph = Graph::read();
    let seen: BTreeSet<(&str, &str)> = graph
        .edges
        .iter()
        .map(|e| (e.from.as_str(), e.to.as_str()))
        .collect();
    let stale: Vec<&str> = CROSS_TIER
        .iter()
        .filter(|&&(a, b, _)| !seen.contains(&(a, b)))
        .map(|&(a, _, _)| a)
        .collect();
    assert!(
        stale.is_empty(),
        "`CROSS_TIER` still excuses an edge that no longer exists, from: {stale:?}. \
         Delete the entry — an exception that outlives its reason is how the next one \
         gets waved through."
    );
}

#[test]
fn every_top_level_module_has_a_tier() {
    let lib = read_lib();
    let placed: BTreeSet<&str> = TIERS
        .iter()
        .flat_map(|t| t.iter().copied())
        .chain(UNRANKED.iter().map(|&(m, _)| m))
        .collect();
    // A `#[cfg(test)]` module is not in the library, so it is not in the layering — this
    // one among them.
    let test_only = test_only_mod_names(&read(&src_dir().join("lib.rs")));
    let missing: Vec<&String> = lib
        .modules
        .iter()
        .filter(|m| !placed.contains(m.as_str()) && !test_only.contains(m))
        .collect();
    assert!(
        missing.is_empty(),
        "lib.rs declares {missing:?} and `TIERS` does not place them. A new top-level \
         module has to say where it sits before anything may name it."
    );
}

#[test]
fn every_tier_entry_is_a_name_the_crate_root_offers() {
    let lib = read_lib();
    let owners: BTreeSet<&str> = lib.reexports.values().map(String::as_str).collect();
    let mut seen = BTreeSet::new();
    let mut unknown = Vec::new();
    for name in TIERS.iter().flat_map(|t| t.iter().copied()) {
        assert!(seen.insert(name), "`TIERS` names {name} in two tiers");
        if name != ROOT && !lib.modules.contains(name) && !owners.contains(name) {
            unknown.push(name);
        }
    }
    assert!(
        unknown.is_empty(),
        "`TIERS` names {unknown:?}, which lib.rs neither declares as a module nor \
         re-exports from. A stale tier entry ranks nothing."
    );
}

// ---------------------------------------------------------------------------
// reading the source
// ---------------------------------------------------------------------------

/// One `crate::<top>::…` path, as written.
struct Edge {
    from: String,
    to: String,
    file: String,
    line: usize,
    path: String,
}

struct Graph {
    edges: Vec<Edge>,
}

impl Graph {
    fn read() -> Self {
        let lib = read_lib();
        let root = src_dir();
        let files = rust_files(&root);
        let test_only = test_only_module_files(&root, &files);

        let mut edges = Vec::new();
        for file in &files {
            let rel = relative(&root, file);
            // This module is the table; scanning it would find the needles in it. The
            // crate root is not a source, because it declares every module it names.
            if rel == "layering.rs" || rel == "lib.rs" || test_only.contains(file) {
                continue;
            }
            let from = top_module(&rel);
            let text = read(file);
            for (line, code) in without_test_items(&rel, &text) {
                for name in crate_paths(&code) {
                    let to = lib.resolve(&name);
                    edges.push(Edge {
                        from: from.clone(),
                        to,
                        file: rel.clone(),
                        line,
                        path: format!("{}{}", marker(), name),
                    });
                }
            }
        }
        Self { edges }
    }
}

/// Tier per module name.
fn ranks() -> BTreeMap<&'static str, usize> {
    TIERS
        .iter()
        .enumerate()
        .flat_map(|(i, tier)| tier.iter().map(move |&m| (m, i)))
        .collect()
}

/// `crate::`, assembled rather than written, so this file's own table and prose are not
/// needles for the scan that reads it — the crate root's items are matched the same way
/// `stark-model`'s transcendental check assembles its own.
fn marker() -> String {
    format!("{}{}", "crate", "::")
}

fn src_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

fn read(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

fn relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .expect("walked from root")
        .to_string_lossy()
        .replace('\\', "/")
}

fn rust_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(at) = stack.pop() {
        let entries =
            fs::read_dir(&at).unwrap_or_else(|e| panic!("cannot list {}: {e}", at.display()));
        for entry in entries {
            let path = entry.expect("readable entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

/// The top-level module a file belongs to: `gpu/stroke/tips.rs` is `gpu`'s, `error.rs`
/// is its own.
fn top_module(rel: &str) -> String {
    match rel.split_once('/') {
        Some((dir, _)) => dir.to_string(),
        None => rel.trim_end_matches(".rs").to_string(),
    }
}

/// Files whose `mod` declaration carries `#[cfg(test)]`, so nothing in them is compiled
/// into the library.
fn test_only_module_files(root: &Path, files: &[PathBuf]) -> BTreeSet<PathBuf> {
    let mut out = BTreeSet::new();
    for file in files {
        let text = read(file);
        let rel = relative(root, file);
        // Where a submodule of *this* file lives: beside `lib.rs`, or in the directory
        // named after the parent.
        let children = match rel.as_str() {
            "lib.rs" => root.to_path_buf(),
            _ => root.join(rel.trim_end_matches(".rs")),
        };
        for name in test_only_mod_names(&text) {
            out.insert(children.join(format!("{name}.rs")));
        }
    }
    out
}

/// The `mod x;` names a `#[cfg(test)]` immediately precedes.
fn test_only_mod_names(text: &str) -> Vec<String> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if line.trim() != "#[cfg(test)]" {
            continue;
        }
        let Some(next) = lines.get(i + 1) else {
            continue;
        };
        let decl = next.trim();
        let decl = decl.strip_prefix("pub ").unwrap_or(decl);
        let decl = match decl.find(") ") {
            Some(at) if decl.starts_with("pub(") => &decl[at + 2..],
            _ => decl,
        };
        if let Some(rest) = decl.strip_prefix("mod ")
            && let Some(name) = rest.strip_suffix(';')
        {
            out.push(name.trim().to_string());
        }
    }
    out
}

/// `(line number, code)` for every line outside a `#[cfg(test)]` item, with whole-line
/// and trailing comments dropped.
///
/// # Panics
///
/// If a `#[cfg(test)]` block has no closing brace at the attribute's own indentation.
/// Loudly, on purpose: the alternative is taking the rest of a file out of the scan and
/// reporting that the graph is clean.
fn without_test_items(rel: &str, text: &str) -> Vec<(usize, String)> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        if line.trim() != "#[cfg(test)]" {
            if let Some(code) = code_of(line) {
                out.push((i + 1, code));
            }
            i += 1;
            continue;
        }
        let indent = line.len() - line.trim_start().len();
        let closer = format!("{}}}", " ".repeat(indent));
        // Whatever else decorates the item, and any doc above it.
        let mut item = i + 1;
        while lines
            .get(item)
            .is_some_and(|l| l.trim_start().starts_with('#') || l.trim_start().starts_with("//"))
        {
            item += 1;
        }
        let Some(head) = lines.get(item) else {
            panic!("{rel}:{}: #[cfg(test)] decorates nothing", i + 1);
        };
        if head.trim_end().ends_with(';') {
            // A one-line item: a `use`, or a `mod x;` handled elsewhere.
            i = item + 1;
            continue;
        }
        let end = (item + 1..lines.len())
            .find(|&k| lines[k].trim_end() == closer)
            .unwrap_or_else(|| {
                panic!(
                    "{rel}:{}: no `{closer}` closes this #[cfg(test)] item. The scan \
                     would silently skip the rest of the file, so it stops instead.",
                    i + 1
                )
            });
        i = end + 1;
    }
    out
}

/// A line's code, or `None` if it is only a comment.
///
/// A trailing `//` is cut when nothing before it is quoted — enough to keep a comment
/// that mentions a path from reading as one, without pretending to lex Rust.
fn code_of(line: &str) -> Option<String> {
    if line.trim_start().starts_with("//") {
        return None;
    }
    match line.find("//") {
        Some(at) if !line[..at].contains('"') => Some(line[..at].to_string()),
        _ => Some(line.to_string()),
    }
}

/// The identifier after each `crate::` in a line.
fn crate_paths(line: &str) -> Vec<String> {
    let marker = marker();
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(at) = line[from..].find(&marker) {
        let start = from + at;
        from = start + marker.len();
        // `crate` has to be a whole word: `stark_crate::x` is not ours.
        let leads = line[..start]
            .chars()
            .next_back()
            .is_none_or(|c| !c.is_alphanumeric() && c != '_');
        if !leads {
            continue;
        }
        let rest = &line[from..];
        let end = rest
            .find(|c: char| !c.is_alphanumeric() && c != '_')
            .unwrap_or(rest.len());
        if end > 0 {
            out.push(rest[..end].to_string());
        }
    }
    out
}

/// What the crate root declares and what it re-exports, so `crate::Engine` can be read
/// as the edge to `engine` that it is.
struct Lib {
    modules: BTreeSet<String>,
    /// Re-exported name → the module it comes from.
    reexports: BTreeMap<String, String>,
}

impl Lib {
    /// Which top-level module a `crate::<name>` path reaches. An unplaced name is one of
    /// the root's own items ([`ROOT`]).
    fn resolve(&self, name: &str) -> String {
        if self.modules.contains(name) {
            return name.to_string();
        }
        self.reexports
            .get(name)
            .cloned()
            .unwrap_or_else(|| ROOT.to_string())
    }
}

fn read_lib() -> Lib {
    let text = read(&src_dir().join("lib.rs"));
    let mut modules = BTreeSet::new();
    let mut reexports = BTreeMap::new();

    for line in text.lines() {
        let decl = strip_visibility(line.trim());
        if let Some(rest) = decl.strip_prefix("mod ") {
            let name = rest.trim_end_matches([';', '{']).trim();
            if !name.is_empty() {
                modules.insert(name.to_string());
            }
        }
    }

    // One statement per `;`, so a `use a::{b, c}` spanning lines still reads as one.
    for statement in text.split(';') {
        let Some(at) = statement.rfind("use ") else {
            continue;
        };
        let spec: String = statement[at + 4..]
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect();
        let Some((owner, rest)) = spec.split_once("::") else {
            continue;
        };
        if owner.is_empty() || owner.contains(['{', '}', '(']) {
            continue;
        }
        let names = match (rest.rfind('{'), rest.rfind('}')) {
            (Some(open), Some(close)) if open < close => rest[open + 1..close]
                .split(',')
                .map(str::to_string)
                .collect::<Vec<_>>(),
            _ => vec![rest.rsplit("::").next().unwrap_or(rest).to_string()],
        };
        for name in names {
            let name = name.trim();
            if !name.is_empty() && name != "self" && name != "*" {
                reexports.insert(name.to_string(), owner.to_string());
            }
        }
    }

    Lib { modules, reexports }
}

fn strip_visibility(decl: &str) -> &str {
    if let Some(rest) = decl.strip_prefix("pub ") {
        return rest;
    }
    if decl.starts_with("pub(")
        && let Some(at) = decl.find(") ")
    {
        return &decl[at + 2..];
    }
    decl
}
