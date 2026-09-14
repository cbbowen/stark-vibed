//! No element spells its classes as string `class:` attributes and nothing else.
//!
//! On Dioxus 0.7.10, `class: "a", class: "{b}"` (in either order) compiles and renders an
//! **empty** class. A single formatted string (`class: "a {b}"`) renders what it says, and so
//! does a merge with an `if` among its parts. `widgets::Segmented` lost every run's
//! `.segmented`, and `widgets::SliderTrack` every track's `.slider`, to the first spelling.
//! Render tests in `widgets` guard those two; this guards the spelling everywhere else.

use std::path::{Path, PathBuf};

#[test]
fn no_element_splits_its_classes_across_string_attributes() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut found = Vec::new();
    for path in rust_files(&src) {
        for line in split_class_elements(&read(&path)) {
            found.push(format!("{}:{line}", path.display()));
        }
    }
    assert!(
        found.is_empty(),
        "an element's classes are split across string `class:` attributes, which renders an \
         empty class; write them as one formatted string (`class: \"a {{b}}\"`):\n{}",
        found.join("\n")
    );
}

#[test]
fn the_scan_sees_the_spelling_that_renders_nothing() {
    let flagged = |src: &str| !split_class_elements(src).is_empty();

    assert!(flagged(r#"div { class: "a", class: "{b}", "x" }"#));
    assert!(flagged(r#"div { class: "{b}", class: "a", "x" }"#));
    assert!(flagged(
        "button {\n    class: \"a\",\n    // a note\n    title: \"t\",\n    class: \"{b}\",\n}"
    ));
    // Past a raw identifier and a quoted name, which is where `SliderTrack` hid it.
    assert!(flagged(
        r#"input { r#type: "range", "data-x": "y", class: "slider", class: "{class}", }"#
    ));

    assert!(!flagged(r#"div { class: "a {b}", "x" }"#));
    assert!(!flagged(r#"div { class: "a", class: if on { "b" }, "x" }"#));
    assert!(!flagged(
        r#"button { class: "a", class: "{b}", class: if on { "c" }, "x" }"#
    ));
    // A component prop named `class` is one attribute, and a child's is its own.
    assert!(!flagged(
        r#"Segmented { class: "run", onpick: move |v| { pick(v) }, div { class: "x" } }"#
    ));
}

/// The line numbers of the elements in `source` whose `class:` attributes are two or more
/// string literals with no `if` among them.
fn split_class_elements(source: &str) -> Vec<usize> {
    let bytes = source.as_bytes();
    let mut found = Vec::new();
    for (at, _) in source.match_indices('{') {
        let head = source[..at].trim_end_matches(' ');
        let ident = head
            .bytes()
            .rev()
            .take_while(|b| b.is_ascii_alphanumeric() || *b == b'_')
            .count();
        let starts_lower = ident > 0 && bytes[head.len() - ident].is_ascii_lowercase();
        if !starts_lower {
            continue;
        }
        let (strings, ifs) = class_parts(source, at + 1);
        if strings >= 2 && ifs == 0 {
            found.push(source[..at].matches('\n').count() + 1);
        }
    }
    found
}

/// Walks the `name: value,` attributes opening an element body at `pos`, and counts its
/// `class:` values that are string literals and those that are `if` expressions.
fn class_parts(source: &str, mut pos: usize) -> (usize, usize) {
    let (mut strings, mut ifs) = (0, 0);
    loop {
        let rest = skip_trivia(&source[pos..]);
        let name_len = attribute_name_len(rest);
        if name_len == 0 {
            return (strings, ifs);
        }
        let after_name = rest[name_len..].trim_start_matches(' ');
        if !after_name.starts_with(':') || after_name.starts_with("::") {
            return (strings, ifs);
        }
        let value = after_name[1..].trim_start_matches(' ');
        if &rest[..name_len] == "class" {
            if value.starts_with('"') {
                strings += 1;
            } else if value.starts_with("if ") {
                ifs += 1;
            }
        }
        let value_at = source.len() - value.len();
        match past_value(source, value_at) {
            Some(next) => pos = next,
            None => return (strings, ifs),
        }
    }
}

/// The length of the attribute name `rest` opens with: an identifier, a raw identifier
/// (`r#type`) or a quoted name (`"data-popout"`); 0 if it opens with none.
fn attribute_name_len(rest: &str) -> usize {
    if let Some(quoted) = rest.strip_prefix('"') {
        return quoted.find('"').map_or(0, |end| end + 2);
    }
    let raw = if rest.starts_with("r#") { 2 } else { 0 };
    let ident = rest[raw..]
        .bytes()
        .take_while(|b| b.is_ascii_alphanumeric() || *b == b'_')
        .count();
    if ident == 0 { 0 } else { raw + ident }
}

/// `source` without leading whitespace and `//` comment lines.
fn skip_trivia(mut source: &str) -> &str {
    loop {
        source = source.trim_start();
        match source.strip_prefix("//") {
            Some(comment) => source = comment.split_once('\n').map_or("", |(_, rest)| rest),
            None => return source,
        }
    }
}

/// The index just past the value starting at `at` and its comma, or `None` when the value
/// ends the element's attributes (no comma at depth 0 before the element closes).
fn past_value(source: &str, at: usize) -> Option<usize> {
    let bytes = source.as_bytes();
    let (mut depth, mut in_string, mut i) = (0usize, false, at);
    while i < bytes.len() {
        match bytes[i] {
            b'\\' if in_string => i += 1,
            b'"' => in_string = !in_string,
            _ if in_string => {}
            b'(' | b'{' | b'[' => depth += 1,
            b')' | b'}' | b']' if depth == 0 => return None,
            b')' | b'}' | b']' => depth -= 1,
            b',' if depth == 0 => return Some(i + 1),
            _ => {}
        }
        i += 1;
    }
    None
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
