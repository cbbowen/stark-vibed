//! Carrying a WESL comment over as the generated item's documentation.

/// The `//` comment run immediately preceding a member, as doc-comment text.
///
/// Walked backwards from the member, because what makes a comment *this member's*
/// documentation is that nothing stands between the two. The final line of `between`
/// is the member's own indentation and is skipped; anything else that is not a `//`
/// line ends the run.
pub(crate) fn doc_lines(between: &str) -> Vec<String> {
    let mut lines: Vec<&str> = between.lines().collect();
    if lines.last().is_some_and(|l| l.trim().is_empty()) {
        lines.pop();
    }
    let mut docs: Vec<String> = lines
        .iter()
        .rev()
        .map_while(|l| {
            l.trim()
                .strip_prefix("//")
                .map(|r| r.trim_end().to_string())
        })
        .collect();
    docs.reverse();
    fence_indented(docs)
}

/// Fence every indented run of carried-over shader prose as `text`.
///
/// **Shader prose is never Rust.** Markdown reads a four-space indent after a blank
/// line as a code block, and `rustdoc` then reads that block as a *doctest* — so
/// `dynamics.wesl`'s
///
/// ```text
///     owed = prefix(l),   received = rowtotal − prefix(l)
/// ```
///
/// came through as two failing doctests complaining that `−` is not a Rust token. The
/// indentation is the shader author's, marking an equation or a table; every one of
/// them is prose about a WGSL kernel and none is a Rust example. Fencing as `text` is
/// what says so, and it keeps the layout the author chose rather than flattening it.
///
/// Only runs that markdown would actually take as code are fenced — an indent that
/// continues a paragraph, with no blank line before it, is left alone.
fn fence_indented(docs: Vec<String>) -> Vec<String> {
    // A doc line carries one conventional leading space (`// Text` → ` Text`), so a
    // markdown code indent is that plus four.
    let indented = |l: &String| l.starts_with("     ");
    let blank = |l: &String| l.trim().is_empty();
    let mut out: Vec<String> = Vec::with_capacity(docs.len());
    let mut open = false;
    for line in docs {
        if open && !indented(&line) && !blank(&line) {
            out.push(" ```".to_string());
            open = false;
        } else if !open && indented(&line) && out.last().is_some_and(blank) {
            out.push(" ```text".to_string());
            open = true;
        }
        out.push(line);
    }
    if open {
        out.push(" ```".to_string());
    }
    out
}
