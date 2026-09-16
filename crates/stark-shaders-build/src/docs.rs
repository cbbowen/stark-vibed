//! Carrying a WESL comment over as the generated item's documentation.

/// The opening paragraph of the `//` run a module begins with — its header's summary.
///
/// The first paragraph rather than the whole header, for the reason rustdoc takes the
/// same one: it says which pass this is, and the rest of a header is the kernel, read
/// where the kernel is. Nothing is fenced, since a paragraph has no blank line for a
/// markdown code block to open after.
pub(crate) fn header_summary(src: &str) -> Vec<String> {
    src.lines()
        .map_while(|l| l.trim().strip_prefix("//").map(str::trim_end))
        .take_while(|l| !l.trim().is_empty())
        .map(str::to_string)
        .collect()
}

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

/// The argument throughout is what `lay_out` hands this: everything between the
/// previous member and this one.
#[cfg(test)]
mod tests {
    use super::*;

    /// The header's first paragraph, and where it stops: the blank comment line that
    /// ends it, and the first line that is not a comment at all.
    #[test]
    fn a_modules_opening_paragraph_is_its_summary() {
        assert_eq!(
            header_summary(
                "// Compositing pass A.\n// One instanced quad per tile.\n//\n// The rest.\n\nimport x;\n"
            ),
            [" Compositing pass A.", " One instanced quad per tile."]
        );
    }

    #[test]
    fn a_module_that_opens_with_code_has_no_summary() {
        assert!(header_summary("import x;\n// Not a header.\n").is_empty());
    }

    #[test]
    fn the_comment_run_abutting_a_member_is_its_documentation() {
        assert_eq!(
            doc_lines("    // How far the edge falls off.\n    // In canvas px.\n    "),
            [" How far the edge falls off.", " In canvas px."]
        );
    }

    #[test]
    fn a_blank_line_ends_the_run() {
        assert_eq!(
            doc_lines("    // A note about the struct.\n\n    // This member's own.\n    "),
            [" This member's own."]
        );
    }

    #[test]
    fn a_member_with_nothing_before_it_has_no_documentation() {
        assert!(doc_lines("    ").is_empty());
        assert!(doc_lines("").is_empty());
    }

    /// Markdown would take the indent for a code block and `rustdoc` for a doctest, so
    /// it is fenced as `text`. Shader prose is never Rust.
    ///
    /// A blank line does not close the fence — only the next unindented line does, so
    /// the blank ends up inside it.
    #[test]
    fn an_indented_block_after_a_blank_line_is_fenced_as_text() {
        assert_eq!(
            doc_lines(
                "    // The owed mass:\n    //\n    //     owed = prefix(l)\n    //\n    // and the rest.\n    "
            ),
            [
                " The owed mass:",
                "",
                " ```text",
                "     owed = prefix(l)",
                "",
                " ```",
                " and the rest.",
            ]
        );
    }

    #[test]
    fn an_indented_block_running_to_the_end_is_closed() {
        assert_eq!(
            doc_lines("    // The owed mass:\n    //\n    //     owed = prefix(l)\n    "),
            [
                " The owed mass:",
                "",
                " ```text",
                "     owed = prefix(l)",
                " ```"
            ]
        );
    }

    /// An indent that continues a paragraph is not a code block, so it is left alone.
    #[test]
    fn an_indent_with_no_blank_line_before_it_is_left_alone() {
        assert_eq!(
            doc_lines("    // A sentence that wraps\n    //     onto an indented line.\n    "),
            [" A sentence that wraps", "     onto an indented line."]
        );
    }

    /// **A run that opens indented is not fenced**, because a fence opens only after a
    /// blank line and there is none to open after. `rustdoc` strips the one
    /// conventional leading space and reads the remaining four as a doctest — the very
    /// failure this function exists to prevent. No comment in the tree opens that way.
    #[test]
    fn a_run_whose_first_line_is_indented_is_not_fenced() {
        assert_eq!(
            doc_lines("    //     owed = prefix(l)\n    // and the rest.\n    "),
            ["     owed = prefix(l)", " and the rest."]
        );
    }
}
