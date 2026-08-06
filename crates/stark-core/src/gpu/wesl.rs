//! Checks across the CPU↔shader boundary (§6.7, §7).
//!
//! What remains here is the **constants**. The structs used to be checked here too,
//! by a `mirrors_wesl!` that pinned a Rust uniform's size against a number written
//! beside it — a mechanism that existed because three declarations had already
//! drifted in the one place a maintainer goes to check (`ViewUniform` said 32 and
//! was 48; `MediaUniform` said 80 and was 96, §18.1.2; `GuideUniform` said 240 and
//! was 304, §20.8). It only ever pinned the size, so a permuted lane passed it.
//!
//! Every one of those structs is now *generated* from the WESL that reads it
//! (`stark-shaders/build/mirror.rs`), which is why the macro is gone rather than
//! improved: there is no second declaration left for it to check. A constant is the
//! remaining case, because both sides still write one out and both go on producing
//! plausible pixels when they disagree.

/// The value of a `const NAME` in some linked WESL source, as an `f64`.
///
/// The constants a comment can only ask to match, and whose mismatch is silent by
/// construction.
///
/// Enough of a parser for a scalar `const`, and no more: anything it cannot find is a
/// failed test rather than a silently skipped one. Three limits worth knowing before
/// reaching for it:
///
///   · **Stripping.** The WESL linker drops declarations no entry point reaches, so a
///     constant that survives only in prose cannot be read at all. Check it through
///     one the shader computes with: `dynamics.wesl`'s `WICK_RATE` is checked through
///     `WICK_HALF`, and `stamp_common.wesl`'s `SWEEP_VERTS` through `SWEEP_SLICES`.
///   · **Reachability.** It reads the *linked* artifact, so the constant must be
///     reachable from the entry points of whichever module you pass.
///     `lib/paint_common.wesl`'s tooth constants are read through `stamp()`,
///     whose fragment stage gates its deposit on them.
///   · **Comparison.** It returns `f64` because that is what a decimal literal parses
///     to, but both sides hold `f32`. Narrow before asserting (`… as f32`), or a
///     constant that is not a power of two fails on the widening alone: the host's
///     `0.06f32` widens to 0.059999998…, which is not the source's `0.06`.
///   · **Mangling.** A constant the root module *imported* arrives renamed —
///     `TOOTH_RISE` links as `package_lib__1paint_common__1TOOTH_RISE` — while one
///     declared in the root module keeps its name (the linker does not mangle root
///     declarations). Hence the suffix match below, which is why this scans lines
///     rather than looking for the literal `const NAME:`.
#[cfg(test)]
pub(crate) fn wesl_const(src: &str, name: &str) -> f64 {
    let decl = src
        .lines()
        .map(str::trim_start)
        .find_map(|line| {
            let rest = line.strip_prefix("const ")?;
            let (ident, value) = rest.split_once(':')?;
            // Either the root module's own name, or an import's mangled form, which
            // is always the qualified path with the original name on the end.
            let ident = ident.trim();
            (ident == name || (ident.starts_with("package") && ident.ends_with(name)))
                .then_some(value)
        })
        .unwrap_or_else(|| panic!("the linked shader has no `const {name}` (stripped?)"));
    let eq = decl.find('=').expect("a const has a value");
    let end = decl.find(';').expect("a const ends");
    decl[eq + 1..end]
        .trim()
        .trim_end_matches(['u', 'i', 'f'])
        .parse()
        .unwrap_or_else(|e| panic!("`const {name}` is not a scalar: {e}"))
}
