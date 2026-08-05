//! Checks across the CPU↔shader boundary (§6.7, §7).
//!
//! Every `#[repr(C)]` uniform in this subsystem is one half of a pair the compiler
//! cannot see across: the shader decides how the lanes are *read*, and nothing on
//! this side knows what it decided. Whatever can be checked here should be, because
//! the failure is quiet — a wgpu validation error at best, a silently misread lane
//! at worst.

/// Pin a uniform struct's size against the WESL declaration it mirrors.
///
/// The convention had been to write the size into the doc comment — which is worse
/// than writing nothing, because a stale number reads as a verified one. Three had
/// drifted by the time this existed: `ViewUniform` said 32 and was 48,
/// `MediaUniform` said 80 and was 96 (`surf_m`, §18.1.2), and `GuideUniform` said
/// 240 and was 304 (the fisheye's second set of poles, §20.8). None of the three
/// was wrong in a way a pixel could show; all three were wrong in the one place a
/// maintainer goes to check.
///
/// So the number moves out of the prose and into the build. It does **not** prove
/// the lanes line up — only the shader-side declaration can say that, which is what
/// `the_stamp_struct_has_the_same_nine_lanes_on_both_sides` reads out of the WESL
/// source for `Stamp`, the one struct here whose mismatch would be silent rather
/// than a validation error. What this catches is the realistic change: a lane
/// appended to one side and not the other.
macro_rules! mirrors_wesl {
    ($t:ty, $bytes:expr) => {
        const _: () = assert!(
            std::mem::size_of::<$t>() == $bytes,
            concat!(
                stringify!($t),
                " is no longer the size of the WESL struct it mirrors",
            ),
        );
    };
}
pub(crate) use mirrors_wesl;

/// The value of a `const NAME` in some linked WESL source, as an `f64`.
///
/// The other half of this module's job. `mirrors_wesl!` pins a *struct*; this pins a
/// *scalar* that both sides compute with — the constants a comment can only ask to
/// match, and whose mismatch is silent by construction because both sides go on
/// producing plausible pixels.
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
///     `lib/paint_common.wesl`'s tooth constants are read through `stamp_oklab()`,
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
