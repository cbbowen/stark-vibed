//! Transliterating Mixbox's pigment polynomial from the vendored GLSL.

use std::path::{Path, PathBuf};

/// Read `mixbox_eval_polynomial` out of the vendored Mixbox GLSL at `glsl` and emit an
/// equivalent WESL function under `out_dir`, returning the directory to mount it from.
///
/// Transpiled rather than copied so the trained coefficients stay sourced from the
/// licensed submodule (CC BY-NC 4.0; see `vendor/mixbox/LICENSE`) rather than living in
/// this repo (§6.7).
pub(crate) fn generate(glsl: &Path, out_dir: &Path) -> PathBuf {
    let at = glsl.display().to_string();
    let source = std::fs::read_to_string(glsl).unwrap_or_else(|e| {
        panic!(
            "cannot read {at}: {e}. Check out the git submodule: \
             `git submodule update --init vendor/mixbox`"
        )
    });

    // Under `OUT_DIR`, which nothing else reads and which cargo already treats as this
    // script's output — so the write can be unconditional, where writing into the
    // source tree needed a read-compare-skip dance to keep its own mtime stable.
    let dir = out_dir.join("gen");
    std::fs::create_dir_all(&dir).expect("create the generated-shader dir");
    std::fs::write(dir.join("mixbox_poly.wesl"), transliterate(&source, &at))
        .expect("write generated mixbox_poly.wesl");
    dir
}

/// The WESL module, from the GLSL that `from` names.
fn transliterate(source: &str, from: &str) -> String {
    // Extract the single `vec3 mixbox_eval_polynomial(vec3 c) { ... }` function.
    // It has no nested braces, so the first `\n}` after it is its close.
    let sig = "vec3 mixbox_eval_polynomial(vec3 c)";
    let start = source
        .find(sig)
        .expect("mixbox_eval_polynomial not found in vendored GLSL");
    let end = source[start..]
        .find("\n}")
        .map(|i| start + i + 2)
        .expect("unterminated mixbox_eval_polynomial");

    // GLSL → WGSL/WESL transliteration (this function is pure arithmetic).
    let wgsl = source[start..end]
        .replace(sig, "fn mixbox_eval_polynomial(c: vec3<f32>) -> vec3<f32>")
        .replace("float ", "let ")
        .replace("vec3(", "vec3<f32>(")
        .replace("c[0]", "c.x")
        .replace("c[1]", "c.y")
        .replace("c[2]", "c.z");
    let wgsl = strip_unary_plus(&wgsl);

    format!(
        "// GENERATED at build time from {from} — do not edit.\n\
         // Mixbox 2.0 (c) 2022 Secret Weapons, authors Sarka Sochorova and Ondrej\n\
         // Jamriska. Licensed CC BY-NC 4.0; see vendor/mixbox/LICENSE.\n\n{wgsl}\n"
    )
}

/// Drop GLSL unary `+` before numeric literals; WGSL has no unary-plus operator.
/// Binary `+` (term separators) is always followed by whitespace, so it's safe.
fn strip_unary_plus(s: &str) -> String {
    let ch: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    for i in 0..ch.len() {
        if ch[i] == '+' && ch.get(i + 1).is_some_and(|c| c.is_ascii_digit()) {
            continue;
        }
        out.push(ch[i]);
    }
    out
}
