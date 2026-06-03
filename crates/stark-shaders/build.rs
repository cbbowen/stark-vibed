//! Compiles WESL shader modules to WGSL at build time (DESIGN.md §2).
//!
//! Each `build_artifact` call links a module and its imports into a single WGSL
//! string deposited in `OUT_DIR`, retrievable in the crate via `include_wesl!`.

use std::path::Path;

/// The vendored Mixbox shader (git submodule), source of the pigment-mixing
/// polynomial. Licensed CC BY-NC 4.0 — see `vendor/mixbox/LICENSE`.
const MIXBOX_GLSL: &str = "../../vendor/mixbox/shaders/mixbox.glsl";

fn main() {
    // Transpile Mixbox's `mixbox_eval_polynomial` from the vendored GLSL into a
    // WESL module so the trained coefficients stay sourced from the licensed
    // submodule rather than copied into this repo (DESIGN.md §6.7).
    generate_mixbox_poly();

    let compiler = wesl::Wesl::new("src/shaders");

    // The brush stamp rasterization shader (DESIGN.md §6.2).
    compiler.build_artifact(&"package::stamp_oklab".parse().unwrap(), "stamp_oklab");

    // Compositing (pass A) and media/lighting (pass B) shaders (DESIGN.md §6.3).
    compiler.build_artifact(&"package::composite".parse().unwrap(), "composite");
    compiler.build_artifact(&"package::media_oklab".parse().unwrap(), "media_oklab");

    // Mixbox color space: latent→RGB polynomial in the media pass (DESIGN.md §6.7).
    compiler.build_artifact(&"package::media_mixbox".parse().unwrap(), "media_mixbox");

    // Wet-mixing reservoir scan (compute), color-space-agnostic (DESIGN.md §6.2).
    compiler.build_artifact(&"package::mixer".parse().unwrap(), "mixer");

    // Mutable-medium combine pass for subtractive/wet brushes (DESIGN.md §6.2).
    compiler.build_artifact(&"package::medium".parse().unwrap(), "medium");

    println!("cargo::rerun-if-changed=src/shaders");
    println!("cargo::rerun-if-changed={MIXBOX_GLSL}");
}

/// Read `mixbox_eval_polynomial` out of the vendored Mixbox GLSL and emit an
/// equivalent WESL function at `src/shaders/mixbox_poly.wesl` (gitignored). Only
/// written when the output changes, so it doesn't retrigger builds.
fn generate_mixbox_poly() {
    let glsl = std::fs::read_to_string(MIXBOX_GLSL).unwrap_or_else(|e| {
        panic!(
            "cannot read {MIXBOX_GLSL}: {e}. Check out the git submodule: \
             `git submodule update --init vendor/mixbox`"
        )
    });

    // Extract the single `vec3 mixbox_eval_polynomial(vec3 c) {{ ... }}` function.
    // It has no nested braces, so the first `\n}` after it is its close.
    let sig = "vec3 mixbox_eval_polynomial(vec3 c)";
    let start = glsl.find(sig).expect("mixbox_eval_polynomial not found in vendored GLSL");
    let end = glsl[start..]
        .find("\n}")
        .map(|i| start + i + 2)
        .expect("unterminated mixbox_eval_polynomial");
    let func = &glsl[start..end];

    // GLSL → WGSL/WESL transliteration (this function is pure arithmetic).
    let wgsl = func
        .replace(sig, "fn mixbox_eval_polynomial(c: vec3<f32>) -> vec3<f32>")
        .replace("float ", "let ")
        .replace("vec3(", "vec3<f32>(")
        .replace("c[0]", "c.x")
        .replace("c[1]", "c.y")
        .replace("c[2]", "c.z");
    let wgsl = strip_unary_plus(&wgsl);

    let out = format!(
        "// GENERATED at build time from {MIXBOX_GLSL} — do not edit.\n\
         // Mixbox 2.0 (c) 2022 Secret Weapons, authors Sarka Sochorova and Ondrej\n\
         // Jamriska. Licensed CC BY-NC 4.0; see vendor/mixbox/LICENSE.\n\n{wgsl}\n"
    );

    // Write only on change so the file's mtime stays stable (the directory is a
    // `rerun-if-changed` input).
    let path = Path::new("src/shaders/mixbox_poly.wesl");
    let unchanged = std::fs::read_to_string(path).map(|c| c == out).unwrap_or(false);
    if !unchanged {
        std::fs::write(path, out).expect("write generated mixbox_poly.wesl");
    }
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
