//! Compiles WESL shader modules to WGSL at build time (§2).
//!
//! The work is `stark-shaders-build`'s. What is left here is what belongs to *this*
//! crate: the entry-point list, the one thing the shaders cannot say about
//! themselves, and the paths.

use std::path::{Path, PathBuf};

// The one list, shared with `lib.rs` (see there).
include!("src/entry_points.rs");

/// The vendored Mixbox shader (git submodule), source of the pigment-mixing
/// polynomial. Licensed CC BY-NC 4.0 — see `vendor/mixbox/LICENSE`.
const MIXBOX_GLSL: &str = "../../vendor/mixbox/shaders/mixbox.glsl";

/// Where the shader tree lives, relative to this crate.
const SHADER_DIR: &str = "src/shaders";

fn main() {
    // A build script is compiled *without* the crate's features, so `cfg(feature =
    // ..)` is not available here — cargo passes the answer in the environment
    // instead. `entry_points.rs` owns what this then means (`entry_point_enabled`),
    // so the two sides cannot drift.
    let mixbox = std::env::var_os("CARGO_FEATURE_MIXBOX").is_some();
    let entry_points: Vec<&str> = ENTRY_POINTS
        .iter()
        .copied()
        .filter(|name| entry_point_enabled(name, mixbox))
        .collect();
    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));

    stark_shaders_build::run(&stark_shaders_build::Config {
        shader_dir: Path::new(SHADER_DIR),
        out_dir: &out_dir,
        mixbox_glsl: mixbox.then(|| Path::new(MIXBOX_GLSL)),
        entry_points: &entry_points,
        resid_entry_points: RESID_ENTRY_POINTS,
        ceiling_entry_points: CEILING_ENTRY_POINTS,
        resid_feature: RESID_FEATURE,
        ceiling_feature: CEILING_FEATURE,
    });

    // The tree, by directory: cargo scans a named directory recursively, so this
    // covers `lib/` and anything nested under it. `src/entry_points.rs` needs no line
    // of its own — it is compiled into this script, and a script that recompiles is
    // re-run.
    println!("cargo::rerun-if-changed={SHADER_DIR}");
    // Only when it is actually read: naming a path that need not exist would make
    // cargo re-run this script on every build in a configuration without the
    // submodule checked out.
    if mixbox {
        println!("cargo::rerun-if-changed={MIXBOX_GLSL}");
    }
}
