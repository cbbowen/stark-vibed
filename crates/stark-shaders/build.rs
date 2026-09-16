//! Compiles WESL shader modules to WGSL at build time (§2).
//!
//! The work is `stark-shaders-build`'s. What is left here is what belongs to *this*
//! crate: the entry-point list, the two things the shaders cannot say about
//! themselves, and the paths.

use std::path::{Path, PathBuf};

// The one list, shared with `lib.rs` (see there).
include!("src/entry_points.rs");

/// The structs **two or more modules declare identically** against one host type.
///
/// `View` is the reason this exists: three shaders write it out separately against one
/// host type, and generating from one of them while ignoring the rest would move the
/// drift rather than remove it. Every *other* uniform struct is discovered.
const SHARED: &[(&[&str], &str)] = &[(&["composite", "matte", "overlay"], "View")];

/// The per-instance records a vertex entry point's `@location` parameters describe, as
/// `(module, entry point, Rust name)`.
///
/// The name is the one thing here the shader cannot supply — a parameter list has no
/// name of its own. The *membership* is not a choice: the generator fails the build for
/// any `@vertex` entry point taking `@location` parameters that is missing here.
///
/// `composite`'s record is generated once and used twice: pass A draws the layer stack
/// with it and the brush-dynamics loop composites its working region through the very
/// same shader (§6.3).
const VERTEX: &[(&str, &str, &str)] = &[
    ("composite", "vs_main", "Instance"),
    ("mask_region", "vs_main", "MaskInstance"),
    ("matte", "vs_main", "MatteInstance"),
    ("overlay", "vs_main", "OverlayInstance"),
    ("stamp", "vs_main", "SegmentInstance"),
];

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
        shared: SHARED,
        vertex: VERTEX,
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
