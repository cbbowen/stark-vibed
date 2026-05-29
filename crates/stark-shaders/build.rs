//! Compiles WESL shader modules to WGSL at build time (DESIGN.md §2).
//!
//! Each `build_artifact` call links a module and its imports into a single WGSL
//! string deposited in `OUT_DIR`, retrievable in the crate via `include_wesl!`.

fn main() {
    let compiler = wesl::Wesl::new("src/shaders");

    // The canvas → surface presentation shader (DESIGN.md §6.4).
    compiler.build_artifact(&"package::present".parse().unwrap(), "present");

    println!("cargo::rerun-if-changed=src/shaders");
}
