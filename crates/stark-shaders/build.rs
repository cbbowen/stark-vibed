//! Compiles WESL shader modules to WGSL at build time (DESIGN.md §2).
//!
//! Each `build_artifact` call links a module and its imports into a single WGSL
//! string deposited in `OUT_DIR`, retrievable in the crate via `include_wesl!`.

fn main() {
    let compiler = wesl::Wesl::new("src/shaders");

    // The canvas → surface presentation shader (DESIGN.md §6.4).
    compiler.build_artifact(&"package::present".parse().unwrap(), "present");

    // The brush stamp rasterization shader (DESIGN.md §6.2).
    compiler.build_artifact(&"package::stamp_oklab".parse().unwrap(), "stamp_oklab");

    // Compositing (pass A) and media/lighting (pass B) shaders (DESIGN.md §6.3).
    compiler.build_artifact(&"package::composite".parse().unwrap(), "composite");
    compiler.build_artifact(&"package::media_oklab".parse().unwrap(), "media_oklab");

    // Pigment color space: additive stamp + Kubelka–Munk media (DESIGN.md §6.7).
    compiler.build_artifact(&"package::stamp_pigment".parse().unwrap(), "stamp_pigment");
    compiler.build_artifact(&"package::media_pigment".parse().unwrap(), "media_pigment");

    println!("cargo::rerun-if-changed=src/shaders");
}
