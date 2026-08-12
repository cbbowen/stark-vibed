//! Compiles WESL shader modules to WGSL at build time (§2).
//!
//! Each entry point in [`ENTRY_POINTS`] is linked with its imports into a single WGSL
//! string deposited in `OUT_DIR`, retrievable in the crate via `include_wesl!`.

use std::path::{Path, PathBuf};

// `build.rs` is a crate root, so its modules resolve beside it rather than in a
// directory named for it. The path keeps the generator out of the package root.
#[path = "build/mirror.rs"]
mod mirror;

// The one list, shared with `lib.rs` (see there).
include!("src/entry_points.rs");

/// The WESL structs the host fills in, and which are therefore generated into Rust
/// rather than transcribed (`build/mirror.rs`).
///
/// The first module of each entry is where the mirror is generated from and what its
/// Rust module is named after; any others declare the same struct for their own
/// pipeline and are checked to agree with it. `View` is the reason that exists: three
/// shaders write it out separately against one host type.
///
/// Kept sorted by module, like [`ENTRY_POINTS`], so a missing entry is visible at a
/// glance.
const MIRRORS: &[(&[&str], &str)] = &[
    (&["blend_common"], "Blend"),
    (&["composite", "matte", "overlay"], "View"),
    (&["dynamics"], "Stamp"),
    (&["fill"], "Fill"),
    (&["filter_common"], "Filter"),
    (&["guides"], "Guide"),
    (&["mask_region"], "Region"),
    (&["media_common"], "Media"),
    (&["merge"], "Merge"),
    (&["resolve"], "Resolve"),
    (&["selection"], "Params"),
    (&["stamp_common"], "TileXform"),
    (&["transform"], "Combine"),
    (&["transform"], "Gated"),
    (&["transform"], "Quad"),
];

/// The WESL constants the host also computes with, generated rather than
/// transcribed (`build/mirror.rs`).
///
/// The other half of the same problem the mirrors solve, and the worse-behaved half:
/// a struct that disagrees is usually a wgpu validation error, while a constant that
/// disagrees leaves both sides rendering perfectly plausible pixels that no longer
/// add up. The tooth's three are the sharpest case — the CPU averages the gate over
/// the ground's rise distribution for the *tool's* half of a transfer and the shader
/// evaluates it per texel for the *canvas* half, so drift is a conservation leak
/// proportional to how far they moved (§6.4).
const CONSTS: &[(&str, &str)] = &[
    ("dynamics", "BAKE_RES"),
    ("dynamics", "BLEED_LADDER_TAPS"),
    ("dynamics", "BLEED_SHARE_LADDER"),
    ("dynamics", "BLEED_SHARE_NEAR"),
    ("filter_common", "CA_CAUCHY_SPAN"),
    ("filter_common", "CA_LAMBDA_BLUE"),
    ("filter_common", "CA_LAMBDA_RED"),
    ("filter_common", "CONTRAST_PIVOT"),
    ("filter_common", "FILTER_CHROMATIC"),
    ("filter_common", "FILTER_COLOR"),
    ("lib/paint_common", "RISE_LIMIT"),
    ("lib/paint_common", "TOOTH_RISE"),
    ("lib/paint_common", "TOOTH_SOFTNESS"),
    ("stamp_common", "SWEEP_SLICES"),
    ("stamp_common", "SWEEP_VERTS"),
];

/// The per-instance records a vertex entry point's `@location` parameters describe,
/// as `(module, entry point, Rust name)`.
///
/// The name is the one thing here the shader cannot supply — a parameter list has no
/// name of its own — so it is written down once, next to the declaration it names.
///
/// `composite`'s record is generated once and used twice: pass A draws the layer
/// stack with it and the brush-dynamics loop composites its working region through
/// the very same shader (§6.3), and the two had a `#[repr(C)]` struct each.
const VERTEX: &[(&str, &str, &str)] = &[
    ("composite", "vs_main", "Instance"),
    ("mask_region", "vs_main", "MaskInstance"),
    ("matte", "vs_main", "MatteInstance"),
    ("overlay", "vs_main", "OverlayInstance"),
    ("stamp", "vs_main", "SegmentInstance"),
];

/// The WESL modules whose `@binding` indices are generated as named consts
/// (`build/mirror.rs::emit_bindings`) — the third transcription of the host/shader
/// boundary after the structs and the constants. `dynamics` is here because its
/// ~37 indices were maintained in three places (the WESL declarations, the
/// bind-group layouts, the bind-group entries) with margin comments as the only
/// map.
const BINDINGS: &[&str] = &["dynamics"];

/// The vendored Mixbox shader (git submodule), source of the pigment-mixing
/// polynomial. Licensed CC BY-NC 4.0 — see `vendor/mixbox/LICENSE`.
const MIXBOX_GLSL: &str = "../../vendor/mixbox/shaders/mixbox.glsl";

/// Where the shader tree lives, relative to this crate.
const SHADER_DIR: &str = "src/shaders";

/// The module prefix generated code is mounted under. Importers say
/// `package::gen::mixbox_poly`, so the import site itself says "build-time
/// generated" — which is half the benefit of it no longer being a file in the tree.
const GEN_PREFIX: &str = "package::gen";

fn main() {
    // A build script is compiled *without* the crate's features, so `cfg(feature =
    // ..)` is not available here — cargo passes the answer in the environment
    // instead. `entry_points.rs` owns what this then means (`entry_point_enabled`),
    // so the two sides cannot drift.
    let mixbox = std::env::var_os("CARGO_FEATURE_MIXBOX").is_some();

    // Transpile Mixbox's `mixbox_eval_polynomial` from the vendored GLSL into a
    // WESL module so the trained coefficients stay sourced from the licensed
    // submodule rather than copied into this repo (§6.7).
    //
    // Skipped entirely without the feature, which is the point of it: the two
    // shaders that import the generated module are the two this build then leaves
    // out, so nothing resolves `package::gen::mixbox_poly` and the submodule need
    // not be checked out at all. A `Router` with nothing mounted under the prefix is
    // exactly right — an import of it would be a resolve error, and there is none.
    let gen_dir = mixbox.then(generate_mixbox_poly);

    // Mirror the host-shared WESL structs into Rust. Read from the *unlinked*
    // sources: the linker mangles `Stamp` to `package__1dynamics_Stamp`, emits it
    // once per artifact that reaches it, and has already stripped whatever no entry
    // point uses — none of which the declaration in the tree has done.
    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    mirror::generate(
        Path::new(SHADER_DIR),
        &out_dir.join("mirror.rs"),
        MIRRORS,
        CONSTS,
        VERTEX,
        BINDINGS,
    );

    // Generated modules resolve out of `OUT_DIR`; everything else out of the tree.
    //
    // The generated file **used to be written into `src/shaders`**, and that one fact
    // is what the whole of this script's freshness apparatus existed to survive: the
    // directory this script *reads* was also one it *wrote*, so the directory's
    // fingerprint was entangled with the script's own output and cargo would
    // sometimes call the artifacts fresh after a shader edit. A stale
    // `composite.wgsl` paired with a freshly built `media_oklab.wgsl` is two halves
    // of two different compositing models — tile-shaped artifacts that survive edits
    // and vanish on `cargo clean`, the worst failure mode there is, because it
    // discredits whatever you happened to be changing at the time.
    //
    // Writing to `OUT_DIR` instead retires the entanglement, the write-only-on-change
    // mtime guard that mitigated it, and the `.gitignore` entry for a generated file
    // sitting in the source tree.
    let mut router = wesl::Router::new();
    if let Some(dir) = &gen_dir {
        router.mount_resolver(
            GEN_PREFIX.parse().expect("the gen prefix is a module path"),
            wesl::FileResolver::new(dir),
        );
    }
    router.mount_fallback_resolver(wesl::FileResolver::new(SHADER_DIR));
    let mut compiler = wesl::Wesl::new(SHADER_DIR).set_custom_resolver(router);

    // Two passes over the tree, differing only in whether the tile's residual channel
    // exists (§6.7). `Feature::Disable` is `condcomp`'s default, so the first pass is
    // the build that was here before this channel was added — the `@if(resid)`
    // declarations simply are not in it.
    compiler.set_feature(RESID_FEATURE, false);
    for name in ENTRY_POINTS {
        if entry_point_enabled(name, mixbox) {
            build_one(&compiler, name, name);
        }
    }
    // The residual pass belongs to the pigment space, so it goes with it: without
    // `mixbox` no space in the build declares a `resid_format`, and these would be
    // eight artifacts nothing could select.
    if mixbox {
        compiler.set_feature(RESID_FEATURE, true);
        for name in RESID_ENTRY_POINTS {
            build_one(&compiler, name, &format!("{name}_resid"));
        }
    }

    // Every module by name, not just the directory — a directory's mtime does not
    // move when a file inside it is edited in place, which is every shader edit.
    // `src/shaders/lib` is walked too: the binding-free leaves live there
    // (`lib/paint_common.wesl` alone reaches six pipelines), so a module missed here
    // is exactly the stale-half failure above.
    for dir in [SHADER_DIR.to_string(), format!("{SHADER_DIR}/lib")] {
        for entry in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("read {dir}: {e}")) {
            let path = entry.expect("shader dir entry").path();
            if path.extension().is_some_and(|e| e == "wesl") {
                println!("cargo::rerun-if-changed={}", path.display());
            }
        }
        println!("cargo::rerun-if-changed={dir}");
    }
    println!("cargo::rerun-if-changed=src/entry_points.rs");
    // Only when it is actually read: naming a path that need not exist would make
    // cargo re-run this script on every build in a configuration without the
    // submodule checked out.
    if mixbox {
        println!("cargo::rerun-if-changed={MIXBOX_GLSL}");
    }
}

/// Link `module` with its imports and deposit the WGSL under `artifact`.
///
/// The two names differ only for a residual variant (`stamp` → `stamp_resid`), which
/// is the whole reason this is a function: `build_artifact` takes the artifact name
/// separately, and the loop that used to pass the module name for both had nowhere to
/// say so.
fn build_one(compiler: &wesl::Wesl<impl wesl::Resolver>, module: &str, artifact: &str) {
    let path = format!("package::{module}");
    compiler.build_artifact(
        &path
            .parse()
            .unwrap_or_else(|e| panic!("`{path}` is not a module path: {e}")),
        artifact,
    );
}

/// Read `mixbox_eval_polynomial` out of the vendored Mixbox GLSL and emit an
/// equivalent WESL function under `OUT_DIR`, returning the directory to mount it
/// from.
fn generate_mixbox_poly() -> PathBuf {
    let glsl = std::fs::read_to_string(MIXBOX_GLSL).unwrap_or_else(|e| {
        panic!(
            "cannot read {MIXBOX_GLSL}: {e}. Check out the git submodule: \
             `git submodule update --init vendor/mixbox`"
        )
    });

    // Extract the single `vec3 mixbox_eval_polynomial(vec3 c) {{ ... }}` function.
    // It has no nested braces, so the first `\n}` after it is its close.
    let sig = "vec3 mixbox_eval_polynomial(vec3 c)";
    let start = glsl
        .find(sig)
        .expect("mixbox_eval_polynomial not found in vendored GLSL");
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

    // Under `OUT_DIR`, which nothing else reads and which cargo already treats as
    // this script's output — so the write can be unconditional, where writing into
    // the source tree needed a read-compare-skip dance to keep its own mtime stable.
    let dir = Path::new(&std::env::var("OUT_DIR").expect("cargo sets OUT_DIR")).join("gen");
    std::fs::create_dir_all(&dir).expect("create the generated-shader dir");
    std::fs::write(dir.join("mixbox_poly.wesl"), out).expect("write generated mixbox_poly.wesl");
    dir
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
