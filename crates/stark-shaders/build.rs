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

/// The structs **two or more modules declare identically** against one host type.
///
/// The first module of each entry is where the mirror is generated from and what its
/// Rust module is named after; the others are checked to agree with it, member for
/// member and offset for offset, and then skipped so discovery does not emit a second
/// copy. `View` is the reason this exists: three shaders write it out separately
/// against one host type, and generating from one of them while ignoring the rest
/// would move the drift rather than remove it.
///
/// Every *other* uniform struct is discovered — see `mirror::generate`. This list is
/// only about which declarations are claimed to be the same declaration.
const SHARED: &[(&[&str], &str)] = &[(&["composite", "matte", "overlay"], "View")];

/// The per-instance records a vertex entry point's `@location` parameters describe,
/// as `(module, entry point, Rust name)`.
///
/// The name is the one thing here the shader cannot supply — a parameter list has no
/// name of its own — so it is written down once, next to the declaration it names.
/// The *membership* is not a choice: `mirror::generate` fails the build for any
/// `@vertex` entry point taking `@location` parameters that is missing here, so the
/// list cannot go stale by the tree gaining one.
///
/// `composite`'s record is generated once and used twice: pass A draws the layer
/// stack with it and the brush-dynamics loop composites its working region through
/// the very same shader (§6.3). One declaration, so the two callers cannot disagree
/// about the record they both feed.
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

/// The module prefix generated code is mounted under. Importers say
/// `package::gen::mixbox_poly`, so the import site itself says "build-time
/// generated" — which is half the benefit of keeping it out of the source tree.
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
        SHARED,
        VERTEX,
    );

    // Generated modules resolve out of `OUT_DIR`; everything else out of the tree.
    //
    // **This script must never write into a directory it also reads.** `src/shaders`
    // is an input, and an input whose fingerprint includes this script's own output is
    // entangled with it — cargo can then call the artifacts fresh after a shader edit.
    // A stale `composite.wgsl` paired with a freshly built `media_oklab.wgsl` is two
    // halves of two different compositing models: tile-shaped artifacts that survive
    // edits and vanish on `cargo clean`, the worst failure mode there is, because it
    // discredits whatever you happened to be changing at the time.
    //
    // Writing to `OUT_DIR` keeps the two apart by construction, which is why there is
    // no mtime guard here and no `.gitignore` entry for a generated shader.
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
    // exists (§6.7). `Feature::Disable` is `condcomp`'s default, so the first pass
    // simply does not contain the `@if(resid)` declarations.
    compiler.set_feature(RESID_FEATURE, false);
    compiler.set_feature(CEILING_FEATURE, false);
    for name in ENTRY_POINTS {
        if entry_point_enabled(name, mixbox) {
            build_one(&compiler, name, name);
        }
    }
    // The sweep with its ceiling lane (§6.2): a target the pipeline has to list,
    // so a second artifact rather than a flag (`CEILING_ENTRY_POINTS`).
    compiler.set_feature(CEILING_FEATURE, true);
    for name in CEILING_ENTRY_POINTS {
        build_one(&compiler, name, &format!("{name}_ceiling"));
    }
    compiler.set_feature(CEILING_FEATURE, false);
    // The residual pass belongs to the pigment space, so it goes with it: without
    // `mixbox` no space in the build declares a `resid_format`, and these would be
    // eight artifacts nothing could select.
    if mixbox {
        compiler.set_feature(RESID_FEATURE, true);
        for name in RESID_ENTRY_POINTS {
            build_one(&compiler, name, &format!("{name}_resid"));
        }
        compiler.set_feature(CEILING_FEATURE, true);
        for name in CEILING_ENTRY_POINTS {
            build_one(&compiler, name, &format!("{name}_resid_ceiling"));
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

/// Link `module` with its imports, check the result's bindings, and deposit the WGSL
/// under `artifact`.
///
/// The two names differ only for a residual variant (`stamp` → `stamp_resid`), which
/// is the whole reason this is a function: the artifact name is stated separately, so
/// the module a variant links and the file it lands in are stated independently at the
/// one call site that needs them to differ.
fn build_one(compiler: &wesl::Wesl<impl wesl::Resolver>, module: &str, artifact: &str) {
    let path = format!("package::{module}");
    let root = path
        .parse()
        .unwrap_or_else(|e| panic!("`{path}` is not a module path: {e}"));
    let compiled = compiler.compile(&root).unwrap_or_else(|e| {
        panic!("failed to build WESL shader `{path}`.\n{e}");
    });
    bindings_do_not_collide(&compiled.syntax, artifact);
    let wgsl = compiled.to_string();
    typechecks(&wgsl, artifact);
    compiled.write_artifact(artifact);
}

/// Fail unless the linked WGSL passes the same front end `wgpu` will run on it.
///
/// **`wesl`'s own validation is not a type check.** It resolves names, counts call
/// arguments and rejects cycles — real checks, and the ones that catch a typo — but it
/// has no opinion about types. A `Sweep` passed to a parameter declared `vec3<f32>`
/// linked without complaint, deposited an artifact, and failed at
/// `create_shader_module`: at run time, on a GPU, in the half of the suite CI runs
/// without comparing pixels. It cost eleven red tests and a bisect to find something
/// the compiler knew.
///
/// So the artifact is parsed and validated here, by `naga` — which is what `wgpu`
/// itself uses, so this is the *same* answer, moved from the run to the build and
/// attributed to the shader that earned it.
///
/// Validated with no capabilities beyond the default set, which is the honest bound: a
/// shader this rejects is one some target would reject too.
fn typechecks(wgsl: &str, artifact: &str) {
    let module = naga::front::wgsl::parse_str(wgsl).unwrap_or_else(|e| {
        panic!(
            "`{artifact}` is not valid WGSL:\n{}",
            e.emit_to_string(wgsl)
        );
    });
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    )
    .validate(&module)
    .unwrap_or_else(|e| {
        panic!(
            "`{artifact}` does not validate:\n{}",
            e.emit_to_string(wgsl)
        );
    });
}

/// Fail unless the modules a pipeline links agree about where each one's share of a
/// group's index space stops.
///
/// A pipeline's bindings come from several files — `blend_mixbox` takes 0–4 from
/// `blend_common`, 5–6 from `mixbox_lut` and 7–8 from itself — and that partition is
/// held by nothing but a comment in each. `mixbox_lut.wesl` says so on its face: *"If
/// `blend_common` ever grows a sixth binding, it collides here, and the error will name
/// a mangled identifier rather than any of the files."* This is that check, taken where
/// the answer is: the linked artifact holds exactly the declarations one pipeline
/// compiles, post-`@if`, imports resolved — so the collision is arithmetic here, in
/// terms of the two *files*, rather than a `naga` error naming
/// `package__1mixbox_lut_pigment_lut` at pipeline creation.
///
/// **Only collisions between two different modules are a fault**, and that is the
/// distinction the hazard is actually about rather than a tolerance. A module may
/// deliberately declare two things at one slot when no entry point reaches both:
/// `transform.wesl` puts `Quad` and `Gated` at `@group(0) @binding(0)` because the
/// affine and the rect-scoped maps are different pipelines, and its header carries the
/// rule that keeps it sound ("if a fourth map is added, give it its own module rather
/// than a fourth struct here"). One file can state that about itself; two files
/// splitting a group cannot, which is why one is checked and the other is not.
fn bindings_do_not_collide(linked: &wesl::syntax::TranslationUnit, artifact: &str) {
    use wesl::syntax::{Attribute, GlobalDeclaration};
    use wesl::{EscapeMangler, Mangler};

    // An `ExpressionNode` renders back to its source text, which for the literal every
    // one of these is *is* the number. A const-evaluated `@binding` would need the eval
    // context the linker has already discharged, so it is passed over rather than
    // guessed at — and there are none in this tree.
    let literal = |e: &wesl::syntax::ExpressionNode| e.to_string().trim().parse::<u32>().ok();
    // The root module's own declarations are not mangled, so `unmangle` returning
    // `None` *is* the answer "this one is the root's".
    let source = |name: &str| {
        EscapeMangler
            .unmangle(name)
            .map_or_else(|| artifact.to_string(), |(path, _)| path.to_string())
    };

    let mut seen: Vec<(u32, u32, String, String)> = Vec::new();
    for d in &linked.global_declarations {
        let GlobalDeclaration::Declaration(decl) = &**d else {
            continue;
        };
        let g = decl.attributes.iter().find_map(|a| match &**a {
            Attribute::Group(e) => literal(e),
            _ => None,
        });
        let b = decl.attributes.iter().find_map(|a| match &**a {
            Attribute::Binding(e) => literal(e),
            _ => None,
        });
        let (Some(g), Some(b)) = (g, b) else { continue };
        let name = decl.ident.name().to_string();
        let from = source(&name);
        if let Some((_, _, other, other_from)) = seen
            .iter()
            .find(|(sg, sb, _, sf)| *sg == g && *sb == b && *sf != from)
        {
            panic!(
                "`{artifact}` links `{other_from}`'s `{other}` and `{from}`'s `{name}` \
                 at the same `@group({g}) @binding({b})`. The modules a pipeline links \
                 partition a group's index space between them (see `mixbox_lut.wesl`), \
                 and two of them have claimed one slot."
            );
        }
        seen.push((g, b, name, from));
    }
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
