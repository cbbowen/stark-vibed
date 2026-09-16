//! Linking each entry point with its imports and depositing the WGSL.

use std::path::Path;

use crate::{Config, check, collide};

/// Link and deposit every artifact this configuration asks for.
///
/// `gen_dir` is where [`crate::GEN_PREFIX`] resolves, when there is anything mounted
/// under it. A `Router` with nothing there is exactly right: an import of it would be a
/// resolve error, and without Mixbox there is none.
pub(crate) fn compile_all(cfg: &Config<'_>, gen_dir: Option<&Path>) {
    let mut router = wesl::Router::new();
    if let Some(dir) = gen_dir {
        router.mount_resolver(
            crate::GEN_PREFIX
                .parse()
                .expect("the gen prefix is a module path"),
            wesl::FileResolver::new(dir),
        );
    }
    router.mount_fallback_resolver(wesl::FileResolver::new(cfg.shader_dir));
    let mut compiler = wesl::Wesl::new(cfg.shader_dir).set_custom_resolver(router);

    // Two passes over the tree, differing only in whether the tile's residual channel
    // exists (§6.7). `Feature::Disable` is `condcomp`'s default, so the first pass
    // simply does not contain the `@if(resid)` declarations.
    compiler.set_feature(cfg.resid_feature, false);
    compiler.set_feature(cfg.ceiling_feature, false);
    for name in cfg.entry_points {
        build_one(&compiler, cfg.out_dir, name, name);
    }
    // The sweep with its ceiling lane (§6.2): a target the pipeline has to list, so a
    // second artifact rather than a flag.
    compiler.set_feature(cfg.ceiling_feature, true);
    for name in cfg.ceiling_entry_points {
        build_one(&compiler, cfg.out_dir, name, &format!("{name}_ceiling"));
    }
    compiler.set_feature(cfg.ceiling_feature, false);
    // The residual pass belongs to the pigment space, so it goes with it.
    if cfg.mixbox_glsl.is_some() {
        compiler.set_feature(cfg.resid_feature, true);
        for name in cfg.resid_entry_points {
            build_one(&compiler, cfg.out_dir, name, &format!("{name}_resid"));
        }
        compiler.set_feature(cfg.ceiling_feature, true);
        for name in cfg.ceiling_entry_points {
            build_one(
                &compiler,
                cfg.out_dir,
                name,
                &format!("{name}_resid_ceiling"),
            );
        }
    }
}

/// Link `module` with its imports, check the result's bindings, and deposit the WGSL
/// under `artifact`.
///
/// The two names differ only for a residual variant (`stamp` → `stamp_resid`), which
/// is the whole reason this is a function: the artifact name is stated separately, so
/// the module a variant links and the file it lands in are stated independently at the
/// one call site that needs them to differ.
fn build_one(
    compiler: &wesl::Wesl<impl wesl::Resolver>,
    out_dir: &Path,
    module: &str,
    artifact: &str,
) {
    let path = format!("package::{module}");
    let root = path
        .parse()
        .unwrap_or_else(|e| panic!("`{path}` is not a module path: {e}"));
    let compiled = compiler.compile(&root).unwrap_or_else(|e| {
        panic!("failed to build WESL shader `{path}`.\n{e}");
    });
    collide::bindings_do_not_collide(&compiled.syntax, &compiled.sourcemap, artifact);
    // Rendered once and deposited as it stands: `write_artifact` would render the
    // linked tree a second time to write the same bytes the type check just read.
    let wgsl = compiled.to_string();
    check::typechecks(&wgsl, artifact);
    let out = out_dir.join(format!("{artifact}.wgsl"));
    std::fs::write(&out, &wgsl).unwrap_or_else(|e| panic!("write {}: {e}", out.display()));
}
