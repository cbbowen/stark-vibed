//! Linking each entry point with its imports and depositing the WGSL.

use std::path::Path;

use crate::entries::{Build, Entry};
use crate::{Config, check, collide};

/// Link and deposit every artifact this configuration asks for.
///
/// `gen_dir` is where [`crate::GEN_PREFIX`] resolves, when there is anything mounted
/// under it. A `Router` with nothing there is exactly right: an import of it would be a
/// resolve error, and without Mixbox there is none.
pub(crate) fn compile_all(cfg: &Config<'_>, entries: &[Entry<'_>], gen_dir: Option<&Path>) {
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

    for entry in entries {
        let builds = entry.builds();
        let linked: Vec<String> = builds
            .iter()
            .map(|build| {
                // **Every** feature, every time. Setting only the ones this build turns
                // on would leave each pass reading whatever the pass before it left
                // behind, so the artifacts would depend on the order of this loop.
                for axis in cfg.axes {
                    let on = entry
                        .axes
                        .iter()
                        .position(|a| a.feature == axis.feature)
                        .is_some_and(|i| build.on[i]);
                    compiler.set_feature(axis.feature, on);
                }
                build_one(&compiler, cfg.out_dir, &entry.module.path, &build.artifact)
            })
            .collect();
        varies(entry, &builds, &linked);
    }
}

/// Fail unless every axis that is *on* in a build actually changed that build.
///
/// A `@if` spelt wrong, or a feature renamed on one side only, deposits the plain pass
/// under the variant's name: every pipeline is still created, and the variant is
/// silently the shader it was meant to differ from — a residual with no target to
/// write to, a ceiling lane that draws as the dial. Nothing downstream can see it,
/// because the two artifacts are both valid WGSL.
///
/// `linked` is `builds`'s WGSL, in the same order.
fn varies(entry: &Entry<'_>, builds: &[Build], linked: &[String]) {
    // `builds` is the whole cross-product of the axes this build links, so turning one
    // of them off stays inside it.
    let at = |on: &[bool]| {
        let i = builds
            .iter()
            .position(|b| b.on == on)
            .expect("every combination of an entry point's axes was linked");
        (builds[i].artifact.as_str(), linked[i].as_str())
    };
    for (build, wgsl) in builds.iter().zip(linked) {
        for (i, axis) in entry.axes.iter().enumerate() {
            if !build.on[i] {
                continue;
            }
            let mut without = build.on.clone();
            without[i] = false;
            let (base, base_wgsl) = at(&without);
            let feature = axis.feature;
            // A plain `assert!`, not `assert_ne!`: the two are whole shaders, and what
            // the reader needs is which pair, not several hundred lines of WGSL twice.
            assert!(
                wgsl != base_wgsl,
                "`{}` is byte-identical to `{base}`: linking `{}.wesl` with the \
                 `{feature}` feature on changed nothing, so nothing in it is gated on \
                 `@if({feature})`. Both artifacts build and every pipeline is created, \
                 so the variant would simply be the plain pass under another name.",
                build.artifact,
                entry.module.path,
            );
        }
    }
}

/// Link `module` with its imports, check the result's bindings, and deposit the WGSL
/// under `artifact`, returning what was written.
///
/// The two names differ for every variant (`stamp` → `stamp_resid`): the module a
/// variant links and the file it lands in are stated independently.
fn build_one(
    compiler: &wesl::Wesl<impl wesl::Resolver>,
    out_dir: &Path,
    module: &str,
    artifact: &str,
) -> String {
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
    wgsl
}

/// The difference check, over an entry point's builds without linking anything: the
/// pairing is what it is for, and a probe tree would only exercise `wesl`.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::entries::{Axis, discover};
    use crate::tree::Module;

    const RESID: Axis<'static> = Axis {
        feature: "resid",
        what: "a residual",
        ty: "Resid",
        off: "Without",
        on: "With",
        pigment_only: false,
        modules: &["stamp"],
    };

    const CEILING: Axis<'static> = Axis {
        feature: "ceiling",
        what: "a ceiling lane",
        ty: "Lane",
        off: "Plain",
        on: "Ceiling",
        pigment_only: false,
        modules: &["stamp"],
    };

    fn stamp() -> Vec<Module> {
        vec![Module::parse(
            "stamp",
            "@fragment\nfn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.0); }\n",
        )]
    }

    /// Four distinct artifacts pass, which is the shape the sweep actually has.
    #[test]
    fn every_combination_differing_from_its_neighbours_is_accepted() {
        let tree = stamp();
        let entries = discover(&tree, &[RESID, CEILING], true);
        let builds = entries[0].builds();
        let linked: Vec<String> = (0..builds.len()).map(|i| i.to_string()).collect();
        varies(&entries[0], &builds, &linked);
    }

    /// A gate that changed nothing: `stamp_resid_ceiling` came out as `stamp_ceiling`,
    /// so the *ceiling* build is what it is compared against — not the plain one, which
    /// differs from it for the other axis's sake and would have hidden this.
    #[test]
    #[should_panic(expected = "`stamp_resid_ceiling` is byte-identical to `stamp_ceiling`")]
    fn a_gate_that_changes_nothing_under_another_axis_is_still_caught() {
        let tree = stamp();
        let entries = discover(&tree, &[RESID, CEILING], true);
        let builds = entries[0].builds();
        let linked: Vec<String> = builds
            .iter()
            .map(|b| match b.artifact.as_str() {
                "stamp_resid_ceiling" | "stamp_ceiling" => "ceiling".to_string(),
                other => other.to_string(),
            })
            .collect();
        varies(&entries[0], &builds, &linked);
    }
}
