//! Which modules become artifacts, and along which axes (§6.10).
//!
//! **Discovered, not listed.** A module declaring an `@vertex`, `@fragment` or
//! `@compute` function links as its own artifact; every other module in the tree is
//! reached only by import and would fail to link as a root. So is the pigment set: a
//! module *reaching* the transpiled polynomial ([`GEN_PREFIX`](crate::GEN_PREFIX)),
//! through any depth of import, is one the `mixbox` cargo feature builds.
//!
//! What stays declared is what a shader cannot say about itself: which *host* choice
//! an `@if(feature)` gate is.

use wesl::syntax::{GlobalDeclaration, ImportContent, ImportStatement, ModulePath};

use crate::tree::Module;

/// The axis a generated `Binding`'s `resid` flag reports.
///
/// [`crate::emit`]'s binding tables read `@if(resid)` off a declaration to say whether
/// the residual build is the only one that has the slot, so the name is this crate's as
/// well as the caller's table's — and naming it here keeps the two from being two
/// spellings that happen to match.
pub const RESID_FEATURE: &str = "resid";

/// One axis a shader is linked along a second time: a WESL conditional-compilation
/// feature, and what the host calls the choice between the two builds.
#[derive(Debug, Clone, Copy)]
pub struct Axis<'a> {
    /// The feature the shader's `@if` names — and the suffix its artifact takes.
    pub feature: &'a str,
    /// What the feature turns on, as a noun phrase: the generated type's
    /// documentation.
    pub what: &'a str,
    /// The generated Rust enum, and its off/on variants. One word, since the
    /// accessor's parameter is the type name lowercased.
    pub ty: &'a str,
    pub off: &'a str,
    pub on: &'a str,
    /// Whether the axis exists at all only in a build carrying the pigment space — in
    /// which case its `on` variant is not generated either, so the choice is
    /// unrepresentable rather than merely wrong.
    pub pigment_only: bool,
    /// The entry points linked along it, sorted.
    ///
    /// Checked in one direction: each named module must be an entry point, and turning
    /// the feature on must change its artifact. A module the feature *would* change
    /// that is missing from this list is **not** caught — that would cost a second link
    /// of every entry point on every axis.
    pub modules: &'a [&'a str],
}

impl Axis<'_> {
    /// Whether this build links the axis's `on` side at all — the one predicate that
    /// decides both which artifacts exist and whether the variant naming them is
    /// generated.
    pub(crate) fn live(&self, pigment: bool) -> bool {
        pigment || !self.pigment_only
    }
}

/// One module that links as its own artifact.
pub(crate) struct Entry<'a> {
    pub(crate) module: &'a Module,
    /// The axes that name this module, in declaration order — every one of them,
    /// including an axis this build does not link, whose parameter the accessor still
    /// takes with only its `off` variant to pass.
    pub(crate) axes: Vec<&'a Axis<'a>>,
    /// Which of [`Self::axes`] this build links the `on` side of, in the same order.
    ///
    /// Settled once, here, rather than passed to everything that asks: the linker and
    /// the generator disagreeing about it is an accessor naming an artifact nobody
    /// deposited, and a `pigment` flag threaded through both is a way for them to.
    live: Vec<bool>,
}

/// One artifact of an entry point.
pub(crate) struct Build {
    /// Which of [`Entry::axes`] are on, in the same order. An axis this build does not
    /// link is always `false`.
    pub(crate) on: Vec<bool>,
    /// The file it is deposited as, without the extension.
    pub(crate) artifact: String,
}

impl Entry<'_> {
    /// Every artifact this entry point is deposited as: the plain build, then one per
    /// combination of the axes this build links.
    pub(crate) fn builds(&self) -> Vec<Build> {
        let live: Vec<usize> = (0..self.axes.len()).filter(|&i| self.live[i]).collect();
        (0..1usize << live.len())
            .map(|mask| {
                let mut on = vec![false; self.axes.len()];
                for (bit, &i) in live.iter().enumerate() {
                    on[i] = (mask >> bit) & 1 == 1;
                }
                // The Rust module name, not the WESL path: `OUT_DIR` is flat, and
                // `read_tree` already refuses two modules that share this name.
                let mut artifact = self.module.rust.clone();
                for (axis, is_on) in self.axes.iter().zip(&on) {
                    if *is_on {
                        artifact.push('_');
                        artifact.push_str(axis.feature);
                    }
                }
                Build { on, artifact }
            })
            .collect()
    }
}

/// The tree's entry points, each with the axes it varies on.
///
/// `pigment` is whether this build carries the Mixbox space: it drops the shaders that
/// import the polynomial, and with them every axis only a pigment space could select.
///
/// # Panics
/// On anything the declared table gets wrong — two axes claiming one feature or one
/// host type, a module list out of order, a module that declares no entry point. Left
/// alone each shows up as an artifact nobody deposited, several frames from the line
/// that caused it.
pub(crate) fn discover<'a>(
    modules: &'a [Module],
    axes: &'a [Axis<'a>],
    pigment: bool,
) -> Vec<Entry<'a>> {
    let generated: ModulePath = crate::GEN_PREFIX
        .parse()
        .expect("the gen prefix is a module path");
    let entries: Vec<&Module> = modules.iter().filter(|m| has_entry_point(m)).collect();

    for (i, axis) in axes.iter().enumerate() {
        let before = &axes[..i];
        assert!(
            !before.iter().any(|a| a.feature == axis.feature),
            "two axes name the `{}` feature. The linker sets a feature per axis, so the \
             second would be the only one a build ever saw.",
            axis.feature,
        );
        assert!(
            !before.iter().any(|a| a.ty == axis.ty),
            "two axes generate `{}`, which would be one Rust enum defined twice.",
            axis.ty,
        );
        assert!(
            axis.ty
                .strip_prefix(|c: char| c.is_ascii_uppercase())
                .is_some_and(|rest| rest.chars().all(|c| c.is_ascii_alphanumeric())),
            "the `{}` axis generates `{}`, which is not one capitalized word — the \
             accessor's parameter is that name lowercased, and anything else makes a \
             parameter no reader would connect to the type.",
            axis.feature,
            axis.ty,
        );
        assert!(
            axis.modules.windows(2).all(|w| w[0] < w[1]),
            "the `{}` axis's modules are not sorted, which is what makes a missing one \
             visible at a glance.",
            axis.feature,
        );
        for name in axis.modules {
            assert!(
                entries.iter().any(|m| m.path == *name),
                "the `{}` axis names `{name}`, which declares no `@vertex`, `@fragment` \
                 or `@compute` entry point — so it is not a module that links as an \
                 artifact of its own, and there is nothing to build a variant of.",
                axis.feature,
            );
        }
    }

    entries
        .iter()
        .copied()
        .filter(|m| pigment || !reaches_generated(m, modules, &generated))
        .map(|module| {
            let axes: Vec<&Axis<'_>> = axes
                .iter()
                .filter(|a| a.modules.contains(&module.path.as_str()))
                .collect();
            Entry {
                live: axes.iter().map(|a| a.live(pigment)).collect(),
                axes,
                module,
            }
        })
        .collect()
}

/// Whether `m` declares a shader entry point, and so links as an artifact of its own.
fn has_entry_point(m: &Module) -> bool {
    m.tu.global_declarations.iter().any(|d| match &**d {
        GlobalDeclaration::Function(f) => f.attributes.iter().any(|a| a.is_entry_point()),
        _ => false,
    })
}

/// Whether `m` reaches anything under `prefix`, through any depth of import.
///
/// **Transitive, because the polynomial is behind a helper.** `lib/mixbox.wesl` holds
/// the pigment round trip and imports the transpiled polynomial; its three importers
/// name neither. Asking only about a module's own imports would leave all three looking
/// colorimetric, and a build without the pigment space would keep them and then fail to
/// resolve `package::gen` while linking them.
///
fn reaches_generated(m: &Module, modules: &[Module], prefix: &ModulePath) -> bool {
    let mut seen: Vec<&str> = vec![m.path.as_str()];
    let mut stack: Vec<&Module> = vec![m];
    while let Some(at) = stack.pop() {
        if at.tu.imports.iter().any(|st| reaches(st, prefix)) {
            return true;
        }
        for path in imported_modules(at, prefix) {
            let Some(next) = modules.iter().find(|other| other.path == path) else {
                continue;
            };
            if !seen.contains(&next.path.as_str()) {
                seen.push(next.path.as_str());
                stack.push(next);
            }
        }
    }
    false
}

/// Every module path `m`'s imports name, as the tree spells them (`lib/color`).
///
/// A candidate list rather than a resolution: an import ends in an *item*, and
/// `import package::lib::color;` would name the module itself, so both readings are
/// offered and [`reaches_generated`] keeps whichever the tree holds.
///
/// Only imports rooted the way `origin` is, because that is what makes the components
/// a tree path: `super::stamp::{x}` from a `lib/` module would otherwise offer `stamp`,
/// which *is* a module here, and the walk would follow an edge that does not exist.
fn imported_modules(m: &Module, origin: &ModulePath) -> Vec<String> {
    let mut out = Vec::new();
    for st in &m.tu.imports {
        if let Some(path) = &st.path
            && path.origin == origin.origin
        {
            walk_content(&path.components, &st.content, &mut out);
        }
    }
    out
}

/// One import's own module paths, appended — an item's module and the item read as a
/// module, and a collection's every branch.
fn walk_content(base: &[String], content: &ImportContent, out: &mut Vec<String>) {
    match content {
        ImportContent::Item(item) => {
            out.push(base.join("/"));
            let mut full = base.to_vec();
            full.push(item.ident.name().to_string());
            out.push(full.join("/"));
        }
        ImportContent::Collection(items) => {
            for i in items {
                let mut full = base.to_vec();
                full.extend(i.path.iter().cloned());
                walk_content(&full, &i.content, out);
            }
        }
    }
}

/// Whether one import statement names anything under `prefix`.
///
/// The nesting is walked rather than the statement's own path matched, because
/// `import package::{gen::poly::x, lib::color::y}` puts the components inside the
/// collection — the same import by another spelling, and one a prefix test on the
/// statement alone would read as "not the generated module".
fn reaches(st: &ImportStatement, prefix: &ModulePath) -> bool {
    let Some(path) = &st.path else { return false };
    path.origin == prefix.origin && content_reaches(&path.components, &st.content, prefix)
}

fn content_reaches(base: &[String], content: &ImportContent, prefix: &ModulePath) -> bool {
    let under = |full: &[String]| full.starts_with(&prefix.components);
    match content {
        ImportContent::Item(item) => {
            let mut full = base.to_vec();
            full.push(item.ident.name().to_string());
            under(&full)
        }
        ImportContent::Collection(items) => items.iter().any(|i| {
            let mut full = base.to_vec();
            full.extend(i.path.iter().cloned());
            under(&full) || content_reaches(&full, &i.content, prefix)
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `resid`'s shape: an axis only a build with the pigment space links.
    const RESID: Axis<'static> = Axis {
        feature: "resid",
        what: "a residual",
        ty: "Resid",
        off: "Without",
        on: "With",
        pigment_only: true,
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

    fn tree() -> Vec<Module> {
        vec![
            Module::parse(
                "stamp",
                "@fragment\nfn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.0); }\n",
            ),
            Module::parse(
                "media_mixbox",
                "import package::gen::mixbox_poly::{mixbox_eval_polynomial};\n\
                 @fragment\nfn fs_main() -> @location(0) vec4<f32> {\n\
                 \x20   return vec4<f32>(mixbox_eval_polynomial(0.0));\n}\n",
            ),
            // Reached only by import: no entry point of its own.
            Module::parse("lib/color", "const K: f32 = 1.0;\n"),
        ]
    }

    fn named<'a>(entries: &'a [Entry<'a>]) -> Vec<&'a str> {
        entries.iter().map(|e| e.module.path.as_str()).collect()
    }

    /// A `@fragment` is what makes a module an artifact; a binding-free leaf is not one.
    #[test]
    fn a_module_declaring_an_entry_point_is_one() {
        let tree = tree();
        assert_eq!(
            named(&discover(&tree, &[], true)),
            ["stamp", "media_mixbox"]
        );
    }

    /// The pigment shaders leave with the polynomial they import — which is what
    /// unlinks the vendored CC BY-NC 4.0 code rather than merely leaving it unreached.
    #[test]
    fn a_shader_importing_the_polynomial_is_left_out_without_the_pigment_space() {
        let tree = tree();
        assert_eq!(named(&discover(&tree, &[], false)), ["stamp"]);
    }

    /// The shape the tree actually has: the polynomial is behind `lib/mixbox`, and its
    /// importers name only that. One level of indirection was enough to keep every
    /// pigment shader in a build that cannot resolve `package::gen`.
    #[test]
    fn a_shader_reaching_the_polynomial_through_a_leaf_is_left_out_too() {
        let tree = [
            Module::parse(
                "lib/mixbox",
                "import package::gen::mixbox_poly::{mixbox_eval_polynomial};\n\
                 fn to_linear(c: vec3<f32>) -> vec3<f32> {\n\
                 \x20   return mixbox_eval_polynomial(c);\n}\n",
            ),
            Module::parse(
                "media_mixbox",
                "import package::lib::mixbox::{to_linear};\n\
                 @fragment\nfn fs_main() -> @location(0) vec4<f32> {\n\
                 \x20   return vec4<f32>(to_linear(vec3<f32>(0.0)), 1.0);\n}\n",
            ),
            Module::parse(
                "media_oklab",
                "@fragment\nfn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.0); }\n",
            ),
        ];
        assert_eq!(named(&discover(&tree, &[], false)), ["media_oklab"]);
        assert_eq!(
            named(&discover(&tree, &[], true)),
            ["media_mixbox", "media_oklab"]
        );
    }

    /// The collection spelling of the same import. Nothing in the tree writes it this
    /// way today, and a prefix test on the statement's own path would miss it.
    #[test]
    fn the_collection_spelling_of_a_generated_import_is_still_one() {
        let tree = [Module::parse(
            "probe",
            "import package::{gen::mixbox_poly::{mixbox_eval_polynomial}, lib::color::{K}};\n\
             @fragment\nfn fs_main() -> @location(0) vec4<f32> {\n\
             \x20   return vec4<f32>(mixbox_eval_polynomial(K));\n}\n",
        )];
        assert!(discover(&tree, &[], false).is_empty());
    }

    /// An importer of something else is not mistaken for one.
    #[test]
    fn a_shader_importing_no_generated_module_stays() {
        let tree = [Module::parse(
            "probe",
            "import package::lib::color::{K};\n\
             @fragment\nfn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(K); }\n",
        )];
        assert_eq!(named(&discover(&tree, &[], false)), ["probe"]);
    }

    /// The artifact names, and the order the suffixes come out in: the axes'.
    #[test]
    fn an_entry_point_is_built_once_per_combination_of_its_axes() {
        let tree = tree();
        let entries = discover(&tree, &[RESID, CEILING], true);
        let builds = entries[0].builds();
        let names: Vec<&str> = builds.iter().map(|b| b.artifact.as_str()).collect();
        assert_eq!(
            names,
            [
                "stamp",
                "stamp_resid",
                "stamp_ceiling",
                "stamp_resid_ceiling"
            ]
        );
    }

    /// Without the pigment space the residual axis is not linked, so its parameter
    /// survives with one value and its artifacts do not exist.
    #[test]
    fn an_axis_this_build_does_not_link_contributes_no_artifact() {
        let tree = tree();
        let entries = discover(&tree, &[RESID, CEILING], false);
        let builds = entries[0].builds();
        let names: Vec<&str> = builds.iter().map(|b| b.artifact.as_str()).collect();
        assert_eq!(names, ["stamp", "stamp_ceiling"]);
        // Two axes still, so the accessor still takes two parameters.
        assert!(builds.iter().all(|b| b.on.len() == 2));
        assert!(builds.iter().all(|b| !b.on[0]));
    }

    /// A nested module would otherwise deposit into a directory `OUT_DIR` does not have.
    #[test]
    fn a_nested_entry_point_is_deposited_under_its_leaf_name() {
        let tree = [Module::parse(
            "lib/ramp/paint",
            "@fragment\nfn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.0); }\n",
        )];
        let entries = discover(&tree, &[], true);
        assert_eq!(entries[0].builds()[0].artifact, "paint");
    }

    #[test]
    #[should_panic(expected = "the `resid` axis names `nowhere`")]
    fn an_axis_naming_a_module_that_is_no_entry_point_is_refused() {
        let tree = tree();
        let axis = Axis {
            modules: &["nowhere"],
            ..RESID
        };
        discover(&tree, &[axis], true);
    }

    /// The linker sets one feature per axis, so a second axis on the same feature would
    /// be the only one any build saw.
    #[test]
    #[should_panic(expected = "two axes name the `resid` feature")]
    fn two_axes_on_one_feature_are_refused() {
        let tree = tree();
        let twin = Axis {
            ty: "Other",
            ..RESID
        };
        discover(&tree, &[RESID, twin], true);
    }

    /// The type name is the accessor's parameter, lowercased, so "one word" is what
    /// makes `resid: Resid` readable — and nothing else said so.
    #[test]
    #[should_panic(expected = "which is not one capitalized word")]
    fn a_multi_word_axis_type_is_refused() {
        let tree = tree();
        let axis = Axis {
            ty: "Resid_Lane",
            ..RESID
        };
        discover(&tree, &[axis], true);
    }

    #[test]
    #[should_panic(expected = "two axes generate `Resid`")]
    fn two_axes_generating_one_type_are_refused() {
        let tree = tree();
        let twin = Axis {
            feature: "other",
            ..RESID
        };
        discover(&tree, &[RESID, twin], true);
    }

    /// Sorted is what makes a missing entry visible, so it is checked rather than asked
    /// for in a comment.
    #[test]
    #[should_panic(expected = "the `resid` axis's modules are not sorted")]
    fn an_unsorted_module_list_is_refused() {
        let tree = tree();
        let axis = Axis {
            modules: &["media_mixbox", "stamp", "media_mixbox"],
            ..RESID
        };
        discover(&tree, &[axis], true);
    }
}
