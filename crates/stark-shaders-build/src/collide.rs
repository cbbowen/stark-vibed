//! Where one module's share of a group's index space stops.

use wesl::SourceMap;
use wesl::syntax::{GlobalDeclaration, TranslationUnit};

use crate::eval::{group_binding, module_context};

/// Fail unless the modules a pipeline links agree about where each one's share of a
/// group's index space stops.
///
/// **The arrangement this guards is now "a group each".** `blend_common`,
/// `filter_common` and `media_common` own `@group(0)` entire and the pigment module
/// beside each owns `@group(1)` entire, so neither has to know where the other's slots
/// stop. That leaves one family sharing a group: the wet loop's, where `dynamics` and
/// `liquify` interleave ~40 slots of `@group(0)` with the five `dynamics_common`
/// declares, and the two cannot be parted without an exact second bind group per kernel
/// on the hottest path in the app. So the rule the generator can assert is still the
/// weaker one — **two modules may not claim one slot** — rather than "one module per
/// group".
///
/// Taken on the *linked* artifact, which holds exactly the declarations one pipeline
/// compiles, post-`@if`, imports resolved. So a collision is arithmetic here, in terms
/// of the two **files**, rather than a `naga` error naming
/// `package__1filter_mixbox_pigment_lut` at pipeline creation.
///
/// **Only collisions between two different modules are a fault**, and that is the
/// distinction the hazard is actually about rather than a tolerance. A module may
/// deliberately declare two things at one slot when no entry point reaches both:
/// `transform.wesl` puts `Quad` and `Gated` at `@group(0) @binding(0)` because the
/// affine and the rect-scoped maps are different pipelines, and its header carries the
/// rule that keeps it sound. One file can state that about itself; two files splitting a
/// group cannot, which is why one is checked and the other is not.
///
/// `sourcemap` is the compiler's own record of which module a mangled name came from,
/// so this does not have to agree with whichever mangler the compiler was configured
/// with. A name it does not hold is the root's, which is not mangled.
pub(crate) fn bindings_do_not_collide(
    linked: &TranslationUnit,
    sourcemap: &impl SourceMap,
    artifact: &str,
) {
    let mut ctx = module_context(linked);
    let source = |name: &str| {
        sourcemap
            .get_decl(name)
            .map_or_else(|| artifact.to_string(), |(path, _)| path.to_string())
    };
    let mut seen: Vec<(u32, u32, String, String)> = Vec::new();
    for d in &linked.global_declarations {
        let GlobalDeclaration::Declaration(decl) = &**d else {
            continue;
        };
        let name = decl.ident.name();
        let Some((g, b)) = group_binding(decl, &mut ctx, || {
            format!("`{}`'s `{name}`", source(name.as_str()))
        }) else {
            continue;
        };
        let (name, from) = (name.to_string(), source(name.as_str()));
        if let Some((_, _, other, other_from)) = seen
            .iter()
            .find(|(sg, sb, _, sf)| *sg == g && *sb == b && *sf != from)
        {
            panic!(
                "`{artifact}` links `{other_from}`'s `{other}` and `{from}`'s `{name}` \
                 at the same `@group({g}) @binding({b})`. The modules a pipeline links \
                 partition a group's index space between them, and two of them have \
                 claimed one slot."
            );
        }
        seen.push((g, b, name, from));
    }
}

/// A linked artifact is two or more modules' declarations in one translation unit, with
/// every import mangled to the module it came from — so these probes are written the way
/// the linker leaves them, and the sourcemap says which name is whose.
#[cfg(test)]
mod tests {
    use super::*;
    use wesl::{BasicSourceMap, NoSourceMap};

    /// The linker's own mangling, as `dynamics_common`'s `st` reaches an artifact.
    fn sourcemap(decls: &[(&str, &str, &str)]) -> BasicSourceMap {
        let mut map = BasicSourceMap::new();
        for (mangled, module, item) in decls {
            map.add_decl(
                (*mangled).to_string(),
                module.parse().expect("a module path"),
                (*item).to_string(),
            );
        }
        map
    }

    fn check(src: &str, map: &BasicSourceMap) {
        let tu: TranslationUnit = src.parse().expect("the probe parses");
        bindings_do_not_collide(&tu, map, "dynamics");
    }

    /// The message names the two *files*, which is the whole point of checking here.
    #[test]
    #[should_panic(
        expected = "`package::dynamics_common`'s `package__1dynamics_common_st` and \
                    `dynamics`'s `region_color` at the same `@group(0) @binding(0)`"
    )]
    fn two_modules_at_one_slot_are_named_by_the_sourcemap() {
        check(
            "@group(0) @binding(0) var<uniform> package__1dynamics_common_st: Stamp;\n\
             @group(0) @binding(0) var region_color: texture_2d<f32>;\n\
             struct Stamp { a: vec4<f32> }\n",
            &sourcemap(&[(
                "package__1dynamics_common_st",
                "package::dynamics_common",
                "st",
            )]),
        );
    }

    /// The arrangement the colour-space passes now have: a group each, so neither
    /// module's indices say anything about the other's.
    #[test]
    fn a_group_per_module_is_the_shape_that_passes() {
        check(
            "@group(0) @binding(0) var package__1blend_common_src: texture_2d<f32>;\n\
             @group(1) @binding(0) var pigment_lut: texture_2d<f32>;\n\
             @group(1) @binding(1) var back_resid: texture_2d<f32>;\n",
            &sourcemap(&[("package__1blend_common_src", "package::blend_common", "src")]),
        );
    }

    /// One file may put two declarations at one slot when no entry point reaches both
    /// (`transform.wesl`), and the root module's names are not mangled at all — so the
    /// sourcemap holding neither is the answer "both are the root's".
    #[test]
    fn one_module_at_one_slot_twice_is_allowed() {
        check(
            "@group(0) @binding(0) var<uniform> quad: Quad;\n\
             @group(0) @binding(0) var<uniform> gated: Gated;\n\
             struct Quad { a: vec4<f32> }\n\
             struct Gated { b: vec4<f32> }\n",
            &sourcemap(&[]),
        );
    }

    /// The same index in a different group is a different slot, and half the tree
    /// declares more than one group.
    #[test]
    fn the_same_index_in_two_groups_is_two_slots() {
        check(
            "@group(0) @binding(1) var package__1a_x: texture_2d<f32>;\n\
             @group(1) @binding(1) var package__1b_y: texture_2d<f32>;\n",
            &sourcemap(&[
                ("package__1a_x", "package::a", "x"),
                ("package__1b_y", "package::b", "y"),
            ]),
        );
    }

    /// What used to be skipped in silence: the rendered expression is not a literal, so
    /// the declaration dropped out of the check. It is evaluated now, and the collision
    /// it was hiding is found.
    #[test]
    #[should_panic(expected = "at the same `@group(0) @binding(5)`")]
    fn a_const_valued_binding_still_collides() {
        check(
            "const BASE: u32 = 4u;\n\
             @group(0) @binding(5) var package__1blend_common_src: texture_2d<f32>;\n\
             @group(0) @binding(BASE + 1u) var pigment_lut: texture_2d<f32>;\n",
            &sourcemap(&[("package__1blend_common_src", "package::blend_common", "src")]),
        );
    }

    /// And one the evaluator cannot reach is refused rather than passed over.
    #[test]
    #[should_panic(expected = "has a `@binding` that does not evaluate")]
    fn a_binding_the_evaluator_cannot_reach_is_refused() {
        let tu: TranslationUnit = "@group(0) @binding(NOWHERE) var x: texture_2d<f32>;\n"
            .parse()
            .expect("the probe parses");
        bindings_do_not_collide(&tu, &NoSourceMap, "dynamics");
    }
}
