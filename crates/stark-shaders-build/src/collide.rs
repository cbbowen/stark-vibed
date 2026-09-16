//! Where one module's share of a group's index space stops.

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
pub(crate) fn bindings_do_not_collide(linked: &wesl::syntax::TranslationUnit, artifact: &str) {
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
