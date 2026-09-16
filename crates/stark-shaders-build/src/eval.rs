//! One const evaluator, for everything that reads a number off a declaration.
//!
//! There were three. Two const-evaluated an attribute and refused what they could not
//! discharge; [`crate::collide`] parsed the *rendered* expression as a literal and
//! **skipped** the declaration when that failed — so a `@binding` the tree does not
//! happen to spell as a digit would have quietly left the collision check covering less
//! than it claims. One function answers now, and its answer is the number or a refusal.

use wesl::eval::{Context, Eval, Exec, Instance, LiteralInstance};
use wesl::syntax::{Attribute, Declaration, ExpressionNode, TranslationUnit};

/// A [`Context`] in which `tu`'s own module-scope names can be evaluated.
///
/// [`Context::new`] opens a *function* scope, where the evaluator looks a name up in the
/// scope and never in the module's declarations — so `const X: f32 = TAU / 4.0;` was
/// dropped with a header note and `@binding(SLOT + 1)` was refused, both for naming a
/// neighbour. Executing the translation unit first is what fills that scope: the
/// `const`s get their values, and `var`s and `override`s land deferred, which is what
/// they are before there is a pipeline.
///
/// Best-effort. A declaration that will not execute ends the walk, and whatever wanted
/// it is refused at its own site with the evaluator's reason.
pub(crate) fn module_context(tu: &TranslationUnit) -> Context<'_> {
    let mut ctx = Context::new(tu);
    let _ = tu.exec(&mut ctx);
    ctx
}

/// The `u32` an attribute's expression evaluates to.
///
/// The error reads as a predicate — the caller puts its own declaration in front of it.
/// There is deliberately no "pass over it" answer: an attribute this cannot reach is one
/// the generated mirror would be silently wrong about.
pub(crate) fn const_u32(e: &ExpressionNode, ctx: &mut Context<'_>) -> Result<u32, String> {
    let value = e
        .eval_value(ctx)
        .map_err(|why| format!("does not evaluate: {why}"))?;
    let n = match &value {
        Instance::Literal(LiteralInstance::AbstractInt(n)) => u32::try_from(*n).ok(),
        Instance::Literal(LiteralInstance::U32(n)) => Some(*n),
        Instance::Literal(LiteralInstance::I32(n)) => u32::try_from(*n).ok(),
        _ => None,
    };
    n.ok_or_else(|| format!("is `{value}`, which is no `u32`"))
}

/// The `@group` and `@binding` a declaration states, when it states a binding.
///
/// `None` only when there is no `@binding` at all. A `@binding` without a `@group` is
/// refused, and so is either expression the evaluator cannot discharge: the pair *is*
/// the slot, and half of one is nothing a host layout or a collision check can use.
///
/// `at` names the declaration — `` `stamp.wesl`'s `st` `` — and the message continues it.
pub(crate) fn group_binding(
    decl: &Declaration,
    ctx: &mut Context<'_>,
    at: &str,
) -> Option<(u32, u32)> {
    let expr = |pick: fn(&Attribute) -> Option<&ExpressionNode>| {
        decl.attributes.iter().find_map(|a| pick(a))
    };
    let binding = expr(|a| match a {
        Attribute::Binding(e) => Some(e),
        _ => None,
    })?;
    let group = expr(|a| match a {
        Attribute::Group(e) => Some(e),
        _ => None,
    })
    .unwrap_or_else(|| panic!("{at} has a `@binding` but no `@group`"));
    Some((
        read(group, "group", ctx, at),
        read(binding, "binding", ctx, at),
    ))
}

fn read(e: &ExpressionNode, what: &str, ctx: &mut Context<'_>, at: &str) -> u32 {
    const_u32(e, ctx).unwrap_or_else(|why| panic!("{at} has a `@{what}` that {why}"))
}

/// Whether an `@if` gates this declaration.
///
/// The one place that names `Attribute::If`, which WESL's conditional translation puts
/// behind a cargo feature. What a mirror does about one is its caller's business: a
/// `@binding` carries the flag through, a struct member and a vertex parameter are
/// refused (§6.10).
pub(crate) fn is_gated(attributes: &[wesl::syntax::AttributeNode]) -> bool {
    attributes.iter().any(|a| matches!(**a, Attribute::If(_)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use wesl::syntax::GlobalDeclaration;

    /// The declaration named `name`, in a module parsed from `src`.
    fn slot_of(src: &str, name: &str) -> Option<(u32, u32)> {
        let tu: TranslationUnit = src.parse().expect("the probe parses");
        let decl = tu
            .global_declarations
            .iter()
            .find_map(|d| match &**d {
                GlobalDeclaration::Declaration(d) if d.ident.name().as_str() == name => Some(d),
                _ => None,
            })
            .expect("the probe declares it")
            .clone();
        group_binding(&decl, &mut module_context(&tu), "`probe.wesl`'s `x`")
    }

    /// The point of evaluating rather than reading: the shader gets to name its own
    /// partition of a group's index space, and the generator still knows the number.
    #[test]
    fn a_binding_given_by_a_const_expression_evaluates() {
        assert_eq!(
            slot_of(
                "const BASE: u32 = 4u;\n\
                 @group(0) @binding(BASE + 1u) var x: texture_2d<f32>;\n",
                "x",
            ),
            Some((0, 5))
        );
    }

    #[test]
    fn a_plain_literal_evaluates() {
        assert_eq!(
            slot_of("@group(1) @binding(2) var x: texture_2d<f32>;\n", "x"),
            Some((1, 2))
        );
    }

    #[test]
    fn a_declaration_with_no_binding_has_no_slot() {
        assert_eq!(slot_of("var<private> x: f32;\n", "x"), None);
    }

    /// Refused, not skipped — the whole reason there is one of these.
    #[test]
    #[should_panic(expected = "has a `@binding` that does not evaluate")]
    fn a_binding_naming_nothing_is_refused() {
        slot_of("@group(0) @binding(NOWHERE) var x: texture_2d<f32>;\n", "x");
    }

    #[test]
    #[should_panic(expected = "is `-1`, which is no `u32`")]
    fn a_negative_binding_is_refused() {
        slot_of("@group(0) @binding(-1) var x: texture_2d<f32>;\n", "x");
    }

    #[test]
    #[should_panic(expected = "has a `@binding` but no `@group`")]
    fn a_binding_without_a_group_is_refused() {
        slot_of("@binding(0) var x: texture_2d<f32>;\n", "x");
    }
}
