//! One const evaluator, for everything that reads a number off a declaration.
//!
//! There were three, and [`crate::collide`]'s parsed the *rendered* attribute text and
//! **skipped** the declaration when that was not a literal — quietly narrowing the check
//! it belongs to. The answer here is the number or a refusal; there is no "pass over it".

use wesl::eval::{Context, Eval, Exec, Instance, LiteralInstance};
use wesl::syntax::{
    Attribute, AttributeNode, Declaration, ExpressionNode, GlobalDeclaration, TranslationUnit,
};

/// A [`Context`] in which `tu`'s own `const`s can be named.
///
/// [`Context::new`] opens a *function* scope, where the evaluator looks a name up in the
/// scope and never in the module's declarations — so `const X: f32 = TAU / 4.0;` and
/// `@binding(SLOT + 1)` failed merely for naming a neighbour. Executing the declarations
/// fills that scope.
///
/// **One at a time, repeatedly, and only the `const`s.** `TranslationUnit::exec` does it
/// in one call but stops at the first declaration that will not execute — and
/// `dynamics.wesl` has one: `var<workgroup> ws_lm: array<Latent, BAKE_RES>`, whose
/// element type is *imported* and so unresolvable in the unlinked source this reads. That
/// walk ended there, leaving the nine `const`s below it unable to name a neighbour, in
/// the largest module of the tree, silently. Repeatedly, because WGSL hoists at module
/// scope: a constant may name one declared after it.
///
/// A `const` that never executes is left out, and whatever wanted it is refused at its
/// own site with the evaluator's reason.
pub(crate) fn module_context(tu: &TranslationUnit) -> Context<'_> {
    let mut ctx = Context::new(tu);
    let consts: Vec<&GlobalDeclaration> = tu
        .global_declarations
        .iter()
        .map(|d| &**d)
        .filter(|d| matches!(d, GlobalDeclaration::Declaration(d) if d.kind.is_const()))
        .collect();
    let mut done = vec![false; consts.len()];
    loop {
        let mut added = false;
        for (d, done) in consts.iter().zip(&mut done) {
            if !*done && d.exec(&mut ctx).is_ok() {
                (*done, added) = (true, true);
            }
        }
        if !added {
            return ctx;
        }
    }
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
/// `at` names the declaration — `` `stamp.wesl`'s `st` `` — and a message continues it.
/// It is called only to build one, which is why it is a closure: most declarations have
/// no `@binding` at all.
///
/// # Panics
/// On a `@binding` with no `@group`, and on either expression the evaluator cannot
/// discharge. [`const_u32`] returns those as an error; here they are fatal, because a
/// slot the generator cannot name is one the host would bind by guess.
pub(crate) fn group_binding(
    decl: &Declaration,
    ctx: &mut Context<'_>,
    at: impl Fn() -> String,
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
    .unwrap_or_else(|| panic!("{} has a `@binding` but no `@group`", at()));
    Some((
        read(group, "group", ctx, &at),
        read(binding, "binding", ctx, &at),
    ))
}

fn read(e: &ExpressionNode, what: &str, ctx: &mut Context<'_>, at: &impl Fn() -> String) -> u32 {
    const_u32(e, ctx).unwrap_or_else(|why| panic!("{} has a `@{what}` that {why}", at()))
}

/// Whether an `@if` gates this declaration.
///
/// Refused on a struct member, where a mirror would otherwise have to be the layout of
/// two artifacts (§6.10) — of a uniform, or of a vertex record.
pub(crate) fn is_gated(attributes: &[AttributeNode]) -> bool {
    attributes.iter().any(|a| matches!(**a, Attribute::If(_)))
}

/// Whether `@if(feature)` gates this declaration, read from the module's `src` because
/// the condition is an expression and what a mirror carries through is the name the
/// shader wrote.
///
/// With [`is_gated`], the only two places that name `Attribute::If`.
pub(crate) fn gated_on(attributes: &[AttributeNode], src: &str, feature: &str) -> bool {
    attributes.iter().any(|a| match &**a {
        Attribute::If(e) => src[e.span().range()].trim() == feature,
        _ => false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
        group_binding(&decl, &mut module_context(&tu), || {
            "`probe.wesl`'s `x`".to_string()
        })
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

    /// And the walk that fills the scope may not stop at a declaration it cannot
    /// execute: `dynamics.wesl` has one above nine constants (see [`module_context`]).
    #[test]
    fn a_declaration_that_will_not_execute_does_not_end_the_walk() {
        assert_eq!(
            slot_of(
                "var<workgroup> ws: array<Latent, 4>;\n\
                 const BASE: u32 = 4u;\n\
                 @group(0) @binding(BASE + 1u) var x: texture_2d<f32>;\n",
                "x",
            ),
            Some((0, 5))
        );
    }

    /// WGSL hoists at module scope, so one pass over the declarations is not enough.
    #[test]
    fn a_const_naming_a_later_const_evaluates() {
        assert_eq!(
            slot_of(
                "const SLOT: u32 = BASE + 1u;\n\
                 const BASE: u32 = 4u;\n\
                 @group(0) @binding(SLOT) var x: texture_2d<f32>;\n",
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
