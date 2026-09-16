//! The constants both sides compute with.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use wesl::eval::{Convert, Eval, Instance, LiteralInstance, Type, ty_eval_ty};
use wesl::syntax::GlobalDeclaration;

use crate::docs::doc_lines;
use crate::eval::module_context;
use crate::tree::Module;

/// Emit every `const` of `m` that has a Rust spelling, with what was skipped.
///
/// **Evaluated, not read.** The value is whatever `wesl`'s const evaluator makes of
/// the initializer, so `7.0 / 64.0` comes out as the number the shader will actually
/// compute with. The check this replaces parsed a decimal literal out of the *linked*
/// source and could do neither: a derived constant is not a literal, and the linker had
/// already stripped anything no entry point reached.
///
/// **A constant naming another evaluates too**, in [`module_context`]'s scope — see
/// `emit::tests::a_const_derived_from_its_neighbours_mirrors_as_its_value`. No `const`
/// in the tree names another today.
pub(super) fn emit(m: &Module) -> (TokenStream, Vec<String>) {
    let (tu, src, module) = (&m.tu, m.src.as_str(), m.path.as_str());
    let mut ctx = module_context(tu);
    let mut out = TokenStream::new();
    let mut skipped = Vec::new();
    for d in &tu.global_declarations {
        let GlobalDeclaration::Declaration(decl) = &**d else {
            continue;
        };
        if !decl.kind.is_const() {
            continue;
        }
        let name = decl.ident.name();
        let name = name.as_str();

        // An explicit type is required rather than inferred from the value. WGSL's
        // *abstract* numerics have no Rust counterpart to pick — `const N = 4` could
        // honestly become an `i32`, a `u32` or an `f32` — and guessing is how a host
        // constant ends up a different type from the one the shader computes with.
        //
        // Passed over in silence rather than noted: an untyped const is the shader
        // deferring the choice, not a mirror that failed. `dynamics.wesl`'s
        // `BLEED_OFFS` is one, and no host wants it.
        let Some(declared) = decl.ty.as_ref() else {
            continue;
        };
        let Some(init) = decl.initializer.as_ref() else {
            continue;
        };
        let at = || format!("`{module}.wesl`'s `const {name}`");
        let Ok(value) = init.eval_value(&mut ctx) else {
            skipped.push(format!("{} does not const-evaluate", at()));
            continue;
        };
        let Ok(ty) = ty_eval_ty(declared, &mut ctx) else {
            skipped.push(format!("{} has no resolvable type", at()));
            continue;
        };
        // `2` evaluates to an *abstract* int, and `4.0` to an abstract float — WGSL
        // defers the choice of a concrete type to the declaration. Converting to the
        // declared type is what the shader itself does, and doing it here is why the
        // generated constant is the type the shader computes with rather than whichever
        // one the literal happened to look like.
        let Some(value) = value.convert_to(&ty) else {
            skipped.push(format!("{} is a `{value}`, which is not a `{ty}`", at()));
            continue;
        };
        let Instance::Literal(lit) = &value else {
            // A *typed* array, matrix or struct: a real declaration that a host constant
            // cannot be. The tree has none — its stencil tables are untyped and exit
            // above — so this branch is reached only by the test that pins it.
            skipped.push(format!("{} is not a scalar", at()));
            continue;
        };
        let (rust, literal) = match (&ty, lit) {
            (Type::F32, LiteralInstance::F32(v)) => {
                assert!(
                    v.is_finite(),
                    "{} is {v}, which no Rust literal spells",
                    at()
                );
                // `{:?}` on an `f32` prints the shortest decimal that reads back to the
                // same bits, so the generated literal *is* this value — which is the whole
                // difficulty the check this replaces documented, having compared the
                // host's rounded `0.06f32` against the source's exact decimal as `f64`.
                (quote!(f32), format!("{v:?}"))
            }
            (Type::U32, LiteralInstance::U32(v)) => (quote!(u32), format!("{v}")),
            (Type::I32, LiteralInstance::I32(v)) => (quote!(i32), format!("{v}")),
            (Type::Bool, LiteralInstance::Bool(v)) => (quote!(bool), format!("{v}")),
            _ => {
                skipped.push(format!("{} is a `{ty}`, which has no host constant", at()));
                continue;
            }
        };
        let literal: TokenStream = literal.parse().expect("a scalar literal is one token");

        let ident = format_ident!("{name}");
        let mut docs = doc_lines(&src[..d.span().range().start]);
        docs.push(String::new());
        docs.push(format!(
            " Generated from `{module}.wesl`'s `const {name}` — the shader's declaration is",
        ));
        docs.push(" the only one.".to_string());
        out.extend(quote! {
            #(#[doc = #docs])*
            pub const #ident: #rust = #literal;
        });
    }
    (out, skipped)
}
