//! The constants both sides compute with.

use proc_macro2::{Literal, TokenStream};
use quote::{format_ident, quote};
use wesl::eval::{Convert, Eval, Instance, LiteralInstance, Ty, ty_eval_ty};
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
/// A constant naming another evaluates too, in [`module_context`]'s scope. No `const` in
/// the tree names another today.
///
/// Vectors and arrays come through as Rust arrays ([`spell`]), which is what lets a
/// shader state a colour or a table of coefficients once. Only a shape with no Rust
/// spelling at all — a matrix, a struct — keeps its note.
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
        let (rust, literal) = match spell(&value) {
            Ok(pair) => pair,
            Err(why) => {
                skipped.push(format!("{} {why}", at()));
                continue;
            }
        };

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

/// The Rust type and the literal occupying it, or why the value has neither.
///
/// The error reads as a predicate — the caller puts the declaration in front of it.
///
/// **A value's own shape, not a buffer's.** A `vec3<f32>` is `[f32; 3]` here, where
/// [`crate::layout::rust_ty`] widens it to the `[f32; 4]` a uniform member's 16-byte
/// stride occupies: nobody reads a constant out of a buffer, so there is no stride to
/// meet. Composites recurse, which is what gives `array<vec4<f32>, N>` a spelling
/// without a second case for it.
fn spell(value: &Instance) -> Result<(TokenStream, TokenStream), String> {
    // Parsed rather than built: a negative is a `-` and a literal, which is two tokens
    // in an expression position and no `proc_macro2::Literal` at all.
    let scalar =
        |ty: TokenStream, text: String| Ok((ty, text.parse().expect("a scalar literal parses")));
    match value {
        // `{:?}` on an `f32` prints the shortest decimal that reads back to the same
        // bits, so the generated literal *is* this value — which is the whole difficulty
        // the check this replaces documented, having compared the host's rounded
        // `0.06f32` against the source's exact decimal as `f64`.
        Instance::Literal(LiteralInstance::F32(v)) if v.is_finite() => {
            scalar(quote!(f32), format!("{v:?}"))
        }
        Instance::Literal(LiteralInstance::F32(v)) => {
            Err(format!("is {v}, which no Rust literal spells"))
        }
        Instance::Literal(LiteralInstance::U32(v)) => scalar(quote!(u32), format!("{v}")),
        Instance::Literal(LiteralInstance::I32(v)) => scalar(quote!(i32), format!("{v}")),
        Instance::Literal(LiteralInstance::Bool(v)) => scalar(quote!(bool), format!("{v}")),
        Instance::Vec(v) => compose(v.iter()),
        Instance::Array(a) => compose(a.iter()),
        other => Err(format!("is a `{}`, which has no host constant", other.ty())),
    }
}

/// [`spell`] for a vector's components or an array's elements — a Rust array of
/// whatever the elements spell as.
///
/// Every element has the same type (both instances enforce it on construction), so the
/// first one's spelling is the array's.
fn compose<'a>(
    elements: impl Iterator<Item = &'a Instance>,
) -> Result<(TokenStream, TokenStream), String> {
    let spelled = elements
        .map(spell)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|why| format!("has an element that {why}"))?;
    let (elem, _) = spelled.first().ok_or("has no elements")?;
    let n = Literal::usize_unsuffixed(spelled.len());
    let values = spelled.iter().map(|(_, v)| v);
    Ok((quote!([#elem; #n]), quote!([#(#values),*])))
}
