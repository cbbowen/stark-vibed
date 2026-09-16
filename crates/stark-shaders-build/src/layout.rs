//! WGSL layout, and the Rust spelling that occupies it.
//!
//! **The layout is the whole point, and it is not the layout `#[repr(C)]` would
//! give.** WGSL aligns a `vec3<f32>` to 16 bytes and sizes it 12; it rounds a struct up
//! to its own alignment; it pads an array's elements and a matrix's columns out to a
//! stride. A Rust struct of the obvious field types agrees with none of that in
//! general — it agrees only when every member is a `vec4`, which is a coincidence to be
//! generated past, not relied on.
//!
//! None of those *type* rules are implemented here. `wesl` resolves a type expression
//! ([`ty_eval_ty`]) and `wgsl-types` gives that type its WGSL [`Type::size_of`] and
//! [`Type::align_of`] — the spec's own tables, `f16` and nested structs included. The
//! member's own `@size`/`@align` overrides are read with `wgsl-types`' own accessors
//! ([`EvalAttrs`]), which is the reading [`Type::size_of`] uses when it sizes a whole
//! struct — and so the one `min_binding_size` is computed from, two paths that
//! `emit::bindings` now asserts agree. What is left is where the *host* has a choice:
//! which Rust spelling occupies a given stride, and the padding that gets the real
//! members onto their offsets.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use wesl::eval::{EvalAttrs, Type, ty_eval_ty};
use wesl::syntax::Struct;

use crate::docs::doc_lines;
use crate::eval::{is_gated, module_context};
use crate::tree::Module;

/// One field of the generated struct: a member, or the padding WGSL puts before one.
pub(crate) struct Field {
    pub(crate) docs: Vec<String>,
    pub(crate) ident: proc_macro2::Ident,
    pub(crate) ty: TokenStream,
    pub(crate) offset: u32,
    /// Padding is emitted but not asserted on — it is the mechanism by which the real
    /// members land where they should, and those are what carry the assertions.
    pub(crate) real: bool,
}

/// A struct placed at its WGSL offsets.
pub(crate) struct Laid {
    pub(crate) fields: Vec<Field>,
    pub(crate) size: u32,
    pub(crate) align: u32,
}

/// Place `s`'s members at their WGSL offsets.
///
/// Fallible rather than panicking, because discovery calls it on every uniform in the
/// tree (`emit::structs`) and one it cannot spell has to be skipped with a note, not
/// stop the build. A member it *can* reach but cannot place is still fatal to the
/// struct as a whole — every member after it would land at the wrong offset — which is
/// what the error says.
pub(crate) fn lay_out(s: &Struct, module: &Module) -> Result<Laid, String> {
    let (src, tu, path) = (module.src.as_str(), &module.tu, module.path.as_str());
    let name = s.ident.name();
    if s.members.is_empty() {
        return Err(format!("`{path}::{name}` has no members"));
    }
    let mut ctx = module_context(tu);

    let (mut fields, mut offset, mut align) = (Vec::new(), 0u32, 1u32);
    // Documentation for the first member runs from the opening brace.
    let mut prev_end = src[..s.members[0].span().range().start]
        .rfind('{')
        .expect("a struct body opens")
        + 1;

    for m in &s.members {
        let member = m.ident.name();
        let at_member = || format!("`{path}::{name}.{member}`");
        let fail = |what: &str| -> String {
            format!(
                "`{path}::{name}.{member}` is a `{}`, which {what}, so `{name}` is not \
                 mirrored",
                m.ty,
            )
        };

        // **`@if` is refused, never evaluated.** A mirror is generated from the
        // *unlinked* source, which carries no feature set — so one Rust struct would
        // have to answer for both the plain and the residual artifact, and could match
        // at most one. `matte.wesl` states the rule in prose beside the attribute it
        // cost; this is it made structural.
        assert!(
            !is_gated(&m.attributes),
            "{} is `@if`-gated. A uniform struct is mirrored from the unlinked source, \
             which has no feature set to evaluate, so one Rust struct would have to \
             answer for every build of it. Declare the member unconditionally (§6.10).",
            at_member(),
        );

        let ty = match ty_eval_ty(&m.ty, &mut ctx) {
            Ok(ty) => ty,
            Err(e) => return Err(fail(&format!("did not resolve: {e}"))),
        };
        let (Some(ty_size), Some(ty_align)) = (ty.size_of(), ty.align_of()) else {
            return Err(fail("is not host-shareable"));
        };
        let Some(spelling) = rust_ty(&ty, ty_size) else {
            return Err(fail("has no Rust spelling"));
        };

        // What the *member* says about its own stride, over what its type says. Read
        // with `wgsl-types`' accessors, so this and `Type::size_of` — which is where
        // `min_binding_size` comes from — cannot read one attribute two ways.
        let attr = |what: &str, got: Result<Option<u32>, wesl::eval::EvalError>| {
            got.unwrap_or_else(|e| {
                panic!(
                    "{} has an `@{what}` that does not evaluate: {e}",
                    at_member()
                )
            })
        };
        let m_align = attr("align", m.attr_align(&mut ctx)).unwrap_or(ty_align);
        assert!(
            m_align.is_power_of_two(),
            "{} declares `@align({m_align})`, which is not a power of two",
            at_member(),
        );
        let m_size = attr("size", m.attr_size(&mut ctx)).unwrap_or(ty_size);
        assert!(
            m_size >= ty_size,
            "{} declares `@size({m_size})` for a {ty_size}-byte `{}`. `@size` pads a \
             member out; it cannot truncate one.",
            at_member(),
            m.ty,
        );

        // WGSL places a member at the next offset meeting its alignment.
        let at = round_up(m_align, offset);
        if at != offset {
            fields.push(pad(fields.len(), offset, at - offset, to_align(m_align)));
        }

        let span = m.span().range();
        fields.push(Field {
            docs: doc_lines(&src[prev_end..span.start]),
            ident: ident(member.as_str()),
            ty: spelling,
            offset: at,
            real: true,
        });

        // An `@size` past the type's own is the member's trailing padding. The lane the
        // shader reads is still the type's, so the Rust spelling stays the type's too —
        // widening it would tell the host there are lanes the shader never wrote.
        if m_size > ty_size {
            fields.push(pad(
                fields.len(),
                at + ty_size,
                m_size - ty_size,
                format!(" Padding to the `@size({m_size})` the member declares."),
            ));
        }

        offset = at + m_size;
        align = align.max(m_align);
        prev_end = span.end;
    }

    // WGSL rounds a struct up to its own alignment. Spelling that as a trailing member
    // rather than leaving it to `#[repr(align)]` keeps the struct free of *implicit*
    // padding, which is what `Pod` requires.
    let size = round_up(align, offset);
    if size != offset {
        fields.push(pad(fields.len(), offset, size - offset, to_align(align)));
    }
    Ok(Laid {
        fields,
        size,
        align,
    })
}

fn pad(index: usize, offset: u32, width: u32, why: String) -> Field {
    let n = lit(width);
    Field {
        docs: vec![why],
        ident: format_ident!("_pad_{index}"),
        ty: quote!([u8; #n]),
        offset,
        real: false,
    }
}

fn to_align(align: u32) -> String {
    format!(" Padding to the {align}-byte WGSL alignment that follows.")
}

/// The Rust spelling of `ty` occupying exactly `stride` bytes.
///
/// `stride` is what makes this more than a name lookup. An array of `vec3<f32>` puts
/// its elements 16 bytes apart though each holds 12, and a `mat3x3<f32>` does the same
/// with its columns; the Rust type has to *say* that, because unlike WGSL it has no
/// separate notion of stride. So a padded vector widens to the lanes it actually
/// occupies — `vec3<f32>` at a stride of 16 is `[f32; 4]` — which is the only place
/// this module chooses a representation rather than reading one off the spec.
pub(crate) fn rust_ty(ty: &Type, stride: u32) -> Option<TokenStream> {
    let scalar = |t: &Type| match t {
        Type::F32 => Some(quote!(f32)),
        Type::U32 => Some(quote!(u32)),
        Type::I32 => Some(quote!(i32)),
        _ => None,
    };
    match ty {
        Type::F32 | Type::U32 | Type::I32 => (stride == 4).then(|| scalar(ty)).flatten(),
        Type::Vec(_, inner) => {
            let elem = scalar(inner)?;
            let width = inner.size_of()?;
            // Exactly divides for every vector WGSL can pad: the stride is a multiple
            // of the component size in all of them.
            let lanes = lit(stride.is_multiple_of(width).then(|| stride / width)?);
            Some(quote!([#elem; #lanes]))
        }
        Type::Mat(cols, rows, inner) => {
            let column = Type::Vec(*rows, inner.clone());
            let col_stride = column.align_of()?;
            let n = lit(*cols as u32);
            (col_stride * *cols as u32 == stride)
                .then(|| rust_ty(&column, col_stride))
                .flatten()
                .map(|c| quote!([#c; #n]))
        }
        Type::Array(inner, Some(count)) => {
            let elem_stride = round_up(inner.align_of()?, inner.size_of()?);
            let n = lit(*count as u32);
            (elem_stride * *count as u32 == stride)
                .then(|| rust_ty(inner, elem_stride))
                .flatten()
                .map(|e| quote!([#e; #n]))
        }
        // A nested struct would need its own generated mirror, which is a reasonable
        // thing to add and not something to guess at.
        _ => None,
    }
}

fn round_up(align: u32, n: u32) -> u32 {
    n.div_ceil(align) * align
}

/// An unsuffixed integer literal — `4`, not `4usize`, which is what an array length
/// and a `#[repr(align(_))]` want and what reads as a number in the generated file.
pub(crate) fn lit(n: u32) -> proc_macro2::Literal {
    proc_macro2::Literal::usize_unsuffixed(n as usize)
}

/// A WGSL identifier as a Rust one, raw-escaped where the two languages disagree
/// about what is a keyword (`type`, `ref`, `become`, …).
pub(crate) fn ident(name: &str) -> proc_macro2::Ident {
    match syn::parse_str::<syn::Ident>(name) {
        Ok(id) => id,
        Err(_) => proc_macro2::Ident::new_raw(name, proc_macro2::Span::call_site()),
    }
}
