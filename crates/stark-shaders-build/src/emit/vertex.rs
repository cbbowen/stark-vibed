//! The per-instance records a `@vertex` entry point takes.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use wesl::eval::{Type, ty_eval_ty};
use wesl::syntax::{Attribute, ExpressionNode, Function, GlobalDeclaration, Struct, StructMember};

use crate::docs::doc_lines;
use crate::eval::{const_u32, is_gated, module_context};
use crate::layout::{Field, ident, lit, rust_ty};
use crate::tree::Module;

/// Emit the record every `@vertex` entry point of `m` takes, where it takes one.
///
/// **The shader names it.** A vertex input is declared as a named struct the entry
/// point takes whole, so the Rust record's name is the struct's — the one thing a
/// bare parameter list cannot supply, and the last thing the build script had to be
/// told (§6.10). An entry point taking `@location` parameters directly is refused
/// here rather than listed somewhere, and so is a gate on the parameter that names
/// the struct.
pub(super) fn emit(m: &Module) -> TokenStream {
    let mut out = TokenStream::new();
    // One record per struct, however many entry points take it: two `@vertex`
    // functions over one instance buffer is a shape nothing forbids, and the second
    // emission would be a duplicate definition in the generated file.
    let mut done: Vec<String> = Vec::new();
    for f in m.tu.global_declarations.iter().filter_map(|d| match &**d {
        GlobalDeclaration::Function(f)
            if f.attributes
                .iter()
                .any(|a| matches!(**a, Attribute::Vertex)) =>
        {
            Some(f)
        }
        _ => None,
    }) {
        out.extend(record(m, f, &mut done));
    }
    out
}

/// The record `f` takes, if it takes one this module has not already emitted.
///
/// A vertex-stage parameter carries `@builtin` or `@location`, directly or through the
/// members of a struct — so a parameter with no attribute at all *is* the record, and
/// there is nothing else it could be.
fn record(m: &Module, f: &Function, done: &mut Vec<String>) -> TokenStream {
    let (module, entry) = (m.path.as_str(), f.ident.name());
    let mut out = TokenStream::new();
    for p in &f.parameters {
        let member = p.ident.name();
        let at = || format!("`{module}.wesl`'s `{entry}.{member}`");
        // `@builtin(vertex_index)` and friends come from the pipeline, not the buffer.
        if p.attributes
            .iter()
            .any(|a| matches!(**a, Attribute::Builtin(_)))
        {
            continue;
        }
        assert!(
            !p.attributes
                .iter()
                .any(|a| matches!(**a, Attribute::Location(_))),
            "{} is a `@location` parameter. A parameter list has no name, so nothing can \
             name the Rust record the host fills from it. Declare the attributes as the \
             members of a named struct and take one of those (§6.10).",
            at(),
        );
        // Whatever is left is the record, and an attribute on it cannot be passed over
        // the way a `@builtin` is: an `@if` here decides whether the host has the struct
        // at all, out of a source with no feature set to evaluate.
        assert!(
            p.attributes.is_empty(),
            "{} is the record `{entry}` takes, and it carries an attribute. Take it \
             unattributed — an `@if` above all (§6.10).",
            at(),
        );
        let name = p.ty.ident.name();
        let name = name.as_str();
        let s = m.struct_named(name).unwrap_or_else(|| {
            panic!(
                "{} is a `{name}`, which this module declares no `struct` for. A record is \
                 mirrored from the unlinked source, where an import is only a name — \
                 declare the struct here (§6.10).",
                at(),
            )
        });
        if done.iter().any(|d| d == name) {
            continue;
        }
        done.push(name.to_string());
        out.extend(from_struct(m, &entry, s));
    }
    out
}

/// Emit `s` as a Rust struct plus the `wgpu::VertexAttribute` array that reads it.
///
/// **Three transcriptions collapse into one here, not two.** A vertex input was
/// written out as the shader's declaration, as a host `#[repr(C)]` struct, and
/// *again* as a `vertex_attr_array![0 => Float32x2, 1 => Float32]` — where the
/// formats restate the types and the offsets are implied by the order. Nothing tied
/// the three together, and the third is the one with no redundancy to catch it: swap
/// two same-sized attributes and every instance silently reads its neighbour's lane.
///
/// **The layout rule is not the one the uniforms use.** A vertex attribute's offset
/// is the host's to choose — WGSL's alignment tables do not reach a vertex buffer at
/// all — and what `vertex_attr_array!` chooses, and therefore what the shaders were
/// built against, is *tight packing*. So members follow one another with no padding,
/// which for these types is also exactly what `#[repr(C)]` does (every one is
/// 4-byte-aligned and a multiple of 4 in size). The emitted `offset_of` assertions
/// are what say those two rules still agree.
fn from_struct(m: &Module, entry: &str, s: &Struct) -> TokenStream {
    let (src, module) = (m.src.as_str(), m.path.as_str());
    let name = s.ident.name();
    let name = name.as_str();
    // A struct of `@builtin`s alone describes no buffer, and is the same shape as an
    // entry point taking its builtins loose — answered the same way, with no record.
    if !s.members.iter().any(|p| location_of(p).is_some()) {
        return TokenStream::new();
    }
    let mut ctx = module_context(&m.tu);
    let (mut fields, mut attrs, mut offset) = (Vec::new(), Vec::new(), 0u32);
    // Documentation for the first member runs from the opening brace, exactly as
    // `layout::lay_out` reads a uniform struct's: a member's span starts at its
    // attributes, so what precedes it is whatever the author wrote about it.
    let mut prev_end = src[..s.members[0].span().range().start]
        .rfind('{')
        .expect("a struct body opens")
        + 1;

    for p in &s.members {
        let member = p.ident.name();
        let span = p.span().range();
        let docs = doc_lines(&src[prev_end..span.start]);
        prev_end = span.end;

        // `@builtin(vertex_index)` and friends come from the pipeline, not the buffer.
        let Some(location) = location_of(p) else {
            continue;
        };
        let location = const_u32(location, &mut ctx).unwrap_or_else(|why| {
            panic!("`{module}.wesl`'s `{name}.{member}` has a `@location` that {why}")
        });

        // Refused for the same reason a uniform struct's `@if` member is (`layout`).
        assert!(
            !is_gated(&p.attributes),
            "`{module}.wesl`'s `{name}.{member}` is an `@if`-gated `@location` member. \
             The record is mirrored from the unlinked source, which has no feature set to \
             evaluate, so one Rust struct would have to answer for every build of it. \
             Declare the attribute unconditionally (§6.10)."
        );
        let ty = ty_eval_ty(&p.ty, &mut ctx)
            .unwrap_or_else(|e| panic!("`{module}.wesl`'s `{name}.{member}` has no type: {e}"));
        let (format, size) = vertex_format(&ty).unwrap_or_else(|| {
            panic!("`{module}.wesl`'s `{name}.{member}` is a `{ty}`, which is not a vertex format")
        });
        let spelling = rust_ty(&ty, size)
            .unwrap_or_else(|| panic!("`{module}.wesl`'s `{name}.{member}` has no Rust spelling"));

        fields.push(Field {
            docs,
            ident: ident(member.as_str()),
            ty: spelling,
            offset,
            real: true,
        });
        let (loc, at) = (lit(location), lit_u64(offset));
        attrs.push(quote! {
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::#format,
                offset: #at,
                shader_location: #loc,
            }
        });
        offset += size;
    }

    let ident = format_ident!("{name}");
    let members = fields.iter().map(|f| {
        let (docs, id, ty) = (&f.docs, &f.ident, &f.ty);
        quote! {
            #(#[doc = #docs])*
            pub #id: #ty,
        }
    });
    let checks = fields.iter().map(|f| {
        let (id, want) = (&f.ident, lit(f.offset));
        let at = format_ident!("OFFSET_OF_{}", f.ident.to_string().to_uppercase());
        let msg = format!(
            "`{name}.{}` is not at the tightly-packed offset {} its vertex attribute reads",
            f.ident, f.offset,
        );
        quote! {
            const #at: usize = core::mem::offset_of!(#ident, #id);
            assert!(#at == #want, #msg);
        }
    });

    let count = lit(fields.len() as u32);
    let snake = snake_case(name);
    let attrs_ident = format_ident!("{}_ATTRIBUTES", snake.to_uppercase());
    let layout_fn = format_ident!("{snake}_layout");
    let size = lit(offset);
    let doc = format!(
        " `{name}`, generated from `{module}.wesl`'s `struct {name}` — the record\n \
         `@vertex fn {entry}` takes, and the only declaration of it.",
    );
    let attrs_doc = format!(" The vertex attributes reading a [`{name}`], in declaration order.");
    let layout_doc = format!(
        " The buffer layout for a slice of [`{name}`].\n\
         \n\
         `step_mode` is the caller's: nothing in the shader says whether the host means\n \
         to advance this buffer per vertex or per instance. Everything else — the\n \
         stride, the formats, the offsets — comes from the declaration.",
    );
    let msg_size = format!("`{name}` is not the {size} bytes its attributes span");

    quote! {
        #[doc = #doc]
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        pub struct #ident {
            #(#members)*
        }

        impl Default for #ident {
            fn default() -> Self {
                bytemuck::Zeroable::zeroed()
            }
        }

        #[doc = #attrs_doc]
        pub const #attrs_ident: [wgpu::VertexAttribute; #count] = [#(#attrs),*];

        #[doc = #layout_doc]
        pub const fn #layout_fn(
            step_mode: wgpu::VertexStepMode,
        ) -> wgpu::VertexBufferLayout<'static> {
            wgpu::VertexBufferLayout {
                array_stride: core::mem::size_of::<#ident>() as u64,
                step_mode,
                attributes: &#attrs_ident,
            }
        }

        const _: () = {
            // The stride the layout above declares is `size_of`, so a Rust struct
            // larger than its attributes span would read every instance after the
            // first from the wrong place.
            const SIZE: usize = core::mem::size_of::<#ident>();
            assert!(SIZE == #size, #msg_size);
            #(#checks)*
        };
    }
}

/// The `@location` a struct member declares, when it declares one.
fn location_of(m: &StructMember) -> Option<&ExpressionNode> {
    m.attributes.iter().find_map(|a| match &**a {
        Attribute::Location(e) => Some(e),
        _ => None,
    })
}

/// The `wgpu::VertexFormat` for `ty`, and the bytes it occupies.
///
/// Deliberately narrower than [`rust_ty`]: a vertex format is a closed set, and a
/// WGSL type outside it (a matrix, an array, a struct) has to be split across
/// several attributes by hand rather than guessed at.
fn vertex_format(ty: &Type) -> Option<(TokenStream, u32)> {
    let (lanes, scalar) = match ty {
        Type::Vec(n, inner) => (u32::from(*n), &**inner),
        other => (1, other),
    };
    let (stem, width) = match scalar {
        Type::F32 => ("Float32", 4),
        Type::U32 => ("Uint32", 4),
        Type::I32 => ("Sint32", 4),
        _ => return None,
    };
    let ident = format_ident!(
        "{}",
        if lanes == 1 {
            stem.to_string()
        } else {
            format!("{stem}x{lanes}")
        }
    );
    Some((quote!(#ident), lanes * width))
}

/// A `u64` literal — what `wgpu::VertexAttribute::offset` is typed as.
fn lit_u64(n: u32) -> proc_macro2::Literal {
    proc_macro2::Literal::u64_suffixed(n.into())
}

/// `SegmentInstance` as `segment_instance`, for the constant and function named after
/// a generated struct.
fn snake_case(name: &str) -> String {
    let mut out = String::new();
    for (i, c) in name.char_indices() {
        if c.is_uppercase() && i > 0 {
            out.push('_');
        }
        out.extend(c.to_lowercase());
    }
    out
}
