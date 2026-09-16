//! The per-instance records a `@vertex` entry point's `@location` parameters describe.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use wesl::eval::{Type, ty_eval_ty};
use wesl::syntax::{Attribute, GlobalDeclaration};

use crate::docs::doc_lines;
use crate::eval::{const_u32, is_gated, module_context};
use crate::layout::{Field, ident, lit, rust_ty};
use crate::tree::Module;

/// Every `@vertex` entry point in `m` that takes at least one `@location` parameter —
/// i.e. that reads a per-instance record the host has to lay out.
pub(super) fn instanced_entries(m: &Module) -> Vec<String> {
    m.tu.global_declarations
        .iter()
        .filter_map(|d| match &**d {
            GlobalDeclaration::Function(f)
                if f.attributes
                    .iter()
                    .any(|a| matches!(**a, Attribute::Vertex))
                    && f.parameters.iter().any(|p| {
                        p.attributes
                            .iter()
                            .any(|a| matches!(**a, Attribute::Location(_)))
                    }) =>
            {
                Some(f.ident.name().to_string())
            }
            _ => None,
        })
        .collect()
}

/// Emit the per-instance record `entry`'s `@location` parameters describe, as a Rust
/// struct plus the `wgpu::VertexAttribute` array that reads it.
///
/// **Three transcriptions collapse into one here, not two.** A vertex input was
/// written out as the shader's parameter list, as a host `#[repr(C)]` struct, and
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
pub(super) fn emit(m: &Module, entry: &str, name: &str) -> TokenStream {
    let (tu, src, module) = (&m.tu, m.src.as_str(), m.path.as_str());
    let (func, body) = tu
        .global_declarations
        .iter()
        .find_map(|d| match &**d {
            GlobalDeclaration::Function(f)
                if f.ident.name().as_str() == entry
                    && f.attributes
                        .iter()
                        .any(|a| matches!(**a, Attribute::Vertex)) =>
            {
                Some((f, d.span().range()))
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("`{module}.wesl` has no `@vertex fn {entry}`"));

    // A `FormalParameter` is not a spanned node, so a parameter's own documentation is
    // found by locating its `@location` within the function's span. That is what lets
    // the prose describing a lane live beside the lane, in the shader, the way the
    // uniforms' does — rather than in a host struct the shader cannot see.
    let doc_for = |location: u32| {
        let needle = format!("@location({location})");
        let at = src[body.clone()].find(&needle)? + body.start;
        let line = src[..at].rfind('\n').map_or(0, |i| i + 1);
        Some(doc_lines(&src[..line]))
    };

    let mut ctx = module_context(tu);
    let (mut fields, mut attrs, mut offset) = (Vec::new(), Vec::new(), 0u32);

    for p in &func.parameters {
        // `@builtin(vertex_index)` and friends come from the pipeline, not the buffer.
        let Some(location) = p.attributes.iter().find_map(|a| match &**a {
            Attribute::Location(e) => Some(e),
            _ => None,
        }) else {
            continue;
        };
        let location = const_u32(location, &mut ctx).unwrap_or_else(|why| {
            panic!("`{module}.wesl`'s `{entry}` has a `@location` that {why}")
        });

        let member = p.ident.name();
        // Refused for the same reason a uniform struct's `@if` member is (`layout`).
        assert!(
            !is_gated(&p.attributes),
            "`{module}.wesl`'s `{entry}.{member}` is an `@if`-gated `@location` \
             parameter. The record is mirrored from the unlinked source, which has no \
             feature set to evaluate, so one Rust struct would have to answer for every \
             build of it. Declare the attribute unconditionally (§6.10)."
        );
        let ty = ty_eval_ty(&p.ty, &mut ctx)
            .unwrap_or_else(|e| panic!("`{module}.wesl`'s `{entry}.{member}` has no type: {e}"));
        let (format, size) = vertex_format(&ty).unwrap_or_else(|| {
            panic!("`{module}.wesl`'s `{entry}.{member}` is a `{ty}`, which is not a vertex format")
        });
        let spelling = rust_ty(&ty, size)
            .unwrap_or_else(|| panic!("`{module}.wesl`'s `{entry}.{member}` has no Rust spelling"));

        fields.push(Field {
            docs: doc_for(location).unwrap_or_default(),
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
    assert!(
        !fields.is_empty(),
        "`{module}.wesl`'s `{entry}` takes no `@location` parameters, so there is no \
         per-instance record to generate"
    );

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
        " `{name}`, generated from `{module}.wesl`'s `@vertex fn {entry}` — the shader's\n \
         parameter list is the only declaration.",
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
