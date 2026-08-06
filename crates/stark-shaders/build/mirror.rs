//! Rust mirrors of the WESL structs the host writes and the shader reads (§7).
//!
//! Every uniform on this boundary is one half of a pair the compiler cannot see
//! across, and the two halves used to be written out separately: nine `vec4` lanes
//! in `dynamics.wesl` and nine `[f32; 4]` fields in `dynamics.rs`, each with its own
//! copy of what the lanes mean. Nothing checked the correspondence, so what the two
//! copies actually did was drift — `Stamp::e`'s Rust doc still described `.zw` as
//! "the midpoint `exchange` samples the canvas at" long after the shader had stopped
//! reading the lane at all.
//!
//! So the shader-side declaration becomes the only one. This walks the WESL AST that
//! `build.rs` already holds and emits the Rust struct from it — fields, padding, and
//! the lane documentation, which now lives exactly once.
//!
//! **The layout is the whole point, and it is not the layout `#[repr(C)]` would give.**
//! WGSL aligns a `vec3<f32>` to 16 bytes and sizes it 12; it rounds a struct up to
//! its own alignment; it pads an array's elements and a matrix's columns out to a
//! stride. A Rust struct of the obvious field types agrees with none of that in
//! general — it only happens to agree when every member is a `vec4`, which is why the
//! hand-written mirrors have survived so far.
//!
//! None of those rules are implemented here. `wesl` resolves a type expression
//! ([`ty_eval_ty`]) and `wgsl-types` gives that type its WGSL [`Type::size_of`] and
//! [`Type::align_of`] — the spec's own tables, including the `@size`/`@align` member
//! attributes and `f16`. What is left for this module is where the *host* has a
//! choice: which Rust spelling occupies a given stride, and the padding that gets the
//! real members onto their offsets.
//!
//! And then the compiler proves it landed: every generated struct carries `size_of`,
//! `align_of` and per-field `offset_of` assertions, so a mistake here is a build
//! failure at the struct it got wrong rather than a lane misread at run time.

use std::path::Path;

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use wesl::eval::{Context, Type, ty_eval_ty};
use wesl::syntax::{GlobalDeclaration, Struct, TranslationUnit};

/// Generate mirrors for `wanted` into `dest`.
///
/// Each entry is `(modules, name)`. The **first** module is where the struct is
/// declared for the host's purposes and what the generated Rust module is named
/// after; any further modules declare the same struct for their own pipeline and are
/// checked to agree with it, member for member and offset for offset.
///
/// That second part is not a nicety. `View` is written out three times in the shader
/// tree — `composite`, `matte` and `overlay` each declare their own — against one
/// `ViewUniform` on the host. Generating from one of them and ignoring the rest would
/// move the drift rather than remove it.
///
/// An explicit list rather than "every struct reachable from a binding", which is the
/// rule this should end at: entries are added as each hand-written mirror is retired,
/// so nothing is generated that nothing uses.
pub fn generate(shader_dir: &Path, dest: &Path, wanted: &[(&[&str], &str)]) {
    // Grouped by the module a struct is emitted under, in first-seen order.
    // `Params` is declared in both `selection.wesl` and `slice.wesl` with *different*
    // members, so the WESL module has to be part of the Rust path.
    let mut modules: Vec<(&str, TokenStream)> = Vec::new();

    for (sources, name) in wanted {
        let (canonical, others) = sources.split_first().expect("a mirror names a module");
        let read = |module: &str| {
            let path = shader_dir.join(format!("{module}.wesl"));
            let src = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
            // The *unlinked* source. The linker mangles `Stamp` to
            // `package__1dynamics_Stamp`, emits it once per artifact that reaches it,
            // strips whatever no entry point uses — and it drops the comments that are
            // half of what is being generated here.
            let tu: TranslationUnit = src
                .parse()
                .unwrap_or_else(|e| panic!("cannot parse {}: {e}", path.display()));
            (src, tu)
        };

        let (src, tu) = read(canonical);
        let laid = lay_out(find(&tu, canonical, name), &src, canonical, &tu);

        for other in others {
            let (o_src, o_tu) = read(other);
            let o_laid = lay_out(find(&o_tu, other, name), &o_src, other, &o_tu);
            agrees(name, canonical, &laid, other, &o_laid);
        }

        let item = emit(name, sources, &laid);
        match modules.iter_mut().find(|(m, _)| m == canonical) {
            Some((_, items)) => items.extend(item),
            None => modules.push((canonical, item)),
        }
    }

    let items = modules.iter().map(|(module, items)| {
        let ident = format_ident!("{module}");
        let doc = format!(" Host mirrors of the structs declared in `{module}.wesl`.");
        quote! {
            #[doc = #doc]
            pub mod #ident {
                #items
            }
        }
    });

    let file = syn::parse2(quote!(#(#items)*)).expect("the generator emits a parseable file");
    let text = format!(
        "// @generated by `build/mirror.rs` from the WESL sources — do not edit.\n\n{}",
        prettyplease::unparse(&file),
    );
    std::fs::write(dest, text).unwrap_or_else(|e| panic!("write {}: {e}", dest.display()));
}

/// The `struct name` declared in `tu`.
fn find<'a>(tu: &'a TranslationUnit, module: &str, name: &str) -> &'a Struct {
    tu.global_declarations
        .iter()
        .find_map(|d| match &**d {
            GlobalDeclaration::Struct(s) if s.ident.name().as_str() == name => Some(s),
            _ => None,
        })
        .unwrap_or_else(|| panic!("`{module}.wesl` declares no `struct {name}`"))
}

/// Fail unless two shader modules lay `name` out identically.
///
/// Only the layout is compared, not the prose: three shaders documenting the same
/// lanes in their own words is fine and is why the comments differ, but a member
/// renamed, retyped or reordered in one of them is a divergence the host cannot see.
fn agrees(name: &str, canonical: &str, a: &Laid, other: &str, b: &Laid) {
    let lanes = |l: &Laid| {
        l.fields
            .iter()
            .filter(|f| f.real)
            .map(|f| format!("{}: {} @{}", f.ident, f.ty, f.offset))
            .collect::<Vec<_>>()
    };
    assert!(
        (a.size, a.align) == (b.size, b.align) && lanes(a) == lanes(b),
        "`{name}` is declared differently in `{canonical}.wesl` and `{other}.wesl`, \
         which share one host mirror:\n  {canonical}: {:?} ({} bytes)\n  {other}: {:?} \
         ({} bytes)",
        lanes(a),
        a.size,
        lanes(b),
        b.size,
    );
}

/// One field of the generated struct: a member, or the padding WGSL puts before one.
struct Field {
    docs: Vec<String>,
    ident: proc_macro2::Ident,
    ty: TokenStream,
    offset: u32,
    /// Padding is emitted but not asserted on — it is the mechanism by which the real
    /// members land where they should, and those are what carry the assertions.
    real: bool,
}

/// A struct placed at its WGSL offsets.
struct Laid {
    fields: Vec<Field>,
    size: u32,
    align: u32,
}

fn emit(name: &str, sources: &[&str], laid: &Laid) -> TokenStream {
    let Laid {
        fields,
        size,
        align,
    } = laid;
    let (size, align) = (*size, *align);

    let ident = format_ident!("{name}");
    let members = fields.iter().map(|f| {
        let (docs, id, ty) = (&f.docs, &f.ident, &f.ty);
        quote! {
            #(#[doc = #docs])*
            pub #id: #ty,
        }
    });

    // What makes the generator trustworthy rather than merely plausible: the rules
    // above computed these offsets, and the compiler now confirms the Rust type
    // really has them. An error in the spelling/padding below stops the build at the
    // struct it got wrong, instead of shifting a lane by four bytes at run time.
    //
    // Each quantity is bound to a `const` before being asserted on, because a macro's
    // arguments are opaque tokens that no formatter can lay out — put the `size_of`
    // inside the `assert!` and the generated file reads `size_of :: < Stamp > ()`.
    let checks = fields.iter().filter(|f| f.real).map(|f| {
        let (id, offset) = (&f.ident, lit(f.offset));
        let at = format_ident!("OFFSET_OF_{}", f.ident.to_string().to_uppercase());
        let msg = format!("`{name}.{}` is not at WGSL offset {}", f.ident, f.offset);
        quote! {
            const #at: usize = core::mem::offset_of!(#ident, #id);
            assert!(#at == #offset, #msg);
        }
    });

    let doc = match sources {
        [one] => format!(
            " `{name}`, generated from `{one}.wesl` — the shader's declaration is the only one.",
        ),
        [first, rest @ ..] => format!(
            " `{name}`, generated from `{first}.wesl`, which {} declare identically.",
            rest.iter()
                .map(|m| format!("`{m}.wesl`"))
                .collect::<Vec<_>>()
                .join(" and "),
        ),
        [] => unreachable!("a mirror names a module"),
    };
    let doc_size = format!(" WGSL size {size}, alignment {align}.");
    let msg_size = format!("`{name}` is not {size} bytes");
    let msg_align = format!("`{name}` is not {align}-byte aligned");
    let (size, align) = (lit(size), lit(align));

    quote! {
        #[doc = #doc]
        #[doc = ""]
        #[doc = #doc_size]
        #[repr(C, align(#align))]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        pub struct #ident {
            #(#members)*
        }

        // So a caller can write `..Default::default()` and leave the padding alone.
        // Padding has to be a real field to keep the struct free of the implicit
        // kind, which would make it unsound to read as bytes — but a caller filling
        // in lanes should no more have to name it than the shader does.
        impl Default for #ident {
            fn default() -> Self {
                bytemuck::Zeroable::zeroed()
            }
        }

        const _: () = {
            const SIZE: usize = core::mem::size_of::<#ident>();
            const ALIGN: usize = core::mem::align_of::<#ident>();
            assert!(SIZE == #size, #msg_size);
            assert!(ALIGN == #align, #msg_align);
            #(#checks)*
        };
    }
}

/// An unsuffixed integer literal — `4`, not `4usize`, which is what an array length
/// and a `#[repr(align(_))]` want and what reads as a number in the generated file.
fn lit(n: u32) -> proc_macro2::Literal {
    proc_macro2::Literal::usize_unsuffixed(n as usize)
}

/// Place `s`'s members at their WGSL offsets.
fn lay_out(s: &Struct, src: &str, module: &str, tu: &TranslationUnit) -> Laid {
    let name = s.ident.name();
    assert!(!s.members.is_empty(), "`{module}::{name}` has no members");
    let mut ctx = Context::new(tu);

    let (mut fields, mut offset, mut align) = (Vec::new(), 0u32, 1u32);
    // Documentation for the first member runs from the opening brace.
    let mut prev_end = src[..s.members[0].span().range().start]
        .rfind('{')
        .expect("a struct body opens")
        + 1;

    for m in &s.members {
        let member = m.ident.name();
        let fail = |what: &str| -> ! {
            panic!(
                "`{module}::{name}.{member}` is a `{}`, which {what}. Every member \
                 after it would be placed at the wrong offset, so this is a build \
                 failure rather than a skipped field.",
                m.ty,
            )
        };
        let ty =
            ty_eval_ty(&m.ty, &mut ctx).unwrap_or_else(|e| fail(&format!("did not resolve: {e}")));
        let (Some(m_size), Some(m_align)) = (ty.size_of(), ty.align_of()) else {
            fail("is not host-shareable")
        };
        let Some(spelling) = rust_ty(&ty, m_size) else {
            fail("has no Rust spelling")
        };

        // WGSL places a member at the next offset meeting its alignment.
        let at = round_up(m_align, offset);
        if at != offset {
            fields.push(pad(fields.len(), offset, at - offset, m_align));
        }

        let span = m.span().range();
        fields.push(Field {
            docs: doc_lines(&src[prev_end..span.start]),
            ident: ident(member.as_str()),
            ty: spelling,
            offset: at,
            real: true,
        });

        offset = at + m_size;
        align = align.max(m_align);
        prev_end = span.end;
    }

    // WGSL rounds a struct up to its own alignment. Spelling that as a trailing member
    // rather than leaving it to `#[repr(align)]` keeps the struct free of *implicit*
    // padding, which is what `Pod` requires.
    let size = round_up(align, offset);
    if size != offset {
        fields.push(pad(fields.len(), offset, size - offset, align));
    }
    Laid {
        fields,
        size,
        align,
    }
}

fn pad(index: usize, offset: u32, width: u32, align: u32) -> Field {
    let n = lit(width);
    Field {
        docs: vec![format!(
            " Padding to the {align}-byte WGSL alignment that follows."
        )],
        ident: format_ident!("_pad_{index}"),
        ty: quote!([u8; #n]),
        offset,
        real: false,
    }
}

/// The Rust spelling of `ty` occupying exactly `stride` bytes.
///
/// `stride` is what makes this more than a name lookup. An array of `vec3<f32>` puts
/// its elements 16 bytes apart though each holds 12, and a `mat3x3<f32>` does the same
/// with its columns; the Rust type has to *say* that, because unlike WGSL it has no
/// separate notion of stride. So a padded vector widens to the lanes it actually
/// occupies — `vec3<f32>` at a stride of 16 is `[f32; 4]` — which is the only place
/// this module chooses a representation rather than reading one off the spec.
fn rust_ty(ty: &Type, stride: u32) -> Option<TokenStream> {
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

/// A WGSL identifier as a Rust one, raw-escaped where the two languages disagree
/// about what is a keyword (`type`, `ref`, `become`, …).
fn ident(name: &str) -> proc_macro2::Ident {
    match syn::parse_str::<syn::Ident>(name) {
        Ok(id) => id,
        Err(_) => proc_macro2::Ident::new_raw(name, proc_macro2::Span::call_site()),
    }
}

/// The `//` comment run immediately preceding a member, as doc-comment text.
///
/// Walked backwards from the member, because what makes a comment *this member's*
/// documentation is that nothing stands between the two. The final line of `between`
/// is the member's own indentation and is skipped; anything else that is not a `//`
/// line ends the run.
fn doc_lines(between: &str) -> Vec<String> {
    let mut lines: Vec<&str> = between.lines().collect();
    if lines.last().is_some_and(|l| l.trim().is_empty()) {
        lines.pop();
    }
    let mut docs: Vec<String> = lines
        .iter()
        .rev()
        .map_while(|l| {
            l.trim()
                .strip_prefix("//")
                .map(|r| r.trim_end().to_string())
        })
        .collect();
    docs.reverse();
    docs
}
