//! The `@binding` declarations, three ways.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use wesl::eval::Context;
use wesl::syntax::{
    AddressSpace, Attribute, Declaration, DeclarationKind, Expression, ExpressionNode,
    GlobalDeclaration, TypeExpression,
};

use crate::docs::doc_lines;
use crate::eval::{group_binding, module_context};
use crate::layout::{lay_out, lit};
use crate::tree::Module;

use super::uniform_type;

/// Emit the `@binding` declarations of `m` three ways: `binding::NAME` (the index, for
/// a `match` arm), `decl::NAME` (the whole declaration, for a slot list), and
/// `BINDINGS` (all of them, for a structural check).
///
/// The third transcription of the same boundary, and the one with the least
/// redundancy to catch it: a binding number was written in the WESL declaration, in
/// the layout entry, and in the bind-group entry, with margin comments as the only
/// map between them. The struct mirrors made the *lanes* single-sourced; this does
/// the same for the *slots*, so a renumbering in the shader is a one-file change
/// that the host follows by name.
///
/// **`decl::` is what makes the group unambiguous.** A slot list naming
/// `decl::REGION_COLOR` carries the declaration itself, so nothing looks a binding up
/// by index — which it could not do correctly anyway, since `@binding(0)` means a
/// different slot in each of a module's groups and half the tree declares more than
/// one (`stamp_common` has three).
///
/// Every declaration is emitted, `@if`-gated ones included — the unlinked source
/// keeps them, and a host that binds one does so exactly when the matching feature
/// build declares it. The name is the WESL variable's, uppercased; two declarations
/// that collide there are a build failure rather than a silent shadowing.
pub(super) fn emit(m: &Module) -> TokenStream {
    let (tu, src, module) = (&m.tu, m.src.as_str(), m.path.as_str());
    let mut ctx = module_context(tu);
    let mut names: Vec<String> = Vec::new();
    let mut indices = Vec::new();
    let mut decls = Vec::new();
    let mut table = Vec::new();
    for d in &tu.global_declarations {
        let GlobalDeclaration::Declaration(decl) = &**d else {
            continue;
        };
        let member = decl.ident.name();
        // The group as well as the index. A bind group layout is for exactly one group,
        // so this is what lets the host assert that a slot list names one — and what a
        // table keyed on the index alone could never have said.
        let Some((group, index)) =
            group_binding(decl, &mut ctx, &format!("`{module}.wesl`'s `{member}`"))
        else {
            continue;
        };
        // `lib/` holds the binding-free leaves (§2) — a rule that had been prose alone,
        // though this is the loop that sees every binding in the tree.
        assert!(
            !m.under_lib(),
            "`{module}.wesl` declares `@group({group}) @binding({index}) var {member}`, \
             but a module under `lib/` may not declare a binding (§2): those are the \
             leaves a pipeline's modules import, and a binding in one lands in every \
             artifact that reaches it, at a slot no importer chose. Move it to the \
             module that owns the pipeline."
        );
        let name = member.to_uppercase();
        assert!(
            !names.contains(&name),
            "`{module}.wesl` declares two bindings that both mirror as `{name}`"
        );
        names.push(name.clone());
        let docs = doc_lines(&src[..d.span().range().start]);
        let ident = format_ident!("{name}");
        let (group_lit, index_lit) = (lit(group), lit(index));
        indices.push(quote! {
            #(#[doc = #docs])*
            pub const #ident: u32 = #index_lit;
        });

        // The rest of what the declaration decides: what kind of thing occupies the
        // slot, and whether it exists at all in a build without the residual.
        let kind = bind_kind(decl, m, &member, &mut ctx);
        // `@if(resid)` — the shader's own gate on the slot, carried through so a
        // layout never has to restate it as an element count (`[..12 + 4 *
        // usize::from(resid)]`).
        let resid = decl.attributes.iter().any(|a| match &**a {
            Attribute::If(e) => src[e.span().range()].trim() == "resid",
            _ => false,
        });
        // `super::`, because `binding` is `decl`'s sibling inside the shader's module,
        // not its child — the bare path resolved from nowhere and every one of these
        // (132 of them, one per declared binding) was a broken intra-doc link. Nothing
        // in CI ran `cargo doc`, and a generated doc is exactly the kind nobody reads
        // in the source, so the whole set stayed broken silently.
        let decl_doc = format!(
            " `@group({group}) @binding({index}) var {member}` — see [`super::binding::{name}`]."
        );
        decls.push(quote! {
            #[doc = #decl_doc]
            pub const #ident: Binding = Binding {
                group: #group_lit,
                index: #index_lit,
                name: #name,
                kind: #kind,
                resid: #resid,
            };
        });
        table.push(quote!(decl::#ident));
    }
    if indices.is_empty() {
        // Every module under `lib/` and every leaf that owns no pipeline. Not an
        // error: discovery reaches the whole tree, and a binding-free module is the
        // rule there rather than an omission (§2).
        return TokenStream::new();
    }
    let index_doc = format!(
        " The `@binding` indices `{module}.wesl` declares, named for their WESL\n \
         variables — the shader's declarations are the only ones.\n\n \
         The index alone does **not** identify a slot when a module declares more than\n \
         one group; it is what a bind-group entry is keyed on once the group is fixed.\n \
         To *name* a slot, use [`decl`].",
    );
    let decl_doc = format!(
        " Every `@binding` `{module}.wesl` declares, whole: its group and index, what\n \
         kind of thing occupies it, and whether it is `@if(resid)`-gated.\n\n \
         The host builds both its bind-group **layouts** and its bind **groups** from\n \
         these, so the two cannot disagree about a slot's type, its storage format, or\n \
         whether the residual build has it.",
    );
    let table_doc = " Every declaration in [`decl`], in declaration order — for the checks that ask\n \
         about the set rather than about one slot.";
    quote! {
        // The descriptor types are hand-written in `lib.rs` — they are the host's
        // vocabulary, not the shader's — and this generated module sits two levels
        // below the crate root, so it names them absolutely.
        use crate::{BindKind, Binding};

        #[doc = #index_doc]
        pub mod binding {
            #(#indices)*
        }

        #[doc = #decl_doc]
        pub mod decl {
            use super::{BindKind, Binding};
            #(#decls)*
        }

        #[doc = #table_doc]
        pub const BINDINGS: &[Binding] = &[#(#table),*];
    }
}

/// What kind of thing a `@binding` declaration puts in its slot, as a `BindKind`
/// expression.
///
/// Read off the declared type, which is where the answer already is: a
/// `texture_storage_2d<rgba32float, write>` is a storage texture of that format, and
/// the host had been choosing between `stor` and `stor32` by hand at every layout that
/// named one.
///
/// **The `wgpu` types are emitted, not their WGSL spellings.** `stark-shaders` depends
/// on `wgpu` anyway — the generated vertex layouts need it — so a `BindKind` carrying
/// `&'static str` bought nothing but a pair of string matches on the host, each with a
/// runtime panic for a fact known here. Now an unmapped format stops *this* build,
/// naming the declaration.
fn bind_kind(decl: &Declaration, m: &Module, member: &str, ctx: &mut Context<'_>) -> TokenStream {
    let module = m.path.as_str();
    let ty = decl
        .ty
        .as_ref()
        .unwrap_or_else(|| panic!("`{module}.wesl`'s `{member}` has a `@binding` but no type"));
    let name = ty.ident.name();
    let name = name.as_str();
    // `var<uniform> x: T` — the size is `T`'s, by the same WGSL layout rules the
    // struct mirrors are laid out under, so `min_binding_size` cannot drift from the
    // struct it guards.
    if matches!(
        &decl.kind,
        DeclarationKind::Var(Some((AddressSpace::Uniform, _)))
    ) {
        let size = uniform_size(ty, m, member, ctx);
        let size = proc_macro2::Literal::u64_unsuffixed(size);
        return quote!(BindKind::Uniform { min_size: #size });
    }
    if name == "sampler" {
        return quote!(BindKind::Sampler);
    }
    let at = || format!("`{module}.wesl`'s `{member}`");
    if let Some(dim) = name.strip_prefix("texture_storage_") {
        // `<format, access>`; only the format reaches the host's descriptor, the
        // access mode being implied by the layout entry the host builds.
        let args = ty
            .template_args
            .as_ref()
            .unwrap_or_else(|| panic!("{} has no storage format", at()));
        let format = expr_ident(&args[0].expression)
            .unwrap_or_else(|| panic!("{} has a storage format that is not a name", at()));
        let format = texture_format(&format, &at());
        let dim = view_dimension(dim, &at());
        return quote!(BindKind::Storage { dim: #dim, format: #format });
    }
    if let Some(dim) = name.strip_prefix("texture_") {
        let dim = view_dimension(dim, &at());
        return quote!(BindKind::Texture { dim: #dim });
    }
    panic!("{} has type `{name}`, which is not a binding kind", at());
}

/// A WGSL storage-format name as a `wgpu::TextureFormat` path.
///
/// Only the formats the shader tree declares. A new one is a deliberate addition here
/// — the two spellings are close but not mechanically derivable (`rg11b10ufloat` is
/// `Rg11b10Ufloat`), and guessing is how a host ends up binding a format the shader
/// does not write.
fn texture_format(wgsl: &str, at: &str) -> TokenStream {
    let ident = match wgsl {
        "rgba8unorm" => "Rgba8Unorm",
        "rgba8snorm" => "Rgba8Snorm",
        "r8unorm" => "R8Unorm",
        "r16float" => "R16Float",
        "rg16float" => "Rg16Float",
        "rgba16float" => "Rgba16Float",
        "r32float" => "R32Float",
        "rg32float" => "Rg32Float",
        "rgba32float" => "Rgba32Float",
        other => panic!("{at} declares storage format `{other}`, which has no `wgpu` mapping here"),
    };
    let ident = format_ident!("{ident}");
    quote!(wgpu::TextureFormat::#ident)
}

/// A WGSL texture type's dimension suffix as a `wgpu::TextureViewDimension` path.
fn view_dimension(wgsl: &str, at: &str) -> TokenStream {
    let ident = match wgsl {
        "1d" => "D1",
        "2d" => "D2",
        "2d_array" => "D2Array",
        "3d" => "D3",
        "cube" => "Cube",
        "cube_array" => "CubeArray",
        other => panic!("{at} is a `texture_{other}`, which has no `wgpu` mapping here"),
    };
    let ident = format_ident!("{ident}");
    quote!(wgpu::TextureViewDimension::#ident)
}

/// The identifier a template argument names, e.g. `rgba16float`.
fn expr_ident(expr: &ExpressionNode) -> Option<String> {
    match &**expr {
        Expression::TypeOrIdentifier(t) => Some(t.ident.name().to_string()),
        _ => None,
    }
}

/// The WGSL size of a uniform binding's declared type — its `min_binding_size`.
///
/// **Two paths reach one number, and this is where they are made to agree.** The size is
/// `wgsl-types`' own, over the resolved type; the generated struct's `size_of` assertion
/// comes from [`lay_out`] walking the members itself. Nothing compared them, so a member
/// attribute one honoured and the other did not left a host struct and the
/// `min_binding_size` guarding it quietly out of step — a buffer the right size for a
/// struct of the wrong one.
fn uniform_size(ty: &TypeExpression, m: &Module, member: &str, ctx: &mut Context<'_>) -> u64 {
    let resolved = uniform_type(ty, m, member, ctx);
    let at = || format!("`{}.wesl`'s `{member}`", m.path);
    let size = resolved
        .size_of()
        .unwrap_or_else(|| panic!("{} has an unsized uniform type", at()));

    // Only a struct this module declares and `lay_out` can spell: anything else has no
    // second answer to compare against.
    if let Some(laid) = m
        .struct_named(ty.ident.name().as_str())
        .and_then(|s| lay_out(s, m).ok())
    {
        let align = resolved
            .align_of()
            .expect("a type with a size has an alignment");
        assert_eq!(
            (laid.size, laid.align),
            (size, align),
            "{} is laid out at {} bytes / {}-byte alignment by `layout::lay_out`, where \
             `wgsl-types` sizes the same type at {size} / {align}. The two read the \
             declaration differently, and `min_binding_size` is the one the host binds \
             against.",
            at(),
            laid.size,
            laid.align,
        );
    }
    size as u64
}
