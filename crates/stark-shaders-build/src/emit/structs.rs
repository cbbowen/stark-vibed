//! The uniform structs the host fills in.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use wesl::syntax::{AddressSpace, DeclarationKind, GlobalDeclaration};

use crate::layout::{Laid, lay_out, lit};
use crate::tree::Module;

/// Emit a mirror for every struct a `var<uniform>` in `m` names — the boundary the
/// host writes across, discovered rather than listed (§2).
///
/// `aliased` holds the `(module, struct)` pairs a `shared` entry already generated
/// under another module, so the two do not both emit one. The second return is what
/// discovery reached and could not spell.
pub(super) fn discover(m: &Module, aliased: &[(String, String)]) -> (TokenStream, Vec<String>) {
    let mut out = TokenStream::new();
    let mut skipped = Vec::new();
    let mut done: Vec<String> = Vec::new();
    for d in &m.tu.global_declarations {
        let GlobalDeclaration::Declaration(decl) = &**d else {
            continue;
        };
        if !matches!(
            &decl.kind,
            DeclarationKind::Var(Some((AddressSpace::Uniform, _)))
        ) {
            continue;
        }
        let Some(ty) = decl.ty.as_ref() else { continue };
        let name = ty.ident.name();
        let name = name.as_str();
        // The same struct can be named by two uniforms of one module; and a `shared`
        // entry has already generated this one somewhere else.
        if done.iter().any(|n| n == name)
            || aliased
                .iter()
                .any(|(md, n)| md == &m.path && n.as_str() == name)
        {
            continue;
        }
        // Not a struct at all — a `var<uniform> x: vec4<f32>` is legal WGSL and needs
        // no mirror, since the host already has the type.
        let Some(s) = m.struct_named(name) else {
            continue;
        };
        done.push(name.to_string());
        match lay_out(s, m) {
            Ok(laid) => out.extend(emit(name, &[m.path.as_str()], &laid)),
            // The reason discovery must not panic: it reaches every uniform in the
            // tree, and one that a host has never asked for being unmirrorable is not
            // a reason to stop. A caller that needed it fails at its own use site.
            Err(why) => skipped.push(why),
        }
    }
    (out, skipped)
}

/// Fail unless two shader modules lay `name` out identically.
///
/// Only the layout is compared, not the prose: three shaders documenting the same
/// lanes in their own words is fine and is why the comments differ, but a member
/// renamed, retyped or reordered in one of them is a divergence the host cannot see.
pub(super) fn agrees(name: &str, canonical: &str, a: &Laid, other: &str, b: &Laid) {
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

/// Emit `name` as a `Pod` Rust struct at its WGSL offsets, documented from the WESL
/// comments and asserted to have landed.
///
/// `sources` is the module it was generated from, then any that declare it
/// identically.
pub(super) fn emit(name: &str, sources: &[&str], laid: &Laid) -> TokenStream {
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

    // What makes the generator trustworthy rather than merely plausible: the layout
    // rules computed these offsets, and the compiler now confirms the Rust type really
    // has them. An error in the spelling/padding stops the build at the struct it got
    // wrong, instead of shifting a lane by four bytes at run time.
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
