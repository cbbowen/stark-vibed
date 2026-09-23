//! The `@location` each member of a fragment-output struct writes.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use wesl::eval::Context;
use wesl::syntax::{Attribute, GlobalDeclaration, Struct};

use crate::docs::doc_lines;
use crate::eval::{const_u32, module_context};
use crate::layout::{lit, snake_case};
use crate::tree::Module;

use super::location_of;

/// The structs some `@fragment` entry point in the tree returns.
///
/// **Tree-wide, and by name.** A fragment output is routinely imported — `stamp.wesl`'s
/// `fs_main` returns a `FsOut` that `stamp_common.wesl` declares — and the mirror belongs
/// under the module that *declares* the struct, since that is the only file whose
/// `@location`s it states. Resolving the import would be the second resolver §6.10 is
/// about, so what crosses the module boundary here is the set of names and nothing else:
/// [`emit`] looks each one up among its own declarations.
///
/// A `-> @location(0) vec4<f32>` return contributes a predeclared type's name, which no
/// module declares a struct for; it is looked up and missed, not special-cased.
pub(super) fn returned(modules: &[Module]) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for m in modules {
        for f in m.tu.global_declarations.iter().filter_map(|d| match &**d {
            GlobalDeclaration::Function(f)
                if f.attributes
                    .iter()
                    .any(|a| matches!(**a, Attribute::Fragment)) =>
            {
                Some(f)
            }
            _ => None,
        }) {
            let Some(ty) = f.return_type.as_ref() else {
                continue;
            };
            let name = ty.ident.name();
            if !names.iter().any(|n| n.as_str() == name.as_str()) {
                names.push(name.to_string());
            }
        }
    }
    names
}

/// Emit the `@location`s of every struct `m` declares that [`returned`] found.
pub(super) fn emit(m: &Module, returned: &[String]) -> TokenStream {
    let mut ctx = module_context(&m.tu);
    let mut out = TokenStream::new();
    for s in m.tu.global_declarations.iter().filter_map(|d| match &**d {
        GlobalDeclaration::Struct(s) => {
            let name = s.ident.name();
            returned
                .iter()
                .any(|n| n.as_str() == name.as_str())
                .then_some(s)
        }
        _ => None,
    }) {
        out.extend(locations(m, s, &mut ctx));
    }
    out
}

/// `s`'s members as a module of `@location` constants, documented from the WESL comment
/// abutting each one.
///
/// **A fragment `@location` was a host constant twice over** — the swept path spelled
/// `COLOR = 0, AUX = 1, RESID = 2, CEILING = 3` against `stamp_common.wesl`'s `FsOut` and
/// the erase spelled `MASS, CEILING, MOMENT` against `FsErase` — and a renumbering in the
/// shader moved neither. Nothing would have failed: the attachments still bind, in the
/// host's order, and the pass writes its lanes into each other's targets.
///
/// `usize`, because a fragment location reaches the host only as a position in a color
/// target or attachment list (`desc::targets_for`). A `@if`-gated member is emitted like
/// any other: the location is fixed whether or not a given build declares the member,
/// which is the whole reason the swept path's ceiling sits at 3 with or without a
/// residual at 2.
fn locations(m: &Module, s: &Struct, ctx: &mut Context<'_>) -> TokenStream {
    let (src, module) = (m.src.as_str(), m.path.as_str());
    let name = s.ident.name();
    let name = name.as_str();
    if s.members.is_empty() {
        return TokenStream::new();
    }
    let mut consts = Vec::new();
    let mut names: Vec<String> = Vec::new();
    // Documentation for the first member runs from the opening brace, exactly as
    // `layout::lay_out` reads a uniform struct's.
    let mut prev_end = src[..s.members[0].span().range().start]
        .rfind('{')
        .expect("a struct body opens")
        + 1;

    for p in &s.members {
        let member = p.ident.name();
        let span = p.span().range();
        let docs = doc_lines(&src[prev_end..span.start]);
        prev_end = span.end;

        // A `@builtin` output — `frag_depth`, a sample mask — is not an attachment.
        let Some(location) = location_of(p) else {
            continue;
        };
        let location = const_u32(location, ctx).unwrap_or_else(|why| {
            panic!("`{module}.wesl`'s `{name}.{member}` has a `@location` that {why}")
        });

        let upper = member.to_uppercase();
        assert!(
            !names.contains(&upper),
            "`{module}.wesl`'s `{name}` declares two members that both mirror as `{upper}`"
        );
        names.push(upper.clone());
        let (ident, at) = (format_ident!("{upper}"), lit(location));
        consts.push(quote! {
            #(#[doc = #docs])*
            pub const #ident: usize = #at;
        });
    }

    if consts.is_empty() {
        // A struct of `@builtin`s alone, which describes no attachment.
        return TokenStream::new();
    }
    let ident = format_ident!("{}", snake_case(name));
    let doc = format!(
        " The color attachment each member of `{name}` writes, from `{module}.wesl`'s own\n \
         `@location`s — a position in a target or attachment list, which is the only shape\n \
         a fragment location reaches the host in.\n\n \
         `@if`-gated members are here too: a location is fixed whether or not a build\n \
         declares the member.",
    );
    quote! {
        #[doc = #doc]
        pub mod #ident {
            #(#consts)*
        }
    }
}
