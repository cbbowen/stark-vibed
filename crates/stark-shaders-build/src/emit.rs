//! Rust mirrors of what the host writes and the shader reads (§6.10, §7).
//!
//! Four kinds, each generated from the WESL declaration that decides how it is read:
//! the **uniform structs** ([`structs`]), the **constants** both sides compute with
//! ([`consts`]), the **`@binding` declarations** ([`bindings`]), and the
//! **per-instance vertex records** a vertex entry point's `@location` parameters
//! describe ([`vertex`]).
//!
//! Every one of them is half of a pair the compiler cannot see across. A hand-written
//! Rust half is a second declaration with its own copy of what the lanes mean, and
//! nothing checks the correspondence: the two drift, and a lane the shader has stopped
//! reading goes on being documented as though it were live.
//!
//! So the shader-side declaration is the only one. And then the compiler proves the
//! mirror landed: every generated struct carries `size_of`, `align_of` and per-field
//! `offset_of` assertions, so a mistake in [`crate::layout`] is a build failure at the
//! struct it got wrong rather than a lane misread at run time.

mod bindings;
mod consts;
mod structs;
mod vertex;

use std::path::Path;

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use wesl::eval::{Context, Eval, Instance, LiteralInstance};
use wesl::syntax::{ExpressionNode, Struct};

use crate::layout::lay_out;
use crate::tree::{Module, read_tree};

/// The `u32` an attribute's expression evaluates to.
///
/// `None` for anything that is not a plain number: an attribute the const evaluator
/// cannot discharge is the caller's to complain about, naming its own declaration.
fn const_u32(e: &ExpressionNode, ctx: &mut Context<'_>) -> Option<u32> {
    e.eval_value(ctx).ok().and_then(|i| match i {
        Instance::Literal(LiteralInstance::AbstractInt(n)) => u32::try_from(n).ok(),
        Instance::Literal(LiteralInstance::U32(n)) => Some(n),
        Instance::Literal(LiteralInstance::I32(n)) => u32::try_from(n).ok(),
        _ => None,
    })
}

/// Generate the host mirrors of everything the shader tree at `shader_dir` declares,
/// into `dest`.
pub(crate) fn generate(
    shader_dir: &Path,
    dest: &Path,
    shared: &[(&[&str], &str)],
    vertex: &[(&str, &str, &str)],
) {
    let text = mirrors(&read_tree(shader_dir), shared, vertex);
    std::fs::write(dest, text).unwrap_or_else(|e| panic!("write {}: {e}", dest.display()));
}

/// The generated file's text, from an already-parsed tree.
///
/// **Discovered, not listed.** Every `@binding`, every `const` with a Rust spelling,
/// and every struct a `var<uniform>` names is mirrored, for every module in the tree.
/// That is the rule the four hand-kept lists this replaces were converging on: a list
/// makes "the host transcribed something the shader already says" an instance to
/// notice rather than a class that cannot arise, and it was already half-kept — the
/// filter's kind codes were generated while the blend's were transcribed, and the
/// binding tables covered one module of twenty-one.
///
/// `shared` and `vertex` are what the shader does not say about itself; see
/// [`crate::Config`].
///
/// Anything discovery cannot spell in Rust — a nested struct, a non-scalar const — is
/// **skipped with a note in the generated file's header** rather than failing the
/// build. It has to be: discovery reaches declarations no host has ever asked for, and
/// one of them being unmirrorable is not a reason to stop. A caller that needed it
/// still fails, at its own use site.
fn mirrors(
    modules: &[Module],
    shared: &[(&[&str], &str)],
    vertex: &[(&str, &str, &str)],
) -> String {
    let find_module = |path: &str| {
        modules
            .iter()
            .find(|m| m.path == path)
            .unwrap_or_else(|| panic!("`{path}.wesl` is not in the shader tree"))
    };

    // Grouped by the module a declaration is emitted under, in first-seen order.
    // One struct name can be declared by two modules with *different* members
    // (`selection.wesl`'s `Params` and `slice.wesl`'s once were exactly that), so
    // the WESL module has to be part of the Rust path.
    let mut items: Vec<(String, TokenStream)> = Vec::new();
    // What discovery could not spell, in the order it was reached.
    let mut skipped: Vec<String> = Vec::new();

    // `(module path, struct name)` pairs discovery must not emit, because the loop
    // below has already emitted them: every module of a `shared` entry, the canonical
    // one included — it is generated here, with the doc naming the modules it answers
    // for, and discovery reaching it again would be a second definition of one type.
    let mut aliased: Vec<(String, String)> = Vec::new();
    for (sources, name) in shared {
        let (canonical, others) = sources.split_first().expect("a mirror names a module");
        let cm = find_module(canonical);
        let laid = lay_out(require(cm, name), cm).unwrap_or_else(|e| panic!("{e}"));
        for other in others {
            let om = find_module(other);
            let o_laid = lay_out(require(om, name), om).unwrap_or_else(|e| panic!("{e}"));
            structs::agrees(name, canonical, &laid, other, &o_laid);
        }
        aliased.extend(
            sources
                .iter()
                .map(|s| ((*s).to_string(), (*name).to_string())),
        );
        push(&mut items, &cm.rust, structs::emit(name, sources, &laid));
    }

    for m in modules {
        let (consts, skips) = consts::emit(m);
        skipped.extend(skips);
        push(&mut items, &m.rust, consts);
        push(&mut items, &m.rust, bindings::emit(m));
        let (uniforms, skips) = structs::discover(m, &aliased);
        skipped.extend(skips);
        push(&mut items, &m.rust, uniforms);
    }

    for (module, entry, name) in vertex {
        let m = find_module(module);
        push(&mut items, &m.rust, vertex::emit(m, entry, name));
    }
    // The other half of that list being a name and not a membership statement: an
    // entry point the host would have to write a record for by hand is a build
    // failure here instead.
    for m in modules {
        for entry in vertex::instanced_entries(m) {
            assert!(
                vertex.iter().any(|(md, e, _)| *md == m.path && *e == entry),
                "`{}.wesl`'s `@vertex fn {entry}` takes `@location` parameters but is \
                 not named in `VERTEX`, so nothing generates the record it reads",
                m.path,
            );
        }
    }

    let items = items.iter().map(|(module, items)| {
        let ident = format_ident!("{module}");
        let doc = format!(" Host mirrors of what `{module}.wesl` declares.");
        quote! {
            #[doc = #doc]
            pub mod #ident {
                #items
            }
        }
    });

    let file = syn::parse2(quote!(#(#items)*)).expect("the generator emits a parseable file");
    let skipped = if skipped.is_empty() {
        String::new()
    } else {
        // In the header rather than beside the item, because the item is exactly what
        // is *not* there: a reader hunting a mirror that does not exist finds the
        // reason at the top of the file it looked in.
        format!(
            "//\n// Declarations discovery reached and could not spell in Rust:\n{}",
            skipped
                .iter()
                .map(|n| format!("//   {n}\n"))
                .collect::<String>(),
        )
    };
    format!(
        "// @generated by `build/mirror.rs` from the WESL sources — do not edit.\n\
         {skipped}\n{}",
        prettyplease::unparse(&file),
    )
}

/// The `struct name` `m` declares, where the caller named it and a miss is its typo.
fn require<'a>(m: &'a Module, name: &str) -> &'a Struct {
    m.struct_named(name)
        .unwrap_or_else(|| panic!("`{}.wesl` declares no `struct {name}`", m.path))
}

/// Add `item` to the Rust module `rust`'s bag, in first-seen order.
fn push(items: &mut Vec<(String, TokenStream)>, rust: &str, item: TokenStream) {
    if item.is_empty() {
        return;
    }
    match items.iter_mut().find(|(m, _)| m == rust) {
        Some((_, bag)) => bag.extend(item),
        None => items.push((rust.to_string(), item)),
    }
}
