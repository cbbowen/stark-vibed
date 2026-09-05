//! Rust mirrors of what the host writes and the shader reads (§6.10, §7).
//!
//! Three kinds, each generated from the WESL declaration that decides how it is
//! read: the **uniform structs** ([`emit`]), the **constants** both sides compute
//! with ([`emit_consts`]), and the **per-instance vertex records** a vertex entry
//! point's `@location` parameters describe ([`emit_vertex`]).
//!
//! Every uniform on this boundary is one half of a pair the compiler cannot see
//! across. A hand-written Rust half is a second declaration of the same lanes with
//! its own copy of what they mean, and nothing checks the correspondence: the two
//! drift, and a lane the shader has stopped reading goes on being documented as
//! though it were live.
//!
//! So the shader-side declaration is the only one. This walks the WESL AST that
//! `build.rs` already holds and emits the Rust struct from it — fields, padding, and
//! the lane documentation, which lives exactly once.
//!
//! **The layout is the whole point, and it is not the layout `#[repr(C)]` would give.**
//! WGSL aligns a `vec3<f32>` to 16 bytes and sizes it 12; it rounds a struct up to
//! its own alignment; it pads an array's elements and a matrix's columns out to a
//! stride. A Rust struct of the obvious field types agrees with none of that in
//! general — it agrees only when every member is a `vec4`, which is a coincidence to
//! be generated past, not relied on.
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
use wesl::eval::{Context, Convert, Eval, Instance, LiteralInstance, Type, ty_eval_ty};
use wesl::syntax::{
    AddressSpace, Attribute, DeclarationKind, GlobalDeclaration, Struct, TranslationUnit,
};

/// One module of the shader tree, read once.
///
/// The *unlinked* source, always. The linker mangles `Stamp` to
/// `package__1dynamics_Stamp`, emits it once per artifact that reaches it, and strips
/// whatever no entry point uses — the reason the check this replaces could not see a
/// constant that survived only in prose (the retired wick's `WICK_RATE` was the case
/// that proved it). And it drops the comments that are half of what is generated here.
pub struct Module {
    /// The WESL path — `dynamics`, or `lib/paint_common`. How a diagnostic names it.
    path: String,
    /// The Rust module the mirrors land in: the file's own name, without the
    /// directory. `lib` holds the binding-free leaves and is a placement rule rather
    /// than a namespace, so it does not reach the generated paths.
    rust: String,
    src: String,
    tu: TranslationUnit,
}

/// Read every `.wesl` in the tree, sorted by path so the generated file is
/// deterministic (`read_dir` order is not).
fn read_tree(shader_dir: &Path) -> Vec<Module> {
    let mut out = Vec::new();
    let mut claimed: Vec<(String, String)> = Vec::new();
    for dir in ["", "lib"] {
        let at = if dir.is_empty() {
            shader_dir.to_path_buf()
        } else {
            shader_dir.join(dir)
        };
        let mut paths: Vec<_> = std::fs::read_dir(&at)
            .unwrap_or_else(|e| panic!("read {}: {e}", at.display()))
            .map(|e| e.expect("shader dir entry").path())
            .filter(|p| p.extension().is_some_and(|e| e == "wesl"))
            .collect();
        paths.sort();
        for p in paths {
            let stem = p
                .file_stem()
                .and_then(|s| s.to_str())
                .expect("a shader file has a name")
                .to_string();
            let path = if dir.is_empty() {
                stem.clone()
            } else {
                format!("{dir}/{stem}")
            };
            // Two shaders with the same file name in different directories would land
            // in one Rust module and silently merge their items. Refused rather than
            // merged — the mirror's whole job is that one declaration answers for one
            // thing.
            if let Some((other, _)) = claimed.iter().find(|(_, r)| *r == stem) {
                panic!("`{other}.wesl` and `{path}.wesl` would both mirror as `{stem}`");
            }
            claimed.push((path.clone(), stem.clone()));
            let src = std::fs::read_to_string(&p)
                .unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()));
            let tu: TranslationUnit = src
                .parse()
                .unwrap_or_else(|e| panic!("cannot parse {}: {e}", p.display()));
            out.push(Module {
                path,
                rust: stem,
                src,
                tu,
            });
        }
    }
    out
}

/// Generate the host mirrors of everything the shader tree declares, into `dest`.
///
/// **Discovered, not listed.** Every `@binding`, every `const` with a Rust spelling,
/// and every struct a `var<uniform>` names is mirrored, for every module in the tree.
/// That is the rule the four hand-kept lists this replaces were converging on: a list
/// makes "the host transcribed something the shader already says" an instance to
/// notice rather than a class that cannot arise, and it was already half-kept — the
/// filter's kind codes were generated while the blend's were transcribed, and the
/// binding tables covered one module of twenty-one.
///
/// Two things are still named, because the shader does not say them:
///
/// * `shared` — the structs **two or more modules declare identically** against one
///   host type. `View` is written out three times (`composite`, `matte`, `overlay`)
///   and there is one `ViewUniform`; the first module named is where it is generated
///   from, the rest are checked to agree member for member and offset for offset, and
///   then skipped so discovery does not emit a second copy. Generating from one and
///   ignoring the others would move the drift rather than remove it.
/// * `vertex` — the **name** of a per-instance record, since a parameter list has no
///   name of its own. Every `@vertex` entry point that takes `@location` parameters
///   must be named here; one that is not is a build failure rather than a record the
///   host would then write by hand.
///
/// Anything discovery cannot spell in Rust — a nested struct, a non-scalar const — is
/// **skipped with a note in the generated file** rather than failing the build. It has
/// to be: discovery reaches declarations no host has ever asked for, and one of them
/// being unmirrorable is not a reason to stop. A caller that needed it still fails, at
/// its own use site.
pub fn generate(
    shader_dir: &Path,
    dest: &Path,
    shared: &[(&[&str], &str)],
    vertex: &[(&str, &str, &str)],
) {
    let modules = read_tree(shader_dir);
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

    // `(module path, struct name)` pairs discovery must not emit, because the loop
    // below has already emitted them: every module of a `shared` entry, the canonical
    // one included — it is generated here, with the doc naming the modules it answers
    // for, and discovery reaching it again would be a second definition of one type.
    let mut aliased: Vec<(String, String)> = Vec::new();
    for (sources, name) in shared {
        let (canonical, others) = sources.split_first().expect("a mirror names a module");
        let cm = find_module(canonical);
        let laid = lay_out(find(cm, name), cm).unwrap_or_else(|e| panic!("{e}"));
        for other in others {
            let om = find_module(other);
            let o_laid = lay_out(find(om, name), om).unwrap_or_else(|e| panic!("{e}"));
            agrees(name, canonical, &laid, other, &o_laid);
        }
        aliased.extend(
            sources
                .iter()
                .map(|s| ((*s).to_string(), (*name).to_string())),
        );
        push(&mut items, &cm.rust, emit(name, sources, &laid));
    }

    for m in &modules {
        push(&mut items, &m.rust, emit_consts(m));
        push(&mut items, &m.rust, emit_bindings(m));
        push(&mut items, &m.rust, emit_uniform_structs(m, &aliased));
    }

    for (module, entry, name) in vertex {
        let m = find_module(module);
        push(&mut items, &m.rust, emit_vertex(m, entry, name));
    }
    // The other half of that list being a name and not a membership statement: an
    // entry point the host would have to write a record for by hand is a build
    // failure here instead.
    for m in &modules {
        for entry in instanced_vertex_entries(m) {
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
    let skipped = SKIPPED.with_borrow(|s| {
        if s.is_empty() {
            String::new()
        } else {
            // In the header rather than beside the item, because the item is exactly
            // what is *not* there: a reader hunting a mirror that does not exist finds
            // the reason at the top of the file it looked in.
            format!(
                "//\n// Declarations discovery reached and could not spell in Rust:\n{}",
                s.iter().map(|n| format!("//   {n}\n")).collect::<String>(),
            )
        }
    });
    let text = format!(
        "// @generated by `build/mirror.rs` from the WESL sources — do not edit.\n\
         {skipped}\n{}",
        prettyplease::unparse(&file),
    );
    std::fs::write(dest, text).unwrap_or_else(|e| panic!("write {}: {e}", dest.display()));
}

thread_local! {
    /// What discovery could not spell, reported in the generated file's header.
    ///
    /// A thread-local rather than a parameter threaded through eight emitters: a skip
    /// is a diagnostic about the *run*, not a value any caller acts on, and the build
    /// script is single-threaded.
    static SKIPPED: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn skipped(note: String) {
    SKIPPED.with_borrow_mut(|s| s.push(note));
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

/// Every `@vertex` entry point in `m` that takes at least one `@location` parameter —
/// i.e. that reads a per-instance record the host has to lay out.
fn instanced_vertex_entries(m: &Module) -> Vec<String> {
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
fn emit_vertex(m: &Module, entry: &str, name: &str) -> TokenStream {
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

    let mut ctx = Context::new(tu);
    let (mut fields, mut attrs, mut offset) = (Vec::new(), Vec::new(), 0u32);

    for p in &func.parameters {
        // `@builtin(vertex_index)` and friends come from the pipeline, not the buffer.
        let Some(location) = p.attributes.iter().find_map(|a| match &**a {
            Attribute::Location(e) => Some(e),
            _ => None,
        }) else {
            continue;
        };
        let location = match location.eval_value(&mut ctx).ok().and_then(|i| match i {
            Instance::Literal(LiteralInstance::AbstractInt(n)) => u32::try_from(n).ok(),
            Instance::Literal(LiteralInstance::U32(n)) => Some(n),
            Instance::Literal(LiteralInstance::I32(n)) => u32::try_from(n).ok(),
            _ => None,
        }) {
            Some(n) => n,
            None => panic!("`{module}.wesl`'s `{entry}` has a `@location` that is not a number"),
        };

        let member = p.ident.name();
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
fn emit_bindings(m: &Module) -> TokenStream {
    let (tu, src, module) = (&m.tu, m.src.as_str(), m.path.as_str());
    let mut names: Vec<String> = Vec::new();
    let mut indices = Vec::new();
    let mut decls = Vec::new();
    let mut table = Vec::new();
    for d in &tu.global_declarations {
        let GlobalDeclaration::Declaration(decl) = &**d else {
            continue;
        };
        let Some(expr) = decl.attributes.iter().find_map(|a| match &**a {
            Attribute::Binding(e) => Some(e),
            _ => None,
        }) else {
            continue;
        };
        let member = decl.ident.name();
        let at = |what: &str, e: &wesl::syntax::ExpressionNode| -> u32 {
            match e
                .eval_value(&mut Context::new(tu))
                .ok()
                .and_then(|i| match i {
                    Instance::Literal(LiteralInstance::AbstractInt(n)) => u32::try_from(n).ok(),
                    Instance::Literal(LiteralInstance::U32(n)) => Some(n),
                    Instance::Literal(LiteralInstance::I32(n)) => u32::try_from(n).ok(),
                    _ => None,
                }) {
                Some(n) => n,
                None => panic!("`{module}.wesl`'s `{member}` has a `@{what}` that is not a number"),
            }
        };
        let index = at("binding", expr);
        // The group the slot is in. A bind group layout is for exactly one group, so
        // this is what lets the host assert that a slot list names one — and what a
        // table keyed on the index alone could never have said.
        let group = decl
            .attributes
            .iter()
            .find_map(|a| match &**a {
                Attribute::Group(e) => Some(at("group", e)),
                _ => None,
            })
            .unwrap_or_else(|| {
                panic!("`{module}.wesl`'s `{member}` has a `@binding` but no `@group`")
            });
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
        let kind = bind_kind(decl, module, &member, tu);
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
fn bind_kind(
    decl: &wesl::syntax::Declaration,
    module: &str,
    member: &str,
    tu: &TranslationUnit,
) -> TokenStream {
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
        let size = uniform_size(ty, module, member, tu);
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
fn expr_ident(expr: &wesl::syntax::ExpressionNode) -> Option<String> {
    match &**expr {
        wesl::syntax::Expression::TypeOrIdentifier(t) => Some(t.ident.name().to_string()),
        _ => None,
    }
}

/// The WGSL size of a uniform binding's declared type — its `min_binding_size`.
fn uniform_size(
    ty: &wesl::syntax::TypeExpression,
    module: &str,
    member: &str,
    tu: &TranslationUnit,
) -> u64 {
    let mut ctx = Context::new(tu);
    let resolved = ty_eval_ty(ty, &mut ctx).unwrap_or_else(|e| {
        panic!("`{module}.wesl`'s `{member}` has an unresolvable uniform type: {e}")
    });
    resolved
        .size_of()
        .unwrap_or_else(|| panic!("`{module}.wesl`'s `{member}` has an unsized uniform type"))
        as u64
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

/// Emit `const NAME` from `module` as a Rust constant of the same type and value.
///
/// **Evaluated, not read.** The value is whatever `wesl`'s const evaluator makes of
/// the initializer, so a constant derived from its neighbours comes out as the
/// number the shader will actually compute with. The check this replaces parsed a
/// decimal literal out of the *linked* source and could do neither: a derived
/// constant is not a literal, and the linker had already stripped anything no entry
/// point reached.
fn emit_consts(m: &Module) -> TokenStream {
    let (tu, src, module) = (&m.tu, m.src.as_str(), m.path.as_str());
    let mut out = TokenStream::new();
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
        let mut ctx = Context::new(tu);
        let at = || format!("`{module}.wesl`'s `const {name}`");
        let Ok(value) = init.eval_value(&mut ctx) else {
            skipped(format!("{} does not const-evaluate", at()));
            continue;
        };
        let Ok(ty) = ty_eval_ty(declared, &mut ctx) else {
            skipped(format!("{} has no resolvable type", at()));
            continue;
        };
        // `2` evaluates to an *abstract* int, and `4.0` to an abstract float — WGSL
        // defers the choice of a concrete type to the declaration. Converting to the
        // declared type is what the shader itself does, and doing it here is why the
        // generated constant is the type the shader computes with rather than whichever
        // one the literal happened to look like.
        let Some(value) = value.convert_to(&ty) else {
            skipped(format!("{} is a `{value}`, which is not a `{ty}`", at()));
            continue;
        };
        let Instance::Literal(lit) = &value else {
            // An array, a matrix, a struct: real declarations that a host constant
            // cannot be. `BLEED_OFFS`'s stencil offsets are the case, and the host
            // derives its own from `BLEED_LADDER_TAPS` beside it.
            skipped(format!("{} is not a scalar", at()));
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
                skipped(format!("{} is a `{ty}`, which has no host constant", at()));
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
    out
}

/// Emit a mirror for every struct a `var<uniform>` in `m` names — the boundary the
/// host writes across, discovered rather than listed (§2).
///
/// `aliased` holds the `(module, struct)` pairs a `shared` entry already generated
/// under another module, so the two do not both emit one.
fn emit_uniform_structs(m: &Module, aliased: &[(String, String)]) -> TokenStream {
    let mut out = TokenStream::new();
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
        let Some(s) = find_opt(m, name) else { continue };
        done.push(name.to_string());
        match lay_out(s, m) {
            Ok(laid) => out.extend(emit(name, &[m.path.as_str()], &laid)),
            // The reason discovery must not panic: it reaches every uniform in the
            // tree, and one that a host has never asked for being unmirrorable is not
            // a reason to stop. A caller that needed it fails at its own use site.
            Err(why) => skipped(why),
        }
    }
    out
}

/// The `struct name` declared in `m`, if it declares one.
fn find_opt<'a>(m: &'a Module, name: &str) -> Option<&'a Struct> {
    m.tu.global_declarations.iter().find_map(|d| match &**d {
        GlobalDeclaration::Struct(s) if s.ident.name().as_str() == name => Some(s),
        _ => None,
    })
}

/// [`find_opt`] where the caller named the struct and a miss is its typo.
fn find<'a>(m: &'a Module, name: &str) -> &'a Struct {
    find_opt(m, name).unwrap_or_else(|| panic!("`{}.wesl` declares no `struct {name}`", m.path))
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

/// A `u64` literal — what `wgpu::VertexAttribute::offset` is typed as.
fn lit_u64(n: u32) -> proc_macro2::Literal {
    proc_macro2::Literal::u64_suffixed(n.into())
}

/// Place `s`'s members at their WGSL offsets.
///
/// Fallible rather than panicking, because discovery calls it on every uniform in the
/// tree (`emit_uniform_structs`) and one it cannot spell has to be skipped with a
/// note, not stop the build. A member it *can* reach but cannot place is still fatal
/// to the struct as a whole — every member after it would land at the wrong offset —
/// which is what the error says.
fn lay_out(s: &Struct, module: &Module) -> Result<Laid, String> {
    let (src, tu, path) = (module.src.as_str(), &module.tu, module.path.as_str());
    let name = s.ident.name();
    if s.members.is_empty() {
        return Err(format!("`{path}::{name}` has no members"));
    }
    let mut ctx = Context::new(tu);

    let (mut fields, mut offset, mut align) = (Vec::new(), 0u32, 1u32);
    // Documentation for the first member runs from the opening brace.
    let mut prev_end = src[..s.members[0].span().range().start]
        .rfind('{')
        .expect("a struct body opens")
        + 1;

    for m in &s.members {
        let member = m.ident.name();
        let fail = |what: &str| -> String {
            format!(
                "`{path}::{name}.{member}` is a `{}`, which {what}, so `{name}` is not \
                 mirrored",
                m.ty,
            )
        };
        let ty = match ty_eval_ty(&m.ty, &mut ctx) {
            Ok(ty) => ty,
            Err(e) => return Err(fail(&format!("did not resolve: {e}"))),
        };
        let (Some(m_size), Some(m_align)) = (ty.size_of(), ty.align_of()) else {
            return Err(fail("is not host-shareable"));
        };
        let Some(spelling) = rust_ty(&ty, m_size) else {
            return Err(fail("has no Rust spelling"));
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
    Ok(Laid {
        fields,
        size,
        align,
    })
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
    fence_indented(docs)
}

/// Fence every indented run of carried-over shader prose as `text`.
///
/// **Shader prose is never Rust.** Markdown reads a four-space indent after a blank
/// line as a code block, and `rustdoc` then reads that block as a *doctest* — so
/// `dynamics.wesl`'s
///
/// ```text
///     owed = prefix(l),   received = rowtotal − prefix(l)
/// ```
///
/// came through as two failing doctests complaining that `−` is not a Rust token. The
/// indentation is the shader author's, marking an equation or a table; every one of
/// them is prose about a WGSL kernel and none is a Rust example. Fencing as `text` is
/// what says so, and it keeps the layout the author chose rather than flattening it.
///
/// Only runs that markdown would actually take as code are fenced — an indent that
/// continues a paragraph, with no blank line before it, is left alone.
fn fence_indented(docs: Vec<String>) -> Vec<String> {
    // A doc line carries one conventional leading space (`// Text` → ` Text`), so a
    // markdown code indent is that plus four.
    let indented = |l: &String| l.starts_with("     ");
    let blank = |l: &String| l.trim().is_empty();
    let mut out: Vec<String> = Vec::with_capacity(docs.len());
    let mut open = false;
    for line in docs {
        if open && !indented(&line) && !blank(&line) {
            out.push(" ```".to_string());
            open = false;
        } else if !open && indented(&line) && out.last().is_some_and(blank) {
            out.push(" ```text".to_string());
            open = true;
        }
        out.push(line);
    }
    if open {
        out.push(" ```".to_string());
    }
    out
}
