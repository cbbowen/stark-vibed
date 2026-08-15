//! Rust mirrors of what the host writes and the shader reads (§6.10, §7).
//!
//! Three kinds, each generated from the WESL declaration that decides how it is
//! read: the **uniform structs** ([`emit`]), the **constants** both sides compute
//! with ([`emit_const`]), and the **per-instance vertex records** a vertex entry
//! point's `@location` parameters describe ([`emit_vertex`]).
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
use wesl::eval::{Context, Convert, Eval, Instance, LiteralInstance, Type, ty_eval_ty};
use wesl::syntax::{
    AddressSpace, Attribute, DeclarationKind, GlobalDeclaration, Struct, TranslationUnit,
};

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
pub fn generate(
    shader_dir: &Path,
    dest: &Path,
    wanted: &[(&[&str], &str)],
    consts: &[(&str, &str)],
    vertex: &[(&str, &str, &str)],
    bindings: &[&str],
) {
    // Grouped by the module a declaration is emitted under, in first-seen order.
    // One struct name can be declared by two modules with *different* members
    // (`selection.wesl`'s `Params` and `slice.wesl`'s once were exactly that), so
    // the WESL module has to be part of the Rust path.
    let mut modules: Vec<(String, TokenStream)> = Vec::new();
    let mut push = |module: &str, item: TokenStream| {
        // A module under `lib/` is named for its file, not its path: `lib` holds the
        // binding-free leaves and is a placement rule rather than a namespace. Two
        // shaders with the same file name in different directories would collide, so
        // the assertion below refuses rather than silently merging them.
        let rust = module.rsplit('/').next().expect("a module has a name");
        match modules.iter_mut().find(|(m, _)| m == rust) {
            Some((_, items)) => items.extend(item),
            None => modules.push((rust.to_string(), item)),
        }
    };
    let read = |module: &str| {
        let path = shader_dir.join(format!("{module}.wesl"));
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        // The *unlinked* source. The linker mangles `Stamp` to
        // `package__1dynamics_Stamp`, emits it once per artifact that reaches it, and
        // strips whatever no entry point uses — the reason the check this replaces
        // could not see a constant that survived only in prose (the retired wick's
        // `WICK_RATE` was the case that proved it). And it drops the comments that
        // are half of what is being generated here.
        let tu: TranslationUnit = src
            .parse()
            .unwrap_or_else(|e| panic!("cannot parse {}: {e}", path.display()));
        (src, tu)
    };

    for (sources, name) in wanted {
        let (canonical, others) = sources.split_first().expect("a mirror names a module");
        let (src, tu) = read(canonical);
        let laid = lay_out(find(&tu, canonical, name), &src, canonical, &tu);

        for other in others {
            let (o_src, o_tu) = read(other);
            let o_laid = lay_out(find(&o_tu, other, name), &o_src, other, &o_tu);
            agrees(name, canonical, &laid, other, &o_laid);
        }
        push(canonical, emit(name, sources, &laid));
    }

    for (module, name) in consts {
        let (src, tu) = read(module);
        push(module, emit_const(&tu, &src, module, name));
    }

    for (module, entry, name) in vertex {
        let (src, tu) = read(module);
        push(module, emit_vertex(&tu, &src, module, entry, name));
    }

    for module in bindings {
        let (src, tu) = read(module);
        push(module, emit_bindings(&tu, &src, module));
    }

    let items = modules.iter().map(|(module, items)| {
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
    let text = format!(
        "// @generated by `build/mirror.rs` from the WESL sources — do not edit.\n\n{}",
        prettyplease::unparse(&file),
    );
    std::fs::write(dest, text).unwrap_or_else(|e| panic!("write {}: {e}", dest.display()));
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
fn emit_vertex(
    tu: &TranslationUnit,
    src: &str,
    module: &str,
    entry: &str,
    name: &str,
) -> TokenStream {
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

/// Emit `pub mod binding` holding one `u32` const per `@binding` declaration in
/// `module` — the indices the host's bind-group layouts and bind-group entries name.
///
/// The third transcription of the same boundary, and the one with the least
/// redundancy to catch it: a binding number was written in the WESL declaration, in
/// the layout entry, and in the bind-group entry, with margin comments as the only
/// map between them. The struct mirrors made the *lanes* single-sourced; this does
/// the same for the *slots*, so a renumbering in the shader is a one-file change
/// that the host follows by name.
///
/// Every declaration is emitted, `@if`-gated ones included — the unlinked source
/// keeps them, and a host that binds one does so exactly when the matching feature
/// build declares it. The name is the WESL variable's, uppercased; two declarations
/// that collide there are a build failure rather than a silent shadowing.
fn emit_bindings(tu: &TranslationUnit, src: &str, module: &str) -> TokenStream {
    let mut ctx = Context::new(tu);
    let mut names: Vec<String> = Vec::new();
    let mut consts = Vec::new();
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
        let index = match expr.eval_value(&mut ctx).ok().and_then(|i| match i {
            Instance::Literal(LiteralInstance::AbstractInt(n)) => u32::try_from(n).ok(),
            Instance::Literal(LiteralInstance::U32(n)) => Some(n),
            Instance::Literal(LiteralInstance::I32(n)) => u32::try_from(n).ok(),
            _ => None,
        }) {
            Some(n) => n,
            None => panic!("`{module}.wesl`'s `{member}` has a `@binding` that is not a number"),
        };
        let name = member.to_uppercase();
        assert!(
            !names.contains(&name),
            "`{module}.wesl` declares two bindings that both mirror as `{name}`"
        );
        names.push(name.clone());
        let docs = doc_lines(&src[..d.span().range().start]);
        let ident = format_ident!("{name}");
        let index_lit = lit(index);
        consts.push(quote! {
            #(#[doc = #docs])*
            pub const #ident: u32 = #index_lit;
        });

        // The rest of what the declaration decides: what kind of thing occupies the
        // slot, and whether it exists at all in a build without the residual.
        let kind = bind_kind(decl, module, &member, tu);
        // `@if(resid)` — the gate that used to be transcribed as a per-layout element
        // count (`[..12 + 4 * usize::from(resid)]`, seven of them).
        let resid = decl.attributes.iter().any(|a| match &**a {
            Attribute::If(e) => src[e.span().range()].trim() == "resid",
            _ => false,
        });
        // No doc comment here: an element of an array expression is not an item, so a
        // `///` on one is inert (and a warning). The prose lives on the `binding::`
        // constant above, which is the name a reader looks the slot up by anyway.
        table.push(quote! {
            Binding {
                index: #index_lit,
                name: #name,
                kind: #kind,
                resid: #resid,
            }
        });
    }
    assert!(
        !consts.is_empty(),
        "`{module}.wesl` declares no `@binding`s, so there is nothing to mirror"
    );
    let index_doc = format!(
        " The `@binding` indices `{module}.wesl` declares, named for their WESL\n \
         variables — the shader's declarations are the only ones.",
    );
    let table_doc = format!(
        " Every `@binding` `{module}.wesl` declares, in declaration order: its index,\n \
         what kind of thing occupies it, and whether it is `@if(resid)`-gated.\n\n \
         The host builds both its bind-group **layouts** and its bind **groups** from\n \
         this, so the two cannot disagree about a slot's type, its storage format, or\n \
         whether the residual build has it. Look up a slot with [`binding_of`].",
    );
    quote! {
        // The descriptor types are hand-written in `lib.rs` — they are the host's
        // vocabulary, not the shader's — and this generated module sits two levels
        // below the crate root, so it names them absolutely.
        use crate::{BindKind, Binding};

        #[doc = #index_doc]
        pub mod binding {
            #(#consts)*
        }

        #[doc = #table_doc]
        pub const BINDINGS: &[Binding] = &[#(#table),*];

        /// The declaration at `index`, or `None` if this module declares no such
        /// binding — which for a host naming its own layout is a typo, and worth a
        /// loud one.
        pub fn binding_of(index: u32) -> Option<&'static Binding> {
            BINDINGS.iter().find(|b| b.index == index)
        }
    }
}

/// What kind of thing a `@binding` declaration puts in its slot, as a `BindKind`
/// expression.
///
/// Read off the declared type, which is where the answer already is: a
/// `texture_storage_2d<rgba32float, write>` is a storage texture of that format, and
/// the host had been choosing between `stor` and `stor32` by hand at every layout that
/// named one.
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
    if let Some(dim) = name.strip_prefix("texture_storage_") {
        // `<format, access>`; only the format reaches the host's descriptor, the
        // access mode being implied by the layout entry the host builds.
        let args = ty
            .template_args
            .as_ref()
            .unwrap_or_else(|| panic!("`{module}.wesl`'s `{member}` has no storage format"));
        let format = expr_ident(&args[0].expression).unwrap_or_else(|| {
            panic!("`{module}.wesl`'s `{member}` has a storage format that is not a name")
        });
        return quote!(BindKind::Storage { dim: #dim, format: #format });
    }
    if let Some(dim) = name.strip_prefix("texture_") {
        return quote!(BindKind::Texture { dim: #dim });
    }
    panic!("`{module}.wesl`'s `{member}` has type `{name}`, which is not a binding kind");
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
fn emit_const(tu: &TranslationUnit, src: &str, module: &str, name: &str) -> TokenStream {
    let decl = tu
        .global_declarations
        .iter()
        .find_map(|d| match &**d {
            GlobalDeclaration::Declaration(decl)
                if decl.kind.is_const() && decl.ident.name().as_str() == name =>
            {
                Some((decl, d.span().range()))
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("`{module}.wesl` declares no `const {name}`"));
    let (decl, span) = decl;

    let init = decl
        .initializer
        .as_ref()
        .unwrap_or_else(|| panic!("`{module}.wesl`'s `const {name}` has no value"));
    let mut ctx = Context::new(tu);
    let value = init
        .eval_value(&mut ctx)
        .unwrap_or_else(|e| panic!("`{module}.wesl`'s `const {name}` does not evaluate: {e}"));

    // An explicit type is required rather than inferred from the value. WGSL's
    // *abstract* numerics have no Rust counterpart to pick — `const N = 4` could
    // honestly become an `i32`, a `u32` or an `f32` — and guessing is how a host
    // constant ends up a different type from the one the shader computes with.
    let declared = decl.ty.as_ref().unwrap_or_else(|| {
        panic!(
            "`{module}.wesl`'s `const {name}` has no declared type, so there is no \
             Rust type to generate. Write one (`const {name}: f32 = …`)."
        )
    });
    let ty = ty_eval_ty(declared, &mut ctx)
        .unwrap_or_else(|e| panic!("`{module}.wesl`'s `const {name}` has no type: {e}"));

    // `2` evaluates to an *abstract* int, and `4.0` to an abstract float — WGSL
    // defers the choice of a concrete type to the declaration. Converting to the
    // declared type is what the shader itself does, and doing it here is why the
    // generated constant is the type the shader computes with rather than whichever
    // one the literal happened to look like.
    let value = value.convert_to(&ty).unwrap_or_else(|| {
        panic!("`{module}.wesl`'s `const {name}` is a `{value}`, which is not a `{ty}`")
    });
    let Instance::Literal(lit) = &value else {
        panic!(
            "`{module}.wesl`'s `const {name}` is not a scalar, which is all a host constant can be"
        )
    };
    let (rust, literal) = match (&ty, lit) {
        (Type::F32, LiteralInstance::F32(v)) => {
            assert!(
                v.is_finite(),
                "`{module}.wesl`'s `const {name}` is {v}, which no Rust literal spells"
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
        _ => panic!(
            "`{module}.wesl`'s `const {name}` is a `{ty}` holding `{value}`, which has \
             no host constant. Scalars only."
        ),
    };
    let literal: TokenStream = literal.parse().expect("a scalar literal is one token");

    let ident = format_ident!("{name}");
    let mut docs = doc_lines(&src[..span.start]);
    docs.push(String::new());
    docs.push(format!(
        " Generated from `{module}.wesl`'s `const {name}` — the shader's declaration is",
    ));
    docs.push(" the only one.".to_string());
    quote! {
        #(#[doc = #docs])*
        pub const #ident: #rust = #literal;
    }
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
