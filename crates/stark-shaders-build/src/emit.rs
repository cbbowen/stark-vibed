//! Rust mirrors of what the host writes and the shader reads (§6.10, §7).
//!
//! Four kinds, each generated from the WESL declaration that decides how it is read:
//! the **uniform structs** ([`structs`]), the **constants** both sides compute with
//! ([`consts`]), the **`@binding` declarations** ([`bindings`]), and the
//! **per-instance vertex records** a vertex entry point takes ([`vertex`]).
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
use wesl::eval::{Context, Type, ty_eval_ty};
use wesl::syntax::TypeExpression;

use crate::tree::{Module, read_tree};

/// The WGSL type a `var<uniform>` names, resolved against the module that declares it.
///
/// **A name this module neither declares nor shares with WGSL is an imported struct, and
/// that is refused.** Not resolved: an import path (`self::`, `super::`, `package::`,
/// aliases, item collections) would be a second resolver written here, which is the
/// transcription §6.10 is about — and the mirror it produced would land under the
/// *importing* module, naming a file that does not declare the struct. A host type
/// several pipelines want is a shared module they import the *binding* from
/// (`view.wesl`), which puts the declaration in one file and the mirror under it.
fn uniform_type(ty: &TypeExpression, m: &Module, member: &str, ctx: &mut Context<'_>) -> Type {
    ty_eval_ty(ty, ctx).unwrap_or_else(|e| {
        panic!(
            "`{}.wesl`'s `var<uniform> {member}: {}` names a type the module does not \
             declare ({e}). A struct reached through an import has no mirror: the \
             generator reads the unlinked source, where the import is only a name. \
             Declare it here, or import the binding itself from the module that does \
             (`view.wesl`).",
            m.path,
            ty.ident.name(),
        )
    })
}

/// [`uniform_type`] for a caller that wants the refusal and not the type.
///
/// `structs::discover` has already looked the struct up locally and missed; what is left
/// to decide is whether that was a type the host already has or an import.
fn refuse_imported_uniform(ty: &TypeExpression, m: &Module, member: &str, ctx: &mut Context<'_>) {
    uniform_type(ty, m, member, ctx);
}

/// Generate the host mirrors of everything the shader tree at `shader_dir` declares,
/// into `dest`.
pub(crate) fn generate(shader_dir: &Path, dest: &Path) {
    let text = mirrors(&read_tree(shader_dir));
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
/// Anything discovery cannot spell in Rust — a nested struct, a matrix const — is
/// **skipped with a note in the generated file's header** rather than failing the
/// build. It has to be: discovery reaches declarations no host has ever asked for, and
/// one of them being unmirrorable is not a reason to stop. A caller that needed it
/// still fails, at its own use site.
fn mirrors(modules: &[Module]) -> String {
    // Grouped by the module a declaration is emitted under, in first-seen order.
    // One struct name can be declared by two modules with *different* members
    // (`selection.wesl`'s `Params` and `slice.wesl`'s once were exactly that), so
    // the WESL module has to be part of the Rust path.
    let mut items: Vec<(String, TokenStream)> = Vec::new();
    // What discovery could not spell, in the order it was reached.
    let mut skipped: Vec<String> = Vec::new();

    for m in modules {
        let (consts, skips) = consts::emit(m);
        skipped.extend(skips);
        push(&mut items, &m.rust, consts);
        push(&mut items, &m.rust, bindings::emit(m));
        let (uniforms, skips) = structs::discover(m);
        skipped.extend(skips);
        push(&mut items, &m.rust, uniforms);
        push(&mut items, &m.rust, vertex::emit(m));
    }

    let items = items.iter().map(|(module, items)| {
        let ident = format_ident!("{module}");
        // Named by the file, `lib/` and all: a mirror module is the leaf's name, but
        // the file a reader has to open is the one the tree holds.
        let path = modules
            .iter()
            .find(|m| m.rust == *module)
            .map_or(module.as_str(), |m| m.path.as_str());
        let doc = format!(" Host mirrors of what `{path}.wesl` declares.");
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
    // The one convention of this file a reader of it cannot derive: `vec3<f32>` is
    // `[f32; 3]` as a constant and `[f32; 4]` as a uniform member, because the second
    // has a stride to fill and the first is only a number.
    format!(
        "// @generated by `stark-shaders-build` from the WESL sources — do not edit.\n\
         //\n\
         // A `vecN` or `array` constant is its own lanes. The same WGSL type as a uniform\n\
         // member is instead the padded stride it occupies there, which can be wider.\n\
         {skipped}\n{}",
        prettyplease::unparse(&file),
    )
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

/// What the generator does today, written down so that changing it is deliberate.
///
/// These are **characterisation** tests: the expectations are the current output rather
/// than an independently-derived one. A correctness fix is supposed to fail them, and
/// the diff is what it changed.
///
/// [`mirrors`] takes an already-parsed module list, so none of this needs a shader tree
/// on disk.
#[cfg(test)]
mod tests {
    use super::*;

    /// The generated file's **body** for one module of inline WESL. The header is one
    /// test's business ([`a_const_derived_from_its_neighbours_mirrors_as_its_value`])
    /// rather than every test's.
    fn generated(src: &str) -> String {
        past_header(&mirrors(&[Module::parse("probe", src)]))
    }

    fn past_header(out: &str) -> String {
        out.split_once("\n\n")
            .expect("a header, then the body")
            .1
            .to_string()
    }

    /// `params` carries no `@group`/`@binding` — legal only here, and deliberate: it
    /// keeps this test's whole-file expectation about the struct.
    #[test]
    fn a_uniform_struct_mirrors_at_its_wgsl_offsets() {
        let out = generated(
            r"
struct Params {
    // How far the edge falls off, in canvas px.
    feather: f32,
    // The rectangle the mask covers.
    //
    //     x0 y0 x1 y1
    rect: vec4<f32>,
    // The gradient's stops.
    stops: array<vec4<f32>, 2>,
    // A packed triple: WGSL aligns it to 16 and sizes it 12.
    tint: vec3<f32>,
}

var<uniform> params: Params;
",
        );
        assert_eq!(
            out,
            r#"/// Host mirrors of what `probe.wesl` declares.
pub mod probe {
    /// `Params`, generated from `probe.wesl` — the shader's declaration is the only one.
    ///
    /// WGSL size 80, alignment 16.
    #[repr(C, align(16))]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct Params {
        /// How far the edge falls off, in canvas px.
        pub feather: f32,
        /// Padding to the 16-byte WGSL alignment that follows.
        pub _pad_1: [u8; 12],
        /// The rectangle the mask covers.
        ///
        /// ```text
        ///     x0 y0 x1 y1
        /// ```
        pub rect: [f32; 4],
        /// The gradient's stops.
        pub stops: [[f32; 4]; 2],
        /// A packed triple: WGSL aligns it to 16 and sizes it 12.
        pub tint: [f32; 3],
        /// Padding to the 16-byte WGSL alignment that follows.
        pub _pad_5: [u8; 4],
    }
    impl Default for Params {
        fn default() -> Self {
            bytemuck::Zeroable::zeroed()
        }
    }
    const _: () = {
        const SIZE: usize = core::mem::size_of::<Params>();
        const ALIGN: usize = core::mem::align_of::<Params>();
        assert!(SIZE == 80, "`Params` is not 80 bytes");
        assert!(ALIGN == 16, "`Params` is not 16-byte aligned");
        const OFFSET_OF_FEATHER: usize = core::mem::offset_of!(Params, feather);
        assert!(OFFSET_OF_FEATHER == 0, "`Params.feather` is not at WGSL offset 0");
        const OFFSET_OF_RECT: usize = core::mem::offset_of!(Params, rect);
        assert!(OFFSET_OF_RECT == 16, "`Params.rect` is not at WGSL offset 16");
        const OFFSET_OF_STOPS: usize = core::mem::offset_of!(Params, stops);
        assert!(OFFSET_OF_STOPS == 32, "`Params.stops` is not at WGSL offset 32");
        const OFFSET_OF_TINT: usize = core::mem::offset_of!(Params, tint);
        assert!(OFFSET_OF_TINT == 64, "`Params.tint` is not at WGSL offset 64");
    };
}
"#,
        );
    }

    /// `SHARE` is the one that says *evaluated, not read*: `7.0 / 64.0` is no literal,
    /// and `0.109375` is the shortest decimal that reads back to the same `f32` bits.
    /// `UNTYPED` is passed over in silence — WGSL's abstract numerics have no Rust
    /// counterpart to pick, and that is the shader deferring the choice, not a mirror
    /// that failed. A const with no comment of its own still gets the blank `///`.
    #[test]
    fn a_scalar_const_of_each_type_mirrors_as_that_type() {
        let out = generated(
            r"
// How many taps the ladder walks.
const TAPS: i32 = 4;
const WIDTH: u32 = 3u;
const RISE: f32 = 0.05;
const SHARE: f32 = 7.0 / 64.0;
const WRAPS: bool = true;
const UNTYPED = 7;
",
        );
        assert_eq!(
            out,
            r"/// Host mirrors of what `probe.wesl` declares.
pub mod probe {
    /// How many taps the ladder walks.
    ///
    /// Generated from `probe.wesl`'s `const TAPS` — the shader's declaration is
    /// the only one.
    pub const TAPS: i32 = 4;
    ///
    /// Generated from `probe.wesl`'s `const WIDTH` — the shader's declaration is
    /// the only one.
    pub const WIDTH: u32 = 3;
    ///
    /// Generated from `probe.wesl`'s `const RISE` — the shader's declaration is
    /// the only one.
    pub const RISE: f32 = 0.05;
    ///
    /// Generated from `probe.wesl`'s `const SHARE` — the shader's declaration is
    /// the only one.
    pub const SHARE: f32 = 0.109375;
    ///
    /// Generated from `probe.wesl`'s `const WRAPS` — the shader's declaration is
    /// the only one.
    pub const WRAPS: bool = true;
}
",
        );
    }

    /// The header, and the one shape that still reaches it. `DOUBLE` used to be here
    /// too — see [`crate::eval::module_context`] — and so did an `array` of scalars,
    /// until one could be spelled.
    #[test]
    fn a_const_derived_from_its_neighbours_mirrors_as_its_value() {
        let out = mirrors(&[Module::parse(
            "probe",
            r"
const RISE: f32 = 0.05;
const DOUBLE: f32 = RISE * 2.0;
const FLIP: mat2x2<f32> = mat2x2<f32>(vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0));
",
        )]);
        let (header, body) = out.split_once("\n\n").expect("a header, then the body");
        assert_eq!(
            header,
            "// @generated by `stark-shaders-build` from the WESL sources — do not edit.\n\
             //\n\
             // A `vecN` or `array` constant is its own lanes. The same WGSL type as a uniform\n\
             // member is instead the padded stride it occupies there, which can be wider.\n\
             //\n\
             // Declarations discovery reached and could not spell in Rust:\n\
             //   `probe.wesl`'s `const FLIP` is a `mat2x2<f32>`, which has no host constant",
        );
        assert!(body.contains("pub const RISE: f32 = 0.05;"), "{body}");
        assert!(body.contains("pub const DOUBLE: f32 = 0.1;"), "{body}");
    }

    /// A vector is its lanes and an array its elements — the value's own shape, not the
    /// padded stride a uniform member of the same type would occupy. `HUE` is
    /// `guides.wesl`'s case (§20.4) and `LOBES` is `filter_common.wesl`'s (§21.10): a
    /// colour and a table of coefficients, each stated once.
    #[test]
    fn a_vector_and_an_array_const_mirror_as_rust_arrays() {
        let out = generated(
            r"
// The x axis, display sRGB.
const HUE: vec3<f32> = vec3<f32>(0.9349, 0.3629, 0.3803);
const TAPS: array<u32, 3> = array<u32, 3>(1u, 2u, 4u);
const LOBES: array<vec2<f32>, 2> = array<vec2<f32>, 2>(
    vec2<f32>(1.056, 599.8),
    vec2<f32>(-0.065, 501.1),
);
",
        );
        assert!(
            out.contains("pub const HUE: [f32; 3] = [0.9349, 0.3629, 0.3803];"),
            "{out}"
        );
        assert!(out.contains("/// The x axis, display sRGB."), "{out}");
        assert!(
            out.contains("pub const TAPS: [u32; 3] = [1, 2, 4];"),
            "{out}"
        );
        assert!(
            out.contains("pub const LOBES: [[f32; 2]; 2] = [[1.056, 599.8], [-0.065, 501.1]];"),
            "{out}"
        );
    }

    /// The abstract numerics a composite defers to its declaration, converted by the
    /// same rule a scalar's are: `vec3(1, 2, 3)` typed `vec3<f32>` is three `f32`s, not
    /// three integers that happened to be written without a point.
    #[test]
    fn a_composite_const_takes_the_declared_element_type() {
        let out = generated("const STEP: vec3<f32> = vec3(1, 2, 3);\n");
        assert!(
            out.contains("pub const STEP: [f32; 3] = [1.0, 2.0, 3.0];"),
            "{out}"
        );
    }

    #[test]
    fn a_binding_of_each_kind_mirrors_whole() {
        let out = generated(
            r"
// What the pass draws through.
@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var src: texture_2d<f32>;
@group(1) @binding(0) var dst: texture_storage_2d<rgba16float, write>;
@if(resid) @group(1) @binding(1) var dst_resid: texture_storage_2d<rgba16float, write>;

struct View { origin: vec4<f32> }
",
        );
        assert_eq!(
            module_body(&out, "decl"),
            r#"        use super::{BindKind, Binding};
        /// `@group(0) @binding(0) var view` — see [`super::binding::VIEW`].
        pub const VIEW: Binding = Binding {
            group: 0,
            index: 0,
            name: "VIEW",
            kind: BindKind::Uniform { min_size: 16 },
            resid: false,
        };
        /// `@group(0) @binding(1) var samp` — see [`super::binding::SAMP`].
        pub const SAMP: Binding = Binding {
            group: 0,
            index: 1,
            name: "SAMP",
            kind: BindKind::Sampler,
            resid: false,
        };
        /// `@group(0) @binding(2) var src` — see [`super::binding::SRC`].
        pub const SRC: Binding = Binding {
            group: 0,
            index: 2,
            name: "SRC",
            kind: BindKind::Texture {
                dim: wgpu::TextureViewDimension::D2,
            },
            resid: false,
        };
        /// `@group(1) @binding(0) var dst` — see [`super::binding::DST`].
        pub const DST: Binding = Binding {
            group: 1,
            index: 0,
            name: "DST",
            kind: BindKind::Storage {
                dim: wgpu::TextureViewDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
            },
            resid: false,
        };
        /// `@group(1) @binding(1) var dst_resid` — see [`super::binding::DST_RESID`].
        pub const DST_RESID: Binding = Binding {
            group: 1,
            index: 1,
            name: "DST_RESID",
            kind: BindKind::Storage {
                dim: wgpu::TextureViewDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
            },
            resid: true,
        };"#,
        );
        // The index alone does not identify a slot: `SAMP` and `DST_RESID` are both 1,
        // in different groups, and the `binding` module says so without comment. The
        // declaration's own comment is carried over.
        assert_eq!(
            module_body(&out, "binding"),
            "        /// What the pass draws through.\n\
             \x20       pub const VIEW: u32 = 0;\n\
             \x20       pub const SAMP: u32 = 1;\n\
             \x20       pub const SRC: u32 = 2;\n\
             \x20       pub const DST: u32 = 0;\n\
             \x20       pub const DST_RESID: u32 = 1;",
        );
        assert!(
            out.contains("pub const BINDINGS: &[Binding] = &[\n        decl::VIEW,"),
            "{out}"
        );
        // `min_binding_size` is the struct's WGSL size reached by another route, so the
        // two answers are visible in one file.
        assert!(out.contains("/// WGSL size 16, alignment 16."), "{out}");
    }

    #[test]
    fn a_vertex_record_struct_mirrors_as_a_tightly_packed_record() {
        let out = generated(
            r"
struct CornerInstance {
    // Where the corner lands, in canvas px.
    @location(0) at: vec2<f32>,
    // Its half-extent.
    @location(1) half_extent: f32,
    // Which slot it reads.
    @location(2) slot: u32,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    inst: CornerInstance,
) -> @builtin(position) vec4<f32> {
    return vec4<f32>(inst.at, f32(inst.slot) * inst.half_extent, f32(vi));
}
",
        );
        assert_eq!(
            out,
            r#"/// Host mirrors of what `probe.wesl` declares.
pub mod probe {
    /** `CornerInstance`, generated from `probe.wesl`'s `struct CornerInstance` — the record
 `@vertex fn vs_main` takes, and the only declaration of it.*/
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct CornerInstance {
        /// Where the corner lands, in canvas px.
        pub at: [f32; 2],
        /// Its half-extent.
        pub half_extent: f32,
        /// Which slot it reads.
        pub slot: u32,
    }
    impl Default for CornerInstance {
        fn default() -> Self {
            bytemuck::Zeroable::zeroed()
        }
    }
    /// The vertex attributes reading a [`CornerInstance`], in declaration order.
    pub const CORNER_INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 3] = [
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: 0u64,
            shader_location: 0,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32,
            offset: 8u64,
            shader_location: 1,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Uint32,
            offset: 12u64,
            shader_location: 2,
        },
    ];
    /** The buffer layout for a slice of [`CornerInstance`].

`step_mode` is the caller's: nothing in the shader says whether the host means
 to advance this buffer per vertex or per instance. Everything else — the
 stride, the formats, the offsets — comes from the declaration.*/
    pub const fn corner_instance_layout(
        step_mode: wgpu::VertexStepMode,
    ) -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: core::mem::size_of::<CornerInstance>() as u64,
            step_mode,
            attributes: &CORNER_INSTANCE_ATTRIBUTES,
        }
    }
    const _: () = {
        const SIZE: usize = core::mem::size_of::<CornerInstance>();
        assert!(SIZE == 16, "`CornerInstance` is not the 16 bytes its attributes span");
        const OFFSET_OF_AT: usize = core::mem::offset_of!(CornerInstance, at);
        assert!(
            OFFSET_OF_AT == 0,
            "`CornerInstance.at` is not at the tightly-packed offset 0 its vertex attribute reads"
        );
        const OFFSET_OF_HALF_EXTENT: usize = core::mem::offset_of!(
            CornerInstance, half_extent
        );
        assert!(
            OFFSET_OF_HALF_EXTENT == 8,
            "`CornerInstance.half_extent` is not at the tightly-packed offset 8 its vertex attribute reads"
        );
        const OFFSET_OF_SLOT: usize = core::mem::offset_of!(CornerInstance, slot);
        assert!(
            OFFSET_OF_SLOT == 12,
            "`CornerInstance.slot` is not at the tightly-packed offset 12 its vertex attribute reads"
        );
    };
}
"#,
        );
    }

    /// `dynamics.wesl`'s shape: a declaration no context built from the unlinked source
    /// can execute, with constants below it (see [`crate::eval::module_context`]).
    #[test]
    fn a_const_below_a_declaration_that_will_not_execute_still_mirrors() {
        let out = generated(
            r"
var<workgroup> ws: array<Latent, 4>;
const RISE: f32 = 0.05;
const DOUBLE: f32 = RISE * 2.0;
",
        );
        assert!(out.contains("pub const DOUBLE: f32 = 0.1;"), "{out}");
    }

    /// `@size` pads a member out and `@align` moves it; `wgsl-types` honours both when
    /// it sizes a struct, and `lay_out` walking the members did not — so a member
    /// carrying one gave a Rust struct that disagreed with the shader from that member
    /// on. The padding is explicit, and the member keeps its type's own spelling: the
    /// lane the shader reads is the type's, and widening it would claim lanes nothing
    /// writes.
    #[test]
    fn a_member_size_and_align_move_the_lanes_after_it() {
        let out = generated(
            r"
struct Params {
    @size(16) feather: f32,
    @align(32) rect: vec4<f32>,
}

var<uniform> params: Params;
",
        );
        assert!(out.contains("pub feather: f32,"), "{out}");
        assert!(
            out.contains(
                "/// Padding to the `@size(16)` the member declares.\n        pub _pad_1: [u8; 12],"
            ),
            "{out}"
        );
        assert!(
            out.contains("/// Padding to the 32-byte WGSL alignment that follows.\n        pub _pad_2: [u8; 16],"),
            "{out}"
        );
        assert!(out.contains("pub rect: [f32; 4],"), "{out}");
        // 32 for `rect`, 16 of it, rounded up to the struct's 32-byte alignment.
        assert!(
            out.contains("assert!(SIZE == 64, \"`Params` is not 64 bytes\");"),
            "{out}"
        );
        assert!(
            out.contains("assert!(ALIGN == 32, \"`Params` is not 32-byte aligned\");"),
            "{out}"
        );
        assert!(
            out.contains(
                "assert!(OFFSET_OF_RECT == 32, \"`Params.rect` is not at WGSL offset 32\");"
            ),
            "{out}"
        );
    }

    /// And `min_binding_size`, reached by `wgsl-types`' own tables over the resolved
    /// type, says the same 64 — which is the assertion `emit::bindings` now makes for
    /// every uniform struct rather than leaving the two answers uncompared.
    #[test]
    fn the_laid_out_size_and_the_min_binding_size_agree() {
        let out = generated(
            r"
struct Params { @size(16) feather: f32, @align(32) rect: vec4<f32> }

@group(0) @binding(0) var<uniform> params: Params;
",
        );
        assert!(out.contains("BindKind::Uniform { min_size: 64 }"), "{out}");
        assert!(out.contains("/// WGSL size 64, alignment 32."), "{out}");
    }

    /// `@size` may pad a member out; it cannot make one smaller than its type.
    #[test]
    #[should_panic(expected = "`@size(8)` for a 16-byte `vec4<f32>`")]
    fn a_size_smaller_than_the_type_is_refused() {
        generated("struct P { @size(8) a: vec4<f32> }\nvar<uniform> p: P;\n");
    }

    /// WGSL requires a power of two, and `round_up` divides by it.
    #[test]
    #[should_panic(expected = "`@align(12)`, which is not a power of two")]
    fn an_alignment_that_is_not_a_power_of_two_is_refused() {
        generated("struct P { @align(12) a: f32 }\nvar<uniform> p: P;\n");
    }

    /// A mirror is generated from the unlinked source, which has no feature set — so one
    /// Rust struct would have to be the layout of both artifacts and could match at most
    /// one. `matte.wesl` keeps its residual attribute unconditional for this reason;
    /// here it is a build failure rather than prose.
    #[test]
    #[should_panic(expected = "is `@if`-gated")]
    fn an_if_gated_uniform_member_is_refused() {
        generated("struct P { a: vec4<f32>, @if(resid) r: vec4<f32> }\nvar<uniform> p: P;\n");
    }

    /// The same rule on the other half of the boundary.
    #[test]
    #[should_panic(expected = "is an `@if`-gated `@location` member")]
    fn an_if_gated_vertex_member_is_refused() {
        generated(
            "struct CornerInstance {\n\
             \x20   @location(0) at: vec2<f32>,\n\
             \x20   @if(resid) @location(1) resid: vec4<f32>,\n\
             }\n\
             @vertex\nfn vs_main(inst: CornerInstance) -> @builtin(position) vec4<f32> {\n\
             \x20   return vec4<f32>(inst.at, 0.0, 1.0);\n}\n",
        );
    }

    /// A gated *binding* is the case that must keep working: the declaration is whole
    /// either way, and the flag is what the host reads to know the residual build has it.
    #[test]
    fn an_if_gated_binding_is_still_carried_through() {
        let out = generated(
            "@if(resid) @group(0) @binding(0) var dst: texture_storage_2d<rgba16float, write>;\n",
        );
        assert!(out.contains("resid: true"), "{out}");
    }

    /// `lib/` holds the binding-free leaves (§2). A binding there lands in every
    /// artifact that imports the leaf, at a slot no importer chose — and this is the loop
    /// that sees every binding in the tree, so the rule stops being prose.
    #[test]
    #[should_panic(expected = "a module under `lib/` may not declare a binding")]
    fn a_binding_in_a_lib_module_fails_the_build() {
        mirrors(&[Module::parse(
            "lib/store",
            "@group(0) @binding(0) var st: texture_2d<f32>;\n",
        )]);
    }

    /// The rule is the directory at any depth, not the one level the walk used to reach.
    #[test]
    #[should_panic(expected = "a module under `lib/` may not declare a binding")]
    fn a_binding_below_lib_fails_the_build_too() {
        mirrors(&[Module::parse(
            "lib/ramp/store",
            "@group(0) @binding(0) var st: texture_2d<f32>;\n",
        )]);
    }

    /// And what a leaf is *for*: everything but a binding still mirrors.
    #[test]
    fn a_lib_module_mirrors_everything_else() {
        let out = past_header(&mirrors(&[Module::parse(
            "lib/ramp",
            "const STOPS: u32 = 4u;\n",
        )]));
        assert!(out.contains("pub mod ramp {"), "{out}");
        assert!(out.contains("pub const STOPS: u32 = 4;"), "{out}");
    }

    /// A struct reached through an import has no mirror — the generator reads the
    /// unlinked source, where the import is only a name — and the lookup coming back
    /// empty used to mean the uniform was passed over without a word.
    #[test]
    #[should_panic(expected = "names a type the module does not declare")]
    fn a_uniform_whose_struct_is_imported_is_refused() {
        generated("var<uniform> view: View;\n");
    }

    /// The same refusal by the other route: `bindings` asks for the type first, to size
    /// `min_binding_size` from it.
    #[test]
    #[should_panic(expected = "names a type the module does not declare")]
    fn an_imported_uniform_struct_is_refused_at_its_binding_too() {
        generated("@group(0) @binding(0) var<uniform> view: View;\n");
    }

    /// What the silence was *right* about, and the reason it cannot simply become a
    /// panic: a uniform of a predeclared type needs no mirror, the host having the type
    /// already.
    #[test]
    fn a_uniform_of_a_predeclared_type_needs_no_mirror() {
        assert_eq!(generated("var<uniform> origin: vec4<f32>;\n"), "");
    }

    /// What replaced the hand-kept `VERTEX` list: the record's name came from there
    /// because a parameter list has none, so a parameter list is what is refused.
    #[test]
    #[should_panic(expected = "is a `@location` parameter")]
    fn a_bare_location_parameter_list_fails_the_build() {
        generated(
            r"
@vertex
fn vs_main(@location(0) at: vec2<f32>) -> @builtin(position) vec4<f32> {
    return vec4<f32>(at, 0.0, 1.0);
}
",
        );
    }

    /// A record reached through an import has no mirror, for the reason a uniform
    /// struct's does: the generator reads the unlinked source, where the import is only
    /// a name.
    #[test]
    #[should_panic(expected = "which this module declares no `struct` for")]
    fn a_vertex_record_from_another_module_is_refused() {
        generated(
            "@vertex\nfn vs_main(inst: CornerInstance) -> @builtin(position) vec4<f32> {\n\
             \x20   return vec4<f32>(inst.at, 0.0, 1.0);\n}\n",
        );
    }

    /// A gate on the *parameter* decides whether the host has the struct at all, out of
    /// a source with no feature set to evaluate — the member rule one level up, and the
    /// one shape the `VERTEX` list refused that discovery could quietly have skipped
    /// past on its way to `@builtin`.
    #[test]
    #[should_panic(expected = "is the record `vs_main` takes, and it carries an attribute")]
    fn an_if_gated_record_parameter_is_refused() {
        generated(
            "struct CornerInstance { @location(0) at: vec2<f32> }\n\
             @vertex\n\
             fn vs_main(@if(resid) inst: CornerInstance) -> @builtin(position) vec4<f32> {\n\
             \x20   return vec4<f32>(inst.at, 0.0, 1.0);\n}\n",
        );
    }

    /// Two entry points over one instance buffer is a shape nothing forbids, and the
    /// second emission would be a duplicate definition in the generated file.
    #[test]
    fn one_record_two_entry_points_is_generated_once() {
        let out = generated(
            "struct CornerInstance { @location(0) at: vec2<f32> }\n\
             @vertex\n\
             fn vs_main(inst: CornerInstance) -> @builtin(position) vec4<f32> {\n\
             \x20   return vec4<f32>(inst.at, 0.0, 1.0);\n}\n\
             @vertex\n\
             fn vs_flipped(inst: CornerInstance) -> @builtin(position) vec4<f32> {\n\
             \x20   return vec4<f32>(inst.at.yx, 0.0, 1.0);\n}\n",
        );
        assert_eq!(out.matches("pub struct CornerInstance").count(), 1, "{out}");
    }

    /// The common shape — `fill`, `erase`, `integrate` and the filters all draw a
    /// fullscreen triangle off the vertex index alone. There is no record to generate
    /// and no list to join, so the module reaches the file not at all.
    #[test]
    fn a_vertex_entry_with_only_builtins_needs_no_record() {
        let out = generated(
            "@vertex\nfn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {\n\
             \x20   return vec4<f32>(f32(vi));\n}\n",
        );
        assert_eq!(out, "");
    }

    /// And the same builtins bundled into a struct, which is the sibling of the case
    /// above and has to be answered the same way — describing no buffer is not an error
    /// that becomes one by being written down differently.
    #[test]
    fn a_record_struct_of_builtins_alone_needs_no_record() {
        let out = generated(
            "struct VsIn { @builtin(vertex_index) vi: u32 }\n\
             @vertex\nfn vs_main(in: VsIn) -> @builtin(position) vec4<f32> {\n\
             \x20   return vec4<f32>(f32(in.vi));\n}\n",
        );
        assert_eq!(out, "");
    }

    /// A struct discovery reaches and cannot spell does not stop the build — it reaches
    /// declarations no host has ever asked for.
    #[test]
    fn a_struct_with_a_member_that_has_no_rust_spelling_is_skipped_with_a_note() {
        let out = mirrors(&[Module::parse(
            "probe",
            "struct Inner { a: f32 }\nstruct Outer { inner: Inner }\nvar<uniform> o: Outer;\n",
        )]);
        assert_eq!(
            out,
            "// @generated by `stark-shaders-build` from the WESL sources — do not edit.\n\
             //\n\
             // A `vecN` or `array` constant is its own lanes. The same WGSL type as a uniform\n\
             // member is instead the padded stride it occupies there, which can be wider.\n\
             //\n\
             // Declarations discovery reached and could not spell in Rust:\n\
             //   `probe::Outer.inner` is a `Inner`, which has no Rust spelling, so `Outer` is not mirrored\n\
             \n",
        );
    }

    /// The body of `pub mod <name>` inside the generated file, at its own indentation.
    fn module_body<'a>(out: &'a str, name: &str) -> &'a str {
        out.split_once(&format!("mod {name} {{\n"))
            .and_then(|(_, rest)| rest.split_once("\n    }\n"))
            .unwrap_or_else(|| panic!("no `mod {name}` in:\n{out}"))
            .0
    }
}
