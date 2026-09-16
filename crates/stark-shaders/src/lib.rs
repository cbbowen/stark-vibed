//! Compiled WGSL shader sources for Stark, embedded at build time from WESL.
//!
//! Keeping shaders in their own crate (§2) means the WESL build step
//! never pollutes the engine crate and the same artifacts can be reused by tools.

/// The WGSL `build.rs` deposited for one entry point, embedded.
///
/// `wesl`'s own `include_wesl!` is this line, and taking it costs the engine — and
/// every wasm build of it — a compile of the whole WESL compiler for a macro that
/// names a file.
macro_rules! include_wesl {
    ($artifact:literal) => {
        include_str!(concat!(env!("OUT_DIR"), "/", $artifact, ".wgsl"))
    };
}

// One accessor per module declaring an entry point, discovered from the tree, and the
// types that choose between a shader's builds (§6.10). The `on` variant of an axis
// (`Resid`, `Lane`) is generated only where the build linked that variant, so nothing
// here is `cfg`-split and no combination a caller can write is unbuilt.
include!(concat!(env!("OUT_DIR"), "/accessors.rs"));

/// What one `@binding` declaration **is**, as the WESL says it (§6.10).
///
/// The host's bind-group layout has to name a `wgpu::BindingType` for every slot, and
/// that type is almost entirely decided by the shader: `texture_storage_2d<rgba32float,
/// write>` is a storage texture of that exact format, `sampler` is a sampler,
/// `var<uniform>` is a uniform buffer of the declared struct's size. All of that used
/// to be transcribed by hand into `desc::` calls — `stor` versus `stor32` chosen per
/// entry, the uniform's `min_binding_size` written out again — with nothing checking
/// either.
///
/// The `wgpu` types are carried directly rather than their WGSL spellings. This crate
/// depends on `wgpu` anyway (the generated vertex buffer layouts are `wgpu`'s), and a
/// `&'static str` here bought only a pair of string matches on the host, each with a
/// runtime panic for a fact the generator knew at build time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindKind {
    /// `var<uniform> x: T` — carries `T`'s WGSL size, which is the layout's
    /// `min_binding_size`.
    Uniform { min_size: u64 },
    /// `sampler`. Whether it is *filtering* is a property of how an entry point uses
    /// it rather than of the declaration, so it is not here — see [`Binding`].
    Sampler,
    /// `texture_2d<f32>` and friends.
    Texture { dim: wgpu::TextureViewDimension },
    /// `texture_storage_2d<format, access>`. The access mode is not here: it is
    /// implied by the layout entry the host builds.
    Storage {
        dim: wgpu::TextureViewDimension,
        format: wgpu::TextureFormat,
    },
}

/// One `@binding` declaration, generated from the WESL (§6.10).
///
/// **What is here is what the declaration decides.** What is *not* here is
/// filterability, and its absence is a statement rather than an omission: the same
/// texture is `textureLoad`ed by one entry point of `dynamics.wesl` and
/// `textureSample`d by another (`region_color`, between `snapshot` and `exchange`), so
/// it is a property of the pair and not of the slot. The host says it, once, in the
/// list that names the entry point's bindings.
///
/// **A host names a slot by taking the whole declaration**, from the generated `decl`
/// module, rather than by looking one up by index. There was a `lookup(table, index)`
/// here, and it was correct only for a module declaring a single group: `@binding(0)`
/// means a different slot in each of a module's groups, and more than half the tree
/// declares two or three (`stamp_common` has `xf` at `@group(0)`, `prefix_tex` at
/// `@group(1)` and `noise_tex` at `@group(2)`, all at index 0). Carrying the
/// declaration instead removes the question rather than answering it — and takes a
/// panicking lookup out of the layout path with it.
#[derive(Clone, Copy, Debug)]
pub struct Binding {
    /// The `@group` this slot is in. A bind group layout is for exactly one group, so
    /// this is what lets the host check that a slot list names one.
    pub group: u32,
    /// The `@binding` index — unique within [`group`](Self::group), and what a
    /// bind-group entry is keyed on.
    pub index: u32,
    /// The WESL variable's name, uppercased — the same spelling as the `binding::` and
    /// `decl::` constants, so a diagnostic can name the slot the shader names.
    pub name: &'static str,
    pub kind: BindKind,
    /// Whether the declaration is `@if(resid)`-gated, i.e. exists only in the
    /// residual build of the shader (§6.7).
    ///
    /// This is what retired the `[..12 + 4 * usize::from(resid)]` slices: a layout
    /// listed its residual entries at the end of an array and then counted them by
    /// hand, per entry point, seven times over. The gate is in the declaration; now it
    /// is in the table.
    pub resid: bool,
}

impl Binding {
    /// The format this storage slot is declared at — what a texture bound here must
    /// be created as.
    ///
    /// The point of asking rather than writing it down: a `texture_storage_2d<f, _>`
    /// names its own format, so a host constant repeating it is the second copy §6.10
    /// exists to forbid. `create_texture` would reject a mismatch loudly, which makes
    /// this the cheap kind of drift — but only after somebody found the six literals.
    ///
    /// # Panics
    /// If this declaration is not a storage texture. That is a mis-*named* slot rather
    /// than a mismatched value, so it cannot be a runtime condition worth handling:
    /// the caller wrote the wrong constant, and the constants are generated.
    pub const fn storage_format(&self) -> wgpu::TextureFormat {
        match self.kind {
            BindKind::Storage { format, .. } => format,
            _ => panic!("`storage_format` asked of a slot that is not a storage texture"),
        }
    }
}

/// Rust mirrors of what the WESL declares — the uniform structs the host fills in,
/// the constants both sides compute with, the `@binding` declarations, and the
/// per-instance vertex records — generated from the shader sources at build time
/// (`stark-shaders-build`).
///
/// The shader decides how the lanes are read, so the shader's declaration is the
/// only one: these are not transcriptions to be kept in step, and the lane
/// documentation on each field is the WESL comment itself.
///
/// **Everything the tree declares is here, not a chosen subset.** The generator
/// discovers rather than being given a list, so a new uniform, constant or binding
/// arrives mirrored — which means some of what follows has no caller. That is the
/// trade and it is the right way round: an unused mirror costs a few lines of
/// generated code, while a *missing* one costs a hand-written second declaration that
/// nothing checks. `mirror.rs`'s header names anything discovery could not spell.
///
/// The lint waiver is what "generated" means here rather than a suppression: WESL
/// writes `const PI: f32 = 3.14159265359`, and `approx_constant` is advice to an
/// author about a literal this module has no author for.
#[expect(
    clippy::approx_constant,
    reason = "the literal is WESL's, in generated code with no author to advise"
)]
pub mod mirror {
    include!(concat!(env!("OUT_DIR"), "/mirror.rs"));
}
