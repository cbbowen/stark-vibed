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

// One record and one accessor per module declaring an entry point, discovered from the
// tree, and the types that choose between a shader's builds (§6.10). The `on` variant
// of an axis (`Resid`, `Lane`) is generated only where the build linked that variant,
// so nothing here is `cfg`-split and no combination a caller can write is unbuilt.
include!(concat!(env!("OUT_DIR"), "/accessors.rs"));

mod layout;

pub use layout::{Stages, layout_of, layout_shared_by};

/// One entry point of a linked artifact, as `naga` reports it (§6.10).
///
/// The generated record of a shader has a field per entry point, so a pipeline names
/// one at compile time; this is what each of those fields carries.
///
/// **Everything here is read off the linked WGSL**, which is the only place some of it
/// is knowable: half of a pipeline's bindings arrive by import, and which of them an
/// entry point reaches is a fact about the whole call graph rather than about the
/// module that declares them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EntryPoint {
    /// The function's name — what `wgpu` is handed as `entry_point`.
    pub name: &'static str,
    /// The deposited artifact that declares it.
    ///
    /// A pipeline is a module plus entry points, and nothing in `wgpu` ties the two:
    /// the plain stamp module built with the *ceiling* record's entry points would
    /// attach location 3 and never write it — different pixels, no error. This is what
    /// [`Artifact`] is checked against.
    ///
    /// It is also what tells two variants' entry points apart where every other field
    /// agrees: `fs_levels` is not `@if(ceiling)`-gated, so the plain and ceiling builds
    /// declare the same one, and only the artifact says which is which.
    pub artifact: &'static str,
    /// Exactly one stage. A [`ShaderStages`](wgpu::ShaderStages) rather than a
    /// one-of-three enum because that is what a bind group layout's `visibility` wants,
    /// and combining two entry points' is then `|`.
    pub stage: wgpu::ShaderStages,
    /// `@workgroup_size`, the grid a compute dispatch is counted in — `[0, 0, 0]` for
    /// every other stage, which declares none.
    pub workgroup_size: [u32; 3],
    /// The `@location`s a fragment entry point writes, ascending — its color
    /// attachments. Empty for every other stage.
    pub targets: &'static [u32],
    /// The bindings this entry point **actually reaches**, ascending by slot, callees
    /// included.
    pub uses: &'static [Use],
}

impl EntryPoint {
    /// The workgroup counts covering `extent`, at this kernel's own
    /// `@workgroup_size`.
    ///
    /// The host used to divide by a mirrored `const` and trust that the kernel
    /// declared the same one — `TILE_WG`, `BLUR_WG`, the bake's scan width. The
    /// declaration is right here, so the division goes through it (§6.10).
    ///
    /// # Panics
    /// On anything but a compute entry point, which declares `[0, 0, 0]`, and on a
    /// kernel whose `@workgroup_size` has a third dimension: a 2-D extent is not the
    /// question such a kernel is asking, and covering `z` with one group would run its
    /// depth once over.
    pub const fn groups(&self, extent: (u32, u32)) -> (u32, u32) {
        let [x, y, z] = self.workgroup_size;
        assert!(
            x > 0 && y > 0,
            "`groups` asked of an entry point that declares no workgroup size",
        );
        assert!(
            z == 1,
            "`groups` covers a 2-D extent, and this kernel's `@workgroup_size` is 3-D",
        );
        (extent.0.div_ceil(x), extent.1.div_ceil(y))
    }
}

/// One **linked artifact**: its WGSL and the name it was deposited under.
///
/// Every generated record implements it, and it is the one argument
/// `desc::Module` takes — so a shader module and the entry points a pipeline builds
/// over it come from one value rather than two arguments a call site could mix.
pub trait Artifact {
    /// The linked WGSL, for [`wgpu::ShaderSource::Wgsl`].
    fn wgsl(&self) -> &'static str;
    /// The deposited artifact's name — [`EntryPoint::artifact`] for every entry point
    /// of it.
    fn name(&self) -> &'static str;
}

/// One binding an entry point reaches, and how.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Use {
    /// The shader's declaration of the slot.
    pub decl: Binding,
    /// Whether **this** entry point reads it through a sampler — the one thing about a
    /// texture's layout entry the declaration cannot decide (see [`Binding`]).
    ///
    /// The image side of the pair only: a sampler's own entry is a filtering sampler
    /// either way, so the flag would say nothing about it.
    pub sampled: bool,
}

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
    Texture {
        dim: wgpu::TextureViewDimension,
        /// The template argument's scalar. Filterability is **not** part of it, which
        /// is why this is not a `wgpu::TextureSampleType` — see [`Sample`].
        sample: Sample,
    },
    /// `texture_storage_2d<format, access>`.
    Storage {
        dim: wgpu::TextureViewDimension,
        format: wgpu::TextureFormat,
        access: wgpu::StorageTextureAccess,
    },
}

/// The scalar a sampled texture is declared over: the `T` of `texture_2d<T>` (§6.10).
///
/// **Not `wgpu::TextureSampleType`**, which folds filterability into its `Float`
/// variant — and filterability is the host's to say, per entry point, since the same
/// texture is `textureLoad`ed by one and `textureSample`d by the next. [`Self::of`]
/// puts the two together, and is the only place that knows the flag means nothing to
/// an integer texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sample {
    /// `texture_2d<f32>`.
    Float,
    /// `texture_2d<u32>`.
    Uint,
    /// `texture_2d<i32>`.
    Sint,
}

impl Sample {
    /// The `wgpu` sample type, given whether this entry point reads it through a
    /// sampler.
    pub const fn of(self, filterable: bool) -> wgpu::TextureSampleType {
        match self {
            Self::Float => wgpu::TextureSampleType::Float { filterable },
            Self::Uint => wgpu::TextureSampleType::Uint,
            Self::Sint => wgpu::TextureSampleType::Sint,
        }
    }
}

/// One `@binding` declaration, generated from the WESL (§6.10).
///
/// **What is here is what the declaration decides.** What is *not* here is
/// filterability, and its absence is a statement rather than an omission: the same
/// texture is `textureLoad`ed by one entry point of `dynamics.wesl` and
/// `textureSample`d by another (`region_color`, between `snapshot` and `exchange`), so
/// it is a property of the pair and not of the slot. [`Use::sampled`] is where it is
/// reported, per entry point, and the layout folds it over the ones that share one.
///
/// **A host names a slot by taking the whole declaration**, from the generated `decl`
/// module, rather than by looking one up by index. There was a `lookup(table, index)`
/// here, and it was correct only for a module declaring a single group: `@binding(0)`
/// means a different slot in each of a module's groups, and more than half the tree
/// declares two or three (`stamp_common` has `xf` at `@group(0)`, `prefix_tex` at
/// `@group(1)` and `noise_tex` at `@group(2)`, all at index 0). Carrying the
/// declaration instead removes the question rather than answering it — and takes a
/// panicking lookup out of the layout path with it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Binding {
    /// The `@group` this slot is in. A bind group layout is for exactly one group, so
    /// this is what an anchor declaration names it by ([`layout_of`]).
    pub group: u32,
    /// The `@binding` index — unique within [`group`](Self::group), and what a
    /// bind-group entry is keyed on.
    pub index: u32,
    /// The WESL variable's name, uppercased — the same spelling as the `binding::` and
    /// `decl::` constants, so a diagnostic can name the slot the shader names.
    pub name: &'static str,
    /// The `.wesl` that declares it, as the tree holds it (`lib/` and all).
    ///
    /// A slot is identified by its module and its name together, not by either alone:
    /// three modules partition one group between them in half the pipelines here
    /// (`blend_common`, `mixbox_lut`, `blend_mixbox`), and two modules may declare the
    /// same name.
    pub module: &'static str,
    pub kind: BindKind,
    /// Whether the declaration is `@if(resid)`-gated, i.e. exists only in the residual
    /// build of the shader (§6.7).
    ///
    /// No layout reads it: a gated slot is simply absent from the entry points of a
    /// build that does not declare it. It is here because the mirror carries what the
    /// declaration says, and it is what a test names to say a fixture really is gated.
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
