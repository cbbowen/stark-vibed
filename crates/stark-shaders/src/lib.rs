//! Compiled WGSL shader sources for Stark, embedded at build time from WESL.
//!
//! Keeping shaders in their own crate (§2) means the WESL build step
//! never pollutes the engine crate and the same artifacts can be reused by tools.

use wesl::include_wesl;

mod entry_points;
pub use entry_points::{
    CEILING_ENTRY_POINTS, CEILING_FEATURE, ENTRY_POINTS, MIXBOX_ENTRY_POINTS, RESID_ENTRY_POINTS,
    RESID_FEATURE, entry_point_enabled,
};

/// One pass's two builds, chosen by whether the document's color space carries a
/// residual (§6.7).
///
/// Defined twice under a `cfg` rather than written as one `if`, because without the
/// `mixbox` feature the `_resid` artifact **does not exist**: a residual is something
/// a pigment space has, Mixbox is the only one, and `build.rs` skips that whole pass.
/// `include_wesl!` of an artifact that was never deposited is a *compile* error, so
/// the arm has to disappear with the pass rather than merely go untaken.
///
/// The `debug_assert!` in that build is the honest statement of what is left: `resid`
/// can only be `true` if some space returned a `resid_format`, and none can.
#[cfg(feature = "mixbox")]
macro_rules! resid_variant {
    ($flag:expr, $plain:literal, $resid:literal) => {
        if $flag {
            include_wesl!($resid)
        } else {
            include_wesl!($plain)
        }
    };
}

#[cfg(not(feature = "mixbox"))]
macro_rules! resid_variant {
    ($flag:expr, $plain:literal, $resid:literal) => {{
        debug_assert!(
            !$flag,
            concat!(
                "`",
                $resid,
                "` is not built without the `mixbox` feature, and no \
                 color space in this build declares a `resid_format`",
            ),
        );
        include_wesl!($plain)
    }};
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
/// (`build/mirror.rs`).
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

/// WGSL swept-segment stamp pass (§6.2). Color-space agnostic — both spaces use
/// it, since the deposit is the same premultiplied "over" whatever the channels mean.
///
/// Takes `resid` because this pass carries a tile's color, so it is built in two
/// variants (see [`RESID_ENTRY_POINTS`]): with the residual channel a pigment space
/// needs (§6.7), and without it. And `ceiling`, for the variant that also sweeps
/// the ceiling lane (see [`CEILING_ENTRY_POINTS`]) — four artifacts in all.
pub fn stamp(resid: bool, ceiling: bool) -> &'static str {
    if ceiling {
        resid_variant!(resid, "stamp_ceiling", "stamp_resid_ceiling")
    } else {
        resid_variant!(resid, "stamp", "stamp_resid")
    }
}

/// WGSL source for the tile compositing pass (§6.3, pass A).
///
/// Takes `resid` because this pass carries a tile's color, so it is built in two
/// variants (see [`RESID_ENTRY_POINTS`]): with the residual channel a pigment space
/// needs (§6.7), and without it.
pub fn composite(resid: bool) -> &'static str {
    resid_variant!(resid, "composite", "composite_resid")
}

/// WGSL matte-layer fill, drawn inside pass A at the matte's place in the layer
/// stack — §15.4.
///
/// Takes `resid` because this pass carries a tile's color, so it is built in two
/// variants (see [`RESID_ENTRY_POINTS`]): with the residual channel a pigment space
/// needs (§6.7), and without it.
pub fn matte(resid: bool) -> &'static str {
    resid_variant!(resid, "matte", "matte_resid")
}

/// WGSL media/lighting pass for the Oklab color space (§6.3, pass B).
pub fn media_oklab() -> &'static str {
    include_wesl!("media_oklab")
}

/// WGSL presentation resolve: the supersampled render box-averaged down to the
/// target, the last pass of a zoomed-out render — §6.4.
pub fn resolve() -> &'static str {
    include_wesl!("resolve")
}

/// WGSL media pass for the Mixbox color space (pigment polynomial) — §6.7.
///
/// Only in a build carrying the `mixbox` feature: this is one of the two shaders
/// that import the polynomial transpiled from the vendored CC BY-NC 4.0 submodule
/// (see [`MIXBOX_ENTRY_POINTS`]), so without it there is no artifact to embed.
#[cfg(feature = "mixbox")]
pub fn media_mixbox() -> &'static str {
    include_wesl!("media_mixbox")
}

/// WGSL layer-blend pass for the Oklab color space: an isolated layer merged into
/// the accumulator through a light-combining mode — §18.0.4.
pub fn blend_oklab() -> &'static str {
    include_wesl!("blend_oklab")
}

/// WGSL layer-blend pass for the Mixbox color space — §18.0.4. Same
/// light algebra as [`blend_oklab`]; the round trip runs through Mixbox's pigment
/// polynomial and its inverse LUT.
///
/// Only in a build carrying the `mixbox` feature, like [`media_mixbox`].
#[cfg(feature = "mixbox")]
pub fn blend_mixbox() -> &'static str {
    include_wesl!("blend_mixbox")
}

/// WGSL **filter layer** pass for the Oklab color space: the accumulator beneath a
/// filter layer, read and rewritten — §21.
pub fn filter_oklab() -> &'static str {
    include_wesl!("filter_oklab")
}

/// WGSL compute passes for the focal blur's FFT convolution (§21.12): the Stockham
/// transform, the aperture kernel, and the frequency-domain multiply.
///
/// Takes no `resid` and no space: by the time these run, every texel is XYZ light —
/// each space's own `fs_blur_decode` (in its filter shader) paid the decode, and
/// its `fs_main` pays the re-entry.
pub fn blur() -> &'static str {
    include_wesl!("blur")
}

/// WGSL filter-layer pass for the Mixbox color space — §21. Same adjustment as
/// [`filter_oklab`]; the round trip runs through Mixbox's pigment polynomial and its
/// inverse LUT.
///
/// Only in a build carrying the `mixbox` feature, like [`media_mixbox`].
#[cfg(feature = "mixbox")]
pub fn filter_mixbox() -> &'static str {
    include_wesl!("filter_mixbox")
}

/// WGSL stroke integrate pass: merge a stroke's scratch slab into the layer over
/// the base — §6.2/§6.1.
///
/// Takes `resid` because this pass carries a tile's color, so it is built in two
/// variants (see [`RESID_ENTRY_POINTS`]): with the residual channel a pigment space
/// needs (§6.7), and without it.
pub fn integrate(resid: bool) -> &'static str {
    resid_variant!(resid, "integrate", "integrate_resid")
}

/// WGSL compute shader for the brush-dynamics **sequential stamp loop**
/// (snapshot / pickup / deposit entry points) — §6.2.
///
/// Takes `resid` because this pass carries a tile's color, so it is built in two
/// variants (see [`RESID_ENTRY_POINTS`]): with the residual channel a pigment space
/// needs (§6.7), and without it.
pub fn dynamics(resid: bool) -> &'static str {
    resid_variant!(resid, "dynamics", "dynamics_resid")
}

/// WGSL compute shader for the **liquify field** (§6.13): the field's snapshot, the
/// composition of one segment's step into it, and the one resample of a piece
/// through it — over the stamp loop's region machinery and the slot it shares
/// (`dynamics_common.wesl`).
///
/// Takes `resid` because the resample writes a tile's color, residual included, so
/// it is built in two variants (see [`RESID_ENTRY_POINTS`]).
pub fn liquify(resid: bool) -> &'static str {
    resid_variant!(resid, "liquify", "liquify_resid")
}

/// WGSL stroke **erase** pass: the stroke's accumulated extent turned on the
/// layer's visible opacity, inverted through the slab law into a height — §6.12.
///
/// Takes `resid` because this pass rewrites a tile whole, residual included —
/// untouched, since an erase changes no pigment — so it is built in two variants
/// (see [`RESID_ENTRY_POINTS`]).
pub fn erase(resid: bool) -> &'static str {
    resid_variant!(resid, "erase", "erase_resid")
}

/// WGSL region-aux narrowing for the stamp loop's write-back — §6.2/§6.4.
///
/// Takes no `resid`: the write-back's color and residual leave the region as plain
/// texture copies, so the pass under this name touches only the aux, which every
/// color space has and neither varies.
pub fn slice() -> &'static str {
    include_wesl!("slice")
}

/// WGSL affine-transform passes: the moved parcel, the cut+stack combine, and
/// the carried selection mask — §16.
///
/// Takes `resid` because this pass carries a tile's color, so it is built in two
/// variants (see [`RESID_ENTRY_POINTS`]): with the residual channel a pigment space
/// needs (§6.7), and without it.
pub fn transform(resid: bool) -> &'static str {
    resid_variant!(resid, "transform", "transform_resid")
}

/// WGSL region fill: a parcel of paint laid through a coverage mask, stacked by
/// the shared parcel law — §18.0.4.
///
/// Takes `resid` because this pass carries a tile's color, so it is built in two
/// variants (see [`RESID_ENTRY_POINTS`]): with the residual channel a pigment space
/// needs (§6.7), and without it.
pub fn fill(resid: bool) -> &'static str {
    resid_variant!(resid, "fill", "fill_resid")
}

/// WGSL layer merge-down: two stacked layers' tiles reduced to the one tile that
/// composites to the same texels — §14.11.
///
/// Takes `resid` because this pass carries a tile's color, so it is built in two
/// variants (see [`RESID_ENTRY_POINTS`]): with the residual channel a pigment space
/// needs (§6.7), and without it.
pub fn merge(resid: bool) -> &'static str {
    resid_variant!(resid, "merge", "merge_resid")
}

/// WGSL slab-law conversions, both ways: a stored tile expanded into what it
/// composites to, and a composite stored back as a tile — §14.11.
///
/// What lets a blend-mode merge run the *real* blend pass on tile-sized targets
/// instead of a second copy of its algebra. Takes `resid` because both directions
/// carry a tile's color, so it is built in two variants (see [`RESID_ENTRY_POINTS`]).
pub fn slab(resid: bool) -> &'static str {
    resid_variant!(resid, "slab", "slab_resid")
}

/// WGSL selection-mask rasterization: one op's shape combined into a mask tile —
/// §6.8.
pub fn selection() -> &'static str {
    include_wesl!("selection")
}

/// WGSL selection mask → stroke region gather, for the brush-dynamics loop —
/// §6.8/§6.2.
pub fn mask_region() -> &'static str {
    include_wesl!("mask_region")
}

/// WGSL selection outline ("marching ants") drawn over the finished image —
/// §6.8.
pub fn overlay() -> &'static str {
    include_wesl!("overlay")
}

/// WGSL drawing-guides overlay: the perspective grid, drawn over everything —
/// §20.4.
pub fn guides() -> &'static str {
    include_wesl!("guides")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every name in [`ENTRY_POINTS`] has an accessor here, and every accessor has a
    /// name in [`ENTRY_POINTS`].
    ///
    /// The list drives `build.rs`; the accessors are what callers reach for. They used
    /// to be two independent transcriptions of the same seventeen names, so adding a
    /// shader to one and not the other failed at the wrong layer — a missing
    /// `include_wesl!` artifact reports as a build-script problem several frames from
    /// the shader you just wrote, and an accessor with no `build_artifact` behind it
    /// does not fail until link time.
    ///
    /// Now one of them is generated from the other's evidence: an accessor that is not
    /// in the list, or a list entry with no accessor, is a failed assertion naming the
    /// offender.
    /// One accessor, paired with the [`ENTRY_POINTS`] name it must correspond to.
    type Accessor = (&'static str, fn() -> &'static str);

    /// One flagged accessor, paired with its [`RESID_ENTRY_POINTS`] name — the same
    /// correspondence [`Accessor`] states for the plain builds, for the eight passes
    /// that take a `resid` bool. With the test that uses it: without `mixbox` there
    /// are no residual variants to pair up.
    #[cfg(feature = "mixbox")]
    type ResidAccessor = (&'static str, fn(bool) -> &'static str);

    #[test]
    fn the_entry_point_list_and_the_accessors_are_the_same_set() {
        let accessors: &[Accessor] = &[
            #[cfg(feature = "mixbox")]
            ("blend_mixbox", blend_mixbox),
            ("blend_oklab", blend_oklab),
            ("blur", blur),
            ("composite", || composite(false)),
            ("dynamics", || dynamics(false)),
            ("erase", || erase(false)),
            ("fill", || fill(false)),
            #[cfg(feature = "mixbox")]
            ("filter_mixbox", filter_mixbox),
            ("filter_oklab", filter_oklab),
            ("guides", guides),
            ("integrate", || integrate(false)),
            ("liquify", || liquify(false)),
            ("mask_region", mask_region),
            ("matte", || matte(false)),
            #[cfg(feature = "mixbox")]
            ("media_mixbox", media_mixbox),
            ("media_oklab", media_oklab),
            ("merge", || merge(false)),
            ("overlay", overlay),
            ("resolve", resolve),
            ("selection", selection),
            ("slab", || slab(false)),
            ("slice", slice),
            ("stamp", || stamp(false, false)),
            ("transform", || transform(false)),
        ];

        let mut names: Vec<&str> = accessors.iter().map(|(n, _)| *n).collect();
        names.sort_unstable();
        // Filtered by the same predicate `build.rs` filters by, so this checks the
        // *third* way the set can be stated as well: an accessor that survived the
        // cargo feature while its artifact did not — or the reverse — is a link error
        // this catches by name instead.
        let mut listed: Vec<&str> = ENTRY_POINTS
            .iter()
            .copied()
            .filter(|n| entry_point_enabled(n, cfg!(feature = "mixbox")))
            .collect();
        listed.sort_unstable();
        assert_eq!(
            names, listed,
            "`ENTRY_POINTS` (which `build.rs` compiles) and the accessors in this \
             module have diverged",
        );

        // And each one actually links to something, which is what proves the list is
        // the *build's* list and not just a matching pair of transcriptions.
        for (name, accessor) in accessors {
            assert!(
                !accessor().is_empty(),
                "`{name}` linked to an empty artifact",
            );
        }
    }

    /// Every module in [`RESID_ENTRY_POINTS`] is also in [`ENTRY_POINTS`], and its
    /// residual variant links to something *different* from its plain one.
    ///
    /// The second half is what makes this worth a test. `build.rs` sets the feature
    /// and then calls `build_artifact` twice; if the flag failed to take — a renamed
    /// feature, a `@if` spelt wrong — both passes would deposit the same WGSL, every
    /// pipeline would build, and a Mixbox document would simply go on dropping its
    /// residual with no target to write it to.
    ///
    /// Only meaningful with the `mixbox` feature: without it `build.rs` skips the
    /// residual pass outright, so there is no second artifact to differ — which the
    /// set check above already states, since neither list carries one.
    #[cfg(feature = "mixbox")]
    #[test]
    fn every_residual_variant_differs_from_its_plain_build() {
        let resid: &[ResidAccessor] = &[
            ("composite", composite),
            ("dynamics", dynamics),
            ("erase", erase),
            ("fill", fill),
            ("integrate", integrate),
            ("liquify", liquify),
            ("matte", matte),
            ("merge", merge),
            ("slab", slab),
            ("stamp", |r| stamp(r, false)),
            ("transform", transform),
        ];

        let mut names: Vec<&str> = resid.iter().map(|(n, _)| *n).collect();
        names.sort_unstable();
        let mut listed = RESID_ENTRY_POINTS.to_vec();
        listed.sort_unstable();
        assert_eq!(
            names, listed,
            "`RESID_ENTRY_POINTS` and the flagged accessors have diverged"
        );

        for (name, accessor) in resid {
            assert!(
                ENTRY_POINTS.contains(name),
                "`{name}` has a residual variant but no plain build to vary from",
            );
            assert_ne!(
                accessor(true),
                accessor(false),
                "`{name}_resid` is byte-identical to `{name}` — the `{RESID_FEATURE}`                  feature did not reach it",
            );
        }
    }

    /// The ceiling-lane sweep is a real second artifact, in every residual
    /// configuration this build has — the residual test's argument, for the
    /// other feature: a `@if(ceiling)` spelt wrong would deposit the plain sweep
    /// twice, every pipeline would build, and a pen-driven ceiling would draw as
    /// the dial with no lane to say otherwise.
    #[test]
    fn every_ceiling_variant_differs_from_its_plain_build() {
        for name in CEILING_ENTRY_POINTS {
            assert!(
                ENTRY_POINTS.contains(name),
                "`{name}` has a ceiling variant but no plain build to vary from",
            );
        }
        assert_ne!(
            stamp(false, true),
            stamp(false, false),
            "`stamp_ceiling` is byte-identical to `stamp` — the `{CEILING_FEATURE}` feature did not reach it",
        );
        #[cfg(feature = "mixbox")]
        {
            assert_ne!(stamp(true, true), stamp(true, false));
            assert_ne!(stamp(true, true), stamp(false, true));
        }
    }
}
