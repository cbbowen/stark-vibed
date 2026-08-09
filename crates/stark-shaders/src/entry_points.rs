// The WESL modules that become standalone WGSL artifacts — the single list.
//
// `include!`d by **both** `build.rs` (which compiles each one) and `lib.rs` (which
// embeds each one), because the two used to carry the list separately with nothing
// checking that they agreed. Adding a shader to one and not the other failed at the
// wrong layer: a missing `include_wesl!` artifact is a build-script-ordering error
// several frames from the shader you just wrote.
//
// Everything *not* in this list is a module reached only by import — the binding-free
// leaves under `shaders/lib/`, and the binding-owning shared modules
// (`blend_common`, `media_common`, `stamp_common`, `mixbox_lut`). Those have no entry
// point of their own and would fail to link as a root.

/// Every WESL module compiled to its own WGSL artifact, by module name.
///
/// Kept sorted, which is not cosmetic: the pipeline this list drives is a `for` loop,
/// so the order decides the order build errors surface in, and an alphabetical list is
/// the one a reader can check for a missing entry at a glance.
pub const ENTRY_POINTS: &[&str] = &[
    "blend_mixbox",
    "blend_oklab",
    "composite",
    "dynamics",
    "fill",
    "guides",
    "integrate",
    "mask_region",
    "matte",
    "media_mixbox",
    "media_oklab",
    "overlay",
    "resolve",
    "selection",
    "slice",
    "stamp",
    "transform",
];

/// The WESL conditional-compilation feature that turns on a tile's **residual**
/// channel — the third colour texture a pigment space needs (§6.7).
pub const RESID_FEATURE: &str = "resid";

/// The subset of [`ENTRY_POINTS`] that also gets built a *second* time with
/// [`RESID_FEATURE`] enabled, deposited as `<module>_resid`.
///
/// These are the passes that carry a tile's colour, and a residual goes wherever a
/// latent goes — it is the same premultiplied "over" on the same coverage, so every
/// one of them does to `resid` exactly what it already does to `color` (§6.7).
///
/// A **variant** rather than one shader that always carries the channel, because
/// Oklab has no residual to carry: its three channels reproduce every sRGB colour
/// exactly, so a third target there would be eight bytes per texel of zeroes written
/// on the default space's hot path. `@if(resid)` is what keeps the two laws in one
/// file instead of a `*_resid.wesl` beside each of these.
///
/// `blend_mixbox` and `media_mixbox` are **not** here and need no feature: they are
/// reached only by the space that has a residual, so they declare the extra binding
/// unconditionally — the trick `mixbox_lut.wesl` already plays on the blend group.
///
/// Kept sorted, like [`ENTRY_POINTS`] and for the same reason.
pub const RESID_ENTRY_POINTS: &[&str] = &[
    "composite",
    "dynamics",
    "fill",
    "integrate",
    "matte",
    "slice",
    "stamp",
    "transform",
];
