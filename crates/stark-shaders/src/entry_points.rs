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
