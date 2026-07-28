//! Compiled WGSL shader sources for Stark, embedded at build time from WESL.
//!
//! Keeping shaders in their own crate (DESIGN.md §2) means the WESL build step
//! never pollutes the engine crate and the same artifacts can be reused by tools.

use wesl::include_wesl;

/// WGSL stamp pass for the Oklab color space (DESIGN.md §6.2).
pub fn stamp_oklab() -> &'static str {
    include_wesl!("stamp_oklab")
}

/// WGSL source for the tile compositing pass (DESIGN.md §6.3, pass A).
pub fn composite() -> &'static str {
    include_wesl!("composite")
}

/// WGSL matte-layer fill, drawn inside pass A at the matte's place in the layer
/// stack — FRAME_DESIGN.md §4.
pub fn matte() -> &'static str {
    include_wesl!("matte")
}

/// WGSL media/lighting pass for the Oklab color space (DESIGN.md §6.3, pass B).
pub fn media_oklab() -> &'static str {
    include_wesl!("media_oklab")
}

/// WGSL media pass for the Mixbox color space (pigment polynomial) — DESIGN §6.7.
pub fn media_mixbox() -> &'static str {
    include_wesl!("media_mixbox")
}

/// WGSL stroke integrate pass: merge a stroke's scratch slab into the layer over
/// the base — DESIGN §6.2/§6.1.
pub fn integrate() -> &'static str {
    include_wesl!("integrate")
}

/// WGSL compute shader for the brush-dynamics **sequential stamp loop**
/// (snapshot / pickup / deposit entry points) — DESIGN §6.2.
pub fn dynamics() -> &'static str {
    include_wesl!("dynamics")
}

/// WGSL region→tile write-back for the stamp loop — DESIGN §6.2/§6.4.
pub fn slice() -> &'static str {
    include_wesl!("slice")
}

/// WGSL affine-transform passes: the moved parcel, the cut+stack combine, and
/// the carried selection mask — TRANSFORM_DESIGN.md.
pub fn transform() -> &'static str {
    include_wesl!("transform")
}

/// WGSL selection-mask rasterization: one op's shape combined into a mask tile —
/// DESIGN §6.8.
pub fn selection() -> &'static str {
    include_wesl!("selection")
}

/// WGSL selection mask → stroke region gather, for the brush-dynamics loop —
/// DESIGN §6.8/§6.2.
pub fn mask_region() -> &'static str {
    include_wesl!("mask_region")
}

/// WGSL selection outline ("marching ants") drawn over the finished image —
/// DESIGN §6.8.
pub fn overlay() -> &'static str {
    include_wesl!("overlay")
}
