//! Compiled WGSL shader sources for Stark, embedded at build time from WESL.
//!
//! Keeping shaders in their own crate (§2) means the WESL build step
//! never pollutes the engine crate and the same artifacts can be reused by tools.

use wesl::include_wesl;

/// WGSL stamp pass for the Oklab color space (§6.2).
pub fn stamp_oklab() -> &'static str {
    include_wesl!("stamp_oklab")
}

/// WGSL source for the tile compositing pass (§6.3, pass A).
pub fn composite() -> &'static str {
    include_wesl!("composite")
}

/// WGSL matte-layer fill, drawn inside pass A at the matte's place in the layer
/// stack — §15.4.
pub fn matte() -> &'static str {
    include_wesl!("matte")
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
pub fn blend_mixbox() -> &'static str {
    include_wesl!("blend_mixbox")
}

/// WGSL stroke integrate pass: merge a stroke's scratch slab into the layer over
/// the base — §6.2/§6.1.
pub fn integrate() -> &'static str {
    include_wesl!("integrate")
}

/// WGSL compute shader for the brush-dynamics **sequential stamp loop**
/// (snapshot / pickup / deposit entry points) — §6.2.
pub fn dynamics() -> &'static str {
    include_wesl!("dynamics")
}

/// WGSL region→tile write-back for the stamp loop — §6.2/§6.4.
pub fn slice() -> &'static str {
    include_wesl!("slice")
}

/// WGSL affine-transform passes: the moved parcel, the cut+stack combine, and
/// the carried selection mask — §16.
pub fn transform() -> &'static str {
    include_wesl!("transform")
}

/// WGSL region fill: a parcel of paint laid through a coverage mask, stacked by
/// the shared parcel law — §18.0.4.
pub fn fill() -> &'static str {
    include_wesl!("fill")
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
