//! Compiled WGSL shader sources for Stark, embedded at build time from WESL.
//!
//! Keeping shaders in their own crate (§2) means the WESL build step
//! never pollutes the engine crate and the same artifacts can be reused by tools.

use wesl::include_wesl;

mod entry_points;
pub use entry_points::ENTRY_POINTS;

/// WGSL swept-segment stamp pass (§6.2). Colour-space agnostic — both spaces use
/// it, since the deposit is the same premultiplied "over" whatever the channels mean.
pub fn stamp() -> &'static str {
    include_wesl!("stamp")
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

    #[test]
    fn the_entry_point_list_and_the_accessors_are_the_same_set() {
        let accessors: &[Accessor] = &[
            ("blend_mixbox", blend_mixbox),
            ("blend_oklab", blend_oklab),
            ("composite", composite),
            ("dynamics", dynamics),
            ("fill", fill),
            ("guides", guides),
            ("integrate", integrate),
            ("mask_region", mask_region),
            ("matte", matte),
            ("media_mixbox", media_mixbox),
            ("media_oklab", media_oklab),
            ("overlay", overlay),
            ("resolve", resolve),
            ("selection", selection),
            ("slice", slice),
            ("stamp", stamp),
            ("transform", transform),
        ];

        let mut names: Vec<&str> = accessors.iter().map(|(n, _)| *n).collect();
        names.sort_unstable();
        let mut listed = ENTRY_POINTS.to_vec();
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
}
