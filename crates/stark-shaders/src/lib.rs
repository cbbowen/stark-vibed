//! Compiled WGSL shader sources for Stark, embedded at build time from WESL.
//!
//! Keeping shaders in their own crate (DESIGN.md §2) means the WESL build step
//! never pollutes the engine crate and the same artifacts can be reused by tools.

use wesl::include_wesl;

/// WGSL source for the canvas → surface presentation pass (DESIGN.md §6.4).
pub fn present() -> &'static str {
    include_wesl!("present")
}

/// WGSL stamp pass for the Oklab color space (DESIGN.md §6.2).
pub fn stamp_oklab() -> &'static str {
    include_wesl!("stamp_oklab")
}

/// WGSL source for the tile compositing pass (DESIGN.md §6.3, pass A).
pub fn composite() -> &'static str {
    include_wesl!("composite")
}

/// WGSL media/lighting pass for the Oklab color space (DESIGN.md §6.3, pass B).
pub fn media_oklab() -> &'static str {
    include_wesl!("media_oklab")
}

/// WGSL media pass for the Mixbox color space (pigment polynomial) — DESIGN §6.7.
pub fn media_mixbox() -> &'static str {
    include_wesl!("media_mixbox")
}

/// WGSL compute pass for wet-mixing pickup (the reservoir scan) — DESIGN §6.2.
pub fn mixer() -> &'static str {
    include_wesl!("mixer")
}
