//! Compiled WGSL shader sources for Stark, embedded at build time from WESL.
//!
//! Keeping shaders in their own crate (DESIGN.md §2) means the WESL build step
//! never pollutes the engine crate and the same artifacts can be reused by tools.

use wesl::include_wesl;

/// WGSL source for the canvas → surface presentation pass (DESIGN.md §6.4).
pub fn present() -> &'static str {
    include_wesl!("present")
}
