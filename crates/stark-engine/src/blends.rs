//! The blend pass's host-side numbers (§6.3).
//!
//! What the chrome needs in order to *say* what the blend pass does, taken through the
//! generated shader mirror (§6.10) rather than transcribed — `filters.rs`'s reason, on
//! the other pass.

/// `BlendMode::Illuminate`'s gray card, as **linear light** — the layer color that
/// leaves the backdrop alone, which is what a frontend has to be able to name for a
/// painter (§6.3). A neutral, so it is the same number on every channel of linear sRGB;
/// `stark_model::color::linear_to_srgb` gives the code a picker shows.
///
/// **Generated from `blend_common.wesl`'s own declaration**, which is the copy that
/// actually runs.
pub const ILLUMINATE_GRAY: f32 = stark_shaders::mirror::blend_common::ILLUMINATE_GRAY;
