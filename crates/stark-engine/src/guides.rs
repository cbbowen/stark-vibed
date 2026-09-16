//! The guide overlay's host-side numbers (§20.4).
//!
//! What the chrome needs in order to *match* what the guide pass draws, taken through
//! the generated shader mirror (§6.10) rather than transcribed — `filters.rs`'s reason,
//! on the other pass.

/// The three world-axis hues the guide pass draws with (§20.3), as **display sRGB**,
/// in X, Y, Z order.
///
/// **Generated from `guides.wesl`'s own declaration**, which is the copy that actually
/// runs. The stylesheet's `--axis-x/y/z` are the statement of them, in Oklab; the
/// shader carries the conversion because it cannot read a stylesheet, and the frontend
/// checks the two agree against *this* rather than against a third copy of the numbers.
pub const AXIS_HUES: [[f32; 3]; 3] = [
    stark_shaders::mirror::guides::AXIS_X,
    stark_shaders::mirror::guides::AXIS_Y,
    stark_shaders::mirror::guides::AXIS_Z,
];
