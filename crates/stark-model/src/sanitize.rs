//! The crate's **NaN policy**: the gates every number a log carries comes back
//! through, cited from the `sanitized()` that hold to it.
//!
//! `max`-then-`min` rather than `clamp` is what makes the NaN clause true —
//! `f32::max`/`min` return the non-NaN operand where `clamp` returns the NaN — so
//! clippy's suggestion in [`clamp01`] is the wrong one.

/// `x` into `[0, 1]`, with NaN landing on 0 — the module's rule at its simplest.
pub(crate) const fn clamp01(x: f32) -> f32 {
    x.max(0.0).min(1.0)
}

/// `x` if it is finite, else `fallback` — [`clamp01`]'s companion for a knob with
/// **no upper bound** to clamp to.
///
/// Pass the field's own default rather than zero: `NaN` says nothing about which end
/// was meant, and a radius rounded to 0 is a brush that paints nothing.
pub(crate) fn finite_or(x: f32, fallback: f32) -> f32 {
    if x.is_finite() { x } else { fallback }
}

/// `x` as a non-negative length or rate: finite first, *then* floored at zero.
///
/// A bare `x.max(0.0)` would turn a `NaN` into 0 but pass an infinity through, and it
/// is the infinity a shader notices — an infinite feather reaches `selection.wesl` as
/// a coverage ramp of infinite width, half-strength across the whole plane.
pub(crate) fn at_least_zero(x: f32, fallback: f32) -> f32 {
    finite_or(x, fallback).max(0.0)
}

/// `x` held to `[lo, hi]`, with a non-finite `x` landing on `neutral` —
/// [`finite_or`]'s companion for a knob bounded at **both** ends.
///
/// `neutral` is a third number rather than one of the bounds because `NaN` says
/// nothing about which end was meant: pass the setting that cannot make a picture
/// worse — 0 for an exposure, 1 for a contrast,
/// [`DRAGO_K`](crate::document::DRAGO_K) for a blend's bend.
pub(crate) fn finite_in(x: f32, neutral: f32, (lo, hi): (f32, f32)) -> f32 {
    if x.is_finite() {
        x.clamp(lo, hi)
    } else {
        neutral
    }
}
