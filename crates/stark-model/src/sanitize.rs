//! The crate's **NaN policy**: the gates every number a log carries comes back
//! through, cited from the `sanitized()` that hold to it.
//!
//! `max`-then-`min` rather than `clamp`, which is what makes the NaN clause true:
//! `f32::max`/`min` return the non-NaN operand where `clamp` returns the NaN. That is
//! why clippy's suggestion in [`clamp01`] is the wrong one.

/// `x` into `[0, 1]`, with NaN landing on 0 — the module's rule at its simplest.
pub(crate) const fn clamp01(x: f32) -> f32 {
    x.max(0.0).min(1.0)
}

/// `x` if it is a number this parameter can be, else `fallback` — [`clamp01`]'s
/// companion for a knob with **no upper bound** to clamp to.
///
/// Falling back to the field's own default rather than to zero: `NaN` says nothing
/// about which end was meant, and a radius silently rounded to 0 is a brush that
/// paints nothing, which is a worse answer than the one the slider ships at.
pub(crate) fn finite_or(x: f32, fallback: f32) -> f32 {
    if x.is_finite() { x } else { fallback }
}

/// `x` as a non-negative length or rate: finite, and floored at zero.
///
/// **Finite first, then floored**, and that order is the whole of it. A bare
/// `x.max(0.0)` turns a `NaN` into 0 but passes an infinity through — and the
/// infinity is the half a shader notices: an infinite feather reaches
/// `selection.wesl` as a coverage ramp of infinite width, where `0.5 - sd/w` is
/// `0.5` at every texel that is not itself infinitely far away. A selection nobody
/// asked for, drawn at half strength across the plane.
pub(crate) fn at_least_zero(x: f32, fallback: f32) -> f32 {
    finite_or(x, fallback).max(0.0)
}

/// `x` held to `[lo, hi]`, with a non-finite `x` landing on `neutral` —
/// [`finite_or`]'s companion for a knob that has a range at **both** ends.
///
/// `is_finite` **first**, then `clamp`, for [`clamp01`]'s reason: a bare `clamp`
/// returns the `NaN` it exists to catch.
///
/// It takes three numbers rather than two because `NaN` says nothing about which end
/// was meant, so the fallback is the setting that cannot make a picture worse — 0
/// for an exposure, 1 for a contrast, [`DRAGO_K`](crate::document::DRAGO_K) for a
/// blend's bend. A bound would have to pick one end and be wrong half the time.
pub(crate) fn finite_in(x: f32, neutral: f32, (lo, hi): (f32, f32)) -> f32 {
    if x.is_finite() {
        x.clamp(lo, hi)
    } else {
        neutral
    }
}
