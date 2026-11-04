//! The crate's **NaN policy**: the gates every number a log carries comes back
//! through, cited from the `sanitized()` that hold to it.
//!
//! `max`-then-`min` rather than `clamp` is what makes the NaN clause true —
//! `f32::max`/`min` return the non-NaN operand where `clamp` returns the NaN — so
//! clippy's suggestion in [`clamp01`] is the wrong one.
//!
//! The gates a **session** setter wants are `pub`: a number or a name arriving from
//! outside is held to the same range on its way into session state as it would be on
//! its way into the log, and one rule spelled twice is two rules one edit apart.

/// `x` into `[0, 1]`, with NaN landing on 0 — the module's rule at its simplest.
pub const fn clamp01(x: f32) -> f32 {
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

/// `x` into `[0, 1]`, with a non-finite `x` landing on `fallback` — [`clamp01`] for a
/// knob whose `NaN` should mean the field's own default rather than the near end.
pub(crate) fn unit_or(x: f32, fallback: f32) -> f32 {
    clamp01(finite_or(x, fallback))
}

/// `x` as a non-negative length or rate: finite first, *then* floored at zero.
///
/// A bare `x.max(0.0)` would turn a `NaN` into 0 but pass an infinity through, and it
/// is the infinity a shader notices — an infinite feather reaches `selection.wesl` as
/// a coverage ramp of infinite width, half-strength across the whole plane.
pub fn at_least_zero(x: f32, fallback: f32) -> f32 {
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

/// A name that travels, held to [`MAX_NAME`](crate::MAX_NAME) `char`s, with an empty
/// one made *no* name — the one funnel for every name the log or the presence wire
/// carries: a layer's, a guide's, a placed picture's and a peer's display name.
///
/// Cut in place and by `char`, so the cut cannot land inside one and a name that was
/// already short keeps the buffer it arrived in. Empty is `None` because `None` is
/// what each of those fields already spells for "unnamed", and a `Some("")` would be a
/// second spelling of it that a reader has to know to treat alike.
///
/// Idempotent, like everything here. Surrounding whitespace is left alone: trimming is
/// what a *frontend's* text field owes its user, not a bound a stranger is held to.
pub fn name(name: Option<String>) -> Option<String> {
    name.map(|mut name| {
        if let Some((cut, _)) = name.char_indices().nth(crate::MAX_NAME) {
            name.truncate(cut);
        }
        name
    })
    .filter(|name| !name.is_empty())
}
