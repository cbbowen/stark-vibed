//! **The input tolerance**: how finely the pointer resolves, in canvas px (§6.2).
//!
//! The one fact about the input nothing downstream can work out for itself, so the
//! frontend states it (`resolution / zoom`) and every stage that prices the input reads
//! it: the fit's two thresholds, the towed tip's emission reach (§6.11), the assist's
//! bars (§6.9), and the hover window's drift gate, extent and run-up reach (§18.1.10).

/// The input tolerance assumed when a caller does not supply one: one canvas px,
/// which is a mouse over a 1:1 view — what the fit's two prices (`KNOT_COST`,
/// `KNOT_SPACING`) were measured against.
pub const DEFAULT_TOLERANCE: f32 = 1.0;

/// Bounds a supplied tolerance is held to.
///
/// Zero (or negative) would make a control point free and every sample worth one.
/// The ceiling is 64 canvas px per input px — past the most zoomed-out view
/// [`ViewTransform`](crate::view::ViewTransform) allows — and is what keeps the
/// window advancing, and so the per-sample work bounded, rather than letting a
/// stroke never grow a control point at all.
///
/// `pub` so the tolerance sweep in `benches/path.rs`, which holds the fit's cost flat
/// *to the top of this range*, can name the bound rather than restate it.
pub const MIN_TOLERANCE: f32 = 1.0 / 64.0;

/// See [`MIN_TOLERANCE`].
pub const MAX_TOLERANCE: f32 = 64.0;

/// A declared input tolerance, held to [`MIN_TOLERANCE`]`..=`[`MAX_TOLERANCE`] —
/// **the** statement of what a tolerance may be.
///
/// Built only by [`new`](Self::new), so every stage that reads one reads the same held
/// number: none can price the input in a tolerance another stage clamped away, and
/// none has a clamp of its own to forget. The session's doors convert the frontend's
/// raw `f32` once.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct InputTolerance(f32);

impl InputTolerance {
    /// [`DEFAULT_TOLERANCE`].
    pub(crate) const DEFAULT: Self = Self::new(DEFAULT_TOLERANCE);

    /// `raw` held to the range, and [`DEFAULT_TOLERANCE`] for a non-finite one: a
    /// number that is not one says nothing in either direction.
    pub(crate) const fn new(raw: f32) -> Self {
        Self(if raw.is_finite() {
            raw.clamp(MIN_TOLERANCE, MAX_TOLERANCE)
        } else {
            DEFAULT_TOLERANCE
        })
    }

    /// The tolerance, in canvas px.
    pub(crate) const fn get(self) -> f32 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A usable tolerance passes through to the bit; the rest land on the nearest
    /// bound, or on the default when there is no nearest.
    #[test]
    fn a_tolerance_is_held_to_its_range() {
        for raw in [MIN_TOLERANCE, 0.25, DEFAULT_TOLERANCE, 3.7, MAX_TOLERANCE] {
            assert_eq!(InputTolerance::new(raw).get().to_bits(), raw.to_bits());
        }
        for (raw, held) in [
            (0.0, MIN_TOLERANCE),
            (-1.0, MIN_TOLERANCE),
            (1e-6, MIN_TOLERANCE),
            (1e9, MAX_TOLERANCE),
            (f32::NAN, DEFAULT_TOLERANCE),
            (f32::INFINITY, DEFAULT_TOLERANCE),
            (f32::NEG_INFINITY, DEFAULT_TOLERANCE),
        ] {
            assert_eq!(InputTolerance::new(raw).get(), held, "tolerance {raw}");
        }
    }
}
