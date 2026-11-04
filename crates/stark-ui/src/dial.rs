//! A panel's continuous knob, as a row of a table (§11.2).
//!
//! Three panels have a short, closed set of them — the Lighting shelf's media and
//! display tracks, the Guides bar's two, the Select panel's three — and each is an enum
//! in its own module, because what a dial *reads* and *writes* is that panel's business:
//! a projection and a preference for the lighting, a camera for a guide, the selection
//! state for a mask. What a dial *is* — its seat, its mark, its words, its track and
//! how its number prints — is the same question every time, and [`Dial`] is that
//! question asked once.
//!
//! Three things follow from asking it once:
//!
//! - **A seat is the declaration order**, `self as usize`, so a view holding one state
//!   per dial indexes into an array built from `VARIANTS` and the two cannot disagree.
//!   The three enums each numbered their own seats by hand, and a test had to check
//!   each table of numbers against its own roster.
//! - **The checks are written once** (`tests::check`): the label leads the tip, so a
//!   knob is not called two things; every track is one a hand can move; every dial
//!   sits in its seat. They were three copies of the same three tests.
//! - **A frontend can build every panel's tracks one way** — the native app's widget
//!   layer is generic over this trait rather than written three times.

use std::fmt::Debug;

use crate::icons::Icon;

/// One continuous knob of a panel, described.
///
/// Implemented by a fieldless enum whose `VARIANTS` are the panel's order, so
/// [`index`](Self::index) is `self as usize` in every implementation.
pub trait Dial: Copy + Eq + Debug + strum::VariantArray + 'static {
    /// Which seat this dial has: its place in `VARIANTS`.
    fn index(self) -> usize;
    /// The mark the control wears.
    fn glyph(self) -> Icon;
    /// The one word (or two) a caption gives it.
    fn label(self) -> &'static str;
    /// What it says on hover — the [`label`](Self::label) and then what the knob is
    /// actually *for*, since a caption cannot carry it.
    fn tip(self) -> &'static str;
    /// The ends of its track: the model's own bound, or a ceiling the control owns.
    fn range(self) -> (f32, f32);
    /// The step a track moves in.
    fn step(self) -> f32;
    /// How the number beside the track prints `v`. Two places unless a dial says
    /// otherwise.
    fn readout(self, v: f32) -> String {
        format!("{v:.2}")
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::Dial;

    /// Every check a dial table owes, for `D`'s whole roster: each dial says something
    /// on hover and says it about itself, each track is one a hand can move back off,
    /// and each dial sits in the seat its index names.
    pub(crate) fn check<D: Dial>() {
        for (i, dial) in D::VARIANTS.iter().copied().enumerate() {
            assert!(!dial.label().is_empty(), "{dial:?} has no caption");
            assert!(
                dial.tip().starts_with(dial.label()),
                "{dial:?}'s hover does not lead with its own name: {:?}",
                dial.tip()
            );
            let (lo, hi) = dial.range();
            assert!(lo < hi, "{dial:?} has an empty track: {lo}..={hi}");
            assert!(dial.step() > 0.0, "{dial:?} steps by nothing");
            assert!(
                dial.step() <= hi - lo,
                "{dial:?} steps further than its own track"
            );
            assert_eq!(dial.index(), i, "{dial:?} names a seat it does not sit in");
        }
    }
}
