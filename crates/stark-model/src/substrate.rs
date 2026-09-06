//! Which canvas a document is painted on (§6.4) — the id, not the map.

use serde::{Deserialize, Serialize};

use stark_assetid::AssetId;

/// Which physical substrate a document is painted on. Saved in `CanvasMeta` (§8), so
/// a piece replays on the canvas it was painted on.
///
/// **Two variants, and there is deliberately no third.** `Flat` is procedural and
/// needs no bytes; every other substrate *is* its bytes, named by the hash of them, so
/// a peer or a replay meeting an id it has never seen can ask for it by content and
/// verify what comes back. A substrate named by a label instead could only be looked
/// up in a table the asker might not have, and the miss would be silent — the tooth
/// reads a flat stand-in and bakes it into the tiles (§6.4). Brush shapes make the
/// same bargain (§6.6), which leaves "built-in" a property of the frontend's asset
/// list and of nothing downstream.
#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    Default,
    carbonite::Schema,
)]
pub enum SubstrateId {
    /// Perfectly smooth: zero height everywhere, hence no relief. Paint behaves
    /// exactly as if there were no substrate.
    #[default]
    Flat,
    /// A height map, named by the BLAKE3 hash of its canonical decoded form
    /// (`stark-engine`'s `substrate::identify`). Shipped and user-brought substrates
    /// are indistinguishable here.
    Image(AssetId),
}

/// How large the substrate is laid on the canvas, as a **percentage of its natural
/// size** — one map tile per `SUBSTRATE_TILE_PX` canvas px (§6.4).
///
/// Document state, saved and replicated, because it decides what the tooth bites as
/// surely as *which* substrate does: at 200% a tip crosses half as many threads per
/// px. So it rides beside [`SubstrateId`] everywhere that one goes.
///
/// # Why a quantized integer and not an `f32`
///
/// - **It is a key.** The engine caches a substrate bake per (id, scale) pair, and an
///   `f32` is neither `Eq` nor `Hash`.
/// - **It replicates exactly.** Two peers that landed on 1.37 by different arithmetic
///   would deposit two different marks; `137` is `137` on both.
/// - **It bounds what a document can cost.** Each distinct scale is a substrate
///   texture held for as long as the log can be replayed. [`STEP`](Self::STEP) keeps a
///   dragged slider from naming three hundred of them, and 5% is under the smallest
///   change anyone can see.
#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    carbonite::Schema,
)]
#[serde(from = "u16", into = "u16")]
#[carbonite(as = "u16")]
pub struct SubstrateScale(u16);

impl SubstrateScale {
    /// The substrate at the size the map was authored at — one tile per
    /// `SUBSTRATE_TILE_PX` canvas px.
    pub const NATURAL: Self = Self(100);
    /// The finest substrate offered: a quarter size, four tiles to the natural one.
    pub const MIN: u16 = 25;
    /// The coarsest: four times natural, past which a 2048-texel map is under one
    /// texel per canvas px and the grain is a blur rather than a tooth.
    pub const MAX: u16 = 400;
    /// The lattice every scale lands on. See the type's note for why there is one.
    pub const STEP: u16 = 5;

    /// The scale nearest `percent`, held to the [`STEP`](Self::STEP) ladder and to
    /// `[MIN, MAX]`. The one door, it cannot fail, and `Deserialize` runs it too.
    pub const fn new(percent: u16) -> Self {
        let clamped = if percent < Self::MIN {
            Self::MIN
        } else if percent > Self::MAX {
            Self::MAX
        } else {
            percent
        };
        Self((clamped + Self::STEP / 2) / Self::STEP * Self::STEP)
    }

    /// The scale as a percentage — what the slider shows and what the wire carries.
    pub const fn percent(self) -> u16 {
        self.0
    }

    /// The multiplier the renderer wants: `1.0` at natural size.
    pub fn factor(self) -> f32 {
        self.0 as f32 / 100.0
    }
}

impl Default for SubstrateScale {
    fn default() -> Self {
        Self::NATURAL
    }
}

impl From<u16> for SubstrateScale {
    fn from(percent: u16) -> Self {
        Self::new(percent)
    }
}

impl From<SubstrateScale> for u16 {
    fn from(scale: SubstrateScale) -> Self {
        scale.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scale off the ladder or outside the range cannot arrive from a file or a
    /// peer, since `Deserialize` runs the constructor.
    #[test]
    fn every_scale_lands_on_the_ladder_inside_the_range() {
        for percent in 0..=1000u16 {
            let scale = SubstrateScale::new(percent);
            assert_eq!(
                scale.percent() % SubstrateScale::STEP,
                0,
                "{percent} is off the ladder"
            );
            assert!(
                (SubstrateScale::MIN..=SubstrateScale::MAX).contains(&scale.percent()),
                "{percent} escaped the range as {}",
                scale.percent()
            );
        }
    }

    /// Rounding, not truncation: a value between two rungs takes the nearer one, so a
    /// slider dragged to 138 shows 140 rather than backing up to 135.
    #[test]
    fn a_scale_between_two_rungs_takes_the_nearer_one() {
        assert_eq!(SubstrateScale::new(138).percent(), 140);
        assert_eq!(SubstrateScale::new(137).percent(), 135);
        assert_eq!(SubstrateScale::NATURAL.percent(), 100);
        assert_eq!(SubstrateScale::NATURAL.factor(), 1.0);
    }

    /// Sanitizing is idempotent — the property §8's funnel rests on, since a value
    /// read out of a file has already been through this door once.
    #[test]
    fn holding_a_held_scale_leaves_it_alone() {
        for percent in 0..=1000u16 {
            let once = SubstrateScale::new(percent);
            assert_eq!(SubstrateScale::new(once.percent()), once);
        }
    }
}
