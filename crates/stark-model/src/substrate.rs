//! Which canvas a document is painted on (§6.4) — the id, not the map.

use serde::{Deserialize, Serialize};

use stark_assetid::AssetId;

/// Which physical substrate a document is painted on. Saved in `CanvasMeta` (§8)
/// because which canvas a piece was painted on is part of the document, so it is
/// reproducible.
///
/// **Two variants, and the split is the point.** `Flat` is procedural and needs no
/// bytes; every other substrate *is* its bytes, named by the hash of them. There is no
/// third case — no substrate named by a label whose image the engine would have to be
/// told about separately — because that case is exactly the one that can go missing
/// (§6.4). A peer, a save file or a replay that meets an
/// [`Image`](Self::Image) id it has never seen can always ask for it by content, and
/// verify what comes back; a substrate called "Rough" could only be looked up in a
/// table the asker might not have, and the miss was silent — the tooth read a flat
/// stand-in and baked it into the tiles.
///
/// So this is the same bargain brush shapes already make (§6.6): the id
/// comes *from* the image (`stark-engine`'s `Engine::import_substrate`),
/// which is what makes "built-in" a property of the frontend's asset list and of
/// nothing downstream. The engine still embeds no image bytes.
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
    /// Perfectly smooth: zero height everywhere, so the
    /// constant height has zero gradient (no relief). Paint behaves exactly as if
    /// there were no substrate — the orthogonal default.
    #[default]
    Flat,
    /// A height map, named by the BLAKE3 hash of its canonical decoded form
    /// (`stark-engine`'s `substrate::identify`). Covers the substrates that ship with the app and the ones a
    /// user brings, identically — the engine cannot tell them apart, which is why
    /// neither can go missing in a way the other wouldn't.
    Image(AssetId),
}

/// How large the substrate is laid on the canvas, as a **percentage of its
/// natural size** (§6.4).
///
/// The natural size is one map tile per `SUBSTRATE_TILE_PX` canvas px, which is the
/// engine's; this is the document's say over it. Document state, saved and
/// replicated, because it decides what the tooth bites as surely as *which* substrate
/// does: at 200% a tip crosses half as many threads per px, so it bridges further
/// and rides fewer faces. A stroke replayed from before a change has to be deposited
/// at the scale it was painted at, exactly as it has to be deposited on the substrate it
/// was painted on — so this rides beside [`SubstrateId`] everywhere that one goes.
///
/// # Why a quantized integer and not an `f32`
///
/// Three reasons, and the third is the one that decided it.
///
/// - **It is a key.** The engine bakes a substrate *per scale* — the rise a tip meets
///   over its reach is measured in the map's own texels, so the reach in texels
///   moves when the scale does — and that bake is cached under the pair. An `f32`
///   is neither `Eq` nor `Hash`, and quantizing at the cache would be the same
///   decision made somewhere it could drift from the log.
/// - **It replicates exactly.** Two peers that landed on 1.37 by different
///   arithmetic would bake two substrates and deposit two different marks; `137` is
///   `137` on both.
/// - **It bounds what a document can cost.** Each distinct scale a document names
///   is a substrate texture held for as long as the log can be replayed across it.
///   [`STEP`](Self::STEP) is what keeps a slider dragged from end to end from
///   naming three hundred of them, and 5% is comfortably under the smallest change
///   in a substrate anyone can see.
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

    /// The scale nearest `percent`, held to the ladder and to the range — the one
    /// door, and it cannot fail.
    ///
    /// Rounds to the nearest step rather than truncating, so a slider handed a value
    /// between two rungs lands on the one it is closer to.
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

    /// The constructor is the only door, and `Deserialize` runs it too — so a scale
    /// off the ladder or outside the range cannot arrive from a file or a peer any
    /// more than it can be written by hand. That is the whole reason the field is
    /// private and the wire is a `u16`.
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

    /// Sanitizing is idempotent — the property §8's funnel rests on: a value read
    /// back out of a file has already been through this door, so passing it through
    /// again must not move it.
    #[test]
    fn holding_a_held_scale_leaves_it_alone() {
        for percent in 0..=1000u16 {
            let once = SubstrateScale::new(percent);
            assert_eq!(SubstrateScale::new(once.percent()), once);
        }
    }
}
