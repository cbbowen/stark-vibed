//! What a document's pixels depend on besides its log (§6.6, §6.4).
//!
//! A stroke names the shape it stamps with, a `SetSubstrate` the substrate it moves
//! onto, a `PlaceImage` the picture it lands — all three by content id, with the bytes
//! outside the log. So anything replaying a log must first ask "what does this need,
//! and have I got it?", whether it is loading a file, joining a session or receiving a
//! peer's action. This module is the one answer for all three.
//!
//! The three kinds are one hash and travel one way, parting only at decode — a brush
//! mask as luminance × alpha, a substrate as channel 0, a picture as all four channels
//! kept — so a receiver has to be *told* which it is being handed, and the action that
//! referenced it is what knows.

use crate::AssetId;
use crate::SubstrateId;
use crate::document::{Action, ActionKind, BrushShape};
use crate::io::DocumentFile;
use crate::peer::{GestureFrame, PeerFrame};

/// Content a document needs before it can be replayed faithfully, and which store
/// it belongs in.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    serde::Serialize,
    serde::Deserialize,
    carbonite::Schema,
)]
pub enum AssetNeed {
    /// A brush shape a stroke stamps with.
    Brush(AssetId),
    /// The canvas substrate a `SetSubstrate` moves the document onto — named by the
    /// [`AssetId`] inside its [`SubstrateId`], the only kind there are bytes to move
    /// for.
    ///
    /// Missing it is worse than missing a brush: an unresolved shape degrades to the
    /// round tip, whereas an unresolved substrate silently drops the deposition tooth
    /// (§6.4) and bakes a smooth deposit into tiles no later arrival un-bakes.
    Substrate(AssetId),
    /// A picture a `PlaceImage` lands as paint (§23).
    ///
    /// A picture has no degraded form at all — a placement without its pixels is an
    /// empty layer, not a worse version of the action — so `stark-net`'s `content`
    /// never gives up on one.
    Picture(AssetId),
}

impl AssetNeed {
    /// The need a document moving onto `substrate` creates — `None` for `Flat`, which
    /// is procedural and so is never waited on.
    ///
    /// The only place that case is answered: past it a need always carries an
    /// [`AssetId`], so every later question about it has an answer rather than an
    /// answer and a special case.
    pub fn for_substrate(substrate: SubstrateId) -> Option<Self> {
        match substrate {
            SubstrateId::Flat => None,
            SubstrateId::Image(id) => Some(AssetNeed::Substrate(id)),
        }
    }

    /// The id the bytes are named and transferred under.
    pub fn content(self) -> AssetId {
        match self {
            AssetNeed::Brush(id) | AssetNeed::Substrate(id) | AssetNeed::Picture(id) => id,
        }
    }

    /// The substrate a [`Substrate`](Self::Substrate) need names, for
    /// `stark-engine`'s `Engine::accept_substrate`.
    pub fn substrate(self) -> Option<SubstrateId> {
        match self {
            AssetNeed::Brush(_) | AssetNeed::Picture(_) => None,
            AssetNeed::Substrate(id) => Some(SubstrateId::Image(id)),
        }
    }
}

/// The content one action depends on, if any — the single definition, so a new action
/// kind cannot be taught to the loader and forgotten by the transport.
///
/// **Exhaustive, with no `_` arm**: a wildcard would answer "nothing" for an action
/// added later carrying an id, which would then save a document that silently fails to
/// bundle it.
pub fn action_content(action: &Action) -> Option<AssetNeed> {
    match &action.kind {
        ActionKind::CommitStroke(rec) => match rec.brush.shape {
            BrushShape::Stamp(id) => Some(AssetNeed::Brush(id)),
            BrushShape::Round { .. } => None,
        },
        ActionKind::SetSubstrate(id) => AssetNeed::for_substrate(*id),
        ActionKind::PlaceImage { image, .. } => Some(AssetNeed::Picture(*image)),
        ActionKind::AddLayer { .. }
        | ActionKind::AddMatte { .. }
        | ActionKind::AddFilter { .. }
        | ActionKind::DuplicateLayer { .. }
        | ActionKind::RemoveLayer { .. }
        | ActionKind::MergeLayerDown { .. }
        | ActionKind::MoveLayer { .. }
        | ActionKind::SetLayerBlend(..)
        | ActionKind::SetLayerClip(..)
        | ActionKind::SetLayerOpacity(..)
        | ActionKind::SetLayerVisible(..)
        | ActionKind::SetLayerName(..)
        | ActionKind::SetFilter(..)
        | ActionKind::SetMatteRect(..)
        | ActionKind::SetMattePaint(..)
        | ActionKind::SetSubstrateColor(_)
        // The *substrate* names content; the scale it is laid at is only a number.
        | ActionKind::SetSubstrateScale(_)
        | ActionKind::Select(_)
        | ActionKind::InvertSelection
        | ActionKind::SetSelectionOpacity(_)
        | ActionKind::Transform { .. }
        | ActionKind::TransformPerspective { .. }
        | ActionKind::TransformWarp { .. }
        | ActionKind::Fill { .. }
        // Offsets and a cut: geometry and ids, nothing that travels beside the log.
        | ActionKind::TranslateLayers { .. }
        | ActionKind::FloatSelection { .. }
        // A guide is geometry all the way down — a camera and a lattice — so it names
        // nothing that travels beside the log (§20.5).
        | ActionKind::AddGuide { .. }
        | ActionKind::RemoveGuide(_)
        | ActionKind::SetGuide(..)
        | ActionKind::SetGuideName(..)
        | ActionKind::MoveGuide { .. }
        | ActionKind::Undo(_) => None,
    }
}

/// The content one presence frame depends on, if any — [`action_content`]'s twin for
/// the half of the wire that is not the log (§17.5). Only head and resync frames name
/// a shape; a delta frame extends the path of a head already seen.
///
/// **Exhaustive, with no `_` arm**, for [`action_content`]'s reason.
pub fn presence_content(frame: &PeerFrame) -> Option<AssetNeed> {
    match frame.gesture.as_ref()? {
        GestureFrame::Stroke { head, .. } => match head.as_deref()?.brush.shape {
            BrushShape::Stamp(id) => Some(AssetNeed::Brush(id)),
            BrushShape::Round { .. } => None,
        },
        // Geometry and paint carried whole, matching their committed twins above.
        GestureFrame::Selection { .. } | GestureFrame::Fill { .. } => None,
    }
}

impl DocumentFile {
    /// Everything this document's log names, including the substrate it starts on —
    /// which the container names rather than any action.
    pub fn required_content(&self) -> Vec<AssetNeed> {
        let mut needs: Vec<AssetNeed> = self
            .actions
            .iter()
            .filter_map(action_content)
            .chain(AssetNeed::for_substrate(self.canvas.substrate))
            .collect();
        needs.sort_unstable();
        needs.dedup();
        needs
    }

    /// What the log names that the file does not carry — the bill for an incomplete
    /// bundle (§8, §12.4).
    ///
    /// Whoever opens the document must settle this *before* replaying it: a
    /// `SetSubstrate` whose height map is not registered when its strokes replay
    /// deposits them through the flat stand-in, and those pixels are stored (§6.4).
    ///
    /// **A need is answered only by its own store.** An [`AssetId`] is a *content*
    /// hash, so one image imported as a stamp and placed as a picture is one id in two
    /// stores that cannot stand in for each other — hence the bag is keyed by
    /// [`AssetNeed`], the id plus its store.
    pub fn unbundled_content(&self) -> Vec<AssetNeed> {
        let held: std::collections::HashSet<AssetNeed> =
            self.content.iter().map(|(need, _)| *need).collect();
        self.required_content()
            .into_iter()
            .filter(|need| !held.contains(need))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{ActionId, ActionKind, ActorId, BrushParams, LayerId, StrokeRecord};

    fn act(kind: ActionKind) -> Action {
        Action {
            id: ActionId {
                lamport: 1,
                actor: ActorId::SOLO,
            },
            kind,
        }
    }

    fn stroke_with(shape: BrushShape) -> Action {
        act(ActionKind::CommitStroke(StrokeRecord {
            layer: LayerId::ROOT,
            brush: BrushParams {
                shape,
                ..BrushParams::default()
            },
            path: Vec::new(),
            seed: 0,
            start: 0.0,
            translation: crate::geom::IVec2::ZERO,
        }))
    }

    /// **One content hash, three needs, and each answered only by its own.**
    ///
    /// The same image imported as a stamp, laid as a substrate and placed as a picture
    /// is one id filed three ways, and the three decode differently. A bundle
    /// answering "present" for any of them would be short by two, nothing would refuse
    /// the replay, and every stroke on that substrate would deposit through the flat
    /// stand-in (§6.4).
    #[test]
    fn one_id_filed_three_ways_is_three_separate_needs() {
        let id = AssetId([7u8; 32]);
        let bytes = vec![1u8, 2, 3];

        let doc = || {
            DocumentFile::new(vec![
                stroke_with(BrushShape::Stamp(id)),
                act(ActionKind::SetSubstrate(SubstrateId::Image(id))),
                act(ActionKind::PlaceImage {
                    id: LayerId::solo(1),
                    carrier: None,
                    above: None,
                    at: crate::geom::IVec2::ZERO,
                    name: None,
                    image: id,
                }),
            ])
        };
        let all = [
            AssetNeed::Brush(id),
            AssetNeed::Substrate(id),
            AssetNeed::Picture(id),
        ];

        // Each need in turn: carrying one leaves exactly the other two owed.
        for held in all {
            let mut d = doc();
            d.content.push((held, bytes.clone()));
            let mut got = d.unbundled_content();
            got.sort_unstable();
            let mut want: Vec<AssetNeed> = all.into_iter().filter(|n| *n != held).collect();
            want.sort_unstable();
            assert_eq!(got, want, "{held:?} answered for a need that is not it");
        }

        // All three, and the bill is settled.
        let mut d = doc();
        for need in all {
            d.content.push((need, bytes.clone()));
        }
        assert!(d.unbundled_content().is_empty());
    }

    /// `Flat` is procedural: no bytes to move, so it is never a need
    /// ([`AssetNeed::for_substrate`]).
    #[test]
    fn a_flat_substrate_is_never_owed() {
        let doc = DocumentFile::new(vec![act(ActionKind::SetSubstrate(SubstrateId::Flat))]);
        assert!(doc.required_content().is_empty());
        assert!(doc.unbundled_content().is_empty());
    }
}
