//! What a document's pixels depend on besides its log (§6.6, §6.4).
//!
//! A stroke names the shape it stamps with, a `SetSubstrate` names the substrate it moves
//! onto, and a `PlaceImage` names the picture it lands — all three by content id.
//! None of them is in the log — the log carries the *name* — so anything that
//! replays a log has first to answer "what does this need, and have I got it?".
//!
//! That question is asked from three places — loading a file, joining a session, and
//! receiving a peer's action — and it is the same question each time. So the answer
//! lives here, beside the engine's two stores, rather than three times over in
//! whichever crate asked first.
//!
//! The three kinds are one hash and travel one way; they part only at the far end,
//! where a brush mask decodes as luminance × alpha, a substrate as channel 0, and a
//! picture as all four channels kept. So a receiver has to be *told* which it is
//! being handed, and the thing that knows is the action that referenced it.

use crate::AssetId;
use crate::SubstrateId;
use crate::document::{Action, ActionKind, BrushShape};
use crate::io::DocumentFile;

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
    /// [`AssetId`] inside its [`SubstrateId`], the only kind of substrate there is
    /// bytes to move for.
    ///
    /// Missing it is worse than missing a brush: an unresolved shape degrades to
    /// the round tip and the stroke is still visibly a stroke, whereas an
    /// unresolved substrate silently drops the deposition tooth (§6.4) and bakes a
    /// smooth deposit into tiles that no later arrival un-bakes.
    Substrate(AssetId),
    /// A picture a `PlaceImage` lands as paint (§23).
    ///
    /// Missing it is the substrate's case rather than the brush's, and for a sharper
    /// reason than either: a brush degrades to the round tip and a substrate to a wrong
    /// deposit, but a picture has no degraded form at all — a placement without its
    /// pixels is an empty layer, which is not a worse version of the action, it is the
    /// absence of it. So a picture is never given up on (`stark-net`'s `content`).
    Picture(AssetId),
}

impl AssetNeed {
    /// The need a document moving onto `substrate` creates — `None` for `Flat`,
    /// which is procedural, has no bytes to move, and so is never waited on.
    ///
    /// This is the only place a substrate's `Flat` case is answered. Past it the need
    /// carries an [`AssetId`], so every question about it — what it transfers
    /// under, which store it belongs in, whether this peer holds it — has an
    /// answer instead of an answer and a special case.
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

/// The content one action depends on, if any.
///
/// The single definition of "what does this action need", so a new action kind
/// that references content cannot be taught to the loader and forgotten by the
/// transport.
///
/// **Exhaustive, with no `_` arm.** That is what makes the sentence above true of
/// the *future* and not only of today: a wildcard answers "nothing" for every
/// variant that does not exist yet, so an action added later carrying an id would
/// save a document that silently fails to bundle it — and an unresolved substrate
/// bakes a smooth deposit into tiles no later arrival un-bakes. Adding a variant
/// stops this function compiling instead, which is the device `minted_layers` and
/// `tests/footprint.rs`'s `slot` already use.
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
        // The *substrate* names content; the scale it is laid at is a number, and lands
        // here beside the rest of the log's plain numbers.
        | ActionKind::SetSubstrateScale(_)
        | ActionKind::Select(_)
        | ActionKind::InvertSelection
        | ActionKind::SetSelectionOpacity(_)
        | ActionKind::Transform { .. }
        | ActionKind::TransformPerspective { .. }
        | ActionKind::TransformWarp { .. }
        | ActionKind::Fill { .. }
        // A guide is geometry all the way down — a camera and a lattice — so it
        // names nothing that has to travel beside the log (§20.5). It is the one
        // document entity with no content at all.
        | ActionKind::AddGuide { .. }
        | ActionKind::RemoveGuide(_)
        | ActionKind::SetGuide(..)
        | ActionKind::SetGuideName(..)
        | ActionKind::MoveGuide { .. }
        | ActionKind::Undo(_) => None,
    }
}

impl DocumentFile {
    /// Everything this document's log names, including the substrate it starts on —
    /// which is named by the container rather than by any action, and would
    /// otherwise be the one piece of content nothing asks for.
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

    /// What the log names that the file does not carry — the bill for a bundle
    /// that was deliberately left incomplete (§8, §12.4).
    ///
    /// Whoever opens the document has to make this good *before* replaying it: a
    /// `SetSubstrate` whose height map is not registered when its strokes replay
    /// deposits them through the flat stand-in, and those pixels are stored
    /// (§6.4).
    ///
    /// **A need is answered by its own store, and that is now a fact about the
    /// type rather than a rule this function keeps.** The bundle was three
    /// `Vec`s once — brushes, substrates, pictures — because their bytes decode
    /// differently (a mask is luminance × alpha, a substrate is channel 0, a
    /// picture is all four channels kept), and an [`AssetId`] is a *content* hash,
    /// so one image imported as a stamp and placed as a picture carries one id in
    /// two stores that cannot stand in for each other. Asking one flattened set
    /// whether the id was present anywhere answered "bundled" for a substrate whose
    /// bytes were only in the brush bag — and the miss was the silent one: nothing
    /// refused the replay, and every stroke made on that substrate deposited
    /// through the flat stand-in.
    ///
    /// The bag is keyed by [`AssetNeed`] now, which *is* "the id, plus which store
    /// it belongs in". So the wrong answer is no longer something to avoid giving;
    /// it is something there is no longer a way to ask for (§1).
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
        }))
    }

    /// **One content hash, three needs, and each answered only by its own.**
    ///
    /// The hazard this guards is that an [`AssetId`] is a *content* hash, so one image
    /// imported as a stamp, laid as a substrate and placed as a picture carries **one
    /// id** filed three ways — and the three decode differently (luminance × alpha,
    /// channel 0, all four channels kept). A bundle that answered "present" for any of
    /// them because the id was somewhere was short by two, nothing refused the replay,
    /// and every stroke on that substrate deposited through the flat stand-in (§6.4).
    ///
    /// That used to be three `Vec`s and a three-armed match that had to remember not
    /// to cross them. It is one bag keyed by [`AssetNeed`] now, so what is left to
    /// check is not "does it remember" but "does it match" — the id *and* the kind
    /// together, which is the whole content of the key.
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

    /// `Flat` is procedural: it has no bytes to move, so it is never a need and
    /// never waited on — the one place that case is answered
    /// ([`AssetNeed::substrate`]).
    #[test]
    fn a_flat_substrate_is_never_owed() {
        let doc = DocumentFile::new(vec![act(ActionKind::SetSubstrate(SubstrateId::Flat))]);
        assert!(doc.required_content().is_empty());
        assert!(doc.unbundled_content().is_empty());
    }
}
