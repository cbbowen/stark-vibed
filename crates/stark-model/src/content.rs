//! What a document's pixels depend on besides its log (§6.6, §6.4).
//!
//! A stroke names the shape it stamps with and a `SetSurface` names the ground it
//! moves onto, both by content id. Neither is in the log — the log carries the
//! *name* — so anything that replays a log has first to answer "what does this
//! need, and have I got it?".
//!
//! That question is asked from three places — loading a file, joining a session, and
//! receiving a peer's action — and it is the same question each time. So the answer
//! lives here, beside the engine's two stores, rather than three times over in
//! whichever crate asked first.
//!
//! The two kinds are one hash and travel one way; they part only at the far end,
//! where a brush mask decodes as luminance × alpha and a ground as channel 0. So
//! a receiver has to be *told* which it is being handed, and the thing that knows
//! is the action that referenced it.

use crate::AssetId;
use crate::SurfaceId;
use crate::document::{Action, ActionKind, BrushShape};
use crate::io::DocumentFile;

/// Content a document needs before it can be replayed faithfully, and which store
/// it belongs in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AssetNeed {
    /// A brush shape a stroke stamps with.
    Brush(AssetId),
    /// The canvas ground a `SetSurface` moves the document onto — named by the
    /// [`AssetId`] inside its [`SurfaceId`], the only kind of ground there is
    /// bytes to move for.
    ///
    /// Missing it is worse than missing a brush: an unresolved shape degrades to
    /// the round tip and the stroke is still visibly a stroke, whereas an
    /// unresolved ground silently drops the deposition tooth (§6.4) and bakes a
    /// smooth deposit into tiles that no later arrival un-bakes.
    Ground(AssetId),
}

impl AssetNeed {
    /// The need a document moving onto `surface` creates — `None` for `Flat`,
    /// which is procedural, has no bytes to move, and so is never waited on.
    ///
    /// This is the only place a ground's `Flat` case is answered. Past it the need
    /// carries an [`AssetId`], so every question about it — what it transfers
    /// under, which store it belongs in, whether this peer holds it — has an
    /// answer instead of an answer and a special case.
    pub fn ground(surface: SurfaceId) -> Option<Self> {
        match surface {
            SurfaceId::Flat => None,
            SurfaceId::Image(id) => Some(AssetNeed::Ground(id)),
        }
    }

    /// The id the bytes are named and transferred under.
    pub fn content(self) -> AssetId {
        match self {
            AssetNeed::Brush(id) | AssetNeed::Ground(id) => id,
        }
    }

    /// The surface a `Ground` need names, for
    /// `stark-engine`'s `Engine::accept_surface`.
    pub fn surface(self) -> Option<SurfaceId> {
        match self {
            AssetNeed::Brush(_) => None,
            AssetNeed::Ground(id) => Some(SurfaceId::Image(id)),
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
/// save a document that silently fails to bundle it — and an unresolved ground
/// bakes a smooth deposit into tiles no later arrival un-bakes. Adding a variant
/// stops this function compiling instead, which is the device `minted_layers` and
/// `tests/footprint.rs`'s `slot` already use.
pub fn action_content(action: &Action) -> Option<AssetNeed> {
    match &action.kind {
        ActionKind::CommitStroke(rec) => match rec.brush.shape {
            BrushShape::Stamp(id) => Some(AssetNeed::Brush(id)),
            BrushShape::Round { .. } => None,
        },
        ActionKind::SetSurface(id) => AssetNeed::ground(*id),
        ActionKind::AddLayer { .. }
        | ActionKind::AddMatte { .. }
        | ActionKind::AddFilter { .. }
        | ActionKind::DuplicateLayer { .. }
        | ActionKind::RemoveLayer(_)
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
        | ActionKind::SetBackground(_)
        | ActionKind::Select(_)
        | ActionKind::InvertSelection
        | ActionKind::Transform { .. }
        | ActionKind::TransformPerspective { .. }
        | ActionKind::TransformWarp { .. }
        | ActionKind::Fill { .. }
        | ActionKind::Undo(_) => None,
    }
}

impl DocumentFile {
    /// Everything this document's log names, including the ground it starts on —
    /// which is named by the container rather than by any action, and would
    /// otherwise be the one piece of content nothing asks for.
    pub fn required_content(&self) -> Vec<AssetNeed> {
        let mut needs: Vec<AssetNeed> = self
            .actions
            .iter()
            .filter_map(action_content)
            .chain(AssetNeed::ground(self.canvas.surface))
            .collect();
        needs.sort_unstable();
        needs.dedup();
        needs
    }

    /// What the log names that the file does not carry — the bill for a bundle
    /// that was deliberately left incomplete (§8, §12.4).
    ///
    /// Whoever opens the document has to make this good *before* replaying it: a
    /// `SetSurface` whose height map is not registered when its strokes replay
    /// deposits them through the flat stand-in, and those pixels are stored
    /// (§6.4).
    pub fn unbundled_content(&self) -> Vec<AssetNeed> {
        let bundled: std::collections::HashSet<AssetId> = self
            .assets
            .iter()
            .map(|(id, _)| *id)
            .chain(
                self.surfaces
                    .iter()
                    .filter_map(|(id, _)| AssetNeed::ground(*id).map(AssetNeed::content)),
            )
            .collect();
        self.required_content()
            .into_iter()
            .filter(|need| !bundled.contains(&need.content()))
            .collect()
    }
}
