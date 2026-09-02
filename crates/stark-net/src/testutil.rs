//! Shared test fixtures: the action builders, endpoints and channel drains the
//! module test suites otherwise each re-declared.

use bytes::Bytes;
use iroh::{EndpointId, SecretKey};
use iroh_blobs::Hash;
use stark_model::document::{
    Action, ActionId, ActionKind, ActorId, BrushParams, BrushShape, LayerId, StrokeRecord,
};
use stark_model::geom::IVec2;
use stark_model::{AssetId, DocumentFile, Srgb};
use tokio::sync::mpsc;

use crate::backend::{self, Bound};
use crate::events::{NetOptions, RemoteEvent};
use crate::mirror::{Mirror, Served, SharedMirror};

/// An [`ActionId`] by the fixture actor.
pub fn id(lamport: u64) -> ActionId {
    ActionId {
        lamport,
        actor: ActorId(1),
    }
}

/// A cheap, uniquely identifiable action that references no content — only
/// that it propagates matters.
pub fn action(lamport: u64) -> Action {
    action_by(ActorId(1), lamport)
}

/// [`action`], authored by `actor`.
pub fn action_by(actor: ActorId, lamport: u64) -> Action {
    Action {
        id: ActionId { lamport, actor },
        kind: ActionKind::SetSubstrateColor(Srgb::new([0.0; 3])),
    }
}

/// An action naming the brush `AssetId([tag; 32])` — the door derives the need
/// from the action, so a test cannot pair them inconsistently.
pub fn action_needing(lamport: u64, tag: u8) -> Action {
    action_needing_by(ActorId(1), lamport, tag)
}

/// [`action_needing`], authored by `actor`.
pub fn action_needing_by(actor: ActorId, lamport: u64, tag: u8) -> Action {
    Action {
        id: ActionId { lamport, actor },
        kind: ActionKind::CommitStroke(StrokeRecord {
            layer: LayerId::ROOT,
            brush: BrushParams {
                shape: BrushShape::Stamp(AssetId([tag; 32])),
                ..BrushParams::default()
            },
            path: Vec::new(),
            seed: 0,
            start: 0.0,
            translation: IVec2::ZERO,
        }),
    }
}

/// A deterministic endpoint identity per tag.
pub fn endpoint(tag: u8) -> EndpointId {
    SecretKey::from_bytes(&[tag; 32]).public()
}

/// Distinct content bytes per tag, with the hash they transfer under.
pub fn content(tag: u8) -> (Bytes, Hash) {
    let bytes = Bytes::from(vec![tag; 16]);
    let hash = Hash::new(&bytes);
    (bytes, hash)
}

/// Everything queued on an event channel right now.
pub fn drain(rx: &mut mpsc::UnboundedReceiver<RemoteEvent>) -> Vec<RemoteEvent> {
    let mut out = Vec::new();
    while let Ok(event) = rx.try_recv() {
        out.push(event);
    }
    out
}

/// A bound local stack serving `served` — which may deliberately hold nothing.
pub async fn bound(served: Served) -> Bound {
    backend::bind(served, &NetOptions::local())
        .await
        .expect("bind a local endpoint")
}

/// A bound member serving `log`. The [`Served`] handle comes back too, for its
/// request count.
pub async fn member(log: Vec<Action>) -> (Bound, SharedMirror, Served) {
    let served = Served::default();
    let stack = bound(served.clone()).await;
    let mirror = SharedMirror::new(Mirror::from_file(&DocumentFile::new(log)));
    served.publish(mirror.clone());
    (stack, mirror, served)
}
