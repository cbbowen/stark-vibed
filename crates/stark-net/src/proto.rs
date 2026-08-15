//! The wire: message formats and the catch-up/asset protocol.
//!
//! Three channels, one vocabulary (§12.4):
//!
//! - **Gossip** carries [`Stamped`] messages — one committed action or
//!   presence frame each, postcard-encoded. Actions are small (fitted control
//!   points, ids, params); pixels and image bytes never ride gossip.
//! - **The `stark/collab/1` ALPN** answers [`Request`]s over one bi-stream per
//!   request: the full session [`Snapshot`](Request::Snapshot) (the save-format
//!   container, assets bundled) for joins.
//! - **The `iroh-blobs` ALPN** serves individual brush images to peers that
//!   see a stroke referencing one they don't hold — hash-verified, addressed
//!   by the blob hash a [`Stamped`] message carries alongside such a stroke.

use std::sync::Mutex;

use iroh::EndpointId;
use serde::{Deserialize, Serialize};
use stark_core::AssetId;
use stark_core::document::Action;
use stark_core::peer::PeerFrame;

use crate::mirror::Mirror;

/// The catch-up (snapshot) protocol. The trailing number moves with the wire:
/// gossip payloads carry no version of their own, so two builds whose action
/// encoding differs must fail to *meet* rather than decode each other's
/// messages wrong — bumped with `WIRE_VERSION` whenever an action reshapes
/// (1: `FillOp`'s parcel; 2: the matte's paint and anchor, §22.4, §15.4;
/// 3: a fill's strength became one field and a coverage, §6.8; 4: `SelectionOp`
/// gained its opacity, §6.8; 5: `BlendMode::Drago` gained its bend, §6.3).
pub(crate) const ALPN: &[u8] = b"stark/collab/5";

/// One gossip broadcast: the payload plus who authored it. Postcard-encoded.
///
/// Gossip forwards messages through intermediate peers and reports only the
/// *delivering* neighbor, so the author travels in the payload. It is
/// self-declared — the same trust already placed in the payload itself, since
/// anyone holding the ticket can write anything (§12.5 defers
/// authentication).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Stamped {
    /// Who produced the message — the authoritative source for anything it
    /// references (a presence frame's author, a stroke's brush asset).
    pub origin: EndpointId,
    /// The blob hash of the brush image the payload references, if any. An
    /// [`AssetId`] names the *decoded coverage* (encoding-independent), so it
    /// is not itself fetchable over blobs — the author, who holds the bytes,
    /// supplies the transfer hash here. Trusted like the rest of the payload;
    /// the engine re-derives the real `AssetId` from the fetched bytes.
    pub asset: Option<iroh_blobs::Hash>,
    pub wire: Wire,
}

/// A live-wire message. Postcard-encoded, inside [`Stamped`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum Wire {
    /// A freshly committed action for the shared log.
    Action(Action),
    /// One client's presence: cursor, selected layer, the gesture it is drawing
    /// (§17.4). **Never mirrored and never snapshotted** — it is not
    /// part of the document, and nothing in the log refers to it, which is the whole
    /// reason it may be dropped, coalesced or delayed without affecting convergence.
    ///
    /// The author is not in the frame: the receiver takes it from the
    /// [`Stamped`] envelope, whose `origin` names exactly one author
    /// (§17.7).
    Presence(PeerFrame),
}

/// A request over the collab ALPN (one per bi-stream; the response is the
/// stream's full contents).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum Request {
    /// The whole session: a [`DocumentFile`](stark_core::DocumentFile) container.
    Snapshot,
    /// The session, minus the content the joiner says it can resolve without help
    /// — the ids of the assets that ship with its build (§12.4).
    ///
    /// A separate variant rather than a field on [`Request::Snapshot`] because
    /// postcard encodes enums by index and appending a variant is the safe
    /// change (§8): an older peer that only knows `Snapshot` keeps working, and a
    /// newer one asking an older host gets a decode error on a request rather
    /// than a silently full bundle.
    ///
    /// The list is a **promise**, not an inventory — "I can get these", not "I
    /// have these loaded". The joiner has to make it good before replaying, and
    /// the blob fetch is what catches it if it cannot.
    SnapshotWithout(Vec<AssetId>),
}

/// Answer one request from the shared [`Mirror`] — every peer is a provider, so
/// the session survives the original sharer leaving.
///
/// This is the whole protocol; the transports below only move the bytes.
pub(crate) fn answer(mirror: &Mutex<Mirror>, req: Request) -> crate::Result<Vec<u8>> {
    Ok(match req {
        Request::Snapshot => snapshot_bytes(mirror, &[])?,
        Request::SnapshotWithout(have) => snapshot_bytes(mirror, &have)?,
    })
}

/// Encode the session snapshot, leaving out any content in `have`.
fn snapshot_bytes(mirror: &Mutex<Mirror>, have: &[AssetId]) -> crate::Result<Vec<u8>> {
    // Cloned under the lock, materialized and encoded outside it: asset payloads
    // are refcounted handles in the mirror, so the only real work the lock covers
    // is the log — and a joiner arriving mid-session does not stall this peer's
    // receive loop for the size of its own brush library.
    let snapshot = mirror.lock().expect("mirror poisoned").snapshot();
    Ok(snapshot.without(have).into_file().to_bytes()?)
}

/// Decode a request received over any transport.
pub(crate) fn decode_request(bytes: &[u8]) -> crate::Result<Request> {
    Ok(postcard::from_bytes(bytes)?)
}

/// The iroh plumbing: the protocol handler and the client-side request call.
pub(crate) use iroh_wire::{CollabProto, request};

mod iroh_wire {
    use std::sync::{Arc, Mutex};

    use iroh::endpoint::Connection;
    use iroh::protocol::{AcceptError, ProtocolHandler};

    use super::{Request, answer, decode_request};
    use crate::mirror::Mirror;

    /// Upper bound on an encoded request.
    ///
    /// A request carries the joiner's list of resolvable content ids, 32 bytes each,
    /// so the ceiling has to clear a catalog that grows rather than the variant tag
    /// alone: 64 KiB is two thousand of them.
    const MAX_REQUEST: usize = 64 * 1024;
    /// Upper bound on a response: a whole session snapshot (log + brush PNGs).
    /// A session that outgrows it stops accepting new members, so crossing most
    /// of the way there is worth saying out loud while joining still works.
    const MAX_RESPONSE: usize = 64 * 1024 * 1024;
    /// Fraction of [`MAX_RESPONSE`] a snapshot may reach before it is reported.
    const RESPONSE_WARN_AT: usize = MAX_RESPONSE / 2;

    /// Serves [`Request`]s over iroh connections.
    #[derive(Debug, Clone)]
    pub(crate) struct CollabProto {
        pub mirror: Arc<Mutex<Mirror>>,
    }

    impl ProtocolHandler for CollabProto {
        async fn accept(&self, connection: Connection) -> Result<(), AcceptError> {
            // Serve requests until the peer closes the connection.
            loop {
                let Ok((mut send, mut recv)) = connection.accept_bi().await else {
                    return Ok(());
                };
                let req = recv
                    .read_to_end(MAX_REQUEST)
                    .await
                    .map_err(AcceptError::from_err)?;
                let req = decode_request(&req).map_err(AcceptError::from_err)?;
                let response = answer(&self.mirror, req).map_err(AcceptError::from_err)?;
                if response.len() > RESPONSE_WARN_AT {
                    tracing::warn!(
                        bytes = response.len(),
                        limit = MAX_RESPONSE,
                        "session snapshot is approaching the response ceiling; past \
                         it no new member can join"
                    );
                }
                send.write_all(&response)
                    .await
                    .map_err(AcceptError::from_err)?;
                send.finish().map_err(AcceptError::from_err)?;
            }
        }
    }

    /// Issue one request over an open connection and return the raw response.
    pub(crate) async fn request(conn: &Connection, req: Request) -> crate::Result<Vec<u8>> {
        let (mut send, mut recv) = conn.open_bi().await?;
        send.write_all(&postcard::to_allocvec(&req)?).await?;
        send.finish()
            .map_err(|e| crate::NetError::Other(e.to_string()))?;
        Ok(recv.read_to_end(MAX_RESPONSE).await?)
    }
}
