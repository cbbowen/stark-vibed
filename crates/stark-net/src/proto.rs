//! The wire: message formats and the catch-up/asset protocol.
//!
//! Two channels, one vocabulary (DESIGN.md §12.4):
//!
//! - **The [`mesh`](crate::mesh)** carries [`Wire`] messages — one committed
//!   action each, postcard-encoded. Actions are small (fitted control points,
//!   ids, params); pixels and image bytes never ride the mesh.
//! - **The `stark/collab/0` ALPN** answers [`Request`]s over one bi-stream per
//!   request: the full session [`Snapshot`](Request::Snapshot) (the save-format
//!   container, assets bundled) for joins, and individual content-addressed
//!   [`Asset`](Request::Asset) blobs for strokes that reference a brush image
//!   the peer hasn't seen.
//!
//! [`answer`] is the whole protocol; each backend supplies only the plumbing
//! that carries the bytes.

use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use stark_core::document::Action;
use stark_core::AssetId;

use crate::mirror::Mirror;

/// The catch-up / asset-fetch protocol.
pub const ALPN: &[u8] = b"stark/collab/0";

/// A mesh broadcast. Postcard-encoded; the enum leaves room for presence
/// (cursors, in-flight strokes — session state, never historized) later.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Wire {
    /// A freshly committed action for the shared log.
    Action(Action),
}

/// A request over the collab ALPN (one per bi-stream; the response is the
/// stream's full contents).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Request {
    /// The whole session: a [`DocumentFile`](stark_core::DocumentFile) container.
    Snapshot,
    /// One content-addressed brush image (canonical grayscale PNG bytes).
    Asset(AssetId),
}

/// Response to [`Request::Asset`]: the bytes, if this peer has them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetResponse(pub Option<Vec<u8>>);

/// Answer one request from the shared [`Mirror`] — every peer is a provider, so
/// the session survives the original sharer leaving.
///
/// This is the whole protocol; the transports below only move the bytes.
pub(crate) fn answer(mirror: &Mutex<Mirror>, req: Request) -> crate::Result<Vec<u8>> {
    Ok(match req {
        Request::Snapshot => {
            let file = mirror.lock().expect("mirror poisoned").document_file();
            file.to_bytes()?
        }
        Request::Asset(id) => {
            let bytes = mirror.lock().expect("mirror poisoned").asset(id);
            postcard::to_allocvec(&AssetResponse(bytes))?
        }
    })
}

/// Decode a request received over any transport.
pub(crate) fn decode_request(bytes: &[u8]) -> crate::Result<Request> {
    Ok(postcard::from_bytes(bytes)?)
}

/// The iroh plumbing. Compiled only when the iroh backend is active; the WebRTC
/// backend carries the same protocol over facade streams instead.
#[cfg(not(all(feature = "webrtc", target_family = "wasm", target_os = "unknown")))]
pub(crate) use iroh_wire::{request, CollabProto};

#[cfg(not(all(feature = "webrtc", target_family = "wasm", target_os = "unknown")))]
mod iroh_wire {
    use std::sync::{Arc, Mutex};

    use iroh::endpoint::Connection;
    use iroh::protocol::{AcceptError, ProtocolHandler};

    use super::{answer, decode_request, Request};
    use crate::mirror::Mirror;

    /// Upper bound on an encoded request (a tag + a 32-byte asset id).
    const MAX_REQUEST: usize = 256;
    /// Upper bound on a response: a whole session snapshot (log + brush PNGs).
    const MAX_RESPONSE: usize = 64 * 1024 * 1024;

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
                send.write_all(&response).await.map_err(AcceptError::from_err)?;
                send.finish().map_err(AcceptError::from_err)?;
            }
        }
    }

    /// Issue one request over an open connection and return the raw response.
    pub(crate) async fn request(conn: &Connection, req: Request) -> crate::Result<Vec<u8>> {
        let (mut send, mut recv) = conn.open_bi().await?;
        send.write_all(&postcard::to_allocvec(&req)?).await?;
        send.finish().map_err(|e| crate::NetError::Other(e.to_string()))?;
        Ok(recv.read_to_end(MAX_RESPONSE).await?)
    }
}
