//! The wire: message formats and the catch-up/asset protocol handler.
//!
//! Two channels, one vocabulary (DESIGN.md §12.4):
//!
//! - **Gossip** carries [`Wire`] messages — one committed action each, postcard-
//!   encoded. Actions are small (fitted control points, ids, params); pixels and
//!   image bytes never ride gossip.
//! - **The `stark/collab/0` ALPN** answers [`Request`]s over one bi-stream per
//!   request: the full session [`Snapshot`](Request::Snapshot) (the save-format
//!   container, assets bundled) for joins, and individual content-addressed
//!   [`Asset`](Request::Asset) blobs for strokes that reference a brush image
//!   the peer hasn't seen.

use std::sync::{Arc, Mutex};

use iroh::endpoint::Connection;
use iroh::protocol::{AcceptError, ProtocolHandler};
use serde::{Deserialize, Serialize};
use stark_core::document::Action;
use stark_core::AssetId;

use crate::mirror::Mirror;

/// The catch-up / asset-fetch protocol.
pub const ALPN: &[u8] = b"stark/collab/0";

/// Upper bound on an encoded request (a tag + a 32-byte asset id).
const MAX_REQUEST: usize = 256;
/// Upper bound on a response: a whole session snapshot (log + brush PNGs).
const MAX_RESPONSE: usize = 64 * 1024 * 1024;

/// A gossip broadcast. Postcard-encoded; the enum leaves room for presence
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

/// Serves [`Request`]s from the shared [`Mirror`] — every peer is a provider,
/// so the session survives the original sharer leaving.
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
            let req: Request = postcard::from_bytes(&req).map_err(AcceptError::from_err)?;

            let response = match req {
                Request::Snapshot => {
                    let file = self.mirror.lock().expect("mirror poisoned").document_file();
                    file.to_bytes().map_err(AcceptError::from_err)?
                }
                Request::Asset(id) => {
                    let bytes = self.mirror.lock().expect("mirror poisoned").asset(id);
                    postcard::to_allocvec(&AssetResponse(bytes)).map_err(AcceptError::from_err)?
                }
            };
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
