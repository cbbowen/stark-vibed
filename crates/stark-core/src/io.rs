//! The native save format: the serialized action log (DESIGN.md §8).
//!
//! The document *is* the list of actions, so the file is a compact action log
//! rather than pixels — replaying it rebuilds the canvas, the full undo
//! timeline, and (via [`Engine::replay_timelapse`](crate::Engine)) a timelapse.
//!
//! ## File size
//!
//! Two levers keep files small:
//! 1. **postcard** — a dense binary encoding with no field names and varint
//!    integers; far smaller than JSON/CBOR for this data.
//! 2. **deflate** — sampled stroke paths are smooth and highly compressible.
//!
//! Both are pure Rust (deflate via miniz_oxide), so the format also works in the
//! wasm/Dioxus frontend. Further wins (path simplification, delta/quantized
//! samples, and the advisory raster `checkpoints` of §8) are future schema
//! additions gated by the wire version.

use std::io::{Read, Write};

use flate2::{read::DeflateDecoder, write::DeflateEncoder, Compression};
use serde::{Deserialize, Serialize};

use crate::assets::AssetId;
use crate::document::Action;
use crate::error::{EngineError, Result};
use crate::geom::TILE_SIZE;

/// Container magic; identifies a Stark document.
const MAGIC: &[u8; 8] = b"STARKDOC";
/// On-disk schema version. Bump when the serialized layout changes.
const WIRE_VERSION: u32 = 1;

/// Build identity, recorded so cross-build replay differences are explainable
/// (DESIGN.md §8). Replay is bit-exact within a build; shader/algorithm changes
/// across builds may shift pixels.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildId {
    pub app_version: String,
}

impl Default for BuildId {
    fn default() -> Self {
        Self {
            app_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

/// The working color space of the document's color channel.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorSpace {
    Oklab,
}

/// Canvas-wide metadata needed to reproduce the document (DESIGN.md §8).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanvasMeta {
    pub tile_size: u32,
    pub color_space: ColorSpace,
}

impl Default for CanvasMeta {
    fn default() -> Self {
        Self {
            tile_size: TILE_SIZE,
            color_space: ColorSpace::Oklab,
        }
    }
}

/// A complete saved document: metadata plus the replayable action log
/// (DESIGN.md §8). The advisory raster `checkpoints` of §8 are deferred to a
/// later wire version.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DocumentFile {
    pub app_build: BuildId,
    pub canvas: CanvasMeta,
    pub actions: Vec<Action>,
    /// Brush-shape images any stroke references, content-addressed and stored as
    /// compact grayscale PNGs (DESIGN.md §6.6, §8). Bundled so the file is
    /// self-contained and replayable.
    pub assets: Vec<(AssetId, Vec<u8>)>,
}

impl DocumentFile {
    pub fn new(actions: Vec<Action>) -> Self {
        Self {
            app_build: BuildId::default(),
            canvas: CanvasMeta::default(),
            actions,
            assets: Vec::new(),
        }
    }

    /// Encode to the on-disk container: `MAGIC | version | deflate(postcard)`.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let body = postcard::to_allocvec(self)
            .map_err(|e| EngineError::Serialize(e.to_string()))?;

        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::best());
        encoder.write_all(&body)?;
        let compressed = encoder.finish()?;

        let mut out = Vec::with_capacity(MAGIC.len() + 4 + compressed.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&WIRE_VERSION.to_le_bytes());
        out.extend_from_slice(&compressed);
        Ok(out)
    }

    /// Decode a container produced by [`DocumentFile::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let header = MAGIC.len() + 4;
        if bytes.len() < header || &bytes[..MAGIC.len()] != MAGIC {
            return Err(EngineError::BadMagic);
        }
        let version = u32::from_le_bytes(
            bytes[MAGIC.len()..header]
                .try_into()
                .expect("4-byte version"),
        );
        if version != WIRE_VERSION {
            return Err(EngineError::UnsupportedVersion(version));
        }

        let mut body = Vec::new();
        DeflateDecoder::new(&bytes[header..]).read_to_end(&mut body)?;
        postcard::from_bytes(&body).map_err(|e| EngineError::Deserialize(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{Action, ActionId, ActionKind, ActorId, LayerId};

    fn sample_doc() -> DocumentFile {
        DocumentFile::new(vec![Action {
            id: ActionId {
                lamport: 3,
                actor: ActorId(7),
            },
            kind: ActionKind::AddLayer {
                id: LayerId(2),
                above: None,
            },
        }])
    }

    #[test]
    fn container_roundtrips() {
        let doc = sample_doc();
        let bytes = doc.to_bytes().unwrap();
        let back = DocumentFile::from_bytes(&bytes).unwrap();
        assert_eq!(back.actions.len(), 1);
        assert_eq!(back.actions[0].id.lamport, 3);
        assert_eq!(back.canvas, doc.canvas);
        assert!(matches!(
            back.actions[0].kind,
            ActionKind::AddLayer { id: LayerId(2), .. }
        ));
    }

    #[test]
    fn rejects_foreign_bytes() {
        assert!(matches!(
            DocumentFile::from_bytes(b"not a stark file"),
            Err(EngineError::BadMagic)
        ));
    }
}
