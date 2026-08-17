//! The native save format: the serialized action log (§8).
//!
//! The document *is* the list of actions, so the file is a compact action log
//! rather than pixels — replaying it rebuilds the canvas, the full undo
//! timeline, and (via `stark-engine`'s `Engine::replay_timelapse`) a timelapse.
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

use flate2::{Compression, read::DeflateDecoder, write::DeflateEncoder};
use serde::{Deserialize, Serialize};

use crate::AssetId;
use crate::ColorSpaceId;
use crate::SurfaceId;
use crate::document::Action;
use crate::error::{DocError, Result};
use crate::geom::TILE_SIZE;

/// Container magic; identifies a Stark document.
const MAGIC: &[u8; 8] = b"STARKDOC";

/// Largest decompressed container this build will hold, in bytes.
///
/// A ceiling on what a *stranger* can make this process allocate, which is the
/// only reason there is one: a document arrives from a peer as readily as from
/// disk (§12.4), and deflate's ratio means a few kilobytes on the wire can name
/// as many gigabytes as it likes. Generous against real documents — the log is
/// a few MB at ten thousand actions, and the bundle is dominated by ground height
/// maps at ~3 MB each — so this is roughly two orders of magnitude of headroom
/// over anything a session produces, and refuses only what no session would.
const MAX_DECOMPRESSED: u64 = 256 << 20;
/// On-disk schema version. Bump when the serialized layout changes.
///
/// **What each bump was and what forced it is §8.1**, which also names the three
/// shapes a break comes in. It lived here, and had grown to a hundred and fifty
/// lines hanging off a `const u32` — a record worth keeping and not worth reading
/// every time someone hovers the constant (§1: cite sections, not line numbers).
///
/// What matters at this line is only the rule the history is evidence for:
/// postcard writes a struct's fields **in order**, with no names and no lengths,
/// and an enum by **variant index**. So appending a variant is free, appending a
/// field to a variant is not, and a file written to an older schema has to be
/// *refused* rather than decoded into whatever its bytes happen to mean now —
/// which is what `rejects_an_older_schema_rather_than_misreading_it` pins. Files
/// are alpha (§19), so old ones are refused rather than migrated.
const WIRE_VERSION: u32 = 12;

/// Build identity, recorded so cross-build replay differences are explainable
/// (§8). Replay is bit-exact within a build; shader/algorithm changes
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

/// Canvas-wide metadata needed to reproduce the document (§8).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanvasMeta {
    /// The tile stride the log was recorded against, in canvas px
    /// ([`TILE_SIZE`](crate::geom::TILE_SIZE)).
    ///
    /// **Checked on load, which is what makes it worth writing.** It was recorded
    /// on every save and read by nothing at all — so a file from a build with a
    /// different stride loaded silently and rendered wrong, which is precisely the
    /// reproducibility question the field was added to answer. Every tile boundary
    /// moves with this number: a stroke's footprint quantizes against it (§12.6),
    /// an apron sits one texel inside it (§6.4), and a fill's written tiles are
    /// derived from it. None of that degrades gracefully.
    ///
    /// A constant rather than something a document may choose. It is stored so the
    /// file can be *refused* by a build that would read it differently, exactly as
    /// the wire version is — see [`DocError::TileSize`](crate::DocError::TileSize).
    pub tile_size: u32,
    pub color_space: ColorSpaceId,
    /// The ground the log *starts* from — the initial condition of the empty
    /// document it replays onto, exactly as `color_space` is (§6.4). A
    /// mid-document switch is an `ActionKind::SetSurface` in the log, so this is not
    /// the current ground and loading has to replay to learn that.
    ///
    /// The image behind it rides in [`DocumentFile::surfaces`]; this is only its
    /// name.
    #[serde(default)]
    pub surface: SurfaceId,
}

impl Default for CanvasMeta {
    fn default() -> Self {
        Self {
            tile_size: TILE_SIZE,
            color_space: ColorSpaceId::Oklab,
            surface: SurfaceId::Flat,
        }
    }
}

/// A complete saved document: metadata plus the replayable action log
/// (§8). The advisory raster `checkpoints` of §8 are deferred to a
/// later wire version.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DocumentFile {
    pub app_build: BuildId,
    pub canvas: CanvasMeta,
    pub actions: Vec<Action>,
    /// Brush-shape images any stroke references, content-addressed and stored as
    /// compact grayscale PNGs (§6.6, §8). Bundled so the file is
    /// self-contained and replayable.
    pub assets: Vec<(AssetId, Vec<u8>)>,
    /// The canvas grounds the document names — `CanvasMeta::surface` plus every
    /// `SetSurface` in the log — as canonical grayscale height maps (§6.4, §8).
    ///
    /// Here for the same reason `assets` is, and it took the deposition tooth to
    /// make that visible: once the ground gates how much paint lands, a height map
    /// is a *replay input*, no different from a brush's coverage mask, and a file
    /// that omits one does not contain its own painting. Keyed separately from
    /// `assets` because the two decode differently — a mask is luminance × alpha,
    /// a ground is channel 0 — so a single bag would hand each store the other's
    /// bytes to reinterpret.
    ///
    /// `Flat` contributes nothing: it is procedural, and the empty vector of a
    /// document that never left it is the honest encoding of that.
    pub surfaces: Vec<(SurfaceId, Vec<u8>)>,
}

impl DocumentFile {
    pub fn new(actions: Vec<Action>) -> Self {
        Self {
            app_build: BuildId::default(),
            canvas: CanvasMeta::default(),
            actions,
            assets: Vec::new(),
            surfaces: Vec::new(),
        }
    }

    /// Encode to the on-disk container: `MAGIC | version | deflate(postcard)`.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let body = postcard::to_allocvec(self).map_err(|e| DocError::Serialize(e.to_string()))?;

        // `default` (level 6) rather than `best` (9). Saving is latency the artist
        // waits through, and level 9 spends a large multiple of 6's time hunting
        // longer matches for a fraction of a percent on data that is already this
        // compressible — fitted paths are smooth, so the wins deflate finds here it
        // finds early. The bundled PNGs, which dominate a large file, are
        // incompressible either way.
        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
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
            return Err(DocError::BadMagic);
        }
        let version = u32::from_le_bytes(
            bytes[MAGIC.len()..header]
                .try_into()
                .expect("4-byte version"),
        );
        if version != WIRE_VERSION {
            return Err(DocError::UnsupportedVersion(version));
        }

        // Bounded before it is expanded, not after: deflate takes a long run down
        // to almost nothing, so the compressed length says nothing about the
        // decompressed one and a reader that finds out by expanding has already
        // spent the memory. `take` one byte past the limit, so "filled the buffer"
        // and "reached the limit" are distinguishable.
        let mut body = Vec::new();
        DeflateDecoder::new(&bytes[header..])
            .take(MAX_DECOMPRESSED + 1)
            .read_to_end(&mut body)?;
        if body.len() as u64 > MAX_DECOMPRESSED {
            return Err(DocError::TooLarge {
                limit: MAX_DECOMPRESSED,
            });
        }
        let file: Self =
            postcard::from_bytes(&body).map_err(|e| DocError::Deserialize(e.to_string()))?;

        // A sibling of the version check above, and refused for the same reason:
        // the bytes decode perfectly and simply do not mean what this build would
        // read them as. Every tile boundary in the document moves with this number
        // (see `CanvasMeta::tile_size`), so there is nothing to degrade to.
        if file.canvas.tile_size != TILE_SIZE {
            return Err(DocError::TileSize {
                expected: TILE_SIZE,
                found: file.canvas.tile_size,
            });
        }

        Ok(file)
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
                carrier: None,
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

    /// A container whose body expands past the cap is **refused rather than
    /// expanded**, and refused without ever holding the expansion.
    ///
    /// Deflate takes a long run down to almost nothing, so the compressed length
    /// says nothing about the decompressed one — the whole point of the bound.
    /// The body here is a few KB on the wire and a third of a gigabyte off it,
    /// which is exactly the shape a peer can send (§12.4).
    #[test]
    fn refuses_a_body_that_expands_past_the_cap() {
        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::best());
        let zeros = vec![0u8; 1 << 20];
        for _ in 0..((MAX_DECOMPRESSED >> 20) + 8) {
            encoder.write_all(&zeros).expect("deflate");
        }
        let bomb = encoder.finish().expect("deflate");
        let mut bytes = Vec::from(&MAGIC[..]);
        bytes.extend_from_slice(&WIRE_VERSION.to_le_bytes());
        bytes.extend_from_slice(&bomb);
        assert!(
            bytes.len() < 1 << 20,
            "the point is that a small file names a huge one; this one is {} bytes",
            bytes.len(),
        );
        assert!(matches!(
            DocumentFile::from_bytes(&bytes),
            Err(DocError::TooLarge { limit }) if limit == MAX_DECOMPRESSED
        ));
    }

    /// …and the cap is nowhere near an honest document: the roundtrip above has to
    /// keep working, which is what stops the bound being tightened into a bug.
    #[test]
    fn an_ordinary_document_is_nowhere_near_the_cap() {
        let bytes = sample_doc().to_bytes().expect("encode");
        assert!((bytes.len() as u64) < MAX_DECOMPRESSED / 1000);
        assert!(DocumentFile::from_bytes(&bytes).is_ok());
    }

    /// A stride this build does not address is **refused**, not loaded.
    ///
    /// The field was written on every save and read by nothing, so this file used
    /// to open clean — and then quantize every footprint, apron and fill against a
    /// grid the log was not recorded on.
    #[test]
    fn rejects_a_document_recorded_on_another_tile_grid() {
        let mut doc = sample_doc();
        doc.canvas.tile_size = TILE_SIZE * 2;
        let bytes = doc.to_bytes().expect("encode");
        assert!(matches!(
            DocumentFile::from_bytes(&bytes),
            Err(DocError::TileSize { expected, found })
                if expected == TILE_SIZE && found == TILE_SIZE * 2
        ));
        // …and this build's own stride is of course accepted.
        assert!(DocumentFile::from_bytes(&sample_doc().to_bytes().expect("encode")).is_ok());
    }

    #[test]
    fn rejects_foreign_bytes() {
        assert!(matches!(
            DocumentFile::from_bytes(b"not a stark file"),
            Err(DocError::BadMagic)
        ));
    }

    /// The version check is what makes a layout change safe to make at all: a
    /// file written to an older schema has to be **refused**, not decoded into
    /// whatever its bytes happen to mean now.
    ///
    /// Postcard writes no field names and no lengths, so nothing downstream can
    /// notice. Wire 5 is the sharpest case so far — it dropped a field from the
    /// middle of every stroke, so a version-4 file read as a version-5 one would
    /// take the old `Tool` byte for the first byte of the brush color and slide
    /// every number after it along. That decodes; it just is not the painting.
    #[test]
    fn rejects_an_older_schema_rather_than_misreading_it() {
        let mut bytes = sample_doc().to_bytes().unwrap();
        let at = MAGIC.len();
        bytes[at..at + 4].copy_from_slice(&(WIRE_VERSION - 1).to_le_bytes());
        assert!(matches!(
            DocumentFile::from_bytes(&bytes),
            Err(DocError::UnsupportedVersion(v)) if v == WIRE_VERSION - 1
        ));
    }
}
