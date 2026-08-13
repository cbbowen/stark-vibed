//! The native save format: the serialized action log (§8).
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

use flate2::{Compression, read::DeflateDecoder, write::DeflateEncoder};
use serde::{Deserialize, Serialize};

use crate::assets::AssetId;
use crate::colorspace::ColorSpaceId;
use crate::document::Action;
use crate::error::{EngineError, Result};
use crate::geom::TILE_SIZE;
use crate::gpu::SurfaceId;

/// Container magic; identifies a Stark document.
const MAGIC: &[u8; 8] = b"STARKDOC";
/// On-disk schema version. Bump when the serialized layout changes.
///
/// **2** — layer groups (§14). `AddLayer`, `AddMatte` and `MoveLayer`
/// each grew a `carrier` field, and postcard writes a struct variant's fields in
/// order with no names and no length, so a version-1 file's actions no longer
/// decode. Every earlier schema change could be *appended* — a new enum variant
/// takes the next index and leaves the old ones where they were — but a field in
/// the middle of an existing variant cannot, and inventing a second `MoveLayer`
/// to preserve the layout would have put the duplication this design exists to
/// remove straight back into the log.
///
/// **3** — brush parameter modulation (§6.2) and the deposition tooth (§6.4).
/// `BrushParams` grew a `modulation` field and a `tooth`. It is *appended*, but that is no help here for the reason above: postcard
/// writes no field names and no length, so a version-2 reader would run straight off
/// the end of a brush and into the path behind it. `#[serde(default)]` is likewise
/// no help — it fills a field the *format* left out, which a self-describing format
/// can signal and postcard cannot. A brush's parameters decide its pixels, so the
/// mapping had to be in the record rather than in session state; the version is what
/// that costs. Files are alpha (§19), so old ones are refused rather than migrated.
///
/// **4** — the canvas ground became content-addressed and is bundled (§6.4).
/// `SurfaceId` was a closed enum of names (`Flat | Linen | Gesso`) and is now
/// `Flat | Image(AssetId)`, so a version-3 file's `Linen` would decode as an
/// `Image` whose hash is whatever bytes followed — and the new `surfaces` field
/// pushes `assets` along besides. Both are the same break twice over.
///
/// It is worth what it costs. The tooth reads the ground, so a document's pixels
/// depend on a height map the file did not carry and named only by a label: open it
/// on a build whose `Gesso.png` had been re-authored and the strokes came back
/// different, silently, with nothing in the file able to notice. A file that bundles
/// the ground it was painted on is a file that means one thing.
///
/// **5** — `StrokeRecord` dropped its `tool` field. Removing one is the same
/// break as inserting one, and worse placed: it sat second, so a version-4 file's
/// `Tool` byte would be read as the first byte of the brush colour and every
/// number after it would be off by one. The version is what refuses that, and it
/// is the whole reason a removal is allowed to be this cheap.
///
/// The field could only ever hold `Brush` — the selection tools produce a
/// [`SelectionOp`](crate::document::SelectionOp), never a stroke — and nothing
/// read it back. So it was a constant written into every stroke of every
/// document, and §1's rule about inert scaffolding applies to a field that has
/// stopped meaning something just as much as to one that never did. If a tool
/// ever needs recording, it comes back with whatever distinguishes the tools,
/// which is not this enum.
/// **6** — the bundle may be incomplete (§8, §12.4). A file may now leave out
/// content the opening app can produce itself, and the ids of the assets that
/// ship with the app are exactly that: a document painted on the built-in gesso
/// ground carried 2.8 MB of a height map every install already had.
///
/// Nothing about the *layout* changed, which is why this bump is about meaning
/// rather than decoding. A version-5 file promises a complete bundle; a
/// version-6 one does not, and a reader that does not know to resolve the
/// difference would install what it found, replay, and deposit every stroke made
/// on that ground through the flat stand-in — silently, into stored pixels
/// (§6.4). Refusing the file is the only safe thing an older build can do, and
/// the version is what lets it.
///
/// This walks back part of version 4's reasoning, so it is worth saying what has
/// changed since and what has not. Version 4's problem was that a ground was a
/// *label*: `Gesso` resolved through a table, and re-authoring `Gesso.png` gave
/// you different pixels with nothing able to notice. That is not what this is. A
/// ground is a content id now, the id stays in the file, and content that does
/// not hash to it is refused rather than substituted — so the failure mode of a
/// re-authored asset is a document that will not open, not one that opens wrong.
///
/// What is genuinely given up is self-containment: a lean file needs the app that
/// wrote it, where a version-5 file needed nothing. That is a real cost and the
/// reason `save_bytes` still writes everything unless it is told what may be left
/// out.
///
/// **7** — `FillOp::color` became `paint: Parcel`, an enum whose first variant
/// (`Solid`) carries the old four floats and whose second is the gradient
/// parcel (§22.4). Postcard has no slot for the discriminant that now precedes
/// them, so a version-6 `Fill` action misreads rather than degrades. The clean
/// reshape was chosen over an appended `Option` deliberately: files are alpha
/// (§19), the bump is the whole cost, and a fill's paint being one field is
/// what every consumer of the op gets to rely on from here on.
const WIRE_VERSION: u32 = 7;

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
        let body =
            postcard::to_allocvec(self).map_err(|e| EngineError::Serialize(e.to_string()))?;

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
        let file: Self =
            postcard::from_bytes(&body).map_err(|e| EngineError::Deserialize(e.to_string()))?;

        // **The one place an untrusted colour space enters.** Checked here rather than
        // where the document is adopted, for the reason the magic and the version are:
        // this is the boundary between bytes someone else wrote and a value the engine
        // treats as its own, and a refusal here leaves whatever is open alone.
        //
        // Every `ColorSpaceId` decodes — the enum is unconditional so the save format's
        // indices cannot shift with a build's features (§8, §19) — and what a build may
        // lack is the *implementation*. So this is not a decode failure, and saying so
        // is what lets a frontend offer "this document needs a Mixbox build" instead of
        // "this file is corrupt".
        if !file.canvas.color_space.available() {
            return Err(EngineError::UnsupportedColorSpace(file.canvas.color_space));
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

    #[test]
    fn rejects_foreign_bytes() {
        assert!(matches!(
            DocumentFile::from_bytes(b"not a stark file"),
            Err(EngineError::BadMagic)
        ));
    }

    /// The version check is what makes a layout change safe to make at all: a
    /// file written to an older schema has to be **refused**, not decoded into
    /// whatever its bytes happen to mean now.
    ///
    /// Postcard writes no field names and no lengths, so nothing downstream can
    /// notice. Wire 5 is the sharpest case so far — it dropped a field from the
    /// middle of every stroke, so a version-4 file read as a version-5 one would
    /// take the old `Tool` byte for the first byte of the brush colour and slide
    /// every number after it along. That decodes; it just is not the painting.
    #[test]
    fn rejects_an_older_schema_rather_than_misreading_it() {
        let mut bytes = sample_doc().to_bytes().unwrap();
        let at = MAGIC.len();
        bytes[at..at + 4].copy_from_slice(&(WIRE_VERSION - 1).to_le_bytes());
        assert!(matches!(
            DocumentFile::from_bytes(&bytes),
            Err(EngineError::UnsupportedVersion(v)) if v == WIRE_VERSION - 1
        ));
    }
}
