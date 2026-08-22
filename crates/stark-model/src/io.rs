//! The native save format: the serialized action log (§8).
//!
//! The document *is* the list of actions, so the file is a compact action log
//! rather than pixels — replaying it rebuilds the canvas, the full undo
//! timeline, and (via `stark-engine`'s `Engine::replay_timelapse`) a timelapse.
//!
//! ## No version number
//!
//! There is nothing in this container to bump. The body is a **carbonite** frame:
//! the writer's schema — every struct, field name and variant the log used — sits
//! at the head of the file, and loading reconciles it against *this* build's types
//! by name, exactly as reading JSON would (§8). A field added since is filled from
//! its `#[serde(default)]`, one removed is skipped, one renamed is found through
//! its `#[serde(alias)]`, and a variant may be inserted anywhere rather than only
//! appended. So a file this build writes stays readable by later ones, and the
//! version check that used to *refuse* older files — and the thirteen numbered
//! layouts behind it (§8.1) — is gone with the encoding that needed it.
//!
//! The schema is built at **compile time** (`#[derive(carbonite::Schema)]` on every
//! type the log names), so writing a file discovers nothing at runtime. That is not a
//! speed choice: a schema can also be found by *tracing* a type's `Deserialize` impl,
//! but tracing drives it with synthetic values, and the three types here that gate
//! their own invariants — `FillOp`, `SelectionOp`, `Gradient` — refuse those on
//! principle. Each states its wire shape with `#[carbonite(as = "...")]` instead,
//! which is what lets a funnel keep refusing (§8).
//!
//! [`DocError::Legacy`] is what remains of the version ratchet: enough to recognize a
//! pre-carbonite container and say so, since postcard wrote no field names and those
//! bytes cannot be read without the exact schema that produced them.
//!
//! ## Two doors
//!
//! [`DocumentFile::from_bytes`] opens a file the user owns;
//! [`DocumentFile::from_untrusted_bytes`] opens one that arrived from somewhere else
//! (§12.4). They differ in one thing — whether the decompressed body is bounded —
//! and the split exists because a single answer was wrong in both directions. A
//! bound low enough to be worth having against a stranger is one an honest document
//! can cross, since nothing caps how many pictures a document places (§23); so the
//! one bound refused paintings this very build had saved. There is no threat model
//! in which the artist's own file is the attacker.
//!
//! ## File size
//!
//! Two levers keep files small:
//! 1. **carbonite** — a columnar binary encoding: field names are paid for once, in
//!    the schema, and not per value; integers are varints.
//! 2. **deflate** — columnar layout is what a compressor wants, since one field's
//!    values sit back to back, and sampled stroke paths are smooth.
//!
//! Both are pure Rust (deflate via miniz_oxide), so the format also works in the
//! wasm/Dioxus frontend. Further wins (path simplification, delta/quantized
//! samples, and the advisory raster `checkpoints` of §8) are additions a later
//! build can make without a break, which is the point of the paragraph above.

use std::io::{Read, Write};

use flate2::{Compression, read::DeflateDecoder, write::DeflateEncoder};
use serde::{Deserialize, Serialize};

use crate::AssetId;
use crate::ColorSpaceId;
use crate::SurfaceId;
use crate::document::Action;
use crate::error::{DocError, Result};

/// Container magic; identifies a Stark document.
const MAGIC: &[u8; 8] = b"STARKDOC";

/// Largest decompressed container [`DocumentFile::from_untrusted_bytes`] will
/// hold, in bytes.
///
/// A ceiling on what a *stranger* can make this process allocate, which is the
/// only reason there is one: deflate's ratio means a few kilobytes on the wire
/// can name as many gigabytes as it likes. Generous against real documents — the
/// log is a few MB at ten thousand actions, and the bundle is dominated by ground
/// height maps at ~3 MB each — so this is roughly two orders of magnitude of
/// headroom over anything a *session* produces.
///
/// **It does not bound [`DocumentFile::from_bytes`], and that is the fix rather
/// than an oversight.** Applied to both doors it was a bound on the writer that
/// only the reader enforced: nothing caps how many pictures a document places
/// (§23), each is up to `MAX_PICTURE_DIM²` of RGBA, and a dozen photographic
/// placements clear this — so Stark saved a file it then refused to open, which is
/// the one failure a save format may not have. The bound belongs where the bytes
/// are a stranger's, and a painting on the artist's own disk is not that.
const MAX_DECOMPRESSED: u64 = 256 << 20;

/// The last schema version the **pre-carbonite** container carried, and the only
/// reason this build still knows any such number (§8.1).
///
/// A tombstone. It is read from one place — [`legacy_header`], on the way to saying
/// "this file predates the format" — and never to decide how to read a current
/// document, which carries no version at all. Nothing will bump it; when no
/// pre-carbonite files are left to open, the constant and its sniff go together.
const LAST_VERSIONED_SCHEMA: u32 = 13;

/// Build identity, recorded so cross-build replay differences are explainable
/// (§8). Replay is bit-exact within a build; shader/algorithm changes
/// across builds may shift pixels.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, carbonite::Schema)]
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
///
/// **The tile stride is deliberately not here.** It was recorded on every save and
/// then, once something finally read it, used to *refuse* a file written against a
/// different `TILE_SIZE` — on the argument that every tile boundary moves with it,
/// so there was nothing to degrade to. The argument was about pixels, and it proves
/// less than it looks: nothing in a log is expressed in tile units. `TileCoord`,
/// `TileRect` and `Extent2` are not `Serialize` at all, and every action states
/// itself in canvas px. So the stride reaches only *derived* things — which tiles a
/// footprint quantizes to (§12.6), where an apron sits (§6.4), whether an action
/// clears a tile cap — and a document whose pixels come back slightly differently is
/// exactly what §19 permits. Recording it bought one thing: making `TILE_SIZE`
/// unchangeable for the life of the format, since the first change would orphan
/// every file ever saved. An implementation detail is not a fact about a painting,
/// and a field older files carry is skipped on the way in like any other (§8).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, carbonite::Schema)]
pub struct CanvasMeta {
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
            color_space: ColorSpaceId::Oklab,
            surface: SurfaceId::Flat,
        }
    }
}

/// A complete saved document: metadata plus the replayable action log (§8).
///
/// The advisory raster `checkpoints` of §8 are still to come, and are now an ordinary
/// addition: a `#[serde(default)]` field that older files simply do not carry.
#[derive(Clone, Debug, Serialize, Deserialize, carbonite::Schema)]
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
    /// The pictures the document places (§23), as canonical RGBA PNGs — one entry
    /// per `PlaceImage` in the log, deduplicated by content id.
    ///
    /// A **third** bag rather than a third use of `assets`, for the reason the second
    /// one exists: the three decode differently — a mask is luminance × alpha, a
    /// ground is channel 0, a picture is all four channels kept — so a single bag
    /// would hand each store the others' bytes to reinterpret. An id is a *content*
    /// hash, so one image imported as a stamp and placed as a picture carries one id
    /// in two bags that cannot stand in for each other, and
    /// [`unbundled_content`](DocumentFile::unbundled_content) is where that is
    /// enforced.
    ///
    /// This is what a document with placed images weighs, and by far the largest
    /// thing in the container — which is the bargain §23 takes deliberately: the log
    /// stays a log, and the pixels are content beside it, fetched and deduplicated
    /// like every other kind.
    pub pictures: Vec<(AssetId, Vec<u8>)>,
}

impl DocumentFile {
    pub fn new(actions: Vec<Action>) -> Self {
        Self {
            app_build: BuildId::default(),
            canvas: CanvasMeta::default(),
            actions,
            assets: Vec::new(),
            surfaces: Vec::new(),
            pictures: Vec::new(),
        }
    }

    /// Encode to the on-disk container: `MAGIC | deflate(carbonite(self))`.
    ///
    /// The carbonite frame carries its own magic and its own schema, so the eight
    /// bytes prepended here are the whole of Stark's framing — what a file *is*,
    /// not which layout it happens to be in.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let body = carbonite::to_vec_static(self).map_err(DocError::Serialize)?;

        // `default` (level 6) rather than `best` (9). Saving is latency the artist
        // waits through, and level 9 spends a large multiple of 6's time hunting
        // longer matches for a fraction of a percent on data that is already this
        // compressible — fitted paths are smooth, so the wins deflate finds here it
        // finds early. The bundled PNGs, which dominate a large file, are
        // incompressible either way.
        //
        // **The magic goes in the sink, not on the front afterwards.** The encoder
        // appends to whatever `Vec` it is given and never rewinds it, so eight bytes
        // put there first come out ahead of the deflate stream uncompressed — which
        // is what the container wants — and the whole compressed body is spared a
        // copy. Prepending instead cost a third full pass over the file on every
        // save, on top of carbonite's buffer and deflate's: a document is dominated
        // by the pictures it places (§23), and those are the bytes being moved.
        let mut encoder = DeflateEncoder::new(Vec::from(&MAGIC[..]), Compression::default());
        encoder.write_all(&body)?;
        Ok(encoder.finish()?)
    }

    /// Decode a container produced by [`DocumentFile::to_bytes`] that **this user
    /// owns** — a file off their own disk, or bytes this process just wrote.
    ///
    /// Unbounded in what it will expand to, which is what
    /// [`from_untrusted_bytes`](Self::from_untrusted_bytes) exists to be the other
    /// half of. A painting is as large as the artist made it: nothing caps how many
    /// pictures a document places (§23), so a bound low enough to be worth having
    /// against a stranger is one an honest document can cross — and refusing to open
    /// a painting this very build saved is a worse failure than any it prevents. The
    /// bytes here came from somewhere the user already trusts with their files.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::decode(bytes, None)
    }

    /// Decode a container that **arrived from somewhere else** — a peer's snapshot
    /// (§12.4), or any bytes whose author is not the person opening them.
    ///
    /// Identical to [`from_bytes`](Self::from_bytes) except that it refuses a body
    /// expanding past [`MAX_DECOMPRESSED`], since deflate's ratio means a few
    /// kilobytes on the wire can name as many gigabytes as they like.
    pub fn from_untrusted_bytes(bytes: &[u8]) -> Result<Self> {
        Self::decode(bytes, Some(MAX_DECOMPRESSED))
    }

    fn decode(bytes: &[u8], limit: Option<u64>) -> Result<Self> {
        let Some(body) = bytes.strip_prefix(&MAGIC[..]) else {
            return Err(DocError::BadMagic);
        };

        // A bound, where there is one, is applied *before* the body is expanded and
        // not after: deflate takes a long run down to almost nothing, so the
        // compressed length says nothing about the decompressed one, and a reader
        // that finds out by expanding has already spent the memory. `take` one byte
        // past the limit, so "filled the buffer" and "reached the limit" stay
        // distinguishable — `u64::MAX` being the unbounded door's way of saying it
        // will never reach one.
        let mut frame = Vec::new();
        let ceiling = limit.map_or(u64::MAX, |limit| limit + 1);
        DeflateDecoder::new(body)
            .take(ceiling)
            .read_to_end(&mut frame)
            // A pre-carbonite container fails *here*, and only here: its four version
            // bytes sit exactly where the deflate stream has to start, and none of the
            // thirteen makes a well-formed one. So the sniff hangs off the inflate error
            // rather than running first — a current file that inflates is never asked
            // whether it looks old.
            .map_err(|e| legacy_header(body).unwrap_or(DocError::Io(e)))?;
        if let Some(limit) = limit
            && frame.len() as u64 > limit
        {
            return Err(DocError::TooLarge { limit });
        }

        carbonite::from_slice_static(&frame).map_err(DocError::Deserialize)
    }
}

/// Recognize the pre-carbonite container header — `MAGIC` followed by a schema
/// version in `1..=LAST_VERSIONED_SCHEMA` — so an old file can be *named* instead
/// of reported as a corrupt deflate stream (§8.1).
///
/// `body` is what follows the magic. Consulted only after inflating has already
/// failed, so this never decides how a current document is read.
fn legacy_header(body: &[u8]) -> Option<DocError> {
    let head: [u8; 4] = body.get(..4)?.try_into().ok()?;
    match u32::from_le_bytes(head) {
        found @ 1..=LAST_VERSIONED_SCHEMA => Some(DocError::Legacy(found)),
        _ => None,
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

    /// **The cap is the stranger's door, and only the stranger's.**
    ///
    /// One body, the two doors, opposite answers. Deflate takes a long run down to
    /// almost nothing, so the compressed length says nothing about the decompressed
    /// one — the whole point of the bound, and exactly the shape a peer can send
    /// (§12.4). The same bytes off the artist's own disk are expanded, because a
    /// bound low enough to stop that is one an honest document can cross, and a save
    /// format that will not open what it wrote is the worse failure.
    #[test]
    fn the_cap_guards_the_untrusted_door_and_not_the_trusted_one() {
        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::best());
        let zeros = vec![0u8; 1 << 20];
        for _ in 0..((MAX_DECOMPRESSED >> 20) + 8) {
            encoder.write_all(&zeros).expect("deflate");
        }
        let bomb = encoder.finish().expect("deflate");
        let mut bytes = Vec::from(&MAGIC[..]);
        bytes.extend_from_slice(&bomb);
        assert!(
            bytes.len() < 1 << 20,
            "the point is that a small file names a huge one; this one is {} bytes",
            bytes.len(),
        );
        assert!(matches!(
            DocumentFile::from_untrusted_bytes(&bytes),
            Err(DocError::TooLarge { limit }) if limit == MAX_DECOMPRESSED
        ));
        // The trusted door expands it and then finds it is not a document — which is
        // the assertion that matters: it got *past* the size, so a real painting of
        // this weight would open.
        assert!(
            matches!(
                DocumentFile::from_bytes(&bytes),
                Err(DocError::Deserialize(_))
            ),
            "the trusted door must not refuse on size",
        );
    }

    /// …and the cap is nowhere near an honest document, so the untrusted door still
    /// opens everything a session actually produces.
    #[test]
    fn an_ordinary_document_is_nowhere_near_the_cap() {
        let bytes = sample_doc().to_bytes().expect("encode");
        assert!((bytes.len() as u64) < MAX_DECOMPRESSED / 1000);
        assert!(DocumentFile::from_untrusted_bytes(&bytes).is_ok());
    }

    #[test]
    fn rejects_foreign_bytes() {
        assert!(matches!(
            DocumentFile::from_bytes(b"not a stark file"),
            Err(DocError::BadMagic)
        ));
    }

    /// A container from before the format change is **named**, not reported as a
    /// corrupt deflate stream (§8.1).
    ///
    /// Those bytes are postcard: no field names, no lengths, meaningful only to the
    /// exact schema that wrote them. Nothing a newer build does can read them, so
    /// the whole job here is to say which thing went wrong to someone holding a
    /// painting they made — `DocError::Legacy` is the last trace of the version
    /// ratchet this format removed.
    #[test]
    fn a_document_from_before_the_format_change_says_so() {
        for version in [1u32, 6, LAST_VERSIONED_SCHEMA] {
            let mut bytes = Vec::from(&MAGIC[..]);
            bytes.extend_from_slice(&version.to_le_bytes());
            // Whatever followed the header: an old body is not a deflate stream
            // starting four bytes early, which is what makes the sniff sound.
            let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
            encoder
                .write_all(b"a postcard action log")
                .expect("deflate");
            bytes.extend_from_slice(&encoder.finish().expect("deflate"));

            assert!(
                matches!(DocumentFile::from_bytes(&bytes), Err(DocError::Legacy(v)) if v == version),
                "version {version} should be named, got {:?}",
                DocumentFile::from_bytes(&bytes).err(),
            );
        }
    }

    /// …and a *current* file is never asked whether it looks old. The sniff reads
    /// the four bytes where a deflate stream now starts, so a save whose compressed
    /// body happens to open with a small little-endian integer must still load.
    ///
    /// The guard is that the sniff hangs off the inflate error rather than running
    /// first; this is the assertion that would fail if it were reordered.
    #[test]
    fn a_current_document_is_not_mistaken_for_an_old_one() {
        // Enough distinct documents that the leading compressed bytes vary widely.
        for lamport in 0..64 {
            let doc = DocumentFile::new(vec![Action {
                id: ActionId {
                    lamport,
                    actor: ActorId(7),
                },
                kind: ActionKind::AddLayer {
                    id: LayerId(2),
                    carrier: None,
                    above: None,
                },
            }]);
            let bytes = doc.to_bytes().expect("encode");
            assert!(
                DocumentFile::from_bytes(&bytes).is_ok(),
                "a document this build wrote must load back",
            );
        }
    }

    /// **The whole point of the format**: a file written against an older shape of
    /// these types loads, rather than being refused by a version number (§8).
    ///
    /// `Old` is what `DocumentFile` and `CanvasMeta` looked like at some earlier
    /// build — no `surface` on the canvas (a field this build added), and a
    /// `tile_size` this build has since dropped — spelled with the names the real
    /// types carry, because names are what carbonite reconciles on. Both moves are
    /// exercised at once: the added field arrives from its `#[serde(default)]`, the
    /// removed one is skipped, and everything either shape shares comes through.
    ///
    /// `tile_size` is not a hypothetical here: every document written before the
    /// stride stopped being recorded carries one, and this is the assertion that
    /// those files still open (see [`CanvasMeta`]).
    #[test]
    fn a_file_written_against_an_older_shape_still_loads() {
        #[derive(Serialize, Deserialize, carbonite::Schema)]
        #[serde(rename = "CanvasMeta")]
        struct OldCanvas {
            /// Carried then, gone now: a reader has to step over its column.
            tile_size: u32,
            color_space: ColorSpaceId,
        }

        #[derive(Serialize, Deserialize, carbonite::Schema)]
        #[serde(rename = "DocumentFile")]
        struct OldFile {
            app_build: BuildId,
            canvas: OldCanvas,
            actions: Vec<Action>,
            assets: Vec<(AssetId, Vec<u8>)>,
            surfaces: Vec<(SurfaceId, Vec<u8>)>,
            pictures: Vec<(AssetId, Vec<u8>)>,
        }

        let old = OldFile {
            app_build: BuildId::default(),
            canvas: OldCanvas {
                tile_size: crate::geom::TILE_SIZE,
                color_space: ColorSpaceId::Oklab,
            },
            actions: sample_doc().actions,
            assets: Vec::new(),
            surfaces: Vec::new(),
            pictures: vec![(AssetId([3; 32]), vec![1, 2, 3])],
        };

        // Framed exactly as `to_bytes` frames one, since it is the container that
        // used to carry the version this test proves is unnecessary.
        let body = carbonite::to_vec_static(&old).expect("encode the old shape");
        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&body).expect("deflate");
        let mut bytes = Vec::from(&MAGIC[..]);
        bytes.extend_from_slice(&encoder.finish().expect("deflate"));

        let back = DocumentFile::from_bytes(&bytes).expect("an older shape still loads");
        assert_eq!(back.canvas.color_space, ColorSpaceId::Oklab);
        assert_eq!(
            back.canvas.surface,
            SurfaceId::default(),
            "a field added since arrives from its serde default",
        );
        assert_eq!(back.actions.len(), 1);
        assert_eq!(back.actions[0].id.lamport, 3);
        assert_eq!(back.pictures.len(), 1, "and the fields both shapes share");
        assert_eq!(back.pictures[0].1, vec![1, 2, 3]);
    }
}
