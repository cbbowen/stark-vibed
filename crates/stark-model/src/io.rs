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
//! appended. So a file this build writes stays readable by later ones, with no
//! version to check.
//!
//! **What it cannot absorb is a type changing shape.** Reconciliation is by name over
//! a *given* type: a scalar that became a struct is not a field to default or a name
//! to alias, so a file written against the old one does not open, and says so as a
//! decode failure rather than as a named refusal. That has happened once —
//! `LayerId` became the id of the action that minted it (§17.9) — under §19's
//! unclaimed beta rung, which is the permission such a change is taken with and the
//! reason it is a decision rather than an accident. A build that has claimed that rung
//! has no such permission, and a change of this kind then needs the type's old shape
//! kept beside the new one.
//!
//! The schema is built at **compile time** (`#[derive(carbonite::Schema)]` on every
//! type the log names), which is what lets a funnel keep refusing: the alternative,
//! tracing a type's `Deserialize` impl, drives it with synthetic values, and the three
//! types here that gate their own invariants — `FillOp`, `SelectionOp`, `Gradient` —
//! turn those away. Each states its wire shape with `#[carbonite(as = "...")]`
//! instead (§8).
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
//! and the split exists because a single answer is wrong in both directions: a bound
//! low enough to be worth having against a stranger is one an honest document can
//! cross, since nothing caps how many pictures a document places (§23). There is no
//! threat model in which the artist's own file is the attacker.
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
//! wasm/Dioxus frontend. Further wins (path simplification, delta/quantized samples,
//! and the advisory raster `checkpoints` of §8) are additions a later build can make
//! without a break.

use std::io::{Read, Write};

use flate2::{Compression, read::DeflateDecoder, write::DeflateEncoder};
use serde::{Deserialize, Serialize};

use crate::AssetId;
use crate::AssetNeed;
use crate::ColorSpaceId;
use crate::SubstrateId;
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
/// log is a few MB at ten thousand actions, and the bundle is dominated by substrate
/// height maps at ~3 MB each — so this is roughly two orders of magnitude of
/// headroom over anything a *session* produces.
///
/// **It deliberately does not bound [`DocumentFile::from_bytes`].** Nothing caps how
/// many pictures a document places (§23), each up to `MAX_PICTURE_DIM²` of RGBA, so a
/// dozen photographic placements clear this — and a save format that refuses to open
/// what it wrote is the one failure it may not have. The bound belongs where the
/// bytes are a stranger's.
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
/// **The tile stride is deliberately not here.** Nothing in a log is expressed in
/// tile units — `TileCoord`, `TileRect` and `Extent2` are not `Serialize` at all, and
/// every action states itself in canvas px — so the stride reaches only *derived*
/// things: which tiles a footprint quantizes to (§12.6), where an apron sits (§6.4),
/// whether an action clears a tile cap. A document whose pixels come back slightly
/// differently is exactly what §19 permits, whereas recording the stride would make
/// `TILE_SIZE` unchangeable for the life of the format. An implementation detail is
/// not a fact about a painting; older files that carry one are skipped over like any
/// other dropped field (§8).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, carbonite::Schema)]
pub struct CanvasMeta {
    pub color_space: ColorSpaceId,
    /// The substrate the log *starts* from — the initial condition of the empty
    /// document it replays onto, exactly as `color_space` is (§6.4). A
    /// mid-document switch is an `ActionKind::SetSubstrate` in the log, so this is not
    /// the current substrate and loading has to replay to learn that.
    ///
    /// The image behind it rides in [`DocumentFile::content`], under its own
    /// [`AssetNeed`]; this is only its name.
    #[serde(default, alias = "surface")]
    pub substrate: SubstrateId,
}

impl Default for CanvasMeta {
    fn default() -> Self {
        Self {
            color_space: ColorSpaceId::Oklab,
            substrate: SubstrateId::Flat,
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
    /// **Everything the log names that is not in the log**, each entry saying which
    /// store its bytes belong in: the brush shapes strokes stamp with (§6.6), the
    /// canvas substrates a `SetSubstrate` moves onto (§6.4), and the pictures a
    /// `PlaceImage` lands (§23). Bundled so the file is self-contained and replayable.
    ///
    /// # One bag, keyed by the thing that knows which store
    ///
    /// The three kinds decode differently — a mask is luminance × alpha, a substrate
    /// is channel 0, a picture is all four channels kept — and an [`AssetId`] is a
    /// **content** hash, so one image imported as a stamp and placed as a picture
    /// carries one id under two decodings that cannot stand in for each other. Keying
    /// by [`AssetNeed`], which is "the id, plus which store it belongs in", leaves no
    /// way to ask one store about another's bytes (§1).
    ///
    /// This is by far the largest thing in the container when a document places
    /// pictures, which is the bargain §23 takes deliberately: the log stays a log, and
    /// the pixels are content beside it, fetched and deduplicated like every other
    /// kind.
    #[serde(default)]
    pub content: Vec<(AssetNeed, Vec<u8>)>,
    /// Brush shapes, in the bag a pre-`content` document wrote them to.
    #[serde(default, rename = "assets")]
    legacy_assets: Vec<(AssetId, Vec<u8>)>,
    /// Canvas substrates, keyed by [`SubstrateId`] as that bundle was.
    #[serde(default, rename = "substrates", alias = "surfaces")]
    legacy_substrates: Vec<(SubstrateId, Vec<u8>)>,
    /// Placed pictures (§23).
    #[serde(default, rename = "pictures")]
    legacy_pictures: Vec<(AssetId, Vec<u8>)>,
}

impl DocumentFile {
    pub fn new(actions: Vec<Action>) -> Self {
        Self {
            app_build: BuildId::default(),
            canvas: CanvasMeta::default(),
            actions,
            content: Vec::new(),
            legacy_assets: Vec::new(),
            legacy_substrates: Vec::new(),
            legacy_pictures: Vec::new(),
        }
    }

    /// Folds the three tombstoned bags into `content`, each under the need its own
    /// store answers — expressible only because the old shape said which store it
    /// meant by which field it used.
    ///
    /// `assets`, `substrates` and `pictures` are what the bundle was before one bag
    /// keyed by [`AssetNeed`] replaced them. They are kept, hollow, for the reason §8
    /// keeps a retired action's variant: a field this build no longer declares is one
    /// an older file's bytes are silently *dropped* through, and a document that loads
    /// without the brush shape its strokes stamp with degrades to a round tip with
    /// nothing saying why. When no three-bag files are left to open, the three go as
    /// `#[carbonite(removed("assets", "substrates", "surfaces", "pictures"))]` rather
    /// than by deletion — as `LAST_VERSIONED_SCHEMA` will — since a later field taking
    /// one of those names, the alias included, would read the dead column.
    ///
    /// **The three are private, which is what makes this unskippable**: nothing outside
    /// this module can leave one non-empty, and [`decode`](Self::decode) is the only
    /// path in the workspace that deserializes a `DocumentFile`. They therefore go out
    /// empty without anything having to write them so.
    fn fold_legacy_bags(&mut self) {
        self.content.extend(
            std::mem::take(&mut self.legacy_assets)
                .into_iter()
                .map(|(id, b)| (AssetNeed::Brush(id), b)),
        );
        self.content.extend(
            std::mem::take(&mut self.legacy_substrates)
                .into_iter()
                .filter_map(|(id, b)| AssetNeed::for_substrate(id).map(|n| (n, b))),
        );
        self.content.extend(
            std::mem::take(&mut self.legacy_pictures)
                .into_iter()
                .map(|(id, b)| (AssetNeed::Picture(id), b)),
        );
    }

    /// Encode to the on-disk container: `MAGIC | deflate(carbonite(self))`.
    ///
    /// The carbonite frame carries its own magic and its own schema, so the eight
    /// bytes prepended here are the whole of Stark's framing — what a file *is*,
    /// not which layout it happens to be in.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let body = carbonite::to_vec_static(self).map_err(DocError::Serialize)?;

        // `default` (level 6) rather than `best` (9). Saving is latency the artist
        // waits through, and level 9 spends a large multiple of 6's time for a
        // fraction of a percent on data that is already this compressible — fitted
        // paths are smooth, and the bundled PNGs are incompressible either way.
        //
        // **The magic goes in the sink, not on the front afterwards.** The encoder
        // appends to whatever `Vec` it is given and never rewinds it, so eight bytes
        // put there first come out ahead of the deflate stream uncompressed — which is
        // what the container wants — and the whole compressed body is spared a copy.
        let mut encoder = DeflateEncoder::new(Vec::from(&MAGIC[..]), Compression::default());
        encoder.write_all(&body)?;
        Ok(encoder.finish()?)
    }

    /// Decode a container produced by [`DocumentFile::to_bytes`] that **this user
    /// owns** — a file off their own disk, or bytes this process just wrote.
    ///
    /// Unbounded in what it will expand to, which is what
    /// [`from_untrusted_bytes`](Self::from_untrusted_bytes) is the other half of. A
    /// painting is as large as the artist made it, and the bytes here came from
    /// somewhere the user already trusts with their files (see `MAX_DECOMPRESSED`).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::decode(bytes, None)
    }

    /// Decode a container that **arrived from somewhere else** — a peer's snapshot
    /// (§12.4), or any bytes whose author is not the person opening them.
    ///
    /// Identical to [`from_bytes`](Self::from_bytes) except that it refuses a body
    /// expanding past `MAX_DECOMPRESSED`, since deflate's ratio means a few
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

        let mut file: Self = carbonite::from_slice_static(&frame).map_err(DocError::Deserialize)?;
        file.fold_legacy_bags();
        Ok(file)
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
                id: LayerId::solo(2),
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
            ActionKind::AddLayer { id, .. } if id == LayerId::solo(2)
        ));
    }

    /// **The cap is the stranger's door, and only the stranger's.**
    ///
    /// One body, the two doors, opposite answers. Deflate takes a long run down to
    /// almost nothing, so the compressed length says nothing about the decompressed
    /// one — exactly the shape a peer can send (§12.4). The same bytes off the
    /// artist's own disk are expanded.
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

    /// **A damaged container is an `Err`, never a panic** — the claim
    /// [`DocumentFile::from_untrusted_bytes`] exists to support, and the one every
    /// other test here leaves unmade by feeding the decoder well-formed bytes.
    ///
    /// Deterministic and dependency-free: every prefix of a real document, then
    /// single-byte flips at offsets a seeded LCG walks. Raw deflate carries no
    /// checksum, so a flipped stream usually still inflates — which is what puts
    /// garbage in front of the carbonite frame rather than stopping at the container.
    /// The two halves are budgeted separately because a truncation almost always
    /// fails at inflate, and the flips are what reach past it.
    ///
    /// Both doors, since only one of them is bounded, and an `Ok` is made to answer
    /// for itself: the needs are read off the log *and* the bundle, so a document
    /// that decoded into a shell falls over there rather than counting as a pass.
    #[test]
    fn a_damaged_container_is_refused_rather_than_a_panic() {
        fn open_both(bytes: &[u8]) {
            for doc in [
                DocumentFile::from_bytes(bytes),
                DocumentFile::from_untrusted_bytes(bytes),
            ]
            .into_iter()
            .flatten()
            {
                assert!(doc.unbundled_content().len() <= doc.required_content().len());
            }
        }

        const FLIPS: usize = 1000;
        let bytes = sample_doc().to_bytes().expect("encode");

        for len in 0..=bytes.len() {
            open_both(&bytes[..len]);
        }

        let mut seed = 0x2026_0904_5354_524bu64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) as usize
        };
        for _ in 0..FLIPS {
            let mut damaged = bytes.clone();
            let at = next() % damaged.len();
            damaged[at] ^= 1 << (next() % 8);
            open_both(&damaged);
        }
    }

    /// A container from before the format change is **named**, not reported as a
    /// corrupt deflate stream (§8.1).
    ///
    /// Those bytes are postcard: no field names, no lengths, meaningful only to the
    /// exact schema that wrote them. Nothing a newer build can do will read them, so
    /// the whole job here is to say which thing went wrong.
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
                    id: LayerId::solo(2),
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

    /// **A color newtype is not a format change**, which is the claim `Srgb` makes
    /// and the one a save format has to be held to.
    ///
    /// `Old` spells the payloads the way they were written *before* the newtype — a
    /// bare `[f32; 3]`, under the variant name that build used (`SetBackground`, since
    /// renamed to `SetSubstrateColor`). `#[carbonite(as = "[f32; 3]")]` keeps the
    /// column a plain triple on both sides of the change, and `#[serde(alias)]` finds
    /// today's variant under yesterday's name (§8).
    ///
    /// The hostile value is the second half: an older build could write a color
    /// outside the cube, and such a file must **load** rather than be refused —
    /// clamped on the way in, which is §19's stance on tightening an invariant.
    #[test]
    fn a_document_written_before_the_color_newtype_still_loads() {
        use crate::document::{ActionKind, LayerId, MatteRegion, Parcel, Place};

        #[derive(Serialize, Deserialize, carbonite::Schema)]
        #[serde(rename = "Parcel")]
        enum OldPaint {
            Solid([f32; 3]),
        }

        #[derive(Serialize, Deserialize, carbonite::Schema)]
        #[serde(rename = "ActionKind")]
        enum OldKind {
            SetBackground([f32; 3]),
            AddMatte {
                id: LayerId,
                carrier: Option<LayerId>,
                at: Place,
                region: MatteRegion,
                paint: OldPaint,
            },
        }

        #[derive(Serialize, Deserialize, carbonite::Schema)]
        #[serde(rename = "Action")]
        struct OldAction {
            id: ActionId,
            kind: OldKind,
        }

        #[derive(Serialize, Deserialize, carbonite::Schema)]
        #[serde(rename = "DocumentFile")]
        struct OldFile {
            app_build: BuildId,
            canvas: CanvasMeta,
            actions: Vec<OldAction>,
            assets: Vec<(AssetId, Vec<u8>)>,
            surfaces: Vec<(SubstrateId, Vec<u8>)>,
            pictures: Vec<(AssetId, Vec<u8>)>,
        }

        let at = |lamport| ActionId {
            lamport,
            actor: ActorId::SOLO,
        };
        let old = OldFile {
            app_build: BuildId::default(),
            canvas: CanvasMeta::default(),
            actions: vec![
                OldAction {
                    id: at(1),
                    kind: OldKind::SetBackground([0.25, 0.5, 0.75]),
                },
                // What an older build could write and this one cannot: outside the
                // cube, and a `NaN`.
                OldAction {
                    id: at(2),
                    kind: OldKind::SetBackground([-1.0, 2.0, f32::NAN]),
                },
                OldAction {
                    id: at(3),
                    kind: OldKind::AddMatte {
                        id: LayerId::solo(1),
                        carrier: None,
                        at: Place::Bottom,
                        region: MatteRegion::Everything,
                        paint: OldPaint::Solid([0.1, 0.2, 0.3]),
                    },
                },
            ],
            assets: Vec::new(),
            surfaces: Vec::new(),
            pictures: Vec::new(),
        };

        let body = carbonite::to_vec_static(&old).expect("encode the old shape");
        let mut encoder = DeflateEncoder::new(Vec::from(&MAGIC[..]), Compression::default());
        encoder.write_all(&body).expect("deflate");
        let bytes = encoder.finish().expect("deflate");

        let back = DocumentFile::from_bytes(&bytes).expect("a pre-newtype document still loads");
        let colors: Vec<[f32; 3]> = back
            .actions
            .iter()
            .map(|a| match &a.kind {
                ActionKind::SetSubstrateColor(c) => c.get(),
                ActionKind::AddMatte {
                    paint: Parcel::Solid(c),
                    ..
                } => c.get(),
                other => panic!("unexpected {other:?}"),
            })
            .collect();
        assert_eq!(
            colors,
            vec![
                [0.25, 0.5, 0.75],
                // Funnelled on the way in rather than refused — the file opens. Not
                // *clamped* to the cube any more: a color outside it is a wide-gamut
                // color now, and only an unbounded one is held (§6.5, `Srgb`).
                [-1.0, 2.0, 0.0],
                [0.1, 0.2, 0.3],
            ],
        );
    }

    /// **The whole point of the format**: a file written against an older shape of
    /// these types loads, rather than being refused by a version number (§8).
    ///
    /// `Old` is what `DocumentFile` and `CanvasMeta` looked like at an earlier build —
    /// no `substrate` on the canvas (a field this build added), and a `tile_size` this
    /// build has since dropped — spelled with the names the real types carry, since
    /// names are what carbonite reconciles on. Both moves at once: the added field
    /// arrives from its `#[serde(default)]`, the removed one is skipped.
    ///
    /// `tile_size` is not hypothetical — every document written before the stride
    /// stopped being recorded carries one (see [`CanvasMeta`]).
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
            surfaces: Vec<(SubstrateId, Vec<u8>)>,
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
            back.canvas.substrate,
            SubstrateId::default(),
            "a field added since arrives from its serde default",
        );
        assert_eq!(back.actions.len(), 1);
        assert_eq!(back.actions[0].id.lamport, 3);
        // …and the bundle both shapes share, folded out of the tombstoned bag it was
        // written to and into the one keyed by need.
        assert_eq!(
            back.content,
            vec![(AssetNeed::Picture(AssetId([3; 32])), vec![1, 2, 3])],
            "a pre-`content` bundle has to survive the way in",
        );
    }
}
