//! The Stark document: the action log, its vocabulary, and its file format (§2).
//!
//! **The document is a list of actions, not a bag of pixels.** This crate is the
//! first half of that sentence. It holds what an [`Action`](document::Action) *is*, what each one
//! reads and writes ([`Footprint`](document::Footprint), §12.6), and how a log is written to a file
//! (§8) or handed to a peer (§12) — and it compiles without wgpu, without
//! `stark-shaders`, and without a build step.
//!
//! `stark-engine` is the other half: `DocState` and the tile pool, the
//! renderers, the compositor and the controller that drives them. It depends on
//! this crate; nothing here depends on it. Pixels are a cached function of the
//! log, so the log does not know what a pixel is.
//!
//! # Which side of the line a type belongs on
//!
//! An **id** is in the log; a **resource** is in the engine. The pairs were there
//! before the crates were: [`AssetId`]/`AssetStore`, [`SubstrateId`]/`SubstrateMap`,
//! [`ColorSpaceId`]/`ColorSpace`, [`LayerId`](document::LayerId)/`Layer`,
//! [`SelectionOp`](document::SelectionOp)/`Selection`,
//! [`Action`](document::Action)/`DocState`.
//!
//! The mechanical form of the same test is `#[derive(Serialize)]`: if a type is
//! serializable it is a fact about the document and lives here; if it holds a tile
//! it is a cache and lives there.

pub mod color;
pub mod colorspace;
pub mod content;
pub mod document;
pub(crate) mod error;
pub mod geom;
pub mod gradient;
pub mod io;
pub mod path;
pub mod peer;
pub(crate) mod sanitize;
pub mod substrate;

// A module above is `pub` because its whole vocabulary is the API, and that is the
// path nearly every consumer spells. A name is lifted here *as well* only when it is
// this crate's headline and its module is incidental — `Srgb`, `AssetId`,
// `DocumentFile`.
//
// `document`, `geom` and `path` lift nothing, because no one name stands for them.
// `document` re-exports a curated list of its own over crate-private submodules (see
// its header), so a type there has exactly one public path already. `geom`'s lift was
// dropped after a count: ~280 sites spelled `geom::` against a dozen that did not, so
// all the short path bought was making those dozen read differently from every
// neighbour.
pub use color::Srgb;
pub use colorspace::ColorSpaceId;
pub use content::{AssetNeed, action_content, presence_content};
pub use error::{DocError, Result};
pub use gradient::{Gradient, GradientStop};
pub use io::{BuildId, CanvasMeta, DocumentFile};
pub use peer::{GestureFrame, PeerFrame, StrokeHead};
/// What a content id *is* — decode, cap, hash (§19). Re-exported rather than
/// redefined: `stark-assetid` is a crate of its own so a *build script* can compute
/// an id, which is what lets the frontend know a bundled asset's id before fetching
/// it. This crate is the same argument one level up, and has no reason to restate it.
pub use stark_assetid::{AssetId, MAX_SHAPE_DIM};
pub use substrate::{SubstrateId, SubstrateScale};

/// Longest name that travels, in `char`s.
///
/// A bound on what one client can make every other client hold: a layer's or a
/// guide's name is replicated to every peer and saved with the document, and a
/// presence frame's display name is republished to the whole session — and nothing
/// about a text field stops a paste from being a megabyte. Counted in `char`s and
/// not bytes because it is user text, so the cut can never land inside one.
///
/// Here rather than in [`peer`], which is only the presence half: `stark-engine`'s
/// `normalize_name` caps a *layer* name to this too. Here rather than in the engine
/// because the model cannot depend on the engine (§2).
pub const MAX_NAME: usize = 64;
