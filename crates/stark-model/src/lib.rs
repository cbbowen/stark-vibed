//! The Stark document: the action log, its vocabulary, and its file format (§2).
//!
//! **The document is a list of actions, not a bag of pixels.** This crate holds what
//! an [`Action`](document::Action) *is*, what each one reads and writes
//! ([`Footprint`](document::Footprint), §12.6), and how a log is written to a file
//! (§8) or handed to a peer (§12). It compiles without wgpu, without `stark-shaders`
//! and without a build step. `stark-engine` holds the derived view; it depends on
//! this crate and nothing here depends on it.
//!
//! # Which side of the line a type belongs on
//!
//! An **id** is in the log; a **resource** is in the engine —
//! [`AssetId`]/`AssetStore`, [`SubstrateId`]/`SubstrateMap`,
//! [`ColorSpaceId`]/`ColorSpace`, [`LayerId`](document::LayerId)/`Layer`,
//! [`SelectionOp`](document::SelectionOp)/`Selection`,
//! [`Action`](document::Action)/`DocState`. Mechanically: if a type is
//! `Serialize` it is a fact about the document and lives here; if it holds a tile it
//! is a cache and lives there.

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

// A name is lifted to the crate root only when it is this crate's headline and its
// module is incidental. `document`, `geom` and `path` lift nothing: no one name stands
// for them, and every consumer spells the module path anyway.
pub use color::Srgb;
pub use colorspace::ColorSpaceId;
pub use content::{AssetNeed, action_content, presence_content};
pub use error::{DocError, Result};
pub use gradient::{Gradient, GradientStop};
pub use io::{BuildId, CanvasMeta, DocumentFile};
pub use peer::{GestureFrame, PeerFrame, StrokeHead};
/// What a content id *is* — decode, cap, hash (§19). Defined in `stark-assetid` so a
/// build script can compute an id without this crate.
pub use stark_assetid::{AssetId, MAX_SHAPE_DIM};
pub use substrate::{SubstrateId, SubstrateScale};

/// Longest name that travels, in `char`s — layer names, guide names, and a presence
/// frame's display name alike.
///
/// A bound on what one client can make every other client hold, since every such name
/// is replicated and saved. Counted in `char`s rather than bytes so a cut can never
/// land inside one.
pub const MAX_NAME: usize = 64;
