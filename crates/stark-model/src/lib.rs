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
//! before the crates were: [`AssetId`]/`AssetStore`, [`SurfaceId`]/`Surface`,
//! [`ColorSpaceId`]/`ColorSpace`, [`LayerId`]/`Layer`, [`SelectionOp`]/`Selection`,
//! [`Action`](document::Action)/`DocState`.
//!
//! The mechanical form of the same test is `#[derive(Serialize)]`: if a type is
//! serializable it is a fact about the document and lives here; if it holds a tile
//! it is a cache and lives there. That is not a judgement call — it is the
//! invariant §8 already enforces, which is why the boundary can be checked rather
//! than remembered.
//!

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
pub mod surface;

// # What the root re-exports, and what it does not
//
// The small modules below are flat: `color`, `colorspace`, `content`, `geom`,
// `gradient`, `io`, `path`, `peer` and `surface` each hold a handful of items, so
// their headline types are lifted here and the module stays available for the
// rest. That is the ordinary prelude shape.
//
// **`document` is the exception, and it is deliberate.** It has a curated
// re-export list of its own over crate-private submodules (see its header), so a
// type there already has exactly one public path — and lifting a subset of them
// again gave `LayerId` two, `SelectionOp` two, and `ActionId` and `FillOp` one
// apiece, with nothing choosing between them and no rule saying which four were
// special. Nothing in the workspace ever took the short path; the four are gone
// rather than the other twenty added, so `document::` means what it says.
pub use colorspace::ColorSpaceId;
pub use content::{AssetNeed, action_content};
pub use error::{DocError, Result};
pub use geom::{Extent2, TILE_SIZE, TileCoord, Vec2, ViewTransform};
pub use gradient::{Gradient, GradientStop};
pub use io::{BuildId, CanvasMeta, DocumentFile};
pub use peer::{GestureFrame, PeerFrame, StrokeHead};
/// What a content id *is* — decode, cap, hash (§19). Re-exported rather than
/// redefined: `stark-assetid` is a crate of its own so a *build script* can compute
/// an id, which is what lets the frontend know a bundled asset's id before fetching
/// it. This crate is the same argument one level up, and has no reason to restate it.
pub use stark_assetid::{AssetId, MAX_SHAPE_DIM};
pub use surface::SurfaceId;
