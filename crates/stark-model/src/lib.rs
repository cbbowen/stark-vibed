//! The Stark document: the action log, its vocabulary, and its file format (§2).
//!
//! **The document is a list of actions, not a bag of pixels.** This crate is the
//! first half of that sentence. It holds what an [`Action`] *is*, what each one
//! reads and writes ([`Footprint`], §12.6), and how a log is written to a file
//! (§8) or handed to a peer (§12) — and it compiles without wgpu, without
//! `stark-shaders`, and without a build step.
//!
//! `stark-engine` is the other half: [`DocState`] and the tile pool, the
//! renderers, the compositor and the controller that drives them. It depends on
//! this crate; nothing here depends on it. Pixels are a cached function of the
//! log, so the log does not know what a pixel is.
//!
//! # Which side of the line a type belongs on
//!
//! An **id** is in the log; a **resource** is in the engine. The pairs were there
//! before the crates were: [`AssetId`]/`AssetStore`, [`SurfaceId`]/`Surface`,
//! [`ColorSpaceId`]/`ColorSpace`, [`LayerId`]/`Layer`, [`SelectionOp`]/`Selection`,
//! [`Action`]/`DocState`.
//!
//! The mechanical form of the same test is `#[derive(Serialize)]`: if a type is
//! serializable it is a fact about the document and lives here; if it holds a tile
//! it is a cache and lives there. That is not a judgement call — it is the
//! invariant §8 already enforces, which is why the boundary can be checked rather
//! than remembered.
//!
//! [`DocState`]: https://docs.rs/stark-engine

pub mod color;
pub mod colorspace;
pub mod document;
pub mod geom;
pub mod gradient;
pub mod path;
pub mod surface;

pub use colorspace::ColorSpaceId;
pub use document::{LayerId, SelectionMode, SelectionOp, SelectionShape};
pub use geom::{Extent2, TILE_SIZE, TileCoord, Vec2, ViewTransform};
pub use gradient::{Gradient, GradientStop};
/// What a content id *is* — decode, cap, hash (§19). Re-exported rather than
/// redefined: `stark-assetid` is a crate of its own so a *build script* can compute
/// an id, which is what lets the frontend know a bundled asset's id before fetching
/// it. This crate is the same argument one level up, and has no reason to restate it.
pub use stark_assetid::{AssetId, MAX_SHAPE_DIM};
pub use surface::SurfaceId;
