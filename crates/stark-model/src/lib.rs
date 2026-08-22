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
pub use color::Srgb;
pub use colorspace::ColorSpaceId;
pub use content::{AssetNeed, action_content};
pub use error::{DocError, Result};
pub use geom::{Extent2, TILE_SIZE, TileCoord, Vec2};
pub use gradient::{Gradient, GradientStop};
pub use io::{BuildId, CanvasMeta, DocumentFile};
pub use peer::{GestureFrame, PeerFrame, StrokeHead};
/// What a content id *is* — decode, cap, hash (§19). Re-exported rather than
/// redefined: `stark-assetid` is a crate of its own so a *build script* can compute
/// an id, which is what lets the frontend know a bundled asset's id before fetching
/// it. This crate is the same argument one level up, and has no reason to restate it.
pub use stark_assetid::{AssetId, MAX_SHAPE_DIM};
pub use surface::SurfaceId;

/// `x` into [0, 1], with NaN landing on 0 — **the crate's NaN policy**, in one
/// place because it is one policy.
///
/// `max`-then-`min` rather than `clamp`, which is what makes the NaN clause true:
/// `f32::max`/`min` return the non-NaN operand where `clamp` returns the NaN. Same
/// argument as [`BrushParams::taper_px`](document::BrushParams::taper_px), and the
/// reason clippy's suggestion here is the wrong one.
///
/// It grew up in `document::brush`, which is where the values it guards were first
/// coming from, and moved here when the fourth caller was a
/// [`Gradient`](gradient::Gradient) — a type outside `document` entirely. Every
/// deserialization gate in the crate spells the bound this way now
/// ([`SelectionOp::at`](document::SelectionOp::at),
/// [`FillOp::with_paint`](document::FillOp::with_paint),
/// `Gradient::clamped`), and each of them at some point spelled it `clamp` instead
/// — passing a `NaN` opacity through the very funnel that exists to stop one. One
/// definition, so the policy cannot be half-remembered at the next gate.
#[allow(clippy::manual_clamp)]
pub(crate) const fn clamp01(x: f32) -> f32 {
    x.max(0.0).min(1.0)
}

/// `x` if it is a number this parameter can be, else `fallback` — [`clamp01`]'s
/// companion for a knob with **no upper bound** to clamp to.
///
/// Falling back to the field's own default rather than to zero is
/// [`ColorAdjust::sanitized`](document::ColorAdjust)'s argument, read for any
/// parameter: `NaN` says nothing about which end was meant, and a radius silently
/// rounded to 0 is a brush that paints nothing, which is a worse answer than the
/// one the slider ships at.
pub(crate) fn finite_or(x: f32, fallback: f32) -> f32 {
    if x.is_finite() { x } else { fallback }
}

/// `x` as a non-negative length or rate: finite, and floored at zero.
///
/// **Finite first, then floored**, and that order is the whole of it. A bare
/// `x.max(0.0)` turns a `NaN` into 0 and an *infinity* into an infinity, which is
/// exactly half a guard — and the half that was missing is the one a shader
/// notices: an infinite feather reaches `selection.wesl` as a coverage ramp of
/// infinite width, where `0.5 - sd/w` is `0.5` at every texel that is not itself
/// infinitely far away, and `NaN` at the ones that are. A selection nobody asked
/// for, drawn at half strength across the plane.
///
/// Every non-negative length in the crate goes through here now — a brush's radius,
/// drain and tapers, a selection's feather, a fill's feather.
pub(crate) fn at_least_zero(x: f32, fallback: f32) -> f32 {
    finite_or(x, fallback).max(0.0)
}
