//! Which color space a document is in (§6.7) — the id, not the space.
//!
//! The space itself — the tile layout, the blend, the shaders — is
//! `stark-engine`'s `colorspace`, and building one from an id is its `make`.

use serde::{Deserialize, Serialize};

/// Identifies a color space; serialized in the save format (`CanvasMeta`, §8).
///
/// **Every variant is unconditional**, including `Mixbox`, whose implementation sits
/// behind a cargo feature the engine may be built without. A `cfg`'d-away variant
/// would make any file naming it undecodable (§8, and `ActionKind`'s tombstone rule
/// for the general case). Whether a build can *honour* an id is `stark-engine`'s
/// `colorspace::make` to answer, as a `DocError::UnsupportedColorSpace`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize, carbonite::Schema)]
pub enum ColorSpaceId {
    Oklab,
    Mixbox,
}
