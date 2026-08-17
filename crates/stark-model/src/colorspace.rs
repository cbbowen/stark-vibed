//! Which color space a document is in (§6.7) — the id, not the space.
//!
//! The space itself — the tile layout, the blend, the shaders — is
//! `stark-engine`'s `colorspace`, and building one from an id is its `make`.

use serde::{Deserialize, Serialize};

/// Identifies a color space; serialized in the save format (`CanvasMeta`, §8).
///
/// **Every variant is unconditional, including one whose implementation a build may
/// not carry** (`Mixbox`, behind the `mixbox` cargo feature). Postcard encodes enums
/// by index (§8): a build that `cfg`'d a variant away would renumber every variant
/// appended after it, so the same bytes would name different spaces in two builds of
/// the same version — a wire-format break that nothing would report, which is the
/// class §19 exists to rule out. An id is therefore always nameable and always
/// serializable, and whether it can be *honoured* is `stark-engine`'s
/// `colorspace::make` answer.
///
/// Since the split (§2) that is **structural rather than remembered**: this crate has
/// no `mixbox` feature to `cfg` on, so there is no longer a build in which the variant
/// could go missing. The rule ruled out its own class, which is the convention working
/// as intended — the flag that could have broken the format is now in a crate that
/// cannot see this enum.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorSpaceId {
    Oklab,
    Mixbox,
}
