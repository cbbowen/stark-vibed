//! Which color space a document is in (§6.7) — the id, not the space.
//!
//! The space itself — the tile layout, the blend, the shaders — is
//! `stark-engine`'s `colorspace`, and building one from an id is its `make`.

use serde::{Deserialize, Serialize};

/// Identifies a color space; serialized in the save format (`CanvasMeta`, §8).
///
/// **Every variant is unconditional, including one whose implementation a build may
/// not carry** (`Mixbox`, behind the `mixbox` cargo feature). A build that `cfg`'d a
/// variant away could not read a file that names it: a variant is matched by name and
/// an unknown one has nothing to fall back on, so the document is refused outright
/// (§8, and see `ActionKind`'s tombstone rule for the general case). An id is
/// therefore always nameable and always decodable, and whether it can be *honoured*
/// is `stark-engine`'s `colorspace::make` answer — a `DocError::UnsupportedColorSpace`
/// rather than a corrupt file, which is what lets a frontend say "this document needs
/// a Mixbox build".
///
/// The reason this is worth writing down twice has changed under it. It used to be
/// that postcard encoded enums by index, so dropping a variant renumbered every one
/// after it and the same bytes silently named a *different* space in two builds of
/// the same version. Carbonite reconciles by name, so that particular silent break is
/// gone — the failure is now loud, and the conclusion is the same one.
///
/// Since the split (§2) that is **structural rather than remembered**: this crate has
/// no `mixbox` feature to `cfg` on, so there is no longer a build in which the variant
/// could go missing. The rule ruled out its own class, which is the convention working
/// as intended — the flag that could have broken the format is now in a crate that
/// cannot see this enum.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize, carbonite::Schema)]
pub enum ColorSpaceId {
    Oklab,
    Mixbox,
}
