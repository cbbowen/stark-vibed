//! Importing a height map: canonical bytes, and the id that names them (§6.4, §8).
//!
//! No GPU here either. This is the asset half of a surface — decode, cap, hash — and
//! it is separate from the upload because it answers a different question: not "how
//! does this ground light and bite" but "which ground is this, and what exactly do we
//! store and send for it".
//!
//! The decode, the cap and the hash themselves are [`stark_assetid`]'s. They are the
//! identity contract every build has to agree on, and nothing here is free to
//! reinterpret them (§19) — what is left in this module is the part that is a
//! *surface's*: wrapping the id in a [`SurfaceId`], because a ground is named by one
//! and a brush shape is not.

use stark_assetid::Canonical;

use crate::error::Result;

use super::SurfaceId;

/// Import a height map: the id that names it, and the canonical bytes to keep
/// beside it. Re-encoded from the decoded height, so what is stored, bundled into a
/// save file and sent to a peer is the form the id actually names — reload it and
/// you land on the same id.
///
/// The engine's entry point is
/// [`Engine::import_surface`](crate::Engine::import_surface).
pub fn canonicalize(png_bytes: &[u8]) -> Result<(SurfaceId, Vec<u8>)> {
    let canonical = stark_assetid::height(png_bytes)?;
    Ok((SurfaceId::Image(canonical.id()), canonical.encode()?))
}

/// The id of an already-canonical height map — bytes out of a save file or off a
/// peer, which are kept verbatim. Derived rather than taken on trust: a ground whose
/// bytes did not hash to the id that asked for them is a ground that would silently
/// deposit the wrong tooth, so the caller gets the id the bytes *are* and compares.
pub fn identify(png_bytes: &[u8]) -> Result<SurfaceId> {
    Ok(SurfaceId::Image(stark_assetid::height(png_bytes)?.id()))
}

/// Decode a height-map PNG to the canonical field the upload path reads.
pub(super) fn canonical_height(png_bytes: &[u8]) -> Result<Canonical> {
    Ok(stark_assetid::height(png_bytes)?)
}
