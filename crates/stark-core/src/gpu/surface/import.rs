//! Importing a height map: canonical bytes, and the id that names them (§6.4, §8).
//!
//! No GPU here either. This is the asset half of a surface — decode, cap, hash — and
//! it is separate from the upload because it answers a different question: not "how
//! does this ground light and bite" but "which ground is this, and what exactly do we
//! store and send for it".

use crate::assets::{AssetId, downsample_to_limit, encode_gray_png};
use crate::error::{EngineError, Result};
use crate::gpu::context::MAX_TEXTURE_DIM_2D;

use super::SurfaceId;

/// Import a height map: the id that names it, and the canonical bytes to keep
/// beside it. Re-encoded from the decoded height, so what is stored, bundled into a
/// save file and sent to a peer is the form the id actually names — reload it and
/// you land on the same id.
///
/// The engine's entry point is
/// [`Engine::import_surface`](crate::Engine::import_surface).
pub fn canonicalize(png_bytes: &[u8]) -> Result<(SurfaceId, Vec<u8>)> {
    let (w, h, height) = canonical_height(png_bytes)?;
    let id = SurfaceId::Image(surface_id(w, h, &height));
    Ok((id, encode_gray_png(w, h, &height)?))
}

/// The id of an already-canonical height map — bytes out of a save file or off a
/// peer, which are kept verbatim. Derived rather than taken on trust: a ground whose
/// bytes did not hash to the id that asked for them is a ground that would silently
/// deposit the wrong tooth, so the caller gets the id the bytes *are* and compares.
pub fn identify(png_bytes: &[u8]) -> Result<SurfaceId> {
    let (w, h, height) = canonical_height(png_bytes)?;
    Ok(SurfaceId::Image(surface_id(w, h, &height)))
}

/// Content id of a canonical height field: the hash of its dimensions and texels.
///
/// Over the *decoded, downsampled* field rather than the file bytes, for the reason
/// [`AssetId`] names a brush's coverage the same way — it is what actually drives
/// pixels, so two peers who encoded the same weave differently converge on one id.
/// That this is deterministic across peers rests on [`MAX_TEXTURE_DIM_2D`] being a
/// fixed constant rather than a device query: were the downsample factor to follow
/// the adapter's real limit, the same PNG would canonicalize differently on two
/// machines and the id would stop naming one thing.
fn surface_id(width: u32, height: u32, texels: &[u8]) -> AssetId {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&width.to_le_bytes());
    hasher.update(&height.to_le_bytes());
    hasher.update(texels);
    AssetId(*hasher.finalize().as_bytes())
}

/// Decode a height-map PNG to its canonical form: one height byte per texel,
/// box-downsampled by an integer factor to fit [`MAX_TEXTURE_DIM_2D`].
///
/// Channel 0, not luminance — a height map's grey *is* its height, so an RGB source
/// carries it in red and weighting the channels would tilt the ground.
pub(super) fn canonical_height(png_bytes: &[u8]) -> Result<(u32, u32, Vec<u8>)> {
    let decoder = png::Decoder::new(std::io::Cursor::new(png_bytes));
    let mut reader = decoder
        .read_info()
        .map_err(|e| EngineError::Asset(e.to_string()))?;
    let size = reader
        .output_buffer_size()
        .ok_or_else(|| EngineError::Asset("surface: missing png size".into()))?;
    let mut buf = vec![0u8; size];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| EngineError::Asset(e.to_string()))?;
    let (w, h) = (info.width, info.height);

    // Collapse to one height byte per texel (the source is 8-bit grayscale, but
    // accept the common color types defensively).
    let n = (w * h) as usize;
    let height: Vec<u8> = match info.color_type {
        png::ColorType::Grayscale => buf[..n].to_vec(),
        png::ColorType::GrayscaleAlpha => buf.as_chunks::<2>().0.iter().map(|p| p[0]).collect(),
        png::ColorType::Rgb => buf.as_chunks::<3>().0.iter().map(|p| p[0]).collect(),
        png::ColorType::Rgba => buf.as_chunks::<4>().0.iter().map(|p| p[0]).collect(),
        other => {
            return Err(EngineError::Asset(format!(
                "surface: unsupported PNG color type {other:?}"
            )));
        }
    };

    let (height, w, h) = downsample_to_limit(height, w, h, MAX_TEXTURE_DIM_2D);
    Ok((w, h, height))
}
