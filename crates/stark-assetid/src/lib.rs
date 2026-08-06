//! What an asset *id* is — and nothing that renders one (§6.6, §6.4, §19).
//!
//! Stark names a brush shape and a canvas ground by the BLAKE3 hash of its
//! **decoded canonical form**, not its file bytes. That one decision is what lets
//! a peer be handed content it has never seen and know it received the right
//! thing, and what lets two people who encoded the same weave differently
//! converge on one id. Everything downstream — the save format's bundle, the
//! session snapshot, the blob fetch, the check in
//! `accept_surface` — rests on every build agreeing about this function.
//!
//! Which is why it lives in a crate of its own. It is the **identity contract**
//! of the file format (§19), and a contract that can only be evaluated by
//! linking a GPU renderer is one that build scripts, tools and tests cannot
//! check. Concretely, it is what lets the frontend know the id of a bundled
//! ground *before* fetching several megabytes of it.
//!
//! Two kinds, one shape:
//!
//! - a **brush shape** is coverage — luminance × alpha, so white-on-black masks
//!   and alpha-cut masks both work ([`coverage`]);
//! - a **canvas ground** is height — channel 0, because a height map's grey *is*
//!   its height and weighting the channels would tilt the ground ([`height`]).
//!
//! They decode differently and that is the whole of their difference; past this
//! crate both are a single-channel image, an id, and a canonical grayscale PNG.
//!
//! **Changing anything here changes what existing documents mean.** The id is
//! derived from the decoded field, so a change to the decode, the downsample or
//! the caps re-names content that is already on disk and already referenced by
//! saved logs. Treat the constants and the two `*_id` hashes as frozen.

use std::io::Cursor;

use serde::{Deserialize, Serialize};

/// Largest edge (px) an imported brush shape keeps; bigger images are
/// box-downsampled by an integer factor on import (§6.6).
///
/// 1024 matches the largest practical stamp footprint (brush radius caps at ~500
/// canvas px), stays well inside the device's 2048 texture limit, and — via the
/// engine's orientation-layer memory budget — keeps rotated stamps smooth (16
/// slices at 1024², vs 4 if a 2048² source were kept).
pub const MAX_SHAPE_DIM: u32 = 1024;

/// Largest edge (px) a canvas ground keeps (§6.4).
///
/// Matches the engine's `MAX_TEXTURE_DIM_2D`, and is a fixed constant rather than
/// a device query — which is load-bearing: were the downsample factor to follow
/// the adapter's real limit, the same PNG would canonicalize differently on two
/// machines and the id would stop naming one thing.
pub const MAX_GROUND_DIM: u32 = 2048;

/// Stable identity of an asset: the BLAKE3 hash of its **decoded canonical
/// form** — dimensions and texels — not of its file bytes.
///
/// `Ord` so collections of assets have one order rather than a hash map's — what
/// a save file's bundle is written in, so the same document serializes to the
/// same bytes twice running (§8).
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AssetId(pub [u8; 32]);

impl std::fmt::Debug for AssetId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // First 8 hex chars are plenty to identify in logs.
        write!(
            f,
            "AssetId({:02x}{:02x}{:02x}{:02x}…)",
            self.0[0], self.0[1], self.0[2], self.0[3]
        )
    }
}

impl AssetId {
    /// Lowercase hex, for a build script emitting a literal or a tool printing one.
    pub fn to_hex(self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }
}

#[derive(Debug, thiserror::Error)]
#[error("asset: {0}")]
pub struct AssetError(String);

pub type Result<T, E = AssetError> = std::result::Result<T, E>;

fn fail(e: impl std::fmt::Display) -> AssetError {
    AssetError(e.to_string())
}

/// A decoded, canonicalized single-channel image: what an id is taken over and
/// what the stored PNG re-encodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Canonical {
    pub width: u32,
    pub height: u32,
    /// One byte per texel, row-major.
    pub texels: Vec<u8>,
}

impl Canonical {
    /// The content id of this field: the hash of its dimensions and texels.
    ///
    /// Dimensions are hashed too, so two fields with the same bytes at different
    /// aspect ratios cannot collide.
    pub fn id(&self) -> AssetId {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.width.to_le_bytes());
        hasher.update(&self.height.to_le_bytes());
        hasher.update(&self.texels);
        AssetId(*hasher.finalize().as_bytes())
    }

    /// Re-encode as the compact grayscale PNG that is the stored and transferred
    /// form of both kinds (§8).
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut out, self.width, self.height);
            encoder.set_color(png::ColorType::Grayscale);
            encoder.set_depth(png::BitDepth::Eight);
            encoder.set_compression(png::Compression::High);
            let mut writer = encoder.write_header().map_err(fail)?;
            writer.write_image_data(&self.texels).map_err(fail)?;
        }
        Ok(out)
    }
}

/// Decode a brush shape to its canonical coverage field.
///
/// Coverage = luminance × alpha, so white-on-black masks (luminance) and
/// alpha-cut masks both work. Palette/grayscale/16-bit inputs are normalized.
pub fn coverage(png_bytes: &[u8]) -> Result<Canonical> {
    let mut decoder = png::Decoder::new(Cursor::new(png_bytes));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().map_err(fail)?;
    let mut buf = vec![
        0u8;
        reader
            .output_buffer_size()
            .ok_or_else(|| AssetError("missing size".into()))?
    ];
    let info = reader.next_frame(&mut buf).map_err(fail)?;
    buf.truncate(info.buffer_size());

    let n = (info.width * info.height) as usize;
    let mut texels = vec![0u8; n];
    let lum =
        |r: u8, g: u8, b: u8| -> u32 { (77 * r as u32 + 150 * g as u32 + 29 * b as u32) >> 8 };
    match info.color_type {
        png::ColorType::Grayscale => texels.copy_from_slice(&buf[..n]),
        png::ColorType::GrayscaleAlpha => {
            for i in 0..n {
                let g = buf[i * 2] as u32;
                let a = buf[i * 2 + 1] as u32;
                texels[i] = (g * a / 255) as u8;
            }
        }
        png::ColorType::Rgb => {
            for i in 0..n {
                texels[i] = lum(buf[i * 3], buf[i * 3 + 1], buf[i * 3 + 2]) as u8;
            }
        }
        png::ColorType::Rgba => {
            for i in 0..n {
                let l = lum(buf[i * 4], buf[i * 4 + 1], buf[i * 4 + 2]);
                let a = buf[i * 4 + 3] as u32;
                texels[i] = (l * a / 255) as u8;
            }
        }
        // `EXPAND` above turns a palette into RGB(A), so reaching here means the
        // transformation did not apply — not that indexed input is rejected.
        png::ColorType::Indexed => return Err(AssetError("indexed PNG not expanded".into())),
    }
    Ok(downsample(texels, info.width, info.height, MAX_SHAPE_DIM))
}

/// Decode a canvas ground to its canonical height field.
///
/// Channel 0, not luminance — a height map's grey *is* its height, so an RGB
/// source carries it in red and weighting the channels would tilt the ground.
pub fn height(png_bytes: &[u8]) -> Result<Canonical> {
    let decoder = png::Decoder::new(Cursor::new(png_bytes));
    let mut reader = decoder.read_info().map_err(fail)?;
    let size = reader
        .output_buffer_size()
        .ok_or_else(|| AssetError("surface: missing png size".into()))?;
    let mut buf = vec![0u8; size];
    let info = reader.next_frame(&mut buf).map_err(fail)?;
    let (w, h) = (info.width, info.height);

    // Collapse to one height byte per texel (the source is 8-bit grayscale, but
    // accept the common color types defensively).
    let n = (w * h) as usize;
    let texels: Vec<u8> = match info.color_type {
        png::ColorType::Grayscale => buf[..n].to_vec(),
        png::ColorType::GrayscaleAlpha => buf.as_chunks::<2>().0.iter().map(|p| p[0]).collect(),
        png::ColorType::Rgb => buf.as_chunks::<3>().0.iter().map(|p| p[0]).collect(),
        png::ColorType::Rgba => buf.as_chunks::<4>().0.iter().map(|p| p[0]).collect(),
        other => {
            return Err(AssetError(format!(
                "surface: unsupported PNG color type {other:?}"
            )));
        }
    };
    Ok(downsample(texels, w, h, MAX_GROUND_DIM))
}

/// Box-downsample a single-channel image by the smallest integer factor that
/// brings both edges within `limit`. An integer factor keeps a tileable texture
/// tileable; `factor == 1` returns the input unchanged.
///
/// Shared by both kinds because both do the same thing for the same reason: each
/// is capped **before** it is hashed, so the id names the canonical form and
/// reloading the stored PNG lands on the same id (§6.6, §6.4).
fn downsample(src: Vec<u8>, w: u32, h: u32, limit: u32) -> Canonical {
    let factor = w.div_ceil(limit).max(h.div_ceil(limit)).max(1);
    if factor == 1 {
        return Canonical {
            width: w,
            height: h,
            texels: src,
        };
    }
    let (nw, nh) = (w / factor, h / factor);
    let area = factor * factor;
    let mut out = vec![0u8; (nw * nh) as usize];
    for y in 0..nh {
        for x in 0..nw {
            let mut sum = 0u32;
            for dy in 0..factor {
                for dx in 0..factor {
                    let sx = x * factor + dx;
                    let sy = y * factor + dy;
                    sum += src[(sy * w + sx) as usize] as u32;
                }
            }
            out[(y * nw + x) as usize] = (sum / area) as u8;
        }
    }
    Canonical {
        width: nw,
        height: nh,
        texels: out,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fixed, non-trivial RGBA source: the two kinds read it differently, and
    /// nothing about it is allowed to change.
    fn fixture() -> Vec<u8> {
        let mut out = Vec::new();
        let mut px = Vec::with_capacity(37 * 23 * 4);
        for y in 0u32..23 {
            for x in 0u32..37 {
                px.extend_from_slice(&[
                    (x * 7 + y * 3) as u8,
                    ((x * 11) ^ (y * 5)) as u8,
                    (x + y * 13) as u8,
                    ((x * y) % 251) as u8,
                ]);
            }
        }
        {
            let mut enc = png::Encoder::new(&mut out, 37, 23);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Eight);
            let mut w = enc.write_header().expect("header");
            w.write_image_data(&px).expect("data");
        }
        out
    }

    fn gray_png(w: u32, h: u32, f: impl Fn(u32, u32) -> u8) -> Vec<u8> {
        let mut texels = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                texels.push(f(x, y));
            }
        }
        Canonical {
            width: w,
            height: h,
            texels,
        }
        .encode()
        .expect("encode")
    }

    /// The property the whole design rests on: re-encoding a canonical field and
    /// decoding it again must land on the same id, or a document's own bundle
    /// would stop matching the ids its log references.
    #[test]
    fn a_canonical_field_round_trips_to_the_same_id() {
        for limit in [MAX_SHAPE_DIM, MAX_GROUND_DIM] {
            let src = gray_png(64, 48, |x, y| (x * 3 + y * 5) as u8);
            let once = if limit == MAX_SHAPE_DIM {
                coverage(&src)
            } else {
                height(&src)
            }
            .expect("decode");
            let twice = if limit == MAX_SHAPE_DIM {
                coverage(&once.encode().expect("re-encode"))
            } else {
                height(&once.encode().expect("re-encode"))
            }
            .expect("decode again");
            assert_eq!(once.id(), twice.id());
            assert_eq!(once, twice);
        }
    }

    /// Capping happens before hashing, so an oversized source and its stored form
    /// are the same content — otherwise a reload would mint a second id for one
    /// image and a peer would fetch bytes it already had.
    #[test]
    fn an_oversized_shape_is_capped_before_it_is_hashed() {
        let src = gray_png(MAX_SHAPE_DIM * 2, MAX_SHAPE_DIM * 2, |x, y| {
            ((x ^ y) & 0xff) as u8
        });
        let capped = coverage(&src).expect("decode");
        assert_eq!(capped.width, MAX_SHAPE_DIM);
        assert_eq!(
            capped.id(),
            coverage(&capped.encode().expect("encode"))
                .expect("decode")
                .id()
        );
    }

    /// **The ids these functions produce are the file format.** A change to the
    /// decode, the downsample or the hash re-names content that is already on
    /// disk and already referenced by saved logs — and would do it silently,
    /// because every other test derives its ids at runtime on both sides and so
    /// agrees with itself no matter what the derivation became.
    ///
    /// These two literals are the only thing in the workspace that would notice.
    /// Changing them is changing the format (§19), not fixing a test.
    #[test]
    fn the_derivation_is_frozen() {
        let src = fixture();
        assert_eq!(
            coverage(&src).expect("coverage").id().to_hex(),
            "2c30aae5d574d125a86a68dbf103d4581c87f65fb1f1f8f1813cde994dcba3c6",
        );
        assert_eq!(
            height(&src).expect("height").id().to_hex(),
            "40d88cda34a84328109351e49fb10116a6150e662e5ed4e42ad7b44266515054",
        );
    }

    /// Dimensions are hashed, so the same texels at a different aspect ratio are
    /// different content.
    #[test]
    fn dimensions_take_part_in_the_id() {
        let texels = vec![7u8; 24];
        let wide = Canonical {
            width: 12,
            height: 2,
            texels: texels.clone(),
        };
        let tall = Canonical {
            width: 2,
            height: 12,
            texels,
        };
        assert_ne!(wide.id(), tall.id());
    }

    /// A ground reads channel 0 and a shape reads luminance × alpha — the one
    /// difference between the two, and the reason the receiver has to be told
    /// which kind it is being handed (§6.4, §6.6).
    #[test]
    fn coverage_and_height_read_a_source_differently() {
        // Red-only RGB: channel 0 is full, luminance is not.
        let mut out = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut out, 2, 2);
            enc.set_color(png::ColorType::Rgb);
            enc.set_depth(png::BitDepth::Eight);
            let mut w = enc.write_header().expect("header");
            w.write_image_data(&[255, 0, 0].repeat(4)).expect("data");
        }
        assert_eq!(height(&out).expect("height").texels, vec![255; 4]);
        assert_eq!(coverage(&out).expect("coverage").texels, vec![76; 4]);
    }
}
