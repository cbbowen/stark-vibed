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
//! Three kinds, two shapes:
//!
//! - a **brush shape** is coverage — luminance × alpha, so white-on-black masks
//!   and alpha-cut masks both work ([`coverage`]);
//! - a **canvas ground** is height — channel 0, because a height map's grey *is*
//!   its height and weighting the channels would tilt the ground ([`height`]);
//! - a **picture** is all four channels, kept ([`picture`]) — an image placed into
//!   the document as paint (§23), where the other two are read *for* something and
//!   this one is the thing itself.
//!
//! The first two decode differently and that is the whole of their difference;
//! past this crate both are a single-channel image, an id, and a canonical
//! grayscale PNG. A picture is the same three sentences over four channels, which
//! is why it has a type of its own rather than a flag on [`Canonical`]: the
//! difference is not what the bytes *mean* but how many there are per texel, and
//! that is the one thing every consumer indexes by.
//!
//! **Changing anything here changes what existing documents mean.** The id is
//! derived from the decoded field, so a change to the decode, the downsample or
//! the caps re-names content that is already on disk and already referenced by
//! saved logs. Treat the constants and all three `id` derivations as frozen.
//!
//! **And the three dimension caps are a one-way ratchet: they may only ever go up.**
//! They are enforced on the way *in*, so lowering one refuses content that a bundle
//! already carries — and a document whose content will not decode does not open at
//! all, since replay refuses rather than substituting a stand-in
//! (`DocError::MissingContent`, §6.4). That makes them part of the §19 promise that
//! old files keep loading, which is easy to miss because they read like tuning
//! constants, and because what one of them bounds downstream (`stark-model`'s
//! `MAX_IMAGE_TILES`) is *derived* from a cap rather than chosen. Raising one is
//! free: it only admits content no older file contains.

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

/// Largest edge (px) a placed picture keeps (§23).
///
/// Sized by what a picture *is* rather than by what a texture can be, because
/// nothing binds one as a texture: the tiles it becomes are built on the CPU
/// (`stark-engine`'s `gpu::place`). What it bounds is memory — 4096 clears a 4K
/// screen capture on the long edge with room over and holds the decoded worst case
/// to 67 MB of RGBA, the same order as the ground height maps a bundle already
/// carries (§8) — and, through it, how much of a stranger's PNG this process will
/// expand.
pub const MAX_PICTURE_DIM: u32 = 4096;

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
#[derive(
    Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, carbonite::Schema,
)]
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

impl std::fmt::Display for AssetId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_hex())
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

/// A decoded, canonicalized **four**-channel image: what a placed picture's id is
/// taken over, and what its stored PNG re-encodes (§23).
///
/// Straight (un-premultiplied) RGBA8 in sRGB, top-left origin, tightly packed —
/// what a PNG stores and what a browser's `getImageData` returns, so nothing has to
/// be undone on either side of the boundary. Alpha here is coverage *of the source
/// image*; what it becomes in the document is paint, which is the engine's business
/// and not this crate's.
///
/// Beside [`Canonical`] rather than inside it. The two answer the same three
/// questions — what are the texels, what is the id, what is stored — and differ in
/// the one thing every consumer indexes by, so a shared type would put a channel
/// count into the hash of the two derivations §19 calls frozen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Picture {
    pub width: u32,
    pub height: u32,
    /// `width * height * 4` bytes, row-major, no padding.
    pub pixels: Vec<u8>,
}

impl Picture {
    /// The content id of this picture: the hash of its dimensions and pixels.
    ///
    /// The same shape as [`Canonical::id`], and deliberately not a shared helper:
    /// these are two frozen derivations, and a function that computed both would be
    /// a place where changing one changes the other (§19).
    pub fn id(&self) -> AssetId {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.width.to_le_bytes());
        hasher.update(&self.height.to_le_bytes());
        hasher.update(&self.pixels);
        AssetId(*hasher.finalize().as_bytes())
    }

    /// The RGBA bytes at `(x, y)`, or fully transparent outside the picture.
    ///
    /// Answering *outside* rather than panicking is what lets the tile builder walk a
    /// whole tile texture — apron included — with no bounds test of its own at every
    /// texel, and it is the honest answer: past the picture's edge there is no paint.
    ///
    /// `i64` where the dimensions are `u32`, and not as defensiveness: the caller's
    /// index is a *tile* origin less a placement, and a tile index is an `i32` of
    /// tiles, so the product overruns an `i32` of pixels long before the tile grid
    /// runs out. Taking the difference in a type that cannot wrap is what makes the
    /// far reaches of the canvas behave like the near ones.
    pub fn sample(&self, x: i64, y: i64) -> [u8; 4] {
        if x < 0 || y < 0 || x >= i64::from(self.width) || y >= i64::from(self.height) {
            return [0; 4];
        }
        let i = ((y as usize) * self.width as usize + x as usize) * 4;
        [
            self.pixels[i],
            self.pixels[i + 1],
            self.pixels[i + 2],
            self.pixels[i + 3],
        ]
    }

    /// Re-encode as the RGBA PNG that is the stored and transferred form (§8, §23).
    ///
    /// `Fast` where the two single-channel kinds ask for `High`. Saving is latency
    /// the artist waits through — `io`'s own argument for deflate level 6 — and the
    /// sizes are not comparable: a mask is a few hundred kilobytes of smooth coverage
    /// where this is megabytes of photograph, on which the deeper search spends a
    /// large multiple of the time for a fraction of a percent.
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut out, self.width, self.height);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            encoder.set_compression(png::Compression::Fast);
            let mut writer = encoder.write_header().map_err(fail)?;
            writer.write_image_data(&self.pixels).map_err(fail)?;
        }
        Ok(out)
    }
}

/// Decode a **picture** to its canonical RGBA field (§23).
///
/// All four channels, kept: this is the one kind that is not read *for* something —
/// a shape is read for coverage and a ground for height, while a picture is the
/// thing itself, and dropping any channel of it would be dropping part of the
/// image someone imported.
///
/// **Bounded before it is expanded**, which is `io`'s rule and this crate's most
/// exposed application of it: a picture arrives from a file or a peer as readily as
/// from a clipboard, and the PNG says nothing about how much memory it decodes to.
/// The header's dimensions are read first, and a source past four times
/// [`MAX_PICTURE_DIM`] on either edge is refused rather than decoded and then
/// downsampled — what is left is capped by the box filter, exactly as the two kinds
/// above are, so a re-encode lands on the same id.
pub fn picture(png_bytes: &[u8]) -> Result<Picture> {
    let mut decoder = png::Decoder::new(Cursor::new(png_bytes));
    // Palette and 16-bit sources normalize to 8-bit RGB(A) — what varies between
    // encoders is not what the picture is.
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().map_err(fail)?;
    let (w, h) = {
        let info = reader.info();
        (info.width, info.height)
    };
    if w == 0 || h == 0 {
        return Err(AssetError("picture: empty".into()));
    }
    // The ceiling on what is *decoded*, above which the box filter is not offered:
    // downsampling holds the stored form small but has to materialize the source
    // first, so an unbounded one is an unbounded allocation.
    const DECODE_LIMIT: u32 = MAX_PICTURE_DIM * 4;
    if w > DECODE_LIMIT || h > DECODE_LIMIT {
        return Err(AssetError(format!(
            "picture: {w}\u{00D7}{h} is past the {DECODE_LIMIT} px decode limit"
        )));
    }

    let size = reader
        .output_buffer_size()
        .ok_or_else(|| AssetError("picture: missing png size".into()))?;
    let mut buf = vec![0u8; size];
    let info = reader.next_frame(&mut buf).map_err(fail)?;

    let n = (w as usize) * (h as usize);
    let mut pixels = vec![0u8; n * 4];
    let quads = pixels.as_chunks_mut::<4>().0;
    match info.color_type {
        png::ColorType::Rgba => pixels.copy_from_slice(&buf[..n * 4]),
        png::ColorType::Rgb => {
            for (out, src) in quads.iter_mut().zip(buf.as_chunks::<3>().0) {
                *out = [src[0], src[1], src[2], 255];
            }
        }
        png::ColorType::Grayscale => {
            for (out, g) in quads.iter_mut().zip(&buf[..n]) {
                *out = [*g, *g, *g, 255];
            }
        }
        png::ColorType::GrayscaleAlpha => {
            for (out, src) in quads.iter_mut().zip(buf.as_chunks::<2>().0) {
                *out = [src[0], src[0], src[0], src[1]];
            }
        }
        // `EXPAND` above turns a palette into RGB(A), so reaching here means the
        // transformation did not apply — not that indexed input is rejected.
        png::ColorType::Indexed => return Err(AssetError("indexed PNG not expanded".into())),
    }
    Ok(downsample_rgba(pixels, w, h, MAX_PICTURE_DIM))
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

/// Box-downsample RGBA by the smallest integer factor that brings both edges within
/// `limit` — [`downsample`]'s four-channel sibling, capping a picture before it is
/// hashed for that function's reason.
///
/// **Averaged premultiplied, and stored straight**, which is the one thing this does
/// that the single-channel version has no occasion to. A transparent texel still has
/// *some* color in its RGB, and PNG encoders routinely leave that black; averaging the
/// channels as they stand drags the color of every edge towards whatever is stored
/// under the transparent part, which on a cut-out is a dark fringe all the way round —
/// the halo that gives a pasted image away. Weighting by alpha is what makes the
/// average an average of the paint that is actually there.
fn downsample_rgba(src: Vec<u8>, w: u32, h: u32, limit: u32) -> Picture {
    let factor = w.div_ceil(limit).max(h.div_ceil(limit)).max(1);
    if factor == 1 {
        return Picture {
            width: w,
            height: h,
            pixels: src,
        };
    }
    let (nw, nh) = (w / factor, h / factor);
    let area = factor * factor;
    let mut out = vec![0u8; (nw * nh * 4) as usize];
    for y in 0..nh {
        for x in 0..nw {
            let (mut acc, mut alpha) = ([0u32; 3], 0u32);
            for dy in 0..factor {
                for dx in 0..factor {
                    let i = (((y * factor + dy) * w + (x * factor + dx)) * 4) as usize;
                    let a = src[i + 3] as u32;
                    for (c, sum) in acc.iter_mut().enumerate() {
                        *sum += src[i + c] as u32 * a;
                    }
                    alpha += a;
                }
            }
            let o = ((y * nw + x) * 4) as usize;
            // Back to straight alpha. Where the whole cell is transparent there is no
            // color to recover and none to invent, so it stays fully clear — which is
            // what `checked_div`'s `None` says here, rather than a guard beside it.
            for c in 0..3 {
                out[o + c] = acc[c].checked_div(alpha).unwrap_or(0) as u8;
            }
            out[o + 3] = (alpha / area) as u8;
        }
    }
    Picture {
        width: nw,
        height: nh,
        pixels: out,
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
    /// These three literals are the only thing in the workspace that would notice.
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
        assert_eq!(
            picture(&src).expect("picture").id().to_hex(),
            "c4fef97a2d2feb1a03a06108589d32d25bdd2201d5a9bf1af1dd57c8db51c005",
        );
    }

    /// A picture round-trips to the same id, which is the property the whole design
    /// rests on for this kind exactly as it does for the other two: a document's own
    /// bundle has to keep matching the ids its log references.
    #[test]
    fn a_canonical_picture_round_trips_to_the_same_id() {
        let once = picture(&fixture()).expect("decode");
        let twice = picture(&once.encode().expect("re-encode")).expect("decode again");
        assert_eq!(once.id(), twice.id());
        assert_eq!(once, twice);
    }

    /// A picture keeps **all four channels**, which is the whole of what separates it
    /// from the two kinds beside it — they are read *for* something and this one is
    /// the thing itself.
    #[test]
    fn a_picture_keeps_what_the_other_two_kinds_read_past() {
        // Red at half alpha: channel 0 is full, luminance is not, and the alpha is a
        // fact neither of the others carries at all.
        let mut src = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut src, 2, 2);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Eight);
            let mut w = enc.write_header().expect("header");
            w.write_image_data(&[255, 0, 0, 128].repeat(4))
                .expect("data");
        }
        let p = picture(&src).expect("picture");
        assert_eq!(p.sample(0, 0), [255, 0, 0, 128]);
        // …where the other two collapse it to one byte each, differently.
        assert_eq!(height(&src).expect("height").texels[0], 255);
        assert_eq!(coverage(&src).expect("coverage").texels[0], 38);
    }

    /// Outside a picture there is no paint, at any distance and on either axis — the
    /// property that lets the tile builder walk a whole apron'd texture without a
    /// bounds test of its own (§23).
    #[test]
    fn sampling_outside_a_picture_is_transparent() {
        let p = picture(&fixture()).expect("picture");
        let (w, h) = (i64::from(p.width), i64::from(p.height));
        for (x, y) in [
            (-1, 0),
            (0, -1),
            (w, 0),
            (0, h),
            (i64::MIN, i64::MIN),
            (i64::MAX, i64::MAX),
        ] {
            assert_eq!(p.sample(x, y), [0; 4], "({x}, {y})");
        }
        assert_ne!(p.sample(w - 1, h - 1), [0; 4], "…and inside it, there is");
    }

    /// **A downsampled picture must not grow a dark fringe.**
    ///
    /// Averaging RGB as it stands drags every edge towards whatever colour is stored
    /// under the transparent part — which encoders routinely leave black — and the
    /// result is the halo that gives a pasted cut-out away. Weighting by alpha is what
    /// makes the average an average of the paint that is actually there.
    ///
    /// The fixture is the worst case on purpose: opaque white beside transparent
    /// *black*, so an unweighted average would come back mid-grey and a weighted one
    /// stays white.
    #[test]
    fn downsampling_a_cutout_does_not_darken_its_edge() {
        let (w, h) = (4u32, 2u32);
        let mut pixels = Vec::new();
        for _ in 0..h {
            // Two opaque white, two transparent black — one output texel per pair.
            pixels.extend_from_slice(&[255, 255, 255, 255]);
            pixels.extend_from_slice(&[0, 0, 0, 0]);
            pixels.extend_from_slice(&[255, 255, 255, 255]);
            pixels.extend_from_slice(&[0, 0, 0, 0]);
        }
        let small = downsample_rgba(pixels, w, h, 2);
        assert_eq!((small.width, small.height), (2, 1));
        let texel = small.sample(0, 0);
        assert_eq!(
            [texel[0], texel[1], texel[2]],
            [255, 255, 255],
            "the colour is the colour of the paint that was there, not of the hole",
        );
        assert_eq!(texel[3], 127, "…and the coverage is what halved");
    }

    /// A cell with nothing in it stays clear rather than inventing a colour to
    /// average — the division the weighting cannot do.
    #[test]
    fn downsampling_an_empty_cell_stays_empty() {
        let small = downsample_rgba(vec![0u8; 4 * 4 * 4], 4, 4, 2);
        assert_eq!(small.pixels, vec![0u8; 2 * 2 * 4]);
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
