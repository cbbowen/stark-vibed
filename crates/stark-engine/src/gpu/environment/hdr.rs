//! The Radiance RGBE (`.hdr`) decoder (§6.3).
//!
//! A file format and nothing else — no GPU, no lighting model — and the part that has
//! to be defensive: the bytes come from the frontend at runtime, so a malformed file
//! must be an error rather than a panic or a silent misread.

/// Decode a Radiance RGBE (`#?RADIANCE`, `FORMAT=32-bit_rle_rgbe`) file into a
/// linear-RGB equirectangular image (row-major, top row first). Returns
/// `(pixels, width, height)`.
///
/// Supports the new-style per-scanline RLE (the common case for ≥8px-wide images)
/// and falls back to flat RGBE quads otherwise. Errors on a malformed header.
pub(super) fn decode_hdr(bytes: &[u8]) -> Result<(Vec<[f32; 3]>, u32, u32), String> {
    let mut pos = 0usize;

    // --- Header: text lines until a blank line, then the resolution line. ---
    let line = |pos: &mut usize| -> String {
        let start = *pos;
        while *pos < bytes.len() && bytes[*pos] != b'\n' {
            *pos += 1;
        }
        let s = String::from_utf8_lossy(&bytes[start..*pos]).into_owned();
        *pos += 1; // skip '\n'
        s
    };

    let magic = line(&mut pos);
    if !magic.starts_with("#?") {
        return Err(format!("hdr: bad magic {magic:?}"));
    }
    // Consume header lines until the blank separator.
    loop {
        if pos >= bytes.len() {
            return Err("hdr: unexpected EOF in header".into());
        }
        let l = line(&mut pos);
        if l.is_empty() {
            break;
        }
    }

    // Resolution line, e.g. "-Y 512 +X 1024". We only support the standard
    // top-down, left-right orientation (`-Y h +X w`), which HDRIs use.
    let res = line(&mut pos);
    let parts: Vec<&str> = res.split_whitespace().collect();
    if parts.len() != 4 || parts[0] != "-Y" || parts[2] != "+X" {
        return Err(format!("hdr: unsupported resolution line {res:?}"));
    }
    let h: u32 = parts[1].parse().map_err(|_| "hdr: bad height")?;
    let w: u32 = parts[3].parse().map_err(|_| "hdr: bad width")?;
    // The image becomes one texture (`Environment::from_equirect`), which can be
    // neither empty nor past the device's limit — and the header alone sizes the
    // allocation below, so it is bounded before a byte of the body is read.
    let max = crate::gpu::context::MAX_TEXTURE_DIM_2D;
    if w == 0 || h == 0 || w > max || h > max {
        return Err(format!(
            "hdr: a {w}x{h} image is not 1..={max} texels a side"
        ));
    }
    let (wu, hu) = (w as usize, h as usize);
    let texels = wu.checked_mul(hu).ok_or("hdr: image too large")?;
    // The reservation below is 12 bytes a texel, sized by the header alone: a few
    // bytes claiming 8192² would reserve 805 MB before the first scanline ran short.
    let body = bytes.len().saturating_sub(pos);
    if body < min_body_len(wu, hu).ok_or("hdr: image too large")? {
        return Err(format!("hdr: {body} bytes cannot hold a {w}x{h} image"));
    }

    let mut out: Vec<[f32; 3]> = Vec::new();
    out.try_reserve_exact(texels)
        .map_err(|_| format!("hdr: no memory for a {w}x{h} image"))?;
    let mut scan = vec![[0u8; 4]; wu]; // one scanline of RGBE
    for _ in 0..hu {
        read_scanline(bytes, &mut pos, &mut scan, wu)?;
        out.extend(scan.iter().map(|&rgbe| rgbe_to_linear(rgbe)));
    }
    Ok((out, w, h))
}

/// The widths a scanline may be new-style run-length encoded at; any other is flat
/// RGBE quads.
const RLE_WIDTHS: std::ops::Range<usize> = 8..0x8000;

/// The most texels one run covers: its count byte is `128 + n`.
const MAX_RUN: usize = (u8::MAX - 128) as usize;

/// The fewest body bytes that can hold `h` scanlines of `w` texels, as
/// [`read_scanline`] reads them: behind each new-style scanline's 4-byte header, every
/// channel as runs of [`MAX_RUN`] texels at two bytes each — and four bytes a texel
/// where a scanline cannot be encoded. `None` past `usize`.
fn min_body_len(w: usize, h: usize) -> Option<usize> {
    let scanline = if RLE_WIDTHS.contains(&w) {
        4 + 4 * 2 * w.div_ceil(MAX_RUN)
    } else {
        w.checked_mul(4)?
    };
    scanline.checked_mul(h)
}

/// Read one scanline of `w` RGBE pixels into `scan`, advancing `pos`. Handles the
/// new-style RLE header (`0x02 0x02 hi lo`) per channel, else flat/old quads.
fn read_scanline(
    bytes: &[u8],
    pos: &mut usize,
    scan: &mut [[u8; 4]],
    w: usize,
) -> Result<(), String> {
    // New-style RLE is flagged by a leading 0x02 0x02 with the width in the next two
    // bytes.
    let new_rle = RLE_WIDTHS.contains(&w)
        && *pos + 4 <= bytes.len()
        && bytes[*pos] == 2
        && bytes[*pos + 1] == 2
        && ((bytes[*pos + 2] as usize) << 8 | bytes[*pos + 3] as usize) == w;

    if !new_rle {
        // Flat RGBE quads (old-style RLE — repeats flagged by R=G=B=1 — is rare
        // for modern HDRIs; we read straight quads, which covers the non-RLE case).
        for px in scan.iter_mut().take(w) {
            if *pos + 4 > bytes.len() {
                return Err("hdr: EOF in flat scanline".into());
            }
            px.copy_from_slice(&bytes[*pos..*pos + 4]);
            *pos += 4;
        }
        return Ok(());
    }
    *pos += 4; // consume the RLE scanline header

    // Four channel planes (R, G, B, E), each run-length encoded across the row.
    // `ch` indexes the *inner* `[u8; 4]` of `scan[x]`, not `scan` itself, so there is
    // no slice for an iterator rewrite to walk.
    for ch in 0..4 {
        let mut x = 0usize;
        while x < w {
            if *pos >= bytes.len() {
                return Err("hdr: EOF in RLE channel".into());
            }
            let count = bytes[*pos] as usize;
            *pos += 1;
            if count > 128 {
                // A run: (count - 128) copies of the next byte.
                let n = count - 128;
                if *pos >= bytes.len() || x + n > w {
                    return Err("hdr: bad RLE run".into());
                }
                let v = bytes[*pos];
                *pos += 1;
                for i in 0..n {
                    scan[x + i][ch] = v;
                }
                x += n;
            } else {
                // A literal: `count` raw bytes.
                if *pos + count > bytes.len() || x + count > w {
                    return Err("hdr: bad RLE literal".into());
                }
                for i in 0..count {
                    scan[x + i][ch] = bytes[*pos + i];
                }
                *pos += count;
                x += count;
            }
        }
    }
    Ok(())
}

/// RGBE → linear RGB. The shared exponent `e` scales the mantissa by `2^(e-136)`
/// (128 bias + 8 mantissa bits); `e == 0` is exact black. The `+0.5` centers each
/// mantissa in its quantization bucket.
fn rgbe_to_linear(rgbe: [u8; 4]) -> [f32; 3] {
    let e = rgbe[3];
    if e == 0 {
        return [0.0; 3];
    }
    let f = 2.0f32.powi(e as i32 - 136);
    [
        (rgbe[0] as f32 + 0.5) * f,
        (rgbe[1] as f32 + 0.5) * f,
        (rgbe[2] as f32 + 0.5) * f,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::context::MAX_TEXTURE_DIM_2D;

    #[test]
    fn decodes_bundled_studio_hdr() {
        let bytes = stark_testdata::assets::studio_hdr();
        let (pixels, w, h) = decode_hdr(&bytes).expect("decode HDR");
        assert_eq!((w, h), (1024, 512));
        assert_eq!(pixels.len(), (w * h) as usize);
        // All finite and non-negative; a studio HDR has some bright (>1) values.
        assert!(pixels.iter().flatten().all(|c| c.is_finite() && *c >= 0.0));
        let max = pixels.iter().flatten().cloned().fold(0.0f32, f32::max);
        assert!(
            max > 1.0,
            "studio HDR should contain values >1 (got max {max})"
        );
    }

    /// A Radiance header with resolution line `res`, then `body`.
    fn file(res: &str, body: &[u8]) -> Vec<u8> {
        let mut out = format!("#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n{res}\n").into_bytes();
        out.extend_from_slice(body);
        out
    }

    /// An image with no texels decodes to nothing a texture can be made of.
    #[test]
    fn an_empty_image_is_refused() {
        for res in ["-Y 0 +X 0", "-Y 0 +X 16", "-Y 16 +X 0"] {
            assert!(decode_hdr(&file(res, &[])).is_err(), "{res} decoded");
        }
    }

    /// One texel past the largest texture on either side, with a complete body — so
    /// the refusal is for the size, not for a short read.
    #[test]
    fn an_image_past_the_texture_limit_is_refused() {
        let over = MAX_TEXTURE_DIM_2D + 1;
        // Flat RGBE quads, none of which starts an RLE scanline.
        let body = [128u8; 4].repeat(over as usize);
        for res in [format!("-Y 1 +X {over}"), format!("-Y {over} +X 1")] {
            assert!(decode_hdr(&file(&res, &body)).is_err(), "{res} decoded");
        }
    }

    /// A header claiming 65536² texels is refused before anything is allocated for
    /// them — 51 GB here, and a `usize` overflow on 32-bit wasm.
    #[test]
    fn a_huge_header_is_refused_before_allocating() {
        assert!(decode_hdr(&file("-Y 65536 +X 65536", &[])).is_err());
    }

    /// A header naming more texels than its body could hold is refused before the image
    /// is reserved: a few bytes claiming 8192² would otherwise reserve 805 MB.
    #[test]
    fn a_body_too_short_for_its_header_is_refused_before_reserving() {
        let max = MAX_TEXTURE_DIM_2D;
        let err = decode_hdr(&file(&format!("-Y {max} +X {max}"), &[0; 64])).unwrap_err();
        assert!(err.contains("cannot hold"), "{err}");
    }

    /// The bound is the decoder's own least: the shortest bodies it reads — one run per
    /// channel of a new-style scanline, and flat quads at a width too narrow to encode
    /// — meet it exactly and decode, and one byte fewer is refused.
    #[test]
    fn the_shortest_bodies_the_decoder_reads_meet_the_bound() {
        let rle = [2, 2, 0, 8, 136, 128, 136, 64, 136, 32, 136, 129];
        let flat = [128, 64, 32, 129].repeat(4);
        for (res, w, body) in [("-Y 1 +X 8", 8, &rle[..]), ("-Y 1 +X 4", 4, &flat[..])] {
            assert_eq!(min_body_len(w, 1), Some(body.len()), "{res}");
            assert!(decode_hdr(&file(res, body)).is_ok(), "{res} was refused");
            let err = decode_hdr(&file(res, &body[..body.len() - 1])).unwrap_err();
            assert!(err.contains("cannot hold"), "{res}: {err}");
        }
    }
}
