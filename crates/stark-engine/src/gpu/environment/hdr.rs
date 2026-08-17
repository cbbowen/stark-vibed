//! The Radiance RGBE (`.hdr`) decoder (§6.3).
//!
//! A file format and nothing else — no GPU, no lighting model. Split out because it
//! is the one part of an environment that is about *bytes on disk* rather than about
//! how a painting is lit, and because it is the part that has to be defensive: the
//! bytes come from the frontend at runtime and a malformed file must be an error
//! rather than a panic or a silent misread.

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
    let (wu, hu) = (w as usize, h as usize);

    let mut out = vec![[0.0f32; 3]; wu * hu];
    let mut scan = vec![[0u8; 4]; wu]; // one scanline of RGBE
    for y in 0..hu {
        read_scanline(bytes, &mut pos, &mut scan, wu)?;
        let row = &mut out[y * wu..(y + 1) * wu];
        for (px, rgbe) in row.iter_mut().zip(scan.iter()) {
            *px = rgbe_to_linear(*rgbe);
        }
    }
    Ok((out, w, h))
}

/// Read one scanline of `w` RGBE pixels into `scan`, advancing `pos`. Handles the
/// new-style RLE header (`0x02 0x02 hi lo`) per channel, else flat/old quads.
fn read_scanline(
    bytes: &[u8],
    pos: &mut usize,
    scan: &mut [[u8; 4]],
    w: usize,
) -> Result<(), String> {
    // New-style RLE is only used for widths in [8, 0x7fff] and is flagged by a
    // leading 0x02 0x02 with the width in the next two bytes.
    let new_rle = (8..0x8000).contains(&w)
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
    // no slice for clippy's iterator rewrite to walk.
    #[allow(clippy::needless_range_loop)]
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
}
