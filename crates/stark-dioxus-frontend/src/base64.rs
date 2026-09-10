//! Standard base64 with padding (RFC 4648 §4), for the `data:` URLs the chrome puts its
//! thumbnails and swatches in. Encode only: nothing here reads one back but a test.

/// `data` as padded standard base64.
pub fn encode(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let sextet = |n: u32, shift: u32| char::from(ALPHABET[((n >> shift) & 63) as usize]);

    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let byte = |i: usize| u32::from(chunk.get(i).copied().unwrap_or(0));
        let n = (byte(0) << 16) | (byte(1) << 8) | byte(2);
        out.push(sextet(n, 18));
        out.push(sextet(n, 12));
        out.push(if chunk.len() > 1 { sextet(n, 6) } else { '=' });
        out.push(if chunk.len() > 2 { sextet(n, 0) } else { '=' });
    }
    out
}

/// The inverse of [`encode`], for the tests that read a `data:` URL back. Stops at the
/// first `=`.
///
/// Spelled as ranges rather than as a table inverted from [`encode`]'s alphabet, so a
/// round trip compares two independent statements of it.
#[cfg(test)]
pub fn decode(text: &str) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0u32);
    for &c in text.as_bytes() {
        let value = match c {
            b'=' => break,
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return Err(format!("{:?} is not a base64 character", char::from(c))),
        };
        acc = (acc << 6) | u32::from(value);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RFC 4648 §10's vectors, which cover no padding, one `=` and two, and one more for
    /// the two characters past the alphanumerics, which no ASCII text reaches.
    #[test]
    fn encode_matches_the_rfc_vectors() {
        let vectors: [(&[u8], &str); 8] = [
            (b"", ""),
            (b"f", "Zg=="),
            (b"fo", "Zm8="),
            (b"foo", "Zm9v"),
            (b"foob", "Zm9vYg=="),
            (b"fooba", "Zm9vYmE="),
            (b"foobar", "Zm9vYmFy"),
            (&[0xfb, 0xff], "+/8="),
        ];
        for (plain, encoded) in vectors {
            assert_eq!(encode(plain), encoded, "encoding {plain:?}");
        }
    }

    /// Every byte value, at every remainder a length can leave, survives the round trip.
    #[test]
    fn a_byte_ramp_round_trips_at_every_remainder() {
        let ramp: Vec<u8> = (0..=u8::MAX).collect();
        for len in [0, 1, 2, 3, 4, 5, 254, 255, 256] {
            let data = &ramp[..len];
            assert_eq!(decode(&encode(data)).as_deref(), Ok(data), "{len} bytes");
        }
    }

    #[test]
    fn decode_refuses_a_character_outside_the_alphabet() {
        assert!(
            decode("Zm9v!").is_err(),
            "`!` is not in the alphabet, so the text is not base64"
        );
    }
}
