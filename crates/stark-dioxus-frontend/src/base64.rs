//! Standard base64 with padding (RFC 4648 §4), for the `data:` URLs the chrome puts its
//! thumbnails and swatches in. Encode only: nothing reads one back.

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
}
