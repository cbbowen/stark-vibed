//! The images this build ships, and the ones a person brings in (§6.4, §6.6, §11.2 N7).
//!
//! Two halves, and only the first is this frontend's own work. **What** ships, what
//! each is called, what an imported image becomes, and how a library is kept are all
//! `stark_ui::assets`. What is here is the two things a native binary answers
//! differently from a browser: where the shipped bytes come from, and who decodes.
//!
//! # The shipped images are in the binary
//!
//! `include_bytes!`, all 17 MB of them — and that is the *opposite* of the web
//! frontend's decision, for the reason that decides it. A wasm binary is downloaded
//! on every visit, so keeping four large PNGs out of it is the difference between a
//! page that loads and one that does not; that is why `asset!` fetches them at
//! runtime and why the id table exists to break the circularity that creates.
//!
//! A native binary is installed once. Fetching would mean an install layout to get
//! right, a missing-file failure mode, and a `cargo run` that cannot find its own
//! canvas textures — all to save disk on a machine that has already spent it. So the
//! bytes ride along, resolving is a slice, and this frontend's `unresolved_content`
//! problem from N5 goes away: it can open a lean file that names a shipped substrate,
//! because it has one.
//!
//! The id table is still `stark_ui::assets`' and still computed at build time.
//! Not because it is needed to avoid a fetch here — there is no fetch — but because a
//! peer must be told what this build can resolve *before* it hashes 17 MB to find
//! out (§12.4).
//!
//! # Decoding
//!
//! The `image` crate, cut to the raster formats worth the compile. It replaces the
//! browser in exactly one step of four: decode, resample, decide, encode. The middle
//! pair is where the two frontends genuinely differ (two resamplers, so an oversized
//! file can land on two ids — `stark_ui::assets` states it), and the last two are
//! shared.

use stark_ui::assets::{self, Decoded};

/// Every image this build ships, by the catalog path that names it.
///
/// Spelled here rather than derived, so a file that is not in the catalog is a
/// compile error at the `include_bytes!` rather than a card that draws nothing — and
/// checked against the catalog by the tests below.
const BUNDLED: &[(&str, &[u8])] = &[
    (
        "shape/Worn_Bristles.png",
        include_bytes!("../../stark-dioxus-frontend/assets/shape/Worn_Bristles.png"),
    ),
    (
        "shape/Flat.png",
        include_bytes!("../../stark-dioxus-frontend/assets/shape/Flat.png"),
    ),
    (
        "shape/Pencil.png",
        include_bytes!("../../stark-dioxus-frontend/assets/shape/Pencil.png"),
    ),
    (
        "substrate/Linen.png",
        include_bytes!("../../stark-dioxus-frontend/assets/substrate/Linen.png"),
    ),
    (
        "substrate/Rough.png",
        include_bytes!("../../stark-dioxus-frontend/assets/substrate/Rough.png"),
    ),
];

/// The shipped bytes at `path`.
pub fn bundled(path: &str) -> Option<&'static [u8]> {
    BUNDLED.iter().find(|(p, _)| *p == path).map(|(_, b)| *b)
}

/// The shipped bytes a content id names, if this build has them.
///
/// The whole of what a web frontend's `fetch` does, minus the fetching — which is the
/// module note in one function.
pub fn bytes_for(id: stark_model::AssetId) -> Option<&'static [u8]> {
    bundled(assets::shipped_at(id)?.path?)
}

/// The catalog row for each shipped shape, paired with its bytes — for the startup
/// import, which wants both.
pub fn shipped_shape_files() -> impl Iterator<Item = (&'static assets::Shipped, &'static [u8])> {
    assets::SHIPPED_SHAPES
        .iter()
        .filter_map(|row| Some((row, bundled(row.path?)?)))
}

/// The two stamps the shipped preset table needs, by content id.
///
/// **Known without importing anything**, because the ids were hashed at build time —
/// which is the whole of what this frontend gains by carrying the bytes. The web
/// app's equivalent cannot answer until its fetches land, and says so: its two stamp
/// presets stand on the round tip until then.
pub fn builtin_shapes() -> stark_ui::presets::BuiltinShapes {
    let stamp = |name: &str| {
        assets::shipped(name)
            .and_then(|row| row.path)
            .and_then(assets::shipped_id)
            .map(stark_model::document::BrushShape::Stamp)
            .unwrap_or_default()
    };
    stark_ui::presets::BuiltinShapes {
        bristles: stamp(assets::BRISTLES),
        pencil: stamp(assets::PENCIL),
    }
}

/// Decode an image file to straight RGBA8, no larger than `cap` on its longest edge.
///
/// The size comes from `stark_ui::assets::fit` rather than from arithmetic here,
/// so the two frontends ask their own resampler for the same size even though the
/// resamplers differ.
///
/// **Lanczos3**, which is what a downscale of a texture or a scanned brush wants: a
/// box filter would alias a woven substrate into a moiré, and the nearest-neighbour
/// default would do worse. A decode is a once-per-import cost, so the good filter is
/// affordable.
pub fn decode(bytes: &[u8], cap: u32) -> Result<Decoded, String> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| format!("not an image this build can decode: {e}"))?;
    let (w, h) = (img.width(), img.height());
    if w == 0 || h == 0 {
        return Err("the image is empty".to_string());
    }
    let (width, height) = assets::fit(w, h, cap);
    let rgba = if (width, height) == (w, h) {
        img.into_rgba8()
    } else {
        image::imageops::resize(
            &img.into_rgba8(),
            width,
            height,
            image::imageops::FilterType::Lanczos3,
        )
    };
    Ok(Decoded {
        width,
        height,
        rgba: rgba.into_raw(),
    })
}

/// A file's bytes as a brush shape's canonical PNG, and whether the ink was inverted.
pub fn as_shape(bytes: &[u8]) -> Result<(Vec<u8>, bool), String> {
    assets::shape_png(decode(bytes, assets::SHAPE_CAP)?)
}

/// A file's bytes as a canvas substrate's canonical PNG.
pub fn as_substrate(bytes: &[u8]) -> Result<Vec<u8>, String> {
    assets::substrate_png(decode(bytes, assets::SUBSTRATE_CAP)?)
}

/// A card's picture as RGBA8, ready for a texture.
///
/// The texels and their reading are `stark_ui::assets::card`; this is the one
/// step that is a toolkit's — the web frontend encodes the same numbers as a data URL
/// instead (`crate::cards` over there).
///
/// White ink with the coverage in alpha for a stamp, so the panel shows through where
/// it lays nothing; opaque grey for a substrate, which has no gaps.
pub fn card_rgba<K: assets::Kind>(png: &[u8]) -> Option<(u32, u32, Vec<u8>)> {
    let card = assets::card::<K>(png)?;
    let rgba = card
        .texels
        .iter()
        .flat_map(|&t| match card.ink {
            assets::Ink::Coverage => [255, 255, 255, t],
            assets::Ink::Height => [t, t, t, 255],
        })
        .collect();
    Some((card.width, card.height, rgba))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every catalog row that names a file has that file in the binary. A row without
    /// its bytes would be a card that draws nothing and a preset that never resolves
    /// — which is what the web frontend's fetch can *legitimately* produce and this
    /// one cannot, since the bytes are not going anywhere.
    #[test]
    fn every_shipped_row_has_its_bytes() {
        for row in assets::SHIPPED_SHAPES
            .iter()
            .chain(assets::SHIPPED_SUBSTRATES)
        {
            let Some(path) = row.path else { continue };
            assert!(bundled(path).is_some(), "{path} is named but not bundled");
        }
    }

    /// And nothing is bundled that no row offers — dead weight in a binary that pays
    /// for it by the megabyte.
    #[test]
    fn nothing_is_bundled_that_no_row_offers() {
        let claimed: Vec<&str> = assets::SHIPPED_SHAPES
            .iter()
            .chain(assets::SHIPPED_SUBSTRATES)
            .filter_map(|s| s.path)
            .collect();
        for (path, _) in BUNDLED {
            assert!(claimed.contains(path), "{path} is bundled but unreachable");
        }
    }

    /// **The bytes in the binary are the bytes the id table was built from.** Both
    /// come from the same directory, but through different mechanisms — a build
    /// script's `read` and a macro's `include_bytes!` — and if they ever came apart
    /// this build would promise a peer content it could not produce (§12.4).
    #[test]
    fn the_bundled_bytes_hash_to_the_ids_this_build_promises() {
        for (path, bytes) in BUNDLED {
            let want = assets::shipped_id(path).expect("every bundled file is hashed");
            let got = if path.starts_with("shape/") {
                stark_assetid::coverage(bytes).map(|c| c.id())
            } else {
                stark_assetid::height(bytes).map(|c| c.id())
            }
            .expect("a shipped asset decodes");
            assert_eq!(got, want, "{path} in the binary is not what was hashed");
        }
    }

    /// A decode caps the longest edge and leaves a small image alone, which is what
    /// keeps a 64-px stamp a 64-px stamp.
    #[test]
    fn a_decode_is_capped_but_never_enlarged() {
        let bytes = bundled("shape/Pencil.png").expect("the pencil ships");
        let small = decode(bytes, 32).expect("the pencil decodes");
        assert!(small.width <= 32 && small.height <= 32);
        assert_eq!(small.rgba.len(), (small.width * small.height * 4) as usize);
    }

    /// Anything that is not an image is refused in words rather than by a panic.
    #[test]
    fn a_file_that_is_not_an_image_is_refused() {
        assert!(as_shape(b"this is not a png").is_err());
        assert!(as_substrate(b"").is_err());
    }
}
