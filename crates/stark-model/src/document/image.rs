//! Where a placed picture lands (§23).
//!
//! The picture *itself* is [`stark_assetid::Picture`] — decode, cap, hash, the same
//! identity contract a brush shape and a canvas substrate pass through (§19). What is
//! left here is the half that is a fact about the **document**: the geometry of a
//! placement, and the bound on how much of the canvas one can rewrite.
//!
//! The split is the crate's own rule read straight off (§2, `lib.rs`): an **id** is
//! in the log, a **resource** is beside it. `PlaceImage` carries an
//! [`AssetId`](crate::AssetId) exactly as a stamp brush does, the pixels ride in
//! [`DocumentFile::content`](crate::io::DocumentFile) and over the wire as a blob,
//! and `crate::content` is what says a document naming one has to have it.

use crate::geom::{IVec2, TILE_APRON, TILE_SIZE, TileCoord, TileRect, Vec2, tiles_of};
use stark_assetid::{MAX_PICTURE_DIM, Picture};

/// The most tiles one placed picture can write — **derived, not chosen**.
///
/// Every other tile cap in the document is a judgement about how much work one action
/// may ask for ([`MAX_FILL_TILES`](super::fill::MAX_FILL_TILES),
/// [`MAX_TRANSFORM_TILES`](super::transform::MAX_TRANSFORM_TILES)), each carrying the
/// risk of disagreeing with what it bounds. Here [`MAX_PICTURE_DIM`] already bounds
/// the box, so the tile count follows from it and a plan that exceeded this could only
/// mean the arithmetic below is wrong.
pub const MAX_IMAGE_TILES: usize = {
    // The picture's own span, plus the apron band `image_tiles` reaches into, can
    // start anywhere within a tile — hence one partial tile at each end.
    let span = ((MAX_PICTURE_DIM + 2 * TILE_APRON) / TILE_SIZE + 2) as usize;
    span * span
};

/// How far from the origin a placement may sit and still be **exact**.
///
/// `PlaceImage` promises the image's texels land on canvas pixels one for one, with
/// nothing filtered between the file and the tiles — which is why `at` is an
/// [`IVec2`] and why scaling is a separate
/// [`Transform`](super::ActionKind::Transform) where resampling is already
/// pinned (§16.4). That promise is about the *arithmetic*, and the arithmetic goes
/// through `f32`: every integer up to `2^24` is exactly representable, and past it
/// the cast starts rounding — silently, to as much as a whole tile at the far end of
/// the `i32` grid.
///
/// So this is the point at which the placement stops being the one it says it is. The
/// picture's own span comes off the budget because [`extent`] adds it to `at` and that
/// sum has to be exact too. About 16.7 million canvas px from the origin — four orders
/// of magnitude past any painting, and still a refusal rather than a rounding (§16.1).
const MAX_EXACT_PLACEMENT: i32 = (1 << 24) - MAX_PICTURE_DIM as i32;

/// The picture's own extent in canvas px, `[lo, hi)` — `None` for a placement too
/// far from the origin to be stated exactly (see [`MAX_EXACT_PLACEMENT`]).
fn extent(at: IVec2, picture: &Picture) -> Option<(Vec2, Vec2)> {
    if at.x.unsigned_abs() > MAX_EXACT_PLACEMENT as u32
        || at.y.unsigned_abs() > MAX_EXACT_PLACEMENT as u32
    {
        return None;
    }
    let lo = Vec2::new(at.x as f32, at.y as f32);
    Some((
        lo,
        lo + Vec2::new(picture.width as f32, picture.height as f32),
    ))
}

/// Which tiles a placed picture writes, in order.
///
/// Pure geometry with no mask to consult, unlike a fill's plan — a picture lands on
/// its own fresh layer and is bounded by nothing but itself (§23) — so it lives here
/// beside the action rather than in the engine. It is also the *only* quantization of
/// this box anywhere: the footprint claims the whole layer instead of re-deriving one
/// (`footprint.rs`), so the §12.6 hazard `fill_bounds` exists to close cannot arise
/// here at all.
///
/// A tile is written when its **texture** holds any of the picture — the interior plus
/// the apron band, since a tile's texture starts one texel before its interior and
/// content reaching into that band belongs to the neighbour too (§6.4). Filtered
/// rather than merely quantized, because [`TileRect::covering`] floors both bounds: a
/// picture ending exactly on a tile boundary would otherwise name the tile past it,
/// and an all-zero tile pollutes `bounds` and holds pool memory for a texel of
/// nothing.
///
/// `None` refuses the whole action, deterministically, for a placement whose box falls
/// off the `i32` tile grid — or, well before that, one too far from the origin for
/// `f32` to state exactly (`MAX_EXACT_PLACEMENT`). Refused rather than rounded,
/// because a placement is a promise about landing texels on pixels one for one.
/// [`MAX_IMAGE_TILES`] cannot be exceeded by a picture `stark_assetid::picture`
/// admitted, so a `None` from [`tiles_of`] would be a bug in that arithmetic rather
/// than a document to refuse.
pub fn image_tiles(at: IVec2, picture: &Picture) -> Option<Vec<TileCoord>> {
    let (lo, hi) = extent(at, picture)?;
    let apron = Vec2::splat(TILE_APRON as f32);
    let mut tiles = tiles_of(
        TileRect::covering(lo - apron, hi + apron, 0)?,
        MAX_IMAGE_TILES,
    )?;
    tiles.retain(|c| {
        let tex_lo = c.origin() - apron;
        let tex_hi = c.origin() + Vec2::splat((TILE_SIZE + TILE_APRON) as f32);
        tex_lo.x < hi.x && tex_lo.y < hi.y && tex_hi.x > lo.x && tex_hi.y > lo.y
    });
    Some(tiles)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn picture(w: u32, h: u32) -> Picture {
        Picture {
            width: w,
            height: h,
            pixels: vec![0; (w * h * 4) as usize],
        }
    }

    /// **A placement past what `f32` states exactly is refused, not rounded.**
    ///
    /// `PlaceImage` promises the picture's texels land on canvas pixels one for one
    /// (§23). Past `2^24` the cast in [`extent`] starts rounding — quietly, and by up
    /// to a whole tile out at the end of the `i32` grid — so the placement that
    /// happens is not the placement the action names. The `i32` grid itself only runs
    /// out four orders of magnitude later, which is why this bound has to be its own
    /// and cannot be left to `TileRect::covering`.
    #[test]
    fn a_placement_too_far_out_to_state_exactly_is_refused() {
        let img = picture(64, 64);
        // The last exact placement, and the first one past it, on each axis and in
        // each direction.
        let ok = MAX_EXACT_PLACEMENT;
        for at in [IVec2::new(ok, 0), IVec2::new(0, ok), IVec2::new(-ok, -ok)] {
            assert!(
                image_tiles(at, &img).is_some(),
                "{at:?} is exactly representable and must still place",
            );
        }
        for at in [
            IVec2::new(ok + 1, 0),
            IVec2::new(0, ok + 1),
            IVec2::new(-ok - 1, 0),
            IVec2::new(i32::MAX, i32::MAX),
            IVec2::new(i32::MIN, i32::MIN),
        ] {
            assert!(image_tiles(at, &img).is_none(), "{at:?} should be refused");
        }

        // The bound is where it is because the *sum* has to be exact too: the
        // picture's own span is added to `at` before anything is quantized.
        let (lo, hi) = extent(IVec2::splat(ok), &img).expect("the last exact placement");
        assert_eq!(lo.x as i32, ok, "the position survives the cast");
        assert_eq!((hi.x - lo.x) as u32, img.width, "and so does the span");
    }

    /// **Every tile the plan names holds some of the picture, and every tile that
    /// holds some of it is named.**
    ///
    /// Swept across a whole tile stride on both axes rather than checked at one
    /// alignment, because one alignment is exactly what hides a boundary case: the
    /// over-claim appears only when an edge lands *on* a tile boundary, and the
    /// under-claim only when it lands one texel inside the apron band.
    ///
    /// Stated against the tile's own interior rather than against the padded box the
    /// plan quantizes, so the two derivations are independent: a tile is named exactly
    /// when its interior, grown by the apron it must show, meets the picture.
    #[test]
    fn the_plan_names_exactly_the_tiles_the_picture_touches() {
        let img = picture(300, 120);
        let apron = TILE_APRON as f32;
        for offset in 0..TILE_SIZE as i32 {
            let at = IVec2::new(offset - 130, offset);
            let named: std::collections::BTreeSet<TileCoord> = image_tiles(at, &img)
                .expect("an ordinary placement")
                .into_iter()
                .collect();
            let (lo, hi) = extent(at, &img).expect("an ordinary placement");
            // Every tile in a generous window around the picture, asked directly.
            let window = TileRect::covering(lo - Vec2::splat(600.0), hi + Vec2::splat(600.0), 0)
                .expect("an ordinary box");
            for c in window.coords() {
                let tex_lo = c.origin() - Vec2::splat(apron);
                let tex_hi = c.origin() + Vec2::splat(TILE_SIZE as f32 + apron);
                let holds =
                    tex_lo.x < hi.x && tex_lo.y < hi.y && tex_hi.x > lo.x && tex_hi.y > lo.y;
                assert_eq!(
                    named.contains(&c),
                    holds,
                    "{c:?}: named {} but holds {holds}, for a picture at {at:?}",
                    named.contains(&c),
                );
            }
        }
    }

    /// The tile cap is **derived** from the dimension cap, so the largest picture this
    /// build accepts cannot make a plan the cap refuses. If this fails, the arithmetic
    /// in [`MAX_IMAGE_TILES`] is wrong — not the document.
    #[test]
    fn the_largest_admissible_picture_fits_the_derived_cap() {
        // The pixels are never read here, and allocating 67 MB of them to find that
        // out would be the test's whole runtime — the plan is a function of the
        // dimensions alone.
        let widest = Picture {
            width: MAX_PICTURE_DIM,
            height: MAX_PICTURE_DIM,
            pixels: Vec::new(),
        };
        // The worst alignment: one pixel past a tile boundary on both axes, so the box
        // straddles a partial tile at each end.
        let tiles =
            image_tiles(IVec2::new(1, 1), &widest).expect("the cap must admit the largest picture");
        assert!(
            tiles.len() <= MAX_IMAGE_TILES,
            "{} tiles against a cap of {MAX_IMAGE_TILES}",
            tiles.len(),
        );
    }
}
