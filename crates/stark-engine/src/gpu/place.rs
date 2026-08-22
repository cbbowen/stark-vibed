//! Turning an imported image into paint tiles (§23).
//!
//! **The only tile writer in this engine with no shader.** Every other one computes a
//! texel from texels — a stroke's sweep over what is resident, a fill's parcel over a
//! base, a transform's resample of a source quad — so it belongs on the GPU, where the
//! inputs already are. A placed image has no such input: it lands on a layer that did
//! not exist a moment ago (§23), so there is nothing beneath it to stack onto, and its
//! texels are simply the file's texels read through the paint representation. Uploading
//! the image to a texture in order to have a fragment shader copy it into another
//! texture would be a round trip to say nothing.
//!
//! Three things fall out of that, and each is worth more than the pass it replaces:
//!
//! - **Bit-exact everywhere.** These tiles are pure CPU f32 arithmetic, so two peers
//!   and two replays on different adapters produce the same bytes — which is true of no
//!   render pass in this crate (§9's goldens are adapter-specific for exactly that
//!   reason).
//! - **No dimension cap from the hardware.** Nothing here binds the image as a texture,
//!   so the only bound on its size is the document's own
//!   ([`MAX_PICTURE_DIM`](stark_assetid::MAX_PICTURE_DIM)) rather than whatever
//!   `max_texture_dimension_2d` this device reports.
//! - **The apron is free and provably right.** Each texel is computed from its own
//!   canvas position, so a tile's apron is bit-identical to its neighbour's interior by
//!   construction — §6.4's rule, met by the strongest form of the argument rather than
//!   by a pass being careful.
//!
//! The color conversion is the host's `rgb_to_channels` (§6.7), the same function a
//! fill converts its parcel with — so an image and a fill of the same color land the
//! same paint, in an Oklab document and a pigment one alike. That it runs per texel
//! here rather than once per fill is the cost of an image being a picture, and it is
//! paid on import and on replay.

use std::sync::Arc;

use crate::colorspace::ColorSpace;
use crate::gpu::channels::{ChannelFormats, Channels};
use crate::gpu::context::GpuContext;
use crate::gpu::half::f32_to_f16;
use crate::gpu::tile::{AllocSource, TileMap, TilePool};
use stark_assetid::Picture;
use stark_model::Srgb;
use stark_model::document::image_tiles;
use stark_model::geom::{IVec2, TILE_APRON, TILE_SIZE, TILE_TEX};

// The deposit law's two ends, read from the shader that declares them (§6.10) rather
// than restated here. `fill.wesl` inverts the identical law to lay the identical paint;
// a second copy of either number would be a second opinion about what "opaque" means,
// in the one file with no `.wesl` to compare against.
use stark_shaders::mirror::paint_common::{OPACITY_K, OPAQUE_MASS};

/// Builds the tiles of a placed image.
///
/// Holds no pipeline and no bind group layout — there is no pass — so this is the
/// color space and the device handle, and nothing else. `Clone` like its siblings so
/// it rides in the `Action::Context` (§5).
#[derive(Clone)]
pub struct PlaceRenderer {
    ctx: GpuContext,
    color_space: Arc<dyn ColorSpace>,
    /// The channel formats this document's tiles carry — the color space's, resolved
    /// once (§6.7).
    formats: ChannelFormats,
}

/// One texel of paint, as the three channels store it.
///
/// Named rather than returned as a tuple because the residual's presence is a property
/// of the *document* and not of the texel: a colorimetric space simply never reads it
/// (§6.7), and writing that as an `Option` per texel would be a branch on a constant,
/// sixty-five thousand times a tile.
struct Texel {
    /// Latent color premultiplied by per-unit opacity, with that opacity in `.w`.
    color: [f32; 4],
    /// Paint height — the amount of paint (§6.1).
    height: f32,
    /// The rest of the same color (§6.7), premultiplied and carrying its own copy of
    /// the opacity, because the fixed-function blends read each target's own alpha.
    resid: [f32; 4],
}

impl Texel {
    /// Nothing here: no paint, no color, no height.
    const NOTHING: Texel = Texel {
        color: [0.0; 4],
        height: 0.0,
        resid: [0.0; 4],
    };
}

impl PlaceRenderer {
    pub(crate) fn new(ctx: &GpuContext, color_space: Arc<dyn ColorSpace>) -> Self {
        let formats = ChannelFormats::of(color_space.as_ref());
        Self {
            ctx: ctx.clone(),
            color_space,
            formats,
        }
    }

    /// The tiles `image` becomes when its top-left texel is placed at `at`.
    ///
    /// `None` refuses the whole action, deterministically, when the placement falls off
    /// the tile grid an `i32` can address — the one thing [`image_tiles`] can refuse,
    /// and refused rather than clamped for the reason a transform's degenerate map is
    /// (§16.1): a placement somewhere else is not the placement that was asked for.
    pub fn render(&self, pool: &TilePool, at: IVec2, image: &Picture) -> Option<TileMap> {
        let coords = image_tiles(at, image)?;
        let mut tiles = TileMap::new();

        // One block per channel, reused across every tile of the image: they are
        // rewritten in full before each upload, and `write_texture` copies out of them
        // before it returns, so there is nothing to keep alive between tiles. A tile is
        // 256², so this is 512 KB rather than 512 KB times the tile count.
        let texels = (TILE_TEX * TILE_TEX) as usize;
        let mut color = vec![0u8; texels * 8];
        let mut aux = vec![0u8; texels * 2];
        let mut resid = vec![
            0u8;
            if self.formats.has_resid() {
                texels * 8
            } else {
                0
            }
        ];

        for coord in coords {
            // Where this tile's texture starts *within the image*: the canvas position
            // of its top-left texel — apron included, which is what makes the apron a
            // computed band rather than a copied one (§6.4) — less the placement. Whole
            // pixels on both sides, so the image's texels land on canvas pixels one for
            // one and nothing is resampled (§23).
            //
            // Derived in integers rather than through `mask_tex_origin`, whose answer is
            // an `f32`: a tile a few million pixels from the origin has an origin past
            // where `f32` can count in single pixels, and rounding it would slide the
            // image within that tile — differently per tile, which is a seam (§6.4) in
            // the far reaches of an infinite canvas. What is needed here is not the
            // origin but its offset from the image, and that is always small.
            let (ox, oy) = (offset(coord.x, at.x), offset(coord.y, at.y));

            for j in 0..i64::from(TILE_TEX) {
                for i in 0..i64::from(TILE_TEX) {
                    let texel = texel(self.color_space.as_ref(), image.sample(ox + i, oy + j));
                    let n = (j * i64::from(TILE_TEX) + i) as usize;
                    write_half4(&mut color[n * 8..], texel.color);
                    write_half(&mut aux[n * 2..], texel.height);
                    if self.formats.has_resid() {
                        write_half4(&mut resid[n * 8..], texel.resid);
                    }
                }
            }

            let channels = Channels::acquire(pool, self.formats, AllocSource::PlacedImage);
            channels.color.write_block(&self.ctx.queue, &color);
            channels.aux.write_block(&self.ctx.queue, &aux);
            if let Some(target) = &channels.resid {
                target.write_block(&self.ctx.queue, &resid);
            }
            tiles = tiles.insert(coord, channels.into_tile());
        }
        Some(tiles)
    }
}

/// One axis of "where this tile's texture starts within the image": the tile's origin,
/// backed off by the apron, less the placement — in `i64`, so a tile index that would
/// overrun an `i32` of pixels is still exact (see [`Picture::sample`]).
fn offset(tile: i32, at: i32) -> i64 {
    i64::from(tile) * i64::from(TILE_SIZE) - i64::from(TILE_APRON) - i64::from(at)
}

/// What one source pixel becomes as paint.
///
/// **The same law a fill lands its parcel by** (`fill.wesl`), evaluated here because
/// there is no shader to evaluate it in: the source's alpha is a *coverage* — the
/// quantity the eye reads — and the paint that produces it is fully opaque paint of
/// whatever mass the slab law needs, `m = −ln(1 − w)/K`. So an opaque photograph lands
/// opaque paint that takes the light and can be glazed over or scraped back, and a
/// soft-edged cut-out thins to nothing at its edge rather than fading in color.
///
/// A free function over the space rather than a method, so the tests below can hold the
/// arithmetic to its claims without a GPU — this is where every one of them lives, and
/// a copy of it written out in the test module would be a test of the copy.
fn texel(space: &dyn ColorSpace, rgba: [u8; 4]) -> Texel {
    // An exact branch on "nothing here", for `fill.wesl`'s reason: a fully transparent
    // source pixel must produce an *empty* texel rather than a vanishing one, because
    // an all-but-zero height still makes the tile look painted to `bounds` and to the
    // compositor.
    if rgba[3] == 0 {
        return Texel::NOTHING;
    }
    // In the cube by construction — `u8 / 255` cannot leave it — so the constructor
    // here is a statement of what these bytes are, not a repair.
    let rgb = Srgb::new([
        rgba[0] as f32 / 255.0,
        rgba[1] as f32 / 255.0,
        rgba[2] as f32 / 255.0,
    ]);
    let channels = space.rgb_to_channels(rgb);
    let res = space.rgb_to_resid(rgb);

    // Where the slider ends, as mass: coverage is `1 − exp(−K·mass)` and reaches 1 only
    // in the limit, so a fully opaque pixel asks for `OPAQUE_MASS` rather than for the
    // infinity the inversion would otherwise return.
    let want = (rgba[3] as f32 / 255.0).min(1.0 - (-OPACITY_K * OPAQUE_MASS).exp());
    let height = -(1.0 - want).ln() / OPACITY_K;
    // Per-unit opacity 1: the paint itself is opaque, and how much of the ground it
    // hides is the height above (§6.1). The channels are premultiplied by it, which at
    // 1 is the identity — written as a multiplication anyway so the representation is
    // stated rather than implied.
    let op = 1.0;
    Texel {
        color: [channels[0] * op, channels[1] * op, channels[2] * op, op],
        height,
        resid: [res[0] * op, res[1] * op, res[2] * op, op],
    }
}

fn write_half(out: &mut [u8], v: f32) {
    out[..2].copy_from_slice(&f32_to_f16(v).to_le_bytes());
}

fn write_half4(out: &mut [u8], v: [f32; 4]) {
    for (i, c) in v.into_iter().enumerate() {
        write_half(&mut out[i * 2..], c);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colorspace::OkLabColorSpace;
    use crate::gpu::half::f16_to_f32;

    /// The default space, which is the one every test here reasons in.
    fn texel_of(rgba: [u8; 4]) -> Texel {
        texel(&OkLabColorSpace, rgba)
    }

    /// **A tile far from the canvas origin addresses the image exactly**, which the
    /// `f32` origin every other pass reads does not.
    ///
    /// The infinite canvas is addressed by an `i32` of *tiles*, so a legal tile index
    /// has a pixel origin far past where `f32` counts in single pixels. Reading the
    /// offset off `mask_tex_origin` would therefore slide the image within such a tile —
    /// and slide it by a *different* amount per tile, since the rounding depends on the
    /// value — which is a seam (§6.4) at the far reaches of the canvas, invisible until
    /// someone pans there.
    ///
    /// Both halves are asserted: that the integer derivation is right, and that the
    /// `f32` one is genuinely wrong somewhere in an ordinary span of tiles. Without the
    /// second half this would pass just as well if the hazard were imaginary, and the
    /// integer arithmetic would look like caution rather than a fix.
    #[test]
    fn a_distant_tile_addresses_the_image_exactly() {
        use crate::gpu::mask_tex_origin;
        use stark_model::geom::TileCoord;

        // Far enough out that a tile origin needs more than the 24 bits of mantissa an
        // `f32` has, and well inside what a `TileRect` addresses.
        let base = 300_000i32;
        let mut f32_disagreed = false;
        for tile in base..base + 64 {
            // An image starting 7 px into this tile's interior. The tile's *texture*
            // starts one apron earlier, so it begins 8 texels before the image — a
            // small number whatever the tile index, which is the whole point.
            let at = (i64::from(tile) * i64::from(TILE_SIZE) + 7) as i32;
            assert_eq!(
                offset(tile, at),
                -(7 + i64::from(TILE_APRON)),
                "tile {tile} did not begin 8 texels before the image",
            );
            let via_f32 = mask_tex_origin(TileCoord::new(tile, 0)).x as i64 - i64::from(at);
            f32_disagreed |= via_f32 != offset(tile, at);
        }
        assert!(
            f32_disagreed,
            "the f32 origin was exact for every tile in this span, so this test is \
             guarding nothing — pick a span further out",
        );
    }

    /// **The source's alpha is a coverage, and the paint laid reproduces it.**
    ///
    /// The whole of what makes an imported image behave like paint rather than like a
    /// pasted rectangle: half-transparent pixels come out as half-covering paint, so a
    /// cut-out's soft edge thins away instead of fading, and an opaque pixel covers.
    /// Checked through the slab law itself, which is the law the media pass runs
    /// forward (§6.1).
    #[test]
    fn a_source_alpha_lands_as_the_coverage_it_asks_for() {
        for a in [1u8, 64, 128, 200, 255] {
            let texel = texel_of([200, 40, 30, a]);
            let covers = 1.0 - (-OPACITY_K * texel.color[3] * texel.height).exp();
            let want = (a as f32 / 255.0).min(1.0 - (-OPACITY_K * OPAQUE_MASS).exp());
            assert!(
                (covers - want).abs() < 1e-4,
                "alpha {a} asked for {want} and lands {covers}",
            );
        }
    }

    /// A fully transparent pixel is **nothing**, exactly — not almost nothing.
    ///
    /// An all-but-zero height reads as painted to `bounds` and to the compositor, so a
    /// photograph with a transparent margin would place a layer whose extent is the
    /// rectangle rather than the picture.
    #[test]
    fn a_transparent_pixel_lays_no_paint_at_all() {
        let texel = texel_of([200, 40, 30, 0]);
        assert_eq!(texel.color, [0.0; 4]);
        assert_eq!(texel.height, 0.0);
        assert_eq!(texel.resid, [0.0; 4]);
    }

    /// **An opaque pixel keeps its color through the half-float store.**
    ///
    /// The one lossy step between a file and a tile is the `f32 → f16` write, and the
    /// signed encoder is what carries an Oklab latent's `a` and `b` axes through it
    /// (see `gpu::half`). A red that came back green here is the failure the old
    /// non-negative encoder would have produced, silently.
    #[test]
    fn a_color_survives_the_half_float_store() {
        let space = OkLabColorSpace;
        for rgb in [
            [0.9f32, 0.15, 0.1],
            [0.1, 0.4, 0.85],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ] {
            let source = rgb.map(|c| (c * 255.0).round() as u8);
            let texel = texel_of([source[0], source[1], source[2], 255]);
            let stored = [0, 1, 2].map(|i| f16_to_f32(f32_to_f16(texel.color[i])));
            let back = space.channels_to_rgb([stored[0], stored[1], stored[2], 1.0], [0.0; 3]);
            let asked = source.map(|c| c as f32 / 255.0);
            for c in 0..3 {
                assert!(
                    (back[c] - asked[c]).abs() < 0.01,
                    "{asked:?} came back {back:?}",
                );
            }
        }
    }
}
