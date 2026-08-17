//! Content-addressed brush/image assets (§6.6).
//!
//! A brush *shape* is a grayscale coverage mask. Imported images are identified
//! by the BLAKE3 hash of their bytes, so a `StrokeRecord` references a 32-byte
//! [`AssetId`] rather than embedding pixels — keeping the action log tiny and
//! giving deterministic, deduplicated, collaboration-friendly resolution.
//!
//! The store decodes an image to a single-channel `R8` coverage texture and
//! caches it on the GPU. It is `Clone` (`Arc`-backed) so it can ride inside the
//! `Action::Context` alongside the tile pool and stroke renderer.

use std::collections::hash_map::{Entry, HashMap};
use std::sync::{Arc, Mutex};

use crate::error::Result;
use crate::gpu::context::GpuContext;

use stark_model::AssetId;

/// A loaded brush shape: its source bytes and the textures the stroke path reads it
/// through.
///
/// The **views** alone, with no texture beside them. A `wgpu::TextureView` holds its
/// own reference to the texture it was made from, so the texture outlives the view by
/// construction. `StrokeRenderer` drops both textures at the same call and renders.
struct Mask {
    /// Source bytes, retained so the asset can be bundled into the save file.
    bytes: Vec<u8>,
    /// The canonical coverage the id names, retained so the pen volume below can be
    /// baked without decoding the PNG again — one byte per texel, capped at
    /// [`MAX_SHAPE_DIM`]², so at most a megabyte for the largest shape there is.
    coverage: Vec<u8>,
    width: u32,
    height: u32,
    /// The follow-stroke prefix-τ: **one layer, the identity**, since the shape's axis
    /// tracks the tangent and the relative angle is always 0 (§6.6). Built at import,
    /// because it is what nearly every stroke reads and it costs a single integral.
    follow: wgpu::TextureView,
    /// The pen-oriented prefix-τ (§6.6): one **padded** layer per relative angle, built
    /// on first use rather than at import.
    ///
    /// Lazy for the asymmetry between the two: the identity above is one layer and a
    /// linear pass, while this is a rotation per layer over a volume `2·layers` times
    /// the size — and the orientation source is a brush setting the store cannot see at
    /// import, so eagerly baking it would charge every follow-stroke brush in the
    /// library for a mode it never enters.
    pen: Option<wgpu::TextureView>,
    /// The plain (unrotated) coverage mask, for per-stamp footprint sampling in the
    /// brush-dynamics stamp loop (§6.2) — orientation is applied by rotating the sample
    /// coordinates, so one texture serves both sources (unlike the prefix-τ, whose
    /// integration axis is baked in).
    coverage_view: wgpu::TextureView,
}

#[derive(Default)]
struct Inner {
    masks: HashMap<AssetId, Mask>,
}

/// GPU-resident cache of content-addressed coverage masks.
#[derive(Clone)]
pub struct AssetStore {
    ctx: GpuContext,
    inner: Arc<Mutex<Inner>>,
}

impl AssetStore {
    pub fn new(ctx: GpuContext) -> Self {
        Self {
            ctx,
            inner: Arc::new(Mutex::new(Inner::default())),
        }
    }

    /// Import a brush image (any PNG), returning its content id. The id is the
    /// hash of the *decoded coverage* (not the file bytes), so it is independent
    /// of source encoding — and the asset is stored as a compact grayscale PNG.
    pub fn import(&self, bytes: &[u8]) -> Result<AssetId> {
        // Canonicalize: stored form is re-encoded from the coverage.
        self.load(bytes, None)
    }

    /// Re-insert an asset from its saved (already-canonical grayscale PNG) bytes
    /// when loading a document — keeps the stored bytes verbatim.
    pub fn insert_bytes(&self, bytes: &[u8]) -> Result<AssetId> {
        self.load(bytes, Some(bytes.to_vec()))
    }

    fn load(&self, decode_from: &[u8], store_bytes: Option<Vec<u8>>) -> Result<AssetId> {
        // Decode, cap and hash are the identity contract's, not ours (§19): the id
        // names the canonical coverage, so a reload of the stored PNG lands back on
        // the same id.
        let canonical = stark_assetid::coverage(decode_from)?;
        let id = canonical.id();
        let stark_assetid::Canonical {
            width: w,
            height: h,
            texels: coverage,
        } = canonical;
        let mut inner = self.inner.lock().expect("asset store poisoned");
        if let Entry::Vacant(slot) = inner.masks.entry(id) {
            let bytes = match store_bytes {
                Some(b) => b,
                None => stark_assetid::Canonical {
                    width: w,
                    height: h,
                    texels: coverage.clone(),
                }
                .encode()?,
            };
            let cov: Vec<f32> = coverage.iter().map(|&b| b as f32 / 255.0).collect();
            // The follow-stroke volume, which is the whole of the bake for the common
            // brush: one layer, the mask as it stands, integrated over its own width.
            let follow = build_prefix_tau(&self.ctx, w, h, 1, 2.0 / w as f32, &cov);
            let coverage_view = build_coverage_r8(&self.ctx, w, h, &coverage);
            slot.insert(Mask {
                bytes,
                coverage,
                width: w,
                height: h,
                follow,
                pen: None,
                coverage_view,
            });
        }
        Ok(id)
    }

    /// A clonable view of the brush's prefix-τ texture for `id` under `orientation`, if
    /// loaded (the running integral of `−ln(1−coverage)` along the travel axis, §6.2).
    ///
    /// **The two orientation sources read different volumes**, because they ask
    /// different questions of the same mask (§6.6). `FollowStroke` keeps the shape's
    /// axis on the tangent, so the relative angle is always 0 and one identity layer
    /// answers every segment. `Pen` lets the two diverge, which means rotating the mask
    /// inside the frame the sweep integrates along — and a square does not fit in
    /// itself turned, so that volume is padded by [`PEN_PAD`] and read at a frame
    /// scaled to match ([`pen_frame_scale`]).
    ///
    /// The pen volume is built here on first ask and kept. `&self` throughout: the
    /// store is `Arc<Mutex<_>>` behind a `Clone`, and this is the one place the cache
    /// grows after import.
    pub fn prefix_view(
        &self,
        id: AssetId,
        orientation: stark_model::document::OrientationSource,
    ) -> Option<wgpu::TextureView> {
        let mut inner = self.inner.lock().expect("asset store poisoned");
        let mask = inner.masks.get_mut(&id)?;
        if orientation == stark_model::document::OrientationSource::FollowStroke {
            return Some(mask.follow.clone());
        }
        if mask.pen.is_none() {
            let (w, h) = (mask.width, mask.height);
            let cov: Vec<f32> = mask.coverage.iter().map(|&b| b as f32 / 255.0).collect();
            // Padded so the turned square still fits, and at enough texels to keep the
            // shape's own resolution across the diagonal.
            let (pw, ph) = (pad_dim(w), pad_dim(h));
            let layers = orientation_layers(pw, ph);
            let rotated = rotate_layers_padded(&cov, w, h, pw, ph, layers);
            // **The row integral is the mask's, not the padded texture's.** A column of
            // the padded volume is `PEN_PAD` times narrower in the units the mask is
            // measured in, so integrating with the padded width would divide every
            // stroke's optical depth by `PEN_PAD` and lighten the mark. The mask spans
            // `pw / PEN_PAD` columns of it, and this is that span's own `dx` — the same
            // number the unpadded bake above uses, to the rounding of `pad_dim`.
            let dx = 2.0 * PEN_PAD / pw as f32;
            mask.pen = Some(build_prefix_tau(&self.ctx, pw, ph, layers, dx, &rotated));
        }
        mask.pen.clone()
    }

    /// A clonable view of the brush's plain coverage mask for `id`, if loaded —
    /// sampled per stamp by the brush-dynamics loop (§6.2).
    ///
    /// One texture for both orientation sources, unlike the prefix-τ above: the loop
    /// samples this in the **shape's own frame**, rotating the lookup rather than the
    /// mask, so nothing here is ever turned and there is no corner to lose.
    pub fn coverage_view(&self, id: AssetId) -> Option<wgpu::TextureView> {
        self.inner
            .lock()
            .expect("asset store poisoned")
            .masks
            .get(&id)
            .map(|m| m.coverage_view.clone())
    }

    /// Whether `id` is loaded in this store.
    pub fn contains(&self, id: AssetId) -> bool {
        self.inner
            .lock()
            .expect("asset store poisoned")
            .masks
            .contains_key(&id)
    }

    /// The canonical PNG bytes of one asset, if loaded — what a peer mirror or
    /// a second (preview) engine needs to reproduce the shape.
    pub fn bytes(&self, id: AssetId) -> Option<Vec<u8>> {
        self.inner
            .lock()
            .expect("asset store poisoned")
            .masks
            .get(&id)
            .map(|m| m.bytes.clone())
    }

    /// Source bytes of every loaded asset, for bundling into the save file (§8).
    pub fn all_bytes(&self) -> Vec<(AssetId, Vec<u8>)> {
        self.inner
            .lock()
            .expect("asset store poisoned")
            .masks
            .iter()
            .map(|(id, m)| (*id, m.bytes.clone()))
            .collect()
    }
}

/// Largest number of orientation slices a brush's pen-oriented prefix-τ volume holds
/// (§6.6). With linear interpolation between adjacent layers this is ~5.6° resolution
/// — smooth for any practical pen rotation.
pub const MAX_ORIENTATION_LAYERS: u32 = 64;

/// Memory budget (bytes) for one brush's pen-oriented prefix-τ volume. The layer count
/// is chosen so `width × height × layers × 4 (R32Float)` stays under this — so a large
/// detailed stamp keeps its full resolution and trades orientation granularity for
/// memory instead. Only the pen volume is measured against it: the follow-stroke bake
/// is a single layer and has nothing to trade.
const PREFIX_BUDGET_BYTES: u32 = 64 << 20; // 64 MiB

/// How many orientation slices to build for a `width × height` **padded** volume: as
/// many as the memory budget allows, capped at [`MAX_ORIENTATION_LAYERS`] and at least
/// 1.
pub fn orientation_layers(width: u32, height: u32) -> u32 {
    let per_layer = (width * height * 4).max(1);
    (PREFIX_BUDGET_BYTES / per_layer).clamp(1, MAX_ORIENTATION_LAYERS)
}

/// How much wider the pen-oriented volume is than the mask it holds (§6.6): the
/// diagonal of a unit square, since that is the one dimension a square turned by an
/// arbitrary angle needs.
///
/// **This is the whole of the padding argument.** The mask is normalized to a square
/// in brush-local coordinates whatever its pixel aspect, and `Pen` turns it inside the
/// frame the sweep integrates along. A square does not fit in itself turned — a 45°
/// rotation puts its corners `√2` out — so an unpadded bake sampled zero out there and
/// clipped the shape's corners off, silently, at every angle but the four right ones.
pub const PEN_PAD: f32 = std::f32::consts::SQRT_2;

/// The frame a pen-oriented volume is read at, as a multiple of the tip's own radius.
///
/// The padded volume's `[-1, 1]²` is [`PEN_PAD`] tips wide, so the sweep has to be
/// integrated in a frame that much larger for the mask inside it to land at the radius
/// the brush asked for. The renderer scales the segment's frame by exactly this and
/// leaves the tip's own radius alone, which is what keeps a nib's paint rates, bleed
/// cadence and touch-down dab the size of the tip rather than of the box around it.
pub fn pen_frame_scale() -> f32 {
    PEN_PAD
}

/// The padded texel count for one axis of a `n`-texel mask: enough that the shape keeps
/// its own resolution across the diagonal rather than being resampled down into the
/// same grid it came from.
fn pad_dim(n: u32) -> u32 {
    ((n as f32 * PEN_PAD).ceil() as u32).max(n)
}

/// Pre-rotate a normalized `[-1, 1]²` coverage mask into `layers` orientation slices of
/// a **padded** `pw × ph` volume (§6.6).
///
/// Slice `l` rotates the shape by `2π·l/layers` into the travel frame, so the sweep's
/// x-integral yields the swept depth at that orientation. The source's unit square lands
/// in the central `1/PEN_PAD` of the output's, which is what leaves room for its corners
/// at every angle; bilinear sampling, zero outside the source, so the border of the
/// padding is exactly the border the mask had. Returns a `layers × ph × pw` buffer.
///
/// Slice 0 is *not* the identity — it is the mask resampled into the padded grid.
/// Nothing needs it to be: `FollowStroke`, the one caller that would read layer 0 as
/// the shape's native orientation, has its own unpadded single-layer bake and never
/// reads this at all.
fn rotate_layers_padded(
    coverage: &[f32],
    width: u32,
    height: u32,
    pw: u32,
    ph: u32,
    layers: u32,
) -> Vec<f32> {
    let w = width as usize;
    let (pwu, phu) = (pw as usize, ph as usize);
    let mut out = vec![0.0f32; pwu * phu * layers as usize];
    let sample = |sx: f32, sy: f32| -> f32 {
        // sx, sy in normalized [-1, 1]; bilinear sample of the source, 0 outside.
        let fx = (sx * 0.5 + 0.5) * width as f32 - 0.5;
        let fy = (sy * 0.5 + 0.5) * height as f32 - 0.5;
        let x0 = fx.floor();
        let y0 = fy.floor();
        let (tx, ty) = (fx - x0, fy - y0);
        let at = |xi: f32, yi: f32| -> f32 {
            if xi < 0.0 || yi < 0.0 || xi >= width as f32 || yi >= height as f32 {
                0.0
            } else {
                coverage[yi as usize * w + xi as usize]
            }
        };
        let a = at(x0, y0) * (1.0 - tx) + at(x0 + 1.0, y0) * tx;
        let b = at(x0, y0 + 1.0) * (1.0 - tx) + at(x0 + 1.0, y0 + 1.0) * tx;
        a * (1.0 - ty) + b * ty
    };
    for l in 0..layers as usize {
        let theta = std::f32::consts::TAU * l as f32 / layers as f32;
        let (s, c) = theta.sin_cos();
        let base = l * pwu * phu;
        for y in 0..phu {
            let py = ((y as f32 + 0.5) / ph as f32 * 2.0 - 1.0) * PEN_PAD;
            for x in 0..pwu {
                let px = ((x as f32 + 0.5) / pw as f32 * 2.0 - 1.0) * PEN_PAD;
                // Sample the source at R(-θ)·(px, py): the image rotates by +θ. The
                // `PEN_PAD` above is what shrinks the source into the padded square —
                // an output a full padded unit out reads `PEN_PAD` source units out,
                // which is past the mask's own edge and reads the zero border.
                let sx = px * c + py * s;
                let sy = -px * s + py * c;
                out[base + y * pwu + x] = sample(sx, sy);
            }
        }
    }
    out
}

/// A coverage sample's **optical depth**, `κ = −ln(1 − coverage)` — the currency the
/// deposit sums (§6.1), and the one conversion between the two.
///
/// The clamp is what keeps `κ` finite where a mask reaches 1: full coverage is
/// infinite depth, and a mask that says so would carry `+∞` into every prefix sum
/// downstream of it. Capping the *coverage* rather than the depth puts the ceiling
/// somewhere a reader of the mask can see it. `dynamics.wesl`'s own `tau_of` mirrors
/// this, clamp included, so the tool side — which has no prefix to difference — agrees
/// with the volume built here.
pub(crate) fn tau_of(coverage: f32) -> f32 {
    -(1.0 - coverage.clamp(0.0, 0.999)).ln()
}

/// Build a brush's **prefix-τ** volume (§6.2, §6.6): for each orientation
/// `layer` and each row, the running integral of optical depth `κ = −ln(1−coverage)`
/// along the travel axis (x), normalized to brush-local units (x spans `[-1, 1]`, width
/// 2). Stored as a `R32Float` **2D-array** texture (the array axis is orientation, sampled
/// with wrapping) and read via `textureLoad` + manual trilinear by the sweep shader: a
/// segment's swept depth at a point is `prefix(u) − prefix(u−d)` on its layer. (A 2D array
/// rather than a true 3D texture so the mask keeps its full width/height — 3D textures are
/// capped far smaller, e.g. 256px, by `maxTextureDimension3D`.)
///
/// Shared by [`AssetStore`] (image brushes — one identity layer for follow-stroke, a
/// padded stack for pen) and the stroke renderer (the round tip, regenerated per
/// `hardness` — rotation-invariant, 1 layer). `coverage` is `layers × height × width`
/// row-major in `[0, 1]`.
///
/// `dx` is the **brush-local width of one column**, and it is a parameter rather than
/// `2/width` because the two are not the same thing once a volume is padded: what the
/// integral has to be measured in is the span the *mask* occupies, or the stroke's
/// optical depth — and so how dark it comes out — would follow the size of the box
/// around it. An unpadded volume passes `2/width` and is exactly what it always was.
///
/// Returns the view alone: it holds its own reference to the texture, so there is
/// nothing for a caller to keep beside it.
pub fn build_prefix_tau(
    ctx: &GpuContext,
    width: u32,
    height: u32,
    layers: u32,
    dx: f32,
    coverage: &[f32],
) -> wgpu::TextureView {
    let w = width as usize;
    let mut prefix = vec![0.0f32; coverage.len()];
    for y in 0..(height * layers) as usize {
        // Rows are contiguous across layers (layer-major, then row), so one linear pass
        // integrates every layer's rows independently.
        let mut acc = 0.0f32;
        for x in 0..w {
            acc += tau_of(coverage[y * w + x]) * dx;
            prefix[y * w + x] = acc;
        }
    }

    let extent = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: layers,
    };
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("stark brush prefix-tau"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    ctx.queue.write_texture(
        texture.as_image_copy(),
        bytemuck::cast_slice(&prefix),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: Some(height),
        },
        extent,
    );
    // Always a 2D-array view (even for the single-layer round tip) so the shader binds one
    // texture type for every brush.
    texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    })
}

/// Upload a coverage mask as a filterable `R8Unorm` texture — the per-stamp
/// footprint the brush-dynamics loop samples (rotating the sample coordinates
/// for orientation, so no pre-rotated layers are needed). Shared by the asset
/// store (image brushes) and the stroke renderer (the round tip, per hardness).
///
/// Returns the view alone, like [`build_prefix_tau`].
pub fn build_coverage_r8(
    ctx: &GpuContext,
    width: u32,
    height: u32,
    coverage: &[u8],
) -> wgpu::TextureView {
    let extent = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("stark brush coverage"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    ctx.queue.write_texture(
        texture.as_image_copy(),
        coverage,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width),
            rows_per_image: Some(height),
        },
        extent,
    );
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mask with structure on both axes, so an integral over it is a real one.
    fn ramp(w: u32, h: u32) -> Vec<f32> {
        (0..w * h)
            .map(|i| {
                let x = (i % w) as f32 / w as f32 * 2.0 - 1.0;
                let y = (i / w) as f32 / h as f32 * 2.0 - 1.0;
                0.9 * (1.0 - x.abs()) * (1.0 - y.abs() * 0.5)
            })
            .collect()
    }

    /// **Padding must not change how dark the stroke comes out.**
    ///
    /// The mark's optical depth is the volume's row integral, and a padded volume's
    /// columns are narrower *in texels* while standing for the same width *of mask*. So
    /// the integral has to be taken at the mask's own column width — that is the whole
    /// content of [`build_prefix_tau`]'s `dx` being a parameter. Take it at the padded
    /// texture's instead and every pen-oriented stroke lands `PEN_PAD` lighter than the
    /// same brush following the stroke, which is the kind of wrong that reads as a
    /// deliberate difference between the two modes.
    ///
    /// Checked as the integral over the whole layer rather than row by row: the padded
    /// grid has its own rows and they do not line up with the source's, while the
    /// double integral is the same quantity on both sides and is what the row totals
    /// sum to.
    #[test]
    fn a_padded_layer_carries_the_masks_own_optical_depth() {
        let (w, h) = (48u32, 32u32);
        let cov = ramp(w, h);
        let (pw, ph) = (pad_dim(w), pad_dim(h));
        let padded = rotate_layers_padded(&cov, w, h, pw, ph, 1);

        let source: f32 =
            cov.iter().map(|&c| tau_of(c)).sum::<f32>() * (2.0 / w as f32) * (2.0 / h as f32);
        let got: f32 = padded.iter().map(|&c| tau_of(c)).sum::<f32>()
            * (2.0 * PEN_PAD / pw as f32)
            * (2.0 * PEN_PAD / ph as f32);
        let err = (got - source).abs() / source;
        assert!(
            err < 0.02,
            "the padded bake carries {got} of the mask's {source} optical depth \
             ({:.1}% off) — a pen-oriented stroke would not match its own brush",
            err * 100.0,
        );
    }

    /// **Turning a shape must not change how much of it there is** — which is the one
    /// thing an unpadded rotation cannot promise, and the bug the padding exists for.
    ///
    /// A square does not fit in itself turned: rotate one by 45° inside its own bounds
    /// and what survives is the octagon they share, `2(√2−1) ≈ 83%` of it. On a mask
    /// that reaches its corners — which is most bristle and charcoal stamps — that is a
    /// sixth of the tip gone, at every angle but the four right ones, with the loss
    /// swelling and shrinking as the pen turns.
    #[test]
    fn a_padded_layer_keeps_the_corners_an_unpadded_one_clipped() {
        const LAYERS: u32 = 8; // so layer 1 is the worst case, 45°
        let (w, h) = (48u32, 48u32);
        // Opaque to the very corner: the mask that has the most to lose.
        let cov = vec![0.8f32; (w * h) as usize];
        let (pw, ph) = (pad_dim(w), pad_dim(h));
        let padded = rotate_layers_padded(&cov, w, h, pw, ph, LAYERS);

        let plane = (pw * ph) as usize;
        let total = |l: usize| padded[l * plane..(l + 1) * plane].iter().sum::<f32>();
        let flat = total(0);
        for l in 1..LAYERS as usize {
            let err = (total(l) - flat).abs() / flat;
            assert!(
                err < 0.02,
                "layer {l} of {LAYERS} carries {:.1}% of the shape the identity layer \
                 does — the rotation is losing mask off the edge of its own volume",
                total(l) / flat * 100.0,
            );
        }
    }
}
