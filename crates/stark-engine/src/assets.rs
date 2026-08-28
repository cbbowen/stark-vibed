//! Content-addressed brush/image assets (§6.6).
//!
//! A brush *shape* is a grayscale coverage mask. Imported images are identified by
//! the BLAKE3 hash of their **decoded, capped coverage** — not of the bytes they
//! arrived in — so a `StrokeRecord` references a 32-byte [`AssetId`] rather than
//! embedding pixels, keeping the action log tiny and giving deterministic,
//! deduplicated, collaboration-friendly resolution.
//!
//! Hashing the canonical form rather than the file is what makes two encodings of one
//! picture the same asset, and it is the identity contract §19 freezes: the cap and
//! the decode are part of what an id *means*, which is why they live in
//! `stark-assetid` where a build script can compute one without a GPU. See
//! [`AssetStore::import`].
//!
//! The store decodes an image to a single-channel `R8` coverage texture and
//! caches it on the GPU. It is `Clone` (`Arc`-backed) so it can ride inside the
//! `Action::Context` alongside the tile pool and stroke renderer.

use std::collections::hash_map::{Entry, HashMap};
use std::sync::{Arc, Mutex};

use crate::error::Result;
use crate::gpu::context::GpuContext;
use crate::unpoisoned;

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
    /// The pen-oriented prefix-τ (§6.6): one layer per relative angle, built on
    /// first use rather than at import.
    ///
    /// Lazy for the asymmetry between the two: the identity above is one layer and a
    /// linear pass, while this is a rotation per layer over a volume `layers` times
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

/// The two GPU readings of one loaded brush mask, resolved under one store lock.
pub(crate) struct MaskViews {
    pub(crate) prefix: wgpu::TextureView,
    pub(crate) coverage: wgpu::TextureView,
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
        let mut inner = unpoisoned(self.inner.lock());
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
            let follow = build_prefix_tau(&self.ctx, w, h, 1, &cov);
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
    /// answers every segment. `Pen` lets the two diverge, which means rotating the
    /// mask inside the frame the sweep integrates along — safe in the mask's own
    /// square, because a canonical mask's content lies inside the disc inscribed in
    /// it (`stark_assetid::coverage`) and a rotation maps that disc to itself. The
    /// volume was padded by `√2` while a mask could occupy its corners, and every
    /// pen-oriented brush paid double the texels — or, at the memory budget, half
    /// the orientation resolution — for padding that held nothing.
    ///
    /// The pen volume is built here on first ask and kept. `&self` throughout: the
    /// store is `Arc<Mutex<_>>` behind a `Clone`, and this is the one place the cache
    /// grows after import.
    pub fn prefix_view(
        &self,
        id: AssetId,
        orientation: stark_model::document::OrientationSource,
    ) -> Option<wgpu::TextureView> {
        let mut inner = unpoisoned(self.inner.lock());
        let mask = inner.masks.get_mut(&id)?;
        if orientation == stark_model::document::OrientationSource::FollowStroke {
            return Some(mask.follow.clone());
        }
        if mask.pen.is_none() {
            let (w, h) = (mask.width, mask.height);
            let cov: Vec<f32> = mask.coverage.iter().map(|&b| b as f32 / 255.0).collect();
            let layers = orientation_layers(w, h);
            let rotated = rotate_layers(&cov, w, h, layers);
            mask.pen = Some(build_prefix_tau(&self.ctx, w, h, layers, &rotated));
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
        unpoisoned(self.inner.lock())
            .masks
            .get(&id)
            .map(|m| m.coverage_view.clone())
    }

    /// Resolve the prefix-Ï„ volume and plain coverage mask from the same loaded
    /// asset state. The stroke renderer needs both for the dynamics path.
    pub(crate) fn mask_views(
        &self,
        id: AssetId,
        orientation: stark_model::document::OrientationSource,
    ) -> Option<MaskViews> {
        let mut inner = unpoisoned(self.inner.lock());
        let mask = inner.masks.get_mut(&id)?;
        let prefix = if orientation == stark_model::document::OrientationSource::FollowStroke {
            mask.follow.clone()
        } else {
            if mask.pen.is_none() {
                let (w, h) = (mask.width, mask.height);
                let cov: Vec<f32> = mask.coverage.iter().map(|&b| b as f32 / 255.0).collect();
                let layers = orientation_layers(w, h);
                let rotated = rotate_layers(&cov, w, h, layers);
                mask.pen = Some(build_prefix_tau(&self.ctx, w, h, layers, &rotated));
            }
            mask.pen.clone().expect("a pen prefix was built above")
        };
        Some(MaskViews {
            prefix,
            coverage: mask.coverage_view.clone(),
        })
    }

    /// Whether `id` is loaded in this store.
    pub fn contains(&self, id: AssetId) -> bool {
        unpoisoned(self.inner.lock()).masks.contains_key(&id)
    }

    /// The canonical PNG bytes of one asset, if loaded — what a peer mirror or
    /// a second (preview) engine needs to reproduce the shape.
    pub fn bytes(&self, id: AssetId) -> Option<Vec<u8>> {
        unpoisoned(self.inner.lock())
            .masks
            .get(&id)
            .map(|m| m.bytes.clone())
    }

    /// Source bytes of every loaded asset, for bundling into the save file (§8).
    pub fn all_bytes(&self) -> Vec<(AssetId, Vec<u8>)> {
        unpoisoned(self.inner.lock())
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

/// How many orientation slices to build for a `width × height` volume: as many as
/// the memory budget allows, capped at [`MAX_ORIENTATION_LAYERS`] and at least 1.
pub fn orientation_layers(width: u32, height: u32) -> u32 {
    let per_layer = (width * height * 4).max(1);
    (PREFIX_BUDGET_BYTES / per_layer).clamp(1, MAX_ORIENTATION_LAYERS)
}

/// Pre-rotate a normalized `[-1, 1]²` coverage mask into `layers` orientation slices
/// on the mask's own grid (§6.6).
///
/// Slice `l` rotates the shape by `2π·l/layers` into the travel frame, so the sweep's
/// x-integral yields the swept depth at that orientation. Rotating **inside the
/// mask's own square** is sound because a canonical mask's content lies inside the
/// disc inscribed in it (`stark_assetid::coverage`), and a rotation maps that disc to
/// itself — nothing reaches the border at any angle. Bilinear sampling, zero outside
/// the source. Returns a `layers × height × width` buffer.
///
/// Slice 0 is *not* the identity — it is the mask resampled through the rotation
/// arithmetic at θ = 0. Nothing needs it to be: `FollowStroke`, the one caller that
/// would read layer 0 as the shape's native orientation, has its own single-layer
/// bake and never reads this at all.
fn rotate_layers(coverage: &[f32], width: u32, height: u32, layers: u32) -> Vec<f32> {
    let w = width as usize;
    let plane = w * height as usize;
    let mut out = vec![0.0f32; plane * layers as usize];
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
        let base = l * plane;
        for y in 0..height as usize {
            let py = (y as f32 + 0.5) / height as f32 * 2.0 - 1.0;
            for x in 0..w {
                let px = (x as f32 + 0.5) / width as f32 * 2.0 - 1.0;
                // Sample the source at R(-θ)·(px, py): the image rotates by +θ.
                let sx = px * c + py * s;
                let sy = -px * s + py * c;
                out[base + y * w + x] = sample(sx, sy);
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
/// rotated stack for pen) and the stroke renderer (the round tip, regenerated per
/// `hardness` — rotation-invariant, 1 layer). `coverage` is `layers × height × width`
/// row-major in `[0, 1]`. Every volume is baked on the mask's own grid, so one
/// column's brush-local width is `2/width` for all of them — it was a parameter while
/// the pen stack was padded and its columns stood for a wider span than they measured.
///
/// Returns the view alone: it holds its own reference to the texture, so there is
/// nothing for a caller to keep beside it.
pub fn build_prefix_tau(
    ctx: &GpuContext,
    width: u32,
    height: u32,
    layers: u32,
    coverage: &[f32],
) -> wgpu::TextureView {
    let w = width as usize;
    let dx = 2.0 / width as f32;
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

    /// **Turning a canonical shape must not change how much of it there is.**
    ///
    /// This is what the whole unpadded bake rests on: a canonical mask's content lies
    /// inside the disc inscribed in its square (`stark_assetid::coverage`), a rotation
    /// maps that disc to itself, and so no angle carries any of the mask off the edge
    /// of its own volume. The mask here reaches the disc's rim — the most a canonical
    /// mask can occupy, with structure on both axes so a loss would register — and
    /// every layer's total has to match the unrotated one's. While a mask could fill
    /// its corners, this exact property is what forced the `√2` padding.
    #[test]
    fn rotating_a_canonical_mask_loses_nothing() {
        const LAYERS: u32 = 8; // so layer 1 is the worst case, 45°
        let (w, h) = (48u32, 48u32);
        let cov: Vec<f32> = (0..w * h)
            .map(|i| {
                let x = ((i % w) as f32 + 0.5) / w as f32 * 2.0 - 1.0;
                let y = ((i / w) as f32 + 0.5) / h as f32 * 2.0 - 1.0;
                let d = (x * x + y * y).sqrt();
                if d < 0.92 {
                    0.8 * (1.0 - x.abs() * 0.4)
                } else {
                    0.0
                }
            })
            .collect();
        let rotated = rotate_layers(&cov, w, h, LAYERS);

        let plane = (w * h) as usize;
        let total = |l: usize| rotated[l * plane..(l + 1) * plane].iter().sum::<f32>();
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
