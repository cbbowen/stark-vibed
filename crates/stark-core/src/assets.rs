//! Content-addressed brush/image assets (DESIGN.md §6.6).
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
use std::io::Cursor;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::error::{EngineError, Result};
use crate::gpu::context::GpuContext;

/// Stable identity of an asset: the BLAKE3 hash of its source bytes.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AssetId(pub [u8; 32]);

impl std::fmt::Debug for AssetId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // First 8 hex chars are plenty to identify in logs.
        write!(f, "AssetId({:02x}{:02x}{:02x}{:02x}…)", self.0[0], self.0[1], self.0[2], self.0[3])
    }
}

struct Mask {
    /// Source bytes, retained so the asset can be bundled into the save file.
    bytes: Vec<u8>,
    view: wgpu::TextureView,
    #[allow(dead_code)]
    texture: wgpu::Texture,
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
        let (w, h, coverage) = decode_coverage(decode_from)?;
        let id = coverage_id(w, h, &coverage);
        let mut inner = self.inner.lock().expect("asset store poisoned");
        if let Entry::Vacant(slot) = inner.masks.entry(id) {
            let bytes = match store_bytes {
                Some(b) => b,
                None => encode_coverage_png(w, h, &coverage)?,
            };
            let cov: Vec<f32> = coverage.iter().map(|&b| b as f32 / 255.0).collect();
            // One coverage layer per shape orientation (DESIGN.md §6.6): the swept-depth
            // integral runs along the stroke's travel axis, so to orient the footprint at
            // an arbitrary angle we pre-rotate the shape into the travel frame, layer 0
            // being the identity (the native, follow-stroke orientation).
            let layers = orientation_layers(w, h);
            let rotated = rotate_layers(&cov, w, h, layers);
            let (texture, view) = build_prefix_tau(&self.ctx, w, h, layers, &rotated);
            slot.insert(Mask {
                bytes,
                view,
                texture,
            });
        }
        Ok(id)
    }

    /// A clonable view of the brush's prefix-τ texture for `id`, if loaded
    /// (the running integral of `−ln(1−coverage)` along the travel axis, §6.2).
    pub fn prefix_view(&self, id: AssetId) -> Option<wgpu::TextureView> {
        self.inner
            .lock()
            .expect("asset store poisoned")
            .masks
            .get(&id)
            .map(|m| m.view.clone())
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

/// Largest number of orientation slices a brush's prefix-τ volume holds (DESIGN.md
/// §6.6). With linear interpolation between adjacent layers this is ~5.6° resolution
/// — smooth for any practical pen rotation.
pub const MAX_ORIENTATION_LAYERS: u32 = 64;

/// Memory budget (bytes) for one brush's prefix-τ volume. The layer count is chosen so
/// `width × height × layers × 4 (R32Float)` stays under this — so a large detailed
/// stamp keeps its full resolution (layer 0 is the identity, so follow-stroke strokes
/// are unchanged) and trades orientation granularity for memory instead.
const PREFIX_BUDGET_BYTES: u32 = 64 << 20; // 64 MiB

/// How many orientation slices to build for a `width × height` shape: as many as the
/// memory budget allows, capped at [`MAX_ORIENTATION_LAYERS`] and at least 1.
pub fn orientation_layers(width: u32, height: u32) -> u32 {
    let per_layer = (width * height * 4).max(1);
    (PREFIX_BUDGET_BYTES / per_layer).clamp(1, MAX_ORIENTATION_LAYERS)
}

/// Pre-rotate a normalized `[-1, 1]²` coverage mask into `layers` orientation slices
/// (DESIGN.md §6.6). Slice `l` rotates the shape by `2π·l/layers` into the travel frame
/// (so the sweep's x-integral yields the swept depth at that orientation); slice 0 is the
/// identity, kept bit-exact so follow-stroke strokes match the un-rotated mask. Bilinear
/// sampling, zero outside the source. Returns a `layers × height × width` buffer.
fn rotate_layers(coverage: &[f32], width: u32, height: u32, layers: u32) -> Vec<f32> {
    let (w, h) = (width as usize, height as usize);
    let mut out = vec![0.0f32; w * h * layers as usize];
    // Layer 0: identity (bit-exact copy of the source).
    out[..w * h].copy_from_slice(coverage);
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
    for l in 1..layers as usize {
        let theta = std::f32::consts::TAU * l as f32 / layers as f32;
        let (s, c) = theta.sin_cos();
        let base = l * w * h;
        for y in 0..h {
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

/// Build a brush's **prefix-τ** volume (DESIGN.md §6.2, §6.6): for each orientation
/// `layer` and each row, the running integral of optical depth `κ = −ln(1−coverage)`
/// along the travel axis (x), normalized to brush-local units (x spans `[-1, 1]`, width
/// 2). Stored as a `R32Float` **2D-array** texture (the array axis is orientation, sampled
/// with wrapping) and read via `textureLoad` + manual trilinear by the sweep shader: a
/// segment's swept depth at a point is `prefix(u) − prefix(u−d)` on its layer. (A 2D array
/// rather than a true 3D texture so the mask keeps its full width/height — 3D textures are
/// capped far smaller, e.g. 256px, by `maxTextureDimension3D`.)
///
/// Shared by [`AssetStore`] (image brushes, at import — many layers) and the stroke
/// renderer (the round tip, regenerated per `hardness` — rotation-invariant, 1 layer).
/// `coverage` is `layers × height × width` row-major in `[0, 1]`.
pub fn build_prefix_tau(
    ctx: &GpuContext,
    width: u32,
    height: u32,
    layers: u32,
    coverage: &[f32],
) -> (wgpu::Texture, wgpu::TextureView) {
    let w = width as usize;
    let dx = 2.0 / width as f32; // brush-local width of one column
    let mut prefix = vec![0.0f32; coverage.len()];
    for y in 0..(height * layers) as usize {
        // Rows are contiguous across layers (layer-major, then row), so one linear pass
        // integrates every layer's rows independently.
        let mut acc = 0.0f32;
        for x in 0..w {
            let m = coverage[y * w + x].clamp(0.0, 0.999);
            acc += -(1.0 - m).ln() * dx;
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
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    (texture, view)
}

/// Content id of a coverage mask: the hash of its dimensions + pixels. Derived
/// from the decoded coverage (not the file bytes) so it is stable across source
/// encodings and PNG encoder versions — important for replay and collaboration.
fn coverage_id(width: u32, height: u32, coverage: &[u8]) -> AssetId {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&width.to_le_bytes());
    hasher.update(&height.to_le_bytes());
    hasher.update(coverage);
    AssetId(*hasher.finalize().as_bytes())
}

/// Encode a coverage buffer as a compact grayscale PNG for the save file (§8).
fn encode_coverage_png(width: u32, height: u32, coverage: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, width, height);
        encoder.set_color(png::ColorType::Grayscale);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(png::Compression::High);
        let mut writer = encoder
            .write_header()
            .map_err(|e| EngineError::Asset(e.to_string()))?;
        writer
            .write_image_data(coverage)
            .map_err(|e| EngineError::Asset(e.to_string()))?;
    }
    Ok(out)
}

/// Decode a PNG to a `width × height` single-channel coverage buffer.
///
/// Coverage = luminance × alpha, so white-on-black masks (luminance) and
/// alpha-cut masks both work. Palette/grayscale/16-bit inputs are normalized.
fn decode_coverage(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>)> {
    let mut decoder = png::Decoder::new(Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder
        .read_info()
        .map_err(|e| EngineError::Asset(e.to_string()))?;
    let mut buf = vec![0u8; reader.output_buffer_size().ok_or_else(|| EngineError::Asset("missing size".into()))?];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| EngineError::Asset(e.to_string()))?;
    buf.truncate(info.buffer_size());

    let n = (info.width * info.height) as usize;
    let mut coverage = vec![0u8; n];
    let lum = |r: u8, g: u8, b: u8| -> u32 {
        (77 * r as u32 + 150 * g as u32 + 29 * b as u32) >> 8
    };
    match info.color_type {
        png::ColorType::Grayscale => {
            coverage.copy_from_slice(&buf[..n]);
        }
        png::ColorType::GrayscaleAlpha => {
            for i in 0..n {
                let g = buf[i * 2] as u32;
                let a = buf[i * 2 + 1] as u32;
                coverage[i] = (g * a / 255) as u8;
            }
        }
        png::ColorType::Rgb => {
            for i in 0..n {
                coverage[i] = lum(buf[i * 3], buf[i * 3 + 1], buf[i * 3 + 2]) as u8;
            }
        }
        png::ColorType::Rgba => {
            for i in 0..n {
                let l = lum(buf[i * 4], buf[i * 4 + 1], buf[i * 4 + 2]);
                let a = buf[i * 4 + 3] as u32;
                coverage[i] = (l * a / 255) as u8;
            }
        }
        png::ColorType::Indexed => {
            return Err(EngineError::Asset("indexed PNG not expanded".into()));
        }
    }
    Ok((info.width, info.height, coverage))
}
