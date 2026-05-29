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
            let (texture, view) = self.upload_r8(w, h, &coverage);
            slot.insert(Mask {
                bytes,
                view,
                texture,
            });
        }
        Ok(id)
    }

    /// A clonable view of the mask for `id`, if loaded.
    pub fn mask_view(&self, id: AssetId) -> Option<wgpu::TextureView> {
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

    fn upload_r8(&self, width: u32, height: u32, data: &[u8]) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = self.ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("stark brush mask"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.ctx.queue.write_texture(
            texture.as_image_copy(),
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }
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
