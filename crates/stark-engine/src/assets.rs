//! Content-addressed brush/image assets (§6.6).
//!
//! A brush *shape* is a grayscale coverage mask, identified by the BLAKE3 hash of
//! its **decoded, capped coverage** — not of the bytes it arrived in — so a
//! `StrokeRecord` references a 32-byte [`AssetId`] rather than embedding pixels,
//! and two encodings of one picture are the same asset. That is the identity
//! contract §19 freezes: the decode and the cap are part of what an id *means*,
//! which is why they live in `stark-assetid`, where a build script can compute one
//! without a GPU. See [`AssetStore::import`].
//!
//! The store keeps each mask's bytes, its decoded coverage, and the volumes the stroke
//! path reads it through — which the stroke path itself knows how to build
//! ([`volume`]). It is `Clone` (`Arc`-backed) so it can ride inside the
//! `Action::Context` alongside the tile pool and stroke renderer.

use std::collections::hash_map::{Entry, HashMap};
use std::sync::{Arc, Mutex};

use crate::device::context::GpuContext;
use crate::error::Result;
use crate::gpu::stroke::volume::{self, Lazy, ResolvedTip, VolumeCache};
use crate::unpoisoned;

use stark_model::AssetId;
use stark_model::document::OrientationSource;

/// A loaded brush shape: its source bytes and what one import measures off them.
///
/// The **views** alone, with no texture beside them: a `wgpu::TextureView` holds its
/// own reference to the texture it was made from, so the texture outlives it.
struct Mask {
    /// Source bytes, retained so the asset can be bundled into the save file.
    bytes: Vec<u8>,
    /// The canonical coverage the id names, retained so the lazily baked volumes can be
    /// built — and rebuilt after an eviction — without decoding the PNG again. One byte
    /// per texel, capped at [`MAX_SHAPE_DIM`](stark_model::MAX_SHAPE_DIM)², so at most a
    /// megabyte for the largest shape there is.
    coverage: Vec<u8>,
    width: u32,
    height: u32,
    /// The follow-stroke prefix-τ: **one layer, the identity**, since the shape's axis
    /// tracks the tangent and the relative angle is always 0 (§6.6). Built at import,
    /// because it is what nearly every stroke reads and it costs a single integral.
    follow: wgpu::TextureView,
    /// The plain (unrotated) coverage mask, for per-stamp footprint sampling in the
    /// brush-dynamics stamp loop (§6.2). Orientation rotates the sample coordinates,
    /// so one texture serves both sources — unlike the prefix-τ, whose integration
    /// axis is baked in.
    depth: wgpu::TextureView,
    /// The mask's **rise** (§6.13), radii — [`volume::mask_rise`], measured once at
    /// import.
    rise: f32,
}

impl Mask {
    /// The prefix-τ volume this orientation source reads (§6.6): the identity layer
    /// built at import for a follow stroke, and for the pen the rotated stack, baked on
    /// first ask and kept in `volumes` until the budget there evicts it.
    ///
    /// That bake is the largest single piece of work the store does — a rotation per
    /// layer over a volume `layers` times the size, and a `ln` per texel of it — and it
    /// runs inside the store lock, mid-gesture, on whichever stroke first asks.
    /// `asset.pen_bake` is its timing row.
    ///
    /// Lazy because the orientation source is a brush setting the store cannot see at
    /// import: baking it eagerly would charge every follow-stroke brush in the library
    /// for a mode it never enters.
    fn prefix(
        &self,
        ctx: &GpuContext,
        volumes: &VolumeCache,
        id: AssetId,
        orientation: OrientationSource,
    ) -> wgpu::TextureView {
        if orientation == OrientationSource::FollowStroke {
            return self.follow.clone();
        }
        let layers = volume::orientation_layers(self.width, self.height);
        let (w, h) = (self.width, self.height);
        volumes.baked((id, Lazy::Pen), volume::prefix_bytes(w, h, layers), || {
            crate::timing::span!("asset.pen_bake");
            let cov = volume::coverage_f32(&self.coverage);
            let mut rotated = volume::rotate_layers(&cov, w, h, layers);
            // In place: the rotated stack is the largest allocation the store
            // makes, and τ is a per-texel reading of the coverage it holds.
            rotated.iter_mut().for_each(|c| *c = volume::tau_of(*c));
            volume::build_prefix(ctx, w, h, layers, &rotated, volume::PREFIX_TAU_LABEL)
        })
    }

    /// The **coverage prefix** this orientation source's liquify follow reads (§6.13),
    /// baked on first ask like the pen volume above — and for the pen, the same
    /// rotation over again, so a pen-oriented liquify brush holds two volumes of the
    /// pen budget. Lazy because nothing but a liquify stroke reads either.
    /// `asset.warp_bake` is its row.
    fn warp_prefix(
        &self,
        ctx: &GpuContext,
        volumes: &VolumeCache,
        id: AssetId,
        orientation: OrientationSource,
    ) -> wgpu::TextureView {
        let (w, h) = (self.width, self.height);
        let layers = match orientation {
            OrientationSource::FollowStroke => 1,
            OrientationSource::Pen => volume::orientation_layers(w, h),
        };
        volumes.baked(
            (id, Lazy::Warp(orientation)),
            volume::prefix_bytes(w, h, layers),
            || {
                crate::timing::span!("asset.warp_bake");
                let cov = volume::coverage_f32(&self.coverage);
                let field = match orientation {
                    OrientationSource::FollowStroke => cov,
                    OrientationSource::Pen => volume::rotate_layers(&cov, w, h, layers),
                };
                volume::build_prefix(ctx, w, h, layers, &field, volume::COVERAGE_PREFIX_LABEL)
            },
        )
    }
}

/// GPU-resident cache of content-addressed coverage masks.
#[derive(Clone)]
pub(crate) struct AssetStore {
    ctx: GpuContext,
    masks: Arc<Mutex<HashMap<AssetId, Mask>>>,
    /// The volumes baked on first ask rather than at import ([`Mask::prefix`]), held
    /// apart from the masks because the bound on them is a bound across all of them.
    volumes: VolumeCache,
}

impl AssetStore {
    pub(crate) fn new(ctx: GpuContext) -> Self {
        Self {
            ctx,
            masks: Arc::new(Mutex::new(HashMap::new())),
            volumes: VolumeCache::default(),
        }
    }

    /// Import a brush image (any PNG), returning its content id — **the one way in**,
    /// for a shipped catalog entry, a file's bundle and a peer's offer alike.
    ///
    /// The id is the hash of the *decoded coverage* rather than of the file bytes, so it
    /// is independent of source encoding, and the stored form is re-encoded from that
    /// coverage. A saved bundle is already that encoding, so a reload re-derives what it
    /// was handed; anything else is canonicalized on the way in, which is what makes
    /// [`bytes`](Self::bytes)' promise unconditional.
    pub(crate) fn import(&self, bytes: &[u8]) -> Result<AssetId> {
        // Decode, cap and hash are the identity contract's, not ours (§19): the id
        // names the canonical coverage, so a reload of the stored PNG lands back on
        // the same id.
        let canonical = stark_assetid::coverage(bytes)?;
        let id = canonical.id();
        let stark_assetid::Canonical {
            width: w,
            height: h,
            texels: coverage,
        } = canonical;
        let mut masks = unpoisoned(self.masks.lock());
        if let Entry::Vacant(slot) = masks.entry(id) {
            let bytes = stark_assetid::Canonical {
                width: w,
                height: h,
                texels: coverage.clone(),
            }
            .encode()?;
            // Two readings of one τ field: the follow-stroke volume — the whole of the
            // bake for the common brush, one layer of the mask as it stands integrated
            // over its own width — and the same τ unintegrated, which is what the tool
            // side reads.
            let tau: Vec<f32> = volume::coverage_f32(&coverage)
                .iter()
                .map(|&c| volume::tau_of(c))
                .collect();
            slot.insert(Mask {
                bytes,
                follow: volume::build_prefix(&self.ctx, w, h, 1, &tau, volume::PREFIX_TAU_LABEL),
                depth: volume::build_depth_r16(&self.ctx, w, h, &tau),
                rise: volume::mask_rise(&coverage, w, h),
                coverage,
                width: w,
                height: h,
            });
        }
        Ok(id)
    }

    /// Resolve the volumes and plain coverage mask of one loaded asset from the same
    /// store state — **the one way to ask**, under one store lock. The stroke renderer
    /// needs several of them for the dynamics path, and asking twice could find the
    /// prefix hot and the coverage cold.
    ///
    /// **The two orientation sources read different volumes**, because they ask
    /// different questions of the same mask (§6.6). `FollowStroke` keeps the shape's
    /// axis on the tangent, so the relative angle is always 0 and one identity layer
    /// answers every segment. `Pen` lets the two diverge, which means rotating the
    /// mask inside the frame the sweep integrates along — safe in the mask's own
    /// square, because a canonical mask's content lies inside the disc inscribed in
    /// it (`stark_assetid::coverage`) and a rotation maps that disc to itself. The
    /// plain coverage is one texture for both sources: the loop samples it in the
    /// **shape's own frame**, rotating the lookup rather than the mask.
    ///
    /// `&self` throughout: the store is `Arc<Mutex<_>>` behind a `Clone`, and the
    /// lazily baked volumes are the one part of it that grows after import.
    pub(crate) fn mask_views(
        &self,
        id: AssetId,
        orientation: OrientationSource,
        warp: bool,
    ) -> Option<ResolvedTip> {
        let masks = unpoisoned(self.masks.lock());
        let mask = masks.get(&id)?;
        Some(ResolvedTip {
            prefix: mask.prefix(&self.ctx, &self.volumes, id, orientation),
            depth: mask.depth.clone(),
            warp: warp.then(|| mask.warp_prefix(&self.ctx, &self.volumes, id, orientation)),
            rise: mask.rise,
        })
    }

    /// Whether `id` is loaded in this store.
    pub(crate) fn contains(&self, id: AssetId) -> bool {
        unpoisoned(self.masks.lock()).contains_key(&id)
    }

    /// The canonical PNG bytes of one asset, if loaded — what a peer mirror or
    /// a second (preview) engine needs to reproduce the shape.
    pub(crate) fn bytes(&self, id: AssetId) -> Option<Vec<u8>> {
        unpoisoned(self.masks.lock())
            .get(&id)
            .map(|m| m.bytes.clone())
    }

    /// Source bytes of every loaded asset, for bundling into the save file (§8).
    pub(crate) fn all_bytes(&self) -> Vec<(AssetId, Vec<u8>)> {
        unpoisoned(self.masks.lock())
            .iter()
            .map(|(id, m)| (*id, m.bytes.clone()))
            .collect()
    }
}
