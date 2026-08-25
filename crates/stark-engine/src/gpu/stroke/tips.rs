//! The textures a brush resolves to, and the caches behind them (§6.6, §6.2).
//!
//! One type because they are one question — *given a brush, what does the GPU read?*
//! — asked identically by both render paths, and because the answers are the only
//! mutable state in a renderer otherwise documented as holding "only immutable GPU
//! objects plus `Arc`-backed handles". Keeping them here is what lets that sentence
//! stay true of [`StrokeRenderer`](super::StrokeRenderer) itself.
//!
//! `DynamicsKit` is the same rule from the other side: it holds built-once GPU objects
//! and no cache, because a lazily-baked texture in a struct documented as immutable is
//! the thing this module exists to collect.
//!
//! Cheap to clone with its renderer — everything is `Arc`-backed or a wgpu handle.

use std::sync::{Arc, Mutex};

use crate::assets::{AssetStore, build_coverage_r8, build_prefix_tau};
use crate::gpu::context::GpuContext;
use stark_model::document::{BrushParams, BrushShape, ColorDynamics, NoiseKind};

use super::segments::round_coverage;
use crate::unpoisoned;

/// Resolution of the generated round-tip prefix texture.
pub(super) const ROUND_RES: u32 = 256;

/// How many round tips [`TipCache`] keeps baked at once.
///
/// More than one because one brush is not the working set: two peers painting
/// concurrently at different hardness (§12), or a replay interleaving strokes from
/// different brushes, alternate keys on every render — and a single entry re-bakes
/// 256² of `acos`/`exp` plus two texture uploads per miss, per frame. Four covers a
/// handful of simultaneous brushes; a hardness slider still walks through fresh
/// values per frame, which is why this is an LRU of a few rather than a map that
/// banks ~320 KB of GPU texture per position and never hands it back.
const ROUND_TIPS_KEPT: usize = 4;

/// How many color-dynamics tiles [`TipCache`] keeps baked at once.
///
/// A tile is one stroke's (§6.2), so what has to stay hot is the stroke being
/// *re*-rendered: the live one, per pointer move, and the brush editor's pinned
/// preview per edit. A few more cover a peer's live stroke interleaving with it
/// (§12). A replay reuses nothing — every stroke is a fresh seed — and bakes its
/// way through whatever this says. Evicted tiles are `destroy()`ed rather than
/// dropped: they go at the rate strokes do, and a dropped texture is not a freed
/// one (`submit.rs`).
const NOISE_TILES_KEPT: usize = 4;

/// The brush textures both paths resolve, and the lazily-baked caches behind them.
#[derive(Clone)]
pub(super) struct TipCache {
    ctx: GpuContext,
    /// The round tips' baked textures, keyed by `hardness.to_bits()` (§6.6): an LRU
    /// of [`ROUND_TIPS_KEPT`], newest last.
    round_tip: Arc<Mutex<Vec<(u32, RoundTip)>>>,
    /// Color dynamics (§6.2): the shared wrap/linear sampler, the 1×1 zero tile
    /// bound when a brush's jitter is off, and the per-stroke baked fields — an LRU
    /// of [`NOISE_TILES_KEPT`] keyed by (kind, stroke seed), newest last.
    pub(super) noise_sampler: wgpu::Sampler,
    dummy_noise: wgpu::TextureView,
    noise_tiles: Arc<Mutex<Vec<(NoiseKey, NoiseTile)>>>,
}

impl TipCache {
    pub(super) fn new(ctx: &GpuContext) -> Self {
        // Wrapping on both axes — the noise tile tiles (that's the whole point).
        let noise_sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("stark noise sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let (_dummy_tex, dummy_noise) = crate::noise::dummy_noise_texture(ctx);
        Self {
            ctx: ctx.clone(),
            round_tip: Arc::new(Mutex::new(Vec::new())),
            noise_sampler,
            dummy_noise,
            noise_tiles: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// The brush's swept-extent prefix-τ texture: an image brush's from the asset
    /// store, the round tip's generated (and cached) from its hardness.
    ///
    /// Both render paths resolve it the same way — they differ in which bind-group
    /// layout they hang it off, not in how the texture is chosen.
    ///
    /// The **orientation source** is part of the question for an image brush (§6.6):
    /// follow-stroke reads a single identity layer, pen a stack of them. A round
    /// tip is rotation-invariant and answers both with the same one slice, which is why
    /// it is asked only for its hardness.
    pub(super) fn prefix_view(
        &self,
        assets: &AssetStore,
        brush: &BrushParams,
    ) -> wgpu::TextureView {
        match brush.shape {
            BrushShape::Stamp(id) => assets
                .prefix_view(id, brush.orientation)
                .unwrap_or_else(|| self.round_tip(BrushShape::DEFAULT_HARDNESS).prefix),
            BrushShape::Round { hardness } => self.round_tip(hardness).prefix,
        }
    }

    /// The brush's plain coverage mask — the weights a reservoir texel carries
    /// (§6.2). Resolved exactly as [`Self::prefix_view`] is, from the same two
    /// sources; only the stamp loop asks for it.
    pub(super) fn coverage_view(
        &self,
        assets: &AssetStore,
        brush: &BrushParams,
    ) -> wgpu::TextureView {
        match brush.shape {
            BrushShape::Stamp(id) => assets
                .coverage_view(id)
                .unwrap_or_else(|| self.round_tip(BrushShape::DEFAULT_HARDNESS).coverage),
            BrushShape::Round { hardness } => self.round_tip(hardness).coverage,
        }
    }

    /// The round tip's baked textures for a given `hardness`, cached so live preview
    /// — which re-renders per pointer move — doesn't rebuild them each frame.
    ///
    /// The pair is built and cached **together**, off a single [`round_coverage`]
    /// evaluation, because they are two readings of one field: 256² texels of
    /// `acos`/`exp`, which a texture each would run twice for the same hardness.
    /// Cached as one entry for a second reason — held apart, the stamp loop could find
    /// its prefix hot and its coverage cold, and pay the field again anyway.
    fn round_tip(&self, hardness: f32) -> RoundTip {
        let mut cache = unpoisoned(self.round_tip.lock());
        let (tip, _evicted) = lru(&mut cache, hardness.to_bits(), ROUND_TIPS_KEPT, || {
            let cov = round_coverage(hardness, ROUND_RES);
            // The round tip is rotation-invariant, so a single orientation layer
            // suffices — the shader's wrapping lookup reads it for every
            // orientation (§6.6).
            let prefix = build_prefix_tau(&self.ctx, ROUND_RES, ROUND_RES, 1, &cov);
            let bytes: Vec<u8> = cov.iter().map(|c| (c * 255.0).round() as u8).collect();
            let coverage = build_coverage_r8(&self.ctx, ROUND_RES, ROUND_RES, &bytes);
            RoundTip { prefix, coverage }
        });
        // An eviction is dropped rather than `destroy()`ed: unlike the per-stroke
        // resources, they happen at the *rate the brush changes*, not per pointer
        // move, so JS GC keeps up fine.
        tip
    }

    /// The color-dynamics noise tile for a brush on the stroke seeded `seed`: the
    /// field baked for that stroke (`noise.rs`), cached so a live preview — which
    /// re-renders per pointer move — bakes it once; or the 1×1 zero tile when the
    /// jitter is off (amplitudes all 0 ⇒ the shader adds exactly nothing).
    pub(super) fn noise_view(&self, cd: &ColorDynamics, seed: u32) -> wgpu::TextureView {
        if !cd.is_active() {
            return self.dummy_noise.clone();
        }
        let mut cache = unpoisoned(self.noise_tiles.lock());
        let (tile, evicted) = lru(&mut cache, (cd.noise, seed), NOISE_TILES_KEPT, || {
            let (texture, view) = crate::noise::build_noise_texture(&self.ctx, cd.noise, seed);
            NoiseTile { texture, view }
        });
        if let Some(old) = evicted {
            // Safe while its last reader is in flight: a stroke binds its tile and
            // submits within one call, and a destroy defers behind submitted work
            // (`submit.rs`).
            old.texture.destroy();
        }
        tile.view
    }
}

/// `key`'s entry in a small LRU held as a `Vec` newest-last — the hit moved to the
/// back, or `build`'s result pushed there — and, past `kept` entries, the oldest
/// handed back for the caller to release. A GPU resource dropped is not one freed
/// (`submit.rs`), so which release it gets is the caller's to say.
fn lru<K: PartialEq, V: Clone>(
    cache: &mut Vec<(K, V)>,
    key: K,
    kept: usize,
    build: impl FnOnce() -> V,
) -> (V, Option<V>) {
    if let Some(i) = cache.iter().position(|(k, _)| *k == key) {
        let hit = cache.remove(i);
        let v = hit.1.clone();
        cache.push(hit);
        return (v, None);
    }
    let v = build();
    cache.push((key, v.clone()));
    let evicted = (cache.len() > kept).then(|| cache.remove(0).1);
    (v, evicted)
}

/// One stroke's baked color-dynamics field: the texture kept beside its view so an
/// eviction can `destroy()` it.
#[derive(Clone)]
struct NoiseTile {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

/// reservoir texels weight by.
///
/// One type because they are one thing — the same coverage field, read two ways —
/// and keeping them so is what makes a cache entry able to say it holds *the tip*
/// rather than a texture that happens to be a tip's.
#[derive(Clone)]
struct RoundTip {
    prefix: wgpu::TextureView,
    coverage: wgpu::TextureView,
}

/// What a noise tile is cached by: the brush's kind and the stroke's seed.
type NoiseKey = (NoiseKind, u32);
