//! The textures a brush resolves to, and the caches behind them (§6.6, §6.2).
//!
//! One type because they are one question — *given a brush, what does the GPU read?*
//! — asked identically by both render paths, and because the answers are the only
//! mutable state in a renderer otherwise documented as holding "only immutable GPU
//! objects plus `Arc`-backed handles". Keeping them here is what lets that sentence
//! stay true of [`StrokeRenderer`](super::StrokeRenderer) itself.
//!
//! It is a complaint the module has made before, in the other direction:
//! `DynamicsKit` records that it "used to carry the round tip's coverage cache, the
//! one mutable thing in a struct documented as built-once", moved out to the renderer.
//! The renderer is where all of it then piled up.
//!
//! Cheap to clone with its renderer — everything is `Arc`-backed or a wgpu handle.

use std::sync::{Arc, Mutex};

use crate::assets::{AssetStore, build_coverage_r8, build_prefix_tau};
use crate::document::{BrushParams, BrushShape, ColorDynamics, NoiseKind};
use crate::gpu::context::GpuContext;

use super::segments::round_coverage;
use super::unpoisoned;

/// Resolution of the generated round-tip prefix texture.
pub(super) const ROUND_RES: u32 = 256;

/// The brush textures both paths resolve, and the lazily-baked caches behind them.
#[derive(Clone)]
pub(super) struct TipCache {
    ctx: GpuContext,
    /// The round tip's baked textures, keyed by `hardness.to_bits()` (§6.6).
    ///
    /// **One entry, replaced rather than accumulated** — and that is a fact about the
    /// key, not caution. Hardness is a continuous slider, so a live preview walks it
    /// through a fresh value per frame while the user drags one: keeping every
    /// position would bank ~320 KB of GPU texture apiece and never hand it back, while
    /// keeping the last is exactly the working set of *adjust the knob and look*.
    /// [`noise_cache`](Self::noise_cache) below grows without bound for the opposite
    /// reason — its key is a small enum, so the whole domain fits and nothing evicts.
    round_tip: Arc<Mutex<Option<(u32, RoundTip)>>>,
    /// Colour dynamics (§6.2): the shared wrap/linear sampler, the 1×1×1 zero volume
    /// bound when a brush's jitter is off, and the lazily-baked per-kind fields.
    pub(super) noise_sampler: wgpu::Sampler,
    dummy_noise: wgpu::TextureView,
    noise_cache: Arc<Mutex<Vec<(NoiseKind, wgpu::TextureView)>>>,
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
            round_tip: Arc::new(Mutex::new(None)),
            noise_sampler,
            dummy_noise,
            noise_cache: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// The brush's swept-footprint prefix-τ texture: an image brush's from the asset
    /// store, the round tip's generated (and cached) from its hardness.
    ///
    /// Both render paths resolve it the same way — they differ in which bind-group
    /// layout they hang it off, not in how the texture is chosen.
    ///
    /// The **orientation source** is part of the question for an image brush (§6.6):
    /// follow-stroke reads a single identity layer, pen a padded stack of them. A round
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
    /// `acos`/`exp` that used to be run twice for the same hardness, once per texture.
    /// Cached as one entry for a second reason — held apart, the stamp loop could find
    /// its prefix hot and its coverage cold, and pay the field again anyway.
    fn round_tip(&self, hardness: f32) -> RoundTip {
        let key = hardness.to_bits();
        let mut cache = unpoisoned(self.round_tip.lock());
        if let Some((k, tip)) = cache.as_ref()
            && *k == key
        {
            return tip.clone();
        }
        let cov = round_coverage(hardness, ROUND_RES);
        // The round tip is rotation-invariant, so a single orientation layer suffices —
        // the shader's wrapping lookup reads it for every orientation (§6.6).
        let prefix = build_prefix_tau(
            &self.ctx,
            ROUND_RES,
            ROUND_RES,
            1,
            2.0 / ROUND_RES as f32,
            &cov,
        );
        let bytes: Vec<u8> = cov.iter().map(|c| (c * 255.0).round() as u8).collect();
        let coverage = build_coverage_r8(&self.ctx, ROUND_RES, ROUND_RES, &bytes);
        let tip = RoundTip { prefix, coverage };
        *cache = Some((key, tip.clone()));
        tip
    }
    /// The colour-dynamics noise tile for a brush: the baked field for its
    /// kind (built once, cached — the bake is a fixed pure function, so at most
    /// one texture per [`NoiseKind`] ever exists), or the 1×1 zero tile when
    /// the jitter is off (amplitudes all 0 ⇒ the shader adds exactly nothing).
    pub(super) fn noise_view(&self, cd: &ColorDynamics) -> wgpu::TextureView {
        if !cd.is_active() {
            return self.dummy_noise.clone();
        }
        let mut cache = unpoisoned(self.noise_cache.lock());
        if let Some((_, view)) = cache.iter().find(|(k, _)| *k == cd.noise) {
            return view.clone();
        }
        let (_tex, view) = crate::noise::build_noise_texture(&self.ctx, cd.noise);
        cache.push((cd.noise, view.clone()));
        view
    }
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
