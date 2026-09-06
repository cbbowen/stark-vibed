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

use crate::assets::{AssetStore, Integrand, build_coverage_r8, build_prefix};
use crate::gpu::context::GpuContext;
use stark_model::document::{BrushParams, BrushShape, ColorDynamics, NoiseKind};

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
/// banks ~590 KB of GPU texture per position and never hands it back.
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

/// One of [`TipCache`]'s two least-recently-used lists: newest last, shared across
/// the renderer's clones, and bounded by the `*_KEPT` constant beside the field.
///
/// A `Vec` rather than a map because the bound is four: a linear scan of four keys
/// is cheaper than a hash, and the order *is* the recency, which a map would have to
/// carry separately.
type Lru<K, V> = Arc<Mutex<Vec<(K, V)>>>;

/// The brush textures both paths resolve, and the lazily-baked caches behind them.
#[derive(Clone)]
pub(super) struct TipCache {
    ctx: GpuContext,
    /// The round tips' baked textures, keyed by the **effective** hardness's bits —
    /// the brush's own floored by its size (`budget::effective_hardness`, §6.6) — an
    /// LRU of [`ROUND_TIPS_KEPT`], newest last. Only a brush the floor binds (hard
    /// *and* small) re-bakes as its size changes; every other key is the hardness
    /// alone, as it always was.
    round_tip: Lru<u32, RoundTip>,
    /// Color dynamics (§6.2): the shared wrap/linear sampler, the 1×1 zero tile
    /// bound when a brush's jitter is off, and the per-stroke baked fields — an LRU
    /// of [`NOISE_TILES_KEPT`] keyed by (kind, stroke seed), newest last.
    pub(super) noise_sampler: wgpu::Sampler,
    dummy_noise: Arc<NoiseTile>,
    noise_tiles: Lru<NoiseKey, Arc<NoiseTile>>,
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
        let (dummy_tex, dummy_noise) = crate::noise::dummy_noise_texture(ctx);
        Self {
            ctx: ctx.clone(),
            round_tip: Arc::new(Mutex::new(Vec::new())),
            noise_sampler,
            dummy_noise: Arc::new(NoiseTile {
                texture: dummy_tex,
                view: dummy_noise,
            }),
            noise_tiles: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// The brush's swept-extent prefix-τ texture: an image brush's from the asset
    /// store, the round tip's generated (and cached) from its hardness — with the
    /// plain coverage beside it, and, for a liquify brush, the coverage prefix its
    /// follow reads and the tip's rise the step budget prices (§6.13).
    ///
    /// Both render paths resolve it the same way — they differ in which bind-group
    /// layout they hang it off, not in how the texture is chosen.
    ///
    /// The **orientation source** is part of the question for an image brush (§6.6):
    /// follow-stroke reads a single identity layer, pen a stack of them. A round
    /// tip is rotation-invariant and answers both with the same one slice, which is why
    /// it is asked only for its hardness — floored by the brush's own size
    /// (`budget::effective_hardness`, §6.6), so a hard edge keeps a ~px of
    /// antialiased rim at any radius.
    pub(super) fn resolve(&self, assets: &AssetStore, brush: &BrushParams) -> Option<ResolvedTip> {
        let warping = brush.liquify().is_some();
        match brush.shape {
            BrushShape::Stamp(id) => {
                assets
                    .mask_views(id, brush.orientation, warping)
                    .map(|views| ResolvedTip {
                        prefix: views.prefix,
                        coverage: views.coverage,
                        warp: views.warp,
                        rise: views.rise,
                    })
            }
            BrushShape::Round { hardness } => {
                let tip = self.round_tip(super::budget::effective_hardness(hardness, brush.size));
                Some(ResolvedTip {
                    prefix: tip.prefix,
                    coverage: tip.coverage,
                    warp: warping.then_some(tip.warp),
                    rise: tip.rise,
                })
            }
        }
    }

    /// The round tip's baked textures for a given `hardness`, cached so live preview
    /// — which re-renders per pointer move — doesn't rebuild them each frame.
    ///
    /// The set is built and cached **together**, off a single [`round_coverage`]
    /// evaluation, because they are readings of one field: 256² texels of
    /// `acos`/`exp`, which a texture each would run again for the same hardness.
    /// Cached as one entry for a second reason — held apart, the stamp loop could find
    /// its prefix hot and its coverage cold, and pay the field again anyway. The
    /// coverage prefix is baked eagerly with them rather than on a liquify brush's
    /// first ask: it is one linear pass over a field already in hand, against a
    /// cache miss that has just paid the transcendental one.
    fn round_tip(&self, hardness: f32) -> RoundTip {
        let mut cache = unpoisoned(self.round_tip.lock());
        let (tip, _evicted) = lru(&mut cache, hardness.to_bits(), ROUND_TIPS_KEPT, || {
            let cov = round_coverage(hardness, ROUND_RES);
            // The round tip is rotation-invariant, so a single orientation layer
            // suffices — the shader's wrapping lookup reads it for every
            // orientation (§6.6).
            let prefix = build_prefix(&self.ctx, ROUND_RES, ROUND_RES, 1, &cov, Integrand::Tau);
            let warp = build_prefix(
                &self.ctx,
                ROUND_RES,
                ROUND_RES,
                1,
                &cov,
                Integrand::Coverage,
            );
            let bytes: Vec<u8> = cov.iter().map(|c| (c * 255.0).round() as u8).collect();
            let coverage = build_coverage_r8(&self.ctx, ROUND_RES, ROUND_RES, &bytes);
            RoundTip {
                prefix,
                coverage,
                warp,
                rise: round_rise(hardness),
            }
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
    pub(super) fn noise(&self, cd: &ColorDynamics, seed: u32) -> NoiseLease {
        if !cd.is_active() {
            return NoiseLease(Arc::clone(&self.dummy_noise));
        }
        let mut cache = unpoisoned(self.noise_tiles.lock());
        let (tile, evicted) = lru(&mut cache, (cd.noise, seed), NOISE_TILES_KEPT, || {
            let (texture, view) = crate::noise::build_noise_texture(&self.ctx, cd.noise, seed);
            Arc::new(NoiseTile { texture, view })
        });
        drop(evicted);
        NoiseLease(tile)
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
struct NoiseTile {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl Drop for NoiseTile {
    fn drop(&mut self) {
        // A render holds a `NoiseLease` in its submit scope, so an LRU eviction can
        // only destroy this texture after the last command buffer that names it has
        // been submitted.
        self.texture.destroy();
    }
}

/// A color-dynamics texture kept alive until the command buffer that samples it is
/// submitted. A texture view alone does not keep an LRU eviction from destroying its
/// source texture.
#[derive(Clone)]
pub(super) struct NoiseLease(Arc<NoiseTile>);

impl NoiseLease {
    pub(super) fn view(&self) -> &wgpu::TextureView {
        &self.0.view
    }
}

/// The textures selected together for one brush: the prefix-τ volume and the
/// coverage mask every path reads, and — for a liquify brush alone — the coverage
/// prefix its follow reads (§6.13), with the tip's **rise** beside it.
pub(super) struct ResolvedTip {
    pub(super) prefix: wgpu::TextureView,
    pub(super) coverage: wgpu::TextureView,
    /// The **coverage prefix** (`assets::Integrand::Coverage`): the same volume as
    /// `prefix`, integrated linearly, which the liquify `warp` kernel reads at the
    /// prefix-τ's own binding so a follow is the mask's mean over the pass rather
    /// than its optical depth. `None` on every other effect, which never asks.
    pub(super) warp: Option<wgpu::TextureView>,
    /// The tip's rise (§6.13), radii: the shortest travel over which its coverage
    /// climbs by `budget::WARP_CONTRACTION`. What the liquify step budget prices
    /// (`budget::liquify_len`); measured off the mask, whichever kind it is.
    pub(super) rise: f32,
}

/// A baked round tip: the **prefix-τ** volume both render paths integrate the swept
/// deposit against, the plain **coverage** mask the stamp loop's reservoir texels
/// weight by, the **coverage prefix** the liquify follow reads (§6.13), and the
/// tip's rise.
///
/// One type because they are one thing — the same coverage field, read four ways —
/// and keeping them so is what makes a cache entry able to say it holds *the tip*
/// rather than a texture that happens to be a tip's.
#[derive(Clone)]
struct RoundTip {
    prefix: wgpu::TextureView,
    coverage: wgpu::TextureView,
    warp: wgpu::TextureView,
    rise: f32,
}

/// What a noise tile is cached by: the brush's kind and the stroke's seed.
type NoiseKey = (NoiseKind, u32);

/// Generate the round tip's coverage: the soft disc whose *swept* profile across the
/// stroke is `1 − |y|^h`, for `h = 1/(1 − hardness)` and `y` the distance from the
/// centreline in radii.
///
/// The profile is what is being designed here, not the extent. What `hardness`
/// names is how the *stroke* falls off from its centreline; the tip that produces it
/// is whatever it has to be, and it is not the profile's own shape — a swept deposit
/// composes in **optical depth**, so a full pass lays `1 − exp(−τ(y))` where `τ` is
/// this mask's `κ = −ln(1 − coverage)` integrated along the travel axis
/// ([`build_prefix`]). Ask instead for the field
/// whose row integrals are
///
/// ```text
/// τ(y) = −h·ln|y|        (so 1 − exp(−τ(y)) = 1 − |y|^h, as wanted)
/// ```
///
/// and that is an Abel transform, which inverts in closed form: the radial
///
/// ```text
/// κ(r) = (h/π)·acos(r)/r,   r < 1
/// ```
///
/// has exactly those integrals. So the tip is `1 − exp(−κ(r))` and the profile is
/// arrived at rather than approached. Rate scales the exponent rather than leaving the
/// family — a pass at strength `a` lays `1 − |y|^(a·h)`, the same shape at another
/// hardness — and the field is radially symmetric, as a round tip's ought to be.
///
/// What this replaces aimed at the same profile through the *linear* integral: a
/// `1 − r^h` disc divided by its own chord half-length, `1/√(1 − y²)`. The log in
/// between is what it did not account for, and it is not a small correction, because
/// `−ln(1 − c)` weights the high-coverage core far above the rim: the stroke came out
/// fuller than its hardness named everywhere, by 0.08 in coverage at `hardness = 0`
/// and by 0.54 at `hardness = 0.9`, with the whole falloff crushed into the last few
/// texels of the rim — and on a hard tip the flanks left the mask above coverage 1
/// entirely, where the clamp ate the overshoot.
///
/// `κ` diverges at the centre, as it must for a profile that reaches exactly 1 there,
/// so the core saturates against that same 0.999 clamp and lands a shade under 1
/// instead. Outside it the profile is exact to a thousandth (`tests`, below).
fn round_coverage(hardness: f32, res: u32) -> Vec<f32> {
    let mut cov = vec![0.0f32; (res * res) as usize];
    for y in 0..res {
        let fy = (y as f32 + 0.5) / res as f32 * 2.0 - 1.0;
        for x in 0..res {
            let fx = (x as f32 + 0.5) / res as f32 * 2.0 - 1.0;
            cov[(y * res + x) as usize] = round_coverage_at(hardness, (fx * fx + fy * fy).sqrt());
        }
    }
    cov
}

/// [`round_coverage`]'s field at radius `r` ∈ [0, ∞), radii: `1 − exp(−κ(r))`, zero
/// outside the disc, and `+∞` at a centre exactly hit (`acos(0)/0`), which is the
/// one place the profile asks for a coverage of exactly 1.
fn round_coverage_at(hardness: f32, r: f32) -> f32 {
    let h = 1.0 / (1.0 - hardness).max(0.01);
    let kappa = if r < 1.0 {
        h * r.acos() / (std::f32::consts::PI * r)
    } else {
        0.0
    };
    1.0 - (-kappa).exp()
}

/// How finely [`round_rise`] samples the radial profile: the rise is answered to a
/// half-thousandth of a radius, which under the budget's texel floor is finer than
/// any tip narrower than two thousand px can show.
const RISE_SAMPLES: usize = 2048;

/// The round tip's **rise** (§6.13), in radii: the shortest travel over which its
/// coverage climbs by [`WARP_CONTRACTION`](super::budget::WARP_CONTRACTION),
/// anywhere along a pass — what the liquify step budget prices.
///
/// Along the centreline, where the pass crosses the profile head-on; any other
/// chord crosses it obliquely and climbs slower, so this is the worst case. The
/// profile is monotone in the radius, so the climb over a window is monotone in
/// the window's width and the shortest window that climbs enough is a binary
/// search over widths, each a sweep of the sampled profile.
///
/// Sampled rather than solved because the steepest window is not always at the
/// rim: a hard tip climbs fastest where its `√(1 − r)` cusp meets the edge, a soft
/// one somewhere inside, and one search answers both. The rim itself is the last
/// sample — a hard tip's whole climb sits inside the final ten-thousandth of a
/// radius, and a grid that stopped short of it would find no climb at all.
pub(super) fn round_rise(hardness: f32) -> f32 {
    let n = RISE_SAMPLES;
    let cov: Vec<f32> = (0..=n)
        .map(|i| round_coverage_at(hardness, i as f32 / n as f32))
        .collect();
    let climbs = |width: usize| -> bool {
        (0..=n - width).any(|i| cov[i] - cov[i + width] >= super::budget::WARP_CONTRACTION)
    };
    // The centre reaches 1 and the rim 0, so the widest window always climbs;
    // the search is for the narrowest that does.
    let (mut lo, mut hi) = (0usize, n);
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if climbs(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi as f32 / n as f32
}

/// [`round_rise`] of the tip `b` bakes — its hardness floored by its size, as
/// [`TipCache::resolve`] bakes it — and nothing for a stamp, whose rise only the
/// store that holds its mask can measure. For the tests that price a brush without
/// resolving it (`budget::flatten_tolerance`).
#[cfg(test)]
pub(super) fn round_rise_of(b: &BrushParams) -> f32 {
    match b.shape {
        BrushShape::Round { hardness } => {
            round_rise(super::budget::effective_hardness(hardness, b.size))
        }
        BrushShape::Stamp(_) => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rise narrows as the tip hardens (§6.13) — a harder tip climbs its
    /// coverage over a shorter travel — and lands where the profile puts it: at
    /// hardness 0.8 the coverage climbs its ½ over a twelfth of a radius, so
    /// that is the rise; at 1 it is the cusp's last thousandth or so, which the
    /// budget's texel floor then takes over. Only from the middle of the range:
    /// the softest tips climb steepest at their *centre*, where `κ = h/(2r)`
    /// spikes, and that spike is narrower at hardness 0 than the mid-profile
    /// climb of a slightly harder tip, so the two softest rises are merely finite.
    #[test]
    fn the_round_tips_rise_narrows_with_hardness() {
        for hardness in [0.0f32, 0.25] {
            let rise = round_rise(hardness);
            assert!(rise > 0.1 && rise < 1.0, "hardness {hardness}: rise {rise}");
        }
        let mut last = f32::INFINITY;
        for hardness in [0.5f32, 0.8, 0.95, 1.0] {
            let rise = round_rise(hardness);
            assert!(
                rise > 0.0 && rise < last,
                "hardness {hardness}: rise {rise} after {last}"
            );
            last = rise;
        }
        let at = round_rise(0.8);
        assert!(
            (0.06..=0.11).contains(&at),
            "hardness 0.8 rises over {at} radii"
        );
        assert!(
            round_rise(1.0) < 0.01,
            "a hard tip rises inside its last texels"
        );
        // The rise is a rise: the sampled profile really climbs that much over it.
        let n = RISE_SAMPLES;
        let width = (at * n as f32) as usize;
        let climbed = (0..=n - width)
            .map(|i| {
                round_coverage_at(0.8, i as f32 / n as f32)
                    - round_coverage_at(0.8, (i + width) as f32 / n as f32)
            })
            .fold(0.0f32, f32::max);
        assert!(
            climbed >= super::super::budget::WARP_CONTRACTION,
            "over its rise the profile climbs {climbed}, less than the bound",
        );
    }

    /// The whole claim [`round_coverage`] makes, checked where it is a claim: a full
    /// pass of the tip lays `1 − |y|^h` across the stroke.
    ///
    /// Swept through the very integral the GPU volume is built from — the row sum
    /// `assets::build_prefix` does, sharing its `tau_of` so the clamp cannot drift
    /// between the two — which is what makes this a test of the tip rather than of a
    /// restatement of it. Inside `|y| < 0.2` the profile is past 0.99 for every
    /// hardness and the clamped core takes over, so that is where the pin stops; the
    /// centre's saturation is the tip's one documented departure.
    #[test]
    fn the_round_tip_sweeps_to_the_profile_its_hardness_names() {
        const RES: u32 = ROUND_RES;
        for hardness in [0.0, 0.25, 0.5, 0.8, 0.95] {
            let h = 1.0 / (1.0 - hardness);
            let cov = round_coverage(hardness, RES);
            for row in 0..RES {
                let y = ((row as f32 + 0.5) / RES as f32 * 2.0 - 1.0).abs();
                if y < 0.2 {
                    continue;
                }
                // The row's optical depth, as the sweep sees it after the tip has
                // passed over: every column of the mask, in brush-local width.
                let tau: f32 = (0..RES)
                    .map(|x| crate::assets::tau_of(cov[(row * RES + x) as usize]))
                    .sum::<f32>()
                    * (2.0 / RES as f32);
                let laid = 1.0 - (-tau).exp();
                let want = 1.0 - y.powf(h);
                assert!(
                    (laid - want).abs() < 2e-3,
                    "hardness {hardness}: at y = {y:.4} the sweep lays {laid:.5}, \
                     not the {want:.5} its profile names",
                );
            }
        }
    }
}
