//! The textures a brush resolves to, and the caches behind them (§6.6, §6.2).
//!
//! One type because they are one question — *given a brush, what does the GPU read?*
//! — asked identically by both render paths, and because the answers are the only
//! mutable state in a renderer otherwise documented as holding "only immutable GPU
//! objects plus `Arc`-backed handles". Keeping them here is what lets that sentence
//! stay true of [`StrokeRenderer`](super::StrokeRenderer) itself, and it is why
//! `DynamicsKit` beside it holds built-once GPU objects and no cache.
//!
//! Cheap to clone with its renderer — everything is `Arc`-backed or a wgpu handle.

use std::sync::{Arc, Mutex};

use crate::assets::{AssetStore, COVERAGE_PREFIX_LABEL, build_depth_r16, build_prefix};
use crate::gpu::context::GpuContext;
use stark_model::document::{BrushParams, BrushShape, ColorDynamics, NoiseKind};

use crate::unpoisoned;

/// Resolution of the generated round-tip prefix texture.
pub(super) const ROUND_RES: u32 = 256;

/// How many round tips [`TipCache`] keeps baked at once.
///
/// More than one because one brush is not the working set: two peers painting
/// concurrently at different hardness (§12), or a replay interleaving strokes from
/// different brushes, alternate keys on every render, and a miss re-bakes 256² of
/// `acos`/`exp` plus two texture uploads. Four covers a handful of simultaneous
/// brushes; an LRU of a few rather than a map, because a hardness slider walks through
/// fresh values per frame and each banks ~590 KB of GPU texture.
const ROUND_TIPS_KEPT: usize = 4;

/// How many color-dynamics tiles [`TipCache`] keeps baked at once.
///
/// A tile is one stroke's (§6.2), so what has to stay hot is the stroke being
/// *re*-rendered: the live one, per pointer move, and the brush editor's pinned preview
/// per edit, with a few more for a peer's live stroke (§12). Evicted tiles are
/// `destroy()`ed rather than dropped: they go at the rate strokes do, and a dropped
/// texture is not a freed one (`submit.rs`).
const NOISE_TILES_KEPT: usize = 4;

/// One of [`TipCache`]'s two least-recently-used lists: newest last, shared across the
/// renderer's clones, bounded by the `*_KEPT` constant beside the field. A `Vec` rather
/// than a map because the bound is four — a linear scan beats a hash, and the order *is*
/// the recency.
type Lru<K, V> = Arc<Mutex<Vec<(K, V)>>>;

/// The brush textures both paths resolve, and the lazily-baked caches behind them.
#[derive(Clone)]
pub(super) struct TipCache {
    ctx: GpuContext,
    /// The round tips' baked textures, keyed by the **effective** hardness's bits — the
    /// brush's own floored by its size (`budget::effective_hardness`, §6.6) — an LRU of
    /// [`ROUND_TIPS_KEPT`], newest last. Only a brush the floor binds (hard *and* small)
    /// re-bakes as its size changes.
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
    /// unintegrated depth beside it, and, for a liquify brush, the coverage prefix its
    /// follow reads and the tip's rise the step budget prices (§6.13).
    ///
    /// Both render paths resolve it the same way; they differ in which bind-group
    /// layout they hang it off, not in how the texture is chosen.
    ///
    /// The **orientation source** is part of the question for an image brush (§6.6):
    /// follow-stroke reads a single identity layer, pen a stack of them. A round tip is
    /// rotation-invariant and answers both with one slice, so it is asked only for its
    /// hardness — floored by the brush's own size (`budget::effective_hardness`), so a
    /// hard edge keeps a ~px of antialiased rim at any radius.
    pub(super) fn resolve(&self, assets: &AssetStore, brush: &BrushParams) -> Option<ResolvedTip> {
        let warping = brush.liquify().is_some();
        match brush.shape {
            BrushShape::Stamp(id) => {
                assets
                    .mask_views(id, brush.orientation, warping)
                    .map(|views| ResolvedTip {
                        prefix: views.prefix,
                        depth: views.depth,
                        warp: views.warp,
                        rise: views.rise,
                    })
            }
            BrushShape::Round { hardness } => {
                let tip = self.round_tip(super::budget::effective_hardness(hardness, brush.size));
                Some(ResolvedTip {
                    prefix: tip.prefix,
                    depth: tip.depth,
                    warp: warping.then_some(tip.warp),
                    rise: tip.rise,
                })
            }
        }
    }

    /// The round tip's baked textures for a given `hardness`, cached so live preview
    /// — which re-renders per pointer move — doesn't rebuild them each frame.
    ///
    /// Built and cached **together**, off a single [`round_depth`] evaluation,
    /// because they are readings of one field: 256² texels of `acos`/`exp` a texture
    /// each would run again for the same hardness, and held apart the stamp loop could
    /// find its prefix hot and its depth cold. The coverage prefix is baked with them
    /// rather than on a liquify brush's first ask — one linear pass over a field already
    /// in hand.
    ///
    /// **The depth is the field and the coverage is a reading of it**, not the other way
    /// round: the prefix-τ integrates `κ` as [`round_depth`] states it, where handing it
    /// the coverage and letting `assets::tau_of` recover `κ` capped the core at that
    /// function's mask ceiling (τ ≈ 6.9). The two orderings agree wherever `κ` stays
    /// under the ceiling and nowhere else — over the disc where it does not, the
    /// recovered field was flat, which the sweep laid as a plateau across the middle of
    /// every stroke. That disc is 0.07 of the radius at hardness 0 and 0.71 at 0.95.
    fn round_tip(&self, hardness: f32) -> RoundTip {
        let mut cache = unpoisoned(self.round_tip.lock());
        let (tip, _evicted) = lru(&mut cache, hardness.to_bits(), ROUND_TIPS_KEPT, || {
            let depth = round_depth(hardness, ROUND_RES);
            // The round tip is rotation-invariant, so a single orientation layer
            // suffices — the shader's wrapping lookup reads it for every
            // orientation (§6.6).
            let prefix = build_prefix(
                &self.ctx,
                ROUND_RES,
                ROUND_RES,
                1,
                &depth,
                "stark round tip prefix-tau",
            );
            let depth_view = build_depth_r16(&self.ctx, ROUND_RES, ROUND_RES, &depth);
            // The coverage prefix a liquify follow reads (§6.13) is the one place the
            // tip is wanted *as* a coverage: a follow is a fraction of how long the tip
            // covered a texel, which saturates exactly as a coverage does.
            let cov: Vec<f32> = depth.iter().map(|&k| 1.0 - (-k).exp()).collect();
            let warp = build_prefix(
                &self.ctx,
                ROUND_RES,
                ROUND_RES,
                1,
                &cov,
                COVERAGE_PREFIX_LABEL,
            );
            RoundTip {
                prefix,
                depth: depth_view,
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

/// The textures selected together for one brush: the prefix-τ volume the swept
/// deposit differences and the tip's own depth the wet loop's tool side reads — one
/// field, integrated and not — and, for a liquify brush alone, the coverage prefix its
/// follow reads (§6.13), with the tip's **rise** beside it.
pub(super) struct ResolvedTip {
    pub(super) prefix: wgpu::TextureView,
    /// The tip's `κ` per unit travel (`assets::build_depth_r16`): what a reservoir
    /// texel's exposure is (`dynamics.wesl::depth_at`), and the field `prefix` is the
    /// travel integral of. Bound by the wet loop alone — the swept path differences
    /// the prefix instead.
    pub(super) depth: wgpu::TextureView,
    /// The **coverage prefix**: the same volume as
    /// `prefix`, integrated linearly over the coverage, which the liquify `warp` kernel reads at the
    /// prefix-τ's own binding so a follow is the mask's mean over the pass rather
    /// than its optical depth. `None` on every other effect, which never asks.
    pub(super) warp: Option<wgpu::TextureView>,
    /// The tip's rise (§6.13), radii: the shortest travel over which its coverage
    /// climbs by `budget::WARP_CONTRACTION`. What the liquify step budget prices
    /// (`budget::liquify_len`); measured off the mask, whichever kind it is.
    pub(super) rise: f32,
}

/// A baked round tip: the **prefix-τ** volume both render paths integrate the swept
/// deposit against, the **depth** the stamp loop's reservoir texels weigh their exposure
/// by, the **coverage prefix** the liquify follow reads (§6.13), and the tip's rise. One
/// type because they are one thing — the same depth field, read four ways.
#[derive(Clone)]
struct RoundTip {
    prefix: wgpu::TextureView,
    depth: wgpu::TextureView,
    warp: wgpu::TextureView,
    rise: f32,
}

/// What a noise tile is cached by: the brush's kind and the stroke's seed.
type NoiseKey = (NoiseKind, u32);

/// The depth one full pass lays over the centreline, at every hardness.
///
/// The depth a full pass of the **softest** tip lays over its own centreline; the
/// hardest lays [`TAU_PER_PASS`](super::budget::TAU_PER_PASS), and the dial runs
/// geometrically between them ([`RoundProfile::of`]).
///
/// This is the one place the tip's *airiness* is set, and it has to move with the dial
/// because coverage saturates above `τ ≈ 3`: hold the peak at one pass and every
/// hardness reads as a solid bar with a blurred edge, since the only part of the profile
/// left in the visible band is its rim. Two is where a soft pass is a gradient all the
/// way across — 0.865 at the centreline, half of that by four fifths of the radius —
/// while the hard end still covers in one pass, at 0.999.
///
/// So a softer tip does lay a lighter mark, which is what a soft brush *is*: less of it
/// is in contact. It is not a second opacity dial — the swing over the whole range is
/// 0.865 to 0.999 — and a lighter mark than the soft end gives is what **flow** is for,
/// which composes the right way round: flow scales the depth, so a fainter pass has more
/// of its profile in the visible band and fades over a *wider* one.
const SOFT_PEAK_DEPTH: f32 = 2.0;

/// How fast the peak climbs off [`SOFT_PEAK_DEPTH`] as the dial hardens.
///
/// Anchored rather than chosen: at 2.5 a tip at hardness 0.8 lays 6.78, which is what
/// the family this replaced laid there (6.73). Everything from the middle of the dial
/// up therefore covers as it always did, and the whole of the lightening the bounded
/// peak buys is spent in the soft half, where it is the point.
const SOFT_FALLOFF: f32 = 2.5;

/// The 10–90 width of the softest tip's profile, in radii — the widest falloff the
/// dial offers, and the constant `budget::shoulder_per_radius` reports scaled by
/// `1 − hardness`.
///
/// Just under the widest this family can draw at the soft end's own peak (0.524, at
/// `p ≈ 2`): the width rises with `p`, turns over there and falls again, so asking for
/// the maximum itself would put [`RoundProfile::of`]'s scan on a stationary point with
/// two branches meeting on it.
pub(super) const SOFT_SHOULDER: f32 = 0.50;

/// One round tip's profile: what `hardness` resolves to before any radius is involved.
///
/// **The stroke is specified, and the tip is whatever draws it** (§6.6) — that much is
/// unchanged. What changed is the family. A full pass lays
///
/// ```text
/// P(y) = 1 − exp(−T·(1 − y²)^p)
/// ```
///
/// at `y` radii off the centreline, and the field that sweeps to it is the Abel
/// inverse, which for this family is elementary:
///
/// ```text
/// κ(r) = κ₀·(1 − r²)^(p − ½),    κ₀ = T / (2∫₀^{π/2} cos^{2p}θ dθ)
/// ```
///
/// since the transform of `(1 − r²)^m` is `B(½, m+1)·(1 − y²)^{m+½}`.
///
/// **Why this family and not `1 − |y|^h`.** The old profile reached exactly 1 at the
/// centreline, and a coverage of exactly 1 is an *infinite* optical depth. Its `κ` duly
/// diverged — ~9000 at the centre texel of a hard bake — and every ceiling that kept
/// that finite showed up in the picture as a flat disc across the middle of the stroke
/// with the falloff resuming at its rim in a visible crease, worst at low flow, where
/// the strength scales the exponent and the core is not saturated enough to hide it.
/// There is no ceiling that avoids it: the profile wanted a depth the engine cannot
/// carry, and the engine's every stroke budget is denominated in
/// [`TAU_PER_PASS`](super::budget::TAU_PER_PASS) — one pass ≈ 6.9 — so a tip that
/// delivered eighty of those per pass banded at the segment cadence and stopped
/// answering to `flow`.
///
/// This family has a **bounded** peak by construction, `T ≤ TAU_PER_PASS` at every
/// hardness, so the scale the budgets are written in is a property of the tip rather
/// than an accident of a clamp. The profile is a smooth dome — zero derivative at the
/// centreline, no cusp, no flat spot — and *flow scales `T` without touching the
/// shape*, so the same is true at every flow, which is what the old family could not
/// manage at any ceiling.
#[derive(Copy, Clone)]
pub(super) struct RoundProfile {
    /// `p`: the falloff exponent, solved so the shoulder is the one `hardness` names.
    p: f32,
    /// `κ₀`: the field's value at the centre, normalizing it to `peak`.
    k0: f32,
}

impl RoundProfile {
    /// The profile `hardness` names.
    ///
    /// **A flat top is the work of a small exponent, not of a deep tip**, which is what
    /// keeps this dial from turning into a second opacity: leaning on the depth for it
    /// would have
    /// hardness quietly setting how dark a stroke lands — see [`SOFT_PEAK_DEPTH`] for
    /// why it moves anyway, and by how little.
    ///
    /// One condition fixes `p`: the **shoulder**, which is the dial's whole meaning
    /// — `SOFT_SHOULDER·(1 − hardness)` of 10–90 width, swept evenly so the whole dial
    /// does work. Solved rather than inverted, because no closed form does it.
    ///
    /// A softer look than the soft end offers is what **flow** is for, and it composes
    /// the right way round: flow scales the depth, so a fainter pass has more of its
    /// profile inside the visible band and fades over a *wider* band, not a narrower
    /// one.
    pub(super) fn of(hardness: f32) -> Self {
        let hardness = hardness.clamp(0.0, 1.0);
        // Geometric between the two depths, but on `1 − (1 − hardness)^SOFT_FALLOFF`
        // rather than on the dial itself: the airiness belongs to the *soft* end, and
        // spreading it evenly left a mid-hard tip a few percent short of covering,
        // which reads as opaque paint that is not quite opaque and is not what any
        // dial position should mean.
        let soft = 1.0 - hardness;
        let peak = SOFT_PEAK_DEPTH
            * (super::budget::TAU_PER_PASS / SOFT_PEAK_DEPTH).powf(1.0 - soft.powf(SOFT_FALLOFF));
        let want = SOFT_SHOULDER * (1.0 - hardness);
        // Scanned rather than bisected, in two passes. The shoulder rises with `p`,
        // turns over, and falls again: below the turn the profile opens from a disc into
        // a dome, above it the dome bunches around the centreline and stops reaching the
        // tip's own rim. The dial wants the first of those, so the turn is found and the
        // target met at or under it — a bisection would land on whichever endpoint it
        // started nearest, and the turn moves with the peak, so it cannot be a constant.
        // Geometric steps, so the resolution is even in the ratio rather than crowded at
        // the hard end.
        let steps = ((P_MAX / P_MIN).ln() / SCAN_RATIO.ln()) as u32;
        let at = |i: u32| P_MIN * SCAN_RATIO.powi(i as i32);
        let mut turn = 0;
        for i in 0..=steps {
            if shoulder_of(peak, at(i)) >= shoulder_of(peak, at(turn)) {
                turn = i;
            }
        }
        let mut best = (f32::INFINITY, P_MIN);
        for i in 0..=turn {
            let p = at(i);
            let d = (shoulder_of(peak, p) - want).abs();
            if d <= best.0 {
                best = (d, p);
            }
        }
        let p = best.1;
        // `peak` is not kept: it is `k0·2∫cos^{2p}` and nothing at render time asks.
        Self {
            p,
            k0: peak / (2.0 * cos_power_integral(p)),
        }
    }

    /// The tip's optical depth at `r` radii from its centre, per radius of travel.
    ///
    /// `1 − r²` is held at one texel of the bake, which is the finest thing the field
    /// is ever read at: below `p = ½` the exponent is negative and the continuum field
    /// runs away at the rim, where what a texel actually carries is its own average.
    /// That is what bounds `κ` — to about 22 at the hardest, against ~9000 before, and
    /// under [`TAU_PER_PASS`](super::budget::TAU_PER_PASS) itself for every hardness
    /// below 0.8.
    pub(super) fn depth_at(&self, r: f32) -> f32 {
        if r >= 1.0 {
            return 0.0;
        }
        let rim = 2.0 / ROUND_RES as f32;
        self.k0 * (1.0 - r * r).max(rim).powf(self.p - 0.5)
    }

    /// The tip's coverage at `r`: `1 − exp(−κ(r))`. What the mask and the rise read —
    /// never the prefix, which integrates [`depth_at`](Self::depth_at) itself.
    fn coverage_at(&self, r: f32) -> f32 {
        1.0 - (-self.depth_at(r)).exp()
    }

    /// The profile this tip sweeps to at `y` radii off the centreline, for a pass at
    /// full strength. The claim the bake exists to make (`tests`, below).
    #[cfg(test)]
    pub(super) fn profile_at(&self, y: f32) -> f32 {
        let peak = self.k0 * 2.0 * cos_power_integral(self.p);
        1.0 - (-peak * (1.0 - y * y).max(0.0).powf(self.p)).exp()
    }
}

/// The exponents [`RoundProfile::of`] searches between. `P_MAX` is the shoulder's own
/// maximum, so the scan stays on the **broad** branch — past it the width falls again,
/// and the same shoulder is met a second time by a profile bunched around the
/// centreline that never reaches the tip's own rim. `P_MIN` is where the rim clamp has
/// taken over and a smaller exponent changes nothing.
const P_MIN: f32 = 0.03;
const P_MAX: f32 = 2.6;
/// The step [`RoundProfile::of`]'s scan takes between them.
const SCAN_RATIO: f32 = 1.01;

/// `∫₀^{π/2} cos^{2p}θ dθ` — half of `B(½, p+½)`, the constant that normalizes
/// [`RoundProfile`]'s field to its peak.
///
/// In `θ` rather than `r` on purpose: `∫₀¹(1 − r²)^{p−½} dr` is the same number with a
/// singular integrand for `p < ½`, and `r = sin θ` takes the singularity out, leaving
/// a bounded smooth periodic integrand the midpoint rule converges on geometrically.
/// A gamma function would answer it in closed form and is not worth the dependency for
/// one constant per bake.
fn cos_power_integral(p: f32) -> f32 {
    const N: usize = 4096;
    let step = std::f32::consts::FRAC_PI_2 / N as f32;
    (0..N)
        .map(|i| (((i as f32 + 0.5) * step).cos()).powf(2.0 * p))
        .sum::<f32>()
        * step
}

/// The 10–90 width of `1 − exp(−peak·(1 − y²)^p)`, in radii — measured against the
/// profile's **own** peak, since a soft tip's peak is under 0.9 and a fixed 90% would
/// be unreachable. Closed form, so [`RoundProfile::of`] can scan it.
fn shoulder_of(peak: f32, p: f32) -> f32 {
    let top = 1.0 - (-peak).exp();
    let y_at = |f: f32| -> f32 {
        let t = -(1.0 - f * top).ln();
        if t >= peak {
            0.0
        } else {
            (1.0 - (t / peak).powf(1.0 / p)).max(0.0).sqrt()
        }
    };
    y_at(0.1) - y_at(0.9)
}

/// How finely [`round_depth`] samples each texel of the bake, per axis.
///
/// A texel of a prefix volume stands for the field's **mean** over its own square, and
/// below `p = ½` the field rises into the rim steeply enough that its centre is a poor
/// reading of that: at the hard end a midpoint bake misses the swept profile by 6e-3 of
/// coverage well inside the disc, where four-by-four holds it under a thousandth
/// everywhere the picture can show. Paid once per hardness, behind the cache.
const DEPTH_SUBSAMPLES: u32 = 4;

/// Generate the round tip's **optical depth** field on the bake grid (§6.6): the
/// radial `κ` of the [`RoundProfile`] `hardness` names, averaged over each texel.
///
/// Finite everywhere, which `build_prefix` requires and a prefix sum gives no second
/// chance at — `RoundProfile::depth_at` holds the rim, and there is no centre
/// divergence in this family to hold.
fn round_depth(hardness: f32, res: u32) -> Vec<f32> {
    let profile = RoundProfile::of(hardness);
    let n = DEPTH_SUBSAMPLES;
    let inv = 1.0 / (n * n) as f32;
    let mut depth = vec![0.0f32; (res * res) as usize];
    for y in 0..res {
        for x in 0..res {
            let mut acc = 0.0f32;
            for sy in 0..n {
                let fy = (y as f32 + (sy as f32 + 0.5) / n as f32) / res as f32 * 2.0 - 1.0;
                for sx in 0..n {
                    let fx = (x as f32 + (sx as f32 + 0.5) / n as f32) / res as f32 * 2.0 - 1.0;
                    acc += profile.depth_at((fx * fx + fy * fy).sqrt());
                }
            }
            depth[(y * res + x) as usize] = acc * inv;
        }
    }
    depth
}

/// How finely [`round_rise`] samples the radial profile: the rise is answered to a
/// half-thousandth of a radius, which under the budget's texel floor is finer than
/// any tip narrower than two thousand px can show.
const RISE_SAMPLES: usize = 2048;

/// The round tip's **rise** (§6.13), in radii: the shortest travel over which its
/// coverage climbs by [`WARP_CONTRACTION`](super::budget::WARP_CONTRACTION),
/// anywhere along a pass — what the liquify step budget prices.
///
/// Along the centreline, where the pass crosses the profile head-on; any other chord
/// crosses it obliquely and climbs slower, so this is the worst case. The profile is
/// monotone in the radius, so the climb over a window is monotone in the window's width
/// and the shortest window that climbs enough is a binary search over widths.
///
/// Sampled rather than solved because the steepest window is not always at the rim: a
/// hard tip climbs fastest where its `√(1 − r)` cusp meets the edge, a soft one
/// somewhere inside. The rim itself is the last sample — a hard tip's whole climb sits
/// inside the final ten-thousandth of a radius.
pub(super) fn round_rise(hardness: f32) -> f32 {
    let n = RISE_SAMPLES;
    let profile = RoundProfile::of(hardness);
    let cov: Vec<f32> = (0..=n)
        .map(|i| profile.coverage_at(i as f32 / n as f32))
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

    /// The rise narrows as the tip hardens (§6.13) and lands where the profile puts it:
    /// at hardness 0.8 the coverage climbs its ½ over a twelfth of a radius; at 1 it is
    /// the cusp's last thousandth or so, which the budget's texel floor then takes over.
    /// Monotone only from the middle of the range — the softest tips climb steepest at
    /// their *centre*, where `κ = h/(2r)` spikes, and that spike is narrower at hardness
    /// 0 than a slightly harder tip's mid-profile climb, so the two softest rises are
    /// merely finite.
    #[test]
    fn the_round_tips_rise_narrows_with_hardness() {
        // Monotone across the *whole* dial now, where the old family's two softest
        // tips merely had finite rises: that one climbed steepest at its centre,
        // where `κ = h/(2r)` spiked, and the spike was narrower at hardness 0 than a
        // slightly harder tip's mid-profile climb. This family has no centre spike to
        // be non-monotone about — its steepest climb is always at the rim.
        let mut last = f32::INFINITY;
        for hardness in [0.0f32, 0.25, 0.5] {
            let rise = round_rise(hardness);
            assert!(
                rise > 0.0 && rise < last,
                "hardness {hardness}: rise {rise} after {last}"
            );
            last = rise;
        }
        let at = round_rise(0.0);
        assert!(
            (0.15..=0.28).contains(&at),
            "the softest tip rises over {at} radii"
        );
        // Past the middle of the dial the whole climb is inside the sampling floor,
        // and the budget's texel floor takes over from there (§6.13).
        for hardness in [0.8f32, 0.95, 1.0] {
            let rise = round_rise(hardness);
            assert!(
                rise <= 2.0 / RISE_SAMPLES as f32,
                "hardness {hardness} rises over {rise} radii, not inside the sampling",
            );
        }
        // The rise is a rise: the sampled profile really climbs that much over it.
        let n = RISE_SAMPLES;
        let width = (at * n as f32) as usize;
        let soft = RoundProfile::of(0.0);
        let climbed = (0..=n - width)
            .map(|i| {
                soft.coverage_at(i as f32 / n as f32)
                    - soft.coverage_at((i + width) as f32 / n as f32)
            })
            .fold(0.0f32, f32::max);
        assert!(
            climbed >= super::super::budget::WARP_CONTRACTION,
            "over its rise the profile climbs {climbed}, less than the bound",
        );
    }

    /// One row of the baked field, swept: the sum `assets::prefix_data` accumulates
    /// across a row, in brush-local width. What the shader reads as a full pass's τ.
    fn swept_tau(depth: &[f32], res: u32, row: u32) -> f32 {
        (0..res)
            .map(|x| depth[(row * res + x) as usize])
            .sum::<f32>()
            * (2.0 / res as f32)
    }

    /// The whole claim [`RoundProfile`] makes: a pass of the tip at strength `a` lays
    /// `1 − exp(−a·T·(1 − y²)^p)` across the stroke — **at every `y`, the core
    /// included**, and at every strength.
    ///
    /// Swept through the very integral the GPU volume is built from (`swept_tau`), so
    /// the bake and the profile it claims cannot drift. The strengths matter as much as
    /// the hardnesses: `a` scales the depth without touching the shape, so a low flow is
    /// where anything the bake has flattened stops being hidden under a coverage that
    /// had saturated anyway.
    ///
    /// **The regression this pins** (2026-09-07): the tip was `1 − |y|^h`, whose depth
    /// diverges at the centreline because the profile reaches exactly 1 there. Every
    /// ceiling that kept that finite — `assets::tau_of`'s mask clamp at τ ≈ 6.9 — flattened
    /// a disc across the middle of the stroke and left the falloff resuming at its rim
    /// in a visible crease, 0.07 of the radius across at hardness 0 and 0.71 at 0.95,
    /// worst at low flow. This family has a bounded peak by construction, so there is
    /// no ceiling to flatten anything and the claim holds to a thousandth.
    #[test]
    fn the_round_tip_sweeps_to_the_profile_its_hardness_names() {
        const RES: u32 = ROUND_RES;
        for hardness in [0.0, 0.25, 0.5, 0.8, 0.95] {
            let profile = RoundProfile::of(hardness);
            let depth = round_depth(hardness, RES);
            for a in [1.0f32, 0.5, 0.25] {
                for row in RES / 2..RES {
                    let y = (row as f32 + 0.5) / RES as f32 * 2.0 - 1.0;
                    let laid = 1.0 - (-a * swept_tau(&depth, RES, row)).exp();
                    // The profile at strength `a` is the same shape at depth `a·T`.
                    let want = 1.0 - (1.0 - profile.profile_at(y)).powf(a);
                    // Two claims, not one loose one. Over the **body** of the profile
                    // the bake is a reading of the closed form and is held to a
                    // few thousandths. The **outer tenth** is where the disc's own edge and
                    // the rim clamp meet a 256² grid: below `p = ½` the field rises
                    // into the rim, no finite sampling resolves that as well, and what
                    // is left is a few levels across the antialiased edge the sweep's
                    // box filter owns anyway (§6.2). Both are far under the 0.05–0.22
                    // of coverage the ceiling this family replaced cost, and that one
                    // was in the middle of the stroke.
                    // The outermost texel row is the one the rim clamp holds
                    // (`RoundProfile::depth_at`), so it is the tip's antialiased edge
                    // rather than a reading of the profile — the sweep's own box filter
                    // owns it (§6.2), and comparing it to the continuum is comparing
                    // two different claims.
                    if row == RES - 1 {
                        continue;
                    }
                    let bound = if y.abs() <= 0.9 { 4e-3 } else { 3e-2 };
                    assert!(
                        (laid - want).abs() < bound,
                        "hardness {hardness} at strength {a}: at y = {y:.4} the sweep                          lays {laid:.5}, not the {want:.5} its profile names",
                    );
                }
            }
        }
    }

    /// The property the family exists for, and the one the old one could not hold: the
    /// depth a full pass lays is **bounded by one pass**, at every hardness.
    ///
    /// That is the scale every stroke budget in `budget.rs` is denominated in
    /// (`TAU_PER_PASS`) — the exchange step the flattener buys, the coarse cell, the
    /// supersample gate. `1 − |y|^h` broke it by eighty-fold at the hard end, because a
    /// profile reaching exactly 1 at the centreline needs an unbounded depth to get
    /// there, and what that cost was not a wrong picture but a *banded* one: strokes
    /// printing at the segment cadence, a bleed stencil laying displaced copies of an
    /// edge, `flow` no longer scaling a smear.
    ///
    /// The profile's shape is not asserted here — `the_round_tip_sweeps_to_the_profile_its_hardness_names`
    /// pins it against the closed form, which is a stronger statement than monotonicity
    /// and one the bake's own quadrature does not blur.
    #[test]
    fn the_round_tip_never_lays_more_than_one_pass() {
        const RES: u32 = ROUND_RES;
        for hardness in [0.0f32, 0.25, 0.5, 0.8, 0.95, 1.0] {
            let depth = round_depth(hardness, RES);
            for row in RES / 2..RES {
                let tau = swept_tau(&depth, RES, row);
                // To within the bake's own quadrature: `PEAK_DEPTH` normalizes the
                // continuum field and the grid's reading of it lands a fraction of a
                // percent either side.
                assert!(
                    tau <= super::super::budget::TAU_PER_PASS * 1.01,
                    "hardness {hardness}: a pass lays {tau}, past the one pass every                      budget is priced in",
                );
            }
        }
        // And the peak really is at the centreline, which is what makes it the bound —
        // to within the same quadrature, since the profile is flat to second order
        // there (`T ≈ T(1 − p·y²)`) and the grid's reading of the disc's own edge moves
        // by more than that between one row and the next.
        for hardness in [0.0f32, 0.5, 0.95] {
            let depth = round_depth(hardness, RES);
            let core = swept_tau(&depth, RES, RES / 2);
            for row in RES / 2..RES {
                let tau = swept_tau(&depth, RES, row);
                assert!(
                    tau <= core + 1e-2,
                    "hardness {hardness}: row {row} lays {tau}, more than the                      centreline's {core}",
                );
            }
        }
    }
}
