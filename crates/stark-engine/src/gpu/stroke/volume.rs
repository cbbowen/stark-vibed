//! The **prefix volumes** a brush tip is read through, and the bound on the ones baked
//! lazily (§6.2, §6.6, §6.13).
//!
//! One optical-depth field per tip, read four ways: integrated along the travel axis
//! for the swept deposit, plain for the wet loop's tool side, integrated over the
//! coverage for a liquify follow, and differenced for the tip's rise. A stored mask's
//! volumes are built from the coverage its id names ([`tau_of`]), the round tip's from
//! its closed form (`tips::round_depth`) — so the arithmetic lives beside the stroke
//! that reads it rather than inside either of the two callers.
//!
//! The small LRU both tip caches are held in is here for the same reason: bounding the
//! mask volumes ([`VolumeCache`]) is what stopped it being the round tip's own.

use std::sync::{Arc, Mutex};

use stark_model::AssetId;
use stark_model::document::OrientationSource;

use crate::gpu::context::GpuContext;
use crate::gpu::desc;
use crate::unpoisoned;

use super::budget::WARP_CONTRACTION;

/// The textures selected together for one brush: the prefix-τ volume the swept
/// deposit differences and the tip's own depth the wet loop's tool side reads — one
/// field, integrated and not — and, for a liquify brush alone, the coverage prefix its
/// follow reads (§6.13), with the tip's **rise** beside it.
#[derive(Clone)]
pub(crate) struct ResolvedTip {
    pub(crate) prefix: wgpu::TextureView,
    /// The tip's `κ` per unit travel ([`build_depth_r16`]): what a reservoir texel's
    /// exposure is (`dynamics.wesl::depth_at`), and the field `prefix` is the travel
    /// integral of. Bound by the wet loop alone — the swept path differences the prefix
    /// instead.
    pub(crate) depth: wgpu::TextureView,
    /// The **coverage prefix**: the same volume as `prefix`, integrated linearly over
    /// the coverage, which the liquify `warp` kernel reads at the prefix-τ's own binding
    /// so a follow is the mask's mean over the pass rather than its optical depth.
    /// `None` on every other effect, which never asks.
    pub(crate) warp: Option<wgpu::TextureView>,
    /// The tip's rise (§6.13), radii: the shortest travel over which its coverage
    /// climbs by [`WARP_CONTRACTION`]. What the liquify step budget prices
    /// (`budget::liquify_len`); measured off the mask, whichever kind it is.
    pub(crate) rise: f32,
}

// ---- the small LRU both tip caches are held in ---------------------------------

/// A least-recently-used list held as a `Vec`, newest last, shared across the clones of
/// whatever holds it. A `Vec` rather than a map because the bounds here are small — a
/// linear scan beats a hash, and the order *is* the recency.
pub(super) type Lru<K, V> = Arc<Mutex<Vec<(K, V)>>>;

/// `key`'s entry in `lru` — the hit moved to the back, or `build`'s result pushed there
/// — and, while `over` says the list holds too much, the oldest entries handed back for
/// the caller to release. A GPU resource dropped is not one freed (`gpu::scratch`), so
/// which release it gets is the caller's to say.
///
/// **What "too much" is belongs to the caller**: a count for the tip and noise caches,
/// whose entries are all one size, and a byte budget for the mask volumes, which run
/// sixty-fold apart ([`VolumeCache`]). The newest is never evicted — one larger than
/// the whole budget would otherwise be baked and dropped again on every render.
///
/// **`build` runs with the lock released**, because a bake is milliseconds — 256² × 16
/// samples of the round tip's profile — and every clone of the renderer shares this
/// list: a hardness slider dragged on one engine would otherwise stall a sibling's
/// stroke for the length of each frame's bake. Two callers may then bake one key at
/// once; the first insert is what both read and what the cache keeps, and the loser's
/// value comes back in the release slot beside any eviction. One wasted bake against a
/// held lock, and the cache never holds two answers to one key.
pub(super) fn cached<K: PartialEq, V: Clone>(
    lru: &Lru<K, V>,
    key: K,
    over: impl Fn(&[(K, V)]) -> bool,
    build: impl FnOnce() -> V,
) -> (V, Vec<V>) {
    if let Some(v) = hit(&mut unpoisoned(lru.lock()), &key) {
        return (v, Vec::new());
    }
    let built = build();
    let mut cache = unpoisoned(lru.lock());
    if let Some(v) = hit(&mut cache, &key) {
        return (v, vec![built]);
    }
    cache.push((key, built.clone()));
    let mut evicted = Vec::new();
    while cache.len() > 1 && over(&cache) {
        evicted.push(cache.remove(0).1);
    }
    (built, evicted)
}

/// `key`'s value if the list holds it, moved to the back as the newest.
fn hit<K: PartialEq, V: Clone>(cache: &mut Vec<(K, V)>, key: &K) -> Option<V> {
    let i = cache.iter().position(|(k, _)| k == key)?;
    let entry = cache.remove(i);
    let v = entry.1.clone();
    cache.push(entry);
    Some(v)
}

// ---- the mask volumes baked on first ask ---------------------------------------

/// Which of a mask's volumes is baked lazily rather than at import: the pen-oriented
/// prefix-τ (§6.6), and the coverage prefix either orientation's liquify follow reads
/// (§6.13).
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) enum Lazy {
    Pen,
    Warp(OrientationSource),
}

/// What one lazily baked volume is kept under: the mask it was baked from, and which of
/// that mask's volumes it is.
type Key = (AssetId, Lazy);

/// One baked volume and what it cost, so the budget below can be read off the list.
#[derive(Clone)]
struct Volume {
    view: wgpu::TextureView,
    bytes: u64,
}

/// How many bytes of lazily baked mask volume [`VolumeCache`] keeps at once.
///
/// Twice the largest live set one brush can ask for: a pen-oriented liquify brush reads
/// a pen prefix-τ and a coverage prefix of the same extent, each held to
/// [`PREFIX_BUDGET_BYTES`] by its layer count. Under that, a single stroke would evict
/// the volume it had just been handed and re-bake it on the next pointer move; at twice
/// it, two brushes alternating — two peers painting concurrently, or a replay
/// interleaving strokes (§12) — keep both.
const VOLUME_BUDGET_BYTES: u64 = 4 * PREFIX_BUDGET_BYTES;

/// The mask volumes baked on first ask rather than at import, bounded by the bytes they
/// hold.
///
/// Every one of them rebuilds from the coverage its id names, so an eviction costs a
/// bake and never a picture — which is what makes a bound safe to have at all. Without
/// one, a library of stamps banks a pen volume each and gives none of them back: the
/// largest shape there is costs [`PREFIX_BUDGET_BYTES`] for its pen prefix-τ and as
/// much again for its liquify twin.
///
/// Cheap to clone with the store — the list is `Arc`-backed.
#[derive(Clone)]
pub(crate) struct VolumeCache {
    kept: Lru<Key, Volume>,
    budget: u64,
}

impl Default for VolumeCache {
    fn default() -> Self {
        Self {
            kept: Arc::new(Mutex::new(Vec::new())),
            budget: VOLUME_BUDGET_BYTES,
        }
    }
}

impl VolumeCache {
    /// `key`'s volume, `bake`d if the cache does not already hold it, with the oldest
    /// entries dropped until the total is back inside the budget.
    ///
    /// `bytes` is what the bake will cost, which [`prefix_bytes`] answers from the
    /// volume's extent before it exists — the only way to price an entry that is not
    /// yet built.
    pub(crate) fn baked(
        &self,
        key: Key,
        bytes: u64,
        bake: impl FnOnce() -> wgpu::TextureView,
    ) -> wgpu::TextureView {
        let budget = self.budget;
        let (volume, evicted) = cached(
            &self.kept,
            key,
            |kept| kept.iter().map(|(_, v)| v.bytes).sum::<u64>() > budget,
            || Volume {
                view: bake(),
                bytes,
            },
        );
        // An eviction is dropped rather than `destroy()`ed: a view keeps its own
        // texture alive, so a stroke already holding one is unaffected, and these go at
        // the rate the brush changes rather than per pointer move.
        drop(evicted);
        volume.view
    }
}

// ---- the volumes themselves ----------------------------------------------------

/// Largest number of orientation slices a brush's pen-oriented prefix-τ volume holds
/// (§6.6). With linear interpolation between adjacent layers this is ~5.6° resolution
/// — smooth for any practical pen rotation.
const MAX_ORIENTATION_LAYERS: u32 = 64;

/// Memory budget (bytes) for one brush's pen-oriented prefix-τ volume: the layer count
/// is chosen so the whole volume stays under it, so a large detailed stamp keeps its
/// full resolution and trades orientation granularity instead. Only the pen volume is
/// measured against it — the follow-stroke bake is a single layer and has nothing to
/// trade.
const PREFIX_BUDGET_BYTES: u64 = 64 << 20; // 64 MiB

/// What a prefix volume is stored as: an `Rg32Float` **2D-array** texture, the array
/// axis orientation ([`build_prefix`]).
const PREFIX_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Float;

/// The **prefix-τ**'s texture label ([`build_prefix`]): [`tau_of`] integrated along the
/// travel axis, which is what a segment's swept deposit is a difference of.
pub(crate) const PREFIX_TAU_LABEL: &str = "stark brush prefix-tau";

/// The **coverage prefix**'s texture label ([`build_prefix`]): the coverage integrated
/// as it stands, so a difference across a stretch of travel is the mask's mean over it
/// times the travel — what a liquify follow is a fraction of (§6.13). Linear where τ is
/// not, so a texel's follow is how *long* the tip covered it and a full pass over the
/// tip's core is exactly the travel.
pub(crate) const COVERAGE_PREFIX_LABEL: &str = "stark brush coverage prefix";

/// How many bytes a `width × height × layers` prefix volume uploads — what
/// [`VolumeCache`] prices an entry at, off [`PREFIX_FORMAT`]'s own texel size so it
/// cannot drift from what [`build_prefix`] creates.
pub(crate) fn prefix_bytes(width: u32, height: u32, layers: u32) -> u64 {
    let texel = u64::from(
        PREFIX_FORMAT
            .block_copy_size(None)
            .expect("an uncompressed format has one aspect"),
    );
    u64::from(width) * u64::from(height) * u64::from(layers) * texel
}

/// How many orientation slices to build for a `width × height` volume: as many as
/// [`PREFIX_BUDGET_BYTES`] allows, capped at [`MAX_ORIENTATION_LAYERS`] and at least 1.
pub(crate) fn orientation_layers(width: u32, height: u32) -> u32 {
    let per_layer = prefix_bytes(width, height, 1).max(1);
    (PREFIX_BUDGET_BYTES / per_layer).clamp(1, u64::from(MAX_ORIENTATION_LAYERS)) as u32
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
/// arithmetic at θ = 0. `FollowStroke`, the only caller that would read it as the
/// shape's native orientation, has its own single-layer bake instead.
pub(crate) fn rotate_layers(coverage: &[f32], width: u32, height: u32, layers: u32) -> Vec<f32> {
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

/// A mask's **rise** (§6.13), in radii: the shortest travel over which its coverage
/// can climb by [`WARP_CONTRACTION`] — what the liquify step budget prices a stamp at.
///
/// Bounded from the steepest texel-to-texel step along either axis rather than
/// searched for as the round tip's is (`tips::round_rise`): a stamp is any picture
/// and its rows are not monotone. A hard mask — a whole-texel step — gives a rise of
/// half a texel, which the budget's own floor rounds up to the canvas texel; a mask
/// that never climbs has no rise, and prices as none.
pub(crate) fn mask_rise(coverage: &[u8], width: u32, height: u32) -> f32 {
    let (w, h) = (width as usize, height as usize);
    let at = |x: usize, y: usize| coverage[y * w + x] as f32 / 255.0;
    // Steepest climb per radius on each axis: a texel is `2/width` radii across
    // and `2/height` tall.
    let mut slope = 0.0f32;
    for y in 0..h {
        for x in 0..w {
            if x + 1 < w {
                slope = slope.max((at(x + 1, y) - at(x, y)).abs() * width as f32 / 2.0);
            }
            if y + 1 < h {
                slope = slope.max((at(x, y + 1) - at(x, y)).abs() * height as f32 / 2.0);
            }
        }
    }
    if slope <= 0.0 {
        f32::INFINITY
    } else {
        WARP_CONTRACTION / slope
    }
}

/// A stored mask's 8-bit coverage as every bake over it reads it: one float per texel,
/// `0..=1`. The one place the two spellings of a coverage meet.
pub(crate) fn coverage_f32(coverage: &[u8]) -> Vec<f32> {
    coverage.iter().map(|&b| f32::from(b) / 255.0).collect()
}

/// A **stored mask's** optical depth, `κ = −ln(1 − coverage)` — the currency the
/// deposit sums (§6.1), and the one conversion between the two.
///
/// The logarithm is [`detmath`](crate::detmath)'s rather than the host's (§12.1). This
/// is what a stamp brush's whole prefix volume is built out of, and the volume is
/// `Rg32Float` — nothing rounds an ulp away between here and the deposit, so two peers
/// whose libms disagreed on a texel would lay different paint from the same log.
///
/// The clamp keeps `κ` finite where a mask reaches 1: full coverage is infinite
/// depth, and would carry `+∞` into every prefix sum downstream of it.
/// `dynamics.wesl`'s own `tau_of` mirrors this clamp, so the tool side — which has
/// no prefix to difference — agrees with the volume built here.
///
/// **For a mask, and only a mask.** An 8-bit coverage really does saturate, so the
/// depth above the clamp is information the mask never carried and the ceiling costs
/// nothing. A tip that knows its own `κ` in closed form must hand *that* to
/// [`build_prefix`] and [`build_depth_r16`] instead of laundering it through a
/// coverage: the round tip's diverges at the centre by construction
/// (`tips::round_depth`), and round-tripping it through here capped its core at the
/// ceiling this clamp sets, τ ≈ 6.9. What that cost was a flat disc across the middle
/// of every stroke — 0.07 of the radius at hardness 0, 0.71 at 0.95 — with the
/// profile's own falloff resuming at its rim as a visible crease, worst at low flow,
/// where the strength scales the exponent and the core is not saturated enough to hide
/// it.
pub(crate) fn tau_of(coverage: f32) -> f32 {
    -crate::detmath::ln(1.0 - coverage.clamp(0.0, 0.999))
}

/// Build a brush's **prefix** volume (§6.2, §6.6): for each orientation `layer` and
/// each row, the running integral of `field` along the travel axis (x), in
/// brush-local units (x spans `[-1, 1]`, width 2). Stored in [`PREFIX_FORMAT`] as a
/// **2D-array** texture — the array axis is orientation, sampled with wrapping — and
/// read via `textureLoad` + manual trilinear by the sweep shader, so a segment's
/// swept depth at a point is `prefix(u) − prefix(u−d)` on its layer. A 2D array
/// rather than a true 3D texture so the mask keeps its full width/height: 3D textures
/// are capped far smaller, e.g. 256px, by `maxTextureDimension3D`.
///
/// `g` carries the **lateral prefix of `r`** — `∫₋₁^y prefix(x, ·)`, brush units on
/// both axes — so the sweep can read the deposit's exact box average over the pixel's
/// own footprint as one more difference (`stamp_common::prefix_span_box`, §6.2).
/// Baked at the midpoint rule: a row's value is the integral *to* its centre, which
/// is what makes the shader's bilinear tap exact there rather than half a texel off.
///
/// **The caller says what it is integrating**, and hands the samples themselves:
/// [`tau_of`] over a stored mask for a prefix-τ, the coverage as it stands for the
/// coverage prefix a liquify follow reads (§6.13), the closed-form depth for a tip
/// that has one (`tips::round_depth`). An enum picking the conversion in here would
/// have to name the field it was handed as well, and it is that pairing — a τ volume
/// reconstructed from a coverage the tip had already been flattened into — that put a
/// clamped plateau in the round tip's core ([`tau_of`]).
///
/// Shared by the asset store (image brushes — one identity layer for follow-stroke, a
/// rotated stack for pen) and the stroke renderer (the round tip, regenerated per
/// `hardness` — rotation-invariant, 1 layer). `field` is `layers × height × width`
/// row-major, and must be finite: a prefix sum carries one `+∞` to the end of its
/// row. Every volume is baked on the mask's own grid, so one column's brush-local
/// width is `2/width` throughout.
///
/// Returns the view alone: it holds its own reference to the texture, so there is
/// nothing for a caller to keep beside it.
pub(crate) fn build_prefix(
    ctx: &GpuContext,
    width: u32,
    height: u32,
    layers: u32,
    field: &[f32],
    label: &'static str,
) -> wgpu::TextureView {
    let data = prefix_data(width, height, layers, field);
    // `prefix_data` is layer-major, which is the upload's order for an array.
    let texture = desc::texture_with_data(
        ctx,
        &desc::sampled_texture(label, PREFIX_FORMAT, extent(width, height, layers)),
        bytemuck::cast_slice(&data),
    );
    // Always a 2D-array view (even for the single-layer round tip) so the shader binds one
    // texture type for every brush.
    texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    })
}

/// A prefix volume's texels ([`build_prefix`]): interleaved `(r, g)` pairs, `r`
/// the travel prefix of `field` per row, `g` the lateral (midpoint-rule)
/// prefix of `r` per column. Split from the upload so the arithmetic is testable
/// without a device.
fn prefix_data(width: u32, height: u32, layers: u32, field: &[f32]) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    let dx = 2.0 / width as f32;
    let dy = 2.0 / height as f32;
    let mut data = vec![0.0f32; field.len() * 2];
    for l in 0..layers as usize {
        let plane = l * w * h;
        for y in 0..h {
            let mut acc = 0.0f32;
            for x in 0..w {
                acc += field[plane + y * w + x] * dx;
                data[(plane + y * w + x) * 2] = acc;
            }
        }
        // The midpoint rule: a row's stored value is the integral to its *centre* —
        // every earlier row whole, half of its own — so the shader's bilinear read
        // is exact at row centres and piecewise linear between them.
        for x in 0..w {
            let mut acc = 0.0f32;
            for y in 0..h {
                let r = data[(plane + y * w + x) * 2];
                data[(plane + y * w + x) * 2 + 1] = acc + r * (0.5 * dy);
                acc += r * dy;
            }
        }
    }
    data
}

/// Upload a tip's **optical depth** `κ` as a filterable `R16Float` texture — what
/// the brush-dynamics loop's *tool* side reads for a reservoir texel's exposure
/// (`dynamics.wesl::depth_at`), rotating the sample coordinates for orientation, so
/// no pre-rotated layers are needed. Shared by the asset store (image brushes, whose
/// `κ` is [`tau_of`] of their 8-bit coverage) and the stroke renderer (the round tip's
/// closed form, per hardness).
///
/// **`κ` and not the coverage it reads off**, and in a float format for the same
/// reason [`build_prefix`] takes a field rather than a conversion: the two sides of
/// the exchange are two quadratures of one bilinear form and conserve only while they
/// agree texel for paired texel (§6.2). A round tip's `κ` reaches ~9000 at the centre
/// texel of its hardest bake, where a coverage saturates at [`tau_of`]'s ceiling, so a
/// tool side handed the coverage recovered a `τ` its canvas side had not capped, and
/// took from the canvas what the reservoir never gave up. `R16Float` carries that with
/// room to spare (max 65504, and ~5e-4 of relative step there) and is filterable, which
/// `R32Float` is not.
///
/// Returns the view alone, like [`build_prefix`].
pub(crate) fn build_depth_r16(
    ctx: &GpuContext,
    width: u32,
    height: u32,
    depth: &[f32],
) -> wgpu::TextureView {
    let halves: Vec<u16> = depth
        .iter()
        .map(|&k| crate::gpu::half::f32_to_f16(k))
        .collect();
    desc::texture_with_data(
        ctx,
        &desc::sampled_texture(
            "stark brush depth",
            wgpu::TextureFormat::R16Float,
            extent(width, height, 1),
        ),
        bytemuck::cast_slice(&halves),
    )
    .create_view(&wgpu::TextureViewDescriptor::default())
}

/// A `width × height` texture of `layers` array slices.
fn extent(width: u32, height: u32, layers: u32) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: layers,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Turning a canonical shape must not change how much of it there is.**
    ///
    /// The unpadded bake rests on this: a canonical mask's content lies inside the
    /// disc inscribed in its square (`stark_assetid::coverage`), a rotation maps that
    /// disc to itself, so no angle carries any of the mask off the edge of its own
    /// volume. The mask here reaches the disc's rim — the most a canonical mask can
    /// occupy — with structure on both axes so a loss would register.
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

    /// The `g` channel is the lateral integral of `r` — checked per layer with an
    /// asymmetric two-layer mask, so a plane- or stride-indexing slip cannot cancel.
    ///
    /// Against independent summation of `r`: every row's `g` is the rows before it
    /// plus half of itself, which is what makes a bilinear tap at a row centre the
    /// exact integral to that centre (`stamp_common::prefix2_edge` relies on it).
    #[test]
    fn the_lateral_prefix_integrates_the_travel_prefix() {
        let (w, h, layers) = (16u32, 12u32, 2u32);
        let plane = (w * h) as usize;
        let cov: Vec<f32> = (0..plane * layers as usize)
            .map(|i| {
                let l = i / plane;
                let x = (i % w as usize) as f32 / w as f32;
                let y = ((i % plane) / w as usize) as f32 / h as f32;
                (0.9 * x * (0.3 + 0.7 * y) * (1.0 - 0.4 * l as f32)).clamp(0.0, 0.95)
            })
            .collect();
        let tau: Vec<f32> = cov.iter().map(|&c| tau_of(c)).collect();
        let data = prefix_data(w, h, layers, &tau);
        let dy = 2.0 / h as f32;
        let r = |l: usize, x: usize, y: usize| data[(l * plane + y * w as usize + x) * 2];
        let g = |l: usize, x: usize, y: usize| data[(l * plane + y * w as usize + x) * 2 + 1];
        for l in 0..layers as usize {
            for x in 0..w as usize {
                let mut below = 0.0f32;
                for y in 0..h as usize {
                    let want = below + r(l, x, y) * 0.5 * dy;
                    assert!(
                        (g(l, x, y) - want).abs() < 1e-5,
                        "layer {l}, column {x}, row {y}: g = {}, not the midpoint \
                         integral {want}",
                        g(l, x, y),
                    );
                    below += r(l, x, y) * dy;
                }
            }
        }
    }

    /// A stamp's rise is bounded off its steepest texel step (§6.13): a hard disc
    /// climbs a whole texel at its edge, so its rise is half a texel — under the
    /// budget's floor, which is the point — a linear ramp across the mask climbs
    /// half of it in half the width, and a flat mask never climbs at all.
    #[test]
    fn a_stamps_rise_is_read_off_its_steepest_step() {
        let (w, h) = (64u32, 64u32);
        let disc: Vec<u8> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32 + 0.5 - 32.0, (i / w) as f32 + 0.5 - 32.0);
                if x * x + y * y < 28.0 * 28.0 { 255 } else { 0 }
            })
            .collect();
        let texel = 2.0 / w as f32;
        assert!((mask_rise(&disc, w, h) - 0.5 * texel).abs() < 1e-6);
        // One level per texel, so the u8 quantization is exactly the ramp: 255
        // levels over 255 texels of a 256-wide mask, two radii less a texel.
        let (rw, rh) = (256u32, 256u32);
        let ramp: Vec<u8> = (0..rw * rh).map(|i| (i % rw) as u8).collect();
        let rise = mask_rise(&ramp, rw, rh);
        assert!(
            (rise - 1.0).abs() < 0.01,
            "a ramp over two radii rises over one: {rise}"
        );
        assert_eq!(
            mask_rise(&vec![200u8; (w * h) as usize], w, h),
            f32::INFINITY
        );
    }

    /// **What an entry is priced at is what its upload costs**, so the budget bounds
    /// the GPU memory it claims to and not some other number — and the layer count
    /// really does hold a pen volume to [`PREFIX_BUDGET_BYTES`].
    ///
    /// Against [`desc::upload_len`] of the very descriptor [`build_prefix`] creates,
    /// which is the one thing that says how many bytes a texture is made from.
    #[test]
    fn a_volume_is_priced_at_what_its_upload_costs() {
        for (w, h, layers) in [(1u32, 1u32, 1u32), (256, 256, 64), (1024, 1024, 8)] {
            let desc = desc::sampled_texture(PREFIX_TAU_LABEL, PREFIX_FORMAT, extent(w, h, layers));
            assert_eq!(prefix_bytes(w, h, layers), desc::upload_len(&desc) as u64);
        }
        for (w, h) in [(64u32, 64u32), (160, 160), (512, 512), (1024, 1024)] {
            let layers = orientation_layers(w, h);
            assert!((1..=MAX_ORIENTATION_LAYERS).contains(&layers));
            assert!(
                prefix_bytes(w, h, layers) <= PREFIX_BUDGET_BYTES,
                "a {w}×{h} pen volume of {layers} layers is past its own budget",
            );
        }
    }

    /// **A second brush's bake evicts the first once the budget is exceeded**, oldest
    /// first, and the newest is kept whatever it cost — [`VolumeCache`]'s bound, over
    /// the list [`cached`] keeps it in, with each entry's bytes standing in for its
    /// view.
    ///
    /// Eviction is safe because a volume rebuilds from the coverage its id names, and
    /// what comes back is what `build` returns: the cache never hands out a stale value
    /// for a key it has dropped, and never holds two answers to one key.
    #[test]
    fn the_byte_budget_evicts_the_oldest_volume() {
        const BUDGET: u64 = 100;
        let lru: Lru<u32, u64> = Arc::new(Mutex::new(Vec::new()));
        let over = |kept: &[(u32, u64)]| kept.iter().map(|(_, b)| b).sum::<u64>() > BUDGET;
        let bake = |key: u32, bytes: u64| cached(&lru, key, over, || bytes);
        let keys = || -> Vec<u32> { unpoisoned(lru.lock()).iter().map(|(k, _)| *k).collect() };

        // Two that fit together, then a third that does not: the oldest goes.
        assert_eq!(bake(1, 40), (40, vec![]));
        assert_eq!(bake(2, 40), (40, vec![]));
        assert_eq!(bake(3, 40), (40, vec![40]));
        assert_eq!(keys(), vec![2, 3]);

        // A hit is a hit — nothing baked, nothing evicted — and it is now the newest,
        // so the next overflow takes the other one. The `0` would be the answer if it
        // had baked.
        assert_eq!(bake(2, 0), (40, vec![]));
        assert_eq!(keys(), vec![3, 2]);
        assert_eq!(bake(4, 40), (40, vec![40]));
        assert_eq!(keys(), vec![2, 4]);

        // The evicted one rebuilds to exactly what it was, and comes back as the only
        // entry under its key.
        assert_eq!(bake(3, 40), (40, vec![40]));
        assert_eq!(keys(), vec![4, 3]);

        // One larger than the whole budget is still handed back and still kept: baking
        // it again on every render is the only other answer.
        assert_eq!(bake(5, BUDGET * 2), (BUDGET * 2, vec![40, 40]));
        assert_eq!(keys(), vec![5]);
    }
}
