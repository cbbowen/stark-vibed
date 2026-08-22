//! Tileable 2-D noise fields for color dynamics (§6.2).
//!
//! Each [`stark_model::document::NoiseKind`] is baked **once, on the CPU,
//! with fixed constants** into a small `Rgba8Snorm` 2-D texture (three
//! independent signed channels; alpha unused) and sampled in the stamp shaders
//! with a repeat sampler. Baking on the CPU keeps the field bit-identical across
//! GPUs, runs, and peers — the same determinism contract as the sRGB↔Oklab
//! constants (§6.5) — and the bake uses only IEEE add/mul/floor/sqrt (all
//! correctly rounded, no transcendentals), so the bytes are reproducible across
//! platforms.
//!
//! Two axes are enough because the lookup is **stroke-local** — across the
//! stroke and along it — so the field never has to resolve a third, canvas
//! axis.
//!
//! Tileability is exact, not blended:
//! - **White** noise is per-texel hashed, so it wraps trivially.
//! - **Simplex** noise is evaluated on a genuinely periodic simplex grid: a
//!   lattice point's gradient is hashed from `q = 6·(i,j,k) − (i+j+k)·(1,1,1)`
//!   (six times its *unskewed* position — always integral) reduced modulo
//!   `6·PERIOD`. Translating the input by `PERIOD` along an axis maps each
//!   lattice point to one whose `q` differs by exactly `6·PERIOD` on that axis,
//!   so the hash — and the noise — repeats exactly. (`PERIOD` must be a multiple
//!   of 3 for the skewed cell indices to translate integrally.)
//!
//!   The lattice stays **three-dimensional** even though the bake is a plane:
//!   the periodic-gradient trick above needs the unskewed lattice positions to
//!   be integral, which holds in 3-D (`G3 = 1/6`) and *not* in 2-D, where
//!   `G2 = (3−√3)/6` is irrational — a 2-D simplex grid can be made periodic
//!   along its own skewed lattice vectors, but not along the axes, which is
//!   exactly what a tileable texture needs. So the field is the 3-D one
//!   restricted to `z = 0`: still smooth, still exactly axis-periodic.
//! - **Voronoi** (Worley F1) noise puts one feature point per grid cell, placed
//!   by hashing the cell index reduced modulo `PERIOD`, so translating the input
//!   by `PERIOD` lands on cells with identical hashes and the field repeats
//!   exactly. Only the 3×3 cells around the sample are searched, which here is
//!   not an approximation: every feature outside that ring is more than one cell
//!   away, so the search is exact wherever the true `F1 ≤ 1`, and the shaping
//!   flattens everything past 0.8 cells anyway (see [`VORONOI_MEAN`]).
//! - **Mosaic** noise is the same cell grid read discretely — each cell's own
//!   constant value, so the tile is flat polygons with hard edges — and inherits
//!   the same exact wrap: the value, like the site, is hashed from the cell index
//!   modulo `PERIOD`.

use crate::gpu::context::GpuContext;
use stark_model::document::NoiseKind;

/// Texels per side of a baked 2-D noise tile. Enough for fields that vary
/// smoothly across a cell; [`NoiseKind::Mosaic`] needs more (see [`MOSAIC_RES`]).
pub const NOISE_RES: u32 = 64;
/// Texels per side of the mosaic tile. Its cell walls are *steps*, and a step
/// is only ever as sharp as the tile is fine — at [`NOISE_RES`] a wall would be
/// a staircase of 4 canvas px treads at frequency 1. Baking it 4× finer puts the
/// wall inside one canvas pixel, where the sampler's own filtering hides it.
const MOSAIC_RES: u32 = 256;
/// Stroke-local pixels (across the stroke and along its arc) spanned by one
/// noise tile at frequency 1. The shader lookup is
/// `coord · frequency / NOISE_TILE_PX`.
pub const NOISE_TILE_PX: f32 = 256.0;
/// Simplex lattice units per noise tile — the tile holds this many noise
/// "features" per side. Must be a multiple of 3 (see the module docs).
const SIMPLEX_PERIOD: i32 = 6;
/// Voronoi cells per noise tile per side — matched to [`SIMPLEX_PERIOD`] so the
/// frequency knobs mean the same thing whichever kind is chosen.
const VORONOI_PERIOD: i32 = 6;
/// Nearest-feature distance (in cell units) that the Voronoi field maps to 0 —
/// roughly its mean, so the field wanders both ways about the brush color
/// instead of only darkening or only lightening it.
const VORONOI_MEAN: f32 = 0.4;
/// Value per cell unit of nearest-feature distance: the field peaks at +1 on a
/// feature point and bottoms out at −1 at `VORONOI_MEAN + 1/VORONOI_GAIN` = 0.8
/// cells — inside the radius where the 3×3 search is exact (module docs), so the
/// clamp can never expose a missed feature.
const VORONOI_GAIN: f32 = 2.5;
/// Fixed bake seeds, one per color channel.
const CHANNEL_SEEDS: [u32; 3] = [0x51ab_1e01, 0x51ab_1e02, 0x51ab_1e03];
/// Fixed mosaic bake seeds: one places the cell sites, one draws each cell's
/// value. Both are shared by the three channels on purpose — one set of cells in
/// every channel is what makes a facet a facet.
const MOSAIC_SITE_SEED: u32 = 0x51ab_1e11;
const MOSAIC_VALUE_SEED: u32 = 0x51ab_1e12;

/// Bake `kind` and upload it as an `Rgba8Snorm` 2-D texture.
pub fn build_noise_texture(
    ctx: &GpuContext,
    kind: NoiseKind,
) -> (wgpu::Texture, wgpu::TextureView) {
    let res = tile_res(kind);
    let bytes = bake(kind, res);
    upload_2d(ctx, res, &bytes, "stark noise field")
}

/// Texels per side of `kind`'s tile.
fn tile_res(kind: NoiseKind) -> u32 {
    match kind {
        NoiseKind::Mosaic => MOSAIC_RES,
        _ => NOISE_RES,
    }
}

/// A 1×1 zero tile: sampled offset is exactly 0, so binding it makes the
/// jitter a no-op without a second shader path.
pub fn dummy_noise_texture(ctx: &GpuContext) -> (wgpu::Texture, wgpu::TextureView) {
    upload_2d(ctx, 1, &[0u8; 4], "stark noise dummy")
}

fn upload_2d(
    ctx: &GpuContext,
    res: u32,
    bytes: &[u8],
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let extent = wgpu::Extent3d {
        width: res,
        height: res,
        depth_or_array_layers: 1,
    };
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Snorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    ctx.queue.write_texture(
        texture.as_image_copy(),
        bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(res * 4),
            rows_per_image: Some(res),
        },
        extent,
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Bake `kind` at `res`² texels: 4 bytes per texel (`Rgba8Snorm`), three signed
/// noise channels in ≈[-1, 1] plus an unused alpha.
fn bake(kind: NoiseKind, res: u32) -> Vec<u8> {
    let n = res as usize;
    let mut out = vec![0u8; n * n * 4];
    for y in 0..n {
        for x in 0..n {
            let i = (y * n + x) * 4;
            let v = match kind {
                NoiseKind::White => white_at(x as u32, y as u32),
                NoiseKind::Simplex => {
                    // Texel centres over `SIMPLEX_PERIOD` lattice units per side,
                    // on the `z = 0` plane of the periodic 3-D field (module docs).
                    let s = SIMPLEX_PERIOD as f32 / res as f32;
                    let p = [(x as f32 + 0.5) * s, (y as f32 + 0.5) * s, 0.0];
                    [
                        periodic_simplex(p, SIMPLEX_PERIOD, CHANNEL_SEEDS[0]),
                        periodic_simplex(p, SIMPLEX_PERIOD, CHANNEL_SEEDS[1]),
                        periodic_simplex(p, SIMPLEX_PERIOD, CHANNEL_SEEDS[2]),
                    ]
                }
                NoiseKind::Voronoi => {
                    // Texel centres over `VORONOI_PERIOD` cells per side.
                    let s = VORONOI_PERIOD as f32 / res as f32;
                    let p = [(x as f32 + 0.5) * s, (y as f32 + 0.5) * s];
                    [
                        periodic_voronoi(p, VORONOI_PERIOD, CHANNEL_SEEDS[0]),
                        periodic_voronoi(p, VORONOI_PERIOD, CHANNEL_SEEDS[1]),
                        periodic_voronoi(p, VORONOI_PERIOD, CHANNEL_SEEDS[2]),
                    ]
                }
                NoiseKind::Mosaic => {
                    // Same cells as `Voronoi`, read flat: all three channels come
                    // from the owning cell, so the facets are whole polygons.
                    let s = VORONOI_PERIOD as f32 / res as f32;
                    let p = [(x as f32 + 0.5) * s, (y as f32 + 0.5) * s];
                    periodic_mosaic(p, VORONOI_PERIOD)
                }
            };
            for (c, val) in v.iter().enumerate() {
                out[i + c] = (val.clamp(-1.0, 1.0) * 127.0).round() as i8 as u8;
            }
        }
    }
    out
}

/// Three independent white-noise channels for one texel, in [-1, 1].
fn white_at(x: u32, y: u32) -> [f32; 3] {
    let h = pcg4d([x, y, 0, CHANNEL_SEEDS[0]]);
    [unit(h[0]), unit(h[1]), unit(h[2])].map(|u| u * 2.0 - 1.0)
}

/// The pcg4d hash (Jarzynski & Olano, JCGT 2020).
///
/// It has no GPU counterpart to agree with, and deliberately so: the shader
/// samples the texture this bakes rather than re-deriving it, which is the whole
/// reason the field is bit-identical across adapters (see the module header).
/// `lib/noise.wesl` did once carry a mirror of this hash — uncalled by any pass,
/// so nothing ever compared the two — and it was deleted rather than kept as a
/// contract neither side exercised.
fn pcg4d(mut v: [u32; 4]) -> [u32; 4] {
    for x in v.iter_mut() {
        *x = x.wrapping_mul(1664525).wrapping_add(1013904223);
    }
    let mix = |v: &mut [u32; 4]| {
        v[0] = v[0].wrapping_add(v[1].wrapping_mul(v[3]));
        v[1] = v[1].wrapping_add(v[2].wrapping_mul(v[0]));
        v[2] = v[2].wrapping_add(v[0].wrapping_mul(v[1]));
        v[3] = v[3].wrapping_add(v[1].wrapping_mul(v[2]));
    };
    mix(&mut v);
    for x in v.iter_mut() {
        *x ^= *x >> 16;
    }
    mix(&mut v);
    v
}

/// u32 → uniform f32 in [0, 1).
fn unit(h: u32) -> f32 {
    h as f32 * 2.328_306_4e-10 // h / 2^32
}

/// The 12 cube-edge gradients of classic simplex/Perlin noise.
const GRAD3: [[f32; 3]; 12] = [
    [1.0, 1.0, 0.0],
    [-1.0, 1.0, 0.0],
    [1.0, -1.0, 0.0],
    [-1.0, -1.0, 0.0],
    [1.0, 0.0, 1.0],
    [-1.0, 0.0, 1.0],
    [1.0, 0.0, -1.0],
    [-1.0, 0.0, -1.0],
    [0.0, 1.0, 1.0],
    [0.0, -1.0, 1.0],
    [0.0, 1.0, -1.0],
    [0.0, -1.0, -1.0],
];

/// Gradient for the lattice point with skewed integer coords `(i, j, k)`,
/// periodic with `period` in *input* space: hash `q = 6·(i,j,k) − (i+j+k)` (six
/// times the unskewed position, always integral) reduced modulo `6·period`.
fn grad_at(i: i64, j: i64, k: i64, period: i32, seed: u32) -> [f32; 3] {
    let m = 6 * period as i64;
    let s = i + j + k;
    let q = [
        (6 * i - s).rem_euclid(m) as u32,
        (6 * j - s).rem_euclid(m) as u32,
        (6 * k - s).rem_euclid(m) as u32,
    ];
    let h = pcg4d([q[0], q[1], q[2], seed]);
    GRAD3[(h[0] % 12) as usize]
}

/// Classic 3-D simplex noise (Gustavson's formulation), exactly periodic with
/// `period` along every axis (a multiple of 3 — see the module docs). Output
/// ≈[-1, 1].
fn periodic_simplex(p: [f32; 3], period: i32, seed: u32) -> f32 {
    const F3: f32 = 1.0 / 3.0;
    const G3: f32 = 1.0 / 6.0;
    debug_assert!(
        period % 3 == 0,
        "the simplex lattice only closes on a period that is a multiple of 3"
    );

    let s = (p[0] + p[1] + p[2]) * F3;
    let i = (p[0] + s).floor();
    let j = (p[1] + s).floor();
    let k = (p[2] + s).floor();
    let t = (i + j + k) * G3;
    // Distances from the cell origin.
    let x0 = p[0] - (i - t);
    let y0 = p[1] - (j - t);
    let z0 = p[2] - (k - t);

    // Rank the offsets to pick the two middle corners of the simplex.
    let (i1, j1, k1, i2, j2, k2) = if x0 >= y0 {
        if y0 >= z0 {
            (1, 0, 0, 1, 1, 0)
        } else if x0 >= z0 {
            (1, 0, 0, 1, 0, 1)
        } else {
            (0, 0, 1, 1, 0, 1)
        }
    } else if y0 < z0 {
        (0, 0, 1, 0, 1, 1)
    } else if x0 < z0 {
        (0, 1, 0, 0, 1, 1)
    } else {
        (0, 1, 0, 1, 1, 0)
    };

    let corners = [
        (0.0, 0.0, 0.0, 0i64, 0i64, 0i64),
        (i1 as f32, j1 as f32, k1 as f32, i1, j1, k1),
        (i2 as f32, j2 as f32, k2 as f32, i2, j2, k2),
        (1.0, 1.0, 1.0, 1, 1, 1),
    ];
    let (bi, bj, bk) = (i as i64, j as i64, k as i64);

    let mut total = 0.0f32;
    for (c, &(di, dj, dk, oi, oj, ok)) in corners.iter().enumerate() {
        let g = c as f32 * G3;
        let x = x0 - di + g;
        let y = y0 - dj + g;
        let z = z0 - dk + g;
        let t = 0.6 - x * x - y * y - z * z;
        if t > 0.0 {
            let grad = grad_at(bi + oi, bj + oj, bk + ok, period, seed);
            let t2 = t * t;
            total += t2 * t2 * (grad[0] * x + grad[1] * y + grad[2] * z);
        }
    }
    // 32 normalizes classic simplex with the 0.6 kernel to roughly [-1, 1].
    32.0 * total
}

/// Feature point of the cell with integer coords `(i, j)`, as an offset in
/// [0, 1]² from the cell's corner, periodic with `period`: the hash reads the
/// cell index reduced modulo the period, so cells a whole period apart carry the
/// same point.
fn voronoi_feature(i: i64, j: i64, period: i32, seed: u32) -> [f32; 2] {
    let m = period as i64;
    let h = pcg4d([i.rem_euclid(m) as u32, j.rem_euclid(m) as u32, 0, seed]);
    [unit(h[0]), unit(h[1])]
}

/// Voronoi (Worley F1) cellular noise on a periodic jittered grid, exactly
/// periodic with `period` along both axes. Output in [-1, 1]: +1 on a feature
/// point, falling off with distance to the nearest one, with a crease where two
/// cells meet.
fn periodic_voronoi(p: [f32; 2], period: i32, seed: u32) -> f32 {
    // Shift the grid per seed so the three channels' cell walls don't all land
    // on the same lines. A constant translation keeps the field periodic.
    let o = pcg4d([seed, 0, 0, 0x9e37_79b9]);
    let p = [
        p[0] + unit(o[0]) * period as f32,
        p[1] + unit(o[1]) * period as f32,
    ];

    let (cx, cy) = (p[0].floor(), p[1].floor());
    let mut nearest2 = f32::INFINITY;
    for dj in -1..=1i64 {
        for di in -1..=1i64 {
            let (i, j) = (cx as i64 + di, cy as i64 + dj);
            let f = voronoi_feature(i, j, period, seed);
            let dx = (i as f32 + f[0]) - p[0];
            let dy = (j as f32 + f[1]) - p[1];
            nearest2 = nearest2.min(dx * dx + dy * dy);
        }
    }
    ((VORONOI_MEAN - nearest2.sqrt()) * VORONOI_GAIN).clamp(-1.0, 1.0)
}

/// The flat value of the Voronoi cell owning `p` — three channels from one hash
/// of the owning cell, so all three share the *same* polygons (three
/// independently celled channels would read as overlapping patchwork, not
/// facets). Exactly periodic with `period` on both axes; output in [-1, 1].
///
/// The search covers 5×5 cells where [`periodic_voronoi`] needs only 3×3: this
/// field has no clamp behind which a missed feature could hide — picking the
/// wrong owner would draw a wrong polygon, in full contrast. A sample's own cell
/// always holds a site, so the true nearest is at most √2 cells away, while
/// every site outside the 5×5 ring is more than 2 cells away: the owner is
/// always found.
fn periodic_mosaic(p: [f32; 2], period: i32) -> [f32; 3] {
    let (cx, cy) = (p[0].floor(), p[1].floor());
    let (mut nearest2, mut owner) = (f32::INFINITY, (0i64, 0i64));
    for dj in -2..=2i64 {
        for di in -2..=2i64 {
            let (i, j) = (cx as i64 + di, cy as i64 + dj);
            let f = voronoi_feature(i, j, period, MOSAIC_SITE_SEED);
            let dx = (i as f32 + f[0]) - p[0];
            let dy = (j as f32 + f[1]) - p[1];
            let d2 = dx * dx + dy * dy;
            if d2 < nearest2 {
                nearest2 = d2;
                owner = (i, j);
            }
        }
    }
    let m = period as i64;
    let h = pcg4d([
        owner.0.rem_euclid(m) as u32,
        owner.1.rem_euclid(m) as u32,
        0,
        MOSAIC_VALUE_SEED,
    ]);
    [unit(h[0]), unit(h[1]), unit(h[2])].map(|u| u * 2.0 - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The continuous simplex field must repeat exactly every `SIMPLEX_PERIOD`
    /// along each axis — the property that makes the baked volume tileable.
    #[test]
    fn simplex_is_periodic() {
        let p = SIMPLEX_PERIOD as f32;
        for seed in CHANNEL_SEEDS {
            for n in 0..64 {
                let h = pcg4d([n, 7, 11, seed]);
                let base = [unit(h[0]) * p, unit(h[1]) * p, unit(h[2]) * p];
                let v0 = periodic_simplex(base, SIMPLEX_PERIOD, seed);
                for axis in 0..3 {
                    let mut q = base;
                    q[axis] += p;
                    let v1 = periodic_simplex(q, SIMPLEX_PERIOD, seed);
                    assert!(
                        (v0 - v1).abs() < 1e-3,
                        "seed {seed:#x} axis {axis} at {base:?}: {v0} vs {v1}"
                    );
                }
            }
        }
    }

    /// The Voronoi field must repeat exactly every `VORONOI_PERIOD` along each
    /// axis — same tileability contract as the simplex field.
    #[test]
    fn voronoi_is_periodic() {
        let p = VORONOI_PERIOD as f32;
        for seed in CHANNEL_SEEDS {
            for n in 0..64 {
                let h = pcg4d([n, 7, 11, seed]);
                let base = [unit(h[0]) * p, unit(h[1]) * p];
                let v0 = periodic_voronoi(base, VORONOI_PERIOD, seed);
                for axis in 0..2 {
                    let mut q = base;
                    q[axis] += p;
                    let v1 = periodic_voronoi(q, VORONOI_PERIOD, seed);
                    assert!(
                        (v0 - v1).abs() < 1e-3,
                        "seed {seed:#x} axis {axis} at {base:?}: {v0} vs {v1}"
                    );
                }
            }
        }
    }

    /// The Voronoi field is a distance field: continuous everywhere (creases are
    /// kinks, not jumps) and never steeper than `VORONOI_GAIN` per cell unit. A
    /// jump would mean the 3×3 search missed a feature the neighbouring sample
    /// found — the failure mode the search bound rules out.
    #[test]
    fn voronoi_is_continuous() {
        let seed = CHANNEL_SEEDS[0];
        let n = 4000;
        let step = VORONOI_PERIOD as f32 / n as f32;
        for axis in 0..2 {
            for line in 0..8u32 {
                let h = pcg4d([line, 3, 5, 77]);
                let mut p = [
                    unit(h[0]) * VORONOI_PERIOD as f32,
                    unit(h[1]) * VORONOI_PERIOD as f32,
                ];
                p[axis] = 0.0;
                let mut prev = periodic_voronoi(p, VORONOI_PERIOD, seed);
                for i in 0..n {
                    p[axis] = (i as f32 + 1.0) * step;
                    let v = periodic_voronoi(p, VORONOI_PERIOD, seed);
                    assert!(
                        (v - prev).abs() < 1.01 * VORONOI_GAIN * step,
                        "discontinuity at {p:?} (axis {axis}): {prev} -> {v}"
                    );
                    prev = v;
                }
            }
        }
    }

    /// The Voronoi shaping constants must keep the field centred and using its
    /// range: a field biased to one side would tint every stroke rather than let
    /// the color wander both ways.
    #[test]
    fn voronoi_is_centred_and_uses_its_range() {
        let n = 64usize;
        let a = bake(NoiseKind::Voronoi, n as u32);
        for c in 0..3 {
            let vals: Vec<f32> = a[c..]
                .iter()
                .step_by(4)
                .map(|&b| b as i8 as f32 / 127.0)
                .collect();
            let mean = vals.iter().sum::<f32>() / vals.len() as f32;
            let lo = vals.iter().cloned().fold(f32::INFINITY, f32::min);
            let hi = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            assert!(mean.abs() < 0.15, "channel {c} biased: mean {mean}");
            assert!(lo < -0.8 && hi > 0.8, "channel {c} range: [{lo}, {hi}]");
        }
    }

    /// The mosaic field must repeat exactly every `VORONOI_PERIOD` along each
    /// axis — including the *owner* it picks, since a wrapped cell that resolved
    /// to a different owner would put a facet edge on the seam.
    #[test]
    fn mosaic_is_periodic() {
        let p = VORONOI_PERIOD as f32;
        for n in 0..256 {
            let h = pcg4d([n, 7, 11, MOSAIC_SITE_SEED]);
            let base = [unit(h[0]) * p, unit(h[1]) * p];
            let v0 = periodic_mosaic(base, VORONOI_PERIOD);
            for axis in 0..2 {
                let mut q = base;
                q[axis] += p;
                let v1 = periodic_mosaic(q, VORONOI_PERIOD);
                assert_eq!(v0, v1, "axis {axis} at {base:?}");
            }
        }
    }

    /// The mosaic must be *flat polygons with hard edges*: exactly one value per
    /// cell, the same cells in all three channels. A baked tile therefore holds
    /// exactly as many distinct texel values as the grid has cells — more would
    /// mean a channel celled on its own grid, or a wall smeared into a ramp of
    /// in-between values; fewer, a cell the sample grid never reaches.
    #[test]
    fn mosaic_is_one_flat_value_per_cell() {
        let n = MOSAIC_RES as usize;
        let a = bake(NoiseKind::Mosaic, n as u32);
        let facets: std::collections::HashSet<[u8; 3]> = a
            .as_chunks::<4>()
            .0
            .iter()
            .map(|t| [t[0], t[1], t[2]])
            .collect();
        let cells = (VORONOI_PERIOD * VORONOI_PERIOD) as usize;
        assert_eq!(facets.len(), cells, "distinct values vs {cells} cells");
    }

    /// The bake must be deterministic (replay/peers depend on it) and in range,
    /// with real variation in every channel.
    #[test]
    fn bake_is_deterministic_and_varied() {
        for kind in [
            NoiseKind::White,
            NoiseKind::Simplex,
            NoiseKind::Voronoi,
            NoiseKind::Mosaic,
        ] {
            let a = bake(kind, 16);
            let b = bake(kind, 16);
            assert_eq!(a, b);
            for c in 0..3 {
                let vals: Vec<i8> = a[c..].iter().step_by(4).map(|&b| b as i8).collect();
                let lo = *vals.iter().min().unwrap();
                let hi = *vals.iter().max().unwrap();
                assert!(lo < -20 && hi > 20, "{kind:?} channel {c}: [{lo}, {hi}]");
            }
        }
    }

    /// The continuous simplex field must be smooth — no discontinuity anywhere,
    /// which would betray a broken periodic gradient hash or corner selection.
    /// A fine step (6/4000 lattice units) may move the value only by ~that step
    /// times the field's steepest slope (measured ≈ 5.5/unit for this kernel).
    #[test]
    fn simplex_is_continuous() {
        let seed = CHANNEL_SEEDS[0];
        let n = 4000;
        let step = SIMPLEX_PERIOD as f32 / n as f32;
        for axis in 0..3 {
            for line in 0..8u32 {
                let h = pcg4d([line, 3, 5, 77]);
                let mut p = [
                    unit(h[0]) * SIMPLEX_PERIOD as f32,
                    unit(h[1]) * SIMPLEX_PERIOD as f32,
                    unit(h[2]) * SIMPLEX_PERIOD as f32,
                ];
                p[axis] = 0.0;
                let mut prev = periodic_simplex(p, SIMPLEX_PERIOD, seed);
                for i in 0..n {
                    p[axis] = (i as f32 + 1.0) * step;
                    let v = periodic_simplex(p, SIMPLEX_PERIOD, seed);
                    assert!(
                        (v - prev).abs() < 10.0 * step,
                        "discontinuity at {p:?} (axis {axis}): {prev} -> {v}"
                    );
                    prev = v;
                }
            }
        }
    }

    /// Tileability of the *baked* tile: stepping across the wrap seam
    /// (texel N−1 → texel 0) must look exactly like stepping anywhere in the
    /// interior — a broken wrap shows up as an outsized seam step. White noise is
    /// excluded: it is discontinuous by construction, so seam and interior steps
    /// are equally large and the comparison says nothing.
    #[test]
    fn smooth_bake_seams_match_interior() {
        let n = 64usize;
        for kind in [NoiseKind::Simplex, NoiseKind::Voronoi] {
            let a = bake(kind, n as u32);
            let texel = |x: usize, y: usize| a[(y * n + x) * 4] as i8 as i32;
            let (mut interior_max, mut seam_max) = (0, 0);
            for y in 0..n {
                for x in 0..n {
                    for (d, on_seam) in [
                        ((texel(x, y) - texel((x + 1) % n, y)).abs(), x + 1 == n),
                        ((texel(x, y) - texel(x, (y + 1) % n)).abs(), y + 1 == n),
                    ] {
                        if on_seam {
                            seam_max = seam_max.max(d);
                        } else {
                            interior_max = interior_max.max(d);
                        }
                    }
                }
            }
            assert!(
                seam_max <= interior_max,
                "{kind:?} wrap seam steps ({seam_max}) exceed interior steps ({interior_max})"
            );
            // And the field resolves its features: steps stay well under full range.
            assert!(
                interior_max < 100,
                "{kind:?} field under-resolved: step {interior_max}"
            );
        }
    }
}
