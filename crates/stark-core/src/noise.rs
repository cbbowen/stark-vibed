//! Tileable 3-D noise fields for colour dynamics (DESIGN.md §6.2).
//!
//! Each [`crate::document::NoiseKind`] is baked **once, on the CPU,
//! with fixed constants** into a small `Rgba8Snorm` 3-D texture (three
//! independent signed channels; alpha unused) and sampled in the stamp shaders
//! with a repeat sampler. Baking on the CPU keeps the field bit-identical across
//! GPUs, runs, and peers — the same determinism contract as the sRGB↔Oklab
//! constants (§6.5) — and the bake uses only IEEE add/mul/floor (no
//! transcendentals), so the bytes are reproducible across platforms.
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

use crate::document::NoiseKind;
use crate::gpu::context::GpuContext;

/// Texels per side of the baked 3-D noise volume.
pub const NOISE_RES: u32 = 64;
/// Canvas pixels (and arc-length pixels) spanned by one noise tile at
/// frequency 1. The shader lookup is `coord · frequency / NOISE_TILE_PX`.
pub const NOISE_TILE_PX: f32 = 256.0;
/// Simplex lattice units per noise tile — the tile holds this many noise
/// "features" per side. Must be a multiple of 3 (see the module docs).
const SIMPLEX_PERIOD: i32 = 6;
/// Fixed bake seeds, one per colour channel.
const CHANNEL_SEEDS: [u32; 3] = [0x51ab_1e01, 0x51ab_1e02, 0x51ab_1e03];

/// Bake `kind` and upload it as an `Rgba8Snorm` 3-D texture.
pub fn build_noise_texture(
    ctx: &GpuContext,
    kind: NoiseKind,
) -> (wgpu::Texture, wgpu::TextureView) {
    let bytes = bake(kind, NOISE_RES);
    upload_3d(ctx, NOISE_RES, &bytes, "stark noise field")
}

/// A 1×1×1 zero volume: sampled offset is exactly 0, so binding it makes the
/// jitter a no-op without a second shader path.
pub fn dummy_noise_texture(ctx: &GpuContext) -> (wgpu::Texture, wgpu::TextureView) {
    upload_3d(ctx, 1, &[0u8; 4], "stark noise dummy")
}

fn upload_3d(
    ctx: &GpuContext,
    res: u32,
    bytes: &[u8],
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let extent = wgpu::Extent3d {
        width: res,
        height: res,
        depth_or_array_layers: res,
    };
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
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

/// Bake `kind` at `res`³ texels: 4 bytes per texel (`Rgba8Snorm`), three signed
/// noise channels in ≈[-1, 1] plus an unused alpha.
fn bake(kind: NoiseKind, res: u32) -> Vec<u8> {
    let n = res as usize;
    let mut out = vec![0u8; n * n * n * 4];
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let i = ((z * n + y) * n + x) * 4;
                let v = match kind {
                    NoiseKind::White => white_at(x as u32, y as u32, z as u32),
                    NoiseKind::Simplex => {
                        // Texel centres over `SIMPLEX_PERIOD` lattice units per side.
                        let s = SIMPLEX_PERIOD as f32 / res as f32;
                        let p = [
                            (x as f32 + 0.5) * s,
                            (y as f32 + 0.5) * s,
                            (z as f32 + 0.5) * s,
                        ];
                        [
                            periodic_simplex(p, SIMPLEX_PERIOD, CHANNEL_SEEDS[0]),
                            periodic_simplex(p, SIMPLEX_PERIOD, CHANNEL_SEEDS[1]),
                            periodic_simplex(p, SIMPLEX_PERIOD, CHANNEL_SEEDS[2]),
                        ]
                    }
                };
                for (c, val) in v.iter().enumerate() {
                    out[i + c] = (val.clamp(-1.0, 1.0) * 127.0).round() as i8 as u8;
                }
            }
        }
    }
    out
}

/// Three independent white-noise channels for one texel, in [-1, 1].
fn white_at(x: u32, y: u32, z: u32) -> [f32; 3] {
    let h = pcg4d([x, y, z, CHANNEL_SEEDS[0]]);
    [unit(h[0]), unit(h[1]), unit(h[2])].map(|u| u * 2.0 - 1.0)
}

/// The pcg4d hash (Jarzynski & Olano) — mirrors `noise.wesl` so CPU and GPU
/// noise stay in the same family.
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
    debug_assert!(period % 3 == 0);

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

    /// The bake must be deterministic (replay/peers depend on it) and in range,
    /// with real variation in every channel.
    #[test]
    fn bake_is_deterministic_and_varied() {
        for kind in [NoiseKind::White, NoiseKind::Simplex] {
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

    /// Tileability of the *baked* volume: stepping across the wrap seam
    /// (texel N−1 → texel 0) must look exactly like stepping anywhere in the
    /// interior — a broken wrap shows up as an outsized seam step.
    #[test]
    fn simplex_bake_seam_matches_interior() {
        let n = 64usize;
        let a = bake(NoiseKind::Simplex, n as u32);
        let texel = |x: usize, y: usize, z: usize| a[((z * n + y) * n + x) * 4] as i8 as i32;
        let (mut interior_max, mut seam_max) = (0, 0);
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let d = (texel(x, y, z) - texel((x + 1) % n, y, z)).abs();
                    if x + 1 == n {
                        seam_max = seam_max.max(d);
                    } else {
                        interior_max = interior_max.max(d);
                    }
                }
            }
        }
        assert!(
            seam_max <= interior_max,
            "wrap seam steps ({seam_max}) exceed interior steps ({interior_max})"
        );
        // And the field resolves its features: steps stay well under full range.
        assert!(
            interior_max < 100,
            "field under-resolved: step {interior_max}"
        );
    }
}
