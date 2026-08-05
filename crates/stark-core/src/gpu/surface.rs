//! The physical canvas **surface** — a tileable ground that decides both where paint
//! lands (the deposition tooth) and how the relief catches light, §6.4.
//!
//! It is a single global, color-space-independent texture sampled in *canvas*
//! space (so the weave pans and zooms with the canvas), shared by the stroke
//! renderer (deposition) and the compositor (shading). Cloning is cheap — wgpu
//! views/samplers are reference-counted.
//!
//! The texture carries the height in `R` — the media pass's relief, §6.3 — and, in
//! `GB`, the **rise the ground makes one [`TOOTH_REACH`] ahead** along each canvas
//! axis ([`pack_ground`]). The rise is the whole of the deposition model: what a
//! dragged tip contacts is not a level set of the height but the *slope of the ground
//! along its own travel* — it is pressed up by ground rising to meet it and left
//! hanging by ground falling away — so paint catches on the near faces of the grain
//! and bridges the lee sides. The rise is baked here, once, because nothing at draw
//! time should be recomputing a filter — which also means the filter can be as
//! carefully chosen as the model deserves.

use serde::{Deserialize, Serialize};

use crate::assets::{AssetId, encode_gray_png};
use crate::error::{EngineError, Result};
use crate::gpu::context::{GpuContext, MAX_TEXTURE_DIM_2D};

/// Canvas pixels spanned by one full tile of the surface texture. The bump wraps
/// (Repeat sampling), so this sets the apparent weave scale; both the deposition
/// and shading passes must use the same value for the texture to line up.
pub const SURFACE_TILE_PX: f32 = 1024.0;

/// Canvas px → surface-tile uv, which is all a shader needs of [`SURFACE_TILE_PX`].
///
/// A function rather than a second constant so the deposition tooth (§6.4) and the
/// media pass cannot end up quoting different scales: the weave the paint catches on
/// and the weave the light catches on have to be the *same* weave, or the highlights
/// sit beside the grain instead of on it.
pub fn grain_uv_scale() -> f32 {
    1.0 / SURFACE_TILE_PX
}

/// Which physical surface a document is painted on. Saved in `CanvasMeta` (§8)
/// because which canvas a piece was painted on is part of the document, so it is
/// reproducible.
///
/// **Two variants, and the split is the point.** `Flat` is procedural and needs no
/// bytes; every other ground *is* its bytes, named by the hash of them. There is no
/// third case — no ground named by a label whose image the engine would have to be
/// told about separately — because that case is exactly the one that can go missing
/// (§6.4). A peer, a save file or a replay that meets an
/// [`Image`](Self::Image) id it has never seen can always ask for it by content, and
/// verify what comes back; a ground called "Gesso" could only be looked up in a
/// table the asker might not have, and the miss was silent — the tooth read a flat
/// stand-in and baked it into the tiles.
///
/// So this is the same bargain brush shapes already make (§6.6): the id
/// comes *from* the image ([`Engine::import_surface`](crate::Engine::import_surface)),
/// which is what makes "built-in" a property of the frontend's asset list and of
/// nothing downstream. The engine still embeds no image bytes.
#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub enum SurfaceId {
    /// Perfectly smooth: zero height everywhere, so the
    /// constant height has zero gradient (no relief). Paint behaves exactly as if
    /// there were no surface — the orthogonal default.
    #[default]
    Flat,
    /// A height map, named by the BLAKE3 hash of its canonical decoded form
    /// ([`surface_id`]). Covers the grounds that ship with the app and the ones a
    /// user brings, identically — the engine cannot tell them apart, which is why
    /// neither can go missing in a way the other wouldn't.
    Image(AssetId),
}

/// A canvas surface: the ground texture (height + the rise ahead) plus a tiling
/// sampler.
#[derive(Clone)]
pub struct Surface {
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    /// 1.0 if this is a real (image) surface with weave to interact with, 0.0 for
    /// the procedural `Flat`. Lets effects keyed on surface relief (e.g. the knife's
    /// scrape, §6.2) be a no-op on `Flat`, whose height is a constant 0.
    pub relief: f32,
    /// The ground's **rise-along-the-travel** histogram, one row per travel
    /// direction: the fraction of texels whose rise ahead ([`rise_ahead`]) falls in
    /// each of the 256 bins spanning ±[`RISE_LIMIT`], for a tip travelling along
    /// `2πk/BEARING_DIRS`.
    ///
    /// It exists to answer one question — [`bearing`](Self::bearing) — and that
    /// question is what makes a **toothed smear conserve paint** (§6.4). The canvas
    /// side of the exchange gates each texel by the ground under *it*; the tool has
    /// no per-texel ground, so it books its side against the mean, and the mean of a
    /// gate over the rise field is a sum over this table.
    ///
    /// It is a table of rows rather than one row **because the rise is directional**
    /// (§6.4). Contact is decided by `ahead·d̂`, so reversing a stroke negates the
    /// field and lands on the mirrored distribution — on a ground whose faces are
    /// asymmetric, a materially different one. Booking every direction against a
    /// single mean would leak paint at exactly the rate the direction matters.
    ///
    /// Rows are binned on the **decoded 8-bit** rise the shader itself reads, so the
    /// two sides draw from one distribution texel for texel — the same reason the
    /// shaders tap the map with nearest and not bilinear. What is no longer exact is
    /// the direction: the row grid quantizes it (interpolated between neighbours) and
    /// the bins quantize the projection. Both residuals are far under the mean-field
    /// freeze either side of the kernel already carries.
    ///
    /// `Arc` so a `Surface` stays two atomic bumps to clone — it is cloned per
    /// stroke, and 16 KB per clone is not the shape of this type.
    bearing_hist: std::sync::Arc<[[f32; 256]; BEARING_DIRS]>,
}

/// Width of the tooth's contact transition, in the rise's own units — height per
/// [`TOOTH_REACH`] of travel.
///
/// **Must match `TOOTH_SOFTNESS` in `paint_common.wesl`.** The trio below is a
/// deliberate mirror of the shader's, and the mirror is load-bearing rather than
/// convenient: the canvas evaluates the gate per texel on the GPU while the tool
/// books its side against [`Surface::bearing`] on the CPU, so if the two functions
/// disagree the two halves of the transfer disagree and a smear stops conserving.
/// That is also what guards it — `tests/dynamics.rs`'s conservation pair is sensitive
/// to exactly this, so a drift here fails a test rather than quietly leaking paint.
///
/// 0.06 is the bundled grounds' own interquartile rise (±0.023–0.031 on gesso,
/// ±0.047–0.078 on linen), so the transition spans the grain's natural variation:
/// narrower reads as binary speckle, much wider smears the faces into a flat grey.
const TOOTH_SOFTNESS: f32 = 0.06;

/// The **contact scale**: the rise, per [`TOOTH_REACH`] of travel, that a full-tooth
/// tip demands of the ground before it presses — and the unit the knob's follow
/// limit is measured in ([`tooth_level`]).
///
/// **Must match `TOOTH_RISE` in `paint_common.wesl`** (same mirror as
/// [`TOOTH_SOFTNESS`]). Measured, not picked: the mean |rise| over the reach is
/// 0.037–0.043 on gesso and 0.060–0.090 on linen, so 0.05 asks of the ground
/// roughly its own typical face — `tooth = 1` catches the leading edges that stand
/// out of the weave and nothing else, without any real ground gating to nothing.
const TOOTH_RISE: f32 = 0.05;

/// The rise the map's `GB` encoding spans: a stored byte covers ±this, in height
/// units per [`TOOTH_REACH`] ([`encode_rise`]/[`decode_rise`], and the span of
/// [`Surface::bearing_hist`]'s bins).
///
/// A quarter of the height range, because the rise *is* small: it is a difference
/// across a few canvas px of a field the antialias has already smoothed, and across
/// both bundled grounds its 99th percentile is under 0.26. Spanning ±1 would spend
/// three quarters of the byte on values no ground produces and leave the gate's
/// whole transition ([`TOOTH_SOFTNESS`]) eight quanta wide — which prints as tone
/// *steps* across the grain. What a pathological ground loses to the clamp is
/// nothing visible: the gate saturates at `±(TOOTH_RISE + TOOTH_SOFTNESS/2)`, far
/// inside it, and both halves of the transfer read the same clamped byte, so even
/// the booking agrees (§6.4).
const RISE_LIMIT: f32 = 0.25;

/// The span the rise is measured across — how far ahead of itself a moving tip reads
/// the ground, in **canvas px** (§6.4).
///
/// It is a *distance* rather than a gain because that is what makes the rise mean
/// something. A tip dragged across a rough ground does not settle onto the height
/// under it; it rides up onto whatever it is about to meet, so it bears on the near
/// face of every bump and bridges the lee side behind it. The slope that decides
/// contact at a texel is therefore the ground's rise a short way *along the direction
/// of travel* — [`rise_ahead`].
///
/// A canvas px rather than a texel or a fraction of the tip: the reach is a property
/// of the contact, and it must not change when the same weave is stored at a different
/// resolution (which the downsample in [`canonical_height`] does routinely) or when a
/// larger brush paints the same ground. The rise baked into the map is measured across
/// this distance in the map's own texels for exactly that reason.
///
/// **3 px is measured against the grounds rather than picked.** It is the distance at
/// which the rise a tip meets stops growing: across the bundled weaves the mean rise
/// over the reach climbs steeply to about 2 px and then flattens (0.038 → 0.056 →
/// 0.069 → 0.078 on gesso for 1.5, 2, 3, 4 px), because past a feature's own width
/// there is no more face to climb. The reach lands on the shoulder of that curve —
/// short enough that the rise is still the face under the tip rather than a plain
/// translation of the mark, long enough that no face it could catch is missed.
const TOOTH_REACH: f32 = 3.0;

/// The scale the ground is resolved at before its rise is measured, in **canvas px**.
///
/// Half a deposited texel: the smallest blur that answers the map's minification (about
/// two map texels per canvas px at [`SURFACE_TILE_PX`], read nearest) without touching
/// the grain itself. Without it the rise picks up the map's Nyquist noise, which has no
/// direction a tip could catch on and prints as a dither that flips with the stroke;
/// much above it and the faces the tooth exists to find are blurred away with it.
///
/// It is deliberately *not* tied to [`TOOTH_REACH`]. The band-limiting that matters is
/// already done by measuring the rise across the reach — a difference over a span `L`
/// is blind to what repeats faster than `L` — so this one has only the sampling grid to
/// answer for.
const GROUND_ANTIALIAS: f32 = 0.5;

/// How many travel directions [`Surface::bearing_hist`] is tabulated at.
///
/// The bearing is a smooth, low-harmonic function of direction — a constant on an
/// isotropic ground, four-fold on a woven one — so sixteen samples resolve it well
/// past anything a real weave carries, and the lookup interpolates between neighbours
/// so a curving stroke does not step. It is also why the table is affordable: the
/// build is one pass over the map per direction, and the result is 16 KB.
const BEARING_DIRS: usize = 16;

/// The steepest fall the tip can still follow, negated — the level the gate
/// thresholds the rise against, from the `tooth` knob (see
/// `paint_common.wesl::tooth_level`, which explains the `2 − 1/tooth` map and the
/// inert floor under the division).
fn tooth_level(tooth: f32) -> f32 {
    TOOTH_RISE * (2.0 - 1.0 / tooth.max(0.01))
}

/// The share of its paint a texel receives, given the rise `d` of the ground along
/// the tip's travel there (`paint_common.wesl::tooth_gate`).
fn tooth_gate(d: f32, tooth: f32) -> f32 {
    let t = ((d - tooth_level(tooth)) / TOOTH_SOFTNESS + 0.5).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Decode one axis of the baked rise from its stored byte
/// (`paint_common.wesl::rise_ahead`).
///
/// Written as the shader writes it, constant for constant: `e/255` is what a texture
/// unit hands back for a `Rgba8Unorm` channel, and the constants that follow are the
/// shader's own folded literals (`255/512` is dyadic, so the fold is exact), so the
/// CPU's rows bin the numbers the GPU will actually project.
fn decode_rise(e: u8) -> f32 {
    (e as f32 / 255.0) * (255.0 / 512.0) - 0.25
}

/// The **rise ahead** at a texel under a tip travelling along `d̂`: how much higher
/// the ground it is about to meet stands than the ground here, `ahead·d̂` — the
/// height's derivative along the travel, taken across the reach
/// (`paint_common.wesl::rise_ahead`).
fn rise_ahead(ahead: [f32; 2], dir: [f32; 2]) -> f32 {
    ahead[0] * dir[0] + ahead[1] * dir[1]
}

impl Surface {
    /// A perfectly smooth surface: a 1×1 *zero-height* texel. Paint always stands
    /// above it (so it shows everywhere) and the constant height has zero gradient
    /// (no relief) — exactly equivalent to having no surface (§6.4).
    pub fn flat(ctx: &GpuContext) -> Self {
        Self {
            relief: 0.0,
            ..Self::from_height(ctx, &[0u8], 1, 1)
        }
    }

    /// The **bearing fraction** at a given tooth and direction of travel: what share
    /// of the ground a tip with this `tooth`, going this way, stands on — averaged
    /// over the whole height map (§6.4).
    ///
    /// This is the model's own central quantity — the mean of the contact gate over
    /// the ground's rise distribution — and it is where paint conservation comes
    /// from. A canvas texel gates its half of the exchange by the ground under
    /// itself; the tool has no ground of its own, so it gates its half by this. The
    /// two agree in expectation over any footprint that spans many grain features,
    /// which is every usable tip, and the residual is the ground's sampling
    /// fluctuation under the tip — the same order as the mean-field freeze the loop
    /// already carries.
    ///
    /// **`dir` is not optional, it is the field.** Contact is decided by the rise
    /// *along the travel*, so the curve is a property of the map and the direction
    /// crossing it together: a tip crossing a weave along the warp meets a different
    /// distribution of faces than one crossing it diagonally, and one running the
    /// stroke backwards meets the mirrored distribution outright. Interpolated
    /// between the two nearest tabulated rows so a curving stroke's booking is
    /// continuous — the tool must not step when the pen turns.
    ///
    /// Exactly 1 where there is nothing to bite: no tooth, or no relief. Not
    /// *approximately* 1 — the sum over a histogram would land a rounding error away,
    /// and that error would be a systematic leak on every stroke of every ordinary
    /// brush.
    pub fn bearing(&self, tooth: f32, dir: [f32; 2]) -> f32 {
        if tooth <= 0.0 || self.relief <= 0.0 {
            return 1.0;
        }
        let turns = dir[1].atan2(dir[0]) / std::f32::consts::TAU * BEARING_DIRS as f32;
        let lo = turns.floor();
        let f = turns - lo;
        let i0 = (lo as i32).rem_euclid(BEARING_DIRS as i32) as usize;
        let i1 = (i0 + 1) % BEARING_DIRS;
        let b0 = self.row_mean(i0, tooth);
        let b1 = self.row_mean(i1, tooth);
        b0 + (b1 - b0) * f
    }

    /// The mean gate over one tabulated direction's rise distribution. Bins are the
    /// encode lattice itself ([`tabulate_bearing`]), so [`decode_rise`] is what turns
    /// one back into the rise the gate reads.
    fn row_mean(&self, dir: usize, tooth: f32) -> f32 {
        let mut mean = 0.0;
        for (bin, share) in self.bearing_hist[dir].iter().enumerate() {
            mean += share * tooth_gate(decode_rise(bin as u8), tooth);
        }
        mean
    }

    /// Build from a height-map PNG. The bytes reached the registry through
    /// [`canonicalize`] or [`identify`], which decoded them once already, so a
    /// failure here is a broken invariant rather than bad input — hence the
    /// `expect` rather than a `Result` the caller would have nothing to do with.
    pub fn load(ctx: &GpuContext, png_bytes: &[u8]) -> Self {
        let (w, h, height) = canonical_height(png_bytes).expect("surface: registered bytes decode");
        Self::from_height(ctx, &height, w, h)
    }

    /// Upload a height field as the `Rgba8Unorm` **ground** texture — height in `R`,
    /// the rise ahead in `GB` ([`pack_ground`]) — and tabulate the bearing
    /// curve it implies.
    fn from_height(ctx: &GpuContext, height: &[u8], w: u32, h: u32) -> Self {
        let ground = pack_ground(height, w, h);
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("stark surface ground"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        ctx.queue.write_texture(
            texture.as_image_copy(),
            &ground,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("stark surface sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            view,
            sampler,
            relief: 1.0,
            bearing_hist: std::sync::Arc::new(tabulate_bearing(&ground)),
        }
    }
}

/// Bake the ground texture: height in `R`, and in `GB` the **rise ahead** —
/// how much higher the ground stands one [`TOOTH_REACH`] ahead along each canvas
/// axis, in the map's own [0, 1] height units, encoded about 128 over
/// ±[`RISE_LIMIT`].
///
/// **The reach is baked in here rather than applied in the shader**, and that is the
/// choice that keeps the whole axis at one texture tap. The deposit reads its ground
/// once per fragment and gets both terms of `s + ahead·d̂` out of it, so the leading
/// edge costs a dot product on top of what the plain tooth already spent — nothing on
/// the fast path's fill rate, nothing in the loop's inner block. It also puts the
/// filter budget where it is free: the rise can be as carefully computed as we like
/// because nothing at draw time recomputes it.
///
/// **The rise is a difference across the reach, not a slope at the texel**, and that
/// is the whole of the filtering question. What [`rise_ahead`] wants is
/// `s(x + reach·d̂) − s(x)`, so that is what is measured: a central difference over the
/// reach itself, per axis, which the shader's dot product then resolves onto the
/// direction of travel.
///
/// Writing it as `reach·∇s` instead — the obvious form — is the trap. A gradient at a
/// point is a *local* quantity multiplied out to a distance it knows nothing about, so
/// it grows without bound in the reach and reports whatever the finest scale in the map
/// happens to be doing: on the bundled grounds it climbs linearly past the height
/// spread it is supposed to displace, and gating on it prints the map's own Nyquist
/// noise as a dither that flips with the stroke. The difference across the span is
/// self-limiting instead — it saturates once the reach clears a feature's width,
/// because past that there is no more face to climb — and it is *inherently*
/// band-limited: a difference over a span is blind to what repeats faster than the
/// span. Only the sampling grid is left to answer for, which [`GROUND_ANTIALIAS`] does.
///
/// Wrapped at the edges because the map tiles; a clamped kernel would print a false
/// ridge down the seam every `SURFACE_TILE_PX`.
///
/// The span is the reach converted into **each axis's own texels** (`dims /
/// SURFACE_TILE_PX` texels per canvas px), so the same weave reads identically
/// however it was stored: halve a map's resolution and the span in texels halves with
/// it. That is what makes the integer downsample in [`canonical_height`] invisible to
/// the mark, and it is why [`TOOTH_REACH`] can be a physical distance at all. The span
/// rounds to whole texels — never below one — and the half-texel that costs is well
/// inside the blur it is measured on.
fn pack_ground(height: &[u8], w: u32, h: u32) -> Vec<u8> {
    let (wi, hi) = (w as i32, h as i32);
    let per_px = |texels: u32| texels as f32 / SURFACE_TILE_PX;
    let smooth = blur(
        height,
        w,
        h,
        GROUND_ANTIALIAS * per_px(w),
        GROUND_ANTIALIAS * per_px(h),
    );
    let at =
        |x: i32, y: i32| -> f32 { smooth[(y.rem_euclid(hi) * wi + x.rem_euclid(wi)) as usize] };
    // Half the reach each way, so the difference spans the reach.
    let span = |texels: u32| ((0.5 * TOOTH_REACH * per_px(texels)).round() as i32).max(1);
    let (sx, sy) = (span(w), span(h));
    let mut out = vec![255u8; (w as usize) * (h as usize) * 4];
    for y in 0..hi {
        for x in 0..wi {
            let i = ((y * wi + x) as usize) * 4;
            out[i] = height[(y * wi + x) as usize];
            out[i + 1] = encode_rise(at(x + sx, y) - at(x - sx, y));
            out[i + 2] = encode_rise(at(x, y + sy) - at(x, y - sy));
        }
    }
    out
}

/// A separable, wrapping Gaussian blur of a height map into [0, 1] floats.
///
/// Kernels are built and normalized in the same order on every machine, so two peers
/// blurring one ground land on the same floats and the tooth they deposit through is
/// the same tooth — the ground is a replay input like any other (§6.4).
fn blur(height: &[u8], w: u32, h: u32, sigma_x: f32, sigma_y: f32) -> Vec<f32> {
    /// Gaussian weights out to 3σ, normalized to sum to 1. A non-positive σ has
    /// nothing to blur and returns the identity kernel; a σ far under a texel reaches
    /// the same place through the arithmetic, since the off-centre weights underflow.
    fn kernel(sigma: f32) -> Vec<f32> {
        let r = (3.0 * sigma).ceil().max(0.0) as i32;
        if r == 0 {
            return vec![1.0];
        }
        let mut k: Vec<f32> = (-r..=r)
            .map(|i| (-0.5 * (i as f32 / sigma).powi(2)).exp())
            .collect();
        let total: f32 = k.iter().sum();
        for weight in &mut k {
            *weight /= total;
        }
        k
    }
    let (wi, hi) = (w as i32, h as i32);
    let (kx, ky) = (kernel(sigma_x), kernel(sigma_y));
    let (rx, ry) = (kx.len() as i32 / 2, ky.len() as i32 / 2);
    let mut across = vec![0.0f32; (w as usize) * (h as usize)];
    for y in 0..hi {
        for x in 0..wi {
            let mut sum = 0.0;
            for (i, weight) in kx.iter().enumerate() {
                let sx = (x + i as i32 - rx).rem_euclid(wi);
                sum += weight * height[(y * wi + sx) as usize] as f32;
            }
            across[(y * wi + x) as usize] = sum / 255.0;
        }
    }
    let mut out = vec![0.0f32; (w as usize) * (h as usize)];
    for y in 0..hi {
        for x in 0..wi {
            let mut sum = 0.0;
            for (i, weight) in ky.iter().enumerate() {
                let sy = (y + i as i32 - ry).rem_euclid(hi);
                sum += weight * across[(sy * wi + x) as usize];
            }
            out[(y * wi + x) as usize] = sum;
        }
    }
    out
}

/// One axis of the rise, into the byte [`decode_rise`] reads back: ±[`RISE_LIMIT`]
/// across the byte, for the reasons given there — real rises are small, and spending
/// the byte on the range they actually occupy is what keeps the gate's transition
/// tonal rather than stepped.
fn encode_rise(rise: f32) -> u8 {
    ((rise / RISE_LIMIT).clamp(-1.0, 1.0) * 128.0 + 128.0)
        .round()
        .clamp(0.0, 255.0) as u8
}

/// Tabulate the rise's distribution, one row per travel direction
/// ([`Surface::bearing_hist`]).
///
/// One pass per direction over the ground, binning the **decoded** rise rather than
/// the float it came from, so the tool books against the numbers the shader will
/// project. Opposite directions are one negation apart — `ahead·(−d̂) = −(ahead·d̂)`,
/// exactly, in floats too — so the sixteen rows cost eight dot products a texel.
///
/// The bins are the encode lattice itself ([`encode_rise`] in, [`decode_rise`] out),
/// which is not laziness but the boundary condition that matters: zero rise is a
/// *lattice point* (byte 128), so the projections that hover a rounding error either
/// side of flat — every texel of an axis-aligned weave crossed at right angles —
/// land in one bin from both directions instead of straddling an edge. It also means
/// an on-axis crossing bins the map's own stored byte, re-quantization error zero.
fn tabulate_bearing(ground: &[u8]) -> [[f32; 256]; BEARING_DIRS] {
    const HALF: usize = BEARING_DIRS / 2;
    let dirs: [[f32; 2]; HALF] = std::array::from_fn(|k| {
        let a = std::f32::consts::TAU * k as f32 / BEARING_DIRS as f32;
        [a.cos(), a.sin()]
    });
    let mut counts = [[0u32; 256]; BEARING_DIRS];
    let texels = ground.as_chunks::<4>().0;
    for texel in texels {
        let ahead = [decode_rise(texel[1]), decode_rise(texel[2])];
        for (k, d) in dirs.iter().enumerate() {
            let rise = rise_ahead(ahead, *d);
            counts[k][encode_rise(rise) as usize] += 1;
            counts[k + HALF][encode_rise(-rise) as usize] += 1;
        }
    }
    let n = texels.len().max(1) as f32;
    std::array::from_fn(|k| std::array::from_fn(|i| counts[k][i] as f32 / n))
}

/// Import a height map: the id that names it, and the canonical bytes to keep
/// beside it. Re-encoded from the decoded height, so what is stored, bundled into a
/// save file and sent to a peer is the form the id actually names — reload it and
/// you land on the same id.
///
/// The engine's entry point is
/// [`Engine::import_surface`](crate::Engine::import_surface).
pub fn canonicalize(png_bytes: &[u8]) -> Result<(SurfaceId, Vec<u8>)> {
    let (w, h, height) = canonical_height(png_bytes)?;
    let id = SurfaceId::Image(surface_id(w, h, &height));
    Ok((id, encode_gray_png(w, h, &height)?))
}

/// The id of an already-canonical height map — bytes out of a save file or off a
/// peer, which are kept verbatim. Derived rather than taken on trust: a ground whose
/// bytes did not hash to the id that asked for them is a ground that would silently
/// deposit the wrong tooth, so the caller gets the id the bytes *are* and compares.
pub fn identify(png_bytes: &[u8]) -> Result<SurfaceId> {
    let (w, h, height) = canonical_height(png_bytes)?;
    Ok(SurfaceId::Image(surface_id(w, h, &height)))
}

/// Content id of a canonical height field: the hash of its dimensions and texels.
///
/// Over the *decoded, downsampled* field rather than the file bytes, for the reason
/// [`AssetId`] names a brush's coverage the same way — it is what actually drives
/// pixels, so two peers who encoded the same weave differently converge on one id.
/// That this is deterministic across peers rests on [`MAX_TEXTURE_DIM_2D`] being a
/// fixed constant rather than a device query: were the downsample factor to follow
/// the adapter's real limit, the same PNG would canonicalize differently on two
/// machines and the id would stop naming one thing.
fn surface_id(width: u32, height: u32, texels: &[u8]) -> AssetId {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&width.to_le_bytes());
    hasher.update(&height.to_le_bytes());
    hasher.update(texels);
    AssetId(*hasher.finalize().as_bytes())
}

/// Decode a height-map PNG to its canonical form: one height byte per texel,
/// box-downsampled by an integer factor to fit [`MAX_TEXTURE_DIM_2D`].
///
/// Channel 0, not luminance — a height map's grey *is* its height, so an RGB source
/// carries it in red and weighting the channels would tilt the ground.
fn canonical_height(png_bytes: &[u8]) -> Result<(u32, u32, Vec<u8>)> {
    let decoder = png::Decoder::new(std::io::Cursor::new(png_bytes));
    let mut reader = decoder
        .read_info()
        .map_err(|e| EngineError::Asset(e.to_string()))?;
    let size = reader
        .output_buffer_size()
        .ok_or_else(|| EngineError::Asset("surface: missing png size".into()))?;
    let mut buf = vec![0u8; size];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| EngineError::Asset(e.to_string()))?;
    let (w, h) = (info.width, info.height);

    // Collapse to one height byte per texel (the source is 8-bit grayscale, but
    // accept the common color types defensively).
    let n = (w * h) as usize;
    let height: Vec<u8> = match info.color_type {
        png::ColorType::Grayscale => buf[..n].to_vec(),
        png::ColorType::GrayscaleAlpha => buf.as_chunks::<2>().0.iter().map(|p| p[0]).collect(),
        png::ColorType::Rgb => buf.as_chunks::<3>().0.iter().map(|p| p[0]).collect(),
        png::ColorType::Rgba => buf.as_chunks::<4>().0.iter().map(|p| p[0]).collect(),
        other => {
            return Err(EngineError::Asset(format!(
                "surface: unsupported PNG color type {other:?}"
            )));
        }
    };

    let (height, w, h) = downsample_to_limit(height, w, h, MAX_TEXTURE_DIM_2D);
    Ok((w, h, height))
}

/// Box-downsample a single-channel image by the smallest integer factor that
/// brings both edges within `limit`. An integer factor keeps a tileable texture
/// tileable; `factor == 1` returns the input unchanged.
pub(crate) fn downsample_to_limit(src: Vec<u8>, w: u32, h: u32, limit: u32) -> (Vec<u8>, u32, u32) {
    let factor = w.div_ceil(limit).max(h.div_ceil(limit)).max(1);
    if factor == 1 {
        return (src, w, h);
    }
    let (nw, nh) = (w / factor, h / factor);
    let area = factor * factor;
    let mut out = vec![0u8; (nw * nh) as usize];
    for y in 0..nh {
        for x in 0..nw {
            let mut sum = 0u32;
            for dy in 0..factor {
                for dx in 0..factor {
                    let sx = x * factor + dx;
                    let sy = y * factor + dy;
                    sum += src[(sy * w + sx) as usize] as u32;
                }
            }
            out[(y * nw + x) as usize] = (sum / area) as u8;
        }
    }
    (out, nw, nh)
}

impl crate::gpu::registry::Resource for SurfaceId {
    type Gpu = Surface;

    /// `Flat` is a 1x1 full-height texel: a constant
    /// height has zero gradient, so it is exactly equivalent to having no surface
    /// (§6.4). It is the only ground with no bytes behind it, which is what makes
    /// "the id names an image the holder may not have yet" a question with exactly
    /// one shape.
    fn is_builtin(self) -> bool {
        matches!(self, SurfaceId::Flat)
    }

    fn build(self, gpu: &GpuContext, bytes: Option<&[u8]>) -> Surface {
        match bytes {
            Some(bytes) if !self.is_builtin() => Surface::load(gpu, bytes),
            _ => Surface::flat(gpu),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ground of **ramps**: height climbing steadily to a peak, then dropping back
    /// over a few texels. Every feature has a long near face and a short far one, and —
    /// the point of the shape — the height histogram is the same one whichever way a
    /// tip crosses it, so a gate reading only the ground *under* the tip cannot tell
    /// the two runs apart at all.
    fn ramps(w: u32, h: u32, period: u32) -> Vec<u8> {
        let climb = period - period / 8;
        (0..w * h)
            .map(|i| {
                let x = (i % w) % period;
                if x < climb {
                    (255 * x / climb.max(1)) as u8
                } else {
                    (255 - 255 * (x - climb) / (period - climb).max(1)) as u8
                }
            })
            .collect()
    }

    /// The mean gate over one tabulated direction, off a table rather than a `Surface`
    /// — everything under test here is the model's arithmetic, and none of it needs a
    /// GPU to be wrong.
    fn row_mean(hist: &[[f32; 256]; BEARING_DIRS], dir: usize, tooth: f32) -> f32 {
        hist[dir]
            .iter()
            .enumerate()
            .map(|(bin, share)| share * tooth_gate(decode_rise(bin as u8), tooth))
            .sum()
    }

    /// **The sign of the whole model.** Dragged up the near faces of a ramped ground a
    /// tip meets rising ground the entire way, so it contacts more of it than its own
    /// height would say; dragged down them it is bridging a falling ground the entire
    /// way and contacts less. Anything that reversed the derivative — a swapped
    /// subtraction in the bake, a negated projection in the shader's mirror — turns
    /// this inequality round, and no amount of "the mark changed when I reversed the
    /// stroke" would notice.
    ///
    /// Direction 0 is `+x`, the way the ramps climb; direction `DIRS/2` is `−x`.
    #[test]
    fn a_tip_dragged_up_the_faces_contacts_more_than_one_dragged_down_them() {
        let (w, h) = (256, 256);
        let hist = tabulate_bearing(&pack_ground(&ramps(w, h, 32), w, h));
        for tooth in [0.3, 0.5, 0.7] {
            let up = row_mean(&hist, 0, tooth);
            let down = row_mean(&hist, BEARING_DIRS / 2, tooth);
            assert!(
                up > down * 1.05,
                "at tooth {tooth} a tip going up the ramps bore on {up} of the ground \
                 and one going down them on {down} — the anticipation has no sign, or \
                 the wrong one"
            );
        }
    }

    /// Across the ramps — `+y`, where the ground is flat — there is nothing rising or
    /// falling ahead, so the two opposite crossings must land on the *same* bearing.
    /// It is the null case the test above needs to mean anything: without it, a bake
    /// that simply added a constant would satisfy the inequality just as well.
    #[test]
    fn a_tip_crossing_the_ramps_sideways_reads_them_the_same_both_ways() {
        let (w, h) = (256, 256);
        let hist = tabulate_bearing(&pack_ground(&ramps(w, h, 32), w, h));
        let (quarter, three) = (BEARING_DIRS / 4, 3 * BEARING_DIRS / 4);
        for tooth in [0.3, 0.5, 0.7] {
            let (a, b) = (
                row_mean(&hist, quarter, tooth),
                row_mean(&hist, three, tooth),
            );
            assert!(
                (a - b).abs() < 1e-6,
                "at tooth {tooth} the two crossings of a ground with no slope along \
                 them disagreed: {a} vs {b}"
            );
        }
    }

    /// **A brush with no tooth is full contact, exactly, on any ground** — flat or as
    /// steep as the encoding can carry — and this is where that is pinned without a
    /// GPU. It is not the callers' `tooth <= 0` guard being retested: the point is
    /// the *approach* to zero, [`tooth_level`] diving far past any encodable fall as
    /// the give grows, so the gate is 1.0 over the whole rise range well before the
    /// knob reaches zero and a pen mapping that sweeps through it lands on the
    /// guard's value continuously rather than with a pop.
    #[test]
    fn a_brush_with_no_tooth_is_full_contact_on_any_ground() {
        let (w, h) = (64, 64);
        for ground in [vec![90u8; (w * h) as usize], ramps(w, h, 8)] {
            let hist = tabulate_bearing(&pack_ground(&ground, w, h));
            for dir in 0..BEARING_DIRS {
                assert_eq!(
                    row_mean(&hist, dir, 0.0),
                    1.0,
                    "direction {dir} gated a brush with no tooth at less than full \
                     contact — the follow limit is not outrunning the encodable falls \
                     as the knob closes"
                );
            }
        }
    }

    /// **The top of the knob still lays paint** — the reason the gate reads the
    /// ground's slope and not its height. A full-tooth tip demands rising ground, and
    /// a real ground *has* rising ground: dragged up the ramps it bears on the long
    /// climb faces (most of the area), dragged down them on the short steep ones
    /// (little, but strictly some). A height threshold at the top of its range gated
    /// everything to nothing, which made `tooth = 1` a knob position with no use.
    #[test]
    fn full_tooth_still_bears_on_the_faces_that_rise_to_meet_it() {
        let (w, h) = (256, 256);
        let hist = tabulate_bearing(&pack_ground(&ramps(w, h, 32), w, h));
        let up = row_mean(&hist, 0, 1.0);
        let down = row_mean(&hist, BEARING_DIRS / 2, 1.0);
        assert!(
            up > 0.5,
            "dragged up the ramps at full tooth the tip bears on only {up} of the \
             ground — the climb faces should carry it"
        );
        assert!(
            down > 0.05,
            "dragged down them it bears on {down} — even the lee run catches the \
             short faces that rise against it"
        );
        assert!(
            up < 1.0 && down < 1.0,
            "full tooth gated nothing at all: {up}, {down}"
        );
    }
}
