//! The **deposition tooth**: a CPU mirror of `paint_common.wesl`, and the ground
//! statistics it implies (§6.4).
//!
//! No GPU in this file. It is the half of the surface that decides *where paint
//! lands* — the contact gate, the rise the ground makes ahead of a moving tip, and
//! the bearing table that lets the tool book its side of a smear without having a
//! ground of its own.
//!
//! **The mirror is load-bearing, not convenient.** The canvas evaluates the gate per
//! texel on the GPU while the tool books against [`Bearing::at`] here, so if the two
//! functions disagree the two halves of a transfer disagree and a smear stops
//! conserving paint. `tests/dynamics.rs`'s conservation pair is sensitive to exactly
//! that, so a drift fails a test rather than quietly leaking paint — and everything
//! in this file is reachable without an adapter, which is why its own tests are too.

use std::sync::Arc;

use super::SURFACE_TILE_PX;

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
pub(super) fn pack_ground(height: &[u8], w: u32, h: u32) -> Vec<u8> {
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

/// The ground's **rise-along-the-travel** histogram, one row per travel direction:
/// the fraction of texels whose rise ahead ([`rise_ahead`]) falls in each of the 256
/// bins spanning ±[`RISE_LIMIT`], for a tip travelling along `2πk/BEARING_DIRS`.
///
/// It exists to answer one question — [`Self::at`] — and that
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
#[derive(Clone)]
pub(super) struct Bearing(Arc<[[f32; 256]; BEARING_DIRS]>);

impl Bearing {
    pub(super) fn tabulate(ground: &[u8]) -> Self {
        Self(Arc::new(tabulate_bearing(ground)))
    }

    pub(super) fn at(&self, tooth: f32, dir: [f32; 2]) -> f32 {
        if tooth <= 0.0 {
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
        for (bin, share) in self.0[dir].iter().enumerate() {
            mean += share * tooth_gate(decode_rise(bin as u8), tooth);
        }
        mean
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::wesl::wesl_const;

    /// The three constants this module and `lib/paint_common.wesl` **both** compute
    /// with, asserted rather than asked for in a comment.
    ///
    /// These are the worst kind of pair to leave to prose, because the failure has no
    /// symptom of its own. The CPU averages `tooth_gate` over the ground's rise
    /// distribution to get the bearing fraction the *tool* books its half of a
    /// transfer against, while the shader evaluates the same gate per texel for the
    /// *canvas* half (§6.4). Move either constant on one side and the two halves go on
    /// rendering perfectly plausible paint that no longer adds up — a conservation
    /// leak proportional to how far they drifted, which `tests/dynamics.rs` would
    /// eventually notice and no golden would localize.
    ///
    /// `RISE_LIMIT` is here at all because it stopped being folded into the literals
    /// `255.0 / 512.0` and `0.25`, which is what put it beyond reach of this check
    /// while its comment still claimed the folded constants had to match.
    ///
    /// Read through `stamp_oklab()`: the tooth gates its deposit, so all three
    /// survive stripping there. No adapter needed, so this holds in CI.
    #[test]
    fn the_host_and_the_shader_agree_on_the_tooths_constants() {
        let src = stark_shaders::stamp_oklab();
        for (name, ours) in [
            ("TOOTH_SOFTNESS", TOOTH_SOFTNESS),
            ("TOOTH_RISE", TOOTH_RISE),
            ("RISE_LIMIT", RISE_LIMIT),
        ] {
            // Narrowed to `f32`, which is what both sides actually hold. Widening
            // instead compares the host's rounded `0.06f32` against the shader
            // source's exact decimal and fails on every constant that is not a
            // power of two.
            assert_eq!(
                ours,
                wesl_const(src, name) as f32,
                "{name} has drifted between `gpu::surface::tooth` and \
                 `lib/paint_common.wesl`; the two halves of a toothed transfer no \
                 longer book against the same gate",
            );
        }
    }

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
