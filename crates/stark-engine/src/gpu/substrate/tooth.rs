//! The **deposition tooth**: a CPU mirror of `paint_common.wesl`, and the substrate
//! statistics it implies (§6.4).
//!
//! No GPU here. This is the half of the substrate that decides *where paint lands* —
//! the contact gate, the rise the substrate makes ahead of a moving tip, and the
//! bearing table that lets the tool book its side of a smear without a substrate of
//! its own.
//!
//! **The mirror is load-bearing.** The canvas evaluates the gate per texel on the GPU
//! while the tool books against [`Bearing::at`] here; if the two disagree the two
//! halves of a transfer disagree and a smear stops conserving paint, which
//! `tests/dynamics.rs`'s conservation pair is sensitive to.

use std::sync::Arc;

// The three constants this module and `lib/paint_common.wesl` **both** compute with,
// generated from the shader's own declarations (§6.10) so the two cannot drift. This
// module averages `tooth_gate` over the substrate's rise distribution for the *tool's*
// half of a transfer while the shader evaluates the same gate per texel for the
// *canvas* half (§6.4); a drift between them leaks paint with no symptom of its own.
//
// The transition's *width* is not a fourth: it is a brush parameter
// (`ToothParams::softness`), so it arrives as an argument and neither side declares it.
use stark_shaders::mirror::paint_common::{RISE_LIMIT, TOOTH_RISE, TOOTH_SOFTNESS_FLOOR};

/// The span the rise is measured across — how far ahead of itself a moving tip reads
/// the substrate, in **canvas px** (§6.4).
///
/// A *distance* rather than a gain: a tip dragged across a rough substrate does not
/// settle onto the height under it, it rides up onto whatever it is about to meet. So
/// what decides contact at a texel is the substrate's rise a short way *along the
/// direction of travel* — [`rise_ahead`].
///
/// Canvas px rather than texels or a fraction of the tip, so the same substrate bites
/// the same however it was stored (the downsample in
/// [`canonical_height`](super::import::canonical_height) rescales routinely) and whatever
/// brush paints it. It is also why laying a substrate at a different
/// [`SubstrateScale`](stark_model::SubstrateScale) needs a fresh bake rather than a
/// scaled lookup ([`Substrate`](super::Substrate)): the reach stays three canvas px, so
/// the span in texels moves under it.
///
/// **3 px is measured against the substrates rather than picked.** Across the bundled
/// substrates the mean rise over the reach climbs steeply to about 2 px and then
/// flattens (0.038 → 0.056 → 0.069 → 0.078 on the rough substrate for 1.5, 2, 3, 4 px),
/// because past a feature's own width there is no more face to climb. 3 px is the
/// shoulder of that curve.
const TOOTH_REACH: f32 = 3.0;

/// The scale the substrate is resolved at before its rise is measured, in **canvas px**.
///
/// Half a deposited texel: the smallest blur that answers the map's minification (about
/// two map texels per canvas px for a full-size map at natural scale, read nearest)
/// without touching the grain. Below it the rise picks up the map's Nyquist noise, which
/// has no direction a tip could catch on and prints as a dither that flips with the
/// stroke; much above it and the faces the tooth exists to find blur away with it.
///
/// Deliberately *not* tied to [`TOOTH_REACH`]: measuring the rise across the reach
/// already band-limits it, so this answers only for the sampling grid.
const SUBSTRATE_ANTIALIAS: f32 = 0.5;

/// How many travel directions [`SubstrateMap::bearing`](super::SubstrateMap::bearing) is
/// tabulated at. The bearing is a smooth, low-harmonic function of direction — constant
/// on an isotropic substrate, four-fold on a woven one — so sixteen rows resolve it past
/// anything a real substrate carries; the lookup interpolates between neighbours so a
/// curving stroke does not step. The whole table is 16 KB.
const BEARING_DIRS: usize = 16;

/// The steepest fall the tip can still follow, negated — the level the gate
/// thresholds the rise against, from the `tooth_give` knob (see
/// `paint_common.wesl::tooth_level`, which explains the `2 − 1/(1 − give)` map, why the
/// knob is the give rather than its inverse, and the inert floor under the division).
fn tooth_level(give: f32) -> f32 {
    TOOTH_RISE * (2.0 - 1.0 / (1.0 - give).max(0.01))
}

/// The share of its paint a texel receives, given the rise `d` of the substrate along
/// the tip's travel there (`paint_common.wesl::tooth_gate`) — the give the tip settles
/// with, and the width of the band it comes into contact over.
fn tooth_gate(d: f32, give: f32, softness: f32) -> f32 {
    let t = ((d - tooth_level(give)) / softness.max(TOOTH_SOFTNESS_FLOOR) + 0.5).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Decode one axis of the baked rise from its stored byte
/// (`paint_common.wesl::rise_ahead`).
///
/// Written as the shader writes it, constant for constant: `e/255` is what a texture
/// unit hands back for a `Rgba8Unorm` channel, and what follows is the same
/// `255·L/128 − L` the shader spells, so the CPU's rows bin the numbers the GPU will
/// actually project. Neither side folds [`RISE_LIMIT`] away into the literals `255/512`
/// and `0.25`: the fold is exact, but it would put the shared number beyond the reach of
/// any check.
fn decode_rise(e: u8) -> f32 {
    (e as f32 / 255.0) * (255.0 * RISE_LIMIT / 128.0) - RISE_LIMIT
}

/// The **rise ahead** at a texel under a tip travelling along `d̂`: how much higher
/// the substrate it is about to meet stands than the substrate here, `ahead·d̂` — the
/// height's derivative along the travel, taken across the reach
/// (`paint_common.wesl::rise_ahead`).
fn rise_ahead(ahead: [f32; 2], dir: [f32; 2]) -> f32 {
    ahead[0] * dir[0] + ahead[1] * dir[1]
}

/// Bake the substrate texture: height in `R`, and in `GB` the **rise ahead** —
/// how much higher the substrate stands one [`TOOTH_REACH`] ahead along each canvas
/// axis, in the map's own [0, 1] height units, encoded about 128 over
/// ±[`RISE_LIMIT`].
///
/// Baking the reach in rather than applying it in the shader keeps the whole axis at one
/// texture tap: the deposit reads its substrate once per fragment and gets both terms of
/// `s + ahead·d̂` out of it.
///
/// **The rise is a difference across the reach, not a slope at the texel.** What
/// [`rise_ahead`] wants is `s(x + reach·d̂) − s(x)`, so that is what is measured — a
/// central difference over the reach itself, per axis, which the shader's dot product
/// resolves onto the direction of travel. Writing it as `reach·∇s` is the trap: a local
/// gradient multiplied out to a distance it knows nothing about grows without bound in
/// the reach and reports whatever the finest scale in the map is doing, so gating on it
/// prints the map's Nyquist noise as a dither that flips with the stroke. A difference
/// over a span saturates once the reach clears a feature's width, and is blind to what
/// repeats faster than the span; only the sampling grid is left to answer for, which
/// [`SUBSTRATE_ANTIALIAS`] does.
///
/// Wrapped at the edges because the map tiles; a clamped kernel would print a false
/// ridge down the seam every `tile_px`.
///
/// The span is the reach converted into **each axis's own texels** (`dims / tile_px`
/// texels per canvas px), so the same substrate reads identically however it was stored —
/// which is what makes the downsample in
/// [`canonical_height`](super::import::canonical_height) invisible to the mark, and why
/// [`TOOTH_REACH`] can be a physical distance at all. It rounds to whole texels, never
/// below one, and the half-texel that costs is well inside the blur.
///
/// `tile_px` is the canvas px one full tile of the map spans —
/// [`SUBSTRATE_TILE_PX`](super::SUBSTRATE_TILE_PX) times the document's scale
/// ([`Substrate::tile_px`](super::Substrate)). It is the *only* thing the scale changes
/// here, and everything downstream moves with it: the antialias σ, the span, and so the
/// bearing table tabulated off the result.
pub(super) fn pack_substrate(height: &[u8], w: u32, h: u32, tile_px: f32) -> Vec<u8> {
    let (wi, hi) = (w as i32, h as i32);
    let per_px = |texels: u32| texels as f32 / tile_px;
    let smooth = blur(
        height,
        w,
        h,
        SUBSTRATE_ANTIALIAS * per_px(w),
        SUBSTRATE_ANTIALIAS * per_px(h),
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
/// blurring one substrate land on the same floats: the substrate is a replay input like
/// any other (§6.4).
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
/// ([`SubstrateMap::bearing`](super::SubstrateMap::bearing)).
///
/// One pass per direction over the substrate, binning the **decoded** rise rather than
/// the float it came from, so the tool books against the numbers the shader will
/// project. Opposite directions are one exact negation apart, so the sixteen rows cost
/// eight dot products a texel.
///
/// The bins are the encode lattice itself ([`encode_rise`] in, [`decode_rise`] out),
/// which puts zero rise on a *lattice point* (byte 128): the projections that hover a
/// rounding error either side of flat — every texel of an axis-aligned substrate crossed
/// at right angles — land in one bin from both directions instead of straddling an edge.
fn tabulate_bearing(substrate: &[u8]) -> [[f32; 256]; BEARING_DIRS] {
    const HALF: usize = BEARING_DIRS / 2;
    let dirs: [[f32; 2]; HALF] = std::array::from_fn(|k| {
        let a = std::f32::consts::TAU * k as f32 / BEARING_DIRS as f32;
        [a.cos(), a.sin()]
    });
    let mut counts = [[0u32; 256]; BEARING_DIRS];
    let texels = substrate.as_chunks::<4>().0;
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

/// The substrate's **rise-along-the-travel** histogram, one row per travel direction:
/// the fraction of texels whose rise ahead ([`rise_ahead`]) falls in each of the 256
/// bins spanning ±[`RISE_LIMIT`], for a tip travelling along `2πk/BEARING_DIRS`.
///
/// It answers one question — [`Self::at`] — and that question is what makes a **toothed
/// smear conserve paint** (§6.4). The canvas side of the exchange gates each texel by
/// the substrate under *it*; the tool has no per-texel substrate, so it books its side
/// against the mean, and the mean of the gate over the rise field is a sum over this
/// table.
///
/// Rows rather than one row **because the rise is directional** (§6.4). Contact is
/// decided by `ahead·d̂`, so reversing a stroke lands on the mirrored distribution —
/// on a substrate whose faces are asymmetric, a materially different one. Booking every
/// direction against a single mean would leak paint at exactly the rate direction
/// matters.
///
/// Rows are binned on the **decoded 8-bit** rise the shader itself reads, so the two
/// sides draw from one distribution texel for texel — the same reason the shaders tap
/// the map with nearest and not bilinear.
///
/// `Arc` so a `SubstrateMap` stays two atomic bumps to clone; it is cloned per stroke.
#[derive(Clone)]
pub(super) struct Bearing(Arc<[[f32; 256]; BEARING_DIRS]>);

impl Bearing {
    pub(super) fn tabulate(substrate: &[u8]) -> Self {
        Self(Arc::new(tabulate_bearing(substrate)))
    }

    pub(super) fn at(&self, give: f32, softness: f32, dir: [f32; 2]) -> f32 {
        if give >= 1.0 {
            return 1.0;
        }
        let turns = dir[1].atan2(dir[0]) / std::f32::consts::TAU * BEARING_DIRS as f32;
        let lo = turns.floor();
        let f = turns - lo;
        let i0 = (lo as i32).rem_euclid(BEARING_DIRS as i32) as usize;
        let i1 = (i0 + 1) % BEARING_DIRS;
        let b0 = self.row_mean(i0, give, softness);
        let b1 = self.row_mean(i1, give, softness);
        b0 + (b1 - b0) * f
    }

    /// The mean gate over one tabulated direction's rise distribution. Bins are the
    /// encode lattice itself ([`tabulate_bearing`]), so [`decode_rise`] is what turns
    /// one back into the rise the gate reads.
    fn row_mean(&self, dir: usize, give: f32, softness: f32) -> f32 {
        let mut mean = 0.0;
        for (bin, share) in self.0[dir].iter().enumerate() {
            mean += share * tooth_gate(decode_rise(bin as u8), give, softness);
        }
        mean
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::substrate::SUBSTRATE_TILE_PX;
    use stark_model::document::ToothParams;

    /// A substrate of **ramps**: height climbing steadily to a peak, then dropping back
    /// over a few texels. Every feature has a long near face and a short far one, and —
    /// the point of the shape — the height histogram is the same one whichever way a
    /// tip crosses it, so a gate reading only the substrate *under* the tip cannot tell
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

    /// The contact transition every claim below about the *give* is read through.
    ///
    /// **The tests' own, deliberately not `ToothParams::DEFAULT_SOFTNESS`.** What the
    /// shipped default is set to is taste; what is asserted here is the deposition
    /// model, and a change of taste must not be able to fail a claim about the rise
    /// field.
    ///
    /// 0.06 because that is a *narrow* band on these substrates — their own
    /// interquartile rise — so the gate is still recognisably a threshold on the rise
    /// and the level-set language below means what it says. The one test that is about
    /// the width itself names both of its bands
    /// ([`a_softer_contact_bears_on_the_substrate_more_evenly`]).
    const NARROW: f32 = 0.06;

    /// The mean gate over one tabulated direction, off a table rather than a `SubstrateMap`
    /// — everything under test here is the model's arithmetic, and none of it needs a
    /// GPU to be wrong.
    fn row_mean_at(hist: &[[f32; 256]; BEARING_DIRS], dir: usize, give: f32, softness: f32) -> f32 {
        hist[dir]
            .iter()
            .enumerate()
            .map(|(bin, share)| share * tooth_gate(decode_rise(bin as u8), give, softness))
            .sum()
    }

    /// [`row_mean_at`] at the contact transition a brush gets when it does not say —
    /// the width every test below the softness pair reads through, so each of them is
    /// still a statement about the *give* alone.
    fn row_mean(hist: &[[f32; 256]; BEARING_DIRS], dir: usize, give: f32) -> f32 {
        row_mean_at(hist, dir, give, NARROW)
    }

    /// **The sign of the whole model.** Dragged up the near faces of a ramped substrate a
    /// tip meets rising substrate the entire way and contacts more of it than its own
    /// height would say; dragged down them it is bridging a falling substrate and
    /// contacts less. Anything that reversed the derivative — a swapped subtraction in
    /// the bake, a negated projection in the shader's mirror — turns this inequality
    /// round.
    ///
    /// Direction 0 is `+x`, the way the ramps climb; direction `DIRS/2` is `−x`.
    #[test]
    fn a_tip_dragged_up_the_faces_contacts_more_than_one_dragged_down_them() {
        let (w, h) = (256, 256);
        let hist = tabulate_bearing(&pack_substrate(&ramps(w, h, 32), w, h, SUBSTRATE_TILE_PX));
        for give in [0.3, 0.5, 0.7] {
            let up = row_mean(&hist, 0, give);
            let down = row_mean(&hist, BEARING_DIRS / 2, give);
            assert!(
                up > down * 1.05,
                "at give {give} a tip going up the ramps bore on {up} of the substrate \
                 and one going down them on {down} — the anticipation has no sign, or \
                 the wrong one"
            );
        }
    }

    /// Across the ramps — `+y`, where the substrate is flat — there is nothing rising or
    /// falling ahead, so the two opposite crossings must land on the *same* bearing.
    /// It is the null case the test above needs to mean anything: without it, a bake
    /// that simply added a constant would satisfy the inequality just as well.
    #[test]
    fn a_tip_crossing_the_ramps_sideways_reads_them_the_same_both_ways() {
        let (w, h) = (256, 256);
        let hist = tabulate_bearing(&pack_substrate(&ramps(w, h, 32), w, h, SUBSTRATE_TILE_PX));
        let (quarter, three) = (BEARING_DIRS / 4, 3 * BEARING_DIRS / 4);
        for give in [0.3, 0.5, 0.7] {
            let (a, b) = (row_mean(&hist, quarter, give), row_mean(&hist, three, give));
            assert!(
                (a - b).abs() < 1e-6,
                "at give {give} the two crossings of a substrate with no slope along \
                 them disagreed: {a} vs {b}"
            );
        }
    }

    /// **A brush with full give is full contact, exactly, on any substrate** — flat or
    /// as steep as the encoding can carry. Not the callers' `give >= 1` guard retested:
    /// the point is the *approach* to the top, [`tooth_level`] diving far past any
    /// encodable fall as the give grows, so a pen mapping that sweeps through it lands
    /// on the guard's value continuously rather than with a pop.
    #[test]
    fn a_brush_with_full_give_is_full_contact_on_any_substrate() {
        let (w, h) = (64, 64);
        for substrate in [vec![90u8; (w * h) as usize], ramps(w, h, 8)] {
            let hist = tabulate_bearing(&pack_substrate(&substrate, w, h, SUBSTRATE_TILE_PX));
            for dir in 0..BEARING_DIRS {
                assert_eq!(
                    row_mean(&hist, dir, ToothParams::DEFAULT_GIVE),
                    1.0,
                    "direction {dir} gated a brush with full give at less than full \
                     contact — the follow limit is not outrunning the encodable falls \
                     as the knob opens"
                );
            }
        }
    }

    /// **How large the substrate is laid changes what a tip bites**, which is the whole
    /// reason a substrate is baked per scale rather than sampled at a scaled uv
    /// (`super::Substrate`).
    ///
    /// The reach is three *canvas* px whatever the scale, so laying the substrate finer
    /// puts more of the grain inside it, and a tip anticipating several threads ahead
    /// spends more of its travel bridging falling substrate than riding one face. So the
    /// finer the substrate is laid, the *less* of it a climbing tip bears on — the mark
    /// going drier as the canvas gets tighter. Scaling one bake's lookup would report
    /// the rise over the *old* span under a new name, the compensating fudge §1 rules
    /// out.
    ///
    /// `tile_px` is passed directly rather than through a `SubstrateScale`, because what
    /// is under test is the bake's one input — the map's texels per canvas px — and
    /// the two ways to move it (a map's resolution, a document's scale) are the same
    /// number arriving.
    #[test]
    fn a_substrate_laid_finer_is_bridged_more_and_borne_on_less() {
        let (w, h) = (256, 256);
        let ramps = ramps(w, h, 32);
        // One canvas px per texel, then four: the reach spans 2 texels and then 6.
        let coarse = tabulate_bearing(&pack_substrate(&ramps, w, h, 256.0));
        let fine = tabulate_bearing(&pack_substrate(&ramps, w, h, 64.0));
        for give in [0.4, 0.2, 0.0] {
            let (c, f) = (row_mean(&coarse, 0, give), row_mean(&fine, 0, give));
            assert!(
                c > f * 1.05,
                "at give {give} the substrate laid coarser bore on {c} of the substrate and                  the finer one on {f} — the scale is not reaching the bake"
            );
        }
    }

    /// **The bottom of the knob still lays paint** — the reason the gate reads the
    /// substrate's slope and not its height. A tip with no give demands rising substrate,
    /// and a real substrate *has* rising substrate: dragged up the ramps it bears on the
    /// long climb faces (most of the area), dragged down them on the short steep ones
    /// (little, but strictly some). A height threshold at the end of its range would gate
    /// everything to nothing, making `give = 0` a knob position with no use.
    #[test]
    fn a_tip_with_no_give_still_bears_on_the_faces_that_rise_to_meet_it() {
        let (w, h) = (256, 256);
        let hist = tabulate_bearing(&pack_substrate(&ramps(w, h, 32), w, h, SUBSTRATE_TILE_PX));
        let up = row_mean(&hist, 0, 0.0);
        let down = row_mean(&hist, BEARING_DIRS / 2, 0.0);
        assert!(
            up > 0.5,
            "dragged up the ramps with no give the tip bears on only {up} of the \
             substrate — the climb faces should carry it"
        );
        assert!(
            down > 0.05,
            "dragged down them it bears on {down} — even the lee run catches the \
             short faces that rise against it"
        );
        assert!(
            up < 1.0 && down < 1.0,
            "a tip with no give gated nothing at all: {up}, {down}"
        );
    }

    /// **Softening the contact moves the bearing towards a half**, from whichever side
    /// it started — which is what says the second knob is a *width* and not a second
    /// depth. A narrow band is nearly an indicator of "does this texel clear the follow
    /// limit"; widening it takes from the faces that bore fully and gives to the ones
    /// that were bridged entirely, and in the limit every texel takes the same half
    /// share. That is the charcoal: the stick crumbles into the valleys instead of
    /// spanning them, so the mark stops being a level set of the grain and becomes a
    /// tone across it.
    ///
    /// Two gives, so the claim is about the *direction* of the move and cannot be
    /// satisfied by a gate that merely got smaller. Both bands are named here rather
    /// than taken from the brush's default, for [`NARROW`]'s reason.
    #[test]
    fn a_softer_contact_bears_on_the_substrate_more_evenly() {
        let (w, h) = (256, 256);
        let hist = tabulate_bearing(&pack_substrate(&ramps(w, h, 32), w, h, SUBSTRATE_TILE_PX));
        for (dir, give) in [(0, 0.0), (BEARING_DIRS / 2, 0.0), (0, 0.6)] {
            let hard = row_mean_at(&hist, dir, give, NARROW);
            let soft = row_mean_at(&hist, dir, give, 8.0 * NARROW);
            assert!(
                (soft - 0.5).abs() < (hard - 0.5).abs(),
                "direction {dir} at give {give} bore on {hard} of the substrate through the \
                 narrow band and {soft} through one eight times as wide — widening the \
                 transition is not evening the contact out"
            );
        }
    }

    /// **The floor under the division is inert.** A brush asking for no softness at all
    /// asks for a hard threshold, and `TOOTH_SOFTNESS_FLOOR` has to hand it one rather
    /// than a band of its own: the floor is two orders under the encode lattice's step,
    /// so no rise the map can carry lands inside it.
    ///
    /// Checked on the gate rather than through a bearing, because what is pinned is that
    /// the two are the *same function* on every value the lattice produces — which a
    /// mean over them could hide by averaging. The premise is asserted beside the
    /// conclusion: a rise landing *inside* the floor's band would gate to something
    /// between 0 and 1 legitimately, so the test says that none does.
    #[test]
    fn no_softness_at_all_is_the_hard_threshold_it_asks_for() {
        for give in [0.7, 0.4, 0.0] {
            let level = tooth_level(give);
            for bin in 0..=u8::MAX {
                let d = decode_rise(bin);
                assert!(
                    (d - level).abs() > TOOTH_SOFTNESS_FLOOR,
                    "the encode lattice puts a rise ({d}) inside the floor's own band \
                     at give {give} (level {level}) — the floor is no longer inert"
                );
                let gate = tooth_gate(d, give, 0.0);
                let hard = f32::from(d >= level);
                assert_eq!(
                    gate, hard,
                    "at give {give} a rise of {d} gated to {gate} with no softness \
                     asked for, where the threshold it straddles says {hard}"
                );
            }
        }
    }
}
