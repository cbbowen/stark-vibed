//! The Oklab picker's geometry: the display gamut's rim, the wheel fitted to it, and
//! the pictures of both (§6.7, §11.2 N8).
//!
//! **Which gamut is the caller's to say** (§6.5). Every function that touches the rim
//! takes a [`Gamut`], because "the most chroma there is at this lightness" has no
//! answer without one — and a frontend on a Display P3 surface has a wider one to
//! offer than a frontend on an sRGB surface. What comes back is always **extended
//! sRGB**, the space the document carries ([`stark_model::Srgb`]), so a P3 color is a
//! triple with a channel outside `[0, 1]` rather than a second kind of color.
//!
//! **The wheel is fitted to the gamut, and that is the whole of why it is a wheel.**
//! The picker used to draw a fixed square of the Oklab `a`/`b` plane, ±0.32 on each
//! axis, which is the box the *whole* gamut fits in — but one lightness is a thin
//! slice of that box, and the rest is colors sRGB cannot show, drawn clamped. At
//! `L = 0.61` the slice is 28% of the square; at `L = 0.2`, 3.8%. So most of the
//! picker was a flat wash answering every position with the same color, and
//! everything the artist was choosing between was crowded into what was left — which
//! is exactly the complaint, *it moves too fast and I cannot see what I have*. Here
//! the radius is chroma as a fraction of what this lightness and this hue can hold,
//! so the rim *is* the gamut boundary and every point inside it is a distinct color
//! the display can show.
//!
//! Everything here is arithmetic over `stark_model::color`, and every constant in it
//! was measured rather than picked — which is the reason it is in this crate rather
//! than in either frontend. A second copy of a bisection bracket, a gamut bridge and
//! a rim resolution would be a second answer to *which colors exist*, and the two
//! apps would draw different wheels for one document.
//!
//! What is not here is the picture's **carrier**: a `data:` URL for a
//! `background-image` on one side, a texture on the other. Both are built from the
//! same buffers below — `library::Thumbs`' split, applied to a wheel.

use std::f32::consts::TAU;

use stark_model::color::{
    Gamut, linear_p3_to_linear_srgb, linear_srgb_to_linear_p3, linear_to_srgb,
    oklab_to_linear_srgb, oklab_to_srgb, srgb_to_oklab,
};

/// The color a session starts on.
///
/// Here rather than in a panel because two things must agree about it: the picker's
/// fallback seed, and what a frontend's startup pushes into the engine. A panel
/// mounts before the engine exists, so if the two disagreed the picker would show a
/// color the brush does not have and the first stroke would come out black.
pub const INITIAL_COLOR: [f32; 3] = [0.61, 0.04, 0.02];

/// Rendered resolution of the wheel, in texels. The plane is low-frequency and the
/// picture is scaled up smoothly, so a small buffer is enough — and it is rebuilt on
/// every step of an `L` drag, which is what makes small matter.
pub const FIELD_N: usize = 96;

/// Rendered width of the `L` ramp — one texel tall, since the track carries no
/// vertical variation and is stretched down.
pub const RAMP_N: usize = 128;

/// How far past the achromatic axis a display gamut could possibly reach in Oklab
/// chroma — the outer bracket [`max_chroma`] searches in. sRGB's real maximum is
/// ≈0.323 and Display P3's ≈0.363, both saturated blues, so this is outside either at
/// *every* lightness, which is the only thing the search asks of it.
const CHROMA_CEILING: f32 = 0.45;

/// Halvings behind [`max_chroma`]. Twenty of them land within 5e-7 of the boundary,
/// three orders under the 1/255 the display quantizes to, and a fixed count keeps a
/// wheel a pure function of its lightness.
const CHROMA_STEPS: u32 = 20;

/// How many hues [`rim_table`] samples the boundary at. 512 is a step of 0.7°.
///
/// The rim is smooth in hue nearly everywhere, and one place it is not: **a slice at
/// constant lightness has corners, because the sRGB cube has them**. At `L = 0.45`
/// the rim runs 0.154 at −104° up to 0.240 at −96.5° and then turns hard, reaching
/// 0.313 — the blue primary itself — half a degree later, before easing back down.
/// The wheel wears that as a crease: a narrow wedge of deep blue with a hard edge,
/// visible in the picture and steep to drag across.
///
/// It is kept rather than smoothed away, because the two ways out are worse. Shaving
/// the corner to make the rim gentle puts `#0000ff` outside the wheel — a primary no
/// drag can reach. Rounding *outward* instead puts a band of clamped color back
/// inside the rim, which is the thing this picker exists to get rid of. And the
/// crease costs less than it looks: chroma per pixel is bounded by `rim / 124px`
/// everywhere except across that one ray, which is under the flat `0.64 / 248px` the
/// square field this replaced had *everywhere*.
const RIM_N: usize = 512;

/// How far outside a linear channel [`in_gamut`] still calls a color showable.
///
/// **A gamut slice at constant lightness is not star-shaped about the achromatic
/// axis**, which a search out from the centre has to be told. Oklab is a cube root
/// away from linear sRGB, so a straight line at constant `L` is a curve through the
/// sRGB cube, and along the hue of the blue primary that curve dips out of the cube
/// and back in: at `L = 0.452` the red channel falls to −0.0007 between chroma 0.266
/// and 0.313, then returns for the corner that *is* `#0000ff`. Bisection from the
/// centre stops at the first crossing, and the blue primary is stranded on the far
/// side of a gap 0.0006 wide — 11.6% short on chroma, with no drag able to reach it.
///
/// A thousandth of a channel bridges the gap: over a 26³ sample of sRGB, nothing is
/// unreachable at `0.001` and 44 colors are at `0` (two at `0.0005`). What it costs
/// is that a color on the rim may be held, by at most 4 codes in one channel and
/// only where that channel is already near zero — the deep blues. That is the whole
/// price, it is measured rather than hoped for, and [`hold_to`] is what collects it.
///
/// Measured over sRGB and spent on whichever gamut is asked for: the shape of the
/// problem is the cube's corners seen through a cube root, which every RGB gamut
/// has.
const GAMUT_BRIDGE: f32 = 0.001;

/// Whether Oklab `(l, a, b)` is a color `gamut` can show, give or take
/// [`GAMUT_BRIDGE`] on that gamut's own channels.
fn in_gamut(gamut: Gamut, l: f32, a: f32, b: f32) -> bool {
    gamut.contains(oklab_to_linear_srgb([l, a, b]), GAMUT_BRIDGE)
}

/// The color `lin` (linear sRGB) held inside `gamut`, as **extended sRGB**: clamped
/// in the gamut's own coordinates rather than in sRGB's, since those are the channels
/// the display actually has.
fn hold_to(gamut: Gamut, lin: [f32; 3]) -> [f32; 3] {
    let held = match gamut {
        Gamut::Srgb => lin.map(|c| c.clamp(0.0, 1.0)),
        Gamut::DisplayP3 => {
            linear_p3_to_linear_srgb(linear_srgb_to_linear_p3(lin).map(|c| c.clamp(0.0, 1.0)))
        }
    };
    held.map(linear_to_srgb)
}

/// The most chroma `gamut` holds at lightness `l` in the direction `hue` — its rim,
/// which is what the wheel's edge *is*.
///
/// By bisection rather than by Ottosson's analytic approximation of the same
/// boundary, for two reasons: the search only ever moves its lower bracket to a point
/// it has *tested*, so it answers with the gamut this build's own conversion has
/// rather than with a curve fitted to somebody else's — and the approximation is
/// fitted to sRGB alone, where this has to answer for any gamut a display offers. The
/// cost is nothing the picker can feel — a wheel asks this once per hue step
/// ([`rim_table`]), not once per pixel, and a pick asks it once.
pub fn max_chroma(gamut: Gamut, l: f32, hue: f32) -> f32 {
    let (sin, cos) = hue.sin_cos();
    let (mut lo, mut hi) = (0.0f32, CHROMA_CEILING);
    for _ in 0..CHROMA_STEPS {
        let mid = 0.5 * (lo + hi);
        if in_gamut(gamut, l, cos * mid, sin * mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// The whole rim at one lightness: [`RIM_N`] chromas around the circle.
///
/// Only the *drawing* goes through the table; every color the picker reports asks
/// [`max_chroma`] directly. That is the cheap half of a measured trade — a wheel
/// costs 0.8ms through the table and 2.9ms with a bisection per pixel, and the second
/// one is most of a frame, spent while the `L` slider is being dragged. What the
/// table costs back is that a drawn pixel and the color reported for it can disagree
/// by the interpolation error over one 0.7° step: nothing anywhere the rim is smooth,
/// and under a pixel's worth at the one place it is not (see [`RIM_N`]).
fn rim_table(gamut: Gamut, l: f32) -> Vec<f32> {
    (0..RIM_N)
        .map(|i| max_chroma(gamut, l, TAU * i as f32 / RIM_N as f32))
        .collect()
}

/// The rim between two of [`rim_table`]'s samples, linearly and wrapping.
fn rim_at(rim: &[f32], hue: f32) -> f32 {
    let t = hue.rem_euclid(TAU) / TAU * RIM_N as f32;
    let i = t.floor();
    let f = t - i;
    let i = (i as usize) % RIM_N;
    rim[i] * (1.0 - f) + rim[(i + 1) % RIM_N] * f
}

/// Where a straight-sRGB color sits on the wheel: its lightness, its hue, and how
/// much of the chroma available *at that lightness and that hue* it spends — `0` at
/// the centre, `1` on the rim.
///
/// `keep` is the hue to hold on to for a color that has none. A grey is every hue at
/// once, so a triple cannot say which one the artist was on, and the direction they
/// last chose is the only answer that does not spin the marker for them.
pub fn on_wheel(gamut: Gamut, rgb: [f32; 3], keep: f32) -> (f32, f32, f32) {
    let [l, a, b, _] = srgb_to_oklab([rgb[0], rgb[1], rgb[2], 1.0]);
    let c = (a * a + b * b).sqrt();
    if c <= 1e-6 {
        return (l, keep, 0.0);
    }
    let hue = b.atan2(a);
    let rim = max_chroma(gamut, l, hue);
    (l, hue, if rim > 1e-6 { (c / rim).min(1.0) } else { 0.0 })
}

/// The color a wheel position *is*, as extended sRGB — outside the cube where the
/// gamut is (§6.5).
///
/// The hold collects the two hairs the fit leaves: [`GAMUT_BRIDGE`], and the ULP at
/// the ends of the lightness axis, where the conversion matrices' rows sum to
/// 1.00000004 and white comes back a shade over. Neither is the fit failing — a
/// position with `sat ≤ 1` is a color the display has — and it is spent in the
/// *gamut's* channels, so holding a P3 color does not drag it back to sRGB.
pub fn wheel_color(gamut: Gamut, l: f32, hue: f32, sat: f32) -> [f32; 3] {
    let (sin, cos) = hue.sin_cos();
    let c = sat * max_chroma(gamut, l, hue);
    hold_to(gamut, oklab_to_linear_srgb([l, c * cos, c * sin]))
}

/// Where `(hue, sat)` sits in the wheel's box, as fractions of it — the marker's
/// place, and the point a fine drag moves *from*. `+a` runs right and `+b` up, the
/// orientation the flat picture of the same plane keeps ([`ab_field_rgb`]), so warm
/// sits at the top of both.
pub fn wheel_xy(hue: f32, sat: f32) -> (f32, f32) {
    let (sin, cos) = hue.sin_cos();
    (0.5 + 0.5 * sat * cos, 0.5 - 0.5 * sat * sin)
}

/// The wheel position a point in the control's box names — [`wheel_xy`] inverted,
/// with the saturation held inside the rim.
pub fn wheel_at(x: f32, y: f32) -> (f32, f32) {
    let (dx, dy) = (2.0 * x - 1.0, 1.0 - 2.0 * y);
    let r = (dx * dx + dy * dy).sqrt();
    (dy.atan2(dx), r.min(1.0))
}

/// How much of the pointer's travel a fine drag spends. A fifth: the whole width of
/// the control then covers a fifth of its range, which is the difference between
/// landing in *that red* and landing somewhere in the reds.
const FINE_GAIN: f32 = 0.2;

/// What a pointer sample on one of the picker's two controls means.
///
/// Decided once, on the press, and held for the whole gesture — the transform
/// widget's `Grab` rule, for its reason: a drag that changed its mind about what the
/// pointer meant halfway through would rewrite a value the hand was not on.
#[derive(Copy, Clone)]
pub enum Grab {
    /// The pointer *is* the value: where it lands is what is picked, so a press with
    /// no travel is already a complete pick.
    At,
    /// The value moves *with* the pointer at [`FINE_GAIN`], from where it already
    /// stood. The press picks nothing at all, which is the point — the hand gets the
    /// control's whole width to spend on a fraction of its range, and the color under
    /// the marker does not jump away before the adjustment starts.
    ///
    /// `from` is the pointer fraction the press landed at, `held` the marker's place
    /// at that moment; every later sample is `held` plus the geared-down travel.
    Fine { from: (f32, f32), held: (f32, f32) },
}

impl Grab {
    /// What a press at `at` means, given where the marker `held` and whether the
    /// modifier for a fine drag was down.
    pub fn take(at: (f32, f32), held: (f32, f32), fine: bool) -> Grab {
        if fine {
            Grab::Fine { from: at, held }
        } else {
            Grab::At
        }
    }

    /// Where in the control's box this sample points, as a fraction of it.
    pub fn place(self, p: (f32, f32)) -> (f32, f32) {
        match self {
            Grab::At => p,
            Grab::Fine { from, held } => (
                held.0 + (p.0 - from.0) * FINE_GAIN,
                held.1 + (p.1 - from.1) * FINE_GAIN,
            ),
        }
    }
}

/// What the readout shows and the field takes: `#rrggbb` for a color the sRGB cube
/// holds, and CSS Color 4's `color(display-p3 r g b)` for one it does not (§6.5).
///
/// **Two spellings because there are two facts to state**, not as a convenience. A
/// hex triple *is* the sRGB cube — there is no `#` code for a color outside it — so a
/// wide color printed as one would be a lie about what the brush holds, and the
/// nearest thing that is not a lie is the notation the platform already uses for it.
/// A caller that must have a hex code (an HTML color input) asks [`hex_of`].
pub fn notation_of(rgb: [f32; 3]) -> String {
    if rgb.iter().all(|c| (-1e-4..=1.0 + 1e-4).contains(c)) {
        return hex_of(rgb);
    }
    let p3 = stark_model::color::srgb_to_display_p3(rgb);
    format!("color(display-p3 {:.4} {:.4} {:.4})", p3[0], p3[1], p3[2])
}

/// A straight-sRGB color as `#rrggbb`, at the display's own precision — clamped, so
/// a wide color prints as the nearest sRGB one. [`notation_of`] is what a readout
/// wants; this is for a control whose value *is* a hex code.
pub fn hex_of(rgb: [f32; 3]) -> String {
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("#{:02x}{:02x}{:02x}", q(rgb[0]), q(rgb[1]), q(rgb[2]))
}

/// What [`notation_of`] prints, read back as extended sRGB: `#rgb`, `#rrggbb` or
/// either without the hash, and `color(srgb …)` / `color(display-p3 …)` with three
/// numbers. `None` for anything else — including a half-typed one, which is what a
/// field holds most of the time it is being used.
pub fn parse_color(s: &str) -> Option<[f32; 3]> {
    let s = s.trim();
    if let Some(inner) = s
        .strip_prefix("color(")
        .and_then(|rest| rest.strip_suffix(')'))
    {
        let mut parts = inner.split_ascii_whitespace();
        let space = parts.next()?;
        let mut n = [0.0f32; 3];
        for slot in &mut n {
            *slot = parts.next()?.parse().ok()?;
        }
        if parts.next().is_some() || !n.iter().all(|c| c.is_finite()) {
            return None;
        }
        return match space {
            "srgb" => Some(n),
            "display-p3" => Some(stark_model::color::display_p3_to_srgb(n)),
            _ => None,
        };
    }
    let s = s.strip_prefix('#').unwrap_or(s);
    if !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let byte = |i: usize| u8::from_str_radix(&s[i..i + 2], 16).ok();
    // `0x7` is `0x77`: a short code names the byte whose halves match, so the two
    // spellings of the same color agree rather than differing by a 17th.
    let nibble = |i: usize| u8::from_str_radix(&s[i..i + 1], 16).ok().map(|v| v * 17);
    let rgb = match s.len() {
        3 => [nibble(0)?, nibble(1)?, nibble(2)?],
        6 => [byte(0)?, byte(2)?, byte(4)?],
        _ => return None,
    };
    Some(rgb.map(|v| v as f32 / 255.0))
}

// --- the pictures ----------------------------------------------------------------
//
// Three buffers, one byte per channel, row-major from the top, **encoded in the
// gamut they were asked for** — sRGB bytes for `Gamut::Srgb`, Display P3 bytes for
// `Gamut::DisplayP3`, which is what a carrier tagged for that space takes. Neither an
// encoding container nor a texture: those are each frontend's, and the *numbers* are
// what must not differ — a wheel is a picture of which colors exist, and two answers
// to that would be two apps.

/// One texel of extended sRGB as `gamut`'s own encoded bytes, quantized the way both
/// carriers want it. A color inside the gamut lands inside `[0, 1]` there by
/// construction, so the clamp collects rounding and nothing else.
fn quantize(gamut: Gamut, rgb: [f32; 3]) -> [u8; 3] {
    let own = match gamut {
        Gamut::Srgb => rgb,
        Gamut::DisplayP3 => stark_model::color::srgb_to_display_p3(rgb),
    };
    own.map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
}

/// The picker's wheel at lightness `l`, [`FIELD_N`] square: hue by direction, chroma
/// by distance as a fraction of [`max_chroma`] in that direction, so the unit circle
/// is the sRGB boundary and every pixel inside it is a color the display can show.
///
/// Square, and the corners past the rim carry the rim's own color: the control is
/// clipped to a circle, so those texels are never seen — but they are what a scaler
/// mixes into the edge ones, and a corner left black would draw a dark fringe all the
/// way round.
pub fn wheel_rgb(gamut: Gamut, l: f32) -> Vec<u8> {
    let rim = rim_table(gamut, l);
    let last = (FIELD_N - 1) as f32;
    let mut out = Vec::with_capacity(FIELD_N * FIELD_N * 3);
    for y in 0..FIELD_N {
        for x in 0..FIELD_N {
            let dx = 2.0 * x as f32 / last - 1.0;
            let dy = 1.0 - 2.0 * y as f32 / last;
            let r = (dx * dx + dy * dy).sqrt();
            let (ux, uy) = if r > 1e-6 {
                (dx / r, dy / r)
            } else {
                (1.0, 0.0)
            };
            let c = rim_at(&rim, dy.atan2(dx)) * r.min(1.0);
            let rgba = oklab_to_srgb([l, c * ux, c * uy, 1.0]);
            out.extend_from_slice(&quantize(gamut, [rgba[0], rgba[1], rgba[2]]));
        }
    }
    out
}

/// The `L` slider's track, [`RAMP_N`] wide and one tall: this hue at this fraction of
/// the chroma each lightness can hold, black at the left to white at the right.
///
/// Drawn rather than handed to a gradient. A gradient interpolates `(a, b)` linearly,
/// so with a saturated color it leaves the gamut immediately and both ends of the
/// track go flat under the clamp — the slider stops answering exactly where the
/// artist is looking for a highlight or a shadow. Fitting each column to its own
/// lightness is the same fix the wheel makes, on the other axis.
pub fn ramp_rgb(gamut: Gamut, hue: f32, sat: f32) -> Vec<u8> {
    let last = (RAMP_N - 1) as f32;
    (0..RAMP_N)
        .flat_map(|x| quantize(gamut, wheel_color(gamut, x as f32 / last, hue, sat)))
        .collect()
}

/// The Oklab `a`/`b` plane at lightness `l`, flat: `a` runs left→right (−`ab`→+`ab`),
/// `b` runs bottom→top, so warm colors sit at the top. Out-of-gamut colors clamp.
///
/// This is the *other* picture of the plane, and it is deliberately not the picker's.
/// A filter's chroma dial draws an affine map of the `(a, b)` plane — a circle of
/// reference hues and where the filter sends them (§21.5) — so its background has to
/// be that plane, flat and undistorted, or the handle would sit over a picture it
/// disagrees with. The picker's wheel is fitted to the gamut and so is not flat; the
/// two still agree about orientation, which is the part a reader carries between them.
pub fn ab_field_rgb(l: f32, ab: f32) -> Vec<u8> {
    let last = (FIELD_N - 1) as f32;
    let mut out = Vec::with_capacity(FIELD_N * FIELD_N * 3);
    for y in 0..FIELD_N {
        for x in 0..FIELD_N {
            let aa = ab * (2.0 * x as f32 / last - 1.0);
            let bb = ab * (1.0 - 2.0 * y as f32 / last);
            let rgba = oklab_to_srgb([l, aa, bb, 1.0]);
            // sRGB whatever the display's gamut: this is the *filter dial's*
            // background, a flat map of the `(a, b)` plane the handle is placed
            // against (§21.5), not a picture of which colors exist.
            out.extend_from_slice(&quantize(Gamut::Srgb, [rgba[0], rgba[1], rgba[2]]));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The point of the fit: every position the wheel can hold is a color the display
    /// shows as asked. Stated as what the artist would see — how far the clamp in
    /// [`wheel_color`] has to move the color — rather than as a bound on the linear
    /// channels, because that is the number [`GAMUT_BRIDGE`] is spending.
    #[test]
    fn no_wheel_position_needs_more_than_a_hair_of_clamping() {
        for gamut in [Gamut::Srgb, Gamut::DisplayP3] {
            let mut worst = 0u8;
            for li in 0..=20 {
                let l = li as f32 / 20.0;
                for hi in 0..72 {
                    let hue = TAU * hi as f32 / 72.0;
                    for si in 0..=10 {
                        let sat = si as f32 / 10.0;
                        let c = sat * max_chroma(gamut, l, hue);
                        let (sin, cos) = hue.sin_cos();
                        // In the *gamut's* encoded channels, which is where the
                        // bridge is spent: extended sRGB exaggerates it, since the
                        // curve is steep near zero and a wide color sits there.
                        let r = oklab_to_srgb([l, c * cos, c * sin, 1.0]);
                        let raw = quantize(gamut, [r[0], r[1], r[2]]);
                        let fitted = quantize(gamut, wheel_color(gamut, l, hue, sat));
                        for i in 0..3 {
                            worst = worst.max(raw[i].abs_diff(fitted[i]));
                        }
                    }
                }
            }
            // 4 codes is what GAMUT_BRIDGE is measured to cost, and only in a
            // channel already near zero.
            assert!(
                worst <= 4,
                "{gamut:?}: the hold moved a color by {worst} codes"
            );
        }
    }

    /// A color goes onto the wheel and comes back the same color — within a
    /// quantization step, which is all the readout that reports it can carry.
    #[test]
    fn a_color_survives_the_wheel() {
        for rgb in [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.61, 0.04, 0.02],
            [0.2, 0.45, 0.7],
        ] {
            for gamut in [Gamut::Srgb, Gamut::DisplayP3] {
                let (l, hue, sat) = on_wheel(gamut, rgb, 0.0);
                let back = wheel_color(gamut, l, hue, sat);
                assert!(
                    rgb.iter()
                        .zip(back)
                        .all(|(a, b)| (a - b).abs() < 1.0 / 255.0),
                    "{gamut:?}: {rgb:?} → (l {l}, hue {hue}, sat {sat}) → {back:?}"
                );
            }
        }
    }

    /// The most saturated colors sRGB has are *on* the rim, rather than stranded
    /// outside it where no drag reaches them. Within a percent, which is the width of
    /// the gap [`GAMUT_BRIDGE`] crosses — and a rim position renders as the primary
    /// either way, which the round trip above is what actually checks.
    #[test]
    fn the_primaries_reach_the_rim() {
        for rgb in [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ] {
            let (_, _, sat) = on_wheel(Gamut::Srgb, rgb, 0.0);
            assert!(sat > 0.99, "{rgb:?} sat {sat}");
            // …and no further out in a wider one, since Display P3 contains sRGB.
            let (_, _, wide) = on_wheel(Gamut::DisplayP3, rgb, 0.0);
            assert!(
                wide <= sat + 1e-3,
                "{rgb:?} spends {wide} of Display P3's chroma and {sat} of sRGB's"
            );
            // How much room P3 leaves past an sRGB primary is the primary's own
            // business: it widens the red and green corners and shares the blue
            // one exactly (x 0.150, y 0.060), so red and green have somewhere to
            // go and blue has none.
            if rgb == [1.0, 0.0, 0.0] || rgb == [0.0, 1.0, 0.0] {
                assert!(wide < 0.95, "{rgb:?} gained only {} chroma", sat - wide);
            }
            if rgb == [0.0, 0.0, 1.0] {
                assert!(wide > 0.99, "P3 moved the blue primary: {wide}");
            }
        }
    }

    /// Display P3's own primaries reach *its* rim, and are outside the sRGB cube —
    /// which is the whole of what a wide gamut buys the picker (§6.5).
    #[test]
    fn the_wide_primaries_reach_the_wide_rim() {
        for p3 in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
            let rgb = linear_p3_to_linear_srgb(p3).map(linear_to_srgb);
            let (_, _, sat) = on_wheel(Gamut::DisplayP3, rgb, 0.0);
            assert!(sat > 0.99, "P3 {p3:?} spends only {sat} of its own chroma");
            assert!(
                rgb.iter().any(|c| !(-0.01..=1.01).contains(c)),
                "P3 {p3:?} is {rgb:?}, which the sRGB cube could hold"
            );
        }
    }

    /// A wheel position outside sRGB draws as the gamut's own bytes: the picture is
    /// tagged for that space, so a P3 texel that clips in sRGB does not clip here.
    #[test]
    fn a_wide_wheel_draws_in_its_own_space() {
        let l = 0.7;
        let srgb = wheel_rgb(Gamut::Srgb, l);
        let p3 = wheel_rgb(Gamut::DisplayP3, l);
        assert_eq!(srgb.len(), p3.len());
        // The centre is achromatic and lands on the same byte in both; the rim is
        // where the two gamuts differ, and it must differ.
        let mid = ((FIELD_N / 2) * FIELD_N + FIELD_N / 2) * 3;
        assert!(
            srgb[mid].abs_diff(p3[mid]) <= 1,
            "the achromatic centre moved: {} vs {}",
            srgb[mid],
            p3[mid]
        );
        assert!(
            srgb.as_chunks::<3>()
                .0
                .iter()
                .zip(p3.as_chunks::<3>().0)
                .any(|(a, b)| a != b),
            "the wide wheel is the sRGB one",
        );
    }

    /// The rim table wraps rather than falling off its end.
    #[test]
    fn the_rim_wraps() {
        let rim = rim_table(Gamut::Srgb, 0.6);
        for hue in [-TAU, -0.3, 0.0, 0.3, TAU, TAU + 0.3] {
            let exact = max_chroma(Gamut::Srgb, 0.6, hue);
            let table = rim_at(&rim, hue);
            assert!(
                (table - exact).abs() < 1e-3,
                "hue {hue}: table {table} vs exact {exact}"
            );
        }
    }

    /// A grey keeps the hue it was given rather than spinning to zero: sRGB cannot
    /// say what hue a grey is, and the last one chosen is the only answer that does
    /// not move a marker under the hand.
    #[test]
    fn a_grey_keeps_the_hue_it_was_holding() {
        let (_, hue, sat) = on_wheel(Gamut::Srgb, [0.5, 0.5, 0.5], 1.234);
        assert_eq!(hue, 1.234);
        assert_eq!(sat, 0.0);
    }

    /// The readout says which space it is speaking, so a wide color is never printed
    /// as the hex code of a different one — and what it prints reads back.
    #[test]
    fn a_wide_color_says_so_and_reads_back() {
        assert_eq!(notation_of([0.25, 0.5, 0.75]), "#4080bf");
        assert_eq!(
            parse_color("#4080bf"),
            Some([0x40, 0x80, 0xbf].map(|v| v as f32 / 255.0))
        );

        let p3_green = linear_p3_to_linear_srgb([0.0, 1.0, 0.0]).map(linear_to_srgb);
        let said = notation_of(p3_green);
        assert!(said.starts_with("color(display-p3 "), "{said}");
        let back = parse_color(&said).expect("what it prints, it reads");
        assert!(
            back.iter().zip(p3_green).all(|(a, b)| (a - b).abs() < 1e-3),
            "{said} → {back:?}, and should be {p3_green:?}"
        );

        // The other spelling of the same thing, and the ones that are not colors.
        assert_eq!(parse_color("color(srgb 0 0.5 1)"), Some([0.0, 0.5, 1.0]));
        for bad in [
            "color(display-p3 1 0)",
            "color(display-p3 1 0 0 0)",
            "color(rec2020 1 0 0)",
            "color(display-p3 1 0 nan)",
            "#4080b",
            "",
        ] {
            assert_eq!(parse_color(bad), None, "{bad}");
        }
    }

    /// The marker's place and the position it is read back from are inverses.
    #[test]
    fn a_wheel_position_and_its_marker_are_inverses() {
        for (hue, sat) in [(0.0, 1.0), (1.5, 0.5), (-2.0, 0.25)] {
            let (x, y) = wheel_xy(hue, sat);
            let (h2, s2) = wheel_at(x, y);
            assert!((s2 - sat).abs() < 1e-5, "{sat} came back as {s2}");
            if sat > 1e-3 {
                let d = (h2 - hue).rem_euclid(TAU);
                assert!(d < 1e-4 || (TAU - d) < 1e-4, "{hue} came back as {h2}");
            }
        }
    }

    /// A fine drag spends a fifth of the pointer's travel and picks nothing on the
    /// press itself, which is what keeps the color from jumping before an adjustment.
    #[test]
    fn a_fine_drag_is_geared_down_from_where_it_started() {
        let grab = Grab::take((0.5, 0.5), (0.2, 0.8), true);
        assert_eq!(grab.place((0.5, 0.5)), (0.2, 0.8));
        let (x, _) = grab.place((1.0, 0.5));
        assert!((x - (0.2 + 0.5 * FINE_GAIN)).abs() < 1e-6);
        // An ordinary press is the value where it landed, with no memory at all.
        assert_eq!(
            Grab::take((0.5, 0.5), (0.2, 0.8), false).place((0.9, 0.1)),
            (0.9, 0.1)
        );
    }

    /// Both spellings of a hex code, and the short one naming the byte whose halves
    /// match — so `#f00` and `#ff0000` are one color rather than two a 17th apart.
    #[test]
    fn a_hex_code_reads_both_ways_round() {
        for (text, rgb) in [
            ("#000000", [0.0, 0.0, 0.0]),
            ("#ffffff", [1.0, 1.0, 1.0]),
            ("#ff0000", [1.0, 0.0, 0.0]),
        ] {
            assert_eq!(parse_color(text), Some(rgb));
            assert_eq!(hex_of(rgb), text);
        }
        // A short code is the byte whose halves match, and the hash is optional.
        assert_eq!(parse_color("#abc"), parse_color("aabbcc"));
        assert_eq!(parse_color(" #ABC "), parse_color("#abc"));
        for bad in ["", "#", "#12", "#12345", "#gggggg", "12345678"] {
            assert_eq!(parse_color(bad), None, "{bad:?}");
        }
    }

    /// Each picture is the size it claims, so a carrier can build a texture from it
    /// without asking.
    #[test]
    fn every_picture_is_the_size_it_says() {
        for gamut in [Gamut::Srgb, Gamut::DisplayP3] {
            assert_eq!(wheel_rgb(gamut, 0.5).len(), FIELD_N * FIELD_N * 3);
            assert_eq!(ramp_rgb(gamut, 0.0, 1.0).len(), RAMP_N * 3);
        }
        assert_eq!(ab_field_rgb(0.5, 0.32).len(), FIELD_N * FIELD_N * 3);
    }
}
