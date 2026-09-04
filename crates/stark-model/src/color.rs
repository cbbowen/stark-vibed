//! Oklab working color space (§6.5).
//!
//! Color enters as straight sRGB (the picker / image space), is stored and
//! blended in **Oklab** for perceptually uniform mixing, and is converted back
//! to display only in the media pass. These conversions are fixed constants
//! shared with the WGSL side (`shaders/lib/color.wesl`), so ingest and present are
//! reproducible across runs and peers — required by golden tests (§9) and
//! convergence (§12).
//!
//! Oklab transform after Björn Ottosson.
//!
//! One further piece of the same shader library lives here for the same reason: the
//! light space `linear_to_light`/`light_to_linear` state (§18.0.4). The dispersion
//! spectrum that used to sit beside it is in `stark-engine`'s `dispersion`, because
//! it reads the shader mirror (§6.10) and this crate compiles without the shaders.

use std::ops::Deref;

use serde::{Deserialize, Serialize};

/// A straight (un-premultiplied) **extended sRGB** color: the sRGB primaries and
/// transfer, continued past the cube — CSS Color 4's `srgb`, in which a value outside
/// `[0, 1]` names a color outside the sRGB gamut (§6.5). The CPU boundary convention
/// as a type rather than as a promise.
///
/// Every color the *document* carries is one of these: the substrate a painting sits
/// on (§15.5), a matte's paint, a fill's parcel, a gradient stop.
///
/// The only way to build one funnels ([`new`](Self::new)): every channel finite and
/// within [`EXTENT`](Self::EXTENT) of zero. `Deserialize` runs the same funnel, so a
/// `NaN` or an unbounded value cannot arrive from a file or a peer either — §1's
/// preference for ruling out a class over enumerating its instances. **Not the cube:**
/// it was, until wide-gamut paint (§6.5). A build from before reads the same bytes
/// and clamps them — the same log, a narrower picture — which is why the widening
/// bumped the wire (`stark-net::wire`).
///
/// # Why it derefs
///
/// Reading goes straight through to the array, so `c[0]`, `c.iter()` and `c.map(..)`
/// keep working and the sites that only *read* a color did not have to change. There
/// is no `DerefMut` and the field is private, so the projection is out and never in —
/// the constructor stays the only door.
///
/// # What is deliberately not one
///
/// [`PaintEffect::color`](crate::document::PaintEffect::color) stays a bare
/// `[f32; 3]`. The frontend writes it a component at a time — a channel slider
/// assigns `color[1]` — so a wrapper there would need setters that re-clamp, which is
/// the *other* design (a value you may mutate carefully) rather than this one (a value
/// that cannot be built wrong). A brush has a funnel of its own in
/// `BrushParams::sanitized`.
///
/// # The wire
///
/// `[f32; 3]`, in both directions and under the same field names, so this is not a
/// format change — a document written before the type existed reads back into it,
/// funnelled on the way (§8).
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize, Deserialize, carbonite::Schema)]
#[serde(from = "[f32; 3]", into = "[f32; 3]")]
#[carbonite(as = "[f32; 3]")]
pub struct Srgb([f32; 3]);

impl Srgb {
    pub const BLACK: Self = Self([0.0; 3]);
    pub const WHITE: Self = Self([1.0; 3]);

    /// How far from zero a channel may go. Past every display gamut — Rec.2020's
    /// primaries sit within ±2 in extended sRGB — and small enough that a half-float
    /// tile cannot overflow through any pass.
    pub const EXTENT: f32 = 4.0;

    /// The color `c`, funnelled — the one door, and it cannot fail.
    ///
    /// `const`, so a palette or a default can be written as one.
    pub const fn new(c: [f32; 3]) -> Self {
        Self([bound(c[0]), bound(c[1]), bound(c[2])])
    }

    /// The components, for a caller that needs the array by value.
    pub const fn get(self) -> [f32; 3] {
        self.0
    }
}

impl From<[f32; 3]> for Srgb {
    fn from(c: [f32; 3]) -> Self {
        Self::new(c)
    }
}

impl From<Srgb> for [f32; 3] {
    fn from(c: Srgb) -> Self {
        c.0
    }
}

impl Deref for Srgb {
    type Target = [f32; 3];

    fn deref(&self) -> &[f32; 3] {
        &self.0
    }
}

/// `x` held to `[-EXTENT, EXTENT]`, with `NaN` landing on 0 — [`Srgb`]'s funnel.
/// `is_nan` first because a symmetric bound has no end for `max`/`min` to carry a
/// `NaN` to that means anything.
const fn bound(x: f32) -> f32 {
    if x.is_nan() {
        0.0
    } else {
        x.max(-Srgb::EXTENT).min(Srgb::EXTENT)
    }
}

/// sRGB transfer function, decoded: gamma-encoded component → linear. Odd and
/// unbounded — mirrored through 0 and continued past 1 — so it is defined on every
/// extended value (§6.5); on `[0, 1]` it is the sRGB curve exactly.
pub fn srgb_to_linear(c: f32) -> f32 {
    let a = c.abs();
    let lin = if a <= 0.04045 {
        a / 12.92
    } else {
        ((a + 0.055) / 1.055).powf(2.4)
    };
    lin.copysign(c)
}

/// The inverse: linear component → gamma-encoded, odd and unbounded like
/// [`srgb_to_linear`].
pub fn linear_to_srgb(c: f32) -> f32 {
    let a = c.abs();
    let enc = if a <= 0.003_130_8 {
        12.92 * a
    } else {
        1.055 * a.powf(1.0 / 2.4) - 0.055
    };
    enc.copysign(c)
}

/// Linear sRGB → linear Display P3, both D65 — CSS Color 4's matrix. Mirrored in
/// `lib/display.wesl`.
pub fn linear_srgb_to_linear_p3(c: [f32; 3]) -> [f32; 3] {
    let [r, g, b] = c;
    [
        0.822_462_1 * r + 0.177_538 * g,
        0.033_194_1 * r + 0.966_805_9 * g,
        0.017_082_7 * r + 0.072_397_4 * g + 0.910_519_9 * b,
    ]
}

/// The exact inverse of [`linear_srgb_to_linear_p3`].
pub fn linear_p3_to_linear_srgb(c: [f32; 3]) -> [f32; 3] {
    let [r, g, b] = c;
    [
        1.224_940_1 * r - 0.224_940_4 * g,
        -0.042_056_9 * r + 1.042_057_1 * g,
        -0.019_637_6 * r - 0.078_636_1 * g + 1.098_273_5 * b,
    ]
}

/// Extended sRGB, encoded → Display P3, encoded (the sRGB curve over P3 primaries):
/// what a `display-p3` canvas stores, and a Display P3 color's `[0, 1]`.
pub fn srgb_to_display_p3(c: [f32; 3]) -> [f32; 3] {
    linear_srgb_to_linear_p3(c.map(srgb_to_linear)).map(linear_to_srgb)
}

/// The inverse of [`srgb_to_display_p3`].
pub fn display_p3_to_srgb(c: [f32; 3]) -> [f32; 3] {
    linear_p3_to_linear_srgb(c.map(srgb_to_linear)).map(linear_to_srgb)
}

/// Which colors a display can show — the coarse buckets a surface's transfer
/// implies (§6.5). What a picker fits its wheel to.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Gamut {
    /// The sRGB cube.
    #[default]
    Srgb,
    /// Display P3's cube: about a quarter more colors, the greens and reds most of
    /// them.
    DisplayP3,
}

impl Gamut {
    /// Whether linear sRGB `lin` is inside this gamut, give or take `slack` on each
    /// of the gamut's own channels.
    pub fn contains(self, lin: [f32; 3], slack: f32) -> bool {
        let own = match self {
            Self::Srgb => lin,
            Self::DisplayP3 => linear_srgb_to_linear_p3(lin),
        };
        own.iter().all(|c| (-slack..=1.0 + slack).contains(c))
    }
}

/// Linear sRGB `[r,g,b]` → Oklab `[L,a,b]`.
pub fn linear_srgb_to_oklab(rgb: [f32; 3]) -> [f32; 3] {
    let [r, g, b] = rgb;
    let l = 0.412_221_46 * r + 0.536_332_55 * g + 0.051_445_995 * b;
    let m = 0.211_903_5 * r + 0.680_699_5 * g + 0.107_396_96 * b;
    let s = 0.088_302_46 * r + 0.281_718_85 * g + 0.629_978_7 * b;

    let l_ = l.cbrt();
    let m_ = m.cbrt();
    let s_ = s.cbrt();

    [
        0.210_454_26 * l_ + 0.793_617_8 * m_ - 0.004_072_047 * s_,
        1.977_998_5 * l_ - 2.428_592_2 * m_ + 0.450_593_7 * s_,
        0.025_904_037 * l_ + 0.782_771_77 * m_ - 0.808_675_77 * s_,
    ]
}

/// Oklab `[L,a,b]` → linear sRGB `[r,g,b]`.
pub fn oklab_to_linear_srgb(lab: [f32; 3]) -> [f32; 3] {
    let [ll, aa, bb] = lab;
    let l_ = ll + 0.396_337_78 * aa + 0.215_803_76 * bb;
    let m_ = ll - 0.105_561_346 * aa - 0.063_854_17 * bb;
    let s_ = ll - 0.089_484_18 * aa - 1.291_485_5 * bb;

    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;

    [
        4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s,
        -1.268_438 * l + 2.609_757_4 * m - 0.341_319_38 * s,
        -0.004_196_086_3 * l - 0.703_418_6 * m + 1.707_614_7 * s,
    ]
}

/// Straight sRGB RGBA in `[0,1]` → Oklab `[L,a,b]` + unchanged alpha.
pub fn srgb_to_oklab(rgba: [f32; 4]) -> [f32; 4] {
    let lin = [
        srgb_to_linear(rgba[0]),
        srgb_to_linear(rgba[1]),
        srgb_to_linear(rgba[2]),
    ];
    let lab = linear_srgb_to_oklab(lin);
    [lab[0], lab[1], lab[2], rgba[3]]
}

/// Oklab `[L,a,b]` + alpha → straight sRGB RGBA in `[0,1]`.
pub fn oklab_to_srgb(laba: [f32; 4]) -> [f32; 4] {
    let lin = oklab_to_linear_srgb([laba[0], laba[1], laba[2]]);
    [
        linear_to_srgb(lin[0]),
        linear_to_srgb(lin[1]),
        linear_to_srgb(lin[2]),
        laba[3],
    ]
}

/// Linear sRGB → XYZ normalized to D65 white — the space light is combined and
/// scaled in (see `lib/color.wesl`, which holds the argument). Each row of the sRGB
/// primaries divided by its own sum, so `(1,1,1)` maps to `(1,1,1)`.
pub fn linear_to_light(c: [f32; 3]) -> [f32; 3] {
    let [r, g, b] = c;
    [
        0.433_939_5 * r + 0.376_207_8 * g + 0.189_852_8 * b,
        0.212_672_9 * r + 0.715_152_2 * g + 0.072_175 * b,
        0.017_756_6 * r + 0.109_467_1 * g + 0.872_776_3 * b,
    ]
}

/// The exact inverse of [`linear_to_light`].
pub fn light_to_linear(c: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = c;
    [
        3.079_954_5 * x - 1.537_138_5 * y - 0.542_816 * z,
        -0.921_258_3 * x + 1.876_010_8 * y + 0.045_247_7 * z,
        0.052_887_3 * x - 0.204_025_9 * y + 1.151_138_5 * z,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A color that is not finite and bounded cannot arrive from a file or a
    /// peer**, which is the half of [`Srgb`]'s claim that a constructor alone does
    /// not make.
    ///
    /// Asked of the *bytes*, because that is the only place it can still be asked: a
    /// hostile `Parcel::Solid` or `GradientStop` is no longer a value anyone can
    /// build. What a document carries is `[f32; 3]`, and this is that column decoded.
    ///
    /// The `NaN` channel is the one that matters — it is the value `f32::clamp` would
    /// pass through to a shader as a NaN texel. A wide-gamut value passes as it is:
    /// the cube is no longer the bound (§6.5).
    #[test]
    fn a_color_from_the_wire_is_finite_and_bounded() {
        let wire = |c: [f32; 3]| {
            carbonite::from_slice_static::<Srgb>(&carbonite::to_vec_static(&c).expect("encodes"))
                .expect("decodes")
        };
        assert_eq!(wire([-1.0, 2.0, f32::NAN]).get(), [-1.0, 2.0, 0.0]);
        assert_eq!(
            wire([f32::INFINITY, f32::NEG_INFINITY, 1e30]).get(),
            [Srgb::EXTENT, -Srgb::EXTENT, Srgb::EXTENT]
        );

        // …and an ordinary color comes through bit for bit, which is what keeps this
        // from being a format change: a document written before the type existed
        // reads back into it unchanged (§8).
        let ordinary = [0.25, 0.5, 0.75];
        assert_eq!(wire(ordinary).get(), ordinary);
        assert_eq!(
            carbonite::to_vec_static(&Srgb::new(ordinary)).expect("encodes"),
            carbonite::to_vec_static(&ordinary).expect("encodes"),
            "the wire shape is the bare array, in both directions",
        );
    }

    /// The projection is **out and never in**: reading goes through to the array, and
    /// there is no way back that skips the constructor.
    #[test]
    fn the_only_door_is_the_constructor() {
        let c = Srgb::new([0.25, 0.5, 0.75]);
        // Deref gives the array's own API to a reader.
        assert_eq!(c[0], 0.25);
        assert_eq!(c.iter().copied().sum::<f32>(), 1.5);
        assert_eq!(c.map(|x| x * 2.0), [0.5, 1.0, 1.5]);
        assert_eq!(c.get(), [0.25, 0.5, 0.75]);
        // `From` in both directions, and the way in funnels like every other door:
        // a wide color passes, an unbounded one is held.
        assert_eq!(Srgb::from([2.0, -1.0, 0.5]).get(), [2.0, -1.0, 0.5]);
        assert_eq!(<[f32; 3]>::from(c), [0.25, 0.5, 0.75]);
        // The named constants are the corners they say they are.
        assert_eq!(Srgb::BLACK.get(), [0.0; 3]);
        assert_eq!(Srgb::WHITE.get(), [1.0; 3]);
        // `const`, so a default or a palette entry can be one.
        const PICKED: Srgb = Srgb::new([9.0, 0.5, -3.0]);
        assert_eq!(PICKED.get(), [Srgb::EXTENT, 0.5, -3.0]);
    }

    fn close(a: [f32; 4], b: [f32; 4], eps: f32) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() <= eps)
    }

    /// The transfer is the sRGB curve on `[0, 1]`, bit for bit — every golden was
    /// blessed through it — and odd past it, so an extended value round-trips.
    #[test]
    fn the_transfer_is_srgb_inside_and_odd_outside() {
        for c in [0.0, 0.001, 0.04045, 0.2, 0.5, 1.0] {
            let want = if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055f32) / 1.055).powf(2.4)
            };
            assert_eq!(srgb_to_linear(c), want);
        }
        for c in [-1.5, -0.3, -0.01, 0.7, 1.4, 3.0] {
            assert_eq!(srgb_to_linear(-c), -srgb_to_linear(c));
            assert!((linear_to_srgb(srgb_to_linear(c)) - c).abs() < 1e-5, "{c}");
        }
    }

    /// Display P3's primaries, as extended sRGB, lie outside the cube and inside the
    /// P3 gamut — and the two matrices are inverses.
    #[test]
    fn display_p3_is_wider_than_the_cube() {
        for (p3, name) in [
            ([1.0, 0.0, 0.0], "red"),
            ([0.0, 1.0, 0.0], "green"),
            ([0.0, 0.0, 1.0], "blue"),
        ] {
            let lin = linear_p3_to_linear_srgb(p3);
            assert!(
                !Gamut::Srgb.contains(lin, 1e-4) && Gamut::DisplayP3.contains(lin, 1e-4),
                "P3 {name} in linear sRGB: {lin:?}"
            );
            let back = linear_srgb_to_linear_p3(lin);
            assert!(
                back.iter().zip(p3).all(|(a, b)| (a - b).abs() < 1e-5),
                "{name}: {back:?}"
            );
            // Encoded and back, through the odd transfer.
            let enc = srgb_to_display_p3(lin.map(linear_to_srgb));
            assert!(
                enc.iter().zip(p3).all(|(a, b)| (a - b).abs() < 1e-4),
                "{name} encoded: {enc:?}"
            );
        }
        // White is white in both.
        let w = linear_srgb_to_linear_p3([1.0; 3]);
        assert!(w.iter().all(|c| (c - 1.0).abs() < 1e-5), "{w:?}");
    }

    #[test]
    fn light_roundtrip() {
        for c in [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.2, 0.5, 0.8]] {
            let back = light_to_linear(linear_to_light(c));
            assert!(
                c.iter().zip(back).all(|(x, y)| (x - y).abs() <= 1e-4),
                "roundtrip {c:?} -> {back:?}"
            );
        }
    }

    #[test]
    fn srgb_oklab_roundtrip() {
        for c in [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.2, 0.5, 0.8, 0.5],
        ] {
            let back = oklab_to_srgb(srgb_to_oklab(c));
            assert!(close(c, back, 1e-3), "roundtrip {c:?} -> {back:?}");
        }
    }

    #[test]
    fn gray_has_no_chroma() {
        let lab = srgb_to_oklab([0.5, 0.5, 0.5, 1.0]);
        assert!(
            lab[1].abs() < 1e-3 && lab[2].abs() < 1e-3,
            "gray a,b ~ 0: {lab:?}"
        );
    }
}
