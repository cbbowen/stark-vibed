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

use crate::clamp01;

/// A straight (un-premultiplied) sRGB color, components in `[0, 1]` — **the CPU
/// boundary convention (§6.5) as a type rather than as a promise.**
///
/// Every color the *document* carries is one of these: the substrate a painting sits
/// on (§15.5), a matte's paint, a fill's parcel, a gradient stop. They had four
/// separate clamps between them, one per funnel — and the funnels were where three of
/// the four were missing. `SetSubstrateColor` and `MattePaint` had none at all and sat
/// under a comment saying there was nothing to hold; the ramp's was spelled
/// `f32::clamp`, which returns the `NaN` it exists to catch.
///
/// There is nothing left to remember. The only way to build one clamps
/// ([`new`](Self::new)), and that is what `Deserialize` runs too, so a color outside
/// the cube cannot arrive from a file or a peer either. No `sanitized` in the crate
/// mentions color any more, which is the whole point: §1 prefers ruling out a class
/// to enumerating its instances, and a color out of range is the class.
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
/// [`BrushParams::color`](crate::document::BrushParams::color) is RGBA and stays a
/// bare `[f32; 4]`. Not an oversight and not only the extra channel: the frontend
/// writes it a component at a time — an opacity slider assigns `color[3]`, a picker
/// copies into `color[..3]` — so a wrapper there would need setters that re-clamp,
/// which is the *other* design (a value you may mutate carefully) rather than this
/// one (a value that cannot be built wrong). And a brush already has a funnel of its
/// own in `BrushParams::sanitized`, which is exactly what the four sites here did
/// not.
///
/// # The wire
///
/// `[f32; 3]`, in both directions and under the same field names, so this is not a
/// format change — a document written before the type existed reads back into it,
/// clamped on the way (§8). The `serde(from/into)` pair and `carbonite(as)` state
/// that once, the device `FillOp` and `SelectionOp` already use.
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize, Deserialize, carbonite::Schema)]
#[serde(from = "[f32; 3]", into = "[f32; 3]")]
#[carbonite(as = "[f32; 3]")]
pub struct Srgb([f32; 3]);

impl Srgb {
    pub const BLACK: Self = Self([0.0; 3]);
    pub const WHITE: Self = Self([1.0; 3]);

    /// The color `c`, held to the cube — the one door, and it cannot fail.
    ///
    /// `const`, so a palette or a default can be written as one. That rests on
    /// [`clamp01`](crate::clamp01) being const, which rests on `f32::max`/`min`
    /// being const on this toolchain — the crate is on nightly for a different
    /// reason (§CLAUDE.md) and this is not a second one: written with comparisons
    /// instead it is the same function, and the `max`-then-`min` spelling is kept
    /// because it is where the NaN policy is stated.
    pub const fn new(c: [f32; 3]) -> Self {
        Self([clamp01(c[0]), clamp01(c[1]), clamp01(c[2])])
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

/// sRGB transfer function: gamma-encoded component in `[0,1]` → linear.
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Inverse sRGB transfer function: linear component → gamma-encoded.
pub fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
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

    /// **A color outside the cube cannot arrive from a file or a peer**, which is the
    /// half of [`Srgb`]'s claim that a constructor alone does not make.
    ///
    /// Asked of the *bytes*, because that is the only place it can still be asked: a
    /// hostile `Parcel::Solid` or `GradientStop` is no longer a value anyone can
    /// build, so the three tests that used to check this at their own funnels have
    /// nothing left to construct. What a document actually carries is `[f32; 3]`, and
    /// this is that column decoded.
    ///
    /// The `NaN` channel is the one that matters. `f32::clamp` returns it — both of
    /// its comparisons against a NaN are false — so every earlier spelling of this
    /// bound caught the out-of-range values and passed the one that reaches a shader
    /// as a NaN texel. [`clamp01`] is `max`-then-`min` for exactly that.
    #[test]
    fn a_color_from_the_wire_is_inside_the_cube() {
        let wire = |c: [f32; 3]| {
            carbonite::from_slice_static::<Srgb>(&carbonite::to_vec_static(&c).expect("encodes"))
                .expect("decodes")
        };
        assert_eq!(wire([-1.0, 2.0, f32::NAN]).get(), [0.0, 1.0, 0.0]);
        assert_eq!(
            wire([f32::INFINITY, f32::NEG_INFINITY, 1e30]).get(),
            [1.0, 0.0, 1.0]
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
        // `From` in both directions, and the way in clamps like every other door.
        assert_eq!(Srgb::from([2.0, -1.0, 0.5]).get(), [1.0, 0.0, 0.5]);
        assert_eq!(<[f32; 3]>::from(c), [0.25, 0.5, 0.75]);
        // The named constants are the corners they say they are.
        assert_eq!(Srgb::BLACK.get(), [0.0; 3]);
        assert_eq!(Srgb::WHITE.get(), [1.0; 3]);
        // `const`, so a default or a palette entry can be one.
        const PICKED: Srgb = Srgb::new([9.0, 0.5, -3.0]);
        assert_eq!(PICKED.get(), [1.0, 0.5, 0.0]);
    }

    fn close(a: [f32; 4], b: [f32; 4], eps: f32) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() <= eps)
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
