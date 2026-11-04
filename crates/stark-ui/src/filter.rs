//! What a filter bar tunes (§21.6): each kind's knobs as a table, and the geometry of
//! the two pictures that stand in for knobs where a filter's numbers are not separate
//! things — the chroma dial (§21.5) and the dispersion pad (§21.10).
//!
//! The bar is each frontend's. The rows, their ranges and what a point in either picture
//! *means* are one answer, or a filter dialled in one app would read back as a different
//! setting in the other.

use stark_engine::filters::{CONTRAST_PIVOT, dispersion_weight};
use stark_model::color::linear_to_srgb;
use stark_model::document::{Aperture, ChromaticAberration, ColorAdjust, FocalBlur};

use crate::icons::Icon;

/// One slider on a filter bar: what it is called, its range, and the two ends of the
/// round trip through the filter's own parameter struct `F`.
///
/// A table rather than hand-written rows, because a hand-written row is a place for one
/// of these to disagree with the value it displays. `get`/`set` are the pair that makes
/// the whole filter travel on every edit (§21.6): a bar reads the settings off the
/// projection, replaces one number, and sends the result back.
pub struct FilterKnob<F: 'static> {
    pub name: &'static str,
    pub hint: &'static str,
    /// The mark this knob wears — and so whether its word may be hidden. A knob with a
    /// mark can lose its word; one without keeps it, because a row with neither would be
    /// an anonymous track. One field, so the wrong pair is unrepresentable.
    ///
    /// The rule is about a bar, not the set: one marked slider among unmarked ones reads
    /// worse than none. Each bar draws one kind's table, and the color filter's — the
    /// only one with marks — is marked throughout.
    pub glyph: Option<Icon>,
    /// The span, in display units, off the core's own bounds (`ColorAdjust::EXPOSURE`
    /// and friends), so the track and the sanitizer cannot disagree about how far a knob
    /// goes.
    pub range: (f32, f32),
    /// What the track snaps to, in display units, or `None` for continuous. For the one
    /// knob that **counts** — an iris's blades — where the arrow keys' hundredth of the
    /// span rounds back to the blade it started on and reads as a dead key.
    pub step: Option<f32>,
    /// Engine units per display unit: `1 / DEG` for an angle, whose track is in degrees
    /// while the engine holds radians (§21.5), and 1 for the rest. Translating here keeps
    /// how an angle is *presented* out of the log.
    pub scale: f32,
    pub get: fn(&F) -> f32,
    pub set: fn(F, f32) -> F,
    /// How the number beside the track reads, in display units — per knob, so renaming
    /// one cannot change how its value prints.
    pub fmt: fn(f32) -> String,
}

impl<F: 'static> FilterKnob<F> {
    /// The number beside the track for `settings`.
    pub fn readout(&self, settings: &F) -> String {
        (self.fmt)((self.get)(settings) / self.scale)
    }
}

/// Degrees per radian, for the knobs whose display unit is not the engine's.
pub const DEG: f32 = 180.0 / std::f32::consts::PI;

/// A whole number of degrees — an angle's readout to anyone dragging it; decimals of a
/// degree are noise. One spelling for the dial's hue and the pad's axis, so the two ways
/// an angle is shown cannot drift apart.
pub fn fmt_degrees(v: f32) -> String {
    format!("{}\u{00B0}", v.round() as i32)
}

/// The color filter's **lightness** knobs — the two the chroma dial has nothing to say
/// about, because they act on Oklab `L` and the dial is one slice of constant `L`. A
/// plane picture cannot show a move along the axis it is perpendicular to.
pub const COLOR_KNOBS: &[FilterKnob<ColorAdjust>] = &[
    FilterKnob {
        name: "Exposure",
        hint: "Stops of light. +1 is twice as much, \u{2212}1 is half \u{2014} applied \
               to the light itself, so it brightens the way an exposure does rather \
               than the way a brightness slider does.",
        glyph: Some(crate::icons::EXPOSURE),
        range: ColorAdjust::EXPOSURE,
        step: None,
        scale: 1.0,
        get: |c| c.exposure,
        set: |c, v| ColorAdjust { exposure: v, ..c },
        // Signed: a stop is centred on zero, and "0.50" and "+0.50" say different things.
        fmt: |v| format!("{v:+.2}"),
    },
    FilterKnob {
        name: "Contrast",
        hint: "Spread about mid-grey. 1 leaves it alone, 0 flattens the picture to \
               one tone. It moves lightness only \u{2014} the colors keep their \
               saturation, which is not true of a contrast curve in sRGB.",
        glyph: Some(crate::icons::CONTRAST),
        range: ColorAdjust::CONTRAST,
        step: None,
        scale: 1.0,
        get: |c| c.contrast,
        set: |c, v| ColorAdjust { contrast: v, ..c },
        fmt: |v| format!("{v:.2}"),
    },
];

/// The focal blur's **size** knob (§21.12) — the one every aperture has. A radius is one
/// number with no partner to make a vector with, so a track is the right control. No
/// glyph, on [`FilterKnob::glyph`]'s bar rule.
pub const BLUR_KNOBS: &[FilterKnob<FocalBlur>] = &[FilterKnob {
    name: "Radius",
    hint: "The circle of confusion's radius, in canvas pixels \u{2014} how far each \
           point's light is spread. A true convolution with the aperture, so lights \
           bloom into bokeh instead of washing out (\u{a7}21.12); stated on the \
           canvas, so the blur scales with the painting rather than with the window.",
    glyph: None,
    range: FocalBlur::RADIUS,
    step: None,
    scale: 1.0,
    get: |b| b.radius,
    set: |b, v| FocalBlur { radius: v, ..b },
    fmt: |v| format!("{v:.1} px"),
}];

/// The knobs an [`Aperture::Blades`] iris has: how many, and turned how far.
///
/// Every `set` in the aperture tables rebuilds its own variant rather than reaching
/// through an accessor (§21.12): a parameter a shape does not have is not there to be
/// written, so the `other => other` arm is the shape changed underneath a row about to be
/// unmounted, and leaving it alone is the only right answer.
pub const BLADES_KNOBS: &[FilterKnob<FocalBlur>] = &[
    FilterKnob {
        name: "Blades",
        hint: "How many blades the iris has \u{2014} five, six and eight are the \
               common ones, and each makes its own polygon out of every \
               out-of-focus highlight. Wide open a lens shows none of this; it is \
               stopping down that brings the blades into the bokeh.",
        glyph: None,
        range: (Aperture::BLADES.0 as f32, Aperture::BLADES.1 as f32),
        // A count, so the track lands on one — see [`FilterKnob::step`].
        step: Some(1.0),
        scale: 1.0,
        get: |b| match b.aperture {
            Aperture::Blades { count, .. } => count as f32,
            // In range, like every sibling table's unreachable arm: a fallback outside
            // `range` would print a readout the track cannot show.
            _ => Aperture::BLADES.0 as f32,
        },
        set: |b, v| FocalBlur {
            aperture: match b.aperture {
                Aperture::Blades { angle, .. } => Aperture::Blades {
                    count: v.round() as u32,
                    angle,
                },
                other => other,
            },
            ..b
        },
        fmt: |v| format!("{v:.0}"),
    },
    FilterKnob {
        name: "Angle",
        hint: "Which way the polygon points.",
        glyph: None,
        // In degrees, the display unit; `scale` carries it back to radians.
        range: (Aperture::ANGLE.0 * DEG, Aperture::ANGLE.1 * DEG),
        step: None,
        scale: 1.0 / DEG,
        get: |b| match b.aperture {
            Aperture::Blades { angle, .. } => angle,
            _ => 0.0,
        },
        set: |b, v| FocalBlur {
            aperture: match b.aperture {
                Aperture::Blades { count, .. } => Aperture::Blades { count, angle: v },
                other => other,
            },
            ..b
        },
        fmt: fmt_degrees,
    },
];

/// The knob an [`Aperture::Disc`] has: how much of its middle is taken out.
///
/// Also where the mirror lens is advertised: the two are one shape, so the doughnut is a
/// place on this track rather than a chip of its own — and a knob whose point cannot be
/// guessed has to say what it is for.
pub const DISC_KNOBS: &[FilterKnob<FocalBlur>] = &[FilterKnob {
    name: "Obstruction",
    hint: "How much of the aperture's middle is blocked, as a share of the radius. \
           0 is a plain disc; wind it up and you have a mirror lens, whose \
           secondary shadows the centre and pushes every highlight out into a \
           doughnut.",
    glyph: None,
    range: Aperture::OBSTRUCTION,
    step: None,
    scale: 1.0,
    get: |b| match b.aperture {
        Aperture::Disc { obstruction } => obstruction,
        _ => 0.0,
    },
    set: |b, v| FocalBlur {
        aperture: match b.aperture {
            Aperture::Disc { .. } => Aperture::Disc { obstruction: v },
            other => other,
        },
        ..b
    },
    fmt: |v| format!("{v:.2}"),
}];

/// The knobs an [`Aperture::Oval`] has: how hard the squeeze, and along what.
pub const OVAL_KNOBS: &[FilterKnob<FocalBlur>] = &[
    FilterKnob {
        name: "Squeeze",
        hint: "The long axis over the short one \u{2014} 2\u{d7} is the anamorphic \
               cinema means when it says the word. The long axis stays the radius, \
               so squeezing narrows the bokeh rather than stretching it.",
        glyph: None,
        range: Aperture::SQUEEZE,
        step: None,
        scale: 1.0,
        get: |b| match b.aperture {
            Aperture::Oval { squeeze, .. } => squeeze,
            _ => 1.0,
        },
        set: |b, v| FocalBlur {
            aperture: match b.aperture {
                Aperture::Oval { angle, .. } => Aperture::Oval { squeeze: v, angle },
                other => other,
            },
            ..b
        },
        fmt: |v| format!("{v:.2}\u{d7}"),
    },
    FilterKnob {
        name: "Angle",
        hint: "Which way the long axis runs.",
        glyph: None,
        // In degrees, the display unit; `scale` carries it back to radians.
        range: (Aperture::ANGLE.0 * DEG, Aperture::ANGLE.1 * DEG),
        step: None,
        scale: 1.0 / DEG,
        get: |b| match b.aperture {
            Aperture::Oval { angle, .. } => angle,
            _ => 0.0,
        },
        set: |b, v| FocalBlur {
            aperture: match b.aperture {
                Aperture::Oval { squeeze, .. } => Aperture::Oval { squeeze, angle: v },
                other => other,
            },
            ..b
        },
        fmt: fmt_degrees,
    },
];

/// The knobs the chosen aperture adds under the radius — one or two, never none: every
/// shape is a family rather than a single figure.
pub fn aperture_knobs(aperture: &Aperture) -> &'static [FilterKnob<FocalBlur>] {
    match aperture {
        Aperture::Disc { .. } => DISC_KNOBS,
        Aperture::Blades { .. } => BLADES_KNOBS,
        Aperture::Oval { .. } => OVAL_KNOBS,
    }
}

/// The mark a shape's chip wears — the bokeh itself, at the setting the chip hands out.
/// What the shape *makes* is the whole content of the choice, so the picture says more
/// than the word does.
pub fn aperture_glyph(aperture: &Aperture) -> Icon {
    match aperture {
        Aperture::Disc { .. } => crate::icons::APERTURE_DISC,
        Aperture::Blades { .. } => crate::icons::APERTURE_BLADES,
        Aperture::Oval { .. } => crate::icons::APERTURE_OVAL,
    }
}

/// The sentence a shape's chip carries — what the aperture *is*, since the picture it
/// makes is the whole reason to pick one.
pub fn aperture_hint(aperture: &Aperture) -> &'static str {
    match aperture {
        Aperture::Disc { .. } => {
            "A circular aperture \u{2014} the lens wide open, and the circle of \
             confusion at its most ideal. Obstruct the middle of it and you have a \
             mirror lens, whose highlights come out as doughnuts."
        }
        Aperture::Blades { .. } => {
            "The iris stopped down onto its blades: highlights take the shape of \
             the polygon they were let through."
        }
        Aperture::Oval { .. } => {
            "An anamorphic lens: the front element squeezes one axis, and \
             highlights stretch into ovals with it."
        }
    }
}

// —— the chroma dial ————————————————————————————————————————————————————————

/// The dial's field, on screen (px) — square, like the color picker's.
pub const DIAL_PX: f32 = 116.0;

/// The Oklab chroma the rim stands for: the reference color whose whole hue circle the
/// dial tracks, so a point of the drawn ring is *where a color of this chroma ends up*.
///
/// Moderately saturated rather than the gamut's edge, so the ring at rest sits well
/// inside the plane and has somewhere to grow when saturation is pushed past 1.
pub const DIAL_CHROMA: f32 = 0.12;

/// Half-extent of the `(a, b)` plane the dial draws, per axis.
///
/// Derived rather than chosen: as far as the centre can travel, plus as wide as the rim
/// can get. That makes the box hold **every reachable setting**, so no combination of a
/// strong cast and a strong saturation can carry the rim handle somewhere it cannot be
/// grabbed back from.
pub const DIAL_AB: f32 = ColorAdjust::TINT.1 + ColorAdjust::SATURATION.1 * DIAL_CHROMA;

/// The Oklab lightness the dial's plane is drawn at: mid-grey, the lightness the contrast
/// knob pivots about.
///
/// One slice is enough — what the dial shows is a map of `(a, b)` alone, identical at
/// every `L` — so there is no slice control: it could not change a pixel.
pub const DIAL_L: f32 = CONTRAST_PIVOT;

/// px per unit of `a`/`b` — the one conversion every drawn radius goes through.
pub const DIAL_SCALE: f32 = DIAL_PX * 0.5 / DIAL_AB;

/// How near the pointer must come to the rim handle to take it rather than the centre
/// (px). Wider than the drawn dot, as a pointer target should be, and small enough that
/// the rest of the field — all the centre's — stays a big target.
pub const DIAL_GRAB: f32 = 10.0;

/// What a stepped drag (Shift) snaps each quantity to. Their job is to make the round
/// numbers reachable by hand: 0°, a saturation of exactly 1, no tint at all and a whole
/// pixel of spread are single points a pointer will not land on twice.
///
/// One angle step for both pictures, because "the round angles" is a fact about hands:
/// the dial's rotation and the fringe's axis want the same dozen directions.
pub const ANGLE_STEP: f32 = std::f32::consts::PI / 12.0; // 15°
/// The saturation a stepped drag snaps to, as a factor.
pub const SATURATION_STEP: f32 = 0.05;
/// The tint a stepped drag snaps to, per Oklab axis.
pub const TINT_STEP: f32 = 0.01;
/// The spread a stepped drag snaps to, in canvas px.
pub const SPREAD_STEP: f32 = 1.0;

/// Where an Oklab `(a, b)` lands in the dial's box, in px from its top-left. `a` runs
/// left→right and `b` bottom→top — the color picker's own orientation, warm at the top,
/// because they are two pictures of the same plane.
pub fn dial_xy(ab: [f32; 2]) -> (f32, f32) {
    (
        (ab[0] / DIAL_AB * 0.5 + 0.5) * DIAL_PX,
        (0.5 - ab[1] / DIAL_AB * 0.5) * DIAL_PX,
    )
}

/// The inverse: the `(a, b)` under a pointer at `(x, y)` in the box.
///
/// Unclamped, deliberately: [`Filter::sanitized`](stark_model::document::Filter::sanitized)
/// is the single place a number is held to its range, and a clamp here would be a second
/// opinion — which is how a control comes to disagree with the value it displays.
pub fn dial_ab(x: f32, y: f32) -> [f32; 2] {
    [
        (x / DIAL_PX * 2.0 - 1.0) * DIAL_AB,
        (1.0 - y / DIAL_PX * 2.0) * DIAL_AB,
    ]
}

/// `v` to the nearest multiple of `step` — see [`ANGLE_STEP`] and its neighbours.
pub fn snapped(v: f32, step: f32) -> f32 {
    (v / step).round() * step
}

/// [`snapped`] when `step` (Shift is held), `v` as it is otherwise.
fn snap_if(step: bool, v: f32, by: f32) -> f32 {
    if step { snapped(v, by) } else { v }
}

/// A vector's length and direction, the direction `keep` where the vector has none.
///
/// At the centre a direction does not exist and `atan2` would answer 0 — which would
/// fling a hue back to red, or an axis back to horizontal, every time a drag crossed
/// the middle on its way somewhere. Both pictures keep the angle the gesture already
/// had and move the radius alone; it is one fact about polar coordinates.
fn polar(dx: f32, dy: f32, keep: f32) -> (f32, f32) {
    let r = dx.hypot(dy);
    (r, if r > 1e-4 { dy.atan2(dx) } else { keep })
}

/// What a press on the chroma dial takes hold of.
///
/// Decided once, on pointer-down ([`dial_grab`]), and held for the whole gesture — so a
/// rotation swung in past the centre does not become a translation halfway through,
/// which is the one way a live hit test could rewrite a number the hand was not on.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum DialGrab {
    /// The rim handle: its direction and its distance from the centre, which are the
    /// hue and the saturation. Taken when the pointer comes down within [`DIAL_GRAB`]
    /// of it — including when saturation is 0 and it sits *on* the centre, because
    /// then it is the only way back out and the centre is still reachable everywhere
    /// else in the field.
    ///
    /// `hold` is where the handle sat relative to the pointer that took it, carried
    /// for the gesture so the first sample is the setting already held. Without it a
    /// grab at the edge of [`DIAL_GRAB`] would fling the saturation by half a unit
    /// before the hand had moved — a target has to be wider than the dot it takes,
    /// and everything inside it has to mean *this one*, not *this position*.
    Rim { hold: [f32; 2] },
    /// Anywhere else in the field: the centre, which is the tint. Absolute rather than
    /// held, and that is the difference in kind: the rim is a handle to grab, the
    /// field is a place to put the grey.
    Centre,
}

/// Where `c` sends red — the rim handle — in Oklab `(a, b)`.
pub fn dial_rim(c: &ColorAdjust) -> [f32; 2] {
    [
        c.tint[0] + c.saturation * DIAL_CHROMA * c.hue.cos(),
        c.tint[1] + c.saturation * DIAL_CHROMA * c.hue.sin(),
    ]
}

/// What a press at `(x, y)` in the dial's box takes, against where the rim handle is
/// drawn for `c`.
pub fn dial_grab(c: &ColorAdjust, x: f32, y: f32) -> DialGrab {
    let rim = dial_rim(c);
    let (hx, hy) = dial_xy(rim);
    if (x - hx).hypot(y - hy) > DIAL_GRAB {
        return DialGrab::Centre;
    }
    let at = dial_ab(x, y);
    DialGrab::Rim {
        hold: [rim[0] - at[0], rim[1] - at[1]],
    }
}

/// The setting one pointer sample at `(x, y)` in the dial's box means, for a gesture
/// that took `grab` of `c`; `step` snaps (Shift). Unsanitized, for [`dial_ab`]'s reason.
pub fn dial_drag(c: ColorAdjust, grab: DialGrab, (x, y): (f32, f32), step: bool) -> ColorAdjust {
    let at = dial_ab(x, y);
    match grab {
        // Absolute, not by delta: the pointer *is* where the grey goes. Which also
        // makes a single click on the field a complete edit, the way a grading wheel
        // behaves.
        DialGrab::Centre => ColorAdjust {
            tint: [
                snap_if(step, at[0], TINT_STEP),
                snap_if(step, at[1], TINT_STEP),
            ],
            ..c
        },
        DialGrab::Rim { hold } => {
            let (r, hue) = polar(
                at[0] + hold[0] - c.tint[0],
                at[1] + hold[1] - c.tint[1],
                c.hue,
            );
            ColorAdjust {
                hue: snap_if(step, hue, ANGLE_STEP),
                saturation: snap_if(step, r / DIAL_CHROMA, SATURATION_STEP),
                ..c
            }
        }
    }
}

// —— the dispersion pad ——————————————————————————————————————————————————————

/// The pad's field, on screen (px) — the dial's size, because the two are the same bar's
/// picture, and a bar whose height changed with the kind of filter selected would jump
/// under the pointer that selected it.
pub const PAD_PX: f32 = DIAL_PX;

/// The centre of the field, in its own px — where the picture stays put.
pub const PAD_MID: f32 = PAD_PX * 0.5;

/// The handle's drawn radius (px) — the dot at the blue end of the fringe.
pub const PAD_HANDLE: f32 = 5.0;

/// The radius the widest reachable spread is drawn at (px): the half-width, less the
/// handle and its grab slop — so **no reachable setting puts the handle outside the
/// element that receives the pointer**, and a fatter handle moves the stop rather than
/// escaping it.
pub const PAD_R: f32 = PAD_MID - PAD_HANDLE - 2.0;

/// The spreads the graduation rings stand for, in canvas px — 8 and 32, with the ceiling
/// drawn as the stop beyond them.
///
/// Each is a quarter of the next, which under [`pad_radius`]'s square root puts them at
/// exactly half of each other's radius: the ladder draws itself evenly, and rings that
/// crowd outward *are* the compression. Two rungs rather than three, because a third
/// would land 6 px from the centre with its label on the one above.
pub const PAD_RINGS: &[f32] = &[
    ChromaticAberration::SPREAD.1 / 16.0,
    ChromaticAberration::SPREAD.1 / 4.0,
];

/// How many stops the drawn spectrum spends: enough that the bar reads as a continuum at
/// the widest it is drawn (`2 · PAD_R` px) rather than as bands — what the pass's own tap
/// count buys, for the same reason.
pub const PAD_STOPS: usize = 32;

/// Where a spread is drawn, as a radius in the field (px).
///
/// **Square-root, not linear.** The interesting spreads are the small ones — two or three
/// canvas px is a lens, thirty is an effect — and the ceiling is 128, so a linear pad
/// would spend nine tenths of its radius on settings nobody dials. Equal *area* per unit
/// of spread costs nothing in honesty: the rings state the law and the readout states the
/// number.
pub fn pad_radius(spread: f32) -> f32 {
    PAD_R * (spread / ChromaticAberration::SPREAD.1).max(0.0).sqrt()
}

/// The inverse: the spread a handle at radius `r` means. Unclamped, for [`dial_ab`]'s
/// reason — past the rim a drag simply pins at the ceiling once sanitized.
pub fn pad_spread(r: f32) -> f32 {
    let t = r / PAD_R;
    ChromaticAberration::SPREAD.1 * t * t
}

/// Where the blue end of the fringe is drawn, in the field's px.
///
/// `+y` is **down**, unlike the dial's plane: this is a canvas direction, and the canvas
/// is drawn with `y` down, so the bar in the pad runs the way the fringe runs in the
/// painting.
pub fn pad_xy(c: ChromaticAberration) -> (f32, f32) {
    let r = pad_radius(c.spread);
    (PAD_MID + r * c.angle.cos(), PAD_MID + r * c.angle.sin())
}

/// The setting a pointer at `(x, y)` in the pad means: the pointer *is* the blue end,
/// so a single click anywhere in the field is a complete edit. `step` snaps (Shift);
/// unsanitized, for [`dial_ab`]'s reason.
pub fn pad_drag(c: ChromaticAberration, (x, y): (f32, f32), step: bool) -> ChromaticAberration {
    let (r, angle) = polar(x - PAD_MID, y - PAD_MID, c.angle);
    ChromaticAberration {
        spread: snap_if(step, pad_spread(r), SPREAD_STEP),
        angle: snap_if(step, angle, ANGLE_STEP),
    }
}

/// The dispersion spectrum as gradient stops: `(offset, display sRGB)` from the red end at
/// 0 to the blue end at 1.
///
/// **The pass's own weights, at the pass's own wavelengths** (§21.10), which is what makes
/// the bar a statement about the render rather than a rainbow someone drew. Two
/// adjustments, both about *showing* a response rather than integrating one: the run is
/// normalized so its strongest channel is full intensity (the weights' absolute scale
/// means nothing — the pass divides it out too), and encoded to sRGB for a screen. The
/// relative brightness along the run survives both, so the deep ends read dark as the eye
/// finds them.
pub fn spectrum_stops() -> [(f32, [u8; 3]); PAD_STOPS] {
    let offset = |i: usize| i as f32 / (PAD_STOPS - 1) as f32;
    let weights: [[f32; 3]; PAD_STOPS] = std::array::from_fn(|i| dispersion_weight(offset(i)));
    let peak = weights
        .iter()
        .flatten()
        .copied()
        .fold(0.0f32, f32::max)
        .max(1e-6);
    let quantized = |v: f32| (linear_to_srgb(v / peak).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    std::array::from_fn(|i| (offset(i), weights[i].map(quantized)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A press on the rim handle takes it *where it is*: the first sample of the drag,
    /// at the press itself, is the setting already held — however far inside
    /// [`DIAL_GRAB`] the press landed.
    #[test]
    fn taking_the_rim_moves_nothing_until_the_hand_does() {
        let c = ColorAdjust {
            hue: 1.0,
            saturation: 0.8,
            tint: [0.01, -0.02],
            ..ColorAdjust::default()
        };
        let (hx, hy) = dial_xy(dial_rim(&c));
        let press = (hx + 4.0, hy - 3.0);
        let grab = dial_grab(&c, press.0, press.1);
        assert!(matches!(grab, DialGrab::Rim { .. }));
        let next = dial_drag(c, grab, press, false);
        assert!((next.hue - c.hue).abs() < 1e-4, "{} vs {}", next.hue, c.hue);
        assert!((next.saturation - c.saturation).abs() < 1e-4);
        assert_eq!(next.tint, c.tint);
        // Far from the handle, the press is the centre's.
        assert_eq!(dial_grab(&c, 0.0, 0.0), DialGrab::Centre);
    }

    /// Crossing the centre keeps the angle the gesture had, on both pictures.
    #[test]
    fn the_centre_keeps_the_angle() {
        let c = ColorAdjust {
            hue: 2.0,
            ..ColorAdjust::default()
        };
        let centre = dial_xy(c.tint);
        let next = dial_drag(c, DialGrab::Rim { hold: [0.0, 0.0] }, centre, false);
        assert_eq!((next.hue, next.saturation), (2.0, 0.0));
        let fringe = ChromaticAberration {
            spread: 4.0,
            angle: 0.7,
        };
        let next = pad_drag(fringe, (PAD_MID, PAD_MID), false);
        assert_eq!((next.spread, next.angle), (0.0, 0.7));
    }

    /// The pad's handle, pressed where it is drawn, reads back as the setting it shows.
    #[test]
    fn the_pad_reads_back_its_own_handle() {
        let c = ChromaticAberration {
            spread: 20.0,
            angle: -0.5,
        };
        let next = pad_drag(c, pad_xy(c), false);
        assert!((next.spread - c.spread).abs() < 1e-3);
        assert!((next.angle - c.angle).abs() < 1e-4);
        let stepped = pad_drag(c, pad_xy(c), true);
        assert_eq!(stepped.spread, 20.0);
    }

    /// How close two px positions or two Oklab coordinates have to be to count as the
    /// same point. Generous, because what is pinned is that the two halves of a mapping
    /// are *inverses*, not the last bit of an `f32`.
    const EPS: f32 = 1e-4;

    /// The dial's two halves are inverses.
    ///
    /// Getting it wrong is silent and looks like feel: the handle drifts away from the
    /// pointer dragging it, which reads as a slippery control rather than a sign error.
    /// One half flips `b` and the other flips `y`, so there is a sign in each that only
    /// their composition checks.
    #[test]
    fn the_dial_maps_both_ways() {
        for ab in [
            [0.0, 0.0],
            [DIAL_AB, 0.0],
            [-DIAL_AB, 0.0],
            [0.0, DIAL_AB],
            [0.0, -DIAL_AB],
            [DIAL_AB * 0.37, -DIAL_AB * 0.81],
        ] {
            let (x, y) = dial_xy(ab);
            let back = dial_ab(x, y);
            assert!(
                (back[0] - ab[0]).abs() < EPS && (back[1] - ab[1]).abs() < EPS,
                "{ab:?} went to ({x}, {y}) and came back {back:?}"
            );
        }
    }

    /// And the orientation those signs encode: `a` runs left→right, `b` runs
    /// **bottom→top** — warm at the top, the picker's own plane (§21.5). A test of the
    /// round trip alone would pass with both signs flipped.
    #[test]
    fn the_dial_is_warm_at_the_top() {
        let (x0, y0) = dial_xy([0.0, 0.0]);
        let (right, _) = dial_xy([DIAL_AB * 0.5, 0.0]);
        let (_, up) = dial_xy([0.0, DIAL_AB * 0.5]);
        assert!(right > x0, "+a should run rightward");
        assert!(up < y0, "+b should run upward, which is a smaller y");
    }

    /// The centre of the box is the neutral, on both halves — the one point where a scale
    /// error and an offset error cannot hide behind each other.
    #[test]
    fn the_dial_centre_is_the_grey() {
        let (x, y) = dial_xy([0.0, 0.0]);
        assert!((x - DIAL_PX * 0.5).abs() < EPS);
        assert!((y - DIAL_PX * 0.5).abs() < EPS);
        let back = dial_ab(DIAL_PX * 0.5, DIAL_PX * 0.5);
        assert!(back[0].abs() < EPS && back[1].abs() < EPS);
    }

    /// The fringe pad's two halves are inverses too — and this pair is the one that is
    /// *not* linear (§21.10).
    #[test]
    fn the_fringe_pad_maps_both_ways() {
        let (_, top) = ChromaticAberration::SPREAD;
        for spread in [0.0, 1.0, 2.0, 3.0, 30.0, top * 0.5, top] {
            let r = pad_radius(spread);
            let back = pad_spread(r);
            assert!(
                (back - spread).abs() < 1e-3,
                "{spread} went to radius {r} and came back {back}"
            );
        }
    }

    /// The law that pairing states: **equal area per unit of spread**. Checked as the
    /// property rather than as a constant, so the numbers may move and the claim may not.
    #[test]
    fn the_fringe_pad_spends_equal_area_per_unit() {
        let (_, top) = ChromaticAberration::SPREAD;
        // Area inside the handle's radius, per unit of spread; π cancels, so r² / spread.
        let per_unit = |spread: f32| {
            let r = pad_radius(spread);
            r * r / spread
        };
        let (a, b, c) = (per_unit(top * 0.1), per_unit(top * 0.5), per_unit(top));
        assert!(
            (a - b).abs() < 1e-3 && (b - c).abs() < 1e-3,
            "area per unit drifts across the range: {a}, {b}, {c}"
        );
        // And the rim is the top of the range, which is what makes the pad's edge mean
        // something.
        assert!((pad_radius(top) - PAD_R).abs() < EPS);
        assert!((pad_radius(0.0)).abs() < EPS);
    }

    /// Neither half clamps: `Filter::sanitized` is the one place a number is held to its
    /// range. Pinned because "add a clamp" is the obvious-looking edit.
    #[test]
    fn the_two_inverses_leave_the_clamping_to_the_sanitizer() {
        let (_, top) = ChromaticAberration::SPREAD;
        assert!(pad_spread(PAD_R * 2.0) > top, "the pad clamped a drag");
        let past = dial_ab(DIAL_PX * 2.0, -DIAL_PX);
        assert!(
            past[0] > DIAL_AB && past[1] > DIAL_AB,
            "the dial clamped one"
        );
    }

    /// `snapped` is round-half-away-from-zero to a multiple, and symmetric about zero —
    /// the stepped drag reads the same on either side of neutral.
    #[test]
    fn snapping_is_symmetric_about_zero() {
        assert_eq!(snapped(0.0, 0.25), 0.0);
        assert_eq!(snapped(0.3, 0.25), 0.25);
        assert_eq!(snapped(-0.3, 0.25), -0.25);
        assert_eq!(snapped(0.13, 0.25), 0.25);
        assert_eq!(snapped(-0.13, 0.25), -0.25);
    }

    /// The spectrum runs red end to blue end across the whole bar, and is normalized so
    /// its strongest channel is full — the two claims [`spectrum_stops`] makes about how
    /// the pass's response is shown.
    #[test]
    fn the_spectrum_spans_the_bar_at_full_scale() {
        let stops = spectrum_stops();
        assert_eq!(stops.first().map(|s| s.0), Some(0.0));
        assert_eq!(stops.last().map(|s| s.0), Some(1.0));
        assert!(
            stops.windows(2).all(|w| w[0].0 < w[1].0),
            "the stops do not run in order"
        );
        let brightest = stops.iter().flat_map(|s| s.1).max();
        assert_eq!(brightest, Some(255), "the run is not normalized to full");
    }
}
