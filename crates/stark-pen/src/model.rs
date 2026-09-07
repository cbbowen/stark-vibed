//! What a stylus reports and what a frontend claims — the half of this crate that
//! names no platform type, and so the half a test can reach on any host (§11).
//!
//! Everything here is arithmetic over numbers a backend hands in. That is not a
//! tidiness argument: `stark-wgpui-frontend` is excluded from CI for want of X11
//! packages, and a backend is `cfg(windows)`, so this module is the only part of pen
//! support any gate compiles at all.

/// A rectangle in the window's client area, in **physical** px.
///
/// Physical rather than logical because a backend measures in what the platform
/// reports and the platform reports device pixels — the scale factor is the
/// frontend's to apply, once, at the boundary where it already holds one.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rect {
    pub left: f32,
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
}

impl Rect {
    /// Whether `at` falls inside — **left and top inclusive, right and bottom
    /// exclusive**, so two rectangles sharing an edge cannot both claim a point on
    /// it.
    #[must_use]
    pub fn contains(self, at: [f32; 2]) -> bool {
        at[0] >= self.left && at[0] < self.right && at[1] >= self.top && at[1] < self.bottom
    }

    /// Whether the rectangle encloses nothing, which is what an inverted or
    /// zero-sized one means. A claim over an empty rectangle takes no press.
    #[must_use]
    pub fn is_empty(self) -> bool {
        !(self.right > self.left && self.bottom > self.top)
    }
}

/// The region of the window where a stylus press opens a gesture on the picture, and
/// whether it may do so at all this frame.
///
/// **A frontend publishes this; a backend reads it.** It is what decides whether a
/// pen contact is taken over by this crate or left to the platform's own
/// mouse-compatibility path, and the split is not a preference — a press the pen
/// takes never reaches the frontend as a mouse event at all, so a claim that covered
/// a button would be a button the pen could not press.
///
/// `enabled` is the whole-window off switch, for the states where a press over the
/// canvas is not paint: a menu standing open over it, a transform widget live in it.
/// Those want the mouse path, which already handles them, and want nothing this crate
/// offers — a transform handle does not care what a nib weighs.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Claim {
    pub rect: Rect,
    pub enabled: bool,
}

impl Claim {
    /// Claim nothing: what a frontend that has not spoken yet is taken to mean, and
    /// what one publishes to hand the window back for a frame.
    pub const NONE: Self = Self {
        rect: Rect {
            left: 0.0,
            top: 0.0,
            right: 0.0,
            bottom: 0.0,
        },
        enabled: false,
    };

    /// Whether a contact starting at `at` is this crate's to carry.
    #[must_use]
    pub fn takes(self, at: [f32; 2]) -> bool {
        self.enabled && !self.rect.is_empty() && self.rect.contains(at)
    }
}

/// A snapshot of the stylus, in the units the rest of Stark measures a stroke in.
///
/// Normalized here rather than at the consumer because the ranges are the
/// platform's — 0..1024 for one API's pressure, ±90° for its tilt — and a frontend
/// that divided by them would be a second copy of a fact this crate already knows
/// (§11.2). What comes out is what `InputSample` wants: pressure in `0..=1`, and tilt
/// as a fraction of a right angle on each axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    /// Where the nib is, in the window's client area, in **physical** px.
    ///
    /// May be fractional, and may fall outside the window: a stylus goes on reporting
    /// through a drag that has left the frame, which is exactly the report a stroke
    /// being pulled off the edge is made of.
    pub position: [f32; 2],
    /// How hard, `0..=1`. Full pressure on a device that reports none, so a stylus
    /// with no sensor draws like a mouse rather than like nothing.
    pub pressure: f32,
    /// Lean on each axis as a fraction of a right angle, `-1..=1`. `[+, +]` is right
    /// and toward the user — the web frontend's normalization, which is degrees over
    /// ninety, so the two chromes hand the engine the same number for the same lean.
    pub tilt: [f32; 2],
    /// Seconds on a monotonic clock whose epoch is **this crate's own** and means
    /// nothing to anybody else.
    ///
    /// That is enough, and it is why no offset against the frontend's clock is
    /// computed anywhere: the path fitter re-bases every channel time to the first
    /// sample of the gesture it is fitting (`path::fit`), so what has to hold is that
    /// one gesture is timed by one device. Which is what makes the high-resolution
    /// timestamps a digitizer reports usable without trusting them to agree with
    /// anything.
    pub time: f64,
    /// Whether the tail is what faces the surface rather than the nib — the eraser
    /// end (§18.1.8).
    ///
    /// Read at the press and not after: a stylus is not turned over mid-stroke, and a
    /// gesture that changed what it was halfway would be one action the log could not
    /// describe.
    pub inverted: bool,
}

impl Default for Pose {
    /// A nib resting at the origin at full pressure — the mouse's pose, which is the
    /// right thing for a field a device declined to report to fall back to.
    fn default() -> Self {
        Self {
            position: [0.0, 0.0],
            pressure: 1.0,
            tilt: [0.0, 0.0],
            time: 0.0,
            inverted: false,
        }
    }
}

/// What happened, as distinct from where the stylus was when it did.
///
/// Three, and no hover among them. A backend reports only the contact it was given —
/// press to lift — because that is the whole of what it took over from the platform:
/// a stylus in the air goes on driving the chrome through the platform's own path,
/// and a hover this crate also reported would be the same motion arriving twice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Touched down. The frontend opens a gesture here.
    Down,
    /// Moved while down.
    Move,
    /// Lifted. The frontend ends the gesture here.
    ///
    /// **Sent for a lost gesture too**, not only a clean lift: a pointer whose
    /// capture is taken away — a system gesture, a display change, another window
    /// claiming it — reports nothing further, and a consumer that only closed on the
    /// tidy case would hold an open stroke for the rest of the session.
    Up,
}

/// One thing the stylus did, and its pose when it did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Report {
    pub phase: Phase,
    pub pose: Pose,
}

/// Map a point in a pointer device's own high-resolution space onto the screen pixels
/// the window is laid out in.
///
/// A digitizer resolves far below the screen it sits under, and the platform says so
/// twice: once as a rounded pixel and once as a raw reading in the device's own
/// units, alongside the two rectangles that relate them. This is that relation, and
/// it is the whole of what stands between a stroke fitted to the hand and one fitted
/// to a staircase — `stark_ui::input::PEN_RESOLUTION` claims half a pixel, and the
/// rounded report cannot answer for it.
///
/// `None` when either rectangle encloses nothing, which is what an unavailable device
/// reports and would otherwise be a division by zero.
#[must_use]
pub fn map_device(at: [f32; 2], device: Rect, display: Rect) -> Option<[f32; 2]> {
    if device.is_empty() || display.is_empty() {
        return None;
    }
    let sx = (display.right - display.left) / (device.right - device.left);
    let sy = (display.bottom - display.top) / (device.bottom - device.top);
    Some([
        display.left + (at[0] - device.left) * sx,
        display.top + (at[1] - device.top) * sy,
    ])
}

/// The lean of a stylus that reports its orientation as two angles rather than as two
/// tilts, normalized the way [`Pose::tilt`] is.
///
/// Wintab describes a stylus by where it points and how far over it is: `azimuth` is
/// the compass bearing of the lean, clockwise from straight away from the user, and
/// `altitude` is the angle between the barrel and the **tablet surface** — a right
/// angle when the pen stands upright, nothing when it lies flat. Both in radians. The
/// pointer API describes the same lean as two angles from vertical instead, so one of
/// the two has to be converted, and this is it.
///
/// The conversion is the projection of the barrel onto each axis — across the tablet
/// with `sin`, along it with `cos`, the second negated because a pen leaning away from
/// the user reports negative there, which is the convention `POINTER_PEN_INFO` states
/// for `tiltY` and so the one both backends have to hand the engine.
///
/// Written as `atan2` rather than as the `atan(sin(azimuth) / tan(altitude))` the
/// formula is usually quoted in, and not for tidiness: `tan` of a right angle is not
/// an infinity in an `f32` but a large number of *whichever sign* the rounding of
/// `π/2` lands on, so the quotient form needs a guard against a pen lying flat and
/// gets an upright one wrong when the guard is written against the sign. `atan2` has
/// no division to protect and is exact at both ends.
///
/// One reading has no answer rather than a hard one: a pen lying flat in the tablet's
/// own plane is fully over on one axis and **undefined on the other**, because its
/// projection onto that plane is the zero vector and there is no angle in it to
/// report. What comes back there is a finite number in range and nothing more is
/// promised — the reading is unphysical anyway, since a nib in contact is not in the
/// tablet's plane.
#[must_use]
pub fn tilt_from_orientation(azimuth: f32, altitude: f32) -> [f32; 2] {
    let (sin_az, cos_az) = azimuth.sin_cos();
    // Absolute, because a tablet that senses below its own plane reports a negative
    // altitude for a lean this side of it and means the same angle by it.
    let (sin_alt, cos_alt) = altitude.abs().sin_cos();
    let right = std::f32::consts::FRAC_PI_2;
    [
        (sin_az * cos_alt).atan2(sin_alt) / right,
        -(cos_az * cos_alt).atan2(sin_alt) / right,
    ]
}

/// Map a reading on one axis onto another, where each is an origin and a **signed**
/// extent.
///
/// Signed because the flip is the point: a digitizer measures up the tablet and a
/// screen measures down the glass, so the extent that carries one to the other is
/// negative on that axis and the arithmetic has to survive it. A rectangle could not
/// say this — [`Rect`] is normalized, and an inverted one reads as enclosing
/// nothing — which is why this is a pair of numbers rather than two corners.
///
/// `None` for a source that spans nothing, which is what an axis a device does not
/// report looks like and would otherwise be a division by zero.
#[must_use]
pub fn map_axis(value: f32, from: (f32, f32), to: (f32, f32)) -> Option<f32> {
    let (from_origin, from_extent) = from;
    let (to_origin, to_extent) = to;
    if from_extent == 0.0 {
        return None;
    }
    Some(to_origin + (value - from_origin) * to_extent / from_extent)
}

/// How far a [`map_device`] reading may sit from the platform's own rounded pixel
/// before the mapping is judged wrong, in screen px.
///
/// The two describe the same nib, so they agree to within the rounding — a couple of
/// pixels covers that, and nothing covers a device whose rectangles do not mean what
/// this crate reads them to mean. **That is what this constant is for**: the
/// sub-pixel path is worth having and is not worth a stroke landing an inch from the
/// cursor, so a backend measures it against the reading it could have trusted instead
/// and gives it up for good if the two disagree.
pub const DEVICE_AGREEMENT_PX: f32 = 2.0;

/// Whether a mapped reading is close enough to the rounded one to be believed.
#[must_use]
pub fn agrees(mapped: [f32; 2], rounded: [f32; 2]) -> bool {
    (mapped[0] - rounded[0]).abs() <= DEVICE_AGREEMENT_PX
        && (mapped[1] - rounded[1]).abs() <= DEVICE_AGREEMENT_PX
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(left: f32, top: f32, right: f32, bottom: f32) -> Rect {
        Rect {
            left,
            top,
            right,
            bottom,
        }
    }

    /// The half-open edge rule, stated from both sides: a point on the left edge is
    /// inside and the same point on the right edge is not, so the panel column and
    /// the canvas beside it cannot both take one press.
    #[test]
    fn edges_belong_to_one_side() {
        let r = rect(10.0, 20.0, 110.0, 220.0);
        assert!(r.contains([10.0, 20.0]), "the top-left corner is inside");
        assert!(
            !r.contains([110.0, 100.0]),
            "the right edge belongs to whatever is beyond it"
        );
        assert!(
            !r.contains([50.0, 220.0]),
            "the bottom edge belongs to whatever is below it"
        );
    }

    /// An inverted or collapsed rectangle encloses nothing rather than everything —
    /// the shape a window reports mid-resize, and the one that would otherwise hand
    /// the whole frame to the pen.
    #[test]
    fn a_degenerate_rect_encloses_nothing() {
        assert!(rect(10.0, 10.0, 10.0, 200.0).is_empty());
        assert!(rect(200.0, 10.0, 10.0, 200.0).is_empty());
        assert!(!rect(200.0, 10.0, 10.0, 200.0).contains([100.0, 100.0]));
    }

    /// The off switch is checked before the rectangle, so the states where a press
    /// over the canvas is not paint hand the whole window back whatever the layout
    /// said.
    #[test]
    fn a_disabled_claim_takes_nothing() {
        let claim = Claim {
            rect: rect(0.0, 0.0, 1000.0, 1000.0),
            enabled: false,
        };
        assert!(!claim.takes([500.0, 500.0]));
        assert!(
            Claim {
                enabled: true,
                ..claim
            }
            .takes([500.0, 500.0])
        );
        assert!(!Claim::NONE.takes([0.0, 0.0]), "the default takes nothing");
    }

    /// The device mapping carries corners to corners, which is the whole of what it
    /// claims to do.
    #[test]
    fn device_corners_land_on_display_corners() {
        let device = rect(0.0, 0.0, 20000.0, 10000.0);
        let display = rect(0.0, 0.0, 3840.0, 2160.0);
        assert_eq!(
            map_device([0.0, 0.0], device, display),
            Some([0.0, 0.0]),
            "the origin"
        );
        assert_eq!(
            map_device([20000.0, 10000.0], device, display),
            Some([3840.0, 2160.0]),
            "the far corner"
        );
    }

    /// A device whose origin is not the display's, which is what a tablet mapped to
    /// one monitor of several reports.
    #[test]
    fn device_mapping_carries_the_offset() {
        let device = rect(1000.0, 2000.0, 3000.0, 4000.0);
        let display = rect(1920.0, 0.0, 3840.0, 1080.0);
        let mid = map_device([2000.0, 3000.0], device, display).expect("a mappable reading");
        assert!(
            (mid[0] - 2880.0).abs() < 1e-3 && (mid[1] - 540.0).abs() < 1e-3,
            "the device's midpoint is the display's midpoint, got {mid:?}"
        );
    }

    /// The reading a device that is not there gives, which must not be a division by
    /// zero.
    #[test]
    fn an_absent_device_maps_nothing() {
        let empty = rect(0.0, 0.0, 0.0, 0.0);
        let ok = rect(0.0, 0.0, 100.0, 100.0);
        assert_eq!(map_device([1.0, 1.0], empty, ok), None);
        assert_eq!(map_device([1.0, 1.0], ok, empty), None);
    }

    /// A pen standing straight up leans nowhere, whichever way it is pointing.
    #[test]
    fn an_upright_pen_has_no_tilt() {
        let right = std::f32::consts::FRAC_PI_2;
        for eighth in 0..8 {
            let azimuth = eighth as f32 * std::f32::consts::FRAC_PI_4;
            let [x, y] = tilt_from_orientation(azimuth, right);
            assert!(
                x.abs() < 1e-6 && y.abs() < 1e-6,
                "azimuth {azimuth} upright gave {x}, {y}"
            );
        }
    }

    /// The sign convention both backends have to agree about: leaning away from the
    /// user is negative on y, and leaning right is positive on x — which is what
    /// `POINTER_PEN_INFO` says its own `tiltY`/`tiltX` mean.
    #[test]
    fn leaning_away_is_negative_and_leaning_right_is_positive() {
        let eighth = std::f32::consts::FRAC_PI_4;
        let away = tilt_from_orientation(0.0, eighth);
        assert!(
            away[1] < -0.1,
            "leaning away should be negative y, got {away:?}"
        );
        assert!(away[0].abs() < 1e-6, "and nothing on x, got {away:?}");

        let right = tilt_from_orientation(std::f32::consts::FRAC_PI_2, eighth);
        assert!(
            right[0] > 0.1,
            "leaning right should be positive x, got {right:?}"
        );
        assert!(right[1].abs() < 1e-6, "and nothing on y, got {right:?}");
    }

    /// A pen laid flat is a number rather than a NaN, whichever way it points.
    ///
    /// Only that. The lean of a flat pen is fully over on one axis and **undefined on
    /// the other** — its projection onto the perpendicular plane is the zero vector,
    /// so there is no angle in it to report and one answer is as good as another. A
    /// test that pinned one down would be asserting a choice rather than a fact. What
    /// has to hold is that no NaN reaches `InputSample::is_admissible`, which refuses
    /// one and takes the whole gesture with it.
    #[test]
    fn a_flat_pen_is_a_number_whichever_way_it_points() {
        for step in 0..16 {
            let azimuth = step as f32 * std::f32::consts::TAU / 16.0;
            let [x, y] = tilt_from_orientation(azimuth, 0.0);
            assert!(
                x.is_finite() && y.is_finite(),
                "azimuth {azimuth} flat gave {x}, {y}"
            );
        }
    }

    /// Tilt never leaves the range the engine reads it in, at any lean.
    #[test]
    fn tilt_stays_within_a_right_angle() {
        for step in 0..64 {
            let azimuth = step as f32 * std::f32::consts::TAU / 64.0;
            for rung in 0..=16 {
                let altitude = rung as f32 * std::f32::consts::FRAC_PI_2 / 16.0;
                let [x, y] = tilt_from_orientation(azimuth, altitude);
                assert!(
                    (-1.0..=1.0).contains(&x) && (-1.0..=1.0).contains(&y),
                    "azimuth {azimuth} altitude {altitude} gave {x}, {y}"
                );
            }
        }
    }

    /// The flip is the reason this takes signed extents rather than two corners: a
    /// digitizer measures up the tablet and a screen measures down the glass.
    #[test]
    fn a_negative_extent_flips_the_axis() {
        let up = (0.0, 1000.0);
        let down = (1080.0, -1080.0);
        assert_eq!(
            map_axis(0.0, up, down),
            Some(1080.0),
            "the tablet's bottom edge"
        );
        assert_eq!(map_axis(1000.0, up, down), Some(0.0), "and its top");
        assert_eq!(
            map_axis(500.0, up, down),
            Some(540.0),
            "and the middle of both"
        );
    }

    /// An axis the device does not report spans nothing, and must not divide by it.
    #[test]
    fn an_axis_that_spans_nothing_maps_nothing() {
        assert_eq!(map_axis(5.0, (0.0, 0.0), (0.0, 100.0)), None);
    }

    /// The guard the sub-pixel path is worth having *because of*: a mapping that
    /// lands where the rounded report already said is believed, and one that lands a
    /// hundred pixels away is not.
    #[test]
    fn only_a_mapping_that_agrees_is_believed() {
        assert!(agrees([100.4, 200.6], [100.0, 201.0]), "rounding");
        assert!(!agrees([340.0, 200.0], [100.0, 201.0]), "wrong on x");
        assert!(
            !agrees([100.0, 900.0], [100.0, 200.0]),
            "right on one axis is not enough"
        );
    }
}
