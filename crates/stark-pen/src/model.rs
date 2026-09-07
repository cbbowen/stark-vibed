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
