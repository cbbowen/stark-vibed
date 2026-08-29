//! The canvas **view**: how the painting is being looked at (§6.4, §18.1.2).
//!
//! Pan, zoom, rotation and the mirror — per-client, never logged, never sent, and
//! invisible to replay. That is why it is here and not in `stark-model` beside the
//! tile grid it does arithmetic against.
//!
//! **It was in the model, and it did not belong there.** That crate's charter is
//! "the document, and nothing else", and the rule it states for deciding what is
//! the document is mechanical: a type that is `Serialize` is a fact about the log
//! and lives there, a type that holds a tile is a cache and lives here. A view
//! transform is neither — it is session state, as its own doc has always said —
//! and `stark-net`, the consumer the crate split was made for (§2), was compiling
//! four hundred lines of it that it can never use.
//!
//! The tile grid it maps onto stays in the model, because *that* is document
//! vocabulary: a footprint quantizes against `TILE_SIZE` and a saved log is
//! addressed in it. So the split runs between the canvas and the eye, which is the
//! same line §18.1.2 draws when it says two people sharing a drawing can have it
//! at different angles.

use std::f32::consts::{FRAC_PI_2, TAU};

use stark_model::geom::{Extent2, Mat2, Vec2};

/// The pan/zoom/rotate/mirror transform applied when presenting the canvas to a
/// substrate (§6.4). This is session state and is never historized.
///
/// Everything here is *how you are looking at* the painting rather than anything
/// about it, which is what makes turning the canvas and holding it up to a mirror
/// the same kind of act as panning: per-client, never logged, never sent, and
/// invisible to replay (§18.1.2). Two people sharing a drawing can
/// have it at different angles.
/// **Mutation goes through the methods below, and every one of them is total**: an
/// argument that would leave the view unusable ([`usable`](Self::usable)) is refused
/// and the view is left exactly as it was.
///
/// That is a rule about a *class*, and it is the only thing standing between a
/// frontend's arithmetic and a panic three subsystems away. `f32::clamp` passes NaN
/// straight through — both of its comparisons are false — so
/// `pinch(.., scale: NaN, ..)` used to store a NaN zoom despite the clamp, and the
/// next `screen_to_canvas` then handed NaN canvas coordinates to the stroke fitter,
/// whose normal equations are unsolvable at any ridge and whose solve says so by
/// panicking. Every guard that existed caught it *downstream* — `export_view`
/// refusing to render, `footprint` refusing a control point — so the view stayed
/// poisoned and export failed for the rest of the session with no way back.
///
/// The fields stay public because reading them is the whole point (a frontend maps
/// pointer events through this on every report). Writing one directly bypasses the
/// rule; there is no such write in the workspace, and `a_view_never_stores_a_number_it_cannot_use`
/// is what keeps the mutators honest.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ViewTransform {
    /// Canvas-space point shown at the center of the viewport.
    pub center: Vec2,
    /// Zoom factor: screen pixels per canvas pixel (1.0 = 1:1).
    pub zoom: f32,
    /// How far the canvas is turned on screen, in radians, positive **clockwise**
    /// (the sense of a canvas rotated on an easel, in a y-down frame).
    ///
    /// Applied after [`flip_h`](Self::flip_h), so the mirror is about the canvas's
    /// own vertical axis whatever angle it is being viewed at — which is what makes
    /// "flip it to check the drawing" mean the same thing at every rotation.
    pub rotation: f32,
    /// Whether the canvas is mirrored left↔right on screen — the oldest trick for
    /// catching a drawing error, since the eye stops recognising what it expected to
    /// see and starts seeing what is there.
    ///
    /// Only the horizontal one, and that is not a limitation: a vertical flip is this
    /// one composed with a half turn, and the pair "mirror + angle" says which of
    /// those two the artist meant. It also keeps the mirror a *toggle* rather than
    /// two independent booleans that can both be on.
    pub flip_h: bool,
    /// Size of the target substrate, in pixels.
    pub viewport: Extent2,
}

impl ViewTransform {
    /// A 1:1 upright view centered on the canvas origin for a viewport of `size`.
    pub fn identity(size: Extent2) -> Self {
        Self {
            center: Vec2::ZERO,
            zoom: 1.0,
            rotation: 0.0,
            flip_h: false,
            viewport: size,
        }
    }

    /// The same view rendered at `n` samples per axis: the viewport and the zoom
    /// scaled together, everything else left alone (§6.4).
    ///
    /// Scaling *both* is what makes supersampling invisible to every pass. The
    /// canvas→NDC map comes out identical (the zoom and the viewport divide out), so
    /// the same canvas pixel lands at the same place in the picture; and anything a
    /// shader measures in target px — an outline's width, a matte's edge fade, a
    /// guide's line — is measured against the scaled zoom, so it comes out `n` times
    /// wider in a picture that is about to be `n` times smaller. Which is the same
    /// width, drawn with `n²` samples of coverage instead of one.
    pub fn supersampled(self, n: u32) -> Self {
        Self {
            zoom: self.zoom * n as f32,
            viewport: Extent2::new(self.viewport.width * n, self.viewport.height * n),
            ..self
        }
    }

    /// The canvas→screen linear map with the zoom divided out: the mirror, then the
    /// rotation. Orthogonal (`|det| = 1`), so its transpose is its inverse — which is
    /// the whole reason the view keeps an angle and a flag rather than a free matrix
    /// that would have to be inverted, and could drift away from being a rigid motion.
    ///
    /// Frontend chrome that draws a canvas-space shape over the canvas hands exactly
    /// this to CSS: a rotated frame, or the transform widget's ellipse, is its
    /// canvas-space shape composed with this.
    pub fn orientation(self) -> Mat2 {
        Mat2::from_angle(self.rotation) * self.mirror()
    }

    /// The mirror alone, without the turn.
    fn mirror(self) -> Mat2 {
        if self.flip_h {
            Mat2::from_cols(Vec2::new(-1.0, 0.0), Vec2::Y)
        } else {
            Mat2::IDENTITY
        }
    }

    /// The full canvas→screen linear map: [`orientation`](Self::orientation) scaled
    /// by the zoom.
    pub fn linear(self) -> Mat2 {
        self.orientation() * self.zoom
    }

    /// The screen→canvas linear map. The transpose rather than a general inverse,
    /// because [`orientation`](Self::orientation) is orthogonal.
    pub fn inverse_linear(self) -> Mat2 {
        self.orientation().transpose() * (1.0 / self.zoom)
    }

    /// A screen-pixel *delta* as a canvas-space delta — a drag, in other words, which
    /// is a direction rather than a place and so does not go through the centre.
    pub fn canvas_delta(self, screen: Vec2) -> Vec2 {
        self.inverse_linear() * screen
    }

    /// Linear map from canvas pixels to normalized device coordinates:
    /// `ndc = m * canvas_pos + translate`. Derivation in §6.4.
    ///
    /// The y row is negated because canvas y is downward while NDC y is upward. An
    /// upright, unmirrored view leaves `m` diagonal and this is exactly the scale it
    /// always was.
    pub fn canvas_to_ndc(self) -> (Mat2, Vec2) {
        let vw = self.viewport.width.max(1) as f32;
        let vh = self.viewport.height.max(1) as f32;
        // The screen→NDC scale with the zoom already folded in, so an upright view
        // computes bit-for-bit what it did before rotation existed.
        let d = Vec2::new(2.0 * self.zoom / vw, -2.0 * self.zoom / vh);
        let o = self.orientation();
        // Scaling the *rows* of `o` by `d`, which in column-major is each column
        // multiplied componentwise.
        let m = Mat2::from_cols(o.x_axis * d, o.y_axis * d);
        (m, -(m * self.center))
    }

    /// Inverse of the view: map a screen-pixel position (origin top-left) to a
    /// canvas-space point. Used to turn pointer input into stroke samples.
    pub fn screen_to_canvas(self, screen: Vec2) -> Vec2 {
        self.canvas_delta(screen - self.half()) + self.center
    }

    /// Forward of [`screen_to_canvas`](Self::screen_to_canvas): a canvas-space
    /// point to its screen-pixel position (origin top-left). Used by frontend
    /// chrome that has to sit over a canvas-space feature — the frame's handles
    /// (§15.7).
    pub fn canvas_to_screen(self, canvas: Vec2) -> Vec2 {
        self.linear() * (canvas - self.center) + self.half()
    }

    /// The canvas-space **bounding box** of everything the viewport shows.
    ///
    /// A bound rather than the region itself, because under rotation the region is a
    /// turned rectangle and its callers — framing what you are looking at, falling
    /// back to "the visible canvas" when nothing is painted — all want a rect. It
    /// therefore covers a little more than is really on screen at an angle, which is
    /// the safe direction for both.
    pub fn visible_bounds(self) -> (Vec2, Vec2) {
        let half = self.half();
        let (a, b) = (
            self.canvas_delta(half),
            self.canvas_delta(Vec2::new(half.x, -half.y)),
        );
        // The four corners are ±a and ±b, so the bound is the componentwise larger.
        let extent = a.abs().max(b.abs());
        (self.center - extent, self.center + extent)
    }

    /// Half the viewport, in screen px — the centre of the substrate.
    fn half(self) -> Vec2 {
        Vec2::new(self.viewport.width as f32, self.viewport.height as f32) * 0.5
    }

    /// The rotation that would put the canvas-space direction `up` at the top of the
    /// screen. `None` for a direction with no length to speak of, which asks for
    /// nothing.
    ///
    /// A question rather than an instruction, and stated as a direction rather than an
    /// angle, because that is what a drag gives — and because an angle would have to
    /// be measured against a zero the mirror keeps moving. What the frontend does with
    /// the answer is the frontend's: the navigator eases toward it and snaps it to a
    /// quarter turn, both of which are properties of dragging with a hand rather than
    /// of the view (see `panels::navigator`).
    pub fn rotation_for_up(self, up: Vec2) -> Option<f32> {
        // The mirror is applied before the turn, so where `up` *would* fall is
        // independent of the rotation this is asking about.
        let shown = self.mirror() * up;
        if shown.length_squared() < 1e-12 {
            return None;
        }
        // A vector's angle grows with `rotation`, and screen up is −y (angle −π/2),
        // so the turn wanted is the difference of the two.
        Some((-FRAC_PI_2 - shown.to_angle()).rem_euclid(TAU))
    }

    /// Turn the canvas to `radians`, normalised to one revolution so the value stays
    /// bounded however many turns it is given.
    pub fn set_rotation(&mut self, radians: f32) {
        self.commit(Self {
            rotation: radians.rem_euclid(TAU),
            ..*self
        });
    }

    /// Whether this view can be rendered and inverted through: finite everywhere,
    /// and with a zoom that can actually be divided by.
    ///
    /// **One definition, asked by the mutators and by the render path alike.**
    /// `stark-engine`'s `Engine::export_view` spelled the same predicate out
    /// inline, which is how a view could be refused at the render and still be
    /// sitting in the session — the check said what was wrong without stopping it
    /// being stored. Now the store is what refuses, and the render's check is the
    /// same question asked of a value that has already passed it.
    ///
    /// The viewport is absent because it is `u32`: there is no unusable one, and
    /// `canvas_to_ndc` floors it at 1 rather than dividing by zero.
    pub fn usable(self) -> bool {
        self.center.is_finite() && self.rotation.is_finite() && self.zoom.is_finite()
            // Not merely finite: `inverse_linear` divides by it, so a zoom of zero
            // sends every screen→canvas map to infinity just as surely as a NaN does.
            // Nor merely positive — a subnormal zoom is `> 0.0` and still reciprocates
            // to infinity, so the test is the division the map actually performs.
            && self.zoom.recip().is_finite()
    }

    /// Adopt `candidate` if it is one this view could be used as — the funnel every
    /// mutator here ends in, and the whole of the rule stated on the type.
    ///
    /// Whole-view rather than per-field, because the fields are not independent: a
    /// pinch derives its centre *from* the zoom it just set, so a NaN scale reaches
    /// the centre too and refusing one field would store half a poisoned view. This
    /// way a refused mutation is a no-op rather than a partial one.
    fn commit(&mut self, candidate: Self) {
        if candidate.usable() {
            *self = candidate;
        }
    }

    /// Pan by a **screen-pixel** drag: content follows the cursor, so the centre
    /// moves opposite, carried into canvas units through the whole map (a turned or
    /// mirrored canvas sends a screen drag somewhere else entirely).
    ///
    /// A method rather than `view.center -= view.canvas_delta(d)` at the call site,
    /// so that the one door onto the centre is a total one — see the type's note.
    pub fn pan_by(&mut self, screen_delta: Vec2) {
        let delta = self.canvas_delta(screen_delta);
        self.commit(Self {
            center: self.center - delta,
            ..*self
        });
    }

    /// Put `point` (canvas space) at the middle of the viewport.
    pub fn center_on(&mut self, point: Vec2) {
        self.commit(Self {
            center: point,
            ..*self
        });
    }

    /// Frame the canvas-space rect `min..max` on screen whole: centred, upright and
    /// unmirrored, at the largest zoom that leaves `margin` of the viewport — a
    /// fraction of each axis, on each side — clear around it.
    ///
    /// **The easel is straightened, deliberately**, which is what makes this "show me
    /// the piece" rather than "zoom to fit". A turn and a mirror are ways of *looking*
    /// at a painting (§18.1.2), so the caller here is a piece arriving rather than a
    /// hand adjusting one — the same reading that has `stark-engine`'s `ExportPlan::view` write a
    /// file upright at whatever angle the canvas is being worked at. It is also the
    /// only fit this asks: at an angle the rect's *screen* footprint is a larger,
    /// turned box, so fitting one and fitting the other are two questions with two
    /// answers.
    ///
    ///
    /// Refused, like every mutator here, when handed a rect no view could be fitted
    /// to: an inverted or empty one leaves the view exactly as it was rather than
    /// storing a zoom worked out from a negative width.
    pub fn show_rect(&mut self, min: Vec2, max: Vec2, margin: f32) {
        let size = max - min;
        // Also the NaN gate: every comparison against one is false, so a rect with a
        // NaN corner is refused here rather than reaching the clamp below, which
        // would pass it straight through (see the note on the type).
        if !(size.x > 0.0 && size.y > 0.0) {
            return;
        }
        let room = Vec2::new(self.viewport.width as f32, self.viewport.height as f32)
            * (1.0 - 2.0 * margin.clamp(0.0, 0.4));
        self.commit(Self {
            center: (min + max) * 0.5,
            // The smaller of the two scales is the one that fits *both* axes — the
            // binding one. Clamped like a pinch's, because how far this view can be
            // zoomed is a property of the view and not of whoever asked it to move.
            zoom: (room / size)
                .min_element()
                .clamp(Self::MIN_ZOOM, Self::MAX_ZOOM),
            rotation: 0.0,
            flip_h: false,
            ..*self
        });
    }

    /// Note that the target substrate is now `viewport` pixels.
    ///
    /// Nothing to refuse — a pixel size is `u32` — but it goes through the same door
    /// as the rest, so "the view is mutated through its methods" has no exceptions to
    /// remember.
    pub fn resize(&mut self, viewport: Extent2) {
        self.commit(Self { viewport, ..*self });
    }

    /// Mirror what is **on screen**, left↔right — the flip an artist means when they
    /// hold the drawing up to a mirror to check it.
    ///
    /// Screen-relative, not canvas-relative, which is the whole point: at any angle it
    /// swaps what is on the left of the screen with what is on the right, so the check
    /// means the same thing however the easel is turned. Mirroring about the canvas's
    /// own axis instead would, on a canvas turned a quarter, swap top for bottom —
    /// technically a flip, and not the one anyone asked for.
    ///
    /// Reflecting the *result* keeps the view a rotation-and-a-mirror rather than
    /// becoming a free matrix, because a reflection can be pushed back through a
    /// rotation: `M·R(θ) = R(−θ)·M`. So the whole operation is to negate the angle and
    /// toggle the mirror — and doing it twice is exactly the identity, which is what
    /// makes it a toggle rather than a setting.
    pub fn mirror_screen_h(&mut self) {
        // One commit rather than a `set_rotation` and a toggle: the pair *is* the
        // reflection (`M·R(θ) = R(−θ)·M`), so a refused half would leave a view that
        // is neither the one before nor the one after.
        self.commit(Self {
            rotation: (-self.rotation).rem_euclid(TAU),
            flip_h: !self.flip_h,
            ..*self
        });
    }

    /// Scale the zoom by `factor` while keeping the canvas point under `anchor`
    /// (a screen-pixel position) pinned in place — cursor-anchored zoom.
    pub fn zoom_about(&mut self, anchor: Vec2, factor: f32) {
        self.pinch(anchor, anchor, factor, 0.0)
    }

    /// Move, scale and turn the view **in one act**, about a screen-pixel point: the
    /// canvas point under `anchor` ends up under `to`, having been scaled by `scale`
    /// and turned by `turn` radians clockwise about it. The two-finger gesture
    /// (§18.1.7), and — with `to == anchor` and no turn — the wheel's zoom.
    ///
    /// One command rather than a pan, a zoom and a turn, because a pinch is one motion
    /// of one pair of fingers and the three are not independent: each of the three
    /// anchors against the view it is applied to, so sending them in sequence would
    /// have the second and third re-anchor against a view the hand never saw, and the
    /// point being held would slide out from under it. Composed here, what the fingers
    /// hold is held exactly.
    ///
    /// The mirror is left alone and the turn adds straight onto the angle, because the
    /// gesture is stated in **screen** terms: a twist clockwise on the glass is a twist
    /// clockwise on the screen at any angle and either handedness — the same
    /// screen-relative sense [`mirror_screen_h`](Self::mirror_screen_h) is defined in.
    /// (`R(δ)·R(θ)·M = R(θ+δ)·M`, so it stays a rotation-and-a-mirror.)
    /// Refused whole when any of its four arguments is non-finite, which is what the
    /// clamp below cannot do on its own: `f32::clamp` compares, and every comparison
    /// against NaN is false, so `(zoom * NaN).clamp(MIN, MAX)` is NaN and not `MAX`.
    /// A refused pinch leaves the view exactly as the hand found it — see the note on
    /// the type for what a stored NaN used to cost.
    pub fn pinch(&mut self, anchor: Vec2, to: Vec2, scale: f32, turn: f32) {
        // The canvas point the gesture is holding, read through the view as it stands.
        let held = self.screen_to_canvas(anchor);
        // Built as a candidate rather than applied in three steps, because the three
        // are not independent: the centre is solved *through* the new angle and zoom,
        // so a NaN in either reaches it, and committing as they were computed would
        // store a view that was poisoned two assignments ago.
        let mut next = Self {
            rotation: (self.rotation + turn).rem_euclid(TAU),
            zoom: (self.zoom * scale).clamp(Self::MIN_ZOOM, Self::MAX_ZOOM),
            ..*self
        };
        // ...and put back under the hand, through the view as it now is. Solving
        // `screen_to_canvas(to) == held` for the centre, which is the one degree of
        // freedom left once the angle and the zoom are set.
        next.center = held - next.canvas_delta(to - next.half());
        self.commit(next);
    }

    const MIN_ZOOM: f32 = 0.05;
    const MAX_ZOOM: f32 = 64.0;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An upright, unmirrored view of `size` at `zoom`, centred on `center`.
    fn view(center: Vec2, zoom: f32, size: Extent2) -> ViewTransform {
        ViewTransform {
            center,
            zoom,
            ..ViewTransform::identity(size)
        }
    }

    /// **No mutator may store a number the view cannot be used with**, whatever it is
    /// handed — the rule stated on [`ViewTransform`], asked of every mutator there is.
    ///
    /// This is the guard that was missing, and its absence was not theoretical: the
    /// clamp in `pinch` looks like it bounds the zoom, and `f32::clamp` passes NaN
    /// through both of its comparisons, so a NaN scale stored a NaN zoom. From there
    /// `screen_to_canvas` fed NaN canvas positions to the stroke fitter and the
    /// spline solve panicked — and nothing in between could put the view back,
    /// because every other check in the codebase asks at the *render*, by which point
    /// the bad value is already resident.
    ///
    /// Driven off a list of mutations rather than one test each, so that a mutator
    /// added later has somewhere obvious to be added and no way to be quietly
    /// exempt.
    #[test]
    fn a_view_never_stores_a_number_it_cannot_use() {
        /// One way the view can be moved: a name for the failure message, and the
        /// mutation with the number under test threaded into it.
        type Mutation = (&'static str, fn(&mut ViewTransform, f32));

        let poison = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0];
        // Every way the view can be moved. `f` is the poisoned number under test;
        // each mutation puts it wherever that mutation can take one.
        let mutations: [Mutation; 10] = [
            ("pan_by.x", |v, f| v.pan_by(Vec2::new(f, 3.0))),
            ("pan_by.y", |v, f| v.pan_by(Vec2::new(3.0, f))),
            ("center_on", |v, f| v.center_on(Vec2::new(f, f))),
            ("set_rotation", |v, f| v.set_rotation(f)),
            ("show_rect.min", |v, f| {
                v.show_rect(Vec2::splat(f), Vec2::splat(100.0), 0.04)
            }),
            ("show_rect.max", |v, f| {
                v.show_rect(Vec2::ZERO, Vec2::splat(f), 0.04)
            }),
            ("show_rect.margin", |v, f| {
                v.show_rect(Vec2::ZERO, Vec2::splat(100.0), f)
            }),
            ("zoom_about.factor", |v, f| {
                v.zoom_about(Vec2::new(40.0, 30.0), f)
            }),
            ("zoom_about.anchor", |v, f| {
                v.zoom_about(Vec2::splat(f), 1.5)
            }),
            ("pinch.turn", |v, f| {
                v.pinch(Vec2::splat(10.0), Vec2::splat(20.0), 1.5, f)
            }),
        ];
        for (name, apply) in mutations {
            for f in poison {
                let before = view(Vec2::new(10.0, 20.0), 2.0, Extent2::new(800, 600));
                let mut after = before;
                apply(&mut after, f);
                assert!(
                    after.usable(),
                    "{name} stored {f} and left the view unusable: {after:?}",
                );
                // **NaN is the value that must be refused whole**, and it is the only
                // one. An infinite or zero *scale* is a request the clamp can honour —
                // it means "as far as this goes", and `MIN_ZOOM`/`MAX_ZOOM` are where
                // it goes — so those legitimately take effect. NaN cannot be clamped
                // into range (every comparison against it is false, which is the bug
                // this whole rule exists for), so the only total answer is to leave
                // the view alone.
                //
                // Whole, not merely refused: `zoom_about` with a NaN *anchor* has a
                // perfectly good zoom and an unusable centre, and committing the
                // first while dropping the second would leave a view the hand never
                // asked for. That case is in the list above for exactly this reason.
                if f.is_nan() {
                    assert_eq!(
                        after, before,
                        "{name} partly applied a mutation carrying NaN",
                    );
                }
            }
        }
        // A mirror is two changes that are one act, so it must not half-apply either.
        let mut v = view(Vec2::ZERO, 1.0, Extent2::new(800, 600));
        v.mirror_screen_h();
        v.mirror_screen_h();
        assert_eq!(
            v,
            view(Vec2::ZERO, 1.0, Extent2::new(800, 600)),
            "not a toggle"
        );
    }

    /// The clamp in `pinch` really is load-bearing for *finite* input — the test
    /// above would pass on a `pinch` that refused every scale.
    #[test]
    fn a_pinch_still_zooms_and_stays_in_range() {
        let mut v = view(Vec2::ZERO, 1.0, Extent2::new(800, 600));
        v.zoom_about(Vec2::new(400.0, 300.0), 4.0);
        assert_eq!(v.zoom, 4.0);
        // Past the ceiling clamps rather than refusing: a finite request is honoured
        // as far as it can be.
        v.zoom_about(Vec2::new(400.0, 300.0), 1e6);
        assert_eq!(v.zoom, ViewTransform::MAX_ZOOM);
        v.zoom_about(Vec2::new(400.0, 300.0), 1e-9);
        assert_eq!(v.zoom, ViewTransform::MIN_ZOOM);
        // The two ends of "as far as this goes", which the rule above deliberately
        // lets through rather than refusing: they are answerable, and answering them
        // is what keeps a zoom of zero from becoming a division by zero downstream.
        v.zoom_about(Vec2::new(400.0, 300.0), f32::INFINITY);
        assert_eq!(v.zoom, ViewTransform::MAX_ZOOM);
        v.zoom_about(Vec2::new(400.0, 300.0), 0.0);
        assert_eq!(v.zoom, ViewTransform::MIN_ZOOM);
    }

    #[test]
    fn viewport_center_maps_to_view_center() {
        let view = view(Vec2::new(123.0, -45.0), 2.0, Extent2::new(800, 600));
        let center_px = Vec2::new(400.0, 300.0);
        let mapped = view.screen_to_canvas(center_px);
        assert!((mapped - view.center).length() < 1e-3, "got {mapped:?}");
    }

    #[test]
    fn zoom_scales_screen_to_canvas_distance() {
        let view = view(Vec2::ZERO, 2.0, Extent2::new(800, 600));
        // 100 screen px right of center is 50 canvas px at 2x zoom.
        let p = view.screen_to_canvas(Vec2::new(500.0, 300.0));
        assert!((p - Vec2::new(50.0, 0.0)).length() < 1e-3, "got {p:?}");
    }

    #[test]
    fn zoom_about_keeps_cursor_point_fixed() {
        let mut view = view(Vec2::new(10.0, 20.0), 1.0, Extent2::new(800, 600));
        let anchor = Vec2::new(620.0, 130.0); // arbitrary off-center cursor
        let canvas_under = view.screen_to_canvas(anchor);
        view.zoom_about(anchor, 2.5);
        let after = view.screen_to_canvas(anchor);
        assert!(
            (after - canvas_under).length() < 1e-3,
            "anchor drifted: {canvas_under:?} -> {after:?}"
        );
        assert!((view.zoom - 2.5).abs() < 1e-4);
    }

    /// What the two-finger gesture promises (§18.1.7): the canvas stays stuck to the
    /// fingers. Stated as the property rather than as a formula, because the property
    /// is the whole reason the pan, the zoom and the turn are one command — and it is
    /// what a caller composing them by hand would silently lose.
    #[test]
    fn a_pinch_leaves_the_canvas_under_both_fingers() {
        // Fingers before and after, in screen px, arbitrarily placed and moved.
        let from = [Vec2::new(240.0, 500.0), Vec2::new(560.0, 260.0)];
        let to = [Vec2::new(310.0, 380.0), Vec2::new(500.0, 120.0)];
        for flip_h in [false, true] {
            for turn in [0.0, 0.4, FRAC_PI_2, 2.5, TAU - 0.1] {
                let mut v = ViewTransform {
                    center: Vec2::new(-30.0, 90.0),
                    zoom: 1.75,
                    rotation: turn,
                    flip_h,
                    viewport: Extent2::new(800, 600),
                };
                let held = from.map(|f| v.screen_to_canvas(f));
                // Exactly what the frontend measures off the pair: the midpoint it
                // travelled between, how much further apart the fingers ended up, and
                // how far the line between them swung.
                let (u, w) = (from[1] - from[0], to[1] - to[0]);
                v.pinch(
                    0.5 * (from[0] + from[1]),
                    0.5 * (to[0] + to[1]),
                    w.length() / u.length(),
                    u.angle_to(w),
                );
                for (canvas, finger) in held.into_iter().zip(to) {
                    let now = v.canvas_to_screen(canvas);
                    assert!(
                        (now - finger).length() < 1e-2,
                        "flip {flip_h}, turn {turn}: {canvas:?} should be under \
                         {finger:?}, is under {now:?}"
                    );
                }
            }
        }
    }

    /// A pinch broken into steps has to land where the whole one would — the fingers
    /// report one at a time, so every gesture *is* a sequence of steps, and a
    /// composition that drifted would leave the canvas sliding under a hand that had
    /// come back to where it started.
    #[test]
    fn a_pinch_taken_in_steps_lands_where_one_step_would() {
        let path = [
            [Vec2::new(200.0, 400.0), Vec2::new(500.0, 300.0)],
            [Vec2::new(220.0, 380.0), Vec2::new(540.0, 260.0)],
            [Vec2::new(260.0, 300.0), Vec2::new(600.0, 240.0)],
            [Vec2::new(200.0, 400.0), Vec2::new(500.0, 300.0)], // back to the start
        ];
        let start = ViewTransform {
            center: Vec2::new(12.0, -7.0),
            zoom: 0.8,
            rotation: 1.1,
            flip_h: true,
            viewport: Extent2::new(800, 600),
        };
        let mut v = start;
        for pair in path.windows(2) {
            let (u, w) = (pair[0][1] - pair[0][0], pair[1][1] - pair[1][0]);
            v.pinch(
                0.5 * (pair[0][0] + pair[0][1]),
                0.5 * (pair[1][0] + pair[1][1]),
                w.length() / u.length(),
                u.angle_to(w),
            );
        }
        // The fingers ended where they began, so the view must have too.
        assert!(
            (v.center - start.center).length() < 1e-2,
            "centre {:?}",
            v.center
        );
        assert!((v.zoom - start.zoom).abs() < 1e-4, "zoom {}", v.zoom);
        assert!(
            (v.rotation - start.rotation).abs() < 1e-4,
            "rotation {}",
            v.rotation
        );
    }

    /// The turn and the mirror are still a *view*: screen→canvas and canvas→screen
    /// have to remain each other's inverse, or a stroke would land somewhere other
    /// than under the pen (§18.1.2).
    #[test]
    fn the_two_directions_invert_each_other_at_any_orientation() {
        let probes = [
            Vec2::new(0.0, 0.0),
            Vec2::new(37.0, -12.5),
            Vec2::new(-400.0, 250.0),
        ];
        for flip_h in [false, true] {
            for turn in [0.0, 0.4, FRAC_PI_2, 2.5, TAU - 0.1] {
                let v = ViewTransform {
                    center: Vec2::new(-30.0, 90.0),
                    zoom: 1.75,
                    rotation: turn,
                    flip_h,
                    viewport: Extent2::new(800, 600),
                };
                for p in probes {
                    let round = v.screen_to_canvas(v.canvas_to_screen(p));
                    assert!(
                        (round - p).length() < 1e-2,
                        "flip {flip_h}, turn {turn}: {p:?} -> {round:?}"
                    );
                }
                // And the NDC map agrees with the screen map, since the shaders use
                // one and every chrome overlay uses the other.
                let (m, t) = v.canvas_to_ndc();
                for p in probes {
                    let s = v.canvas_to_screen(p);
                    let ndc = m * p + t;
                    let from_screen = Vec2::new(
                        2.0 * s.x / v.viewport.width as f32 - 1.0,
                        1.0 - 2.0 * s.y / v.viewport.height as f32,
                    );
                    assert!(
                        (ndc - from_screen).length() < 1e-4,
                        "flip {flip_h}, turn {turn}: ndc {ndc:?} vs {from_screen:?}"
                    );
                }
            }
        }
    }

    /// An upright, unmirrored view has to go through the matrix path *bit for bit* as
    /// it would through a plain scale-and-offset — the golden images are blessed
    /// against it, and a one-ulp drift in the raster transform is exactly the kind of
    /// change that shows up as a mysterious failure a year later.
    #[test]
    fn an_upright_view_maps_exactly_as_a_plain_scale_does() {
        let v = view(Vec2::new(123.5, -45.25), 2.5, Extent2::new(801, 597));
        let (m, t) = v.canvas_to_ndc();
        let vw = v.viewport.width as f32;
        let vh = v.viewport.height as f32;
        let scale = Vec2::new(2.0 * v.zoom / vw, -2.0 * v.zoom / vh);
        assert_eq!(m.x_axis, Vec2::new(scale.x, 0.0));
        assert_eq!(m.y_axis, Vec2::new(0.0, scale.y));
        assert_eq!(t, -v.center * scale);
    }

    /// The gesture: the direction dragged becomes the direction that points up.
    #[test]
    fn the_rotation_for_a_direction_puts_it_at_the_top_of_the_screen() {
        for flip_h in [false, true] {
            for up in [
                Vec2::new(0.0, -1.0),
                Vec2::new(1.0, 0.0),
                Vec2::new(-3.0, 4.0),
                Vec2::new(0.7, 0.7),
            ] {
                let mut v = ViewTransform {
                    flip_h,
                    ..ViewTransform::identity(Extent2::new(400, 300))
                };
                v.set_rotation(v.rotation_for_up(up).expect("a direction"));
                let shown = (v.linear() * up).normalize();
                assert!(
                    (shown - Vec2::new(0.0, -1.0)).length() < 1e-4,
                    "flip {flip_h}, up {up:?} came out pointing {shown:?}"
                );
            }
        }
        // A drag that has gone nowhere asks for nothing, rather than for whatever
        // `atan2(0, 0)` happens to be.
        let v = ViewTransform::identity(Extent2::new(400, 300));
        assert!(v.rotation_for_up(Vec2::ZERO).is_none());
    }

    /// The mirror is **screen**-relative: at any angle it swaps the left of the screen
    /// with the right. Canvas-relative would, on a canvas turned a quarter, swap top
    /// for bottom instead — a flip, but not the one anyone asked for.
    #[test]
    fn mirroring_swaps_the_screens_left_and_right_at_any_angle() {
        for turn in [0.0, FRAC_PI_2, 1.1, TAU - 0.3] {
            let mut v = ViewTransform {
                rotation: turn,
                ..ViewTransform::identity(Extent2::new(400, 300))
            };
            // Where four canvas directions land on screen, before and after.
            let probes = [Vec2::X, Vec2::Y, Vec2::new(-2.0, 1.0), Vec2::new(3.0, 4.0)];
            let before: Vec<Vec2> = probes.iter().map(|p| v.linear() * *p).collect();
            v.mirror_screen_h();
            for (p, was) in probes.iter().zip(&before) {
                let now = v.linear() * *p;
                assert!(
                    (now - Vec2::new(-was.x, was.y)).length() < 1e-3,
                    "turn {turn}: {p:?} was at {was:?}, mirrored to {now:?}"
                );
            }
            // And twice is the identity — which is what makes it a toggle.
            v.mirror_screen_h();
            for (p, was) in probes.iter().zip(&before) {
                assert!(
                    (v.linear() * *p - *was).length() < 1e-3,
                    "turn {turn}: {p:?}"
                );
            }
        }
    }

    /// What the navigator's marker and "fit to view" both rest on.
    #[test]
    fn visible_bounds_cover_the_viewport_at_any_orientation() {
        let v = ViewTransform {
            center: Vec2::new(10.0, -20.0),
            zoom: 2.0,
            rotation: 0.6,
            flip_h: true,
            viewport: Extent2::new(400, 300),
        };
        let (min, max) = v.visible_bounds();
        let (w, h) = (400.0, 300.0);
        for corner in [
            Vec2::new(0.0, 0.0),
            Vec2::new(w, 0.0),
            Vec2::new(0.0, h),
            Vec2::new(w, h),
        ] {
            let c = v.screen_to_canvas(corner);
            assert!(
                c.x >= min.x - 1e-3
                    && c.x <= max.x + 1e-3
                    && c.y >= min.y - 1e-3
                    && c.y <= max.y + 1e-3,
                "corner {corner:?} -> {c:?} outside {min:?}..{max:?}"
            );
        }
        // Upright, it is exactly the plain half-viewport-over-zoom rect.
        let upright = view(Vec2::ZERO, 2.0, Extent2::new(400, 300));
        assert_eq!(
            upright.visible_bounds(),
            (Vec2::new(-100.0, -75.0), Vec2::new(100.0, 75.0))
        );
    }

    /// "Show me the piece" (§15.6): the whole rect on screen, centred, with the easel
    /// straightened — and *snug* on the axis that binds, since a fit that came in
    /// under the box on both axes would be quietly wasting the window it was given.
    #[test]
    fn show_rect_frames_the_whole_rect_upright() {
        // 400 × 200, nowhere near where the view is looking — the case a document
        // load is, on a canvas with no edges to have been near in the first place.
        let (min, max) = (Vec2::new(-300.0, 40.0), Vec2::new(100.0, 240.0));
        let turned = ViewTransform {
            center: Vec2::splat(9000.0),
            zoom: 7.0,
            rotation: 0.9,
            flip_h: true,
            viewport: Extent2::new(800, 600),
        };

        let mut v = turned;
        v.show_rect(min, max, 0.0);
        assert_eq!(v.center, (min + max) * 0.5);
        assert_eq!(
            (v.rotation, v.flip_h),
            (0.0, false),
            "the easel should be straightened"
        );
        // 800/400 = 2 binds before 600/200 = 3.
        assert!((v.zoom - 2.0).abs() < 1e-6, "zoom is {}", v.zoom);
        let (lo, hi) = v.visible_bounds();
        assert!(lo.x <= min.x + 1e-3 && hi.x >= max.x - 1e-3, "cropped");
        assert!(lo.y <= min.y + 1e-3 && hi.y >= max.y - 1e-3, "cropped");

        // The margin is a fraction of each axis on each side, so it comes off the
        // binding axis rather than off the rect: 800 × 0.9 = 720 usable px.
        let mut m = turned;
        m.show_rect(min, max, 0.05);
        assert!((m.zoom - 1.8).abs() < 1e-6, "zoom is {}", m.zoom);
        let (lo, hi) = m.visible_bounds();
        assert!(
            lo.x < min.x && hi.x > max.x && lo.y < min.y && hi.y > max.y,
            "the piece should have room around it on every side"
        );

        // A rect that is not one — inverted here, empty in the poison list above —
        // leaves the view exactly as it was rather than fitting a negative width.
        let mut bad = turned;
        bad.show_rect(max, min, 0.05);
        assert_eq!(bad, turned, "an inverted rect should be refused whole");
    }
}
