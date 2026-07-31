//! Canvas geometry: tile coordinates and the pan/zoom view transform.
//!
//! Canvas space is in pixels with x to the right and y downward. Tile `(i, j)`
//! covers the square `[i*TILE_SIZE, (i+1)*TILE_SIZE) × [j*TILE_SIZE, ...)`.
//! The infinite canvas (§6) is realized by tiles being sparse and
//! addressed by signed integer coordinates.

use std::f32::consts::{FRAC_PI_2, TAU};

pub use glam::{Affine2, Mat2, Vec2};

/// Apron (halo) width in pixels carried around each tile's interior, replicated
/// from the neighboring canvas content (§6.4). The compositor samples a
/// tile's interior with bilinear filtering; without an apron the filter clamps at
/// the tile edge instead of reaching into the neighbor, leaving a visible seam at
/// every boundary under sub-pixel pan or non-1:1 zoom (the seam is then amplified
/// by the media pass's height→normal gradient). One pixel is all bilinear needs;
/// widen this if a future media effect needs more neighbor context.
pub const TILE_APRON: u32 = 1;

/// Physical edge length of a tile's channel textures: interior plus an apron on
/// every side. Tiles are stored at this size; only the interior is presented.
pub const TILE_TEX: u32 = 256;

/// Edge length of a square tile's *interior*, in canvas pixels (§6.1).
/// This is the addressing stride: tile `(i, j)` owns canvas
/// `[i*TILE_SIZE, (i+1)*TILE_SIZE)` — aprons (below) overlap neighbors and are
/// not owned.
pub const TILE_SIZE: u32 = TILE_TEX - 2 * TILE_APRON;

/// Maps a tile's interior quad corner (`∈ [0, 1]`) to a UV coordinate in the
/// apron'd texture: `uv = corner * INTERIOR_UV_SCALE + INTERIOR_UV_BIAS`. The
/// compositor and presenter sample only the interior sub-rect; bilinear taps at
/// the interior edge then fall into the apron (neighbor content), not a clamp.
pub const INTERIOR_UV_SCALE: f32 = TILE_SIZE as f32 / TILE_TEX as f32;
pub const INTERIOR_UV_BIAS: f32 = TILE_APRON as f32 / TILE_TEX as f32;

/// Integer address of a tile on the infinite canvas.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileCoord {
    pub x: i32,
    pub y: i32,
}

impl TileCoord {
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    /// Canvas-space position of this tile's top-left corner, in pixels.
    pub fn origin(self) -> Vec2 {
        Vec2::new(
            self.x as f32 * TILE_SIZE as f32,
            self.y as f32 * TILE_SIZE as f32,
        )
    }
}

/// A pixel size (e.g. a render target's dimensions).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Extent2 {
    pub width: u32,
    pub height: u32,
}

impl Extent2 {
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

/// The pan/zoom/rotate/mirror transform applied when presenting the canvas to a
/// surface (§6.4). This is session state and is never historized.
///
/// Everything here is *how you are looking at* the painting rather than anything
/// about it, which is what makes turning the canvas and holding it up to a mirror
/// the same kind of act as panning: per-client, never logged, never sent, and
/// invisible to replay (§18.1.2). Two people sharing a drawing can
/// have it at different angles.
#[derive(Copy, Clone, Debug)]
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
    /// Size of the target surface, in pixels.
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

    /// Half the viewport, in screen px — the centre of the surface.
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
        self.rotation = radians.rem_euclid(TAU);
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
        self.set_rotation(-self.rotation);
        self.flip_h = !self.flip_h;
    }

    /// Scale the zoom by `factor` while keeping the canvas point under `anchor`
    /// (a screen-pixel position) pinned in place — cursor-anchored zoom.
    pub fn zoom_about(&mut self, anchor: Vec2, factor: f32) {
        let before = self.screen_to_canvas(anchor);
        self.zoom = (self.zoom * factor).clamp(Self::MIN_ZOOM, Self::MAX_ZOOM);
        let after = self.screen_to_canvas(anchor);
        self.center += before - after;
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

    /// An upright, unmirrored view has to go through the new matrix path *bit for
    /// bit* as it went through the old scale pair — the golden images are blessed
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
        // Upright, it is exactly the old half-viewport-over-zoom rect.
        let upright = view(Vec2::ZERO, 2.0, Extent2::new(400, 300));
        assert_eq!(
            upright.visible_bounds(),
            (Vec2::new(-100.0, -75.0), Vec2::new(100.0, 75.0))
        );
    }
}
