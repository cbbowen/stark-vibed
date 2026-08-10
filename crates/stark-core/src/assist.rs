//! Drag-and-hold drawing assist (§6.9).
//!
//! Drag out a rough line or a rough ellipse and then **hold the pen down without
//! moving it**: the stroke in flight snaps to the ideal shape it resembles, and the
//! rest of the same drag steers that shape rather than extending the path. Lift to
//! commit, exactly as for any other stroke.
//!
//! The feature attaches at the seam §18.1.3 named for it — *a path transform applied
//! between the fitter and the renderer*. Nothing downstream of
//! [`StrokeRecord::path`](crate::document::StrokeRecord) learns that assist exists: a
//! snapped stroke is still a list of [`ControlPoint`]s, so the renderer, the wire
//! format, the save file, replay and the goldens are untouched, and the assisted
//! stroke is undoable, replayable and collaborative for free. That is why this module
//! answers in control points instead of carrying a shape into the action log.
//!
//! Three separable pieces:
//!
//! - **Recognition** ([`recognize`]) — which shape, if any, the raw pointer trace is.
//!   It answers `None` freely: dwelling at the end of a stroke that is *not* a line or
//!   an ellipse has to leave that stroke exactly as it was drawn, because a false
//!   positive (a considered stroke silently replaced) costs far more than a miss
//!   (hold it a little straighter and try again). A line that lands near an axis of a
//!   perspective guide the artist has up is turned onto it (§20.6) — the only place
//!   this module knows a guide exists.
//! - **Adjustment** ([`AssistShape::adjust`]) — what the rest of the drag means. Both
//!   arms are derived from the shape *as recognized* plus the pointer's travel since,
//!   never from the previous frame's shape, so a long adjustment cannot accumulate
//!   drift — the same bargain the transform gesture makes (§16.6).
//! - **Realization** ([`AssistShape::to_path`]) — the shape as a fitted path, carrying
//!   the pen channels the stroke was actually drawn with. A snapped line keeps the
//!   pressure swell that was put into it rather than becoming a plotted vector, which
//!   is the whole difference between this and a shape tool.
//!
//! Everything the recognizer measures against is denominated in the gesture's **input
//! tolerance** — the device's own grain in canvas px, which the frontend already
//! states for the fit (see [`PathFitter::with_tolerance`](crate::path::PathFitter::with_tolerance)).
//! A "close enough to a line" bar fixed in canvas px would mean two different things
//! at two zoom levels, for the same reason the fit's own prices would.

use std::f32::consts::TAU;

use nalgebra::{Const, Dyn, OMatrix};

use crate::geom::{Ellipse, Vec2, principal_axis};
use crate::guides::{AxisPencil, AxisPlane, Scaffold};
use crate::path::{ControlPoint, FLATTEN_TOLERANCE, arc_profile, clamp_tilt, flatten, param_at};
use crate::spline::{CubicBSpline, Observations};

/// Reports below this many are not a shape, whatever they look like: a click, a
/// twitch, or a two-sample flick has no trace to recognize and would snap to noise.
const MIN_SAMPLES: usize = 8;

/// Smallest shape worth snapping, in input tolerances — roughly "the pointer has
/// travelled 32 device pixels". Below it the ideal shape and the drawn one differ by
/// less than the hand does.
const MIN_SIZE: f32 = 32.0;

/// How far the trace may sit from a candidate **line** at its worst, as a fraction of
/// the segment's own length.
///
/// The **worst** sample and not the RMS, which is the whole difference between a bar
/// that discriminates and one that does not. A hand's wobble along a straight drag is
/// noise — its worst is a small multiple of its RMS — while a curve somebody *meant*
/// deviates systematically, so averaging is exactly the operation that hides it. A
/// 300px stroke bowed 40px reads as 4% RMS, indistinguishable from a shaky straight
/// line, and as 9% at its worst, which is not close to anything.
const LINE_RESIDUAL: f32 = 0.035;

/// The same bar for an **ellipse**, as a fraction of its mean radius. Looser than the
/// line's on purpose: the same hand movement is spread over a radius rather than over
/// a length, so the same absolute wobble is a larger fraction of it.
const ELLIPSE_RESIDUAL: f32 = 0.10;

/// How far a recognized line may sit from a **guide axis** and still be read as one of
/// that axis's lines (§20.6), as a fraction of the line's own length.
///
/// Quoted the same way [`LINE_RESIDUAL`] is, and for the same reason — a bar in canvas
/// px would mean two things at two zoom levels — but wider, because it is not the same
/// question. `LINE_RESIDUAL` asks whether the hand drew a line at all, where a false
/// positive replaces a considered curve; this asks *which* line it meant, at a moment
/// when the artist has already dwelt to ask for an ideal one and has a grid up to
/// answer with. A hand aiming at a vanishing point a metre off the page is doing well
/// to be within a few degrees, and a fraction of the length is a fixed cone: 0.1 is
/// 5.7°, so three axes claim a sixth of the compass and a deliberately diagonal line
/// is left alone.
const GUIDE_LINE_RESIDUAL: f32 = 0.1;

/// The same bar for a **perspective circle** (§20.7), as a fraction of the drawn loop's
/// mean radius: how far the trace may sit from the circle-on-a-plane that claims it.
///
/// Wider again than [`ELLIPSE_RESIDUAL`], and this is where the feature actually lives.
/// The hard part of drawing a circle in perspective is not its size or its place, which
/// a hand gets about right, but its **eccentricity and its tilt** — both fixed by where
/// on the plane it sits, and both exactly what the eye cannot judge. A bar that only
/// accepted loops already correct in the two would help nobody.
///
/// Measured, on ellipses a few hundred px across: 0.18 admits a loop about a sixth too
/// round, or leaning 5° out of the tilt its position calls for, or a lesser mix of the
/// two — and declines at around a fifth and 8°, so a loop drawn deliberately across the
/// grid stays the ellipse it is. The cost of the bar being an isotropic fraction of the
/// mean radius is that it forgives eccentricity more readily than tilt on a strongly
/// foreshortened circle, which is the right way round: how *open* a near-edge-on
/// ellipse should be is genuinely hard to see, and which way it leans is not.
const GUIDE_CIRCLE_RESIDUAL: f32 = 0.18;

/// Residual floor, in input tolerances. Without it the fractional bars above scale to
/// nothing on a small shape and no short stroke could ever snap, however straight the
/// hand held it — the residual cannot fall below what the device resolves.
const RESIDUAL_FLOOR: f32 = 4.0;

/// How far apart a trace's ends may be for it to be read as a loop, as a multiple of
/// the radius its own length implies (`length / 2π`). At 1.2 a stroke that goes seven
/// eighths of the way round still closes and a three-quarter arc does not.
const CLOSE_GAP: f32 = 1.2;

/// Points the ellipse fit works over, resampled uniformly along the trace. Enough to
/// average out pointer jitter, few enough that four reweighting passes are free.
const ELLIPSE_SAMPLES: usize = 96;

/// Cap on the ellipse fit's reweighting passes (see [`fit_ellipse`]). It normally
/// settles in three or four; the cap is what bounds a loop left far enough open that
/// each pass is only nudging the estimate.
const ELLIPSE_PASSES: usize = 16;

/// Movement below which the ellipse fit is settled, relative to its own mean radius —
/// well under what the recognizer's bar can distinguish, so stopping there is free.
const ELLIPSE_SETTLED: f32 = 1e-3;

/// Wedges the ellipse's parameter circle is cut into when weighting samples by the
/// share of the shape they stand for (see [`weigh`]).
///
/// Coarse enough that the sparsest part of the loop still occupies its own wedges —
/// arc-length sampling puts few reports near the ends of the *major* axis of an
/// eccentric ellipse, and those are exactly the samples that fix its length.
const PARAM_BINS: usize = 32;

/// How far a snapped ellipse may sit off the true one, in canvas px — what fixes its
/// leg count.
///
/// The binding error is *not* the interior ripple (`r·Δ⁴/384` for a leg of `Δ`
/// radians, which is microscopic at any leg count worth using) but the **clamped end**:
/// the first leg of the path is deformed by the end condition, and its chord bows
/// `r·Δ²/8` off the arc it stands for. Placing the end control points by least squares
/// ([`realize`]) spreads that, leaving a little under a quarter of it — measured at
/// 1.69px for a 200px circle on 30° legs, against `r·Δ²/8 = 6.85`. So the leg count
/// solves `r·Δ²/24 ≤` this: the divisor is rounded *down* from the measured ratio, so
/// the number below is a bound rather than an average of one.
const ELLIPSE_ERROR: f32 = 0.4;

/// Legs a snapped ellipse is built from, whatever [`ELLIPSE_ERROR`] asks for. The floor
/// keeps a thumbnail ellipse from being a polygon; the ceiling keeps a canvas-wide one
/// bounded — and it is not a real cost, since the fitter itself would spend *more*
/// control points than this on a stroke that long ([`KNOT_SPACING`](crate::path::KNOT_SPACING)).
const MIN_LEGS: usize = 12;
const MAX_LEGS: usize = 96;

/// Extra legs drawn past a full turn, at each end.
///
/// A clamped cubic B-spline's **first span is a straight chord** — the clamp collapses
/// three of its four Bézier points onto the first control point — so an ellipse cannot
/// be exact at the two ends of the open path that draws it, however many control
/// points it gets. Overlapping the seam puts that flat sixth-of-a-leg *underneath* the
/// far end's correctly-curved interior instead of beside it, which is also what makes
/// a closed loop join without a notch.
const SEAM_OVERLAP: usize = 2;

/// Ceiling on a snapped path's control points. Nothing here needs to outgrow what the
/// fitter itself would spend on the same stroke.
const MAX_KNOTS: usize = 96;

/// Ideal-shape samples per control point that the realization fits against.
const TARGETS_PER_KNOT: usize = 4;

/// Shortest grip an ellipse adjustment will turn or scale about, in canvas px. Nearer
/// the centre than this the pointer's *direction* is noise, exactly as it is at the
/// start of the navigator's rotate drag (§18.1.2).
const MIN_GRIP: f32 = 1.0;

/// The ideal shape a held gesture snapped to (§6.9).
///
/// Held in the session for the rest of the gesture and thrown away at the end: what is
/// committed is the path it produces, not the shape itself. Post-hoc stroke editing
/// (§18.2.1) is where keeping it would start to pay, and that is the point at which to
/// decide it costs a wire-format change.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum AssistShape {
    /// A straight segment from `a` to `b`. `b` is the end the stroke finished at —
    /// the end under the pointer, and so the end a drag moves.
    Line {
        a: Vec2,
        b: Vec2,
        /// Whether the direction is a perspective guide's rather than the hand's
        /// (§20.6). One bit, because that is all the difference amounts to: the
        /// pencil's line through `a` *is* a straight canvas line, so a guided line
        /// is the same segment held to a direction it may not leave, and the rest
        /// of the drag runs **along** it instead of steering it off.
        on_axis: bool,
    },
    /// An ellipse, drawn as one closed loop.
    Ellipse {
        center: Vec2,
        /// Semi-axes along the ellipse's own frame, major first.
        radii: Vec2,
        /// Rotation of that frame, radians clockwise in canvas space.
        angle: f32,
        /// The perspective plane this is a **circle** on, if it is one (§20.7).
        ///
        /// The ellipse above stays the truth — it is what is drawn, and every
        /// consumer below reads it without knowing a guide exists. This says only
        /// which family the shape came out of and may not leave, which is all
        /// [`adjust`](Self::adjust) needs to go on steering it a circle. The same
        /// bit of bookkeeping [`Line::on_axis`](Self::Line::on_axis) is, for the
        /// same reason.
        plane: Option<AxisPlane>,
        /// Where on the ellipse the loop starts and ends, in the ellipse's **own**
        /// frame — so turning the shape carries the seam with it, and the join stays
        /// where the hand put it rather than sliding round under a rotation.
        phase: f32,
        /// Which way round the loop travels: `+1` or `-1` on [`phase`](Self::Ellipse::phase).
        /// Taken from the drawn trace so the pen channels ride the shape the way they
        /// were laid down.
        winding: f32,
    },
}

/// The shape `trace` resembles, or `None` to leave the stroke as it was drawn.
///
/// `tolerance` is the gesture's input tolerance in canvas px — the device's own grain,
/// which is the unit every bar here is quoted in. `guides` is what the grids on the
/// screen offer: axes a line may be aimed along (§20.6) and planes a loop may be a
/// circle on (§20.7). Pass an empty one and the answer is the hand's own shape.
///
/// A closed trace is offered to the ellipse fit first and **falls through** to the line
/// fit if it misses the bar, rather than being refused outright: that is one rule
/// instead of a decision tree, and it means a rough rectangle (closed, but no ellipse
/// and no line) simply does not snap.
///
/// The guide question is asked strictly *after* the stroke has been accepted as a
/// shape, never instead: recognition decides whether this is a line or an ellipse, and
/// only then does the grid get to say *which* line, or on which plane the ellipse is a
/// circle. A curve that happens to bow along a fan line is still a curve, and the two
/// bars cannot be traded against each other.
pub fn recognize(trace: &[Vec2], tolerance: f32, guides: &Scaffold) -> Option<AssistShape> {
    if trace.len() < MIN_SAMPLES {
        return None;
    }
    let min_size = MIN_SIZE * tolerance;
    let floor = RESIDUAL_FLOOR * tolerance;

    if is_closed(trace)
        && let Some(fit) = fit_ellipse(trace)
    {
        let size = 0.5 * (fit.radii.x + fit.radii.y);
        if size >= min_size && fit.worst <= (ELLIPSE_RESIDUAL * size).max(floor) {
            let bar = (GUIDE_CIRCLE_RESIDUAL * size).max(floor);
            return Some(
                snap_to_plane(trace, &guides.planes, bar).unwrap_or_else(|| fit.seamed(trace)),
            );
        }
    }

    let free = fit_line(trace)?;
    let size = free.a.distance(free.b);
    if size < min_size || free.worst > (LINE_RESIDUAL * size).max(floor) {
        return None;
    }
    let bar = (GUIDE_LINE_RESIDUAL * size).max(floor);
    Some(match snap_to_pencil(trace, free.a, &guides.axes, bar) {
        Some(on) => AssistShape::Line {
            a: on.a,
            b: on.b,
            on_axis: true,
        },
        None => AssistShape::Line {
            a: free.a,
            b: free.b,
            on_axis: false,
        },
    })
}

/// The guide axis the trace reads as, if any: among the pencils' lines through the
/// anchor, the one it strays least from, provided that is within `bar`.
///
/// Through the **anchor** and not through the trace's middle, because that is the point
/// the line is already anchored at — the start, placed deliberately (see [`fit_line`]).
/// Aligning about it turns the stroke onto the grid without sliding it off where the
/// hand began, which is also why this snaps to a *direction* rather than to the nearest
/// drawn fan line: the fans are a sampling of the pencil at whatever cell size the
/// slider says, and there is no reason a stroke's position should quantize to it.
fn snap_to_pencil(
    trace: &[Vec2],
    anchor: Vec2,
    pencils: &[AxisPencil],
    bar: f32,
) -> Option<LineFit> {
    pencils
        .iter()
        .filter_map(|pencil| pencil.through(anchor))
        .map(|dir| line_along(trace, anchor, dir))
        .filter(|fit| fit.worst <= bar)
        .min_by(|x, y| x.worst.total_cmp(&y.worst))
}

/// The **perspective circle** the trace reads as, if any (§20.7): among the planes on
/// the screen, the one whose circle explains the loop best, if that is within `bar`.
///
/// Each plane is asked in its own coordinates, where the question is not "which ellipse
/// is this" but "which *circle*" — and answering it is one call to the same
/// [`fit_ellipse`], because a circle is an ellipse whose radii agree and the measure
/// corrections that fit earned (speed, overshoot, undershoot) are exactly as necessary
/// on a pulled-back trace as on a drawn one. Its two radii are then collapsed to the
/// one of equal area: a loop drawn as a circle in perspective pulls back to a circle,
/// so they already agree, and where they do not it is the canvas score below that
/// declines rather than this choice.
///
/// **Scored on the canvas, never in the plane.** A plane's own metric is stretched by
/// the perspective — unboundedly, toward its vanishing line — so a residual measured
/// there would mean something different at every depth, and the far half of a loop
/// would count for orders of magnitude more than the near half. What decides is the
/// same residual the free ellipse was judged by, in the space the artist drew it in.
fn snap_to_plane(trace: &[Vec2], planes: &[AxisPlane], bar: f32) -> Option<AssistShape> {
    let mut best: Option<(f32, AssistShape)> = None;
    for plane in planes {
        let Some(flat) = plane.chart(trace).and_then(|pulled| fit_ellipse(&pulled)) else {
            continue;
        };
        let radius = (flat.radii.x * flat.radii.y).sqrt();
        let Some(seen) = plane.circle_seen(flat.center, radius) else {
            continue;
        };
        let frame = Frame::of(seen);
        let worst = trace.iter().map(|p| frame.distance(*p)).fold(0.0, f32::max);
        if worst > bar || best.as_ref().is_some_and(|(seen, _)| *seen <= worst) {
            continue;
        }
        let (phase, winding) = seam_of(&frame, trace);
        best = Some((
            worst,
            AssistShape::Ellipse {
                center: seen.center,
                radii: seen.radii,
                angle: seen.angle,
                phase,
                winding,
                plane: Some(*plane),
            },
        ));
    }
    best.map(|(_, shape)| shape)
}

impl AssistShape {
    /// The shape as the pointer's travel from `grip` to `pointer` leaves it.
    ///
    /// Always applied to the shape **as recognized**, with the grip where the pointer
    /// was when it snapped — so what a caller holds is one shape plus one displacement,
    /// and a minute of adjustment is no different from the same drag made at once.
    ///
    /// - A **line** moves the end the pointer is holding. By the pointer's delta rather
    ///   than to the pointer itself: snapping moved that end off the hand by up to the
    ///   fit residual, and driving it absolutely would jump it back on the first move.
    ///   A line that snapped to a **guide axis** keeps that direction and takes only
    ///   the component of the travel along it, so the end runs out and back along the
    ///   grid line for the rest of the drag. Adjustment preserves what recognition
    ///   established — the same bargain that keeps a drawn loop's eccentricity — and
    ///   an alignment that a single sideways nudge could break would not be one.
    /// - An **ellipse** turns and scales about its centre, so that the point being held
    ///   follows the pointer. Turning is what the feature is for; the scale rides along
    ///   because a one-pointer drag has two degrees of freedom and the radius is the
    ///   only other thing a hand at that position could mean. The eccentricity the
    ///   drawn loop established is preserved.
    /// - A **perspective circle** (§20.7) is sized, and only sized, in the plane it is
    ///   a circle on: turning a circle does nothing, so the turn the free arm spends a
    ///   degree of freedom on is not there to spend. Its eccentricity and tilt on the
    ///   canvas then follow from where on the plane it sits — which is the point of the
    ///   thing, and is why it cannot be done by scaling the drawn ellipse.
    pub fn adjust(self, grip: Vec2, pointer: Vec2) -> Self {
        match self {
            Self::Line { a, b, on_axis } => {
                let end = b + (pointer - grip);
                // The base direction, so the constraint is a fixed line and the result
                // stays a pure function of the total travel.
                let held = (b - a).try_normalize().filter(|_| on_axis);
                Self::Line {
                    a,
                    b: match held {
                        Some(u) => a + u * (end - a).dot(u),
                        None => end,
                    },
                    on_axis,
                }
            }
            Self::Ellipse {
                center,
                radii,
                angle,
                phase,
                winding,
                plane: Some(plane),
            } => {
                // A circle has no orientation, so there is nothing here for the drag to
                // turn: what is left of the ellipse's two degrees of freedom is the
                // size, taken in the plane's own coordinates about the centre it has
                // *there*. The canvas ellipse's centre is not the image of the circle's
                // centre, so scaling the drawn shape about it would leave the plane at
                // once.
                let sized = plane
                    .circle_behind(Ellipse::new(center, radii, angle))
                    .zip(plane.to_plane(grip).zip(plane.to_plane(pointer)))
                    .and_then(|((flat, radius), (from, to))| {
                        let l0 = (from - flat).length();
                        (l0 > 0.0)
                            .then(|| plane.circle_seen(flat, radius * (to - flat).length() / l0))
                            .flatten()
                    });
                match sized {
                    Some(seen) => Self::Ellipse {
                        center: seen.center,
                        radii: seen.radii,
                        angle: seen.angle,
                        phase,
                        winding,
                        plane: Some(plane),
                    },
                    // Nothing to draw at the far end of that drag — the circle has been
                    // pulled through its own vanishing line, or down to a point. Holding
                    // still is the honest answer, and dragging back recovers.
                    None => self,
                }
            }
            Self::Ellipse {
                center,
                radii,
                angle,
                phase,
                winding,
                plane: None,
            } => {
                let (from, to) = (grip - center, pointer - center);
                let (l0, l1) = (from.length(), to.length());
                if l0 < MIN_GRIP || l1 < MIN_GRIP {
                    return self;
                }
                Self::Ellipse {
                    center,
                    radii: radii * (l1 / l0),
                    angle: angle + from.perp_dot(to).atan2(from.dot(to)),
                    phase,
                    winding,
                    plane: None,
                }
            }
        }
    }

    /// The point a drag is deemed to be holding, for a gesture that has just snapped.
    ///
    /// Not the pointer's own position: it is the shape that moves from here on, so the
    /// grip has to be a point *of the shape*, or the first adjustment would apply the
    /// fit residual as a jump.
    pub fn grip(&self) -> Vec2 {
        match *self {
            Self::Line { b, .. } => b,
            // The seam: where the loop starts, which is where the hand is holding it.
            Self::Ellipse { .. } => self.at(0.0, 1.0),
        }
    }

    /// The ideal shape as a fitted path, carrying `pen`'s channels along it.
    ///
    /// `knots` is what the drawn stroke itself was fitted to: the shape never gets
    /// *fewer* control points than the stroke it replaces, because the pen channels
    /// ride the same polygon as the geometry and a pressure profile needs somewhere to
    /// live. Geometry decides the floor and the pen decides nothing else — the same
    /// split [`CubicBSpline::fit_channels`] draws.
    ///
    /// The geometry of a **line** is placed in closed form (any collinear control
    /// polygon draws exactly that line, so there is nothing to solve and nothing to
    /// round off). An **ellipse** is *fitted*: a clamped B-spline's ends are pinned to
    /// their own control points, so control points placed analytically on the ellipse
    /// leave an `O(Δ²)` bulge exactly at the seam, and a solve is what places the end
    /// rows to cancel it. The pen channels are fitted either way.
    pub fn to_path(&self, pen: &PenProfile, knots: usize) -> Vec<ControlPoint> {
        let (seed, targets, fit_geometry) = match *self {
            Self::Line { a, b, .. } => {
                let m = knots.clamp(2, MAX_KNOTS);
                let along = |f: f32| a.lerp(b, f);
                let seed = spread(m, along);
                let targets = spread(m * TARGETS_PER_KNOT, along);
                (seed, targets, false)
            }
            Self::Ellipse { radii, .. } => {
                // One leg per `delta` radians, from the ripple a leg leaves against
                // the true ellipse — then widened, if need be, to carry the pen.
                let want = knots.saturating_sub(2 * SEAM_OVERLAP + 1);
                let legs = leg_count(radii.max_element()).max(want).min(MAX_LEGS);
                let m = legs + 2 * SEAM_OVERLAP + 1;
                let delta = TAU / legs as f32;
                let span = (m - 1) as f32 * delta;
                // Control points sit on a slightly *larger* ellipse: a uniform cubic
                // B-spline runs inside its own control polygon by `(1 - cos Δ)/3`, and
                // undoing that is what makes the drawn curve the ellipse asked for
                // rather than a shrunken one. Exact for an ellipse as well as a circle
                // — the construction is an affine image of the circle case, and
                // B-splines commute with affine maps.
                let bulge = 3.0 / (2.0 + delta.cos());
                let start = -(SEAM_OVERLAP as f32) * delta;
                let seed = spread(m, |f| self.at(start + f * span, bulge));
                let targets = spread(m * TARGETS_PER_KNOT, |f| self.at(start + f * span, 1.0));
                (seed, targets, true)
            }
        };
        realize(&seed, &targets, pen, fit_geometry)
    }

    /// A point of the shape's own parameterization, scaled about the centre by `bulge`
    /// (1.0 for the shape itself). For a line the parameter is the fraction from `a`
    /// to `b`; for an ellipse it is radians past the seam, signed by the winding.
    fn at(&self, t: f32, bulge: f32) -> Vec2 {
        match *self {
            Self::Line { a, b, .. } => a.lerp(b, t),
            Self::Ellipse {
                center,
                radii,
                angle,
                phase,
                winding,
                ..
            } => {
                let u = phase + winding * t;
                let local = Vec2::new(radii.x * u.cos(), radii.y * u.sin()) * bulge;
                center + Vec2::from_angle(angle).rotate(local)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The pen channels
// ---------------------------------------------------------------------------

/// The pen channels of the stroke as drawn, against distance along it.
///
/// This is what keeps a snapped stroke *painted*. The geometry is replaced wholesale;
/// the pressure, tilt and time that were put into it are carried onto the ideal shape
/// at the same fraction of the way along, so a line snapped out of a stroke that swelled
/// in the middle still swells in the middle. Without it the feature would produce
/// vector art with a brush texture on it.
///
/// Read off the *fitted* path rather than off the raw reports, because that is already
/// the smoothed, de-jittered version of the same signal and it is what the stroke would
/// have been drawn with had nothing snapped.
pub struct PenProfile {
    /// `(fraction along, [pressure, tilt x, tilt y, time])`, in order.
    at: Vec<(f32, [f32; CHANNELS])>,
}

/// Pen channels riding the control polygon — the same four [`PathFitter`](crate::path::PathFitter)
/// solves for.
const CHANNELS: usize = 4;

impl PenProfile {
    /// The profile of a fitted path.
    pub fn of(path: &[ControlPoint]) -> Self {
        let poly = flatten(path, FLATTEN_TOLERANCE);
        let total = poly.last().map_or(0.0, |s| s.dist);
        let inv = if total > 1e-6 { 1.0 / total } else { 0.0 };
        Self {
            at: poly
                .iter()
                .map(|s| (s.dist * inv, [s.pressure, s.tilt.x, s.tilt.y, s.time]))
                .collect(),
        }
    }

    /// The channels `f` of the way along the drawn stroke, `f ∈ [0, 1]`.
    fn sample(&self, f: f32) -> [f32; CHANNELS] {
        let Some(&(_, first)) = self.at.first() else {
            return [1.0, 0.0, 0.0, 0.0];
        };
        let f = f.clamp(0.0, 1.0);
        let i = self.at.partition_point(|&(x, _)| x < f);
        if i == 0 {
            return first;
        }
        let Some(&(hi, b)) = self.at.get(i) else {
            return self.at[self.at.len() - 1].1;
        };
        let (lo, a) = self.at[i - 1];
        let u = if hi > lo { (f - lo) / (hi - lo) } else { 0.0 };
        std::array::from_fn(|d| a[d] + (b[d] - a[d]) * u)
    }
}

// ---------------------------------------------------------------------------
// Realization: an ideal shape as control points
// ---------------------------------------------------------------------------

/// `n` values of `f` spread over `[0, 1]` inclusive.
fn spread(n: usize, f: impl Fn(f32) -> Vec2) -> Vec<Vec2> {
    let last = n.saturating_sub(1).max(1) as f32;
    (0..n).map(|i| f(i as f32 / last)).collect()
}

/// Fit `seed`'s control polygon to `targets`, and the pen channels alongside it.
///
/// The parameters the targets are fitted at come from the seed curve's own **arc
/// profile**, which is the correction [`PathFitter`](crate::path::PathFitter) makes
/// for the same reason: a clamped B-spline is not parameterized by distance, and
/// assuming otherwise leaves a residual on data the curve could match exactly.
fn realize(
    seed: &[Vec2],
    targets: &[Vec2],
    pen: &PenProfile,
    fit_geometry: bool,
) -> Vec<ControlPoint> {
    let m = seed.len();
    let geom_seed = OMatrix::<f32, Dyn, Const<2>>::from_fn_generic(Dyn(m), Const::<2>, |j, d| {
        if d == 0 { seed[j].x } else { seed[j].y }
    });
    let Ok(spline) = CubicBSpline::from_control_points(geom_seed.clone()) else {
        // Fewer than two control points is not a curve; the caller's clamps rule it
        // out, and there is nothing to draw if they ever did not.
        return Vec::new();
    };
    let spans = spline.num_spans() as f32;
    let profile = arc_profile(&spline, &[]);
    let fractions = arc_fractions(targets);
    let ts: Vec<f32> = fractions
        .iter()
        .map(|&f| param_at(&profile, spans, f))
        .collect();

    let geom = if fit_geometry {
        let values: Vec<[f32; 2]> = targets.iter().map(|p| [p.x, p.y]).collect();
        spline.fit_channels(Observations::even(&ts, &values), 0, 0, &geom_seed, 0.0)
    } else {
        geom_seed
    };

    // Seeded from the profile at each control point's own share of the polygon, not
    // from zero: the solve's proximal ridge pulls towards its prior, and a prior of
    // zero biases every channel low by about a percent (see `spline`'s tests).
    let attr_seed =
        OMatrix::<f32, Dyn, Const<CHANNELS>>::from_fn_generic(Dyn(m), Const::<CHANNELS>, |j, d| {
            pen.sample(j as f32 / (m - 1).max(1) as f32)[d]
        });
    let values: Vec<[f32; CHANNELS]> = fractions.iter().map(|&f| pen.sample(f)).collect();
    let attr = spline.fit_channels(Observations::even(&ts, &values), 0, 0, &attr_seed, 0.0);

    (0..m)
        .map(|j| ControlPoint {
            pos: Vec2::new(geom[(j, 0)], geom[(j, 1)]),
            // The same clamps the fitter applies to its own solved channels: a control
            // value the data barely holds can overshoot the range a pen can report, and
            // pressure is a radius the renderer multiplies by without a ceiling.
            pressure: attr[(j, 0)].clamp(0.0, 1.0),
            tilt: clamp_tilt(Vec2::new(attr[(j, 1)], attr[(j, 2)])),
            time: attr[(j, 3)],
        })
        .collect()
}

/// Cumulative chord length along `pts`, normalized to `[0, 1]`.
fn arc_fractions(pts: &[Vec2]) -> Vec<f32> {
    let mut cum = Vec::with_capacity(pts.len());
    let mut acc = 0.0;
    let mut prev = pts.first().copied().unwrap_or(Vec2::ZERO);
    for p in pts {
        acc += prev.distance(*p);
        cum.push(acc);
        prev = *p;
    }
    match cum.last() {
        Some(&total) if total > 1e-6 => cum.iter().map(|c| c / total).collect(),
        _ => spread_scalar(pts.len()),
    }
}

/// `n` fractions spread over `[0, 1]` — the fallback when a polyline has no length.
fn spread_scalar(n: usize) -> Vec<f32> {
    let last = n.saturating_sub(1).max(1) as f32;
    (0..n).map(|i| i as f32 / last).collect()
}

/// Legs a full turn of an ellipse of this radius is drawn with — `r·Δ²/24 ≤`
/// [`ELLIPSE_ERROR`], which is what makes the count follow the shape's size instead of
/// being a number somebody picked.
fn leg_count(radius: f32) -> usize {
    if !(radius.is_finite() && radius > 0.0) {
        return MIN_LEGS;
    }
    let delta = (24.0 * ELLIPSE_ERROR / radius).sqrt();
    let legs = (TAU / delta).ceil();
    if legs.is_finite() {
        (legs as usize).clamp(MIN_LEGS, MAX_LEGS)
    } else {
        MAX_LEGS
    }
}

// ---------------------------------------------------------------------------
// Recognition
// ---------------------------------------------------------------------------

/// Whether the trace's ends are close enough for it to be read as a loop.
///
/// Measured against the radius the trace's own *length* implies (`length / 2π`) rather
/// than against a fixed distance, so the test says the same thing about a thumbnail
/// circle and a canvas-wide one.
fn is_closed(trace: &[Vec2]) -> bool {
    let (Some(first), Some(last)) = (trace.first(), trace.last()) else {
        return false;
    };
    let length: f32 = trace.windows(2).map(|w| w[0].distance(w[1])).sum();
    length > 0.0 && first.distance(*last) * TAU <= CLOSE_GAP * length
}

/// A line through the trace, and how far the trace sits from it.
struct LineFit {
    /// Where the stroke started — taken as drawn, not fitted.
    a: Vec2,
    b: Vec2,
    /// The furthest any sample sits from the line, perpendicular.
    worst: f32,
}

/// The line **through the first sample** that best fits the rest, minimizing
/// perpendicular distance.
///
/// Anchored rather than free, because the two ends of a drag are not the same kind of
/// thing: where a stroke *starts* is placed deliberately, with the pen at rest on the
/// point the hand chose, while where it ends is wherever the hand had got to. So the
/// start is taken as drawn and the fit spends all its freedom on the direction. That
/// also makes the residual honest — it is measured against the line that will actually
/// be drawn, rather than against a free one the snap then moves off.
///
/// Perpendicular distance and not vertical offset (total least squares, the principal
/// axis of the scatter *about that anchor*), because ordinary least squares would
/// answer differently for the same stroke drawn at a different angle — which for a
/// gesture on a canvas that can itself be rotated (§18.1.2) is not a fit at all.
///
/// The far end is the last sample projected onto the line: it is where the pointer is
/// holding, and so what the rest of the drag moves.
fn fit_line(trace: &[Vec2]) -> Option<LineFit> {
    if trace.len() < 2 {
        return None;
    }
    let anchor = trace[0];
    let (mut sxx, mut sxy, mut syy) = (0.0f32, 0.0f32, 0.0f32);
    for p in trace {
        let d = *p - anchor;
        sxx += d.x * d.x;
        sxy += d.x * d.y;
        syy += d.y * d.y;
    }
    let (_, _, dir) = principal_axis(sxx, sxy, syy);
    Some(line_along(trace, anchor, dir?))
}

/// The trace read as a line from `anchor` along `dir` (unit): where it ends, and how
/// far it strays at its worst.
///
/// The scoring half of [`fit_line`], with the direction supplied rather than solved
/// for — so a guide axis is judged by exactly the measure the free fit is judged by,
/// which is what makes the two residuals comparable at all (§20.6).
fn line_along(trace: &[Vec2], anchor: Vec2, dir: Vec2) -> LineFit {
    let last = trace.last().copied().unwrap_or(anchor);
    LineFit {
        a: anchor,
        b: anchor + dir * (last - anchor).dot(dir),
        worst: trace
            .iter()
            .map(|p| (*p - anchor).perp_dot(dir).abs())
            .fold(0.0, f32::max),
    }
}

/// An ellipse through the trace, and how far the trace sits from it.
struct EllipseFit {
    center: Vec2,
    radii: Vec2,
    angle: f32,
    /// The furthest any sample sits from the ellipse.
    worst: f32,
}

impl EllipseFit {
    /// The fit as a shape, with the seam and the winding taken from the trace — so the
    /// loop starts where the hand started it and travels the way the hand went.
    fn seamed(&self, trace: &[Vec2]) -> AssistShape {
        let frame = Frame::new(self.center, self.radii, self.angle);
        let (phase, winding) = seam_of(&frame, trace);
        AssistShape::Ellipse {
            center: self.center,
            radii: self.radii,
            angle: self.angle,
            phase,
            winding,
            plane: None,
        }
    }
}

/// Where a loop starts on `frame` and which way round it travels, read off the trace.
///
/// Shared by the free fit and the perspective one (§20.7), because the seam is a fact
/// about the *hand* — where it began and which way it went — and not about which
/// ellipse the recognizer settled on.
fn seam_of(frame: &Frame, trace: &[Vec2]) -> (f32, f32) {
    let ts: Vec<f32> = trace.iter().map(|p| frame.param(*p)).collect();
    let turned: f32 = ts.windows(2).map(|w| wrap_pi(w[1] - w[0])).sum();
    (
        ts.first().copied().unwrap_or(0.0),
        if turned < 0.0 { -1.0 } else { 1.0 },
    )
}

/// An ellipse's own frame: what turns a canvas point into the parameter it sits at.
struct Frame {
    center: Vec2,
    radii: Vec2,
    /// The frame's rotation and its inverse, ready to apply.
    rotate: Vec2,
    unrotate: Vec2,
}

impl Frame {
    fn new(center: Vec2, radii: Vec2, angle: f32) -> Self {
        Self {
            center,
            radii,
            rotate: Vec2::from_angle(angle),
            unrotate: Vec2::from_angle(-angle),
        }
    }

    fn of(e: Ellipse) -> Self {
        Self::new(e.center, e.radii, e.angle)
    }

    /// The point of this ellipse at parameter `t`.
    fn point(&self, t: f32) -> Vec2 {
        self.center
            + self
                .rotate
                .rotate(Vec2::new(self.radii.x * t.cos(), self.radii.y * t.sin()))
    }

    /// `p` in the ellipse's own axes.
    fn local(&self, p: Vec2) -> Vec2 {
        self.unrotate.rotate(p - self.center)
    }

    /// The ellipse parameter `p` sits at, in `(-π, π]`.
    fn param(&self, p: Vec2) -> f32 {
        let q = self.local(p);
        (q.y / self.radii.y.max(1e-6)).atan2(q.x / self.radii.x.max(1e-6))
    }

    /// Distance from `p` to the ellipse.
    ///
    /// The algebraic residual divided by its own gradient — the standard first-order
    /// correction, and exact in the limit of a point on the curve. Measuring *radially*
    /// instead (the obvious thing) over-reads by up to the axis ratio at the ends of the
    /// major axis, which would mean a 2:1 ellipse had to be drawn twice as accurately as
    /// a circle to snap at all.
    fn distance(&self, p: Vec2) -> f32 {
        let q = self.local(p);
        let (a2, b2) = (
            (self.radii.x * self.radii.x).max(1e-6),
            (self.radii.y * self.radii.y).max(1e-6),
        );
        let residual = q.x * q.x / a2 + q.y * q.y / b2 - 1.0;
        let grad = 2.0 * Vec2::new(q.x / a2, q.y / b2).length();
        if grad < 1e-9 {
            self.radii.min_element()
        } else {
            residual.abs() / grad
        }
    }
}

/// The ellipse the trace is a loop of, by **reweighted moments**.
///
/// The second moments of points spread uniformly in an ellipse's own parameter are
/// exactly `½·diag(a², b²)` in its own frame, so the covariance gives the shape in
/// closed form — but only for that measure. Pointer reports are spread by the hand's
/// *speed*, and resampling them uniformly by arc length does not fix it either: arc
/// length runs fastest at the ends of the minor axis, so an arc-length measure reads a
/// 2:1 ellipse as roughly 1.7:1.
///
/// So the measure is corrected instead of assumed. Each pass estimates the shape, reads
/// off the parameter every sample sits at, and reweights each sample by the *parameter*
/// it stands for — which makes the true ellipse a fixed point of the iteration rather
/// than something the first estimate has to land on. Four passes; the correspondence is
/// declared from the current estimate and never searched, which is the same discipline
/// [`PathFitter`](crate::path::PathFitter) applies to the stroke fit.
fn fit_ellipse(trace: &[Vec2]) -> Option<EllipseFit> {
    let pts = resample(trace, ELLIPSE_SAMPLES)?;
    // The first pass has no estimate to weigh against, so it takes the resampled points
    // as they come — arc length, which is biased, and only a seed.
    let mut weighted: Vec<(Vec2, f32)> = pts.iter().map(|p| (*p, 1.0)).collect();
    let mut fit: Option<Ellipse> = None;

    for _ in 0..ELLIPSE_PASSES {
        let next = moments(&weighted)?;
        let settled = fit.is_some_and(|prev| settled(prev, next));
        fit = Some(next);
        if settled {
            break;
        }
        weighted = weigh(&Frame::of(next), &pts);
    }

    let fit = fit?;
    let frame = Frame::of(fit);
    // Scored against what was actually drawn, not against the resampled loop: the
    // closing chord is a licence taken to fit with, never evidence that the fit is good.
    Some(EllipseFit {
        center: fit.center,
        radii: fit.radii,
        angle: fit.angle,
        worst: trace.iter().map(|p| frame.distance(*p)).fold(0.0, f32::max),
    })
}

/// Whether two successive estimates agree closely enough to stop, measured against the
/// ellipse's own scale so the test means the same thing at any size.
fn settled(a: Ellipse, b: Ellipse) -> bool {
    let scale = b.scale();
    let moved = a.center.distance(b.center)
        + a.radii.distance(b.radii)
        + wrap_pi(b.angle - a.angle).abs() * scale;
    moved <= ELLIPSE_SETTLED * scale
}

/// The ellipse whose second moments are the weighted samples' own.
///
/// Closed form, and exact for the measure it assumes: points spread uniformly in an
/// ellipse's parameter have covariance `½·diag(a², b²)` in its own frame, so the
/// principal axes of the scatter *are* the axes and its eigenvalues *are* the squared
/// semi-axes. Everything difficult about the fit is in earning that measure, which is
/// [`weigh`]'s job.
fn moments(weighted: &[(Vec2, f32)]) -> Option<Ellipse> {
    let total: f32 = weighted.iter().map(|(_, w)| *w).sum();
    if !(total.is_finite() && total > 0.0) {
        return None;
    }
    let inv = 1.0 / total;
    let center = weighted.iter().fold(Vec2::ZERO, |s, (p, w)| s + *p * *w) * inv;
    let (mut sxx, mut sxy, mut syy) = (0.0f32, 0.0f32, 0.0f32);
    for (p, w) in weighted {
        let d = *p - center;
        sxx += w * d.x * d.x;
        sxy += w * d.x * d.y;
        syy += w * d.y * d.y;
    }
    let (major, minor, dir) = principal_axis(sxx * inv, sxy * inv, syy * inv);
    let dir = dir?;
    let radii = Vec2::new((2.0 * major).max(0.0).sqrt(), (2.0 * minor).max(0.0).sqrt());
    if !(radii.is_finite() && radii.min_element() > 0.0) {
        return None;
    }
    Some(Ellipse::new(center, radii, dir.y.atan2(dir.x)))
}

/// How much of the ellipse's own parameter each sample stands for: the circle is cut
/// into [`PARAM_BINS`] equal wedges, and every **occupied** wedge is worth the same,
/// shared out among the samples that landed in it.
///
/// Coverage, not travel — and that distinction is the whole of it. Weighting by the
/// gap to each neighbour instead counts an arc *twice* when the hand goes over it
/// twice, which is precisely what closing a loop does: you come back past where you
/// started. A 6% overshoot on a 2:1 ellipse walked the estimated centre 78px off and
/// took the worst residual from 4px to 112px, so the shape was rejected for being
/// drawn the way loops are actually drawn.
///
/// An **empty** wedge is worth nothing, which is the other half: a loop that stopped
/// short leaves a gap in the parameter circle, and a gap has no data in it. Sharing its
/// weight out to the samples on either side would let two of them speak for a sixth of
/// the ellipse.
fn weigh(frame: &Frame, pts: &[Vec2]) -> Vec<(Vec2, f32)> {
    let mut occupants = [0u32; PARAM_BINS];
    let bins: Vec<usize> = pts
        .iter()
        .map(|p| {
            let f = (frame.param(*p) + std::f32::consts::PI) / TAU;
            let bin = (f * PARAM_BINS as f32) as usize;
            bin.min(PARAM_BINS - 1)
        })
        .collect();
    for &bin in &bins {
        occupants[bin] += 1;
    }
    let mut out: Vec<(Vec2, f32)> = pts
        .iter()
        .zip(&bins)
        .map(|(p, &bin)| (*p, 1.0 / occupants[bin] as f32))
        .collect();
    // A wedge the hand never reached is **filled from the estimate itself** — one point
    // on the current ellipse at the middle of each empty wedge.
    //
    // Closing the loop is the last thing a hand does and the first thing it gets wrong,
    // and stopping a few degrees short leaves the moments to describe an arc while the
    // closed-form they invert assumes a whole turn: an 8% short loop walked the centre
    // 27px off, which was enough to fail the bar at every eccentricity. Filling is
    // sound because the truth is still a fixed point — an estimate that is already
    // right puts its synthetic points on the true ellipse — and the wedge is at most an
    // eighth of the circle ([`CLOSE_GAP`]), so what was drawn always outvotes it.
    out.extend(
        occupants
            .iter()
            .enumerate()
            .filter(|&(_, &n)| n == 0)
            .map(|(bin, _)| {
                let t = (bin as f32 + 0.5) / PARAM_BINS as f32 * TAU - std::f32::consts::PI;
                (frame.point(t), 1.0)
            }),
    );
    out
}

/// `trace` resampled to `n` points spread uniformly by arc length.
///
/// Purely a decimation — a slow drag reports thousands of times and the fit has no use
/// for them — since [`weigh`] is what actually decides how much each sample
/// counts for. An earlier version closed the loop with a chord from the last sample
/// back to the first, to span the wedge a not-quite-closed loop leaves; that is
/// fabricated data, and once the weights measure *coverage* the wedge correctly counts
/// for nothing on its own.
fn resample(trace: &[Vec2], n: usize) -> Option<Vec<Vec2>> {
    if trace.len() < 2 || n < 3 {
        return None;
    }
    let ring: Vec<Vec2> = trace.to_vec();
    let cum = {
        let mut acc = 0.0;
        let mut out = Vec::with_capacity(ring.len());
        out.push(0.0);
        for w in ring.windows(2) {
            acc += w[0].distance(w[1]);
            out.push(acc);
        }
        out
    };
    let total = *cum.last()?;
    if !(total.is_finite() && total > 0.0) {
        return None;
    }
    Some(
        (0..n)
            .map(|i| {
                let want = total * i as f32 / (n - 1) as f32;
                let k = cum.partition_point(|&c| c < want).clamp(1, ring.len() - 1);
                let (lo, hi) = (cum[k - 1], cum[k]);
                let u = if hi > lo {
                    (want - lo) / (hi - lo)
                } else {
                    0.0
                };
                ring[k - 1].lerp(ring[k], u)
            })
            .collect(),
    )
}

/// `x` folded into `(-π, π]` — the shortest way round.
fn wrap_pi(x: f32) -> f32 {
    let mut x = x % TAU;
    if x > std::f32::consts::PI {
        x -= TAU;
    } else if x <= -std::f32::consts::PI {
        x += TAU;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1.0;

    /// The recognizer with no guides on the screen — what every test but the §20.6
    /// and §20.7 ones is asking about.
    fn free(trace: &[Vec2]) -> Option<AssistShape> {
        recognize(trace, TOL, &Scaffold::default())
    }

    /// A trace along an ideal shape, with `wobble` px of deterministic zig-zag across
    /// it — a stand-in for a hand.
    fn trace_of(n: usize, wobble: f32, f: impl Fn(f32) -> Vec2) -> Vec<Vec2> {
        (0..n)
            .map(|i| {
                let t = i as f32 / (n - 1) as f32;
                f(t) + Vec2::new(0.0, wobble * (i as f32 * 1.7).sin())
            })
            .collect()
    }

    fn line_trace(a: Vec2, b: Vec2, wobble: f32) -> Vec<Vec2> {
        trace_of(40, wobble, |t| a.lerp(b, t))
    }

    fn ellipse_trace(center: Vec2, radii: Vec2, angle: f32, wobble: f32) -> Vec<Vec2> {
        loop_trace(center, radii, angle, wobble, 0.0)
    }

    /// A hand-drawn loop. `turns` is how far past (or short of) one full turn it runs —
    /// nobody closes a loop exactly on the point they started from.
    fn loop_trace(center: Vec2, radii: Vec2, angle: f32, wobble: f32, turns: f32) -> Vec<Vec2> {
        trace_of(120, wobble, |t| {
            let u = t * (1.0 + turns) * TAU;
            center + Vec2::from_angle(angle).rotate(Vec2::new(radii.x * u.cos(), radii.y * u.sin()))
        })
    }

    #[test]
    fn a_rough_drag_is_a_line() {
        let (a, b) = (Vec2::new(10.0, 20.0), Vec2::new(310.0, 90.0));
        let shape = free(&line_trace(a, b, 3.0)).expect("a line");
        let AssistShape::Line {
            a: fa,
            b: fb,
            on_axis,
        } = shape
        else {
            panic!("recognized {shape:?}, not a line");
        };
        assert!(fa.distance(a) < 6.0, "start moved to {fa}");
        assert!(fb.distance(b) < 6.0, "end moved to {fb}");
        assert!(!on_axis, "there were no guides to be on the axis of");
    }

    #[test]
    fn a_rough_loop_is_an_ellipse() {
        let (center, radii) = (Vec2::new(100.0, -40.0), Vec2::new(120.0, 60.0));
        let shape = free(&ellipse_trace(center, radii, 0.4, 3.0)).expect("an ellipse");
        let AssistShape::Ellipse {
            center: c,
            radii: r,
            angle,
            ..
        } = shape
        else {
            panic!("recognized {shape:?}, not an ellipse");
        };
        assert!(c.distance(center) < 8.0, "centre moved to {c}");
        // The eccentricity is the thing the naive moment fit gets wrong — a 2:1
        // ellipse read through an arc-length measure comes back nearer 1.7:1.
        let ratio = r.x / r.y;
        assert!(
            (ratio - 2.0).abs() < 0.2,
            "axis ratio {ratio} (radii {r}), expected 2:1"
        );
        assert!((wrap_pi(angle - 0.4)).abs() < 0.15, "angle {angle}");
    }

    /// Eccentricity is not something the recognizer may be shy about: a long thin loop
    /// is a *more* deliberate thing to draw than a circle, not a less confident one.
    ///
    /// This is the case a naive moment fit cannot reach at all — read through an
    /// arc-length measure a 2:1 ellipse comes back nearer 1.7:1, and the error grows
    /// with the axis ratio until the shape misses the bar for being what it is.
    #[test]
    fn a_long_thin_loop_is_still_an_ellipse() {
        for ratio in [2.0f32, 4.0, 8.0] {
            let radii = Vec2::new(400.0, 400.0 / ratio);
            let shape = free(&ellipse_trace(Vec2::ZERO, radii, -1.2, 4.0))
                .unwrap_or_else(|| panic!("{ratio}:1 declined"));
            let AssistShape::Ellipse { radii: r, .. } = shape else {
                panic!("{ratio}:1 recognized as {shape:?}");
            };
            assert!(
                (r.x / r.y - ratio).abs() < 0.15 * ratio,
                "{ratio}:1 read as {:.2}:1 (radii {r})",
                r.x / r.y
            );
        }
    }

    /// **Where a loop closes is where a hand is least accurate**, so neither running
    /// past the start nor stopping short of it may cost the shape.
    ///
    /// Both used to. Overshoot double-counted the re-traversed wedge and walked the
    /// centre 78px off a 400px ellipse; undershoot left the closed-form inversion
    /// describing an arc while assuming a whole turn. See [`weigh`].
    #[test]
    fn a_loop_that_misses_its_own_start_still_snaps() {
        for turns in [-0.10f32, -0.04, 0.0, 0.04, 0.10] {
            for ratio in [1.0f32, 4.0] {
                let radii = Vec2::new(400.0, 400.0 / ratio);
                let trace = loop_trace(Vec2::ZERO, radii, 0.6, 4.0, turns);
                let shape =
                    free(&trace).unwrap_or_else(|| panic!("{ratio}:1 at {turns:+} turns declined"));
                let AssistShape::Ellipse {
                    center, radii: r, ..
                } = shape
                else {
                    panic!("recognized as {shape:?}");
                };
                assert!(
                    center.length() < 0.05 * radii.x,
                    "{ratio}:1 at {turns:+} turns put the centre at {center}"
                );
                assert!(
                    (r.x - radii.x).abs() < 0.08 * radii.x,
                    "{ratio}:1 at {turns:+} turns read the major axis as {}",
                    r.x
                );
            }
        }
    }

    /// The start of a drag is placed deliberately; the end is wherever the hand had got
    /// to. So a snapped line keeps the first sample exactly and fits only its direction.
    #[test]
    fn a_snapped_line_starts_where_the_stroke_started() {
        let (a, b) = (Vec2::new(10.0, 20.0), Vec2::new(310.0, 90.0));
        let trace = line_trace(a, b, 3.0);
        let AssistShape::Line { a: start, .. } = free(&trace).expect("a line") else {
            panic!("not a line");
        };
        assert_eq!(start, trace[0], "the drawn start was moved to {start}");
    }

    /// The bar has to *decline*. A considered curve that happens to end near where it
    /// started must survive being held.
    #[test]
    fn a_deliberate_curve_does_not_snap() {
        // A cardioid-ish loop: closed, but no ellipse.
        let trace = trace_of(80, 0.0, |t| {
            let u = t * TAU;
            let r = 100.0 * (1.0 + 0.45 * (2.0 * u).cos());
            Vec2::new(r * u.cos(), r * u.sin())
        });
        assert_eq!(free(&trace), None);
    }

    #[test]
    fn a_bowed_stroke_is_not_a_line() {
        // 300px across with a 40px sagitta — a curve somebody meant.
        let trace = trace_of(40, 0.0, |t| Vec2::new(300.0 * t, 160.0 * t * (1.0 - t)));
        assert_eq!(free(&trace), None);
    }

    #[test]
    fn a_twitch_is_not_a_shape() {
        assert_eq!(free(&[Vec2::ZERO; 4]), None);
        let tiny = line_trace(Vec2::ZERO, Vec2::new(6.0, 0.0), 0.0);
        assert_eq!(free(&tiny), None);
    }

    /// The realized path has to *be* the shape, not merely resemble it — including at
    /// the clamped ends, which is what the ellipse's least-squares placement buys.
    #[test]
    fn a_snapped_line_is_straight() {
        let (a, b) = (Vec2::new(-50.0, 30.0), Vec2::new(250.0, 130.0));
        let pen = PenProfile::of(&[ControlPoint::at(a), ControlPoint::at(b)]);
        let path = AssistShape::Line {
            a,
            b,
            on_axis: false,
        }
        .to_path(&pen, 12);
        let poly = flatten(&path, FLATTEN_TOLERANCE);
        let dir = (b - a).normalize();
        let worst = poly
            .iter()
            .map(|s| (s.pos - a).perp_dot(dir).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 0.02, "drifted {worst}px off the line");
    }

    #[test]
    fn a_snapped_ellipse_is_round() {
        for radius in [40.0f32, 200.0, 1200.0] {
            let shape = AssistShape::Ellipse {
                center: Vec2::new(7.0, -13.0),
                radii: Vec2::splat(radius),
                angle: 0.0,
                phase: 0.3,
                winding: 1.0,
                plane: None,
            };
            let pen = PenProfile::of(&[ControlPoint::at(Vec2::ZERO), ControlPoint::at(Vec2::X)]);
            let path = shape.to_path(&pen, 8);
            let poly = flatten(&path, FLATTEN_TOLERANCE);
            let worst = poly
                .iter()
                .map(|s| (s.pos.distance(Vec2::new(7.0, -13.0)) - radius).abs())
                .fold(0.0f32, f32::max);
            assert!(worst <= ELLIPSE_ERROR, "radius {radius}: off by {worst}px");
        }
    }

    /// The pen channels ride the ideal shape rather than being flattened out of it.
    #[test]
    fn a_snapped_stroke_keeps_its_pressure() {
        let drawn: Vec<ControlPoint> = (0..9)
            .map(|i| {
                let t = i as f32 / 8.0;
                ControlPoint {
                    pos: Vec2::new(t * 300.0, 0.0),
                    // A swell: light at both ends, full in the middle.
                    pressure: 0.15 + 0.85 * (t * std::f32::consts::PI).sin(),
                    tilt: Vec2::ZERO,
                    time: t,
                }
            })
            .collect();
        let pen = PenProfile::of(&drawn);
        let path = AssistShape::Line {
            a: Vec2::ZERO,
            b: Vec2::new(300.0, 0.0),
            on_axis: false,
        }
        .to_path(&pen, drawn.len());
        let mid = path[path.len() / 2].pressure;
        let ends = path[0].pressure.max(path[path.len() - 1].pressure);
        assert!(mid > 0.8, "the swell was flattened out (mid {mid})");
        assert!(ends < 0.4, "the light ends were filled in (ends {ends})");
    }

    /// Adjustment is always measured from the shape as recognized, so the same total
    /// travel means the same thing however it is broken up.
    #[test]
    fn adjustment_does_not_accumulate() {
        let shape = AssistShape::Ellipse {
            center: Vec2::ZERO,
            radii: Vec2::new(100.0, 50.0),
            angle: 0.0,
            phase: 0.0,
            winding: 1.0,
            plane: None,
        };
        let grip = shape.grip();
        let target = Vec2::new(0.0, 140.0);
        let direct = shape.adjust(grip, target);
        let stepped = (1..=8).fold(shape, |_, i| {
            shape.adjust(grip, grip.lerp(target, i as f32 / 8.0))
        });
        assert_eq!(direct, stepped);
    }

    #[test]
    fn a_line_adjustment_moves_the_held_end() {
        let shape = AssistShape::Line {
            a: Vec2::ZERO,
            b: Vec2::new(100.0, 0.0),
            on_axis: false,
        };
        let moved = shape.adjust(shape.grip(), Vec2::new(100.0, 60.0));
        assert_eq!(
            moved,
            AssistShape::Line {
                a: Vec2::ZERO,
                b: Vec2::new(100.0, 60.0),
                on_axis: false,
            }
        );
    }

    // --- lines on a guide axis (§20.6) -------------------------------------

    use crate::guides::PerspectiveGuide;

    /// A guide in general position: three finite vanishing points, none of its axes
    /// level, so nothing here can pass by accidentally agreeing with the canvas.
    fn guide() -> PerspectiveGuide {
        PerspectiveGuide {
            center: Vec2::new(120.0, -60.0),
            focal: 700.0,
            rotation: glam::Quat::from_rotation_z(0.2)
                * glam::Quat::from_rotation_x(0.3)
                * glam::Quat::from_rotation_y(0.55),
            ..Default::default()
        }
    }

    /// A trace drawn from `a`, `len` px long, `off` radians away from the guide line of
    /// `axis` through `a` — the shape of every question here: how far off may a hand be?
    fn aimed(
        g: &PerspectiveGuide,
        axis: usize,
        a: Vec2,
        len: f32,
        off: f32,
        wobble: f32,
    ) -> Vec<Vec2> {
        let u = g.pencils()[axis]
            .expect("axis shown")
            .through(a)
            .expect("a line through a");
        let dir = Vec2::from_angle(off).rotate(u);
        line_trace(a, a + dir * len, wobble)
    }

    fn as_line(shape: AssistShape) -> (Vec2, Vec2, bool) {
        match shape {
            AssistShape::Line { a, b, on_axis } => (a, b, on_axis),
            other => panic!("recognized {other:?}, not a line"),
        }
    }

    /// The feature itself: a line drawn roughly toward a vanishing point comes back
    /// aimed exactly at it, *from where the hand started it* — turned onto the grid,
    /// not moved onto it.
    #[test]
    fn a_line_aimed_near_an_axis_snaps_onto_it() {
        let g = guide();
        let vps = g.scene().vps;
        let up = Scaffold::of(std::slice::from_ref(&g));
        let start = Vec2::new(-260.0, 210.0);
        for (axis, vp) in vps.iter().enumerate() {
            let vp = vp.expect("3-point: all finite");
            let trace = aimed(&g, axis, start, 420.0, 0.05, 2.5);
            let (a, b, on_axis) = as_line(recognize(&trace, TOL, &up).expect("a line"));
            assert!(on_axis, "axis {axis} was not taken");
            assert_eq!(a, start, "the drawn start moved to {a}");
            let (u, to_vp) = ((b - a).normalize(), (vp - a).normalize());
            assert!(
                u.perp_dot(to_vp).abs() < 1e-3,
                "axis {axis}: the snapped line points {u} and the VP is at {to_vp}"
            );
        }
    }

    /// ...and it has to decline. A line drawn deliberately across the grid is a line
    /// the artist meant, and the grid does not get to claim it.
    #[test]
    fn a_line_across_the_grid_keeps_its_own_direction() {
        let g = guide();
        let up = Scaffold::of(std::slice::from_ref(&g));
        let start = Vec2::new(-260.0, 210.0);
        // A fifth of a turn off the X axis — and, in this pose, well away from the
        // other two as well.
        let trace = aimed(&g, 0, start, 420.0, 0.6, 2.5);
        let (a, b, on_axis) = as_line(recognize(&trace, TOL, &up).expect("a line"));
        assert!(!on_axis, "a deliberate diagonal was pulled onto an axis");
        let drawn = (trace[trace.len() - 1] - trace[0]).normalize();
        assert!((b - a).normalize().perp_dot(drawn).abs() < 0.02);
    }

    /// The snap is one question about direction, so it cannot depend on how the trace
    /// was *scaled*: the bar is a fraction of the line's own length, which is the same
    /// cone for a short stroke and a long one.
    #[test]
    fn the_snap_window_is_an_angle() {
        let g = guide();
        let up = Scaffold::of(std::slice::from_ref(&g));
        let start = Vec2::new(40.0, 320.0);
        for len in [120.0f32, 400.0, 1600.0] {
            // Inside the cone at every length…
            let near = aimed(&g, 2, start, len, 0.04, 0.0);
            let (_, _, on) = as_line(recognize(&near, TOL, &up).expect("a line"));
            assert!(on, "{len}px missed an axis 2.3° away");
            // …and outside it at every length.
            let far = aimed(&g, 2, start, len, 0.25, 0.0);
            let (_, _, on) = as_line(recognize(&far, TOL, &up).expect("a line"));
            assert!(!on, "{len}px was claimed by an axis 14° away");
        }
    }

    /// What the artist cannot see cannot bend a line: the same trace, with the guide's
    /// eye shut, is the hand's own.
    #[test]
    fn a_hidden_guide_does_not_snap() {
        let mut g = guide();
        let start = Vec2::new(-260.0, 210.0);
        let trace = aimed(&g, 1, start, 420.0, 0.05, 2.5);
        assert!(
            as_line(
                recognize(&trace, TOL, &Scaffold::of(std::slice::from_ref(&g))).expect("a line")
            )
            .2
        );

        g.visible = false;
        assert!(
            !as_line(
                recognize(&trace, TOL, &Scaffold::of(std::slice::from_ref(&g))).expect("a line")
            )
            .2
        );
    }

    /// A curve is still a curve. The grid may only choose *which* line a line is, and
    /// never promote something that was not one — the two bars are not interchangeable.
    #[test]
    fn a_bowed_stroke_near_an_axis_is_still_not_a_line() {
        let g = guide();
        let start = Vec2::new(-260.0, 210.0);
        let u = g.pencils()[0].unwrap().through(start).unwrap();
        // 300px along the axis with a 40px sagitta across it — the §6.9 curve, laid on
        // the grid.
        let trace = trace_of(40, 0.0, |t| {
            start + u * (300.0 * t) + u.perp() * (160.0 * t * (1.0 - t))
        });
        assert_eq!(
            recognize(&trace, TOL, &Scaffold::of(std::slice::from_ref(&g))),
            None
        );
    }

    /// Steering a line that took an axis runs it out along that axis: the pointer's
    /// travel across the line is dropped, and what it means is where along the line the
    /// end lands. An alignment a sideways nudge could break would not be one.
    #[test]
    fn an_axis_line_is_steered_along_its_axis() {
        let (a, b) = (Vec2::new(10.0, 10.0), Vec2::new(210.0, 110.0));
        let shape = AssistShape::Line {
            a,
            b,
            on_axis: true,
        };
        let u = (b - a).normalize();
        // A pointer that has wandered a long way off the line, and some way along it.
        let target = b + u * 90.0 + u.perp() * 140.0;
        let (a2, b2, on) = as_line(shape.adjust(shape.grip(), target));
        assert_eq!((a2, on), (a, true));
        assert!(
            (b2 - a).perp_dot(u).abs() < 1e-3,
            "the end left the axis, at {b2}"
        );
        assert!(
            ((b2 - a).dot(u) - ((b - a).length() + 90.0)).abs() < 1e-3,
            "the end did not run out along the axis, at {b2}"
        );

        // And, like every adjustment, it is a function of the total travel (§16.6).
        let stepped = (1..=8).fold(shape, |_, i| {
            shape.adjust(shape.grip(), shape.grip().lerp(target, i as f32 / 8.0))
        });
        assert_eq!(shape.adjust(shape.grip(), target), stepped);
    }

    // --- circles on a plane (§20.7) ----------------------------------------

    /// Where the §20.7 tests put their circles: near the foot of each plane's normal,
    /// which is the part of a plane that is squarely in front of the eye. Out toward a
    /// plane's own vanishing line a circle straddles it and has no bounded image at
    /// all — a real answer (`circle_seen` declines), and not the case under test.
    const ON_PLANE: Vec2 = Vec2::new(0.05, -0.04);

    /// Mean radius, in canvas px, of the *image* every §20.7 test draws around.
    ///
    /// Sized on the canvas rather than in the plane, because a plane radius is not a
    /// comparable quantity across three planes lying at three depths: the same 0.12 in
    /// plane units images to 130px on one of this guide's planes and to 1500px on
    /// another, and a residual bar quoted as a fraction of the shape means something
    /// quite different at those two sizes. What a hand draws is a few hundred px
    /// across, so that is what these ask about.
    const DRAWN_SIZE: f32 = 180.0;

    fn as_ellipse(shape: AssistShape) -> (Vec2, Vec2, f32, Option<AxisPlane>) {
        match shape {
            AssistShape::Ellipse {
                center,
                radii,
                angle,
                plane,
                ..
            } => (center, radii, angle, plane),
            other => panic!("recognized {other:?}, not an ellipse"),
        }
    }

    /// The circle at [`ON_PLANE`] on plane `k` whose image is [`DRAWN_SIZE`] across the
    /// mean: the plane, its radius there, and the ellipse it is seen as.
    fn perspective_circle(g: &PerspectiveGuide, k: usize) -> (AxisPlane, f32, Ellipse) {
        let plane = g.planes()[k].expect("plane shown");
        let probe = plane.circle_seen(ON_PLANE, 0.02).expect("a bounded image");
        let radius = 0.02 * DRAWN_SIZE / probe.scale();
        let seen = plane
            .circle_seen(ON_PLANE, radius)
            .expect("a bounded image");
        (plane, radius, seen)
    }

    /// A hand-drawn loop around that circle's image, with its eccentricity stretched by
    /// `wrong` and its tilt turned by `tilt` — exactly the pair of things a hand gets
    /// wrong about a circle in perspective, and the two the snap exists to fix.
    fn perspective_loop(g: &PerspectiveGuide, k: usize, wrong: f32, tilt: f32) -> Vec<Vec2> {
        let (
            _,
            _,
            Ellipse {
                center,
                radii,
                angle,
            },
        ) = perspective_circle(g, k);
        loop_trace(
            center,
            Vec2::new(radii.x, radii.y * wrong),
            angle + tilt,
            2.0,
            0.0,
        )
    }

    /// The feature: a loop drawn roughly where a circle on the grid would be comes back
    /// as *that circle*, not as the ellipse the hand actually managed.
    #[test]
    fn a_loop_drawn_in_perspective_snaps_to_a_circle() {
        let g = guide();
        let up = Scaffold::of(std::slice::from_ref(&g));
        for k in 0..3 {
            let (plane, radius, _) = perspective_circle(&g, k);
            // A tenth too round and leaning 3° off — a hand that saw it about right.
            let trace = perspective_loop(&g, k, 1.1, 0.05);
            let (center, radii, angle, on) =
                as_ellipse(recognize(&trace, TOL, &up).expect("an ellipse"));
            let on = on.unwrap_or_else(|| panic!("plane {k} was not taken"));
            assert_eq!(Some(plane), Some(on), "plane {k}: another plane claimed it");

            // ...and it is a circle *there*: at the place and the size it was drawn at.
            let (flat, r) = on
                .circle_behind(Ellipse::new(center, radii, angle))
                .expect("a circle behind it");
            assert!(
                flat.distance(ON_PLANE) < 0.15 * radius,
                "plane {k}: the circle sits at {flat}, not {ON_PLANE}"
            );
            assert!(
                (r - radius).abs() < 0.2 * radius,
                "plane {k}: radius {r}, drawn around {radius}"
            );
        }
    }

    /// The snapped shape has to *be* the perspective circle, not merely resemble one:
    /// its whole outline pulls back to one constant radius. That is the statement the
    /// canvas ellipse on its own cannot make.
    #[test]
    fn the_snapped_loop_is_round_in_its_plane() {
        let g = guide();
        let trace = perspective_loop(&g, 2, 1.1, -0.05);
        let (center, radii, angle, plane) = as_ellipse(
            recognize(&trace, TOL, &Scaffold::of(std::slice::from_ref(&g))).expect("an ellipse"),
        );
        let plane = plane.expect("a perspective circle");
        let (flat, r) = plane
            .circle_behind(Ellipse::new(center, radii, angle))
            .expect("a circle");

        let frame = Frame::new(center, radii, angle);
        let worst = (0..64)
            .map(|i| {
                let p = frame.point(i as f32 / 64.0 * TAU);
                let q = plane.to_plane(p).expect("on the plane");
                (q.distance(flat) - r).abs()
            })
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-3 * r, "the outline strays {worst} off round");
    }

    /// And it has to decline. A loop no plane would put there is the ellipse it was
    /// drawn as — an ellipse is a shape somebody can mean.
    #[test]
    fn a_loop_no_plane_would_put_there_stays_an_ellipse() {
        let g = guide();
        // Twice as eccentric as the plane calls for, and turned 40° out of its tilt.
        let trace = perspective_loop(&g, 2, 1.6, 0.5);
        let (_, _, _, plane) = as_ellipse(
            recognize(&trace, TOL, &Scaffold::of(std::slice::from_ref(&g))).expect("an ellipse"),
        );
        assert!(plane.is_none(), "a deliberate ellipse was made a circle");
    }

    /// What the artist cannot see cannot reshape a loop — and a plane is switched off
    /// directly, being the thing the bar's chips name.
    #[test]
    fn an_unshown_plane_does_not_snap() {
        let mut g = guide();
        let trace = perspective_loop(&g, 2, 1.1, -0.05);
        let claimed = |g: &PerspectiveGuide| {
            as_ellipse(
                recognize(&trace, TOL, &Scaffold::of(std::slice::from_ref(g))).expect("an ellipse"),
            )
            .3
            .is_some()
        };
        assert!(claimed(&g));

        // The loop is a circle on pair 2, and only that plane's own flag governs
        // it: switching off the other two leaves it claimed, switching off its
        // own takes it.
        g.pairs = [false, false, true];
        assert!(claimed(&g), "a plane went with its neighbours");
        g.pairs = [true, true, false];
        assert!(!claimed(&g), "a plane survived being switched off");
        g.pairs = [true; 3];
        g.visible = false;
        assert!(!claimed(&g), "a plane survived the guide's eye");
    }

    /// Steering a perspective circle sizes it *in its plane* and leaves it there, so
    /// the canvas ellipse's eccentricity and tilt go on following the grid rather than
    /// the hand — which is the whole reason the plane is carried at all.
    ///
    /// The pointer is put on the circle half again as wide, at a place of its own
    /// choosing: the drag is therefore mostly *across* the shape, and the answer has to
    /// be exactly the circle half again as wide, because sideways travel means nothing
    /// to a shape with no orientation to turn.
    #[test]
    fn a_perspective_circle_is_steered_in_its_plane() {
        let g = guide();
        let trace = perspective_loop(&g, 2, 1.08, 0.04);
        let shape =
            recognize(&trace, TOL, &Scaffold::of(std::slice::from_ref(&g))).expect("an ellipse");
        let (center, radii, angle, plane) = as_ellipse(shape);
        let plane = plane.expect("a perspective circle");
        let (flat, r) = plane
            .circle_behind(Ellipse::new(center, radii, angle))
            .expect("a circle behind it");

        let wider = plane.circle_seen(flat, r * 1.5).expect("a wider circle");
        let target = Frame::of(wider).point(1.0);
        let (center, radii, angle, kept) = as_ellipse(shape.adjust(shape.grip(), target));
        assert_eq!(kept, Some(plane), "the circle left its plane");

        let (moved, wider) = plane
            .circle_behind(Ellipse::new(center, radii, angle))
            .expect("still a circle");
        assert!(
            moved.distance(flat) < 1e-3 * r,
            "the centre wandered from {flat} to {moved}"
        );
        assert!(
            (wider - 1.5 * r).abs() < 1e-3 * r,
            "radius {wider}, expected {}",
            1.5 * r
        );

        // ...and it is a function of the total travel, like every adjustment (§16.6).
        let stepped = (1..=8).fold(shape, |_, i| {
            shape.adjust(shape.grip(), shape.grip().lerp(target, i as f32 / 8.0))
        });
        assert_eq!(shape.adjust(shape.grip(), target), stepped);
    }
}
