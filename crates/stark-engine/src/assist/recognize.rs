//! **Recognition**: which shape, if any, a raw pointer trace is (§6.9).
//!
//! It answers `None` freely — see the module note in [`super`] for why a false
//! positive costs more than a miss. A line that lands near an axis of a perspective
//! guide the artist has up is turned onto it (§20.6); this is the only part of assist
//! that knows a guide exists.
//!
//! `pub(super)`, here and on the three methods below, and that is what the split cost:
//! the feature tests live in `super` (see the module note there) and build a
//! perspective circle's plane for themselves, which is a use this type would not have
//! had from outside one file. Nothing outside `assist` can name it.

use super::AssistShape;
use stark_model::document::{AxisPencil, AxisPlane, Scaffold};
use stark_model::geom::{Ellipse, Vec2, principal_axis};
use std::f32::consts::TAU;

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
/// line's on purpose, and by a wide margin.
///
/// Two reasons, and they compound. The same hand movement is spread over a radius
/// rather than over a length, so the same absolute wobble is a larger fraction of it —
/// and a loop is a *longer* gesture than a drag of the same size, since going round
/// costs π times the diameter where crossing costs one, with the wrist reversing
/// direction twice on the way. A worst-sample bar on a signal that long is being asked
/// about the one moment the hand was least steady.
///
/// The bar that discriminates is still not the ellipse's: a shape that is not a loop of
/// *some* ellipse fails this by tens of percent, not by ones — the cardioid in the tests
/// misses it several times over — and the shapes near the edge of it are ellipses drawn
/// shakily. So the cost of the miss (draw it again, steadier) buys less here than it
/// does on a line, where a bowed stroke sits just the other side of the bar.
pub(super) const ELLIPSE_RESIDUAL: f32 = 0.15;

/// How far a recognized line may sit from a **guide axis** and still be read as one of
/// that axis's lines (§20.6), as a fraction of the line's own length.
///
/// Quoted the same way [`LINE_RESIDUAL`] is, and for the same reason — a bar in canvas
/// px would mean two things at two zoom levels — but wider, because it is not the same
/// question. `LINE_RESIDUAL` asks whether the hand drew a line at all, where a false
/// positive replaces a considered curve; this asks *which* line it meant, at a moment
/// when the artist has already dwelt to ask for an ideal one and has a grid up to
/// answer with. A hand aiming at a vanishing point a metre off the page is doing well
/// to be within a few degrees, and a fraction of the length is a fixed cone: 0.15 is
/// 8.6°, so three axes claim between a quarter and a third of the compass and a
/// deliberately diagonal line is still left alone.
///
/// The cone has to hold the *aim*, not the wrist. Where a stroke lands is decided at the
/// moment the hand commits to a direction, before the drag exists to be judged, and a
/// vanishing point off the page is not somewhere the eye can look while drawing — so the
/// error being priced is a guess at an angle, and a few degrees is a good guess.
const GUIDE_LINE_RESIDUAL: f32 = 0.15;

/// The same bar for a **perspective circle** (§20.7), as a fraction of the drawn loop's
/// mean radius: how far the trace may sit from the circle-on-a-plane that claims it.
///
/// Wider again than [`ELLIPSE_RESIDUAL`], and this is where the feature actually lives.
/// The hard part of drawing a circle in perspective is not its size or its place, which
/// a hand gets about right, but its **eccentricity and its tilt** — both fixed by where
/// on the plane it sits, and both exactly what the eye cannot judge. A bar that only
/// accepted loops already correct in the two would help nobody.
///
/// Measured, on ellipses a few hundred px across: 0.26 admits a loop about a quarter
/// too round, or leaning 9° out of the tilt its position calls for, or a lesser mix of
/// the two — and declines at around a third and 11°, so a loop drawn deliberately across
/// the grid stays the ellipse it is. Those are the figures for the plane that shows the
/// tilt most; a plane seen closer to face-on forgives 15° or 40°, and correctly, since a
/// rounder image has proportionally less tilt to be wrong about.
///
/// The cost of the bar being an isotropic fraction of the
/// mean radius is that it forgives eccentricity more readily than tilt on a strongly
/// foreshortened circle, which is the right way round: how *open* a near-edge-on
/// ellipse should be is genuinely hard to see, and which way it leans is not.
pub(super) const GUIDE_CIRCLE_RESIDUAL: f32 = 0.26;

/// Residual floor, in input tolerances. Without it the fractional bars above scale to
/// nothing on a small shape and no short stroke could ever snap, however straight the
/// hand held it — the residual cannot fall below what the device resolves.
const RESIDUAL_FLOOR: f32 = 4.0;

/// How far apart a trace's ends may be for it to be read as a loop, as a multiple of
/// the radius its own length implies (`length / 2π`). At 1.5 a stroke that goes four
/// fifths of the way round still closes and a three-quarter arc does not.
///
/// Neither side of that is free. Below it is what a hand actually does — closing a loop
/// is the last thing it does and the first it gets wrong (see [`weigh`]) — and the gap
/// costs the fit little, because the wedges the trace never reached are filled from the
/// estimate rather than guessed at: measured, a fifth-turn gap on a 400px loop still
/// lands the centre and the major axis within 3%. Above it the closed form starts to
/// describe an arc instead — a quarter-turn gap walks the centre 5% off — and, more to
/// the point, an arc is a shape somebody can mean.
pub(super) const CLOSE_GAP: f32 = 1.5;

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

/// The shape `trace` resembles, or `None` to leave the stroke as it was drawn.
///
/// `tolerance` is the gesture's input tolerance in canvas px — the device's own tolerance,
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
///
/// `pub(super)`, here and on the three methods below, and that is what the split cost:
/// the feature tests live in [`super`] (see the module note there) and build a
/// perspective circle's plane for themselves, which is a use this type would not have
/// had from inside one file. Nothing outside `assist` can name it.
pub(super) struct Frame {
    center: Vec2,
    radii: Vec2,
    /// The frame's rotation and its inverse, ready to apply.
    rotate: Vec2,
    unrotate: Vec2,
}

impl Frame {
    pub(super) fn new(center: Vec2, radii: Vec2, angle: f32) -> Self {
        Self {
            center,
            radii,
            rotate: Vec2::from_angle(angle),
            unrotate: Vec2::from_angle(-angle),
        }
    }

    pub(super) fn of(e: Ellipse) -> Self {
        Self::new(e.center, e.radii, e.angle)
    }

    /// The point of this ellipse at parameter `t`.
    pub(super) fn point(&self, t: f32) -> Vec2 {
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
    // right puts its synthetic points on the true ellipse — and the gap is at most a
    // fifth of the circle ([`CLOSE_GAP`]), so what was drawn always outvotes it.
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
    let cum = {
        let mut acc = 0.0;
        let mut out = Vec::with_capacity(trace.len());
        out.push(0.0);
        for w in trace.windows(2) {
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
                let k = cum.partition_point(|&c| c < want).clamp(1, trace.len() - 1);
                let (lo, hi) = (cum[k - 1], cum[k]);
                let u = if hi > lo {
                    (want - lo) / (hi - lo)
                } else {
                    0.0
                };
                trace[k - 1].lerp(trace[k], u)
            })
            .collect(),
    )
}

/// `x` folded into `(-π, π]` — the shortest way round.
pub(super) fn wrap_pi(x: f32) -> f32 {
    let mut x = x % TAU;
    if x > std::f32::consts::PI {
        x -= TAU;
    } else if x <= -std::f32::consts::PI {
        x += TAU;
    }
    x
}
