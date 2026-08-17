//! Drag-and-hold drawing assist, through the gesture it is actually made with
//! (§6.9).
//!
//! [`crate::assist`](stark_engine::assist)'s own tests cover the geometry — which shape
//! a trace is, and how faithfully the ideal one is realized. What is tested here is the
//! part that only exists once a [`Session`] is holding a gesture: that a hold replaces
//! the path in flight, that the rest of the same drag *steers* rather than extends,
//! that the release commits what was previewed, and that a stroke the recognizer
//! declines is left exactly as it was drawn. The guide snap (§20.6) is here for the
//! same reason: the axes a line may take come off the session's own guide list, so
//! whether the feature is wired up at all is a question about this level.
//!
//! No GPU: everything here is the session's own bookkeeping over the fitter, which is
//! the level the whole feature lives at.

use glam::Quat;
use stark_engine::PerspectiveGuide;
use stark_engine::ViewTransform;
use stark_engine::command::InputSample;
use stark_engine::path::{DEFAULT_TOLERANCE, FLATTEN_TOLERANCE, flatten};
use stark_engine::session::Session;
use stark_model::document::{LayerId, Tool};
use stark_model::geom::{Ellipse, Extent2, Vec2};

fn session() -> Session {
    Session::new(ViewTransform::identity(Extent2::new(512, 512)), LayerId(0))
}

/// Drag a stroke out through `f`, sampled `n` times, without ending it.
fn drag(session: &mut Session, n: usize, f: impl Fn(f32) -> Vec2) {
    session.start_stroke(
        Tool::Brush,
        InputSample::at(f(0.0)),
        1,
        DEFAULT_TOLERANCE,
        0.0,
    );
    for i in 1..n {
        session.stroke_to(InputSample::at(f(i as f32 / (n - 1) as f32)));
    }
}

/// A hand's wobble across whatever it is drawing: deterministic, and far too small to
/// be mistaken for intent.
fn wobble(i: usize) -> Vec2 {
    Vec2::new(0.0, 2.5 * (i as f32 * 1.7).sin())
}

/// The furthest the stroke in flight strays from the chord between its own two ends —
/// how straight it is, without assuming *which* straight line it settled on.
///
/// The recognized line is the total-least-squares fit of the trace, so its ends are the
/// drawn ones *projected*, a wobble's width off wherever the hand actually went.
/// Measuring against the nominal endpoints would be scoring the fit, not the snap.
fn bow(session: &Session) -> f32 {
    let record = session.preview_record().expect("a stroke in flight");
    let poly = flatten(&record.path, FLATTEN_TOLERANCE);
    let (a, b) = (poly[0].pos, poly[poly.len() - 1].pos);
    let dir = (b - a).normalize();
    poly.iter()
        .map(|s| (s.pos - a).perp_dot(dir).abs())
        .fold(0.0, f32::max)
}

/// Where the stroke in flight ends.
fn tip(session: &Session) -> Vec2 {
    let record = session.preview_record().expect("a stroke in flight");
    record.path.last().expect("control points").pos
}

#[test]
fn holding_snaps_a_rough_drag_to_a_line() {
    let (a, b) = (Vec2::new(-120.0, -40.0), Vec2::new(180.0, 30.0));
    let mut session = session();
    drag(&mut session, 40, |t| {
        a.lerp(b, t) + wobble((t * 39.0) as usize)
    });

    let crooked = bow(&session);
    assert!(crooked > 1.0, "the drag was already straight ({crooked}px)");

    let ordinal = session.gesture_ordinal();
    assert!(session.assist_stroke(), "a rough drag is a line");
    assert!(session.is_assisted());
    let snapped = bow(&session);
    assert!(snapped < 0.05, "the snap bows {snapped}px");
    // ...and it is the line that was drawn, not some other one.
    assert!(
        tip(&session).distance(b) < 4.0,
        "the snap moved the end to {}",
        tip(&session)
    );
    assert_eq!(
        session.gesture_ordinal(),
        ordinal + 1,
        "a snap replaces the path, so it must read as a discontinuity — otherwise a \
         peer splices a delta onto a path that no longer exists (§17.5)"
    );
    assert_eq!(
        session.frozen_spans(),
        0,
        "steering a shape moves every control point, so nothing may be retired"
    );
}

/// The rest of the drag steers the shape. The *same* `To` command does it, so nothing
/// about the dwell is mirrored in the frontend's pointer handling.
#[test]
fn a_held_gesture_steers_instead_of_extending() {
    let (a, b) = (Vec2::new(0.0, 0.0), Vec2::new(200.0, 0.0));
    let mut session = session();
    drag(&mut session, 40, |t| {
        a.lerp(b, t) + wobble((t * 39.0) as usize)
    });
    assert!(session.assist_stroke());

    // Swing the held end a long way off the original line. A stroke that was still
    // being *extended* would grow a corner and keep everything before it.
    let moved = Vec2::new(200.0, 160.0);
    for i in 1..=10 {
        session.stroke_to(InputSample::at(b.lerp(moved, i as f32 / 10.0)));
    }
    let bowed = bow(&session);
    assert!(
        bowed < 0.05,
        "the stroke bowed {bowed}px following the pointer"
    );
    let last = tip(&session);
    assert!(
        last.distance(moved) < 0.01,
        "the held end {last} did not follow the pointer to {moved}"
    );

    let record = session.end_stroke().expect("a stroke to commit");
    assert_eq!(
        record.path.last().expect("control points").pos,
        last,
        "the release must commit exactly what was previewed"
    );
}

/// The bar has to decline: a considered stroke that happens to pause before the pen
/// lifts must survive intact.
#[test]
fn holding_on_a_squiggle_changes_nothing() {
    let mut session = session();
    drag(&mut session, 60, |t| {
        Vec2::new(t * 300.0, 60.0 * (t * 9.0).sin())
    });
    let before = session.preview_record().expect("in flight").path;

    assert!(!session.assist_stroke(), "a squiggle is not a shape");
    assert!(!session.is_assisted());
    assert_eq!(
        session.preview_record().expect("in flight").path,
        before,
        "a declined hold must leave the stroke bit-for-bit as it was"
    );

    // ...and the gesture carries on through the fitter exactly as before.
    session.stroke_to(InputSample::at(Vec2::new(320.0, 0.0)));
    assert!(session.preview_record().expect("in flight").path.len() >= before.len());
}

#[test]
fn a_rough_loop_snaps_to_an_ellipse() {
    let (center, radii) = (Vec2::new(40.0, -20.0), Vec2::new(150.0, 90.0));
    let mut session = session();
    drag(&mut session, 90, |t| {
        let u = t * std::f32::consts::TAU;
        center + Vec2::new(radii.x * u.cos(), radii.y * u.sin()) + wobble((t * 89.0) as usize)
    });
    assert!(session.assist_stroke(), "a rough loop is an ellipse");

    let record = session.preview_record().expect("in flight");
    let worst = flatten(&record.path, FLATTEN_TOLERANCE)
        .iter()
        .map(|s| {
            let q = s.pos - center;
            // The algebraic residual over its own gradient — distance to the ellipse.
            let (a2, b2) = (radii.x * radii.x, radii.y * radii.y);
            let residual = q.x * q.x / a2 + q.y * q.y / b2 - 1.0;
            residual.abs() / (2.0 * Vec2::new(q.x / a2, q.y / b2).length())
        })
        .fold(0.0f32, f32::max);
    assert!(
        worst < 4.0,
        "the snapped loop sits {worst}px off the ellipse"
    );
}

/// Idempotent, so the frontend may ask without first working out what state the
/// gesture is in.
#[test]
fn holding_twice_is_one_snap() {
    let mut session = session();
    drag(&mut session, 40, |t| {
        Vec2::new(t * 300.0, t * 40.0) + wobble((t * 39.0) as usize)
    });
    assert!(session.assist_stroke());
    let ordinal = session.gesture_ordinal();
    let path = session.preview_record().expect("in flight").path;

    assert!(
        !session.assist_stroke(),
        "a snapped gesture cannot snap again"
    );
    assert_eq!(session.gesture_ordinal(), ordinal);
    assert_eq!(session.preview_record().expect("in flight").path, path);
}

/// A rough drag toward a vanishing point of a guide the artist has up commits a line
/// that aims exactly at it (§20.6) — the whole feature, through the gesture it is made
/// with, and with the guide read off the session rather than handed in.
#[test]
fn a_held_line_takes_the_axis_of_a_visible_guide() {
    let mut session = session();
    let guide = PerspectiveGuide {
        center: Vec2::new(90.0, -40.0),
        focal: 700.0,
        rotation: Quat::from_rotation_x(0.3) * Quat::from_rotation_y(0.55),
        ..Default::default()
    };
    let vp = guide.scene().vps[2].expect("Z vanishes on the canvas");
    session.guides = vec![guide].into();

    // Drawn from `start`, 400px toward the vanishing point but 4° off it.
    let start = Vec2::new(-240.0, 180.0);
    let aim = Vec2::from_angle(4f32.to_radians()).rotate((vp - start).normalize());
    drag(&mut session, 40, |t| {
        start + aim * (400.0 * t) + wobble((t * 39.0) as usize)
    });
    assert!(session.assist_stroke(), "a rough drag is a line");

    let record = session.end_stroke().expect("a stroke to commit");
    let (a, b) = (record.path[0].pos, record.path.last().unwrap().pos);
    assert!(a.distance(start) < 4.0, "the drawn start moved to {a}");
    let (u, to_vp) = ((b - a).normalize(), (vp - a).normalize());
    assert!(
        u.perp_dot(to_vp).abs() < 1e-2,
        "the committed line points {u}, and the vanishing point is at {to_vp}"
    );
}

/// The guide has to be *on the screen* to bend anything: the same drag, with the guide
/// list empty, keeps the direction the hand gave it.
#[test]
fn the_same_drag_without_a_guide_keeps_its_own_direction() {
    let mut session = session();
    let start = Vec2::new(-240.0, 180.0);
    let aim = Vec2::from_angle(0.9).rotate(Vec2::X);
    drag(&mut session, 40, |t| {
        start + aim * (400.0 * t) + wobble((t * 39.0) as usize)
    });
    assert!(session.assist_stroke());

    let record = session.end_stroke().expect("a stroke to commit");
    let (a, b) = (record.path[0].pos, record.path.last().unwrap().pos);
    assert!(
        (b - a).normalize().perp_dot(aim).abs() < 0.02,
        "the line was turned to {}",
        (b - a).normalize()
    );
}

/// Once a line has taken an axis it keeps it for the rest of the drag: steering runs
/// the end out **along** the grid line, so the alignment survives the hand wandering
/// off it. Without this the feature would last exactly one pointer move.
#[test]
fn steering_an_axis_line_stays_on_the_axis() {
    let mut session = session();
    let guide = PerspectiveGuide {
        center: Vec2::new(90.0, -40.0),
        focal: 700.0,
        rotation: Quat::from_rotation_x(0.3) * Quat::from_rotation_y(0.55),
        ..Default::default()
    };
    let vp = guide.scene().vps[2].expect("Z vanishes on the canvas");
    session.guides = vec![guide].into();

    let start = Vec2::new(-240.0, 180.0);
    let aim = Vec2::from_angle(4f32.to_radians()).rotate((vp - start).normalize());
    let end = start + aim * 400.0;
    drag(&mut session, 40, |t| {
        start + aim * (400.0 * t) + wobble((t * 39.0) as usize)
    });
    assert!(session.assist_stroke());

    // Swing the pointer a long way off the line, as a hand steering the far end does.
    let moved = end + Vec2::new(60.0, 220.0);
    for i in 1..=10 {
        session.stroke_to(InputSample::at(end.lerp(moved, i as f32 / 10.0)));
    }

    let record = session.end_stroke().expect("a stroke to commit");
    let (a, b) = (record.path[0].pos, record.path.last().unwrap().pos);
    let to_vp = (vp - a).normalize();
    assert!(
        (b - a).normalize().perp_dot(to_vp).abs() < 1e-2,
        "the end was steered off the axis, to {b}"
    );
    // It did move — along the line, by the pointer's travel resolved onto it.
    let along = (moved - end).dot(to_vp);
    assert!(
        ((b - a).length() - (400.0 + along)).abs() < 6.0,
        "the end ran to {} rather than {}",
        (b - a).length(),
        400.0 + along
    );
}

/// A loop drawn roughly where a circle on the ground would be commits a stroke that
/// **pulls back to a constant radius on that plane** (§20.7) — the statement the canvas
/// shape alone cannot make, and the one the feature is for.
///
/// Asserted on the committed path rather than on the shape, so it covers the whole
/// route: recognition, the realization into control points, and the release.
#[test]
fn a_held_loop_becomes_a_circle_on_a_visible_plane() {
    let mut session = session();
    let guide = PerspectiveGuide {
        center: Vec2::new(90.0, -40.0),
        focal: 700.0,
        rotation: Quat::from_rotation_x(0.35) * Quat::from_rotation_y(0.5),
        ..Default::default()
    };
    // Pair 2 spans (Z, X): the ground. A circle on it, sized to draw at ~180px.
    let plane = guide.planes()[2].expect("the ground plane");
    let at = Vec2::new(0.05, -0.04);
    let probe = plane.circle_seen(at, 0.02).expect("a bounded image");
    let radius = 0.02 * 180.0 / probe.scale();
    let Ellipse {
        center,
        radii,
        angle,
    } = plane.circle_seen(at, radius).expect("a bounded image");
    session.guides = vec![guide].into();

    // Drawn a tenth too round and leaning 3° off — the two things a hand gets wrong.
    let (u, v) = (
        Vec2::from_angle(angle + 0.05),
        Vec2::from_angle(angle + 0.05).perp(),
    );
    drag(&mut session, 90, |t| {
        let a = t * std::f32::consts::TAU;
        center
            + u * (radii.x * a.cos())
            + v * (radii.y * 1.1 * a.sin())
            + wobble((t * 89.0) as usize)
    });
    assert!(session.assist_stroke(), "a rough loop is an ellipse");

    let record = session.end_stroke().expect("a stroke to commit");
    let poly = flatten(&record.path, FLATTEN_TOLERANCE);
    let flat: Vec<Vec2> = poly
        .iter()
        .map(|s| plane.to_plane(s.pos).expect("on the plane"))
        .collect();
    // Measured off the outline's own extent rather than its centroid: the flattened
    // polyline is evenly spaced on the *canvas*, which perspective makes wildly uneven
    // on the plane, so a centroid would sit well off the centre. A bounding box only
    // needs the extremes, and for a circle it is a square about the centre.
    let (lo, hi) = flat.iter().fold(
        (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN)),
        |(lo, hi), q| (lo.min(*q), hi.max(*q)),
    );
    let (hub, span) = ((lo + hi) * 0.5, hi - lo);
    assert!(
        (span.x - span.y).abs() < 0.02 * span.x,
        "the committed stroke spans {span} on the plane — not square, so not a circle"
    );
    let r = 0.25 * (span.x + span.y);
    let worst = flat
        .iter()
        .map(|q| (q.distance(hub) - r).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst < 0.02 * r,
        "the outline strays {worst} off a radius of {r} on the plane"
    );
    assert!(
        hub.distance(at) < 0.2 * radius,
        "the circle landed at {hub}, not {at}"
    );
}

/// The same drag with the guide's eye shut stays the ellipse it was drawn as, wonky
/// eccentricity and all.
#[test]
fn a_hidden_plane_leaves_the_loop_alone() {
    let mut session = session();
    let guide = PerspectiveGuide {
        center: Vec2::new(90.0, -40.0),
        focal: 700.0,
        rotation: Quat::from_rotation_x(0.35) * Quat::from_rotation_y(0.5),
        visible: false,
        ..Default::default()
    };
    assert!(guide.planes()[2].is_none(), "a hidden guide put up a plane");
    session.guides = vec![guide].into();

    let (center, radii) = (Vec2::new(40.0, -20.0), Vec2::new(200.0, 90.0));
    drag(&mut session, 90, |t| {
        let a = t * std::f32::consts::TAU;
        center + Vec2::new(radii.x * a.cos(), radii.y * a.sin()) + wobble((t * 89.0) as usize)
    });
    assert!(session.assist_stroke());

    // Still the drawn ellipse: 200×90, not something a plane chose.
    let record = session.end_stroke().expect("a stroke to commit");
    let poly = flatten(&record.path, FLATTEN_TOLERANCE);
    let worst = poly
        .iter()
        .map(|s| {
            let q = s.pos - center;
            let (a2, b2) = (radii.x * radii.x, radii.y * radii.y);
            let residual = q.x * q.x / a2 + q.y * q.y / b2 - 1.0;
            residual.abs() / (2.0 * Vec2::new(q.x / a2, q.y / b2).length())
        })
        .fold(0.0f32, f32::max);
    assert!(worst < 4.0, "the loop was reshaped, sitting {worst}px off");
}

#[test]
fn a_hold_with_nothing_in_flight_is_a_no_op() {
    let mut session = session();
    assert!(!session.assist_stroke());

    // A shape gesture builds no stroke, and an exact marquee has nothing to snap to.
    session.start_selection(Tool::SelectRect, Vec2::ZERO);
    session.selection_to(Vec2::new(60.0, 60.0));
    assert!(!session.assist_stroke());
    assert!(session.is_selecting(), "the marquee survived the hold");
}
