//! Drag-and-hold drawing assist, through the gesture it is actually made with
//! (§6.9).
//!
//! [`crate::assist`](stark_core::assist)'s own tests cover the geometry — which shape
//! a trace is, and how faithfully the ideal one is realized. What is tested here is the
//! part that only exists once a [`Session`] is holding a gesture: that a hold replaces
//! the path in flight, that the rest of the same drag *steers* rather than extends,
//! that the release commits what was previewed, and that a stroke the recognizer
//! declines is left exactly as it was drawn.
//!
//! No GPU: everything here is the session's own bookkeeping over the fitter, which is
//! the level the whole feature lives at.

use stark_core::command::InputSample;
use stark_core::document::{LayerId, Tool};
use stark_core::geom::{Extent2, Vec2, ViewTransform};
use stark_core::path::{DEFAULT_TOLERANCE, FLATTEN_TOLERANCE, flatten};
use stark_core::session::Session;

fn session() -> Session {
    Session::new(ViewTransform::identity(Extent2::new(512, 512)), LayerId(0))
}

/// Drag a stroke out through `f`, sampled `n` times, without ending it.
fn drag(session: &mut Session, n: usize, f: impl Fn(f32) -> Vec2) {
    session.start_stroke(Tool::Brush, InputSample::at(f(0.0)), 1, DEFAULT_TOLERANCE);
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
