//! Gradient capture (§22.2): tracing a line through paint and getting a ramp back.
//!
//! The promise under test is that every sample of a trace **is** an eyedropper
//! pick — same sources, same patch mean, same raw-channels rule — so the fitted
//! gradient's ends are the colors a pick at those points reports, in both
//! color spaces. And the refusals: a trace over bare canvas has no gradient in
//! it, and a gap of bare canvas crossed mid-trace does not inject the paper.

mod common;

use common::*;
use stark_engine::{Engine, PickOptions};
use stark_model::Gradient;
use stark_model::geom::Vec2;

const RED: [f32; 4] = [0.85, 0.12, 0.1, 1.0];
const BLUE: [f32; 4] = [0.1, 0.2, 0.8, 1.0];

/// The capture the UI takes: a 5×5 patch per sample (§22.2).
fn capture(engine: &mut Engine, path: &[Vec2]) -> Option<Gradient> {
    pollster::block_on(engine.pick_gradient(
        path,
        PickOptions {
            radius: 2,
            ..PickOptions::default()
        },
    ))
}

fn near(a: [f32; 3], b: [f32; 4], tol: f32) -> bool {
    (0..3).all(|i| (a[i] - b[i]).abs() <= tol)
}

/// Two abutting bars, a trace across both: the ramp's ends are the two paints.
///
/// In both color spaces, because it is a different claim in each — in Mixbox the
/// samples are pigment mixtures run back through the polynomial, and a capture
/// that forgot the residual would hand back a ramp through `#383838` (§6.7).
#[test]
fn a_trace_across_two_paints_ends_on_each() {
    for space in stark_engine::colorspace::all_available() {
        let Some(mut engine) = engine_or_skip_with(space) else {
            return;
        };
        // Wide vertical bars butted at x = 0, painted with a wide tip so the
        // trace runs through solid paint on both sides.
        paint(
            &mut engine,
            RED,
            30.0,
            &[Vec2::new(-30.0, -20.0), Vec2::new(-30.0, 20.0)],
        );
        paint(
            &mut engine,
            BLUE,
            30.0,
            &[Vec2::new(30.0, -20.0), Vec2::new(30.0, 20.0)],
        );

        let g = capture(&mut engine, &[Vec2::new(-40.0, 0.0), Vec2::new(40.0, 0.0)])
            .expect("paint under the whole trace");
        let start = g.sample(0.0);
        let end = g.sample(1.0);
        assert!(
            near((start).get(), RED, 0.06),
            "{space:?}: the trace starts on red, got {start:?}"
        );
        assert!(
            near((end).get(), BLUE, 0.06),
            "{space:?}: the trace ends on blue, got {end:?}"
        );

        // Every sample is an eyedropper pick: the ramp's end agrees with a pick
        // at the same point, to well under a visible difference.
        let picked = pollster::block_on(engine.pick_color(
            Vec2::new(-40.0, 0.0),
            PickOptions {
                radius: 2,
                ..PickOptions::default()
            },
        ))
        .expect("paint at the trace's start");
        assert!(
            near((start).get(), [picked[0], picked[1], picked[2], 1.0], 0.02),
            "{space:?}: capture start {start:?} vs pick {picked:?}"
        );
    }
}

/// Bare canvas has no gradient in it — the same refusal as the eyedropper's,
/// stretched along a line.
#[test]
fn a_trace_over_bare_canvas_has_no_gradient() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    assert!(
        capture(&mut engine, &[Vec2::new(-40.0, 0.0), Vec2::new(40.0, 0.0)]).is_none(),
        "empty document"
    );

    paint(
        &mut engine,
        RED,
        24.0,
        &[Vec2::new(-40.0, 0.0), Vec2::new(40.0, 0.0)],
    );
    assert!(
        capture(
            &mut engine,
            &[Vec2::new(-40.0, 200.0), Vec2::new(40.0, 200.0)]
        )
        .is_none(),
        "well clear of the only stroke"
    );
}

/// A gap of bare canvas mid-trace is skipped, not sampled: the ramp runs from
/// paint to paint without the paper appearing between them.
#[test]
fn a_gap_in_the_paint_does_not_join_the_ramp() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // Two bars with 40 px of bare canvas between them.
    paint(
        &mut engine,
        RED,
        16.0,
        &[Vec2::new(-60.0, -20.0), Vec2::new(-60.0, 20.0)],
    );
    paint(
        &mut engine,
        BLUE,
        16.0,
        &[Vec2::new(60.0, -20.0), Vec2::new(60.0, 20.0)],
    );

    let g = capture(&mut engine, &[Vec2::new(-60.0, 0.0), Vec2::new(60.0, 0.0)])
        .expect("paint at both ends");
    // Nowhere along the fitted ramp does anything but a red→blue mixture appear.
    // "Mixture" means: on or near the Oklab segment joining the two paints —
    // which an interpolated ramp is by construction, and which the paper (or a
    // None mis-mapped to black) sits far off.
    let lab = |c: [f32; 3]| {
        let l = stark_model::color::srgb_to_oklab([c[0], c[1], c[2], 1.0]);
        [l[0], l[1], l[2]]
    };
    let a = lab((g.sample(0.0)).get());
    let b = lab((g.sample(1.0)).get());
    for i in 0..=32 {
        let p = lab((g.sample(i as f32 / 32.0)).get());
        // Distance from p to the segment a–b.
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
        let len2 = ab.iter().map(|v| v * v).sum::<f32>();
        let f = (ab.iter().zip(&ap).map(|(x, y)| x * y).sum::<f32>() / len2).clamp(0.0, 1.0);
        let d = (0..3)
            .map(|ch| (ap[ch] - ab[ch] * f).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(
            d < 0.08,
            "a color {d} off the red–blue line joined the ramp: {:?}",
            g.sample(i as f32 / 32.0)
        );
    }
}
