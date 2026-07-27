//! Colour dynamics (colour jitter) — DESIGN.md §6.2: the applied colour varies
//! across the brush and along the stroke, driven by a tileable 2-D noise field
//! sampled in the stroke's own (lateral, arc) frame.
//! These tests pin the three contracts: **off is exactly off** (a zero-amplitude
//! brush deposits bit-identically to one with no dynamics at all), **on is
//! deterministic** (same record ⇒ same pixels, across engines and through
//! save/load — the jitter is a pure function of the record + seed), and the
//! goldens keep the look itself honest for both noise kinds and both deposit
//! paths (the swept fast path and the sequential exchange loop).

mod common;

use common::*;
use stark_core::command::{GestureCommand, InputSample, ViewCommand};
use stark_core::document::{BrushParams, ColorDynamics, NoiseKind, Tool};
use stark_core::geom::Vec2;
use stark_core::path::DEFAULT_TOLERANCE;

/// A muted teal mid-tone: enough headroom in every Oklab channel for the jitter
/// to wander both ways without everything clamping at the gamut edge.
const TEAL: [f32; 4] = [0.25, 0.55, 0.55, 1.0];

/// The standard test stroke: a broad S-curve covering a good patch of canvas.
const S_CURVE: [Vec2; 5] = [
    Vec2::new(-95.0, 55.0),
    Vec2::new(-50.0, -40.0),
    Vec2::new(0.0, 55.0),
    Vec2::new(50.0, -40.0),
    Vec2::new(95.0, 55.0),
];

fn jitter_brush(noise: NoiseKind, frequency: [f32; 2], amplitude: [f32; 3]) -> BrushParams {
    let mut b = brush(TEAL, 42.0);
    b.color_dynamics = ColorDynamics {
        noise,
        frequency,
        amplitude,
    };
    b
}

#[test]
fn golden_color_jitter_simplex() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // Smooth organic wander in all three Oklab channels, at a scale that shows a
    // few colour "clouds" across the stroke and a few more along it.
    let b = jitter_brush(NoiseKind::Simplex, [2.0, 0.5], [0.18, 0.14, 0.14]);
    stroke_with(&mut engine, b, &S_CURVE);
    let img = engine.render_to_image();
    assert_golden("color_jitter_simplex", &img, 6);
}

#[test]
fn golden_color_jitter_white() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // High-frequency white noise: fine speckled grain in lightness + chroma.
    let b = jitter_brush(NoiseKind::White, [6.0, 6.0], [0.15, 0.10, 0.10]);
    stroke_with(&mut engine, b, &S_CURVE);
    let img = engine.render_to_image();
    assert_golden("color_jitter_white", &img, 6);
}

/// Along-stroke-only jitter (`frequency = [0, f]`): the colour drifts along
/// the arc but is uniform across the footprint at any point of the stroke.
#[test]
fn golden_color_jitter_along_stroke() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let b = jitter_brush(NoiseKind::Simplex, [0.0, 4.0], [0.16, 0.16, 0.16]);
    stroke_with(&mut engine, b, &S_CURVE);
    let img = engine.render_to_image();
    assert_golden("color_jitter_along_stroke", &img, 6);
}

/// Zero amplitude must be **exactly** the constant-colour deposit — the shader
/// early-outs and the zero volume is bound, so a brush with dynamics fields set
/// but amplitude 0 renders bit-identically to the default brush.
#[test]
fn zero_amplitude_is_bit_identical_to_off() {
    let (Some(mut a), Some(mut b)) = (engine_or_skip(), engine_or_skip()) else {
        return;
    };
    paint(&mut a, TEAL, 42.0, &S_CURVE);

    let mut jb = brush(TEAL, 42.0);
    jb.color_dynamics = ColorDynamics {
        noise: NoiseKind::White,
        frequency: [5.0, 5.0],
        amplitude: [0.0; 3],
    };
    stroke_with(&mut b, jb, &S_CURVE);

    let (ia, ib) = (a.render_to_image(), b.render_to_image());
    assert!(
        images_match(&ia, &ib, 0),
        "amplitude-0 dynamics changed the deposit"
    );
}

/// The jitter is a pure function of the stroke record: two engines replaying the
/// same script produce identical pixels, and the jitter genuinely does something.
#[test]
fn jitter_is_deterministic_and_effective() {
    let (Some(mut a), Some(mut b), Some(mut plain)) =
        (engine_or_skip(), engine_or_skip(), engine_or_skip())
    else {
        return;
    };
    let jb = jitter_brush(NoiseKind::Simplex, [3.0, 3.0], [0.2, 0.12, 0.12]);
    stroke_with(&mut a, jb, &S_CURVE);
    stroke_with(&mut b, jb, &S_CURVE);
    paint(&mut plain, TEAL, 42.0, &S_CURVE);

    let (ia, ib) = (a.render_to_image(), b.render_to_image());
    assert!(images_match(&ia, &ib, 0), "jitter not deterministic");
    let ip = plain.render_to_image();
    assert!(
        frac_exceeding(&ia, &ip, 10) > 0.01,
        "jitter had no visible effect"
    );
}

/// Save → load replays the jittered stroke bit-for-bit (the dynamics live in the
/// historized `BrushParams`; the lookup offset derives from the recorded seed).
#[test]
fn jitter_survives_save_load() {
    let (Some(mut a), Some(mut b)) = (engine_or_skip(), engine_or_skip()) else {
        return;
    };
    let jb = jitter_brush(NoiseKind::White, [4.0, 2.0], [0.15, 0.1, 0.1]);
    stroke_with(&mut a, jb, &S_CURVE);
    let bytes = a.save_bytes().expect("save");
    b.load_bytes(&bytes).expect("load");
    let (ia, ib) = (a.render_to_image(), b.render_to_image());
    assert!(
        images_match(&ia, &ib, 0),
        "jittered stroke not replay-lossless"
    );
}

/// The sequential exchange loop (a brush that also lifts/deposits canvas paint —
/// DESIGN §6.2) applies the same jitter to the brush's own `add` paint:
/// deterministic across engines, and visibly different from the unjittered loop.
#[test]
fn jitter_applies_in_dynamics_loop() {
    let (Some(mut a), Some(mut b), Some(mut plain)) =
        (engine_or_skip(), engine_or_skip(), engine_or_skip())
    else {
        return;
    };
    let mut jb = jitter_brush(NoiseKind::Simplex, [3.0, 3.0], [0.2, 0.12, 0.12]);
    jb.dynamics.lift = 0.3;
    jb.dynamics.deposit = 0.3; // lift/deposit > 0 ⇒ the exchange-loop path
    stroke_with(&mut a, jb, &S_CURVE);
    stroke_with(&mut b, jb, &S_CURVE);
    let mut pb = jb;
    pb.color_dynamics = ColorDynamics::default();
    stroke_with(&mut plain, pb, &S_CURVE);

    let (ia, ib) = (a.render_to_image(), b.render_to_image());
    assert!(images_match(&ia, &ib, 0), "loop jitter not deterministic");
    let ip = plain.render_to_image();
    assert!(
        frac_exceeding(&ia, &ip, 10) > 0.01,
        "loop jitter had no visible effect"
    );
}

/// Live preview == committed: replaying the samples one-by-one (the live path)
/// then committing renders the same pixels as a single replayed commit.
#[test]
fn jittered_live_preview_matches_commit() {
    let (Some(mut live), Some(mut replayed)) = (engine_or_skip(), engine_or_skip()) else {
        return;
    };
    let jb = jitter_brush(NoiseKind::Simplex, [2.0, 3.0], [0.15, 0.1, 0.1]);

    live.process(ViewCommand::SetBrush(jb));
    let mut it = S_CURVE.iter();
    live.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(*it.next().unwrap()),
        tolerance: DEFAULT_TOLERANCE,
    });
    for &p in it {
        live.process(GestureCommand::To {
            sample: InputSample::at(p),
        });
    }
    live.process(GestureCommand::End);

    replayed.process(ViewCommand::SetBrush(jb));
    let samples: Vec<InputSample> = S_CURVE.iter().map(|&p| InputSample::at(p)).collect();
    replayed.replay_stroke(Tool::Brush, &samples);

    let (il, ir) = (live.render_to_image(), replayed.render_to_image());
    assert!(images_match(&il, &ir, 0), "live jitter != committed jitter");
}
