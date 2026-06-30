//! Palette-knife dynamics (DESIGN.md §6.2): the unified tool driven by per-sample pen
//! **pressure** (scrape) and **tilt toward motion** (deposit), plus a finite pre-`charge`.
//!
//! The existing dynamics goldens (`golden.rs`) all leave `load_pressure = 0`, so they
//! already guard that the new per-pixel scrape path doesn't perturb the constant case. These
//! tests exercise the new behaviour itself.

mod common;

use common::*;
use stark_core::command::{InputCommand, InputSample};
use stark_core::document::{BrushDynamics, BrushParams, Tool};
use stark_core::geom::Vec2;
use stark_core::RgbaImage;

const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];

/// A knife preset: no own paint, scrape on, the rest off unless overridden.
fn knife(load: f32, deposit: f32, dyn_overrides: BrushDynamics) -> BrushParams {
    dry_brush(RED, 22.0, BrushDynamics { add: 0.0, load, deposit, ..dyn_overrides })
}

/// Commit a stroke through explicit samples (so pressure/tilt vary per sample).
fn stroke_samples(engine: &mut stark_core::Engine, b: BrushParams, samples: &[InputSample]) {
    engine.process(InputCommand::SetBrush(b));
    let mut it = samples.iter();
    let first = *it.next().expect("at least one sample");
    engine.process(InputCommand::StartStroke { tool: Tool::Brush, sample: first });
    for &s in it {
        engine.process(InputCommand::StrokeTo { sample: s });
    }
    engine.process(InputCommand::EndStroke);
}

/// The whole new structural path (3-target scrape sweep + the integrate's per-pixel scrape
/// branch) must reduce *exactly* to the unchanged constant-`load` path when pressure is full
/// everywhere — `load_seg = load · mix(1, 1, load_pressure) = load` regardless of
/// `load_pressure`. So a `load_pressure = 1` scrape at pressure 1 must match the
/// `load_pressure = 0` scrape pixel-for-pixel (modulo float precision). This is the strongest
/// guard that the new code is behaviour-preserving (DESIGN §6.2).
#[test]
fn pressure_scrape_matches_constant_at_full_pressure() {
    let scene = |load_pressure: f32| -> Option<RgbaImage> {
        let mut engine = engine_or_skip()?;
        paint(&mut engine, RED, 60.0, &[Vec2::new(-95.0, 0.0), Vec2::new(95.0, 0.0)]);
        // `stroke_with` uses full-pressure samples → load_seg is independent of load_pressure.
        stroke_with(
            &mut engine,
            knife(0.8, 0.0, BrushDynamics { load_pressure, ..Default::default() }),
            &[Vec2::new(0.0, -80.0), Vec2::new(0.0, 80.0)],
        );
        Some(engine.render_to_image(PAPER))
    };
    let (Some(constant), Some(scrape)) = (scene(0.0), scene(1.0)) else {
        return; // no GPU adapter
    };
    let frac = frac_exceeding(&constant, &scrape, 4);
    assert!(
        frac < 0.005,
        "per-pixel scrape at full pressure must match the constant path: {:.3}% px differ",
        frac * 100.0
    );
}

/// Pressure modulates the scrape: a knife dragged across a solid field with a pressure ramp
/// (light → hard) must remove *more* paint where it pressed harder. We compare the painted
/// halves of the field: the high-pressure end of the scrape channel lifts more red, so that
/// region is measurably lighter (closer to paper) than the low-pressure end.
#[test]
fn pressure_ramps_the_scrape() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // Solid red field across the whole canvas.
    paint(&mut engine, RED, 120.0, &[Vec2::new(-110.0, 0.0), Vec2::new(110.0, 0.0)]);
    let before = engine.render_to_image(PAPER);

    // A vertical scrape with pressure rising from top (light) to bottom (hard).
    let ramp = |t: f32| InputSample {
        pos: Vec2::new(0.0, -80.0 + 160.0 * t),
        pressure: 0.15 + 0.85 * t,
        tilt: Vec2::ZERO,
        time: t as f64,
    };
    let samples: Vec<_> = (0..=16).map(|i| ramp(i as f32 / 16.0)).collect();
    stroke_samples(
        &mut engine,
        knife(1.0, 0.0, BrushDynamics { load_pressure: 1.0, ..Default::default() }),
        &samples,
    );
    let after = engine.render_to_image(PAPER);

    // Paint removed at the top (light) vs bottom (hard) of the scrape, sampled on the path.
    // The red field and the light paper both have a high *red* channel, so it can't tell them
    // apart; the **green** channel does — it rises toward paper as the red paint is scraped off.
    let scraped = |y: f32| -> i32 {
        let x = (SIZE.width / 2) as usize;
        let py = ((y + 128.0).clamp(0.0, 255.0)) as usize;
        let idx = (py * SIZE.width as usize + x) * 4 + 1; // green channel
        after.pixels[idx] as i32 - before.pixels[idx] as i32
    };
    let light = scraped(-60.0); // near the light end
    let hard = scraped(60.0); // near the hard end
    assert!(
        hard > light + 10,
        "harder pressure must scrape more: light-end Δgreen={light}, hard-end Δgreen={hard}"
    );
    assert_golden("knife_pressure_ramp", &after, 6);
}

/// Tilt toward the direction of motion modulates the deposit. A pre-`charge`d knife (no scrape)
/// dragged left→right deposits more where the pen leans *forward* (tilt · dir > 0) than where
/// it leans *backward*. Two strokes that differ only in tilt sign must lay different amounts.
#[test]
fn forward_tilt_deposits_more_than_backward() {
    let deposited = |tilt_x: f32| -> Option<i64> {
        let mut engine = engine_or_skip()?;
        let blank = engine.render_to_image(PAPER);
        // A charged knife that only deposits (no scrape), fully tilt-gated, moving +x. A small
        // per-column deposit so the finite charge *spreads* along the stroke body (a large rate
        // dumps it all into the start cap, where coverage → 0; cf. `knife_carry`). The only
        // paint source is the pre-`charge` (add = 0, no lift), so any trail proves the
        // pre-charge works, and the tilt gate decides whether it appears at all.
        let b = knife(
            0.0,
            0.08,
            BrushDynamics { charge: 1.5, deposit_tilt: 1.0, ..Default::default() },
        );
        let tilt = Vec2::new(tilt_x, 0.0); // +x = lean toward motion, −x = away
        let s = |x: f32| InputSample { pos: Vec2::new(x, 0.0), pressure: 1.0, tilt, time: 0.0 };
        stroke_samples(&mut engine, b, &[s(-80.0), s(0.0), s(80.0)]);
        let after = engine.render_to_image(PAPER);
        // Total absolute change from blank = how much paint landed.
        let total: i64 = blank
            .pixels
            .iter()
            .zip(&after.pixels)
            .map(|(a, c)| (*a as i64 - *c as i64).abs())
            .sum();
        Some(total)
    };
    let (Some(forward), Some(backward)) = (deposited(1.0), deposited(-1.0)) else {
        return;
    };
    assert!(
        forward > backward,
        "forward tilt must deposit more than backward: forward={forward}, backward={backward}"
    );
}

/// Regression: the deposit must be **continuous through vertical**. A barely-back-tilted pen
/// and an upright pen are almost the same pose, so they must deposit almost the same amount.
/// The old code's no-pen fallback forced full deposit at ~zero tilt while a back tilt gave
/// zero, so passing through vertical mid-stroke slammed the deposit from 0 → full (the
/// reported discontinuity). With the smooth lean the two poses now agree closely.
#[test]
fn deposit_is_continuous_through_vertical() {
    let deposited = |tilt_x: f32| -> Option<i64> {
        let mut engine = engine_or_skip()?;
        let blank = engine.render_to_image(PAPER);
        let b = knife(
            0.0,
            0.08,
            BrushDynamics { charge: 1.5, deposit_tilt: 1.0, ..Default::default() },
        );
        let tilt = Vec2::new(tilt_x, 0.0);
        let s = |x: f32| InputSample { pos: Vec2::new(x, 0.0), pressure: 1.0, tilt, time: 0.0 };
        stroke_samples(&mut engine, b, &[s(-80.0), s(0.0), s(80.0)]);
        let after = engine.render_to_image(PAPER);
        Some(
            blank
                .pixels
                .iter()
                .zip(&after.pixels)
                .map(|(a, c)| (*a as i64 - *c as i64).abs())
                .sum(),
        )
    };
    let (Some(upright), Some(back)) = (deposited(0.0), deposited(-0.02)) else {
        return;
    };
    // A 0.02 (≈ 1.8°) back tilt must land within ~10% of the upright deposit — a smooth
    // response, not a cliff. (The old fallback gave `back = 0`, `upright = full`.)
    let hi = upright.max(back) as f64;
    let rel = (upright - back).unsigned_abs() as f64 / hi.max(1.0);
    assert!(
        rel < 0.1,
        "deposit must be continuous through vertical: upright={upright}, back={back} ({:.1}% apart)",
        rel * 100.0
    );
}
