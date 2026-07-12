//! Brush-dynamics stamp-loop tests (DESIGN.md §6.2): the sequential pickup ⇄
//! deposit exchange against a 2-D tool reservoir. Covers conservation (paint
//! moves by height, is neither created nor destroyed), smearing onto bare
//! canvas, erasing without re-tinting, the pre-`charge` glob, and determinism.

mod common;

use common::*;
use stark_core::document::{BrushDynamics, BrushParams, Tool};
use stark_core::command::{InputCommand, InputSample};
use stark_core::geom::Vec2;

const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];

/// A brush with the given [`BrushDynamics`].
fn dyn_brush(color: [f32; 4], radius: f32, dynamics: BrushDynamics) -> BrushParams {
    BrushParams {
        dynamics,
        ..brush(color, radius)
    }
}

// Channel-dominance predicates — see tests/stroke.rs for why dominance (not
// saturation) and why the 60-level margin.
fn is_red(c: [u8; 4]) -> bool {
    c[0] as i32 > c[1] as i32 + 60 && c[0] as i32 > c[2] as i32 + 60
}
fn is_green(c: [u8; 4]) -> bool {
    c[1] as i32 > c[0] as i32 + 60 && c[1] as i32 > c[2] as i32 + 60
}
fn center(img: &stark_core::RgbaImage) -> [u8; 4] {
    img.pixel(img.width / 2, img.height / 2)
}

#[test]
fn conservative_smear_preserves_uniform_field() {
    // With `add = 0` the tool only *moves* paint between canvas and reservoir.
    // Smearing entirely inside a uniform field has nothing to move — the lift
    // takes paint up and the deposit lays the same colour straight back — so the
    // field must stay essentially unchanged (paint neither destroyed nor piled).
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    assert!(
        !is_red(center(&engine.render_to_image(PAPER))),
        "lit bare paper reads as red — is_red can't discriminate paint here"
    );
    paint(
        &mut engine,
        RED,
        80.0,
        &[Vec2::new(-110.0, 0.0), Vec2::new(110.0, 0.0)],
    );
    let before = engine.render_to_image(PAPER);

    let b = dyn_brush(
        RED,
        24.0,
        BrushDynamics { add: 0.0, load: 0.5, deposit: 0.5, ..Default::default() },
    );
    stroke_with(&mut engine, b, &[Vec2::new(-50.0, 0.0), Vec2::new(50.0, 0.0)]);
    let after = engine.render_to_image(PAPER);

    assert!(
        is_red(center(&after)),
        "paint must not be destroyed by a conservative smear of a uniform field"
    );
    let frac = frac_exceeding(&before, &after, 40);
    assert!(
        frac < 0.2,
        "smearing a uniform field should move almost nothing, but {:.1}% of pixels differ by >40 levels",
        frac * 100.0
    );
}

#[test]
fn smear_carries_paint_onto_bare_canvas() {
    // The headline behaviour: dragging an `add = 0` brush from a painted patch
    // onto bare canvas deposits the carried paint there. Any red on the
    // previously-bare runway *must* have been lifted off the patch (conservation).
    // Measured via the green channel: red on the warm paper lowers it.
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let y = SIZE.height / 2;
    let run_x = SIZE.width / 2 - 12; // canvas −12: bare runway past the patch

    paint(
        &mut engine,
        RED,
        40.0,
        &[Vec2::new(-90.0, 0.0), Vec2::new(-50.0, 0.0)],
    );
    let run_before = engine.render_to_image(PAPER).pixel(run_x, y);
    assert!(run_before[1] > 170, "the runway should start bare paper: {run_before:?}");

    let b = dyn_brush(
        RED,
        28.0,
        BrushDynamics { add: 0.0, load: 0.9, deposit: 0.3, ..Default::default() },
    );
    stroke_with(
        &mut engine,
        b,
        &[
            Vec2::new(-80.0, 0.0),
            Vec2::new(-30.0, 0.0),
            Vec2::new(30.0, 0.0),
            Vec2::new(90.0, 0.0),
        ],
    );
    let run_after = engine.render_to_image(PAPER).pixel(run_x, y);

    assert!(
        (run_after[1] as i32) < run_before[1] as i32 - 15,
        "with add=0, the bare runway must gain carried paint (green drops): {run_before:?} -> {run_after:?}"
    );
}

#[test]
fn eraser_thins_without_retint() {
    // An eraser (`load > 0, deposit = 0, add = 0`) fills its reservoir but lays
    // nothing back — paint it crosses must only *thin* (height drops), never take
    // on the colour the tool picked up earlier.
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let y = SIZE.height / 2;
    let green_x = SIZE.width / 2 + 60; // inside the green bar

    paint(
        &mut engine,
        RED,
        40.0,
        &[Vec2::new(-90.0, 0.0), Vec2::new(-50.0, 0.0)],
    );
    paint(
        &mut engine,
        GREEN,
        40.0,
        &[Vec2::new(30.0, 0.0), Vec2::new(90.0, 0.0)],
    );
    let before = engine.render_to_image(PAPER).pixel(green_x, y);
    assert!(is_green(before), "the bar should start green: {before:?}");

    // Drag from inside the red patch across the green bar: the tool holds red
    // while it crosses green.
    let b = dyn_brush(
        RED,
        24.0,
        BrushDynamics { add: 0.0, load: 0.5, deposit: 0.0, ..Default::default() },
    );
    stroke_with(&mut engine, b, &[Vec2::new(-80.0, 0.0), Vec2::new(90.0, 0.0)]);
    let after = engine.render_to_image(PAPER).pixel(green_x, y);

    assert!(
        !is_red(after),
        "erasing across the green bar must thin it, not re-tint it red: {before:?} -> {after:?}"
    );
}

#[test]
fn smear_over_empty_canvas_adds_nothing() {
    // Bare canvas has no paint to lift, so an `add = 0` tool must lay nothing —
    // no colour, no height (no relief raised out of nothing).
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let blank = engine.render_to_image(PAPER);
    let b = dyn_brush(
        RED,
        24.0,
        BrushDynamics { add: 0.0, load: 1.0, deposit: 1.0, ..Default::default() },
    );
    stroke_with(&mut engine, b, &[Vec2::new(-40.0, 0.0), Vec2::new(40.0, 0.0)]);
    let after = engine.render_to_image(PAPER);
    assert!(
        images_match(&blank, &after, 2),
        "a pure smear over empty canvas must add no paint or height"
    );
}

#[test]
fn charged_tool_lays_a_finite_glob() {
    // A pre-`charge`d tool (add = 0, no lift) deposits its glob along the stroke:
    // paint appears near the start and runs out with distance — the reservoir is
    // finite, unlike the inexhaustible `add` source.
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let y = SIZE.height / 2;
    let b = dyn_brush(
        RED,
        20.0,
        BrushDynamics { add: 0.0, deposit: 0.12, charge: 2.0, ..Default::default() },
    );
    stroke_with(&mut engine, b, &[Vec2::new(-110.0, 0.0), Vec2::new(110.0, 0.0)]);
    let img = engine.render_to_image(PAPER);

    let start = img.pixel(20, y); // near the stroke start
    assert!(
        (start[1] as i32) < 170,
        "the charged glob must lay paint near the stroke start: {start:?}"
    );
    // The glob depletes: the very end of the stroke carries visibly less paint
    // than the start (compare green-channel loss vs bare paper).
    let end = img.pixel(SIZE.width - 20, y);
    assert!(
        end[1] > start[1],
        "the finite charge must deplete along the stroke: start {start:?} vs end {end:?}"
    );
}

#[test]
fn dynamics_stroke_is_deterministic() {
    // The stamp loop is a pure function of base + record: two engines running the
    // same script must produce identical images (replay / live == committed).
    let render = || {
        let mut engine = engine_or_skip()?;
        paint(
            &mut engine,
            GREEN,
            38.0,
            &[Vec2::new(0.0, -90.0), Vec2::new(0.0, 90.0)],
        );
        let b = dyn_brush(
            RED,
            16.0,
            BrushDynamics { add: 0.1, load: 0.5, deposit: 0.5, ..Default::default() },
        );
        stroke_with(&mut engine, b, &[Vec2::new(-110.0, 0.0), Vec2::new(110.0, 0.0)]);
        Some(engine.render_to_image(PAPER))
    };
    let (Some(a), Some(b)) = (render(), render()) else {
        return;
    };
    assert!(
        images_match(&a, &b, 0),
        "the same dynamics script must render bit-identically"
    );
}

#[test]
fn golden_smudge_drag() {
    // A committed green bar, then a red smudge dragged left→right across it: the
    // tool picks up green where it crosses and carries a fading tint downstream
    // (DESIGN.md §6.2). Guards the look of the sequential stamp loop end-to-end.
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint(
        &mut engine,
        GREEN,
        38.0,
        &[Vec2::new(0.0, -90.0), Vec2::new(0.0, 90.0)],
    );
    let b = dyn_brush(
        RED,
        16.0,
        BrushDynamics { add: 0.1, load: 0.5, deposit: 0.5, ..Default::default() },
    );
    stroke_with(&mut engine, b, &[Vec2::new(-110.0, 0.0), Vec2::new(110.0, 0.0)]);
    let img = engine.render_to_image(PAPER);
    assert_golden("smudge_drag", &img, 6);
}

#[test]
fn golden_self_smear() {
    // A single stroke that crosses its own trail: because pickup reads the
    // *evolving* region, the second pass must smear the paint the first pass just
    // laid — the order-dependence the old base-only reservoir could not express.
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint(
        &mut engine,
        GREEN,
        30.0,
        &[Vec2::new(-60.0, 40.0), Vec2::new(60.0, 40.0)],
    );
    let b = dyn_brush(
        RED,
        18.0,
        BrushDynamics { add: 0.5, load: 0.6, deposit: 0.5, ..Default::default() },
    );
    // Down through the bar, loop around, and back across its own trail.
    engine.process(InputCommand::SetBrush(b));
    engine.process(InputCommand::StartStroke {
        tool: Tool::Brush,
        sample: InputSample::at(Vec2::new(-40.0, -70.0)),
    });
    for &(x, y) in &[(-40.0, 60.0), (20.0, 70.0), (40.0, 0.0), (-70.0, 10.0)] {
        engine.process(InputCommand::StrokeTo {
            sample: InputSample::at(Vec2::new(x, y)),
        });
    }
    engine.process(InputCommand::EndStroke);
    let img = engine.render_to_image(PAPER);
    assert_golden("self_smear", &img, 6);
}
