//! Step-2 stroke MVP tests (DESIGN.md §13 build order, step 2): the
//! command/action split, copy-on-write tiles, and history undo/redo.

mod common;

use common::*;
use stark_core::command::{InputCommand, InputSample};
use stark_core::document::{BrushDynamics, BrushParams, DryParams, Tool};
use stark_core::geom::Vec2;
use stark_core::Engine;

const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];

/// A brush with the given [`DryParams`] dynamics.
fn dry_brush(color: [f32; 4], radius: f32, p: DryParams) -> BrushParams {
    let mut b = brush(color, radius);
    b.dynamics = BrushDynamics::Dry(p);
    b
}

/// Paint and commit a stroke through `points` with an explicit brush.
fn stroke_with(engine: &mut Engine, b: BrushParams, points: &[Vec2]) {
    engine.process(InputCommand::SetBrush(b));
    let mut it = points.iter();
    let first = *it.next().expect("at least one point");
    engine.process(InputCommand::StartStroke {
        tool: Tool::Brush,
        sample: InputSample::at(first),
    });
    for &p in it {
        engine.process(InputCommand::StrokeTo {
            sample: InputSample::at(p),
        });
    }
    engine.process(InputCommand::EndStroke);
}

fn paint_stroke(engine: &mut Engine) {
    paint(
        engine,
        RED,
        40.0,
        &[
            Vec2::new(-30.0, 0.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(30.0, 0.0),
        ],
    );
}

// Lit paint is never a pure primary, so assert channel *dominance* rather than
// near-saturation (the image-based-lighting media pass legitimately shades and
// desaturates color — a warm studio tint and gloss sheen lift the other channels).
fn is_red(c: [u8; 4]) -> bool {
    c[0] as i32 > c[1] as i32 + 20 && c[0] as i32 > c[2] as i32 + 20
}
fn is_blue(c: [u8; 4]) -> bool {
    c[2] as i32 > c[0] as i32 + 20 && c[2] as i32 > c[1] as i32 + 20
}
fn center(img: &stark_core::RgbaImage) -> [u8; 4] {
    img.pixel(img.width / 2, img.height / 2)
}

#[test]
fn live_preview_shows_stroke_before_commit() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // Build an in-flight stroke without ending it.
    engine.process(InputCommand::SetBrush(brush(RED, 40.0)));
    engine.process(InputCommand::StartStroke {
        tool: Tool::Brush,
        sample: InputSample::at(Vec2::new(-30.0, 0.0)),
    });
    engine.process(InputCommand::StrokeTo {
        sample: InputSample::at(Vec2::new(30.0, 0.0)),
    });

    assert!(engine.observe().is_stroking);
    let preview = engine.render_to_image(BG);
    assert!(is_red(center(&preview)), "preview should show the stroke");
}

#[test]
fn stroke_commit_undo_redo() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint_stroke(&mut engine);
    assert!(!engine.observe().is_stroking);
    assert!(engine.observe().can_undo);

    let committed = engine.render_to_image(BG);
    assert!(is_red(center(&committed)), "committed center should be red");
    assert!(
        is_blue(committed.pixel(10, 10)),
        "untouched corner should be background blue"
    );

    engine.process(InputCommand::Undo);
    assert!(engine.observe().can_redo);
    assert!(
        is_blue(center(&engine.render_to_image(BG))),
        "after undo, center should be background"
    );

    engine.process(InputCommand::Redo);
    assert!(
        is_red(center(&engine.render_to_image(BG))),
        "after redo, center should be red again"
    );
}

#[test]
fn stroke_spans_multiple_tiles_via_cow() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint_stroke(&mut engine);

    // A radius-40 stroke straddling the canvas origin touches all four tiles
    // around (0,0); copy-on-write should have populated more than one.
    let populated: usize = engine.document().layers.iter().map(|l| l.tiles.size()).sum();
    assert!(
        populated >= 2,
        "stroke across the origin should populate multiple tiles, got {populated}"
    );
}

// ---- Conservative dry-brush dynamics (DESIGN.md §6.2): add / lift / deposit ----

#[test]
fn conservative_smear_preserves_uniform_field() {
    // With `add = 0` the brush only *moves* paint between canvas and tool. Smearing
    // entirely *within* a uniform field has nothing to move — the lift takes paint up and
    // the deposit lays the same colour straight back — so the field must stay essentially
    // unchanged. (The over-blend deposit is not bit-exact, so a generous tolerance; the
    // point is paint is neither destroyed nor multiplied — an over-eager smear would either
    // gouge the field or pile it up.)
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // A broad solid red field across the middle of the canvas.
    paint(
        &mut engine,
        RED,
        80.0,
        &[Vec2::new(-110.0, 0.0), Vec2::new(110.0, 0.0)],
    );
    let before = engine.render_to_image(PAPER);

    // A conservative smear (add = 0) kept well inside the field.
    let b = dry_brush(RED, 24.0, DryParams { add: 0.0, lift: 0.5, deposit: 0.5, ridge: 0.0 });
    stroke_with(&mut engine, b, &[Vec2::new(-50.0, 0.0), Vec2::new(50.0, 0.0)]);
    let after = engine.render_to_image(PAPER);

    assert!(
        is_red(center(&after)),
        "paint must not be destroyed by a conservative smear of a uniform field"
    );
    let frac = frac_exceeding(&before, &after, 40);
    assert!(
        frac < 0.2,
        "smearing a uniform field should move almost nothing, but {:.1}% of pixels changed",
        frac * 100.0
    );
}

#[test]
fn lift_deposit_carries_paint_onto_bare_canvas() {
    // The headline behaviour: dragging an `add = 0` brush from a painted patch onto bare
    // canvas deposits the carried paint there. Because `add = 0`, the brush lays none of
    // its own paint — so any red that appears on the previously-bare runway *must* be paint
    // lifted off the patch and carried over (conservation). We measure the green channel:
    // depositing red onto the warm paper lowers it. (`is_red` can't be used — the studio-lit
    // paper is itself warm/red-dominant, so it isn't a paint discriminator.)
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let y = SIZE.height / 2;
    let run_x = 116; // canvas ≈ -12, bare runway just past the patch's right edge

    // A red patch on the left; the right runway starts bare.
    paint(
        &mut engine,
        RED,
        40.0,
        &[Vec2::new(-90.0, 0.0), Vec2::new(-50.0, 0.0)],
    );
    let run_before = engine.render_to_image(PAPER).pixel(run_x, y);
    assert!(run_before[1] > 170, "the runway should start bare paper: {run_before:?}");

    // Drag from inside the patch rightward across the bare runway.
    let b = dry_brush(RED, 28.0, DryParams { add: 0.0, lift: 0.9, deposit: 0.3, ridge: 0.0 });
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
fn lift_over_empty_canvas_adds_nothing() {
    // There is no paint to lift on bare canvas, so the tool stays empty and the brush lays
    // nothing — no colour and no height (a regression guard for the old bug that raised
    // relief out of nothing).
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let blank = engine.render_to_image(PAPER);
    let b = dry_brush(RED, 24.0, DryParams { add: 0.0, lift: 1.0, deposit: 1.0, ridge: 0.0 });
    stroke_with(&mut engine, b, &[Vec2::new(-40.0, 0.0), Vec2::new(40.0, 0.0)]);
    let after = engine.render_to_image(PAPER);
    assert!(
        images_match(&blank, &after, 2),
        "lift over empty canvas must add no paint or height"
    );
}
