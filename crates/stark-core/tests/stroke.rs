//! Step-2 stroke MVP tests (DESIGN.md §13 build order, step 2): the
//! command/action split, copy-on-write tiles, and history undo/redo.

mod common;

use common::*;
use stark_core::command::{InputCommand, InputSample};
use stark_core::document::Tool;
use stark_core::geom::Vec2;
use stark_core::Engine;

const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];

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
