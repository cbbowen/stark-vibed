//! Region fill (§18.0.4).
//!
//! The claims worth pinning are the ones that make fill *this* app's fill rather
//! than a paint bucket: it deposits paint with real height (so it lights, and a
//! brush can lift it back), it is gated by the selection exactly as a stroke is,
//! its edge is the same feather ramp a marquee's is, and it is refused — not
//! guessed at — when nothing bounds it. Plus the ordinary action obligations:
//! undo, replay, and leaving distant paint bit-for-bit alone.

mod common;

use common::*;
use stark_core::RgbaImage;
use stark_core::command::{DocCommand, GestureCommand, InputSample, ViewCommand};
use stark_core::document::{
    BrushDynamics, BrushParams, FillOp, SelectionMode, SelectionOp, SelectionShape, ShapeAction,
    Tool,
};
use stark_core::geom::Vec2;
use stark_core::path::DEFAULT_TOLERANCE;

const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const GREEN: [f32; 4] = [0.1, 0.8, 0.2, 1.0];

/// A pixel's screen position for a canvas point, under the tests' identity view.
fn screen_of(canvas: Vec2) -> (u32, u32) {
    let half = Vec2::new(SIZE.width as f32, SIZE.height as f32) * 0.5;
    let p = canvas + half;
    (p.x as u32, p.y as u32)
}

fn texel(img: &RgbaImage, canvas: Vec2) -> [i32; 3] {
    let (x, y) = screen_of(canvas);
    let i = ((y * img.width + x) * 4) as usize;
    [
        img.pixels[i] as i32,
        img.pixels[i + 1] as i32,
        img.pixels[i + 2] as i32,
    ]
}

/// Whether the pixel at a canvas point reads as red paint rather than bare paper.
fn is_red(img: &RgbaImage, canvas: Vec2) -> bool {
    let [r, g, b] = texel(img, canvas);
    r - g > 40 && r - b > 40
}

fn is_green(img: &RgbaImage, canvas: Vec2) -> bool {
    let [r, g, b] = texel(img, canvas);
    g - r > 40 && g - b > 40
}

fn select_rect(engine: &mut stark_core::Engine, min: Vec2, max: Vec2, feather: f32) {
    engine.process(DocCommand::Select(SelectionOp::new(
        SelectionMode::Replace,
        SelectionShape::rect_from_corners(min, max),
        feather,
    )));
}

/// Commit a fill of the rect `min..max` with `color`, through the direct command
/// path. `height` is the amount of paint, as the brush's `add` would supply it.
fn fill_rect(
    engine: &mut stark_core::Engine,
    min: Vec2,
    max: Vec2,
    color: [f32; 4],
    feather: f32,
    height: f32,
) {
    let layer = engine.observe().active_layer;
    engine.process(DocCommand::Fill {
        layer,
        op: FillOp::new(
            SelectionShape::rect_from_corners(min, max),
            feather,
            color,
            height,
        ),
    });
}

/// Arm the Fill action with a red brush. A fill lays *the brush's* paint, so the
/// colour and the amount come off the Color and Brush panels rather than off
/// controls of Fill's own — which is what the gesture tests have to set up.
fn arm_fill(engine: &mut stark_core::Engine, color: [f32; 4]) {
    engine.process(ViewCommand::SetBrush(brush(color, 16.0)));
    engine.process(ViewCommand::SetShapeAction(ShapeAction::Fill));
}

/// Drag a rectangle out with the Fill action armed — the gesture path, which is
/// what a user actually does.
fn drag_fill(engine: &mut stark_core::Engine, min: Vec2, max: Vec2) {
    arm_fill(engine, RED);
    engine.process(GestureCommand::Start {
        tool: Tool::SelectRect,
        sample: InputSample::at(min),
        tolerance: DEFAULT_TOLERANCE,
    });
    engine.process(GestureCommand::To {
        sample: InputSample::at(max),
    });
    engine.process(GestureCommand::End);
}

#[test]
fn a_fill_lays_paint_inside_its_region_and_nothing_outside() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let before = engine.render_to_image();
    fill_rect(
        &mut engine,
        Vec2::new(-40.0, -40.0),
        Vec2::new(40.0, 40.0),
        RED,
        0.0,
        0.6,
    );
    let after = engine.render_to_image();

    assert!(is_red(&after, Vec2::ZERO), "the region should be filled");
    assert!(
        is_red(&after, Vec2::new(-35.0, 35.0)),
        "and filled to its corners"
    );
    assert!(
        !is_red(&after, Vec2::new(80.0, 80.0)),
        "paint outside the region"
    );
    // Far outside, the image must be bit-for-bit what it was: the shader's
    // zero-coverage branch is an exact pass-through, on purpose.
    let (x, y) = screen_of(Vec2::new(110.0, 110.0));
    let i = ((y * after.width + x) * 4) as usize;
    assert_eq!(
        after.pixels[i..i + 4],
        before.pixels[i..i + 4],
        "a fill disturbed paint outside its own coverage"
    );
}

#[test]
fn the_selection_bounds_the_fill() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // Select a small box, then fill a much larger one. The mask is the gate every
    // tool acts through (§6.8), so the fill must stop at the selection —
    // which is also what makes a fill of an infinite canvas well-defined.
    select_rect(
        &mut engine,
        Vec2::new(-20.0, -20.0),
        Vec2::new(20.0, 20.0),
        0.0,
    );
    fill_rect(
        &mut engine,
        Vec2::new(-90.0, -90.0),
        Vec2::new(90.0, 90.0),
        RED,
        0.0,
        0.6,
    );
    let img = engine.render_to_image();

    assert!(is_red(&img, Vec2::ZERO), "inside the selection");
    assert!(
        !is_red(&img, Vec2::new(60.0, 60.0)),
        "the fill escaped the selection"
    );
}

#[test]
fn filling_the_selection_needs_one_and_is_otherwise_refused() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let layer = engine.observe().active_layer;

    // Nothing selected: `All` is unbounded, and the action is rejected rather
    // than inventing a rectangle. The document must be untouched.
    let before = engine.render_to_image();
    engine.process(DocCommand::Fill {
        layer,
        op: FillOp::of_selection(RED, 0.6),
    });
    let after = engine.render_to_image();
    assert!(
        images_match(&before, &after, 0),
        "an unbounded fill changed the document"
    );

    // With a selection in force it is exactly the selection that gets filled —
    // the selection bar's button.
    select_rect(
        &mut engine,
        Vec2::new(-30.0, -30.0),
        Vec2::new(30.0, 30.0),
        0.0,
    );
    engine.process(DocCommand::Fill {
        layer,
        op: FillOp::of_selection(RED, 0.6),
    });
    let filled = engine.render_to_image();
    assert!(is_red(&filled, Vec2::ZERO));
    assert!(!is_red(&filled, Vec2::new(70.0, 70.0)));
}

#[test]
fn a_fill_stacks_over_resident_paint_rather_than_replacing_it() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint(
        &mut engine,
        GREEN,
        20.0,
        &[Vec2::new(-40.0, 0.0), Vec2::new(40.0, 0.0)],
    );
    let painted = engine.render_to_image();
    assert!(is_green(&painted, Vec2::ZERO));

    // A thin, low-opacity wash: a glaze, which must tint rather than cover. The
    // parcel law is the brush's own (`paint_common.wesl`), so this is the same
    // arithmetic a very slow brush over the same area would run.
    fill_rect(
        &mut engine,
        Vec2::new(-60.0, -60.0),
        Vec2::new(60.0, 60.0),
        [1.0, 0.0, 0.0, 0.15],
        0.0,
        0.05,
    );
    let glazed = engine.render_to_image();
    let [gr, gg, _] = texel(&glazed, Vec2::ZERO);
    let [pr, pg, _] = texel(&painted, Vec2::ZERO);
    assert!(
        gr > pr,
        "a thin red glaze should warm what is under it (was {pr},{pg} now {gr},{gg})"
    );
    assert!(
        is_green(&glazed, Vec2::ZERO),
        "a thin glaze covered the paint underneath it entirely \
         (was {pr},{pg} now {gr},{gg})"
    );

    // A heavy one over the same place does cover.
    fill_rect(
        &mut engine,
        Vec2::new(-60.0, -60.0),
        Vec2::new(60.0, 60.0),
        RED,
        0.0,
        2.0,
    );
    assert!(
        is_red(&engine.render_to_image(), Vec2::ZERO),
        "a thick fill should cover what is under it"
    );
}

#[test]
fn a_feathered_fill_thins_towards_its_edge() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    fill_rect(
        &mut engine,
        Vec2::new(-50.0, -50.0),
        Vec2::new(50.0, 50.0),
        RED,
        40.0,
        0.8,
    );
    let img = engine.render_to_image();
    // Coverage scales the *height* — the amount of paint — so a feathered edge is
    // a thinning of the deposit, not a fade of its colour. Sampled well inside,
    // near the boundary, and past the ramp.
    let inner = texel(&img, Vec2::ZERO);
    let edge = texel(&img, Vec2::new(48.0, 0.0));
    let outside = texel(&img, Vec2::new(90.0, 0.0));
    let redness = |t: [i32; 3]| t[0] - (t[1] + t[2]) / 2;
    assert!(
        redness(inner) > redness(edge),
        "the feathered edge is as thick as the middle ({inner:?} vs {edge:?})"
    );
    assert!(
        redness(edge) > redness(outside),
        "the feather ramp did not reach past the boundary"
    );
}

#[test]
fn the_gesture_path_fills_and_leaves_the_tool_armed() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    drag_fill(&mut engine, Vec2::new(-40.0, -40.0), Vec2::new(40.0, 40.0));

    assert!(
        is_red(&engine.render_to_image(), Vec2::ZERO),
        "the gesture laid no paint"
    );
    // The point of the test: unlike the four selecting actions, Fill does *not*
    // hand the canvas back to the brush. Blocking in is done many times in a row.
    assert_eq!(
        engine.observe().tool,
        Tool::SelectRect,
        "Fill disarmed the shape tool"
    );
    // And it committed a fill rather than a selection.
    assert!(
        !engine.observe().has_selection,
        "a fill gesture edited the selection"
    );
}

#[test]
fn a_selecting_gesture_still_disarms_the_tool() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.process(ViewCommand::SetShapeAction(ShapeAction::Select(
        SelectionMode::Replace,
    )));
    engine.process(GestureCommand::Start {
        tool: Tool::SelectRect,
        sample: InputSample::at(Vec2::new(-30.0, -30.0)),
        tolerance: DEFAULT_TOLERANCE,
    });
    engine.process(GestureCommand::To {
        sample: InputSample::at(Vec2::new(30.0, 30.0)),
    });
    engine.process(GestureCommand::End);

    assert_eq!(engine.observe().tool, Tool::Brush);
    assert!(engine.observe().has_selection);
}

#[test]
fn the_drag_previews_the_paint_it_will_commit() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    arm_fill(&mut engine, RED);
    engine.process(GestureCommand::Start {
        tool: Tool::SelectRect,
        sample: InputSample::at(Vec2::new(-40.0, -40.0)),
        tolerance: DEFAULT_TOLERANCE,
    });
    engine.process(GestureCommand::To {
        sample: InputSample::at(Vec2::new(40.0, 40.0)),
    });
    // Mid-drag: the region already reads as paint, not as an outline of where the
    // paint would go.
    let previewed = engine.render_to_image();
    assert!(
        is_red(&previewed, Vec2::ZERO),
        "the fill preview showed no paint"
    );

    engine.process(GestureCommand::End);
    let committed = engine.render_to_image();
    // `preview == committed`, the same invariant a stroke holds: the preview runs
    // the very call the commit does, over the same base.
    assert!(
        images_match(&previewed, &committed, 1),
        "the fill preview and the committed fill disagree"
    );
}

#[test]
fn the_preview_does_not_accumulate_as_the_drag_grows() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    arm_fill(&mut engine, RED);
    engine.process(GestureCommand::Start {
        tool: Tool::SelectRect,
        sample: InputSample::at(Vec2::new(-40.0, -40.0)),
        tolerance: DEFAULT_TOLERANCE,
    });
    // Fifty moves over the same box. Each rebuilds the preview from the committed
    // base rather than glazing over the last one, so the fiftieth must read the
    // same as the first — otherwise dragging slowly would deposit more paint than
    // dragging fast, which is the bug this shape of preview exists to avoid.
    engine.process(GestureCommand::To {
        sample: InputSample::at(Vec2::new(40.0, 40.0)),
    });
    let first = engine.render_to_image();
    for _ in 0..50 {
        engine.process(GestureCommand::To {
            sample: InputSample::at(Vec2::new(40.0, 40.0)),
        });
    }
    let fiftieth = engine.render_to_image();
    assert!(
        images_match(&first, &fiftieth, 1),
        "the fill preview accumulated across pointer moves"
    );
}

#[test]
fn undo_restores_exactly_what_the_fill_covered() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint(
        &mut engine,
        GREEN,
        18.0,
        &[Vec2::new(-50.0, 20.0), Vec2::new(50.0, 20.0)],
    );
    let before = engine.render_to_image();

    fill_rect(
        &mut engine,
        Vec2::new(-60.0, -60.0),
        Vec2::new(60.0, 60.0),
        RED,
        0.0,
        0.9,
    );
    assert!(is_red(&engine.render_to_image(), Vec2::ZERO));

    engine.process(DocCommand::Undo);
    let after = engine.render_to_image();
    assert!(
        images_match(&before, &after, 1),
        "undoing a fill did not restore the paint under it"
    );
}

#[test]
fn a_filled_region_can_be_scraped_back_by_a_lift_brush() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    fill_rect(
        &mut engine,
        Vec2::new(-60.0, -60.0),
        Vec2::new(60.0, 60.0),
        RED,
        0.0,
        0.9,
    );
    let filled = engine.render_to_image();
    assert!(is_red(&filled, Vec2::ZERO));

    // The whole argument for a fill depositing *height* rather than a flat colour:
    // what it lays is paint, so the wet-paint loop can pick it up again. A pure
    // scraper (`add = 0`, `lift = 1`) should take it off.
    let scraper = BrushParams {
        radius: 24.0,
        dynamics: BrushDynamics {
            add: 0.0,
            lift: 1.0,
            deposit: 0.0,
            charge: 0.0,
            bleed: 0.0,
        },
        ..BrushParams::default()
    };
    stroke_with(
        &mut engine,
        scraper,
        &[Vec2::new(-40.0, 0.0), Vec2::new(40.0, 0.0)],
    );
    assert!(
        !is_red(&engine.render_to_image(), Vec2::ZERO),
        "a lift brush could not scrape the fill back — it is not paint"
    );
}

#[test]
fn a_fill_survives_save_and_load() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    select_rect(
        &mut engine,
        Vec2::new(-45.0, -45.0),
        Vec2::new(45.0, 45.0),
        6.0,
    );
    fill_rect(
        &mut engine,
        Vec2::new(-70.0, -70.0),
        Vec2::new(70.0, 70.0),
        RED,
        4.0,
        0.7,
    );
    let before = engine.render_to_image();

    let bytes = engine.save_bytes().expect("save");
    engine.load_bytes(&bytes).expect("load");
    let after = engine.render_to_image();
    assert!(
        images_match(&before, &after, 2),
        "a fill did not replay identically from the log"
    );
}

#[test]
fn fill_golden() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // A feathered ellipse fill over a stroke, so the golden pins the edge ramp,
    // the stack over resident paint, and the lighting of the deposited height all
    // at once.
    paint(
        &mut engine,
        GREEN,
        22.0,
        &[Vec2::new(-70.0, -30.0), Vec2::new(70.0, 30.0)],
    );
    let layer = engine.observe().active_layer;
    engine.process(DocCommand::Fill {
        layer,
        op: FillOp::new(
            SelectionShape::ellipse_from_corners(Vec2::new(-60.0, -60.0), Vec2::new(60.0, 60.0)),
            18.0,
            [0.9, 0.2, 0.1, 0.8],
            0.7,
        ),
    });
    assert_golden("fill_ellipse", &engine.render_to_image(), 2);
}
