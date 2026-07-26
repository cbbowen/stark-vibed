//! Selections: the mask that gates where the brush may paint (DESIGN.md §6.8).
//!
//! Most of these assert the properties the feature rests on rather than an image:
//! paint lands inside and nowhere outside; the boolean modes compose; the *op*, not
//! the mask, is what history and the save file carry, so undo/redo, replay and load
//! all reproduce the same pixels; and the mask gates the brush-dynamics path (the
//! sequential stamp loop) as well as the plain swept deposit.
//!
//! The two **goldens** at the end cover what point probes cannot: the actual shape of
//! the mask across a whole frame — the analytic silhouettes, the width and profile of
//! the feather ramp, continuity across tile boundaries, and the outline overlay — for
//! each of the two masking sites in turn (DESIGN.md §9).

mod common;

use common::*;
use stark_core::command::{InputCommand, InputSample};
use stark_core::document::{BrushDynamics, Tool};
use stark_core::geom::Vec2;
use stark_core::{RgbaImage, SelectionMode, SelectionOp, SelectionShape};

const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];

/// Canvas-space box the tests select and paint across. Comfortably inside the 256²
/// viewport, which is centred on the canvas origin.
const BOX_MIN: Vec2 = Vec2::new(-40.0, -40.0);
const BOX_MAX: Vec2 = Vec2::new(0.0, 40.0);

/// A pixel's screen position for a canvas point, under the tests' identity view.
fn screen_of(canvas: Vec2) -> (u32, u32) {
    let half = Vec2::new(SIZE.width as f32, SIZE.height as f32) * 0.5;
    let p = canvas + half;
    (p.x as u32, p.y as u32)
}

/// Whether the pixel at a canvas point reads as paint rather than bare paper. The
/// substrate is neutral near-white, so any strongly red-dominant pixel is paint.
fn is_painted(img: &RgbaImage, canvas: Vec2) -> bool {
    let (x, y) = screen_of(canvas);
    let i = ((y * img.width + x) * 4) as usize;
    let (r, g, b) = (img.pixels[i] as i32, img.pixels[i + 1] as i32, img.pixels[i + 2] as i32);
    r - g > 40 && r - b > 40
}

fn select(engine: &mut stark_core::Engine, mode: SelectionMode, shape: SelectionShape) {
    engine.process(InputCommand::Select(SelectionOp::new(mode, shape, 0.0)));
}

fn rect(min: Vec2, max: Vec2) -> SelectionShape {
    SelectionShape::rect_from_corners(min, max)
}

/// A horizontal stroke that crosses the selection boundary at x = 0, so the same
/// gesture has a masked half and an unmasked half.
fn crossing_stroke(engine: &mut stark_core::Engine) {
    paint(
        engine,
        RED,
        14.0,
        &[Vec2::new(-30.0, 0.0), Vec2::new(0.0, 0.0), Vec2::new(30.0, 0.0)],
    );
}

#[test]
fn selection_clips_the_brush() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    select(&mut engine, SelectionMode::Replace, rect(BOX_MIN, BOX_MAX));
    crossing_stroke(&mut engine);
    let img = engine.render_to_image(PAPER);

    assert!(
        is_painted(&img, Vec2::new(-20.0, 0.0)),
        "the selected half of the stroke should be painted"
    );
    assert!(
        !is_painted(&img, Vec2::new(20.0, 0.0)),
        "the deselected half of the stroke must not reach the canvas"
    );
}

#[test]
fn no_selection_paints_everywhere() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    crossing_stroke(&mut engine);
    let img = engine.render_to_image(PAPER);
    assert!(is_painted(&img, Vec2::new(-20.0, 0.0)));
    assert!(is_painted(&img, Vec2::new(20.0, 0.0)));
}

#[test]
fn deselecting_restores_the_whole_canvas() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    select(&mut engine, SelectionMode::Replace, rect(BOX_MIN, BOX_MAX));
    assert!(engine.observe().has_selection);
    engine.process(InputCommand::Select(SelectionOp::select_all()));
    assert!(
        !engine.observe().has_selection,
        "selecting everything is indistinguishable from having no selection"
    );

    crossing_stroke(&mut engine);
    let img = engine.render_to_image(PAPER);
    assert!(is_painted(&img, Vec2::new(20.0, 0.0)));
}

#[test]
fn inverting_swaps_which_half_paints() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    select(&mut engine, SelectionMode::Replace, rect(BOX_MIN, BOX_MAX));
    engine.process(InputCommand::InvertSelection);
    crossing_stroke(&mut engine);
    let img = engine.render_to_image(PAPER);

    assert!(!is_painted(&img, Vec2::new(-20.0, 0.0)), "inverted: left is now masked out");
    assert!(is_painted(&img, Vec2::new(20.0, 0.0)), "inverted: right now paints");
}

#[test]
fn union_extends_and_subtract_cuts() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // Select the left box, add the right one, then cut a notch out of the middle.
    select(&mut engine, SelectionMode::Replace, rect(BOX_MIN, BOX_MAX));
    select(
        &mut engine,
        SelectionMode::Union,
        rect(Vec2::new(10.0, -40.0), Vec2::new(40.0, 40.0)),
    );
    select(
        &mut engine,
        SelectionMode::Subtract,
        rect(Vec2::new(-25.0, -40.0), Vec2::new(-15.0, 40.0)),
    );
    crossing_stroke(&mut engine);
    let img = engine.render_to_image(PAPER);

    assert!(is_painted(&img, Vec2::new(-8.0, 0.0)), "original selection, uncut");
    assert!(is_painted(&img, Vec2::new(25.0, 0.0)), "added by the union");
    assert!(!is_painted(&img, Vec2::new(-20.0, 0.0)), "removed by the subtract");
    assert!(!is_painted(&img, Vec2::new(5.0, 0.0)), "never selected at all");
}

#[test]
fn intersect_keeps_only_the_overlap() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    select(&mut engine, SelectionMode::Replace, rect(BOX_MIN, BOX_MAX));
    select(
        &mut engine,
        SelectionMode::Intersect,
        rect(Vec2::new(-20.0, -40.0), Vec2::new(30.0, 40.0)),
    );
    crossing_stroke(&mut engine);
    let img = engine.render_to_image(PAPER);

    assert!(is_painted(&img, Vec2::new(-10.0, 0.0)), "in both");
    assert!(!is_painted(&img, Vec2::new(-30.0, 0.0)), "only in the first");
    assert!(!is_painted(&img, Vec2::new(20.0, 0.0)), "only in the second");
}

#[test]
fn lasso_masks_its_interior() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // A triangle covering the left half of the stroke's path.
    let tri = SelectionShape::Lasso(vec![
        Vec2::new(-60.0, -50.0),
        Vec2::new(-60.0, 50.0),
        Vec2::new(-5.0, 0.0),
    ]);
    select(&mut engine, SelectionMode::Replace, tri);
    crossing_stroke(&mut engine);
    let img = engine.render_to_image(PAPER);

    // Probed within the stroke's body, not out at its cap, so this measures the mask
    // rather than the brush's falloff.
    assert!(is_painted(&img, Vec2::new(-20.0, 0.0)), "inside the triangle");
    assert!(!is_painted(&img, Vec2::new(20.0, 0.0)), "outside it");
    // Just past the apex on the stroke's line, still outside the polygon.
    assert!(!is_painted(&img, Vec2::new(2.0, 0.0)), "beyond the apex");
}

#[test]
fn selection_gates_the_brush_dynamics_path() {
    // `lift`/`deposit` route the stroke through the sequential stamp loop rather than
    // the swept fast path (DESIGN.md §6.2) — a completely separate masking site, so it
    // gets its own check.
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let mut b = brush(RED, 14.0);
    b.dynamics = BrushDynamics {
        add: 1.0,
        lift: 0.3,
        deposit: 0.8,
        charge: 0.5,
        ..BrushDynamics::default()
    };
    select(&mut engine, SelectionMode::Replace, rect(BOX_MIN, BOX_MAX));
    stroke_with(
        &mut engine,
        b,
        &[Vec2::new(-30.0, 0.0), Vec2::new(0.0, 0.0), Vec2::new(30.0, 0.0)],
    );
    let img = engine.render_to_image(PAPER);

    assert!(is_painted(&img, Vec2::new(-20.0, 0.0)), "smearing brush paints inside");
    assert!(
        !is_painted(&img, Vec2::new(20.0, 0.0)),
        "smearing brush must not touch the canvas outside the selection"
    );
}

#[test]
fn dynamics_brush_does_not_lift_paint_from_outside() {
    // A pure smudge (no `add`) dragged from unselected canvas into the selection must
    // not carry paint across the boundary: the lift is masked on the pickup side too.
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // Lay a bar of paint on the right (which will be *outside* the selection).
    paint(&mut engine, RED, 20.0, &[Vec2::new(20.0, 0.0), Vec2::new(60.0, 0.0)]);
    select(&mut engine, SelectionMode::Replace, rect(BOX_MIN, BOX_MAX));

    let mut smudge = brush([0.0, 0.0, 0.0, 1.0], 16.0);
    smudge.dynamics = BrushDynamics {
        add: 0.0,
        lift: 0.9,
        deposit: 0.9,
        ..BrushDynamics::default()
    };
    // Drag right-to-left, from the painted bar into the selected empty region.
    stroke_with(
        &mut engine,
        smudge,
        &[Vec2::new(50.0, 0.0), Vec2::new(0.0, 0.0), Vec2::new(-30.0, 0.0)],
    );
    let img = engine.render_to_image(PAPER);

    assert!(
        !is_painted(&img, Vec2::new(-25.0, 0.0)),
        "no paint may be smeared in from outside the selection"
    );
}

#[test]
fn undo_and_redo_step_through_the_selection() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let bare = engine.render_to_image(PAPER);
    select(&mut engine, SelectionMode::Replace, rect(BOX_MIN, BOX_MAX));
    crossing_stroke(&mut engine);
    let masked = engine.render_to_image(PAPER);

    engine.process(InputCommand::Undo); // the stroke
    assert!(engine.observe().has_selection, "the selection outlives the stroke");
    engine.process(InputCommand::Undo); // the selection itself
    assert!(!engine.observe().has_selection);
    assert!(
        images_match(&bare, &engine.render_to_image(PAPER), 1),
        "undoing back past the selection returns the empty canvas"
    );

    engine.process(InputCommand::Redo); // the selection
    engine.process(InputCommand::Redo); // the stroke
    assert!(engine.observe().has_selection);
    assert!(
        images_match(&masked, &engine.render_to_image(PAPER), 1),
        "redoing through a selection must reproduce the same pixels"
    );
}

#[test]
fn undoing_the_selection_unmasks_later_strokes() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    select(&mut engine, SelectionMode::Replace, rect(BOX_MIN, BOX_MAX));
    engine.process(InputCommand::Undo);
    // The new stroke starts a fresh branch, on a document with no selection: it is
    // gated by the selection *in force at its point in the log*, which is none.
    crossing_stroke(&mut engine);
    let img = engine.render_to_image(PAPER);
    assert!(is_painted(&img, Vec2::new(20.0, 0.0)));
}

#[test]
fn selection_survives_save_and_load() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    select(&mut engine, SelectionMode::Replace, rect(BOX_MIN, BOX_MAX));
    crossing_stroke(&mut engine);
    let before = engine.render_to_image(PAPER);
    let bytes = engine.save_bytes().expect("save");

    let Some(mut loaded) = engine_or_skip() else {
        return;
    };
    loaded.load_bytes(&bytes).expect("load");
    let after = loaded.render_to_image(PAPER);
    assert!(
        images_match(&before, &after, 2),
        "the log carries the selection *op*, so a reload replays the same mask"
    );
    assert!(
        loaded.observe().has_selection,
        "the selection is still in force after loading"
    );
}

#[test]
fn a_selection_gesture_commits_the_same_op_it_previewed() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // Drag a rectangular marquee, then paint through it.
    engine.process(InputCommand::StartStroke {
        tool: Tool::SelectRect,
        sample: InputSample::at(BOX_MIN),
    });
    engine.process(InputCommand::StrokeTo {
        sample: InputSample::at(Vec2::new(-20.0, 0.0)),
    });
    engine.process(InputCommand::StrokeTo {
        sample: InputSample::at(BOX_MAX),
    });
    // Mid-gesture the preview already shows a selection, but nothing is committed.
    assert!(!engine.observe().has_selection, "still just a preview");
    engine.process(InputCommand::EndStroke);
    assert!(engine.observe().has_selection, "release commits the op");

    crossing_stroke(&mut engine);
    let img = engine.render_to_image(PAPER);
    assert!(is_painted(&img, Vec2::new(-20.0, 0.0)));
    assert!(!is_painted(&img, Vec2::new(20.0, 0.0)));
}

#[test]
fn a_click_with_a_marquee_selects_nothing() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.process(InputCommand::StartStroke {
        tool: Tool::SelectRect,
        sample: InputSample::at(Vec2::ZERO),
    });
    engine.process(InputCommand::EndStroke);
    assert!(
        !engine.observe().has_selection,
        "a zero-area marquee is not a selection"
    );
}

#[test]
fn the_outline_is_drawn_on_the_boundary_only() {
    // The overlay pass recovers the contour from the mask's own gradient (§6.8), so
    // this checks it lands on the boundary and nowhere else — including that an
    // unmasked document draws nothing at all.
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let bare = engine.render_to_image(PAPER);
    select(&mut engine, SelectionMode::Replace, rect(BOX_MIN, BOX_MAX));
    let outlined = engine.render_to_image(PAPER);

    assert!(
        !images_match(&bare, &outlined, 8),
        "an active selection must be visible on screen"
    );

    // Differences must hug the rectangle's edges: sample a band across the right
    // boundary (x = 0) and well inside/outside it.
    let changed = |p: Vec2| {
        let (x, y) = screen_of(p);
        let i = ((y * bare.width + x) * 4) as usize;
        (0..3)
            .map(|k| (bare.pixels[i + k] as i32 - outlined.pixels[i + k] as i32).abs())
            .max()
            .unwrap_or(0)
            > 8
    };
    assert!(changed(Vec2::new(0.0, 0.0)), "the boundary is outlined");
    assert!(!changed(Vec2::new(-20.0, 0.0)), "the interior is left alone");
    assert!(!changed(Vec2::new(20.0, 0.0)), "the exterior is left alone");
}

#[test]
fn feathered_edge_fades_the_stroke() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // Same geometry, hard vs. heavily feathered edge. Just inside the boundary the
    // feathered mask must be visibly weaker than the hard one.
    let probe = Vec2::new(-4.0, 0.0);
    engine.process(InputCommand::Select(SelectionOp::new(
        SelectionMode::Replace,
        rect(BOX_MIN, BOX_MAX),
        0.0,
    )));
    crossing_stroke(&mut engine);
    let hard = engine.render_to_image(PAPER);

    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.process(InputCommand::Select(SelectionOp::new(
        SelectionMode::Replace,
        rect(BOX_MIN, BOX_MAX),
        20.0,
    )));
    crossing_stroke(&mut engine);
    let soft = engine.render_to_image(PAPER);

    let redness = |img: &RgbaImage, p: Vec2| {
        let (x, y) = screen_of(p);
        let i = ((y * img.width + x) * 4) as usize;
        img.pixels[i] as i32 - img.pixels[i + 1] as i32
    };
    assert!(
        redness(&soft, probe) + 20 < redness(&hard, probe),
        "feathered edge should be markedly weaker just inside the boundary \
         (soft {}, hard {})",
        redness(&soft, probe),
        redness(&hard, probe)
    );
    // Well inside — further from either vertical boundary than the ramp is wide — the
    // two should agree: feather affects the edge, not the interior.
    let deep = Vec2::new(-20.0, 0.0);
    assert!(
        (redness(&soft, deep) - redness(&hard, deep)).abs() < 20,
        "feather must not change the selection's interior"
    );
}

// --- goldens (DESIGN.md §9) --------------------------------------------------
//
// Point probes can say "paint here, none there"; they cannot say what the mask
// *looks like*. These two do: the analytic silhouettes and how the three boolean
// modes compose into one, the width and profile of the feather ramp, that the mask is
// continuous where it crosses a tile boundary (it is rasterized per tile, so a
// discontinuity there is a real failure mode — §6.4), and the outline overlay, which
// nothing else renders. One golden per masking site, since they are separate code.

/// A stripe field covering the whole viewport, in alternating colours — the paint the
/// selection stencils. Broad and edge-to-edge on purpose: every band crosses the mask
/// boundary, and the bands run past the viewport so no stroke cap is mistaken for a
/// mask edge.
fn paint_stripes(engine: &mut stark_core::Engine) {
    const TEAL: [f32; 4] = [0.0, 0.55, 0.6, 1.0];
    for (i, y) in [-75.0f32, -25.0, 25.0, 75.0].into_iter().enumerate() {
        let color = if i % 2 == 0 { RED } else { TEAL };
        paint(
            engine,
            color,
            22.0,
            &[Vec2::new(-140.0, y), Vec2::new(140.0, y)],
        );
    }
}

#[test]
fn golden_selection_stencil() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // One selection built from all three composing modes and all three analytic
    // shapes, feathered enough that the ramp is several pixels wide on screen.
    const FEATHER: f32 = 6.0;
    let op = |mode, shape| SelectionOp::new(mode, shape, FEATHER);
    engine.process(InputCommand::Select(op(
        SelectionMode::Replace,
        rect(Vec2::new(-100.0, -74.0), Vec2::new(6.0, 74.0)),
    )));
    engine.process(InputCommand::Select(op(
        SelectionMode::Union,
        SelectionShape::Ellipse {
            center: Vec2::new(46.0, -4.0),
            radii: Vec2::new(58.0, 62.0),
        },
    )));
    engine.process(InputCommand::Select(op(
        SelectionMode::Subtract,
        SelectionShape::Lasso(vec![
            Vec2::new(-46.0, -90.0),
            Vec2::new(22.0, 14.0),
            Vec2::new(-84.0, 34.0),
        ]),
    )));

    // With the mask in force, the stripes come out cut to its silhouette, with the
    // feather visible as a fade at every crossing — and the outline drawn over it.
    paint_stripes(&mut engine);

    let img = engine.render_to_image(PAPER);
    assert_golden("selection_stencil", &img, 6);
}

#[test]
fn golden_selection_smear() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // The brush-dynamics path masks somewhere else entirely — the stamp loop's
    // deposit/pickup, against a mask gathered into the stroke's region (§6.8) — so it
    // gets its own frame. Stripes go down *first*, unmasked, to give the tool paint to
    // carry; the selection then decides where that paint may be moved.
    paint_stripes(&mut engine);
    engine.process(InputCommand::Select(SelectionOp::new(
        SelectionMode::Replace,
        SelectionShape::Ellipse {
            center: Vec2::ZERO,
            radii: Vec2::new(84.0, 66.0),
        },
        5.0,
    )));

    // A conservative smudge: no paint of its own, so everything that moves was lifted
    // from the canvas — and none of it may cross the boundary in either direction.
    let mut smudge = brush([0.0, 0.0, 0.0, 1.0], 26.0);
    smudge.dynamics = BrushDynamics {
        add: 0.0,
        lift: 0.7,
        deposit: 0.85,
        ..BrushDynamics::default()
    };
    stroke_with(
        &mut engine,
        smudge,
        &[
            Vec2::new(-120.0, -40.0),
            Vec2::new(-40.0, 30.0),
            Vec2::new(40.0, -30.0),
            Vec2::new(120.0, 40.0),
        ],
    );

    let img = engine.render_to_image(PAPER);
    assert_golden("selection_smear", &img, 6);
}
