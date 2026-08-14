//! Selections: the mask that gates where the brush may paint (§6.8).
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
//! each of the two masking sites in turn (§9).

mod common;

use common::*;
use stark_core::command::{DocCommand, GestureCommand, InputSample, ViewCommand};
use stark_core::document::{BrushDynamics, Tool};
use stark_core::geom::Vec2;
use stark_core::path::DEFAULT_TOLERANCE;
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
    let (r, g, b) = (
        img.pixels[i] as i32,
        img.pixels[i + 1] as i32,
        img.pixels[i + 2] as i32,
    );
    r - g > 40 && r - b > 40
}

fn select(engine: &mut stark_core::Engine, mode: SelectionMode, shape: SelectionShape) {
    engine.process(DocCommand::Select(SelectionOp::new(mode, shape, 0.0)));
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
        &[
            Vec2::new(-30.0, 0.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(30.0, 0.0),
        ],
    );
}

#[test]
fn selection_clips_the_brush() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    select(&mut engine, SelectionMode::Replace, rect(BOX_MIN, BOX_MAX));
    crossing_stroke(&mut engine);
    let img = engine.render_to_image();

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
    let img = engine.render_to_image();
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
    engine.process(DocCommand::Select(SelectionOp::select_all()));
    assert!(
        !engine.observe().has_selection,
        "selecting everything is indistinguishable from having no selection"
    );

    crossing_stroke(&mut engine);
    let img = engine.render_to_image();
    assert!(is_painted(&img, Vec2::new(20.0, 0.0)));
}

#[test]
fn inverting_swaps_which_half_paints() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    select(&mut engine, SelectionMode::Replace, rect(BOX_MIN, BOX_MAX));
    engine.process(DocCommand::InvertSelection);
    crossing_stroke(&mut engine);
    let img = engine.render_to_image();

    assert!(
        !is_painted(&img, Vec2::new(-20.0, 0.0)),
        "inverted: left is now masked out"
    );
    assert!(
        is_painted(&img, Vec2::new(20.0, 0.0)),
        "inverted: right now paints"
    );
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
    let img = engine.render_to_image();

    assert!(
        is_painted(&img, Vec2::new(-8.0, 0.0)),
        "original selection, uncut"
    );
    assert!(is_painted(&img, Vec2::new(25.0, 0.0)), "added by the union");
    assert!(
        !is_painted(&img, Vec2::new(-20.0, 0.0)),
        "removed by the subtract"
    );
    assert!(
        !is_painted(&img, Vec2::new(5.0, 0.0)),
        "never selected at all"
    );
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
    let img = engine.render_to_image();

    assert!(is_painted(&img, Vec2::new(-10.0, 0.0)), "in both");
    assert!(
        !is_painted(&img, Vec2::new(-30.0, 0.0)),
        "only in the first"
    );
    assert!(
        !is_painted(&img, Vec2::new(20.0, 0.0)),
        "only in the second"
    );
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
    let img = engine.render_to_image();

    // Probed within the stroke's body, not out at its cap, so this measures the mask
    // rather than the brush's falloff.
    assert!(
        is_painted(&img, Vec2::new(-20.0, 0.0)),
        "inside the triangle"
    );
    assert!(!is_painted(&img, Vec2::new(20.0, 0.0)), "outside it");
    // Just past the apex on the stroke's line, still outside the polygon.
    assert!(!is_painted(&img, Vec2::new(2.0, 0.0)), "beyond the apex");
}

#[test]
fn selection_gates_the_brush_dynamics_path() {
    // `lift`/`deposit` route the stroke through the sequential stamp loop rather than
    // the swept fast path (§6.2) — a completely separate masking site, so it
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
        bleed: 0.0,
    };
    select(&mut engine, SelectionMode::Replace, rect(BOX_MIN, BOX_MAX));
    stroke_with(
        &mut engine,
        b,
        &[
            Vec2::new(-30.0, 0.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(30.0, 0.0),
        ],
    );
    let img = engine.render_to_image();

    assert!(
        is_painted(&img, Vec2::new(-20.0, 0.0)),
        "smearing brush paints inside"
    );
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
    paint(
        &mut engine,
        RED,
        20.0,
        &[Vec2::new(20.0, 0.0), Vec2::new(60.0, 0.0)],
    );
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
        &[
            Vec2::new(50.0, 0.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(-30.0, 0.0),
        ],
    );
    let img = engine.render_to_image();

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
    let bare = engine.render_to_image();
    select(&mut engine, SelectionMode::Replace, rect(BOX_MIN, BOX_MAX));
    crossing_stroke(&mut engine);
    let masked = engine.render_to_image();

    engine.process(DocCommand::Undo); // the stroke
    assert!(
        engine.observe().has_selection,
        "the selection outlives the stroke"
    );
    engine.process(DocCommand::Undo); // the selection itself
    assert!(!engine.observe().has_selection);
    assert!(
        images_match(&bare, &engine.render_to_image(), 1),
        "undoing back past the selection returns the empty canvas"
    );

    engine.process(DocCommand::Redo); // the selection
    engine.process(DocCommand::Redo); // the stroke
    assert!(engine.observe().has_selection);
    assert!(
        images_match(&masked, &engine.render_to_image(), 1),
        "redoing through a selection must reproduce the same pixels"
    );
}

#[test]
fn undoing_the_selection_unmasks_later_strokes() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    select(&mut engine, SelectionMode::Replace, rect(BOX_MIN, BOX_MAX));
    engine.process(DocCommand::Undo);
    // The new stroke starts a fresh branch, on a document with no selection: it is
    // gated by the selection *in force at its point in the log*, which is none.
    crossing_stroke(&mut engine);
    let img = engine.render_to_image();
    assert!(is_painted(&img, Vec2::new(20.0, 0.0)));
}

#[test]
fn selection_survives_save_and_load() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    select(&mut engine, SelectionMode::Replace, rect(BOX_MIN, BOX_MAX));
    crossing_stroke(&mut engine);
    let before = engine.render_to_image();
    let bytes = engine.save_bytes().expect("save");

    let Some(mut loaded) = engine_or_skip() else {
        return;
    };
    loaded.load_bytes(&bytes).expect("load");
    let after = loaded.render_to_image();
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
    engine.process(GestureCommand::Start {
        tool: Tool::SelectRect,
        sample: InputSample::at(BOX_MIN),
        tolerance: DEFAULT_TOLERANCE,
    });
    engine.process(GestureCommand::To {
        sample: InputSample::at(Vec2::new(-20.0, 0.0)),
    });
    engine.process(GestureCommand::To {
        sample: InputSample::at(BOX_MAX),
    });
    // Mid-gesture the preview already shows a selection, but nothing is committed.
    assert!(!engine.observe().has_selection, "still just a preview");
    engine.process(GestureCommand::End);
    assert!(engine.observe().has_selection, "release commits the op");

    crossing_stroke(&mut engine);
    let img = engine.render_to_image();
    assert!(is_painted(&img, Vec2::new(-20.0, 0.0)));
    assert!(!is_painted(&img, Vec2::new(20.0, 0.0)));
}

#[test]
fn a_click_with_a_marquee_selects_nothing() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.process(GestureCommand::Start {
        tool: Tool::SelectRect,
        sample: InputSample::at(Vec2::ZERO),
        tolerance: DEFAULT_TOLERANCE,
    });
    engine.process(GestureCommand::End);
    assert!(
        !engine.observe().has_selection,
        "a zero-area marquee is not a selection"
    );
    assert_eq!(
        engine.observe().tool,
        Tool::SelectRect,
        "a stray click is not a selection, so it must not disarm the tool either"
    );
}

#[test]
fn drawing_a_selection_disarms_the_tool() {
    // Selection tools are momentary (§6.8): selecting is a step towards
    // painting, so the canvas comes back to the brush of its own accord rather than
    // leaving a mode the next gesture would silently redefine the selection through.
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    for tool in [Tool::SelectRect, Tool::SelectEllipse, Tool::SelectLasso] {
        engine.process(GestureCommand::Start {
            tool,
            sample: InputSample::at(BOX_MIN),
            tolerance: DEFAULT_TOLERANCE,
        });
        // A marquee spans the drag's two corners; a lasso needs a traced interior.
        let path: &[Vec2] = if tool == Tool::SelectLasso {
            &[
                Vec2::new(BOX_MAX.x, BOX_MIN.y),
                BOX_MAX,
                Vec2::new(BOX_MIN.x, BOX_MAX.y),
            ]
        } else {
            &[BOX_MAX]
        };
        for &p in path {
            engine.process(GestureCommand::To {
                sample: InputSample::at(p),
            });
        }
        assert_eq!(
            engine.observe().tool,
            tool,
            "{tool:?} stays armed for the length of its own gesture"
        );

        engine.process(GestureCommand::End);
        assert!(engine.observe().has_selection, "{tool:?} drew a selection");
        assert_eq!(
            engine.observe().tool,
            Tool::Brush,
            "{tool:?} hands the canvas back to the brush once it has drawn"
        );
    }
}

#[test]
fn a_stroke_after_a_selection_paints_rather_than_reselecting() {
    // The point of the momentary tool: the gesture right after a marquee is a brush
    // stroke, gated by the selection just drawn — not a second marquee replacing it.
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.process(GestureCommand::Start {
        tool: Tool::SelectRect,
        sample: InputSample::at(BOX_MIN),
        tolerance: DEFAULT_TOLERANCE,
    });
    engine.process(GestureCommand::To {
        sample: InputSample::at(BOX_MAX),
    });
    engine.process(GestureCommand::End);

    // Drive the next gesture with whatever tool the engine now reports, exactly as a
    // frontend does — no explicit switch back to the brush anywhere.
    engine.process(ViewCommand::SetBrush(brush(RED, 14.0)));
    let tool = engine.observe().tool;
    engine.process(GestureCommand::Start {
        tool,
        sample: InputSample::at(Vec2::new(-30.0, 0.0)),
        tolerance: DEFAULT_TOLERANCE,
    });
    engine.process(GestureCommand::To {
        sample: InputSample::at(Vec2::new(30.0, 0.0)),
    });
    engine.process(GestureCommand::End);

    let img = engine.render_to_image();
    assert!(
        engine.observe().has_selection,
        "the selection is still in force"
    );
    assert!(
        is_painted(&img, Vec2::new(-20.0, 0.0)),
        "the gesture painted"
    );
    assert!(
        !is_painted(&img, Vec2::new(20.0, 0.0)),
        "and was gated by the selection it followed"
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
    let bare = engine.render_to_image();
    select(&mut engine, SelectionMode::Replace, rect(BOX_MIN, BOX_MAX));
    let outlined = engine.render_to_image();

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
    assert!(
        !changed(Vec2::new(-20.0, 0.0)),
        "the interior is left alone"
    );
    assert!(!changed(Vec2::new(20.0, 0.0)), "the exterior is left alone");
}

/// **A selection has a strength, and every tool acts through it** (§6.8).
///
/// The point of putting the opacity on the *mask* rather than on any one tool: the
/// mask has always meant "how much each texel is selected", so a selection at a
/// half is a half-strength brush inside it, with nothing in the stroke path
/// changed. Measured as redness against the same stroke at full strength and
/// against no stroke at all, so the claim is "between", not a pinned number.
#[test]
fn a_partial_selection_dims_every_tool_through_it() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let probe = Vec2::new(-20.0, 0.0);
    let redness = |img: &RgbaImage| {
        let (x, y) = screen_of(probe);
        let i = ((y * img.width + x) * 4) as usize;
        img.pixels[i] as i32 - (img.pixels[i + 1] as i32 + img.pixels[i + 2] as i32) / 2
    };

    let bare = redness(&engine.render_to_image());
    select(&mut engine, SelectionMode::Replace, rect(BOX_MIN, BOX_MAX));
    crossing_stroke(&mut engine);
    let full = redness(&engine.render_to_image());
    assert!(full > bare + 40, "the control stroke did not land");

    // The same selection at a half, and the same stroke through it.
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.process(DocCommand::Select(SelectionOp::at(
        SelectionMode::Replace,
        rect(BOX_MIN, BOX_MAX),
        0.0,
        0.5,
    )));
    crossing_stroke(&mut engine);
    let half = redness(&engine.render_to_image());
    assert!(
        half > bare + 10 && half < full - 10,
        "a half-strength selection should lay paint between none and all \
         (bare {bare}, half {half}, full {full})"
    );
}

/// The marching ants have to survive the mask being dimmed. They are recovered by
/// differencing the mask, so a selection at 0.4 has no 0.5-contour in it anywhere
/// and a fixed threshold would draw nothing at all — the outline reads the
/// selection's own level instead (§6.8).
#[test]
fn a_partial_selection_is_still_outlined() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let bare = engine.render_to_image();
    engine.process(DocCommand::Select(SelectionOp::at(
        SelectionMode::Replace,
        rect(BOX_MIN, BOX_MAX),
        0.0,
        0.4,
    )));
    let outlined = engine.render_to_image();

    let changed = |p: Vec2| {
        let (x, y) = screen_of(p);
        let i = ((y * bare.width + x) * 4) as usize;
        (0..3)
            .map(|k| (bare.pixels[i + k] as i32 - outlined.pixels[i + k] as i32).abs())
            .max()
            .unwrap_or(0)
            > 8
    };
    assert!(
        changed(Vec2::new(0.0, 0.0)),
        "a selection at 0.4 drew no outline"
    );
    assert!(
        !changed(Vec2::new(-20.0, 0.0)),
        "the interior is left alone"
    );
}

/// Inverting reflects through the mask's level, so the complement of a region
/// selected at a half is its *outside* selected at a half — not at full strength,
/// which `1 − m` would have given, and not nothing (§6.8).
#[test]
fn inverting_a_partial_selection_keeps_its_strength() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let probe = Vec2::new(20.0, 0.0); // outside the box, so inversion selects it
    let redness = |img: &RgbaImage| {
        let (x, y) = screen_of(probe);
        let i = ((y * img.width + x) * 4) as usize;
        img.pixels[i] as i32 - (img.pixels[i + 1] as i32 + img.pixels[i + 2] as i32) / 2
    };
    let bare = redness(&engine.render_to_image());

    engine.process(DocCommand::Select(SelectionOp::at(
        SelectionMode::Replace,
        rect(BOX_MIN, BOX_MAX),
        0.0,
        0.5,
    )));
    engine.process(DocCommand::InvertSelection);
    crossing_stroke(&mut engine);
    let inverted = redness(&engine.render_to_image());

    // Full strength through the same inverted geometry, for the upper bound.
    let Some(mut reference) = engine_or_skip() else {
        return;
    };
    select(
        &mut reference,
        SelectionMode::Replace,
        rect(BOX_MIN, BOX_MAX),
    );
    reference.process(DocCommand::InvertSelection);
    crossing_stroke(&mut reference);
    let full = redness(&reference.render_to_image());

    assert!(
        inverted > bare + 10,
        "inverting a partial selection deselected everything \
         (bare {bare}, inverted {inverted})"
    );
    assert!(
        inverted < full - 10,
        "inverting a partial selection promoted it to full strength \
         (inverted {inverted}, full {full})"
    );
}

#[test]
fn feathered_edge_fades_the_stroke() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // Same geometry, hard vs. heavily feathered edge. Just inside the boundary the
    // feathered mask must be visibly weaker than the hard one.
    let probe = Vec2::new(-4.0, 0.0);
    engine.process(DocCommand::Select(SelectionOp::new(
        SelectionMode::Replace,
        rect(BOX_MIN, BOX_MAX),
        0.0,
    )));
    crossing_stroke(&mut engine);
    let hard = engine.render_to_image();

    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.process(DocCommand::Select(SelectionOp::new(
        SelectionMode::Replace,
        rect(BOX_MIN, BOX_MAX),
        20.0,
    )));
    crossing_stroke(&mut engine);
    let soft = engine.render_to_image();

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

// --- goldens (§9) --------------------------------------------------
//
// Point probes can say "paint here, none there"; they cannot say what the mask
// *looks like*. These two do: the analytic silhouettes and how the three boolean
// modes compose into one, the width and profile of the feather ramp, that the mask is
// continuous where it crosses a tile boundary (it is rasterized per tile, so a
// discontinuity there is a real failure mode — §6.4), and the outline overlay, which
// nothing else renders. One golden per masking site, since they are separate code.

/// A stripe field covering the whole viewport, in alternating colors — the paint the
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
    engine.process(DocCommand::Select(op(
        SelectionMode::Replace,
        rect(Vec2::new(-100.0, -74.0), Vec2::new(6.0, 74.0)),
    )));
    engine.process(DocCommand::Select(op(
        SelectionMode::Union,
        SelectionShape::Ellipse {
            center: Vec2::new(46.0, -4.0),
            radii: Vec2::new(58.0, 62.0),
        },
    )));
    engine.process(DocCommand::Select(op(
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

    let img = engine.render_to_image();
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
    engine.process(DocCommand::Select(SelectionOp::new(
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

    let img = engine.render_to_image();
    assert_golden("selection_smear", &img, 6);
}
