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
use stark_engine::RgbaImage;
use stark_engine::command::Tool;
use stark_engine::command::{DocCommand, GestureCommand, InputSample, ViewCommand};
use stark_engine::path::DEFAULT_TOLERANCE;
use stark_model::Srgb;
use stark_model::document::{
    BrushDynamics, BrushEffect, BrushParams, FillOp, SelectionMode, SelectionOp, SelectionShape,
    ShapeAction,
};
use stark_model::geom::Vec2;

/// A fill's paint is color alone — how far it covers is `FillOp::opacity`,
/// which the Select panel's slider sets (§18.0.4). These double as brush
/// colors unchanged: a color is three channels for both (§6.2).
const RED: [f32; 3] = [1.0, 0.0, 0.0];
const GREEN: [f32; 3] = [0.1, 0.8, 0.2];

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
/// Whether two texels show the same paint. "The same" is within one code per
/// channel: the display encode dithers each pixel's rounding independently
/// (§6.5), so two screen pixels of identical paint may legitimately land one
/// code apart.
fn same_paint(a: [i32; 3], b: [i32; 3]) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1)
}

fn is_red(img: &RgbaImage, canvas: Vec2) -> bool {
    let [r, g, b] = texel(img, canvas);
    r - g > 40 && r - b > 40
}

fn is_green(img: &RgbaImage, canvas: Vec2) -> bool {
    let [r, g, b] = texel(img, canvas);
    g - r > 40 && g - b > 40
}

fn select_rect(engine: &mut stark_engine::Engine, min: Vec2, max: Vec2, feather: f32) {
    engine.process(DocCommand::Select(SelectionOp::new(
        SelectionMode::Replace,
        SelectionShape::rect_from_corners(min, max),
        feather,
    )));
}

/// Commit a fill of the rect `min..max` with `color`, through the direct command
/// path. `opacity` is how far it covers, which is what the panel's slider sets.
fn fill_rect(
    engine: &mut stark_engine::Engine,
    min: Vec2,
    max: Vec2,
    color: [f32; 3],
    feather: f32,
    opacity: f32,
) {
    let layer = engine.observe().active_layer;
    engine.process(DocCommand::Fill {
        layer,
        op: FillOp::new(
            SelectionShape::rect_from_corners(min, max),
            feather,
            Srgb::new(color),
            opacity,
        ),
    });
}

/// Arm the Fill action with a red brush. A fill lays *the brush's color*, so
/// that much still comes off the Color panel; how far it covers is the Select
/// panel's own opacity, which defaults to a full one.
fn arm_fill(engine: &mut stark_engine::Engine, color: [f32; 3]) {
    engine.process(ViewCommand::set_brush(brush(color, 16.0)));
    engine.process(ViewCommand::SetShapeAction(ShapeAction::Fill));
}

/// Drag a rectangle out with the Fill action armed — the gesture path, which is
/// what a user actually does.
fn drag_fill(engine: &mut stark_engine::Engine, min: Vec2, max: Vec2) {
    arm_fill(engine, RED);
    engine.process(GestureCommand::Start {
        tool: Tool::SelectRect,
        sample: InputSample::at(min),
        tolerance: DEFAULT_TOLERANCE,
        rope: 0.0,
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
        1.0,
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
        1.0,
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
        op: FillOp::of_selection(Srgb::new(RED)),
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
        op: FillOp::of_selection(Srgb::new(RED)),
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

    // A wash at a twentieth: a glaze, which must tint rather than cover. The
    // parcel law is the brush's own (`paint_common.wesl`), so this is the same
    // arithmetic a very slow brush over the same area would run — the opacity
    // only says how much of it to run.
    fill_rect(
        &mut engine,
        Vec2::new(-60.0, -60.0),
        Vec2::new(60.0, 60.0),
        RED,
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
        1.0,
    );
    assert!(
        is_red(&engine.render_to_image(), Vec2::ZERO),
        "a thick fill should cover what is under it"
    );
}

/// **The claim the Opacity slider exists to make** (§6.8): at 1 a fill *covers*,
/// and at ½ it does not. No setting of the brush's flow could say the first —
/// visible coverage is `1 − exp(−K·mass)`, so the whole flow range at full alpha
/// stopped at 95% and the slider had no top. The shader now inverts that law for
/// the mass instead of taking a thickness on faith.
///
/// Read as a comparison inside one frame rather than against an absolute color:
/// the same fill lands over a green stroke on the left and over bare paper on the
/// right, and "covers" means those two places agree. The residual tolerance is the
/// lighting, not the paint — the stroke under the left one still has height, so
/// the media pass shades the coat over it slightly differently (§6.3).
#[test]
fn opacity_one_covers_what_is_under_it_and_a_half_does_not() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let (over_paint, over_paper) = (Vec2::new(-50.0, 0.0), Vec2::new(50.0, 0.0));
    let spread = |img: &RgbaImage| {
        let (a, b) = (texel(img, over_paint), texel(img, over_paper));
        (0..3).map(|c| (a[c] - b[c]).abs()).max().unwrap_or(0)
    };

    paint(
        &mut engine,
        GREEN,
        20.0,
        &[Vec2::new(-70.0, 0.0), Vec2::new(-30.0, 0.0)],
    );
    let rect = (Vec2::new(-80.0, -40.0), Vec2::new(80.0, 40.0));

    fill_rect(&mut engine, rect.0, rect.1, RED, 0.0, 1.0);
    let covered = spread(&engine.render_to_image());
    assert!(
        covered <= 12,
        "an opaque fill let the paint under it show through (channels differ by {covered})"
    );

    // The same stroke, filled again at half — undo puts the canvas back, so the
    // two fills are compared over identical paint.
    engine.process(DocCommand::Undo);
    fill_rect(&mut engine, rect.0, rect.1, RED, 0.0, 0.5);
    let showing = spread(&engine.render_to_image());
    assert!(
        showing > 3 * covered.max(4),
        "a half-opacity fill hid the paint under it almost as well as an opaque one \
         ({showing} vs {covered}) — the slider's middle is doing nothing"
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
        1.0,
    );
    let img = engine.render_to_image();
    // Coverage scales the *paint* — a feathered edge is a thinning of the deposit,
    // not a fade of its color — and, because the shader asks the slab law for the
    // mass that lands the coverage it was given, what thins on the canvas is the
    // rasterizer's ramp itself. Sampled well inside, near the boundary, and past
    // the ramp.
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
        rope: 0.0,
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
        rope: 0.0,
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
        rope: 0.0,
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
        1.0,
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
        0.6,
    );
    let filled = engine.render_to_image();
    assert!(is_red(&filled, Vec2::ZERO));

    // The whole argument for a fill depositing *paint* rather than a flat color:
    // what it lays has an amount, so the wet-paint loop can pick it up again. A
    // pure scraper (`add = 0`, `lift = 1`) should take it off. Filled at 0.6 rather
    // than opaque because the claim is that the paint is liftable, not that one
    // pass of a scraper out-runs eight units of it.
    let scraper = BrushParams {
        size: 24.0,
        // A pure scraper mints nothing, so its pigment is never seen.
        effect: BrushEffect::paint_with(
            [0.0; 3],
            BrushDynamics {
                flow: 0.0,
                lift: 1.0,
                deposit: 0.0,
                charge: 0.0,
                bleed: 0.0,
            },
        ),
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
            Srgb::new([0.9, 0.2, 0.1]),
            0.7,
        ),
    });
    assert_golden("fill_ellipse", &engine.render_to_image(), 2);
}

// ---------------------------------------------------------------------------
// The gradient fill (§22.4): a parcel whose latent varies with canvas position.
// ---------------------------------------------------------------------------

use stark_engine::PickOptions;
use stark_model::document::{GradientAxis, GradientParcel, Parcel};
use stark_model::{Gradient, GradientStop};

const BLUE: [f32; 3] = [0.1, 0.2, 0.8];

fn red_blue() -> Gradient {
    Gradient::new(vec![
        GradientStop {
            t: 0.0,
            color: Srgb::new(RED),
        },
        GradientStop {
            t: 1.0,
            color: Srgb::new(BLUE),
        },
    ])
    .expect("two stops")
}

fn is_blue(img: &RgbaImage, canvas: Vec2) -> bool {
    let [r, g, b] = texel(img, canvas);
    b - r > 40 && b - g > 40
}

/// Commit a gradient fill of the rect `min..max`, through the direct command path.
fn gradient_fill_rect(
    engine: &mut stark_engine::Engine,
    min: Vec2,
    max: Vec2,
    parcel: GradientParcel,
    feather: f32,
    opacity: f32,
) {
    let layer = engine.observe().active_layer;
    engine.process(DocCommand::Fill {
        layer,
        op: FillOp::with_paint(
            SelectionShape::rect_from_corners(min, max),
            feather,
            Parcel::Gradient(parcel),
            opacity,
        ),
    });
}

/// The ramp lies along its axis, and beyond either end it holds its end stop —
/// a gradient fill covers its whole region; the axis only places the transition.
#[test]
fn a_linear_gradient_fill_ramps_along_its_axis_and_clamps_past_it() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // The region is wider than the axis, so the clamp is observable inside it.
    gradient_fill_rect(
        &mut engine,
        Vec2::new(-90.0, -30.0),
        Vec2::new(90.0, 30.0),
        GradientParcel {
            gradient: red_blue(),
            axis: GradientAxis::Linear {
                from: Vec2::new(-50.0, 0.0),
                to: Vec2::new(50.0, 0.0),
            },
        },
        0.0,
        0.7,
    );
    let img = engine.render_to_image();
    assert!(
        is_red(&img, Vec2::new(-50.0, 0.0)),
        "the axis starts on red"
    );
    assert!(is_blue(&img, Vec2::new(50.0, 0.0)), "and ends on blue");
    assert!(
        is_red(&img, Vec2::new(-80.0, 0.0)),
        "before the axis the ramp holds its first stop"
    );
    assert!(
        is_blue(&img, Vec2::new(80.0, 0.0)),
        "past the axis it holds its last"
    );
    // Constant on perpendiculars: the ramp is a function of the axis alone.
    let (above, below) = (
        texel(&img, Vec2::new(-50.0, -20.0)),
        texel(&img, Vec2::new(-50.0, 20.0)),
    );
    assert!(
        same_paint(above, below),
        "perpendicular offsets read the same ramp position: {above:?} vs {below:?}"
    );
    // And the middle is a mixture, not either end.
    let [r, _, b] = texel(&img, Vec2::ZERO);
    assert!(
        r > 40 && b > 40,
        "the midpoint should mix the two ends, got {:?}",
        texel(&img, Vec2::ZERO)
    );
}

/// A radial axis rings outward: first stop at the centre, last at and past the
/// radius, and the same color at the same distance in every direction.
#[test]
fn a_radial_gradient_fill_rings_outward() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    gradient_fill_rect(
        &mut engine,
        Vec2::new(-80.0, -80.0),
        Vec2::new(80.0, 80.0),
        GradientParcel {
            gradient: red_blue(),
            axis: GradientAxis::Radial {
                center: Vec2::ZERO,
                radius: 50.0,
            },
        },
        0.0,
        0.7,
    );
    let img = engine.render_to_image();
    assert!(is_red(&img, Vec2::ZERO), "the centre is the first stop");
    assert!(
        is_blue(&img, Vec2::new(70.0, 0.0)),
        "past the radius the ramp holds its last stop"
    );
    let (east, south) = (
        texel(&img, Vec2::new(30.0, 0.0)),
        texel(&img, Vec2::new(0.0, 30.0)),
    );
    assert!(
        same_paint(east, south),
        "the ramp is a function of distance alone: {east:?} vs {south:?}"
    );
}

/// The preview is the commit, before the commit — `PreviewTransform`'s bargain,
/// made by the gradient-fill mode (§22.4): `ViewCommand::PreviewFill` runs the
/// same `FillRenderer::apply` over the same committed tiles as
/// `DocCommand::Fill`, so the two frames must be identical.
#[test]
fn the_gradient_preview_matches_the_commit() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint(
        &mut engine,
        GREEN,
        20.0,
        &[Vec2::new(-60.0, -20.0), Vec2::new(60.0, 20.0)],
    );
    select_rect(
        &mut engine,
        Vec2::new(-55.0, -35.0),
        Vec2::new(55.0, 35.0),
        8.0,
    );
    let layer = engine.observe().active_layer;
    let op = FillOp::gradient_of_selection(GradientParcel {
        gradient: red_blue(),
        axis: GradientAxis::Linear {
            from: Vec2::new(-55.0, 0.0),
            to: Vec2::new(55.0, 0.0),
        },
    });

    engine.process(ViewCommand::PreviewFill(Some((layer, op.clone()))));
    let previewed = engine.render_to_image();
    engine.process(ViewCommand::PreviewFill(None));
    engine.process(DocCommand::Fill { layer, op });
    let committed = engine.render_to_image();
    assert!(
        images_match(&previewed, &committed, 0),
        "gradient-fill preview vs commit must be bit-identical"
    );
}

/// A redrawn axis previews a fresh fill, never a stack: fifty previews render
/// the same frame as one — the same claim the drag-fill preview makes.
#[test]
fn the_gradient_preview_does_not_accumulate() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    select_rect(
        &mut engine,
        Vec2::new(-40.0, -40.0),
        Vec2::new(40.0, 40.0),
        0.0,
    );
    let layer = engine.observe().active_layer;
    let op = |to: f32| {
        FillOp::gradient_of_selection(GradientParcel {
            gradient: red_blue(),
            axis: GradientAxis::Linear {
                from: Vec2::new(-40.0, 0.0),
                to: Vec2::new(to, 0.0),
            },
        })
    };
    engine.process(ViewCommand::PreviewFill(Some((layer, op(40.0)))));
    let first = engine.render_to_image();
    for to in 0..50 {
        engine.process(ViewCommand::PreviewFill(Some((layer, op(to as f32)))));
        let _ = engine.render_to_image();
    }
    engine.process(ViewCommand::PreviewFill(Some((layer, op(40.0)))));
    let last = engine.render_to_image();
    assert!(
        images_match(&first, &last, 0),
        "re-previewing the same axis must render the same frame"
    );
}

/// The ramp's ends are its stops, in **both** color spaces — in Mixbox the
/// stops convert to concentrations *and residual*, and an end that forgot the
/// residual would come back as the polynomial's nearest reachable color
/// (§6.7). Read back through the eyedropper, which answers in sRGB.
#[test]
fn the_ramp_ends_are_the_stops_in_both_spaces() {
    for space in stark_engine::colorspace::all_available() {
        let Some(mut engine) = engine_or_skip_with(space) else {
            return;
        };
        gradient_fill_rect(
            &mut engine,
            Vec2::new(-60.0, -30.0),
            Vec2::new(60.0, 30.0),
            GradientParcel {
                gradient: red_blue(),
                axis: GradientAxis::Linear {
                    from: Vec2::new(-40.0, 0.0),
                    to: Vec2::new(40.0, 0.0),
                },
            },
            0.0,
            0.7,
        );
        for (at, want, name) in [
            (Vec2::new(-40.0, 0.0), RED, "start"),
            (Vec2::new(40.0, 0.0), BLUE, "end"),
        ] {
            let got = pollster::block_on(engine.pick_color(at, PickOptions::default()))
                .unwrap_or_else(|| panic!("{space:?}: no paint at the {name}"));
            for ch in 0..3 {
                assert!(
                    (got[ch] - want[ch]).abs() < 0.04,
                    "{space:?}: the {name} should be its stop, got {got:?}"
                );
            }
        }
    }
}

/// A gradient fill replays identically from the log — with every field of the
/// parcel distinct (three stops, a radial axis, partial opacity, a feather), so
/// a mis-ordered decode shows up as a different picture rather than a lucky
/// match (the same stance as the filter's round-trip test).
#[test]
fn a_gradient_fill_survives_save_and_load() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let gradient = Gradient::new(vec![
        GradientStop {
            t: 0.0,
            color: Srgb::new([0.9, 0.1, 0.1]),
        },
        GradientStop {
            t: 0.3,
            color: Srgb::new([0.9, 0.8, 0.1]),
        },
        GradientStop {
            t: 1.0,
            color: Srgb::new([0.1, 0.2, 0.8]),
        },
    ])
    .expect("three stops");
    select_rect(
        &mut engine,
        Vec2::new(-50.0, -50.0),
        Vec2::new(50.0, 50.0),
        6.0,
    );
    let layer = engine.observe().active_layer;
    engine.process(DocCommand::Fill {
        layer,
        op: FillOp::gradient_of_selection(GradientParcel {
            gradient,
            axis: GradientAxis::Radial {
                center: Vec2::new(5.0, -3.0),
                radius: 47.0,
            },
        }),
    });
    let before = engine.render_to_image();

    let bytes = engine.save_bytes().expect("save");
    engine.load_bytes(&bytes).expect("load");
    let after = engine.render_to_image();
    assert!(
        images_match(&before, &after, 2),
        "a gradient fill did not replay identically from the log"
    );
}
