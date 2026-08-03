//! Brush-dynamics stamp-loop tests (§6.2): the sequential pickup ⇄
//! deposit exchange against a 2-D tool reservoir. Covers conservation (paint
//! moves by height, is neither created nor destroyed), smearing onto bare
//! canvas, erasing without re-tinting, the pre-`charge` glob, and determinism.

mod common;

use common::*;
use stark_core::command::{DocCommand, GestureCommand, InputSample, ViewCommand};
use stark_core::document::{BrushDynamics, BrushParams, BrushShape, Tool};
use stark_core::geom::Vec2;
use stark_core::path::DEFAULT_TOLERANCE;

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
        !is_red(center(&engine.render_to_image())),
        "lit bare paper reads as red — is_red can't discriminate paint here"
    );
    paint(
        &mut engine,
        RED,
        80.0,
        &[Vec2::new(-110.0, 0.0), Vec2::new(110.0, 0.0)],
    );
    let before = engine.render_to_image();

    let b = dyn_brush(
        RED,
        24.0,
        BrushDynamics {
            add: 0.0,
            lift: 0.5,
            deposit: 0.5,
            ..Default::default()
        },
    );
    stroke_with(
        &mut engine,
        b,
        &[Vec2::new(-50.0, 0.0), Vec2::new(50.0, 0.0)],
    );
    let after = engine.render_to_image();

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
    let run_before = engine.render_to_image().pixel(run_x, y);
    assert!(
        run_before[1] > 170,
        "the runway should start bare paper: {run_before:?}"
    );

    let b = dyn_brush(
        RED,
        28.0,
        BrushDynamics {
            add: 0.0,
            lift: 0.9,
            deposit: 0.3,
            ..Default::default()
        },
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
    let run_after = engine.render_to_image().pixel(run_x, y);

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
    let before = engine.render_to_image().pixel(green_x, y);
    assert!(is_green(before), "the bar should start green: {before:?}");

    // Drag from inside the red patch across the green bar: the tool holds red
    // while it crosses green.
    let b = dyn_brush(
        RED,
        24.0,
        BrushDynamics {
            add: 0.0,
            lift: 0.5,
            deposit: 0.0,
            ..Default::default()
        },
    );
    stroke_with(
        &mut engine,
        b,
        &[Vec2::new(-80.0, 0.0), Vec2::new(90.0, 0.0)],
    );
    let after = engine.render_to_image().pixel(green_x, y);

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
    let blank = engine.render_to_image();
    let b = dyn_brush(
        RED,
        24.0,
        BrushDynamics {
            add: 0.0,
            lift: 1.0,
            deposit: 1.0,
            ..Default::default()
        },
    );
    stroke_with(
        &mut engine,
        b,
        &[Vec2::new(-40.0, 0.0), Vec2::new(40.0, 0.0)],
    );
    let after = engine.render_to_image();
    assert!(
        images_match(&blank, &after, 2),
        "a pure smear over empty canvas must add no paint or height"
    );
}

/// The pen-up settle (§6.2), from the canvas → tool side: **an eraser must take the
/// paint it stopped on.**
///
/// A stroke ends with the tip still in contact and the transfer still in flight, and
/// on the last footprint alone nothing finishes it — everywhere else on the trail a
/// point sees the whole footprint pass over it and leave by the trailing rim. Without
/// a settle the eraser's own last footprint keeps a tip-shaped patch of the paint it
/// was standing on: measured here, 18 levels of red still there at the pen-up point
/// against a trail that is scraped to bare paper. The settle takes it, and the mark
/// then ends where the pen did, tapering only across the half the tip had not reached.
#[test]
fn an_eraser_takes_the_paint_it_stopped_on() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // A bed far wider than the tip, so the whole stroke runs inside it.
    paint(
        &mut engine,
        RED,
        140.0,
        &[Vec2::new(-220.0, 0.0), Vec2::new(220.0, 0.0)],
    );
    stroke_with(
        &mut engine,
        dyn_brush(
            RED,
            45.0,
            BrushDynamics {
                add: 0.0,
                lift: 0.95,
                deposit: 0.0,
                ..Default::default()
            },
        ),
        &[Vec2::new(-150.0, 0.0), Vec2::new(0.0, 0.0)],
    );
    let img = engine.render_to_image();
    let y = SIZE.height / 2;
    // Red on the warm ground pulls green down, so paint left behind reads as green
    // that failed to come back up. Canvas 0 is the pen-up point.
    let trail = img.pixel(SIZE.width / 2 - 38, y)[1] as i32;
    let stop = img.pixel(SIZE.width / 2, y)[1] as i32;
    assert!(
        trail - stop < 8,
        "the eraser left {} levels of paint under its last footprint that the rest of \
         the trail was scraped clean of — the pen-up transfer is stranded on the tool",
        trail - stop
    );
}

/// The same settle from the tool → canvas side: **a carrying stroke must not break
/// its own slope where it stops.**
///
/// A tool laying its load onto bare canvas leaves a mark that fades as the load runs
/// down — gently, since one tip radius of travel depletes it barely at all. But the
/// last footprint is served only by the tip's *leading* half, which is the depleted
/// half, so without a settle the mark falls off a cliff at the trailing rim of that
/// footprint: measured here it lightens by 17 levels across the final footprint
/// against 2 over the whole trail before it, and the corner reads as a tip-shaped
/// disc. With the settle it is 4.
///
/// The bound is on the *deficit at the pen-up point* rather than on a slope ratio
/// because a gentle fade is only a few levels over the window either way, and a ratio
/// of two small differences is mostly quantization.
#[test]
fn a_carrying_stroke_ends_without_breaking_its_own_slope() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    stroke_with(
        &mut engine,
        BrushParams {
            // No `drain`: it fades the mark on its own and caps the segment length,
            // and both would muddy a measurement about how the mark *ends*. A hard tip
            // because that is where the corner is sharpest — `κ = −ln(1−coverage)`
            // knees wherever coverage approaches 1, and a soft tip spreads the break
            // over its own falloff.
            drain: 0.0,
            shape: BrushShape::Round { hardness: 0.95 },
            ..dyn_brush(
                RED,
                70.0,
                BrushDynamics {
                    add: 0.0,
                    lift: 0.0,
                    deposit: 0.12,
                    charge: 2.0,
                    bleed: 0.0,
                },
            )
        },
        &[Vec2::new(-260.0, 0.0), Vec2::new(0.0, 0.0)],
    );
    let img = engine.render_to_image();
    let y = SIZE.height / 2;
    // Just behind the final footprint (canvas −76) against the pen-up point itself.
    let trail = img.pixel(SIZE.width / 2 - 76, y)[1] as i32;
    let stop = img.pixel(SIZE.width / 2, y)[1] as i32;
    assert!(
        stop - trail < 10,
        "the mark lightened by {} levels across the final footprint against {} of fade \
         over the whole trail before it — the tool's load is stranded at the pen-up",
        stop - trail,
        trail - img.pixel(SIZE.width / 2 - 108, y)[1] as i32
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
        BrushDynamics {
            add: 0.0,
            deposit: 0.12,
            charge: 2.0,
            ..Default::default()
        },
    );
    stroke_with(
        &mut engine,
        b,
        &[Vec2::new(-110.0, 0.0), Vec2::new(110.0, 0.0)],
    );
    let img = engine.render_to_image();

    // Bare paper, read off this very image instead of hardcoded. The old bound was an
    // absolute "reads ~200", which recorded what the tonemap happened to do the day it
    // was written and had to be re-derived every time the lighting model moved — most
    // recently when the media pass became a reference (§6.3) and bare paper
    // brightened by forty levels. What this test is *about* is the difference: red
    // paint pulls the paper's green down where the glob landed, and pulls it down less
    // once the charge has run out. That survives any exposure or tonemap.
    let paper = img.pixel(20, 20); // well above the stroke, never touched
    let green_loss = |px: [u8; 4]| paper[1] as i32 - px[1] as i32;

    let start = img.pixel(20, y); // near the stroke start
    assert!(
        green_loss(start) > 15,
        "the charged glob must lay paint near the stroke start: {start:?} on paper {paper:?}"
    );
    // The glob depletes: the very end of the stroke carries visibly less paint
    // than the start.
    let end = img.pixel(SIZE.width - 20, y);
    assert!(
        green_loss(end) < green_loss(start),
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
            BrushDynamics {
                add: 0.1,
                lift: 0.5,
                deposit: 0.5,
                ..Default::default()
            },
        );
        stroke_with(
            &mut engine,
            b,
            &[Vec2::new(-110.0, 0.0), Vec2::new(110.0, 0.0)],
        );
        Some(engine.render_to_image())
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
    // (§6.2). Guards the look of the sequential stamp loop end-to-end.
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
        BrushDynamics {
            add: 0.1,
            lift: 0.5,
            deposit: 0.5,
            ..Default::default()
        },
    );
    stroke_with(
        &mut engine,
        b,
        &[Vec2::new(-110.0, 0.0), Vec2::new(110.0, 0.0)],
    );
    let img = engine.render_to_image();
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
        BrushDynamics {
            add: 0.5,
            lift: 0.6,
            deposit: 0.5,
            ..Default::default()
        },
    );
    // Down through the bar, loop around, and back across its own trail.
    engine.process(ViewCommand::SetBrush(b));
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(Vec2::new(-40.0, -70.0)),
        tolerance: DEFAULT_TOLERANCE,
    });
    for &(x, y) in &[(-40.0, 60.0), (20.0, 70.0), (40.0, 0.0), (-70.0, 10.0)] {
        engine.process(GestureCommand::To {
            sample: InputSample::at(Vec2::new(x, y)),
        });
    }
    engine.process(GestureCommand::End);
    let img = engine.render_to_image();
    assert_golden("self_smear", &img, 6);
}

/// A dynamics stroke must read as one continuous mark, not a row of stamps.
///
/// The loop's transfers compose only if each is a function of the segment's swept
/// exposure that survives re-cutting the path — additive in `e`, or multiplicative
/// `exp(k·e)`. Anything saturating per segment (`1 − exp(k·e)` applied to a
/// reservoir the segment does not deplete) deposits a steep ramp inside each quad,
/// which shows up as arcs at the flattening's segment spacing. Uniform 2px
/// sampling hid it; adaptive sampling does not.
#[test]
fn dynamics_stroke_reads_as_one_continuous_mark() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // A long straight stroke with a big tip: `drain` fades it along its length, so
    // the profile should be smooth and monotone. Banding shows as ripple against
    // that ramp.
    let b = BrushParams {
        drain: 0.004,
        ..dyn_brush(
            RED,
            60.0,
            BrushDynamics {
                add: 0.6,
                lift: 0.2,
                deposit: 0.9,
                ..Default::default()
            },
        )
    };
    stroke_with(
        &mut engine,
        b,
        &[Vec2::new(-120.0, 0.0), Vec2::new(120.0, 0.0)],
    );
    let img = engine.render_to_image();

    // Walk the centre row across the stroke's body and track the red channel. A
    // continuous stroke fades monotonically; each reversal is a band edge.
    let y = img.height / 2;
    let prof: Vec<i32> = (40..216).map(|x| img.pixel(x, y)[0] as i32).collect();
    let mut reversals = 0;
    for w in prof.windows(3) {
        let (d0, d1) = (w[1] - w[0], w[2] - w[1]);
        // Only count reversals big enough to be a band, not sampling noise.
        if d0 * d1 < 0 && (d0.abs() > 1 || d1.abs() > 1) {
            reversals += 1;
        }
    }
    assert!(
        reversals <= 6,
        "stroke profile reverses {reversals}× across its body — per-segment banding",
    );
}

/// A stroke that is *carrying* paint must still read as one continuous mark.
///
/// The stamp loop's exchange has to survive being re-cut into different segments,
/// which adaptive flattening makes it do (uniform 2px sampling hid it). Two things
/// break that and both show up here as scalloped arcs at the segment spacing: a
/// deposit that is not the exchange ODE's own solution (`dynamics.wesl`'s
/// `laid_window`), and segments long enough that the tool reservoir's single
/// mid-pass sample jumps between them (`gpu::stroke::flatten_tolerance`). Only a
/// loaded tool exposes either — the `add` axis alone is smooth, which is why this
/// test smears existing paint rather than painting on bare canvas.
#[test]
fn a_carrying_stroke_reads_as_one_continuous_mark() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // A slab to load the tool from, then a smear that drags it out across the canvas.
    let mut slab = brush(GREEN, 70.0);
    slab.drain = 0.0;
    stroke_with(
        &mut engine,
        slab,
        &[Vec2::new(-115.0, 0.0), Vec2::new(-60.0, 0.0)],
    );

    let smear = BrushParams {
        shape: BrushShape::Round { hardness: 0.95 },
        drain: 0.0,
        ..dyn_brush(
            RED,
            60.0,
            BrushDynamics {
                add: 0.0,
                lift: 0.4,
                deposit: 0.9,
                ..Default::default()
            },
        )
    };
    stroke_with(
        &mut engine,
        smear,
        &[Vec2::new(-100.0, 0.0), Vec2::new(115.0, 0.0)],
    );
    let img = engine.render_to_image();

    // Green rises monotonically as the carried paint thins out; each reversal past
    // the noise floor is a band edge.
    let y = img.height / 2;
    let prof: Vec<i32> = (100..230).map(|x| img.pixel(x, y)[1] as i32).collect();
    let d: Vec<i32> = prof.windows(2).map(|w| w[1] - w[0]).collect();
    let reversals = d
        .windows(2)
        .filter(|w| w[0] * w[1] < 0 && (w[0].abs() > 1 || w[1].abs() > 1))
        .count();
    assert!(
        reversals <= 4,
        "carried-paint profile reverses {reversals}× — per-segment scalloping",
    );
}

/// A conservative smear (`add = 0`) can move paint around for as long as it likes and
/// must never end up with more of it than it started with.
///
/// The single-pass test above cannot see this. The failure it guards against is a
/// *drift* in the tool's own accounting, and one pass of a drift is nothing — what
/// makes it visible is the tool's load compounding, which needs one **long** gesture
/// rather than many short ones (a fresh gesture starts the tool empty, so repeating
/// short strokes resets the very thing that accumulates).
///
/// The bug it exists for: the tool's drain used to carry a fudge factor of 0.75
/// ("deplete a bit slower than the math would indicate to account for error"), so the
/// tool kept a quarter of every deposit it made *and laid it anyway*. That factor was
/// covering for the old reload cadence — the tool reloading on its own schedule while
/// the canvas was stripped every segment — and once both sides of the transfer measured
/// the same segment it had nothing left to cover for. Reinstating it fails this by ~6%.
///
/// Measured as image darkness rather than height, since there is no height readback:
/// the field is deliberately faint, because height only reaches the render while the
/// paint is still short of opaque.
#[test]
fn a_conservative_smear_does_not_mint_paint_however_long_it_runs() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // A broad field to smear inside, so nothing the tip carries leaves the region.
    let mut faint = brush([1.0, 0.0, 0.0, 0.12], 110.0);
    faint.drain = 0.0;
    stroke_with(
        &mut engine,
        faint,
        &[Vec2::new(-120.0, 0.0), Vec2::new(120.0, 0.0)],
    );

    let before = total_ink(&engine.render_to_image());

    // `add = 0`, so the tool can only move what is already there. Zig-zagging keeps
    // the tip inside the field for the whole gesture.
    let smear = dyn_brush(
        RED,
        30.0,
        BrushDynamics {
            add: 0.0,
            lift: 0.9,
            deposit: 0.9,
            ..Default::default()
        },
    );
    engine.process(ViewCommand::SetBrush(smear));
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(Vec2::new(-70.0, 0.0)),
        tolerance: DEFAULT_TOLERANCE,
    });
    for i in 1..=240 {
        let t = i as f32;
        engine.process(GestureCommand::To {
            sample: InputSample::at(Vec2::new((t * 0.19).sin() * 70.0, (t * 0.11).cos() * 40.0)),
        });
    }
    engine.process(GestureCommand::End);
    let after = total_ink(&engine.render_to_image());

    // Conserved, the tool ends the stroke still holding some of what it lifted, so
    // the canvas can only come out level or slightly lighter. Minting shows up as the
    // one thing conservation forbids: more paint than went in.
    let growth = after / before;
    assert!(
        growth <= 1.0,
        "a conservative smear left {:.1}% more ink on the canvas than it found          — the tool is minting paint",
        (growth - 1.0) * 100.0
    );
}

/// The other half of conservation: a smear must not **destroy** paint either.
///
/// The test above bounds the transfer from one side only, and a leak passes it
/// trivially — losing paint is never "more than went in". That is the side the
/// exchange actually got wrong: the tool used to solve half the coupled pair (the
/// canvas relaxing towards a tool that never took anything back) and then lift from
/// the region as the `deposit` had already left it, so the two halves disagreed about
/// how much had changed hands by `O(lift²)` per segment. At the rates below that is
/// ~39% of the total `canvas + tool` height at *every* segment boundary, and what it
/// draws is the pair of artifacts `golden_heavy_smear_regression` and
/// `golden_lift_end_regression` pin: arcs at the segment spacing through thick paint,
/// and a tip-shaped hole where a stroke stops.
///
/// Measured as the **worst** lightening along the smear rather than as total ink,
/// because that is what the eye is picking up and because a global average dilutes a
/// hole the size of a tip into nothing. Inside a uniform field a conservative smear
/// has nothing to move — the lift takes paint up and the deposit lays the same colour
/// straight back — so the only lightening the law permits is the load the tool is
/// still carrying, one tip's worth of a pass. `conservative_smear_preserves_uniform_field`
/// makes the same argument at `lift = deposit = 0.5`; the rates here are the ones that
/// made the discrepancy an `O(1)` fraction of the transfer instead of a rounding error.
///
/// The bound has room on both sides: the half-solved exchange lightens the field by 91
/// levels here and the complementary one by 35, and those 35 are the tool's own load —
/// what it picked up over its first pass and still holds when the pen comes up (§6.2).
#[test]
fn a_conservative_smear_does_not_destroy_paint_either() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // A field far wider than the tip, so nothing it carries can be left outside it.
    paint(
        &mut engine,
        RED,
        140.0,
        &[Vec2::new(-200.0, 0.0), Vec2::new(200.0, 0.0)],
    );
    let before = engine.render_to_image();

    stroke_with(
        &mut engine,
        dyn_brush(
            RED,
            40.0,
            BrushDynamics {
                add: 0.0,
                lift: 0.95,
                deposit: 0.95,
                ..Default::default()
            },
        ),
        &[Vec2::new(-100.0, 0.0), Vec2::new(0.0, 0.0)],
    );
    let after = engine.render_to_image();

    // Red paint on the warm ground pulls the green channel down, so paint going
    // missing shows as green coming back up.
    let y = SIZE.height / 2;
    let worst = (0..SIZE.width)
        .map(|x| after.pixel(x, y)[1] as i32 - before.pixel(x, y)[1] as i32)
        .max()
        .expect("a row of pixels");
    assert!(
        worst < 60,
        "a conservative smear lightened the field it was smearing by {worst} levels —          the two halves of the exchange are not taking complementary shares of it"
    );
}

/// The same visible stretch of stroke, painted with five different tails, must come
/// out the same however the flattener happened to cut it.
///
/// This is `golden_drained_brush_length_independent` asked as a *number* rather than as
/// five blessed images. The golden pins the artifact but cannot say how far off it is,
/// and the whole story on [`RESERVOIR_EXCHANGE_STEP`](stark_core) is a convergence
/// table — so the quantity it converges in belongs in the suite, not just in a comment.
///
/// The five strokes are collinear, so the *geometry* of the visible stretch is
/// identical in all five; the brush runs dry (`drain = 0.005`, so 200px) a whole tip
/// before entering view, so every visible pixel arrived through the reservoir. All that
/// differs is where the flattener's bisection put the segment boundaries, which depends
/// on the whole path's length. Any spread at all is the loop's first-order splitting
/// error printing as a delay line ringing at the segment cadence.
///
/// The bound is set above the 8-bit quantization floor the step was converged to and
/// below the ~6.6 levels the step before it produced, so it fails on a regression of
/// the transport accuracy without firing on rounding.
#[test]
fn a_carried_stroke_is_independent_of_how_the_path_was_cut() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let brush = BrushParams {
        radius: 80.0,
        shape: BrushShape::Round { hardness: 0.95 },
        drain: 0.005,
        dynamics: BrushDynamics {
            add: 1.0,
            lift: 0.95,
            deposit: 0.95,
            ..BrushDynamics::default()
        },
        ..BrushParams::default()
    };
    let renders: Vec<stark_core::RgbaImage> = [200.0f32, 300.0, 400.0, 500.0, 600.0]
        .into_iter()
        .map(|x| {
            stroke_with(
                &mut engine,
                brush,
                &[
                    Vec2::new(-400.0, 0.0),
                    Vec2::new(0.0, 0.0),
                    Vec2::new(x, 0.0),
                ],
            );
            let img = engine.render_to_image();
            engine.process(DocCommand::Undo);
            img
        })
        .collect();

    let (worst, rms) = cut_spread(&renders);
    eprintln!("length-independence: {worst} levels max, {rms:.2} rms");
    assert!(
        worst <= 4,
        "the same visible stretch of stroke came out {worst} levels apart ({rms:.2} rms)          depending on where the pen went afterwards — the exchange's transport error is          tracking the segment cadence"
    );
}

/// Worst and rms per-pixel spread across renders that are supposed to be identical.
///
/// Measured on green: the paint is red over a light grey ground, so green is the
/// channel it moves furthest. No high-pass is needed the way there is when judging one
/// image by eye — these five share a geometry, so the stroke's own gradient cancels in
/// the spread and what is left is only the disagreement.
fn cut_spread(imgs: &[stark_core::RgbaImage]) -> (u8, f64) {
    let (w, h) = (imgs[0].width, imgs[0].height);
    let mut worst = 0u8;
    let mut sq = 0.0f64;
    for y in 0..h {
        for x in 0..w {
            let (lo, hi) = imgs
                .iter()
                .map(|im| im.pixel(x, y)[1])
                .fold((255u8, 0u8), |(lo, hi), g| (lo.min(g), hi.max(g)));
            let d = hi - lo;
            worst = worst.max(d);
            sq += (d as f64) * (d as f64);
        }
    }
    (worst, (sq / (w * h) as f64).sqrt())
}

/// Mean darkness over the image: with a fixed palette this rises as paint thickens,
/// so it stands in for "how much paint is on the canvas" without a height readback.
fn total_ink(img: &stark_core::RgbaImage) -> f64 {
    let sum: u64 = img
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| (255 - p[1]) as u64 + (255 - p[2]) as u64)
        .sum();
    sum as f64 / (img.width * img.height) as f64
}

/// Flow for the glaze below. Low on purpose: `add` is the amount of paint the brush
/// lays and the only thing that decides it (§6.1), so at full flow a renderer
/// that honours it and one that ignores it both come out red and there is nothing
/// left to measure.
const GLAZE_ADD: f32 = 0.02;

/// Paint a solid vertical bar, then glaze a horizontal stroke across it at `add`
/// flow with the given `deposit`, and return the lit result.
fn bar_then_glaze(add: f32, deposit: f32) -> Option<stark_core::RgbaImage> {
    let mut engine = engine_or_skip()?;
    paint(
        &mut engine,
        GREEN,
        34.0,
        &[Vec2::new(0.0, -110.0), Vec2::new(0.0, 110.0)],
    );
    let mut glaze = dyn_brush(
        RED,
        34.0,
        BrushDynamics {
            add,
            lift: 0.0,
            deposit,
            charge: 0.0,
            bleed: 0.0,
        },
    );
    // No falloff, so the glaze is one uniform parcel over its whole travel and both
    // paths are being asked the same question at every texel of it.
    glaze.drain = 0.0;
    stroke_with(
        &mut engine,
        glaze,
        &[Vec2::new(-110.0, 0.0), Vec2::new(110.0, 0.0)],
    );
    Some(engine.render_to_image())
}

/// **The stamp loop reduces to the swept fast path when there is nothing to
/// manipulate**, and the same brush must therefore draw the same picture whichever
/// one runs (§6.2). A `deposit` of 0.01 with no `lift` and no `charge` puts the
/// stroke through the whole sequential loop while leaving the tool's reservoir empty
/// for all of it — nothing ever gets onto the tip, so `deposit` has nothing to lay
/// and every texel of the mark is the brush's own `add` paint, exactly as on the
/// fast path. `exchange_at` says so algebraically: with `k_lift = 0` the canvas keeps
/// all of its height (`keep = 1`) and the source rides the full exposure
/// (`add_w = e`), which is the fast path's deposit written out.
///
/// It is drawn *over existing paint*, which is where the two can differ without the
/// media pass covering for them. Over bare canvas visible alpha is `opacity × height`
/// and the height was always right, so a path that laid the colour by the brush's
/// per-unit opacity alone still came out faint; over a stroke already on the canvas
/// nothing checks it, and a 2%-flow glaze repainted the bar at full strength. That
/// was the swept path's `integrate`, covering by opacity rather than by the parcel's
/// visible alpha `1 − exp(−K·opacity·height)` — the same defect §6.3 names in
/// the layer composite, one level down.
///
/// The other half of the agreement is that `add` mean the same amount of paint on
/// both paths. It did not: the loop scaled it by a gain of 2, so nudging `deposit`
/// off zero doubled the flow of a slider that has nothing to do with it — 24% of
/// this image more than 4 levels apart on its own. Together the two put the worst
/// pixel 157 levels out; what is left is 2, which is the 8-bit floor.
#[test]
fn a_glaze_lands_the_same_whether_or_not_the_stamp_loop_runs() {
    let Some(swept) = bar_then_glaze(GLAZE_ADD, 0.0) else {
        return;
    };
    let Some(looped) = bar_then_glaze(GLAZE_ADD, 0.01) else {
        return;
    };
    let (_, worst) = diff_fraction(&swept, &looped);
    let frac = frac_exceeding(&swept, &looped, 4);
    assert!(
        frac < 0.002 && worst <= 8,
        "the same glaze drew differently through the stamp loop: {:.2}% of pixels \
         differ by >4 levels, worst {worst} — the loop is not reducing to the fast path",
        frac * 100.0
    );

    // …and the glaze must actually be a glaze, or the two paths above agree only
    // because they are both ignoring `add`. Against the bar alone and against the
    // same stroke at full flow, at the crossing: a fiftieth of the flow has to move
    // the colour a great deal less than all of it.
    let (Some(bar), Some(opaque)) = (bar_then_glaze(0.0, 0.0), bar_then_glaze(1.0, 0.0)) else {
        return;
    };
    let (x, y) = (SIZE.width / 2, SIZE.height / 2);
    let shift = |img: &stark_core::RgbaImage| {
        (0..3)
            .map(|i| (img.pixel(x, y)[i] as i32 - bar.pixel(x, y)[i] as i32).abs())
            .max()
            .expect("three channels")
    };
    let (thin, thick) = (shift(&swept), shift(&opaque));
    assert!(
        thin * 2 < thick,
        "a {GLAZE_ADD}-flow glaze moved the paint under it {thin} levels where full \
         flow moved it {thick} — `add` is not deciding how much paint lands"
    );
}

/// The `bleed` axis is a *lateral* flux: it must move paint between neighbouring
/// canvas texels and do nothing else. Run alone over a uniform field it therefore
/// has to be very nearly the identity — every neighbourhood difference it trades
/// on is zero — and, because it is not a source, none of the brush's own colour
/// may reach the canvas however hard it scrubs. The brush is deliberately GREEN
/// over RED paint so that failure mode has a colour.
#[test]
fn bleed_alone_neither_lays_paint_nor_disturbs_a_uniform_field() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint(
        &mut engine,
        RED,
        80.0,
        &[Vec2::new(-110.0, 0.0), Vec2::new(110.0, 0.0)],
    );
    let before = engine.render_to_image();
    assert!(is_red(center(&before)), "the field never got painted");

    let b = dyn_brush(
        GREEN,
        24.0,
        BrushDynamics {
            add: 0.0,
            bleed: 0.95,
            ..Default::default()
        },
    );
    // Back and forth, so a compounding leak (a source term hiding in the flux)
    // gets several passes to show itself.
    stroke_with(
        &mut engine,
        b,
        &[
            Vec2::new(-50.0, 0.0),
            Vec2::new(50.0, 0.0),
            Vec2::new(-50.0, 0.0),
            Vec2::new(50.0, 0.0),
        ],
    );
    let after = engine.render_to_image();

    assert!(
        is_red(center(&after)),
        "a bleed-only brush re-tinted the field towards its own colour: {:?}",
        center(&after)
    );
    let frac = frac_exceeding(&before, &after, 40);
    assert!(
        frac < 0.2,
        "diffusing a uniform field should move almost nothing, but {:.1}% of pixels \
         differ by >40 levels",
        frac * 100.0
    );
}

/// The headline behaviour: scrubbing a bleed-only brush along the boundary between
/// two colours softens it — the transition spreads, so its *steepest step* drops.
/// Measured on the per-row difference of a signed red-vs-green signal, which is
/// what "blur" means and does not care which side of the image either stripe
/// landed on, or exactly where the dominance predicates would draw their lines.
/// (A first cut counted rows that read as *neither* colour instead, and failed
/// backwards: the stripes met over a sliver of paper, diffusion pulled paint into
/// the gap, and the "mixed" band narrowed while the edge genuinely softened.)
#[test]
fn bleed_softens_a_colour_boundary() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // Two overlapping stripes — no paper gap at the boundary, so the transition
    // is stripe against stripe and its slope belongs to the meeting edge.
    paint(
        &mut engine,
        RED,
        20.0,
        &[Vec2::new(-90.0, -16.0), Vec2::new(90.0, -16.0)],
    );
    paint(
        &mut engine,
        GREEN,
        20.0,
        &[Vec2::new(-90.0, 16.0), Vec2::new(90.0, 16.0)],
    );

    let (cx, cy) = (SIZE.width / 2, SIZE.height / 2);
    // The steepest per-row step of (green − red) down the boundary's column.
    let max_step = |img: &stark_core::RgbaImage| {
        (cy - 12..cy + 12)
            .map(|y| {
                let s = |y: u32| {
                    let p = img.pixel(cx, y);
                    p[1] as i32 - p[0] as i32
                };
                (s(y + 1) - s(y)).abs()
            })
            .max()
            .expect("non-empty window")
    };
    let before = engine.render_to_image();
    let step_before = max_step(&before);
    assert!(
        step_before >= 25,
        "the stripes should meet in a steep edge, got a max step of {step_before} \
         levels/row — too soft for a blur to measure"
    );

    let b = dyn_brush(
        RED,
        28.0,
        BrushDynamics {
            add: 0.0,
            bleed: 0.95,
            ..Default::default()
        },
    );
    // Several passes along the boundary: the axis is a rate, and each pass of the
    // tip buys another helping of variance.
    stroke_with(
        &mut engine,
        b,
        &[
            Vec2::new(-70.0, 0.0),
            Vec2::new(70.0, 0.0),
            Vec2::new(-70.0, 0.0),
            Vec2::new(70.0, 0.0),
            Vec2::new(-70.0, 0.0),
            Vec2::new(70.0, 0.0),
            Vec2::new(-70.0, 0.0),
            Vec2::new(70.0, 0.0),
            Vec2::new(-70.0, 0.0),
        ],
    );
    let after = engine.render_to_image();
    let step_after = max_step(&after);

    assert!(
        step_after * 10 <= step_before * 8,
        "scrubbing bleed = 0.95 along the boundary should soften its steepest step \
         by at least a fifth, got {step_before} -> {step_after} levels/row"
    );
    // The diffusion is local: 15 px out from the boundary each stripe still reads
    // as its own colour, so the edge softened by mixing rather than by erasure.
    assert!(
        is_red(after.pixel(cx, cy - 15)) && is_green(after.pixel(cx, cy + 15))
            || is_green(after.pixel(cx, cy - 15)) && is_red(after.pixel(cx, cy + 15)),
        "the stripes themselves should survive a boundary blur"
    );
}

/// The property the reach fix exists for: `bleed`'s smoothing distance scales
/// with the brush, so a big blender softens far from the boundary in one pass —
/// where a fixed 1-texel stencil tops out near a pixel of σ per pass and cannot
/// move a reading 10 px away no matter the rate (the regression this pins: the
/// axis "worked" but was invisible at any usable brush size).
///
/// Measured as the red-vs-green contrast 10 px to each side of the boundary
/// dropping materially after a single pass, which no orientation, colour-space
/// or dominance-threshold detail can fake: only paint arriving from across the
/// boundary moves those readings toward one another.
#[test]
fn bleed_reach_scales_with_the_brush() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint(
        &mut engine,
        RED,
        20.0,
        &[Vec2::new(-90.0, -16.0), Vec2::new(90.0, -16.0)],
    );
    paint(
        &mut engine,
        GREEN,
        20.0,
        &[Vec2::new(-90.0, 16.0), Vec2::new(90.0, 16.0)],
    );

    let (cx, cy) = (SIZE.width / 2, SIZE.height / 2);
    // Signed red-vs-green contrast at a pixel; |it| is purity of whichever side.
    let contrast = |img: &stark_core::RgbaImage, y: u32| {
        let p = img.pixel(cx, y);
        (p[1] as i32 - p[0] as i32).abs()
    };
    let before = engine.render_to_image();
    let (b_lo, b_hi) = (contrast(&before, cy - 10), contrast(&before, cy + 10));
    assert!(
        b_lo >= 60 && b_hi >= 60,
        "10 px out each stripe should still read nearly pure, got {b_lo}/{b_hi}"
    );

    let b = dyn_brush(
        RED,
        60.0,
        BrushDynamics {
            add: 0.0,
            bleed: 0.95,
            ..Default::default()
        },
    );
    // One pass. The reach has to come from the stencil scaling with the radius,
    // not from scrubbing long enough that any stencil would get there.
    stroke_with(
        &mut engine,
        b,
        &[Vec2::new(-70.0, 0.0), Vec2::new(70.0, 0.0)],
    );
    let after = engine.render_to_image();
    let (a_lo, a_hi) = (contrast(&after, cy - 10), contrast(&after, cy + 10));

    assert!(
        a_lo + 25 <= b_lo && a_hi + 25 <= b_hi,
        "one pass of a radius-60 blur should mix the stripes 10 px out on both \
         sides, got {b_lo} -> {a_lo} and {b_hi} -> {a_hi}"
    );
}

/// The regression the bleed cadence's fire slots and the deposit's rewrite guard
/// exist for (§6.2): a slow hand is fitted at a control point per pointer sample,
/// so a bleed-only stroke can arrive as hundreds of **sub-pixel** knots — and over
/// a flat coat it must change nothing at all. It used to brighten the coat by up
/// to 28 levels: fired per segment, the per-texel exposure is prefix-cancellation
/// noise and the flux sits under the f16 ULP of the heights it edits — and even at
/// zero flux, re-storing an algebraically identical texel walks it down one ULP
/// per rewrite on a backend whose f32→f16 storage conversion truncates (D3D12
/// does). The dense record is appended to the log directly, exactly as the field
/// repro's fitter produced it, rather than through the fitter.
#[test]
fn a_dense_bleed_scribble_over_flat_paint_is_a_no_op() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint(
        &mut engine,
        RED,
        80.0,
        &[Vec2::new(-110.0, 0.0), Vec2::new(110.0, 0.0)],
    );
    let before = engine.render_to_image();

    let mut doc = stark_core::io::DocumentFile::from_bytes(&engine.save_bytes().expect("save"))
        .expect("parse own save");
    let prev = doc.actions.last().expect("the coat stroke").id;
    let mut brush = dyn_brush(
        GREEN,
        40.0,
        BrushDynamics {
            add: 0.0,
            bleed: 0.95,
            ..Default::default()
        },
    );
    brush.drain = 0.0;
    // A shallow 72 px arc over 200 knots — every span ~0.36 px, like the repro.
    let path: Vec<stark_core::path::ControlPoint> = (0..200)
        .map(|i| {
            let ang = (i as f32 / 199.0 - 0.5) * 1.2;
            let mut cp = stark_core::path::ControlPoint::at(Vec2::new(
                ang.sin() * 60.0,
                (ang.cos() - 1.0) * 60.0,
            ));
            cp.pressure = 0.5;
            cp
        })
        .collect();
    doc.actions.push(stark_core::document::Action {
        id: stark_core::document::ActionId {
            lamport: prev.lamport + 1,
            actor: prev.actor,
        },
        kind: stark_core::document::ActionKind::CommitStroke(stark_core::document::StrokeRecord {
            layer: engine.observe().active_layer,
            tool: Tool::Brush,
            brush,
            path,
            seed: 7,
        }),
    });

    let Some(mut replayed) = engine_or_skip() else {
        return;
    };
    replayed
        .load_bytes(&doc.to_bytes().expect("encode"))
        .expect("load + replay");
    let after = replayed.render_to_image();
    assert!(
        images_match(&before, &after, 2),
        "a dense bleed-only scribble over flat paint must leave it flat"
    );
}
