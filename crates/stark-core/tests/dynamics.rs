//! Brush-dynamics stamp-loop tests (DESIGN.md §6.2): the sequential pickup ⇄
//! deposit exchange against a 2-D tool reservoir. Covers conservation (paint
//! moves by height, is neither created nor destroyed), smearing onto bare
//! canvas, erasing without re-tinting, the pre-`charge` glob, and determinism.

mod common;

use common::*;
use stark_core::command::{GestureCommand, InputSample, ViewCommand};
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
    // recently when the media pass became a reference (DESIGN.md §6.3) and bare paper
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
