//! Brush parameter modulation (§6.2): the pen driving parameters other than the
//! radius, and driving the radius with something other than pressure.
//!
//! Every test here holds the **geometry** fixed and varies only what the pen
//! reports, so what it measures is the mapping and not the stroke. Two strokes at
//! two constant pressures (or tilts) is the whole method: a ramp along one stroke
//! would confound the mapping with the taper, the drain and the arc-length noise,
//! and a difference between the ends of one mark is a much weaker claim than a
//! difference between two marks that are otherwise the same object.

mod common;

use common::*;
use stark_engine::command::{DocCommand, GestureCommand, InputSample, ViewCommand};
use stark_engine::document::{
    BrushDynamics, BrushParams, BrushShape, ModSource, Modulation, Modulations, Tool,
};
use stark_engine::geom::Vec2;
use stark_engine::path::DEFAULT_TOLERANCE;

const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];

/// A test brush with no `drain`: the run-dry falloff is the one *other* thing that
/// varies along a stroke, and these tests are about what varies across one.
fn plain(radius: f32, modulation: Modulations) -> BrushParams {
    BrushParams {
        drain: 0.0,
        modulation,
        ..brush(RED, radius)
    }
}

/// Paint and commit a stroke through `points` with the pen held at one `pressure`
/// and one `tilt` throughout.
///
/// Constant rather than ramped on purpose (see the module note). `tilt` is a lean
/// along +x, which with the default `FollowStroke` orientation reaches nothing but
/// the modulations — so a difference this produces cannot be the footprint turning.
fn stroke_pen(engine: &mut stark_engine::Engine, b: BrushParams, points: &[Vec2], pen: (f32, f32)) {
    let sample = |pos| InputSample {
        pos,
        pressure: pen.0,
        tilt: Vec2::new(pen.1, 0.0),
        time: 0.0,
    };
    engine.process(ViewCommand::SetBrush(b));
    let mut it = points.iter();
    let first = *it.next().expect("at least one point");
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: sample(first),
        tolerance: DEFAULT_TOLERANCE,
        rope: 0.0,
    });
    for &p in it {
        engine.process(GestureCommand::To { sample: sample(p) });
    }
    engine.process(GestureCommand::End);
}

/// The stroke this file draws every time: a straight run across the middle.
fn run() -> [Vec2; 2] {
    [Vec2::new(-120.0, 0.0), Vec2::new(120.0, 0.0)]
}

/// How much the stroke changed the canvas: the summed per-pixel worst-channel
/// difference from `before`, in levels.
///
/// A total rather than a count, so it answers *how much paint* rather than *how many
/// pixels were touched* — which is the difference between the flow tests and the size
/// test below, and why they use different measures.
fn ink(before: &stark_engine::RgbaImage, after: &stark_engine::RgbaImage) -> f64 {
    before
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .zip(after.pixels.as_chunks::<4>().0)
        .map(|(a, b)| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (*x as i32 - *y as i32).abs())
                .max()
                .unwrap_or(0) as f64
        })
        .sum()
}

/// Pixels the stroke moved by more than a hair — the mark's **extent**, which is what
/// a size mapping changes.
fn covered(before: &stark_engine::RgbaImage, after: &stark_engine::RgbaImage) -> u64 {
    before
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .zip(after.pixels.as_chunks::<4>().0)
        .filter(|(a, b)| {
            a.iter()
                .zip(b.iter())
                .any(|(x, y)| (*x as i32 - *y as i32).abs() > 8)
        })
        .count() as u64
}

/// Draw `b` twice at two pen states, undoing between, and return the two marks'
/// measures against the same clean canvas.
fn two_marks(
    engine: &mut stark_engine::Engine,
    b: BrushParams,
    a_pen: (f32, f32),
    b_pen: (f32, f32),
    measure: impl Fn(&stark_engine::RgbaImage, &stark_engine::RgbaImage) -> f64,
) -> (f64, f64) {
    let before = engine.render_to_image();
    stroke_pen(engine, b, &run(), a_pen);
    let first = measure(&before, &engine.render_to_image());
    engine.process(DocCommand::Undo);
    stroke_pen(engine, b, &run(), b_pen);
    let second = measure(&before, &engine.render_to_image());
    (first, second)
}

/// The property `None` on a target is *for*: an unmapped parameter is not multiplied
/// by anything, so the input it might have been mapped to reaches nothing at all.
///
/// Asserted as **bit equality** rather than similarity, because that is the actual
/// claim — `Modulations::factor` skips a `None` instead of multiplying by a 1 it
/// computed. A brush with nothing mapped draws the same mark whatever the hand does,
/// which is a marker pen, and something no version of this app could do before.
#[test]
fn an_unmapped_brush_cannot_feel_the_pen() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let b = plain(30.0, Modulations::default());
    let before = engine.render_to_image();
    stroke_pen(&mut engine, b, &run(), (1.0, 0.9));
    let heavy = engine.render_to_image();
    engine.process(DocCommand::Undo);
    stroke_pen(&mut engine, b, &run(), (0.12, 0.0));
    let light = engine.render_to_image();

    let (frac, worst) = diff_fraction(&heavy, &light);
    assert_eq!(
        (frac, worst),
        (0.0, 0),
        "nothing is mapped to the pen, so the two marks must be the same pixels"
    );
    assert!(
        ink(&before, &heavy) > 0.0,
        "the test drew nothing at all — it is measuring an empty canvas"
    );
}

/// Flow mapped to pressure: the same tip lays less paint the lighter the hand, with
/// the mark's *extent* unchanged because size is deliberately left unmapped.
///
/// This is the swept fast path (`add` alone), which is where the rate had to move off
/// the per-tile uniform and onto the segment instance to be reachable at all.
#[test]
fn flow_follows_pressure_when_it_is_mapped_to_it() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let b = plain(
        30.0,
        Modulations {
            flow: Some(Modulation::linear(ModSource::Pressure)),
            ..Modulations::default()
        },
    );
    let (heavy, light) = two_marks(&mut engine, b, (1.0, 0.0), (0.15, 0.0), ink);
    assert!(
        light < heavy * 0.6,
        "a light touch should lay markedly less paint: {light} vs {heavy}"
    );
    assert!(
        light > 0.0,
        "…but a 0.15 press through a linear mapping is not nothing"
    );

    // The control: the same two presses with flow unmapped lay the same paint. What
    // this rules out is the difference above being anything other than the mapping —
    // the geometry, the dab, the fitter's own response to pressure.
    let flat = plain(30.0, Modulations::default());
    let (a, b) = two_marks(&mut engine, flat, (1.0, 0.0), (0.15, 0.0), ink);
    assert_eq!(a, b, "unmapped, the two presses must lay the same paint");
}

/// Size driven by **tilt** instead of pressure — the pencil that widens as it is laid
/// over — and pressure no longer reaching the radius at all.
///
/// The second half is the part worth having. Wire the pressure → size scaling into the
/// segment generator and there is nowhere else to point it: a brush that sizes off
/// tilt cannot exist, and neither can one that sizes off nothing.
#[test]
fn size_can_be_driven_by_tilt_instead_of_pressure() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let b = plain(
        40.0,
        Modulations {
            size: Some(Modulation::linear(ModSource::Tilt)),
            ..Modulations::default()
        },
    );
    let wide = |a: &_, c: &_| covered(a, c) as f64;
    let (flat_pen, upright) = two_marks(&mut engine, b, (1.0, 1.0), (1.0, 0.2), wide);
    assert!(
        upright < flat_pen * 0.5,
        "leaning the pen over should widen the mark: {upright} vs {flat_pen} px"
    );

    // And pressure is now inert on this brush: it is mapped to nothing.
    let (hard, soft) = two_marks(&mut engine, b, (1.0, 0.7), (0.2, 0.7), wide);
    assert_eq!(
        hard, soft,
        "with size mapped to tilt, pressure must not reach the radius"
    );
}

/// Deposit mapped to tilt, on the stamp loop: the palette knife that lays its load
/// when it is put down flat and holds it when it is upright.
///
/// The upright case is exact rather than merely small — a mapped factor of 0 makes
/// `λ_deposit` exactly 0, so the tool gives up nothing, and the settle at pen-up
/// reads the same rate and gives up nothing either. A knife that dribbled at the end
/// of every stroke would be the tell that the pen-up slot had been left on the
/// brush's own rate.
#[test]
fn deposit_follows_tilt_on_the_stamp_loop() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let b = BrushParams {
        dynamics: BrushDynamics {
            add: 0.0,
            lift: 0.0,
            deposit: 0.9,
            charge: 1.5,
            bleed: 0.0,
        },
        shape: BrushShape::Round { hardness: 0.9 },
        modulation: Modulations {
            deposit: Some(Modulation::linear(ModSource::Tilt)),
            ..Modulations::default()
        },
        ..plain(30.0, Modulations::default())
    };

    let before = engine.render_to_image();
    stroke_pen(&mut engine, b, &run(), (1.0, 0.0));
    let upright = engine.render_to_image();
    let (frac, worst) = diff_fraction(&before, &upright);
    assert_eq!(
        (frac, worst),
        (0.0, 0),
        "an upright pen deposits nothing, at no point of the stroke and not at pen-up"
    );

    engine.process(DocCommand::Undo);
    stroke_pen(&mut engine, b, &run(), (1.0, 1.0));
    let laid = ink(&before, &engine.render_to_image());
    assert!(
        laid > 0.0,
        "laid flat, the same knife must put its load down"
    );
}
