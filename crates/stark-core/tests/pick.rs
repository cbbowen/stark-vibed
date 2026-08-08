//! The eyedropper (§18.0.2): sampling colour back off the canvas.
//!
//! The claim it rests on is that it reads the **raw layer channels** rather than the
//! composited, lit result — so what comes back is a colour the palette could have
//! mixed, and in a Mixbox document a pigment mixture that can be picked back up
//! (§6.7). These check the things that would quietly betray that: that a
//! painted colour survives the round trip in *both* colour spaces, that bare canvas
//! answers "nothing" instead of the paper colour, and that the layer and radius
//! options select what they say they do.
//!
//! The one source that *does* answer on bare canvas is the composite over the
//! substrate, and what it has to be held to is the opposite pair: that the ground
//! shows through where the paint does not cover, and that it stays out of the way
//! where the paint does.

mod common;

use common::*;
use stark_core::colorspace::ColorSpaceId;
use stark_core::command::{DocCommand, PeerCommand};
use stark_core::geom::Vec2;
use stark_core::{Engine, LayerId, PickOptions, PickSource};

const RED: [f32; 4] = [0.85, 0.12, 0.1, 1.0];
const BLUE: [f32; 4] = [0.1, 0.2, 0.8, 1.0];

/// A short horizontal stroke through the origin, wide enough that the middle is
/// solidly covered.
const BAR: &[Vec2] = &[Vec2::new(-40.0, 0.0), Vec2::new(40.0, 0.0)];

fn pick(engine: &mut Engine, at: Vec2, options: PickOptions) -> Option<[f32; 3]> {
    pollster::block_on(engine.pick_color(at, options))
}

/// Point-sample the composite — the default an Alt+click takes.
fn pick_point(engine: &mut Engine, at: Vec2) -> Option<[f32; 3]> {
    pick(engine, at, PickOptions::default())
}

/// The same stack with the canvas colour behind it.
fn over_substrate(radius: u32) -> PickOptions {
    PickOptions {
        source: PickSource::CompositeOverSubstrate,
        radius,
    }
}

fn near(a: [f32; 3], b: [f32; 4], tol: f32) -> bool {
    (0..3).all(|i| (a[i] - b[i]).abs() <= tol)
}

fn assert_near(got: Option<[f32; 3]>, want: [f32; 4], tol: f32, what: &str) {
    let got = got.unwrap_or_else(|| panic!("{what}: nothing picked"));
    assert!(
        near(got, want, tol),
        "{what}: picked {got:?}, wanted ~{:?}",
        [want[0], want[1], want[2]]
    );
}

/// The headline: paint a colour, pick it back up, get the colour you painted.
///
/// It has to hold in both colour spaces and it is a *different* claim in each. Oklab
/// stores `(L, a, b)` and Mixbox stores pigment concentrations, so a pick that
/// forgot to run the channels back through the colour space would come out roughly
/// right in one and wildly wrong in the other.
#[test]
fn picks_the_colour_that_was_painted() {
    for space in [ColorSpaceId::Oklab, ColorSpaceId::Mixbox] {
        let Some(mut engine) = engine_or_skip_with(space) else {
            return;
        };
        paint(&mut engine, RED, 24.0, BAR);
        assert_near(
            pick_point(&mut engine, Vec2::ZERO),
            RED,
            0.03,
            &format!("{space:?}"),
        );
    }
}

/// And it is *not* the lit result. The media pass lights, tonemaps and sRGB-encodes
/// what it composites, so a screen pixel over the same paint is a different number —
/// which is the whole reason the eyedropper samples the raw channels instead.
///
/// The one place outside `golden.rs` that asks for the studio HDR, and it has to: the
/// rest of the suite paints under the reference light, whose entire purpose is that
/// what you painted is what you see (§6.3). Under it the lit pixel *is* the paint to
/// within a couple of levels, and the second assertion below — which exists to keep
/// this test from being vacuous — correctly refuses to pass. A test that the picker
/// bypasses the light needs a light that does something.
#[test]
fn picks_the_paint_rather_than_the_lit_pixel() {
    let Some(mut engine) = engine_or_skip_studio() else {
        return;
    };
    paint(&mut engine, RED, 24.0, BAR);
    let picked = pick_point(&mut engine, Vec2::ZERO).expect("paint under the point");

    // The same texel as the screen shows it: the view is 1:1 centred on the origin,
    // so canvas (0,0) is the middle of the viewport.
    let img = engine.render_to_image();
    let lit = img.pixel(img.width / 2, img.height / 2);
    let lit = [
        lit[0] as f32 / 255.0,
        lit[1] as f32 / 255.0,
        lit[2] as f32 / 255.0,
    ];
    assert!(
        near(picked, RED, 0.03),
        "the pick should be the paint's own colour, got {picked:?}"
    );
    assert!(
        !near(lit, RED, 0.03),
        "this test is vacuous unless the studio lighting actually moves the colour; \
         the lit pixel came back {lit:?}"
    );
}

/// Bare canvas has nothing to pick. The substrate is the ground, not something a
/// brush picks up, so an empty patch answers `None` rather than loading the brush
/// with the paper colour.
#[test]
fn bare_canvas_has_nothing_to_pick() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    assert_eq!(pick_point(&mut engine, Vec2::ZERO), None, "empty document");

    paint(&mut engine, RED, 24.0, BAR);
    assert!(pick_point(&mut engine, Vec2::ZERO).is_some(), "on the bar");
    assert_eq!(
        pick_point(&mut engine, Vec2::new(0.0, 120.0)),
        None,
        "well clear of the only stroke"
    );
}

/// Except over the substrate, where the ground *is* what was asked for: an empty
/// patch answers with the canvas colour rather than with nothing. And with the
/// document's colour, read at the moment of the sample — a remembered default would
/// pass every test that never repaints the canvas.
#[test]
fn over_the_substrate_answers_with_the_canvas_colour() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    const GROUND: [f32; 3] = [0.2, 0.55, 0.35];
    engine.process(DocCommand::SetBackground(GROUND));

    assert_near(
        pick(&mut engine, Vec2::ZERO, over_substrate(0)),
        [GROUND[0], GROUND[1], GROUND[2], 1.0],
        0.02,
        "bare canvas is the canvas colour",
    );
    assert_eq!(
        pick_point(&mut engine, Vec2::ZERO),
        None,
        "and the other sources still have nothing to pick there"
    );
}

/// What the source is *for*. Where the paint does not cover, the canvas fills in, and
/// the two composited sources disagree exactly there: the plain composite weighs by
/// opacity, so bare texels count for nothing and it reports the stroke alone, while
/// over the substrate the same patch answers with the mixture an eye sees.
///
/// The ground is blue and the paint red so the disagreement is a channel apart rather
/// than a shade apart.
#[test]
fn thin_coverage_mixes_toward_the_canvas() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    paint(&mut engine, RED, 24.0, BAR);

    // Straddling the bar's edge: the bar reaches y ≈ 24, and a 24-px patch centred at
    // 30 has rather less than half of itself on paint.
    let at = Vec2::new(0.0, 30.0);
    let paint_only = pick(
        &mut engine,
        at,
        PickOptions {
            radius: 24,
            ..PickOptions::default()
        },
    )
    .expect("paint in the patch");
    let with_ground =
        pick(&mut engine, at, over_substrate(24)).expect("the ground is always there");

    assert!(
        near(paint_only, RED, 0.06),
        "an opacity-weighted mean is not diluted by the bare part of the patch, \
         got {paint_only:?}"
    );
    assert!(
        with_ground[2] > paint_only[2] + 0.1 && with_ground[0] < paint_only[0] - 0.1,
        "the blue ground should show through where the paint does not cover: \
         {with_ground:?} against {paint_only:?}"
    );
    assert!(
        with_ground[0] > 0.1,
        "and it is a mixture, not a jump to the ground: {with_ground:?}"
    );

    // Where the paint *does* cover, the ground behind it changes nothing — otherwise
    // the mode would be tinting the paint rather than filling in behind it.
    assert_near(
        pick(&mut engine, Vec2::ZERO, over_substrate(0)),
        RED,
        0.08,
        "opaque paint hides the ground",
    );
}

/// Sampling one layer is the colour that layer would have *alone* — not the colour
/// the stack shows at that point. Painting blue over red and asking each question
/// separately is the only way to tell the two apart.
#[test]
fn one_layer_ignores_the_layers_over_it() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let under = engine.observe().active_layer;
    paint(&mut engine, RED, 24.0, BAR);

    engine.process(DocCommand::AddLayer {
        carrier: None,
        above: None,
    });
    let over = engine.observe().active_layer;
    assert_ne!(over, under, "AddLayer should arm the new layer");
    paint(&mut engine, BLUE, 24.0, BAR);

    assert_near(
        pick_point(&mut engine, Vec2::ZERO),
        BLUE,
        0.05,
        "the composite is what the top layer shows",
    );
    let one_layer = |source| PickOptions {
        source,
        ..PickOptions::default()
    };
    assert_near(
        pick(&mut engine, Vec2::ZERO, one_layer(PickSource::Layer(under))),
        RED,
        0.05,
        "the layer underneath still holds its own paint",
    );
    assert_near(
        pick(&mut engine, Vec2::ZERO, one_layer(PickSource::Layer(over))),
        BLUE,
        0.05,
        "and the top layer holds its own",
    );
    assert_eq!(
        pick(
            &mut engine,
            Vec2::ZERO,
            one_layer(PickSource::Layer(LayerId(9999)))
        ),
        None,
        "a layer that is not there holds nothing"
    );
}

/// A hidden layer is not sampled, because a sample comes off the same stack the
/// screen draws — the option list is shared with rendering precisely so that this
/// cannot drift.
#[test]
fn hidden_layers_are_not_sampled() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let under = engine.observe().active_layer;
    paint(&mut engine, RED, 24.0, BAR);
    engine.process(DocCommand::AddLayer {
        carrier: None,
        above: None,
    });
    let over = engine.observe().active_layer;
    paint(&mut engine, BLUE, 24.0, BAR);

    engine.process(DocCommand::SetLayerVisible(over, false));
    assert_near(
        pick_point(&mut engine, Vec2::ZERO),
        RED,
        0.05,
        "hiding the blue layer should uncover the red one",
    );
    let _ = PeerCommand::SetActiveLayer(under);
}

/// The radius averages the patch, weighted by how much paint is in it. Two bars side
/// by side: a point sample lands on one of them, a wide one comes back between the
/// two — and never outside the range they span.
#[test]
fn radius_averages_the_patch() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // Butted together at x = 0: red to the left, blue to the right.
    paint(
        &mut engine,
        RED,
        8.0,
        &[Vec2::new(-40.0, -8.0), Vec2::new(-40.0, 8.0)],
    );
    paint(
        &mut engine,
        BLUE,
        8.0,
        &[Vec2::new(-24.0, -8.0), Vec2::new(-24.0, 8.0)],
    );

    let at = Vec2::new(-40.0, 0.0);
    assert_near(
        pick_point(&mut engine, at),
        RED,
        0.05,
        "a point sample takes the texel it is on",
    );

    let wide = pick(
        &mut engine,
        at,
        PickOptions {
            radius: 12,
            ..PickOptions::default()
        },
    )
    .expect("paint in the patch");
    assert!(
        wide[2] > RED[2] + 0.05,
        "a patch reaching the blue bar should pull blue in, got {wide:?}"
    );
    assert!(
        wide[2] < BLUE[2] && wide[0] > BLUE[0],
        "and it is a mixture, not a jump to the other bar: {wide:?}"
    );
}

/// The radius is bounded rather than trusted: one sample is a render plus a
/// readback, so an absurd radius must not turn into an absurd texture.
#[test]
fn an_absurd_radius_is_clamped_not_obeyed() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint(&mut engine, RED, 24.0, BAR);
    assert_near(
        pick(
            &mut engine,
            Vec2::ZERO,
            PickOptions {
                radius: u32::MAX,
                ..PickOptions::default()
            },
        ),
        RED,
        0.2,
        "a huge radius should still answer",
    );
}
