//! The eyedropper (§18.0.2): sampling colour back off the canvas.
//!
//! The claim it rests on is that it reads the **raw layer channels** rather than the
//! composited, lit result — so what comes back is a colour the palette could have
//! mixed, and in a Mixbox document a pigment mixture that can be picked back up
//! (§6.7). These check the things that would quietly betray that: that a
//! painted colour survives the round trip in *both* colour spaces, that bare canvas
//! answers "nothing" instead of the paper colour, and that the layer and radius
//! options select what they say they do.

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
#[test]
fn picks_the_paint_rather_than_the_lit_pixel() {
    let Some(mut engine) = engine_or_skip() else {
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
