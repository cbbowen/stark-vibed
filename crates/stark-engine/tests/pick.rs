//! The eyedropper (§18.0.2): sampling color back off the canvas.
//!
//! The claim it rests on is that it reads the **raw layer channels** rather than the
//! composited, lit result — so what comes back is a color the palette could have
//! mixed, and in a Mixbox document a pigment mixture that can be picked back up
//! (§6.7). These check the things that would quietly betray that: that a
//! painted color survives the round trip in *both* color spaces, that bare canvas
//! answers "nothing" instead of the paper color, and that the layer, group, below
//! and radius options select what they say they do.
//!
//! The one source that *does* answer on bare canvas is the composite over the
//! substrate, and what it has to be held to is the opposite pair: that the substrate
//! shows through where the paint does not cover, and that it stays out of the way
//! where the paint does.
//!
//! The **layer** hit test (§16.11) is here for the same reason its implementation
//! is: it is the other question asked of a point on the canvas, and it answers
//! off the same draw list. What it rests on is that "topmost" means *last drawn*
//! and "there" means the screen shows it — so the claims below are about
//! stacking, about hiding (a layer's own and the group's above it), and about the
//! difference between a layer turned down and a layer switched off.

mod common;

use common::*;
use stark_engine::command::{DocCommand, PeerCommand};
use stark_engine::{Engine, PickOptions, PickSource};
use stark_model::Srgb;
use stark_model::document::{BlendMode, LayerId, Place};
use stark_model::geom::Vec2;

const RED: [f32; 3] = [0.85, 0.12, 0.1];
const BLUE: [f32; 3] = [0.1, 0.2, 0.8];

/// The layer every fresh document starts with.
const ROOT: LayerId = LayerId::ROOT;

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

/// The same stack with the canvas color behind it.
fn over_substrate(radius: u32) -> PickOptions {
    PickOptions {
        source: PickSource::CompositeOverSubstrate,
        radius,
    }
}

fn near(a: [f32; 3], b: [f32; 3], tol: f32) -> bool {
    (0..3).all(|i| (a[i] - b[i]).abs() <= tol)
}

fn assert_near(got: Option<[f32; 3]>, want: [f32; 3], tol: f32, what: &str) {
    let got = got.unwrap_or_else(|| panic!("{what}: nothing picked"));
    assert!(
        near(got, want, tol),
        "{what}: picked {got:?}, wanted ~{:?}",
        [want[0], want[1], want[2]]
    );
}

/// The headline: paint a color, pick it back up, get the color you painted.
///
/// It has to hold in both color spaces and it is a *different* claim in each. Oklab
/// stores `(L, a, b)` and Mixbox stores pigment concentrations, so a pick that
/// forgot to run the channels back through the color space would come out roughly
/// right in one and wildly wrong in the other.
#[test]
fn picks_the_color_that_was_painted() {
    for space in stark_engine::colorspace::all_available() {
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
        "the pick should be the paint's own color, got {picked:?}"
    );
    assert!(
        !near(lit, RED, 0.03),
        "this test is vacuous unless the studio lighting actually moves the color; \
         the lit pixel came back {lit:?}"
    );
}

/// Bare canvas has nothing to pick. The substrate is the substrate, not something a
/// brush picks up, so an empty patch answers `None` rather than loading the brush
/// with the paper color.
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

/// Except over the substrate, where it *is* what was asked for: an empty
/// patch answers with the canvas color rather than with nothing. And with the
/// document's color, read at the moment of the sample — a remembered default would
/// pass every test that never repaints the canvas.
#[test]
fn over_the_substrate_answers_with_the_canvas_color() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    const SUBSTRATE: [f32; 3] = [0.2, 0.55, 0.35];
    engine.process(DocCommand::SetSubstrateColor(Srgb::new(SUBSTRATE)));

    assert_near(
        pick(&mut engine, Vec2::ZERO, over_substrate(0)),
        SUBSTRATE,
        0.02,
        "bare canvas is the canvas color",
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
/// The substrate is blue and the paint red so the disagreement is a channel apart rather
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
    let with_substrate =
        pick(&mut engine, at, over_substrate(24)).expect("the substrate is always there");

    assert!(
        near(paint_only, RED, 0.06),
        "an opacity-weighted mean is not diluted by the bare part of the patch, \
         got {paint_only:?}"
    );
    assert!(
        with_substrate[2] > paint_only[2] + 0.1 && with_substrate[0] < paint_only[0] - 0.1,
        "the blue substrate should show through where the paint does not cover: \
         {with_substrate:?} against {paint_only:?}"
    );
    assert!(
        with_substrate[0] > 0.1,
        "and it is a mixture, not a jump to the substrate: {with_substrate:?}"
    );

    // Where the paint *does* cover, the substrate behind it changes nothing — otherwise
    // the mode would be tinting the paint rather than filling in behind it.
    assert_near(
        pick(&mut engine, Vec2::ZERO, over_substrate(0)),
        RED,
        0.08,
        "opaque paint hides the substrate",
    );
}

/// Sampling one layer is the color that layer would have *alone* — not the color
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
            one_layer(PickSource::Layer(LayerId::solo(9999)))
        ),
        None,
        "a layer that is not there holds nothing"
    );
}

/// **A pick over a blended stack builds a scratch, and the scratch has to outlive the
/// submit.**
///
/// A blend mode is not `Normal` "over", so pass A cannot draw the stack in one run:
/// `Plan::build` isolates the blended layer into a scratch level and bounces the two
/// halves together (§6.5). That scratch is the pick's own — a few kilobytes at a
/// patch's size, rather than the render path's window-sized cache — and its
/// attachments `destroy()` themselves on drop, which is right for a texture whose last
/// use has been *submitted* and fatal for one whose commands are only recorded. A
/// gradient trace records every patch into one encoder and submits once, so "recorded"
/// and "in flight" are a hundred patches apart.
///
/// Nothing else here reaches that plan. Every other test in this file paints into a
/// plain stack, where `plan.scratch` is empty and no scratch is built at all — so the
/// whole class was covered by the arithmetic of the picked colour and nothing else.
/// What this asserts is only that the pick still answers, because a submit that fails
/// validation loses the entire command buffer and the readback comes back `None`.
#[test]
fn a_pick_over_a_blended_stack_still_answers() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint(&mut engine, RED, 24.0, BAR);
    engine.process(DocCommand::AddLayer {
        carrier: None,
        above: None,
    });
    let over = engine.observe().active_layer;
    paint(&mut engine, BLUE, 24.0, BAR);
    engine.process(DocCommand::SetLayerBlend(over, BlendMode::Multiply));

    // A single point first, which is the one-patch case and would survive a lost
    // submit only by accident.
    assert!(
        pick_point(&mut engine, Vec2::ZERO).is_some(),
        "a blended stack under the point and nothing came back",
    );

    // Then a trace, which is the shape the batching is for: one prepared plan, many
    // patches, one submit. Along the bar, so every patch has paint under it — and a
    // gradient comes back only if the whole command buffer landed, since a failed
    // submit leaves every patch's target unwritten and every sample answering `None`.
    let traced = pollster::block_on(engine.pick_gradient(
        &[Vec2::new(-30.0, 0.0), Vec2::new(30.0, 0.0)],
        PickOptions::default(),
    ));
    assert!(
        traced.is_some(),
        "the trace over a blended stack found no paint at all — its submit was lost",
    );
}

/// Sampling one layer **ignores its opacity slider** — and stops entirely at zero.
///
/// Turning a layer down does not turn its paint into a paler paint; it says how much
/// of the layer the *document* shows, which is the question the other two sources
/// ask. So `Layer` reports the same color at every setting above zero.
///
/// Zero is the exception, and it is a different kind of statement: a layer turned all
/// the way down contributes nothing, so sampling it answers "nothing here" — the same
/// answer bare canvas gives — rather than reporting paint that is switched off.
///
/// **What actually changed, stated exactly**: not the color, but where the pick
/// answers at all. The pick divides by the coverage it sums, so a layer's opacity
/// always cancelled out of the *color* — the ordinary settings below would have
/// passed before the slider was dropped, and they are here as a pin rather than as a
/// regression. What the opacity did not cancel out of is `PICK_MIN_OPACITY`, the
/// floor beneath which a patch is called empty: it scaled the coverage towards that
/// floor, so far enough down, solid paint reported nothing at all.
///
/// `0.001` is where that bites for paint this thick, and it is the case that fails
/// without the change. It is deliberately an extreme value: the boundary is what is
/// under test, and quoting a realistic one that happened to sit above the floor would
/// be a test that passes either way — which the first draft of this one was.
#[test]
fn one_layer_ignores_its_own_opacity_until_it_reaches_zero() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let layer = engine.observe().active_layer;
    paint(&mut engine, RED, 24.0, BAR);
    let source = PickOptions {
        source: PickSource::Layer(layer),
        ..PickOptions::default()
    };
    let full = pick(&mut engine, Vec2::ZERO, source).expect("opaque paint picks");

    for opacity in [0.75, 0.25, 0.02, 0.001] {
        engine.process(DocCommand::SetLayerOpacity(layer, opacity));
        assert_near(
            pick(&mut engine, Vec2::ZERO, source),
            [full[0], full[1], full[2]],
            0.02,
            &format!("the layer's paint at opacity {opacity}"),
        );
    }

    // Zero is a different statement, not a fainter one: the layer contributes nothing
    // to the document, so there is nothing of it to sample.
    engine.process(DocCommand::SetLayerOpacity(layer, 0.0));
    assert_eq!(
        pick(&mut engine, Vec2::ZERO, source),
        None,
        "a layer turned all the way down has nothing to sample",
    );

    // …and the *composite* still fades with the slider, which is what says this is a
    // property of the one-layer source rather than of the pick as a whole.
    engine.process(DocCommand::SetLayerOpacity(layer, 1.0));
    let opaque = pick_point(&mut engine, Vec2::ZERO).expect("the composite shows paint");
    engine.process(DocCommand::SetLayerOpacity(layer, 0.0));
    assert_eq!(
        pick_point(&mut engine, Vec2::ZERO),
        None,
        "the composite of a layer turned all the way down is bare canvas",
    );
    assert!(
        near(opaque, RED, 0.05),
        "…where at full strength it is paint"
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

const GREEN: [f32; 3] = [0.1, 0.7, 0.2];
const YELLOW: [f32; 3] = [0.85, 0.8, 0.1];

/// Bars for the scoped-source tests, spatially separated so a point sample says
/// which layer answered.
const BAR_100: &[Vec2] = &[Vec2::new(-40.0, 100.0), Vec2::new(40.0, 100.0)];
const BAR_200: &[Vec2] = &[Vec2::new(-40.0, 200.0), Vec2::new(40.0, 200.0)];

/// The document the two scoped sources are held against, and the ids that name
/// its parts:
///
/// ```text
/// root:  L1 [ M2 yellow @100, M1 blue @100, base green @200 ]   (the group)
///        L0 red @0
/// ```
///
/// M2 and M1 share a bar so "above within the group" is a real occlusion; L0's
/// red and the base's green sit outside every other bar so a sample at their y
/// can only have come from them.
fn scoped_doc(engine: &mut Engine) -> (LayerId, LayerId, LayerId, LayerId) {
    let l0 = engine.observe().active_layer;
    paint(engine, RED, 24.0, BAR);
    engine.process(DocCommand::AddLayer {
        carrier: None,
        above: None,
    });
    let l1 = engine.observe().active_layer;
    paint(engine, GREEN, 24.0, BAR_200);
    engine.process(DocCommand::AddLayer {
        carrier: Some(l1),
        above: None,
    });
    let m1 = engine.observe().active_layer;
    paint(engine, BLUE, 24.0, BAR_100);
    engine.process(DocCommand::AddLayer {
        carrier: Some(l1),
        above: None,
    });
    let m2 = engine.observe().active_layer;
    paint(engine, YELLOW, 24.0, BAR_100);
    (l0, l1, m1, m2)
}

fn source(source: PickSource) -> PickOptions {
    PickOptions {
        source,
        ..PickOptions::default()
    }
}

/// The group source samples the interior of the layer's group — its siblings and
/// the carrier's own content — and nothing outside it: paint that only another
/// part of the document holds answers `None`, the fence the source is for. With
/// `below`, members above the layer are cut, as though switched off.
#[test]
fn the_group_source_samples_the_group_and_nothing_outside_it() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let (_l0, _l1, m1, _m2) = scoped_doc(&mut engine);
    let whole = |layer| {
        source(PickSource::Group {
            layer,
            below: false,
        })
    };

    // 0.1 rather than 0.05 where one bar occludes another: a single stroke pass
    // does not reach full coverage (the slab law, §6.1), so a few percent of the
    // layer beneath bleeds through — still a channel away from the blue it hides.
    assert_near(
        pick(&mut engine, Vec2::new(0.0, 100.0), whole(m1)),
        YELLOW,
        0.1,
        "the whole interior shows the member over m1",
    );
    assert_near(
        pick(&mut engine, Vec2::new(0.0, 200.0), whole(m1)),
        GREEN,
        0.05,
        "the carrier's own content is part of the group",
    );
    assert_eq!(
        pick(&mut engine, Vec2::ZERO, whole(m1)),
        None,
        "paint outside the group is behind the fence",
    );
    assert_near(
        pick(&mut engine, Vec2::ZERO, PickOptions::default()),
        RED,
        0.05,
        "…and it is the fence that hid it, not the paint being absent",
    );
    assert_near(
        pick(
            &mut engine,
            Vec2::new(0.0, 100.0),
            source(PickSource::Group {
                layer: m1,
                below: true,
            }),
        ),
        BLUE,
        0.05,
        "`below` cuts the member above m1, uncovering m1's own paint",
    );
}

/// A root layer's group is the root stack, so the group source there is the whole
/// document — and still no substrate: the root "group" is paint, not canvas.
#[test]
fn a_root_layer_reads_the_root_stack_as_its_group() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let (l0, _l1, _m1, _m2) = scoped_doc(&mut engine);
    let whole = source(PickSource::Group {
        layer: l0,
        below: false,
    });

    // 0.1 for the occlusion bleed, as in the group test above.
    assert_near(
        pick(&mut engine, Vec2::new(0.0, 100.0), whole),
        YELLOW,
        0.1,
        "a root anchor sees the whole document",
    );
    assert_eq!(
        pick(&mut engine, Vec2::new(0.0, -200.0), whole),
        None,
        "and bare canvas still answers nothing",
    );
}

/// The below source is the document with everything above the layer switched
/// off, over the substrate: layers beneath the ancestor chain answer, members
/// above the layer do not, and bare canvas answers with the substrate.
#[test]
fn the_below_source_cuts_the_stack_above_the_layer() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    const SUBSTRATE: [f32; 3] = [0.2, 0.55, 0.35];
    engine.process(DocCommand::SetSubstrateColor(Srgb::new(SUBSTRATE)));
    let (l0, _l1, m1, _m2) = scoped_doc(&mut engine);
    let below = |layer| source(PickSource::Below(layer));

    assert_near(
        pick(&mut engine, Vec2::new(0.0, 100.0), below(m1)),
        BLUE,
        0.05,
        "the member above m1 is switched off",
    );
    assert_near(
        pick(&mut engine, Vec2::ZERO, below(m1)),
        RED,
        0.05,
        "root layers beneath m1's carrier still answer",
    );
    assert_near(
        pick(&mut engine, Vec2::new(0.0, 200.0), below(m1)),
        GREEN,
        0.05,
        "the carrier's own content sits beneath its members (§14.2)",
    );
    assert_near(
        pick(&mut engine, Vec2::new(0.0, -200.0), below(m1)),
        SUBSTRATE,
        0.02,
        "the substrate rides this source: bare canvas is the canvas color",
    );
    assert_near(
        pick(&mut engine, Vec2::new(0.0, 100.0), below(l0)),
        SUBSTRATE,
        0.02,
        "below the bottom layer, everything else is switched off",
    );
    assert_near(
        pick(&mut engine, Vec2::ZERO, below(l0)),
        RED,
        0.05,
        "…while the layer itself still answers",
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

// ---------------------------------------------------------------------------
// The layer hit test (§16.11)
// ---------------------------------------------------------------------------

/// A short vertical stroke through the origin, [`BAR`]'s partner: the two cross
/// at the origin and are alone everywhere else, which is what lets one point ask
/// "which is on top" and two others ask "did you find the right one at all".
const UP: &[Vec2] = &[Vec2::new(0.0, -40.0), Vec2::new(0.0, 40.0)];

/// The crossing. Both bars cover it, so the answer here is about **order**.
const BOTH: Vec2 = Vec2::ZERO;
/// On the horizontal bar and clear of the vertical one (whose half-width is 24).
const ONLY_BAR: Vec2 = Vec2::new(-32.0, 0.0);
/// On the vertical bar and clear of the horizontal one.
const ONLY_UP: Vec2 = Vec2::new(0.0, -32.0);
/// Far off both: nothing is painted here on any layer.
const NEITHER: Vec2 = Vec2::new(-200.0, -200.0);

fn hit(engine: &mut Engine, at: Vec2) -> Option<LayerId> {
    pollster::block_on(engine.pick_layer(at))
}

/// Red across the root layer, blue up a layer above it. Returns the upper one.
fn crossed(engine: &mut Engine) -> LayerId {
    paint(engine, RED, 24.0, BAR);
    engine.process(DocCommand::AddLayer {
        carrier: None,
        above: None,
    });
    let top = engine.observe().active_layer;
    paint(engine, BLUE, 24.0, UP);
    top
}

/// The headline: the answer is the **last layer drawn** that has paint there, and
/// on bare canvas there is no answer at all.
///
/// The three points are three different claims, and only the first is about
/// stacking: the other two are the ones that would still pass if the hit test
/// answered with the active layer, or with the topmost layer full stop.
#[test]
fn the_topmost_painted_layer_is_the_one_under_the_pointer() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let top = crossed(&mut engine);

    assert_eq!(hit(&mut engine, BOTH), Some(top), "the upper bar wins");
    assert_eq!(hit(&mut engine, ONLY_BAR), Some(ROOT), "only red is here");
    assert_eq!(hit(&mut engine, ONLY_UP), Some(top), "only blue is here");
    assert_eq!(hit(&mut engine, NEITHER), None, "bare canvas holds nothing");
}

/// A hidden layer is not under the pointer, because it is not on the screen —
/// the press has to find whatever the eye finds.
#[test]
fn a_hidden_layer_is_not_under_the_pointer() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let top = crossed(&mut engine);
    engine.process(DocCommand::SetLayerVisible(top, false));

    assert_eq!(hit(&mut engine, BOTH), Some(ROOT), "red shows through now");
    assert_eq!(hit(&mut engine, ONLY_UP), None, "and blue is nowhere");
}

/// Opacity is not visibility, and the hit test has to keep the difference.
///
/// Zero is a layer switched off and answers like a hidden one; every setting
/// above it is a layer that is *there*, turned down, and must stay grabbable —
/// paint you can see and cannot pick up is the failure this pins.
#[test]
fn a_faded_layer_is_still_there_and_a_zeroed_one_is_not() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let top = crossed(&mut engine);

    engine.process(DocCommand::SetLayerOpacity(top, 0.1));
    assert_eq!(hit(&mut engine, BOTH), Some(top), "faint, but it is there");

    engine.process(DocCommand::SetLayerOpacity(top, 0.0));
    assert_eq!(hit(&mut engine, BOTH), Some(ROOT), "switched off is gone");
}

/// Hiding a **group** hides its members, and the hit test has to follow the whole
/// subtree down rather than ask each layer about itself.
///
/// The one claim that cannot be made by asking the compositor for a single
/// layer's draw list: that list answers for the layer named, which is right when
/// the question is "what color is this layer's paint" (§18.0.2) and wrong when it
/// is "what can I point at".
#[test]
fn a_hidden_group_takes_its_members_with_it() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let top = crossed(&mut engine);
    // The blue layer is now carried by the red one — a group whose base is paint.
    engine.process(DocCommand::MoveLayer {
        id: top,
        carrier: Some(ROOT),
        at: Place::Top,
    });
    assert_eq!(hit(&mut engine, BOTH), Some(top), "grouped, still on top");

    engine.process(DocCommand::SetLayerVisible(ROOT, false));
    assert_eq!(hit(&mut engine, BOTH), None, "the carrier took it down");
    assert_eq!(hit(&mut engine, ONLY_UP), None, "everywhere, not just here");
}

/// A stroke's feathered rim is paint, and it is not what the hand was pointing
/// at — the threshold sits where a texel starts to read as covered.
///
/// Asked just outside the tip rather than at some arbitrary distance: a bar of
/// half-width 24 has its coverage ramp inside that, so a point at 32 is past the
/// paint entirely while the tile it lands in is the stroke's own — which is
/// exactly the case a tile-granular hit test would get wrong.
#[test]
fn the_pointer_has_to_be_on_the_paint_not_merely_near_it() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint(&mut engine, RED, 24.0, BAR);

    assert_eq!(hit(&mut engine, Vec2::new(0.0, 0.0)), Some(ROOT));
    assert_eq!(hit(&mut engine, Vec2::new(0.0, 32.0)), None, "past the rim");
}
