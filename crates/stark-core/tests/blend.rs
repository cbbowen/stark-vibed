//! Per-layer blend modes (MISSING_FEATURES.md §0.4): the light-combining family
//! and the compositor's per-layer isolation that makes it possible.
//!
//! The interesting assertions are not "it looks different" — they are the algebraic
//! properties the modes were derived to have. Each mode is ordinary addition
//! conjugated by a tone curve, `f(a,b) = T(T⁻¹(a) + T⁻¹(b))`, so black is an
//! identity and the operation is associative and commutative. Those are what a
//! painter actually feels: a glow layer over nothing changes nothing, and reordering
//! a stack of them changes nothing either.

mod common;

use common::*;
use stark_core::colorspace::ColorSpaceId;
use stark_core::command::DocCommand;
use stark_core::document::{BlendMode, LayerId};
use stark_core::geom::Vec2;
use stark_core::{Engine, RgbaImage};

const ROOT: LayerId = LayerId(0);
const TOP: LayerId = LayerId(1);

/// Two saturated lights that overlap in the middle of the canvas. Warm and cool so
/// the overlap is unmistakably a *combination* rather than either one of them.
const WARM: [f32; 4] = [0.90, 0.35, 0.10, 1.0];
const COOL: [f32; 4] = [0.10, 0.30, 0.85, 1.0];

const H_STROKE: &[Vec2] = &[Vec2::new(-60.0, 0.0), Vec2::new(60.0, 0.0)];
const V_STROKE: &[Vec2] = &[Vec2::new(0.0, -60.0), Vec2::new(0.0, 60.0)];

fn center(img: &RgbaImage) -> [u8; 4] {
    img.pixel(img.width / 2, img.height / 2)
}

/// Perceived brightness, for "did combining light make it brighter?" — the one
/// question every mode here has to answer yes to.
fn luma(c: [u8; 4]) -> f64 {
    0.2126 * c[0] as f64 + 0.7152 * c[1] as f64 + 0.0722 * c[2] as f64
}

/// Add a layer above the current one and paint `color` along `points` on it.
fn layer_with(engine: &mut Engine, color: [f32; 4], points: &[Vec2]) {
    engine.process(DocCommand::AddLayer { above: None });
    paint(engine, color, 44.0, points);
}

/// Warm on the root, cool on a layer above it, crossing at the canvas origin.
fn crossed(engine: &mut Engine) {
    paint(engine, WARM, 44.0, H_STROKE);
    layer_with(engine, COOL, V_STROKE);
}

// ---------------------------------------------------------------------------
// Identity: black is the neutral element, so a mode over nothing is a no-op.
// ---------------------------------------------------------------------------

/// A blend layer over an **empty** stack must render exactly as `Normal` does.
///
/// This is the identity law, and it is also the whole isolation path under test:
/// the layer is composited alone into the isolation targets, run through the blend
/// pass against a cleared accumulator, and ping-ponged back into the caller's
/// targets. Any error in that plumbing — a stale accumulator, a parity mistake that
/// leaves the result in the scratch pair, a clear that never happened — shows up
/// here as a difference from a path that does none of it.
///
/// Exact, not approximate: with no backdrop the blend interpolates to the source
/// channels in the *working* space, so the colour never makes the trip out to light
/// and back and there is nothing for a conversion to round off.
#[test]
fn blend_over_nothing_is_normal() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    paint(&mut engine, WARM, 44.0, H_STROKE);
    let normal = engine.render_to_image();

    for mode in [BlendMode::Reinhard, BlendMode::Drago] {
        engine.process(DocCommand::SetLayerBlend(ROOT, mode));
        let blended = engine.render_to_image();
        let (frac, worst) = diff_fraction(&normal, &blended);
        assert_eq!(
            (frac, worst),
            (0.0, 0),
            "{mode:?} over an empty stack must be the identity"
        );
    }
}

/// The same, one layer up: an empty layer set to a blend mode contributes nothing.
/// It is dropped before it ever reaches the compositor, so the stack below it is
/// untouched rather than round-tripped through a pass that computes the identity.
#[test]
fn empty_blend_layer_contributes_nothing() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    paint(&mut engine, WARM, 44.0, H_STROKE);
    let before = engine.render_to_image();

    engine.process(DocCommand::AddLayer { above: None });
    engine.process(DocCommand::SetLayerBlend(TOP, BlendMode::Reinhard));
    let after = engine.render_to_image();

    let (frac, worst) = diff_fraction(&before, &after);
    assert_eq!(
        (frac, worst),
        (0.0, 0),
        "an empty glow layer is not a layer"
    );
}

// ---------------------------------------------------------------------------
// The modes combine light.
// ---------------------------------------------------------------------------

/// Where two lights overlap, a light-combining mode is brighter than either of them
/// alone — and brighter than `Normal`, which simply hides the one underneath.
#[test]
fn light_modes_brighten_the_overlap() {
    for mode in [BlendMode::Reinhard, BlendMode::Drago] {
        let Some(mut engine) = engine_or_skip_blue() else {
            return;
        };
        crossed(&mut engine);

        let normal = luma(center(&engine.render_to_image()));
        engine.process(DocCommand::SetLayerBlend(TOP, mode));
        let blended = luma(center(&engine.render_to_image()));

        // The top layer alone, for "brighter than either input".
        engine.process(DocCommand::SetLayerVisible(ROOT, false));
        let top_only = luma(center(&engine.render_to_image()));
        engine.process(DocCommand::SetLayerVisible(ROOT, true));
        engine.process(DocCommand::SetLayerVisible(TOP, false));
        let bottom_only = luma(center(&engine.render_to_image()));

        assert!(
            blended > normal + 4.0,
            "{mode:?}: combining light must beat covering it ({blended:.1} vs {normal:.1})"
        );
        assert!(
            blended > top_only.max(bottom_only) + 4.0,
            "{mode:?}: the overlap must be brighter than either light alone \
             ({blended:.1} vs {top_only:.1}/{bottom_only:.1})"
        );
    }
}

/// Outside the overlap nothing changes: a mode only has an opinion where there is
/// something underneath to have an opinion about. Sampled on the arm of the top
/// layer's stroke that hangs off the end of the bottom one.
#[test]
fn blend_only_acts_where_the_layers_meet() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    crossed(&mut engine);
    let normal = engine.render_to_image();

    engine.process(DocCommand::SetLayerBlend(TOP, BlendMode::Drago));
    let blended = engine.render_to_image();

    // The vertical stroke runs to y = ±60 canvas px; the horizontal one is 44 px
    // wide, so 50 px above centre is on the cool stroke and clear of the warm one.
    let (x, y) = (normal.width / 2, normal.height / 2 - 50);
    assert_eq!(
        normal.pixel(x, y),
        blended.pixel(x, y),
        "away from the backdrop a glow layer is just the layer"
    );
}

// ---------------------------------------------------------------------------
// The laws the derivation buys.
// ---------------------------------------------------------------------------

/// **Associativity and commutativity.** Three overlapping glow layers give the same
/// picture whichever order they are stacked in, because the mode is a conjugation of
/// `+`. This is the property that makes a blend mode safe to build a painting on:
/// reordering the stack is not supposed to be a colour decision.
///
/// The law is about the *colour* algebra, so the assertion is made where the colour
/// algebra is the only thing running: an interior patch every one of the three
/// strokes covers at full opacity. Away from there, at a stroke's antialiased edge,
/// stacking order genuinely does matter — `(1 − αb)·Cs` weighs the top layer's own
/// colour by how much backdrop it found, so which layer is on top is a real
/// difference. That is Porter-Duff, not the blend mode; `Normal` layers have exactly
/// the same property, and no derivation could remove it.
#[test]
fn glow_stacking_is_order_independent() {
    // Three strokes through the origin at different angles: the middle of the canvas
    // is inside all three, well clear of any edge.
    let paths: [&[Vec2]; 3] = [
        &[Vec2::new(-70.0, 0.0), Vec2::new(70.0, 0.0)],
        &[Vec2::new(-50.0, -50.0), Vec2::new(50.0, 50.0)],
        &[Vec2::new(0.0, -70.0), Vec2::new(0.0, 70.0)],
    ];
    let colors = [WARM, COOL, [0.20, 0.80, 0.35, 1.0]];

    let render = |order: [usize; 3]| -> Option<RgbaImage> {
        let mut engine = engine_or_skip_blue()?;
        for (n, &i) in order.iter().enumerate() {
            if n > 0 {
                engine.process(DocCommand::AddLayer { above: None });
            }
            paint(&mut engine, colors[i], 44.0, paths[i]);
            let id = engine.observe().active_layer;
            engine.process(DocCommand::SetLayerBlend(id, BlendMode::Reinhard));
        }
        Some(engine.render_to_image())
    };

    let Some(a) = render([0, 1, 2]) else { return };
    let b = render([2, 0, 1]).expect("second engine");
    let c = render([1, 2, 0]).expect("third engine");

    for (name, other) in [("2,0,1", &b), ("1,2,0", &c)] {
        // 2, not 0: associativity is exact in ℝ and merely very close in the
        // half-float the composite targets carry between passes.
        let worst = max_diff_in_box(&a, other, 12);
        assert!(
            worst <= 2,
            "reordering glow layers ({name}) moved the fully-covered interior by {worst}"
        );
    }
}

/// The worst per-channel difference inside a centred `2·half + 1` square.
fn max_diff_in_box(a: &RgbaImage, b: &RgbaImage, half: u32) -> u8 {
    let (cx, cy) = (a.width / 2, a.height / 2);
    let mut worst = 0u8;
    for y in cy - half..=cy + half {
        for x in cx - half..=cx + half {
            let (pa, pb) = (a.pixel(x, y), b.pixel(x, y));
            for i in 0..4 {
                worst = worst.max((pa[i] as i32 - pb[i] as i32).unsigned_abs() as u8);
            }
        }
    }
    worst
}

/// **Black is the identity — the round trip under test.** `f(a, 0) = a` for both
/// curves, so a light layer over *black paint* must come back out its own colour.
///
/// Unlike [`blend_over_nothing_is_normal`], this one does not take the shortcut:
/// with an opaque backdrop the interpolation weight is 1, so the colour genuinely
/// makes the trip out to normalized XYZ and back. That makes it the check on the
/// conversions themselves — and in a Mixbox document, on `mixbox_lut.wesl`, the one
/// place the engine inverts the pigment polynomial on the GPU. A LUT with its axes
/// or its rows the wrong way round would still be "brighter at the overlap"; it
/// would not survive this.
#[test]
fn black_is_the_identity_through_the_round_trip() {
    // A near-neutral and a saturated colour, because the two spaces cost very
    // different amounts on each.
    const MUTED: [f32; 4] = [0.55, 0.52, 0.48, 1.0];
    let cases = [
        // Oklab's round trip is a pair of matrices and a signed cube root: exact to
        // within the half-float the composite targets carry, whatever the colour.
        (ColorSpaceId::Oklab, MUTED, 2u8),
        (ColorSpaceId::Oklab, WARM, 2u8),
        // Mixbox's is the trained polynomial and its inverse table. A near-neutral is
        // a mixture pigment hits almost dead on, so it has to come back almost exact —
        // this is the case that would catch a LUT read the wrong way round, which
        // would be wrong by a hundred levels rather than a handful.
        (ColorSpaceId::Mixbox, MUTED, 8u8),
        // A saturated orange is not. Three concentrations cannot express it exactly;
        // Mixbox carries the difference in a *residual* that this engine drops so the
        // channels fit alongside coverage (DESIGN.md §6.7), and the CPU conversion
        // loses the same ~12 levels of blue on this colour before any of it reaches
        // the GPU. So the bound here is not slack for a suspect shader — it is the
        // documented cost of saying "this much light" in pigment, and it is why a
        // `Radiance` layer in a Mixbox document is a different proposition from one
        // in an Oklab document.
        (ColorSpaceId::Mixbox, WARM, 40u8),
    ];
    for (space, color, tol) in cases {
        for mode in [BlendMode::Reinhard, BlendMode::Drago] {
            let Some(mut engine) = engine_or_skip_with(space) else {
                return;
            };
            // A wide black ground with the colour laid over the middle of it.
            paint(&mut engine, [0.0, 0.0, 0.0, 1.0], 70.0, H_STROKE);
            layer_with(&mut engine, color, V_STROKE);
            let over_black = center(&engine.render_to_image());

            engine.process(DocCommand::SetLayerBlend(TOP, mode));
            let blended = center(&engine.render_to_image());

            let worst = (0..3)
                .map(|i| (over_black[i] as i32 - blended[i] as i32).unsigned_abs())
                .max()
                .unwrap_or(0);
            assert!(
                worst <= tol as u32,
                "{space:?}/{mode:?} on {color:?}: light over black must be the light \
                 itself, but {blended:?} differs from {over_black:?} by {worst}"
            );
        }
    }
}

/// **Glow cannot blow out.** Reinhard's curve is asymptotic, so no amount of light
/// reaches white — which is the difference from Screen, and the reason to reach for
/// it. Five stacked layers of near-white on a light ground must still leave headroom
/// somewhere; the same five under `Drago`, whose log curve has no asymptote, are
/// allowed to (and do) push into the roll-off.
#[test]
fn glow_leaves_headroom_where_radiance_does_not() {
    let bright = [0.80, 0.80, 0.80, 1.0];
    let stack = |mode: BlendMode| -> Option<f64> {
        let mut engine = engine_or_skip()?;
        for n in 0..5 {
            if n > 0 {
                engine.process(DocCommand::AddLayer { above: None });
            }
            paint(&mut engine, bright, 50.0, H_STROKE);
            let id = engine.observe().active_layer;
            engine.process(DocCommand::SetLayerBlend(id, mode));
        }
        Some(luma(center(&engine.render_to_image())))
    };

    let Some(glow) = stack(BlendMode::Reinhard) else {
        return;
    };
    let radiance = stack(BlendMode::Drago).expect("second engine");
    assert!(
        glow < radiance,
        "five layers deep, Radiance must have overtaken Glow ({glow:.1} vs {radiance:.1})"
    );
    assert!(
        glow < 254.0,
        "Glow reached the clip at {glow:.1}; its curve is supposed to make that impossible"
    );
}

// ---------------------------------------------------------------------------
// The pigment space.
// ---------------------------------------------------------------------------

/// The Mixbox path runs the same algebra, but the trip back from light is a LUT
/// lookup rather than a matrix (`mixbox_lut.wesl`) — a different shader, a different
/// binding, and the only place in the engine that inverts the pigment polynomial on
/// the GPU. Same two claims: identity over nothing, brighter at the overlap.
#[test]
fn blend_works_in_the_pigment_space() {
    let Some(mut engine) = engine_or_skip_with(ColorSpaceId::Mixbox) else {
        return;
    };
    paint(&mut engine, WARM, 44.0, H_STROKE);
    let normal = engine.render_to_image();
    engine.process(DocCommand::SetLayerBlend(ROOT, BlendMode::Reinhard));
    let (frac, worst) = diff_fraction(&normal, &engine.render_to_image());
    assert_eq!(
        (frac, worst),
        (0.0, 0),
        "a glow layer over nothing is the identity in pigment too"
    );

    engine.process(DocCommand::SetLayerBlend(ROOT, BlendMode::Normal));
    layer_with(&mut engine, COOL, V_STROKE);
    let covered = luma(center(&engine.render_to_image()));
    engine.process(DocCommand::SetLayerBlend(TOP, BlendMode::Reinhard));
    let combined = luma(center(&engine.render_to_image()));
    assert!(
        combined > covered + 4.0,
        "pigment glow must still combine light ({combined:.1} vs {covered:.1})"
    );
}

// ---------------------------------------------------------------------------
// Goldens.
// ---------------------------------------------------------------------------

#[test]
fn golden_glow_layer() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    crossed(&mut engine);
    engine.process(DocCommand::SetLayerBlend(TOP, BlendMode::Reinhard));
    assert_golden("blend_glow", &engine.render_to_image(), 6);
}

#[test]
fn golden_radiance_layer() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    crossed(&mut engine);
    engine.process(DocCommand::SetLayerBlend(TOP, BlendMode::Drago));
    assert_golden("blend_radiance", &engine.render_to_image(), 6);
}
