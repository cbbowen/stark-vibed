//! The erase pass (§6.12).
//!
//! The claims worth pinning are the ones the pass exists for: the dial is quoted
//! in what the eye sees (`erase = 0.5` under a saturated stroke leaves half the
//! visible opacity — held here as *equality with the half-covered fill*, which is
//! the same statement with no media arithmetic in it), the opacity is a ceiling
//! a stroke cannot scrub past, separate strokes compound multiplicatively, a full
//! erase is bare canvas again to the bit, and nothing outside the stroke's own
//! extent moves at all. The piecewise-vs-whole and round-trip obligations ride
//! the corpus (`corpus.rs`, case `erase`).

mod common;

use common::*;
use stark_engine::command::DocCommand;
use stark_model::Srgb;
use stark_model::document::{
    BrushEffect, BrushParams, EraseEffect, FillOp, LayerId, ModSource, Modulation, SelectionShape,
};
use stark_model::geom::Vec2;

/// The bed every test erases from: a full-canvas red fill. A fill rather than a
/// stroke, because its coverage is *stated* (`FillOp::opacity`) — which is what
/// lets the headline test compare an erased bed against a bed simply asked for
/// at the target coverage.
const RED: [f32; 3] = [1.0, 0.0, 0.0];

/// An eraser (§6.12): `opacity` is the dial, and its flow the rate — high
/// enough here that one pass saturates the bite to its ceiling over the stroke's
/// core, so a test sampling the core is reading the dial and not a half-built
/// fringe.
fn eraser(opacity: f32, radius: f32) -> BrushParams {
    let mut b = brush([0.0, 0.0, 0.0], radius);
    b.effect = BrushEffect::Erase(EraseEffect {
        opacity,
        flow: 2.5,
        ..EraseEffect::default()
    });
    b.drain = 0.0;
    b
}

/// Fill the whole canvas with `color` at `opacity` — the bed, or the headline
/// test's stated-coverage reference.
fn fill_canvas(engine: &mut stark_engine::Engine, color: [f32; 3], opacity: f32) {
    let layer = engine.observe().active_layer;
    engine.process(DocCommand::Fill {
        layer,
        op: FillOp::new(
            SelectionShape::rect_from_corners(Vec2::new(-128.0, -128.0), Vec2::new(128.0, 128.0)),
            0.0,
            Srgb::new(color),
            opacity,
        ),
    });
}

/// A full-strength eraser returns the canvas to bare — the very pixels it showed
/// before anything was painted — and takes nothing beyond its own extent.
///
/// Exactness is the claim: a full erase inverts the slab law to a height of
/// exactly 0, a zero tile composites as nothing, and the display dither is a
/// function of position (§6.5) — so "erased" and "never painted" are the same
/// image, not merely similar.
#[test]
fn a_full_erase_is_bare_canvas_again() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let bare = engine.render_to_image();
    fill_canvas(&mut engine, RED, 1.0);
    let painted = engine.render_to_image();
    assert!(
        red_dominant(center(&painted)),
        "the bed must read as paint before the claim means anything"
    );

    stroke_with(
        &mut engine,
        eraser(1.0, 24.0),
        &[Vec2::new(-60.0, 0.0), Vec2::new(60.0, 0.0)],
    );
    let after = engine.render_to_image();

    assert_eq!(
        texel(&after, Vec2::ZERO),
        texel(&bare, Vec2::ZERO),
        "a full erase must be bare canvas to the bit at the stroke's core"
    );
    // …and the bed beyond the tip's reach (radius 24, plus the soft edge) is
    // exactly the painted image still: the integrate's untouched branch passes
    // the base through rather than recomputing it.
    for p in [Vec2::new(0.0, 90.0), Vec2::new(-100.0, -80.0)] {
        assert_eq!(
            texel(&after, p),
            texel(&painted, p),
            "paint the stroke never reached moved at {p:?}"
        );
    }
}

/// `erase` is a **ceiling, not a rate**: a stroke scrubbing the same spot five
/// times leaves what one clean pass leaves. The extent accumulates and its
/// coverage saturates at 1, so the removal saturates at the dial (§6.12) —
/// where a per-piece or per-pass law would compound toward bare canvas.
#[test]
fn scrubbing_cannot_pass_the_dial() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    fill_canvas(&mut engine, RED, 1.0);
    let bed = engine.render_to_image();

    // One clean pass across y = −40…
    stroke_with(
        &mut engine,
        eraser(0.5, 20.0),
        &[Vec2::new(-80.0, -40.0), Vec2::new(80.0, -40.0)],
    );
    // …and one stroke worrying the same band at y = 40, five crossings.
    let scrub: Vec<Vec2> = (0..5)
        .flat_map(|i| {
            let x = if i % 2 == 0 { -80.0 } else { 80.0 };
            [Vec2::new(x, 40.0 + i as f32 * 0.25)]
        })
        .collect();
    stroke_with(&mut engine, eraser(0.5, 20.0), &scrub);
    let after = engine.render_to_image();

    let once = texel(&after, Vec2::new(0.0, -40.0));
    let scrubbed = texel(&after, Vec2::new(0.0, 40.0));
    assert!(
        apart(once, scrubbed) <= 2,
        "a scrubbed half-eraser must land where one pass lands: one pass \
         {once:?}, scrubbed {scrubbed:?}"
    );
    // The pass did something, or the equality above is two untouched texels
    // agreeing with each other.
    assert!(
        apart(once, texel(&bed, Vec2::new(0.0, -40.0))) > 20,
        "the half eraser left the bed unmoved, so this test measured nothing"
    );
}

/// Separate strokes compound multiplicatively on what each finds: two passes at
/// 0.5 are one pass at 0.75, exactly as layered glazes are — `(1−½)² = 1−¾`.
#[test]
fn two_half_erases_are_one_three_quarter_erase() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    fill_canvas(&mut engine, RED, 1.0);

    for _ in 0..2 {
        stroke_with(
            &mut engine,
            eraser(0.5, 20.0),
            &[Vec2::new(-80.0, -40.0), Vec2::new(80.0, -40.0)],
        );
    }
    stroke_with(
        &mut engine,
        eraser(0.75, 20.0),
        &[Vec2::new(-80.0, 40.0), Vec2::new(80.0, 40.0)],
    );
    let after = engine.render_to_image();

    let twice_half = texel(&after, Vec2::new(0.0, -40.0));
    let three_quarter = texel(&after, Vec2::new(0.0, 40.0));
    assert!(
        apart(twice_half, three_quarter) <= 2,
        "two half erases {twice_half:?} must equal one three-quarter erase \
         {three_quarter:?}"
    );
}

/// **The dial is quoted in visible opacity** — the claim the pass exists for
/// (§6.12), stated with no media arithmetic in it: a fully covered bed
/// erased at 0.5 *is* the bed filled at 0.5. Same latent, same per-unit
/// opacity, and a height within `exp(−OPAQUE_MASS)` of the same — so the two
/// documents render the same pixels, lighting included.
///
/// A `lift`-built eraser fails this by construction: it removes a fraction of
/// the *height*, and near-opaque paint shows almost none of that.
#[test]
fn a_half_erase_is_the_half_covered_fill() {
    let Some(mut erased) = engine_or_skip() else {
        return;
    };
    fill_canvas(&mut erased, RED, 1.0);
    stroke_with(
        &mut erased,
        eraser(0.5, 24.0),
        &[Vec2::new(-90.0, 0.0), Vec2::new(90.0, 0.0)],
    );
    let erased_img = erased.render_to_image();

    let mut asked = engine_or_skip().expect("the adapter answered once already");
    fill_canvas(&mut asked, RED, 0.5);
    let asked_img = asked.render_to_image();

    for p in [Vec2::ZERO, Vec2::new(-40.0, 0.0), Vec2::new(40.0, 0.0)] {
        let (e, a) = (texel(&erased_img, p), texel(&asked_img, p));
        assert!(
            apart(e, a) <= 2,
            "at {p:?} the half-erased bed {e:?} must be the half-covered fill              {a:?} — the dial is not landing in the coverage domain"
        );
        let (eh, eo) = paint_at(&erased, LayerId::ROOT, p).expect("the erased bed has a tile here");
        let (ah, ao) = paint_at(&asked, LayerId::ROOT, p).expect("the filled bed has a tile here");
        assert!(
            (eo - ao).abs() <= 0.01,
            "at {p:?} the half-erased bed's per-unit opacity {eo} must be the              half-covered fill's {ao} — the dial left the coverage domain, which a              render at this height cannot show"
        );
        // Loose against the two, because the claim is `exp(−OPAQUE_MASS)` and not
        // equality: what the eraser leaves is a bed thin enough to be transparent at
        // that opacity, not the same slab.
        assert!(
            (eh - ah).abs() <= 0.05 * ah.max(1.0),
            "at {p:?} the half-erased height {eh} is nowhere near the half-covered              fill's {ah}"
        );
    }
}

/// Erasing where there is nothing changes nothing — to the bit. A tile the
/// layer does not have is nothing to erase (§6.12): the pass mints no
/// tiles, so the render after is the render before.
#[test]
fn erasing_bare_canvas_changes_nothing() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let before = engine.render_to_image();
    stroke_with(
        &mut engine,
        eraser(1.0, 30.0),
        &[Vec2::new(-80.0, 0.0), Vec2::new(80.0, 0.0)],
    );
    let after = engine.render_to_image();
    let (frac, worst) = diff_fraction(&before, &after);
    assert_eq!(
        (frac, worst),
        (0.0, 0),
        "an eraser over bare canvas moved pixels"
    );
}

/// A straight leg along `y` from `x0` to `x1` at one `pressure` — the two reports
/// a hand makes of a straight run (`tests/opacity.rs` makes the same
/// construction, for the same reason).
fn leg(x0: f32, x1: f32, y: f32, pressure: f32) -> Vec<(Vec2, f32)> {
    vec![(Vec2::new(x0, y), pressure), (Vec2::new(x1, y), pressure)]
}

/// The eraser's ceiling under the pen (§6.12, `EraseModulations::opacity`):
/// mapped to pressure, a light touch thins the bed where a heavy one clears
/// it — and, the first claim winning (§6.2), a light pass back over a cleared
/// band within one stroke takes nothing more from it.
#[test]
fn an_eraser_under_the_pen_removes_what_the_pen_asks() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    fill_canvas(&mut engine, RED, 1.0);
    let bed = texel(&engine.render_to_image(), Vec2::ZERO);
    let mut e = eraser(1.0, 20.0);
    if let BrushEffect::Erase(e) = &mut e.effect {
        e.modulation.opacity = Some(Modulation::linear(ModSource::Pressure));
    }
    // Heavy along y = −40, light along y = 40.
    stroke_pressed(&mut engine, e, &leg(-80.0, 80.0, -40.0, 1.0));
    stroke_pressed(&mut engine, e, &leg(-80.0, 80.0, 40.0, 0.2));
    let after = engine.render_to_image();
    let heavy = texel(&after, Vec2::new(0.0, -40.0));
    let light = texel(&after, Vec2::new(0.0, 40.0));
    assert!(
        apart(heavy, bed) > 100,
        "a heavy press should clear the bed: {heavy:?}"
    );
    assert!(
        apart(light, bed) < apart(heavy, bed) / 2,
        "a light touch should thin it markedly less: {light:?} against {heavy:?}"
    );
    assert!(apart(light, bed) > 10, "…and is not nothing: {light:?}");

    // One stroke: out heavy, back light over the same band.
    let mut twice = engine_or_skip().expect("the adapter answered once already");
    fill_canvas(&mut twice, RED, 1.0);
    // The pressure drop on an excursion away from the band, for the reason
    // `tests/opacity.rs` gives: the fitted pressure is a smoothed spline, and a
    // step in it at the hairpin ripples back along the heavy leg.
    let excursion: Vec<(Vec2, f32)> = (1..=6)
        .map(|i| (Vec2::new(80.0, 10.0 * i as f32), 1.0))
        .chain((0..=6).map(|i| {
            let t = i as f32 / 6.0;
            (Vec2::new(80.0, 60.0 - 60.0 * t), 1.0 - 0.8 * t)
        }))
        .collect();
    let both: Vec<(Vec2, f32)> = leg(-80.0, 80.0, 0.0, 1.0)
        .into_iter()
        .chain(excursion)
        .chain(leg(80.0, -80.0, 0.25, 0.2))
        .collect();
    stroke_pressed(&mut twice, e, &both);
    let crossed = texel(&twice.render_to_image(), Vec2::ZERO);
    assert!(
        apart(crossed, heavy) <= 2,
        "the cleared band crossed back lightly {crossed:?} must be the cleared band {heavy:?}"
    );
}
