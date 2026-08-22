//! The **reference light** invariant (§6.3).
//!
//! `Neutral` is what the whole test suite now paints under ([`engine_or_skip`]), which
//! makes this file the one that says *why that is worth anything*: every other test
//! reads a byte back and holds it against an expectation, and this is where the claim
//! that the byte means what it says is checked.
//!
//! Under the procedural `Neutral` environment — whose exposure is 1.0 — with the
//! relief flattened, the media pass is supposed to be an identity: what you painted is what
//! you see. That is the whole claim the `Neutral` environment makes — it is the light
//! you switch to in order to judge a color rather than enjoy it — and it is a claim
//! about the *entire* tail of the pipeline, so nothing here reaches into it. Each test
//! paints (or grounds) a known sRGB color, renders, and reads the byte back.
//!
//! The pass is not, and cannot be, exactly the identity: paint is a dielectric, so a
//! flat patch still carries the environment's ambient sheen, and the diffuse it leaves
//! behind is `1 - spec_energy` of the albedo — about 1.6%. [`TOL`] is that residue and
//! nothing more. It is deliberately tight: it is what stops the media pass from
//! drifting back into being a *look* applied to every color on its way to the screen.

mod common;

use common::*;
use stark_engine::MediaParams;
use stark_engine::command::{DocCommand, ViewCommand};
use stark_model::Srgb;
use stark_model::document::{BrushParams, BrushShape};
use stark_model::geom::Vec2;

/// The reference configuration: no relief to tilt a normal, no weave, no gloss —
/// every knob that could shape the light turned off, so what is left is only the
/// lighting model itself. Exposure is not among them: it rides with the environment,
/// and these tests run on `Neutral`, whose exposure is 1.0 for exactly this reason.
const REFERENCE: MediaParams = MediaParams {
    height_strength: 0.0,
    specular: 0.0,
    surface_strength: 0.0,
};

/// Largest byte error the reference light is allowed. Two levels out of 255 — the
/// dielectric sheen described above, which peaks at white.
const TOL: i32 = 2;

/// Colors to check: the greyscale ramp (where a tonemap's black point and shoulder
/// do their damage) and a few saturated ones (where a hue-shifting one does).
const PROBES: &[[f32; 3]] = &[
    [0.0, 0.0, 0.0],
    [0.25, 0.25, 0.25],
    [0.5, 0.5, 0.5],
    [0.75, 0.75, 0.75],
    [1.0, 1.0, 1.0],
    [0.8, 0.2, 0.2],
    [0.2, 0.5, 0.9],
    [0.95, 0.85, 0.1],
];

/// The rendered pixel at the centre of the viewport.
fn center_rgb(engine: &mut stark_engine::Engine) -> [u8; 3] {
    let img = engine.render_to_image();
    let i = ((img.height / 2) * img.width + img.width / 2) as usize * 4;
    [img.pixels[i], img.pixels[i + 1], img.pixels[i + 2]]
}

/// `want` is straight sRGB in `[0,1]`, which is exactly what the render target stores,
/// so the expected byte is a plain rescale — no encode, no color-space conversion.
/// Anything the media pass does to the color shows up here as a difference.
fn assert_reproduces(what: &str, want: [f32; 3], got: [u8; 3]) {
    let expect = want.map(|c| (c * 255.0).round() as i32);
    let delta = [
        got[0] as i32 - expect[0],
        got[1] as i32 - expect[1],
        got[2] as i32 - expect[2],
    ];
    assert!(
        delta.iter().all(|d| d.abs() <= TOL),
        "{what} {want:?}: rendered {got:?}, expected {expect:?} (off by {delta:?}, tol {TOL})"
    );
}

/// The substrate on its own, with no paint over it: the lighting tail in isolation —
/// environment lookup, BRDF, tonemap, sRGB encode — with nothing from the stroke or
/// compositing machinery in front of it. If this drifts, the light is at fault.
#[test]
fn reference_light_reproduces_the_substrate() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.process(ViewCommand::SetMediaParams(REFERENCE));
    for &c in PROBES {
        engine.process(DocCommand::SetBackground(Srgb::new(c)));
        let got = center_rgb(&mut engine);
        assert_reproduces("substrate", c, got);
    }
}

/// The same claim for actual paint, which is the one a painter cares about: lay an
/// opaque patch and it must come back its own color.
///
/// Each probe is painted twice, on black and on white. Visible alpha is
/// `1 - exp(-opacity · thickness)` — it approaches 1 without reaching it — so a patch
/// that is merely *nearly* opaque still carries a trace of what it was laid on, and
/// the two grounds are far enough apart that such a trace breaks one of them. Running
/// both is what makes this a test of the light rather than of the ground.
#[test]
fn reference_light_reproduces_opaque_paint() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.process(ViewCommand::SetMediaParams(REFERENCE));

    // Two passes: one is already opaque to within a byte, the second puts the visible
    // alpha far enough into the exponential's tail that the ground cannot reach the
    // surface at all — which is the property the two grounds below check.
    const PASSES: usize = 2;

    for &c in PROBES {
        for ground in [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]] {
            engine.process(DocCommand::SetBackground(Srgb::new(ground)));
            let brush = BrushParams {
                color: [c[0], c[1], c[2], 1.0],
                radius: 48.0,
                shape: BrushShape::Round { hardness: 0.95 },
                drain: 0.0,
                ..BrushParams::default()
            };
            for _ in 0..PASSES {
                stroke_with(
                    &mut engine,
                    brush,
                    &[Vec2::new(-40.0, 0.0), Vec2::new(40.0, 0.0)],
                );
            }
            let got = center_rgb(&mut engine);
            assert_reproduces(&format!("paint on {ground:?}"), c, got);
            // Back to bare canvas — one undo per committed stroke, so the probes
            // cannot tint one another.
            for _ in 0..PASSES {
                engine.process(DocCommand::Undo);
            }
        }
    }
}
