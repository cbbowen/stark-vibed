//! Tile-apron seam regression (§6.4).
//!
//! Under magnification the compositor samples each tile bilinearly. Tiles are
//! separate textures, so without an apron the filter clamps at a tile's edge
//! instead of reaching into the neighbor — a discontinuity the media pass then
//! amplifies into a lighting ridge along every tile boundary. The apron carries
//! a band of the neighbor's content so the edge taps interpolate correctly.
//!
//! The invariant the apron restores is **translation invariance w.r.t. the tile
//! grid**: the lit canvas must not depend on where the tile boundaries happen to
//! fall. So painting a stroke straddling the 4-tile corner at the origin must
//! render identically to painting the *same* stroke shifted by a half-tile (a
//! non-multiple of TILE_SIZE, so it lands inside one tile's interior), with the
//! view shifted to match. They differ only in tile-grid alignment; a seam would
//! appear in the corner case and break the match.

mod common;

use common::*;
use stark_assetid::Picture;
use stark_engine::command::Tool;
use stark_engine::command::{DocCommand, GestureCommand, InputSample, ViewCommand};
use stark_engine::path::DEFAULT_TOLERANCE;
use stark_engine::{MediaParams, RgbaImage};
use stark_model::document::BrushDynamics;
use stark_model::geom::{IVec2, Vec2};

const RED: [f32; 4] = [0.85, 0.15, 0.1, 1.0];

/// Render a diagonal, height-bearing stroke offset by `shift` canvas px, viewed
/// at 2× zoom centered on `shift` so the on-screen result is independent of
/// `shift` — except for how the stroke lands on the tile grid.
fn render_shifted(shift: Vec2) -> RgbaImage {
    let mut engine = engine_or_skip_blue().expect("engine (caller checked adapter)");

    // Exaggerate the impasto relief so any clamped-normal seam is unmistakable. The
    // image-based-lighting specular reflection is *very* normal-sensitive (a sharp
    // env lookup), so it's kept moderate: a gross seam (a real normal discontinuity)
    // still jumps tens of levels, but the apron's sub-pixel compositing residual —
    // which the sharp reflection would otherwise amplify past tolerance — stays
    // small. SubstrateMap relief is turned OFF: the canvas substrate is sampled in canvas
    // space, so it intentionally is *not* tile-grid translation invariant and would
    // mask the apron behavior tested here. Display dither is off by the same
    // argument: it is keyed to the screen pixel (§6.5), so it too would differ
    // between the shifted renders and fog the comparison.
    engine.process(ViewCommand::SetMediaParams(MediaParams {
        height_strength: 2.5,
        specular: 0.3,
        substrate_strength: 0.0,
        dither: false,
    }));

    // Diagonal stroke through the 4-tile corner at `shift` (origin for shift=0).
    // Tooth off for the same reason (it gates deposition by canvas-space substrate),
    // and the deposit jitter with it: its gate is keyed to the canvas texel (§6.2),
    // so it too is deliberately not translation invariant and would differ between
    // the shifted renders.
    let mut b = brush(RED, 28.0);
    b.dynamics.deposit_jitter = 0.0;
    engine.process(ViewCommand::SetBrush(b));
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(shift + Vec2::new(-50.0, -50.0)),
        tolerance: DEFAULT_TOLERANCE,
        rope: 0.0,
    });
    engine.process(GestureCommand::To {
        sample: InputSample::at(shift + Vec2::new(50.0, 50.0)),
    });
    engine.process(GestureCommand::End);

    // Center the view on `shift` (Pan: center -= delta/zoom, at zoom 1), then
    // magnify 2× about the viewport center so the canvas point under the center
    // stays put. Result: identical screen mapping for every `shift`.
    let center_px = Vec2::new(SIZE.width as f32 * 0.5, SIZE.height as f32 * 0.5);
    engine.process(ViewCommand::Pan { delta: -shift });
    engine.process(ViewCommand::Zoom {
        anchor: center_px,
        factor: 2.0,
    });

    engine.render_to_image()
}

#[test]
fn apron_makes_tiles_seamless_under_zoom() {
    if engine_or_skip_blue().is_none() {
        return; // no usable GPU adapter
    }

    // Straddling the origin's 4-tile corner vs. inside tile (0,0)'s interior.
    let corner = render_shifted(Vec2::ZERO);
    let interior = render_shifted(Vec2::new(128.0, 128.0));

    // The apron's compositing is near—but not bit—exact, and image-based lighting
    // (exposure + ACES tonemap) amplifies that sub-pixel residual along the tile
    // seam, so a thin band of boundary pixels differs by ~10 levels. A genuinely
    // *missing* apron is a stark lighting ridge along every boundary — tens of levels
    // over a far larger area — so this threshold still catches what it guards.
    let (frac, worst) = diff_fraction(&corner, &interior);
    assert!(
        worst <= 25 && frac < 0.07,
        "tile seam: corner vs interior render differ by up to {worst} levels \
         on {:.2}% of pixels — the apron is not covering tile boundaries",
        frac * 100.0
    );
}

/// Like `render_shifted`, but what straddles the corner is a **placed image** (§23) —
/// the one tile writer here that is not a render pass.
///
/// It has to meet the same rule, and it meets it in the strongest available form:
/// every texel is computed from its own canvas position on the CPU, so a tile's apron
/// is bit-identical to its neighbour's interior by construction rather than by a pass
/// being careful. This is what says the tile walk really is written that way — an
/// implementation that filled each tile from the image's origin *relative to the tile*
/// would look perfect at one alignment and seam at every boundary.
fn render_shifted_image(shift: Vec2) -> RgbaImage {
    let mut engine = engine_or_skip_blue().expect("engine (caller checked adapter)");
    engine.process(ViewCommand::SetMediaParams(MediaParams {
        height_strength: 2.5,
        specular: 0.3,
        substrate_strength: 0.0,
        dither: false,
    }));

    // A field of varying color and full alpha, big enough to span the 4-tile corner in
    // every direction. Varying rather than flat: a constant image is seamless however
    // wrongly it is addressed.
    const SIDE: u32 = 200;
    let pixels = (0..SIDE * SIDE)
        .flat_map(|i| {
            let (x, y) = (i % SIDE, i / SIDE);
            [
                (x * 5 % 256) as u8,
                (y * 3 % 256) as u8,
                ((x ^ y) % 256) as u8,
                255,
            ]
        })
        .collect();
    let png = Picture {
        width: SIDE,
        height: SIDE,
        pixels,
    }
    .encode()
    .expect("a well-formed field");
    let id = engine.import_picture(&png).expect("import the field");
    engine.process(DocCommand::PlaceImage {
        carrier: None,
        above: None,
        at: IVec2::new(shift.x as i32, shift.y as i32) - IVec2::splat(SIDE as i32 / 2),
        name: None,
        image: id,
    });

    let center_px = Vec2::new(SIZE.width as f32 * 0.5, SIZE.height as f32 * 0.5);
    engine.process(ViewCommand::Pan { delta: -shift });
    engine.process(ViewCommand::Zoom {
        anchor: center_px,
        factor: 2.0,
    });
    engine.render_to_image()
}

#[test]
fn apron_makes_a_placed_image_seamless_under_zoom() {
    if engine_or_skip_blue().is_none() {
        return; // no usable GPU adapter
    }

    let corner = render_shifted_image(Vec2::ZERO);
    let interior = render_shifted_image(Vec2::new(128.0, 128.0));

    // Held far tighter than its two neighbours above, and that is the point of testing
    // this path separately: the tiles are CPU arithmetic on integer canvas positions,
    // so the two runs' *tiles* are bit-identical and only the compositor's sampling
    // stands between them — which is itself a pure function of canvas position. There
    // is no sub-pixel residual to allow for here, so anything but a near-exact match is
    // a real addressing bug rather than lighting amplifying a rounding difference.
    let (frac, worst) = diff_fraction(&corner, &interior);
    assert!(
        worst <= 2 && frac < 0.01,
        "placed-image seam: corner vs interior differ by up to {worst} levels on \
         {:.2}% of pixels — the tile walk is not addressing canvas position",
        frac * 100.0
    );
}

/// Like `render_shifted`, but the height-bearing stroke is a **stamp-loop smudge**
/// (§6.2): lay a red field through the corner, then drag a smearing
/// brush along it. Exercises the region write-back path — the whole-block slice
/// from the shared region must keep aprons bit-identical to neighbour interiors,
/// and the halo composite must give rewritten tiles real neighbour content.
fn render_shifted_smudge(shift: Vec2) -> RgbaImage {
    let mut engine = engine_or_skip_blue().expect("engine (caller checked adapter)");
    engine.process(ViewCommand::SetMediaParams(MediaParams {
        height_strength: 2.5,
        specular: 0.3,
        substrate_strength: 0.0,
        dither: false,
    }));

    // A wide base field along the diagonal, fully containing the smudge's path.
    // Jitter off on both brushes here, by `render_shifted`'s argument: the gate is
    // canvas-anchored (§6.2) and this test compares shifted renders.
    let mut field = brush(RED, 60.0);
    field.dynamics.deposit_jitter = 0.0;
    engine.process(ViewCommand::SetBrush(field));
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(shift + Vec2::new(-60.0, -60.0)),
        tolerance: DEFAULT_TOLERANCE,
        rope: 0.0,
    });
    engine.process(GestureCommand::To {
        sample: InputSample::at(shift + Vec2::new(60.0, 60.0)),
    });
    engine.process(GestureCommand::End);

    // The smudge under test, through the same 4-tile corner.
    let mut smudge = brush(RED, 28.0);
    smudge.dynamics = BrushDynamics {
        flow: 0.0,
        lift: 0.6,
        deposit: 0.5,
        deposit_jitter: 0.0,
        ..Default::default()
    };
    engine.process(ViewCommand::SetBrush(smudge));
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(shift + Vec2::new(-50.0, -50.0)),
        tolerance: DEFAULT_TOLERANCE,
        rope: 0.0,
    });
    engine.process(GestureCommand::To {
        sample: InputSample::at(shift + Vec2::new(50.0, 50.0)),
    });
    engine.process(GestureCommand::End);

    let center_px = Vec2::new(SIZE.width as f32 * 0.5, SIZE.height as f32 * 0.5);
    engine.process(ViewCommand::Pan { delta: -shift });
    engine.process(ViewCommand::Zoom {
        anchor: center_px,
        factor: 2.0,
    });
    engine.render_to_image()
}

#[test]
fn apron_makes_dynamics_writeback_seamless_under_zoom() {
    if engine_or_skip_blue().is_none() {
        return; // no usable GPU adapter
    }

    // Same invariant as above, for the swept-exchange loop's region write-back: a
    // smudge straddling the 4-tile corner must render identically to the same
    // smudge shifted into one tile's interior. A missing halo (or a slice that
    // didn't cover whole blocks) would seam the relief along every tile boundary.
    let corner = render_shifted_smudge(Vec2::ZERO);
    let interior = render_shifted_smudge(Vec2::new(128.0, 128.0));

    // The two runs' *regions* differ in size (the corner stroke spans more tiles),
    // so the pickup's normalized-coordinate bilinear samples (`world / rdim`)
    // round differently at ~1 ulp; through f16 storage and the exaggerated relief
    // lighting that substrates as a broad, imperceptible (≤ a few levels) residual
    // over the smudged area — not a seam. A real missing halo is a *contiguous
    // band* of tens of levels along every boundary, so gate on the significantly-
    // different area instead of the any-difference area.
    let (_, worst) = diff_fraction(&corner, &interior);
    let frac_big = frac_exceeding(&corner, &interior, 12);
    assert!(
        worst <= 25 && frac_big < 0.005,
        "dynamics write-back seam: corner vs interior differ by up to {worst} levels, \
         {:.3}% of pixels by >12 — the region write-back is not covering tile boundaries",
        frac_big * 100.0
    );
}
