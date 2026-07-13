//! Tile-apron seam regression (DESIGN.md §6.4).
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
use stark_core::command::{InputCommand, InputSample};
use stark_core::document::{BrushDynamics, Tool};
use stark_core::geom::Vec2;
use stark_core::{MediaParams, RgbaImage};

const RED: [f32; 4] = [0.85, 0.15, 0.1, 1.0];

/// Render a diagonal, height-bearing stroke offset by `shift` canvas px, viewed
/// at 2× zoom centered on `shift` so the on-screen result is independent of
/// `shift` — except for how the stroke lands on the tile grid.
fn render_shifted(shift: Vec2) -> RgbaImage {
    let mut engine = engine_or_skip().expect("engine (caller checked adapter)");

    // Exaggerate the impasto relief so any clamped-normal seam is unmistakable. The
    // image-based-lighting specular reflection is *very* normal-sensitive (a sharp
    // env lookup), so it's kept moderate: a gross seam (a real normal discontinuity)
    // still jumps tens of levels, but the apron's sub-pixel compositing residual —
    // which the sharp reflection would otherwise amplify past tolerance — stays
    // small. Surface relief is turned OFF: the canvas weave is sampled in canvas
    // space, so it intentionally is *not* tile-grid translation invariant and would
    // mask the apron behavior tested here.
    engine.set_media_params(MediaParams {
        height_strength: 2.5,
        specular: 0.3,
        surface_strength: 0.0,
        // Like the weave, the normal dither is canvas-position-seeded — deliberately
        // not translation invariant — so it must be off for these shift comparisons.
        normal_dither: 0.0,
        ..MediaParams::default()
    });

    // Diagonal stroke through the 4-tile corner at `shift` (origin for shift=0).
    // Tooth off for the same reason (it gates deposition by canvas-space weave).
    let mut b = brush(RED, 28.0);
    b.tooth = 0.0;
    engine.process(InputCommand::SetBrush(b));
    engine.process(InputCommand::StartStroke {
        tool: Tool::Brush,
        sample: InputSample::at(shift + Vec2::new(-50.0, -50.0)),
    });
    engine.process(InputCommand::StrokeTo {
        sample: InputSample::at(shift + Vec2::new(50.0, 50.0)),
    });
    engine.process(InputCommand::EndStroke);

    // Center the view on `shift` (Pan: center -= delta/zoom, at zoom 1), then
    // magnify 2× about the viewport center so the canvas point under the center
    // stays put. Result: identical screen mapping for every `shift`.
    let center_px = Vec2::new(SIZE.width as f32 * 0.5, SIZE.height as f32 * 0.5);
    engine.process(InputCommand::Pan { delta: -shift });
    engine.process(InputCommand::Zoom {
        anchor: center_px,
        factor: 2.0,
    });

    engine.render_to_image(BG)
}

#[test]
fn apron_makes_tiles_seamless_under_zoom() {
    if engine_or_skip().is_none() {
        return; // no usable GPU adapter
    }

    // Straddling the origin's 4-tile corner vs. inside tile (0,0)'s interior.
    let corner = render_shifted(Vec2::ZERO);
    let interior = render_shifted(Vec2::new(128.0, 128.0));

    // The apron's compositing is near—but not bit—exact, and image-based lighting
    // (exposure + ACES tonemap) amplifies that sub-pixel residual along the tile
    // seam a little more than the old directional model did, so a thin band of
    // boundary pixels differs by ~10 levels. A genuinely *missing* apron is a stark
    // lighting ridge along every boundary — tens of levels over a far larger area —
    // so this threshold still catches the regression it guards.
    let (frac, worst) = diff_fraction(&corner, &interior);
    assert!(
        worst <= 25 && frac < 0.07,
        "tile seam: corner vs interior render differ by up to {worst} levels \
         on {:.2}% of pixels — the apron is not covering tile boundaries",
        frac * 100.0
    );
}

/// Like `render_shifted`, but the height-bearing stroke is a **stamp-loop smudge**
/// (DESIGN.md §6.2): lay a red field through the corner, then drag a smearing
/// brush along it. Exercises the region write-back path — the whole-block slice
/// from the shared region must keep aprons bit-identical to neighbour interiors,
/// and the halo composite must give rewritten tiles real neighbour content.
fn render_shifted_smudge(shift: Vec2) -> RgbaImage {
    let mut engine = engine_or_skip().expect("engine (caller checked adapter)");
    engine.set_media_params(MediaParams {
        height_strength: 2.5,
        specular: 0.3,
        surface_strength: 0.0,
        // Canvas-position-seeded (deliberately not translation invariant), so off
        // for these shift comparisons — like the weave.
        normal_dither: 0.0,
        ..MediaParams::default()
    });

    // A wide base field along the diagonal, fully containing the smudge's path.
    let mut field = brush(RED, 60.0);
    field.tooth = 0.0;
    engine.process(InputCommand::SetBrush(field));
    engine.process(InputCommand::StartStroke {
        tool: Tool::Brush,
        sample: InputSample::at(shift + Vec2::new(-60.0, -60.0)),
    });
    engine.process(InputCommand::StrokeTo {
        sample: InputSample::at(shift + Vec2::new(60.0, 60.0)),
    });
    engine.process(InputCommand::EndStroke);

    // The smudge under test, through the same 4-tile corner.
    let mut smudge = brush(RED, 28.0);
    smudge.tooth = 0.0;
    smudge.dynamics = BrushDynamics {
        add: 0.0,
        lift: 0.6,
        deposit: 0.5,
        ..Default::default()
    };
    engine.process(InputCommand::SetBrush(smudge));
    engine.process(InputCommand::StartStroke {
        tool: Tool::Brush,
        sample: InputSample::at(shift + Vec2::new(-50.0, -50.0)),
    });
    engine.process(InputCommand::StrokeTo {
        sample: InputSample::at(shift + Vec2::new(50.0, 50.0)),
    });
    engine.process(InputCommand::EndStroke);

    let center_px = Vec2::new(SIZE.width as f32 * 0.5, SIZE.height as f32 * 0.5);
    engine.process(InputCommand::Pan { delta: -shift });
    engine.process(InputCommand::Zoom {
        anchor: center_px,
        factor: 2.0,
    });
    engine.render_to_image(BG)
}

#[test]
fn apron_makes_dynamics_writeback_seamless_under_zoom() {
    if engine_or_skip().is_none() {
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
    // lighting that surfaces as a broad, imperceptible (≤ a few levels) residual
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

