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
use stark_core::document::{BrushDynamics, DryParams, Tool, WetParams};
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

/// Like `render_shifted`, but the height-bearing stroke is a **medium** (knife)
/// write-back rather than the additive deposit: lay a wide red field through the
/// corner, then scrape a knife along it with carry + ridge on. This exercises the
/// read-modify-write combine path (footprint→scratch→combine→CoW), whose apron
/// must stay bit-identical to the neighbor's interior just like the deposit's
/// (DESIGN.md §6.2/§6.4) — the ridge term is deliberately a function of the local
/// coverage only (no neighbor reads) so it can't introduce a boundary discontinuity.
fn render_shifted_knife(shift: Vec2) -> RgbaImage {
    let mut engine = engine_or_skip().expect("engine (caller checked adapter)");
    engine.set_media_params(MediaParams {
        height_strength: 2.5,
        specular: 0.3,
        surface_strength: 0.0,
        ..MediaParams::default()
    });

    // A wide base field along the diagonal, fully containing the knife's path.
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

    // The Dry scrape+smear+ridge under test, through the same 4-tile corner.
    let mut knife = brush(RED, 28.0);
    knife.tooth = 0.0;
    knife.dynamics = BrushDynamics::Dry(DryParams {
        smear: 0.5,
        remove: 0.5,
        add: 0.0,
        ridge: 1.0,
    });
    engine.process(InputCommand::SetBrush(knife));
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

/// Like `render_shifted`, but the stroke is a **wet** brush whose post-deposit
/// diffusion runs over a composited region and is sliced back into tiles. The region
/// must include a one-tile halo so each rewritten tile's apron reads its neighbour's
/// real interior; otherwise the copy-back overwrites aprons toward unaffected
/// neighbours with empty region content — a seam, glaring in the relief normals.
fn render_shifted_wet(shift: Vec2) -> RgbaImage {
    let mut engine = engine_or_skip().expect("engine (caller checked adapter)");
    engine.set_media_params(MediaParams {
        height_strength: 2.5,
        specular: 0.3,
        surface_strength: 0.0,
        ..MediaParams::default()
    });

    // A broad base field covering all four tiles around the corner, so the corner the
    // view is centred on has paint in every quadrant.
    let mut field = brush(RED, 90.0);
    field.tooth = 0.0;
    engine.process(InputCommand::SetBrush(field));
    engine.process(InputCommand::StartStroke {
        tool: Tool::Brush,
        sample: InputSample::at(shift + Vec2::new(-150.0, 0.0)),
    });
    engine.process(InputCommand::StrokeTo {
        sample: InputSample::at(shift + Vec2::new(150.0, 0.0)),
    });
    engine.process(InputCommand::EndStroke);

    // The wet stroke under test, confined to the corner's +,+ tile (offset from the
    // corner by more than radius+apron so it does NOT touch the other three tiles).
    // The visible corner is therefore an affected/unaffected tile boundary cutting
    // through painted canvas — exactly where a missing-halo apron seams the relief.
    // Both axes on, so the test covers the advect + diffuse write-back together.
    let mut wet = brush(RED, 24.0);
    wet.tooth = 0.0;
    wet.dynamics = BrushDynamics::Wet(WetParams { bleed: 0.9, drag: 0.9 });
    engine.process(InputCommand::SetBrush(wet));
    engine.process(InputCommand::StartStroke {
        tool: Tool::Brush,
        sample: InputSample::at(shift + Vec2::new(40.0, 40.0)),
    });
    engine.process(InputCommand::StrokeTo {
        sample: InputSample::at(shift + Vec2::new(85.0, 85.0)),
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

#[test]
fn apron_makes_medium_writeback_seamless_under_zoom() {
    if engine_or_skip().is_none() {
        return; // no usable GPU adapter
    }

    // Same invariant as above, but for the knife's read-modify-write path: a scrape
    // straddling the 4-tile corner must render identically to the same scrape shifted
    // into one tile's interior. A broken apron in the combine pass (or a ridge that
    // sampled neighbors) would seam along every tile boundary.
    let corner = render_shifted_knife(Vec2::ZERO);
    let interior = render_shifted_knife(Vec2::new(128.0, 128.0));

    // A real missing apron seams a *contiguous* band along every tile boundary — many
    // pixels differing by tens of levels. A smear's laid height rides the per-column
    // reservoir, whose region composite carries an unavoidable f16 sub-texel difference
    // at internal tile boundaries; with the exaggerated relief here that surfaces as a
    // *handful* of isolated specks — all under ~14 levels, none above 20 — not a band.
    // So gate on the significantly-different *area* (>12 levels): a real seam blows past
    // it (a stark ridge, well over 0.5% of pixels), the precision specks sit ~0.01%.
    let frac_any = diff_fraction(&corner, &interior).0;
    let frac_big = frac_exceeding(&corner, &interior, 12);
    assert!(
        frac_big < 0.005 && frac_any < 0.07,
        "medium write-back seam: {:.3}% of pixels differ by >12 levels ({:.2}% differ at \
         all) — the combine pass is not covering tile boundaries",
        frac_big * 100.0,
        frac_any * 100.0,
    );
}

#[test]
fn apron_makes_wet_diffusion_seamless_under_zoom() {
    if engine_or_skip().is_none() {
        return; // no usable GPU adapter
    }

    // The wet brush's region-diffusion write-back must be seam-free: a wet stroke
    // straddling the 4-tile corner must render identically to the same stroke inside
    // one tile's interior. Without the halo composite, rewritten tiles' aprons toward
    // unaffected neighbours land on empty region → a relief seam along the boundary.
    let corner = render_shifted_wet(Vec2::ZERO);
    let interior = render_shifted_wet(Vec2::new(128.0, 128.0));

    let (frac, worst) = diff_fraction(&corner, &interior);
    assert!(
        worst <= 25 && frac < 0.07,
        "wet diffusion seam: corner vs interior differ by up to {worst} levels \
         on {:.2}% of pixels — the region write-back is not covering tile boundaries",
        frac * 100.0
    );
}

