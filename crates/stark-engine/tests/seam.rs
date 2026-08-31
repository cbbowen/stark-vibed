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

const RED: [f32; 3] = [0.85, 0.15, 0.1];

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
    b.jitter = 0.0;
    engine.process(ViewCommand::set_brush(b));
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
    field.jitter = 0.0;
    engine.process(ViewCommand::set_brush(field));
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
    smudge.jitter = 0.0;
    smudge.effect = stark_model::document::BrushEffect::wet_with(
        RED,
        BrushDynamics {
            flow: 0.0,
            lift: 0.6,
            deposit: 0.5,
            ..Default::default()
        },
    );
    engine.process(ViewCommand::set_brush(smudge));
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

// ---------------------------------------------------------------------------
// The other tile writers
//
// §6.4 is a rule about **every pass that writes tiles**, and the three tests above
// ask it of three of them: the swept stroke, the placed image, the dynamics
// write-back. The rest — the fill, the selection mask every tool acts through, the
// transform, the merge — were unguarded, and each is a pass that computes a texel
// from a canvas position and can get the apron wrong in exactly the same way.
//
// One table rather than four more `render_shifted_*` functions, because what differs
// between them is one closure: what to do at `shift`. The view treatment, the media
// settings and the comparison belong to the invariant, not to the operation.

/// The media settings a translation-invariance comparison has to run under, and why
/// each is off: the canvas substrate is sampled in canvas space and the display dither
/// is keyed to the screen pixel (§6.5), so both are *deliberately* not tile-grid
/// invariant and would fog the measurement. Relief stays exaggerated, because a
/// clamped normal at a tile edge is what a missing apron looks like.
const SEAM_MEDIA: MediaParams = MediaParams {
    height_strength: 2.5,
    specular: 0.3,
    substrate_strength: 0.0,
    dither: false,
};

/// A brush whose mark is a pure function of canvas position: no deposit jitter, whose
/// gate is keyed to the canvas texel (§6.2), and no tooth, which reads the substrate.
fn flat_brush(color: [f32; 3], radius: f32) -> stark_model::document::BrushParams {
    let mut b = brush(color, radius);
    b.jitter = 0.0;
    b.tooth.give = 1.0;
    b
}

/// Run `op` at `shift`, viewed so that the screen mapping is identical for every
/// `shift` — the whole of [`render_shifted`]'s treatment, with the operation lifted
/// out.
fn render_shifted_op(shift: Vec2, op: impl FnOnce(&mut stark_engine::Engine, Vec2)) -> RgbaImage {
    let mut engine = engine_or_skip_blue().expect("engine (caller checked adapter)");
    engine.process(ViewCommand::SetMediaParams(SEAM_MEDIA));
    op(&mut engine, shift);
    let center_px = Vec2::new(SIZE.width as f32 * 0.5, SIZE.height as f32 * 0.5);
    engine.process(ViewCommand::Pan { delta: -shift });
    engine.process(ViewCommand::Zoom {
        anchor: center_px,
        factor: 2.0,
    });
    engine.render_to_image()
}

/// An undercoat for the operations that move or fold paint rather than lay it.
fn undercoat(engine: &mut stark_engine::Engine, shift: Vec2) {
    stroke_with(
        engine,
        flat_brush(RED, 30.0),
        &[
            shift + Vec2::new(-60.0, -40.0),
            shift + Vec2::new(60.0, 40.0),
        ],
    );
}

/// **Every tile-writing pass is a pure function of canvas position** (§6.4), asked of
/// the four this file did not previously reach.
///
/// Each case does the same thing twice — once straddling the 4-tile corner at the
/// origin, once shifted a half-tile into one tile's interior — and the two renders
/// must agree. A pass that addressed a texel relative to *its tile* rather than to
/// the canvas looks perfect at one alignment and seams at every boundary.
///
/// The comparison is sensitive to exactly that: dropping the `shift` from one case's
/// region — so it addresses the canvas origin instead of the operation's own — takes
/// it to 219 levels over 57% of the frame.
#[test]
fn every_tile_writer_is_translation_invariant() {
    use stark_engine::command::PeerCommand;
    use stark_model::Srgb;
    use stark_model::document::{FillOp, SelectionMode, SelectionOp, SelectionShape, TransformMap};

    if engine_or_skip_blue().is_none() {
        return; // no usable GPU adapter
    }

    type Case = (&'static str, fn(&mut stark_engine::Engine, Vec2));
    let cases: &[Case] = &[
        // The fill pass: a feathered region laid straight onto bare canvas.
        ("fill", |e, shift| {
            e.process(DocCommand::Fill {
                layer: e.observe().active_layer,
                op: FillOp::new(
                    SelectionShape::rect_from_corners(
                        shift + Vec2::splat(-45.0),
                        shift + Vec2::splat(45.0),
                    ),
                    6.0,
                    Srgb::new([0.2, 0.7, 0.35]),
                    1.0,
                ),
            });
        }),
        // The **selection mask** every tool acts through (§6.8), seen through a fill
        // that covers far more than the mask does — so what shapes the result is the
        // rasterized coverage rather than the fill's own region.
        ("mask", |e, shift| {
            e.process(DocCommand::Select(SelectionOp::new(
                SelectionMode::Replace,
                SelectionShape::ellipse_from_corners(
                    shift + Vec2::splat(-40.0),
                    shift + Vec2::splat(40.0),
                ),
                7.0,
            )));
            e.process(DocCommand::Fill {
                layer: e.observe().active_layer,
                op: FillOp::new(
                    SelectionShape::rect_from_corners(
                        shift + Vec2::splat(-120.0),
                        shift + Vec2::splat(120.0),
                    ),
                    0.0,
                    Srgb::new([0.85, 0.4, 0.1]),
                    1.0,
                ),
            });
        }),
        // The transform's resample. The map is expressed in canvas coordinates and
        // turns about `shift`, so the *content* it produces is the same picture at
        // either alignment — only the tile grid under it moves.
        ("transform", |e, shift| {
            undercoat(e, shift);
            let turn = stark_model::geom::Affine2::from_angle_translation(0.4, Vec2::ZERO);
            let about = stark_model::geom::Affine2::from_angle_translation(0.0, shift)
                * turn
                * stark_model::geom::Affine2::from_angle_translation(0.0, -shift);
            e.process(DocCommand::Transform {
                layer: e.observe().active_layer,
                map: TransformMap::Affine(about),
            });
        }),
        // The merge's tile rewrite (§14.11): two painted layers folded into one.
        ("merge", |e, shift| {
            undercoat(e, shift);
            let lower = e.observe().active_layer;
            e.process(DocCommand::AddLayer {
                carrier: None,
                above: Some(lower),
            });
            let upper = e.observe().active_layer;
            e.process(PeerCommand::SetActiveLayer(upper));
            stroke_with(
                e,
                flat_brush([0.15, 0.35, 0.9], 24.0),
                &[
                    shift + Vec2::new(-60.0, 40.0),
                    shift + Vec2::new(60.0, -40.0),
                ],
            );
            e.process(DocCommand::MergeLayerDown(upper));
        }),
    ];

    // Bare canvas under the same view and the same light — what each case is held
    // against to prove it drew anything at all. Without it a case whose command was
    // declined (a wrong layer id, a refused region) compares blank against blank and
    // passes having measured nothing, which is how five tests in this suite came to
    // be comparing paper with paper (see `corpus`'s `every_case_leaves_a_mark`).
    let blank = render_shifted_op(Vec2::ZERO, |_, _| {});

    for (name, op) in cases {
        let corner = render_shifted_op(Vec2::ZERO, *op);
        let interior = render_shifted_op(Vec2::new(128.0, 128.0), *op);
        assert!(
            !images_match(&corner, &blank, 8),
            "{name}: left no mark, so the comparison below is blank against blank"
        );
        // The corpus's kind of bound rather than this file's older one: these passes
        // write a texel from its own canvas position with no filtering across the
        // boundary, so unlike the lit-relief comparisons above there is no sub-pixel
        // residual to allow for. A seam is a contiguous band of tens of levels.
        let (_, worst) = diff_fraction(&corner, &interior);
        let frac_big = frac_exceeding(&corner, &interior, 12);
        assert!(
            worst <= 25 && frac_big < 0.005,
            "{name}: corner vs interior differ by up to {worst} levels, {:.3}% of \
             pixels by >12 — this pass is not a pure function of canvas position (§6.4)",
            frac_big * 100.0,
        );
    }
}
