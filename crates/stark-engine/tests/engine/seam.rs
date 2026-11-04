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

use crate::common::*;
use stark_engine::command::{DocCommand, Edit, ViewCommand};
use stark_engine::{MediaParams, RgbaImage};
use stark_model::document::BrushDynamics;
use stark_model::geom::{IVec2, Vec2};

// Not the palette's: the seam tolerances below were measured against this red.
const RED: [f32; 3] = [0.85, 0.15, 0.1];

/// The worst height step between any tile's apron and the neighbour interior it
/// duplicates, and the tallest texel either holds. Read off the tiles rather than a
/// render, so the lit image cannot blur a one-texel slip away.
fn apron_mismatch(engine: &stark_engine::Engine) -> (f32, f32, usize) {
    use stark_model::document::LayerId;
    use stark_model::geom::{TILE_APRON, TILE_SIZE, TILE_TEX, TileCoord};

    let layer = LayerId::ROOT;
    let coords = layer_tiles(engine, layer);
    let (lo, hi) = (TILE_APRON, TILE_APRON + TILE_SIZE);
    let at = |row: u32, col: u32| (row * TILE_TEX + col) as usize;
    // Counted and handed back, because a run whose tiles are not adjacent compares no
    // boundary at all and would report a worst of zero.
    let mut pairs = 0usize;
    let (mut worst, mut peak) = (0.0f32, 0.0f32);
    for &a in &coords {
        let here = engine
            .tile_channels(layer, a)
            .expect("a listed tile reads back");
        peak = here.height.iter().copied().fold(peak, f32::max);
        for (dx, dy) in [(1, 0), (0, 1)] {
            let Some(there) = engine.tile_channels(layer, TileCoord::new(a.x + dx, a.y + dy))
            else {
                continue;
            };
            pairs += 1;
            // A texel by its place along the shared edge and across it.
            let rc = |along, across| {
                if dx == 1 {
                    at(along, across)
                } else {
                    at(across, along)
                }
            };
            for k in lo..hi {
                // Each side's last interior texel against the other's apron copy of it.
                worst = worst
                    .max((here.height[rc(k, hi - 1)] - there.height[rc(k, lo - 1)]).abs())
                    .max((there.height[rc(k, lo)] - here.height[rc(k, hi)]).abs());
            }
        }
    }
    (worst, peak, pairs)
}

/// **A toothed stroke keeps its aprons** (§6.4), on the swept path and the loop. The
/// substrate is read in canvas space, so an apron texel must bite the same map texel as
/// the neighbour interior it duplicates; on hard stripes at two map texels per px a
/// one-texel slip is a full step of the gate.
#[test]
fn a_toothed_stroke_keeps_its_aprons() {
    use stark_model::document::{BrushEffect, ToothParams};

    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    on_substrate(&mut engine, &stripe_substrate());
    let effects = [
        BrushEffect::painted(RED),
        BrushEffect::wet_with(RED, BrushDynamics::default()),
    ];
    for effect in effects {
        let mut b = brush(RED, 40.0);
        b.effect = effect;
        b.jitter = 0.0;
        b.drain = 0.0;
        b.tooth = ToothParams {
            give: 0.0,
            softness: 0.06,
        };
        // Diagonal, so the gate varies along the horizontal edges' apron rows too.
        let run = [Vec2::new(-70.0, -60.0), Vec2::new(70.0, 80.0)];
        stroke_with(&mut engine, b, &run);
        let (worst, peak, pairs) = apron_mismatch(&engine);
        engine.process(DocCommand::Undo);
        assert!(peak > 0.0, "the stroke laid nothing");
        assert!(
            pairs > 0,
            "the run spans one tile, so no boundary was compared"
        );
        // Bit-identical on the adapter the suite runs on, whose rasterizer interpolates
        // the same under an integer shift of the target; the slack is for one that does not.
        assert!(
            worst <= 0.01 * peak,
            "an apron differs from the interior it duplicates by {worst} of a {peak} peak — \
             the tooth is not a pure function of canvas position"
        );
    }
}

// ---------------------------------------------------------------------------
// Every tile writer, under zoom
//
// §6.4 is a rule about **every pass that writes tiles**: the swept stroke, the placed
// image, the dynamics write-back, the fill, the selection mask every tool acts
// through, the transform, the merge. Each computes a texel from a canvas position and
// can get the apron wrong in exactly the same way.
//
// One table rather than a `render_shifted_*` function per writer, because what differs
// between them is one closure — what to do at `shift` — and how far the two
// alignments may sit apart. The view treatment, the media settings, the blank-canvas
// control and the comparison belong to the invariant, not to the operation.

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

/// **The apron rule survives a long way from the origin** (§6.4) — and the mark does
/// not quite.
///
/// The swept fast path works in **absolute** f32 canvas px (`stamp_common.wesl`'s
/// `in.canvas`), where the dynamics loop works region-local; at 2²⁰ px out the frame's
/// own ULP is ⅛ px. Every other test in this file paints about the origin, so this is
/// the one that asks the apron-vs-neighbour-interior question at a distance.
///
/// **The seam holds.** Worst apron-to-interior height step against a ~4.05 peak, swept
/// then loop: 0/0 at the origin, one f16 ULP (0.0039) / 0 at every shift from 2⁸ to
/// 2²², with no shift in between where it grows. The region path is exactly zero
/// throughout, which is what working region-local buys it.
///
/// What drifts instead is the **mark itself**: the same stroke lays +0.06% of height at
/// 2¹⁶, +1.2% at 2²⁰ and +5.7% at 2²², because the tip's coverage is evaluated at a
/// position the frame can no longer resolve. A real limitation of the infinite canvas,
/// but not a seam — it moves a tile and its neighbour together, which is why the two
/// bounds below are so far apart.
#[test]
fn a_stroke_far_from_the_origin_keeps_its_aprons() {
    use stark_model::document::{BrushEffect, LayerId};
    use stark_model::geom::{TILE_SIZE, TileCoord};

    // Shift from the origin, and what it may move the stroke's weight by — `None` at
    // the origin, which is what the other two are measured against. Both shifts are
    // whole multiples of `TILE_SIZE` and exact in f32, so the comparison holds tile
    // phase and sub-texel phase fixed and only the frame's own precision varies.
    const CASES: [(f32, Option<f64>); 3] = [
        (0.0, None),
        (1_048_576.0, Some(0.02)),
        (4_194_304.0, Some(0.08)),
    ];

    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    for (what, effect) in [
        ("swept", BrushEffect::painted(RED)),
        ("loop", BrushEffect::wet_with(RED, BrushDynamics::default())),
    ] {
        let mut at_origin = 0.0;
        for (shift, drift_bound) in CASES {
            let mut b = flat_brush(RED, 40.0);
            b.effect = effect;
            b.drain = 0.0;
            let o = Vec2::splat(shift);
            stroke_with(
                &mut engine,
                b,
                &[o + Vec2::new(-70.0, -60.0), o + Vec2::new(70.0, 80.0)],
            );
            let (worst, peak, pairs) = apron_mismatch(&engine);
            let total = total_height(&engine, LayerId::ROOT);
            // The mark is where it was asked for, not back at the origin — without this
            // a stroke the engine had declined would compare an empty layer with itself.
            let out_there = (shift / TILE_SIZE as f32) as i32;
            let landed = engine
                .tile_channels(LayerId::ROOT, TileCoord::new(out_there, out_there))
                .is_some();
            engine.process(DocCommand::Undo);
            assert!(
                peak > 0.0 && landed && pairs > 0,
                "{what}: the stroke at {shift} left no tile at ({out_there}, \
                 {out_there}), or none adjacent to compare across"
            );
            // One f16 ULP at this peak is 0.00098 of it, so this is four of them — a
            // hundredth of what the toothed case above allows, and what says the two
            // sides of a boundary are still the same number rather than merely close.
            assert!(
                worst <= 0.004 * peak,
                "{what}: at {shift} px out, an apron differs from the interior it \
                 duplicates by {worst} of a {peak} peak — the absolute canvas frame has \
                 stopped resolving a tile boundary the same way from both sides"
            );
            let Some(drift_bound) = drift_bound else {
                at_origin = total;
                continue;
            };
            let drift = (total - at_origin) / at_origin;
            assert!(
                drift.abs() <= drift_bound,
                "{what}: at {shift} px out the same stroke laid {total:.1} of height \
                 against {at_origin:.1} at the origin ({:+.2}%, bound ±{:.0}%) — the \
                 f32 canvas frame is resolving the tip differently than it did",
                drift * 100.0,
                drift_bound * 100.0,
            );
        }
    }
}

/// A brush whose mark is a pure function of canvas position: no deposit jitter, whose
/// gate is keyed to the canvas texel (§6.2), and no tooth, which reads the substrate.
fn flat_brush(color: [f32; 3], radius: f32) -> stark_model::document::BrushParams {
    let mut b = brush(color, radius).unjittered();
    b.tooth.give = 1.0;
    b
}

/// Run `op` at `shift`, viewed so that the screen mapping is identical for every
/// `shift` — centred on `shift`, then magnified 2× about the viewport centre, so the
/// two renders differ only in how the operation lands on the tile grid. Under
/// magnification the compositor samples each tile bilinearly, which is where a missing
/// apron shows.
fn render_shifted_op(shift: Vec2, op: impl FnOnce(&mut stark_engine::Engine, Vec2)) -> RgbaImage {
    let mut engine = engine_or_skip_blue().expect("engine (caller checked adapter)");
    engine.process(ViewCommand::SetMediaParams(SEAM_MEDIA));
    op(&mut engine, shift);
    // Pan centres the view on `shift` (center -= delta/zoom, at zoom 1); the zoom keeps
    // the canvas point under the viewport centre put.
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

/// How far a case's two alignments may sit apart: no texel by more than `worst`, and
/// no more than `share` of the frame by more than `over`.
///
/// Per case, because the writers are not equally exact. A seam — a missing apron, a
/// texel addressed relative to its tile — is a contiguous band of tens of levels along
/// every boundary, which each of these bounds refuses; what they differ in is how much
/// of the residual *short* of a seam each writer is entitled to.
#[derive(Clone, Copy)]
struct SeamTol {
    worst: u8,
    over: u8,
    share: f64,
}

/// A writer that computes every texel from its own canvas position and filters nothing
/// across a boundary — the placed image, the fill, the mask. Nothing stands between the
/// two runs but the compositor's sampling, itself a pure function of canvas position,
/// so anything but a near-exact match is an addressing bug. Measured bit-identical.
const ADDRESSED: SeamTol = SeamTol {
    worst: 2,
    over: 0,
    share: 0.01,
};

/// A height-bearing mark under [`SEAM_MEDIA`]'s exaggerated relief — the swept stroke
/// and the dynamics write-back. Measured at a worst of 1 level for both, over 0.4% and
/// 8% of the frame, none of it by more than 2: under the reference light (§6.3) the
/// relief is lit by a function of the normal alone, with nothing to amplify a rounding
/// difference. The bounds these two rows had before were set under image-based
/// lighting, whose sharp specular took the same residual to ~10 levels, and they went
/// with it.
const LIT: SeamTol = SeamTol {
    worst: 6,
    over: 2,
    share: 0.005,
};

/// A writer that reads tiles back and writes them again — the transform's resample,
/// the merge's fold. Measured at a worst of 13 and 14 levels, over 0.002% and 0.009%
/// of the frame by more than 12, a scatter of single texels rather than a band.
const REWRITTEN: SeamTol = SeamTol {
    worst: 25,
    over: 12,
    share: 0.005,
};

/// **Every tile-writing pass is a pure function of canvas position** (§6.4).
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
    use stark_model::document::{
        BrushEffect, FillOp, SelectionMode, SelectionOp, SelectionShape, TransformMap,
    };

    if engine_or_skip_blue().is_none() {
        return; // no usable GPU adapter
    }

    type Case = (&'static str, fn(&mut stark_engine::Engine, Vec2), SeamTol);
    let cases: &[Case] = &[
        // The swept stroke: a diagonal, height-bearing mark through the corner, under
        // the exaggerated relief of [`SEAM_MEDIA`], so a clamped normal at a tile edge
        // shows as a lighting ridge.
        (
            "stroke",
            |e, shift| {
                stroke_with(
                    e,
                    flat_brush(RED, 28.0),
                    &[
                        shift + Vec2::new(-50.0, -50.0),
                        shift + Vec2::new(50.0, 50.0),
                    ],
                );
            },
            LIT,
        ),
        // A **placed image** (§23) — the one tile writer here that is not a render
        // pass. Every texel is computed from its own canvas position on the CPU, so a
        // tile's apron is bit-identical to its neighbour's interior by construction,
        // and only the compositor's sampling stands between the two runs. An
        // implementation that filled each tile from the image's origin *relative to
        // the tile* would look perfect at one alignment and seam at every boundary.
        (
            "image",
            |e, shift| {
                // Varying rather than flat: a constant image is seamless however
                // wrongly it is addressed. Big enough to span the corner every way.
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
                let id = e
                    .import_picture(&picture_png(SIDE, SIDE, pixels))
                    .expect("import the field");
                e.process(DocCommand::PlaceImage {
                    carrier: None,
                    above: None,
                    at: IVec2::new(shift.x as i32, shift.y as i32) - IVec2::splat(SIDE as i32 / 2),
                    name: None,
                    image: id,
                });
            },
            ADDRESSED,
        ),
        // The **dynamics loop's region write-back** (§6.2): a red field through the
        // corner, then a smearing brush dragged along it. The whole-block slice from
        // the shared region must keep aprons bit-identical to neighbour interiors, and
        // the halo composite must give rewritten tiles real neighbour content.
        (
            "smudge",
            |e, shift| {
                stroke_with(
                    e,
                    flat_brush(RED, 60.0),
                    &[
                        shift + Vec2::new(-60.0, -60.0),
                        shift + Vec2::new(60.0, 60.0),
                    ],
                );
                let mut smudge = flat_brush(RED, 28.0);
                smudge.effect = BrushEffect::wet_with(
                    RED,
                    BrushDynamics {
                        add: 0.0,
                        lift: 0.6,
                        deposit: 0.5,
                        ..Default::default()
                    },
                );
                stroke_with(
                    e,
                    smudge,
                    &[
                        shift + Vec2::new(-50.0, -50.0),
                        shift + Vec2::new(50.0, 50.0),
                    ],
                );
            },
            LIT,
        ),
        // The fill pass: a feathered region laid straight onto bare canvas.
        (
            "fill",
            |e, shift| {
                e.process(DocCommand::Set(Edit::Fill {
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
                }));
            },
            ADDRESSED,
        ),
        // The **selection mask** every tool acts through (§6.8), seen through a fill
        // that covers far more than the mask does — so what shapes the result is the
        // rasterized coverage rather than the fill's own region.
        (
            "mask",
            |e, shift| {
                e.process(DocCommand::Select(SelectionOp::new(
                    SelectionMode::Replace,
                    SelectionShape::ellipse_from_corners(
                        shift + Vec2::splat(-40.0),
                        shift + Vec2::splat(40.0),
                    ),
                    7.0,
                )));
                e.process(DocCommand::Set(Edit::Fill {
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
                }));
            },
            ADDRESSED,
        ),
        // The transform's resample. The map is expressed in canvas coordinates and
        // turns about `shift`, so the *content* it produces is the same picture at
        // either alignment — only the tile grid under it moves.
        (
            "transform",
            |e, shift| {
                undercoat(e, shift);
                let turn = stark_model::geom::Affine2::from_angle_translation(0.4, Vec2::ZERO);
                let about = stark_model::geom::Affine2::from_angle_translation(0.0, shift)
                    * turn
                    * stark_model::geom::Affine2::from_angle_translation(0.0, -shift);
                e.process(DocCommand::Set(Edit::Transform {
                    layer: e.observe().active_layer,
                    map: TransformMap::Affine(about),
                }));
            },
            REWRITTEN,
        ),
        // The merge's tile rewrite (§14.11): two painted layers folded into one.
        (
            "merge",
            |e, shift| {
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
            },
            REWRITTEN,
        ),
    ];

    // Bare canvas under the same view and the same light — what each case is held
    // against to prove it drew anything at all. Without it a case whose command was
    // declined (a wrong layer id, a refused region) compares blank against blank and
    // passes having measured nothing, which is how five tests in this suite came to
    // be comparing paper with paper (see the corpus battery's `check_leaves_a_mark`).
    let blank = render_shifted_op(Vec2::ZERO, |_, _| {});

    for (name, op, tol) in cases {
        let corner = render_shifted_op(Vec2::ZERO, *op);
        let interior = render_shifted_op(Vec2::new(128.0, 128.0), *op);
        assert_images_differ(
            &corner,
            &blank,
            8,
            format_args!("{name}: left no mark, so the comparison below is blank against blank"),
        );
        let (_, worst) = diff_fraction(&corner, &interior);
        let share = frac_exceeding(&corner, &interior, tol.over);
        assert!(
            worst <= tol.worst && share <= tol.share,
            "{name}: corner vs interior differ by up to {worst} levels, {:.3}% of \
             pixels by >{} — this pass is not a pure function of canvas position (§6.4)",
            share * 100.0,
            tol.over,
        );
    }
}
