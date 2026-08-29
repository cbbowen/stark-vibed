//! Transforms of the selected paint: the affine (§16), and the rect-scoped
//! perspective (§16.8) and warp (§16.9).
//!
//! The exactness invariants carry most of the weight, because they are what the
//! cut/paste laws were chosen for (§16.2–§16.4): the identity
//! transform is a byte-for-byte no-op at any feather — for all three families —
//! integer translations move the painting without a texel of resampling loss,
//! and involutions (a flip, a there-and-back move of the whole layer) compose
//! to the identity through separate actions. Behavioural probes cover the cut,
//! the carry of the selection, the rect gate, deterministic rejection, undo,
//! and save/load replay; goldens pin genuinely-resampled output.

mod common;

use common::*;
use stark_engine::RgbaImage;
use stark_engine::command::DocCommand;
#[cfg(feature = "mixbox")]
use stark_model::ColorSpaceId;
use stark_model::Srgb;
use stark_model::document::{
    LayerId, MatteRegion, Parcel, PerspectiveMap, Place, SelectionMode, SelectionOp,
    SelectionShape, TransformMap, WarpMap, rect_corners,
};
use stark_model::geom::{Affine2, Vec2};

const RED: [f32; 3] = [1.0, 0.0, 0.0];
const GREEN: [f32; 3] = [0.1, 0.8, 0.2];

/// A compact red blob centred at `c` — two short crossing strokes, so it has
/// height variation for the lighting to bite on.
fn blob(engine: &mut stark_engine::Engine, c: Vec2) {
    paint(
        engine,
        RED,
        16.0,
        &[c + Vec2::new(-18.0, 0.0), c + Vec2::new(18.0, 0.0)],
    );
    paint(
        engine,
        RED,
        10.0,
        &[c + Vec2::new(0.0, -14.0), c + Vec2::new(0.0, 14.0)],
    );
}

fn select_rect(engine: &mut stark_engine::Engine, min: Vec2, max: Vec2, feather: f32) {
    engine.process(DocCommand::Select(SelectionOp::new(
        SelectionMode::Replace,
        SelectionShape::rect_from_corners(min, max),
        feather,
    )));
}

fn transform(engine: &mut stark_engine::Engine, affine: Affine2) {
    transform_map(engine, TransformMap::Affine(affine));
}

fn transform_map(engine: &mut stark_engine::Engine, map: TransformMap) {
    let layer = engine.observe().active_layer;
    engine.process(DocCommand::Transform { layer, map });
}

fn assert_identical(a: &RgbaImage, b: &RgbaImage, what: &str) {
    let (frac, worst) = diff_fraction(a, b);
    if worst == 0 {
        return;
    }
    // Where: the first few differing pixels, to say whether a failure is at the
    // paint, the selection outline, or somewhere structural like a tile seam.
    let mut sites = Vec::new();
    'scan: for y in 0..a.height {
        for x in 0..a.width {
            let i = ((y * a.width + x) * 4) as usize;
            if a.pixels[i..i + 4] != b.pixels[i..i + 4] {
                sites.push(format!(
                    "({x},{y}) {:?} -> {:?}",
                    &a.pixels[i..i + 4],
                    &b.pixels[i..i + 4]
                ));
                if sites.len() >= 8 {
                    break 'scan;
                }
            }
        }
    }
    panic!(
        "{what}: images differ (worst channel delta {worst}, {:.4}% of pixels differ)\nfirst sites: {}",
        frac * 100.0,
        sites.join("; ")
    );
}

// --- The exactness invariants (§16.4) ---------------------------

#[test]
fn identity_transform_is_a_noop() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    blob(&mut engine, Vec2::new(-30.0, 0.0));
    select_rect(
        &mut engine,
        Vec2::new(-60.0, -40.0),
        Vec2::new(0.0, 40.0),
        0.0,
    );
    let before = engine.render_to_image();
    transform(&mut engine, Affine2::IDENTITY);
    let after = engine.render_to_image();
    assert_identical(&before, &after, "identity transform");
}

/// The strong form of the law test: at a *feathered* edge the cut takes `h·m`,
/// the paste puts `h·m` back, and the lift law recombines them to the original —
/// chosen precisely so this holds (§16.2). Inside the ramp the
/// values are genuinely recomputed (`h·(1−m) + h·m`), and the render target's
/// f32→f16 store rounding is implementation-defined, so a ramp texel may land
/// one f16 tile ulp off — at most one display LSB. Coverage 0 and 1 take exact
/// branches and are pinned byte-exact by the other tests; this one asserts the
/// ramp stays within that single ulp.
#[test]
fn identity_transform_is_a_noop_at_a_feathered_edge() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // Dither off (§6.5): the single-ulp drift this pins is sub-code, and the
    // display's own half-code of noise can round it across two boundaries.
    engine.process(stark_engine::command::ViewCommand::SetMediaParams(
        stark_engine::MediaParams {
            dither: false,
            ..Default::default()
        },
    ));
    blob(&mut engine, Vec2::new(-30.0, 0.0));
    select_rect(
        &mut engine,
        Vec2::new(-60.0, -40.0),
        Vec2::new(-10.0, 40.0),
        20.0,
    );
    let before = engine.render_to_image();
    transform(&mut engine, Affine2::IDENTITY);
    let after = engine.render_to_image();
    let (_, worst) = diff_fraction(&before, &after);
    assert!(
        worst <= 1,
        "feathered identity transform drifted past one display LSB (worst {worst})"
    );
}

/// Mixbox-only, so it exists only in a build carrying the `mixbox` feature.
/// `ColorSpaceId::Mixbox` still *names* a space there — the save format's enum
/// indices cannot depend on a feature (§8) — but nothing can open one.
#[cfg(feature = "mixbox")]
#[test]
fn identity_transform_is_a_noop_in_mixbox() {
    let Some(mut engine) = engine_or_skip_with(ColorSpaceId::Mixbox) else {
        return;
    };
    blob(&mut engine, Vec2::new(-30.0, 0.0));
    select_rect(
        &mut engine,
        Vec2::new(-60.0, -40.0),
        Vec2::new(0.0, 40.0),
        4.0,
    );
    let before = engine.render_to_image();
    transform(&mut engine, Affine2::IDENTITY);
    let after = engine.render_to_image();
    let (_, worst) = diff_fraction(&before, &after);
    assert!(worst <= 1, "mixbox identity transform (worst {worst})");
}

/// An integer translation resamples at exact texel centres, so the painting moves
/// with no loss at all: the after-image is the before-image shifted, byte for byte.
#[test]
fn integer_translation_is_exact() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // Dither off (§6.5): the display's dither is keyed to the *screen* pixel by
    // design, so it is the one layer that does not shift with the painting.
    // Byte-exactness is asserted of everything beneath it.
    engine.process(stark_engine::command::ViewCommand::SetMediaParams(
        stark_engine::MediaParams {
            dither: false,
            ..Default::default()
        },
    ));
    blob(&mut engine, Vec2::new(-60.0, -20.0));
    let before = engine.render_to_image();
    let (dx, dy) = (64i32, 32i32);
    transform(
        &mut engine,
        Affine2::from_translation(Vec2::new(dx as f32, dy as f32)),
    );
    let after = engine.render_to_image();

    // Compare `after` against `before` shifted by (dx, dy), over the region both
    // frames cover. The substrate is uniform and the lighting is a local function
    // of the canvas, so the whole frame — paint, relief, everything — must shift.
    let (w, h) = (before.width as i32, before.height as i32);
    let mut worst = 0u8;
    for y in 0..h {
        for x in 0..w {
            let (sx, sy) = (x - dx, y - dy);
            if sx < 0 || sy < 0 || sx >= w || sy >= h {
                continue;
            }
            let ia = ((y * w + x) * 4) as usize;
            let ib = ((sy * w + sx) * 4) as usize;
            for c in 0..4 {
                let d = (after.pixels[ia + c] as i32 - before.pixels[ib + c] as i32).unsigned_abs()
                    as u8;
                worst = worst.max(d);
            }
        }
    }
    assert!(
        worst == 0,
        "integer translation must be exact (worst channel delta {worst})"
    );
}

/// Two whole-layer moves there and back compose to the identity: the cut drops
/// the source tiles, the paste re-creates them elsewhere, and coming home the
/// same way loses nothing.
#[test]
fn translate_there_and_back_is_identity() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    blob(&mut engine, Vec2::new(-30.0, 10.0));
    let before = engine.render_to_image();
    transform(
        &mut engine,
        Affine2::from_translation(Vec2::new(300.0, 0.0)),
    );
    let away = engine.render_to_image();
    assert!(
        !painted(&away, Vec2::new(-30.0, 10.0)),
        "the painting should have left the viewport"
    );
    transform(
        &mut engine,
        Affine2::from_translation(Vec2::new(-300.0, 0.0)),
    );
    let back = engine.render_to_image();
    assert_identical(&before, &back, "translate there and back");
}

/// A horizontal flip about the canvas origin (a half-integer-free axis on the
/// texel grid) maps texel centres to texel centres; two of them are the identity
/// through two separately committed actions.
#[test]
fn flip_twice_is_identity() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    blob(&mut engine, Vec2::new(-45.0, 5.0));
    let before = engine.render_to_image();
    let flip = Affine2::from_scale(Vec2::new(-1.0, 1.0));
    transform(&mut engine, flip);
    let flipped = engine.render_to_image();
    assert!(
        painted(&flipped, Vec2::new(45.0, 5.0)),
        "flip moved the blob"
    );
    assert!(
        !painted(&flipped, Vec2::new(-45.0, 5.0)),
        "flip cleared the source"
    );
    transform(&mut engine, flip);
    let back = engine.render_to_image();
    assert_identical(&before, &back, "flip twice");
}

// --- Behaviour: the cut, the carry, rejection, history --------------------------

#[test]
fn transform_cuts_the_source_and_lands_at_the_destination() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    blob(&mut engine, Vec2::new(-50.0, 0.0));
    select_rect(
        &mut engine,
        Vec2::new(-90.0, -40.0),
        Vec2::new(-10.0, 40.0),
        0.0,
    );
    transform(
        &mut engine,
        Affine2::from_translation(Vec2::new(100.0, 0.0)),
    );
    let img = engine.render_to_image();
    assert!(painted(&img, Vec2::new(50.0, 0.0)), "paint should arrive");
    assert!(!painted(&img, Vec2::new(-50.0, 0.0)), "paint should leave");
}

#[test]
fn moved_paint_stacks_over_what_it_lands_on() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // A green substrate at the destination; a thick red blob moved on top of it.
    paint(
        &mut engine,
        GREEN,
        24.0,
        &[Vec2::new(30.0, 0.0), Vec2::new(70.0, 0.0)],
    );
    blob(&mut engine, Vec2::new(-50.0, 0.0));
    select_rect(
        &mut engine,
        Vec2::new(-90.0, -40.0),
        Vec2::new(-10.0, 40.0),
        0.0,
    );
    transform(
        &mut engine,
        Affine2::from_translation(Vec2::new(100.0, 0.0)),
    );
    let img = engine.render_to_image();
    assert!(
        painted(&img, Vec2::new(50.0, 0.0)),
        "thick red paint should cover the green it lands on"
    );
}

#[test]
fn selection_follows_the_transform() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    select_rect(
        &mut engine,
        Vec2::new(-60.0, -40.0),
        Vec2::new(-20.0, 40.0),
        0.0,
    );
    transform(&mut engine, Affine2::from_translation(Vec2::new(80.0, 0.0)));
    assert!(
        engine.observe().has_selection,
        "the selection must survive the move"
    );
    // A stroke across the whole width may only land inside the *moved* region.
    paint(
        &mut engine,
        RED,
        12.0,
        &[Vec2::new(-80.0, 0.0), Vec2::new(80.0, 0.0)],
    );
    let img = engine.render_to_image();
    assert!(
        painted(&img, Vec2::new(40.0, 0.0)),
        "the moved selection should admit paint"
    );
    assert!(
        !painted(&img, Vec2::new(-40.0, 0.0)),
        "the old selection region must no longer admit paint"
    );
}

#[test]
fn degenerate_affine_is_refused_without_a_history_step() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    blob(&mut engine, Vec2::new(-30.0, 0.0));
    let before = engine.render_to_image();
    let undo_before = engine.observe().can_undo;
    transform(&mut engine, Affine2::from_scale(Vec2::new(0.0, 1.0)));
    transform(
        &mut engine,
        Affine2::from_translation(Vec2::new(f32::NAN, 0.0)),
    );
    let after = engine.render_to_image();
    assert_identical(&before, &after, "degenerate transform");
    assert_eq!(
        engine.observe().can_undo,
        undo_before,
        "a refused transform must not enter history"
    );
}

#[test]
fn oversized_transform_is_refused_whole() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    blob(&mut engine, Vec2::new(-30.0, 0.0));
    let before = engine.render_to_image();
    // Usable affine, but the destination would need more tiles than the cap.
    transform(&mut engine, Affine2::from_scale(Vec2::splat(1e4)));
    let after = engine.render_to_image();
    assert_identical(&before, &after, "oversized transform");
}

#[test]
fn transform_on_a_matte_layer_is_refused() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    blob(&mut engine, Vec2::new(-30.0, 0.0));
    engine.process(DocCommand::AddMatte {
        carrier: None,
        at: Place::Top,
        region: MatteRegion::OutsideRect {
            min: Vec2::new(-100.0, -100.0),
            max: Vec2::new(100.0, 100.0),
        },
        paint: Parcel::Solid(Srgb::new([0.2, 0.2, 0.2])),
    });
    let matte: LayerId = engine
        .observe()
        .layers
        .iter()
        .find(|l| l.matte.is_some())
        .expect("a matte exists")
        .id;
    let before = engine.render_to_image();
    engine.process(DocCommand::Transform {
        layer: matte,
        map: TransformMap::Affine(Affine2::from_translation(Vec2::new(50.0, 0.0))),
    });
    let after = engine.render_to_image();
    assert_identical(&before, &after, "transform aimed at a matte");
}

#[test]
fn undo_restores_the_untransformed_painting() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    blob(&mut engine, Vec2::new(-30.0, 0.0));
    select_rect(
        &mut engine,
        Vec2::new(-70.0, -40.0),
        Vec2::new(10.0, 40.0),
        6.0,
    );
    let before = engine.render_to_image();
    transform(
        &mut engine,
        Affine2::from_translation(Vec2::new(90.0, 20.0)),
    );
    engine.process(DocCommand::Undo);
    let after = engine.render_to_image();
    assert_identical(&before, &after, "undo of a transform");
}

/// A transform is six floats in the log: saving and reloading must replay to the
/// same pixels (§9's replay-equivalence, extended to the new action).
#[test]
fn save_load_reproduces_a_transform() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    blob(&mut engine, Vec2::new(-40.0, -10.0));
    select_rect(
        &mut engine,
        Vec2::new(-80.0, -50.0),
        Vec2::new(0.0, 30.0),
        8.0,
    );
    let about = Vec2::new(-40.0, -10.0);
    let rot = Affine2::from_translation(about)
        * Affine2::from_angle(0.6)
        * Affine2::from_translation(-about);
    transform(&mut engine, rot);
    let before = engine.render_to_image();

    let bytes = engine.save_bytes().expect("save");
    let Some(mut reloaded) = engine_or_skip() else {
        return;
    };
    reloaded.load_bytes(&bytes).expect("load");
    let after = reloaded.render_to_image();
    assert_identical(&before, &after, "save/load of a transform");
}

/// The preview is the commit, before the commit: `ViewCommand::PreviewTransform`
/// runs the same renderer over the same committed tiles as `DocCommand::Transform`,
/// so the previewed frame and the committed frame must be identical — the §1.3
/// live == committed invariant, extended to the transform gesture
/// (§16.6).
#[test]
fn preview_matches_the_commit() {
    use stark_engine::command::ViewCommand;
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    blob(&mut engine, Vec2::new(-40.0, -10.0));
    select_rect(
        &mut engine,
        Vec2::new(-80.0, -50.0),
        Vec2::new(0.0, 30.0),
        8.0,
    );
    let layer = engine.observe().active_layer;
    let map = TransformMap::Affine(Affine2::from_translation(Vec2::new(57.0, 23.0)));

    engine.process(ViewCommand::PreviewTransform(Some((layer, map.clone()))));
    let previewed = engine.render_to_image();
    engine.process(ViewCommand::PreviewTransform(None));
    engine.process(DocCommand::Transform { layer, map });
    let committed = engine.render_to_image();
    assert_identical(&previewed, &committed, "transform preview vs commit");
}

/// The selection hull tracks the ops that build the mask and the transforms that
/// move it — it is what the transform chrome hangs its handles on.
#[test]
fn selection_hull_follows_ops_and_transforms() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    assert_eq!(
        engine.observe().selection_hull,
        None,
        "the unrestricted selection has no hull"
    );
    select_rect(
        &mut engine,
        Vec2::new(-60.0, -40.0),
        Vec2::new(-20.0, 40.0),
        0.0,
    );
    let (lo, hi) = engine.observe().selection_hull.expect("a rect has a hull");
    assert!(lo.x <= -60.0 && hi.x >= -20.0 && lo.y <= -40.0 && hi.y >= 40.0);
    assert!(lo.x > -70.0, "hull should be near the shape, got {lo:?}");

    transform(&mut engine, Affine2::from_translation(Vec2::new(80.0, 0.0)));
    let (lo2, hi2) = engine.observe().selection_hull.expect("hull survives");
    assert!(
        (lo2.x - (lo.x + 80.0)).abs() < 1e-3 && (hi2.x - (hi.x + 80.0)).abs() < 1e-3,
        "hull should translate with the selection: {lo:?}..{hi:?} -> {lo2:?}..{hi2:?}"
    );
}

// --- The rect-scoped families: perspective (§16.8) and warp (§16.9) ------------

/// The rect the gated tests act through — generously around the blob, well
/// inside the viewport.
const GATE: (Vec2, Vec2) = (Vec2::new(-100.0, -60.0), Vec2::new(40.0, 60.0));

fn perspective(corners: [Vec2; 4]) -> TransformMap {
    TransformMap::Perspective(PerspectiveMap {
        min: GATE.0,
        max: GATE.1,
        corners,
    })
}

fn warp_identity() -> WarpMap {
    WarpMap::identity(GATE.0, GATE.1, 4, 4)
}

/// An identity perspective — corners exactly on the rect's — is a byte-for-byte
/// no-op, exactly as the affine's is: the literal-identity homography, the
/// exact combine branches and the box gate's exact interior all have to line up
/// for this to hold (§16.4, §16.8).
#[test]
fn perspective_identity_is_a_noop() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    blob(&mut engine, Vec2::new(-30.0, 0.0));
    select_rect(
        &mut engine,
        Vec2::new(-60.0, -40.0),
        Vec2::new(0.0, 40.0),
        0.0,
    );
    let before = engine.render_to_image();
    transform_map(&mut engine, perspective(rect_corners(GATE.0, GATE.1)));
    let after = engine.render_to_image();
    assert_identical(&before, &after, "identity perspective");
}

/// The warp's twin — the deviation-form subdivision makes every lattice point
/// its base, every sub-cell an exact-identity affine, and the whole action a
/// byte-for-byte no-op (§16.9).
#[test]
fn warp_identity_is_a_noop() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    blob(&mut engine, Vec2::new(-30.0, 0.0));
    select_rect(
        &mut engine,
        Vec2::new(-60.0, -40.0),
        Vec2::new(0.0, 40.0),
        0.0,
    );
    let before = engine.render_to_image();
    transform_map(&mut engine, TransformMap::Warp(warp_identity()));
    let after = engine.render_to_image();
    assert_identical(&before, &after, "identity warp");
}

/// A genuine perspective (the far edge pinched): the paint leaves its source
/// and lands where the homography says it lands — probed at the forward image
/// of the blob's centre.
#[test]
fn perspective_carries_paint_where_the_corners_point() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let c = Vec2::new(-30.0, 0.0);
    blob(&mut engine, c);
    // Pinch the top edge inward and shift everything right: a receding plane.
    let base = rect_corners(GATE.0, GATE.1);
    let corners = [
        base[0] + Vec2::new(90.0, 20.0),
        base[1] + Vec2::new(50.0, 20.0),
        base[2] + Vec2::new(70.0, 0.0),
        base[3] + Vec2::new(70.0, 0.0),
    ];
    let map = PerspectiveMap {
        min: GATE.0,
        max: GATE.1,
        corners,
    };
    let expected = map.forward().expect("usable").apply(c);
    transform_map(&mut engine, TransformMap::Perspective(map));
    let img = engine.render_to_image();
    assert!(
        painted(&img, expected),
        "paint should arrive at the homography's image {expected:?}"
    );
    assert!(!painted(&img, c), "paint should leave the source");
}

/// A warp with a dragged control point: the paint under the substrate lands where
/// the substrate says — probed at the engine's own `eval` of the blob's grid
/// fraction, which is exactly what the exact-follow gesture promises (§16.9).
#[test]
fn warp_carries_paint_with_its_surface() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let c = Vec2::new(-30.0, 0.0);
    blob(&mut engine, c);
    let before = engine.render_to_image();
    let mut map = warp_identity();
    // Drag two interior points down-right; the blob rides the bulge. Gentle
    // enough not to fold — a fold is *refused* (its own test below), and the
    // gesture's clamp never lets a hand compose one.
    map.points[5] += Vec2::new(24.0, 14.0);
    map.points[6] += Vec2::new(18.0, 8.0);
    assert!(map.usable());
    let size = GATE.1 - GATE.0;
    let t = (c - GATE.0) / size;
    let expected = map.eval(t);
    transform_map(&mut engine, TransformMap::Warp(map));
    let img = engine.render_to_image();
    assert!(
        painted(&img, expected),
        "paint should arrive at the substrate's image {expected:?}"
    );
    // A smooth warp *refills* a gently-grabbed spot from upstream paint, so
    // "the source is vacated" is not a property here (the rigid families pin
    // that); what must hold is that the painting genuinely deformed.
    let (frac, _) = diff_fraction(&before, &img);
    assert!(frac > 0.001, "the warp changed almost nothing ({frac})");
}

/// The rect gate: paint outside the map's source rect is untouched — not just
/// visually, byte for byte — however the inside is bent.
#[test]
fn paint_outside_the_rect_is_untouched() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    blob(&mut engine, Vec2::new(-30.0, 0.0));
    let bystander = Vec2::new(90.0, 60.0); // outside GATE
    blob(&mut engine, bystander);
    let before = engine.render_to_image();
    let mut map = warp_identity();
    map.points[5] += Vec2::new(35.0, -20.0);
    transform_map(&mut engine, TransformMap::Warp(map));
    let after = engine.render_to_image();
    assert!(painted(&after, bystander), "the bystander stays painted");
    // Byte-exact over the bystander's own corner of the frame (the warp's
    // image stays far from it).
    let (w, x0, y0) = (before.width, 190u32, 150u32);
    for y in y0..before.height {
        for x in x0..w {
            let i = ((y * w + x) * 4) as usize;
            assert_eq!(
                before.pixels[i..i + 4],
                after.pixels[i..i + 4],
                "bystander pixel ({x},{y}) changed under a warp that must not reach it"
            );
        }
    }
}

/// Concave quads and folded meshes are refused deterministically, leaving no
/// history step — the §16.1 rejection stance, extended to the new families.
#[test]
fn degenerate_gated_maps_are_refused_without_a_history_step() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    blob(&mut engine, Vec2::new(-30.0, 0.0));
    let before = engine.render_to_image();
    let undo_before = engine.observe().can_undo;
    // A bow-tie quad.
    let base = rect_corners(GATE.0, GATE.1);
    transform_map(
        &mut engine,
        perspective([base[1], base[0], base[2], base[3]]),
    );
    // A folded mesh.
    let mut folded = warp_identity();
    folded.points[5] = folded.points[6] + Vec2::new(60.0, 0.0);
    transform_map(&mut engine, TransformMap::Warp(folded));
    let after = engine.render_to_image();
    assert_identical(&before, &after, "degenerate perspective/warp");
    assert_eq!(
        engine.observe().can_undo,
        undo_before,
        "a refused transform must not enter history"
    );
}

/// The selection rides a perspective exactly as it rides an affine: the moved
/// region admits paint, the vacated region does not.
#[test]
fn selection_follows_a_perspective() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    select_rect(
        &mut engine,
        Vec2::new(-60.0, -40.0),
        Vec2::new(-20.0, 40.0),
        0.0,
    );
    // A pure shift expressed as corners (the parallelogram tier).
    let base = rect_corners(GATE.0, GATE.1);
    transform_map(
        &mut engine,
        perspective(base.map(|c| c + Vec2::new(80.0, 0.0))),
    );
    assert!(
        engine.observe().has_selection,
        "the selection must survive the move"
    );
    paint(
        &mut engine,
        RED,
        12.0,
        &[Vec2::new(-80.0, 0.0), Vec2::new(80.0, 0.0)],
    );
    let img = engine.render_to_image();
    assert!(
        painted(&img, Vec2::new(40.0, 0.0)),
        "the moved selection should admit paint"
    );
    assert!(
        !painted(&img, Vec2::new(-40.0, 0.0)),
        "the old selection region must no longer admit paint"
    );
}

/// Undo restores the pre-warp painting — the patch machinery covers the new
/// action kinds through the same tile-diff + selection restore.
#[test]
fn undo_restores_an_unwarped_painting() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    blob(&mut engine, Vec2::new(-30.0, 0.0));
    select_rect(
        &mut engine,
        Vec2::new(-70.0, -40.0),
        Vec2::new(10.0, 40.0),
        6.0,
    );
    let before = engine.render_to_image();
    let mut map = warp_identity();
    map.points[9] += Vec2::new(-10.0, 18.0);
    map.points[10] += Vec2::new(14.0, 10.0);
    // The warp must actually commit, or the undo below would undo the
    // *selection* and this test would measure the wrong thing.
    assert!(map.usable());
    transform_map(&mut engine, TransformMap::Warp(map));
    engine.process(DocCommand::Undo);
    let after = engine.render_to_image();
    assert_identical(&before, &after, "undo of a warp");
}

/// Both new actions are a few floats in the log: saving and reloading must
/// replay to the same pixels (§9's replay-equivalence, extended).
#[test]
fn save_load_reproduces_perspective_and_warp() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    blob(&mut engine, Vec2::new(-40.0, -10.0));
    select_rect(
        &mut engine,
        Vec2::new(-80.0, -50.0),
        Vec2::new(0.0, 30.0),
        8.0,
    );
    let base = rect_corners(GATE.0, GATE.1);
    transform_map(
        &mut engine,
        perspective([
            base[0] + Vec2::new(30.0, 12.0),
            base[1] + Vec2::new(10.0, 8.0),
            base[2] + Vec2::new(18.0, -4.0),
            base[3] + Vec2::new(22.0, 2.0),
        ]),
    );
    let mut map = warp_identity();
    map.points[6] += Vec2::new(-12.0, 16.0);
    transform_map(&mut engine, TransformMap::Warp(map));
    let before = engine.render_to_image();

    let bytes = engine.save_bytes().expect("save");
    let Some(mut reloaded) = engine_or_skip() else {
        return;
    };
    reloaded.load_bytes(&bytes).expect("load");
    let after = reloaded.render_to_image();
    assert_identical(&before, &after, "save/load of a perspective + warp");
}

/// `preview == committed`, extended to the gated families: the preview runs the
/// same gated renderer over the same committed tiles as the commit.
#[test]
fn gated_preview_matches_the_commit() {
    use stark_engine::command::ViewCommand;
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    blob(&mut engine, Vec2::new(-40.0, -10.0));
    select_rect(
        &mut engine,
        Vec2::new(-80.0, -50.0),
        Vec2::new(0.0, 30.0),
        8.0,
    );
    let layer = engine.observe().active_layer;
    let mut wm = warp_identity();
    wm.points[5] += Vec2::new(26.0, -14.0);
    let map = TransformMap::Warp(wm);

    engine.process(ViewCommand::PreviewTransform(Some((layer, map.clone()))));
    let previewed = engine.render_to_image();
    engine.process(ViewCommand::PreviewTransform(None));
    engine.process(DocCommand::Transform { layer, map });
    let committed = engine.render_to_image();
    assert_identical(&previewed, &committed, "gated preview vs commit");
}

/// The one golden: genuinely resampled output — a rotation about the blob's
/// centre — pinning the bilinear quality, the seam-free tiling of the quads, and
/// the carried mask's outline (§9).
#[test]
fn golden_rotate() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    blob(&mut engine, Vec2::new(-20.0, 0.0));
    select_rect(
        &mut engine,
        Vec2::new(-60.0, -40.0),
        Vec2::new(20.0, 40.0),
        3.0,
    );
    let about = Vec2::new(-20.0, 0.0);
    let rot = Affine2::from_translation(about)
        * Affine2::from_angle(std::f32::consts::FRAC_PI_6)
        * Affine2::from_translation(-about);
    transform(&mut engine, rot);
    let img = engine.render_to_image();
    assert_golden("transform_rotate", &img, 3);
}

/// A second golden for the gated families: a receding perspective and a bent
/// mesh over the same blob, pinning the inverse-homography and
/// inverse-bilinear taps, the box gate's ramp, and the unioned mask outline.
#[test]
fn golden_perspective_and_warp() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    blob(&mut engine, Vec2::new(-30.0, 0.0));
    select_rect(
        &mut engine,
        Vec2::new(-60.0, -40.0),
        Vec2::new(0.0, 40.0),
        3.0,
    );
    let base = rect_corners(GATE.0, GATE.1);
    transform_map(
        &mut engine,
        perspective([
            base[0] + Vec2::new(28.0, 14.0),
            base[1] + Vec2::new(-6.0, 10.0),
            base[2] + Vec2::new(8.0, -2.0),
            base[3] + Vec2::new(12.0, 4.0),
        ]),
    );
    let mut map = warp_identity();
    map.points[5] += Vec2::new(18.0, -22.0);
    map.points[10] += Vec2::new(-10.0, 12.0);
    transform_map(&mut engine, TransformMap::Warp(map));
    let img = engine.render_to_image();
    assert_golden("transform_perspective_warp", &img, 3);
}
