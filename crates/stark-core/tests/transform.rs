//! Affine transform of the selected paint (TRANSFORM_DESIGN.md).
//!
//! The exactness invariants carry most of the weight, because they are what the
//! cut/paste laws were chosen for (TRANSFORM_DESIGN.md §2–§4): the identity
//! transform is a byte-for-byte no-op at any feather, integer translations move
//! the painting without a texel of resampling loss, and involutions (a flip, a
//! there-and-back move of the whole layer) compose to the identity through
//! separate actions. Behavioural probes cover the cut, the carry of the
//! selection, deterministic rejection, undo, and save/load replay; one golden
//! pins genuinely-resampled output (a rotation).

mod common;

use common::*;
use stark_core::RgbaImage;
use stark_core::colorspace::ColorSpaceId;
use stark_core::command::DocCommand;
use stark_core::document::{LayerId, MatteRegion, SelectionMode, SelectionOp, SelectionShape};
use stark_core::geom::{Affine2, Vec2};

const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const GREEN: [f32; 4] = [0.1, 0.8, 0.2, 1.0];

/// A pixel's screen position for a canvas point, under the tests' identity view.
fn screen_of(canvas: Vec2) -> (u32, u32) {
    let half = Vec2::new(SIZE.width as f32, SIZE.height as f32) * 0.5;
    let p = canvas + half;
    (p.x as u32, p.y as u32)
}

/// Whether the pixel at a canvas point reads as red paint rather than bare paper.
fn is_painted(img: &RgbaImage, canvas: Vec2) -> bool {
    let (x, y) = screen_of(canvas);
    let i = ((y * img.width + x) * 4) as usize;
    let (r, g, b) = (
        img.pixels[i] as i32,
        img.pixels[i + 1] as i32,
        img.pixels[i + 2] as i32,
    );
    r - g > 40 && r - b > 40
}

/// A compact red blob centred at `c` — two short crossing strokes, so it has
/// height variation for the lighting to bite on.
fn blob(engine: &mut stark_core::Engine, c: Vec2) {
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

fn select_rect(engine: &mut stark_core::Engine, min: Vec2, max: Vec2, feather: f32) {
    engine.process(DocCommand::Select(SelectionOp::new(
        SelectionMode::Replace,
        SelectionShape::rect_from_corners(min, max),
        feather,
    )));
}

fn transform(engine: &mut stark_core::Engine, affine: Affine2) {
    let layer = engine.observe().active_layer;
    engine.process(DocCommand::Transform { layer, affine });
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

// --- The exactness invariants (TRANSFORM_DESIGN.md §4) ---------------------------

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
/// chosen precisely so this holds (TRANSFORM_DESIGN.md §2). Inside the ramp the
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
    assert_identical(&before, &after, "mixbox identity transform");
}

/// An integer translation resamples at exact texel centres, so the painting moves
/// with no loss at all: the after-image is the before-image shifted, byte for byte.
#[test]
fn integer_translation_is_exact() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
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
        !is_painted(&away, Vec2::new(-30.0, 10.0)),
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
        is_painted(&flipped, Vec2::new(45.0, 5.0)),
        "flip moved the blob"
    );
    assert!(
        !is_painted(&flipped, Vec2::new(-45.0, 5.0)),
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
    assert!(
        is_painted(&img, Vec2::new(50.0, 0.0)),
        "paint should arrive"
    );
    assert!(
        !is_painted(&img, Vec2::new(-50.0, 0.0)),
        "paint should leave"
    );
}

#[test]
fn moved_paint_stacks_over_what_it_lands_on() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // A green ground at the destination; a thick red blob moved on top of it.
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
        is_painted(&img, Vec2::new(50.0, 0.0)),
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
        is_painted(&img, Vec2::new(40.0, 0.0)),
        "the moved selection should admit paint"
    );
    assert!(
        !is_painted(&img, Vec2::new(-40.0, 0.0)),
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
        above: None,
        region: MatteRegion::OutsideRect {
            min: Vec2::new(-100.0, -100.0),
            max: Vec2::new(100.0, 100.0),
        },
        color: [0.2, 0.2, 0.2],
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
        affine: Affine2::from_translation(Vec2::new(50.0, 0.0)),
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
/// same pixels (DESIGN.md §9's replay-equivalence, extended to the new action).
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
/// (TRANSFORM_DESIGN.md §6).
#[test]
fn preview_matches_the_commit() {
    use stark_core::command::ViewCommand;
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
    let affine = Affine2::from_translation(Vec2::new(57.0, 23.0));

    engine.process(ViewCommand::PreviewTransform(Some((layer, affine))));
    let previewed = engine.render_to_image();
    engine.process(ViewCommand::PreviewTransform(None));
    engine.process(DocCommand::Transform { layer, affine });
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

/// The one golden: genuinely resampled output — a rotation about the blob's
/// centre — pinning the bilinear quality, the seam-free tiling of the quads, and
/// the carried mask's outline (DESIGN.md §9).
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
