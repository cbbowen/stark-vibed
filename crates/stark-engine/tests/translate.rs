//! The layer frame (§14.12) and the float (§16.12): moving a layer is a
//! property write, and dragging a selection is a cut paid once.
//!
//! The claims divide the way the design does. The **field** must composite
//! exactly as the baked move it replaces — pinned against `Transform`, whose
//! integer-translation exactness (§16.4) is already law — while touching no
//! tile. **Painting** on a translated layer must land under the hand, with the
//! mask gating where the hand sees it, which is the whole of what the actions'
//! `frame` offset exists to arrange. The **float** must leave the picture
//! unchanged (§14.11.1's law run backwards) and, dragged and merged back, come
//! to the same picture the old cut/paste path produced.

mod common;

use common::*;
use stark_engine::RgbaImage;
use stark_engine::command::{DocCommand, ViewCommand};
use stark_model::document::{SelectionMode, SelectionOp, SelectionShape, TransformMap};
use stark_model::geom::{Affine2, IVec2, Vec2};

const RED: [f32; 3] = [1.0, 0.0, 0.0];
const GREEN: [f32; 3] = [0.1, 0.8, 0.2];

/// A compact red blob centred at `c` — two crossing strokes, so the lighting has
/// height variation to bite on and a seam would show.
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

fn translate_active(engine: &mut stark_engine::Engine, to: IVec2) {
    let layer = engine.observe().active_layer;
    engine.process(DocCommand::TranslateLayer { layer, to });
}

fn assert_identical(a: &RgbaImage, b: &RgbaImage, what: &str) {
    assert_within(a, b, 0, what);
}

/// The two images agree within `levels` per channel — 0 is byte-exact.
fn assert_within(a: &RgbaImage, b: &RgbaImage, levels: u8, what: &str) {
    let (frac, worst) = diff_fraction(a, b);
    if worst <= levels {
        return;
    }
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
        "{what}: images differ past {levels} levels (worst channel delta {worst}, {:.4}% of pixels)\nfirst sites: {}",
        frac * 100.0,
        sites.join("; ")
    );
}

// --- The field composites as the baked move (§14.12) ------------------------

/// **The core claim**: a layer whose frame is set to `D` renders bit-for-bit as
/// the same layer with `D` baked through `Transform`'s whole-pixel translation —
/// the path §16.4 already pins exact. One is a property write, the other rewrote
/// every tile; the picture cannot tell them apart.
#[test]
fn a_translated_layer_renders_as_the_baked_move() {
    let Some(mut a) = engine_or_skip() else {
        return;
    };
    let Some(mut b) = engine_or_skip() else {
        return;
    };
    let d = IVec2::new(37, -13);
    blob(&mut a, Vec2::new(-20.0, 8.0));
    blob(&mut b, Vec2::new(-20.0, 8.0));

    translate_active(&mut a, d);

    let layer = b.observe().active_layer;
    b.process(DocCommand::Transform {
        layer,
        map: TransformMap::Affine(Affine2::from_translation(d.as_vec2())),
    });

    assert_identical(
        &a.render_to_image(),
        &b.render_to_image(),
        "field vs baked move",
    );
}

/// Moving a layer moves **no tile**: the map — and with it the content revision
/// every thumbnail is keyed on (§14.6) — rides through untouched, which is the
/// entire feature.
#[test]
fn a_translation_touches_no_tile() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    blob(&mut engine, Vec2::ZERO);
    let layer = engine.observe().active_layer;
    let before = engine.observe();
    let row = |o: &stark_engine::ObservableState| {
        o.layers
            .iter()
            .find(|l| l.id == layer)
            .expect("the active layer is on the roster")
            .clone()
    };
    translate_active(&mut engine, IVec2::new(300, 40));
    let after = engine.observe();
    assert_eq!(
        row(&before).content_revision,
        row(&after).content_revision,
        "a translate minted a fresh tile map",
    );
    assert_eq!(row(&after).translation, IVec2::new(300, 40));
}

/// The drag's preview is the commit's own fold: showing a frame and committing
/// it produce the same picture (§1.3's invariant for the cheapest preview yet).
#[test]
fn a_translation_previews_as_its_commit() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    blob(&mut engine, Vec2::new(10.0, -30.0));
    let layer = engine.observe().active_layer;
    let to = IVec2::new(-90, 55);
    engine.process(ViewCommand::PreviewTranslate(Some((layer, to))));
    let previewed = engine.render_to_image();
    engine.process(DocCommand::TranslateLayer { layer, to });
    assert_identical(
        &previewed,
        &engine.render_to_image(),
        "preview == committed",
    );
}

/// Undo restores the frame — a property write, so it is exact and re-renders
/// nothing.
#[test]
fn undo_restores_a_translation() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    blob(&mut engine, Vec2::new(-40.0, 0.0));
    let base = engine.render_to_image();
    translate_active(&mut engine, IVec2::new(120, 120));
    engine.process(DocCommand::Undo);
    assert_identical(&base, &engine.render_to_image(), "undo of a translate");
    engine.process(DocCommand::Redo);
    translate_active(&mut engine, IVec2::ZERO);
    assert_identical(&base, &engine.render_to_image(), "moved back by hand");
}

// --- Painting in a frame (§14.12's `frame` offset) --------------------------

/// Per-channel levels a stroke solved in one frame may sit from the same stroke
/// solved in another. The record's geometry is exact — whole-pixel offsets are
/// exact in `f32`, so every mark lands in the same texel — but the sweep's
/// coverage is *solved* in `f32` at the shifted magnitudes, where the same
/// arithmetic can round an ulp apart, and the f16 tile store turns a sub-half-ulp
/// disagreement into a stored bit (the swept interior's known speckle; measured
/// worst delta 2). What these tests pin is that nothing *shifts*: a grid
/// confusion would land whole texels of paint elsewhere, orders past this band.
const FRAME_SOLVE_LEVELS: u8 = 2;

/// A stroke on a translated layer lands **under the hand**: painting at `P` on a
/// layer whose frame sits at `D` shows what painting at `P` on an untranslated
/// layer shows — same texels, within [`FRAME_SOLVE_LEVELS`]. The record's path
/// is converted at the press and the compositor adds the frame back.
#[test]
fn painting_on_a_translated_layer_lands_under_the_hand() {
    let Some(mut a) = engine_or_skip() else {
        return;
    };
    let Some(mut b) = engine_or_skip() else {
        return;
    };
    // Not tile-aligned, so any confusion of the two grids would shear the stroke
    // across a seam rather than landing it whole.
    translate_active(&mut a, IVec2::new(37, -13));
    let line = [Vec2::new(-80.0, 0.0), Vec2::new(80.0, 0.0)];
    paint(&mut a, GREEN, 12.0, &line);
    paint(&mut b, GREEN, 12.0, &line);
    assert_within(
        &a.render_to_image(),
        &b.render_to_image(),
        FRAME_SOLVE_LEVELS,
        "a stroke in a frame",
    );
}

/// The author's mask gates a framed stroke **where the hand sees it**: the same
/// selection over the same stroke reads identically whatever the layer's frame,
/// because the mask is brought into the frame by the record's own offset — the
/// exact integer shift, so the *mask's* half adds nothing past the stroke's own
/// solve band ([`FRAME_SOLVE_LEVELS`]).
#[test]
fn a_selection_gates_a_framed_stroke_where_the_hand_sees_it() {
    let Some(mut a) = engine_or_skip() else {
        return;
    };
    let Some(mut b) = engine_or_skip() else {
        return;
    };
    let region = (Vec2::new(-40.0, -40.0), Vec2::new(40.0, 40.0));
    translate_active(&mut a, IVec2::new(37, -13));
    for e in [&mut a, &mut b] {
        select_rect(e, region.0, region.1, 6.0);
        paint(
            e,
            GREEN,
            12.0,
            &[Vec2::new(-80.0, 0.0), Vec2::new(80.0, 0.0)],
        );
    }
    assert_within(
        &a.render_to_image(),
        &b.render_to_image(),
        FRAME_SOLVE_LEVELS,
        "a gated stroke in a frame",
    );
}

// --- The float (§16.12) -----------------------------------------------------

/// Floating a selection leaves the picture where it stood: the cut is §16.2's
/// lift and the child stacks back by the merge law run backwards (§14.11.1).
/// Boundary texels are genuinely recomputed through the coverage ramp, so the
/// bound is the ramp's single display level; and the selection is **consumed**,
/// so what the float is compared against is the canvas before anything was
/// selected — no outline on either side.
#[test]
fn a_float_leaves_the_picture_unchanged() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    blob(&mut engine, Vec2::new(-10.0, 5.0));
    let base = engine.render_to_image();
    let source = engine.observe().active_layer;
    select_rect(
        &mut engine,
        Vec2::new(-40.0, -60.0),
        Vec2::new(15.0, 60.0),
        4.0,
    );
    engine.process(DocCommand::FloatSelection { layer: source });

    let o = engine.observe();
    assert!(!o.has_selection, "the float consumes the selection");
    let child = o.active_layer;
    assert_ne!(child, source, "the float armed the child");
    let row = o
        .layers
        .iter()
        .find(|l| l.id == child)
        .expect("the child is on the roster");
    assert_eq!(row.carrier, Some(source), "carried by the source");
    assert_eq!(
        row.translation,
        IVec2::ZERO,
        "an unframed source cuts at zero"
    );

    assert_within(&base, &engine.render_to_image(), 1, "the float's picture");
}

/// The float from a **translated** source: the same picture-preservation claim,
/// with the cut's gates fed by the mask brought into a nonzero, non-tile-aligned
/// frame — the §14.12.2 shift's pixel witness (the audit in `tests/footprint.rs`
/// drives this path too, but for footprint containment, not pixels).
#[test]
fn a_float_from_a_translated_source_leaves_the_picture_unchanged() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    blob(&mut engine, Vec2::new(-10.0, 5.0));
    let source = engine.observe().active_layer;
    translate_active(&mut engine, IVec2::new(37, -13));
    let base = engine.render_to_image();
    // Over where the paint now *shows* — the mask is drawn on the canvas, and
    // bringing it into the frame is exactly what is under test.
    select_rect(
        &mut engine,
        Vec2::new(-10.0, -70.0),
        Vec2::new(55.0, 50.0),
        4.0,
    );
    engine.process(DocCommand::FloatSelection { layer: source });
    let o = engine.observe();
    let child = o.active_layer;
    assert_ne!(child, source, "the float was made");
    let row = o
        .layers
        .iter()
        .find(|l| l.id == child)
        .expect("the child is on the roster");
    assert_eq!(
        row.translation,
        IVec2::new(37, -13),
        "the child stands at the frame the cut was made in",
    );
    assert_within(
        &base,
        &engine.render_to_image(),
        1,
        "a framed float's picture",
    );
}

/// A perspective on a translated layer is a perspective on the same picture:
/// `a_translated_layer_renders_as_the_baked_move`, extended through a gated
/// resample. Both engines paint the blob identically at frame zero; `a` then
/// moves it with the **field** and `b` with the baked cut/paste (exact,
/// §16.4), so their canvases are bit-equal — and the same canvas-space
/// perspective under the same mask must keep them so. `a`'s paint side runs
/// the map conjugated into its frame over unshifted tiles, `b`'s the plain
/// map over shifted ones; the taps land at the same canvas positions over the
/// same texel values, within the conjugation's own ulps
/// ([`FRAME_SOLVE_LEVELS`]).
#[test]
fn a_perspective_in_a_frame_lands_under_the_hand() {
    let Some(mut a) = engine_or_skip() else {
        return;
    };
    let Some(mut b) = engine_or_skip() else {
        return;
    };
    let d = IVec2::new(37, -13);
    blob(&mut a, Vec2::new(-10.0, 5.0));
    blob(&mut b, Vec2::new(-10.0, 5.0));
    translate_active(&mut a, d);
    let layer = b.observe().active_layer;
    b.process(DocCommand::Transform {
        layer,
        map: TransformMap::Affine(Affine2::from_translation(d.as_vec2())),
    });

    let map = {
        use stark_model::document::{PerspectiveMap, rect_corners};
        let (lo, hi) = (Vec2::new(-20.0, -60.0), Vec2::new(90.0, 50.0));
        let mut corners = rect_corners(lo, hi);
        corners[1] += Vec2::new(0.0, 9.0);
        corners[3] += Vec2::new(0.0, -7.0);
        TransformMap::Perspective(PerspectiveMap {
            min: lo,
            max: hi,
            corners,
        })
    };
    for e in [&mut a, &mut b] {
        select_rect(e, Vec2::new(0.0, -40.0), Vec2::new(70.0, 30.0), 4.0);
        let layer = e.observe().active_layer;
        e.process(DocCommand::Transform {
            layer,
            map: map.clone(),
        });
    }
    assert_within(
        &a.render_to_image(),
        &b.render_to_image(),
        FRAME_SOLVE_LEVELS,
        "a perspective in a frame",
    );
}

/// A float with nothing selected on the layer — or nothing selected at all — is
/// refused before an action is spent, and refused identically by `apply`.
#[test]
fn a_float_of_nothing_floats_nothing() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    blob(&mut engine, Vec2::ZERO);
    let layer = engine.observe().active_layer;
    let rows = engine.observe().layers.len();
    // Nothing selected at all.
    engine.process(DocCommand::FloatSelection { layer });
    assert_eq!(
        engine.observe().layers.len(),
        rows,
        "universal mask refused"
    );
    // A selection that holds none of this layer's paint.
    select_rect(
        &mut engine,
        Vec2::new(4000.0, 4000.0),
        Vec2::new(4100.0, 4100.0),
        0.0,
    );
    engine.process(DocCommand::FloatSelection { layer });
    assert_eq!(engine.observe().layers.len(), rows, "empty cut refused");
}

/// The whole gesture, end to end: float, drag the child, merge it back down —
/// against the picture the old path (`Transform`'s cut/paste) produces from the
/// same selection and the same move. The merge across frames bakes the child's
/// offset through the same exactness invariant the reference rides, so the two
/// pictures agree within the ramp's recompute band.
#[test]
fn a_float_dragged_and_merged_back_is_the_baked_move() {
    let Some(mut a) = engine_or_skip() else {
        return;
    };
    let Some(mut b) = engine_or_skip() else {
        return;
    };
    let d = IVec2::new(64, 23);
    let region = (Vec2::new(-40.0, -60.0), Vec2::new(15.0, 60.0));
    for e in [&mut a, &mut b] {
        blob(e, Vec2::new(-10.0, 5.0));
        select_rect(e, region.0, region.1, 4.0);
    }

    // The new path: float, drag the child, merge it home.
    let source = a.observe().active_layer;
    a.process(DocCommand::FloatSelection { layer: source });
    let child = a.observe().active_layer;
    a.process(DocCommand::TranslateLayer {
        layer: child,
        to: d,
    });
    a.process(DocCommand::MergeLayerDown(child));

    // The old path: one cut/paste, then deselect to match the consumed mask.
    let layer = b.observe().active_layer;
    b.process(DocCommand::Transform {
        layer,
        map: TransformMap::Affine(Affine2::from_translation(d.as_vec2())),
    });
    b.process(DocCommand::Select(SelectionOp::new(
        SelectionMode::Replace,
        SelectionShape::All,
        0.0,
    )));

    assert_eq!(
        a.observe().layers.len(),
        b.observe().layers.len(),
        "the merge folded the float away",
    );
    assert_within(
        &a.render_to_image(),
        &b.render_to_image(),
        2,
        "float+drag+merge vs cut/paste",
    );
}

/// Undoing a float restores both halves at once: the paint back on its source,
/// and the author's selection back in force.
#[test]
fn undo_restores_a_float_and_its_selection() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    blob(&mut engine, Vec2::new(-10.0, 5.0));
    let source = engine.observe().active_layer;
    select_rect(
        &mut engine,
        Vec2::new(-40.0, -60.0),
        Vec2::new(15.0, 60.0),
        4.0,
    );
    let selected = engine.render_to_image();
    engine.process(DocCommand::FloatSelection { layer: source });
    engine.process(DocCommand::Undo);
    let o = engine.observe();
    assert!(o.has_selection, "the selection came back with the paint");
    assert_eq!(
        o.layers
            .iter()
            .filter(|l| l.carrier == Some(source))
            .count(),
        0
    );
    assert_identical(&selected, &engine.render_to_image(), "undo of a float");
}
