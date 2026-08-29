//! Matte layers — the frame (§15.2, §15.4).
//!
//! A matte is a layer whose content is a *region and a fill* rather than a map of
//! tiles. These cover what makes it a layer (it composites in stack order, it
//! takes layer opacity, it undoes) and the two things the compositing model is
//! most likely to get wrong: that it is visible at all (§15.4.1 — the media pass
//! derives visibility from height, not composited alpha), and that an opaque one
//! *erases* the relief beneath it rather than letting underlying impasto emboss
//! ghost ridges through it (§15.4.2).
//!
//! The `previews_without_logging` tests are here together rather than split by
//! subject: the frame drag, the frame's color and the substrate's are one
//! mechanism — an unlogged stand-in document while the pointer is down, one action
//! on release — and they are worth reading side by side.

mod common;

use common::*;
use stark_engine::command::{DocCommand, PeerCommand, ViewCommand};
use stark_engine::{Engine, RgbaImage};
use stark_model::Srgb;
use stark_model::document::{LayerId, MatteRegion, Parcel, Place};
use stark_model::geom::Vec2;

const RED: [f32; 3] = [0.85, 0.1, 0.1];
const BLACK: [f32; 3] = [0.0, 0.0, 0.0];

/// A frame around the middle of the 256² viewport. The canvas origin is at the
/// viewport centre, so this is a centred 100×100 hole.
const HOLE: MatteRegion = MatteRegion::OutsideRect {
    min: Vec2::new(-50.0, -50.0),
    max: Vec2::new(50.0, 50.0),
};

/// A stroke long enough to run well outside the frame on both sides.
const WIDE_STROKE: &[Vec2] = &[Vec2::new(-110.0, 0.0), Vec2::new(110.0, 0.0)];

fn at(img: &RgbaImage, x: u32, y: u32) -> [u8; 4] {
    img.pixel(x, y)
}
/// Well outside the 100×100 hole, but still on the stroke.
fn outside(img: &RgbaImage) -> [u8; 4] {
    at(img, 12, img.height / 2)
}
fn is_dark(c: [u8; 4]) -> bool {
    c[0] < 60 && c[1] < 60 && c[2] < 60
}

fn add_frame(engine: &mut Engine) {
    engine.process(DocCommand::AddMatte {
        carrier: None,
        at: Place::Top,
        region: HOLE,
        paint: Parcel::Solid(Srgb::new(BLACK)),
    });
}

/// The core claim: a frame matte covers everything outside its rect and nothing
/// inside it. Also the §15.4.1 regression — a matte that failed to write the aux
/// (thickness) target would be perfectly invisible, and `outside` would stay red.
#[test]
fn frame_covers_outside_and_spares_inside() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint(&mut engine, RED, 30.0, WIDE_STROKE);
    let before = engine.render_to_image();
    assert!(red_dominant(center(&before)), "stroke should cross centre");
    assert!(
        red_dominant(outside(&before)),
        "stroke should reach outside the frame"
    );

    add_frame(&mut engine);
    let after = engine.render_to_image();
    assert!(
        is_dark(outside(&after)),
        "outside the frame should be covered by the matte, got {:?}",
        outside(&after)
    );
    assert!(
        red_dominant(center(&after)),
        "inside the frame should be untouched, got {:?}",
        center(&after)
    );
}

/// §15.4.2 — the ghost-relief regression.
///
/// The aux (height) target's blend for a matte must be `over`, not the color
/// space's additive. Additive would keep the height of the paint *underneath*,
/// and the media pass — which builds its normal from the height field — would
/// emboss that paint's impasto through an opaque mat board.
///
/// Formulated as: an opaque matte over a heavy stroke must render the same as the
/// same matte over bare canvas. The comparison is on the *lit* image and the
/// stroke is thick, so a surviving height field shows up as shading differences.
#[test]
fn opaque_matte_erases_relief_beneath() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // A fat, heavy stroke right through the region the matte will cover.
    paint(&mut engine, RED, 45.0, WIDE_STROKE);
    add_frame(&mut engine);
    let over_paint = engine.render_to_image();

    let Some(mut bare) = engine_or_skip() else {
        return;
    };
    add_frame(&mut bare);
    let over_nothing = bare.render_to_image();

    // Compare only the matte-covered band, well clear of the frame's lit edge
    // (a genuine height cliff, present in both images but sensitive to sampling).
    let (w, h) = (over_paint.width, over_paint.height);
    let mut worst = 0i32;
    for y in (h / 2 - 20)..(h / 2 + 20) {
        for x in 4..40 {
            let (a, b) = (at(&over_paint, x, y), at(&over_nothing, x, y));
            for c in 0..3 {
                worst = worst.max((a[c] as i32 - b[c] as i32).abs());
            }
        }
    }
    assert!(
        worst <= 2,
        "impasto beneath an opaque matte is embossing through it \
         (max channel difference {worst}); the matte's aux blend must be `over`, \
         not additive — §15.4.2"
    );
    let _ = w;
}

/// A matte takes layer opacity like any other layer — this is what makes the
/// crop scrim need no machinery of its own (§15.3).
#[test]
fn matte_honors_layer_opacity_and_visibility() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint(&mut engine, RED, 30.0, WIDE_STROKE);
    add_frame(&mut engine);
    let matte_id = engine
        .observe()
        .layers
        .last()
        .expect("matte layer present")
        .id;
    let opaque = engine.render_to_image();

    engine.process(DocCommand::SetLayerVisible(matte_id, false));
    let hidden = engine.render_to_image();
    assert!(
        red_dominant(outside(&hidden)),
        "a hidden matte should cover nothing"
    );

    engine.process(DocCommand::SetLayerVisible(matte_id, true));
    engine.process(DocCommand::SetLayerOpacity(matte_id, 0.5));
    let half = engine.render_to_image();
    let (o, hf, hd) = (outside(&opaque), outside(&half), outside(&hidden));
    // Compared on total brightness, not per channel: against a red stroke the
    // blue channel is already floored at both ends, so it has no range to be
    // "between" in. Monotonic only — the slab law makes opacity markedly
    // non-linear (§15.4.3), so this deliberately does not assert a midpoint.
    let lum = |c: [u8; 4]| c[0] as i32 + c[1] as i32 + c[2] as i32;
    assert!(
        lum(o) < lum(hf) && lum(hf) < lum(hd),
        "a half-opacity matte should sit between opaque and hidden: \
         opaque {o:?}, half {hf:?}, hidden {hd:?}"
    );
}

/// A matte composites at its own place in the stack, so one *below* the paint
/// covers nothing of it (§15.4.4). Guards the ordered item walk
/// against being flattened back into "all tiles, then all mattes".
#[test]
fn matte_below_paint_does_not_cover_it() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint(&mut engine, RED, 30.0, WIDE_STROKE);
    // Below the root layer: `above: None` appends on top, so insert under the
    // root by moving it after the fact.
    add_frame(&mut engine);
    let matte_id = engine.observe().layers.last().expect("matte").id;
    engine.process(DocCommand::MoveLayer {
        carrier: None,
        id: LayerId::ROOT,
        at: Place::Above(matte_id),
    });
    let img = engine.render_to_image();
    assert!(
        red_dominant(outside(&img)),
        "paint above a matte must not be covered by it, got {:?}",
        outside(&img)
    );
}

/// Bounds are paint-only: a matte covers the infinite plane, and counting it
/// would make `bounds` unbounded — breaking "frame to content" and export's
/// no-frame fallback (§15.6).
#[test]
fn matte_does_not_extend_canvas_bounds() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint(&mut engine, RED, 20.0, &[Vec2::ZERO, Vec2::new(10.0, 0.0)]);
    let painted = engine.observe().bounds.tile_range();
    assert!(painted.is_some(), "the stroke should populate tiles");

    add_frame(&mut engine);
    assert_eq!(
        engine.observe().bounds.tile_range(),
        painted,
        "a matte must not contribute to canvas bounds"
    );
}

/// A matte may be **selected** like any other layer — that is what lets the
/// frontend have one selection concept rather than a paint target plus a separate
/// frame focus (§15.7) — but a stroke aimed at it draws nothing,
/// refused in `apply` rather than swallowed or magically rasterized.
///
/// Enforced in the engine, not the frontend, so replay and peers agree about a log
/// that names one.
#[test]
fn a_matte_can_be_selected_but_takes_no_paint() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint(&mut engine, RED, 30.0, WIDE_STROKE);
    add_frame(&mut engine);
    let matte_id = engine.observe().layers.last().expect("matte").id;

    // Adding a matte does not *steal* the painting target...
    assert_ne!(
        engine.observe().active_layer,
        matte_id,
        "adding a frame should not move the selection on its own"
    );
    // ...but selecting one is allowed, and reported back.
    engine.process(PeerCommand::SetActiveLayer(matte_id));
    assert_eq!(
        engine.observe().active_layer,
        matte_id,
        "a matte must be selectable, so the frontend needs only one selection"
    );
    assert!(
        !engine
            .observe()
            .layers
            .iter()
            .find(|l| l.id == matte_id)
            .expect("matte")
            .is_paintable(),
        "and it must report that it takes no paint"
    );

    // Painting on it does nothing at all — no panic, no silent landing elsewhere.
    let before = engine.render_to_image();
    paint(
        &mut engine,
        RED,
        30.0,
        &[Vec2::new(-30.0, 40.0), Vec2::new(30.0, 40.0)],
    );
    let after = engine.render_to_image();
    assert!(
        images_match(&before, &after, 1),
        "a stroke on a matte must change nothing"
    );

    // Selecting a paint layer again resumes painting, so nothing is stuck.
    engine.process(PeerCommand::SetActiveLayer(LayerId::ROOT));
    paint(
        &mut engine,
        RED,
        30.0,
        &[Vec2::new(-30.0, 40.0), Vec2::new(30.0, 40.0)],
    );
    assert!(
        !images_match(&before, &engine.render_to_image(), 1),
        "selecting a paint layer again must resume painting"
    );
}

/// A frame-handle drag previews live but logs once (§15.7): the
/// canvas follows the pointer, and the whole drag costs one undo step.
#[test]
fn dragging_a_frame_previews_without_logging() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint(&mut engine, RED, 30.0, WIDE_STROKE);
    add_frame(&mut engine);
    let matte_id = engine.observe().layers.last().expect("matte").id;
    let framed = engine.render_to_image();

    // Three "pointer moves" of a drag that shrinks the hole.
    for w in [45.0f32, 40.0, 35.0] {
        engine.process(ViewCommand::PreviewMatteRect(Some((
            matte_id,
            Vec2::new(-w, -w),
            Vec2::new(w, w),
        ))));
    }
    let dragging = engine.render_to_image();
    assert!(
        !images_match(&framed, &dragging, 4),
        "the preview should move the frame on screen"
    );
    // `observe` reports the previewed rect, which is what keeps the handles under
    // the pointer instead of a frame behind on the committed value.
    let previewed = engine.observe().layers.last().and_then(|l| l.matte.clone());
    assert_eq!(previewed.and_then(|m| m.dims()).map(|d| d.0), Some(70.0));

    // Undoing now must take back the *frame*, not a drag step — nothing about the
    // drag has been logged yet.
    engine.process(DocCommand::Undo);
    let undone = engine.render_to_image();
    assert!(
        red_dominant(outside(&undone)),
        "undo during a drag should remove the frame, so nothing was logged by it"
    );
    engine.process(DocCommand::Redo);

    // Releasing commits exactly one action, and drops the preview.
    engine.process(DocCommand::SetMatteRect(
        matte_id,
        Vec2::new(-35.0, -35.0),
        Vec2::new(35.0, 35.0),
    ));
    let committed = engine.render_to_image();
    assert!(
        images_match(&dragging, &committed, 2),
        "the committed rect should match what the drag previewed"
    );
    engine.process(DocCommand::Undo);
    let back = engine.render_to_image();
    assert!(
        images_match(&framed, &back, 2),
        "one undo should take back the whole drag"
    );
}

/// A canvas-color drag previews live but logs once (§15.5) — the
/// substrate's half of the bargain the test above makes for the frame, and here
/// beside it because both ride the same preview slot. A color picker reports a
/// value per pointer move, so without this a single drag would bury the history
/// under a hundred one-shade-apart edits.
#[test]
fn dragging_the_canvas_color_previews_without_logging() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let blank = engine.render_to_image();
    paint(&mut engine, RED, 30.0, WIDE_STROKE);
    let painted = engine.render_to_image();
    let substrate = engine.observe().substrate_color;

    // Three "pointer moves" of a drag towards a dark substrate.
    for v in [0.5f32, 0.3, 0.1] {
        engine.process(ViewCommand::PreviewSubstrateColor(Some(Srgb::new([
            v, v, v,
        ]))));
    }
    let dragging = engine.render_to_image();
    assert!(
        !images_match(&painted, &dragging, 4),
        "the preview should recolor the canvas on screen"
    );
    // `observe` reports the previewed color, so the panel's swatch agrees with the
    // canvas it controls instead of trailing a commit behind it.
    assert_eq!(engine.observe().substrate_color, Srgb::new([0.1, 0.1, 0.1]));

    // Undoing now must take back the *stroke*, not a drag step, and drop the
    // preview with it — landing on the untouched document, because nothing about
    // the drag has been logged.
    engine.process(DocCommand::Undo);
    assert!(
        images_match(&blank, &engine.render_to_image(), 2),
        "undo during a drag should take back the stroke and drop the preview"
    );
    engine.process(DocCommand::Redo);

    // Releasing commits exactly one action, and drops the preview.
    engine.process(DocCommand::SetSubstrateColor(Srgb::new([0.1, 0.1, 0.1])));
    let committed = engine.render_to_image();
    assert!(
        images_match(&dragging, &committed, 2),
        "the committed color should match what the drag previewed"
    );
    engine.process(DocCommand::Undo);
    assert_eq!(
        engine.observe().substrate_color,
        substrate,
        "one undo should take back the whole drag"
    );
    assert!(
        images_match(&painted, &engine.render_to_image(), 2),
        "and restore the image it started from"
    );
}

/// A frame-color pick previews live but logs once (§15.7) — the frame's own
/// color, next to the substrate's above, because the two are the same control
/// making the same bargain against different state — the same Oklab picker, even,
/// asked about a different flat expanse. It reports a color per pointer move, so
/// without this, choosing a mat board costs an undo step per shade the pointer
/// crossed on the way to it.
#[test]
fn picking_a_frame_color_previews_without_logging() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint(&mut engine, RED, 30.0, WIDE_STROKE);
    add_frame(&mut engine);
    let matte_id = engine.observe().layers.last().expect("matte").id;
    let black = engine.render_to_image();

    // Three "pointer moves" of a pick towards a pale mat board.
    for v in [0.3f32, 0.6, 0.9] {
        engine.process(ViewCommand::PreviewParcel(Some((
            matte_id,
            Parcel::Solid(Srgb::new([v, v, v])),
        ))));
    }
    let picking = engine.render_to_image();
    assert!(
        !is_dark(outside(&picking)),
        "the preview should recolor the frame on screen"
    );
    // `observe` reports the previewed color, so the swatch agrees with the canvas
    // it controls instead of trailing a commit behind it.
    let previewed = engine.observe().layers.last().and_then(|l| l.matte.clone());
    assert_eq!(
        previewed.map(|m| m.paint),
        Some(Parcel::Solid(Srgb::new([0.9, 0.9, 0.9])))
    );

    // A history step during the pick drops the preview and finds nothing of it
    // logged, so undo-then-redo lands back on the black frame.
    engine.process(DocCommand::Undo);
    engine.process(DocCommand::Redo);
    assert!(
        images_match(&black, &engine.render_to_image(), 2),
        "nothing about the pick should have been logged by it"
    );

    // Settling commits exactly one action, which supersedes the preview it matches.
    for v in [0.3f32, 0.6, 0.9] {
        engine.process(ViewCommand::PreviewParcel(Some((
            matte_id,
            Parcel::Solid(Srgb::new([v, v, v])),
        ))));
    }
    engine.process(DocCommand::SetMattePaint(
        matte_id,
        Parcel::Solid(Srgb::new([0.9, 0.9, 0.9])),
    ));
    assert!(
        images_match(&picking, &engine.render_to_image(), 2),
        "the committed color should match what the pick previewed"
    );
    engine.process(DocCommand::Undo);
    assert!(
        images_match(&black, &engine.render_to_image(), 2),
        "one undo should take back the whole pick"
    );
}

/// A pick that lands back on the color it opened on is not an edit — but it still
/// has to commit, because every release commits (`panels::color::end_pick`) and the
/// preview it left standing must be superseded by something. The refusal that
/// reconciles those two lives in the engine, which is what this pins (§15.7).
#[test]
fn a_frame_color_pick_that_changes_nothing_logs_nothing() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint(&mut engine, RED, 30.0, WIDE_STROKE);
    add_frame(&mut engine);
    let matte_id = engine.observe().layers.last().expect("matte").id;
    let black = engine.render_to_image();

    for v in [0.3f32, 0.6, 0.0] {
        engine.process(ViewCommand::PreviewParcel(Some((
            matte_id,
            Parcel::Solid(Srgb::new([v, v, v])),
        ))));
    }
    engine.process(DocCommand::SetMattePaint(
        matte_id,
        Parcel::Solid(Srgb::new(BLACK)),
    ));
    assert!(
        images_match(&black, &engine.render_to_image(), 2),
        "the settled pick should leave the document as it found it, preview and all"
    );

    // Nothing was logged, so one undo reaches past it to the frame itself.
    engine.process(DocCommand::Undo);
    assert!(
        red_dominant(outside(&engine.render_to_image())),
        "one undo should remove the frame, so the settled pick logged no step"
    );
}

/// A matte is an ordinary logged action, so undo steps through it.
#[test]
fn matte_undoes() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint(&mut engine, RED, 30.0, WIDE_STROKE);
    let before = engine.render_to_image();

    add_frame(&mut engine);
    assert!(is_dark(outside(&engine.render_to_image())));

    engine.process(DocCommand::Undo);
    let undone = engine.render_to_image();
    assert!(
        images_match(&before, &undone, 2),
        "undoing a matte should restore the un-framed image"
    );

    engine.process(DocCommand::Redo);
    assert!(is_dark(outside(&engine.render_to_image())));
}

// ---------------------------------------------------------------------------
// The whole-plane region and the gradient paint (§15.5, §22.4).
// ---------------------------------------------------------------------------

use stark_model::document::Parcel as MP;
use stark_model::document::{GradientAxis, GradientParcel};
use stark_model::{Gradient, GradientStop};

fn red_blue() -> Gradient {
    Gradient::new(vec![
        GradientStop {
            t: 0.0,
            color: Srgb::new([0.85, 0.1, 0.1]),
        },
        GradientStop {
            t: 1.0,
            color: Srgb::new([0.1, 0.2, 0.85]),
        },
    ])
    .expect("two stops")
}

/// A backing: an `Everything` matte born at the bottom of the stack covers the
/// whole view — and stays *under* the paint, which is the §15.5 claim that a
/// backing is an underpainting, not a wash over the picture.
#[test]
fn an_everything_matte_backs_the_whole_view_beneath_the_paint() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint(&mut engine, RED, 16.0, WIDE_STROKE);
    engine.process(DocCommand::AddMatte {
        carrier: None,
        at: Place::Bottom,
        region: MatteRegion::Everything,
        paint: Parcel::Solid(Srgb::new([0.1, 0.5, 0.15])),
    });
    let img = engine.render_to_image();
    // Green everywhere the paint is not…
    let corner = at(&img, 6, 6);
    assert!(
        corner[1] > corner[0] + 30 && corner[1] > corner[2] + 30,
        "the backing should cover bare canvas, got {corner:?}"
    );
    // …and the stroke still on top of it.
    assert!(
        red_dominant(outside(&img)),
        "a backing at the bottom must not cover the paint"
    );
}

/// A gradient matte grades across the canvas — the ramp is the §22.4 fill's,
/// read from the same axis at composite time, so the left of a left-to-right
/// ramp is its first stop and the right its last.
#[test]
fn a_gradient_matte_grades_across_the_canvas() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.process(DocCommand::AddMatte {
        carrier: None,
        at: Place::Bottom,
        region: MatteRegion::Everything,
        paint: MP::Gradient(GradientParcel {
            gradient: red_blue(),
            axis: GradientAxis::Linear {
                from: Vec2::new(-100.0, 0.0),
                to: Vec2::new(100.0, 0.0),
            },
        }),
    });
    let img = engine.render_to_image();
    let (w, h) = (img.width, img.height);
    let left = at(&img, w / 8, h / 2);
    let right = at(&img, w - w / 8, h / 2);
    assert!(
        left[0] > left[2] + 40,
        "the ramp starts on its red stop, got {left:?}"
    );
    assert!(
        right[2] > right[0] + 40,
        "and ends on its blue stop, got {right:?}"
    );
    // Constant on perpendiculars: same color up and down a vertical line.
    assert_eq!(
        at(&img, w / 4, h / 4),
        at(&img, w / 4, 3 * h / 4),
        "a linear ramp is a function of its axis alone"
    );
}

/// The gradient survives the log: undo restores the paint it replaced (the
/// patch restores the *value*, not a rect), and a reload replays it.
#[test]
fn a_gradient_matte_survives_undo_and_reload() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    add_frame(&mut engine);
    let matte_id = engine
        .observe()
        .layers
        .last()
        .expect("the frame is topmost")
        .id;
    let black = engine.render_to_image();
    engine.process(DocCommand::SetMattePaint(
        matte_id,
        MP::Gradient(GradientParcel {
            gradient: red_blue(),
            axis: GradientAxis::Radial {
                center: Vec2::ZERO,
                radius: 90.0,
            },
        }),
    ));
    let graded = engine.render_to_image();
    assert!(
        !images_match(&black, &graded, 4),
        "the gradient paint should change the frame"
    );

    engine.process(DocCommand::Undo);
    assert!(
        images_match(&black, &engine.render_to_image(), 1),
        "undo should restore the solid paint exactly"
    );
    engine.process(DocCommand::Redo);

    let bytes = engine.save_bytes().expect("save");
    engine.load_bytes(&bytes).expect("load");
    assert!(
        images_match(&graded, &engine.render_to_image(), 2),
        "a gradient matte did not replay identically from the log"
    );
}

/// An `Everything` matte defines no export frame: naming it exports the painted
/// bounds, the same answer as naming no frame at all (§15.6).
#[test]
fn an_everything_matte_defines_no_export_frame() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint(
        &mut engine,
        RED,
        16.0,
        &[Vec2::new(-30.0, 0.0), Vec2::new(30.0, 0.0)],
    );
    engine.process(DocCommand::AddMatte {
        carrier: None,
        at: Place::Bottom,
        region: MatteRegion::Everything,
        paint: Parcel::Solid(Srgb::new([0.9, 0.9, 0.85])),
    });
    let backing = engine
        .observe()
        .layers
        .first()
        .expect("the backing is at the bottom")
        .id;
    let with_backing = engine
        .export_plan(Some(backing), stark_engine::ExportScale::Factor(1.0))
        .expect("plan");
    let without = engine
        .export_plan(None, stark_engine::ExportScale::Factor(1.0))
        .expect("plan");
    assert_eq!(
        (with_backing.min, with_backing.max),
        (without.min, without.max),
        "a rect-less matte must fall back to the painted bounds"
    );
}
