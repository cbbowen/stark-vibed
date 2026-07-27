//! Export (FRAME_DESIGN.md §6): rendering a frame to an image.
//!
//! The claim export rests on is that it is *the same path the screen takes* — every
//! visible layer through the media pass, just with the view centred on the frame at
//! `zoom = scale`. So these check the things that would quietly betray that: the
//! output size, that a frame contributes nothing to its own export, that scale is a
//! resolution and not a crop, and that chrome never reaches a file.

mod common;

use common::*;
use stark_core::command::{DocCommand, GestureCommand, InputSample, PeerCommand};
use stark_core::document::{MatteRegion, SelectionOp, Tool};
use stark_core::geom::Vec2;
use stark_core::path::DEFAULT_TOLERANCE;
use stark_core::{Background, Engine, ExportScale, LayerId, RgbaImage};

const RED: [f32; 4] = [0.85, 0.1, 0.1, 1.0];
const BLACK: [f32; 3] = [0.0, 0.0, 0.0];

/// A 120×80 frame centred on the canvas origin.
const FRAME: MatteRegion = MatteRegion::OutsideRect {
    min: Vec2::new(-60.0, -40.0),
    max: Vec2::new(60.0, 40.0),
};

/// A stroke through the frame that runs well past it on both sides, so anything
/// that failed to crop would show the overshoot.
const WIDE: &[Vec2] = &[Vec2::new(-110.0, 0.0), Vec2::new(110.0, 0.0)];

fn add_frame(engine: &mut Engine) -> LayerId {
    engine.process(DocCommand::AddMatte {
        above: None,
        region: FRAME,
        color: BLACK,
    });
    engine.observe().layers.last().expect("matte").id
}

fn is_dark(c: [u8; 4]) -> bool {
    c[0] < 60 && c[1] < 60 && c[2] < 60
}
fn red_dominant(c: [u8; 4]) -> bool {
    c[0] as i32 > c[1] as i32 + 30 && c[0] as i32 > c[2] as i32 + 30
}
fn corners(img: &RgbaImage) -> [[u8; 4]; 4] {
    let (w, h) = (img.width - 1, img.height - 1);
    [
        img.pixel(0, 0),
        img.pixel(w, 0),
        img.pixel(0, h),
        img.pixel(w, h),
    ]
}

/// The frame's canvas rect becomes the image's pixel size, and — the part that is
/// easy to get wrong — the frame matte contributes *nothing to its own export*.
/// It covers only outside its rect, which is exactly what got cropped away, so
/// this falls out with no special case. If export were framing the wrong rect (or
/// off by the matte's own coverage) the corners would come back mat-board black.
#[test]
fn exports_the_frame_rect_without_its_own_matte() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint(&mut engine, RED, 30.0, WIDE);
    let frame = add_frame(&mut engine);

    let img = pollster::block_on(
        engine
            .export(Some(frame), ExportScale::Factor(1.0), Background::Substrate)
            .expect("export"),
    );
    assert_eq!(
        (img.width, img.height),
        (120, 80),
        "1x = one px per canvas px"
    );
    for (i, c) in corners(&img).iter().enumerate() {
        assert!(
            !is_dark(*c),
            "corner {i} is mat-board black: the frame is covering its own export ({c:?})"
        );
    }
    // The stroke crosses the middle, so the piece is actually in there.
    assert!(red_dominant(img.pixel(img.width / 2, img.height / 2)));
}

/// Scale is a resolution, not a crop: 2× is the same picture at twice the size.
#[test]
fn scale_changes_resolution_not_framing() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint(&mut engine, RED, 30.0, WIDE);
    let frame = add_frame(&mut engine);

    let one = pollster::block_on(
        engine
            .export(Some(frame), ExportScale::Factor(1.0), Background::Substrate)
            .expect("1x"),
    );
    let two = pollster::block_on(
        engine
            .export(Some(frame), ExportScale::Factor(2.0), Background::Substrate)
            .expect("2x"),
    );
    assert_eq!((two.width, two.height), (one.width * 2, one.height * 2));

    // Same framing: the four corners and the centre agree between the two, which
    // they could not if 2× had zoomed in on a sub-rect instead of resolving finer.
    let near = |a: [u8; 4], b: [u8; 4]| (0..3).all(|c| (a[c] as i32 - b[c] as i32).abs() <= 12);
    for (a, b) in corners(&one).iter().zip(corners(&two).iter()) {
        assert!(
            near(*a, *b),
            "corner differs between 1x and 2x: {a:?} vs {b:?}"
        );
    }
    assert!(near(
        one.pixel(one.width / 2, one.height / 2),
        two.pixel(two.width / 2, two.height / 2)
    ));

    // An explicit width is the same thing said the other way round.
    let by_width = pollster::block_on(
        engine
            .export(Some(frame), ExportScale::Width(240), Background::Substrate)
            .expect("by width"),
    );
    assert_eq!((by_width.width, by_width.height), (240, 160));
}

/// A transparent export is a cut-out: bare canvas is genuinely absent, painted
/// texels are not. The substrate composite is skipped rather than the alpha being
/// tweaked afterwards, so bare canvas must not carry substrate colour either
/// (FRAME_DESIGN.md §6).
#[test]
fn transparent_export_cuts_out_the_paint() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // A short stroke through the middle, leaving the frame's edges bare.
    paint(
        &mut engine,
        RED,
        24.0,
        &[Vec2::new(-20.0, 0.0), Vec2::new(20.0, 0.0)],
    );
    let frame = add_frame(&mut engine);

    let cut = pollster::block_on(
        engine
            .export(
                Some(frame),
                ExportScale::Factor(1.0),
                Background::Transparent,
            )
            .expect("transparent"),
    );
    let opaque = pollster::block_on(
        engine
            .export(Some(frame), ExportScale::Factor(1.0), Background::Substrate)
            .expect("substrate"),
    );

    for (i, c) in corners(&cut).iter().enumerate() {
        assert_eq!(
            c[3], 0,
            "corner {i} of a cut-out should be fully transparent"
        );
    }
    let mid = cut.pixel(cut.width / 2, cut.height / 2);
    assert!(
        mid[3] > 200,
        "painted texels should stay opaque, got {mid:?}"
    );
    assert!(red_dominant(mid), "and keep their own colour, got {mid:?}");

    // The substrate export is opaque everywhere — the two really are different
    // renders, not the same one relabelled.
    assert!(corners(&opaque).iter().all(|c| c[3] == 255));
}

/// The substrate is document state now (FRAME_DESIGN.md §5), so it travels with
/// the export instead of depending on whichever frontend asked for it.
#[test]
fn export_uses_the_documents_ground() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let frame = add_frame(&mut engine);
    let paper = pollster::block_on(
        engine
            .export(Some(frame), ExportScale::Factor(1.0), Background::Substrate)
            .expect("paper"),
    );
    assert!(!is_dark(paper.pixel(4, 4)), "default ground is near-white");

    engine.process(DocCommand::SetBackground([0.02, 0.02, 0.03]));
    let ink = pollster::block_on(
        engine
            .export(Some(frame), ExportScale::Factor(1.0), Background::Substrate)
            .expect("ink"),
    );
    assert!(
        is_dark(ink.pixel(4, 4)),
        "a dark ground should reach the export, got {:?}",
        ink.pixel(4, 4)
    );

    // And it undoes like any other edit.
    engine.process(DocCommand::Undo);
    assert!(!is_dark(
        pollster::block_on(
            engine
                .export(Some(frame), ExportScale::Factor(1.0), Background::Substrate)
                .expect("undone")
        )
        .pixel(4, 4)
    ));
}

/// Chrome never reaches a file: an active selection outlines on screen but must
/// not appear in an export (FRAME_DESIGN.md §6).
#[test]
fn export_omits_the_selection_outline() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let frame = add_frame(&mut engine);
    let clean = pollster::block_on(
        engine
            .export(Some(frame), ExportScale::Factor(1.0), Background::Substrate)
            .expect("clean"),
    );

    // A marquee well inside the frame, so its outline would land in the export.
    engine.process(GestureCommand::Start {
        tool: Tool::SelectRect,
        sample: InputSample::at(Vec2::new(-30.0, -20.0)),
        tolerance: DEFAULT_TOLERANCE,
    });
    engine.process(GestureCommand::To {
        sample: InputSample::at(Vec2::new(30.0, 20.0)),
    });
    engine.process(GestureCommand::End);
    assert!(
        engine.observe().has_selection,
        "the marquee should have taken"
    );

    let selected = pollster::block_on(
        engine
            .export(Some(frame), ExportScale::Factor(1.0), Background::Substrate)
            .expect("with selection"),
    );
    assert!(
        images_match(&clean, &selected, 2),
        "the selection outline leaked into the export"
    );

    // It is still on screen, though — the outline is chrome, not a deletion.
    engine.process(DocCommand::Select(SelectionOp::select_all()));
    let _ = engine.render_to_image();
}

/// With no frame, export falls back to the painted bounds, then to the viewport —
/// so it always means *something* on an unbounded canvas (FRAME_DESIGN.md §6).
#[test]
fn export_without_a_frame_falls_back() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // Nothing painted: the viewport is the only thing left to mean.
    let empty = pollster::block_on(
        engine
            .export(None, ExportScale::Factor(1.0), Background::Substrate)
            .expect("empty"),
    );
    assert_eq!((empty.width, empty.height), (SIZE.width, SIZE.height));

    // Painted: the populated tiles' bounds, which are tile-aligned and so at least
    // one tile across.
    paint(&mut engine, RED, 20.0, &[Vec2::ZERO, Vec2::new(10.0, 0.0)]);
    let plan = engine
        .export_plan(None, ExportScale::Factor(1.0))
        .expect("plan");
    assert!(plan.size.width >= stark_core::TILE_SIZE);
    assert!(plan.size.height >= stark_core::TILE_SIZE);
}

/// A frame too small to have any pixels, or a scale that would ask the device for
/// an absurd texture, is an error rather than a wgpu validation panic.
#[test]
fn impossible_exports_are_errors() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let frame = add_frame(&mut engine);

    engine.process(DocCommand::SetMatteRect(
        frame,
        Vec2::ZERO,
        Vec2::new(0.2, 0.2),
    ));
    assert!(
        engine
            .export_plan(Some(frame), ExportScale::Factor(1.0))
            .is_err(),
        "a sub-pixel frame cannot be exported"
    );

    engine.process(DocCommand::SetMatteRect(
        frame,
        Vec2::new(-60.0, -40.0),
        Vec2::new(60.0, 40.0),
    ));
    assert!(
        engine
            .export_plan(Some(frame), ExportScale::Factor(1000.0))
            .is_err(),
        "an absurd scale should be refused, not handed to the device"
    );
    assert!(
        engine
            .export_plan(Some(frame), ExportScale::Factor(0.0))
            .is_err(),
        "a zero scale should be refused"
    );
}

/// An export is RGBA whatever the render target's own channel order is.
///
/// This is the bug that shipped: every test renders to `Rgba8Unorm`, but a browser
/// surface is typically `Bgra8Unorm`, and the readback handed those bytes straight
/// to an RGBA image — so the exported PNG came out with red and blue swapped.
/// Green, black and white are all fixed points of a R↔B swap, which is exactly why
/// it read as a colour-space problem rather than a byte-order one.
///
/// Painting the same thing on the two formats and demanding identical *bytes* is
/// the check no single-format test could make.
#[test]
fn export_is_rgba_whatever_the_target_format_is() {
    let Some(mut rgba) = engine_or_skip() else {
        return;
    };
    let Some(mut bgra) = engine_or_skip_in_format(wgpu::TextureFormat::Bgra8Unorm) else {
        return;
    };

    for engine in [&mut rgba, &mut bgra] {
        // A colour with three distinct channels, so a swap cannot hide in it.
        paint(engine, [0.9, 0.35, 0.1, 1.0], 30.0, WIDE);
        add_frame(engine);
    }
    let a = pollster::block_on(
        rgba.export(
            Some(LayerId(1)),
            ExportScale::Factor(1.0),
            Background::Substrate,
        )
        .expect("rgba export"),
    );
    let b = pollster::block_on(
        bgra.export(
            Some(LayerId(1)),
            ExportScale::Factor(1.0),
            Background::Substrate,
        )
        .expect("bgra export"),
    );

    assert_eq!((a.width, a.height), (b.width, b.height));
    assert!(
        images_match(&a, &b, 2),
        "a BGRA target exported different colours than an RGBA one — the readback \
         is not normalizing channel order. Centre texel: {:?} vs {:?}",
        a.pixel(a.width / 2, a.height / 2),
        b.pixel(b.width / 2, b.height / 2),
    );
    // And the paint really is warm in both, so this is not two identically-wrong
    // images agreeing with each other.
    for img in [&a, &b] {
        let c = img.pixel(img.width / 2, img.height / 2);
        assert!(
            red_dominant(c),
            "warm paint should stay warm after export, got {c:?}"
        );
    }
}

/// The frame's *live* rect is what exports — a preview drag included, since the
/// dialog reports a size while the user is still dragging (FRAME_DESIGN.md §7).
#[test]
fn export_plan_reports_the_size_it_will_produce() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let frame = add_frame(&mut engine);
    let plan = engine
        .export_plan(Some(frame), ExportScale::Factor(2.0))
        .expect("plan");
    assert_eq!((plan.size.width, plan.size.height), (240, 160));
    assert_eq!(plan.zoom, 2.0);

    let img = pollster::block_on(
        engine
            .export(Some(frame), ExportScale::Factor(2.0), Background::Substrate)
            .expect("export"),
    );
    assert_eq!(
        (img.width, img.height),
        (plan.size.width, plan.size.height),
        "the plan must describe what export actually produces"
    );
    let _ = PeerCommand::SetActiveLayer(frame);
}
