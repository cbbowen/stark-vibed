//! Export (§15.6): rendering a frame to an image.
//!
//! The claim export rests on is that it is *the same path the screen takes* — every
//! visible layer through the media pass, just with the view centred on the frame at
//! `zoom = scale`. So these check the things that would quietly betray that: the
//! output size, that a frame contributes nothing to its own export, that scale is a
//! resolution and not a crop, and that chrome never reaches a file.

mod common;

use common::*;
use stark_engine::command::{DocCommand, GestureCommand, InputSample, PeerCommand, ViewCommand};
use stark_engine::document::{MattePaint, MatteRegion, Place, SelectionOp, Tool};
use stark_engine::geom::Vec2;
use stark_engine::path::DEFAULT_TOLERANCE;
use stark_engine::{Background, Engine, ExportScale, LayerId, Offscreen, Rendered, RgbaImage};

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
        carrier: None,
        at: Place::Top,
        region: FRAME,
        paint: MattePaint::Solid(BLACK),
    });
    engine.observe().layers.last().expect("matte").id
}

fn is_dark(c: [u8; 4]) -> bool {
    c[0] < 60 && c[1] < 60 && c[2] < 60
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
            .export(
                &mut Offscreen::default(),
                Some(frame),
                ExportScale::Factor(1.0),
                Background::Substrate,
                Rendered::Live,
            )
            .expect("export"),
    )
    .expect("the readback completes");
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
            .export(
                &mut Offscreen::default(),
                Some(frame),
                ExportScale::Factor(1.0),
                Background::Substrate,
                Rendered::Live,
            )
            .expect("1x"),
    )
    .expect("the readback completes");
    let two = pollster::block_on(
        engine
            .export(
                &mut Offscreen::default(),
                Some(frame),
                ExportScale::Factor(2.0),
                Background::Substrate,
                Rendered::Live,
            )
            .expect("2x"),
    )
    .expect("the readback completes");
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
            .export(
                &mut Offscreen::default(),
                Some(frame),
                ExportScale::Width(240),
                Background::Substrate,
                Rendered::Live,
            )
            .expect("by width"),
    )
    .expect("the readback completes");
    assert_eq!((by_width.width, by_width.height), (240, 160));
}

/// A transparent export is a cut-out: bare canvas is genuinely absent, painted
/// texels are not. The substrate composite is skipped rather than the alpha being
/// tweaked afterwards, so bare canvas must not carry substrate color either
/// (§15.6).
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
                &mut Offscreen::default(),
                Some(frame),
                ExportScale::Factor(1.0),
                Background::Transparent,
                Rendered::Live,
            )
            .expect("transparent"),
    )
    .expect("the readback completes");
    let opaque = pollster::block_on(
        engine
            .export(
                &mut Offscreen::default(),
                Some(frame),
                ExportScale::Factor(1.0),
                Background::Substrate,
                Rendered::Live,
            )
            .expect("substrate"),
    )
    .expect("the readback completes");

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
    assert!(red_dominant(mid), "and keep their own color, got {mid:?}");

    // The substrate export is opaque everywhere — the two really are different
    // renders, not the same one relabelled.
    assert!(corners(&opaque).iter().all(|c| c[3] == 255));
}

/// The substrate is document state now (§15.5), so it travels with
/// the export instead of depending on whichever frontend asked for it.
#[test]
fn export_uses_the_documents_ground() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let frame = add_frame(&mut engine);
    let paper = pollster::block_on(
        engine
            .export(
                &mut Offscreen::default(),
                Some(frame),
                ExportScale::Factor(1.0),
                Background::Substrate,
                Rendered::Live,
            )
            .expect("paper"),
    )
    .expect("the readback completes");
    assert!(!is_dark(paper.pixel(4, 4)), "default ground is near-white");

    engine.process(DocCommand::SetBackground([0.02, 0.02, 0.03]));
    let ink = pollster::block_on(
        engine
            .export(
                &mut Offscreen::default(),
                Some(frame),
                ExportScale::Factor(1.0),
                Background::Substrate,
                Rendered::Live,
            )
            .expect("ink"),
    )
    .expect("the readback completes");
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
                .export(
                    &mut Offscreen::default(),
                    Some(frame),
                    ExportScale::Factor(1.0),
                    Background::Substrate,
                    Rendered::Live
                )
                .expect("undone")
        )
        .expect("the readback completes")
        .pixel(4, 4)
    ));
}

/// Chrome never reaches a file: an active selection outlines on screen but must
/// not appear in an export (§15.6).
#[test]
fn export_omits_the_selection_outline() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let frame = add_frame(&mut engine);
    let clean = pollster::block_on(
        engine
            .export(
                &mut Offscreen::default(),
                Some(frame),
                ExportScale::Factor(1.0),
                Background::Substrate,
                Rendered::Live,
            )
            .expect("clean"),
    )
    .expect("the readback completes");

    // A marquee well inside the frame, so its outline would land in the export.
    engine.process(GestureCommand::Start {
        tool: Tool::SelectRect,
        sample: InputSample::at(Vec2::new(-30.0, -20.0)),
        tolerance: DEFAULT_TOLERANCE,
        rope: 0.0,
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
            .export(
                &mut Offscreen::default(),
                Some(frame),
                ExportScale::Factor(1.0),
                Background::Substrate,
                Rendered::Live,
            )
            .expect("with selection"),
    )
    .expect("the readback completes");
    assert!(
        images_match(&clean, &selected, 2),
        "the selection outline leaked into the export"
    );

    // It is still on screen, though — the outline is chrome, not a deletion.
    engine.process(DocCommand::Select(SelectionOp::select_all()));
    let _ = engine.render_to_image();
}

/// With no frame, export falls back to the painted bounds, then to the viewport —
/// so it always means *something* on an unbounded canvas (§15.6).
#[test]
fn export_without_a_frame_falls_back() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // Nothing painted: the viewport is the only thing left to mean.
    let empty = pollster::block_on(
        engine
            .export(
                &mut Offscreen::default(),
                None,
                ExportScale::Factor(1.0),
                Background::Substrate,
                Rendered::Live,
            )
            .expect("empty"),
    )
    .expect("the readback completes");
    assert_eq!((empty.width, empty.height), (SIZE.width, SIZE.height));

    // Painted: the populated tiles' bounds, which are tile-aligned and so at least
    // one tile across.
    paint(&mut engine, RED, 20.0, &[Vec2::ZERO, Vec2::new(10.0, 0.0)]);
    let plan = engine
        .export_plan(None, ExportScale::Factor(1.0))
        .expect("plan");
    assert!(plan.size.width >= stark_engine::TILE_SIZE);
    assert!(plan.size.height >= stark_engine::TILE_SIZE);
}

/// **Showing the piece frames what an export would write** (§15.6): the same rect,
/// asked of the same rule, so the view and a file cannot come to disagree about where
/// the piece ends. This is the framing a document load does — a view is per-client
/// session state and so is not in the file, and without it a painting opens at
/// whatever pan and zoom the last one was left at.
///
/// Stated against `export_plan` rather than against numbers of its own: a test that
/// wrote the expected rect out by hand would be a second copy of the rule, and would
/// go on passing if the two implementations drifted apart, which is the only failure
/// worth catching here.
#[test]
fn showing_the_piece_frames_what_an_export_would_write() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    /// Turned, mirrored, zoomed in and looking somewhere else entirely — a session
    /// left where the *previous* document had it.
    fn look_elsewhere(engine: &mut Engine) {
        engine.process(ViewCommand::CenterOn(Vec2::splat(9000.0)));
        engine.process(ViewCommand::SetRotation(0.9));
        engine.process(ViewCommand::MirrorH);
        engine.process(ViewCommand::Zoom {
            anchor: Vec2::ZERO,
            factor: 8.0,
        });
    }

    /// Frame the piece from over there, and check it against the export's own plan.
    fn frames_like_an_export(engine: &mut Engine, frame: Option<LayerId>) {
        look_elsewhere(engine);
        let plan = engine
            .export_plan(frame, ExportScale::Factor(1.0))
            .expect("plan");
        engine.process(ViewCommand::ShowPiece(frame));
        let view = engine.observe().view;

        assert_eq!(
            view.center,
            (plan.min + plan.max) * 0.5,
            "{frame:?}: centred on a different rect than the export's"
        );
        assert_eq!(
            (view.rotation, view.flip_h),
            (0.0, false),
            "{frame:?}: the easel should be straightened, as a file is written upright"
        );
        // Every corner of what a file would hold is on screen, with room to spare —
        // the margin, which is the whole difference between framing a piece and
        // cropping to it.
        let (lo, hi) = view.visible_bounds();
        assert!(
            lo.x < plan.min.x && lo.y < plan.min.y && hi.x > plan.max.x && hi.y > plan.max.y,
            "{frame:?}: {lo:?}..{hi:?} does not show {:?}..{:?} whole",
            plan.min,
            plan.max
        );
        // And snug: the piece fills the window but for that margin, rather than
        // sitting as a speck in the middle of it.
        let (shown, piece) = (hi - lo, plan.max - plan.min);
        assert!(
            (shown.x / piece.x).min(shown.y / piece.y) < 1.2,
            "{frame:?}: zoomed out to {shown:?} for a {piece:?} piece"
        );
    }

    // Nothing painted and no frame: there is no piece to show, so the view holds
    // still rather than framing the window onto itself — which is where an *export*
    // falls back to, and the one place the two part company.
    look_elsewhere(&mut engine);
    let held = engine.observe().view;
    engine.process(ViewCommand::ShowPiece(None));
    assert_eq!(
        engine.observe().view,
        held,
        "an empty document has nothing to frame"
    );

    // Painted and unframed, the piece is the painted bounds...
    paint(&mut engine, RED, 30.0, WIDE);
    frames_like_an_export(&mut engine, None);
    // ...and once there is a frame, it is the frame — a stroke running well past it
    // on both sides, so a fit that took the paint instead would come out wider.
    let frame = add_frame(&mut engine);
    frames_like_an_export(&mut engine, Some(frame));
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

/// The export limit is **the device's own**, and everything inside it really renders.
///
/// Not the literal 8192, which is only where `wgpu::Limits::default()` happens to cap
/// 2D textures — the frontend requests those, so on the app's device a hardcoded
/// number would agree by coincidence. The headless device asks for
/// `downlevel_defaults` (2048, the web/WebGL2 floor), and there every size from 2049
/// to 8192 would pass a check written against a limit the device does not have, then
/// ask wgpu for a texture it was never granted: the permissive direction, from the one
/// guard whose whole job is to report that in words instead.
///
/// Both halves matter. Refusing one past the limit is the easy one; *rendering* the
/// one exactly at it is what says the number is real and not merely arithmetic
/// agreeing with itself.
#[test]
fn the_export_limit_is_the_devices_own() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let limit = engine.gpu().device.limits().max_texture_dimension_2d;
    let frame = add_frame(&mut engine);

    // A square frame `limit` px on a side: at 1:1 that is the largest export this
    // device can make, whatever this device happens to be.
    engine.process(DocCommand::SetMatteRect(
        frame,
        Vec2::ZERO,
        Vec2::splat(limit as f32),
    ));
    let plan = engine
        .export_plan(Some(frame), ExportScale::Factor(1.0))
        .expect("an export exactly at the device limit is allowed");
    assert_eq!((plan.size.width, plan.size.height), (limit, limit));
    let img = pollster::block_on(
        engine
            .export(
                &mut Offscreen::default(),
                Some(frame),
                ExportScale::Factor(1.0),
                Background::Substrate,
                Rendered::Committed,
            )
            .expect("export at the limit"),
    )
    .expect("the readback completes");
    assert_eq!((img.width, img.height), (limit, limit));

    // One pixel more is refused in the engine's words, not wgpu's.
    engine.process(DocCommand::SetMatteRect(
        frame,
        Vec2::ZERO,
        Vec2::splat(limit as f32 + 1.0),
    ));
    let err = engine
        .export_plan(Some(frame), ExportScale::Factor(1.0))
        .expect_err("one px past the device limit must be refused");
    assert!(
        format!("{err}").contains(&limit.to_string()),
        "the refusal should name the limit it is against, got: {err}"
    );
}

/// A piece too large to export at 1:1 can still be *previewed* (§15.6).
///
/// The navigator asks for the largest plan that fits its panel, and that question has
/// to be answerable for any piece at all — the miniature matters most on the ones too
/// big to see. Asked as a 1× plan (to learn the rect) followed by a scaled one, a
/// piece past the device's texture limit fails the query for a render that was never
/// going to happen: `draw_overview` returns `None` and the panel goes on quietly
/// showing a stale picture.
#[test]
fn a_piece_past_the_export_limit_still_has_an_overview() {
    use stark_engine::geom::Extent2;
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let frame = add_frame(&mut engine);
    // Past any device's texture limit, and not square, so the fit has to respect the
    // binding axis rather than whichever it looked at first.
    engine.process(DocCommand::SetMatteRect(
        frame,
        Vec2::ZERO,
        Vec2::new(20000.0, 12000.0),
    ));
    assert!(
        engine
            .export_plan(Some(frame), ExportScale::Factor(1.0))
            .is_err(),
        "20000 × 12000 px should still be refused as an export"
    );

    let panel = Extent2::new(252, 176);
    let plan = engine
        .export_plan(Some(frame), ExportScale::Fit(panel))
        .expect("a piece of any size has an overview");
    assert!(
        plan.size.width <= panel.width && plan.size.height <= panel.height,
        "the overview is {} × {}, which does not fit {} × {}",
        plan.size.width,
        plan.size.height,
        panel.width,
        panel.height
    );
    // The piece is 5:3 against a panel nearer 1.4:1, so it is width that binds — and
    // the fit is *tight* on that axis. A preview that fitted by coming in under the
    // box on both axes would be quietly wasting the panel it was given.
    assert_eq!(plan.size.width, panel.width);
    assert!(plan.size.height < panel.height);

    // The overview really is renderable, which is the thing the panel gave up on.
    let image = pollster::block_on(
        engine
            .export(
                &mut Offscreen::default(),
                Some(frame),
                ExportScale::Fit(panel),
                Background::Substrate,
                Rendered::Committed,
            )
            .expect("export"),
    )
    .expect("the readback completes");
    assert_eq!(
        (image.width, image.height),
        (plan.size.width, plan.size.height)
    );
}

/// Fitting scales a small piece *up*: the overview shows the whole of it at a glance,
/// and a 60-px sketch shown at 60 px says less than the empty panel around it.
#[test]
fn fitting_a_small_piece_fills_the_box() {
    use stark_engine::geom::Extent2;
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let frame = add_frame(&mut engine); // 120 × 80
    let plan = engine
        .export_plan(Some(frame), ExportScale::Fit(Extent2::new(252, 176)))
        .expect("plan");
    assert!(
        plan.zoom > 1.0,
        "a 120 × 80 piece should fit at {}×",
        plan.zoom
    );
    // 252/120 = 2.1 binds before 176/80 = 2.2.
    assert_eq!((plan.size.width, plan.size.height), (252, 168));
}

/// An export is RGBA whatever the render target's own channel order is.
///
/// This is the bug that shipped: every test renders to `Rgba8Unorm`, but a browser
/// surface is typically `Bgra8Unorm`, and the readback handed those bytes straight
/// to an RGBA image — so the exported PNG came out with red and blue swapped.
/// Green, black and white are all fixed points of a R↔B swap, which is exactly why
/// it read as a color-space problem rather than a byte-order one.
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
        // A color with three distinct channels, so a swap cannot hide in it.
        paint(engine, [0.9, 0.35, 0.1, 1.0], 30.0, WIDE);
        add_frame(engine);
    }
    let a = pollster::block_on(
        rgba.export(
            &mut Offscreen::default(),
            Some(LayerId(1)),
            ExportScale::Factor(1.0),
            Background::Substrate,
            Rendered::Live,
        )
        .expect("rgba export"),
    )
    .expect("the readback completes");
    let b = pollster::block_on(
        bgra.export(
            &mut Offscreen::default(),
            Some(LayerId(1)),
            ExportScale::Factor(1.0),
            Background::Substrate,
            Rendered::Live,
        )
        .expect("bgra export"),
    )
    .expect("the readback completes");

    assert_eq!((a.width, a.height), (b.width, b.height));
    assert!(
        images_match(&a, &b, 2),
        "a BGRA target exported different colors than an RGBA one — the readback \
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
/// dialog reports a size while the user is still dragging (§15.7).
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
            .export(
                &mut Offscreen::default(),
                Some(frame),
                ExportScale::Factor(2.0),
                Background::Substrate,
                Rendered::Live,
            )
            .expect("export"),
    )
    .expect("the readback completes");
    assert_eq!(
        (img.width, img.height),
        (plan.size.width, plan.size.height),
        "the plan must describe what export actually produces"
    );
    let _ = PeerCommand::SetActiveLayer(frame);
}

/// [`Rendered::Committed`] leaves the in-flight stroke out, and
/// [`Rendered::Live`] keeps it in.
///
/// This is what lets a frontend render a *stand-in* for the document — the
/// navigator's miniature — on the cadence of the undo history rather than of the
/// pointer: it can only refresh per committed change if what it renders is
/// committed-only, or it would show whichever gesture happened to be in flight when
/// its timer fired and never correct itself afterwards.
#[test]
fn committed_renders_omit_the_in_flight_stroke() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let frame = add_frame(&mut engine);
    let bare = pollster::block_on(
        engine
            .export(
                &mut Offscreen::default(),
                Some(frame),
                ExportScale::Factor(1.0),
                Background::Substrate,
                Rendered::Committed,
            )
            .expect("bare"),
    )
    .expect("the readback completes");

    // A stroke *held down* across the middle of the frame: nothing is committed yet.
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(WIDE[0]),
        tolerance: DEFAULT_TOLERANCE,
        rope: 0.0,
    });
    engine.process(GestureCommand::To {
        sample: InputSample::at(WIDE[1]),
    });
    assert!(engine.observe().is_stroking, "the stroke should be in hand");

    let live = pollster::block_on(
        engine
            .export(
                &mut Offscreen::default(),
                Some(frame),
                ExportScale::Factor(1.0),
                Background::Substrate,
                Rendered::Live,
            )
            .expect("live"),
    )
    .expect("the readback completes");
    let committed = pollster::block_on(
        engine
            .export(
                &mut Offscreen::default(),
                Some(frame),
                ExportScale::Factor(1.0),
                Background::Substrate,
                Rendered::Committed,
            )
            .expect("committed"),
    )
    .expect("the readback completes");
    assert!(
        !images_match(&bare, &live, 2),
        "the stroke in hand must show in a live render"
    );
    assert!(
        images_match(&bare, &committed, 2),
        "a committed render must not see the stroke in hand"
    );

    // And on release the two agree again: the difference is only ever "in flight",
    // never "invisible to one of them".
    engine.process(GestureCommand::End);
    let after = pollster::block_on(
        engine
            .export(
                &mut Offscreen::default(),
                Some(frame),
                ExportScale::Factor(1.0),
                Background::Substrate,
                Rendered::Committed,
            )
            .expect("after"),
    )
    .expect("the readback completes");
    assert!(
        images_match(&live, &after, 6),
        "the committed stroke should look like the one that was previewed"
    );
}

/// The revision a frontend keys a rendered stand-in on: it moves when the committed
/// document does, and — the half that decides whether a navigator is affordable —
/// stays put through the whole of a gesture, however many samples it takes.
#[test]
fn doc_revision_tracks_commits_and_not_gestures() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let at_rest = engine.observe().doc_revision;

    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(WIDE[0]),
        tolerance: DEFAULT_TOLERANCE,
        rope: 0.0,
    });
    for i in 1..8 {
        engine.process(GestureCommand::To {
            sample: InputSample::at(Vec2::new(-110.0 + 30.0 * i as f32, 0.0)),
        });
        assert_eq!(
            engine.observe().doc_revision,
            at_rest,
            "a stroke in flight is not a change to the document"
        );
    }
    engine.process(GestureCommand::End);
    let committed = engine.observe().doc_revision;
    assert_ne!(committed, at_rest, "the commit is a change");

    // Panning is not, and neither is an unlogged preview drag — but an undo is.
    engine.process(ViewCommand::CenterOn(Vec2::new(500.0, -250.0)));
    engine.process(ViewCommand::PreviewBackground(Some([0.1, 0.1, 0.1])));
    assert_eq!(engine.observe().doc_revision, committed);
    engine.process(DocCommand::Undo);
    assert_ne!(engine.observe().doc_revision, committed);
}

/// `CenterOn` puts a canvas point at the middle of the viewport and leaves the zoom
/// where it was — what a navigator click means.
#[test]
fn center_on_moves_the_view_without_zooming() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.process(ViewCommand::Zoom {
        anchor: Vec2::new(10.0, 10.0),
        factor: 2.0,
    });
    let zoom = engine.view().zoom;
    let target = Vec2::new(-321.0, 654.0);
    engine.process(ViewCommand::CenterOn(target));
    let view = engine.view();
    assert_eq!(view.center, target);
    assert_eq!(view.zoom, zoom);
    // Which is to say: the point is under the middle of the window.
    let middle = Vec2::new(
        view.viewport.width as f32 * 0.5,
        view.viewport.height as f32 * 0.5,
    );
    assert!(view.screen_to_canvas(middle).distance(target) < 1e-3);
}
