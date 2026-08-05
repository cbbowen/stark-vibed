//! Turning and mirroring the view (§18.1.2).
//!
//! **The claim.** How you are looking at the painting is not part of the painting.
//! Rotation and the mirror are view state: they change every pixel the *screen*
//! shows, and nothing at all about the document, what a file gets, or where a stroke
//! lands relative to the pen.
//!
//! So these check the two halves of that separately — that the screen really does
//! turn, and that nothing else moved with it.

mod common;

use common::*;
use stark_core::command::{GestureCommand, InputSample, ViewCommand};
use stark_core::document::{MatteRegion, Tool};
use stark_core::geom::Vec2;
use stark_core::path::DEFAULT_TOLERANCE;
use stark_core::{Background, Engine, ExportScale, Offscreen, Rendered, RgbaImage};

const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];

/// Channel dominance, with the margin `stroke.rs` justifies: over the blue ground,
/// this cleanly separates lit paint from lit substrate.
fn is_red(c: [u8; 4]) -> bool {
    c[0] as i32 > c[1] as i32 + 60 && c[0] as i32 > c[2] as i32 + 60
}

/// The four probes around the centre of a `SIZE`-square render, at `d` px out.
fn probes(img: &RgbaImage, d: u32) -> (bool, bool, bool, bool) {
    let (cx, cy) = (img.width / 2, img.height / 2);
    (
        is_red(img.pixel(cx - d, cy)),
        is_red(img.pixel(cx + d, cy)),
        is_red(img.pixel(cx, cy - d)),
        is_red(img.pixel(cx, cy + d)),
    )
}

/// A horizontal bar of paint through the canvas origin.
fn bar(engine: &mut Engine, from: f32, to: f32) {
    paint(
        engine,
        RED,
        14.0,
        &[Vec2::new(from, 0.0), Vec2::new(to, 0.0)],
    );
}

/// Turning the canvas turns what the screen shows. A bar painted across the canvas
/// reads across the screen upright, and up the screen after a quarter turn — which is
/// the whole feature, and the thing every uniform and shader change has to preserve.
#[test]
fn a_quarter_turn_stands_the_painting_on_end() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    bar(&mut engine, -90.0, 90.0);

    let (left, right, above, below) = probes(&engine.render_to_image(), 60);
    assert!(
        left && right && !above && !below,
        "upright, the bar should run across the screen"
    );

    // Canvas −x to the top of the screen: a quarter turn clockwise. Asked for the
    // way the navigator's drag asks — as a direction — since that is the path with
    // something to get wrong.
    let up = engine.view().rotation_for_up(Vec2::new(-1.0, 0.0)).unwrap();
    engine.process(ViewCommand::SetRotation(up));
    assert!((engine.view().rotation - std::f32::consts::FRAC_PI_2).abs() < 1e-4);
    let (left, right, above, below) = probes(&engine.render_to_image(), 60);
    assert!(
        !left && !right && above && below,
        "turned, the same bar should run up the screen"
    );
}

/// The mirror is a mirror: paint on the left of the canvas shows on the right of the
/// screen, and the canvas point under a given screen point moves with it — which is
/// what keeps the pen landing where the artist is looking.
#[test]
fn mirroring_swaps_the_sides() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    // Only to the left of the origin.
    bar(&mut engine, -90.0, -40.0);
    let (left, right, ..) = probes(&engine.render_to_image(), 65);
    assert!(left && !right, "the bar was painted on the left");

    engine.process(ViewCommand::MirrorH);
    let (left, right, ..) = probes(&engine.render_to_image(), 65);
    assert!(!left && right, "mirrored, it should read on the right");

    let middle = Vec2::new(
        engine.view().viewport.width as f32 * 0.5,
        engine.view().viewport.height as f32 * 0.5,
    );
    let probe = middle + Vec2::new(65.0, 0.0);
    assert!(
        engine.view().screen_to_canvas(probe).x < 0.0,
        "mirrored, the right of the screen has to be the left of the canvas"
    );
}

/// Turning and mirroring the easel changes nothing about the piece: the document is
/// untouched, and a file — or the navigator's overview, which frames itself the same
/// way — comes out upright and pixel-identical.
#[test]
fn the_view_never_reaches_the_document() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    bar(&mut engine, -90.0, 40.0);
    engine.process(stark_core::command::DocCommand::AddMatte {
        carrier: None,
        above: None,
        region: MatteRegion::OutsideRect {
            min: Vec2::new(-100.0, -60.0),
            max: Vec2::new(100.0, 60.0),
        },
        color: [0.0, 0.0, 0.0],
    });
    let frame = engine.observe().layers.last().expect("matte").id;
    let shot = |engine: &mut Engine| {
        pollster::block_on(
            engine
                .export(
                    &mut Offscreen::default(),
                    Some(frame),
                    ExportScale::Factor(1.0),
                    Background::Substrate,
                    Rendered::Committed,
                )
                .expect("export"),
        )
    };
    let upright = shot(&mut engine);
    let before = engine.observe();

    let up = engine.view().rotation_for_up(Vec2::new(0.6, -0.4)).unwrap();
    engine.process(ViewCommand::SetRotation(up));
    engine.process(ViewCommand::MirrorH);

    let after = engine.observe();
    assert_eq!(
        before.doc_revision, after.doc_revision,
        "no document changed"
    );
    assert_eq!(
        before.can_undo, after.can_undo,
        "nothing entered the history"
    );
    let turned = shot(&mut engine);
    assert!(
        images_match(&upright, &turned, 0),
        "an export must show the piece, not the easel"
    );
}

/// A stroke lands under the pen at any orientation. The gesture arrives already in
/// canvas space — the frontend maps it through the same view — so this is really a
/// check that the two directions of that map agree with what is rendered.
#[test]
fn a_stroke_lands_under_the_pen_on_a_turned_canvas() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    let up = engine.view().rotation_for_up(Vec2::new(-1.0, 0.0)).unwrap();
    engine.process(ViewCommand::SetRotation(up));
    engine.process(ViewCommand::MirrorH);
    engine.process(ViewCommand::SetBrush(brush(RED, 14.0)));

    // Two screen points, mapped to canvas the way the frontend maps a pointer.
    let view = engine.view();
    let (a, b) = (Vec2::new(60.0, 128.0), Vec2::new(196.0, 128.0));
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(view.screen_to_canvas(a)),
        tolerance: DEFAULT_TOLERANCE,
    });
    engine.process(GestureCommand::To {
        sample: InputSample::at(view.screen_to_canvas(b)),
    });
    engine.process(GestureCommand::End);

    // It has to come back out along the line the pen drew, on screen.
    let img = engine.render_to_image();
    for x in [70, 128, 186] {
        let c = img.pixel(x, 128);
        assert!(is_red(c), "no paint at screen ({x}, 128): {c:?}");
    }
    for y in [70, 186] {
        let c = img.pixel(128, y);
        assert!(!is_red(c), "paint away from the line at (128, {y}): {c:?}");
    }
}

/// Grab-and-drag means the content follows the pointer — at any angle. The pan
/// delta arrives in *screen* px, so a turned canvas has to carry it back through the
/// whole map rather than dividing by the zoom.
#[test]
fn panning_follows_the_pointer_on_a_turned_canvas() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    let up = engine.view().rotation_for_up(Vec2::new(1.0, -2.0)).unwrap();
    engine.process(ViewCommand::SetRotation(up));
    let anchor = Vec2::new(200.0, 90.0);
    let delta = Vec2::new(35.0, -20.0);
    // The canvas point that should end up under the anchor: the one that is under
    // where the pointer *was* before it dragged there.
    let grabbed = engine.view().screen_to_canvas(anchor - delta);
    engine.process(ViewCommand::Pan { delta });
    let landed = engine.view().screen_to_canvas(anchor);
    assert!(
        (landed - grabbed).length() < 1e-2,
        "content did not follow the drag: {grabbed:?} -> {landed:?}"
    );
}

/// The mirror is *exact* geometry — no angle, no resampling — so what it does to the
/// screen can be checked pixel for pixel: mirrored pixel `x` has to be upright pixel
/// `W − 1 − x`, because both look at the same canvas point.
///
/// That makes it the sharpest check available on the screen path at a non-identity
/// orientation, and it covers all three separately-written copies of the view's map:
/// the composite's canvas→NDC matrix, the matte shader's own inverse of it, and the
/// media pass's screen→canvas matrix for the canvas weave. A sign wrong in any one of
/// them shows up here and nowhere else — each was checked by breaking it.
///
/// The weave is on, and its **embossing** deliberately off. With the light fixed to
/// the room and the canvas moving under it, a mirrored canvas genuinely catches the
/// light differently — the shading is not mirrored, and must not be, or turning the
/// easel would stop changing how impasto reads, which is half of why anyone turns it.
/// That is a real ~130-level difference and it is the lighting answering correctly, so
/// it is taken out of the question here; what is left of the weave is where it is
/// *sampled*, which is exactly the thing under test.
#[test]
fn mirroring_reflects_every_pixel_of_the_screen_path() {
    use stark_core::command::DocCommand;

    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    // The weave, embossed hard enough to see — this is what `surf_m` carries.
    let linen = engine
        .import_surface(&stark_testdata::assets::linen())
        .expect("the linen height map imports");
    engine.process(DocCommand::SetSurface(linen));
    engine.process(ViewCommand::SetMediaParams(stark_core::MediaParams {
        // The weave *on*, so where it is sampled is part of the answer, but its
        // embossing *off* — see the note above.
        surface_strength: 1.0,
        height_strength: 0.0,
        ..Default::default()
    }));
    // Paint off to one side, so the reflection has something asymmetric to move…
    bar(&mut engine, -90.0, 20.0);
    // …and a frame, so the matte's own inverse of the view is in the picture too.
    engine.process(DocCommand::AddMatte {
        carrier: None,
        above: None,
        region: MatteRegion::OutsideRect {
            min: Vec2::new(-70.0, -50.0),
            max: Vec2::new(90.0, 40.0),
        },
        color: [0.0, 0.0, 0.0],
    });

    // At two orientations, because the mirror is *screen*-relative: it has to swap the
    // left of the screen with the right whatever angle the canvas is at, and a
    // canvas-relative one would pass the upright case and fail the turned one.
    for turn in [0.0, std::f32::consts::FRAC_PI_2] {
        engine.process(ViewCommand::SetRotation(turn));
        let before = engine.render_to_image();
        engine.process(ViewCommand::MirrorH);
        let mirrored = engine.render_to_image();

        let (w, h) = (before.width, before.height);
        let mut worst = 0i32;
        for y in 0..h {
            for x in 0..w {
                let a = mirrored.pixel(x, y);
                let b = before.pixel(w - 1 - x, y);
                for c in 0..4 {
                    worst = worst.max((a[c] as i32 - b[c] as i32).abs());
                }
            }
        }
        assert!(
            worst <= 2,
            "turned {turn}: the mirrored screen is not the reflection of it (worst {worst})"
        );
        // Back to unmirrored for the next orientation — twice is the identity.
        engine.process(ViewCommand::MirrorH);
    }
}
