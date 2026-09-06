//! `Engine::new_sharing` (§11): a second engine over the first one's pipelines,
//! assets and decoded-resource caches, with a document of its own.
//!
//! The claims worth pinning are the two the sharing could silently break: the
//! shared machinery must render **identically** (a preview engine exists to show
//! strokes exactly as the donor's canvas would — sharing is only sound if shared
//! pipelines and pools change nothing about the pixels), and the *documents* must
//! stay isolated (painting on the preview must not put paint on the canvas).
//! Plus the seams the frontends lean on: the asset store and the resource caches
//! are live-shared, not snapshots, and `export_view` frames exactly the view it
//! is handed.

mod common;

use common::palette::RED;
use common::*;
use stark_engine::Extent2;
use stark_engine::ViewTransform;
use stark_engine::command::DocCommand;
use stark_engine::{Background, Engine, Offscreen, Rendered};
use stark_model::geom::Vec2;

/// The suites' standard diagonal, replayed identically on both engines.
fn diagonal(engine: &mut Engine) {
    paint(
        engine,
        RED,
        24.0,
        &[Vec2::new(-80.0, -80.0), Vec2::new(80.0, 80.0)],
    );
}

/// A stroke rendered by a sharing engine is the stroke the donor renders —
/// bit-identical, same device, same submission stream. This is the whole bargain:
/// if shared pipelines, pools or caches could shift a pixel, a "preview" through
/// them would be a picture of something else.
#[test]
fn a_shared_engine_renders_the_donor_s_pixels() {
    let Some(mut donor) = engine_or_skip() else {
        return;
    };
    let mut shared = Engine::new_sharing(&donor, SIZE);
    diagonal(&mut donor);
    diagonal(&mut shared);
    let a = donor.render_to_image();
    let b = shared.render_to_image();
    assert!(
        images_match(&a, &b, 0),
        "shared machinery must not change a pixel"
    );
}

/// The documents are isolated: painting on the sharing engine leaves the donor's
/// canvas untouched, byte for byte — the preview never leaks onto the painting.
#[test]
fn painting_on_a_shared_engine_leaves_the_donor_alone() {
    let Some(mut donor) = engine_or_skip() else {
        return;
    };
    let before = donor.render_to_image();
    let mut shared = Engine::new_sharing(&donor, SIZE);
    diagonal(&mut shared);
    shared.process(DocCommand::Undo);
    diagonal(&mut shared);
    let after = donor.render_to_image();
    assert!(
        images_match(&before, &after, 0),
        "a preview stroke reached the donor's document"
    );
}

/// The asset store is **live**-shared, not copied at construction: a stamp
/// imported on the donor after the sharing engine exists is already in it, under
/// the same content id. This is what lets the brush editor drop its
/// copy-the-bytes-over dance.
#[test]
fn brush_assets_are_shared_live() {
    let Some(donor) = engine_or_skip() else {
        return;
    };
    let shared = Engine::new_sharing(&donor, SIZE);
    let png = gray_png(16, 16, 255);
    let id = donor.import_brush(&png).expect("import a valid stamp");
    assert!(
        shared.has_asset(id),
        "an import on the donor must be visible to the sibling"
    );
    assert_eq!(shared.asset_bytes(id), donor.asset_bytes(id));
}

/// The substrate byte/build cache is live-shared the same way: a height map imported
/// on the donor can be served (and stood on) by the sibling with no bytes handed
/// across.
#[test]
fn substrates_are_shared_live() {
    let Some(mut donor) = engine_or_skip() else {
        return;
    };
    let shared = Engine::new_sharing(&donor, SIZE);
    let png = gray_png(64, 64, 128);
    let id = donor
        .import_substrate(&png)
        .expect("import a valid substrate");
    assert!(shared.substrate_bytes(id).is_some());
    assert_eq!(shared.substrate_bytes(id), donor.substrate_bytes(id));
}

/// The compositor's view settings stayed per-engine when the passes moved behind
/// the shared `Arc`: tuning the sibling's lighting must not move the donor's.
#[test]
fn media_params_diverge_per_engine() {
    let Some(donor) = engine_or_skip() else {
        return;
    };
    let mut shared = Engine::new_sharing(&donor, SIZE);
    let before = donor.observe().media.height_strength;
    let mut media = shared.observe().media;
    media.height_strength = before + 1.0;
    shared.process(stark_engine::command::ViewCommand::SetMediaParams(media));
    assert_eq!(
        donor.observe().media.height_strength,
        before,
        "a sibling's media params leaked into the donor"
    );
    assert_eq!(
        shared.observe().media.height_strength,
        before + 1.0,
        "the sibling's own setting did not take"
    );
}

/// `export_view` renders the view it is handed: the caller's viewport becomes the
/// image size, and the caller's centre is the image's centre. (Distinct from
/// `export`, whose framing is derived from the document and tile-aligned in the
/// fallback — the crop a thumbnail must not inherit.)
#[test]
fn export_view_frames_the_callers_view() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let at = Vec2::new(300.0, -140.0);
    paint(&mut engine, RED, 20.0, &[at, at + Vec2::new(1.0, 0.0)]);
    let size = Extent2::new(64, 32);
    let view = ViewTransform {
        center: at,
        zoom: 1.0,
        ..ViewTransform::identity(size)
    };
    let readback = engine
        .export_view(
            &mut Offscreen::default(),
            view,
            None,
            Background::Substrate,
            Rendered::Committed,
        )
        .expect("a finite view within device limits");
    let img = pollster::block_on(readback).expect("the readback completes");
    assert_eq!((img.width, img.height), (64, 32));
    assert!(
        red_dominant(center(&img)),
        "the paint at the view's centre must land at the image's centre"
    );
    // And the guard refuses what the device would have panicked on.
    let huge = ViewTransform {
        center: at,
        zoom: 1.0,
        ..ViewTransform::identity(Extent2::new(1 << 20, 8))
    };
    assert!(
        engine
            .export_view(
                &mut Offscreen::default(),
                huge,
                None,
                Background::Substrate,
                Rendered::Committed
            )
            .is_err()
    );
}

/// A minimal grayscale PNG — a valid stamp or height map without reaching for
/// bundled assets.
fn gray_png(w: u32, h: u32, level: u8) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, w, h);
        enc.set_color(png::ColorType::Grayscale);
        enc.set_depth(png::BitDepth::Eight);
        let mut writer = enc.write_header().expect("png header");
        writer
            .write_image_data(&vec![level; (w * h) as usize])
            .expect("png data");
    }
    out
}

/// **An engine built from a held `EngineShared` is the same engine
/// `new_sharing` builds** — the point of exposing the facet at all.
///
/// The value outlives the engine it came from, which is what a consumer wanted:
/// the preset thumbnails held a live renderer purely to reach the device, so they
/// could not run before one existed. Here the donor is *dropped* before the
/// sibling is built, which is the strongest form of that claim and one
/// `new_sharing(&donor, ..)` cannot make at all.
#[test]
fn a_held_shared_half_outlives_the_engine_it_came_from() {
    let Some(mut donor) = engine_or_skip() else {
        return;
    };
    diagonal(&mut donor);
    let reference = donor.render_to_image();

    let shared = donor.shared();
    drop(donor);

    let mut sibling = Engine::on_shared(shared, SIZE);
    diagonal(&mut sibling);
    let got = sibling.render_to_image();

    assert_eq!((got.width, got.height), (reference.width, reference.height),);
    assert_eq!(
        got.pixels, reference.pixels,
        "an engine on a held shared half painted a different stroke",
    );
}

/// The shared half can seed **many** siblings, and each keeps its own document.
///
/// The isolation claim above, made once the shared half is a value that can be
/// cloned freely rather than a donor that has to be borrowed one call at a time.
#[test]
fn one_shared_half_seeds_independent_engines() {
    let Some(donor) = engine_or_skip() else {
        return;
    };
    let shared = donor.shared();

    let mut a = Engine::on_shared(shared.clone(), SIZE);
    let b = Engine::on_shared(shared, SIZE);
    diagonal(&mut a);

    // `b` never saw a stroke, so it must still be bare canvas — and `a`'s paint
    // must not have reached it through the shared pool.
    assert!(
        a.document().bounds().tile_range().is_some(),
        "the painted engine has no extent",
    );
    assert_eq!(
        b.document().bounds().tile_range(),
        None,
        "a sibling picked up the other's paint",
    );
}
