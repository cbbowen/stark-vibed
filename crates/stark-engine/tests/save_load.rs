//! Step-5 save/load + timelapse tests (§8).
//!
//! Completes the replay-equivalence set from step 3: save → load reproduces the
//! exact pixels, undo works after loading, the saved file is compact, and a
//! timelapse yields one frame per action ending at the final image.

mod common;

use common::*;
use stark_engine::Engine;
use stark_engine::command::Tool;
use stark_engine::command::{DocCommand, GestureCommand, InputSample, ViewCommand};
use stark_engine::path::DEFAULT_TOLERANCE;
#[cfg(feature = "mixbox")]
use stark_model::ColorSpaceId;
use stark_model::SurfaceId;
use stark_model::document::BrushShape;
use stark_model::geom::Vec2;

const RED: [f32; 4] = [0.85, 0.1, 0.1, 1.0];
const GREEN: [f32; 4] = [0.1, 0.8, 0.2, 1.0];

const STROKE_A: &[Vec2] = &[Vec2::new(-40.0, -20.0), Vec2::new(40.0, 20.0)];
const STROKE_B: &[Vec2] = &[Vec2::new(-40.0, 40.0), Vec2::new(40.0, -40.0)];

/// A dry, toothed stroke: it reaches only for the ground's peaks, so its mark *is*
/// the ground and a missing height map is visible in the pixels.
fn paint_toothed(engine: &mut Engine) {
    stroke_with(
        engine,
        stark_model::document::BrushParams {
            color: RED,
            radius: 30.0,
            tooth: 0.55,
            drain: 0.15,
            ..Default::default()
        },
        &[Vec2::new(-60.0, 0.0), Vec2::new(60.0, 0.0)],
    );
}

fn paint_two(engine: &mut Engine) {
    paint(engine, RED, 30.0, STROKE_A);
    paint(engine, GREEN, 30.0, STROKE_B);
}

#[test]
fn save_load_roundtrip_is_lossless() {
    let Some(mut original) = engine_or_skip_blue() else {
        return;
    };
    paint_two(&mut original);
    let before = original.render_to_image();
    let bytes = original.save_bytes().expect("serialize");

    let mut loaded = engine_or_skip_blue().expect("adapter available (original built)");
    loaded.load_bytes(&bytes).expect("deserialize + replay");
    let after = loaded.render_to_image();

    assert!(
        images_match(&before, &after, 0),
        "save → load must reproduce identical pixels"
    );
    // The full undo timeline is available after loading (undo-after-load).
    assert!(loaded.observe().can_undo);
}

#[test]
fn undo_after_load_drops_last_stroke() {
    let Some(mut original) = engine_or_skip_blue() else {
        return;
    };
    paint_two(&mut original);
    let bytes = original.save_bytes().expect("serialize");

    // Reference: a document that only ever had stroke A.
    let mut just_a = engine_or_skip_blue().expect("adapter");
    paint(&mut just_a, RED, 30.0, STROKE_A);
    let only_a = just_a.render_to_image();

    // Load both strokes, then undo the second.
    let mut loaded = engine_or_skip_blue().expect("adapter");
    loaded.load_bytes(&bytes).expect("load");
    loaded.process(DocCommand::Undo);
    let undone = loaded.render_to_image();

    assert!(
        images_match(&undone, &only_a, 0),
        "undo after load must drop the last stroke exactly"
    );
}

#[test]
fn timelapse_yields_one_frame_per_action() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    paint_two(&mut engine);
    let file = engine.document_file();
    let final_image = engine.render_to_image();

    let mut frames = Vec::new();
    engine
        .replay_timelapse(&file, |frame| frames.push(frame))
        .expect("timelapse");

    assert_eq!(frames.len(), file.actions.len(), "one frame per action");
    assert!(
        images_match(frames.last().unwrap(), &final_image, 0),
        "last timelapse frame must equal the fully-replayed image"
    );
}

/// A timelapse replays through the **document's** color space, not through whichever
/// one the engine running it happened to be in.
///
/// The channel layouts differ between spaces (§6.7), so replaying a Mixbox document
/// on an Oklab engine is not a slightly-off picture but a stroke deposited through
/// the wrong shaders and lit by the wrong media pass. Loading has always matched the
/// space first; the timelapse had the preamble written out separately and did not, so
/// the two are driven from the same fresh Oklab engine here — a timelapse that agrees
/// with the load is one that adopted the file rather than approximated it.
/// Mixbox-only, so it exists only in a build carrying the `mixbox` feature.
/// `ColorSpaceId::Mixbox` still *names* a space there — the save format's enum
/// indices cannot depend on a feature (§8) — but nothing can open one.
#[cfg(feature = "mixbox")]
#[test]
fn a_timelapse_replays_in_the_documents_color_space() {
    let Some(mut original) = engine_or_skip_with(ColorSpaceId::Mixbox).map(on_blue) else {
        return;
    };
    paint_two(&mut original);
    let file = original.document_file();
    let expected = original.render_to_image();

    // Two fresh Oklab engines: one loads, one timelapses. Both must land on the
    // Mixbox picture.
    let mut loaded = engine_or_skip_blue().expect("adapter available");
    loaded.load_document(&file).expect("load");
    assert!(
        images_match(&loaded.render_to_image(), &expected, 0),
        "loading already matches the document's color space"
    );

    let mut lapsing = engine_or_skip_blue().expect("adapter available");
    let mut frames = Vec::new();
    lapsing
        .replay_timelapse(&file, |frame| frames.push(frame))
        .expect("timelapse");
    assert!(
        images_match(frames.last().expect("frames"), &expected, 0),
        "and so must a timelapse — same file, same preamble, same pixels"
    );
}

#[test]
fn brush_assets_survive_save_load() {
    let Some(mut original) = engine_or_skip_blue() else {
        return;
    };
    // Paint with an image brush shape (the asset lives only in this engine).
    let id = original
        .import_brush(&stark_testdata::assets::bristles())
        .expect("import");
    let mut brush = brush(RED, 60.0);
    brush.shape = BrushShape::Stamp(id);
    original.process(ViewCommand::SetBrush(brush));
    original.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(Vec2::new(-70.0, 0.0)),
        tolerance: DEFAULT_TOLERANCE,
        rope: 0.0,
    });
    original.process(GestureCommand::To {
        sample: InputSample::at(Vec2::new(70.0, 0.0)),
    });
    original.process(GestureCommand::End);
    let before = original.render_to_image();

    let bytes = original.save_bytes().expect("serialize");

    // A fresh engine that never imported the brush must still reproduce it,
    // because the asset is bundled in the file.
    let mut loaded = engine_or_skip_blue().expect("adapter");
    loaded.load_bytes(&bytes).expect("load");
    let after = loaded.render_to_image();

    assert!(
        images_match(&before, &after, 0),
        "image-brush stroke must round-trip through save/load via bundled assets"
    );
}

#[test]
fn saved_file_is_compact() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    // A long, many-sample stroke — the kind that could bloat a file.
    let path: Vec<Vec2> = (0..400)
        .map(|i| Vec2::new(-100.0 + i as f32 * 0.5, (i as f32 * 0.2).sin() * 30.0))
        .collect();
    paint(&mut engine, RED, 12.0, &path);

    let bytes = engine.save_bytes().expect("serialize");
    let rendered_bytes = (SIZE.width * SIZE.height * 4) as usize;
    assert!(
        bytes.len() < rendered_bytes,
        "action log ({}) should be far smaller than a raster ({rendered_bytes})",
        bytes.len()
    );
    eprintln!("400-sample stroke document: {} bytes", bytes.len());
}

/// The canvas surface is document state (§6.4), so a mid-document switch
/// is a logged action: it survives save/load, and undo takes it back.
///
/// `CanvasMeta::surface` records the surface the log *starts* from; the switch
/// itself rides in the log, which is why loading has to replay to learn the
/// current one rather than reading it off the header.
#[test]
fn a_surface_switch_is_historized() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    let start = engine.surface();
    assert_eq!(
        start,
        SurfaceId::Flat,
        "an engine holding no height map has exactly one ground it can name"
    );
    // A real ground to switch *to*, imported so its id exists at all (§6.4).
    let target = engine
        .import_surface(&stark_testdata::assets::linen())
        .expect("the linen height map imports");

    paint(
        &mut engine,
        RED,
        20.0,
        &[Vec2::new(-40.0, 0.0), Vec2::new(40.0, 0.0)],
    );
    engine.process(DocCommand::SetSurface(target));
    assert_eq!(engine.surface(), target);

    // Undo reaches it, because it is an action like any other.
    engine.process(DocCommand::Undo);
    assert_eq!(engine.surface(), start, "undo must revert the surface");
    engine.process(DocCommand::Redo);
    assert_eq!(engine.surface(), target);

    // And it round-trips: the header still says the document *started* on `start`.
    let file = engine.document_file();
    assert_eq!(
        file.canvas.surface, start,
        "CanvasMeta records the initial surface, not the current one"
    );
    let bytes = engine.save_bytes().expect("serialize");

    let Some(mut loaded) = engine_or_skip_blue() else {
        return;
    };
    loaded.load_bytes(&bytes).expect("load");
    assert_eq!(
        loaded.surface(),
        target,
        "replaying the log must land on the switched-to surface"
    );
}

/// **A document bundles every ground it names, not just the one it ends on**
/// (§6.4, §8).
///
/// The tooth reads whichever ground was in force when a stroke was made, so a
/// height map is a replay input exactly as a brush's coverage mask is — and a file
/// that carries only the last one cannot reproduce a document that switched
/// part-way. Checked on the container rather than on pixels because pixels cannot
/// distinguish a ground faithfully reproduced from one that fell back to the flat
/// stand-in; the bundle either holds the bytes or it does not.
#[test]
fn a_document_bundles_every_ground_it_names() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    let linen = engine
        .import_surface(&stark_testdata::assets::linen())
        .expect("the linen height map imports");
    let rough = engine
        .import_surface(&stark_testdata::assets::rough())
        .expect("the rough height map imports");
    assert_ne!(linen, rough, "two grounds, two content ids");

    // Paint across a switch, so the log names both and ends on neither of them
    // first: linen, then rough, then back to smooth.
    engine.process(DocCommand::SetSurface(linen));
    paint(
        &mut engine,
        RED,
        20.0,
        &[Vec2::new(-40.0, -20.0), Vec2::new(40.0, -20.0)],
    );
    engine.process(DocCommand::SetSurface(rough));
    paint(
        &mut engine,
        RED,
        20.0,
        &[Vec2::new(-40.0, 20.0), Vec2::new(40.0, 20.0)],
    );
    engine.process(DocCommand::SetSurface(SurfaceId::Flat));

    let file = engine.document_file();
    let bundled: Vec<SurfaceId> = file.surfaces.iter().map(|(id, _)| *id).collect();
    assert!(
        bundled.contains(&linen) && bundled.contains(&rough),
        "both grounds the log names must ride with it, got {bundled:?}"
    );
    assert!(
        !bundled.contains(&SurfaceId::Flat),
        "`Flat` is procedural and has no bytes to bundle"
    );
    for (id, bytes) in &file.surfaces {
        assert!(!bytes.is_empty(), "{id:?} was bundled with no height map");
    }

    // And the bundle survives the container, which is what a loader reads.
    let encoded = engine.save_bytes().expect("serialize");
    let back = stark_model::DocumentFile::from_bytes(&encoded).expect("deserialize");
    assert_eq!(
        back.surfaces.len(),
        file.surfaces.len(),
        "the grounds must survive the round trip"
    );
}

/// Bytes offered under the wrong id are **refused**, not installed (§6.4).
///
/// This is the one way a content-addressed ground could still deposit the wrong
/// tooth: a peer or a file handing over an image that is not the one the id names.
/// Since the id is derived from the bytes on the way in, the disagreement is
/// detectable, so it is detected.
#[test]
fn a_ground_that_is_not_what_it_claims_is_refused() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    let linen = engine
        .import_surface(&stark_testdata::assets::linen())
        .expect("the linen height map imports");
    assert!(
        engine
            .accept_surface(linen, &stark_testdata::assets::rough())
            .is_err(),
        "rough's bytes must not install themselves as linen"
    );
    // The honest pairing still works, and is idempotent.
    assert_eq!(
        engine
            .accept_surface(linen, &stark_testdata::assets::linen())
            .expect("linen's own bytes are accepted under linen's id"),
        linen
    );
}

/// **A lean file names the ground it was painted on but does not carry it, and
/// still replays to the same pixels** (§8's version 6, §12.4).
///
/// The bundle is what a `.stark` file weighs — a log is fitted paths and a canvas
/// ground is megabytes — so a document painted on an asset the app ships with
/// carried a copy of it for nothing. The id stays in the file; only the bytes go.
///
/// What has to hold is that resolving the id back and installing it **before the
/// replay** lands on the identical picture. A toothed brush on the irregular rough ground
/// ground is the only configuration where getting that wrong shows up at all:
/// without the height map the deposition gate is 1.0 everywhere and the stroke
/// comes back smooth, into stored pixels that no later arrival un-bakes (§6.4).
#[test]
fn a_lean_file_replays_identically_once_its_content_is_resolved() {
    let Some(mut original) = engine_or_skip_blue() else {
        return;
    };
    let rough_bytes = stark_testdata::assets::rough();
    let rough = original
        .import_surface(&rough_bytes)
        .expect("import ground");
    original.process(DocCommand::SetSurface(rough));
    paint_toothed(&mut original);
    let expected = original.render_to_image();

    // Saved twice: once bundling everything, once leaving out the ground on the
    // promise that whoever opens it can produce that id.
    let SurfaceId::Image(ground_id) = rough else {
        panic!("an imported ground is an image");
    };
    let fat = original.save_bytes().expect("serialize");
    let lean = original
        .save_bytes_resolvable(&[ground_id])
        .expect("serialize lean");
    assert!(
        lean.len() * 4 < fat.len(),
        "leaving the ground out should dominate the file: {} vs {} bytes",
        lean.len(),
        fat.len()
    );

    let file = stark_model::DocumentFile::from_bytes(&lean).expect("decode");
    assert!(
        file.surfaces.is_empty(),
        "the ground was promised, so it should not be in the bundle"
    );

    // Opening it: the bill, settled before a single action replays.
    let mut loaded = engine_or_skip_blue().expect("adapter");
    let owed = loaded.unresolved_content(&file);
    assert_eq!(
        owed,
        vec![stark_model::AssetNeed::Ground(ground_id)],
        "a lean file has to say what it is missing, or the opener replays without it"
    );
    for need in &owed {
        loaded
            .accept_surface(need.surface().expect("a ground"), &rough_bytes)
            .expect("resolve locally");
    }
    loaded
        .load_document(&file)
        .expect("the bill was settled above");

    assert!(
        images_match(&loaded.render_to_image(), &expected, 0),
        "a resolved lean file must replay to the same pixels as the fat one"
    );
}

/// The other half: a file that bundles everything owes nothing, so the ordinary
/// open path is untouched by any of this.
#[test]
fn a_complete_file_owes_nothing() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    let rough = engine
        .import_surface(&stark_testdata::assets::rough())
        .expect("import ground");
    engine.process(DocCommand::SetSurface(rough));
    paint_toothed(&mut engine);

    let file = stark_model::DocumentFile::from_bytes(&engine.save_bytes().expect("serialize"))
        .expect("decode");
    let mut fresh = engine_or_skip_blue().expect("adapter");
    assert!(fresh.unresolved_content(&file).is_empty());
    fresh
        .load_document(&file)
        .expect("a complete file owes nothing");
    assert!(images_match(
        &fresh.render_to_image(),
        &engine.render_to_image(),
        0
    ));
}

/// **A lean file whose bill was never settled is refused, and the open document is
/// left exactly as it was.**
///
/// The companion to [`a_lean_file_replays_identically_once_its_content_is_resolved`],
/// and the reason that test's ordering is a guarantee rather than a convention. Logged
/// and returned as `Ok(())` instead, the load reports success and replays every toothed
/// stroke through the flat stand-in, into *stored* pixels that no later arrival
/// un-bakes (§6.4) — a document that opens perfectly smooth and gives no sign which
/// ground it was painted through.
///
/// Two claims, and the second is the one a log line could not make. It fails — with
/// the bill in the error, so a caller can act on it — and it fails *before* adopting
/// anything, so a refused open is not a half-replaced painting.
#[test]
fn an_unsettled_lean_file_is_refused_and_changes_nothing() {
    let Some(mut original) = engine_or_skip_blue() else {
        return;
    };
    let rough = original
        .import_surface(&stark_testdata::assets::rough())
        .expect("import ground");
    original.process(DocCommand::SetSurface(rough));
    paint_toothed(&mut original);
    let SurfaceId::Image(ground_id) = rough else {
        panic!("an imported ground is an image");
    };
    let lean = original
        .save_bytes_resolvable(&[ground_id])
        .expect("serialize lean");

    // A second engine with a painting of its own already open, and no rough ground.
    let mut opener = engine_or_skip_blue().expect("adapter");
    paint(&mut opener, GREEN, 30.0, STROKE_B);
    let before = opener.render_to_image();

    let err = opener
        .load_bytes(&lean)
        .expect_err("an unpaid bill must refuse");
    match err {
        stark_engine::EngineError::Document(stark_model::DocError::MissingContent(missing)) => {
            assert_eq!(
                missing,
                vec![stark_model::AssetNeed::Ground(ground_id)],
                "the refusal has to name what is owed, or the caller cannot settle it"
            )
        }
        other => panic!("expected MissingContent, got {other}"),
    }
    assert!(
        images_match(&opener.render_to_image(), &before, 0),
        "a refused load must leave the document that was open untouched"
    );
}

/// The bundled-asset table `stark-testdata` derives at runtime names the same ids the
/// engine's own importers do — which is the whole load-bearing claim behind a harness
/// being able to open a lean capture at all (§8, §19).
///
/// Two derivations, deliberately kept apart: a shape is hashed from its decoded
/// *coverage* and a ground from its *height*, so the same file filed as both earns two
/// ids. This asserts the round trip in the direction that matters — engine derives the
/// id, table hands back bytes that install under it — because the failure mode is
/// silent. A table keyed on the wrong hash resolves nothing, `unresolved_content` stays
/// non-empty, and the only symptom is a load that refuses.
#[test]
fn the_bundled_asset_table_agrees_with_the_engine_on_every_id() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let SurfaceId::Image(ground) = engine
        .import_surface(&stark_testdata::assets::rough())
        .expect("import ground")
    else {
        panic!("an imported ground is an image");
    };
    let shape = engine
        .import_brush(&stark_testdata::assets::bristles())
        .expect("import shape");

    for (what, id) in [("ground", ground), ("shape", shape)] {
        let bytes = stark_testdata::assets::bundled(id)
            .unwrap_or_else(|| panic!("the table ships no {what} for the id the engine derived"));
        // Installing under the id it was looked up by is the check: `accept_surface`
        // re-derives and refuses a mismatch, and `import_brush` returns the id it
        // actually computed.
        let landed = if what == "ground" {
            match engine.accept_surface(SurfaceId::Image(id), &bytes) {
                Ok(SurfaceId::Image(back)) => back,
                other => panic!("installing the {what} did not land on an image: {other:?}"),
            }
        } else {
            engine.import_brush(&bytes).expect("install the shape")
        };
        assert_eq!(
            landed, id,
            "the table's {what} bytes hash to a different id"
        );
    }
}
