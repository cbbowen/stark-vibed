//! Step-5 save/load + timelapse tests (§8).
//!
//! Completes the replay-equivalence set from step 3: save → load reproduces the
//! exact pixels, undo works after loading, the saved file is compact, and a
//! timelapse yields one frame per action ending at the final image.

use crate::common::palette::{GREEN_SOFT, RED_SOFT};
use crate::common::*;
use stark_engine::Engine;
use stark_engine::command::DocCommand;
#[cfg(feature = "mixbox")]
use stark_model::ColorSpaceId;
use stark_model::SubstrateId;
use stark_model::document::BrushShape;
use stark_model::geom::Vec2;

const STROKE_A: &[Vec2] = &[Vec2::new(-40.0, -20.0), Vec2::new(40.0, 20.0)];
const STROKE_B: &[Vec2] = &[Vec2::new(-40.0, 40.0), Vec2::new(40.0, -40.0)];

/// A dry, toothed stroke: it reaches only for the substrate's peaks, so its mark *is*
/// the substrate and a missing height map is visible in the pixels.
fn paint_toothed(engine: &mut Engine) {
    stroke_with(
        engine,
        stark_model::document::BrushParams {
            effect: stark_model::document::BrushEffect::painted(RED_SOFT),
            size: 30.0,
            tooth: stark_model::document::ToothParams {
                give: 0.45,
                ..Default::default()
            },
            drain: 0.15,
            ..Default::default()
        },
        &[Vec2::new(-60.0, 0.0), Vec2::new(60.0, 0.0)],
    );
}

fn paint_two(engine: &mut Engine) {
    paint(engine, RED_SOFT, 30.0, STROKE_A);
    paint(engine, GREEN_SOFT, 30.0, STROKE_B);
}

#[test]
fn save_load_roundtrip_is_lossless() {
    let Some(mut original) = engine_or_skip_blue() else {
        return;
    };
    paint_two(&mut original);
    let before = original.render_to_image();
    let mut loaded = reloaded(&original);
    let after = loaded.render_to_image();

    assert_images(
        &before,
        &after,
        0,
        "save → load must reproduce identical pixels",
    );
    // The full undo timeline is available after loading (undo-after-load).
    assert!(loaded.observe().can_undo);
}

/// `ObservableState::edited` is the baseline half of "there is unsaved work here":
/// it says the committed document has moved since it *arrived*, whichever way it
/// arrived. A frontend that asks before the page unloads (`stark-dioxus-frontend`'s
/// `files::unsaved`, §11) is built on it, and what it must never do is object to
/// closing a canvas nobody has touched — a warning raised when there is nothing to
/// lose is one the user learns to click through before the day it is right.
///
/// The load is the case worth a test rather than the fresh document: `reset_document`
/// takes the baseline before the replay, and every action of that replay moves the
/// counter again, so a load that did not re-take it afterwards would report a
/// just-opened file as edited work.
#[test]
fn a_document_is_edited_only_once_it_moves_from_what_arrived() {
    // Not `engine_or_skip_blue`, whose substrate color is itself a committed change:
    // the claim here is about a document nothing has been done to at all.
    let Some(mut original) = engine_or_skip() else {
        return;
    };
    assert!(
        !original.observe().edited,
        "a document nobody has touched has not been edited"
    );
    paint_two(&mut original);
    assert!(original.observe().edited, "two strokes are two edits");
    let mut loaded = reloaded(&original);
    assert!(
        !loaded.observe().edited,
        "a document just loaded stands exactly as the file it came out of"
    );

    paint(&mut loaded, RED_SOFT, 30.0, STROKE_A);
    assert!(
        loaded.observe().edited,
        "a stroke after the load is work that file does not hold"
    );

    // And a new document is a fresh baseline, not the loaded one carried over.
    loaded
        .new_document(loaded.observe().color_space, SubstrateId::Flat)
        .expect("a new document on the flat substrate");
    assert!(
        !loaded.observe().edited,
        "replacing the document starts the count again"
    );
}

/// A history step with nothing to step over is not an edit: it leaves both
/// `edited` and `doc_revision` where they were, so neither the unsaved-work prompt
/// nor anything memoized on the revision is disturbed by a stray Ctrl+Z.
#[test]
fn a_step_with_nowhere_to_go_leaves_the_document_unedited() {
    let unmoved = |engine: &Engine, at: u64, what: &str| {
        let obs = engine.observe();
        assert!(!obs.edited, "{what} marked the document edited");
        assert_eq!(obs.doc_revision, at, "{what} moved the revision");
    };

    let Some(mut fresh) = engine_or_skip() else {
        return;
    };
    let at = fresh.observe().doc_revision;
    fresh.process(DocCommand::Undo);
    unmoved(&fresh, at, "an undo on a blank document");
    fresh.process(DocCommand::Redo);
    unmoved(&fresh, at, "a redo on a blank document");

    paint_two(&mut fresh);
    let mut loaded = reloaded(&fresh);
    let at = loaded.observe().doc_revision;
    loaded.process(DocCommand::Redo);
    unmoved(&loaded, at, "a redo straight after a load");
    let (applied, _) = loaded.scrub_range().expect("a solo document scrubs");
    loaded.process(DocCommand::Seek(applied));
    unmoved(&loaded, at, "a seek to where the playhead stands");
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
    paint(&mut just_a, RED_SOFT, 30.0, STROKE_A);
    let only_a = just_a.render_to_image();

    // Load both strokes, then undo the second.
    let mut loaded = engine_or_skip_blue().expect("adapter");
    loaded.load_bytes(&bytes).expect("load");
    loaded.process(DocCommand::Undo);
    let undone = loaded.render_to_image();

    assert_images(
        &undone,
        &only_a,
        0,
        "undo after load must drop the last stroke exactly",
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
    assert_images(
        frames.last().unwrap(),
        &final_image,
        0,
        "last timelapse frame must equal the fully-replayed image",
    );
    // A timelapse ends with the file loaded, and a loaded file is unedited.
    assert!(
        !engine.observe().edited,
        "a timelapse left the document it replayed marked edited"
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
    assert_images(
        &loaded.render_to_image(),
        &expected,
        0,
        "loading already matches the document's color space",
    );

    let mut lapsing = engine_or_skip_blue().expect("adapter available");
    let mut frames = Vec::new();
    lapsing
        .replay_timelapse(&file, |frame| frames.push(frame))
        .expect("timelapse");
    assert_images(
        frames.last().expect("frames"),
        &expected,
        0,
        "and so must a timelapse — same file, same preamble, same pixels",
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
    let mut brush = brush(RED_SOFT, 60.0);
    brush.shape = BrushShape::Stamp(id);
    stroke_with(
        &mut original,
        brush,
        &[Vec2::new(-70.0, 0.0), Vec2::new(70.0, 0.0)],
    );
    let before = original.render_to_image();

    // A fresh engine that never imported the brush must still reproduce it,
    // because the asset is bundled in the file.
    let mut loaded = reloaded(&original);
    let after = loaded.render_to_image();

    assert_images(
        &before,
        &after,
        0,
        "image-brush stroke must round-trip through save/load via bundled assets",
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
    paint(&mut engine, RED_SOFT, 12.0, &path);

    let bytes = engine.save_bytes().expect("serialize");
    let rendered_bytes = (SIZE.width * SIZE.height * 4) as usize;
    assert!(
        bytes.len() < rendered_bytes,
        "action log ({}) should be far smaller than a raster ({rendered_bytes})",
        bytes.len()
    );
    eprintln!("400-sample stroke document: {} bytes", bytes.len());
}

/// The canvas substrate is document state (§6.4), so a mid-document switch
/// is a logged action: it survives save/load, and undo takes it back.
///
/// `CanvasMeta::substrate` records the substrate the log *starts* from; the switch
/// itself rides in the log, which is why loading has to replay to learn the
/// current one rather than reading it off the header.
#[test]
fn a_substrate_switch_is_historized() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    let start = engine.substrate();
    assert_eq!(
        start,
        SubstrateId::Flat,
        "an engine holding no height map has exactly one substrate it can name"
    );
    // A real substrate to switch *to*, imported so its id exists at all (§6.4).
    let target = engine
        .import_substrate(&stark_testdata::assets::linen())
        .expect("the linen height map imports");

    paint(
        &mut engine,
        RED_SOFT,
        20.0,
        &[Vec2::new(-40.0, 0.0), Vec2::new(40.0, 0.0)],
    );
    engine.process(DocCommand::SetSubstrate(target));
    assert_eq!(engine.substrate(), target);

    // Undo reaches it, because it is an action like any other.
    engine.process(DocCommand::Undo);
    assert_eq!(engine.substrate(), start, "undo must revert the substrate");
    engine.process(DocCommand::Redo);
    assert_eq!(engine.substrate(), target);

    // And it round-trips: the header still says the document *started* on `start`.
    let file = engine.document_file();
    assert_eq!(
        file.canvas.substrate, start,
        "CanvasMeta records the initial substrate, not the current one"
    );
    let bytes = engine.save_bytes().expect("serialize");

    let Some(mut loaded) = engine_or_skip_blue() else {
        return;
    };
    loaded.load_bytes(&bytes).expect("load");
    assert_eq!(
        loaded.substrate(),
        target,
        "replaying the log must land on the switched-to substrate"
    );
}

/// **A document bundles every substrate it names, not just the one it ends on**
/// (§6.4, §8).
///
/// The tooth reads whichever substrate was in force when a stroke was made, so a
/// height map is a replay input exactly as a brush's coverage mask is — and a file
/// that carries only the last one cannot reproduce a document that switched
/// part-way. Checked on the container rather than on pixels because pixels cannot
/// distinguish a substrate faithfully reproduced from one that fell back to the flat
/// stand-in; the bundle either holds the bytes or it does not.
#[test]
fn a_document_bundles_every_substrate_it_names() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    let linen = engine
        .import_substrate(&stark_testdata::assets::linen())
        .expect("the linen height map imports");
    let rough = engine
        .import_substrate(&stark_testdata::assets::rough())
        .expect("the rough height map imports");
    assert_ne!(linen, rough, "two substrates, two content ids");

    // Paint across a switch, so the log names both and ends on neither of them
    // first: linen, then rough, then back to smooth.
    engine.process(DocCommand::SetSubstrate(linen));
    paint(
        &mut engine,
        RED_SOFT,
        20.0,
        &[Vec2::new(-40.0, -20.0), Vec2::new(40.0, -20.0)],
    );
    engine.process(DocCommand::SetSubstrate(rough));
    paint(
        &mut engine,
        RED_SOFT,
        20.0,
        &[Vec2::new(-40.0, 20.0), Vec2::new(40.0, 20.0)],
    );
    engine.process(DocCommand::SetSubstrate(SubstrateId::Flat));

    let file = engine.document_file();
    let bundled: Vec<SubstrateId> = file
        .content
        .iter()
        .filter_map(|(need, _)| need.substrate())
        .collect();
    assert!(
        bundled.contains(&linen) && bundled.contains(&rough),
        "both substrates the log names must ride with it, got {bundled:?}"
    );
    assert!(
        !bundled.contains(&SubstrateId::Flat),
        "`Flat` is procedural and has no bytes to bundle"
    );
    for (need, bytes) in &file.content {
        assert!(!bytes.is_empty(), "{need:?} was bundled with no bytes");
    }

    // And the bundle survives the container, which is what a loader reads.
    let encoded = engine.save_bytes().expect("serialize");
    let back = stark_model::DocumentFile::from_bytes(&encoded).expect("deserialize");
    assert_eq!(
        back.content, file.content,
        "the bundle must survive the round trip"
    );
}

/// Bytes offered under the wrong id are **refused**, not installed (§6.4).
///
/// This is the one way a content-addressed substrate could still deposit the wrong
/// tooth: a peer or a file handing over an image that is not the one the id names.
/// Since the id is derived from the bytes on the way in, the disagreement is
/// detectable, so it is detected.
#[test]
fn a_substrate_that_is_not_what_it_claims_is_refused() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    let linen = engine
        .import_substrate(&stark_testdata::assets::linen())
        .expect("the linen height map imports");
    assert!(
        engine
            .accept_substrate(linen, &stark_testdata::assets::rough())
            .is_err(),
        "rough's bytes must not install themselves as linen"
    );
    // The honest pairing still works, and is idempotent.
    assert_eq!(
        engine
            .accept_substrate(linen, &stark_testdata::assets::linen())
            .expect("linen's own bytes are accepted under linen's id"),
        linen
    );
}

/// **A brush's bytes are checked against the id they were filed under too** — the one
/// arm of the install that did not ask (§8, §19).
///
/// Every install goes through one door now ([`Engine::install`]), so a file offering a
/// stamp under someone else's id is refused the way a substrate's already was. It used
/// to be filed under whatever it *did* hash to and answered `Ok`: the need stayed unmet,
/// the stroke naming it drew with no shape at all, and nothing was said about it.
///
/// The second half is what the one door costs. It canonicalizes on the way in, so a
/// bundle's bytes have to come back out unchanged — otherwise loading a file and saving
/// it again would rewrite every stamp it carries.
///
/// [`Engine::install`]: stark_engine::Engine::install
#[test]
fn a_brush_that_is_not_what_it_claims_is_refused() {
    let (Some(donor), Some(mut opener)) = (engine_or_skip_blue(), engine_or_skip_blue()) else {
        return;
    };
    let bristles = stark_testdata::assets::bristles();
    let id = donor
        .import_brush(&bristles)
        .expect("the bristle stamp imports");
    let need = stark_model::AssetNeed::Brush(id);
    // A different shape, so its coverage hashes elsewhere — which is the whole premise.
    let impostor = stark_testdata::assets::flat_tip();

    let refused = opener.install(need, &impostor);
    assert!(
        matches!(
            refused,
            Err(stark_engine::EngineError::Document(
                stark_model::DocError::Misnamed { .. }
            ))
        ),
        "the flat tip must not install itself as the bristle stamp: {refused:?}"
    );
    assert!(
        !opener.holds(need),
        "nothing may be held under the id that asked"
    );

    // What the file actually carries: the store's own canonical form, which has to
    // survive being installed again.
    let canonical = donor
        .content_bytes(need)
        .expect("the store keeps a canonical form of what it imported");
    opener
        .install(need, &canonical)
        .expect("bytes under their own id install");
    assert_eq!(
        opener.content_bytes(need),
        Some(canonical),
        "a bundle's stamp bytes have to survive the install, or a re-save would rewrite them"
    );
}

/// **A lean file names the substrate it was painted on but does not carry it, and
/// still replays to the same pixels** (§8's version 6, §12.4).
///
/// The bundle is what a `.stark` file weighs — a log is fitted paths and a canvas
/// substrate is megabytes — so a document painted on an asset the app ships with
/// carried a copy of it for nothing. The id stays in the file; only the bytes go.
///
/// What has to hold is that resolving the id back and installing it **before the
/// replay** lands on the identical picture. A toothed brush on the irregular rough substrate
/// substrate is the only configuration where getting that wrong shows up at all:
/// without the height map the deposition gate is 1.0 everywhere and the stroke
/// comes back smooth, into stored pixels that no later arrival un-bakes (§6.4).
#[test]
fn a_lean_file_replays_identically_once_its_content_is_resolved() {
    let Some(mut original) = engine_or_skip_blue() else {
        return;
    };
    let rough_bytes = stark_testdata::assets::rough();
    let rough = on_substrate(&mut original, &rough_bytes);
    paint_toothed(&mut original);
    let expected = original.render_to_image();

    // Saved twice: once bundling everything, once leaving out the substrate on the
    // promise that whoever opens it can produce that id.
    let SubstrateId::Image(substrate_id) = rough else {
        panic!("an imported substrate is an image");
    };
    let fat = original.save_bytes().expect("serialize");
    let lean = original
        .save_bytes_resolvable(&[substrate_id])
        .expect("serialize lean");
    assert!(
        lean.len() * 4 < fat.len(),
        "leaving the substrate out should dominate the file: {} vs {} bytes",
        lean.len(),
        fat.len()
    );

    let file = stark_model::DocumentFile::from_bytes(&lean).expect("decode");
    assert!(
        !file
            .content
            .iter()
            .any(|(need, _)| need.substrate().is_some()),
        "the substrate was promised, so it should not be in the bundle"
    );

    // Opening it: the bill, settled before a single action replays.
    let mut loaded = engine_or_skip_blue().expect("adapter");
    let owed = loaded.unresolved_content(&file);
    assert_eq!(
        owed,
        vec![stark_model::AssetNeed::Substrate(substrate_id)],
        "a lean file has to say what it is missing, or the opener replays without it"
    );
    for need in &owed {
        loaded
            .accept_substrate(need.substrate().expect("a substrate"), &rough_bytes)
            .expect("resolve locally");
    }
    loaded
        .load_document(&file)
        .expect("the bill was settled above");

    assert_images(
        &loaded.render_to_image(),
        &expected,
        0,
        "a resolved lean file must replay to the same pixels as the fat one",
    );
}

/// The other half: a file that bundles everything owes nothing, so the ordinary
/// open path is untouched by any of this.
#[test]
fn a_complete_file_owes_nothing() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    on_substrate(&mut engine, &stark_testdata::assets::rough());
    paint_toothed(&mut engine);

    let file = stark_model::DocumentFile::from_bytes(&engine.save_bytes().expect("serialize"))
        .expect("decode");
    let mut fresh = engine_or_skip_blue().expect("adapter");
    assert!(fresh.unresolved_content(&file).is_empty());
    fresh
        .load_document(&file)
        .expect("a complete file owes nothing");
    assert_images(
        &fresh.render_to_image(),
        &engine.render_to_image(),
        0,
        "a complete file reloads as the document it was written from",
    );
}

/// **A lean file whose bill was never settled is refused, and the open document is
/// left exactly as it was.**
///
/// The companion to [`a_lean_file_replays_identically_once_its_content_is_resolved`],
/// and the reason that test's ordering is a guarantee rather than a convention. Logged
/// and returned as `Ok(())` instead, the load reports success and replays every toothed
/// stroke through the flat stand-in, into *stored* pixels that no later arrival
/// un-bakes (§6.4) — a document that opens perfectly smooth and gives no sign which
/// substrate it was painted through.
///
/// Two claims, and the second is the one a log line could not make. It fails — with
/// the bill in the error, so a caller can act on it — and it fails *before* adopting
/// anything, so a refused open is not a half-replaced painting.
#[test]
fn an_unsettled_lean_file_is_refused_and_changes_nothing() {
    let Some(mut original) = engine_or_skip_blue() else {
        return;
    };
    let rough = on_substrate(&mut original, &stark_testdata::assets::rough());
    paint_toothed(&mut original);
    let SubstrateId::Image(substrate_id) = rough else {
        panic!("an imported substrate is an image");
    };
    let lean = original
        .save_bytes_resolvable(&[substrate_id])
        .expect("serialize lean");

    // A second engine with a painting of its own already open, and no rough substrate.
    let mut opener = engine_or_skip_blue().expect("adapter");
    paint(&mut opener, GREEN_SOFT, 30.0, STROKE_B);
    let before = opener.render_to_image();

    let err = opener
        .load_bytes(&lean)
        .expect_err("an unpaid bill must refuse");
    match err {
        stark_engine::EngineError::Document(stark_model::DocError::MissingContent(missing)) => {
            assert_eq!(
                missing,
                vec![stark_model::AssetNeed::Substrate(substrate_id)],
                "the refusal has to name what is owed, or the caller cannot settle it"
            )
        }
        other => panic!("expected MissingContent, got {other}"),
    }
    assert_images(
        &opener.render_to_image(),
        &before,
        0,
        "a refused load must leave the document that was open untouched",
    );
}

/// The bundled-asset table `stark-testdata` derives at runtime names the same ids the
/// engine's own importers do — which is the whole load-bearing claim behind a harness
/// being able to open a lean capture at all (§8, §19).
///
/// Two derivations, deliberately kept apart: a shape is hashed from its decoded
/// *coverage* and a substrate from its *height*, so the same file filed as both earns two
/// ids. This asserts the round trip in the direction that matters — engine derives the
/// id, table hands back bytes that install under it — because the failure mode is
/// silent. A table keyed on the wrong hash resolves nothing, `unresolved_content` stays
/// non-empty, and the only symptom is a load that refuses.
#[test]
fn the_bundled_asset_table_agrees_with_the_engine_on_every_id() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let SubstrateId::Image(substrate) = engine
        .import_substrate(&stark_testdata::assets::rough())
        .expect("import substrate")
    else {
        panic!("an imported substrate is an image");
    };
    let shape = engine
        .import_brush(&stark_testdata::assets::bristles())
        .expect("import shape");

    for (what, id) in [("substrate", substrate), ("shape", shape)] {
        let bytes = stark_testdata::assets::bundled(id)
            .unwrap_or_else(|| panic!("the table ships no {what} for the id the engine derived"));
        // Installing under the id it was looked up by is the check: `accept_substrate`
        // re-derives and refuses a mismatch, and `import_brush` returns the id it
        // actually computed.
        let landed = if what == "substrate" {
            match engine.accept_substrate(SubstrateId::Image(id), &bytes) {
                Ok(SubstrateId::Image(back)) => back,
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
