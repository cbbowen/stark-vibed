//! Step-6a layer tests (§6, build order 6a): active-layer painting,
//! per-layer opacity/visibility, reordering, and undo of layer operations.

mod common;

use common::*;
use stark_engine::Engine;
use stark_engine::command::{DocCommand, ViewCommand};
use stark_engine::document::{BlendMode, DRAGO_K, LayerId, Place};
use stark_model::geom::Vec2;

const RED: [f32; 4] = [0.85, 0.1, 0.1, 1.0];
const GREEN: [f32; 4] = [0.1, 0.8, 0.2, 1.0];

const ROOT: LayerId = LayerId(0);
const TOP: LayerId = LayerId(1);

const H_STROKE: &[Vec2] = &[Vec2::new(-25.0, 0.0), Vec2::new(25.0, 0.0)];
const V_STROKE: &[Vec2] = &[Vec2::new(0.0, -25.0), Vec2::new(0.0, 25.0)];

fn green_dominant(c: [u8; 4]) -> bool {
    c[1] as i32 > c[0] as i32 + 30 && c[1] as i32 > c[2] as i32 + 30
}

/// Paint red on the root layer, then add a layer and paint green on it. Both
/// strokes cross the canvas origin (screen center), green on top.
fn two_layers(engine: &mut Engine) {
    paint(engine, RED, 40.0, H_STROKE);
    engine.process(DocCommand::AddLayer {
        carrier: None,
        above: None,
    });
    // AddLayer makes the new layer active.
    assert_eq!(engine.observe().active_layer, TOP);
    paint(engine, GREEN, 40.0, V_STROKE);
}

#[test]
fn active_layer_directs_paint_and_stacks_on_top() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    two_layers(&mut engine);

    let obs = engine.observe();
    assert_eq!(obs.layers.len(), 2, "root + added layer");

    // Green was painted on the top layer, so it wins at the center.
    assert!(green_dominant(center(&engine.render_to_image())));
}

#[test]
fn hiding_a_layer_removes_its_contribution() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    two_layers(&mut engine);

    engine.process(DocCommand::SetLayerVisible(TOP, false));
    assert!(
        red_dominant(center(&engine.render_to_image())),
        "hiding the green top layer reveals red beneath"
    );

    engine.process(DocCommand::SetLayerVisible(TOP, true));
    assert!(green_dominant(center(&engine.render_to_image())));
}

#[test]
fn zero_opacity_layer_is_invisible() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    two_layers(&mut engine);

    engine.process(DocCommand::SetLayerOpacity(TOP, 0.0));
    assert!(red_dominant(center(&engine.render_to_image())));

    // Undo the opacity change → green returns (layer ops are historized).
    engine.process(DocCommand::Undo);
    assert!(green_dominant(center(&engine.render_to_image())));
}

#[test]
fn reordering_changes_which_layer_wins() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    two_layers(&mut engine);

    // Move the root (red) layer above the top (green) layer.
    engine.process(DocCommand::MoveLayer {
        carrier: None,
        id: ROOT,
        at: Place::Above(TOP),
    });
    assert!(
        red_dominant(center(&engine.render_to_image())),
        "red now sits on top"
    );

    engine.process(DocCommand::Undo);
    assert!(
        green_dominant(center(&engine.render_to_image())),
        "undo restores green on top"
    );
}

/// Duplicating a layer puts a copy of it directly above it, holding the same paint
/// and the same name, and arms the copy (§14.8).
///
/// The paint half is asserted by *hiding* the copy rather than by hiding the source:
/// the two layers hold the same stroke, and a stroke composited over itself is not
/// the stroke — its antialiased edge gains coverage. So "the copy is the only
/// difference" is the exact claim, and it is exact to the byte.
#[test]
fn duplicating_a_layer_copies_it_above_itself() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    two_layers(&mut engine);
    engine.process(DocCommand::SetLayerName(TOP, Some("Sky".into())));
    let before = engine.render_to_image();

    engine.process(DocCommand::DuplicateLayer(TOP));
    let obs = engine.observe();
    let copy = obs.active_layer;
    assert_eq!(obs.layers.len(), 3, "root, the layer, and its copy");
    assert_ne!(
        copy, TOP,
        "the copy is armed for the next stroke, not the source"
    );
    assert_eq!(
        obs.layers.iter().map(|l| l.id).collect::<Vec<_>>(),
        vec![ROOT, TOP, copy],
        "the copy sits directly above the layer it was copied from"
    );
    // The author's own word travels with the copy rather than being decorated into
    // one they never typed.
    assert_eq!(name_of(&engine, copy), Some("Sky".to_string()));

    engine.process(DocCommand::SetLayerVisible(copy, false));
    assert!(
        images_match(&before, &engine.render_to_image(), 0),
        "the copy holds exactly what the source holds"
    );

    engine.process(DocCommand::Undo);
    engine.process(DocCommand::Undo);
    assert_eq!(engine.observe().layers.len(), 2, "undo takes the copy back");
    assert!(images_match(&before, &engine.render_to_image(), 0));
}

/// The copy shares its source's tiles (copy-on-write, §5.2) — which must be
/// invisible: painting on one leaves the other alone.
#[test]
fn painting_on_a_copy_leaves_its_source_alone() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    two_layers(&mut engine);
    engine.process(DocCommand::DuplicateLayer(TOP));
    let copy = engine.observe().active_layer;

    // A red stroke over the copy's green, then hide the copy. The source must still
    // be green at the center: the stroke went to one layer's tiles, not to the pair
    // of handles they were sharing a moment ago.
    paint(&mut engine, RED, 20.0, V_STROKE);
    assert!(red_dominant(center(&engine.render_to_image())));

    engine.process(DocCommand::SetLayerVisible(copy, false));
    assert!(
        green_dominant(center(&engine.render_to_image())),
        "the source still holds only what it held"
    );
}

/// A duplicate mints an id per copied layer, and those ids are in the log — so a
/// reloaded document must resume its counter past them (§17.9). Reusing one would
/// put two layers under a single id, which `layer_index` resolves to whichever
/// comes first.
#[test]
fn a_duplicates_ids_are_not_reused_after_a_reload() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    two_layers(&mut engine);
    engine.process(DocCommand::DuplicateLayer(TOP));
    let bytes = engine.save_bytes().expect("serialize");

    let mut loaded = engine_or_skip_blue().expect("adapter");
    loaded.load_bytes(&bytes).expect("load");
    let existing: Vec<LayerId> = loaded.observe().layers.iter().map(|l| l.id).collect();
    assert_eq!(existing.len(), 3, "the copy came back with the log");

    loaded.process(DocCommand::AddLayer {
        carrier: None,
        above: None,
    });
    let added = loaded.observe().active_layer;
    assert!(
        !existing.contains(&added),
        "the next layer takes a fresh id: {added:?} against {existing:?}"
    );
}

/// **Undo can remove a layer too**, so the brush has to be repointed after one the
/// same way it is after a `RemoveLayer` (§17.9). `AddLayer` arms the layer it added;
/// undoing it withdraws exactly that layer, which leaves the most ordinary two-step
/// sequence in the app — add a layer, change your mind — pointing the brush at a
/// layer that no longer exists. `apply` then refuses every stroke silently, with
/// nothing on screen to say why.
#[test]
fn undoing_an_add_leaves_the_brush_somewhere_it_can_paint() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    engine.process(DocCommand::AddLayer {
        carrier: None,
        above: None,
    });
    assert_eq!(
        engine.observe().active_layer,
        TOP,
        "the add armed its layer"
    );

    engine.process(DocCommand::Undo);
    let obs = engine.observe();
    assert!(
        obs.layers.iter().any(|l| l.id == obs.active_layer),
        "the active layer must exist; got {:?} of {:?}",
        obs.active_layer,
        obs.layers.iter().map(|l| l.id).collect::<Vec<_>>(),
    );
    // And it can actually take paint, which is the property the repoint is for.
    paint(&mut engine, RED, 40.0, H_STROKE);
    assert!(
        engine.document().bounds().tile_range().is_some(),
        "the stroke after the undo landed nowhere",
    );
}

/// What `id` is called right now, or `None` if it has never been named.
fn name_of(engine: &Engine, id: LayerId) -> Option<String> {
    engine
        .observe()
        .layers
        .iter()
        .find(|l| l.id == id)
        .and_then(|l| l.name.as_ref().map(|n| n.to_string()))
}

#[test]
fn renaming_a_layer_is_undoable() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    two_layers(&mut engine);
    assert_eq!(name_of(&engine, TOP), None, "a new layer starts unnamed");

    engine.process(DocCommand::SetLayerName(TOP, Some("Sky".into())));
    assert_eq!(name_of(&engine, TOP), Some("Sky".to_string()));

    // A name is part of the document, so taking one back is an undo step — which is
    // what makes a mistyped rename recoverable the way a mis-set opacity is.
    engine.process(DocCommand::Undo);
    assert_eq!(name_of(&engine, TOP), None);
    engine.process(DocCommand::Redo);
    assert_eq!(name_of(&engine, TOP), Some("Sky".to_string()));

    // Clearing it is its own step rather than an undo of the rename.
    engine.process(DocCommand::SetLayerName(TOP, None));
    assert_eq!(name_of(&engine, TOP), None);
    engine.process(DocCommand::Undo);
    assert_eq!(name_of(&engine, TOP), Some("Sky".to_string()));
}

#[test]
fn a_name_is_either_absent_or_readable() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    two_layers(&mut engine);

    engine.process(DocCommand::SetLayerName(TOP, Some("  Sky  ".into())));
    assert_eq!(
        name_of(&engine, TOP),
        Some("Sky".to_string()),
        "surrounding whitespace is not part of the name"
    );

    // A field emptied out clears the name rather than setting a blank one, so the
    // row goes back to being described by its place in the stack.
    engine.process(DocCommand::SetLayerName(TOP, Some("   ".into())));
    assert_eq!(name_of(&engine, TOP), None);

    // A name is replicated and saved, so its length is bounded — by `char`s, so the
    // cut can never land inside one.
    let long = "\u{1F308}".repeat(200);
    engine.process(DocCommand::SetLayerName(TOP, Some(long)));
    let stored = name_of(&engine, TOP).expect("named");
    assert_eq!(stored.chars().count(), 64);
    assert!(stored.chars().all(|c| c == '\u{1F308}'));
}

#[test]
fn renaming_to_the_same_name_is_not_an_edit() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    two_layers(&mut engine);
    engine.process(DocCommand::SetLayerName(TOP, Some("Sky".into())));

    // Commit-on-blur means the frontend re-sends the current name whenever a field
    // is left untouched. That must cost nothing: an undo step that appears to do
    // nothing when reached is worse than no step at all.
    engine.process(DocCommand::SetLayerName(TOP, Some("Sky".into())));
    engine.process(DocCommand::SetLayerName(TOP, Some(" Sky ".into())));
    engine.process(DocCommand::Undo);
    assert_eq!(
        name_of(&engine, TOP),
        None,
        "one undo passes the whole rename, so only one step was logged"
    );
}

/// **Every setter makes the same bargain**, not just the four that were written
/// with a check of their own (§5.4).
///
/// `renaming_to_the_same_name_is_not_an_edit` states the rule for one command;
/// this states it for the ones that had no check at all, and they are the ordinary
/// ones — a visibility toggle, a clip toggle, the canvas color. Setting a value to
/// the value it already holds spent an undo step that appears to do nothing when
/// reached, which is the failure that rule exists to prevent, and which of the
/// commands avoided it was an accident of who wrote which arm.
///
/// Counted through `scrub_range`, whose applied count *is* the number of logged
/// steps — a cheaper and more exact question than walking undos back.
#[test]
fn setting_a_value_to_the_one_it_already_holds_is_not_an_edit() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    two_layers(&mut engine);
    engine.process(DocCommand::SetLayerVisible(TOP, false));
    engine.process(DocCommand::SetLayerClip(TOP, true));

    let logged = |e: &Engine| e.scrub_range().expect("solo history").0;
    let before = logged(&engine);
    let picture = engine.render_to_image();

    // Each of these is the value the document already reads.
    engine.process(DocCommand::SetLayerVisible(TOP, false));
    engine.process(DocCommand::SetLayerClip(TOP, true));
    engine.process(DocCommand::SetLayerOpacity(TOP, 1.0));
    engine.process(DocCommand::SetBackground([
        BG.r as f32,
        BG.g as f32,
        BG.b as f32,
    ]));
    assert_eq!(
        logged(&engine),
        before,
        "a setter that changes nothing must log nothing",
    );
    // …and the document is where it was, which is what makes the claim above about
    // the log rather than about a refusal to apply.
    assert!(images_match(&picture, &engine.render_to_image(), 0));

    // The rule is not "never log": a real change still does.
    engine.process(DocCommand::SetLayerVisible(TOP, true));
    assert_eq!(logged(&engine), before + 1);
}

/// **A commit supersedes the unlogged drag**, whichever commit it is (§17.6).
///
/// The rule was written out at the commit sites that remembered it and absent from
/// thirteen that did not, so a drag left in flight while some *other* edit landed
/// pinned the canvas to the dragged value and shadowed it. `RemoveLayer` is the
/// case here because it is one of the thirteen and because its effect is
/// unmistakable: the layer being previewed at 0% opacity is the layer that goes
/// away, so a preview that survived the commit would still be drawing it.
#[test]
fn a_commit_supersedes_a_drag_it_knows_nothing_about() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    two_layers(&mut engine);
    // Only the root layer's red will be left once the top layer is gone.
    engine.process(DocCommand::SetLayerVisible(TOP, false));
    let without_top = engine.render_to_image();
    engine.process(DocCommand::SetLayerVisible(TOP, true));

    // A drag in flight on the root layer, never released.
    engine.process(ViewCommand::PreviewLayerOpacity(Some((ROOT, 0.0))));
    // An unrelated commit lands mid-drag.
    engine.process(DocCommand::RemoveLayer(TOP));
    assert!(
        images_match(&without_top, &engine.render_to_image(), 2),
        "the drag preview outlived the commit and is still fading the root layer",
    );
    assert_eq!(
        opacity_of(&engine, ROOT),
        Some(1.0),
        "and the projection agrees: nothing of the drag was kept",
    );
}

#[test]
fn layer_names_survive_save_load() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    two_layers(&mut engine);
    engine.process(DocCommand::SetLayerName(TOP, Some("Sky".into())));
    let bytes = engine.save_bytes().expect("serialize");

    let mut loaded = engine_or_skip_blue().expect("adapter");
    loaded.load_bytes(&bytes).expect("load");
    assert_eq!(name_of(&loaded, TOP), Some("Sky".to_string()));
    assert_eq!(name_of(&loaded, ROOT), None);
}

/// The opacity of a layer, off the projection the panel reads — which is the
/// *previewed* document while a drag is in flight.
fn opacity_of(engine: &Engine, id: LayerId) -> Option<f32> {
    engine
        .observe()
        .layers
        .iter()
        .find(|l| l.id == id)
        .map(|l| l.opacity)
}

/// The blend mode of a layer, off the same projection and for the same reason.
fn blend_of(engine: &Engine, id: LayerId) -> Option<BlendMode> {
    engine
        .observe()
        .layers
        .iter()
        .find(|l| l.id == id)
        .map(|l| l.blend)
}

/// `Radiance` at the bend a fresh layer wears — what a Bend drag opens on.
const RADIANCE: BlendMode = BlendMode::Drago { k: DRAGO_K };

/// An opacity drag previews live but logs once (§14.6) — the third rider on the
/// preview slot, beside the frame drag and the canvas color (`tests/matte.rs`),
/// and here for the reason those two are: a slider reports a value per pointer
/// *move*, so without this one adjustment buries the history under a hundred
/// one-percent-apart edits and undo stops being able to take it back.
#[test]
fn dragging_layer_opacity_previews_without_logging() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    two_layers(&mut engine);
    let opaque = engine.render_to_image();

    // Three "pointer moves" of a drag towards transparent.
    for v in [0.6f32, 0.3, 0.0] {
        engine.process(ViewCommand::PreviewLayerOpacity(Some((TOP, v))));
    }
    assert!(
        red_dominant(center(&engine.render_to_image())),
        "the preview should fade the green layer on screen"
    );
    // `observe` reports the previewed opacity, so the slider's own track agrees with
    // the canvas it controls instead of trailing a commit behind it.
    assert_eq!(opacity_of(&engine, TOP), Some(0.0));

    // Undoing now must reach the *stroke*, not a drag step, and drop the preview with
    // it — so undo-then-redo lands exactly back on the opaque document, which nothing
    // about the drag has been logged into.
    engine.process(DocCommand::Undo);
    engine.process(DocCommand::Redo);
    assert!(
        images_match(&opaque, &engine.render_to_image(), 2),
        "a history step during a drag should drop the preview and log nothing of it"
    );

    // Release: one commit, which supersedes the preview it matches.
    for v in [0.6f32, 0.3, 0.0] {
        engine.process(ViewCommand::PreviewLayerOpacity(Some((TOP, v))));
    }
    let dragging = engine.render_to_image();
    engine.process(DocCommand::SetLayerOpacity(TOP, 0.0));
    assert!(
        images_match(&dragging, &engine.render_to_image(), 2),
        "the committed opacity should match what the drag previewed"
    );
    engine.process(DocCommand::Undo);
    assert!(
        images_match(&opaque, &engine.render_to_image(), 2),
        "one undo should take back the whole drag"
    );
}

/// A drag that travels out and comes back is not an edit — the same rule
/// `renaming_to_the_same_name_is_not_an_edit` states, and the case that makes it
/// here is the one the frontend cannot avoid: a slider released on the value it was
/// pressed on still has to commit, because the preview it left up must be
/// superseded by *something*.
#[test]
fn an_opacity_drag_that_ends_where_it_started_logs_nothing() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    two_layers(&mut engine);
    let opaque = engine.render_to_image();

    for v in [0.6f32, 0.3, 1.0] {
        engine.process(ViewCommand::PreviewLayerOpacity(Some((TOP, v))));
    }
    engine.process(DocCommand::SetLayerOpacity(TOP, 1.0));
    assert!(
        images_match(&opaque, &engine.render_to_image(), 2),
        "the settled drag should leave the document as it found it, preview and all"
    );

    engine.process(DocCommand::Undo);
    assert!(
        red_dominant(center(&engine.render_to_image())),
        "one undo should reach the green stroke, so the settled drag logged no step"
    );
}

/// The **Bend** slider makes the same bargain, and it is worth asserting separately
/// rather than trusting the shape: it drags a parameter that lives inside the blend
/// mode (§6.3), so what previews and what commits is the whole `BlendMode`, and the
/// "unchanged commits nothing" rule now has to compare two modes rather than two
/// numbers. Both halves in one test, because they are one bargain.
#[test]
fn dragging_a_blend_parameter_previews_without_logging() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    two_layers(&mut engine);
    engine.process(DocCommand::SetLayerBlend(TOP, RADIANCE));
    let rest = engine.render_to_image();

    // Three "pointer moves" of a drag towards the hot end.
    for k in [1.0f32, 2.0, 4.0] {
        engine.process(ViewCommand::PreviewLayerBlend(Some((
            TOP,
            BlendMode::Drago { k },
        ))));
    }
    let dragging = engine.render_to_image();
    assert!(
        !images_match(&rest, &dragging, 2),
        "the preview should reach the canvas"
    );
    // `observe` reports the previewed mode, so the track agrees with the canvas.
    assert_eq!(blend_of(&engine, TOP), Some(BlendMode::Drago { k: 4.0 }));

    // Release: one commit, which supersedes the preview it matches.
    engine.process(DocCommand::SetLayerBlend(TOP, BlendMode::Drago { k: 4.0 }));
    assert!(
        images_match(&dragging, &engine.render_to_image(), 2),
        "the committed bend should match what the drag previewed"
    );
    engine.process(DocCommand::Undo);
    assert!(
        images_match(&rest, &engine.render_to_image(), 2),
        "one undo should take back the whole drag"
    );

    // …and a drag that travels out and comes back is not an edit at all.
    for k in [1.0f32, 2.0, DRAGO_K] {
        engine.process(ViewCommand::PreviewLayerBlend(Some((
            TOP,
            BlendMode::Drago { k },
        ))));
    }
    engine.process(DocCommand::SetLayerBlend(TOP, RADIANCE));
    assert!(
        images_match(&rest, &engine.render_to_image(), 2),
        "the settled drag should leave the document as it found it, preview and all"
    );
    engine.process(DocCommand::Undo);
    assert_eq!(
        blend_of(&engine, TOP),
        Some(BlendMode::Normal),
        "one undo should reach the mode itself, so the settled drag logged no step",
    );
}

#[test]
fn layer_state_survives_save_load() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    two_layers(&mut engine);
    engine.process(DocCommand::SetLayerOpacity(TOP, 0.4));
    let before = engine.render_to_image();
    let bytes = engine.save_bytes().expect("serialize");

    let mut loaded = engine_or_skip_blue().expect("adapter");
    loaded.load_bytes(&bytes).expect("load");
    let after = loaded.render_to_image();

    assert!(
        images_match(&before, &after, 0),
        "layer ordering + opacity must round-trip through save/load"
    );
    assert_eq!(loaded.observe().layers.len(), 2);
}

/// The layer projection is **the same list** for as long as the document it
/// describes stands still — the property that lets a frontend take a projection
/// after every command, including the pan and brush commands that arrive at
/// pointer rate and cannot move a layer (`Engine::projected_layers`).
///
/// Asserted on *identity* rather than on equality, because equality was always
/// true: what this pins is that the walk did not run again. `Layers` derefs to a
/// slice, so `as_ptr` is the address of the shared buffer — two projections
/// sharing one is exactly the cache having answered.
#[test]
fn a_still_document_projects_the_same_layer_list() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    two_layers(&mut engine);

    let first = engine.observe().layers;
    // Every command here is one a frontend sends at pointer rate, and none of them
    // is a change to the document.
    engine.process(ViewCommand::Pan {
        delta: Vec2::new(13.0, -7.0),
    });
    engine.process(ViewCommand::Zoom {
        anchor: Vec2::ZERO,
        factor: 1.25,
    });
    let after = engine.observe().layers;
    assert!(
        std::ptr::eq(first.as_ptr(), after.as_ptr()),
        "a view command rebuilt the layer projection"
    );
    assert_eq!(first, after, "and it is still the same list");
}

/// …and a **new** list whenever the document does move, by either of the two
/// routes the cache is keyed on: a committed edit, and an unlogged preview.
///
/// The second is the one worth a test. A preview replaces the *shown* document
/// without committing anything, so `doc_revision` does not stir — the preview's
/// own epoch is what says so, and a cache keyed on the revision alone would hand
/// back a list describing the document behind the drag.
#[test]
fn a_moved_document_projects_a_fresh_layer_list() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    two_layers(&mut engine);

    // A commit: the opacity really changes, so the list must too.
    let before = engine.observe().layers;
    engine.process(DocCommand::SetLayerOpacity(TOP, 0.5));
    let committed = engine.observe().layers;
    assert_ne!(before, committed, "a commit must project the new opacity");

    // A preview: nothing is logged, `doc_revision` stands, and the projection
    // still has to report what is on screen (§14.6).
    let revision = engine.observe().doc_revision;
    engine.process(ViewCommand::PreviewLayerOpacity(Some((TOP, 0.125))));
    let previewed = engine.observe();
    assert_eq!(
        previewed.doc_revision, revision,
        "a preview commits nothing"
    );
    let shown = previewed
        .layers
        .iter()
        .find(|l| l.id == TOP)
        .expect("the top layer")
        .opacity;
    assert!(
        (shown - 0.125).abs() < 1e-6,
        "the projection reported {shown}, not the preview"
    );
}
