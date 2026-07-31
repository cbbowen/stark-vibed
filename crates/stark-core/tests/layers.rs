//! Step-6a layer tests (§6, build order 6a): active-layer painting,
//! per-layer opacity/visibility, reordering, and undo of layer operations.

mod common;

use common::*;
use stark_core::command::DocCommand;
use stark_core::document::LayerId;
use stark_core::geom::Vec2;
use stark_core::{Engine, RgbaImage};

const RED: [f32; 4] = [0.85, 0.1, 0.1, 1.0];
const GREEN: [f32; 4] = [0.1, 0.8, 0.2, 1.0];

const ROOT: LayerId = LayerId(0);
const TOP: LayerId = LayerId(1);

const H_STROKE: &[Vec2] = &[Vec2::new(-25.0, 0.0), Vec2::new(25.0, 0.0)];
const V_STROKE: &[Vec2] = &[Vec2::new(0.0, -25.0), Vec2::new(0.0, 25.0)];

fn center(img: &RgbaImage) -> [u8; 4] {
    img.pixel(img.width / 2, img.height / 2)
}
fn red_dominant(c: [u8; 4]) -> bool {
    c[0] as i32 > c[1] as i32 + 30 && c[0] as i32 > c[2] as i32 + 30
}
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
        above: Some(TOP),
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
