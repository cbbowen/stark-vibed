//! Step-6a layer tests (DESIGN.md §6, build order 6a): active-layer painting,
//! per-layer opacity/visibility, reordering, and undo of layer operations.

mod common;

use common::*;
use stark_core::document::LayerId;
use stark_core::geom::Vec2;
use stark_core::{Engine, InputCommand, RgbaImage};

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
    engine.process(InputCommand::AddLayer { above: None });
    // AddLayer makes the new layer active.
    assert_eq!(engine.observe().active_layer, TOP);
    paint(engine, GREEN, 40.0, V_STROKE);
}

#[test]
fn active_layer_directs_paint_and_stacks_on_top() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    two_layers(&mut engine);

    let obs = engine.observe();
    assert_eq!(obs.layers.len(), 2, "root + added layer");

    // Green was painted on the top layer, so it wins at the center.
    assert!(green_dominant(center(&engine.render_to_image(BG))));
}

#[test]
fn hiding_a_layer_removes_its_contribution() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    two_layers(&mut engine);

    engine.process(InputCommand::SetLayerVisible(TOP, false));
    assert!(
        red_dominant(center(&engine.render_to_image(BG))),
        "hiding the green top layer reveals red beneath"
    );

    engine.process(InputCommand::SetLayerVisible(TOP, true));
    assert!(green_dominant(center(&engine.render_to_image(BG))));
}

#[test]
fn zero_opacity_layer_is_invisible() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    two_layers(&mut engine);

    engine.process(InputCommand::SetLayerOpacity(TOP, 0.0));
    assert!(red_dominant(center(&engine.render_to_image(BG))));

    // Undo the opacity change → green returns (layer ops are historized).
    engine.process(InputCommand::Undo);
    assert!(green_dominant(center(&engine.render_to_image(BG))));
}

#[test]
fn reordering_changes_which_layer_wins() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    two_layers(&mut engine);

    // Move the root (red) layer above the top (green) layer.
    engine.process(InputCommand::MoveLayer {
        id: ROOT,
        above: Some(TOP),
    });
    assert!(
        red_dominant(center(&engine.render_to_image(BG))),
        "red now sits on top"
    );

    engine.process(InputCommand::Undo);
    assert!(
        green_dominant(center(&engine.render_to_image(BG))),
        "undo restores green on top"
    );
}

#[test]
fn layer_state_survives_save_load() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    two_layers(&mut engine);
    engine.process(InputCommand::SetLayerOpacity(TOP, 0.4));
    let before = engine.render_to_image(BG);
    let bytes = engine.save_bytes().expect("serialize");

    let mut loaded = engine_or_skip().expect("adapter");
    loaded.load_bytes(&bytes).expect("load");
    let after = loaded.render_to_image(BG);

    assert!(
        images_match(&before, &after, 0),
        "layer ordering + opacity must round-trip through save/load"
    );
    assert_eq!(loaded.observe().layers.len(), 2);
}
