//! The floating Brush panel: the everyday quick controls. The full parameter set
//! lives in the brush editor dialog.

use dioxus::prelude::*;

use crate::state::{AppState, update_brush};
use crate::widgets::Slider;
use stark_core::document::{BrushDynamics, BrushShape, OrientationSource};

/// Built-in assets, bundled as static files and **fetched at runtime** so they
/// stay out of the wasm binary (DESIGN.md §6.6). The engine is handed the bytes.
pub const BRISTLE_BRUSH: Asset = asset!("/assets/shape/WornBristles.png");

/// The maximum brush radius (`BrushParams::radius`).
pub const MAX_RADIUS: f32 = 500.0;

/// The floating Brush panel: the everyday quick controls (shape, size, opacity,
/// amount). Everything else — the full grouped parameter set with a live test
/// stroke — lives in the brush editor dialog ("Edit brush…").
#[component]
pub fn BrushPanel() -> Element {
    let state = use_context::<AppState>();
    let brush = state
        .obs
        .read()
        .as_ref()
        .map(|o| o.brush)
        .unwrap_or_default();
    let is_round = matches!(brush.shape, BrushShape::Round);

    let chip = |active: bool| if active { "chip active" } else { "chip" };

    rsx! {
        div { class: "brush-shapes",
            button {
                class: chip(is_round),
                onclick: move |_| set_shape(state, BrushShape::Round),
                "Round"
            }
            button {
                class: chip(!is_round),
                onclick: move |_| set_bristles(state),
                "Bristles"
            }
        }
        Slider { label: "Size", min: 1.0, max: MAX_RADIUS, value: brush.radius,
            oninput: move |v| update_brush(state, move |b| b.radius = v) }
        Slider { label: "Opacity", min: 0.0, max: 1.0, value: brush.color[3],
            oninput: move |v| update_brush(state, move |b| b.color[3] = v) }
        Slider { label: "Amount", min: 0.0, max: 1.5, value: brush.dynamics.add,
            oninput: move |v| update_brush(state, move |b| b.dynamics.add = v) }
        button {
            class: "be-open",
            onclick: move |_| {
                let mut open = state.brush_editor_open;
                open.set(true);
            },
            "Edit brush\u{2026}"
        }
    }
}

/// Switch to a shape, also setting a sensible default spacing for it.
pub fn set_shape(state: AppState, shape: BrushShape) {
    update_brush(state, move |b| b.shape = shape);
}

/// Set what orients the brush shape as it sweeps (DESIGN.md §6.6).
pub fn set_orientation(state: AppState, orientation: OrientationSource) {
    update_brush(state, move |b| b.orientation = orientation);
}

/// Edit the unified brush dynamics in place (DESIGN.md §6.2).
pub fn set_dyn(state: AppState, f: impl FnOnce(&mut BrushDynamics)) {
    update_brush(state, move |b| f(&mut b.dynamics));
}

/// Reset to the everyday brush: lay the brush's own paint, manipulate nothing.
pub fn set_brush_preset(state: AppState) {
    set_dyn(state, |d| *d = BrushDynamics::default());
}

/// The palette knife (DESIGN.md §6.2): no own paint (`add = 0`), a finite pre-`charge`
/// it carries, and it scrapes what it rides over (`lift`) back down as it goes
/// (`deposit`). A hard edge so it reads as a blade riding the weave.
pub fn set_knife(state: AppState) {
    update_brush(state, |b| {
        b.shape = BrushShape::Round;
        b.hardness = 0.9;
        b.dynamics = BrushDynamics {
            add: 0.0,
            lift: 1.0,
            deposit: 0.6,
            charge: 0.5,
        };
    });
}

/// Select the built-in bristle brush. It's fetched + imported once at startup
/// (DESIGN.md §6.6), so this is a no-op until those bytes have loaded.
pub fn set_bristles(state: AppState) {
    let id = state.renderer.read().as_ref().and_then(|r| r.bristle());
    let Some(id) = id else { return };
    set_shape(state, BrushShape::Stamp(id));
}
