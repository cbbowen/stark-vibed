//! The floating Drawing Guides panel (§20): the perspective grid.
//!
//! One camera, three familiar cases. The panel does not pretend "1-point",
//! "2-point" and "3-point" are different tools: the chips are *presets* that
//! turn the same camera, and which chip lights up is **derived from the
//! geometry** — the count of finite vanishing points — so dragging the sliders
//! between cases shows the cases flowing into each other rather than a mode
//! switching (§20.1).
//!
//! These sliders are the milestone's controls, not the tool's final shape:
//! direct manipulation of the vanishing points, the center of view and the 45°
//! circle on the canvas is designed separately (§20.5) and will subsume most
//! of them.

use dioxus::prelude::*;

use crate::state::{AppState, dispatch};
use crate::widgets::Slider;
use stark_core::PerspectiveGuide;
use stark_core::command::ViewCommand;

#[component]
pub fn GuidesPanel() -> Element {
    let state = use_context::<AppState>();
    // Read off the engine's own projection rather than a local copy: a shadow
    // seeded from `Default` goes stale the moment anything else changes the
    // guide (§4).
    let obs = state.obs.read();
    let g = obs.as_ref().map(|o| o.guide).unwrap_or_default();
    let view_center = obs
        .as_ref()
        .map(|o| o.view.center)
        .unwrap_or(stark_core::Vec2::ZERO);
    drop(obs);

    // Which classical case the camera *is*, read from the geometry: how many
    // axes vanish on the canvas. Roll does not change it, and no slider is
    // "the mode" — that is the point.
    let finite_vps = g.scene().vps.iter().flatten().count();
    let case_chip = |n: usize| {
        if g.enabled && finite_vps == n {
            "chip active"
        } else {
            "chip"
        }
    };
    let axis_chip = |on: bool| if on { "chip active" } else { "chip" };

    // A preset turns the camera to a canonical orientation for its case,
    // brings the grid to where you are looking, and switches it on — three
    // acts, because reaching for a preset *is* asking to see that grid.
    let preset = move |yaw_deg: f32, pitch_deg: f32| {
        update(state, move |g| {
            g.yaw = yaw_deg.to_radians();
            g.pitch = pitch_deg.to_radians();
            g.roll = 0.0;
            g.center = view_center;
            g.enabled = true;
        });
    };

    rsx! {
        div { class: "slider-row",
            div { class: "slider-label", "Grid" }
            button {
                class: axis_chip(g.enabled),
                title: "Draw the perspective grid over the canvas",
                onclick: move |_| update(state, |g| g.enabled = !g.enabled),
                if g.enabled { "On" } else { "Off" }
            }
            button {
                class: "chip",
                title: "Put the center of view where you are looking",
                onclick: move |_| update(state, move |g| g.center = view_center),
                "Recenter"
            }
        }
        div { class: "slider-row",
            div { class: "slider-label", "Case" }
            button {
                class: case_chip(1),
                title: "View straight down an axis: transverse lines stay parallel, \
                        one point vanishes at the center of view",
                onclick: move |_| preset(0.0, 0.0),
                "1-point"
            }
            button {
                class: case_chip(2),
                title: "View turned but level: verticals stay parallel, two points \
                        vanish on the horizon",
                onclick: move |_| preset(30.0, 0.0),
                "2-point"
            }
            button {
                class: case_chip(3),
                title: "View turned and tilted: all three axes vanish on the canvas",
                onclick: move |_| preset(30.0, 18.0),
                "3-point"
            }
        }
        // The lens. Also the radius of the 45° circle — they are the same
        // number, and the circle on the canvas is this slider made visible.
        Slider { label: "Focal", min: 250.0, max: 3000.0, value: g.focal,
            oninput: move |v| update(state, move |g| g.focal = v) }
        Slider { label: "Yaw", min: -180.0, max: 180.0, value: g.yaw.to_degrees(),
            oninput: move |v: f32| update(state, move |g| g.yaw = v.to_radians()) }
        Slider { label: "Pitch", min: -80.0, max: 80.0, value: g.pitch.to_degrees(),
            oninput: move |v: f32| update(state, move |g| g.pitch = v.to_radians()) }
        Slider { label: "Roll", min: -180.0, max: 180.0, value: g.roll.to_degrees(),
            oninput: move |v: f32| update(state, move |g| g.roll = v.to_radians()) }
        Slider { label: "Density", min: 4.0, max: 36.0, value: g.density as f32,
            oninput: move |v: f32| update(state, move |g| g.density = v.round() as u32) }
        Slider { label: "Opacity", min: 0.1, max: 1.0, value: g.opacity,
            oninput: move |v| update(state, move |g| g.opacity = v) }
        div { class: "slider-row",
            div { class: "slider-label", "Axes" }
            for (i, name) in ["X", "Y", "Z"].into_iter().enumerate() {
                button {
                    class: axis_chip(g.axes[i]),
                    title: "Show this world axis's fan of guide lines",
                    onclick: move |_| update(state, move |g| g.axes[i] = !g.axes[i]),
                    "{name}"
                }
            }
        }
    }
}

/// Read the current guide off the projection, adjust it, and push it back —
/// the same read-modify-commit shape as `update_media` (§4).
fn update(state: AppState, f: impl FnOnce(&mut PerspectiveGuide)) {
    let mut g = state
        .obs
        .read()
        .as_ref()
        .map(|o| o.guide)
        .unwrap_or_default();
    f(&mut g);
    dispatch(state, ViewCommand::SetGuide(g));
}
