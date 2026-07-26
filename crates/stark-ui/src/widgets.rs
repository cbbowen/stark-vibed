//! Small reusable controls shared by the panels and the brush editor.

use dioxus::prelude::*;

use crate::platform::capture_pointer;

#[component]
pub fn Slider(
    label: String,
    min: f32,
    max: f32,
    value: f32,
    oninput: EventHandler<f32>,
) -> Element {
    rsx! {
        div { class: "slider-row",
            div { class: "slider-label", "{label}" }
            input {
                class: "slider",
                r#type: "range", min: "{min}", max: "{max}", step: "any", value: "{value}",
                oninput: move |e| {
                    if let Ok(v) = e.value().parse::<f32>() { oninput.call(v); }
                },
            }
        }
    }
}

// `TernaryPad` and these three constants are complete and styled (`.ternary*` in
// stark.css) but **nothing renders the component** — it is wired to no panel. Kept
// rather than deleted pending a call on whether it is still wanted; see CLEANUP.md.
/// Ternary pad triangle, on screen (px): width matches the colour field, height makes
/// it equilateral (`w·√3/2`). Mirrored by `.ternary`/`.ternary-tri` in stark.css.
#[allow(dead_code)]
const TRI_W: f32 = 220.0;
#[allow(dead_code)]
const TRI_H: f32 = 190.0;
/// Vertical room above/below the ternary triangle for its vertex labels (px).
#[allow(dead_code)]
const TRI_LBL: f32 = 16.0;

/// Ternary (barycentric) pad: drag a marker inside a triangle whose vertices are the
/// pure axes (`labels[0]` top, `labels[1]` bottom-left, `labels[2]` bottom-right).
/// Reports the marker's barycentric coordinates — weights ≥ 0 summing to 1 — so three
/// knobs whose common scale is redundant (overactuated against a separate rate/strength
/// control) collapse to the two degrees of freedom that matter. Controlled: the marker
/// tracks `value` (normalized defensively, so legacy non-normalized params display
/// sensibly) and every drag reports through `onchange`. Used for the dry brush's
/// add/lift/deposit; the wet brush's mix gets the same treatment (DESIGN.md §6.2).
#[component]
pub fn TernaryPad(
    labels: [String; 3],
    value: [f32; 3],
    onchange: EventHandler<[f32; 3]>,
) -> Element {
    let mut picking = use_signal(|| false);

    // Marker position from the (normalized) weights: p = Σ wᵢ·Vᵢ over the triangle's
    // vertices V₀=(W/2, 0), V₁=(0, H), V₂=(W, H), shifted down by the label band.
    let s: f32 = value.iter().sum();
    let v = if s > 1e-4 {
        value.map(|x| x / s)
    } else {
        [1.0, 0.0, 0.0]
    };
    let mx = v[0] * TRI_W * 0.5 + v[2] * TRI_W;
    let my = (v[1] + v[2]) * TRI_H + TRI_LBL;

    let pick = move |e: &Event<PointerData>| {
        let c = e.element_coordinates();
        onchange.call(ternary_weights(c.x as f32, c.y as f32 - TRI_LBL));
    };

    rsx! {
        div {
            class: "ternary",
            // Pointer capture keeps the drag streaming here while the button is held,
            // even outside the pad (weights clamp onto the simplex); the drag ends on
            // up/cancel, never on leaving the bounds.
            onpointerdown: move |e| { capture_pointer(&e); picking.set(true); pick(&e); },
            onpointermove: move |e| { if picking() { pick(&e); } },
            onpointerup: move |_| picking.set(false),
            onpointercancel: move |_| picking.set(false),
            div { class: "ternary-tri" }
            div { class: "ternary-label ternary-top", "{labels[0]}" }
            div { class: "ternary-label ternary-left", "{labels[1]}" }
            div { class: "ternary-label ternary-right", "{labels[2]}" }
            div { class: "ternary-marker", style: "left:{mx}px; top:{my}px;" }
        }
    }
}

/// Barycentric weights of a pointer position in the ternary triangle's local space
/// (origin at the label band's bottom-left, vertices as in [`TernaryPad`]). Positions
/// outside the triangle clamp onto it: negative weights drop to 0 and the rest
/// renormalize — so dragging past an edge or vertex pins the opposite weights to
/// exactly 0, which is also how a pure single- or two-axis mix is dialled in.
#[allow(dead_code)] // see the note on TRI_W: TernaryPad is not wired to any panel
pub fn ternary_weights(px: f32, py: f32) -> [f32; 3] {
    let (x0, y0) = (TRI_W * 0.5, 0.0f32);
    let (x1, y1) = (0.0f32, TRI_H);
    let (x2, y2) = (TRI_W, TRI_H);
    let denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
    let w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom;
    let w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom;
    let w = [w0, w1, 1.0 - w0 - w1].map(|x| x.max(0.0));
    let s: f32 = w.iter().sum();
    if s > 0.0 {
        w.map(|x| x / s)
    } else {
        [1.0, 0.0, 0.0]
    }
}

// --- command dispatch ---
