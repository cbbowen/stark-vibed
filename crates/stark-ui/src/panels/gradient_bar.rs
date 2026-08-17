//! The shared gradient bar (§22.4): one interface for laying a ramp, wherever
//! it lands.
//!
//! The transform mode's shape, with a **target**: entered from the Selection
//! bar's Gradient button (fill the selection) or the frame bar's (repaint the
//! matte), it swaps the raising bar for this one — the ramp in hand,
//! Linear/Radial, Done — and mounts a full-viewport catcher where **the drag
//! is the axis**: press where the ramp starts, release where it ends, radial
//! reads the same drag as centre and reach. Every mutation funnels through one
//! [`update`], which previews the real result through the target's own preview
//! command — the same renderer as its commit, so what is shown is what "Done"
//! produces, and re-dragging replaces the preview rather than stacking.
//!
//! The two targets differ in one deliberate way. A **fill** reads its ramp live
//! off the library (the Gradients panel's highlighted row), so a click there
//! re-previews; a **matte** carries its ramp in the target, seeded from the
//! matte's own paint — re-composing an old gradient's axis must not silently
//! swap its colors for whatever the library happens to have selected. A
//! library click mid-mode still replaces it, because a click is a choice.

use dioxus::html::input_data::MouseButton;
use dioxus::prelude::*;

use crate::gradients;
use crate::icons::{self, icon, label};
use crate::input::{Nav, page_xy};
use crate::layout::chrome_class;
use crate::platform::capture_pointer;
use crate::preview;
use crate::state::{AppState, GradientAxisKind, GradientTarget, GradientUi};
use stark_model::Gradient;
use stark_model::document::{FillOp, GradientAxis, GradientParcel, MattePaint};
use stark_model::geom::Vec2;

/// Enter the mode for a **fill of the selection**. The target layer is the
/// transform's choice — the active layer if paintable, else the topmost
/// paintable. Refuses quietly when the library is empty: the bar's button is
/// disabled then, and says why.
///
/// Nothing about strength is captured, because the ramp lands through the
/// selection and the selection already carries it (§6.8).
pub fn begin_fill(state: AppState) {
    if gradients::current(state).is_none() {
        return;
    }
    // One composing mode at a time (`crate::modes`), the transform's rule.
    crate::modes::leave(state);
    let obs = state.obs.read();
    let Some(o) = obs.as_ref() else { return };
    let layer = o
        .layers
        .iter()
        .find(|l| l.id == o.active_layer && l.is_paintable())
        .or_else(|| o.layers.iter().rev().find(|l| l.is_paintable()))
        .map(|l| l.id);
    let Some(layer) = layer else { return };
    drop(obs);
    let mut mode = state.gradient_bar;
    mode.set(Some(GradientUi {
        target: GradientTarget::Fill { layer },
        kind: GradientAxisKind::Linear,
        drag: None,
    }));
}

/// Enter the mode for a **matte's paint** (§15.4), from the frame bar's
/// Gradient chip. A matte already wearing a gradient re-opens on its own ramp
/// and axis, so the mode edits what is there; a solid one starts on the
/// library's current ramp with a vertical axis across its rect (or the view,
/// for a ground) — a graded sky's default, previewed immediately so entering
/// the mode already shows something to adjust.
pub fn begin_matte(state: AppState, layer: stark_model::document::LayerId, paint: &MattePaint) {
    // One composing mode at a time (`crate::modes`), as for the fill above.
    crate::modes::leave(state);
    let (gradient, kind, drag) = match paint {
        MattePaint::Gradient { gradient, axis } => {
            let (kind, drag) = match axis {
                GradientAxis::Linear { from, to } => (GradientAxisKind::Linear, (*from, *to)),
                GradientAxis::Radial { center, radius } => (
                    GradientAxisKind::Radial,
                    (*center, *center + Vec2::new(*radius, 0.0)),
                ),
            };
            (gradient.clone(), kind, drag)
        }
        MattePaint::Solid(_) => {
            let Some(gradient) = gradients::current(state) else {
                return;
            };
            let (lo, hi) = default_axis_rect(state, layer);
            let cx = (lo.x + hi.x) * 0.5;
            (
                gradient,
                GradientAxisKind::Linear,
                (Vec2::new(cx, lo.y), Vec2::new(cx, hi.y)),
            )
        }
    };
    let ui = GradientUi {
        target: GradientTarget::Matte { layer, gradient },
        kind,
        drag: Some(drag),
    };
    update(state, ui);
}

/// The rect a fresh matte axis spans: the matte's own, or the view for a
/// ground that has none.
fn default_axis_rect(state: AppState, layer: stark_model::document::LayerId) -> (Vec2, Vec2) {
    let obs = state.obs.read();
    let matte_rect = obs
        .as_ref()
        .and_then(|o| o.layers.iter().find(|l| l.id == layer))
        .and_then(|l| l.matte.as_ref())
        .and_then(|m| m.rect);
    matte_rect.unwrap_or_else(|| {
        obs.as_ref()
            .map(|o| o.view.visible_bounds())
            .unwrap_or((Vec2::splat(-256.0), Vec2::splat(256.0)))
    })
}

/// Update the gesture and show its consequence — every mutation funnels through
/// here, so the preview can never lag the state (the transform's rule).
fn update(state: AppState, ui: GradientUi) {
    // Resolved against `ui` before it moves into the signal.
    let laid = compose(state, &ui);
    if laid.is_none() {
        // Nothing composed yet — drop whatever this target was showing. Order is
        // free here, unlike the show below: there is no preview to lag the state.
        clear_preview(state, &ui.target);
    }
    let mut mode = state.gradient_bar;
    mode.set(Some(ui));
    if let Some(laid) = laid {
        laid.show(state);
    }
}

/// Re-preview the composing gesture, if one is composing — how a gradient
/// picked in the panel mid-mode reaches the canvas. For a matte target the
/// click *replaces* the carried ramp (a click is a choice); for a fill the
/// ramp is read live anyway.
pub fn refresh(state: AppState) {
    let ui = state.gradient_bar.peek().clone();
    let Some(mut ui) = ui else { return };
    if let GradientTarget::Matte { gradient, .. } = &mut ui.target
        && let Some(current) = gradients::current(state)
    {
        *gradient = current;
    }
    update(state, ui);
}

/// A composed ramp, ready to go: **which layer, and the paint it lays there**.
///
/// The layer and the payload travel together, and that is the point. They used to
/// be resolved apart — a payload enum matched against the target it came from —
/// so every site had to spell out two arms that could not happen (a fill payload
/// on a matte target) alongside the two that could. Carried together the mismatch
/// is not a case to handle; it is a value that cannot be built.
enum Laid {
    Fill(stark_model::document::LayerId, FillOp),
    Matte(stark_model::document::LayerId, MattePaint),
}

impl Laid {
    /// Show it on the canvas, logging nothing.
    fn show(self, state: AppState) {
        match self {
            Laid::Fill(layer, op) => preview::FILL.show(state, (layer, op)),
            Laid::Matte(layer, paint) => preview::MATTE_PAINT.show(state, (layer, paint)),
        }
    }

    /// Lay it down — one undo step. The engine refuses an unchanged matte paint,
    /// so re-opening a gradient matte and pressing Done spends none.
    fn commit(self, state: AppState) {
        match self {
            Laid::Fill(layer, op) => preview::FILL.commit(state, (layer, op)),
            Laid::Matte(layer, paint) => preview::MATTE_PAINT.commit(state, (layer, paint)),
        }
    }
}

/// Drop whichever preview a gesture aimed at `target` is showing.
///
/// The one place that answers "which preview does this target use" for the case
/// where there is no [`Laid`] to ask — nothing composed yet, or the mode abandoned
/// (`crate::modes`). Public for that second caller.
pub fn clear_preview(state: AppState, target: &GradientTarget) {
    match target {
        GradientTarget::Fill { .. } => preview::FILL.clear(state),
        GradientTarget::Matte { .. } => preview::MATTE_PAINT.clear(state),
    }
}

/// What the current gesture would lay, or `None` before the drag has composed an
/// axis (or with no ramp in the library to lay).
fn compose(state: AppState, ui: &GradientUi) -> Option<Laid> {
    let axis = ui.axis()?;
    Some(match &ui.target {
        GradientTarget::Fill { layer } => Laid::Fill(
            *layer,
            FillOp::gradient_of_selection(GradientParcel {
                gradient: gradients::current(state)?,
                axis,
            }),
        ),
        GradientTarget::Matte { layer, gradient } => Laid::Matte(
            *layer,
            MattePaint::Gradient {
                gradient: gradient.clone(),
                axis,
            },
        ),
    })
}

/// Leave the mode: commit the composed result, or just drop the preview when
/// there is nothing to lay. The commit clears the preview itself, so there is
/// no intermediate frame showing the document without it (the transform's
/// Done). For a matte, re-selecting the layer is untouched — the frame bar
/// comes straight back.
fn finish(state: AppState) {
    let ui = state.gradient_bar.peek().clone();
    if let Some(ui) = ui {
        match compose(state, &ui) {
            Some(laid) => laid.commit(state),
            None => clear_preview(state, &ui.target),
        }
    }
    let mut mode = state.gradient_bar;
    mode.set(None);
}

/// The ramp the bar (and the axis chrome) is currently laying.
fn ramp_in_hand(state: AppState, ui: &GradientUi) -> Option<Gradient> {
    match &ui.target {
        GradientTarget::Fill { .. } => gradients::current(state),
        GradientTarget::Matte { gradient, .. } => Some(gradient.clone()),
    }
}

/// The mode's bar, standing in for whichever bar raised it.
#[component]
pub fn GradientBar() -> Element {
    let state = use_context::<AppState>();
    let Some(ui) = state.gradient_bar.read().clone() else {
        return rsx! {};
    };
    // Read reactively so a pick in the Gradients panel repaints the strip.
    let strip = ramp_in_hand(state, &ui).map(|g| gradients::css_strip(&g));
    let kind_chip = |kind: GradientAxisKind, glyph: &'static str, name: &'static str| {
        let active = ui.kind == kind;
        let chip_ui = ui.clone();
        rsx! {
            button {
                class: if active { "chip active" } else { "chip" },
                title: match kind {
                    GradientAxisKind::Linear => "The drag is the ramp: press at its start, release at its end",
                    GradientAxisKind::Radial => "The drag is the reach: press at the centre, release at the rim",
                },
                onclick: move |_| {
                    // Reinterpret the drag already made rather than losing it —
                    // `GradientUi::axis` reads the same two points either way.
                    update(state, GradientUi { kind, ..chip_ui.clone() });
                },
                {icon(glyph)}
                {label(name)}
            }
        }
    };

    let done_title = if ui.drag.is_some() {
        match ui.target {
            GradientTarget::Fill { .. } => "Lay the gradient (undo takes it back)",
            GradientTarget::Matte { .. } => "Keep this paint (undo takes it back)",
        }
    } else {
        "Nothing dragged yet \u{2014} leave without laying anything"
    };

    rsx! {
        div { class: chrome_class(state, "selection-bar gradient-fill-bar"),
            // The Gradients panel's mark: the bar is that library's ramp being
            // put to work, so it wears the library's glyph.
            span { class: "bar-label",
                {icon(icons::GRADIENT)}
                {label("Gradient")}
            }
            if let Some(strip) = strip {
                span {
                    class: "bar-gradient-strip",
                    title: "The ramp being laid \u{2014} pick another in the Gradients panel",
                    style: "background: {strip};",
                }
            }
            span { class: "bar-sep" }
            {kind_chip(GradientAxisKind::Linear, icons::GRADIENT_LINEAR, "Linear")}
            {kind_chip(GradientAxisKind::Radial, icons::GRADIENT_RADIAL, "Radial")}
            span { class: "bar-sep" }
            button {
                class: "chip",
                title: done_title,
                onclick: move |_| finish(state),
                {icon(icons::DONE)}
                {label("Done")}
            }
        }
    }
}

/// The mode's catcher and axis chrome: while composing, the pointer draws the
/// axis, it does not paint — the transform catcher's bargain, with `Nav` live
/// so the view stays reachable mid-compose.
#[component]
pub fn GradientBarOverlay() -> Element {
    let state = use_context::<AppState>();
    let mut dragging = use_signal(|| false);
    let nav = Nav::use_nav(state);

    let Some(ui) = state.gradient_bar.read().clone() else {
        return rsx! {};
    };
    let view = match state.obs.read().as_ref() {
        Some(o) => o.view,
        None => return rsx! {},
    };
    let to_canvas = move |e: &Event<PointerData>| view.screen_to_canvas(page_xy(e));

    let panning = (state.space_down)();
    let catcher_class = if panning {
        "gradient-catcher pan"
    } else {
        "gradient-catcher"
    };
    let down_ui = ui.clone();
    let move_ui = ui.clone();

    rsx! {
        div {
            class: "{catcher_class}",
            onpointerdown: move |e| {
                if nav.begin(&e) {
                    // Navigation takes the press; the composed axis stands — it
                    // is canvas-space, so the view moving under it costs nothing.
                    dragging.set(false);
                    return;
                }
                if e.trigger_button() != Some(MouseButton::Primary) {
                    return;
                }
                e.stop_propagation();
                capture_pointer(&e);
                dragging.set(true);
                let p = to_canvas(&e);
                update(state, GradientUi { drag: Some((p, p)), ..down_ui.clone() });
            },
            onpointermove: move |e| {
                if nav.advance(&e) || !dragging() {
                    return;
                }
                if let Some((from, _)) = move_ui.drag {
                    update(state, GradientUi { drag: Some((from, to_canvas(&e))), ..move_ui.clone() });
                }
            },
            onpointerup: move |e| if !nav.release(&e) { nav.stop(); dragging.set(false); },
            onpointercancel: move |e| if !nav.release(&e) { nav.stop(); dragging.set(false); },
            onwheel: move |e| nav.wheel(e),
        }

        if let Some((from, to)) = ui.drag {
            {axis_chrome(ui.kind, from, to, view)}
        }
    }
}

/// The axis, drawn in the trace's own visual language (dark casing, dashed
/// light core, a dot on the anchor): linear is the line the ramp runs along,
/// radial is the reach circle with its centre marked. A plain function chosen
/// by an `if`, like the transform's visuals.
fn axis_chrome(
    kind: GradientAxisKind,
    from: Vec2,
    to: Vec2,
    view: stark_model::ViewTransform,
) -> Element {
    let a = view.canvas_to_screen(from);
    let b = view.canvas_to_screen(to);
    match kind {
        GradientAxisKind::Linear => {
            let d = format!("M{:.2} {:.2} L{:.2} {:.2}", a.x, a.y, b.x, b.y);
            rsx! {
                svg { class: "gradient-trace-svg",
                    path { class: "gradient-trace-casing", d: "{d}" }
                    path { class: "gradient-trace-core", d: "{d}" }
                    circle { class: "gradient-trace-anchor", cx: "{a.x}", cy: "{a.y}", r: "4" }
                    circle { class: "gradient-trace-anchor", cx: "{b.x}", cy: "{b.y}", r: "3" }
                }
            }
        }
        GradientAxisKind::Radial => {
            // The circle is drawn about the centre with the *screen* radius; a
            // rotated or mirrored view moves the rim point, never the circle.
            let r = from.distance(to) * view.zoom;
            let d = format!("M{:.2} {:.2} L{:.2} {:.2}", a.x, a.y, b.x, b.y);
            rsx! {
                svg { class: "gradient-trace-svg",
                    circle { class: "gradient-axis-circle", cx: "{a.x}", cy: "{a.y}", r: "{r}" }
                    path { class: "gradient-trace-core", d: "{d}" }
                    circle { class: "gradient-trace-anchor", cx: "{a.x}", cy: "{a.y}", r: "4" }
                }
            }
        }
    }
}
