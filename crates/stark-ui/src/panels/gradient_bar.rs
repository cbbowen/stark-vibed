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
//! off the library (the pop-out's highlighted row), so a click there
//! re-previews; a **matte** carries its ramp in the target, seeded from the
//! matte's own paint — re-composing an old gradient's axis must not silently
//! swap its colors for whatever the library happens to have selected. A
//! library click mid-mode still replaces it, because a click is a choice.
//!
//! The library itself lives on this bar: the strip showing the ramp in hand is
//! a [`GradientWell`], and clicking it flies the library out — the rows to pick
//! from and the Trace that makes a new one (§22.3). Which is why an empty
//! library no longer refuses the mode: the bar is where ramps come *from*, so
//! it has to be reachable before there are any.

use dioxus::html::input_data::MouseButton;
use dioxus::prelude::*;

use crate::commands::{self, Command};
use crate::gradients;
use crate::icons::{self, icon, label};
use crate::input::{Nav, page_xy};
use crate::layout::chrome_class;
use crate::modes::Composing;
use crate::panels::gradients::GradientWell;
use crate::platform::capture_pointer;
use crate::preview;
use crate::state::{AppState, GradientAxisKind, GradientTarget, GradientUi, use_obs};
use crate::widgets::CommandButton;
use stark_model::Gradient;
use stark_model::document::{FillOp, GradientAxis, GradientParcel, MattePaint};
use stark_model::geom::Vec2;

/// Enter the mode for a **fill of the selection**. The target layer is the
/// transform's choice — the active layer if paintable, else the topmost
/// paintable. An empty library does not refuse: the ramp is read live, so
/// nothing previews until the bar's well supplies one — and the well is where
/// one is picked or traced.
///
/// Nothing about strength is captured, because the ramp lands through the
/// selection and the selection already carries it (§6.8).
pub fn begin_fill(state: AppState) {
    // One composing mode at a time (`crate::modes`), the transform's rule.
    crate::modes::leave(state);
    let obs = state.obs.peek();
    let Some(o) = obs.as_ref() else { return };
    let layer = o
        .layers
        .iter()
        .find(|l| l.id == o.active_layer && l.is_paintable())
        .or_else(|| o.layers.iter().rev().find(|l| l.is_paintable()))
        .map(|l| l.id);
    let Some(layer) = layer else { return };
    drop(obs);
    // `enter`, which puts down whatever else was composing — the only way in
    // (`crate::modes`). No preview yet: there is no drag, so there is no axis.
    crate::modes::enter(
        state,
        Composing::GradientFill(GradientUi {
            target: GradientTarget::Fill { layer },
            kind: GradientAxisKind::Linear,
            drag: None,
        }),
    );
}

/// Enter the mode for a **matte's paint** (§15.4), from the frame bar's
/// Gradient chip. A matte already wearing a gradient re-opens on its own ramp
/// and axis, so the mode edits what is there; a solid one starts on the
/// library's current ramp with a vertical axis across its rect (or the view,
/// for a ground) — a graded sky's default, previewed immediately so entering
/// the mode already shows something to adjust. With the library empty the mode
/// still opens, carrying no ramp: the bar's well is where the first one is
/// picked or traced, and the default axis stands ready for it.
pub fn begin_matte(state: AppState, layer: stark_model::document::LayerId, paint: &MattePaint) {
    // One composing mode at a time (`crate::modes`), as for the fill above —
    // and **before** the reads below rather than left to the `enter` inside
    // `open`, which is the ordering the defaults depend on: putting a mode down
    // drops its preview, and `default_axis_rect` asks the projection for a rect
    // that a preview still standing could be moving.
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
            (Some(gradient.clone()), kind, drag)
        }
        MattePaint::Solid(_) => {
            let (lo, hi) = default_axis_rect(state, layer);
            let cx = (lo.x + hi.x) * 0.5;
            (
                gradients::current(state),
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
    open(state, ui);
}

/// The rect a fresh matte axis spans: the matte's own, or the view for a
/// ground that has none.
fn default_axis_rect(state: AppState, layer: stark_model::document::LayerId) -> (Vec2, Vec2) {
    let obs = state.obs.peek();
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
    // `advance` replaces what the mode in hand is composing and refuses to change
    // *which* mode that is; entering is [`open`]'s (`crate::modes`).
    crate::modes::advance(state, Composing::GradientFill(ui.clone()));
    show_axis(state, &ui);
}

/// Enter the mode on `ui` and preview it — [`update`]'s other half, split from
/// it when the modes became one signal: the write that opens a mode puts down
/// whatever was already composing and the write that advances one must not, so
/// they cannot be the same call.
fn open(state: AppState, ui: GradientUi) {
    // The mode first, and this order is load-bearing: `enter` drops the
    // *previous* mode's preview, so a `show_axis` ahead of it would have its own
    // preview taken down again on the way in.
    crate::modes::enter(state, Composing::GradientFill(ui.clone()));
    show_axis(state, &ui);
}

/// Show what `ui` would lay, or drop the preview when it would lay nothing.
///
/// **After the signal, always**, which is why both callers above write the mode
/// first: the preview can then never lag the state it is a picture of.
fn show_axis(state: AppState, ui: &GradientUi) {
    match compose(state, ui) {
        Some(laid) => laid.show(state),
        None => clear_preview(state, &ui.target),
    }
}

/// Re-preview the composing gesture, if one is composing — how a gradient
/// picked in the pop-out mid-mode reaches the canvas. For a matte target the
/// click *replaces* the carried ramp (a click is a choice); for a fill the
/// ramp is read live anyway.
pub fn refresh(state: AppState) {
    let ui = crate::modes::composing_now(state).and_then(Composing::gradient_fill);
    let Some(mut ui) = ui else { return };
    if let GradientTarget::Matte { gradient, .. } = &mut ui.target
        && let Some(current) = gradients::current(state)
    {
        *gradient = Some(current);
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

/// Set the composing gesture aside for the **trace** (§22.2), and hand it to the
/// caller to hold — the half of the mode swap that is not an abandonment.
///
/// Arming a trace from this bar's own well is the artist reaching into the
/// library the bar is holding, not putting the bar down; but the trace mounts a
/// catcher of its own and two of those cannot share the canvas (`crate::modes`),
/// so the gesture — target, kind, and the axis already dragged — is parked
/// rather than lost. [`resume`] brings it back.
///
/// The preview goes down with it, and that is not tidiness: a capture samples the
/// **composite** (§22.2), so a fill preview left standing would be traced as if it
/// were paint and the new ramp would be fitted through the old one.
pub fn suspend(state: AppState) -> Option<GradientUi> {
    let held = crate::modes::composing_now(state).and_then(Composing::gradient_fill);
    if held.is_some() {
        // `leave` puts the mode down and drops its preview, which is the whole of
        // what suspending costs — the gesture itself is the thing being handed
        // back, and it is in `held`. Nothing else may write the mode
        // (`crate::modes`).
        crate::modes::leave(state);
    }
    held
}

/// Take the suspended gesture back up, if there is one — [`suspend`]'s other
/// half, called when the trace mode ends however it ended.
///
/// Through [`update`], so the preview returns with the bar instead of the axis
/// coming back to a canvas showing nothing: what it composes against is whatever
/// ramp is now in hand. A capture is still in flight at this point (the sampling
/// is a readback), so the ramp it lands is delivered a moment later by the
/// [`refresh`] inside [`gradients::select`] — and a trace that captured nothing
/// leaves the bar exactly as it was picked up.
pub fn resume_from(state: AppState, ui: GradientUi) {
    open(state, ui);
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
                gradient: gradient.clone()?,
                axis,
            },
        ),
    })
}

/// Leave the mode: commit the composed result, or just drop the preview when
/// there is nothing to lay. The commit clears the preview itself, so there is
/// no intermediate frame showing the document without it (the transform's
/// Done). For a matte, re-selecting the layer is untouched — the frame bar
/// comes straight back. The bar's "Done", and Enter's (`crate::modes::finish`).
pub fn finish(state: AppState) {
    let ui = crate::modes::composing_now(state).and_then(Composing::gradient_fill);
    if let Some(ui) = ui {
        match compose(state, &ui) {
            Some(laid) => laid.commit(state),
            None => clear_preview(state, &ui.target),
        }
    }
    // `leave_settled`, not `leave`: the commit above supersedes the preview, and
    // dropping it again would show the document without what was just laid for a
    // frame (`crate::modes`).
    crate::modes::leave_settled(state);
}

/// The ramp the bar (and the axis chrome) is currently laying, or `None` while
/// the library has yet to supply one.
fn ramp_in_hand(state: AppState, ui: &GradientUi) -> Option<Gradient> {
    match &ui.target {
        GradientTarget::Fill { .. } => gradients::current(state),
        GradientTarget::Matte { gradient, .. } => gradient.clone(),
    }
}

/// The mode's bar, standing in for whichever bar raised it.
///
/// While a trace has the gesture parked (`suspend`), the bar stays on screen
/// **recessed** rather than vanishing — the one genuine park-and-resume in the
/// app, drawn as what it is: the place the trace hands back to
/// (MODAL_DESIGN.md). Recessed chrome is inert (`pointer-events: none`), so
/// its controls promise nothing the parked gesture cannot honour.
#[component]
pub fn GradientBar() -> Element {
    let state = use_context::<AppState>();
    let (ui, parked) =
        if let Some(ui) = crate::modes::composing(state).and_then(Composing::gradient_fill) {
            (ui, false)
        } else if let Some(ui) = state.gradient_resume.read().clone() {
            (ui, true)
        } else {
            return rsx! {};
        };
    // Read reactively so a pick in the library pop-out repaints the strip.
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

    // A drag alone is not enough to have something to lay: entered with an
    // empty library, the axis can be composed before any ramp exists to run
    // along it, and Done then leaves without laying (`compose`).
    let done_title = if ui.drag.is_some() && strip.is_some() {
        match ui.target {
            GradientTarget::Fill { .. } => "Lay the gradient (undo takes it back)",
            GradientTarget::Matte { .. } => "Keep this paint (undo takes it back)",
        }
    } else if strip.is_none() {
        "No ramp yet \u{2014} pick or trace one from the strip, or leave without laying anything"
    } else {
        "Nothing dragged yet \u{2014} leave without laying anything"
    };
    let well_title = if strip.is_some() {
        "The ramp being laid \u{2014} click to pick another or trace a new one"
    } else {
        "No ramp yet \u{2014} click to pick or trace one"
    };

    rsx! {
        div {
            // The composing register (`mode-bar`) comes off with the mode:
            // parked, this is a shelved gesture behind the trace's bar, not
            // the thing the canvas answers to.
            class: chrome_class(
                state,
                if parked {
                    "selection-bar gradient-fill-bar recessed"
                } else {
                    "selection-bar gradient-fill-bar mode-bar"
                },
            ),
            // The library's mark: the bar is that library's ramp being
            // put to work, so it wears the library's glyph.
            span { class: "bar-label",
                {icon(icons::GRADIENT)}
                {label("Gradient")}
            }
            // The ramp in hand, and the library behind it: clicking the strip
            // flies the pop-out up (§22.3).
            GradientWell { strip, title: well_title }
            span { class: "bar-sep" }
            {kind_chip(GradientAxisKind::Linear, icons::GRADIENT_LINEAR, "Linear")}
            {kind_chip(GradientAxisKind::Radial, icons::GRADIENT_RADIAL, "Radial")}
            span { class: "bar-sep" }
            // The way out that keeps nothing, worn whole off the registry with
            // its Esc advertisement (MODAL_DESIGN.md).
            CommandButton { command: Command::CancelMode }
            button {
                class: "chip",
                title: commands::advertised(done_title, Command::FinishMode, &state.bindings.read()),
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
    // The view through a memo, unconditionally and ahead of the early returns
    // below like any `use_*`. Not a straight read of the projection, which is
    // what this was: that woke the overlay on every engine write rather than on
    // the one field it draws with (`state::use_obs`).
    let live_view = use_obs(state, |o| o.view);

    let Some(ui) = crate::modes::composing(state).and_then(Composing::gradient_fill) else {
        return rsx! {};
    };
    let Some(view) = live_view() else {
        return rsx! {};
    };
    let to_canvas = move |e: &Event<PointerData>| view.screen_to_canvas(page_xy(e));

    let panning = (state.space_down)();
    let catcher_class = if panning {
        "gradient-catcher pan"
    } else {
        "gradient-catcher"
    };
    // What to do, said the way the trace says it (§22.2): the catcher is
    // invisible and the axis has no chrome until it is dragged, so before the
    // first drag the mode is a pointer that mysteriously stopped painting.
    // Down once an axis exists — from there the chrome explains itself.
    let axis_hint = match ui.kind {
        GradientAxisKind::Linear => {
            "Drag the axis: press where the ramp starts, release where it ends."
        }
        GradientAxisKind::Radial => "Drag the reach: press at the centre, release at the rim.",
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

        if ui.drag.is_none() {
            div { class: "gradient-trace-hint", {axis_hint} }
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
    view: stark_engine::ViewTransform,
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
