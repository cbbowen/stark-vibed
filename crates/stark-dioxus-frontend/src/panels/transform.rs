//! The transform mode's chrome (§16.6, §16.8, §16.9): one bar, three
//! families, each with a control surface whose **shape is the deformation**:
//!
//! - **Free** (the affine) — an ellipse, not a box of handles: the image of a
//!   reference circle under the accumulated linear map. Drag inside to
//!   translate, on the rim to rotate/scale, outside to stretch/shear — the
//!   grabbed point follows the pointer exactly in every family.
//! - **Perspective** — the image of the source rect is a quad with corner
//!   handles; the map is *defined* as "the homography putting the corners
//!   where the hand put them", so the widget cannot disagree with the paint.
//!   Edges shift whole sides (the foreshortening gesture); inside translates.
//!   The receding grid drawn through the quad is the transformed space itself.
//! - **Warp** — a 4×4 control mesh whose drawn curves are sampled from the
//!   same smooth surface the paint resamples through. Drag a point, or grab
//!   the surface anywhere: the least-norm control move keeps the grabbed
//!   *paint* under the pointer — the hand holds the painting, not a handle.
//!
//! Which polylines those are is `stark_ui::transform::{outline, grid, handles}`,
//! in canvas px; what is left here is the SVG. The affine keeps its own drawing,
//! because a CSS `matrix()` on a round div is a different *technique* rather than a
//! second derivation of the same geometry.
//!
//! The bar selects the family. A switch **carries** the deformation when the
//! new family contains the old one exactly (affine ⊂ perspective, and the
//! smooth mesh reproduces any affine); otherwise it **commits** first — one
//! honest undo step — and reopens fresh around the moved paint, because a
//! lossy carry would silently change what "Done" produces.
//!
//! Everything before "Done" is a **lossless preview**: every change previews
//! through `preview::TRANSFORM` — the same renderer the commit
//! runs, over the committed tiles, so what is on screen is exactly what "Done"
//! will produce. "Done" commits a single `DocCommand::Transform`: one undo
//! step per gesture, like the frame drag (§15.7).
//!
//! While the mode is active the full-viewport catcher owns the pointer, so a
//! stray drag cannot paint — but navigation still works: middle-drag and
//! space-drag pan, the wheel zooms (see `input::Nav`). All gesture maths is in
//! canvas space, so panning or zooming mid-gesture cannot corrupt it.

use dioxus::html::input_data::MouseButton;
use dioxus::prelude::*;

use crate::icons::{icon, label};
use crate::input::{Nav, page_xy};
use crate::layout::chrome_dimmed;
use crate::modes::Composing;
use crate::preview;
use crate::state::{AppState, use_obs};
use crate::widgets::CommandButton;
use stark_engine::ViewTransform;
use stark_model::geom::Vec2;
use stark_ui::commands::Command;
use stark_ui::transform::{Bands, Family, Grab, Hint, Switch, TransformState, TransformUi};

/// Enter transform mode around the current selection, in the Free (affine)
/// family.
///
/// The widget mounts around the selection's analytic hull; an unbounded
/// selection (select-all, an inversion) falls back to the painted content's
/// bounds — the whole layer is what an unbounded selection holds — and an
/// empty canvas to the view, so the widget always exists. The target layer is
/// the active one, or the topmost paintable layer when a matte is selected (a
/// matte refuses transforms the same way it refuses strokes).
pub fn begin_transform(state: AppState) {
    // One composing mode at a time (`crate::modes`): whatever was in hand is put
    // down before this takes the canvas, so the case of two catchers over one
    // pointer never arises.
    crate::modes::leave(state);
    let obs = state.obs.peek();
    let Some(o) = obs.as_ref() else { return };
    // Which layer and which rectangle are both `stark_ui::transform`'s answers,
    // and they come off one read: a hull from before a layer change with a layer
    // from after would mount the widget around paint it is not holding.
    let Some(entry) = stark_ui::transform::entry(o) else {
        return;
    };
    let zoom = o.view.zoom;
    drop(obs);
    // `enter`, not a write: it is what puts down whatever else was composing,
    // and the only way in (`crate::modes`).
    crate::modes::enter(
        state,
        Composing::Transform(stark_ui::transform::mount(
            entry.layer,
            Family::Free,
            entry.hull,
            Bands::at(zoom),
        )),
    );
}

/// Update the gesture and show its consequence — every mutation funnels through
/// here, so the preview can never lag the state.
fn update(state: AppState, ui: TransformUi) {
    // `advance`, which replaces what the mode in hand is composing and refuses
    // to change *which* mode that is (`crate::modes`).
    crate::modes::advance(state, Composing::Transform(ui));
    preview::TRANSFORM.show(state, (ui.layer(), ui.map()));
}

/// Switch the composing family — the bar's half of a decision
/// `stark_ui::transform::switch` makes: carry the deformation, or commit it
/// first (one honest undo step) and reopen around the moved paint.
fn switch_family(state: AppState, ui: TransformUi, to: Family) {
    let zoom = state
        .obs
        .peek()
        .as_ref()
        .map(|o| o.view.zoom)
        .unwrap_or(1.0);
    match stark_ui::transform::switch(ui, to, Bands::at(zoom)) {
        Switch::Nothing => {}
        // `update`: the carried map is the new family's own, exact to within a
        // resample, and the preview owes that rather than the one it replaced.
        Switch::Carried(next) => update(state, next),
        // `advance`, not `update`: there is no deformation to show, and previewing
        // the identity would resample the selected paint for nothing.
        Switch::Fresh(next) => crate::modes::advance(state, Composing::Transform(next)),
        Switch::Commit { map, then } => {
            preview::TRANSFORM.commit(state, (ui.layer(), map));
            crate::modes::advance(state, Composing::Transform(then));
        }
    }
}

/// Commit the gesture and leave the mode — the bar's "Done", and Enter's
/// (`crate::modes::finish`). An identity transform just drops the preview
/// rather than spending an undo step on a no-op.
pub fn finish(state: AppState) {
    let Some(ui) = crate::modes::composing_now(state).and_then(Composing::transform) else {
        return;
    };
    if ui.is_identity() {
        preview::TRANSFORM.clear(state);
    } else {
        // The commit clears the preview itself, so there is no intermediate
        // frame showing the untransformed document.
        preview::TRANSFORM.commit(state, (ui.layer(), ui.map()));
    }
    // `leave_settled`, not `leave`: both arms above have already dealt with the
    // preview — one dropped it, the other superseded it with a commit — and
    // dropping it again after the commit would show the untransformed document
    // for a frame (`crate::modes`).
    crate::modes::leave_settled(state);
}

/// The transform bar: the family selector, the affine's two flips, Cancel and
/// "Done". Mounted only while the gesture is in flight, in the same bottom
/// column as the selection and frame bars — wearing the composing register
/// (`mode-bar`) those two do not, because this bar fronts a catcher that has
/// taken the pointer away from painting (MODAL_DESIGN.md).
#[component]
pub fn TransformBar() -> Element {
    let state = use_context::<AppState>();
    let Some(ui) = crate::modes::composing(state).and_then(Composing::transform) else {
        return rsx! {};
    };
    let family = ui.family();
    let chip = |on: bool| if on { "chip active" } else { "chip" };

    rsx! {
        div {
            class: "transform-bar mode-bar chrome",
            class: if chrome_dimmed(state) { "dimmed" },
            // The same mark the selection bar's Transform chip wears — this bar stands
            // in for that one for the gesture's duration, so it carries the glyph of
            // the button that raised it.
            span { class: "bar-label",
                {icon(stark_ui::icons::TRANSFORM)}
                {label("Transform")}
            }

            span { class: "bar-sep" }

            // The three families. Switching carries the deformation when the new
            // family holds it exactly (free → perspective, free → warp), and
            // commits it first when it cannot — never a silent approximation.
            button {
                class: chip(family == Family::Free),
                title: "Move, scale, rotate, shear — the ellipse widget",
                onclick: move |_| switch_family(state, ui, Family::Free),
                {icon(stark_ui::icons::TRANSFORM)}
                {label("Free")}
            }
            button {
                class: chip(family == Family::Perspective),
                title: "Drag the corners into a perspective (§16.8)",
                onclick: move |_| switch_family(state, ui, Family::Perspective),
                {icon(stark_ui::icons::PERSPECTIVE)}
                {label("Perspective")}
            }
            button {
                class: chip(family == Family::Warp),
                title: "Bend the paint through a mesh (§16.9)",
                onclick: move |_| switch_family(state, ui, Family::Warp),
                {icon(stark_ui::icons::WARP)}
                {label("Warp")}
            }
            if family == Family::Free {
                span { class: "bar-sep" }
                // The axis was already the only thing distinguishing these two buttons,
                // and it is carried by the glyph: a picture of the mirroring itself.
                button {
                    class: "chip",
                    title: "Mirror left \u{2194} right",
                    onclick: move |_| {
                        if let TransformUi::Affine { rect, ts } = ui {
                            update(state, TransformUi::Affine { rect, ts: ts.flipped_h() });
                        }
                    },
                    {icon(stark_ui::icons::FLIP_H)}
                    {label("Flip")}
                }
                button {
                    class: "chip",
                    title: "Mirror top \u{2195} bottom",
                    onclick: move |_| {
                        if let TransformUi::Affine { rect, ts } = ui {
                            update(state, TransformUi::Affine { rect, ts: ts.flipped_v() });
                        }
                    },
                    {icon(stark_ui::icons::FLIP_V)}
                    {label("Flip")}
                }
            }
            span { class: "bar-sep" }
            // The way out that keeps nothing — the act `modes::leave` performs
            // for every other entry point, finally offered as itself. Worn
            // whole off the registry, Esc advertisement included.
            CommandButton { command: Command::CancelMode }
            button {
                class: "chip",
                title: stark_ui::commands::advertised(
                    "Apply the transform \u{2014} one undo step",
                    Command::FinishMode,
                    &state.bindings.read(),
                ),
                onclick: move |_| finish(state),
                {icon(stark_ui::icons::DONE)}
                {label("Done")}
            }
        }
    }
}

/// The transform widget: a full-viewport catcher that owns every pointer event
/// and classifies it against the current family's control surface (the maths
/// lives in `stark_ui::transform`), plus the purely visual chrome — the affine's
/// ellipse, or an SVG of the quad/mesh whose lines are the deformation itself. No
/// per-handle DOM: the whole viewport is the control surface.
#[component]
pub fn TransformOverlay() -> Element {
    let state = use_context::<AppState>();
    let mut drag = use_signal(|| None::<Grab>);
    // The cursor the resting pointer has earned, for feedback only.
    let mut hover = use_signal(|| "");
    // The canvas's own navigation bindings, live on the catcher: composing a
    // transform must not cost the view (see `input::Nav`).
    let nav = Nav::use_nav(state);
    // The view through a memo, unconditionally and ahead of the early returns
    // below like any `use_*`. Not a straight read of the projection, which is
    // what this was: that woke the overlay on every engine write rather than on
    // the one field it draws with (`state::use_obs`).
    let live_view = use_obs(state, |o| o.view);

    let Some(ui) = crate::modes::composing(state).and_then(Composing::transform) else {
        return rsx! {};
    };
    let Some(view) = live_view() else {
        return rsx! {};
    };

    let to_canvas = move |e: &Event<PointerData>| view.screen_to_canvas(page_xy(e));
    // The grab widths in canvas px — one value, and the division by the zoom is
    // inside it (`stark_ui::transform::Bands`).
    let bands = Bands::at(view.zoom);

    // How this frontend spells the cursor a resting pointer has earned. The
    // classification is `stark_ui::transform`'s — the CSS is a spelling of it, and a
    // spelling is all a frontend owes (§11.2).
    let cursor_at = move |pc: Vec2| -> &'static str {
        match stark_ui::transform::hint_at(&ui, pc, bands) {
            Hint::Move => "cursor: move;",
            Hint::Hold => "cursor: grab;",
            Hint::Shape => "cursor: crosshair;",
        }
    };

    let mut follow = move |e: &Event<PointerData>| {
        if nav.advance(e) {
            return;
        }
        let pc = to_canvas(e);
        // Resting: report what a press here would do, for the cursor. Asked with a
        // `peek` and answered before the write below, because a `with_mut` dirties
        // the signal whether or not it changed anything — which on a hovering
        // pointer is a re-render of the overlay per move, for nothing.
        if drag.peek().is_none() {
            hover.set(cursor_at(pc));
            return;
        }
        // The grab carries the drag's start *and* the last shape the family could
        // express, so this is the whole of a move: no second read of live state, and
        // nothing that could hand back a shape from another family.
        if let Some(next) = drag.with_mut(|d| d.as_mut().map(|g| g.follow(pc, bands))) {
            update(state, next);
        }
    };
    let mut finish = move |e: &Event<PointerData>| {
        follow(e);
        nav.stop();
        drag.set(None);
    };

    // The cursor announces the region the press would grab; the pan class wins
    // while space is held (an inline cursor would override it, so none is set).
    let panning = (state.space_down)();
    let catcher_class = if panning {
        "transform-catcher pan"
    } else {
        "transform-catcher"
    };
    let cursor = match (panning, drag()) {
        (true, _) => "",
        (_, Some(_)) => "cursor: grabbing;",
        (_, None) => hover(),
    };

    rsx! {
        div {
            class: "{catcher_class}",
            style: "{cursor}",
            onpointerdown: move |e| {
                if nav.begin(&e) {
                    // A second finger turns the drag into navigation (§18.1.7).
                    // The preview it had built stands — a transform commits on
                    // "Done", not on release, so letting go to look around costs
                    // nothing.
                    drag.set(None);
                    return;
                }
                if e.trigger_button() != Some(MouseButton::Primary) {
                    return;
                }
                e.stop_propagation();
                crate::platform::capture_pointer(&e);
                let pc = to_canvas(&e);
                drag.set(Some(Grab::take(ui, pc, bands)));
            },
            onpointermove: move |e| follow(&e),
            // Fingers still on the glass mean the gesture is not over — see the
            // canvas's own release handler.
            onpointerup: move |e| if !nav.release(&e) { finish(&e) },
            onpointercancel: move |e| if !nav.release(&e) { nav.stop(); drag.set(None); },
            onwheel: move |e| nav.wheel(e),
        }

        {match ui {
            TransformUi::Affine { ts, .. } => affine_ellipse(state, ts, view),
            TransformUi::Perspective(_) | TransformUi::Warp(_) => shape_overlay(state, ui, view),
        }}
    }
}

/// The affine family's widget: a circle of the reference radius, deformed by
/// the linear map via CSS about its centre — the same composition the affine
/// applies to the paint, so the widget and the preview cannot disagree. It
/// stays a circle exactly as long as the transform is a similarity;
/// eccentricity *is* the distortion.
///
/// A plain function rather than a `#[component]`, for [`shape_overlay`]'s reason.
fn affine_ellipse(state: AppState, ts: TransformState, view: ViewTransform) -> Element {
    let cs = view.canvas_to_screen(ts.center);
    let r = ts.radius * view.zoom;
    // The gesture's linear map is canvas-space; what CSS draws is on screen, so the
    // view's own orientation composes onto it. The zoom is already in `r`, and the
    // orientation carries no scale, so this is the whole difference — a turned or
    // mirrored canvas turns and mirrors the widget with the paint it stands for.
    let l = view.orientation() * ts.linear;
    let ellipse_style = format!(
        "left: {}px; top: {}px; width: {}px; height: {}px; \
         transform: matrix({}, {}, {}, {}, 0, 0);",
        cs.x - r,
        cs.y - r,
        2.0 * r,
        2.0 * r,
        l.x_axis.x,
        l.x_axis.y,
        l.y_axis.x,
        l.y_axis.y,
    );
    rsx! {
        div {
            class: "transform-ellipse chrome",
            class: if chrome_dimmed(state) { "dimmed" },
            style: "{ellipse_style}",
            // The north dot rides the same CSS transform as its parent, so it
            // marks the reference ellipse's "up" wherever the deformation has
            // carried it — without it a rotated circle looks unrotated.
            div { class: "transform-north" }
        }
    }
}

/// One SVG polyline through screen-space points.
fn polyline(points: impl Iterator<Item = Vec2>) -> String {
    let mut d = String::new();
    for (i, p) in points.enumerate() {
        d.push(if i == 0 { 'M' } else { 'L' });
        d.push_str(&format!("{:.2} {:.2} ", p.x, p.y));
    }
    d
}

/// The two rect-scoped families' widget: the outline, the lines drawn through it,
/// and the handles — all three straight off `stark_ui::transform`, in canvas px.
///
/// One function for both because the only difference is a word: a perspective's
/// interior lines are context (`transform-grid`) and a warp's are draggable paint
/// (`transform-mesh`), which the stylesheet says at two weights. Everything else —
/// which runs there are, how finely they are sampled, where the handles sit — is the
/// same question, and it was answered here twice before it was answered once (§16.9).
///
/// A plain function rather than a `#[component]`: it is chosen by a `match` (a
/// component would be conditionally-mounted hooks) and its inputs are the parent's
/// already-read state, not props to diff.
fn shape_overlay(state: AppState, ui: TransformUi, view: ViewTransform) -> Element {
    let to_screen = move |c: Vec2| view.canvas_to_screen(c);
    let (w, h) = (view.viewport.width, view.viewport.height);
    let path = |runs: Vec<Vec<Vec2>>| -> String {
        runs.into_iter()
            .map(|run| polyline(run.into_iter().map(to_screen)))
            .collect()
    };
    let grid = path(stark_ui::transform::grid(&ui));
    let outline = path(stark_ui::transform::outline(&ui));
    let grid_class = if ui.family() == Family::Warp {
        "transform-mesh"
    } else {
        "transform-grid"
    };

    rsx! {
        svg {
            class: "transform-svg chrome",
            class: if chrome_dimmed(state) { "dimmed" },
            width: "{w}",
            height: "{h}",
            view_box: "0 0 {w} {h}",
            path { class: "{grid_class}", d: "{grid}" }
            path { class: "transform-outline", d: "{outline}" }
            for c in stark_ui::transform::handles(&ui) {
                circle {
                    class: "transform-handle",
                    cx: "{to_screen(c).x}",
                    cy: "{to_screen(c).y}",
                    r: "5",
                }
            }
        }
    }
}
