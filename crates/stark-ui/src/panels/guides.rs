//! Drawing guides (§20.5): the panel that keeps the list, the edit mode that
//! shapes one guide, and the bar that serves the mode.
//!
//! Three pieces, one list. The **panel** is the roster — add a perspective,
//! show or hide one, pick one up to work on — deliberately shaped like the
//! Layers panel, because it answers the same question about a different stack.
//! Selecting a row (or adding a guide) enters the **edit mode**: a
//! full-viewport catcher owns the pointer, exactly as transform mode does
//! (§16.6), and dragging on the canvas *is* the manipulation:
//!
//! - **anywhere** — grab the world: the direction under the pointer follows
//!   it, and the rotation snaps to a pure turn about a world axis whenever the
//!   drag nearly is one ([`PerspectiveGuide::dragged`], §20.5);
//! - **the 45° circle** — drag the lens: the circle's radius *is* the focal
//!   length, so it follows the hand exactly;
//! - **the center-of-view crosshair** — move the whole construction.
//!
//! The **Perspective Guide bar** stands at the bottom for the mode's duration:
//! per-axis locks (constraining the drag — lock the vertical and 2-point
//! stays 2-point under any gesture), per-axis visibility, density, opacity,
//! and "Done". There is no case switch anywhere: which of 1/2/3-point you are
//! in is something the canvas *shows* (the count of finite vanishing points),
//! not something a control stores.

use dioxus::html::input_data::MouseButton;
use dioxus::prelude::*;

use crate::icons::{self, icon};
use crate::input::{Nav, page_xy};
use crate::layout::chrome_class;
use crate::state::{AppState, GuideEdit, dispatch};
use stark_core::PerspectiveGuide;
use stark_core::command::ViewCommand;
use stark_core::geom::Vec2;

/// The axis hues, as CSS — the same values `guides.wesl` draws the fans in,
/// so a lock chip and the lines it governs read as one thing.
const AXIS_CSS: [&str; 3] = ["#e8575c", "#54b566", "#618cf0"];
const AXIS_NAMES: [&str; 3] = ["X", "Y", "Z"];

/// Grab radius of the center-of-view crosshair, screen px.
const CENTER_GRAB_PX: f32 = 14.0;
/// Half-width of the 45° circle's grab band, screen px — converted by the
/// zoom, so the ring is equally grabbable at any magnification.
const CIRCLE_BAND_PX: f32 = 10.0;
/// The lens's travel, canvas px: wide enough for any drawing, floored so the
/// circle cannot be dragged through its own center into a degenerate camera.
const FOCAL_RANGE: (f32, f32) = (120.0, 12000.0);

/// The engine's guide list, cloned for a read-modify-commit.
fn guides_of(state: AppState) -> Vec<PerspectiveGuide> {
    state
        .obs
        .read()
        .as_ref()
        .map(|o| o.guides.clone())
        .unwrap_or_default()
}

/// Adjust one guide and push the whole list back — every mutation funnels
/// through here, the same shape as `update_media` (§4).
fn update_guide(state: AppState, index: usize, f: impl FnOnce(&mut PerspectiveGuide)) {
    let mut guides = guides_of(state);
    let Some(g) = guides.get_mut(index) else {
        return;
    };
    f(g);
    dispatch(state, ViewCommand::SetGuides(guides));
}

/// Enter the edit mode on guide `index` — fresh locks each time: a lock is a
/// constraint on the hand for one sitting, not a fact about the guide.
pub fn begin_guide_edit(state: AppState, index: usize) {
    let mut mode = state.guide_edit;
    mode.set(Some(GuideEdit {
        index,
        locked: [false; 3],
    }));
}

/// Leave the edit mode.
fn end_guide_edit(state: AppState) {
    let mut mode = state.guide_edit;
    mode.set(None);
}

/// Add a perspective guide where the artist is looking, and pick it up: adding
/// *is* asking to shape it, so the mode opens on the new row.
fn add_perspective(state: AppState) {
    let center = state
        .obs
        .read()
        .as_ref()
        .map(|o| o.view.center)
        .unwrap_or(Vec2::ZERO);
    let mut guides = guides_of(state);
    guides.push(PerspectiveGuide {
        center,
        ..Default::default()
    });
    let index = guides.len() - 1;
    dispatch(state, ViewCommand::SetGuides(guides));
    begin_guide_edit(state, index);
}

/// Remove a guide, keeping the edit mode pointed at the row it was on: the
/// indices above the removed one all shift down, and the mode follows —
/// unless it was the removed guide itself, in which case it ends.
fn remove_guide(state: AppState, index: usize) {
    let mut guides = guides_of(state);
    if index >= guides.len() {
        return;
    }
    guides.remove(index);
    dispatch(state, ViewCommand::SetGuides(guides));
    let mut mode = state.guide_edit;
    let current = *mode.peek();
    let adjusted = current.and_then(|e| match e.index {
        i if i == index => None,
        i if i > index => Some(GuideEdit { index: i - 1, ..e }),
        _ => Some(e),
    });
    mode.set(adjusted);
}

/// The Drawing Guides panel: the roster of guides, shaped like the Layers
/// panel — a header that adds, rows that select, an eye per row.
#[component]
pub fn GuidesPanel() -> Element {
    let state = use_context::<AppState>();
    let guides = guides_of(state);
    let editing = (*state.guide_edit.read()).map(|e| e.index);

    rsx! {
        div { class: "layer-header",
            button {
                class: "layer-add",
                title: "Add a perspective grid where you are looking",
                onclick: move |_| add_perspective(state),
                {icon(icons::ADD_LAYER)}
                "Perspective"
            }
        }
        if guides.is_empty() {
            div { class: "guide-empty",
                "No guides yet. Add a perspective grid to draw through."
            }
        }
        for (i, g) in guides.into_iter().enumerate() {
            div {
                key: "{i}",
                class: if editing == Some(i) { "guide-row active" } else { "guide-row" },
                // The name selects: picking a guide up to look at it and
                // picking it up to shape it are the same act here, because
                // shaping is all there is to do to one.
                span {
                    class: "guide-name",
                    onclick: move |_| begin_guide_edit(state, i),
                    "Perspective {i + 1}"
                }
                // Remove then the eye, the order the Layers panel's rows put them in
                // — the two rosters answer the same question about different stacks,
                // so a hand that has learned one has learned the other. The ✕ this
                // wore was the one mark in either panel drawn as a character rather
                // than a glyph; it is `icons::REMOVE` in both places now.
                button {
                    class: "guide-remove",
                    title: "Remove this guide",
                    onclick: move |_| remove_guide(state, i),
                    {icon(icons::REMOVE)}
                }
                button {
                    class: if g.visible { "guide-eye" } else { "guide-eye hidden" },
                    title: if g.visible { "Hide this guide" } else { "Show this guide" },
                    onclick: move |_| update_guide(state, i, |g| g.visible = !g.visible),
                    {icon(if g.visible { icons::VISIBLE } else { icons::HIDDEN })}
                }
            }
        }
    }
}

/// The Perspective Guide bar (§20.5): the edit mode's controls, in the same
/// bottom column as the selection and transform bars. Mounted only while a
/// guide is being composed.
#[component]
pub fn PerspectiveGuideBar() -> Element {
    let state = use_context::<AppState>();
    let Some(edit) = *state.guide_edit.read() else {
        return rsx! {};
    };
    let guides = guides_of(state);
    let Some(g) = guides.get(edit.index).copied() else {
        // The guide went away under the mode (a stale index would edit the
        // wrong guide); fold the bar rather than pointing it at nothing.
        end_guide_edit(state);
        return rsx! {};
    };
    let index = edit.index;

    rsx! {
        div { class: chrome_class(state, "guide-bar"),
            span { class: "bar-label", "Perspective {index + 1}" }
            // Locks: hold a world axis fixed, constraining the canvas drag to
            // turns about it — lock the vertical and every gesture keeps the
            // verticals parallel. Colored as the axis's own lines are.
            span { class: "bar-sub", "Lock" }
            for i in 0..3 {
                button {
                    class: if edit.locked[i] { "chip axis-chip active" } else { "chip axis-chip" },
                    style: "--axis: {AXIS_CSS[i]}",
                    title: "Hold the {AXIS_NAMES[i]} axis fixed under the drag",
                    onclick: move |_| {
                        let mut mode = state.guide_edit;
                        if let Some(e) = mode.write().as_mut() {
                            e.locked[i] = !e.locked[i];
                        }
                    },
                    "{AXIS_NAMES[i]}"
                }
            }
            span { class: "bar-sep" }
            span { class: "bar-sub", "Show" }
            for i in 0..3 {
                button {
                    class: if g.axes[i] { "chip axis-chip active" } else { "chip axis-chip" },
                    style: "--axis: {AXIS_CSS[i]}",
                    title: "Show the {AXIS_NAMES[i]} axis's fan of guide lines",
                    onclick: move |_| update_guide(state, index, move |g| g.axes[i] = !g.axes[i]),
                    "{AXIS_NAMES[i]}"
                }
            }
            span { class: "bar-sep" }
            span { class: "bar-sub", "Density" }
            input {
                class: "slider",
                r#type: "range", min: "4", max: "36", step: "1",
                value: "{g.density}",
                title: "Guide lines per half turn",
                oninput: move |e| {
                    if let Ok(v) = e.value().parse::<f32>() {
                        update_guide(state, index, move |g| g.density = v.round() as u32);
                    }
                },
            }
            span { class: "bar-sub", "Opacity" }
            input {
                class: "slider",
                r#type: "range", min: "0.1", max: "1", step: "any",
                value: "{g.opacity}",
                title: "How strongly the guide reads over the paint",
                oninput: move |e| {
                    if let Ok(v) = e.value().parse::<f32>() {
                        update_guide(state, index, move |g| g.opacity = v);
                    }
                },
            }
            span { class: "bar-sep" }
            button {
                class: "chip",
                title: "Leave the guide as it stands",
                onclick: move |_| end_guide_edit(state),
                "Done"
            }
        }
    }
}

/// What a press at a point would grab, tried nearest-first: the crosshair
/// moves the construction, the ring is the lens, and everywhere else is the
/// world.
#[derive(Copy, Clone, PartialEq)]
enum GuideRegion {
    Center,
    Focal,
    Orbit,
}

/// An in-flight guide drag: what it grabbed, where it started in canvas px,
/// and the guide as it was then. Recomputed from the start on every move —
/// the same discipline as the transform drag (§16.6), and here it is also
/// what keeps the axis *snap* stable: the snap classifies the whole drag from
/// its origin, so it cannot flicker between axes mid-gesture.
#[derive(Clone, Copy)]
struct Drag {
    region: GuideRegion,
    from: Vec2,
    start: PerspectiveGuide,
}

/// The edit mode's catcher: a full-viewport surface that owns every pointer
/// event while a guide is being composed, so a stray drag cannot paint — but
/// navigation still works (middle-drag and space-drag pan, the wheel zooms;
/// see `input::Nav`). All gesture math is in canvas space, so panning or
/// zooming mid-drag cannot corrupt it.
#[component]
pub fn GuideEditOverlay() -> Element {
    let state = use_context::<AppState>();
    let mut drag = use_signal(|| None::<Drag>);
    let mut hover = use_signal(|| None::<GuideRegion>);
    let nav = Nav::use_nav(state);

    let Some(edit) = *state.guide_edit.read() else {
        return rsx! {};
    };
    let (view, guide) = match state.obs.read().as_ref() {
        Some(o) => match o.guides.get(edit.index) {
            Some(g) => (o.view, *g),
            None => return rsx! {},
        },
        None => return rsx! {},
    };
    let index = edit.index;
    let locked = edit.locked;

    let to_canvas = move |e: &Event<PointerData>| view.screen_to_canvas(page_xy(e));
    let classify = move |g: &PerspectiveGuide, pc: Vec2| {
        let on_screen = view.canvas_to_screen(g.center);
        if (on_screen - view.canvas_to_screen(pc)).length() < CENTER_GRAB_PX {
            GuideRegion::Center
        } else if ((pc - g.center).length() - g.focal).abs() * view.zoom < CIRCLE_BAND_PX {
            GuideRegion::Focal
        } else {
            GuideRegion::Orbit
        }
    };

    let mut follow = move |e: &Event<PointerData>| {
        if nav.advance(e) {
            return;
        }
        let pc = to_canvas(e);
        let Some(d) = drag() else {
            hover.set(Some(classify(&guide, pc)));
            return;
        };
        match d.region {
            GuideRegion::Center => update_guide(state, index, move |g| {
                g.center = d.start.center + (pc - d.from);
            }),
            GuideRegion::Focal => update_guide(state, index, move |g| {
                g.focal = (pc - d.start.center)
                    .length()
                    .clamp(FOCAL_RANGE.0, FOCAL_RANGE.1);
            }),
            GuideRegion::Orbit => update_guide(state, index, move |g| {
                *g = d.start.dragged(d.from, pc, locked);
            }),
        }
    };
    let mut finish = move |e: &Event<PointerData>| {
        follow(e);
        nav.stop();
        drag.set(None);
    };

    let panning = (state.space_down)();
    let catcher_class = if panning {
        "guide-catcher pan"
    } else {
        "guide-catcher"
    };
    let cursor = match (panning, drag(), hover()) {
        (true, ..) => "",
        (_, Some(d), _) => match d.region {
            GuideRegion::Center => "cursor: move;",
            _ => "cursor: grabbing;",
        },
        (_, None, Some(GuideRegion::Center)) => "cursor: move;",
        (_, None, Some(GuideRegion::Focal)) => "cursor: grab;",
        (_, None, Some(GuideRegion::Orbit)) => "cursor: crosshair;",
        (_, None, None) => "",
    };

    rsx! {
        div {
            class: "{catcher_class}",
            style: "{cursor}",
            onpointerdown: move |e| {
                if nav.begin(&e) {
                    // A second finger turns the drag into navigation
                    // (§18.1.7). The guide keeps whatever the drag had done —
                    // there is nothing to commit, view state is already live.
                    drag.set(None);
                    return;
                }
                if e.trigger_button() != Some(MouseButton::Primary) {
                    return;
                }
                e.stop_propagation();
                crate::platform::capture_pointer(&e);
                let pc = to_canvas(&e);
                drag.set(Some(Drag {
                    region: classify(&guide, pc),
                    from: pc,
                    start: guide,
                }));
            },
            onpointermove: move |e| follow(&e),
            onpointerup: move |e| if !nav.release(&e) { finish(&e) },
            onpointercancel: move |e| if !nav.release(&e) { nav.stop(); drag.set(None); },
            onwheel: move |e| nav.wheel(e),
        }
    }
}
