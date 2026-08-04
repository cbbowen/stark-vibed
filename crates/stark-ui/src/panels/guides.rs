//! Drawing guides (§20.5): the panel that keeps the list, the edit mode that
//! shapes one guide, and the bar that serves the mode.
//!
//! Three pieces, one list. The **panel** is the roster — add a perspective,
//! name one, remove one, show or hide one, pick one up to work on —
//! deliberately shaped like the Layers panel, because it answers the same
//! question about a different stack, down to the row's controls and the
//! double-click that renames.
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

use dioxus::html::Key;
use dioxus::html::input_data::MouseButton;
use dioxus::prelude::*;

use crate::icons::{self, icon, label};
use crate::input::{Nav, page_xy};
use crate::layout::chrome_class;
use crate::platform::select_all;
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

/// What to call a guide that has never been named: its place in the list.
///
/// The Layers panel's counterpart numbers by [`LayerId::ordinal`], which is stable
/// for the layer's whole life; a guide has no id, so this numbers by *position* and
/// the labels below a removed guide shift up. That is the honest reading either
/// way — an unnamed row is being described, not named, and the description of the
/// second row is "the second one". Naming it is how you stop it moving.
///
/// [`LayerId::ordinal`]: stark_core::LayerId::ordinal
fn guide_label(index: usize, guide: &PerspectiveGuide) -> String {
    match &guide.name {
        Some(name) => name.clone(),
        None => format!("Perspective {}", index + 1),
    }
}

/// The Drawing Guides panel: the roster of guides, shaped like the Layers
/// panel — a header that adds, rows that select, rename, remove and hide.
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
                {label("Perspective")}
            }
        }
        if guides.is_empty() {
            div { class: "guide-empty",
                "No guides yet. Add a perspective grid to draw through."
            }
        }
        for (i, g) in guides.into_iter().enumerate() {
            GuideRow { key: "{i}", index: i, guide: g, active: editing == Some(i) }
        }
    }
}

/// One guide in the roster. A component rather than markup inlined in the loop
/// above for the reason [`LayerRow`](super::layer::LayerRow) is one: the rename
/// field's draft is *row-local* state, so opening one leaves every other row alone
/// and closing it needs nothing cleaned up — and a hook cannot live inside a `for`.
#[component]
fn GuideRow(index: usize, guide: PerspectiveGuide, active: bool) -> Element {
    let state = use_context::<AppState>();
    // The rename in progress on this row, or `None` while the row is just a row.
    // Held here rather than read back off the field on commit because both commit
    // paths — Enter and blur — need it, and one of them fires while the field is on
    // its way out.
    let mut draft = use_signal(|| None::<String>);
    // `take` is what makes the two commit paths safe to both fire: whichever runs
    // second finds no draft. An emptied field is a *removed* name rather than a
    // blank one, which is the engine's rule for every name (`normalize_name`), so
    // the row goes back to describing its position.
    let mut commit = move || {
        let text = draft.write().take();
        if let Some(text) = text {
            update_guide(state, index, move |g| g.name = Some(text));
        }
    };
    let label = guide_label(index, &guide);
    // What the field opens on: the guide's *name*, which for one never named is
    // empty. Deliberately not the label — seeding with the generated "Perspective 2"
    // would turn opening the field and pressing Enter into a rename to "Perspective
    // 2", quietly making a description into a name, and this panel's descriptions
    // move when a guide is removed. The placeholder carries the label instead.
    let seed = guide.name.clone().unwrap_or_default();
    let visible = guide.visible;

    rsx! {
        div {
            class: if active { "guide-row active" } else { "guide-row" },
            if let Some(text) = draft() {
                input {
                    class: "guide-name",
                    class: "guide-rename",
                    r#type: "text",
                    value: "{text}",
                    placeholder: "{label}",
                    // The field is the point of the double-click, so it takes focus
                    // as it appears rather than asking for a second click.
                    onmounted: move |e: Event<MountedData>| {
                        spawn(async move {
                            let _ = e.set_focus(true).await;
                            // Selected, not merely focused: the usual reason to open
                            // the field is to replace the name rather than add to it.
                            select_all(&e);
                        });
                    },
                    oninput: move |e| draft.set(Some(e.value())),
                    // Committing on blur is what makes this feel like a label rather
                    // than a form. Enter commits directly rather than by blurring — a
                    // focused element that is removed does not reliably fire `blur`.
                    onblur: move |_| commit(),
                    onkeydown: move |e| match e.key() {
                        Key::Enter => commit(),
                        // Escape abandons the edit — dropping the draft first, so the
                        // blur that follows has nothing left to commit.
                        Key::Escape => draft.set(None),
                        _ => {}
                    },
                }
            } else {
                // The name selects, and selecting *is* picking the guide up to shape
                // it, because shaping is all there is to do to one. Double-click
                // renames, as it does on a layer row — the first click of the pair
                // landing you in the edit mode is no cost, since the guide you are
                // renaming is the one you were about to work on.
                button {
                    class: "guide-name",
                    title: "Shape this guide \u{2014} double-click to rename",
                    onclick: move |_| begin_guide_edit(state, index),
                    ondoubleclick: move |_| draft.set(Some(seed.clone())),
                    "{label}"
                }
            }
            // Remove then the eye, the order the Layers panel's rows put them in —
            // the two rosters answer the same question about different stacks, so a
            // hand that has learned one has learned the other. The ✕ this wore was
            // the one mark in either panel drawn as a character rather than a glyph;
            // it is `icons::REMOVE` in both places now.
            button {
                class: "guide-remove",
                title: "Remove this guide",
                onclick: move |_| remove_guide(state, index),
                {icon(icons::REMOVE)}
            }
            button {
                class: if visible { "guide-eye" } else { "guide-eye hidden" },
                title: if visible { "Hide this guide" } else { "Show this guide" },
                onclick: move |_| update_guide(state, index, |g| g.visible = !g.visible),
                {icon(if visible { icons::VISIBLE } else { icons::HIDDEN })}
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
    let Some(g) = guides.get(edit.index) else {
        // The guide went away under the mode (a stale index would edit the
        // wrong guide); fold the bar rather than pointing it at nothing.
        end_guide_edit(state);
        return rsx! {};
    };
    let index = edit.index;
    // The bar names the guide the same way its row does, so a renamed guide is
    // called the same thing in both places.
    let name = guide_label(index, g);
    let (density, opacity, axes) = (g.density, g.opacity, g.axes);

    rsx! {
        div { class: chrome_class(state, "guide-bar"),
            // The Guides panel's own mark, on the bar its rows raise — and the reason
            // the words here can be the guide's *name*: the glyph says what kind of
            // thing is being shaped, so the text is free to say which one.
            //
            // Hideable, even though a named guide's name is the artist's own. The rule
            // that protects names is about the places a name is *kept* — the roster in
            // the panel, which is where a guide is found, chosen and renamed, and which
            // keeps its text. This bar is not that; it is the mode indicator for the one
            // guide already in hand, and every other bottom bar's label goes. Leaving
            // this one standing would make the guide bar the odd bar out for a word that
            // is legible one panel away.
            span { class: "bar-label",
                {icon(icons::PERSPECTIVE_GRID)}
                {name}
            }
            // Locks: hold a world axis fixed, constraining the canvas drag to
            // turns about it — lock the vertical and every gesture keeps the
            // verticals parallel. Colored as the axis's own lines are.
            //
            // All four of the bar's group labels wear a mark, and it is the *label*
            // that is the optional half rather than the glyph. That is not a rule
            // about this bar: a mark is what a control still has when its word is
            // taken away, so anything that can be reached for has to carry one, and a
            // label with nothing but a word is a control that would vanish.
            span { class: "bar-sub",
                {icon(icons::LOCK)}
                {label("Lock")}
            }
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
            // The same eye the guide's own row wears, asked of a fan of guide lines
            // rather than of the whole guide.
            span { class: "bar-sub",
                {icon(icons::VISIBLE)}
                {label("Show")}
            }
            for i in 0..3 {
                button {
                    class: if axes[i] { "chip axis-chip active" } else { "chip axis-chip" },
                    style: "--axis: {AXIS_CSS[i]}",
                    title: "Show the {AXIS_NAMES[i]} axis's fan of guide lines",
                    onclick: move |_| update_guide(state, index, move |g| g.axes[i] = !g.axes[i]),
                    "{AXIS_NAMES[i]}"
                }
            }
            span { class: "bar-sep" }
            // A fan of lines from a point, which is what this number counts: the
            // guide's fans are its parametrization (§20.5), so the mark is a picture
            // of the thing the slider makes more or fewer of.
            span { class: "bar-sub",
                {icon(icons::DENSITY)}
                {label("Density")}
            }
            input {
                class: "slider",
                r#type: "range", min: "4", max: "36", step: "1",
                value: "{density}",
                title: "Guide lines per half turn",
                oninput: move |e| {
                    if let Ok(v) = e.value().parse::<f32>() {
                        update_guide(state, index, move |g| g.density = v.round() as u32);
                    }
                },
            }
            // The ghost the Layers panel and the brush editor wear: how much of what
            // is under this shows through, asked of a guide over the paint.
            span { class: "bar-sub",
                {icon(icons::OPACITY)}
                {label("Opacity")}
            }
            input {
                class: "slider",
                r#type: "range", min: "0.1", max: "1", step: "any",
                value: "{opacity}",
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
                {icon(icons::DONE)}
                {label("Done")}
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
#[derive(Clone)]
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
            Some(g) => (o.view, g.clone()),
            None => return rsx! {},
        },
        None => return rsx! {},
    };
    let index = edit.index;
    let locked = edit.locked;
    // The two camera numbers the hit test needs, read out here. Taken as values
    // rather than off `guide` so the closures below capture nothing that is not
    // `Copy` — a guide carries a name now, and a handler that captured the whole
    // guide could not be shared between the move and the release.
    let (center, focal) = (guide.center, guide.focal);

    let to_canvas = move |e: &Event<PointerData>| view.screen_to_canvas(page_xy(e));
    let classify = move |pc: Vec2| {
        let on_screen = view.canvas_to_screen(center);
        if (on_screen - view.canvas_to_screen(pc)).length() < CENTER_GRAB_PX {
            GuideRegion::Center
        } else if ((pc - center).length() - focal).abs() * view.zoom < CIRCLE_BAND_PX {
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
            hover.set(Some(classify(pc)));
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
            // The turn only, rather than the whole guide the drag was started
            // from: a drag is a statement about the camera's orientation, and
            // writing back a snapshot would also write back the name, opacity and
            // density as they stood at the press. Assigning the one field the drag
            // computes leaves nothing for a mid-drag edit elsewhere to lose.
            GuideRegion::Orbit => update_guide(state, index, move |g| {
                g.rotation = d.start.dragged(d.from, pc, locked).rotation;
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
                    region: classify(pc),
                    from: pc,
                    start: guide.clone(),
                }));
            },
            onpointermove: move |e| follow(&e),
            onpointerup: move |e| if !nav.release(&e) { finish(&e) },
            onpointercancel: move |e| if !nav.release(&e) { nav.stop(); drag.set(None); },
            onwheel: move |e| nav.wheel(e),
        }
    }
}
