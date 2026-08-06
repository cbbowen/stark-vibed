//! Drawing guides (§20.5): the panel that keeps the list, the edit mode that
//! shapes one guide, and the bar that serves the mode.
//!
//! Three pieces, one list. The **panel** is the roster — add a perspective,
//! name one, remove one, show or hide one, reorder by dragging, pick one up to
//! work on — deliberately shaped like the Layers panel, because it answers the
//! same question about a different stack, down to the row's controls, the
//! double-click that renames, and the drag that moves a row (`panels::reorder`,
//! §14.6). What that drag *means* is all that differs: a guide list is flat, so
//! a landing is an index and nothing sideways is asked of the hand.
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
use crate::panels::reorder::{self, Grab, Motion, Slide};
use crate::platform::{capture_pointer, guide_boxes, select_all};
use crate::state::{AppState, GuideEdit, dispatch};
use stark_core::command::ViewCommand;
use stark_core::geom::Vec2;
use stark_core::{Lens, PerspectiveGuide};

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

/// Copy a guide into the row directly below the one it was copied from, and pick
/// the copy up — duplicating one is asking to shape a variant of it, which is the
/// same reason [`add_perspective`] opens the mode on what it made.
///
/// The copy carries the source's name as it stands, if it has one. The engine's
/// layer duplicate makes the same choice for the same reason (§14.8): a name is
/// the author's own word, and decorating it into "Horizon copy" would be inventing
/// one they never typed. An unnamed guide keeps being described by its position,
/// so the copy simply reads as the row it now is.
fn duplicate_guide(state: AppState, index: usize) {
    let mut guides = guides_of(state);
    let Some(guide) = guides.get(index).cloned() else {
        return;
    };
    guides.insert(index + 1, guide);
    dispatch(state, ViewCommand::SetGuides(guides));
    begin_guide_edit(state, index + 1);
}

/// Move the guide at `from` so that it sits at index `to`, and keep the edit mode
/// pointed at the **guide** it was pointed at rather than at the index.
///
/// That second half is not a nicety. This panel's rows are addressed by position —
/// a guide has no id — so every index in flight is a claim about a list that has
/// just changed underneath it, and the mode holds one. [`remove_guide`] carries the
/// same correction for the same reason; this is that rule for a move rather than a
/// removal.
///
/// The list is view state, so it goes back whole — the read-modify-commit shape
/// every mutation here takes ([`update_guide`]) — and there is no undo step to spend.
/// A drag that lands where it began is still declined, but for the plainer reason
/// that it is not a move.
fn move_guide(state: AppState, from: usize, to: usize) {
    let mut guides = guides_of(state);
    if from >= guides.len() || to >= guides.len() {
        return;
    }
    let guide = guides.remove(from);
    guides.insert(to, guide);
    dispatch(state, ViewCommand::SetGuides(guides));
    let mut mode = state.guide_edit;
    let current = *mode.peek();
    mode.set(current.map(|e| GuideEdit {
        index: moved(e.index, from, to),
        ..e
    }));
}

/// Where the guide at `i` ends up once the one at `from` has been taken out and put
/// back at `to` — the two steps, in that order, which is the only reading under which
/// `to` is an index into the list the drag was drawn against.
fn moved(i: usize, from: usize, to: usize) -> usize {
    if i == from {
        return to;
    }
    let out = if i > from { i - 1 } else { i };
    if out >= to { out + 1 } else { out }
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
        Some(name) => name.to_string(),
        None => format!("Perspective {}", index + 1),
    }
}

/// The Drawing Guides panel: the roster of guides, shaped like the Layers panel — a
/// header that adds, rows that select, rename, remove, hide, and drag to reorder.
///
/// The drag is the layer panel's, sharing its code (`panels::reorder`): the press,
/// the lift, the slot opening under the hand, the release. All this panel adds is
/// what a landing means, which for a flat list is an index — and the correction that
/// costs, since a guide is addressed by *position* and the mode holds one of those
/// (see [`move_guide`]).
#[component]
pub fn GuidesPanel() -> Element {
    let state = use_context::<AppState>();
    let guides = guides_of(state);
    let editing = (*state.guide_edit.read()).map(|e| e.index);
    // The in-flight row drag, if any — panel-local, and delimited by the browser's
    // own gesture rather than by a timer (§11).
    let mut drag = use_signal(|| None::<Grab>);
    // Resolved once here rather than read by each row, so the rows that do not move
    // do not re-render as the pointer travels. `lift` is where the dragged row is
    // drawn: straight down the column, because a flat list has no depth for a
    // sideways drag to choose — the one thing the layer tree asks of the hand that
    // this roster has nothing to ask.
    let (land, lift) = match drag.read().as_ref().filter(|d| d.live()) {
        Some(d) => {
            let keys: Vec<String> = (0..guides.len()).map(|i| i.to_string()).collect();
            let dy = d.delta().1;
            let slide = d
                .resolve(&keys)
                .and_then(|(from, boxes)| Slide::resolve(&boxes, (from, from), dy));
            (slide, (0.0, dy))
        }
        None => (None, (0.0, 0.0)),
    };

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
            GuideRow {
                key: "{i}",
                index: i,
                guide: g,
                active: editing == Some(i),
                motion: land.map_or_else(Motion::default, |s| s.motion(i, lift)),
                drag,
                onland: move |from: usize| {
                    // A press that never travelled is a click, and the browser is
                    // about to send one; nothing here has anything to say about it.
                    if drag.peek().as_ref().is_none_or(|d| !d.live()) {
                        drag.set(None);
                        return;
                    }
                    // The disarm first, so no frame carries both the new order and
                    // the transforms that were describing the old one — spent rather
                    // than dropped, so the click behind the release can be swallowed
                    // (`reorder::claimed`). It has to be: this panel's rows are
                    // addressed by position, so that click names whichever guide has
                    // just taken the dragged one's place, and acting on it would put
                    // the artist in the wrong guide's edit mode.
                    if let Some(d) = drag.write().as_mut() {
                        d.spend();
                    }
                    let Some(slide) = land else {
                        return;
                    };
                    if slide.inert() {
                        // A drag that went nowhere is the click it nearly was, and on
                        // this row a click is picking the guide up to shape it.
                        begin_guide_edit(state, from);
                    } else {
                        // Deliberately *not* an edit-mode entry: reordering the roster
                        // is tidying, and tidying must not take over the canvas. What
                        // you were shaping stays what you are shaping.
                        move_guide(state, from, slide.gap);
                    }
                },
            }
        }
    }
}

/// One guide in the roster. A component rather than markup inlined in the loop
/// above for the reason [`LayerRow`](super::layer::LayerRow) is one: the rename
/// field's draft is *row-local* state, so opening one leaves every other row alone
/// and closing it needs nothing cleaned up — and a hook cannot live inside a `for`.
#[component]
fn GuideRow(
    index: usize,
    guide: PerspectiveGuide,
    active: bool,
    motion: Motion,
    drag: Signal<Option<Grab>>,
    onland: EventHandler<usize>,
) -> Element {
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
            update_guide(state, index, move |g| g.name = Some(text.into()));
        }
    };
    let label = guide_label(index, &guide);
    // What the field opens on: the guide's *name*, which for one never named is
    // empty. Deliberately not the label — seeding with the generated "Perspective 2"
    // would turn opening the field and pressing Enter into a rename to "Perspective
    // 2", quietly making a description into a name, and this panel's descriptions
    // move when a guide is removed. The placeholder carries the label instead.
    let seed = guide.name.as_deref().unwrap_or_default().to_string();
    let visible = guide.visible;
    // The row's transform, written by `Motion` so every declaration is stated on
    // every render — including the ones that are "off" (see `reorder::Motion::css`).
    let class = format!(
        "guide-row{}{}",
        if active { " active" } else { "" },
        motion.class()
    );
    let shift = motion.css();

    rsx! {
        div {
            class: "{class}",
            style: "{shift}",
            // Which row this element is, for `platform::guide_boxes` to read back.
            // The index *is* the identity here — a guide has no id — which is sound
            // for the length of one gesture, since nothing reorders the list while a
            // pointer is down on it.
            "data-guide": "{index}",
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
                    title: "Shape this guide \u{2014} drag to reorder, double-click to rename",
                    // The click a drag leaves behind is not this row's: the drop has
                    // already said what it meant, and on a roster addressed by
                    // position that click names whichever guide took this one's place.
                    onclick: move |_| {
                        if !reorder::claimed(&mut drag) {
                            begin_guide_edit(state, index);
                        }
                    },
                    ondoubleclick: move |_| draft.set(Some(seed.clone())),
                    // The name is the grip, as it is on a layer row: the thing you
                    // would reach for to move a guide is the guide. Capture is what
                    // makes the release certain — it is delivered to the capturing
                    // element whatever the pointer is over by then, and this is a drag
                    // where everything under the pointer moves as you drag it.
                    onpointerdown: move |e: Event<PointerData>| {
                        capture_pointer(&e);
                        let p = e.client_coordinates();
                        drag.set(Some(Grab::begin(
                            index.to_string(),
                            guide_boxes(),
                            (p.x as f32, p.y as f32),
                        )));
                    },
                    onpointermove: move |e: Event<PointerData>| {
                        // The armed check first: it keeps every pointer move over the
                        // panel from dirtying the whole roster.
                        if drag.peek().is_none() {
                            return;
                        }
                        let p = e.client_coordinates();
                        if let Some(d) = drag.write().as_mut() {
                            d.track((p.x as f32, p.y as f32));
                        }
                    },
                    onpointerup: move |_| onland.call(index),
                    // A cancel — the browser taking the gesture, a pen leaving the
                    // tablet — ends it the same way.
                    onpointercancel: move |_| onland.call(index),
                    "{label}"
                }
            }
            // Duplicate, then Remove, then the eye — the order the Layers panel's rows
            // put them in, and the same three glyphs, because the two rosters differ in
            // what they list rather than in what these controls mean.
            button {
                class: "guide-duplicate",
                title: "Duplicate this guide",
                onclick: move |_| duplicate_guide(state, index),
                {icon(icons::DUPLICATE)}
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
    let (density, opacity, axes, lens) = (g.density, g.opacity, g.axes, g.lens);

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
            // The lens (§20.8): one toggle, because everything else about the
            // camera means the same thing under both projections. Lit, the
            // straight guide lines bow into the circles the stereographic
            // fisheye truly images them to, the second pole of every axis comes
            // into view, and the 90° ring — the classical 5-point grid's
            // boundary — appears around the center.
            button {
                class: if lens == Lens::Fisheye { "chip active" } else { "chip" },
                title: "Curvilinear (fisheye): a stereographic lens \u{2014} straight \
                        world lines bow into circles, and both poles of every axis \
                        come into view",
                onclick: move |_| update_guide(state, index, |g| {
                    g.lens = match g.lens {
                        Lens::Rectilinear => Lens::Fisheye,
                        Lens::Fisheye => Lens::Rectilinear,
                    };
                }),
                {icon(icons::FISHEYE)}
                {label("Fisheye")}
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
/// moves the construction, a ring is the lens, and everywhere else is the
/// world.
#[derive(Copy, Clone, PartialEq)]
enum GuideRegion {
    Center,
    /// A view-cone ring was grabbed; the payload is that ring's radius **per
    /// unit focal length** ([`Lens::ring_factors`]), so the drag divides the
    /// hand's distance back into a focal length. Carried in the drag because
    /// the fisheye shows two rings and the one grabbed must stay the one held
    /// — the 90° ring dragged inward must not hand off to the 45°.
    Focal(f32),
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
    // The camera numbers the hit test needs, read out here. Taken as values
    // rather than off `guide` so the closures below capture nothing that is not
    // `Copy` — a guide carries a name now, and a handler that captured the whole
    // guide could not be shared between the move and the release.
    let (center, focal, lens) = (guide.center, guide.focal, guide.lens);

    let to_canvas = move |e: &Event<PointerData>| view.screen_to_canvas(page_xy(e));
    let classify = move |pc: Vec2| {
        let on_screen = view.canvas_to_screen(center);
        if (on_screen - view.canvas_to_screen(pc)).length() < CENTER_GRAB_PX {
            return GuideRegion::Center;
        }
        // The lens's rings, nearest first: the fisheye shows two (45° and 90°,
        // §20.8) and a press between them grabs whichever is closer.
        let dist = (pc - center).length();
        let (r45, r90) = lens.ring_factors();
        let grabbed = [Some(r45), r90]
            .into_iter()
            .flatten()
            .map(|factor| ((dist - focal * factor).abs() * view.zoom, factor))
            .filter(|(err, _)| *err < CIRCLE_BAND_PX)
            .min_by(|a, b| a.0.total_cmp(&b.0));
        match grabbed {
            Some((_, factor)) => GuideRegion::Focal(factor),
            None => GuideRegion::Orbit,
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
            GuideRegion::Focal(factor) => update_guide(state, index, move |g| {
                g.focal =
                    ((pc - d.start.center).length() / factor).clamp(FOCAL_RANGE.0, FOCAL_RANGE.1);
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
        (_, None, Some(GuideRegion::Focal(_))) => "cursor: grab;",
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

#[cfg(test)]
mod tests {
    use super::*;

    /// [`moved`] must agree with the list surgery it describes, for **every** pair of
    /// positions: take the guide at `from` out, put it back at `to`, and every other
    /// guide's new index is the one it claims.
    ///
    /// Exhaustive rather than sampled, because this is off-by-one arithmetic over two
    /// steps that shift indices in opposite directions, and the pair that is wrong is
    /// never the one anyone would think to write down. It is worth the certainty: the
    /// edit mode holds one of these indices, so an error here points the Perspective
    /// bar — and every drag on the canvas — at a guide the artist did not touch.
    #[test]
    fn the_index_remap_agrees_with_the_move_it_describes() {
        for n in 1..7usize {
            let list: Vec<usize> = (0..n).collect();
            for from in 0..n {
                for to in 0..n {
                    let mut after = list.clone();
                    let guide = after.remove(from);
                    after.insert(to, guide);
                    for i in 0..n {
                        assert_eq!(
                            after[moved(i, from, to)],
                            i,
                            "n={n} from={from} to={to}: guide {i} is not where it was sent"
                        );
                    }
                }
            }
        }
    }

    /// A flat list's landing is an index, and the one the shared gesture reports is
    /// the index to insert at **once the row has been taken out**.
    ///
    /// Asserted end to end rather than trusted, because "counted in the rows that stay
    /// put" and "index into the list after the removal" are the same number for a
    /// reason that is one sentence long and easy to get backwards — and getting it
    /// backwards is off by one only in the direction you dragged.
    #[test]
    fn a_row_lands_where_it_was_dropped() {
        const H: f32 = 20.0;
        let boxes: Vec<(f32, f32)> = (0..4).map(|i| (i as f32 * H, H)).collect();
        let order = |from: usize, dy: f32| {
            let slide = Slide::resolve(&boxes, (from, from), dy).expect("resolves");
            let mut list: Vec<usize> = (0..4).collect();
            let row = list.remove(from);
            list.insert(slide.gap, row);
            list
        };
        assert_eq!(order(0, H), vec![1, 0, 2, 3], "one row down");
        assert_eq!(order(0, 3.0 * H), vec![1, 2, 3, 0], "to the foot");
        assert_eq!(order(3, -3.0 * H), vec![3, 0, 1, 2], "to the head");
        assert_eq!(order(1, 0.0), vec![0, 1, 2, 3], "nowhere at all");
    }
}
