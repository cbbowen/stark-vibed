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
//! - **anywhere** — grab the world: the direction under the pointer follows it
//!   exactly, the free arc ([`PerspectiveGuide::dragged`], §20.5);
//! - **a horizon** — turn about one axis: the vanishing line between two axes'
//!   vanishing points belongs to the third, so grabbing the line between the X
//!   and Z vanishing points orbits Y and nothing else
//!   ([`PerspectiveGuide::horizons`]);
//! - **the 45° circle** — drag the lens: the circle's radius *is* the focal
//!   length, so it follows the hand exactly;
//! - **the center-of-view crosshair** — move the whole construction.
//!
//! A constrained turn is therefore something the hand *reaches for*, not
//! something a free drag falls into on its way past an axis.
//!
//! The **Perspective Guide bar** stands at the bottom for the mode's duration:
//! per-axis locks (constraining the drag — lock the vertical and 2-point
//! stays 2-point under any gesture), per-plane visibility (XY / YZ / ZX, each
//! chip lettered in its two axes' own hues), the cell count, opacity,
//! and "Done". Locks name axes and visibility names planes because a rotation
//! is about an axis and a guide line is drawn in a plane. There is no case switch anywhere: which of 1/2/3-point you are
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
use stark_core::{Lens, PairTrace, PerspectiveGuide};

/// The axis hues, by **name**: `stark.css` declares `--axis-x/y/z` and this
/// side never learns what they are. The colours belong to the app rather than
/// to this bar — the guide's own lines are drawn in them too — so they are
/// stated once, in the stylesheet, in Oklab and at one shared lightness, and
/// everything that wears one refers to it.
///
/// A control that needed to *compute* with a hue would have to be given the
/// numbers; none does. A chip either wears a hue or interpolates between two,
/// and CSS does both from the variable — which is also why the plane chip's
/// gradient can run `in oklab` without this file knowing that word.
///
/// `guides.wesl` is the one place that cannot follow, needing shader constants;
/// [`the_chips_are_painted_in_the_shader_s_own_axis_hues`] parses the
/// stylesheet and holds the two together.
const AXIS_CSS: [&str; 3] = ["var(--axis-x)", "var(--axis-y)", "var(--axis-z)"];
const AXIS_NAMES: [&str; 3] = ["X", "Y", "Z"];

/// The two axes of each pair plane, in the order the chip shows them: XY, YZ,
/// ZX.
///
/// The model's own cyclic order, pair `k` being spanned by axes `(k, k+1)`, and
/// the chips are read in it rather than sorted: the three then run X→Y→Z→X, so
/// each chip picks up where the last left off and every axis letter appears
/// exactly twice, once on each side. Sorting the last one to "XZ" would break
/// that at the only place it shows.
const PAIR_AXES: [[usize; 2]; 3] = [[0, 1], [1, 2], [2, 0]];

/// Grab radius of the center-of-view crosshair, screen px.
const CENTER_GRAB_PX: f32 = 14.0;
/// Half-width of the grab band around a drawn curve — a view-cone ring or a
/// horizon — in screen px, so a handle is equally grabbable at any
/// magnification. One number for both because the two are the same ask of the
/// hand: put the pointer on a line about a pixel wide.
const LINE_BAND_PX: f32 = 10.0;
/// The lens's travel, canvas px: wide enough for any drawing, floored so the
/// circle cannot be dragged through its own center into a degenerate camera.
const FOCAL_RANGE: (f32, f32) = (120.0, 12000.0);

/// The cell scale the bar offers, as **halvings** of the default lattice
/// (§20.3): two steps coarser to two steps finer, and nothing in between.
///
/// The model draws a valid grid at any scale; this is what the control offers,
/// and the reason is that a grid meant to be counted on should *refine* rather
/// than slide. Double the cells and every line of the coarser grid is still a
/// line of the finer one with a new line between each pair — nothing an artist
/// has already counted against moves. At any other ratio the whole family slides
/// along its pencil toward the corner's own edge, which reads as the grid
/// drifting sideways rather than as a change of scale.
const CELL_OCTAVES: (i32, i32) = (-2, 2);

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
    // The grid's scale is the *length* of the lattice: how many cells lie
    // between the eye and its corner, which is all a camera with no world scale
    // of its own can say about the size of a cell (§20.3). The bar states it in
    // halvings of the default ([`CELL_OCTAVES`]) — the guide's own length is the
    // ladder's rung, so there is no separate number to keep in step.
    let base = PerspectiveGuide::default().lattice;
    let octave = (g.lattice.length() / base.length()).log2().round();
    let (opacity, pairs, lens) = (g.opacity, g.pairs, g.lens);

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
            div {
                class: "segmented",
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
            }
            span { class: "bar-sep" }
            // The same eye the guide's own row wears, asked of one plane of the
            // grid rather than of the whole guide. A plane rather than an axis
            // because a plane is what is drawn — a guide line lies *in* one
            // (§20.3) — and because the three planes are independently
            // switchable where three axes were not: two axes could never show
            // one plane without a second coming free with them.
            span { class: "bar-sub",
                {icon(icons::VISIBLE)}
                {label("Show")}
            }
            div {
                class: "segmented",
                for k in 0..3 {
                    {
                        let [a, b] = PAIR_AXES[k];
                        rsx! {
                            button {
                                class: if pairs[k] { "chip plane-chip active" } else { "chip plane-chip" },
                                style: "--axis-a: {AXIS_CSS[a]}; --axis-b: {AXIS_CSS[b]}",
                                title: "Show the {AXIS_NAMES[a]}{AXIS_NAMES[b]} plane \u{2014} \
                                        its two fans of guide lines, its horizon and its \
                                        station point",
                                onclick: move |_| update_guide(state, index, move |g| {
                                    g.pairs[k] = !g.pairs[k];
                                }),
                                // Each letter in its own axis's hue, so the chip
                                // names the plane by the two colors ruling it.
                                span { class: "ax-a", "{AXIS_NAMES[a]}" }
                                span { class: "ax-b", "{AXIS_NAMES[b]}" }
                            }
                        }
                    }
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
                {label("Cells")}
            }
            input {
                class: "slider",
                r#type: "range",
                min: "{CELL_OCTAVES.0}", max: "{CELL_OCTAVES.1}", step: "1",
                value: "{octave}",
                title: "How fine the grid is \u{2014} each step halves the cell, so \
                        every line of the coarser grid is still a line of this one",
                // Stepped off the *default* rather than off the guide's current
                // lattice, so a rung is the same grid however it was reached.
                // Where the corner sits is the drag's business, and there is no
                // drag for it yet (§20.5) — when there is, this scales about it,
                // and the grid stays hung on the viewer either way (§20.3).
                oninput: move |e| {
                    if let Ok(k) = e.value().parse::<f32>() {
                        update_guide(state, index, move |g| {
                            g.lattice = base * 2f32.powi(k.round() as i32);
                        });
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

/// What a press at a point would grab, nearest wins: the crosshair moves the
/// construction, a ring is the lens, a horizon is a turn about one axis, and
/// everywhere else is the world.
#[derive(Copy, Clone, Debug, PartialEq)]
enum GuideRegion {
    Center,
    /// A view-cone ring was grabbed; the payload is that ring's radius **per
    /// unit focal length** ([`Lens::ring_factors`]), so the drag divides the
    /// hand's distance back into a focal length. Carried in the drag because
    /// the fisheye shows two rings and the one grabbed must stay the one held
    /// — the 90° ring dragged inward must not hand off to the 45°.
    Focal(f32),
    /// A **horizon** was grabbed (§20.5): the vanishing line between two axes'
    /// vanishing points, which is the plane normal to the third — so the
    /// payload is that third axis, and the drag turns about it. Grab the line
    /// between the X and Z vanishing points and the camera orbits Y.
    Horizon(usize),
    Orbit,
}

/// An in-flight guide drag: what it grabbed, where it started in canvas px,
/// and the guide as it was then. Recomputed from the start on every move — the
/// same discipline as the transform drag (§16.6), and here it is also what
/// makes a constrained turn a decision of the *press*: the region is
/// classified once, so which axis a drag turns about is settled before the
/// hand has moved and cannot change under it.
#[derive(Clone)]
struct Drag {
    region: GuideRegion,
    from: Vec2,
    start: PerspectiveGuide,
}

/// The guide's grabbable geometry, read out once when the overlay renders: all
/// canvas-space, and `Copy`, so the pointer handlers can share the hit test
/// without any of them holding the guide itself (which carries a name, and is
/// not `Copy`).
#[derive(Copy, Clone)]
struct Handles {
    center: Vec2,
    focal: f32,
    lens: Lens,
    /// Horizon `n` is the one that turns about axis `n`, and it is `None`
    /// where the guide does not draw it — you cannot grab a line that is not
    /// on the screen ([`PerspectiveGuide::horizons`]).
    horizons: [Option<PairTrace>; 3],
}

impl Handles {
    fn of(g: &PerspectiveGuide) -> Self {
        Self {
            center: g.center,
            focal: g.focal,
            lens: g.lens,
            horizons: g.horizons(),
        }
    }

    /// What a press at canvas point `p` grabs, with the view at `zoom`.
    ///
    /// The crosshair is topmost, as it is drawn; below it the rings and the
    /// horizons compete on **distance in screen px**, so a press between two
    /// curves takes the one it is nearer and every handle is equally grabbable
    /// at any magnification. A tie goes to the ring: under the fisheye a pair
    /// trace can *be* a ring (in a 1-point pose the 90° ring is the X/Y
    /// horizon, §20.8), and the lens drag is the older, more-reached-for
    /// gesture to leave in the artist's hand where the two coincide.
    fn at(self, p: Vec2, zoom: f32) -> GuideRegion {
        if (p - self.center).length() * zoom < CENTER_GRAB_PX {
            return GuideRegion::Center;
        }
        let dist = (p - self.center).length();
        let (r45, r90) = self.lens.ring_factors();
        let rings = [Some(r45), r90]
            .into_iter()
            .flatten()
            .map(|factor| ((dist - self.focal * factor).abs() * zoom, factor));
        let horizons = self
            .horizons
            .into_iter()
            .enumerate()
            .filter_map(|(n, trace)| Some((trace?.distance(p) * zoom, n)));
        let ring = rings
            .filter(|(err, _)| *err < LINE_BAND_PX)
            .min_by(|a, b| a.0.total_cmp(&b.0));
        let horizon = horizons
            .filter(|(err, _)| *err < LINE_BAND_PX)
            .min_by(|a, b| a.0.total_cmp(&b.0));
        match (ring, horizon) {
            (Some((re, _)), Some((he, n))) if he < re => GuideRegion::Horizon(n),
            (Some((_, factor)), _) => GuideRegion::Focal(factor),
            (None, Some((_, n))) => GuideRegion::Horizon(n),
            (None, None) => GuideRegion::Orbit,
        }
    }
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
    // The grabbable geometry, derived once. Taken as a value rather than off
    // `guide` so the closures below capture nothing that is not `Copy` — a
    // guide carries a name now, and a handler that captured the whole guide
    // could not be shared between the move and the release.
    let handles = Handles::of(&guide);

    let to_canvas = move |e: &Event<PointerData>| view.screen_to_canvas(page_xy(e));
    let classify = move |pc: Vec2| handles.at(pc, view.zoom);

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
            // One drag under two constraints (§20.5). Grabbing a horizon holds
            // the axis it belongs to for the gesture's duration, and holding an
            // axis fixed is exactly turning about it — the same thing a lock
            // chip says, so it arrives as one and there is no third rotation
            // path to keep in step. Two constraints that cannot both hold —
            // the Y lock lit and the X horizon grabbed — pin the frame, which
            // is the standing rule for two locks rather than a new one.
            //
            // The turn only, rather than the whole guide the drag was started
            // from: a drag is a statement about the camera's orientation, and
            // writing back a snapshot would also write back the name, opacity and
            // lattice as they stood at the press. Assigning the one field the drag
            // computes leaves nothing for a mid-drag edit elsewhere to lose.
            GuideRegion::Orbit | GuideRegion::Horizon(_) => {
                let mut held = locked;
                if let GuideRegion::Horizon(n) = d.region {
                    held[n] = true;
                }
                update_guide(state, index, move |g| {
                    g.rotation = d.start.dragged(d.from, pc, held).rotation;
                })
            }
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
        // A ring and a horizon are both handles lying on the canvas, so both
        // read as something to take hold of; the free world grab is the one
        // that is not a handle, and says so.
        (_, None, Some(GuideRegion::Focal(_) | GuideRegion::Horizon(_))) => "cursor: grab;",
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

    /// The three axis hues as the shipped stylesheet declares them — parsed out
    /// of the file itself, so what the tests below hold is what the browser
    /// gets rather than a copy of it that could drift.
    fn declared_hues() -> [[f32; 3]; 3] {
        const CSS: &str = include_str!("../../assets/stark.css");
        std::array::from_fn(|i| {
            let decl = format!("--axis-{}: oklab(", ["x", "y", "z"][i]);
            let at = CSS
                .find(&decl)
                .unwrap_or_else(|| panic!("stark.css declares no `{decl}…)`"));
            let rest = &CSS[at + decl.len()..];
            let body = &rest[..rest.find(')').expect("an unclosed oklab()")];
            let lab: Vec<f32> = body
                .split_whitespace()
                .map(|n| n.parse().expect("a number"))
                .collect();
            assert_eq!(lab.len(), 3, "`{decl}{body})` is not an L, a, b");
            [lab[0], lab[1], lab[2]]
        })
    }

    /// The bar is painted in the guide pass's own axis hues (§20.4) — a lock
    /// chip, a plane chip and the lines they govern are one colour, or the
    /// controls stop looking like they belong to what they control.
    ///
    /// The two declarations cannot be merged: `guides.wesl` needs its colours as
    /// shader constants and cannot read a stylesheet, and the mirror carries
    /// scalars only (a `vec3` has no host constant). So they are two statements
    /// of one fact, and this is the thing that notices when they part — which
    /// matters more than it did, now that one reads
    /// `oklab(0.667 0.1675 0.0664)` and the other
    /// `vec3(0.9349, 0.3629, 0.3803)`. `#e8575c` beside
    /// `vec3(0.91, 0.34, 0.36)` could be checked by a reader who cared to;
    /// these cannot be checked by anyone.
    ///
    /// The tolerance is a **quantization step**, not a fudge: what has to
    /// survive both roundings is the 8-bit colour the screen shows, and
    /// anything inside 1/255 is the same pixel. It is also why the shader's
    /// constants carry four decimals — at two, the conversion of a round Oklab
    /// lightness misses by more than that.
    #[test]
    fn the_chips_are_painted_in_the_shader_s_own_axis_hues() {
        // `guides.wesl`'s AXIS_X / AXIS_Y / AXIS_Z, display sRGB.
        const SHADER: [[f32; 3]; 3] = [
            [0.9349, 0.3629, 0.3803],
            [0.2932, 0.6746, 0.3667],
            [0.3922, 0.5631, 0.9544],
        ];
        for (i, (lab, want)) in declared_hues().iter().zip(&SHADER).enumerate() {
            // The bar wears the variable rather than a value, so the name has to
            // be the declared one — a typo'd `var()` is simply no colour.
            let name = format!("var(--axis-{})", ["x", "y", "z"][i]);
            assert_eq!(AXIS_CSS[i], name, "the bar points at nothing");

            let got = stark_core::color::oklab_to_srgb([lab[0], lab[1], lab[2], 1.0]);
            for c in 0..3 {
                assert!(
                    (got[c] - want[c]).abs() < 1.0 / 255.0,
                    "axis {i}: {name} is {:?}, but the shader draws {want:?}",
                    &got[..3]
                );
            }
        }
    }

    /// **No axis reads heavier than the others**: all three hues sit at one
    /// Oklab lightness, and only the hue tells them apart.
    ///
    /// The three are laid side by side on the bar, and two of them are
    /// interpolated into a plane chip's gradient, so a difference in lightness
    /// reads as one control being more emphatic than its neighbours — and as a
    /// gradient with a bright end. Written as hex they had drifted 0.046 of `L`
    /// apart with the green on top, and nothing in the source could show it.
    /// That is the argument for stating a colour in a space with a lightness
    /// axis, and this is the part of it that can be checked.
    ///
    /// Exact equality rather than a tolerance: the claim is about how the three
    /// are *written*, and what it asks is that one number appears in all of
    /// them.
    #[test]
    fn the_axis_hues_carry_equal_weight() {
        let [x, y, z] = declared_hues();
        assert_eq!(
            [x[0], y[0], z[0]],
            [x[0]; 3],
            "the axis hues are at different lightnesses"
        );
    }

    /// Where each of the default guide's horizons sits, as the one coordinate
    /// it is a level set of.
    ///
    /// The default is 2-point at 30° of yaw, centred on the origin at a focal
    /// length of 900, so all three horizons are canvas-aligned: a vertical
    /// through each transverse vanishing point, and the level horizon through
    /// the center of view. That is enough distinct geometry to press against
    /// and simple enough to write the answers down — and it is asserted rather
    /// than assumed, so a change to the default pose fails here instead of
    /// quietly aiming every press below at empty canvas.
    fn horizons_of(g: &PerspectiveGuide) -> [f32; 3] {
        std::array::from_fn(|n| match g.horizons()[n] {
            Some(PairTrace::Line { normal, offset }) => {
                let axial = normal.x + normal.y;
                assert!(
                    (axial.abs() - 1.0).abs() < 1e-4,
                    "horizon {n} is not canvas-aligned: {normal:?}"
                );
                -offset * axial
            }
            other => panic!("horizon {n} should be a straight line, got {other:?}"),
        })
    }

    /// Grabbing a horizon asks to turn about **its own** axis (§20.5) — the
    /// line between two axes' vanishing points belongs to the third, and the
    /// press has to come back with that third one.
    ///
    /// The index is the whole risk here: every horizon is a line in the same
    /// list and picking the wrong one is a gesture that turns the guide about
    /// an axis the artist did not reach for — which looks like a bug in the
    /// rotation, not in a subscript.
    #[test]
    fn a_press_on_a_horizon_grabs_the_axis_it_turns_about() {
        let g = PerspectiveGuide::default();
        let h = Handles::of(&g);
        let [x_at, y_at, z_at] = horizons_of(&g);
        // Axis 1's horizon is the level one through the center of view (the
        // classical horizon); the other two are the verticals through the
        // transverse vanishing points.
        assert_eq!(h.at(Vec2::new(x_at, 400.0), 1.0), GuideRegion::Horizon(0));
        assert_eq!(h.at(Vec2::new(300.0, y_at), 1.0), GuideRegion::Horizon(1));
        assert_eq!(h.at(Vec2::new(z_at, 500.0), 1.0), GuideRegion::Horizon(2));
    }

    /// Everything else the press can land on still does, and the band is in
    /// **screen** px: the same canvas point is a horizon grab zoomed out and
    /// open world zoomed in, because what the hand can hit is a distance on the
    /// screen, not on the canvas.
    #[test]
    fn the_other_regions_survive_the_horizons() {
        let g = PerspectiveGuide::default();
        let h = Handles::of(&g);
        let [x_at, ..] = horizons_of(&g);
        assert_eq!(h.at(Vec2::new(4.0, -3.0), 1.0), GuideRegion::Center);
        assert_eq!(h.at(Vec2::new(0.0, g.focal), 1.0), GuideRegion::Focal(1.0));
        assert_eq!(h.at(Vec2::new(100.0, 300.0), 1.0), GuideRegion::Orbit);

        let near = Vec2::new(x_at + 50.0, 400.0);
        assert_eq!(h.at(near, 1.0), GuideRegion::Orbit, "50px is a miss");
        assert_eq!(
            h.at(near, 0.1),
            GuideRegion::Horizon(0),
            "…and 5 screen px is a hit"
        );
    }

    /// Two handles in reach: the nearer one wins, and an exact tie goes to the
    /// ring. The tie is not hypothetical — a horizon crosses the 45° circle in
    /// every pose, and under the fisheye a pair trace can *be* a ring (§20.8).
    #[test]
    fn the_nearer_handle_wins_and_a_tie_goes_to_the_ring() {
        let g = PerspectiveGuide::default();
        let h = Handles::of(&g);
        let [_, y_at, _] = horizons_of(&g);
        // Where the level horizon crosses the 45° ring, both errors are zero.
        assert_eq!(
            h.at(Vec2::new(g.focal, y_at), 1.0),
            GuideRegion::Focal(1.0),
            "a dead tie is the lens"
        );
        // 2px off the horizon and 8px outside the ring.
        assert_eq!(
            h.at(Vec2::new(g.focal + 8.0, y_at + 2.0), 1.0),
            GuideRegion::Horizon(1)
        );
        // 4px off the horizon, all but on the ring.
        assert_eq!(
            h.at(Vec2::new(g.focal, y_at + 4.0), 1.0),
            GuideRegion::Focal(1.0)
        );
    }

    /// A horizon that is not drawn is not grabbable, and the press falls
    /// through to the free world grab — the rule the guide states
    /// ([`PerspectiveGuide::horizons`]) carried all the way to the hand.
    /// Switching a plane off takes the one horizon that turns about its normal
    /// and leaves the other two.
    #[test]
    fn an_undrawn_horizon_cannot_be_grabbed() {
        let mut g = PerspectiveGuide::default();
        let [x_at, y_at, _] = horizons_of(&g);
        // Pair 1 is the Y/Z plane, normal to X.
        g.pairs = [true, false, true];
        let h = Handles::of(&g);
        assert_eq!(h.at(Vec2::new(x_at, 400.0), 1.0), GuideRegion::Orbit);
        assert_eq!(h.at(Vec2::new(300.0, y_at), 1.0), GuideRegion::Horizon(1));
    }

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
