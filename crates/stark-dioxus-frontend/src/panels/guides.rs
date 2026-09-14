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

use dioxus::html::input_data::MouseButton;
use dioxus::prelude::*;

use crate::icons::{icon, label};
use crate::input::{Nav, canvas_xy};
use crate::panels::reorder::{Grip, RowKey};
use crate::preview;
use crate::state::{AppState, dispatch, use_obs_opt};
use crate::widgets::{ActChip, Bar, Chip, CommandButton, InlineRename, PreviewSlider, SliderShape};
use stark_engine::GuideInfo;
use stark_engine::command::{DocCommand, ViewCommand};
use stark_model::document::{GuideId, Lens, PerspectiveGuide};
use stark_model::geom::Vec2;
use stark_ui::commands::Command;
use stark_ui::guides::{
    AXIS_NAMES, CELL_OCTAVES, FOCAL_RANGE, GuideRegion, Handles, PAIR_AXES, anchor_at,
};
use stark_ui::modes::{Composing, GuideEdit};
use stark_ui::reorder::{Grab, Motion, Slide};

/// The axis hues, by **name**: `stark.css` declares `--axis-x/y/z` and this
/// side never learns what they are. The colors belong to the app rather than
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
/// `tests::the_chips_are_painted_in_the_shader_s_own_axis_hues` parses the
/// stylesheet and holds the two together.
const AXIS_CSS: [&str; 3] = ["var(--axis-x)", "var(--axis-y)", "var(--axis-z)"];

/// The opacity track's ends, off the dial that owns them
/// (`stark_ui::guides::Dial::Opacity`) rather than spelled here three times as the
/// fill, the `min` and the `max`.
const OPACITY_RANGE: (f32, f32) = stark_ui::guides::Dial::Opacity.range();

/// The engine's guide roster, as this client sees it (§20.5) — the document's
/// guides, each row carrying whether this client's eye on it is open.
fn guides_of(state: AppState) -> Vec<GuideInfo> {
    state
        .obs
        .read()
        .as_ref()
        .map(|o| o.guides.to_vec())
        .unwrap_or_default()
}

/// The camera of the guide `id` as the engine holds it now, or `None` if it has
/// gone.
///
/// Read back per edit rather than kept beside the mode, and that is worth saying:
/// the roster is projected off the **previewed** document, so mid-drag this
/// reports what the canvas is showing. Every reader below wants that — a chip
/// clicked during nothing is reading the committed camera, and a chip clicked
/// while a slider is still open is reading the one under the hand.
fn camera_of(state: AppState, id: GuideId) -> Option<PerspectiveGuide> {
    guides_of(state)
        .into_iter()
        .find(|g| g.id == id)
        .map(|g| g.guide)
}

/// Adjust one guide's camera and **lay it down** — one logged action, one undo
/// step (§20.5).
///
/// For the controls that settle the moment they are used: a plane chip, the lens
/// toggle, a keyboard nudge. A drag previews instead — [`drag_guide`] on the canvas,
/// a `PreviewSlider` on the bar — so the artist sees every value the pointer crosses
/// and pays for one.
fn edit_guide(state: AppState, id: GuideId, f: impl FnOnce(&mut PerspectiveGuide)) {
    let Some(mut camera) = camera_of(state, id) else {
        return;
    };
    f(&mut camera);
    preview::GUIDE.commit(state, (id, camera));
}

/// Adjust one guide's camera and **show it without logging it** — one sample of a
/// drag (§20.5).
///
/// The pending half of [`preview::GUIDE`]: the value is held so the release lays
/// down exactly what the canvas is showing, and the caller wires
/// [`settle_guide`] to every event that can end the gesture.
fn drag_guide(
    state: AppState,
    pending: Signal<Option<(GuideId, PerspectiveGuide)>>,
    id: GuideId,
    f: impl FnOnce(&mut PerspectiveGuide),
) {
    let Some(mut camera) = camera_of(state, id) else {
        return;
    };
    f(&mut camera);
    preview::GUIDE.during(state, pending, (id, camera));
}

/// End a guide drag: lay down what the previews have been showing, once.
/// Idempotent, which is what lets it hang off every event that can end one
/// ([`Preview::settle`](crate::preview::Preview::settle)).
fn settle_guide(state: AppState, pending: Signal<Option<(GuideId, PerspectiveGuide)>>) {
    preview::GUIDE.settle(state, pending);
}

/// Enter the edit mode on guide `id` — fresh locks each time: a lock is a
/// constraint on the hand for one sitting, not a fact about the guide.
///
/// **Opens the guide's eye**, and this is the one place that does it besides the
/// row's own eye control. A guide is hidden until asked for (§20.5), so shaping
/// one you cannot see would be dragging an invisible construction — and picking a
/// guide up *is* asking to see it. Adding and duplicating come free: both end by
/// picking up what they made, so the rule is written once here rather than at
/// each path that produces a guide.
///
/// It stays open afterwards. Leaving the mode does not shut it again, because by
/// then it is an opinion the artist expressed rather than one the tool assumed.
pub fn begin_guide_edit(state: AppState, id: GuideId) {
    dispatch(state, ViewCommand::SetGuideVisible(id, true));
    // One composing mode at a time (`crate::modes`): picking a guide up puts
    // down a transform or a gradient axis rather than stacking a second catcher
    // over the first, and `enter` is what does that — it is the only way in.
    crate::modes::enter(
        state,
        Composing::GuideEdit(GuideEdit {
            id,
            locked: [false; 3],
        }),
    );
}

/// Leave the edit mode — the bar's "Done", and Enter's (`crate::modes::finish`).
///
/// Every control on the bar lays its own value down as it settles (§20.5), so
/// there is nothing half-composed here for a commit and a cancel to differ
/// about — which is why the bar offers no Cancel chip. The preview is dropped on
/// the way out all the same: a mode left while a pointer is still down would
/// otherwise strand the pose under the hand on the canvas with no commit coming
/// to supersede it.
pub fn end_guide_edit(state: AppState) {
    // `leave` drops this mode's preview itself (`crate::modes`), which is the
    // whole of what leaving costs here — every control on the bar has already
    // laid its own value down.
    crate::modes::leave(state);
}

/// Add a perspective guide where the artist is looking, and pick it up: adding
/// *is* asking to shape it, so the mode opens on the new row
/// (`Command::AddPerspective`).
pub fn add_perspective(state: AppState) {
    let center = state
        .obs
        .peek()
        .as_ref()
        .map(|o| o.view.center)
        .unwrap_or(Vec2::ZERO);
    let after = guides_of(state).last().map(|g| g.id);
    add_and_edit(
        state,
        DocCommand::AddGuide {
            guide: PerspectiveGuide {
                center,
                ..Default::default()
            },
            after,
            name: None,
        },
    );
}

/// Dispatch `add` — a command that adds one guide — and pick up the guide it added.
///
/// The engine mints no id to hand back — a guide's identity is the id of the action that
/// added it (§20.5) — so the new guide is *found*, by comparing the roster before and
/// after (`stark_ui::mint`). `dispatch` refreshes the projection inside the engine's own
/// borrow, so the second read already has it.
fn add_and_edit(state: AppState, add: DocCommand) {
    let before: Vec<GuideId> = guides_of(state).iter().map(|g| g.id).collect();
    dispatch(state, add);
    let added = stark_ui::mint::minted(&before, guides_of(state).into_iter().map(|g| g.id));
    if let Some(added) = added {
        begin_guide_edit(state, added);
    }
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
///
/// One action rather than an add and then a rename, which is why `AddGuide`
/// carries a name at all: two would put one gesture two undo steps deep with a
/// nameless guide in between.
fn duplicate_guide(state: AppState, id: GuideId) {
    let Some(source) = guides_of(state).into_iter().find(|g| g.id == id) else {
        return;
    };
    add_and_edit(
        state,
        DocCommand::AddGuide {
            guide: source.guide,
            after: Some(id),
            name: source.name.as_deref().map(str::to_owned),
        },
    );
}

/// Move the guide `id` so that it sits at index `to` in the roster.
///
/// The index is the drag's own answer — `reorder::Slide` speaks in gaps — and it
/// is turned into the anchor the action carries here, which is the guide the
/// dragged one lands *after*. The two readings differ by one at exactly one
/// place, the head of the roster, and naming a guide rather than a position is
/// what makes the action mean the same thing on a peer whose roster has moved.
///
/// Nothing to re-point afterwards, unlike the version this replaces: the mode
/// holds a `GuideId` now, and an id does not shift when the list closes up
/// behind a row.
fn move_guide(state: AppState, id: GuideId, to: usize) {
    let ids: Vec<GuideId> = guides_of(state).iter().map(|g| g.id).collect();
    let Some(from) = ids.iter().position(|g| *g == id) else {
        return;
    };
    dispatch(
        state,
        DocCommand::MoveGuide {
            id,
            after: anchor_at(&ids, from, to),
        },
    );
}

/// Remove a guide, and end the edit mode if it was the guide being shaped.
///
/// The re-pointing this used to carry is gone with the index it corrected: the
/// mode names a guide, so every other row's mode survives a removal untouched.
fn remove_guide(state: AppState, id: GuideId) {
    dispatch(state, DocCommand::RemoveGuide(id));
    let editing = crate::modes::composing_now(state).and_then(Composing::guide_edit);
    if editing.is_some_and(|e| e.id == id) {
        crate::modes::leave(state);
    }
}

/// What to call a guide that has never been named: its place in the roster.
///
/// The Layers panel's counterpart numbers by *mint* order
/// ([`layer_label`](stark_ui::layer_tree::layer_label)), which survives a reorder;
/// this numbers by *position*, which does not. Both shift when a row above is
/// removed, and that is the honest reading either way — an unnamed row is being
/// described, not named, and the description of the second row is "the second one".
/// Naming it is how you stop it moving.
fn guide_label(index: usize, guide: &GuideInfo) -> String {
    stark_ui::guides::label(index, guide)
}

/// The Drawing Guides panel: the roster of guides, shaped like the Layers panel — a
/// header that adds, rows that select, rename, remove, hide, and drag to reorder.
///
/// The drag is the layer panel's, sharing its code (`panels::reorder`): the press,
/// the lift, the slot opening under the hand, the release. All this panel adds is
/// what a landing means, which for a flat list is a gap between rows — turned into
/// the guide it lands *after* at the one place that knows both (see
/// [`move_guide`]), because an action has to name a guide rather than a position
/// to mean the same thing on a peer.
#[component]
pub fn GuidesPanel() -> Element {
    let state = use_context::<AppState>();
    let guides = guides_of(state);
    let editing = crate::modes::composing(state)
        .and_then(Composing::guide_edit)
        .map(|e| e.id);
    // The in-flight row drag, if any — panel-local, and delimited by the browser's
    // own gesture rather than by a timer (§11).
    let drag = use_signal(|| None::<Grab>);
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
            CommandButton { command: Command::AddPerspective, class: "layer-add" }
        }
        // The roster, in a well of its own — the Layers panel's tree in the same
        // ink (`.guide-list`, `.layer-tree`), because it is the same statement: the
        // rows are the panel's one *picture* (the guides, listed), and the ground
        // under a picture is not the ground the controls stand on. The header keeps
        // its inset above it, so the well never reaches a corner — which is what lets
        // this panel go on not clipping, as its dragged row needs (`.panel`).
        //
        // The empty line is *inside* it, so the region a guide would appear in is
        // drawn whether or not one has been added yet, and the panel does not change
        // shape under the first add.
        div { class: "guide-list",
            if guides.is_empty() {
                div { class: "guide-empty",
                    "No guides yet. Add a perspective grid to draw through."
                }
            }
            for (i, g) in guides.into_iter().enumerate() {
                GuideRow {
                    key: "{i}",
                    index: i,
                    active: editing == Some(g.id),
                    guide: g,
                    motion: land.map_or_else(Motion::default, |s| s.motion(i, lift)),
                    drag,
                    onland: move |from: usize| {
                        let Some(slide) = land else {
                            return;
                        };
                        // The row the drag was drawn against, by *position*, which is all
                        // a drag over drawn rows can name. Turned into the guide's own id
                        // here and never carried further: everything downstream of this
                        // handler addresses a guide by id (§20.5).
                        let Some(id) = guides_of(state).get(from).map(|g| g.id) else {
                            return;
                        };
                        if slide.inert() {
                            // A drag that went nowhere is the click it nearly was, and on
                            // this row a click is picking the guide up to shape it.
                            begin_guide_edit(state, id);
                        } else {
                            // Deliberately *not* an edit-mode entry: reordering the roster
                            // is tidying, and tidying must not take over the canvas. What
                            // you were shaping stays what you are shaping.
                            move_guide(state, id, slide.gap);
                        }
                    },
                }
            }
        }
    }
}

/// One guide in the roster. A component rather than markup inlined in the loop
/// above for the reason [`LayerRow`](super::layer::LayerRow) is one: whether its name
/// is open for renaming is *row-local* state, so opening one leaves every other row
/// alone — and a hook cannot live inside a `for`.
#[component]
fn GuideRow(
    index: usize,
    guide: GuideInfo,
    active: bool,
    motion: Motion,
    drag: Signal<Option<Grab>>,
    onland: EventHandler<usize>,
) -> Element {
    let state = use_context::<AppState>();
    let mut editing = use_signal(|| false);
    let id = guide.id;
    let label = guide_label(index, &guide);
    let seed = guide.name.as_deref().unwrap_or_default().to_string();
    let visible = guide.visible;
    // The row's transform, written by `Motion` so every declaration is stated on
    // every render — including the ones that are "off" (see `super::reorder::css`).
    let shift = super::reorder::css(motion);

    rsx! {
        div {
            class: "guide-row",
            class: if active { "active" },
            class: if motion.lifted { "dragging" },
            style: "{shift}",
            // Which row this element is, for `platform::guide_boxes` to read back.
            // A *position*, not the guide's id, and deliberately: what the drag
            // measures is drawn rows against each other, and `reorder` speaks in
            // rows. The landing is turned back into a guide by the one handler that
            // knows both (`GuidesPanel`'s `onland`).
            "data-guide": "{index}",
            if editing() {
                // Through the engine's name funnel like a layer's: trimmed, capped, and one
                // that comes out blank clears the name, so the row goes back to describing
                // its position (`normalize_name`). One logged action, so a mistyped rename
                // is undoable the way a mis-set opacity is (§20.5).
                InlineRename {
                    class: "guide-name guide-rename",
                    seed,
                    placeholder: label,
                    oncommit: move |text: String| {
                        dispatch(state, DocCommand::SetGuideName(id, Some(text)));
                    },
                    onclose: move |_| editing.set(false),
                }
            } else {
                // Selecting *is* picking the guide up to shape it, because shaping is all
                // there is to do to one. The first click of a rename's pair landing you in
                // the edit mode is no cost, since the guide you are renaming is the one
                // you were about to work on.
                Grip {
                    class: "guide-name",
                    title: "Shape this guide \u{2014} drag to reorder, double-click to rename",
                    row: RowKey::Guide(index),
                    drag,
                    onclick: move |_| begin_guide_edit(state, id),
                    ondoubleclick: move |_| editing.set(true),
                    onland: move |_| onland.call(index),
                    "{label}"
                }
            }
            // Duplicate, then Remove, then the eye — the order the Layers panel's rows
            // put them in, and the same three glyphs, because the two rosters differ in
            // what they list rather than in what these controls mean.
            button {
                class: "guide-duplicate",
                title: "Duplicate this guide",
                onclick: move |_| duplicate_guide(state, id),
                {icon(stark_ui::icons::DUPLICATE)}
            }
            // Remove then the eye, the order the Layers panel's rows put them in —
            // the two rosters answer the same question about different stacks, so a
            // hand that has learned one has learned the other. The ✕ this wore was
            // the one mark in either panel drawn as a character rather than a glyph;
            // it is `stark_ui::icons::REMOVE` in both places now.
            button {
                class: "guide-remove",
                title: "Remove this guide",
                onclick: move |_| remove_guide(state, id),
                {icon(stark_ui::icons::REMOVE)}
            }
            // The eye, and the one control on this row that is **not** a document
            // edit (§20.5): whether *you* are looking at a guide is not a fact about
            // the drawing, so it is a view command — unsaved, unsent, and not an undo
            // step. Everything else here — the name, the order, the camera, the
            // guide's existence — is logged.
            button {
                class: if visible { "guide-eye" } else { "guide-eye hidden" },
                title: if visible { "Hide this guide" } else { "Show this guide" },
                onclick: move |_| dispatch(state, ViewCommand::SetGuideVisible(id, !visible)),
                {icon(if visible { stark_ui::icons::VISIBLE } else { stark_ui::icons::HIDDEN })}
            }
        }
    }
}

/// The Perspective Guide bar (§20.5): the controls for the guide `edit` shapes, in the
/// same bottom column as the selection and transform bars.
#[component]
pub fn PerspectiveGuideBar(edit: GuideEdit) -> Element {
    let state = use_context::<AppState>();
    // What the bar's sliders are showing but have not laid down (§20.5). Bar-local,
    // and above the early return because a hook has to be: a bar that unmounts
    // mid-drag takes the pending value with it, and the mode's own exit drops the
    // preview it was showing (`end_guide_edit`).
    let pending = use_signal(|| None::<(GuideId, PerspectiveGuide)>);
    let guides = guides_of(state);
    let Some((index, row)) = guides
        .iter()
        .position(|g| g.id == edit.id)
        .map(|i| (i, &guides[i]))
    else {
        // The guide went away under the mode — a peer removed it, or an undo
        // crossed the add. Fold the bar rather than pointing it at nothing.
        //
        // A guide is *found* here rather than indexed, which is the whole of what
        // its having an id buys the mode: a removal or a reorder elsewhere in the
        // roster used to have to re-point this, and now moves nothing it holds.
        end_guide_edit(state);
        return rsx! {};
    };
    let id = edit.id;
    let g = &row.guide;
    // The bar names the guide the same way its row does, so a renamed guide is
    // called the same thing in both places.
    let name = guide_label(index, row);
    // The grid's scale is the *length* of the lattice: how many cells lie
    // between the eye and its corner, which is all a camera with no world scale
    // of its own can say about the size of a cell (§20.3). The bar states it in
    // halvings of the default ([`CELL_OCTAVES`]) — the guide's own length is the
    // ladder's rung, so there is no separate number to keep in step.
    let octave = stark_ui::guides::octave(g);
    let (opacity, pairs, lens) = (g.opacity, g.pairs, g.lens);
    let locked = edit.locked;

    rsx! {
        // No Cancel chip, alone among the mode bars, because it would be a lie here: a
        // guide is shaped live (§20.5), so there is nothing uncommitted for a cancel to
        // keep back — Esc and Done are one act, and the bar says so by offering it once.
        //
        // The words beside the Guides panel's mark are the guide's *name*: the glyph says
        // what kind of thing is being shaped, so the text is free to say which one. It
        // goes in minimal mode like every bar's, though a name is the artist's own — the
        // roster in the panel is where a name is kept, and this is only the indicator for
        // the guide already in hand.
        Bar {
            class: "guide-bar",
            glyph: stark_ui::icons::PERSPECTIVE_GRID,
            word: name,
            mode: true,

            span { class: "bar-sep" }

            // Locks: hold a world axis fixed, constraining the canvas drag to
            // turns about it — lock the vertical and every gesture keeps the
            // verticals parallel. Any may be held together, so they stand apart
            // rather than as a run (§25.9). Colored as the axis's own lines are.
            span { class: "bar-sub",
                {icon(stark_ui::icons::LOCK)}
                {label("Lock")}
            }
            div { class: "row",
                for (i, name) in AXIS_NAMES.into_iter().enumerate() {
                    Chip {
                        key: "{name}",
                        active: locked[i],
                        class: "axis-chip",
                        style: "--axis: {AXIS_CSS[i]}",
                        title: "Hold the {name} axis fixed under the drag",
                        onclick: move |_| {
                            // Read live rather than from the render's `edit`, and
                            // written through `advance`, the one writer that may not
                            // change *which* mode is live (`crate::modes`).
                            let live = crate::modes::composing_now(state);
                            let Some(mut edit) = live.and_then(Composing::guide_edit) else {
                                return;
                            };
                            edit.locked[i] = !edit.locked[i];
                            crate::modes::advance(state, Composing::GuideEdit(edit));
                        },
                        "{name}"
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
                {icon(stark_ui::icons::VISIBLE)}
                {label("Show")}
            }
            div { class: "row",
                for (k, [a, b]) in PAIR_AXES.into_iter().enumerate() {
                    // Each letter in its own axis's hue, so the chip names the plane by
                    // the two colors ruling it.
                    Chip {
                        key: "{k}",
                        active: pairs[k],
                        class: "plane-chip",
                        style: "--axis-a: {AXIS_CSS[a]}; --axis-b: {AXIS_CSS[b]}",
                        title: "Show the {AXIS_NAMES[a]}{AXIS_NAMES[b]} plane \u{2014} its two \
                                fans of guide lines, its horizon and its station point",
                        onclick: move |_| edit_guide(state, id, move |g| g.pairs[k] = !g.pairs[k]),
                        span { class: "ax-a", "{AXIS_NAMES[a]}" }
                        span { class: "ax-b", "{AXIS_NAMES[b]}" }
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
            Chip {
                active: lens == Lens::Fisheye,
                title: "Curvilinear (fisheye): a stereographic lens \u{2014} straight \
                        world lines bow into circles, and both poles of every axis \
                        come into view",
                onclick: move |_| edit_guide(state, id, |g| {
                    g.lens = match g.lens {
                        Lens::Rectilinear => Lens::Fisheye,
                        Lens::Fisheye => Lens::Rectilinear,
                    };
                }),
                {icon(stark_ui::icons::FISHEYE)}
                {label("Fisheye")}
            }
            span { class: "bar-sep" }
            // A fan of lines from a point, which is what this number counts: the guide's
            // fans are its parametrization (§20.5), so the mark is a picture of the thing
            // the slider makes more or fewer of. Stepped off the *default* rather than off
            // the guide's current lattice, so a rung is the same grid however it was
            // reached; where the corner sits is the drag's business, and there is no drag
            // for it yet (§20.5) — the grid stays hung on the viewer either way (§20.3).
            PreviewSlider {
                shape: SliderShape::Bar,
                label: "Cells",
                glyph: stark_ui::icons::DENSITY,
                min: CELL_OCTAVES.0 as f32,
                max: CELL_OCTAVES.1 as f32,
                step: 1.0,
                value: octave,
                title: "How fine the grid is \u{2014} each step halves the cell, so \
                        every line of the coarser grid is still a line of this one",
                preview: preview::GUIDE,
                pending,
                map: move |k: f32| {
                    camera_of(state, id).map(|g| (id, stark_ui::guides::with_octave(g, k)))
                },
            }
            // The ghost the Layers panel and the brush editor wear: how much of what
            // is under this shows through, asked of a guide over the paint.
            PreviewSlider {
                shape: SliderShape::Bar,
                label: "Opacity",
                glyph: stark_ui::icons::OPACITY,
                min: OPACITY_RANGE.0,
                max: OPACITY_RANGE.1,
                value: opacity,
                title: "How strongly the guide reads over the paint",
                preview: preview::GUIDE,
                pending,
                map: move |v: f32| {
                    camera_of(state, id).map(|g| (id, PerspectiveGuide { opacity: v, ..g }))
                },
            }
            span { class: "bar-sep" }
            ActChip {
                command: Command::FinishMode,
                title: "Leave the guide as it stands",
                onclick: move |_| end_guide_edit(state),
            }
        }
    }
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

/// The edit mode's catcher: a full-viewport surface that owns every pointer
/// event while a guide is being composed, so a stray drag cannot paint — but
/// navigation still works (middle-drag and space-drag pan, the wheel zooms;
/// see `input::Nav`). All gesture math is in canvas space, so panning or
/// zooming mid-drag cannot corrupt it.
#[component]
pub fn GuideEditOverlay(edit: GuideEdit) -> Element {
    let state = use_context::<AppState>();
    let mut drag = use_signal(|| None::<Drag>);
    let mut hover = use_signal(|| None::<GuideRegion>);
    // What the drag is showing but has not laid down (§20.5). A guide is document
    // state, so a canvas drag makes the bargain every continuous control in the app
    // makes: preview per sample, one logged action on release (`crate::preview`).
    let pending = use_signal(|| None::<(GuideId, PerspectiveGuide)>);
    let nav = Nav::use_nav(state);

    // The two things this overlay draws with, through **one** memo — the pair
    // moves together (a drag writes the guide, and the pose it is judged against
    // is the view it is drawn in), which is the case `state::use_obs` asks for a
    // tuple in. Unconditionally, ahead of the early return, like any `use_*`. The guide's
    // id holds for the overlay's life, since `ModeCatcher` keys it by the id.
    let GuideEdit { id, locked } = edit;
    let look = use_obs_opt(state, move |o| {
        // Found by id, not by index. The roster is read off the *previewed*
        // document, so mid-drag this is the pose under the hand — which is what the
        // hit test wants: a handle has to be where it is drawn.
        let o = o?;
        let g = o.guides.iter().find(|g| g.id == id)?;
        Some((o.view, g.guide))
    });

    let Some((view, guide)) = look() else {
        return rsx! {};
    };
    // The grabbable geometry, derived once and `Copy`, so the pointer handlers can
    // share the hit test.
    let handles = Handles::of(&guide);

    let classify = move |pc: Vec2| handles.at(pc, view.zoom);

    let mut follow = move |e: &Event<PointerData>| {
        if nav.advance(e) {
            return;
        }
        let pc = canvas_xy(view, e);
        let Some(d) = drag() else {
            hover.set(Some(classify(pc)));
            return;
        };
        match d.region {
            GuideRegion::Center => drag_guide(state, pending, id, move |g| {
                g.center = d.start.center + (pc - d.from);
            }),
            GuideRegion::Focal(factor) => drag_guide(state, pending, id, move |g| {
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
                drag_guide(state, pending, id, move |g| {
                    g.rotation = d.start.dragged(d.from, pc, held).rotation;
                })
            }
        }
    };
    let mut finish = move |e: &Event<PointerData>| {
        follow(e);
        // One logged action for the whole gesture, laying down exactly the pose the
        // canvas is showing (§20.5). After `follow`, so a release that also moved the
        // pointer commits where it ended rather than where the last move left it.
        settle_guide(state, pending);
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
                    // A second finger turns the drag into navigation (§18.1.7).
                    // What the drag had reached is *laid down* rather than dropped:
                    // the pose on the canvas is the one the artist was looking at
                    // when they reached for the second finger, and abandoning it
                    // would take the work back for navigating (§20.5).
                    settle_guide(state, pending);
                    drag.set(None);
                    return;
                }
                if e.trigger_button() != Some(MouseButton::Primary) {
                    return;
                }
                e.stop_propagation();
                crate::platform::capture_pointer(&e);
                let pc = canvas_xy(view, &e);
                drag.set(Some(Drag {
                    region: classify(pc),
                    from: pc,
                    start: guide,
                }));
            },
            onpointermove: move |e| follow(&e),
            onpointerup: move |e| if !nav.release(&e) { finish(&e) },
            onpointercancel: move |e| if !nav.release(&e) {
                // A cancel lays down what was reached, like a release: the browser
                // taking the gesture or a pen leaving the tablet is not the artist
                // asking for the pose back (§20.5). `settle` is idempotent, so a
                // cancel that follows a release costs nothing.
                settle_guide(state, pending);
                nav.stop();
                drag.set(None);
            },
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
    /// chip, a plane chip and the lines they govern are one color, or the
    /// controls stop looking like they belong to what they control.
    ///
    /// The two declarations cannot be merged: `guides.wesl` needs its colors as
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
    /// survive both roundings is the 8-bit color the screen shows, and
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
            // be the declared one — a typo'd `var()` is simply no color.
            let name = format!("var(--axis-{})", ["x", "y", "z"][i]);
            assert_eq!(AXIS_CSS[i], name, "the bar points at nothing");

            let got = stark_model::color::oklab_to_srgb([lab[0], lab[1], lab[2], 1.0]);
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
    /// That is the argument for stating a color in a space with a lightness
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
        let boxes: Vec<stark_ui::reorder::Extent> = (0..4)
            .map(|i| stark_ui::reorder::Extent {
                top: i as f32 * H,
                height: H,
            })
            .collect();
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
