//! The floating Layers panel: the layer tree, with per-layer opacity, visibility,
//! blend mode and clipping (§6 step 6a, §14.6).
//!
//! The tree is drawn the way clipping masks are drawn everywhere: **the base at the
//! bottom, what it carries indented above it**. That picture is already how a
//! painter reads a clipping group in Photoshop; here it is simply the truth, because
//! a group *is* the layer at its base (§14.2).
//!
//! The one thing a Photoshop refugee has to unlearn is that the indent means
//! clipping. Here indent means **membership** and the rail down the left of a row
//! means **clipping**, and they are drawn as different marks because they are
//! different facts — a layer can be in a group without being clipped to it, which is
//! a state Photoshop's panel cannot draw at all.
//!
//! Because that picture already says where every layer sits, the moves between those
//! places belong **on the rows** rather than in a pair of buttons above them: Carry at
//! the head of a row's line, Release standing in the indent that row's membership
//! opened, the fold triangle on the top edge it shares with what it carries. A
//! selection-scoped button has to name the layer it would act on and go inert when
//! there is none; a control drawn *in* the row has already named it, and simply is not
//! there when the move it makes has nowhere to go.
//!
//! Remove is there for the same reason, and it was the last header button to move: it
//! acted on "the selected layer" and had to grey out when removing that layer would
//! empty the document. On a row it names its own layer, and the row that would empty
//! the document simply has no Remove — which is also what makes the Guides panel's
//! rows and these ones one shape rather than two (`panels::guides`).
//!
//! And because the panel draws where every layer *is*, the way to put one somewhere
//! else is to drag it there. That gesture is one move, not three, for the reason the
//! model has one command: a drop lands in some stack, at some place in it (§14.8).
//! Carry and Release stay — they are the two moves worth having a one-click name for,
//! and they say what they do without being tried — but reordering *within* a stack
//! had no control at all before this, because it is the one move neither of them can
//! express. See [`landing`] for what a drop means and [`Motion`] for what a row does
//! about it.

use std::collections::HashSet;

use dioxus::prelude::*;

use crate::collab::css_color;
use crate::icons::{icon, label};
use crate::panels::filter::AddFilterButton;
use crate::panels::reorder::{Grip, RowKey};
use crate::preview;
use crate::state::{AppState, dispatch, use_obs};
use crate::widgets::{Chip, CommandButton, InlineRename, PreviewSlider, Select};
use stark_engine::LayerInfo;
use stark_engine::command::{DocCommand, PeerCommand};
use stark_model::document::{BlendMode, LayerId};
use stark_ui::collab::Peer;
use stark_ui::commands::Command;
use stark_ui::layer_tree::{
    self, BEND_HINT, INDENT, Row, bend_ends, blend_hint, clip_hint, landing, opacity_hint, rows,
};
use stark_ui::reorder::{Grab, Motion};

/// Add a paint layer where the artist is working (`Command::AddLayer`): into
/// the selected layer's own stack, above it — not always the document's,
/// because adding while working inside a group should land in that group,
/// which is where you are looking.
pub fn add_layer(state: AppState) {
    let (carrier, above) = state
        .obs
        .peek()
        .as_ref()
        .map(|o| {
            let selected = o.layers.iter().find(|l| l.id == o.active_layer);
            (selected.and_then(|l| l.carrier), selected.map(|l| l.id))
        })
        .unwrap_or((None, None));
    dispatch(state, DocCommand::AddLayer { carrier, above });
}

/// Dispatch `add` — a command that adds one layer — and select the layer it added, so a
/// new frame, backing or filter comes up with its bar and handles without a second click.
///
/// Found by comparing the roster before and after (`stark_ui::mint`) rather than by where
/// the layer should have landed; `dispatch` refreshes the projection before it returns, so
/// the second read already has it. [`add_layer`] needs none of this: the engine makes a
/// new paint layer the active one itself.
pub fn add_and_select(state: AppState, add: DocCommand) {
    let ids = |state: AppState| -> Vec<LayerId> {
        state
            .obs
            .peek()
            .as_ref()
            .map(|o| o.layers.iter().map(|l| l.id).collect())
            .unwrap_or_default()
    };
    let before = ids(state);
    dispatch(state, add);
    if let Some(id) = stark_ui::mint::minted(&before, ids(state)) {
        dispatch(state, PeerCommand::SetActiveLayer(id));
    }
}

#[component]
pub fn LayerPanel() -> Element {
    let state = use_context::<AppState>();
    // Which groups are shut. Panel-local view state — which is the whole point of
    // it not being in the document: whether *you* have a group folded away is not
    // part of the painting, is not saved, and is not something a collaborator
    // should see happen to their panel.
    let mut collapsed = use_signal(HashSet::<LayerId>::new);
    // The in-flight row drag, if any. Panel-local for the same reason `collapsed` is:
    // it exists only between a press and its release, is nobody else's business, and
    // — like the panel stack's — is delimited by the browser's own gesture, so it
    // cannot be left armed by a timer that failed to fire (§11).
    let drag = use_signal(|| None::<Grab>);

    // The tree and which row is selected, through **one** memo (`state::use_obs`).
    // Both move on a commit; nothing here has anything to say about a pan or a
    // stroke in flight. Read straight off `obs` this panel re-rendered on every
    // engine write, and each of those re-renders cloned the whole layer list — with
    // every layer's name — to redraw rows that had not changed.
    //
    // One memo rather than two because the pair is compared together: a selection
    // change moves `active_layer` while the list stands, and a commit usually moves
    // both, so splitting them would buy one extra comparison and no extra sleep.
    let tree = use_obs(state, |o| {
        (
            o.layers.clone(),
            o.layers.iter().find(|l| l.id == o.active_layer).cloned(),
        )
    });
    let (layers, selected) = tree().unwrap_or_default();
    // Keep the rows' pictures up to date (§14.6; `crate::layer_thumbs`).
    //
    // **Driven from the panel, not from the app root**, which is the opposite of the
    // brush thumbnails' arrangement and the opposite for a reason. Those have a second
    // viewer that appears only while a key is held, far too late to start rendering;
    // these have exactly one viewer, and it is this panel. Rendering a document's
    // worth of layers for a panel that is closed would be spending the canvas's own
    // engine borrow on pictures nobody has asked to see.
    //
    // On `layers` rather than on `doc_revision`: the memo above already collapses
    // every engine write down to "the tree or its tiles moved", which is exactly the
    // question, and `content_revision` is what carries a stroke into it. `refresh`
    // is idempotent and returns immediately when nothing is stale, so an effect that
    // fires for an unrelated change costs a scan of the list.
    use_effect(use_reactive!(|layers| {
        crate::layer_thumbs::prune(state, &layers);
        crate::layer_thumbs::refresh(state);
    }));
    let shut = collapsed.read().clone();
    let rows = rows(&layers, &shut);
    // The rows as the panel actually shows them (`layer_tree::display`): top of the
    // document first, with whatever is folded away left out. One list, used three
    // times — to draw, to resolve the drag against, and to say what a drop means — so
    // the gesture is reasoning about the same picture the user is looking at. The
    // turn itself is the tree's, because `landing` is written against it and a list
    // handed over unturned resolves to the mirror image of the right drop.
    let display = layer_tree::display(&rows);
    // The drag preview, resolved to numbers here rather than read by each row: the
    // rows that do not move do not re-render as the pointer travels, and the drop's
    // meaning is decided in one place instead of once per row.
    let land = drag
        .read()
        .as_ref()
        .filter(|d| d.live())
        .and_then(|d| landing(&display, d));

    // `LayerInfo` carries the layer's name now, so it is `Clone` rather than `Copy`
    // and cannot be read again after a handler has moved it. The id is all most
    // handlers here want, and it still copies.
    let selected_id = selected.as_ref().map(|l| l.id);

    // Whether the two relational controls have anything to say about the selected
    // layer. The row answers both (`layer_tree::Row::blend_inert`), and carries the
    // argument for why they part on a filter. Off `rows` rather than `display`, so a
    // selection folded away under a shut group still answers.
    let picked = selected_id.and_then(|id| rows.iter().find(|r| r.info.id == id));
    let blend_inert = picked.is_none_or(Row::blend_inert);
    let clip_inert = picked.is_none_or(Row::clip_inert);
    rsx! {
        if let Some(layer) = selected {
            SelectedLayerControls { layer, blend_inert, clip_inert }
        }

        hr {}

        div { class: "layer-header",
            // A frame is a layer, so making one belongs here rather than in a
            // panel of its own (§15.7). Both adds render their command whole
            // ([`add_layer`], `panels::frame::add_frame`).
            CommandButton { command: Command::AddLayer, class: "layer-add" }
            // No "+ Background" beside it: the substrate is made at most once per
            // painting, so it is a chip in the frame bar instead (§15.5) rather
            // than a button standing here for the rest of the session.
            // The third kind of layer (§21). Beside the other two rather than in a
            // menu of its own, because that is what it is: a filter is a layer, and
            // where it lands is the whole of what it acts on.
            AddFilterButton {}
            CommandButton { command: Command::AddFrame, class: "layer-add" }
        }

        // Top of the document first, which is what a stack looks like from in front
        // of it — and within a group, what it carries above its base. In a well of
        // its own (`.layer-tree`): the rows are the panel's one *picture* — the
        // stack, drawn — and the ground under a picture is not the ground the
        // controls stand on.
        div { class: "layer-tree",
            for (i, row) in display.iter().copied().enumerate() {
                LayerRow {
                    // Keyed by the layer, so a reorder *moves* the row's element instead
                    // of repainting whichever row now stands in that position. Positional
                    // diffing was harmless while the panel only ever grew and shrank; a
                    // drop reorders, and it would leave the click that follows the release
                    // landing on the row that took the dragged one's place.
                    key: "{row.info.id}",
                    row: row.clone(),
                    motion: land.map_or_else(Motion::default, |l| l.motion(i)),
                    // The one mark that is about a row *other* than the one moving: the
                    // layer that would carry the drop. Beside `motion` rather than in it
                    // because it is a fact about the landing's meaning, which is this
                    // panel's alone — a flat roster has no such row.
                    carrying: land.is_some_and(|l| l.carrier == Some(row.info.id)),
                    active: selected_id == Some(row.info.id),
                    drag,
                    ontoggle: move |id| {
                        let mut shut = collapsed.write();
                        if !shut.remove(&id) {
                            shut.insert(id);
                        }
                    },
                    onland: move |id: LayerId| {
                        // Dragging a layer selects it, drop or no drop: it is the one you
                        // just had in your hand. Said here rather than left to the click
                        // that follows, which this gesture has taken.
                        dispatch(state, PeerCommand::SetActiveLayer(id));
                        let Some(l) = land.filter(|l| !l.inert) else {
                            return;
                        };
                        // A layer dropped into a folded group would otherwise vanish into
                        // it. Opening the fold is not a second decision — it is the panel
                        // showing the move it just made.
                        if let Some(c) = l.carrier {
                            collapsed.write().remove(&c);
                        }
                        dispatch(state, l.move_layer(id));
                    },
                }
            }
        }
    }
}

/// The properties of **whichever layer is selected**, once, rather than repeated per row
/// and again in the frame bar — a frame is a layer, so it needs no copies of its own
/// (§15.7). A component for its two drags' pending values, which live exactly as long as
/// there is a selected layer to drag the properties of.
#[component]
fn SelectedLayerControls(layer: LayerInfo, blend_inert: bool, clip_inert: bool) -> Element {
    let state = use_context::<AppState>();
    // What an opacity drag and a Bend drag would lay. The whole mode for Bend rather
    // than the number, because that is what `SetLayerBlend` takes — a parameter alone
    // would have to be put back into a mode at commit time, off the very projection the
    // preview is feeding.
    let fading = use_signal(|| None::<(LayerId, f32)>);
    let bending = use_signal(|| None::<(LayerId, BlendMode)>);
    let id = layer.id;
    let clip = layer.clip;
    let modes: Vec<&'static str> = BlendMode::ALL.iter().map(|m| m.label()).collect();
    // `same_mode`, not `==`: the list is of modes at their default settings, and a
    // Radiance layer whose Bend has been dragged is still on the Radiance row. Under `==`
    // it would show no row selected at all — and picking one to fix that would reset the
    // very number the drag set.
    let mode = BlendMode::ALL.iter().position(|m| m.same_mode(layer.blend));
    rsx! {
        div { class: "slider-row marked",
            div { class: "slider-label",
                {icon(stark_ui::icons::BLEND)}
                {label(if layer.is_group { "Blend \u{2014} of the group" } else { "Blend" })}
            }
            // Blend and clip are one row because they are one question — *how does this
            // layer meet what is below it* — and they share the answer's two halves: the
            // mode says how the paint combines, the toggle says where it is allowed to
            // land. Both go inert together at the bottom of the document, which is the
            // other thing the shared row makes visible.
            //
            // On a **filter** the two halves come apart, and the row is where you can see
            // that they were always two: a filter has no source, so the mode has nothing
            // to describe and goes inert — but "where is this allowed to land" still has
            // an answer, and the chip stays live to give it (§21.4).
            div { class: "row blend-row",
                // The mode's own description, so the difference between the two light
                // modes is readable without painting a test stroke. Inert at the bottom
                // of the document, where every mode is the identity (§14.4.3) — shown
                // rather than hidden, since the control belongs to the layer wherever it
                // sits.
                Select {
                    title: blend_hint(layer.blend, &layer),
                    disabled: blend_inert,
                    options: modes,
                    selected: mode,
                    onchange: move |i: usize| {
                        dispatch(state, DocCommand::SetLayerBlend(id, BlendMode::ALL[i]));
                    },
                }
                // A lit chip rather than a tick-box and a sentence: the sentence is the
                // tooltip, and the glyph carries what the word "Clip" could not. Inert
                // only where there is nothing beneath — where a mode over nothing is
                // harmlessly the identity, a clip over nothing would erase the layer
                // (§14.4.3).
                Chip {
                    active: clip,
                    title: clip_hint(&layer),
                    disabled: clip_inert,
                    onclick: move |_| dispatch(state, DocCommand::SetLayerClip(id, !clip)),
                    {icon(stark_ui::icons::CLIP)}
                }
            }
        }
        // A filter's opacity is its **strength** (§21.4): "50% opacity" on a color
        // adjustment invites the reading that the filter is half transparent, when it is
        // half applied. The "— of the group" qualifier rides inside the hideable word: it
        // is the sentence saying what *this* opacity fades (§14.3), and half a sentence
        // left standing in minimal mode would read as a bug.
        PreviewSlider {
            label: match (layer.is_group, layer.filter.is_some()) {
                (true, _) => "Opacity \u{2014} of the group",
                (false, true) => "Strength",
                (false, false) => "Opacity",
            },
            glyph: stark_ui::icons::OPACITY,
            min: 0.0,
            max: 100.0,
            value: layer.opacity * 100.0,
            title: opacity_hint(&layer),
            preview: preview::LAYER_OPACITY,
            pending: fading,
            map: move |v: f32| Some((id, v / 100.0)),
        }
        // Radiance's own parameter — the first a mode has had (§18.0.4). The row is here
        // only while the mode is: a Bend on a Multiply layer would be a control for a
        // number that mode's curve has no place for, and the document could not hold the
        // setting it appeared to offer.
        if let BlendMode::Drago { k } = layer.blend {
            PreviewSlider {
                label: "Bend",
                glyph: stark_ui::icons::BEND,
                // In **octaves of `k`**, not in `k`. The bend is a scale, so what it does
                // to the curve is a matter of ratio: half of 0.2 is a different mode and
                // half of 3 is barely a change. A linear track would spend most of its
                // travel in the flat end.
                min: bend_ends().0,
                max: bend_ends().1,
                value: k.log2(),
                title: BEND_HINT,
                // Inert with its mode: a bend over nothing bends nothing.
                disabled: blend_inert,
                preview: preview::LAYER_BLEND,
                pending: bending,
                map: move |stops: f32| Some((id, BlendMode::Drago { k: stops.exp2() })),
            }
        }
    }
}

#[component]
pub fn LayerRow(
    row: Row,
    motion: Motion,
    carrying: bool,
    /// Whether this is the selected row. A prop rather than a read of the
    /// projection, for the reason `motion` and `carrying` are: the panel has
    /// already resolved it, and a row that asked for itself would subscribe every
    /// row in the tree to every engine write — one selection change re-rendering
    /// the whole list, and a stroke doing it per sample.
    active: bool,
    drag: Signal<Option<Grab>>,
    ontoggle: EventHandler<LayerId>,
    onland: EventHandler<LayerId>,
) -> Element {
    let state = use_context::<AppState>();
    let info = row.info.clone();
    // Whether this row's name is open for renaming — row-local, so opening one leaves
    // every other row alone.
    let mut editing = use_signal(|| false);
    let id = info.id;
    // The row's own fields, read out before the handlers below capture them:
    // `LayerInfo` is `Clone` rather than `Copy` now that it carries the name, and
    // several handlers want a piece of it.
    let visible = info.visible;
    let matte = info.matte.is_some();
    let filter = info.filter.is_some();
    let label = stark_ui::layer_tree::layer_label(&info);
    let seed = info.name.as_deref().unwrap_or_default().to_string();
    // This layer's own paint in miniature, or `None` for a row that has no picture
    // to show (§14.6). The two `None`s are deliberately different things and the
    // `map` is what keeps them apart: a layer with no *tiles* gets no thumbnail box
    // at all, while a paint layer whose *first* render has not landed yet gets an
    // empty one, so the row does not change shape underneath the pointer when the
    // image arrives. Only the first: a later render replaces the picture in place
    // rather than emptying the box while it works (§14.6). Subscribes, which is how
    // a row learns its picture is ready.
    //
    // The style is built here rather than in the markup because it is one: an `if`
    // inside an rsx attribute is a Rust expression, so `{url}` in its arms is literal
    // text rather than an interpolation — the row would ask the browser for a picture
    // called `{url}`. Every other thumbnail in the app is written this way for the
    // same reason (`panels::brush`, `slots`).
    let thumb = info.content_revision.map(|_| {
        match crate::layer_thumbs::url(state, &info) {
            // `none` written out rather than the property omitted: Dioxus merges
            // inline style per property, so a declaration simply left off is stranded
            // on a reused node and the row goes on showing the *previous* layer's
            // picture after a reorder.
            Some(url) if !url.is_empty() => format!("background-image: url({url});"),
            _ => "background-image: none;".to_string(),
        }
    });
    let indent = info.depth * INDENT;
    let is_group = info.is_group;
    let collapsed = row.collapsed;
    // The two moves the row can make of itself (§14.2). They were a pair of buttons
    // acting on "the selected layer"; here each acts on the row it is drawn in, which
    // is the layer being talked about anyway — and the row already knows both answers,
    // so neither has an inapplicable state to sit in.
    //
    // The whole command each, not the ingredients: §14.2's rule is the tree's to
    // spell, and it was spelled here and again in the native panel.
    let carry = row.carry();
    let release = row.release();
    let removable = row.removable;
    // The layer this one folds into, or `None` where no merge preserves the picture
    // (§14.11). Read straight off the projection rather than worked out here: whether a
    // pair composites as one layer is a question about blend modes, clipping and the
    // isolation each is stated against, and a second opinion in the panel is how a
    // button ends up offering an edit the engine then declines.
    let merge_down = info.merge_down;
    // Which layer it lands in is worth saying, because "down" is not always the row
    // below: the bottom member of a group folds into the layer carrying it (§14.1),
    // which the panel draws *under* the indent rather than directly beneath. Chosen
    // out here rather than inside the attribute so the branch is a plain `if` in
    // ordinary code, which is where a reader looks for one.
    //
    // A **filter** row says something else again, because what the click does there is
    // not to move paint but to bake an adjustment into the paint it was adjusting
    // (§14.11.7) — "merge this layer down" would describe a layer with nothing in it.
    let merge_title = if filter {
        "Bake this filter into the paint it is filtering \u{2014} the picture stays \
         the same, and the row goes"
    } else if merge_down.is_some() && merge_down == info.carrier {
        "Merge this layer into the one carrying it \u{2014} the picture stays the same"
    } else {
        "Merge this layer down into the one below \u{2014} the picture stays the same"
    };

    let title = if matte {
        "Compose this frame — double-click to rename"
    } else if filter {
        "Tune this filter — it adjusts everything below it in its own stack. \
         Double-click to rename"
    } else {
        "Paint on this layer — double-click to rename"
    };
    // The two kinds that are a *what* rather than a place to paint share one treatment:
    // a dimmed, un-pressable-looking name. The mark that used to lead it is in the row's
    // right-hand slot, with the thumbnails.
    let name_class = if matte || filter {
        "layer-name layer-name-kind"
    } else {
        "layer-name"
    };

    // A row is one line — Carry, the name that selects it, then Duplicate, Remove and
    // the eye hard against the right edge — with two marks outside that line: the group's
    // triangle straddling its top edge, and Release standing in the indent. The
    // per-layer opacity slider lives in the panel's single set of controls for
    // whatever is selected.
    // The row's transform, written by `Motion` so that every declaration is stated on
    // every render — including the ones that are "off", which is the whole of that
    // rule (see `super::reorder::css`).
    let shift = super::reorder::css(motion);

    rsx! {
        // The indent is padding on the wrapper rather than a margin on the row,
        // because the space it opens is not empty any more: Release is drawn in it.
        // Stated twice on purpose: as the padding that lays the row out, and as
        // `--indent` for the stylesheet's stratum band, which paints exactly the
        // gutter the padding opens (`.layer-item::before`) — one value, two hands.
        div {
            class: "layer-item",
            class: if motion.lifted { "dragging" },
            style: "--indent:{indent}px; padding-left:{indent}px; {shift}",
            // Which layer this element is, for `platform::layer_boxes` to read back.
            // A drag measures the DOM and then talks about rows, so the two have to
            // agree; this is what lets it match on identity rather than assume an
            // order the panel does not promise.
            "data-layer": "{id}",
            // Release, in the last step of the indent — the space this layer's own
            // membership carved out, which is the only place in the panel that means
            // "the group you are in" without a word. A layer in no group has no such
            // space, and needs no Release; the control cannot exist where it would be
            // inapplicable, rather than existing there greyed out. That is also what
            // makes the offset safe to subtract: `carrier` is `Some` exactly when
            // `depth` is at least one, both being read off the same walk in
            // `observe()`, so the button never asks for a step the indent has not got.
            if let Some(release) = release {
                button {
                    class: "layer-release",
                    style: "left:{indent - INDENT}px",
                    title: "Lift this layer out of its group",
                    onclick: move |_| dispatch(state, release.clone()),
                    {icon(stark_ui::icons::RELEASE)}
                }
            }
            div {
                class: "layer-row row",
                // Three kinds of row, and the two that are not paint wear their own
                // substrate: a frame is dashed (§15.7) and a filter is ruled (§21.6),
                // because in both cases "the brush has nowhere to go here" is the thing
                // to see before reaching for it. A filter's mark is a *line* rather than
                // a dash: a frame bounds the piece, and a filter runs across everything
                // under it.
                class: if matte { "matte" } else if filter { "filter" },
                // One selection, one highlight. A matte is selected exactly the way a
                // paint layer is (§15.7) — selecting it raises the frame bar and its
                // on-canvas handles, and the brush simply has nowhere to go until a
                // paint layer is selected again. Because there is only one thing to
                // highlight, "exactly one row is highlighted" is a consequence rather
                // than a rule to keep — and because `active` arrives as a prop resolved
                // once by the panel, it is a consequence of one comparison rather than
                // of one per row.
                class: if active { "active" },
                // Membership is an indent; clipping is a rail. Two marks, because they
                // are two facts (§14.6) — and a row can wear one without the other,
                // which is the state Photoshop's single arrow cannot express.
                class: if info.clip { "clipped" },
                // The layer that would carry what is being dropped. Marked while a drag
                // is over it because that is the one part of the landing the indent
                // leaves to be inferred — the seam says *where*, the block's own indent
                // says *how deep*, and this says *whose stack that depth is*.
                class: if carrying { "carrying" },
                // Only a group gets a triangle, and it sits centred on the row's top
                // edge, aimed at what it carries — which this panel draws *above* the
                // base (§14.6). Which way the caret points therefore says nothing; what
                // the two states differ by is a lid (see `stark_ui::icons::FOLD_OPEN`). It is out
                // of the line rather than in it because the line is full: the slot at
                // the head of the row is where Carry goes, and a mark about the rows
                // above belongs on the edge it shares with them.
                if is_group {
                    button {
                        class: "layer-fold",
                        title: if collapsed { "Show what this layer carries" }
                               else { "Fold away what this layer carries" },
                        onclick: move |_| ontoggle.call(id),
                        {icon(if collapsed { stark_ui::icons::FOLD_SHUT } else { stark_ui::icons::FOLD_OPEN })}
                    }
                }
                // Carry, at the head of the line: put this layer on the one below it in
                // its own stack, and the two become a group. There is no third command
                // — "clip to the layer below" is Carry followed by the Clip toggle,
                // because clipping to exactly one layer *is* that layer carrying this
                // one (§14.4). The space is held either way, so the names down the panel
                // still start in one column at each depth.
                //
                // Rests hidden and arrives with Release and the eye on hover, the three
                // together (`.layer-item:hover` in `stark.css`) — a move and its undo
                // should not be discovered one at a time. The glyph pair says the rest:
                // an elbow turning right here, the same elbow turning left out in the
                // indent, each drawn the way the row's own indent is about to move.
                // They are the only pair in the panel drawn as one picture mirrored,
                // which is what makes a move and its undo readable as such.
                if let Some(carry) = carry {
                    button {
                        class: "layer-carry",
                        title: "Put this layer on the one below it \u{2014} they become a group",
                        onclick: move |_| dispatch(state, carry.clone()),
                        {icon(stark_ui::icons::CARRY)}
                    }
                } else {
                    span { class: "layer-carry" }
                }
                if editing() {
                    // The engine drops a rename to the name the layer already has, so
                    // leaving an untouched field spends no undo step.
                    InlineRename {
                        class: "layer-name layer-rename",
                        seed,
                        placeholder: label.to_string(),
                        oncommit: move |text: String| {
                            dispatch(state, DocCommand::SetLayerName(id, Some(text)));
                        },
                        onclose: move |_| editing.set(false),
                    }
                } else {
                    Grip {
                        class: name_class,
                        title,
                        row: RowKey::Layer(id),
                        drag,
                        onclick: move |_| dispatch(state, PeerCommand::SetActiveLayer(id)),
                        ondoubleclick: move |_| editing.set(true),
                        onland: move |_| onland.call(id),
                        "{label}"
                    }
                }
                // Who else is working here (§17.4). The selected layer is
                // per-client, so this is the only place that answers "am I about to
                // paint over what someone else is doing?" before it happens.
                for peer in peers_on(state, id) {
                    div {
                        class: "peer-chip",
                        style: "background:{css_color(&peer)}",
                        title: "{peer.name} is working on this layer",
                        "{peer.initials()}"
                    }
                }
                // Merge down (§14.11): this layer's paint folded into the one beneath
                // it, and this row gone. On the row for the same reason Carry and
                // Duplicate are — it names its own layer, so there is no "the selected
                // layer" to read.
                //
                // **Absent rather than inert** where the pair cannot be merged, which
                // is the one place this panel departs from its own habit of greying a
                // control out. A merge that would change the picture is not a weaker
                // merge, it is a different edit — and a disabled button here would
                // invite the reading that the document is temporarily in the way, when
                // what is actually true is that these two layers do not describe one
                // layer. The engine answers the same question before it logs anything
                // (`LayerInfo::merge_down`), so the two cannot disagree.
                //
                // Its slot is held either way, like Carry's and Remove's: the eyes are
                // a column to glance down, and a row without a merge must not push its
                // neighbours sideways.
                if merge_down.is_some() {
                    button {
                        class: "layer-merge",
                        title: "{merge_title}",
                        onclick: move |_| dispatch(state, DocCommand::MergeLayerDown(id)),
                        {icon(stark_ui::icons::MERGE_DOWN)}
                    }
                } else {
                    span { class: "layer-merge" }
                }
                // Duplicate, beside Remove: a second copy of this layer directly above
                // it, carrying its tiles, its name and everything it carries
                // (§14.8). On the row for the same reason every other move is —
                // it names its own layer, so there is no "the selected layer" to
                // read and no inapplicable state to grey out. Unlike Remove it has
                // none: every row can be copied, including the last one standing.
                button {
                    class: "layer-duplicate",
                    title: if is_group { "Duplicate this layer, and everything it carries" }
                           else { "Duplicate this layer" },
                    onclick: move |_| dispatch(state, DocCommand::DuplicateLayer(id)),
                    {icon(stark_ui::icons::DUPLICATE)}
                }
                // Remove, next to last: the destructive control on the row it destroys.
                // It rests hidden and arrives on hover with Carry, Release and an open
                // eye, which is also the whole safety story — a control you have to
                // reach for is cheaper than a confirmation, and the history makes the
                // click undoable anyway (§5).
                //
                // Absent rather than inert on the row whose removal would empty the
                // document, on the same argument Release is: a control that cannot
                // apply here has nothing to say, and the last stack standing is
                // already legible as the last one. Its slot is still held, the way
                // Carry's is — the eyes are a column to glance down, and one row's
                // eye stepping right would cost exactly what that column buys.
                if removable {
                    button {
                        class: "layer-remove",
                        title: if is_group { "Remove this layer, and everything it carries" }
                               else { "Remove this layer" },
                        onclick: move |_| dispatch(state, DocCommand::RemoveLayer(id)),
                        {icon(stark_ui::icons::REMOVE)}
                    }
                } else {
                    span { class: "layer-remove" }
                }
                // The last *control* on the line, so the eyes stand in one column down the
                // whole panel however deep the tree goes: a row is indented from the left,
                // and its right edge is where the panel's is. The kind slot below sits
                // outboard of the column rather than in it — it is flush with the row's
                // edge and is not a control, so the eyes still line up against a fixed
                // rule. That column is the thing being bought — the
                // tick-boxes this replaces marched *rightwards* with the indent, so reading
                // "what is hidden?" off the panel meant reading every row rather than
                // glancing down an edge. It shows the eye the layer *is*, not the one
                // clicking would give you (see `stark_ui::icons::VISIBLE`).
                //
                // An open eye now rests hidden with Carry and Release, which is that same
                // argument taken one step: a layer you did not hide is showing, and the
                // legible row is already saying so. Leaving only the struck ones standing
                // turns the column from one to scan into one to glance at. Nothing is lost
                // that a hover does not give back, and the class is still on the button
                // either way, so the state is what the DOM says it is.
                button {
                    class: if visible { "layer-eye" } else { "layer-eye hidden" },
                    title: match (is_group, visible) {
                        (true, true) => "Hide this layer and what it carries",
                        (true, false) => "Show this layer and what it carries",
                        (false, true) => "Hide this layer",
                        (false, false) => "Show this layer",
                    },
                    onclick: move |_| dispatch(state, DocCommand::SetLayerVisible(id, !visible)),
                    {icon(if visible { stark_ui::icons::VISIBLE } else { stark_ui::icons::HIDDEN })}
                }
                // **What this layer is**, in one slot flush with the row's right edge and
                // as tall as the row (§14.6). Every row fills it and they all line up, so
                // the panel gains a second column to read down beside the eyes' — and the
                // three kinds of layer answer the same question in the same place, each in
                // the terms it has:
                //
                // - a **paint** layer shows its own paint (`crate::layer_thumbs`);
                // - a **frame** shows the crop marks, because its content is a rect and a
                //   color the row is already drawing — a picture of it would be a flat
                //   rectangle saying less than the mark does;
                // - a **filter** shows the funnel, because it has no content at all
                //   (§21.3) and an empty picture would say "blank" about a layer that is
                //   an operation.
                //
                // The marks led the name until the thumbnails arrived. They belong here
                // instead because the question they answer is the thumbnail's — what kind
                // of thing is this — and not the name's, which is what the author calls it.
                //
                // A `<div>` with a background rather than an `<img>`, and
                // `pointer-events: none` in the stylesheet, because **the whole row is the
                // grip**: a drag starts anywhere on it, and an element that took the press
                // would put a dead patch in the middle of the one gesture this panel is
                // built around.
                if let Some(style) = thumb {
                    div { class: "layer-thumb", style: "{style}" }
                } else if matte {
                    div { class: "layer-thumb layer-thumb-kind", {icon(stark_ui::icons::FRAME)} }
                } else if filter {
                    div { class: "layer-thumb layer-thumb-kind", {icon(stark_ui::icons::FILTER)} }
                }
            }
        }
    }
}

/// The collaborators whose selected layer is `id`.
fn peers_on(state: AppState, id: stark_model::document::LayerId) -> Vec<Peer> {
    state
        .collab
        .peers
        .read()
        .iter()
        .filter(|p| p.active_layer == id)
        .cloned()
        .collect()
}
