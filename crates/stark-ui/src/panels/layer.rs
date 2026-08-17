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
//! express. See [`landing`] for what a drop means and [`Motion`] for how it is drawn.

use std::collections::HashSet;

use dioxus::html::Key;
use dioxus::prelude::*;

use crate::icons::{self, icon, label};
use crate::panels::filter::AddFilterButton;
use crate::panels::frame::AddFrameButton;
use crate::panels::layer_tree::{INDENT, Row, landing, rows};
use crate::panels::reorder::{self, Grab, Motion};
use crate::platform::{capture_pointer, layer_boxes, select_all};
use crate::preview;
use crate::render::PeerInfo;
use crate::state::{AppState, dispatch, use_obs};
use stark_engine::LayerInfo;
use stark_engine::command::{DocCommand, PeerCommand};
use stark_model::document::LayerId;
use stark_model::document::{BlendMode, DRAGO_K_RANGE, Place};

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
    let mut drag = use_signal(|| None::<Grab>);
    // The opacity being previewed by a slider drag, if one is in flight — the drag's
    // own "there is something to commit", panel-local like `drag` and delimited by the
    // same browser gesture. It is the *value*, not a flag, so the commit says what the
    // last preview showed rather than reading it back off a projection that the
    // in-flight preview is itself feeding (§14.6).
    let fading = use_signal(|| None::<(LayerId, f32)>);
    // The blend mode being previewed by a Bend drag, on `fading`'s pattern and for its
    // reasons. The whole mode rather than the number, because that is what
    // `SetLayerBlend` takes — a parameter alone would have to be put back into a mode
    // at commit time, off the very projection the preview is feeding.
    let bending = use_signal(|| None::<(LayerId, BlendMode)>);

    // The tree and which row is selected, through **one** memo (`state::use_obs`).
    // Both move on a commit; nothing here has anything to say about a pan or a
    // stroke in flight. Read straight off `obs` this panel re-rendered on every
    // engine write, and each of those re-renders cloned the whole layer list — with
    // every layer's name — to redraw rows that had not changed.
    //
    // One memo rather than two because the pair is compared together: a selection
    // change moves `active_layer` while the list stands, and a commit usually moves
    // both, so splitting them would buy one extra comparison and no extra sleep.
    //
    // The properties that belong to *whichever* layer is selected live here, once,
    // rather than being repeated per row and again in the frame bar. A frame is a
    // layer, so it needs no copies of its own (§15.7).
    let tree = use_obs(state, |o| {
        (
            o.layers.clone(),
            o.layers.iter().find(|l| l.id == o.active_layer).cloned(),
        )
    });
    let (layers, selected) = tree().unwrap_or_default();
    let shut = collapsed.read().clone();
    let rows = rows(&layers, &shut);
    // The rows as the panel actually shows them: top of the document first, with
    // whatever is folded away left out. One list, used three times — to draw, to
    // resolve the drag against, and to say what a drop means — so the gesture is
    // reasoning about the same picture the user is looking at.
    let display: Vec<Row> = rows.iter().rev().filter(|r| !r.hidden).cloned().collect();
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
    // layer. **They part on a filter**, which is the one row where a shared condition
    // for the two would be wrong (§21.4): a mode describes how a *source*
    // meets a backdrop and a filter has no source, while a clip says where the layer
    // may land — a question a filter still answers, by being confined to the coverage
    // it read. Both go inert with nothing beneath them (§14.4.3), which is the half
    // they do share.
    //
    // And they read *different predicates* for that half, which is the second thing
    // the split brings out. A blend is positional, so it takes `has_backdrop`. A
    // filter's clip is inert exactly where the **filter** is, so it takes
    // `has_underlay` — the renderer's own answer (§21.2), which counts a carrier's
    // base as beneath what it carries. `has_backdrop` would say no there, and that
    // arrangement is "filter just this layer": the chip would be dead in the one
    // place it is reached for most.
    let blend_inert = selected
        .as_ref()
        .is_none_or(|l| !l.has_backdrop || l.filter.is_some());
    let clip_inert = selected.as_ref().is_none_or(|l| match l.filter {
        Some(_) => !l.has_underlay,
        None => !l.has_backdrop,
    });
    // Where "Add layer" puts one: into the selected layer's own stack, above it.
    // Read out here rather than in the handler because the row block below consumes
    // `selected`, and this is the only part of it the handler wants.
    let add_at = (
        selected.as_ref().map(|l| l.carrier).unwrap_or(None),
        selected_id,
    );

    rsx! {
        div { class: "layer-header",
            // A frame is a layer, so making one belongs here rather than in a
            // panel of its own (§15.7).
            button {
                class: "layer-add",
                title: "Add a paint layer above the selected one",
                // Into the selected layer's own stack, not always the document's:
                // adding a layer while working inside a group should land in that
                // group, which is where you are looking.
                onclick: move |_| {
                    let (carrier, above) = add_at;
                    dispatch(state, DocCommand::AddLayer { carrier, above });
                },
                {icon(icons::ADD_LAYER)}
                {label("Layer")}
            }
            // No "+ Background" beside it: the ground is made at most once per
            // painting, so it is a chip in the frame bar instead (§15.5) rather
            // than a button standing here for the rest of the session.
            // The third kind of layer (§21). Beside the other two rather than in a
            // menu of its own, because that is what it is: a filter is a layer, and
            // where it lands is the whole of what it acts on.
            AddFilterButton {}
            AddFrameButton {}
        }

        // Top of the document first, which is what a stack looks like from in front
        // of it — and within a group, what it carries above its base.
        for (i, row) in display.iter().enumerate() {
            LayerRow {
                // Keyed by the layer, so a reorder *moves* the row's element instead
                // of repainting whichever row now stands in that position. Positional
                // diffing was harmless while the panel only ever grew and shrank; a
                // drop reorders, and it would leave the click that follows the release
                // landing on the row that took the dragged one's place.
                key: "{row.info.id.0}",
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
                    // A press that never travelled is a click, and the browser is
                    // about to send one; nothing here has anything to say about it.
                    if drag.peek().as_ref().is_none_or(|d| !d.live()) {
                        drag.set(None);
                        return;
                    }
                    // **The disarm goes first**, and for the reason the panel stack's
                    // does: a row's shift is stated against the panel as it stood when
                    // the press landed, so a frame carrying the new order while the
                    // transforms are still on would be the move applied twice. It is
                    // *spent* rather than dropped so the click behind the release can
                    // be recognized and swallowed (`reorder::claimed`) — on a panel
                    // that has just reordered, that click names whichever row took
                    // this one's place.
                    if let Some(d) = drag.write().as_mut() {
                        d.spend();
                    }
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
                    dispatch(state, DocCommand::MoveLayer {
                        id,
                        carrier: l.carrier,
                        at: l.at,
                    });
                },
            }
        }

        hr {}

        if let Some(l) = selected {
            // `marked`, as `widgets::Slider` sets it on the rows it builds: these two are
            // hand-rolled (one splits its samples between a preview and one commit, the
            // other holds a picker and a chip rather than a track), but they wear a glyph,
            // so they fold onto one line in minimal mode exactly as the component's rows do.
            div { class: "slider-row marked",
                // The "— of the group" qualifier rides inside the hideable word rather
                // than beside it. It is not a second fact about the control; it is the
                // sentence saying what *this* opacity fades (§14.3), and half a
                // sentence left standing in minimal mode would read as a bug.
                div { class: "slider-label",
                    {icon(icons::OPACITY)}
                    // A filter's opacity is its **strength** (§21.4), and the word is
                    // worth changing: "50% opacity" on a color adjustment invites
                    // the reading that the filter is half transparent, when what it
                    // is is half applied.
                    {label(match (l.is_group, l.filter.is_some()) {
                        (true, _) => "Opacity \u{2014} of the group",
                        (false, true) => "Strength",
                        (false, false) => "Opacity",
                    })}
                }
                input {
                    class: "slider",
                    r#type: "range", min: "0", max: "100", step: "any",
                    value: "{(l.opacity * 100.0) as i32}",
                    title: "{opacity_hint(&l)}",
                    // Previewed per sample, committed once when the drag settles: a
                    // layer's opacity is document state, so one adjustment must cost
                    // one undo step — and one replicated action — rather than one per
                    // pointer move, which is the bargain the frame drag and the canvas
                    // color already make (§14.6). The engine renders the preview and
                    // reports it back through `observe`, so the track and the canvas
                    // both follow the pointer.
                    oninput: move |e| {
                        if let Ok(v) = e.value().parse::<f32>() {
                            preview::LAYER_OPACITY.during(state, fading, (l.id, v / 100.0));
                        }
                    },
                    // Three ways to end, because a range control has three — see
                    // `Preview::settle`, which holds the why (and is idempotent, so
                    // arriving twice is free).
                    onchange: move |_| preview::LAYER_OPACITY.settle(state, fading),
                    onpointerup: move |_| preview::LAYER_OPACITY.settle(state, fading),
                    onpointercancel: move |_| preview::LAYER_OPACITY.settle(state, fading),
                }
            }
            div { class: "slider-row marked",
                div { class: "slider-label",
                    {icon(icons::BLEND)}
                    {label(if l.is_group { "Blend \u{2014} of the group" } else { "Blend" })}
                }
                // Blend and clip are one row because they are one question — *how does
                // this layer meet what is below it* — and they share the answer's two
                // halves: the mode says how the paint combines, the toggle says where it
                // is allowed to land. Both go inert together at the bottom of the
                // document, which is the other thing the shared row makes visible.
                //
                // On a **filter** the two halves come apart, and the row is where you
                // can see that they were always two: a filter has no source, so the
                // mode has nothing to describe and goes inert — but "where is this
                // allowed to land" still has an answer, and the chip stays live to
                // give it (§21.4).
                div { class: "row blend-row",
                    select {
                        class: "select",
                        // The mode's own description, so the difference between the two
                        // light modes is readable without painting a test stroke.
                        title: "{blend_hint(l.blend, &l)}",
                        // Inert at the bottom of the document, where there is nothing to
                        // blend with and every mode is the identity
                        // (§14.4.3). Shown rather than hidden: the control belongs to the
                        // layer wherever it sits, and a row that loses a control when it
                        // is dragged to the bottom reads as a bug.
                        disabled: blend_inert,
                        onchange: move |e| {
                            if let Some(m) = BlendMode::ALL.iter().find(|m| m.label() == e.value()) {
                                dispatch(state, DocCommand::SetLayerBlend(l.id, *m));
                            }
                        },
                        for mode in BlendMode::ALL {
                            option {
                                value: "{mode.label()}",
                                // `same_mode`, not `==`: the list is of modes at their
                                // default settings, and a Radiance layer whose Bend has
                                // been dragged is still on the Radiance row. Under `==`
                                // it would show no row selected at all — and picking one
                                // to fix that would reset the very number the drag set.
                                selected: mode.same_mode(l.blend),
                                "{mode.label()}"
                            }
                        }
                    }
                    // A lit chip rather than a tick-box and a sentence. The sentence was
                    // there because the *word* "Clip" is the thing nobody guesses the
                    // meaning of — but a sentence is not a label, it is a tooltip that
                    // had been promoted into the panel, and it cost the control a row of
                    // its own. It is a tooltip again here, and the glyph carries what a
                    // one-word label could not.
                    button {
                        class: if l.clip { "chip active" } else { "chip" },
                        title: "{clip_hint(&l)}",
                        // Inert only where there is nothing beneath — and where a mode
                        // over nothing is harmlessly the identity, a clip over nothing
                        // would erase the layer, which is the whole reason this one has
                        // to be stopped rather than merely left to do nothing
                        // (§14.4.3).
                        disabled: clip_inert,
                        onclick: move |_| dispatch(state, DocCommand::SetLayerClip(l.id, !l.clip)),
                        {icon(icons::CLIP)}
                    }
                }
            }
            // Radiance's own parameter — the first a mode has had (§18.0.4). The row
            // is here only while the mode is: a Bend on a Multiply layer would be a
            // control for a number that mode's curve has no place for, and the
            // document could not hold the setting it appeared to offer. That is the
            // same argument that put `k` on the variant rather than beside it, read
            // out into the panel.
            if let BlendMode::Drago { k } = l.blend {
                div { class: "slider-row marked",
                    div { class: "slider-label",
                        {icon(icons::BEND)}
                        {label("Bend")}
                    }
                    input {
                        class: "slider",
                        r#type: "range", step: "any",
                        // In **octaves of `k`**, not in `k`. The bend is a scale, so
                        // what it does to the curve is a matter of ratio: half of 0.2
                        // is a different mode and half of 3 is barely a change. A
                        // linear track would spend most of its travel in the flat end
                        // and cross the whole interesting range in its first few px.
                        min: "{bend_ends().0}", max: "{bend_ends().1}",
                        // The document's own value, which during a drag is the
                        // preview's — the engine renders it and reports it back
                        // through `observe`, so the track and the canvas follow the
                        // pointer together, exactly as the opacity slider above does.
                        value: "{k.log2()}",
                        title: "{BEND_HINT}",
                        // Inert with its mode: a bend over nothing bends nothing.
                        disabled: blend_inert,
                        // Previewed per sample, committed once when the drag settles —
                        // the same bargain, through the same pair. The whole mode
                        // travels rather than the number, because that is what both
                        // ends of the bargain take.
                        oninput: move |e| {
                            if let Ok(stops) = e.value().parse::<f32>() {
                                let next = BlendMode::Drago { k: stops.exp2() };
                                preview::LAYER_BLEND.during(state, bending, (l.id, next));
                            }
                        },
                        onchange: move |_| preview::LAYER_BLEND.settle(state, bending),
                        onpointerup: move |_| preview::LAYER_BLEND.settle(state, bending),
                        onpointercancel: move |_| preview::LAYER_BLEND.settle(state, bending),
                    }
                }
            }
        }
    }
}

/// What a blend mode does, in one line, for the picker's tooltip.
///
/// Here rather than beside [`BlendMode`] for the same reason [`layer_label`] is: the
/// mode's *name* is part of what it is and travels with the document, but how you
/// explain it to someone hovering a drop-down is a frontend's business. The core
/// says "Glow"; deciding that a painter wants to hear "cannot blow out" rather than
/// "conjugate of addition under `x/(1+x)`" is a presentation call.
fn blend_hint(mode: BlendMode, layer: &LayerInfo) -> &'static str {
    // The two cases where the control is not saying what it usually says come
    // first, because they are about *this row* rather than about the mode.
    if !layer.has_backdrop {
        return "Nothing composites under this layer, so every mode looks the same here.";
    }
    if layer.is_group {
        return match mode {
            BlendMode::Normal => "This group sits on top of what is below it.",
            _ => {
                "How this group \u{2014} everything it carries, composited \u{2014} \
                  meets what is below it."
            }
        };
    }
    match mode {
        BlendMode::Normal => "The layer sits on top of what is below it.",
        BlendMode::Reinhard => {
            "Combines light instead of covering it \u{2014} softer than Screen, and it \
             cannot blow out however deep you stack it. For glazes, mist and rim light."
        }
        BlendMode::Drago { .. } => {
            "Combines light on a log curve \u{2014} hotter, and where two lights coincide \
             it pushes past white into the highlight roll-off. For flame and speculars."
        }
        BlendMode::Multiply => {
            "Takes light away instead of adding it, the way stacked glazes do \u{2014} \
             white leaves the layer below alone, black hides it. For shadows and tinting."
        }
    }
}

/// What the Bend slider does, in one line — a frontend's business for
/// [`blend_hint`]'s reason.
///
/// Written as the two ends rather than as "the curve's `k`", because the ends are the
/// part a painter can act on: one of them is the mode a painter reaches for when a
/// specular should read as *hot*, and the other is what to pull back to when it has
/// started eating the drawing underneath.
const BEND_HINT: &str = "How hard Radiance's curve bends. Left, coincident lights \
                         barely add and the brighter one simply wins; right, they add \
                         outright and reach the highlight roll-off sooner.";

/// The Bend slider's ends, in **octaves** of the mode's `k` — [`DRAGO_K_RANGE`] read
/// in the unit the track travels in (see the row for why that unit).
fn bend_ends() -> (f32, f32) {
    (DRAGO_K_RANGE.0.log2(), DRAGO_K_RANGE.1.log2())
}

/// What the opacity slider fades, in one line.
///
/// Three answers, and the first is the one worth having: on a group, opacity is the
/// property that could *not* be borrowed from the base the way blend and clip are
/// (§14.3), so it fades the base and everything it carries as one unit.
fn opacity_hint(layer: &LayerInfo) -> &'static str {
    if layer.is_group {
        "Fades this layer and everything it carries, as one"
    } else if layer.filter.is_some() {
        "How much of the adjustment lands \u{2014} at 0 the filter is the identity"
    } else if layer.is_paintable() {
        "Opacity of the selected layer"
    } else {
        "Frame opacity \u{2014} drag down to see through it while composing"
    }
}

/// What clipping would do to *this* layer, in one line.
///
/// Three different sentences, because the control means three different things
/// depending on where the row sits — and the difference is the part users get wrong
/// everywhere else (§14.4).
fn clip_hint(layer: &LayerInfo) -> &'static str {
    if !layer.has_backdrop {
        return "Nothing composites under this layer, so clipping it would leave nothing \
                to show.";
    }
    // A filter's clip is about **where its result is allowed to land** rather than
    // where the layer shows, because a filter has nothing of its own to show
    // (§21.4). Worth its own sentence for exactly that reason: the word is the same
    // one every other row wears, and what it bounds here is a fringe rather than
    // paint.
    if layer.filter.is_some() {
        return "Clip: keep this filter inside the paint it is filtering \u{2014} it may \
                change the color that is there, never spread past its edge.";
    }
    if layer.is_group {
        return "Clip: this group shows only where there is paint under the group.";
    }
    match layer.carrier {
        // Inside a group the bound is the group, which is the whole reason groups
        // and clipping are one feature rather than two.
        Some(_) => {
            "Clip: show only where there is paint under this layer *within its group* \
             \u{2014} the whole stack below it, not just the one layer."
        }
        None => {
            "Clip: show only where there is paint under this layer. To clip to one \
             layer alone, Carry it onto that layer first."
        }
    }
}

/// What to call a layer that has never been named: its place in the stack, or what
/// it *is* when that says more (§15.7 — there is only ever one frame,
/// so numbering it would be noise).
///
/// Kept here rather than in the core because it is a way of *presenting* a stack,
/// not a fact about the document — which is exactly why an unnamed layer stores no
/// name (see [`LayerInfo::name`]). A layer the author has named shows that name,
/// frame or not.
///
/// The word alone, with no mark in it. A `\u{25F1}` leading an unnamed frame's label
/// would stand in for a glyph the set already has (`icons::FRAME`, which the frame bar
/// wears), and putting it in the *string* costs twice over: this label is also the
/// rename field's placeholder, so opening the field on a frame would show a corner mark
/// inside a text box. The row draws the glyph, which leaves the placeholder a name.
fn layer_label(info: &LayerInfo) -> String {
    match (&info.name, info.matte.as_ref(), info.filter.as_ref()) {
        (Some(name), ..) => name.to_string(),
        // The two kinds of matte, told apart by the one thing that differs:
        // a frame is defined against a rect, a background against none (§15.5).
        (None, Some(m), _) if m.rect.is_some() => "Frame".to_string(),
        (None, Some(_), _) => "Background".to_string(),
        // The *filter's* own name rather than the word "Filter" (§21.6): unlike a
        // frame, of which there is only ever one kind, which filter this is is the
        // first thing to know about the row — and a stack of three rows all reading
        // "Filter" would say nothing at all.
        (None, _, Some(f)) => f.label().to_string(),
        (None, None, None) => format!("Layer {}", info.id.ordinal()),
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
    // The rename in progress on *this* row, or `None` while the row is just a row.
    // Row-local, so opening one leaves every other row alone and closing it needs
    // nothing cleaned up. The draft is held here rather than read back off the
    // field on commit because both commit paths — Enter and blur — need it, and one
    // of them fires while the field is on its way out.
    let mut draft = use_signal(|| None::<String>);
    let id = info.id;
    // Commit whatever the field holds, and close it. `take` is what makes the two
    // commit paths safe to both fire: whichever runs second finds no draft. Leaving
    // an untouched field costs nothing either — the engine drops a rename to the
    // name the layer already has, so no undo step is spent on it.
    let mut commit = move || {
        let text = draft.write().take();
        if let Some(text) = text {
            dispatch(state, DocCommand::SetLayerName(id, Some(text)));
        }
    };
    // The row's own fields, read out before the handlers below capture them:
    // `LayerInfo` is `Clone` rather than `Copy` now that it carries the name, and
    // several handlers want a piece of it.
    let visible = info.visible;
    let matte = info.matte.is_some();
    let filter = info.filter.is_some();
    let label = layer_label(&info);
    // What the field opens on: the layer's *name*, which for one that has never been
    // named is empty. Deliberately not the label — seeding with the generated
    // "Layer 3" would turn opening the field and pressing Enter into a rename to
    // "Layer 3", quietly making a description into a name. The placeholder carries
    // the label instead, so the row still says what it is called while empty.
    let seed = info.name.as_deref().unwrap_or_default().to_string();
    // One selection, one highlight. A matte is selected exactly the way a paint
    // layer is (§15.7) — selecting it raises the frame bar and its
    // on-canvas handles, and the brush simply has nowhere to go until a paint layer
    // is selected again. Because there is only one thing to highlight, "exactly one
    // row is highlighted" is a consequence rather than a rule to keep — and because
    // `active` arrives as a prop resolved once by the panel, it is a consequence of
    // one comparison rather than of one per row.
    //
    // Three kinds of row, and the two that are not paint wear their own ground: a
    // frame is dashed (§15.7) and a filter is ruled (§21.6), because in both cases
    // "the brush has nowhere to go here" is the thing to see before reaching for it.
    // A filter's mark is a *line* rather than a dash: a frame bounds the piece, and a
    // filter runs across everything under it.
    let kind = if matte {
        " matte"
    } else if filter {
        " filter"
    } else {
        ""
    };
    let row_class = format!("layer-row{kind}{}", if active { " active" } else { "" });
    // Membership is an indent; clipping is a rail. Two marks, because they are two
    // facts (§14.6) — and a row can wear one without the other, which
    // is the state Photoshop's single arrow cannot express.
    let mut row_class = if info.clip {
        format!("{row_class} clipped")
    } else {
        row_class
    };
    // The layer that would carry what is being dropped. Marked while a drag is over
    // it because that is the one part of the landing the indent leaves to be inferred
    // — the seam says *where*, the block's own indent says *how deep*, and this says
    // *whose stack that depth is*.
    if carrying {
        row_class.push_str(" carrying");
    }
    let indent = info.depth * INDENT;
    let is_group = info.is_group;
    let collapsed = row.collapsed;
    // The two moves the row can make of itself (§14.2). They were a pair of buttons
    // acting on "the selected layer"; here each acts on the row it is drawn in, which
    // is the layer being talked about anyway — and the row already knows both answers,
    // so neither has an inapplicable state to sit in.
    let carry_onto = row.carry_onto;
    let release_to = row.release_to;
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

    // A row is one line — Carry, the name that selects it, then Duplicate, Remove and
    // the eye hard against the right edge — with two marks outside that line: the group's
    // triangle straddling its top edge, and Release standing in the indent. The
    // per-layer opacity slider lives in the panel's single set of controls for
    // whatever is selected.
    // The row's transform, written by `Motion` so that every declaration is stated on
    // every render — including the ones that are "off", which is the whole of that
    // rule (see `reorder::Motion::css`).
    let item_class = format!("layer-item{}", motion.class());
    let shift = motion.css();

    rsx! {
        // The indent is padding on the wrapper rather than a margin on the row,
        // because the space it opens is not empty any more: Release is drawn in it.
        div {
            class: "{item_class}",
            style: "padding-left:{indent}px; {shift}",
            // Which layer this element is, for `platform::layer_boxes` to read back.
            // A drag measures the DOM and then talks about rows, so the two have to
            // agree; this is what lets it match on identity rather than assume an
            // order the panel does not promise.
            "data-layer": "{id.0}",
            // Release, in the last step of the indent — the space this layer's own
            // membership carved out, which is the only place in the panel that means
            // "the group you are in" without a word. A layer in no group has no such
            // space, and needs no Release; the control cannot exist where it would be
            // inapplicable, rather than existing there greyed out. That is also what
            // makes the offset safe to subtract: `carrier` is `Some` exactly when
            // `depth` is at least one, both being read off the same walk in
            // `observe()`, so the button never asks for a step the indent has not got.
            if let Some((group, outer)) = release_to {
                button {
                    class: "layer-release",
                    style: "left:{indent - INDENT}px",
                    title: "Lift this layer out of its group",
                    onclick: move |_| {
                        dispatch(state, DocCommand::MoveLayer {
                            id,
                            carrier: outer,
                            at: Place::Above(group),
                        });
                    },
                    {icon(icons::RELEASE)}
                }
            }
            div {
                class: "{row_class} row",
                // Only a group gets a triangle, and it sits centred on the row's top
                // edge, aimed at what it carries — which this panel draws *above* the
                // base (§14.6). Which way the caret points therefore says nothing; what
                // the two states differ by is a lid (see `icons::FOLD_OPEN`). It is out
                // of the line rather than in it because the line is full: the slot at
                // the head of the row is where Carry goes, and a mark about the rows
                // above belongs on the edge it shares with them.
                if is_group {
                    button {
                        class: "layer-fold",
                        title: if collapsed { "Show what this layer carries" }
                               else { "Fold away what this layer carries" },
                        onclick: move |_| ontoggle.call(id),
                        {icon(if collapsed { icons::FOLD_SHUT } else { icons::FOLD_OPEN })}
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
                if let Some(onto) = carry_onto {
                    button {
                        class: "layer-carry",
                        title: "Put this layer on the one below it \u{2014} they become a group",
                        onclick: move |_| {
                            dispatch(state, DocCommand::MoveLayer {
                                id,
                                carrier: Some(onto),
                                at: Place::Top,
                            });
                        },
                        {icon(icons::CARRY)}
                    }
                } else {
                    span { class: "layer-carry" }
                }
                if let Some(text) = draft() {
                    input {
                        class: "layer-name",
                        class: "layer-rename",
                        r#type: "text",
                        value: "{text}",
                        placeholder: "{label}",
                        // The field is the point of the double-click, so it takes focus
                        // as it appears rather than asking for a second click. The DOM
                        // node exists by the time `onmounted` runs, which is what the
                        // `autofocus` attribute cannot promise for an element inserted
                        // after load.
                        onmounted: move |e: Event<MountedData>| {
                            spawn(async move {
                                let _ = e.set_focus(true).await;
                                // Selected, not merely focused: the field opens on the
                                // name the layer already has, and the usual reason to
                                // open it is to replace that name rather than add to it.
                                // Typing over is one keystroke; keeping a word of it is
                                // one click. Ordered after the focus rather than left to
                                // `select`'s own — awaiting it is what puts the two in a
                                // known order.
                                select_all(&e);
                            });
                        },
                        oninput: move |e| draft.set(Some(e.value())),
                        // Committing on blur is what makes this feel like a label rather
                        // than a form: clicking away is an ordinary way to be finished,
                        // and nothing is lost by it.
                        //
                        // Enter commits directly rather than by blurring — a focused
                        // element that is removed does not reliably fire `blur` (the very
                        // thing `platform::on_window_key` exists to work around), so the
                        // field closing itself cannot be the commit. The two paths cannot
                        // double up: `commit` *takes* the draft, so whichever runs second
                        // finds nothing to send.
                        onblur: move |_| commit(),
                        // Everything else typed here is left alone: the global shortcuts
                        // already stand aside for a text field (`input::bind_shortcuts`),
                        // which is what leaves the browser's own Ctrl+Z editing this text
                        // instead of the document.
                        onkeydown: move |e| match e.key() {
                            Key::Enter => commit(),
                            // Escape abandons the edit — dropping the draft first, so the
                            // blur that follows the field's removal has nothing left to
                            // commit.
                            Key::Escape => draft.set(None),
                            _ => {}
                        },
                    }
                } else {
                    button {
                        // The two kinds that are a *what* rather than a place to
                        // paint share one treatment: a glyph leading a dimmed,
                        // un-pressable-looking name.
                        class: if matte || filter { "layer-name layer-name-kind" } else { "layer-name" },
                        title,
                        // The click a drag leaves behind is not this row's — the
                        // drop has already said which layer is selected, and on a
                        // panel that reordered under the release this click names
                        // whichever row took the dragged one's place.
                        onclick: move |_| {
                            if !reorder::claimed(&mut drag) {
                                dispatch(state, PeerCommand::SetActiveLayer(id));
                            }
                        },
                        ondoubleclick: move |_| draft.set(Some(seed.clone())),
                        // The name **is** the grip, as the panel's title is
                        // (`layout::Panel`): the thing you would reach for to move a
                        // layer is the layer, and a separate handle beside it would be
                        // the one part of the row that can be dragged while looking
                        // like the rest. Its three gestures share one press — a click
                        // selects, two rename, and a press that travels
                        // [`GRAB_SLOP`] is a move — so the drag arms here and only
                        // *becomes* one once the pointer has said so.
                        //
                        // Capture is what makes the release certain: it is delivered
                        // to the capturing element whatever the pointer is over by
                        // then, and this is a drag where everything under the pointer
                        // moves as you drag it.
                        onpointerdown: move |e: Event<PointerData>| {
                            capture_pointer(&e);
                            let p = e.client_coordinates();
                            drag.set(Some(Grab::begin(
                                id.0.to_string(),
                                layer_boxes(),
                                (p.x as f32, p.y as f32),
                            )));
                        },
                        onpointermove: move |e: Event<PointerData>| {
                            // The armed check first: it is what keeps every pointer
                            // move over the panel from dirtying the whole tree — and a
                            // finished grab is not armed, it is a receipt waiting for
                            // the click behind it (`reorder::claimed`).
                            if drag.peek().as_ref().is_none_or(Grab::over) {
                                return;
                            }
                            let p = e.client_coordinates();
                            // Whether the press that armed this is still down. A row's
                            // name is both the grip and a thing you hover, so this
                            // handler hears every pass of the pointer over the panel,
                            // and a drag whose release went somewhere else would
                            // otherwise be steered by them (`Grab::track`).
                            let held = !e.held_buttons().is_empty();
                            if let Some(d) = drag.write().as_mut() {
                                d.track((p.x as f32, p.y as f32), held);
                            }
                        },
                        onpointerup: move |_| onland.call(id),
                        // A cancel — the browser taking the gesture, a pen leaving the
                        // tablet — ends it the same way, and `onland` declines a drag
                        // that never went live or that lands where it began.
                        onpointercancel: move |_| onland.call(id),
                        // The frame's crop marks, on the row as on the bar — the only
                        // kind of layer that is a *what* rather than a place to paint,
                        // and the one row in the panel whose dashed border is already
                        // saying so. It leads the name whether or not the frame has
                        // been renamed, because the mark is about what the layer is
                        // and the name is about what the author calls it.
                        if matte {
                            {icon(icons::FRAME)}
                        } else if filter {
                            // The funnel the filter bar wears, for the same reason
                            // the crop marks lead a frame's name: the mark is about
                            // what the layer *is*, the name about what it is called.
                            {icon(icons::FILTER)}
                        }
                        "{label}"
                    }
                }
                // Who else is working here (§17.4). The selected layer is
                // per-client, so this is the only place that answers "am I about to
                // paint over what someone else is doing?" before it happens.
                for peer in peers_on(state, id) {
                    div {
                        class: "peer-chip",
                        style: "background:{peer.css_color()}",
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
                        {icon(icons::MERGE_DOWN)}
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
                    {icon(icons::DUPLICATE)}
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
                        {icon(icons::REMOVE)}
                    }
                } else {
                    span { class: "layer-remove" }
                }
                // Last on the line, so the eyes stand in one column down the whole panel
                // however deep the tree goes: a row is indented from the left, and its right
                // edge is where the panel's is. That column is the thing being bought — the
                // tick-boxes this replaces marched *rightwards* with the indent, so reading
                // "what is hidden?" off the panel meant reading every row rather than
                // glancing down an edge. It shows the eye the layer *is*, not the one
                // clicking would give you (see `icons::VISIBLE`).
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
                    {icon(if visible { icons::VISIBLE } else { icons::HIDDEN })}
                }
            }
        }
    }
}

/// The collaborators whose selected layer is `id`.
fn peers_on(state: AppState, id: stark_model::document::LayerId) -> Vec<PeerInfo> {
    state
        .collab
        .peers
        .read()
        .iter()
        .filter(|p| p.active_layer == id)
        .cloned()
        .collect()
}
