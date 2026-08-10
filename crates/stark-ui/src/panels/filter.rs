//! The filter layer: its "add" picker and the bar that tunes it (§21.6).
//!
//! A filter is a **layer** — a function of what is composited beneath it in its own
//! stack — so almost everything about it is already the Layers panel's job:
//! visibility, opacity (which here is the filter's *strength*), reordering, delete,
//! and above all *placement*, which is the whole of how far the filter reaches. What
//! is left is the filter's own numbers, and they live in a bar mounted only while a
//! filter layer is **selected**, on the argument the frame bar is made this way
//! (`panels::frame`, §15.7): controls meaningless without a filter should be absent
//! rather than greyed out, and a bar that is simply present says "you are grading"
//! more directly than a mode indicator would.
//!
//! **There is no scope control, and that absence is the design.** Every other
//! application answers "what does this adjustment apply to?" with a clipping toggle,
//! a mask, or a group the layer has to be dragged into. Here the answer is where the
//! row sits: at the root it grades the painting, carried onto a layer it grades that
//! layer. So the control already exists and it is the Layers panel — which is why
//! this file has a slider per parameter and nothing else.
//!
//! With more than one *kind* of filter (§21.10), "+ Filter" grew the one thing it
//! was always going to need: a picker over [`Filter::ALL`], which is the core's own
//! list in the core's own order. The bar itself keys everything per-kind off the
//! selected filter — its knob table, its label, its neutral — so a new kind is a
//! new table here and nothing else.

use dioxus::prelude::*;

use crate::icons::{self, icon, label};
use crate::layout::chrome_class;
use crate::state::{AppState, dispatch};
use crate::widgets::settle;
use stark_core::command::{DocCommand, PeerCommand, ViewCommand};
use stark_core::document::{ChromaticAberration, ColorAdjust, Filter};
use stark_core::{LayerId, LayerInfo};

/// One slider on the bar: what it is called, its range, and the two ends of the
/// round trip through the filter's own parameter struct `F`.
///
/// A table rather than hand-written rows because the rows differ in nothing but
/// these five things, and a hand-written row is a place for one of them to disagree
/// with the value it displays. `get`/`set` are the pair that makes the whole filter
/// travel on every edit (§21.6) — the bar reads the current settings off the
/// projection, replaces one number, and sends the result back. Generic over the
/// parameter struct so each filter kind is one more `const` table and no more
/// bar code.
///
/// **No glyphs, deliberately unmarked.** The icon set has a mark for exactly one of
/// the colour knobs (a tonal ramp for contrast), and a bar where one slider is
/// marked and the rest are not reads worse than one where none is — so these keep
/// their words, and keep them in minimal mode too, which is the rule an unmarked
/// control already carries (`widgets::Slider`). A to-do rather than a decision: the
/// day there is a mark per knob, this grows a `glyph` and the words become hideable
/// together.
struct Knob<F: 'static> {
    name: &'static str,
    hint: &'static str,
    /// The slider's span, in display units — derived from the core's own bounds
    /// (`ColorAdjust::EXPOSURE` and friends) so the track and the sanitizer cannot
    /// disagree about how far a knob goes.
    range: (f32, f32),
    /// What the slider shows, in the unit the *hand* thinks in — degrees for an
    /// angle, the number itself for the rest. The engine's unit is radians (§21.5),
    /// and translating here is what keeps "how an angle is presented" out of the log.
    scale: f32,
    get: fn(&F) -> f32,
    set: fn(F, f32) -> F,
    /// How the number beside the track reads, in the slider's own unit — in the
    /// table with everything else per-knob, so renaming a knob cannot silently
    /// change how its value prints.
    fmt: fn(f32) -> String,
}

/// Degrees per radian, for the knobs whose display unit is not the engine's.
const DEG: f32 = 180.0 / std::f32::consts::PI;

/// A whole number of degrees — an angle's readout to anyone dragging it; decimals
/// of a degree are noise. Shared by the two angle knobs so they cannot drift apart.
fn fmt_degrees(v: f32) -> String {
    format!("{}\u{00B0}", v.round() as i32)
}

const COLOR_KNOBS: &[Knob<ColorAdjust>] = &[
    Knob {
        name: "Exposure",
        hint: "Stops of light. +1 is twice as much, \u{2212}1 is half \u{2014} applied \
               to the light itself, so it brightens the way an exposure does rather \
               than the way a brightness slider does.",
        range: ColorAdjust::EXPOSURE,
        scale: 1.0,
        get: |c| c.exposure,
        set: |c, v| ColorAdjust { exposure: v, ..c },
        // The `+` is worth the arm: a stop is a signed quantity centred on zero,
        // and "0.50" and "+0.50" say different things.
        fmt: |v| format!("{v:+.2}"),
    },
    Knob {
        name: "Contrast",
        hint: "Spread about mid-grey. 1 leaves it alone, 0 flattens the picture to \
               one tone. It moves lightness only \u{2014} the colours keep their \
               saturation, which is not true of a contrast curve in sRGB.",
        range: ColorAdjust::CONTRAST,
        scale: 1.0,
        get: |c| c.contrast,
        set: |c, v| ColorAdjust { contrast: v, ..c },
        fmt: |v| format!("{v:.2}"),
    },
    Knob {
        name: "Saturation",
        hint: "How far the colours sit from grey. 0 is a greyscale that keeps every \
               lightness exactly where it was, which a luminance average does not.",
        range: ColorAdjust::SATURATION,
        scale: 1.0,
        get: |c| c.saturation,
        set: |c, v| ColorAdjust { saturation: v, ..c },
        fmt: |v| format!("{v:.2}"),
    },
    Knob {
        name: "Hue",
        hint: "Turn every colour around the wheel, in degrees. Lightness and \
               saturation are held, so a hue shift is a hue shift.",
        range: (ColorAdjust::HUE.0 * DEG, ColorAdjust::HUE.1 * DEG),
        scale: 1.0 / DEG,
        get: |c| c.hue,
        set: |c, v| ColorAdjust { hue: v, ..c },
        fmt: fmt_degrees,
    },
];

const CHROMATIC_KNOBS: &[Knob<ChromaticAberration>] = &[
    Knob {
        name: "Spread",
        hint: "How far the spectrum is pulled apart, in canvas pixels \u{2014} the \
               width of the fringe every edge grows. The whole rainbow is in \
               between, not three offset copies: the effect is the lens's own \
               dispersion, integrated (\u{a7}21.10).",
        range: ChromaticAberration::SPREAD,
        scale: 1.0,
        get: |c| c.spread,
        set: |c, v| ChromaticAberration { spread: v, ..c },
        fmt: |v| format!("{v:.1}"),
    },
    Knob {
        name: "Angle",
        hint: "The axis the colours part along, in degrees \u{2014} the way the blue \
               end of the spectrum is carried, with the red end opposite. Stated on \
               the canvas, so it turns with the painting rather than the window.",
        range: (
            ChromaticAberration::ANGLE.0 * DEG,
            ChromaticAberration::ANGLE.1 * DEG,
        ),
        scale: 1.0 / DEG,
        get: |c| c.angle,
        set: |c, v| ChromaticAberration { angle: v, ..c },
        fmt: fmt_degrees,
    },
];

/// The filter being tuned, if the **selected layer** is one.
///
/// There is deliberately no separate filter-selection state, for the reason
/// [`selected_frame`](super::frame::selected_frame) has none: `active_layer` is the
/// selected layer and a filter may be it, so the bar keys off the same thing the
/// Layers panel highlights — and a filter that is removed, undone, or replaced by a
/// document load stops being tuned with no invalidation to remember.
pub fn selected_filter(state: AppState) -> Option<(LayerInfo, Filter)> {
    let obs = state.obs.read();
    let o = obs.as_ref()?;
    o.layers
        .iter()
        .find(|l| l.id == o.active_layer)
        .and_then(|l| l.filter.map(|f| (l.clone(), f)))
}

/// Stop tuning: select the topmost paint layer instead — a filter, like a frame, is
/// only "deselected" by selecting something else.
fn done_grading(state: AppState) {
    super::frame::done_composing(state);
}

/// Land `filter` above the selected layer, in that layer's own stack, and select it
/// — the body of the picker's items, shared so each kind is one entry there.
///
/// The placement is the same one "+ Layer" uses, and is the one decision that
/// matters here: adding a filter while working inside a group grades that group.
/// The new filter is selected immediately, so its bar comes up without a second
/// click.
fn add_filter(state: AppState, at: Option<(Option<LayerId>, Option<LayerId>)>, filter: Filter) {
    let (carrier, above) = at.unwrap_or((None, None));
    // The ids that already exist, taken before the dispatch: `AddFilter` mints the
    // new id engine-side, so the new layer is the one the projection gains — not
    // "the topmost filter", which is somebody else's filter the moment one already
    // sits above the insertion point.
    let before: Vec<LayerId> = state
        .obs
        .read()
        .as_ref()
        .map(|o| o.layers.iter().map(|l| l.id).collect())
        .unwrap_or_default();
    dispatch(
        state,
        DocCommand::AddFilter {
            carrier,
            above,
            filter,
        },
    );
    let new_id = state.obs.read().as_ref().and_then(|o| {
        o.layers
            .iter()
            .find(|l| l.filter.is_some() && !before.contains(&l.id))
            .map(|l| l.id)
    });
    if let Some(id) = new_id {
        dispatch(state, PeerCommand::SetActiveLayer(id));
    }
}

/// The "+ Filter" button, for the Layers panel's header — a filter *is* a layer, so
/// that is where making one belongs, beside "+ Layer" and "+ Frame".
///
/// It opens a small picker over [`Filter::ALL`] — every kind this build offers, in
/// the core's own order, each landing at its neutral setting so adding one changes
/// nothing until it is dialled. A picker rather than a fourth and fifth header
/// button, because the header is a row of *what a layer can be* and the kinds of
/// filter are one of those things, not three.
#[component]
pub fn AddFilterButton() -> Element {
    let state = use_context::<AppState>();
    let mut open = use_signal(|| false);
    // Where it goes, read out here rather than in the handler: the same pair
    // `LayerPanel` computes for "+ Layer", for the same reason.
    let at = state.obs.read().as_ref().and_then(|o| {
        o.layers
            .iter()
            .find(|l| l.id == o.active_layer)
            .map(|l| (l.carrier, Some(l.id)))
    });
    rsx! {
        div {
            class: "filter-add",
            // Clicking anywhere else blurs the trigger and folds the picker away.
            // The items act on `pointerdown` below, which the browser delivers
            // *before* this blur — so choosing a kind wins the race with the menu
            // unmounting under the pointer.
            onfocusout: move |_| open.set(false),
            button {
                class: "layer-add",
                title: "Add a filter: it adjusts everything composited below it in \
                        its own stack \u{2014} the whole painting at the top, one \
                        group inside one",
                onclick: move |_| {
                    let show = !open();
                    open.set(show);
                },
                {icon(icons::ADD_FILTER)}
                {label("Filter")}
            }
            if open() {
                div { class: "filter-add-menu",
                    for f in Filter::ALL {
                        button {
                            key: "{f.label()}",
                            class: "filter-add-item",
                            onpointerdown: move |_| {
                                open.set(false);
                                add_filter(state, at, f);
                            },
                            "{f.label()}"
                        }
                    }
                }
            }
        }
    }
}

/// The rows of sliders for one filter kind — the generic half of the bar, so a new
/// kind brings a knob table and nothing else. `current` is the whole filter's
/// settings off the projection; each edit replaces one number and sends the whole
/// thing back through `wrap`, which is what keeps "the filter travels entire" true
/// per kind without the bar knowing any kind's shape (§21.6).
fn knob_rows<F: Copy + 'static>(
    state: AppState,
    id: LayerId,
    current: F,
    knobs: &'static [Knob<F>],
    wrap: fn(F) -> Filter,
    mut tuning: Signal<Option<Filter>>,
) -> Element {
    rsx! {
        for knob in knobs {
            div {
                key: "{knob.name}",
                class: "filter-knob",
                title: "{knob.hint}",
                span { class: "filter-knob-label", "{knob.name}" }
                span { class: "filter-knob-value", "{readout(knob, &current)}" }
                input {
                    class: "slider",
                    r#type: "range",
                    min: "{knob.range.0}", max: "{knob.range.1}", step: "any",
                    value: "{(knob.get)(&current) / knob.scale}",
                    // Previewed per sample, committed once when the drag settles.
                    // A filter is judged *by looking*, so every value the pointer
                    // crosses has to reach the canvas — and only the answer belongs
                    // in the log (§21.6).
                    oninput: move |e| {
                        if let Ok(v) = e.value().parse::<f32>() {
                            let next = wrap((knob.set)(current, v * knob.scale));
                            tuning.set(Some(next));
                            dispatch(state, ViewCommand::PreviewFilter(Some((id, next))));
                        }
                    },
                    // Three ways to end, because a range control has three — see
                    // `widgets::settle`, which holds the why (and is idempotent,
                    // so arriving twice is free).
                    onchange: move |_| settle(state, tuning, move |f| DocCommand::SetFilter(id, f)),
                    onpointerup: move |_| settle(state, tuning, move |f| DocCommand::SetFilter(id, f)),
                    onpointercancel: move |_| settle(state, tuning, move |f| DocCommand::SetFilter(id, f)),
                }
            }
        }
    }
}

/// The selected filter's controls, in a bar at the bottom of the screen. Mounted
/// only while a filter layer is selected — see the module docs for why that is the
/// whole interaction model rather than a panel with an empty state.
#[component]
pub fn FilterBar() -> Element {
    let state = use_context::<AppState>();
    // The value a drag is showing, or `None` between drags — the drag's own "there is
    // something to commit", panel-local and delimited by the browser's own gesture, so
    // it cannot be left armed. It is the *filter*, not a flag, so the commit says what
    // the last preview showed rather than reading it back off a projection the preview
    // is itself feeding (§21.6, and §14.6 for the opacity slider that does the same).
    //
    // **Before** the early return, because a hook that runs only when a filter is
    // selected is a hook that runs sometimes.
    let tuning = use_signal(|| None::<Filter>);
    let Some((info, filter)) = selected_filter(state) else {
        return rsx! {};
    };
    // A filter with nothing composited beneath it in its own stack does nothing at
    // all (§21.2) — the foot of a stack, or a stack whose lower layers are all
    // hidden or empty. Said once, in the bar, rather than greying out the sliders,
    // which would each have to explain the same thing.
    let inert = !info.has_underlay;

    // The one place the bar knows the kinds apart: which table its rows come from.
    let rows = match filter {
        Filter::Color(c) => knob_rows(state, info.id, c, COLOR_KNOBS, Filter::Color, tuning),
        Filter::Chromatic(c) => knob_rows(
            state,
            info.id,
            c,
            CHROMATIC_KNOBS,
            Filter::Chromatic,
            tuning,
        ),
    };

    rsx! {
        div { class: chrome_class(state, "filter-bar"),
            // The glyph rides the bar's *label*, as the frame bar's crop marks do:
            // no single slider here is "the filter", so what the mark identifies is
            // the bar, and through it the layer you are tuning.
            span { class: "bar-label",
                {icon(icons::FILTER)}
                {label(filter.label())}
            }

            if inert {
                span {
                    class: "filter-inert",
                    title: "A filter adjusts what is composited below it in its own \
                            stack. Move it above a layer with something visible on \
                            it and it will have something to work on.",
                    "nothing below it"
                }
            }

            span { class: "bar-sep" }

            {rows}

            span { class: "bar-sep" }

            button {
                class: "chip",
                title: "Put every slider back to neutral \u{2014} the filter stays, \
                        doing nothing, until it is dialled again",
                disabled: filter.is_neutral(),
                onclick: move |_| {
                    dispatch(state, DocCommand::SetFilter(info.id, filter.neutral()));
                },
                {icon(icons::RESET)}
                {label("Neutral")}
            }
            button {
                class: "chip",
                title: "Stop tuning and go back to painting (the filter stays)",
                onclick: move |_| done_grading(state),
                {icon(icons::DONE)}
                {label("Done")}
            }
        }
    }
}

/// What the number reads as beside its track: the knob's own `fmt`, on the value in
/// the slider's unit — per-knob in the table, so a label edit cannot change how a
/// value prints.
fn readout<F>(knob: &Knob<F>, settings: &F) -> String {
    (knob.fmt)((knob.get)(settings) / knob.scale)
}
