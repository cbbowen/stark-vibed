//! The filter layer: its "add" button and the bar that tunes it (§21.6).
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

use dioxus::prelude::*;

use crate::icons::{self, icon, label};
use crate::layout::chrome_class;
use crate::state::{AppState, dispatch};
use crate::widgets::settle;
use stark_core::command::{DocCommand, PeerCommand, ViewCommand};
use stark_core::document::{ColorAdjust, Filter};
use stark_core::{LayerId, LayerInfo};

/// One slider on the bar: what it is called, its range, and the two ends of the
/// round trip through [`ColorAdjust`].
///
/// A table rather than four hand-written rows because the four differ in nothing but
/// these five things, and a hand-written row is a place for one of them to disagree
/// with the value it displays. `get`/`set` are the pair that makes the whole filter
/// travel on every edit (§21.6) — the bar reads the current adjustment off the
/// projection, replaces one number, and sends the result back.
///
/// **No glyphs, deliberately unmarked.** The icon set has a mark for exactly one of
/// these four (a tonal ramp for contrast), and a bar where one slider is marked and
/// three are not reads worse than one where none is — so these keep their words, and
/// keep them in minimal mode too, which is the rule an unmarked control already
/// carries (`widgets::Slider`). A to-do rather than a decision: the day there are
/// four marks, this grows a `glyph` and the words become hideable together.
struct Knob {
    name: &'static str,
    hint: &'static str,
    /// The slider's span, in display units — derived from the core's own bounds
    /// (`ColorAdjust::EXPOSURE` and friends) so the track and the sanitizer cannot
    /// disagree about how far a knob goes.
    range: (f32, f32),
    /// What the slider shows, in the unit the *hand* thinks in — degrees for the
    /// hue, the number itself for the rest. The engine's unit is radians (§21.5),
    /// and translating here is what keeps "how an angle is presented" out of the log.
    scale: f32,
    get: fn(&ColorAdjust) -> f32,
    set: fn(ColorAdjust, f32) -> ColorAdjust,
    /// How the number beside the track reads, in the slider's own unit — in the
    /// table with everything else per-knob, so renaming a knob cannot silently
    /// change how its value prints.
    fmt: fn(f32) -> String,
}

/// Degrees per radian, for the one knob whose display unit is not the engine's.
const DEG: f32 = 180.0 / std::f32::consts::PI;

const KNOBS: &[Knob] = &[
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
        // A hue is a whole number of degrees to anyone reading it; decimals of a
        // degree are noise.
        fmt: |v| format!("{}\u{00B0}", v.round() as i32),
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

/// The "+ Filter" button, for the Layers panel's header — a filter *is* a layer, so
/// that is where making one belongs, beside "+ Layer" and "+ Frame".
///
/// It lands **above the selected layer, in that layer's own stack**, which is the
/// same placement "+ Layer" uses and is the one decision that matters here: it means
/// adding a filter while working inside a group grades that group, and adding one
/// with a top-level layer selected grades everything under it. The new filter is
/// selected immediately, so its bar comes up without a second click.
#[component]
pub fn AddFilterButton() -> Element {
    let state = use_context::<AppState>();
    // Where it goes, read out here rather than in the handler: the same pair
    // `LayerPanel` computes for "+ Layer", for the same reason.
    let at = state.obs.read().as_ref().and_then(|o| {
        o.layers
            .iter()
            .find(|l| l.id == o.active_layer)
            .map(|l| (l.carrier, Some(l.id)))
    });
    rsx! {
        button {
            class: "layer-add",
            title: "Add a filter: it adjusts everything composited below it in its \
                    own stack \u{2014} the whole painting at the top, one group \
                    inside one",
            onclick: move |_| {
                let (carrier, above) = at.unwrap_or((None, None));
                // The ids that already exist, taken before the dispatch: `AddFilter`
                // mints the new id engine-side, so the new layer is the one the
                // projection gains — not "the topmost filter", which is somebody
                // else's filter the moment one already sits above the insertion
                // point.
                let before: Vec<LayerId> = state
                    .obs
                    .read()
                    .as_ref()
                    .map(|o| o.layers.iter().map(|l| l.id).collect())
                    .unwrap_or_default();
                dispatch(state, DocCommand::AddFilter {
                    carrier,
                    above,
                    // Neutral, so adding one changes nothing until it is dialled.
                    filter: Filter::Color(ColorAdjust::NEUTRAL),
                });
                // Select it, so its bar comes up without a second click.
                let new_id = state.obs.read().as_ref().and_then(|o| {
                    o.layers
                        .iter()
                        .find(|l| l.filter.is_some() && !before.contains(&l.id))
                        .map(|l| l.id)
                });
                if let Some(id) = new_id {
                    dispatch(state, PeerCommand::SetActiveLayer(id));
                }
            },
            {icon(icons::ADD_FILTER)}
            {label("Filter")}
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
    let mut tuning = use_signal(|| None::<Filter>);
    let Some((info, filter)) = selected_filter(state) else {
        return rsx! {};
    };
    let Filter::Color(adjust) = filter;
    // A filter with nothing composited beneath it in its own stack does nothing at
    // all (§21.2) — the foot of a stack, or a stack whose lower layers are all
    // hidden or empty. Said once, in the bar, rather than greying out four sliders
    // that would each have to explain the same thing.
    let inert = !info.has_underlay;

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

            for knob in KNOBS {
                div {
                    key: "{knob.name}",
                    class: "filter-knob",
                    title: "{knob.hint}",
                    span { class: "filter-knob-label", "{knob.name}" }
                    span { class: "filter-knob-value", "{readout(knob, &adjust)}" }
                    input {
                        class: "slider",
                        r#type: "range",
                        min: "{knob.range.0}", max: "{knob.range.1}", step: "any",
                        value: "{(knob.get)(&adjust) / knob.scale}",
                        // Previewed per sample, committed once when the drag settles.
                        // A colour adjustment is judged *by looking*, so every value
                        // the pointer crosses has to reach the canvas — and only the
                        // answer belongs in the log (§21.6).
                        oninput: move |e| {
                            if let Ok(v) = e.value().parse::<f32>() {
                                let next = Filter::Color((knob.set)(adjust, v * knob.scale));
                                tuning.set(Some(next));
                                dispatch(state, ViewCommand::PreviewFilter(Some((info.id, next))));
                            }
                        },
                        // Three ways to end, because a range control has three — see
                        // `widgets::settle`, which holds the why (and is idempotent,
                        // so arriving twice is free).
                        onchange: move |_| settle(state, tuning, |f| DocCommand::SetFilter(info.id, f)),
                        onpointerup: move |_| settle(state, tuning, |f| DocCommand::SetFilter(info.id, f)),
                        onpointercancel: move |_| settle(state, tuning, |f| DocCommand::SetFilter(info.id, f)),
                    }
                }
            }

            span { class: "bar-sep" }

            button {
                class: "chip",
                title: "Put every slider back to neutral \u{2014} the filter stays, \
                        doing nothing, until it is dialled again",
                disabled: filter.is_neutral(),
                onclick: move |_| {
                    dispatch(state, DocCommand::SetFilter(
                        info.id, Filter::Color(ColorAdjust::NEUTRAL),
                    ));
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
fn readout(knob: &Knob, adjust: &ColorAdjust) -> String {
    (knob.fmt)((knob.get)(adjust) / knob.scale)
}
