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
//! this file has the filter's own numbers and nothing else.
//!
//! With more than one *kind* of filter (§21.10), "+ Filter" grew the one thing it
//! was always going to need: a picker over [`Filter::ALL`], which is the core's own
//! list in the core's own order. The bar itself keys everything per-kind off the
//! selected filter — its knob table, its label, its neutral — so a new kind is a
//! new table here and nothing else.
//!
//! **The colour filter is the one kind that is not only sliders**, and the reason is
//! that three of its numbers are not three things. `hue`, `saturation` and `tint` are
//! a rotation, a scale and a translation of one Oklab `(a, b)` plane, so what they
//! are between them is a single affine map — and the honest picture of an affine map
//! of a plane is the image of a circle. [`chroma_dial`] draws exactly that, over the
//! same Oklab slice the colour picker shows, and every part of it is a fact rather
//! than a metaphor: the rim is where a colour of chroma [`DIAL_CHROMA`] ends up, the
//! centre is where a grey ends up, and the arm is where red ends up. Three tracks
//! could say the same thing, but only one at a time, and none of them could say what
//! the picture says at rest — *this is what the filter will do to a colour*.

use std::sync::LazyLock;

use dioxus::prelude::*;

use crate::icons::{self, icon, label};
use crate::layout::chrome_class;
use crate::panels::color::ab_field_data_url;
use crate::platform::capture_pointer;
use crate::state::{AppState, dispatch};
use crate::widgets::settle;
use stark_core::command::{DocCommand, PeerCommand, ViewCommand};
use stark_core::document::{CONTRAST_PIVOT, ChromaticAberration, ColorAdjust, Filter};
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
/// `glyph` is an `Option` on [`widgets::Slider`](crate::widgets::Slider)'s exact
/// terms, and the same fact is read off it twice: a knob that has a mark wraps its
/// word as hideable, and a knob that has none keeps it, because a row with neither
/// would be an anonymous track. Reading the two off one field is what makes the wrong
/// pair unrepresentable rather than merely unlikely.
///
/// The chromatic filter's two are the `None`s — a to-do, not a decision. The rule
/// that used to keep *all* of these unmarked was about a bar, not about the set: one
/// marked slider among unmarked ones reads worse than none marked. That still holds
/// and is still satisfied, because each bar draws one kind's table and the colour
/// filter's is now marked throughout.
struct Knob<F: 'static> {
    name: &'static str,
    hint: &'static str,
    /// The mark this knob wears, and — see above — whether its word may be hidden.
    glyph: Option<&'static str>,
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
/// of a degree are noise. Shared by the chromatic filter's Angle knob and the dial's
/// hue readout, so the two ways this application shows an angle cannot drift apart.
fn fmt_degrees(v: f32) -> String {
    format!("{}\u{00B0}", v.round() as i32)
}

/// The colour filter's **lightness** knobs — the two the dial has nothing to say
/// about, because they act on Oklab `L` and the dial is one slice of constant `L`.
///
/// That split is the whole reason the bar is a dial *and* two tracks rather than one
/// or the other: a plane picture cannot show a move along the axis it is
/// perpendicular to, and a track cannot show three coupled numbers at once.
const COLOR_KNOBS: &[Knob<ColorAdjust>] = &[
    Knob {
        name: "Exposure",
        hint: "Stops of light. +1 is twice as much, \u{2212}1 is half \u{2014} applied \
               to the light itself, so it brightens the way an exposure does rather \
               than the way a brightness slider does.",
        glyph: Some(icons::EXPOSURE),
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
        glyph: Some(icons::CONTRAST),
        range: ColorAdjust::CONTRAST,
        scale: 1.0,
        get: |c| c.contrast,
        set: |c, v| ColorAdjust { contrast: v, ..c },
        fmt: |v| format!("{v:.2}"),
    },
];

const CHROMATIC_KNOBS: &[Knob<ChromaticAberration>] = &[
    Knob {
        name: "Spread",
        hint: "How far the spectrum is pulled apart, in canvas pixels \u{2014} the \
               width of the fringe every edge grows. The whole rainbow is in \
               between, not three offset copies: the effect is the lens's own \
               dispersion, integrated (\u{a7}21.10).",
        glyph: None,
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
        glyph: None,
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
                span { class: "filter-knob-label",
                    // Mark then word, and the word hideable only because the mark is
                    // there to survive it — the pair `widgets::Slider` reads off one
                    // `Option`, for the reason on [`Knob::glyph`].
                    match knob.glyph {
                        Some(glyph) => rsx! { {icon(glyph)} {label(knob.name)} },
                        None => rsx! { "{knob.name}" },
                    }
                }
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

// —— the chroma dial ————————————————————————————————————————————————————————

/// The dial's field, on screen (px) — square, like the picker's.
const DIAL_PX: f32 = 116.0;

/// The Oklab chroma the rim stands for: the reference colour whose whole hue circle
/// the dial tracks, so a point of the drawn ring is *where a colour of this chroma
/// ends up*.
///
/// A moderately saturated colour rather than the gamut's edge, so the ring at rest
/// sits well inside the plane and has somewhere to grow when saturation is pushed
/// past 1.
const DIAL_CHROMA: f32 = 0.12;

/// Half-extent of the `(a, b)` plane the dial draws, per axis.
///
/// Derived rather than chosen: as far as the centre can travel, plus as wide as the
/// rim can get. That is what makes the box big enough to hold **every reachable
/// setting**, and so what makes the rim handle always inside the element that
/// receives the pointer — no combination of a strong cast and a strong saturation can
/// carry a handle somewhere it cannot be grabbed back from. A hand-picked extent
/// would be a bound that has to be re-checked every time the core's are edited.
const DIAL_AB: f32 = ColorAdjust::TINT.1 + ColorAdjust::SATURATION.1 * DIAL_CHROMA;

/// The Oklab lightness the dial's plane is drawn at: mid-grey, the very lightness the
/// contrast knob pivots about.
///
/// One slice, and one is enough — what the dial shows is a map of `(a, b)` alone,
/// identical at every `L`, so a second slice would draw the same circle over a
/// different backdrop. Hence no slice control: it could not change a pixel.
const DIAL_L: f32 = CONTRAST_PIVOT;

/// px per unit of `a`/`b` in the box above — the one conversion every drawn radius
/// goes through.
const DIAL_SCALE: f32 = DIAL_PX * 0.5 / DIAL_AB;

/// How near the pointer must come to the rim handle to take it rather than the centre
/// (px). Wider than the drawn dot, as a pointer target should be, and small enough
/// that the rest of the field — which is all the centre's — stays a big target.
const DIAL_GRAB: f32 = 10.0;

/// What Shift steps each axis by. Their whole job is to make the round numbers
/// reachable by hand: 0°, a saturation of exactly 1, and a tint of exactly nothing
/// are single points in a continuum a pointer will not land on twice.
const HUE_STEP: f32 = std::f32::consts::PI / 12.0; // 15°
const SATURATION_STEP: f32 = 0.05;
const TINT_STEP: f32 = 0.01;

/// The plane itself, rendered once for the process. Unlike the picker's field there is
/// nothing to invalidate — [`DIAL_L`] and [`DIAL_AB`] are constants — so this is a
/// `LazyLock` rather than a memo per mount, and selecting a filter costs no BMP.
static DIAL_FIELD: LazyLock<String> = LazyLock::new(|| ab_field_data_url(DIAL_L, DIAL_AB));

/// Which of the dial's two handles a drag has hold of.
///
/// Decided once, on pointer-down, and held for the whole gesture — so a rotation
/// swung in past the centre does not become a translation halfway through, which is
/// the one way a live hit test could rewrite a number the hand was not on.
#[derive(Copy, Clone, PartialEq)]
enum Grab {
    /// The rim handle: its direction and its distance from the centre, which are the
    /// hue and the saturation. Taken when the pointer comes down within [`DIAL_GRAB`]
    /// of it — including when saturation is 0 and it sits *on* the centre, because
    /// then it is the only way back out and the centre is still reachable everywhere
    /// else in the field.
    ///
    /// `hold` is where the handle sat relative to the pointer that took it, carried
    /// for the gesture so the first sample is the setting already held. Without it a
    /// grab at the edge of [`DIAL_GRAB`] would fling the saturation by half a unit
    /// before the hand had moved — a target has to be wider than the dot it takes,
    /// and everything inside it has to mean *this one*, not *this position*.
    Rim { hold: [f32; 2] },
    /// Anywhere else in the field: the centre, which is the tint. Absolute rather than
    /// held, and that is the difference in kind: the rim is a handle to grab, the
    /// field is a place to put the grey.
    Centre,
}

/// Where an Oklab `(a, b)` lands in the dial's box, in px from its top-left. `a` runs
/// left→right and `b` bottom→top — the picker's own orientation, warm at the top,
/// because they are two pictures of the same plane.
fn dial_xy(ab: [f32; 2]) -> (f32, f32) {
    (
        (ab[0] / DIAL_AB * 0.5 + 0.5) * DIAL_PX,
        (0.5 - ab[1] / DIAL_AB * 0.5) * DIAL_PX,
    )
}

/// The inverse: the `(a, b)` under a pointer at `(x, y)` in the box.
///
/// Unclamped, deliberately — [`Filter::sanitized`] is the single place a number is
/// held to its range, and it is on the path every edit takes. A clamp here would be a
/// second opinion about the stops, which is exactly how a slider comes to disagree
/// with the value it displays.
fn dial_ab(x: f32, y: f32) -> [f32; 2] {
    [
        (x / DIAL_PX * 2.0 - 1.0) * DIAL_AB,
        (1.0 - y / DIAL_PX * 2.0) * DIAL_AB,
    ]
}

/// `v` to the nearest multiple of `step` — see the `*_STEP` constants.
fn snapped(v: f32, step: f32) -> f32 {
    (v / step).round() * step
}

/// One pointer sample on the dial: the filter that sample means, previewed to the
/// canvas and stashed for the settle — the same live-preview/log-once bargain the
/// sliders make (§21.6), through the same [`settle`].
fn drag_dial(
    state: AppState,
    id: LayerId,
    c: ColorAdjust,
    mut tuning: Signal<Option<Filter>>,
    grab: Grab,
    e: &Event<PointerData>,
) {
    let p = e.element_coordinates();
    let at = dial_ab(p.x as f32, p.y as f32);
    let step = e.modifiers().contains(Modifiers::SHIFT);
    let next = match grab {
        // Absolute, not by delta: the pointer *is* where the grey goes. Which also
        // makes a single click on the field a complete edit, the way a grading wheel
        // behaves — and a click that lands on the setting already held is refused
        // engine-side, so it costs no undo step (§21.6).
        Grab::Centre => ColorAdjust {
            tint: if step {
                [snapped(at[0], TINT_STEP), snapped(at[1], TINT_STEP)]
            } else {
                at
            },
            ..c
        },
        Grab::Rim { hold } => {
            let (dx, dy) = (at[0] + hold[0] - c.tint[0], at[1] + hold[1] - c.tint[1]);
            let r = dx.hypot(dy);
            // At the centre a direction does not exist and `atan2` would answer 0 —
            // which would fling the hue back to red every time a drag crossed the
            // middle on its way somewhere. Keep the angle the gesture already had and
            // move the radius alone.
            let hue = if r > 1e-4 { dy.atan2(dx) } else { c.hue };
            let saturation = r / DIAL_CHROMA;
            ColorAdjust {
                hue: if step { snapped(hue, HUE_STEP) } else { hue },
                saturation: if step {
                    snapped(saturation, SATURATION_STEP)
                } else {
                    saturation
                },
                ..c
            }
        }
    };
    // Through the core's own funnel *before* it is shown, so the pointer cannot
    // preview a setting the commit would then clamp: at the stops the canvas, the
    // readout and the log all say the same number.
    let next = Filter::Color(next).sanitized();
    tuning.set(Some(next));
    dispatch(state, ViewCommand::PreviewFilter(Some((id, next))));
}

/// The colour filter's `hue`, `saturation` and `tint` as the one thing they are: the
/// image of a circle of the Oklab plane under the map they make (see the module docs).
///
/// Everything drawn is a claim about a colour, and each is checkable by eye against
/// the plane it sits on:
///
/// - the **dashed circle** is where the reference colours are — chroma
///   [`DIAL_CHROMA`], every hue — untouched;
/// - the **solid circle** is where the filter sends them;
/// - the **arm and its dot** are where it sends red, which is what makes the circle
///   *directed* and a rotation visible at all;
/// - the **centre** is where it sends grey, which is the tint.
///
/// So the picture at rest already answers "what will this do to a colour", which is
/// the question three tracks can only answer one number at a time.
fn chroma_dial(
    state: AppState,
    id: LayerId,
    c: ColorAdjust,
    tuning: Signal<Option<Filter>>,
    mut grabbed: Signal<Option<Grab>>,
) -> Element {
    let (ox, oy) = dial_xy([0.0, 0.0]);
    let (cx, cy) = dial_xy(c.tint);
    let ring = c.saturation * DIAL_CHROMA * DIAL_SCALE;
    let (hx, hy) = dial_xy([
        c.tint[0] + c.saturation * DIAL_CHROMA * c.hue.cos(),
        c.tint[1] + c.saturation * DIAL_CHROMA * c.hue.sin(),
    ]);
    // How far the centre may travel — a good deal less than the plane is drawn, since
    // the plane is drawn wide enough to hold the *rim* as well (see `DIAL_AB`). Shown
    // only while the centre is the thing being dragged: the rest of the time it is one
    // more line over a picture that is already saying something, and while the tint is
    // in hand it is the difference between a control with a stop and one that has
    // stopped responding.
    let (tx, ty) = dial_xy([ColorAdjust::TINT.0, ColorAdjust::TINT.1]);
    let tspan = (ColorAdjust::TINT.1 - ColorAdjust::TINT.0) * DIAL_SCALE;
    let bounding = grabbed() == Some(Grab::Centre);
    // Which handle a pointer-down takes, decided against where the handle *is* on this
    // render — the closure is over this render's `c`, which is what makes the answer a
    // fact about the picture the user is looking at.
    let take = move |e: &Event<PointerData>| {
        let p = e.element_coordinates();
        let (px, py) = (p.x as f32, p.y as f32);
        if (px - hx).hypot(py - hy) > DIAL_GRAB {
            return Grab::Centre;
        }
        let at = dial_ab(px, py);
        Grab::Rim {
            hold: [
                c.tint[0] + c.saturation * DIAL_CHROMA * c.hue.cos() - at[0],
                c.tint[1] + c.saturation * DIAL_CHROMA * c.hue.sin() - at[1],
            ],
        }
    };
    let commit = move |_| settle(state, tuning, move |f| DocCommand::SetFilter(id, f));

    rsx! {
        div { class: "filter-dial",
            div {
                class: "dial-field",
                style: "background-image: {*DIAL_FIELD};",
                title: "The Oklab plane at mid-grey, and what this filter does to it. \
                        Drag the dot on the rim to turn the hue and pull the \
                        saturation out; drag anywhere else to tint \u{2014} the centre \
                        is the colour a grey becomes. Hold Shift for round steps.",
                // Pointer capture, as the colour picker's field takes: the drag keeps
                // reporting once it leaves the box, and what it reports is clamped by
                // the sanitizer rather than by the element's edge.
                onpointerdown: move |e| {
                    capture_pointer(&e);
                    let grab = take(&e);
                    grabbed.set(Some(grab));
                    drag_dial(state, id, c, tuning, grab, &e);
                },
                onpointermove: move |e| {
                    if let Some(grab) = grabbed() {
                        drag_dial(state, id, c, tuning, grab, &e);
                    }
                },
                // Three ways to end, for `widgets::settle`'s reasons; clearing the grab
                // is what makes a stray move afterwards not a drag.
                onpointerup: move |e| { grabbed.set(None); commit(e); },
                onpointercancel: move |e| { grabbed.set(None); commit(e); },
                svg {
                    class: "dial-svg",
                    width: "{DIAL_PX}",
                    height: "{DIAL_PX}",
                    view_box: "0 0 {DIAL_PX} {DIAL_PX}",
                    if bounding {
                        rect {
                            class: "dial-bound",
                            x: "{tx}", y: "{ty}",
                            width: "{tspan}", height: "{tspan}",
                        }
                    }
                    circle {
                        class: "dial-rest",
                        cx: "{ox}", cy: "{oy}",
                        r: "{DIAL_CHROMA * DIAL_SCALE}",
                    }
                    line { class: "dial-arm", x1: "{cx}", y1: "{cy}", x2: "{hx}", y2: "{hy}" }
                    circle { class: "dial-ring", cx: "{cx}", cy: "{cy}", r: "{ring}" }
                    circle { class: "dial-centre", cx: "{cx}", cy: "{cy}", r: "3" }
                    circle { class: "dial-handle", cx: "{hx}", cy: "{hy}", r: "5" }
                }
            }
            div { class: "dial-readout",
                DialRow {
                    name: "Saturation",
                    value: format!("{:.2}", c.saturation),
                    hint: "How far the ring sits from the centre. 0 is a greyscale that \
                           keeps every lightness exactly where it was, which a luminance \
                           average does not.",
                }
                DialRow {
                    name: "Hue",
                    value: fmt_degrees(c.hue * DEG),
                    hint: "How far the ring is turned. Lightness and saturation are \
                           held, so a hue shift is a hue shift.",
                }
                DialRow {
                    name: "Tint a",
                    value: format!("{:+.3}", c.tint[0]),
                    hint: "Where the centre sits along green \u{2192} red \u{2014} half \
                           of the colour a grey becomes.",
                }
                DialRow {
                    name: "Tint b",
                    value: format!("{:+.3}", c.tint[1]),
                    hint: "Where the centre sits along blue \u{2192} yellow \u{2014} the \
                           other half of the colour a grey becomes.",
                }
            }
        }
    }
}

/// One line of the dial's readout, in the knobs' own label/value pair — the dial
/// replaced two tracks, and the numbers they showed should not have been replaced
/// with them. Shown rather than typed into: the dial is where these are set.
#[component]
fn DialRow(name: &'static str, value: String, hint: &'static str) -> Element {
    rsx! {
        span { class: "filter-knob-label", title: hint, "{name}" }
        span { class: "filter-knob-value", title: hint, "{value}" }
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
    // The dial's half of the same story: which handle a drag has hold of, `None`
    // between drags. Here rather than inside [`chroma_dial`] for the reason above —
    // the dial is mounted for one kind of filter, and a hook that runs for one kind
    // of filter is a hook that runs sometimes.
    let grabbed = use_signal(|| None::<Grab>);
    let Some((info, filter)) = selected_filter(state) else {
        return rsx! {};
    };
    // A filter with nothing composited beneath it in its own stack does nothing at
    // all (§21.2) — the foot of a stack, or a stack whose lower layers are all
    // hidden or empty. Said once, in the bar, rather than greying out the sliders,
    // which would each have to explain the same thing.
    let inert = !info.has_underlay;

    // The one place the bar knows the kinds apart: which controls it puts up. Still a
    // knob table each, and the colour filter's dial ahead of its two — the plane's
    // three numbers first, then the two that move along the axis the plane has none of.
    let rows = match filter {
        Filter::Color(c) => rsx! {
            {chroma_dial(state, info.id, c, tuning, grabbed)}
            span { class: "bar-sep" }
            // Stacked rather than side by side, which is the dial's doing: beside a
            // 116px square, two tracks in a row make a bar half again as wide as it
            // is tall for no reason, and stacking spends height the dial has already
            // bought. They are also a pair — the two ways to move Oklab `L` — so a
            // column reads as one group where a row read as the tail of the dial's.
            div { class: "filter-knob-stack",
                {knob_rows(state, info.id, c, COLOR_KNOBS, Filter::Color, tuning)}
            }
        },
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
