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
//! The clip chip a filter row *does* wear (§21.4.1) is not that control and does not
//! reopen this: it bounds what the pass may **write** — a clipped filter cannot push
//! coverage past the paint it found — while what the pass **reads** is still, only,
//! where the row sits.
//!
//! With more than one *kind* of filter (§21.10), "+ Filter" grew the one thing it
//! was always going to need: a picker over [`Filter::ALL`], which is the core's own
//! list in the core's own order. The bar itself keys everything per-kind off the
//! selected filter — its controls, its label, its neutral — so a new kind lands in
//! one `match` arm and nothing else. [`knob_rows`] is still the cheap way to fill
//! that arm: a `const` table of [`Knob`]s and no more bar code.
//!
//! **Neither kind of filter here is only sliders**, and both for the same reason:
//! their numbers are not separate things, so separate tracks would be the wrong
//! picture of them.
//!
//! `hue`, `saturation` and `tint` are a rotation, a scale and a translation of one
//! Oklab `(a, b)` plane, so what they are between them is a single affine map — and
//! the honest picture of an affine map of a plane is the image of a circle.
//! [`chroma_dial`] draws exactly that, over the same Oklab slice the color picker
//! shows, and every part of it is a fact rather than a metaphor: the rim is where a
//! color of chroma [`DIAL_CHROMA`] ends up, the centre is where a grey ends up, and
//! the arm is where red ends up. Three tracks could say the same thing, but only one
//! at a time, and none of them could say what the picture says at rest — *this is
//! what the filter will do to a color*.
//!
//! The chromatic filter's `spread` and `angle` are a length and a direction, which is
//! to say they are **one vector** — the displacement from where the red end of the
//! spectrum lands to where the blue end does (§21.10). [`fringe_pad`] draws that
//! vector, and draws it as the thing it describes: a bar of the real dispersion
//! spectrum, painted with the pass's own [`dispersion_weight`] at the pass's own
//! wavelengths, growing out of the centre in the direction the fringe will run.
//! Dragging it *is* pulling the spectrum apart. Two tracks could not show that the
//! two numbers are one arrow, and no slider can show what the fringe will look like.

use std::sync::LazyLock;

use dioxus::prelude::*;

use crate::commands::{self, Command};
use crate::icons::{self, icon, label};
use crate::layout::chrome_class;
use crate::panels::color::ab_field_data_url;
use crate::panels::gradients::GradientWell;
use crate::platform::capture_pointer;
use crate::preview;
use crate::state::{AppState, dispatch, use_obs, use_obs_opt};
use stark_engine::LayerInfo;
use stark_engine::command::{DocCommand, PeerCommand};
use stark_engine::filters::{CONTRAST_PIVOT, dispersion_weight};
use stark_model::color::linear_to_srgb;
use stark_model::document::LayerId;
use stark_model::document::{ChromaticAberration, ColorAdjust, Filter};
use stark_model::gradient::Gradient;

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
/// The rule is about a bar, not about the set: one marked slider among unmarked ones
/// reads worse than none marked. It is satisfied here — each bar draws one kind's
/// table, and the color filter's, the only table with marks, is marked throughout.
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
/// of a degree are noise. Shared by the dial's hue and the dispersion pad's axis, so
/// the two ways this application shows an angle cannot drift apart.
fn fmt_degrees(v: f32) -> String {
    format!("{}\u{00B0}", v.round() as i32)
}

/// The color filter's **lightness** knobs — the two the dial has nothing to say
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
               one tone. It moves lightness only \u{2014} the colors keep their \
               saturation, which is not true of a contrast curve in sRGB.",
        glyph: Some(icons::CONTRAST),
        range: ColorAdjust::CONTRAST,
        scale: 1.0,
        get: |c| c.contrast,
        set: |c, v| ColorAdjust { contrast: v, ..c },
        fmt: |v| format!("{v:.2}"),
    },
];

/// The filter being tuned, if the **selected layer** is one.
///
/// There is deliberately no separate filter-selection state, for the reason
/// [`selected_frame`](super::frame::selected_frame) has none: `active_layer` is the
/// selected layer and a filter may be it, so the bar keys off the same thing the
/// Layers panel highlights — and a filter that is removed, undone, or replaced by a
/// document load stops being tuned with no invalidation to remember.
/// The rule itself, asked of a projection already in hand.
fn selected_filter_of(o: &stark_engine::ObservableState) -> Option<(LayerInfo, Filter)> {
    o.layers
        .iter()
        .find(|l| l.id == o.active_layer)
        .and_then(|l| l.filter.clone().map(|f| (l.clone(), f)))
}

/// [`selected_filter_of`] asked from an **event handler** — `peek`, so a handler
/// does not subscribe the scope it happens to run under to the whole projection.
/// The handler-time half, exactly as [`modes::is_composing`](crate::modes::is_composing)
/// is to `modes::composing`.
pub fn selected_filter(state: AppState) -> Option<(LayerInfo, Filter)> {
    state.obs.peek().as_ref().and_then(selected_filter_of)
}

/// [`selected_filter_of`] asked from a **component** — a memo, so the bar wakes
/// when the selected filter changes and sleeps through every pan and stroke sample
/// that merely rewrites the projection (`state::use_obs`). The render-time half.
///
/// A hook: call unconditionally, and above any early return.
fn use_selected_filter(state: AppState) -> Memo<Option<(LayerInfo, Filter)>> {
    use_obs_opt(state, |o| o.and_then(selected_filter_of))
}

/// Hand the library's current ramp to the selected **gradient map** filter — the
/// body of a library pop-out row click while one is being tuned (§21.11).
///
/// Called from [`gradients::select`](crate::gradients::select), so "clicking a row
/// takes the ramp" means the same thing here it means for a composing fill or a
/// matte: a click is a choice, and the one consumer with a bar up receives it. A
/// discrete click commits directly — there is nothing continuous to preview-then-
/// settle — and re-clicking the ramp the filter already wears is refused
/// engine-side, so it costs no undo step (§21.6). While any mode is composing
/// this stands down (`crate::modes`): under the gradient *bar* the click belongs
/// to that composition instead — the case this began as — and under the other
/// three it would commit a document edit beneath a live preview, which is the
/// same refusal the canvas and the shortcuts make.
pub fn apply_ramp(state: AppState) {
    if crate::modes::is_composing(state) {
        return;
    }
    let Some((info, Filter::GradientMap(_))) = selected_filter(state) else {
        return;
    };
    let Some(g) = crate::gradients::current(state) else {
        return;
    };
    dispatch(
        state,
        DocCommand::SetFilter(info.id, Filter::GradientMap(Some(g))),
    );
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
        .peek()
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
    let new_id = state.obs.peek().as_ref().and_then(|o| {
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
    let at = use_obs(state, |o| {
        o.layers
            .iter()
            .find(|l| l.id == o.active_layer)
            .map(|l| (l.carrier, Some(l.id)))
    })()
    .flatten();
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
                        {
                            let name = f.label();
                            rsx! {
                                button {
                                    key: "{name}",
                                    class: "filter-add-item",
                                    onpointerdown: move |_| {
                                        open.set(false);
                                        add_filter(state, at, f.clone());
                                    },
                                    "{name}"
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The rows of sliders for one filter kind — the generic half of the bar, so a kind
/// whose numbers really are separate is a knob table and nothing else. `current` is
/// the whole filter's
/// settings off the projection; each edit replaces one number and sends the whole
/// thing back through `wrap`, which is what keeps "the filter travels entire" true
/// per kind without the bar knowing any kind's shape (§21.6).
fn knob_rows<F: Copy + 'static>(
    state: AppState,
    id: LayerId,
    current: F,
    knobs: &'static [Knob<F>],
    wrap: fn(F) -> Filter,
    tuning: Signal<Option<(LayerId, Filter)>>,
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
                            preview::FILTER.during(state, tuning, (id, next));
                        }
                    },
                    // Three ways to end, because a range control has three — see
                    // `Preview::settle`, which holds the why (and is idempotent,
                    // so arriving twice is free).
                    onchange: move |_| preview::FILTER.settle(state, tuning),
                    onpointerup: move |_| preview::FILTER.settle(state, tuning),
                    onpointercancel: move |_| preview::FILTER.settle(state, tuning),
                }
            }
        }
    }
}

// —— the chroma dial ————————————————————————————————————————————————————————

/// The dial's field, on screen (px) — square, like the picker's.
const DIAL_PX: f32 = 116.0;

/// The Oklab chroma the rim stands for: the reference color whose whole hue circle
/// the dial tracks, so a point of the drawn ring is *where a color of this chroma
/// ends up*.
///
/// A moderately saturated color rather than the gamut's edge, so the ring at rest
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
/// reachable by hand: 0°, a saturation of exactly 1, a tint of exactly nothing and a
/// spread of a whole pixel are single points in a continuum a pointer will not land
/// on twice.
///
/// One angle step for both pictures, because "the round angles" is a fact about
/// hands and not about hue: the dial's rotation and the fringe's axis want the same
/// dozen directions, and two constants of the same value would be two chances to
/// disagree.
const ANGLE_STEP: f32 = std::f32::consts::PI / 12.0; // 15°
const SATURATION_STEP: f32 = 0.05;
const TINT_STEP: f32 = 0.01;
const SPREAD_STEP: f32 = 1.0; // one canvas px

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
    tuning: Signal<Option<(LayerId, Filter)>>,
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
                hue: if step { snapped(hue, ANGLE_STEP) } else { hue },
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
    preview::FILTER.during(state, tuning, (id, next));
}

/// The color filter's `hue`, `saturation` and `tint` as the one thing they are: the
/// image of a circle of the Oklab plane under the map they make (see the module docs).
///
/// Everything drawn is a claim about a color, and each is checkable by eye against
/// the plane it sits on:
///
/// - the **dashed circle** is where the reference colors are — chroma
///   [`DIAL_CHROMA`], every hue — untouched;
/// - the **solid circle** is where the filter sends them;
/// - the **arm and its dot** are where it sends red, which is what makes the circle
///   *directed* and a rotation visible at all;
/// - the **centre** is where it sends grey, which is the tint.
///
/// So the picture at rest already answers "what will this do to a color", which is
/// the question three tracks can only answer one number at a time.
fn chroma_dial(
    state: AppState,
    id: LayerId,
    c: ColorAdjust,
    tuning: Signal<Option<(LayerId, Filter)>>,
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
    let commit = move |_| preview::FILTER.settle(state, tuning);

    rsx! {
        div { class: "filter-dial",
            div {
                class: "dial-field",
                style: "background-image: {*DIAL_FIELD};",
                title: "The Oklab plane at mid-grey, and what this filter does to it. \
                        Drag the dot on the rim to turn the hue and pull the \
                        saturation out; drag anywhere else to tint \u{2014} the centre \
                        is the color a grey becomes. Hold Shift for round steps.",
                // Pointer capture, as the color picker's field takes: the drag keeps
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
                // Three ways to end, for `Preview::settle`'s reasons; clearing the grab
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
                           of the color a grey becomes.",
                }
                DialRow {
                    name: "Tint b",
                    value: format!("{:+.3}", c.tint[1]),
                    hint: "Where the centre sits along blue \u{2192} yellow \u{2014} the \
                           other half of the color a grey becomes.",
                }
            }
        }
    }
}

/// One line of a picture's readout, in the knobs' own label/value pair — a picture
/// replaced some tracks, and the numbers they showed should not have been replaced
/// with them. Shown rather than typed into: the picture is where these are set.
#[component]
fn DialRow(name: &'static str, value: String, hint: &'static str) -> Element {
    rsx! {
        span { class: "filter-knob-label", title: hint, "{name}" }
        span { class: "filter-knob-value", title: hint, "{value}" }
    }
}

// —— the dispersion pad ——————————————————————————————————————————————————————

/// The pad's field, on screen (px) — square, and the dial's size, because the two are
/// the same bar's picture and a bar whose height changed with the kind of filter
/// selected would jump under the pointer that selected it.
const PAD_PX: f32 = DIAL_PX;

/// The centre of the field, in its own px — where the picture stays put.
const PAD_MID: f32 = PAD_PX * 0.5;

/// The radius the widest reachable spread is drawn at (px): the field's half-width,
/// less room for the handle itself.
///
/// Derived rather than picked, on [`DIAL_AB`]'s argument turned the other way round:
/// what has to be true is that **no reachable setting puts a handle outside the
/// element that receives the pointer**, so the margin is the handle's own radius plus
/// its grab slop, and a fatter handle moves the stop rather than escaping it.
const PAD_R: f32 = PAD_MID - PAD_HANDLE - 2.0;

/// The handle's drawn radius (px) — the dot at the blue end of the fringe.
const PAD_HANDLE: f32 = 5.0;

/// The spreads the graduation rings stand for, in canvas px — 8 and 32, with the
/// ceiling itself drawn as the stop beyond them.
///
/// Each is a quarter of the next, which under the square-root law below puts them at
/// exactly half of each other's radius — so the ladder draws itself evenly and the
/// picture states its own scale: rings that crowd outward *are* the compression. Two
/// rungs and the rim rather than three and the rim, because a fourth would land 6px
/// from the centre and its label on top of the one above it.
const PAD_RINGS: &[f32] = &[
    ChromaticAberration::SPREAD.1 / 16.0,
    ChromaticAberration::SPREAD.1 / 4.0,
];

/// How many stops the drawn spectrum spends. Enough that the bar reads as a continuum
/// at the widest it is ever drawn (`2 · PAD_R` px) rather than as bands — which is the
/// same thing the pass's tap count buys, for the same reason.
const PAD_STOPS: usize = 32;

/// Where a spread is drawn, as a radius in the field (px).
///
/// **Square-root, not linear**, and this is the one place the pad is a scale rather
/// than a picture. The interesting spreads are the small ones — a fringe of two or
/// three canvas px is a lens, thirty is an effect — and `SPREAD`'s ceiling is 128, so
/// a linear pad would spend nine tenths of its radius on settings nobody dials and
/// leave "2" and "3" a pixel apart. Equal *area* per unit of spread is the standard
/// fix and it costs nothing in honesty, because the law is stated by the rings and
/// the number itself is in the readout: `d(spread)/d(radius)` goes to zero at the
/// centre, so the fine end is the fine end.
fn pad_radius(spread: f32) -> f32 {
    PAD_R * (spread / ChromaticAberration::SPREAD.1).max(0.0).sqrt()
}

/// The inverse: the spread a handle at radius `r` means.
///
/// Unclamped, for [`dial_ab`]'s reason — [`Filter::sanitized`] is the single place a
/// number is held to its range, and a second opinion here is how a control comes to
/// disagree with the value it displays. Past the rim a drag simply pins at 128.
fn pad_spread(r: f32) -> f32 {
    let t = r / PAD_R;
    ChromaticAberration::SPREAD.1 * t * t
}

/// Where the blue end of the fringe is drawn, in the field's px.
///
/// `+y` is **down**, unlike the dial's plane: this is a canvas direction, and the
/// canvas is drawn on the screen with `y` down. So the bar in the pad runs the way the
/// fringe runs in the painting, which is the whole claim the picture makes.
fn pad_xy(c: ChromaticAberration) -> (f32, f32) {
    let r = pad_radius(c.spread);
    (PAD_MID + r * c.angle.cos(), PAD_MID + r * c.angle.sin())
}

/// The dispersion spectrum as SVG gradient stops: `(offset, css color)` from the red
/// end at 0 to the blue end at 1.
///
/// **The pass's own weights, at the pass's own wavelengths** (§21.10) — this is what
/// makes the bar a statement about the render rather than a rainbow someone drew. Two
/// honest adjustments, both about *showing* a response rather than integrating one:
/// the run is normalized so its strongest channel is full intensity (the weights are a
/// response, and their absolute scale means nothing — the pass divides them out too),
/// and the result is encoded to sRGB, because that is what a screen takes. The
/// relative brightness along the run survives both, so the deep ends read dark exactly
/// as the eye finds them.
///
/// A `LazyLock` for [`DIAL_FIELD`]'s reason: it is a fixed function of nothing, so a
/// filter selection costs no arithmetic.
static PAD_SPECTRUM: LazyLock<Vec<(f32, String)>> = LazyLock::new(|| {
    let weights: Vec<[f32; 3]> = (0..PAD_STOPS)
        .map(|i| dispersion_weight(i as f32 / (PAD_STOPS - 1) as f32))
        .collect();
    let peak = weights
        .iter()
        .flatten()
        .copied()
        .fold(0.0f32, f32::max)
        .max(1e-6);
    weights
        .iter()
        .enumerate()
        .map(|(i, w)| {
            let q = |v: f32| (linear_to_srgb(v / peak).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            (
                i as f32 / (PAD_STOPS - 1) as f32,
                format!("#{:02x}{:02x}{:02x}", q(w[0]), q(w[1]), q(w[2])),
            )
        })
        .collect()
});

/// One pointer sample on the dispersion pad: the pointer *is* the blue end.
///
/// Absolute rather than by delta, like the dial's centre and for the same reason — a
/// single click anywhere in the field is a complete edit, and there is no handle to
/// miss, because the one handle's position is the whole of what is being set. Previews
/// and settles through the same funnel everything else here does (§21.6).
fn drag_fringe(
    state: AppState,
    id: LayerId,
    c: ChromaticAberration,
    tuning: Signal<Option<(LayerId, Filter)>>,
    e: &Event<PointerData>,
) {
    let p = e.element_coordinates();
    let (dx, dy) = (p.x as f32 - PAD_MID, p.y as f32 - PAD_MID);
    let r = dx.hypot(dy);
    // At the centre a direction does not exist and `atan2` would answer 0, snapping the
    // axis back to horizontal every time a drag crossed the middle. Keep the axis the
    // gesture already had and move the spread alone — `drag_dial`'s hue does the same,
    // and it is the same fact about polar coordinates.
    let angle = if r > 1e-4 { dy.atan2(dx) } else { c.angle };
    let spread = pad_spread(r);
    let step = e.modifiers().contains(Modifiers::SHIFT);
    let next = Filter::Chromatic(ChromaticAberration {
        spread: if step {
            snapped(spread, SPREAD_STEP)
        } else {
            spread
        },
        angle: if step {
            snapped(angle, ANGLE_STEP)
        } else {
            angle
        },
    })
    .sanitized();
    preview::FILTER.during(state, tuning, (id, next));
}

/// The chromatic filter's `spread` and `angle` as the one thing they are: the vector
/// the spectrum is pulled apart along (see the module docs).
///
/// Everything drawn is a claim about the render, checkable by eye against the fringe
/// the canvas grows:
///
/// - the **bar** is the spectrum — the pass's own weights at the pass's own
///   wavelengths, red end to blue end, painted along the axis they part on;
/// - its **length** is the spread, which is the full width of the fringe an edge
///   grows;
/// - the **handle** is where the blue end lands, and the bar's other tip is the red;
/// - the **centre** is the picture, which does not move: the two ends part around it.
///
/// So the drag is the effect: pull the rainbow out of the middle and turn it, and the
/// painting does what the pad just did.
fn fringe_pad(
    state: AppState,
    id: LayerId,
    c: ChromaticAberration,
    tuning: Signal<Option<(LayerId, Filter)>>,
    mut pulling: Signal<bool>,
) -> Element {
    let (bx, by) = pad_xy(c);
    // The red end: the blue end reflected in the centre, because the spread is
    // symmetric — the fringe parts around the picture rather than dragging it.
    let (rx, ry) = (2.0 * PAD_MID - bx, 2.0 * PAD_MID - by);
    // A gradient over a zero-length line paints nothing, and at spread 0 the two ends
    // are the same point. Nothing is drawn there either, which is the correct picture
    // of "no wavelength moves" — but the *stops* still have to describe a real line,
    // so the bar is simply omitted rather than degenerating.
    let dispersed = c.spread > 0.0;
    let commit = move |_| preview::FILTER.settle(state, tuning);

    rsx! {
        div { class: "filter-fringe",
            div {
                class: "fringe-field",
                title: "The dispersion, as the vector it is. Drag to pull the spectrum \
                        apart \u{2014} how far out is the width of the fringe every \
                        edge grows, and which way is the axis it parts on, stated on \
                        the canvas so it turns with the painting. The dot is where the \
                        blue end lands and the far tip is the red; the picture itself \
                        stays in the middle. Hold Shift for round steps.",
                // Pointer capture, as the dial's field takes: the drag keeps reporting
                // once it leaves the box, and what it reports is held to the stops by
                // the sanitizer rather than by the element's edge.
                onpointerdown: move |e| {
                    capture_pointer(&e);
                    pulling.set(true);
                    drag_fringe(state, id, c, tuning, &e);
                },
                onpointermove: move |e| {
                    if pulling() {
                        drag_fringe(state, id, c, tuning, &e);
                    }
                },
                onpointerup: move |e| { pulling.set(false); commit(e); },
                onpointercancel: move |e| { pulling.set(false); commit(e); },
                svg {
                    class: "fringe-svg",
                    width: "{PAD_PX}",
                    height: "{PAD_PX}",
                    view_box: "0 0 {PAD_PX} {PAD_PX}",
                    defs {
                        linearGradient {
                            id: "fringe-spectrum",
                            gradient_units: "userSpaceOnUse",
                            x1: "{rx}", y1: "{ry}", x2: "{bx}", y2: "{by}",
                            for (at, hex) in PAD_SPECTRUM.iter() {
                                stop { key: "{at}", offset: "{at}", stop_color: "{hex}" }
                            }
                        }
                    }
                    // The graduation, and the stop at the end of it. Both faint and
                    // both always up: unlike the dial's bound they are not a stop that
                    // only matters in hand — they are what says the radius is a
                    // square root, and a picture whose scale appeared only while it
                    // was being dragged would be read wrong at rest.
                    for spread in PAD_RINGS {
                        circle {
                            key: "{spread}",
                            class: "fringe-grid",
                            cx: "{PAD_MID}", cy: "{PAD_MID}", r: "{pad_radius(*spread)}",
                        }
                    }
                    circle {
                        class: "fringe-bound",
                        cx: "{PAD_MID}", cy: "{PAD_MID}", r: "{PAD_R}",
                    }
                    // The rings' spreads, in hand only — at rest the ladder is enough
                    // to say "this is a scale", and three numbers standing over the
                    // spectrum the rest of the time would be three numbers in the way.
                    if pulling() {
                        for spread in PAD_RINGS.iter().chain([&ChromaticAberration::SPREAD.1]) {
                            text {
                                key: "{spread}",
                                class: "fringe-tick",
                                x: "{PAD_MID}",
                                y: "{PAD_MID - pad_radius(*spread)}",
                                "{spread}"
                            }
                        }
                    }
                    if dispersed {
                        // Drawn twice through one gradient: a wide soft pass under a
                        // narrow bright one, which is what a fringe *is* — light, not
                        // a line. Cheaper than an SVG blur filter and, unlike one, it
                        // costs nothing at spread 0 because neither is drawn.
                        line {
                            class: "fringe-glow",
                            x1: "{rx}", y1: "{ry}", x2: "{bx}", y2: "{by}",
                            stroke: "url(#fringe-spectrum)",
                        }
                        line {
                            class: "fringe-bar",
                            x1: "{rx}", y1: "{ry}", x2: "{bx}", y2: "{by}",
                            stroke: "url(#fringe-spectrum)",
                        }
                    }
                    // Where the picture stays. Small and always up, because "the two
                    // ends part around this" is the claim that separates this filter
                    // from the three shifted copies everyone else draws.
                    circle { class: "fringe-origin", cx: "{PAD_MID}", cy: "{PAD_MID}", r: "1.5" }
                    circle {
                        class: "fringe-handle",
                        cx: "{bx}", cy: "{by}", r: "{PAD_HANDLE}",
                    }
                }
            }
            div { class: "dial-readout",
                DialRow {
                    name: "Spread",
                    value: format!("{:.1} px", c.spread),
                    hint: "How far the spectrum is pulled apart, in canvas pixels \
                           \u{2014} the width of the fringe every edge grows. The \
                           whole rainbow is in between, not three offset copies: the \
                           effect is the lens's own dispersion, integrated \
                           (\u{a7}21.10).",
                }
                DialRow {
                    name: "Axis",
                    value: fmt_degrees(c.angle * DEG),
                    hint: "The direction the blue end of the spectrum is carried, with \
                           the red end opposite. Stated on the canvas, so it turns \
                           with the painting rather than with the window.",
                }
            }
        }
    }
}

// —— the gradient map ————————————————————————————————————————————————————————

/// The gradient map's controls (§21.11): the ramp it wears, and the one edit a
/// ramp affords in place.
///
/// Deliberately thin, because the *choosing* already has a home: the library
/// pop-out flies out of the ramp strip itself ([`GradientWell`], §22.3), and
/// while this bar is up a row click there hands the clicked ramp to this filter
/// ([`apply_ramp`]). So the bar shows the ramp in the library's own strip (the
/// same `in oklab` CSS, which is the same interpolation the pass runs — §22.3's
/// invariant doing its job a third time), opens on an empty well while there is
/// none, and offers **Reverse** — the one edit that is about the *mapping*
/// rather than the ramp: a trace runs in whatever direction the hand drew, and
/// the map reads dark at 0, so a ramp captured light-to-dark is one click from
/// meaning what was meant instead of re-tracing backwards.
fn map_rows(state: AppState, id: LayerId, ramp: Option<Gradient>) -> Element {
    match ramp {
        None => rsx! {
            span {
                class: "filter-inert",
                title: "A gradient map repaints what is beneath it with a ramp \
                        indexed by lightness \u{2014} dark paint takes the ramp's \
                        start, light paint its end. Click the well beside this to \
                        choose the ramp, or trace one off the canvas there.",
                "no ramp yet"
            }
            span { class: "bar-sep" }
            GradientWell {
                title: "No ramp yet \u{2014} click to pick one, or trace one off the canvas",
            }
        },
        Some(g) => {
            let strip = crate::gradients::css_strip(&g);
            rsx! {
                GradientWell {
                    strip,
                    title: "The ramp, dark paint's end to the left \u{2014} click \
                            to pick another or trace a new one",
                }
                span { class: "bar-sep" }
                button {
                    class: "chip",
                    title: "Run the ramp the other way \u{2014} what dark paint \
                            takes trades places with what light paint takes",
                    onclick: move |_| {
                        dispatch(
                            state,
                            DocCommand::SetFilter(id, Filter::GradientMap(Some(g.reversed()))),
                        );
                    },
                    {icon(icons::SWAP)}
                    {label("Reverse")}
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
    let tuning = use_signal(|| None::<(LayerId, Filter)>);
    // The dial's half of the same story: which handle a drag has hold of, `None`
    // between drags. Here rather than inside [`chroma_dial`] for the reason above —
    // the dial is mounted for one kind of filter, and a hook that runs for one kind
    // of filter is a hook that runs sometimes.
    let grabbed = use_signal(|| None::<Grab>);
    // And the dispersion pad's: whether its one handle is in hand. A `bool` rather
    // than a second [`Grab`], because the pad has one thing to grab and an enum of one
    // variant would be a bool that took longer to read. Declared here for the reason
    // above.
    let pulling = use_signal(|| false);
    let Some((info, filter)) = use_selected_filter(state)() else {
        return rsx! {};
    };
    // A filter with nothing composited beneath it in its own stack does nothing at
    // all (§21.2) — the foot of a stack, or a stack whose lower layers are all
    // hidden or empty. Said once, in the bar, rather than greying out the sliders,
    // which would each have to explain the same thing.
    let inert = !info.has_underlay;
    // While a mode is composing, this bar recedes with the other standing bars
    // (MODAL_DESIGN.md) — a trace armed from this bar's own gradient well is the
    // live case: the bar is the place the capture returns to, and its sliders
    // must not be pressable under a catcher that owns the pointer.
    let composing = crate::modes::composing(state).is_some();

    // The whole-filter facts the chrome needs, read before the match consumes the
    // filter — its `Clone` is spent on the arms, not on the label.
    let bar_label = filter.label();
    let at_neutral = filter.is_neutral();
    let neutral = filter.neutral();

    // The one place the bar knows the kinds apart: which controls it puts up. A
    // picture each — the color filter's plane ahead of the two tracks that move along
    // the axis a plane has nothing to say about, the chromatic filter's vector, which
    // is the whole of it, and the gradient map's ramp, likewise.
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
        Filter::Chromatic(c) => fringe_pad(state, info.id, c, tuning, pulling),
        Filter::GradientMap(g) => map_rows(state, info.id, g),
    };

    rsx! {
        div {
            class: chrome_class(
                state,
                if composing { "filter-bar recessed" } else { "filter-bar" },
            ),
            // The glyph rides the bar's *label*, as the frame bar's crop marks do:
            // no single slider here is "the filter", so what the mark identifies is
            // the bar, and through it the layer you are tuning.
            span { class: "bar-label",
                {icon(icons::FILTER)}
                {label(bar_label)}
            }

            span { class: "bar-sep" }

            if inert {
                span {
                    class: "filter-inert",
                    title: "A filter adjusts what is composited below it in its own \
                            stack. Move it above a layer with something visible on \
                            it and it will have something to work on.",
                    "nothing below it"
                }

                span { class: "bar-sep" }
            }

            {rows}

            span { class: "bar-sep" }

            button {
                class: "chip",
                title: "Put every slider back to neutral \u{2014} the filter stays, \
                        doing nothing, until it is dialled again",
                disabled: at_neutral,
                onclick: move |_| {
                    dispatch(state, DocCommand::SetFilter(info.id, neutral.clone()));
                },
                {icon(icons::RESET)}
                {label("Neutral")}
            }
            button {
                class: "chip",
                // Esc performs this same act (`commands`' ladder) — advertised
                // through the registry, the frame bar's reason.
                title: commands::advertised(
                    "Stop tuning and go back to painting \u{2014} the filter stays",
                    Command::CancelMode,
                    &state.bindings.read(),
                ),
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
