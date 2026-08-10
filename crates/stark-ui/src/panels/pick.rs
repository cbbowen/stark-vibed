//! The eyedropper's options, in a floating bar (§18.0.2).
//!
//! A bar rather than a panel, and mounted only while Alt is held, on the same
//! argument the selection and frame bars are mounted only while the thing they act
//! on exists: the eyedropper is not a tool you switch to — Alt over the brush *is*
//! the binding, as in Clip Studio Paint and Rebelle — so it has no resting state for
//! a panel to occupy. Coming up on the modifier also makes the binding
//! discoverable, which is the standing problem with modifier bindings: press Alt and
//! the options appear beside the cursor you are about to sample with.
//!
//! It goes away again the moment the drag starts, because from then on the thing to
//! look at is the canvas and the colour coming off it.

use dioxus::prelude::*;

use crate::icons::{self, icon, label};
use crate::layout::chrome_class;
use crate::panels::select::current_tool;
use crate::state::{AppState, PickScope};

/// The eyedropper's sampled patch, as the prior art names it: a point sample, or the
/// mean of an N×N square around it. A radius is what the engine takes
/// (`PickOptions::radius`), and `2r + 1` is the square it describes.
const PATCHES: [(&str, u32); 4] = [
    ("Point", 0),
    ("3\u{00D7}3", 1),
    ("5\u{00D7}5", 2),
    ("11\u{00D7}11", 5),
];

/// What a sample comes off, as `(scope, icon, label, what it means)`.
///
/// Ordered by how much each one lets in — one layer, then the whole stack, then the
/// stack over the ground it was painted on — so the row reads as one question, *how
/// far does this sample see*, rather than as three unrelated buttons. The default
/// sits where that ordering puts it rather than at the head of the row, which is what
/// an ordering worth having costs.
const SOURCES: [(PickScope, &str, &str, &str); 3] = [
    (
        PickScope::ThisLayer,
        icons::ONE_LAYER,
        "This layer",
        "Sample the selected layer alone, ignoring anything over or under it",
    ),
    (
        PickScope::AllLayers,
        icons::ALL_LAYERS,
        "All layers",
        "Sample the colour the canvas shows, through every visible layer",
    ),
    (
        PickScope::AllLayersAndCanvas,
        icons::ALL_LAYERS_AND_CANVAS,
        "+ Canvas",
        "Sample every visible layer over the canvas colour, so bare canvas and thin \
         paint answer with what the eye sees rather than nothing",
    ),
];

#[component]
pub fn PickBar() -> Element {
    let state = use_context::<AppState>();
    // Armed, but not yet in use. `canvas_active` covers a stroke or a pan already in
    // hand — Alt pressed mid-stroke must not pop a bar up over the painting — and
    // `dragging` covers the pick itself, which deliberately leaves `canvas_active`
    // alone so the Color panel stays legible while sampling.
    let armed = (state.pick.alt_down)()
        && !(state.pick.dragging)()
        && !(state.canvas_active)()
        && !(state.space_down)()
        && !current_tool(state).is_selection();
    if !armed {
        return rsx! {};
    }

    let (mut scope, mut radius) = (state.pick.scope, state.pick.radius);
    let (source, r) = (scope(), radius());
    let chip = |on: bool| if on { "chip active" } else { "chip" };

    rsx! {
        div { class: chrome_class(state, "pick-bar"),
            // The tool Alt has just armed, drawn as well as named. This bar exists to
            // make a modifier binding discoverable (see the module docs), and a picture
            // of the eyedropper appearing beside the cursor is the shortest version of
            // that argument the bar can make.
            span { class: "bar-label",
                {icon(icons::EYEDROPPER)}
                {label("Eyedropper")}
            }

            span { class: "bar-sep" }

            div {
                class: "segmented",
                for (want, glyph, name, why) in SOURCES {
                    button {
                        class: chip(source == want),
                        title: why,
                        onclick: move |_| scope.set(want),
                        {icon(glyph)}
                        {label(name)}
                    }
                }
            }

            span { class: "bar-sep" }

            div {
                class: "segmented",
                for (label, want) in PATCHES {
                    button {
                        class: chip(r == want),
                        title: "How much canvas one sample averages",
                        onclick: move |_| radius.set(want),
                        "{label}"
                    }
                }
            }
        }
    }
}
