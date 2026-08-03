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

use crate::icons::{self, icon};
use crate::layout::chrome_class;
use crate::panels::select::current_tool;
use crate::state::AppState;

/// The eyedropper's sampled patch, as the prior art names it: a point sample, or the
/// mean of an N×N square around it. A radius is what the engine takes
/// (`PickOptions::radius`), and `2r + 1` is the square it describes.
const PATCHES: [(&str, u32); 4] = [
    ("Point", 0),
    ("3\u{00D7}3", 1),
    ("5\u{00D7}5", 2),
    ("11\u{00D7}11", 5),
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

    let (mut all_layers, mut radius) = (state.pick.all_layers, state.pick.radius);
    let (all, r) = (all_layers(), radius());
    let chip = |on: bool| if on { "chip active" } else { "chip" };

    rsx! {
        div { class: chrome_class(state, "pick-bar"),
            // The tool Alt has just armed, drawn as well as named. This bar exists to
            // make a modifier binding discoverable (see the module docs), and a picture
            // of the eyedropper appearing beside the cursor is the shortest version of
            // that argument the bar can make.
            span { class: "bar-label",
                {icon(icons::EYEDROPPER)}
                "Eyedropper"
            }
            button {
                class: chip(all),
                title: "Sample the colour the canvas shows, through every visible layer",
                onclick: move |_| all_layers.set(true),
                "All layers"
            }
            button {
                class: chip(!all),
                title: "Sample the selected layer alone, ignoring anything over or under it",
                onclick: move |_| all_layers.set(false),
                "This layer"
            }

            span { class: "bar-sep" }

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
