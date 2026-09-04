//! The eyedropper's options, in a floating bar (§18.0.2).
//!
//! A bar rather than a panel, and mounted only while the eyedropper's chord is
//! held (Alt by default — the drag table's row, `crate::drags`), on the same
//! argument the selection and frame bars are mounted only while the thing they act
//! on exists: the eyedropper is not a tool you switch to — the chord over the brush
//! *is* the binding, as in Clip Studio Paint and Rebelle — so it has no resting
//! state for a panel to occupy. Coming up on the modifier also makes the binding
//! discoverable, which is the standing problem with modifier bindings: press it and
//! the options appear beside the cursor you are about to sample with.
//!
//! It goes away again the moment the drag starts, because from then on the thing to
//! look at is the canvas and the color coming off it.

use dioxus::prelude::*;

use crate::icons::{icon, label};
use crate::layout::chrome_dimmed;
use crate::panels::select::current_tool;
use crate::state::AppState;
use crate::widgets::CommandButton;
use stark_ui::commands::Command;
use stark_ui::commands::PickScope;
use stark_ui::drags::DragAction;

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
    // Armed, but not yet in use — the drag table's own answer to the held
    // modifiers, so the bar comes up on whatever chord the pick actually wears.
    // `canvas_active` covers a stroke or a pan already in hand — the chord pressed
    // mid-stroke must not pop a bar up over the painting — and `dragging` covers
    // the pick itself, which deliberately leaves `canvas_active` alone so the
    // Color panel stays legible while sampling.
    let armed = stark_ui::drags::armed(&state.drags.read(), (state.held_mods)())
        == Some(DragAction::PickColor)
        && !(state.pick.dragging)()
        && !(state.canvas_active)()
        && !(state.space_down)()
        && !current_tool(state).is_selection();
    if !armed {
        return rsx! {};
    }

    let mut radius = state.pick.radius;
    let mut group_only = state.pick.group_only;
    let (r, grouped) = (radius(), group_only());
    let chip = |on: bool| if on { "chip active" } else { "chip" };

    rsx! {
        div {
            class: "pick-bar chrome",
            class: if chrome_dimmed(state) { "dimmed" },
            // The tool Alt has just armed, drawn as well as named. This bar exists to
            // make a modifier binding discoverable (see the module docs), and a picture
            // of the eyedropper appearing beside the cursor is the shortest version of
            // that argument the bar can make.
            span { class: "bar-label",
                {icon(stark_ui::icons::EYEDROPPER)}
                {label("Eyedropper")}
            }

            span { class: "bar-sep" }

            // Each chip is its command worn whole (`crate::commands`): the
            // mark, the terse word, the sentence in the tooltip, and which one
            // is lit are the registry's, so the row cannot describe a reach
            // differently from the palette. Which is also what buys these three
            // a keyboard — Alt+Q / Alt+A / Alt+Z reach the same act the chip
            // does, held under the very modifier that raised this bar, and each
            // chip advertises the key that reaches it. `PickScope::ALL` is the
            // ordering: how much each one lets in, one layer to all of them, so
            // the row reads as one question — *how far does this sample see* —
            // rather than as three unrelated buttons.
            div {
                class: "segmented",
                for want in PickScope::ALL {
                    CommandButton { key: "{want:?}", command: Command::SetPickScope(want) }
                }
            }

            // The fence the reach runs inside — a toggle beside the row rather
            // than a fourth position in it, because it composes with every
            // reach instead of competing with them. On, only the selected
            // layer's group answers; off, the whole document does, and the
            // canvas color stands behind it — the canvas is a fact about the
            // picture, not about any group of paint, so it arrives exactly when
            // the fence comes down.
            button {
                class: chip(grouped),
                title: "Only the selected layer's group answers \u{2014} its \
                        siblings and the layer carrying them. Turn off to sample \
                        the whole document over the canvas color, so bare canvas \
                        and thin paint answer with what the eye sees",
                onclick: move |_| group_only.set(!grouped),
                {icon(stark_ui::icons::GROUP_ONLY)}
                {label("Group")}
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
