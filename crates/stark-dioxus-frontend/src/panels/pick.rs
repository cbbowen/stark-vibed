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
use crate::panels::select::current_tool;
use crate::state::AppState;
use crate::widgets::{Bar, Chip, Choice, CommandButton, Face, Segmented};
use stark_ui::commands::Command;
use stark_ui::commands::PickScope;
use stark_ui::pick::{PATCHES, patch_word};
use strum::VariantArray;

#[component]
pub fn PickBar() -> Element {
    let state = use_context::<AppState>();
    // Armed, but not yet in use — `stark_ui::pick` over the drag table's own answer
    // to the held modifiers, so the bar comes up on whatever chord the pick actually
    // wears and stands down for exactly what the press stands down for.
    // `canvas_active` is the `busy` half: a stroke or a pan already in hand, since a
    // chord pressed mid-stroke must not pop a bar up over the painting. `dragging`
    // is the pick itself, which deliberately leaves `canvas_active` alone so the
    // Color panel stays legible while sampling.
    let hand = crate::input::hand(state, current_tool(state));
    if !hand.shows_options(&state.drags.read(), (state.held_mods)()) {
        return rsx! {};
    }

    let mut radius = state.pick.radius;
    let mut group_only = state.pick.group_only;
    let (r, grouped) = (radius(), group_only());
    let patches: Vec<_> = PATCHES
        .map(|want| {
            Choice::new(
                want,
                Face::Word(patch_word(want).into()),
                "How much canvas one sample averages",
            )
        })
        .into();

    rsx! {
        // The tool Alt has just armed, drawn as well as named: a picture of the
        // eyedropper beside the cursor is the shortest way to make the binding
        // discoverable (see the module docs).
        Bar { class: "pick-bar", glyph: stark_ui::icons::EYEDROPPER, word: "Eyedropper",
            span { class: "bar-sep" }

            // Each chip is its command worn whole (`crate::commands`): the
            // mark, the terse word, the sentence in the tooltip, and which one
            // is lit are the registry's, so the row cannot describe a reach
            // differently from the palette. Which is also what buys these three
            // a keyboard — Alt+Q / Alt+A / Alt+Z reach the same act the chip
            // does, held under the very modifier that raised this bar, and each
            // chip advertises the key that reaches it. `PickScope::VARIANTS` is the
            // ordering: how much each one lets in, one layer to all of them, so
            // the row reads as one question — *how far does this sample see* —
            // rather than as three unrelated buttons.
            div {
                class: "segmented",
                for want in PickScope::VARIANTS.iter().copied() {
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
            Chip {
                active: grouped,
                title: stark_ui::pick::GROUP_TIP,
                onclick: move |_| group_only.set(!grouped),
                {icon(stark_ui::icons::GROUP_ONLY)}
                {label("Group")}
            }

            span { class: "bar-sep" }

            Segmented {
                choices: patches,
                selected: r,
                onpick: move |want: u32| radius.set(want),
            }
        }
    }
}
