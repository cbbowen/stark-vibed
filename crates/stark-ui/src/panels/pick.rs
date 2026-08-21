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

use crate::drags::{self, DragAction};
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

/// How far a sample sees, as `(scope, icon, label, what it means)`.
///
/// Ordered by how much each one lets in — one layer, the layers beneath it too,
/// then every layer — so the row reads as one question, *how far does this sample
/// see*, rather than as three unrelated buttons. The default sits where that
/// ordering puts it rather than at the head of the row, which is what an ordering
/// worth having costs. What the reach runs *through* — the selected layer's group,
/// or the whole document over the canvas color — is the Group chip beside the row
/// (§18.0.2).
const SOURCES: [(PickScope, &str, &str, &str); 3] = [
    (
        PickScope::ThisLayer,
        icons::ONE_LAYER,
        "This layer",
        "Sample the selected layer alone, ignoring anything over or under it",
    ),
    (
        PickScope::AndBelow,
        icons::AND_BELOW,
        "+ Below",
        "Sample the selected layer and everything beneath it \u{2014} what the \
         canvas would show with the layers above switched off",
    ),
    (
        PickScope::AllLayers,
        icons::ALL_LAYERS,
        "All layers",
        "Sample every visible layer \u{2014} the color the canvas shows",
    ),
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
    let armed = drags::armed(&state.drags.read(), (state.held_mods)())
        == Some(DragAction::PickColor)
        && !(state.pick.dragging)()
        && !(state.canvas_active)()
        && !(state.space_down)()
        && !current_tool(state).is_selection();
    if !armed {
        return rsx! {};
    }

    let (mut scope, mut radius) = (state.pick.scope, state.pick.radius);
    let mut group_only = state.pick.group_only;
    let (source, r, grouped) = (scope(), radius(), group_only());
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
                {icon(icons::GROUP_ONLY)}
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
