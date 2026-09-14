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

/// The eyedropper's signals (§18.0.2), grouped because they are one
/// feature's worth of view state: the two options a sample is taken with, the latch
/// that keeps a picking drag from asking for samples faster than the GPU answers
/// them, and the flag that says the drag is under way. (Whether the eyedropper is
/// *armed* is no longer a flag of its own: it is the drag table's answer to the
/// modifiers currently held — `stark_ui::drags::armed` over
/// [`Signals::held_mods`](crate::state::Signals::held_mods).)
///
/// The options live here rather than in the engine because nothing in the engine
/// reads them between calls — [`Engine::pick_color`](stark_engine::Engine::pick_color)
/// is a request and they are its arguments, so a copy projected back through
/// `observe()` would be state with no owner.
#[derive(Clone, Copy)]
pub struct PickState {
    /// How far a sample sees: the selected layer, it and what is beneath it, or
    /// every layer.
    pub scope: Signal<stark_ui::commands::PickScope>,
    /// Whether the sample is confined to the selected layer's **group** — its
    /// siblings and the layer carrying them (§14.2). On by default: sampling near
    /// paint usually means sampling the passage being worked, not whatever other
    /// group happens to show through at that point. Off, the whole document
    /// answers, over the canvas color — the canvas is behind the picker exactly
    /// when this is off, since a group is paint and the document is a picture.
    pub group_only: Signal<bool>,
    /// Half-width of the averaged square, in canvas px (0 = point sample).
    pub radius: Signal<u32>,
    /// Whether a sample is in flight — see [`crate::input::pick_color`].
    pub busy: Signal<bool>,
    /// Whether a picking drag is actually sampling. Shared rather than local to
    /// the canvas, unlike `drawing`/`panning`, because the options bar is mounted
    /// on *armed but not yet dragging* and so has to be able to tell the two
    /// apart.
    pub dragging: Signal<bool>,
    /// Where a **held touch** pick is showing its answer, element (CSS) px: the
    /// finger's own position, with the swatch drawn clear of it (§18.1.11).
    ///
    /// `None` for every other way of sampling, and that is the field's content
    /// rather than an oversight. A mouse or a pen puts a cursor on the point it is
    /// asking about and leaves the Color panel in plain view; a finger covers the
    /// point and, on the tablet this gesture exists for, most of the panel with the
    /// hand behind it. The loupe is the answer for the one gesture that cannot
    /// otherwise see one.
    pub loupe: Signal<Option<stark_model::geom::Vec2>>,
}

impl PickState {
    pub(crate) fn new() -> Self {
        use crate::state::root_signal;
        Self {
            scope: root_signal(stark_ui::commands::PickScope::default),
            group_only: root_signal(|| true),
            radius: root_signal(|| 0),
            busy: root_signal(|| false),
            dragging: root_signal(|| false),
            loupe: root_signal(|| None),
        }
    }
}
