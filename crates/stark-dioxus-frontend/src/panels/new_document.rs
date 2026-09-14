//! The New Document dialog (§6.7, §6.4): the color space and the canvas surface a fresh
//! document opens on. Opening it is `crate::substrates::new_document`'s.

use dioxus::prelude::*;
use stark_model::{ColorSpaceId, SubstrateId};

use crate::state::{AppState, use_obs};
use crate::substrates::{new_document, resolved};
use crate::widgets::Modal;
use stark_ui::assets::{self, Pick};

/// Modal for starting a fresh document. Today it carries the color-space choice
/// (§6.7); it's a dialog so more document settings can join it later.
#[component]
pub fn NewDocumentModal(on_close: EventHandler<()>) -> Element {
    let state = use_context::<AppState>();
    // Off the projection, not off the renderer. Both facts are in `obs`, and
    // reading the renderer signal in a render body subscribes the dialog to every
    // engine write — so it re-rendered on every command for the whole time it was
    // open, to re-seed two `use_signal`s that are seeded once (U9, and `PeerCursors`
    // carries the same warning).
    let document = use_obs(state, |o| (o.color_space, o.substrate));
    let (current, current_surface) = match document() {
        Some((space, surface)) => (space, Some(surface)),
        None => (ColorSpaceId::Oklab, None),
    };
    let choice = use_signal(|| current);

    // The substrate is chosen as a `Pick`, not as an id: a bundled one's id is the hash
    // of a height map, so it is not knowable until that map has been fetched — and
    // this dialog runs before any of them have (§6.4). A substrate from the library
    // already has its id, which is exactly the asymmetry `Pick` exists to carry. Both
    // are resolved at Create, once the bytes are in hand.
    let library = state.substrates.entries;
    let current_pick = resolved(state)
        .into_iter()
        .find(|(_, id)| *id == current_surface)
        .map(|(g, _)| Pick::Builtin(g.name))
        .or_else(|| match current_surface {
            Some(SubstrateId::Image(id)) => library
                .read()
                .iter()
                .any(|e| e.id == id)
                .then_some(Pick::Custom(id)),
            _ => None,
        })
        .unwrap_or(assets::DEFAULT_SUBSTRATE);
    let surf_choice = use_signal(|| current_pick);
    // Set while a Create is running: a second one would leave the session twice and
    // rebuild the GPU state twice.
    let mut pending = use_signal(|| false);

    // One selectable color-space card; `selected` toggles the highlight.
    let card = |id: ColorSpaceId, title: &str, desc: &str| {
        let class = if choice() == id {
            "space-card selected"
        } else {
            "space-card"
        };
        rsx! {
            div {
                class,
                onclick: move |_| { let mut choice = choice; choice.set(id); },
                div { class: "space-card-title", "{title}" }
                div { class: "space-card-desc", "{desc}" }
            }
        }
    };

    // Every surface on offer, as a row: the bundled ones and then the library's,
    // which is the order the Lighting panel's gallery uses too — so a substrate the
    // artist imported is offered here exactly as a bundled one is.
    //
    // A memo, as the gallery's own list is — it hands back an owned `Vec`, so no
    // `Signal` guard is held open across a loop body that runs arbitrary render code,
    // and the dialog wakes when the library changes rather than on every read of it.
    let surfaces = use_memo(move || {
        assets::SHIPPED_SUBSTRATES
            .iter()
            .map(|g| {
                (
                    Pick::Builtin(g.name),
                    g.name.to_string(),
                    g.blurb.to_string(),
                )
            })
            .chain(library.read().iter().map(|e| {
                (
                    Pick::Custom(e.id),
                    e.name.clone(),
                    "A surface you imported.".to_string(),
                )
            }))
            .collect::<Vec<_>>()
    });

    rsx! {
        Modal { on_close,
            div { class: "modal-title", "New Document" }
            div { class: "modal-subtitle", "Starting a new document replaces the current canvas." }

            div { class: "modal-section-label", "COLOR SPACE" }
            {card(ColorSpaceId::Oklab, "Oklab", "Perceptual color with smooth, predictable blending. The standard choice for digital painting.")}
            // Offered only where the engine carries it. `ColorSpaceId::Mixbox` is
            // a variant in every build — the save format's enum indices cannot
            // depend on a feature (§8) — so the id below still compiles; what a
            // build without the `mixbox` feature lacks is the space behind it, and
            // `ColorSpaceId::available` is the same question this asks.
            {cfg!(feature = "mixbox").then(|| card(ColorSpaceId::Mixbox, "Mixbox", "Realistic pigment mixing (Mixbox): blue + yellow makes green, like real paint. For natural media."))}

            div { class: "modal-section-label", "SURFACE" }
            for (pick, title, desc) in surfaces() {
                div {
                    key: "{pick:?}",
                    class: if surf_choice() == pick { "space-card selected" } else { "space-card" },
                    onclick: move |_| { let mut c = surf_choice; c.set(pick); },
                    div { class: "space-card-title", "{title}" }
                    div { class: "space-card-desc", "{desc}" }
                }
            }

            div { class: "modal-actions",
                button {
                    class: "btn btn-secondary",
                    onclick: move |_| on_close.call(()),
                    "Cancel"
                }
                button {
                    class: "btn btn-primary",
                    disabled: pending(),
                    onclick: move |_| {
                        // Checked here too: a second click can be handled before the
                        // render that disables the button.
                        if pending.replace(true) {
                            return;
                        }
                        new_document(state, choice(), surf_choice(), pending);
                    },
                    "Create"
                }
            }
        }
    }
}
