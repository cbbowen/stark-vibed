//! The canvas surface's chrome in the Lighting panel (§6.4): the well that names the
//! substrate in force, and the gallery it flies out. Which substrates exist and how a
//! pick becomes an id are `crate::substrates`'.

use dioxus::dioxus_core::spawn_forever;
use dioxus::prelude::*;
use stark_model::{AssetId, SubstrateId};

use crate::library;
use crate::state::{AppState, use_obs};
use crate::substrates::{resolve_signal, resolved, select};
use crate::widgets::{AssetCard, AssetGallery, PopoutId, Thumb, popout_open, toggle_popout};
use stark_ui::assets::{self, Pick, Substrates};

/// **Fetch the bundled substrates' height maps, once, on the way in.**
///
/// A map is several megabytes and deliberately not fetched at startup (§6.6), but a
/// surface being *looked at* is the moment the maps are wanted: [`SubstrateGallery`] is
/// about to draw every card, and [`SubstrateWell`] cannot so much as name a built-in
/// without its id, which is the hash of a file that may not be here yet (see [`Pick`]).
///
/// `use_hook`, so it runs once per mount and costs nothing after the first —
/// [`resolve_signal`] answers from the cached id. `spawn_forever`, because the panel can
/// be closed mid-fetch and the substrate it readies is worth keeping. Whoever asked
/// redraws when a map lands ([`resolved`] watches for one).
fn use_bundled(state: AppState) {
    use_hook(|| {
        spawn_forever(async move {
            for g in assets::SHIPPED_SUBSTRATES {
                resolve_signal(state, Pick::Builtin(g.name)).await;
            }
        });
    });
}

/// What the document's substrate is called, for the one control that has room for a
/// name and not for a grid.
///
/// Three places to look, in the order a substrate can be known: this build's catalog,
/// this browser's library, and neither — a substrate a peer brought or that came in
/// with a file, which [`SubstrateGallery`] draws as its own card and names the same
/// way here. `None` while a built-in's map is still in flight, which is the one
/// moment the app genuinely does not know the answer.
fn current_name(state: AppState, current: SubstrateId) -> Option<String> {
    if let Some((g, _)) = resolved(state).iter().find(|(_, id)| *id == Some(current)) {
        return Some(g.name.to_string());
    }
    if let SubstrateId::Image(asset) = current
        && let Some(entry) = state
            .substrates
            .entries
            .read()
            .iter()
            .find(|e| e.id == asset)
    {
        return Some(entry.name.clone());
    }
    matches!(current, SubstrateId::Image(_)).then(|| STRAY_NAME.to_string())
}

/// What a substrate in neither the catalog nor the library is called.
const STRAY_NAME: &str = "From the document";

/// [`select`] for a substrate in the library, by the id it is held under — what an
/// import does once the file is in.
fn select_custom(state: AppState, id: AssetId) {
    select(state, Pick::Custom(id));
}

/// The Lighting panel's surface **well**: the substrate the document is on, as a
/// picture and a name, and the press that flies [`SubstrateGallery`] out beside the
/// panel (`PopoutId::SubstrateGallery`).
///
/// A well says which surface is in force in one row and asks for the grid only when
/// somebody wants to change it; the grid is the rarest choice in the panel, made
/// between passages rather than during one.
#[component]
pub fn SubstrateWell() -> Element {
    let state = use_context::<AppState>();
    use_bundled(state);
    // Through a memo (`state::use_obs`), like the gallery's own highlight: the well
    // moves when a substrate is chosen and at no other time.
    let current = use_obs(state, |o| o.substrate)().unwrap_or_default();
    let name = current_name(state, current).unwrap_or_default();
    let thumb = match current {
        SubstrateId::Image(id) => library::thumbnail::<Substrates>(state, id),
        SubstrateId::Flat => None,
    };
    // Lit while the gallery it opened is standing beside the panel, for the swatch
    // above it's reason (`panels::lighting`).
    let well = if popout_open(state, PopoutId::SubstrateGallery) {
        "surface-well open"
    } else {
        "surface-well"
    };
    rsx! {
        button {
            class: well,
            title: "Choose the canvas surface",
            onclick: move |_| toggle_popout(state, PopoutId::SubstrateGallery),
            // `Smooth` has no height map and so no picture; the class draws the
            // absence rather than leaving an empty box, exactly as its card does.
            div {
                class: if thumb.is_some() { "asset-thumb" } else { "asset-thumb flat" },
                style: thumb.as_deref().map(|url| crate::cards::thumb_style(Some(url))),
            }
            // Blank for the one moment the name is unknown rather than guessed at: a
            // built-in whose map has not landed is a substrate this browser cannot yet
            // name, and "Smooth" would be a lie about the picture beside it.
            div { class: "asset-name", "{name}" }
        }
    }
}

/// The Lighting panel's surface picker: every bundled substrate, every substrate in the
/// user's library, the substrate the document is on when it is in neither list, and an
/// import card.
///
/// The brush editor's shape gallery with substrates in it, because they are the same
/// kind of thing (`crate::substrates`). `Smooth` is a real substrate with a real id, so
/// it is an ordinary row of [`assets::SHIPPED_SUBSTRATES`] rather than a round-tip card.
#[component]
pub fn SubstrateGallery() -> Element {
    let state = use_context::<AppState>();
    use_bundled(state);

    // Which card wears the selected ring, through a memo (`state::use_obs`): it moves
    // when a substrate is chosen and at no other time, while the projection behind it
    // moves on every command.
    let current = use_obs(state, |o| o.substrate)().unwrap_or_default();
    let catalog = resolved(state);
    let entries = state.substrates.entries;
    // Memoized so the list is rebuilt when the library changes rather than on every
    // obs refresh; the encode behind each url is itself remembered per content id, so
    // a card that survives a rebuild costs a scan.
    let thumbs = use_memo(move || {
        entries
            .read()
            .iter()
            .map(|e| {
                (
                    e.id,
                    e.name.clone(),
                    library::thumbnail::<Substrates>(state, e.id),
                )
            })
            .collect::<Vec<_>>()
    });
    let owned = thumbs();

    // A substrate in neither list — one a peer brought, or one that came in with a file
    // — still has to be *shown*, or the gallery would claim the document is on
    // whichever card happened to sort first. The engine has its bytes, so it gets a
    // real picture; it cannot be picked back, which is honest, since this browser
    // cannot produce it again.
    let stray = match current {
        SubstrateId::Image(id)
            if !catalog.iter().any(|(_, c)| *c == Some(current))
                && !owned.iter().any(|(e, ..)| *e == id) =>
        {
            Some(AssetCard {
                key: format!("stray:{}", id.to_hex()),
                name: STRAY_NAME.to_string(),
                thumb: Thumb::Picture(library::thumbnail::<Substrates>(state, id)),
                selected: true,
                blurb: Some("A surface this document came with — not in your library."),
                pick: None,
                remove: None,
            })
        }
        _ => None,
    };
    let builtins = catalog.into_iter().map(|(g, id)| {
        let thumb = match id {
            Some(SubstrateId::Image(a)) => library::thumbnail::<Substrates>(state, a),
            _ => None,
        };
        AssetCard {
            key: format!("builtin:{}", g.name),
            name: g.name.to_string(),
            // `Smooth` has no height map and so no picture; the flat ground stands in.
            thumb: thumb.map_or(Thumb::Flat, |url| Thumb::Picture(Some(url))),
            selected: id == Some(current),
            blurb: Some(g.blurb),
            pick: Some(Pick::Builtin(g.name)),
            remove: None,
        }
    });
    let library_cards = owned.into_iter().map(|(id, name, url)| AssetCard {
        key: format!("library:{}", id.to_hex()),
        name,
        thumb: Thumb::Picture(url),
        selected: current == SubstrateId::Image(id),
        blurb: None,
        pick: Some(Pick::Custom(id)),
        remove: Some(id),
    });
    let cards: Vec<_> = builtins.chain(library_cards).chain(stray).collect();

    rsx! {
        AssetGallery {
            cards,
            notice: (state.substrates.notice)(),
            onpick: move |pick| select(state, pick),
            onremove: move |id| library::remove::<Substrates>(state, id),
            onimport: move |()| {
                crate::platform::pick_file("image/*", move |name, bytes| {
                    library::import_file::<Substrates>(state, name, bytes, select_custom);
                });
            },
            ondrop: move |files| library::import_dropped::<Substrates>(state, files, select_custom),
        }
    }
}
