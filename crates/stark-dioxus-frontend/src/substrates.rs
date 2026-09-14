//! The canvas substrates: the ones that ship with the app, and the ones a user brings
//! (§6.4).
//!
//! A built-in substrate is a grayscale height map under `assets/substrate/`, bundled as
//! a static file and **fetched at runtime** so several megabytes of substrate stay out
//! of the wasm binary. The engine keys a substrate by the hash of its decoded height
//! field, so every engine that imports the same file lands on the same
//! [`SubstrateId`] — which is what lets a built-in be referenced the way a substrate a
//! peer brought is: by content id, with no notion of "built-in" anywhere
//! downstream.
//!
//! This is `crate::builtins` for substrates instead of brush shapes, deliberately so.
//! Both are frontend-held images the engine names by content, and the catch is the
//! same in both: **an id is only knowable once the bytes have arrived**, so nothing
//! that runs before the fetch can name one. For shapes that shows up as presets
//! being seeded after startup; here it shows up in [`open_default`], which has to
//! fetch before it can open, and in [`SubstrateGallery`], whose cards have no pictures
//! until the maps it asks for on the way in have landed.
//!
//! Substrates are on this footing because a *name* — `Linen`, `Rough` — is only as good
//! as the table the reader holds, and that diverges collaborators (§6.4, §12.4): a
//! peer who has never picked Rough receives `SetSubstrate(Rough)`, has no bytes for it,
//! silently falls back to the flat stand-in, and from then on bakes every stroke with
//! no deposition tooth at all. A content id cannot go quietly missing that way. The
//! holder either has those exact bytes or knows precisely what to ask a peer for, and
//! what comes back is verified against the id that asked for it.
//!
//! **Adding a built-in substrate is a file plus a row in [`assets::SHIPPED_SUBSTRATES`]**,
//! and its `asset!` row in `crate::shipped` — it then appears in the Lighting panel's
//! picker and in the New-document dialog.
//!
//! # The user's own substrates
//!
//! And that is the whole reason a user can bring one at all. Because the engine names
//! a substrate by content, "built-in" is a fact about *this frontend's asset list* and
//! about nothing downstream: a substrate imported from a file is the same kind of thing as
//! `Linen`, saved into a document the same way, bundled into a save file the same way,
//! and fetched by a peer the same way. There is no second mechanism here — only a
//! second place the bytes come from.
//!
//! So the library is the brush stamps' library with substrates in it, down to the
//! storage — rows in `localStorage`, height maps in the blob store beside them (§25.6)
//! — and it is written once, in `crate::library`. The one real difference between the
//! two is decided below this frontend (`stark_ui::assets::substrate_png`): a stamp's
//! polarity is a spelling and a substrate's polarity *is the substrate*, so nothing
//! inverts one.
//!
//! [`Pick`] is what the two halves have in common at the picker: a built-in is chosen
//! by catalog name because its id is not knowable until its map has been fetched, and
//! a user's substrate is chosen by the id the library already holds. Everything past
//! [`resolve_signal`] deals in ids alone.

use dioxus::dioxus_core::spawn_forever;
use dioxus::html::HasFileData;
use dioxus::prelude::*;
use stark_engine::command::DocCommand;
use stark_model::{AssetId, SubstrateId};

use crate::library;
use crate::render::Renderer;
use crate::state::{AppState, dispatch, use_obs};
use crate::widgets::Modal;
use stark_model::ColorSpaceId;
use stark_ui::assets::{self, Pick, Substrates};

// --- resolving and switching ------------------------------------------------

/// The catalog row for `name`.
fn substrate(name: &str) -> Option<&'static assets::Shipped> {
    assets::SHIPPED_SUBSTRATES.iter().find(|g| g.name == name)
}

/// The height map behind a bundled substrate. `None` for a procedural substrate (which
/// has no image) or a fetch that failed — callers treat both as "nothing to
/// import" rather than as an error to surface.
///
/// Split out from the import so the two halves can be sequenced differently by
/// callers that own a `Renderer` and callers that reach it through a signal: this
/// is the awaiting half, and it borrows nothing.
async fn fetch(name: &str) -> Option<Vec<u8>> {
    crate::shipped::fetch_bytes(substrate(name)?.path?).await
}

/// Whether `name` needs no image: `Flat` is its own id, known without a fetch.
fn is_procedural(name: &str) -> bool {
    substrate(name).is_some_and(|g| g.path.is_none())
}

/// Resolve a pick to its content id, fetching and importing a bundled height map the
/// first time it is asked for.
///
/// For callers holding a `Renderer` outright — app startup and the brush editor's
/// preview. Repeat asks cost nothing: the id is remembered once known.
pub async fn resolve(r: &mut Renderer, pick: Pick) -> Option<SubstrateId> {
    let Pick::Builtin(name) = pick else {
        // A user's substrate is named by an id it already has; there is nothing to fetch
        // and nothing this half can add. `resolve_signal` is the path that can install
        // it, because the library lives on `AppState` and a bare `Renderer` has none.
        return None;
    };
    if let Some(id) = r.builtin_substrate(name) {
        return Some(id);
    }
    if is_procedural(name) {
        return Some(SubstrateId::Flat);
    }
    let bytes = fetch(name).await?;
    r.load_substrate(name, &bytes)
}

/// Open a fresh document on `pick`, in `color_space`.
///
/// Fetch *then* open, because the id is what `new_document` is given and the id
/// comes out of the image. A substrate that cannot be fetched leaves the document on
/// `Flat` — smooth, and honestly so, rather than claiming a substrate it has not got.
pub async fn open_default(r: &mut Renderer, color_space: stark_model::ColorSpaceId) {
    let surface = resolve(r, assets::DEFAULT_SUBSTRATE)
        .await
        .unwrap_or_default();
    r.new_document(color_space, surface);
}

/// [`resolve`], for callers that reach the engine through [`AppState`] rather than
/// holding one — and the only path that can resolve a [`Pick::Custom`], since the
/// library lives on the state.
///
/// The engine is held only for the synchronous halves and never across the fetch —
/// holding a `Signal`'s guard over an `await` is a borrow that outlives every other
/// reader for the duration of a network round trip. Falls back to `Flat` rather than
/// failing: a substrate that will not load leaves a smooth canvas, which the document
/// then records as the truth.
///
/// The import is quiet: it readies a substrate and names it, and the document does not
/// move onto it until the `SetSubstrate` in [`select`] — which is a command, and
/// publishes.
pub async fn resolve_signal(state: AppState, pick: Pick) -> SubstrateId {
    let name = match pick {
        Pick::Custom(id) => {
            return library::ensure::<Substrates>(state, id)
                .map(SubstrateId::Image)
                .unwrap_or_default();
        }
        Pick::Builtin(name) => name,
    };
    let known = state
        .renderer
        .peek()
        .as_ref()
        .and_then(|r| r.builtin_substrate(name));
    if let Some(id) = known {
        return id;
    }
    if is_procedural(name) {
        return SubstrateId::Flat;
    }
    let Some(bytes) = fetch(name).await else {
        return SubstrateId::Flat;
    };
    let landed =
        crate::state::with_engine_quiet(state, |r| r.load_substrate(name, &bytes)).flatten();
    if landed.is_some() {
        crate::shipped::landed(state);
    }
    landed.unwrap_or_default()
}

/// Switch the document's substrate and repaint — the painting is preserved; existing
/// paint re-reads against the new substrate (§6.4). A logged action, so it undoes and
/// replicates like any other edit.
///
/// A bundled height map is fetched on first use, which is why this is async: the id
/// the action carries is the hash of an image that may not be in hand yet.
pub fn select(state: AppState, pick: Pick) {
    // `spawn_forever`: the caller is a panel's scope, and hiding the panel
    // mid-fetch must not cancel the switch (only root-owned signals are touched,
    // so outliving the panel is safe).
    spawn_forever(async move {
        let id = resolve_signal(state, pick).await;
        // One out of the library was offered on its way through `library::ensure`.
        if matches!(pick, Pick::Builtin(_)) {
            seed_session(state, id);
        }
        dispatch(state, DocCommand::SetSubstrate(id));
    });
}

/// [`select`] for a substrate in the library, by the id it is held under — what an
/// import does once the file is in.
fn select_custom(state: AppState, id: AssetId) {
    select(state, Pick::Custom(id));
}

/// Register a shipped substrate with a live session so peers can fetch it by hash. A
/// no-op when solo; idempotent when repeated (content-addressed).
///
/// Called *before* the `SetSubstrate` is dispatched, and the order matters: the
/// broadcast attaches a transfer hash looked up from what has been registered here,
/// so a substrate registered afterwards would go out unfetchable — the peer would see
/// the action, find no hash, and be left on the flat stand-in, which is the very
/// failure this design removes.
fn seed_session(state: AppState, id: SubstrateId) {
    // `Flat` is procedural: no bytes to register, and no peer can be waiting on it.
    let SubstrateId::Image(asset) = id else {
        return;
    };
    // Solo there is nobody to offer it to, and a height map is megabytes to copy out
    // of the engine for nobody.
    if state.collab.session.peek().is_none() {
        return;
    }
    let bytes = state
        .renderer
        .peek()
        .as_ref()
        .and_then(|r| r.substrate_bytes(id));
    if let Some(bytes) = bytes {
        library::seed_session::<Substrates>(state, asset, bytes);
    }
}

/// Every catalog substrate paired with the id it resolved to — `None` for one whose
/// height map has not been fetched. For the pickers, which list them and mark the
/// one the document is on, and which redraw when a fetch lands
/// ([`shipped::watch`](crate::shipped::watch)) rather than on every engine write.
///
/// The library's substrates are not here: those are `state.substrates.entries`, and each
/// already carries the id it is named by. A document may also be on a substrate that is
/// in *neither* — one a peer brought, or one loaded from a file whose substrate was never
/// in this browser's library — which is why a picker asks this rather than assuming
/// its own list is exhaustive ([`SubstrateGallery`] draws that case as its own card).
pub fn resolved(state: AppState) -> Vec<(&'static assets::Shipped, Option<SubstrateId>)> {
    crate::shipped::watch(state);
    let renderer = state.renderer.peek();
    assets::SHIPPED_SUBSTRATES
        .iter()
        .map(|g| {
            let id = match g.path {
                None => Some(SubstrateId::Flat),
                Some(_) => renderer.as_ref().and_then(|r| r.builtin_substrate(g.name)),
            };
            (g, id)
        })
        .collect()
}

// --- the pickers ------------------------------------------------------------

/// **Fetch the bundled substrates' height maps, once, on the way in.**
///
/// A substrate's map is several megabytes and is deliberately not fetched at startup
/// (§6.6) — but a card with no picture is a `select` with extra steps, and a surface
/// being *looked at* is exactly the moment the maps are wanted. So the lazy fetch is
/// kept and its trigger moved to the first moment it pays for itself.
///
/// `use_hook`, so it runs once per mount, and cheap on every mount after the first —
/// [`resolve_signal`] answers from the id already cached. `spawn_forever` because the
/// panel can be closed mid-fetch and the substrate it readies is worth keeping.
/// Whoever asked redraws when a map lands: [`resolved`] watches for one, and
/// [`resolve_signal`] says when one has (`crate::shipped`).
///
/// **Two callers, and they are not the same moment.** [`SubstrateGallery`] asks
/// because it is about to draw every card; [`SubstrateWell`] asks because it cannot
/// so much as *name* the substrate the document is on without it — a built-in is
/// known by content id and its id is the hash of a file that may not be here yet
/// (see [`Pick`]). The well is what the Lighting panel shows at rest, so this is
/// where the fetch now happens for most artists, which is no later than before: the
/// gallery used to stand open in that panel.
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
    matches!(current, SubstrateId::Image(_)).then(|| "From the document".to_string())
}

/// The Lighting panel's surface **well**: the substrate the document is on, as a
/// picture and a name, and the press that flies [`SubstrateGallery`] out beside the
/// panel (`widgets::PopoutId::SubstrateGallery`).
///
/// The gallery used to stand open in that panel, which made choosing a surface cost
/// the artist a grid's worth of column for as long as the panel was open — and the
/// grid is the *rarest* thing in it, chosen between passages rather than during one.
/// A well says which surface is in force in one row and asks for the grid only when
/// somebody wants to change it.
#[component]
pub fn SubstrateWell() -> Element {
    let state = use_context::<AppState>();
    // The well names a built-in, so it needs what names one — see [`use_bundled`].
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
    let well = match crate::widgets::popout_open(state, crate::widgets::PopoutId::SubstrateGallery)
    {
        true => "surface-well open",
        false => "surface-well",
    };
    rsx! {
        button {
            class: well,
            title: "Choose the canvas surface",
            onclick: move |_| crate::widgets::toggle_popout(
                state,
                crate::widgets::PopoutId::SubstrateGallery,
            ),
            // `Smooth` has no height map and so no picture; the class draws the
            // absence rather than leaving an empty box, exactly as its card does.
            div {
                class: if thumb.is_some() { "asset-thumb" } else { "asset-thumb flat" },
                style: crate::cards::thumb_style(thumb.as_deref()),
            }
            // The name is blank for the one moment it is unknown rather than
            // guessed at: a built-in whose map has not landed is a substrate this
            // browser cannot yet name, and "Smooth" would be a lie about the
            // picture beside it.
            div { class: "asset-name", "{name}" }
        }
    }
}

/// The Lighting panel's surface picker: every bundled substrate, every substrate in the
/// user's library (thumbnail + name, with a hover ✕ to remove), the substrate the
/// document is on when it is in neither list, and an import card. Images can also be
/// dropped anywhere on the grid.
///
/// The brush editor's `ShapeGallery` with substrates in it — deliberately the same
/// picture, because they are the same kind of thing (see the module note). What it
/// does *not* borrow is the flat card's stand-in: `Smooth` is a real substrate with a
/// real id, so it is an ordinary row of [`assets::SHIPPED_SUBSTRATES`] rather than a special case.
#[component]
pub fn SubstrateGallery() -> Element {
    let state = use_context::<AppState>();
    let mut dropping = use_signal(|| false);

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
    // A substrate in neither list — one a peer brought, or one that came in with a file
    // — still has to be *shown*, or the gallery would claim the document is on
    // whichever card happened to sort first. The engine has its bytes (it is
    // depositing through them), so it gets a real picture; switching away from it is
    // an ordinary pick, and switching back is not offered, which is honest — it is
    // not in the library and this browser cannot produce it again.
    let stray = match current {
        SubstrateId::Image(id)
            if !catalog.iter().any(|(_, c)| *c == Some(current))
                && !thumbs().iter().any(|(e, ..)| *e == id) =>
        {
            Some((id, library::thumbnail::<Substrates>(state, id)))
        }
        _ => None,
    };

    // One row per bundled substrate, flattened to what a card draws *before* the rsx.
    // A `for` whose body is anything but a plain element loses its `key` to positional
    // diffing, which is how a gallery ends up with one card wearing another's picture
    // — the trap `crate::shapes`' gallery is written around too.
    let builtins = catalog.into_iter().map(move |(g, id)| {
        let thumb = match id {
            Some(SubstrateId::Image(a)) => library::thumbnail::<Substrates>(state, a),
            _ => None,
        };
        (g.name, g.blurb, thumb, id == Some(current))
    });

    let card = |active: bool| {
        if active {
            "asset-card selected"
        } else {
            "asset-card"
        }
    };

    rsx! {
        div {
            class: if dropping() { "asset-grid dropping" } else { "asset-grid" },
            // `preventDefault` on dragover is what makes the element a drop target at
            // all; the class is just the highlight. And `stopPropagation`, which is
            // what claims the drop: the app root takes every drop the window sees and
            // places what it gets as a *picture* (§23.4), so a substrate dropped here has
            // to say it is a different act (`ShapeGallery` carries the same note).
            ondragover: move |e| {
                e.prevent_default();
                e.stop_propagation();
                dropping.set(true);
            },
            ondragleave: move |_| dropping.set(false),
            ondrop: move |e| {
                e.prevent_default();
                e.stop_propagation();
                dropping.set(false);
                library::import_dropped::<Substrates>(state, e.files(), select_custom);
            },

            for (name, blurb, thumb, active) in builtins {
                div {
                    key: "{name}",
                    class: card(active),
                    title: "{blurb}",
                    onclick: move |_| select(state, Pick::Builtin(name)),
                    // `Smooth` has no height map and so no picture; the class draws
                    // the absence rather than leaving an empty box.
                    div {
                        class: if thumb.is_some() { "asset-thumb" } else { "asset-thumb flat" },
                        style: crate::cards::thumb_style(thumb.as_deref()),
                    }
                    div { class: "asset-name", "{name}" }
                }
            }
            for (id, name, url) in thumbs() {
                div {
                    key: "{id.to_hex()}",
                    class: card(current == SubstrateId::Image(id)),
                    onclick: move |_| select(state, Pick::Custom(id)),
                    div { class: "asset-thumb", style: crate::cards::thumb_style(url.as_deref()) }
                    div { class: "asset-name", title: "{name}", "{name}" }
                    button {
                        class: "asset-remove",
                        title: "Remove from library",
                        onclick: move |e| {
                            e.stop_propagation();
                            library::remove::<Substrates>(state, id);
                        },
                        {crate::icons::icon(stark_ui::icons::REMOVE)}
                    }
                }
            }
            if let Some((id, url)) = stray {
                div {
                    key: "{id.to_hex()}",
                    class: "asset-card selected",
                    title: "A surface this document came with — not in your library.",
                    div { class: "asset-thumb", style: crate::cards::thumb_style(url.as_deref()) }
                    div { class: "asset-name", "From the document" }
                }
            }
            div { class: "asset-card import",
                // `pick_file` must run inside the click gesture — no task hop.
                onclick: move |_| {
                    crate::platform::pick_file("image/*", move |name, bytes| {
                        library::import_file::<Substrates>(state, name, bytes, select_custom);
                    });
                },
                div { class: "asset-thumb plus", {crate::icons::icon(stark_ui::icons::ADD)} }
                div { class: "asset-name", "Import\u{2026}" }
            }
        }
        if let Some(notice) = (state.substrates.notice)() {
            div { class: "asset-notice", "{notice}" }
        }
    }
}

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

    // Same card, for the canvas substrate choice — one row per pick, whichever half it
    // came from, so a substrate the artist imported is offered here exactly as a bundled
    // one is.
    let scard = |pick: Pick, title: String, desc: String| {
        let class = if surf_choice() == pick {
            "space-card selected"
        } else {
            "space-card"
        };
        rsx! {
            div {
                class,
                onclick: move |_| { let mut c = surf_choice; c.set(pick); },
                div { class: "space-card-title", "{title}" }
                div { class: "space-card-desc", "{desc}" }
            }
        }
    };

    // Every surface on offer, as a row: the bundled ones and then the library's,
    // which is the order the Lighting panel's gallery uses too.
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
                {scard(pick, title, desc)}
            }

            div { class: "modal-actions",
                button {
                    class: "btn btn-secondary",
                    onclick: move |_| on_close.call(()),
                    "Cancel"
                }
                button {
                    class: "btn btn-primary",
                    onclick: move |_| new_document(state, choice(), surf_choice()),
                    "Create"
                }
            }
        }
    }
}

/// Replace the document with a fresh one in the chosen color space, on the chosen
/// substrate, then repaint. A bundled substrate's height map is fetched on first use (the
/// large bump maps stay out of the wasm binary — §6.6), so this runs async: `pick` is
/// a name or an id and what `new_document` needs is the resolved `SubstrateId`.
///
/// It owns closing the modal, once the work is done. A scope-tied `spawn` from the
/// modal's own handler, so the task is the dialog instance's: dismissing it during the
/// fetch cancels the new document before anything has changed, and a dialog reopened
/// meanwhile is a new instance this task cannot close.
fn new_document(state: AppState, color: ColorSpaceId, pick: Pick) {
    spawn(async move {
        // A substrate that will not fetch opens the document smooth rather than
        // refusing to open it — and the document then honestly *says* it is smooth
        // instead of claiming a substrate it hasn't got.
        let surface = resolve_signal(state, pick).await;
        // After the fetch, so a dismissal during it leaves the session alone too.
        // Replacing the document abandons any shared session (and clears the ticket
        // from the URL) — the fresh canvas is private until re-shared.
        crate::collab::leave(state);
        // Framing a fresh document leaves the view alone: nothing is painted and
        // nothing framed, so there is no piece to show yet.
        crate::state::replace_document(state, |r| {
            r.new_document(color, surface);
            Ok(())
        });
        tracing::info!(?color, ?pick, ?surface, "new document ready");
        crate::dialogs::close(state, crate::dialogs::DialogId::NewDocument);
    });
}
