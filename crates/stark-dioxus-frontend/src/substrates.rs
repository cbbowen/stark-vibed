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
//! fetch before it can open, and in the Lighting panel's gallery
//! ([`SubstrateGallery`](crate::panels::substrates::SubstrateGallery)), whose cards have
//! no pictures until the maps it asks for on the way in have landed.
//!
//! The chrome — the well, the gallery and the New Document dialog — is
//! `crate::panels::substrates` and `crate::panels::new_document`; what is here is the
//! catalog, resolving a pick to an id, and opening a document on one.
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
use dioxus::prelude::*;
use stark_engine::command::DocCommand;
use stark_model::SubstrateId;

use crate::library;
use crate::render::Renderer;
use crate::state::{AppState, dispatch};
use stark_model::ColorSpaceId;
use stark_ui::assets::{self, Pick, Substrates};
use stark_ui::desk::{Desk, Replacement};

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
    if let Some(id) = r.desk.builtin_substrate(name) {
        return Some(id);
    }
    if is_procedural(name) {
        return Some(SubstrateId::Flat);
    }
    let bytes = fetch(name).await?;
    load(&mut r.desk, name, &bytes)
}

/// Load a shipped substrate's fetched height map into `desk` under `name`, logging a
/// refusal — `None` for one.
fn load(desk: &mut Desk, name: &'static str, png: &[u8]) -> Option<SubstrateId> {
    desk.load_builtin_substrate(name, png)
        .inspect_err(|e| tracing::warn!("canvas substrate failed to import: {e}"))
        .ok()
}

/// Open a fresh document on `pick`, in `color_space`.
///
/// Fetch *then* open, because the id is what the replacement is given and the id
/// comes out of the image. A substrate that cannot be fetched leaves the document on
/// `Flat` — smooth, and honestly so, rather than claiming a substrate it has not got.
pub async fn open_default(r: &mut Renderer, color_space: stark_model::ColorSpaceId) {
    let surface = resolve(r, assets::DEFAULT_SUBSTRATE)
        .await
        .unwrap_or_default();
    // Nobody can have touched the document it replaces: the renderer is unpublished.
    if let Err(e) = r.desk.replace(Replacement::New(color_space, surface), &[]) {
        // The engine keeps the document it was built with, so startup carries on.
        tracing::error!(
            ?color_space,
            ?surface,
            "cannot open the first document: {e}"
        );
    }
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
        .and_then(|r| r.desk.builtin_substrate(name));
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
        crate::state::with_engine_quiet(state, |r| load(&mut r.desk, name, &bytes)).flatten();
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
        .and_then(|r| r.desk.engine().substrate_bytes(id));
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
/// its own list is exhaustive (the Lighting panel's gallery draws that case as its own
/// card, `panels::substrates`).
pub fn resolved(state: AppState) -> Vec<(&'static assets::Shipped, Option<SubstrateId>)> {
    crate::shipped::watch(state);
    let renderer = state.renderer.peek();
    assets::SHIPPED_SUBSTRATES
        .iter()
        .map(|g| {
            let id = match g.path {
                None => Some(SubstrateId::Flat),
                Some(_) => renderer
                    .as_ref()
                    .and_then(|r| r.desk.builtin_substrate(g.name)),
            };
            (g, id)
        })
        .collect()
}

/// Replace the document with a fresh one in the chosen color space, on the chosen
/// substrate, then repaint. A bundled substrate's height map is fetched on first use (the
/// large bump maps stay out of the wasm binary — §6.6), so this runs async: `pick` is
/// a name or an id and what the replacement needs is the resolved `SubstrateId`.
///
/// It owns closing the modal, once the document is replaced; otherwise it clears
/// `pending` and leaves the dialog up. A scope-tied `spawn` from the modal's own
/// handler, so the task is the dialog instance's: dismissing it during the fetch
/// cancels the new document before anything has changed, and a dialog reopened
/// meanwhile is a new instance this task cannot close.
pub fn new_document(state: AppState, color: ColorSpaceId, pick: Pick, mut pending: Signal<bool>) {
    spawn(async move {
        // A substrate that will not fetch opens the document smooth rather than
        // refusing to open it — and the document then honestly *says* it is smooth
        // instead of claiming a substrate it hasn't got.
        let surface = resolve_signal(state, pick).await;
        match crate::state::replace_document(state, Replacement::New(color, surface), &[]) {
            Some(Ok(())) => {
                tracing::info!(?color, ?pick, ?surface, "new document ready");
                crate::dialogs::close(state, crate::dialogs::DialogId::NewDocument);
            }
            // Unreachable from the cards, which offer only the spaces this build
            // carries; the dialog has nowhere to say it.
            Some(Err(e)) => {
                tracing::error!(
                    ?color,
                    ?surface,
                    "cannot start a document in this space: {e}"
                );
                pending.set(false);
            }
            None => pending.set(false),
        }
    });
}
