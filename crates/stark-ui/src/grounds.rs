//! The canvas grounds that ship with the app (§6.4).
//!
//! A built-in ground is a grayscale height map under `assets/surface/`, bundled as
//! a static file and **fetched at runtime** so several megabytes of weave stay out
//! of the wasm binary. The engine keys a ground by the hash of its decoded height
//! field, so every engine that imports the same file lands on the same
//! [`SurfaceId`] — which is what lets a built-in be referenced the way a ground a
//! peer brought is: by content id, with no notion of "built-in" anywhere
//! downstream.
//!
//! This is `crate::builtins` for grounds instead of brush shapes, deliberately so.
//! Both are frontend-held images the engine names by content, and the catch is the
//! same in both: **an id is only knowable once the bytes have arrived**, so nothing
//! that runs before the fetch can name one. For shapes that shows up as presets
//! being seeded after startup; here it shows up as the opening document being on
//! `Flat` until linen lands ([`open_default`]).
//!
//! Grounds are on this footing because a *name* — `Linen`, `Gesso` — is only as good
//! as the table the reader holds, and that diverges collaborators (§6.4, §12.4): a
//! peer who has never picked Gesso receives `SetSurface(Gesso)`, has no bytes for it,
//! silently falls back to the flat stand-in, and from then on bakes every stroke with
//! no deposition tooth at all. A content id cannot go quietly missing that way. The
//! holder either has those exact bytes or knows precisely what to ask a peer for, and
//! what comes back is verified against the id that asked for it.
//!
//! **Adding a ground is a file plus a row in [`GROUNDS`]** — it then appears in the
//! Lighting panel's picker and in the New-document dialog.

use dioxus::dioxus_core::spawn_forever;
use dioxus::prelude::*;
use stark_engine::command::DocCommand;
use stark_model::SurfaceId;

use crate::render::Renderer;
use crate::state::{AppState, dispatch};

/// One ground selectable in the UI.
pub struct BuiltinGround {
    /// The picker's label, and the name the rest of the app asks for it by. Not
    /// persisted anywhere — a document stores the resolved content id — so
    /// renaming one is a cosmetic change.
    pub name: &'static str,
    /// The bundled height map, fetched at runtime. `None` for a ground that is
    /// procedural and needs no image, which is `Smooth` and only ever `Smooth`.
    pub asset: Option<Asset>,
    /// The same file's path under `assets/`, which is how `crate::builtin_ids`
    /// knows this ground's content id without fetching it. Spelled twice because
    /// `asset!` needs a literal and a lookup needs a string; a test checks the two
    /// agree. `None` exactly when `asset` is.
    pub path: Option<&'static str>,
    /// The picker's one-line description.
    pub blurb: &'static str,
}

pub const SMOOTH: &str = "Smooth";
pub const LINEN: &str = "Linen";
pub const GESSO: &str = "Gesso";

/// Every ground that ships with the app, in picker order.
pub const GROUNDS: &[BuiltinGround] = &[
    BuiltinGround {
        name: SMOOTH,
        asset: None,
        path: None,
        blurb: "A perfectly smooth surface — paint lies flat, no canvas texture.",
    },
    BuiltinGround {
        name: LINEN,
        asset: Some(asset!("/assets/surface/Linen.png")),
        path: Some("surface/Linen.png"),
        blurb: "A regular woven grid — the honest painter's canvas.",
    },
    BuiltinGround {
        name: GESSO,
        asset: Some(asset!("/assets/surface/Gesso.png")),
        path: Some("surface/Gesso.png"),
        blurb: "Brushed acrylic ground: irregular knife strokes with a fine crackle.",
    },
];

/// The ground a fresh document opens on. Linen — the weave is what paint has to sit
/// in, and its relief is read by the stroke pass whether or not the lighting shows
/// it, so it is the honest starting substrate and `Smooth` is the deliberate switch
/// away.
///
/// That opinion lives here rather than in the engine because the engine embeds no
/// image bytes and a ground now *is* its bytes: core naming linen would be core
/// naming an image it cannot produce. `document::DEFAULT_SURFACE` is `Flat`
/// accordingly, and [`open_default`] is what moves a starting document off it.
pub const DEFAULT_GROUND: &str = SMOOTH;

/// The catalog row for `name`.
fn ground(name: &str) -> Option<&'static BuiltinGround> {
    GROUNDS.iter().find(|g| g.name == name)
}

/// The height map behind a bundled ground. `None` for a procedural ground (which
/// has no image) or a fetch that failed — callers treat both as "nothing to
/// import" rather than as an error to surface.
///
/// Split out from the import so the two halves can be sequenced differently by
/// callers that own a `Renderer` and callers that reach it through a signal: this
/// is the awaiting half, and it borrows nothing.
async fn fetch(name: &str) -> Option<Vec<u8>> {
    let asset = ground(name)?.asset?;
    tracing::info!(ground = name, url = %asset, "fetching canvas ground");
    match dioxus::asset_resolver::read_asset_bytes(asset).await {
        Ok(bytes) => Some(bytes),
        Err(e) => {
            tracing::warn!("could not fetch the canvas ground “{name}”: {e}");
            None
        }
    }
}

/// Whether `name` needs no image: `Flat` is its own id, known without a fetch.
fn is_procedural(name: &str) -> bool {
    ground(name).is_some_and(|g| g.asset.is_none())
}

/// Resolve a bundled ground to its content id, fetching and importing its height
/// map the first time it is asked for.
///
/// For callers holding a `Renderer` outright — app startup and the brush editor's
/// preview. Repeat asks cost nothing: the id is remembered once known.
pub async fn resolve(r: &mut Renderer, name: &'static str) -> Option<SurfaceId> {
    if let Some(id) = r.ground(name) {
        return Some(id);
    }
    if is_procedural(name) {
        return Some(SurfaceId::Flat);
    }
    let bytes = fetch(name).await?;
    r.load_ground(name, &bytes)
}

/// Open a fresh document on a bundled ground, in `color_space`.
///
/// Fetch *then* open, because the id is what `new_document` is given and the id
/// comes out of the image. A ground that cannot be fetched leaves the document on
/// `Flat` — smooth, and honestly so, rather than claiming a weave it has not got.
pub async fn open_default(r: &mut Renderer, color_space: stark_model::ColorSpaceId) {
    let surface = resolve(r, DEFAULT_GROUND).await.unwrap_or_default();
    r.new_document(color_space, surface);
}

/// [`resolve`], for callers that reach the engine through [`AppState`] rather than
/// holding one.
///
/// The engine is held only for the synchronous halves and never across the fetch —
/// holding a `Signal`'s guard over an `await` is a borrow that outlives every other
/// reader for the duration of a network round trip. Falls back to `Flat` rather than
/// failing: a ground that will not load leaves a smooth canvas, which the document
/// then records as the truth.
///
/// The import is quiet: it readies a ground and names it, and the document does not
/// move onto it until the `SetSurface` in [`select`] — which is a command, and
/// publishes.
pub async fn resolve_signal(state: AppState, name: &'static str) -> SurfaceId {
    let known = state.renderer.peek().as_ref().and_then(|r| r.ground(name));
    if let Some(id) = known {
        return id;
    }
    if is_procedural(name) {
        return SurfaceId::Flat;
    }
    let Some(bytes) = fetch(name).await else {
        return SurfaceId::Flat;
    };
    crate::state::with_engine_quiet(state, |r| r.load_ground(name, &bytes))
        .flatten()
        .unwrap_or_default()
}

/// Switch the document's ground and repaint — the painting is preserved; existing
/// paint re-reads against the new weave (§6.4). A logged action, so it undoes and
/// replicates like any other edit.
///
/// The height map is fetched on first use, which is why this is async: the id the
/// action carries is the hash of an image that may not be in hand yet.
pub fn select(state: AppState, name: &'static str) {
    // `spawn_forever`: the caller is a panel's scope, and hiding the panel
    // mid-fetch must not cancel the switch (only root-owned signals are touched,
    // so outliving the panel is safe).
    spawn_forever(async move {
        let id = resolve_signal(state, name).await;
        seed_session(state, id);
        dispatch(state, DocCommand::SetSurface(id));
    });
}

/// Register a ground with a live session so peers can fetch it by hash. A no-op
/// when solo; idempotent when repeated (content-addressed).
///
/// Called *before* the `SetSurface` is dispatched, and the order matters: the
/// broadcast attaches a transfer hash looked up from what has been registered here,
/// so a ground registered afterwards would go out unfetchable — the peer would see
/// the action, find no hash, and be left on the flat stand-in, which is the very
/// failure this design removes.
fn seed_session(state: AppState, id: SurfaceId) {
    let Some(broadcaster) = state
        .collab
        .session
        .read()
        .as_ref()
        .map(|s| s.broadcaster())
    else {
        return;
    };
    let bytes = state
        .renderer
        .peek()
        .as_ref()
        .and_then(|r| r.surface_bytes(id));
    // `Flat` is procedural: no bytes to register, and no peer can be waiting on it.
    if let (Some(bytes), Some(need)) = (bytes, stark_net::AssetNeed::ground(id)) {
        broadcaster.add_content(need, bytes);
    }
}

/// Every catalog ground paired with the id it resolved to — `None` for one whose
/// height map has not been fetched. For the pickers, which list them and mark the
/// one the document is on; `read`, so a row settles when its fetch lands.
///
/// A document may be on a ground that is in no catalog — one a peer brought — which
/// is why a picker asks this rather than assuming its own list is exhaustive.
pub fn resolved(state: AppState) -> Vec<(&'static BuiltinGround, Option<SurfaceId>)> {
    let renderer = state.renderer.read();
    GROUNDS
        .iter()
        .map(|g| {
            let id = match g.asset {
                None => Some(SurfaceId::Flat),
                Some(_) => renderer.as_ref().and_then(|r| r.ground(g.name)),
            };
            (g, id)
        })
        .collect()
}
