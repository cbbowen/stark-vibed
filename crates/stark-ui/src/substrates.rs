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
//! **Adding a built-in substrate is a file plus a row in [`SUBSTRATES`]** — it then appears
//! in the Lighting panel's picker and in the New-document dialog.
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
//! So the library is `crate::shapes` for substrates, down to the storage: rows (a name
//! and an id) in `localStorage`, height maps in the blob store beside them (§25.6),
//! and the shared parts of both in `crate::library`. Import runs through
//! [`normalize_substrate_image`], which is where
//! the one real difference between the two lives — a stamp's polarity is a spelling
//! and a substrate's polarity *is the substrate*, so nothing here inverts anything.
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

use crate::library::{self, Thumbs};
use crate::platform::{base64_encode, normalize_substrate_image};
use crate::render::Renderer;
use crate::state::{AppState, dispatch, use_obs};
use crate::storage::{self, Store};
use crate::widgets::Modal;
use stark_model::ColorSpaceId;

/// One substrate selectable in the UI.
pub struct BuiltinSubstrate {
    /// The picker's label, and the name the rest of the app asks for it by. Not
    /// persisted anywhere — a document stores the resolved content id — so
    /// renaming one is a cosmetic change.
    pub name: &'static str,
    /// The bundled height map, fetched at runtime. `None` for a substrate that is
    /// procedural and needs no image, which is `Smooth` and only ever `Smooth`.
    pub asset: Option<Asset>,
    /// The same file's path under `assets/`, which is how `crate::builtin_ids`
    /// knows this substrate's content id without fetching it. Spelled twice because
    /// `asset!` needs a literal and a lookup needs a string; a test checks the two
    /// agree. `None` exactly when `asset` is.
    pub path: Option<&'static str>,
    /// The picker's one-line description.
    pub blurb: &'static str,
}

pub const SMOOTH: &str = "Smooth";
pub const LINEN: &str = "Linen";
pub const ROUGH: &str = "Rough";

/// Every substrate that ships with the app, in picker order.
pub const SUBSTRATES: &[BuiltinSubstrate] = &[
    BuiltinSubstrate {
        name: SMOOTH,
        asset: None,
        path: None,
        blurb: "A perfectly smooth surface — paint lies flat, no canvas texture.",
    },
    BuiltinSubstrate {
        name: LINEN,
        asset: Some(asset!("/assets/substrate/Linen.png")),
        path: Some("substrate/Linen.png"),
        blurb: "A regular woven grid — the honest painter's canvas.",
    },
    BuiltinSubstrate {
        name: ROUGH,
        asset: Some(asset!("/assets/substrate/Rough.png")),
        path: Some("substrate/Rough.png"),
        blurb: "Brushed acrylic substrate: irregular knife strokes with a fine crackle.",
    },
];

/// The substrate a fresh document opens on: `Smooth`, the one that is procedural.
///
/// The opinion lives here rather than in the engine because the engine embeds no
/// image bytes and a substrate now *is* its bytes: naming a substrate in core would be core
/// naming an image it cannot produce. `document::DEFAULT_SUBSTRATE` is `Flat`
/// accordingly, and [`open_default`] is the hook that would move a starting document
/// off it — which today it does not, because the two agree.
///
/// It is a `Pick` rather than a `SubstrateId` for that same reason, and the shape is
/// what matters more than the value: naming a *woven* default here is a one-word
/// change, and every fetch-then-open it needs is already written.
pub const DEFAULT_SUBSTRATE: Pick = Pick::Builtin(SMOOTH);

/// **What the picker hands back** — the two ways a substrate can be named before it is
/// resolved.
///
/// The asymmetry is real and is the whole reason this is an enum rather than an id:
/// a built-in's id is the hash of a file that may not have been fetched yet, so it
/// can only be named by catalog *name* until then, while a substrate in the library was
/// hashed when it was imported and is named by the id it already has. Both are
/// resolved to a `SubstrateId` by [`resolve_signal`], which is the last place either
/// spelling exists.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Pick {
    /// A substrate this build ships with, by its row in [`SUBSTRATES`].
    Builtin(&'static str),
    /// A substrate the user imported, by the content id the library holds it under.
    Custom(AssetId),
}

// --- the user's library ----------------------------------------------------

/// One imported substrate, **with its bytes in hand**.
///
/// `crate::shapes`'s `ShapeEntry` for substrates, and the same bargain: constructing one
/// means the bytes are here — [`load`] drops a row whose blob is gone rather than
/// admitting a byte-less entry — so nothing downstream has a half-loaded substrate to draw
/// a blank card for.
#[derive(Clone, PartialEq)]
pub struct SubstrateEntry {
    /// Display name, defaulted from the imported file's stem.
    pub name: String,
    /// Canonical grayscale height map (what the engine stores under `id`).
    pub png: Vec<u8>,
    /// Content id of `png` — the `AssetId` inside the `SubstrateId::Image` a document
    /// records. If an engine upgrade ever re-canonicalizes differently, [`ensure`]
    /// heals the entry from the id the import actually returns.
    pub id: AssetId,
}

/// One row of the stored library: **a name and an id, and no bytes at all** — see
/// `crate::shapes`'s `StoredShape`, which this is the sibling of.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct StoredSubstrate {
    name: String,
    #[serde(with = "storage::hex")]
    id: AssetId,
}

impl storage::Entry for StoredSubstrate {
    const STORE: Store = Store::Substrates;
}

impl storage::Blob for SubstrateEntry {
    const STORE: Store = Store::Substrates;
}

/// This library's gallery thumbnails ([`Thumbs`]) — the *height field* each id names.
///
/// Its own cache, not `crate::shapes`': a grayscale PNG canonicalizes to one id under
/// both readings, so one table would hand a substrate the picture of a stamp.
static THUMBS: Thumbs = Thumbs::new();

/// A `background-image` data URL showing the height field `id` names.
///
/// **The field itself, in grey, opaque** — which is what a substrate *is*, and the one
/// place this differs from a stamp's card. A stamp is drawn as white ink over nothing,
/// because coverage says where paint lands and the card must show the panel through
/// the gaps; a substrate has no gaps. Its low substrate is as much a part of it as its high
/// substrate, and drawing the lows transparent would show a canvas full of holes.
///
/// Bytes come from the engine's store first and the library second, on [`ensure`]'s
/// order and for its reason: a built-in is only ever in the engine, and a substrate
/// imported in an earlier session is only ever in the library until it is picked.
/// `None` while a built-in's fetch is still in flight — the same moment its card is
/// not yet clickable — or if the bytes do not decode.
pub fn thumbnail(state: AppState, id: AssetId) -> Option<String> {
    if let Some(url) = THUMBS.get(id) {
        return Some(url);
    }
    let bytes = {
        let renderer = state.renderer;
        let guard = renderer.read();
        guard
            .as_ref()
            .and_then(|r| r.substrate_bytes(SubstrateId::Image(id)))
    }
    .or_else(|| {
        state
            .substrates
            .entries
            .read()
            .iter()
            .find(|e| e.id == id)
            .map(|e| e.png.clone())
    })?;
    let url = encode_thumb(&bytes)?;
    THUMBS.put(id, url.clone());
    Some(url)
}

/// Decode `png` to its height field, reduce it to a thumbnail, and encode that as an
/// opaque grayscale PNG.
///
/// Through `stark_assetid::height`, not through the file's own pixels: the card then
/// shows the field the *tooth will read*, so a source that put its height somewhere
/// the engine does not look shows up as a card that looks wrong rather than as a mark
/// that is.
fn encode_thumb(png: &[u8]) -> Option<String> {
    let stark_assetid::Canonical {
        width,
        height,
        texels,
    } = library::reduce(stark_assetid::height(png).ok()?);
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, width, height);
        encoder.set_color(png::ColorType::Grayscale);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(png::Compression::High);
        let mut writer = encoder.write_header().ok()?;
        writer.write_image_data(&texels).ok()?;
    }
    Some(format!("data:image/png;base64,{}", base64_encode(&out)))
}

/// Populate the library signal from storage. Called once at app start.
///
/// Two reads, because the library is kept in two stores — see `crate::shapes`'s
/// `load`, which this is the sibling of, for the whole of why a row whose bytes are
/// gone is **dropped** rather than kept as a card that draws nothing.
pub async fn load(state: AppState) {
    let rows = storage::load_list::<StoredSubstrate>().unwrap_or_default();
    let ids: Vec<AssetId> = rows.iter().map(|r| r.id).collect();
    let blobs = storage::blob_load_all::<SubstrateEntry>(&ids).await;

    let kept: Vec<SubstrateEntry> = rows
        .into_iter()
        .zip(blobs)
        .filter_map(|(row, png)| {
            png.map(|png| SubstrateEntry {
                name: row.name,
                png,
                id: row.id,
            })
        })
        .collect();
    if kept.len() != ids.len() {
        tracing::warn!(
            "{} surface(s) had lost their height map and were dropped from the library",
            ids.len() - kept.len()
        );
        persist(&kept);
    }

    let mut entries = state.substrates.entries;
    entries.set(kept);
}

/// Import an image file as a new substrate: normalize in the browser, canonicalize in
/// the engine, add to the library, and **switch the document onto it**.
///
/// Switching is the difference from a shape import, and it is not a flourish: a shape
/// joins a gallery you then paint with, while a substrate is the thing you are painting
/// *on*. Importing one and having to click it afterwards would be two acts where the
/// artist made one. The switch goes through [`select`], so it is a logged action like
/// any other pick.
pub fn import_file(state: AppState, file_name: String, bytes: Vec<u8>) {
    let mut notice = state.substrates.notice;
    notice.set(None);
    spawn_forever(async move {
        let png = match normalize_substrate_image(bytes).await {
            Ok(png) => png,
            Err(e) => {
                notice.set(Some(format!("Couldn't import “{file_name}”: {e}.")));
                return;
            }
        };
        // The canonical bytes, not the file's: what the id names, what a save file
        // bundles, and what a peer is served are all one representation (§8, §19).
        //
        // Quiet: readying a substrate names it and changes no document state. What
        // publishes is the `SetSubstrate` in [`select`] below, which is a command.
        let imported = crate::state::with_engine_quiet(state, |r| {
            r.import_substrate(&png)
                .map(|id| (id, r.substrate_bytes(id).unwrap_or(png)))
        })
        .flatten();
        let Some((surface, canonical)) = imported else {
            notice.set(Some(format!(
                "Couldn't import “{file_name}”: it is not a height map this build can read."
            )));
            return;
        };
        let SubstrateId::Image(id) = surface else {
            // Unreachable: `import_substrate` hashes an image, and `Flat` is the one
            // substrate that has no image. Said rather than unwrapped, because what it
            // would mean is that the engine invented a procedural substrate out of bytes.
            notice.set(Some(format!(
                "Couldn't import “{file_name}”: it canonicalized to the flat substrate."
            )));
            return;
        };

        let name = library::display_name(&file_name, "Imported surface");
        let mut entries = state.substrates.entries;
        let known = entries.read().iter().any(|e| e.id == id);
        if !known {
            // Bytes before the row that names them (`storage::blob_save`): the other
            // order can leave a library pointing at a substrate that was never stored.
            storage::blob_save::<SubstrateEntry>(id, &canonical).await;
            entries.write().push(SubstrateEntry {
                name: name.clone(),
                png: canonical,
                id,
            });
            persist(&entries.read());
        }
        select(state, Pick::Custom(id));
        notice.set(known.then(|| format!("“{name}” is already in your library — selected it.")));
    });
}

/// Import files dropped onto the surface gallery. Each file reads and imports
/// independently; when several are dropped the last to finish is the one the document
/// ends on (arbitrary but harmless — every one lands in the library).
pub fn import_dropped(state: AppState, files: Vec<dioxus::html::FileData>) {
    for file in files {
        spawn_forever(async move {
            let name = file.name();
            match file.read_bytes().await {
                Ok(bytes) => import_file(state, name, bytes.to_vec()),
                Err(e) => {
                    let mut notice = state.substrates.notice;
                    notice.set(Some(format!("Couldn't read “{name}”: {e}.")));
                }
            }
        });
    }
}

/// Make sure `id` is usable as a substrate: its bytes in this document's engine,
/// returning the id to reference it by — healed when the stored id predates a
/// canonicalization change. `None` when the bytes are nowhere to be found, the canvas
/// isn't up yet, or the import failed.
///
/// `crate::shapes`'s `ensure` for substrates, and it matters more here: a `SetSubstrate`
/// naming a substrate the engine has no bytes for does not merely look wrong, it bakes a
/// flat deposit into stored tiles that nothing later un-bakes (§6.4).
pub fn ensure(state: AppState, id: AssetId) -> Option<SubstrateId> {
    let surface = SubstrateId::Image(id);
    // Already in this document's engine — imported here, or arrived with a loaded
    // file, or fetched off a peer.
    let held = state
        .renderer
        .peek()
        .as_ref()
        .is_some_and(|r| r.substrate_bytes(surface).is_some());
    if held {
        return Some(surface);
    }

    let entry = state
        .substrates
        .entries
        .read()
        .iter()
        .find(|e| e.id == id)
        .cloned()?;
    let actual =
        crate::state::with_engine_quiet(state, |r| r.import_substrate(&entry.png)).flatten();
    let Some(actual) = actual else {
        let mut notice = state.substrates.notice;
        notice.set(Some(format!("“{}” failed to load.", entry.name)));
        return None;
    };
    if actual != surface
        && let SubstrateId::Image(healed) = actual
    {
        // The stored id predates a canonicalization change; heal it in place, in the
        // order `storage::blob_save` states — bytes under the new name, then the row,
        // then the old name dropped. `crate::shapes::ensure` has the whole argument.
        let mut entries = state.substrates.entries;
        if let Some(e) = entries.write().iter_mut().find(|e| e.id == id) {
            e.id = healed;
        }
        let bytes = entry.png;
        spawn_forever(async move {
            storage::blob_save::<SubstrateEntry>(healed, &bytes).await;
            persist(&entries.read());
            storage::blob_remove::<SubstrateEntry>(id).await;
        });
    }
    Some(actual)
}

/// Drop an entry from the library. Paintings already made on it are untouched — the
/// engine's per-document store keeps every imported substrate, and save files bundle
/// whatever the log names (§8). A document *currently* on the removed substrate stays on
/// it: the substrate is what the painting is on, so taking it out of the picker must not
/// silently repaint the canvas.
pub fn remove(state: AppState, id: AssetId) {
    let mut entries = state.substrates.entries;
    entries.write().retain(|e| e.id != id);
    persist(&entries.read());
    // The row first, then the bytes (`storage::blob_save`): a crash between the two
    // strands some bytes, which costs space; the other order strands the *row*, which
    // costs a substrate that cannot be painted on.
    spawn_forever(async move { storage::blob_remove::<SubstrateEntry>(id).await });
}

// --- resolving and switching ------------------------------------------------

/// The catalog row for `name`.
fn substrate(name: &str) -> Option<&'static BuiltinSubstrate> {
    SUBSTRATES.iter().find(|g| g.name == name)
}

/// The height map behind a bundled substrate. `None` for a procedural substrate (which
/// has no image) or a fetch that failed — callers treat both as "nothing to
/// import" rather than as an error to surface.
///
/// Split out from the import so the two halves can be sequenced differently by
/// callers that own a `Renderer` and callers that reach it through a signal: this
/// is the awaiting half, and it borrows nothing.
async fn fetch(name: &str) -> Option<Vec<u8>> {
    let asset = substrate(name)?.asset?;
    tracing::info!(substrate = name, url = %asset, "fetching canvas substrate");
    match dioxus::asset_resolver::read_asset_bytes(asset).await {
        Ok(bytes) => Some(bytes),
        Err(e) => {
            tracing::warn!("could not fetch the canvas substrate “{name}”: {e}");
            None
        }
    }
}

/// Whether `name` needs no image: `Flat` is its own id, known without a fetch.
fn is_procedural(name: &str) -> bool {
    substrate(name).is_some_and(|g| g.asset.is_none())
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
    let surface = resolve(r, DEFAULT_SUBSTRATE).await.unwrap_or_default();
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
        Pick::Custom(id) => return ensure(state, id).unwrap_or_default(),
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
    crate::state::with_engine_quiet(state, |r| r.load_substrate(name, &bytes))
        .flatten()
        .unwrap_or_default()
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
        seed_session(state, id);
        dispatch(state, DocCommand::SetSubstrate(id));
    });
}

/// Register a substrate with a live session so peers can fetch it by hash. A no-op
/// when solo; idempotent when repeated (content-addressed).
///
/// Called *before* the `SetSubstrate` is dispatched, and the order matters: the
/// broadcast attaches a transfer hash looked up from what has been registered here,
/// so a substrate registered afterwards would go out unfetchable — the peer would see
/// the action, find no hash, and be left on the flat stand-in, which is the very
/// failure this design removes.
fn seed_session(state: AppState, id: SubstrateId) {
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
        .and_then(|r| r.substrate_bytes(id));
    // `Flat` is procedural: no bytes to register, and no peer can be waiting on it.
    if let (Some(bytes), Some(need)) = (bytes, stark_net::AssetNeed::for_substrate(id)) {
        broadcaster.add_content(need, bytes);
    }
}

/// Every catalog substrate paired with the id it resolved to — `None` for one whose
/// height map has not been fetched. For the pickers, which list them and mark the
/// one the document is on; `read`, so a row settles when its fetch lands.
///
/// The library's substrates are not here: those are `state.substrates.entries`, and each
/// already carries the id it is named by. A document may also be on a substrate that is
/// in *neither* — one a peer brought, or one loaded from a file whose substrate was never
/// in this browser's library — which is why a picker asks this rather than assuming
/// its own list is exhaustive ([`SubstrateGallery`] draws that case as its own card).
pub fn resolved(state: AppState) -> Vec<(&'static BuiltinSubstrate, Option<SubstrateId>)> {
    let renderer = state.renderer.read();
    SUBSTRATES
        .iter()
        .map(|g| {
            let id = match g.asset {
                None => Some(SubstrateId::Flat),
                Some(_) => renderer.as_ref().and_then(|r| r.builtin_substrate(g.name)),
            };
            (g, id)
        })
        .collect()
}

// --- persistence ------------------------------------------------------------

/// Write the library's rows — [`SubstrateEntry`] narrowed to what is durable about it.
/// The bytes are not this function's to write; every caller reaching here has put
/// them down already (`crate::shapes`'s `persist` has the argument).
fn persist(entries: &[SubstrateEntry]) {
    let rows: Vec<StoredSubstrate> = entries
        .iter()
        .map(|e| StoredSubstrate {
            name: e.name.clone(),
            id: e.id,
        })
        .collect();
    storage::save_list(&rows);
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
/// Whoever asked redraws when a map lands without being told to: [`resolved`] and
/// [`thumbnail`] both read the renderer signal, and the quiet import writes it.
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
            for g in SUBSTRATES {
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
        SubstrateId::Image(id) => thumbnail(state, id),
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
                style: library::thumb_style(thumb.as_deref()),
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
/// real id, so it is an ordinary row of [`SUBSTRATES`] rather than a special case.
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
            .map(|e| (e.id, e.name.clone(), thumbnail(state, e.id)))
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
            Some((id, thumbnail(state, id)))
        }
        _ => None,
    };

    // One row per bundled substrate, flattened to what a card draws *before* the rsx.
    // A `for` whose body is anything but a plain element loses its `key` to positional
    // diffing, which is how a gallery ends up with one card wearing another's picture
    // — the trap `crate::shapes`' gallery is written around too.
    let builtins = catalog.into_iter().map(move |(g, id)| {
        let thumb = match id {
            Some(SubstrateId::Image(a)) => thumbnail(state, a),
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
                import_dropped(state, e.files());
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
                        style: library::thumb_style(thumb.as_deref()),
                    }
                    div { class: "asset-name", "{name}" }
                }
            }
            for (id, name, url) in thumbs() {
                div {
                    key: "{id.to_hex()}",
                    class: card(current == SubstrateId::Image(id)),
                    onclick: move |_| select(state, Pick::Custom(id)),
                    div { class: "asset-thumb", style: library::thumb_style(url.as_deref()) }
                    div { class: "asset-name", title: "{name}", "{name}" }
                    button {
                        class: "asset-remove",
                        title: "Remove from library",
                        onclick: move |e| {
                            e.stop_propagation();
                            remove(state, id);
                        },
                        {crate::icons::icon(crate::icons::REMOVE)}
                    }
                }
            }
            if let Some((id, url)) = stray {
                div {
                    key: "{id.to_hex()}",
                    class: "asset-card selected",
                    title: "A surface this document came with — not in your library.",
                    div { class: "asset-thumb", style: library::thumb_style(url.as_deref()) }
                    div { class: "asset-name", "From the document" }
                }
            }
            div { class: "asset-card import",
                // `pick_file` must run inside the click gesture — no task hop.
                onclick: move |_| {
                    crate::platform::pick_file("image/*", move |name, bytes| {
                        import_file(state, name, bytes);
                    });
                },
                div { class: "asset-thumb plus", {crate::icons::icon(crate::icons::ADD)} }
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
        .unwrap_or(DEFAULT_SUBSTRATE);
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
        SUBSTRATES
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
                    onclick: move |_| new_document(state, choice(), surf_choice(), on_close),
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
/// It owns closing the modal (`on_close`), calling it only once the work is done.
/// `spawn_forever`, not `spawn`: a plain spawn would tie the task to the
/// modal's scope, and the backdrop/Cancel still work during the fetch — a
/// dismissal would cancel it mid-flight *after* `collab::leave` already ran
/// (session gone, document never replaced). The task must outlive the modal;
/// calling `on_close` after it unmounted is harmless (the callback lives in
/// CommandRail's scope, which persists).
fn new_document(state: AppState, color: ColorSpaceId, pick: Pick, on_close: EventHandler<()>) {
    // Replacing the document abandons any shared session (and clears the
    // ticket from the URL) — the fresh canvas is private until re-shared.
    crate::collab::leave(state);
    spawn_forever(async move {
        // A substrate that will not fetch opens the document smooth rather than
        // refusing to open it — and the document then honestly *says* it is smooth
        // instead of claiming a substrate it hasn't got.
        let surface = resolve_signal(state, pick).await;
        crate::state::with_engine(state, |r| {
            r.new_document(color, surface);
            r.paint();
        });
        tracing::info!(?color, ?pick, ?surface, "new document ready");
        on_close.call(());
    });
}
