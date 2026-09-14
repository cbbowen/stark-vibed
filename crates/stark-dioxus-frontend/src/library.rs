//! The two asset libraries this browser keeps — the brush stamps and the canvas
//! substrates a person brought in (§6.4, §6.6, §25.6).
//!
//! One object twice over (`stark_ui::assets`), so written once here too: reading a
//! library in, importing into it, making an entry usable in this document, removing
//! one, its gallery card, and offering it to a live session. What a kind adds is the
//! half only this frontend can say: where its signals are and how the browser decodes a
//! file for it ([`Shelf`]), and which engine call answers for it ([`Engine`]). The
//! stores, the card and the fallback name are the crate's.
//!
//! Every task here is `spawn_forever` and every result lands in root-owned signals: an
//! import is started from a gallery, and closing its panel must not cancel or dangle it
//! (`collab.rs` has the same rule).

use dioxus::dioxus_core::spawn_forever;
use dioxus::prelude::*;
use stark_model::{AssetId, SubstrateId};
use stark_ui::assets::{self, Entry, Kind, Shapes, Substrates, Unread};
use stark_ui::desk::Desk;
use stark_ui::library::Thumbs;

use crate::platform::{normalize_shape_image, normalize_substrate_image};
use crate::state::{AppState, gpu_lost, root_signal, with_engine_quiet};

/// One library's signals. Root-owned: an import is started from a gallery's scope and
/// must outlive it.
#[derive(Clone, Copy)]
pub struct LibraryState {
    /// The entries, read at start from the browser's two stores (§25.6). Empty until
    /// that read lands, which is why [`load`] is awaited ahead of the first thing that
    /// resolves an id out of it.
    pub entries: Signal<Vec<Entry>>,
    /// The stored rows whose bytes would not read this session. Never shown, and written
    /// back beside `entries` by every write of the library.
    pub unread: Signal<Vec<Unread>>,
    /// A transient line under the gallery: why an import was refused, or what it did to
    /// the file. `None` when quiet.
    pub notice: Signal<Option<String>>,
}

impl LibraryState {
    pub(crate) fn new() -> Self {
        Self {
            entries: root_signal(Vec::new),
            unread: root_signal(Vec::new),
            notice: root_signal(|| None),
        }
    }
}

/// The engine half of one library: which engine call answers "do you hold these
/// bytes", and which canonicalizes them into an id.
///
/// A trait over the engine rather than two more functions on the kind, so the
/// operations below can run in a test against a fake one — a renderer needs a GPU.
pub trait Engine<K: Kind> {
    /// Whether this document holds `id`, asked without copying the bytes.
    fn holds(&self, id: AssetId) -> bool;
    /// The canonical bytes this document holds under `id`.
    fn held(&self, id: AssetId) -> Option<Vec<u8>>;
    /// Canonicalize `png` into this document, answering the id it is held under.
    fn import(&mut self, png: &[u8]) -> Result<AssetId, String>;
}

impl Engine<Shapes> for Desk {
    fn holds(&self, id: AssetId) -> bool {
        self.engine().holds(Shapes::need(id))
    }

    fn held(&self, id: AssetId) -> Option<Vec<u8>> {
        self.engine().asset_bytes(id)
    }

    fn import(&mut self, png: &[u8]) -> Result<AssetId, String> {
        self.engine().import_brush(png).map_err(|e| e.to_string())
    }
}

impl Engine<Substrates> for Desk {
    fn holds(&self, id: AssetId) -> bool {
        self.engine().holds(Substrates::need(id))
    }

    fn held(&self, id: AssetId) -> Option<Vec<u8>> {
        self.engine().substrate_bytes(SubstrateId::Image(id))
    }

    fn import(&mut self, png: &[u8]) -> Result<AssetId, String> {
        match self.engine_mut().import_substrate(png) {
            Ok(SubstrateId::Image(id)) => Ok(id),
            // Said rather than unwrapped: `Flat` is the one substrate with no image, so
            // this would be the engine inventing a procedural substrate out of bytes.
            Ok(SubstrateId::Flat) => Err("it canonicalized to the flat substrate".to_string()),
            // The engine's reason goes to the log; the gallery is told what it means.
            Err(e) => {
                tracing::warn!("canvas substrate failed to import: {e}");
                Err("it is not a height map this build can read".to_string())
            }
        }
    }
}

/// A decoded file on its way into a library: its PNG, and what the import did to it when
/// that is worth saying — or why it could not be decoded.
type Normalized = Result<(Vec<u8>, Option<&'static str>), String>;

/// A library as this frontend keeps it.
pub trait Shelf: Kind {
    /// Where this library's signals are.
    fn library(state: AppState) -> LibraryState;
    /// This library's gallery thumbnails — its own cache, because one grayscale PNG lands
    /// on one id under both readings and is two different pictures (`stark_ui::library`).
    fn thumbs() -> &'static Thumbs<String>;
    /// Decode a picked file into this library's PNG, through the browser.
    async fn normalize(bytes: Vec<u8>) -> Normalized;
}

impl Shelf for Shapes {
    fn library(state: AppState) -> LibraryState {
        state.shapes
    }

    fn thumbs() -> &'static Thumbs<String> {
        static THUMBS: Thumbs<String> = Thumbs::new();
        &THUMBS
    }

    async fn normalize(bytes: Vec<u8>) -> Normalized {
        let (png, inverted) = normalize_shape_image(bytes).await?;
        let note = "read as dark ink on light paper, so it was inverted — white now paints.";
        Ok((png, inverted.then_some(note)))
    }
}

impl Shelf for Substrates {
    fn library(state: AppState) -> LibraryState {
        state.substrates
    }

    fn thumbs() -> &'static Thumbs<String> {
        static THUMBS: Thumbs<String> = Thumbs::new();
        &THUMBS
    }

    async fn normalize(bytes: Vec<u8>) -> Normalized {
        Ok((normalize_substrate_image(bytes).await?, None))
    }
}

/// A `background-image` data URL of what `id` names under `K`'s reading: the coverage a
/// stamp lays, or the height field a substrate bites (`stark_ui::assets::card`).
///
/// Bytes from the engine first and the library second: a shipped asset is only ever in
/// the engine, and one imported in an earlier session only in the library until it is
/// picked. `None` while a shipped asset's fetch is in flight — the same moment its card
/// is not clickable — or when the bytes do not decode.
///
/// The engine is peeked rather than read: a gallery redraws when a shipped asset lands
/// (`shipped::watch`) or its library changes, and a read would redraw it per command.
pub fn thumbnail<K: Shelf>(state: AppState, id: AssetId) -> Option<String>
where
    Desk: Engine<K>,
{
    if let Some(url) = K::thumbs().get(id) {
        return Some(url);
    }
    let held = state
        .renderer
        .peek()
        .as_ref()
        .and_then(|r| <Desk as Engine<K>>::held(&r.desk, id));
    let bytes = held.or_else(|| {
        K::library(state)
            .entries
            .read()
            .iter()
            .find(|e| e.id == id)
            .map(|e| e.png.clone())
    })?;
    let url = card_url::<K>(&bytes)?;
    K::thumbs().put(id, url.clone());
    Some(url)
}

/// The field `png` names under `K`'s reading, reduced to a card and carried as a data
/// URL — the numbers are the crate's, the carrier this frontend's (`crate::cards`).
fn card_url<K: Kind>(png: &[u8]) -> Option<String> {
    crate::cards::data_url(assets::card::<K>(png)?)
}

/// Fill `K`'s library from the browser's two stores — `stark_ui::assets::load`, which
/// says what a row whose bytes are gone costs, and what one whose bytes would not read
/// does. Awaited once at start.
pub async fn load<K: Shelf>(state: AppState) {
    let LibraryState {
        mut entries,
        mut unread,
        ..
    } = K::library(state);
    let loaded = assets::load::<K>().await;
    entries.set(loaded.entries);
    unread.set(loaded.unread);
}

/// Import a picked file into `K`'s library — decoded in the browser, canonicalized in the
/// engine, added — and hand its id to `then`, which picks it.
///
/// A file the library holds already adds nothing and says so, and `then` runs all the
/// same: importing a file twice picks it rather than refusing it. So does a file the
/// store would not take, which is in this document and not in the library.
pub fn import_file<K: Shelf>(
    state: AppState,
    file_name: String,
    bytes: Vec<u8>,
    then: fn(AppState, AssetId),
) where
    Desk: Engine<K>,
{
    let library = K::library(state);
    let mut notice = library.notice;
    notice.set(None);
    spawn_forever(async move {
        let refused = |why: String| Some(format!("Couldn't import “{file_name}”: {why}."));
        let (png, note) = match K::normalize(bytes).await {
            Ok(normalized) => normalized,
            Err(why) => {
                notice.set(refused(why));
                return;
            }
        };
        // The canonical bytes, not the file's: what the id names, what a save file bundles
        // and what a peer is served are one representation (§8, §19). Quiet, because
        // readying an asset changes no document state.
        let imported = with_engine_quiet(state, |r| canonicalize::<K>(&mut r.desk, png))
            .unwrap_or_else(|| Err("the canvas is still starting".to_string()));
        let (id, canonical) = match imported {
            Ok(imported) => imported,
            Err(why) => {
                notice.set(refused(why));
                return;
            }
        };
        let name = stark_ui::library::display_name(&file_name, K::FALLBACK_NAME);
        let entry = Entry {
            name: name.clone(),
            png: canonical,
            id,
        };
        let admitted = admit::<K>(library, entry).await;
        then(state, id);
        notice.set(said(&name, &admitted, note));
    });
}

/// Canonicalize `png` in the engine: the id it is held under, and the bytes that id names
/// — the engine's own, which are what a library row has to match.
fn canonicalize<K: Kind>(
    engine: &mut impl Engine<K>,
    png: Vec<u8>,
) -> Result<(AssetId, Vec<u8>), String> {
    let id = engine.import(&png)?;
    Ok((id, engine.held(id).unwrap_or(png)))
}

/// What adding an import to its library came to.
#[derive(Debug, PartialEq)]
enum Admitted {
    /// Its id was in the library already.
    Known,
    Added,
    /// The store would not take its bytes, for this reason, so no row names them: it is
    /// in this document and not in the library.
    Unsaved(String),
}

/// Add an imported entry to the library — its bytes stored before the row that names
/// them, and no row when they did not land (`assets::store_bytes`) — unless its id is
/// there already.
///
/// Asked again once the bytes are down: a second drop of the same file can land while
/// this one's are being written, and a library holding one id twice draws two cards for
/// one thing.
async fn admit<K: Kind>(library: LibraryState, entry: Entry) -> Admitted {
    let id = entry.id;
    let holds = |entries: &[Entry]| entries.iter().any(|e| e.id == id);
    let LibraryState {
        mut entries,
        unread,
        ..
    } = library;
    if holds(&entries.peek()) {
        return Admitted::Known;
    }
    let stored = assets::store_bytes::<K>(id, &entry.png).await;
    if holds(&entries.peek()) {
        return Admitted::Known;
    }
    if let Err(why) = stored {
        return Admitted::Unsaved(why);
    }
    entries.write().push(entry);
    assets::persist::<K>(&entries.peek(), &unread.peek());
    Admitted::Added
}

/// What an import says under the gallery: that the file was there already, that the
/// library could not keep it, what the import did to it, or nothing.
fn said(name: &str, admitted: &Admitted, note: Option<&str>) -> Option<String> {
    match (admitted, note) {
        (Admitted::Known, _) => Some(format!(
            "“{name}” is already in your library — selected it."
        )),
        (Admitted::Unsaved(why), note) => Some(format!(
            "“{name}” is in this document only: your library couldn't store it ({why}).{}",
            note.map(|note| format!(" It {note}")).unwrap_or_default()
        )),
        (Admitted::Added, Some(note)) => Some(format!("“{name}” {note}")),
        (Admitted::Added, None) => None,
    }
}

/// Import files dropped onto a gallery, each read and imported on its own. When several
/// land, the last to finish is the one `then` leaves picked — arbitrary but harmless,
/// since every one of them joins the library.
pub fn import_dropped<K: Shelf>(
    state: AppState,
    files: Vec<dioxus::html::FileData>,
    then: fn(AppState, AssetId),
) where
    Desk: Engine<K>,
{
    for file in files {
        spawn_forever(async move {
            let name = file.name();
            match file.read_bytes().await {
                Ok(bytes) => import_file::<K>(state, name, bytes.to_vec(), then),
                Err(e) => {
                    let mut notice = K::library(state).notice;
                    notice.set(Some(format!("Couldn't read “{name}”: {e}.")));
                }
            }
        });
    }
}

/// Make `id` usable in this document: its bytes in the engine and offered to a live
/// session, answering the id to reference it by — **healed** when the stored id predates
/// a canonicalization change (§19). `None` when neither the engine nor the library has
/// the bytes, the canvas is not up, or the import was refused, which the gallery is told.
///
/// Shared by both galleries' picks and by preset application (`crate::presets`), which
/// resurrects stamps from snapshots taken in other documents. It matters more for a
/// substrate: a `SetSubstrate` naming bytes the engine lacks bakes a flat deposit into
/// tiles that nothing later un-bakes (§6.4).
///
/// An id the engine holds already is answered under `peek`, which is every pick after
/// the first; only an import takes the engine's write door, `with_engine_quiet` —
/// readying an asset changes no document state — once and synchronously, so nothing is
/// borrowed across the heal's await.
pub fn ensure<K: Shelf>(state: AppState, id: AssetId) -> Option<AssetId>
where
    Desk: Engine<K>,
{
    if gpu_lost(state) {
        return None;
    }
    let held = state
        .renderer
        .peek()
        .as_ref()
        .map(|r| <Desk as Engine<K>>::holds(&r.desk, id))?;
    let id = if held {
        id
    } else {
        let library = K::library(state);
        let Reached { id, healed } =
            with_engine_quiet(state, |r| reach::<K>(&mut r.desk, library, id)).flatten()?;
        if let Some(Healed { stale, png }) = healed {
            heal::<K>(library, stale, id, png);
        }
        id
    };
    offer::<K>(state, id);
    Some(id)
}

/// What [`reach`] found with the engine in hand.
struct Reached {
    /// The id to reference the asset by.
    id: AssetId,
    /// The row's move onto `id`, when it had to move.
    healed: Option<Healed>,
}

/// A library row moved off the id it was stored under.
struct Healed {
    stale: AssetId,
    /// The row's bytes, to store under the id it moved onto.
    png: Vec<u8>,
}

/// [`ensure`]'s import half, for an id the engine does not hold: the library's bytes
/// imported, and the row moved onto the id the engine gave when the two differ.
fn reach<K: Kind>(
    engine: &mut impl Engine<K>,
    library: LibraryState,
    id: AssetId,
) -> Option<Reached> {
    let imported = {
        let entries = library.entries.peek();
        let entry = entries.iter().find(|e| e.id == id)?;
        engine
            .import(&entry.png)
            .map_err(|why| format!("“{}” failed to load: {why}.", entry.name))
    };
    let actual = match imported {
        Ok(actual) => actual,
        Err(said) => {
            let mut notice = library.notice;
            notice.set(Some(said));
            return None;
        }
    };
    if actual == id {
        return Some(Reached { id, healed: None });
    }
    let mut entries = library.entries;
    let png = entries.write().iter_mut().find(|e| e.id == id).map(|row| {
        row.id = actual;
        row.png.clone()
    });
    Some(Reached {
        id: actual,
        healed: png.map(|png| Healed { stale: id, png }),
    })
}

/// Follow a row's move onto `actual` into the two stores, in the order `assets::heal`
/// states — the rows read once the bytes have landed, so an import or removal made
/// during the write is written with them.
///
/// The signal moved already, since the caller is about to paint with `actual`. When the
/// bytes do not land it moves back: the stores still name them by `stale`, and a later
/// write this session would otherwise name bytes that were never stored. The next pick
/// heals it again.
fn heal<K: Kind>(library: LibraryState, stale: AssetId, actual: AssetId, png: Vec<u8>) {
    let LibraryState {
        mut entries,
        unread,
        ..
    } = library;
    spawn_forever(async move {
        let rows = move || assets::persist::<K>(&entries.peek(), &unread.peek());
        if assets::heal::<K>(stale, actual, &png, rows).await.is_err()
            && let Some(row) = entries.write().iter_mut().find(|e| e.id == actual)
        {
            row.id = stale;
        }
    });
}

/// Offer what this document holds under `id` to a live session. Copied only when there
/// is one: a height map is megabytes to copy for nobody.
fn offer<K: Kind>(state: AppState, id: AssetId)
where
    Desk: Engine<K>,
{
    if state.collab.session.peek().is_none() {
        return;
    }
    let bytes = state
        .renderer
        .peek()
        .as_ref()
        .and_then(|r| <Desk as Engine<K>>::held(&r.desk, id));
    if let Some(bytes) = bytes {
        seed_session::<K>(state, id, bytes);
    }
}

/// Drop an entry from `K`'s library — the row first, then the bytes, an import's order
/// reversed and for its reason (`storage::blob_save`). Paint made with it is untouched,
/// and a document on a removed substrate stays on it: the engine's per-document store
/// keeps every imported asset, and a save file bundles whatever its log names (§8).
pub fn remove<K: Shelf>(state: AppState, id: AssetId) {
    let LibraryState {
        mut entries,
        mut unread,
        ..
    } = K::library(state);
    assets::remove::<K>(&mut entries.write(), &mut unread.write(), id);
    spawn_forever(async move { assets::drop_bytes::<K>(id).await });
}

/// Offer `bytes` to a live session as `K`'s need, so a peer can fetch them by hash the
/// moment an action names `id` (§12.4). A no-op when solo; idempotent when repeated,
/// being content-addressed.
pub fn seed_session<K: Kind>(state: AppState, id: AssetId, bytes: Vec<u8>) {
    if let Some(broadcaster) = state
        .collab
        .session
        .read()
        .as_ref()
        .map(|s| s.broadcaster())
    {
        broadcaster.add_content(K::need(id), bytes);
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::sync::Once;

    use dioxus::dioxus_core::{ScopeId, VirtualDom};
    use stark_ui::storage::{Backend, BlobRead, Stored, at_once};

    use super::*;

    /// An engine that holds what it imports, under an id the test chooses — so it can
    /// disagree with the id a library row carries, which is what a canonicalization
    /// change looks like from here.
    struct Fake {
        held: Vec<(AssetId, Vec<u8>)>,
        /// The id every import lands on.
        lands_on: AssetId,
        imports: usize,
    }

    impl Fake {
        fn landing_on(id: AssetId) -> Self {
            Self {
                held: Vec::new(),
                lands_on: id,
                imports: 0,
            }
        }
    }

    impl<K: Kind> Engine<K> for Fake {
        fn holds(&self, id: AssetId) -> bool {
            self.held.iter().any(|(held, _)| *held == id)
        }

        fn held(&self, id: AssetId) -> Option<Vec<u8>> {
            self.held
                .iter()
                .find(|(held, _)| *held == id)
                .map(|(_, bytes)| bytes.clone())
        }

        fn import(&mut self, png: &[u8]) -> Result<AssetId, String> {
            self.imports += 1;
            self.held.push((self.lands_on, png.to_vec()));
            Ok(self.lands_on)
        }
    }

    /// Run `f` where a signal can be made: the root scope of an app with nothing in it.
    fn in_app<R>(f: impl FnOnce() -> R) -> R {
        let dom = VirtualDom::new(|| rsx! {});
        dom.in_scope(ScopeId::ROOT, f)
    }

    fn shelf(entries: Vec<Entry>) -> LibraryState {
        LibraryState {
            entries: Signal::new(entries),
            unread: Signal::new(Vec::new()),
            notice: Signal::new(None),
        }
    }

    fn pencil(n: u8) -> Entry {
        Entry {
            name: "Pencil".to_string(),
            png: vec![n; 4],
            id: AssetId([n; 32]),
        }
    }

    /// **The heal path.** A row whose id predates a canonicalization change is imported,
    /// lands on another id, and the row moves onto it — so the caller paints with the id
    /// the engine will serve, and the next ask finds the bytes held rather than importing
    /// them again.
    #[test]
    fn a_stale_row_heals_onto_the_id_the_engine_gives() {
        in_app(|| {
            let (stale, actual) = (AssetId([1; 32]), AssetId([2; 32]));
            let library = shelf(vec![Entry {
                name: "Bristles".to_string(),
                png: vec![7; 4],
                id: stale,
            }]);
            let mut engine = Fake::landing_on(actual);

            let reached =
                reach::<Shapes>(&mut engine, library, stale).expect("the library has the bytes");
            assert_eq!(reached.id, actual);
            let healed = reached.healed.expect("the row moved");
            assert_eq!(healed.stale, stale);
            assert_eq!(
                healed.png,
                vec![7; 4],
                "the row's bytes go to the id it moved onto"
            );
            assert_eq!(
                library.entries.peek()[0].id,
                actual,
                "and the row moved onto the id the engine gave"
            );
            assert!(
                Engine::<Shapes>::holds(&engine, actual),
                "so the next ask finds it held rather than importing it again"
            );
            assert_eq!(engine.imports, 1);
        });
    }

    /// What this test thread asked the store to write — the process's store, but each
    /// thread's own tape — and why the store refuses blob writes while it does.
    ///
    /// Installed rather than handed in, because `storage::install` is the only way in and
    /// keeps the first backend it is given — which is why the tests using it show the
    /// tape is live before relying on it saying nothing.
    struct Tape;

    thread_local! {
        static TAPE: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
        static FULL: RefCell<Option<String>> = const { RefCell::new(None) };
    }

    fn record(what: &str, key: &str) {
        TAPE.with_borrow_mut(|tape| tape.push(format!("{what} {key}")));
    }

    /// The tape since this was last asked, oldest first.
    fn tape() -> Vec<String> {
        TAPE.take()
    }

    fn install_tape() {
        static INSTALL: Once = Once::new();
        INSTALL.call_once(|| stark_ui::storage::install(Tape));
        TAPE.take();
        FULL.set(None);
    }

    impl Backend for Tape {
        fn get(&self, _: &str) -> Option<String> {
            None
        }

        fn set(&self, key: &str, _: &str) -> bool {
            record("set", key);
            true
        }

        fn remove(&self, _: &str) {}

        fn blob_get_many<'a>(
            &'a self,
            keys: &'a [String],
        ) -> Stored<'a, Result<Vec<BlobRead>, String>> {
            Box::pin(std::future::ready(Ok(vec![Ok(None); keys.len()])))
        }

        fn blob_put<'a>(&'a self, key: &'a str, _: &'a [u8]) -> Stored<'a, Result<(), String>> {
            if let Some(why) = FULL.with_borrow(Clone::clone) {
                return Box::pin(std::future::ready(Err(why)));
            }
            record("put", key);
            Box::pin(std::future::ready(Ok(())))
        }

        fn blob_delete<'a>(&'a self, key: &'a str) -> Stored<'a, ()> {
            record("delete", key);
            Box::pin(std::future::ready(()))
        }
    }

    /// **A file the library already holds stores nothing twice** — no second write of its
    /// bytes or of the rows, and no second card — and the gallery is told it was picked
    /// rather than imported.
    #[test]
    fn an_import_the_library_already_holds_stores_nothing_twice() {
        install_tape();
        in_app(|| {
            let id = AssetId([3; 32]);
            let mut engine = Fake::landing_on(id);
            let library = shelf(Vec::new());
            let take = |engine: &mut Fake| {
                let (id, canonical) =
                    canonicalize::<Shapes>(engine, vec![5; 4]).expect("the fake takes anything");
                let entry = Entry {
                    name: "Pencil".to_string(),
                    png: canonical,
                    id,
                };
                at_once(admit::<Shapes>(library, entry))
            };

            assert_eq!(
                take(&mut engine),
                Admitted::Added,
                "the first import is new"
            );
            assert!(!tape().is_empty(), "and its bytes and row were stored");

            assert_eq!(
                take(&mut engine),
                Admitted::Known,
                "the second is content the library holds"
            );
            assert_eq!(tape(), Vec::<String>::new(), "so nothing is stored again");
            assert_eq!(
                library.entries.peek().len(),
                1,
                "and no second card is drawn"
            );
            assert_eq!(
                said("Pencil", &Admitted::Known, Some("was inverted")).as_deref(),
                Some("“Pencil” is already in your library — selected it."),
            );
        });
    }

    /// **An import whose bytes the store refuses is not added and writes no row** — a row
    /// naming bytes that never landed is dropped by the next load, and the import with
    /// it. The gallery is told it is in this document only, and why.
    #[test]
    fn an_import_the_store_refuses_adds_no_row_and_writes_nothing() {
        install_tape();
        in_app(|| {
            let library = shelf(Vec::new());
            assert_eq!(
                at_once(admit::<Shapes>(library, pencil(1))),
                Admitted::Added
            );
            assert!(!tape().is_empty(), "the tape is live");

            FULL.set(Some("QuotaExceededError".to_string()));
            let refused = at_once(admit::<Shapes>(library, pencil(2)));
            assert_eq!(refused, Admitted::Unsaved("QuotaExceededError".to_string()));
            assert_eq!(tape(), Vec::<String>::new(), "no row is written");
            assert_eq!(
                library.entries.peek().len(),
                1,
                "and no card is drawn for it"
            );
            assert_eq!(
                said("Pencil", &refused, None).as_deref(),
                Some(
                    "“Pencil” is in this document only: your library couldn't store it \
                     (QuotaExceededError)."
                ),
            );
        });
    }

    /// A 64×64 disc, written the two ways a brush shape may be authored: `alpha` puts the
    /// coverage in the alpha channel over transparent black, `value` puts it in the color
    /// over an opaque black substrate.
    fn disc(alpha: bool) -> Vec<u8> {
        const N: u32 = 64;
        let mut pixels = Vec::with_capacity((N * N * 4) as usize);
        for y in 0..N {
            for x in 0..N {
                let (dx, dy) = (x as f32 - 31.5, y as f32 - 31.5);
                let r = (dx * dx + dy * dy).sqrt() / 30.0;
                let cov = (255.0 * (1.0 - r).clamp(0.0, 1.0)) as u8;
                pixels.extend_from_slice(&if alpha {
                    [255, 255, 255, cov]
                } else {
                    [cov, cov, cov, 255]
                });
            }
        }
        let mut out = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut out, N, N);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&pixels).unwrap();
        }
        out
    }

    /// A stamp's card is taken over its *coverage*, so the two spellings of one disc draw
    /// the same card — an imported shape cannot look out of place beside a built-in for
    /// having put its coverage somewhere else.
    #[test]
    fn a_shape_draws_the_same_card_however_its_coverage_was_authored() {
        let (alpha, value) = (
            card_url::<Shapes>(&disc(true)),
            card_url::<Shapes>(&disc(false)),
        );
        assert!(alpha.is_some());
        assert_eq!(alpha, value);
    }

    /// And the card the two agree on has a *transparent* ground, not a black one: where
    /// the disc does not cover, nothing is drawn, so the panel shows through however the
    /// source spelled that.
    #[test]
    fn where_a_shape_does_not_cover_the_thumbnail_is_clear() {
        let card = assets::card::<Shapes>(&disc(false)).expect("a card");
        let bytes = crate::cards::encode_png(card).expect("a thumbnail");

        let mut reader = png::Decoder::new(std::io::Cursor::new(bytes))
            .read_info()
            .unwrap();
        let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut buf).unwrap();
        assert_eq!(info.color_type, png::ColorType::GrayscaleAlpha);

        // The corner is outside the disc; the centre is the whole of it.
        let at = |x: u32, y: u32| buf[((y * info.width + x) * 2 + 1) as usize];
        assert_eq!(at(0, 0), 0, "the corner is clear");
        // Not 255: the disc falls off from its exact centre, and the centre texel is
        // half a texel off it. What is being pinned is the axis, not the peak.
        assert!(
            at(info.width / 2, info.height / 2) > 240,
            "the centre covers"
        );
    }
}
