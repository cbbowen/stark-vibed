//! The two asset libraries this browser keeps — the brush stamps and the canvas
//! substrates a person brought in (§6.4, §6.6, §25.6).
//!
//! One object twice over (`stark_ui::assets`), so written once here too: reading a
//! library in, importing into it, making an entry usable in this document, removing
//! one, its gallery card, and offering it to a live session. What a kind adds is the
//! half only this frontend can say: where its signals are and how the browser decodes a
//! file for it ([`Shelf`]), and which renderer call answers for it ([`Engine`]). The
//! stores, the card and the fallback name are the crate's.
//!
//! Every task here is `spawn_forever` and every result lands in root-owned signals: an
//! import is started from a gallery, and closing its panel must not cancel or dangle it
//! (`collab.rs` has the same rule).

use dioxus::dioxus_core::spawn_forever;
use dioxus::prelude::*;
use stark_model::{AssetId, SubstrateId};
use stark_ui::assets::{self, Entry, Kind, Shapes, Substrates};
use stark_ui::library::Thumbs;

use crate::platform::{normalize_shape_image, normalize_substrate_image};
use crate::render::Renderer;
use crate::state::{AppState, root_signal, with_engine_quiet};

/// One library's signals. Root-owned: an import is started from a gallery's scope and
/// must outlive it.
#[derive(Clone, Copy)]
pub struct LibraryState {
    /// The entries, read at start from the browser's two stores (§25.6). Empty until
    /// that read lands, which is why [`load`] is awaited ahead of the first thing that
    /// resolves an id out of it.
    pub entries: Signal<Vec<Entry>>,
    /// A transient line under the gallery: why an import was refused, or what it did to
    /// the file. `None` when quiet.
    pub notice: Signal<Option<String>>,
}

impl LibraryState {
    pub(crate) fn new() -> Self {
        Self {
            entries: root_signal(Vec::new),
            notice: root_signal(|| None),
        }
    }
}

/// The engine half of one library: which renderer call answers "do you hold these
/// bytes", and which canonicalizes them into an id.
///
/// A trait over the engine rather than two more functions on the kind, so the
/// operations below can run in a test against a fake one — a renderer needs a GPU.
pub trait Engine<K: Kind> {
    /// The canonical bytes this document holds under `id`.
    fn held(&self, id: AssetId) -> Option<Vec<u8>>;
    /// Canonicalize `png` into this document, answering the id it is held under.
    fn import(&mut self, png: &[u8]) -> Result<AssetId, String>;
}

impl Engine<Shapes> for Renderer {
    fn held(&self, id: AssetId) -> Option<Vec<u8>> {
        self.asset_bytes(id)
    }

    fn import(&mut self, png: &[u8]) -> Result<AssetId, String> {
        self.import_brush_id(png)
    }
}

impl Engine<Substrates> for Renderer {
    fn held(&self, id: AssetId) -> Option<Vec<u8>> {
        self.substrate_bytes(SubstrateId::Image(id))
    }

    fn import(&mut self, png: &[u8]) -> Result<AssetId, String> {
        // The renderer logs why it refused an import and hands back only that it did.
        match self.import_substrate(png) {
            Some(SubstrateId::Image(id)) => Ok(id),
            // Said rather than unwrapped: `Flat` is the one substrate with no image, so
            // this would be the engine inventing a procedural substrate out of bytes.
            Some(SubstrateId::Flat) => Err("it canonicalized to the flat substrate".to_string()),
            None => Err("it is not a height map this build can read".to_string()),
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
    Renderer: Engine<K>,
{
    if let Some(url) = K::thumbs().get(id) {
        return Some(url);
    }
    let held = state
        .renderer
        .peek()
        .as_ref()
        .and_then(|r| <Renderer as Engine<K>>::held(r, id));
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
/// says what a row whose bytes are gone costs. Awaited once at start.
pub async fn load<K: Shelf>(state: AppState) {
    let mut entries = K::library(state).entries;
    entries.set(assets::load::<K>().await);
}

/// Import a picked file into `K`'s library — decoded in the browser, canonicalized in the
/// engine, added — and hand its id to `then`, which picks it.
///
/// A file the library holds already adds nothing and says so, and `then` runs all the
/// same: importing a file twice picks it rather than refusing it.
pub fn import_file<K: Shelf>(
    state: AppState,
    file_name: String,
    bytes: Vec<u8>,
    then: fn(AppState, AssetId),
) where
    Renderer: Engine<K>,
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
        let imported = with_engine_quiet(state, |r| canonicalize::<K>(r, png))
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
        let known = admit::<K>(library, entry).await;
        then(state, id);
        notice.set(said(&name, known, note));
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

/// Add an imported entry to the library — its bytes stored before the row that names
/// them (`storage::blob_save`) — unless its id is there already. Answers whether it was.
///
/// Asked again once the bytes are down: a second drop of the same file can land while
/// this one's are being written, and a library holding one id twice draws two cards for
/// one thing.
async fn admit<K: Kind>(library: LibraryState, entry: Entry) -> bool {
    let id = entry.id;
    let holds = |entries: &[Entry]| entries.iter().any(|e| e.id == id);
    let mut entries = library.entries;
    if holds(&entries.peek()) {
        return true;
    }
    assets::store_bytes::<K>(id, &entry.png).await;
    if holds(&entries.peek()) {
        return true;
    }
    entries.write().push(entry);
    assets::persist::<K>(&entries.peek());
    false
}

/// What an import says under the gallery: that the file was there already, what the
/// import did to it, or nothing.
fn said(name: &str, known: bool, note: Option<&str>) -> Option<String> {
    match (known, note) {
        (true, _) => Some(format!(
            "“{name}” is already in your library — selected it."
        )),
        (false, Some(note)) => Some(format!("“{name}” {note}")),
        (false, None) => None,
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
    Renderer: Engine<K>,
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
/// One door to the engine, `with_engine_quiet` — readying an asset changes no document
/// state — taken once and synchronously, so nothing is borrowed across the heal's await.
pub fn ensure<K: Shelf>(state: AppState, id: AssetId) -> Option<AssetId>
where
    Renderer: Engine<K>,
{
    let library = K::library(state);
    let Reached {
        id,
        bytes,
        healed_from,
    } = with_engine_quiet(state, |r| reach::<K>(r, library, id)).flatten()?;
    if let Some(stale) = healed_from {
        // The signal has moved already — the caller is about to paint with `id` — and
        // the two stores follow in a task, in the order `assets::heal` states.
        let (entries, png) = (library.entries, bytes.clone());
        spawn_forever(async move {
            let rows = entries.peek().to_vec();
            assets::heal::<K>(&rows, stale, id, &png).await;
        });
    }
    seed_session::<K>(state, id, bytes);
    Some(id)
}

/// What [`ensure`] found with the engine in hand.
struct Reached {
    /// The id to reference the asset by.
    id: AssetId,
    /// The canonical bytes it names, to offer a session.
    bytes: Vec<u8>,
    /// The id the library row held before it moved onto `id`, when it had to.
    healed_from: Option<AssetId>,
}

/// [`ensure`]'s engine half: held already, or imported out of the library — and the row
/// moved onto the id the engine gave, when the two differ.
fn reach<K: Kind>(
    engine: &mut impl Engine<K>,
    library: LibraryState,
    id: AssetId,
) -> Option<Reached> {
    // Already in this document: imported here, arrived with a loaded file, or fetched
    // off a peer.
    if let Some(bytes) = engine.held(id) {
        return Some(Reached {
            id,
            bytes,
            healed_from: None,
        });
    }
    let (name, png) = library
        .entries
        .peek()
        .iter()
        .find(|e| e.id == id)
        .map(|e| (e.name.clone(), e.png.clone()))?;
    let actual = match engine.import(&png) {
        Ok(actual) => actual,
        Err(why) => {
            let mut notice = library.notice;
            notice.set(Some(format!("“{name}” failed to load: {why}.")));
            return None;
        }
    };
    let healed_from = (actual != id).then(|| {
        let mut entries = library.entries;
        if let Some(row) = entries.write().iter_mut().find(|e| e.id == id) {
            row.id = actual;
        }
        id
    });
    Some(Reached {
        id: actual,
        bytes: png,
        healed_from,
    })
}

/// Drop an entry from `K`'s library — the row first, then the bytes, an import's order
/// reversed and for its reason (`storage::blob_save`). Paint made with it is untouched:
/// the engine's per-document store keeps every imported asset, and a save file bundles
/// whatever its log names (§8).
pub fn remove<K: Shelf>(state: AppState, id: AssetId) {
    let mut entries = K::library(state).entries;
    entries.write().retain(|e| e.id != id);
    assets::persist::<K>(&entries.peek());
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
    use std::sync::{Mutex, MutexGuard, Once, PoisonError};

    use dioxus::dioxus_core::{ScopeId, VirtualDom};
    use stark_ui::storage::{Backend, Stored};

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
            notice: Signal::new(None),
        }
    }

    /// Run what the store below answers at once.
    fn ready<F: Future>(future: F) -> F::Output {
        let mut future = std::pin::pin!(future);
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        match future.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(out) => out,
            std::task::Poll::Pending => panic!("the store answers at once"),
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
            assert_eq!((reached.id, reached.healed_from), (actual, Some(stale)));
            assert_eq!(
                reached.bytes,
                vec![7; 4],
                "the session is offered the row's bytes"
            );
            assert_eq!(
                library.entries.peek()[0].id,
                actual,
                "and the row moved onto the id the engine gave"
            );

            let again = reach::<Shapes>(&mut engine, library, actual).expect("held now");
            assert_eq!((again.id, again.healed_from), (actual, None));
            assert_eq!(engine.imports, 1, "so it was not imported twice");
        });
    }

    /// What this process's store was asked to write under the shape library's key.
    ///
    /// Installed rather than handed in, because `storage::install` is the only way in and
    /// keeps the first backend it is given — which is why the test using it shows the
    /// tape is live before relying on it saying nothing.
    struct Tape;

    static TAPE: Mutex<Vec<String>> = Mutex::new(Vec::new());

    fn tape() -> MutexGuard<'static, Vec<String>> {
        TAPE.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn record(what: &str, key: &str) {
        // Only this library's keys: another test in the process may keep a record too.
        if key.starts_with(Shapes::STORE.named().0) {
            tape().push(format!("{what} {key}"));
        }
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
        ) -> Stored<'a, Result<Vec<Option<Vec<u8>>>, String>> {
            Box::pin(std::future::ready(Ok(vec![None; keys.len()])))
        }

        fn blob_put<'a>(&'a self, key: &'a str, _: &'a [u8]) -> Stored<'a, Result<(), String>> {
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
        static INSTALL: Once = Once::new();
        INSTALL.call_once(|| stark_ui::storage::install(Tape));
        tape().clear();
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
                ready(admit::<Shapes>(library, entry))
            };

            assert!(!take(&mut engine), "the first import is new");
            assert!(
                !std::mem::take(&mut *tape()).is_empty(),
                "and its bytes and row were stored"
            );

            assert!(take(&mut engine), "the second is content the library holds");
            assert_eq!(*tape(), Vec::<String>::new(), "so nothing is stored again");
            assert_eq!(
                library.entries.peek().len(),
                1,
                "and no second card is drawn"
            );
            assert_eq!(
                said("Pencil", true, Some("was inverted")).as_deref(),
                Some("“Pencil” is already in your library — selected it."),
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
