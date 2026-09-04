//! What an imported image *becomes*, and what a client's asset library is made of
//! (§6.4, §6.6, §19, §25.6).
//!
//! Two libraries, and they are the same object twice over: the brush stamps a user
//! brought in and the canvas substrates. An entry is a canonical image keyed by its
//! content id, the id is the whole of the reference, the rows go to the text store and
//! the bytes to the blob store beside them, and each is shown as a card the size of a
//! fingernail. What differs is the one thing [`Kind`] carries — what the id is a
//! *reading of*.
//!
//! # The half a decoder does not answer
//!
//! Importing an image is decode, resample, **decide**, encode. Only the middle two
//! words are a platform's: the browser decodes through `createImageBitmap` and the
//! native app through a Rust decoder, and neither of them knows that a scan of ink on
//! paper has to be inverted or that a substrate's transparency is not a trench. Those
//! decisions are here, over straight RGBA8, so the two frontends cannot come to
//! disagree about what a file means.
//!
//! **What is not shared is the resample**, and that is a real difference rather than
//! an oversight: `drawImage` is the only thing in the browser's chain that can scale
//! without first materializing the full-size buffer (a 48-megapixel photograph is
//! 190 MB of RGBA before anything has looked at it), so the web keeps it. Two
//! resamplers means the same oversized file imported on the two frontends lands on two
//! ids. Nothing depends on it landing on one — an id names bytes, and these are
//! different bytes — but it is worth knowing before someone tries to prove otherwise.
//! Under the cap there is no resample and the two agree exactly.

use std::marker::PhantomData;

use serde::{Deserialize, Serialize};
use stark_assetid::{AssetId, Canonical};

use crate::storage::{self, Store};

/// The longest edge an imported brush shape is kept at.
///
/// `stark_assetid::coverage` caps it anyway; capping at the import is what keeps the
/// full-size decode from ever reaching a library entry.
pub const SHAPE_CAP: u32 = stark_assetid::MAX_SHAPE_DIM;

/// The same for a canvas substrate, which is allowed to be larger: a stamp is a mark
/// and a substrate is a whole surface (§6.4).
pub const SUBSTRATE_CAP: u32 = stark_assetid::MAX_SUBSTRATE_DIM;

/// The size an import should be decoded to: `w`×`h` brought within `cap`, never
/// enlarged.
///
/// Shared so both frontends ask their own resampler for the *same* size, which is the
/// half of the divergence above that did not have to exist. Floored at one texel: a
/// zero-sized image is not an image, and the callers below say so.
pub fn fit(w: u32, h: u32, cap: u32) -> (u32, u32) {
    let scale = (f64::from(cap) / f64::from(w.max(h).max(1))).min(1.0);
    (
        ((f64::from(w) * scale) as u32).max(1),
        ((f64::from(h) * scale) as u32).max(1),
    )
}

/// A decoded image on its way into a library: straight (un-premultiplied) sRGB RGBA8,
/// top-left origin, tightly packed.
///
/// The form a PNG stores and the form `getImageData` returns, so nothing has to be
/// undone on either side of the platform boundary.
pub struct Decoded {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl Decoded {
    fn check(&self) -> Result<(), String> {
        if self.width == 0 || self.height == 0 {
            return Err("the image is empty".to_string());
        }
        let want = (self.width as usize) * (self.height as usize) * 4;
        if self.rgba.len() != want {
            return Err(format!(
                "the decoder gave {} bytes for a {}×{} image, not {want}",
                self.rgba.len(),
                self.width,
                self.height
            ));
        }
        Ok(())
    }
}

/// The luminance weights the whole app reads grey by, so "grey" means one thing
/// across it — the same three the engine's coverage read uses.
fn luminance(p: [u8; 4]) -> u32 {
    (77 * p[0] as u32 + 150 * p[1] as u32 + 29 * p[2] as u32) >> 8
}

/// Turn a decoded image into a **brush shape**'s PNG, and say whether its ink was
/// inverted.
///
/// The engine reads coverage as `luminance × alpha` — white paints, black does not —
/// but a scanned or drawn brush image is usually dark ink on light paper, which would
/// import as a solid rectangle with a shape-shaped hole in it. So the border ring is
/// measured: mostly-covered means ink-on-paper, and the luminance is inverted.
///
/// The **ring**, not the mean, and not a corner. A shape is a mark on a field, so what
/// the field is doing at the edge is what says which way round the image was authored
/// — where the mean is dominated by whichever of the two happens to cover more area,
/// and a single corner is one texel's worth of evidence.
pub fn shape_png(mut img: Decoded) -> Result<(Vec<u8>, bool), String> {
    img.check()?;
    let (w, h) = (img.width, img.height);
    let px = img.rgba.as_chunks_mut::<4>().0;

    let mut border = (0u64, 0u64);
    let mut ring = |i: usize| {
        let p = px[i];
        border.0 += (luminance(p) * p[3] as u32 / 255) as u64;
        border.1 += 1;
    };
    for x in 0..w {
        ring((x) as usize);
        ring(((h - 1) * w + x) as usize);
    }
    for y in 1..h.saturating_sub(1) {
        ring((y * w) as usize);
        ring((y * w + w - 1) as usize);
    }
    let inverted = border.0 / border.1.max(1) > 127;
    if inverted {
        for p in px.iter_mut() {
            let inv = (255 - luminance(*p)) as u8;
            (p[0], p[1], p[2]) = (inv, inv, inv);
        }
    }
    Ok((encode_rgba(w, h, &img.rgba)?, inverted))
}

/// Turn a decoded image into a **canvas substrate**'s PNG.
///
/// [`shape_png`]'s sibling, and the differences are the whole of what a substrate is
/// as against a stamp (§6.4):
///
/// - **Grey by luminance.** `stark_assetid::height` reads channel 0 and says why: an
///   authored height map's grey *is* its height, so weighting its channels would tilt
///   it. This is the step before that one, where a photograph of a canvas is being
///   turned into a height map, and the red channel of a photograph is not a height
///   field. Once it is grey the two readings agree — which is the point: the engine's
///   rule is left untouched and this policy stops it from ever mattering.
/// - **No inversion.** A stamp's polarity is a spelling; a substrate's polarity *is*
///   the substrate. Inverting one would turn its ridges into its valleys, and nothing
///   in the image says which was meant.
/// - **Alpha composited over white**, not multiplied in. A mask's transparency means
///   "no ink here", but a substrate has no such thing as an absent height: a PNG with
///   a transparent border would otherwise import as a deep trench around a canvas that
///   has none. White rather than black, because an unpainted substrate reads as its
///   own top surface.
pub fn substrate_png(mut img: Decoded) -> Result<Vec<u8>, String> {
    img.check()?;
    for p in img.rgba.as_chunks_mut::<4>().0 {
        let a = p[3] as u32;
        let over = (luminance(*p) * a + 255 * (255 - a)) / 255;
        *p = [over as u8; 4];
        p[3] = 255;
    }
    encode_rgba(img.width, img.height, &img.rgba)
}

/// Encode straight RGBA8 as a PNG — the form both imports hand to the engine, which
/// canonicalizes it again and takes the id from what it finds.
fn encode_rgba(w: u32, h: u32, rgba: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, w, h);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(png::Compression::Fast);
        let mut writer = encoder
            .write_header()
            .map_err(|e| format!("could not encode the image: {e}"))?;
        writer
            .write_image_data(rgba)
            .map_err(|e| format!("could not encode the image: {e}"))?;
    }
    Ok(out)
}

// --- the two libraries ----------------------------------------------------

/// Which library — the one thing that differs between them, as a type.
///
/// A type rather than a value because the store a row goes to is a `const` on the
/// storage traits, and because the two are never mixed: nothing takes a library and
/// asks which it is.
pub trait Kind: 'static {
    /// The record both halves of this library are kept under (§25.6).
    const STORE: Store;
    /// What one is called, for a message a person reads.
    const NOUN: &'static str;
    /// The field an id is a hash of, under this library's reading.
    ///
    /// **The whole of the difference.** The same grayscale PNG canonicalizes to the
    /// same id under both readings and means two different things, which is why a
    /// thumbnail cache is per-library and why this is a method rather than a flag.
    fn canonical(png: &[u8]) -> Option<Canonical>;
    /// What a gallery card is a picture of.
    fn ink() -> Ink;
}

/// The brush stamps a user brought in (§6.6).
pub struct Shapes;

impl Kind for Shapes {
    const STORE: Store = Store::Shapes;
    const NOUN: &'static str = "shape";

    fn canonical(png: &[u8]) -> Option<Canonical> {
        stark_assetid::coverage(png).ok()
    }

    fn ink() -> Ink {
        Ink::Coverage
    }
}

/// The canvas substrates a user brought in (§6.4).
pub struct Substrates;

impl Kind for Substrates {
    const STORE: Store = Store::Substrates;
    const NOUN: &'static str = "substrate";

    fn canonical(png: &[u8]) -> Option<Canonical> {
        stark_assetid::height(png).ok()
    }

    fn ink() -> Ink {
        Ink::Height
    }
}

/// How a card's texels are to be read — which is not a style but a fact about what
/// the field *is*.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ink {
    /// White ink with the field in alpha. A stamp's coverage says where paint lands,
    /// so the card must show the panel through the gaps.
    Coverage,
    /// Opaque grey. A substrate has no gaps — its low ground is as much a part of it
    /// as its high ground, and drawing the lows transparent would show a canvas full
    /// of holes.
    Height,
}

/// A gallery card's picture: the field reduced to a card's size, and how to read it.
///
/// Texels rather than an encoded image, because the two frontends put a picture on
/// screen by different means — a data URL in a `background-image`, a texture in an
/// element tree — and the *numbers* are what must not differ. A card drawn from the
/// canonical field rather than from the file it was authored from is the whole point:
/// where the author put the coverage, or which channel they left the height in, stops
/// being something either app has to be careful about.
pub struct Card {
    pub width: u32,
    pub height: u32,
    /// One byte per texel, row-major — read according to [`ink`](Self::ink).
    pub texels: Vec<u8>,
    pub ink: Ink,
}

/// The card for `png` under this library's reading. `None` when the bytes do not
/// decode, which is the same answer as no card yet.
pub fn card<K: Kind>(png: &[u8]) -> Option<Card> {
    let Canonical {
        width,
        height,
        texels,
    } = crate::library::reduce(K::canonical(png)?);
    Some(Card {
        width,
        height,
        texels,
        ink: K::ink(),
    })
}

// --- what this build ships -------------------------------------------------

/// One asset that ships with the app.
///
/// The catalog is here rather than in either frontend because a *name* is what a
/// preset asks for and what a picker labels, and two catalogs would be two answers to
/// "what does this build have". What stays in each frontend is how the bytes are
/// reached — a fetched URL on the web, bytes in the binary natively — which is the
/// one part the two genuinely differ on.
pub struct Shipped {
    /// The picker's label, and the name the rest of the app asks for it by. Not
    /// persisted anywhere — a document stores the resolved content id — so renaming
    /// one is a cosmetic change.
    pub name: &'static str,
    /// The file's path under the assets directory, which is how a build knows the
    /// asset's content id without reading it. `None` for one that is procedural and
    /// needs no image, which is [`SMOOTH`] and only ever `SMOOTH`.
    pub path: Option<&'static str>,
    /// The picker's one-line description. Empty for a shape: the shape gallery shows
    /// the name alone, and writing three sentences no surface renders would be
    /// content this catalog cannot check.
    pub blurb: &'static str,
}

/// The worn bristle shape: a dry, broken-edged tip.
pub const BRISTLES: &str = "Worn Bristles";
/// The flat shape.
pub const FLAT_TIP: &str = "Flat";
/// The pencil shape.
pub const PENCIL: &str = "Pencil";

/// Every brush shape that ships with the app, in gallery order.
pub const SHIPPED_SHAPES: &[Shipped] = &[
    Shipped {
        name: BRISTLES,
        path: Some("shape/Worn_Bristles.png"),
        blurb: "",
    },
    Shipped {
        name: FLAT_TIP,
        path: Some("shape/Flat.png"),
        blurb: "",
    },
    Shipped {
        name: PENCIL,
        path: Some("shape/Pencil.png"),
        blurb: "",
    },
];

pub const SMOOTH: &str = "Smooth";
pub const LINEN: &str = "Linen";
pub const ROUGH: &str = "Rough";

/// Every canvas substrate that ships with the app, in picker order.
pub const SHIPPED_SUBSTRATES: &[Shipped] = &[
    Shipped {
        name: SMOOTH,
        path: None,
        blurb: "A perfectly smooth surface — paint lies flat, no canvas texture.",
    },
    Shipped {
        name: LINEN,
        path: Some("substrate/Linen.png"),
        blurb: "A regular woven grid — the honest painter's canvas.",
    },
    Shipped {
        name: ROUGH,
        path: Some("substrate/Rough.png"),
        blurb: "Brushed acrylic substrate: irregular knife strokes with a fine crackle.",
    },
];

/// **What a substrate picker hands back** — the two ways one can be named before it is
/// resolved.
///
/// The asymmetry is real and is the whole reason this is an enum rather than an id: a
/// shipped substrate's id is the hash of a file that may not have been read yet, so it
/// can only be named by catalog *name* until then, while one in the library was hashed
/// when it was imported and is named by the id it already has.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Pick {
    /// A substrate this build ships with, by its row in [`SHIPPED_SUBSTRATES`].
    Builtin(&'static str),
    /// A substrate the user imported, by the content id the library holds it under.
    Custom(AssetId),
}

/// The substrate a fresh document opens on: [`SMOOTH`], the one that is procedural.
///
/// The opinion lives here rather than in the engine because the engine embeds no image
/// bytes and a substrate now *is* its bytes: naming one in core would be core naming
/// an image it cannot produce. `document::DEFAULT_SUBSTRATE` is `Flat` accordingly.
pub const DEFAULT_SUBSTRATE: Pick = Pick::Builtin(SMOOTH);

/// The shipped row called `name`, in either catalog.
pub fn shipped(name: &str) -> Option<&'static Shipped> {
    SHIPPED_SHAPES
        .iter()
        .chain(SHIPPED_SUBSTRATES)
        .find(|s| s.name == name)
}

include!(concat!(env!("OUT_DIR"), "/shipped_ids.rs"));

/// The content id of the shipped asset at `path`, **without reading it** (§12.4).
///
/// Hashed at build time (`build.rs`), which is what breaks the circularity the
/// catalogs otherwise live with: an id is a hash of the decoded image, so nothing that
/// runs before the bytes arrive can name one. This one can.
pub fn shipped_id(path: &str) -> Option<AssetId> {
    SHIPPED_IDS
        .iter()
        .find(|(p, _)| *p == path)
        .map(|(_, i)| *i)
}

/// The catalog row a content id names, if this build ships it — [`shipped_id`]
/// reversed, for making good on what [`resolvable`] promised.
pub fn shipped_at(id: AssetId) -> Option<&'static Shipped> {
    let path = SHIPPED_IDS
        .iter()
        .find(|(_, i)| *i == id)
        .map(|(p, _)| *p)?;
    SHIPPED_SHAPES
        .iter()
        .chain(SHIPPED_SUBSTRATES)
        .find(|s| s.path == Some(path))
}

/// Every content id this build can resolve out of its own bundle.
///
/// Handed to a session at join time so the host can leave them out of the snapshot:
/// the joiner is not saying "I have these loaded", it is saying "I can get these
/// without you" (§12.4).
pub fn resolvable() -> Vec<AssetId> {
    SHIPPED_IDS.iter().map(|(_, id)| *id).collect()
}

/// One entry in a library, **with its bytes in hand**.
///
/// Constructing one means the bytes are here: [`load`] drops a row whose blob is gone
/// rather than admitting a byte-less entry, so nothing downstream has a half-loaded
/// asset to draw a blank card for.
#[derive(Clone, PartialEq)]
pub struct Entry {
    /// Display name, defaulted from the imported file's stem.
    pub name: String,
    /// The canonical PNG the engine stores under `id`.
    pub png: Vec<u8>,
    /// Content id of `png`. If a build ever canonicalizes differently, the id the
    /// import actually returns is what heals the entry ([`heal`]).
    pub id: AssetId,
}

/// One row of the stored library: **a name and an id, and no bytes at all.**
///
/// The id is the whole of the reference — it *names* the PNG (§19) — which is what
/// makes a row small enough to keep in a text store the settings, the chord table and
/// the tour's ledger are also spending. The bytes it names are a blob under the same
/// record's key.
///
/// The kind rides in a `PhantomData` that is skipped, so the JSON is `{name, id}` for
/// both libraries — the shape every already-stored library is written in.
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Row<K> {
    name: String,
    #[serde(with = "storage::hex")]
    id: AssetId,
    #[serde(skip)]
    kind: PhantomData<K>,
}

impl<K: Kind> storage::Entry for Row<K> {
    const STORE: Store = K::STORE;
}

impl<K: Kind> storage::Blob for Row<K> {
    const STORE: Store = K::STORE;
}

/// Read a library out of its two stores.
///
/// Two reads because it is kept in two: the rows out of the text store, then their
/// PNGs out of the blob store in a single batch. **A row whose bytes are not there is
/// dropped**, and the library written back without it — a blob store is evictable
/// under storage pressure, so that is a state to expect rather than one that only
/// follows a crash, and an entry that cannot be painted with is not an asset. Leaving
/// it in would show a card that draws nothing and fails every time it is clicked.
pub async fn load<K: Kind>() -> Vec<Entry> {
    let rows = storage::load_list::<Row<K>>().unwrap_or_default();
    let ids: Vec<AssetId> = rows.iter().map(|r| r.id).collect();
    let blobs = storage::blob_load_all::<Row<K>>(&ids).await;
    let kept: Vec<Entry> = rows
        .into_iter()
        .zip(blobs)
        .filter_map(|(row, png)| {
            png.map(|png| Entry {
                name: row.name,
                png,
                id: row.id,
            })
        })
        .collect();
    if kept.len() != ids.len() {
        tracing::warn!(
            "{} {}(s) had lost their image and were dropped from the library",
            ids.len() - kept.len(),
            K::NOUN
        );
        persist::<K>(&kept);
    }
    kept
}

/// Write the library's rows — [`Entry`] narrowed to what is durable about it.
///
/// The bytes are not this function's to write: an entry reaching here always has them
/// stored already, because every caller put them there first.
///
/// Written whole, and that is what the split bought. A library of a few dozen names
/// and ids is a couple of kilobytes; what happened at this line when the PNGs rode
/// inline was a base64 of every image in the library, on the thread the canvas paints
/// on, per change.
pub fn persist<K: Kind>(entries: &[Entry]) {
    let rows: Vec<Row<K>> = entries
        .iter()
        .map(|e| Row {
            name: e.name.clone(),
            id: e.id,
            kind: PhantomData,
        })
        .collect();
    storage::save_list(&rows);
}

/// Store one entry's bytes. **Before the row that names them**: the other order can
/// leave a library pointing at an asset that was never stored.
pub async fn store_bytes<K: Kind>(id: AssetId, png: &[u8]) {
    storage::blob_save::<Row<K>>(id, png).await;
}

/// Forget one entry's bytes. **After the row that named them**, which is the same rule
/// read the other way: a crash between the two strands some bytes, which costs space,
/// where the other order strands the row, which costs an asset that cannot be used.
pub async fn drop_bytes<K: Kind>(id: AssetId) {
    storage::blob_remove::<Row<K>>(id).await;
}

/// Move an entry to the id its bytes actually canonicalize to.
///
/// For a stored id that predates a canonicalization change: the bytes under their new
/// name, then the rows, then the old name dropped — [`store_bytes`]'s order, for its
/// reason. `entries` is expected to already carry the new id, since the caller is
/// about to paint with it.
pub async fn heal<K: Kind>(entries: &[Entry], stale: AssetId, actual: AssetId, png: &[u8]) {
    store_bytes::<K>(actual, png).await;
    persist::<K>(entries);
    drop_bytes::<K>(stale).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, p: [u8; 4]) -> Decoded {
        Decoded {
            width: w,
            height: h,
            rgba: std::iter::repeat_n(p, (w * h) as usize).flatten().collect(),
        }
    }

    /// An import is capped, never enlarged — a 64-px stamp is a 64-px stamp.
    #[test]
    fn fitting_caps_without_enlarging() {
        assert_eq!(fit(4096, 2048, 1024), (1024, 512));
        assert_eq!(fit(64, 64, 1024), (64, 64));
        // A sliver still has a texel on its short edge rather than none.
        assert_eq!(fit(4096, 1, 1024), (1024, 1));
    }

    /// A light border reads as ink on paper and is inverted; a dark one is already
    /// the way the engine reads coverage and is left alone.
    #[test]
    fn dark_ink_on_light_paper_is_inverted() {
        let (_, inverted) = shape_png(solid(8, 8, [255, 255, 255, 255])).unwrap();
        assert!(
            inverted,
            "a white field is paper with the ink somewhere on it"
        );
        let (_, inverted) = shape_png(solid(8, 8, [0, 0, 0, 255])).unwrap();
        assert!(!inverted, "a black field is already coverage");
    }

    /// A **transparent** border is not paper: coverage is luminance × alpha, so
    /// nothing is covered there whatever the colour underneath says. A white
    /// transparent border inverting would turn every alpha cut-out inside out.
    #[test]
    fn a_transparent_border_is_not_paper() {
        let (_, inverted) = shape_png(solid(8, 8, [255, 255, 255, 0])).unwrap();
        assert!(!inverted);
    }

    /// A substrate's transparency is composited over white rather than multiplied
    /// in, so a PNG with a transparent border does not import as a trench.
    #[test]
    fn a_substrates_transparency_is_its_own_top_surface() {
        let png = substrate_png(solid(4, 4, [0, 0, 0, 0])).unwrap();
        let field = Substrates::canonical(&png).expect("a substrate PNG reads as height");
        assert!(
            field.texels.iter().all(|&t| t == 255),
            "fully transparent black came out as {:?}, not the top surface",
            &field.texels[..4]
        );
    }

    /// The two readings of one image are different fields, which is the whole reason
    /// `Kind` exists — and why a thumbnail cache is per-library.
    #[test]
    fn the_two_kinds_read_one_image_differently() {
        // Mid grey at half alpha: coverage halves it, height ignores alpha entirely.
        let png = encode_rgba(2, 2, &[128, 128, 128, 128].repeat(4)).unwrap();
        let coverage = Shapes::canonical(&png).expect("reads as coverage");
        let height = Substrates::canonical(&png).expect("reads as height");
        assert_ne!(coverage.id(), height.id());
    }

    /// An empty image is refused rather than encoded, and a decoder that miscounts is
    /// caught here rather than by a panic inside the encoder.
    #[test]
    fn a_malformed_decode_is_refused() {
        assert!(shape_png(solid(0, 4, [0; 4])).is_err());
        let short = Decoded {
            width: 4,
            height: 4,
            rgba: vec![0; 8],
        };
        assert!(substrate_png(short).is_err());
    }

    /// Exactly one shipped row is procedural, and it is a substrate: a brush shape
    /// with no image would be a stamp with nothing to stamp.
    #[test]
    fn only_a_substrate_may_be_procedural() {
        assert!(SHIPPED_SHAPES.iter().all(|s| s.path.is_some()));
        assert_eq!(
            SHIPPED_SUBSTRATES
                .iter()
                .filter(|s| s.path.is_none())
                .count(),
            1
        );
        assert_eq!(shipped(SMOOTH).and_then(|s| s.path), None);
    }

    /// Every shipped name is distinct across *both* catalogs, because [`shipped`]
    /// looks a name up in the pair — and a substrate answering a shape's name would
    /// hand a picker the wrong file.
    #[test]
    fn no_two_shipped_assets_share_a_name() {
        let names: Vec<&str> = SHIPPED_SHAPES
            .iter()
            .chain(SHIPPED_SUBSTRATES)
            .map(|s| s.name)
            .collect();
        for (i, name) in names.iter().enumerate() {
            assert!(!names[i + 1..].contains(name), "{name} is claimed twice");
        }
    }

    /// The default a fresh document opens on is a substrate this build actually
    /// ships, so the one `Pick` nothing chose still resolves.
    #[test]
    fn the_default_substrate_is_one_that_ships() {
        let Pick::Builtin(name) = DEFAULT_SUBSTRATE else {
            panic!("the default is named by catalog, not by id");
        };
        assert!(SHIPPED_SUBSTRATES.iter().any(|s| s.name == name));
    }

    /// The catalog names its files as a path here and again as an `asset!` literal in
    /// the web frontend, because a proc macro needs a literal and a lookup needs a
    /// string. Two spellings of one filename is exactly the kind of thing that
    /// drifts, so it is checked rather than trusted: every catalog row must resolve
    /// to a hashed file, and every hashed file must be claimed by a row.
    #[test]
    fn the_catalog_and_the_shipped_files_name_the_same_images() {
        let claimed: Vec<&str> = SHIPPED_SHAPES
            .iter()
            .chain(SHIPPED_SUBSTRATES)
            .filter_map(|s| s.path)
            .collect();
        for path in &claimed {
            assert!(
                shipped_id(path).is_some(),
                "the catalog names {path}, which is not among the shipped assets"
            );
        }
        for (path, _) in SHIPPED_IDS {
            assert!(
                claimed.contains(path),
                "{path} ships and is hashed but no catalog row offers it"
            );
        }
    }

    /// **The catalog is append-only.** Every id here has been shipped, so a saved
    /// document may reference one and rely on this build to supply it (§8's version
    /// 6). Re-authoring a shipped image, or dropping one, does not break a picker —
    /// it strands every painting made on it, which will then refuse to open rather
    /// than open wrong.
    ///
    /// Adding a row is free. Changing or removing one is a decision about other
    /// people's files, so it fails here first: add the new asset alongside, and retire
    /// the old one only when nothing can still be pointing at it.
    ///
    /// **Keyed on the id, not the path**, because the id is the whole of what a
    /// document holds (§19). A file may be renamed — `Gesso.png` became `Rough.png` —
    /// and every painting made on it still opens, while re-authoring it under the same
    /// name strands them all. The path beside each hash says which asset it was, and
    /// nothing looks it up.
    ///
    /// **The two shape rows were re-canonicalized once, in alpha** (2026-08-23), when
    /// a brush shape's canonical form became reach-normalized so that a brush's `size`
    /// names the disc its mark fits in for every shape (`stark_assetid::coverage`,
    /// §6.6). That re-derived every shape id and stranded documents painted on the old
    /// ones — a decision about other people's files that §19 puts at exactly this
    /// rung, and one the alpha window is *for*. The substrate rows did not move, the
    /// height derivation being untouched. Past alpha the rule above stands as
    /// written: append.
    #[test]
    fn the_shipped_catalog_is_append_only() {
        // Shipped ids, oldest first. Append; do not edit.
        const SHIPPED_EVER: &[(&str, &str)] = &[
            (
                "shape/Flat.png",
                "2c484d9d80ad2d087996fe17aade9b6618cf4e91a97a570df8b378cd3de563bb",
            ),
            (
                "shape/Worn_Bristles.png",
                "e572234345c9a1ffdf7df457569a3bc14709437512266f4b9c4fb7f4214bab0d",
            ),
            (
                "substrate/Rough.png",
                "0b88d740a6b3f35f57b5f1d6e4064ac7b4ace0d2c2abab417bbcce762602deb6",
            ),
            (
                "substrate/Linen.png",
                "9d8105e76895f6e47b456177da890816a2983112548d7d748cd42c5d67cd5dc1",
            ),
        ];
        for (path, want) in SHIPPED_EVER {
            assert!(
                SHIPPED_IDS.iter().any(|(_, id)| id.to_hex() == *want),
                "{path} shipped as {want} and this build has no such content; \
                 documents painted on it can no longer be opened. Add the new asset \
                 as a new row rather than re-authoring the old one."
            );
        }
    }

    /// Distinct files must be distinct content; two rows sharing an id would make a
    /// picker offer one thing twice.
    #[test]
    fn every_shipped_asset_is_distinct_content() {
        for (i, (path, id)) in SHIPPED_IDS.iter().enumerate() {
            for (other, other_id) in &SHIPPED_IDS[i + 1..] {
                assert_ne!(id, other_id, "{path} and {other} hash to one id");
            }
        }
    }

    /// The two directions agree: a row's id names the row back.
    #[test]
    fn an_id_names_the_row_it_came_from() {
        for row in SHIPPED_SHAPES.iter().chain(SHIPPED_SUBSTRATES) {
            let Some(path) = row.path else { continue };
            let id = shipped_id(path).expect("every catalog row is hashed");
            assert_eq!(shipped_at(id).map(|r| r.name), Some(row.name));
        }
    }

    /// A row is `{name, id}` in both libraries — the shape every already-stored
    /// library is written in, which the kind must ride along without disturbing.
    #[test]
    fn a_row_is_written_the_way_it_always_was() {
        let row: Row<Shapes> = Row {
            name: "Bristles".to_string(),
            id: AssetId([0xab; 32]),
            kind: PhantomData,
        };
        let json = serde_json::to_string(&row).unwrap();
        assert_eq!(
            json,
            format!(r#"{{"name":"Bristles","id":"{}"}}"#, "ab".repeat(32))
        );
        // And reads back, including into the other library's row type: the JSON
        // carries no kind, which is what makes the two stores the only thing that
        // tells them apart.
        let back: Row<Substrates> = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "Bristles");
    }
}
