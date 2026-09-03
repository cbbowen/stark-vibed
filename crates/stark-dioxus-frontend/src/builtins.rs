//! The brush shapes that ship with the app (§6.6).
//!
//! A built-in is a PNG under `assets/shape/`, bundled as a static file and
//! **fetched at runtime** so the stamps stay out of the wasm binary. Every
//! engine imports the bytes once at startup — the main canvas ([`import_all`]
//! from `main`) and the brush editor's preview — and because the engine keys an
//! asset by the hash of its decoded coverage, all of them land on the same
//! [`AssetId`](stark_model::AssetId). That is what lets a built-in be referenced
//! the way any other
//! stamp is: by content id, with no notion of "built-in" anywhere downstream.
//!
//! The catch is *when*: an id is only knowable once the bytes have arrived, so
//! nothing that runs before the fetch can name one. The app's own brush presets
//! are exactly that case, which is why `crate::presets` installs them after
//! startup rather than at load.
//!
//! Unlike the user's imported shapes (`crate::shapes`) a built-in is not in the
//! library and not in `localStorage`: it comes back with the app, so storing a
//! copy per browser would only be a second, staler one.
//!
//! **Adding a shape is a file plus a row in the catalog**
//! ([`stark_chrome::assets::SHIPPED_SHAPES`]) plus its `asset!` literal below — it
//! then appears in
//! the brush editor's gallery and can be named by a default preset.

use dioxus::prelude::*;
use stark_chrome::assets;
use stark_model::document::BrushShape;

use crate::render::Renderer;
use crate::state::AppState;

/// The bundled PNG behind a catalog row, by the row's path.
///
/// **The one thing the catalog cannot carry.** Which shapes ship and what they are
/// called are `stark_chrome::assets::SHIPPED_SHAPES`, because a name is what a preset
/// asks for and a second catalog would be a second answer to what this build has. An
/// `Asset` is not: `asset!` is a proc macro that demands a path literal inside this
/// crate, so the files are spelled here — and `crate::builtin_ids` has the test that
/// the two spellings agree.
const BUNDLED: &[(&str, Asset)] = &[
    (
        "shape/Worn_Bristles.png",
        asset!("/assets/shape/Worn_Bristles.png"),
    ),
    ("shape/Flat.png", asset!("/assets/shape/Flat.png")),
    ("shape/Pencil.png", asset!("/assets/shape/Pencil.png")),
];

/// The bundled file at `path`, if this catalog claims it.
pub fn bundled_at(path: &str) -> Option<Asset> {
    BUNDLED.iter().find(|(p, _)| *p == path).map(|(_, a)| *a)
}

/// The bundled file for a catalog row.
pub fn bundled(row: &assets::Shipped) -> Option<Asset> {
    bundled_at(row.path?)
}

/// Fetch every bundled shape and import it into `r`, so strokes and presets can
/// reference one by content id. Run on each engine that has to render them: the
/// main canvas at startup, and the brush editor's preview (whose fetches hit the
/// browser cache, and whose ids line up with the main engine's by construction).
///
/// A shape that fails to fetch is logged and skipped — the rest still load, and
/// callers already treat an unresolved built-in as "not available yet".
pub async fn import_all(r: &mut Renderer) {
    for builtin in assets::SHIPPED_SHAPES {
        let Some(file) = bundled(builtin) else {
            continue;
        };
        match dioxus::asset_resolver::read_asset_bytes(file).await {
            Ok(bytes) => r.load_builtin(builtin.name, &bytes),
            Err(e) => tracing::warn!("could not fetch the built-in shape “{}”: {e}", builtin.name),
        }
    }
}

/// The stamp shape `name` refers to, or `None` while its bytes have yet to be
/// fetched and imported (or the canvas is still starting). `peek`, not `read`:
/// this answers commands (a gallery click, seeding the presets), and a command
/// has no business subscribing its caller to the renderer.
pub fn shape(state: AppState, name: &str) -> Option<BrushShape> {
    let renderer = state.renderer.peek();
    let id = renderer.as_ref()?.builtin(name)?;
    Some(BrushShape::Stamp(id))
}

/// Make a built-in the active brush shape. A no-op until its bytes have loaded,
/// like the rest of the gallery's cards.
pub fn select(state: AppState, name: &str) {
    if let Some(shape) = shape(state, name) {
        crate::panels::brush::set_shape(state, shape);
    }
}

/// Every built-in paired with the id it resolved to — `None` for one still in
/// flight. For the gallery, which draws a card per built-in and highlights the
/// one the brush is holding; `read`, so the cards light up when the fetch lands.
pub fn resolved(state: AppState) -> Vec<(&'static assets::Shipped, Option<stark_model::AssetId>)> {
    let renderer = state.renderer.read();
    assets::SHIPPED_SHAPES
        .iter()
        .map(|b| (b, renderer.as_ref().and_then(|r| r.builtin(b.name))))
        .collect()
}
