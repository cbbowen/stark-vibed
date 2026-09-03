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
//! **Adding a shape is a file plus a row in [`SHAPES`]** — it then appears in
//! the brush editor's gallery and can be named by a default preset.

use dioxus::prelude::*;
use stark_model::document::BrushShape;

use crate::render::Renderer;
use crate::state::AppState;

/// One shape bundled with the app.
pub struct BuiltinShape {
    /// The gallery's label, and the name a default preset asks for it by
    /// ([`shape`]). Not persisted anywhere — a preset stores the resolved
    /// content id — so renaming one is a cosmetic change.
    pub name: &'static str,
    /// The bundled PNG, fetched at runtime.
    pub asset: Asset,
    /// The same file's path under `assets/`, which is how `crate::builtin_ids`
    /// knows this shape's content id without fetching it. Spelled twice because
    /// `asset!` needs a literal and a lookup needs a string; a test checks the two
    /// agree.
    pub path: &'static str,
}

/// The worn bristle shape: a dry, broken-edged tip.
pub const BRISTLES: &str = "Worn Bristles";

/// The flat shape.
pub const FLAT_TIP: &str = "Flat";

/// The pencil shape
pub const PENCIL: &str = "Pencil";

/// Every shape that ships with the app, in gallery order.
pub const SHAPES: &[BuiltinShape] = &[
    BuiltinShape {
        name: BRISTLES,
        asset: asset!("/assets/shape/Worn_Bristles.png"),
        path: "shape/Worn_Bristles.png",
    },
    BuiltinShape {
        name: FLAT_TIP,
        asset: asset!("/assets/shape/Flat.png"),
        path: "shape/Flat.png",
    },
    BuiltinShape {
        name: PENCIL,
        asset: asset!("/assets/shape/Pencil.png"),
        path: "shape/Pencil.png",
    },
];

/// Fetch every bundled shape and import it into `r`, so strokes and presets can
/// reference one by content id. Run on each engine that has to render them: the
/// main canvas at startup, and the brush editor's preview (whose fetches hit the
/// browser cache, and whose ids line up with the main engine's by construction).
///
/// A shape that fails to fetch is logged and skipped — the rest still load, and
/// callers already treat an unresolved built-in as "not available yet".
pub async fn import_all(r: &mut Renderer) {
    for builtin in SHAPES {
        match dioxus::asset_resolver::read_asset_bytes(builtin.asset).await {
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
pub fn resolved(state: AppState) -> Vec<(&'static BuiltinShape, Option<stark_model::AssetId>)> {
    let renderer = state.renderer.read();
    SHAPES
        .iter()
        .map(|b| (b, renderer.as_ref().and_then(|r| r.builtin(b.name))))
        .collect()
}
