//! The floating Lighting panel: the image-based-lighting media pass and the canvas
//! surface (§6.3, §6.4).

use dioxus::prelude::*;

use crate::panels::color::OklabPicker;
use crate::state::{AppState, dispatch};
use crate::widgets::Slider;
use dioxus::dioxus_core::spawn_forever;
use stark_core::command::{DocCommand, ViewCommand};
use stark_core::{EnvironmentId, MediaParams, SurfaceId};

/// Built-in assets, bundled as static files and **fetched at runtime** so they
/// stay out of the wasm binary (§6.6). The engine is handed the bytes.
pub const SURFACE_LINEN: Asset = asset!("/assets/surface/Linen.png");
pub const SURFACE_GESSO: Asset = asset!("/assets/surface/Gesso.png");
pub const ENV_FERNDALE: Asset = asset!("/assets/environment/ferndale_studio_11_1k.hdr");

/// The selectable canvas surfaces, in display order (§6.4). Adding a
/// surface = one row here (plus its asset fetch in [`set_surface`]); the Lighting
/// panel's drop-down renders this table.
pub const SURFACES: &[(SurfaceId, &str)] = &[
    (SurfaceId::Flat, "Smooth"),
    (SurfaceId::Linen, "Linen"),
    (SurfaceId::Gesso, "Gesso"),
];

/// The selectable lighting environments, in display order (§6.3). Same
/// shape as [`SURFACES`]: one row per environment, its bytes (if any) resolved by
/// [`environment_asset`]. `Neutral` leads because it is the reference light — the
/// achromatic one you switch to to judge colour; the HDRs are the room you paint in.
pub const ENVIRONMENTS: &[(EnvironmentId, &str)] = &[
    (EnvironmentId::Neutral, "Neutral"),
    (EnvironmentId::Ferndale, "Ferndale studio"),
];

/// What the app lights the canvas with on startup: the achromatic reference light,
/// which is also what the engine boots on. Paint reads as its own colour under it —
/// at `Neutral`'s exposure of 1.0 the media pass is an identity (§6.3) — so
/// what you mix is what you see, and the studio HDR is the deliberate switch into a
/// room. Kept a named constant because the startup hook in `main.rs` fetches its
/// bytes if it has any; `Neutral` is procedural, so today that fetch is skipped.
pub const DEFAULT_ENVIRONMENT: EnvironmentId = EnvironmentId::Neutral;

/// Lighting controls for the image-based-lighting media pass (§6.3).
/// The canvas is lit by the chosen environment; these tune how it reads. Exposure is
/// not among them — it rides with the environment, so picking a light picks it.
#[component]
pub fn LightingPanel() -> Element {
    let state = use_context::<AppState>();
    // Read off the engine's own projection rather than a local copy: a shadow seeded
    // from `Default` goes stale the moment anything else changes these (§4).
    let obs = state.obs.read();
    let p = obs.as_ref().map(|o| o.media).unwrap_or_default();
    let surf = obs.as_ref().map(|o| o.surface).unwrap_or_default();
    let env = obs.as_ref().map(|o| o.environment).unwrap_or_default();
    // The canvas substrate colour (straight sRGB), shown as a swatch that pops out an
    // Oklab picker. Read from the engine's projection rather than a local signal:
    // it is document state now (§15.5), so a copy here would go stale
    // the moment an undo or a document load moved it (§4).
    let c = obs
        .as_ref()
        .map(|o| o.background)
        .unwrap_or(stark_core::document::DEFAULT_BACKGROUND);
    drop(obs);
    let mut show_bg_picker = use_signal(|| false);
    let swatch = format!(
        "background: rgb({:.1}% {:.1}% {:.1}%);",
        c[0] * 100.0,
        c[1] * 100.0,
        c[2] * 100.0
    );
    rsx! {
        Slider { label: "Impasto", min: 0.0, max: 1.0, value: p.height_strength,
            oninput: move |v| update_media(state, move |m| m.height_strength = v) }
        Slider { label: "Texture", min: 0.0, max: 1.0, value: p.surface_strength,
            oninput: move |v| update_media(state, move |m| m.surface_strength = v) }
        Slider { label: "Gloss", min: 0.0, max: 0.35, value: p.specular,
            oninput: move |v| update_media(state, move |m| m.specular = v) }
        div { class: "slider-row",
            div { class: "slider-label", "Canvas" }
            button {
                class: "swatch",
                style: "{swatch}",
                onclick: move |_| show_bg_picker.set(!show_bg_picker()),
            }
        }
        div { class: "slider-row",
            div { class: "slider-label", "Surface" }
            select {
                class: "select",
                onchange: move |e| {
                    if let Some((id, _)) = SURFACES.iter().find(|(s, _)| format!("{s:?}") == e.value()) {
                        set_surface(state, *id);
                    }
                },
                for (id, name) in SURFACES.iter().copied() {
                    option { value: "{id:?}", selected: surf == id, "{name}" }
                }
            }
        }
        div { class: "slider-row",
            div { class: "slider-label", "Light" }
            select {
                class: "select",
                onchange: move |e| {
                    if let Some((id, _)) = ENVIRONMENTS.iter().find(|(v, _)| format!("{v:?}") == e.value()) {
                        set_environment(state, *id);
                    }
                },
                for (id, name) in ENVIRONMENTS.iter().copied() {
                    option { value: "{id:?}", selected: env == id, "{name}" }
                }
            }
        }
        // Pop-out colour selector: mounted only while open, so the picker re-seeds from
        // the current colour each time. Positioned by `.color-popout` (flies out beside
        // the panel, whose `.panel` is the nearest positioned ancestor).
        if show_bg_picker() {
            div { class: "color-popout",
                OklabPicker {
                    init: c,
                    // Previewed while the pointer is down, committed once on release:
                    // the substrate colour is document state, so one drag has to cost
                    // one undo step (and one replicated action) rather than one per
                    // pointer sample — the same bargain the frame drag makes.
                    onchange: move |rgb: [f32; 3]| preview_background(state, rgb),
                    oncommit: move |rgb: [f32; 3]| update_background(state, rgb),
                }
            }
        }
    }
}

/// Mutate the lighting params in place, push them to the engine, and repaint.
/// Read the current media params off the observable projection, mutate a copy, and
/// push it back — the same read-modify-commit shape as `update_brush`.
fn update_media(state: AppState, f: impl FnOnce(&mut MediaParams)) {
    let mut p = state
        .obs
        .read()
        .as_ref()
        .map(|o| o.media)
        .unwrap_or_default();
    f(&mut p);
    dispatch(state, ViewCommand::SetMediaParams(p));
}

/// Show a substrate colour without logging it — every sample of a picker drag
/// (§15.5). The engine renders it and reports it back through
/// `observe`, so the canvas and the swatch both track the pointer.
fn preview_background(state: AppState, rgb: [f32; 3]) {
    dispatch(state, ViewCommand::PreviewBackground(Some(rgb)));
}

/// Commit the canvas substrate colour (straight sRGB) — once, when the pick ends.
/// A logged document edit, not a view setting: the ground a piece was painted on is
/// part of what it is, and it is saved with it (§15.5).
fn update_background(state: AppState, rgb: [f32; 3]) {
    dispatch(state, DocCommand::SetBackground(rgb));
}

/// The bundled asset behind an image-backed surface (`None` for procedural ones,
/// which need no bytes). The one place to map a new [`SURFACES`] row to its file.
pub fn surface_asset(id: SurfaceId) -> Option<Asset> {
    match id {
        SurfaceId::Flat => None,
        SurfaceId::Linen => Some(SURFACE_LINEN),
        SurfaceId::Gesso => Some(SURFACE_GESSO),
    }
}

/// Switch the canvas surface in place and repaint — the document is preserved;
/// existing paint re-reads against the new weave (§6.4). Image-backed
/// surfaces are fetched on first use (the bump maps stay out of the wasm binary),
/// so this runs async, like `new_document`'s fetch.
pub fn set_surface(state: AppState, id: SurfaceId) {
    let mut renderer = state.renderer;
    // `spawn_forever`: the caller is LightingPanel's scope, and hiding the
    // panel mid-fetch must not cancel the switch (only root-owned signals are
    // touched, so outliving the panel is safe).
    spawn_forever(async move {
        let needs_bytes = renderer
            .read()
            .as_ref()
            .is_some_and(|r| !r.surface_loaded(id));
        if needs_bytes && let Some(asset) = surface_asset(id) {
            tracing::info!(surface = ?id, url = %asset, "fetching surface asset");
            match dioxus::asset_resolver::read_asset_bytes(asset).await {
                Ok(bytes) => {
                    if let Some(r) = renderer.write().as_mut() {
                        r.register_surface(id, bytes);
                    }
                }
                Err(e) => {
                    tracing::warn!("surface fetch failed: {e}");
                    return;
                }
            }
        }
        if let Some(r) = renderer.write().as_mut() {
            r.set_surface(id);
            r.paint();
        }
    });
}

/// The bundled HDR behind an image-backed environment (`None` for the procedural
/// `Neutral`, which is generated on the GPU side and needs no bytes). The one place
/// to map a new [`ENVIRONMENTS`] row to its file.
pub fn environment_asset(id: EnvironmentId) -> Option<Asset> {
    match id {
        EnvironmentId::Neutral => None,
        EnvironmentId::Ferndale => Some(ENV_FERNDALE),
    }
}

/// Re-light the canvas with `id` and repaint. A view setting: no stored pixel moves,
/// only how the relief catches the light (§6.3). HDR-backed environments
/// are fetched on first use — the same `spawn_forever` + register-then-switch shape
/// as [`set_surface`], for the same reason: closing the panel mid-fetch must not
/// cancel the switch.
pub fn set_environment(state: AppState, id: EnvironmentId) {
    let mut renderer = state.renderer;
    spawn_forever(async move {
        let needs_bytes = renderer
            .read()
            .as_ref()
            .is_some_and(|r| !r.environment_loaded(id));
        if needs_bytes && let Some(asset) = environment_asset(id) {
            tracing::info!(environment = ?id, url = %asset, "fetching environment asset");
            match dioxus::asset_resolver::read_asset_bytes(asset).await {
                Ok(bytes) => {
                    if let Some(r) = renderer.write().as_mut() {
                        r.register_environment(id, bytes);
                    }
                }
                Err(e) => {
                    tracing::warn!("environment fetch failed: {e}");
                    return;
                }
            }
        }
        if let Some(r) = renderer.write().as_mut() {
            r.set_environment(id);
            r.paint();
        }
    });
}
