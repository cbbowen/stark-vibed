//! The floating Lighting panel: the image-based-lighting media pass and the canvas
//! surface (DESIGN.md §6.3, §6.4).

use dioxus::prelude::*;

use crate::panels::color::OklabPicker;
use crate::render::BG;
use crate::state::AppState;
use crate::widgets::Slider;
use dioxus::dioxus_core::spawn_forever;
use stark_core::{MediaParams, SurfaceId};

/// Built-in assets, bundled as static files and **fetched at runtime** so they
/// stay out of the wasm binary (DESIGN.md §6.6). The engine is handed the bytes.
pub const SURFACE_LINEN: Asset = asset!("/assets/surface/Linen.png");
pub const ENV_FERNDALE: Asset = asset!("/assets/environment/ferndale_studio_11_1k.hdr");

/// The selectable canvas surfaces, in display order (DESIGN.md §6.4). Adding a
/// surface = one row here (plus its asset fetch in [`set_surface`]); the Lighting
/// panel's drop-down renders this table.
pub const SURFACES: &[(SurfaceId, &str)] =
    &[(SurfaceId::Flat, "Smooth"), (SurfaceId::Linen, "Linen")];

/// Lighting controls for the image-based-lighting media pass (DESIGN.md §6.3).
/// The canvas is lit by the studio HDR environment; these tune how it reads.
#[component]
pub fn LightingPanel() -> Element {
    let state = use_context::<AppState>();
    // Seeded from the engine defaults; this panel owns the live values (lighting is
    // a view setting, not part of the observable document state).
    let media = use_signal(MediaParams::default);
    let p = media();
    // The canvas substrate colour (straight sRGB), shown as a swatch that pops out an
    // Oklab picker. Like the sliders, a view setting owned here (`Renderer::set_background`).
    let mut bg = use_signal(|| [BG.r as f32, BG.g as f32, BG.b as f32]);
    let mut show_bg_picker = use_signal(|| false);
    let c = bg();
    let swatch = format!(
        "background: rgb({:.1}% {:.1}% {:.1}%);",
        c[0] * 100.0,
        c[1] * 100.0,
        c[2] * 100.0
    );
    // The canvas surface (weave), switchable in place — the document is preserved;
    // existing paint re-reads against the new bump (DESIGN.md §6.4). Reading the
    // renderer signal keeps the drop-down in sync after the async switch lands.
    let surf = state
        .renderer
        .read()
        .as_ref()
        .map(|r| r.surface())
        .unwrap_or_default();

    rsx! {
        Slider { label: "Exposure", min: 0.1, max: 2.0, value: p.exposure,
            oninput: move |v| update_media(state, media, move |m| m.exposure = v) }
        Slider { label: "Relief", min: 0.0, max: 0.6, value: p.height_strength,
            oninput: move |v| update_media(state, media, move |m| m.height_strength = v) }
        Slider { label: "Weave", min: 0.0, max: 1.5, value: p.surface_strength,
            oninput: move |v| update_media(state, media, move |m| m.surface_strength = v) }
        Slider { label: "Wet gloss", min: 0.0, max: 0.35, value: p.specular,
            oninput: move |v| update_media(state, media, move |m| m.specular = v) }
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
        // Pop-out colour selector: mounted only while open, so the picker re-seeds from
        // the current colour each time. Positioned by `.color-popout` (flies out beside
        // the panel, whose `.panel` is the nearest positioned ancestor).
        if show_bg_picker() {
            div { class: "color-popout",
                OklabPicker {
                    init: bg(),
                    onchange: move |rgb: [f32; 3]| {
                        bg.set(rgb);
                        update_background(state, rgb);
                    },
                }
            }
        }
    }
}

/// Mutate the lighting params in place, push them to the engine, and repaint.
fn update_media(state: AppState, mut media: Signal<MediaParams>, f: impl FnOnce(&mut MediaParams)) {
    let mut p = media();
    f(&mut p);
    media.set(p);
    let mut renderer = state.renderer;
    let mut guard = renderer.write();
    if let Some(r) = guard.as_mut() {
        r.set_media_params(p);
        r.paint();
    }
}

/// Set the canvas substrate colour (straight sRGB, a view setting) and repaint.
fn update_background(state: AppState, rgb: [f32; 3]) {
    let mut renderer = state.renderer;
    let mut guard = renderer.write();
    if let Some(r) = guard.as_mut() {
        r.set_background(rgb);
        r.paint();
    }
}

/// The bundled asset behind an image-backed surface (`None` for procedural ones,
/// which need no bytes). The one place to map a new [`SURFACES`] row to its file.
pub fn surface_asset(id: SurfaceId) -> Option<Asset> {
    match id {
        SurfaceId::Flat => None,
        SurfaceId::Linen => Some(SURFACE_LINEN),
    }
}

/// Switch the canvas surface in place and repaint — the document is preserved;
/// existing paint re-reads against the new weave (DESIGN.md §6.4). Image-backed
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
