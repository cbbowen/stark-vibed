//! The floating Lighting panel: the image-based-lighting media pass and the canvas
//! substrate (§6.3, §6.4).

use dioxus::prelude::*;
use stark_model::Srgb;

use crate::icons::{icon, label};
use crate::panels::color::OklabPicker;
use crate::preview;
use crate::state::{AppState, dispatch, use_obs, with_engine, with_engine_quiet};
use crate::widgets::{PopoutId, PreviewSlider, Select, Slider, use_popout};
use dioxus::dioxus_core::spawn_forever;
use stark_engine::command::ViewCommand;
use stark_engine::{EnvironmentId, MediaParams};
use stark_model::{SubstrateId, SubstrateScale};
use stark_ui::lighting::{Dial, ENVIRONMENTS};

/// Built-in assets, bundled as static files and **fetched at runtime** so they
/// stay out of the wasm binary (§6.6). The engine is handed the bytes.
///
/// Canvas substrates are *not* here — they are `crate::substrates`, because a substrate is
/// content-addressed and so needs a resolved-id cache that a bare `Asset` constant has
/// nowhere to keep (§6.4).
pub const ENV_FERNDALE: Asset = asset!("/assets/environment/ferndale_studio_11_1k.hdr");
pub const ENV_BLOEM_HILL: Asset = asset!("/assets/environment/bloem_hill_01_1k.hdr");
pub const ENV_KLOOFENDAL_OVERCAST: Asset =
    asset!("/assets/environment/kloofendal_overcast_puresky_1k.hdr");
pub const ENV_QWANTANI_DUSK: Asset = asset!("/assets/environment/qwantani_dusk_2_puresky_1k.hdr");

/// Lighting controls for the image-based-lighting media pass (§6.3).
/// The canvas is lit by the chosen environment; these tune how it reads. Exposure is
/// not among them — it rides with the environment, so picking a light picks it.
#[component]
pub fn LightingPanel() -> Element {
    let state = use_context::<AppState>();
    // Read off the engine's own projection rather than a local copy: a shadow seeded
    // from `Default` goes stale the moment anything else changes these (§4).
    // …and through a memo, so the panel wakes when one of these moves and sleeps
    // through the strokes and pans that merely rewrite the projection
    // (`state::use_obs`). These are the slowest-moving values in it — a light and a
    // substrate are chosen between passages, not during one.
    //
    // The canvas substrate color (straight sRGB) is shown as a swatch that flies out
    // an Oklab picker ([`SubstrateColorPicker`]). Read from the projection rather than
    // a local signal for the same reason as the rest: it is document state now
    // (§15.5), so a copy here would go stale the moment an undo or a document load
    // moved it (§4).
    let scene = use_obs(state, |o| {
        (
            o.media,
            o.substrate,
            o.substrate_scale,
            o.environment,
            o.substrate_color,
        )
    });
    let (p, surf, scale, env, c) = scene().unwrap_or_else(|| {
        (
            MediaParams::default(),
            SubstrateId::default(),
            SubstrateScale::NATURAL,
            EnvironmentId::default(),
            stark_engine::document::DEFAULT_SUBSTRATE_COLOR,
        )
    });
    // What a release would lay down (`preview::SUBSTRATE_SCALE`). Held rather than read
    // back off `scale` at commit time, which reports the *preview* mid-drag.
    let laying = use_signal(|| None::<SubstrateScale>);
    // The display (§6.5): this browser's choice, and what the canvas in front of it
    // can do. `renderer_ready` is the subscription and `peek` the read: the renderer
    // signal is written on every dispatch, and this panel must not wake per pointer
    // sample.
    let hdr = crate::state::use_pref(state, |p| p.hdr)();
    let ready = *state.renderer_ready.read();
    let (hdr_capable, display_headroom) = if ready {
        state.renderer.peek().as_ref().map_or((false, None), |r| {
            (r.session.hdr_capable(), r.display_headroom())
        })
    } else {
        (false, None)
    };
    let swatch = crate::cards::swatch_style(c.get());
    // Lit while its pop-out is open — the well is a column away from what it opened,
    // and nothing else says which press put that surface there.
    let color_open = use_popout(state, PopoutId::SubstrateColor);
    use_popout(state, PopoutId::SubstrateGallery);
    let swatch_class = if color_open { "swatch open" } else { "swatch" };
    let environment = ENVIRONMENTS.iter().position(|(id, _)| *id == env);
    rsx! {
        // Each track's ends, its mark and its caption come off the dial rather than
        // being spelled here: the gloss ceiling in particular was a literal on this
        // side and a named constant on the native one whose doc said it was copying
        // this slider (§11.2).
        Slider { label: Dial::Impasto.label(), glyph: Dial::Impasto.glyph(),
            min: Dial::Impasto.range().0, max: Dial::Impasto.range().1, value: p.height_strength,
            oninput: move |v| update_media(state, move |m| m.height_strength = v) }
        Slider { label: Dial::Texture.label(), glyph: Dial::Texture.glyph(),
            min: Dial::Texture.range().0, max: Dial::Texture.range().1, value: p.substrate_strength,
            oninput: move |v| update_media(state, move |m| m.substrate_strength = v) }
        Slider { label: Dial::Gloss.label(), glyph: Dial::Gloss.glyph(),
            min: Dial::Gloss.range().0, max: Dial::Gloss.range().1, value: p.specular,
            oninput: move |v| update_media(state, move |m| m.specular = v) }
        // The canvas colour, and the surface it is laid on: the two choices in this
        // panel that are made by *looking*, and so the two that want more room than a
        // 300px column has. Both fly out beside the panel now
        // (`panels::popout::StackPopouts`); what stays here is the well that says
        // which one is in force.
        //
        // `marked` by hand on the rows that hold a well or a drop-down rather than a
        // `Slider`, since each wears a glyph and that is all `marked` claims: the panel
        // folds as one column in minimal mode (§11).
        //
        // The `data-popout` attribute is on the **row**, not on the well inside it,
        // and it is what the pop-out is placed against (`PopoutId::in_stack`).
        div { class: "slider-row marked", "data-popout": PopoutId::SubstrateColor.key(),
            div { class: "slider-label", {icon(stark_ui::icons::CANVAS)} {label("Background")} }
            button {
                class: swatch_class,
                style: "{swatch}",
                onclick: move |_| crate::widgets::toggle_popout(state, PopoutId::SubstrateColor),
            }
        }
        div { class: "slider-row marked", "data-popout": PopoutId::SubstrateGallery.key(),
            div { class: "slider-label", {icon(stark_ui::icons::SURFACE)} {label("Surface")} }
            super::substrates::SubstrateWell {}
        }
        // How large the substrate is laid (§6.4). The percentage is the readout
        // because "the same as last time" is judged by the figure.
        PreviewSlider {
            label: "Scale",
            glyph: stark_ui::icons::SUBSTRATE_SCALE,
            min: f32::from(SubstrateScale::MIN),
            max: f32::from(SubstrateScale::MAX),
            // The track steps on the value's own ladder, so it cannot offer a position
            // `SubstrateScale::new` would move the handle off.
            step: f32::from(SubstrateScale::STEP),
            value: f32::from(scale.percent()),
            readout: "{scale.percent()}%",
            // Inert on a flat surface, whose height is a constant: there is nothing to size.
            disabled: surf == SubstrateId::Flat,
            preview: preview::SUBSTRATE_SCALE,
            pending: laying,
            map: |v: f32| Some(SubstrateScale::new(v.round() as u16)),
        }
        div { class: "slider-row marked",
            div { class: "slider-label", {icon(stark_ui::icons::LIGHT)} {label("Light")} }
            Select {
                options: ENVIRONMENTS.iter().map(|(_, name)| *name).collect::<Vec<_>>(),
                selected: environment,
                onchange: move |i: usize| set_environment(state, ENVIRONMENTS[i].0),
            }
        }
        // The HDR switch (§6.5): off is the picture an export makes. Only on a
        // canvas that can show more; `marked` by hand on the rows above's terms.
        if hdr_capable {
            div { class: "slider-row marked",
                div { class: "slider-label", {icon(stark_ui::icons::HDR)} {label("HDR")} }
                input {
                    class: "setting-check",
                    r#type: "checkbox",
                    checked: hdr.on,
                    onchange: move |e| crate::prefs::set(state, |p| p.hdr.on = e.checked()),
                }
            }
            // The slider stands in for a headroom the platform will not report
            // (`Renderer::display_headroom`); a figure the screen states is used as is.
            if hdr.on && display_headroom.is_none() {
                Slider { label: Dial::Headroom.label(), glyph: Dial::Headroom.glyph(),
                    min: Dial::Headroom.range().0, max: Dial::Headroom.range().1,
                    value: hdr.clamped_headroom(),
                    oninput: move |v| crate::prefs::set_unsaved(state, move |p| p.hdr.headroom = v),
                    onsettle: move |_| crate::prefs::save(state) }
            }
        }
    }
}

/// Tell the engine what the screen is (§6.5) — `Session::apply_output` over this
/// browser's choice and the canvas in front of it. Run once the renderer is up
/// (`prefs::load_engine`) and whenever either half moves.
///
/// Through [`with_engine`] rather than `dispatch`: the output is no deed the tour
/// counts and commits nothing to broadcast, and the projection it moves is published.
pub fn apply_output(state: AppState) {
    let choice = state.prefs.peek().hdr;
    with_engine(state, |r| {
        let display = r.display_headroom();
        r.session.apply_output(choice, display);
    });
}

/// The canvas colour's picker, as flown out beside the Lighting panel
/// (`widgets::PopoutId::SubstrateColor`).
///
/// It stood *in* the panel until the column it grew by turned out to be the whole
/// complaint: a picker is 220px of wheel plus a track and a field, which is most of a
/// screen's worth of stack, and it pushed the light and the substrate scale below the
/// fold for as long as it was open. Beside the panel it costs the column nothing.
///
/// Mounted only while open — by the layer that places it, which is what makes this a
/// component rather than a block of markup — so the picker re-seeds from the current
/// colour each time it is opened, the same bargain the frame bar's copy makes.
#[component]
pub fn SubstrateColorPicker() -> Element {
    let state = use_context::<AppState>();
    // The colour the picker opens on, off the projection like the swatch that opened
    // it: document state, so a local copy would go stale under an undo (§4, §15.5).
    let c = use_obs(state, |o| o.substrate_color)()
        .unwrap_or(stark_engine::document::DEFAULT_SUBSTRATE_COLOR);
    rsx! {
        OklabPicker {
            init: (c).get(),
            // Previewed while the pointer is down, committed once on release:
            // the substrate color is document state, so one drag has to cost
            // one undo step (and one replicated action) rather than one per
            // pointer sample — the same bargain the frame drag makes.
            onchange: move |rgb: [f32; 3]| preview::BACKGROUND.show(state, Srgb::new(rgb)),
            oncommit: move |rgb: [f32; 3]| preview::BACKGROUND.commit(state, Srgb::new(rgb)),
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

/// The bundled HDR behind an image-backed environment (`None` for the procedural
/// `Neutral`, which is generated on the GPU side and needs no bytes). The one place
/// to map a new [`ENVIRONMENTS`] row to its file.
pub fn environment_asset(id: EnvironmentId) -> Option<Asset> {
    match id {
        EnvironmentId::Neutral => None,
        EnvironmentId::Ferndale => Some(ENV_FERNDALE),
        EnvironmentId::BloemHill => Some(ENV_BLOEM_HILL),
        EnvironmentId::KloofendalOvercast => Some(ENV_KLOOFENDAL_OVERCAST),
        EnvironmentId::QwantaniDusk => Some(ENV_QWANTANI_DUSK),
    }
}

/// Re-light the canvas with `id` and repaint. A view setting: no stored pixel moves,
/// only how the relief catches the light (§6.3). HDR-backed environments
/// are fetched on first use — the same `spawn_forever` + register-then-switch shape
/// as [`crate::substrates::select`], for the same reason: closing the panel mid-fetch
/// must not cancel the switch.
///
/// The switch itself goes through [`dispatch`], not through the renderer signal, and
/// that is not merely tidiness: the picker reads the environment off the observable
/// projection (§4), so a switch that skipped the `observe` refresh would leave the
/// panel re-rendering the *previous* light onto the `select` — the pick would show,
/// flash back and stay wrong until some other command happened to refresh `obs`.
pub fn set_environment(state: AppState, id: EnvironmentId) {
    spawn_forever(async move {
        let needs_bytes = state
            .renderer
            .read()
            .as_ref()
            .is_some_and(|r| !r.session.engine().environment_loaded(id));
        if needs_bytes && let Some(asset) = environment_asset(id) {
            tracing::info!(environment = ?id, url = %asset, "fetching environment asset");
            match dioxus::asset_resolver::read_asset_bytes(asset).await {
                Ok(bytes) => {
                    // Quiet: registering bytes readies a light without switching to
                    // it. The switch on the next line is the state change, and it is
                    // a command, so it publishes.
                    // A light that will not decode is reported and skipped, exactly
                    // as one that would not fetch is on the arm below: the canvas
                    // keeps the light it has rather than losing the tab to a
                    // decoder panic.
                    if let Some(Err(e)) = with_engine_quiet(state, |r| {
                        r.session.engine_mut().register_environment(id, bytes)
                    }) {
                        tracing::warn!(environment = ?id, "environment will not decode: {e}");
                        return;
                    }
                }
                Err(e) => {
                    tracing::warn!("environment fetch failed: {e}");
                    return;
                }
            }
        }
        dispatch(state, ViewCommand::SetEnvironment(id));
    });
}
