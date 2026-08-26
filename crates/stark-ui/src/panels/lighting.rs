//! The floating Lighting panel: the image-based-lighting media pass and the canvas
//! substrate (§6.3, §6.4).

use dioxus::prelude::*;
use stark_model::Srgb;

use crate::icons::{self, icon, label};
use crate::panels::color::OklabPicker;
use crate::preview;
use crate::state::{AppState, dispatch, use_obs, with_engine_quiet};
use crate::widgets::{PopoutId, Slider, slider_fill};
use dioxus::dioxus_core::spawn_forever;
use stark_engine::command::ViewCommand;
use stark_engine::{EnvironmentId, MediaParams};
use stark_model::{SubstrateId, SubstrateScale};

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

/// The selectable lighting environments, in display order (§6.3). One row per
/// environment, its bytes (if any) resolved by
/// [`environment_asset`]. `Neutral` leads because it is the reference light — the
/// achromatic one you switch to to judge color; the HDRs are the room you paint in.
pub const ENVIRONMENTS: &[(EnvironmentId, &str)] = &[
    (EnvironmentId::Neutral, "Neutral"),
    (EnvironmentId::Ferndale, "Ferndale studio"),
    (EnvironmentId::BloemHill, "Bloem hill"),
    (EnvironmentId::KloofendalOvercast, "Kloofendal overcast"),
    (EnvironmentId::QwantaniDusk, "Qwantani dusk"),
];

/// What the app lights the canvas with on startup: the achromatic reference light,
/// which is also what the engine boots on. Paint reads as its own color under it —
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
    let swatch = format!(
        "background: rgb({:.1}% {:.1}% {:.1}%);",
        c[0] * 100.0,
        c[1] * 100.0,
        c[2] * 100.0
    );
    // Lit while its pop-out is open — the well is a column away from what it opened,
    // and nothing else says which press put that surface there.
    let swatch_class = if crate::widgets::popout_open(state, PopoutId::SubstrateColor) {
        "swatch open"
    } else {
        "swatch"
    };
    // Both of this panel's pop-outs go when the panel does. They are mounted at the
    // app root rather than in here (`panels::popout`), so nothing takes one off the
    // screen on the way out — and the next time the panel was opened the picker
    // would be standing beside a row nobody had pressed. The frame bar's own
    // pop-out makes the same promise for the same reason (`panels::frame`).
    use_drop(move || {
        let mine = matches!(
            *state.popout.peek(),
            Some(PopoutId::SubstrateColor | PopoutId::SubstrateGallery)
        );
        if mine {
            crate::widgets::close_popout(state);
        }
    });
    rsx! {
        Slider { label: "Impasto", glyph: icons::IMPASTO, min: 0.0, max: 1.0, value: p.height_strength,
            oninput: move |v| update_media(state, move |m| m.height_strength = v) }
        Slider { label: "Texture", glyph: icons::TEXTURE, min: 0.0, max: 1.0, value: p.substrate_strength,
            oninput: move |v| update_media(state, move |m| m.substrate_strength = v) }
        Slider { label: "Gloss", glyph: icons::GLOSS, min: 0.0, max: 0.35, value: p.specular,
            oninput: move |v| update_media(state, move |m| m.specular = v) }
        // The canvas colour, and the surface it is laid on: the two choices in this
        // panel that are made by *looking*, and so the two that want more room than a
        // 300px column has. Both fly out beside the panel now
        // (`panels::popout::StackPopouts`); what stays here is the well that says
        // which one is in force.
        //
        // `marked` by hand on these four, as `widgets::Slider` sets it on the three
        // above: each holds a well, a native `select` or a range that previews and
        // commits, so none of them can be the component — but each wears a glyph, and
        // that is the whole of what `marked` claims. The panel folds as one column in
        // minimal mode rather than three rows folding and four standing on (§11).
        //
        // The `data-popout` attribute is on the **row**, not on the well inside it,
        // and it is what the pop-out is placed against — see `PopoutId::in_stack`
        // for why the row is the right box to measure.
        div { class: "slider-row marked", "data-popout": "substrate-color",
            div { class: "slider-label", {icon(icons::CANVAS)} {label("Background")} }
            button {
                class: swatch_class,
                style: "{swatch}",
                onclick: move |_| crate::widgets::toggle_popout(state, PopoutId::SubstrateColor),
            }
        }
        div { class: "slider-row marked", "data-popout": "substrate-gallery",
            div { class: "slider-label", {icon(icons::SURFACE)} {label("Surface")} }
            crate::substrates::SubstrateWell {}
        }
        // How large the substrate is laid (§6.4). A raw range rather than `Slider`,
        // because this one is document state: it previews per sample and commits
        // once, which needs the three drag-ending events `Slider` does not carry
        // (`preview::SUBSTRATE_SCALE`, and `Preview::settle` for why there are three).
        //
        // The percentage is in the label because it is the one number here worth
        // reading back — a substrate is judged by eye, but "the same as last time" is
        // judged by the figure. It sits *beside* the hideable word rather than inside
        // it: minimal mode takes a control's name, never its value, so what is left is
        // the mark and the figure — which is the pair that was worth keeping anyway.
        div { class: "slider-row marked",
            div { class: "slider-label",
                {icon(icons::SUBSTRATE_SCALE)}
                {label("Scale")}
                "{scale.percent()}%"
            }
            input {
                class: "slider",
                style: slider_fill(
                    SubstrateScale::MIN as f32,
                    SubstrateScale::MAX as f32,
                    scale.percent() as f32,
                ),
                r#type: "range",
                min: "{SubstrateScale::MIN}",
                max: "{SubstrateScale::MAX}",
                // The control steps on the same ladder the value does, so the track
                // cannot offer a position `SubstrateScale::new` would move the handle
                // off (§6.4).
                step: "{SubstrateScale::STEP}",
                value: "{scale.percent()}",
                // Inert on `Smooth`, whose height is a constant: there is no substrate to
                // size, and a live slider would claim otherwise.
                disabled: surf == SubstrateId::Flat,
                oninput: move |e| {
                    if let Ok(v) = e.value().parse::<u16>() {
                        preview::SUBSTRATE_SCALE.during(state, laying, SubstrateScale::new(v));
                    }
                },
                onchange: move |_| preview::SUBSTRATE_SCALE.settle(state, laying),
                onpointerup: move |_| preview::SUBSTRATE_SCALE.settle(state, laying),
                onpointercancel: move |_| preview::SUBSTRATE_SCALE.settle(state, laying),
            }
        }
        div { class: "slider-row marked",
            div { class: "slider-label", {icon(icons::LIGHT)} {label("Light")} }
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
    }
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
            .is_some_and(|r| !r.environment_loaded(id));
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
                    if let Some(Err(e)) =
                        with_engine_quiet(state, |r| r.register_environment(id, bytes))
                    {
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
