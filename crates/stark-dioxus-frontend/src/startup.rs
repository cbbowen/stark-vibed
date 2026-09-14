//! What the app does once, when it starts: read this browser's records, then build
//! the engine and everything that has to wait for one.

use dioxus::prelude::*;

use crate::render::{self, CANVAS_ID};
use crate::state::{self, AppState, update_brush};
use crate::{builtins, collab, commands, drags, files, gradients, images, library};
use crate::{panels, platform, presets, slots, substrates, tutor};
use stark_engine::command::ViewCommand;
use stark_ui::lighting::DEFAULT_ENVIRONMENT;

/// Put this browser's stored records where the chrome reads them — before the first
/// render, so it is already the screen the artist left and the first keystroke or
/// press asks the tables they set.
///
/// The shape and substrate libraries are not here: their bytes are a fetch
/// (§25.6), so [`run`] loads them.
pub fn load_records(state: AppState) {
    presets::load(state);
    gradients::load(state);
    slots::load(state);
    commands::load(state);
    drags::load(state);
}

/// Build the engine and bring the app up on it. Nothing runs past a canvas that
/// cannot be drawn on (`render::StartupFailure`).
///
/// The order, and what holds each step where it is:
///
/// 1. **The bundled brush shapes, before any preset names one.** A stamp is named by
///    the hash of its bytes, so the ids do not exist until the import.
/// 2. **The default substrate, before the document opens on it** — for the same
///    reason. Through a new document rather than `SetSubstrate`, so no empty first
///    step lands in every fresh document's undo history. It replaces a document
///    nobody can have touched, because the renderer is not published until step 5.
/// 3. **The default light.** Only a view setting; any later step would do.
/// 4. **Re-read the canvas size after the last fetch**, with no `await` before the
///    publish: every resize reported during the fetches was dropped.
/// 5. **Publish the renderer.** Everything below needs the engine.
/// 6. **The engine-owned preferences, with no `await` since step 5.** Until they land
///    the projection holds the engine's defaults, and a preference changed in that
///    gap would save those over the stored ones (`prefs::set` reads the projection).
///    Also before a session is joined, which reads the peer-outline switch.
/// 7. **The shape and substrate libraries, before the first preset is worn** — a
///    library not yet arrived puts a custom-stamped preset on the round tip, and
///    leaves a file's substrate unnamed.
/// 8. **The app's own presets, after the imports of step 1, then the rack from
///    them**: a built-in names a bundled stamp by id, and a slot takes the preset
///    that declares its digit.
/// 9. **The first preset and the opening color**, pushed to an engine that opened on
///    black.
/// 10. **Join a session named in the URL; bind the file launch and paste**, which
///     both need an engine to load into.
/// 11. **The tour, last**: every step above dispatches on the user's behalf, and a
///     tour listening to them would count the app starting as the artist at work.
pub async fn run(state: AppState) {
    let mut r = match render::init(platform::canvas_by_id(CANVAS_ID)).await {
        Ok(r) => r,
        Err(why) => {
            tracing::error!(%why, "the canvas cannot be drawn on");
            let mut failed = state.startup_failure;
            failed.set(Some(why));
            return;
        }
    };

    // 1.
    builtins::import_all(&mut r).await;
    // 2.
    let color_space = r.desk.engine().color_space();
    substrates::open_default(&mut r, color_space).await;
    // 3.
    light_default(&mut r).await;
    // 4.
    r.sync_to_canvas();
    r.paint();
    // 5.
    state::publish_renderer(state, r);
    // 6.
    crate::prefs::load_engine(state);

    // 7.
    library::load::<stark_ui::assets::Shapes>(state).await;
    library::load::<stark_ui::assets::Substrates>(state).await;
    // 8. Every start, not only a first one, so an improved default reaches a browser
    // that has been running Stark for months.
    presets::install_builtins(state);
    slots::seed_defaults(state);
    // 9. Both are `SetBrush`, which is session state: no undo step.
    presets::apply_first(state);
    update_brush(state, |_, t| t.color = stark_ui::color::INITIAL_COLOR);
    // 10.
    if let Some(ticket) = collab::url_ticket() {
        tracing::info!("joining shared session from URL fragment");
        collab::join(state, ticket);
    }
    files::bind_file_launch(state);
    images::bind_paste(state);
    // 11.
    tutor::begin(state);
}

/// Light the canvas with the bundled default environment. A no-op while the
/// default is the procedural one, and the procedural one stays if the bytes will
/// not fetch or decode.
async fn light_default(r: &mut render::Renderer) {
    let Some(asset) = panels::lighting::environment_asset(DEFAULT_ENVIRONMENT) else {
        return;
    };
    let Ok(bytes) = dioxus::asset_resolver::read_asset_bytes(asset).await else {
        return;
    };
    match r
        .desk
        .engine_mut()
        .register_environment(DEFAULT_ENVIRONMENT, bytes)
    {
        Ok(()) => r.process(ViewCommand::SetEnvironment(DEFAULT_ENVIRONMENT)),
        Err(e) => tracing::warn!("the bundled environment will not decode: {e}"),
    }
}
