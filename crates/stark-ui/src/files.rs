//! Getting work in and out as files: save, open, and export (§15.6, §8).
//!
//! Two different things share this module because they are the same gesture from
//! the user's side — a file leaves or arrives — but they are not the same object:
//!
//! - **Save** writes the *document*: the action log, replayable, still editable,
//!   with its whole undo history intact when reopened. This is the native format
//!   and the thing that must never be lossy.
//! - **Export** writes a *picture*: one frame, flattened and lit, at a chosen
//!   resolution. It is an output, not a document, and nothing can be recovered
//!   from it.
//!
//! Naming them apart in the menu matters more than it looks: an artist who
//! "exports" thinking they saved has lost the painting.

use dioxus::dioxus_core::spawn_forever;
use dioxus::prelude::*;

use crate::icons::{self, icon};
use crate::panels::frame::{piece_frame, use_selected_frame};
use crate::platform::{download_bytes, pick_file};
use crate::state::AppState;
use stark_engine::command::ViewCommand;
use stark_engine::{Background, ExportScale, Rendered};
use stark_model::document::LayerId;

/// Extension for the native (replayable) document format.
const DOC_EXT: &str = "stark";

/// Write the document — the action log, not the pixels (§8).
///
/// Lean: content this build ships with is named but not carried, because the
/// bundle is what a `.stark` file weighs. A log is fitted paths; a canvas ground is
/// megabytes, so carrying one puts a doodle on the built-in gesso at 2.8 MB of which
/// almost none is the painting. Opening one resolves those ids out of the app's
/// own files (`crate::builtin_ids`), and the catalog is append-only so that keeps
/// working.
pub fn save_document(state: AppState) {
    let resolvable = crate::builtin_ids::resolvable();
    let bytes = state
        .renderer
        .read()
        .as_ref()
        .map(|r| r.save_bytes_resolvable(&resolvable));
    match bytes {
        Some(Ok(bytes)) => {
            if let Err(e) = download_bytes(
                &bytes,
                &format!("painting.{DOC_EXT}"),
                "application/octet-stream",
            ) {
                tracing::error!("save failed: {e}");
            }
        }
        Some(Err(e)) => tracing::error!("could not serialize the document: {e}"),
        None => {}
    }
}

/// Replace the document with one read from disk. Its full undo history comes back
/// with it, because the file *is* the history (§8).
pub fn open_document(state: AppState) {
    // Called straight from the menu handler: a file picker may only be opened
    // inside the user gesture that asked for it.
    //
    // The handler is `Fn`, not `FnMut` — it may be re-entered per selected file —
    // so the signals are copied out of the capture on each call rather than
    // mutated in place. `Signal` is `Copy`, which is what makes that free.
    pick_file(&format!(".{DOC_EXT}"), move |_name, bytes| {
        open_bytes(state, bytes)
    });
}

/// Take the app's second way in: the file the OS launched it with (§11).
///
/// Registered on the browser's `launchQueue`, which the manifest's `file_handlers`
/// entry is what fills — double-clicking a `.stark` in a file manager, once the
/// app is installed. Two things about *when* this may be called:
///
/// - **After the renderer exists.** A queued launch is delivered the moment a
///   consumer is set, and [`open_bytes`] needs an engine to load into — with none
///   it would return having silently dropped the file. So `main` binds this at the
///   end of its startup task rather than in the root's body.
/// - **Possibly more than once.** The manifest asks for `focus-existing`, so a
///   second launch reaches the *running* app instead of reloading it. That is the
///   least destructive of the choices on offer: a reload would throw the open
///   painting away before the new file had even been read, whereas this path
///   refuses a bad file with the current one still on screen.
pub fn bind_file_launch(state: AppState) {
    crate::platform::on_file_launch(move |name, bytes| {
        tracing::info!(name, bytes = bytes.len(), "opening a launched file");
        open_bytes(state, bytes);
    });
}

/// Replace the document with one decoded from `bytes` — the half of [`open_document`]
/// after the file is in hand, shared with [`bind_file_launch`].
fn open_bytes(state: AppState, bytes: Vec<u8>) {
    let file = match stark_engine::DocumentFile::from_bytes(&bytes) {
        Ok(file) => file,
        Err(e) => return tracing::error!("could not open that file: {e}"),
    };
    // What the file names but does not carry, less whatever this session has
    // already loaded. Empty for a document saved with everything bundled.
    let owed = {
        let renderer = state.renderer;
        let guard = renderer.read();
        let Some(r) = guard.as_ref() else { return };
        r.unresolved_content(&file)
    };
    spawn_forever(async move {
        // Resolved out of this build's own assets, and *before* the replay:
        // the file carries the grounds it was painted on or names ones this
        // app ships, and either way they have to be registered before a single
        // action runs, or every stroke made on one deposits through the flat
        // stand-in into stored pixels (§6.4, §8).
        let supplied = crate::builtin_ids::fetch(&owed).await;
        if supplied.len() != owed.len() {
            // Refused rather than opened wrong. A file has no peer to fall
            // back on, so content this build cannot produce is the end of it —
            // which is the whole reason the shipped catalog is append-only.
            tracing::error!(
                ?owed,
                "this painting uses content this version of Stark does not have"
            );
            return;
        }
        // A load replaces the document wholesale, so this is the loud door: the
        // publish on the way out is what re-describes the layer list, the undo
        // flags and the rest to the chrome.
        let loaded = crate::state::with_engine(state, |r| {
            for (need, content) in &supplied {
                crate::builtin_ids::install(r, *need, content);
            }
            // The engine checks the bill again rather than trusting that it was
            // settled, and refuses without touching the open document. Reaching this
            // arm means an install above failed — bytes that hash to something other
            // than the id they were fetched for — so the painting on screen is left
            // exactly as it was.
            if let Err(e) = r.load_document(&file) {
                tracing::error!("could not open that painting: {e}");
                return false;
            }
            // Frame what has just arrived. A view is per-client session state and
            // so is not in the file (§18.1.2) — without this the painting opens at
            // whatever pan and zoom the *last* one was left at, which on an
            // unbounded canvas is routinely an empty stretch nowhere near it.
            //
            // Asked of the loaded document's own projection: the frame that says
            // where the piece ends belongs to the document that just replaced the
            // one on screen. And asked *before* the paint below, so the first frame
            // drawn is already the framed one rather than a flash of the old view.
            let frame = piece_frame(&r.observe());
            r.process(ViewCommand::ShowPiece(frame));
            r.paint();
            true
        });
        if loaded == Some(true) {
            tracing::info!(bytes = bytes.len(), "document loaded");
        }
    });
}

/// Scale presets for the export dialog. A multiplier is the honest unit here: the
/// frame's canvas size is the piece, and everything else is a resampling of it.
const SCALES: [(&str, f32); 4] = [
    ("1\u{00D7}", 1.0),
    ("2\u{00D7}", 2.0),
    ("3\u{00D7}", 3.0),
    ("4\u{00D7}", 4.0),
];

/// The export dialog: pick a resolution and a ground, see the pixel size you will
/// get, and write the PNG.
#[component]
pub fn ExportModal(on_close: EventHandler<()>) -> Element {
    let state = use_context::<AppState>();
    let mut scale = use_signal(|| 1.0f32);
    let mut transparent = use_signal(|| false);
    let mut error = use_signal(|| None::<String>);
    // A full-resolution readback is not instant, and the button is the only place
    // to say so — without it a 4x export looks like a dead click.
    let mut busy = use_signal(|| false);

    // Export frames against a *matte layer*, which need not be the selected one —
    // the dialog is opened from a menu, and the user may be on a paint layer.
    // Prefer whatever is selected, else the piece's own frame, else nothing (which
    // falls back to the painted bounds, §15.6). Only mattes **with a rect**
    // count: a background (§15.5) frames nothing, so it must not make
    // the dialog claim a frame it is not using.
    let frame: Option<LayerId> = use_selected_frame(state)()
        .filter(|(_, m)| m.rect.is_some())
        .map(|(l, _)| l.id)
        .or_else(|| state.obs.read().as_ref().and_then(piece_frame));

    // What we are about to produce, reported by the engine rather than recomputed
    // here — so the number on screen cannot drift from the render.
    let plan = state
        .renderer
        .read()
        .as_ref()
        .map(|r| r.export_plan(frame, ExportScale::Factor(scale())));

    let (size_label, plan_error) = match &plan {
        Some(Ok(p)) => (
            format!("{} \u{00D7} {} px", p.size.width, p.size.height),
            None,
        ),
        Some(Err(e)) => ("\u{2014}".to_string(), Some(e.to_string())),
        None => ("\u{2014}".to_string(), None),
    };
    let blocked = plan_error.is_some();

    rsx! {
        div {
            class: "modal-backdrop",
            onclick: move |_| on_close.call(()),
            div {
                class: "modal-dialog",
                onclick: move |e| e.stop_propagation(),

                div { class: "modal-title", "Export image" }
                div { class: "modal-subtitle",
                    if frame.is_some() {
                        "Writes the frame as a PNG. This is a picture, not the painting \u{2014} use Save to keep an editable document."
                    } else {
                        "No frame, so everything painted is exported. Add a frame to choose the crop."
                    }
                }

                div { class: "modal-section-label", "RESOLUTION" }
                select {
                    class: "select",
                    onchange: move |e| {
                        if let Some((_, factor)) = SCALES.iter().find(|(l, _)| *l == e.value()) {
                            scale.set(*factor);
                        }
                    },
                    for (label, factor) in SCALES {
                        option {
                            value: "{label}",
                            selected: (scale() - factor).abs() < 1e-3,
                            "{label}"
                        }
                    }
                }
                div { class: "export-size", "{size_label}" }

                div { class: "modal-section-label", "BACKGROUND" }
                div { class: "tool-row",
                    button {
                        class: if transparent() { "chip" } else { "chip active" },
                        onclick: move |_| transparent.set(false),
                        "Canvas"
                    }
                    button {
                        class: if transparent() { "chip active" } else { "chip" },
                        onclick: move |_| transparent.set(true),
                        "Transparent"
                    }
                }

                if let Some(message) = plan_error.clone().or_else(|| error.cloned()) {
                    div { class: "export-error", "{message}" }
                }

                div { class: "modal-actions",
                    button {
                        class: "btn btn-secondary",
                        onclick: move |_| on_close.call(()),
                        "Cancel"
                    }
                    button {
                        class: "btn btn-primary",
                        disabled: blocked || busy(),
                        onclick: move |_| {
                            busy.set(true);
                            // Scope-tied `spawn`, deliberately: this task writes
                            // `busy` and `error`, which belong to *this* modal's
                            // scope. A detached `spawn_forever` would outlive them
                            // and write through dangling handles. Dismissing the
                            // dialog therefore cancels the export, which is also
                            // what dismissing it should mean.
                            spawn(async move {
                                let result = export_png(state, frame, scale(), transparent()).await;
                                busy.set(false);
                                match result {
                                    Ok(()) => on_close.call(()),
                                    Err(e) => error.set(Some(e)),
                                }
                            });
                        },
                        // The same mark the menu entry that opened this dialog wears,
                        // and it stays put while the word swaps to "Exporting…".
                        {icon(icons::EXPORT)}
                        if busy() { "Exporting\u{2026}" } else { "Export" }
                    }
                }
            }
        }
    }
}

/// Render the frame and hand the PNG to the browser.
async fn export_png(
    state: AppState,
    frame: Option<LayerId>,
    scale: f32,
    transparent: bool,
) -> Result<(), String> {
    let background = if transparent {
        Background::Transparent
    } else {
        Background::Substrate
    };
    // Render, then **drop the guard before awaiting**. The readback future owns
    // everything it needs, so nothing holds the renderer while the browser's event
    // loop runs the copy — which it must be free to do, since the UI re-renders
    // during that time and would panic reading a renderer we still had borrowed.
    // `with_engine_quiet` is what bounds that hold to the render itself.
    let readback = crate::state::with_engine_quiet(state, |r| {
        // What the artist is looking at, in-flight gesture and all — a picture of
        // the canvas as it stands, not of the last commit.
        // No repaint afterwards: the export renders through its own view into its
        // own target, and a render at any size but the surface's builds its own
        // attachments (`Compositor::render`) — so the frame already on screen, and
        // everything it was composited from, is left exactly as it was.
        r.export(
            frame,
            ExportScale::Factor(scale),
            background,
            Rendered::Live,
        )
        .map_err(|e| e.to_string())
    })
    .ok_or("the canvas is not ready yet")??;
    let png = readback
        .await
        .map_err(|e| e.to_string())?
        .to_png()
        .map_err(|e| e.to_string())?;
    download_bytes(&png, "painting.png", "image/png")
}
