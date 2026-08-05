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

use dioxus::prelude::*;

use crate::icons::{self, icon};
use crate::panels::frame::selected_frame;
use crate::platform::{download_bytes, pick_file};
use crate::state::AppState;
use stark_core::{Background, ExportScale, LayerId, Rendered};

/// Extension for the native (replayable) document format.
const DOC_EXT: &str = "stark";

/// Write the document — the action log, not the pixels (§8).
pub fn save_document(state: AppState) {
    let bytes = state.renderer.read().as_ref().map(|r| r.save_bytes());
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
        let (mut renderer, mut obs) = (state.renderer, state.obs);
        let mut guard = renderer.write();
        let Some(r) = guard.as_mut() else { return };
        // One replay, and it is right the first time: the file carries the grounds
        // it was painted on (§6.4, §8), and `load_document` installs them before
        // replaying a single action. That is what this used to spend a second full
        // replay on — fetch the ground the document *ended* on, then run the whole
        // log again — and it never covered the case of a document that switched
        // grounds part-way, since it only ever fetched one.
        match r.load_bytes(&bytes) {
            Ok(()) => {
                r.paint();
                obs.set(Some(r.observe()));
                tracing::info!(bytes = bytes.len(), "document loaded");
            }
            Err(e) => tracing::error!("could not open that file: {e}"),
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
    // Prefer whatever is selected, else the topmost frame, else nothing (which
    // falls back to the painted bounds, §15.6).
    let frame: Option<LayerId> = selected_frame(state).map(|(l, _)| l.id).or_else(|| {
        state.obs.read().as_ref().and_then(|o| {
            o.layers
                .iter()
                .rev()
                .find(|l| l.matte.is_some())
                .map(|l| l.id)
        })
    });

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
    let mut renderer = state.renderer;

    // Render, then **drop the guard before awaiting**. The readback future owns
    // everything it needs, so nothing holds the renderer while the browser's event
    // loop runs the copy — which it must be free to do, since the UI re-renders
    // during that time and would panic reading a renderer we still had borrowed.
    let readback = {
        let mut guard = renderer.write();
        let r = guard.as_mut().ok_or("the canvas is not ready yet")?;
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
        .map_err(|e| e.to_string())?
    };
    let png = readback.await.to_png().map_err(|e| e.to_string())?;
    download_bytes(&png, "painting.png", "image/png")
}
