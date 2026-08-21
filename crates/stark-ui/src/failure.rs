//! What the app does when the GPU dies (§5).
//!
//! `stark-engine` projects [`ObservableState::gpu_failure`] rather than panicking,
//! and states the contract this module is the other half of: *stop dispatching and
//! offer to save; do not keep painting, since nothing after this point reaches a
//! pixel.* The field is an `Arc` so the projection stays cheap to clone at pointer
//! rate — it was put there for a frontend to read.
//!
//! **The painting survives the device, and that is the whole point.** A document is
//! an action log in ordinary memory (§8); the tiles are a derived, cached view of
//! it. So a lost device costs the *picture on screen* and nothing else — the file
//! this dialog writes is byte-identical to the one a healthy session would have
//! written. Discovering the same fact by freezing, which is what a canvas with no
//! report does, loses the session instead.
//!
//! Two halves, and they key off the same field so they cannot disagree
//! ([`crate::state::gpu_lost`]): the doors in `state` stop moving the engine, and
//! this mounts. A stop with no dialog is a frozen canvas; a dialog over an app that
//! is still dispatching is a lie.
//!
//! [`ObservableState::gpu_failure`]: stark_engine::ObservableState::gpu_failure

use dioxus::prelude::*;

use crate::icons::{self, icon};
use crate::state::AppState;
use crate::widgets::Modal;

/// The report, mounted for as long as the projection carries a failure — which is
/// forever, since a failure is never cleared and the device is never rebuilt.
///
/// **Not dismissible, and it has no ✕.** Every other dialog in the app closes back
/// onto something you can carry on doing; there is nothing behind this one. A close
/// button would offer a return to a canvas that cannot take another stroke, which
/// is exactly the state this exists to stop the artist from spending time in.
///
/// Mounted at the app root rather than inside any panel, on
/// [`crate::brush_editor`]'s reason: a panel is a `backdrop-filter` containing
/// block, so a dialog rendered inside one is trapped in its box.
#[component]
pub fn GpuFailureModal() -> Element {
    let state = use_context::<AppState>();
    // Cloned out of the projection rather than held across the render: `Arc`, so
    // this is a refcount bump, and the guard must not be live when Save reads the
    // renderer.
    let Some(failure) = state
        .obs
        .read()
        .as_ref()
        .and_then(|o| o.gpu_failure.clone())
    else {
        return rsx! {};
    };
    // Saved once. The button stays — a download the browser silently declined is
    // the one case where asking again is the right move — but the word changes, so
    // a click that appeared to do nothing is distinguishable from one that worked.
    let mut saved = use_signal(|| false);

    rsx! {
        // No `on_close`, unlike every other dialog in the app: there is nothing
        // behind this one to go back to.
        Modal {
            div { class: "modal-title", "The GPU stopped responding" }
            div { class: "modal-subtitle",
                "The canvas can't be drawn any more. Your painting is safe \u{2014} \
                 Stark stores it as the list of actions that made it, not as pixels, \
                 so saving now loses nothing. Reload the page to start again."
            }

            // What the driver said. Small and dim: nothing branches on it and
            // most artists will not read it, but it is the only thing that makes
            // a bug report about this actionable.
            div { class: "failure-detail", "{failure}" }

            div { class: "modal-actions",
                button {
                    class: "btn btn-primary",
                    onclick: move |_| {
                        crate::files::save_document(state);
                        saved.set(true);
                    },
                    {icon(icons::SAVE)}
                    if saved() { "Save again" } else { "Save the painting" }
                }
            }
        }
    }
}
