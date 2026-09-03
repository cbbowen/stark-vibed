//! What the app does when the GPU dies (§5) — or never arrives at all.
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
//! # The device that never arrived
//!
//! [`Signals::startup_failure`](crate::state::Signals::startup_failure)(crate::state::AppState) is the other way the
//! canvas ends up unable to take a mark, and it is not the same fact
//! ([`StartupFailure`](crate::render::StartupFailure)): there is no engine, so
//! there is no projection to carry it and no document to offer to save. It is
//! the **more likely** of the two by a wide margin — a browser without WebGPU is
//! Safari before 26, Firefox without the flag, a blocklisted driver, any
//! headless browser — and until it was given this surface it was three `expect`s
//! in `render::init`, which killed the startup task and left the chrome standing
//! over a blank canvas with the explanation in the console.
//!
//! One component for both, with two arms, because they are one thing from the
//! artist's side: the canvas will not take a mark, here is why, and here is
//! whatever can still be done about it. What differs is that arm's last
//! sentence, and whether there is a button under it.
//!
//! [`ObservableState::gpu_failure`]: stark_engine::ObservableState::gpu_failure

use dioxus::prelude::*;

use crate::icons::{self, icon};
use crate::state::{AppState, use_obs};
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
    // The app never started, which is asked first: it is the arm that means there
    // is no projection for the read below to find.
    if let Some(why) = (state.startup_failure)() {
        return rsx! { NoGpu { why } };
    }
    // Cloned out of the projection rather than held across the render: `Arc`, so
    // this is a refcount bump, and the guard must not be live when Save reads the
    // renderer.
    //
    // Through a memo, and this is the component in the app that most wants one:
    // it is mounted for the whole life of the page and draws nothing on a healthy
    // device, so a straight read of the projection — which is what this was — woke
    // it on every command of every session to decide, again, that there was
    // nothing to report. The `Option<Arc<_>>` compares by pointer, so the memo
    // propagates exactly once, when the device dies (`state::use_obs`).
    let Some(failure) = use_obs(state, |o| o.gpu_failure.clone())().flatten() else {
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

/// The report for a browser that never gave us a GPU
/// ([`StartupFailure`](crate::render::StartupFailure)).
///
/// **No Save**, unlike its sibling above, and that is the honest difference
/// between the two: a device that dies takes the picture on screen and leaves
/// the action log that made it, which is a file worth writing. A device that
/// never arrived leaves an empty document, and a button offering to save it
/// would hand the artist a file with nothing in it.
///
/// What it offers instead is the one thing that can help: which browsers can run
/// this. Named rather than described, because "a browser with WebGPU" is not
/// something anybody can go and check.
///
/// Not dismissible, for the sibling's reason and more strongly: there is not
/// merely nothing behind this dialog, there is nothing behind it *yet* —
/// dismissing it would leave a canvas that has never been able to take a mark.
#[component]
fn NoGpu(why: crate::render::StartupFailure) -> Element {
    rsx! {
        Modal {
            div { class: "modal-title", "This browser can\u{2019}t run Stark" }
            div { class: "modal-subtitle",
                "Stark paints on the GPU through WebGPU, and this browser either \
                 doesn\u{2019}t have it or isn\u{2019}t allowed to use it. Nothing \
                 has been lost \u{2014} the canvas never opened. Chrome, Edge and \
                 Safari 26 have WebGPU on by default; Firefox needs \
                 dom.webgpu.enabled set in about:config."
            }

            // What the browser said, in the same small dim line the lost-device
            // report uses. Nothing branches on it and most people will not read
            // it, but a bug report about this is unactionable without it.
            div { class: "failure-detail", "{why}" }
        }
    }
}
