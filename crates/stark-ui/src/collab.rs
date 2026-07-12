//! Shared-drawing UI glue (DESIGN.md §12): hosting and joining sessions, plus
//! the two pumps between the engine and the network —
//!
//! - **outgoing**: after every dispatched command, [`flush_outbox`] broadcasts
//!   what the engine committed;
//! - **incoming**: a spawned task feeds [`RemoteEvent`]s into the engine and
//!   repaints.
//!
//! The session itself lives in a signal beside the renderer; iroh runs in the
//! browser over its relay transport, so this is the same code path native
//! tests exercise over UDP.

use dioxus::prelude::*;
use stark_net::{
    actor_from_endpoint_id, Broadcaster, CollabSession, NetOptions, RemoteEvent, SecretKey,
    SessionTicket,
};

use crate::AppState;

/// The UI's view of the collaboration state.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum CollabPhase {
    #[default]
    Solo,
    /// Session setup (bind/online/join) in flight.
    Connecting,
    /// Live in a shared session.
    Shared,
}

/// Start hosting the current document. Async: binds the endpoint, waits
/// (bounded) for relay readiness so the ticket is dialable, then flips the
/// engine into shared mode and stores the ticket for the dialog.
pub fn share(state: AppState) {
    if (state.collab_phase)() != CollabPhase::Solo {
        return;
    }
    set_phase(state, CollabPhase::Connecting);
    spawn(async move {
        // The actor id derives from the endpoint identity, and the shared log
        // must carry it before the snapshot is served — so generate the key
        // first, convert the engine, then bind the session around it.
        let secret = SecretKey::generate();
        let actor = actor_from_endpoint_id(secret.public());
        let (doc, assets) = {
            let mut renderer = state.renderer;
            let mut guard = renderer.write();
            let Some(r) = guard.as_mut() else {
                set_phase(state, CollabPhase::Solo);
                return;
            };
            r.start_collaboration(actor);
            (r.document_file(), r.all_asset_bytes())
        };

        let opts = NetOptions { secret: Some(secret), ..Default::default() };
        match CollabSession::host(doc, opts).await {
            Ok(session) => {
                // Seed every locally-imported brush so peers can fetch any the
                // snapshot didn't already bundle (DESIGN.md §12.4).
                for (id, bytes) in assets {
                    session.add_asset(id, bytes);
                }
                install(state, session);
            }
            Err(e) => {
                tracing::warn!("share failed: {e}");
                fail(state, format!("Sharing failed: {e}"));
            }
        }
    });
}

/// Join a session from a pasted ticket. Replaces the current document.
pub fn join(state: AppState, ticket_text: String) {
    if (state.collab_phase)() != CollabPhase::Solo {
        return;
    }
    let ticket: SessionTicket = match ticket_text.parse() {
        Ok(t) => t,
        Err(e) => {
            fail(state, format!("Bad ticket: {e}"));
            return;
        }
    };
    set_phase(state, CollabPhase::Connecting);
    spawn(async move {
        match CollabSession::join(&ticket, NetOptions::default()).await {
            Ok((session, file)) => {
                let assets = {
                    let mut renderer = state.renderer;
                    let mut obs = state.obs;
                    let mut guard = renderer.write();
                    let Some(r) = guard.as_mut() else {
                        set_phase(state, CollabPhase::Solo);
                        return;
                    };
                    r.join_collaboration(&file, session.actor_id());
                    r.paint();
                    obs.set(Some(r.observe()));
                    r.all_asset_bytes()
                };
                for (id, bytes) in assets {
                    session.add_asset(id, bytes);
                }
                install(state, session);
            }
            Err(e) => {
                tracing::warn!("join failed: {e}");
                fail(state, format!("Joining failed: {e}"));
            }
        }
    });
}

/// Leave the session: tear down the network side and keep painting solo on the
/// current canvas (the shared log stays loaded; the engine just stops queueing
/// broadcasts).
pub fn leave(state: AppState) {
    let mut session_sig = state.collab_session;
    let Some(session) = session_sig.write().take() else {
        return;
    };
    {
        let mut renderer = state.renderer;
        if let Some(r) = renderer.write().as_mut() {
            r.end_collaboration();
        }
    }
    let mut ticket = state.collab_ticket;
    ticket.set(None);
    set_phase(state, CollabPhase::Solo);
    spawn(async move {
        session.shutdown().await;
    });
}

/// After a dispatched command: broadcast whatever the engine just committed.
/// Cheap when solo (the outbox is empty and no session exists).
pub fn flush_outbox(state: AppState) {
    let actions = {
        let mut renderer = state.renderer;
        let mut guard = renderer.write();
        match guard.as_mut() {
            Some(r) => r.take_outbox(),
            None => return,
        }
    };
    if actions.is_empty() {
        return;
    }
    let Some(tx): Option<Broadcaster> =
        state.collab_session.read().as_ref().map(|s| s.broadcaster())
    else {
        return;
    };
    spawn(async move {
        for action in actions {
            if let Err(e) = tx.broadcast(action).await {
                tracing::warn!("broadcast failed: {e}");
            }
        }
    });
}

/// Store the live session and start the incoming pump.
fn install(state: AppState, mut session: CollabSession) {
    let mut ticket = state.collab_ticket;
    ticket.set(Some(session.ticket().to_string()));

    let mut events = session.take_events().expect("fresh session events");
    let mut session_sig = state.collab_session;
    session_sig.set(Some(session));
    set_phase(state, CollabPhase::Shared);

    spawn(async move {
        let mut renderer = state.renderer;
        let mut obs = state.obs;
        while let Some(event) = events.recv().await {
            let snapshot = {
                let mut guard = renderer.write();
                let Some(r) = guard.as_mut() else { continue };
                match event {
                    RemoteEvent::Asset { bytes } => {
                        r.import_brush(&bytes);
                        None
                    }
                    RemoteEvent::Action(action) => {
                        r.merge_remote(action);
                        r.paint();
                        Some(r.observe())
                    }
                }
            };
            if snapshot.is_some() {
                obs.set(snapshot);
            }
        }
        tracing::info!("collab event stream ended");
    });
}

fn set_phase(state: AppState, phase: CollabPhase) {
    let mut p = state.collab_phase;
    p.set(phase);
    if phase != CollabPhase::Solo {
        let mut err = state.collab_error;
        err.set(None);
    }
}

fn fail(state: AppState, message: String) {
    let mut err = state.collab_error;
    err.set(Some(message));
    set_phase(state, CollabPhase::Solo);
}

/// The "Shared drawing" dialog: start sharing, join from a ticket, or (while
/// live) read out the ticket and leave.
#[component]
pub fn SessionModal(on_close: EventHandler<()>) -> Element {
    let state = use_context::<AppState>();
    let phase = (state.collab_phase)();
    let ticket = (state.collab_ticket)();
    let error = (state.collab_error)();
    let mut join_text = use_signal(String::new);

    rsx! {
        div {
            class: "modal-backdrop",
            onclick: move |_| on_close.call(()),
            div {
                class: "modal-dialog",
                onclick: move |e| e.stop_propagation(),

                div { class: "modal-title", "Shared Drawing" }

                if let Some(message) = error {
                    div { class: "collab-error", "{message}" }
                }

                match phase {
                    CollabPhase::Solo => rsx! {
                        div { class: "modal-subtitle",
                            "Paint together in real time, peer-to-peer. Share this canvas, or join someone else's."
                        }
                        div { class: "modal-section-label", "SHARE THIS CANVAS" }
                        button {
                            class: "btn btn-primary",
                            onclick: move |_| share(state),
                            "Start sharing"
                        }
                        div { class: "modal-section-label", "JOIN A SESSION" }
                        div { class: "modal-subtitle", "Joining replaces your current canvas with the shared one." }
                        input {
                            class: "ticket-input",
                            placeholder: "Paste a ticket (stark…)",
                            value: "{join_text}",
                            oninput: move |e| join_text.set(e.value()),
                        }
                        div { class: "modal-actions",
                            button {
                                class: "btn btn-secondary",
                                onclick: move |_| on_close.call(()),
                                "Close"
                            }
                            button {
                                class: "btn btn-primary",
                                disabled: join_text().trim().is_empty(),
                                onclick: move |_| join(state, join_text()),
                                "Join"
                            }
                        }
                    },
                    CollabPhase::Connecting => rsx! {
                        div { class: "modal-subtitle", "Connecting…" }
                        div { class: "modal-actions",
                            button {
                                class: "btn btn-secondary",
                                onclick: move |_| on_close.call(()),
                                "Close"
                            }
                        }
                    },
                    CollabPhase::Shared => rsx! {
                        div { class: "modal-subtitle",
                            "Live. Send this ticket to anyone who should paint with you — every member can share it."
                        }
                        textarea {
                            class: "ticket-text",
                            readonly: true,
                            value: ticket.unwrap_or_default(),
                        }
                        div { class: "modal-actions",
                            button {
                                class: "btn btn-secondary",
                                onclick: move |_| { leave(state); },
                                "Leave session"
                            }
                            button {
                                class: "btn btn-primary",
                                onclick: move |_| on_close.call(()),
                                "Close"
                            }
                        }
                    },
                }
            }
        }
    }
}
