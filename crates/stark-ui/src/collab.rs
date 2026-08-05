//! Shared-drawing UI glue (§12): hosting and joining sessions, plus
//! the two pumps between the engine and the network —
//!
//! - **outgoing**: after every dispatched command, [`flush_outbox`] broadcasts
//!   what the engine committed;
//! - **incoming**: a spawned task feeds [`RemoteEvent`]s into the engine and
//!   repaints.
//!
//! Both ends of the invitation are one gesture. Sharing starts the moment
//! "Share…" is picked, and [`SessionModal`] only hands over the resulting link;
//! joining has no UI at all — opening a link whose fragment carries a ticket
//! joins on load.
//!
//! The session itself lives in a signal beside the renderer; iroh runs in the
//! browser over its relay transport, so this is the same code path native
//! tests exercise over UDP.
//!
//! Every task in this module is spawned with `spawn_forever`: `spawn` ties a
//! task to the calling component's scope, and these calls originate in modal
//! button handlers — closing the modal must not cancel session work. The
//! incoming pump outlives even its entry point, so its handle is kept in
//! `AppState`'s `collab.pump` and cancelled when the session it serves is
//! replaced ([`install`]) or torn down ([`leave`]).

use dioxus::dioxus_core::spawn_forever;
use dioxus::prelude::*;
use stark_core::peer::Identity;
use stark_net::{
    AssetNeed, Broadcaster, CollabSession, LinkKind, NetOptions, RemoteEvent, SessionTicket,
    actor_from_endpoint_id,
};

use crate::icons::{self, icon};
use crate::state::AppState;

/// How often this client publishes its presence, in ms. Fast enough that another
/// painter's stroke grows smoothly, slow enough that a 240 Hz pen does not put 240
/// frames a second on a flood mesh.
const PRESENCE_TICK_MS: i32 = 33;

/// Presence ticks between polls of how each peer is reached (WebRTC / relay /
/// hole-punched — the session dialog's badges). ~2 s: links change when a
/// connection is made, lost, or upgraded by hole punching, never per frame.
const LINK_POLL_TICKS: u32 = 60;

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
    if (state.collab.phase)() != CollabPhase::Solo {
        return;
    }
    set_phase(state, CollabPhase::Connecting);
    spawn_forever(async move {
        // The actor id derives from the endpoint identity, and the shared log
        // must carry it before the snapshot is served — so settle the key first,
        // convert the engine, then bind the session around it. The key is this
        // browser's persisted one, so sharing the same document twice is the same
        // author twice (`crate::identity`).
        let id = crate::identity::get();
        let actor = actor_from_endpoint_id(id.secret.public());
        let (doc, assets) = {
            let mut renderer = state.renderer;
            let mut guard = renderer.write();
            let Some(r) = guard.as_mut() else {
                set_phase(state, CollabPhase::Solo);
                return;
            };
            r.start_collaboration(Identity::new(actor, id.boot));
            (r.document_file(), r.all_asset_bytes())
        };

        let opts = NetOptions {
            secret: Some(id.secret),
            ..Default::default()
        };
        match CollabSession::host(doc, opts).await {
            Ok(session) => {
                // Seed every locally-imported brush so peers can fetch any the
                // snapshot didn't already bundle (§12.4). The grounds need no
                // equivalent: the snapshot carries every one the log names, and
                // hosting seeds the blob store from it — a ground imported *later*
                // is seeded by `grounds::select` before its `SetSurface` goes out.
                for (id, bytes) in assets {
                    session.add_content(AssetNeed::Brush(id), bytes);
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

/// Join the session a ticket names. Replaces the current document. The only
/// caller is the page-load path in `main.rs`, which reads the ticket out of the
/// URL fragment — a shared link is the whole of the joining UI.
pub fn join(state: AppState, ticket_text: String) {
    if (state.collab.phase)() != CollabPhase::Solo {
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
    spawn_forever(async move {
        // Same persisted key as hosting uses: joining is not a different person.
        let id = crate::identity::get();
        let opts = NetOptions {
            secret: Some(id.secret),
            ..Default::default()
        };
        match CollabSession::join(&ticket, opts).await {
            Ok((session, file)) => {
                let assets = {
                    let mut renderer = state.renderer;
                    let mut obs = state.obs;
                    let mut guard = renderer.write();
                    let Some(r) = guard.as_mut() else {
                        set_phase(state, CollabPhase::Solo);
                        return;
                    };
                    r.join_collaboration(&file, Identity::new(session.actor_id(), id.boot));
                    r.paint();
                    obs.set(Some(r.observe()));
                    r.all_asset_bytes()
                };
                for (id, bytes) in assets {
                    session.add_content(AssetNeed::Brush(id), bytes);
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
    let mut session_sig = state.collab.session;
    let Some(session) = session_sig.write().take() else {
        return;
    };
    let mut pump = state.collab.pump;
    if let Some(task) = pump.write().take() {
        task.cancel();
    }
    let mut presence = state.collab.presence;
    if let Some(task) = presence.write().take() {
        task.cancel();
    }
    let mut peers = state.collab.peers;
    peers.set(Vec::new());
    let mut links = state.collab.links;
    links.set(Vec::new());
    // Say goodbye before the transport goes: peers drop this client at once rather
    // than waiting out the presence timeout with a stale cursor on their canvas.
    let farewell = {
        let mut renderer = state.renderer;
        let mut guard = renderer.write();
        guard.as_mut().map(|r| {
            let frame = r.leaving_presence();
            r.end_collaboration();
            r.paint();
            frame
        })
    };
    let mut ticket = state.collab.ticket;
    ticket.set(None);
    set_url_ticket(None);
    set_phase(state, CollabPhase::Solo);
    spawn_forever(async move {
        if let Some(frame) = farewell {
            let _ = session.broadcaster().publish(frame).await;
        }
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
    let Some(tx): Option<Broadcaster> = state
        .collab
        .session
        .read()
        .as_ref()
        .map(|s| s.broadcaster())
    else {
        return;
    };
    spawn_forever(async move {
        for action in actions {
            if let Err(e) = tx.broadcast(action).await {
                tracing::warn!("broadcast failed: {e}");
            }
        }
    });
}

/// Store the live session and start the incoming pump.
fn install(state: AppState, mut session: CollabSession) {
    let ticket_text = session.ticket().to_string();
    // The page URL *is* the invitation: anyone opening it joins this session
    // (via this peer — every member is a valid entry point).
    set_url_ticket(Some(&ticket_text));
    let mut ticket = state.collab.ticket;
    ticket.set(Some(ticket_text));

    let mut events = session.take_events().expect("fresh session events");
    let mut session_sig = state.collab.session;
    session_sig.set(Some(session));
    set_phase(state, CollabPhase::Shared);

    let task = spawn_forever(async move {
        let mut renderer = state.renderer;
        let mut obs = state.obs;
        while let Some(event) = events.recv().await {
            let (snapshot, repaint) = {
                let mut guard = renderer.write();
                let Some(r) = guard.as_mut() else { continue };
                match event {
                    // Repaint: an asset resolved off a *presence* head arrives
                    // while the peer's live stroke is already on screen as a
                    // round-tip fallback — the import is what upgrades it. (On
                    // the commit path the following Action repaints anyway;
                    // assets are rare enough that one extra request is free.)
                    RemoteEvent::Asset { need, bytes } => {
                        // The transport says which store these bytes belong in —
                        // the action that referenced them is the only thing that
                        // knows, and a brush mask and a canvas ground decode
                        // differently (§6.6, §6.4).
                        match need {
                            AssetNeed::Brush(_) => r.import_brush(&bytes),
                            // Arrives *before* the `SetSurface` that wanted it, so
                            // the tooth reads the real weave from the very first
                            // stroke after the switch rather than baking a flat
                            // deposit that no later arrival un-bakes.
                            AssetNeed::Ground(id) => r.accept_surface(id, &bytes),
                        }
                        (None, true)
                    }
                    RemoteEvent::Action(action) => {
                        r.merge_remote(action);
                        (Some(r.observe()), true)
                    }
                    // A peer moved, switched layer, or drew another stretch of a
                    // live stroke (§17.4). Repaint only when the frame
                    // reached the *canvas*, which `merge_presence` is what decides:
                    // a cursor and a name are DOM chrome drawn from the roster, which
                    // the presence pump pushes on its own cadence, so a remote
                    // pointer move owes no compositor pass at all. And leave `obs`
                    // alone regardless: presence changes nothing the chrome renders
                    // from, and refreshing it at pointer rate would re-run the whole
                    // component tree.
                    RemoteEvent::Presence { actor, frame } => {
                        // Dated here, with the same clock the presence pump ticks
                        // the expiry with. The engine's own clock is no substitute:
                        // it advances only when that pump has something to drain,
                        // which on a client that is just watching is the heartbeat
                        // — and a frame stamped a whole heartbeat stale is what
                        // used to trip `GESTURE_TIMEOUT` mid-stroke.
                        (None, r.merge_presence(actor, frame, now_seconds()))
                    }
                }
            };
            // Requested, not painted inline: peer gesture frames arrive at ~30 Hz
            // *per stroking peer*, on top of the local pointer rate — the request
            // latch is what folds all of it into one paint per displayed frame.
            if repaint {
                crate::state::request_paint(state);
            }
            if snapshot.is_some() {
                obs.set(snapshot);
            }
        }
        tracing::info!("collab event stream ended");
    });
    let mut pump = state.collab.pump;
    if let Some(old) = pump.write().replace(task) {
        old.cancel();
    }
    start_presence_pump(state);
}

/// Publish this client's presence on a fixed cadence for as long as the session
/// lives (§17.5).
///
/// A *pull* loop rather than a push from `dispatch`: presence is a latch, so what
/// matters is its current value, and pulling on a fixed tick is what turns a 240 Hz
/// pen into a 30 Hz stream without dropping anything (the path delta is computed
/// against what was actually sent). It also gives the engine the clock it has none of
/// — the same tick expires peers who have gone quiet.
///
/// **Waking is not working.** The loop wakes on a fixed cadence but does nothing at
/// all on a tick where nothing has moved: `presence_due` and the roster revision are
/// both `&self`, read through `peek` — so an idle session costs two comparisons, with
/// no mutable borrow of the engine, no roster allocation, and no signal write. The
/// last of those matters most: `Signal::write` marks its subscribers dirty whether or
/// not the value changed, so taking one every tick would re-render every component
/// that reads the renderer, thirty times a second, for the entire life of a session
/// in which nobody was doing anything.
fn start_presence_pump(state: AppState) {
    let mut presence = state.collab.presence;
    let task = spawn_forever(async move {
        let mut sent_revision = 0;
        let mut ticks: u32 = 0;
        loop {
            // The tick runs *first* and the sleep last, so the engine's clock is set
            // before the incoming pump can hand it a frame to date. A labelled block
            // rather than `continue`, so that stays true however the body grows —
            // `continue` here would skip the sleep and spin.
            'tick: {
                let Some(tx): Option<Broadcaster> = state
                    .collab
                    .session
                    .read()
                    .as_ref()
                    .map(|s| s.broadcaster())
                else {
                    // The session is gone; so is the reason for this loop.
                    return;
                };
                // Refresh how each peer is reached, on its own slow cadence and
                // before the idle check — a link upgrading from relay to WebRTC
                // moves no cursor. On the first tick too, so the dialog isn't
                // blank for the poll interval after sharing starts. Write only
                // on change: the signal re-renders the session dialog.
                if ticks.is_multiple_of(LINK_POLL_TICKS) {
                    let links = tx.links().await;
                    let mut links_sig = state.collab.links;
                    if *links_sig.peek() != links {
                        links_sig.set(links);
                    }
                }
                ticks = ticks.wrapping_add(1);
                let now = now_seconds();
                // `peek`, not `read`: this runs outside any component, and subscribing
                // a background task to a signal is meaningless anyway.
                let work = state
                    .renderer
                    .peek()
                    .as_ref()
                    .map(|r| (r.presence_due(now), r.peers_revision() != sent_revision));
                let Some((due, roster_stale)) = work else {
                    break 'tick;
                };
                if !due && !roster_stale {
                    break 'tick;
                }

                let (frame, repaint, roster) = {
                    let mut renderer = state.renderer;
                    let mut guard = renderer.write();
                    match guard.as_mut() {
                        Some(r) => {
                            let tick = due.then(|| r.take_presence(now));
                            let (frame, repaint) = match tick {
                                Some(t) => (t.frame, t.repaint),
                                None => (None, false),
                            };
                            // Re-read the revision *after* the drain, which may itself
                            // have expired a peer — and compare against what was last
                            // handed to the signal, so a change made here is not
                            // skipped by the watermark advancing past it.
                            let revision = r.peers_revision();
                            let stale = revision != sent_revision;
                            sent_revision = revision;
                            (frame, repaint, stale.then(|| r.peers()))
                        }
                        None => break 'tick,
                    }
                };
                // The drain's expiry may have taken a stalled gesture or a departed
                // peer's paint off the canvas. Nothing else notices: the incoming
                // pump repaints only for frames that *arrive*, and expiry is exactly
                // the case where they stopped — skip this and the dead stroke stays
                // on screen, frozen, until something unrelated forces a paint.
                if repaint {
                    crate::state::request_paint(state);
                }
                if let Some(roster) = roster {
                    let mut peers = state.collab.peers;
                    peers.set(roster);
                }
                if let Some(frame) = frame
                    && let Err(e) = tx.publish(frame).await
                {
                    // Best effort by design: the next frame supersedes this one, and
                    // nothing in the log depends on it.
                    tracing::debug!("presence publish failed: {e}");
                }
            }
            crate::platform::sleep_ms(PRESENCE_TICK_MS).await;
        }
    });
    if let Some(old) = presence.write().replace(task) {
        old.cancel();
    }
}

/// Seconds on a **monotonic** clock — the clock `stark-core` deliberately does not
/// own.
///
/// `performance.now()` rather than `Date.now()`, because every use of this is a
/// *duration*: `PEER_TIMEOUT`, `HEARTBEAT`, `GESTURE_TIMEOUT`, `GESTURE_RESYNC`.
/// Nothing compares it across clients, and nothing needs an epoch. A wall clock
/// stepping — an NTP correction, a user changing the system time — broke those
/// durations in both directions: backwards, `now - pub_at` went negative, the
/// heartbeat stopped coming due and every peer dropped this client after six seconds
/// of apparent silence; forwards, the whole roster expired in a single tick.
///
/// `performance.now()` is missing only in environments with no `performance` at all,
/// where `Date.now()` is the best available answer.
pub(crate) fn now_seconds() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map_or_else(js_sys::Date::now, |p| p.now())
        / 1000.0
}

fn set_phase(state: AppState, phase: CollabPhase) {
    let mut p = state.collab.phase;
    p.set(phase);
    if phase != CollabPhase::Solo {
        let mut err = state.collab.error;
        err.set(None);
    }
}

fn fail(state: AppState, message: String) {
    let mut err = state.collab.error;
    err.set(Some(message));
    set_phase(state, CollabPhase::Solo);
}

// --- tickets in the URL fragment ---
//
// A live session's ticket rides the page URL as `…#stark…`, so sharing a
// drawing is just sharing the address, and opening such a link joins on load.
// The fragment is the right vehicle: it never leaves the browser (not sent to
// the server), and tickets are base32 — no percent-encoding surprises.

/// The session ticket in the current page URL, if any.
pub fn url_ticket() -> Option<String> {
    let hash = web_sys::window()?.location().hash().ok()?;
    let ticket = hash.strip_prefix('#').unwrap_or(&hash);
    ticket.starts_with("stark").then(|| ticket.to_string())
}

/// The invitation to hand out: this page's address with `ticket` in the fragment.
///
/// Rebuilt from the ticket rather than read back out of `location.href`, so the
/// dialog re-renders when the ticket signal changes — `location` is not reactive,
/// and reading it during a render that beat `replaceState` would show the old URL.
fn invite_url(ticket: &str) -> String {
    let Some(location) = web_sys::window().map(|w| w.location()) else {
        return format!("#{ticket}");
    };
    format!(
        "{}{}{}#{ticket}",
        location.origin().unwrap_or_default(),
        location.pathname().unwrap_or_default(),
        location.search().unwrap_or_default()
    )
}

/// Put `text` on the system clipboard. Fire-and-forget: the returned promise is
/// dropped, and a browser that denies the permission just leaves the readonly
/// field on screen to select by hand.
fn copy_to_clipboard(text: &str) {
    if let Some(window) = web_sys::window() {
        let _ = window.navigator().clipboard().write_text(text);
    }
}

/// Reflect (or clear) the live session's ticket in the URL bar. Uses
/// `replaceState` so joining/leaving doesn't pollute tab history.
fn set_url_ticket(ticket: Option<&str>) {
    let Some(window) = web_sys::window() else {
        return;
    };
    let url = match ticket {
        Some(t) => format!("#{t}"),
        // Rebuild path + query without a fragment (an empty replaceState URL
        // would keep the current one, hash included).
        None => {
            let location = window.location();
            format!(
                "{}{}",
                location.pathname().unwrap_or_default(),
                location.search().unwrap_or_default()
            )
        }
    };
    if let Ok(history) = window.history()
        && let Err(e) = history.replace_state_with_url(&wasm_bindgen::JsValue::NULL, "", Some(&url))
    {
        tracing::warn!("failed to update URL fragment: {e:?}");
    }
}

/// The label and style for how a peer's connection reaches us, from the link
/// the mesh reports for it — or `None`, meaning no direct connection at all:
/// the mesh forwards that peer's traffic through the members that do have one.
fn link_badge(kind: Option<LinkKind>) -> (&'static str, &'static str) {
    match kind {
        Some(LinkKind::WebRtc) => ("direct · WebRTC", "peer-link peer-link-direct"),
        Some(LinkKind::Direct) => ("direct", "peer-link peer-link-direct"),
        Some(LinkKind::Relay) => ("via relay", "peer-link peer-link-relay"),
        Some(LinkKind::Unknown) => ("connecting…", "peer-link"),
        None => ("via peers", "peer-link"),
    }
}

/// The "Share" dialog. Sharing has already started by the time this opens (the
/// menu item calls [`share`]), so the dialog's whole job is to hand over the
/// link — and to let this client leave again. There is no join half: opening a
/// shared link *is* joining (see `url_ticket`), so nothing here asks for a
/// ticket.
#[component]
pub fn SessionModal(on_close: EventHandler<()>) -> Element {
    let state = use_context::<AppState>();
    let phase = (state.collab.phase)();
    let ticket = (state.collab.ticket)();
    let error = (state.collab.error)();
    let mut copied = use_signal(|| false);

    rsx! {
        div {
            class: "modal-backdrop",
            onclick: move |_| on_close.call(()),
            div {
                class: "modal-dialog",
                onclick: move |e| e.stop_propagation(),

                div { class: "modal-title", "Share" }

                if let Some(message) = error {
                    div { class: "collab-error", {message} }
                }

                match phase {
                    // Only reached when sharing failed — the menu starts it before
                    // this dialog mounts, so there is nothing to wait for otherwise.
                    CollabPhase::Solo => rsx! {
                        div { class: "modal-subtitle",
                            "This canvas isn't shared."
                        }
                        button {
                            class: "btn btn-primary",
                            onclick: move |_| share(state),
                            "Try again"
                        }
                    },
                    CollabPhase::Connecting => rsx! {
                        div { class: "modal-subtitle", "Creating a link…" }
                    },
                    CollabPhase::Shared => rsx! {
                        div { class: "modal-subtitle",
                            "Anyone who opens this link paints here with you, in real time. Every member can pass it on."
                        }
                        {
                            let url = invite_url(ticket.as_deref().unwrap_or_default());
                            let to_copy = url.clone();
                            rsx! {
                                div { class: "invite-row",
                                    input {
                                        class: "invite-url",
                                        readonly: true,
                                        value: "{url}",
                                    }
                                    // The glyph is the half of this button that holds
                                    // still: the word swaps to "Copied" and back on a
                                    // timer, so what says *what the button is* has to
                                    // be the part that does not change.
                                    button {
                                        class: "btn btn-primary",
                                        onclick: move |_| {
                                            copy_to_clipboard(&to_copy);
                                            copied.set(true);
                                            // Back to "Copy" after a beat, so the
                                            // button reads as a button again.
                                            spawn(async move {
                                                crate::platform::sleep_ms(1600).await;
                                                copied.set(false);
                                            });
                                        },
                                        {icon(icons::COPY_TO_CLIPBOARD)}
                                        if copied() { "Copied" } else { "Copy" }
                                    }
                                }
                            }
                        }
                        // Who is here, and how each one is reached — the only
                        // place the relay-or-direct question is answered. Names
                        // come from the presence roster; link kinds from the
                        // mesh, polled by the presence pump.
                        {
                            let peers = (state.collab.peers)();
                            let links = (state.collab.links)();
                            rsx! {
                                div { class: "peer-list",
                                    if peers.is_empty() {
                                        div { class: "peer-list-empty",
                                            "No one else has joined yet."
                                        }
                                    }
                                    for peer in peers {
                                        {
                                            let kind = links
                                                .iter()
                                                .find(|l| l.actor == peer.actor)
                                                .map(|l| l.kind);
                                            let (label, class) = link_badge(kind);
                                            let color = peer.css_color();
                                            rsx! {
                                                div { class: "peer-row",
                                                    span {
                                                        class: "peer-dot",
                                                        style: "background: {color}",
                                                    }
                                                    span { class: "peer-name", {peer.name.clone()} }
                                                    span { class: "{class}", {label} }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        button {
                            class: "btn btn-secondary",
                            onclick: move |_| { leave(state); },
                            "Stop sharing"
                        }
                    },
                }

                div { class: "modal-actions",
                    button {
                        class: "btn btn-primary",
                        onclick: move |_| on_close.call(()),
                        {icon(icons::DONE)}
                        "Done"
                    }
                }
            }
        }
    }
}
