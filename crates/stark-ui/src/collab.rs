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
use stark_engine::command::ViewCommand;
use stark_engine::peer::Identity;
use stark_model::SubstrateId;
use stark_net::{
    AssetNeed, Broadcaster, CollabSession, Events, Joined, LinkKind, NetOptions, RemoteEvent,
    SessionTicket, actor_from_endpoint_id,
};

use crate::icons::{self, icon};
use crate::state::AppState;
use crate::widgets::Modal;

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
        // Quiet: hosting attaches an identity and starts queueing broadcasts, which
        // no part of the projection shows — the roster is its own signal (§17.4).
        let Some((doc, assets)) = crate::state::with_engine_quiet(state, |r| {
            r.start_collaboration(Identity::new(actor, id.boot));
            (r.document_file(), r.all_asset_bytes())
        }) else {
            set_phase(state, CollabPhase::Solo);
            return;
        };

        let opts = NetOptions {
            secret: Some(id.secret),
            resolvable: crate::builtin_ids::resolvable(),
            ..Default::default()
        };
        match CollabSession::host(doc, opts).await {
            Ok((session, events)) => {
                // Seed every locally-imported brush so peers can fetch any the
                // snapshot didn't already bundle (§12.4). The substrates need no
                // equivalent: the snapshot carries every one the log names, and
                // hosting seeds the blob store from it — a substrate imported *later*
                // is seeded by `substrates::select` before its `SetSubstrate` goes out.
                for (id, bytes) in assets {
                    session.add_content(AssetNeed::Brush(id), bytes);
                }
                let ticket_text = session.ticket().await.to_string();
                install(state, session, events, ticket_text);
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
        // What this build ships with, offered so the host can leave it out of the
        // snapshot — the bundled substrates canonicalize to 2.0 and 2.8 MB of substrate
        // every install already has (§12.4). A promise, settled below before anything is
        // replayed, and called in again by `ResolveLocally` for the rest of the
        // session.
        let opts = NetOptions {
            secret: Some(id.secret),
            resolvable: crate::builtin_ids::resolvable(),
            ..Default::default()
        };
        match CollabSession::join(&ticket, opts).await {
            Ok(Joined {
                session,
                events,
                document: file,
                owed,
            }) => {
                // Fetched *before* `join_collaboration`, which replays the log: a
                // substrate that is not registered when its `SetSubstrate` replays
                // deposits every later stroke against the flat stand-in, and
                // those pixels are stored (§6.4). Awaited out here because the
                // renderer guard must not be held across a fetch.
                let owed_bytes = crate::builtin_ids::fetch(&owed).await;
                // Joining replaces the whole document, so the publish is the point:
                // `with_engine` takes it on the way out, and the inline paint stays
                // for the reason `state::resize` keeps one — the peer's canvas
                // should not wait a frame to appear.
                let Some(assets) = crate::state::with_engine(state, |r| {
                    for (need, bytes) in &owed_bytes {
                        crate::builtin_ids::install(r, *need, bytes);
                    }
                    r.join_collaboration(&file, Identity::new(session.actor_id(), id.boot));
                    // Frame what arrived, the same as opening a file does
                    // (`files::open_bytes`): a view is per-client and never sent
                    // (§18.1.2), so a joiner starts at the origin at 1:1 while the
                    // drawing they came to see can be anywhere on an unbounded
                    // canvas — including entirely off their screen.
                    let frame = crate::panels::frame::piece_frame(&r.observe());
                    r.process(ViewCommand::ShowPiece(frame));
                    r.paint();
                    r.all_asset_bytes()
                }) else {
                    set_phase(state, CollabPhase::Solo);
                    return;
                };
                for (id, bytes) in assets {
                    session.add_content(AssetNeed::Brush(id), bytes);
                }
                let ticket_text = session.ticket().await.to_string();
                install(state, session, events, ticket_text);
            }
            Err(e) => {
                tracing::warn!("join failed: {e}");
                fail(state, format!("Joining failed: {e}"));
            }
        }
    });
}

/// Make good on [`RemoteEvent::ResolveLocally`]: read the content out of this
/// app's own bundle, install it, and register it with the session — which is what
/// releases the remote action that was waiting on it.
///
/// Doing nothing here would also be correct: the transport dials a peer after a
/// grace period, which is what it did before any of this existed. What this saves
/// is the transfer — a collaborator switching to a substrate the app ships with costs
/// a read from its own files instead of megabytes off a peer (§12.4).
fn supply_locally(state: AppState, need: AssetNeed) {
    let Some(broadcaster): Option<Broadcaster> = state
        .collab
        .session
        .read()
        .as_ref()
        .map(|s| s.broadcaster())
    else {
        return;
    };
    spawn_forever(async move {
        let Some((need, bytes)) = crate::builtin_ids::fetch(&[need]).await.into_iter().next()
        else {
            return;
        };
        // Into the engine *before* the session is told. `add_content` releases
        // the action parked on this content, and that action is applied on the
        // assumption its content is already installed — for a substrate, getting
        // that order wrong is the flat stand-in again (§6.4).
        //
        // Quiet: bytes arriving change how a later action *renders*, not anything
        // the chrome shows, so this asks for the frame and nothing else.
        if crate::state::with_engine_quiet(state, |r| crate::builtin_ids::install(r, need, &bytes))
            .is_none()
        {
            return;
        }
        broadcaster.add_content(need, bytes);
        crate::state::request_paint(state);
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
    // Quiet: the peers' paint coming off the canvas is a repaint, and the roster
    // emptying is `state.collab`'s business — the projection says nothing about
    // either.
    let farewell = crate::state::with_engine_quiet(state, |r| {
        let frame = r.leaving_presence();
        r.end_collaboration();
        r.paint();
        frame
    });
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
    // Quiet, and on the interactive path: this runs after *every* dispatch, which
    // has just published — a second `observe` walk here would be paid per command
    // to report exactly what the first one did.
    let Some(actions) = crate::state::with_engine_quiet(state, |r| r.take_outbox()) else {
        return;
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
    // Inline, not spawned. Broadcasting queues; the session's one send task puts
    // things on the wire. A task per dispatch used to mean two dispatches in the
    // same frame raced onto the same sender, and every inversion bought a timeline
    // resync on every receiver.
    for action in actions {
        if let Err(e) = tx.broadcast(action) {
            tracing::warn!("broadcast failed: {e}");
        }
    }
}

/// Store the live session and start the incoming pump. `ticket_text` is minted
/// by the async caller — minting asks the network which members are reachable.
fn install(state: AppState, session: CollabSession, mut events: Events, ticket_text: String) {
    // The page URL *is* the invitation: anyone opening it joins this session
    // (via this peer — every member is a valid entry point).
    set_url_ticket(Some(&ticket_text));
    let mut ticket = state.collab.ticket;
    ticket.set(Some(ticket_text));

    let mut session_sig = state.collab.session;
    session_sig.set(Some(session));
    set_phase(state, CollabPhase::Shared);

    let task = spawn_forever(async move {
        while let Some(event) = events.recv().await {
            // Held quietly and published per *event* rather than per hold of the
            // engine: only a merged action moves the document the chrome renders
            // from, and presence arrives at pointer rate — publishing on that
            // cadence would drag a full component tree behind every peer's pointer.
            let Some((publish, repaint)) = crate::state::with_engine_quiet(state, |r| {
                match event {
                    // Repaint: an asset resolved off a *presence* head arrives
                    // while the peer's live stroke is already on screen as a
                    // round-tip fallback — the import is what upgrades it. (On
                    // the commit path the following Action repaints anyway;
                    // assets are rare enough that one extra request is free.)
                    RemoteEvent::Asset { need, bytes } => {
                        // The transport says which store these bytes belong in —
                        // the action that referenced them is the only thing that
                        // knows, and a brush mask and a canvas substrate decode
                        // differently (§6.6, §6.4).
                        match need {
                            AssetNeed::Brush(_) => r.import_brush(&bytes),
                            // Arrives *before* the `SetSubstrate` that wanted it, so
                            // the tooth reads the real substrate from the very first
                            // stroke after the switch rather than baking a flat
                            // deposit that no later arrival un-bakes.
                            AssetNeed::Substrate(id) => {
                                r.accept_substrate(SubstrateId::Image(id), &bytes)
                            }
                            // Likewise before the `PlaceImage` that wanted it: the
                            // transport parks that action until the pixels land,
                            // because a placement without them is not a degraded
                            // placement, it is an empty layer (§23).
                            AssetNeed::Picture(id) => r.accept_picture(id, &bytes),
                        }
                        (false, true)
                    }
                    RemoteEvent::Action(action) => {
                        r.merge_remote(action);
                        (true, true)
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
                        // — and a frame stamped a whole heartbeat stale trips
                        // `GESTURE_TIMEOUT` mid-stroke.
                        (false, r.merge_presence(actor, frame, now_seconds()))
                    }
                    // The promise `join` made, called in: a peer named content
                    // this build ships with. Handled off this task — the read is
                    // a fetch and the pump is holding the renderer guard.
                    RemoteEvent::ResolveLocally { need } => {
                        supply_locally(state, need);
                        (false, false)
                    }
                }
            }) else {
                continue;
            };
            // Requested, not painted inline: peer gesture frames arrive at ~30 Hz
            // *per stroking peer*, on top of the local pointer rate — the request
            // latch is what folds all of it into one paint per displayed frame.
            if repaint {
                crate::state::request_paint(state);
            }
            if publish {
                crate::state::publish_observation(state);
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
                    // The invitation re-mints on the same cadence: which members
                    // a fresh link should name changes exactly when the links do
                    // — someone arrived, someone left, a path was proven. Kept
                    // current so the URL a host copies an hour in still works
                    // after that host closes the tab. Write only on change: the
                    // ticket signal re-renders the dialog, and minting sorts its
                    // members so the same membership always spells the same text.
                    let ticket_text = tx.ticket().await.to_string();
                    let mut ticket_sig = state.collab.ticket;
                    if ticket_sig.peek().as_deref() != Some(ticket_text.as_str()) {
                        set_url_ticket(Some(&ticket_text));
                        ticket_sig.set(Some(ticket_text));
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

                // Quiet, and this one is load-bearing: the tick runs at the presence
                // cadence for as long as a session is open, and the roster it does
                // publish has its own signal (§17.4) — nothing here is in the
                // projection.
                let tick = crate::state::with_engine_quiet(state, |r| {
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
                    (
                        frame,
                        repaint,
                        revision,
                        (revision != sent_revision).then(|| r.peers()),
                    )
                });
                let Some((frame, repaint, revision, roster)) = tick else {
                    break 'tick;
                };
                sent_revision = revision;
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

/// Seconds on the monotonic clock `stark-engine` deliberately does not own — see
/// [`platform::now_seconds`](crate::platform::now_seconds), which holds the whole
/// argument for which clock and why.
pub(crate) fn now_seconds() -> f64 {
    crate::platform::now_seconds()
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
// the server), and a ticket's alphabet is base64**url** — no percent-encoding
// surprises. It is case-sensitive, though, which the fragment is careful with and
// nothing here may lowercase.

/// The session ticket in the current page URL, if any.
pub fn url_ticket() -> Option<String> {
    let fragment = crate::platform::url_fragment()?;
    fragment.starts_with("stark").then_some(fragment)
}

/// The invitation to hand out: this page's address with `ticket` in the fragment.
///
/// Rebuilt from the ticket rather than read back out of `location.href`, so the
/// dialog re-renders when the ticket signal changes — `location` is not reactive,
/// and reading it during a render that beat `replaceState` would show the old URL.
fn invite_url(ticket: &str) -> String {
    crate::platform::url_with_fragment(ticket)
}

/// Reflect (or clear) the live session's ticket in the URL bar. Uses
/// `replaceState` so joining/leaving doesn't pollute tab history.
fn set_url_ticket(ticket: Option<&str>) {
    crate::platform::set_url_fragment(ticket);
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
        Modal { on_close,
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
                                        crate::platform::copy_to_clipboard(&to_copy);
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
                                                span { class: "peer-name", {peer.name} }
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
