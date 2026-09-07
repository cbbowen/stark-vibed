//! Sharing this window's painting, and joining someone else's (§12.4).
//!
//! The engine's half of a session is the same on both frontends — it is one
//! `ReplicatedTimeline` over the action log — and so is the wire. What is here is the
//! three things a native window answers differently from a browser tab.
//!
//! # The link names a client that exists
//!
//! A shared session travels as a ticket, and a ticket travels as a link. The web app
//! puts one in its own page URL, because the page *is* the client: sharing a drawing
//! is sharing the address. A window has no address, so the invitation it hands out
//! names the hosted web build ([`stark_ui::collab::WEB_APP`]) — anyone opening it gets
//! a client and joins through it, including someone with no copy of the app at all.
//!
//! Joining is the same fact read backwards, and it is the half that has no browser to
//! do it: nothing opens a window by URL, so **the clipboard is the door**. Share puts
//! a link on it, Join takes one off it, and the link is the same string either way —
//! which is why what is stripped off a paste is `stark_ui::collab`'s and not this
//! module's. There is no field to type into because there is no dialog to put one in
//! (§11.2); when there is, it reads the same function.
//!
//! # iroh is tokio's and wgpui is not
//!
//! The web app has one executor and everything is spawned on it. Here there are two:
//! wgpui's, which owns the `Canvas`, and tokio's, which is what iroh's timers and
//! sockets are written against. [`on_net`] is the bridge for anything that waits:
//! network work is spawned onto the runtime and *awaited* from a wgpui task, so the
//! code that touches the engine stays where the engine is and nothing takes a lock
//! across a socket.
//!
//! **Nothing of the transport's runs off that runtime**, and the rule is that blunt
//! because the exceptions are not visible from the call site: a `Broadcaster` method
//! returns immediately and *spawns* — a blob insert, a queue drain — and a tokio spawn
//! from a thread with no runtime in scope panics rather than failing. So the two
//! synchronous ones ([`send`], [`offer`], and the seeding beside them) run inside
//! [`in_net`], which enters the runtime without leaving the thread, and the event
//! stream is read on the runtime by [`pump`] and forwarded over a plain channel. The
//! forwarding hop is what that rule costs, and it is one move of a `RemoteEvent`.
//!
//! Which leaves the `Canvas` calling nothing of `stark-net`'s directly, and that is
//! the point rather than a side effect: "remember to enter the runtime" is a thing a
//! call site can forget, and what it produces when forgotten is a panic in a thread
//! that has nothing to do with the mistake.
//!
//! # What is owed can be answered without asking
//!
//! A joining peer is billed for the content the snapshot left out, and the web app
//! must fetch it before it may replay. This build has it: the shipped images are in
//! the binary (`crate::assets`), so settling the bill is a slice — and the promise
//! that lets a *peer* leave it out in the first place is the same catalog, passed as
//! [`NetOptions::resolvable`](stark_net::NetOptions::resolvable).

use std::future::Future;
use std::sync::OnceLock;

use stark_model::{AssetId, AssetNeed, DocumentFile, SubstrateId};
use stark_net::{
    Broadcaster, CollabSession, Events, Joined, NetOptions, RemoteEvent, SessionTicket,
};

use crate::render::Renderer;

/// Where a session stands, as the chrome needs to know it.
///
/// Three states rather than a `bool`, because binding and relay-readiness take a
/// noticeable moment and a second Share pressed inside it would bind a second
/// endpoint over the first.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Phase {
    #[default]
    Solo,
    /// Setup in flight — hosting or joining. Both acts are refused while it lasts.
    Connecting,
    /// Live in a shared session.
    Shared,
}

/// The live session and where it stands, which are one fact and so one value.
#[derive(Default)]
pub struct Collab {
    pub phase: Phase,
    /// The session itself. Held here rather than inside the runtime because
    /// [`Broadcaster::broadcast`] is synchronous and runs on the dispatch path: every
    /// committed action goes out through it, and a hop through a task per command is
    /// what put two dispatches in one frame onto the wire out of order.
    pub session: Option<CollabSession>,
    /// The incoming pump. Its life is the session's — replaced when one starts,
    /// dropped (and so cancelled) when one ends.
    pub pump: Option<wgpui::Task<()>>,
    /// The act in flight: a share, a join, or a link being minted. Held rather than
    /// detached, for `crate::files`' reason — a wgpui `Task` cancels when it is
    /// dropped, and a share dropped mid-bind is a link the user asked for and did not
    /// get.
    pub task: Option<wgpui::Task<()>>,
}

impl Collab {
    /// The session's sender, if there is a session.
    pub fn broadcaster(&self) -> Option<Broadcaster> {
        self.session.as_ref().map(CollabSession::broadcaster)
    }
}

/// What a session act produced, handed back to the view — `crate::files::Done`'s
/// shape and for its reason: the acts finish on a task and the state they settle
/// belongs to the `Canvas`, so they report rather than reach in.
pub enum Done {
    /// A session is live and this is the invitation to it.
    Started {
        session: Box<CollabSession>,
        events: Events,
        link: String,
    },
    /// A joined session, whose document has yet to be installed — the pieces come
    /// back before the log is replayed, because replaying is the engine's and the
    /// engine is not on this task.
    Arrived {
        session: Box<CollabSession>,
        events: Events,
        file: Box<DocumentFile>,
        owed: Vec<AssetNeed>,
    },
    /// A fresh link for a session that was already live.
    Link(String),
    /// Something went wrong, in words a person could act on.
    Failed(String),
}

/// The one tokio runtime this binary keeps, built on first use.
///
/// A leaked `static` rather than a field: dropping a multi-threaded runtime *blocks*
/// until its workers stop, and the place that would drop this is a window closing.
/// The threads go when the process does, which is the same moment they would have
/// gone anyway.
///
/// `None` if the runtime will not build, which is a machine with no threads to give —
/// reported as a failed share rather than as a panic, like every other way this window
/// can find it has no resource.
fn net() -> Option<&'static tokio::runtime::Runtime> {
    static NET: OnceLock<Option<tokio::runtime::Runtime>> = OnceLock::new();
    NET.get_or_init(|| tokio::runtime::Runtime::new().ok())
        .as_ref()
}

/// Run `work` on the network runtime and await its answer from anywhere.
///
/// The bridge, and deliberately the only one. Everything iroh does — binding,
/// dialing, relaying, minting a ticket — is written against tokio's timers and its
/// reactor, and polling that from wgpui's executor is not a slower version of working,
/// it is a hang. So the work is *spawned*, and what crosses back is a join handle,
/// which needs no runtime to poll.
///
/// `None` when there is no runtime, and also when the work panicked — the caller has
/// one thing to report either way, and neither is a state a painting should be lost
/// over.
fn on_net<T: Send + 'static>(
    work: impl Future<Output = T> + Send + 'static,
) -> impl Future<Output = Option<T>> {
    let handle = net().map(|rt| rt.spawn(work));
    async move {
        match handle {
            Some(handle) => handle.await.ok(),
            None => None,
        }
    }
}

/// Do `work` with the network runtime in scope, on the calling thread.
///
/// [`on_net`]'s synchronous twin, and the one that is easy to leave out: a
/// `Broadcaster`'s sending methods return immediately, which reads as "this does not
/// need an executor" and is exactly wrong — they hand the work to a spawned task, and
/// spawning with no runtime in scope is a panic rather than a refusal.
///
/// Entering costs a thread-local set and restore, which is what lets the dispatch path
/// stay synchronous — the thing that must not change, since ordering on the wire is
/// the order `broadcast` was called in (`Canvas::send`).
fn in_net<T>(work: impl FnOnce() -> T) -> Option<T> {
    let rt = net()?;
    let _guard = rt.enter();
    Some(work())
}

/// What this build promises it can produce without asking a peer (§12.4) — the
/// shipped catalog, which here is bytes in the binary rather than a fetch away.
fn options(secret: stark_net::SecretKey) -> NetOptions {
    NetOptions {
        secret: Some(secret),
        resolvable: stark_ui::assets::resolvable(),
        ..Default::default()
    }
}

/// Start hosting `doc`: bind the endpoint, seed the assets a peer might want, and
/// mint the first invitation.
///
/// `assets` is seeded rather than left to be fetched because the snapshot bundles
/// only what the *log* names: a stamp imported in this session and not yet painted
/// with is content a peer will ask for the moment it is (§12.4).
pub async fn host(doc: DocumentFile, assets: Vec<(AssetId, Vec<u8>)>) -> Done {
    let id = crate::identity::get();
    let opts = options(id.secret);
    let started = on_net(async move {
        let (session, events) = CollabSession::host(doc, opts).await?;
        let tx = session.broadcaster();
        for (id, bytes) in assets {
            tx.add_content(AssetNeed::Brush(id), bytes);
        }
        let link = stark_ui::collab::invite_link(&tx.ticket().await.to_string());
        Ok::<_, stark_net::NetError>((session, events, link))
    })
    .await;
    match started {
        Some(Ok((session, events, link))) => Done::Started {
            session: Box::new(session),
            events,
            link,
        },
        Some(Err(e)) => Done::Failed(format!("sharing failed: {e}")),
        None => Done::Failed("sharing failed: this build has no network runtime".to_string()),
    }
}

/// Join the session a pasted `link` names, bringing back the pieces the engine needs.
///
/// The document is **not** installed here: replaying it is the engine's, the engine
/// belongs to the view, and a future holding it across a dial is the one thing that
/// would freeze the window while a relay thought about it.
pub async fn join(link: String) -> Done {
    let Some(text) = stark_ui::collab::ticket_in(&link) else {
        return Done::Failed("there is no session link on the clipboard".to_string());
    };
    let ticket: SessionTicket = match text.parse() {
        Ok(ticket) => ticket,
        Err(e) => return Done::Failed(format!("that is not a session link: {e}")),
    };
    let id = crate::identity::get();
    let opts = options(id.secret);
    let arrived = on_net(async move { CollabSession::join(&ticket, opts).await }).await;
    match arrived {
        Some(Ok(Joined {
            session,
            events,
            document,
            owed,
        })) => Done::Arrived {
            session: Box::new(session),
            events,
            file: Box::new(document),
            owed,
        },
        Some(Err(e)) => Done::Failed(format!("joining failed: {e}")),
        None => Done::Failed("joining failed: this build has no network runtime".to_string()),
    }
}

/// Mint a fresh invitation to a session already live.
///
/// Minted per ask rather than kept from the share, because what a link is *worth*
/// changes: it names this peer and the members it could vouch were alive when it was
/// made, so a link handed out an hour in still works after its minter has gone.
pub async fn link(tx: Broadcaster) -> Done {
    match on_net(async move { tx.ticket().await.to_string() }).await {
        Some(ticket) => Done::Link(stark_ui::collab::invite_link(&ticket)),
        None => Done::Failed("could not mint a link".to_string()),
    }
}

/// Put this client's presence frame on the wire (§17.5).
///
/// Detached, and **best effort by design**: the next frame supersedes this one and
/// nothing in the log depends on it, so a caller that waited for it would be waiting
/// on a relay to say what it is about to say again anyway.
pub fn publish(tx: Broadcaster, frame: stark_model::PeerFrame) {
    let Some(rt) = net() else {
        return;
    };
    // The result is dropped rather than reported: a frame that did not go is one the
    // next tick makes again, and a title bar that said so would be saying it thirty
    // times a second.
    rt.spawn(async move {
        let _ = tx.publish(frame).await;
    });
}

/// Seed the session with everything this client has imported, so a peer can fetch any
/// asset the snapshot did not already carry.
pub fn seed(tx: &Broadcaster, assets: Vec<(AssetId, Vec<u8>)>) {
    in_net(|| {
        for (id, bytes) in assets {
            tx.add_content(AssetNeed::Brush(id), bytes);
        }
    });
}

/// Hand the session one asset's bytes, under the id an action is about to name.
pub fn offer(tx: &Broadcaster, need: AssetNeed, bytes: Vec<u8>) {
    in_net(|| tx.add_content(need, bytes));
}

/// Put committed actions on the wire, in the order they were committed.
///
/// One call for the whole drain rather than one per action, so the runtime is entered
/// once per dispatch — and so that the dispatch path has one thing to say about a
/// failure rather than one per action.
pub fn send(tx: &Broadcaster, actions: Vec<stark_model::document::Action>) -> Option<String> {
    let mut trouble = None;
    in_net(|| {
        for action in actions {
            if let Err(e) = tx.broadcast(action) {
                trouble = Some(format!("this stroke did not reach the session: {e}"));
            }
        }
    });
    trouble
}

/// Read the session's event stream on the runtime, and hand back a channel the view
/// can receive from.
///
/// The hop exists because of what [`Events::recv`] is: the transport's, so what it
/// touches is the transport's business, and this module has already been wrong once
/// about which of those touches need a reactor. A plain unbounded channel needs
/// nothing at all, and one is cheap to be sure about.
///
/// Cancellation stays the view's: dropping the receiver — which is what dropping the
/// pump task does — fails the next send and ends the forwarder with it.
pub fn pump(mut events: Events) -> tokio::sync::mpsc::UnboundedReceiver<RemoteEvent> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    if let Some(rt) = net() {
        rt.spawn(async move {
            while let Some(event) = events.recv().await {
                if tx.send(event).is_err() {
                    return;
                }
            }
        });
    }
    rx
}

/// Put content into the store `need` names.
///
/// The transport says which store, because the action that referenced the bytes is
/// the only thing that knows and a brush mask, a canvas substrate and a picture all
/// decode differently (§6.4, §6.6, §23).
fn install(r: &mut Renderer, need: AssetNeed, bytes: &[u8]) -> Result<(), String> {
    match need {
        AssetNeed::Brush(_) => r.import_brush_id(bytes).map(|_| ()),
        AssetNeed::Substrate(id) => r.accept_substrate(SubstrateId::Image(id), bytes),
        AssetNeed::Picture(id) => r.accept_picture(id, bytes),
    }
}

/// Settle what a joined session's snapshot left out, off this build's own catalog.
///
/// **Before the log is replayed**, which is the whole of why it is a separate step: a
/// substrate that is not registered when its `SetSubstrate` replays deposits every
/// later stroke against the flat stand-in, and those pixels are stored (§6.4).
///
/// An `Err` names content nobody here can produce, which is a session this build
/// cannot render — refused with the painting on screen untouched, exactly as a file
/// naming the same thing is (`Canvas::load`).
pub fn settle_owed(r: &mut Renderer, owed: &[AssetNeed]) -> Result<(), String> {
    for need in owed {
        let Some(png) = crate::assets::bytes_for(need.content()) else {
            return Err("this session uses content this build does not carry".to_string());
        };
        install(r, *need, png)?;
    }
    Ok(())
}

/// Make good on [`RemoteEvent::ResolveLocally`]: read the content out of this
/// build's own catalog, install it, and hand it back so the session can release the
/// action that was waiting on it.
///
/// Into the engine **before** the session is told, for [`settle_owed`]'s reason:
/// `add_content` releases the parked action, and that action is applied assuming its
/// content is already installed.
///
/// Doing nothing here would also be correct — the transport dials a peer after a
/// grace period. What this saves is the transfer.
fn supply(r: &mut Renderer, tx: &Broadcaster, need: AssetNeed) -> Option<String> {
    // The promise was made off the same table this reads, so a call it cannot answer
    // means the table disagrees with itself — which is worth saying out loud, because
    // what it costs is a peer's action parked until the transport gives up on us.
    let Some(png) = crate::assets::bytes_for(need.content()) else {
        return Some(
            "a collaborator asked for content this build promised and does not have".to_string(),
        );
    };
    if let Err(e) = install(r, need, png) {
        return Some(format!("content this build promised would not load: {e}"));
    }
    offer(tx, need, png.to_vec());
    None
}

/// What one remote event did, so the caller knows what it owes the screen.
#[derive(Default)]
pub struct Wake {
    /// The document moved, so what the panels draw from is stale.
    pub observe: bool,
    /// The canvas moved, so a frame is owed.
    pub repaint: bool,
    /// What went wrong taking the event, if anything.
    ///
    /// Carried back rather than logged, because this frontend logs nowhere: what a
    /// person can act on goes to the title bar (`Canvas::report`) and the rest does
    /// not exist. Content from a peer that will not install is exactly the kind of
    /// thing they should hear about — it is a stroke that will land looking wrong.
    pub trouble: Option<String>,
}

/// Feed one remote event into the engine (§12).
///
/// The whole of the incoming pump that touches the document, which is what makes it
/// worth its own function: the loop around it is a wgpui task and this is not.
pub fn apply(r: &mut Renderer, tx: &Broadcaster, event: RemoteEvent, now: f64) -> Wake {
    match event {
        // Repaint: an asset resolved off a *presence* head arrives while the peer's
        // live stroke is already on screen as a round-tip fallback, and the import is
        // what upgrades it. On the commit path the following action repaints anyway.
        RemoteEvent::Asset { need, bytes } => Wake {
            observe: false,
            repaint: true,
            trouble: install(r, need, &bytes)
                .err()
                .map(|e| format!("content from a collaborator would not load: {e}")),
        },
        RemoteEvent::Action(action) => {
            r.merge_remote(action);
            Wake {
                observe: true,
                repaint: true,
                trouble: None,
            }
        }
        // A peer moved, switched layer, or drew another stretch of a live stroke
        // (§17.4). Only the canvas half matters here: the roster is chrome this
        // frontend does not draw yet, so a remote pointer move owes nothing at all.
        RemoteEvent::Presence { actor, frame } => Wake {
            observe: false,
            repaint: r.merge_presence(actor, frame, now),
            trouble: None,
        },
        RemoteEvent::ResolveLocally { need } => Wake {
            trouble: supply(r, tx, need),
            ..Wake::default()
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The bug this module was born with.** A `Broadcaster`'s sending methods look
    /// synchronous and are not — they spawn — so calling one from the window's own
    /// thread panicked with "there is no reactor running", which is a crash in a
    /// thread that has nothing to do with the mistake.
    ///
    /// What is checked is the fix rather than the symptom: inside [`in_net`] there is
    /// a runtime for a spawn to find. Nothing here binds an endpoint, so it is the
    /// seam that is tested and not the transport — which is the point, since the seam
    /// is the part this crate owns.
    #[test]
    fn work_inside_the_net_has_a_runtime_to_spawn_on() {
        let found = in_net(|| tokio::runtime::Handle::try_current().is_ok());
        assert_eq!(found, Some(true), "in_net did not enter the runtime");
        // And the other way round, which is the state every call site is in by
        // default: this thread is the window's, and it has no runtime at all.
        assert!(tokio::runtime::Handle::try_current().is_err());
    }

    /// The asynchronous half of the same seam: work goes *onto* the runtime, and its
    /// answer comes back to a thread that has none — which is what lets the code that
    /// touches the engine stay where the engine is.
    #[test]
    fn work_sent_to_the_net_answers_a_thread_without_one() {
        let answered = pollster::block_on(on_net(async {
            // Something only a runtime can do, so the test fails if the work were
            // quietly running on the caller's thread instead.
            tokio::runtime::Handle::try_current().is_ok()
        }));
        assert_eq!(answered, Some(true));
    }
}
