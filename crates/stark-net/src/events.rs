//! The engine-facing seam (§12.4): what the UI pump consumes — [`RemoteEvent`]s
//! off an [`Events`] stream — and what a frontend hands in, the [`NetOptions`]
//! and the identity mapping. No swarm machinery lives here;
//! [`session`](crate::session) owns that and feeds this.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use bytes::Bytes;
use iroh::{EndpointId, SecretKey};
use stark_model::document::{Action, ActorId};
use stark_model::peer::PeerFrame;
use stark_model::{AssetId, AssetNeed};
use tokio::sync::mpsc;

use crate::cancel::Cancel;

/// How many presence frames may sit queued for the engine at once, across all
/// peers.
///
/// Presence is a latch, so a deep queue is not resilience — it is staleness. The
/// engine drops any frame that does not advance `(boot, seq)` anyway (§17.5), so
/// everything queued behind the newest is work the UI will do and throw away.
/// This cap is what stops a UI that has fallen behind from accumulating them
/// without bound: roughly half a second of five peers stroking at 30 Hz, long
/// enough to ride out a slow frame and short enough that what does arrive is
/// still current.
pub(crate) const PRESENCE_QUEUE: usize = 64;

/// Map an iroh endpoint identity to the engine's author id (§12.4:
/// "an iroh node id *is* the `ActorId`"). `ActorId` is 8 bytes to keep every
/// action id small, so this takes the key's first 8 bytes — collisions across
/// the handful of peers in a drawing session are negligible (birthday bound
/// ≈ n²/2⁶⁵), and a collision would only merge two peers' undo scopes.
pub fn actor_from_endpoint_id(id: EndpointId) -> ActorId {
    let bytes = id.as_bytes();
    ActorId(u64::from_le_bytes(
        bytes[..8].try_into().expect("32-byte key"),
    ))
}

/// Something a peer did, to be applied to the local engine. Apply in order:
/// assets arrive before the action that references them.
#[derive(Debug, Clone)]
pub enum RemoteEvent {
    /// Content a remote action references, resolved off a peer — feed to the store
    /// `need` names before the action that wanted it: a brush image to
    /// `Engine::import_brush`, a canvas substrate to
    /// `Engine::accept_substrate`.
    Asset { need: AssetNeed, bytes: Bytes },
    /// A committed remote action — feed to
    /// `Engine::merge_remote`.
    Action(Action),
    /// A peer's presence — feed to
    /// `Engine::merge_presence`
    /// (§17.4). Unlike an action this may be dropped freely: nothing in
    /// the log refers to it, so losing one costs a frame of someone else's cursor
    /// and nothing else.
    Presence { actor: ActorId, frame: PeerFrame },
    /// A remote action needs content this client said it could resolve itself
    /// ([`NetOptions::resolvable`]) — the promise being called in.
    ///
    /// Read the bytes from wherever you promised they were and hand them back
    /// with [`Broadcaster::add_content`](crate::Broadcaster::add_content);
    /// the action waiting on them is released
    /// when you do. Nothing else is expected of you and nothing is on fire.
    ///
    /// **Ignoring it is safe.** After a short grace period the transport dials a
    /// peer for the content exactly as it would have without the promise, so a
    /// frontend that cannot deliver — or does not handle this at all — loses a
    /// little time and nothing else.
    ResolveLocally { need: AssetNeed },
}

/// The stream of remote edits, handed out once by
/// [`CollabSession::host`](crate::CollabSession::host) /
/// [`CollabSession::join`](crate::CollabSession::join). Pump it into the engine.
///
/// It is a type rather than a bare channel because the presence quota is
/// accounted here: a slot is taken when a frame is queued and freed when one is
/// handed over, so "queued presence" means the same number on both ends without
/// the consumer knowing there is a quota at all.
#[derive(Debug)]
pub struct Events {
    pub(crate) rx: mpsc::UnboundedReceiver<RemoteEvent>,
    pub(crate) presence: Arc<PresenceQuota>,
    /// The session's stop signal — what makes `recv`'s "once the session has
    /// ended" a fact it can see rather than infer from the channel.
    pub(crate) cancel: Cancel,
}

impl Events {
    /// The next remote event, or `None` once the session has ended.
    ///
    /// Ending does not discard what already arrived: events queued when the
    /// session ended are still handed out, and `None` comes only once they are
    /// drained. Never hangs after the end — the stop arm of the race completes
    /// immediately once the signal is up.
    pub async fn recv(&mut self) -> Option<RemoteEvent> {
        let rx = &mut self.rx;
        let cancel = &self.cancel;
        let received = n0_future::future::race(rx.recv(), async {
            cancel.stopped_wait().await;
            None
        })
        .await;
        // `None` is the stop signal (or every sender gone): close the channel
        // so the end is terminal — a straggling sender now fails fast instead
        // of queueing into the void and un-ending the stream — while what was
        // already queued still drains first.
        let event = match received {
            Some(event) => event,
            None => {
                rx.close();
                rx.recv().await?
            }
        };
        self.took(&event);
        Some(event)
    }

    /// The next remote event if one is already queued — for a pump driven by
    /// something other than this stream, such as a frame clock.
    pub fn try_recv(&mut self) -> Option<RemoteEvent> {
        let event = self.rx.try_recv().ok()?;
        self.took(&event);
        Some(event)
    }

    fn took(&self, event: &RemoteEvent) {
        if matches!(event, RemoteEvent::Presence { .. }) {
            self.presence.release();
        }
    }
}

/// Slots for presence frames in flight to the engine (see [`PRESENCE_QUEUE`]).
#[derive(Debug, Default)]
pub(crate) struct PresenceQuota(AtomicUsize);

impl PresenceQuota {
    /// Take a slot, or `false` when the engine is already this far behind — in
    /// which case the frame is dropped, which is the one thing presence is
    /// allowed to have happen to it. Nothing in the log refers to it, the newest
    /// frame supersedes it, and the author re-sends its whole gesture on the
    /// next resync frame (§17.4, §17.5).
    pub fn reserve(&self) -> bool {
        self.0
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |queued| {
                (queued < PRESENCE_QUEUE).then_some(queued + 1)
            })
            .is_ok()
    }

    pub fn release(&self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Connectivity configuration for a session.
#[derive(Debug, Default, Clone)]
pub struct NetOptions {
    /// Reuse a persisted identity; a fresh key is generated otherwise.
    pub secret: Option<SecretKey>,
    /// Skip the public n0 relay + address-lookup infrastructure and rely on
    /// the ticket's direct socket addresses only — for LAN use and tests.
    pub local_only: bool,
    /// Content this client can produce **without asking anyone** — the ids of the
    /// assets that ship with its build (§12.4).
    ///
    /// It does two things, and they are the same promise read from either end. A
    /// joiner's list is left out of the snapshot it is sent
    /// ([`Joined::owed`](crate::Joined::owed) is
    /// the bill). And for the rest of the session, a remote action naming one of
    /// these raises [`RemoteEvent::ResolveLocally`] instead of dialling a peer —
    /// so a collaborator switching to a substrate this app ships with costs a read
    /// from disk rather than megabytes over the wire.
    ///
    /// Empty promises nothing, and is the safe default: every id then travels the
    /// wire as it would with no local catalog at all.
    pub resolvable: Vec<AssetId>,
}

impl NetOptions {
    /// LAN/test configuration: no relays, no external lookups.
    pub fn local() -> Self {
        Self {
            local_only: true,
            ..Self::default()
        }
    }
}
