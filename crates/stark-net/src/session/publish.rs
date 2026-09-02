//! The send side (§12.4): the [`Broadcaster`] handle the UI feeds, and the
//! session's one [`send_loop`] task, which is what makes the wire order the
//! commit order.

use std::sync::Arc;

use bytes::Bytes;
use iroh::{EndpointAddr, EndpointId, TransportAddr};
use iroh_gossip::api::GossipSender;
use iroh_gossip::proto::TopicId;
use stark_model::AssetNeed;
use stark_model::document::{Action, ActorId};
use stark_model::peer::PeerFrame;
use tokio::sync::mpsc;

use crate::Result;
use crate::backend::Dialer;
use crate::cancel::Cancel;
use crate::events::actor_from_endpoint_id;
use crate::neighbors::Neighbors;
use crate::ticket::SessionTicket;
use crate::waitlist::Waitlist;
use crate::wire::{StampedRef, WireRef};

/// How many members besides the minter a ticket names (§12.4).
///
/// Naming more buys insurance against the minter having left by the time the
/// link is opened, and a few is enough: any one live member admits the joiner to
/// the whole swarm. A link is also something a person pastes into a chat window,
/// so it has a length budget — a member costs tens of bytes, of which the spelling
/// (`ticket`: deflated, then base64url) gives back most but not all — and a joiner
/// pays up to [`DIAL_TIMEOUT`](super::join::DIAL_TIMEOUT) per dead name in it.
const TICKET_NEIGHBORS: usize = 3;

/// How a direct connection to a session member currently travels. Gossip
/// links may migrate (relay first, then direct once hole punching or a WebRTC
/// bootstrap lands), so this is sampled at query time rather than recorded at
/// connection time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkKind {
    /// A peer-to-peer WebRTC data channel (the browser's direct path).
    WebRtc,
    /// A hole-punched UDP path (native iroh's direct path).
    Direct,
    /// Via an iroh relay server.
    Relay,
    /// No path selected yet.
    Unknown,
}

/// How this client's connection to one session member currently travels —
/// the answer to "are we peer-to-peer or riding a relay?".
///
/// Keyed by the member's [`ActorId`] so the UI can join it against the
/// presence roster. Members with no entry are not gossip neighbors of this
/// client; their traffic is forwarded by whoever is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeerLink {
    /// The author id the peer's endpoint identity maps to.
    pub actor: ActorId,
    /// How the connection reaches them right now.
    pub kind: LinkKind,
}

/// A detached publishing handle onto a
/// [`CollabSession`](crate::CollabSession): broadcast actions,
/// register assets and mint tickets without holding the session itself. All
/// clones share the same gossip topic and mirror.
#[derive(Clone)]
pub struct Broadcaster {
    pub(super) local_id: EndpointId,
    /// Presence goes straight out: it is best-effort by design and on its own
    /// cadence, so it must not queue behind a long stroke.
    pub(super) sender: GossipSender,
    /// Committed actions go through one queue drained by one task, which is what
    /// makes their wire order the order they were committed in.
    pub(super) outgoing: mpsc::UnboundedSender<Bytes>,
    pub(super) dialer: Dialer,
    pub(super) neighbors: Neighbors,
    pub(super) waitlist: Arc<Waitlist>,
    pub(super) topic: TopicId,
    /// How this peer is reached, minted at bind time — the first name on every
    /// ticket this peer hands out.
    pub(super) ticket_addr: EndpointAddr,
}

impl Broadcaster {
    /// Broadcast one locally-committed action (from
    /// `Engine::take_outbox`) to the swarm.
    ///
    /// Returns once the action is mirrored and queued; the session's one send task
    /// puts it on the wire. That task is what makes the wire order the order things
    /// were committed in — a caller spawning a send per dispatch raced two of them
    /// onto the same sender, and every inversion cost a timeline resync on every
    /// receiver.
    pub fn broadcast(&self, action: Action) -> Result<()> {
        let need = stark_model::action_content(&action);
        let bytes = self.encode(WireRef::Action(&action), need)?;
        // Mirrored after encoding and before queueing, which is what lets the action
        // be moved rather than duplicated — and mirrored whether or not it ever goes
        // out, since that is what lets `reconcile` hand it to a peer that missed it.
        self.waitlist.published(action);
        // Queued, not sent. The caller returns immediately and the session's one
        // send task puts it on the wire; a closed queue means the session is over,
        // which is not this caller's problem.
        let _ = self.outgoing.send(bytes);
        Ok(())
    }

    /// Publish this client's presence (§17.4).
    ///
    /// Deliberately *not* mirrored: presence is not part of the document, so it is
    /// never served to a joiner and never reaches a file. And deliberately
    /// best-effort — a frame that cannot be sent is dropped rather than retried,
    /// because the next one supersedes it anyway.
    pub async fn publish(&self, frame: PeerFrame) -> Result<()> {
        let need = stark_model::presence_content(&frame);
        let bytes = self.encode(WireRef::Presence(&frame), need)?;
        Ok(self.sender.broadcast(bytes).await?)
    }

    /// Stamp and encode one payload, attaching the transfer hash for whatever
    /// content it references.
    fn encode(&self, wire: WireRef<'_>, need: Option<AssetNeed>) -> Result<Bytes> {
        // Attach the blob hash for the referenced content, so receivers that lack
        // it know what to fetch. `add_content` accompanies the import, for a
        // substrate as for a brush, so a registered lookup is the normal case and a
        // miss means that ordering was broken.
        let asset = need.and_then(|need| {
            let hash = self.waitlist.transfer_hash(need.content());
            if hash.is_none() {
                // What a call site that committed before registering looks like
                // from here. Reported at the fault rather than left to surface
                // as a warning on someone else's canvas: with no transfer hash
                // the receiver cannot fetch the bytes at all, and for a substrate
                // that is a permanent divergence (§6.4).
                //
                // Sent anyway. The action is already committed locally, so
                // withholding it would guarantee the divergence this is warning
                // about rather than merely risk it — and a peer that gets the
                // content some other way still converges.
                tracing::error!(
                    "broadcasting a payload referencing unregistered {need:?}; \
                     add_content must precede the commit that references it"
                );
            }
            hash
        });
        let stamped = StampedRef {
            origin: self.local_id,
            asset,
            wire,
        };
        Ok(crate::codec::encode_stamped_ref(&stamped)?.into())
    }

    /// Register content so joiners can be served and peers can fetch it — a brush
    /// image alongside
    /// `Engine::import_brush`, a canvas substrate
    /// alongside `Engine::import_substrate`.
    ///
    /// Call it *before* committing an action that references the content: the
    /// broadcast attaches a transfer hash looked up here, and an action that goes out
    /// without one leaves receivers unable to fetch what it needs. Getting that order
    /// wrong logs an error naming the content, from the client that committed it —
    /// the fault is only visible there, since what it produces at the far end is
    /// indistinguishable from content that has not arrived yet.
    pub fn add_content(&self, need: AssetNeed, bytes: impl Into<Bytes>) {
        let bytes = bytes.into();
        let hash = self.dialer.add_blob(bytes.clone());
        // Through the waitlist, not straight into the mirror: a remote action
        // may already be parked on exactly this content, and a local import
        // satisfies it as well as a fetch would.
        self.waitlist.imported(need, bytes, hash);
    }

    /// The ticket others use to join — every member can hand one out, so the
    /// session survives the host leaving.
    ///
    /// It names *this* peer first, then up to [`TICKET_NEIGHBORS`] members it is
    /// connected to right now — so the link also survives this peer leaving
    /// between the minting and the pasting: a joiner tries members in order, and
    /// any one of them admits it. Minted per call rather than stored, because
    /// the insurance is only as good as it is current. Sorted
    /// ([`Neighbors::snapshot_sorted`]) so one membership always spells one link.
    pub async fn ticket(&self) -> SessionTicket {
        let neighbors = self.neighbors.snapshot_sorted();
        let mut members = vec![self.ticket_addr.clone()];
        for id in neighbors.into_iter().take(TICKET_NEIGHBORS) {
            // The proven path only — the one this peer's traffic rides right
            // now. Wrong is worse than missing here: a bare id still resolves
            // through address lookup on a WAN session, while a joiner spends
            // [`DIAL_TIMEOUT`] discovering that a stale address does not answer.
            // Custom (WebRTC) addrs are left off for the reason `Dialer::learn`
            // drops them: a peer derives one from the endpoint id itself, so a
            // link gains nothing by fixing in where a channel happened to be
            // attached at minting time.
            let hops = self
                .dialer
                .selected_addr(id)
                .await
                .filter(|addr| !matches!(addr, TransportAddr::Custom(_)));
            members.push(EndpointAddr::from_parts(id, hops));
        }
        SessionTicket {
            members,
            topic: self.topic,
        }
    }

    /// How each gossip-neighbor session member is reached right now — direct
    /// (WebRTC or hole-punched UDP) or via a relay. Sampled per call; a link
    /// migrates from relay to direct when hole punching or a WebRTC bootstrap
    /// lands, so poll rather than cache.
    pub async fn links(&self) -> Vec<PeerLink> {
        let neighbors = self.neighbors.snapshot_sorted();
        let mut links = Vec::with_capacity(neighbors.len());
        for id in neighbors {
            let kind = match self.dialer.selected_addr(id).await {
                Some(TransportAddr::Custom(_)) => LinkKind::WebRtc,
                Some(TransportAddr::Ip(_)) => LinkKind::Direct,
                Some(TransportAddr::Relay(_)) => LinkKind::Relay,
                _ => LinkKind::Unknown,
            };
            links.push(PeerLink {
                actor: actor_from_endpoint_id(id),
                kind,
            });
        }
        links
    }
}

/// The session's one send task: committed actions reach the wire in the order they
/// were queued, which is the order they were committed in.
///
/// A failed send is not retried here. The action is already in this peer's mirror,
/// so the next member to sweep collects it ([`reconcile`](crate::reconcile)) — and
/// retrying in place would hold every action behind it up for a peer that is about
/// to ask for it anyway.
pub(super) async fn send_loop(
    cancel: Cancel,
    sender: GossipSender,
    mut queue: mpsc::UnboundedReceiver<Bytes>,
) {
    loop {
        // Raced against the stop signal: what is still queued when the session
        // ends is already mirrored, so a peer's sweep recovers it.
        let bytes = n0_future::future::race(queue.recv(), async {
            cancel.stopped_wait().await;
            None
        })
        .await;
        let Some(bytes) = bytes else { return };
        if let Err(e) = sender.broadcast(bytes).await {
            tracing::warn!("broadcast failed: {e}; a peer's sweep will recover it");
        }
    }
}
