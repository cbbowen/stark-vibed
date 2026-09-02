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
    /// the insurance is only as good as it is current. The sampling is here;
    /// the policy — order, cap, which hops travel — is [`mint`]'s.
    pub async fn ticket(&self) -> SessionTicket {
        let mut neighbors = Vec::new();
        for id in self
            .neighbors
            .snapshot_sorted()
            .into_iter()
            .take(TICKET_NEIGHBORS)
        {
            // The proven path only — the one this peer's traffic rides right
            // now. Wrong is worse than missing here: a bare id still resolves
            // through address lookup on a WAN session, while a joiner spends
            // [`DIAL_TIMEOUT`] discovering that a stale address does not answer.
            neighbors.push((id, self.dialer.selected_addr(id).await));
        }
        mint(self.ticket_addr.clone(), &neighbors, self.topic)
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

/// The minting policy, pure: assemble a link from this peer's address, the
/// neighbors sampled beside it — each with the one path its traffic was riding
/// — and the topic.
///
/// Three decisions live here, where a test can hold them still:
///
/// - **One membership spells one link.** Neighbors are sorted by key bytes and
///   capped at [`TICKET_NEIGHBORS`], so re-minting over an unchanged set yields
///   byte-identical text — the frontend re-mints on a cadence and rewrites the
///   invitation only on change.
/// - **Custom (WebRTC) addrs are left off** the extra members, for the reason
///   `Dialer::learn` drops them: a peer derives one from the endpoint id
///   itself, so a link gains nothing by fixing in where a channel happened to
///   be attached at minting time.
/// - **The minter comes first**, addrs untouched — it is the member most
///   recently known alive.
pub(crate) fn mint(
    local: EndpointAddr,
    neighbors: &[(EndpointId, Option<TransportAddr>)],
    topic: TopicId,
) -> SessionTicket {
    let mut named = neighbors.to_vec();
    named.sort_unstable_by_key(|(id, _)| *id.as_bytes());
    let mut members = vec![local];
    for (id, addr) in named.into_iter().take(TICKET_NEIGHBORS) {
        let hops = addr.filter(|addr| !matches!(addr, TransportAddr::Custom(_)));
        members.push(EndpointAddr::from_parts(id, hops));
    }
    SessionTicket { members, topic }
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

/// The minting policy ([`mint`]), without an endpoint under it.
#[cfg(test)]
mod tests {
    use iroh_base::CustomAddr;

    use super::*;
    use crate::testutil::endpoint;

    const TOPIC: TopicId = TopicId::from_bytes([7u8; 32]);

    fn ip(port: u16) -> TransportAddr {
        TransportAddr::Ip(format!("127.0.0.1:{port}").parse().expect("a socket"))
    }

    fn local() -> EndpointAddr {
        EndpointAddr::from_parts(endpoint(9), [ip(4433)])
    }

    /// One membership spells one link: however the neighbor set was enumerated,
    /// a re-mint over the same members is byte-identical text — the frontend
    /// rewrites the invitation only when the text moves.
    #[test]
    fn a_remint_of_an_unchanged_membership_spells_the_same_link() {
        let neighbors = [
            (endpoint(3), Some(ip(1))),
            (endpoint(1), Some(ip(2))),
            (endpoint(2), None),
        ];
        let mut permuted = neighbors.clone();
        permuted.rotate_left(1);

        let first = mint(local(), &neighbors, TOPIC);
        let again = mint(local(), &permuted, TOPIC);
        assert_eq!(first.to_string(), again.to_string());

        // The minter is first; the rest follow in key order.
        assert_eq!(first.members[0].id, endpoint(9));
        let mut sorted = vec![endpoint(1), endpoint(2), endpoint(3)];
        sorted.sort_unstable_by_key(|id| *id.as_bytes());
        let named: Vec<_> = first.members[1..].iter().map(|m| m.id).collect();
        assert_eq!(named, sorted);
    }

    /// A link has a length budget, so it names at most [`TICKET_NEIGHBORS`]
    /// members besides its minter — the first of the sorted set, so which ones
    /// survive the cap is a fact of the membership rather than of enumeration.
    #[test]
    fn a_link_names_at_most_its_cap_of_neighbors() {
        let neighbors: Vec<_> = (1..=TICKET_NEIGHBORS as u8 + 2)
            .map(|tag| (endpoint(tag), None))
            .collect();

        let ticket = mint(local(), &neighbors, TOPIC);
        assert_eq!(ticket.members.len(), 1 + TICKET_NEIGHBORS);

        let mut sorted: Vec<_> = neighbors.iter().map(|(id, _)| *id).collect();
        sorted.sort_unstable_by_key(|id| *id.as_bytes());
        let named: Vec<_> = ticket.members[1..].iter().map(|m| m.id).collect();
        assert_eq!(named, sorted[..TICKET_NEIGHBORS]);
    }

    /// A WebRTC addr is derived from the endpoint id, so fixing one into a link
    /// buys nothing and can wedge a dial (`Dialer::learn`): an extra member
    /// riding one travels as a bare id, while a proven IP or relay path is kept
    /// — and the minter's own addrs are not the policy's to touch.
    #[test]
    fn a_webrtc_addr_is_left_off_the_extra_members() {
        let custom = TransportAddr::Custom(CustomAddr::from_parts(7, b"a data channel"));
        let neighbors = [(endpoint(1), Some(custom)), (endpoint(2), Some(ip(1)))];

        let ticket = mint(local(), &neighbors, TOPIC);
        for member in &ticket.members {
            assert!(
                !member
                    .addrs
                    .iter()
                    .any(|addr| matches!(addr, TransportAddr::Custom(_))),
                "a custom addr was minted into the link"
            );
        }
        let by_id = |tag: u8| {
            ticket
                .members
                .iter()
                .find(|m| m.id == endpoint(tag))
                .expect("named")
        };
        assert!(by_id(1).addrs.is_empty(), "bare id, not a stale channel");
        assert_eq!(by_id(2).addrs.len(), 1, "the proven path is kept");
        assert!(
            !ticket.members[0].addrs.is_empty(),
            "the minter keeps its own"
        );
    }
}
