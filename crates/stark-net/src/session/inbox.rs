//! The receive side (§12.4): the gossip loop, and the per-payload decisions it
//! plumbs.
//!
//! [`recv_loop`] is the plumbing — the stop-signal race, the neighbor
//! bookkeeping, the decode; [`Inbox::on_payload`] is everything after decode,
//! kept apart so the decisions are testable without a live swarm. The loop
//! itself never waits on the network: an action that references content this
//! peer lacks is parked on the [`Waitlist`] and released by the resolver that
//! fetches it; everything else keeps flowing past.

use std::ops::ControlFlow;
use std::sync::Arc;

use iroh_gossip::api::{Event as GossipEvent, GossipReceiver};
use n0_future::StreamExt;
use tokio::sync::mpsc;

use crate::events::{PresenceQuota, RemoteEvent, actor_from_endpoint_id};
use crate::reconcile::Wiring;
use crate::waitlist::{MustFetch, Waitlist};
use crate::wire::{Stamped, Wire};

/// The per-payload half of the receive side: decode has happened, and what is
/// left is deciding — park, drop, or hand to the engine.
pub(super) struct Inbox {
    waitlist: Arc<Waitlist>,
    presence: Arc<PresenceQuota>,
    tx: mpsc::UnboundedSender<RemoteEvent>,
}

impl Inbox {
    /// One decoded gossip payload's whole decision. A fetch it requires is
    /// handed back rather than started — only the caller holds the endpoints to
    /// ask and the spawn hook ([`Wiring::spawn_resolver`]).
    /// [`ControlFlow::Break`] means the engine is gone and the loop should end.
    fn on_payload(&self, stamped: Stamped) -> ControlFlow<(), Option<MustFetch>> {
        let Stamped {
            origin,
            asset: asset_hash,
            wire,
        } = stamped;
        let action = match wire {
            Wire::Action(action) => action,
            // Presence bypasses the mirror entirely: it is not part of the document,
            // so it is never served to a joiner and never reaches a file.
            Wire::Presence(frame) => {
                // A live stroke's head names its brush image just like the
                // eventual commit will. Admitted detached — presence must never
                // wait on a fetch — so the rest of the gesture renders with the
                // real shape as soon as the bytes land; until then the
                // receiver's preview degrades to the round tip. The commit that
                // follows names the same content and parks behind *this*
                // resolver rather than starting a second one.
                let fetch = stark_model::presence_content(&frame)
                    .and_then(|need| self.waitlist.admit_detached(need, asset_hash));
                // Dropped rather than queued when the engine is already
                // `PRESENCE_QUEUE` frames behind: a frame the UI would reach
                // late is one the engine rejects as stale anyway.
                if !self.presence.reserve() {
                    tracing::trace!("presence frame dropped; the engine is behind");
                    return ControlFlow::Continue(fetch);
                }
                let event = RemoteEvent::Presence {
                    actor: actor_from_endpoint_id(origin),
                    frame,
                };
                if self.tx.send(event).is_err() {
                    return ControlFlow::Break(());
                }
                return ControlFlow::Continue(fetch);
            }
        };

        // Two identities travel with every action and nothing upstream ties them
        // together: `origin` picks who to fetch its content from, while
        // `id.actor` owns its undo scope (§12.3) and half the total order key.
        // Gossip reports only the delivering neighbour, so `origin` is
        // self-declared — the same trust the payload already carries (§12.5) —
        // which makes this a consistency check and not authentication.
        //
        // Dropped rather than accepted, because an action whose author is wrong
        // is one whose undo scope is wrong: applying it puts something in the log
        // that the peer who appears to have written it cannot take back.
        if actor_from_endpoint_id(origin) != action.id.actor {
            tracing::warn!(
                origin = %origin.fmt_short(),
                actor = ?action.id.actor,
                "dropping an action whose author does not match its sender"
            );
            return ControlFlow::Continue(None);
        }

        // Whatever the action references has to reach the engine first, so the
        // engine can apply it faithfully. The action waits for it — parked, not
        // awaited here, so nothing else in the session waits with it
        // ([`Waitlist::admit`]). The origin authored the action and so
        // definitely holds the content; the neighbour that forwarded it may not.
        ControlFlow::Continue(self.waitlist.admit(action, asset_hash))
    }
}

/// The gossip receive loop: decode, park what is waiting on content, mirror,
/// forward to the engine. Also maintains the neighbor set and kicks off the
/// WebRTC bootstrap for every new neighbor.
pub(super) async fn recv_loop(
    wiring: Wiring,
    mut gossip: GossipReceiver,
    presence: Arc<PresenceQuota>,
    tx: mpsc::UnboundedSender<RemoteEvent>,
) {
    let inbox = Inbox {
        waitlist: wiring.waitlist.clone(),
        presence,
        tx,
    };
    loop {
        // Raced against the stop signal: the stream on its own ends only as a
        // side effect of the endpoint closing, which teardown must not wait on.
        let next = n0_future::future::race(gossip.next(), async {
            wiring.cancel.stopped_wait().await;
            None
        })
        .await;
        let Some(event) = next else {
            return;
        };
        let message = match event {
            Ok(GossipEvent::Received(message)) => message,
            Ok(GossipEvent::Lagged) => {
                // The swarm outran this peer and skipped the rest. Which actions
                // went is not knowable from here, so this is one of the two things
                // reconciliation exists for: compare logs with a neighbour shortly,
                // once what is still in flight has landed (§12.5). A lagged
                // *presence* stream needs no recovery at all — the author re-sends
                // its whole gesture on the next resync frame (§17.5).
                tracing::warn!("gossip lagged; reconciling to recover what was missed");
                wiring.prompt.raise();
                continue;
            }
            Ok(GossipEvent::NeighborUp(peer)) => {
                tracing::debug!(%peer, "gossip neighbor up");
                wiring
                    .neighbors
                    .lock()
                    .expect("neighbors poisoned")
                    .insert(peer);
                wiring.dialer.ensure_direct(peer);
                continue;
            }
            Ok(GossipEvent::NeighborDown(peer)) => {
                tracing::debug!(%peer, "gossip neighbor down");
                wiring
                    .neighbors
                    .lock()
                    .expect("neighbors poisoned")
                    .remove(&peer);
                continue;
            }
            Err(e) => {
                tracing::debug!("gossip receiver closed: {e}");
                return;
            }
        };

        let stamped: Stamped = match crate::codec::decode(&message.content) {
            Ok(stamped) => stamped,
            Err(e) => {
                tracing::warn!("undecodable gossip payload: {e}");
                continue;
            }
        };
        let (origin, from) = (stamped.origin, message.delivered_from);
        match inbox.on_payload(stamped) {
            ControlFlow::Continue(Some(fetch)) => {
                wiring.spawn_resolver(fetch.need, fetch.hash, origin, from);
            }
            ControlFlow::Continue(None) => {}
            ControlFlow::Break(()) => return,
        }
    }
}

/// The per-payload decisions, without a swarm under them.
///
/// These construct [`Stamped`] values directly and assert through the waitlist
/// and the event channel — the two places a decision lands — because none of
/// the properties here (who is dropped, what parks, what a full quota does) is
/// about the network that normally delivers the payload.
#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use bytes::Bytes;
    use iroh::{EndpointId, SecretKey};
    use iroh_blobs::Hash;
    use stark_model::document::{
        Action, ActionId, ActionKind, ActorId, BrushParams, BrushShape, LayerId, StrokeRecord,
    };
    use stark_model::geom::IVec2;
    use stark_model::peer::PeerFrame;
    use stark_model::{AssetId, AssetNeed, DocumentFile, Srgb};
    use tokio::sync::mpsc;

    use super::*;
    use crate::cancel::Cancel;
    use crate::events::{Events, PRESENCE_QUEUE};
    use crate::mirror::Mirror;

    fn endpoint(tag: u8) -> EndpointId {
        SecretKey::from_bytes(&[tag; 32]).public()
    }

    /// An inbox over an empty mirror, and the [`Events`] stream its decisions
    /// surface on — the same pair `finish` wires, minus the swarm.
    fn setup() -> (Inbox, Events) {
        let mirror = Arc::new(Mutex::new(Mirror::from_file(
            &DocumentFile::new(Vec::new()),
        )));
        let (tx, rx) = mpsc::unbounded_channel();
        let presence = Arc::new(PresenceQuota::default());
        let inbox = Inbox {
            waitlist: Arc::new(Waitlist::new(mirror, tx.clone(), &[])),
            presence: presence.clone(),
            tx,
        };
        let events = Events {
            rx,
            presence,
            cancel: Cancel::default(),
        };
        (inbox, events)
    }

    /// An action that references no content at all.
    fn action_by(actor: ActorId, lamport: u64) -> Action {
        Action {
            id: ActionId { lamport, actor },
            kind: ActionKind::SetSubstrateColor(Srgb::new([0.0; 3])),
        }
    }

    /// An action naming the brush `AssetId([tag; 32])`.
    fn action_needing(actor: ActorId, lamport: u64, tag: u8) -> Action {
        Action {
            id: ActionId { lamport, actor },
            kind: ActionKind::CommitStroke(StrokeRecord {
                layer: LayerId::ROOT,
                brush: BrushParams {
                    shape: BrushShape::Stamp(AssetId([tag; 32])),
                    ..BrushParams::default()
                },
                path: Vec::new(),
                seed: 0,
                start: 0.0,
                translation: IVec2::ZERO,
            }),
        }
    }

    fn stamped(origin: EndpointId, asset: Option<Hash>, wire: Wire) -> Stamped {
        Stamped {
            origin,
            asset,
            wire,
        }
    }

    fn frame(seq: u64) -> PeerFrame {
        PeerFrame {
            boot: 1,
            seq,
            name: None,
            active_layer: LayerId::ROOT,
            cursor: None,
            gesture: None,
            leaving: false,
        }
    }

    /// The author/origin consistency check: an action whose declared author is
    /// not its sender is dropped whole — not surfaced, and not mirrored, so
    /// this peer never serves it onward either.
    #[test]
    fn an_action_is_dropped_unless_its_author_matches_its_origin() {
        let (inbox, mut events) = setup();
        let origin = endpoint(1);
        let honest = actor_from_endpoint_id(origin);
        let forged = action_by(ActorId(honest.0 ^ 1), 1);

        assert_eq!(
            inbox.on_payload(stamped(origin, None, Wire::Action(forged.clone()))),
            ControlFlow::Continue(None)
        );
        assert!(
            events.try_recv().is_none(),
            "a forged action reaches nothing"
        );
        assert_eq!(
            inbox.waitlist.missing_from(&[forged.id]),
            vec![forged.id],
            "and is not mirrored either"
        );

        // The contrast: the same payload from its true author goes through.
        let owned = action_by(honest, 2);
        assert_eq!(
            inbox.on_payload(stamped(origin, None, Wire::Action(owned.clone()))),
            ControlFlow::Continue(None)
        );
        assert!(matches!(events.try_recv(), Some(RemoteEvent::Action(a)) if a.id == owned.id));
    }

    /// The presence quota: a frame past [`PRESENCE_QUEUE`] is dropped rather
    /// than queued, and a recv is what frees a slot — so exactly one more frame
    /// fits per event the engine takes.
    #[test]
    fn presence_past_the_quota_is_dropped_and_a_recv_frees_a_slot() {
        let (inbox, mut events) = setup();
        let origin = endpoint(1);
        let presence = |seq: u64| stamped(origin, None, Wire::Presence(frame(seq)));

        for seq in 0..PRESENCE_QUEUE as u64 {
            assert_eq!(inbox.on_payload(presence(seq)), ControlFlow::Continue(None));
        }
        // Full: the next frame is dropped, not queued behind stale ones.
        assert_eq!(
            inbox.on_payload(presence(1_000)),
            ControlFlow::Continue(None)
        );

        // Taking one event releases exactly one slot...
        assert!(matches!(
            events.try_recv(),
            Some(RemoteEvent::Presence { .. })
        ));
        // ...so one more frame fits, and the one after is dropped again.
        assert_eq!(
            inbox.on_payload(presence(1_001)),
            ControlFlow::Continue(None)
        );
        assert_eq!(
            inbox.on_payload(presence(1_002)),
            ControlFlow::Continue(None)
        );

        let mut received = 1;
        while events.try_recv().is_some() {
            received += 1;
        }
        assert_eq!(
            received,
            PRESENCE_QUEUE + 1,
            "the two dropped frames never reached the queue"
        );
    }

    /// Admission routing: an action naming content this peer lacks parks —
    /// nothing surfaces, the fetch is handed back to the caller — and the
    /// content landing is what releases it, asset first. One that names
    /// nothing passes straight through.
    #[test]
    fn an_action_parks_behind_its_content_and_a_plain_one_passes() {
        let (inbox, mut events) = setup();
        let origin = endpoint(1);
        let actor = actor_from_endpoint_id(origin);
        let needing = action_needing(actor, 1, 7);
        let bytes = Bytes::from_static(b"a brush image");
        let hash = Hash::new(&bytes);

        let flow = inbox.on_payload(stamped(origin, Some(hash), Wire::Action(needing.clone())));
        assert_eq!(
            flow,
            ControlFlow::Continue(Some(MustFetch {
                need: AssetNeed::Brush(AssetId([7; 32])),
                hash,
            })),
            "the caller is told to start the fetch"
        );
        assert!(events.try_recv().is_none(), "parked, not surfaced");
        assert!(
            inbox.waitlist.missing_from(&[needing.id]).is_empty(),
            "parked counts as held, so a sweep does not re-claim it"
        );

        inbox
            .waitlist
            .resolved(AssetNeed::Brush(AssetId([7; 32])), bytes, hash);
        assert!(matches!(events.try_recv(), Some(RemoteEvent::Asset { .. })));
        assert!(matches!(events.try_recv(), Some(RemoteEvent::Action(a)) if a.id == needing.id));

        // An action that references nothing needs no fetch and waits on nothing.
        let plain = action_by(actor, 2);
        assert_eq!(
            inbox.on_payload(stamped(origin, None, Wire::Action(plain.clone()))),
            ControlFlow::Continue(None)
        );
        assert!(matches!(events.try_recv(), Some(RemoteEvent::Action(a)) if a.id == plain.id));
    }
}
