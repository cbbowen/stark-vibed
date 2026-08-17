//! A session must outlive the peer that started it (§12.4: "every
//! member can hand out a ticket").
//!
//! These run over real iroh endpoints on loopback but need no GPU: they work on
//! the action log and the catch-up protocol directly.

use std::time::Duration;

use stark_engine::DocumentFile;
use stark_model::document::{Action, ActionId, ActionKind, ActorId, LayerId};
use stark_net::{CollabSession, Events, Joined, NetOptions, RemoteEvent, SessionTicket};

/// A cheap, uniquely identifiable action — the content is irrelevant, only that
/// it propagates.
fn action(actor: ActorId, lamport: u64) -> Action {
    Action {
        id: ActionId { lamport, actor },
        kind: ActionKind::SetLayerVisible(LayerId(1), true),
    }
}

fn ticket_of(session: &CollabSession) -> SessionTicket {
    session
        .ticket()
        .to_string()
        .parse()
        .expect("ticket round-trips")
}

/// Wait (bounded) for one remote action to arrive.
async fn next_action(events: &mut Events) -> Action {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    loop {
        match tokio::time::timeout_at(deadline, events.recv()).await {
            Ok(Some(RemoteEvent::Action(action))) => return action,
            Ok(Some(
                RemoteEvent::Asset { .. }
                | RemoteEvent::Presence { .. }
                | RemoteEvent::ResolveLocally { .. },
            )) => continue,
            Ok(None) => panic!("event stream ended"),
            Err(_) => panic!("timed out waiting for a remote action"),
        }
    }
}

/// The reported failure: with the founder gone, can a newcomer still join
/// through a peer that is still present?
#[tokio::test(flavor = "multi_thread")]
async fn a_newcomer_can_join_through_any_member_after_the_founder_leaves() {
    let (host, _host_events) =
        CollabSession::host(DocumentFile::new(Vec::new()), NetOptions::local())
            .await
            .expect("host session");
    let marker = action(host.actor_id(), 1);
    host.broadcast(marker.clone()).expect("broadcast");

    // A second peer joins the founder, the normal way.
    let Joined {
        session: peer,
        events: mut peer_events,
        document: doc,
        ..
    } = CollabSession::join(&ticket_of(&host), NetOptions::local())
        .await
        .expect("join via founder");
    assert!(
        doc.actions.iter().any(|a| a.id == marker.id),
        "joiner should have caught up through the snapshot"
    );

    // The founder leaves for good.
    host.shutdown().await;

    // A newcomer arrives with a ticket from the *remaining* member.
    let Joined {
        session: newcomer,
        events: _newcomer_events,
        document: newcomer_doc,
        ..
    } = CollabSession::join(&ticket_of(&peer), NetOptions::local())
        .await
        .expect("join via a remaining member after the founder left");
    assert!(
        newcomer_doc.actions.iter().any(|a| a.id == marker.id),
        "newcomer should have caught up from the remaining member"
    );

    // And the newcomer is a full participant, not just a reader.
    let fresh = action(newcomer.actor_id(), 2);
    newcomer.broadcast(fresh.clone()).expect("broadcast");
    assert_eq!(next_action(&mut peer_events).await.id, fresh.id);

    newcomer.shutdown().await;
    peer.shutdown().await;
}
