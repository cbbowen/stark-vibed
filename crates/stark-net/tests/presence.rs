//! Presence on the wire (§17.4, §8): it reaches peers, it is
//! attributed to its author by the transport rather than by its payload, and it
//! never reaches the document.
//!
//! Over real iroh endpoints on loopback, but with no GPU: the properties under test
//! are all about the wire and the mirror.

use std::time::Duration;

use stark_model::DocumentFile;
use stark_model::document::{Action, ActionId, ActionKind, ActorId, LayerId};
use stark_model::peer::PeerFrame;
use stark_net::{CollabSession, Events, Joined, NetOptions, RemoteEvent, SessionTicket};

async fn ticket_of(session: &CollabSession) -> SessionTicket {
    session
        .broadcaster()
        .ticket()
        .await
        .to_string()
        .parse()
        .expect("ticket round-trips")
}

fn frame(seq: u64, name: &str) -> PeerFrame {
    PeerFrame {
        boot: 0,
        seq,
        name: Some(name.to_string()),
        active_layer: LayerId::ROOT,
        cursor: Some(stark_model::Vec2::new(12.0, 34.0)),
        gesture: None,
        leaving: false,
    }
}

/// Wait (bounded) for one presence frame.
async fn next_presence(events: &mut Events) -> (ActorId, PeerFrame) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    loop {
        match tokio::time::timeout_at(deadline, events.recv()).await {
            Ok(Some(RemoteEvent::Presence { actor, frame })) => return (actor, frame),
            Ok(Some(_)) => continue,
            Ok(None) => panic!("event stream ended"),
            Err(_) => panic!("timed out waiting for presence"),
        }
    }
}

/// Presence crosses the mesh, and the receiver attributes it to the **sender's**
/// identity — taken from the transport's authenticated origin, never from the frame,
/// so a peer can publish its own presence and nobody else's.
#[tokio::test(flavor = "multi_thread")]
async fn presence_reaches_peers_attributed_to_its_sender() {
    let (host, _host_events) =
        CollabSession::host(DocumentFile::new(Vec::new()), NetOptions::local())
            .await
            .expect("host session");
    let Joined {
        session: _peer,
        mut events,
        document: _doc,
        ..
    } = CollabSession::join(&ticket_of(&host).await, NetOptions::local())
        .await
        .expect("join session");

    host.broadcaster()
        .publish(frame(1, "Ada"))
        .await
        .expect("publish presence");

    let (actor, got) = next_presence(&mut events).await;
    assert_eq!(actor, host.actor_id(), "attributed to the sender");
    assert_eq!(got.name.as_deref(), Some("Ada"));
    assert_eq!(got.cursor, Some(stark_model::Vec2::new(12.0, 34.0)));
}

/// Presence is **not part of the document**: it never enters the mirror, so a peer
/// joining afterwards gets a snapshot with nothing of it in it. That one rule is why
/// the save format and the catch-up protocol needed no changes at all.
#[tokio::test(flavor = "multi_thread")]
async fn presence_never_enters_the_snapshot() {
    let (host, _host_events) =
        CollabSession::host(DocumentFile::new(Vec::new()), NetOptions::local())
            .await
            .expect("host session");

    // One real action, so the snapshot is demonstrably non-empty — and a burst of
    // presence around it, which must leave no trace.
    let tx = host.broadcaster();
    for seq in 1..=5 {
        tx.publish(frame(seq, "Ada")).await.expect("publish");
    }
    tx.broadcast(Action {
        id: ActionId {
            lamport: 1,
            actor: host.actor_id(),
        },
        kind: ActionKind::SetLayerVisible(LayerId::ROOT, true),
    })
    .expect("broadcast action");
    for seq in 6..=10 {
        tx.publish(frame(seq, "Ada")).await.expect("publish");
    }

    let Joined {
        session: _peer,
        events: _peer_events,
        document: doc,
        ..
    } = CollabSession::join(&ticket_of(&host).await, NetOptions::local())
        .await
        .expect("join session");
    assert_eq!(
        doc.actions.len(),
        1,
        "the snapshot must carry the action and nothing of the presence: {:?}",
        doc.actions
    );
}
