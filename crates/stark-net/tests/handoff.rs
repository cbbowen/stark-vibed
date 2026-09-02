//! A session must outlive the peer that started it (§12.4: "every
//! member can hand out a ticket").
//!
//! These run over real iroh endpoints on loopback but need no GPU: they work on
//! the action log and the catch-up protocol directly.

mod util;

use std::time::Duration;

use stark_model::DocumentFile;
use stark_model::document::Action;
use stark_net::{CollabSession, Events, Joined, NetOptions, RemoteEvent};

use util::{action, next_matching, ticket_of};

/// Wait (bounded) for one remote action to arrive.
async fn next_action(events: &mut Events) -> Action {
    next_matching(events, |event| match event {
        RemoteEvent::Action(action) => Some(action),
        _ => None,
    })
    .await
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
    host.broadcaster()
        .broadcast(marker.clone())
        .expect("broadcast");

    // A second peer joins the founder, the normal way.
    let Joined {
        session: peer,
        events: mut peer_events,
        document: doc,
        ..
    } = CollabSession::join(&ticket_of(&host).await, NetOptions::local())
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
    } = CollabSession::join(&ticket_of(&peer).await, NetOptions::local())
        .await
        .expect("join via a remaining member after the founder left");
    assert!(
        newcomer_doc.actions.iter().any(|a| a.id == marker.id),
        "newcomer should have caught up from the remaining member"
    );

    // And the newcomer is a full participant, not just a reader.
    let fresh = action(newcomer.actor_id(), 2);
    newcomer
        .broadcaster()
        .broadcast(fresh.clone())
        .expect("broadcast");
    assert_eq!(next_action(&mut peer_events).await.id, fresh.id);

    newcomer.shutdown().await;
    peer.shutdown().await;
}

/// The other half of surviving the founder: surviving the **minter**. A link
/// names the members its minter could vouch were alive as well as the minter
/// itself, so a pasted link keeps working after the peer that handed it out has
/// left — the joiner walks past the dead name to a live one.
#[tokio::test(flavor = "multi_thread")]
async fn a_link_outlives_its_minter() {
    let (host, mut host_events) =
        CollabSession::host(DocumentFile::new(Vec::new()), NetOptions::local())
            .await
            .expect("host session");
    let host_id = ticket_of(&host).await.members[0].id;
    let marker = action(host.actor_id(), 1);
    host.broadcaster()
        .broadcast(marker.clone())
        .expect("broadcast");

    let Joined { session: peer, .. } =
        CollabSession::join(&ticket_of(&host).await, NetOptions::local())
            .await
            .expect("join via founder");

    // The peer's link vouches for the founder as well as itself. Minting reads
    // the live neighbor set; the join has met the founder already, but give the
    // set a moment rather than assuming it.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    let link = loop {
        let link = ticket_of(&peer).await;
        if link.members.len() >= 2 {
            break link;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the peer's link never named the founder"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    assert_ne!(link.members[0].id, host_id, "a link names its minter first");
    assert!(
        link.members.iter().any(|m| m.id == host_id),
        "a link vouches for the members its minter is connected to"
    );

    // The minter leaves before the link is ever used.
    peer.shutdown().await;

    let Joined {
        session: newcomer,
        document: newcomer_doc,
        ..
    } = CollabSession::join(&link, NetOptions::local())
        .await
        .expect("join through the second member the link names");
    assert!(
        newcomer_doc.actions.iter().any(|a| a.id == marker.id),
        "the newcomer should have caught up through the surviving member"
    );

    // And is in the live swarm through that member, not just snapshotted.
    let fresh = action(newcomer.actor_id(), 2);
    newcomer
        .broadcaster()
        .broadcast(fresh.clone())
        .expect("broadcast");
    assert_eq!(next_action(&mut host_events).await.id, fresh.id);

    newcomer.shutdown().await;
    host.shutdown().await;
}
