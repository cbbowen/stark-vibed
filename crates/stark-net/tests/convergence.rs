//! End-to-end: two engines converge over *real* iroh endpoints on loopback
//! (§12.4, build-order step 12) — no relays, no external services.
//! This is the wire-level counterpart of stark-engine's `tests/collab.rs`.

mod util;

use stark_engine::command::DocCommand;
use stark_model::geom::Vec2;
use stark_net::NetOptions;

use util::{
    engine_or_skip, flush_outbox, host_session, identical, join_session, paint, wait_for_actions,
};

/// Host paints, shares; a peer joins over loopback iroh, both paint
/// concurrently, and after the gossip settles the canvases are identical.
/// Also exercises join-time catch-up (the pre-share stroke) and undo across
/// the wire.
#[tokio::test(flavor = "multi_thread")]
async fn two_peers_converge_over_iroh() {
    let (Some(mut host), Some(mut peer)) = (engine_or_skip(), engine_or_skip()) else {
        return;
    };

    // --- host side: paint before sharing, then share ---
    paint(
        &mut host,
        [0.9, 0.1, 0.1],
        &[Vec2::new(40.0, 60.0), Vec2::new(216.0, 60.0)],
    );
    let (host_session, mut host_events, ticket) = host_session(&mut host).await;

    // --- peer side: join, catch up ---
    let (peer_session, mut peer_events) =
        join_session(&mut peer, &ticket, NetOptions::local()).await;

    // The pre-share stroke arrived via the snapshot.
    assert!(identical(&host.render_to_image(), &peer.render_to_image()));

    // --- concurrent edits, crossing on the canvas ---
    paint(
        &mut host,
        [0.1, 0.8, 0.2],
        &[Vec2::new(40.0, 128.0), Vec2::new(216.0, 128.0)],
    );
    paint(
        &mut peer,
        [0.1, 0.2, 0.9],
        &[Vec2::new(128.0, 40.0), Vec2::new(128.0, 216.0)],
    );
    flush_outbox(&mut host, &host_session).await;
    flush_outbox(&mut peer, &peer_session).await;

    wait_for_actions(&mut host_events, &mut host, 1).await;
    wait_for_actions(&mut peer_events, &mut peer, 1).await;
    assert!(
        identical(&host.render_to_image(), &peer.render_to_image()),
        "peers diverged after concurrent strokes over iroh"
    );

    // --- undo across the wire: host undoes its green stroke ---
    host.process(DocCommand::Undo);
    flush_outbox(&mut host, &host_session).await;
    wait_for_actions(&mut peer_events, &mut peer, 1).await;
    assert!(
        identical(&host.render_to_image(), &peer.render_to_image()),
        "peers diverged after undo over iroh"
    );

    host_session.shutdown().await;
    peer_session.shutdown().await;
}
