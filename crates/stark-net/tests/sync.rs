//! End-to-end: two engines converge over *real* iroh endpoints on loopback
//! (DESIGN.md §12.4, build-order step 12) — no relays, no external services.
//! This is the wire-level counterpart of stark-core's `tests/collab.rs`.

use std::time::Duration;

use stark_core::command::{InputCommand, InputSample};
use stark_core::document::Tool;
use stark_core::engine::headless_engine;
use stark_core::geom::{Extent2, Vec2};
use stark_core::{Engine, RgbaImage};
use stark_net::{CollabSession, NetOptions, RemoteEvent, SessionTicket};
use tokio::sync::mpsc::UnboundedReceiver;

const SIZE: Extent2 = Extent2 { width: 256, height: 256 };
const TARGET: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const PAPER: wgpu::Color = wgpu::Color { r: 0.97, g: 0.97, b: 0.97, a: 1.0 };

fn engine_or_skip() -> Option<Engine> {
    match pollster::block_on(headless_engine(TARGET, SIZE)) {
        Ok(e) => Some(e),
        Err(e) => {
            eprintln!("skipping GPU test: {e}");
            None
        }
    }
}

fn paint(engine: &mut Engine, color: [f32; 4], points: &[Vec2]) {
    let brush = stark_core::document::BrushParams {
        color,
        radius: 12.0,
        ..Default::default()
    };
    engine.process(InputCommand::SetBrush(brush));
    let mut it = points.iter();
    engine.process(InputCommand::StartStroke {
        tool: Tool::Brush,
        sample: InputSample::at(*it.next().unwrap()),
    });
    for &p in it {
        engine.process(InputCommand::StrokeTo { sample: InputSample::at(p) });
    }
    engine.process(InputCommand::EndStroke);
}

fn identical(a: &RgbaImage, b: &RgbaImage) -> bool {
    a.width == b.width && a.height == b.height && a.pixels == b.pixels
}

/// Apply every queued remote event to the engine (the UI pump, DESIGN.md §12.4).
fn drain_events(events: &mut UnboundedReceiver<RemoteEvent>, engine: &mut Engine) -> usize {
    let mut applied = 0;
    while let Ok(event) = events.try_recv() {
        match event {
            RemoteEvent::Asset { bytes } => {
                engine.import_brush(&bytes).expect("import remote brush");
            }
            RemoteEvent::Action(action) => {
                engine.merge_remote(action);
                applied += 1;
            }
        }
    }
    applied
}

/// Broadcast everything the engine committed since the last pump.
async fn flush_outbox(engine: &mut Engine, session: &CollabSession) {
    for action in engine.take_outbox() {
        session.broadcast(action).await.expect("broadcast");
    }
}

/// Wait (bounded) until `engine` has applied `n` more remote actions.
async fn wait_for_actions(
    events: &mut UnboundedReceiver<RemoteEvent>,
    engine: &mut Engine,
    mut n: usize,
) {
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    while n > 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "timed out waiting for {n} remote action(s)"
        );
        n = n.saturating_sub(drain_events(events, engine));
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

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
    paint(&mut host, [0.9, 0.1, 0.1, 1.0], &[Vec2::new(40.0, 60.0), Vec2::new(216.0, 60.0)]);

    let secret = stark_net::SecretKey::generate();
    let host_actor = stark_net::actor_from_endpoint_id(secret.public());
    host.start_collaboration(host_actor);

    let mut host_session = CollabSession::host(
        host.document_file(),
        NetOptions { secret: Some(secret), local_only: true },
    )
    .await
    .expect("host session");
    let mut host_events = host_session.take_events().expect("host events");
    let ticket: SessionTicket = host_session.ticket().to_string().parse().expect("ticket text");

    // --- peer side: join, catch up ---
    let (mut peer_session, snapshot) = CollabSession::join(&ticket, NetOptions::local())
        .await
        .expect("join session");
    let mut peer_events = peer_session.take_events().expect("peer events");
    peer.join_collaboration(&snapshot, peer_session.actor_id());

    // The pre-share stroke arrived via the snapshot.
    assert!(identical(
        &host.render_to_image(PAPER),
        &peer.render_to_image(PAPER)
    ));

    // --- concurrent edits, crossing on the canvas ---
    paint(&mut host, [0.1, 0.8, 0.2, 1.0], &[Vec2::new(40.0, 128.0), Vec2::new(216.0, 128.0)]);
    paint(&mut peer, [0.1, 0.2, 0.9, 1.0], &[Vec2::new(128.0, 40.0), Vec2::new(128.0, 216.0)]);
    flush_outbox(&mut host, &host_session).await;
    flush_outbox(&mut peer, &peer_session).await;

    wait_for_actions(&mut host_events, &mut host, 1).await;
    wait_for_actions(&mut peer_events, &mut peer, 1).await;
    assert!(
        identical(&host.render_to_image(PAPER), &peer.render_to_image(PAPER)),
        "peers diverged after concurrent strokes over iroh"
    );

    // --- undo across the wire: host undoes its green stroke ---
    host.process(InputCommand::Undo);
    flush_outbox(&mut host, &host_session).await;
    wait_for_actions(&mut peer_events, &mut peer, 1).await;
    assert!(
        identical(&host.render_to_image(PAPER), &peer.render_to_image(PAPER)),
        "peers diverged after undo over iroh"
    );

    host_session.shutdown().await;
    peer_session.shutdown().await;
}
