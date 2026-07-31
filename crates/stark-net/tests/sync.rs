//! End-to-end: two engines converge over *real* iroh endpoints on loopback
//! (§12.4, build-order step 12) — no relays, no external services.
//! This is the wire-level counterpart of stark-core's `tests/collab.rs`.

use std::time::Duration;

use stark_core::command::{DocCommand, GestureCommand, InputSample, ViewCommand};
use stark_core::document::{BrushParams, BrushShape, Tool};
use stark_core::engine::headless_engine;
use stark_core::geom::{Extent2, Vec2};
use stark_core::path::DEFAULT_TOLERANCE;
use stark_core::peer::{GestureFrame, PeerFrame, StrokeHead};
use stark_core::{Engine, RgbaImage};
use stark_net::{CollabSession, NetOptions, RemoteEvent, SessionTicket};
use tokio::sync::mpsc::UnboundedReceiver;

const SIZE: Extent2 = Extent2 {
    width: 256,
    height: 256,
};
const TARGET: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
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
    let brush = BrushParams {
        color,
        radius: 12.0,
        ..Default::default()
    };
    paint_with(engine, brush, points);
}

fn paint_with(engine: &mut Engine, brush: BrushParams, points: &[Vec2]) {
    engine.process(ViewCommand::SetBrush(brush));
    let mut it = points.iter();
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(*it.next().unwrap()),
        tolerance: DEFAULT_TOLERANCE,
    });
    for &p in it {
        engine.process(GestureCommand::To {
            sample: InputSample::at(p),
        });
    }
    engine.process(GestureCommand::End);
}

fn identical(a: &RgbaImage, b: &RgbaImage) -> bool {
    a.width == b.width && a.height == b.height && a.pixels == b.pixels
}

/// Apply every queued remote event to the engine (the UI pump, §12.4).
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
            RemoteEvent::Presence { actor, frame } => {
                // A fixed clock is fine here: nothing in these tests waits out an
                // expiry, and 0.0 keeps them deterministic.
                engine.merge_presence(actor, frame, 0.0);
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
    paint(
        &mut host,
        [0.9, 0.1, 0.1, 1.0],
        &[Vec2::new(40.0, 60.0), Vec2::new(216.0, 60.0)],
    );

    let secret = stark_net::SecretKey::generate();
    let host_actor = stark_net::actor_from_endpoint_id(secret.public());
    host.start_collaboration(host_actor);

    let mut host_session = CollabSession::host(
        host.document_file(),
        NetOptions {
            secret: Some(secret),
            local_only: true,
        },
    )
    .await
    .expect("host session");
    let mut host_events = host_session.take_events().expect("host events");
    let ticket: SessionTicket = host_session
        .ticket()
        .to_string()
        .parse()
        .expect("ticket text");

    // --- peer side: join, catch up ---
    let (mut peer_session, snapshot) = CollabSession::join(&ticket, NetOptions::local())
        .await
        .expect("join session");
    let mut peer_events = peer_session.take_events().expect("peer events");
    peer.join_collaboration(&snapshot, peer_session.actor_id());

    // The pre-share stroke arrived via the snapshot.
    assert!(identical(&host.render_to_image(), &peer.render_to_image()));

    // --- concurrent edits, crossing on the canvas ---
    paint(
        &mut host,
        [0.1, 0.8, 0.2, 1.0],
        &[Vec2::new(40.0, 128.0), Vec2::new(216.0, 128.0)],
    );
    paint(
        &mut peer,
        [0.1, 0.2, 0.9, 1.0],
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

/// Encode a grayscale radial-blob PNG — a stand-in for a user's custom brush
/// shape. `size` varies the pixels, so different sizes give different ids.
fn blob_png(size: u32) -> Vec<u8> {
    let c = size as f32 / 2.0;
    let mut pixels = vec![0u8; (size * size) as usize];
    for y in 0..size {
        for x in 0..size {
            let d = ((x as f32 - c).powi(2) + (y as f32 - c).powi(2)).sqrt() / c;
            pixels[(y * size + x) as usize] = (255.0 * (1.0 - d).clamp(0.0, 1.0)) as u8;
        }
    }
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, size, size);
        encoder.set_color(png::ColorType::Grayscale);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().expect("png header");
        writer.write_image_data(&pixels).expect("png data");
    }
    out
}

/// A custom shape imported *mid-session* reaches the other side by content
/// hash on both wires: a live-preview head triggers a detached background
/// fetch (presence must never block on assets), and a committed stroke with a
/// second, never-previewed shape pulls the blob inline before the action so
/// the render is faithful immediately. The importing side seeds its session
/// mirror the way the UI's import path does.
#[tokio::test(flavor = "multi_thread")]
async fn custom_shapes_replicate_mid_session() {
    let (Some(mut host), Some(mut peer)) = (engine_or_skip(), engine_or_skip()) else {
        return;
    };

    let secret = stark_net::SecretKey::generate();
    let host_actor = stark_net::actor_from_endpoint_id(secret.public());
    host.start_collaboration(host_actor);
    let host_session = CollabSession::host(
        host.document_file(),
        NetOptions {
            secret: Some(secret),
            local_only: true,
        },
    )
    .await
    .expect("host session");
    let ticket: SessionTicket = host_session
        .ticket()
        .to_string()
        .parse()
        .expect("ticket text");

    let (mut peer_session, snapshot) = CollabSession::join(&ticket, NetOptions::local())
        .await
        .expect("join session");
    let mut peer_events = peer_session.take_events().expect("peer events");
    peer.join_collaboration(&snapshot, peer_session.actor_id());

    // --- live-preview path: a stroke head names a shape the peer lacks ---
    let live = host.import_brush(&blob_png(96)).expect("import live shape");
    host_session.add_asset(live, host.asset_bytes(live).expect("canonical bytes"));
    let layer = host.observe().active_layer;
    let brush = BrushParams {
        radius: 24.0,
        shape: BrushShape::Stamp(live),
        ..Default::default()
    };
    host_session
        .broadcaster()
        .publish(PeerFrame {
            boot: 1,
            seq: 1,
            name: None,
            active_layer: layer,
            cursor: Some(Vec2::new(10.0, 10.0)),
            gesture: Some(GestureFrame::Stroke {
                id: 1,
                head: Some(StrokeHead {
                    layer,
                    tool: Tool::Brush,
                    brush,
                    seed: 7,
                }),
                from: 0,
                points: vec![],
            }),
            leaving: false,
        })
        .await
        .expect("publish presence");

    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    while !peer.has_asset(live) {
        assert!(
            std::time::Instant::now() < deadline,
            "presence-referenced shape never arrived"
        );
        drain_events(&mut peer_events, &mut peer);
        tokio::time::sleep(Duration::from_millis(25)).await;
    }

    // --- commit path: a stroke with a second, never-previewed shape ---
    let committed = host
        .import_brush(&blob_png(64))
        .expect("import committed shape");
    host_session.add_asset(
        committed,
        host.asset_bytes(committed).expect("canonical bytes"),
    );
    paint_with(
        &mut host,
        BrushParams {
            color: [0.8, 0.2, 0.1, 1.0],
            radius: 20.0,
            shape: BrushShape::Stamp(committed),
            ..Default::default()
        },
        &[Vec2::new(60.0, 128.0), Vec2::new(196.0, 128.0)],
    );
    flush_outbox(&mut host, &host_session).await;
    wait_for_actions(&mut peer_events, &mut peer, 1).await;

    assert!(
        peer.has_asset(committed),
        "commit-referenced shape must be fetched before the action is applied"
    );
    assert!(
        identical(&host.render_to_image(), &peer.render_to_image()),
        "peers diverged on a mid-session custom-shape stroke"
    );

    host_session.shutdown().await;
    peer_session.shutdown().await;
}
