//! End-to-end: two engines converge over *real* iroh endpoints on loopback
//! (§12.4, build-order step 12) — no relays, no external services.
//! This is the wire-level counterpart of stark-engine's `tests/collab.rs`.

use std::time::Duration;

use stark_engine::command::{DocCommand, GestureCommand, InputSample, ViewCommand};
use stark_engine::engine::headless_engine;
use stark_engine::path::DEFAULT_TOLERANCE;
use stark_engine::peer::{GestureFrame, PeerFrame, StrokeHead};
use stark_engine::{Engine, RgbaImage};
use stark_model::SurfaceId;
use stark_model::document::{BrushParams, BrushShape, Tool};
use stark_model::geom::{Extent2, Vec2};
use stark_net::{AssetNeed, CollabSession, Events, Joined, NetOptions, RemoteEvent, SessionTicket};

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
        rope: 0.0,
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

/// Local options that promise this client can produce `ids` without the network
/// — what the frontend passes from its build-time manifest of bundled assets.
fn promising(ids: &[stark_model::AssetId]) -> NetOptions {
    NetOptions {
        resolvable: ids.to_vec(),
        ..NetOptions::local()
    }
}

/// Apply every queued remote event to the engine (the UI pump, §12.4).
fn drain_events(events: &mut Events, engine: &mut Engine) -> usize {
    let mut applied = 0;
    while let Some(event) = events.try_recv() {
        match event {
            RemoteEvent::Asset { need, bytes } => match need {
                AssetNeed::Brush(_) => {
                    engine.import_brush(&bytes).expect("import remote brush");
                }
                AssetNeed::Ground(id) => {
                    engine
                        .accept_surface(SurfaceId::Image(id), &bytes)
                        .expect("install remote ground");
                }
            },
            RemoteEvent::Action(action) => {
                engine.merge_remote(action);
                applied += 1;
            }
            RemoteEvent::Presence { actor, frame } => {
                // A fixed clock is fine here: nothing in these tests waits out an
                // expiry, and 0.0 keeps them deterministic.
                engine.merge_presence(actor, frame, 0.0);
            }
            // These tests promise nothing they do not then supply by hand, so
            // nothing should be asking. Ignoring it is safe either way — the
            // transport falls back to a peer — but it would mean the test is
            // measuring the fallback rather than what it says it measures.
            RemoteEvent::ResolveLocally { need } => {
                panic!("unexpected local-resolution request for {need:?}")
            }
        }
    }
    applied
}

/// Broadcast everything the engine committed since the last pump.
async fn flush_outbox(engine: &mut Engine, session: &CollabSession) {
    for action in engine.take_outbox() {
        session.broadcast(action).expect("broadcast");
    }
}

/// Wait (bounded) until `engine` has applied `n` more remote actions.
async fn wait_for_actions(events: &mut Events, engine: &mut Engine, mut n: usize) {
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

    let (host_session, mut host_events) = CollabSession::host(
        host.document_file(),
        NetOptions {
            secret: Some(secret),
            local_only: true,
            ..Default::default()
        },
    )
    .await
    .expect("host session");
    let ticket: SessionTicket = host_session
        .ticket()
        .to_string()
        .parse()
        .expect("ticket text");

    // --- peer side: join, catch up ---
    let Joined {
        session: peer_session,
        events: mut peer_events,
        document: snapshot,
        ..
    } = CollabSession::join(&ticket, NetOptions::local())
        .await
        .expect("join session");
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
    let (host_session, _host_events) = CollabSession::host(
        host.document_file(),
        NetOptions {
            secret: Some(secret),
            local_only: true,
            ..Default::default()
        },
    )
    .await
    .expect("host session");
    let ticket: SessionTicket = host_session
        .ticket()
        .to_string()
        .parse()
        .expect("ticket text");

    let Joined {
        session: peer_session,
        events: mut peer_events,
        document: snapshot,
        ..
    } = CollabSession::join(&ticket, NetOptions::local())
        .await
        .expect("join session");
    peer.join_collaboration(&snapshot, peer_session.actor_id());

    // --- live-preview path: a stroke head names a shape the peer lacks ---
    let live = host.import_brush(&blob_png(96)).expect("import live shape");
    host_session.add_content(
        AssetNeed::Brush(live),
        host.asset_bytes(live).expect("canonical bytes"),
    );
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
    host_session.add_content(
        AssetNeed::Brush(committed),
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

/// **A peer that has never seen a ground still paints on it** (§6.4, §12.4).
///
/// What content-addressing the ground buys. Name a ground instead, and the name is
/// only as good as the table the reader holds: the host switches to one the peer has
/// never fetched, the peer's registry silently falls back to the flat stand-in, and
/// from then on every stroke it merges deposits with no deposition tooth. The canvases
/// diverge with nothing on either screen to say why — the peer's pixels are a
/// perfectly plausible painting, just not the same one.
///
/// It has to be a *toothed* brush on an irregular ground, because that is the only
/// thing the fallback changes: with `tooth: 0.0` the gate is 1.0 everywhere and a
/// missing ground is invisible.
#[tokio::test(flavor = "multi_thread")]
async fn a_peer_paints_on_a_ground_it_has_never_seen() {
    let (Some(mut host), Some(mut peer)) = (engine_or_skip(), engine_or_skip()) else {
        return;
    };
    let secret = stark_net::SecretKey::generate();
    let host_actor = stark_net::actor_from_endpoint_id(secret.public());
    host.start_collaboration(host_actor);
    let (host_session, _host_events) = CollabSession::host(
        host.document_file(),
        NetOptions {
            secret: Some(secret),
            local_only: true,
            ..Default::default()
        },
    )
    .await
    .expect("host session");
    let ticket: SessionTicket = host_session
        .ticket()
        .to_string()
        .parse()
        .expect("ticket text");
    let Joined {
        session: peer_session,
        events: mut peer_events,
        document: snapshot,
        ..
    } = CollabSession::join(&ticket, NetOptions::local())
        .await
        .expect("join session");
    peer.join_collaboration(&snapshot, peer_session.actor_id());

    // The host takes up a ground mid-session. The peer has never held these bytes:
    // it joined a document that was on `Flat`, and nothing has offered it since.
    let gesso = host
        .import_surface(&stark_testdata::assets::gesso())
        .expect("the gesso height map imports");
    host_session.add_content(
        AssetNeed::ground(gesso).expect("the gesso ground is an image"),
        host.surface_bytes(gesso).expect("canonical bytes"),
    );
    host.process(DocCommand::SetSurface(gesso));
    flush_outbox(&mut host, &host_session).await;
    wait_for_actions(&mut peer_events, &mut peer, 1).await;

    assert_eq!(
        peer.surface(),
        gesso,
        "the peer's document must move to the host's ground"
    );

    // A dry brush: it reaches only for the peaks, so its mark *is* the ground.
    let dry = BrushParams {
        color: [0.85, 0.15, 0.1, 1.0],
        radius: 30.0,
        tooth: 0.55,
        drain: 0.005,
        ..Default::default()
    };
    paint_with(
        &mut host,
        dry,
        &[Vec2::new(40.0, 128.0), Vec2::new(216.0, 128.0)],
    );
    flush_outbox(&mut host, &host_session).await;
    wait_for_actions(&mut peer_events, &mut peer, 1).await;

    assert!(
        identical(&host.render_to_image(), &peer.render_to_image()),
        "peers diverged over a ground the peer had never seen — the stroke deposited \
         through a different tooth on each"
    );

    host_session.shutdown().await;
    peer_session.shutdown().await;
}

/// **An unregistered shape must not strand the stroke that names it.**
///
/// `add_content` is documented to precede the commit that references its content,
/// and the sender now reports the violation (it is the only end that can tell one
/// from content merely still in flight). What the *receiver* does with it is this
/// test: the action arrives with no transfer hash, so there is nothing to fetch,
/// and parking it would be parking forever. It applies at once and draws with the
/// round tip — the same degradation an unreachable brush gets, reached without a
/// single dial.
#[tokio::test(flavor = "multi_thread")]
async fn a_stroke_whose_shape_was_never_registered_still_arrives() {
    let (Some(mut host), Some(mut peer)) = (engine_or_skip(), engine_or_skip()) else {
        return;
    };

    let secret = stark_net::SecretKey::generate();
    host.start_collaboration(stark_net::actor_from_endpoint_id(secret.public()));
    let (host_session, _host_events) = CollabSession::host(
        host.document_file(),
        NetOptions {
            secret: Some(secret),
            local_only: true,
            ..Default::default()
        },
    )
    .await
    .expect("host session");
    let ticket: SessionTicket = host_session
        .ticket()
        .to_string()
        .parse()
        .expect("ticket text");
    let Joined {
        session: peer_session,
        events: mut peer_events,
        document: snapshot,
        ..
    } = CollabSession::join(&ticket, NetOptions::local())
        .await
        .expect("join session");
    peer.join_collaboration(&snapshot, peer_session.actor_id());

    // Imported into the engine but deliberately NOT registered with the session:
    // the call `stark-ui` makes beside every import is the one being skipped.
    let orphan = host.import_brush(&blob_png(80)).expect("import shape");
    paint_with(
        &mut host,
        BrushParams {
            color: [0.2, 0.4, 0.9, 1.0],
            radius: 22.0,
            shape: BrushShape::Stamp(orphan),
            ..Default::default()
        },
        &[Vec2::new(50.0, 128.0), Vec2::new(206.0, 128.0)],
    );
    flush_outbox(&mut host, &host_session).await;

    // The point of the test is that this returns at all. Before parking, an
    // action with no hash was applied inline; the hazard parking introduced is
    // that a need with nothing to fetch waits for a resolver nobody started.
    wait_for_actions(&mut peer_events, &mut peer, 1).await;
    assert!(
        !peer.has_asset(orphan),
        "nothing registered the bytes, so there was nothing for the peer to fetch"
    );

    host_session.shutdown().await;
    peer_session.shutdown().await;
}

/// **Content replicates through an intermediary, not just from its author.**
///
/// A peer that fetched a shape records the hash it transferred under, which is
/// what lets it both serve the bytes onward and *announce* them on its own
/// actions. Without that a swarm would only ever be as good as its original
/// author's reachability — and the two-peer tests cannot tell the difference,
/// because there the author is always the deliverer.
#[tokio::test(flavor = "multi_thread")]
async fn a_shape_reaches_a_peer_that_joined_through_an_intermediary() {
    let (Some(mut host), Some(mut middle), Some(mut far)) =
        (engine_or_skip(), engine_or_skip(), engine_or_skip())
    else {
        return;
    };

    let secret = stark_net::SecretKey::generate();
    host.start_collaboration(stark_net::actor_from_endpoint_id(secret.public()));
    let (host_session, _host_events) = CollabSession::host(
        host.document_file(),
        NetOptions {
            secret: Some(secret),
            local_only: true,
            ..Default::default()
        },
    )
    .await
    .expect("host session");

    let ticket = |s: &CollabSession| -> SessionTicket {
        s.ticket().to_string().parse().expect("ticket text")
    };
    let Joined {
        session: middle_session,
        events: mut middle_events,
        document: middle_doc,
        ..
    } = CollabSession::join(&ticket(&host_session), NetOptions::local())
        .await
        .expect("middle joins the host");
    middle.join_collaboration(&middle_doc, middle_session.actor_id());

    // The host paints with a shape imported mid-session; `middle` fetches it.
    let shape = host.import_brush(&blob_png(72)).expect("import shape");
    host_session.add_content(
        AssetNeed::Brush(shape),
        host.asset_bytes(shape).expect("canonical bytes"),
    );
    paint_with(
        &mut host,
        BrushParams {
            color: [0.1, 0.7, 0.3, 1.0],
            radius: 18.0,
            shape: BrushShape::Stamp(shape),
            ..Default::default()
        },
        &[Vec2::new(60.0, 120.0), Vec2::new(190.0, 120.0)],
    );
    flush_outbox(&mut host, &host_session).await;
    wait_for_actions(&mut middle_events, &mut middle, 1).await;
    assert!(
        middle.has_asset(shape),
        "the intermediary fetched the shape"
    );

    // A newcomer joins through the *intermediary*, whose snapshot must carry the
    // shape it fetched rather than only the ones it was handed at join time.
    let Joined {
        session: far_session,
        events: mut far_events,
        document: far_doc,
        ..
    } = CollabSession::join(&ticket(&middle_session), NetOptions::local())
        .await
        .expect("far joins through the intermediary");
    far.join_collaboration(&far_doc, far_session.actor_id());
    drain_events(&mut far_events, &mut far);

    assert!(
        far.has_asset(shape),
        "a shape the intermediary fetched must reach whoever joins through it"
    );
    assert!(
        identical(&host.render_to_image(), &far.render_to_image()),
        "the newcomer diverged from the host over a shape it reached second-hand"
    );

    host_session.shutdown().await;
    middle_session.shutdown().await;
    far_session.shutdown().await;
}

/// **A joiner that already has a ground is not sent it — and still replays on it.**
///
/// The whole point of the promise (§12.4). Ground bytes are the biggest thing that
/// moves in this system: the app's own grounds canonicalize to 2.0 and 2.8 MB,
/// against a log that is a handful of fitted paths. Before this, every join pulled
/// a copy over the network into an install that shipped with it.
///
/// The hazard the omission introduces is *replay*, not live painting. The snapshot
/// here already contains a toothed stroke made on the gesso ground, so the joiner
/// has to have those bytes registered **before** `join_collaboration` replays the
/// log — otherwise the stroke re-deposits through the flat stand-in and the result
/// is stored (§6.4). A toothed brush on an irregular ground is the only
/// configuration where that shows up in pixels at all.
#[tokio::test(flavor = "multi_thread")]
async fn a_promised_ground_is_left_out_of_the_snapshot_and_still_replays() {
    let (Some(mut host), Some(mut peer)) = (engine_or_skip(), engine_or_skip()) else {
        return;
    };

    // Painted *before* sharing, so the stroke is in the snapshot's log and the
    // joiner reaches it by replay rather than by gossip.
    let gesso_bytes = stark_testdata::assets::gesso();
    let gesso = host.import_surface(&gesso_bytes).expect("import ground");
    host.process(DocCommand::SetSurface(gesso));
    paint_with(
        &mut host,
        BrushParams {
            color: [0.85, 0.15, 0.1, 1.0],
            radius: 30.0,
            tooth: 0.55,
            drain: 0.005,
            ..Default::default()
        },
        &[Vec2::new(40.0, 128.0), Vec2::new(216.0, 128.0)],
    );

    let secret = stark_net::SecretKey::generate();
    host.start_collaboration(stark_net::actor_from_endpoint_id(secret.public()));
    let (host_session, _host_events) = CollabSession::host(
        host.document_file(),
        NetOptions {
            secret: Some(secret),
            local_only: true,
            ..Default::default()
        },
    )
    .await
    .expect("host session");
    let ticket: SessionTicket = host_session
        .ticket()
        .to_string()
        .parse()
        .expect("ticket text");

    // The joiner says it can resolve the gesso ground itself — which, being a
    // ground that ships with the app, it can.
    let stark_model::SurfaceId::Image(promised) = gesso else {
        panic!("an imported ground is an image");
    };
    let Joined {
        session: peer_session,
        events: _peer_events,
        document: snapshot,
        owed,
    } = CollabSession::join(&ticket, promising(&[promised]))
        .await
        .expect("join session");

    assert!(
        snapshot.surfaces.is_empty(),
        "the host still sent a ground the joiner said it had"
    );
    assert!(
        snapshot.actions.len() >= 2,
        "the snapshot must carry the stroke, or replay has nothing to get wrong"
    );
    assert_eq!(
        owed,
        vec![AssetNeed::Ground(promised)],
        "the omission has to come back as a bill, or the joiner replays without it"
    );

    // Settle the bill the way the frontend does: install, *then* replay.
    for need in &owed {
        let id = need.surface().expect("a ground need names a surface");
        peer.accept_surface(id, &gesso_bytes)
            .expect("install the promised ground");
    }
    peer.join_collaboration(&snapshot, peer_session.actor_id());

    assert!(
        identical(&host.render_to_image(), &peer.render_to_image()),
        "the peer replayed a toothed stroke against a ground it resolved locally and          landed somewhere else — the omission is not sound"
    );

    host_session.shutdown().await;
    peer_session.shutdown().await;
}

/// **Mid-session, a promised ground is asked of the frontend rather than a peer.**
///
/// The join negotiation covers the snapshot; this is the rest of the session. The
/// transport asks the frontend first and only dials if the answer does not come, so a
/// collaborator switching to a ground that ships with the app costs every other client
/// a read from disk rather than a transfer of the canonical weave.
///
/// The peer here plays the frontend by hand: it promises the ground, waits to be
/// asked, and supplies the bytes from its own copy — which is exactly what
/// `stark-ui`'s `supply_locally` does with its bundle. The proof that the local
/// answer was *used* rather than merely offered is in the pixels: the toothed
/// stroke that follows lands identically, which it cannot do against the flat
/// stand-in.
#[tokio::test(flavor = "multi_thread")]
async fn a_promised_ground_is_asked_of_the_frontend_mid_session() {
    let (Some(mut host), Some(mut peer)) = (engine_or_skip(), engine_or_skip()) else {
        return;
    };

    let secret = stark_net::SecretKey::generate();
    host.start_collaboration(stark_net::actor_from_endpoint_id(secret.public()));
    let (host_session, _host_events) = CollabSession::host(
        host.document_file(),
        NetOptions {
            secret: Some(secret),
            local_only: true,
            ..Default::default()
        },
    )
    .await
    .expect("host session");
    let ticket: SessionTicket = host_session
        .ticket()
        .to_string()
        .parse()
        .expect("ticket text");

    // The peer promises the gesso ground before it has any reason to want it.
    let gesso_bytes = stark_testdata::assets::gesso();
    // Derived without a GPU, exactly as `stark-ui`'s build script derives the ids
    // of the assets it bundles — and the `assert_eq!` below is what checks that
    // this route and the engine's `import_surface` agree on the name.
    let probe = stark_assetid::height(&gesso_bytes)
        .expect("decode the ground")
        .id();
    let Joined {
        session: peer_session,
        events: mut peer_events,
        document: snapshot,
        ..
    } = CollabSession::join(&ticket, promising(&[probe]))
        .await
        .expect("join session");
    peer.join_collaboration(&snapshot, peer_session.actor_id());

    // The host takes it up mid-session and paints through a dry, toothed brush,
    // whose mark *is* the ground.
    let gesso = host.import_surface(&gesso_bytes).expect("import ground");
    host_session.add_content(
        AssetNeed::ground(gesso).expect("an image ground"),
        host.surface_bytes(gesso).expect("canonical bytes"),
    );
    host.process(DocCommand::SetSurface(gesso));
    paint_with(
        &mut host,
        BrushParams {
            color: [0.85, 0.15, 0.1, 1.0],
            radius: 30.0,
            tooth: 0.55,
            drain: 0.005,
            ..Default::default()
        },
        &[Vec2::new(40.0, 128.0), Vec2::new(216.0, 128.0)],
    );
    flush_outbox(&mut host, &host_session).await;

    // Play the frontend: wait to be asked, then answer out of our own copy. The
    // stroke is not parked (a round tip references nothing), so it can arrive
    // while we are still waiting — merged as it comes, and counted, because what
    // is left to wait for afterwards depends on how much got through here.
    let mut merged = 0usize;
    let asked = tokio::time::timeout(Duration::from_secs(20), async {
        loop {
            match peer_events.recv().await.expect("event stream ended") {
                RemoteEvent::ResolveLocally { need } => return need,
                RemoteEvent::Asset { need, .. } => {
                    panic!("{need:?} was fetched off a peer instead of asked for")
                }
                RemoteEvent::Action(action) => {
                    peer.merge_remote(action);
                    merged += 1;
                }
                RemoteEvent::Presence { .. } => {}
            }
        }
    })
    .await
    .expect("timed out waiting to be asked for the promised ground");
    assert_eq!(asked, AssetNeed::ground(gesso).expect("an image ground"));

    // Supply it the way `supply_locally` does: into the engine, then the session,
    // which releases the `SetSurface` parked on it.
    peer.accept_surface(asked.surface().expect("a ground"), &gesso_bytes)
        .expect("install locally");
    peer_session.add_content(asked, gesso_bytes.clone());

    // The `SetSurface` and the stroke, less whatever landed before we were asked.
    wait_for_actions(&mut peer_events, &mut peer, 2 - merged).await;
    assert!(
        identical(&host.render_to_image(), &peer.render_to_image()),
        "the peer answered locally but still deposited through a different tooth"
    );

    host_session.shutdown().await;
    peer_session.shutdown().await;
}
