//! The shared harness for stark-net's integration tests: session setup over
//! loopback iroh, the bounded event waits, and the engine-side pump helpers
//! (the UI pump, §12.4). A directory module so cargo does not take it for a
//! test binary of its own.

#![expect(dead_code, reason = "not every test binary uses every helper")]

use std::time::Duration;

use stark_engine::command::{GestureCommand, InputSample, Tool, ViewCommand};
use stark_engine::path::DEFAULT_TOLERANCE;
use stark_engine::{Engine, RgbaImage, headless_engine};
use stark_model::document::{
    Action, ActionId, ActionKind, ActorId, BrushParams, BrushShape, LayerId, StrokeRecord,
};
use stark_model::geom::{Extent2, IVec2, Vec2};
use stark_model::{AssetId, SubstrateId};
use stark_net::{AssetNeed, CollabSession, Events, Joined, NetOptions, RemoteEvent, SessionTicket};

pub const SIZE: Extent2 = Extent2 {
    width: 256,
    height: 256,
};
const TARGET: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// A headless engine, or `None` under `STARK_ALLOW_NO_GPU=1` — a missing GPU is
/// otherwise a failure, for the reason CLAUDE.md gives: a skipped test reports `ok`.
pub fn engine_or_skip() -> Option<Engine> {
    stark_engine::testing::or_skip(
        pollster::block_on(headless_engine(TARGET, SIZE)),
        "a stark-net integration test",
    )
}

/// A cheap, uniquely identifiable action — the content is irrelevant, only that
/// it propagates.
pub fn action(actor: ActorId, lamport: u64) -> Action {
    Action {
        id: ActionId { lamport, actor },
        kind: ActionKind::SetLayerVisible(LayerId::solo(1), true),
    }
}

/// A committed stroke naming `shape` — what a session broadcasts when a client
/// paints with a custom brush, minus the client.
pub fn stroke_naming(actor: ActorId, lamport: u64, shape: AssetId) -> Action {
    Action {
        id: ActionId { lamport, actor },
        kind: ActionKind::CommitStroke(StrokeRecord {
            layer: LayerId::ROOT,
            brush: BrushParams {
                shape: BrushShape::Stamp(shape),
                ..BrushParams::default()
            },
            path: Vec::new(),
            seed: 0,
            start: 0.0,
            translation: IVec2::ZERO,
        }),
    }
}

/// The session's link, round-tripped through its pasted form — the way a
/// joiner actually gets one.
pub async fn ticket_of(session: &CollabSession) -> SessionTicket {
    session
        .broadcaster()
        .ticket()
        .await
        .to_string()
        .parse()
        .expect("ticket round-trips")
}

/// Wait (bounded, ~20 s) for the first event `f` claims — events it declines
/// are dropped. `f` may also panic on an event that must not arrive first.
pub async fn next_matching<T>(
    events: &mut Events,
    mut f: impl FnMut(RemoteEvent) -> Option<T>,
) -> T {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    loop {
        match tokio::time::timeout_at(deadline, events.recv()).await {
            Ok(Some(event)) => {
                if let Some(found) = f(event) {
                    return found;
                }
            }
            Ok(None) => panic!("event stream ended"),
            Err(_) => panic!("timed out waiting for a matching event"),
        }
    }
}

/// Share `engine`'s document over loopback iroh: the identity is minted first,
/// so the engine collaborates under the actor id the session broadcasts as,
/// and the ticket comes back parsed, the way a joiner gets it.
pub async fn host_session(engine: &mut Engine) -> (CollabSession, Events, SessionTicket) {
    let secret = stark_net::SecretKey::generate();
    engine.start_collaboration(stark_net::actor_from_endpoint_id(secret.public()));
    let (session, events) = CollabSession::host(
        engine.document_file(),
        NetOptions {
            secret: Some(secret),
            local_only: true,
            ..Default::default()
        },
    )
    .await
    .expect("host session");
    let ticket = ticket_of(&session).await;
    (session, events, ticket)
}

/// Join through `ticket` and replay the snapshot into `engine` — the whole
/// joiner preamble. A test that must look at the raw [`Joined`] first (its
/// `owed`, or the snapshot before replay) takes [`join_only`] instead.
pub async fn join_session(
    engine: &mut Engine,
    ticket: &SessionTicket,
    opts: NetOptions,
) -> (CollabSession, Events) {
    let Joined {
        session,
        events,
        document,
        ..
    } = join_only(ticket, opts).await;
    engine
        .join_collaboration(&document, session.actor_id())
        .expect("join a session this build can render");
    (session, events)
}

/// [`CollabSession::join`] with the expectation attached — the un-wrapped half
/// of [`join_session`], for tests that settle `owed` by hand before replaying.
pub async fn join_only(ticket: &SessionTicket, opts: NetOptions) -> Joined {
    CollabSession::join(ticket, opts)
        .await
        .expect("join session")
}

/// Local options that promise this client can produce `ids` without the network
/// — what the frontend passes from its build-time manifest of bundled assets.
pub fn promising(ids: &[AssetId]) -> NetOptions {
    NetOptions {
        resolvable: ids.to_vec(),
        ..NetOptions::local()
    }
}

pub fn paint(engine: &mut Engine, color: [f32; 3], points: &[Vec2]) {
    let brush = BrushParams {
        effect: stark_model::document::BrushEffect::painted(color),
        size: 12.0,
        ..Default::default()
    };
    paint_with(engine, brush, points);
}

pub fn paint_with(engine: &mut Engine, brush: BrushParams, points: &[Vec2]) {
    engine.process(ViewCommand::set_brush(brush));
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

pub fn identical(a: &RgbaImage, b: &RgbaImage) -> bool {
    a.width == b.width && a.height == b.height && a.pixels == b.pixels
}

/// Encode a grayscale radial-blob PNG — a stand-in for a user's custom brush
/// shape. `size` varies the pixels, so different sizes give different ids.
pub fn blob_png(size: u32) -> Vec<u8> {
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

/// Apply every queued remote event to the engine (the UI pump, §12.4).
pub fn drain_events(events: &mut Events, engine: &mut Engine) -> usize {
    let mut applied = 0;
    while let Some(event) = events.try_recv() {
        match event {
            RemoteEvent::Asset { need, bytes } => match need {
                AssetNeed::Brush(_) => {
                    engine.import_brush(&bytes).expect("import remote brush");
                }
                AssetNeed::Substrate(id) => {
                    engine
                        .accept_substrate(SubstrateId::Image(id), &bytes)
                        .expect("install remote substrate");
                }
                AssetNeed::Picture(id) => {
                    engine
                        .accept_picture(id, &bytes)
                        .expect("install remote picture");
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
pub async fn flush_outbox(engine: &mut Engine, session: &CollabSession) {
    let tx = session.broadcaster();
    for action in engine.take_outbox() {
        tx.broadcast(action).expect("broadcast");
    }
}

/// Wait (bounded) until `engine` has applied `n` more remote actions.
pub async fn wait_for_actions(events: &mut Events, engine: &mut Engine, mut n: usize) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    while n > 0 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "timed out waiting for {n} remote action(s)"
        );
        n = n.saturating_sub(drain_events(events, engine));
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}
