//! Brush-shape replication (§6.6, §12.4): content named by a stroke reaches
//! whoever needs it — off the author, off an intermediary, or not at all when
//! nothing was ever registered.
//!
//! Over real iroh endpoints on loopback. Only the mid-session test builds
//! engines: it proves the fetched shape *renders*, where the other two assert
//! wire facts — what arrives, what is fetched — and need no pixels for it.

mod util;

use std::time::Duration;

use stark_model::DocumentFile;
use stark_model::document::{BrushParams, BrushShape};
use stark_model::geom::Vec2;
use stark_model::peer::{GestureFrame, PeerFrame, StrokeHead};
use stark_net::{AssetNeed, CollabSession, Joined, NetOptions, RemoteEvent};

use util::{
    blob_png, drain_events, engine_or_skip, flush_outbox, host_session, identical, join_session,
    next_matching, paint_with, stroke_naming, ticket_of, wait_for_actions,
};

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

    let (host_session, _host_events, ticket) = host_session(&mut host).await;
    let (peer_session, mut peer_events) =
        join_session(&mut peer, &ticket, NetOptions::local()).await;

    // --- live-preview path: a stroke head names a shape the peer lacks ---
    let live = host.import_brush(&blob_png(96)).expect("import live shape");
    host_session.broadcaster().add_content(
        AssetNeed::Brush(live),
        host.asset_bytes(live).expect("canonical bytes"),
    );
    let layer = host.observe().active_layer;
    let brush = BrushParams {
        size: 24.0,
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
                head: Some(Box::new(StrokeHead {
                    layer,
                    brush,
                    seed: 7,
                    translation: stark_model::geom::IVec2::ZERO,
                })),
                from: 0,
                points: vec![],
                start: 0.0,
            }),
            leaving: false,
        })
        .await
        .expect("publish presence");

    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    while !peer.has_asset(live) {
        assert!(
            tokio::time::Instant::now() < deadline,
            "presence-referenced shape never arrived"
        );
        drain_events(&mut peer_events, &mut peer);
        tokio::time::sleep(Duration::from_millis(25)).await;
    }

    // --- commit path: a stroke with a second, never-previewed shape ---
    let committed = host
        .import_brush(&blob_png(64))
        .expect("import committed shape");
    host_session.broadcaster().add_content(
        AssetNeed::Brush(committed),
        host.asset_bytes(committed).expect("canonical bytes"),
    );
    paint_with(
        &mut host,
        BrushParams {
            effect: stark_model::document::BrushEffect::painted([0.8, 0.2, 0.1]),
            size: 20.0,
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

/// **An unregistered shape must not strand the stroke that names it.**
///
/// `add_content` is documented to precede the commit that references its content,
/// and the sender reports the violation (it is the only end that can tell one
/// from content merely still in flight). What the *receiver* does with it is this
/// test: the action arrives with no transfer hash, so there is nothing to fetch,
/// and parking it would be parking forever. It applies at once — the same
/// degradation an unreachable brush gets, reached without a single dial.
///
/// A wire fact, so no engine: the stroke is hand-built naming a shape nobody
/// registered — the call `stark-dioxus-frontend` makes beside every import is the
/// one being skipped — and what is asserted is that the action arrives and nothing
/// is fetched for it.
#[tokio::test(flavor = "multi_thread")]
async fn a_stroke_whose_shape_was_never_registered_still_arrives() {
    let (host, _host_events) =
        CollabSession::host(DocumentFile::new(Vec::new()), NetOptions::local())
            .await
            .expect("host session");
    let Joined {
        session: peer,
        events: mut peer_events,
        ..
    } = CollabSession::join(&ticket_of(&host).await, NetOptions::local())
        .await
        .expect("join session");

    // A stroke naming a shape the session was never handed bytes for.
    let orphan = stark_model::AssetId([0xAB; 32]);
    let stroke = stroke_naming(host.actor_id(), 1, orphan);
    host.broadcaster()
        .broadcast(stroke.clone())
        .expect("broadcast");

    // The point of the test is that this arrives at all. Before parking, an
    // action with no hash was applied inline; the hazard parking introduced is
    // that a need with nothing to fetch waits for a resolver nobody started.
    let arrived = next_matching(&mut peer_events, |event| match event {
        RemoteEvent::Action(action) => Some(action),
        RemoteEvent::Asset { need, .. } => {
            panic!("nothing registered the bytes, yet {need:?} was fetched")
        }
        _ => None,
    })
    .await;
    assert_eq!(arrived.id, stroke.id);

    host.shutdown().await;
    peer.shutdown().await;
}

/// **Content replicates through an intermediary, not just from its author.**
///
/// A peer that fetched a shape records the hash it transferred under, which is
/// what lets it both serve the bytes onward and *announce* them on its own
/// actions. Without that a swarm would only ever be as good as its original
/// author's reachability — and the two-peer tests cannot tell the difference,
/// because there the author is always the deliverer.
///
/// No engine: the shape's id is minted CPU-side the way the frontend's
/// build-time manifest mints one, and arrival is asserted on the wire — the
/// [`RemoteEvent::Asset`] bytes at the intermediary, and the snapshot bundle a
/// newcomer joining *through* it is served.
#[tokio::test(flavor = "multi_thread")]
async fn a_shape_reaches_a_peer_that_joined_through_an_intermediary() {
    let (host, _host_events) =
        CollabSession::host(DocumentFile::new(Vec::new()), NetOptions::local())
            .await
            .expect("host session");
    let Joined {
        session: middle,
        events: mut middle_events,
        ..
    } = CollabSession::join(&ticket_of(&host).await, NetOptions::local())
        .await
        .expect("middle joins the host");

    // The host paints with a shape registered mid-session; `middle` fetches it.
    let canonical = stark_assetid::coverage(&blob_png(72)).expect("decode the shape");
    let shape = canonical.id();
    let bytes = canonical.encode().expect("canonical bytes");
    host.broadcaster()
        .add_content(AssetNeed::Brush(shape), bytes.clone());
    let stroke = stroke_naming(host.actor_id(), 1, shape);
    host.broadcaster()
        .broadcast(stroke.clone())
        .expect("broadcast");

    // The intermediary fetched the shape — and got the bytes, not just a name —
    // before the stroke that waited on it was released.
    let fetched = next_matching(&mut middle_events, |event| match event {
        RemoteEvent::Asset {
            need: AssetNeed::Brush(id),
            bytes,
        } if id == shape => Some(bytes),
        _ => None,
    })
    .await;
    assert_eq!(fetched.as_ref(), &bytes[..], "the canonical bytes arrived");
    let arrived = next_matching(&mut middle_events, |event| match event {
        RemoteEvent::Action(action) => Some(action),
        _ => None,
    })
    .await;
    assert_eq!(arrived.id, stroke.id);

    // A newcomer joins through the *intermediary*, whose snapshot must carry the
    // shape it fetched rather than only the ones it was handed at join time.
    let far = CollabSession::join(&ticket_of(&middle).await, NetOptions::local())
        .await
        .expect("far joins through the intermediary");
    assert!(
        far.document
            .content
            .iter()
            .any(|(need, held)| *need == AssetNeed::Brush(shape) && held == &bytes),
        "a shape the intermediary fetched must reach whoever joins through it"
    );
    assert!(
        far.document.actions.iter().any(|a| a.id == stroke.id),
        "the stroke that named it is in the newcomer's log"
    );

    host.shutdown().await;
    middle.shutdown().await;
    far.session.shutdown().await;
}
