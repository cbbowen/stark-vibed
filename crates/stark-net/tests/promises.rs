//! The substrate on the wire and the local-resolution promise (§6.4, §12.4):
//! a substrate a peer never held still deposits identically, and content a
//! client promises to produce itself is billed rather than transferred.
//!
//! Over real iroh endpoints on loopback. The three substrate tests genuinely
//! need pixels — the deposition tooth is the only witness that the *right*
//! bytes were used — so they build engines; the brush-side promise is a fact
//! about the snapshot and the bill, and needs none.

mod util;

use std::time::Duration;

use stark_engine::command::DocCommand;
use stark_model::DocumentFile;
use stark_model::document::{ActorId, BrushParams};
use stark_model::geom::Vec2;
use stark_net::{AssetNeed, CollabSession, NetOptions, RemoteEvent};

use util::{
    blob_png, engine_or_skip, flush_outbox, host_session, identical, join_only, join_session,
    paint_with, promising, stroke_naming, ticket_of, wait_for_actions,
};

/// **A peer that has never seen a substrate still paints on it** (§6.4, §12.4).
///
/// What content-addressing the substrate buys. Name a substrate instead, and the name is
/// only as good as the table the reader holds: the host switches to one the peer has
/// never fetched, the peer's registry silently falls back to the flat stand-in, and
/// from then on every stroke it merges deposits with no deposition tooth. The canvases
/// diverge with nothing on either screen to say why — the peer's pixels are a
/// perfectly plausible painting, just not the same one.
///
/// It has to be a *toothed* brush on an irregular substrate, because that is the only
/// thing the fallback changes: at full `tooth_give` the gate is 1.0 everywhere and a
/// missing substrate is invisible.
#[tokio::test(flavor = "multi_thread")]
async fn a_peer_paints_on_a_substrate_it_has_never_seen() {
    let (Some(mut host), Some(mut peer)) = (engine_or_skip(), engine_or_skip()) else {
        return;
    };
    let (host_session, _host_events, ticket) = host_session(&mut host).await;
    let (peer_session, mut peer_events) =
        join_session(&mut peer, &ticket, NetOptions::local()).await;

    // The host takes up a substrate mid-session. The peer has never held these bytes:
    // it joined a document that was on `Flat`, and nothing has offered it since.
    let rough = host
        .import_substrate(&stark_testdata::assets::rough())
        .expect("the rough height map imports");
    host_session.broadcaster().add_content(
        AssetNeed::for_substrate(rough).expect("the rough substrate is an image"),
        host.substrate_bytes(rough).expect("canonical bytes"),
    );
    host.process(DocCommand::SetSubstrate(rough));
    flush_outbox(&mut host, &host_session).await;
    wait_for_actions(&mut peer_events, &mut peer, 1).await;

    assert_eq!(
        peer.substrate(),
        rough,
        "the peer's document must move to the host's substrate"
    );

    // A dry brush: it reaches only for the peaks, so its mark *is* the substrate.
    let dry = BrushParams {
        effect: stark_model::document::BrushEffect::painted([0.85, 0.15, 0.1]),
        size: 30.0,
        tooth: stark_model::document::ToothParams {
            give: 0.45,
            ..Default::default()
        },
        drain: 0.15,
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
        "peers diverged over a substrate the peer had never seen — the stroke deposited \
         through a different tooth on each"
    );

    host_session.shutdown().await;
    peer_session.shutdown().await;
}

/// **A joiner that already has a substrate is not sent it — and still replays on it.**
///
/// The whole point of the promise (§12.4). Substrate bytes are the biggest thing that
/// moves in this system: the app's own substrates canonicalize to 2.0 and 2.8 MB,
/// against a log that is a handful of fitted paths. Before this, every join pulled
/// a copy over the network into an install that shipped with it.
///
/// The hazard the omission introduces is *replay*, not live painting. The snapshot
/// here already contains a toothed stroke made on the rough substrate, so the joiner
/// has to have those bytes registered **before** `join_collaboration` replays the
/// log — otherwise the stroke re-deposits through the flat stand-in and the result
/// is stored (§6.4). A toothed brush on an irregular substrate is the only
/// configuration where that shows up in pixels at all.
#[tokio::test(flavor = "multi_thread")]
async fn a_promised_substrate_is_left_out_of_the_snapshot_and_still_replays() {
    let (Some(mut host), Some(mut peer)) = (engine_or_skip(), engine_or_skip()) else {
        return;
    };

    // Painted *before* sharing, so the stroke is in the snapshot's log and the
    // joiner reaches it by replay rather than by gossip.
    let rough_bytes = stark_testdata::assets::rough();
    let rough = host
        .import_substrate(&rough_bytes)
        .expect("import substrate");
    host.process(DocCommand::SetSubstrate(rough));
    paint_with(
        &mut host,
        BrushParams {
            effect: stark_model::document::BrushEffect::painted([0.85, 0.15, 0.1]),
            size: 30.0,
            tooth: stark_model::document::ToothParams {
                give: 0.45,
                ..Default::default()
            },
            drain: 0.15,
            ..Default::default()
        },
        &[Vec2::new(40.0, 128.0), Vec2::new(216.0, 128.0)],
    );
    let (host_session, _host_events, ticket) = host_session(&mut host).await;

    // The joiner says it can resolve the rough substrate itself — which, being a
    // substrate that ships with the app, it can.
    let stark_model::SubstrateId::Image(promised) = rough else {
        panic!("an imported substrate is an image");
    };
    let joined = join_only(&ticket, promising(&[promised])).await;

    assert!(
        !joined
            .document
            .content
            .iter()
            .any(|(need, _)| need.substrate().is_some()),
        "the host still sent a substrate the joiner said it had"
    );
    assert!(
        joined.document.actions.len() >= 2,
        "the snapshot must carry the stroke, or replay has nothing to get wrong"
    );
    assert_eq!(
        joined.owed,
        vec![AssetNeed::Substrate(promised)],
        "the omission has to come back as a bill, or the joiner replays without it"
    );

    // Settle the bill the way the frontend does: install, *then* replay.
    for need in &joined.owed {
        let id = need
            .substrate()
            .expect("a substrate need names a substrate");
        peer.accept_substrate(id, &rough_bytes)
            .expect("install the promised substrate");
    }
    peer.join_collaboration(&joined.document, joined.session.actor_id())
        .expect("join a session this build can render");

    assert!(
        identical(&host.render_to_image(), &peer.render_to_image()),
        "the peer replayed a toothed stroke against a substrate it resolved locally and landed somewhere else — the omission is not sound"
    );

    host_session.shutdown().await;
    joined.session.shutdown().await;
}

/// **Mid-session, a promised substrate is asked of the frontend rather than a peer.**
///
/// The join negotiation covers the snapshot; this is the rest of the session. The
/// transport asks the frontend first and only dials if the answer does not come, so a
/// collaborator switching to a substrate that ships with the app costs every other client
/// a read from disk rather than a transfer of the canonical substrate.
///
/// The peer here plays the frontend by hand: it promises the substrate, waits to be
/// asked, and supplies the bytes from its own copy — which is exactly what
/// `stark-ui`'s `supply_locally` does with its bundle. The proof that the local
/// answer was *used* rather than merely offered is in the pixels: the toothed
/// stroke that follows lands identically, which it cannot do against the flat
/// stand-in.
#[tokio::test(flavor = "multi_thread")]
async fn a_promised_substrate_is_asked_of_the_frontend_mid_session() {
    let (Some(mut host), Some(mut peer)) = (engine_or_skip(), engine_or_skip()) else {
        return;
    };
    let (host_session, _host_events, ticket) = host_session(&mut host).await;

    // The peer promises the rough substrate before it has any reason to want it.
    let rough_bytes = stark_testdata::assets::rough();
    // Derived without a GPU, exactly as `stark-ui`'s build script derives the ids
    // of the assets it bundles — and the `assert_eq!` below is what checks that
    // this route and the engine's `import_substrate` agree on the name.
    let probe = stark_assetid::height(&rough_bytes)
        .expect("decode the substrate")
        .id();
    let (peer_session, mut peer_events) =
        join_session(&mut peer, &ticket, promising(&[probe])).await;

    // The host takes it up mid-session and paints through a dry, toothed brush,
    // whose mark *is* the substrate.
    let rough = host
        .import_substrate(&rough_bytes)
        .expect("import substrate");
    host_session.broadcaster().add_content(
        AssetNeed::for_substrate(rough).expect("an image substrate"),
        host.substrate_bytes(rough).expect("canonical bytes"),
    );
    host.process(DocCommand::SetSubstrate(rough));
    paint_with(
        &mut host,
        BrushParams {
            effect: stark_model::document::BrushEffect::painted([0.85, 0.15, 0.1]),
            size: 30.0,
            tooth: stark_model::document::ToothParams {
                give: 0.45,
                ..Default::default()
            },
            drain: 0.15,
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
    .expect("timed out waiting to be asked for the promised substrate");
    assert_eq!(
        asked,
        AssetNeed::for_substrate(rough).expect("an image substrate")
    );

    // Supply it the way `supply_locally` does: into the engine, then the session,
    // which releases the `SetSubstrate` parked on it.
    peer.accept_substrate(asked.substrate().expect("a substrate"), &rough_bytes)
        .expect("install locally");
    peer_session
        .broadcaster()
        .add_content(asked, rough_bytes.clone());

    // The `SetSubstrate` and the stroke, less whatever landed before we were asked.
    wait_for_actions(&mut peer_events, &mut peer, 2 - merged).await;
    assert!(
        identical(&host.render_to_image(), &peer.render_to_image()),
        "the peer answered locally but still deposited through a different tooth"
    );

    host_session.shutdown().await;
    peer_session.shutdown().await;
}

/// **The promise has a brush arm, and it works the same way** — the snapshot
/// omits a promised shape's bytes, and the omission comes back as a bill.
///
/// All three substrate tests prove their half in pixels; this one needs none,
/// because what it pins is the negotiation: `resolvable` reaches the host as
/// `SnapshotWithout`, the host strips exactly the promised payload while the
/// log keeps naming it, and `Joined::owed` — `unbundled_content` asked of the
/// trimmed file — says a *brush* need, not merely an id.
#[tokio::test(flavor = "multi_thread")]
async fn a_promised_brush_shape_is_left_out_of_the_snapshot_and_billed() {
    // A hosted document whose log names a custom shape and whose bundle
    // carries it — what a session looks like after an import and a stroke.
    let canonical = stark_assetid::coverage(&blob_png(48)).expect("decode the shape");
    let shape = canonical.id();
    let bytes = canonical.encode().expect("canonical bytes");
    let mut file = DocumentFile::new(vec![stroke_naming(ActorId(1), 1, shape)]);
    file.content = vec![(AssetNeed::Brush(shape), bytes)];
    let (host, _host_events) = CollabSession::host(file, NetOptions::local())
        .await
        .expect("host session");

    let joined = join_only(&ticket_of(&host).await, promising(&[shape])).await;

    assert!(
        joined.document.content.is_empty(),
        "the host still sent a shape the joiner said it had"
    );
    assert!(
        joined.document.actions.iter().any(|a| a.id.lamport == 1),
        "the log still names the stroke — only the payload is spared"
    );
    assert_eq!(
        joined.owed,
        vec![AssetNeed::Brush(shape)],
        "the omission has to come back as a brush bill, in the store that owns it"
    );

    host.shutdown().await;
    joined.session.shutdown().await;
}
