//! Collaboration semantics, headlessly (§12, build-order step 12):
//! two engines exchanging action logs must converge to identical pixels —
//! convergence as a test — including out-of-order merges, undo-as-an-action,
//! and late joins. No network here; `stark-net` adds the wire on top.

mod common;

use common::collab::{pair, snap, sync, sync_into};
use common::palette::{BLUE_VIVID, GREEN_SOFT, RED_VIVID};
use common::{engine_or_skip, images_match, paint, whole_render};
use stark_engine::command::{DocCommand, PeerCommand};
use stark_model::document::ActorId;
use stark_model::geom::Vec2;

/// Two peers paint overlapping strokes and merge them in *different orders*
/// (B commits its own stroke first, then receives A's earlier-ordered one, so
/// B exercises the rewind-and-replay insert; A appends). Both must render the
/// same pixels, and a peer joining afterwards from the shared log must too.
#[test]
fn concurrent_strokes_converge() {
    let Some((mut a, mut b)) = pair() else {
        return;
    };
    let Some(mut c) = engine_or_skip() else {
        return;
    };

    // Concurrent, overlapping edits: same lamport, actor id breaks the tie, so
    // A's stroke orders before B's even though B applies it second.
    paint(
        &mut a,
        RED_VIVID,
        12.0,
        &[Vec2::new(40.0, 128.0), Vec2::new(216.0, 128.0)],
    );
    paint(
        &mut b,
        GREEN_SOFT,
        12.0,
        &[Vec2::new(128.0, 40.0), Vec2::new(128.0, 216.0)],
    );
    sync(&mut a, &mut b);

    let img_a = snap(&mut a);
    let img_b = snap(&mut b);
    assert!(
        images_match(&img_a, &img_b, 0),
        "peers diverged after cross-merge"
    );

    // A late joiner catching up from the full shared log sees the same canvas.
    c.join_collaboration(&a.document_file(), ActorId(3))
        .expect("join a session this build can render");
    let img_c = snap(&mut c);
    assert!(
        images_match(&img_a, &img_c, 0),
        "late joiner diverged from the session"
    );
}

/// Undo in a shared session is a logged action targeting *my* stroke, skipping
/// the peer's intervening one (§12.3); redo is an undo of that undo.
#[test]
fn shared_undo_skips_peer_actions() {
    let Some((mut a, mut b)) = pair() else {
        return;
    };

    paint(
        &mut a,
        RED_VIVID,
        12.0,
        &[Vec2::new(40.0, 128.0), Vec2::new(216.0, 128.0)],
    );
    // B's canvas before it has seen anything of A: exactly "only B's stroke".
    paint(
        &mut b,
        GREEN_SOFT,
        12.0,
        &[Vec2::new(128.0, 40.0), Vec2::new(128.0, 216.0)],
    );
    let only_b = snap(&mut b);

    sync(&mut a, &mut b);

    // A undoes *its* stroke — B's later stroke must survive.
    a.process(DocCommand::Undo);
    sync(&mut a, &mut b);
    let img_a = snap(&mut a);
    let img_b = snap(&mut b);
    assert!(images_match(&img_a, &img_b, 0), "peers diverged after undo");
    assert!(
        images_match(&img_a, &only_b, 0),
        "undo of A's stroke should leave exactly B's stroke"
    );

    // Redo brings A's stroke back on both peers — at the *top* of the stack
    // (§12.3), over B's stroke rather than back under it, so the
    // crossing no longer matches the pre-undo image. What must hold: the peers
    // agree, a late joiner materializing the log canonically agrees, and the
    // redone stroke is genuinely back (the image changed from "only B").
    a.process(DocCommand::Redo);
    sync(&mut a, &mut b);
    let img_a = snap(&mut a);
    let img_b = snap(&mut b);
    assert!(images_match(&img_a, &img_b, 0), "peers diverged after redo");
    assert!(
        !images_match(&img_a, &only_b, 0),
        "redo should bring A's stroke back"
    );
    let Some(mut c) = engine_or_skip() else {
        return;
    };
    c.join_collaboration(&a.document_file(), ActorId(3))
        .expect("join a session this build can render");
    assert!(
        images_match(&img_a, &snap(&mut c), 0),
        "redo-at-top diverged from a late joiner's canonical materialization"
    );

    // A fresh edit clears the redo stack, like solo editing.
    assert!(!a.observe().can_redo);
}

/// A multi-step undo/redo chain (Z Z Y Y) walks back and forth correctly —
/// the redo-of-an-undo bookkeeping, all through logged actions.
#[test]
fn shared_undo_redo_chain() {
    let Some(mut a) = engine_or_skip() else {
        return;
    };
    a.start_collaboration(ActorId(1));

    let blank = snap(&mut a);
    paint(
        &mut a,
        RED_VIVID,
        12.0,
        &[Vec2::new(40.0, 100.0), Vec2::new(216.0, 100.0)],
    );
    let one = snap(&mut a);
    paint(
        &mut a,
        BLUE_VIVID,
        12.0,
        &[Vec2::new(40.0, 156.0), Vec2::new(216.0, 156.0)],
    );
    let two = snap(&mut a);

    a.process(DocCommand::Undo);
    assert!(images_match(&snap(&mut a), &one, 0), "first undo");
    a.process(DocCommand::Undo);
    assert!(images_match(&snap(&mut a), &blank, 0), "second undo");
    assert!(!a.observe().can_undo, "nothing left to undo");
    a.process(DocCommand::Redo);
    assert!(images_match(&snap(&mut a), &one, 0), "first redo");
    a.process(DocCommand::Redo);
    assert!(images_match(&snap(&mut a), &two, 0), "second redo");
    assert!(!a.observe().can_redo, "nothing left to redo");
}

/// Sharing rewrites solo-authored actions to the sharer's actor id, so the
/// host can still undo strokes painted before the session started.
#[test]
fn host_can_undo_pre_share_strokes() {
    let Some(mut a) = engine_or_skip() else {
        return;
    };
    let blank = snap(&mut a);
    paint(
        &mut a,
        RED_VIVID,
        12.0,
        &[Vec2::new(40.0, 128.0), Vec2::new(216.0, 128.0)],
    );

    a.start_collaboration(ActorId(7));
    assert!(a.observe().can_undo, "pre-share stroke should be undoable");
    a.process(DocCommand::Undo);
    assert!(
        images_match(&snap(&mut a), &blank, 0),
        "undo should remove the pre-share stroke"
    );
}

/// A file saved mid-session carries the full log (undos included); a solo load
/// flattens it to the effective sequence and reproduces the same pixels.
#[test]
fn shared_save_solo_load_roundtrip() {
    let (Some(mut a), Some(mut solo)) = (engine_or_skip(), engine_or_skip()) else {
        return;
    };
    a.start_collaboration(ActorId(1));

    paint(
        &mut a,
        RED_VIVID,
        12.0,
        &[Vec2::new(40.0, 100.0), Vec2::new(216.0, 100.0)],
    );
    paint(
        &mut a,
        GREEN_SOFT,
        12.0,
        &[Vec2::new(40.0, 156.0), Vec2::new(216.0, 156.0)],
    );
    a.process(DocCommand::Undo); // green gone; log still contains it + the undo
    let expected = snap(&mut a);

    let bytes = a.save_bytes().expect("save shared doc");
    solo.load_bytes(&bytes).expect("load shared doc");
    assert!(
        images_match(&snap(&mut solo), &expected, 0),
        "solo load of a shared log should reproduce the session's canvas"
    );
    // The undone stroke was flattened away; the surviving one is undoable.
    assert!(solo.observe().can_undo);
    solo.process(DocCommand::Undo);
    let after_undo = snap(&mut solo);
    let blank = {
        // A fresh document renders bare paper.
        let Some(mut fresh) = engine_or_skip() else {
            return;
        };
        snap(&mut fresh)
    };
    assert!(
        images_match(&after_undo, &blank, 0),
        "undo after load should reach the blank canvas"
    );
}

/// Duplicate deliveries (redelivery by the transport) are idempotent.
#[test]
fn merge_is_idempotent() {
    let Some((mut a, mut b)) = pair() else {
        return;
    };

    paint(
        &mut a,
        RED_VIVID,
        12.0,
        &[Vec2::new(40.0, 128.0), Vec2::new(216.0, 128.0)],
    );
    let actions = a.take_outbox();
    for action in &actions {
        assert!(b.merge_remote(action.clone()));
    }
    let img = snap(&mut b);
    for action in &actions {
        assert!(
            !b.merge_remote(action.clone()),
            "duplicate must be rejected"
        );
    }
    assert!(
        images_match(&snap(&mut b), &img, 0),
        "duplicate merge changed pixels"
    );
}

/// A shared document has no single playhead to scrub (§18.2.4).
///
/// Not a gap: the state is a function of a log peers are still appending to, so a
/// scrub would be silently undone by the next arrival. Answering `None` is what
/// lets the frontend say *why* the mode is unavailable instead of offering a
/// control that quietly does nothing.
#[test]
fn a_shared_timeline_reports_no_scrub_range() {
    let Some(mut a) = engine_or_skip() else {
        return;
    };
    paint(
        &mut a,
        RED_VIVID,
        12.0,
        &[Vec2::new(40.0, 128.0), Vec2::new(216.0, 128.0)],
    );
    assert!(a.scrub_range().is_some(), "solo, the history is walkable");

    a.start_collaboration(ActorId(1));
    assert_eq!(a.scrub_range(), None);
    assert!(a.scrub_labels().is_empty());

    // And a seek there is inert rather than destructive.
    let before = snap(&mut a);
    a.process(DocCommand::Seek(0));
    assert!(images_match(&before, &snap(&mut a), 0));
}

/// **Leaving the session hands the history back** (§18.2.4).
///
/// The refusal above is made on behalf of peers still appending to the log, so it
/// has to end when they do: a document nobody else is writing to is one client's
/// to walk, whoever painted what is in it. Asked here rather than left to the bar,
/// because "the scrubber is unavailable" has no pixels of its own — the mode simply
/// goes on saying a session that has ended is still running.
///
/// Three claims, in the order they can fail: the conversion moves no paint, the
/// walk that comes back spans the *whole* document (the peer's stroke included, and
/// exactly the actions a save would write), and travelling it is lossless.
#[test]
fn leaving_a_session_hands_the_history_back() {
    let Some((mut a, mut b)) = pair() else {
        return;
    };

    paint(
        &mut a,
        RED_VIVID,
        12.0,
        &[Vec2::new(40.0, 128.0), Vec2::new(216.0, 128.0)],
    );
    paint(
        &mut a,
        BLUE_VIVID,
        12.0,
        &[Vec2::new(40.0, 60.0), Vec2::new(216.0, 60.0)],
    );
    // Undone *as an action* while shared, so the log A leaves with holds a step it
    // must not walk back through: the `Undo` and its target are both still in it.
    a.process(DocCommand::Undo);
    sync(&mut a, &mut b);
    paint(
        &mut b,
        GREEN_SOFT,
        12.0,
        &[Vec2::new(128.0, 40.0), Vec2::new(128.0, 216.0)],
    );
    sync(&mut a, &mut b);

    let shared_log = a.document_file().actions.len();
    let shared = snap(&mut a);
    a.end_collaboration();
    assert!(
        images_match(&shared, &snap(&mut a), 0),
        "leaving repainted the canvas"
    );

    let (at, total) = a.scrub_range().expect("the history is this client's again");
    assert_eq!(at, total, "the playhead stands where the session left it");
    assert_eq!(
        total,
        a.document_file().actions.len(),
        "every action a save would write is a step the scrubber can reach"
    );
    assert_eq!(
        total,
        shared_log - 2,
        "the undone stroke and its `Undo` resolved away, as a solo load flattens them"
    );
    assert_eq!(a.scrub_labels().len(), total, "a caption per step");

    // The reference is the *replay* render: a scrub re-folds each stroke through
    // `apply`, where a live commit took its preview's own tiles (§6.2).
    let reference = whole_render(&mut a);
    a.process(DocCommand::Seek(0));
    assert!(
        !images_match(&reference, &snap(&mut a), 0),
        "a seek to the empty canvas changed nothing"
    );
    a.process(DocCommand::Seek(total));
    assert!(
        images_match(&reference, &snap(&mut a), 0),
        "the round trip through the peer's stroke did not come back"
    );
}

/// **Sharing the same document twice does not mint a layer id twice** (§17.9).
///
/// Partitioning the id space by author is what makes starting a counter at 1 safe,
/// and the half it partitions off is *this actor's* — which the actor of a second
/// share already has ids in, being the first one back again: an identity is a
/// browser's persisted key, not a session's. Starting over there mints an id the
/// document holds, the convergence failure `LayerId::mint` exists to rule out, and
/// locally it is worse than it sounds: `layer_index` resolves an id to whichever
/// layer comes first, so paint aimed at the new layer lands on the old.
///
/// Two roads in, because the counter is resumed from the log rather than from
/// whether the actor changed, and only the second road can tell those apart.
#[test]
fn re_sharing_a_document_does_not_re_mint_a_layer_id() {
    let Some(mut a) = engine_or_skip() else {
        return;
    };

    a.start_collaboration(ActorId(1));
    a.process(DocCommand::AddLayer {
        carrier: None,
        above: None,
    });
    let first = a.observe().active_layer;

    // The same browser sharing the same painting again: `leave` kept the canvas, and
    // the identity that comes back is the one that just left.
    a.end_collaboration();
    a.start_collaboration(ActorId(1));
    a.process(DocCommand::AddLayer {
        carrier: None,
        above: None,
    });
    let second = a.observe().active_layer;

    assert_ne!(
        first, second,
        "the second session re-minted an id the document already holds"
    );
    let ids: Vec<_> = a.observe().layers.iter().map(|l| l.id).collect();
    assert_eq!(ids.len(), 3, "root and one layer per session");
    assert_eq!(
        ids.iter().collect::<std::collections::HashSet<_>>().len(),
        ids.len(),
        "every layer in the stack is reachable by its own id"
    );

    // The second road: opened off disk, where nothing about this engine's own state
    // remembers that the ids in the file are this client's — the load starts it at
    // `SOLO`, so the actor does change, and only the log can say so.
    let bytes = a.save_bytes().expect("save the shared document");
    let Some(mut reopened) = engine_or_skip() else {
        return;
    };
    reopened.load_bytes(&bytes).expect("load it back");
    reopened.start_collaboration(ActorId(1));
    reopened.process(DocCommand::AddLayer {
        carrier: None,
        above: None,
    });
    assert!(
        !ids.contains(&reopened.observe().active_layer),
        "sharing a reopened file minted an id the file already holds"
    );
}

/// **A peer's merge-down removes a layer too**, so it has to repoint the brush the
/// same way a peer's `RemoveLayer` does (§17.9).
///
/// The defect §17.9 records as fixed was reintroduced by a feature added after it:
/// `merge_apply` ends in `remove_layer(source)`, but the repoint was keyed on the
/// `RemoveLayer` *variant* rather than on the fact of a layer going away. B is left
/// pointing at a layer that no longer exists, after which `apply` refuses every
/// stroke silently — the failure has no pixels of its own, which is why it is asked
/// here rather than left to a render comparison.
#[test]
fn a_remote_merge_down_does_not_strand_the_active_layer() {
    let (Some(mut a), Some(mut b)) = (engine_or_skip(), engine_or_skip()) else {
        return;
    };
    a.start_collaboration(ActorId(1));
    paint(
        &mut a,
        RED_VIVID,
        12.0,
        &[Vec2::new(40.0, 128.0), Vec2::new(216.0, 128.0)],
    );
    a.process(DocCommand::AddLayer {
        carrier: None,
        above: None,
    });
    let top = a.observe().active_layer;
    paint(
        &mut a,
        GREEN_SOFT,
        12.0,
        &[Vec2::new(40.0, 100.0), Vec2::new(216.0, 100.0)],
    );

    b.join_collaboration(&a.document_file(), ActorId(2))
        .expect("join a session this build can render");
    b.process(PeerCommand::SetActiveLayer(top));
    assert_eq!(
        b.observe().active_layer,
        top,
        "B is working on the top layer"
    );

    a.process(DocCommand::MergeLayerDown(top));
    assert!(
        a.document().layer(top).is_none(),
        "the merge folded the source away on A",
    );
    sync_into(&mut a, &mut b);
    assert!(b.document().layer(top).is_none(), "B received the merge");

    let obs = b.observe();
    assert!(
        obs.layers.iter().any(|l| l.id == obs.active_layer),
        "B's active layer {:?} no longer exists, so every stroke is silently refused",
        obs.active_layer,
    );
}
