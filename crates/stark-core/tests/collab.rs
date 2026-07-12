//! Collaboration semantics, headlessly (DESIGN.md §12, build-order step 12):
//! two engines exchanging action logs must converge to identical pixels —
//! convergence as a test — including out-of-order merges, undo-as-an-action,
//! and late joins. No network here; `stark-net` adds the wire on top.

mod common;

use common::{engine_or_skip, images_match, paint, PAPER};
use stark_core::command::InputCommand;
use stark_core::document::ActorId;
use stark_core::{Engine, RgbaImage};
use stark_core::geom::Vec2;

const RED: [f32; 4] = [0.9, 0.1, 0.1, 1.0];
const GREEN: [f32; 4] = [0.1, 0.8, 0.2, 1.0];
const BLUE: [f32; 4] = [0.1, 0.2, 0.9, 1.0];

fn snap(e: &mut Engine) -> RgbaImage {
    e.render_to_image(PAPER)
}

/// Pump every pending local action from `from` into `into`.
fn sync_into(from: &mut Engine, into: &mut Engine) {
    for action in from.take_outbox() {
        into.merge_remote(action);
    }
}

/// Exchange outboxes both ways.
fn sync(a: &mut Engine, b: &mut Engine) {
    sync_into(a, b);
    sync_into(b, a);
}

/// Two peers paint overlapping strokes and merge them in *different orders*
/// (B commits its own stroke first, then receives A's earlier-ordered one, so
/// B exercises the rewind-and-replay insert; A appends). Both must render the
/// same pixels, and a peer joining afterwards from the shared log must too.
#[test]
fn concurrent_strokes_converge() {
    let (Some(mut a), Some(mut b), Some(mut c)) =
        (engine_or_skip(), engine_or_skip(), engine_or_skip())
    else {
        return;
    };

    a.start_collaboration(ActorId(1));
    b.join_collaboration(&a.document_file(), ActorId(2));

    // Concurrent, overlapping edits: same lamport, actor id breaks the tie, so
    // A's stroke orders before B's even though B applies it second.
    paint(&mut a, RED, 12.0, &[Vec2::new(40.0, 128.0), Vec2::new(216.0, 128.0)]);
    paint(&mut b, GREEN, 12.0, &[Vec2::new(128.0, 40.0), Vec2::new(128.0, 216.0)]);
    sync(&mut a, &mut b);

    let img_a = snap(&mut a);
    let img_b = snap(&mut b);
    assert!(
        images_match(&img_a, &img_b, 0),
        "peers diverged after cross-merge"
    );

    // A late joiner catching up from the full shared log sees the same canvas.
    c.join_collaboration(&a.document_file(), ActorId(3));
    let img_c = snap(&mut c);
    assert!(
        images_match(&img_a, &img_c, 0),
        "late joiner diverged from the session"
    );
}

/// Undo in a shared session is a logged action targeting *my* stroke, skipping
/// the peer's intervening one (DESIGN.md §12.3); redo is an undo of that undo.
#[test]
fn shared_undo_skips_peer_actions() {
    let (Some(mut a), Some(mut b)) = (engine_or_skip(), engine_or_skip()) else {
        return;
    };

    a.start_collaboration(ActorId(1));
    b.join_collaboration(&a.document_file(), ActorId(2));

    paint(&mut a, RED, 12.0, &[Vec2::new(40.0, 128.0), Vec2::new(216.0, 128.0)]);
    // B's canvas before it has seen anything of A: exactly "only B's stroke".
    paint(&mut b, GREEN, 12.0, &[Vec2::new(128.0, 40.0), Vec2::new(128.0, 216.0)]);
    let only_b = snap(&mut b);

    sync(&mut a, &mut b);
    let both = snap(&mut a);

    // A undoes *its* stroke — B's later stroke must survive.
    a.process(InputCommand::Undo);
    sync(&mut a, &mut b);
    let img_a = snap(&mut a);
    let img_b = snap(&mut b);
    assert!(images_match(&img_a, &img_b, 0), "peers diverged after undo");
    assert!(
        images_match(&img_a, &only_b, 0),
        "undo of A's stroke should leave exactly B's stroke"
    );

    // Redo brings A's stroke back on both peers.
    a.process(InputCommand::Redo);
    sync(&mut a, &mut b);
    let img_a = snap(&mut a);
    let img_b = snap(&mut b);
    assert!(images_match(&img_a, &img_b, 0), "peers diverged after redo");
    assert!(
        images_match(&img_a, &both, 0),
        "redo should restore both strokes"
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
    paint(&mut a, RED, 12.0, &[Vec2::new(40.0, 100.0), Vec2::new(216.0, 100.0)]);
    let one = snap(&mut a);
    paint(&mut a, BLUE, 12.0, &[Vec2::new(40.0, 156.0), Vec2::new(216.0, 156.0)]);
    let two = snap(&mut a);

    a.process(InputCommand::Undo);
    assert!(images_match(&snap(&mut a), &one, 0), "first undo");
    a.process(InputCommand::Undo);
    assert!(images_match(&snap(&mut a), &blank, 0), "second undo");
    assert!(!a.observe().can_undo, "nothing left to undo");
    a.process(InputCommand::Redo);
    assert!(images_match(&snap(&mut a), &one, 0), "first redo");
    a.process(InputCommand::Redo);
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
    paint(&mut a, RED, 12.0, &[Vec2::new(40.0, 128.0), Vec2::new(216.0, 128.0)]);

    a.start_collaboration(ActorId(7));
    assert!(a.observe().can_undo, "pre-share stroke should be undoable");
    a.process(InputCommand::Undo);
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

    paint(&mut a, RED, 12.0, &[Vec2::new(40.0, 100.0), Vec2::new(216.0, 100.0)]);
    paint(&mut a, GREEN, 12.0, &[Vec2::new(40.0, 156.0), Vec2::new(216.0, 156.0)]);
    a.process(InputCommand::Undo); // green gone; log still contains it + the undo
    let expected = snap(&mut a);

    let bytes = a.save_bytes().expect("save shared doc");
    solo.load_bytes(&bytes).expect("load shared doc");
    assert!(
        images_match(&snap(&mut solo), &expected, 0),
        "solo load of a shared log should reproduce the session's canvas"
    );
    // The undone stroke was flattened away; the surviving one is undoable.
    assert!(solo.observe().can_undo);
    solo.process(InputCommand::Undo);
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

/// Duplicate deliveries (gossip redelivery) are idempotent.
#[test]
fn merge_is_idempotent() {
    let (Some(mut a), Some(mut b)) = (engine_or_skip(), engine_or_skip()) else {
        return;
    };
    a.start_collaboration(ActorId(1));
    b.join_collaboration(&a.document_file(), ActorId(2));

    paint(&mut a, RED, 12.0, &[Vec2::new(40.0, 128.0), Vec2::new(216.0, 128.0)]);
    let actions = a.take_outbox();
    for action in &actions {
        assert!(b.merge_remote(action.clone()));
    }
    let img = snap(&mut b);
    for action in &actions {
        assert!(!b.merge_remote(action.clone()), "duplicate must be rejected");
    }
    assert!(
        images_match(&snap(&mut b), &img, 0),
        "duplicate merge changed pixels"
    );
}
