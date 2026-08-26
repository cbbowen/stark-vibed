//! The commutation fast path (§12.6): an undo whose target commutes
//! with everything materialized after it is shifted out of the history —
//! `Footprint` as the action's `Centralizer`, `Action::inverse` restoring what
//! it wrote — instead of rewinding and replaying; redo re-materializes at the
//! top of the stack (§12.3), so it is a plain append.
//!
//! Pixels can't tell the paths apart — that's the point — so each test asserts
//! two things: the *stats* say the fast path ran (or correctly refused to), and
//! the pixels still match a fresh peer materializing the same log canonically.

mod common;

use common::{engine_or_skip, engine_or_skip_sized, images_match, paint};
use stark_engine::command::DocCommand;
use stark_engine::{Engine, RgbaImage};
use stark_model::document::{ActorId, LayerId};
use stark_model::geom::{Extent2, Vec2};

const RED: [f32; 3] = [0.9, 0.1, 0.1];
const GREEN: [f32; 3] = [0.1, 0.8, 0.2];
const BLUE: [f32; 3] = [0.1, 0.2, 0.9];

/// A viewport wide enough to hold two tile columns (`TILE_SIZE` is 254 canvas
/// px): same-layer strokes only commute when their padded extents share no
/// tile — the splice patches whole tile handles, so a shared tile is a genuine
/// conflict — and the default 256² canvas is a single tile.
const WIDE: Extent2 = Extent2 {
    width: 600,
    height: 300,
};

/// In `WIDE`, x-extents that keep two strokes in disjoint tile columns even
/// after footprint padding.
const LEFT: (f32, f32) = (20.0, 90.0);
const RIGHT: (f32, f32) = (300.0, 500.0);

fn snap(e: &mut Engine) -> RgbaImage {
    e.render_to_image()
}

fn sync_into(from: &mut Engine, into: &mut Engine) {
    for action in from.take_outbox() {
        into.merge_remote(action);
    }
}

fn sync(a: &mut Engine, b: &mut Engine) {
    sync_into(a, b);
    sync_into(b, a);
}

/// The canonical materialization of `engine`'s log: a fresh peer joining from
/// the full shared document, which rewinds nothing and splices nothing. The
/// fast paths must produce *identical* pixels to this, tolerance zero — that
/// is the convergence claim of §12.6.
///
/// **A scenario with a selection in it must not use this one.** The joiner has an
/// actor of its own, and a selection is per-author (§17.3) whose outline is chrome
/// drawn for its owner — `show_peer_selections` is off by default — so a stranger
/// renders every selection without the outline the engine under test draws, and
/// reports a difference about who is looking rather than about what the log says.
/// The scenarios below are strokes, which no viewpoint changes.
/// `commute_pairs::spliced_and_canonical` joins as the *authoring* actor for exactly
/// this reason, and is the place to put a selection scenario.
fn canonical_snap(of: &mut Engine, size: Extent2) -> Option<RgbaImage> {
    let mut fresh = engine_or_skip_sized(size)?;
    fresh
        .join_collaboration(&of.document_file(), ActorId(99))
        .expect("join a session this build can render");
    Some(snap(&mut fresh))
}

/// A short horizontal stroke centred at `y`, in `x0..x1`.
fn bar(e: &mut Engine, color: [f32; 3], (x0, x1): (f32, f32), y: f32) {
    paint(e, color, 8.0, &[Vec2::new(x0, y), Vec2::new(x1, y)]);
}

/// Undoing a stroke buried under a peer's non-overlapping strokes must splice,
/// not replay — on both the author and the peer — and still match a canonical
/// rebuild exactly. Redo (an undo of the undo) must ride the insert fast path
/// the same way.
#[test]
fn undo_and_redo_splice_past_disjoint_peer_strokes() {
    let (Some(mut a), Some(mut b)) = (engine_or_skip_sized(WIDE), engine_or_skip_sized(WIDE))
    else {
        return;
    };
    a.start_collaboration(ActorId(1));
    b.join_collaboration(&a.document_file(), ActorId(2))
        .expect("join a session this build can render");

    // A paints in the left tile column; B concurrently paints two bars in the
    // right one (same layer, disjoint tiles). Absorbing A's earlier-ordered
    // stroke costs B one shallow rewind (mid-sequence arrivals have no fast
    // path, §12.6); the undo and redo below must add nothing on either peer.
    bar(&mut a, RED, LEFT, 80.0);
    bar(&mut b, GREEN, RIGHT, 100.0);
    bar(&mut b, BLUE, RIGHT, 220.0);
    sync(&mut a, &mut b);
    let both = snap(&mut a);
    let before_a = a.timeline_stats();
    let before_b = b.timeline_stats();

    // A undoes its buried stroke: a pure commuting shift on both peers.
    a.process(DocCommand::Undo);
    sync(&mut a, &mut b);

    for (name, e, before) in [("author", &mut a, before_a), ("peer", &mut b, before_b)] {
        let stats = e.timeline_stats();
        assert_eq!(
            stats.rebuilds, before.rebuilds,
            "{name} rebuilt instead of shifting"
        );
        assert_eq!(stats.replayed, before.replayed, "{name} replayed actions");
        assert_eq!(
            stats.fast_removes,
            before.fast_removes + 1,
            "{name} missed the undo fast path"
        );
    }
    let img_a = snap(&mut a);
    let img_b = snap(&mut b);
    assert!(images_match(&img_a, &img_b, 0), "peers diverged after undo");
    let Some(canon) = canonical_snap(&mut a, WIDE) else {
        return;
    };
    assert!(
        images_match(&img_a, &canon, 0),
        "spliced undo diverged from canonical materialization"
    );

    // Redo materializes the revived stroke at the *top* of the stack
    // (§12.3), so it is a plain append on both peers: no further
    // rebuilds, nothing replayed.
    let before_a = a.timeline_stats();
    let before_b = b.timeline_stats();
    a.process(DocCommand::Redo);
    sync(&mut a, &mut b);
    for (name, e, before) in [("author", &mut a, before_a), ("peer", &mut b, before_b)] {
        let stats = e.timeline_stats();
        assert_eq!(stats.rebuilds, before.rebuilds, "{name} rebuilt on redo");
        assert_eq!(stats.replayed, before.replayed, "{name} replayed on redo");
    }
    let img_a = snap(&mut a);
    assert!(
        images_match(&img_a, &snap(&mut b), 0),
        "peers diverged after redo"
    );
    assert!(
        images_match(&img_a, &both, 0),
        "redo should restore both strokes"
    );
}

/// A peer's stroke arriving *behind* newer local strokes (smaller Lamport
/// slot) takes the rewind — mid-sequence inserts have no fast path by design
/// (§12.6) — but the rewind is shallow: a concurrent arrival sits
/// near the top of the stack by construction, so only the tail replays.
#[test]
fn late_arrival_replays_only_the_tail() {
    let (Some(mut a), Some(mut b)) = (engine_or_skip_sized(WIDE), engine_or_skip_sized(WIDE))
    else {
        return;
    };
    a.start_collaboration(ActorId(1));
    b.join_collaboration(&a.document_file(), ActorId(2))
        .expect("join a session this build can render");

    bar(&mut a, RED, LEFT, 80.0);
    sync(&mut a, &mut b);

    // B commits one bar at Lamport 2 but holds it back while A keeps painting
    // (Lamport 2, 3): when it finally lands on A it belongs mid-log, under
    // exactly one newer local stroke.
    bar(&mut b, GREEN, RIGHT, 150.0);
    bar(&mut a, BLUE, LEFT, 150.0);
    bar(&mut a, RED, LEFT, 220.0);
    sync_into(&mut b, &mut a);

    let stats = a.timeline_stats();
    assert_eq!(stats.rebuilds, 1, "a mid-sequence arrival rewinds");
    assert_eq!(
        stats.replayed, 2,
        "the rewind should replay only the arrival and the one stroke above it"
    );

    sync_into(&mut a, &mut b);
    let img_a = snap(&mut a);
    assert!(images_match(&img_a, &snap(&mut b), 0), "peers diverged");
    let Some(canon) = canonical_snap(&mut a, WIDE) else {
        return;
    };
    assert!(
        images_match(&img_a, &canon, 0),
        "rewound insert diverged from canonical materialization"
    );
}

/// Strokes on *different layers* always commute, whatever their geometry — so
/// an undo splices past a peer's overlapping stroke when it lives on the
/// peer's own layer, even on a canvas small enough to be a single tile.
#[test]
fn undo_splices_past_another_layers_stroke() {
    let (Some(mut a), Some(mut b)) = (engine_or_skip(), engine_or_skip()) else {
        return;
    };
    a.start_collaboration(ActorId(1));
    b.join_collaboration(&a.document_file(), ActorId(2))
        .expect("join a session this build can render");

    paint(
        &mut a,
        RED,
        12.0,
        &[Vec2::new(40.0, 128.0), Vec2::new(216.0, 128.0)],
    );
    sync(&mut a, &mut b);
    // B adds its own layer (which becomes B's active layer) and paints a
    // crossing stroke there.
    b.process(DocCommand::AddLayer {
        carrier: None,
        above: None,
    });
    paint(
        &mut b,
        GREEN,
        12.0,
        &[Vec2::new(128.0, 40.0), Vec2::new(128.0, 216.0)],
    );
    sync(&mut b, &mut a);

    a.process(DocCommand::Undo);
    sync(&mut a, &mut b);

    for (name, e) in [("author", &mut a), ("peer", &mut b)] {
        let stats = e.timeline_stats();
        assert_eq!(stats.rebuilds, 0, "{name} rebuilt instead of splicing");
        assert_eq!(stats.fast_removes, 1, "{name} missed the undo fast path");
    }
    let img_a = snap(&mut a);
    assert!(images_match(&img_a, &snap(&mut b), 0), "peers diverged");
    let Some(canon) = canonical_snap(&mut a, common::SIZE) else {
        return;
    };
    assert!(
        images_match(&img_a, &canon, 0),
        "spliced undo diverged from canonical materialization"
    );
}

/// A rename after the stroke must not block the stroke's undo fast path — and
/// must survive the splice untouched.
#[test]
fn undo_splices_past_a_rename() {
    let (Some(mut a), Some(mut b)) = (engine_or_skip(), engine_or_skip()) else {
        return;
    };
    a.start_collaboration(ActorId(1));
    b.join_collaboration(&a.document_file(), ActorId(2))
        .expect("join a session this build can render");

    bar(&mut a, RED, (20.0, 230.0), 128.0);
    sync(&mut a, &mut b);
    b.process(DocCommand::SetLayerName(
        LayerId(0),
        Some("underpainting".into()),
    ));
    sync(&mut b, &mut a);

    a.process(DocCommand::Undo);
    sync(&mut a, &mut b);

    let stats = a.timeline_stats();
    assert_eq!(stats.rebuilds, 0, "rename blocked the undo fast path");
    assert_eq!(stats.fast_removes, 1);
    for e in [&mut a, &mut b] {
        let observed = e.observe();
        let layer = observed.layers.iter().find(|l| l.id == LayerId(0));
        assert_eq!(
            layer.and_then(|l| l.name.as_deref()),
            Some("underpainting"),
            "rename lost across the splice"
        );
    }
    let img_a = snap(&mut a);
    assert!(images_match(&img_a, &snap(&mut b), 0), "peers diverged");
}

/// **A stroke does not commute with the substrate it was painted on**, so an undo
/// of a `SetSubstrate` buried under later strokes must refuse the splice (§6.4,
/// §12.6).
///
/// The tooth gates how much paint lands by the substrate's rise, and `apply` reads
/// it off the state being folded over — so the substrate in force is part of what a
/// stroke *is*. The footprint said otherwise for a while, and this is what that
/// cost.
///
/// # Why there are five strokes
///
/// One would pass either way, which is the trap. `shift_late` walks the *cached*
/// states, and at each one it inverts the shifted action out and re-applies the
/// single action at that version — so a lone stroke is re-rendered onto the
/// pre-substrate state and comes out right by accident, footprint or no footprint.
/// It is the run between two snapshots that is left behind: the whole run rotates
/// past in one step, only its last action is re-applied, and every stroke before
/// that keeps tiles toothed by a substrate the log no longer names.
///
/// # What the failure looks like
///
/// Not a disagreement between peers — **both take the same splice and converge on
/// the same wrong picture** (`a == b` held throughout while `a == canonical` did
/// not). The document is simply no longer what its own log says, so it changes the
/// next time anyone opens the file. That is why the canonical comparison is the
/// assertion that matters here and peer agreement is not enough.
///
/// Shaped like [`undo_splices_past_a_rename`], and the two are the pair worth
/// reading together: there the buried action genuinely commutes and the splice is
/// the right answer, here it does not and the rebuild is. The brush is given a
/// **biting tooth** (`give` below its inert default of 1.0) so the substrate
/// reaches the paint at all.
#[test]
fn undo_of_a_substrate_does_not_splice_past_the_strokes_it_toothed() {
    let (Some(mut a), Some(mut b)) = (engine_or_skip(), engine_or_skip()) else {
        return;
    };
    a.start_collaboration(ActorId(1));

    // Set before `b` joins, so the height map rides in the snapshot `b` joins from
    // and both peers tooth against the same bytes (§8, §12.4).
    let rough = a
        .import_substrate(&stark_testdata::assets::rough())
        .expect("the rough height map imports");
    a.process(DocCommand::SetSubstrate(rough));
    b.join_collaboration(&a.document_file(), ActorId(2))
        .expect("join a session this build can render");

    let mut biting = common::brush(RED, 8.0);
    biting.tooth.give = 0.2;
    for y in [60.0f32, 100.0, 140.0, 180.0, 220.0] {
        common::stroke_with(&mut b, biting, &[Vec2::new(20.0, y), Vec2::new(230.0, y)]);
    }
    sync(&mut a, &mut b);

    // `a`'s own last action is the substrate, now buried under `b`'s five strokes.
    a.process(DocCommand::Undo);
    sync(&mut a, &mut b);

    let stats = a.timeline_stats();
    assert_eq!(
        stats.fast_removes, 0,
        "a substrate was spliced out past strokes it toothed",
    );
    assert!(stats.rebuilds >= 1, "the strokes should have re-rendered");

    let img_a = snap(&mut a);
    assert!(images_match(&img_a, &snap(&mut b), 0), "peers diverged");
    let Some(canon) = canonical_snap(&mut a, common::SIZE) else {
        return;
    };
    assert!(
        images_match(&img_a, &canon, 0),
        "the spliced state kept a tooth the log no longer names",
    );
}

/// Overlapping strokes genuinely don't commute: the fast path must refuse and
/// fall back to the rewind — and the peers must still converge exactly.
#[test]
fn overlapping_strokes_fall_back_to_replay() {
    let (Some(mut a), Some(mut b)) = (engine_or_skip(), engine_or_skip()) else {
        return;
    };
    a.start_collaboration(ActorId(1));
    b.join_collaboration(&a.document_file(), ActorId(2))
        .expect("join a session this build can render");

    // Crossing strokes through the canvas centre.
    paint(
        &mut a,
        RED,
        12.0,
        &[Vec2::new(40.0, 128.0), Vec2::new(216.0, 128.0)],
    );
    paint(
        &mut b,
        GREEN,
        12.0,
        &[Vec2::new(128.0, 40.0), Vec2::new(128.0, 216.0)],
    );
    sync(&mut a, &mut b);

    a.process(DocCommand::Undo);
    sync(&mut a, &mut b);

    for (name, e) in [("author", &mut a), ("peer", &mut b)] {
        let stats = e.timeline_stats();
        assert_eq!(
            stats.fast_removes, 0,
            "{name} spliced out a stroke its neighbour overlaps"
        );
        assert!(stats.rebuilds >= 1, "{name} should have rebuilt");
    }
    let img_a = snap(&mut a);
    assert!(
        images_match(&img_a, &snap(&mut b), 0),
        "peers diverged after conflicted undo"
    );
    let Some(canon) = canonical_snap(&mut a, common::SIZE) else {
        return;
    };
    assert!(
        images_match(&img_a, &canon, 0),
        "fallback diverged from canonical materialization"
    );
}
