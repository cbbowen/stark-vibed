//! Per-client state that every client reads and only its owner writes
//! (PEER_DESIGN.md).
//!
//! Two mechanisms, tested apart because the rule that separates them is the whole
//! design: the **selection** is per-actor *document* state, because a stroke's pixels
//! depend on the mask it was drawn through and replay has to reconstruct it; a peer's
//! selected layer and live gesture are **presence**, unlogged, because replay needs
//! neither.
//!
//! Everything here is headless and needs no network: the semantics live in
//! `stark-core`, which is the point of putting them there.

mod common;

use common::{engine_or_skip, images_match, paint};
use stark_core::command::{DocCommand, GestureCommand, InputSample, PeerCommand, ViewCommand};
use stark_core::document::{ActorId, LayerId, Tool};
use stark_core::geom::Vec2;
use stark_core::path::DEFAULT_TOLERANCE;
use stark_core::peer::{GestureFrame, PeerFrame, StrokeHead};
use stark_core::{Engine, RgbaImage};
use stark_core::{SelectionMode, SelectionOp, SelectionShape};

const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const GREEN: [f32; 4] = [0.1, 0.8, 0.2, 1.0];

/// The half of the canvas actor A will select: everything left of x = 0.
const LEFT_MIN: Vec2 = Vec2::new(-400.0, -400.0);
const LEFT_MAX: Vec2 = Vec2::new(0.0, 400.0);

/// A horizontal stroke crossing x = 0, so the same gesture has a half inside A's
/// selection and a half outside it.
const CROSSING: [Vec2; 2] = [Vec2::new(-80.0, 0.0), Vec2::new(80.0, 0.0)];

fn snap(e: &mut Engine) -> RgbaImage {
    e.render_to_image()
}

fn sync(a: &mut Engine, b: &mut Engine) {
    for action in a.take_outbox() {
        b.merge_remote(action);
    }
    for action in b.take_outbox() {
        a.merge_remote(action);
    }
}

fn select_left(engine: &mut Engine) {
    engine.process(DocCommand::Select(SelectionOp::new(
        SelectionMode::Replace,
        SelectionShape::rect_from_corners(LEFT_MIN, LEFT_MAX),
        0.0,
    )));
}

/// Whether the pixel at a canvas point reads as paint rather than bare paper. The
/// substrate is a neutral near-white, so a strongly red-dominant pixel is paint.
fn is_painted(img: &RgbaImage, canvas: Vec2) -> bool {
    let half = Vec2::new(img.width as f32, img.height as f32) * 0.5;
    let p = canvas + half;
    let i = ((p.y as u32 * img.width + p.x as u32) * 4) as usize;
    let (r, g, b) = (
        img.pixels[i] as i32,
        img.pixels[i + 1] as i32,
        img.pixels[i + 2] as i32,
    );
    r - g > 40 && r - b > 40
}

/// **The defect this design exists to fix.** One collaborator's selection must not
/// clip another's brush.
///
/// Before selections were keyed by actor, `DocState` held one mask and
/// `CommitStroke` read it from the state it folded over — so A's lasso silently
/// clipped B's next stroke, on B's own screen.
#[test]
fn one_peers_selection_does_not_clip_anothers_stroke() {
    let (Some(mut a), Some(mut b)) = (engine_or_skip(), engine_or_skip()) else {
        return;
    };
    a.start_collaboration(ActorId(1));
    b.join_collaboration(&a.document_file(), ActorId(2));

    // A masks the left half of the canvas, and B hears about it.
    select_left(&mut a);
    sync(&mut a, &mut b);
    assert!(a.observe().has_selection, "A masked the canvas");
    assert!(
        !b.observe().has_selection,
        "A's selection must not become B's"
    );

    // B paints across the boundary. Both halves must land: B has no mask.
    paint(&mut b, RED, 12.0, &CROSSING);
    sync(&mut a, &mut b);

    for (name, engine) in [("author", &mut b), ("peer", &mut a)] {
        let img = snap(engine);
        assert!(
            is_painted(&img, Vec2::new(-60.0, 0.0)),
            "{name}: inside A's selection"
        );
        assert!(
            is_painted(&img, Vec2::new(60.0, 0.0)),
            "{name}: outside A's selection — B is unmasked, so this must be painted"
        );
    }
}

/// The author's own selection still gates the author's stroke, and every peer
/// reproduces that gating — which is what makes replicating the mask necessary
/// rather than merely tidy.
#[test]
fn a_peers_stroke_is_reproduced_through_the_authors_own_mask() {
    let (Some(mut a), Some(mut b)) = (engine_or_skip(), engine_or_skip()) else {
        return;
    };
    a.start_collaboration(ActorId(1));
    b.join_collaboration(&a.document_file(), ActorId(2));

    select_left(&mut a);
    paint(&mut a, RED, 12.0, &CROSSING);
    sync(&mut a, &mut b);

    for (name, engine) in [("author", &mut a), ("peer", &mut b)] {
        let img = snap(engine);
        assert!(
            is_painted(&img, Vec2::new(-60.0, 0.0)),
            "{name}: inside A's own selection"
        );
        assert!(
            !is_painted(&img, Vec2::new(60.0, 0.0)),
            "{name}: outside A's own selection — the author's mask must gate it"
        );
    }
}

/// Two peers may hold *different* selections at once, and each one's strokes are
/// gated by their own. Convergence is unaffected: both render the same canvas.
#[test]
fn peers_hold_independent_selections_and_still_converge() {
    let (Some(mut a), Some(mut b), Some(mut c)) =
        (engine_or_skip(), engine_or_skip(), engine_or_skip())
    else {
        return;
    };
    a.start_collaboration(ActorId(1));
    b.join_collaboration(&a.document_file(), ActorId(2));

    // A takes the left half, B the right.
    select_left(&mut a);
    b.process(DocCommand::Select(SelectionOp::new(
        SelectionMode::Replace,
        SelectionShape::rect_from_corners(Vec2::new(0.0, -400.0), Vec2::new(400.0, 400.0)),
        0.0,
    )));
    sync(&mut a, &mut b);

    // Each paints the same crossing line; each keeps only their own half.
    paint(&mut a, RED, 12.0, &CROSSING);
    paint(
        &mut b,
        GREEN,
        12.0,
        &[Vec2::new(-80.0, 40.0), Vec2::new(80.0, 40.0)],
    );
    sync(&mut a, &mut b);

    let masked = snap(&mut a);
    assert!(is_painted(&masked, Vec2::new(-60.0, 0.0)), "A's kept half");
    assert!(
        !is_painted(&masked, Vec2::new(60.0, 0.0)),
        "A's clipped half"
    );

    // Convergence is about the *artwork*, not the chrome: the marching ants are
    // drawn for whoever's selection is in force on this client, so a peer with a
    // different mask legitimately shows a different outline. Deselect both, so what
    // is compared is the paint.
    a.process(DocCommand::Select(SelectionOp::select_all()));
    b.process(DocCommand::Select(SelectionOp::select_all()));
    sync(&mut a, &mut b);

    let img_a = snap(&mut a);
    assert!(images_match(&img_a, &snap(&mut b), 0), "peers diverged");

    // A late joiner replays the whole log — both peers' masks and the strokes they
    // gated — and lands on the same pixels.
    c.join_collaboration(&a.document_file(), ActorId(3));
    assert!(
        images_match(&img_a, &snap(&mut c), 0),
        "a late joiner must reconstruct every actor's mask from the log"
    );
}

/// Undo is already scoped to *my* actions (DESIGN.md §12.3), so undoing my selection
/// leaves yours alone. This is a consequence of keying by the action's own author,
/// not a rule anything has to enforce.
#[test]
fn undoing_my_selection_leaves_yours() {
    let (Some(mut a), Some(mut b)) = (engine_or_skip(), engine_or_skip()) else {
        return;
    };
    a.start_collaboration(ActorId(1));
    b.join_collaboration(&a.document_file(), ActorId(2));

    select_left(&mut a);
    b.process(DocCommand::Select(SelectionOp::new(
        SelectionMode::Replace,
        SelectionShape::rect_from_corners(Vec2::new(0.0, -400.0), Vec2::new(400.0, 400.0)),
        0.0,
    )));
    sync(&mut a, &mut b);
    assert!(a.observe().has_selection && b.observe().has_selection);

    a.process(DocCommand::Undo);
    sync(&mut a, &mut b);
    assert!(!a.observe().has_selection, "A undid its own selection");
    assert!(
        b.observe().has_selection,
        "B's selection must survive A's undo"
    );
}

/// A document that was never shared behaves exactly as before: one actor, one mask.
/// This is the check that the re-keying is invisible where there is nothing to key
/// by — and the reason every existing golden is unchanged.
#[test]
fn a_solo_document_is_unaffected() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    select_left(&mut engine);
    assert!(engine.observe().has_selection);
    paint(&mut engine, RED, 12.0, &CROSSING);
    let img = snap(&mut engine);
    assert!(is_painted(&img, Vec2::new(-60.0, 0.0)));
    assert!(!is_painted(&img, Vec2::new(60.0, 0.0)));
}

// --- layer ids (PEER_DESIGN.md §9) --------------------------------------

/// Two peers adding a layer at the same moment must mint **different** ids.
///
/// With a counter resynced from the log they minted the same one: both saw `n`
/// layers, both minted `n + 1`, and the merged log then held two different layers
/// under a single id, which `layer_index` resolved to whichever came first.
#[test]
fn concurrent_layer_adds_do_not_collide() {
    let (Some(mut a), Some(mut b)) = (engine_or_skip(), engine_or_skip()) else {
        return;
    };
    a.start_collaboration(ActorId(1));
    b.join_collaboration(&a.document_file(), ActorId(2));

    a.process(DocCommand::AddLayer { above: None });
    b.process(DocCommand::AddLayer { above: None });
    let id_a = a.observe().active_layer;
    let id_b = b.observe().active_layer;
    assert_ne!(id_a, id_b, "concurrent adds minted the same layer id");

    sync(&mut a, &mut b);
    let ids: Vec<LayerId> = a.observe().layers.iter().map(|l| l.id).collect();
    assert_eq!(ids.len(), 3, "root plus one layer from each peer: {ids:?}");
    assert!(ids.contains(&id_a) && ids.contains(&id_b));
    let mut sorted = ids.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), ids.len(), "duplicate layer ids: {ids:?}");
}

/// A solo document keeps the small, readable ids it always had — `ActorId::SOLO`
/// maps to id-space 0, so nothing about the partitioning is visible until a document
/// is actually shared.
#[test]
fn solo_layer_ids_stay_small() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.process(DocCommand::AddLayer { above: None });
    assert_eq!(engine.observe().active_layer, LayerId(1));
    engine.process(DocCommand::AddLayer { above: None });
    assert_eq!(engine.observe().active_layer, LayerId(2));
}

/// A peer deleting the layer this client is painting on must not leave it pointed at
/// a layer that no longer exists — after which every stroke was silently refused by
/// `apply` with nothing on screen to explain it.
#[test]
fn a_remote_removal_repoints_the_active_layer() {
    let (Some(mut a), Some(mut b)) = (engine_or_skip(), engine_or_skip()) else {
        return;
    };
    a.start_collaboration(ActorId(1));
    b.join_collaboration(&a.document_file(), ActorId(2));

    a.process(DocCommand::AddLayer { above: None });
    sync(&mut a, &mut b);
    let added = a.observe().active_layer;

    // B selects the new layer; A deletes it out from under them.
    b.process(PeerCommand::SetActiveLayer(added));
    assert_eq!(b.observe().active_layer, added);
    a.process(DocCommand::RemoveLayer(added));
    sync(&mut a, &mut b);

    let obs = b.observe();
    assert!(
        obs.layers.iter().any(|l| l.id == obs.active_layer),
        "active layer {:?} is not in the stack {:?}",
        obs.active_layer,
        obs.layers.iter().map(|l| l.id).collect::<Vec<_>>()
    );
    // And painting works again, rather than silently doing nothing.
    paint(&mut b, RED, 12.0, &CROSSING);
    assert!(is_painted(&snap(&mut b), Vec2::new(0.0, 0.0)));
}

// --- presence (PEER_DESIGN.md §4) ---------------------------------------

/// A peer's in-flight stroke is drawn before it commits, and the *committed* action
/// replaces it with the same pixels — the live-equals-committed invariant, across
/// the wire.
#[test]
fn a_peers_live_stroke_previews_and_the_commit_matches_it() {
    let (Some(mut a), Some(mut b)) = (engine_or_skip(), engine_or_skip()) else {
        return;
    };
    a.start_collaboration(ActorId(1));
    b.join_collaboration(&a.document_file(), ActorId(2));

    // A draws, publishing as it goes; B reassembles and previews.
    let brush = common::brush(RED, 12.0);
    a.process(stark_core::command::ViewCommand::SetBrush(brush));
    a.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(CROSSING[0]),
        tolerance: DEFAULT_TOLERANCE,
    });
    let mut now = 0.0;
    let pump = |a: &mut Engine, b: &mut Engine, now: &mut f64| {
        *now += 0.05;
        if let Some(frame) = a.take_presence(*now) {
            b.merge_presence(ActorId(1), frame, *now);
        }
    };
    pump(&mut a, &mut b, &mut now);
    for i in 1..=8 {
        let t = i as f32 / 8.0;
        a.process(GestureCommand::To {
            sample: InputSample::at(CROSSING[0] + (CROSSING[1] - CROSSING[0]) * t),
        });
        pump(&mut a, &mut b, &mut now);
    }

    let preview = snap(&mut b);
    assert!(
        is_painted(&preview, Vec2::new(0.0, 0.0)),
        "B should see A's stroke while it is being drawn"
    );

    // A releases; the action lands and supersedes the preview.
    a.process(GestureCommand::End);
    sync(&mut a, &mut b);
    let committed = snap(&mut b);
    assert!(is_painted(&committed, Vec2::new(0.0, 0.0)));
    assert!(
        images_match(&committed, &snap(&mut a), 0),
        "the commit must land the same pixels on both peers"
    );
}

/// A peer that stops publishing loses its live gesture, then leaves the roster —
/// so a crash mid-stroke does not leave a smear on everyone else's canvas.
#[test]
fn a_silent_peer_loses_its_gesture_then_its_place() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.start_collaboration(ActorId(1));

    let frame = PeerFrame {
        boot: 0,
        seq: 1,
        name: Some("Ada".into()),
        active_layer: LayerId(0),
        cursor: Some(Vec2::ZERO),
        gesture: Some(GestureFrame::Stroke {
            id: 0,
            head: Some(StrokeHead {
                layer: LayerId(0),
                tool: Tool::Brush,
                brush: common::brush(RED, 12.0),
                seed: 1,
            }),
            from: 0,
            points: CROSSING
                .iter()
                .map(|p| stark_core::path::ControlPoint::at(*p))
                .collect(),
        }),
        leaving: false,
    };
    assert!(engine.merge_presence(ActorId(2), frame, 0.0));
    assert!(
        is_painted(&snap(&mut engine), Vec2::new(0.0, 0.0)),
        "a peer's live stroke should be visible"
    );

    // Silence: the gesture expires first, then the peer.
    engine.take_presence(stark_core::peer::GESTURE_TIMEOUT + 0.1);
    assert_eq!(engine.peers().count(), 1, "still on the roster");
    assert!(
        !is_painted(&snap(&mut engine), Vec2::new(0.0, 0.0)),
        "a stalled gesture must stop being drawn"
    );
    engine.take_presence(stark_core::peer::PEER_TIMEOUT + 0.1);
    assert_eq!(engine.peers().count(), 0, "the peer left");
}

/// A collaborator's selection outline is off by default and drawn only when this
/// client asks for it — a view setting, so it changes what you look at and nothing
/// about the drawing (PEER_DESIGN.md §3). Your own outline is unaffected either way.
#[test]
fn peer_selection_outlines_are_opt_in() {
    let (Some(mut a), Some(mut b)) = (engine_or_skip(), engine_or_skip()) else {
        return;
    };
    a.start_collaboration(ActorId(1));
    b.join_collaboration(&a.document_file(), ActorId(2));

    // B is present and has a selection whose edge crosses the viewport.
    b.process(DocCommand::Select(SelectionOp::new(
        SelectionMode::Replace,
        SelectionShape::rect_from_corners(LEFT_MIN, LEFT_MAX),
        0.0,
    )));
    sync(&mut a, &mut b);
    a.merge_presence(
        ActorId(2),
        PeerFrame {
            boot: 0,
            seq: 1,
            name: None,
            active_layer: LayerId(0),
            cursor: None,
            gesture: None,
            leaving: false,
        },
        0.0,
    );

    let hidden = snap(&mut a);
    assert!(
        !a.observe().show_peer_selections,
        "collaborators' outlines are off until asked for"
    );

    a.process(ViewCommand::SetShowPeerSelections(true));
    let shown = snap(&mut a);
    assert!(
        !images_match(&hidden, &shown, 0),
        "turning the setting on should draw B's outline"
    );

    a.process(ViewCommand::SetShowPeerSelections(false));
    assert!(
        images_match(&hidden, &snap(&mut a), 0),
        "turning it back off should restore exactly the canvas without it"
    );

    // And it reaches only the peers' outlines: A's own is drawn regardless.
    select_left(&mut a);
    let mine = snap(&mut a);
    assert!(
        !images_match(&mine, &hidden, 0),
        "this client's own selection outline is not gated by the setting"
    );
}

/// `presence_due` is the cheap `&self` test a pump uses to skip an idle tick
/// entirely. It is allowed to be pessimistic, but it must **never** say no where
/// `take_presence` would have produced a frame — a pump that trusted it would drop
/// that frame on the floor, and the peer's cursor or stroke would stall until the
/// next heartbeat.
#[test]
fn presence_due_never_hides_a_frame() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.start_collaboration(ActorId(1));

    let mut now = 0.0;
    let mut idle_ticks = 0;
    // Every kind of change a frame can carry, plus long runs of nothing between
    // them — which is where the skipping actually happens.
    let step = |engine: &mut Engine, now: &mut f64, idle: &mut u32| {
        *now += 0.033;
        let due = engine.presence_due(*now);
        let frame = engine.take_presence(*now);
        assert!(
            due || frame.is_none(),
            "presence_due said no at t={now} but a frame was produced"
        );
        if !due {
            *idle += 1;
        }
    };

    for _ in 0..10 {
        step(&mut engine, &mut now, &mut idle_ticks);
    }
    engine.process(PeerCommand::SetCursor(Some(Vec2::new(3.0, 4.0))));
    for _ in 0..10 {
        step(&mut engine, &mut now, &mut idle_ticks);
    }
    engine.process(PeerCommand::SetName("Ada".into()));
    for _ in 0..10 {
        step(&mut engine, &mut now, &mut idle_ticks);
    }
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(CROSSING[0]),
        tolerance: DEFAULT_TOLERANCE,
    });
    for i in 0..10 {
        engine.process(GestureCommand::To {
            sample: InputSample::at(CROSSING[0] + (CROSSING[1] - CROSSING[0]) * (i as f32 / 10.0)),
        });
        step(&mut engine, &mut now, &mut idle_ticks);
    }
    // The end of a gesture is the case most easily missed: the frame that *clears*
    // it is what stops peers drawing a stroke nobody is making any more.
    engine.process(GestureCommand::End);
    for _ in 0..10 {
        step(&mut engine, &mut now, &mut idle_ticks);
    }
    // And the heartbeat still gets through a long silence.
    now += stark_core::peer::HEARTBEAT;
    let due = engine.presence_due(now);
    assert!(due, "the heartbeat must come due");
    assert!(engine.take_presence(now).is_some(), "heartbeat frame");

    assert!(
        idle_ticks > 20,
        "only {idle_ticks} ticks were skippable — the test is not exercising the \
         idle path the optimisation exists for"
    );
}

/// Presence never reaches the document: it is not in `DocState`, so it cannot be
/// saved, and a file written mid-gesture contains only what committed.
#[test]
fn presence_never_reaches_the_save_file() {
    let (Some(mut a), Some(mut b)) = (engine_or_skip(), engine_or_skip()) else {
        return;
    };
    a.start_collaboration(ActorId(1));
    let before = a.document_file().actions.len();

    // A peer's whole presence: cursor, layer, a live stroke.
    let frame = PeerFrame {
        boot: 0,
        seq: 1,
        name: None,
        active_layer: LayerId(0),
        cursor: Some(Vec2::new(10.0, 10.0)),
        gesture: Some(GestureFrame::Selection {
            id: 0,
            op: SelectionOp::new(
                SelectionMode::Replace,
                SelectionShape::rect_from_corners(LEFT_MIN, LEFT_MAX),
                0.0,
            ),
        }),
        leaving: false,
    };
    a.merge_presence(ActorId(2), frame, 0.0);
    b.load_bytes(&a.save_bytes().expect("save")).expect("load");

    assert_eq!(
        a.document_file().actions.len(),
        before,
        "presence must not become an action"
    );
    assert!(
        !b.observe().has_selection,
        "a peer's *in-flight* marquee is not a committed selection"
    );
}
