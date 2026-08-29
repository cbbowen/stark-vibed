//! Drawing guides as **document state** (§20.5), and the one half of a guide
//! that is not.
//!
//! A perspective set up over a drawing is part of its construction, so a guide is
//! logged like a layer: saved with the file, replicated to peers, and reachable by
//! undo. What stayed per-client is whether *you* are looking at one, which
//! defaults to **not**: an eye must not reach across a session, must not be
//! saved, and must not cost an undo step — and a document that carries every
//! perspective ever built over it must not lay them all on the canvas when it
//! opens.
//!
//! Those are four claims about where a guide lives, and each of them is silent
//! when it breaks: a guide that failed to save looks exactly like a guide, right
//! up until the file is reopened. So each has a test here, asked of the roster the
//! engine projects rather than of pixels — a guide draws chrome, and chrome never
//! reaches an exported image (§20.4), which is precisely why no golden can see any
//! of this.
//!
//! The geometry these guides describe is `stark-model`'s own concern and is
//! covered there; what is tested here is only where the roster lives.

mod common;

use common::engine_or_skip;
use stark_engine::Engine;
use stark_engine::command::{DocCommand, ViewCommand};
use stark_model::document::{ActorId, GuideId, Lens, PerspectiveGuide};
use stark_model::geom::Vec2;

/// A guide that is nothing like the default, so a roster that quietly rebuilt one
/// from scratch would not pass for it.
fn distinctive() -> PerspectiveGuide {
    PerspectiveGuide {
        center: Vec2::new(37.0, -104.0),
        focal: 612.0,
        lens: Lens::Fisheye,
        opacity: 0.42,
        pairs: [true, false, true],
        ..PerspectiveGuide::default()
    }
}

/// The **document's** half of the roster: what is saved, replicated and undoable.
///
/// Deliberately without the guide visibility. Two engines looking at one document agree about
/// every row of this and are free to disagree about what is on screen, which is
/// the whole shape of §20.5 — so a test that compares rosters says so by
/// comparing this, and a test about guide visibility asks [`visible_guides`] instead.
fn roster(e: &Engine) -> Vec<(GuideId, Option<String>, PerspectiveGuide)> {
    e.observe()
        .guides
        .iter()
        .map(|g| (g.id, g.name.as_deref().map(str::to_owned), g.guide))
        .collect()
}

/// The **client's** half: which of those rows this engine is drawing.
fn visible_guides(e: &Engine) -> Vec<bool> {
    e.observe().guides.iter().map(|g| g.visible).collect()
}

/// Add one guide and answer its id — which the engine mints no counter for: a
/// guide's identity is the id of the action that added it (§20.5), so it is found
/// on the roster afterwards rather than returned.
///
/// The eye stays **shut**, which is what this door does and is worth knowing when
/// reading the tests below: opening one is the frontend picking a guide up to
/// shape it (`panels::guides`' `begin_guide_edit`), not a consequence of the
/// guide existing.
fn add(e: &mut Engine, guide: PerspectiveGuide, name: Option<&str>) -> GuideId {
    let after = e.observe().guides.last().map(|g| g.id);
    e.process(DocCommand::AddGuide {
        guide,
        after,
        name: name.map(str::to_owned),
    });
    e.observe().guides.last().expect("the guide just added").id
}

/// Add a guide and open this client's eye on it — what the panel's "Add
/// Perspective" amounts to, in the two commands it is made of.
fn add_and_show(e: &mut Engine, guide: PerspectiveGuide, name: Option<&str>) -> GuideId {
    let id = add(e, guide, name);
    e.process(ViewCommand::SetGuideVisible(id, true));
    id
}

/// **Sharing a document does not disturb the guides it already had.**
///
/// `start_collaboration` rewrites every solo-authored action's `ActorId` to the
/// sharer's, once, before any peer has seen them (§12.3). A `GuideId` used to be
/// *derived* from the action id inside the fold rather than carried in the action, so
/// that rewrite moved it — while every `RemoveGuide`, `SetGuide`, `SetGuideName` and
/// `MoveGuide` in the same log went on naming the old one, and each of those no-ops on
/// an id it cannot find. Pressing Share therefore reverted every guide edit made
/// before it, brought back every deleted guide, and closed every open eye — because
/// `visible_guides` is a set of ids too.
///
/// Every other test in this file shares *first* and adds guides after, which is why
/// none of them saw it. This one is deliberately the other order.
#[test]
fn guides_survive_the_moment_a_document_is_shared() {
    let Some(mut e) = engine_or_skip() else {
        return;
    };

    // A guide that is edited, one that is deleted, and one that is merely renamed —
    // the three later actions that name a guide by id.
    let edited = add_and_show(&mut e, PerspectiveGuide::default(), Some("the ground"));
    e.process(DocCommand::SetGuide(edited, distinctive()));
    let doomed = add(&mut e, PerspectiveGuide::default(), Some("a mistake"));
    e.process(DocCommand::RemoveGuide(doomed));
    let renamed = add(&mut e, PerspectiveGuide::default(), None);
    e.process(DocCommand::SetGuideName(renamed, Some("the wall".into())));

    let before = roster(&e);
    let eyes = visible_guides(&e);
    assert_eq!(before.len(), 2, "one guide was removed before sharing");

    e.start_collaboration(ActorId(1));

    assert_eq!(
        roster(&e),
        before,
        "sharing changed the guides: an id the later actions name moved under them",
    );
    assert_eq!(
        visible_guides(&e),
        eyes,
        "sharing closed an eye, which is a set of ids and not of guides",
    );
}

/// **Saved.** A guide survives the round trip through the file, camera, name and
/// arrangement alike — because it is in the log, which *is* the save format (§8).
#[test]
fn guides_survive_save_load() {
    let (Some(mut a), Some(mut b)) = (engine_or_skip(), engine_or_skip()) else {
        return;
    };
    add_and_show(&mut a, distinctive(), Some("the ground"));
    add_and_show(&mut a, PerspectiveGuide::default(), None);
    let before = roster(&a);
    assert_eq!(before.len(), 2);

    let bytes = a.save_bytes().expect("save");
    b.load_bytes(&bytes).expect("load");
    assert_eq!(
        roster(&b),
        before,
        "the roster did not survive the file, and nothing about the pixels would say so"
    );
    // …and it arrives with both guides hidden, though the client that saved it had
    // both visible. A document carries every perspective ever built over it (§20.5),
    // so opening one must not lay them all on the canvas at once.
    assert_eq!(visible_guides(&a), vec![true, true]);
    assert_eq!(
        visible_guides(&b),
        vec![false, false],
        "a saved eye came back open"
    );
}

/// **Replicated.** Everything about a guide reaches a peer: its arrival, its
/// camera, its name, and its removal.
///
/// Asserted on the *whole* roster rather than on a field, because a partial
/// replication is the failure that reads as working — the guide is there, and the
/// artist on the other end is constructing against a different perspective.
#[test]
fn guides_reach_a_peer() {
    let (Some(mut a), Some(mut b)) = (engine_or_skip(), engine_or_skip()) else {
        return;
    };
    a.start_collaboration(ActorId(1));
    b.join_collaboration(&a.document_file(), ActorId(2))
        .expect("join a session this build can render");

    let first = add(&mut a, distinctive(), Some("the ground"));
    let second = add(&mut a, PerspectiveGuide::default(), None);
    a.process(DocCommand::SetGuideName(
        second,
        Some("  the wall  ".into()),
    ));
    for action in a.take_outbox() {
        b.merge_remote(action);
    }
    assert_eq!(roster(&b), roster(&a), "a peer sees a different roster");
    assert_eq!(
        roster(&b)[1].1.as_deref(),
        Some("the wall"),
        "a name reached the peer un-normalized"
    );

    // The eye is the one thing that did not travel: `a` opened both, `b` sees the
    // guides and draws neither until it asks (§20.5).
    a.process(ViewCommand::SetGuideVisible(first, true));
    a.process(ViewCommand::SetGuideVisible(second, true));
    assert_eq!(visible_guides(&a), vec![true, true]);
    assert_eq!(
        visible_guides(&b),
        vec![false, false],
        "an eye crossed the wire"
    );

    // …and a removal is a fact about the document like the rest of it.
    a.process(DocCommand::RemoveGuide(first));
    for action in a.take_outbox() {
        b.merge_remote(action);
    }
    assert_eq!(roster(&b), roster(&a));
    assert_eq!(roster(&b).len(), 1);
}

/// **Undoable.** One undo step per settled edit, and the step puts back exactly
/// what was there — a camera, a name, an arrangement, a guide's whole existence.
///
/// Walked backwards through four different kinds of edit rather than tested on
/// one, because the patch that services an undo is driven by the footprint's write
/// list (`document::patch`), and `Resource::Guides` is one coarse resource for all
/// of them: a mistake there restores the roster to the wrong *epoch* rather than
/// to the wrong field, which one edit undone cannot show.
#[test]
fn every_guide_edit_is_one_undo_step() {
    let Some(mut e) = engine_or_skip() else {
        return;
    };
    let first = add(&mut e, distinctive(), Some("the ground"));
    let added = roster(&e);
    let second = add(&mut e, PerspectiveGuide::default(), None);
    let two = roster(&e);

    e.process(DocCommand::SetGuide(first, PerspectiveGuide::default()));
    let reshaped = roster(&e);
    assert_ne!(reshaped, two, "the reshape changed nothing");

    e.process(DocCommand::SetGuideName(first, None));
    let renamed = roster(&e);
    assert_eq!(renamed[0].1, None, "the name was not taken away");

    e.process(DocCommand::MoveGuide {
        id: first,
        after: Some(second),
    });
    let moved = roster(&e);
    assert_eq!(
        moved.iter().map(|g| g.0).collect::<Vec<_>>(),
        vec![second, first],
        "the move did not reorder the roster"
    );

    // Back up the ladder, one rung at a time.
    for want in [renamed, reshaped, two, added, Vec::new()] {
        assert!(e.observe().can_undo, "the edit spent no undo step");
        e.process(DocCommand::Undo);
        assert_eq!(e.observe().guides.len(), want.len());
        assert_eq!(roster(&e), want, "an undo put back the wrong roster");
    }
}

/// **Not** saved, replicated or undoable: the eye.
///
/// The complement of the three above, and the reason a guide is two things kept in
/// two places (§20.5). Opening one must not travel, must not be written to the
/// file, and must not be something an undo takes back — an artist who opened a
/// guide to look at it and then pressed Ctrl-Z expects their *last edit* back, not
/// the guide.
#[test]
fn an_eye_is_this_client_s_alone() {
    let (Some(mut a), Some(mut b)) = (engine_or_skip(), engine_or_skip()) else {
        return;
    };
    a.start_collaboration(ActorId(1));
    b.join_collaboration(&a.document_file(), ActorId(2))
        .expect("join a session this build can render");
    let id = add(&mut a, distinctive(), Some("the ground"));
    for action in a.take_outbox() {
        b.merge_remote(action);
    }
    assert_eq!(
        visible_guides(&a),
        vec![false],
        "a guide arrived already drawn"
    );
    assert_eq!(visible_guides(&b), vec![false]);

    a.process(ViewCommand::SetGuideVisible(id, true));
    assert_eq!(visible_guides(&a), vec![true], "the eye did not open");
    assert!(
        a.take_outbox().is_empty(),
        "opening an eye put something on the wire"
    );
    assert_eq!(
        visible_guides(&b),
        vec![false],
        "a peer's guide lit up with ours"
    );

    // Not in the file either: a document written while it was open reloads shut.
    let Some(mut c) = engine_or_skip() else {
        return;
    };
    c.load_bytes(&a.save_bytes().expect("save")).expect("load");
    assert_eq!(roster(&c).len(), 1, "the guide did not survive the file");
    assert_eq!(
        visible_guides(&c),
        vec![false],
        "an open eye was saved with the document"
    );

    // And not an undo step. Nothing has been logged since the add, so the one undo
    // available is the add itself — which is exactly the point: the eye did not
    // interpose a step between the artist and their last real edit.
    a.process(DocCommand::Undo);
    assert!(
        roster(&a).is_empty(),
        "the undo took back the eye instead of the edit"
    );
}

/// An open eye is remembered **through** the guide going away and coming back.
///
/// The state is a set of opened ids, so a removal leaves the id in it, and that is
/// deliberate: a removal can be undone, and a guide that shut its own eye on the
/// way back would be the tool changing what the artist is looking at — and here
/// that would mean the guide returning invisible, which reads as the undo having
/// only half worked.
#[test]
fn an_open_eye_survives_an_undone_removal() {
    let Some(mut e) = engine_or_skip() else {
        return;
    };
    let id = add_and_show(&mut e, distinctive(), None);
    e.process(DocCommand::RemoveGuide(id));
    assert!(e.observe().guides.is_empty());

    e.process(DocCommand::Undo);
    let back = roster(&e);
    assert_eq!(back.len(), 1, "the undo did not put the guide back");
    assert_eq!(back[0].0, id, "it came back under a different id");
    assert_eq!(
        visible_guides(&e),
        vec![true],
        "the guide came back with its eye shut"
    );
}

/// Two guides added by two peers at the same moment take different ids, with no
/// counter resynced anywhere to make it so (§20.5, §17.9).
///
/// The property is structural — a `GuideId` *is* an `ActionId`, and two actions
/// cannot share one — which is exactly why it is worth a test that a
/// counter-shaped answer would fail: concurrent adds are the case a per-client
/// counter gets wrong, and the way it goes wrong is two guides under one id, with
/// the roster answering with whichever comes first.
#[test]
fn concurrent_adds_cannot_collide() {
    let (Some(mut a), Some(mut b)) = (engine_or_skip(), engine_or_skip()) else {
        return;
    };
    a.start_collaboration(ActorId(1));
    b.join_collaboration(&a.document_file(), ActorId(2))
        .expect("join a session this build can render");

    // Same lamport on both sides: neither has heard the other.
    add(&mut a, distinctive(), Some("mine"));
    add(&mut b, PerspectiveGuide::default(), Some("theirs"));
    let (out_a, out_b) = (a.take_outbox(), b.take_outbox());
    for action in out_a {
        b.merge_remote(action);
    }
    for action in out_b {
        a.merge_remote(action);
    }

    let ids: Vec<GuideId> = roster(&a).iter().map(|g| g.0).collect();
    assert_eq!(ids.len(), 2, "one add landed on top of the other");
    assert_ne!(ids[0], ids[1], "two guides minted the same id");
    assert_eq!(roster(&a), roster(&b), "the two peers did not converge");
}
