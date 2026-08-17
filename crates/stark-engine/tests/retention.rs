//! History retention (§5): the engine gives up **undo depth** under memory
//! pressure, and never gives up any of the document.
//!
//! The distinction is the whole subject. `DocState` is cheap to clone and tiles are
//! copy-on-write, so history retention drives GPU memory reclamation for free — but
//! only if something ever retires history, and for a long time nothing did. What
//! makes retiring it *safe* is that the log is the document (§1, §8): folding an
//! action out of the undo stack must not fold it out of the file, the timelapse, or
//! what a joining peer is sent.
//!
//! So the load-bearing test here is the round trip. A version of this feature that
//! shortened `clone_actions` would pass every "did memory go down" check and quietly
//! write files missing their early strokes, with the in-memory picture still showing
//! the whole painting.

mod common;

use common::{brush, engine_or_skip, stroke_with};
use stark_engine::command::{DocCommand, ViewCommand};
use stark_engine::geom::Vec2;
use stark_engine::{Engine, RgbaImage};

/// Enough strokes to build a history worth folding, spread so they touch different
/// tiles — retention only pins a tile version that a *later* action replaced.
fn paint_a_session(engine: &mut Engine, strokes: usize) {
    for i in 0..strokes {
        let x = 20.0 + (i % 8) as f32 * 24.0;
        let y = 20.0 + (i / 8) as f32 * 24.0;
        let b = brush([0.9, 0.1, 0.1, 1.0], 4.0);
        stroke_with(engine, b, &[Vec2::new(x, y), Vec2::new(x + 16.0, y + 16.0)]);
    }
}

fn pixels(engine: &mut Engine) -> RgbaImage {
    engine.render_to_image()
}

/// Every byte of two images, so "the same picture" means the same picture.
fn identical(a: &RgbaImage, b: &RgbaImage) -> bool {
    a.width == b.width && a.height == b.height && a.pixels == b.pixels
}

/// **The document survives what the undo stack does not.**
///
/// The test the unsafe version of this feature would have failed: fold the history
/// under a budget of zero, then save, load into a fresh engine, and require the
/// picture to be identical. A timeline that shortened its log instead of keeping a
/// folded prefix would come back missing its early strokes — and nothing about the
/// engine that saved it would have looked wrong, because its in-memory state still
/// held the whole painting.
#[test]
fn folding_history_does_not_change_the_saved_document() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // Trim as hard as the floor allows, on every commit.
    engine.process(ViewCommand::SetHistoryBudget(0));
    paint_a_session(&mut engine, 48);

    let before = pixels(&mut engine);
    let scrub = engine.scrub_range().expect("a solo document scrubs");
    assert!(
        scrub.0 < 48,
        "nothing was folded ({} steps of 48 still undoable), so this proves nothing",
        scrub.0,
    );

    let bytes = engine.save_bytes().expect("save");
    let Some(mut reopened) = engine_or_skip() else {
        return;
    };
    reopened.load_bytes(&bytes).expect("load");
    let after = pixels(&mut reopened);

    assert!(
        identical(&before, &after),
        "a folded session saved a different painting than it was showing",
    );
}

/// And the same document is what a **collaborator** would be sent.
///
/// `start_collaboration` replays `clone_actions()` from an empty document, so a log
/// that had lost its folded prefix would hand every joiner a different painting —
/// the same failure as the file, reached by the other door.
#[test]
fn folding_history_does_not_change_what_a_peer_is_sent() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.process(ViewCommand::SetHistoryBudget(0));
    paint_a_session(&mut engine, 48);

    let before = pixels(&mut engine);
    engine.start_collaboration(stark_engine::document::ActorId(7));
    let after = pixels(&mut engine);

    assert!(
        identical(&before, &after),
        "re-materializing the shared log produced a different painting",
    );
}

/// Undo keeps working up to the fold, and stops cleanly at it.
///
/// Both halves matter. A folded history that still reported `can_undo` would light
/// an Undo button that does nothing — which is what comparing against
/// `Version::default()` did, latent until folding existed.
#[test]
fn undo_reaches_the_fold_and_stops() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.process(ViewCommand::SetHistoryBudget(0));
    paint_a_session(&mut engine, 48);

    let reach = engine.scrub_range().expect("scrubs").0;
    assert!(reach > 0, "everything was folded, leaving nothing to check");

    // Every remaining step undoes.
    for i in 0..reach {
        assert!(
            engine.observe().can_undo,
            "step {i} of {reach} reported nothing to undo",
        );
        engine.process(DocCommand::Undo);
    }
    // And then it stops, saying so.
    assert!(
        !engine.observe().can_undo,
        "a fully unwound folded history still offers an undo",
    );
    // Asking anyway is a no-op rather than a panic or a silent corruption.
    let at_floor = pixels(&mut engine);
    engine.process(DocCommand::Undo);
    assert!(
        identical(&at_floor, &pixels(&mut engine)),
        "an undo past the fold changed the document",
    );
}

/// **The default budget does not trim an ordinary session.** Retention is meant to
/// be invisible until memory is actually tight; a policy that folded the undo stack
/// of a fifty-stroke document would be a bug, not a tuning choice.
#[test]
fn an_ordinary_session_keeps_its_whole_undo_stack() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint_a_session(&mut engine, 48);
    assert_eq!(
        engine.scrub_range().expect("scrubs").0,
        48,
        "the default budget folded an ordinary session's history",
    );
}

/// A shared session never folds, whatever the budget says — its document is
/// re-materialized from the whole log on every arriving action (§12.2), so there is
/// nothing there to fold into a base state. Structural: `Timeline::forget_oldest`
/// defaults to folding nothing and `ReplicatedTimeline` does not override it.
#[test]
fn a_shared_session_never_folds() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.start_collaboration(stark_engine::document::ActorId(3));
    engine.process(ViewCommand::SetHistoryBudget(0));
    paint_a_session(&mut engine, 48);

    // A replicated timeline reports no scrub range at all, which is the same refusal
    // — so the check is that the log kept every action.
    let log = engine.save_bytes().expect("save");
    let file = stark_engine::DocumentFile::from_bytes(&log).expect("decode");
    assert_eq!(
        file.actions.len(),
        48,
        "a shared log lost actions to a fold it should have declined",
    );
}
