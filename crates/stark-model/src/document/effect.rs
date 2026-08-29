//! **Which actions in a log are effective** — the undo algebra (§12.3).
//!
//! The third of the three facts the model owns about a log, beside *that* it folds
//! ([`fold`](super::fold)) and *which actions commute*
//! ([`footprint`](super::footprint)). It lived in `stark-engine`'s timeline, which is
//! the one place the split had left on the wrong side: not a line of it names a
//! `DocState`, an `ApplyCtx`, a tile or a renderer — it is `Action`, `ActionId` and
//! `ActorId` and nothing else, and the boundary was already reporting itself, since
//! [`ActionKind::Undo`](super::ActionKind::Undo)'s own doc had to cite it in prose
//! because an intra-doc link cannot point from the model into the engine.
//!
//! Two things follow. `stark-net`'s mirror can reason about an effective log without
//! ever naming an engine type, which is what §2 says the split bought; and the
//! property this most wants tested — random `(actor, lamport, kind)` logs with undo,
//! redo and late arrival, asserting that splicing agrees with rewind-and-replay —
//! becomes a test with no GPU and no `ApplyCtx` anywhere near it.
//!
//! # The rules
//!
//! An `Undo` is an ordinary logged action naming its target, so undo survives a
//! reload and replicates like anything else. What it *means* is resolved here rather
//! than in the fold: undo needs the whole log, not just the prior state.

use std::collections::{HashMap, HashSet};

use super::{Action, ActionId, ActionKind, ActorId};

/// Which action an [`ActionKind::Undo`] targets, if `action` is one.
pub fn undo_target_of(action: &Action) -> Option<ActionId> {
    match action.kind {
        ActionKind::Undo(target) => Some(target),
        _ => None,
    }
}

/// The set of action ids suppressed by *effective* `Undo` actions in `log`
/// (which must be sorted by [`ActionId`], the total order).
///
/// One descending pass suffices: an `Undo` always has a larger id than its
/// target (its author had seen the target, so its Lamport clock is larger), so
/// by the time we reach an action we already know whether it is undone. An
/// `Undo` that is itself undone (a redo happened) contributes nothing.
pub fn undone_ids(log: &[Action]) -> HashSet<ActionId> {
    let mut undone = HashSet::new();
    for action in log.iter().rev() {
        if let Some(target) = undo_target_of(action)
            && !undone.contains(&action.id)
        {
            undone.insert(target);
        }
    }
    undone
}

/// The materialization key of every **revived** ordinary action in `log`
/// (which must be sorted by id): the id of the latest effective redo (an
/// `Undo` of an `Undo`) that revived it.
///
/// This is what implements *redo-at-top* (§12.3): a redone action
/// re-materializes at the redo's own slot — the top of the stack as of the
/// redo — rather than its original position. For the redoing client that makes
/// redo a plain append; peers converge because the key is a pure function of
/// the shared log. The semantic trade is deliberate: a redone stroke lands
/// *over* work that happened while it was undone, not under it.
fn revival_keys(log: &[Action], undone: &HashSet<ActionId>) -> HashMap<ActionId, ActionId> {
    let by_id = |id: ActionId| {
        let pos = log.partition_point(|a| a.id < id);
        log.get(pos).filter(|a| a.id == id)
    };
    let mut keys: HashMap<ActionId, ActionId> = HashMap::new();
    for action in log {
        // An effective redo: an `Undo` of an `Undo` of an ordinary action.
        let Some(undo_id) = undo_target_of(action) else {
            continue;
        };
        if undone.contains(&action.id) {
            continue;
        }
        let Some(target_id) = by_id(undo_id).and_then(undo_target_of) else {
            continue;
        };
        if by_id(target_id).is_none_or(|t| undo_target_of(t).is_some()) {
            continue;
        }
        let key = keys.entry(target_id).or_insert(action.id);
        *key = (*key).max(action.id);
    }
    keys
}

/// Indices of the effective actions of a **sorted** log, in materialization
/// order: every ordinary action no effective `Undo` suppresses, ordered by its
/// own id — or, once a redo has revived it, by the reviving redo's id
/// ([`revival_keys`]).
///
/// Takes `undone` rather than deriving it, because every caller wants it for
/// something else too — the target resolution below, or its own filtering — and
/// the set is a whole pass over the log.
pub fn effective_indices(log: &[Action], undone: &HashSet<ActionId>) -> Vec<usize> {
    let keys = revival_keys(log, undone);
    let mut indices: Vec<usize> = log
        .iter()
        .enumerate()
        .filter(|(_, a)| undo_target_of(a).is_none() && !undone.contains(&a.id))
        .map(|(i, _)| i)
        .collect();
    indices.sort_by_key(|&i| {
        let id = log[i].id;
        keys.get(&id).copied().unwrap_or(id)
    });
    indices
}

/// The **effective sequence** of a shared action log (§12.3): the
/// actions that actually shape the document — every non-`Undo` action that no
/// effective `Undo` suppresses — in materialization order (total order by id,
/// except that a redone action sits at its reviving redo's slot, see
/// [`revival_keys`]). This is what gets materialized (and what a solo load
/// replays); the `Undo` actions themselves are resolved here and never reach
/// [`Materialize::fold`](super::Materialize::fold).
///
/// Sorts a copy of `log` by id first, so callers may pass a file's action list
/// as-is (solo logs are already ordered; shared saves are written in total
/// order, but files are external input).
pub fn effective_actions(log: &[Action]) -> Vec<Action> {
    let mut sorted: Vec<Action> = log.to_vec();
    sorted.sort_by_key(|a| a.id);
    let undone = undone_ids(&sorted);
    effective_indices(&sorted, &undone)
        .into_iter()
        .map(|i| sorted[i].clone())
        .collect()
}

/// What a local undo and redo would target.
///
/// Both are pure functions of `(log, actor, undone)`, which is what makes caching the
/// pair sound — `stark-engine`'s replicated timeline does, and its field says why.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Targets {
    pub undo: Option<ActionId>,
    pub redo: Option<ActionId>,
}

/// Resolve both targets against a **sorted** log and its `undone` set. A pure
/// function of the three, which is what makes caching it sound.
pub fn targets(log: &[Action], actor: ActorId, undone: &HashSet<ActionId>) -> Targets {
    let undo = undo_target(log, actor, undone);
    Targets {
        undo,
        redo: redo_target(log, actor, undone, undo),
    }
}

/// The most recent still-effective action *by* `actor` — what a local undo
/// should target. `Undo` actions themselves aren't candidates (they're redone
/// via [`redo_target`], not undone).
pub(crate) fn undo_target(
    log: &[Action],
    actor: ActorId,
    undone: &HashSet<ActionId>,
) -> Option<ActionId> {
    log.iter()
        .rev()
        .find(|a| a.id.actor == actor && undo_target_of(a).is_none() && !undone.contains(&a.id))
        .map(|a| a.id)
}

/// The `Undo` a local redo should suppress: `actor`'s most recent effective
/// `Undo` whose (non-`Undo`) target is currently undone — but only if it is more
/// recent than `latest_ordinary`, so a fresh edit "clears" the redo stack,
/// matching single-user expectations.
///
/// `latest_ordinary` is [`undo_target`]'s own answer, passed in rather than
/// recomputed: the action a local undo would take is exactly the one an edit has
/// to be newer than to count as clearing the stack, and the two spelling that
/// predicate separately is two places for it to drift.
pub(crate) fn redo_target(
    log: &[Action],
    actor: ActorId,
    undone: &HashSet<ActionId>,
    latest_ordinary: Option<ActionId>,
) -> Option<ActionId> {
    log.iter()
        .rev()
        .filter(|a| a.id.actor == actor && !undone.contains(&a.id))
        .take_while(|a| latest_ordinary.is_none_or(|m| a.id > m))
        .find(|a| {
            // An effective Undo whose target is an ordinary action that is
            // (still) undone. Undos-of-Undos are skipped: their effect is
            // already reflected in the effectiveness of what they target.
            undo_target_of(a).is_some_and(|t| {
                undone.contains(&t)
                    && action_by_id(log, t).is_some_and(|target| undo_target_of(target).is_none())
            })
        })
        .map(|a| a.id)
}

/// The action with this id in a **sorted** log.
fn action_by_id(log: &[Action], id: ActionId) -> Option<&Action> {
    let pos = log.partition_point(|a| a.id < id);
    log.get(pos).filter(|a| a.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Srgb;

    /// Logs here are built in increasing id order, which is the state
    /// [`ReplicatedTimeline`] keeps its own in — and what `action_by_id`'s binary
    /// search and `undone_ids`' single descending pass both rest on.
    fn resolve(log: &[Action], actor: u64) -> Targets {
        targets(log, ActorId(actor), &undone_ids(log))
    }

    fn id(lamport: u64, actor: u64) -> ActionId {
        ActionId {
            lamport,
            actor: ActorId(actor),
        }
    }

    /// An ordinary action — which kind is immaterial here; only `Undo` is read.
    fn edit(lamport: u64, actor: u64) -> Action {
        Action {
            id: id(lamport, actor),
            kind: ActionKind::SetSubstrateColor(Srgb::new([0.0; 3])),
        }
    }

    fn undo_of(lamport: u64, actor: u64, target: ActionId) -> Action {
        Action {
            id: id(lamport, actor),
            kind: ActionKind::Undo(target),
        }
    }

    #[test]
    fn undo_takes_this_actors_latest_and_never_anyone_elses() {
        let log = [edit(1, 1), edit(2, 2), edit(3, 1)];
        assert_eq!(
            resolve(&log, 1),
            Targets {
                undo: Some(id(3, 1)),
                redo: None
            }
        );
        assert_eq!(
            resolve(&log, 2),
            Targets {
                undo: Some(id(2, 2)),
                redo: None
            }
        );
        // An actor who has done nothing has nothing to undo, however full the log.
        assert_eq!(resolve(&log, 3), Targets::default());
    }

    #[test]
    fn an_undo_steps_the_target_back_and_offers_itself_as_the_redo() {
        let log = [edit(1, 1), edit(2, 1), undo_of(3, 1, id(2, 1))];
        assert_eq!(
            resolve(&log, 1),
            Targets {
                undo: Some(id(1, 1)),
                redo: Some(id(3, 1)),
            }
        );
    }

    /// Redo is an `Undo` of an `Undo` (§12.3): the revived action becomes
    /// undoable again, and there is nothing further to redo.
    #[test]
    fn a_redo_revives_its_target_and_empties_the_redo_stack() {
        let log = [
            edit(1, 1),
            edit(2, 1),
            undo_of(3, 1, id(2, 1)),
            undo_of(4, 1, id(3, 1)),
        ];
        assert_eq!(
            resolve(&log, 1),
            Targets {
                undo: Some(id(2, 1)),
                redo: None,
            }
        );
    }

    /// A fresh edit clears the redo stack, matching what a single-user timeline
    /// does when a commit truncates the future.
    #[test]
    fn an_edit_after_an_undo_clears_the_redo() {
        let log = [edit(1, 1), edit(2, 1), undo_of(3, 1, id(2, 1)), edit(5, 1)];
        assert_eq!(
            resolve(&log, 1),
            Targets {
                undo: Some(id(5, 1)),
                redo: None,
            }
        );
    }

    /// Undo is owned per actor (§12.3): a peer undoing your action leaves you
    /// with nothing to redo — it was not your undo to reverse.
    #[test]
    fn a_peers_undo_is_not_this_actors_redo() {
        let log = [edit(1, 1), undo_of(2, 2, id(1, 1))];
        assert_eq!(resolve(&log, 1), Targets::default());
        // …while its author may reverse it, which is what makes redo theirs.
        assert_eq!(resolve(&log, 2).redo, Some(id(2, 2)));
    }

    /// `insert`'s fast path skips deriving the effective sequence, on the claim
    /// that an ordinary action newer than the whole log lands at the end of that
    /// sequence and moves nothing else. Here is the claim, checked against the
    /// derivation it skips: same effective order, same undone set.
    #[test]
    fn appending_an_ordinary_action_only_extends_the_effective_sequence() {
        // A log with an undo and a redo in it, so the sequence being extended is
        // one whose effectiveness and slots were genuinely rearranged.
        let before = [
            edit(1, 1),
            edit(2, 2),
            edit(3, 1),
            undo_of(4, 1, id(3, 1)),
            edit(5, 2),
            undo_of(6, 1, id(4, 1)),
        ];
        let mut after = before.to_vec();
        after.push(edit(7, 2));

        let (u0, u1) = (undone_ids(&before), undone_ids(&after));
        assert_eq!(u0, u1, "an ordinary action suppresses nothing");

        let eff0 = effective_indices(&before, &u0);
        let eff1 = effective_indices(&after, &u1);
        assert_eq!(
            eff1.split_last().map(|(_, rest)| rest.to_vec()),
            Some(eff0),
            "the sequence it had is untouched…"
        );
        assert_eq!(
            eff1.last().map(|&i| after[i].id),
            Some(id(7, 2)),
            "…and the new action is what went on the end"
        );
    }

    /// The other half of the fast path: it resolves the targets itself rather
    /// than scanning, so its shortcut has to agree with the scan. A peer's commit
    /// moves neither target; ours becomes the undo target and clears the redo.
    #[test]
    fn appending_moves_only_its_own_actors_targets() {
        let before = [edit(1, 1), edit(2, 1), undo_of(3, 1, id(2, 1))];
        let ours = resolve(&before, 1);
        assert_eq!(
            ours.redo,
            Some(id(3, 1)),
            "a redo is on offer to start with"
        );

        // A peer's action: `retarget_appended` does nothing, so the scan must
        // agree that nothing moved.
        let mut peer = before.to_vec();
        peer.push(edit(4, 2));
        assert_eq!(
            resolve(&peer, 1),
            ours,
            "a peer's commit is not ours to undo"
        );

        // Ours: the shortcut says (undo = it, redo = none).
        let mut mine = before.to_vec();
        mine.push(edit(5, 1));
        assert_eq!(
            resolve(&mine, 1),
            Targets {
                undo: Some(id(5, 1)),
                redo: None,
            },
        );
    }
}
