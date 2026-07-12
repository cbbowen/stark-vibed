//! The timeline abstraction (DESIGN.md §5, §12).
//!
//! `Session`/`Engine` only ever see the [`Timeline`] trait, so the storage
//! strategy can change without touching them. [`LinearTimeline`] is the
//! single-user implementation (a thin wrapper over `history::History`);
//! [`ReplicatedTimeline`] is the multi-peer one — a totally-ordered, grow-only
//! action log (a replicated-log CRDT) materialized through the very same
//! `history::History` as a snapshot cache.

use std::collections::HashSet;

use history::{History, Version};

use super::action::{Action, ActionId, ActionKind, ActorId, ApplyCtx};
use super::state::DocState;

/// A versioned document: the source of the current [`DocState`] plus undo/redo.
pub trait Timeline {
    /// The current document state (`O(1)`).
    fn current(&self) -> &DocState;

    /// Commit an action, advancing the timeline. Clears any redo stack.
    fn push(&mut self, action: Action, ctx: &mut ApplyCtx);

    /// Step back one action if possible; returns whether anything was undone.
    fn undo(&mut self, ctx: &mut ApplyCtx) -> bool;

    /// Re-apply the most recently undone action; returns whether anything was redone.
    fn redo(&mut self, ctx: &mut ApplyCtx) -> bool;

    fn can_undo(&self) -> bool;
    fn can_redo(&self) -> bool;

    /// All committed actions, oldest to newest — the basis of the save format
    /// (DESIGN.md §8). For a replicated timeline this is the **full** log,
    /// including `Undo` actions and the actions they suppress — exactly the
    /// payload a joining peer needs (DESIGN.md §12.4).
    fn clone_actions(&self) -> Vec<Action>;

    /// Shared-mode undo (DESIGN.md §5.4): the action an "undo" should target by
    /// logging an [`ActionKind::Undo`], or `None` if undo is plain timeline
    /// navigation (the solo path). The engine asks this first and only falls
    /// back to [`Timeline::undo`] when it returns `None`.
    fn undo_as_action(&self) -> Option<ActionId> {
        None
    }

    /// Shared-mode redo: the **`Undo` action** a "redo" should itself undo
    /// (redo is an `Undo` of an `Undo`, DESIGN.md §12.3), or `None`.
    fn redo_as_action(&self) -> Option<ActionId> {
        None
    }

    /// Integrate an action authored elsewhere (DESIGN.md §12.1). Returns whether
    /// it was new (false = duplicate or unsupported by this timeline).
    fn merge(&mut self, _action: Action, _ctx: &mut ApplyCtx) -> bool {
        false
    }
}

/// Single-user timeline: a linear undo/redo stack over `history::History`.
pub struct LinearTimeline {
    history: History<Action>,
    /// Actions popped by `undo`, awaiting `redo`. Cleared on a fresh `push`.
    redo: Vec<Action>,
}

impl LinearTimeline {
    pub fn new(initial: DocState) -> Self {
        Self {
            history: History::new(initial),
            redo: Vec::new(),
        }
    }

    pub fn actions(&self) -> impl Iterator<Item = &Action> {
        self.history.actions()
    }
}

impl Timeline for LinearTimeline {
    fn current(&self) -> &DocState {
        self.history.last_state()
    }

    fn push(&mut self, action: Action, ctx: &mut ApplyCtx) {
        self.redo.clear();
        // Infallible apply (DESIGN.md §5) — no error to handle.
        self.history.push_action_with(action, ctx);
    }

    fn undo(&mut self, ctx: &mut ApplyCtx) -> bool {
        match self.history.pop_action_with(ctx) {
            Some(action) => {
                self.redo.push(action);
                true
            }
            None => false,
        }
    }

    fn redo(&mut self, ctx: &mut ApplyCtx) -> bool {
        match self.redo.pop() {
            Some(action) => {
                self.history.push_action_with(action, ctx);
                true
            }
            None => false,
        }
    }

    fn can_undo(&self) -> bool {
        self.history.last_version() != Version::default()
    }

    fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    fn clone_actions(&self) -> Vec<Action> {
        self.actions().cloned().collect()
    }
}

/// Which action an [`ActionKind::Undo`] targets, if `action` is one.
fn undo_target_of(action: &Action) -> Option<ActionId> {
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
fn undone_ids(log: &[Action]) -> HashSet<ActionId> {
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

/// The **effective sequence** of a shared action log (DESIGN.md §12.3): the
/// actions that actually shape the document — every non-`Undo` action that no
/// effective `Undo` suppresses — in total order. This is what gets materialized
/// (and what a solo load replays); the `Undo` actions themselves are resolved
/// here and never reach [`history::Action::apply`].
///
/// Sorts a copy of `log` by id first, so callers may pass a file's action list
/// as-is (solo logs are already ordered; shared saves are written in total
/// order, but files are external input).
pub fn effective_actions(log: &[Action]) -> Vec<Action> {
    let mut sorted: Vec<Action> = log.to_vec();
    sorted.sort_by_key(|a| a.id);
    let undone = undone_ids(&sorted);
    sorted
        .into_iter()
        .filter(|a| undo_target_of(a).is_none() && !undone.contains(&a.id))
        .collect()
}

/// Multi-peer timeline (DESIGN.md §12): a grow-only set of actions with the
/// total order given by [`ActionId`] `(lamport, actor)`. The canonical state is
/// the deterministic replay of the *effective* actions in that order; two peers
/// that have seen the same set of actions therefore compute identical pixels
/// (strong eventual consistency). `history::History` serves as the
/// materialization cache: when an action arrives out of order (or an `Undo`
/// changes effectiveness mid-log), we pop back to the first divergence and
/// replay forward — cost scales with how far back the change lands, kept
/// shallow by the history crate's dense snapshot retention (DESIGN.md §12.2).
pub struct ReplicatedTimeline {
    /// Whose undo/redo this timeline answers for ([`Timeline::undo_as_action`]).
    actor: ActorId,
    /// The full log, sorted by id, no duplicates. Grow-only.
    log: Vec<Action>,
    /// Membership index for O(1) duplicate rejection.
    ids: HashSet<ActionId>,
    /// Materialization of the effective sequence, in order (the initial empty
    /// document is `History`'s version 0, which pops never remove).
    history: History<Action>,
}

impl ReplicatedTimeline {
    /// Build from a full shared log (a joining peer's snapshot, or the local
    /// linear history at share time). Replays the effective sequence.
    pub fn from_log(
        actor: ActorId,
        initial: DocState,
        log: Vec<Action>,
        ctx: &mut ApplyCtx,
    ) -> Self {
        let mut this = Self {
            actor,
            log: Vec::new(),
            ids: HashSet::new(),
            history: History::new(initial),
        };
        let mut log = log;
        log.sort_by_key(|a| a.id);
        for action in log {
            if this.ids.insert(action.id) {
                this.log.push(action);
            }
        }
        this.resync(ctx);
        this
    }

    /// Insert one action (local commit or remote merge). Returns whether it was
    /// new. Idempotent: duplicates (gossip redelivery) are rejected by id.
    fn insert(&mut self, action: Action, ctx: &mut ApplyCtx) -> bool {
        if !self.ids.insert(action.id) {
            return false;
        }
        // Sorted insert; the common case (a fresh local or causally-newest
        // remote action) lands at the end.
        let pos = self.log.partition_point(|a| a.id < action.id);
        self.log.insert(pos, action);
        self.resync(ctx);
        true
    }

    /// Make `history` match the current effective sequence: find the first
    /// index where they diverge, pop back to it, replay forward. Untouched
    /// prefixes keep their snapshots (and their tiles' `Arc`s) as-is.
    fn resync(&mut self, ctx: &mut ApplyCtx) {
        let undone = undone_ids(&self.log);
        let effective: Vec<&Action> = self
            .log
            .iter()
            .filter(|a| undo_target_of(a).is_none() && !undone.contains(&a.id))
            .collect();

        let diverge = self
            .history
            .actions()
            .zip(effective.iter())
            .take_while(|(h, e)| h.id == e.id)
            .count();

        let mut materialized = self.history.actions().count();
        while materialized > diverge {
            self.history.pop_action_with(ctx);
            materialized -= 1;
        }
        for action in &effective[diverge..] {
            self.history.push_action_with((*action).clone(), ctx);
        }
    }

    /// The most recent still-effective action *by this actor* — what a local
    /// undo should target. `Undo` actions themselves aren't candidates (they're
    /// redone via [`Self::redo_target`], not undone).
    fn undo_target(&self) -> Option<ActionId> {
        let undone = undone_ids(&self.log);
        self.log
            .iter()
            .rev()
            .find(|a| {
                a.id.actor == self.actor
                    && undo_target_of(a).is_none()
                    && !undone.contains(&a.id)
            })
            .map(|a| a.id)
    }

    /// The `Undo` a local redo should suppress: this actor's most recent
    /// effective `Undo` whose (non-`Undo`) target is currently undone —
    /// but only if it is more recent than this actor's latest effective
    /// ordinary action, so a fresh edit "clears" the redo stack, matching
    /// single-user expectations.
    fn redo_target(&self) -> Option<ActionId> {
        let undone = undone_ids(&self.log);
        let latest_ordinary = self
            .log
            .iter()
            .rev()
            .find(|a| {
                a.id.actor == self.actor
                    && undo_target_of(a).is_none()
                    && !undone.contains(&a.id)
            })
            .map(|a| a.id);
        self.log
            .iter()
            .rev()
            .filter(|a| a.id.actor == self.actor && !undone.contains(&a.id))
            .take_while(|a| latest_ordinary.is_none_or(|m| a.id > m))
            .find(|a| {
                // An effective Undo whose target is an ordinary action that is
                // (still) undone. Undos-of-Undos are skipped: their effect is
                // already reflected in the effectiveness of what they target.
                undo_target_of(a).is_some_and(|t| {
                    undone.contains(&t)
                        && self
                            .action_by_id(t)
                            .is_some_and(|target| undo_target_of(target).is_none())
                })
            })
            .map(|a| a.id)
    }

    fn action_by_id(&self, id: ActionId) -> Option<&Action> {
        let pos = self.log.partition_point(|a| a.id < id);
        self.log.get(pos).filter(|a| a.id == id)
    }
}

impl Timeline for ReplicatedTimeline {
    fn current(&self) -> &DocState {
        self.history.last_state()
    }

    fn push(&mut self, action: Action, ctx: &mut ApplyCtx) {
        self.insert(action, ctx);
    }

    /// Navigation undo doesn't exist in a shared session — undo is a logged
    /// action so peers can order it (DESIGN.md §12.3). The engine routes
    /// through [`Timeline::undo_as_action`] first, so this is unreachable in
    /// practice; it conservatively does nothing.
    fn undo(&mut self, _ctx: &mut ApplyCtx) -> bool {
        false
    }

    fn redo(&mut self, _ctx: &mut ApplyCtx) -> bool {
        false
    }

    fn can_undo(&self) -> bool {
        self.undo_target().is_some()
    }

    fn can_redo(&self) -> bool {
        self.redo_target().is_some()
    }

    fn clone_actions(&self) -> Vec<Action> {
        self.log.clone()
    }

    fn undo_as_action(&self) -> Option<ActionId> {
        self.undo_target()
    }

    fn redo_as_action(&self) -> Option<ActionId> {
        self.redo_target()
    }

    fn merge(&mut self, action: Action, ctx: &mut ApplyCtx) -> bool {
        self.insert(action, ctx)
    }
}
