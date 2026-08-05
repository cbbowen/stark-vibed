//! The timeline abstraction (§5, §12).
//!
//! `Session`/`Engine` only ever see the [`Timeline`] trait, so the storage
//! strategy can change without touching them. [`LinearTimeline`] is the
//! single-user implementation (a thin wrapper over `history::History`);
//! [`ReplicatedTimeline`] is the multi-peer one — a totally-ordered, grow-only
//! action log (a replicated-log CRDT) materialized through the very same
//! `history::History` as a snapshot cache.

use std::collections::{HashMap, HashSet};

use history::{History, Version};

use super::action::{Action, ActionId, ActionKind, ActorId, ApplyCtx};
use super::footprint::footprint;
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
    /// (§8). For a replicated timeline this is the **full** log,
    /// including `Undo` actions and the actions they suppress — exactly the
    /// payload a joining peer needs (§12.4).
    fn clone_actions(&self) -> Vec<Action>;

    /// Shared-mode undo (§5.4): the action an "undo" should target by
    /// logging an [`ActionKind::Undo`], or `None` if undo is plain timeline
    /// navigation (the solo path). The engine asks this first and only falls
    /// back to [`Timeline::undo`] when it returns `None`.
    fn undo_as_action(&self) -> Option<ActionId> {
        None
    }

    /// Shared-mode redo: the **`Undo` action** a "redo" should itself undo
    /// (redo is an `Undo` of an `Undo`, §12.3), or `None`.
    fn redo_as_action(&self) -> Option<ActionId> {
        None
    }

    /// Integrate an action authored elsewhere (§12.1). Returns whether
    /// it was new (false = duplicate or unsupported by this timeline).
    fn merge(&mut self, _action: Action, _ctx: &mut ApplyCtx) -> bool {
        false
    }

    /// Where the playhead stands and how far it can travel — `(applied, total)`,
    /// both counted in actions — or `None` for a timeline that cannot be scrubbed
    /// (§18.2.4).
    ///
    /// `None` rather than `(n, n)` because the two say different things: a
    /// timeline with nowhere to go is still a timeline, while a
    /// [`ReplicatedTimeline`] has no single playhead to move at all — its
    /// materialization is a function of a log that peers are still appending to,
    /// so a scrub would be undone by the next arrival. The frontend needs to tell
    /// "there is no history yet" from "this document's history is not yours alone
    /// to walk", and only the second is a reason to say so.
    fn scrub_range(&self) -> Option<(usize, usize)> {
        None
    }

    /// Move the playhead to `to` (clamped to the range), applying or withdrawing
    /// whatever lies between. Returns whether the document changed.
    ///
    /// The withdrawn actions are *kept*, in the same place undo keeps them, which
    /// is the whole reason this is navigation rather than deletion: scrubbing back
    /// and forward is lossless, and committing a fresh edit at a scrubbed-back
    /// position truncates the future exactly as painting after an undo does.
    fn seek(&mut self, _to: usize, _ctx: &mut ApplyCtx) -> bool {
        false
    }

    /// A caption per action, oldest first and spanning the **whole** range
    /// [`scrub_range`](Self::scrub_range) reports — the withdrawn ones included,
    /// since a scrubber has to label the steps it can travel *to*.
    fn scrub_labels(&self) -> Vec<&'static str> {
        Vec::new()
    }

    /// How materializations have been serviced (§12.6). Solo timelines
    /// report zeros — the counters exist for the replicated fast paths.
    fn stats(&self) -> TimelineStats {
        TimelineStats::default()
    }
}

/// Counters for how the replicated timeline absorbed log changes
/// (§12.6) — the observable difference between the commutation fast path and a
/// rewind-and-replay, which pixels alone can't show (that's the point).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct TimelineStats {
    /// Undos absorbed by shifting the target out through a fully-commuting
    /// suffix — no replay, no re-render at all.
    pub fast_removes: u64,
    /// Changes that replayed something: a partially-commuting undo, a
    /// concurrent arrival landing mid-sequence, a joining peer's whole log
    /// (§12.2).
    pub rebuilds: u64,
    /// Actions re-applied by those rebuilds — the work the fast path avoids.
    pub replayed: u64,
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

    /// How many actions are currently applied — the playhead's position.
    ///
    /// Asked of the history rather than tracked beside it: a second copy of a
    /// number the history already knows is a number that can disagree with it,
    /// and `Version` is opaque, so there was nothing else to ask.
    ///
    /// That principle used to cost a walk per call, which
    /// [`scrub_range`](Timeline::scrub_range) makes a walk per render. It costs
    /// nothing now — `actions()` reports its own length — so the answer that
    /// cannot go stale is also the cheap one, and neither has to be traded for
    /// the other.
    fn applied(&self) -> usize {
        self.history.actions().len()
    }
}

impl Timeline for LinearTimeline {
    fn current(&self) -> &DocState {
        self.history.last_state()
    }

    fn push(&mut self, action: Action, ctx: &mut ApplyCtx) {
        self.redo.clear();
        // Infallible apply (§5) — no error to handle.
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

    fn scrub_range(&self) -> Option<(usize, usize)> {
        let applied = self.applied();
        Some((applied, applied + self.redo.len()))
    }

    /// Scrubbing **is** the undo/redo split, moved in bulk rather than one step at
    /// a time (§18.2.4). Nothing new is stored: the applied prefix
    /// and the withheld suffix are the two halves this type already had, so a
    /// scrub leaves the timeline in a state undo could equally have produced —
    /// which is what makes it safe to paint from wherever the playhead was left.
    ///
    /// Backwards goes through `pop_actions_with`, which rebuilds the snapshot
    /// cache for the shorter history *once* instead of computing and discarding
    /// the intermediate cache states of `k` successive pops. That is the
    /// difference between a drag to the start of a long session being one rebuild
    /// and being one per stroke crossed.
    ///
    /// Forwards has no such shortcut and wants none: re-applying an action is
    /// re-rendering its stroke on the GPU, which is the work itself rather than
    /// bookkeeping around it.
    fn seek(&mut self, to: usize, ctx: &mut ApplyCtx) -> bool {
        let applied = self.applied();
        let to = to.min(applied + self.redo.len());
        match to.cmp(&applied) {
            std::cmp::Ordering::Less => {
                // `pop_actions_with` yields newest-first, which is exactly the
                // order successive `undo`s push onto `redo` — so the stack stays
                // one thing however it was filled.
                let popped = self.history.pop_actions_with(applied - to, ctx);
                self.redo.extend(popped);
                true
            }
            std::cmp::Ordering::Greater => {
                // The oldest withheld action sits at the *end* of `redo` (it is
                // the next one `redo()` would pop), so the batch to re-apply is
                // the tail, taken in reverse.
                let batch: Vec<Action> = self
                    .redo
                    .drain(self.redo.len() - (to - applied)..)
                    .rev()
                    .collect();
                for action in batch {
                    self.history.push_action_with(action, ctx);
                }
                true
            }
            std::cmp::Ordering::Equal => false,
        }
    }

    fn scrub_labels(&self) -> Vec<&'static str> {
        self.history
            .actions()
            .map(|a| a.kind.label())
            // The withheld half is stored newest-first; the scrubber reads
            // oldest-first throughout.
            .chain(self.redo.iter().rev().map(|a| a.kind.label()))
            .collect()
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
fn effective_indices(log: &[Action], undone: &HashSet<ActionId>) -> Vec<usize> {
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
/// [`history::Action::apply`].
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

/// What a local undo and redo would target — the pair
/// [`ReplicatedTimeline`] caches, so see its field for *why* it is cached.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
struct Targets {
    undo: Option<ActionId>,
    redo: Option<ActionId>,
}

/// Resolve both targets against a **sorted** log and its `undone` set. A pure
/// function of the three, which is what makes caching it sound.
fn targets(log: &[Action], actor: ActorId, undone: &HashSet<ActionId>) -> Targets {
    let undo = undo_target(log, actor, undone);
    Targets {
        undo,
        redo: redo_target(log, actor, undone, undo),
    }
}

/// The most recent still-effective action *by* `actor` — what a local undo
/// should target. `Undo` actions themselves aren't candidates (they're redone
/// via [`redo_target`], not undone).
fn undo_target(log: &[Action], actor: ActorId, undone: &HashSet<ActionId>) -> Option<ActionId> {
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
fn redo_target(
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

/// Multi-peer timeline (§12): a grow-only set of actions with the
/// total order given by [`ActionId`] `(lamport, actor)`. The canonical state is
/// the deterministic replay of the *effective* actions in that order; two peers
/// that have seen the same set of actions therefore compute identical pixels
/// (strong eventual consistency). `history::History` serves as the
/// materialization cache: when an action arrives out of order (or an `Undo`
/// changes effectiveness mid-log), we pop back to the first divergence and
/// replay forward — cost scales with how far back the change lands, kept
/// shallow by the history crate's dense snapshot retention (§12.2).
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
    stats: TimelineStats,
    /// What a local undo and redo would target as the log now stands — a pure
    /// function of [`log`](Self::log) and [`actor`](Self::actor), resolved once
    /// per log change in [`resync`](Self::resync).
    ///
    /// Cached because the *questions* are asked far more often than the log
    /// changes. `can_undo`/`can_redo` reach `Engine::observe`, which the frontend
    /// refreshes after every command — including the pointer samples of a stroke
    /// in flight, which arrive at digitizer rate and commit nothing at all.
    /// Answering from scratch there meant two backwards passes over the whole log
    /// per pen event, each building a fresh [`undone_ids`] set, and each getting
    /// slower as the session grew. The log itself is untouched for the whole
    /// gesture, so every one of those passes recomputed the same two answers.
    ///
    /// A derived copy is a thing that can disagree with what it was derived from,
    /// so this is not kept in step by remembering to: the log is written in
    /// exactly one place ([`insert`](Self::insert), which
    /// [`from_log`](Self::from_log) also funnels through), that place calls
    /// `resync`, and `resync` writes this from the same `undone` set it
    /// materializes against — before any of its own early returns.
    targets: Targets,
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
            stats: TimelineStats::default(),
            targets: Targets::default(),
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
    /// new. Idempotent: duplicates (redelivery by the transport) are rejected by id.
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

    /// Make `history` match the current effective sequence.
    ///
    /// The change is almost always a single action entering or leaving the
    /// sequence (one call per log insert), so this classifies it
    /// (§12.6):
    ///
    /// - an action **removed** (an undo landed) is handed to the history's
    ///   `remove_action_with`, which shifts it past everything it commutes with
    ///   (via [`Footprint`], its `Centralizer`) using [`Action`]'s `inverse` —
    ///   no re-render at all when the whole suffix commutes, and a replay only
    ///   of what sits past the first conflict otherwise;
    /// - an action **appended** (a fresh local commit, a causally-newest remote
    ///   arrival, or a redo — which materializes at the *top* of the stack,
    ///   §12.3) is just pushed;
    /// - anything else — a concurrent arrival landing mid-sequence, a joining
    ///   peer's whole log — rewinds to the first divergence and replays forward
    ///   (§12.2). Cheap in practice: a concurrent arrival sits near
    ///   the top of the stack by construction, so the rewind is shallow.
    ///
    /// Untouched prefixes keep their snapshots (and their tiles' `Arc`s) as-is
    /// in every case. Convergence is untouched by the fast paths: disjoint
    /// footprints mean the shifted materialization computes the *same pixels*
    /// the canonical replay would, because every `apply` reads only what its
    /// footprint declares — so peers still agree state-for-state however each
    /// one got there.
    ///
    /// [`Footprint`]: super::footprint::Footprint
    fn resync(&mut self, ctx: &mut ApplyCtx) {
        let undone = undone_ids(&self.log);
        // Before the arms below, every one of which can return: this is the one
        // point a log change passes through, which is what keeps the cache from
        // being something a future path could forget (see [`Self::targets`]).
        self.targets = targets(&self.log, self.actor, &undone);
        // Indices into `log`, not borrows, so the arms below can take `&mut
        // self` and clone only what they materialize.
        let eff = effective_indices(&self.log, &undone);
        let mat: Vec<ActionId> = self.history.actions().map(|a| a.id).collect();
        let diverge = (0..mat.len().min(eff.len()))
            .take_while(|&i| mat[i] == self.log[eff[i]].id)
            .count();

        // The history is a prefix of the sequence (usually an exact one, or one
        // fresh action short): append what's missing.
        if diverge == mat.len() {
            for &i in &eff[diverge..] {
                let action = self.log[i].clone();
                self.history.push_action_with(action, ctx);
            }
            return;
        }

        // Exactly one materialized action left the sequence (an undo landed):
        // let the history shift it out through the commuting run after it.
        if eff.len() + 1 == mat.len()
            && (diverge..eff.len()).all(|i| self.log[eff[i]].id == mat[i + 1])
        {
            // The history doesn't report which path it took, and pixels can't
            // show it (that's the point) — so re-derive it for the stats.
            let commuting = {
                let mut suffix = self.history.actions().skip(diverge);
                let fp = footprint(suffix.next().expect("diverge < mat.len()"));
                suffix.take_while(|a| !fp.conflicts(&footprint(a))).count()
            };
            let suffix = mat.len() - diverge - 1;
            if commuting == suffix {
                self.stats.fast_removes += 1;
            } else {
                self.stats.rebuilds += 1;
                self.stats.replayed += (suffix - commuting) as u64;
            }
            self.history.remove_action_with(diverge, ctx);
            return;
        }

        // Rewind to the first divergence and replay forward (§12.2).
        self.stats.rebuilds += 1;
        let mut materialized = mat.len();
        while materialized > diverge {
            self.history.pop_action_with(ctx);
            materialized -= 1;
        }
        for &i in &eff[diverge..] {
            let action = self.log[i].clone();
            self.stats.replayed += 1;
            self.history.push_action_with(action, ctx);
        }
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
    /// action so peers can order it (§12.3). The engine routes
    /// through [`Timeline::undo_as_action`] first, so this is unreachable in
    /// practice; it conservatively does nothing.
    fn undo(&mut self, _ctx: &mut ApplyCtx) -> bool {
        false
    }

    fn redo(&mut self, _ctx: &mut ApplyCtx) -> bool {
        false
    }

    fn can_undo(&self) -> bool {
        self.targets.undo.is_some()
    }

    fn can_redo(&self) -> bool {
        self.targets.redo.is_some()
    }

    fn clone_actions(&self) -> Vec<Action> {
        self.log.clone()
    }

    fn undo_as_action(&self) -> Option<ActionId> {
        self.targets.undo
    }

    fn redo_as_action(&self) -> Option<ActionId> {
        self.targets.redo
    }

    fn merge(&mut self, action: Action, ctx: &mut ApplyCtx) -> bool {
        self.insert(action, ctx)
    }

    fn stats(&self) -> TimelineStats {
        self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            kind: ActionKind::SetBackground([0.0; 3]),
        }
    }

    fn undo_of(lamport: u64, actor: u64, target: ActionId) -> Action {
        Action {
            id: id(lamport, actor),
            kind: ActionKind::Undo(target),
        }
    }

    /// Logs here are built in increasing id order, which is the state
    /// [`ReplicatedTimeline`] keeps its own in — and what `action_by_id`'s binary
    /// search and `undone_ids`' single descending pass both rest on.
    fn resolve(log: &[Action], actor: u64) -> Targets {
        targets(log, ActorId(actor), &undone_ids(log))
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
}
