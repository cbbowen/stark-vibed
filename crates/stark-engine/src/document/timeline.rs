//! The timeline (§5, §12): where the current [`DocState`] comes from, and how far back
//! it can be walked.
//!
//! [`LinearTimeline`] is the single-user body; [`ReplicatedTimeline`] is the multi-peer
//! one — a totally-ordered, grow-only action log (a replicated-log CRDT). Callers hold
//! the [`Timeline`] enum, never a body.

use std::collections::HashSet;

use history::History;

/// The log entry `history` actually stores: an [`Action`] addressed to the state it
/// folds into (§2). [`Logged`] lives in `stark-model` because the orphan rule puts the
/// `history::Action` impl there. Nothing outside this module sees it.
type Entry = Logged<DocState>;

use super::apply::ApplyCtx;
use super::state::DocState;
use stark_model::document::{Action, ActionId, ActorId, Logged};
// The undo algebra is the model's (§12.3, `document::effect`): which actions in a log
// are effective is a fact about the log, like *that* it folds and which actions
// commute, and it names nothing this crate owns.
use stark_model::document::{Targets, effective_indices, targets, undo_target_of, undone_ids};

/// A versioned document: the source of the current [`DocState`] plus undo/redo —
/// **solo or shared**, which are the only two it will ever be.
pub enum Timeline {
    /// One client's own history: a linear undo/redo stack (§5).
    Linear(LinearTimeline),
    /// A shared session's: a grow-only log, totally ordered, materialized through the
    /// same `history::History` as a cache (§12).
    Replicated(ReplicatedTimeline),
}

impl Timeline {
    /// The current document state (`O(1)`).
    pub fn current(&self) -> &DocState {
        match self {
            Timeline::Linear(t) => t.current(),
            Timeline::Replicated(t) => t.current(),
        }
    }

    /// Commit an action, advancing the timeline. Clears any redo stack.
    pub fn push(&mut self, action: Action, ctx: &mut ApplyCtx) {
        match self {
            Timeline::Linear(t) => t.push(action, ctx),
            Timeline::Replicated(t) => {
                t.insert(action, ctx);
            }
        }
    }

    /// Step back one action if possible; returns whether anything was undone.
    ///
    /// **Shared sessions never come through here.** Undo is a logged action there so
    /// peers can order it (§12.3): ask [`undo_as_action`](Self::undo_as_action) first,
    /// which answers `Some` for exactly the timeline this arm refuses.
    pub fn undo(&mut self, ctx: &mut ApplyCtx) -> bool {
        match self {
            Timeline::Linear(t) => t.undo(ctx),
            Timeline::Replicated(_) => false,
        }
    }

    /// Re-apply the most recently undone action; returns whether anything was redone.
    /// Refused in a shared session for [`undo`](Self::undo)'s reason.
    pub fn redo(&mut self, ctx: &mut ApplyCtx) -> bool {
        match self {
            Timeline::Linear(t) => t.redo(ctx),
            Timeline::Replicated(_) => false,
        }
    }

    pub fn can_undo(&self) -> bool {
        match self {
            Timeline::Linear(t) => t.can_undo(),
            Timeline::Replicated(t) => t.targets.undo.is_some(),
        }
    }

    pub fn can_redo(&self) -> bool {
        match self {
            Timeline::Linear(t) => t.can_redo(),
            Timeline::Replicated(t) => t.targets.redo.is_some(),
        }
    }

    /// All committed actions, oldest to newest — the basis of the save format
    /// (§8). For a replicated timeline this is the **full** log, including `Undo`
    /// actions and the actions they suppress — exactly the payload a joining peer
    /// needs (§12.4).
    pub fn clone_actions(&self) -> Vec<Action> {
        match self {
            Timeline::Linear(t) => t.actions().cloned().collect(),
            Timeline::Replicated(t) => t.log.clone(),
        }
    }

    /// Shared-mode undo (§5.4): the action an "undo" should target by logging an
    /// [`ActionKind::Undo`](stark_model::document::ActionKind::Undo), or `None` if undo
    /// is plain timeline navigation. Ask this first and fall back to
    /// [`undo`](Self::undo) only on `None`.
    pub fn undo_as_action(&self) -> Option<ActionId> {
        match self {
            Timeline::Linear(_) => None,
            Timeline::Replicated(t) => t.targets.undo,
        }
    }

    /// Shared-mode redo: the **`Undo` action** a "redo" should itself undo (redo is an
    /// `Undo` of an `Undo`, §12.3), or `None`.
    pub fn redo_as_action(&self) -> Option<ActionId> {
        match self {
            Timeline::Linear(_) => None,
            Timeline::Replicated(t) => t.targets.redo,
        }
    }

    /// Integrate an action authored elsewhere (§12.1). Returns whether it was new
    /// (false = a duplicate, or a solo timeline, which has no peers to hear from).
    pub fn merge(&mut self, action: Action, ctx: &mut ApplyCtx) -> bool {
        match self {
            Timeline::Linear(_) => false,
            Timeline::Replicated(t) => t.insert(action, ctx),
        }
    }

    /// The same history, one client's own again, once the session that shared it has
    /// ended (§12.3, §18.2.4).
    ///
    /// **The log does not come with it**: the `Undo` actions and the actions they
    /// suppress are dropped, so [`clone_actions`](Self::clone_actions) afterwards
    /// reports only the effective walk.
    pub fn unshare(self) -> Self {
        match self {
            Timeline::Linear(t) => Timeline::Linear(t),
            Timeline::Replicated(t) => Timeline::Linear(LinearTimeline::from_history(t.history)),
        }
    }

    /// Where the playhead stands and how far it can travel — `(applied, total)`, both
    /// counted in actions — or `None` for a timeline that cannot be scrubbed
    /// (§18.2.4).
    ///
    /// `None` rather than `(n, n)`: a [`ReplicatedTimeline`] has no single playhead —
    /// its materialization is a function of a log peers are still appending to — and a
    /// frontend has to tell that from a history that is merely empty.
    pub fn scrub_range(&self) -> Option<(usize, usize)> {
        match self {
            Timeline::Linear(t) => Some((t.applied(), t.applied() + t.redo.len())),
            Timeline::Replicated(_) => None,
        }
    }

    /// Move the playhead to `to` (clamped to the range), applying or withdrawing
    /// whatever lies between. Returns whether the document changed.
    ///
    /// Withdrawn actions are *kept*, as undo keeps them: scrubbing is lossless in both
    /// directions, and committing a fresh edit at a scrubbed-back position truncates the
    /// future exactly as painting after an undo does.
    ///
    /// A shared session declines, for [`scrub_range`](Self::scrub_range)'s reason.
    pub fn seek(&mut self, to: usize, ctx: &mut ApplyCtx) -> bool {
        match self {
            Timeline::Linear(t) => t.seek(to, ctx),
            Timeline::Replicated(_) => false,
        }
    }

    /// A caption per action, oldest first and spanning the **whole** range
    /// [`scrub_range`](Self::scrub_range) reports — the withdrawn ones included, since
    /// a scrubber has to label the steps it can travel *to*. Empty where there is no
    /// scrubber, which is the shared case.
    pub fn scrub_labels(&self) -> Vec<&'static str> {
        match self {
            Timeline::Linear(t) => t.scrub_labels(),
            Timeline::Replicated(_) => Vec::new(),
        }
    }

    /// How materializations have been serviced (§12.6). A solo timeline reports zeros
    /// — the counters exist for the replicated fast paths, and there are none here to
    /// count.
    pub fn stats(&self) -> TimelineStats {
        match self {
            Timeline::Linear(_) => TimelineStats::default(),
            Timeline::Replicated(t) => t.stats,
        }
    }

    /// Give up the ability to undo past the oldest `count` actions, folding them into
    /// the base state, and report how many were actually folded (§5).
    ///
    /// **The log is not shortened — only the reach of undo is.** What is folded is
    /// still returned by [`clone_actions`](Self::clone_actions), so the file, a
    /// timelapse and a joining peer still get the whole painting (§1, §8). What goes is
    /// the retained *snapshots*, and with them the tile handles they were pinning.
    ///
    /// May fold **fewer** than asked, or none. A shared session folds **nothing**, and
    /// must: its document is re-materialized from the whole log on every arriving
    /// action (§12.2), so a folded action is one the next merge cannot replay.
    pub fn forget_oldest(&mut self, count: usize) -> usize {
        match self {
            Timeline::Linear(t) => t.forget_oldest(count),
            Timeline::Replicated(_) => 0,
        }
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
    history: History<Entry>,
    /// Actions popped by `undo`, awaiting `redo`. Cleared on a fresh `push`.
    redo: Vec<Action>,
    /// Actions folded out of the undo stack by [`LinearTimeline::forget_oldest`],
    /// oldest first — **still part of the document** (§5), so
    /// [`clone_actions`](Timeline::clone_actions) goes on reporting every action ever
    /// committed. Only the snapshots between them are gone.
    ///
    /// Grows without bound, and that is correct: a `CommitStroke` is a few hundred
    /// bytes, against the GPU tiles at ~640 KB apiece this exists to reclaim.
    forgotten: Vec<Action>,
}

impl LinearTimeline {
    pub fn new(initial: DocState) -> Self {
        Self {
            history: History::new(initial),
            redo: Vec::new(),
            forgotten: Vec::new(),
        }
    }

    /// Adopt a history materialized elsewhere — what [`Timeline::unshare`] hands
    /// over when a shared session ends.
    ///
    /// The playhead starts at the newest step with the whole walk behind it, and `redo`
    /// starts empty: a suppressed action was never materialized.
    fn from_history(history: History<Entry>) -> Self {
        Self {
            history,
            redo: Vec::new(),
            forgotten: Vec::new(),
        }
    }

    /// Every action this timeline holds, oldest first — the folded prefix and then
    /// the live history.
    pub fn actions(&self) -> impl Iterator<Item = &Action> {
        self.forgotten
            .iter()
            .chain(self.history.actions().map(|e| &**e))
    }

    /// How many actions are currently applied — the playhead's position. `O(1)`, which
    /// matters because [`scrub_range`](Timeline::scrub_range) asks it once a render.
    fn applied(&self) -> usize {
        self.history.actions().len()
    }
}

impl LinearTimeline {
    fn current(&self) -> &DocState {
        self.history.last_state()
    }

    fn push(&mut self, action: Action, ctx: &mut ApplyCtx) {
        self.redo.clear();
        // Infallible apply (§5) — no error to handle.
        self.history.push_action_with(Entry::new(action), ctx);
    }

    fn undo(&mut self, ctx: &mut ApplyCtx) -> bool {
        match self.history.pop_action_with(ctx).map(Entry::into_action) {
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
                self.history.push_action_with(Entry::new(action), ctx);
                true
            }
            None => false,
        }
    }

    /// Compared against the history's **own** oldest version, not against
    /// `Version::default()`: the two part once
    /// [`forget_oldest`](Timeline::forget_oldest) folds anything, and against the
    /// constant a fully folded history would claim an undo it cannot perform.
    fn can_undo(&self) -> bool {
        self.history.last_version() != self.history.initial_version()
    }

    fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// Scrubbing **is** the undo/redo split, moved in bulk rather than one step at a
    /// time (§18.2.4): it leaves the timeline in a state undo could equally have
    /// produced, which is what makes it safe to paint from wherever the playhead stops.
    fn seek(&mut self, to: usize, ctx: &mut ApplyCtx) -> bool {
        let applied = self.applied();
        let to = to.min(applied + self.redo.len());
        match to.cmp(&applied) {
            std::cmp::Ordering::Less => {
                // `pop_actions_with` yields newest-first, which is exactly the
                // order successive `undo`s push onto `redo` — so the stack stays
                // one thing however it was filled.
                let popped = self.history.pop_actions_with(applied - to, ctx);
                self.redo.extend(popped.into_iter().map(Entry::into_action));
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
                    self.history.push_action_with(Entry::new(action), ctx);
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

    /// Folds through to `History::forget_actions`, keeping what it hands back.
    ///
    /// **The redo stack is deliberately untouched**: its actions are not applied, so
    /// they pin no snapshot and cost no tiles.
    ///
    /// Reports the actions actually folded, which is not always the number asked for —
    /// see [`Timeline::forget_oldest`].
    fn forget_oldest(&mut self, count: usize) -> usize {
        if count == 0 {
            return 0;
        }
        let folded = self.history.forget_actions(count);
        let n = folded.len();
        // Oldest first out of `forget_actions`, and oldest first here, so the
        // concatenation in `actions()` is the log in order.
        self.forgotten
            .extend(folded.into_iter().map(Entry::into_action));
        n
    }
}

/// Multi-peer timeline (§12): a grow-only set of actions, totally ordered by
/// [`ActionId`] `(lamport, actor)`. The canonical state is the deterministic replay of
/// the *effective* actions in that order, so two peers that have seen the same set
/// compute identical pixels (strong eventual consistency, §12.2).
pub struct ReplicatedTimeline {
    /// Whose undo/redo this timeline answers for ([`Timeline::undo_as_action`]).
    actor: ActorId,
    /// The full log, sorted by id, no duplicates. Grow-only.
    log: Vec<Action>,
    /// Membership index for O(1) duplicate rejection.
    ids: HashSet<ActionId>,
    /// Materialization of the effective sequence, in order (the initial empty
    /// document is `History`'s version 0, which pops never remove).
    history: History<Entry>,
    stats: TimelineStats,
    /// What a local undo and redo would target as the log now stands — a pure
    /// function of [`log`](Self::log) and [`actor`](Self::actor), cached because
    /// `can_undo`/`can_redo` are asked far more often than the log changes.
    ///
    /// **Every write to [`log`](Self::log) must resolve this**, or it can disagree with
    /// what it was derived from.
    targets: Targets,
    /// Ids suppressed by effective `Undo`s ([`undone_ids`]) — the set every question
    /// about effectiveness starts from. Held rather than rederived per insert, because
    /// the insert that dominates a session cannot change it: an ordinary action
    /// suppresses nothing, and nothing already in the log can suppress an action newer
    /// than all of it.
    undone: HashSet<ActionId>,
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
            undone: HashSet::new(),
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
        // Both read before the move, and both decide the fast path below.
        let id = action.id;
        let ordinary = undo_target_of(&action).is_none();
        let appended = self.log.last().is_none_or(|last| last.id < id);
        // Sorted insert; the common case (a fresh local or causally-newest
        // remote action) lands at the end.
        let pos = self.log.partition_point(|a| a.id < id);
        self.log.insert(pos, action);

        // An ordinary action newer than everything already in the log appends to the
        // effective sequence and changes nothing else about it: nothing can suppress it
        // (an `Undo` carries a larger id than its target), it suppresses and revives
        // nothing itself, and its id sorts last. `undone` is consulted regardless, since
        // a log arrives from files and peers and a malformed one need not respect the id
        // ordering.
        if ordinary && appended && !self.undone.contains(&id) {
            let action = self.log[pos].clone();
            self.history.push_action_with(Entry::new(action), ctx);
            self.retarget_appended(id);
        } else {
            self.resync(ctx);
        }
        true
    }

    /// Move the undo/redo targets over an action the fast path appended.
    ///
    /// Both targets are about **this actor's** actions only, so a peer's commit moves
    /// neither. Ours moves both to answers already in hand: a fresh ordinary edit is
    /// the newest thing we could undo, and it clears our redo stack.
    fn retarget_appended(&mut self, id: ActionId) {
        if id.actor == self.actor {
            self.targets = Targets {
                undo: Some(id),
                redo: None,
            };
        }
    }

    /// Make `history` match the current effective sequence, rederiving that
    /// sequence from the log.
    ///
    /// Reached when [`insert`](Self::insert) cannot say for itself what changed — an
    /// `Undo`, or an arrival that lands mid-log. The change is still almost always a
    /// single action entering or leaving, so this classifies it (§12.6): a removed
    /// action is shifted out past everything it commutes with, an appended one is
    /// pushed (a redo materializes at the *top* of the stack, §12.3), and anything else
    /// rewinds to the first divergence and replays forward (§12.2).
    ///
    /// Untouched prefixes keep their snapshots, and their tiles' `Arc`s, in every case.
    /// The fast paths converge with the canonical replay only because every `apply`
    /// reads and writes no more than its [`Footprint`] declares (§12.6): disjoint
    /// footprints then mean the shifted materialization computes the *same pixels*.
    ///
    /// [`Footprint`]: stark_model::document::Footprint
    fn resync(&mut self, ctx: &mut ApplyCtx) {
        self.undone = undone_ids(&self.log);
        // Before the arms below, every one of which can return: with `insert`'s fast
        // path this is where every log change resolves the derived state
        // (see [`Self::targets`]).
        self.targets = targets(&self.log, self.actor, &self.undone);
        // Indices into `log`, not borrows, so the arms below can take `&mut
        // self` and clone only what they materialize.
        let eff = effective_indices(&self.log, &self.undone);
        let mat: Vec<ActionId> = self.history.actions().map(|a| a.id).collect();
        let diverge = (0..mat.len().min(eff.len()))
            .take_while(|&i| mat[i] == self.log[eff[i]].id)
            .count();

        // The history is a prefix of the sequence (usually an exact one, or one
        // fresh action short): append what's missing.
        if diverge == mat.len() {
            for &i in &eff[diverge..] {
                let action = self.log[i].clone();
                self.history.push_action_with(Entry::new(action), ctx);
            }
            return;
        }

        // Exactly one materialized action left the sequence (an undo landed):
        // let the history shift it out through the commuting run after it.
        if eff.len() + 1 == mat.len()
            && (diverge..eff.len()).all(|i| self.log[eff[i]].id == mat[i + 1])
        {
            // The history doesn't report which path it took, so re-derive it for the
            // stats — off the **cached** footprints each `Logged` carries. `Logged`
            // derefs to `Action`, so the free `compute_footprint` would resolve here
            // silently and re-derive the whole commuting suffix for a counter, a
            // `TransformWarp` in that run being a 57×57 fine-lattice solve apiece.
            let commuting = {
                let mut suffix = self.history.actions().skip(diverge);
                let fp = suffix.next().expect("diverge < mat.len()").footprint();
                suffix.take_while(|a| !fp.conflicts(a.footprint())).count()
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
            self.history.push_action_with(Entry::new(action), ctx);
        }
    }
}

impl ReplicatedTimeline {
    fn current(&self) -> &DocState {
        self.history.last_state()
    }
}

// No test module: the undo algebra's tests live with it in
// `stark_model::document::effect`, and what is left here — materializing a sequence
// into `history` — is covered by the integration suite, which drives it through an
// `Engine` because that is the only thing that has one.
