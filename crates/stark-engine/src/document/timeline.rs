//! The timeline (§5, §12).
//!
//! [`Timeline`] is an **enum** over the two storage strategies, not a trait: an
//! operation a shared log must refuse is an arm with the refusal written beside it,
//! where a trait default would let a new operation come to do nothing in one mode
//! without anyone deciding it should. [`LinearTimeline`] is the single-user body (a
//! thin wrapper over `history::History`); [`ReplicatedTimeline`] is the multi-peer
//! one — a totally-ordered, grow-only action log (a replicated-log CRDT)
//! materialized through the very same `history::History` as a snapshot cache. Both
//! are the enum's variant bodies and neither is named by `Engine`.

use std::collections::HashSet;

use history::History;

/// The log entry `history` actually stores: an [`Action`] addressed to the state it
/// folds into (§2). [`Logged`] is `stark-model`'s so that the `history::Action` impl
/// can be written at all — the orphan rule put it there, and the division it forces
/// is the right one. Nothing outside this module sees it.
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
///
/// An enum rather than a trait object, and the difference is not style. There are two
/// implementations and the engine already branches on which it holds at every point
/// that matters — `is_shared()`, `undo_as_action().is_none()`, `scrub_range().is_none()`
/// — so what the trait bought was not polymorphism but a place to hide the branch.
/// What it *cost* was four methods whose default was to do nothing: a
/// [`ReplicatedTimeline`] cannot [`seek`](Self::seek) or
/// [`forget_oldest`](Self::forget_oldest) and a [`LinearTimeline`] has no
/// [`merge`](Self::merge), and each of those refusals arrived as an inherited default
/// nobody had written down at the implementation it applied to. A silent default is
/// how a *new* operation comes to do nothing in one mode without anyone deciding it
/// should: the arm that would have had to be written is the arm that never appears.
///
/// Every one of those refusals is an arm below now, with the reason beside it.
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
    /// peers can order it (§12.3), and the engine asks
    /// [`undo_as_action`](Self::undo_as_action) first — which answers `Some` for
    /// exactly the timeline this arm refuses.
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
    /// [`ActionKind::Undo`], or `None` if undo is plain timeline navigation (the solo
    /// path). The engine asks this first and only falls back to [`undo`](Self::undo)
    /// when it answers `None`.
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
    /// A [`Linear`](Self::Linear) is already what this returns. A
    /// [`Replicated`](Self::Replicated) hands over its **materialization**, which *is*
    /// a linear history: the effective sequence, applied in order, with the snapshots
    /// stepping back through it needs. So the walk comes back for the cost of a move
    /// rather than of a replay — and the log does not come with it, because with
    /// nobody left to merge, the `Undo` actions and the actions they suppress are
    /// answers to a question nobody will ask again.
    ///
    /// Consuming, so nothing is left holding a shared log that nothing materializes.
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
    /// `None` rather than `(n, n)` because the two say different things: a timeline
    /// with nowhere to go is still a timeline, while a [`ReplicatedTimeline`] has no
    /// single playhead to move at all — its materialization is a function of a log
    /// peers are still appending to, so a scrub would be undone by the next arrival.
    /// The frontend needs to tell "there is no history yet" from "this document's
    /// history is not yours alone to walk", and only the second is a reason to say so.
    pub fn scrub_range(&self) -> Option<(usize, usize)> {
        match self {
            Timeline::Linear(t) => Some((t.applied(), t.applied() + t.redo.len())),
            Timeline::Replicated(_) => None,
        }
    }

    /// Move the playhead to `to` (clamped to the range), applying or withdrawing
    /// whatever lies between. Returns whether the document changed.
    ///
    /// The withdrawn actions are *kept*, in the same place undo keeps them, which is
    /// the whole reason this is navigation rather than deletion: scrubbing back and
    /// forward is lossless, and committing a fresh edit at a scrubbed-back position
    /// truncates the future exactly as painting after an undo does.
    ///
    /// A shared session declines, and it is the same refusal
    /// [`scrub_range`](Self::scrub_range) makes: there is no single playhead to move.
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
    /// timelapse and a joining peer all still get the whole painting; what goes is the
    /// retained *snapshots* between here and there, and with them the tile handles they
    /// were pinning. That distinction is the whole of why this is safe: the document is
    /// its log (§1, §8), so a timeline that dropped actions would be saving a different
    /// painting, silently, with nothing on screen to say so.
    ///
    /// May fold **fewer** than asked, or none: `history` only folds as far as a cached
    /// state it can reach without replaying, and its cache is geometrically spaced.
    ///
    /// A shared session folds **nothing**, and must: its document is re-materialized
    /// from the whole log on every arriving action (§12.2), so an action folded into a
    /// base state is one the next merge cannot replay. The log is not this client's to
    /// shorten in any sense at all.
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
    /// Actions folded out of the undo stack by [`forget_oldest`], oldest first —
    /// **still part of the document** (§5).
    ///
    /// This is what makes retention safe. `History::forget_actions` hands its
    /// forgotten actions back precisely so a caller can keep them, and keeping them
    /// is the difference between giving up undo depth and losing a painting: the log
    /// *is* the document (§1, §8), so [`clone_actions`](Timeline::clone_actions) —
    /// which is what the save file, a timelapse and a joining peer are built from —
    /// goes on reporting every action ever committed. Only the snapshots between
    /// them are gone, and with them the tile handles history retention was pinning.
    ///
    /// Grows without bound, and that is correct rather than an oversight: it is the
    /// document, and a `CommitStroke` is a fitted control-point list of a few hundred
    /// bytes. Tens of thousands of strokes are tens of megabytes of CPU memory, where
    /// what this exists to reclaim is GPU tiles at ~640 KB apiece.
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
    /// Nothing is withheld and nothing is folded: the whole effective sequence is
    /// applied, which is where the session left the document, so the playhead
    /// starts at the newest step with the entire walk behind it. What was undone
    /// during the session is *not* waiting in `redo` — a suppressed action was
    /// never materialized, so it is not here to re-apply, which is the same
    /// flattening a solo load of the same file performs (`Engine::load_document`).
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

    /// How many actions are currently applied — the playhead's position.
    ///
    /// Asked of the history rather than tracked beside it: a second copy of a
    /// number the history already knows is a number that can disagree with it,
    /// and `Version` is opaque, so there is nothing else to ask.
    ///
    /// It costs nothing — `actions()` reports its own length — so the answer that
    /// cannot go stale is also the cheap one, and neither has to be traded for the
    /// other. Which matters because [`scrub_range`](Timeline::scrub_range) asks it
    /// once a render.
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
    /// `Version::default()`.
    ///
    /// The two are the same thing right up until [`forget_oldest`](Timeline::forget_oldest)
    /// folds anything: `History::initial_version` is `Version::default()` "until
    /// `forget_actions` folds actions into the initial state, and the version of the
    /// state it folded them into thereafter". Against the constant, a fully folded
    /// history reports that it can undo and then does nothing when asked — an Undo
    /// button that is lit and inert. Asking the history where its own floor is has no
    /// such failure mode, and reads better besides.
    fn can_undo(&self) -> bool {
        self.history.last_version() != self.history.initial_version()
    }

    fn can_redo(&self) -> bool {
        !self.redo.is_empty()
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
    /// **The redo stack is deliberately untouched.** It holds actions that are *not*
    /// applied, so they pin no snapshot and cost no tiles; folding them would give up
    /// a redo the user can still see offered, for nothing.
    ///
    /// Reported in terms of the actions actually folded, which is not always the
    /// number asked for — see [`Timeline::forget_oldest`].
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
    history: History<Entry>,
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
    /// [`from_log`](Self::from_log) also funnels through), and that place either
    /// resolves both directly or hands off to `resync`, which writes them from
    /// the same `undone` set it materializes against.
    targets: Targets,
    /// Ids suppressed by effective `Undo`s ([`undone_ids`]) — the set every
    /// question about effectiveness starts from.
    ///
    /// Held rather than rederived per insert, because the insert that dominates
    /// a session cannot change it: an ordinary action suppresses nothing, and
    /// nothing already in the log can be suppressing an action newer than all of
    /// it. Deriving it anyway built a `HashSet` over the whole log per commit to
    /// rediscover the same answer.
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

        // The insert that dominates a session — an ordinary action newer than
        // everything already in the log — puts itself at the end of the effective
        // sequence and changes nothing else about it:
        //
        // - **nothing suppresses it.** An `Undo` carries a larger id than its
        //   target, its author having seen the target to undo it, so no action
        //   already here can be undoing one newer than all of them. `undone` is
        //   consulted regardless, since a log arrives from files and peers and a
        //   malformed one need not respect that;
        // - **it suppresses and revives nothing**, being ordinary, so no other
        //   action's effectiveness moves and no other action's slot moves;
        // - **its own key is its id**, the largest, so it sorts last.
        //
        // Which leaves the whole of `resync` a push. Deriving the sequence again
        // instead is a pass over the log, a `HashSet` and a sort *per commit* —
        // so a session paid for its own length squared to learn, each time, that
        // the newest action goes last.
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
    /// Both questions are asked only about **this actor's** actions, so a peer's
    /// commit moves neither and there is nothing to do. Ours moves both, to
    /// answers already in hand: a fresh ordinary edit is the newest thing we
    /// could undo, and it clears our redo stack — which is `redo_target`'s
    /// "newer than the latest ordinary action" bound, reached directly instead of
    /// by scanning the log for it.
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
    /// Reached when [`insert`](Self::insert) cannot say for itself what changed —
    /// an `Undo`, whose whole job is to change other actions' effectiveness, or
    /// an arrival that lands mid-log. Everything else appends, and appending is
    /// the one case whose effect on the sequence is known without deriving it.
    ///
    /// The change is still almost always a single action entering or leaving,
    /// so this classifies it (§12.6):
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
    /// [`Footprint`]: stark_model::document::Footprint
    fn resync(&mut self, ctx: &mut ApplyCtx) {
        self.undone = undone_ids(&self.log);
        // Before the arms below, every one of which can return: together with
        // `insert`'s fast path this is where every log change resolves the
        // derived state, which is what keeps it from being something a future
        // path could forget (see [`Self::targets`]).
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
            // The history doesn't report which path it took, and pixels can't
            // show it (that's the point) — so re-derive it for the stats.
            //
            // Off the **cached** footprints each `Logged` carries, not
            // `compute_footprint`. `Logged` derefs to `Action`, so the free function
            // used to resolve here silently and re-do the whole derivation over the
            // commuting suffix — for a counter. A `TransformWarp` in that run is a
            // 57×57 fine-lattice solve apiece, which is the exact cost `Logged` was
            // given a cache to stop paying (see its note).
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

// No test module: every test this file held was the undo algebra's, and they went with
// it to `stark_model::document::effect`. What is left here — materializing a sequence
// into `history`, and the two storage strategies — is covered by the integration suite,
// which drives it through an `Engine` because that is the only thing that has one.
