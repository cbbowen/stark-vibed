//! What logging an action costs (§5, §12.4).
//!
//! Every action enters the log through one of four doors here — [`Engine::commit`],
//! which asks the no-op question, and the three that may not — and everything a
//! commit owes is paid inside them: the sanitizing funnel, the copy the wire is owed,
//! the drag preview it supersedes, the counter that says the document moved, the
//! brush repointed off a layer that is gone, and the undo depth given up to memory.
//! An arm of `input` names a door and nothing else.

use super::Engine;
use crate::document::{ApplyCtx, Timeline};
use stark_model::document::{Action, ActionId, ActionKind, LayerId, StrokeRecord};

/// How much resident tile memory the engine will let **history retention** hold
/// before it starts giving up undo depth (§5).
///
/// `DocState` is cheap to clone and tiles are copy-on-write, so history retention
/// drives GPU memory reclamation for free — but only if something retires history.
/// `history` keeps its snapshots geometrically spaced, so what is retained is
/// `O(log n)` states rather than `O(n)`; each still pins every tile version that has
/// changed since, and on a large canvas a tile pair is ~640 KB.
///
/// **This is a bound on a cost that has not been measured**, in the sense
/// [`MAX_RELEASE_PER_EPOCH`](crate::gpu::TilePool) and the compositor's flush cadence
/// are: 2 GiB is about 3200 tile pairs, comfortably past a large painting's working
/// set and well past what an ordinary session reaches. Raising it costs memory and
/// buys undo depth; the honest way to change it is to measure a session and say so.
///
/// The **default**, not the value — a frontend that knows what it is running on
/// sets its own ([`ViewCommand::SetHistoryBudget`](crate::command::ViewCommand)), and
/// Stark's own offers it as a slider. A default has to be safe on the smallest
/// machine that will meet it and generous on the largest, and where those disagree
/// it errs generous: reaching this at all takes a long session on a big canvas, and
/// the cost of being wrong upwards is memory pressure the browser reports, where
/// being wrong downwards is undo steps silently gone.
///
/// It is a *ceiling on retention*, not on the document. Paint that is on the canvas
/// now is held by the current state and no amount of trimming frees it — see
/// [`Engine::trim_history`] for why that is what [`MIN_UNDO_DEPTH`] guards.
pub const DEFAULT_HISTORY_BUDGET: u64 = 2 << 30;

/// Whether a stroke's commit takes the tiles its live preview already drew, rather
/// than rendering the stroke again at pen-up (`document::PreparedStroke`, §6.2).
///
/// **On**, because the two renders are the same picture to within a level or two and
/// only one of them is paid for while the artist is waiting: a long stroke rendered
/// a second time is a hitch at exactly the moment the incremental repaint exists to
/// remove one. What the other setting buys is not a better picture but an *identical*
/// one — the stroke drawn the single way a file, an undo and a collaborator all draw
/// it, so the drawing reproduces bit for bit (§8, §9) rather than within the seam a
/// cut costs. That is worth offering and not worth defaulting to.
///
/// The **default**, not the value —
/// [`ViewCommand::SetFastCommit`](crate::command::ViewCommand::SetFastCommit) sets it, and
/// Stark's own settings dialog offers it. Here rather than in the frontend's stored
/// preferences for the reason [`DEFAULT_HISTORY_BUDGET`] is: two defaults for one
/// behaviour is two answers to what Stark does out of the box, and this is a
/// behaviour whose two paths are nearly indistinguishable in pixels — a disagreement
/// would be invisible in everything but [`Engine::strokes_reused`].
pub const DEFAULT_FAST_COMMIT: bool = true;

/// Undo steps the engine will not trim below, however tight memory is.
///
/// **The guard against trimming for nothing.** Resident tiles are held by the
/// *current* document as much as by history, so a session with four full-canvas
/// layers can exceed any budget with almost no history at all — and there, folding
/// the undo stack away frees nothing and costs the user every step they might want
/// back. A floor makes that failure bounded: the worst case is a document that sits
/// over budget with [`MIN_UNDO_DEPTH`] steps of undo, which is the true answer rather
/// than an unbounded march to zero.
const MIN_UNDO_DEPTH: usize = 10;

impl Engine {
    /// Note that the **committed** document has been replaced: every cached
    /// [`FrozenHead`](super::live::FrozenHead) built against the old one is stale, and anything
    /// the frontend
    /// derived from it is out of date.
    ///
    /// One call rather than two counters bumped side by side at each of seven sites —
    /// a commit, either half of undo/redo, a merge, a share, a join, a reset —
    /// because "these advance together" is the property that has to hold, and a site
    /// that remembered one and forgot the other would be silent. The preview path
    /// deliberately does *not* come through here: it moves what is drawn without
    /// changing the document
    /// (see [`ObservableState::doc_revision`](super::ObservableState::doc_revision)).
    ///
    /// **Repointing the brush belongs here too**, for the same argument one scope up.
    /// The rule is not "a `RemoveLayer` removes a layer" — an undo of an `AddLayer`
    /// withdraws one, a merge folds one away, a peer's merge arrives having done the
    /// same, and a seek crosses additions wholesale (§17.9). All of them come through
    /// here, and none has to know it:
    /// [`repoint_active_layer`](Self::repoint_active_layer) returns on its first line
    /// when the layer still exists, which is every ordinary commit.
    pub(super) fn committed_changed(&mut self) {
        self.preview.invalidate();
        self.doc_revision += 1;
        self.repoint_active_layer();
    }

    /// Move the history playhead one step, the way [`DocCommand::Undo`](crate::command::DocCommand::Undo) and
    /// [`DocCommand::Redo`](crate::command::DocCommand::Redo) each do it.
    ///
    /// The two are one operation named twice: a shared session logs the step as an
    /// `Undo` action peers can order (§5.4, §12.3) and a solo one navigates, and redo
    /// is an `Undo` of an `Undo` — so the *only* thing that differs is which pair of
    /// timeline methods is asked. Passing the pair rather than writing the body out
    /// twice is what stops the two drifting: dropping the preview, bumping the
    /// revision on the navigating branch and re-reading the document's substrate
    /// afterwards are all things one arm could have grown and the other not.
    pub(super) fn navigate(
        &mut self,
        as_action: impl Fn(&Timeline) -> Option<ActionId>,
        step: impl Fn(&mut Timeline, &mut ApplyCtx) -> bool,
    ) {
        self.preview.set_doc(None);
        if let Some(target) = as_action(&self.timeline) {
            // The unconditional door: a step the timeline has just said it can take
            // must land, and a declined `Undo` would report a move the playhead did
            // not make.
            let id = self.next_action_id();
            self.commit_with_id(id, ActionKind::Undo(target));
        } else {
            step(&mut self.timeline, &mut self.shared.apply);
            self.committed_changed();
        }
        // A step across a `SetSubstrate` — or a `SetSubstrateScale` — moves the
        // document's substrate (§6.4).
        self.apply_document_substrate();
    }

    /// Log one action and apply it — **unless the document already reads that way**
    /// ([`is_noop_on`](crate::document::apply::is_noop_on)), in which case nothing is
    /// logged and the drag in flight is still dropped.
    ///
    /// **One door, so an arm of [`process_doc_inner`](Self::process_doc_inner) is one
    /// word.** A second entry point that skipped the no-op question would make which
    /// door an arm reaches for a per-arm judgement nothing checks — and that judgement
    /// has produced a bug before, `SetLayerVisible` logging a step for setting the
    /// value it already held while `SetLayerOpacity` did not. Ruling out the class
    /// costs nothing here, because every kind that would want the unchecked door — a
    /// stroke, a fill, a transform, a selection, a removal, a merge, a move — sits in
    /// `is_noop_on`'s exhaustive `false` arm and answers by construction (CLAUDE.md).
    ///
    /// **The unlogged drag in flight is dropped on every path through here**, once,
    /// rather than at each commit site that remembered to. A drag preview is a whole
    /// document standing in for the committed one (§17.6), so anything that moves the
    /// committed document supersedes it — leaving it up pins the canvas to the last
    /// dragged value and shadows every later edit.
    ///
    /// The declining path drops it too, which is the other half of a setter's
    /// bargain: a slider dragged out and back must log nothing *and* must still
    /// supersede the preview it left up, because a preview is superseded by something
    /// or not at all.
    pub(super) fn commit(&mut self, kind: ActionKind) {
        // Sanitized before the comparison, not after: `is_noop_on` compares the
        // payload against the one already in the document, and the stored one has
        // been through this funnel. Left raw, a slider released on the value it was
        // pressed on would compare unequal to its own sanitized twin and log an
        // action that changes nothing — the case the check exists to catch.
        let kind = kind.sanitized();
        if crate::document::apply::is_noop_on(&kind, self.document(), self.actor()) {
            self.preview.set_doc(None);
            return;
        }
        // Drawn only once the action is known to be worth logging, so a slider
        // dragged out and back spends no Lamport tick.
        let id = self.next_action_id();
        self.commit_sanitized(id, kind);
    }

    /// [`commit`](Self::commit) with the action id drawn by the caller and **the
    /// no-op question not asked** — the unconditional door.
    ///
    /// The id comes from outside because two callers cannot let this draw it:
    /// [`commit_minting`](Self::commit_minting) has to build the kind *from* the id,
    /// since a layer's id is the id of the action that mints it, and
    /// [`replay_stroke_seeded`](Self::replay_stroke_seeded) answers with it.
    ///
    /// The question is not asked because for these callers a decline would be silent
    /// damage rather than a saved undo step, which is why the exemption is a door and
    /// not a list of kinds. `commit_minting` has already baked the id into the kind's
    /// layer ids, so declining would leave them naming an action that never happened;
    /// [`commit_stroke`](Self::commit_stroke) has an offer of already-rendered tiles
    /// riding the context for exactly this push (`PreparedStroke`, §6.2); and an
    /// `Undo` that declined would leave the playhead where it was
    /// ([`navigate`](Self::navigate)). That none of their kinds *could* answer "no-op"
    /// today is a fact about `is_noop_on`, not a contract they rest on.
    pub(super) fn commit_with_id(&mut self, id: ActionId, kind: ActionKind) {
        self.commit_sanitized(id, kind.sanitized());
    }

    /// What both doors above open onto: log `kind` under `id` and apply it, with the
    /// payload already through the funnel.
    ///
    /// The precondition is in the name for a reason that is not the saved pass:
    /// `Logged::new` runs the funnel again on the way into the log, deliberately, so
    /// that a footprint is built from what the fold will actually see. What the
    /// precondition buys is that [`commit`](Self::commit) **logs the very value it
    /// compared** — the no-op answer and the stored payload cannot come apart, and a
    /// reader need not know `sanitized` is idempotent to trust that they agree.
    fn commit_sanitized(&mut self, id: ActionId, kind: ActionKind) {
        // Every logged edit, whatever kind — a stroke landing at pen-up, a fill, a
        // layer move. One row rather than one per `ActionKind`, because what a
        // profile is being asked here is "is a commit the hitch the artist felt",
        // and the answer is read against `input.fit` and `frame` beside it. Which
        // *kind* of commit was slow is a question the phases underneath it already
        // answer — `stroke.range` and its parts for a stroke that had to render,
        // where one that took its preview's tiles (`commit_stroke`) has none.
        crate::timing::span!("doc.commit");
        self.preview.set_doc(None);
        // `kind` arrives having been through the **minted** half of the sanitizing
        // funnel (§21.5); `Logged::new` is the "enters state" half. It runs in the two
        // doors above rather than inside the timeline so that the log and the wire
        // carry *what was applied* — the broadcast clone below is taken from this
        // action, so a peer that received the raw kind and cleaned it on arrival would
        // agree about pixels while disagreeing about the log.
        //
        // One call for every kind, rather than a payload-level call per kind that
        // carries a knob — a list every new action-with-a-knob has to be added to.
        let action = Action { id, kind };
        // Cloned only when there is somewhere for the copy to go: a `CommitStroke`
        // carries the stroke's whole fitted control-point list, the largest thing in
        // the log, and a solo session has nowhere to put it.
        let broadcast = self.is_shared().then(|| action.clone());
        let ctx = &mut self.shared.apply;
        self.timeline.push(action, ctx);
        // The committed document is what every in-flight preview is drawn over, so
        // every cached head built against the old one is now stale.
        self.committed_changed();
        if let Some(action) = broadcast
            && let Some(outbox) = self.authoring.outbox.as_mut()
        {
            outbox.push(action);
        }
        // Committing is the only thing that grows the undo stack, so it is the only
        // place retention has to be reconsidered. Deliberately not in
        // `committed_changed`, which an undo also comes through — giving up undo
        // depth *while the user is undoing* is the one moment it would be felt.
        self.trim_history();
    }

    /// Log and apply a stroke, offering the fold the tiles the preview already drew
    /// for it (`PreparedStroke`, §6.2) — what makes pen-up cost a tail rather than a
    /// stroke.
    ///
    /// The offer rides the context for exactly the one push, and the fold accepts it
    /// by taking it: a slot still full afterwards was declined — the record moved
    /// between the last fold and the release, or the base did — and is dropped here
    /// rather than left for a later fold to find.
    ///
    /// **This is the whole of what [`fast_commit`](Self::fast_commit) switches**, and
    /// the reason the setting can be one line: with nothing offered, the fold below
    /// renders the stroke exactly as a replay does, which is what makes the switched-
    /// off path bit-for-bit rather than merely close (`DEFAULT_FAST_COMMIT`).
    pub(super) fn commit_stroke(&mut self, rec: StrokeRecord) {
        // Taken either way. The tiles describe a stroke that is being committed this
        // instant, so they are no use to the next fold whichever path lands it, and
        // holding them past here would pin a stroke's worth of tiles on the one
        // setting that is about *not* using them.
        let prepared = self.preview.take_prepared();
        let offered = self
            .shared
            .apply
            .offer(prepared.filter(|_| self.fast_commit));
        // The unconditional door, and the one place a decline would cost paint rather
        // than an undo step: `reclaim` would take the offered tiles back safely enough,
        // but a `CommitStroke` that never reached the log is a stroke the artist drew
        // and the document does not have.
        let id = self.next_action_id();
        self.commit_with_id(id, ActionKind::CommitStroke(rec));
        // A slot still full after the push was declined; an empty one was taken.
        if offered && !self.shared.apply.reclaim() {
            self.strokes_reused += 1;
        }
    }

    /// Give up the oldest undo steps if history retention is holding more tile
    /// memory than [`DEFAULT_HISTORY_BUDGET`] allows (§5).
    ///
    /// **Half of what is left, not down to a target.** Halving converges in a few
    /// commits and leaves a cushion, where trimming to exactly the budget would fold
    /// the whole stack away the moment one large layer pushed it over — the same
    /// hysteresis argument, and the same arithmetic, as the tile pool's own surplus
    /// policy (`surplus_to_release`).
    ///
    /// **Asked of the pool, answered by the pool.** What is measured is resident tile
    /// bytes, and what a fold releases is the snapshots between here and there: the
    /// handles they pinned drop, their textures return to the free list, and the
    /// pool's epoch boundary hands the surplus back to the driver. So this does not
    /// free memory itself — it stops history being the reason the pool cannot.
    ///
    /// A shared session declines by construction, because
    /// [`Timeline::forget_oldest`] does: its document is re-materialized from the
    /// whole log (§12.2), so nothing there is foldable. Nothing about this call site
    /// has to know that.
    fn trim_history(&mut self) {
        if self.shared.apply.pool.resident_bytes() <= self.history_budget {
            return;
        }
        // How far back undo can currently travel. `None` is a timeline whose history
        // is not this client's alone to walk, which is also one that cannot be
        // trimmed — the same question, so the same answer.
        let Some((applied, _)) = self.timeline.scrub_range() else {
            return;
        };
        let Some(excess) = applied.checked_sub(MIN_UNDO_DEPTH).filter(|e| *e > 0) else {
            return;
        };
        let forgotten = self.timeline.forget_oldest((applied / 2).min(excess));
        if forgotten > 0 {
            tracing::debug!(
                forgotten,
                remaining = applied - forgotten,
                resident_mb = self.shared.apply.pool.resident_bytes() / (1 << 20),
                "gave up undo depth to release retained tiles",
            );
        }
    }

    /// Point the active layer at something that exists, preferring a paintable one:
    /// a matte may legitimately be selected, but someone who just lost the layer they
    /// were painting on wants to keep painting, not to land on the frame.
    /// Searches the whole tree, not just the root stack: removing a group takes
    /// carried layers with it, and the replacement may itself be carried by
    /// something (§14.2).
    fn repoint_active_layer(&mut self) {
        if self.document().contains_layer(self.session.active_layer) {
            return;
        }
        let (mut paintable, mut any) = (None, None);
        self.document().visit(&mut |l, _| {
            any = any.or(Some(l.id));
            if paintable.is_none() && l.is_paintable() {
                paintable = Some(l.id);
            }
        });
        if let Some(id) = paintable.or(any) {
            self.session.active_layer = id;
        }
    }

    pub(super) fn next_action_id(&mut self) -> ActionId {
        let id = ActionId {
            lamport: self.authoring.clock,
            actor: self.actor(),
        };
        self.authoring.clock += 1;
        id
    }

    /// Point the brush at `id` if it landed and can take a stroke.
    ///
    /// **Both halves matter**, and there are four callers (`AddLayer`, `PlaceImage`,
    /// `DuplicateLayer`, `MergeLayerDown`) that would otherwise each choose which to
    /// ask: an unknown carrier adds nothing (§14.8), so arming an id no layer has
    /// leaves the next stroke with nowhere to go — and a matte or a filter has no tile
    /// map, so arming one swallows that stroke instead (§15.7, §21.4).
    ///
    /// A no-op otherwise, deliberately: the brush stays where it was, which is a place
    /// that exists, rather than moving somewhere that cannot be painted on.
    pub(super) fn arm_active(&mut self, id: LayerId) {
        if self.document().layer(id).is_some_and(|l| l.is_paintable()) {
            self.session.active_layer = id;
        }
    }

    /// Commit an action **whose kind names the layers it mints**, and hand back the
    /// id it was given.
    ///
    /// The door [`LayerId`]'s shape asks for: a layer's id is the id of the action
    /// that minted it, so the kind cannot be built until that id exists — and here it
    /// cannot be built any other way. Peeking at the clock and committing separately
    /// would work exactly as long as nothing committed in between, which is a rule a
    /// call site could forget and no test would notice: the ids would name an action
    /// that never happened, and the layers would still be distinct.
    ///
    /// Everything else goes through [`commit`](Self::commit), which is this with the
    /// id thrown away.
    pub(super) fn commit_minting(
        &mut self,
        build: impl FnOnce(ActionId) -> ActionKind,
    ) -> ActionId {
        let id = self.next_action_id();
        let kind = build(id);
        // What the door is for, asked of the kind that came back rather than trusted
        // of the closure that built it: `ActionKind::minted_layers` is the exhaustive
        // list of what an action claims to mint, so a variant that grew a layer and
        // forgot to derive its id from `id` is caught here and not by two peers
        // disagreeing about which layer a stroke landed on.
        //
        // **Both halves of the id**, since `k` is the half that does the work for a
        // duplicate: sharing an action id is what makes the ids this action's, and
        // differing in `k` is what makes them each other's.
        let minted: Vec<LayerId> = kind.minted_layers().collect();
        debug_assert!(
            minted.iter().all(|layer| layer.action == id),
            "{} mints a layer id that is not this action's",
            kind.label(),
        );
        debug_assert!(
            minted
                .iter()
                .enumerate()
                .all(|(i, l)| !minted[..i].contains(l)),
            "{} mints one layer id twice",
            kind.label(),
        );
        self.commit_with_id(id, kind);
        id
    }
}
