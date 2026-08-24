//! Collaboration and presence: the two channels a shared session runs on (§12, §17).
//!
//! They are deliberately separate, and the rule that separates them is the one §4
//! runs on — *does replay need it to reproduce a pixel?* The **action** channel is
//! the document: ordered, replicated, and the thing convergence is a property of.
//! The **presence** channel is everything else a collaborator can see — a cursor, a
//! selected layer, a gesture that has not committed — and nothing in the action log
//! ever references it, which is exactly what lets the transport drop, coalesce or
//! delay a presence frame without touching convergence.
//!
//! The engine stays network-agnostic either way: it owns the merge semantics and
//! these hooks, and `stark-net` owns the wire.

use super::{Engine, ROOT_LAYER};
use crate::document::{DocState, LinearTimeline, ReplicatedTimeline, Timeline, TimelineStats};
use crate::peer::{Identity, Peer};
use stark_model::DocumentFile;
use stark_model::document::{Action, ActorId};
use stark_model::peer::PeerFrame;

/// What one presence-pump tick moved ([`Engine::take_presence`]).
///
/// The two halves reach different places: `frame` is owed to the wire, `repaint`
/// to the compositor. They travel together because the pump is the engine's only
/// clock, so the expiry that produces `repaint` can only run on its cadence.
#[derive(Debug)]
pub struct PresenceTick {
    /// This client's presence, if anything a peer would care about has changed
    /// since the last drain — `None` when solo or when nothing moved.
    pub frame: Option<PeerFrame>,
    /// Expiry took something off the canvas — a stalled gesture or a departed
    /// peer — so a repaint is owed. Without it the last composite, stale stroke
    /// and all, stays on screen until something else forces a paint.
    pub repaint: bool,
}

impl Engine {
    // --- the action channel (§12) ---------------------------------------

    /// Whether this engine is **broadcasting**: authoring into a shared session,
    /// with an outbox for the transport to drain (§12.4).
    ///
    /// The same question as "is this document's history a shared log", and kept
    /// that way deliberately: [`end_collaboration`](Self::end_collaboration) gives
    /// the history back in the same breath as it drops the outbox, so there is no
    /// state in which this and [`scrub_range`](Self::scrub_range) disagree about
    /// whose the document is.
    pub fn is_shared(&self) -> bool {
        self.authoring.outbox.is_some()
    }

    /// This engine's author id for new actions.
    pub fn actor(&self) -> ActorId {
        self.authoring.actor
    }

    /// Start sharing the **current** document as `actor` (the host side).
    /// Converts the linear history into a [`ReplicatedTimeline`] over the same
    /// log. Solo-authored actions ([`ActorId::SOLO`]) are rewritten to `actor`
    /// — done once, before any peer has seen them — so the sharer can still
    /// undo their pre-share strokes (undo targets *my* actions, §12.3).
    pub fn start_collaboration(&mut self, identity: impl Into<Identity>) {
        if self.is_shared() {
            return;
        }
        let identity = identity.into();
        let actor = identity.actor;
        let mut log = self.timeline.clone_actions();
        for a in &mut log {
            if a.id.actor == ActorId::SOLO {
                a.id.actor = actor;
            }
        }
        // The layer-id space belongs to the actor, so this client's counter resumes
        // past whatever *it* has already minted here (§17.9). A first share finds
        // nothing and so does start at 1: the layer ids in a solo log are `SOLO`'s,
        // and unlike the action ids above they are deliberately left that way. What
        // restarting regardless got wrong is that the actor of a second share is the
        // first one back again — an identity is a browser's persisted key, not a
        // session's — as is the actor of a file this client shared before and has
        // just loaded.
        self.authoring.next_layer = Self::next_ordinal(actor, &log);
        // Replay from the substrate this document's log *starts* from, not from the
        // default — same base state `reset_document` builds, so re-hosting a document
        // that was created on a non-default canvas doesn't silently move it.
        let initial = DocState::with_layer(ROOT_LAYER).with_substrate(self.initial_substrate);
        let ctx = &mut self.shared.apply;
        self.timeline = Box::new(ReplicatedTimeline::from_log(actor, initial, log, ctx));
        self.authoring.actor = actor;
        self.session.adopt_identity(identity);
        self.authoring.outbox = Some(Vec::new());
        self.preview.set_doc(None);
        self.committed_changed();
        self.mark_live_stale();
    }

    /// Join a shared session (the peer side): replace the document with the
    /// session's **full** log — including `Undo` actions, which the replicated
    /// timeline resolves — and author future actions as `actor`.
    pub fn join_collaboration(&mut self, file: &DocumentFile, identity: impl Into<Identity>) {
        let identity = identity.into();
        let actor = identity.actor;
        // Everything the shared log needs before it can be replayed, in the order
        // that makes it a replay rather than an approximation ([`Self::adopt`]). A
        // joiner replays the whole painting, so this is where getting it wrong costs
        // the most.
        self.adopt(file);
        let ctx = &mut self.shared.apply;
        let initial = DocState::with_layer(ROOT_LAYER).with_substrate(self.initial_substrate);
        self.timeline = Box::new(ReplicatedTimeline::from_log(
            actor,
            initial,
            file.actions.clone(),
            ctx,
        ));
        self.authoring.actor = actor;
        self.session.adopt_identity(identity);
        self.resync_counters(&file.actions);
        self.authoring.outbox = Some(Vec::new());
        // Whatever the replayed log left the document on.
        self.apply_document_substrate();
        self.committed_changed();
        self.mark_live_stale();
    }

    /// Leave a shared session: stop queueing broadcasts, forget everyone who was in
    /// it, and take the history back as this client's own to walk. Editing continues
    /// solo on the same canvas — same pixels, same active layer — and a later
    /// [`Self::start_collaboration`] shares it again.
    ///
    /// The peers' *selections* stay in the document, because replay still needs them
    /// to reproduce their strokes; they simply stop being drawn, since the roster is
    /// what decides that (§17.3).
    pub fn end_collaboration(&mut self) {
        // Taking the queue away is what stops the broadcast — there is no second
        // flag to leave disagreeing with it, and whatever was still queued goes with
        // it, since there is no longer anyone owed it.
        self.authoring.outbox = None;
        self.peers.clear();
        // A [`ReplicatedTimeline`] refuses to seek, to undo by navigation and to fold
        // its oldest actions away, and every one of those refusals is made on behalf
        // of peers who are still appending to the log (§12.2, §18.2.4). None of them
        // is, once this returns — so the timeline stops being one that refuses,
        // rather than the refusals outliving the session that justified them.
        //
        // `unshare` consumes, which is what leaves nothing behind still claiming a
        // shared log; the swap therefore needs somewhere to park, and an empty
        // timeline is the cheapest valid thing there is (a `DocState` is persistent
        // maps, §5.1).
        let parked: Box<dyn Timeline> =
            Box::new(LinearTimeline::new(DocState::with_layer(ROOT_LAYER)));
        self.timeline = std::mem::replace(&mut self.timeline, parked).unshare();
        // What the document's history *offers* moved with it: a peer's stroke is
        // this document's to undo now, and the scrubber comes back. Published for
        // the reason [`Self::start_collaboration`] publishes the conversion it makes
        // in the other direction — the pixels are untouched either way, and nothing
        // else in the projection would say so.
        self.committed_changed();
        self.mark_live_stale();
    }

    /// Integrate an action authored by a peer (§12.1). Idempotent —
    /// duplicates are rejected by id. Advances the Lamport clock past the
    /// remote action so future local ids order after everything seen.
    pub fn merge_remote(&mut self, action: Action) -> bool {
        self.authoring.clock = self.authoring.clock.max(action.id.lamport + 1);
        let author = action.id.actor;
        let ctx = &mut self.shared.apply;
        let merged = self.timeline.merge(action, ctx);
        if merged {
            // Replaces the document every frozen head was composited onto — and
            // repoints the brush if the arriving action took the layer this client
            // was painting on. Asked of the document rather than of the action's
            // *variant*, which is how a peer's `MergeLayerDown` came to strand it:
            // `merge_apply` ends in `remove_layer(source)`, so keying on
            // `RemoveLayer` answered a question about deletion by naming one of the
            // two actions that delete (§17.9).
            self.committed_changed();
            // A gesture is a thing that becomes an action, so the action's arrival is
            // the end-of-gesture signal — no id to correlate, and no window in which
            // both the live copy and the committed one are drawn.
            self.peers.clear_gesture(author);
            self.mark_live_stale();
        }
        // A peer may have switched the substrate (§6.4).
        self.apply_document_substrate();
        merged
    }

    /// Drain locally-committed actions awaiting broadcast (empty when solo).
    ///
    /// Drains the queue rather than taking it, so a shared session stays shared: the
    /// only thing that ends the broadcast is [`end_collaboration`](Self::end_collaboration).
    pub fn take_outbox(&mut self) -> Vec<Action> {
        self.authoring
            .outbox
            .as_mut()
            .map(std::mem::take)
            .unwrap_or_default()
    }

    /// How the timeline has serviced materializations (§12.6): the
    /// commutation fast paths versus rewind-and-replay. Zeros when solo. For
    /// tests and diagnostics — pixels can't tell the paths apart, by design.
    pub fn timeline_stats(&self) -> TimelineStats {
        self.timeline.stats()
    }

    /// How many in-flight strokes the preview fold is caching a settled head for
    /// (§17.6) — at most one per actor who is *currently* drawing one.
    ///
    /// For tests and diagnostics, beside [`timeline_stats`](Self::timeline_stats) and
    /// for the same reason: a head is a cache, so pixels cannot show whether one is
    /// held. What they also cannot show is a head held for a gesture that has *ended*,
    /// which is not a wrong picture but a `DocState`'s worth of tile handles the pool
    /// cannot reclaim — invisible until the GPU runs out. Countable here instead.
    /// `&mut`, because the fold it counts within is rebuilt lazily: the count is
    /// only meaningful of a serviced fold, so this flushes first — the same
    /// picture the next paint would build.
    pub fn live_head_count(&mut self) -> usize {
        self.flush_live();
        self.preview.head_count()
    }

    /// How many of this client's stroke commits have taken the preview's tiles
    /// rather than rendering the stroke again at pen-up (`PreparedStroke`, §6.2).
    ///
    /// For tests and diagnostics, beside [`live_head_count`](Self::live_head_count)
    /// and for its reason: the two ways a commit can land are the same pixels by
    /// design, so only a count can say which one ran — and a commit that quietly
    /// fell back to the whole render is the hitch this path exists to remove,
    /// reported by nothing else.
    pub fn strokes_reused(&self) -> u64 {
        self.strokes_reused
    }

    /// The preview's invalidation epoch (§17.6): a counter that advances whenever the
    /// document the in-flight gestures are composited onto is replaced — by a commit,
    /// an undo, a remote merge, a load, or an unlogged drag preview being installed or
    /// dropped. A cached head stamped with an older value is discarded.
    ///
    /// For tests and diagnostics, beside [`live_head_count`](Self::live_head_count).
    /// Pixels cannot stand in for it: a drag preview that changes no *tiles* — the
    /// substrate color, say — leaves a stale head drawing exactly the right paint, so
    /// the picture is right while the rule that keeps it right has been broken. It only
    /// becomes visible for a preview that does move tiles, by which point the cause is
    /// several commands behind.
    pub fn preview_epoch(&self) -> u64 {
        self.preview.epoch()
    }

    // --- the presence channel (§17.4) -----------------------------------
    //
    // Symmetric with the action hooks above, and separate for the reason the module
    // doc gives: nothing in the action log ever references presence.

    /// Whether [`take_presence`](Self::take_presence) would do anything at `now` —
    /// a `&self` test a pump can run without borrowing the engine mutably.
    ///
    /// This is what keeps an idle shared session free. The pump has to wake on a
    /// fixed cadence (that is what makes the latch coalesce, §5.1, and it is the
    /// engine's only clock), but *waking* need not mean working: a tick where
    /// nothing has moved and no peer is due to expire should cost this comparison
    /// and nothing else — no mutable borrow, no roster rebuild, and above all no
    /// write to the signal the engine lives in, which would mark it dirty and
    /// re-render every component that reads it.
    ///
    /// Conservative in the same direction as [`Session::publish_due`]: it may say
    /// yes where the drain then finds nothing, never the reverse.
    pub fn presence_due(&self, now: f64) -> bool {
        self.peers.expiry_due(now) || (self.is_shared() && self.session.publish_due(now))
    }

    /// A counter that changes whenever the peer roster does, so a frontend can tell
    /// that its projection is stale without rebuilding it (§17.4).
    pub fn peers_revision(&self) -> u64 {
        self.peers.revision()
    }

    /// This client's presence, if anything a peer would care about has changed since
    /// the last call (§17.5). Also expires peers that have gone quiet,
    /// since this is called on the frontend's publish cadence — the only clock
    /// `stark-engine` has, because it deliberately owns none.
    ///
    /// `frame` is `None` when solo: presence with nobody to read it is pure cost.
    /// `repaint` reports whether the expiry changed the canvas — a stalled gesture
    /// or a departed peer takes paint off it, and a caller that drops that bit
    /// leaves the stale stroke on screen until something else forces a paint.
    pub fn take_presence(&mut self, now: f64) -> PresenceTick {
        self.now = now.max(self.now);
        let repaint = self.peers.tick(self.now).canvas;
        if repaint {
            self.mark_live_stale();
        }
        let frame = self
            .is_shared()
            .then(|| self.session.publish(self.now))
            .flatten();
        PresenceTick { frame, repaint }
    }

    /// The farewell frame, so peers drop this client at once instead of waiting out
    /// [`PEER_TIMEOUT`](stark_model::peer::PEER_TIMEOUT). Send it before tearing the
    /// transport down.
    pub fn leaving_presence(&mut self) -> PeerFrame {
        self.session.publish_leaving()
    }

    /// Integrate presence published by `actor`, whose identity comes from the
    /// transport's authenticated origin and never from the frame body — a peer can
    /// publish its own presence and nobody else's (§17.7).
    ///
    /// Returns whether the **canvas** changed, i.e. whether a repaint is owed. A
    /// frame that only moved a cursor or a selected layer returns `false`: those are
    /// chrome, drawn from the roster projection, which a caller notices moved through
    /// [`peers_revision`](Self::peers_revision) instead. Presence arrives at pointer
    /// rate from every peer at once, so the difference between the two questions is
    /// the difference between a compositor pass per remote pointer move and none.
    ///
    /// Dated by `now`, the **caller's** clock — the same one it hands
    /// [`take_presence`](Self::take_presence) — folded into [`Self::now`] so the
    /// engine's clock stays monotonic. Dating by `self.now` alone would assume the
    /// pump advances it every tick, and the pump skips `take_presence` on a tick with
    /// nothing to publish: on a client that is only *watching*, the clock would
    /// advance per [`HEARTBEAT`](stark_model::peer::HEARTBEAT), and every frame merged in
    /// between would age a whole heartbeat at once when the expiry finally ran —
    /// taking down live gestures whose frames arrive thirty times a second.
    pub fn merge_presence(&mut self, actor: ActorId, frame: PeerFrame, now: f64) -> bool {
        self.now = now.max(self.now);
        let now = self.now;
        if actor == self.actor() {
            // Our own frame, echoed back by a flood transport. The local session is
            // the authority on this client; taking it back off the wire would fight
            // with it.
            return false;
        }
        let change = self.peers.merge(actor, frame, now);
        if change.canvas {
            self.mark_live_stale();
        }
        change.canvas
    }

    /// Everyone else in the session, in ascending [`ActorId`] order (empty solo).
    pub fn peers(&self) -> impl Iterator<Item = &Peer> {
        self.peers.iter()
    }

    /// This client's display name, as peers see it.
    pub fn name(&self) -> &str {
        self.session.name()
    }
}
