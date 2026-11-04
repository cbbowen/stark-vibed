//! Collaboration and presence: the two channels a shared session runs on (§12, §17).
//!
//! They are separate, and the rule that separates them is the one §4 runs on — *does
//! replay need it to reproduce a pixel?* The **action** channel is the document:
//! ordered, replicated, and the thing convergence is a property of. The **presence**
//! channel is everything else a collaborator can see — a cursor, a selected layer, an
//! uncommitted gesture — and nothing in the action log ever references it, which is
//! what lets the transport drop, coalesce or delay a presence frame without touching
//! convergence.
//!
//! The engine owns the merge semantics and these hooks; `stark-net` owns the wire.

use super::commit::Drag;
use super::{Engine, ROOT_LAYER};
use crate::Result;
use crate::document::DocState;
use crate::fold::{LinearTimeline, ReplicatedTimeline, Timeline};
use crate::peer::Peer;
use crate::presence::Identity;
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
    /// The same question as "is this document's history a shared log":
    /// [`end_collaboration`](Self::end_collaboration) gives the history back in the
    /// same breath as it drops the outbox, so there is no state in which this and
    /// [`scrub_range`](Self::scrub_range) disagree about whose the document is.
    pub(crate) fn is_shared(&self) -> bool {
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
    ///
    /// **A scrubbed-away future does not survive the share**, and cannot: what is
    /// shared is [`clone_actions`](Timeline::clone_actions), the log up to the
    /// playhead, and the withheld suffix a `LinearTimeline` holds for redo is not part
    /// of it. That is the same document `save_bytes` would write and the same one a
    /// joiner will see — but it is the moment the redo stack stops existing, and
    /// `can_redo` goes false straight after with nothing else said. A frontend that
    /// wants to warn has to ask before calling.
    ///
    /// **Answers whether it took effect**, which is what §4 asks of anything that
    /// mutates and is not a command: `false` is "already sharing", the one case this
    /// declines.
    pub fn start_collaboration(&mut self, identity: impl Into<Identity>) -> bool {
        if self.is_shared() {
            return false;
        }
        let identity = identity.into();
        let actor = identity.actor;
        let mut log = self.timeline.clone_actions();
        for a in &mut log {
            if a.id.actor == ActorId::SOLO {
                a.id.actor = actor;
            }
        }
        // **The layer ids inside those actions still say `SOLO`, and stay that way.**
        // A `LayerId` names the action that minted it, so rewriting the log's actor
        // above did not rewrite them — and must not: every reference to a layer, in
        // every later stroke, move and merge, would have to move with it, to change
        // nothing that can be observed. What an id has to be is unique, and it still
        // is: `SOLO` authors no action in a shared session, so nothing this actor mints
        // from here on can land on one of these — and there is no per-actor counter to
        // recover (§17.9).
        self.enter_shared(identity, log);
        true
    }

    /// Become `identity` in a shared session whose log is `log` — the steps hosting and
    /// joining share, so the two cannot drift: the document re-materialized from the
    /// log onto the origin (which supersedes a drag over it), this client authoring as
    /// the identity's actor and presenting as it, and the outbox opened.
    ///
    /// What it leaves to the caller is the one thing they differ on. A join resumes the
    /// clock past the log it was handed (`resync_counters`); a host must not, since
    /// lowering the clock to its shared prefix could reissue an id a withheld redo
    /// suffix already holds.
    fn enter_shared(&mut self, identity: Identity, log: Vec<Action>) {
        let actor = identity.actor;
        let initial = self.origin();
        self.change_committed(Drag::Supersede, |timeline, ctx| {
            *timeline =
                Timeline::Replicated(ReplicatedTimeline::from_log(actor, initial, log, ctx));
            true
        });
        self.authoring.actor = actor;
        self.presence.adopt_identity(identity);
        self.authoring.outbox = Some(Vec::new());
    }

    /// Join a shared session (the peer side): replace the document with the
    /// session's **full** log — including `Undo` actions, which the replicated
    /// timeline resolves — and author future actions as `actor`.
    ///
    /// **Fails, and changes nothing, on a document whose color space this build
    /// lacks** ([`DocError::UnsupportedColorSpace`](stark_model::DocError)) — the same
    /// refusal [`Engine::load_bytes`] makes about a file, through the same door
    /// ([`ValidatedFile`]). It is a `Result` because a session arrives over a transport:
    /// this is the one adoption path whose input nobody in this process has ever looked
    /// at, and an unchecked one would reach `adopt`'s `expect` with the painting
    /// unsaved. A frontend shows the refusal and stays where it is.
    ///
    /// [`ValidatedFile`]: super::file::ValidatedFile
    pub fn join_collaboration(
        &mut self,
        file: &DocumentFile,
        identity: impl Into<Identity>,
    ) -> Result<()> {
        // Before anything is disturbed, which is the whole of what the type buys:
        // `adopt` empties the document on its second line.
        let validated = super::file::ValidatedFile::new(file)?;
        // Everything the shared log needs before it can be replayed, in the order that
        // makes it a replay rather than an approximation.
        self.adopt(&validated);
        self.enter_shared(identity.into(), file.actions.clone());
        self.resync_counters(&file.actions);
        Ok(())
    }

    /// Leave a shared session: stop queueing broadcasts, forget everyone who was in
    /// it, and take the history back as this client's own to walk. Editing continues
    /// solo on the same canvas — same pixels, same active layer — and a later
    /// [`Self::start_collaboration`] shares it again.
    ///
    /// The peers' *selections* stay in the document, because replay still needs them
    /// to reproduce their strokes; they simply stop being drawn, since the roster is
    /// what decides that (§17.3).
    ///
    /// **Answers whether a shared session was ended**, and declines when there was
    /// none — §4's rule, and what keeps the guard structural rather than a line a
    /// frontend has to hold. Running the body regardless would bump the document
    /// revision, drop the preview and force a full re-projection for nothing.
    pub fn end_collaboration(&mut self) -> bool {
        if !self.is_shared() {
            return false;
        }
        // Taking the queue away is what stops the broadcast — there is no second
        // flag to leave disagreeing with it, and whatever was still queued goes with
        // it, since there is no longer anyone owed it.
        self.authoring.outbox = None;
        self.peers.clear();
        // A `ReplicatedTimeline` refuses to seek, to undo by navigation and to fold its
        // oldest actions away, on behalf of peers who are still appending to the log
        // (§12.2, §18.2.4). None of them is, once this returns, so the refusals must
        // not outlive the session that justified them.
        //
        // `unshare` consumes, which is what leaves nothing behind still claiming a
        // shared log; the swap therefore needs somewhere to park, and an empty timeline
        // is the cheapest valid thing there is (a `DocState` is persistent maps, §5.1).
        //
        // A committed change although the pixels are untouched: what the history
        // *offers* moved with it — a peer's stroke is this document's to undo now, and
        // the scrubber comes back — and nothing else in the projection would say so.
        // The document is the same one, so a drag held across leaving stays up.
        self.change_committed(Drag::Keep, |timeline, _| {
            let parked = Timeline::Linear(LinearTimeline::new(DocState::with_layer(ROOT_LAYER)));
            *timeline = std::mem::replace(timeline, parked).unshare();
            true
        });
        true
    }

    /// Integrate an action authored by a peer (§12.1). Idempotent —
    /// duplicates are rejected by id. Advances the Lamport clock past the
    /// remote action so future local ids order after everything seen.
    ///
    /// **Refused once the device has died** (§5): nothing applied after that point
    /// could render, and a fold that half-ran on a lost device is a state no peer
    /// holds. The refusal is here, on the one door every peer action comes through,
    /// rather than asked of each frontend's collaboration pump — which is how it went
    /// unasked. The action is not lost to the session: a client that recovers rejoins,
    /// and reconciliation hands it back (§17.5).
    pub fn merge_remote(&mut self, action: Action) -> bool {
        if self.shared.gpu.health().has_failed() {
            return false;
        }
        // Saturating, as a layer number does (§14.6): the id is a peer's or a file's
        // to state, and `u64::MAX + 1` would panic a debug build and wrap a release
        // one onto 0 — ordering every later local action before the one just seen.
        self.authoring.clock = self.authoring.clock.max(action.id.lamport.saturating_add(1));
        let author = action.id.actor;
        let merged = self.change_committed(Drag::Keep, |timeline, ctx| timeline.merge(action, ctx));
        if merged {
            // A gesture is a thing that becomes an action, so the action's arrival is
            // the end-of-gesture signal — no id to correlate, and no window in which
            // both the live copy and the committed one are drawn.
            self.peers.clear_gesture(author);
        }
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

    // --- the presence channel (§17.4) -----------------------------------
    //
    // Separate for the reason the module doc gives: nothing in the action log ever
    // references presence.

    /// Whether [`take_presence`](Self::take_presence) would do anything at `now` —
    /// a `&self` test a pump can run without borrowing the engine mutably.
    ///
    /// This is what keeps an idle shared session free. The pump has to wake on a fixed
    /// cadence (that is what makes the latch coalesce, §5.1, and it is the engine's
    /// only clock), but *waking* need not mean working: a tick where nothing has moved
    /// and no peer is due to expire should cost this comparison and nothing else — no
    /// mutable borrow, no roster rebuild, and above all no write to the signal the
    /// engine lives in, which would re-render every component that reads it.
    ///
    /// Conservative in the same direction as `PresenceTx::publish_due`: it may say yes
    /// where the drain then finds nothing, never the reverse.
    ///
    /// Judged at the instant [`take_presence`](Self::take_presence) would act at —
    /// `now` through the roster's clock, which never steps back — so a caller whose
    /// clock did step back is not told "nothing due" by one and expired by the other.
    pub fn presence_due(&self, now: f64) -> bool {
        let now = self.peers.clock_at(now);
        self.peers.expiry_due(now)
            || (self.is_shared()
                && self.presence.publish_due(
                    now,
                    self.session.active_layer(),
                    self.session.is_gesturing(),
                ))
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
    /// `repaint` reports whether the expiry changed the canvas — a caller that drops
    /// that bit leaves a stale stroke on screen until something else forces a paint.
    pub fn take_presence(&mut self, now: f64) -> PresenceTick {
        let now = self.peers.advance(now);
        let repaint = self.peers.tick(now).canvas();
        if repaint {
            self.mark_live_stale();
        }
        let frame = self.is_shared().then(|| self.publish(now)).flatten();
        PresenceTick { frame, repaint }
    }

    /// What this client has to say about itself right now, if the latch says anything
    /// (§17.5). **The one place the two halves of presence meet**: the session says
    /// what is being dragged out and where it is aimed, the publisher decides what of
    /// that the wire has not already been told.
    fn publish(&mut self, now: f64) -> Option<PeerFrame> {
        self.presence.publish(
            now,
            self.session.active_layer(),
            self.session.gesture_ordinal(),
            self.session.live(),
        )
    }

    /// The farewell frame, so peers drop this client at once instead of waiting out
    /// [`PEER_TIMEOUT`](stark_model::peer::PEER_TIMEOUT). Send it before tearing the
    /// transport down.
    pub fn leaving_presence(&mut self) -> PeerFrame {
        self.presence.publish_leaving(self.session.active_layer())
    }

    /// Integrate presence published by `actor`, whose identity comes from the
    /// transport's authenticated origin and never from the frame body — a peer can
    /// publish its own presence and nobody else's (§17.7).
    ///
    /// Returns whether the **canvas** changed, i.e. whether a repaint is owed. A frame
    /// that only moved a cursor or a selected layer returns `false`: those are chrome,
    /// drawn from the roster projection, which a caller notices moved through
    /// [`peers_revision`](Self::peers_revision) instead. Presence arrives at pointer
    /// rate from every peer at once, so the two questions differ by a compositor pass
    /// per remote pointer move.
    ///
    /// Dated by `now`, the **caller's** clock — the same one it hands
    /// [`take_presence`](Self::take_presence) — folded into the roster's clock so it
    /// stays monotonic (`Peers::advance`). Reading that clock without advancing it would
    /// assume the pump moves it every tick, and the pump skips `take_presence` on a tick
    /// with nothing to publish: on a client that is only *watching*, every frame merged
    /// in between would age a whole [`HEARTBEAT`](stark_model::peer::HEARTBEAT) at once
    /// when the expiry finally ran, taking down live gestures whose frames arrive thirty
    /// times a second.
    pub fn merge_presence(&mut self, actor: ActorId, frame: PeerFrame, now: f64) -> bool {
        let now = self.peers.advance(now);
        if actor == self.actor() {
            // Our own frame, echoed back by a flood transport. The local session is
            // the authority on this client; taking it back off the wire would fight
            // with it.
            return false;
        }
        let change = self.peers.merge(actor, frame, now);
        if change.canvas() {
            self.mark_live_stale();
        }
        change.canvas()
    }

    /// Everyone else in the session, in ascending [`ActorId`] order (empty solo).
    pub fn peers(&self) -> impl Iterator<Item = &Peer> {
        self.peers.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::DocCommand;
    use crate::view::Extent2;

    fn engine(gpu: crate::GpuContext) -> Engine {
        Engine::new(gpu, wgpu::TextureFormat::Rgba8Unorm, Extent2::new(64, 64))
    }

    /// **A device that has failed takes no peer action** (§5) — and one that has not
    /// takes the same action, so the refusal is the device's and not the action's.
    ///
    /// The failure is the cheapest real one there is: a texture past the device's
    /// limits, which the uncaptured-error handler records as `Invalid`. Not a loss —
    /// but every failure means the operation that hit it produced nothing, and the
    /// frontend stops (`GpuHealth::has_failed`).
    #[test]
    fn a_failed_device_refuses_peer_actions() {
        let (Some(host_gpu), Some(peer_gpu)) = (
            crate::testing::context_or_skip(),
            crate::testing::context_or_skip(),
        ) else {
            return;
        };
        let mut host = engine(host_gpu);
        host.start_collaboration(ActorId(1));
        let mut peer = engine(peer_gpu.clone());
        peer.join_collaboration(&host.document_file(), ActorId(2))
            .expect("join a session this build can render");

        host.process(DocCommand::AddLayer {
            carrier: None,
            above: None,
        });
        host.process(DocCommand::AddLayer {
            carrier: None,
            above: None,
        });
        let mut sent = host.take_outbox().into_iter();
        let (first, second) = (sent.next().expect("one"), sent.next().expect("two"));
        assert!(peer.merge_remote(first), "a healthy device merges");

        let over = peer_gpu.device.limits().max_texture_dimension_2d + 1;
        let _refused = peer_gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("stark merge test: past the limit"),
            size: wgpu::Extent3d {
                width: over,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let _ = peer_gpu.device.poll(wgpu::PollType::wait_indefinitely());
        assert!(peer_gpu.health().has_failed(), "the failure was recorded");

        let clock = peer.authoring.clock;
        assert!(!peer.merge_remote(second), "a failed device merges nothing");
        assert_eq!(peer.authoring.clock, clock, "nor does it take the action's clock");
    }
}
