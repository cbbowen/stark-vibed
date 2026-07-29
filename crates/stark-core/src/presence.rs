//! The presence gesture protocol — **both ends, adjacent** (PEER_DESIGN.md §5).
//!
//! [`GestureTx`] encodes the gesture a client has in flight into [`GestureFrame`]s;
//! [`GestureRx`] reassembles them. That is one protocol, not two: a sent-path
//! watermark, a head/delta split, gap detection, and a periodic resync that repairs
//! any receiver which missed a delta. The rules that make it work are properties of
//! the *pair* —
//!
//! - a `head` rides exactly the frames that start from index 0;
//! - only **frozen** control points may be counted as sent, because the provisional
//!   tail can still move (DESIGN §6.2);
//! - the frozen watermark only ever advances, so a resync frame (which says nothing
//!   new about what is frozen) cannot walk it back;
//! - a watermark may only advance for a frame that is actually **sent**;
//! - a delta may only be spliced onto the frame that *immediately* precedes it,
//!   because the retained prefix is trusted on the strength of `from` values the
//!   receiver may never have seen.
//!
//! — and none of them is checkable from one side alone. Split across the sender's
//! module and the receiver's, each half was individually defensible and they
//! disagreed about states the other could produce; that seam is where both of the
//! bugs fixed in `77f0f69` lived, and the last rule above is a third that only
//! surfaced once the two ends could be driven against each other. Here the state
//! machines sit side by side, the rules are `debug_assert`ed where they are cheap,
//! and `tests::round_trip_survives_a_lossy_channel` drives one through a dropping,
//! duplicating, delaying channel into the other.
//!
//! The wire *types* stay in [`crate::peer`], which is the public surface. What lives
//! here is the state that interprets them.

use crate::document::{SelectionOp, StrokeRecord};
use crate::path::{ControlPoint, frozen_spans_for};
use crate::peer::{GESTURE_RESYNC, GestureFrame, LiveGesture, StrokeHead};

/// The gesture a sender has in flight, in the form [`GestureTx::encode`] reads it.
///
/// Built by the caller from whatever it actually holds — for
/// [`Session`](crate::session::Session) that is the live [`PathFitter`](crate::path::PathFitter),
/// which stays the single source of truth for the local stroke. This type is the
/// narrow window the encoder sees it through, not a second copy of it.
pub(crate) enum GestureSource {
    Stroke {
        head: StrokeHead,
        path: Vec<ControlPoint>,
        /// How many leading control points are final (DESIGN §6.2).
        frozen: usize,
    },
    Selection(SelectionOp),
}

/// The sending half: what this client has told the wire about its gesture.
pub(crate) struct GestureTx {
    /// Ordinal of the gesture the last frame carried, or `None` if it carried none.
    sent_id: Option<u64>,
    /// Control points of that gesture the wire has already carried. Only *frozen*
    /// ones count: the provisional tail is resent because it can still move.
    sent: usize,
    /// When the last resync frame went out.
    resync_at: f64,
}

impl GestureTx {
    pub(crate) fn new() -> Self {
        Self {
            sent_id: None,
            sent: 0,
            resync_at: f64::NEG_INFINITY,
        }
    }

    /// Whether a resync is due at `now` — time to re-send the gesture's invariant
    /// head and its whole path, repairing any receiver that missed a delta and
    /// priming any client that arrived mid-stroke, without either having to ask.
    pub(crate) fn resync_due(&self, now: f64) -> bool {
        GESTURE_RESYNC.is_some_and(|interval| now - self.resync_at >= interval)
    }

    /// Record that a frame carrying a resync went out at `now`.
    pub(crate) fn stamp_resync(&mut self, now: f64) {
        self.resync_at = now;
    }

    /// Whether the wire is currently carrying a gesture — so one that has just ended
    /// still owes the frame that clears it.
    pub(crate) fn in_flight(&self) -> bool {
        self.sent_id.is_some()
    }

    /// Encode `source` for the wire, advancing the watermarks.
    ///
    /// **Call this only for a frame that is actually going out.** The watermarks
    /// record what the receiver has been *told*; advancing them for a frame the
    /// caller then drops would skip control points nobody ever saw, and the gap would
    /// not be repaired until the next resync.
    pub(crate) fn encode(
        &mut self,
        id: u64,
        source: Option<GestureSource>,
        resync: bool,
    ) -> Option<GestureFrame> {
        let Some(source) = source else {
            self.sent_id = None;
            self.sent = 0;
            return None;
        };
        match source {
            GestureSource::Selection(op) => {
                // A marquee or lasso goes whole: it is already decimated, and unlike a
                // stroke its tail is not append-only — the closing edge follows the
                // cursor, so there is no frozen prefix to exploit.
                self.sent_id = Some(id);
                self.sent = 0;
                Some(GestureFrame::Selection { id, op })
            }
            GestureSource::Stroke { head, path, frozen } => {
                let frozen = frozen.min(path.len());
                debug_assert!(
                    self.sent_id != Some(id) || frozen >= self.sent,
                    "the frozen watermark must not walk back within one gesture"
                );
                // A frame is *fresh* — it carries the head and starts from 0 — for a
                // gesture the wire has not carried yet, and on every resync.
                let fresh = resync || self.sent_id != Some(id);
                let from = if fresh { 0 } else { self.sent.min(path.len()) };
                let points = path[from..].to_vec();
                self.sent_id = Some(id);
                self.sent = frozen;
                Some(GestureFrame::Stroke {
                    id,
                    head: fresh.then_some(head),
                    from: from as u32,
                    points,
                })
            }
        }
    }
}

/// The receiving half: one peer's gesture, reassembled from the frames that arrived.
#[derive(Clone, Debug, Default)]
pub(crate) struct GestureRx {
    /// The ordinal being tracked. Held separately from both `drawn` and `stroke`
    /// because neither is present in every state this can be in: a selection leaves
    /// no `stroke`, and a stroke whose head has not arrived leaves nothing `drawn`.
    /// It is the only thing that reliably tells this gesture from the one before it.
    id: Option<u64>,
    /// The stroke being reassembled, and the path so far. Kept apart from `drawn`
    /// because a stroke that has lost frames stops *growing* while still being
    /// tracked, awaiting the next resync frame.
    stroke: Option<StrokeAssembly>,
    /// What a renderer should draw for this peer right now.
    drawn: Option<LiveGesture>,
    /// [`PeerFrame::seq`](crate::peer::PeerFrame::seq) of the last frame spliced into
    /// `stroke`. A delta is only safe to splice if it *immediately* follows the frame
    /// before it — see [`Self::apply`].
    last_seq: Option<u64>,
}

#[derive(Clone, Debug)]
struct StrokeAssembly {
    id: u64,
    head: StrokeHead,
    path: Vec<ControlPoint>,
    /// Control points the sender has stopped resending — i.e. the ones its fitter
    /// froze. The receiver learns this for free from the delta's `from`, and it is
    /// exactly what the incremental repaint needs to know (PEER_DESIGN.md §6).
    frozen: usize,
}

impl GestureRx {
    /// What to draw for this peer, if anything.
    pub(crate) fn drawn(&self) -> Option<&LiveGesture> {
        self.drawn.as_ref()
    }

    /// The ordinal of the gesture in flight — what tells a cached render of this
    /// peer's stroke apart from a render of the one before it.
    pub(crate) fn id(&self) -> Option<u64> {
        self.id
    }

    /// How many leading control points the sender has declared final. Conservative:
    /// it is learned from a delta's `from`, which is the sender's frozen count as of
    /// the *previous* frame, so it always lags the truth rather than outrunning it.
    ///
    /// Exists for the round-trip test, which needs the exact prefix the protocol
    /// promises; a renderer wants [`frozen_spans`](Self::frozen_spans) instead.
    #[cfg(test)]
    pub(crate) fn frozen(&self) -> usize {
        self.stroke.as_ref().map_or(0, |s| s.frozen)
    }

    /// How many spans of the reassembled stroke are settled, so the preview can
    /// repaint only the tail (DESIGN.md §6.2, PEER_DESIGN.md §6).
    pub(crate) fn frozen_spans(&self) -> usize {
        self.stroke
            .as_ref()
            .map_or(0, |s| frozen_spans_for(s.frozen, s.path.len()))
    }

    /// Take the gesture down. Returns whether anything was being drawn.
    pub(crate) fn clear(&mut self) -> bool {
        let was_drawing = self.drawn.is_some();
        self.id = None;
        self.stroke = None;
        self.drawn = None;
        self.last_seq = None;
        was_drawing
    }

    /// Integrate one frame, carrying the [`PeerFrame::seq`](crate::peer::PeerFrame::seq)
    /// it arrived under. Returns whether what is *drawn* changed.
    pub(crate) fn apply(&mut self, frame: GestureFrame, seq: u64) -> bool {
        // A new ordinal is a different gesture: drop whatever was being drawn or
        // assembled rather than splicing two of them together.
        //
        // Keyed on the ordinal rather than on the stroke assembly, because a
        // *selection* leaves no assembly. Keyed on the assembly, a stroke delta whose
        // head had been lost found nothing to clear, took the early return below, and
        // left the peer's last marquee sitting on the canvas.
        let mut changed = false;
        if self.id != Some(frame.id()) {
            changed = self.clear();
        }
        self.id = Some(frame.id());
        match frame {
            GestureFrame::Selection { op, .. } => {
                self.stroke = None;
                self.last_seq = Some(seq);
                self.drawn = Some(LiveGesture::Selection(op));
                true
            }
            GestureFrame::Stroke {
                id,
                head,
                from,
                points,
            } => {
                let from = from as usize;
                // Start (or restart, on a resync frame) whenever the head is present.
                debug_assert!(
                    head.is_none() || from == 0,
                    "a head rides only a frame that starts the path over"
                );
                // A frame that resends the path whole stands on its own; a delta has
                // to be spliced onto what came before it.
                let whole = head.is_some() && from == 0;
                if let Some(head) = head
                    && (self.stroke.as_ref().is_none_or(|s| s.id != id) || from == 0)
                {
                    // A resync restarts the *assembly*, not the gesture: for the same
                    // ordinal the frozen watermark carries over, because the resent
                    // path's prefix is exactly the frozen points already held — a
                    // frozen control point never moves, and a resync says nothing new
                    // about freezing. Reset to zero here, every resync frame
                    // discarded the renderer's cached head (`Engine::refresh_live`
                    // keys on `frozen_spans`) and redrew the whole stroke from
                    // scratch — once a second, per stroking peer.
                    let frozen = match self.stroke.as_ref() {
                        Some(s) if s.id == id => s.frozen,
                        _ => 0,
                    };
                    self.stroke = Some(StrokeAssembly {
                        id,
                        head,
                        path: Vec::new(),
                        frozen,
                    });
                    self.last_seq = None;
                }
                let Some(assembly) = self.stroke.as_mut() else {
                    // A delta for a gesture whose head we never saw. Nothing to draw
                    // until the next resync frame — and nothing of the *previous*
                    // gesture either, which the ordinal check above has seen to.
                    return changed;
                };
                // **A delta is only safe to splice if nothing was lost before it.**
                //
                // `truncate(from); extend(points)` keeps indices below `from` and the
                // sender only guarantees those are final *because* it froze them —
                // which it announced by the `from` of the frames in between. Miss one
                // of those and the retained prefix can hold a value that was still
                // provisional when it was last written, then get promoted to "frozen"
                // by a later frame's `from`. The path then has no hole and the right
                // length, and is quietly wrong in its middle — which is why testing
                // `from > path.len()` alone never caught it, and why the round-trip
                // test below does.
                //
                // Frames from one client are numbered without gaps and `Peers::merge`
                // only ever hands over strictly newer ones, so "nothing was lost" is
                // exactly "this seq follows the last one spliced".
                if !whole && self.last_seq.is_none_or(|last| last + 1 != seq) {
                    // What we already hold is a true prefix of the stroke, every
                    // control point of it final and sent by the author for this
                    // gesture. So it stays on the canvas and simply stops growing,
                    // rather than blinking out for up to a `GESTURE_RESYNC`. Short is
                    // provisional, which a live preview always is; absent is a
                    // flicker. The resync frame repairs it.
                    return changed;
                }
                if from > assembly.path.len() {
                    // Unreachable from an honest sender once continuity holds — its
                    // `from` is a frozen count, which never exceeds its own path. Kept
                    // because the frame came off a wire.
                    return changed;
                }
                self.last_seq = Some(seq);
                assembly.path.truncate(from);
                assembly.path.extend(points);
                // The watermark only ever advances (`max`) — a resync frame, whose
                // `from` is 0, leaves the carried-over count standing. Clamped rather
                // than asserted against the path: an honest sender never resends
                // fewer points than it froze, but the frame came off a wire.
                assembly.frozen = assembly.frozen.max(from).min(assembly.path.len());
                self.drawn = Some(LiveGesture::Stroke(StrokeRecord {
                    layer: assembly.head.layer,
                    tool: assembly.head.tool,
                    brush: assembly.head.brush,
                    path: assembly.path.clone(),
                    seed: assembly.head.seed,
                }));
                true
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::command::InputSample;
    use crate::document::{ActorId, LayerId, Tool};
    use crate::geom::{Extent2, Vec2, ViewTransform};
    use crate::path::DEFAULT_TOLERANCE;
    use crate::peer::{PeerFrame, Peers};
    use crate::session::Session;

    /// A fixed-seed generator, so a failure is a bug report rather than a coin toss.
    /// `proptest` would add shrinking, but the inputs here are already minimal —
    /// what varies is *which frames survive the channel* — and the project keeps its
    /// dev-dependencies to the two it cannot do without.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 33
        }

        /// True with probability `n`/256.
        fn chance(&mut self, n: u64) -> bool {
            self.next() % 256 < n
        }
    }

    /// The invariant that binds the two halves, exercised across a lossy channel.
    ///
    /// Drives a real [`Session`] with pointer samples, publishes on a tick, and
    /// pushes the frames through a channel that drops, duplicates and delays them
    /// into a real [`Peers`]. Three things must hold at every step, whatever the
    /// channel did:
    ///
    /// 1. the receiver agrees with the sender **exactly** on the frozen prefix — the
    ///    part the protocol promises never moves — and never holds more control
    ///    points than the sender ever had. Only the frozen part: the provisional tail
    ///    legitimately differs, because the fit is still refining it and the receiver
    ///    holds an older snapshot of it. (Asserting agreement on the whole path is
    ///    what this test rejected first, which is the distinction the watermark
    ///    exists to draw.)
    /// 2. its frozen watermark never outruns the path it indexes;
    /// 3. once a resync frame is delivered, the receiver's path matches the sender's
    ///    entirely — the protocol's whole promise, that loss costs latency and not
    ///    correctness;
    /// 4. within one gesture the watermark never walks back — the module-level rule
    ///    a resync frame used to violate by resetting the assembly's frozen count,
    ///    which discarded the renderer's cached head and redrew the whole stroke
    ///    from scratch once a second.
    #[test]
    fn round_trip_survives_a_lossy_channel() {
        for seed in 0..64u64 {
            let mut rng = Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
            let mut session =
                Session::new(ViewTransform::identity(Extent2::new(64, 64)), LayerId(0));
            let mut peers = Peers::new();
            let actor = ActorId(1);
            let mut inflight: Vec<PeerFrame> = Vec::new();
            let mut now = 0.0;
            // Three gestures back to back, so a restart has to be unambiguous too.
            for gesture in 0..3u32 {
                // Invariant 4's memory: the highest frozen count seen this gesture.
                let mut watermark = 0usize;
                let seedling = u64::from(gesture);
                session.start_stroke(
                    Tool::Brush,
                    InputSample::at(Vec2::new(0.0, f32::from(gesture as u16) * 10.0)),
                    seedling,
                    DEFAULT_TOLERANCE,
                );
                for step in 1..40u32 {
                    let t = f32::from(step as u16);
                    session.stroke_to(InputSample::at(Vec2::new(
                        t * 3.0,
                        f32::from(gesture as u16) * 10.0 + (t * 0.7).sin() * 12.0,
                    )));
                    now += 0.033;
                    if let Some(frame) = session.publish(now) {
                        // 25% loss, plus the occasional duplicate and a one-tick delay
                        // — everything a flood mesh is allowed to do to presence.
                        if !rng.chance(64) {
                            inflight.push(frame.clone());
                        }
                        if rng.chance(32) {
                            inflight.push(frame);
                        }
                    }
                    if !inflight.is_empty() && !rng.chance(48) {
                        for frame in inflight.drain(..) {
                            peers.merge(actor, frame, now);
                        }
                    }
                    check(&session, &peers, actor, &mut watermark);
                }
                session.end_stroke();
                now += 0.033;
                if let Some(frame) = session.publish(now) {
                    inflight.push(frame);
                }
                for frame in inflight.drain(..) {
                    peers.merge(actor, frame, now);
                }
                check(&session, &peers, actor, &mut watermark);
                assert!(
                    peers.get(actor).and_then(|p| p.live_stroke()).is_none(),
                    "seed {seed}: the frame clearing a finished gesture must land"
                );
            }
        }
    }

    /// Invariants 1, 2 and 4. The receiver may lag the sender, but never disagree —
    /// and never *unlearn*: `watermark` remembers the highest frozen count seen this
    /// gesture, which no later frame (a resync in particular) may fall below.
    fn check(session: &Session, peers: &Peers, actor: ActorId, watermark: &mut usize) {
        let Some(peer) = peers.get(actor) else { return };
        let Some(shown) = peer.live_stroke() else {
            return;
        };
        assert!(
            peer.live_frozen_spans() <= crate::path::span_count(shown.path.len()),
            "frozen watermark outran the path it indexes"
        );
        assert!(
            peer.live_frozen_points() >= *watermark,
            "the frozen watermark walked back within one gesture"
        );
        *watermark = peer.live_frozen_points();
        let Some(truth) = session.preview_record() else {
            // The sender has no stroke in flight, so what is drawn is a leftover from
            // one that ended — the clearing frame has not landed yet. It must still be
            // a coherent path, which the checks above have already established.
            return;
        };
        assert!(
            shown.path.len() <= truth.path.len(),
            "the receiver holds more control points than the sender ever had"
        );
        let frozen = peer.live_frozen_points();
        assert!(frozen <= shown.path.len());
        assert_eq!(
            shown.path[..frozen],
            truth.path[..frozen],
            "a frozen control point never moves, so the two ends must agree on every \
             one the receiver has been told about — no hole, and no splice across \
             gestures"
        );
    }

    /// Invariant 3, isolated: a resync repairs a receiver that has missed everything.
    #[test]
    fn a_resync_makes_the_receiver_exact() {
        let mut session = Session::new(ViewTransform::identity(Extent2::new(64, 64)), LayerId(0));
        let mut peers = Peers::new();
        let actor = ActorId(1);
        session.start_stroke(
            Tool::Brush,
            InputSample::at(Vec2::ZERO),
            7,
            DEFAULT_TOLERANCE,
        );
        let mut now = 0.0;
        // Every frame of the first half-second is dropped on the floor.
        for step in 1..20u32 {
            session.stroke_to(InputSample::at(Vec2::new(
                f32::from(step as u16) * 4.0,
                0.0,
            )));
            now += 0.025;
            let _ = session.publish(now);
        }
        assert!(peers.get(actor).is_none(), "nothing has been delivered");

        // The next resync frame is delivered, and it alone must be enough.
        if let Some(resync_interval) = crate::peer::GESTURE_RESYNC {
            now += resync_interval;
            session.stroke_to(InputSample::at(Vec2::new(120.0, 0.0)));
            let frame = session.publish(now).expect("a resync frame is due");
            peers.merge(actor, frame, now);

            let truth = session.preview_record().expect("in flight");
            let shown = peers
                .get(actor)
                .and_then(|p| p.live_stroke())
                .expect("repaired from one frame");
            assert_eq!(
                shown.path, truth.path,
                "a resync carries the whole path, so one frame is a complete repair"
            );
        }
    }
}
