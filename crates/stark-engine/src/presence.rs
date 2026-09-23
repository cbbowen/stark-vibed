//! The presence gesture protocol — **both ends, adjacent** (§17.5).
//!
//! [`GestureTx`] encodes the gesture a client has in flight into [`GestureFrame`]s;
//! [`GestureRx`] reassembles them. The rules that make it work are properties of the
//! *pair*, checkable from neither end alone, which is why both ends live here:
//!
//! - a `head` rides exactly the frames that start from index 0;
//! - only **frozen** control points may be counted as sent, because the provisional
//!   tail can still move (§6.2);
//! - the frozen watermark only ever advances, so a resync frame (which says nothing
//!   new about what is frozen) cannot walk it back;
//! - a watermark may only advance for a frame that is actually **sent**;
//! - a delta may only be spliced onto the frame that *immediately* precedes it,
//!   because the retained prefix is trusted on the strength of `from` values the
//!   receiver may never have seen.
//!
//! [`PresenceTx`] is the frame around the gesture: the name, the cursor, and the
//! latch that decides when a [`PeerFrame`] goes out at all (§17.4, §17.5). It holds
//! the [`GestureTx`], so the whole sending half is one value the engine owns beside
//! the roster; the receiving half of the *frame* is
//! [`Peers::merge`](crate::peer::Peers::merge).
//!
//! The wire *types* stay in [`crate::peer`]. What lives here is the state that
//! interprets them.

use crate::gesture::LiveGesture;
use crate::path::frozen_spans_for;
use crate::peer::{Identity, default_name};
use stark_model::document::{ActorId, LayerId, StrokeRecord};
use stark_model::geom::Vec2;
use stark_model::path::ControlPoint;
use stark_model::peer::{GESTURE_RESYNC, GestureFrame, HEARTBEAT, PeerFrame, StrokeHead};

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

    /// Encode the gesture `live` for the wire, advancing the watermarks.
    ///
    /// `live` is the session's own answer
    /// ([`Session::live`](crate::session::Session::live)) — the gesture and how many
    /// of its leading control points are final (§6.2) — so the wire reads the same
    /// value the preview fold draws rather than a second window onto the fitter.
    ///
    /// **Call this only for a frame that is actually going out.** The watermarks
    /// record what the receiver has been *told*; advancing them for a frame the caller
    /// then drops skips control points nobody saw, and the gap stands until the next
    /// resync.
    pub(crate) fn encode(
        &mut self,
        id: u64,
        live: Option<(LiveGesture, usize)>,
        resync: bool,
    ) -> Option<GestureFrame> {
        let Some((gesture, frozen)) = live else {
            self.sent_id = None;
            self.sent = 0;
            return None;
        };
        match gesture {
            LiveGesture::Selection(op) => {
                // A marquee or lasso goes whole: it is already decimated, and unlike a
                // stroke its tail is not append-only — the closing edge follows the
                // cursor, so there is no frozen prefix to exploit.
                self.sent_id = Some(id);
                self.sent = 0;
                Some(GestureFrame::Selection { id, op })
            }
            // A fill's region is a shape, sent whole for exactly the reasons a
            // selection's is. **The layer is dropped here**: the frame's own
            // `PeerFrame::active_layer` is it, and a second copy could disagree with
            // that one. The pinned layer therefore travels only in-process, so a
            // peer's preview can drift from the commit across the retarget race the
            // pin closes locally — a preview, not the document (§17.5).
            LiveGesture::Fill {
                op, translation, ..
            } => {
                self.sent_id = Some(id);
                self.sent = 0;
                Some(GestureFrame::Fill {
                    id,
                    op,
                    translation,
                })
            }
            LiveGesture::Stroke(rec) => {
                let frozen = frozen.min(rec.path.len());
                debug_assert!(
                    self.sent_id != Some(id) || frozen >= self.sent,
                    "the frozen watermark must not walk back within one gesture"
                );
                // A frame is *fresh* — it carries the head and starts from 0 — for a
                // gesture the wire has not carried yet, and on every resync.
                let fresh = resync || self.sent_id != Some(id);
                let from = if fresh {
                    0
                } else {
                    self.sent.min(rec.path.len())
                };
                let points = rec.path[from..].to_vec();
                self.sent_id = Some(id);
                self.sent = frozen;
                Some(GestureFrame::Stroke {
                    id,
                    // The record's invariant half, split out only for a frame that
                    // carries one — which most are not. Boxed because a `StrokeHead`
                    // is several times the rest of the frame.
                    head: fresh.then(|| {
                        Box::new(StrokeHead {
                            layer: rec.layer,
                            brush: rec.brush,
                            seed: rec.seed,
                            translation: rec.translation,
                        })
                    }),
                    from: from as u32,
                    points,
                    // Per frame rather than in the head: a curve *parameter* names a
                    // different place as the path grows, and it refines until the
                    // entry spans freeze (§17.5).
                    start: rec.start,
                })
            }
        }
    }
}

/// What this client last put on the wire, for change detection (see
/// [`PresenceTx::publish`]). Compared rather than dirty-flagged: a comparison cannot
/// be forgotten at a call site, and these are three cheap fields.
///
/// The gesture is deliberately absent — what the wire has been told about it is
/// [`GestureTx::in_flight`]'s answer, not a second copy to keep in step.
#[derive(Clone, PartialEq)]
struct Published {
    active_layer: LayerId,
    cursor: Option<Vec2>,
    name: String,
}

impl Published {
    /// Whether what was last published still describes the client — **the one place
    /// the published field list lives**, because [`PresenceTx::publish_due`] asks the
    /// same question and must never answer `false` where `publish` would produce a
    /// frame.
    ///
    /// Borrows the name rather than taking one, so a caller can ask before deciding to
    /// allocate.
    fn matches(&self, active_layer: LayerId, cursor: Option<Vec2>, name: &str) -> bool {
        self.active_layer == active_layer && self.cursor == cursor && self.name == name
    }
}

/// The sending half of presence (§17.4): what this client publishes about itself
/// that is not a gesture — its name and its cursor — and the latch that decides when
/// a [`PeerFrame`] goes out.
///
/// The active layer and the gesture in flight ride every frame too, but they are the
/// session's own facts and arrive as arguments rather than as a second copy here.
/// `Engine` holds both and is where they meet.
pub(crate) struct PresenceTx {
    /// This client's display name.
    ///
    /// Empty only before any identity is adopted and anything typed — a solo client,
    /// which publishes nothing. Past that it is always a name a peer can show, because
    /// `None` on the wire means *unchanged* and so cannot be used to clear one.
    ///
    /// Private because it has an invariant `name_chosen` carries — see
    /// [`Self::set_name`].
    name: String,
    /// Whether [`name`](Self::name) is one the *user* chose rather than the
    /// id-derived default. Sharing or joining a document mints this client a new
    /// actor id and wants to refresh that default, but must not overwrite a name
    /// somebody typed.
    name_chosen: bool,
    /// When the name last rode a frame. Its own clock beside [`sent_at`](Self::sent_at):
    /// a gesture publishes thirty times a second, so the frame heartbeat never fires
    /// while one is in flight, and the name would never be repeated to a client that
    /// joined during it.
    name_sent_at: f64,
    /// Who this client is, for the name [`default_name`] derives from it. The
    /// [`Identity`]'s other half, kept because clearing the name has to write that
    /// default *then*, not leave the wire to infer it.
    actor: ActorId,
    /// Where this client's pointer is, canvas space; `None` when it is off the canvas.
    ///
    /// Sent to peers and read on *this* side too — a guide draws its rays through the
    /// hand (§20.9) — so both are hung on one fact rather than two that could disagree.
    cursor: Option<Vec2>,
    // The latch, not a queue (§17.5).
    published: Option<Published>,
    /// Which run of this client is publishing; see [`Identity::boot`]. Rides every
    /// frame so a peer can order this run's frames against the previous one's.
    boot: u64,
    seq: u64,
    sent_at: f64,
    /// The gesture protocol's sending half — the watermarks recording what the wire
    /// has been told about the gesture.
    gesture: GestureTx,
}

impl PresenceTx {
    pub(crate) fn new() -> Self {
        Self {
            name: String::new(),
            name_chosen: false,
            name_sent_at: f64::NEG_INFINITY,
            actor: ActorId::SOLO,
            cursor: None,
            published: None,
            boot: 0,
            seq: 0,
            sent_at: f64::NEG_INFINITY,
            gesture: GestureTx::new(),
        }
    }

    /// This client's display name, as peers see it.
    #[cfg(test)]
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    /// Set the display name the user chose. **Sticky**: hosting or joining mints this
    /// client a new actor id, and [`adopt_identity`](Self::adopt_identity) will not
    /// overwrite a name set here.
    ///
    /// Setting it empty gives the choice back, and **writes the id-derived default
    /// here and now** rather than leaving the field empty: an empty name is `None` on
    /// the wire ([`PeerFrame::sanitized`](stark_model::peer::PeerFrame::sanitized)),
    /// `None` means *unchanged*, and peers would go on showing the name that was just
    /// cleared.
    pub(crate) fn set_name(&mut self, name: &str) {
        let name = name.trim();
        self.name_chosen = !name.is_empty();
        self.name.clear();
        if self.name_chosen {
            self.name.push_str(name);
        } else {
            self.name.push_str(&default_name(self.actor));
        }
    }

    /// Where this client's pointer is, or `None` when it is off the canvas.
    ///
    /// **Non-finite positions are refused here**, since `screen_to_canvas` can produce
    /// one and this is the last gate before both the wire and the guide rays drawn
    /// through the cursor (§20.9).
    pub(crate) fn set_cursor(&mut self, at: Option<Vec2>) {
        self.cursor = at.filter(|p| p.is_finite());
    }

    /// Where this client's pointer is.
    pub(crate) fn cursor(&self) -> Option<Vec2> {
        self.cursor
    }

    /// Adopt the identity a session has given this client.
    ///
    /// Records the run counter every published frame carries, and takes the id-derived
    /// name as a default — unless the user has chosen one, which stands.
    ///
    /// **The latch starts over**, because a new identity is a new audience: nobody who
    /// will hear the next frame heard the last one. Left standing, a name chosen before
    /// a re-share would read as already published and never go out again.
    pub(crate) fn adopt_identity(&mut self, identity: Identity) {
        self.boot = identity.boot;
        self.actor = identity.actor;
        if !self.name_chosen {
            self.name = default_name(identity.actor);
        }
        self.published = None;
        self.name_sent_at = f64::NEG_INFINITY;
    }

    /// The frame to send, if anything a peer would care about has changed since the
    /// last call — otherwise `None` (§17.5).
    ///
    /// A **latch, not a queue**: it reports the *current* state, and the path delta is
    /// computed here against what has actually been sent. A pen reporting at 240 Hz
    /// against a 30 Hz publish tick therefore coalesces losslessly — eight moves
    /// produce one frame carrying all eight control points.
    ///
    /// A frame goes out when something changed, when a gesture is in flight (its path
    /// just grew), or every [`HEARTBEAT`] regardless, so a silent peer still proves it
    /// is here. `ordinal` and `live` are the session's gesture in flight, as
    /// [`GestureTx::encode`] takes them.
    pub(crate) fn publish(
        &mut self,
        now: f64,
        active_layer: LayerId,
        ordinal: u64,
        live: Option<(LiveGesture, usize)>,
    ) -> Option<PeerFrame> {
        // Repairs any receiver that missed a delta and primes any that arrived
        // mid-stroke, without either having to ask — which is what keeps the wire
        // one-way and the sender stateless about its audience.
        let resync = self.gesture.resync_due(now);

        // Asked before anything is built, so the overwhelming majority of ticks — the
        // ones returning `None` two lines below — allocate nothing.
        let changed = self
            .published
            .as_ref()
            .is_none_or(|p| !p.matches(active_layer, self.cursor, &self.name))
            // A live gesture's path grows every move, so its frame always differs.
            || live.is_some()
            // ...and one that has just ended still owes the frame that clears it.
            || self.gesture.in_flight();
        if !changed && now - self.sent_at < HEARTBEAT {
            return None;
        }

        // The name rides a frame when it changed, on a resync, and **once every
        // [`HEARTBEAT`] regardless**. A peer that already knows it does not need it
        // thirty times a second; a peer that started listening after the frame that
        // carried it has no way to ask, and nothing else repairs that — the gesture
        // resync would, but [`GESTURE_RESYNC`] is `None`. So the repair is time.
        let name_due = resync
            || now - self.name_sent_at >= HEARTBEAT
            || self.published.as_ref().is_none_or(|p| p.name != self.name);
        // An empty name is the pre-identity state alone, and rides no frame: `None` is
        // what the receiver reads it as anyway, and `None` means *unchanged*.
        let name = (name_due && !self.name.is_empty()).then(|| self.name.clone());

        // Everything below this line commits to sending, which is the only point at
        // which the watermarks may move: they record what the receiver has been told.
        let gesture = self.gesture.encode(ordinal, live, resync);
        if resync {
            self.gesture.stamp_resync(now);
        }
        if name.is_some() {
            self.name_sent_at = now;
        }
        self.sent_at = now;
        self.seq += 1;
        self.published = Some(Published {
            active_layer,
            cursor: self.cursor,
            name: self.name.clone(),
        });
        Some(PeerFrame {
            boot: self.boot,
            seq: self.seq,
            name,
            active_layer,
            cursor: self.cursor,
            gesture,
            leaving: false,
        })
    }

    /// Whether [`publish`](Self::publish) could produce a frame. `gesturing` is
    /// whether the session has a gesture in flight — a stroke or a shape drag.
    ///
    /// Deliberately **conservative**: it may say yes where `publish` then returns
    /// `None`, but it must never say no where `publish` would have produced a frame —
    /// a pump that trusts it would drop that frame on the floor.
    ///
    /// Exists so an idle session costs nothing: without it the pump takes a mutable
    /// borrow of the engine thirty times a second to learn there was nothing to send.
    pub(crate) fn publish_due(&self, now: f64, active_layer: LayerId, gesturing: bool) -> bool {
        now - self.sent_at >= HEARTBEAT
            || gesturing
            // A gesture that just ended: the frame clearing it is the one that stops
            // peers drawing a stroke nobody is making any more.
            || self.gesture.in_flight()
            || self
                .published
                .as_ref()
                .is_none_or(|p| !p.matches(active_layer, self.cursor, &self.name))
    }

    /// The farewell frame: one publish that removes this client from every peer's
    /// roster at once, rather than making them wait out
    /// [`PEER_TIMEOUT`](stark_model::peer::PEER_TIMEOUT).
    pub(crate) fn publish_leaving(&mut self, active_layer: LayerId) -> PeerFrame {
        self.seq += 1;
        PeerFrame {
            boot: self.boot,
            seq: self.seq,
            name: None,
            active_layer,
            cursor: None,
            gesture: None,
            leaving: true,
        }
    }
}

/// The receiving half: one peer's gesture, reassembled from the frames that arrived.
///
/// **The states are the protocol's, so an invalid combination cannot be built.** They
/// were four `Option`s whose valid pairings lived in prose — a selection left no
/// assembly, a stroke whose head had not arrived left nothing drawn, and the last seq
/// spliced meant nothing at all outside a stroke.
#[derive(Clone, Debug, Default)]
pub(crate) enum GestureRx {
    #[default]
    Idle,
    /// A marquee, lasso or fill region. Sent whole (`GestureTx::encode`), so there is
    /// nothing to assemble and what arrived *is* what to draw.
    Region { id: u64, drawn: LiveGesture },
    /// A stroke. `assembly` is `None` until a head arrives — a delta for a gesture
    /// this receiver joined mid-flight names an ordinal but draws nothing.
    Stroke { id: u64, assembly: Option<Assembly> },
}

/// A stroke being reassembled: the record the fold will draw, spliced into in place.
#[derive(Clone, Debug)]
pub(crate) struct Assembly {
    /// The very type the author committed from, so the two ends cannot disagree about
    /// what a live stroke is — and the path the frames splice into, rather than a
    /// second buffer copied into it per frame.
    record: StrokeRecord,
    /// Control points the sender has stopped resending — i.e. the ones its fitter
    /// froze. The receiver learns this for free from the delta's `from`, and it is
    /// exactly what the incremental repaint needs to know (§17.6).
    frozen: usize,
    /// [`PeerFrame::seq`](stark_model::peer::PeerFrame::seq) of the last frame spliced
    /// in, or `None` for an assembly nothing has reached yet. A delta is only safe to
    /// splice if it *immediately* follows the frame before it — see [`Self::splice`].
    last_seq: Option<u64>,
}

impl Assembly {
    /// The assembly a head opens: the record's invariant half, an empty path, and the
    /// frozen watermark `carried` over from the assembly this one replaces.
    fn opening(head: StrokeHead, start: f32, carried: usize) -> Self {
        Self {
            record: StrokeRecord {
                layer: head.layer,
                brush: head.brush,
                path: Vec::new(),
                seed: head.seed,
                start,
                // The sender's: the path frames carry is in the layer's frame, and the
                // fold needs the very offset the author's commit will use (§14.12).
                translation: head.translation,
            },
            frozen: carried,
            last_seq: None,
        }
    }

    /// Splice one frame's points in, answering whether the path moved. `whole` is
    /// whether the frame resends the path entire and so stands on its own.
    fn splice(
        &mut self,
        seq: u64,
        whole: bool,
        from: usize,
        points: Vec<ControlPoint>,
        start: f32,
    ) -> bool {
        // **A delta is only safe to splice if nothing was lost before it.**
        // `truncate(from); extend(points)` keeps everything below `from`, which is
        // sound only because the sender froze those points — a fact it announced
        // through the `from` of the frames in between. Miss one and a still-provisional
        // control point is retained, then promoted to "frozen" by a later frame's
        // `from`: right length, no hole, wrong in the middle.
        //
        // Frames from one client are numbered without gaps and `Peers::merge` only ever
        // hands over strictly newer ones, so "nothing was lost" is exactly "this seq
        // follows the last one spliced".
        //
        // What is held is still a true prefix of the stroke, so a refusal leaves it on
        // the canvas and stops it growing until the next resync repairs it. Short is
        // provisional, which a live preview always is; absent is a flicker.
        if !whole && self.last_seq.is_none_or(|last| last + 1 != seq) {
            return false;
        }
        if from > self.record.path.len() {
            // Unreachable from an honest sender once continuity holds — its `from` is a
            // frozen count, which never exceeds its own path. Kept because the frame
            // came off a wire.
            return false;
        }
        self.last_seq = Some(seq);
        self.record.path.truncate(from);
        self.record.path.extend(points);
        // The marker as this frame states it — newest wins, unlike the watermark below,
        // because it is a statement about the curve rather than a count of what was sent.
        self.record.start = start;
        // The watermark only ever advances (`max`) — a resync frame, whose `from` is 0,
        // leaves the carried-over count standing. Clamped rather than asserted against
        // the path: an honest sender never resends fewer points than it froze, but the
        // frame came off a wire.
        self.frozen = self.frozen.max(from).min(self.record.path.len());
        true
    }
}

impl GestureRx {
    /// The ordinal being tracked — the only thing that reliably tells this gesture
    /// from the one before it.
    fn id(&self) -> Option<u64> {
        match self {
            Self::Idle => None,
            Self::Region { id, .. } | Self::Stroke { id, .. } => Some(*id),
        }
    }

    /// What to draw for this peer and which gesture it is, or `None` when nothing is.
    /// The pair, because the ordinal is what tells a cached render of this gesture from
    /// a render of the one before it, and the two are present together or not at all.
    ///
    /// **The one copy of the path the receiving side makes**: it is assembled in place
    /// and cloned here, where the fold takes ownership of it.
    pub(crate) fn drawn(&self) -> Option<(u64, LiveGesture)> {
        match self {
            Self::Idle | Self::Stroke { assembly: None, .. } => None,
            Self::Region { id, drawn } => Some((*id, drawn.clone())),
            Self::Stroke {
                id,
                assembly: Some(a),
            } => Some((*id, LiveGesture::Stroke(a.record.clone()))),
        }
    }

    /// Whether there is a gesture on the canvas for this peer.
    pub(crate) fn is_drawing(&self) -> bool {
        match self {
            Self::Idle | Self::Stroke { assembly: None, .. } => false,
            Self::Region { .. } | Self::Stroke { .. } => true,
        }
    }

    /// The live stroke being reassembled, if that is what this peer is drawing.
    pub(crate) fn live_stroke(&self) -> Option<&StrokeRecord> {
        match self {
            Self::Idle | Self::Region { .. } | Self::Stroke { assembly: None, .. } => None,
            Self::Stroke {
                assembly: Some(a), ..
            } => Some(&a.record),
        }
    }

    /// How many leading control points the sender has declared final. Conservative:
    /// learned from a delta's `from`, the sender's frozen count as of the *previous*
    /// frame, so it lags the truth rather than outrunning it. A renderer wants
    /// [`frozen_spans`](Self::frozen_spans) instead.
    #[cfg(test)]
    pub(crate) fn frozen(&self) -> usize {
        match self {
            Self::Idle | Self::Region { .. } | Self::Stroke { assembly: None, .. } => 0,
            Self::Stroke {
                assembly: Some(a), ..
            } => a.frozen,
        }
    }

    /// How many spans of the reassembled stroke are settled, so the preview can
    /// repaint only the tail (§6.2, §17.6).
    pub(crate) fn frozen_spans(&self) -> usize {
        match self {
            Self::Idle | Self::Region { .. } | Self::Stroke { assembly: None, .. } => 0,
            Self::Stroke {
                assembly: Some(a), ..
            } => frozen_spans_for(a.frozen, a.record.path.len()),
        }
    }

    /// Take the gesture down. Returns whether anything was being drawn.
    pub(crate) fn clear(&mut self) -> bool {
        let was_drawing = self.is_drawing();
        *self = Self::Idle;
        was_drawing
    }

    /// Integrate one frame, carrying the [`PeerFrame::seq`](stark_model::peer::PeerFrame::seq)
    /// it arrived under. Returns whether what is *drawn* changed.
    pub(crate) fn apply(&mut self, frame: GestureFrame, seq: u64, active_layer: LayerId) -> bool {
        // A new ordinal is a different gesture: drop whatever was being drawn or
        // assembled rather than splicing two of them together. Keyed on the ordinal
        // rather than on the stroke assembly, because a *selection* leaves no assembly
        // and its marquee would then outlive the gesture that drew it.
        let id = frame.id();
        let changed = if self.id() == Some(id) {
            false
        } else {
            self.clear()
        };
        match frame {
            GestureFrame::Selection { op, .. } => {
                *self = Self::Region {
                    id,
                    drawn: LiveGesture::Selection(op),
                };
                true
            }
            GestureFrame::Fill {
                op, translation, ..
            } => {
                *self = Self::Region {
                    id,
                    drawn: LiveGesture::Fill {
                        layer: active_layer,
                        op,
                        translation,
                    },
                };
                true
            }
            GestureFrame::Stroke {
                head,
                from,
                points,
                start,
                ..
            } => {
                let from = from as usize;
                debug_assert!(
                    head.is_none() || from == 0,
                    "a head rides only a frame that starts the path over"
                );
                // A frame that resends the path whole stands on its own; a delta has to
                // be spliced onto what came before it.
                let whole = head.is_some() && from == 0;
                // A *region* under this ordinal has no assembly and keeps its outline
                // unless the head below replaces it.
                let held = match self {
                    Self::Stroke { assembly, .. } => assembly.take(),
                    Self::Idle | Self::Region { .. } => None,
                };
                // Start — or restart, on a resync frame — whenever the head is present.
                // A resync restarts the *assembly*, not the gesture: within one ordinal
                // the frozen watermark carries over, because a frozen control point
                // never moves and a resync says nothing new about freezing. Reset to
                // zero it would discard the renderer's cached head, which keys on
                // `frozen_spans`, once per resync per peer.
                //
                // The head arrives already sanitized: `Peers::merge` is the one door,
                // and gating here would cover a stroke's brush while leaving the frame's
                // other two free-form payloads as they arrived (§21.5).
                let opened = match head {
                    Some(head) if held.is_none() || from == 0 => Some(Assembly::opening(
                        *head,
                        start,
                        held.map_or(0, |a| a.frozen),
                    )),
                    Some(_) | None => held,
                };
                let Some(mut assembly) = opened else {
                    // A delta for a gesture whose head we never saw. Nothing to draw
                    // until the next resync frame — and nothing of the *previous*
                    // gesture either, which the ordinal check above has seen to.
                    if let Self::Idle | Self::Stroke { .. } = self {
                        *self = Self::Stroke { id, assembly: None };
                    }
                    return changed;
                };
                let spliced = assembly.splice(seq, whole, from, points, start);
                *self = Self::Stroke {
                    id,
                    // An assembly nothing has ever reached holds no stroke: continuity
                    // refuses every later delta, so only the next head can fill it, and
                    // that opens a new one regardless.
                    assembly: assembly.last_seq.is_some().then_some(assembly),
                };
                changed || spliced
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::command::{InputSample, Tool};
    use crate::path::DEFAULT_TOLERANCE;
    use crate::peer::Peers;
    use crate::session::{Outline, Session};
    use stark_model::document::{ActorId, LayerId, ShapeAction};
    use stark_model::geom::Vec2;
    use stark_model::peer::PeerFrame;

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

    /// The two halves of presence, put together the way `Engine` puts them: what the
    /// session is dragging out and where it is aimed, handed to the publisher.
    fn publish(tx: &mut PresenceTx, session: &Session, now: f64) -> Option<PeerFrame> {
        tx.publish(
            now,
            session.active_layer(),
            session.gesture_ordinal(),
            session.live(),
        )
    }

    /// The invariant that binds the two halves, exercised across a lossy channel.
    ///
    /// Drives a real [`Session`] with pointer samples, publishes on a tick, and pushes
    /// the frames through a channel that drops, duplicates and delays them into a real
    /// [`Peers`]. Whatever the channel did:
    ///
    /// 1. the receiver agrees with the sender **exactly** on the frozen prefix, and
    ///    never holds more control points than the sender ever had. Only the frozen
    ///    part: the provisional tail legitimately differs, since the fit is still
    ///    refining it while the receiver holds an older snapshot;
    /// 2. its frozen watermark never outruns the path it indexes;
    /// 3. once a resync frame is delivered the receiver's path matches the sender's
    ///    entirely — loss costs latency, not correctness;
    /// 4. within one gesture the watermark never walks back, which a resync frame is
    ///    the most likely to break.
    #[test]
    fn round_trip_survives_a_lossy_channel() {
        for seed in 0..64u64 {
            let mut rng = Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
            let mut session = Session::new(LayerId::ROOT);
            let mut tx = PresenceTx::new();
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
                    InputSample::at(Vec2::new(0.0, f32::from(gesture as u16) * 10.0)),
                    seedling,
                    DEFAULT_TOLERANCE,
                    0.0,
                    stark_model::geom::IVec2::ZERO,
                );
                for step in 1..40u32 {
                    let t = f32::from(step as u16);
                    session.gesture_to(InputSample::at(Vec2::new(
                        t * 3.0,
                        f32::from(gesture as u16) * 10.0 + (t * 0.7).sin() * 12.0,
                    )));
                    now += 0.033;
                    if let Some(frame) = publish(&mut tx, &session, now) {
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
                session.end_gesture();
                now += 0.033;
                if let Some(frame) = publish(&mut tx, &session, now) {
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

    /// The control points a receiver would draw.
    fn shown(rx: &GestureRx) -> Vec<ControlPoint> {
        rx.live_stroke()
            .map(|rec| rec.path.clone())
            .unwrap_or_default()
    }

    /// The latch (§17.5), driven without a `Session`: a frame goes out on the first
    /// tick, on a change, and on the heartbeat — and never otherwise. `publish_due`
    /// is asked before every `publish`, because its one rule is that it may not say
    /// no where `publish` then produces a frame.
    #[test]
    fn the_latch_sends_on_change_and_on_the_heartbeat_only() {
        fn tick(presence: &mut PresenceTx, now: f64) -> Option<PeerFrame> {
            let due = presence.publish_due(now, LayerId::ROOT, false);
            let frame = presence.publish(now, LayerId::ROOT, 0, None);
            assert!(
                due || frame.is_none(),
                "publish_due said no at {now} and publish produced a frame"
            );
            frame
        }
        let mut presence = PresenceTx::new();

        let first = tick(&mut presence, 0.0).expect("the first tick proves the client is here");
        assert_eq!(first.seq, 1);
        assert_eq!(first.name, None, "an unset name rides no frame");
        assert!(
            tick(&mut presence, HEARTBEAT * 0.1).is_none(),
            "nothing changed"
        );
        let beat = tick(&mut presence, HEARTBEAT).expect("a silent peer still proves it is here");
        assert_eq!(beat.seq, 2);

        presence.set_name(" ada ");
        let named = tick(&mut presence, HEARTBEAT * 1.1).expect("a new name is a change");
        assert_eq!(named.name.as_deref(), Some("ada"), "the name is trimmed");
        presence.set_cursor(Some(Vec2::new(f32::NAN, 1.0)));
        assert!(
            tick(&mut presence, HEARTBEAT * 1.2).is_none(),
            "a NaN cursor is refused at the door, so nothing changed"
        );
        presence.set_cursor(Some(Vec2::new(2.0, 3.0)));
        let moved = tick(&mut presence, HEARTBEAT * 1.3).expect("a cursor move is a change");
        assert_eq!(moved.cursor, Some(Vec2::new(2.0, 3.0)));
        assert_eq!(
            moved.name, None,
            "a peer that already knows the name is not told it again"
        );

        let bye = presence.publish_leaving(LayerId::ROOT);
        assert!(bye.leaving);
        assert_eq!(bye.seq, 5, "the farewell follows in sequence");
        assert_eq!(bye.cursor, None);
    }

    /// **A name is repaired, not announced once.** A client that starts listening
    /// after the frame that carried one has no way to ask for it, and nothing else
    /// tells it: [`GESTURE_RESYNC`] is `None`, so the only repair the protocol has is
    /// the heartbeat. Without it a late joiner shows the id-derived default for the
    /// rest of the session.
    #[test]
    fn a_late_listener_learns_the_name_within_a_heartbeat() {
        let actor = ActorId(1);
        let mut tx = PresenceTx::new();
        tx.adopt_identity(Identity::new(actor, 1));
        tx.set_name("ada");

        let first = tx.publish(0.0, LayerId::ROOT, 0, None).expect("a frame");
        assert_eq!(first.name.as_deref(), Some("ada"), "the naming frame");

        // A receiver that was not there for it. The sender knows nothing about its
        // audience, so only time says the name again: the publish cadence, run out
        // past one heartbeat.
        const TICK: f64 = 0.033;
        let mut late = Peers::new();
        let mut now = 0.0;
        for _ in 0..64 {
            now += TICK;
            if let Some(frame) = tx.publish(now, LayerId::ROOT, 0, None) {
                late.merge(actor, frame, now);
            }
        }
        assert!(now >= HEARTBEAT, "the run has to cross a heartbeat");
        assert_eq!(
            late.get(actor).expect("the peer arrived").name,
            "ada",
            "a listener that missed the naming frame must still learn the name",
        );
    }

    /// Clearing the name gives the id-derived default back **to the peers**, not just
    /// to this client: an empty name is `None` on the wire, `None` means *unchanged*,
    /// and everyone would go on showing the name that was just cleared.
    #[test]
    fn a_cleared_name_reaches_the_receiver() {
        let actor = ActorId(9);
        let mut tx = PresenceTx::new();
        tx.adopt_identity(Identity::new(actor, 1));
        tx.set_name("ada");
        let mut peers = Peers::new();
        let named = tx.publish(0.0, LayerId::ROOT, 0, None).expect("a frame");
        peers.merge(actor, named, 0.0);
        assert_eq!(peers.get(actor).expect("peer").name, "ada");

        tx.set_name("  ");
        let cleared = tx
            .publish(0.1, LayerId::ROOT, 0, None)
            .expect("a cleared name is a change");
        peers.merge(actor, cleared, 0.1);
        assert_eq!(
            peers.get(actor).expect("peer").name,
            default_name(actor),
            "peers must resume showing the id-derived default",
        );
    }

    /// **Re-sharing is a new audience.** A name published in the last session was
    /// heard by nobody who will hear this one's frames, so the latch starts over with
    /// the identity — it did not, which left a chosen name unsent for the whole of
    /// every session after the first.
    #[test]
    fn a_fresh_identity_republishes_the_name() {
        let mut tx = PresenceTx::new();
        tx.adopt_identity(Identity::new(ActorId(1), 1));
        tx.set_name("ada");
        let first = tx.publish(0.0, LayerId::ROOT, 0, None).expect("a frame");
        assert_eq!(first.name.as_deref(), Some("ada"));

        // Leaving and sharing again mints a fresh actor id; the chosen name stands.
        tx.adopt_identity(Identity::new(ActorId(2), 2));
        let rejoined = tx
            .publish(0.1, LayerId::ROOT, 0, None)
            .expect("a fresh identity is a change");
        assert_eq!(
            rejoined.name.as_deref(),
            Some("ada"),
            "the new session's peers have never been told it",
        );
    }

    /// [`PresenceTx::adopt_identity`] takes the id-derived name as a default and leaves
    /// a chosen one alone; emptying the name gives the choice back.
    #[test]
    fn adopting_an_identity_does_not_overwrite_a_chosen_name() {
        let mut presence = PresenceTx::new();
        presence.adopt_identity(Identity::new(ActorId(7), 3));
        assert_eq!(presence.name(), default_name(ActorId(7)));
        presence.set_name("ada");
        presence.adopt_identity(Identity::new(ActorId(8), 4));
        assert_eq!(presence.name(), "ada");
        presence.set_name("   ");
        presence.adopt_identity(Identity::new(ActorId(9), 5));
        assert_eq!(presence.name(), default_name(ActorId(9)));
        let bye = presence.publish_leaving(LayerId::ROOT);
        assert_eq!(bye.boot, 5, "the run counter is the newest identity's");
    }

    /// **What a gesture puts on the wire, pinned** — the fingerprint the sending half
    /// may not move (§17.5).
    ///
    /// A gossip message carries no schema (§8), so reshaping anything the presence
    /// channel touches means bumping the ALPN. Two records, because they fail for
    /// different reasons: [`WIRE_SHAPE`] is the protocol's own behaviour — which
    /// frames carry a head, where each delta starts, how many points it holds — and
    /// [`WIRE_PAYLOAD`] is one hash over every byte in every frame, which catches a
    /// value that moved inside a shape that did not.
    ///
    /// **Re-blessing either is only correct when the wire was meant to move**, and
    /// the ALPN moves with it.
    #[test]
    fn the_wire_a_gesture_puts_out_is_unchanged() {
        let (shape, payload) = published_gestures();
        assert_eq!(shape, WIRE_SHAPE, "the frames a gesture published changed");
        assert_eq!(payload, WIRE_PAYLOAD, "a frame's contents changed");
    }

    /// Every gesture frame a scripted stroke, lasso and fill published, as
    /// [`the_wire_a_gesture_puts_out_is_unchanged`] records them.
    const WIRE_SHAPE: &[&str] = &[
        "stroke id=1 head=true from=0 n=2",
        "stroke id=1 head=false from=1 n=1",
        "stroke id=1 head=false from=1 n=2",
        "stroke id=1 head=false from=1 n=3",
        "stroke id=1 head=false from=1 n=4",
        "stroke id=1 head=false from=1 n=5",
        "stroke id=1 head=false from=2 n=4",
        "stroke id=1 head=false from=2 n=5",
        "stroke id=1 head=false from=3 n=4",
        "stroke id=1 head=false from=3 n=4",
        "stroke id=1 head=false from=3 n=5",
        "cleared",
        "select id=2",
        "select id=2",
        "cleared",
        "fill id=3",
        "fill id=3",
        "fill id=3",
        "cleared",
    ];

    /// blake3 over all of those frames' contents, same run.
    const WIRE_PAYLOAD: &str = "817764edfcb6697fc3f27a32e29300f23c43a9d9c90d9b4c0ce9a9e15336206d";

    /// Drive a stroke, a lasso and a fill through a real [`Session`], publishing on a
    /// tick, and report what went out: one line per frame naming what it carries, and
    /// one hash over the whole run.
    fn published_gestures() -> (Vec<String>, String) {
        let mut session = Session::new(LayerId::ROOT);
        let mut tx = PresenceTx::new();
        let mut now = 0.0;
        let mut shape = Vec::new();
        let mut payload = blake3::Hasher::new();
        let mut tick = |session: &mut Session, shape: &mut Vec<String>| {
            now += 0.033;
            let Some(frame) = publish(&mut tx, session, now) else {
                return;
            };
            payload.update(format!("{:?}", frame.gesture).as_bytes());
            // A frame carrying no gesture is the one that *clears* one, which is as
            // much a part of the protocol as the deltas before it.
            let Some(gesture) = frame.gesture else {
                shape.push("cleared".to_owned());
                return;
            };
            shape.push(match &gesture {
                GestureFrame::Stroke {
                    id,
                    head,
                    from,
                    points,
                    ..
                } => format!(
                    "stroke id={id} head={} from={from} n={}",
                    head.is_some(),
                    points.len()
                ),
                GestureFrame::Selection { id, .. } => format!("select id={id}"),
                GestureFrame::Fill { id, .. } => format!("fill id={id}"),
            });
        };

        session.start_stroke(
            InputSample::at(Vec2::ZERO),
            7,
            DEFAULT_TOLERANCE,
            0.0,
            stark_model::geom::IVec2::ZERO,
        );
        for step in 1..12u32 {
            let t = f32::from(step as u16);
            session.gesture_to(InputSample::at(Vec2::new(t * 5.0, (t * 0.4).sin() * 20.0)));
            tick(&mut session, &mut shape);
        }
        session.end_gesture();
        tick(&mut session, &mut shape);

        for (action, tool, outline) in [
            (
                ShapeAction::default(),
                Tool::SelectLasso,
                Outline::Lasso(Vec::new()),
            ),
            (ShapeAction::Fill, Tool::SelectRect, Outline::Rect),
        ] {
            session.shape_action = action;
            session.start_selection(
                tool,
                outline,
                Vec2::ZERO,
                false,
                stark_model::geom::IVec2::ZERO,
            );
            for step in 1..4u32 {
                let t = f32::from(step as u16);
                session.gesture_to(InputSample::at(Vec2::new(t * 9.0, t * t * 3.0)));
                tick(&mut session, &mut shape);
            }
            session.end_gesture();
            tick(&mut session, &mut shape);
        }

        (shape, payload.finalize().to_hex().to_string())
    }

    /// Invariant 3, isolated: **one resync frame repairs a receiver that has missed
    /// everything.** The protocol's whole promise — loss costs latency, not
    /// correctness.
    ///
    /// Driven through the two halves directly rather than through [`Session::publish`],
    /// which decides *when* to resync from [`GESTURE_RESYNC`]. That constant is
    /// currently `None`, so a body gated on it would check nothing and still report
    /// `ok`. The mechanism is testable whatever the cadence is, because `encode` takes
    /// `resync` as a parameter.
    #[test]
    fn a_resync_repairs_a_receiver_that_missed_everything() {
        let mut tx = GestureTx::new();
        let mut rx = GestureRx::default();
        let layer = LayerId::ROOT;
        let path: Vec<ControlPoint> = (0..12)
            .map(|i| ControlPoint::at(Vec2::new(f32::from(i as u16) * 4.0, 0.0)))
            .collect();
        let stroke = |n: usize, frozen: usize| {
            (
                LiveGesture::Stroke(StrokeRecord {
                    layer,
                    brush: stark_model::document::BrushParams::default(),
                    path: path[..n].to_vec(),
                    seed: 7,
                    start: 0.25,
                    translation: stark_model::geom::IVec2::ZERO,
                }),
                frozen,
            )
        };

        // The gesture's first frame lands, and the receiver is exact.
        let first = tx.encode(0, Some(stroke(3, 1)), false).expect("a frame");
        assert!(rx.apply(first, 1, layer));
        assert_eq!(shown(&rx), path[..3], "the first frame carries the head");
        assert_eq!(
            rx.live_stroke()
                .expect("a stroke frame draws a stroke")
                .start,
            0.25,
            "the marker rides every frame into the drawn record"
        );

        // Then every frame for a while is sent and lost. The watermarks move — they
        // record what the wire was *told*, and the wire was told — so the sender goes
        // on trimming its deltas to a prefix the receiver never saw.
        for n in 4..=9 {
            tx.encode(0, Some(stroke(n, n - 2)), false)
                .expect("a frame");
        }
        assert_eq!(
            shown(&rx),
            path[..3],
            "nothing was delivered, so nothing grew"
        );
        let held = rx.frozen();

        // One resync frame — head present, whole path, `from` back to 0 — and it must
        // be a complete repair on its own. Its seq is far past the last one spliced,
        // which is exactly the gap a delta would be refused for: a whole frame stands
        // on its own and is not subject to the continuity rule.
        let repair = tx.encode(0, Some(stroke(12, 9)), true).expect("a frame");
        assert!(rx.apply(repair, 40, layer));
        assert_eq!(
            shown(&rx),
            path,
            "a resync carries the whole path, so one frame is a complete repair"
        );
        assert!(
            rx.frozen() >= held,
            "a resync says nothing new about what is frozen, so it must not walk the \
             watermark back"
        );
    }
}
