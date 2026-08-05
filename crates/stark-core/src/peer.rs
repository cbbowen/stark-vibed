//! Presence: per-client state that every client reads and only its owner writes
//! (§17.4).
//!
//! This is the **unlogged** half of per-client state. The rule that puts something
//! here rather than in [`DocState`](crate::document::DocState) is the one §4
//! already runs on — *does replay need it to reproduce pixels?* — and the answer for
//! everything in this module is no:
//!
//! - the **selected layer** is already closed over by [`StrokeRecord::layer`], so
//!   logging it would make every click in the layers panel an undo step for no
//!   reproducible consequence;
//! - a **cursor** paints nothing;
//! - a **live gesture** is by definition the thing that has not committed yet — when
//!   it does, the [`Action`](crate::document::Action) is authoritative and the live
//!   copy is discarded.
//!
//! (The selection *does* pass that test, so it lives in `DocState` keyed by actor —
//! see §17.3. It is the one piece of per-client state that is not here.)
//!
//! # Why this may be lossy
//!
//! **Nothing in the action log ever references presence.** That single invariant is
//! what lets the transport drop, coalesce, reorder or arbitrarily delay these frames
//! without touching convergence: the worst outcome of losing every presence frame in
//! a session is that it looks like a session without presence — strokes appear when
//! they commit. So the wire is free to shed presence first under congestion, and a
//! receiver is free to discard a frame it cannot use (see [`Peers::merge`]).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::document::{ActorId, BrushParams, FillOp, LayerId, SelectionOp, StrokeRecord};
use crate::geom::Vec2;
use crate::path::ControlPoint;
use crate::presence::GestureRx;

/// How long a peer may go unheard-from before it leaves the roster, in seconds.
/// Peers publish at least every [`HEARTBEAT`] even when idle, so this is several
/// missed heartbeats rather than a tight race.
pub const PEER_TIMEOUT: f64 = 6.0;

/// How often a peer publishes even when nothing changed (seconds).
pub const HEARTBEAT: f64 = 2.0;

/// How long a live gesture survives without an update before it is dropped
/// (seconds). Shorter than [`PEER_TIMEOUT`]: a peer that crashes mid-stroke should
/// stop smearing paint well before it leaves the roster. Strictly *longer* than
/// [`HEARTBEAT`]: the expiry clock advances at least per heartbeat but no faster
/// than the pump bothers to tick it, so a timeout inside that window sat on a
/// knife edge — equal to it, every live stroke on an idle receiver died at the
/// first heartbeat boundary.
pub const GESTURE_TIMEOUT: f64 = HEARTBEAT + 1.0;

/// How often the sender re-sends a gesture's invariant head and its whole path
/// (seconds), repairing any receiver that missed a delta and priming any client
/// that arrived mid-stroke (§17.5).
pub const GESTURE_RESYNC: Option<f64> = None;

/// Who this client is on the wire.
///
/// `actor` is meant to be **durable**: derived from a key the frontend persists, so
/// the same person reloading is the same author. That matters beyond presence —
/// undo targets *your* actions, and `DocState` keeps a selection per actor forever
/// because replay needs it, so minting a fresh id per session orphaned your own
/// history and left a dead entry in the log for every session anyone ever opened.
///
/// `boot` distinguishes *runs* of that identity, and is what makes durability safe:
/// [`PeerFrame::seq`] restarts at zero every process, so a peer that reloads within
/// [`PEER_TIMEOUT`] would otherwise have every frame rejected as stale until it
/// out-numbered its previous run. It need only increase, not mean anything.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Identity {
    pub actor: ActorId,
    pub boot: u64,
}

impl Identity {
    pub fn new(actor: ActorId, boot: u64) -> Self {
        Self { actor, boot }
    }
}

impl From<ActorId> for Identity {
    /// An identity with no run counter — correct for a client whose key is freshly
    /// generated, since a brand-new [`ActorId`] has no previous run to be confused
    /// with, and for tests.
    fn from(actor: ActorId) -> Self {
        Self { actor, boot: 0 }
    }
}

/// What a peer is doing right now — the preview of the action it will become.
///
/// [`Stroke`](Self::Stroke) carries a [`StrokeRecord`]: literally the type the
/// commit will carry, rendered through the same entry point. There is no second
/// stroke representation to keep in step, and therefore no second way for live and
/// committed pixels to disagree.
#[derive(Clone, Debug)]
pub enum LiveGesture {
    Stroke(StrokeRecord),
    Selection(SelectionOp),
    /// A region being dragged out to fill (§18.0.4). Carries its
    /// layer, because unlike a [`StrokeRecord`] a [`FillOp`] is the *region*, not
    /// the whole edit — the layer is the author's active one at the time.
    Fill {
        layer: LayerId,
        op: FillOp,
    },
}

/// One gesture in flight, from whoever is making it — the shape the preview fold
/// consumes, and the only place the local client and a peer are interchangeable.
///
/// They are deliberately *not* stored alike. The local gesture is derived from the
/// session's live [`PathFitter`](crate::path::PathFitter), which stays the single
/// source of truth for the one thing `preview == committed` rests on; a peer's is
/// reassembled from frames. Copying the local one into the roster to make them
/// uniform would make two sources of truth for exactly that. So they meet here
/// instead, at the point of use — which is all the fold ever needed, and gives it one
/// shape to reason about with every field present, rather than a tuple whose ordinal
/// had to be defaulted.
#[derive(Clone, Debug)]
pub struct GestureView {
    pub actor: ActorId,
    pub gesture: LiveGesture,
    /// Per-actor ordinal: what tells a cached render of *this* gesture apart from a
    /// render of the one before it.
    pub ordinal: u64,
    /// How many leading spans are settled, so an incremental repaint can retire them
    /// (§6.2, §17.6).
    pub frozen_spans: usize,
}

/// The invariant part of a live stroke: everything but the path. Sent on the
/// gesture's first frame and repeated on every resync frame, so a client that joined
/// mid-stroke or missed a delta can start rendering without asking for anything.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StrokeHead {
    pub layer: LayerId,
    pub brush: BrushParams,
    pub seed: u64,
}

/// One gesture update on the wire (§17.5).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum GestureFrame {
    /// A stroke in flight. The path grows by appending, because the fitter *freezes*
    /// a prefix of control points that is final and never revised (§6.2) —
    /// so `points` is everything frozen since the last frame plus the provisional
    /// knot under the cursor, and the receiver's reassembly
    /// (`truncate(from); extend(points)`) is exact rather than an approximation.
    Stroke {
        /// Per-actor ordinal, so a restart is unambiguous without a clock.
        id: u64,
        /// Present on the gesture's first frame and on every resync frame.
        head: Option<StrokeHead>,
        /// Index of the first control point in `points`; 0 on a resync frame.
        from: u32,
        points: Vec<ControlPoint>,
    },
    /// A marquee or lasso being dragged. Sent whole: it is already decimated
    /// (`LASSO_MIN_STEP`), and unlike a stroke path its tail is not append-only —
    /// the closing edge moves with the cursor.
    Selection { id: u64, op: SelectionOp },
    /// A region being dragged out to fill — sent whole for the same reason a
    /// selection is (§18.0.4). The layer is not in the payload:
    /// [`PeerFrame::active_layer`] already carries it, and a second copy could
    /// disagree with it.
    Fill { id: u64, op: FillOp },
}

impl GestureFrame {
    /// The gesture's per-actor ordinal.
    pub fn id(&self) -> u64 {
        match self {
            Self::Stroke { id, .. } | Self::Selection { id, .. } | Self::Fill { id, .. } => *id,
        }
    }
}

/// One published frame of a client's presence — the publishable half of a
/// [`Session`](crate::session::Session).
///
/// The author is **not** in the payload: [`Peers::merge`] takes it from the
/// transport's authenticated origin, the same discipline `Action` gets for free from
/// its [`ActionId`](crate::document::ActionId) (§17.7).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PeerFrame {
    /// Which run of this client published the frame ([`Identity::boot`]). Ordered
    /// *before* `seq`, which restarts at zero when a client does.
    #[serde(default)]
    pub boot: u64,
    /// Monotonic within a run. A frame that does not advance `(boot, seq)` is stale —
    /// a duplicate or an overtaken one — and is dropped.
    pub seq: u64,
    /// Sent on change and on resync frames; `None` means "unchanged".
    pub name: Option<String>,
    pub active_layer: LayerId,
    /// Hover position in canvas space; `None` when the pointer is off the canvas.
    pub cursor: Option<Vec2>,
    pub gesture: Option<GestureFrame>,
    /// This peer is leaving. Everything else in the frame is ignored.
    #[serde(default)]
    pub leaving: bool,
}

/// A participant, as everyone else sees them.
#[derive(Clone, Debug)]
pub struct Peer {
    pub actor: ActorId,
    pub name: String,
    /// Derived from `actor` by hash ([`peer_color`]), so every client agrees on it
    /// with no negotiation and no allocation protocol.
    pub color: [f32; 3],
    pub active_layer: LayerId,
    pub cursor: Option<Vec2>,
    /// The run of this peer we are listening to; see [`Identity::boot`].
    boot: u64,
    seq: u64,
    /// The receiving half of the gesture protocol. Its counterpart,
    /// [`GestureTx`](crate::presence::GestureTx), lives beside it in
    /// [`crate::presence`] rather than in the sender's module — they are one protocol
    /// and their invariants are properties of the pair.
    rx: GestureRx,
    last_seen: f64,
    gesture_seen: f64,
}

impl Peer {
    fn new(actor: ActorId, active_layer: LayerId, now: f64) -> Self {
        Self {
            actor,
            name: default_name(actor),
            color: peer_color(actor),
            active_layer,
            cursor: None,
            boot: 0,
            seq: 0,
            rx: GestureRx::default(),
            last_seen: now,
            gesture_seen: now,
        }
    }

    /// What to draw for this peer right now, if anything.
    pub fn gesture(&self) -> Option<&LiveGesture> {
        self.rx.drawn()
    }

    /// This peer's gesture as the preview fold wants it — the same shape
    /// [`Session::gesture_view`](crate::session::Session::gesture_view) produces for
    /// the local client.
    pub fn gesture_view(&self) -> Option<GestureView> {
        Some(GestureView {
            actor: self.actor,
            gesture: self.rx.drawn()?.clone(),
            // Set and cleared with `drawn`, so this is never the fallback the tuple
            // form needed — where `unwrap_or(0)` quietly gave every peer's selection
            // the same ordinal.
            ordinal: self.rx.id()?,
            frozen_spans: self.rx.frozen_spans(),
        })
    }

    /// Whether this peer has a gesture on the canvas.
    pub fn is_gesturing(&self) -> bool {
        self.rx.drawn().is_some()
    }

    /// The live stroke this peer is drawing, if any — what the preview fold renders.
    pub fn live_stroke(&self) -> Option<&StrokeRecord> {
        match self.rx.drawn() {
            Some(LiveGesture::Stroke(rec)) => Some(rec),
            _ => None,
        }
    }

    /// How many spans of [`live_stroke`](Self::live_stroke) are settled, so the
    /// preview can repaint only the tail (§6.2, §17.6).
    pub fn live_frozen_spans(&self) -> usize {
        self.rx.frozen_spans()
    }

    /// How many leading control points of [`live_stroke`](Self::live_stroke) the
    /// author has declared final. The protocol's core promise is about exactly these:
    /// a frozen control point never moves again, so they must equal the author's.
    #[cfg(test)]
    pub(crate) fn live_frozen_points(&self) -> usize {
        self.rx.frozen()
    }

    /// Integrate a frame. Returns whether the **canvas** changed, which is a much
    /// narrower question than whether the frame changed anything: a cursor is chrome
    /// and a selected layer is a fact about the peer, so neither belongs to the
    /// composited picture. Only the live gesture does (§17.6).
    fn apply(&mut self, frame: PeerFrame, now: f64) -> bool {
        self.seq = frame.seq;
        self.last_seen = now;
        if let Some(name) = frame.name {
            self.name = name;
        }
        self.active_layer = frame.active_layer;
        self.cursor = frame.cursor;
        match frame.gesture {
            None => self.rx.clear(),
            Some(gesture) => {
                self.gesture_seen = now;
                // `active_layer` is set just above, so a fill frame is paired with
                // the layer from the very frame that carried it rather than from
                // whichever one happened to arrive last.
                self.rx.apply(gesture, frame.seq, self.active_layer)
            }
        }
    }

    fn end_gesture(&mut self) {
        self.rx.clear();
    }
}

/// What integrating presence actually moved.
///
/// Presence arrives at pointer rate, but very little of it reaches the *canvas*: a
/// cursor is DOM chrome drawn over the artwork, and a peer's selected layer is a fact
/// about them. Only a live gesture makes the composited picture stale — plus a peer
/// arriving or leaving, which changes the set of actors whose selection outline may be
/// drawn.
///
/// Reporting one undifferentiated "changed" made every remote cursor move cost a
/// rebuild of the live fold — a `DocState` clone and a re-render of every stroke in
/// flight — thirty times a second per peer, and worst of all *while* someone was
/// painting, which is exactly when the frame budget is already spoken for.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct PresenceChange {
    /// Anything at all moved, so a cached projection of the roster is stale.
    pub roster: bool,
    /// The composited picture is stale and the live fold must be rebuilt. Implies
    /// [`roster`](Self::roster).
    pub canvas: bool,
}

impl PresenceChange {
    /// Nothing moved.
    pub const NONE: Self = Self {
        roster: false,
        canvas: false,
    };
    /// The canvas moved — which is also a roster change.
    pub const CANVAS: Self = Self {
        roster: true,
        canvas: true,
    };

    /// A roster change that reached the canvas only if `canvas`.
    fn roster(canvas: bool) -> Self {
        Self {
            roster: true,
            canvas,
        }
    }
}

/// Everyone else in the session (§17.4).
///
/// The **local** client is deliberately not in here. It would make the fold read
/// more uniformly, but the local live gesture is *derived* from
/// [`Session`](crate::session::Session)'s in-flight fitter, and copying it into the
/// roster would make two sources of truth for the one thing the
/// `preview == committed` invariant depends on. The engine merges the two at the
/// point of use instead (`Engine::live_strokes`), which keeps the ordering uniform
/// without duplicating the state.
#[derive(Default)]
pub struct Peers {
    map: BTreeMap<ActorId, Peer>,
    /// Bumped on every change. Lets a caller notice that the roster moved without
    /// rebuilding and comparing a projection of it — which, on a pump that wakes
    /// thirty times a second, is the difference between an allocation per tick and
    /// none.
    revision: u64,
}

impl Peers {
    pub fn new() -> Self {
        Self::default()
    }

    /// A counter that changes whenever anything in the roster does.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Integrate a frame published by `actor`, reporting what it moved.
    ///
    /// `actor` comes from the transport's authenticated origin, never from the
    /// frame: a peer can publish its own presence and nobody else's, which is the
    /// same guarantee `Action` gets from its id (§17.7).
    pub fn merge(&mut self, actor: ActorId, frame: PeerFrame, now: f64) -> PresenceChange {
        let change = self.merge_inner(actor, frame, now);
        self.revision += u64::from(change.roster);
        change
    }

    fn merge_inner(&mut self, actor: ActorId, frame: PeerFrame, now: f64) -> PresenceChange {
        if frame.leaving {
            // A departure takes the peer's gesture — and its selection outline — off
            // the canvas with it.
            return match self.map.remove(&actor) {
                Some(_) => PresenceChange::CANVAS,
                None => PresenceChange::NONE,
            };
        }
        match self.map.get_mut(&actor) {
            // Stale or duplicate: presence is a snapshot, so an overtaken frame
            // carries nothing a newer one has not already said. Ordered on
            // `(boot, seq)`, because `seq` alone cannot tell a client that restarted
            // from one whose frames arrived late.
            Some(peer) if (frame.boot, frame.seq) <= (peer.boot, peer.seq) => PresenceChange::NONE,
            Some(peer) if frame.boot > peer.boot => {
                // A new run of a client we already know — it reloaded. Its `seq` and
                // its gesture ordinals both restart at zero, so nothing from the
                // previous run may be carried over or compared against.
                peer.rx.clear();
                peer.boot = frame.boot;
                peer.apply(frame, now);
                PresenceChange::CANVAS
            }
            Some(peer) => PresenceChange::roster(peer.apply(frame, now)),
            None => {
                // An arrival changes which actors' selections may be outlined, so the
                // fold is stale whatever else the frame carries.
                let mut peer = Peer::new(actor, frame.active_layer, now);
                peer.boot = frame.boot;
                peer.apply(frame, now);
                self.map.insert(actor, peer);
                PresenceChange::CANVAS
            }
        }
    }

    /// Drop peers that have gone quiet and live gestures that have stalled. Called
    /// on the frontend's publish cadence, which is the only clock `stark-core` has —
    /// the engine deliberately owns none, so it runs on wasm and native alike.
    ///
    /// Reports what it moved, so the caller knows whether to redraw. A stalled
    /// *gesture* changes what is on the canvas without changing the roster's size, so
    /// counting peers is not enough to notice it. Everything expiry does reaches the
    /// canvas: a departure takes the peer's selection outline with it.
    pub fn tick(&mut self, now: f64) -> PresenceChange {
        let mut change = PresenceChange::NONE;
        self.map.retain(|_, p| {
            let live = now - p.last_seen < PEER_TIMEOUT;
            if !live {
                change = PresenceChange::CANVAS;
            }
            live
        });
        for peer in self.map.values_mut() {
            if peer.is_gesturing() && now - peer.gesture_seen >= GESTURE_TIMEOUT {
                peer.end_gesture();
                change = PresenceChange::CANVAS;
            }
        }
        self.revision += u64::from(change.roster);
        change
    }

    /// Whether [`tick`](Self::tick) would change anything — the cheap test, so a
    /// pump can skip the work on an idle session rather than take a mutable borrow
    /// to discover there was none.
    pub fn expiry_due(&self, now: f64) -> bool {
        self.map.values().any(|p| {
            now - p.last_seen >= PEER_TIMEOUT
                || (p.is_gesturing() && now - p.gesture_seen >= GESTURE_TIMEOUT)
        })
    }

    /// A gesture is a thing that becomes an action, so an action from its author is
    /// the end-of-gesture signal — no id to correlate, and no window in which both
    /// the live copy and the committed one are drawn.
    pub fn clear_gesture(&mut self, actor: ActorId) {
        if let Some(peer) = self.map.get_mut(&actor)
            && peer.is_gesturing()
        {
            peer.end_gesture();
            self.revision += 1;
        }
    }

    /// Forget everyone — leaving a session, or loading a different document.
    pub fn clear(&mut self) {
        if !self.map.is_empty() {
            self.map.clear();
            self.revision += 1;
        }
    }

    /// Every peer, in ascending [`ActorId`] order. That order is what every client
    /// can compute alike, which is what makes the preview fold agree across clients
    /// (§17.6).
    pub fn iter(&self) -> impl Iterator<Item = &Peer> {
        self.map.values()
    }

    pub fn get(&self, actor: ActorId) -> Option<&Peer> {
        self.map.get(&actor)
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

/// A peer's display colour, derived from its id so every client agrees with no
/// negotiation and no allocation protocol.
///
/// The hue is a mixing hash of the id, which *decorrelates* ids that are numerically
/// close or share bytes (as endpoint-derived ones do — [`ActorId`] takes its bytes
/// verbatim from a public key). It does not *space* them: two peers can land on
/// neighbouring hues, and with no coordination there is no way to prevent that
/// without an allocation protocol, which is a worse trade than an occasional
/// similar pair. Saturation and value are fixed high enough to read as a
/// person-marker over paint of any value.
pub fn peer_color(actor: ActorId) -> [f32; 3] {
    // splitmix64 finalizer: cheap, and it decorrelates the low bits that
    // `actor_from_endpoint_id` happens to take from a public key.
    let mut z = actor.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let hue = (z >> 40) as f32 / 16777216.0 * 360.0;
    hsv_to_rgb(hue, 0.72, 0.95)
}

/// A peer's name until they publish one: short, stable, and derived from the id, so
/// two unnamed peers are still distinguishable.
pub fn default_name(actor: ActorId) -> String {
    format!("{:04x}", actor.0 as u16)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let c = v * s;
    let hp = (h / 60.0).rem_euclid(6.0);
    let x = c * (1.0 - (hp % 2.0 - 1.0).abs());
    let (r, g, b) = match hp as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    [r + m, g + m, b + m]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{SelectionMode, SelectionShape};
    use crate::path::ControlPoint;

    fn frame(seq: u64, gesture: Option<GestureFrame>) -> PeerFrame {
        PeerFrame {
            boot: 0,
            seq,
            name: None,
            active_layer: LayerId(0),
            cursor: None,
            gesture,
            leaving: false,
        }
    }

    fn head() -> StrokeHead {
        StrokeHead {
            layer: LayerId(0),
            brush: BrushParams::default(),
            seed: 7,
        }
    }

    fn pts(n: usize) -> Vec<ControlPoint> {
        (0..n)
            .map(|i| ControlPoint::at(Vec2::new(i as f32, 0.0)))
            .collect()
    }

    #[test]
    fn a_stale_frame_is_dropped() {
        let mut peers = Peers::new();
        let a = ActorId(1);
        assert!(peers.merge(a, frame(5, None), 0.0).roster);
        assert!(
            !peers.merge(a, frame(4, None), 1.0).roster,
            "seq went backwards"
        );
        assert!(peers.merge(a, frame(6, None), 2.0).roster);
    }

    /// A client that reloads keeps its `ActorId` — that is what persisting the key
    /// buys — but restarts `seq` at zero. Ordered on `seq` alone, every frame of the
    /// new run is an "overtaken duplicate" until it out-numbers the old one, which
    /// for a long session is minutes of silence from someone who is right there.
    #[test]
    fn a_reloaded_peer_is_not_mistaken_for_a_stale_one() {
        let mut peers = Peers::new();
        let a = ActorId(1);
        let stroke = |id, n| {
            Some(GestureFrame::Stroke {
                id,
                head: Some(head()),
                from: 0,
                points: pts(n),
            })
        };

        let mut old = frame(50, stroke(7, 4));
        old.boot = 3;
        assert!(peers.merge(a, old, 0.0).roster);
        assert_eq!(
            peers
                .get(a)
                .expect("peer")
                .live_stroke()
                .expect("live")
                .path
                .len(),
            4
        );

        // The same person, reloaded: `seq` back to 1, and the gesture ordinals it
        // hands out start over too — so the run counter has to lead the comparison.
        let mut fresh = frame(1, stroke(7, 2));
        fresh.boot = 4;
        assert!(peers.merge(a, fresh, 0.1).canvas, "a new run must be heard");
        assert_eq!(
            peers
                .get(a)
                .expect("peer")
                .live_stroke()
                .expect("live")
                .path
                .len(),
            2,
            "and must not inherit the previous run's path, ordinal collision or not"
        );

        // A straggler from the old run is still stale, which is the property the
        // `seq` check existed for and must survive the change.
        let mut straggler = frame(51, None);
        straggler.boot = 3;
        assert!(!peers.merge(a, straggler, 0.2).roster);
    }

    /// A cursor is chrome drawn over the artwork, not part of it. Saying so is what
    /// keeps one peer's pointer from costing everyone a re-fold thirty times a second.
    #[test]
    fn a_cursor_move_is_not_a_canvas_change() {
        let mut peers = Peers::new();
        let a = ActorId(1);
        // The arrival itself is: it changes whose selection may be outlined.
        assert!(peers.merge(a, frame(1, None), 0.0).canvas, "arrival");

        let mut moved = frame(2, None);
        moved.cursor = Some(Vec2::new(4.0, 9.0));
        let change = peers.merge(a, moved, 0.1);
        assert!(change.roster, "the roster projection is stale");
        assert!(!change.canvas, "a cursor is not painted into the document");
        assert_eq!(
            peers.get(a).expect("peer").cursor,
            Some(Vec2::new(4.0, 9.0))
        );

        // A layer selection is likewise a fact about the peer, not about the picture.
        let mut relayered = frame(3, None);
        relayered.active_layer = LayerId(7);
        assert!(!peers.merge(a, relayered, 0.2).canvas);

        // But a gesture is.
        let stroking = frame(
            4,
            Some(GestureFrame::Stroke {
                id: 0,
                head: Some(head()),
                from: 0,
                points: pts(3),
            }),
        );
        assert!(peers.merge(a, stroking, 0.3).canvas, "a live stroke is");
        assert!(peers.merge(a, frame(5, None), 0.4).canvas, "so is its end");
    }

    /// The ordinal, not the stroke assembly, is what says "this is a different
    /// gesture" — a selection leaves no assembly, so keying on one let a peer's
    /// marquee outlive the gesture that drew it.
    #[test]
    fn a_stroke_delta_clears_a_stale_selection() {
        let mut peers = Peers::new();
        let a = ActorId(1);
        peers.merge(
            a,
            frame(
                1,
                Some(GestureFrame::Selection {
                    id: 0,
                    op: SelectionOp::new(
                        SelectionMode::Replace,
                        SelectionShape::rect_from_corners(Vec2::ZERO, Vec2::splat(10.0)),
                        0.0,
                    ),
                }),
            ),
            0.0,
        );
        assert!(matches!(
            peers.get(a).expect("peer").gesture(),
            Some(LiveGesture::Selection(_))
        ));

        // The frame clearing the marquee and the new stroke's head frame are both
        // lost; a headless delta for the *next* gesture is the first thing to arrive.
        let change = peers.merge(
            a,
            frame(
                4,
                Some(GestureFrame::Stroke {
                    id: 1,
                    head: None,
                    from: 3,
                    points: pts(1),
                }),
            ),
            0.1,
        );
        assert!(change.canvas, "taking the marquee down is a canvas change");
        assert!(
            peers.get(a).expect("peer").gesture().is_none(),
            "a gesture that ended must not be left on the canvas by the frames of the \
             one that replaced it"
        );
    }

    #[test]
    fn deltas_reassemble_into_the_growing_path() {
        let mut peers = Peers::new();
        let a = ActorId(1);
        peers.merge(
            a,
            frame(
                1,
                Some(GestureFrame::Stroke {
                    id: 0,
                    head: Some(head()),
                    from: 0,
                    points: pts(3),
                }),
            ),
            0.0,
        );
        peers.merge(
            a,
            frame(
                2,
                Some(GestureFrame::Stroke {
                    id: 0,
                    head: None,
                    // The provisional last knot is revised, the frozen two are not.
                    from: 2,
                    points: pts(4)[2..].to_vec(),
                }),
            ),
            0.1,
        );
        let peer = peers.get(a).expect("peer");
        let rec = peer.live_stroke().expect("live stroke");
        assert_eq!(rec.path.len(), 4);
        assert_eq!(rec.seed, 7);
    }

    #[test]
    fn a_gap_holds_the_prefix_and_a_resync_repairs_it() {
        let mut peers = Peers::new();
        let a = ActorId(1);
        peers.merge(
            a,
            frame(
                1,
                Some(GestureFrame::Stroke {
                    id: 0,
                    head: Some(head()),
                    from: 0,
                    points: pts(2),
                }),
            ),
            0.0,
        );
        // Frame 2 never arrived; frame 3 starts past what we hold.
        let change = peers.merge(
            a,
            frame(
                3,
                Some(GestureFrame::Stroke {
                    id: 0,
                    head: None,
                    from: 5,
                    points: pts(1),
                }),
            ),
            0.1,
        );
        assert!(
            !change.canvas,
            "nothing drawn changed, so nothing to repaint"
        );
        assert_eq!(
            peers
                .get(a)
                .expect("peer")
                .live_stroke()
                .expect("the prefix stays on the canvas")
                .path
                .len(),
            2,
            "the points already held are a true prefix of the stroke — short is \
             provisional, which a live preview always is; absent is a flicker"
        );
        peers.merge(
            a,
            frame(
                4,
                Some(GestureFrame::Stroke {
                    id: 0,
                    head: Some(head()),
                    from: 0,
                    points: pts(6),
                }),
            ),
            0.2,
        );
        assert_eq!(
            peers
                .get(a)
                .expect("peer")
                .live_stroke()
                .expect("repaired")
                .path
                .len(),
            6
        );
    }

    /// A lost frame that leaves **no hole** is the dangerous one.
    ///
    /// Deltas splice with `truncate(from); extend(points)`, keeping everything below
    /// `from` — which is only sound if the sender had frozen those points, and it
    /// announces that through the `from` of the frames in between. Lose one and a
    /// still-provisional control point is retained, then promoted to "frozen" by the
    /// next frame's `from`. The result has the right length and no gap, and is wrong
    /// in the middle. Only frame continuity catches it.
    #[test]
    fn a_lost_frame_that_leaves_no_hole_is_still_a_gap() {
        let mut peers = Peers::new();
        let a = ActorId(1);
        peers.merge(
            a,
            frame(
                1,
                Some(GestureFrame::Stroke {
                    id: 0,
                    head: Some(head()),
                    from: 0,
                    points: pts(3),
                }),
            ),
            0.0,
        );
        // Frame 2 (which would have revised control point 1 and 2) never arrived.
        // Frame 3 starts at 2 — inside the path, so nothing looks amiss.
        peers.merge(
            a,
            frame(
                3,
                Some(GestureFrame::Stroke {
                    id: 0,
                    head: None,
                    from: 2,
                    points: pts(2),
                }),
            ),
            0.1,
        );
        let path = &peers
            .get(a)
            .expect("peer")
            .live_stroke()
            .expect("the prefix stays up")
            .path;
        assert_eq!(
            path.len(),
            3,
            "a delta that is not the next frame must not be spliced, however \
             plausible its `from` looks"
        );
    }

    /// A resync frame restarts the *assembly*, not the gesture: the frozen
    /// watermark must survive it. Reset to zero, every resync discarded the
    /// renderer's cached head (`Engine::render_live_stroke` keys on it) and the
    /// whole stroke was redrawn from scratch — once a second per stroking peer,
    /// which is the multi-client slowdown.
    #[test]
    fn a_resync_does_not_walk_the_frozen_watermark_back() {
        let mut peers = Peers::new();
        let a = ActorId(1);
        peers.merge(
            a,
            frame(
                1,
                Some(GestureFrame::Stroke {
                    id: 0,
                    head: Some(head()),
                    from: 0,
                    points: pts(3),
                }),
            ),
            0.0,
        );
        // A delta whose `from` announces the sender froze two control points.
        peers.merge(
            a,
            frame(
                2,
                Some(GestureFrame::Stroke {
                    id: 0,
                    head: None,
                    from: 2,
                    points: pts(4)[2..].to_vec(),
                }),
            ),
            0.1,
        );
        assert_eq!(peers.get(a).expect("peer").live_frozen_points(), 2);

        // The resync: same gesture, head present, whole path from 0.
        peers.merge(
            a,
            frame(
                3,
                Some(GestureFrame::Stroke {
                    id: 0,
                    head: Some(head()),
                    from: 0,
                    points: pts(5),
                }),
            ),
            1.2,
        );
        let peer = peers.get(a).expect("peer");
        assert_eq!(peer.live_stroke().expect("live").path.len(), 5);
        assert_eq!(
            peer.live_frozen_points(),
            2,
            "a resync says nothing new about what is frozen, so it must not walk \
             the watermark back"
        );

        // A new gesture must not inherit it.
        peers.merge(
            a,
            frame(
                4,
                Some(GestureFrame::Stroke {
                    id: 1,
                    head: Some(head()),
                    from: 0,
                    points: pts(2),
                }),
            ),
            1.3,
        );
        assert_eq!(peers.get(a).expect("peer").live_frozen_points(), 0);
    }

    #[test]
    fn a_new_gesture_ordinal_starts_over() {
        let mut peers = Peers::new();
        let a = ActorId(1);
        peers.merge(
            a,
            frame(
                1,
                Some(GestureFrame::Stroke {
                    id: 0,
                    head: Some(head()),
                    from: 0,
                    points: pts(5),
                }),
            ),
            0.0,
        );
        peers.merge(
            a,
            frame(
                2,
                Some(GestureFrame::Stroke {
                    id: 1,
                    head: Some(head()),
                    from: 0,
                    points: pts(2),
                }),
            ),
            0.1,
        );
        assert_eq!(
            peers
                .get(a)
                .expect("peer")
                .live_stroke()
                .expect("stroke")
                .path
                .len(),
            2,
            "the second gesture must not inherit the first's path"
        );
    }

    #[test]
    fn silence_expires_the_gesture_then_the_peer() {
        let mut peers = Peers::new();
        let a = ActorId(1);
        peers.merge(
            a,
            frame(
                1,
                Some(GestureFrame::Stroke {
                    id: 0,
                    head: Some(head()),
                    from: 0,
                    points: pts(3),
                }),
            ),
            0.0,
        );
        peers.tick(GESTURE_TIMEOUT + 0.1);
        assert!(peers.get(a).expect("still present").gesture().is_none());
        peers.tick(PEER_TIMEOUT + 0.1);
        assert!(peers.is_empty());
    }

    #[test]
    fn leaving_removes_the_peer_at_once() {
        let mut peers = Peers::new();
        let a = ActorId(1);
        peers.merge(a, frame(1, None), 0.0);
        let mut bye = frame(2, None);
        bye.leaving = true;
        assert!(peers.merge(a, bye, 0.1).roster);
        assert!(peers.is_empty());
    }

    #[test]
    fn a_committed_action_clears_the_live_gesture() {
        let mut peers = Peers::new();
        let a = ActorId(1);
        peers.merge(
            a,
            frame(
                1,
                Some(GestureFrame::Stroke {
                    id: 0,
                    head: Some(head()),
                    from: 0,
                    points: pts(3),
                }),
            ),
            0.0,
        );
        peers.clear_gesture(a);
        assert!(peers.get(a).expect("peer").gesture().is_none());
    }

    #[test]
    fn peer_colors_are_stable_distinct_and_spread() {
        // Stable: every client derives the same colour for the same peer, which is
        // what replaces an allocation protocol.
        assert_eq!(peer_color(ActorId(7)), peer_color(ActorId(7)));

        // Distinct: no two of a realistic session's worth of ids collide outright.
        // Deliberately *not* asserting they are far apart — a hash decorrelates
        // hues, it does not space them, and pretending otherwise would be a test
        // asserting a property the code cannot have (see `peer_color`).
        let colors: Vec<[f32; 3]> = (0..32).map(|i| peer_color(ActorId(i))).collect();
        for (i, a) in colors.iter().enumerate() {
            for b in &colors[i + 1..] {
                assert_ne!(a, b);
            }
        }

        // Spread: over many ids the hues do cover the wheel rather than clustering,
        // which is the property that actually makes peers distinguishable in
        // aggregate. Every 60° sector should see some of 128 ids.
        let mut sectors = [0usize; 6];
        for i in 0..128 {
            let [r, g, b] = peer_color(ActorId(i * 2_654_435_761));
            let max = r.max(g).max(b);
            let min = r.min(g).min(b);
            let hue = if max - min <= f32::EPSILON {
                0.0
            } else if max == r {
                (60.0 * ((g - b) / (max - min))).rem_euclid(360.0)
            } else if max == g {
                60.0 * ((b - r) / (max - min)) + 120.0
            } else {
                60.0 * ((r - g) / (max - min)) + 240.0
            };
            sectors[((hue / 60.0) as usize).min(5)] += 1;
        }
        assert!(
            sectors.iter().all(|n| *n > 0),
            "hues clustered: {sectors:?}"
        );
    }
}
