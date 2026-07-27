//! Presence: per-client state that every client reads and only its owner writes
//! (PEER_DESIGN.md §4).
//!
//! This is the **unlogged** half of per-client state. The rule that puts something
//! here rather than in [`DocState`](crate::document::DocState) is the one DESIGN §4
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
//! see PEER_DESIGN.md §3. It is the one piece of per-client state that is not here.)
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

use crate::document::{ActorId, BrushParams, LayerId, SelectionOp, StrokeRecord, Tool};
use crate::geom::Vec2;
use crate::path::ControlPoint;

/// How long a peer may go unheard-from before it leaves the roster, in seconds.
/// Peers publish at least every [`HEARTBEAT`] even when idle, so this is several
/// missed heartbeats rather than a tight race.
pub const PEER_TIMEOUT: f64 = 6.0;

/// How often a peer publishes even when nothing changed (seconds).
pub const HEARTBEAT: f64 = 2.0;

/// How long a live gesture survives without an update before it is dropped
/// (seconds). Shorter than [`PEER_TIMEOUT`]: a peer that crashes mid-stroke should
/// stop smearing paint well before it leaves the roster.
pub const GESTURE_TIMEOUT: f64 = 2.0;

/// How often the sender re-sends a gesture's invariant head and its whole path
/// (seconds), repairing any receiver that missed a delta and priming any client
/// that arrived mid-stroke (PEER_DESIGN.md §5).
pub const GESTURE_RESYNC: f64 = 1.0;

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
}

/// The invariant part of a live stroke: everything but the path. Sent on the
/// gesture's first frame and repeated on every resync frame, so a client that joined
/// mid-stroke or missed a delta can start rendering without asking for anything.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StrokeHead {
    pub layer: LayerId,
    pub tool: Tool,
    pub brush: BrushParams,
    pub seed: u64,
}

/// One gesture update on the wire (PEER_DESIGN.md §5).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum GestureFrame {
    /// A stroke in flight. The path grows by appending, because the fitter *freezes*
    /// a prefix of control points that is final and never revised (DESIGN §6.2) —
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
}

impl GestureFrame {
    /// The gesture's per-actor ordinal.
    pub fn id(&self) -> u64 {
        match self {
            Self::Stroke { id, .. } | Self::Selection { id, .. } => *id,
        }
    }
}

/// One published frame of a client's presence — the publishable half of a
/// [`Session`](crate::session::Session).
///
/// The author is **not** in the payload: [`Peers::merge`] takes it from the
/// transport's authenticated origin, the same discipline `Action` gets for free from
/// its [`ActionId`](crate::document::ActionId) (PEER_DESIGN.md §7).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PeerFrame {
    /// Monotonic per actor. A frame that does not advance it is stale — a duplicate
    /// or an overtaken one — and is dropped.
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
    pub gesture: Option<LiveGesture>,
    seq: u64,
    /// The gesture currently being reassembled, and the path so far. Kept apart from
    /// `gesture` because a gesture with a gap in its path is *dropped from view*
    /// while still being tracked, awaiting the next resync frame.
    stroke: Option<StrokeAssembly>,
    last_seen: f64,
    gesture_seen: f64,
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

impl Peer {
    fn new(actor: ActorId, active_layer: LayerId, now: f64) -> Self {
        Self {
            actor,
            name: default_name(actor),
            color: peer_color(actor),
            active_layer,
            cursor: None,
            gesture: None,
            seq: 0,
            stroke: None,
            last_seen: now,
            gesture_seen: now,
        }
    }

    /// The live stroke this peer is drawing, if any — what the preview fold renders.
    pub fn live_stroke(&self) -> Option<&StrokeRecord> {
        match &self.gesture {
            Some(LiveGesture::Stroke(rec)) => Some(rec),
            _ => None,
        }
    }

    /// The marquee or lasso this peer is dragging, if any.
    pub fn live_selection(&self) -> Option<&SelectionOp> {
        match &self.gesture {
            Some(LiveGesture::Selection(op)) => Some(op),
            _ => None,
        }
    }

    /// The per-actor ordinal of the gesture in flight — what tells a cached render
    /// of this peer's stroke apart from a render of the one before it.
    pub fn gesture_id(&self) -> Option<u64> {
        self.stroke.as_ref().map(|s| s.id)
    }

    /// How many spans of [`live_stroke`](Self::live_stroke) are settled, so the
    /// preview can repaint only the tail (DESIGN.md §6.2, PEER_DESIGN.md §6).
    pub fn live_frozen_spans(&self) -> usize {
        self.stroke
            .as_ref()
            .map_or(0, |s| crate::path::frozen_spans_for(s.frozen, s.path.len()))
    }

    fn apply(&mut self, frame: PeerFrame, now: f64) {
        self.seq = frame.seq;
        self.last_seen = now;
        if let Some(name) = frame.name {
            self.name = name;
        }
        self.active_layer = frame.active_layer;
        self.cursor = frame.cursor;
        match frame.gesture {
            None => self.end_gesture(),
            Some(frame) => {
                self.gesture_seen = now;
                self.apply_gesture(frame);
            }
        }
    }

    fn apply_gesture(&mut self, frame: GestureFrame) {
        // A new ordinal is a different gesture: drop whatever was being assembled
        // rather than splicing two strokes together.
        if self.stroke.as_ref().is_some_and(|s| s.id != frame.id()) {
            self.stroke = None;
            self.gesture = None;
        }
        match frame {
            GestureFrame::Selection { op, .. } => {
                self.stroke = None;
                self.gesture = Some(LiveGesture::Selection(op));
            }
            GestureFrame::Stroke {
                id,
                head,
                from,
                points,
            } => {
                let from = from as usize;
                // Start (or restart, on a resync frame) whenever the head is present.
                if let Some(head) = head
                    && (self.stroke.as_ref().is_none_or(|s| s.id != id) || from == 0)
                {
                    self.stroke = Some(StrokeAssembly {
                        id,
                        head,
                        path: Vec::new(),
                        frozen: 0,
                    });
                }
                let Some(assembly) = self.stroke.as_mut() else {
                    // A delta for a gesture whose head we never saw. Nothing to do
                    // but wait for the next resync frame (PEER_DESIGN.md §5).
                    return;
                };
                if from > assembly.path.len() {
                    // A gap: frames were lost. Losing a live preview is cosmetic —
                    // the committed action always follows and is authoritative — so
                    // drop what is shown and let the resync frame repair it, rather
                    // than rendering a path with a hole in it.
                    self.gesture = None;
                    return;
                }
                assembly.path.truncate(from);
                assembly.path.extend(points);
                // A resync frame resends the whole path, which says nothing new about
                // what is frozen — so the watermark only ever advances.
                assembly.frozen = assembly.frozen.max(from);
                self.gesture = Some(LiveGesture::Stroke(StrokeRecord {
                    layer: assembly.head.layer,
                    tool: assembly.head.tool,
                    brush: assembly.head.brush,
                    path: assembly.path.clone(),
                    seed: assembly.head.seed,
                }));
            }
        }
    }

    fn end_gesture(&mut self) {
        self.gesture = None;
        self.stroke = None;
    }
}

/// Everyone else in the session (PEER_DESIGN.md §4).
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

    /// Integrate a frame published by `actor`. Returns whether anything changed.
    ///
    /// `actor` comes from the transport's authenticated origin, never from the
    /// frame: a peer can publish its own presence and nobody else's, which is the
    /// same guarantee `Action` gets from its id (PEER_DESIGN.md §7).
    pub fn merge(&mut self, actor: ActorId, frame: PeerFrame, now: f64) -> bool {
        let changed = self.merge_inner(actor, frame, now);
        self.revision += u64::from(changed);
        changed
    }

    fn merge_inner(&mut self, actor: ActorId, frame: PeerFrame, now: f64) -> bool {
        if frame.leaving {
            return self.map.remove(&actor).is_some();
        }
        match self.map.get_mut(&actor) {
            // Stale or duplicate: presence is a snapshot, so an overtaken frame
            // carries nothing a newer one has not already said.
            Some(peer) if frame.seq <= peer.seq => false,
            Some(peer) => {
                peer.apply(frame, now);
                true
            }
            None => {
                let mut peer = Peer::new(actor, frame.active_layer, now);
                peer.apply(frame, now);
                self.map.insert(actor, peer);
                true
            }
        }
    }

    /// Drop peers that have gone quiet and live gestures that have stalled. Called
    /// on the frontend's publish cadence, which is the only clock `stark-core` has —
    /// the engine deliberately owns none, so it runs on wasm and native alike.
    ///
    /// Returns whether anything changed, so the caller knows whether to redraw. A
    /// stalled *gesture* changes what is on the canvas without changing the roster's
    /// size, so counting peers is not enough to notice it.
    pub fn tick(&mut self, now: f64) -> bool {
        let before = self.map.len();
        self.map.retain(|_, p| now - p.last_seen < PEER_TIMEOUT);
        let mut changed = self.map.len() != before;
        for peer in self.map.values_mut() {
            if peer.gesture.is_some() && now - peer.gesture_seen >= GESTURE_TIMEOUT {
                peer.end_gesture();
                changed = true;
            }
        }
        self.revision += u64::from(changed);
        changed
    }

    /// Whether [`tick`](Self::tick) would change anything — the cheap test, so a
    /// pump can skip the work on an idle session rather than take a mutable borrow
    /// to discover there was none.
    pub fn expiry_due(&self, now: f64) -> bool {
        self.map.values().any(|p| {
            now - p.last_seen >= PEER_TIMEOUT
                || (p.gesture.is_some() && now - p.gesture_seen >= GESTURE_TIMEOUT)
        })
    }

    /// A gesture is a thing that becomes an action, so an action from its author is
    /// the end-of-gesture signal — no id to correlate, and no window in which both
    /// the live copy and the committed one are drawn.
    pub fn clear_gesture(&mut self, actor: ActorId) {
        if let Some(peer) = self.map.get_mut(&actor)
            && peer.gesture.is_some()
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
    /// (PEER_DESIGN.md §6).
    pub fn iter(&self) -> impl Iterator<Item = &Peer> {
        self.map.values()
    }

    pub fn get(&self, actor: ActorId) -> Option<&Peer> {
        self.map.get(&actor)
    }

    pub fn len(&self) -> usize {
        self.map.len()
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
    use crate::path::ControlPoint;

    fn frame(seq: u64, gesture: Option<GestureFrame>) -> PeerFrame {
        PeerFrame {
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
            tool: Tool::Brush,
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
        assert!(peers.merge(a, frame(5, None), 0.0));
        assert!(!peers.merge(a, frame(4, None), 1.0), "seq went backwards");
        assert!(peers.merge(a, frame(6, None), 2.0));
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
    fn a_gap_drops_the_preview_and_a_resync_repairs_it() {
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
        peers.merge(
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
            peers.get(a).expect("peer").live_stroke().is_none(),
            "a path with a hole in it is not shown"
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
        assert!(peers.get(a).expect("still present").gesture.is_none());
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
        assert!(peers.merge(a, bye, 0.1));
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
        assert!(peers.get(a).expect("peer").gesture.is_none());
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
