//! The presence **wire** (§17.4): the frames a client publishes about itself.
//!
//! Presence is per-client state every client reads and only its owner writes, held
//! outside the timeline because replay does not need it. What travels is here; what
//! a receiver *builds* from it — the roster, the gesture latch, the interpolated
//! view of a peer's live stroke — is `stark-engine`'s `peer`, because it is state
//! rather than a message.
//!
//! The one piece of per-client state that *is* needed by replay — the selection —
//! is not presence at all: it lives in the document keyed by
//! [`ActorId`](crate::document::ActorId) instead (§17.3).
//!
//! `stark-net` speaks exactly this module and no more (§2, §12.4).

use serde::{Deserialize, Serialize};

use crate::document::{BrushParams, FillOp, LayerId, SelectionOp};
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
/// stop smearing paint well before it leaves the roster. Strictly *longer* than
/// [`HEARTBEAT`]: the expiry clock advances at least per heartbeat but no faster
/// than the pump bothers to tick it, so a timeout inside that window sat on a
/// knife edge — equal to it, every live stroke on an idle receiver died at the
/// first heartbeat boundary.
pub const GESTURE_TIMEOUT: f64 = HEARTBEAT + 1.0;

/// How often the sender re-sends a gesture's invariant head and its whole path
/// (seconds), repairing any receiver that missed a delta and priming any client
/// that arrived mid-stroke (§17.5) — or `None` to send no resync frames at all.
///
/// **Currently `None`, deliberately.** This is a cadence, not a switch on the
/// feature: the encoder takes `resync` as a parameter and both halves implement the
/// repair in full, which is why
/// `presence::tests::a_resync_repairs_a_receiver_that_missed_everything` exercises it
/// regardless of what this says. What is deferred is only *how often it is worth
/// paying for*, and that needs a measurement this crate cannot make — a resync frame
/// carries the whole path, so on a long stroke it is the largest presence frame a
/// session sends, and whether it earns its place depends on the loss rate and the
/// added latency on a real transport rather than on anything visible from here.
///
/// Setting it is a one-line change with no other edit anywhere: what a receiver does
/// with a resync frame is already settled, already tested, and already the reason
/// `GestureRx` carries its frozen watermark across one.
pub const GESTURE_RESYNC: Option<f64> = None;

/// The invariant part of a live stroke: everything but the path. Sent on the
/// gesture's first frame and repeated on every resync frame, so a client that joined
/// mid-stroke or missed a delta can start rendering without asking for anything.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
pub struct StrokeHead {
    pub layer: LayerId,
    pub brush: BrushParams,
    pub seed: u64,
}

/// One gesture update on the wire (§17.5).
#[derive(Clone, Debug, Serialize, Deserialize, carbonite::Schema)]
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
/// `stark-engine`'s `Session`.
///
/// The author is **not** in the payload: `stark-engine`'s `Peers::merge` takes it from the
/// transport's authenticated origin, the same discipline `Action` gets for free from
/// its [`ActionId`](crate::document::ActionId) (§17.7).
#[derive(Clone, Debug, Serialize, Deserialize, carbonite::Schema)]
pub struct PeerFrame {
    /// Which run of this client published the frame (`stark-engine`'s `Identity::boot`). Ordered
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
