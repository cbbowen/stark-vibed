//! The presence **wire** (§17.4): the frames a client publishes about itself.
//!
//! Presence is per-client state every client reads and only its owner writes, held
//! outside the timeline because replay does not need it. What travels is here; what
//! a receiver *builds* from it — the roster, the gesture latch, the interpolated
//! view of a peer's live stroke — is `stark-engine`'s `peer`.
//!
//! The selection is the one piece of per-client state replay *does* need, and it is
//! not presence: it lives in the document keyed by
//! [`ActorId`](crate::document::ActorId) (§17.3).
//!
//! `stark-net` speaks exactly this module and no more (§2, §12.4).

use serde::{Deserialize, Serialize};

use crate::MAX_NAME;
use crate::document::action::clamp_frame;
use crate::document::{BrushParams, FillOp, LayerId, SelectionOp};
use crate::geom::{IVec2, Vec2};
use crate::path::ControlPoint;
use crate::sanitize::at_least_zero;

/// How long a peer may go unheard-from before it leaves the roster, in seconds.
/// Peers publish at least every [`HEARTBEAT`] even when idle, so this is several
/// missed heartbeats rather than a tight race.
pub const PEER_TIMEOUT: f64 = 6.0;

/// How often a peer publishes even when nothing changed (seconds).
pub const HEARTBEAT: f64 = 2.0;

/// How long a live gesture survives without an update before it is dropped
/// (seconds). Must stay shorter than [`PEER_TIMEOUT`], so a peer that crashes
/// mid-stroke stops smearing paint before it leaves the roster, and strictly *longer*
/// than [`HEARTBEAT`]: the expiry clock advances no faster than the pump ticks it, so
/// equal to it every live stroke on an idle receiver dies at the first heartbeat.
pub const GESTURE_TIMEOUT: f64 = HEARTBEAT + 1.0;

/// How often the sender re-sends a gesture's invariant head and its whole path
/// (seconds), repairing any receiver that missed a delta and priming any client
/// that arrived mid-stroke (§17.5) — or `None` to send no resync frames at all.
///
/// **Currently `None`, deliberately.** This is a cadence, not a switch on the
/// feature: both halves implement the repair in full. What is deferred is how often
/// it is worth paying for — a resync frame carries the whole path, so on a long
/// stroke it is the largest presence frame a session sends, and whether it earns its
/// place depends on a real transport's loss rate and latency.
pub const GESTURE_RESYNC: Option<f64> = None;

/// The invariant part of a live stroke: everything but the path. Sent on the
/// gesture's first frame and repeated on every resync frame, so a client that joined
/// mid-stroke or missed a delta can start rendering without asking for anything.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
pub struct StrokeHead {
    pub layer: LayerId,
    pub brush: BrushParams,
    pub seed: u64,
    /// The layer's translation at the press
    /// ([`StrokeRecord::translation`](crate::document::StrokeRecord::translation)):
    /// the path's coordinates are relative to it, so a receiver folding the live
    /// stroke needs the same offset the commit will. Zero from an older sender, under
    /// which translation and canvas coincide.
    #[serde(default)]
    pub translation: IVec2,
}

/// One gesture update on the wire (§17.5).
#[derive(Clone, Debug, Serialize, Deserialize, carbonite::Schema)]
pub enum GestureFrame {
    /// A stroke in flight. The path grows by appending, because the fitter *freezes*
    /// a prefix of control points that is final and never revised (§6.2), so `points`
    /// is everything frozen since the last frame plus the provisional knot under the
    /// cursor and the receiver's `truncate(from); extend(points)` is exact.
    Stroke {
        /// Per-actor ordinal, so a restart is unambiguous without a clock.
        id: u64,
        /// Present on the gesture's first frame and on every resync frame. Boxed for
        /// the enum's sake, not the wire's — serde sees straight through, so the
        /// bytes are the unboxed shape's.
        head: Option<Box<StrokeHead>>,
        /// Index of the first control point in `points`; 0 on a resync frame.
        from: u32,
        points: Vec<ControlPoint>,
        /// Where on the assembled curve the stroke begins
        /// ([`StrokeRecord::start`](crate::document::StrokeRecord::start)).
        ///
        /// On every frame rather than in the head: it is a curve *parameter*, so
        /// the same number names a different place as the path grows. It stops
        /// moving exactly when the sender freezes the spans behind it (§6.2).
        #[serde(default)]
        start: f32,
    },
    /// A marquee or lasso being dragged. Sent whole: it is already decimated
    /// (`LASSO_MIN_STEP`), and unlike a stroke path its tail is not append-only —
    /// the closing edge moves with the cursor.
    Selection { id: u64, op: SelectionOp },
    /// A region being dragged out to fill — sent whole for the same reason a
    /// selection is (§18.0.4). The layer is not in the payload:
    /// [`PeerFrame::active_layer`] already carries it, and a second copy could
    /// disagree with it.
    Fill {
        id: u64,
        op: FillOp,
        /// The layer's translation at the press — [`StrokeHead::translation`], for a fill.
        #[serde(default)]
        translation: IVec2,
    },
}

impl GestureFrame {
    /// The gesture's per-actor ordinal.
    pub fn id(&self) -> u64 {
        match self {
            Self::Stroke { id, .. } | Self::Selection { id, .. } | Self::Fill { id, .. } => *id,
        }
    }

    /// [`PeerFrame::sanitized`]'s gesture half — private, so a gesture is gated only
    /// because the frame carrying it was, and a receiver cannot hold one that came
    /// through no door.
    ///
    /// **Exhaustive, with no `_` arm**: a fourth shape of gesture stops this
    /// compiling until it says whether it carries a number, where a wildcard would
    /// answer "nothing to hold" on its behalf.
    fn sanitized(self) -> Self {
        match self {
            Self::Stroke {
                id,
                head,
                from,
                points,
                start,
            } => Self::Stroke {
                id,
                // A committed stroke's brush is held to exactly this by
                // `ActionKind::sanitized`. A live one is drawn by the same renderer
                // and never becomes an action on the way, so nothing else would.
                head: head.map(|head| {
                    Box::new(StrokeHead {
                        layer: head.layer,
                        brush: head.brush.sanitized(),
                        seed: head.seed,
                        translation: clamp_frame(head.translation),
                    })
                }),
                from,
                // Gated already: a `ControlPoint` is `#[serde(from)]` through
                // `ControlPoint::clamped`. It leaves `pos` alone, but `stroke_rect`
                // claims the whole layer on a non-finite point — so a walk here would
                // find nothing and would cost per point at pointer rate.
                points,
                // The marker's ceiling is the path's span count, which is the
                // flattening's to know. What a frame can state alone is that it is a
                // number, and not before the curve it marks a point on.
                start: at_least_zero(start, 0.0),
            },
            // Gated already, and structurally rather than by a call: the op cannot
            // be deserialized except through `SelectionOp::at` (`#[serde(from)]`),
            // so there is nothing left here for a second pass to find.
            Self::Selection { .. } => self,
            // The fill's op likewise, through `FillOp::with_paint`. Its frame is not
            // part of the op, so it is clamped here.
            Self::Fill {
                id,
                op,
                translation,
            } => Self::Fill {
                id,
                op,
                translation: clamp_frame(translation),
            },
        }
    }
}

/// One published frame of a client's presence — the publishable half of a
/// `stark-engine`'s `Session`.
///
/// The author is **not** in the payload: `stark-engine`'s `Peers::merge` takes it from
/// the transport's authenticated origin, as `Action` gets it from its
/// [`ActionId`](crate::document::ActionId) (§17.7).
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

impl PeerFrame {
    /// The same frame with every free-form payload finite and in range — **the one
    /// funnel a frame passes through on its way into a roster** (§21.5), and
    /// [`ActionKind::sanitized`](crate::document::ActionKind::sanitized)'s twin for
    /// the half of the wire that is not the log.
    ///
    /// A frame is never an action, so it meets that funnel nowhere else, and it
    /// carries several things no type of its own bounds: a name republished to
    /// everyone, a cursor, a stroke's `start`, and a brush whose radius sizes a
    /// dispatch and whose rates reach the dynamics loop. One call, at the door, rather
    /// than a gate per field to forget (§1).
    ///
    /// **Every field written out, no `..self`**: a field added to the frame later
    /// stops this compiling until it says whether it is a number.
    ///
    /// **Idempotent**, so the door may hold it without first establishing that no
    /// other gate already did.
    #[must_use]
    pub fn sanitized(self) -> Self {
        Self {
            boot: self.boot,
            seq: self.seq,
            name: self
                .name
                .map(|mut name| {
                    // In place and by `char`: the cut cannot land inside one, and a
                    // name that was already short keeps the buffer it arrived in.
                    if let Some((cut, _)) = name.char_indices().nth(MAX_NAME) {
                        name.truncate(cut);
                    }
                    name
                })
                // An empty name is *no* name, which is what `None` already means on
                // this field. `Peer::apply` overwrites the id-derived `default_name`
                // unconditionally, so a `Some("")` would blank that peer's row for the
                // rest of the session.
                .filter(|name| !name.is_empty()),
            active_layer: self.active_layer,
            // The sender filters this too, and says why (`Session::set_cursor`) —
            // but the sender is not the end facing the wire.
            cursor: self.cursor.filter(|p| p.is_finite()),
            gesture: self.gesture.map(GestureFrame::sanitized),
            leaving: self.leaving,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{BrushDynamics, BrushEffect, BrushParams, FRAME_LIMIT, LayerId};
    use crate::path::ControlPoint;

    /// A frame carrying every shape of bad number this funnel is for.
    fn hostile() -> PeerFrame {
        PeerFrame {
            boot: 1,
            seq: 1,
            name: Some("€".repeat(MAX_NAME * 2)),
            active_layer: LayerId::ROOT,
            cursor: Some(Vec2::new(f32::NAN, 3.0)),
            gesture: Some(GestureFrame::Stroke {
                id: 1,
                head: Some(Box::new(StrokeHead {
                    layer: LayerId::ROOT,
                    brush: BrushParams {
                        size: f32::NAN,
                        effect: BrushEffect::wet_with(
                            [0.0; 3],
                            BrushDynamics {
                                lift: f32::INFINITY,
                                ..Default::default()
                            },
                        ),
                        ..BrushParams::default()
                    },
                    seed: 0,
                    translation: IVec2::splat(i32::MAX),
                })),
                from: 0,
                points: vec![ControlPoint::at(Vec2::ZERO)],
                start: f32::NAN,
            }),
            leaving: false,
        }
    }

    /// The gate holds, stated beside the funnel rather than only in the crate
    /// downstream that calls it.
    #[test]
    fn a_frame_arrives_finite_and_in_range() {
        let f = hostile().sanitized();
        assert_eq!(
            f.name.as_deref().map(str::chars).map(Iterator::count),
            Some(MAX_NAME),
            "a name is cut to the bound, and cut by `char`",
        );
        assert_eq!(f.cursor, None, "a NaN cursor is not a place on the canvas");
        let Some(GestureFrame::Stroke { head, start, .. }) = f.gesture else {
            panic!("the gesture kept its shape");
        };
        assert_eq!(
            start, 0.0,
            "a marker is a place on the curve, not before it"
        );
        let head = head.expect("the head survives");
        assert!(head.brush.size.is_finite(), "a radius sizes a dispatch");
        assert_eq!(
            head.translation,
            IVec2::splat(FRAME_LIMIT),
            "a live stroke's frame is spent by the renderer a committed one is",
        );

        // The fill arm carries a frame too, and it is the same offset.
        let f = PeerFrame {
            gesture: Some(GestureFrame::Fill {
                id: 1,
                op: FillOp::of_selection(crate::Srgb::new([0.0; 3])),
                translation: IVec2::splat(i32::MIN),
            }),
            ..hostile()
        }
        .sanitized();
        let Some(GestureFrame::Fill { translation, .. }) = f.gesture else {
            panic!("the gesture kept its shape");
        };
        assert_eq!(translation, IVec2::splat(-FRAME_LIMIT));
    }

    /// [`PeerFrame::sanitized`] claims to be idempotent, which is what lets the door
    /// hold it without first establishing that nothing else already did.
    #[test]
    fn sanitizing_a_frame_twice_does_not_move_it_again() {
        let once = hostile().sanitized();
        let twice = once.clone().sanitized();
        assert_eq!(format!("{once:?}"), format!("{twice:?}"));
    }

    /// An empty name is *no* name: `None` is what this field already spells for
    /// "unchanged", and `Peer::apply` overwrites the id-derived default
    /// unconditionally — so a `Some("")` would blank that peer's row for the rest of
    /// the session with nothing left to restore it.
    #[test]
    fn an_empty_name_is_no_name() {
        let f = PeerFrame {
            name: Some(String::new()),
            ..hostile()
        }
        .sanitized();
        assert_eq!(f.name, None);
    }
}
