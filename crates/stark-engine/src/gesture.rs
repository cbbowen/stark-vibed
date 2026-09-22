//! **One vocabulary for a gesture in flight** (§17.4, §17.6).
//!
//! What this client is dragging out and what a peer is dragging out are the same
//! three things, so they are named once, here — by the session that produces one
//! ([`Session::live`](crate::session::Session::live)), by the protocol that sends and
//! reassembles it ([`crate::presence`]), by the roster that holds a peer's
//! ([`crate::peer`]) and by the fold that draws either ([`GestureView`]).
//!
//! They are deliberately not *stored* alike — the local client is not in the roster
//! ([`Peers`](crate::peer::Peers)) — so the two meet at the point of use, in a
//! [`GestureView`], rather than in a field.

use stark_model::document::{ActorId, FillOp, LayerId, SelectionOp, StrokeRecord};

/// What somebody is doing right now — the preview of the action it will become.
///
/// [`Stroke`](Self::Stroke) carries a [`StrokeRecord`] — the very type the commit will
/// carry, rendered through the same entry point, so there is no second representation
/// for live and committed pixels to disagree over. The other two are the op their
/// release commits, for the same reason.
#[derive(Clone, Debug, PartialEq)]
pub enum LiveGesture {
    Stroke(StrokeRecord),
    Selection(SelectionOp),
    /// A region being dragged out to fill (§18.0.4). Carries its
    /// layer, because unlike a [`StrokeRecord`] a [`FillOp`] is the *region*, not
    /// the whole edit — the layer is the author's active one at the time.
    Fill {
        layer: LayerId,
        op: FillOp,
        /// The layer's translation the op is in (§14.12) — a record carries its own,
        /// and a fill op is not a record.
        translation: stark_model::geom::IVec2,
    },
}

/// One gesture in flight, from whoever is making it — the shape the preview fold
/// consumes, and the only place the local client and a peer are interchangeable.
#[derive(Clone, Debug)]
pub struct GestureView {
    pub actor: ActorId,
    pub gesture: LiveGesture,
    /// Per-actor ordinal: what tells a cached render of *this* gesture apart from a
    /// render of the one before it.
    pub ordinal: u64,
    /// How many leading spans are settled, so an incremental repaint can retire them
    /// (§6.2, §17.6). Derived from the author's frozen control-point count through
    /// `path::frozen_spans_for` at both ends — the session's and a peer's — so the
    /// wire's partition and the fold's are one number.
    pub frozen_spans: usize,
}
