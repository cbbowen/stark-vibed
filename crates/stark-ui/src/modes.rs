//! Which whole-canvas **composing mode** is live, and what it composes (§16.6, §20.5,
//! §22.2, §22.4): the values a chrome holds for the length of a mode, with no signal and
//! no toolkit in them. Entering a mode, leaving one and dropping its preview are each
//! frontend's.

use stark_model::Gradient;
use stark_model::document::{GradientAxis, GuideId, LayerId};
use stark_model::geom::Vec2;

use crate::transform::TransformUi;

/// The composing mode in flight, and what it is composing.
///
/// Four gestures take the canvas away from the brush for the length of a composition, and
/// **one value holds whichever is live**, so two modes at once is a state a chrome cannot
/// express rather than one every entry point has to remember to decline.
///
/// Named rather than a `bool` because leaving a mode has to know *what* to put down: each
/// holds a preview of a different kind, and dropping the wrong one leaves the canvas
/// showing a composition nothing is composing. It carries the payload rather than pointing
/// at where one is kept, so a mode live with its gesture missing is unspellable.
#[derive(Clone, PartialEq, Debug)]
pub enum Composing {
    /// The transform widget (§16.6).
    Transform(TransformUi),
    /// A perspective guide being shaped (§20.5).
    GuideEdit(GuideEdit),
    /// The gradient library's trace, armed from its pop-out (§22.2).
    ///
    /// The one mode with no payload: a trace's path is held by its own overlay, since an
    /// abandoned trace leaves nothing and a fresh arm should start clean either way.
    GradientTrace,
    /// The gradient fill's axis, on the shared gradient bar (§22.4).
    GradientFill(GradientUi),
}

impl Composing {
    /// The transform in hand, if that is the mode.
    ///
    /// An extractor per payload rather than a `match` at each call site, so that a site
    /// reads as its question: `composing.and_then(Composing::transform)`.
    pub fn transform(self) -> Option<TransformUi> {
        match self {
            Self::Transform(ui) => Some(ui),
            _ => None,
        }
    }

    /// The guide being shaped, if that is the mode (§20.5).
    pub fn guide_edit(self) -> Option<GuideEdit> {
        match self {
            Self::GuideEdit(edit) => Some(edit),
            _ => None,
        }
    }

    /// The gradient axis being composed, if that is the mode (§22.4).
    pub fn gradient_fill(self) -> Option<GradientUi> {
        match self {
            Self::GradientFill(ui) => Some(ui),
            _ => None,
        }
    }

    /// Whether these are the same *mode*, whatever each is composing.
    ///
    /// What a drag sample asks before it replaces the payload: it may change what the live
    /// mode composes but never which mode is live, since that swap would skip dropping the
    /// old mode's preview.
    pub fn same_mode(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }
}

/// A drawing guide selected for composing (§20.5): which guide the mode edits, and the
/// per-axis locks constraining the canvas drag.
///
/// The locks live here rather than on the guide because they are gesture state — a
/// constraint on the hand for one sitting, not a fact about the guide worth saving.
/// Leaving the mode releases them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GuideEdit {
    /// Which guide is being shaped
    /// ([`ObservableState::guides`](stark_engine::ObservableState::guides)).
    ///
    /// An id rather than an index, because the roster is document state: a peer's edit or
    /// an undo can reorder it or take a row out from under the hand, and an index would
    /// silently address a *different* guide.
    pub id: GuideId,
    /// World axes held fixed under the orbit drag: one lock constrains the drag to turning
    /// about that axis, two pin the frame entirely
    /// ([`PerspectiveGuide::dragged`](stark_model::document::PerspectiveGuide::dragged)).
    pub locked: [bool; 3],
}

/// The gradient gesture composed on the shared gradient bar (§22.4): what the ramp lands
/// on, how the drag is read, and the drag itself.
///
/// The drag is kept as its two raw points and the axis **derived** per kind, so switching
/// kinds reinterprets the drag the hand already made instead of throwing it away.
#[derive(Clone, PartialEq, Debug)]
pub struct GradientUi {
    /// What "Done" commits. One bar and one catcher for both targets, because laying a
    /// ramp is the same interface wherever it lands.
    pub target: GradientTarget,
    /// How the drag becomes an axis — the bar's Linear/Radial chips.
    pub kind: GradientAxisKind,
    /// Canvas space: the anchor and the current end. `None` until the first drag, when
    /// there is no axis to preview.
    pub drag: Option<(Vec2, Vec2)>,
}

/// What the gradient bar's ramp lands on (§22.4).
#[derive(Clone, PartialEq, Debug)]
pub enum GradientTarget {
    /// Fill the selection on `layer`, the ramp read live from the library.
    ///
    /// Opaque paint through the mask, so how strongly it lands is the selection's
    /// business and there is nothing here to capture (§6.8).
    Fill { layer: LayerId },
    /// Repaint the matte `layer` (§15.4).
    ///
    /// The ramp rides the target, seeded from the matte's own paint, so re-composing an
    /// old gradient's axis does not swap its colors for the library's current row.
    /// `None` where there was nothing to seed from — an empty library — and then there
    /// is nothing to preview, and "Done" lays nothing, until a ramp arrives.
    Matte {
        layer: LayerId,
        gradient: Option<Gradient>,
    },
}

/// How a gradient drag is read as an axis (§22.4).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GradientAxisKind {
    /// From the press to the release.
    Linear,
    /// Centred on the press, reaching to the release.
    Radial,
}

impl GradientUi {
    /// The axis the drag composes, or `None` before the first drag.
    pub fn axis(&self) -> Option<GradientAxis> {
        let (from, to) = self.drag?;
        Some(match self.kind {
            GradientAxisKind::Linear => GradientAxis::Linear { from, to },
            GradientAxisKind::Radial => GradientAxis::Radial {
                center: from,
                radius: from.distance(to),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stark_model::document::{ActionId, ActorId};

    fn fill(kind: GradientAxisKind, drag: Option<(Vec2, Vec2)>) -> GradientUi {
        GradientUi {
            target: GradientTarget::Fill {
                layer: LayerId::ROOT,
            },
            kind,
            drag,
        }
    }

    /// A radial drag is a centre and a reach: the press, and how far the release is from
    /// it.
    #[test]
    fn a_radial_axis_reaches_as_far_as_the_drag() {
        let (from, to) = (Vec2::new(10.0, 20.0), Vec2::new(13.0, 24.0));
        assert_eq!(
            fill(GradientAxisKind::Radial, Some((from, to))).axis(),
            Some(GradientAxis::Radial {
                center: from,
                radius: from.distance(to),
            }),
        );
        assert_eq!(
            from.distance(to),
            5.0,
            "a 3-4-5 drag, so the reach is exact"
        );
    }

    /// The same two points read the other way: the drag is the ramp.
    #[test]
    fn a_linear_axis_is_the_drag_itself() {
        let (from, to) = (Vec2::new(10.0, 20.0), Vec2::new(13.0, 24.0));
        assert_eq!(
            fill(GradientAxisKind::Linear, Some((from, to))).axis(),
            Some(GradientAxis::Linear { from, to }),
        );
    }

    /// Before the first drag there is nothing to preview, whichever kind is chosen.
    #[test]
    fn there_is_no_axis_before_the_first_drag() {
        for kind in [GradientAxisKind::Linear, GradientAxisKind::Radial] {
            assert_eq!(fill(kind, None).axis(), None, "{kind:?}");
        }
    }

    fn a_guide() -> Composing {
        Composing::GuideEdit(GuideEdit {
            id: GuideId(ActionId {
                lamport: 1,
                actor: ActorId(1),
            }),
            locked: [false; 3],
        })
    }

    /// An extractor answers for **its own** mode and for no other, which is the whole of
    /// what the call sites lean on: `and_then(Composing::guide_edit)` has to be `None`
    /// while a gradient axis is being composed, or a guide bar would mount over a
    /// gradient's catcher.
    #[test]
    fn an_extractor_answers_only_for_its_own_mode() {
        assert!(a_guide().guide_edit().is_some());
        assert!(a_guide().transform().is_none());
        assert!(a_guide().gradient_fill().is_none());

        assert!(Composing::GradientTrace.guide_edit().is_none());
        assert!(Composing::GradientTrace.transform().is_none());
        assert!(Composing::GradientTrace.gradient_fill().is_none());
    }

    /// A drag sample's gate: the same mode with a different payload passes, a different
    /// mode does not — which is what keeps replacing a payload from being a second entry
    /// that swaps the mode without dropping the preview the old one was showing.
    #[test]
    fn same_mode_ignores_the_payload_and_nothing_else() {
        let mut moved = match a_guide() {
            Composing::GuideEdit(mut e) => {
                e.locked[1] = true;
                Composing::GuideEdit(e)
            }
            _ => unreachable!(),
        };
        assert!(
            a_guide().same_mode(&moved),
            "a lock is not a different mode"
        );
        assert_ne!(a_guide(), moved, "and it is a different value");

        moved = Composing::GradientTrace;
        assert!(!a_guide().same_mode(&moved));
        assert!(!moved.same_mode(&a_guide()));
    }
}
