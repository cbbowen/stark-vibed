//! What a composing mode composes (§22.4): the values a chrome holds for the length of
//! a mode, with no signal and no toolkit in them.

use stark_model::Gradient;
use stark_model::document::{GradientAxis, LayerId};
use stark_model::geom::Vec2;

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
}
