//! The drawing guides' vocabulary (§20): how a perspective quad's three planes are
//! named, and which rung of the cell ladder a lattice stands on.
//!
//! What a guide *is* is `stark_model`'s and reaches a shader without passing through
//! here (§20.4). What is here is the half a panel needs and the model does not state:
//! that the pair planes are read in the model's own cyclic order, that the cell scale
//! is offered in halvings rather than continuously, and the round trip between a
//! lattice and the rung it sits on.
//!
//! Both frontends held all of it. The native copy's doc said it was "the web app's own
//! figures (`panels::guides::CELL_OCTAVES`), because a rung has to be the same grid in
//! both apps" — a comment where a `use` should have been, and a rung that stopped
//! being the same grid the moment either side was edited.
//!
//! **The canvas gesture is not here.** Shaping a guide by dragging on the painting is
//! a composing mode with a full-viewport catcher, which one frontend has and the other
//! defers (§11.2), so its grab radii and its focal range stay with it. When the native
//! gesture lands, `FOCAL_RANGE` is the next thing that belongs in this module.

use stark_model::document::PerspectiveGuide;

use crate::icons::Icon;

/// The cell scale a control offers, as **halvings** of the default lattice (§20.3):
/// two steps coarser to two steps finer, and nothing in between.
///
/// The model draws a valid grid at any scale; this is what the control offers, and the
/// reason is that a grid meant to be counted on should *refine* rather than slide.
/// Double the cells and every line of the coarser grid is still a line of the finer one
/// with a new line between each pair — nothing an artist has already counted against
/// moves. At any other ratio the whole family slides along its pencil toward the
/// corner's own edge, which reads as the grid drifting sideways rather than as a change
/// of scale.
pub const CELL_OCTAVES: (i32, i32) = (-2, 2);

/// The floor the opacity track offers. A guide at zero is a guide that is on and
/// invisible, which the eye already says better.
pub const MIN_OPACITY: f32 = 0.1;

/// The two axes of each pair plane, in the order a control shows them: XY, YZ, ZX.
///
/// The model's own cyclic order, pair `k` being spanned by axes `(k, k+1)`, and the
/// chips are read in it rather than sorted: the three then run X→Y→Z→X, so each chip
/// picks up where the last left off and every axis letter appears exactly twice.
pub const PAIR_AXES: [[usize; 2]; 3] = [[0, 1], [1, 2], [2, 0]];

/// What each axis is called. Positional, because the model's axes are.
pub const AXIS_NAMES: [&str; 3] = ["X", "Y", "Z"];

/// Which rung of the cell ladder `guide` stands on.
///
/// The grid's scale is the *length* of the lattice: how many cells lie between the eye
/// and its corner, which is all a camera with no world scale of its own can say about
/// the size of a cell (§20.3). The guide's own length is the rung, so there is no
/// separate number to keep in step.
///
/// A degenerate lattice answers rung zero rather than `-inf` or `NaN`: the two apps had
/// two answers here, and this is the careful one.
pub fn octave(guide: &PerspectiveGuide) -> f32 {
    let base = PerspectiveGuide::default().lattice.length();
    let here = guide.lattice.length();
    if base <= 0.0 || here <= 0.0 {
        return 0.0;
    }
    (here / base).log2().round()
}

/// `guide` moved to rung `value` of the cell ladder.
///
/// Stepped off the **default** lattice rather than off the guide's current one, so a
/// rung is the same grid however it was reached — which is the whole of what
/// [`CELL_OCTAVES`] promises, and would not hold if each step compounded the last.
pub fn with_octave(mut guide: PerspectiveGuide, value: f32) -> PerspectiveGuide {
    let base = PerspectiveGuide::default().lattice;
    guide.lattice = base * 2f32.powi(value.round() as i32);
    guide
}

/// What to call a guide that has never been named: its place in the roster.
///
/// Numbered by *position*, which shifts when a row above it goes. That is the honest
/// reading — an unnamed row is being described rather than named, and the description
/// of the second row is "the second one". Naming it is how you stop it moving.
pub fn unnamed_label(index: usize) -> String {
    format!("Perspective {}", index + 1)
}

/// A guides panel's two continuous knobs.
#[derive(Clone, Copy, PartialEq, Eq, Debug, strum::VariantArray, strum::EnumCount)]
pub enum Dial {
    /// How fine the grid is, in halvings of the default lattice.
    Cells,
    /// How strongly the whole overlay reads over the paint.
    Opacity,
}

impl Dial {
    /// Which seat this dial has, so a view holding one state per dial can index
    /// rather than search. Exhaustive, so a third dial does not compile until it
    /// says where it sits.
    pub fn index(self) -> usize {
        match self {
            Dial::Cells => 0,
            Dial::Opacity => 1,
        }
    }

    /// The mark the control wears.
    pub fn glyph(self) -> Icon {
        match self {
            // A fan of lines from a point, which is exactly what this number counts:
            // the guide's fans are its parametrization (§20.5).
            Dial::Cells => crate::icons::DENSITY,
            Dial::Opacity => crate::icons::OPACITY,
        }
    }

    /// The one word a caption gives it.
    pub fn label(self) -> &'static str {
        match self {
            Dial::Cells => "Cells",
            Dial::Opacity => "Opacity",
        }
    }

    /// What it says on hover.
    pub fn tip(self) -> &'static str {
        match self {
            Dial::Cells => {
                "Cells \u{2014} how fine the grid is; each step halves the cell, so every \
                 line of the coarser grid is still a line of this one"
            }
            Dial::Opacity => "Opacity \u{2014} how strongly the guide reads over the paint",
        }
    }

    /// The ends of its track.
    pub fn range(self) -> (f32, f32) {
        match self {
            Dial::Cells => (CELL_OCTAVES.0 as f32, CELL_OCTAVES.1 as f32),
            Dial::Opacity => (MIN_OPACITY, 1.0),
        }
    }

    /// Whole rungs for the ladder, a hundredth for a strength.
    pub fn step(self) -> f32 {
        match self {
            Dial::Cells => 1.0,
            Dial::Opacity => 0.01,
        }
    }

    /// Where the dial stands for `guide`.
    pub fn read(self, guide: &PerspectiveGuide) -> f32 {
        match self {
            Dial::Cells => octave(guide),
            Dial::Opacity => guide.opacity,
        }
    }

    /// The guide `value` asks for. Stated as a whole edit of the camera rather than as
    /// a field write, because that is the shape every act on a guide takes:
    /// `DocCommand::SetGuide` carries the camera entire (§20.5).
    pub fn write(self, guide: PerspectiveGuide, value: f32) -> PerspectiveGuide {
        match self {
            Dial::Cells => with_octave(guide, value),
            Dial::Opacity => {
                let mut guide = guide;
                guide.opacity = value.clamp(MIN_OPACITY, 1.0);
                guide
            }
        }
    }

    /// How a readout prints it: a signed rung, two places for a strength.
    pub fn readout(self, v: f32) -> String {
        match self {
            Dial::Cells => format!("{v:+.0}"),
            Dial::Opacity => format!("{v:.2}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use strum::VariantArray;

    /// The default guide is rung zero by construction — the ladder is defined off it,
    /// so anything else means `octave` and [`with_octave`] disagree about their origin.
    #[test]
    fn the_default_lattice_is_rung_zero() {
        assert_eq!(octave(&PerspectiveGuide::default()), 0.0);
    }

    /// The round trip is the claim [`CELL_OCTAVES`] rests on: a rung names one grid,
    /// so writing a rung and reading it back gives the rung, at every offered step.
    #[test]
    fn a_rung_names_one_grid_however_it_is_reached() {
        for rung in CELL_OCTAVES.0..=CELL_OCTAVES.1 {
            let g = with_octave(PerspectiveGuide::default(), rung as f32);
            assert_eq!(octave(&g), rung as f32, "rung {rung} did not come back");
        }
    }

    /// Stepping is off the default, not off what is in hand — so reaching a rung by
    /// two moves lands where reaching it by one does. Compounding would drift.
    #[test]
    fn stepping_twice_lands_where_stepping_once_does() {
        let once = with_octave(PerspectiveGuide::default(), 2.0);
        let twice = with_octave(with_octave(PerspectiveGuide::default(), -1.0), 2.0);
        assert_eq!(once.lattice, twice.lattice);
    }

    /// A degenerate lattice answers a rung rather than `-inf` or `NaN`. One frontend
    /// guarded this and the other did not; the guard is the one that travelled.
    #[test]
    fn a_collapsed_lattice_still_names_a_rung() {
        let mut g = PerspectiveGuide::default();
        // Zeroed without naming its type: the lattice is a `glam::Vec3`, and this
        // crate does not depend on glam directly.
        g.lattice *= 0.0;
        assert_eq!(octave(&g), 0.0);
    }

    /// Every axis appears exactly twice across the three planes, which is what makes
    /// the cyclic order readable as X→Y→Z→X rather than as an arbitrary list.
    #[test]
    fn the_pairs_run_once_around_the_axes() {
        let mut seen = [0; 3];
        for [a, b] in PAIR_AXES {
            seen[a] += 1;
            seen[b] += 1;
        }
        assert_eq!(seen, [2, 2, 2], "the pair planes do not close a cycle");
    }

    /// A track a hand cannot move back off is the failure this table rules out.
    #[test]
    fn every_track_is_one_a_hand_can_move() {
        for dial in Dial::VARIANTS {
            let (lo, hi) = dial.range();
            assert!(lo < hi, "{dial:?} has an empty track: {lo}..={hi}");
            assert!(dial.step() > 0.0, "{dial:?} steps by nothing");
            assert!(
                dial.step() <= hi - lo,
                "{dial:?} steps further than its own track"
            );
        }
    }

    /// The opacity floor holds on the way in, so a value from anywhere — a track, a
    /// stored record, a peer — cannot land a guide on "on and invisible".
    #[test]
    #[expect(
        clippy::float_cmp_const,
        reason = "the clamp lands on MIN_OPACITY itself, so the assertion is identity rather than proximity"
    )]
    fn the_opacity_floor_holds_whatever_is_written() {
        let g = Dial::Opacity.write(PerspectiveGuide::default(), 0.0);
        assert_eq!(g.opacity, MIN_OPACITY);
        let g = Dial::Opacity.write(PerspectiveGuide::default(), 5.0);
        assert_eq!(g.opacity, 1.0);
    }

    /// An unnamed row is described by where it is, counting from one.
    #[test]
    fn an_unnamed_guide_is_the_nth_one() {
        assert_eq!(unnamed_label(0), "Perspective 1");
        assert_eq!(unnamed_label(107), "Perspective 108");
    }

    /// The seat and the roster are one order. They are two statements, and a view
    /// indexes by the first into a list built from the second.
    #[test]
    fn every_dial_sits_in_the_seat_its_index_names() {
        for (i, dial) in Dial::VARIANTS.iter().enumerate() {
            assert_eq!(dial.index(), i, "{dial:?} names a seat it does not sit in");
        }
    }
}
