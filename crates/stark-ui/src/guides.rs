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
//! **The canvas gesture's arithmetic is here too** — what a press grabs ([`Handles`]),
//! how far the lens travels ([`FOCAL_RANGE`]) and where a dragged row lands
//! ([`anchor_at`]). The gesture itself is a composing mode with a full-viewport catcher,
//! which one frontend has and the other defers (§11.2), so the catcher stays with it.

use stark_model::document::{GuideId, Lens, PerspectiveGuide, PlaneTrace};
use stark_model::geom::Vec2;

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

/// What a roster row calls `guide`: its own name if it has been given one, and its
/// place if it has not ([`unnamed_label`]).
pub fn label(index: usize, guide: &stark_engine::GuideInfo) -> String {
    match &guide.name {
        Some(name) => name.to_string(),
        None => unnamed_label(index),
    }
}

/// How near a press must land to the center-of-view crosshair to grab it, screen px.
pub const CENTER_GRAB_PX: f32 = 14.0;

/// Half-width of the grab band around a drawn curve — a view-cone ring or a horizon — in
/// screen px, so a handle is equally grabbable at any magnification. One number for both,
/// because they are the same ask of the hand: put the pointer on a line about a pixel
/// wide.
pub const LINE_BAND_PX: f32 = 10.0;

/// The lens's travel, canvas px: wide enough for any drawing, floored so the circle cannot
/// be dragged through its own center into a degenerate camera.
pub const FOCAL_RANGE: (f32, f32) = (120.0, 12000.0);

/// What a press on a guide would grab (§20.5), nearest wins: the crosshair moves the
/// construction, a ring is the lens, a horizon is a turn about one axis, and everywhere
/// else is the world.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum GuideRegion {
    Center,
    /// A view-cone ring; the payload is that ring's radius **per unit focal length**
    /// ([`Lens::ring_factors`]), so a drag divides the hand's distance back into a focal
    /// length. Carried in the drag because the fisheye shows two rings, and the one
    /// grabbed must stay the one held — the 90° ring dragged inward must not hand off to
    /// the 45°.
    Focal(f32),
    /// A **horizon**: the vanishing line between two axes' vanishing points, which is the
    /// plane normal to the third — so the payload is that third axis, and a drag turns
    /// about it.
    Horizon(usize),
    Orbit,
}

/// A guide's grabbable geometry, read out once when its overlay draws: canvas-space and
/// `Copy`, so pointer handlers can share the hit test without holding the guide itself
/// (which carries a name).
#[derive(Copy, Clone, Debug)]
pub struct Handles {
    center: Vec2,
    focal: f32,
    lens: Lens,
    /// Horizon `n` is the one that turns about axis `n`, and it is `None` where the guide
    /// does not draw it — a line that is not on the screen cannot be grabbed
    /// ([`PerspectiveGuide::horizons`]).
    horizons: [Option<PlaneTrace>; 3],
}

impl Handles {
    /// The handles `g` draws.
    pub fn of(g: &PerspectiveGuide) -> Self {
        Self {
            center: g.center,
            focal: g.focal,
            lens: g.lens,
            horizons: g.horizons(),
        }
    }

    /// What a press at canvas point `p` grabs, with the view at `zoom`.
    ///
    /// The crosshair is topmost, as it is drawn; below it the rings and the horizons
    /// compete on **distance in screen px**, so a press between two curves takes the
    /// nearer and every handle is equally grabbable at any magnification. A tie goes to
    /// the ring: under the fisheye a pair trace can *be* a ring (in a 1-point pose the
    /// 90° ring is the X/Y horizon, §20.8), and the lens drag is the older gesture to
    /// leave in the hand where the two coincide.
    pub fn at(self, p: Vec2, zoom: f32) -> GuideRegion {
        if (p - self.center).length() * zoom < CENTER_GRAB_PX {
            return GuideRegion::Center;
        }
        let dist = (p - self.center).length();
        let (r45, r90) = self.lens.ring_factors();
        let rings = [Some(r45), r90]
            .into_iter()
            .flatten()
            .map(|factor| ((dist - self.focal * factor).abs() * zoom, factor));
        let horizons = self
            .horizons
            .into_iter()
            .enumerate()
            .filter_map(|(n, trace)| Some((trace?.distance(p) * zoom, n)));
        let ring = rings
            .filter(|(err, _)| *err < LINE_BAND_PX)
            .min_by(|a, b| a.0.total_cmp(&b.0));
        let horizon = horizons
            .filter(|(err, _)| *err < LINE_BAND_PX)
            .min_by(|a, b| a.0.total_cmp(&b.0));
        match (ring, horizon) {
            (Some((re, _)), Some((he, n))) if he < re => GuideRegion::Horizon(n),
            (Some((_, factor)), _) => GuideRegion::Focal(factor),
            (None, Some((_, n))) => GuideRegion::Horizon(n),
            (None, None) => GuideRegion::Orbit,
        }
    }
}

/// The guide a row taken from `from` and dropped at gap `to` lands **after** — `None` for
/// the head of the roster.
///
/// The one piece of off-by-one arithmetic in a guide list's drag: the gap is counted in
/// the rows that *stay put*, so it is an index into the roster with the dragged row
/// removed, and the anchor is the entry one before it in that shortened list. The engine
/// resolves the anchor against exactly that list (`DocState::move_guide`), which is what
/// makes the round trip exact — and naming a guide rather than a position is what makes
/// the action mean the same thing on a peer whose roster has moved.
pub fn anchor_at(ids: &[GuideId], from: usize, to: usize) -> Option<GuideId> {
    let rest: Vec<GuideId> = ids
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != from)
        .map(|(_, id)| *id)
        .collect();
    to.min(rest.len()).checked_sub(1).map(|i| rest[i])
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
    ///
    /// `const`, so a frontend that wants one end as its own named constant can take it
    /// from here rather than spelling the number again.
    pub const fn range(self) -> (f32, f32) {
        match self {
            Dial::Cells => (CELL_OCTAVES.0 as f32, CELL_OCTAVES.1 as f32),
            Dial::Opacity => (MIN_OPACITY, 1.0),
        }
    }

    /// Whole rungs for the ladder, a hundredth for a strength.
    pub const fn step(self) -> f32 {
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

    /// Both dials say something on hover, and say it about themselves — what a column
    /// with no captions stands on. The label leads the tip so the two cannot come to
    /// call one knob two things.
    #[test]
    fn every_dial_says_what_it_does() {
        for dial in Dial::VARIANTS {
            assert!(!dial.label().is_empty(), "{dial:?} has no caption");
            assert!(
                dial.tip().starts_with(dial.label()),
                "{dial:?}'s hover does not lead with its own name: {:?}",
                dial.tip()
            );
        }
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

    /// Where each of the default guide's horizons sits, as the one coordinate it is a
    /// level set of.
    ///
    /// The default is 2-point at 30° of yaw, centred on the origin at a focal length of
    /// 900, so all three horizons are canvas-aligned: a vertical through each transverse
    /// vanishing point, and the level horizon through the center of view. Asserted rather
    /// than assumed, so a change to the default pose fails here instead of quietly aiming
    /// every press below at empty canvas.
    fn horizons_of(g: &PerspectiveGuide) -> [f32; 3] {
        std::array::from_fn(|n| match g.horizons()[n] {
            Some(PlaneTrace::Line { normal, offset }) => {
                let axial = normal.x + normal.y;
                assert!(
                    (axial.abs() - 1.0).abs() < 1e-4,
                    "horizon {n} is not canvas-aligned: {normal:?}"
                );
                -offset * axial
            }
            other => panic!("horizon {n} should be a straight line, got {other:?}"),
        })
    }

    /// Grabbing a horizon asks to turn about **its own** axis (§20.5) — the line between
    /// two axes' vanishing points belongs to the third. The index is the whole risk: the
    /// wrong one is a gesture that turns the guide about an axis the artist did not reach
    /// for, which looks like a bug in the rotation, not in a subscript.
    #[test]
    fn a_press_on_a_horizon_grabs_the_axis_it_turns_about() {
        let g = PerspectiveGuide::default();
        let h = Handles::of(&g);
        let [x_at, y_at, z_at] = horizons_of(&g);
        // Axis 1's horizon is the level one through the center of view; the other two
        // are the verticals through the transverse vanishing points.
        assert_eq!(h.at(Vec2::new(x_at, 400.0), 1.0), GuideRegion::Horizon(0));
        assert_eq!(h.at(Vec2::new(300.0, y_at), 1.0), GuideRegion::Horizon(1));
        assert_eq!(h.at(Vec2::new(z_at, 500.0), 1.0), GuideRegion::Horizon(2));
    }

    /// Everything else a press can land on still does, and the band is in **screen** px:
    /// the same canvas point is a horizon grab zoomed out and open world zoomed in.
    #[test]
    fn the_other_regions_survive_the_horizons() {
        let g = PerspectiveGuide::default();
        let h = Handles::of(&g);
        let [x_at, ..] = horizons_of(&g);
        assert_eq!(h.at(Vec2::new(4.0, -3.0), 1.0), GuideRegion::Center);
        assert_eq!(h.at(Vec2::new(0.0, g.focal), 1.0), GuideRegion::Focal(1.0));
        assert_eq!(h.at(Vec2::new(100.0, 300.0), 1.0), GuideRegion::Orbit);

        let near = Vec2::new(x_at + 50.0, 400.0);
        assert_eq!(h.at(near, 1.0), GuideRegion::Orbit, "50px is a miss");
        assert_eq!(
            h.at(near, 0.1),
            GuideRegion::Horizon(0),
            "…and 5 screen px is a hit"
        );
    }

    /// Two handles in reach: the nearer one wins, and an exact tie goes to the ring. The
    /// tie is not hypothetical — a horizon crosses the 45° circle in every pose, and under
    /// the fisheye a pair trace can *be* a ring (§20.8).
    #[test]
    fn the_nearer_handle_wins_and_a_tie_goes_to_the_ring() {
        let g = PerspectiveGuide::default();
        let h = Handles::of(&g);
        let [_, y_at, _] = horizons_of(&g);
        // Where the level horizon crosses the 45° ring, both errors are zero.
        assert_eq!(
            h.at(Vec2::new(g.focal, y_at), 1.0),
            GuideRegion::Focal(1.0),
            "a dead tie is the lens"
        );
        // 2px off the horizon and 8px outside the ring.
        assert_eq!(
            h.at(Vec2::new(g.focal + 8.0, y_at + 2.0), 1.0),
            GuideRegion::Horizon(1)
        );
        // 4px off the horizon, all but on the ring.
        assert_eq!(
            h.at(Vec2::new(g.focal, y_at + 4.0), 1.0),
            GuideRegion::Focal(1.0)
        );
    }

    /// A horizon that is not drawn is not grabbable, and the press falls through to the
    /// free world grab ([`PerspectiveGuide::horizons`]). Switching a plane off takes the
    /// one horizon that turns about its normal and leaves the other two.
    #[test]
    fn an_undrawn_horizon_cannot_be_grabbed() {
        let mut g = PerspectiveGuide::default();
        let [x_at, y_at, _] = horizons_of(&g);
        // Pair 1 is the Y/Z plane, normal to X.
        g.pairs = [true, false, true];
        let h = Handles::of(&g);
        assert_eq!(h.at(Vec2::new(x_at, 400.0), 1.0), GuideRegion::Orbit);
        assert_eq!(h.at(Vec2::new(300.0, y_at), 1.0), GuideRegion::Horizon(1));
    }

    /// [`anchor_at`] names the guide that puts the dragged row exactly where it was
    /// dropped, for **every** pair of positions — checked against the surgery the engine
    /// performs, spelled out here rather than called, so the two are held together by
    /// agreeing rather than by sharing code.
    ///
    /// Exhaustive rather than sampled: this is off-by-one arithmetic over two steps that
    /// shift indices in opposite directions, and the wrong pair is never the one anyone
    /// would think to write down. An error here is a logged, replicated undo step that
    /// rearranges the roster in a way the artist never asked for.
    #[test]
    fn the_anchor_lands_the_row_where_it_was_dropped() {
        let id = |i: usize| {
            GuideId(stark_model::document::ActionId {
                lamport: i as u64,
                actor: stark_model::document::ActorId(1),
            })
        };
        for n in 1..7usize {
            let ids: Vec<GuideId> = (0..n).map(id).collect();
            for from in 0..n {
                for to in 0..n {
                    // What the panel sends…
                    let after = anchor_at(&ids, from, to);
                    // …and what `DocState::move_guide` does with it: take the row out,
                    // then land it one past the anchor in what is left.
                    let mut rest = ids.clone();
                    let moved = rest.remove(from);
                    let slot = after
                        .and_then(|a| rest.iter().position(|g| *g == a))
                        .map_or(0, |i| i + 1);
                    rest.insert(slot, moved);
                    // …against the list surgery the drag was drawn against.
                    let mut want = ids.clone();
                    let row = want.remove(from);
                    want.insert(to.min(want.len()), row);
                    assert_eq!(
                        rest, want,
                        "n={n} from={from} to={to}: the anchor landed the row elsewhere"
                    );
                }
            }
        }
    }
}
