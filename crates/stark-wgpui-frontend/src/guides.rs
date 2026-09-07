//! The Drawing Guides shelf: the roster of perspective guides, and the dressing of
//! whichever one is in hand (§20.5).
//!
//! Shaped like the Layers shelf beside it, because it answers the same question about
//! a different stack: a row per guide, an eye saying whether *this client* draws it, a
//! trash, and a press that takes one up. What a guide *is* is `stark_model`'s and
//! reaches a shader without passing through here (§20.4).
//!
//! # The canvas gesture is not here yet
//!
//! The web app shapes a guide by dragging on the painting (§20.5) — the free arc, the
//! constrained turn about a horizon, the 45° circle that *is* the focal length. That is
//! a composing mode with a full-viewport catcher, and this frontend has one of those
//! (`crate::transform`) rather than a place to put a second, so it is a stage of its
//! own (§11.2).
//!
//! So a guide added here is `PerspectiveGuide::default` placed where the artist is
//! looking: two-point, turned 30°, drawing immediately. This shelf then dresses it —
//! how fine the grid is, how strongly it reads, which planes are ruled, which lens.

use stark_engine::ObservableState;
use stark_engine::command::{DocCommand, ViewCommand};
use stark_model::document::{GuideId, Lens, PerspectiveGuide};
use stark_ui::commands::{Bindings, Command};
use stark_ui::icons::Icon;
use wgpui::{Bounds, IntoElement, Pixels, Point, SharedString, canvas, div, prelude::*, px};

use crate::controls::Controls;
use crate::style::{self, StyleExt};

/// The cell scale the shelf offers, as **halvings** of the default lattice (§20.3):
/// two steps coarser to two steps finer, and nothing in between.
///
/// The model draws a valid grid at any scale; this is what the control offers, and
/// the reason is that a grid meant to be counted on should *refine* rather than
/// slide. Double the cells and every line of the coarser grid is still a line of the
/// finer one with a new line between each pair — nothing already counted against
/// moves. At any other ratio the whole family slides along its pencil toward the
/// corner's own edge, which reads as the grid drifting sideways.
///
/// The web app's own figures (`panels::guides::CELL_OCTAVES`), because a rung has to
/// be the same grid in both apps.
pub const CELL_OCTAVES: (i32, i32) = (-2, 2);

/// The floor the opacity track offers. A guide at zero is a guide that is on and
/// invisible, which the eye already says better.
const MIN_OPACITY: f32 = 0.1;

/// The two axes of each pair plane, in the order the chip shows them: XY, YZ, ZX.
///
/// The model's own cyclic order, pair `k` being spanned by axes `(k, k+1)`, and the
/// chips are read in it rather than sorted: the three then run X→Y→Z→X, so each chip
/// picks up where the last left off and every axis letter appears exactly twice.
const PAIR_AXES: [[usize; 2]; 3] = [[0, 1], [1, 2], [2, 0]];

const AXIS_NAMES: [&str; 3] = ["X", "Y", "Z"];

/// The shelf's two continuous knobs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dial {
    /// How fine the grid is, in halvings of the default lattice.
    Cells,
    /// How strongly the whole overlay reads over the paint.
    Opacity,
}

/// Both, in the order the shelf draws them.
pub const DIALS: [Dial; 2] = [Dial::Cells, Dial::Opacity];

impl Dial {
    pub fn glyph(self) -> Icon {
        match self {
            // A fan of lines from a point, which is exactly what this number counts:
            // the guide's fans are its parametrization (§20.5).
            Dial::Cells => stark_ui::icons::DENSITY,
            Dial::Opacity => stark_ui::icons::OPACITY,
        }
    }

    fn tip(self) -> &'static str {
        match self {
            Dial::Cells => {
                "Cells \u{2014} how fine the grid is; each step halves the cell, so every \
                 line of the coarser grid is still a line of this one"
            }
            Dial::Opacity => "Opacity \u{2014} how strongly the guide reads over the paint",
        }
    }

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

    /// The guide `value` asks for. Stated as a whole edit of the camera rather than
    /// as a field write, because that is the shape every act on this shelf takes:
    /// `DocCommand::SetGuide` carries the camera entire (§20.5).
    pub fn write(self, mut guide: PerspectiveGuide, value: f32) -> PerspectiveGuide {
        match self {
            Dial::Cells => {
                // Stepped off the *default* rather than off the guide's current
                // lattice, so a rung is the same grid however it was reached.
                let base = PerspectiveGuide::default().lattice;
                guide.lattice = base * 2f32.powi(value.round() as i32);
            }
            Dial::Opacity => guide.opacity = value.clamp(MIN_OPACITY, 1.0),
        }
        guide
    }

    fn readout(self, v: f32) -> String {
        match self {
            Dial::Cells => format!("{v:+.0}"),
            Dial::Opacity => format!("{v:.2}"),
        }
    }
}

/// Which rung of the cell ladder `guide` stands on.
///
/// The grid's scale is the *length* of the lattice: how many cells lie between the
/// eye and its corner, which is all a camera with no world scale of its own can say
/// about the size of a cell (§20.3).
fn octave(guide: &PerspectiveGuide) -> f32 {
    let base = PerspectiveGuide::default().lattice.length();
    if base <= 0.0 || guide.lattice.length() <= 0.0 {
        return 0.0;
    }
    (guide.lattice.length() / base).log2().round()
}

/// What to call a guide that has never been named: its place in the roster.
///
/// Numbered by *position*, which shifts when a row above it goes — the honest reading
/// for a row that is being described rather than named. Naming it is how you stop it
/// moving. The web app's roster says the same thing the same way.
pub fn label(index: usize, guide: &stark_engine::GuideInfo) -> String {
    match &guide.name {
        Some(name) => name.to_string(),
        None => format!("Perspective {}", index + 1),
    }
}

/// Which control a press on the shelf landed on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Region {
    /// Add a perspective, placed where the artist is looking.
    Add,
    /// A row's body: take this guide up, so the dressing below acts on it.
    Row(usize),
    /// Its eye — whether **this client** draws it (§20.5). Never saved, never sent.
    Visible(usize),
    /// Its trash.
    Remove(usize),
    /// One of the three pair planes of the guide in hand, by index into [`PAIR_AXES`].
    Pair(usize),
    /// The lens toggle (§20.8).
    Lens,
}

/// Where each was laid out — `crate::panel`'s device, for its reason.
pub type Regions = std::rc::Rc<std::cell::RefCell<Vec<(Region, Bounds<Pixels>)>>>;

fn probe(regions: &Regions, region: Region) -> impl IntoElement {
    let regions = regions.clone();
    canvas(
        move |bounds, _, _| regions.borrow_mut().push((region, bounds)),
        |_, (), _, _| {},
    )
    .absolute()
    .size_full()
}

/// Which control a press landed on.
pub fn hit(regions: &Regions, at: Point<Pixels>) -> Option<Region> {
    // Reversed, so a row's eye and trash win over the row under them: the three
    // overlap by construction and the list is built parent-first.
    regions
        .borrow()
        .iter()
        .rev()
        .find(|(_, bounds)| bounds.contains(&at))
        .map(|(region, _)| *region)
}

/// The guide this shelf's dressing acts on: the one last taken up, or — for a client
/// that has taken none, or whose choice an undo has removed — the newest.
///
/// Resolved against the roster every frame rather than kept in step with it, which is
/// the same bargain the Layers shelf's active row makes: a guide can go away under a
/// peer's edit or an undo, and a held id that no longer names one would leave the
/// tracks pointed at nothing.
pub fn chosen(o: Option<&ObservableState>, held: Option<GuideId>) -> Option<GuideId> {
    let guides = &o?.guides;
    held.filter(|id| guides.iter().any(|g| g.id == *id))
        .or_else(|| guides.last().map(|g| g.id))
}

/// Build the shelf's body.
pub fn guides_body(
    o: Option<&ObservableState>,
    held: Option<GuideId>,
    bindings: &Bindings,
    controls: &Controls,
    regions: &Regions,
) -> impl IntoElement + use<> {
    let guides: Vec<stark_engine::GuideInfo> = o.map(|o| o.guides.to_vec()).unwrap_or_default();
    let taken = chosen(o, held);
    let camera = guides.iter().find(|g| Some(g.id) == taken).map(|g| g.guide);

    div()
        .flex()
        .flex_col()
        .gap_1()
        .child(
            div().flex().gap_1().child(style::tip(
                div()
                    .id("guide-add")
                    .relative()
                    .chip()
                    .flex_1()
                    .flex()
                    .items_center()
                    .justify_center()
                    .py_1p5()
                    .resting()
                    .child(probe(regions, Region::Add))
                    .child(crate::icons::icon(
                        Command::AddPerspective.icon(),
                        style::INK_MARK,
                    )),
                Command::AddPerspective.tooltip(bindings),
            )),
        )
        // The roster. Empty is a real state and says so in words: there is no mark
        // for "nothing here yet", and a blank strip would read as a broken panel.
        .children(guides.is_empty().then(|| {
            div()
                .py_1()
                .caption()
                .child("No guides yet \u{2014} add a perspective to draw through.")
        }))
        .children(guides.iter().enumerate().map(|(i, g)| {
            let worn = Some(g.id) == taken;
            div()
                .relative()
                .flex()
                .items_center()
                .gap_1()
                .px_1()
                .py_0p5()
                .rounded_sm()
                .lit_row(worn)
                .child(chip(
                    regions,
                    Region::Visible(i),
                    if g.visible {
                        stark_ui::icons::VISIBLE
                    } else {
                        stark_ui::icons::HIDDEN
                    },
                    g.visible,
                    if g.visible {
                        "Stop drawing this guide"
                    } else {
                        "Draw this guide"
                    },
                ))
                .child(
                    // The name takes the slack, so the trash stays put down the column
                    // however long a guide is called. A name the artist chose, so it
                    // stays a word (`crate::panel`).
                    div()
                        .relative()
                        .flex_1()
                        .text_sm()
                        .truncate()
                        .cursor_pointer()
                        .child(probe(regions, Region::Row(i)))
                        .child(label(i, g)),
                )
                .child(chip(
                    regions,
                    Region::Remove(i),
                    stark_ui::icons::REMOVE,
                    false,
                    "Remove this guide",
                ))
        }))
        // The dressing of whichever guide is in hand. Absent rather than dim with no
        // guide at all: there is nothing for these to be about, and a run of dead
        // tracks says less than the empty line above does.
        .children(camera.map(|g| {
            div()
                .flex()
                .flex_col()
                .gap_1()
                .pt_1()
                // Which of the three pair planes are ruled (§20.3). A plane rather
                // than an axis because a guide line lies *in* one, and because three
                // planes are independently switchable where three axes are not: two
                // axes could never show one plane without a second coming free.
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap_2()
                        .child(crate::icons::icon(
                            stark_ui::icons::VISIBLE,
                            style::INK_MARK,
                        ))
                        .child(div().flex().flex_1().gap_1().children((0..3).map(|k| {
                            let [a, b] = PAIR_AXES[k];
                            let on = g.pairs[k];
                            style::tip(
                                div()
                                    .id(SharedString::from(format!("plane-{k}")))
                                    .relative()
                                    .chip()
                                    .flex_1()
                                    .py_1()
                                    .text_center()
                                    .lit(on)
                                    .child(probe(regions, Region::Pair(k)))
                                    .child(format!("{}{}", AXIS_NAMES[a], AXIS_NAMES[b])),
                                format!(
                                    "Show the {}{} plane \u{2014} its two fans of guide \
                                         lines, its horizon and its station point",
                                    AXIS_NAMES[a], AXIS_NAMES[b]
                                ),
                            )
                        }))),
                )
                // The lens (§20.8): one toggle, because everything else about the
                // camera means the same thing under both projections.
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap_2()
                        .child(crate::icons::icon(
                            stark_ui::icons::PERSPECTIVE_GRID,
                            style::INK_MARK,
                        ))
                        .child(style::tip(
                            div()
                                .id("guide-lens")
                                .relative()
                                .chip()
                                .flex_1()
                                .flex()
                                .items_center()
                                .justify_center()
                                .py_1p5()
                                .lit(g.lens == Lens::Fisheye)
                                .child(probe(regions, Region::Lens))
                                .child(crate::icons::icon(
                                    stark_ui::icons::FISHEYE,
                                    if g.lens == Lens::Fisheye {
                                        style::INK_LIT
                                    } else {
                                        style::INK_MARK
                                    },
                                )),
                            "Fisheye \u{2014} a stereographic lens: straight world lines bow \
                             into circles, and both poles of every axis come into view",
                        )),
                )
                .children(DIALS.into_iter().map(|dial| {
                    crate::panel::Slider::new(
                        dial.glyph(),
                        dial.tip(),
                        dial.readout(dial.read(&g)),
                        controls.guide(dial),
                    )
                }))
        }))
}

/// A small square control on a roster row — an eye, a trash.
fn chip(
    regions: &Regions,
    region: Region,
    glyph: Icon,
    on: bool,
    tip: &'static str,
) -> impl IntoElement + use<> {
    style::tip(
        div()
            .id(SharedString::from(format!("{region:?}")))
            .relative()
            .w(px(20.))
            .h(px(18.))
            .flex()
            .items_center()
            .justify_center()
            .rounded_sm()
            .cursor_pointer()
            .child(probe(regions, region))
            .child(crate::icons::icon(
                glyph,
                if on { style::INK_LIT } else { style::INK_MARK },
            )),
        tip,
    )
}

/// What a press means, given the roster it was drawn over and the guide in hand.
///
/// A function over the projection rather than a `match` in the view, so the mapping
/// is testable — which is the same split `crate::layers::act` makes, and for the same
/// reason: every arm is a claim about §20.5's vocabulary.
pub fn act(
    region: Region,
    guides: &[stark_engine::GuideInfo],
    taken: Option<GuideId>,
    center: stark_model::geom::Vec2,
) -> Option<Act> {
    let row = |i: usize| guides.get(i);
    // The camera in hand, for the three acts that edit one.
    let held = || {
        let id = taken?;
        guides.iter().find(|g| g.id == id).map(|g| (id, g.guide))
    };
    Some(match region {
        // Placed where the artist is looking, and appended: the roster reads in the
        // order guides were added.
        Region::Add => Act::Doc(DocCommand::AddGuide {
            guide: PerspectiveGuide {
                center,
                ..Default::default()
            },
            after: guides.last().map(|g| g.id),
            name: None,
        }),
        // Whether this client draws it is **session** state, not the document's: a
        // collaborator's eye is theirs (§20.5).
        Region::Visible(i) => {
            let g = row(i)?;
            Act::View(ViewCommand::SetGuideVisible(g.id, !g.visible))
        }
        Region::Remove(i) => Act::Doc(DocCommand::RemoveGuide(row(i)?.id)),
        Region::Row(i) => Act::Take(row(i)?.id),
        Region::Pair(k) => {
            let (id, mut guide) = held()?;
            let on = *guide.pairs.get(k)?;
            guide.pairs[k] = !on;
            Act::Doc(DocCommand::SetGuide(id, guide))
        }
        Region::Lens => {
            let (id, mut guide) = held()?;
            guide.lens = match guide.lens {
                Lens::Rectilinear => Lens::Fisheye,
                Lens::Fisheye => Lens::Rectilinear,
            };
            Act::Doc(DocCommand::SetGuide(id, guide))
        }
    })
}

/// What a press turns into. Three kinds, because they are three kinds of state (§4).
pub enum Act {
    /// A document edit: logged, undoable, replicated.
    Doc(DocCommand),
    /// A view setting — whether *this* client draws the guide.
    View(ViewCommand),
    /// Which guide this shelf's dressing acts on. The panel's own state: nothing
    /// about the document changes, and a collaborator's choice is theirs.
    Take(GuideId),
}

#[cfg(test)]
mod tests {
    use super::*;
    use stark_model::document::{ActionId, ActorId};

    fn info(n: u64, guide: PerspectiveGuide) -> stark_engine::GuideInfo {
        stark_engine::GuideInfo {
            id: GuideId(ActionId {
                lamport: n,
                actor: ActorId::SOLO,
            }),
            name: None,
            visible: false,
            guide,
        }
    }

    fn roster() -> Vec<stark_engine::GuideInfo> {
        vec![
            info(1, PerspectiveGuide::default()),
            info(2, PerspectiveGuide::default()),
        ]
    }

    /// Both dials wear a mark and say what they do — what a column with no labels
    /// stands on (`crate::panel`).
    #[test]
    fn every_dial_is_marked_and_says_what_it_does() {
        for dial in DIALS {
            assert!(dial.glyph().svg().is_some(), "{dial:?} has no glyph");
            assert!(!dial.tip().is_empty());
            let (lo, hi) = dial.range();
            assert!(hi > lo);
        }
    }

    /// A rung of the cell ladder reads back as the rung it was written at, which is
    /// what lets the track draw its handle where the hand left it.
    #[test]
    fn a_cell_rung_reads_back_as_itself() {
        for rung in CELL_OCTAVES.0..=CELL_OCTAVES.1 {
            let g = Dial::Cells.write(PerspectiveGuide::default(), rung as f32);
            assert_eq!(Dial::Cells.read(&g), rung as f32, "rung {rung}");
        }
    }

    /// The default guide stands at rung zero, so the ladder is centred on what an add
    /// produces rather than on an arbitrary lattice.
    #[test]
    fn the_default_guide_stands_at_the_middle_rung() {
        assert_eq!(octave(&PerspectiveGuide::default()), 0.0);
    }

    /// A degenerate lattice — one a peer could send, since the model clamps for
    /// finiteness and not for length — reads as the middle rung rather than as a
    /// negative infinity the track would place off its own end.
    #[test]
    fn a_collapsed_lattice_still_lands_on_a_rung() {
        let mut g = PerspectiveGuide::default();
        // Through the default's own lattice rather than a named zero: `Vec3` is
        // glam's and the model does not re-export it, and what matters here is the
        // length rather than the type.
        g.lattice *= 0.0;
        assert_eq!(octave(&g), 0.0);
    }

    /// The eye is **session** state and the planes are the document's — the split
    /// §20.5 draws, and the one thing this mapping could get wrong invisibly.
    #[test]
    fn the_eye_is_this_clients_and_the_plane_is_the_documents() {
        let guides = roster();
        let taken = Some(guides[0].id);
        assert!(matches!(
            act(Region::Visible(0), &guides, taken, Default::default()),
            Some(Act::View(ViewCommand::SetGuideVisible(_, true)))
        ));
        match act(Region::Pair(1), &guides, taken, Default::default()) {
            Some(Act::Doc(DocCommand::SetGuide(_, g))) => {
                assert!(!g.pairs[1], "the chip toggles the plane it names");
                assert!(g.pairs[0] && g.pairs[2], "and leaves the other two alone");
            }
            _ => panic!("a plane chip edits the guide"),
        }
    }

    /// An act on the guide in hand asks for nothing when there is none — rather than
    /// reaching for a row that is not there.
    #[test]
    fn the_dressing_needs_a_guide_in_hand() {
        let guides = roster();
        assert!(act(Region::Lens, &guides, None, Default::default()).is_none());
        assert!(act(Region::Pair(0), &guides, None, Default::default()).is_none());
        // Add is the exception: it is what there being none is for.
        assert!(matches!(
            act(Region::Add, &guides, None, Default::default()),
            Some(Act::Doc(DocCommand::AddGuide { .. }))
        ));
    }

    /// A choice an undo has removed falls back to the newest guide rather than
    /// leaving the dressing pointed at nothing.
    #[test]
    fn a_choice_that_has_gone_falls_back_to_the_newest() {
        let guides = roster();
        let gone = info(99, PerspectiveGuide::default()).id;
        let last = guides.last().unwrap().id;
        let held = |held| held_chosen(&guides, held);
        assert_eq!(held(Some(guides[0].id)), Some(guides[0].id));
        assert_eq!(held(Some(gone)), Some(last));
        assert_eq!(held(None), Some(last));
        assert_eq!(held_chosen(&[], None), None);
    }

    /// [`chosen`]'s reading, without a projection to hang it off — the half worth
    /// testing.
    fn held_chosen(guides: &[stark_engine::GuideInfo], held: Option<GuideId>) -> Option<GuideId> {
        held.filter(|id| guides.iter().any(|g| g.id == *id))
            .or_else(|| guides.last().map(|g| g.id))
    }
}
