//! What the Layers panel draws, worked out from the flat list the engine
//! projects — and what a drop into it means (§14.6, §14.8).
//!
//! **Nothing here renders anything.** It is the panel's arithmetic: which rows are
//! folded away, what a row's own Carry and Release would do, and — given a drag —
//! where the block would land, at what depth, and what that spells as a
//! [`MoveLayer`](stark_engine::command::DocCommand::MoveLayer). The rsx! that draws
//! it is [`layer`](crate::panels::layer), which is 900 lines of chrome this used to be buried in.
//!
//! It is a file of its own for `crate::gesture`'s reason: this is the part of the
//! panel that can be *tested*, and ten of the crate's tests are here — the ones
//! that can say a drop between two rows means one place in the tree rather than
//! another. Inside 1500 lines of markup they were the tests hardest to find and
//! the code most likely to be skimmed as layout.
//!
//! The column arithmetic underneath — which rows yield, and by how much — is not
//! here either. That is [`reorder`](crate::panels::reorder), shared with the guide list; what is
//! here is the part only a *tree* has.

use std::collections::{HashMap, HashSet};

use super::reorder::{Grab, Motion, Slide};
use stark_engine::LayerInfo;
use stark_model::document::LayerId;
use stark_model::document::Place;

/// How far one level of membership indents a row, in pixels. Named because three
/// things are measured in it: the row's own offset, the slot the indent leaves empty
/// to its left, which is where Release sits, and — since a drag can change a row's
/// depth as well as its place — how far sideways the pointer must travel to mean one
/// level of it.
pub(super) const INDENT: usize = 14;

/// A row as the panel draws it: the layer, plus what its neighbours in the flat
/// list say about it that the layer alone cannot.
#[derive(Clone, PartialEq)]
pub struct Row {
    pub info: LayerInfo,
    /// Collapsed away under a group whose triangle is shut.
    pub(super) hidden: bool,
    /// Shut, for a group. Meaningless for a layer that carries nothing.
    pub(super) collapsed: bool,
    /// The layer directly below this one *in its own stack* — what Carry would put
    /// it on, and the layer a clip would be bounded by. `None` at the foot of a
    /// stack, where there is nothing to be carried by.
    pub(super) carry_onto: Option<LayerId>,
    /// The group this layer is in, and what carries *that* — between them, where
    /// Release would put it: out of the group and directly above it. `None` for a
    /// layer that is not in a group, which is the only state Release has nothing to
    /// say about.
    pub(super) release_to: Option<(LayerId, Option<LayerId>)>,
    /// Whether Remove would leave a document behind. Removing a group takes what it
    /// carries with it (§14.2), so the floor is not "more than one row" but
    /// "something would be left" — which for a row deep in a group is nearly always
    /// true, and for the sole top-level stack never is.
    pub(super) removable: bool,
}

/// What a drag would commit, resolved against the rows as they stand now.
///
/// The column arithmetic — which rows yield, and by how much — is [`Slide`], shared
/// with the other roster that is dragged this way (`panels::reorder`). What is here
/// is the part only a *tree* has: which depth the hand is asking for, and what that
/// spells as a place in the document.
#[derive(Clone, Copy, PartialEq)]
pub(super) struct Landing {
    /// Which rows travel and what the rest do about it. The block is the dragged row
    /// and everything it carries — contiguous in the panel, because a base is drawn
    /// under exactly the rows it carries and nothing may come between.
    pub(super) slide: Slide,
    /// The depth it lands at, which the pointer's *horizontal* travel chooses among
    /// the depths this seam can express.
    pub(super) depth: usize,
    /// Where the block is drawn while in flight.
    pub(super) shift: (f32, f32),
    /// The move this commits to (§14.8).
    pub(super) carrier: Option<LayerId>,
    pub(super) at: Place,
    /// Whether that move would change anything. A drag that ends where it began must
    /// not spend an undo step saying so — in a tree that means the same slot *and*
    /// the same depth, since one seam can hold several depths.
    pub(super) inert: bool,
}

impl Landing {
    /// How to draw the row at display index `i`.
    pub(super) fn motion(&self, i: usize) -> Motion {
        self.slide.motion(i, self.shift)
    }
}

/// Resolve an in-flight drag against the displayed rows: where the block would land,
/// what that spells as a
/// [`MoveLayer`](stark_engine::command::DocCommand::MoveLayer), and how to draw the
/// panel meanwhile.
///
/// `None` when the drag cannot be resolved — a row without a box, which is what a
/// document that changed under the pointer looks like (a peer's edit, §17). The panel
/// then draws itself at rest and the release commits nothing, rather than acting on
/// geometry that describes a tree that is gone.
///
/// **A gap, not a row, is the drop target**, and the pointer's *x* chooses the depth
/// within it. That is the whole gesture: the same drop point between two rows can
/// mean several different places in several different stacks — all of which draw at
/// that one seam — and which one it means is how far right you are holding it. Where
/// a gap can only mean one thing, which is most of them, x does nothing at all.
///
/// The depths a gap can express run from the depth of the row **above** it to one
/// past the depth of the row **below** it. Both bounds are the panel's own picture
/// read back: a drop cannot be shallower than the row it is directly under (that
/// position renders somewhere else entirely — outside the group, below its base), and
/// one deeper than the row below it is that row carrying the drop, since what a layer
/// carries is drawn directly above it.
///
/// Two things fall out rather than being checked. A cycle is impossible: every place
/// this can name is stated against a row that **stays put**, and a row inside the
/// travelling block is not one. And the landing is total — every depth in the range
/// names exactly one real position, because the ancestors of the row below the gap
/// cover every depth beneath it without a gap.
pub(super) fn landing(display: &[Row], drag: &Grab) -> Option<Landing> {
    // Each row's box, found by the id it wears rather than by where it sits.
    let keys: Vec<String> = display.iter().map(|r| r.info.id.to_string()).collect();
    let (from, boxes) = drag.resolve(&keys)?;
    let deep = display[from].info.depth;
    // The block reaches back over the rows this one carries — everything above it
    // that is deeper than it, up to the first row that is not.
    let start = display[..from]
        .iter()
        .rposition(|r| r.info.depth <= deep)
        .map_or(0, |i| i + 1);
    let (dx, dy) = drag.delta();
    let slide = Slide::resolve(&boxes, (start, from), dy)?;
    let gap = slide.gap;
    // The rows that stay put, in display order — what `gap` is counted in.
    let rest: Vec<usize> = (0..display.len())
        .filter(|i| *i < start || *i > from)
        .collect();

    let above = gap.checked_sub(1).map(|k| &display[rest[k]]);
    let below = rest.get(gap).map(|&k| &display[k]);
    let low = above.map_or(0, |r| r.info.depth);
    // One past the row below is that row *carrying* the drop — unless the row below
    // is a filter, which never carries (§21.2): the engine would refuse the move, so
    // the panel does not draw a place it cannot drop into.
    let high = below
        .map_or(0, |r| r.info.depth + usize::from(r.info.filter.is_none()))
        .max(low);
    // Relative to the depth it was grabbed at, so a straight-down drag keeps the
    // nesting it had wherever that is still legal, and only a sideways one changes it.
    let want = (deep as f32 + dx / INDENT as f32).round().max(0.0) as usize;
    let depth = want.clamp(low, high);

    let (carrier, at) = match below {
        // Under everything: the foot of the document's own stack, the one place no
        // sibling can name.
        None => (None, Place::Bottom),
        Some(b) if depth > b.info.depth => (
            Some(b.info.id),
            // What a layer carries is drawn above it, so the seam directly over a row
            // is the *foot* of its carried stack — unless those rows are folded away,
            // in which case this seam stands for the whole subtree and a drop belongs
            // on top of it, where it will be when the fold opens.
            if b.collapsed && b.info.is_group {
                Place::Top
            } else {
                Place::Bottom
            },
        ),
        // Otherwise the drop is in an ancestor's stack: `depth` picks which one, and
        // it goes directly above the ancestor that sits at that depth.
        Some(b) => {
            let mut anchor = b;
            while anchor.info.depth > depth {
                let up = anchor.info.carrier?;
                anchor = display.iter().find(|r| r.info.id == up)?;
            }
            (anchor.info.carrier, Place::Above(anchor.info.id))
        }
    };

    Some(Landing {
        slide,
        depth,
        // The block keeps its own shape and moves as one: every row in it takes the
        // same shift, so the indents *within* a dragged group stay where they were
        // while the whole of it steps to the depth it is going to.
        shift: ((depth as f32 - deep as f32) * INDENT as f32, dy),
        carrier,
        at,
        // Same slot, same depth, same tree.
        inert: slide.inert() && depth == deep,
    })
}

/// The flat list decorated with what the panel needs and the projection does not
/// carry: which rows are folded away under a shut group, and what the row's own
/// Carry and Release would do.
///
/// Walks bottom-to-top, the order `observe()` produces, keeping the depth at which
/// the enclosing group was shut. Everything deeper than that is hidden until the
/// walk comes back out — which is exactly "hidden iff some ancestor is collapsed",
/// computed in one pass without ever looking a parent up.
///
/// The two move targets fall out of the same order for the same reason, which is why
/// they are computed here rather than per row: the nearest sibling *below* a layer is
/// simply the last one seen in its stack, and a group's base is walked before
/// anything it carries, so what carries the group is known by the time a member asks.
/// Both are one map lookup per row rather than a scan of the list per row.
pub(super) fn rows(layers: &[LayerInfo], collapsed: &HashSet<LayerId>) -> Vec<Row> {
    let mut out = Vec::with_capacity(layers.len());
    let mut shut_at: Option<usize> = None;
    // The topmost layer seen so far in each stack, and whether it is a filter —
    // which the Carry answer below has to know, since a filter never carries
    // (§21.2) and a button that spelled a move the engine refuses would be a lie.
    let mut top_of: HashMap<Option<LayerId>, (LayerId, bool)> = HashMap::new();
    // What carries each layer, for the rows that will ask about their group.
    let mut outer_of: HashMap<LayerId, Option<LayerId>> = HashMap::new();
    for info in layers {
        if shut_at.is_some_and(|d| info.depth <= d) {
            shut_at = None;
        }
        let hidden = shut_at.is_some();
        let collapsed = collapsed.contains(&info.id);
        if !hidden && collapsed && info.is_group {
            shut_at = Some(info.depth);
        }
        let carry_onto = top_of
            .get(&info.carrier)
            .and_then(|&(id, filter)| (!filter).then_some(id));
        let release_to = info
            .carrier
            .map(|group| (group, outer_of.get(&group).copied().flatten()));
        // After the reads, so neither answer is the layer itself. A collapsed group's
        // members are still in their stack, so this happens for hidden rows too.
        top_of.insert(info.carrier, (info.id, info.filter.is_some()));
        outer_of.insert(info.id, info.carrier);
        out.push(Row {
            info: info.clone(),
            hidden,
            collapsed,
            carry_onto,
            release_to,
            removable: subtree_len(layers, info.id) < layers.len(),
        });
    }
    out
}

/// How many rows `id` takes with it if removed: itself, plus everything it carries
/// at any depth. Those are exactly the rows that follow it while deeper than it.
fn subtree_len(layers: &[LayerInfo], id: LayerId) -> usize {
    let Some(at) = layers.iter().position(|l| l.id == id) else {
        return 0;
    };
    let depth = layers[at].depth;
    1 + layers[at + 1..]
        .iter()
        .take_while(|l| l.depth > depth)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use stark_model::document::BlendMode;

    /// A row's height and the gap between two of them, as the stylesheet leaves them.
    /// The exact numbers do not matter — every answer here is a comparison against a
    /// row's centre — but they have to be *consistent*, which is what makes a drag of
    /// `STEP` "one row down".
    const H: f32 = 20.0;
    const GAP: f32 = 8.0;
    const STEP: f32 = H + GAP;
    const FIRST: f32 = 100.0;

    fn info(id: u64, depth: usize, carrier: Option<u64>, is_group: bool) -> LayerInfo {
        LayerInfo {
            id: LayerId::solo(id),
            blend: BlendMode::Normal,
            clip: false,
            opacity: 1.0,
            visible: true,
            carrier: carrier.map(LayerId::solo),
            depth,
            is_group,
            has_backdrop: true,
            name: None,
            matte: None,
            filter: None,
            has_underlay: true,
            // These tests are about where a drag lands, which is geometry — nothing
            // here reads the merge, and offering one would say something about a pair
            // of rows that the fixture's `(id, depth)` pairs do not describe.
            merge_down: None,
            // Every fixture row is a paint layer, and one revision serves them all:
            // nothing here draws a thumbnail, and a distinct number per row would be
            // saying something about their contents that these tests do not measure.
            content_revision: Some(0),
            translation: stark_model::geom::IVec2::ZERO,
        }
    }

    /// A display list from `(id, depth)` pairs given **as the panel draws them** — top
    /// of the document first, a base under the rows it carries.
    ///
    /// Each row's carrier is derived rather than stated, because the picture already
    /// says it: a row's carrier is the first row *below* it that is one level
    /// shallower. Deriving it is also what keeps a malformed spec from quietly
    /// describing a tree the panel could never draw.
    ///
    /// `shut` names the groups whose carried rows are folded away — the one thing a
    /// display list cannot show, since those rows are not in it.
    fn display(spec: &[(u64, usize)], shut: &[u64]) -> Vec<Row> {
        spec.iter()
            .enumerate()
            .map(|(i, &(id, depth))| {
                let carrier = spec[i + 1..]
                    .iter()
                    .find(|(_, d)| *d < depth)
                    .map(|(base, _)| *base);
                let carries = spec[..i]
                    .iter()
                    .rev()
                    .take_while(|(_, d)| *d > depth)
                    .count()
                    > 0;
                Row {
                    info: info(id, depth, carrier, carries || shut.contains(&id)),
                    hidden: false,
                    collapsed: shut.contains(&id),
                    carry_onto: None,
                    release_to: None,
                    removable: true,
                }
            })
            .collect()
    }

    /// Every row's box, in display order — with `missing` left out, which is what a
    /// list that changed under the hand looks like.
    fn boxes(rows: &[Row], missing: Option<u64>) -> Vec<(String, f32, f32)> {
        rows.iter()
            .enumerate()
            .filter(|(_, r)| missing.map(LayerId::solo).as_ref() != Some(&r.info.id))
            .map(|(i, r)| (r.info.id.to_string(), FIRST + i as f32 * STEP, H))
            .collect()
    }

    /// A drag of row `id` by `(dx, dy)`, taken from the middle of its own row.
    fn drag(rows: &[Row], id: u64, dx: f32, dy: f32) -> Grab {
        drag_of(boxes(rows, None), rows, id, dx, dy)
    }

    fn drag_of(measured: Vec<(String, f32, f32)>, rows: &[Row], id: u64, dx: f32, dy: f32) -> Grab {
        let at = rows
            .iter()
            .position(|r| r.info.id == LayerId::solo(id))
            .expect("the dragged row is displayed");
        let anchor = (200.0, FIRST + at as f32 * STEP + H * 0.5);
        let mut grab = Grab::begin(LayerId::solo(id).to_string(), measured, anchor);
        grab.track((anchor.0 + dx, anchor.1 + dy), true);
        grab
    }

    /// A flat document: three layers in the root stack, drawn top-first.
    fn flat() -> Vec<Row> {
        display(&[(1, 0), (2, 0), (3, 0)], &[])
    }

    /// Dragging a row past its neighbour reorders it within its own stack — the move
    /// the panel had no control for at all, since Carry and Release only ever changed
    /// *whose* stack a layer is in.
    #[test]
    fn a_drag_down_lands_above_the_row_it_passed() {
        let rows = flat();
        let l = landing(&rows, &drag(&rows, 1, 0.0, STEP)).expect("resolves");
        assert_eq!((l.carrier, l.at), (None, Place::Above(LayerId::solo(3))));
        assert!(!l.inert);
    }

    /// The foot of a stack — the place `Place::Above` cannot name, and the reason the
    /// anchor grew a third state at all.
    #[test]
    fn a_drag_to_the_bottom_lands_under_everything() {
        let rows = flat();
        let l = landing(&rows, &drag(&rows, 1, 0.0, 3.0 * STEP)).expect("resolves");
        assert_eq!((l.carrier, l.at), (None, Place::Bottom));
    }

    /// Sideways is what nests. The same drop point one indent to the right is not a
    /// place in the same stack — it is the row below the seam carrying the layer,
    /// which is exactly what the Carry button spells (§14.2).
    #[test]
    fn a_drag_to_the_right_is_what_carries() {
        let rows = flat();
        let straight = landing(&rows, &drag(&rows, 1, 0.0, STEP)).expect("resolves");
        let over = landing(&rows, &drag(&rows, 1, INDENT as f32, STEP)).expect("resolves");
        assert_eq!(straight.depth, 0);
        assert_eq!(
            (over.carrier, over.at),
            (Some(LayerId::solo(3)), Place::Bottom),
            "one indent right of the same seam, layer 3 carries it"
        );
    }

    /// The seam directly over a **filter** row never means "carried by the filter":
    /// a filter never carries (§21.2) and the engine refuses the move, so the panel
    /// must not draw a place it cannot drop into. The same rightward travel that
    /// nests into an ordinary row (the test above) lands beside a filter instead.
    #[test]
    fn a_filter_row_offers_no_carry_depth() {
        use stark_model::document::{ColorAdjust, Filter};
        let mut rows = flat();
        rows.iter_mut()
            .find(|r| r.info.id == LayerId::solo(3))
            .expect("row 3 is displayed")
            .info
            .filter = Some(Filter::Color(ColorAdjust::NEUTRAL));
        let over = landing(&rows, &drag(&rows, 1, INDENT as f32, STEP)).expect("resolves");
        assert_eq!(
            (over.depth, over.carrier, over.at),
            (0, None, Place::Above(LayerId::solo(3))),
            "an indent right of the seam over a filter stays in the filter's stack"
        );
    }

    /// A drag that ends where it began commits nothing. Undo is cheap but it is not
    /// free, and a step that does nothing when you reach it is worse than no step.
    #[test]
    fn a_drag_that_goes_nowhere_is_inert() {
        let rows = flat();
        for dy in [0.0, 6.0, -6.0] {
            let l = landing(&rows, &drag(&rows, 2, 0.0, dy)).expect("resolves");
            assert!(l.inert, "a {dy}px drag stayed in the same slot");
        }
    }

    /// A tree three deep, drawn as the panel draws it:
    ///
    /// ```text
    ///   D  1        depth 0
    ///   B  2        depth 0
    ///   K  3        depth 2   -,
    ///   H  4        depth 1    | G's subtree
    ///   G  5        depth 0   -'
    ///   A  6        depth 0
    /// ```
    fn nested() -> Vec<Row> {
        display(&[(1, 0), (2, 0), (3, 2), (4, 1), (5, 0), (6, 0)], &[])
    }

    /// **One seam, four meanings.** The gap between `B` and `K` is where four
    /// different stacks all end, so the drop point alone cannot say which is meant —
    /// and the pointer's horizontal travel is what says it.
    ///
    /// This is the whole reason depth is a live part of the gesture rather than a
    /// consequence of it: without it the deepest three of these are unreachable by
    /// dragging, and a panel that can draw a place it cannot drop into is a panel that
    /// has to grow a second control to reach it.
    #[test]
    fn one_seam_can_mean_four_different_stacks() {
        let rows = nested();
        let landed = |steps: f32| {
            let l = landing(&rows, &drag(&rows, 1, steps * INDENT as f32, STEP)).expect("resolves");
            (l.depth, l.carrier, l.at)
        };
        assert_eq!(
            landed(0.0),
            (0, None, Place::Above(LayerId::solo(5))),
            "the document's own stack, above the group"
        );
        assert_eq!(
            landed(1.0),
            (1, Some(LayerId::solo(5)), Place::Above(LayerId::solo(4))),
            "inside G, above H"
        );
        assert_eq!(
            landed(2.0),
            (2, Some(LayerId::solo(4)), Place::Above(LayerId::solo(3))),
            "inside H, above K"
        );
        assert_eq!(
            landed(3.0),
            (3, Some(LayerId::solo(3)), Place::Bottom),
            "and one deeper still is K carrying it"
        );
    }

    /// A group travels as one: the block is the whole subtree, which is contiguous in
    /// the panel because a base is drawn under exactly the rows it carries.
    #[test]
    fn dragging_a_group_lifts_everything_it_carries() {
        let rows = nested();
        let l = landing(&rows, &drag(&rows, 5, 0.0, STEP)).expect("resolves");
        assert_eq!(l.slide.block, (2, 4), "K, H and G move together");
        // Three rows and the gaps between them, so the slot opened is the size of what
        // is going into it.
        assert!(
            (l.slide.step - 3.0 * STEP).abs() < 0.01,
            "step {}",
            l.slide.step
        );
    }

    /// Nothing can be dropped into what it is carrying. Not checked at the drop —
    /// **ruled out by construction**: every place a landing can name is stated against
    /// a row that stays put, and no row of the travelling block is one. The engine
    /// declines a cycle anyway (§14.8), but a panel that offers a move the engine will
    /// silently refuse is a panel that lies.
    #[test]
    fn a_group_cannot_be_dropped_inside_itself() {
        let rows = nested();
        let block = [LayerId::solo(3), LayerId::solo(4), LayerId::solo(5)];
        for steps in 0..8 {
            for dy in [-3.0 * STEP, -STEP, 0.0, STEP, 3.0 * STEP] {
                let d = drag(&rows, 5, steps as f32 * INDENT as f32, dy);
                let l = landing(&rows, &d).expect("resolves");
                assert!(
                    !l.carrier.is_some_and(|c| block.contains(&c)),
                    "landed inside its own subtree at {steps} indents, {dy}px"
                );
            }
        }
    }

    /// A drop into a folded group goes on **top** of what it carries rather than under
    /// it. The seam over a shut group stands for the whole subtree rather than for the
    /// foot of it, so that is where the layer will be when the fold opens — which the
    /// drop then does, so the answer is looked at rather than inferred.
    #[test]
    fn a_folded_group_takes_the_drop_on_top() {
        let rows = display(&[(1, 0), (2, 0), (3, 0), (4, 0)], &[3]);
        // One row down puts the seam directly over the shut group, and one indent
        // right is the depth that goes into it.
        let l = landing(&rows, &drag(&rows, 1, INDENT as f32, STEP)).expect("resolves");
        assert_eq!((l.carrier, l.at), (Some(LayerId::solo(3)), Place::Top));
    }

    /// A row the drag has no box for resolves to nothing at all: the panel draws
    /// itself at rest and the release commits nothing. That is what a document
    /// changing under the pointer looks like — a peer's edit landing mid-drag (§17) —
    /// and the alternative is acting on geometry that describes a tree that is gone.
    #[test]
    fn a_row_that_was_not_measured_abandons_the_drag() {
        let rows = flat();
        let d = drag_of(boxes(&rows, Some(2)), &rows, 1, 0.0, STEP);
        assert!(landing(&rows, &d).is_none());
    }
}
