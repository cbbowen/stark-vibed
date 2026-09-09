//! What the Layers panel draws, worked out from the flat list the engine
//! projects — and what a drop into it means (§14.6, §14.8).
//!
//! **Nothing here renders anything.** It is the panel's arithmetic: which rows are
//! folded away, what a row's own Carry and Release would do, and — given a drag —
//! where the block would land, at what depth, and what that spells as a
//! [`MoveLayer`](stark_engine::command::DocCommand::MoveLayer). The rsx! that draws
//! it is `stark-dioxus-frontend`'s `panels::layer`, 900 lines of chrome this used to be
//! buried in.
//!
//! It is a file of its own for [`transform`](crate::transform)'s reason: this is the part of the
//! panel that can be *tested*, and a dozen of the crate's tests are here — the ones
//! that can say a drop between two rows means one place in the tree rather than
//! another. Inside 1500 lines of markup they were the tests hardest to find and
//! the code most likely to be skimmed as layout.
//!
//! The column arithmetic underneath — which rows yield, and by how much — is not
//! here either. That is [`reorder`](crate::reorder), shared with the guide list; what is
//! here is the part only a *tree* has.

use std::borrow::Cow;
use std::collections::HashSet;

use crate::reorder::{Grab, Motion, Slide};
use stark_engine::LayerInfo;
use stark_engine::command::DocCommand;
use stark_model::document::LayerId;
use stark_model::document::Place;

/// How far one level of membership indents a row, in pixels. Named because three
/// things are measured in it: the row's own offset, the slot the indent leaves empty
/// to its left, which is where Release sits, and — since a drag can change a row's
/// depth as well as its place — how far sideways the pointer must travel to mean one
/// level of it.
pub const INDENT: usize = 14;

/// A row as the panel draws it: the layer, plus what its neighbours in the flat
/// list say about it that the layer alone cannot.
#[derive(Clone, PartialEq)]
pub struct Row {
    pub info: LayerInfo,
    /// Collapsed away under a group whose triangle is shut.
    pub hidden: bool,
    /// Shut, **and it is a group**. A frontend keeps its folded ids in a set nothing
    /// prunes, so a layer that was a group, was folded, then had its last member
    /// released is still in that set — and a row that reported the fold anyway would
    /// leave every use site to remember `&& is_group`. It is spent here instead.
    pub collapsed: bool,
    /// The layer directly below this one *in its own stack* — what [`carry`] would
    /// put it on, and the layer a clip would be bounded by (§14.4). `None` at the
    /// foot of a stack, where there is nothing to be carried by.
    ///
    /// [`carry`]: Self::carry
    carry_onto: Option<LayerId>,
    /// The group this layer is in, and what carries *that* — between them, where
    /// [`release`] would put it: out of the group and directly above it. `None` for a
    /// layer that is not in a group, which is the only state Release has nothing to
    /// say about.
    ///
    /// [`release`]: Self::release
    release_to: Option<(LayerId, Option<LayerId>)>,
    /// Whether Remove would leave a document behind. Removing a group takes what it
    /// carries with it (§14.2), so the floor is not "more than one row" but
    /// "something would be left" — which for a row deep in a group is nearly always
    /// true, and for the sole top-level stack never is.
    pub removable: bool,
}

impl Row {
    /// Put this layer on the one below it in its own stack, so the two become a
    /// group (§14.2, §14.8). `None` at the foot of a stack, and where the row below
    /// is a filter, which never carries (§21.2).
    ///
    /// The command whole rather than the layer to carry onto: what a frontend does
    /// with the answer is spell §14.2's rule, and two frontends spelling it apart is
    /// two apps. A row with nothing to carry onto simply has no button.
    pub fn carry(&self) -> Option<DocCommand> {
        Some(DocCommand::MoveLayer {
            id: self.info.id,
            carrier: Some(self.carry_onto?),
            at: Place::Top,
        })
    }

    /// Lift this layer out of the group it is in, to directly above it (§14.2).
    /// `None` for a layer that is in no group.
    pub fn release(&self) -> Option<DocCommand> {
        let (group, outer) = self.release_to?;
        Some(DocCommand::MoveLayer {
            id: self.info.id,
            carrier: outer,
            at: Place::Above(group),
        })
    }

    /// Whether the blend picker has anything to say about this layer.
    ///
    /// Blend and clip go inert together with nothing beneath them (§14.4.3), and
    /// **they part on a filter** — the one row where a shared condition would be
    /// wrong (§21.4). A mode describes how a *source* meets a backdrop and a filter
    /// has no source; a clip says where the layer may land, a question a filter still
    /// answers by being confined to the coverage it read.
    ///
    /// They also read *different predicates* for the half they share, which is the
    /// second thing the split brings out. A blend is positional, so it takes
    /// [`has_backdrop`]. A filter's clip is inert exactly where the **filter** is, so
    /// it takes [`has_underlay`] — the renderer's own answer (§21.2), which counts a
    /// carrier's base as beneath what it carries. `has_backdrop` says no there, and
    /// that arrangement is "filter just this layer": the chip would be dead in the one
    /// place it is reached for most.
    ///
    /// [`has_backdrop`]: LayerInfo::has_backdrop
    /// [`has_underlay`]: LayerInfo::has_underlay
    pub fn blend_inert(&self) -> bool {
        !self.info.has_backdrop || self.info.filter.is_some()
    }

    /// Whether the clip chip has anything to say about this layer — see
    /// [`blend_inert`](Self::blend_inert) for why the two are not one answer.
    pub fn clip_inert(&self) -> bool {
        match self.info.filter {
            Some(_) => !self.info.has_underlay,
            None => !self.info.has_backdrop,
        }
    }
}

/// What to call a layer that has never been named: which one it was, or what
/// it *is* when that says more (§15.7 — there is only ever one frame,
/// so numbering it would be noise).
///
/// The number is [`LayerInfo::number`] — minted into the action that created the
/// layer and frozen there, so it counts from 1 and nothing an artist does to another
/// row moves it. It was the layer id's own Lamport clock, which is unique across peers
/// for free and reads as "Layer 108" by the time a session has painted, since every
/// action moves that clock. Two peers adding a layer at once can now land on the same
/// number, and it costs nothing: nothing is looked up by a label (a row is keyed by
/// its id), and the pair is one rename apart.
///
/// Here rather than in the engine because it is a way of *presenting* a stack,
/// not a fact about the document — which is exactly why an unnamed layer stores no
/// name (see [`LayerInfo::name`]). A layer the author has named shows that name,
/// frame or not.
///
/// The word alone, with no mark in it. A `\u{25F1}` leading an unnamed frame's label
/// would stand in for a glyph the set already has (`icons::FRAME`, which the frame bar
/// wears), and putting it in the *string* costs twice over: this label is also a
/// rename field's placeholder, so opening the field on a frame would show a corner mark
/// inside a text box. The row draws the glyph, which leaves the placeholder a name.
///
/// Borrowed wherever there is something to borrow, which is every arm but one: four
/// of the five answers are already text somebody else owns, and only "Layer 7" has to
/// be built. This runs once per row per frame on both frontends.
pub fn layer_label(info: &LayerInfo) -> Cow<'_, str> {
    match (&info.name, info.matte.as_ref(), info.filter.as_ref()) {
        (Some(name), ..) => Cow::Borrowed(&**name),
        // The two kinds of matte, told apart by the one thing that differs:
        // a frame is defined against a rect, a background against none (§15.5).
        (None, Some(m), _) if m.rect.is_some() => Cow::Borrowed("Frame"),
        (None, Some(_), _) => Cow::Borrowed("Background"),
        // The *filter's* own name rather than the word "Filter" (§21.6): unlike a
        // frame, of which there is only ever one kind, which filter this is is the
        // first thing to know about the row — and a stack of three rows all reading
        // "Filter" would say nothing at all.
        (None, _, Some(f)) => Cow::Borrowed(f.label()),
        // Every paint layer is minted with a number, so the second arm is dead —
        // written out rather than a panic, because a row's label is not worth taking
        // the panel down over.
        (None, None, None) => match info.number {
            Some(n) => Cow::Owned(format!("Layer {n}")),
            None => Cow::Borrowed("Layer"),
        },
    }
}

/// The rows as a panel **shows** them: top of the document first, with whatever a
/// shut group folded away left out.
///
/// Not the order [`rows`] hands back, which is the engine's — bottom-to-top, a
/// group's base *before* what it carries — and the difference is not cosmetic.
/// [`landing`]'s reach-back over a dragged block, its depth bounds and its walk up to
/// an ancestor are all written against *this* order. While both lists were `&[Row]`,
/// the only thing between a correct drag and a silently inverted one was a line each
/// frontend wrote for itself — one had written it, the other had not yet grown a drag
/// to need it. A type is what makes the turn unskippable.
///
/// Borrows rather than clones. A `LayerInfo` carries a name, a filter with a
/// gradient's stops and a matte with another, and this list is rebuilt every frame.
pub struct Display<'a>(Vec<&'a Row>);

/// The displayed rows of `rows`, in the order a panel draws them — see [`Display`].
pub fn display(rows: &[Row]) -> Display<'_> {
    Display(rows.iter().rev().filter(|r| !r.hidden).collect())
}

impl<'a> std::ops::Deref for Display<'a> {
    type Target = [&'a Row];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// What a drag would commit, resolved against the rows as they stand now.
///
/// The column arithmetic — which rows yield, and by how much — is [`Slide`], shared
/// with the other roster that is dragged this way (`panels::reorder`). What is here
/// is the part only a *tree* has: which depth the hand is asking for, and what that
/// spells as a place in the document.
#[derive(Clone, Copy, PartialEq)]
pub struct Landing {
    /// Which rows travel and what the rest do about it. The block is the dragged row
    /// and everything it carries — contiguous in the panel, because a base is drawn
    /// under exactly the rows it carries and nothing may come between.
    pub slide: Slide,
    /// The depth it lands at, which the pointer's *horizontal* travel chooses among
    /// the depths this seam can express.
    pub depth: usize,
    /// Where the block is drawn while in flight.
    pub shift: (f32, f32),
    /// Whose stack the drop lands in — public because the panel asks a question of
    /// it that is not the move: which row to *mark* as the one taking the drop, and
    /// which fold to open so a layer dropped into a shut group does not vanish.
    pub carrier: Option<LayerId>,
    /// Where in that stack. Read through [`move_layer`](Self::move_layer), which is
    /// the only thing this and `carrier` are together.
    at: Place,
    /// Whether that move would change anything. A drag that ends where it began must
    /// not spend an undo step saying so — in a tree that means the same slot *and*
    /// the same depth, since one seam can hold several depths.
    pub inert: bool,
}

impl Landing {
    /// How to draw the row at display index `i`.
    pub fn motion(&self, i: usize) -> Motion {
        self.slide.motion(i, self.shift)
    }

    /// What dropping `id` here commits (§14.8) — the whole command, since `carrier`
    /// and `at` are literally [`MoveLayer`]'s own fields minus the id, and a frontend
    /// reassembling them is a frontend that can reassemble them differently.
    ///
    /// [`MoveLayer`]: DocCommand::MoveLayer
    pub fn move_layer(&self, id: LayerId) -> DocCommand {
        DocCommand::MoveLayer {
            id,
            carrier: self.carrier,
            at: self.at,
        }
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
pub fn landing(display: &Display<'_>, drag: &Grab) -> Option<Landing> {
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

    let above = gap.checked_sub(1).map(|k| display[rest[k]]);
    let below = rest.get(gap).map(|&k| display[k]);
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
            if b.collapsed {
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
                let outer = *display.iter().find(|r| r.info.id == up)?;
                // A carrier chain that does not climb is not a tree, and this walk is
                // over a list the *caller* supplies: a peer's edit landing half-applied
                // (§17), or a fixture. Refusing it is what makes the loop finite —
                // strictly decreasing depth bounds it by the deepest row there is.
                if outer.info.depth >= anchor.info.depth {
                    return None;
                }
                anchor = outer;
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
///
/// One `Vec` answers both, indexed by depth — a stack of cursors, which is how the
/// walk that *builds* this list keeps its place (`observe`'s `Cursor`). Entry `d` is
/// the last row seen at depth `d`, **truncated on
/// the way out** of a subtree, and that truncation is what makes it the current row's
/// ancestor chain as well as its stack: the sibling below is entry `depth`, and what
/// carries this row's group is entry `depth - 2`. Leaving a stack means passing
/// through a shallower row, which drops everything deeper than it.
///
/// So the whole projection is one pass with two hash maps and a per-row scan of the
/// list fewer than it was — which is worth the paragraph because it runs on both
/// frontends' render path, once a frame.
pub fn rows(layers: &[LayerInfo], collapsed: &HashSet<LayerId>) -> Vec<Row> {
    let mut out = Vec::with_capacity(layers.len());
    let mut shut_at: Option<usize> = None;
    // The last row seen at each depth, and whether it is a filter — which the Carry
    // answer below has to know, since a filter never carries (§21.2) and a button
    // that spelled a move the engine refuses would be a lie.
    let mut seen: Vec<(LayerId, bool)> = Vec::new();
    // Removing a row takes everything it carries with it, so the only row that would
    // empty the document is the base of a *sole* top-level stack — which is the first
    // row, the engine's order putting a base before what it carries.
    let roots = layers.iter().filter(|l| l.depth == 0).count();
    for (i, info) in layers.iter().enumerate() {
        if shut_at.is_some_and(|d| info.depth <= d) {
            shut_at = None;
        }
        let hidden = shut_at.is_some();
        let collapsed = collapsed.contains(&info.id) && info.is_group;
        if !hidden && collapsed {
            shut_at = Some(info.depth);
        }
        let carry_onto = seen
            .get(info.depth)
            .and_then(|&(id, filter)| (!filter).then_some(id));
        let release_to = info.carrier.map(|group| {
            // The group sits at `depth - 1`, so what carries *it* is one further down.
            // `get` rather than an index: this reads a list the engine supplies, and a
            // panel is not the place to report a malformed one with a panic.
            let outer = info
                .depth
                .checked_sub(2)
                .and_then(|d| seen.get(d))
                .map(|&(id, _)| id);
            (group, outer)
        });
        // After the reads, so neither answer is the layer itself, and the truncation
        // is what drops a subtree this row has just walked back out of. A collapsed
        // group's members are still in their stack, so this happens for hidden rows
        // too.
        seen.truncate(info.depth);
        seen.push((info.id, info.filter.is_some()));
        out.push(Row {
            info: info.clone(),
            hidden,
            collapsed,
            carry_onto,
            release_to,
            removable: !(i == 0 && roots == 1),
        });
    }
    out
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
            // Its own id, so a label in a failure message says which row it came from.
            number: Some(id as u32),
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
    fn drawn(spec: &[(u64, usize)], shut: &[u64]) -> Vec<Row> {
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

    /// The fixture's rows as the [`Display`] [`landing`] takes.
    ///
    /// The identity rather than [`display`]'s reversal, because [`drawn`] writes them
    /// in display order already — which is what lets a spec be read as the picture.
    /// That [`display`] produces this order from the engine's is its own test below.
    fn shown(rows: &[Row]) -> Display<'_> {
        Display(rows.iter().collect())
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
        drawn(&[(1, 0), (2, 0), (3, 0)], &[])
    }

    /// Dragging a row past its neighbour reorders it within its own stack — the move
    /// the panel had no control for at all, since Carry and Release only ever changed
    /// *whose* stack a layer is in.
    #[test]
    fn a_drag_down_lands_above_the_row_it_passed() {
        let rows = flat();
        let l = landing(&shown(&rows), &drag(&rows, 1, 0.0, STEP)).expect("resolves");
        assert_eq!((l.carrier, l.at), (None, Place::Above(LayerId::solo(3))));
        assert!(!l.inert);
    }

    /// The foot of a stack — the place `Place::Above` cannot name, and the reason the
    /// anchor grew a third state at all.
    #[test]
    fn a_drag_to_the_bottom_lands_under_everything() {
        let rows = flat();
        let l = landing(&shown(&rows), &drag(&rows, 1, 0.0, 3.0 * STEP)).expect("resolves");
        assert_eq!((l.carrier, l.at), (None, Place::Bottom));
    }

    /// Sideways is what nests. The same drop point one indent to the right is not a
    /// place in the same stack — it is the row below the seam carrying the layer,
    /// which is exactly what the Carry button spells (§14.2).
    #[test]
    fn a_drag_to_the_right_is_what_carries() {
        let rows = flat();
        let straight = landing(&shown(&rows), &drag(&rows, 1, 0.0, STEP)).expect("resolves");
        let over = landing(&shown(&rows), &drag(&rows, 1, INDENT as f32, STEP)).expect("resolves");
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
        let over = landing(&shown(&rows), &drag(&rows, 1, INDENT as f32, STEP)).expect("resolves");
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
            let l = landing(&shown(&rows), &drag(&rows, 2, 0.0, dy)).expect("resolves");
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
        drawn(&[(1, 0), (2, 0), (3, 2), (4, 1), (5, 0), (6, 0)], &[])
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
            let l = landing(&shown(&rows), &drag(&rows, 1, steps * INDENT as f32, STEP))
                .expect("resolves");
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
        let l = landing(&shown(&rows), &drag(&rows, 5, 0.0, STEP)).expect("resolves");
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
                let l = landing(&shown(&rows), &d).expect("resolves");
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
        let rows = drawn(&[(1, 0), (2, 0), (3, 0), (4, 0)], &[3]);
        // One row down puts the seam directly over the shut group, and one indent
        // right is the depth that goes into it.
        let l = landing(&shown(&rows), &drag(&rows, 1, INDENT as f32, STEP)).expect("resolves");
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
        assert!(landing(&shown(&rows), &d).is_none());
    }

    /// The four things a row can be called, and the order they win in. What the
    /// *number* is is the engine's question (`DocState::next_layer_number`); what is
    /// here is that a row wears the one it was minted with, and that a kind speaks
    /// over it while a name speaks over everything.
    #[test]
    fn a_row_is_called_its_name_then_its_kind_then_its_number() {
        use stark_engine::MatteInfo;
        use stark_model::color::Srgb;
        use stark_model::document::{ColorAdjust, Filter, Parcel};
        use stark_model::geom::Vec2;

        let plain = info(7, 0, None, false);
        assert_eq!(layer_label(&plain), "Layer 7");

        let named = LayerInfo {
            name: Some("Sky".into()),
            ..plain.clone()
        };
        assert_eq!(layer_label(&named), "Sky", "a name speaks over the number");

        let frame = LayerInfo {
            matte: Some(MatteInfo {
                rect: Some((Vec2::ZERO, Vec2::new(64.0, 64.0))),
                paint: Parcel::Solid(Srgb::BLACK),
            }),
            number: None,
            ..plain.clone()
        };
        assert_eq!(layer_label(&frame), "Frame");

        let filter = LayerInfo {
            filter: Some(Filter::Color(ColorAdjust::NEUTRAL)),
            number: None,
            ..plain
        };
        assert_eq!(
            layer_label(&filter),
            Filter::Color(ColorAdjust::NEUTRAL).label()
        );
    }

    /// A document in the order the engine keeps it, for the projection's own tests:
    /// bottom-to-top, a group's base before what it carries. The reverse of what
    /// [`drawn`] takes, deliberately — that difference is what [`display`] is.
    fn projected(spec: &[(u64, usize, Option<u64>, bool)]) -> Vec<LayerInfo> {
        spec.iter()
            .map(|&(id, depth, carrier, group)| info(id, depth, carrier, group))
            .collect()
    }

    /// [`display`] is the one place the panel's order is stated: top of the document
    /// first, folded rows gone. The engine's order is the other way round and a base
    /// comes *before* what it carries, so a list handed to [`landing`] unturned is not
    /// a wrong drop — it is the mirror of the right one, and the whole gesture is
    /// stated against it.
    #[test]
    fn the_displayed_order_is_the_engines_turned_over() {
        // A group `1` carrying `2`, then a root `3` above them both.
        let layers = projected(&[
            (1, 0, None, true),
            (2, 1, Some(1), false),
            (3, 0, None, false),
        ]);
        let rows = rows(&layers, &HashSet::from([LayerId::solo(1)]));
        let shown: Vec<u64> = display(&rows)
            .iter()
            .map(|r| r.info.id.action.lamport)
            .collect();
        assert_eq!(
            shown,
            [3, 1],
            "top of the document first, the fold's member gone"
        );
    }

    /// A carrier chain that does not climb is refused rather than walked forever.
    /// The list is the *caller's* — a peer's edit can land half-applied mid-drag
    /// (§17) — so the loop cannot be left to trust that it terminates.
    #[test]
    fn a_carrier_chain_that_does_not_climb_lands_nothing() {
        let mut rows = drawn(&[(1, 0), (2, 0), (3, 1), (4, 0)], &[]);
        // Row 3 sits at depth 1 and says row 3 carries it — a cycle of one, which is
        // a tree no walk up can leave.
        rows[2].info.carrier = Some(LayerId::solo(3));
        // Straight down onto the seam over row 3, at the root's own depth: the arm
        // that walks up to an ancestor.
        let d = drag(&rows, 1, 0.0, STEP);
        assert!(landing(&shown(&rows), &d).is_none());
    }

    /// **A fold is a group's.** A frontend's set of shut ids is not pruned when a
    /// group's last member is released, so the id of a layer that is no longer a group
    /// is still in it — and a row that reported that fold would hide whatever it sat
    /// under and take a drop on top of itself.
    #[test]
    fn a_stale_fold_is_not_a_fold() {
        // Two roots, neither carrying anything, and both ids shut.
        let layers = projected(&[(1, 0, None, false), (2, 0, None, false)]);
        let shut = HashSet::from([LayerId::solo(1), LayerId::solo(2)]);
        let rows = rows(&layers, &shut);
        assert!(rows.iter().all(|r| !r.collapsed), "neither is a group");
        assert!(
            rows.iter().all(|r| !r.hidden),
            "so neither folds anything away"
        );
    }

    /// Blend and clip part on a filter, and the halves they share read different
    /// predicates — the case that is easy to get backwards, since three of the four
    /// answers agree (§21.4).
    #[test]
    fn a_filters_clip_outlives_its_blend() {
        use stark_model::document::{ColorAdjust, Filter};

        let row = |filter: bool, backdrop: bool, underlay: bool| Row {
            info: LayerInfo {
                filter: filter.then_some(Filter::Color(ColorAdjust::NEUTRAL)),
                has_backdrop: backdrop,
                has_underlay: underlay,
                ..info(1, 0, None, false)
            },
            hidden: false,
            collapsed: false,
            carry_onto: None,
            release_to: None,
            removable: true,
        };

        // A paint layer: one answer, twice, off `has_backdrop`.
        let paint = row(false, true, true);
        assert!(!paint.blend_inert() && !paint.clip_inert());
        let floor = row(false, false, false);
        assert!(floor.blend_inert() && floor.clip_inert());

        // A filter: the mode has no source to describe wherever it sits, and the clip
        // follows the *renderer* instead — which counts a carrier's base as beneath
        // what it carries, so "filter just this layer" keeps its chip.
        let filter = row(true, true, true);
        assert!(filter.blend_inert(), "a filter has no source to blend");
        assert!(!filter.clip_inert(), "but it still says where it may land");
        let alone = row(true, false, true);
        assert!(
            !alone.clip_inert(),
            "has_backdrop would have killed the chip in the one place it is used most"
        );
        let empty = row(true, true, false);
        assert!(empty.clip_inert(), "nothing under it in its own stack");
    }

    /// Carry and Release come out of the row whole. Each was spelled twice — once per
    /// frontend — out of ingredients the row handed over untyped, and §14.2's rule is
    /// not a thing two apps get to answer separately.
    #[test]
    fn carry_and_release_are_the_move_they_mean() {
        // A group `1` carrying `2` and `3`, with a root `4` above the lot.
        let layers = projected(&[
            (1, 0, None, true),
            (2, 1, Some(1), false),
            (3, 1, Some(1), false),
            (4, 0, None, false),
        ]);
        let rows = rows(&layers, &HashSet::new());

        // `3` sits on `2` inside the group; carrying it makes the two a group of
        // their own, on top of what `2` already carries (nothing).
        assert!(matches!(
            rows[2].carry(),
            Some(DocCommand::MoveLayer { id, carrier: Some(c), at: Place::Top })
                if id == LayerId::solo(3) && c == LayerId::solo(2)
        ));
        // Releasing it puts it in the document's own stack, directly above the group.
        assert!(matches!(
            rows[2].release(),
            Some(DocCommand::MoveLayer { id, carrier: None, at: Place::Above(g) })
                if id == LayerId::solo(3) && g == LayerId::solo(1)
        ));
        // The foot of the document's stack has nothing under it to be carried by, and
        // nothing to be released from.
        assert!(rows[0].carry().is_none() && rows[0].release().is_none());
        // And the bottom of a group is in one without sitting on anything.
        assert!(rows[1].carry().is_none() && rows[1].release().is_some());
    }

    /// What Remove would leave behind, over a document whose only root is a group:
    /// the base takes everything with it, and every row inside it leaves that base.
    #[test]
    fn only_the_base_of_a_sole_stack_refuses_to_go() {
        let layers = projected(&[(1, 0, None, true), (2, 1, Some(1), false)]);
        let sole = rows(&layers, &HashSet::new());
        assert!(!sole[0].removable, "removing it would empty the document");
        assert!(sole[1].removable);

        // A second root, and the first becomes removable — what is left is a document.
        let layers = projected(&[
            (1, 0, None, true),
            (2, 1, Some(1), false),
            (3, 0, None, false),
        ]);
        assert!(rows(&layers, &HashSet::new()).iter().all(|r| r.removable));
    }
}
