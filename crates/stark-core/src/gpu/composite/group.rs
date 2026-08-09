//! What a render is asked to draw: the layer stack, cut into blend groups (§14.7,
//! §18.0.4).
//!
//! Pure description — no GPU anywhere in this file. The stack arrives here already
//! decided by the document, and the only judgement it makes is [`CompositeGroup::stack`]'s:
//! whether a group can tell itself apart from no group at all.

use crate::document::BlendMode;
use crate::geom::TileCoord;
use crate::gpu::tile::TilePairHandle;

/// A matte layer's draw parameters (§15.4).
#[derive(Copy, Clone, Debug)]
pub struct MatteDraw {
    /// The region's rect in canvas px: `min.xy, max.xy`. For a frame this is the
    /// *hole* — the fill covers everything outside it.
    pub rect: [f32; 4],
    /// Fill color in the document's working color space.
    pub channels: [f32; 4],
    /// The same colour's **residual** in `.xyz` (§6.7); `.w` unused. Zero in a space
    /// that has none, which is the true value rather than a stand-in.
    pub resid: [f32; 4],
    /// The layer's opacity.
    pub opacity: f32,
}

/// One item of compositing pass A, in bottom-to-top stack order.
///
/// An ordered list rather than a flat tile array because a matte composites at
/// its own place in the stack — a frame over the painting, a ground under it
/// (§15.4.4). Tiles already cost one draw each (each needs its own
/// bind group), so interleaving mattes adds no per-tile overhead.
#[derive(Clone)]
pub enum CompositeItem {
    Tile {
        coord: TileCoord,
        handle: TilePairHandle,
        opacity: f32,
    },
    Matte(MatteDraw),
}

/// One **blend group** of pass A: something that composites on its own, and how
/// its result merges into everything below it (§18.0.4,
/// §14.7).
///
/// A group is defined against *what is underneath it* — which means it has to be
/// composited alone, on nothing, before it can be merged. That is the per-layer
/// isolation §6.3 names as the prerequisite for richer modes, and layer
/// groups are the same investment recursed: [`GroupContent::Stack`] is a group
/// whose members are themselves groups.
#[derive(Clone)]
pub struct CompositeGroup {
    pub blend: BlendMode,
    /// Clip to the coverage of what this group composites onto
    /// (§14.4). Costs the same isolation a blend mode does, and for the same reason:
    /// the merge has to *read* the backdrop's alpha.
    pub clip: bool,
    /// Applied to this group's whole composited result at the merge, not to its
    /// members one at a time.
    ///
    /// That distinction is only visible on a [`Stack`](GroupContent::Stack),
    /// whose members overlap — a leaf layer's tiles do not, so its opacity rides
    /// on [`CompositeItem::Tile`] and this stays 1.0 for it. Two granularities of
    /// the same fact, and the cheaper one is used wherever it is equivalent.
    pub opacity: f32,
    pub content: GroupContent,
}

/// What a [`CompositeGroup`] is made of — and the fast path, expressed as a
/// shape rather than as a condition inside the encoder.
#[derive(Clone)]
pub enum GroupContent {
    /// A run of the stack that composites internally under plain premultiplied
    /// "over", with **no isolation**.
    ///
    /// Consecutive `Normal`, unclipped layers carrying nothing compose correctly
    /// against each other *and* against the accumulator, so a document that uses
    /// no modes, no clipping and no groups is a single `Run` and costs exactly
    /// what the flat tile list cost before any of this existed.
    Run(Vec<CompositeItem>),
    /// Members composited bottom-to-top, each merging into the one below through
    /// its own blend mode and clip — a **layer group** (§14.2).
    ///
    /// The builder collapses a `Stack` that could not tell itself apart from a
    /// `Run` into one, which is what makes "organization is free" structural
    /// rather than a promise (§14.7 rule 2).
    Stack(Vec<CompositeGroup>),
}

impl CompositeGroup {
    /// A run of drawables that merges outward through `blend` and `clip`.
    pub fn run(blend: BlendMode, clip: bool, items: Vec<CompositeItem>) -> Self {
        Self {
            blend,
            clip,
            opacity: 1.0,
            content: GroupContent::Run(items),
        }
    }

    /// A group of `members` — **collapsed into a plain [`Run`](GroupContent::Run)
    /// when nothing about it could tell itself apart from one**
    /// (§14.7 rule 2).
    ///
    /// This is where "organization is free" is made structural rather than
    /// promised. A group that merges normally, unclipped, at full opacity, and
    /// whose every member draws directly, changes no blending scope: its members
    /// were composing against everything below them under `over` already, and
    /// isolating them would produce the same pixels through two extra render
    /// passes per member. So it produces the identical draw list to no group at
    /// all — which is the property the golden test pins, and the answer to
    /// "grouping my layers changed my painting".
    ///
    /// The condition cannot be relaxed to "the group itself is normal": a member
    /// with a mode of its own *does* blend against a different backdrop once
    /// isolated, and that difference is the feature (§14.5).
    pub fn stack(blend: BlendMode, clip: bool, opacity: f32, members: Vec<Self>) -> Self {
        let free = blend.is_normal() && !clip && opacity >= 1.0;
        if free && members.iter().all(|m| m.as_direct_run().is_some()) {
            let items = members
                .into_iter()
                .flat_map(|m| match m.content {
                    GroupContent::Run(items) => items,
                    // The guard on the line above is this arm's proof: a direct
                    // group is a `Run`. It is the one place the implication is
                    // still asserted rather than carried by a return type, because
                    // this consumes the members and `as_direct_run` borrows them.
                    GroupContent::Stack(_) => {
                        unreachable!("a direct group is a Run by construction")
                    }
                })
                .collect();
            return Self::run(blend, clip, items);
        }
        Self {
            blend,
            clip,
            opacity,
            content: GroupContent::Stack(members),
        }
    }

    /// The items this group draws straight into the accumulator, or `None` if it
    /// needs isolating — an unclipped `Normal` `Run` at full opacity is the fast
    /// path, and everything else is a merge.
    ///
    /// **Returning the run is what deletes the invariant rather than checking it.**
    /// Three call sites used to ask a `bool` `is_direct` and then re-match the
    /// content: two behind an `unreachable!`, and one (the stack builder in
    /// `Engine`) behind an `if let` whose failure branch silently dropped the merge.
    /// Three places that could disagree about what "direct" implies, and the third
    /// would not even have said so.
    pub fn as_direct_run(&self) -> Option<&[CompositeItem]> {
        match &self.content {
            GroupContent::Run(items) if self.is_free() => Some(items),
            _ => None,
        }
    }

    /// [`Self::as_direct_run`] for a caller that appends to the run it gets back —
    /// the stack builder folding a group into the one below it.
    pub fn as_direct_run_mut(&mut self) -> Option<&mut Vec<CompositeItem>> {
        let free = self.is_free();
        match &mut self.content {
            GroupContent::Run(items) if free => Some(items),
            _ => None,
        }
    }

    /// Whether this group's *merge* is a no-op — normal, unclipped, opaque. Says
    /// nothing about its content, which is the other half of "direct".
    fn is_free(&self) -> bool {
        self.blend.is_normal() && !self.clip && self.opacity >= 1.0
    }

    /// How deep the isolation nests below this group: 0 for a `Run`, one more
    /// than its deepest member for a `Stack`. The scratch stack is sized by this
    /// (§14.7).
    fn depth(&self) -> usize {
        match &self.content {
            GroupContent::Run(_) => 0,
            GroupContent::Stack(members) => 1 + members.iter().map(Self::depth).max().unwrap_or(0),
        }
    }

    /// Every drawable in this group, in composite order — the flat streams pass A
    /// uploads (the draw loop walks the tree, but the instance buffers do not
    /// need to).
    pub(super) fn items(&self) -> Vec<&CompositeItem> {
        let mut out = Vec::new();
        fn walk<'a>(g: &'a CompositeGroup, out: &mut Vec<&'a CompositeItem>) {
            match &g.content {
                GroupContent::Run(items) => out.extend(items.iter()),
                GroupContent::Stack(members) => members.iter().for_each(|m| walk(m, out)),
            }
        }
        walk(self, &mut out);
        out
    }
}

/// How many scratch levels compositing `members` as one stack takes: none if every
/// member draws straight into the accumulator, else one for this stack plus however
/// many the deepest nested group below it needs.
pub(super) fn scratch_levels(members: &[CompositeGroup]) -> usize {
    if members.iter().all(|m| m.as_direct_run().is_some()) {
        return 0;
    }
    1 + members.iter().map(CompositeGroup::depth).max().unwrap_or(0)
}
