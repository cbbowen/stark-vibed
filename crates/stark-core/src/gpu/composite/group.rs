//! What a render is asked to draw: the layer stack, cut into blend groups (§14.7,
//! §18.0.4).
//!
//! Pure description — no GPU anywhere in this file. The stack arrives here already
//! decided by the document, and the only judgement it makes is [`CompositeGroup::stack`]'s:
//! whether a group can tell itself apart from no group at all.

use crate::document::{CompositeParams, Filter};
use crate::geom::TileCoord;
use crate::gpu::tile::TilePairHandle;

/// A matte layer's draw parameters (§15.4).
// No `Debug`: the mirrored `Ramp` (§6.10) derives only what `bytemuck`
// needs, and 544 bytes of stop lanes would be noise in a dump anyway.
#[derive(Clone)]
pub struct MatteDraw {
    /// The region's rect in canvas px: `min.xy, max.xy`. For a frame this is the
    /// *hole* — the fill covers everything outside it. Zeros, never read, when
    /// `flags` says the region is the whole plane.
    pub rect: [f32; 4],
    /// Region kind, as the shader reads it: 0 = outside `rect`, 1 = everything
    /// (§15.5).
    pub flags: f32,
    /// Fill color in the document's working color space. Zeros for a gradient
    /// matte, whose color comes from `ramp` per fragment.
    pub channels: [f32; 4],
    /// The same color's **residual** in `.xyz` (§6.7); `.w` unused. Zero in a space
    /// that has none, which is the true value rather than a stand-in.
    pub resid: [f32; 4],
    /// The layer's opacity.
    pub opacity: f32,
    /// A gradient paint's ramp (§22.4), already packed for the
    /// shader's per-matte uniform — stops in working-space channels, axis in
    /// canvas px. `None` for a solid matte, which binds the shared zeroed ramp.
    /// Boxed so the rare gradient does not carry its 544 bytes of stop lanes
    /// into every [`CompositeItem`] — a tile is the common item by thousands.
    pub ramp: Option<Box<stark_shaders::mirror::matte::Ramp>>,
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

impl CompositeItem {
    /// Draw this item at `opacity` instead of the one it was built with.
    ///
    /// **Private, with exactly one caller**: [`CompositeGroup::leaf`], which is where
    /// a layer's opacity is folded into items that do not overlap. Building the items
    /// already faded and letting a group apply its own on top is precisely the double
    /// fade §14.7 describes, so the only way to set an item's opacity is to go through
    /// the constructor that takes it off the group in the same breath.
    fn set_opacity(&mut self, opacity: f32) {
        match self {
            CompositeItem::Tile { opacity: o, .. } => *o = opacity,
            CompositeItem::Matte(m) => m.opacity = opacity,
        }
    }
}

/// A filter layer's pass parameters (§21).
///
/// Deliberately **not** a `Filter`: what the shader reads is a code and two uniform
/// lanes of floats, and which code a filter is numbered is a fact about
/// `filter_common.wesl` rather than about the document (the same split
/// [`blend_code`](super::blend::blend_code) makes for a blend mode). Flattening it
/// here also puts the layer's opacity where the pass wants it — as a strength — so
/// the encoder has one thing to write rather than two to remember to combine.
///
/// **Both of the two [`CompositeParams`] a filter reads live here**, for that reason
/// and no other. They are not on the enclosing [`CompositeGroup`] because a filter
/// merges nothing outward for them to describe (see [`GroupContent::Filter`]) — they
/// are settings of the pass, and this is the pass's description.
#[derive(Clone, Debug, PartialEq)]
pub struct FilterDraw {
    /// The filter's shader code — see `filter_common.wesl`.
    pub kind: u32,
    /// How much of the adjustment lands: the layer's opacity (§21.4).
    pub strength: f32,
    /// Whether the filter is confined to the coverage it read: the layer's clip
    /// (§21.4, §14.4). A **point** filter is already confined, so this changes
    /// nothing for one; the gather is what it bounds.
    pub clip: bool,
    /// The filter's own numbers, read according to `kind` — two lanes of four, so a
    /// kind with more than four has somewhere to put them without the bind group
    /// layout learning anything (`filter_common.wesl`).
    pub params: [f32; 4],
    pub params2: [f32; 4],
    /// The gradient map's ramp (§21.11), already in the shader's own terms: Oklab
    /// `(L, a, b)` and the stop's position per lane, `params[0]` of them live.
    /// `None` for every other kind — the uniform's lanes are zeroed at write.
    /// Boxed on [`MatteDraw::ramp`]'s argument: the 256 bytes ride only the draws
    /// that carry a ramp.
    pub stops: Option<Box<[[f32; 4]; MAX_MAP_STOPS]>>,
}

/// The stop lanes the filter uniform carries — the shader's literal `16`
/// (`filter_common.wesl`, §6.10's generator cannot resolve a named length), held
/// to the fitter's own bound here so the two cannot drift: every `Gradient` fits.
const MAX_MAP_STOPS: usize = crate::gradient::MAX_STOPS;

impl FilterDraw {
    /// The draw parameters for `filter` under the layer's own `params`.
    ///
    /// The whole [`CompositeParams`] rather than the two scalars it reads, because
    /// the rule about which of the three a filter has is a rule about the value as a
    /// whole (§21.4): the opacity becomes the strength, the clip travels as itself,
    /// and the blend is the one a filter never has — state declines to store one, so
    /// there is no mode here to drop.
    ///
    /// The one place a `Filter` becomes numbers, so the ABI is stated once — and
    /// each arm's code is the **mirrored** `FILTER_*` constant from
    /// `filter_common.wesl` (§6.10), never a literal. That is what makes the claim
    /// structural: a filter kind added to the document without a declaration in the
    /// shader has no constant to name here, so this match fails to compile rather
    /// than rendering as whichever kind happens to share its index. (The shader's
    /// `filtered()` still needs its arm; what cannot happen is the two agreeing on
    /// the *wrong* number.)
    pub fn new(filter: Filter, composite: CompositeParams) -> Self {
        // Named apart from the `params` lanes below, which are the *filter's* own
        // numbers and have nothing to do with the layer's.
        let (strength, clip) = (composite.opacity, composite.clip);
        match filter {
            Filter::Color(c) => Self {
                kind: stark_shaders::mirror::filter_common::FILTER_COLOR,
                strength,
                clip,
                params: [c.exposure, c.contrast, c.saturation, c.hue],
                params2: [c.tint[0], c.tint[1], 0.0, 0.0],
                stops: None,
            },
            // The spread and angle stay in canvas terms here: which *texels* they
            // become depends on the view, and this description deliberately has
            // none — the encoder derives the uniform's dispersion vector per frame
            // (§21.10).
            Filter::Chromatic(c) => Self {
                kind: stark_shaders::mirror::filter_common::FILTER_CHROMATIC,
                strength,
                clip,
                params: [c.spread, c.angle, 0.0, 0.0],
                params2: [0.0; 4],
                stops: None,
            },
            // The one conversion the ramp pays, here and once: stops are straight
            // sRGB in the document and Oklab in the pass, because Oklab is where
            // `Gradient::sample`'s interpolation — the thing the map *is* — is
            // defined (§21.11). A rampless map never reaches this constructor:
            // it is neutral, and the draw list dropped it (§21.3); the `None` arm
            // exists so the match stays total rather than as a path.
            Filter::GradientMap(g) => {
                let stops = g.as_ref().map(|g| {
                    let mut lanes = Box::new([[0.0f32; 4]; MAX_MAP_STOPS]);
                    for (lane, stop) in lanes.iter_mut().zip(g.stops()) {
                        let lab = crate::color::srgb_to_oklab([
                            stop.color[0],
                            stop.color[1],
                            stop.color[2],
                            1.0,
                        ]);
                        *lane = [lab[0], lab[1], lab[2], stop.t];
                    }
                    lanes
                });
                Self {
                    kind: stark_shaders::mirror::filter_common::FILTER_GRADIENT_MAP,
                    strength,
                    clip,
                    params: [
                        g.as_ref().map_or(0.0, |g| g.stops().len() as f32),
                        0.0,
                        0.0,
                        0.0,
                    ],
                    params2: [0.0; 4],
                    stops,
                }
            }
        }
    }
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
    /// How this group's composited result merges into everything below it — the
    /// **layer's own** params, applied once, to the whole (§14.4.3).
    ///
    /// The same [`CompositeParams`] the document holds, carried through unchanged,
    /// which is what stops the render path taking them apart and putting two of them
    /// back in different places. Clipping costs the same isolation a blend mode does,
    /// and for the same reason: the merge has to *read* the backdrop's alpha.
    ///
    /// [`Self::leaf`] is the one constructor that moves anything out of here, and it
    /// moves only the opacity, only onto items that provably do not overlap.
    pub params: CompositeParams,
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
    /// A **filter layer**: no content of its own, one pass that reads the
    /// accumulator this stack has built so far and writes it back adjusted (§21).
    ///
    /// The odd member of this enum, and the shape says why: a `Run` and a `Stack`
    /// are both *things to composite*, which a filter is not — it is a function of
    /// what has been composited already. That is exactly the blend pass's own
    /// relationship to the accumulator with the isolated source removed, so it takes
    /// the same ping-pong and the same scratch and costs the same one pass
    /// (`composite/filter.rs`).
    ///
    /// Its enclosing [`CompositeGroup`]'s params are [`CompositeParams::IDENTITY`],
    /// always: those describe a **merge**, and a filter has no source to merge. The
    /// two of them a filter does read are inside the draw instead — the opacity as
    /// the filter's strength, the clip as the bound on what the pass may write
    /// (§21.4) — which is why they are settings of a pass here rather than
    /// parameters of a merge that does not happen.
    Filter(FilterDraw),
}

impl CompositeGroup {
    /// A run of drawables that merges outward through `params`, **each item carrying
    /// whatever opacity it was built with**.
    ///
    /// The raw constructor, for a caller that already holds finished items — the
    /// collapse in [`Self::stack`], which concatenates runs whose opacities are
    /// already folded in. A caller turning *one layer* into a run wants
    /// [`Self::leaf`] instead, which is the difference between "these items are
    /// ready" and "fold this layer's opacity into them".
    pub fn run(params: CompositeParams, items: Vec<CompositeItem>) -> Self {
        Self {
            params,
            content: GroupContent::Run(items),
        }
    }

    /// **One layer's** drawables, merging outward through `params` — the constructor
    /// every layer goes through, and the only place opacity is ever folded into an
    /// item.
    ///
    /// The fold is what keeps a faded layer on the fast path: `params.opacity` on the
    /// group would make it non-direct and cost two render passes, while a layer's
    /// tiles do not overlap, so scaling each of them is identical to scaling the
    /// composited layer. Two granularities of one fact, and the cheaper one is taken
    /// wherever it is equivalent.
    ///
    /// **Equivalent, and therefore exclusive.** The opacity comes *off* the params on
    /// the way past, so what is folded is never also applied at a merge. That is not
    /// a nicety: it is the bug this constructor exists to make unrepresentable. A
    /// group's base composites with [`CompositeParams::IDENTITY`] — the layer's own
    /// params belong to the group as a whole — and for as long as the base's items
    /// were tagged by the item builder instead, a base at 0.5 drew its paint at 0.25
    /// while everything it carried drew at 0.5 (§14.7).
    pub fn leaf(params: CompositeParams, mut items: Vec<CompositeItem>) -> Self {
        for item in &mut items {
            item.set_opacity(params.opacity);
        }
        Self::run(
            CompositeParams {
                opacity: 1.0,
                ..params
            },
            items,
        )
    }

    /// A filter layer's pass, which merges nothing and is merged through nothing
    /// (§21) — see [`GroupContent::Filter`] for why the params are fixed rather than
    /// taken from the layer, and [`FilterDraw`] for where the two the layer really
    /// does state have gone.
    pub fn filter(draw: FilterDraw) -> Self {
        Self {
            params: CompositeParams::IDENTITY,
            content: GroupContent::Filter(draw),
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
    pub fn stack(params: CompositeParams, members: Vec<Self>) -> Self {
        if params.is_free() && members.iter().all(|m| m.as_direct_run().is_some()) {
            let items = members
                .into_iter()
                .flat_map(|m| match m.content {
                    GroupContent::Run(items) => items,
                    // The guard on the line above is this arm's proof: a direct
                    // group is a `Run`. It is the one place the implication is
                    // still asserted rather than carried by a return type, because
                    // this consumes the members and `as_direct_run` borrows them.
                    GroupContent::Stack(_) | GroupContent::Filter(_) => {
                        unreachable!("a direct group is a Run by construction")
                    }
                })
                .collect();
            // `run`, not `leaf`: these items already carry their own layers' folded
            // opacities, and re-folding this group's — which the guard above has just
            // proved is 1.0 — would overwrite every one of them.
            return Self::run(params, items);
        }
        Self {
            params,
            content: GroupContent::Stack(members),
        }
    }

    /// The items this group draws straight into the accumulator, or `None` if it
    /// needs isolating — an unclipped `Normal` `Run` at full opacity is the fast
    /// path, and everything else is a merge.
    ///
    /// **Returning the run is what deletes the invariant rather than checking it.**
    /// A `bool` `is_direct` leaves every call site to re-match the content: behind an
    /// `unreachable!`, or — as in the stack builder in `Engine` — behind an `if let`
    /// whose failure branch silently drops the merge. Three places that can disagree
    /// about what "direct" implies, and the third
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
        self.params.is_free()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::BlendMode;

    /// A drawable with a known opacity and nothing else — a matte, because it is the
    /// one [`CompositeItem`] that is plain data. A tile would need a GPU to make one,
    /// and none of what is under test here has anything to do with paint.
    fn item(opacity: f32) -> CompositeItem {
        CompositeItem::Matte(MatteDraw {
            rect: [0.0; 4],
            flags: 0.0,
            channels: [0.0; 4],
            resid: [0.0; 4],
            opacity,
            ramp: None,
        })
    }

    /// The opacity of every drawable in a group, in composite order.
    ///
    /// A test-local walk, deliberately not a method on [`CompositeGroup`] the render
    /// path could also call: the plan's traversal (`composite::plan`) is the only one
    /// of this tree, and a second one shared with the tests is a second one to agree
    /// with.
    fn opacities(group: &CompositeGroup) -> Vec<f32> {
        match &group.content {
            GroupContent::Run(items) => items
                .iter()
                .map(|i| match i {
                    CompositeItem::Matte(m) => m.opacity,
                    CompositeItem::Tile { opacity, .. } => *opacity,
                })
                .collect(),
            GroupContent::Stack(members) => members.iter().flat_map(opacities).collect(),
            // Nothing to draw: a filter is one fullscreen triangle off its uniform
            // slot, with no per-item stream behind it.
            GroupContent::Filter(_) => Vec::new(),
        }
    }

    fn faded(opacity: f32) -> CompositeParams {
        CompositeParams {
            opacity,
            ..CompositeParams::IDENTITY
        }
    }

    /// [`CompositeGroup::leaf`] folds the opacity into the items **and takes it off
    /// the merge**. Both halves, because either alone is a bug: folding without
    /// clearing fades the layer twice (§14.7), and clearing without folding loses the
    /// fade entirely.
    ///
    /// The third assertion is what the fold is *for*: a faded leaf stays on the fast
    /// path, where the same opacity left on the group would cost it two render passes.
    #[test]
    fn a_leaf_folds_its_opacity_into_its_items_and_off_its_merge() {
        let group = CompositeGroup::leaf(faded(0.4), vec![item(1.0), item(1.0)]);
        assert_eq!(opacities(&group), vec![0.4, 0.4], "the fold did not happen");
        assert_eq!(group.params.opacity, 1.0, "the merge would fade it again");
        assert!(
            group.as_direct_run().is_some(),
            "a faded leaf must stay on the fast path",
        );
    }

    /// The blend mode and the clip travel the other way: they describe a merge, so
    /// they stay on the group and never reach an item.
    #[test]
    fn a_leaf_keeps_its_blend_and_clip_on_the_merge() {
        let params = CompositeParams {
            blend: BlendMode::Multiply,
            clip: true,
            opacity: 0.5,
        };
        let group = CompositeGroup::leaf(params, vec![item(1.0)]);
        assert_eq!(group.params.blend, BlendMode::Multiply);
        assert!(group.params.clip);
        assert!(
            group.as_direct_run().is_none(),
            "a mode and a clip still need isolating",
        );
    }

    /// Collapsing a free group must **not** re-fold: its members' items already carry
    /// their own layers' opacities, and the collapse is a concatenation rather than a
    /// new layer.
    ///
    /// This is the trap [`CompositeGroup::run`] exists to keep open. `stack` ends in a
    /// constructor call, and reaching for `leaf` there — the one every other caller
    /// uses — would set every item to the group's own opacity, which the collapse has
    /// just proved is 1.0. Two faded layers tidied into a folder would come back out
    /// at full strength, which is exactly the "grouping changed my painting" failure
    /// §14.7 rule 2 exists to rule out.
    #[test]
    fn collapsing_a_free_group_keeps_its_members_folded_opacities() {
        let group = CompositeGroup::stack(
            CompositeParams::IDENTITY,
            vec![
                CompositeGroup::leaf(faded(0.25), vec![item(1.0)]),
                CompositeGroup::leaf(faded(0.75), vec![item(1.0)]),
            ],
        );
        assert!(
            group.as_direct_run().is_some(),
            "a free group of direct members collapses into a run (§14.7 rule 2)",
        );
        assert_eq!(opacities(&group), vec![0.25, 0.75]);
    }

    /// A group whose own params do something keeps them — and its **base** does not.
    ///
    /// The arrangement `composite_stack` builds: the base's content as a member at
    /// [`CompositeParams::IDENTITY`], the layer's own params on the group. Applied at
    /// both, an opacity of `a` would reach the base's paint as `a²`, which is the bug
    /// this shape was rebuilt to make unrepresentable.
    #[test]
    fn a_groups_params_ride_the_group_and_never_its_base() {
        let group = CompositeGroup::stack(
            faded(0.5),
            vec![
                CompositeGroup::leaf(CompositeParams::IDENTITY, vec![item(1.0)]),
                CompositeGroup::leaf(CompositeParams::IDENTITY, vec![item(1.0)]),
            ],
        );
        assert!(
            group.as_direct_run().is_none(),
            "a faded group cannot collapse — its opacity has to be applied at a merge",
        );
        assert_eq!(group.params.opacity, 0.5, "the group carries the fade");
        assert_eq!(
            opacities(&group),
            vec![1.0, 1.0],
            "no member, base included, may carry the group's fade as well",
        );
    }
}
