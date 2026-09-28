//! What a render is asked to draw: the layer stack, cut into blend groups (§14.7,
//! §18.0.4).
//!
//! Pure description — no GPU anywhere in this file. The stack arrives here already
//! decided by the document, and the only judgement it makes is [`CompositeGroup::stack`]'s:
//! whether a group can tell itself apart from no group at all.

use crate::colorspace::ColorSpace;
use crate::document::CompositeParams;
use crate::gpu::tile::TilePairHandle;
use stark_model::document::{
    ChromaticAberration, ColorAdjust, Filter, FocalBlur, GradientAxis, GradientParcel, MatteRegion,
    Parcel,
};
use stark_model::geom::Vec2;
use stark_shaders::mirror::matte::{REGION_EVERYTHING, REGION_OUTSIDE};

/// A matte layer's draw parameters (§15.4).
// No `Debug`: the mirrored `Ramp` (§6.10) derives only what `bytemuck`
// needs, and 544 bytes of stop lanes would be noise in a dump anyway.
#[derive(Clone)]
pub(crate) struct MatteDraw {
    /// The region's rect in canvas px: `min.xy, max.xy`. For a frame this is the
    /// *hole* — the fill covers everything outside it. Zeros, never read, when
    /// `flags` says the region is the whole plane.
    pub rect: [f32; 4],
    /// Region kind, as the shader reads it (§15.5): `matte.wesl`'s
    /// [`REGION_OUTSIDE`] or [`REGION_EVERYTHING`].
    pub flags: f32,
    /// Fill color in the document's working color space. Zeros for a gradient
    /// matte, whose color comes from `ramp` per fragment.
    pub channels: [f32; 4],
    /// The same color's **residual** in `.xyz` (§6.7); `.w` unused. Zero in a space
    /// that has none, which is the true value rather than a stand-in.
    pub resid: [f32; 4],
    /// The layer's opacity.
    pub opacity: f32,
    /// A gradient paint's ramp (§22.4), packed for the shader's per-matte uniform:
    /// stops in working-space channels, axis in canvas px. `None` for a solid matte,
    /// which binds the shared zeroed ramp. Boxed so the rare gradient's 544 bytes of
    /// stop lanes do not ride every [`CompositeItem`].
    pub ramp: Option<Box<stark_shaders::mirror::matte::Ramp>>,
}

impl MatteDraw {
    /// The draw for a matte layer's `region` and `paint`, placed by the layer's
    /// `offset` (§15.4). Region, rect and gradient axis are all stated in the layer's
    /// own frame (§14.12), and this is where the three of them land on the canvas.
    ///
    /// sRGB in the log, working-space channels on the GPU — the same conversion the
    /// brush colour gets, so a matte means the same colour in an Oklab and a Mixbox
    /// document. A gradient converts every stop that way and the shader interpolates
    /// in the working space (§22.4).
    ///
    /// Comes out at **opacity 1**: the layer's own is a [`CompositeParams`] field that
    /// [`CompositeGroup::leaf`] folds in, or not, and setting it here as well is how a
    /// group's base gets faded twice (§14.7).
    pub(crate) fn new(
        region: &MatteRegion,
        paint: &Parcel,
        offset: Vec2,
        cs: &dyn ColorSpace,
    ) -> Self {
        let rect = match region.translated(offset).rect() {
            Some((min, max)) => [min.x, min.y, max.x, max.y],
            // The whole plane: the shader never reads the rect (`flags` routes past
            // it), so zeros rather than sentinels.
            None => [0.0; 4],
        };
        let flags = match region {
            MatteRegion::OutsideRect { .. } => REGION_OUTSIDE,
            MatteRegion::Everything => REGION_EVERYTHING,
        };
        let (channels, resid, ramp) = match paint {
            Parcel::Solid(color) => {
                let l = cs.rgb_to_latent(*color);
                (
                    [l.lat[0], l.lat[1], l.lat[2], 1.0],
                    [l.res[0], l.res[1], l.res[2], 0.0],
                    None,
                )
            }
            Parcel::Gradient(GradientParcel { gradient, axis }) => {
                let mut ramp = stark_shaders::mirror::matte::Ramp::default();
                let stops = gradient.stops();
                ramp.stops = stops.len() as f32;
                ramp.axis = match axis.translated(offset) {
                    GradientAxis::Linear { from, to } => [from.x, from.y, to.x, to.y],
                    GradientAxis::Radial { center, radius } => {
                        ramp.axis_kind = 1.0;
                        [center.x, center.y, radius, 0.0]
                    }
                };
                // Bounded by `Gradient`'s own invariant rather than by a check here: a
                // ramp holds at most `gradient::MAX_STOPS`, the length `matte.wesl`
                // declares `stop_c` at. Truncating instead would hide a broken
                // invariant behind a wrong picture.
                for (i, stop) in stops.iter().enumerate() {
                    let l = cs.rgb_to_latent(stop.color);
                    ramp.stop_c[i] = [l.lat[0], l.lat[1], l.lat[2], stop.t];
                    ramp.stop_r[i] = [l.res[0], l.res[1], l.res[2], 0.0];
                }
                ([0.0; 4], [0.0; 4], Some(Box::new(ramp)))
            }
        };
        Self {
            rect,
            flags,
            channels,
            resid,
            opacity: 1.0,
            ramp,
        }
    }
}

/// One item of compositing pass A, in bottom-to-top stack order.
///
/// An ordered list rather than a flat tile array because a matte composites at its own
/// place in the stack — a frame over the painting, a substrate under it (§15.4.4).
/// Tiles already cost one draw each, so interleaving mattes adds no per-tile overhead.
#[derive(Clone)]
pub(crate) enum CompositeItem {
    Tile {
        /// Where the tile's interior sits **on the canvas**: its own origin plus the
        /// layer's translation (§14.12), resolved by the item builder because that is
        /// the one place that knows whose layer the tile is. This is the whole of how
        /// a translated layer reaches the screen.
        origin: Vec2,
        handle: TilePairHandle,
        opacity: f32,
    },
    Matte(MatteDraw),
}

impl CompositeItem {
    /// Draw this item at `opacity` instead of the one it was built with.
    ///
    /// **Private, with one caller**: [`CompositeGroup::leaf`], which folds a layer's
    /// opacity into items that do not overlap. Setting it anywhere else risks the
    /// double fade §14.7 describes — the item faded, then the group's merge fading it
    /// again.
    fn set_opacity(&mut self, opacity: f32) {
        match self {
            CompositeItem::Tile { opacity: o, .. } => *o = opacity,
            CompositeItem::Matte(m) => m.opacity = opacity,
        }
    }
}

/// A filter layer's pass (§21): what it does, how much of it lands, and where.
///
/// **Both [`CompositeParams`] a filter reads live here** rather than on the enclosing
/// [`CompositeGroup`], because a filter merges nothing outward for them to describe
/// (see [`GroupContent::Filter`]): they are settings of a pass.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FilterDraw {
    pub op: FilterOp,
    /// How much of the adjustment lands: the layer's opacity (§21.4).
    pub strength: f32,
    /// Whether the filter is confined to the coverage it read: the layer's clip
    /// (§21.4, §14.4). A **point** filter is already confined, so this changes
    /// nothing for one; the gather is what it bounds.
    pub clip: bool,
}

/// What a filter layer does, in the terms its pass runs it in (§21).
///
/// The document's own settings, in canvas terms: which *texels* a spread, an angle or
/// a radius becomes depends on the view, and a draw deliberately has none (§21.10,
/// §21.12). The one conversion made here is the gradient map's, whose ramp does not
/// depend on the view. Which `u32` each variant is numbered, and which lanes carry it,
/// is `filter_common.wesl`'s — stated once, in
/// [`filter_uniform`](super::filter::filter_uniform).
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum FilterOp {
    Color(ColorAdjust),
    Chromatic(ChromaticAberration),
    FocalBlur(FocalBlur),
    GradientMap {
        /// The ramp in the shader's own terms: Oklab `(L, a, b)` and the stop's
        /// position per lane, `count` of them live. Boxed for [`MatteDraw::ramp`]'s
        /// reason.
        stops: Box<[[f32; 4]; MAX_MAP_STOPS]>,
        count: usize,
    },
}

/// The stop lanes the filter uniform carries — the shader's literal `16`
/// (`filter_common.wesl`, §6.10's generator cannot resolve a named length), held
/// to the fitter's own bound here so the two cannot drift: every `Gradient` fits.
const MAX_MAP_STOPS: usize = stark_model::gradient::MAX_STOPS;

impl FilterDraw {
    /// The draw for `filter` under the layer's own `params`: the opacity becomes the
    /// strength, the clip travels as itself, and the blend is the one a filter never
    /// has (§21.4).
    pub(crate) fn new(filter: Filter, composite: CompositeParams) -> Self {
        let op = match filter {
            Filter::Color(c) => FilterOp::Color(c),
            Filter::Chromatic(c) => FilterOp::Chromatic(c),
            Filter::FocalBlur(b) => FilterOp::FocalBlur(b),
            // Stops are straight sRGB in the document and Oklab in the pass, because
            // Oklab is where `Gradient::sample`'s interpolation — the thing the map
            // *is* — is defined (§21.11). A rampless map is neutral and the draw list
            // dropped it (§21.3); `None` only keeps the match total.
            Filter::GradientMap(g) => {
                let mut stops = Box::new([[0.0f32; 4]; MAX_MAP_STOPS]);
                let count = g.as_ref().map_or(0, |g| {
                    for (lane, stop) in stops.iter_mut().zip(g.stops()) {
                        let lab = stark_model::color::srgb_to_oklab([
                            stop.color[0],
                            stop.color[1],
                            stop.color[2],
                            1.0,
                        ]);
                        *lane = [lab[0], lab[1], lab[2], stop.t];
                    }
                    g.stops().len()
                });
                FilterOp::GradientMap { stops, count }
            }
        };
        Self {
            op,
            strength: composite.opacity,
            clip: composite.clip,
        }
    }

    /// Whether this is the focal blur (§21.12) — the one kind whose pass is preceded
    /// by a convolution.
    pub(crate) fn is_focal_blur(&self) -> bool {
        matches!(self.op, FilterOp::FocalBlur(_))
    }

    /// Whether this filter **gathers**: reads texels other than the one it writes
    /// (§21.10, §21.12) — [`Filter::resamples`]'s answer, which the merge must refuse
    /// ([`MergeRenderer::apply_filter`](crate::gpu::merge::MergeRenderer::apply_filter)).
    pub(crate) fn resamples(&self) -> bool {
        match self.op {
            FilterOp::Color(_) | FilterOp::GradientMap { .. } => false,
            FilterOp::Chromatic(_) | FilterOp::FocalBlur(_) => true,
        }
    }
}

/// One **blend group** of pass A: something that composites on its own, and how its
/// result merges into everything below it (§18.0.4, §14.7).
///
/// A group is defined against *what is underneath it*, so it has to be composited
/// alone, on nothing, before it can be merged — the per-layer isolation §6.3 names as
/// the prerequisite for richer modes, recursed: [`GroupContent::Stack`] is a group
/// whose members are themselves groups.
#[derive(Clone)]
pub(crate) struct CompositeGroup {
    /// How this group's composited result merges into everything below it — the
    /// **layer's own** params, applied once, to the whole (§14.4.3).
    ///
    /// The same [`CompositeParams`] the document holds, carried through unchanged.
    /// Clipping costs the same isolation a blend mode does, and for the same reason:
    /// the merge has to *read* the backdrop's alpha. [`Self::leaf`] is the one
    /// constructor that moves anything out of here, and it moves only the opacity,
    /// only onto items that provably do not overlap.
    pub params: CompositeParams,
    pub content: GroupContent,
}

/// What a [`CompositeGroup`] is made of — and the fast path, expressed as a
/// shape rather than as a condition inside the encoder.
#[derive(Clone)]
pub(crate) enum GroupContent {
    /// A run of the stack that composites internally under plain premultiplied
    /// "over", with **no isolation**.
    ///
    /// Consecutive `Normal`, unclipped layers carrying nothing compose correctly
    /// against each other *and* against the accumulator, so a document with no modes,
    /// no clipping and no groups is a single `Run`.
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
    /// Not a *thing to composite* like a `Run` or a `Stack`, but a function of what has
    /// been composited already — which is the blend pass's own relationship to the
    /// accumulator with the isolated source removed, so it takes the same ping-pong and
    /// the same scratch and costs the same one pass (`composite/filter.rs`).
    ///
    /// Its enclosing [`CompositeGroup`]'s params are always
    /// [`CompositeParams::IDENTITY`]: those describe a **merge**, and a filter has no
    /// source to merge. The two it does read live in the draw instead — the opacity as
    /// the filter's strength, the clip as the bound on what the pass may write (§21.4).
    Filter(FilterDraw),
}

impl CompositeGroup {
    /// A run of drawables that merges outward through `params`, **each item carrying
    /// whatever opacity it was built with**.
    ///
    /// For a caller that already holds finished items — the collapse in
    /// [`Self::stack`], which concatenates runs whose opacities are already folded in.
    /// A caller turning *one layer* into a run wants [`Self::leaf`], which folds that
    /// layer's opacity in.
    pub(crate) fn run(params: CompositeParams, items: Vec<CompositeItem>) -> Self {
        Self {
            params,
            content: GroupContent::Run(items),
        }
    }

    /// **One layer's** drawables, merging outward through `params` — the constructor
    /// every layer goes through, and the only place opacity is folded into an item.
    ///
    /// The fold keeps a faded layer on the fast path: `params.opacity` left on the
    /// group would make it non-direct and cost two render passes, while a layer's
    /// tiles do not overlap, so scaling each of them is identical to scaling the
    /// composited layer.
    ///
    /// **Equivalent, and therefore exclusive.** The opacity comes *off* the params on
    /// the way past, so what is folded is never also applied at a merge — the double
    /// fade §14.7 describes, which this constructor exists to make unrepresentable.
    pub(crate) fn leaf(params: CompositeParams, mut items: Vec<CompositeItem>) -> Self {
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
    pub(crate) fn filter(draw: FilterDraw) -> Self {
        Self {
            params: CompositeParams::IDENTITY,
            content: GroupContent::Filter(draw),
        }
    }

    /// A group of `members` — **collapsed into a plain [`Run`](GroupContent::Run)
    /// when nothing about it could tell itself apart from one**
    /// (§14.7 rule 2).
    ///
    /// A group that merges normally, unclipped, at full opacity, and whose every
    /// member draws directly, changes no blending scope: its members were already
    /// composing against everything below them under `over`, and isolating them would
    /// produce the same pixels through two extra render passes each. So it produces
    /// the identical draw list to no group at all, which is what makes "organization
    /// is free" structural rather than promised.
    ///
    /// The condition cannot be relaxed to "the group itself is normal": a member with
    /// a mode of its own *does* blend against a different backdrop once isolated, and
    /// that difference is the feature (§14.5).
    pub(crate) fn stack(params: CompositeParams, members: Vec<Self>) -> Self {
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
    /// **Returning the run is what deletes the invariant rather than checking it.** A
    /// `bool` `is_direct` leaves every call site to re-match the content, and a call
    /// site that gets that wrong silently drops the merge.
    pub(crate) fn as_direct_run(&self) -> Option<&[CompositeItem]> {
        match &self.content {
            GroupContent::Run(items) if self.is_free() => Some(items),
            _ => None,
        }
    }

    /// [`Self::as_direct_run`] for a caller that appends to the run it gets back —
    /// the stack builder folding a group into the one below it.
    pub(crate) fn as_direct_run_mut(&mut self) -> Option<&mut Vec<CompositeItem>> {
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
    use stark_model::document::BlendMode;

    /// A drawable with a known opacity and nothing else — a matte, because it is the
    /// one [`CompositeItem`] that is plain data. A tile would need a GPU to make one,
    /// and none of what is under test here has anything to do with paint.
    fn item(opacity: f32) -> CompositeItem {
        CompositeItem::Matte(MatteDraw {
            rect: [0.0; 4],
            flags: REGION_OUTSIDE,
            channels: [0.0; 4],
            resid: [0.0; 4],
            opacity,
            ramp: None,
        })
    }

    /// The opacity of every drawable in a group, in composite order.
    ///
    /// Test-local rather than a method the render path could also call: the plan's
    /// traversal (`composite::plan`) is the only walk of this tree, and a shared second
    /// one is a second one to agree with.
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

    /// [`CompositeGroup::leaf`] folds the opacity into the items **and takes it off the
    /// merge**. Either half alone is a bug: folding without clearing fades the layer
    /// twice (§14.7), clearing without folding loses the fade. The third assertion is
    /// what the fold is *for* — a faded leaf stays on the fast path.
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
    /// This is the trap [`CompositeGroup::run`] exists to keep open: reaching for
    /// `leaf` in `stack` instead would set every item to the group's own opacity —
    /// which the collapse has just proved is 1.0 — so two faded layers tidied into a
    /// folder would come back out at full strength (§14.7 rule 2).
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
    /// both, an opacity of `a` would reach the base's paint as `a²`.
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

    /// Every filter this build offers says what the **model** says about gathering.
    ///
    /// [`Filter::resamples`] and [`FilterDraw::resamples`] are two matches over two
    /// enums, so this is the one hop where they could part — and the merge that
    /// refuses a gather (§14.11.7) asks the draw.
    #[test]
    fn every_filter_draw_says_what_the_model_says_about_gathering() {
        for filter in Filter::ALL {
            let expected = filter.resamples();
            let label = filter.label();
            let draw = FilterDraw::new(filter, CompositeParams::IDENTITY);
            assert_eq!(
                draw.resamples(),
                expected,
                "the draw for the {label} filter disagrees with `Filter::resamples`",
            );
        }
    }
}
