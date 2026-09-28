//! The compositor's draw list: what a document draws, bottom-to-top, cut into blend
//! groups (§6.3, §14.7, §18.0.4).
//!
//! **A function of the document alone.** The colour space and the view's tile cull are
//! the whole of what a list is built *with*, so every rule here — the run merge, the
//! group, the clip, the filter's backdrop, and the three restrictions an eyedropper
//! samples through — is decided without a device and tested without one.

use super::pick::PickSource;
use crate::colorspace::ColorSpace;
use crate::document::{CompositeParams, DocState, Layer, LayerContent};
use crate::gpu::composite::{CompositeGroup, CompositeItem, FilterDraw, MatteDraw};
use stark_model::document::LayerId;
use stark_model::geom::TileRect;

/// The walk that builds a draw list, and the two things it is built with.
///
/// Held as a value rather than passed as a pair of arguments because every step of the
/// walk needs both: a matte packs its colour through `cs`, and every stack passes
/// `visible` down to its members.
pub(super) struct DrawList<'a> {
    /// The document's colour space (§6.7): what a matte's sRGB paint becomes on the
    /// GPU. A tile needs none — its channels are already in the working space.
    cs: &'a dyn ColorSpace,
    /// The view-AABB cull (§6.3): only tiles this names are built into the list.
    /// `None` culls nothing — see [`ViewTransform::visible_tiles`].
    ///
    /// [`ViewTransform::visible_tiles`]: crate::view::ViewTransform::visible_tiles
    visible: Option<TileRect>,
}

impl<'a> DrawList<'a> {
    pub(super) fn new(cs: &'a dyn ColorSpace, visible: Option<TileRect>) -> Self {
        Self { cs, visible }
    }

    /// The whole document, bottom-to-top: every visible layer's tiles and mattes, each
    /// tagged with its layer opacity, cut into blend groups.
    ///
    /// Consecutive layers that need no isolation share one `Run`, composing correctly
    /// against each other under premultiplied "over" — so a document with no blend
    /// modes, no clipping and no groups produces exactly one. Anything else becomes a
    /// group of its own, its mode and its clip both being defined against *what is
    /// underneath it*, which has to be composited in isolation first.
    ///
    /// A layer that **carries** others is a group, and composites as a `Stack`: its own
    /// content at the bottom, then each carried layer merging into what is beneath it
    /// *within the group* (§14.2), the whole then merging outward through the base's
    /// blend mode, clip and opacity.
    ///
    /// Within a run this is an *ordered* item list rather than a flat tile list because
    /// a matte has to composite at its own place in the stack — a frame over the
    /// painting, a substrate under it (§15.4.4). The compositor re-batches consecutive
    /// tiles into one instanced draw, so an all-paint document costs nothing for it.
    pub(super) fn document(&self, doc: &DocState) -> Vec<CompositeGroup> {
        // The root stack has nothing under its first member, by definition — see
        // `stack`'s `under`.
        self.stack(doc.root().iter(), false)
    }

    /// One layer's **own** paint, with its mode, clip and opacity dropped — the
    /// eyedropper's sample-one-layer option (§18.0.2) and the layer panel's thumbnails
    /// (§14.6).
    ///
    /// What it carries is left out, and the three composite params go with it, since a
    /// sample is of the paint that is there rather than of the part of it that survives
    /// its surroundings. Sharing this with rendering is what makes a sample come off
    /// the same stack the screen draws.
    pub(super) fn layer_alone(&self, doc: &DocState, id: LayerId) -> Vec<CompositeGroup> {
        // Fully transparent is the one exception to dropping the opacity below: a layer
        // turned all the way down contributes nothing to the document, so sampling it
        // answers "nothing here" — the answer bare canvas gives — rather than reporting
        // paint that is switched off. Hidden reads the same way, and for the reason it
        // does everywhere else: a sample comes off the same stack the screen draws.
        let Some(layer) = doc.layer(id).filter(|l| l.is_shown()) else {
            return Vec::new();
        };
        let items = self.items(layer);
        if items.is_empty() {
            return Vec::new();
        }
        // **All three composite params are dropped**, opacity included: opacity says
        // how much of the layer the *document* shows, so "sample this layer" answers
        // the same colour at any setting.
        //
        // The colour would survive either way — the pick divides by the coverage it
        // sums, so opacity cancels (`mean_channels`). What this changes is where the
        // pick answers **at all**: a faded layer's coverage scales towards
        // `PICK_MIN_OPACITY`, where a thin glaze would report "nothing here".
        vec![CompositeGroup::leaf(CompositeParams::IDENTITY, items)]
    }

    /// The list an eyedropper source samples off (§18.0.2): the whole document or one
    /// layer, plus the two scoped questions — a group's interior, and the document cut
    /// above a layer.
    ///
    /// Here rather than in `engine::pick` because every one of them is a restriction of
    /// the same walk, and keeping the restrictions beside the walk is what keeps a
    /// sample coming off the same stack the screen draws.
    pub(super) fn pick(&self, doc: &DocState, source: PickSource) -> Vec<CompositeGroup> {
        match source {
            PickSource::Composite | PickSource::CompositeOverSubstrate => self.document(doc),
            PickSource::Layer(id) => self.layer_alone(doc, id),
            PickSource::Group { layer, below } => self.group_interior(doc, layer, below),
            PickSource::Below(layer) => self.below(doc, layer),
        }
    }

    /// The interior of the group that carries `layer`: the carrier's own content at the
    /// bottom, then the members — all of them, or with `below` only those up to and
    /// including `layer` (`PickSource::Group`).
    ///
    /// The carrier's outward params are dropped rather than applied — the same trade
    /// [`layer_alone`](Self::layer_alone) makes: they say how the group meets what is
    /// beneath it, and beneath it is what this source excludes. The members keep
    /// theirs, a sibling's mode against the base being part of what the group looks
    /// like *inside*. A layer in the root stack reads the root as its group, which
    /// makes this the whole document.
    ///
    /// A carrier that is hidden or turned all the way down answers nothing, on the
    /// `Layer` source's rule. The layer itself is only the anchor: hidden or not, its
    /// *group* is what is being asked about.
    fn group_interior(&self, doc: &DocState, layer: LayerId, below: bool) -> Vec<CompositeGroup> {
        let (base, members): (Option<&Layer>, &rpds::Vector<Layer>) = match doc.carrier_of(layer) {
            Some(cid) => {
                let Some(carrier) = doc.layer(cid).filter(|l| l.is_shown()) else {
                    return Vec::new();
                };
                (Some(carrier), carrier.carries())
            }
            // `carrier_of` answers `None` for a root layer and for a layer that does not
            // exist, and only the first has a group to sample.
            None if doc.contains_layer(layer) => (None, doc.root()),
            None => return Vec::new(),
        };
        // The anchor is in `members` by construction — `carrier_of` just said so.
        let end = match members.iter().position(|l| l.id == layer) {
            Some(at) if below => at + 1,
            _ => members.len(),
        };
        let own = base.map_or_else(Vec::new, |b| self.items(b));
        let mut groups = Vec::new();
        if !own.is_empty() {
            groups.push(CompositeGroup::leaf(CompositeParams::IDENTITY, own));
        }
        let under = !groups.is_empty();
        groups.extend(self.stack(members.iter().take(end), under));
        groups
    }

    /// The document with everything above `layer` switched off, bottom of the tree up
    /// to and including the layer itself (`PickSource::Below`).
    ///
    /// The chain of carriers from the root down to the layer is the only part of the
    /// tree that is *partially* included, so the walk follows exactly that path: whole
    /// stacks beneath each ancestor, then the ancestor cut above the next link
    /// ([`stack_below`](Self::stack_below)).
    fn below(&self, doc: &DocState, layer: LayerId) -> Vec<CompositeGroup> {
        if !doc.contains_layer(layer) {
            return Vec::new();
        }
        // Root-first: the last link is the layer itself.
        let mut path = vec![layer];
        let mut cur = layer;
        while let Some(carrier) = doc.carrier_of(cur) {
            path.push(carrier);
            cur = carrier;
        }
        path.reverse();
        self.stack_below(doc.root(), &path, false)
    }

    /// One stack's worth of [`below`](Self::below): everything beneath `path[0]` whole,
    /// then `path[0]` itself — whole when it is the target, cut above `path[1]` when it
    /// is an ancestor carrying the rest of the chain.
    ///
    /// The ancestor's own composite params are **kept**, unlike the one-layer and
    /// group-interior sources: this source asks what the screen would show with the
    /// upper layers hidden, and hiding a member does not lift the group's mode, clip or
    /// opacity off what remains. A hidden ancestor takes the whole chain with it; the
    /// layers beneath it still answer.
    fn stack_below(
        &self,
        layers: &rpds::Vector<Layer>,
        path: &[LayerId],
        under: bool,
    ) -> Vec<CompositeGroup> {
        let Some(&head) = path.first() else {
            return Vec::new();
        };
        let Some(at) = layers.iter().position(|l| l.id == head) else {
            return Vec::new();
        };
        if path.len() == 1 {
            // The target itself: it and everything beneath it in this stack, whole —
            // the ordinary walk, stopped early.
            return self.stack(layers.iter().take(at + 1), under);
        }
        let mut groups = self.stack(layers.iter().take(at), under);
        let ancestor = layers.get(at).expect("position() names an element");
        if !ancestor.is_shown() {
            return groups;
        }
        // `stack`'s group branch, with the carried stack cut by the rest of the path. An
        // ancestor is a carrier, and a filter never carries (§21.2), so the filter
        // branch cannot arise here.
        let own = self.items(ancestor);
        let carried = self.stack_below(ancestor.carries(), &path[1..], !own.is_empty());
        let Some(group) = group_of(ancestor.composite, own, carried) else {
            return groups;
        };
        // Through the same merge the draw list uses, so the two walks cannot come to
        // disagree about the shape of what they build.
        push_merging(&mut groups, group);
        groups
    }

    /// One stack's worth of groups — the root's, or a layer's carried stack.
    ///
    /// `under` says whether something already composites beneath this stack's first
    /// member: false for the document's own stack, and for a carried stack whether the
    /// **base's own content** draws anything, since a group's members composite over
    /// the base (§14.1).
    ///
    /// Only a **filter** asks (§21.2), which is why it is a `bool` rather than the
    /// base's items: everything else in this walk is defined against what it is drawn
    /// *into*, and a filter is the one thing defined against what has already been
    /// drawn.
    fn stack<'l>(
        &self,
        layers: impl Iterator<Item = &'l Layer>,
        under: bool,
    ) -> Vec<CompositeGroup> {
        let mut groups: Vec<CompositeGroup> = Vec::new();
        for layer in layers {
            // Hiding a layer hides what it carries: the group is the layer (§14.3), so
            // its visibility is the group's.
            if !layer.is_shown() {
                continue;
            }
            // A **filter layer** rewrites what is already composited beneath it *in its
            // own stack* (§21.2) — the same set a clip reads, which is what makes
            // "filter just this layer" the single gesture of carrying it onto that layer
            // rather than a scoping mode of its own.
            //
            // Two ways it reaches nothing, and both drop it from the draw list rather
            // than encoding a pass that provably cannot change a texel: **nothing is
            // beneath it here** (the foot of a stack, or a stack whose lower members
            // were all culled), or the filter is at its **neutral** setting, which is
            // what a freshly added one holds (§21.3). The first is also where a **clip**
            // would have nothing to say (§14.4.3), so the one place the two properties
            // could disagree is a pass that is not encoded at all.
            if let Some(f) = layer.filter() {
                debug_assert!(
                    layer.carries().is_empty(),
                    "a filter never carries (§21.2) — the state refuses the arrangement",
                );
                if (under || !groups.is_empty()) && !f.is_neutral() {
                    groups.push(CompositeGroup::filter(FilterDraw::new(f, layer.composite)));
                }
                continue;
            }
            let own = self.items(layer);
            let carried = self.stack(layer.carries().iter(), !own.is_empty());
            // An empty layer is dropped rather than given a group: for a blend mode or a
            // clip that saves two render passes which provably compute the identity,
            // which is what keeps a stack of empty glow layers free. A layer that
            // carries something visible is not empty, whatever its own content.
            let Some(group) = group_of(layer.composite, own, carried) else {
                continue;
            };
            push_merging(&mut groups, group);
        }
        groups
    }

    /// What one layer's own content draws, without what it carries.
    ///
    /// Every item comes out at **opacity 1**, and that is not a stub: the layer's
    /// opacity is a [`CompositeParams`] field, folded in — or not — by
    /// [`CompositeGroup::leaf`], the one place that decision is made (§14.7).
    ///
    /// Paint is culled to [`visible`](Self::visible) (§6.3); a matte is not. A matte's
    /// rect can be the *hole* in a frame, whose fill covers everything outside it
    /// (§15.4.4), so there is no box to test it against — and there is at most one per
    /// layer.
    fn items(&self, layer: &Layer) -> Vec<CompositeItem> {
        // The layer's frame (§14.12): its tiles are keyed in it, the view rect is on the
        // canvas, and this is the one place the two meet — the cull asks in the frame,
        // the item answers on the canvas.
        let d = layer.translation;
        let offset = stark_model::geom::Vec2::new(d.x as f32, d.y as f32);
        match &layer.content {
            LayerContent::Paint(tiles) => culled(tiles.map(), visible_in_frame(self.visible, d))
                .map(|(coord, handle)| CompositeItem::Tile {
                    origin: coord.origin() + offset,
                    handle: handle.clone(),
                    opacity: 1.0,
                })
                .collect(),
            LayerContent::Matte { region, paint } => vec![CompositeItem::Matte(MatteDraw::new(
                region, paint, offset, self.cs,
            ))],
            // A filter draws no items at all: it is a pass over what the *stack* has
            // built, not content of its own, so `stack` gives it a group rather than
            // asking here. The eyedropper's sample-one-layer option reads that same
            // nothing (§18.0.2), rather than the picture the filter happens to be
            // sitting over.
            LayerContent::Filter(_) => Vec::new(),
        }
    }
}

/// The view's tile rect, restated in a layer frame placed at `translation` on the
/// canvas (§14.12): what is visible of a translated layer's **local** tiles.
/// Rounded outward — a frame off the tile grid lands the rect astride tiles, and
/// a cull may only ever keep too much. Saturating, so a frame at the integer
/// horizon degrades to a wider rect rather than wrapping to the far side.
fn visible_in_frame(
    visible: Option<TileRect>,
    translation: stark_model::geom::IVec2,
) -> Option<TileRect> {
    if translation == stark_model::geom::IVec2::ZERO {
        return visible;
    }
    let rect = visible?;
    let t = stark_model::geom::TILE_SIZE as i64;
    // The rect moves by −frame: floor for the low edge, ceil for the high one.
    // In `i64`, like `sources_in`'s spans: the frame is clamped at the funnel
    // (`FRAME_LIMIT`), but a cull may not rest a wrap away from a value one
    // unclamped writer could someday hand it.
    let lo = |d: i32| (-(d as i64)).div_euclid(t);
    let hi = |d: i32| (-(d as i64) + t - 1).div_euclid(t);
    let edge = |base: i32, shift: i64| {
        (base as i64 + shift).clamp(i32::MIN as i64, i32::MAX as i64) as i32
    };
    Some(TileRect {
        min: (
            edge(rect.min.0, lo(translation.x)),
            edge(rect.min.1, lo(translation.y)),
        ),
        max: (
            edge(rect.max.0, hi(translation.x)),
            edge(rect.max.1, hi(translation.y)),
        ),
    })
}

/// The entries of `map` that `visible` admits, **walking whichever side is smaller**.
///
/// The cull is the same set either way — `TileRect::contains` and a probe of
/// `TileRect::coords` agree by construction — so the only question is which walk to
/// pay for, and the two differ by orders of magnitude in opposite directions. Scanning
/// the layer and filtering is right when the viewport admits more tiles than the layer
/// holds: every zoomed-out frame, which is the case the cull was written for (§6.3),
/// where the visible count scales as 1/zoom². Probing costs one hash lookup per
/// *visible* tile instead, which is what keeps a 1:1 view of a large painting off the
/// whole trie.
///
/// `None` claims everything: the box could not be measured (a non-finite view, or one
/// so far out that whole tiles leave the `i32` grid), and an optimization that cannot
/// see its input must do nothing rather than guess — see [`ViewTransform::visible_tiles`].
///
/// [`ViewTransform::visible_tiles`]: crate::view::ViewTransform::visible_tiles
fn culled<V>(
    map: &rpds::HashTrieMap<stark_model::geom::TileCoord, V>,
    visible: Option<TileRect>,
) -> Box<dyn Iterator<Item = (stark_model::geom::TileCoord, &V)> + '_> {
    match visible {
        Some(rect) if rect.count() < map.size() as u64 => Box::new(
            rect.coords()
                .filter_map(move |c| map.get(&c).map(|v| (c, v))),
        ),
        Some(rect) => Box::new(
            map.iter()
                .filter(move |(coord, _)| rect.contains(**coord))
                .map(|(coord, v)| (*coord, v)),
        ),
        None => Box::new(map.iter().map(|(coord, v)| (*coord, v))),
    }
}

/// **Where a layer's composite params go** (§14.4.3, §14.7) — the one statement of
/// the rule, and `None` for a layer that draws nothing at all.
///
/// - A **leaf** is the whole of what it draws, so its params are its run's.
/// - A **group's base** is a *member* of the group, not the group. Its own content
///   composites with `IDENTITY` and the layer's params are applied once, to the
///   composited whole. Every one of the three would be wrong applied twice: a blend
///   mode would combine the base with itself, a clip would clip the base to its own
///   coverage, and an opacity would fade it to `a²`.
///
/// The empty case answers `None` rather than an empty group because **the cull can
/// empty a layer that has paint**, so it fires for a document scrolled away from as
/// well as one not yet painted on. That is sound on the identity above rather than on
/// the two cases coinciding: `blend_common.wesl::merge` with a transparent source is
/// `cb` exactly — `cs.a` is 0, so both source terms vanish and the aux sum adds
/// nothing — for every mode and both clip states.
///
/// A function because two walks ask it: [`DrawList::stack`] builds the draw list, and
/// [`DrawList::stack_below`] builds the restriction of it a `PickSource::Below` sample
/// comes off.
fn group_of(
    params: CompositeParams,
    own: Vec<CompositeItem>,
    carried: Vec<CompositeGroup>,
) -> Option<CompositeGroup> {
    if own.is_empty() && carried.is_empty() {
        return None;
    }
    Some(if carried.is_empty() {
        CompositeGroup::leaf(params, own)
    } else {
        let mut members = Vec::with_capacity(carried.len() + 1);
        if !own.is_empty() {
            members.push(CompositeGroup::leaf(CompositeParams::IDENTITY, own));
        }
        members.extend(carried);
        CompositeGroup::stack(params, members)
    })
}

/// Push `group`, merging it into the run below when neither side needs isolating —
/// the fast path, and the reason an ordinary document is one group.
///
/// `as_direct_run_mut` is the test and the run in one, which is what makes this
/// total: a group that answered "direct" while holding something other than a `Run`
/// would otherwise be counted as merged and then silently dropped.
fn push_merging(groups: &mut Vec<CompositeGroup>, mut group: CompositeGroup) {
    let merged = match (
        groups
            .last_mut()
            .and_then(CompositeGroup::as_direct_run_mut),
        group.as_direct_run_mut(),
    ) {
        (Some(items), Some(more)) => {
            items.append(more);
            true
        }
        _ => false,
    };
    if !merged {
        groups.push(group);
    }
}

#[cfg(test)]
mod tests {
    //! The walk over documents built from the fold's pure half — the `DocState` calls
    //! `apply_pure` makes — so every rule it states runs where there is no GPU.
    //!
    //! A **matte** is what draws in these fixtures: it is the one content that covers
    //! without a tile, and a tile needs a device to lay. What is asserted is the
    //! *shape* the walk cuts, which is where every rule here shows up.

    use super::*;
    use crate::document::layer::PropValue;
    use crate::gpu::composite::GroupContent;
    use stark_model::Srgb;
    use stark_model::document::{BlendMode, ColorAdjust, Filter, MatteRegion, Parcel, Place};
    use stark_model::geom::TileCoord;

    /// The empty paint layer every document opens with. It draws nothing, so it is
    /// absent from every shape below.
    const ROOT: LayerId = LayerId::ROOT;
    const A: LayerId = LayerId::solo(1);
    const B: LayerId = LayerId::solo(2);
    const C: LayerId = LayerId::solo(3);
    const D: LayerId = LayerId::solo(4);
    const E: LayerId = LayerId::solo(5);

    /// A filter that does something, so it is never dropped as neutral (§21.3).
    const GREY: Filter = Filter::Color(ColorAdjust {
        saturation: 0.0,
        ..ColorAdjust::NEUTRAL
    });

    /// Oklab needs no device, which is the whole of why this module's rules are
    /// testable here.
    fn oklab() -> std::sync::Arc<dyn ColorSpace> {
        crate::colorspace::make(stark_model::ColorSpaceId::Oklab).expect("Oklab is unconditional")
    }

    /// The walk with nothing culled — every fixture here draws mattes, which a cull
    /// never touches.
    fn walk(cs: &dyn ColorSpace) -> DrawList<'_> {
        DrawList::new(cs, None)
    }

    /// `state` with a matte on top of `carrier`'s stack, or of the root's.
    fn matte(state: &DocState, id: LayerId, carrier: Option<LayerId>) -> DocState {
        state.insert_matte(
            id,
            carrier,
            Place::Top,
            MatteRegion::Everything,
            Parcel::Solid(Srgb::new([0.0, 0.0, 0.0])),
        )
    }

    /// The group tree as a readable shape: `run<n>` for a run of `n` drawables,
    /// `stack[..]` for a group that has to composite in isolation, `filter` for a
    /// pass. Every assertion below is written against this, since what the walk
    /// decides *is* the shape.
    fn shape(groups: &[CompositeGroup]) -> String {
        groups
            .iter()
            .map(|g| match &g.content {
                GroupContent::Run(items) => format!("run{}", items.len()),
                GroupContent::Stack(members) => format!("stack[{}]", shape(members)),
                GroupContent::Filter(_) => "filter".to_string(),
            })
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Root `[ROOT, A, B, C]`, the last three mattes: one run of three.
    fn three() -> DocState {
        let d = matte(&DocState::with_layer(ROOT), A, None);
        let d = matte(&d, B, None);
        matte(&d, C, None)
    }

    /// [`three`] with `A` carrying `D` and `E`, and blending, so the group is a group
    /// rather than a collapse: root `[ROOT, A[D, E], B, C]`.
    fn grouped() -> DocState {
        let d = matte(&matte(&three(), D, Some(A)), E, Some(A));
        d.set_prop(A, PropValue::Blend(BlendMode::Multiply))
    }

    /// **The two walks are the same cull.** [`culled`] picks between scanning the
    /// layer and probing the viewport purely on which is cheaper, so the answer must
    /// not depend on that choice — a disagreement would drop paint from the picture at
    /// exactly one zoom level. Driven over map/rect pairs that put the branch on both
    /// sides of its own threshold, including the equality case where it flips.
    #[test]
    fn both_culling_walks_pick_the_same_tiles() {
        let map: rpds::HashTrieMap<TileCoord, i32> = (0..40)
            .map(|i| (TileCoord::new(i % 7, i / 7), i))
            .collect::<Vec<_>>()
            .into_iter()
            .fold(rpds::HashTrieMap::new(), |m, (c, v)| m.insert(c, v));

        let sorted = |mut v: Vec<(TileCoord, i32)>| {
            v.sort_by_key(|(c, _)| (c.y, c.x));
            v
        };
        // Both routes, asked directly, so the test does not depend on which one
        // `culled` would have chosen for a given rect.
        let by_scan = |rect: TileRect| {
            sorted(
                map.iter()
                    .filter(|(c, _)| rect.contains(**c))
                    .map(|(c, v)| (*c, *v))
                    .collect(),
            )
        };
        let by_probe = |rect: TileRect| {
            sorted(
                rect.coords()
                    .filter_map(|c| map.get(&c).map(|v| (c, *v)))
                    .collect(),
            )
        };

        for rect in [
            TileRect::ALL,
            TileRect::EMPTY,
            // Wholly inside the painted region, wholly outside, and straddling it —
            // the three ways a viewport can sit over a painting.
            TileRect {
                min: (1, 1),
                max: (3, 3),
            },
            TileRect {
                min: (50, 50),
                max: (60, 60),
            },
            TileRect {
                min: (-4, 2),
                max: (2, 9),
            },
            // A single tile, which is the probe branch at its most lopsided.
            TileRect {
                min: (3, 2),
                max: (3, 2),
            },
        ] {
            // The scan is the reference: it is defined for every rect, including
            // [`TileRect::ALL`].
            let want = by_scan(rect);
            let got = sorted(culled(&map, Some(rect)).map(|(c, v)| (c, *v)).collect());
            assert_eq!(got, want, "culled disagreed on {rect:?}");
            // The probe is only asked where probing is *finite*, which is the size
            // guard inside `culled` stated from the outside: `TileRect::ALL` counts
            // 1.8e19 tiles, so walking its coords is not a slow test but a hung one.
            if rect.count() <= 10_000 {
                assert_eq!(want, by_probe(rect), "the walks disagree on {rect:?}");
            }
        }
        // An unmeasurable box claims everything, which is the safe direction.
        assert_eq!(culled(&map, None).count(), map.size());
    }

    /// **"An ordinary document is one `Run`"**, which [`DrawList::document`] claims.
    ///
    /// It is the whole argument for cutting the draw list into groups at all: a
    /// document that uses no blend modes, no clipping and no groups must reach the
    /// compositor as one run, or the feature is a tax on everyone who does not use
    /// it — a *performance* claim asserted like a correctness one.
    #[test]
    fn plain_layers_merge_into_one_run() {
        let cs = oklab();
        let doc = matte(&matte(&three(), D, None), E, None);
        assert_eq!(
            shape(&walk(cs.as_ref()).document(&doc)),
            "run5",
            "plain layers did not collapse into one run",
        );

        // And a layer that *does* need isolating breaks the run — otherwise the test
        // above would pass on a rule that merged everything unconditionally, which
        // would composite blend modes against the wrong backdrop.
        let blended = doc.set_prop(C, PropValue::Blend(BlendMode::Multiply));
        assert_eq!(
            shape(&walk(cs.as_ref()).document(&blended)),
            "run2,run1,run2",
            "a blended layer did not break the runs around it",
        );
    }

    /// A layer that draws nothing produces no group at all — the empty-skip both walks
    /// make, which is what lets a culled-away document cost nothing.
    #[test]
    fn a_layer_with_nothing_to_draw_makes_no_group() {
        assert!(group_of(CompositeParams::IDENTITY, Vec::new(), Vec::new()).is_none());
        let cs = oklab();
        let doc = DocState::with_layer(ROOT).insert_layer(A, None, Some(ROOT), None);
        assert!(
            walk(cs.as_ref()).document(&doc).is_empty(),
            "a document of empty layers drew something",
        );
    }

    /// **A group composites as a `Stack` with its base at the bottom** (§14.2), and the
    /// carrier's params are the group's, applied once to the whole.
    ///
    /// The base's own content composites with `IDENTITY` inside: applied twice, a group
    /// base at 0.5 would draw its paint at 0.25 (§14.7).
    #[test]
    fn a_carrier_composites_as_a_stack_with_its_own_content_first() {
        let cs = oklab();
        let groups = walk(cs.as_ref()).document(&grouped());
        assert_eq!(
            shape(&groups),
            "stack[run1,run2],run2",
            "the group's base did not come first, or its members did not merge",
        );
        let GroupContent::Stack(members) = &groups[0].content else {
            unreachable!("the shape above is a stack")
        };
        assert_eq!(
            groups[0].params.blend,
            BlendMode::Multiply,
            "the carrier's mode belongs to the group, applied once",
        );
        assert_eq!(
            members[0].params,
            CompositeParams::IDENTITY,
            "the base is a member, so it composites with no params of its own",
        );
    }

    /// **Organization is free**: the same group with nothing to isolate collapses into
    /// the run around it, so a document reaches the compositor as one run whether or
    /// not the artist put its layers in folders (§14.7).
    #[test]
    fn a_plain_group_collapses_into_the_run_around_it() {
        let cs = oklab();
        let plain = matte(&matte(&three(), D, Some(A)), E, Some(A));
        assert_eq!(shape(&walk(cs.as_ref()).document(&plain)), "run5");
    }

    /// **A clip costs the same isolation a blend mode does** (§14.4.3): it has to read
    /// the backdrop's alpha, so a clipped layer cannot join the run below it, and the
    /// flag travels on the group rather than being folded into its items.
    #[test]
    fn a_clipped_layer_breaks_the_run_and_keeps_its_flag() {
        let cs = oklab();
        let doc = three().set_prop(B, PropValue::Clip(true));
        let groups = walk(cs.as_ref()).document(&doc);
        assert_eq!(
            shape(&groups),
            "run1,run1,run1",
            "a clipped layer merged into the run below it",
        );
        assert!(groups[1].params.clip, "the clip did not reach the group");
        assert!(
            !groups[0].params.clip && !groups[2].params.clip,
            "the clip leaked onto a neighbour",
        );
    }

    /// **A filter that would reach nothing is not encoded at all** (§21.2): the pass
    /// provably cannot change a texel, and the same emptiness is where a clip would
    /// have nothing to say (§14.4.3).
    #[test]
    fn a_filter_is_dropped_wherever_it_would_reach_nothing() {
        let cs = oklab();
        let over_a_matte = || matte(&DocState::with_layer(ROOT), A, None);
        let cases = [
            // Nothing composited beneath it in the root stack.
            (
                "at the foot",
                DocState::with_layer(ROOT).insert_filter(B, None, None, GREY),
                "",
            ),
            // Every lower member culled — here by hiding the one that draws.
            (
                "over a hidden matte",
                over_a_matte()
                    .set_prop(A, PropValue::Visible(false))
                    .insert_filter(B, None, Some(A), GREY),
                "",
            ),
            // The foot of a *carried* stack whose base draws nothing: `under` is false
            // there too, and the empty base takes the whole group with it.
            (
                "at the foot of an empty group",
                DocState::with_layer(ROOT).insert_filter(B, Some(ROOT), None, GREY),
                "",
            ),
            // And the two that land. In the root stack, over a matte:
            (
                "over a matte",
                over_a_matte().insert_filter(B, None, Some(A), GREY),
                "run1,filter",
            ),
            // And at the foot of a group whose **base** draws — the one thing `under`
            // carries in, since a group's members composite over the base (§14.1).
            (
                "at the foot of a group whose base draws",
                over_a_matte().insert_filter(B, Some(A), None, GREY),
                "stack[run1,filter]",
            ),
        ];
        for (name, doc, want) in cases {
            assert_eq!(shape(&walk(cs.as_ref()).document(&doc)), want, "{name}");
        }
    }

    /// A filter at its **neutral** setting draws no pass even where it would reach
    /// something — which is what a freshly added one holds (§21.3).
    #[test]
    fn a_neutral_filter_draws_no_pass() {
        let cs = oklab();
        let doc = matte(&DocState::with_layer(ROOT), A, None).insert_filter(
            B,
            None,
            Some(A),
            Filter::Color(ColorAdjust::NEUTRAL),
        );
        assert_eq!(shape(&walk(cs.as_ref()).document(&doc)), "run1");
    }

    /// **`PickSource::Below` is the document cut above a layer**: everything beneath it
    /// whole, the layer itself, and nothing above — and the cut follows the chain of
    /// carriers rather than stopping at the root stack.
    #[test]
    fn below_takes_the_document_up_to_the_layer_and_no_further() {
        let cs = oklab();
        let doc = grouped();
        let walk = walk(cs.as_ref());
        // Inside the group: the carrier's own content and the one member, and nothing
        // of the sibling above it.
        assert_eq!(
            shape(&walk.pick(&doc, PickSource::Below(D))),
            "stack[run1,run1]",
            "cutting at a group's lower member kept more than the member",
        );
        // The carrier itself is the ordinary walk stopped early, so it comes whole —
        // hiding what is *above* a group does not empty the group.
        assert_eq!(
            shape(&walk.pick(&doc, PickSource::Below(A))),
            "stack[run1,run2]",
        );
        // Nothing above: `C` is left out, where the whole document has it.
        assert_eq!(
            shape(&walk.pick(&doc, PickSource::Below(B))),
            "stack[run1,run2],run1",
        );
        // The topmost layer of the root stack is the whole document.
        assert_eq!(
            shape(&walk.pick(&doc, PickSource::Below(C))),
            shape(&walk.document(&doc)),
        );
    }

    /// A **hidden ancestor** takes its whole chain with it, and the layers beneath it
    /// still answer — the one place `Below` differs from hiding the upper rows by hand.
    #[test]
    fn below_through_a_hidden_carrier_keeps_what_is_under_it() {
        let cs = oklab();
        let doc = matte(&three(), D, Some(B)).set_prop(B, PropValue::Visible(false));
        assert_eq!(
            shape(&walk(cs.as_ref()).pick(&doc, PickSource::Below(D))),
            "run1",
            "a hidden carrier hid the layer beneath it as well, or drew its own",
        );
    }

    /// **`PickSource::Group` is the interior of the group that carries the layer**: the
    /// carrier's own content first, then the members — all of them, or cut above the
    /// anchor — with the carrier's *outward* params dropped and the members' kept.
    #[test]
    fn a_group_source_drops_the_carriers_outward_params_and_keeps_the_members() {
        let cs = oklab();
        let doc = grouped().set_prop(D, PropValue::Clip(true));
        let walk = walk(cs.as_ref());
        let whole = walk.pick(
            &doc,
            PickSource::Group {
                layer: D,
                below: false,
            },
        );
        assert_eq!(
            shape(&whole),
            "run1,run1,run1",
            "the base, the clipped member and its sibling are three groups",
        );
        assert_eq!(
            whole[0].params,
            CompositeParams::IDENTITY,
            "the carrier's mode travelled into its own group's interior",
        );
        assert!(whole[1].params.clip, "a member lost its own clip");
        // Cut at the anchor: the base and `D`, never `E`.
        assert_eq!(
            shape(&walk.pick(
                &doc,
                PickSource::Group {
                    layer: D,
                    below: true,
                },
            )),
            "run1,run1",
        );
        // A layer in the root stack reads the root as its group, which is the whole
        // document.
        assert_eq!(
            shape(&walk.pick(
                &doc,
                PickSource::Group {
                    layer: B,
                    below: false,
                },
            )),
            shape(&walk.document(&doc)),
        );
    }

    /// A hidden carrier's interior is not on the screen to be sampled, and a layer that
    /// does not exist has no group and nothing beneath it — the three ways a scoped
    /// source's anchor comes to nothing.
    #[test]
    fn a_scoped_source_over_nothing_answers_nothing() {
        let cs = oklab();
        let doc = matte(&three(), D, Some(B)).set_prop(B, PropValue::Visible(false));
        let walk = walk(cs.as_ref());
        let group_of_ = |layer| PickSource::Group {
            layer,
            below: false,
        };
        assert!(walk.pick(&doc, group_of_(D)).is_empty());
        let absent = LayerId::solo(99);
        assert!(walk.pick(&doc, PickSource::Below(absent)).is_empty());
        assert!(walk.pick(&doc, group_of_(absent)).is_empty());
    }

    /// **One layer alone is its own paint at any setting**: all three composite params
    /// are dropped, and a layer that is hidden or turned all the way down answers
    /// nothing at all (§18.0.2).
    #[test]
    fn one_layer_alone_is_its_own_paint_at_any_setting() {
        let cs = oklab();
        let doc = three()
            .set_prop(B, PropValue::Blend(BlendMode::Multiply))
            .set_prop(B, PropValue::Clip(true))
            .set_prop(B, PropValue::Opacity(0.25));
        let walk = walk(cs.as_ref());
        let groups = walk.layer_alone(&doc, B);
        assert_eq!(shape(&groups), "run1");
        assert_eq!(
            groups[0].params,
            CompositeParams::IDENTITY,
            "a sample of the paint carried the settings that say how it meets its neighbours",
        );
        assert!(
            walk.layer_alone(&doc.set_prop(B, PropValue::Opacity(0.0)), B)
                .is_empty(),
            "a layer switched off answered as though it were there",
        );
        assert!(
            walk.layer_alone(&doc.set_prop(B, PropValue::Visible(false)), B)
                .is_empty(),
        );
        // What it carries is left out: the layer's *own* paint is the question.
        assert_eq!(
            shape(&walk.layer_alone(&matte(&doc, D, Some(B)), B)),
            "run1"
        );
    }
}
