//! Merging a layer **down** onto the one beneath it (§14.11): when the pair can be
//! replaced by one layer, and how the paint lands when it can.
//!
//! Pure CPU, no GPU anywhere in this file — [`crate::gpu::merge`] does the tiles. What
//! is decided here is the harder half, because merge-down has exactly one law:
//!
//! > **A merge must not change what the document looks like.**
//!
//! That is not a nicety, it is what distinguishes a merge from a destructive edit. A
//! painter merges to spend fewer layers on a thing that is finished, not to accept a
//! new picture — so a merge that shifts a pixel is a bug with no way to notice it
//! until the work is saved and the layers are gone.
//!
//! The law is not free, and this module is where it costs something: **it is not
//! always possible**, so this returns an `Option` and the panel offers the control
//! only where there is one.
//!
//! # What has to hold
//!
//! Write `B` for everything composited beneath the pair, `D` for the lower layer
//! (the **destination**, which survives) and `S` for the upper (the **source**, which
//! is consumed). The document currently shows `merge_S(merge_D(B, D), S)`; after the
//! merge it shows `merge_D(B, D ⊕ S)` for whatever tile-space `⊕` this module names.
//! Those two agreeing for **every** backdrop `B` is the whole question, and it splits
//! into two independent ones:
//!
//! - **Does `S` reach the accumulator by plain "over"?** Only then is `⊕` the stacking
//!   law, and only then does over's associativity — `over(over(B,D),S) =
//!   over(B, over(D,S))` — carry the result across. A blend mode does not associate
//!   with over at all, and a clip is a deletion rather than a stack.
//! - **Is the backdrop `S` is defined against exactly `D`?** A clip reads the
//!   backdrop's coverage, so the answer decides whether "clipped to `D`" is even what
//!   `S` means. It is `D` alone in exactly two places: `S` is the bottom of the stack
//!   its carrier `D` opens (a group's members composite over its base, §14.1), or `S`
//!   sits second from the bottom of the **root** stack, whose accumulator starts
//!   cleared.
//!
//! # Two kinds of source
//!
//! Everything above is about a source made of **paint**, which is stacked into the
//! destination. A **filter layer** is the other kind, and it merges by a different
//! sentence (§14.11.7): it has nothing to stack, being a function of what it sits on,
//! so the merge *rewrites* the destination's channels and leaves every other thing
//! about that layer alone — its blend, its clip, its opacity, its place.
//!
//! Only the second question above binds it, and it binds absolutely: a filter rewrites
//! the accumulator beneath it, so baking it into `D` is the same picture exactly when
//! that accumulator is `D` alone. The first question is vacuous — nothing arrives, so
//! there is no "over" to associate and no mode to agree about.
//!
//! What it costs is one refusal of its own. The merged tiles are written by a pass
//! that must be a pure function of canvas position (§6.4) — the rule that keeps a
//! tile's apron bit-identical to its neighbour's interior — and a filter that reads
//! *neighbouring* texels (§21.10) is not one, at any apron width, since its reach is
//! the document's to set. So [`Filter::resamples`](super::filter::Filter::resamples)
//! is asked, and a gather is declined.
//!
//! # What is deliberately refused
//!
//! **Two layers sharing a blend mode — unimplemented, not unsound.** The soundness
//! question is settled: the emissive modes weigh coverage in emission, where the
//! combination is addition, so they are associative at any coverage (§18.0.4,
//! `blend_common::added_light`). What it still needs is the mode's algebra evaluated
//! in **tile** space: the light conversion is per color space, and in a pigment
//! document that means binding the Mixbox LUT into a pass that has never needed it.
//! The slab-law inversion the result then has to be stored through already exists
//! (`merge.wesl::optical_mass`).
//!
//! **A source with a blend mode, merged into its carrier.** Sound for a different and
//! simpler reason — the group's isolated content is what the merge rewrites, so the
//! outward merge is unchanged whatever the mode — and blocked on the same tile-space
//! conversion. The two arrive together when it does.
//!
//! **Groups.** A source that carries layers would have to flatten a whole subtree, and
//! a destination that carries layers is not what sits beneath the source — its whole
//! group is. Both are refused rather than partially handled. (A **carrier** as the
//! destination is not this case and is offered: there the source is the bottom member
//! of the stack that carrier opens, so what sits beneath it *is* the carrier's own
//! content, §14.1.)
//!
//! **A filter as the destination, and a matte as either.** Neither holds tiles, so
//! there is nothing to rewrite and nothing to rewrite into. A solid matte is the one
//! case with an obvious answer — its composited color is its fill, so a filter merged
//! into one would be that matte with a filtered fill — and it is left out on purpose
//! rather than overlooked: a *gradient* matte has no such answer, because the
//! adjustment is nonlinear and filtering the stops is not filtering the ramp.

use super::layer::{CompositeParams, Layer, LayerContent, LayerId};
use super::state::DocState;

/// The merge a [`plan`] found: which layer is consumed, which survives, how the paint
/// lands, and what each side's tiles are worth on their own.
///
/// The destination is derived rather than chosen — "down" names exactly one layer —
/// but it is carried anyway, because a [`Footprint`] is built from the action alone
/// and cannot go looking for it (§12.6). The applying side re-derives the plan and
/// declines if it names a different destination, which is what keeps a peer's action
/// honest against a tree that has moved under it.
///
/// [`Footprint`]: super::footprint::Footprint
#[derive(Clone, Debug, PartialEq)]
pub struct MergePlan {
    pub source: LayerId,
    pub dest: LayerId,
    /// Which of the two merges this is, and everything that one of them needs.
    pub kind: MergeKind,
}

/// The two things a merge can be, which are two because a source can be two things.
///
/// A layer of paint is **stacked into** the destination — two slabs become one, and
/// every number in [`Stack`](Self::Stack) is about reconciling two layers' params
/// into the one set that speaks for both afterwards. A **filter** is not stacked into
/// anything; it is a function of what it sits on, so merging it **rewrites** the
/// destination's channels and leaves everything else about that layer alone. Nothing
/// is reconciled, because nothing arrives.
///
/// One enum rather than one struct with fields that go quiet: `dest_opacity` and
/// `keeps` have no meaning for a filter merge, and a field that cannot change a pixel
/// is a field somebody eventually reads anyway.
#[derive(Clone, Debug, PartialEq)]
pub enum MergeKind {
    /// The source's paint stacked into the destination's, through the source's own
    /// params (§14.11.3).
    Stack {
        /// How the source's paint meets the destination's — **the source layer's own**
        /// params, always. A merge folds the upper layer into the lower through exactly
        /// the merge the compositor would have run between the two, which is the whole of
        /// why the result is the same picture.
        ///
        /// Its `opacity` is the source's own, and it is folded into the merged tiles.
        source_params: CompositeParams,
        /// What the destination's tiles are worth on their own: its own opacity when the
        /// two are siblings, and **1.0 when the destination is the carrier** — a group's
        /// base composites at [`CompositeParams::IDENTITY`] inside the isolation, its
        /// slider belonging to the group as a whole (§14.7).
        dest_opacity: f32,
        /// What the surviving layer's params become. The destination's own, untouched,
        /// when it is a carrier; otherwise the pair's shared mode at full opacity, both
        /// sliders having been folded into the tiles.
        keeps: CompositeParams,
    },
    /// The source is a **filter layer**, and the merge runs it over the destination's
    /// stored channels (§14.11.7).
    ///
    /// The filter travels rather than being re-read on the far side, on
    /// `source_params`' argument: the plan is the whole description of the operation,
    /// and the applying side should not have to go back to the tree for half of it.
    ///
    /// **No `dest_opacity` and no `keeps`**, and their absence is the content of this
    /// variant. The destination keeps every one of its params — a filter merge does
    /// not stack anything onto it, so there is nothing for its sliders to have to
    /// absorb — and its opacity never enters the arithmetic at all: pass A's slab law
    /// scales coverage and height and leaves the color alone, so the un-premultiplied
    /// color the filter is defined on is the same whatever the slider says.
    Filter {
        filter: super::filter::Filter,
        /// The filter layer's own params, which is what the **compositor** reads of
        /// one (§21.4): the opacity is the filter's strength and has to be baked in,
        /// or a half-applied grade would merge to a fully applied one.
        ///
        /// Carried whole rather than as that one number, so the merge builds its draw
        /// through the same `FilterDraw::new` the draw list does. The clip rides along
        /// having nothing to do — it is the identity for every filter that may come
        /// this way (§21.4.1) — and dropping it here would mean this path and the
        /// screen's read the layer differently, which is the divergence the shared
        /// constructor exists to rule out.
        source_params: CompositeParams,
    },
}

/// The merge of `source` into the layer beneath it, or `None` when there is none that
/// preserves the document's appearance — see the module header for which is which.
///
/// A **pure function of the state**, which is what lets the action carry only the two
/// ids: every peer and every replay asks this question of the same document and gets
/// the same answer, so a merge is accepted or declined identically everywhere without
/// the log having to carry the reasoning.
pub fn plan(state: &DocState, source: LayerId) -> Option<MergePlan> {
    let site = state.site_of(source)?;
    let s = state.layer(source)?;
    let (dest, backdrop_is_dest) = match site.carrier {
        // In a carried stack, the bottom member's backdrop is the carrier's **own
        // content** — a group's members composite over its base (§14.1) — so
        // "beneath" walks out of the stack rather than off the end of it.
        None if site.index == 0 => return None, // the foot of the document: nothing below
        Some(carrier) if site.index == 0 => (carrier, true),
        // Otherwise the layer directly below is the lower sibling. Its backdrop *is*
        // the destination only at the foot of the root stack, where the accumulator
        // starts cleared; anywhere else there is a carrier's content or a lower
        // sibling under it as well.
        _ => {
            let stack = match site.carrier {
                None => state.root(),
                Some(carrier) => &state.layer(carrier)?.carries,
            };
            let below = stack.get(site.index - 1)?;
            (
                below.id,
                site.carrier.is_none() && site.index == 1 && !below.composite.clip,
            )
        }
    };
    plan_at(&MergeSite {
        source: s,
        dest: state.layer(dest)?,
        backdrop_is_dest,
        dest_is_carrier: site.carrier == Some(dest),
    })
}

/// Where a candidate source sits, as a **compositing walk of the tree already knows
/// it** — the pure input [`plan_at`] decides from.
///
/// Split out from [`plan`] because the projection asks this question of *every*
/// layer (the control is offered per row, §14.11), and `plan`'s own `site_of` is a
/// walk of the whole tree: asking it per layer made `Engine::observe` quadratic in
/// the layer count. A walk in composite order already knows each of these — the
/// lower sibling it just visited, the carrier it descended through — so the search
/// was for an answer the caller had in hand, which is the argument `LayerInfo`'s
/// own `has_backdrop` already makes one field over.
pub(crate) struct MergeSite<'a> {
    /// The layer that would be consumed.
    pub source: &'a Layer,
    /// The layer directly beneath it: its lower sibling, or — at the foot of a
    /// carried stack — the carrier whose own content it composites over (§14.1).
    pub dest: &'a Layer,
    /// Whether [`dest`](Self::dest) is the whole of what the source composites onto
    /// (§14.11.2). True where the source is the bottom member of the stack its
    /// carrier opens, and where it sits second from the foot of the **root** stack
    /// over an unclipped layer, whose accumulator starts cleared.
    pub backdrop_is_dest: bool,
    /// Whether the destination is the source's **carrier** — the group case, where
    /// the base composites at full strength inside the isolation and its slider
    /// belongs to the group as a whole (§14.7).
    pub dest_is_carrier: bool,
}

/// The merge on offer at a located site, or `None` when there is none that
/// preserves the document's appearance — [`plan`] without the search.
///
/// Everything below this point is a question about the two layers themselves, which
/// is why the split falls here: the tree is consulted only to find `dest` and to
/// answer the two flags, and nothing after that reads it again.
pub(crate) fn plan_at(site: &MergeSite) -> Option<MergePlan> {
    let (s, d) = (site.source, site.dest);
    let backdrop_is_dest = site.backdrop_is_dest;
    // A source is one of two things, and which one decides the whole shape of the
    // merge (see [`MergeKind`]): paint that carries nothing — a subtree cannot travel
    // as one tile — or a **filter**, which carries nothing by construction (§21.2).
    let filter = s.filter();
    if filter.is_none() && !is_plain_paint(s) {
        return None;
    }

    // Hiding a layer hides what it carries (§14.3), so a merge across a difference in
    // visibility would either reveal paint that is hidden or hide paint that is not.
    // Two hidden layers merge fine: nothing shows either way, before or after.
    if s.visible != d.visible {
        return None;
    }

    // A **filter** source, which is the whole of the second kind of merge (§14.11.7).
    //
    // It asks §14.11.2's second question and nothing else, because for a filter that
    // question *is* the operation: a filter rewrites the accumulator beneath it, so
    // baking it into the destination is the same picture exactly when the accumulator
    // beneath it is that destination alone. Which is `backdrop_is_dest` — the same
    // predicate a clipped layer answers, asked of a layer that is nothing but its
    // backdrop, rewritten.
    //
    // The first question — does the source reach the accumulator by plain "over"? —
    // has no bearing here. Nothing arrives to be stacked, so there is no
    // associativity to lean on and no mode to agree about; the filter's own blend is
    // refused by state (§21.4) and its clip is absorbed, being the identity for every
    // filter that may come this way.
    if let Some(f) = filter {
        // A filter that **resamples** is refused, and by a law rather than a
        // preference: the merged tiles are written by a pass that must be a pure
        // function of canvas position (§6.4), and a gather is not one at any apron
        // width. `Filter::resamples` is where that is decided, once.
        //
        // The destination must be **paint**, and — unless it is the carrier, which by
        // definition carries this filter — must carry nothing itself: what a group's
        // base composites to is not what the group composites to, so rewriting the
        // base would not be rewriting what the filter read.
        if !backdrop_is_dest || f.resamples() {
            return None;
        }
        if !d.content_is_paint() || (!site.dest_is_carrier && d.is_group()) {
            return None;
        }
        return Some(MergePlan {
            source: s.id,
            dest: d.id,
            kind: MergeKind::Filter {
                filter: f,
                source_params: s.composite,
            },
        });
    }

    // A clip is stated against the backdrop, so it can only be folded into a layer
    // that **is** the backdrop.
    if s.composite.clip && !backdrop_is_dest {
        return None;
    }

    let keeps = if site.dest_is_carrier {
        // Into the **carrier**. Everything about it survives untouched: its blend and
        // its clip point outward — they describe how the group meets what lies under
        // the group (§14.4.3) — and its opacity is applied to the group's composited
        // whole, which is exactly what this merge rewrites the inside of. So the
        // merge runs against the base's content at full strength and the slider stays
        // where it is, on the layer.
        //
        // The source may carry **any** mode here, and that is not a special case: the
        // group's isolated content is `merge_source(base, source)` before and after,
        // so what the group merges outward is unchanged whatever the source's mode is.
        if !d.content_is_paint() {
            return None;
        }
        d.composite
    } else {
        // Into a **sibling**. Here the destination is a leaf whose own opacity rides
        // on its tiles, so it folds in with the source's and the survivor stands at
        // full strength.
        //
        // The two must **agree** about how they meet the backdrop, because after the
        // merge one set of params speaks for both: same mode, and neither clipped
        // (the destination's clip would start applying to the source's paint, and the
        // source's is refused above unless the destination *is* its whole backdrop —
        // which a sibling with anything under it is not).
        //
        // Same-mode siblings merge because the modes are associative at any coverage
        // (§18.0.4): `merge(merge(B,D),S)` is `merge(B, merge(D,S))`, so the pair
        // composites as one layer carrying `merge(D,S)`. That was false until the
        // emissive modes stopped weighing coverage in the working space, and the merge
        // was refused for exactly that reason.
        if !is_plain_paint(d) || d.composite.clip {
            return None;
        }
        // At the foot of the root stack neither mode is stated against anything,
        // so they need not agree — both are the identity there, and so is
        // whichever one the survivor ends up wearing. Anywhere else one set of
        // params has to speak for both afterwards, so they must.
        //
        // `!=` and not "the same mode": a mode that carries parameters is a
        // *family* of curves (§18.0.4), and the associativity that carries this
        // merge is each curve's own — two `Drago`s at different bends are two
        // different functions, and folding one through the other's curve would be
        // exactly the silently-different picture a merge promises not to be.
        if !backdrop_is_dest && d.composite.blend != s.composite.blend {
            return None;
        }
        CompositeParams {
            blend: d.composite.blend,
            clip: false,
            opacity: 1.0,
        }
    };

    Some(MergePlan {
        source: s.id,
        dest: d.id,
        kind: MergeKind::Stack {
            source_params: s.composite,
            // A carrier's own content composites at full strength inside the
            // isolation; a sibling's rides its own slider (§14.7).
            dest_opacity: if site.dest_is_carrier {
                1.0
            } else {
                d.composite.opacity
            },
            keeps,
        },
    })
}

/// Paint that carries nothing: the only shape either side of a merge may take today.
fn is_plain_paint(l: &Layer) -> bool {
    l.content_is_paint() && !l.is_group()
}

impl Layer {
    /// Whether this layer's own content is painted tiles — a **matte** is neither
    /// merged nor merged into (§14.11), and a filter is merged but never merged into
    /// (§14.11.7), having no tiles to rewrite.
    fn content_is_paint(&self) -> bool {
        matches!(self.content, LayerContent::Paint(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{BlendMode, ChromaticAberration, ColorAdjust, DRAGO_K, Filter, Place};

    const A: LayerId = LayerId(0);
    const B: LayerId = LayerId(1);
    const C: LayerId = LayerId(2);
    const D: LayerId = LayerId(3);

    const MODES: [BlendMode; 3] = [
        BlendMode::Reinhard,
        BlendMode::Drago { k: DRAGO_K },
        BlendMode::Multiply,
    ];

    /// Three paint layers in the root stack, bottom-to-top: A, B, C.
    fn flat() -> DocState {
        DocState::with_layer(A)
            .insert_layer(B, None, Some(A))
            .insert_layer(C, None, Some(B))
    }

    /// The layer `source` would merge down onto, if any.
    fn dest(state: &DocState, source: LayerId) -> Option<LayerId> {
        plan(state, source).map(|p| p.dest)
    }

    /// What each side is worth and what the survivor keeps — the whole plan bar the
    /// two ids, which `dest` covers.
    ///
    /// `None` for a **filter** merge as well as for no merge at all, and that is the
    /// right shape rather than a lossy one: none of these three exists there (see
    /// [`MergeKind`]), and the filter merge has [`merged_filter`] to ask instead.
    fn terms(state: &DocState, source: LayerId) -> Option<(CompositeParams, f32, CompositeParams)> {
        match plan(state, source)?.kind {
            MergeKind::Stack {
                source_params,
                dest_opacity,
                keeps,
            } => Some((source_params, dest_opacity, keeps)),
            MergeKind::Filter { .. } => None,
        }
    }

    /// The filter a merge of `source` would run over its destination, if that is the
    /// merge on offer.
    fn merged_filter(state: &DocState, source: LayerId) -> Option<Filter> {
        match plan(state, source)?.kind {
            MergeKind::Filter { filter, .. } => Some(filter),
            MergeKind::Stack { .. } => None,
        }
    }

    fn faded(opacity: f32) -> CompositeParams {
        CompositeParams {
            opacity,
            ..CompositeParams::IDENTITY
        }
    }

    /// The ordinary case, and the one that must never grow a condition: plain layers
    /// merge onto the plain layer below, and the bottom of the document has nothing to
    /// merge into.
    #[test]
    fn a_plain_layer_merges_onto_the_plain_layer_below() {
        let state = flat();
        assert_eq!(dest(&state, C), Some(B));
        assert_eq!(dest(&state, B), Some(A));
        assert_eq!(dest(&state, A), None, "the foot of the stack has no `down`");
        assert_eq!(
            terms(&state, C),
            Some((CompositeParams::IDENTITY, 1.0, CompositeParams::IDENTITY)),
        );
    }

    /// Between siblings both sliders are folded into the merged tiles, so the survivor
    /// stands at full strength — the destination's is a leaf's, and a leaf's opacity
    /// rides on its tiles (§14.7).
    #[test]
    fn sibling_opacities_are_folded_into_the_tiles() {
        let state = flat().set_layer_opacity(C, 0.4).set_layer_opacity(B, 0.25);
        assert_eq!(
            terms(&state, C),
            Some((faded(0.4), 0.25, CompositeParams::IDENTITY)),
            "both sides expand at their own opacity and the survivor is left at 1",
        );
    }

    /// **Siblings sharing a blend mode merge**, and that turns on the modes being
    /// associative at any coverage (§18.0.4). While they weighed coverage in the
    /// working space they were not, and this was refused for exactly that reason.
    ///
    /// Modes that *disagree* are still refused, and always will be: after the merge one
    /// set of params speaks for both, and there is no third mode that means "multiply
    /// here and glow there".
    #[test]
    fn siblings_merge_when_their_modes_agree_and_not_otherwise() {
        for mode in MODES {
            let both = flat().set_layer_blend(C, mode).set_layer_blend(B, mode);
            assert_eq!(dest(&both, C), Some(B), "{mode:?} on both");
            assert_eq!(
                terms(&both, C).map(|(_, _, keeps)| keeps.blend),
                Some(mode),
                "the survivor has to go on meeting the backdrop the same way",
            );

            let source_only = flat().set_layer_blend(C, mode);
            assert_eq!(dest(&source_only, C), None, "{mode:?} on the source alone");
            let dest_only = flat().set_layer_blend(B, mode);
            assert_eq!(
                dest(&dest_only, C),
                None,
                "{mode:?} on the destination alone"
            );
        }
        // …and two *different* modes are no better than one.
        let mixed = flat()
            .set_layer_blend(C, BlendMode::Reinhard)
            .set_layer_blend(B, BlendMode::Multiply);
        assert_eq!(dest(&mixed, C), None);
    }

    /// **Two bends are two modes.** A mode's parameters are part of which function it
    /// is (§18.0.4), so `Radiance` at one bend and `Radiance` at another are as
    /// unmergeable as `Glow` and `Multiply` — associativity is each curve's own, and
    /// folding a layer through a curve that is not its own is exactly the silent
    /// change of picture a merge promises not to be.
    ///
    /// The one that would have been easy to get wrong: this is the test that fails if
    /// the agreement check is ever relaxed to "the same mode", which reads like a
    /// tidy-up and is a hole.
    #[test]
    fn siblings_at_different_bends_do_not_merge() {
        let apart = flat()
            .set_layer_blend(C, BlendMode::Drago { k: 0.3 })
            .set_layer_blend(B, BlendMode::Drago { k: 1.5 });
        assert_eq!(dest(&apart, C), None, "same mode, different curve");
        // The same pair at the same bend merges, so what is refused above is the
        // difference and not the parameter's presence.
        let together = flat()
            .set_layer_blend(C, BlendMode::Drago { k: 0.3 })
            .set_layer_blend(B, BlendMode::Drago { k: 0.3 });
        assert_eq!(dest(&together, C), Some(B));
    }

    /// At the foot of the root stack a blend mode is the identity — there is nothing
    /// under it to combine with — so neither layer's mode has to agree with anything.
    #[test]
    fn modes_need_not_agree_where_neither_is_stated_against_anything() {
        let state = flat()
            .set_layer_blend(A, BlendMode::Multiply)
            .set_layer_blend(B, BlendMode::Reinhard);
        assert_eq!(dest(&state, B), Some(A));
        // …but a clip on the bottom layer erases it rather than going inert, so it is
        // still refused.
        assert_eq!(dest(&state.set_layer_clip(A, true), B), None);
    }

    /// A clipped layer clips to **everything beneath it in its own stack** (§14.4), so
    /// it may only be folded into a destination that is the whole of that backdrop.
    #[test]
    fn a_clipped_layer_merges_only_where_the_destination_is_its_whole_backdrop() {
        // Second from the foot of the root stack: the accumulator holds A alone.
        let state = flat().set_layer_clip(B, true);
        assert_eq!(dest(&state, B), Some(A));
        assert!(
            terms(&state, B).is_some_and(|(source, _, keeps)| source.clip && !keeps.clip),
            "the clip is spent on the merge, not carried by the survivor",
        );
        // One row higher, C is clipped to A *and* B, which no merge into B can carry.
        assert_eq!(dest(&flat().set_layer_clip(C, true), C), None);
    }

    /// A group's members composite over its base (§14.1), so the bottom carried layer
    /// merges into the carrier — **whatever mode it carries**. The group's isolated
    /// content is `merge_source(base, source)` either way, so what the group merges
    /// outward is unchanged, and that is the whole argument.
    #[test]
    fn the_bottom_of_a_group_merges_into_its_base_under_any_mode() {
        let state = flat().move_layer(C, Some(B), Place::Top);
        for mode in MODES {
            let with_mode = state.set_layer_blend(C, mode);
            assert_eq!(dest(&with_mode, C), Some(B), "{mode:?} into a carrier");
            assert_eq!(
                terms(&with_mode, C).map(|(source, _, _)| source.blend),
                Some(mode),
                "the merge runs through the source's own mode",
            );
        }
        // A clipped member clips to exactly the base, which is the gesture "clip to
        // this one layer" is spelled with (§14.4).
        assert_eq!(dest(&state.set_layer_clip(C, true), C), Some(B));
    }

    /// The carrier's own params survive the merge untouched — all three of them.
    ///
    /// Its blend and its clip point outward, describing how the *group* meets what lies
    /// under it, and its opacity is applied to the group's composited whole. The merge
    /// rewrites the inside of that whole, so none of the three has anything to do with
    /// it — which is why the base expands at **1.0** rather than at its own slider.
    #[test]
    fn a_carrier_keeps_all_of_its_own_params() {
        let state = flat()
            .move_layer(C, Some(B), Place::Top)
            .set_layer_blend(B, BlendMode::Multiply)
            .set_layer_clip(B, true)
            .set_layer_opacity(B, 0.5);
        let carrier = state.layer(B).expect("the carrier exists").composite;
        assert_eq!(
            terms(&state, C),
            Some((CompositeParams::IDENTITY, 1.0, carrier)),
            "the base composites at full strength inside the group, and keeps its own",
        );
    }

    /// What sits beneath a layer whose lower sibling is a **group** is that whole
    /// group, not the group's base — so there is nothing here a two-tile merge can do.
    /// A group as the *source* is refused for the plainer reason that it is a subtree.
    #[test]
    fn groups_are_refused_on_both_sides() {
        // B carries A — the stack is [B[A], C], with C sitting above the whole group.
        let dest_group = flat().move_layer(A, Some(B), Place::Top);
        assert_eq!(dest(&dest_group, C), None, "the destination is a group");

        // The other side: [A, B[C]], so B is a group with plain paint directly under
        // it. Everything else about the pair is mergeable; that B is a subtree is the
        // whole of what refuses it.
        let source_group = flat().move_layer(C, Some(B), Place::Top);
        assert_eq!(dest(&source_group, B), None, "the source is a group");
        // …and the member inside it still merges into its base, which is what says the
        // refusal above is about B being a subtree rather than about the tree having
        // become unreadable here.
        assert_eq!(dest(&source_group, C), Some(B));
    }

    /// Merging across a difference in visibility would reveal hidden paint or hide
    /// visible paint; two hidden layers show nothing either way and merge fine.
    #[test]
    fn visibility_has_to_match() {
        assert_eq!(dest(&flat().set_layer_visible(C, false), C), None);
        assert_eq!(dest(&flat().set_layer_visible(B, false), C), None);
        assert_eq!(
            dest(
                &flat()
                    .set_layer_visible(C, false)
                    .set_layer_visible(B, false),
                C,
            ),
            Some(B),
        );
    }

    /// A **matte** has no tile map, so it is neither merged nor merged into (§15.2) —
    /// the same refusal a stroke aimed at one gets — and neither is a **filter**
    /// merged *into*, for the same reason read the other way: there are no channels
    /// there to rewrite. (A filter as the *source* is the second kind of merge, and
    /// the tests below are its.)
    #[test]
    fn only_paint_is_merged_into() {
        use crate::document::MatteRegion;
        use crate::geom::Vec2;

        let region = MatteRegion::OutsideRect {
            min: Vec2::ZERO,
            max: Vec2::splat(100.0),
        };
        let over_matte = DocState::with_layer(A)
            .insert_matte(
                B,
                None,
                crate::document::Place::Above(A),
                region,
                crate::document::MattePaint::Solid([1.0; 3]),
            )
            .insert_layer(C, None, Some(B));
        assert_eq!(dest(&over_matte, C), None, "a matte destination");
        assert_eq!(dest(&over_matte, B), None, "a matte source");

        let over_filter = DocState::with_layer(A)
            .insert_filter(B, None, Some(A), GREY)
            .insert_layer(C, None, Some(B));
        assert_eq!(dest(&over_filter, C), None, "a filter destination");
    }

    /// A point filter, dialled — the merge has to be offered for a filter that does
    /// something, since a neutral one is dropped from the draw list entirely (§21.3).
    const GREY: Filter = Filter::Color(ColorAdjust {
        saturation: 0.0,
        ..ColorAdjust::NEUTRAL
    });

    /// A filter that reads its neighbours (§21.10).
    const FRINGE: Filter = Filter::Chromatic(ChromaticAberration {
        spread: 8.0,
        angle: 0.0,
    });

    /// **A filter merges exactly where its backdrop is the destination alone**
    /// (§14.11.7) — the same predicate §14.11.2 asks of every source, and for a
    /// filter the only one that binds.
    ///
    /// Both positions, because they are two different sentences that happen to agree:
    /// carried onto a layer, the base composites at the bottom of the group it opens
    /// (§14.1); second from the foot of the root, the accumulator starts cleared.
    #[test]
    fn a_filter_merges_where_its_backdrop_is_its_destination() {
        let carried = DocState::with_layer(A).insert_filter(B, Some(A), None, GREY);
        assert_eq!(dest(&carried, B), Some(A), "carried onto a painted layer");
        assert_eq!(merged_filter(&carried, B), Some(GREY));

        let foot = DocState::with_layer(A).insert_filter(B, None, Some(A), GREY);
        assert_eq!(dest(&foot, B), Some(A), "second from the foot of the root");
        assert_eq!(merged_filter(&foot, B), Some(GREY));
    }

    /// Anywhere else it is filtering more than the destination, and baking it into
    /// that one layer would change the picture — the merge law itself (§14.11).
    #[test]
    fn a_filter_above_more_than_its_destination_does_not_merge() {
        // Root [A, B, F]: the filter reads A composited with B, not B alone.
        let state = flat().insert_filter(D, None, Some(C), GREY);
        assert_eq!(dest(&state, D), None, "third from the foot of the root");

        // Root [A, G[X], F]: the destination is a group, so what the filter read is
        // the group's composite and rewriting the base is not rewriting that.
        let over_group = DocState::with_layer(A)
            .insert_layer(C, Some(A), None)
            .insert_filter(B, None, Some(A), GREY);
        assert_eq!(dest(&over_group, B), None, "a group as the destination");
    }

    /// **A resampling filter never merges**, in either position — the refusal
    /// `Filter::resamples` exists to make (§14.11.7, §6.4). Not a judgement about the
    /// picture but about the pass: a gather is not a function of canvas position, and
    /// a tile's apron is only as wide as one.
    #[test]
    fn a_resampling_filter_never_merges() {
        let carried = DocState::with_layer(A).insert_filter(B, Some(A), None, FRINGE);
        assert_eq!(dest(&carried, B), None, "carried onto a painted layer");

        let foot = DocState::with_layer(A).insert_filter(B, None, Some(A), FRINGE);
        assert_eq!(dest(&foot, B), None, "second from the foot of the root");
    }

    /// **The destination keeps everything**, and the filter's own strength travels.
    ///
    /// Two halves of one claim, which is why they are one test. A filter merge
    /// rewrites channels and stacks nothing, so there is no reconciling to do and
    /// [`terms`] has nothing to report — while the *strength* is exactly what a merge
    /// that dropped it would get silently wrong, turning a half-applied grade into a
    /// fully applied one with the layer that said so already gone.
    #[test]
    fn a_filter_merge_bakes_its_strength_and_reconciles_nothing() {
        let state = DocState::with_layer(A)
            .insert_filter(B, Some(A), None, GREY)
            .set_layer_opacity(B, 0.4)
            .set_layer_opacity(A, 0.25)
            .set_layer_blend(A, BlendMode::Multiply);
        assert_eq!(
            terms(&state, B),
            None,
            "a filter merge reconciles no params"
        );
        let MergeKind::Filter {
            filter,
            source_params,
        } = plan(&state, B).expect("the merge is offered").kind
        else {
            unreachable!("a filter source plans a filter merge")
        };
        assert_eq!(filter, GREY);
        assert_eq!(
            source_params.opacity, 0.4,
            "the strength has to be baked in"
        );
        // The destination's own params are not in the plan at all, which is the
        // structural half: there is no field here that could carry them wrongly.
        let after = state.layer(A).expect("the destination");
        assert_eq!(after.composite.opacity, 0.25);
        assert_eq!(after.composite.blend, BlendMode::Multiply);
    }
}
