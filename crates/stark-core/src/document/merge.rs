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
//! # What is deliberately refused
//!
//! **Two layers sharing a blend mode — for now, and no longer on principle.** This
//! was refused as *unsound*: the blend functions are associative by construction, but
//! the Porter-Duff wrapper applied them to the accumulator's coverage-averaged
//! colour, and averaging does not commute with a curve, so a same-mode merge was only
//! approximately the stack. That is fixed at the source rather than worked around
//! here — the emissive modes now weigh coverage in emission, where the combination is
//! addition, and are associative at any coverage (§18.0.4, `blend_common::added_light`).
//!
//! So the merge is now *sound* and merely unimplemented, which is a different
//! sentence and a smaller one. What it still needs is the mode's algebra evaluated in
//! **tile** space: the light conversion is per colour space, and in a pigment document
//! that means binding the Mixbox LUT into a pass that has never needed it. The slab-law
//! inversion the result then has to be stored through already exists
//! (`merge.wesl::optical_mass`).
//!
//! **A source with a blend mode, merged into its carrier.** Sound for a different and
//! simpler reason — the group's isolated content is what the merge rewrites, so the
//! outward merge is unchanged whatever the mode — and blocked on the same tile-space
//! conversion. The two arrive together when it does.
//!
//! **Groups.** A source that carries layers would have to flatten a whole subtree, and
//! a destination that carries layers is not what sits beneath the source — its whole
//! group is. Both are refused rather than partially handled.

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
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MergePlan {
    pub source: LayerId,
    pub dest: LayerId,
    /// How the source's paint meets the destination's — **the source layer's own**
    /// params, always. A merge folds the upper layer into the lower through exactly
    /// the merge the compositor would have run between the two, which is the whole of
    /// why the result is the same picture.
    ///
    /// Its `opacity` is the source's own, and it is folded into the merged tiles.
    pub source_params: CompositeParams,
    /// What the destination's tiles are worth on their own: its own opacity when the
    /// two are siblings, and **1.0 when the destination is the carrier** — a group's
    /// base composites at [`CompositeParams::IDENTITY`] inside the isolation, its
    /// slider belonging to the group as a whole (§14.7).
    pub dest_opacity: f32,
    /// What the surviving layer's params become. The destination's own, untouched,
    /// when it is a carrier; otherwise the pair's shared mode at full opacity, both
    /// sliders having been folded into the tiles.
    pub keeps: CompositeParams,
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
    // Both sides have to be paint that carries nothing — the source because a subtree
    // cannot travel as one tile, the destination because what sits beneath the source
    // would then be a whole group rather than that layer's content.
    if !is_plain_paint(s) {
        return None;
    }

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
    let d = state.layer(dest)?;

    // Hiding a layer hides what it carries (§14.3), so a merge across a difference in
    // visibility would either reveal paint that is hidden or hide paint that is not.
    // Two hidden layers merge fine: nothing shows either way, before or after.
    if s.visible != d.visible {
        return None;
    }
    // A clip is stated against the backdrop, so it can only be folded into a layer
    // that **is** the backdrop.
    if s.composite.clip && !backdrop_is_dest {
        return None;
    }

    let keeps = match site.carrier {
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
        Some(_) if site.index == 0 => {
            if !d.content_is_paint() {
                return None;
            }
            d.composite
        }
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
        _ => {
            if !is_plain_paint(d) || d.composite.clip {
                return None;
            }
            // At the foot of the root stack neither mode is stated against anything,
            // so they need not agree — both are the identity there, and so is
            // whichever one the survivor ends up wearing. Anywhere else one set of
            // params has to speak for both afterwards, so they must.
            if !backdrop_is_dest && d.composite.blend != s.composite.blend {
                return None;
            }
            CompositeParams {
                blend: d.composite.blend,
                clip: false,
                opacity: 1.0,
            }
        }
    };

    Some(MergePlan {
        source,
        dest,
        source_params: s.composite,
        // A carrier's own content composites at full strength inside the isolation;
        // a sibling's rides its own slider (§14.7).
        dest_opacity: if site.index == 0 && site.carrier.is_some() {
            1.0
        } else {
            d.composite.opacity
        },
        keeps,
    })
}

/// Paint that carries nothing: the only shape either side of a merge may take today.
fn is_plain_paint(l: &Layer) -> bool {
    l.content_is_paint() && !l.is_group()
}

impl Layer {
    /// Whether this layer's own content is painted tiles — a matte and a filter are
    /// neither merged nor merged into (§14.11).
    fn content_is_paint(&self) -> bool {
        matches!(self.content, LayerContent::Paint(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{BlendMode, Place};

    const A: LayerId = LayerId(0);
    const B: LayerId = LayerId(1);
    const C: LayerId = LayerId(2);

    const MODES: [BlendMode; 3] = [BlendMode::Reinhard, BlendMode::Drago, BlendMode::Multiply];

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
    fn terms(state: &DocState, source: LayerId) -> Option<(CompositeParams, f32, CompositeParams)> {
        plan(state, source).map(|p| (p.source_params, p.dest_opacity, p.keeps))
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

    /// Neither a matte nor a filter has a tile map, so neither is merged nor merged
    /// into (§15.2, §21) — the same refusal a stroke aimed at one gets.
    #[test]
    fn only_paint_merges() {
        use crate::document::{ColorAdjust, Filter, MatteRegion};
        use crate::geom::Vec2;

        let region = MatteRegion::OutsideRect {
            min: Vec2::ZERO,
            max: Vec2::splat(100.0),
        };
        let over_matte = DocState::with_layer(A)
            .insert_matte(B, None, Some(A), region, [1.0; 3])
            .insert_layer(C, None, Some(B));
        assert_eq!(dest(&over_matte, C), None, "a matte destination");
        assert_eq!(dest(&over_matte, B), None, "a matte source");

        let with_filter = DocState::with_layer(A)
            .insert_filter(B, None, Some(A), Filter::Color(ColorAdjust::NEUTRAL))
            .insert_layer(C, None, Some(B));
        assert_eq!(dest(&with_filter, C), None, "a filter destination");
        assert_eq!(dest(&with_filter, B), None, "a filter source");
    }
}
