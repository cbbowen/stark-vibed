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
//! **Two layers sharing a blend mode.** The tempting rule — "same mode, so merge them
//! in that mode" — is false for the light-combining modes, and quietly. Their blend
//! *functions* are associative (that is the point of building each as a conjugation of
//! addition, [`BlendMode`]), but the Porter-Duff wrapper around them is not: the
//! middle term applies the blend function to the accumulator's **coverage-averaged**
//! colour, and averaging does not commute with a curve. Two 50%-covered
//! [`Reinhard`](BlendMode::Reinhard) layers over a backdrop come out at 0.6446
//! stacked and 0.6442 merged — a difference no golden would flag as a *rule* being
//! wrong, only as one image being slightly off. ([`Multiply`](BlendMode::Multiply)
//! happens to survive, its blend function being bilinear; a rule that holds for one
//! mode of four is a special case, not a law, and it is not worth a shader path that
//! has to be right about which.)
//!
//! **A source with a blend mode, merged into its carrier.** This one *is* sound — the
//! group's isolated content is unchanged by construction — but it needs the blend
//! algebra evaluated in tile space and inverted back through the slab law, and
//! `blend_common.wesl` owns bindings that a tile pass cannot inherit. Left out rather
//! than approximated; the layers it covers keep their Merge control hidden.
//!
//! **Groups.** A source that carries layers would have to flatten a whole subtree, and
//! a destination that carries layers is not what sits beneath the source — its whole
//! group is. Both are refused rather than partially handled.

use super::layer::{Layer, LayerContent, LayerId};
use super::state::DocState;

/// How the source layer's paint lands on the destination's — the one thing the tile
/// pass has to be told, and the only shape of merge each case admits.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MergeKind {
    /// Premultiplied "over": the source stacks on the destination, heights adding.
    /// What an unclipped `Normal` layer does to everything below it.
    Over,
    /// The source **clipped** to the destination (§14.4): deleted where the
    /// destination has no coverage, and its height suppressed with it, so no relief
    /// is left lighting over paint that is not there.
    Clip,
}

/// The merge a [`plan`] found: which layer is consumed, which survives, and how the
/// paint lands.
///
/// The destination is derived rather than chosen — "down" names exactly one layer —
/// but it is carried anyway, because a [`Footprint`] is built from the action alone
/// and cannot go looking for it (§12.6). The applying side re-derives the plan and
/// declines if it names a different destination, which is what keeps a peer's action
/// honest against a tree that has moved under it.
///
/// [`Footprint`]: super::footprint::Footprint
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MergePlan {
    pub source: LayerId,
    pub dest: LayerId,
    pub kind: MergeKind,
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
    // The source has to be paint that carries nothing, and has to reach the
    // accumulator by "over" — a mode of its own is refused here, before either
    // position case, because neither can absorb it (see the header).
    if !is_plain_paint(s) || !s.composite.blend.is_normal() {
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

    let kind = if s.composite.clip {
        MergeKind::Clip
    } else {
        MergeKind::Over
    };
    // A clip is stated against the backdrop, so it can only be folded into a layer
    // that *is* the backdrop.
    if kind == MergeKind::Clip && !backdrop_is_dest {
        return None;
    }

    match site.carrier {
        // Into the **carrier**. Its blend and its clip point outward — they describe
        // how the group meets what lies under the group (§14.4.3) — and the group's
        // isolated content is exactly what this merge rewrites, so both survive
        // untouched whatever they are.
        //
        // Its **opacity** does not, and that is why it is pinned: a group's opacity is
        // applied to the composited whole at the merge (§14.7), which is not something
        // a tile can carry, so the merged layer would have to keep it — and the source's
        // paint, which was never scaled by it, would start being.
        Some(_) if site.index == 0 => {
            if !d.content_is_paint() || d.composite.opacity != 1.0 {
                return None;
            }
        }
        // Into a **sibling**. Here the destination is a leaf whose own opacity rides on
        // its tiles, so it folds in with the source's; what it may not have is a
        // relational property of its own, since after the merge that property would
        // apply to the source's paint too.
        //
        // The exception is the foot of the root stack, where there is no backdrop for
        // either to be stated against: a mode there is the identity and stays the
        // identity, which is the same fact `LayerInfo::has_backdrop` reports (§14.4.3).
        // A *clip* there is not inert — it erases the layer — so it is still refused.
        _ => {
            if !is_plain_paint(d) || d.composite.clip {
                return None;
            }
            if !backdrop_is_dest && !d.composite.blend.is_normal() {
                return None;
            }
        }
    }

    Some(MergePlan { source, dest, kind })
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

    /// Three paint layers in the root stack, bottom-to-top: A, B, C.
    fn flat() -> DocState {
        DocState::with_layer(A)
            .insert_layer(B, None, Some(A))
            .insert_layer(C, None, Some(B))
    }

    fn kind(state: &DocState, source: LayerId) -> Option<(LayerId, MergeKind)> {
        plan(state, source).map(|p| (p.dest, p.kind))
    }

    /// The ordinary case, and the one that must never grow a condition: plain layers
    /// merge onto the plain layer below, and the bottom of the document has nothing to
    /// merge into.
    #[test]
    fn a_plain_layer_merges_onto_the_plain_layer_below() {
        let state = flat();
        assert_eq!(kind(&state, C), Some((B, MergeKind::Over)));
        assert_eq!(kind(&state, B), Some((A, MergeKind::Over)));
        assert_eq!(kind(&state, A), None, "the foot of the stack has no `down`");
    }

    /// Opacity is folded into the merged tiles, so a faded layer is mergeable — the
    /// one relational property that is not a property at all.
    #[test]
    fn opacity_does_not_stop_a_merge_between_siblings() {
        let state = flat().set_layer_opacity(C, 0.4).set_layer_opacity(B, 0.25);
        assert_eq!(kind(&state, C), Some((B, MergeKind::Over)));
    }

    /// A blend mode on **either** side is refused between siblings: the source's
    /// because "over" is the only law this merges by, the destination's because after
    /// the merge it would apply to the source's paint as well.
    ///
    /// Sharing a mode does not help, which is the rule this test exists to pin — see
    /// the module header for why it is false rather than merely unimplemented.
    #[test]
    fn a_blend_mode_refuses_a_sibling_merge_even_when_both_share_it() {
        for mode in [BlendMode::Reinhard, BlendMode::Drago, BlendMode::Multiply] {
            let source = flat().set_layer_blend(C, mode);
            assert_eq!(kind(&source, C), None, "{mode:?} on the source");

            let dest = flat().set_layer_blend(B, mode);
            assert_eq!(kind(&dest, C), None, "{mode:?} on the destination");

            let both = flat().set_layer_blend(C, mode).set_layer_blend(B, mode);
            assert_eq!(kind(&both, C), None, "{mode:?} on both");
        }
    }

    /// At the foot of the root stack a blend mode is the identity — there is nothing
    /// under it to combine with — so it neither blocks the merge nor changes it.
    #[test]
    fn a_mode_on_the_bottom_layer_is_inert_and_allows_the_merge() {
        let state = flat().set_layer_blend(A, BlendMode::Multiply);
        assert_eq!(kind(&state, B), Some((A, MergeKind::Over)));
        // …but a clip there erases the layer rather than going inert, so it is still
        // refused.
        let clipped = state.set_layer_clip(A, true);
        assert_eq!(kind(&clipped, B), None);
    }

    /// A clipped layer clips to **everything beneath it in its own stack** (§14.4), so
    /// it may only be folded into a destination that is the whole of that backdrop.
    #[test]
    fn a_clipped_layer_merges_only_where_the_destination_is_its_whole_backdrop() {
        // Second from the foot of the root stack: the accumulator holds A alone.
        let state = flat().set_layer_clip(B, true);
        assert_eq!(kind(&state, B), Some((A, MergeKind::Clip)));
        // One row higher, C is clipped to A *and* B, which no merge into B can carry.
        let higher = flat().set_layer_clip(C, true);
        assert_eq!(kind(&higher, C), None);
    }

    /// A group's members composite over its base (§14.1), so the bottom carried layer
    /// merges into the carrier — and does so whatever the carrier's own blend and clip
    /// are, since those describe how the *group* meets what lies under it and the
    /// group's content is exactly what this rewrites.
    #[test]
    fn the_bottom_of_a_carried_stack_merges_into_its_carrier() {
        let state = flat().move_layer(C, Some(B), Place::Top);
        assert_eq!(kind(&state, C), Some((B, MergeKind::Over)));

        let styled = state
            .set_layer_blend(B, BlendMode::Multiply)
            .set_layer_clip(B, true);
        assert_eq!(
            kind(&styled, C),
            Some((B, MergeKind::Over)),
            "the carrier's outward properties are untouched by the merge",
        );
        // And a clipped member clips to exactly the base, which is the gesture
        // "clip to this one layer" is spelled with (§14.4).
        let clipped = state.set_layer_clip(C, true);
        assert_eq!(kind(&clipped, C), Some((B, MergeKind::Clip)));
    }

    /// A group's opacity is applied to its composited whole, which a tile cannot carry
    /// — so the one property of a carrier that blocks the merge is that one.
    #[test]
    fn a_faded_carrier_refuses_the_merge() {
        let state = flat()
            .move_layer(C, Some(B), Place::Top)
            .set_layer_opacity(B, 0.5);
        assert_eq!(kind(&state, C), None);
    }

    /// What sits beneath a layer whose lower sibling is a **group** is that whole
    /// group, not the group's base — so there is nothing here a two-tile merge can do.
    /// A group as the *source* is refused for the plainer reason that it is a subtree.
    #[test]
    fn groups_are_refused_on_both_sides() {
        // B carries A — the stack is [B[A], C], with C sitting above the whole group.
        let dest_group = flat().move_layer(A, Some(B), Place::Top);
        assert_eq!(kind(&dest_group, C), None, "the destination is a group");

        // The other side: [A, B[C]], so B is a group with plain paint directly under
        // it. Everything else about the pair is mergeable; that B is a subtree is the
        // whole of what refuses it.
        let source_group = flat().move_layer(C, Some(B), Place::Top);
        assert_eq!(kind(&source_group, B), None, "the source is a group");
        // …and the member inside it still merges into its base, which is what says the
        // refusal above is about B being a subtree rather than about the tree having
        // become unreadable here.
        assert_eq!(kind(&source_group, C), Some((B, MergeKind::Over)));
    }

    /// Merging across a difference in visibility would reveal hidden paint or hide
    /// visible paint; two hidden layers show nothing either way and merge fine.
    #[test]
    fn visibility_has_to_match() {
        assert_eq!(kind(&flat().set_layer_visible(C, false), C), None);
        assert_eq!(kind(&flat().set_layer_visible(B, false), C), None);
        assert_eq!(
            kind(
                &flat()
                    .set_layer_visible(C, false)
                    .set_layer_visible(B, false),
                C,
            ),
            Some((B, MergeKind::Over)),
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
        assert_eq!(kind(&over_matte, C), None, "a matte destination");
        assert_eq!(kind(&over_matte, B), None, "a matte source");

        let with_filter = DocState::with_layer(A)
            .insert_filter(B, None, Some(A), Filter::Color(ColorAdjust::NEUTRAL))
            .insert_layer(C, None, Some(B));
        assert_eq!(kind(&with_filter, C), None, "a filter destination");
        assert_eq!(kind(&with_filter, B), None, "a filter source");
    }
}
