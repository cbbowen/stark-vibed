//! Merging a layer **down** onto the one beneath it (§14.11): when the pair can be
//! replaced by one layer, and what the survivor's params become when it can.
//!
//! Pure CPU — [`crate::gpu::merge`] does the tiles. The one law:
//!
//! > **A merge must not change what the document looks like.**
//!
//! It cannot always be obeyed, which is why [`plan`] returns an `Option`.
//!
//! # What has to hold
//!
//! Write `B` for everything composited beneath the pair, `D` for the lower layer (the
//! **destination**, which survives) and `S` for the upper (the **source**, which is
//! consumed). The document shows `merge_S(merge_D(B, D), S)` and must go on showing
//! `merge_D(B, D ⊕ S)` for **every** backdrop `B`. That splits in two:
//!
//! - **Do the two meet the backdrop by the same law?** Only then is `⊕` that law, and
//!   only then does its associativity (§18.0.4) carry the result across. A clip is a
//!   deletion rather than a stack, so it never associates.
//! - **Is the backdrop `S` is defined against exactly `D`?** It is `D` alone in exactly
//!   two places: `S` is the bottom of the stack its carrier `D` opens (a group's members
//!   composite over its base, §14.1), or `S` sits second from the bottom of the **root**
//!   stack, whose accumulator starts cleared.
//!
//! A **filter layer** source is the other kind of merge (§14.11.7): it stacks nothing,
//! being a function of what it sits on, so the merge *rewrites* the destination's
//! channels. Only the second question binds it, and it binds absolutely. It carries one
//! refusal of its own: the merged tiles are written by a pass that must be a pure
//! function of canvas position (§6.4), which a filter reading *neighbouring* texels
//! (§21.10) is not at any apron width.
//!
//! # What is deliberately refused
//!
//! **Groups**, on either side: a source that carries layers would have to flatten a
//! subtree, and what sits beneath a source is a destination group's *whole* group
//! rather than its base. (A **carrier** as the destination is not this case and is
//! offered, §14.1.)
//!
//! **A filter as the destination, and a matte as either**: neither holds tiles. A solid
//! matte is left out on purpose rather than overlooked — a *gradient* matte has no such
//! answer, the adjustment being nonlinear, so filtering the stops is not filtering the
//! ramp.

use super::layer::{CompositeParams, Layer, LayerContent};
use super::state::DocState;
use stark_model::document::LayerId;

/// The merge a [`plan`] found: which layer is consumed, which survives, how the paint
/// lands, and what each side's tiles are worth on their own.
///
/// The destination is derived rather than chosen — "down" names exactly one layer —
/// but it is carried anyway, because a [`Footprint`] is built from the action alone and
/// cannot go looking for it (§12.6). The applying side re-derives the plan and declines
/// if it names a different destination.
///
/// [`Footprint`]: stark_model::document::Footprint
#[derive(Clone, Debug, PartialEq)]
pub struct MergePlan {
    pub source: LayerId,
    pub dest: LayerId,
    /// Which of the two merges this is, and everything that one of them needs.
    pub kind: MergeKind,
}

/// The two things a merge can be, which are two because a source can be two things.
///
/// A layer of paint is **stacked into** the destination, and every number in
/// [`Stack`](Self::Stack) reconciles the two layers' params into the one set that
/// speaks for both afterwards. A **filter** stacks nothing, so merging it **rewrites**
/// the destination's channels and there is nothing to reconcile.
#[derive(Clone, Debug, PartialEq)]
pub enum MergeKind {
    /// The source's paint stacked into the destination's, through the source's own
    /// params (§14.11.3).
    Stack {
        /// How the source's paint meets the destination's — **the source layer's own**
        /// params, always, since the same picture requires exactly the merge the
        /// compositor would have run between the two. Its `opacity` is folded into the
        /// merged tiles.
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
    /// **No `dest_opacity` and no `keeps`**: the destination keeps every one of its
    /// params, and its opacity never enters the arithmetic — pass A's slab law scales
    /// coverage and height and leaves the color alone, so the un-premultiplied color
    /// the filter is defined on is the same whatever the slider says.
    Filter {
        filter: stark_model::document::Filter,
        /// The filter layer's own params, which is what the **compositor** reads of one
        /// (§21.4): the opacity is the filter's strength and has to be baked in, or a
        /// half-applied grade would merge to a fully applied one.
        ///
        /// Carried whole rather than as that one number, so this path and the screen's
        /// build the draw through the same `FilterDraw::new`. The clip rides along
        /// inert, being the identity for every filter that may come this way (§21.4.1).
        source_params: CompositeParams,
    },
}

/// The merge of `source` into the layer beneath it, or `None` when there is none that
/// preserves the document's appearance — see the module header for which is which.
///
/// A **pure function of the state**, which is what lets the action carry only the two
/// ids: every peer and every replay asks this of the same document and gets the same
/// answer, so a merge is accepted or declined identically everywhere.
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

/// Where a candidate source sits — the pure input [`plan_at`] decides from.
///
/// Reach for this over [`plan`] when the caller is already walking in composite order:
/// it knows the lower sibling and the carrier already, where `plan`'s own `site_of`
/// searches the whole tree — once per layer, the control being offered per row (§14.11).
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

/// The merge on offer at a located site, or `None` when there is none that preserves
/// the document's appearance — [`plan`] without the search. Reads nothing but the site,
/// so a caller that already has one need not consult the tree again.
pub(crate) fn plan_at(site: &MergeSite<'_>) -> Option<MergePlan> {
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

    // A filter rewrites the accumulator beneath it, so baking it into the destination
    // is the same picture exactly when that accumulator is the destination alone
    // (§14.11.7). Nothing arrives to be stacked, so the first question is vacuous.
    if let Some(f) = filter {
        // A resampling filter is refused by a law rather than a preference: the merged
        // tiles are written by a pass that must be a pure function of canvas position
        // (§6.4), and a gather is not one at any apron width.
        //
        // Unless the destination is the carrier — which by definition carries this
        // filter — it must carry nothing itself: what a group's base composites to is
        // not what the group composites to.
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
        // Into the **carrier**. Its blend and clip point outward, describing how the
        // group meets what lies under it (§14.4.3), and its opacity applies to the
        // group's composited whole — whose inside is exactly what this rewrites — so
        // all three survive untouched and the base merges at full strength. The source
        // may carry **any** mode: the group's isolated content is
        // `merge_source(base, source)` either way.
        if !d.content_is_paint() {
            return None;
        }
        d.composite
    } else {
        // Into a **sibling**. The destination is a leaf whose own opacity rides on its
        // tiles, so it folds in with the source's and the survivor stands at full
        // strength. The two must **agree** about how they meet the backdrop, one set of
        // params speaking for both afterwards: the same mode — which merges because the
        // modes are associative at any coverage (§18.0.4) — and neither clipped, since
        // the destination's clip would start applying to the source's paint.
        if !is_plain_paint(d) || d.composite.clip {
            return None;
        }
        // At the foot of the root stack neither mode is stated against anything, so
        // they need not agree. `!=` and not "the same mode": a mode that carries
        // parameters is a *family* of curves, and the associativity above is each
        // curve's own — two `Drago`s at different bends are two different functions.
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
    use stark_model::Srgb;
    use stark_model::document::{
        BlendMode, ChromaticAberration, ColorAdjust, DRAGO_K, Filter, Place,
    };

    const A: LayerId = LayerId::ROOT;
    const B: LayerId = LayerId::solo(1);
    const C: LayerId = LayerId::solo(2);
    const D: LayerId = LayerId::solo(3);

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

    /// What each side is worth and what the survivor keeps — the whole plan bar the two
    /// ids, which `dest` covers. `None` for a **filter** merge as well as for no merge
    /// at all: none of these three exists there — ask [`merged_filter`] instead.
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

    /// **Siblings sharing a blend mode merge**, which turns on the modes being
    /// associative at any coverage (§18.0.4). Modes that *disagree* are refused: one
    /// set of params speaks for both afterwards, and there is no third mode meaning
    /// "multiply here and glow there".
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
    /// unmergeable as `Glow` and `Multiply`. Fails if the agreement check is ever
    /// relaxed to "the same mode", which reads like a tidy-up and is a hole.
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

    /// The carrier's own params survive the merge untouched — all three of them. Its
    /// blend and clip point outward and its opacity applies to the group's composited
    /// whole, whose *inside* is what the merge rewrites, which is why the base expands
    /// at **1.0** rather than at its own slider.
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

    /// A **matte** has no tile map, so it is neither merged nor merged into (§15.2),
    /// and a **filter** is never merged *into* for the same reason: no channels to
    /// rewrite. (A filter as the *source* is the second kind of merge, tested below.)
    #[test]
    fn only_paint_is_merged_into() {
        use stark_model::document::MatteRegion;
        use stark_model::geom::Vec2;

        let region = MatteRegion::OutsideRect {
            min: Vec2::ZERO,
            max: Vec2::splat(100.0),
        };
        let over_matte = DocState::with_layer(A)
            .insert_matte(
                B,
                None,
                stark_model::document::Place::Above(A),
                region,
                stark_model::document::Parcel::Solid(Srgb::new([1.0; 3])),
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
    /// (§14.11.7), in both positions: carried onto a layer, whose base its members
    /// composite over (§14.1); and second from the foot of the root, whose accumulator
    /// starts cleared.
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

    /// **The destination keeps everything**, and the filter's own strength travels: a
    /// merge that dropped it would turn a half-applied grade into a fully applied one,
    /// with the layer that said so already gone.
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
