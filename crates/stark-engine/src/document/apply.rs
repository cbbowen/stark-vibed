//! Applying an action: the fold that turns the log into tiles (§4, §5).
//!
//! [`Action`] itself — what the log carries, what a peer receives, what a file
//! stores — is `stark-model`'s. This is the other half of the same sentence: what
//! *happens* when one is applied, which needs the renderers, the tile pool and the
//! canvas substrates, and so cannot live where the action does.
//!
//! The two are tied together by [`Materialize`]:
//! the model owns *that* an action folds over some state and which actions commute,
//! and [`DocState`] is this crate's answer to what the state is. The orphan rule
//! forced that shape and it is the right one — see §2.

use std::rc::Rc;
use std::sync::Arc;

use crate::gpu::substrate::Substrate;
use stark_model::document::ActorId;
use stark_model::document::{
    Action, ActionId, ActionKind, Footprint, Materialize, Resource, compute_footprint,
};

use super::layer::Layer;
use super::liquify::LiquifyRun;
use super::state::DocState;
use crate::gpu::selection::SelectionRenderer;
use crate::gpu::stroke::StrokeRenderer;
use crate::gpu::tile::TilePool;
use stark_model::document::LayerId;

/// Side-channel passed through [`Materialize::fold`]: the GPU resources needed
/// to render a stroke (§5). It owns cheap `Arc`-backed clones, so it
/// has no borrow lifetime — which is what lets it be the `Action::Context`.
///
/// `Clone` for the same reason it has no lifetime: every field is a handle, so a
/// copy is a fistful of refcount bumps and shares the thing rather than doubling
/// it. That is exactly what a *preview* engine wants (`Engine::new_sharing`,
/// §11) — and cloning the context whole is what keeps a renderer added here from
/// being shared by everything except the one constructor that listed its
/// siblings by hand.
pub struct ApplyCtx {
    pub(crate) pool: TilePool,
    pub(crate) stroke: StrokeRenderer,
    pub(crate) assets: crate::assets::AssetStore,
    pub(crate) selection: SelectionRenderer,
    pub(crate) transform: crate::gpu::transform::TransformRenderer,
    pub(crate) fill: crate::gpu::fill::FillRenderer,
    pub(crate) merge: crate::gpu::merge::MergeRenderer,
    /// Builds the tiles of an image brought in from outside the document (§23).
    /// The one renderer here holding no pipeline, because that path has no pass.
    pub(crate) place: crate::gpu::place::PlaceRenderer,
    /// The pictures a `PlaceImage` names (§23) — the third content-addressed store,
    /// beside `assets` and `substrates` and for their reason: the log carries the id
    /// and the pixels ride here.
    pub(crate) pictures: crate::pictures::PictureStore,
    /// The device, so a canvas substrate can be built here on demand.
    pub(crate) gpu: crate::gpu::context::GpuContext,
    /// The canvas substrates and the bytes registered for them (§6.4).
    ///
    /// It lives here, rather than beside the compositor it also feeds, because the
    /// **deposit reads it**: the tooth gates the paint a stroke lays by the substrate
    /// under it, so which substrate a stroke sees is part of applying that stroke.
    /// Asked with [`DocState::substrate`](super::state::DocState) and its scale *as the
    /// log stood at that action*, which is the whole reason `SetSubstrate` was made a
    /// logged action rather than a view setting — a stroke from before a mid-document
    /// switch deposits against the substrate it was actually painted on, at the size it
    /// was laid at, on replay and on a peer alike.
    pub(crate) substrates: crate::gpu::registry::Registry<Substrate>,
    /// A stroke the live preview has already drawn, for the `CommitStroke` fold to
    /// take rather than render again — see [`PreparedStroke`].
    ///
    /// Transient: the engine fills it immediately before the push that commits the
    /// stroke and empties it immediately after, so at every other moment — and in
    /// every clone handed to a sibling engine — it is `None`. The fold empties it
    /// only by *taking* it, which is how the engine learns whether the offer was
    /// accepted: a slot still full after the push was declined.
    ///
    /// Written through [`offer`](Self::offer) and [`reclaim`](Self::reclaim), and
    /// emptied by the fold that takes it — those three, and the `None` an engine is
    /// built with.
    pub(crate) prepared: Option<PreparedStroke>,
}

impl ApplyCtx {
    /// Offer the fold the tiles the preview already drew for this stroke (§6.2),
    /// answering whether anything was offered at all.
    pub(crate) fn offer(&mut self, prepared: Option<PreparedStroke>) -> bool {
        self.prepared = prepared;
        self.prepared.is_some()
    }

    /// Take back an offer the fold declined — `true` when there was one to take back,
    /// which is exactly "the fold did not use it".
    ///
    /// The slot is emptied either way, so the transience the field's doc claims holds
    /// whichever answer this gives.
    pub(crate) fn reclaim(&mut self) -> bool {
        self.prepared.take().is_some()
    }
}

/// Every field is a handle, so a copy is a fistful of refcount bumps — **except
/// [`prepared`](Self::prepared), which a clone always leaves empty.**
///
/// Written out rather than derived because that exception is the whole point. The
/// slot is a message to one fold, and a sibling preview engine (`Engine::new_sharing`,
/// §11) will never take it; a derive taken while the slot was full would hand that
/// engine a whole stroke's worth of fresh `Arc<GpuTile>` handles and pin them for its
/// life. The field's doc says a clone's slot is `None`, and the engine's
/// fill-push-empty around a single commit is what made that true. Here it is a
/// property of the type instead (CLAUDE.md: rule out a class).
impl Clone for ApplyCtx {
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
            stroke: self.stroke.clone(),
            assets: self.assets.clone(),
            selection: self.selection.clone(),
            transform: self.transform.clone(),
            fill: self.fill.clone(),
            merge: self.merge.clone(),
            place: self.place.clone(),
            pictures: self.pictures.clone(),
            gpu: self.gpu.clone(),
            substrates: self.substrates.clone(),
            prepared: None,
        }
    }
}

/// A stroke whose tiles the live preview has already rendered, offered to the fold
/// so that its `CommitStroke` takes them instead of rendering the whole stroke again
/// at pen-up (§6.2, §17.6).
///
/// The preview *is* the commit, drawn early: `preview == committed` (§1.3) is the
/// claim that the fold's frozen head plus its live tail and one whole-stroke render
/// are the same picture, and everything in `engine::live` exists to keep it so.
/// Rendering the stroke a second time when the pointer lifts therefore buys nothing
/// and costs the stroke's whole length in one hitch — at exactly the moment the
/// incremental repaint was built to stop paying for it. What lands in the document
/// is then the picture the artist watched being painted, to the bit.
///
/// **What the fold checks, and what it cannot.** Two things are verifiable from
/// inside the fold and are verified there, so that no caller can hand it tiles for
/// the wrong stroke: the record is this action's (`rec`), and the tiles were drawn
/// over this state's own layer — `base`, compared by *identity*, which is exact for
/// a persistent map (one is replaced, never edited, so the same root is the same
/// tiles). What the fold cannot see is the rest of the scene the render read — the
/// author's selection and the canvas substrate — and those are the engine's to
/// vouch for: an offer is only made while nothing has replaced the document since
/// the tiles were drawn, which is the same epoch rule the preview's own cached heads
/// are trusted by (`Preview::invalidate`).
///
/// Kept for one push rather than cached: the tiles share structure with the
/// committed document's and hold every fresh tile the stroke wrote, so a slot that
/// outlived its commit would pin a stroke's worth of GPU memory for nothing.
#[derive(Clone)]
pub(crate) struct PreparedStroke {
    /// The record the tiles are a render of — compared whole, because the commit
    /// re-derives its record from the fitter at pen-up and sanitizes it on the way
    /// in, and either step disagreeing with what was previewed means the preview was
    /// of a different stroke.
    pub(crate) rec: stark_model::document::StrokeRecord,
    /// The layer's tiles the render read as its base.
    pub(crate) base: crate::gpu::tile::TileMap,
    /// The liquify run the render read beside them (§6.13) — part of the base a
    /// liquify stroke composes through, so part of what the offer is checked against.
    pub(crate) base_run: Option<Rc<LiquifyRun>>,
    /// The layer's tiles with the stroke on them: `base` with a fresh handle at
    /// every tile the stroke reached.
    pub(crate) tiles: crate::gpu::tile::TileMap,
    /// The run the render left (§6.13): a liquify stroke's, or `None` for every
    /// other effect, which leaves the run in place ([`Layer::with_painted`]).
    pub(crate) liquify: Option<Rc<LiquifyRun>>,
}

impl PreparedStroke {
    /// Whether these tiles are the render of `rec` over `base` and `run` — the
    /// fold's half of the check described on the type.
    fn is_render_of(
        &self,
        rec: &stark_model::document::StrokeRecord,
        base: &crate::gpu::tile::TileMap,
        run: Option<&Rc<LiquifyRun>>,
    ) -> bool {
        self.base.ptr_eq(base) && LiquifyRun::same(self.base_run.as_ref(), run) && self.rec == *rec
    }
}

impl ApplyCtx {
    /// The canvas substrate `substrate` names, built on demand ([`Registry::get`]).
    ///
    /// Returns an owned handle rather than a borrow — `SubstrateMap` is a pair of
    /// reference-counted wgpu objects, so the clone is two atomic bumps — because
    /// the caller then borrows other fields of `self` to build the scene around it.
    ///
    /// [`Registry::get`]: crate::gpu::registry::Registry::get
    pub(crate) fn substrate(&self, substrate: Substrate) -> crate::gpu::substrate::SubstrateMap {
        self.substrates.get(&self.gpu, substrate)
    }
}

/// The shared body of the three transform actions (§16): cut the author's
/// selected paint, restack it under `map`, and carry the author's mask with it.
/// Gated and keyed exactly as a stroke is — the mask comes off the state being
/// folded over, the actor off the action's own id. A matte or absent layer
/// refuses it, like a stroke; an unusable or oversized map is rejected
/// deterministically, so peers and replays agree.
/// **The gate every action that lays or moves paint passes through** — the layer's
/// tiles, the author's mask, the renderer's answer, and the rewrite — in one place.
///
/// Four actions lay or move paint (a stroke, a fill, a transform, a merge), and the
/// three things this settles are the three each would otherwise get right on its
/// own:
///
/// - a **matte or an absent layer refuses** the edit rather than swallowing it
///   ([`paint_base`], §15.7) — here, in the engine, so replay and peers agree about a
///   log that contains one;
/// - the mask is the **author's**, keyed off `action.id.actor` and never off the
///   session (§17.3), so a collaborator's lasso cannot clip this edit;
/// - a renderer that declines leaves the state alone, **deterministically**, and says
///   so once.
///
/// `edit` answers `None` to decline, or the new tiles — with the liquify run a liquify
/// stroke leaves behind them (§6.13), `None` from every other edit, which keeps the
/// run in place ([`Layer::with_painted`]).
fn paint_edit(
    state: DocState,
    layer: LayerId,
    actor: ActorId,
    refused: &'static str,
    edit: impl FnOnce(
        &DocState,
        &Layer,
        &super::selection::Selection,
    ) -> Option<(crate::gpu::tile::TileMap, Option<Rc<LiquifyRun>>)>,
) -> DocState {
    let Some(target) = state.layer(layer).filter(|l| l.tiles().is_some()).cloned() else {
        return state;
    };
    let selection = state.selection_of(actor);
    match edit(&state, &target, &selection) {
        Some((tiles, run)) => state.map_layer(layer, |l| l.with_painted(tiles, run)),
        None => {
            tracing::warn!("{refused}");
            state
        }
    }
}

/// [`paint_edit`] for the one family that also **moves the author's mask**: the three
/// transforms, which cut the selected paint out and restack it under the map, and
/// carry the mask with it (§16).
///
/// A second function rather than an `Option<Selection>` in the first's return. That
/// sentinel was `Some` in exactly one of three callers and `None` hard-coded in the
/// other two, and the `&DocState` beside it was bound by exactly one — the opposite
/// one — so a reader had to check all three call sites to learn which half of the
/// signature was live at each. Each helper's parameters are now what its callers
/// actually use.
fn paint_and_mask_edit(
    state: DocState,
    layer: LayerId,
    actor: ActorId,
    refused: &'static str,
    edit: impl FnOnce(
        &crate::gpu::tile::TileMap,
        &super::selection::Selection,
    ) -> Option<(crate::gpu::tile::TileMap, super::selection::Selection)>,
) -> DocState {
    let Some(base) = paint_base(&state, layer) else {
        return state;
    };
    let selection = state.selection_of(actor);
    match edit(&base, &selection) {
        Some((tiles, moved)) => state
            .map_layer(layer, |l| l.with_tiles(tiles))
            .with_selection(actor, moved),
        None => {
            tracing::warn!("{refused}");
            state
        }
    }
}

fn transform_apply(
    state: DocState,
    ctx: &ApplyCtx,
    actor: ActorId,
    layer: LayerId,
    map: &stark_model::document::TransformMap,
    translation: stark_model::geom::IVec2,
) -> DocState {
    // The mask travels with the paint it gated (§16), which is what the second
    // helper is for. `frame` is the action's own (§14.12): the renderer runs the
    // map conjugated into the layer's frame on the paint and the canvas map on
    // the mask.
    paint_and_mask_edit(
        state,
        layer,
        actor,
        "transform rejected (unusable map or too many tiles); ignored",
        |base, selection| {
            ctx.transform
                .apply(&ctx.pool, base, selection, map, translation)
        },
    )
}

/// The body of [`ActionKind::MergeLayerDown`] (§14.11): fold `source` into the
/// layer beneath it, which is the one action whose promise is about pixels that do
/// **not** move.
///
/// Beside [`transform_apply`] and for its reason — an `apply` arm says *what* an
/// action does, and reconciling a plan, a renderer and a state rewrite is a
/// paragraph rather than a line. [`merge::plan`](super::merge::plan) stays where it
/// is: it is pure CPU and its module says so, while this half holds a [`TilePool`].
///
/// The plan is **re-derived from the state being folded over** rather than trusted
/// from the log, so a replay and a peer decide from the same document — and a plan
/// that no longer names `dest` (a concurrent reorder, a mode set on either layer
/// since) declines the whole action, leaving the document untouched. Deterministic,
/// so everyone declines together.
fn merge_apply(state: DocState, ctx: &ApplyCtx, source: LayerId, dest: LayerId) -> DocState {
    let Some(plan) = super::merge::plan(&state, source) else {
        return state;
    };
    if plan.dest != dest {
        tracing::warn!("merge down no longer names this destination; ignored");
        return state;
    }
    // Cloned out before the rewrite for the reason `paint_base` clones: a handful of
    // `Arc` bumps, and it is what keeps the borrow of the state off the tree being
    // rebuilt. `plan` has already said the destination is paint.
    let Some(lower) = paint_base(&state, dest) else {
        return state;
    };
    // What the survivor's tiles become, and what its params become — `None` for
    // "its own, untouched", which is the whole content of a filter merge (see
    // [`MergeKind`](super::merge::MergeKind)). Said once here rather than by two
    // `return`ing branches that would each have to state it.
    let (tiles, keeps) = match plan.kind {
        // A **filter** source is the other kind of merge (§14.11.7): nothing is
        // stacked, so the destination's channels are rewritten where they stand and
        // every other thing about that layer — its blend, its clip, its opacity, its
        // place — is left exactly as it was.
        super::merge::MergeKind::Filter {
            filter,
            source_params,
        } => {
            // A **neutral** filter is not run at all, and that is the honest answer
            // rather than a shortcut: the draw list already leaves one out (§21.3), so
            // what it contributes to the picture is nothing, and the merge that must
            // not change the picture therefore has nothing to write. Rewriting the
            // tiles anyway would spend a pass per tile to land the identity *plus* one
            // round trip's rounding.
            let merged = if filter.is_neutral() {
                lower
            } else {
                ctx.merge.apply_filter(
                    &ctx.pool,
                    &lower,
                    // Built through the very constructor the draw list goes through,
                    // off the very params the compositor reads (§21.4) — so the merged
                    // tile is what the screen was showing, not a second reading of the
                    // same layer.
                    &crate::gpu::composite::FilterDraw::new(filter, source_params),
                )
            };
            (merged, None)
        }
        // Every number here is the plan's: what each side's tiles are worth on their
        // own, how the upper meets the lower, and what the survivor carries
        // afterwards. The two kinds differ by where the destination's slider belongs
        // — folded into the tiles beside a sibling, left on the layer when the
        // destination is a carrier and the slider is the group's (§14.7) — and that
        // decision is made once, in `plan`, rather than twice here and there.
        super::merge::MergeKind::Stack {
            source_params,
            dest_opacity,
            keeps,
        } => {
            // Both sides are paint that carries nothing — `plan` said so — so the
            // tile map is there to be read.
            let Some(upper) = paint_base(&state, source) else {
                return state;
            };
            // The source restated in the destination's frame (§14.12): the two
            // sides must share one tile space before they can be stacked, and the
            // survivor keeps its own frame. A whole-pixel shift, run through the
            // transform's own exactness invariant (§16.4) under a universal
            // selection — so the merge that must not change the picture does not.
            // Declined past the transform's tile caps, deterministically, like
            // every refusal here.
            let frame_of = |id: LayerId| {
                state
                    .layer(id)
                    .map_or(stark_model::geom::IVec2::ZERO, |l| l.translation)
            };
            let shift = frame_of(source) - frame_of(dest);
            let upper = if shift == stark_model::geom::IVec2::ZERO {
                upper
            } else {
                let map = stark_model::document::TransformMap::Affine(
                    stark_model::geom::Affine2::from_translation(shift.as_vec2()),
                );
                let everything = super::selection::Selection::everything();
                match ctx.transform.apply(
                    &ctx.pool,
                    &upper,
                    &everything,
                    &map,
                    stark_model::geom::IVec2::ZERO,
                ) {
                    Some((tiles, _)) => tiles,
                    None => {
                        tracing::warn!("merge across frames exceeds the tile caps; ignored");
                        return state;
                    }
                }
            };
            let tiles = ctx.merge.apply(
                &ctx.pool,
                crate::gpu::merge::MergeScene {
                    lower: crate::gpu::merge::MergeSide {
                        tiles: &lower,
                        opacity: dest_opacity,
                    },
                    upper: crate::gpu::merge::MergeSide {
                        tiles: &upper,
                        opacity: source_params.opacity,
                    },
                    blend: source_params.blend,
                    clip: source_params.clip,
                },
            );
            (tiles, Some(keeps))
        }
    };
    state
        .map_layer(dest, |l| Layer {
            composite: keeps.unwrap_or(l.composite),
            ..l.with_tiles(tiles)
        })
        .remove_layer(source)
}

/// The body of [`ActionKind::FloatSelection`] (§16.12): cut what the author's
/// selection holds on `layer` into the minted `child`, carried at the foot of
/// the source's stack, and consume the selection — the float *is* the selection
/// now, and an outline left behind would sit over paint that is no longer there.
///
/// Every refusal is deterministic, so peers and replays agree: a matte, filter
/// or absent source ([`paint_base`]); a child id the document already holds
/// (two layers under one id is §17.9's failure, and half a float — the cut
/// taken, nothing landed — would be worse than either); a universal selection
/// (nothing is *selected*; moving everything is `TranslateLayers`); an empty or
/// oversized cut (`plan_float`).
fn float_apply(
    state: DocState,
    ctx: &ApplyCtx,
    actor: ActorId,
    layer: LayerId,
    child: LayerId,
    translation: stark_model::geom::IVec2,
) -> DocState {
    let Some(base) = paint_base(&state, layer) else {
        return state;
    };
    if state.contains_layer(child) {
        tracing::warn!("float names a child the document already holds; ignored");
        return state;
    }
    let selection = state.selection_of(actor);
    if selection.is_universal() {
        tracing::warn!("float with nothing selected; ignored");
        return state;
    }
    let Some((remaining, lifted)) = ctx
        .transform
        .float(&ctx.pool, &base, &selection, translation)
    else {
        tracing::warn!("float rejected (empty cut or too many tiles); ignored");
        return state;
    };
    // Unnamed, identity params, shown — and standing at the frame the cut was
    // made in, which is the action's own field rather than the state's, so a
    // replay mints what the recording run minted (§16.12).
    let lifted = Layer {
        translation,
        ..Layer::new(child).with_tiles(lifted)
    };
    state
        .map_layer(layer, |l| l.with_tiles(remaining))
        .insert_float(layer, lifted)
        .with_selection(actor, super::selection::Selection::everything())
}

/// The tiles `layer` paints into, or `None` if it has none — the gate every
/// action that lays or moves paint passes through first.
///
/// **The refusal is the point.** A matte has no tile map, so a stroke, a fill or
/// a transform aimed at one is turned away rather than swallowed or magically
/// rasterized (§15.7) — and turned away *here*, in the engine, not only in the
/// frontend, which is what keeps replay and peers agreeing about a log that
/// happens to contain such an action. An absent layer reads the same way: there
/// is nothing there to paint on.
///
/// Cloned out of the tree before anything is rebuilt: the map is persistent, so
/// this is a handful of `Arc` bumps, and it is what keeps the borrow of the state
/// from outliving the rewrite that follows.
fn paint_base(state: &DocState, layer: LayerId) -> Option<crate::gpu::tile::TileMap> {
    state.layer(layer).and_then(|l| l.tiles()).cloned()
}

/// [`DocState`] is what this crate folds a log into (§2, §5).
///
/// The model owns *that* a log folds and which actions commute; this is the answer
/// to what it folds into — a persistent map of copy-on-write GPU tiles, which is
/// precisely the thing the model cannot name.
impl Materialize for DocState {
    type Ctx = ApplyCtx;

    fn fold(self, action: &Action, ctx: &mut ApplyCtx) -> DocState {
        apply(action, self, ctx)
    }

    /// Hold this fold to its own footprint (§12.6) — see [`super::audit`] for why the
    /// check lives here rather than in a test, and what it costs.
    #[cfg(debug_assertions)]
    fn audit(before: &Self, after: &Self, action: &Action, footprint: &Footprint) {
        super::audit::audit(before, after, action, footprint);
    }

    /// `DocState`'s clone is a handful of `Arc` bumps (§5.1), which is what makes
    /// keeping the previous state to compare against affordable at all.
    #[cfg(debug_assertions)]
    const AUDITED: bool = true;

    /// Remove this action's effect by restoring what it wrote from `previous` — the
    /// values under its footprint, nothing more, so the edits of commuting actions
    /// applied after it survive. Tiles come back as the same shared handles
    /// (copy-on-write means identity is equality), so this re-renders nothing.
    ///
    /// **Audited like the fold, and for a sharper reason.** The forward direction is
    /// driven hundreds of times by every test in the workspace; this one runs only
    /// where the history shifts an undone action past a commuting suffix, so its
    /// coverage was the handful of scenarios in `tests/commute.rs`. It is also the
    /// direction §12.6 is really about: what it writes becomes the replay base for
    /// every later version, and a peer that patched differently diverges with no
    /// pixel able to say so. Same checker, same `cfg`, same argument for the cost
    /// (see [`super::audit`]).
    fn unfold(&mut self, action: &Action, footprint: &Footprint, previous: &DocState) {
        #[cfg(debug_assertions)]
        let before = self.clone();
        *self = super::patch::unapply(action, footprint, previous, self);
        #[cfg(debug_assertions)]
        super::audit::audit(&before, self, action, footprint);
    }
}

/// Fold one action into the state — the whole of what applying means (§4).
///
/// A free function taking the action rather than a method on it, because [`Action`]
/// is `stark-model`'s now and this needs the renderers: the division the orphan rule
/// forced is the one the split is about (§2).
///
/// **What is left here is the ten kinds that need a renderer** — a stroke, a placed
/// image, the two selection ops, the three transforms, a merge, a float and a fill.
/// Everything else folds without touching one and is [`apply_pure`]'s, which this
/// match ends by handing over to. Both matches are exhaustive with no `_` arm, so a
/// kind added later has to name itself on one side of the line rather than defaulting
/// into either.
///
/// **Total.** An action that cannot be honoured — a stroke on a missing layer, a
/// transform past the tile caps — returns the state unchanged rather than an error,
/// so every peer declines it identically (`Materialize::fold`).
fn apply(action: &Action, state: DocState, ctx: &mut ApplyCtx) -> DocState {
    match &action.kind {
        // The **author's** mask gates the stroke, read from the state being folded
        // over so replay reproduces it exactly and keyed by the author so a
        // collaborator's lasso never clips it (§6.8, §17.3) — both of which
        // [`paint_edit`] settles, along with the refusal on a matte.
        ActionKind::CommitStroke(rec) => paint_edit(
            state,
            rec.layer,
            action.id.actor,
            // Unreachable: a stroke either takes the preview's tiles or renders, and
            // neither declines. The gate above it — a matte, an absent layer — is
            // `paint_edit`'s own and returns before this is read.
            "stroke rendered nothing; ignored",
            |state, target, selection| {
                let base = target.tiles().expect("paint_edit hands over a paint layer");
                let run = target.liquify_run();
                // The preview's tiles, where the preview drew this very stroke over
                // this very base — see `PreparedStroke`. Otherwise the stroke is
                // rendered here, which is every replay, every peer's copy and every
                // redo: the log carries the stroke, never its pixels.
                let painted = match ctx.prepared.take_if(|p| p.is_render_of(rec, base, run)) {
                    Some(prepared) => (prepared.tiles, prepared.liquify),
                    None => {
                        // The substrate this stroke was painted on, as the log stood
                        // here — not as it stands now (§6.4). The tooth gates the
                        // deposit by it, so a mid-document `SetSubstrate` changes what
                        // comes *after* it and nothing before, on replay exactly as it
                        // did live.
                        let substrate = ctx.substrate(state.substrate());
                        // The author's canvas mask, brought into the layer's frame
                        // the record was made in (§14.12) — scoped to the stroke's
                        // own claim, and the mask itself, by handle, whenever the
                        // frame is zero.
                        let gate = ctx.transform.shifted_selection_in(
                            &ctx.pool,
                            selection,
                            rec.translation,
                            stark_model::document::stroke_rect(rec),
                        );
                        let painted = ctx.stroke.render(
                            crate::gpu::stroke::StrokeScene {
                                pool: &ctx.pool,
                                assets: &ctx.assets,
                                base,
                                liquify: run,
                                selection: &gate,
                                substrate: &substrate,
                            },
                            rec,
                        );
                        (painted.tiles, painted.liquify)
                    }
                };
                // A stroke lays paint through the author's mask and never moves it.
                Some(painted)
            },
        ),
        // An image from outside the document, as a layer holding it (§23). The layer
        // arrives first and by exactly the same call an `AddLayer` makes, so an unknown
        // carrier declines it identically — and the tiles are only built once the layer
        // is known to have landed, since building them for a layer that is not there is
        // a photograph's worth of GPU memory for nothing.
        //
        // The tiles come from the CPU rather than from a pass, which is what makes them
        // the same bytes on every adapter: see `gpu::place`.
        ActionKind::PlaceImage {
            id,
            carrier,
            above,
            at,
            name,
            image,
        } => {
            let state = state
                .insert_layer(*id, *carrier, *above)
                .set_layer_name(*id, name.as_deref().map(Into::into));
            if !state.contains_layer(*id) {
                return state;
            }
            // The picture, by the id the log carries (§23). Absent means it has not
            // arrived — which the loader and the transport both make sure cannot
            // happen before this runs (`unresolved_content`, and the waitlist parking
            // the action until its content lands), so reaching the `None` arm is a
            // caller that skipped the bill rather than a state to design around.
            let Some(picture) = ctx.pictures.get(*image) else {
                tracing::warn!(?image, "placing an image this session does not hold");
                return state;
            };
            match ctx.place.render(&ctx.pool, *at, &picture) {
                Some(tiles) => state.map_layer(*id, |l| l.with_tiles(tiles)),
                None => {
                    // Off the tile grid an `i32` can address. The layer stays — it is
                    // what the action minted, and withdrawing it here would make the
                    // action half-applied — and it is simply empty, which is the honest
                    // picture of an image placed where no tile exists.
                    tracing::warn!("placed image is off the addressable canvas; no tiles written");
                    state
                }
            }
        }
        // The author's own selection, and only ever the author's: the key is
        // taken from `action.id.actor`, never from the payload, so an action
        // cannot address anyone else's mask (§17.3).
        //
        // An op too large to rasterize (see `MAX_SELECTION_TILES`) leaves the
        // selection alone — deterministically, since the bound is a pure
        // function of the op, so peers and replays agree.
        ActionKind::Select(op) => {
            let prev = state.selection_of(action.id.actor);
            match ctx.selection.apply(&ctx.pool, &prev, op) {
                Some(selection) => state.with_selection(action.id.actor, selection),
                None => {
                    tracing::warn!("selection op too large to rasterize; ignored");
                    state
                }
            }
        }
        ActionKind::InvertSelection => {
            let prev = state.selection_of(action.id.actor);
            let selection = ctx.selection.invert(&ctx.pool, &prev);
            state.with_selection(action.id.actor, selection)
        }
        // Cut the author's selected paint, restack it under the affine, and
        // carry the author's mask with it (§16). Gated and
        // keyed exactly as a stroke is: the mask comes off the state being
        // folded over, the actor off the action's own id. A matte or absent
        // layer refuses it, like a stroke; an unusable or oversized transform
        // is rejected deterministically, so peers and replays agree.
        ActionKind::Transform {
            layer,
            affine,
            translation: frame,
        } => transform_apply(
            state,
            ctx,
            action.id.actor,
            *layer,
            &stark_model::document::TransformMap::Affine(*affine),
            *frame,
        ),
        // The rect-scoped siblings (§16.8, §16.9): identical shape — cut,
        // restack, carry the mask — differing only in the map handed to
        // the renderer.
        ActionKind::TransformPerspective {
            layer,
            map,
            translation: frame,
        } => transform_apply(
            state,
            ctx,
            action.id.actor,
            *layer,
            &stark_model::document::TransformMap::Perspective(*map),
            *frame,
        ),
        ActionKind::TransformWarp {
            layer,
            map,
            translation: frame,
        } => transform_apply(
            state,
            ctx,
            action.id.actor,
            *layer,
            &stark_model::document::TransformMap::Warp(map.clone()),
            *frame,
        ),
        // Fold two layers into one without moving a pixel of the composite
        // (§14.11) — the orchestration is `merge_apply`, beside `transform_apply`
        // and for its reason.
        ActionKind::MergeLayerDown { source, dest } => merge_apply(state, ctx, *source, *dest),
        // Reify the author's selection as a layer a drag can move for the price
        // of a property write (§16.12) — the orchestration is `float_apply`.
        ActionKind::FloatSelection {
            layer,
            child,
            translation: frame,
        } => float_apply(state, ctx, action.id.actor, *layer, *child, *frame),
        // Lay a parcel of paint through the region's coverage, gated by the
        // author's selection — the same gate a stroke passes through, so a fill
        // is clipped by a selection exactly as a brush is
        // (§18.0.4). Refused on a matte or absent layer like a stroke; refused
        // deterministically when unbounded or oversized, so peers and replays
        // agree about a log that contains one.
        ActionKind::Fill {
            layer,
            op,
            translation: frame,
        } => paint_edit(
            state,
            *layer,
            action.id.actor,
            "fill rejected (unbounded region or too many tiles); ignored",
            // A fill lays paint through a mask and never moves it, which is now what
            // taking `paint_edit` says rather than a `None` it has to pass. The mask
            // is brought into the layer's frame first, as a stroke's is (§14.12).
            |_, target, selection| {
                let base = target.tiles().expect("paint_edit hands over a paint layer");
                let gate = ctx.transform.shifted_selection_in(
                    &ctx.pool,
                    selection,
                    *frame,
                    stark_model::document::fill_rect(op),
                );
                // A fill leaves the run in place (§6.13).
                ctx.fill
                    .apply(&ctx.pool, base, &gate, op)
                    .map(|t| (t, None))
            },
        ),

        // Everything that needs no renderer, written out rather than swept up by a
        // `_`: which side of the line a new kind belongs on is a question the
        // compiler should be the one to ask.
        kind @ (ActionKind::AddLayer { .. }
        | ActionKind::DuplicateLayer { .. }
        | ActionKind::RemoveLayer { .. }
        | ActionKind::SetLayerBlend(..)
        | ActionKind::SetLayerClip(..)
        | ActionKind::SetLayerOpacity(..)
        | ActionKind::SetLayerVisible(..)
        | ActionKind::SetLayerName(..)
        | ActionKind::MoveLayer { .. }
        | ActionKind::Undo(_)
        | ActionKind::SetSelectionOpacity(_)
        | ActionKind::SetSubstrate(_)
        | ActionKind::SetSubstrateScale(_)
        | ActionKind::SetSubstrateColor(_)
        | ActionKind::AddMatte { .. }
        | ActionKind::SetMatteRect(..)
        | ActionKind::SetMattePaint(..)
        | ActionKind::AddFilter { .. }
        | ActionKind::SetFilter(..)
        | ActionKind::AddGuide { .. }
        | ActionKind::RemoveGuide(_)
        | ActionKind::SetGuide(..)
        | ActionKind::SetGuideName(..)
        | ActionKind::MoveGuide { .. }
        | ActionKind::TranslateLayers { .. }) => {
            apply_pure(kind, state, action.id.actor).into_state()
        }
    }
}

/// The arms of [`apply`] that are a [`DocState`] call and nothing else — two thirds
/// of the fold, answered without a tile pool, a renderer or an adapter.
///
/// [`Folded::NeedsRenderer`] for the ten kinds that do need one, so [`apply`] keeps
/// those and hands everything else here. **The partition is the compiler's**: this
/// function has no [`ApplyCtx`] to name, so an arm that reaches for a renderer does
/// not compile — which is the whole point of the shape, and the reason nothing is
/// threaded in here against the day one might.
///
/// Both matches are exhaustive, so a new kind must name itself on each side of the
/// line — but naming it on the *wrong* side of both compiles, and no match can say so.
/// **That is why every answer here carries a document**: the mistake costs an ignored
/// action rather than a panic on every remote peer's fold and on every action of a
/// loaded file, in release as well as debug.
///
/// Split out because these arms were being *reconstructed* to be tested. `patch.rs`
/// checks that folding an action and unapplying it puts the document back, and the
/// fold it was checking against was a hand-written match in its own test module: a
/// third statement of the same mutation, with nothing tying it to this one. A
/// sanitize or a refusal added to an arm here would leave that round trip passing,
/// having tested a fold nobody runs. It calls this now.
///
/// Takes the actor rather than the [`Action`] because that is all any arm reads of
/// it: the author's own mask is keyed by it (§17.3) and `SetSelectionOpacity` writes
/// through it. Nothing here reads the Lamport clock.
pub(super) fn apply_pure(kind: &ActionKind, state: DocState, actor: ActorId) -> Folded {
    Folded::Done(match kind {
        ActionKind::AddLayer { id, carrier, above } => state.insert_layer(*id, *carrier, *above),
        // The copy's tiles are the shared handles the source already holds, so
        // duplicating a layer costs no GPU memory until one of the two is
        // painted on — copy-on-write is what makes this a cheap action rather
        // than a re-render of everything under it (§5.2).
        ActionKind::DuplicateLayer { ids } => state.duplicate_layer(ids),
        // **Declined unless the subtree is what the action names** (§12.6). Every id
        // in it is a `Resource::Layer` write, so a group holding one the action does
        // not name — a peer's concurrent add — would write state nothing declared,
        // which is the divergence `ActionKind::RemoveLayer` describes. Asked of the
        // state being folded, so peers and replays decline the same action;
        // `duplicate_layer` refuses on the same terms.
        ActionKind::RemoveLayer { id, carried } => {
            match state.carried_ids(*id) {
                // Absent: nothing to remove, and the arm below would say the same.
                None => state,
                Some(holds) if holds == *carried => state.remove_layer(*id),
                Some(holds) => {
                    tracing::warn!(
                        ?id,
                        named = carried.len(),
                        holds = holds.len(),
                        "a group removal names a subtree the document no longer holds; ignored",
                    );
                    state
                }
            }
        }
        ActionKind::SetLayerBlend(id, blend) => state.set_layer_blend(*id, *blend),
        ActionKind::SetLayerClip(id, clip) => state.set_layer_clip(*id, *clip),
        ActionKind::SetLayerOpacity(id, opacity) => state.set_layer_opacity(*id, *opacity),
        ActionKind::SetLayerVisible(id, visible) => state.set_layer_visible(*id, *visible),
        ActionKind::SetLayerName(id, name) => {
            state.set_layer_name(*id, name.as_deref().map(Into::into))
        }
        ActionKind::MoveLayer { id, carrier, at } => state.move_layer(*id, *carrier, *at),
        // Resolved at the timeline layer (effective-sequence filtering); an
        // `Undo` should never be materialized through `apply`. Identity, so
        // a stray one is harmless rather than wrong.
        ActionKind::Undo(_) => state,
        // The author's own mask, keyed like every other selection edit off
        // `action.id.actor`. Rasterizes nothing at all: the strength is how the mask
        // is read, so this touches one number and no tile (§6.8) — which is exactly
        // why it is here and its two siblings are not.
        ActionKind::SetSelectionOpacity(opacity) => state.with_selection_opacity(actor, *opacity),
        ActionKind::SetSubstrate(id) => state.with_substrate(*id),
        ActionKind::SetSubstrateScale(scale) => state.with_substrate_scale(*scale),
        ActionKind::SetSubstrateColor(rgb) => state.with_substrate_color(*rgb),
        ActionKind::AddMatte {
            id,
            carrier,
            at,
            region,
            paint,
        } => state.insert_matte(*id, *carrier, *at, *region, paint.clone()),
        ActionKind::SetMatteRect(id, min, max) => state.set_matte_rect(*id, *min, *max),
        ActionKind::SetMattePaint(id, paint) => state.set_matte_paint(*id, paint.clone()),
        // The payload is sanitized inside `insert_filter`/`set_filter` — the
        // funnel sits in state, where replayed files and remote peers land too,
        // not only where a local command is minted (§21.6).
        ActionKind::AddFilter {
            id,
            carrier,
            above,
            filter,
        } => state.insert_filter(*id, *carrier, *above, filter.clone()),
        ActionKind::SetFilter(id, filter) => state.set_filter(*id, filter.clone()),

        // The drawing guides (§20.5). The one family of actions with no pixel on
        // the other side of it: a guide is geometry to construct through, so
        // applying one moves a roster and nothing else — which is also why none
        // of these needed `ctx` even before there was somewhere to say so.
        //
        // **The id is the adding action's own** (`GuideId`), and the action carries
        // it rather than the fold deriving it: an id derived here is not part of the
        // action, and `start_collaboration` rewrites solo-authored `ActionId`s
        // (§17.9). Everything else names one that was minted earlier and no-ops when
        // it is not there, the way every action naming an absent layer does.
        ActionKind::AddGuide {
            id,
            guide,
            after,
            name,
        } => state.insert_guide(*id, *guide, *after, name.as_deref().map(Arc::from)),
        ActionKind::RemoveGuide(id) => state.remove_guide(*id),
        ActionKind::SetGuide(id, guide) => state.set_guide(*id, *guide),
        ActionKind::SetGuideName(id, name) => {
            state.set_guide_name(*id, name.as_deref().map(Arc::from))
        }
        ActionKind::MoveGuide { id, after } => state.move_guide(*id, *after),

        // Move layers without moving a tile (§14.12): a property write, so it
        // belongs to this half — which is what makes its drag preview free.
        ActionKind::TranslateLayers { moves } => state.translate_layers(moves),

        // The ten whose answer is pixels, and so is a renderer's. Declined here
        // rather than absent, so that a kind added later has to say which side it is
        // on in this match too — and so that [`apply`]'s list and this one cannot
        // both quietly omit it.
        ActionKind::CommitStroke(_)
        | ActionKind::PlaceImage { .. }
        | ActionKind::Select(_)
        | ActionKind::InvertSelection
        | ActionKind::Transform { .. }
        | ActionKind::TransformPerspective { .. }
        | ActionKind::TransformWarp { .. }
        | ActionKind::MergeLayerDown { .. }
        | ActionKind::FloatSelection { .. }
        | ActionKind::Fill { .. } => return Folded::NeedsRenderer(state),
    })
}

/// What [`apply_pure`] made of the kind it was handed — the document either way.
///
/// Not a `Result`, because neither answer is a failure: a kind this half does not fold
/// is one [`apply`] renders itself.
pub(super) enum Folded {
    /// Folded here: the document with the action in it.
    Done(DocState),
    /// Not folded: the state as it arrived, for the caller that has a renderer.
    NeedsRenderer(DocState),
}

impl Folded {
    /// The document, whichever answer this is — what [`apply`] takes, having already
    /// decided by its own match that this kind needs no renderer.
    fn into_state(self) -> DocState {
        let (Folded::Done(state) | Folded::NeedsRenderer(state)) = self;
        state
    }
}

/// The document as committing `kind` would leave it, **without logging it** — the
/// preview half of every setter command (§21.6).
///
/// A drag previews by folding the very action its release will commit: the same
/// [`apply`] arm, behind the same [`ActionKind::sanitized`] funnel. That is the whole
/// of it, and the reason it is one function rather than a line in each preview arm.
/// Written per arm the sanitize is a habit, and two arms had already forgotten it.
/// `SetMattePaint` reached [`DocState::set_matte_paint`] directly, so it previewed a
/// gradient whose axis nobody can place — the first sample of an axis drag has
/// `from == to`, which `Parcel::sanitized` collapses to the ramp's anchor.
/// `SetLayerOpacity` went through `f32::clamp`, which passes a NaN straight out
/// (both of NaN's comparisons are false) where the commit's `finite_in` lands it on
/// 1.0. Both are reachable from a drag, and both showed the artist a document the
/// release would then decline to store — a `preview == committed` break (§1.3) in the
/// one class of action with no pixels of its own to give it away.
///
/// **Only the kinds that move state and nothing else.** A preview mints no layer,
/// folds no stroke and takes no prepared tiles. The two previews that *do* touch the
/// GPU — the transform and the fill — keep their own entry points, because what they
/// have to answer first is whether the parcel can be cut at all.
///
/// The id is provisional. No kind that reaches here reads the Lamport clock, so
/// nothing is spent by not advancing it; the actor is real, because the author's own
/// selection is keyed by it (§17.3) and `SetSelectionOpacity` previews through here.
pub(crate) fn preview_of(
    kind: ActionKind,
    state: &DocState,
    actor: ActorId,
    ctx: &mut ApplyCtx,
) -> DocState {
    debug_assert!(
        !matches!(kind, ActionKind::CommitStroke(_)),
        "a preview folds a setter; `CommitStroke` would consume the prepared tiles",
    );
    let action = Action {
        id: ActionId { lamport: 0, actor },
        kind: kind.sanitized(),
    };
    apply(&action, state.clone(), ctx)
}

/// Whether applying this to `state` would leave it exactly as it found it — so a
/// command that would spend an undo step on nothing can decline to log one (§5.4).
///
/// **Answered by folding, not by a mirror of the fold.** This was a match with an
/// arm per setter, each comparing the payload against the value it read out of the
/// state — a fourth statement of every action's effect, beside the footprint, the
/// fold and the patch, and the one nothing tied to the other three. It drifted
/// twice: a blend on a filter and a rect the matte refuses both answered "an edit"
/// here while `apply` folded them to nothing, and each spent an undo step that does
/// nothing when reached. Now the fold answers, through the one enumeration of what
/// can differ between two states ([`audit::changes_nothing`](super::audit)), and the
/// match is gone.
///
/// Three answers are given before folding, and they are the whole of what is
/// stated here by hand:
///
/// - **An `Undo` is `false`.** The timeline resolves it (§5.4); its fold is the
///   identity whatever it undoes, so the fold is the one witness that cannot say.
///   Not asked today — `commit_with_id` is its door — but the answer has to be right
///   for the day it is.
/// - **A write to paint, to which layers exist, or to the tree's shape is `false`**,
///   read off the footprint. Finding out whether a stroke left a layer byte-identical
///   means rendering it, and a structural edit's refusals — a cycle, a subtree the
///   action no longer names — are the deterministic declines §12.1 wants in the log
///   so that every peer declines together. A false "no" costs an undo step; a false
///   "yes" would silently drop an edit.
/// - **An action naming a layer the document lacks is `false`**, read off the same
///   footprint: inert *here*, but a peer whose concurrent undo has just put that layer
///   back applies it, and a log that omitted it would be a different log on the two
///   clients (§12.1). The guide roster is one resource with no per-guide existence to
///   read ([`Resource::Guides`]), so a guide edit naming an absent guide takes the
///   fold's answer instead — the roster it leaves is the roster it found.
///
/// Everything else folds through [`apply_pure`] and is diffed. Asked of the action
/// **as it will be logged**: the engine runs the sanitizing funnel (§21.6) first, so
/// the payload folded here is the payload that would be stored.
pub(crate) fn is_noop_on(kind: &ActionKind, state: &DocState, actor: ActorId) -> bool {
    if matches!(kind, ActionKind::Undo(_)) {
        return false;
    }
    let footprint = compute_footprint(&Action {
        id: ActionId { lamport: 0, actor },
        kind: kind.clone(),
    });
    if footprint.writes.iter().any(never_a_noop) {
        return false;
    }
    let named = footprint.reads.iter().chain(&footprint.writes);
    if named
        .filter_map(layer_named)
        .any(|id| !state.contains_layer(id))
    {
        return false;
    }
    match apply_pure(kind, state.clone(), actor) {
        Folded::Done(after) => super::audit::changes_nothing(state, &after),
        // The renderer's half: pixels, whose answer costs the work the question
        // exists to avoid.
        Folded::NeedsRenderer(_) => false,
    }
}

/// Whether a write to this resource is one [`is_noop_on`] answers `false` to without
/// folding: paint, a layer's presence, everything about a layer, the tree's shape.
///
/// Exhaustive, so a resource added later has to say which side it is on.
fn never_a_noop(resource: &Resource) -> bool {
    match resource {
        Resource::Paint(..)
        | Resource::Existence(_)
        | Resource::Layer(_)
        | Resource::LiquifyRun(_)
        | Resource::StackOrder => true,
        Resource::Prop(..)
        | Resource::Selection(_)
        | Resource::Substrate
        | Resource::SubstrateColor
        | Resource::Guides => false,
    }
}

/// The layer this resource names, if it names one — what [`is_noop_on`] checks the
/// document holds.
fn layer_named(resource: &Resource) -> Option<LayerId> {
    match resource {
        Resource::Paint(id, _)
        | Resource::Existence(id)
        | Resource::Prop(id, _)
        | Resource::Layer(id)
        | Resource::LiquifyRun(id) => Some(*id),
        Resource::StackOrder
        | Resource::Selection(_)
        | Resource::Substrate
        | Resource::SubstrateColor
        | Resource::Guides => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stark_model::Srgb;
    use stark_model::document::{
        BlendMode, ColorAdjust, Filter, GuideId, MatteRegion, Parcel, PerspectiveGuide, Place,
    };
    use stark_model::geom::{IVec2, Vec2};

    const A: LayerId = LayerId::ROOT;
    const MATTE: LayerId = LayerId::solo(3);
    const FILTER: LayerId = LayerId::solo(4);
    /// An id the document does not hold.
    const ABSENT: LayerId = LayerId::solo(9);
    const ACTOR: ActorId = ActorId(1);
    const G1: GuideId = GuideId(ActionId {
        lamport: 1,
        actor: ACTOR,
    });
    const NO_GUIDE: GuideId = GuideId(ActionId {
        lamport: 8,
        actor: ACTOR,
    });

    const HOLE: (Vec2, Vec2) = (Vec2::ZERO, Vec2::splat(64.0));

    /// A paint layer, a framed matte, a filter and one guide — one of everything the
    /// ctx-free fold can refuse.
    fn furnished() -> DocState {
        DocState::with_layer(A)
            .insert_matte(
                MATTE,
                None,
                Place::Bottom,
                MatteRegion::OutsideRect {
                    min: HOLE.0,
                    max: HOLE.1,
                },
                Parcel::Solid(Srgb::new([0.2, 0.4, 0.6])),
            )
            .insert_filter(FILTER, None, None, Filter::Color(ColorAdjust::NEUTRAL))
            .insert_guide(G1, PerspectiveGuide::default(), None, None)
    }

    fn noop(kind: ActionKind, state: &DocState) -> bool {
        is_noop_on(&kind, state, ACTOR)
    }

    /// The first drift: `set_layer_blend` refuses a filter (§21.4), so the action
    /// folds to nothing — and the hand-written arm compared the mode anyway.
    #[test]
    fn a_blend_the_filter_refuses_is_not_an_edit() {
        let state = furnished();
        assert!(noop(
            ActionKind::SetLayerBlend(FILTER, BlendMode::Multiply),
            &state
        ));
        // The same write on paint is the edit it looks like.
        assert!(!noop(
            ActionKind::SetLayerBlend(A, BlendMode::Multiply),
            &state
        ));
    }

    /// The second drift: `set_matte_rect` refuses a rect it cannot measure and a
    /// region with no rect to move, and the hand-written arm compared corners anyway.
    #[test]
    fn a_rect_the_matte_refuses_is_not_an_edit() {
        let state = furnished();
        let nan = Vec2::splat(f32::NAN);
        assert!(noop(ActionKind::SetMatteRect(MATTE, nan, nan), &state));
        // The rect it already holds, and a real move.
        assert!(noop(
            ActionKind::SetMatteRect(MATTE, HOLE.0, HOLE.1),
            &state
        ));
        assert!(!noop(
            ActionKind::SetMatteRect(MATTE, HOLE.0, HOLE.1 + Vec2::ONE),
            &state
        ));
        // A backing has no rect (§15.5): `with_rect` leaves it alone.
        let backing = state.set_matte_region(MATTE, MatteRegion::Everything);
        assert!(noop(
            ActionKind::SetMatteRect(MATTE, HOLE.0, HOLE.1),
            &backing
        ));
    }

    /// A structural write answers `false` off the footprint, before any fold: an add
    /// of an id the document already holds folds to nothing, and still logs.
    #[test]
    fn a_structural_write_is_never_a_noop() {
        let state = furnished();
        assert!(!noop(
            ActionKind::AddLayer {
                id: A,
                carrier: None,
                above: None,
            },
            &state
        ));
        assert!(!noop(
            ActionKind::RemoveLayer {
                id: ABSENT,
                carried: Vec::new(),
            },
            &state
        ));
        assert!(!noop(
            ActionKind::MoveLayer {
                id: A,
                carrier: None,
                at: Place::Bottom,
            },
            &state
        ));
        // The renderer's half declines without folding at all — this one writes
        // only the author's mask, so it is the fold's partition that answers.
        assert!(!noop(ActionKind::InvertSelection, &state));
        assert!(!noop(
            ActionKind::Undo(ActionId {
                lamport: 0,
                actor: ACTOR,
            }),
            &state
        ));
    }

    /// An action naming a layer the document lacks is logged (§12.1) — and the
    /// engine's expansion of an absent layer into *no* moves is still the no-op it
    /// relies on.
    #[test]
    fn naming_an_absent_layer_is_an_edit() {
        let state = furnished();
        assert!(!noop(ActionKind::SetLayerOpacity(ABSENT, 1.0), &state));
        assert!(!noop(ActionKind::SetLayerName(ABSENT, None), &state));
        assert!(!noop(
            ActionKind::TranslateLayers {
                moves: vec![(ABSENT, IVec2::ZERO)],
            },
            &state
        ));
        assert!(noop(
            ActionKind::TranslateLayers { moves: Vec::new() },
            &state
        ));
        // …where the same value on a layer that is there is not one.
        assert!(noop(ActionKind::SetLayerOpacity(A, 1.0), &state));
        assert!(noop(ActionKind::SetLayerName(A, None), &state));
    }

    /// A guide edit answers off the roster it leaves: the same camera is nothing,
    /// a guide the roster lacks is nothing, and a real reshape is an edit.
    #[test]
    fn a_guide_edit_answers_off_the_roster_it_leaves() {
        let state = furnished();
        assert!(noop(
            ActionKind::SetGuide(G1, PerspectiveGuide::default()),
            &state
        ));
        assert!(noop(
            ActionKind::SetGuide(NO_GUIDE, PerspectiveGuide::default()),
            &state
        ));
        assert!(noop(ActionKind::RemoveGuide(NO_GUIDE), &state));
        assert!(!noop(ActionKind::RemoveGuide(G1), &state));
        assert!(!noop(
            ActionKind::SetGuideName(G1, Some("horizon".into())),
            &state
        ));
    }
}
