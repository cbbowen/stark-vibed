//! Applying an action: the fold that turns the log into tiles (§4, §5).
//!
//! [`Action`] itself — what the log carries, what a peer receives, what a file
//! stores — is `stark-model`'s. This is the other half of the same sentence: what
//! *happens* when one is applied, which needs the renderers, the tile pool and the
//! canvas substrates, and so cannot live where the action does.
//!
//! The two are tied together by [`Materialize`](stark_model::document::Materialize):
//! the model owns *that* an action folds over some state and which actions commute,
//! and [`DocState`] is this crate's answer to what the state is. The orphan rule
//! forced that shape and it is the right one — see §2.

use std::sync::Arc;

use crate::gpu::substrate::Substrate;
use stark_model::document::ActorId;
use stark_model::document::{Action, ActionKind, Footprint, GuideId, Materialize};

use super::layer::{Layer, LayerContent};
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
#[derive(Clone)]
pub struct ApplyCtx {
    pub pool: TilePool,
    pub stroke: StrokeRenderer,
    pub assets: crate::assets::AssetStore,
    pub selection: SelectionRenderer,
    pub transform: crate::gpu::transform::TransformRenderer,
    pub fill: crate::gpu::fill::FillRenderer,
    pub merge: crate::gpu::merge::MergeRenderer,
    /// Builds the tiles of an image brought in from outside the document (§23).
    /// The one renderer here holding no pipeline, because that path has no pass.
    pub place: crate::gpu::place::PlaceRenderer,
    /// The pictures a `PlaceImage` names (§23) — the third content-addressed store,
    /// beside `assets` and `substrates` and for their reason: the log carries the id
    /// and the pixels ride here.
    pub pictures: crate::pictures::PictureStore,
    /// The device, so a canvas substrate can be built here on demand.
    pub gpu: crate::gpu::context::GpuContext,
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
    pub substrates: crate::gpu::registry::Registry<Substrate>,
    /// A stroke the live preview has already drawn, for the `CommitStroke` fold to
    /// take rather than render again — see [`PreparedStroke`].
    ///
    /// Transient: the engine fills it immediately before the push that commits the
    /// stroke and empties it immediately after, so at every other moment — and in
    /// every clone handed to a sibling engine — it is `None`. The fold empties it
    /// only by *taking* it, which is how the engine learns whether the offer was
    /// accepted: a slot still full after the push was declined.
    pub prepared: Option<PreparedStroke>,
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
pub struct PreparedStroke {
    /// The record the tiles are a render of — compared whole, because the commit
    /// re-derives its record from the fitter at pen-up and sanitizes it on the way
    /// in, and either step disagreeing with what was previewed means the preview was
    /// of a different stroke.
    pub rec: stark_model::document::StrokeRecord,
    /// The layer's tiles the render read as its base.
    pub base: crate::gpu::tile::TileMap,
    /// The layer's tiles with the stroke on them: `base` with a fresh handle at
    /// every tile the stroke reached.
    pub tiles: crate::gpu::tile::TileMap,
}

impl PreparedStroke {
    /// Whether these tiles are the render of `rec` over `base` — the fold's half of
    /// the check described on the type.
    fn is_render_of(
        &self,
        rec: &stark_model::document::StrokeRecord,
        base: &crate::gpu::tile::TileMap,
    ) -> bool {
        self.base.ptr_eq(base) && self.rec == *rec
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
    pub fn substrate(&self, substrate: Substrate) -> crate::gpu::substrate::SubstrateMap {
        self.substrates.get(&self.gpu, substrate)
    }
}

/// The shared body of the three transform actions (§16): cut the author's
/// selected paint, restack it under `map`, and carry the author's mask with it.
/// Gated and keyed exactly as a stroke is — the mask comes off the state being
/// folded over, the actor off the action's own id. A matte or absent layer
/// refuses it, like a stroke; an unusable or oversized map is rejected
/// deterministically, so peers and replays agree.
fn transform_apply(
    state: DocState,
    ctx: &ApplyCtx,
    actor: ActorId,
    layer: LayerId,
    map: &stark_model::document::TransformMap,
) -> DocState {
    let Some(base) = paint_base(&state, layer) else {
        return state;
    };
    let selection = state.selection_of(actor);
    match ctx.transform.apply(&ctx.pool, &base, &selection, map) {
        Some((tiles, moved_selection)) => state
            .map_layer(layer, |l| l.with_tiles(tiles))
            .with_selection(actor, moved_selection),
        None => {
            tracing::warn!("transform rejected (unusable map or too many tiles); ignored");
            state
        }
    }
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
    // [`MergeKind`](super::merge::MergeKind)). Saying the difference once here is
    // what the two `return`ing branches this replaced could not.
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
    fn unfold(&mut self, action: &Action, footprint: &Footprint, previous: &DocState) {
        *self = super::patch::unapply(action, footprint, previous, self);
    }
}

/// Fold one action into the state — the whole of what applying means (§4).
///
/// A free function taking the action rather than a method on it, because [`Action`]
/// is `stark-model`'s now and this needs the renderers: the division the orphan rule
/// forced is the one the split is about (§2).
///
/// **Total.** An action that cannot be honoured — a stroke on a missing layer, a
/// transform past the tile caps — returns the state unchanged rather than an error,
/// so every peer declines it identically (`Materialize::fold`).
fn apply(action: &Action, state: DocState, ctx: &mut ApplyCtx) -> DocState {
    match &action.kind {
        ActionKind::CommitStroke(rec) => {
            let Some(base) = paint_base(&state, rec.layer) else {
                return state;
            };
            // The preview's tiles, where the preview drew this very stroke over this
            // very base — see `PreparedStroke`. Otherwise the stroke is rendered here,
            // which is every replay, every peer's copy and every redo: the log carries
            // the stroke, never its pixels.
            let tiles = match ctx.prepared.take_if(|p| p.is_render_of(rec, &base)) {
                Some(prepared) => prepared.tiles,
                None => {
                    // The **author's** selection, as it stood at this point in the
                    // log, gates the stroke (§6.8, §17.3). Read from the state being
                    // folded over, so replay reproduces it exactly; keyed by the
                    // author, so a collaborator's lasso never clips this stroke.
                    let selection = state.selection_of(action.id.actor);
                    // The substrate this stroke was painted on, as the log stood here —
                    // not as it stands now (§6.4). The tooth gates the deposit by it,
                    // so a mid-document `SetSubstrate` changes what comes *after* it and
                    // nothing before, on replay exactly as it did live.
                    let substrate = ctx.substrate(state.substrate());
                    ctx.stroke.render(
                        crate::gpu::stroke::StrokeScene {
                            pool: &ctx.pool,
                            assets: &ctx.assets,
                            base: &base,
                            selection: &selection,
                            substrate: &substrate,
                        },
                        rec,
                    )
                }
            };
            state.map_layer(rec.layer, |l| l.with_tiles(tiles))
        }
        ActionKind::AddLayer { id, carrier, above } => state.insert_layer(*id, *carrier, *above),
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
        // The author's own mask, keyed like every other selection edit off
        // `action.id.actor`. Rasterizes nothing at all: the strength is how the mask
        // is read, so this touches one number and no tile (§6.8).
        ActionKind::SetSelectionOpacity(opacity) => {
            state.with_selection_opacity(action.id.actor, *opacity)
        }
        ActionKind::SetSubstrate(id) => state.with_substrate(*id),
        ActionKind::SetSubstrateScale(scale) => state.with_substrate_scale(*scale),
        ActionKind::AddMatte {
            id,
            carrier,
            at,
            region,
            paint,
        } => state.insert_matte(*id, *carrier, *at, *region, paint.clone()),
        // The payload is sanitized inside `insert_filter`/`set_filter` — the
        // funnel sits in state, where replayed files and remote peers land too,
        // not only where a local command is minted (§21.5).
        ActionKind::AddFilter {
            id,
            carrier,
            above,
            filter,
        } => state.insert_filter(*id, *carrier, *above, filter.clone()),
        ActionKind::SetFilter(id, filter) => state.set_filter(*id, filter.clone()),
        ActionKind::SetMatteRect(id, min, max) => state.set_matte_rect(*id, *min, *max),
        ActionKind::SetMattePaint(id, paint) => state.set_matte_paint(*id, paint.clone()),
        ActionKind::SetSubstrateColor(rgb) => state.with_substrate_color(*rgb),

        // The drawing guides (§20.5). The one family of actions with no pixel on
        // the other side of it: a guide is geometry to construct through, so
        // applying one moves a roster and nothing else — which is also why none
        // of these touches `ctx`.
        //
        // **The id is the action's own** (`GuideId`), which is what an add mints
        // by being applied at all. Everything else names one that was minted
        // earlier and no-ops when it is not there, the way every action naming an
        // absent layer does.
        ActionKind::AddGuide { guide, after, name } => state.insert_guide(
            GuideId(action.id),
            *guide,
            *after,
            name.as_deref().map(Arc::from),
        ),
        ActionKind::RemoveGuide(id) => state.remove_guide(*id),
        ActionKind::SetGuide(id, guide) => state.set_guide(*id, *guide),
        ActionKind::SetGuideName(id, name) => {
            state.set_guide_name(*id, name.as_deref().map(Arc::from))
        }
        ActionKind::MoveGuide { id, after } => state.move_guide(*id, *after),
        // Cut the author's selected paint, restack it under the affine, and
        // carry the author's mask with it (§16). Gated and
        // keyed exactly as a stroke is: the mask comes off the state being
        // folded over, the actor off the action's own id. A matte or absent
        // layer refuses it, like a stroke; an unusable or oversized transform
        // is rejected deterministically, so peers and replays agree.
        ActionKind::Transform { layer, affine } => transform_apply(
            state,
            ctx,
            action.id.actor,
            *layer,
            &stark_model::document::TransformMap::Affine(*affine),
        ),
        // The rect-scoped siblings (§16.8, §16.9): identical shape — cut,
        // restack, carry the mask — differing only in the map handed to
        // the renderer.
        ActionKind::TransformPerspective { layer, map } => transform_apply(
            state,
            ctx,
            action.id.actor,
            *layer,
            &stark_model::document::TransformMap::Perspective(*map),
        ),
        ActionKind::TransformWarp { layer, map } => transform_apply(
            state,
            ctx,
            action.id.actor,
            *layer,
            &stark_model::document::TransformMap::Warp(map.clone()),
        ),
        // Fold two layers into one without moving a pixel of the composite
        // (§14.11) — the orchestration is `merge_apply`, beside `transform_apply`
        // and for its reason.
        ActionKind::MergeLayerDown { source, dest } => merge_apply(state, ctx, *source, *dest),
        // Lay a parcel of paint through the region's coverage, gated by the
        // author's selection — the same gate a stroke passes through, so a fill
        // is clipped by a selection exactly as a brush is
        // (§18.0.4). Refused on a matte or absent layer like a stroke; refused
        // deterministically when unbounded or oversized, so peers and replays
        // agree about a log that contains one.
        ActionKind::Fill { layer, op } => {
            let Some(base) = paint_base(&state, *layer) else {
                return state;
            };
            let selection = state.selection_of(action.id.actor);
            match ctx.fill.apply(&ctx.pool, &base, &selection, op) {
                Some(tiles) => state.map_layer(*layer, |l| l.with_tiles(tiles)),
                None => {
                    tracing::warn!("fill rejected (unbounded region or too many tiles); ignored");
                    state
                }
            }
        }
    }
}

/// Whether applying this to `state` would leave it exactly as it found it —
/// so a command that would spend an undo step on nothing can decline to log
/// one (§5.4).
///
/// **The question belongs here, beside `apply`.** It was asked six times over
/// in the engine's command handler, in four different shapes, each reaching
/// into a layer's content to ask what kind of layer it is — and the comments
/// there say why that is uncomfortable: it is "a second rule about what a matte
/// is, kept somewhere `apply` cannot see". The four that had a check were not
/// the four that needed one, either: `SetLayerVisible`, `SetLayerClip`,
/// `SetMatteRect` and `SetSubstrateColor` had none, so setting a value to the value
/// it already held cost an undo step that appears to do nothing when reached.
///
/// Asked of the action **as it will be logged**. The engine sanitizes at mint
/// (§21.5, §6.3) so that replay puts back what was applied rather than
/// re-deriving it, which means the payload compared here is the payload that
/// would be stored — a slider reporting `0.6000001` is clamped before it gets
/// this far.
///
/// Conservative in the one safe direction: everything whose effect is pixels
/// answers `false`. A stroke, a fill or a transform *could* leave a layer
/// byte-identical, but finding that out means doing the work, and the point of
/// the question is to avoid it. A false "no" costs an undo step; a false "yes"
/// would silently drop an edit.
///
/// An action naming a layer that does not exist answers `false` too, and
/// deliberately: it is inert *here*, but the same action reaching a peer whose
/// tree is one step ahead is not, and a log that omits it would be a different
/// log on the two clients (§12.1).
///
/// **Exhaustive, with no `_` arm**, for [`minted_layers`](Self::minted_layers)'s
/// reason: a variant added later must be made to answer rather than defaulted
/// into the safe answer and forgotten.
pub(crate) fn is_noop_on(kind: &ActionKind, state: &DocState, actor: ActorId) -> bool {
    // The layer this action names, or `false` from every arm below when it is
    // absent — see the doc comment.
    let layer = |id: LayerId| state.layer(id);
    match kind {
        ActionKind::SetLayerOpacity(id, opacity) => {
            layer(*id).is_some_and(|l| l.composite.opacity == *opacity)
        }
        // Asked of the author's own mask, since that is the only one an action can
        // address (§17.3) — and answerable at all only because this action moves no
        // tiles: a slider dragged out and back logs nothing. With nothing selected
        // the number is still a fact — the strength the next region takes, and
        // the whole canvas's meanwhile (`Selection::opacity`) — so it is compared
        // like any other, and logged when it moves.
        ActionKind::SetSelectionOpacity(opacity) => {
            state.selection_of(actor).opacity() == *opacity
        }
        ActionKind::SetLayerBlend(id, blend) => {
            layer(*id).is_some_and(|l| l.composite.blend == *blend)
        }
        ActionKind::SetLayerClip(id, clip) => layer(*id).is_some_and(|l| l.composite.clip == *clip),
        ActionKind::SetLayerVisible(id, visible) => {
            layer(*id).is_some_and(|l| l.visible == *visible)
        }
        // Compared against the *stored* name, which is `None` for a layer that
        // has never been named — so clearing an unnamed layer's name is the
        // no-op it looks like. Commit-on-blur makes this the common case:
        // leaving a field you only looked at must cost nothing.
        ActionKind::SetLayerName(id, name) => {
            layer(*id).is_some() && state.layer_name(*id) == name.as_deref()
        }
        // Asked of the layer's *content*, since only a filter has a filter to
        // compare. A non-filter layer answers `false` and logs an action `apply`
        // will no-op, rather than growing a rule about what a filter is.
        ActionKind::SetFilter(id, filter) => {
            layer(*id).and_then(|l| l.filter()).as_ref() == Some(filter)
        }
        ActionKind::SetMattePaint(id, paint) => matches!(
            layer(*id).map(|l| &l.content),
            Some(LayerContent::Matte { paint: current, .. }) if current == paint
        ),
        // A region with no rect answers `false`: `with_rect` leaves it alone, so
        // the action is inert, but saying so here would be this file's third
        // opinion about what an `Everything` matte is.
        ActionKind::SetMatteRect(id, min, max) => {
            layer(*id)
                .and_then(|l| l.matte_region())
                .and_then(|r| r.rect())
                == Some((*min, *max))
        }
        ActionKind::SetSubstrateColor(rgb) => state.substrate_color == *rgb,
        ActionKind::SetSubstrate(id) => state.substrate == *id,
        ActionKind::SetSubstrateScale(scale) => state.substrate_scale == *scale,
        // A guide edit that changes nothing, asked of the guide as it stands —
        // and answering `false` for one that is not there, on this function's
        // general rule: inert here, live on a peer whose roster is a step ahead.
        ActionKind::SetGuide(id, guide) => {
            state.guide(*id).is_some_and(|g| g.camera == *guide)
        }
        // Commit-on-blur makes a rename to the name it already has the common
        // case: leaving a field you only looked at must cost nothing.
        ActionKind::SetGuideName(id, name) => {
            state.guide(*id).is_some_and(|g| g.name.as_deref() == name.as_deref())
        }
        // A move is nothing when the guide already sits directly after the anchor.
        // Asked of the roster rather than of the drag, so a row dropped back where
        // it was picked up costs no undo step however the frontend spelled it.
        ActionKind::MoveGuide { id, after } => {
            let roster = state.guides();
            let Some(i) = roster.iter().position(|g| g.id == *id) else {
                return false;
            };
            // What it sits after today — `None` at the head of the roster.
            let before = i
                .checked_sub(1)
                .map(|j| roster.get(j).expect("indexed from the roster").id);
            // An anchor no longer in the roster lands the guide at the head, so it
            // asks for the same place naming nothing asks for.
            let anchor = after.filter(|a| roster.iter().any(|g| g.id == *a));
            before == anchor
        }
        // Everything whose effect is pixels, or whose effect depends on a tree
        // walk this question will not pay for.
        ActionKind::CommitStroke(_)
        | ActionKind::Fill { .. }
        | ActionKind::Transform { .. }
        | ActionKind::TransformPerspective { .. }
        | ActionKind::TransformWarp { .. }
        | ActionKind::Select(_)
        | ActionKind::InvertSelection
        | ActionKind::AddLayer { .. }
        | ActionKind::AddMatte { .. }
        | ActionKind::AddFilter { .. }
        | ActionKind::PlaceImage { .. }
        | ActionKind::DuplicateLayer { .. }
        | ActionKind::RemoveLayer { .. }
        | ActionKind::MergeLayerDown { .. }
        | ActionKind::MoveLayer { .. }
        // An add always changes the roster; a remove of an absent guide answers
        // `false` for the reason every other absent-target arm does.
        | ActionKind::AddGuide { .. }
        | ActionKind::RemoveGuide(_)
        | ActionKind::Undo(_) => false,
    }
}
