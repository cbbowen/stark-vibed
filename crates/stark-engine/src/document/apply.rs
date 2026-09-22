//! Applying an action: the fold that turns the log into tiles (§4, §5).
//!
//! [`Action`] is `stark-model`'s; what *happens* when one is applied needs the
//! renderers, the tile pool and the canvas substrates, so it lives here. The two
//! halves meet at [`Materialize`], with [`DocState`] as the state (§2).

use std::rc::Rc;
use std::sync::Arc;

use crate::gpu::substrate::Substrate;
use stark_model::AssetId;
use stark_model::document::{
    Action, ActionId, ActionKind, ActorId, FillOp, Footprint, LayerId, Materialize, Resource,
    SelectionOp, StrokeRecord, TransformMap, compute_footprint,
};
use stark_model::geom::{IVec2, TileRect};

use super::layer::PropValue;
use super::liquify::LiquifyRun;
use super::selection::Selection;
use super::state::DocState;
use crate::gpu::selection::SelectionRenderer;
use crate::gpu::stroke::StrokeRenderer;
use crate::gpu::tile::TilePool;

/// Side-channel passed through [`Materialize::fold`]: the GPU resources needed to
/// render a stroke (§5).
///
/// Every field is a handle, so the type has no borrow lifetime — which is what lets
/// it be the `Action::Context` — and a sibling's copy shares the resources rather
/// than doubling them ([`for_sibling`](Self::for_sibling)).
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
    /// and the pixels ride here. Unlike those two it is the **document's**: a reset
    /// keeps only what the next log names, and a sibling starts empty.
    pub(crate) pictures: crate::pictures::PictureStore,
    /// The device, so a canvas substrate can be built here on demand.
    pub(crate) gpu: crate::gpu::context::GpuContext,
    /// The canvas substrates and the bytes registered for them (§6.4).
    ///
    /// Here rather than only beside the compositor, because the **deposit reads it**:
    /// the tooth gates the paint a stroke lays by the substrate under it. Asked with
    /// [`DocState::substrate`](super::state::DocState) and its scale *as the log stood
    /// at that action*, so a stroke from before a mid-document `SetSubstrate` deposits
    /// against the substrate it was painted on, at the size it was laid at, on replay
    /// and on a peer alike.
    pub(crate) substrates: crate::gpu::registry::Registry<Substrate>,
    /// A stroke the live preview has already drawn, for the `CommitStroke` fold to
    /// take rather than render again — see [`PreparedStroke`].
    ///
    /// Transient: filled immediately before the push that commits the stroke and
    /// `None` at every other moment, including in every sibling. The fold empties it by
    /// *taking* it, so a slot still full after the push means the offer was declined.
    /// Written through [`offer`](Self::offer) and [`reclaim`](Self::reclaim).
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
    /// which is exactly "the fold did not use it". The slot is emptied either way.
    pub(crate) fn reclaim(&mut self) -> bool {
        self.prepared.take().is_some()
    }

    /// The context a sibling engine opens with (`Engine::new_sharing`, §11): every
    /// renderer, the pool and the stores shared by handle — **except** the pictures,
    /// which start afresh, and the prepared slot, which starts empty.
    ///
    /// The pictures are this document's, and a shared store would keep them alive in
    /// every sibling — and in any `EngineShared` a frontend holds on to — past the reset
    /// that lets them go. The slot is a message to one fold, and a copy taken while it
    /// was full would pin a stroke's worth of fresh tiles for the sibling's life. A
    /// method rather than `Clone`, since a clone that drops both is not one.
    pub(crate) fn for_sibling(&self) -> Self {
        Self {
            pool: self.pool.clone(),
            stroke: self.stroke.clone(),
            assets: self.assets.clone(),
            selection: self.selection.clone(),
            transform: self.transform.clone(),
            fill: self.fill.clone(),
            merge: self.merge.clone(),
            place: self.place.clone(),
            pictures: crate::pictures::PictureStore::new(),
            gpu: self.gpu.clone(),
            substrates: self.substrates.clone(),
            prepared: None,
        }
    }
}

/// A stroke whose tiles the live preview has already rendered, offered to the fold
/// so that its `CommitStroke` takes them instead of rendering the whole stroke again
/// at pen-up (§6.2, §17.6). `preview == committed` (§1.3) is what makes the two the
/// same picture, to the bit.
///
/// **What the fold checks, and what it cannot.** The fold verifies that the record is
/// this action's (`rec`) and that the tiles were drawn over this state's own layer —
/// `base`, compared by *identity*, which is exact for a persistent map. It cannot see
/// the rest of the scene the render read — the author's selection and the canvas
/// substrate — so the engine vouches for those: an offer is only made while nothing
/// has replaced the document since the tiles were drawn (`Preview::invalidate`).
///
/// Kept for one push rather than cached: the tiles hold every fresh tile the stroke
/// wrote, so a slot that outlived its commit would pin a stroke's worth of GPU memory.
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
    /// other effect, which leaves the run in place
    /// ([`Layer::with_painted`](super::layer::Layer::with_painted)).
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
    /// Returns an owned handle rather than a borrow — two atomic bumps — so the caller
    /// can borrow other fields of `self` to build the scene around it.
    ///
    /// [`Registry::get`]: crate::gpu::registry::Registry::get
    pub(crate) fn substrate(&self, substrate: Substrate) -> crate::gpu::substrate::SubstrateMap {
        self.substrates.get(&self.gpu, substrate)
    }

    /// The mask a paint edit by `actor` lays through: the **author's** (§17.3), brought
    /// into the frame the edit is stated in (§14.12) and scoped to the tiles `claim`
    /// reaches — the mask itself, by handle, whenever the frame is zero.
    pub(crate) fn author_gate(
        &self,
        state: &DocState,
        actor: ActorId,
        frame: IVec2,
        claim: TileRect,
    ) -> Selection {
        self.transform
            .shifted_selection_in(&self.pool, &state.selection_of(actor), frame, claim)
    }
}

/// A deterministic refusal (§12.1): the state as it arrived, and a warning saying why.
/// Only the fold says so — a preview of the same refusal shows the committed document
/// and says nothing, since it would say it on every pointer move.
fn declined(state: DocState, why: &'static str) -> DocState {
    tracing::warn!("{why}; ignored");
    state
}

/// Why [`transform_onto`] declines, for the three arms that call it.
const TRANSFORM_DECLINED: &str =
    "transform declined (no paint layer, unusable map or too many tiles)";

/// `state` with `op` filled into `layer` through the author's mask (§18.0.4): the
/// `Fill` arm of the fold, and every preview of a fill (§22.4).
///
/// `op` and `frame` are the action's: the op stated in the layer's frame, and where
/// that frame sits (§14.12). `None` is a deterministic refusal — a layer holding no
/// paint (§15.7), or an op the renderer declines as unbounded or past the tile cap.
pub(crate) fn fill_onto(
    state: &DocState,
    ctx: &ApplyCtx,
    actor: ActorId,
    layer: LayerId,
    op: &FillOp,
    frame: IVec2,
) -> Option<DocState> {
    let base = paint_base(state, layer)?;
    let gate = ctx.author_gate(state, actor, frame, stark_model::document::fill_rect(op));
    let tiles = ctx.fill.apply(&ctx.pool, base, &gate, op)?;
    // A fill leaves the liquify run in place (§6.13).
    Some(state.map_layer(layer, |l| l.with_tiles(tiles)))
}

/// `state` with the author's selected paint on `layer` cut, restacked under `map` and
/// the mask carried with it (§16): the three transform arms of the fold, and their
/// preview (§16.6).
///
/// `map` is stated on the canvas and `frame` is the action's (§14.12): the renderer runs
/// the map conjugated into the frame on the paint and the canvas map on the mask. `None`
/// is a deterministic refusal — a layer holding no paint, an unusable map, or a plan
/// past the tile caps (§16.1).
pub(crate) fn transform_onto(
    state: &DocState,
    ctx: &ApplyCtx,
    actor: ActorId,
    layer: LayerId,
    map: &TransformMap,
    frame: IVec2,
) -> Option<DocState> {
    let base = paint_base(state, layer)?;
    let (tiles, moved) =
        ctx.transform
            .apply(&ctx.pool, base, &state.selection_of(actor), map, frame)?;
    Some(
        state
            .map_layer(layer, |l| l.with_tiles(tiles))
            .with_selection(actor, moved),
    )
}

/// `state` with `op` folded into the author's own mask (§6.8, §17.3): the `Select` arm
/// of the fold, and the marquee's live preview. `None` for an op too large to
/// rasterize (`MAX_SELECTION_TILES`), a bound that is a pure function of the op.
pub(crate) fn select_onto(
    state: &DocState,
    ctx: &ApplyCtx,
    actor: ActorId,
    op: &SelectionOp,
) -> Option<DocState> {
    let selection = ctx
        .selection
        .apply(&ctx.pool, &state.selection_of(actor), op)?;
    Some(state.with_selection(actor, selection))
}

/// The body of [`ActionKind::CommitStroke`] (§6.2): the stroke laid on its layer
/// through the author's mask — from the preview's tiles where the preview drew this
/// very stroke over this very base ([`PreparedStroke`]), and rendered here otherwise,
/// which is every replay, every peer's copy and every redo.
///
/// Refused, and deterministically, on a layer holding no paint (§15.7) and for a record
/// every render refuses whole ([`StrokeRenderer::refuses`]).
fn stroke_apply(
    state: DocState,
    ctx: &mut ApplyCtx,
    actor: ActorId,
    rec: &StrokeRecord,
) -> DocState {
    if ctx.stroke.refuses(rec) {
        return state;
    }
    let Some(target) = state.layer(rec.layer) else {
        return state;
    };
    let Some(base) = target.tiles() else {
        return state;
    };
    let run = target.liquify_run();
    let (tiles, liquify) = match ctx.prepared.take_if(|p| p.is_render_of(rec, base, run)) {
        Some(prepared) => (prepared.tiles, prepared.liquify),
        // The gate and the substrate are resolved only here: a pen-up that takes the
        // preview's tiles on a translated layer would otherwise pay for a mask shift.
        None => {
            // The substrate as the log stood at this action (§6.4), so a mid-document
            // `SetSubstrate` changes only what comes after it.
            let substrate = ctx.substrate(state.substrate());
            let gate = ctx.author_gate(
                &state,
                actor,
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
    state.map_layer(rec.layer, |l| l.with_painted(tiles, liquify))
}

/// The body of [`ActionKind::PlaceImage`] (§23): the layer `add` mints — by exactly
/// the call an `AddLayer` makes, so an unknown carrier declines it identically —
/// holding the picture `image` with its top-left texel at `at`.
///
/// The tiles are built only once the layer is known to have landed: for a layer that
/// is not there they would be a photograph's worth of GPU memory for nothing. They
/// come from the CPU rather than from a pass, which makes them the same bytes on every
/// adapter (`gpu::place`).
fn place_apply(
    state: DocState,
    ctx: &ApplyCtx,
    id: LayerId,
    add: impl FnOnce(&DocState) -> DocState,
    at: IVec2,
    image: AssetId,
) -> DocState {
    // A held id is refused outright, as float and duplicate refuse theirs:
    // `insert_layer` would decline it, but the name and the tiles would still land on
    // the layer that holds it.
    if state.contains_layer(id) {
        tracing::warn!(
            ?id,
            "placement names a layer the document already holds; ignored"
        );
        return state;
    }
    let state = add(&state);
    if !state.contains_layer(id) {
        return state;
    }
    // Absent means the picture has not arrived, which the loader and the transport
    // both rule out before this runs (`unresolved_content`, and the waitlist parking
    // the action until its content lands).
    let Some(picture) = ctx.pictures.get(image) else {
        tracing::warn!(?image, "placing an image this session does not hold");
        return state;
    };
    match ctx.place.render(&ctx.pool, at, &picture) {
        Some(tiles) => state.map_layer(id, |l| l.with_tiles(tiles)),
        None => {
            // Off the tile grid an `i32` can address. The layer stays — it is what the
            // action minted, and withdrawing it would leave the action half-applied.
            tracing::warn!("placed image is off the addressable canvas; no tiles written");
            state
        }
    }
}

/// The body of [`ActionKind::MergeLayerDown`] (§14.11): fold `source` into the
/// layer beneath it — the one action whose promise is about pixels that do **not**
/// move.
///
/// The plan is **re-derived from the state being folded over** rather than trusted
/// from the log ([`merge::offer`](super::merge::offer), which the command and the
/// panel ask too), so a replay and a peer decide from the same document — and a plan
/// that no longer names `dest` (a concurrent reorder, a mode set on either layer
/// since) declines the whole action, leaving the document untouched. Deterministic,
/// so everyone declines together.
fn merge_apply(state: DocState, ctx: &ApplyCtx, source: LayerId, dest: LayerId) -> DocState {
    let Some(plan) = super::merge::offer(&state, source) else {
        return declined(state, "merge down no longer offered here");
    };
    if plan.dest != dest {
        tracing::warn!("merge down no longer names this destination; ignored");
        return state;
    }
    let Some(lower) = paint_base(&state, dest) else {
        return state;
    };
    // What the survivor's tiles become, and what its params become — `None` for
    // "its own, untouched", which is the whole content of a filter merge.
    let (tiles, keeps) = match plan.kind {
        // A **filter** source is the other kind of merge (§14.11.7): nothing is
        // stacked, so the destination's channels are rewritten where they stand and
        // every other thing about that layer — its blend, its clip, its opacity, its
        // place — is left exactly as it was.
        super::merge::MergeKind::Filter {
            filter,
            source_params,
        } => {
            // A **neutral** filter contributes nothing to the picture — the draw list
            // leaves one out (§21.3) — so the merge that must not change the picture
            // has nothing to write. Running it would spend a pass per tile to land the
            // identity *plus* one round trip's rounding.
            let merged = if filter.is_neutral() {
                lower.clone()
            } else {
                ctx.merge.apply_filter(
                    &ctx.pool,
                    lower,
                    // Built through the very constructor the draw list goes through,
                    // off the very params the compositor reads (§21.4) — so the merged
                    // tile is what the screen was showing, not a second reading of the
                    // same layer.
                    &crate::gpu::composite::FilterDraw::new(filter, source_params),
                )
            };
            (merged, None)
        }
        // Every number here is the plan's. The two kinds differ by where the
        // destination's slider belongs — folded into the tiles beside a sibling, left
        // on the layer when the destination is a carrier and the slider is the group's
        // (§14.7) — and `plan` decides that once rather than twice.
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
            // The source restated in the destination's frame (§14.12), by the
            // whole-pixel shift whose tile caps `offer` has already checked
            // (`merge::frame_bakes`), so the transform's exactness (§16.4) holds and the
            // merge that must not change the picture does not.
            let frame_of = |id: LayerId| state.layer(id).map_or(IVec2::ZERO, |l| l.translation);
            let shift = frame_of(source) - frame_of(dest);
            let upper = if shift == IVec2::ZERO {
                upper.clone()
            } else {
                let map = TransformMap::Affine(stark_model::geom::Affine2::from_translation(
                    shift.as_vec2(),
                ));
                match ctx.transform.apply(
                    &ctx.pool,
                    upper,
                    &Selection::everything(),
                    &map,
                    IVec2::ZERO,
                ) {
                    Some((tiles, _)) => tiles,
                    None => return declined(state, "merge across frames exceeds the tile caps"),
                }
            };
            let tiles = ctx.merge.apply(
                &ctx.pool,
                crate::gpu::merge::MergeScene {
                    lower: crate::gpu::merge::MergeSide {
                        tiles: lower,
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
        .map_layer(dest, |l| {
            let mut merged = l.with_tiles(tiles);
            merged.composite = keeps.unwrap_or(l.composite);
            merged
        })
        .remove_layer(source)
}

/// Whether a float of `layer` by `actor`, at the frame `frame`, would cut anything
/// (§16.12) — every refusal [`float_apply`] makes of the source and the selection: a
/// layer holding no paint ([`paint_base`]), a universal selection (nothing is
/// *selected*; moving everything is `TranslateLayers`), and an empty or oversized cut
/// (`plan_float`). What the command asks before it spends an action.
pub(crate) fn float_cut_offered(
    state: &DocState,
    actor: ActorId,
    layer: LayerId,
    frame: IVec2,
) -> bool {
    let selection = state.selection_of(actor);
    !selection.is_universal()
        && paint_base(state, layer)
            .is_some_and(|base| super::transform::plan_float(base, &selection, frame).is_some())
}

/// The body of [`ActionKind::FloatSelection`] (§16.12): cut what the author's
/// selection holds on `layer` into the minted `child`, carried at the foot of the
/// source's stack, and consume the selection — the float *is* the selection now.
///
/// Every refusal is deterministic, so peers and replays agree: those of
/// [`float_cut_offered`], and a child id the document already holds (two layers under
/// one id is §17.9's failure).
fn float_apply(
    state: DocState,
    ctx: &ApplyCtx,
    actor: ActorId,
    layer: LayerId,
    child: LayerId,
    translation: IVec2,
    number: Option<u32>,
) -> DocState {
    if !float_cut_offered(&state, actor, layer, translation) {
        return declined(
            state,
            "float declined (no paint layer, nothing selected, or an empty or oversized cut)",
        );
    }
    if state.contains_layer(child) {
        return declined(state, "float names a child the document already holds");
    }
    let selection = state.selection_of(actor);
    let cut = paint_base(&state, layer).and_then(|base| {
        ctx.transform
            .float(&ctx.pool, base, &selection, translation)
    });
    let Some((remaining, lifted)) = cut else {
        return declined(state, "float cut nothing");
    };
    // Unnamed but numbered — a float is new content rather than a copy, so it is
    // described the way a fresh layer is (§14.6) — at identity params, shown, and
    // standing at the frame the cut was made in. Both the frame and the number are
    // the action's own fields rather than the state's, so a replay mints what the
    // recording run minted (§16.12).
    let mut lifted = state.numbered(child, number).with_tiles(lifted);
    lifted.translation = translation;
    state
        .map_layer(layer, |l| l.with_tiles(remaining))
        .insert_float(layer, lifted)
        .with_selection(actor, Selection::everything())
}

/// The tiles `layer` paints into, or `None` if it has none — the gate every action
/// that lays or moves paint passes through first.
///
/// **The refusal is the point.** A matte has no tile map, so a stroke, a fill or a
/// transform aimed at one is turned away rather than swallowed or magically
/// rasterized (§15.7) — and turned away *here*, in the engine, not only in the
/// frontend, which is what keeps replay and peers agreeing about a log that happens
/// to contain such an action. An absent layer reads the same way.
fn paint_base(state: &DocState, layer: LayerId) -> Option<&crate::gpu::tile::TileMap> {
    state.layer(layer).and_then(|l| l.tiles())
}

/// [`DocState`] is what this crate folds a log into (§2, §5): a persistent map of
/// copy-on-write GPU tiles, which is precisely the thing the model cannot name.
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
    /// **Audited like the fold**, and it is the direction §12.6 is really about: what
    /// this writes becomes the replay base for every later version, and a peer that
    /// patched differently diverges with no pixel able to say so. Same checker, gated
    /// the way the fold's is: on [`AUDITED`](Materialize::AUDITED), which only a debug
    /// build sets (see [`super::audit`]).
    fn unfold(&mut self, action: &Action, footprint: &Footprint, previous: &DocState) {
        let before = Self::AUDITED.then(|| self.clone());
        *self = super::patch::unapply(footprint, previous, self);
        if let Some(before) = before {
            Self::audit(&before, self, action, footprint);
        }
    }
}

/// Fold one action into the state — the whole of what applying means (§4).
///
/// **The ten kinds that need a renderer** — a stroke, a placed image, the two
/// selection ops, the three transforms, a merge, a float and a fill — each dispatch to
/// the one function that folds them, which the previews call too. Everything else
/// folds without touching one and is [`apply_pure`]'s, which this match ends by
/// handing over to. Both matches are exhaustive with no `_` arm, so a kind added
/// later has to name itself on one side of the line rather than defaulting into
/// either.
///
/// **Total.** An action that cannot be honoured — a stroke on a missing layer, a
/// transform past the tile caps — returns the state unchanged rather than an error,
/// so every peer declines it identically (`Materialize::fold`).
fn apply(action: &Action, state: DocState, ctx: &mut ApplyCtx) -> DocState {
    let actor = action.id.actor;
    match &action.kind {
        ActionKind::CommitStroke(rec) => stroke_apply(state, ctx, actor, rec),
        ActionKind::PlaceImage {
            id,
            carrier,
            above,
            at,
            name,
            image,
            number,
        } => place_apply(
            state,
            ctx,
            *id,
            |state| {
                state
                    .insert_layer(*id, *carrier, *above, *number)
                    .set_prop(*id, PropValue::Name(name.as_deref().map(Into::into)))
            },
            *at,
            *image,
        ),
        ActionKind::Select(op) => select_onto(&state, ctx, actor, op)
            .unwrap_or_else(|| declined(state, "selection op too large to rasterize")),
        ActionKind::InvertSelection => {
            let prev = state.selection_of(actor);
            let selection = ctx.selection.invert(&ctx.pool, &prev);
            state.with_selection(actor, selection)
        }
        ActionKind::Transform {
            layer,
            affine,
            translation: frame,
        } => transform_onto(
            &state,
            ctx,
            actor,
            *layer,
            &TransformMap::Affine(*affine),
            *frame,
        )
        .unwrap_or_else(|| declined(state, TRANSFORM_DECLINED)),
        ActionKind::TransformPerspective {
            layer,
            map,
            translation: frame,
        } => transform_onto(
            &state,
            ctx,
            actor,
            *layer,
            &TransformMap::Perspective(*map),
            *frame,
        )
        .unwrap_or_else(|| declined(state, TRANSFORM_DECLINED)),
        ActionKind::TransformWarp {
            layer,
            map,
            translation: frame,
        } => transform_onto(
            &state,
            ctx,
            actor,
            *layer,
            &TransformMap::Warp(map.clone()),
            *frame,
        )
        .unwrap_or_else(|| declined(state, TRANSFORM_DECLINED)),
        ActionKind::MergeLayerDown { source, dest } => merge_apply(state, ctx, *source, *dest),
        ActionKind::FloatSelection {
            layer,
            child,
            translation: frame,
            number,
        } => float_apply(state, ctx, actor, *layer, *child, *frame, *number),
        ActionKind::Fill {
            layer,
            op,
            translation: frame,
        } => fill_onto(&state, ctx, actor, *layer, op, *frame).unwrap_or_else(|| {
            declined(
                state,
                "fill declined (no paint layer, unbounded region or too many tiles)",
            )
        }),

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
/// of the fold, answered without a tile pool, a renderer or an adapter. Also what
/// `patch.rs` folds with, so its round trip is checked against the real fold.
///
/// [`Folded::NeedsRenderer`] for the ten kinds that do need one. **The partition is
/// the compiler's**: this function has no [`ApplyCtx`] to name, so an arm that
/// reaches for a renderer does not compile.
///
/// Both matches are exhaustive, so a new kind must name itself on each side of the
/// line — but naming it on the *wrong* side of both compiles, and no match can say so.
/// **That is why every answer here carries a document**: the mistake costs an ignored
/// action rather than a panic on every remote peer's fold and on every action of a
/// loaded file, in release as well as debug.
///
/// Takes the actor rather than the [`Action`] because that is all any arm reads of
/// it: the author's own mask is keyed by it (§17.3) and `SetSelectionOpacity` writes
/// through it. Nothing here reads the Lamport clock.
pub(super) fn apply_pure(kind: &ActionKind, state: DocState, actor: ActorId) -> Folded {
    Folded::Done(match kind {
        ActionKind::AddLayer {
            id,
            carrier,
            above,
            number,
        } => state.insert_layer(*id, *carrier, *above, *number),
        // The copy's tiles are the shared handles the source already holds, so
        // duplicating a layer costs no GPU memory until one of the two is
        // painted on — copy-on-write is what makes this a cheap action rather
        // than a re-render of everything under it (§5.2).
        ActionKind::DuplicateLayer { ids, number } => state.duplicate_layer(ids, *number),
        // **Declined unless the subtree is what the action names** (§12.6). Every id
        // in it is a `Resource::Layer` write, so a group holding one the action does
        // not name — a peer's concurrent add — would write state nothing declared.
        // Asked of the state being folded, so peers and replays decline the same
        // action; `duplicate_layer` refuses on the same terms.
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
        ActionKind::SetLayerBlend(id, blend) => state.set_prop(*id, PropValue::Blend(*blend)),
        ActionKind::SetLayerClip(id, clip) => state.set_prop(*id, PropValue::Clip(*clip)),
        ActionKind::SetLayerOpacity(id, opacity) => {
            state.set_prop(*id, PropValue::Opacity(*opacity))
        }
        ActionKind::SetLayerVisible(id, visible) => {
            state.set_prop(*id, PropValue::Visible(*visible))
        }
        ActionKind::SetLayerName(id, name) => {
            state.set_prop(*id, PropValue::Name(name.as_deref().map(Into::into)))
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
        // The payload is sanitized inside `insert_filter`/`Layer::with_prop` — the
        // funnel sits in state, where replayed files and remote peers land too,
        // not only where a local command is minted (§21.6).
        ActionKind::AddFilter {
            id,
            carrier,
            above,
            filter,
        } => state.insert_filter(*id, *carrier, *above, filter.clone()),
        ActionKind::SetFilter(id, filter) => state.set_prop(*id, PropValue::Filter(filter.clone())),

        // The drawing guides (§20.5). The one family of actions with no pixel on the
        // other side of it: a guide is geometry to construct through, so applying one
        // moves a roster and nothing else.
        //
        // **The id is the adding action's own** (`GuideId`), and the action carries it
        // rather than the fold deriving it: an id derived here is not part of the
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
/// preview half of every setter command (§21.6), or `None` where the commit would be
/// declined, which the preview shows as the committed document.
///
/// A drag previews by folding the very action its release will commit: behind the same
/// [`ActionKind::sanitized`] funnel, through the same function [`apply`]'s arm calls —
/// [`fill_onto`], [`transform_onto`], or the pure half. A preview that skipped the
/// funnel would show the artist a document the release then declines to store — a
/// `preview == committed` break (§1.3).
///
/// **The context is shared, not exclusive**, so a preview cannot take the prepared
/// tiles, and a kind the pure half does not fold and no arm here names — a stroke, and
/// everything that mints or removes a layer — has no preview.
pub(crate) fn preview_of(
    kind: ActionKind,
    state: &DocState,
    actor: ActorId,
    ctx: &ApplyCtx,
) -> Option<DocState> {
    let kind = kind.sanitized();
    match &kind {
        ActionKind::Fill {
            layer,
            op,
            translation: frame,
        } => fill_onto(state, ctx, actor, *layer, op, *frame),
        ActionKind::Transform {
            layer,
            affine,
            translation: frame,
        } => transform_onto(
            state,
            ctx,
            actor,
            *layer,
            &TransformMap::Affine(*affine),
            *frame,
        ),
        ActionKind::TransformPerspective {
            layer,
            map,
            translation: frame,
        } => transform_onto(
            state,
            ctx,
            actor,
            *layer,
            &TransformMap::Perspective(*map),
            *frame,
        ),
        ActionKind::TransformWarp {
            layer,
            map,
            translation: frame,
        } => transform_onto(
            state,
            ctx,
            actor,
            *layer,
            &TransformMap::Warp(map.clone()),
            *frame,
        ),
        // The pure half's partition is exhaustive, so this arm is every other kind.
        other => match apply_pure(other, state.clone(), actor) {
            Folded::Done(shown) => Some(shown),
            Folded::NeedsRenderer(_) => {
                debug_assert!(false, "{} has no preview", other.label());
                None
            }
        },
    }
}

/// Whether applying this to `state` would leave it exactly as it found it — so a
/// command that would spend an undo step on nothing can decline to log one (§5.4).
///
/// **Answered by folding, not by a mirror of the fold**, through the one enumeration
/// of what can differ between two states
/// ([`audit::changes_nothing`](super::audit)).
///
/// Three answers are given before folding, and they are the whole of what is
/// stated here by hand:
///
/// - **An `Undo` is `false`.** The timeline resolves it (§5.4); its fold is the
///   identity whatever it undoes, so the fold is the one witness that cannot say.
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
///   fold's answer instead.
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

    /// The first drift: `Layer::with_prop` refuses a blend on a filter (§21.4), so the
    /// action folds to nothing — and the hand-written arm compared the mode anyway.
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
        let backing = state.set_prop(
            MATTE,
            PropValue::Matte(MatteRegion::Everything, Parcel::Solid(Srgb::WHITE)),
        );
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
                number: None,
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
