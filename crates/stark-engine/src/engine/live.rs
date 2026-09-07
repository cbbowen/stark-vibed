//! The live preview fold: the committed document with every in-flight gesture drawn
//! over it (§17.6).
//!
//! Two caches make it affordable — the folded document, and a [`FrozenHead`] per
//! stroke in flight — and an **epoch** keeps them honest: anything that replaces the
//! document a head was composited onto bumps [`Preview::epoch`], and a head stamped
//! with an older one is discarded rather than drawn over a canvas that no longer
//! exists. [`Preview`] owns that state so the slot cannot move without the epoch
//! moving with it.

use std::collections::{BTreeMap, BTreeSet};

use super::Engine;
use crate::document::{ApplyCtx, DocState, PreparedStroke};
use crate::gpu::StrokeSpans;
use crate::gpu::stroke::Progress;
use crate::peer::{GestureView, LiveGesture, Peer};
use crate::projection::Revision;
use stark_model::document::{ActorId, LayerId, StrokeRecord};
use stark_model::geom::{TileCoord, TileRect};

/// What is being *shown* over the committed document, and the caches that make
/// showing it affordable (§17.6).
///
/// The two fields that can invalidate a cached head — the drag-preview document and
/// the committed one underneath it — move only through
/// [`set_doc`](Self::set_doc) and [`invalidate`](Self::invalidate), each of which
/// bumps the epoch as it goes.
#[derive(Default)]
pub(super) struct Preview {
    /// The unlogged document edit in flight: a whole document standing in for the
    /// committed one, since what these edits change — a matte's rect (§15.7), the
    /// substrate color (§15.5) — is document state rather than a tile edit. One
    /// slot, not one per kind: only one such drag can be in flight at a time, and a
    /// second kind starting mid-drag supersedes the first rather than compounding
    /// with it. `None` when nothing is being dragged.
    doc: Option<DocState>,
    /// The committed state (or `doc`) with every in-flight gesture — this client's
    /// and every peer's — drawn over it. `None` when nobody is mid-gesture.
    live: Option<DocState>,
    /// The settled head of each in-flight stroke, keyed by its author (see
    /// [`FrozenHead`]). Every head is rooted at the *committed* document rather than
    /// at the previous peer's preview: chaining would invalidate peer *k*'s cache on
    /// every move by peers before it, collapsing the incremental repaint exactly
    /// when two people are painting at once.
    heads: BTreeMap<ActorId, FrozenHead>,
    /// Bumped whenever the document the previews are composited onto changes. A
    /// [`FrozenHead`] stamped with an older epoch is stale and discarded.
    epoch: Revision,
    /// Bumped every time the fold is **rebuilt** — "what is shown has moved", where
    /// [`epoch`](Self::epoch) is "what is shown was *replaced*".
    ///
    /// The two cannot be one: bumping the epoch per fold would discard every live
    /// stroke's head, yet a fold changes what a renderer should draw while both
    /// `doc_revision` and `epoch` stand still, so a draw list keyed on those two
    /// alone would hold the frame at the moment the stroke began (C4).
    fold: Revision,
    /// Whether the fold no longer reflects the gestures and document it folds — set
    /// by [`Engine::mark_live_stale`], cleared by the [`Engine::flush_live`] that
    /// services it, so N mutations per frame cost one fold.
    stale: bool,
    /// This client's own stroke as the last fold drew it, for its commit to take
    /// rather than render again ([`PreparedStroke`], §6.2). Filled only by a fold
    /// that drew that stroke over the committed document alone, and emptied by any
    /// other fold and by [`invalidate`](Self::invalidate), so it cannot outlive the
    /// base it was drawn over.
    prepared: Option<PreparedStroke>,
}

impl Preview {
    /// Note that the document the previews are drawn over has been replaced, so
    /// every cached head is stale.
    pub(super) fn invalidate(&mut self) {
        self.epoch.bump();
        self.prepared = None;
    }

    /// Install (or, with `None`, drop) the stand-in document for an unlogged edit in
    /// flight. **The only way to move that slot**, so the epoch cannot be left
    /// behind: the gesture previews are composited onto this.
    pub(super) fn set_doc(&mut self, doc: Option<DocState>) {
        self.doc = doc;
        self.invalidate();
    }

    /// Forget everything: no drag in flight, no fold, no heads. For loading a
    /// different document, where none of it describes anything any more.
    pub(super) fn clear(&mut self) {
        self.doc = None;
        self.live = None;
        self.heads.clear();
        self.invalidate();
    }

    /// The document as it should be *shown*: the fold if there is one, else the
    /// unlogged edit in flight, else `committed`.
    pub(super) fn presented<'a>(&'a self, committed: &'a DocState) -> &'a DocState {
        self.live
            .as_ref()
            .or(self.doc.as_ref())
            .unwrap_or(committed)
    }

    /// The unlogged edit in flight, for the presentation properties that must track a
    /// drag rather than lag on the committed value (§15.5, §15.7).
    pub(super) fn doc(&self) -> Option<&DocState> {
        self.doc.as_ref()
    }

    /// How many strokes in flight have a settled head cached — at most one per actor
    /// currently drawing one.
    pub(super) fn head_count(&self) -> usize {
        self.heads.len()
    }

    /// The invalidation epoch a cached head is stamped against, as the number
    /// `render::DrawKey` keys on.
    pub(super) fn epoch(&self) -> u64 {
        self.epoch.get()
    }

    /// This client's stroke as the last fold drew it, for its commit to take — see
    /// [`prepared`](Self::prepared). Taken, not read: the tiles are the commit's now.
    pub(super) fn take_prepared(&mut self) -> Option<PreparedStroke> {
        self.prepared.take()
    }

    /// How many times the fold has been rebuilt — see [`fold`](Self::fold).
    pub(super) fn fold(&self) -> u64 {
        self.fold.get()
    }

    /// Rebuild the fold: `committed` (or the drag standing in for it) with every
    /// in-flight gesture composited over it, in ascending [`ActorId`] order (§17.6).
    ///
    /// The order is fixed and derivable, so every client folds the same picture. A
    /// stroke is rendered against the *committed* base and overlaid tile-wise rather
    /// than chained peer-over-peer, since chaining would invalidate one peer's cached
    /// head on every move of the peers before it.
    ///
    /// **Unless two live strokes share tiles.** The overlay copies whole tiles, so a
    /// tile they share carries the committed pixels plus one stroke's paint whichever
    /// way it is copied, and the other stroke disappears from it until it commits. So
    /// a stroke whose reach meets one already in the fold is rendered over **the
    /// fold** instead, and gives up its cached head to do it. `preview == committed`
    /// still holds tile-wise for each stroke; what stays provisional is which of two
    /// concurrent strokes ends up on top, since that depends on a total order neither
    /// is in until both commit.
    ///
    /// The head cache is rebuilt into a **fresh** map rather than edited in place,
    /// which is what bounds it: a head whose gesture is no longer in flight is dropped
    /// by construction, instead of leaking a whole `DocState`'s worth of
    /// `Arc<GpuTile>` handles the pool cannot reclaim.
    fn rebuild(
        &mut self,
        ctx: &ApplyCtx,
        committed: &DocState,
        local: ActorId,
        gestures: Vec<GestureView>,
    ) {
        // Before the early return as much as after it: dropping the fold is a change
        // to what is shown exactly as building one is, and it is the transition a
        // pen-up makes.
        self.fold.bump();
        // The last fold's prepared tiles are not the ones to commit: this fold either
        // draws the stroke afresh below, or has no such stroke.
        self.prepared = None;
        if gestures.is_empty() {
            self.live = None;
            self.heads.clear();
            return;
        }
        let base = self.doc.clone().unwrap_or_else(|| committed.clone());
        let mut out = base.clone();
        // Last fold's cache to draw from, and this fold's to fill: what is not moved
        // across is released when `cached` drops at the end of the call.
        let mut cached = std::mem::take(&mut self.heads);
        let mut heads = BTreeMap::new();
        // The paint every gesture already folded in has claimed, so a stroke can tell
        // whether the tiles it is about to overlay still hold nothing but the base.
        let mut claimed: Vec<(LayerId, TileRect)> = Vec::new();
        // Which layers any earlier gesture has written at all — a different question
        // from `claimed`. `claimed` asks whether an earlier gesture *overlaps* this
        // stroke, which decides whether the head can be cached; this asks whether
        // `out`'s tile map for the layer is still the base's, which decides whether
        // the overlay can be a swap.
        let mut touched: BTreeSet<LayerId> = BTreeSet::new();
        for GestureView {
            actor,
            gesture,
            ordinal,
            frozen_spans: frozen,
        } in gestures
        {
            match gesture {
                LiveGesture::Selection(op) => {
                    // A marquee previews as the mask it will commit — the very same
                    // call `Select` makes (§6.8), so what is previewed is what lands.
                    let prev = base.selection_of(actor);
                    if let Some(selection) = ctx.selection.apply(&ctx.pool, &prev, &op) {
                        out = out.with_selection(actor, selection);
                    }
                }
                // A fill previews through the same `FillRenderer::apply` the commit
                // makes, over the same base, so what is on screen mid-drag is the
                // result. Redone from the base on each move rather than accumulated,
                // so dragging a rectangle out does not stack a hundred glazes.
                LiveGesture::Fill {
                    layer,
                    op,
                    translation: frame,
                } => {
                    // Already chained: it reads `out`, not `base`, and replaces the
                    // layer's whole tile map rather than copying tiles across.
                    let rect = stark_model::document::fill_rect(&op);
                    claimed.push((layer, rect));
                    touched.insert(layer);
                    if let Some(tiles) = out.layer(layer).and_then(|l| l.tiles()).cloned() {
                        // The author's canvas mask, in the layer's frame the op is
                        // in — the very shift the commit makes (§14.12).
                        let gate = ctx.transform.shifted_selection_in(
                            &ctx.pool,
                            &base.selection_of(actor),
                            frame,
                            rect,
                        );
                        if let Some(filled) = ctx.fill.apply(&ctx.pool, &tiles, &gate, &op) {
                            out = out.map_layer(layer, |l| l.with_tiles(filled));
                        }
                    }
                }
                LiveGesture::Stroke(rec) => {
                    let reach = stark_model::document::stroke_rect(&rec);
                    let contested = claimed
                        .iter()
                        .any(|(layer, rect)| *layer == rec.layer && rect.intersects(&reach));
                    claimed.push((rec.layer, reach));
                    // A contested stroke draws over the fold, a different document on
                    // every move, so its head is unrepeatable: nothing cached is
                    // reused, and what this fold caches is marked never to be.
                    let head = cached.remove(&actor).filter(|h| {
                        !contested
                            && !h.contested
                            && h.epoch == self.epoch
                            && h.gesture == ordinal
                            && h.spans <= frozen
                    });
                    let root = if contested { &out } else { &base };
                    let (head, tail_state) = self.render_live_stroke(
                        ctx,
                        LiveTail {
                            author: actor,
                            gesture: ordinal,
                            frozen,
                            contested,
                        },
                        root,
                        &rec,
                        head,
                    );
                    // **The first stroke on a layer swaps the map instead of copying
                    // it tile by tile.** `head.dirty` is every tile the stroke has
                    // *ever* touched, so walking it is work that grows with the stroke
                    // — exactly what `FrozenHead` exists to stop. Where nothing earlier
                    // in this fold has written the layer, the tail state's map already
                    // *is* the overlay. `overlay_tiles` is for the contested case: two
                    // peers painting one layer.
                    out = if touched.insert(rec.layer) {
                        match tail_state.layer(rec.layer).filter(|l| l.tiles().is_some()) {
                            Some(l) => {
                                let tiles = l.tiles().cloned().expect("filtered on tiles");
                                let run = l.liquify_run().cloned();
                                out.map_layer(rec.layer, |l| l.with_painted(tiles, run))
                            }
                            None => out,
                        }
                    } else {
                        overlay_tiles(&out, rec.layer, &tail_state, &head.dirty)
                    };
                    heads.insert(actor, head);
                    // The tail state is the document this stroke's commit will
                    // produce — provided it is this client's stroke, drawn over the
                    // committed document alone (not a drag's stand-in, not another
                    // stroke's paint) and in its own color rather than the
                    // diagnostic's.
                    let lands_at_commit = actor == local
                        && !contested
                        && self.doc.is_none()
                        && !cfg!(feature = "debug-unfrozen");
                    if lands_at_commit
                        && let Some(from) = base.layer(rec.layer).filter(|l| l.tiles().is_some())
                        && let Some(to) = tail_state.layer(rec.layer)
                        && let Some(tiles) = to.tiles()
                    {
                        self.prepared = Some(PreparedStroke {
                            base: from.tiles().cloned().expect("filtered on tiles"),
                            base_run: from.liquify_run().cloned(),
                            tiles: tiles.clone(),
                            // The run the tail left, if the stroke is a liquify one;
                            // the fold keeps the run in place otherwise, exactly as
                            // `with_painted` does with a `None` (§6.13).
                            liquify: rec.brush.liquify().and_then(|_| to.liquify_run().cloned()),
                            rec,
                        });
                    }
                }
            }
        }
        self.heads = heads;
        self.live = Some(out);
    }

    /// Advance one stroke's frozen head and render its live tail, returning the head
    /// to keep and the state the tail left behind.
    ///
    /// Uses the same entry point a commit does (`StrokeRenderer::render_range`), so
    /// the live preview and the `Action::apply` that replaces it draw the same pixels.
    fn render_live_stroke(
        &self,
        ctx: &ApplyCtx,
        tail: LiveTail,
        base: &DocState,
        rec: &StrokeRecord,
        head: Option<FrozenHead>,
    ) -> (FrozenHead, DocState) {
        let LiveTail {
            author,
            gesture,
            frozen,
            contested,
        } = tail;
        // A span whose pixels are still measured against a length the stroke has not
        // reached cannot be frozen — the taper runs from the *ends* of the whole
        // stroke, which are still under the pointer. Clamped here rather than in the
        // fitter because it is a fact about the brush, not the curve (see
        // `safe_frozen`). Monotone: a prefix admitted once stays admissible, so this
        // can only slow a head down, never invalidate it.
        let frozen = crate::gpu::stroke::safe_frozen(rec, frozen);
        // Nothing cached, or the fit went backwards (a new stroke): start over from
        // the committed document, with a fresh (uncharged) brush.
        let mut head = head.unwrap_or_else(|| FrozenHead {
            spans: 0,
            dist: 0.0,
            tool: None,
            state: base.clone(),
            gesture,
            epoch: self.epoch,
            contested,
            dirty: BTreeSet::new(),
        });
        if frozen > head.spans {
            head = advance_head(ctx, author, head, rec, frozen);
        }

        let all = crate::path::span_count(rec.path.len());
        let tail = StrokeSpans::from_parts(rec, head.spans..all, head.dist);
        // The diagnostic recolors only what this move redrew, so the seam between
        // tinted and untinted paint *is* the freezing boundary. Build-time only: a
        // shipping build paints the tail in the stroke's own color.
        #[cfg(feature = "debug-unfrozen")]
        let tinted = {
            let mut r = rec.clone();
            // An erase stroke has no pigment to tint; its tail simply stays
            // undiagnosed, which the boundary this exists to show survives.
            if let Some(p) = r.brush.paint_mut() {
                p.color = DEBUG_UNFROZEN_COLOR;
            } else if let Some(w) = r.brush.wet_mut() {
                w.color = DEBUG_UNFROZEN_COLOR;
            }
            r
        };
        #[cfg(feature = "debug-unfrozen")]
        let tail_rec = &tinted;
        #[cfg(not(feature = "debug-unfrozen"))]
        let tail_rec = rec;
        // The tail reaches the end of the stroke, so the brush state it leaves is
        // thrown away and rebuilt from the head on the next move — which is what makes
        // the tail re-renderable. Its dirty tiles still join the head's for the
        // overlay.
        let (state, carry) =
            render_span_range(ctx, author, &head.state, tail_rec, tail, head.tool.as_ref());
        if let Progress::Finished { dirty, .. } = carry.progress {
            head.dirty.extend(dirty);
        }
        (head, state)
    }
}

/// **Which** in-flight stroke a render is of, and how much of it has settled — as
/// against what it draws with, which is the rest of
/// [`render_live_stroke`](Preview::render_live_stroke)'s arguments.
///
/// One value because three of the four are the identity a cached [`FrozenHead`] is
/// matched against: a head is kept only if it belongs to this author's *this*
/// gesture, at this contest state, and has not run past what is frozen.
struct LiveTail {
    /// The peer authoring the stroke, whose selection the render reads (§17.3).
    author: ActorId,
    /// The gesture's ordinal, which is what distinguishes this stroke from the next
    /// one by the same author.
    gesture: u64,
    /// How many leading spans the fitter has frozen — an upper bound on what the
    /// head may absorb, and clamped again by `safe_frozen` before it is used.
    frozen: usize,
    /// Whether another peer is painting the same layer, which decides whether the
    /// tail is drawn over the committed document or over the overlay so far (§17.6).
    contested: bool,
}

/// Color the live tail is drawn in under the `debug-unfrozen` feature.
/// Full-opacity magenta: it has to read against paint of any hue, and against the
/// stroke's own color in particular.
#[cfg(feature = "debug-unfrozen")]
const DEBUG_UNFROZEN_COLOR: [f32; 3] = [1.0, 0.0, 1.0];

/// The part of the in-flight stroke that has stopped changing, already composited
/// onto the committed document.
///
/// Re-rendering a live stroke costs (segments × tiles covered), both of which grow
/// with its length. The fitter freezes control points behind the pointer and never
/// revises them
/// ([`PathFitter::frozen_spans`](crate::path::PathFitter::frozen_spans)), so those
/// spans are final: render them once and each move draws only the short live tail
/// over the result. Work per move then follows the tail rather than the stroke.
pub(super) struct FrozenHead {
    /// How many leading spans `state` already has drawn on it.
    spans: usize,
    /// Arc length at the end of those spans — where the tail's `dist` resumes.
    /// Not recoverable from `spans` alone, and the `drain` falloff and color
    /// dynamics both read it (see `gpu::stroke::StrokeSpans`).
    dist: f32,
    /// The brush state the tail resumes from: the sequential stamp loop's reservoir
    /// (`lift`/`deposit`/`charge`), or the erase pass's accumulated extent (§6.12).
    /// `None` for the swept path, which carries nothing between segments. See
    /// [`ToolState`](crate::gpu::stroke::ToolState).
    tool: Option<crate::gpu::stroke::ToolState>,
    state: DocState,
    /// Which gesture this is the head of — its author's ordinal. A head may only be
    /// reused within the *same* gesture, and the span count alone cannot tell a new
    /// stroke from a continued one once the new one has grown past where the old was
    /// frozen.
    gesture: u64,
    /// The base this head was composited onto ([`Preview::epoch`]). Anything that
    /// replaces that base — a commit, an undo, a remote merge, a load, a drag preview
    /// — bumps the epoch, and a head from an earlier one is discarded.
    epoch: Revision,
    /// Whether this head was rooted at the *fold* rather than at the committed
    /// document, because another live stroke had claimed tiles it reaches. The fold is
    /// rebuilt on every move, so such a head can never be reused — it is kept only
    /// because the overlay reads it and [`Preview::head_count`] counts it. The epoch
    /// cannot say this: nothing about the *document* changed.
    contested: bool,
    /// Every tile the head has rewritten so far, so the fold knows what to overlay
    /// (§17.6). Accumulated because a head grows across many advances.
    dirty: BTreeSet<TileCoord>,
}

impl Engine {
    /// The document as it should be *shown*: the fold if there is one, else the
    /// unlogged edit in flight, else the committed state.
    pub(super) fn presented(&self) -> &DocState {
        self.preview.presented(self.timeline.current())
    }

    /// The document as a `Transform` commit of `map` would leave it, built through
    /// the **same renderer** the commit uses, so what is shown is what "Done" will
    /// produce (§16.6). `None` when the layer cannot be transformed (a matte, absent)
    /// or the transform is rejected — the preview then shows the committed document,
    /// matching the commit's refusal.
    pub(super) fn preview_transform(
        &self,
        layer: LayerId,
        map: &stark_model::document::TransformMap,
    ) -> Option<DocState> {
        let doc = self.timeline.current();
        let target = doc.layer(layer)?;
        // The frame the commit will stamp — the layer's, read off the same
        // committed document (`Engine::frame_of`), so preview == committed
        // holds for the conjugation too.
        let frame = target.translation;
        let base = target.tiles()?;
        let selection = doc.selection_of(self.actor());
        let (tiles, moved) = self.shared.apply.transform.apply(
            &self.shared.apply.pool,
            base,
            &selection,
            map,
            frame,
        )?;
        Some(
            doc.map_layer(layer, |l| l.with_tiles(tiles))
                .with_selection(self.actor(), moved),
        )
    }

    /// The document as a `Fill` commit of `op` would leave it — the gradient fill's
    /// composing preview (§22.4), built through the **same** `FillRenderer::apply` the
    /// commit uses, for `preview_transform`'s reason. `None` when the layer cannot
    /// take paint or the fill is refused (unbounded, over the cap), matching the
    /// commit.
    pub(super) fn preview_fill(
        &self,
        layer: LayerId,
        op: &stark_model::document::FillOp,
    ) -> Option<DocState> {
        let doc = self.timeline.current();
        let target = doc.layer(layer)?;
        // The op arrives on the canvas, as the command that will commit it does;
        // both convert into the layer's frame here and at `DocCommand::Fill`
        // through the same pair of calls (§14.12).
        let frame = target.translation;
        let base = target.tiles()?.clone();
        let op = op.translated(-frame.as_vec2());
        let gate = self.shared.apply.transform.shifted_selection_in(
            &self.shared.apply.pool,
            &doc.selection_of(self.actor()),
            frame,
            stark_model::document::fill_rect(&op),
        );
        let tiles = self
            .shared
            .apply
            .fill
            .apply(&self.shared.apply.pool, &base, &gate, &op)?;
        Some(doc.map_layer(layer, |l| l.with_tiles(tiles)))
    }

    /// Install (or, with `None`, drop) the unlogged edit in flight and refold — the
    /// shared tail of every `Preview*` command.
    pub(super) fn set_doc_preview(&mut self, preview: Option<DocState>) {
        self.preview.set_doc(preview);
        self.mark_live_stale();
    }

    /// Note that the fold no longer shows what is there — the one call every
    /// mutation that can change what is on screen ends with.
    ///
    /// A note, not a rebuild: mutations arrive at input rate while presentation is
    /// one frame per rAF, so [`flush_live`](Self::flush_live) rebuilds the fold at the
    /// read, once per frame actually shown. Painting the latest state once shows
    /// exactly what folding it per event would have.
    pub(super) fn mark_live_stale(&mut self) {
        self.preview.stale = true;
    }

    /// Service a pending [`mark_live_stale`](Self::mark_live_stale): rebuild the fold
    /// if anything has changed since it was last built. Every reader of
    /// [`presented`](Self::presented) reaches it through an entry point that flushes
    /// first, so a stale fold is never observable. Idempotent.
    ///
    /// `pub` for harnesses, not frontends: a frontend gets its flush from
    /// [`render`](Self::render) with the frame that shows the result, while the stroke
    /// benchmark must pay the fold explicitly to time it. Not a command, deliberately
    /// (§4): a command is an *input* — logged, replicated, countable — and this
    /// mutates no document or session state, only a cache their content determines.
    #[doc(hidden)]
    pub fn flush_live(&mut self) {
        if !self.preview.stale {
            return;
        }
        // **After the early return**, so the row counts folds rather than calls: this
        // is asked once per painted frame whether or not anything moved, and a mean
        // diluted by the no-op case would answer neither question.
        crate::timing::span!("live.fold");
        self.preview.stale = false;
        let gestures = self.live_gestures();
        // Disjoint fields, so the fold reads the context and the committed document
        // while holding the preview mutably.
        self.preview.rebuild(
            &self.shared.apply,
            self.timeline.current(),
            self.actor(),
            gestures,
        );
    }

    /// Every gesture in flight, in ascending [`ActorId`] order.
    ///
    /// The local client's is *derived* from the session's fitter rather than kept in
    /// the roster: copying it there would make two sources of truth for the one thing
    /// `preview == committed` rests on. Merging the two here gives the uniform
    /// ordering without the duplication (§17.4).
    ///
    /// A hand that is only *hovering* folds the §18.1.10 mark instead — the stroke its
    /// recent reports would have committed, in the same [`GestureView`] shape, so the
    /// fold cannot tell it from a real gesture and `hover == committed` is inherited
    /// rather than maintained. The `match` is what guarantees the two never coexist:
    /// an actor folds at most one gesture, and a fact outranks a hypothesis.
    fn live_gestures(&self) -> Vec<GestureView> {
        let mut out: Vec<GestureView> = Vec::new();
        match self.session.gesture_view(self.actor()) {
            Some(g) => out.push(g),
            None => out.extend(self.session.hover_view(
                self.actor(),
                self.authoring.clock,
                self.frame_of(self.session.active_layer()),
            )),
        }
        out.extend(self.peers.iter().filter_map(Peer::gesture_view));
        out.sort_by_key(|g| g.actor);
        out
    }

    /// Dump the finished stroke's raw input as a pasteable Rust literal.
    ///
    /// A misfit seen in the app is otherwise unreproducible: the fit depends on the
    /// exact sequence of pointer reports — their spacing carries the pen's speed,
    /// which the density policy and the freezing both key off — and no synthetic curve
    /// stands in for a real hand. Compiled away without the feature, along with the
    /// buffer it prints (see [`Engine::note_debug_sample`]).
    #[cfg(feature = "debug-unfrozen")]
    pub(super) fn log_debug_samples(&mut self) {
        if self.debug_samples.is_empty() {
            return;
        }
        // Positions *and* the pen channels: pressure sizes the brush and tilt steers
        // it, both fitted as their own least-squares channels, so a capture without
        // them cannot reproduce a fault in either.
        let mut lit = String::from("&[");
        for (i, s) in self.debug_samples.iter().enumerate() {
            if i > 0 {
                lit.push(',');
            }
            lit.push_str(&format!(
                "[{:.2},{:.2},{:.3},{:.3},{:.3}]",
                s.pos.x, s.pos.y, s.pressure, s.tilt.x, s.tilt.y
            ));
        }
        lit.push(']');
        tracing::info!(
            samples = self.debug_samples.len(),
            "raw stroke [x,y,pressure,tiltx,tilty]: {lit}"
        );
        self.debug_samples.clear();
    }

    #[cfg(not(feature = "debug-unfrozen"))]
    pub(super) fn log_debug_samples(&mut self) {}
}

/// Draw spans `head.spans..frozen` onto the frozen head, so the next move need
/// not draw them again.
fn advance_head(
    ctx: &ApplyCtx,
    author: ActorId,
    head: FrozenHead,
    rec: &StrokeRecord,
    frozen: usize,
) -> FrozenHead {
    {
        let spans = StrokeSpans::from_parts(rec, head.spans..frozen, head.dist);
        // The renderer reports where it stopped: arc length accumulates along the
        // *emitted* polyline, and only the renderer knows the budget it flattened at,
        // so a second measurement here could hand the tail a distance the head never
        // reached.
        let (state, carry) =
            render_span_range(ctx, author, &head.state, rec, spans, head.tool.as_ref());
        // A deferred range leaves the head as it was, for the next move to try again:
        // a head only ever grows, so freezing one would carry its short arc length
        // into every later tail.
        let Progress::Finished { tool, dirty: fresh } = carry.progress else {
            return head;
        };
        let mut dirty = head.dirty;
        dirty.extend(fresh);
        FrozenHead {
            spans: frozen,
            dist: carry.dist,
            // `None` from a range means "unchanged", not "reset" — a range with no
            // geometry runs nothing and leaves the brush as it found it.
            tool: tool.or(head.tool),
            state,
            gesture: head.gesture,
            epoch: head.epoch,
            contested: head.contested,
            dirty,
        }
    }
}

/// Render one span range of the in-flight stroke over an arbitrary base, resuming
/// the brush from `tool` and reporting what the next range must resume from.
///
/// Uses the same entry point a commit does (`StrokeRenderer::render_range`), so
/// the live preview and the `Action::apply` that replaces it draw the same pixels.
fn render_span_range(
    ctx: &ApplyCtx,
    author: ActorId,
    base: &DocState,
    rec: &StrokeRecord,
    spans: StrokeSpans,
    tool: Option<&crate::gpu::stroke::ToolState>,
) -> (DocState, crate::gpu::stroke::StrokeCarry) {
    let carry_only = crate::gpu::stroke::StrokeCarry::unchanged;
    // A matte has no tile map, so it previews as nothing — matching the commit, which
    // refuses the stroke outright (§15.7). Preview and commit agreeing is the §1.3
    // invariant, so the two refusals must line up.
    let Some(target) = base.layer(rec.layer).filter(|l| l.tiles().is_some()) else {
        return (base.clone(), carry_only(spans.dist()));
    };
    let tiles_base = target.tiles().expect("filtered on tiles");
    // The **author's** mask, exactly as the commit reads it, which is what lets one
    // client's live stroke be reproduced on another's screen while their selections
    // differ (§17.3). Brought into the record's frame by the record's own offset, as
    // the commit brings it (§14.12).
    let selection = ctx.transform.shifted_selection_in(
        &ctx.pool,
        &base.selection_of(author),
        rec.translation,
        stark_model::document::stroke_rect(rec),
    );
    // The substrate this stroke is being laid on (§6.4) — the same texture
    // `CommitStroke`'s apply will resolve, which is what `preview == committed` needs
    // of the tooth.
    //
    // The registry's *in-use* substrate, not a lookup by `base.substrate()`: this is a
    // `&self` path and cannot bake one on demand. `apply_document_substrate` holds
    // `current()` level with `document().substrate()`, and a live gesture cannot
    // straddle a logged `SetSubstrate` — a merge that moved the substrate bumped the
    // preview's epoch and threw every frozen head away with it.
    //
    // **The pair can still part, and only on the scale.**
    // `ViewCommand::PreviewSubstrateScale` moves `base.substrate().scale` without
    // baking, so a mark live across that drag — a hover probe, which no gesture is in
    // flight to suppress — draws on the scale the registry still holds. That costs a
    // preview lagging the slider, not a divergence: the commit re-bakes and renders
    // from the log. The **id** is what a live stroke may never be wrong about, and no
    // command previews it.
    debug_assert_eq!(
        base.substrate().id,
        ctx.substrates.id().id,
        "a live stroke is being drawn against a substrate image the document is not on",
    );
    let substrate = ctx.substrates.current();
    let (painted, carry) = ctx.stroke.render_range(
        crate::gpu::stroke::StrokeScene {
            pool: &ctx.pool,
            assets: &ctx.assets,
            base: tiles_base,
            liquify: target.liquify_run(),
            selection: &selection,
            substrate: &substrate,
        },
        rec,
        spans,
        tool,
    );
    (
        base.map_layer(rec.layer, |l| {
            l.with_painted(painted.tiles, painted.liquify)
        }),
        carry,
    )
}

/// Copy `dirty`'s tiles from `src`'s `layer` into `out` — the overlay step of the
/// preview fold (§17.6).
///
/// Only the named tiles move, which keeps two peers painting on one layer from
/// erasing each other's work back to the committed state.
fn overlay_tiles(
    out: &DocState,
    layer: LayerId,
    src: &DocState,
    dirty: &BTreeSet<TileCoord>,
) -> DocState {
    if dirty.is_empty() {
        return out.clone();
    }
    let Some(src_layer) = src.layer(layer).filter(|l| l.tiles().is_some()) else {
        return out.clone();
    };
    let src_tiles = src_layer.tiles().expect("filtered on tiles");
    let Some(tiles) = out.layer(layer).and_then(|l| l.tiles()) else {
        return out.clone();
    };
    let mut tiles = tiles.clone();
    for coord in dirty {
        match src_tiles.get(coord) {
            Some(handle) => tiles = tiles.insert(*coord, handle.clone()),
            None => tiles = tiles.remove(coord),
        }
    }
    // The run travels with the tiles it composed (§6.13): a contested liquify
    // stroke's overlay carries the run its tail left, as the uncontested swap does.
    let run = src_layer.liquify_run().cloned();
    out.map_layer(layer, |l| l.with_painted(tiles, run))
}
