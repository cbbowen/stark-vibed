//! The live preview fold: the committed document with every in-flight gesture drawn
//! over it (§17.6).
//!
//! Two caches make this affordable — the folded document itself, and a
//! [`FrozenHead`] per stroke in flight — and one rule keeps them honest. The fitter
//! freezes a prefix of control points it will never revise, so the spans they
//! determine are final: render them once, keep the result, and each pointer move
//! only draws the short live tail over it.
//!
//! The rule is an **epoch**. Anything that replaces the document a head was
//! composited onto bumps [`Preview::epoch`], and a head stamped with an older one is
//! discarded rather than drawn over a canvas that no longer exists — which rules out
//! the whole class of "drawn over a base that has since moved" instead of
//! enumerating the ways it arises.
//!
//! [`Preview`] exists so that stays true. The four pieces of state it holds are one
//! thing with one invariant, so they are one type rather than four fields of
//! [`Engine`] that any method could set. **The slot cannot move without the epoch
//! moving with it**, because [`Preview::set_doc`] is the only way to move it — where
//! five bumps spelled out at their call sites would be five chances to forget one.

use std::collections::{BTreeMap, BTreeSet};

use super::Engine;
use crate::document::{ApplyCtx, DocState};
use crate::gpu::StrokeSpans;
use crate::peer::{GestureView, LiveGesture, Peer};
use stark_model::document::{ActorId, LayerId, StrokeRecord};
use stark_model::geom::{TileCoord, TileRect};

/// What is being *shown* over the committed document, and the caches that make
/// showing it affordable (§17.6).
///
/// The fields are private to this module and there is no setter for any of them
/// individually: the two that can invalidate a cached head — the drag-preview
/// document and the committed one underneath it — move only through
/// [`set_doc`](Self::set_doc) and [`invalidate`](Self::invalidate), each of which
/// bumps the epoch as it goes. That is the whole reason the type exists.
#[derive(Default)]
pub(super) struct Preview {
    /// The unlogged document edit in flight: a whole document that stands in for
    /// the committed one, because what these edits change — a matte's rect
    /// (§15.7), the substrate color (§15.5) — is document state rather than a
    /// tile edit, so there is nothing to draw *over* the document the way a stroke
    /// preview does. One slot, not one per kind: only one such drag can be in
    /// flight at a time (they all belong to a single held pointer), and the
    /// stand-in is built from the committed state each time, so a second kind
    /// starting mid-drag supersedes the first rather than compounding with it.
    /// `None` when nothing is being dragged.
    doc: Option<DocState>,
    /// The committed state (or `doc`) with every in-flight gesture — this client's
    /// and every peer's — drawn over it. `None` when nobody is mid-gesture.
    live: Option<DocState>,
    /// The settled head of each in-flight stroke, keyed by its author (see
    /// [`FrozenHead`]). Every head is rooted at the *committed* document rather than
    /// at the previous peer's preview: chaining would be marginally more faithful
    /// for two strokes overlapping in the same instant, and would invalidate peer
    /// *k*'s cache on every move by peers before it — collapsing the incremental
    /// repaint exactly when two people are painting at once.
    heads: BTreeMap<ActorId, FrozenHead>,
    /// Bumped whenever the document the previews are composited onto changes. A
    /// [`FrozenHead`] stamped with an older epoch is stale and discarded.
    epoch: u64,
    /// Bumped every time the fold is **rebuilt** — a counter for "what is shown has
    /// moved", where [`epoch`](Self::epoch) is "what is shown was *replaced*".
    ///
    /// The two cannot be one. `epoch` throws away every cached [`FrozenHead`], so
    /// bumping it per fold would discard the settled head of every live stroke on
    /// every pointer move and undo the whole incremental repaint. But a fold does
    /// change what a renderer should draw, and nothing else moves when it does: a
    /// stroke in flight commits nothing, so `doc_revision` is still, and it replaces
    /// no document, so `epoch` is still. A draw list keyed on those two alone would
    /// hold the frame at the moment the stroke began (C4).
    fold: u64,
    /// Whether the fold no longer reflects the gestures and document it folds —
    /// set by [`Engine::mark_live_stale`], cleared by the [`Engine::flush_live`]
    /// that services it. The deferral is what turns N mutations per frame (every
    /// pointer sample, every peer gesture frame) into one fold per frame actually
    /// painted.
    stale: bool,
}

impl Preview {
    /// Note that the document the previews are drawn over has been replaced, so
    /// every cached head is stale. The committed document moving is one cause; the
    /// drag preview standing in for it is the other, and [`set_doc`](Self::set_doc)
    /// routes through here rather than restating it.
    pub(super) fn invalidate(&mut self) {
        self.epoch += 1;
    }

    /// Install (or, with `None`, drop) the stand-in document for an unlogged edit in
    /// flight. **The only way to move that slot**, so the epoch cannot be left behind
    /// — the gesture previews are composited onto this, and moving it invalidates
    /// every cached head exactly as a commit would.
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

    /// The invalidation epoch a cached head is stamped against.
    pub(super) fn epoch(&self) -> u64 {
        self.epoch
    }

    /// How many times the fold has been rebuilt — see [`fold`](Self::fold).
    pub(super) fn fold(&self) -> u64 {
        self.fold
    }

    /// Rebuild the fold: `committed` (or the drag standing in for it) with every
    /// in-flight gesture composited over it, in ascending [`ActorId`] order (§17.6).
    ///
    /// The order is fixed and derivable, so every client folds the same picture. A
    /// stroke is rendered against the *committed* base and then overlaid tile-wise
    /// rather than chained peer-over-peer, because chaining would invalidate one
    /// peer's cached head on every move of the peers before it — precisely when two
    /// people are painting at once and the cache matters most.
    ///
    /// **Unless they are painting on the same tiles.** The overlay copies whole tiles,
    /// so a tile two live strokes share carries the committed pixels plus one
    /// stroke's paint whichever way it is copied: the second copy puts back exactly
    /// what the first had drawn there, and one of the two strokes disappears from that
    /// tile until it commits. So a stroke whose reach meets one already in the fold is
    /// rendered over **the fold** instead, and gives up its cached head to do it —
    /// which is the honest price, and one paid only while two people are painting the
    /// same tiles rather than merely at the same time. `preview == committed` still
    /// holds tile-wise for each stroke; what stays provisional is which of two
    /// concurrent strokes ends up on top, since that depends on a total order neither
    /// is in until both commit.
    ///
    /// The head cache is rebuilt into a **fresh** map rather than edited in place,
    /// which is what bounds it: a head is kept by being carried over, so one whose
    /// gesture is no longer in flight is dropped by construction rather than by a call
    /// somebody has to remember to make. Edited in place, the only cleanup would be
    /// the wholesale clear on the "nobody is gesturing" path — so a peer that lifts
    /// while another goes on painting leaves its head behind, and with it a whole
    /// `DocState`'s worth of `Arc<GpuTile>` handles the pool cannot reclaim. Exactly
    /// while two people are painting, which is when there is least to spare.
    fn rebuild(&mut self, ctx: &ApplyCtx, committed: &DocState, gestures: Vec<GestureView>) {
        // Before the early return as much as after it: dropping the fold is a change
        // to what is shown exactly as building one is, and it is the transition a
        // pen-up makes.
        self.fold += 1;
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
                // A fill previews as the paint it will lay, not as an outline of
                // where it would go — the same `FillRenderer::apply` the commit
                // makes, over the same base, so what is on screen mid-drag is
                // literally the result. Losslessly, and thrown away and redone on
                // each move rather than accumulated, which is what keeps dragging a
                // rectangle out from stacking a hundred glazes.
                LiveGesture::Fill { layer, op } => {
                    // Already chained: it reads `out`, not `base`, and replaces the
                    // layer's whole tile map rather than copying tiles across.
                    claimed.push((layer, stark_model::document::fill_rect(&op)));
                    if let Some(tiles) = out.layer(layer).and_then(|l| l.tiles()).cloned() {
                        let gate = base.selection_of(actor);
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
                    // A contested stroke draws over the fold, which is a different
                    // document on every move — so its head is unrepeatable by
                    // construction, and both halves of the cache follow from that one
                    // fact: nothing cached is reused, and what this fold caches is
                    // marked never to be.
                    let head = cached.remove(&actor).filter(|h| {
                        !contested
                            && !h.contested
                            && h.epoch == self.epoch
                            && h.gesture == ordinal
                            && h.spans <= frozen
                    });
                    let root = if contested { &out } else { &base };
                    let (head, tail_state) = self.render_live_stroke(
                        ctx, actor, root, &rec, frozen, head, ordinal, contested,
                    );
                    out = overlay_tiles(&out, rec.layer, &tail_state, &head.dirty);
                    heads.insert(actor, head);
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
    #[allow(clippy::too_many_arguments)]
    fn render_live_stroke(
        &self,
        ctx: &ApplyCtx,
        author: ActorId,
        base: &DocState,
        rec: &StrokeRecord,
        frozen: usize,
        head: Option<FrozenHead>,
        ordinal: u64,
        contested: bool,
    ) -> (FrozenHead, DocState) {
        // A stroke cannot freeze a span whose pixels are still measured against a
        // length the stroke has not reached — the taper, from the *ends* of the whole
        // stroke, and the touch-down dab, which the stroke has not yet outrun. Both
        // are ends still under the pointer. Held back here rather than in the fitter
        // because it is a fact about the brush, not about the curve — the same control
        // points freeze at the same place for every brush of the same size (see
        // `safe_frozen`). An already-kept head is unaffected: a prefix this admitted
        // once stays admissible, so the clamp can only slow the head down, never
        // invalidate it.
        let frozen = crate::gpu::stroke::safe_frozen(rec, frozen);
        // Nothing cached, or the fit went backwards (a new stroke): start over from
        // the committed document, with a fresh (uncharged) brush.
        let mut head = head.unwrap_or_else(|| FrozenHead {
            spans: 0,
            dist: 0.0,
            tool: None,
            state: base.clone(),
            gesture: ordinal,
            epoch: self.epoch,
            contested,
            dirty: BTreeSet::new(),
        });
        if frozen > head.spans {
            head = advance_head(ctx, author, head, rec, frozen);
        }

        let all = crate::path::span_count(rec.path.len());
        let tail = StrokeSpans {
            range: head.spans..all,
            dist: head.dist,
        };
        // The diagnostic recolors only what this move actually redrew, so the seam
        // between tinted and untinted paint *is* the freezing boundary. Build-time
        // only (the `debug-unfrozen` feature): a shipping build has no code path that
        // paints the tail in anything but the stroke's own color.
        #[cfg(feature = "debug-unfrozen")]
        let tinted = {
            let mut r = rec.clone();
            r.brush.color = DEBUG_UNFROZEN_COLOR;
            r
        };
        #[cfg(feature = "debug-unfrozen")]
        let tail_rec = &tinted;
        #[cfg(not(feature = "debug-unfrozen"))]
        let tail_rec = rec;
        // The tail reaches the end of the stroke, so the state it leaves the brush in
        // is handed to nobody — it is thrown away and rebuilt from the head on the next
        // move, which is exactly what makes the tail re-renderable. Its dirty tiles,
        // though, are part of what the fold has to overlay, so they join the head's.
        let (state, carry) =
            render_span_range(ctx, author, &head.state, tail_rec, tail, head.tool.as_ref());
        head.dirty.extend(carry.dirty);
        (head, state)
    }
}

/// Color the live tail is drawn in under the `debug-unfrozen` feature.
/// Full-opacity magenta: it has to read against paint of any hue, and against the
/// stroke's own color in particular.
#[cfg(feature = "debug-unfrozen")]
const DEBUG_UNFROZEN_COLOR: [f32; 4] = [1.0, 0.0, 1.0, 1.0];

/// The part of the in-flight stroke that has stopped changing, already composited
/// onto the committed document.
///
/// A live stroke is re-rendered on every pointer move, and rendering it costs
/// (segments × tiles it covers) — both of which grow with its length, so a long
/// stroke gets quadratically expensive to keep drawing. But the fitter freezes
/// control points behind the pointer and never revises them
/// ([`PathFitter::frozen_spans`](crate::path::PathFitter::frozen_spans)), so the
/// spans they determine are final: render them once, keep the result, and each move
/// only has to draw the short live tail over it. Work per move then follows the tail
/// rather than the stroke.
pub(super) struct FrozenHead {
    /// How many leading spans `state` already has drawn on it.
    spans: usize,
    /// Arc length at the end of those spans — where the tail's `dist` resumes.
    /// Not recoverable from `spans` alone, and the `drain` falloff and color
    /// dynamics both read it (see `gpu::stroke::StrokeSpans`).
    dist: f32,
    /// The brush state the tail must resume from, for a stroke that runs the
    /// sequential stamp loop (`lift`/`deposit`/`charge`). `None` for the swept path,
    /// which carries nothing between segments. See
    /// [`ToolState`](crate::gpu::stroke::ToolState).
    tool: Option<crate::gpu::stroke::ToolState>,
    state: DocState,
    /// Which gesture this is the head of — its author's ordinal. A head is only ever
    /// legitimately reused across consecutive moves of the *same* gesture, and the
    /// span count alone cannot tell a new stroke from a continued one when the new
    /// one has already grown past where the old was frozen.
    gesture: u64,
    /// The base this head was composited onto ([`Preview::epoch`]). Anything that
    /// replaces the base — a commit, an undo, a remote merge, a load, a drag preview
    /// — bumps the epoch, and a head from an earlier one is discarded rather than
    /// drawn over a canvas that no longer exists.
    epoch: u64,
    /// Whether this head was rooted at the *fold* rather than at the committed
    /// document, because another live stroke had already claimed tiles it reaches.
    /// The fold is rebuilt from scratch on every move, so such a head can never be
    /// reused — it is still kept, because it is what the overlay reads and what
    /// [`Preview::head_count`] has to account for, but it is stale the moment it
    /// is stored. The epoch cannot say this: nothing about the *document* changed.
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

    /// The document as a `Transform` commit of `map` would leave it, built
    /// through the **same renderer** the commit uses — which is what makes
    /// the preview lossless and exact: what is shown is what "Done" will produce
    /// (§16.6). `None` when the layer cannot be transformed (a
    /// matte, absent) or the transform is rejected — the preview then simply
    /// shows the committed document, matching the commit's refusal.
    pub(super) fn preview_transform(
        &self,
        layer: LayerId,
        map: &stark_model::document::TransformMap,
    ) -> Option<DocState> {
        let doc = self.timeline.current();
        let base = doc.layer(layer)?.tiles()?;
        let selection = doc.selection_of(self.actor());
        let (tiles, moved) =
            self.shared
                .apply
                .transform
                .apply(&self.shared.apply.pool, base, &selection, map)?;
        Some(
            doc.map_layer(layer, |l| l.with_tiles(tiles))
                .with_selection(self.actor(), moved),
        )
    }

    /// The document as a `Fill` commit of `op` would leave it — the gradient
    /// fill's composing preview (§22.4), built through the **same**
    /// `FillRenderer::apply` the commit uses, for `preview_transform`'s reason:
    /// what is shown is what "Done" will produce. `None` when the layer cannot
    /// take paint or the fill is refused (unbounded, over the cap) — the
    /// preview then shows the committed document, matching the commit.
    pub(super) fn preview_fill(
        &self,
        layer: LayerId,
        op: &stark_model::document::FillOp,
    ) -> Option<DocState> {
        let doc = self.timeline.current();
        let base = doc.layer(layer)?.tiles()?.clone();
        let selection = doc.selection_of(self.actor());
        let tiles = self
            .shared
            .apply
            .fill
            .apply(&self.shared.apply.pool, &base, &selection, op)?;
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
    /// A note, not a rebuild: mutations arrive at input rate (every pointer
    /// sample; every peer's gesture frame off the pump), while presentation is one
    /// frame per rAF — so the fold is rebuilt by [`flush_live`](Self::flush_live)
    /// at the read, once per frame actually shown, rather than here, once per
    /// event whose intermediate picture nobody sees. Painting the latest state
    /// once shows exactly what folding it per event would have.
    pub(super) fn mark_live_stale(&mut self) {
        self.preview.stale = true;
    }

    /// Service a pending [`mark_live_stale`](Self::mark_live_stale): rebuild the
    /// fold if anything has changed since it was last built. Every reader of
    /// [`presented`](Self::presented) reaches it through an entry point that
    /// flushes first — the render, the eyedropper, the head-count diagnostic — so
    /// a stale fold is never observable, only never-shown intermediate ones.
    ///
    /// `pub` for harnesses, not frontends: a frontend gets its flush from
    /// [`render`](Self::render) with the frame that shows the result, while the
    /// stroke benchmark has no surface and must pay the fold explicitly to time
    /// it. Not a command, deliberately (§4): a command is an *input* — logged,
    /// replicated, countable — and this mutates no document or session state at
    /// all, only services a cache whose content is fully determined by them.
    /// Idempotent; calling it twice does the work once.
    pub fn flush_live(&mut self) {
        if !self.preview.stale {
            return;
        }
        // **After the early return**, so the row counts folds rather than calls: this
        // is asked once per painted frame whether or not anything moved, and a row
        // whose count was "frames" and whose mean was diluted by the no-op case would
        // answer neither question. Below the flag rather than above it for the same
        // reason a `#[inline]` guard goes first — what is being measured is the work.
        crate::timing::span!("live.fold");
        self.preview.stale = false;
        let gestures = self.live_gestures();
        // Disjoint fields, so the fold reads the context and the committed document
        // while holding the preview mutably.
        self.preview
            .rebuild(&self.shared.apply, self.timeline.current(), gestures);
    }

    /// Every gesture in flight, in ascending [`ActorId`] order.
    ///
    /// The local client's is *derived* from the session's fitter rather than kept in
    /// the roster: copying it there would make two sources of truth for the one thing
    /// the `preview == committed` invariant rests on. Merging the two here is what
    /// gives the uniform ordering without the duplication (§17.4).
    ///
    /// A hand that is only *hovering* folds the §18.1.10 mark instead — the
    /// stroke its recent reports would have committed, in the same
    /// [`GestureView`] shape, so the fold cannot tell it from a real gesture and
    /// `hover == committed` is inherited rather than maintained. The `match` is
    /// the guarantee the two never coexist: an actor folds at most one gesture,
    /// and a fact always outranks a hypothesis. The seed is the one a gesture
    /// starting this instant would take (`GestureCommand::Start`), which is
    /// exactly the claim the mark makes.
    fn live_gestures(&self) -> Vec<GestureView> {
        let mut out: Vec<GestureView> = Vec::new();
        match self.session.gesture_view(self.actor()) {
            Some(g) => out.push(g),
            None => out.extend(self.session.hover_view(self.actor(), self.authoring.clock)),
        }
        out.extend(self.peers.iter().filter_map(Peer::gesture_view));
        out.sort_by_key(|g| g.actor);
        out
    }

    /// Dump the finished stroke's raw input as a pasteable Rust literal.
    ///
    /// A misfit seen in the app is otherwise unreproducible: the fit depends on the
    /// exact sequence of pointer reports — their spacing carries the pen's speed,
    /// which is what the density policy and the freezing both key off — and no
    /// synthetic curve stands in for a real hand. This turns one into a test case.
    /// Compiled away without the feature, along with the buffer it prints: a
    /// diagnostic that cannot reach a shipping build should not be *carried* by one
    /// either (see [`Engine::note_debug_sample`]).
    #[cfg(feature = "debug-unfrozen")]
    pub(super) fn log_debug_samples(&mut self) {
        if self.debug_samples.is_empty() {
            return;
        }
        // Positions *and* the pen channels. Position alone is not the input: pressure
        // sizes the brush and tilt steers it, both are fitted as their own least-squares
        // channels, and a capture without them cannot reproduce a fault in either.
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
        let spans = StrokeSpans {
            range: head.spans..frozen,
            dist: head.dist,
        };
        // The renderer reports where it stopped rather than the caller recomputing it:
        // arc length is accumulated along the *emitted* polyline, and only the renderer
        // knows the budget it flattened at (a dynamics brush may have coarsened it), so
        // a second measurement here could hand the tail a distance the head never
        // reached — which `drain` and the color-dynamics noise would both show.
        let (state, carry) =
            render_span_range(ctx, author, &head.state, rec, spans, head.tool.as_ref());
        let mut dirty = head.dirty;
        dirty.extend(carry.dirty);
        FrozenHead {
            spans: frozen,
            dist: carry.dist,
            // `None` from a range means "unchanged", not "reset" — a range with no
            // geometry runs nothing and leaves the brush as it found it.
            tool: carry.tool.or(head.tool),
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
    let carry_only = |dist| crate::gpu::stroke::StrokeCarry {
        dist,
        tool: None,
        dirty: Vec::new(),
    };
    // A matte has no tile map, so it previews as nothing — matching the commit,
    // which refuses the stroke outright (§15.7). Preview and
    // commit agreeing is the §1.3 invariant, so the two refusals must line up.
    let Some(tiles_base) = base.layer(rec.layer).and_then(|l| l.tiles()) else {
        return (base.clone(), carry_only(spans.dist));
    };
    // The **author's** mask, exactly as the commit will read it — which is what
    // lets one client's live stroke be reproduced faithfully on another's screen
    // while their selections differ (§17.3).
    let selection = base.selection_of(author);
    // The ground this stroke is being laid on (§6.4) — the same texture
    // `CommitStroke`'s apply will resolve, which is what `preview == committed`
    // needs of the tooth.
    //
    // The registry's *in-use* surface, not a lookup by `base.surface`, and the two
    // are the same thing here rather than nearly so: `apply_document_surface`
    // holds `current()` equal to `document().surface` after every commit, undo,
    // load and merge, and a live gesture cannot straddle a `SetSurface` because a
    // gesture is not a logged action — any merge that switched the surface bumped
    // the preview's epoch and threw every frozen head away with it. Asserted rather
    // than only argued, since it is a `&self` path that cannot build one on demand.
    debug_assert_eq!(
        base.surface,
        ctx.surfaces.id(),
        "a live stroke is being drawn against a surface the document is not on",
    );
    let surface = ctx.surfaces.current();
    let (tiles, carry) = ctx.stroke.render_range(
        crate::gpu::stroke::StrokeScene {
            pool: &ctx.pool,
            assets: &ctx.assets,
            base: tiles_base,
            selection: &selection,
            surface: &surface,
        },
        rec,
        spans,
        tool,
    );
    (base.map_layer(rec.layer, |l| l.with_tiles(tiles)), carry)
}

/// Copy `dirty`'s tiles from `src`'s `layer` into `out` — the overlay step of the
/// preview fold (§17.6).
///
/// Only the named tiles move, which is what keeps two peers painting on one layer
/// from erasing each other's work back to the committed state: each contributes
/// exactly the tiles its own stroke touched.
fn overlay_tiles(
    out: &DocState,
    layer: LayerId,
    src: &DocState,
    dirty: &BTreeSet<TileCoord>,
) -> DocState {
    if dirty.is_empty() {
        return out.clone();
    }
    let Some(src_tiles) = src.layer(layer).and_then(|l| l.tiles()) else {
        return out.clone();
    };
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
    out.map_layer(layer, |l| l.with_tiles(tiles))
}
