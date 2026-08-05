//! The live preview fold: the committed document with every in-flight gesture drawn
//! over it (§17.6).
//!
//! Two caches make this affordable, and both are stated here rather than spread
//! across the call sites that would otherwise have to maintain them.
//!
//! [`FrozenHead`] is the per-stroke one: the fitter freezes a prefix of control
//! points it will never revise, so the spans they determine are final and can be
//! rendered once instead of on every pointer move. [`Engine::doc_epoch`] is the
//! invalidation rule that makes keeping one safe — anything that replaces the
//! document a head was composited onto bumps it, and a head from an earlier epoch is
//! discarded rather than drawn over a canvas that no longer exists.

use std::collections::{BTreeMap, BTreeSet};

use super::Engine;
use crate::document::{ActorId, DocState, LayerId, StrokeRecord};
use crate::geom::TileCoord;
use crate::gpu::StrokeSpans;
use crate::peer::{GestureView, LiveGesture, Peer};

/// Colour the live tail is drawn in under the `debug-unfrozen` feature.
/// Full-opacity magenta: it has to read against paint of any hue, and against the
/// stroke's own colour in particular.
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
    /// Not recoverable from `spans` alone, and the `drain` falloff and colour
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
    /// The base this head was composited onto ([`Engine::doc_epoch`]). Anything that
    /// replaces the base — a commit, an undo, a remote merge, a load — bumps the
    /// epoch, and a head from an earlier one is discarded rather than drawn over a
    /// canvas that no longer exists.
    epoch: u64,
    /// Every tile the head has rewritten so far, so the fold knows what to overlay
    /// (§17.6). Accumulated because a head grows across many advances.
    dirty: BTreeSet<TileCoord>,
}
impl Engine {
    /// Install (or, with `None`, drop) the stand-in document for an unlogged edit
    /// in flight — the shared tail of every `Preview*` command.
    /// The document as a `Transform` commit of `map` would leave it, built
    /// through the **same renderer** the commit uses — which is what makes
    /// the preview lossless and exact: what is shown is what "Done" will produce
    /// (§16.6). `None` when the layer cannot be transformed (a
    /// matte, absent) or the transform is rejected — the preview then simply
    /// shows the committed document, matching the commit's refusal.
    pub(super) fn preview_transform(
        &self,
        layer: LayerId,
        map: &crate::document::TransformMap,
    ) -> Option<DocState> {
        let doc = self.timeline.current();
        let base = doc.layer(layer)?.tiles()?;
        let selection = doc.selection_of(self.actor);
        let (tiles, moved) = self
            .apply
            .transform
            .apply(&self.apply.pool, base, &selection, map)?;
        Some(
            doc.map_layer(layer, |l| l.with_tiles(tiles))
                .with_selection(self.actor, moved),
        )
    }

    pub(super) fn set_doc_preview(&mut self, preview: Option<DocState>) {
        self.doc_preview = preview;
        // The gesture previews are composited onto this, so moving it invalidates
        // every cached head exactly as a commit would.
        self.doc_epoch += 1;
        self.refresh_live();
    }

    /// The document as it should be *shown*: the committed state, or the unlogged
    /// edit in flight standing in for it, with every in-flight gesture drawn over
    /// the top.
    pub(super) fn presented(&self) -> &DocState {
        self.live
            .as_ref()
            .or(self.doc_preview.as_ref())
            .unwrap_or_else(|| self.timeline.current())
    }

    /// Rebuild [`Self::live`]: the committed document (or the frame drag standing in
    /// for it) with every in-flight gesture composited over it, this client's and
    /// every peer's, in ascending [`ActorId`] order (§17.6).
    ///
    /// The order is fixed and derivable, so every client folds the same picture. Each
    /// stroke is rendered against the *committed* base and then overlaid tile-wise,
    /// rather than chained peer-over-peer: chaining would invalidate one peer's
    /// cached head on every move of the peers before it, which is precisely when two
    /// people are painting at once and the cache matters most. Where two live strokes
    /// touch the same tile in the same instant, the higher `ActorId` wins it — and a
    /// preview of concurrent strokes is provisional in any case, because the true
    /// result depends on the total order, which is not known until both commit.
    ///
    /// The cache is rebuilt into a **fresh** map rather than edited in place, which is
    /// what bounds it: a head is kept by being carried over, so one whose gesture is no
    /// longer in flight is dropped by construction rather than by a call somebody has
    /// to remember to make. Edited in place it was kept instead — the only cleanup was
    /// the wholesale `clear()` on the "nobody is gesturing" path above, so a peer that
    /// lifted while another went on painting left its head behind, and with it a whole
    /// `DocState`'s worth of `Arc<GpuTile>` handles that the pool could not reclaim.
    /// Exactly while two people are painting, which is when there is least to spare.
    pub(super) fn refresh_live(&mut self) {
        let gestures = self.live_gestures();
        if gestures.is_empty() {
            self.live = None;
            self.heads.clear();
            return;
        }
        let base = self
            .doc_preview
            .clone()
            .unwrap_or_else(|| self.timeline.current().clone());
        let mut out = base.clone();
        // Last fold's cache to draw from, and this fold's to fill: what is not moved
        // across is released when `cached` drops at the end of the call.
        let mut cached = std::mem::take(&mut self.heads);
        let mut heads = BTreeMap::new();
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
                    if let Some(selection) =
                        self.apply.selection.apply(&self.apply.pool, &prev, &op)
                    {
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
                    if let Some(tiles) = out.layer(layer).and_then(|l| l.tiles()).cloned() {
                        let gate = base.selection_of(actor);
                        if let Some(filled) =
                            self.apply.fill.apply(&self.apply.pool, &tiles, &gate, &op)
                        {
                            out = out.map_layer(layer, |l| l.with_tiles(filled));
                        }
                    }
                }
                LiveGesture::Stroke(rec) => {
                    let head = cached.remove(&actor).filter(|h| {
                        h.epoch == self.doc_epoch && h.gesture == ordinal && h.spans <= frozen
                    });
                    let (head, tail_state) =
                        self.render_live_stroke(actor, &base, &rec, frozen, head, ordinal);
                    out = overlay_tiles(&out, rec.layer, &tail_state, &head.dirty);
                    heads.insert(actor, head);
                }
            }
        }
        self.heads = heads;
        self.live = Some(out);
    }

    /// Every gesture in flight, in ascending [`ActorId`] order.
    ///
    /// The local client's is *derived* from the session's fitter rather than kept in
    /// the roster: copying it there would make two sources of truth for the one thing
    /// the `preview == committed` invariant rests on. Merging the two here is what
    /// gives the uniform ordering without the duplication (§17.4).
    fn live_gestures(&self) -> Vec<GestureView> {
        let mut out: Vec<GestureView> = Vec::new();
        out.extend(self.session.gesture_view(self.actor));
        out.extend(self.peers.iter().filter_map(Peer::gesture_view));
        out.sort_by_key(|g| g.actor);
        out
    }

    /// Advance one stroke's frozen head and render its live tail, returning the head
    /// to keep and the state the tail left behind.
    ///
    /// Uses the same entry point a commit does (`StrokeRenderer::render_range`), so
    /// the live preview and the `Action::apply` that replaces it draw the same pixels.
    fn render_live_stroke(
        &self,
        author: ActorId,
        base: &DocState,
        rec: &StrokeRecord,
        frozen: usize,
        head: Option<FrozenHead>,
        ordinal: u64,
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
            epoch: self.doc_epoch,
            dirty: BTreeSet::new(),
        });
        if frozen > head.spans {
            head = self.advance_head(author, head, rec, frozen);
        }

        let all = crate::path::span_count(rec.path.len());
        let tail = StrokeSpans {
            range: head.spans..all,
            dist: head.dist,
        };
        // The diagnostic recolours only what this move actually redrew, so the seam
        // between tinted and untinted paint *is* the freezing boundary. Build-time
        // only (the `debug-unfrozen` feature): a shipping build has no code path that
        // paints the tail in anything but the stroke's own colour.
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
            self.render_span_range(author, &head.state, tail_rec, tail, head.tool.as_ref());
        head.dirty.extend(carry.dirty);
        (head, state)
    }

    /// Dump the finished stroke's raw input as a pasteable Rust literal.
    ///
    /// A misfit seen in the app is otherwise unreproducible: the fit depends on the
    /// exact sequence of pointer reports — their spacing carries the pen's speed,
    /// which is what the density policy and the freezing both key off — and no
    /// synthetic curve stands in for a real hand. This turns one into a test case.
    pub(super) fn log_debug_samples(&mut self) {
        if !cfg!(feature = "debug-unfrozen") || self.debug_samples.is_empty() {
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

    /// Draw spans `head.spans..frozen` onto the frozen head, so the next move need
    /// not draw them again.
    fn advance_head(
        &self,
        author: ActorId,
        head: FrozenHead,
        rec: &StrokeRecord,
        frozen: usize,
    ) -> FrozenHead {
        let spans = StrokeSpans {
            range: head.spans..frozen,
            dist: head.dist,
        };
        // The renderer reports where it stopped rather than the caller recomputing it:
        // arc length is accumulated along the *emitted* polyline, and only the renderer
        // knows the budget it flattened at (a dynamics brush may have coarsened it), so
        // a second measurement here could hand the tail a distance the head never
        // reached — which `drain` and the colour-dynamics noise would both show.
        let (state, carry) =
            self.render_span_range(author, &head.state, rec, spans, head.tool.as_ref());
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
            dirty,
        }
    }

    /// Render one span range of the in-flight stroke over an arbitrary base, resuming
    /// the brush from `tool` and reporting what the next range must resume from.
    ///
    /// Uses the same entry point a commit does (`StrokeRenderer::render_range`), so
    /// the live preview and the `Action::apply` that replaces it draw the same pixels.
    fn render_span_range(
        &self,
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
        // `doc_epoch` and threw every frozen head away with it. Asserted rather than
        // only argued, since it is a `&self` path that cannot build one on demand.
        debug_assert_eq!(
            base.surface,
            self.apply.surfaces.id(),
            "a live stroke is being drawn against a surface the document is not on",
        );
        let surface = self.apply.surfaces.current();
        let (tiles, carry) = self.apply.stroke.render_range(
            crate::gpu::stroke::StrokeScene {
                pool: &self.apply.pool,
                assets: &self.apply.assets,
                base: tiles_base,
                selection: &selection,
                surface,
            },
            rec,
            spans,
            tool,
        );
        (base.map_layer(rec.layer, |l| l.with_tiles(tiles)), carry)
    }
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
