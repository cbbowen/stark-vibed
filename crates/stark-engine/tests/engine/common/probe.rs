//! What the suite reads off an engine that pixels cannot show — tiles, heights, the
//! action log — and the two protocols written over them: an export, and a drag that
//! must preview without logging.

// The siblings of this module, and the colours and corpus beside them: every
// helper here is re-exported from `common`, so one glob reaches all of them.
use super::*;
use stark_engine::command::Tool;
use stark_engine::command::{DocCommand, Edit, InputSample, ViewCommand};
use stark_engine::path::DEFAULT_TOLERANCE;
use stark_engine::testing::Probe;
use stark_engine::{Engine, RgbaImage};
use stark_model::document::BrushParams;
use stark_model::document::LayerId;
use stark_model::geom::{TILE_APRON, TILE_SIZE, TILE_TEX, Vec2};
use std::fmt;

/// Export `frame` and block on the readback — `Engine::export`'s two-phase result
/// (§15.6) collapsed to the image, which is all a test wants from it.
///
/// Both halves panic rather than answer: the outer refusal is a bad *request* and
/// the inner one is the GPU failing underneath (§5), and no caller here is asking
/// either question. `tests/engine/export.rs` asks the outer one of `export_plan`
/// directly, where it is the claim rather than the setup.
pub fn export(
    engine: &mut Engine,
    frame: Option<LayerId>,
    scale: stark_engine::ExportScale,
    background: stark_engine::Background,
    content: stark_engine::Rendered,
) -> RgbaImage {
    export_into(
        engine,
        &mut stark_engine::Offscreen::default(),
        frame,
        scale,
        background,
        content,
    )
}

/// [`export`] into an `Offscreen` the caller keeps — for the two composite tests
/// whose claim *is* that a kept slot is reused.
pub fn export_into(
    engine: &mut Engine,
    into: &mut stark_engine::Offscreen,
    frame: Option<LayerId>,
    scale: stark_engine::ExportScale,
    background: stark_engine::Background,
    content: stark_engine::Rendered,
) -> RgbaImage {
    pollster::block_on(
        engine
            .export(into, frame, scale, background, content)
            .expect("the export is a request the device can meet"),
    )
    .expect("the readback completes")
}

/// The committed document rendered again **whole**: undo and redo re-fold the last
/// action through `apply`, which has no preview to take, so a stroke committed from
/// its live preview (`PreparedStroke`, §6.2) comes back as the one-pass render a
/// replay, a file or a peer would make of it.
///
/// What every `preview == committed` claim is held against. A live commit *is* its
/// preview now, so "the cut between head and tail does not matter" is a claim
/// about this render, not about the frame after pen-up. Solo engines only: a
/// shared session logs its undo rather than navigating.
pub fn whole_render(engine: &mut Engine) -> RgbaImage {
    engine.process(DocCommand::Undo);
    engine.process(DocCommand::Redo);
    engine.render_to_image()
}

/// Per-channel levels a stroke committed from its live preview may sit from the
/// same stroke rendered whole (§6.2). The preview is a frozen head with the tail
/// drawn over it, and the head is stored in f16 at the cut where one pass carries
/// f32 through, so the two need not agree to the bit.
///
/// **They nearly do.** Measured over this bound's four consumers with it set to 0:
/// a brush shape arriving mid-stroke, a baked stamp's neighbour and a live stroke
/// under a feathered mask are all *exact*, and only `color_dynamics`'s jittered
/// preview moves — by 2 levels. So 3, which is that figure and a level. It was 4,
/// set when the flattener cut the preview's head and the replay's whole stroke at
/// different tangents; they are cut the same way now, and the corpus reads the same
/// way (`corpus::Tol::seam`).
pub const SEAM_LEVELS: u8 = 3;

/// How far apart two renders of one dial's positions may sit. The dials differ, so
/// each caller says.
#[derive(Clone, Copy)]
pub struct DragTol {
    /// The preview must move the canvas by **more** than this, or the drag showed
    /// nothing and the rest of the protocol is vacuous.
    pub shows: u8,
    /// How far a commit may sit from the preview it supersedes, and an undo from the
    /// render the drag opened on.
    pub settles: u8,
}

/// **The bargain every dial makes with the preview slot** (§1.3), asked of one dial.
///
/// A slider or a picker reports a value per pointer *move*. Logging each would bury
/// the history under a hundred one-step-apart edits and leave undo unable to take
/// the adjustment back, so a drag previews per move and logs once, on release. That
/// is one protocol, and it was written out at seven dials — the frame rect, the
/// canvas color, the frame color, layer opacity, the bend, a filter's sliders.
///
/// `edit` builds the edit for a value; `values` is the drag, its last entry the value
/// released on. In order:
///
/// - the preview reaches the canvas, so the rest is not vacuous, and logs nothing;
/// - a history step mid-drag drops the preview and finds nothing of the drag logged;
/// - the release costs **exactly one** action and renders what the last preview did;
/// - one undo takes the whole drag back.
///
/// "Logged nothing" is read off the timeline's own length, not off pixels: a drag
/// step that did log would usually render identically to one that did not, and the
/// claim is about the history.
///
/// `during` is called with the drag up and `mid_undo` right after the mid-drag undo,
/// for each file's own reading of what `observe()` reports there — which is the one
/// thing these seven dials do not share, since each answers about its own state.
/// [`settled_drag_logs_nothing`] is the fifth claim, for the dials that make it.
pub fn drag_protocol<V>(
    engine: &mut Engine,
    edit: impl Fn(&V) -> Edit,
    values: &[V],
    tol: DragTol,
    during: impl Fn(&mut Engine),
    mid_undo: impl Fn(&mut Engine),
) {
    let opening = engine.render_to_image();
    let steps = logged(engine);
    let last = values.last().expect("a drag moves somewhere");

    let drag = |engine: &mut Engine| {
        for v in values {
            engine.process(ViewCommand::Preview(Some(edit(v))));
        }
    };

    drag(engine);
    let dragging = engine.render_to_image();
    assert_images_differ(
        &opening,
        &dragging,
        tol.shows,
        "the drag previewed nothing, so the rest of this claim is vacuous",
    );
    assert_eq!(logged(engine), steps, "a preview logged an action");
    during(engine);

    engine.process(DocCommand::Undo);
    mid_undo(engine);
    engine.process(DocCommand::Redo);
    assert_eq!(
        logged(engine),
        steps,
        "a history step mid-drag logged a step of the drag"
    );
    assert_images(
        &opening,
        &engine.render_to_image(),
        tol.settles,
        "a history step mid-drag should drop the preview and log nothing of it",
    );

    // The drag again, since the history step above dropped it, then the release.
    drag(engine);
    engine.process(DocCommand::Set(edit(last)));
    assert_eq!(
        logged(engine),
        steps + 1,
        "the whole drag should cost exactly one action"
    );
    assert_images(
        &dragging,
        &engine.render_to_image(),
        tol.settles,
        "the commit rendered something other than what the drag previewed",
    );

    engine.process(DocCommand::Undo);
    assert_images(
        &opening,
        &engine.render_to_image(),
        tol.settles,
        "one undo should take back the whole drag",
    );
}

/// A drag released on the value it was pressed on is **not an edit** — but it still
/// has to commit, because every release commits and the preview it left standing must
/// be superseded by something. The refusal that reconciles those two is the engine's,
/// and this is where it is asked of one dial.
///
/// `rest` is the value the drag opened on, and `values` its travel, the last entry
/// being `rest` again.
pub fn settled_drag_logs_nothing<V>(
    engine: &mut Engine,
    edit: impl Fn(&V) -> Edit,
    values: &[V],
    rest: &V,
    tol: u8,
) {
    let opening = engine.render_to_image();
    let steps = logged(engine);
    for v in values {
        engine.process(ViewCommand::Preview(Some(edit(v))));
    }
    engine.process(DocCommand::Set(edit(rest)));
    assert_eq!(
        logged(engine),
        steps,
        "a drag that ended where it started logged a step"
    );
    assert_images(
        &opening,
        &engine.render_to_image(),
        tol,
        "the settled drag should leave the document as it found it, preview and all",
    );
}

/// How many actions this engine's timeline holds — what "the drag logged nothing"
/// is read off.
fn logged(engine: &Engine) -> usize {
    engine.scrub_range().expect("a solo timeline scrubs").1
}

/// A reading of the committed document, to hold a later one against.
///
/// The `DocState` whole — cheap to clone, since it is persistent maps of tile
/// handles (§5.1) — plus the two readings that are the *timeline's* rather than the
/// document's.
pub struct Committed {
    doc: stark_engine::testing::DocState,
    revision: u64,
    log: Option<(usize, usize)>,
}

impl Committed {
    pub fn of(engine: &Engine) -> Self {
        Self {
            doc: engine.document().clone(),
            revision: engine.observe().doc_revision,
            log: engine.scrub_range(),
        }
    }
}

/// Assert the committed document has not moved since `before`.
///
/// **What may differ is not listed here.** `testing::undeclared` is the audit's own
/// enumeration of it (§12.6) — the one `Materialize::audit` runs on every fold of
/// every debug build — asked with an empty `Footprint`, so *any* difference is
/// undeclared and reported by name. A `Layer` or `DocState` field added later is
/// covered by construction, where a hand-listed set of readings goes quietly stale
/// and says nothing when it does.
///
/// The revision and the scrub range are read beside it because they are the
/// timeline's: a preview that wrote the timeline's current state without committing
/// would leave the document identical and the log moved.
#[track_caller]
pub fn assert_document_unchanged(engine: &Engine, before: &Committed, what: impl fmt::Display) {
    let now = Committed::of(engine);
    let loose = stark_engine::testing::undeclared(
        &before.doc,
        &now.doc,
        &stark_model::document::Footprint::default(),
    );
    assert!(
        loose.is_empty(),
        "{what}: the committed document changed: {}",
        loose.join(", ")
    );
    assert_eq!(
        (before.revision, before.log),
        (now.revision, now.log),
        "{what}: the document is intact, but the timeline moved"
    );
}

/// The coordinates of the tiles `layer` holds in the **committed** document.
///
/// Empty where it holds none, which *is* the answer for bare canvas: a tile that was
/// never minted is not a tile of zeroes. The order is the map's; a caller comparing
/// two of these sorts them.
pub fn layer_tiles(engine: &Engine, layer: LayerId) -> Vec<stark_model::geom::TileCoord> {
    engine
        .document()
        .layer(layer)
        .and_then(|l| l.tiles())
        .map(|t| t.keys().copied().collect())
        .unwrap_or_default()
}

/// Every tile of `layer` on `a` and `b`, compared **bitwise** lane by lane: `None`
/// when they agree, else how many lanes differ, the largest difference and where. A
/// residual one side holds and the other does not is a difference too.
pub fn tiles_differ(a: &Engine, b: &Engine, layer: LayerId) -> Option<String> {
    let keys = |e: &Engine| -> Vec<stark_model::geom::TileCoord> {
        let mut k = layer_tiles(e, layer);
        k.sort();
        k
    };
    let (ka, kb) = (keys(a), keys(b));
    if ka != kb {
        return Some(format!(
            "tile sets differ: {} against {}",
            ka.len(),
            kb.len()
        ));
    }
    // (lanes differing, worst |Δ|, where: tile, channel, lane index)
    let mut diff = (0usize, 0.0f32, None);
    for c in &ka {
        let (ta, tb) = (
            a.tile_channels(layer, *c).expect("listed"),
            b.tile_channels(layer, *c).expect("listed"),
        );
        if ta.resid.is_some() != tb.resid.is_some() {
            return Some(format!("one side of {c:?} has a residual, the other none"));
        }
        let channels = [
            ("color", ta.color, tb.color),
            ("height", ta.height, tb.height),
            (
                "resid",
                ta.resid.unwrap_or_default(),
                tb.resid.unwrap_or_default(),
            ),
        ];
        for (name, la, lb) in channels {
            for (i, (x, y)) in la.iter().zip(&lb).enumerate() {
                if x.to_bits() != y.to_bits() {
                    diff.0 += 1;
                    let d = (x - y).abs();
                    if diff.2.is_none() || d > diff.1 {
                        diff = (diff.0, d, Some((*c, name, i)));
                    }
                }
            }
        }
    }
    let (n, worst, at) = diff;
    at.map(|(c, name, i)| {
        format!(
            "{n} lanes differ over {} tiles, worst {worst:e} in {name}[{i}] of {c:?}",
            ka.len()
        )
    })
}

/// One straight run along `y = 0` from `−half` to `+half`, delivered coarsely or
/// finely, folded whole, measured, and taken back off the canvas.
///
/// `step` of `None` is the two reports a hand makes of a straight run, fitted at
/// [`DEFAULT_TOLERANCE`]: the brush's own budget cuts it, about a radius a segment.
/// `Some(s)` reports every `s` canvas px and fits at `MIN_TOLERANCE`, which cuts a
/// segment about every px. **The last report lands on `+half` either way** — a
/// truncated final step would shorten the fine run's end cap, a deficit in the same
/// direction as what the callers here measure and of the same order.
///
/// The stroke is folded whole before it is read, so the live preview's own head/tail
/// cut is not what is measured.
///
/// **An engine of its own, and that is not thrift forgone.** The deposit jitter is
/// keyed on the stroke's seed, which is the document clock at the press (§6.2) — so
/// two deliveries on one engine jitter differently and stop cancelling, while two on
/// engines of their own cancel exactly. Measured, comparing a coarse delivery with a
/// fine one at a 40 px tip: the mid-band column moves 0.27% on a shared engine
/// against a bound of 0.10%, and 0.00% on engines of their own. A caller wanting one
/// engine for a table of these would have to turn the jitter off, which is a
/// different measurement.
///
/// Answers the layer's total height and the height down each column of the middle
/// half of the run (§6.1), or `None` where there is no adapter and skipping is
/// permitted.
pub fn cut_run(b: BrushParams, half: f32, step: Option<f32>) -> Option<(f64, Vec<f64>)> {
    let mut engine = engine_or_skip()?;
    let (samples, tolerance): (Vec<Vec2>, f32) = match step {
        Some(s) => {
            let n = (2.0 * half / s).ceil() as usize;
            let at = |i: usize| Vec2::new((-half + i as f32 * s).min(half), 0.0);
            ((0..=n).map(at).collect(), stark_engine::path::MIN_TOLERANCE)
        }
        None => (
            vec![Vec2::new(-half, 0.0), Vec2::new(half, 0.0)],
            DEFAULT_TOLERANCE,
        ),
    };
    engine.process(ViewCommand::set_brush(b));
    Gesture::new(Tool::Brush)
        .tolerance(tolerance)
        .drive(&mut engine, samples.into_iter().map(InputSample::at))
        .end();
    engine.process(DocCommand::Undo);
    engine.process(DocCommand::Redo);
    let band = (half / 2.0) as i32;
    Some((
        total_height(&engine, LayerId::ROOT),
        column_height(&engine, LayerId::ROOT, -band..band),
    ))
}

/// The total **height** on `layer` — the channel §6.1 says is conserved, summed off
/// the tiles themselves rather than inferred from how dark the render came out.
///
/// **Interiors only.** Every tile's apron is a bit-identical copy of its neighbour's
/// edge (§6.4), so summing whole blocks counts the seams twice — which is invisible on
/// a one-tile stroke and grows with the tile count, exactly the way a conservation
/// drift would.
///
/// The committed document, so a caller mid-gesture is asking about the state before
/// the live tail.
pub fn total_height(engine: &Engine, layer: LayerId) -> f64 {
    let lo = TILE_APRON as usize;
    let hi = lo + TILE_SIZE as usize;
    layer_tiles(engine, layer)
        .iter()
        .filter_map(|c| engine.tile_channels(layer, *c))
        .map(|ch| {
            (lo..hi)
                .flat_map(|y| (lo..hi).map(move |x| y * TILE_TEX as usize + x))
                .map(|i| ch.height[i] as f64)
                .sum::<f64>()
        })
        .sum()
}

/// Height summed down each canvas column in `xs` on `layer`, off the tiles' interiors
/// — [`total_height`] resolved along x, for a claim about how a deposit varies along a
/// stroke.
pub fn column_height(engine: &Engine, layer: LayerId, xs: std::ops::Range<i32>) -> Vec<f64> {
    let mut cols = vec![0.0f64; xs.len()];
    for coord in layer_tiles(engine, layer) {
        let ch = engine
            .tile_channels(layer, coord)
            .expect("a listed tile reads back");
        let x0 = coord.origin().x as i32;
        for lx in 0..TILE_SIZE {
            let x = x0 + lx as i32;
            if !xs.contains(&x) {
                continue;
            }
            let col = &mut cols[(x - xs.start) as usize];
            for ly in 0..TILE_SIZE {
                let i = (ly + TILE_APRON) * TILE_TEX + lx + TILE_APRON;
                *col += f64::from(ch.height[i as usize]);
            }
        }
    }
    cols
}

/// The **height** and per-unit **opacity** at one canvas point on `layer` — §6.1's two
/// channels, read off the tile rather than inferred from the pixel they produced.
///
/// `None` where the layer holds no tile there, which *is* the answer for bare canvas:
/// a tile that was never minted is not a tile of zeroes.
///
/// The interior, never the apron: a point maps to exactly one tile's interior texel,
/// and the copy of it in a neighbour's apron is the same value by §6.4 — so taking the
/// interior is a choice about which of two identical readings to name, and naming the
/// owner is the one that stays right if the seam rule ever breaks.
pub fn paint_at(engine: &Engine, layer: LayerId, canvas: Vec2) -> Option<(f32, f32)> {
    let tile = |v: f32| (v / TILE_SIZE as f32).floor() as i32;
    let coord = stark_model::geom::TileCoord::new(tile(canvas.x), tile(canvas.y));
    let local = canvas - coord.origin();
    let (x, y) = (
        local.x as usize + TILE_APRON as usize,
        local.y as usize + TILE_APRON as usize,
    );
    let ch = engine.tile_channels(layer, coord)?;
    let i = y * TILE_TEX as usize + x;
    Some((ch.height[i], ch.color[i * 4 + 3]))
}
