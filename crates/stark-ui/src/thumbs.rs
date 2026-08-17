//! Rendered thumbnails for brushes: a miniature test stroke per brush, generated
//! offscreen and cached (§11). Two viewers share the one cache — the preset
//! library's rows, and the quick-brush rack the number keys draw (§18.1.8) —
//! and they share it exactly because the key is the brush itself, so a brush
//! that is in both places is rendered once.
//!
//! Each thumbnail is a red stroke drawn across a canvas laid **entirely in
//! paint** — a light-gray slab on the left half, a dark-gray slab on the right —
//! rendered by a **shared** engine (`Renderer::shared_engine`), so it costs no
//! pipeline compiles, no fetches and no decodes: one engine is built lazily on
//! first use and kept, and each thumbnail is two fills, one replayed stroke, one
//! small offscreen render, and a readback.
//!
//! Paint under the whole stroke, not bare substrate, because the ground half of
//! what a brush *is* here is what it does to the paint already down (§6.2):
//! a smudge drags gray into its wake, an eraser bites a gap, a wet brush's lift
//! muddies its own red — none of which bare canvas can show. Two grays rather
//! than one so the stroke's opacity and its pickup each read against a ground
//! that contrasts with them somewhere along the run.
//!
//! # The look is pinned, deliberately
//!
//! Thumbnails render on the flat ground under the neutral light at default media
//! parameters — never the current canvas's look. A thumbnail is the *brush's*
//! identity card, not a preview of today's lighting; pinning the look is what
//! lets the cache be keyed on the brush snapshot alone ([`lookup`]), so a
//! library's thumbnails survive every lighting, ground and background change
//! without regenerating. (The brush editor's live preview is the opposite
//! choice, made for the opposite reason.)
//!
//! The cache is per-session, in memory: with generation this cheap, re-rendering
//! a library at startup is a few frames of background GPU work, which is less
//! machinery than persisting image blobs against a `localStorage` quota.
//!
//! Like the brush editor's preview engine, the rig's engine deliberately skips
//! `state::with_engine`: it has no observable projection and no chrome reading
//! one back, so there is no publish to pair a mutation with.

use dioxus::dioxus_core::spawn_forever;
use dioxus::prelude::*;

use stark_engine::ViewTransform;
use stark_engine::command::{DocCommand, ViewCommand};
use stark_engine::{
    Background, Engine, EnvironmentId, InputSample, MediaParams, Offscreen, Rendered,
};
use stark_model::SurfaceId;
use stark_model::document::{BrushParams, FillOp, SelectionShape, Tool};
use stark_model::geom::{Extent2, Vec2};

use crate::platform::base64_encode;
use crate::presets::Wearable;
use crate::state::AppState;

/// Thumbnail pixel size: 2× the box a preset row shows it in (a full-bleed row,
/// `.preset-row` — 36 px tall in a 300 px panel), so it stays crisp on a dense
/// display. `cover` centres whatever width the row actually has.
const THUMB_W: u32 = 520;
const THUMB_H: u32 = 72;

/// Fixed jitter seed, for the reason the brush editor pins one: the thumbnail of
/// an unchanged preset must be byte-stable, or the cache key would be a lie.
const THUMB_SEED: u64 = 0x7B1D_EA0F_57A2;

/// The test stroke's fixed red — the one saturated thing in the picture, over
/// the two grays it is judged against.
const STROKE_COLOR: [f32; 3] = [0.82, 0.15, 0.12];

/// The two paint slabs the stroke crosses: light on the left, dark on the
/// right. Both sit away from the white substrate beneath them, so an eraser's
/// bite reads on either half.
const LIGHT_PAINT: [f32; 3] = [0.80, 0.80, 0.80];
const DARK_PAINT: [f32; 3] = [0.28, 0.28, 0.28];

/// The thumbnail machinery's signals. All root-owned (`state::root_signal`):
/// generation runs in `spawn_forever` tasks that outlive whichever panel asked.
#[derive(Clone, Copy)]
pub struct ThumbState {
    /// Finished thumbnails: a `data:image/png` URL per brush snapshot, found by
    /// comparing the snapshot itself ([`lookup`]).
    pub cache: Signal<Vec<(Wearable, String)>>,
    /// The kept engine + offscreen attachments; `None` until first use.
    pub rig: Signal<Option<Rig>>,
    /// The device and compiled pipelines to build that rig on, published once the
    /// main renderer lands (`state::publish_renderer`).
    ///
    /// **Held rather than fetched**, which is what `stark_engine::EngineShared` is for.
    /// The generator used to reach into `state.renderer` and borrow the whole live
    /// renderer — its canvas surface, its document, its in-flight gesture — for the
    /// length of building a thumbnail rig, purely to get at the device. So it could
    /// not run before a renderer existed and had to check for one every time round
    /// the loop. This is a handful of refcount bumps and outlives whoever published
    /// it.
    pub shared: Signal<Option<stark_engine::EngineShared>>,
    /// Whether the generator task is running — at most one at a time.
    pub busy: Signal<bool>,
}

/// The offscreen renderer the thumbnails share: a document-less-looking engine
/// (its document only ever holds the two strokes of the thumbnail in flight) and
/// the render attachments, kept because this render repeats at a steady size.
pub struct Rig {
    engine: Engine,
    off: Offscreen,
}

/// The cached entry for `w`, found by **comparing the brush snapshot itself**.
///
/// There is no digest, and that is the point. A cache key has one job — say
/// whether two brushes would render the same picture — and `Wearable` already
/// answers it exactly, by a derived `PartialEq` the compiler extends whenever
/// `BrushParams` gains a field. Every alternative is a second, hand-maintained
/// opinion about what a brush *is*: a hand-written hash silently ignores the new
/// field and serves a stale thumbnail for a brush that has changed, which is a
/// wrong picture with nothing anywhere to say so.
///
/// It was a digest of the JSON encoding, which had the drift ruled out but cost a
/// `serde_json::to_string` **per brush per call** — paid, before U2, on every
/// engine write. A linear scan of `PartialEq` over a library of a few dozen `Copy`
/// structs is cheaper than one of those serializations, so nothing was bought with
/// it. `Renderer::builtins` is a `Vec` looked up the same way and for the same
/// reason.
fn lookup(state: AppState, w: &Wearable) -> Option<String> {
    state
        .thumbs
        .cache
        .read()
        .iter()
        .find(|(cached, _)| cached == w)
        .map(|(_, url)| url.clone())
}

/// The thumbnail for `w`, if it has been generated. Subscribes, so a row showing
/// a placeholder re-renders when its image lands.
pub fn url(state: AppState, w: &Wearable) -> Option<String> {
    lookup(state, w)
}

/// Make sure every brush that has a picture to show has a thumbnail, generating
/// the missing ones in the background. Idempotent and cheap when nothing is
/// missing; safe to call from a render effect.
///
/// One generator at a time: the task re-scans after each thumbnail, so a brush
/// that appears while it runs is picked up by the running task rather than
/// needing a second one.
pub fn refresh(state: AppState) {
    if *state.thumbs.busy.peek() || next_missing(state).is_none() {
        return;
    }
    let mut busy = state.thumbs.busy;
    busy.set(true);
    spawn_forever(async move {
        while let Some(w) = next_missing(state) {
            if !generate(state, w).await {
                // No engine to render with (startup, or a lost device). The
                // library effect calls `refresh` again when the renderer lands.
                break;
            }
            // Generation just cached an entry for `w`, so the cache must now answer
            // for it — and if it does not, `next_missing` will hand back the same
            // brush forever and this loop will never end. That is reachable in
            // exactly one way, since the lookup is `PartialEq`: a brush holding a
            // parameter that is not equal to itself, which is to say a NaN. Nothing
            // in the app can produce one (sliders clamp, and JSON cannot even carry
            // it), so this is a guarantee of termination rather than a case that
            // happens — the loop cannot spin, whatever is in the library.
            if lookup(state, &w).is_none() {
                tracing::warn!(
                    "a brush parameter does not compare equal to itself; \
                                skipping the rest of the thumbnails"
                );
                break;
            }
        }
        let mut busy = state.thumbs.busy;
        busy.set(false);
    });
}

/// The first brush with no thumbnail yet: the library in order, then the
/// quick-brush rack (§18.1.8), whose overlay shows the same picture per slot.
///
/// The rack is scanned as well as the library rather than instead of being
/// assumed to be a subset of it, because it is not one: a slot tuned under a
/// hold keeps a brush no preset holds (`slots::release`), and that is exactly the
/// slot whose row would otherwise be the only blank one in the column. Presets
/// first, so the list a user is looking at fills in before the rack they have to
/// hold a key to see — and a slot that *did* come from a preset costs nothing
/// here, since the two hash to one key.
fn next_missing(state: AppState) -> Option<Wearable> {
    let cache = state.thumbs.cache.peek();
    let presets = state.presets.peek();
    let rack = state.slots.brushes.peek();
    presets
        .iter()
        .map(|e| e.brush)
        .chain(rack.iter().flatten().copied())
        .find(|w| !cache.iter().any(|(cached, _)| cached == w))
}

/// Render one thumbnail and put it in the cache. `false` when there is no main
/// renderer to share an engine from yet — the one condition worth stopping for;
/// a preset whose render fails for its own reasons is skipped by caching a blank
/// entry rather than retried forever.
async fn generate(state: AppState, w: Wearable) -> bool {
    // Everything up to the readback happens under the rig borrow, which must end
    // before the await — the same borrow bargain as `Engine::export` itself.
    let readback = {
        let mut rig_signal = state.thumbs.rig;
        let mut guard = rig_signal.write();
        if guard.is_none() {
            // The shared half, not the renderer: nothing here needs the canvas, the
            // document or the gesture in flight.
            let Some(shared) = state.thumbs.shared.peek().clone() else {
                return false;
            };
            let mut engine = Engine::on_shared(shared, Extent2::new(THUMB_W, THUMB_H));
            // Pin the look the module doc promises: flat ground, neutral light,
            // default media, white substrate (what an eraser's bite reveals).
            // The document opened on the donor's ground (`Engine::new_sharing`),
            // so the surface is set back explicitly.
            engine.process(DocCommand::SetSurface(SurfaceId::default()));
            engine.process(DocCommand::SetBackground([1.0, 1.0, 1.0]));
            engine.process(ViewCommand::SetEnvironment(EnvironmentId::default()));
            engine.process(ViewCommand::SetMediaParams(MediaParams::default()));
            *guard = Some(Rig {
                engine,
                off: Offscreen::default(),
            });
        }
        let rig = guard.as_mut().expect("just built");
        let view = thumb_view(&w.params);
        // The scene, in stack order: the two paint slabs the stroke acts on —
        // the whole ground is paint, so smearing, lifting and bleeding read
        // everywhere along the run — then the stroke itself, towed through the
        // preset's own smoothing. A fill carries its own region (§18.0.4), so
        // no selection is involved and the stroke is gated by nothing.
        let layer = rig.engine.observe().active_layer;
        let h = half_extent(&view);
        // Overhang every outer edge so no sliver of substrate survives the
        // rounding of the view's edges; the halves meet exactly at x = 0.
        let (over_x, over_y) = (h.x * 1.05, h.y * 1.05);
        for (min_x, max_x, color) in [(-over_x, 0.0, LIGHT_PAINT), (0.0, over_x, DARK_PAINT)] {
            rig.engine.process(DocCommand::Fill {
                layer,
                op: FillOp::new(
                    SelectionShape::rect_from_corners(
                        Vec2::new(min_x, -over_y),
                        Vec2::new(max_x, over_y),
                    ),
                    0.0,
                    color,
                    1.0,
                ),
            });
        }
        let mut brush = w.params;
        brush.color[..3].copy_from_slice(&STROKE_COLOR);
        rig.engine.process(ViewCommand::SetBrush(brush));
        let rope = crate::input::rope_in(view, w.smoothing);
        rig.engine
            .replay_stroke_seeded(Tool::Brush, &test_stroke(&view), THUMB_SEED, rope);
        // The whole rig document, which is only ever the thumbnail in flight — there
        // is no layer here to single out.
        let readback = rig.engine.export_view(
            &mut rig.off,
            view,
            None,
            Background::Substrate,
            Rendered::Committed,
        );
        // The render is already submitted; put the document back — one undo per
        // action above — before the await, so the rig is clean whoever borrows
        // it next.
        for _ in 0..3 {
            rig.engine.process(DocCommand::Undo);
        }
        readback
    };
    let url = match readback {
        // Two ways to come back empty, and both cache the miss so the generator
        // cannot spin on it: a view the device refuses (unreachable at these
        // sizes), and a readback that failed because the GPU did (§5) — which the
        // canvas will be reporting through `ObservableState::gpu_failure` anyway,
        // so a thumbnail is not the place to raise it.
        Ok(f) => f
            .await
            .ok()
            .and_then(|image| image.to_png().ok())
            .map(|png| format!("data:image/png;base64,{}", base64_encode(&png)))
            .unwrap_or_default(),
        Err(_) => String::new(),
    };
    let mut cache = state.thumbs.cache;
    cache.write().push((w, url));
    true
}

/// The view a preset's thumbnail renders through, scaled to the brush: the test
/// stroke runs ~18 radii, so the same view shows a pen as a line and an airbrush
/// as the soft mass it is, each filling the thumbnail rather than being drawn at
/// some one zoom that suits neither.
fn thumb_view(b: &BrushParams) -> ViewTransform {
    let r = b.radius.max(1.0);
    // Canvas-space width shown; the wide row's aspect leaves the height ~3
    // radii, which is what bounds the S's swing in `test_stroke`.
    let span = 22.0 * r;
    let size = Extent2::new(THUMB_W, THUMB_H);
    ViewTransform {
        center: Vec2::ZERO,
        zoom: THUMB_W as f32 / span,
        ..ViewTransform::identity(size)
    }
}

/// The canvas-space rect `view` shows, as (half-width, half-height) about the
/// origin — both stroke layouts are placed from it.
fn half_extent(view: &ViewTransform) -> Vec2 {
    Vec2::new(
        view.viewport.width as f32 / view.zoom * 0.5,
        view.viewport.height as f32 / view.zoom * 0.5,
    )
}

/// The test stroke: an S-curve **across** the thumbnail with a pressure bell and
/// a ramping forward tilt — the brush editor's seeded stroke, laid along the
/// wide axis. Same shape for every preset, so the thumbnails read as one family
/// and what differs between them is only the brush.
fn test_stroke(view: &ViewTransform) -> Vec<InputSample> {
    let h = half_extent(view);
    let run = h.x * 0.84;
    // The S's swing: as much of the height as the full-pressure tip (radius =
    // span/22, from `thumb_view`) leaves free.
    let swing = (h.y * 0.9 - h.x / 11.0).max(0.0);
    const N: usize = 48;
    (0..N)
        .map(|i| {
            let t = i as f32 / (N - 1) as f32;
            InputSample {
                pos: Vec2::new(
                    -run + t * 2.0 * run,
                    (t * std::f32::consts::TAU).sin() * swing,
                ),
                pressure: (t * std::f32::consts::PI).sin().clamp(0.08, 1.0),
                // Lean **across** the run, growing over the stroke, so tilt-driven
                // settings read. Across and not along, because the axis a lean
                // aims is now anisotropic (§6.6): a pen leaned along its own
                // travel draws its tip out *forwards*, which lays more paint
                // without widening the mark and would show a pencil doing
                // nothing visible. Held at one azimuth while the S turns under
                // it, so one preview shows both readings — broad where the
                // stroke runs across the lean, dark and fine where it runs
                // along it, which is the whole of what the axis does.
                tilt: Vec2::new(0.0, 0.65 * t),
                time: (t * 0.7) as f64,
            }
        })
        .collect()
}
