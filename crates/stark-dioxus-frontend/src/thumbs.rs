//! Rendered thumbnails for brushes: a miniature test stroke per brush, generated
//! offscreen and cached (§11). Two viewers share the one cache — the preset
//! library's rows, and the quick-brush rack the number keys draw (§18.1.8) —
//! and they share it exactly because the key is the brush itself, so a brush
//! that is in both places is rendered once: a slot is its preset at a size and
//! flow (`slots::resolve`), and one still at the preset's own is the preset's
//! picture. Itself *as the picture paints it*, painting color aside
//! ([`keyed`]): one preset saved in two colors is one picture.
//!
//! Each thumbnail is a mid-gray stroke drawn across a canvas laid **entirely in
//! paint** — a light-gray slab on the left half, a dark-gray slab on the right —
//! rendered by a **shared** engine (`Renderer::shared_engine`), so it costs no
//! pipeline compiles, no fetches and no decodes: one engine is built lazily on
//! first use and kept, and each thumbnail is two fills, one replayed stroke, one
//! small offscreen render, and a readback.
//!
//! Paint under the whole stroke, not bare substrate, because the substrate half of
//! what a brush *is* here is what it does to the paint already down (§6.2):
//! a smudge drags gray into its wake, an eraser bites a gap, a wet brush's lift
//! muddies its own body — none of which bare canvas can show. Two grays rather
//! than one so the stroke's opacity and its pickup each read against a substrate
//! that contrasts with them somewhere along the run.
//!
//! # The look is pinned, deliberately
//!
//! Thumbnails render on the flat substrate under the neutral light at default media
//! parameters — never the current canvas's look. A thumbnail is the *brush's*
//! identity card, not a preview of today's lighting; pinning the look is what
//! lets the cache be keyed on the brush snapshot alone ([`lookup`]), so a
//! library's thumbnails survive every lighting, substrate and substrate-color change
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
use stark_engine::command::Tool;
use stark_model::Srgb;

use stark_engine::Extent2;
use stark_engine::ViewTransform;
use stark_engine::command::InputSample;
use stark_engine::command::{DocCommand, ViewCommand};
use stark_engine::{Background, Engine, EnvironmentId, MediaParams, Offscreen, Rendered};
use stark_model::SubstrateId;
use stark_model::document::{FillOp, SelectionShape};
use stark_model::geom::Vec2;

use crate::platform::base64_encode;
use crate::state::{AppState, root_signal};
use stark_ui::brush_config::{BrushConfig, Transient};

/// Thumbnail pixel size: 2× the box a preset row shows it in (a full-bleed row,
/// `.preset-row` — 36 px tall in a 300 px panel), so it stays crisp on a dense
/// display. `cover` centres whatever width the row actually has.
const THUMB_W: u32 = 520;
const THUMB_H: u32 = 72;

/// Fixed jitter seed, for the reason the brush editor pins one: the thumbnail of
/// an unchanged preset must be byte-stable, or the cache key would be a lie.
const THUMB_SEED: u64 = 0x7B1D_EA0F_57A2;

/// The test stroke's fixed neutral: a mid gray pitched between the two slabs,
/// so it reads darker over the light half and lighter over the dark half. Gray
/// rather than any hue for the chrome's own reason (stark.css): the library is
/// a column of these standing beside the painting, and one saturated color
/// repeated down it is a thumb on the scale of every color the artist weighs.
const STROKE_COLOR: [f32; 3] = [0.55, 0.55, 0.55];

/// The two paint slabs the stroke crosses: light on the left, dark on the
/// right. Both sit away from the white substrate beneath them, so an eraser's
/// bite reads on either half.
const LIGHT_PAINT: [f32; 3] = [0.80, 0.80, 0.80];
const DARK_PAINT: [f32; 3] = [0.28, 0.28, 0.28];

/// The thumbnail machinery's signals. All root-owned (`state::root_signal`):
/// generation runs in `spawn_forever` tasks that outlive whichever panel asked.
#[derive(Clone, Copy)]
pub struct ThumbState {
    /// Finished thumbnails: a `data:image/png` URL per brush snapshot — both
    /// halves, since a slot's tune is part of its picture — found by comparing
    /// the snapshot itself ([`lookup`]).
    pub cache: Signal<Vec<((BrushConfig, Transient), String)>>,
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

impl ThumbState {
    /// Its signals, root-owned like every other group of them
    /// (`state::root_signal`); built here rather than in `AppState::new` so the
    /// fields and the values they open on stay in one place.
    pub(crate) fn new() -> Self {
        Self {
            cache: root_signal(Vec::new),
            rig: root_signal(|| None),
            shared: root_signal(|| None),
            busy: root_signal(|| false),
        }
    }
}

/// The offscreen renderer the thumbnails share: a document-less-looking engine
/// (its document only ever holds the two strokes of the thumbnail in flight) and
/// the render attachments, kept because this render repeats at a steady size.
pub struct Rig {
    engine: Engine,
    off: Offscreen,
}

/// The cached entry for `w`, found by **comparing the brush snapshot itself**
/// ([`keyed`] first, which is the one field the picture does not take from it).
///
/// There is no digest, and that is the point. A cache key has one job — say
/// whether two brushes would render the same picture — and `BrushConfig` already
/// answers it exactly, by a derived `PartialEq` the compiler extends whenever
/// the brush gains a field. Every alternative is a second, hand-maintained
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
fn lookup(state: AppState, w: &BrushConfig, t: Transient) -> Option<String> {
    let key = (*w, keyed(t));
    state
        .thumbs
        .cache
        .read()
        .iter()
        .find(|(cached, _)| *cached == key)
        .map(|(_, url)| url.clone())
}

/// The tune **as the thumbnail paints it**: the test stroke's own gray in place
/// of the painting color. Both the render and the cache key go through it, so
/// the key cannot come to disagree with the picture about what a thumbnail is.
///
/// The picture ignores the RGB it is handed — every thumbnail is drawn in
/// [`STROKE_COLOR`], over the same two grays — so two tunes differing only in
/// color are one picture, and keying on the raw snapshot would file
/// that one picture under a fresh name for every color a tune happened to be
/// saved at: a stored tune carries the color the hand held when the snapshot
/// was taken (`presets::wear` keeps the live one on the way back in,
/// §18.1.8), so one tool saved twice in two colors would be rendered twice
/// for one row's worth of picture.
///
/// The effect's opacity is deliberately untouched: it is the brush's own — the
/// stroke really is laid under it (§6.2) — and so are the size and flow beside
/// the color, for the same reason: a slot tuned off its preset is a different
/// stroke, and gets one.
fn keyed(t: Transient) -> Transient {
    Transient {
        color: STROKE_COLOR,
        ..t
    }
}

/// The thumbnail for `w` at `t`, if it has been generated. Subscribes, so a row
/// showing a placeholder re-renders when its image lands.
pub fn url(state: AppState, w: &BrushConfig, t: Transient) -> Option<String> {
    lookup(state, w, t)
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
        while let Some((w, t)) = next_missing(state) {
            if !generate(state, w, t).await {
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
            if lookup(state, &w, t).is_none() {
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
/// assumed to be a subset of it, because it is not one: a slot is its preset at
/// a size and flow of its own (`slots::resolve`), and one tuned off the preset's
/// is a brush no preset holds — exactly the slot whose row would otherwise be
/// the only blank one in the column. Presets first, so the list a user is
/// looking at fills in before the rack they have to hold a key to see — and a
/// slot still at its preset's own size costs nothing here, since the two are one
/// key.
fn next_missing(state: AppState) -> Option<(BrushConfig, Transient)> {
    let cache = state.thumbs.cache.peek();
    let presets = state.presets.peek();
    let rack = state.slots.brushes.peek();
    presets
        .iter()
        .map(|e| (e.brush, e.transient))
        .chain(
            rack.iter()
                .flatten()
                .filter_map(|slot| crate::slots::resolve(&presets, slot)),
        )
        // Asked on the same terms the cache answers on ([`keyed`]) — and it has
        // to be, or a brush filed under its rendered color would be reported
        // missing forever and `refresh` would never finish its scan.
        .find(|(w, t)| {
            let key = (*w, keyed(*t));
            !cache.iter().any(|(cached, _)| *cached == key)
        })
}

/// Render one thumbnail and put it in the cache. `false` when there is no main
/// renderer to share an engine from yet — the one condition worth stopping for;
/// a preset whose render fails for its own reasons is skipped by caching a blank
/// entry rather than retried forever.
async fn generate(state: AppState, w: BrushConfig, t: Transient) -> bool {
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
            // Pin the look the module doc promises: flat substrate, neutral light,
            // default media, white substrate (what an eraser's bite reveals).
            // The document opened on the donor's substrate (`Engine::new_sharing`),
            // so the substrate is set back explicitly.
            engine.process(DocCommand::SetSubstrate(SubstrateId::default()));
            engine.process(DocCommand::SetSubstrateColor(Srgb::new([1.0, 1.0, 1.0])));
            engine.process(ViewCommand::SetEnvironment(EnvironmentId::default()));
            engine.process(ViewCommand::SetMediaParams(MediaParams::default()));
            *guard = Some(Rig {
                engine,
                off: Offscreen::default(),
            });
        }
        let rig = guard.as_mut().expect("just built");
        let view = thumb_view(t.size);
        // The scene, in stack order: the two paint slabs the stroke acts on —
        // the whole substrate is paint, so smearing, lifting and bleeding read
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
                    Srgb::new(color),
                    1.0,
                ),
            });
        }
        // The tune the cache will file this under, which is the tune the stroke
        // is laid at: one statement of "the color a thumbnail is painted in".
        let keyed_tune = keyed(t);
        rig.engine.process(ViewCommand::SetBrush {
            brush: w.params(keyed_tune),
            color: keyed_tune.color,
        });
        let rope = stark_ui::input::rope(view, w.smoothing);
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
    cache.write().push(((w, keyed(t)), url));
    true
}

/// The view a preset's thumbnail renders through, scaled to the brush: the test
/// stroke runs ~18 radii, so the same view shows a pen as a line and an airbrush
/// as the soft mass it is, each filling the thumbnail rather than being drawn at
/// some one zoom that suits neither.
fn thumb_view(size: f32) -> ViewTransform {
    let r = size.max(1.0);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn tune(color: [f32; 3]) -> Transient {
        Transient {
            color,
            ..Default::default()
        }
    }

    /// The second render this key exists to rule out: a stored tune carries the
    /// color the hand held and `presets::wear` keeps the live one over it
    /// (§18.1.8), so one tool saved twice in two colors is one picture — the
    /// thumbnail is painted in its own gray either way — and must not cost two.
    #[test]
    fn two_painting_colors_of_one_brush_are_one_thumbnail() {
        assert_eq!(keyed(tune([0.9, 0.1, 0.1])), keyed(tune([0.1, 0.2, 0.9])));
    }

    /// The size and flow are the tune's other two knobs, and they really are the
    /// picture's — a slot tuned off its preset is a different stroke (§6.2) —
    /// so the key normalizes the color alone.
    #[test]
    fn the_tune_is_part_of_the_picture_though() {
        let resized = Transient {
            size: 80.0,
            ..tune([0.9, 0.1, 0.1])
        };
        assert_ne!(keyed(tune([0.9, 0.1, 0.1])), keyed(resized));
    }
}
