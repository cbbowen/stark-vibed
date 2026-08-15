//! Rendered thumbnails for brush presets: a miniature test stroke per preset,
//! generated offscreen and cached (§11).
//!
//! Each thumbnail is the brush editor's preview scene in miniature — a red
//! reference band with the preset's own stroke laid across it — rendered by a
//! **shared** engine (`Renderer::shared_engine`), so it costs no pipeline
//! compiles, no fetches and no decodes: one engine is built lazily on first use
//! and kept, and each thumbnail is two replayed strokes, one small offscreen
//! render, and a readback.
//!
//! # The look is pinned, deliberately
//!
//! Thumbnails render on the flat ground under the neutral light at default media
//! parameters — never the current canvas's look. A thumbnail is the *brush's*
//! identity card, not a preview of today's lighting; pinning the look is what
//! lets the cache key be the brush snapshot alone ([`key`]), so a library's
//! thumbnails survive every lighting, ground and background change without
//! regenerating. (The brush editor's live preview is the opposite choice, made
//! for the opposite reason.)
//!
//! The cache is per-session, in memory: with generation this cheap, re-rendering
//! a library at startup is a few frames of background GPU work, which is less
//! machinery than persisting image blobs against a `localStorage` quota.
//!
//! Like the brush editor's preview engine, the rig's engine deliberately skips
//! `state::with_engine`: it has no observable projection and no chrome reading
//! one back, so there is no publish to pair a mutation with.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use dioxus::dioxus_core::spawn_forever;
use dioxus::prelude::*;

use stark_core::command::{DocCommand, ViewCommand};
use stark_core::document::{BrushParams, BrushShape, Tool};
use stark_core::geom::{Extent2, Vec2};
use stark_core::{
    Background, Engine, EnvironmentId, InputSample, MediaParams, Offscreen, Rendered, SurfaceId,
    ViewTransform,
};

use crate::platform::base64_encode;
use crate::presets::Wearable;
use crate::state::AppState;

/// Thumbnail pixel size: 2.5× the CSS box the preset rows display it in
/// (`.preset-thumb`, 60 × 24), same aspect exactly, so it stays crisp on a
/// dense display and `cover` never crops it.
const THUMB_W: u32 = 150;
const THUMB_H: u32 = 60;

/// Fixed jitter seed, for the reason the brush editor pins one: the thumbnail of
/// an unchanged preset must be byte-stable, or the cache key would be a lie.
const THUMB_SEED: u64 = 0x7B1D_EA0F_57A2;

/// The test stroke's fixed color — the brush editor's preview gold, so the two
/// places a brush is shown as a stroke show it the same way.
const STROKE_COLOR: [f32; 3] = [0.852, 0.645, 0.125];

/// The reference band's fixed red — likewise the editor's.
const BAND_COLOR: [f32; 4] = [0.82, 0.15, 0.12, 1.0];

/// The thumbnail machinery's signals. All root-owned (`state::root_signal`):
/// generation runs in `spawn_forever` tasks that outlive whichever panel asked.
#[derive(Clone, Copy)]
pub struct ThumbState {
    /// Finished thumbnails: a `data:image/png` URL per brush snapshot digest.
    pub cache: Signal<HashMap<u64, String>>,
    /// The kept engine + offscreen attachments; `None` until first use.
    pub rig: Signal<Option<Rig>>,
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

/// The cache key: a digest of the whole brush snapshot — parameters and feel —
/// via its JSON encoding, which `crate::presets` already maintains as the stable
/// wire form of a stored brush. Session-local (the hasher is `std`'s, unkeyed
/// only within one run), which matches the cache being session-local.
pub fn key(w: &Wearable) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    serde_json::to_string(&w.params)
        .unwrap_or_default()
        .hash(&mut h);
    w.smoothing.to_bits().hash(&mut h);
    h.finish()
}

/// The thumbnail for `w`, if it has been generated. Subscribes, so a row showing
/// a placeholder re-renders when its image lands.
pub fn url(state: AppState, w: &Wearable) -> Option<String> {
    state.thumbs.cache.read().get(&key(w)).cloned()
}

/// Make sure every preset in the library has a thumbnail, generating the missing
/// ones in the background. Idempotent and cheap when nothing is missing; safe to
/// call from a render effect.
///
/// One generator at a time: the task re-scans the library after each thumbnail,
/// so presets saved while it runs are picked up by the running task rather than
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
        }
        let mut busy = state.thumbs.busy;
        busy.set(false);
    });
}

/// The first library entry with no thumbnail yet, oldest first.
fn next_missing(state: AppState) -> Option<Wearable> {
    let cache = state.thumbs.cache.peek();
    state
        .presets
        .peek()
        .iter()
        .map(|e| e.brush)
        .find(|w| !cache.contains_key(&key(w)))
}

/// Render one thumbnail and put it in the cache. `false` when there is no main
/// renderer to share an engine from yet — the one condition worth stopping for;
/// a preset whose render fails for its own reasons is skipped by caching a blank
/// entry rather than retried forever.
async fn generate(state: AppState, w: Wearable) -> bool {
    let k = key(&w);
    // Everything up to the readback happens under the rig borrow, which must end
    // before the await — the same borrow bargain as `Engine::export` itself.
    let readback = {
        let mut rig_signal = state.thumbs.rig;
        let mut guard = rig_signal.write();
        if guard.is_none() {
            let renderer = state.renderer.peek();
            let Some(main) = renderer.as_ref() else {
                return false;
            };
            let mut engine = main.shared_engine(Extent2::new(THUMB_W, THUMB_H));
            // Pin the look the module doc promises: flat ground, neutral light,
            // default media. The document opened on the donor's ground
            // (`Engine::new_sharing`), so the surface is set back explicitly.
            engine.process(DocCommand::SetSurface(SurfaceId::default()));
            engine.process(ViewCommand::SetEnvironment(EnvironmentId::default()));
            engine.process(ViewCommand::SetMediaParams(MediaParams::default()));
            *guard = Some(Rig {
                engine,
                off: Offscreen::default(),
            });
        }
        let rig = guard.as_mut().expect("just built");
        let view = thumb_view(&w.params);
        // The scene, in stack order: the band the stroke has to cross — what
        // makes an eraser or a smudge legible at all — then the stroke itself,
        // towed through the preset's own smoothing.
        rig.engine.process(ViewCommand::SetBrush(band_brush(&view)));
        rig.engine
            .replay_stroke_seeded(Tool::Brush, &band_stroke(&view), THUMB_SEED, 0.0);
        let mut brush = w.params;
        brush.color[..3].copy_from_slice(&STROKE_COLOR);
        rig.engine.process(ViewCommand::SetBrush(brush));
        let rope = crate::input::rope_in(view, w.smoothing);
        rig.engine
            .replay_stroke_seeded(Tool::Brush, &test_stroke(&view), THUMB_SEED, rope);
        let readback = rig.engine.export_view(
            &mut rig.off,
            view,
            Background::Substrate,
            Rendered::Committed,
        );
        // The render is already submitted; put the document back before the
        // await so the rig is clean whoever borrows it next.
        rig.engine.process(DocCommand::Undo);
        rig.engine.process(DocCommand::Undo);
        readback
    };
    let url = match readback {
        Ok(f) => {
            let image = f.await;
            image
                .to_png()
                .map(|png| format!("data:image/png;base64,{}", base64_encode(&png)))
                .unwrap_or_default()
        }
        // A view the device refuses (unreachable at these sizes) — cache the
        // miss so the generator cannot spin on it.
        Err(_) => String::new(),
    };
    let mut cache = state.thumbs.cache;
    cache.write().insert(k, url);
    true
}

/// The view a preset's thumbnail renders through, scaled to the brush: the test
/// stroke runs ~10 radii, so the same view shows a pen as a line and an airbrush
/// as the soft mass it is, each filling the thumbnail rather than being drawn at
/// some one zoom that suits neither.
fn thumb_view(b: &BrushParams) -> ViewTransform {
    let r = b.radius.max(1.0);
    // Canvas-space width shown: the stroke's run plus a radius of margin each side.
    let span = 12.0 * r;
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

/// The vertical reference band, crossing the (horizontal) test stroke at its
/// middle and running off both edges — the editor's red band, in miniature and
/// turned to fit the wide frame. Scaled with the view so it reads the same
/// thickness in every thumbnail.
fn band_brush(view: &ViewTransform) -> BrushParams {
    let h = half_extent(view);
    BrushParams {
        color: BAND_COLOR,
        radius: h.y * 0.42,
        shape: BrushShape::Round { hardness: 0.9 },
        drain: 0.0,
        ..BrushParams::default()
    }
}

fn band_stroke(view: &ViewTransform) -> Vec<InputSample> {
    let h = half_extent(view);
    const N: usize = 8;
    (0..N)
        .map(|i| {
            let t = i as f32 / (N - 1) as f32;
            InputSample {
                pos: Vec2::new(0.0, -1.5 * h.y + t * 3.0 * h.y),
                pressure: 1.0,
                ..Default::default()
            }
        })
        .collect()
}

/// The test stroke: an S-curve **across** the thumbnail with a pressure bell and
/// a ramping forward tilt — the brush editor's seeded stroke, laid along the
/// wide axis. Same shape for every preset, so the thumbnails read as one family
/// and what differs between them is only the brush.
fn test_stroke(view: &ViewTransform) -> Vec<InputSample> {
    let h = half_extent(view);
    let run = h.x * 0.84;
    // The S's swing: as much of the height as the full-pressure tip leaves free.
    let swing = (h.y * 0.9 - h.x / 6.0).max(0.0);
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
                // Lean along the travel, growing over the stroke, so
                // tilt-driven settings read (the pencil widens, the knife lays
                // on more and more).
                tilt: Vec2::new(0.65 * t, 0.0),
                time: (t * 0.7) as f64,
            }
        })
        .collect()
}
