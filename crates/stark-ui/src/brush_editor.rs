//! The brush editor pop-up (DESIGN.md §11): a Procreate-style dialog with a live
//! test-stroke preview over grouped settings.
//!
//! The preview is a second `Engine` on its **own document** but the **shared GPU
//! device** ([`render::init_shared`]), mirroring the main canvas's surface,
//! environment, lighting, and background so a stroke reads exactly like it will
//! on the real canvas. One test stroke — a seeded default with a pressure bell
//! and a ramping forward tilt (so pressure/tilt-driven settings respond even
//! with a mouse), or whatever the user last drew on the preview — is re-stroked
//! (undo → set brush → replay → paint) as settings change — throttled to one
//! re-stroke per [`RESTROKE_THROTTLE_MS`] so slider drags stay responsive while
//! the stroke still updates live.
//!
//! Settings are grouped into collapsible sections by what they affect, with
//! rarely-used knobs behind a per-section "Show more" and modulation sliders
//! (pressure→load, tilt→deposit) indented under — and only shown alongside — the
//! axis they modulate.

use dioxus::html::input_data::MouseButton;
use dioxus::prelude::*;

use stark_core::document::{BrushParams, BrushShape, OrientationSource, Tool};
use stark_core::geom::Vec2;
use stark_core::{EnvironmentId, InputCommand, InputSample};

use crate::render::{self, Renderer};
use crate::{
    capture_pointer, set_bristles, set_brush_preset, set_knife, set_orientation, set_shape,
    sleep_ms, update_brush, AppState, Slider,
};

/// The preview `<canvas>`'s DOM id (the main canvas is `render::CANVAS_ID`).
const PREVIEW_CANVAS_ID: &str = "brush-preview-canvas";

/// Minimum gap between slider-driven preview re-strokes. A re-stroke replays the
/// whole test stroke through the engine, and slider `input` events fire faster
/// than that can run, so unthrottled drags back up and the slider lags. ~10
/// updates a second still reads as live.
const RESTROKE_THROTTLE_MS: i32 = 100;

/// Shared `Copy` handle to the preview's signals.
#[derive(Clone, Copy)]
struct Preview {
    /// The preview surface + engine; `None` until its async init completes.
    renderer: Signal<Option<Renderer>>,
    /// The test stroke (canvas-space samples), replayed on every setting change.
    samples: Signal<Vec<InputSample>>,
    /// Samples of an in-progress user stroke on the preview canvas.
    rec: Signal<Vec<InputSample>>,
    /// Whether the user is mid-stroke on the preview canvas.
    drawing: Signal<bool>,
    /// Whether a committed stroke is on the preview document (undo it before replaying).
    committed: Signal<bool>,
    /// Re-stroke throttle gate: `None` when idle; while the post-re-stroke
    /// cooldown runs, `Some(dirty)` records whether an edit arrived during it
    /// (and so a trailing re-stroke is owed).
    throttle: Signal<Option<bool>>,
}

/// The brush editor dialog. Mounted only while open (so each open re-inits the
/// preview against the current canvas look and re-seeds the section state).
#[component]
pub fn BrushEditorModal(on_close: EventHandler<()>) -> Element {
    let state = use_context::<AppState>();
    let preview = Preview {
        renderer: use_signal(|| None),
        samples: use_signal(Vec::new),
        rec: use_signal(Vec::new),
        drawing: use_signal(|| false),
        committed: use_signal(|| false),
        throttle: use_signal(|| None),
    };

    // Section fold state: the everyday groups start open, the specialised ones closed.
    let tip_open = use_signal(|| true);
    let paint_open = use_signal(|| true);
    let pickup_open = use_signal(|| false);
    let flow_open = use_signal(|| false);
    let surface_open = use_signal(|| false);
    // Per-section "Show more" for the rarely-touched knobs.
    let tip_more = use_signal(|| false);
    let paint_more = use_signal(|| false);
    let pickup_more = use_signal(|| false);

    let brush = state
        .obs
        .read()
        .as_ref()
        .map(|o| o.brush)
        .unwrap_or_default();
    let is_round = matches!(brush.shape, BrushShape::Round);
    let d = brush.dynamics;
    let chip = |active: bool| if active { "chip active" } else { "chip" };

    rsx! {
        div {
            class: "modal-backdrop",
            onclick: move |_| on_close.call(()),
            div {
                class: "modal-dialog be-dialog",
                onclick: move |e| e.stop_propagation(),

                div { class: "be-header",
                    div { class: "modal-title", "Brush" }
                    button { class: "btn btn-primary", onclick: move |_| on_close.call(()), "Done" }
                }

                // Live test canvas: draw on it to replace the test stroke; ↺ restores
                // the default. The stroke re-renders as every setting changes.
                div { class: "be-preview-wrap",
                    canvas {
                        id: PREVIEW_CANVAS_ID,
                        class: "brush-preview",
                        onmounted: move |_| { spawn(init_preview(state, preview)); },
                        onpointerdown: move |e| {
                            if e.trigger_button() == Some(MouseButton::Primary) {
                                capture_pointer(&e);
                                start_preview_stroke(preview, &e);
                            }
                        },
                        onpointermove: move |e| {
                            if (preview.drawing)() { move_preview_stroke(preview, &e); }
                        },
                        onpointerup: move |_| end_preview_stroke(preview),
                        onpointercancel: move |_| cancel_preview_stroke(state, preview),
                    }
                    div { class: "be-preview-hint", "Test stroke — draw here to replace it" }
                    button {
                        class: "be-preview-reset",
                        title: "Restore the default test stroke",
                        onclick: move |_| reset_stroke(state, preview),
                        "\u{21BA}"
                    }
                }

                // One-click starting points (DESIGN §6.2); every knob below stays live.
                div { class: "be-presets",
                    div { class: "be-presets-label", "Preset" }
                    button { class: "chip", onclick: move |_| { set_brush_preset(state); restroke(state, preview); }, "Brush" }
                    button { class: "chip", onclick: move |_| { set_knife(state); restroke(state, preview); }, "Palette Knife" }
                }

                div { class: "be-sections",
                    Section {
                        title: "Tip", desc: "The footprint the stroke sweeps along the path.",
                        open: tip_open,
                        div { class: "brush-shapes",
                            button { class: chip(is_round),
                                onclick: move |_| { set_shape(state, BrushShape::Round, 0.25); restroke(state, preview); },
                                "Round" }
                            button { class: chip(!is_round),
                                onclick: move |_| { set_bristles(state); restroke(state, preview); },
                                "Bristles" }
                        }
                        // Orientation only matters for non-round tips (per-orientation
                        // footprint slices, DESIGN §6.6); hardness only for the
                        // procedural round tip.
                        if !is_round {
                            div { class: "brush-shapes",
                                button { class: chip(brush.orientation == OrientationSource::FollowStroke),
                                    onclick: move |_| { set_orientation(state, OrientationSource::FollowStroke); restroke(state, preview); },
                                    "Follow stroke" }
                                button { class: chip(brush.orientation == OrientationSource::Pen),
                                    onclick: move |_| { set_orientation(state, OrientationSource::Pen); restroke(state, preview); },
                                    "Pen angle" }
                            }
                        }
                        Slider { label: "Size", min: 1.0, max: 120.0, value: brush.radius,
                            oninput: move |v| edit(state, preview, move |b| b.radius = v) }
                        if is_round {
                            Slider { label: "Hardness", min: 0.0, max: 0.95, value: brush.hardness,
                                oninput: move |v| edit(state, preview, move |b| b.hardness = v) }
                        }
                        More { open: tip_more,
                            Slider { label: "Spacing", min: 0.02, max: 1.0, value: brush.spacing,
                                oninput: move |v| edit(state, preview, move |b| b.spacing = v) }
                        }
                    }

                    Section {
                        title: "Paint", desc: "The brush's own paint: how much goes down and what it's like.",
                        open: paint_open,
                        // `add` is the only source term of the six-axis tool (§6.2).
                        Slider { label: "Amount", min: 0.0, max: 1.0, value: d.add,
                            oninput: move |v| edit(state, preview, move |b| b.dynamics.add = v) }
                        Slider { label: "Opacity", min: 0.0, max: 1.0, value: brush.color[3],
                            oninput: move |v| edit(state, preview, move |b| b.color[3] = v) }
                        Slider { label: "Rate", min: 0.05, max: 1.0, value: brush.flow,
                            oninput: move |v| edit(state, preview, move |b| b.flow = v) }
                        More { open: paint_more,
                            Slider { label: "Thickness", min: 0.0, max: 1.5, value: brush.height,
                                oninput: move |v| edit(state, preview, move |b| b.height = v) }
                            Slider { label: "Wetness", min: 0.0, max: 1.0, value: brush.wetness,
                                oninput: move |v| edit(state, preview, move |b| b.wetness = v) }
                            // Reservoir depletion per px travelled — the stroke runs dry.
                            Slider { label: "Drain", min: 0.0, max: 0.01, value: brush.drain,
                                oninput: move |v| edit(state, preview, move |b| b.drain = v) }
                        }
                    }

                    Section {
                        title: "Pickup", desc: "Canvas paint moving on and off the tool — smudge, knife, eraser.",
                        open: pickup_open,
                        // Each axis and how the pen modulates it live together: the
                        // modulation slider sits indented under its axis, and only
                        // appears while that axis is active (it has no effect at 0).
                        Slider { label: "Lift", min: 0.0, max: 1.0, value: d.lift,
                            oninput: move |v| edit(state, preview, move |b| b.dynamics.lift = v) }
                        if d.lift > 0.005 {
                            div { class: "be-sub",
                                Slider { label: "Pressure \u{2192} Lift", min: 0.0, max: 1.0, value: d.load_pressure,
                                    oninput: move |v| edit(state, preview, move |b| b.dynamics.load_pressure = v) }
                            }
                        }
                        Slider { label: "Deposit", min: 0.0, max: 1.0, value: d.deposit,
                            oninput: move |v| edit(state, preview, move |b| b.dynamics.deposit = v) }
                        if d.deposit > 0.005 {
                            div { class: "be-sub",
                                Slider { label: "Tilt \u{2192} Deposit", min: 0.0, max: 1.0, value: d.deposit_tilt,
                                    oninput: move |v| edit(state, preview, move |b| b.dynamics.deposit_tilt = v) }
                            }
                        }
                        More { open: pickup_more,
                            // The finite glob pre-loaded on the tool (palette knife, §6.2).
                            Slider { label: "Charge", min: 0.0, max: 2.0, value: d.charge,
                                oninput: move |v| edit(state, preview, move |b| b.dynamics.charge = v) }
                        }
                    }

                    Section {
                        title: "Wet flow", desc: "Paint already on the canvas moving under the stroke.",
                        open: flow_open,
                        Slider { label: "Drag", min: 0.0, max: 1.0, value: d.drag,
                            oninput: move |v| edit(state, preview, move |b| b.dynamics.drag = v) }
                        Slider { label: "Bleed", min: 0.0, max: 1.0, value: d.bleed,
                            oninput: move |v| edit(state, preview, move |b| b.dynamics.bleed = v) }
                        Slider { label: "Ridge", min: 0.0, max: 1.0, value: d.ridge,
                            oninput: move |v| edit(state, preview, move |b| b.dynamics.ridge = v) }
                    }

                    Section {
                        title: "Surface", desc: "How the canvas weave gates dry strokes.",
                        open: surface_open,
                        Slider { label: "Tooth", min: 0.0, max: 1.0, value: brush.tooth,
                            oninput: move |v| edit(state, preview, move |b| b.tooth = v) }
                    }
                }
            }
        }
    }
}

// --- grouping chrome ---

/// A collapsible settings group: a chevron header (click toggles) over the body.
#[component]
fn Section(title: String, desc: String, open: Signal<bool>, children: Element) -> Element {
    let mut open = open;
    rsx! {
        div { class: "be-section",
            button {
                class: "be-section-header",
                onclick: move |_| { let v = open(); open.set(!v); },
                span { class: if open() { "be-chevron open" } else { "be-chevron" }, "\u{25B8}" }
                "{title}"
            }
            if open() {
                div { class: "be-section-body",
                    div { class: "be-section-desc", "{desc}" }
                    {children}
                }
            }
        }
    }
}

/// In-section disclosure for rarely-touched knobs: hidden behind "Show more".
#[component]
fn More(open: Signal<bool>, children: Element) -> Element {
    let mut open = open;
    rsx! {
        if open() { {children} }
        button {
            class: "be-more",
            onclick: move |_| { let v = open(); open.set(!v); },
            if open() { "Show less" } else { "Show more\u{2026}" }
        }
    }
}

// --- preview engine ---

/// Build the preview renderer on the shared GPU device, mirror the main canvas's
/// look (surface, environment, lighting, background), import the built-in stamp
/// brush (same content-addressed id as the main engine's), seed the default test
/// stroke, and paint it with the current brush.
async fn init_preview(state: AppState, mut preview: Preview) {
    // Copy everything out of the main renderer before any await (no held borrows).
    let Some((gpu, surface_id, env_id, media, bg)) = state
        .renderer
        .peek()
        .as_ref()
        .map(|r| (r.gpu(), r.surface(), r.environment(), r.media_params(), r.background()))
    else {
        return;
    };

    let mut r = render::init_shared(render::canvas_element(PREVIEW_CANVAS_ID), gpu).await;

    // The asset bytes were all fetched at app startup, so these hit the browser
    // cache; content-addressed ids make the imports line up with the main engine.
    if let Ok(bytes) = dioxus::asset_resolver::read_asset_bytes(crate::BRISTLE_BRUSH).await {
        r.load_bristle(&bytes);
    }
    if let Some(asset) = crate::surface_asset(surface_id)
        && let Ok(bytes) = dioxus::asset_resolver::read_asset_bytes(asset).await
    {
        r.register_surface(surface_id, bytes);
    }
    r.set_surface(surface_id);
    if env_id == EnvironmentId::Ferndale
        && let Ok(bytes) = dioxus::asset_resolver::read_asset_bytes(crate::ENV_FERNDALE).await
    {
        r.register_environment(env_id, bytes);
        r.set_environment(env_id);
    }
    r.set_media_params(media);
    r.set_background(bg);

    // Seed and paint the default test stroke with the current brush.
    let samples = default_stroke(&r);
    if let Some(brush) = state.obs.peek().as_ref().map(|o| o.brush) {
        r.process(InputCommand::SetBrush(brush));
        replay(&mut r, &samples);
        preview.committed.set(true);
    }
    r.paint();
    preview.samples.set(samples);
    preview.renderer.set(Some(r));
}

/// The seeded test stroke: an S-curve across the preview with a pressure bell
/// (light → full → light) and a forward tilt that ramps in — so pressure- and
/// tilt-modulated settings visibly shape the stroke even for mouse users.
fn default_stroke(r: &Renderer) -> Vec<InputSample> {
    let (w, h) = r.size();
    let (w, h) = (w as f32, h as f32);
    let view = r.view();
    const N: usize = 64;
    (0..N)
        .map(|i| {
            let t = i as f32 / (N - 1) as f32;
            let x = w * 0.06 + t * w * 0.88;
            let y = h * 0.5 - (t * std::f32::consts::TAU).sin() * h * 0.26;
            InputSample {
                pos: view.screen_to_canvas(Vec2::new(x, y)),
                pressure: (t * std::f32::consts::PI).sin().clamp(0.08, 1.0),
                // Lean along the (mostly +x) travel direction, growing over the
                // stroke, so tilt→deposit reads as a knife laying down more and more.
                tilt: Vec2::new(0.65 * t, 0.0),
                time: (t * 0.7) as f64,
            }
        })
        .collect()
}

/// Feed a recorded test stroke through the preview engine's normal stroke path.
fn replay(r: &mut Renderer, samples: &[InputSample]) {
    let mut it = samples.iter();
    let Some(first) = it.next() else { return };
    r.process(InputCommand::StartStroke { tool: Tool::Brush, sample: *first });
    for s in it {
        r.process(InputCommand::StrokeTo { sample: *s });
    }
    r.process(InputCommand::EndStroke);
}

/// Re-render the test stroke with the current brush: undo the committed one,
/// push the brush, replay, paint. No-op while the user is drawing on the preview.
fn restroke(state: AppState, mut preview: Preview) {
    if *preview.drawing.peek() {
        return;
    }
    let Some(brush) = state.obs.peek().as_ref().map(|o| o.brush) else {
        return;
    };
    let samples = preview.samples.peek().clone();
    let mut renderer = preview.renderer;
    let mut guard = renderer.write();
    let Some(r) = guard.as_mut() else { return };
    if *preview.committed.peek() {
        r.process(InputCommand::Undo);
    }
    r.process(InputCommand::SetBrush(brush));
    replay(r, &samples);
    r.paint();
    drop(guard);
    preview.committed.set(true);
}

/// Apply a brush edit to the real document brush, then re-stroke the preview —
/// throttled, so a slider drag re-strokes at most every [`RESTROKE_THROTTLE_MS`]
/// instead of once per `input` event. The brush itself always updates
/// immediately; only the preview render is deferred.
///
/// Leading + trailing: an edit while idle re-strokes at once and starts a
/// cooldown; edits during a cooldown just mark it dirty, and the cooldown task
/// re-strokes with the latest brush (repeating until a window passes clean), so
/// the preview always settles on the final slider value.
///
/// Scope invariant: the cooldown task is the only thing that resets `throttle`
/// to `None`, and `spawn` ties it to the scope whose rsx wrote the `oninput`
/// closure. Today that's `BrushEditorModal` itself, which also owns the
/// `Preview` signals — task and state die together on close, which is why a
/// plain `spawn` (not `spawn_forever`) is correct. Don't move the slider rows
/// into a child `#[component]`: the task would then die on a section fold with
/// `throttle` stuck at `Some`, gating all further re-strokes.
fn edit(state: AppState, mut preview: Preview, f: impl FnOnce(&mut BrushParams)) {
    update_brush(state, f);
    if preview.throttle.peek().is_some() {
        preview.throttle.set(Some(true));
        return;
    }
    preview.throttle.set(Some(false));
    restroke(state, preview);
    spawn(async move {
        loop {
            sleep_ms(RESTROKE_THROTTLE_MS).await;
            if *preview.throttle.peek() != Some(true) {
                break;
            }
            preview.throttle.set(Some(false));
            restroke(state, preview);
        }
        preview.throttle.set(None);
    });
}

/// Restore the default test stroke and re-render it.
fn reset_stroke(state: AppState, mut preview: Preview) {
    let samples = match preview.renderer.peek().as_ref() {
        Some(r) => default_stroke(r),
        None => return,
    };
    preview.samples.set(samples);
    restroke(state, preview);
}

// --- drawing a new test stroke on the preview canvas ---

/// Map a pointer event on the preview canvas to a canvas-space input sample
/// (same mapping as the main canvas's `sample`).
fn preview_sample(r: &Renderer, e: &Event<PointerData>) -> InputSample {
    let c = e.element_coordinates();
    InputSample {
        pos: r.view().screen_to_canvas(Vec2::new(c.x as f32, c.y as f32)),
        pressure: e.pressure(),
        tilt: Vec2::new(e.tilt_x() as f32, e.tilt_y() as f32) / 90.0,
        ..Default::default()
    }
}

/// Begin a user test stroke: clear the committed one and start recording.
fn start_preview_stroke(mut preview: Preview, e: &Event<PointerData>) {
    let mut renderer = preview.renderer;
    let mut guard = renderer.write();
    let Some(r) = guard.as_mut() else { return };
    if *preview.committed.peek() {
        r.process(InputCommand::Undo);
        preview.committed.set(false);
    }
    let s = preview_sample(r, e);
    r.process(InputCommand::StartStroke { tool: Tool::Brush, sample: s });
    r.paint();
    drop(guard);
    preview.rec.set(vec![s]);
    preview.drawing.set(true);
}

/// Extend the in-progress user test stroke.
fn move_preview_stroke(mut preview: Preview, e: &Event<PointerData>) {
    let mut renderer = preview.renderer;
    let mut guard = renderer.write();
    let Some(r) = guard.as_mut() else { return };
    let s = preview_sample(r, e);
    r.process(InputCommand::StrokeTo { sample: s });
    r.paint();
    drop(guard);
    preview.rec.write().push(s);
}

/// Commit the user's stroke as the new test stroke.
fn end_preview_stroke(mut preview: Preview) {
    if !*preview.drawing.peek() {
        return;
    }
    let mut renderer = preview.renderer;
    let mut guard = renderer.write();
    if let Some(r) = guard.as_mut() {
        r.process(InputCommand::EndStroke);
        r.paint();
    }
    drop(guard);
    preview.drawing.set(false);
    preview.committed.set(true);
    let rec = preview.rec.peek().clone();
    if !rec.is_empty() {
        preview.samples.set(rec);
    }
}

/// A cancelled pointer aborts the in-progress stroke and restores the last one.
fn cancel_preview_stroke(state: AppState, mut preview: Preview) {
    if !*preview.drawing.peek() {
        return;
    }
    let mut renderer = preview.renderer;
    if let Some(r) = renderer.write().as_mut() {
        r.process(InputCommand::CancelStroke);
    }
    preview.drawing.set(false);
    restroke(state, preview);
}
