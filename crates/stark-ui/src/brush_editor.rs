//! The brush editor pop-up (§11): a Procreate-style dialog with a live
//! test-stroke preview over grouped settings.
//!
//! The preview is a second `Engine` on its **own document** but the **shared GPU
//! device** ([`render::init_shared`]), mirroring the main canvas's surface,
//! environment, lighting, and background so a stroke reads exactly like it will
//! on the real canvas. One test stroke — a seeded default with a pressure bell
//! and a ramping forward tilt (so pressure/tilt-driven settings respond even
//! with a mouse), or whatever the user last drew on the preview — is re-stroked
//! (undo → set brush → replay → paint) as settings change. Slider edits are
//! throttled to one apply per [`EDIT_THROTTLE_MS`], and the replay commits the
//! whole stroke as a single render (`Engine::replay_stroke` — no per-sample
//! live-preview refresh, which would be O(n²) and starve the GPU), so drags
//! stay responsive and the finished stroke appears in one go.
//!
//! Settings are grouped into collapsible sections by what they affect, with
//! rarely-used knobs behind a per-section "Show more". Every slider here drives a
//! parameter the engine actually reads — a knob that moves but changes nothing is
//! worse than no knob, and a knob the engine reads but that hides behind "Show
//! more" (as `drain` did) may as well not exist.

use dioxus::html::input_data::MouseButton;
use dioxus::prelude::*;

use stark_core::document::{
    BrushParams, BrushShape, ModSource, Modulation, Modulations, NoiseKind, OrientationSource,
    PenState, Tool,
};
use stark_core::geom::Vec2;
use stark_core::{ColorSpaceId, InputSample, SurfaceId};

use dioxus::html::HasFileData;

use crate::icons::{self, icon};
use crate::panels::brush::{MAX_RADIUS, MAX_TAPER, set_orientation, set_shape};
use crate::platform::{capture_pointer, pick_file, sleep_ms};
use crate::render::{self, Renderer};
use crate::state::{AppState, update_brush};
use crate::widgets::Slider;
use stark_core::command::{DocCommand, GestureCommand, ViewCommand};

/// The preview `<canvas>`'s DOM id (the main canvas is `render::CANVAS_ID`).
const PREVIEW_CANVAS_ID: &str = "brush-preview-canvas";

/// The test stroke's fixed RGB (straight sRGB): a pleasant blue, so it reads
/// clearly over the red reference stroke beneath it — the preview is about the
/// brush's *behaviour*, not its colour. Only RGB is forced; the brush's own
/// alpha (the Opacity slider) still applies.
const PREVIEW_STROKE_COLOR: [f32; 3] = [0.852, 0.645, 0.125];

/// Fixed jitter seed for the previewed test stroke. Every edit re-strokes, and
/// a stroke's seed is normally the document clock — which advances with each
/// replay's commit, re-rolling the colour dynamics and dither each time and
/// hiding the parameter change behind fresh noise. Pinning it means only the
/// edited setting moves between renders. Arbitrary value; it just never changes.
const PREVIEW_STROKE_SEED: u64 = 0x5747_1CED_57A2_4B11;

/// Minimum gap between slider edits taking effect. Each edit dispatches to the
/// engine, repaints the main canvas, refreshes `obs` (re-rendering this whole
/// dialog and every other `obs` reader), and replays the test stroke (~a
/// frame's worth of GPU) — fine at this rate, not at slider `input`-event
/// rate. The slider thumb itself is a native range input and keeps moving
/// smoothly between commits.
const EDIT_THROTTLE_MS: i32 = 50;

/// A deferred brush mutation: the latest slider edit during a throttle window.
type BrushEdit = Box<dyn FnOnce(&mut BrushParams)>;

/// The parameters the pen can drive (§6.2) — one variant per field of
/// [`Modulations`], and the addressing for the one open mapping row.
///
/// It carries everything about a row that differs: its word, its range, where its
/// base value lives on the brush, and which mapping slot belongs to it. That is what
/// lets [`mod_slider`] take a `ModRow` and nothing else, and it is why the rows
/// cannot drift out of step with the engine's set — adding a target to
/// `Modulations` and not here fails to compile at [`Self::slot`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ModRow {
    Size,
    Flow,
    Tooth,
    Lift,
    Deposit,
    Bleed,
}

impl ModRow {
    /// The word on the row, which is also the word the section already used for the
    /// parameter — "Flow" for `add`, not "Add".
    fn label(self) -> &'static str {
        match self {
            Self::Size => "Size",
            Self::Flow => "Flow",
            Self::Tooth => "Tooth",
            Self::Lift => "Lift",
            Self::Deposit => "Deposit",
            Self::Bleed => "Bleed",
        }
    }

    /// The base slider's range. The three pickup axes stop at 0.95 for the reason
    /// they always did — λ diverges at 1 (§6.2).
    fn range(self) -> (f32, f32) {
        match self {
            Self::Size => (1.0, MAX_RADIUS),
            Self::Flow => (0.0, 3.0),
            Self::Tooth => (0.0, 1.0),
            Self::Lift | Self::Deposit | Self::Bleed => (0.0, 0.95),
        }
    }

    fn get(self, b: &BrushParams) -> f32 {
        match self {
            Self::Size => b.radius,
            Self::Flow => b.dynamics.add,
            Self::Tooth => b.tooth,
            Self::Lift => b.dynamics.lift,
            Self::Deposit => b.dynamics.deposit,
            Self::Bleed => b.dynamics.bleed,
        }
    }

    fn set(self, b: &mut BrushParams, v: f32) {
        match self {
            Self::Size => b.radius = v,
            Self::Flow => b.dynamics.add = v,
            Self::Tooth => b.tooth = v,
            Self::Lift => b.dynamics.lift = v,
            Self::Deposit => b.dynamics.deposit = v,
            Self::Bleed => b.dynamics.bleed = v,
        }
    }

    fn slot(self, m: &mut Modulations) -> &mut Option<Modulation> {
        match self {
            Self::Size => &mut m.size,
            Self::Flow => &mut m.flow,
            Self::Tooth => &mut m.tooth,
            Self::Lift => &mut m.lift,
            Self::Deposit => &mut m.deposit,
            Self::Bleed => &mut m.bleed,
        }
    }

    fn of(self, m: &Modulations) -> Option<Modulation> {
        match self {
            Self::Size => m.size,
            Self::Flow => m.flow,
            Self::Tooth => m.tooth,
            Self::Lift => m.lift,
            Self::Deposit => m.deposit,
            Self::Bleed => m.bleed,
        }
    }
}

/// The class a two-state chip wears. Shared by every chip row in the dialog, so a
/// selected shape, a selected noise kind and a selected pen source all light the
/// same way.
fn chip(active: bool) -> &'static str {
    if active { "chip active" } else { "chip" }
}

/// The word a source wears on its chip.
fn source_label(s: ModSource) -> &'static str {
    match s {
        ModSource::Pressure => "Pressure",
        ModSource::Tilt => "Tilt",
    }
}

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
    /// Edit throttle gate: whether the post-edit cooldown is running.
    cooling: Signal<bool>,
    /// The latest edit deferred during the cooldown, owed a trailing apply.
    pending: Signal<Option<BrushEdit>>,
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
        cooling: use_signal(|| false),
        pending: use_signal(|| None),
    };

    // If the dialog closes mid-cooldown the throttle task is cancelled with it;
    // commit any deferred edit so the tail of a drag isn't lost.
    use_drop(move || {
        let mut pending = preview.pending;
        if let Some(f) = pending.write().take() {
            update_brush(state, f);
        }
    });

    // Re-stroke whenever the brush *shape* changes, wherever from — a gallery
    // click, an async import finishing after its click handler returned, the
    // quick panel, a preset. The memo keeps this from firing on unrelated
    // `obs` churn (sliders restroke through `edit`'s throttle instead), and
    // the initial run is a no-op because the preview renderer isn't up yet.
    let shape = use_memo(move || state.obs.read().as_ref().map(|o| o.brush.shape));
    use_effect(move || {
        let _ = shape();
        restroke(state, preview);
    });

    // Section fold state: the everyday groups start open, the specialised ones closed.
    let tip_open = use_signal(|| true);
    let paint_open = use_signal(|| true);
    let color_open = use_signal(|| false);
    let pickup_open = use_signal(|| false);
    let _surface_open = use_signal(|| false);
    // Per-section "Show more" for the rarely-touched knobs.
    let pickup_more = use_signal(|| false);
    // Which parameter's pen mapping is open, at most one at a time — so the dialog
    // grows by one sub-row while a mapping is being edited and by nothing otherwise
    // (see [`mod_slider`]). Held here rather than per row because the rows are plain
    // calls, not components, and because "one at a time" is the behaviour wanted.
    let mod_open = use_signal(|| None::<ModRow>);

    let brush = state
        .obs
        .read()
        .as_ref()
        .map(|o| o.brush)
        .unwrap_or_default();
    let is_round = matches!(brush.shape, BrushShape::Round { .. });
    let d = brush.dynamics;
    let cd = brush.color_dynamics;
    // The jitter channels are the *colour space's* channels — label them for
    // whatever space the document is in.
    let space = state
        .renderer
        .read()
        .as_ref()
        .map(|r| r.color_space())
        .unwrap_or(ColorSpaceId::Oklab);
    // The ground the document is on (§6.4) — what the tooth has to bite into.
    let surface = state
        .renderer
        .read()
        .as_ref()
        .map(|r| r.surface())
        .unwrap_or_default();
    let ch_labels = match space {
        ColorSpaceId::Mixbox => ["Pigment 1", "Pigment 2", "Pigment 3"],
        _ => ["Lightness", "Green \u{2194} red", "Blue \u{2194} yellow"],
    };

    rsx! {
        div {
            class: "modal-backdrop",
            onclick: move |_| on_close.call(()),
            div {
                class: "modal-dialog be-dialog",
                onclick: move |e| e.stop_propagation(),

                div { class: "be-header",
                    div { class: "modal-title", "Brush" }
                    button { class: "btn btn-primary", onclick: move |_| on_close.call(()),
                        {icon(icons::DONE)}
                        "Done"
                    }
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
                    // The turning arrow Undo wears, because putting the preview back
                    // *is* an undo — narrowed to the one stroke this dialog owns
                    // (`icons::RESET`).
                    button {
                        class: "be-preview-reset",
                        title: "Restore the default test stroke",
                        onclick: move |_| reset_stroke(state, preview),
                        {icon(icons::RESET)}
                    }
                }

                div { class: "be-sections",
                    Section {
                        title: "Tip", desc: "The footprint the stroke sweeps along the path.",
                        glyph: icons::TIP,
                        open: tip_open,
                        ShapeGallery {}
                        // Orientation only matters for non-round tips (per-orientation
                        // footprint slices, §6.6); hardness only for the
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
                        {mod_slider(state, preview, mod_open, ModRow::Size, brush)}
                        if let BrushShape::Round { hardness } = brush.shape {
                            Slider { label: "Hardness", min: 0.0, max: 1.0, value: hardness,
                                oninput: move |v| edit(state, preview, move |b| {
                                    if let BrushShape::Round { hardness } = &mut b.shape {
                                        *hardness = v;
                                    }
                                }) }
                        }
                        // The two tapers — the run over which the tip widens from a
                        // point (§6.2). In *radii*, so a taper keeps its shape as the
                        // brush is resized, which is why the labels say so.
                        Slider { label: "Start taper (radii)", min: 0.0, max: MAX_TAPER, value: brush.start_taper_length,
                            oninput: move |v| edit(state, preview, move |b| b.start_taper_length = v) }
                        Slider { label: "End taper (radii)", min: 0.0, max: MAX_TAPER, value: brush.end_taper_length,
                            oninput: move |v| edit(state, preview, move |b| b.end_taper_length = v) }
                    }

                    Section {
                        title: "Paint", desc: "The brush's own paint: how much goes down and how far it lasts.",
                        glyph: icons::PAINT,
                        open: paint_open,
                        // `add` is the tool's only source term (§6.2) and its only amount
                        // knob: the paint height laid per unit swept optical depth.
                        {mod_slider(state, preview, mod_open, ModRow::Flow, brush)}
                        // How deeply the tip bites into the canvas's own tooth
                        // (§6.4): at 0 the ground is irrelevant and the mark is solid,
                        // and turned up the paint catches on the weave's peaks and
                        // skips its valleys, which is what a dry brush leaves.
                        {mod_slider(state, preview, mod_open, ModRow::Tooth, brush)}
                        // The ground is the *document's*, not the brush's — a pencil
                        // and a loaded brush on one canvas see one tooth — so on a
                        // smooth canvas this knob has nothing to bite and says so,
                        // rather than moving and changing nothing.
                        if brush.tooth > 0.0 && surface == SurfaceId::Flat {
                            div { class: "be-note",
                                "This canvas is smooth, so there is no tooth to catch on. \
                                 Pick a surface in the Lighting panel."
                            }
                        }
                        // Per-unit opacity, independent of the amount laid (§6.1).
                        // The ghost the Layers panel's opacity wears: the same question
                        // — how much of what is under this shows through — asked of the
                        // paint a stroke lays rather than of a whole layer.
                        Slider { label: "Opacity", glyph: icons::OPACITY, min: 0.0, max: 1.0, value: brush.color[3],
                            oninput: move |v| edit(state, preview, move |b| b.color[3] = v) }
                        // Depletion per px travelled — the stroke runs dry. 0 is what a
                        // pen or a digital brush wants; not behind "Show more", because
                        // it is the only knob that decides whether a tool runs out.
                        Slider { label: "Drain", min: 0.0, max: 0.01, value: brush.drain,
                            oninput: move |v| edit(state, preview, move |b| b.drain = v) }
                    }

                    Section {
                        title: "Color dynamics", desc: "The colour wanders across the brush and along the stroke, following a noise field.",
                        glyph: icons::COLOR,
                        open: color_open,
                        div { class: "brush-shapes",
                            button { class: chip(cd.noise == NoiseKind::Simplex),
                                onclick: move |_| edit(state, preview, |b| b.color_dynamics.noise = NoiseKind::Simplex),
                                "Simplex" }
                            button { class: chip(cd.noise == NoiseKind::White),
                                onclick: move |_| edit(state, preview, |b| b.color_dynamics.noise = NoiseKind::White),
                                "White" }
                            button { class: chip(cd.noise == NoiseKind::Voronoi),
                                onclick: move |_| edit(state, preview, |b| b.color_dynamics.noise = NoiseKind::Voronoi),
                                "Voronoi" }
                            button { class: chip(cd.noise == NoiseKind::Mosaic),
                                onclick: move |_| edit(state, preview, |b| b.color_dynamics.noise = NoiseKind::Mosaic),
                                "Mosaic" }
                        }
                        // How far each colour channel wanders (± in the channel's units).
                        for i in 0..3 {
                            Slider { label: ch_labels[i].to_string(), min: 0.0, max: 0.5, value: cd.amplitude[i],
                                oninput: move |v| edit(state, preview, move |b| b.color_dynamics.amplitude[i] = v) }
                        }
                        // How fast it wanders along each lookup axis; the modulation
                        // sliders live only while some channel is active (no effect at 0).
                        if cd.amplitude.iter().any(|a| *a > 0.0) {
                            div { class: "be-sub",
                                Slider { label: "Scale \u{2192} across stroke", min: 0.0, max: 8.0, value: cd.frequency[0],
                                    oninput: move |v| edit(state, preview, move |b| b.color_dynamics.frequency[0] = v) }
                                Slider { label: "Scale \u{2192} along stroke", min: 0.0, max: 8.0, value: cd.frequency[1],
                                    oninput: move |v| edit(state, preview, move |b| b.color_dynamics.frequency[1] = v) }
                            }
                        }
                    }

                    Section {
                        title: "Pickup", desc: "Canvas paint on the move — smudge, knife, eraser, blur.",
                        glyph: icons::PICKUP,
                        open: pickup_open,
                        // The three axes a palette knife is built out of, and the three
                        // most worth mapping onto the pen: a knife that lifts with
                        // pressure and lays back with tilt is those two chips
                        // (§6.2).
                        {mod_slider(state, preview, mod_open, ModRow::Lift, brush)}
                        {mod_slider(state, preview, mod_open, ModRow::Deposit, brush)}
                        // The lateral axis: the paint under the tip diffuses towards its
                        // neighbours (§6.2). Alone it is a blur brush; under `add` it
                        // melts the ridges of the strokes being painted over. Capped at
                        // 0.95 like the two vertical rates — the λ diverges at 1.
                        {mod_slider(state, preview, mod_open, ModRow::Bleed, brush)}
                        More { open: pickup_more,
                            // The finite glob pre-loaded on the tool (palette knife, §6.2).
                            Slider { label: "Charge", min: 0.0, max: 2.0, value: d.charge,
                                oninput: move |v| edit(state, preview, move |b| b.dynamics.charge = v) }
                        }
                    }

                }
            }
        }
    }
}

/// A parameter slider with its **pen mapping** hung off the end (§6.2): the base
/// slider exactly as it was, plus a chip naming what drives it, which opens the
/// mapping's own controls in place.
///
/// The chip is the whole design decision. A brush with no mapping on this row reads
/// as one word and one track, as it always did; the mapping is a second line only
/// while it is being edited, and only ever one row at a time (`open` holds a single
/// [`ModRow`]). The alternative — a "pen response" section listing every target —
/// puts the control a fold away from the parameter it drives, so reading a brush
/// means holding two lists against each other.
///
/// **A plain function, not a `#[component]`.** The sub-row's own sliders write their
/// `oninput` closures wherever this is expanded, and [`edit`]'s throttle task is
/// `spawn`ed into that scope. A child component would be unmounted by a section fold
/// or by the chip closing, killing the task with `cooling` stuck at `true` and gating
/// every later edit — the same hazard [`edit`]'s own note describes. Called inline it
/// runs in `BrushEditorModal`'s scope, which is where the state it touches lives.
fn mod_slider(
    state: AppState,
    preview: Preview,
    open: Signal<Option<ModRow>>,
    row: ModRow,
    brush: BrushParams,
) -> Element {
    let mut open = open;
    let (min, max) = row.range();
    let m = row.of(&brush.modulation);
    let expanded = open() == Some(row);
    // The chip says what is driving the parameter, which is the one thing a glance
    // needs; unmapped it is a bare "+", the same invitation it wears everywhere else.
    let chip_class = if m.is_some() {
        "mod-chip active"
    } else {
        "mod-chip"
    };

    rsx! {
        div { class: "mod-slider",
            Slider { label: row.label().to_string(), min, max, value: row.get(&brush),
                oninput: move |v| edit(state, preview, move |b| row.set(b, v)) }
            button {
                class: chip_class,
                title: "What the pen drives this with",
                onclick: move |_| open.set(if expanded { None } else { Some(row) }),
                match m {
                    Some(m) => rsx! { "{source_label(m.source)}" },
                    None => rsx! { {icon(icons::MODULATE)} },
                }
            }
        }
        if expanded {
            div { class: "be-sub mod-panel",
                div { class: "brush-shapes",
                    button { class: chip(m.is_none()),
                        onclick: move |_| set_source(state, preview, row, None),
                        "Off" }
                    for src in [ModSource::Pressure, ModSource::Tilt] {
                        button {
                            key: "{source_label(src)}",
                            class: chip(m.is_some_and(|m| m.source == src)),
                            onclick: move |_| set_source(state, preview, row, Some(src)),
                            "{source_label(src)}"
                        }
                    }
                }
                if let Some(m) = m {
                    // The two shape knobs, and the curve they describe drawn beside
                    // them — the factor is what actually reaches the renderer, and a
                    // number pair for a curve is the one thing a picture reads better.
                    div { class: "mod-shape",
                        div { class: "mod-shape-knobs",
                            // What survives a feather touch (or an upright pen). This
                            // is what makes a tilt-driven brush usable with a mouse,
                            // which reports no tilt at all.
                            Slider { label: "At zero".to_string(), min: 0.0, max: 1.0, value: m.floor,
                                oninput: move |v| edit(state, preview, move |b| {
                                    if let Some(m) = row.slot(&mut b.modulation) { m.floor = v; }
                                }) }
                            // Negative = late, positive = early; 0 is linear,
                            // exactly. Unlabelled at its ends because the plot
                            // beside it moves as the knob does, which says which
                            // way is which better than two words either side of a
                            // track this narrow.
                            Slider { label: "Response".to_string(),
                                min: -1.0, max: 1.0, value: m.curve,
                                oninput: move |v| edit(state, preview, move |b| {
                                    if let Some(m) = row.slot(&mut b.modulation) { m.curve = v; }
                                }) }
                        }
                        {curve_plot(m)}
                    }
                }
            }
        }
    }
}

/// Set (or clear) a row's mapping source, keeping the shape it already had — so
/// switching pressure → tilt is one edit rather than three.
fn set_source(state: AppState, preview: Preview, row: ModRow, source: Option<ModSource>) {
    edit(state, preview, move |b| {
        let held = row.of(&b.modulation);
        *row.slot(&mut b.modulation) = source.map(|source| Modulation {
            source,
            ..held.unwrap_or(Modulation::linear(source))
        });
    });
}

/// The mapping's response drawn as a curve: input left → right, the factor it
/// multiplies the parameter by bottom → top.
///
/// Sampled from [`Modulation::factor`] itself rather than redrawn from the formula,
/// so the picture cannot disagree with the renderer — including about the floor and
/// about the clamps. Both sources are fed the same sweep, which is what makes one
/// plot serve either.
fn curve_plot(m: Modulation) -> Element {
    const N: usize = 25;
    const W: f32 = 56.0;
    const H: f32 = 30.0;
    let pad = 1.5;
    let pts: String = (0..N)
        .map(|i| {
            let x = i as f32 / (N - 1) as f32;
            let f = m.factor(PenState {
                pressure: x,
                tilt: x,
            });
            let px = pad + x * (W - 2.0 * pad);
            let py = H - pad - f * (H - 2.0 * pad);
            format!("{px:.2},{py:.2} ")
        })
        .collect();
    rsx! {
        svg {
            class: "mod-curve",
            view_box: "0 0 {W} {H}",
            width: "{W}",
            height: "{H}",
            polyline {
                points: "{pts}",
                fill: "none",
                stroke: "currentColor",
                stroke_width: "1.5",
                stroke_linejoin: "round",
            }
        }
    }
}

/// The Tip section's shape gallery: the procedural round tip, every shape
/// bundled with the app (`crate::builtins`), every shape in the user's library
/// (thumbnail + name, with a hover ✕ to remove), and an import card. Images can
/// also be dropped anywhere on the grid.
///
/// No restroke calls here: selection and import go through the brush's `shape`,
/// and the modal's shape effect re-strokes on any change — which is what lets
/// an async import (finishing long after its click handler returned) still
/// refresh the preview. Safe as a child component (unlike the slider rows):
/// nothing here spawns into this scope — imports are `spawn_forever` in
/// `crate::shapes`.
#[component]
fn ShapeGallery() -> Element {
    let state = use_context::<AppState>();
    let mut dropping = use_signal(|| false);

    let brush_shape = state
        .obs
        .read()
        .as_ref()
        .map(|o| o.brush.shape)
        .unwrap_or_default();
    // One card per bundled shape, in table order. A built-in whose fetch is
    // still in flight has no id yet, so it simply never reads as selected —
    // clicking it is the same no-op, and both settle when the bytes land.
    let builtins: Vec<(&'static str, String, bool)> = crate::builtins::resolved(state)
        .into_iter()
        .map(|(builtin, id)| {
            let active = matches!(brush_shape, BrushShape::Stamp(s) if Some(s) == id);
            (builtin.name, builtin.asset.to_string(), active)
        })
        .collect();
    let entries = state.shapes.entries;
    // Thumbnails are base64 data URLs of the canonical PNGs — memoized so they
    // re-encode only when the library changes, not on every obs refresh.
    let thumbs = use_memo(move || {
        entries
            .read()
            .iter()
            .map(|e| {
                (
                    e.id,
                    crate::shapes::id_hex(e.id),
                    e.name.clone(),
                    crate::shapes::data_url(e),
                )
            })
            .collect::<Vec<_>>()
    });

    let card = |active: bool| {
        if active {
            "shape-card selected"
        } else {
            "shape-card"
        }
    };
    let is_round = matches!(brush_shape, BrushShape::Round { .. });

    rsx! {
        div {
            class: if dropping() { "be-shapes dropping" } else { "be-shapes" },
            // `preventDefault` on dragover is what makes the element a drop
            // target at all; the class is just the highlight.
            ondragover: move |e| {
                e.prevent_default();
                dropping.set(true);
            },
            ondragleave: move |_| dropping.set(false),
            ondrop: move |e| {
                e.prevent_default();
                dropping.set(false);
                crate::shapes::import_dropped(state, e.files());
            },

            div { class: card(is_round),
                onclick: move |_| set_shape(state, BrushShape::default()),
                div { class: "shape-thumb round" }
                div { class: "shape-name", "Round" }
            }
            for (name, url, active) in builtins {
                div {
                    key: "{name}",
                    class: card(active),
                    onclick: move |_| crate::builtins::select(state, name),
                    div { class: "shape-thumb", style: "background-image: url({url});" }
                    div { class: "shape-name", title: "{name}", "{name}" }
                }
            }
            for (id, key, name, url) in thumbs() {
                div {
                    key: "{key}",
                    class: card(brush_shape == BrushShape::Stamp(id)),
                    onclick: move |_| crate::shapes::select(state, id),
                    div { class: "shape-thumb", style: "background-image: url({url});" }
                    div { class: "shape-name", title: "{name}", "{name}" }
                    // `icons::REMOVE`, as on every other row the application lets you
                    // take something out of — the library of stamps is one more roster.
                    button {
                        class: "shape-remove",
                        title: "Remove from library",
                        onclick: move |e| {
                            e.stop_propagation();
                            crate::shapes::remove(state, id);
                        },
                        {icon(icons::REMOVE)}
                    }
                }
            }
            div { class: "shape-card import",
                // `pick_file` must run inside the click gesture — no task hop.
                onclick: move |_| {
                    pick_file("image/*", move |name, bytes| {
                        crate::shapes::import_file(state, name, bytes);
                    });
                },
                div { class: "shape-thumb plus", {icon(icons::ADD)} }
                div { class: "shape-name", "Import\u{2026}" }
            }
        }
        if let Some(notice) = (state.shapes.notice)() {
            div { class: "be-shape-notice", "{notice}" }
        }
        div { class: "be-shape-hint",
            "Import any image or drop one on the grid — white paints, black doesn't, transparency counts."
        }
    }
}

// --- grouping chrome ---

/// A collapsible settings group: a chevron header (click toggles) over the body.
///
/// `glyph` says what the group is *about* — the same job the `desc` sentence does,
/// except that the sentence is inside the fold and the mark is not. A shut section
/// is a word on a line, and four words in a column are read one at a time; four
/// marks are read at once, which is what makes the dialog navigable while collapsed.
///
/// The chevron beside it stays a character on purpose. It is the one mark here whose
/// meaning is its *rotation*, and the set has no right-pointing caret to rotate —
/// borrowing the Layers panel's would put the glyph that deliberately refuses to
/// rotate (`icons::FOLD_OPEN`) into a control that must.
#[component]
fn Section(
    title: String,
    desc: String,
    glyph: &'static str,
    open: Signal<bool>,
    children: Element,
) -> Element {
    let mut open = open;
    rsx! {
        div { class: "be-section",
            button {
                class: "be-section-header",
                onclick: move |_| { let v = open(); open.set(!v); },
                span { class: if open() { "be-chevron open" } else { "be-chevron" }, "\u{25B8}" }
                {icon(glyph)}
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
/// look (surface, environment, lighting, background), import the bundled stamp
/// shapes (same content-addressed ids as the main engine's), seed the default test
/// stroke, and paint it with the current brush.
async fn init_preview(state: AppState, mut preview: Preview) {
    // Copy everything out of the main renderer before any await (no held borrows).
    let Some((gpu, surface_id, env_id, media, bg)) = state.renderer.peek().as_ref().map(|r| {
        (
            r.gpu(),
            r.surface(),
            r.environment(),
            r.media_params(),
            r.observe().background,
        )
    }) else {
        return;
    };

    let mut r = render::init_shared(render::canvas_element(PREVIEW_CANVAS_ID), gpu).await;

    // The asset bytes were all fetched at app startup, so these hit the browser
    // cache; content-addressed ids make the imports line up with the main engine.
    crate::builtins::import_all(&mut r).await;
    if let Some(asset) = crate::panels::lighting::surface_asset(surface_id)
        && let Ok(bytes) = dioxus::asset_resolver::read_asset_bytes(asset).await
    {
        r.register_surface(surface_id, bytes);
    }
    r.set_surface(surface_id);
    if let Some(asset) = crate::panels::lighting::environment_asset(env_id)
        && let Ok(bytes) = dioxus::asset_resolver::read_asset_bytes(asset).await
    {
        r.register_environment(env_id, bytes);
    }
    r.set_environment(env_id);
    r.set_media_params(media);
    r.process(DocCommand::SetBackground(bg));

    // Lay the fixed red reference stroke: committed once, beneath the
    // replayable test stroke, so the user can preview how the brush interacts
    // with paint already on the canvas. `restroke`'s single undo never reaches
    // it (it only ever removes the test stroke committed on top).
    paint_reference_stroke(&mut r);

    // Seed the default test stroke and render it with the current brush.
    r.paint();
    preview.samples.set(default_stroke(&r));
    preview.renderer.set(Some(r));
    restroke(state, preview);
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

/// The fixed reference stroke laid on the preview canvas before any test
/// stroke: a simple, hard-edged, opaque **red vertical** line down the middle,
/// committed once at init so the user can see how the brush being edited
/// interacts with paint already on the canvas (smudge, drag, bleed, …). Plain
/// `add` paint, no dynamics, no drain — a clean, unchanging target.
fn paint_reference_stroke(r: &mut Renderer) {
    let (w, h) = r.size();
    let (w, h) = (w as f32, h as f32);
    let view = r.view();
    let x = w * 0.5;
    const N: usize = 8;
    let samples: Vec<InputSample> = (0..N)
        .map(|i| {
            let t = i as f32 / (N - 1) as f32;
            let y = h * -0.5 + t * h * 1.5;
            InputSample {
                pos: view.screen_to_canvas(Vec2::new(x, y)),
                pressure: 1.0,
                ..Default::default()
            }
        })
        .collect();
    let brush = BrushParams {
        color: [0.82, 0.15, 0.12, 1.0],
        radius: 75.0,
        shape: BrushShape::Round { hardness: 0.9 },
        drain: 0.0,
        ..BrushParams::default()
    };
    r.process(ViewCommand::SetBrush(brush));
    r.replay_stroke(Tool::Brush, &samples);
}

/// Re-render the test stroke with the current brush: undo the committed one,
/// push the brush, replay as a single commit, paint. `Renderer::replay_stroke`
/// skips the per-sample live-preview refresh (O(n²) across a replay), so the
/// whole re-stroke is one full-stroke render — about a frame's worth of GPU —
/// and the finished stroke is presented in one go, no progressive redraw.
/// The replay is seeded with [`PREVIEW_STROKE_SEED`] so the jitter stays put
/// across edits and only the changed parameter moves.
/// No-op while the user is drawing on the preview.
fn restroke(state: AppState, mut preview: Preview) {
    if *preview.drawing.peek() {
        return;
    }
    let Some(mut brush) = state.obs.peek().as_ref().map(|o| o.brush) else {
        return;
    };
    // A custom shape selected on the main engine may be missing from the
    // preview engine (its store is per-document). Copy the canonical bytes
    // over — content-addressing lines the ids up, the same trick the preview's
    // built-in imports rely on.
    let shape_bytes = match brush.shape {
        BrushShape::Stamp(id) => state
            .renderer
            .peek()
            .as_ref()
            .and_then(|main| main.asset_bytes(id))
            .map(|bytes| (id, bytes)),
        BrushShape::Round { .. } => None,
    };
    // Force the test stroke to the fixed preview blue so it reads over the red
    // reference stroke; the brush's own alpha (Opacity) is left untouched.
    brush.color[..3].copy_from_slice(&PREVIEW_STROKE_COLOR);
    let samples = preview.samples.peek().clone();
    let mut renderer = preview.renderer;
    let mut guard = renderer.write();
    let Some(r) = guard.as_mut() else { return };
    if let Some((id, bytes)) = shape_bytes
        && !r.has_asset(id)
    {
        r.import_brush(&bytes);
    }
    if *preview.committed.peek() {
        r.process(DocCommand::Undo);
    }
    r.process(ViewCommand::SetBrush(brush));
    r.replay_stroke_seeded(Tool::Brush, &samples, PREVIEW_STROKE_SEED);
    r.paint();
    drop(guard);
    preview.committed.set(true);
}

/// Apply a brush edit to the real document brush and re-stroke the preview —
/// throttled to one apply per [`EDIT_THROTTLE_MS`] instead of one per `input`
/// event, since the apply itself (engine dispatch, main-canvas repaint, `obs`
/// refresh re-rendering the dialog, preview re-stroke) is what makes an
/// unthrottled drag choppy. The slider thumb is a native range input, so it
/// keeps moving smoothly between commits.
///
/// Leading + trailing: an edit while idle applies at once and starts a
/// cooldown; edits during a cooldown are deferred (latest wins — slider edits
/// set absolute values, and one pointer drags one slider at a time), and the
/// cooldown task applies them each window until one passes clean, so the brush
/// always settles on the final slider value.
///
/// Scope invariant: the cooldown task is the only thing that resets `cooling`
/// to `false`, and `spawn` ties it to the scope whose rsx wrote the `oninput`
/// closure. Today that's `BrushEditorModal` itself, which also owns the
/// `Preview` signals — task and state die together on close (the `use_drop`
/// there flushes a still-pending edit), which is why a plain `spawn` (not
/// `spawn_forever`) is correct. Don't move the slider rows into a child
/// `#[component]`: the task would then die on a section fold with `cooling`
/// stuck at `true`, gating all further edits.
fn edit(state: AppState, mut preview: Preview, f: impl FnOnce(&mut BrushParams) + 'static) {
    if *preview.cooling.peek() {
        preview.pending.set(Some(Box::new(f)));
        return;
    }
    preview.cooling.set(true);
    update_brush(state, f);
    restroke(state, preview);
    spawn(async move {
        loop {
            sleep_ms(EDIT_THROTTLE_MS).await;
            let Some(f) = preview.pending.write().take() else {
                break;
            };
            update_brush(state, f);
            restroke(state, preview);
        }
        preview.cooling.set(false);
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
        r.process(DocCommand::Undo);
        preview.committed.set(false);
    }
    let s = preview_sample(r, e);
    r.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: s,
        tolerance: crate::input::input_tolerance_in(r.view(), e),
    });
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
    r.process(GestureCommand::To { sample: s });
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
        r.process(GestureCommand::End);
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
        r.process(GestureCommand::Cancel);
    }
    preview.drawing.set(false);
    restroke(state, preview);
}
