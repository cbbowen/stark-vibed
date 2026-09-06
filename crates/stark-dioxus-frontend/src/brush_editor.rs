//! The brush editor pop-up (§11): a Procreate-style dialog with a live
//! test-stroke preview beside grouped settings.
//!
//! The preview is the dialog's **right column**, full height, with the stroke
//! running down it. A stroke is long and thin, so the tall column is the shape
//! that fits it — it buys a longer run of paint than a letterbox strip of the
//! same area would, and it leaves the settings a full-width column of their own
//! rather than one shared with nothing. Being full height, it is resized by
//! anything that changes the dialog's height (a section folding, a window
//! resize), which [`resize_preview`] follows.
//!
//! The preview is a second `Engine` on its **own document**, built by **sharing
//! the main engine's state** ([`Renderer::shared`]): same device, same compiled
//! pipelines, same imported shapes, and it opens on the canvas's substrate under its
//! lighting — so a stroke reads exactly like it will on the real canvas, and the
//! dialog opens without fetching or decoding anything. One test stroke — a seeded default with a pressure bell
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
use stark_engine::command::Tool;
use stark_ui::icons::Icon;

use stark_engine::command::InputSample;
use stark_model::ColorSpaceId;
use stark_model::SubstrateId;
use stark_model::document::{
    BrushEffect, BrushParams, BrushShape, ModSource, Modulation, NoiseKind, OrientationSource,
    PenState,
};
use stark_model::geom::Vec2;

use dioxus::html::HasFileData;

use crate::commands;
use crate::icons::icon;
use crate::panels::brush::{MAX_TAPER, MAX_TOOTH_SOFTNESS, set_orientation, set_shape};
use crate::platform::{capture_pointer, pick_file, sleep_ms};
use crate::presets;
use crate::render::Renderer;
use crate::state::{AppState, update_brush};
use crate::widgets::{Modal, Slider};
use stark_engine::command::{DocCommand, GestureCommand, ViewCommand};
use stark_ui::brush_config::{BrushConfig, BrushEffectType, Transient};
use stark_ui::brush_config::{MAX_RADIUS, MIN_RADIUS};
use stark_ui::commands::Command;

/// The preview `<canvas>`'s DOM id (the main canvas is `render::CANVAS_ID`).
const PREVIEW_CANVAS_ID: &str = "brush-preview-canvas";

/// The test stroke's fixed RGB (straight sRGB): a pleasant blue, so it reads
/// clearly over the red reference stroke beneath it — the preview is about the
/// brush's *behaviour*, not its color. Only the color is forced; the effect's
/// own opacity (the Opacity slider) still applies.
const PREVIEW_STROKE_COLOR: [f32; 3] = [0.852, 0.645, 0.125];

/// Fixed jitter seed for the previewed test stroke. Every edit re-strokes, and
/// a stroke's seed is normally the document clock — which advances with each
/// replay's commit, re-rolling the color dynamics and dither each time and
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
type BrushEdit = Box<dyn FnOnce(&mut BrushConfig, &mut Transient)>;

/// The parameters the pen can drive (§6.2) — one variant per modulation
/// target the brush carries, and the addressing for the one open mapping row.
///
/// It carries everything about a row that differs: its word, its range, where its
/// base value lives on the brush, and which mapping slot belongs to it. That is what
/// lets [`mod_slider`] take a `ModRow` and nothing else, and it is why the rows
/// cannot drift out of step with the engine's set — adding a target to any of the
/// modulation tables (`BrushModulations`, `PaintModulations`, `EraseModulations`)
/// and not here fails to compile at [`Self::slot`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ModRow {
    Size,
    Opacity,
    Flow,
    Stretch,
    ToothGive,
    Add,
    Lift,
    Deposit,
    Bleed,
}

impl ModRow {
    /// The word on the row, which is also the word the section already used for
    /// the parameter. Takes the brush because the Flow row *is* the in-force
    /// effect's rate, and the liquify effect's rate is not a flow of anything:
    /// it is how hard the paint follows (§6.13).
    fn label(self, b: &BrushConfig) -> &'static str {
        match self {
            Self::Size => "Size",
            Self::Opacity => "Opacity",
            Self::Flow => match b.effect {
                BrushEffectType::Liquify => "Strength",
                _ => "Flow",
            },
            Self::Stretch => "Stretch",
            Self::ToothGive => "Tooth give",
            Self::Add => "Add",
            Self::Lift => "Lift",
            Self::Deposit => "Deposit",
            Self::Bleed => "Bleed",
        }
    }

    /// The glyph beside the row's word, where the parameter has one it wears
    /// everywhere else — the opacity's, which the layer and selection panels
    /// show against their own opacity sliders.
    fn glyph(self) -> Option<Icon> {
        match self {
            Self::Opacity => Some(stark_ui::icons::OPACITY),
            _ => None,
        }
    }

    /// The base slider's range, for the brush being edited. The three wet fluxes
    /// stop at 0.95 for the reason they always did — λ diverges at 1 (§6.2).
    ///
    /// Takes both halves because one row's top is not a constant: the Flow
    /// row's is the in-force effect's, and `Stretch`'s asks the engine about
    /// the tip the transient sizes.
    fn range(self, b: &BrushConfig, t: Transient) -> (f32, f32) {
        match self {
            Self::Size => (MIN_RADIUS, MAX_RADIUS),
            // A ceiling: the fraction of a full stroke (§6.2, §6.12).
            Self::Opacity => (0.0, 1.0),
            // The in-force effect's own range (`BrushConfig::max_flow`) — the
            // liquify strength stops at its quoted 1, the rates at the slider's
            // own top.
            Self::Flow => (0.0, b.max_flow()),
            // The knob is `1 − 1/s`, so its own top is an infinitely long tip. Two
            // things stop it short, and the smaller wins: the elongation saturates
            // at `MAX_ELONGATION`, past which the slider stops meaning anything
            // (§6.6) — and the *renderer* cannot draw a tip reaching further than
            // one region holds, which for a large brush bites first
            // (`stark_engine::max_stretch`, §6.2).
            //
            // Asking the engine rather than restating its arithmetic is the whole
            // point: a tip past that limit does not draw a coarser stroke, it
            // silently stops lifting and depositing altogether. A slider that
            // offered one would be offering a broken brush, and no note beside it
            // would make that better than not offering it.
            Self::Stretch => (0.0, stark_engine::max_stretch(&b.params(t))),
            // Full range, and it reads right-to-left: 1 is all the give there is, so
            // the substrate gates nothing, and 0 is the driest tip (§6.4). Quoted that
            // way round for the pen's sake — see `BrushParams::tooth_give`.
            Self::ToothGive => (0.0, 1.0),
            // The full share (`BrushDynamics::add`): 1 is a wet brush laying
            // exactly what a paint brush at the same flow would.
            Self::Add => (0.0, 1.0),
            Self::Lift | Self::Deposit | Self::Bleed => (0.0, 0.95),
        }
    }

    fn get(self, b: &BrushConfig, t: Transient) -> f32 {
        match self {
            Self::Size => t.size,
            // The ceiling of whichever effect is in force — the laying side's
            // or the eraser's own (`BrushConfig::opacity`).
            Self::Opacity => b.opacity(),
            // The overall rate of whichever effect is in force (§6.2, §6.12)
            // — the transient's, like the size beside it.
            Self::Flow => t.flow,
            Self::Stretch => b.stretch,
            Self::ToothGive => b.tooth.give,
            Self::Add => b.wet.add,
            Self::Lift => b.wet.lift,
            Self::Deposit => b.wet.deposit,
            Self::Bleed => b.wet.bleed,
        }
    }

    /// The three wet-only rows write the wet half directly: the configuration
    /// holds every effect, so a write racing the pen's eraser end (§18.1.8) —
    /// [`edit`] defers its closure a throttle window — lands on the remembered
    /// wet half instead of being dropped. The effect switch is the user's own
    /// and never moves under an edit (`BrushConfig::effect`).
    fn set(self, b: &mut BrushConfig, t: &mut Transient, v: f32) {
        match self {
            Self::Size => t.size = v,
            Self::Opacity => b.set_opacity(v),
            Self::Flow => t.flow = v,
            Self::Stretch => b.stretch = v,
            Self::ToothGive => b.tooth.give = v,
            Self::Add => b.wet.add = v,
            Self::Lift => b.wet.lift = v,
            Self::Deposit => b.wet.deposit = v,
            Self::Bleed => b.wet.bleed = v,
        }
    }

    /// Where this row's mapping lives on the brush: the tip's own table
    /// (`BrushModulations`), or the effect's — which for Flow is whichever
    /// effect is in force, that being the row's whole point (§6.12).
    fn slot(self, b: &mut BrushConfig) -> &mut Option<Modulation> {
        match self {
            Self::Size => &mut b.modulation.size,
            Self::Stretch => &mut b.modulation.stretch,
            Self::ToothGive => &mut b.modulation.tooth_give,
            Self::Flow => match b.effect {
                BrushEffectType::Paint | BrushEffectType::Wet => &mut b.flow_modulation,
                BrushEffectType::Erase => &mut b.erase.flow_modulation,
                BrushEffectType::Liquify => &mut b.liquify.strength_modulation,
            },
            // The laying side's or the eraser's, like the dial itself. A liquify
            // brush shows no Opacity row at all (§6.13), so its arm is never
            // reached; the laying side's slot is what the dial would write if it
            // were.
            Self::Opacity => match b.effect {
                BrushEffectType::Erase => &mut b.erase.opacity_modulation,
                BrushEffectType::Paint | BrushEffectType::Wet | BrushEffectType::Liquify => {
                    &mut b.opacity_modulation
                }
            },
            Self::Add => &mut b.wet.add_modulation,
            Self::Lift => &mut b.wet.lift_modulation,
            Self::Deposit => &mut b.wet.deposit_modulation,
            Self::Bleed => &mut b.wet.bleed_modulation,
        }
    }

    fn of(self, b: &BrushConfig) -> Option<Modulation> {
        match self {
            Self::Size => b.modulation.size,
            Self::Stretch => b.modulation.stretch,
            Self::ToothGive => b.modulation.tooth_give,
            Self::Flow => match b.effect {
                BrushEffectType::Paint | BrushEffectType::Wet => b.flow_modulation,
                BrushEffectType::Erase => b.erase.flow_modulation,
                BrushEffectType::Liquify => b.liquify.strength_modulation,
            },
            Self::Opacity => match b.effect {
                BrushEffectType::Erase => b.erase.opacity_modulation,
                BrushEffectType::Paint | BrushEffectType::Wet | BrushEffectType::Liquify => {
                    b.opacity_modulation
                }
            },
            Self::Add => b.wet.add_modulation,
            Self::Lift => b.wet.lift_modulation,
            Self::Deposit => b.wet.deposit_modulation,
            Self::Bleed => b.wet.bleed_modulation,
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
    ///
    /// A writable `Signal`, unlike [`Signals::renderer`](crate::state::Signals::renderer)(crate::state::AppState),
    /// and deliberately: this engine has no observable projection and no chrome
    /// reading one back — the dialog renders from its own signals. There is
    /// therefore no publish to pair a mutation with, which is the whole of what
    /// `state::with_engine` exists to enforce.
    renderer: Signal<Option<Renderer>>,
    /// The test stroke (canvas-space samples), replayed on every setting change.
    samples: Signal<Vec<InputSample>>,
    /// Whether `samples` is the user's own stroke rather than the seeded default
    /// — the one thing [`resize_preview`] must not re-seed over.
    drawn: Signal<bool>,
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

/// A part of this dialog the guided tour can point at (§24.3).
///
/// The tour's cards are placed against a measured box, so each part it names has to
/// be findable in the DOM — and findable by something that **cannot drift from the
/// markup**. This enum is that something: the `data-be` attribute below is written
/// from [`key`](Self::key) and the tour's selector is built from the same function,
/// exactly as a panel's `data-panel` and the drag that resolves against it share
/// `layout::panel_key`. A section renamed on screen keeps its key; a section deleted
/// stops compiling on both sides at once.
///
/// Only the parts a lesson names are here. The dialog has more boxes than this and
/// they are none of the tour's business.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum BrushPart {
    /// The dialog itself.
    Dialog,
    /// The live test stroke down the right-hand column.
    Preview,
    /// The four folding parameter groups, in the order they are laid out.
    Tip,
    Paint,
    Color,
    Wet,
}

impl BrushPart {
    /// The name this part wears in the DOM, and the one a selector finds it by.
    pub fn key(self) -> &'static str {
        match self {
            BrushPart::Dialog => "dialog",
            BrushPart::Preview => "preview",
            BrushPart::Tip => "tip",
            BrushPart::Paint => "paint",
            BrushPart::Color => "color",
            BrushPart::Wet => "wet",
        }
    }
}

/// The brush editor dialog. Mounted only while open (so each open re-inits the
/// preview against the current canvas look and re-seeds the section state).
#[component]
pub fn BrushEditorModal(on_close: EventHandler<()>) -> Element {
    let state = use_context::<AppState>();
    let preview = Preview {
        renderer: use_signal(|| None),
        samples: use_signal(Vec::new),
        drawn: use_signal(|| false),
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
    // brush edits (sliders restroke through `edit`'s throttle instead), and
    // the initial run is a no-op because the preview renderer isn't up yet.
    let shape = use_memo(move || (state.brush)().shape);
    use_effect(move || {
        let _ = shape();
        restroke(state, preview);
    });

    // Section fold state: the everyday groups start open, the specialised ones closed.
    let tip_open = use_signal(|| true);
    let paint_open = use_signal(|| true);
    let color_open = use_signal(|| false);
    let wet_open = use_signal(|| false);
    let _surface_open = use_signal(|| false);
    // Per-section "Show more" for the rarely-touched knobs.
    let wet_more = use_signal(|| false);
    // Which parameter's pen mapping is open, at most one at a time — so the dialog
    // grows by one sub-row while a mapping is being edited and by nothing otherwise
    // (see [`mod_slider`]). Held here rather than per row because the rows are plain
    // calls, not components, and because "one at a time" is the behaviour wanted.
    let mod_open = use_signal(|| None::<ModRow>);

    // The frontend's own signals (`state::AppState::{brush, transient}`): the
    // dialog's rows are the brush, and they wake for brush edits and nothing
    // else.
    let brush = (state.brush)();
    let tune = (state.transient)();
    let is_round = matches!(brush.shape, BrushShape::Round { .. });
    // Which effect the brush is (§6.2, §6.12) — what gates the laying-only
    // sections below, what the effect chips read, and what the effect section
    // calls itself.
    let erases = brush.effect == BrushEffectType::Erase;
    let liquifies = brush.effect == BrushEffectType::Liquify;
    // Whether the effect in force lays pigment at all — what gates the sections
    // that are properties of laying it: the color dynamics, and the opacity
    // ceiling on the amount laid (§6.12, §6.13).
    let lays = !erases && !liquifies;
    let (effect_title, effect_desc) = match brush.effect {
        BrushEffectType::Paint => (
            "Paint",
            "The brush's own paint: how much goes down and how far it lasts.",
        ),
        BrushEffectType::Wet => (
            "Wet",
            "The paint mixes with what is on the canvas: lift, deposit, bleed.",
        ),
        BrushEffectType::Erase => (
            "Erase",
            "The stroke removes what the eye sees, instead of laying paint.",
        ),
        BrushEffectType::Liquify => (
            "Liquify",
            "The stroke drags the picture with it — paint warps instead of mixing.",
        ),
    };
    let charge = brush.wet.charge;
    let cd = brush.color_dynamics;
    // The jitter channels are the *color space's* channels — label them for
    // whatever space the document is in.
    let space = state
        .renderer
        .read()
        .as_ref()
        .map(|r| r.color_space())
        .unwrap_or(ColorSpaceId::Oklab);
    // The substrate the document is on (§6.4) — what the tooth has to bite into.
    let substrate = state
        .renderer
        .read()
        .as_ref()
        .map(|r| r.substrate())
        .unwrap_or_default();
    let ch_labels = match space {
        ColorSpaceId::Mixbox => ["Pigment 1", "Pigment 2", "Pigment 3"],
        _ => ["Lightness", "Green \u{2194} red", "Blue \u{2194} yellow"],
    };

    // What the header's "Overwrite preset" can do (`stark_ui::presets::overwrite`), asked
    // of the library, the name in hand and the brush as they are *now* — so the
    // button wakes with the first edit that moves the brush off its preset.
    let in_hand = (state.preset_in_hand)();
    let verdict =
        stark_ui::presets::overwrite(&state.presets.read(), in_hand.as_deref(), &(brush, tune));
    let overwrite_ready = matches!(verdict, stark_ui::presets::Overwrite::Ready(_));
    let overwrite_title = match &verdict {
        stark_ui::presets::Overwrite::Nothing => {
            "The brush was not taken from a preset \u{2014} save a new one".to_string()
        }
        stark_ui::presets::Overwrite::Builtin(name) => format!(
            "\u{201C}{name}\u{201D} is one of the app's own presets, which it keeps up to date \
             \u{2014} save a new one instead"
        ),
        stark_ui::presets::Overwrite::Unchanged(name) => {
            format!("\u{201C}{name}\u{201D} already is this brush")
        }
        stark_ui::presets::Overwrite::Ready(name) => {
            format!("Replace \u{201C}{name}\u{201D} with the brush in hand")
        }
    };
    let save_new_title = stark_ui::commands::advertised(
        "Keep the brush in hand under a new name",
        Command::SavePreset,
        &state.bindings.read(),
    );

    rsx! {
        Modal { class: "be-dialog", "data-be": "{BrushPart::Dialog.key()}", on_close,
            div { class: "be-header",
                div { class: "modal-title",
                    "Brush"
                    // Which preset the brush descends from — what "Overwrite preset"
                    // would write over, said where it can be read without hovering
                    // the button for it.
                    if let Some(name) = &in_hand {
                        span { class: "be-from", "{name}" }
                    }
                }
                // The three ways out of the dialog, in the order the work happens:
                // keep the tuned brush where it came from, keep it under a new name,
                // or keep it only in hand. Each one closes the dialog — kept is done.
                // Saving lives here rather than on the Brush panel because the brush
                // worth keeping is the one that has just been tuned.
                div { class: "be-header-actions",
                    // Dead in every arm of `stark_ui::presets::overwrite` but one, and the
                    // tooltip says which: a built-in is rebuilt next start, so the
                    // write would not last; a brush still equal to its preset has
                    // nothing to write.
                    button {
                        class: "btn btn-secondary",
                        disabled: !overwrite_ready,
                        title: "{overwrite_title}",
                        onclick: move |_| {
                            // The tail of a drag may still be waiting out the
                            // throttle (`edit`), and the snapshot has to be of the
                            // brush as tuned — so it lands first. The `use_drop`
                            // flush would land it too, a moment after the preset
                            // had been written without it.
                            let mut pending = preview.pending;
                            if let Some(f) = pending.write().take() {
                                update_brush(state, f);
                            }
                            presets::overwrite_in_hand(state);
                            on_close.call(());
                        },
                        {icon(stark_ui::icons::SAVE)}
                        "Overwrite preset"
                    }
                    // Its own words, the registry's act (§25.1): the command is
                    // what the palette and a chord reach, so the tooltip advertises
                    // it. The dialog it raises stacks over this one and takes it
                    // down with the name confirmed (`PresetSaveModal`); cancelled,
                    // it leaves the editor as it was.
                    button {
                        class: "btn btn-secondary",
                        title: "{save_new_title}",
                        onclick: move |_| commands::run(Command::SavePreset, state),
                        {icon(stark_ui::icons::ADD)}
                        "Save new preset"
                    }
                    button { class: "btn btn-primary", onclick: move |_| on_close.call(()),
                        {icon(stark_ui::icons::DONE)}
                        "Done"
                    }
                }
            }

            // Live test canvas. First in the markup — it is what the dialog is
            // about — but the grid puts it in the right-hand column, full
            // height (the header spans both, the sections take the left).
            // Draw on it to replace the test stroke; ↺ restores the default.
            // The stroke re-renders as every setting changes.
            div { class: "be-preview-wrap", "data-be": "{BrushPart::Preview.key()}",
                canvas {
                    id: PREVIEW_CANVAS_ID,
                    class: "brush-preview",
                    onmounted: move |_| { spawn(init_preview(state, preview)); },
                    onresize: move |e| {
                        if let Ok(size) = e.get_content_box_size() {
                            resize_preview(state, preview, size.width as u32, size.height as u32);
                        }
                    },
                    onpointerdown: move |e| {
                        if e.trigger_button() == Some(MouseButton::Primary) {
                            capture_pointer(&e);
                            start_preview_stroke(state, preview, &e);
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
                // (`stark_ui::icons::RESET`).
                button {
                    class: "be-preview-reset",
                    title: "Restore the default test stroke",
                    onclick: move |_| reset_stroke(state, preview),
                    {icon(stark_ui::icons::RESET)}
                }
            }

            div { class: "be-sections",
                Section {
                    part: BrushPart::Tip,
                    title: "Tip", desc: "The footprint the stroke sweeps along the path.",
                    glyph: stark_ui::icons::TIP,
                    open: tip_open,
                    ShapeGallery {}
                    // Orientation is what aims the footprint (§6.6), and there
                    // are two ways for that to matter: a non-round tip has a
                    // silhouette to turn, and **any** tip that stretches has an
                    // axis to draw out along. A round tip that does neither is the
                    // one case where the chips would decide nothing, so it is the
                    // one case that does not show them. Hardness stays the
                    // procedural tip's alone.
                    if !is_round || brush.stretch > 0.0 {
                        div { class: "brush-shapes",
                            button { class: chip(brush.orientation == OrientationSource::FollowStroke),
                                onclick: move |_| { set_orientation(state, OrientationSource::FollowStroke); restroke(state, preview); },
                                "Follow stroke" }
                            button { class: chip(brush.orientation == OrientationSource::Pen),
                                onclick: move |_| { set_orientation(state, OrientationSource::Pen); restroke(state, preview); },
                                "Pen angle" }
                        }
                    }
                    {mod_slider(state, preview, mod_open, ModRow::Size, brush, tune)}
                    // How far the footprint is drawn out along the axis above
                    // (§6.6). Pointed at Tilt with "Pen angle" this is the pencil:
                    // lean the pen and the contact patch elongates along the lean,
                    // the way a real conical tip's does. Held at a value with no
                    // mapping it is a chisel nib, off a plain round tip.
                    {mod_slider(state, preview, mod_open, ModRow::Stretch, brush, tune)}
                    // Stretching along the *tangent* is a coherent thing to ask
                    // for — the tip lays more paint per unit travel — but it is not
                    // the one people reach for this slider wanting, and the
                    // difference is invisible until the pen is leaned. So say
                    // which axis is in force rather than second-guess the setting.
                    if brush.stretch > 0.0 && brush.orientation == OrientationSource::FollowStroke {
                        div { class: "be-note",
                            "Stretching along the stroke, so the mark gets heavier rather                                  than wider. Switch to Pen angle for a tip that broadens as                                  the pen leans." }
                    }
                    // Why the slider stopped short of where it stops on a smaller
                    // brush. Said only when it actually did: below ~110 px the
                    // whole range is there and there is nothing to explain.
                    if stark_engine::max_stretch(&brush.params(tune)) < BrushParams::MAX_STRETCH {
                        div { class: "be-note",
                            "This tip is too big to draw out any further — a stroke                                  that lifts and deposits works over a copy of the canvas                                  beneath it, and that has a size limit. Lower Size to                                  stretch it more." }
                    }
                    if let BrushShape::Round { hardness } = brush.shape {
                        Slider { label: "Hardness", min: 0.0, max: 1.0, value: hardness,
                            oninput: move |v| edit(state, preview, move |b, _| {
                                if let BrushShape::Round { hardness } = &mut b.shape {
                                    *hardness = v;
                                }
                            }) }
                    }
                    // The two tapers — the run over which the tip widens from a
                    // point (§6.2). In *radii*, so a taper keeps its shape as the
                    // brush is resized, which is why the labels say so.
                    Slider { label: "Start taper (radii)", min: 0.0, max: MAX_TAPER, value: brush.start_taper_length,
                        oninput: move |v| edit(state, preview, move |b, _| b.start_taper_length = v) }
                    Slider { label: "End taper (radii)", min: 0.0, max: MAX_TAPER, value: brush.end_taper_length,
                        oninput: move |v| edit(state, preview, move |b, _| b.end_taper_length = v) }
                    // Stroke smoothing (§6.11): the towed tip. The one
                    // slider here whose knob never reaches the engine — the
                    // stored path already embodies it, so the amount is the
                    // frontend's own (`BrushConfig::smoothing`), riding
                    // presets and the rack with the rest of the brush.
                    Slider { label: "Smoothing", min: 0.0, max: 1.0, value: brush.smoothing,
                        oninput: move |v| edit(state, preview, move |b, _| b.smoothing = v) }
                }

                Section {
                    part: BrushPart::Paint,
                    title: effect_title,
                    desc: effect_desc,
                    glyph: stark_ui::icons::PAINT,
                    open: paint_open,
                    // What a stroke of this brush **does** (§6.2, §6.12):
                    // paint, wet paint, or erase. Chips rather than a dial,
                    // because it is the tool's identity and not an amount — the
                    // sections below come and go with it, which a slider
                    // position would not say. The user's own choice: no slider
                    // moves this switch (`BrushConfig::effect`).
                    div { class: "brush-shapes",
                        button { class: chip(brush.effect == BrushEffectType::Paint),
                            onclick: move |_| set_effect(state, preview, BrushEffectType::Paint),
                            "Paint" }
                        button { class: chip(brush.effect == BrushEffectType::Wet),
                            onclick: move |_| set_effect(state, preview, BrushEffectType::Wet),
                            "Wet" }
                        button { class: chip(erases),
                            onclick: move |_| set_effect(state, preview, BrushEffectType::Erase),
                            "Erase" }
                        button { class: chip(liquifies),
                            onclick: move |_| set_effect(state, preview, BrushEffectType::Liquify),
                            "Liquify" }
                    }
                    // The effect's ceiling (§6.2, §6.12), whichever it is: the
                    // fraction of a full stroke this stroke lays — or, erasing,
                    // removes. 0.5 really shows (or leaves) half, however hard
                    // the spot is scrubbed. Not a rate: the rate is Flow below.
                    // The pen can drive it like the rate — a light touch lays a
                    // faint mark a heavy one fills in — which is the chip on the
                    // row. A liquify brush has no such ceiling — scrubbing keeps
                    // carrying (§6.13) — so the row is not shown rather than
                    // shown and vetoed (`BrushConfig::set_opacity`).
                    if !liquifies {
                        {mod_slider(state, preview, mod_open, ModRow::Opacity, brush, tune)}
                    }
                    // The effect's overall rate (§6.2): how much a pass lays — and,
                    // wet, how hard it works the canvas; erasing, how fast the bite
                    // builds toward its ceiling (§6.12). Not a wet axis: what the
                    // tool *does* per unit of this is the Wet section's business.
                    {mod_slider(state, preview, mod_open, ModRow::Flow, brush, tune)}
                    // How finely a liquify drag is stepped (§6.13): at 1 every
                    // step is a contraction and a hard tip steps by the texel;
                    // lower is the same field from fewer steps, faster, with the
                    // paint ahead of a hard edge squashed rather than carried. A
                    // cost dial, not a rate, so no pen chip.
                    if liquifies {
                        Slider { label: "Quality", min: 0.0, max: 1.0, value: brush.liquify.quality,
                            oninput: move |v| edit(state, preview, move |b, _| b.liquify.quality = v) }
                    }
                    // How far the tip settles into the canvas's own tooth (§6.4):
                    // at 1 it follows every fall, the substrate is irrelevant and the
                    // mark is solid; turned *down* the paint catches on the
                    // substrate's peaks and skips its valleys, which is what a dry
                    // brush leaves.
                    //
                    // The one slider here whose interesting end is the left, and that
                    // is the model rather than an oversight: a modulation only scales
                    // down, so quoting the knob as the give is what makes a pressure
                    // mapping the charcoal — light touch dry, borne down solid —
                    // instead of its opposite (`ToothParams::give`).
                    {mod_slider(state, preview, mod_open, ModRow::ToothGive, brush, tune)}
                    // ...and how *abruptly* it meets the grain, which is the other
                    // half of contact and a different question (§6.4). Narrow and the mark is
                    // a level set of the grain — the faces print and the valleys do
                    // not, which is paint sitting on the substrate. Wide and the tip
                    // crumbles into the valleys instead of spanning them, so the grain
                    // reads as a tone rather than a pattern: the charcoal.
                    //
                    // Not a `mod_slider`, and that is the model rather than an
                    // omission: the pen presses a tip harder, it does not make it out
                    // of something else (`BrushModulations::tooth_give`).
                    Slider { label: "Tooth softness", min: 0.0, max: MAX_TOOTH_SOFTNESS, value: brush.tooth.softness,
                        oninput: move |v| edit(state, preview, move |b, _| b.tooth.softness = v) }
                    // The substrate is the *document's*, not the brush's — a pencil
                    // and a loaded brush on one canvas see one tooth — so on a
                    // smooth canvas this knob has nothing to bite and says so,
                    // rather than moving and changing nothing.
                    if brush.tooth.give < 1.0 && substrate == SubstrateId::Flat {
                        div { class: "be-note",
                            "This canvas is smooth, so there is no tooth to catch on. \
                             Pick a substrate in the Lighting panel."
                        }
                    }
                    // The deposit jitter (§6.2): every texel scales the paint it
                    // takes by its own factor in (1 − j, 1 + j), fixed for the
                    // stroke — what keeps the wet loop's accumulation from banding,
                    // at the 1% default; the rest of the track is grain as a look.
                    // The slider stops at 0.2 where the field runs to 1
                    // (`BrushParams::jitter`): past strong grain the gate
                    // is only noise, and a ceiling the model does not own belongs to
                    // the slider's end rather than to the quantity.
                    Slider { label: "Jitter", min: 0.0, max: 0.2, value: brush.jitter,
                        oninput: move |v| edit(state, preview, move |b, _| b.jitter = v) }
                    // Depletion per *radius* travelled — the stroke runs dry. 0 is
                    // what a pen or a digital brush wants; not behind "Show more",
                    // because it is the only knob that decides whether a tool runs
                    // out. In radii, so this slider's top means the same thing at
                    // every brush size (§6.2): dry two radii past the press. Quoted
                    // per canvas px it did not — the same setting was a gentle fade
                    // on a small tip and a stub on a large one, which is the whole
                    // of why `radius` had to be read as something other than scale.
                    Slider { label: "Drain", min: 0.0, max: 0.5, value: brush.drain,
                        oninput: move |v| edit(state, preview, move |b, _| b.drain = v) }
                }

                // Pigment wander is a property of laying pigment, so the whole
                // section is the laying side's (§6.12, §6.13) — an eraser or a
                // liquify brush shows no rows that reach nothing. The closures
                // write the paint side directly: it is always there to take them
                // (`BrushConfig`), even when the pen's eraser end swaps the
                // effect inside `edit`'s throttle window (§18.1.8).
                if lays {
                    Section {
                        part: BrushPart::Color,
                        title: "Color dynamics", desc: "The color wanders across the brush and along the stroke, following a noise field.",
                        glyph: stark_ui::icons::COLOR,
                        open: color_open,
                        div { class: "brush-shapes",
                            for kind in [NoiseKind::Simplex, NoiseKind::White, NoiseKind::Voronoi, NoiseKind::Mosaic] {
                                button {
                                    key: "{noise_label(kind)}",
                                    class: chip(cd.noise == kind),
                                    onclick: move |_| edit(state, preview, move |b, _| b.color_dynamics.noise = kind),
                                    "{noise_label(kind)}"
                                }
                            }
                        }
                        // How far each color channel wanders (± in the channel's units).
                        for i in 0..3 {
                            Slider { label: ch_labels[i].to_string(), min: 0.0, max: 0.5, value: cd.amplitude[i],
                                oninput: move |v| edit(state, preview, move |b, _| b.color_dynamics.amplitude[i] = v) }
                        }
                        // How fast it wanders along each lookup axis; the modulation
                        // sliders live only while some channel is active (no effect at 0).
                        if cd.amplitude.iter().any(|a| *a > 0.0) {
                            div { class: "be-sub",
                                Slider { label: "Scale \u{2192} across stroke", min: 0.0, max: 8.0, value: cd.frequency[0],
                                    oninput: move |v| edit(state, preview, move |b, _| b.color_dynamics.frequency[0] = v) }
                                Slider { label: "Scale \u{2192} along stroke", min: 0.0, max: 8.0, value: cd.frequency[1],
                                    oninput: move |v| edit(state, preview, move |b, _| b.color_dynamics.frequency[1] = v) }
                            }
                        }
                    }
                }

                // The fluxes are the wet effect's own (§6.2), so the section
                // goes with the chip that names it: a paint brush lays and an
                // eraser removes, and neither has an axis here to show.
                if brush.effect == BrushEffectType::Wet {
                    Section {
                        part: BrushPart::Wet,
                        title: "Wet", desc: "Canvas paint on the move — smudge, knife, blur.",
                        glyph: stark_ui::icons::WET,
                        open: wet_open,
                        // The source axis (§6.2): how much of the brush's own paint
                        // is in the mix, as a share the shared Flow scales. At 0 the
                        // tool only works what is there — the blender — and the Flow
                        // slider strengthens the blend instead of laying paint.
                        {mod_slider(state, preview, mod_open, ModRow::Add, brush, tune)}
                        // The three fluxes a palette knife is built out of, and the
                        // three most worth mapping onto the pen: a knife that lifts
                        // with pressure and lays back with tilt is those two chips
                        // (§6.2).
                        {mod_slider(state, preview, mod_open, ModRow::Lift, brush, tune)}
                        {mod_slider(state, preview, mod_open, ModRow::Deposit, brush, tune)}
                        // The lateral axis: the paint under the tip diffuses towards its
                        // neighbours (§6.2). Alone it is a blur brush; under `add` it
                        // melts the ridges of the strokes being painted over. Capped at
                        // 0.95 like the two vertical rates — the λ diverges at 1.
                        {mod_slider(state, preview, mod_open, ModRow::Bleed, brush, tune)}
                        More { open: wet_more,
                            // The finite glob pre-loaded on the tool (palette knife, §6.2).
                            Slider { label: "Charge", min: 0.0, max: 2.0, value: charge,
                                oninput: move |v| edit(state, preview, move |b, _| b.wet.charge = v) }
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
    brush: BrushConfig,
    tune: Transient,
) -> Element {
    let mut open = open;
    let (min, max) = row.range(&brush, tune);
    let m = row.of(&brush);
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
            Slider { label: row.label(&brush).to_string(), glyph: row.glyph(), min, max, value: row.get(&brush, tune),
                oninput: move |v| edit(state, preview, move |b, t| row.set(b, t, v)) }
            button {
                class: chip_class,
                title: "What the pen drives this with",
                onclick: move |_| open.set(if expanded { None } else { Some(row) }),
                match m {
                    Some(m) => rsx! { "{source_label(m.source)}" },
                    None => rsx! { {icon(stark_ui::icons::MODULATE)} },
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
                                oninput: move |v| edit(state, preview, move |b, _| {
                                    if let Some(m) = row.slot(b) { m.floor = v; }
                                }) }
                            // Negative = late, positive = early; 0 is linear,
                            // exactly. Unlabelled at its ends because the plot
                            // beside it moves as the knob does, which says which
                            // way is which better than two words either side of a
                            // track this narrow.
                            Slider { label: "Response".to_string(),
                                min: -1.0, max: 1.0, value: m.curve,
                                oninput: move |v| edit(state, preview, move |b, _| {
                                    if let Some(m) = row.slot(b) { m.curve = v; }
                                }) }
                        }
                        {curve_plot(m)}
                    }
                }
            }
        }
    }
}

/// The word a noise kind wears on its chip.
fn noise_label(kind: NoiseKind) -> &'static str {
    match kind {
        NoiseKind::Simplex => "Simplex",
        NoiseKind::White => "White",
        NoiseKind::Voronoi => "Voronoi",
        NoiseKind::Mosaic => "Mosaic",
    }
}

/// Switch what the brush does (§6.2, §6.12). One field moves and nothing is
/// forgotten: the brush carries every effect's configuration (`BrushConfig`),
/// so switching to Erase and back costs a tuned smudge none of its axes —
/// across dialog closes, preset swaps and sessions alike, where the stash this
/// replaced remembered only for the dialog's own length.
fn set_effect(state: AppState, preview: Preview, kind: BrushEffectType) {
    edit(state, preview, move |b, _| b.effect = kind);
}

/// Set (or clear) a row's mapping source, keeping the shape it already had — so
/// switching pressure → tilt is one edit rather than three.
fn set_source(state: AppState, preview: Preview, row: ModRow, source: Option<ModSource>) {
    edit(state, preview, move |b, _| {
        let held = row.of(b);
        *row.slot(b) = source.map(|source| Modulation {
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

    // The one field the gallery reads: which card wears the selected ring moves
    // when a shape is chosen and at no other time.
    let brush_shape = (state.brush)().shape;
    // One card per bundled shape, in table order. A built-in whose fetch is
    // still in flight has no id yet, so it simply never reads as selected —
    // clicking it is the same no-op, and both settle when the bytes land. Its
    // picture waits on the same moment, because the picture is the *coverage the
    // engine imported* rather than the bundled file (`shapes::thumbnail`): a
    // built-in is authored the same way a user's shape is, so it is shown the
    // same way, and neither has to have put its coverage in an alpha channel.
    let builtins = crate::builtins::resolved(state)
        .into_iter()
        .map(|(builtin, id)| {
            let active = matches!(brush_shape, BrushShape::Stamp(s) if Some(s) == id);
            let thumb = id.and_then(|id| crate::shapes::thumbnail(state, id));
            (builtin.name, thumb, active)
        });
    let entries = state.shapes.entries;
    // Memoized so the list is rebuilt when the library changes rather than on
    // every obs refresh; the encode behind each url is itself remembered per
    // content id, so a card that survives a rebuild costs a scan.
    let thumbs = use_memo(move || {
        entries
            .read()
            .iter()
            .map(|e| {
                (
                    e.id,
                    crate::shapes::id_hex(e.id),
                    e.name.clone(),
                    crate::shapes::thumbnail(state, e.id),
                )
            })
            .collect::<Vec<_>>()
    });

    let card = |active: bool| {
        if active {
            "asset-card selected"
        } else {
            "asset-card"
        }
    };
    let is_round = matches!(brush_shape, BrushShape::Round { .. });

    rsx! {
        div {
            class: if dropping() { "asset-grid dropping" } else { "asset-grid" },
            // `preventDefault` on dragover is what makes the element a drop
            // target at all; the class is just the highlight.
            //
            // **And `stopPropagation`, which is what claims the drop.** The app root
            // takes every drop the window sees, so that one landing on a panel is not
            // handled by the browser navigating away from an unsaved painting (§23.4)
            // — and it places what it gets as a picture. A stamp dropped into this
            // library is a different act, so this handler says so rather than letting
            // the same file be imported twice, two ways.
            ondragover: move |e| {
                e.prevent_default();
                e.stop_propagation();
                dropping.set(true);
            },
            ondragleave: move |_| dropping.set(false),
            ondrop: move |e| {
                e.prevent_default();
                e.stop_propagation();
                dropping.set(false);
                crate::shapes::import_dropped(state, e.files());
            },

            div { class: card(is_round),
                onclick: move |_| set_shape(state, BrushShape::default()),
                div { class: "asset-thumb round" }
                div { class: "asset-name", "Round" }
            }
            for (name, url, active) in builtins {
                div {
                    key: "{name}",
                    class: card(active),
                    onclick: move |_| crate::builtins::select(state, name),
                    div { class: "asset-thumb", style: stark_ui::library::thumb_style(url.as_deref()) }
                    div { class: "asset-name", title: "{name}", "{name}" }
                }
            }
            for (id, key, name, url) in thumbs() {
                div {
                    key: "{key}",
                    class: card(brush_shape == BrushShape::Stamp(id)),
                    onclick: move |_| crate::shapes::select(state, id),
                    div { class: "asset-thumb", style: stark_ui::library::thumb_style(url.as_deref()) }
                    div { class: "asset-name", title: "{name}", "{name}" }
                    // `stark_ui::icons::REMOVE`, as on every other row the application lets you
                    // take something out of — the library of stamps is one more roster.
                    button {
                        class: "asset-remove",
                        title: "Remove from library",
                        onclick: move |e| {
                            e.stop_propagation();
                            crate::shapes::remove(state, id);
                        },
                        {icon(stark_ui::icons::REMOVE)}
                    }
                }
            }
            div { class: "asset-card import",
                // `pick_file` must run inside the click gesture — no task hop.
                onclick: move |_| {
                    pick_file("image/*", move |name, bytes| {
                        crate::shapes::import_file(state, name, bytes);
                    });
                },
                div { class: "asset-thumb plus", {icon(stark_ui::icons::ADD)} }
                div { class: "asset-name", "Import\u{2026}" }
            }
        }
        if let Some(notice) = (state.shapes.notice)() {
            div { class: "asset-notice", "{notice}" }
        }
        div { class: "asset-hint",
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
/// rotate (`stark_ui::icons::FOLD_OPEN`) into a control that must.
#[component]
fn Section(
    part: BrushPart,
    title: String,
    desc: String,
    glyph: Icon,
    open: Signal<bool>,
    children: Element,
) -> Element {
    let mut open = open;
    rsx! {
        div { class: "be-section", "data-be": "{part.key()}",
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

/// Build the preview renderer by **sharing** the main engine's state
/// (`Renderer::shared`): pipelines, imported stamp shapes, the decoded substrate and
/// environment all arrive with it, and the document opens on the canvas's substrate
/// under its lighting — so there is nothing to fetch and nothing to mirror but the
/// background, which is document state. Then seed the default test stroke and
/// paint it with the current brush.
async fn init_preview(state: AppState, mut preview: Preview) {
    // One layout frame, so the freshly-mounted column measures as styled rather
    // than at the canvas's 300×150 intrinsic size. Still only a seed: the element
    // is re-read below, right before anything is placed against it.
    crate::platform::next_frame().await;
    let built = {
        let renderer = state.renderer.peek();
        renderer.as_ref().map(|main| {
            let mut r = main.shared(crate::platform::canvas_by_id(PREVIEW_CANVAS_ID));
            r.process(DocCommand::SetSubstrateColor(
                main.observe().substrate_color,
            ));
            r
        })
    };
    let Some(mut r) = built else { return };

    // Re-read the element before anything is measured against it: both strokes are
    // placed from `r.size()`, so a stale viewport would put them off the column as
    // well as leaving the surface stretched (`Renderer::sync_to_canvas`).
    r.sync_to_canvas();

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

/// The seeded test stroke: an S-curve **down** the preview column with a pressure
/// bell (light → full → light) and a forward tilt that ramps in — so pressure- and
/// tilt-modulated settings visibly shape the stroke even for mouse users.
///
/// Downward because the preview is a tall column, and because it is the direction
/// a hand draws a test stroke in: the run is along the long axis, and the S's swing
/// is across the short one.
fn default_stroke(r: &Renderer) -> Vec<InputSample> {
    let (w, h) = r.size();
    let (w, h) = (w as f32, h as f32);
    let view = r.view();
    const N: usize = 64;
    (0..N)
        .map(|i| {
            let t = i as f32 / (N - 1) as f32;
            let x = w * 0.5 + (t * std::f32::consts::TAU).sin() * w * 0.26;
            let y = h * 0.06 + t * h * 0.88;
            InputSample {
                pos: view.screen_to_canvas(Vec2::new(x, y)),
                pressure: (t * std::f32::consts::PI).sin().clamp(0.08, 1.0),
                // Lean along the (mostly +y) travel direction, growing over the
                // stroke, so tilt→deposit reads as a knife laying down more and more.
                tilt: Vec2::new(0.0, 0.65 * t),
                time: (t * 0.7) as f64,
            }
        })
        .collect()
}

/// The fixed reference stroke laid on the preview canvas before any test
/// stroke: a simple, hard-edged, opaque **red horizontal** band across the
/// middle, committed once at init so the user can see how the brush being
/// edited interacts with paint already on the canvas (smudge, drag, bleed, …).
/// Plain `add` paint, no dynamics, no drain — a clean, unchanging target.
///
/// Across, because the test stroke runs down: the two have to *cross*, or the
/// brush never meets the paint it is meant to be shown moving. It runs off both
/// edges so the crossing is never near an end of it.
fn paint_reference_stroke(r: &mut Renderer) {
    let (w, h) = r.size();
    let (w, h) = (w as f32, h as f32);
    let view = r.view();
    let y = h * 0.5;
    const N: usize = 8;
    let samples: Vec<InputSample> = (0..N)
        .map(|i| {
            let t = i as f32 / (N - 1) as f32;
            let x = w * -0.25 + t * w * 1.5;
            InputSample {
                pos: view.screen_to_canvas(Vec2::new(x, y)),
                pressure: 1.0,
                ..Default::default()
            }
        })
        .collect();
    const REFERENCE_COLOR: [f32; 3] = [0.82, 0.15, 0.12];
    let brush = BrushParams {
        size: 75.0,
        shape: BrushShape::Round { hardness: 0.9 },
        drain: 0.0,
        effect: BrushEffect::painted(REFERENCE_COLOR),
        ..BrushParams::default()
    };
    r.process(ViewCommand::SetBrush {
        brush,
        color: REFERENCE_COLOR,
    });
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
    let brush = *state.brush.peek();
    let mut tune = *state.transient.peek();
    // Force the test stroke to the fixed preview blue so it reads over the red
    // reference stroke; the effect's own opacity (the Opacity slider) is left
    // untouched. A stamp shape needs no handing over: the preview engine shares
    // the main engine's content-addressed asset store, so whatever the brush
    // holds is already here.
    tune.color = PREVIEW_STROKE_COLOR;
    let samples = preview.samples.peek().clone();
    let mut renderer = preview.renderer;
    let mut guard = renderer.write();
    let Some(r) = guard.as_mut() else { return };
    if *preview.committed.peek() {
        r.process(DocCommand::Undo);
    }
    r.process(ViewCommand::SetBrush {
        brush: brush.params(tune),
        color: tune.color,
    });
    // The §6.11 rope the smoothing slider means *on this canvas*: the recorded
    // test stroke is a hand, and replaying it through the tow is what lets the
    // slider show its work on the stroke beside it.
    let rope = stark_ui::input::rope(r.view(), brush.smoothing);
    r.replay_stroke_seeded(Tool::Brush, &samples, PREVIEW_STROKE_SEED, rope);
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
fn edit(
    state: AppState,
    mut preview: Preview,
    f: impl FnOnce(&mut BrushConfig, &mut Transient) + 'static,
) {
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

/// Match the preview surface to a new canvas size. The column runs the dialog's
/// full height, so folding a section or resizing the window changes it, and the
/// drawing buffer — sized once in `finish_init` — would otherwise stretch.
///
/// The **seeded** stroke is then re-laid to the new extent, so the default keeps
/// running the length of the column instead of ending short of it. A stroke the
/// user drew is left exactly where they drew it: it is theirs, and it is in
/// canvas space, so it survives the resize untouched.
fn resize_preview(state: AppState, mut preview: Preview, width: u32, height: u32) {
    let mut renderer = preview.renderer;
    let mut guard = renderer.write();
    let Some(r) = guard.as_mut() else { return };
    if r.size() == (width, height) {
        return;
    }
    r.resize(width, height);
    let reseeded = (!*preview.drawn.peek()).then(|| default_stroke(r));
    drop(guard);
    if let Some(samples) = reseeded {
        preview.samples.set(samples);
    }
    restroke(state, preview);
}

/// Restore the default test stroke and re-render it.
fn reset_stroke(state: AppState, mut preview: Preview) {
    let samples = match preview.renderer.peek().as_ref() {
        Some(r) => default_stroke(r),
        None => return,
    };
    preview.samples.set(samples);
    preview.drawn.set(false);
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
fn start_preview_stroke(state: AppState, mut preview: Preview, e: &Event<PointerData>) {
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
        // Towed like the main canvas (§6.11): the preview is where the brush
        // is felt out, so drawing on it under smoothing has to feel like the
        // brush, not like the brush with its string cut.
        rope: stark_ui::input::rope(r.view(), state.brush.peek().smoothing),
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
        preview.drawn.set(true);
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
