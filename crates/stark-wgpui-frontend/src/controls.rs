//! The dials and the picker the widget layer holds for the panels (§11.1).
//!
//! A library slider owns its value between frames where the hand-built troughs owned
//! nothing, so the view keeps one `SliderState` per knob: written from the model each
//! frame ([`Controls::sync`]) — the brush moves under the keyboard too — and read
//! back on `SliderEvent::Change`. The subscriptions live here because a dropped one
//! is a dial nobody hears.
//!
//! What a drag *means* is still the view's. The handlers call the same `turn`,
//! `turn_dial` and `set_opacity` the measured troughs called, and the mask's strength
//! keeps its bargain — previewed while the hand is on it, spent on release (§6.8).
//! The blend picker is the drop-down §25.9's ladder asked for once the modes stopped
//! fitting on a chip, in place of the cycle that stood in for it.

use stark_engine::ObservableState;
use stark_model::document::{BlendMode, Modulation, PerspectiveGuide};
use stark_ui::prefs::Hdr;
use wgpui::{Context, Entity, Focusable, SharedString, Subscription, Window, prelude::*};
use wgpui_component::IndexPath;
use wgpui_component::input::{InputEvent, InputState};
use wgpui_component::select::{SelectEvent, SelectState};
use wgpui_component::slider::{SliderEvent, SliderState};

use stark_ui::brush_editor::{self, Knob, ModRow, Shown};

use crate::brush::Brush;
use crate::brush_editor::Shape;
use crate::canvas::Canvas;
use crate::panel::KNOBS;
use crate::select::Dial;
use stark_ui::guides as gd;
use stark_ui::lighting as light;

/// Every dial the Select section can mount, in one order, so a state exists for each
/// whether or not this frame shows it.
pub const DIALS: [Dial; 3] = [Dial::Feather, Dial::FillOpacity, Dial::MaskOpacity];

/// The states, and the subscriptions that make them heard.
pub struct Controls {
    /// The brush panel's four, in [`KNOBS`]' order.
    pub knobs: [Entity<SliderState>; KNOBS.len()],
    /// The Select section's, in [`DIALS`]' order — see [`Controls::dial`].
    dials: [Entity<SliderState>; DIALS.len()],
    /// The selected layer's opacity.
    pub opacity: Entity<SliderState>,
    /// The selected layer's blend mode, over [`BlendMode::ALL`]'s labels.
    pub blend: Entity<SelectState<Vec<SharedString>>>,
    /// The Lighting shelf's five, in `stark_ui::lighting::Dial`'s own order —
    /// see [`Controls::light`].
    lights: [Entity<SliderState>; <light::Dial as strum::EnumCount>::COUNT],
    /// The Guides shelf's two, in `stark_ui::guides::Dial`'s own order.
    guide_dials: [Entity<SliderState>; <gd::Dial as strum::EnumCount>::COUNT],
    /// Which room the canvas is lit in, over `light::ENVIRONMENTS`' names.
    pub environment: Entity<SelectState<Vec<SharedString>>>,
    /// The color panel's notation field: what the picker stands on, as text a
    /// person can read, copy, or type over (`stark_ui::color::parse_color`).
    pub hex: Entity<InputState>,
    /// The menu bar's command search (`crate::palette`).
    pub search: Entity<InputState>,
    /// The brush editor's modulatable tracks, in `brush_editor::MOD_ROWS`' order.
    editor_mods: [Entity<SliderState>; brush_editor::MOD_ROWS.len()],
    /// Its plain ones, in `brush_editor::KNOBS`' order.
    editor_knobs: [Entity<SliderState>; brush_editor::KNOBS.len()],
    /// The open mapping's two shape knobs — one pair rather than a pair per row,
    /// because only ever one mapping is open at a time (`crate::brush_editor::Editor`).
    pub mod_floor: Entity<SliderState>,
    pub mod_curve: Entity<SliderState>,
    _subscriptions: Vec<Subscription>,
}

/// **Every track in this chrome that shows a brush runs 0..=1**, whatever the
/// parameter under it does.
///
/// A `SliderState`'s bounds are set when it is built and there is no way to move them
/// on a live one — and three of the ranges are not constants: the editor's Flow row and
/// the Brush shelf's Flow dial both end where the in-force effect does
/// (`BrushConfig::max_flow`), and the Stretch row's top is what the renderer can draw
/// at the size in hand (`stark_ui::brush_editor::ModRow::range`). So the trough is a
/// fraction and the *view* maps it. The figure beside the track prints the real value,
/// so none of this reaches the artist.
const EDITOR_STEP: f32 = 0.005;

impl Controls {
    pub fn new(window: &mut Window, cx: &mut Context<'_, Canvas>) -> Self {
        let mut subs = Vec::new();
        // Fractions, like the editor's tracks below and for the same reason: the Flow
        // knob's top is the in-force effect's (`panel::Knob::range`) and a
        // `SliderState`'s bounds are fixed when it is built.
        let knobs = KNOBS.map(|knob| {
            let state = cx.new(|_| SliderState::new().min(0.0).max(1.0).step(knob.step()));
            subs.push(
                cx.subscribe(&state, move |this, _, event: &SliderEvent, cx| {
                    if let SliderEvent::Change(v) = event {
                        this.turn(knob, v.start(), cx);
                    }
                }),
            );
            state
        });
        let dials = DIALS.map(|dial| {
            let (lo, hi) = dial.range();
            let state = cx.new(|_| SliderState::new().min(lo).max(hi).step(dial.step()));
            subs.push(cx.subscribe(
                &state,
                move |this, _, event: &SliderEvent, cx| match event {
                    SliderEvent::Change(v) => this.turn_dial(dial, fraction(v.start(), lo, hi), cx),
                    SliderEvent::Release(v) => {
                        this.settle_dial(dial, fraction(v.start(), lo, hi), cx);
                    }
                },
            ));
            state
        });
        let opacity = cx.new(|_| SliderState::new().min(0.0).max(1.0).step(0.01));
        subs.push(cx.subscribe(&opacity, |this, _, event: &SliderEvent, cx| {
            if let SliderEvent::Change(v) = event {
                this.set_opacity(v.start(), cx);
            }
        }));
        let labels: Vec<SharedString> = BlendMode::ALL
            .iter()
            .map(|mode| SharedString::from(mode.label()))
            .collect();
        let blend = cx.new(|cx| SelectState::new(labels, Some(IndexPath::default()), window, cx));
        subs.push(cx.subscribe(
            &blend,
            |this, _, event: &SelectEvent<Vec<SharedString>>, cx| {
                let SelectEvent::Confirm(Some(label)) = event else {
                    return;
                };
                if let Some(mode) = BlendMode::ALL.iter().find(|m| m.label() == label.as_ref()) {
                    this.set_blend(*mode, cx);
                }
            },
        ));
        // The Lighting shelf's tracks. Three are a view setting, one is document
        // state and one is this client's own preference — a split the view answers
        // (`Canvas::turn_light`) rather than the track, which knows only its range.
        let lights = std::array::from_fn(|i| {
            let dial = <light::Dial as strum::VariantArray>::VARIANTS[i];
            let (lo, hi) = dial.range();
            let state = cx.new(|_| SliderState::new().min(lo).max(hi).step(dial.step()));
            subs.push(cx.subscribe(
                &state,
                move |this, _, event: &SliderEvent, cx| match event {
                    SliderEvent::Change(v) => this.turn_light(dial, v.start(), cx),
                    // One dial is a stored preference rather than engine state, so it
                    // is shown per sample and written down once (`Canvas::settle_light`).
                    SliderEvent::Release(v) => this.settle_light(dial, v.start(), cx),
                },
            ));
            state
        });
        let guide_dials = std::array::from_fn(|i| {
            let dial = <gd::Dial as strum::VariantArray>::VARIANTS[i];
            let (lo, hi) = dial.range();
            let state = cx.new(|_| SliderState::new().min(lo).max(hi).step(dial.step()));
            subs.push(
                cx.subscribe(&state, move |this, _, event: &SliderEvent, cx| {
                    if let SliderEvent::Change(v) = event {
                        this.turn_guide(dial, v.start(), cx);
                    }
                }),
            );
            state
        });
        let lights_labels: Vec<SharedString> = light::ENVIRONMENTS
            .iter()
            .map(|(_, name)| SharedString::from(*name))
            .collect();
        let environment =
            cx.new(|cx| SelectState::new(lights_labels, Some(IndexPath::default()), window, cx));
        subs.push(cx.subscribe(
            &environment,
            |this, _, event: &SelectEvent<Vec<SharedString>>, cx| {
                let SelectEvent::Confirm(Some(label)) = event else {
                    return;
                };
                if let Some((id, _)) = light::ENVIRONMENTS
                    .iter()
                    .find(|(_, name)| *name == label.as_ref())
                {
                    this.set_environment(*id, cx);
                }
            },
        ));
        let hex = cx.new(|cx| InputState::new(window, cx));
        subs.push(cx.subscribe_in(
            &hex,
            window,
            |this, field, event: &InputEvent, window, cx| match event {
                InputEvent::PressEnter { .. } => {
                    let text = field.read(cx).value();
                    this.take_hex(&text, cx);
                    // Enter is the end of typing; the keyboard goes back to the canvas.
                    this.take_focus(window, cx);
                }
                InputEvent::Blur => {
                    let text = field.read(cx).value();
                    this.take_hex(&text, cx);
                }
                InputEvent::Change | InputEvent::Focus => {}
            },
        ));
        // The brush editor's tracks. Fractions rather than the parameters' own ranges
        // (see `EDITOR_STEP`), so the view is what maps a drag onto whichever range the
        // row has *this* frame.
        let editor_mods = brush_editor::MOD_ROWS.map(|row| {
            let state = cx.new(|_| SliderState::new().min(0.0).max(1.0).step(EDITOR_STEP));
            subs.push(
                cx.subscribe(&state, move |this, _, event: &SliderEvent, cx| {
                    if let SliderEvent::Change(v) = event {
                        this.turn_mod_row(row, v.start(), cx);
                    }
                }),
            );
            state
        });
        let editor_knobs = brush_editor::KNOBS.map(|knob| {
            let state = cx.new(|_| SliderState::new().min(0.0).max(1.0).step(EDITOR_STEP));
            subs.push(
                cx.subscribe(&state, move |this, _, event: &SliderEvent, cx| {
                    if let SliderEvent::Change(v) = event {
                        this.turn_editor_knob(knob, v.start(), cx);
                    }
                }),
            );
            state
        });
        let mod_floor = cx.new(|_| SliderState::new().min(0.0).max(1.0).step(EDITOR_STEP));
        subs.push(
            cx.subscribe(&mod_floor, |this, _, event: &SliderEvent, cx| {
                if let SliderEvent::Change(v) = event {
                    this.turn_mapping(Shape::Floor, v.start(), cx);
                }
            }),
        );
        // The response is quoted −1..=1 — late, linear, early — and the track is a
        // fraction like every other here, so the view spends the one mapping.
        let mod_curve = cx.new(|_| SliderState::new().min(0.0).max(1.0).step(EDITOR_STEP));
        subs.push(
            cx.subscribe(&mod_curve, |this, _, event: &SliderEvent, cx| {
                if let SliderEvent::Change(v) = event {
                    this.turn_mapping(Shape::Curve, v.start(), cx);
                }
            }),
        );
        let search = cx.new(|cx| InputState::new(window, cx).placeholder("Find a command\u{2026}"));
        subs.push(cx.subscribe_in(
            &search,
            window,
            |this, field, event: &InputEvent, window, cx| match event {
                InputEvent::PressEnter { .. } => {
                    let query = field.read(cx).value();
                    this.pick_first(&query, window, cx);
                }
                InputEvent::Focus => this.set_searching(true, cx),
                InputEvent::Blur => this.set_searching(false, cx),
                InputEvent::Change => this.repaint(cx),
            },
        ));
        Self {
            knobs,
            dials,
            opacity,
            blend,
            lights,
            guide_dials,
            environment,
            hex,
            search,
            editor_mods,
            editor_knobs,
            mod_floor,
            mod_curve,
            _subscriptions: subs,
        }
    }

    /// The state behind one of the Lighting shelf's tracks.
    pub fn light(&self, dial: light::Dial) -> &Entity<SliderState> {
        // Seated rather than searched: the index is exhaustive, so a sixth dial
        // is a compile error where the old `expect` was a panic on opening.
        &self.lights[dial.index()]
    }

    /// The state behind one of the Guides shelf's tracks.
    pub fn guide(&self, dial: gd::Dial) -> &Entity<SliderState> {
        // Seated rather than searched, as the lighting dials are.
        let i = <gd::Dial as strum::VariantArray>::VARIANTS
            .iter()
            .position(|d| *d == dial)
            .expect("every guide dial has a state");
        &self.guide_dials[i]
    }

    /// The state behind one of the brush editor's modulatable tracks.
    ///
    /// An index rather than a search: the roster is derived from the enum's own order
    /// (`stark_ui::brush_editor::MOD_ROWS`), so `ModRow::index` names a seat that
    /// exists by construction. It used to be a `position().expect()`, which a tenth
    /// row wired up and left out of the roster would have turned into a panic on the
    /// frame the dialog opened.
    pub fn editor_mod(&self, row: ModRow) -> &Entity<SliderState> {
        &self.editor_mods[row.index()]
    }

    /// The state behind one of its plain ones.
    pub fn editor_knob(&self, knob: Knob) -> &Entity<SliderState> {
        let i = brush_editor::KNOBS
            .iter()
            .position(|k| *k == knob)
            .expect("every editor knob is in `stark_ui::brush_editor::KNOBS`");
        &self.editor_knobs[i]
    }

    /// The state behind one of the Select section's dials.
    pub fn dial(&self, dial: Dial) -> &Entity<SliderState> {
        let i = DIALS
            .iter()
            .position(|d| *d == dial)
            .expect("every dial has a state");
        &self.dials[i]
    }

    /// Bring every state up to what the model says, once per frame.
    ///
    /// A state that already agrees is left alone, so a drag in progress — whose
    /// last value the model has just taken — is not written back under the hand.
    pub fn sync(&self, what: Sync<'_>, window: &mut Window, cx: &mut Context<'_, Canvas>) {
        let Sync {
            brush,
            obs,
            opacity,
            blend,
            hdr,
            guide,
            editor,
            mapping,
        } = what;
        for (knob, state) in KNOBS.iter().zip(&self.knobs) {
            let (lo, hi) = knob.range(brush);
            settle(state, fraction(knob.read(brush), lo, hi), window, cx);
        }
        if let Some(o) = obs {
            for (dial, state) in DIALS.iter().zip(&self.dials) {
                settle(state, dial.read(o), window, cx);
            }
        }
        for (dial, state) in <light::Dial as strum::VariantArray>::VARIANTS
            .iter()
            .zip(&self.lights)
        {
            settle(state, dial.read(obs, hdr), window, cx);
        }
        // Only where there is a guide in hand: with none the tracks are not drawn at
        // all (`crate::guides`), and writing them would be settling a control nobody
        // can see onto a camera that does not exist.
        if let Some(g) = guide {
            for (dial, state) in <gd::Dial as strum::VariantArray>::VARIANTS
                .iter()
                .zip(&self.guide_dials)
            {
                settle(state, dial.read(&g), window, cx);
            }
        }
        // Only while the dialog is up: two dozen settles a frame is real work, and with
        // no editor open every one of them would be writing a control nobody can see.
        if let Some(shown) = editor {
            for (row, state) in brush_editor::MOD_ROWS.iter().zip(&self.editor_mods) {
                let (lo, hi) = row.range(&shown.brush, shown.tune);
                settle(
                    state,
                    fraction(row.get(&shown.brush, shown.tune), lo, hi),
                    window,
                    cx,
                );
            }
            for (knob, state) in brush_editor::KNOBS.iter().zip(&self.editor_knobs) {
                let (lo, hi) = knob.range();
                settle(state, fraction(knob.get(&shown.brush), lo, hi), window, cx);
            }
        }
        // The shape knobs follow whichever mapping is open, so opening a second row's
        // shows that row's floor and response rather than the last one's.
        if let Some(mapping) = mapping {
            settle(&self.mod_floor, mapping.floor, window, cx);
            settle(&self.mod_curve, (mapping.curve + 1.0) * 0.5, window, cx);
        }
        settle(&self.opacity, opacity, window, cx);
        let want = obs
            .map(|o| o.environment)
            .and_then(|env| light::ENVIRONMENTS.iter().position(|(id, _)| *id == env))
            .map(IndexPath::new);
        if self.environment.read(cx).selected_index(cx) != want {
            self.environment
                .update(cx, |s, cx| s.set_selected_index(want, window, cx));
        }
        let want = BlendMode::ALL
            .iter()
            .position(|m| m.same_mode(blend))
            .map(IndexPath::new);
        if self.blend.read(cx).selected_index(cx) != want {
            self.blend
                .update(cx, |s, cx| s.set_selected_index(want, window, cx));
        }
        // The field says what the brush holds — unless a person is typing into it,
        // which is the one time it says what they mean instead.
        let hex = self.hex.read(cx);
        if !hex.focus_handle(cx).is_focused(window) {
            let want = stark_ui::color::notation_of(brush.tune.color);
            if hex.value().as_ref() != want {
                self.hex.update(cx, |s, cx| s.set_value(want, window, cx));
            }
        }
    }
}

/// What one frame's worth of model state is, for [`Controls::sync`].
///
/// A struct rather than six more arguments, for `panel::Sections`' reason: they are
/// mostly numbers, and a caller that shuffled two of them would settle the layer's
/// opacity onto the brush's flow with nothing for the compiler to say.
pub struct Sync<'a> {
    pub brush: &'a Brush,
    pub obs: Option<&'a ObservableState>,
    /// The selected layer's opacity and blend mode.
    pub opacity: f32,
    pub blend: BlendMode,
    /// This client's HDR choice — the one dial on the Lighting shelf that is neither
    /// the document's nor the engine's (§6.5).
    pub hdr: Hdr,
    /// The camera the Guides shelf's tracks are about, where one is in hand.
    pub guide: Option<PerspectiveGuide>,
    /// The brush being edited, while the editor is open — what its two dozen tracks are
    /// settled from, and `None` the rest of the time so they cost nothing.
    pub editor: Option<&'a Shown>,
    /// The mapping whose two shape knobs are showing, if any.
    pub mapping: Option<Modulation>,
}

/// Where `v` stands in `lo..=hi`, which is what the view's handlers speak.
fn fraction(v: f32, lo: f32, hi: f32) -> f32 {
    // A zero-width range is not a mistake here: `ModRow::Stretch`'s top is what the
    // renderer can draw at the size in hand, and a large enough brush leaves it at
    // zero (§6.2). The track then stands at its left end rather than at NaN.
    if (hi - lo).abs() < f32::EPSILON {
        return 0.0;
    }
    ((v - lo) / (hi - lo)).clamp(0.0, 1.0)
}

/// Set a slider to `v` if it does not already say so.
fn settle(state: &Entity<SliderState>, v: f32, window: &mut Window, cx: &mut Context<'_, Canvas>) {
    if (state.read(cx).value().start() - v).abs() > 1e-4 {
        state.update(cx, |s, cx| s.set_value(v, window, cx));
    }
}
