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
use stark_model::document::{BlendMode, PerspectiveGuide};
use stark_ui::prefs::Hdr;
use wgpui::{Context, Entity, Focusable, SharedString, Subscription, Window, prelude::*};
use wgpui_component::IndexPath;
use wgpui_component::input::{InputEvent, InputState};
use wgpui_component::select::{SelectEvent, SelectState};
use wgpui_component::slider::{SliderEvent, SliderState};

use crate::brush::Brush;
use crate::canvas::Canvas;
use crate::guides;
use crate::lighting;
use crate::panel::KNOBS;
use crate::select::Dial;

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
    /// The Lighting shelf's five, in `lighting::DIALS`' order — see
    /// [`Controls::light`].
    lights: [Entity<SliderState>; lighting::DIALS.len()],
    /// The Guides shelf's two, in `guides::DIALS`' order.
    guide_dials: [Entity<SliderState>; guides::DIALS.len()],
    /// Which room the canvas is lit in, over `lighting::ENVIRONMENTS`' names.
    pub environment: Entity<SelectState<Vec<SharedString>>>,
    /// The color panel's notation field: what the picker stands on, as text a
    /// person can read, copy, or type over (`stark_ui::color::parse_color`).
    pub hex: Entity<InputState>,
    /// The menu bar's command search (`crate::palette`).
    pub search: Entity<InputState>,
    _subscriptions: Vec<Subscription>,
}

impl Controls {
    pub fn new(window: &mut Window, cx: &mut Context<'_, Canvas>) -> Self {
        let mut subs = Vec::new();
        let knobs = KNOBS.map(|knob| {
            let (lo, hi) = knob.range();
            let state = cx.new(|_| SliderState::new().min(lo).max(hi).step(knob.step()));
            subs.push(
                cx.subscribe(&state, move |this, _, event: &SliderEvent, cx| {
                    if let SliderEvent::Change(v) = event {
                        this.turn(knob, fraction(v.start(), lo, hi), cx);
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
        let lights = lighting::DIALS.map(|dial| {
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
        let guide_dials = guides::DIALS.map(|dial| {
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
        let lights_labels: Vec<SharedString> = lighting::ENVIRONMENTS
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
                if let Some((id, _)) = lighting::ENVIRONMENTS
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
            _subscriptions: subs,
        }
    }

    /// The state behind one of the Lighting shelf's tracks.
    pub fn light(&self, dial: lighting::Dial) -> &Entity<SliderState> {
        let i = lighting::DIALS
            .iter()
            .position(|d| *d == dial)
            .expect("every lighting dial has a state");
        &self.lights[i]
    }

    /// The state behind one of the Guides shelf's tracks.
    pub fn guide(&self, dial: guides::Dial) -> &Entity<SliderState> {
        let i = guides::DIALS
            .iter()
            .position(|d| *d == dial)
            .expect("every guide dial has a state");
        &self.guide_dials[i]
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
        } = what;
        for (knob, state) in KNOBS.iter().zip(&self.knobs) {
            settle(state, knob.read(brush), window, cx);
        }
        if let Some(o) = obs {
            for (dial, state) in DIALS.iter().zip(&self.dials) {
                settle(state, dial.read(o), window, cx);
            }
        }
        for (dial, state) in lighting::DIALS.iter().zip(&self.lights) {
            settle(state, dial.read(obs, hdr), window, cx);
        }
        // Only where there is a guide in hand: with none the tracks are not drawn at
        // all (`crate::guides`), and writing them would be settling a control nobody
        // can see onto a camera that does not exist.
        if let Some(g) = guide {
            for (dial, state) in guides::DIALS.iter().zip(&self.guide_dials) {
                settle(state, dial.read(&g), window, cx);
            }
        }
        settle(&self.opacity, opacity, window, cx);
        let want = obs
            .map(|o| o.environment)
            .and_then(|env| lighting::ENVIRONMENTS.iter().position(|(id, _)| *id == env))
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
}

/// Where `v` stands in `lo..=hi`, which is what the view's handlers speak.
fn fraction(v: f32, lo: f32, hi: f32) -> f32 {
    ((v - lo) / (hi - lo)).clamp(0.0, 1.0)
}

/// Set a slider to `v` if it does not already say so.
fn settle(state: &Entity<SliderState>, v: f32, window: &mut Window, cx: &mut Context<'_, Canvas>) {
    if (state.read(cx).value().start() - v).abs() > 1e-4 {
        state.update(cx, |s, cx| s.set_value(v, window, cx));
    }
}
