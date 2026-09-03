//! The window's one view: the brush panel, the engine's canvas beside it, and the
//! mouse gestures that drive both (§6.2, §11).
//!
//! One view rather than two, and that is a decision rather than an economy. A press
//! on a slider and a press on the canvas are the same event arriving at the same
//! place, and which of them it is depends on where the panel *ends* — so the split
//! lives in one hit test ([`panel::hit`]) instead of in two elements racing for the
//! pointer. It is also what lets a slider drag keep working once the pointer has left
//! the track, which an element handler cannot do: wgpui gates `on_mouse_move` on the
//! hitbox, so an element that loses the pointer stops hearing about it.
//!
//! What the hit test tests against is the layout wgpui actually produced, not one
//! this side derived — see `panel::Regions`.

use stark_chrome::brush_config::{BrushEffectType, MAX_FLOW, MAX_RADIUS, MIN_RADIUS};
use stark_chrome::commands::{Bindings, Command};
use stark_chrome::drags::{DragAction, DragBindings, DragButton, DragChord};
use stark_chrome::input as chrome_input;
use stark_chrome::keys::Mods;
use stark_engine::ViewTransform;
use stark_engine::command::{DocCommand, GestureCommand, InputSample, Tool};
use stark_model::Vec2;
use wgpui::{
    AnyElement, Context, FocusHandle, KeyDownEvent, MouseButton, MouseDownEvent, MouseMoveEvent,
    MouseUpEvent, Pixels, Point, Render, Window, div, prelude::*, rgb, wgpu_surface,
};

use crate::brush::Brush;
use crate::panel::{self, Knob, Region, Regions};
use crate::render::Renderer;

/// The effects the panel offers, with the word each wears.
///
/// All four of the model's, minus nothing: an eraser is a brush whose effect is
/// `Erase` and a blur is one with `bleed` up (§6.12), so this is the whole tool
/// vocabulary rather than a selection from it.
const EFFECTS: &[(BrushEffectType, &str)] = &[
    (BrushEffectType::Paint, "Paint"),
    (BrushEffectType::Wet, "Wet"),
    (BrushEffectType::Erase, "Erase"),
    (BrushEffectType::Liquify, "Liquify"),
];

/// How far one press of the bracket keys moves the brush's size, as a factor.
///
/// A ratio rather than a step, because size is perceived logarithmically: a pixel
/// added to a 4-px tip is a quarter of it and nothing at all to a 400-px one. The same
/// figure the web frontend steps by.
const SIZE_STEP: f32 = 1.1;

/// How much of the size range one horizontal pixel of a tuning drag spends, as an
/// exponent — so the knob moves multiplicatively, for `SIZE_STEP`'s reason.
///
/// `MIN_RADIUS..MAX_RADIUS` is about nine doublings, and this spends them over some
/// 900 px: a drag across a window covers the range once, and a short one is fine.
const TUNE_SIZE_PER_PX: f32 = 0.007;

/// How much flow one vertical pixel spends — the range over some 300 px, which is
/// shorter because there is far less of it to cross.
const TUNE_FLOW_PER_PX: f32 = 0.01;

/// What a press took hold of.
enum Held {
    /// A stroke on the canvas.
    Stroke,
    /// A knob on the panel, kept for the whole drag — see the module note.
    Knob(Knob),
    /// A **bound modifier drag** over the canvas (§18.1.9): the size sideways, the
    /// flow up and down, from where the press landed.
    Tune {
        /// Where the drag started, in logical px, and the tune it started from — so
        /// a long drag is one map from the press rather than a chain of steps, which
        /// is `stark_chrome::transform`'s rule applied to a knob.
        from: Point<Pixels>,
        size: f32,
        flow: f32,
    },
}

pub struct Canvas {
    /// `None` when wgpui is not on its wgpu renderer, so there is no device to paint
    /// with — reported on screen rather than as a panic.
    renderer: Option<Renderer>,
    /// The tool in hand and the library it can be swapped for.
    brush: Brush,
    /// What the pointer is holding, if anything.
    held: Option<Held>,
    /// This browser's chord table, and the drag table beside it — both shipped
    /// defaults for now: rebinding needs a settings surface, which is N8's.
    bindings: Bindings,
    drags: DragBindings,
    /// The keyboard needs somewhere to be focused, or nothing is dispatched at all.
    focus: FocusHandle,
    /// Whether the canvas owes the surface a frame.
    dirty: bool,
    /// Where the panel's controls were laid out, as of the last painted frame — the
    /// panel measures rather than predicts, and this is where it reports (`panel`).
    regions: Regions,
    /// The clock `InputSample::time` is read off, and the raw reading it counts
    /// from. `quanta` rather than `std::time::Instant` (clippy.toml): this binary
    /// is native-only, but the clock the rest of the tree uses is the one that
    /// works in both places, and it is already compiled here through the engine's
    /// timing layer (§7.1).
    clock: quanta::Clock,
    epoch: u64,
}

impl Canvas {
    pub fn new(window: &mut Window, cx: &mut Context<'_, Self>) -> Self {
        let mut renderer = Renderer::new(window);
        let brush = Brush::default();
        if let Some(r) = renderer.as_mut() {
            r.process(brush.set());
        }
        let clock = quanta::Clock::new();
        let epoch = clock.raw();
        let focus = cx.focus_handle();
        Self {
            renderer,
            brush,
            held: None,
            bindings: Bindings::default(),
            drags: DragBindings::default(),
            focus,
            // The first frame has a canvas nobody has painted yet.
            dirty: true,
            regions: Regions::default(),
            clock,
            epoch,
        }
    }

    fn press(&mut self, ev: &MouseDownEvent, window: &mut Window, cx: &mut Context<'_, Self>) {
        // The panel first: its column is where a press stops being paint.
        match panel::hit(&self.regions, ev.position) {
            Some(Region::Knob(knob)) => {
                self.held = Some(Held::Knob(knob));
                if let Some(f) = panel::fraction_at(&self.regions, knob, ev.position) {
                    self.turn(knob, f, cx);
                }
                return;
            }
            Some(Region::Effect(i)) => {
                if let Some((effect, _)) = EFFECTS.get(i) {
                    self.brush.config.effect = *effect;
                    self.brush.tuned_off_preset();
                    self.send_brush(cx);
                }
                return;
            }
            Some(Region::Preset(i)) => {
                if let Some(name) = self.brush.library.get(i).map(|e| e.name.clone()) {
                    self.brush.wear(&name);
                    self.send_brush(cx);
                }
                return;
            }
            None if panel::within(ev.position) => {
                // Somewhere on the panel that is not a control. Not paint either.
                return;
            }
            None => {}
        }

        // The drag table before the paint path, exactly as the web canvas asks it:
        // a modified press is a *gesture*, and which one is the table's answer rather
        // than a ladder of modifier tests here (§25.3).
        let mods = Mods {
            ctrl: ev.modifiers.control || ev.modifiers.platform,
            shift: ev.modifiers.shift,
            alt: ev.modifiers.alt,
        };
        let chord = DragChord {
            mods,
            button: DragButton::Left,
        };
        if self.drags.lookup(mods, DragButton::Left) == Some(DragAction::TuneBrush) {
            let _ = chord;
            self.held = Some(Held::Tune {
                from: ev.position,
                size: self.brush.tune.size,
                flow: self.brush.tune.flow,
            });
            return;
        }

        let (scale, now) = (window.scale_factor(), self.elapsed());
        let smoothing = self.brush.config.smoothing;
        let Some(r) = self.renderer.as_mut() else {
            return;
        };
        let view = r.view();
        r.process(GestureCommand::Start {
            tool: Tool::Brush,
            sample: sample_at(view, ev.position, scale, now),
            // Both are canvas-space lengths the frontend alone can state, and both
            // are mapped by `stark_chrome::input` rather than here — which is the
            // point of that module: this frontend had its own copy of the rope's
            // constant and its own quadratic for exactly one commit (§11.2).
            //
            // The resolution is a *mouse's*, in this surface's device px: winit gives
            // no pen, so there is nothing finer to report yet.
            tolerance: chrome_input::tolerance(view, chrome_input::MOUSE_RESOLUTION),
            rope: chrome_input::rope(view, smoothing),
        });
        self.held = Some(Held::Stroke);
        self.repaint(cx);
    }

    fn drag(&mut self, ev: &MouseMoveEvent, window: &mut Window, cx: &mut Context<'_, Self>) {
        match self.held {
            Some(Held::Knob(knob)) => {
                // Recomputed from the pointer's x alone, so a drag that has wandered
                // off the track vertically still moves the knob it took hold of.
                if let Some(f) = panel::fraction_at(&self.regions, knob, ev.position) {
                    self.turn(knob, f, cx);
                }
            }
            Some(Held::Tune { from, size, flow }) => {
                // Both knobs from the press rather than from the last move, so a long
                // drag is one map and rounding cannot walk over its length.
                let dx = f32::from(ev.position.x) - f32::from(from.x);
                let dy = f32::from(ev.position.y) - f32::from(from.y);
                self.brush.tune.size =
                    (size * (TUNE_SIZE_PER_PX * dx).exp()).clamp(MIN_RADIUS, MAX_RADIUS);
                // Up is more, which is the direction every slider in the app grows in
                // and the opposite of the screen's y.
                self.brush.tune.flow = (flow - dy * TUNE_FLOW_PER_PX).clamp(0.0, MAX_FLOW);
                self.send_brush(cx);
            }
            Some(Held::Stroke) => {
                let (scale, now) = (window.scale_factor(), self.elapsed());
                let Some(r) = self.renderer.as_mut() else {
                    return;
                };
                let view = r.view();
                r.process(GestureCommand::To {
                    sample: sample_at(view, ev.position, scale, now),
                });
                self.repaint(cx);
            }
            None => {}
        }
    }

    /// End whatever the press took hold of — for a stroke, the one edge that commits
    /// an action (§4).
    fn release(&mut self, _ev: &MouseUpEvent, _window: &mut Window, cx: &mut Context<'_, Self>) {
        if matches!(self.held.take(), Some(Held::Stroke))
            && let Some(r) = self.renderer.as_mut()
        {
            r.process(GestureCommand::End);
        }
        self.repaint(cx);
    }

    /// Run whatever the shipped chord table says this keystroke asks for.
    ///
    /// **The whole of the keyboard, and it is nine lines**, because the table is
    /// shared: what Ctrl+Z means was settled once (§25) and this frontend only has to
    /// say what a keystroke *is* (`crate::keys`) and what an act *does* below.
    fn key(&mut self, ev: &KeyDownEvent, _window: &mut Window, cx: &mut Context<'_, Self>) {
        let Some(command) = self.bindings.lookup(&crate::keys::stroke(&ev.keystroke)) else {
            return;
        };
        self.run(command, cx);
    }

    /// Do what a command means here.
    ///
    /// A short list, and short *honestly*: the registry has thirty-odd acts and this
    /// frontend has five of them. What the others need is a
    /// surface — a dialog, a layer list, a selection — and each arrives with its own
    /// stage (§11.2). An act with nothing to act on is left alone rather than given a
    /// no-op arm, so the day it lands the compiler has nothing to say and the reader
    /// does.
    fn run(&mut self, command: Command, cx: &mut Context<'_, Self>) {
        let doc = match command {
            Command::Undo => Some(DocCommand::Undo),
            Command::Redo => Some(DocCommand::Redo),
            _ => None,
        };
        match command {
            Command::BrushSmaller => self.step_size(1.0 / SIZE_STEP, cx),
            Command::BrushLarger => self.step_size(SIZE_STEP, cx),
            _ => {}
        }
        if let Some(doc) = doc
            && let Some(r) = self.renderer.as_mut()
        {
            r.process(doc);
            self.repaint(cx);
        }
    }

    /// Step the brush's size by a factor, clamped to the range the panel offers.
    fn step_size(&mut self, factor: f32, cx: &mut Context<'_, Self>) {
        self.brush.tune.size = (self.brush.tune.size * factor).clamp(MIN_RADIUS, MAX_RADIUS);
        self.send_brush(cx);
    }

    /// Move a knob and put the changed brush in the engine's hand.
    fn turn(&mut self, knob: Knob, fraction: f32, cx: &mut Context<'_, Self>) {
        if panel::drag_knob(&mut self.brush, knob, fraction) {
            self.brush.tuned_off_preset();
        }
        self.send_brush(cx);
    }

    fn send_brush(&mut self, cx: &mut Context<'_, Self>) {
        let command = self.brush.set();
        if let Some(r) = self.renderer.as_mut() {
            r.process(command);
        }
        // The canvas does not change until the next stroke, but the panel does, and
        // both are this one view.
        self.repaint(cx);
    }

    /// Seconds since the window opened, for `InputSample::time` — which the stroke
    /// dynamics read as velocity and the timelapse (§8) replays against.
    fn elapsed(&self) -> f64 {
        self.clock.delta(self.epoch, self.clock.raw()).as_secs_f64()
    }

    /// Note that the frame has changed. `notify` schedules it; `dirty` is what that
    /// frame reads to decide whether the *engine* has to render at all.
    fn repaint(&mut self, cx: &mut Context<'_, Self>) {
        self.dirty = true;
        cx.notify();
    }
}

impl Render for Canvas {
    fn render(&mut self, window: &mut Window, cx: &mut Context<'_, Self>) -> impl IntoElement {
        // **The frame after a resize exists only because of this.** The surface
        // element resizes its textures during prepaint, which is after this runs, so
        // the new size is first visible here one frame later — and a window resize
        // schedules no further frame of its own. Cheap to ask for: the tree is small,
        // and the engine renders only when something has actually changed.
        window.request_animation_frame();

        let dragging = match self.held {
            Some(Held::Knob(k)) => Some(k),
            _ => None,
        };
        let chrome = panel::brush_panel(&self.brush, dragging, EFFECTS, &self.regions);

        let Some(r) = self.renderer.as_mut() else {
            return unavailable();
        };
        if self.dirty || r.resized() {
            r.paint();
            self.dirty = false;
        }
        div()
            .size_full()
            .flex()
            // The keyboard answers whatever has focus, and nothing has it unless
            // something asks: an unfocused window dispatches no chord at all.
            .track_focus(&self.focus)
            .on_key_down(cx.listener(Self::key))
            .child(chrome)
            .child(
                div()
                    .flex_1()
                    .h_full()
                    .child(wgpu_surface(r.surface()).size_full()),
            )
            .on_mouse_down(MouseButton::Left, cx.listener(Self::press))
            .on_mouse_move(cx.listener(Self::drag))
            .on_mouse_up(MouseButton::Left, cx.listener(Self::release))
            // A release the view never saw still ends what it was holding: the
            // pointer can leave the window mid-drag, and the alternative is a gesture
            // the engine never closes and a knob that keeps following the mouse.
            .on_mouse_up_out(MouseButton::Left, cx.listener(Self::release))
            .into_any_element()
    }
}

/// A pointer position mapped to a canvas-space sample.
///
/// `position` is window-relative and **logical**, while the view is denominated in
/// the surface's device px — so the scale factor is the whole of the conversion.
///
/// The panel's width comes off first: the surface begins where the panel ends, and
/// `screen_to_canvas` maps out of the *surface's* space rather than the window's.
fn sample_at(view: ViewTransform, position: Point<Pixels>, scale: f32, time: f64) -> InputSample {
    let x = f32::from(position.x) - panel::WIDTH;
    let screen = Vec2::new(x * scale, f32::from(position.y) * scale);
    InputSample {
        pos: view.screen_to_canvas(screen),
        // A mouse is always pressed home (`ModSource::Pressure`), and reports no
        // tilt at all.
        pressure: 1.0,
        tilt: Vec2::ZERO,
        time,
    }
}

/// What the window shows when there is no wgpu device to paint with.
fn unavailable() -> AnyElement {
    div()
        .size_full()
        .flex()
        .items_center()
        .justify_center()
        .bg(rgb(0x1b1b1b))
        .text_color(rgb(0xd0d0d0))
        .child("wgpui is not using its wgpu renderer here, so there is no canvas to paint on.")
        .into_any_element()
}
