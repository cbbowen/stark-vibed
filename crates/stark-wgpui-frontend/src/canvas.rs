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

use stark_chrome::brush_config::BrushEffectType;
use stark_chrome::input as chrome_input;
use stark_engine::ViewTransform;
use stark_engine::command::{GestureCommand, InputSample, Tool};
use stark_model::Vec2;
use wgpui::{
    AnyElement, Context, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, Pixels, Point,
    Render, Window, div, prelude::*, rgb, wgpu_surface,
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

/// What a press took hold of.
enum Held {
    /// A stroke on the canvas.
    Stroke,
    /// A knob on the panel, kept for the whole drag — see the module note.
    Knob(Knob),
}

pub struct Canvas {
    /// `None` when wgpui is not on its wgpu renderer, so there is no device to paint
    /// with — reported on screen rather than as a panic.
    renderer: Option<Renderer>,
    /// The tool in hand and the library it can be swapped for.
    brush: Brush,
    /// What the pointer is holding, if anything.
    held: Option<Held>,
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
    pub fn new(window: &mut Window, _cx: &mut Context<'_, Self>) -> Self {
        let mut renderer = Renderer::new(window);
        let brush = Brush::default();
        if let Some(r) = renderer.as_mut() {
            r.process(brush.set());
        }
        let clock = quanta::Clock::new();
        let epoch = clock.raw();
        Self {
            renderer,
            brush,
            held: None,
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
