//! The window's one view: the engine's canvas, and the mouse gesture that paints on
//! it (§6.2, §11).

use stark_engine::ViewTransform;
use stark_engine::command::{GestureCommand, InputSample, Tool, ViewCommand};
use stark_model::Vec2;
use wgpui::{
    AnyElement, Context, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, Pixels, Point,
    Render, Window, div, prelude::*, rgb, wgpu_surface,
};

use crate::brush;
use crate::render::Renderer;

/// How finely a mouse resolves position, in **device px**: it walks the screen in
/// whole physical pixels, so one is its floor. What the fitter needs in order to
/// tell jitter from detail (`GestureCommand::Start::tolerance`); a pen would report
/// finer, and this frontend has no pen.
const MOUSE_RESOLUTION_PX: f32 = 1.0;

/// The longest smoothing string a brush can ask for, in **device px** — what
/// `SMOOTHING = 1` would mean (§6.11). Screen-denominated because wobble is a fact
/// about the hand: the same tremor spans 64× more canvas zoomed out than in.
const ROPE_MAX_SCREEN_PX: f32 = 160.0;

pub struct Canvas {
    /// `None` when wgpui is not on its wgpu renderer, so there is no device to paint
    /// with — reported on screen rather than as a panic.
    renderer: Option<Renderer>,
    /// Whether a stroke is in flight, so a move extends only a press this view saw.
    drawing: bool,
    /// Whether the canvas owes the surface a frame.
    dirty: bool,
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
        if let Some(r) = renderer.as_mut() {
            r.process(ViewCommand::SetBrush {
                brush: brush::hard_round(),
                color: brush::INK,
            });
        }
        let clock = quanta::Clock::new();
        let epoch = clock.raw();
        Self {
            renderer,
            drawing: false,
            // The first frame has a canvas nobody has painted yet.
            dirty: true,
            clock,
            epoch,
        }
    }

    fn press(&mut self, ev: &MouseDownEvent, window: &mut Window, cx: &mut Context<'_, Self>) {
        let (scale, now) = (window.scale_factor(), self.elapsed());
        let Some(r) = self.renderer.as_mut() else {
            return;
        };
        let view = r.view();
        r.process(GestureCommand::Start {
            tool: Tool::Brush,
            sample: sample_at(view, ev.position, scale, now),
            // Both are canvas-space lengths the frontend alone can state, and both
            // get there the same way: they are screen quantities, so they divide by
            // the zoom to reach the space the fit measures its error in.
            tolerance: MOUSE_RESOLUTION_PX / view.zoom,
            rope: brush::SMOOTHING * brush::SMOOTHING * ROPE_MAX_SCREEN_PX / view.zoom,
        });
        self.drawing = true;
        self.repaint(cx);
    }

    fn drag(&mut self, ev: &MouseMoveEvent, window: &mut Window, cx: &mut Context<'_, Self>) {
        if !self.drawing {
            return;
        }
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

    /// End the stroke — the one edge that commits an action (§4).
    fn release(&mut self, _ev: &MouseUpEvent, _window: &mut Window, cx: &mut Context<'_, Self>) {
        if !self.drawing {
            return;
        }
        self.drawing = false;
        if let Some(r) = self.renderer.as_mut() {
            r.process(GestureCommand::End);
        }
        self.repaint(cx);
    }

    /// Seconds since the window opened, for `InputSample::time` — which the stroke
    /// dynamics read as velocity and the timelapse (§8) replays against.
    fn elapsed(&self) -> f64 {
        self.clock.delta(self.epoch, self.clock.raw()).as_secs_f64()
    }

    /// Note that the canvas has changed. `notify` schedules the frame; `dirty` is
    /// what that frame reads to decide whether the engine has to render at all.
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
        // schedules no further frame of its own. Cheap to ask for: the tree is one
        // element, and the engine renders only when something has actually changed.
        window.request_animation_frame();

        let Some(r) = self.renderer.as_mut() else {
            return unavailable();
        };
        if self.dirty || r.resized() {
            r.paint();
            self.dirty = false;
        }
        div()
            .size_full()
            .child(wgpu_surface(r.surface()).absolute().inset_0())
            .on_mouse_down(MouseButton::Left, cx.listener(Self::press))
            .on_mouse_move(cx.listener(Self::drag))
            .on_mouse_up(MouseButton::Left, cx.listener(Self::release))
            // A release the canvas never saw still ends the stroke: the pointer can
            // leave the window mid-drag, and the alternative is a gesture the engine
            // never closes.
            .on_mouse_up_out(MouseButton::Left, cx.listener(Self::release))
            .into_any_element()
    }
}

/// A pointer position mapped to a canvas-space sample.
///
/// `position` is window-relative and **logical**, while the view is denominated in
/// the surface's device px — the canvas fills the window's drawable area, so the
/// scale factor is the whole of the conversion.
fn sample_at(view: ViewTransform, position: Point<Pixels>, scale: f32, time: f64) -> InputSample {
    let screen = Vec2::new(f32::from(position.x) * scale, f32::from(position.y) * scale);
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
