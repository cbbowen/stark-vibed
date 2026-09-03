//! Stark's native frontend (§11): a wgpui window with the engine's canvas in it.
//!
//! **A second frontend over the same [`stark_engine`], which is why it exists.**
//! "Dependencies point one way: frontend → engine → model" (CLAUDE.md) was a claim
//! about a tree with one frontend in it, and a claim of that shape is only tested by
//! a second consumer. What this one proves is deliberately small — a canvas that
//! renders and a brush that paints on it — but it is proved against a toolkit that
//! shares nothing with the first: no DOM, no browser, no `<canvas>`, and a wgpu
//! device this frontend does not own.
//!
//! It found one thing already. [`GpuContext`](stark_engine::GpuContext) used to carry
//! a `wgpu::Instance` and `wgpu::Adapter` that the engine never read, and wgpui hands
//! out neither — so the two moved to the side that actually uses them, which is the
//! web frontend's own `Renderer` and its three canvases.
//!
//! # What is here and what is not
//!
//! Three modules, in the order a frame moves through them: [`brush`] is the one tool,
//! [`render`] is the surface the engine paints into, and [`canvas`] is the view that
//! turns mouse events into a gesture. There is no chrome, no document state, no
//! panels — none of that is what a second frontend has to prove first.

mod brush;
mod canvas;
mod render;

use wgpui::{
    App, Application, Bounds, TitlebarOptions, WindowBounds, WindowOptions, prelude::*, px, size,
};

use crate::canvas::Canvas;

fn main() {
    Application::new().run(|cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(1280.), px(800.)), cx);
        cx.open_window(
            WindowOptions {
                titlebar: Some(TitlebarOptions {
                    title: Some("Stark".into()),
                    ..Default::default()
                }),
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |window, cx| cx.new(|cx| Canvas::new(window, cx)),
        )
        .expect("open the painting window");
        cx.activate(true);
    });
}
