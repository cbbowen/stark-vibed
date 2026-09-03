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
//! It has found two things so far. [`GpuContext`] used to carry a `wgpu::Instance`
//! and `wgpu::Adapter` that the engine never read, and wgpui hands out neither — so
//! the two moved to the side that actually uses them, which is the web frontend's own
//! `Renderer` and its three canvases. And the device wgpui built asked for
//! `wgpu::Limits::default()`, which is four storage textures per stage where the
//! Mixbox stamp loop writes six (§6.7) — so a whole colour space was unreachable
//! here for want of a number nobody could state. [`device_descriptor`] is where it is
//! stated now, over a vendored patch that threads it down (`vendor/wgpui`).
//!
//! # What is here and what is not
//!
//! Five modules. Three are the frame, in the order it moves through them: [`brush`]
//! is the one tool, [`render`] is the surface the engine paints into, and [`canvas`]
//! is the view that turns mouse events into a gesture. Two are what this frontend
//! *keeps*: [`store`] is where a record goes on this platform — the native half of
//! `stark_chrome::storage` — and [`window`] is the one record that is nobody else's.
//!
//! There is still no chrome, no document state and no panels. The plan from here is
//! §11.2.

mod brush;
mod canvas;
mod render;
mod store;
mod window;

use stark_engine::GpuContext;
use wgpui::{App, Application, TitlebarOptions, WindowOptions, prelude::*};

use crate::canvas::Canvas;

/// What the engine needs of the device, raised over what wgpui needs of it.
///
/// **The frontend states this because it is the only thing that can.** wgpui creates
/// the device (§11.1) and every element in the window draws with it, so there is one
/// descriptor for both consumers and neither of them can write it alone. Before the
/// vendored patch that threads this down, wgpui wrote it — as
/// [`Limits::default()`](wgpu::Limits::default) — and the engine got whatever that
/// happened to cover, which was not the Mixbox stamp loop's six storage textures
/// (§6.7).
///
/// `or_better_values_from` rather than either side's limits alone, and the direction
/// matters: starting from `Limits::default()` keeps every limit wgpui's renderer was
/// written against, and raises only the fields where
/// [`minimum_required_limits`](GpuContext::minimum_required_limits) asks for more.
/// Handing the engine's minimum over on its own would *lower* the rest to the
/// downlevel floor it is built on and take wgpui's own headroom with it.
///
/// The engine's half of that answer is `#[cfg]`-dependent — six storage textures with
/// the Mixbox feature, four without — so this asks for exactly what the build in hand
/// can use, with no second copy of the number here to drift. The same line as the web
/// frontend's `render::init`, for the same reason.
fn device_descriptor() -> wgpu::DeviceDescriptor<'static> {
    wgpu::DeviceDescriptor {
        label: Some("stark native device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default()
            .or_better_values_from(&GpuContext::minimum_required_limits()),
        experimental_features: wgpu::ExperimentalFeatures::default(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }
}

fn main() {
    // Before anything reads a record, say where records go (§11.2). Two directories,
    // and `None` where the platform will not name them — which the format already
    // treats as "nothing stored" rather than as a failure, so there is nothing to
    // handle here beyond declining to install.
    if let Some(files) = store::Files::resolve() {
        stark_chrome::storage::install(files);
    }

    Application::new(&device_descriptor()).run(|cx: &mut App| {
        // Read before `open_window` borrows `cx`, not inside its argument list.
        let bounds = window::opening(cx);
        cx.open_window(
            WindowOptions {
                titlebar: Some(TitlebarOptions {
                    title: Some("Stark".into()),
                    ..Default::default()
                }),
                window_bounds: Some(bounds),
                ..Default::default()
            },
            |window, cx| {
                // Where the window ends up is worth keeping, and the moment it is
                // *worth writing* is once: a record saved as the bounds change would
                // be a file written per frame of a resize drag.
                window.on_window_should_close(cx, |window, _cx| {
                    window::remember(window.window_bounds());
                    true
                });
                cx.new(|cx| Canvas::new(window, cx))
            },
        )
        .expect("open the painting window");
        cx.activate(true);
    });
}
