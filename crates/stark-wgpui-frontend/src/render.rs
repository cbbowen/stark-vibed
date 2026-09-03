//! The wgpui surface the engine paints into (§6.4, §11).
//!
//! The web frontend's `Renderer` builds the device, binds a `wgpu::Surface` to a
//! `<canvas>` and configures it. None of that happens here: **wgpui owns the device**
//! and hands out a double-buffered pair of textures instead, so what this holds is a
//! [`WgpuSurfaceHandle`] and an [`Engine`] built on the device behind it. The engine
//! renders straight into the back buffer and the swap is a pointer swap — the same
//! bargain the browser canvas makes, with no readback and no encode.

use stark_engine::command::{InputCommand, ViewCommand};
use stark_engine::{Engine, GpuContext, ObservableState, ViewTransform};
use stark_model::geom::Extent2;
use wgpui::{WgpuSurfaceHandle, Window};

/// The format the engine renders through, and so the format the surface's two
/// buffers carry.
///
/// **Not an sRGB format**: the media pass already encodes display sRGB (§6.5), so an
/// sRGB target would encode it twice. wgpui picks its own window format by the same
/// test (`!f.is_srgb()`), so what this writes reaches the screen unconverted.
pub const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// One surface, one engine, and the resize that keeps them agreeing.
pub struct Renderer {
    surface: WgpuSurfaceHandle,
    engine: Engine,
    /// The viewport the engine was last told about, in device px — see
    /// [`paint`](Self::paint), which is where it is corrected.
    viewport: (u32, u32),
}

impl Renderer {
    /// Bind a surface to `window` and build an engine on the device behind it.
    ///
    /// `None` when wgpui is not on its wgpu renderer, which is the one platform
    /// answer that leaves nothing to paint with — the view reports it rather than
    /// panicking, for the reason the web frontend's `StartupFailure` gives.
    pub fn new(window: &Window) -> Option<Self> {
        let (width, height) = device_pixels(window);
        let surface = window.create_wgpu_surface(width, height, TARGET_FORMAT)?;
        // The engine is *given* its wgpu resources (CLAUDE.md), and here that is
        // forced rather than chosen: the handle carries a device and a queue and
        // nothing else. It also replaces wgpui's device callbacks with the engine's
        // (`GpuContext::from_parts`), which is the right way round — the engine is
        // what has to stop issuing work when the device dies.
        let gpu = GpuContext::from_parts(surface.device().clone(), surface.queue().clone());
        let engine = Engine::new(gpu, TARGET_FORMAT, Extent2::new(width, height));
        Some(Self {
            surface,
            engine,
            viewport: (width, height),
        })
    }

    /// Send a command to the engine — the **only** way to move engine state through a
    /// `Renderer` (§4), deliberately, and for the reason the web frontend's
    /// `Renderer::process` spells out: a named `set_*` beside it is a second spelling
    /// that skips whatever the first one also did.
    pub fn process(&mut self, command: impl Into<InputCommand>) {
        self.engine.process(command);
    }

    /// The engine's cheap UI-facing projection, read back after each command (§5).
    ///
    /// Cheap by construction — the layer roster is shared rather than copied — but
    /// not free, so the view keeps the answer rather than asking per frame.
    pub fn observe(&self) -> ObservableState {
        self.engine.observe()
    }

    /// The view a pointer position is mapped through.
    pub fn view(&self) -> ViewTransform {
        self.engine.view()
    }

    /// The handle the element composites. Cloned per frame, which costs two atomic
    /// bumps — the element wants it by value.
    pub fn surface(&self) -> WgpuSurfaceHandle {
        self.surface.clone()
    }

    /// Whether the element has resized the surface out from under the viewport the
    /// engine was last told about.
    ///
    /// A repaint the frame owes even though no command asked for one: the resize
    /// happens in the element's *prepaint*, so the frame that caused it has already
    /// drawn the old picture stretched to the new size.
    pub fn resized(&self) -> bool {
        self.surface.size() != self.viewport
    }

    /// Render the canvas into the back buffer and swap it to the front.
    ///
    /// `swap_buffers` rather than `present`: this runs inside a frame wgpui is
    /// already building, so the swap is what that frame composites, and the
    /// `request_present` the other half of `present` would add asks for a further
    /// frame nobody needs.
    pub fn paint(&mut self) {
        // Both at once, so a concurrent resize cannot hand back a view of one size
        // and dimensions of another.
        let Some((target, size)) = self.surface.back_view_with_size() else {
            return;
        };
        // The viewport is taken from the buffer rather than from the window, because
        // the buffer is what the render lands in: a view that disagreed with its
        // target would put every stroke somewhere other than under the pointer.
        if size != self.viewport {
            self.viewport = size;
            self.engine
                .process(ViewCommand::Resize(Extent2::new(size.0, size.1)));
        }
        self.engine.render(&target);
        self.surface.swap_buffers();
    }
}

/// The window's drawable area in **device px** — what a surface is sized in, where
/// every [`Pixels`](wgpui::Pixels) the layout speaks in is logical.
fn device_pixels(window: &Window) -> (u32, u32) {
    let scale = window.scale_factor();
    let size = window.viewport_size();
    // Floored at 1: a zero-sized surface is not a texture, and a minimized window
    // is an ordinary state rather than a failure.
    let px = |v: wgpui::Pixels| ((f32::from(v) * scale).round() as u32).max(1);
    (px(size.width), px(size.height))
}
