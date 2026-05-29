//! WebGPU surface rendering (DESIGN.md §6.4, §11).
//!
//! The engine renders directly into the canvas's `wgpu::Surface` texture — no
//! readback, no encode. A [`Renderer`] bundles the surface and the engine; the
//! app stores it in a signal and calls [`Renderer::paint`] after each command.

use stark_core::geom::Extent2;
use stark_core::{Engine, GpuContext, InputCommand, ObservableState, ViewTransform};
use wasm_bindgen::JsCast;

/// Canvas size in physical pixels (also the engine viewport).
pub const W: u32 = 900;
pub const H: u32 = 600;
/// Warm paper-white background, in linear RGB.
pub const BG: wgpu::Color = wgpu::Color {
    r: 0.92,
    g: 0.90,
    b: 0.85,
    a: 1.0,
};
pub const CANVAS_ID: &str = "stark-canvas";

/// Owns the canvas surface and the painting engine.
pub struct Renderer {
    surface: wgpu::Surface<'static>,
    engine: Engine,
}

impl Renderer {
    pub fn process(&mut self, command: InputCommand) {
        self.engine.process(command);
    }

    pub fn observe(&self) -> ObservableState {
        self.engine.observe()
    }

    pub fn view(&self) -> ViewTransform {
        self.engine.view()
    }

    /// Render the current canvas straight into the surface texture and present.
    pub fn paint(&mut self) {
        use wgpu::CurrentSurfaceTexture::{Success, Suboptimal};
        let frame = match self.surface.get_current_texture() {
            Success(frame) | Suboptimal(frame) => frame,
            // Timeout/Outdated/Lost/etc.: skip; the next command repaints.
            _ => return,
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.engine.render(&view, BG);
        frame.present();
    }
}

/// Look up the canvas element the app rendered into the DOM.
pub fn canvas_element() -> web_sys::HtmlCanvasElement {
    web_sys::window()
        .expect("window")
        .document()
        .expect("document")
        .get_element_by_id(CANVAS_ID)
        .expect("canvas element present")
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .expect("element is a canvas")
}

/// Asynchronously create the WebGPU device, configure the surface, and build the
/// engine (DESIGN.md §7: the frontend supplies GPU handles via `from_parts`).
pub async fn init(canvas: web_sys::HtmlCanvasElement) -> Renderer {
    canvas.set_width(W);
    canvas.set_height(H);

    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::BROWSER_WEBGPU;
    let instance = wgpu::Instance::new(desc);

    let surface: wgpu::Surface<'static> = instance
        .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
        .expect("create canvas surface");

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        })
        .await
        .expect("request adapter (WebGPU unavailable?)");

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("stark web device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("request device");

    // Pick a non-sRGB format: the media pass already encodes display sRGB, so an
    // sRGB surface would double-encode (DESIGN.md §6.5).
    let caps = surface.get_capabilities(&adapter);
    let format = caps
        .formats
        .iter()
        .copied()
        .find(|f| !f.is_srgb())
        .unwrap_or(caps.formats[0]);

    surface.configure(
        &device,
        &wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: W,
            height: H,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        },
    );

    let gpu = GpuContext::from_parts(instance, adapter, device, queue);
    let engine = Engine::new(gpu, format, Extent2::new(W, H));
    Renderer { surface, engine }
}
