//! WebGPU surface rendering (DESIGN.md §6.4, §11).
//!
//! The engine renders directly into the canvas's `wgpu::Surface` texture — no
//! readback, no encode. A [`Renderer`] bundles the surface and the engine; the
//! app stores it in a signal, calls [`Renderer::paint`] after each command, and
//! [`Renderer::resize`] when the canvas (window) changes size.

use stark_core::geom::Extent2;
use stark_core::{
    ColorSpaceId, Engine, GpuContext, InputCommand, ObservableState, SurfaceId, ViewTransform,
};
use wasm_bindgen::JsCast;

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
    canvas: web_sys::HtmlCanvasElement,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    engine: Engine,
    /// The built-in bristle brush, imported once its bytes are fetched (§6.6).
    bristle: Option<stark_core::AssetId>,
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

    /// The document's current color space (DESIGN.md §6.7).
    pub fn color_space(&self) -> ColorSpaceId {
        self.engine.color_space()
    }

    /// Start a fresh document in `id`'s color space. Clears the canvas, since the
    /// channel layouts differ between spaces (DESIGN.md §6.7).
    pub fn set_color_space(&mut self, id: ColorSpaceId) {
        self.engine.set_color_space(id);
    }

    /// The document's current canvas surface (DESIGN.md §6.4).
    pub fn surface(&self) -> SurfaceId {
        self.engine.surface()
    }

    /// Switch the canvas surface (resets the document — it affects deposition).
    pub fn set_surface(&mut self, id: SurfaceId) {
        self.engine.set_surface(id);
    }

    /// Whether a surface's bytes are loaded (Flat always is).
    pub fn surface_loaded(&self, id: SurfaceId) -> bool {
        self.engine.surface_loaded(id)
    }

    /// Register frontend-fetched image bytes for a surface (DESIGN.md §6.4).
    pub fn register_surface(&mut self, id: SurfaceId, png_bytes: Vec<u8>) {
        self.engine.register_surface(id, png_bytes);
    }

    /// The built-in bristle brush's asset id, once its bytes have been imported.
    pub fn bristle(&self) -> Option<stark_core::AssetId> {
        self.bristle
    }

    /// Import the built-in bristle brush from fetched bytes, caching its id.
    pub fn load_bristle(&mut self, png_bytes: &[u8]) {
        match self.engine.import_brush(png_bytes) {
            Ok(id) => self.bristle = Some(id),
            Err(e) => tracing::warn!("bristle import failed: {e}"),
        }
    }

    /// Match the surface + engine viewport to a new canvas size (CSS pixels).
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 || (width == self.config.width && height == self.config.height)
        {
            return;
        }
        self.canvas.set_width(width);
        self.canvas.set_height(height);
        self.config.width = width;
        self.config.height = height;
        self.surface
            .configure(&self.engine.gpu().device, &self.config);
        self.engine.resize(Extent2::new(width, height));
    }

    /// Render the current canvas straight into the surface texture and present.
    pub fn paint(&mut self) {
        use wgpu::CurrentSurfaceTexture::{Suboptimal, Success};
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

/// The canvas element's laid-out size in CSS pixels (≥1). Measures the element
/// itself — no full-window assumption — so an embedded/sub-window canvas works.
fn canvas_size(canvas: &web_sys::HtmlCanvasElement) -> (u32, u32) {
    (
        canvas.client_width().max(1) as u32,
        canvas.client_height().max(1) as u32,
    )
}

/// Await one animation frame, so a layout pass (and any just-applied stylesheet)
/// is reflected before we measure the canvas.
async fn next_frame() {
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        web_sys::window()
            .expect("window")
            .request_animation_frame(&resolve)
            .expect("request_animation_frame");
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
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

/// Asynchronously create the WebGPU device, configure the surface to the
/// canvas's current size, and build the engine (DESIGN.md §7).
pub async fn init(canvas: web_sys::HtmlCanvasElement) -> Renderer {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::BROWSER_WEBGPU;
    let instance = wgpu::Instance::new(desc);

    let surface: wgpu::Surface<'static> = instance
        .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
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
            required_limits: wgpu::Limits::default()
                .or_better_values_from(&GpuContext::minimum_required_limits()),
            experimental_features: wgpu::ExperimentalFeatures::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("request device");

    // Size the drawing buffer to the canvas's laid-out size (CSS pixels). We
    // measure the *element*, not the window, so an embedded/sub-window canvas
    // works too. Crucially we do it here — after the async device setup and a
    // layout frame — so the stylesheet (linked via `document::Stylesheet`) has
    // applied: measuring up front would read the unstyled 300×150 intrinsic size,
    // and Dioxus `onresize` only delivers *later* resizes, so it wouldn't correct
    // the seed. Subsequent resizes are handled by `onresize`.
    next_frame().await;
    let (width, height) = canvas_size(&canvas);
    canvas.set_width(width);
    canvas.set_height(height);

    // Pick a non-sRGB format: the media pass already encodes display sRGB, so an
    // sRGB surface would double-encode (DESIGN.md §6.5).
    let caps = surface.get_capabilities(&adapter);
    let format = caps
        .formats
        .iter()
        .copied()
        .find(|f| !f.is_srgb())
        .unwrap_or(caps.formats[0]);

    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width,
        height,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &config);

    let gpu = GpuContext::from_parts(instance, adapter, device, queue);
    let engine = Engine::new(gpu, format, Extent2::new(width, height));
    Renderer {
        canvas,
        surface,
        config,
        engine,
        bristle: None,
    }
}
