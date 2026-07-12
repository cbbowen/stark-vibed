//! WebGPU surface rendering (DESIGN.md §6.4, §11).
//!
//! The engine renders directly into the canvas's `wgpu::Surface` texture — no
//! readback, no encode. A [`Renderer`] bundles the surface and the engine; the
//! app stores it in a signal, calls [`Renderer::paint`] after each command, and
//! [`Renderer::resize`] when the canvas (window) changes size.

use stark_core::geom::Extent2;
use stark_core::{
    ColorSpaceId, Engine, EnvironmentId, GpuContext, InputCommand, MediaParams, ObservableState,
    SurfaceId, ViewTransform,
};
use wasm_bindgen::JsCast;

/// Default paper-white background, as straight sRGB (the engine's
/// `rgb_to_channels` linearizes it). Kept neutral (not warm) — the studio
/// environment light already tints the scene warm, and a neutral substrate keeps
/// paint colours legible against it. The live value is [`Renderer::set_background`].
pub const BG: wgpu::Color = wgpu::Color {
    r: 0.97,
    g: 0.97,
    b: 0.97,
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
    /// Canvas substrate colour (straight sRGB), a view setting like the lighting —
    /// it feeds the media pass's `background_color()`, not the document (§6.3).
    background: wgpu::Color,
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

    /// Switch the canvas surface in place — the document is preserved; existing
    /// paint re-reads against the new weave (DESIGN.md §6.4).
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

    /// The current lighting environment (DESIGN.md §6.3).
    pub fn environment(&self) -> EnvironmentId {
        self.engine.environment()
    }

    /// Register frontend-fetched HDR bytes for a lighting environment (§6.3).
    pub fn register_environment(&mut self, id: EnvironmentId, hdr_bytes: Vec<u8>) {
        self.engine.register_environment(id, hdr_bytes);
    }

    /// Switch the lighting environment (a view setting — never resets the canvas).
    pub fn set_environment(&mut self, id: EnvironmentId) {
        self.engine.set_environment(id);
    }

    /// Tune the media/lighting parameters (exposure, gloss, relief — §6.3).
    pub fn set_media_params(&mut self, params: MediaParams) {
        self.engine.set_media_params(params);
    }

    /// The current media/lighting parameters (so a second renderer — the brush
    /// editor's preview — can mirror the main canvas's look).
    pub fn media_params(&self) -> MediaParams {
        self.engine.media_params()
    }

    /// The shared GPU handles (cheap `Arc` clones), so a second renderer can be
    /// built on the same device via [`init_shared`].
    pub fn gpu(&self) -> GpuContext {
        self.engine.gpu().clone()
    }

    /// The surface's current size in CSS pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// The current canvas substrate colour, as straight sRGB components.
    pub fn background(&self) -> [f32; 3] {
        [self.background.r as f32, self.background.g as f32, self.background.b as f32]
    }

    /// Set the canvas substrate colour (straight sRGB). A view setting — it does not
    /// touch the document, only how bare canvas renders (§6.3).
    pub fn set_background(&mut self, rgb: [f32; 3]) {
        self.background = wgpu::Color {
            r: rgb[0] as f64,
            g: rgb[1] as f64,
            b: rgb[2] as f64,
            a: 1.0,
        };
    }

    /// The built-in bristle brush's asset id, once its bytes have been imported.
    pub fn bristle(&self) -> Option<stark_core::AssetId> {
        self.bristle
    }

    // --- collaboration (DESIGN.md §12) — thin engine delegates for the
    // session glue in `collab.rs`. ---

    /// Convert the current document into a shared one, authored as `actor`.
    pub fn start_collaboration(&mut self, actor: stark_core::document::ActorId) {
        self.engine.start_collaboration(actor);
    }

    /// Replace the document with a joined session's log.
    pub fn join_collaboration(
        &mut self,
        file: &stark_core::DocumentFile,
        actor: stark_core::document::ActorId,
    ) {
        self.engine.join_collaboration(file, actor);
    }

    /// Leave a shared session (keep the canvas, stop broadcasting).
    pub fn end_collaboration(&mut self) {
        self.engine.end_collaboration();
    }

    /// Snapshot the document (full shared log + referenced assets).
    pub fn document_file(&self) -> stark_core::DocumentFile {
        self.engine.document_file()
    }

    /// Integrate one remote action; returns whether it was new.
    pub fn merge_remote(&mut self, action: stark_core::document::Action) -> bool {
        self.engine.merge_remote(action)
    }

    /// Drain locally-committed actions awaiting broadcast.
    pub fn take_outbox(&mut self) -> Vec<stark_core::document::Action> {
        self.engine.take_outbox()
    }

    /// Import a remote peer's brush image so its strokes render faithfully.
    pub fn import_brush(&self, png_bytes: &[u8]) {
        if let Err(e) = self.engine.import_brush(png_bytes) {
            tracing::warn!("remote brush import failed: {e}");
        }
    }

    /// Every imported brush asset, to seed a session's asset mirror.
    pub fn all_asset_bytes(&self) -> Vec<(stark_core::AssetId, Vec<u8>)> {
        self.engine.all_asset_bytes()
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
        self.engine.render(&view, self.background);
        self.engine.gpu().queue.present(frame);
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

/// Look up a canvas element the app rendered into the DOM (the main painting
/// canvas, or the brush editor's preview canvas).
pub fn canvas_element(id: &str) -> web_sys::HtmlCanvasElement {
    web_sys::window()
        .expect("window")
        .document()
        .expect("document")
        .get_element_by_id(id)
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
            compatible_surface: Some(&surface),
            ..Default::default()
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

    let gpu = GpuContext::from_parts(instance, adapter, device, queue);
    finish_init(canvas, surface, gpu).await
}

/// Build a second [`Renderer`] on the app's **existing** GPU device: a new
/// surface bound to `canvas` plus its own `Engine` (its own document), sharing
/// `gpu`'s instance/device/queue. Used by the brush editor's preview canvas —
/// cheap (no adapter/device request), and the preview document stays fully
/// isolated from the real one.
pub async fn init_shared(canvas: web_sys::HtmlCanvasElement, gpu: GpuContext) -> Renderer {
    let surface: wgpu::Surface<'static> = gpu
        .instance
        .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
        .expect("create preview canvas surface");
    finish_init(canvas, surface, gpu).await
}

/// Shared tail of [`init`]/[`init_shared`]: size the drawing buffer, pick the
/// surface format, configure, and build the engine.
async fn finish_init(
    canvas: web_sys::HtmlCanvasElement,
    surface: wgpu::Surface<'static>,
    gpu: GpuContext,
) -> Renderer {
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
    let caps = surface.get_capabilities(&gpu.adapter);
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
        color_space: wgpu::SurfaceColorSpace::default(),
    };
    surface.configure(&gpu.device, &config);

    let engine = Engine::new(gpu, format, Extent2::new(width, height));
    Renderer {
        canvas,
        surface,
        config,
        engine,
        bristle: None,
        background: BG,
    }
}
