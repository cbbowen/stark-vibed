//! WebGPU surface rendering (§6.4, §11).
//!
//! The engine renders directly into the canvas's `wgpu::Surface` texture — no
//! readback, no encode. A [`Renderer`] bundles the surface and the [`Desk`] holding
//! the engine; the app stores it in a signal, requests a paint after each command
//! (coalesced to one [`Renderer::paint`] per animation frame —
//! [`request_paint`](crate::state::request_paint)), and calls
//! [`Renderer::resize`] when the canvas (window) changes size.
//!
//! What is here is what the surface adds. The engine's own methods are reached
//! through [`Renderer::desk`], never re-spelled.

use stark_engine::command::Tool;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::platform::Canvas;
use stark_engine::Extent2;
use stark_engine::command::ViewCommand;
use stark_engine::command::{InputCommand, InputSample};
use stark_engine::{Engine, GpuContext};
use stark_ui::desk::Desk;

pub const CANVAS_ID: &str = "stark-canvas";

/// How many painted frames may still be executing on the GPU before
/// [`request_paint`](crate::state::request_paint) skips a frame instead of
/// submitting another.
///
/// This is the back-pressure the surface cannot give on the WebGPU backend:
/// `present` is a no-op there, `PresentMode::Fifo` and
/// `desired_maximum_frame_latency` are dead values, and `get_current_texture`
/// hands out a fresh canvas texture every frame without blocking — so nothing
/// stops a paint per rAF from deepening the GPU queue without bound whenever a
/// frame's work exceeds the frame budget. Two is what
/// `desired_maximum_frame_latency` would have asked for: enough depth that the
/// CPU and GPU pipeline instead of alternating, while a GPU more than two
/// frames behind sheds presentation frames rather than queueing them.
/// Skipping is safe because ingestion is decoupled from presentation — samples
/// keep reaching the fitter per event, and the first fold after a skip shows
/// exactly what the skipped folds would have (`Engine::flush_live`).
const MAX_FRAMES_IN_FLIGHT: u32 = 2;

/// Owns the canvas surface and the painting engine.
pub struct Renderer {
    canvas: Canvas,
    /// The two handles a *surface* is made from, kept on this side because they are
    /// this frontend's business and not the engine's ([`stark_engine::GpuContext`]):
    /// the app binds three `<canvas>` elements over one device — the painting
    /// canvas, the navigator's miniature and the brush editor's preview — and each
    /// of them needs the instance to create the surface and the adapter to ask what
    /// that surface can do. Both are cheap `Arc` handles, and wgpu does not require
    /// either to be kept alive; keeping them is how a *second* surface stays
    /// possible after the first is bound.
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    /// The engine and the shipped assets it has loaded by name — `&mut` only inside
    /// the `state` doors, like the rest of the renderer.
    pub desk: Desk,
    /// The Navigator panel's canvas and everything that draws into it — `None` until
    /// the panel mounts one ([`Renderer::attach_overview`]).
    overview: Option<Overview>,
    /// The compositing attachments the Layers panel's thumbnails render through
    /// ([`Renderer::export_layer`]).
    ///
    /// Kept rather than allocated per call, unlike [`export`](Self::export)'s, and the
    /// decision is here rather than at the call site for the reason `Offscreen`'s own
    /// doc gives: whether a slot outlives its call is the *caller's* to state, and this
    /// is the caller that knows. A file export happens once and may be enormous; a
    /// thumbnail is 64 px, is rendered once per layer, and is rendered again on the
    /// next commit — so allocating and dropping a pair per row would be the cost of
    /// the feature.
    layer_thumbs: stark_engine::Offscreen,
    /// Painted frames whose GPU work has not yet completed (see
    /// [`MAX_FRAMES_IN_FLIGHT`]). Incremented per [`paint`](Self::paint),
    /// decremented by the `on_submitted_work_done` callback that paint registers
    /// — an atomic behind an `Arc` because the callback must be `Send` and may
    /// not touch a signal.
    frames_in_flight: Arc<AtomicU32>,
}

/// A second WebGPU surface showing the same document: the Navigator panel's canvas,
/// the surface bound to it, and the compositing attachments the miniature renders
/// through (`panels::navigator`).
///
/// The miniature is a *rendered surface*, not an image the UI carries — the same
/// bargain the painting canvas makes (§11). It began as an `export`: render
/// to an offscreen texture, copy the pixels back to the CPU, hand them to a 2D canvas
/// through `ImageData`. Every part of that after "render" existed only because the
/// miniature had nowhere of its own to draw, and giving it a surface deleted all of
/// it — the GPU→CPU copy and its frame of latency, the pixel buffer held in a signal,
/// and the imperative repaint that had to be re-run whenever the element remounted.
///
/// One document, two surfaces, on one device: exactly what the brush editor's preview
/// canvas already does ([`Renderer::shared`]), except that this one shares the
/// *engine* too, so it is a second view of the real painting rather than a second
/// painting.
struct Overview {
    /// Kept so the drawing buffer can be resized with the surface: the miniature's
    /// pixel size follows the piece's aspect, not the window's.
    canvas: Canvas,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    /// Pass A's attachments, kept between refreshes — see [`stark_engine::Offscreen`].
    /// This render repeats for as long as the panel is open, so it is the one that has
    /// to reuse them; step 2 of this design is what makes a refresh allocate nothing.
    targets: stark_engine::Offscreen,
}

impl Renderer {
    /// Send a command to the engine. Its other `&mut` methods are reached through
    /// [`Renderer::desk`], and replacing the document is [`Desk::replace`]'s alone.
    ///
    /// **No named `set_*` wrappers sit beside it.** A one-line
    /// `engine.process(…)` per setting is a second spelling of a command, and the
    /// second spelling is the one that skips
    /// [`state::dispatch`](crate::state::dispatch): a panel reaching one through the
    /// renderer signal moves engine state without refreshing the observable projection
    /// the chrome reads back, so its own control re-renders showing the *previous*
    /// value and stays wrong until some unrelated command happens to refresh `obs`
    /// (§4, §7). Core declines to expose such a setter for the canvas surface for the
    /// same reason (`Engine::apply_document_substrate`).
    ///
    /// Frontend code holding an `AppState` calls `state::dispatch`, which is this
    /// plus that refresh, the repaint and the outbox flush. This is for the callers
    /// that own a `Renderer` outright and have no chrome to keep in step: app
    /// startup, and the brush editor's private preview engine.
    pub fn process(&mut self, command: impl Into<InputCommand>) {
        self.desk.engine_mut().process(command);
    }

    /// Replay a full stroke as one commit — a single render, no per-sample
    /// live-preview refresh (see `Engine::replay_stroke`).
    pub fn replay_stroke(&mut self, tool: Tool, samples: &[InputSample]) {
        self.desk.engine_mut().replay_stroke(tool, samples);
    }

    /// Replay a full stroke with a caller-chosen jitter seed, so repeated
    /// replays of the same samples keep the same color dynamics and dither
    /// (see `Engine::replay_stroke_seeded`).
    ///
    /// **Answers whether it committed one.** Samples that hold no stroke — none at
    /// all, or a hand that never left its first point — commit nothing, and the brush
    /// editor's preview has to know: it undoes the committed stroke before each
    /// replay, and an undo for a commit that never happened reaches past into the
    /// reference band beneath (`stark_ui::brush_editor::TestStroke`).
    pub fn replay_stroke_seeded(
        &mut self,
        tool: Tool,
        samples: &[InputSample],
        seed: u64,
        rope: f32,
    ) -> bool {
        self.desk
            .engine_mut()
            .replay_stroke_seeded(tool, samples, seed, rope)
            .is_some()
    }

    /// How far above SDR white the display says it can go (§6.5). On the web:
    /// `Some(1.0)` for an SDR display, `None` for an HDR one — the browser says only
    /// which — and the headroom slider stands in for `None`.
    pub fn display_headroom(&self) -> Option<f32> {
        self.surface
            .display_hdr_info(&self.adapter)
            .tone_map_headroom()
    }

    /// The surface's current size in CSS pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Render a frame and return a future for its readback (§15.6).
    ///
    /// The future does **not** borrow the renderer, which is the whole point: the
    /// caller can drop its write guard before awaiting, so the UI is free to
    /// re-render (and read the renderer) while the GPU→CPU copy is in flight.
    ///
    /// A one-shot: the attachments are allocated for this render and dropped with it,
    /// which is what keeps a 4× export of a large frame from parking its
    /// several-hundred-megabyte pair for the rest of the session.
    pub fn export(
        &mut self,
        frame: Option<stark_model::document::LayerId>,
        scale: stark_engine::ExportScale,
        background: stark_engine::Background,
        content: stark_engine::Rendered,
    ) -> stark_engine::Result<
        impl std::future::Future<Output = stark_engine::Result<stark_engine::RgbaImage>> + use<>,
    > {
        self.desk.engine_mut().export(
            &mut stark_engine::Offscreen::default(),
            frame,
            scale,
            background,
            content,
        )
    }

    /// Render **one layer alone** through `plan`'s view, for a Layers panel row
    /// (§14.6). Returns a future for the readback, on [`export`](Self::export)'s
    /// borrow bargain — the caller drops its write guard before awaiting.
    ///
    /// The layer's blend mode, clip and opacity are dropped by the isolate this
    /// renders through (`Engine::export_view`), so a row shows the paint that is
    /// there rather than the part of it the document lets through. Cut out rather
    /// than over the substrate, so a row says where the layer *has* paint.
    ///
    /// `plan` is the caller's, and deliberately: it is the same plan the navigator
    /// frames its miniature with, so an overview and a row cannot come to disagree
    /// about where the piece is.
    pub fn export_layer(
        &mut self,
        layer: stark_model::document::LayerId,
        plan: &stark_engine::ExportPlan,
    ) -> stark_engine::Result<
        impl std::future::Future<Output = stark_engine::Result<stark_engine::RgbaImage>> + use<>,
    > {
        self.desk.engine_mut().export_view(
            &mut self.layer_thumbs,
            plan.view(),
            Some(layer),
            stark_engine::Background::Transparent,
            stark_engine::Rendered::Committed,
        )
    }

    /// Bind the Navigator panel's `<canvas>` as a second surface onto this engine's
    /// device, ready for [`paint_overview`](Self::paint_overview).
    ///
    /// Called from the panel's `onmounted`, and again on every remount: a closed panel
    /// takes its element with it, and the element is what a surface is bound to, so
    /// the old one is dropped here rather than reused. Nothing is measured — the
    /// miniature's size comes from the piece's proportions, not from layout, so the
    /// drawing buffer is sized on the first paint instead.
    ///
    /// Configured to the engine's own target format, not to a format picked from this
    /// surface's capabilities: the engine's pipelines are built for one format, and a
    /// second surface that chose differently would fail validation rather than merely
    /// look wrong. The format the main canvas settled on is available here (both
    /// surfaces are canvases on the same adapter, so it is in this one's caps too).
    pub fn attach_overview(&mut self, canvas: Canvas) {
        let surface = match self.instance.create_surface(canvas.surface_target()) {
            Ok(surface) => surface,
            Err(e) => {
                tracing::warn!("navigator surface unavailable: {e}");
                return;
            }
        };
        let caps = surface.get_capabilities(&self.adapter);
        // Zero says "not configured yet", and cannot collide with a real plan size (a
        // plan's edges are floored at 1), so the first paint always configures before
        // it asks for a texture. The color space is the main surface's: the engine
        // encodes for one transfer (§6.5).
        let config = surface_config(
            &caps,
            self.desk.engine().target_format(),
            self.config.color_space,
            (0, 0),
        );
        self.overview = Some(Overview {
            canvas,
            surface,
            config,
            targets: stark_engine::Offscreen::default(),
        });
    }

    /// Draw the miniature `plan` describes straight into the Navigator's surface and
    /// present it. `false` if no canvas is attached (the panel is closed).
    ///
    /// The committed document, over the substrate: an overview is a picture of the
    /// piece as it stands, and it is refreshed per commit, so following the stroke in
    /// hand would mean re-rendering at pointer rate to show what the canvas beside it
    /// is already showing full size.
    ///
    /// Synchronous, which is the whole point of the surface: there is no readback to
    /// await, so a refresh is one render and a present.
    pub fn paint_overview(&mut self, plan: &stark_engine::ExportPlan) -> bool {
        use wgpu::CurrentSurfaceTexture::{Suboptimal, Success};
        let Some(ov) = self.overview.as_mut() else {
            return false;
        };
        let size = plan.size;
        if (ov.config.width, ov.config.height) != (size.width, size.height) {
            ov.canvas.set_buffer_size(size.width, size.height);
            ov.config.width = size.width;
            ov.config.height = size.height;
            ov.surface
                .configure(&self.desk.engine().gpu().device, &ov.config);
        }
        let frame = match ov.surface.get_current_texture() {
            Success(frame) | Suboptimal(frame) => frame,
            // Timeout/Outdated/Lost: skip it. The next committed edit repaints, and a
            // miniature one revision stale is not worth a retry loop.
            _ => return false,
        };
        let target = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.desk.engine_mut().render_into(
            &mut ov.targets,
            &target,
            plan.view(),
            stark_engine::Background::Substrate,
            stark_engine::Rendered::Committed,
        );
        self.desk.engine().gpu().queue.present(frame);
        true
    }

    /// Sample the canvas color at `at` — the eyedropper (§18.0.2).
    ///
    /// The same borrow bargain as [`Renderer::export`], and it matters more here:
    /// the sample is taken mid-gesture, so the caller has to be able to drop its
    /// write guard before awaiting or the UI's own re-render will panic on a
    /// renderer it still holds borrowed.
    pub fn pick_color(
        &mut self,
        at: stark_model::geom::Vec2,
        options: stark_engine::PickOptions,
    ) -> impl std::future::Future<Output = Option<[f32; 3]>> + use<> {
        self.desk.engine_mut().pick_color(at, options)
    }

    /// Sample a gradient along a traced path — the gradient capture (§22.2).
    /// The same borrow bargain as [`Renderer::pick_color`].
    pub fn pick_gradient(
        &mut self,
        path: &[stark_model::geom::Vec2],
        options: stark_engine::PickOptions,
    ) -> impl std::future::Future<Output = Option<stark_model::Gradient>> + use<> {
        self.desk.engine_mut().pick_gradient(path, options)
    }

    /// Which layer's paint the canvas shows at `at` — the layer carry's hit test
    /// (§16.11). The same borrow bargain as [`Renderer::pick_color`], and it
    /// matters here for that reason exactly: this one opens a drag, so the
    /// gesture goes on driving the renderer while the readback is in flight.
    pub fn pick_layer(
        &mut self,
        at: stark_model::geom::Vec2,
    ) -> impl std::future::Future<Output = Option<stark_model::document::LayerId>> + use<> {
        self.desk.engine_mut().pick_layer(at)
    }

    /// Match the surface + engine viewport to a new canvas size (CSS pixels).
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 || (width == self.config.width && height == self.config.height)
        {
            return;
        }
        self.canvas.set_buffer_size(width, height);
        self.config.width = width;
        self.config.height = height;
        self.surface
            .configure(&self.desk.engine().gpu().device, &self.config);
        self.desk
            .engine_mut()
            .process(ViewCommand::Resize(Extent2::new(width, height)));
    }

    /// Re-measure the canvas element and match the surface to it. A no-op when it
    /// already agrees.
    ///
    /// The size [`finish_init`] seeds from is a *guess*. It is read one animation
    /// frame in, which is not the same thing as the stylesheet having applied: until
    /// it does, `.paint-canvas` is not in force and the element measures the canvas's
    /// intrinsic 300×150 rather than the window. Nothing corrects that on its own,
    /// because the only correction is the DOM resize observer, and it reports through
    /// [`crate::state::resize`], which can act only once the renderer signal is
    /// published — while everything between `init` and that publish is a *network
    /// fetch* (shape assets, the substrate's height map, the environment HDR). The
    /// corrected size therefore lands squarely inside the window where it is dropped,
    /// and the viewport keeps a size the canvas has not had since the first frame:
    /// the view's `half()` is off by the difference, so every stroke lands away from
    /// the pointer until something else changes the layout.
    ///
    /// So the seed is treated as provisional and this re-reads the element at the
    /// first moment a resize could no longer be missed — the statement immediately
    /// before the renderer is published, with no `await` between the two.
    pub fn sync_to_canvas(&mut self) {
        let (width, height) = self.canvas.laid_out_size();
        self.resize(width, height);
    }

    /// Whether the device is still usable — the `&self` request form of what
    /// [`ObservableState::gpu_failure`](stark_engine::ObservableState::gpu_failure)
    /// projects (§5).
    ///
    /// For the one caller that has no projection to hand: the paint loop
    /// (`state::schedule_paint`). Everything else asks the projection, which is
    /// what the chrome mounts its report on — see [`crate::failure`].
    pub fn gpu_healthy(&self) -> bool {
        self.desk.engine().gpu().health().is_ok()
    }

    /// Whether the GPU still owes the work of [`MAX_FRAMES_IN_FLIGHT`] painted
    /// frames — the signal for [`request_paint`](crate::state::request_paint) to
    /// skip a frame rather than deepen the queue. Because submissions on one
    /// queue complete in order, a paint's completion also vouches for every
    /// submission before it (commit renders, fills), so queue depth from
    /// non-paint work is counted too, one frame later.
    pub fn gpu_behind(&self) -> bool {
        self.frames_in_flight.load(Ordering::Relaxed) >= MAX_FRAMES_IN_FLIGHT
    }

    /// Render the current canvas straight into the surface texture and present.
    pub fn paint(&mut self) {
        use wgpu::CurrentSurfaceTexture::{Suboptimal, Success};
        // Its own row because it is the one part of a frame that is not Stark's
        // work: acquiring a surface texture is where a browser compositor makes the
        // page wait, and folded into `frame` that wait would read as time the engine
        // spent. On WebGPU it should be free — `get_current_texture` never blocks
        // there — so a row that grows is a finding rather than a cost.
        let frame = {
            stark_engine::timing::span!("frame.acquire");
            match self.surface.get_current_texture() {
                Success(frame) | Suboptimal(frame) => frame,
                // Timeout/Outdated/Lost/etc.: skip; the next command repaints.
                _ => return,
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.desk.engine_mut().render(&view);
        // Count this frame against the in-flight budget until the GPU finishes
        // it. Registered after the render's submit, so the callback fires once
        // everything this paint queued has executed. The WebGPU spec resolves
        // the underlying promise even on device loss, so the count cannot wedge.
        let in_flight = Arc::clone(&self.frames_in_flight);
        in_flight.fetch_add(1, Ordering::Relaxed);
        self.desk
            .engine()
            .gpu()
            .queue
            .on_submitted_work_done(move || {
                in_flight.fetch_sub(1, Ordering::Relaxed);
            });
        // A no-op on the web, where the canvas is presented by the page — measured
        // anyway, and cheaply, because "present is free here" is a claim about wgpu's
        // WebGPU backend that a version bump could quietly stop being true.
        stark_engine::timing::span!("frame.present");
        self.desk.engine().gpu().queue.present(frame);
    }
}

/// The format and color space the main canvas is configured with (§6.5): as much
/// range and as much gamut as the display in front of it actually has.
///
/// **Both halves ask what the display *is*, not what the format allows.** wgpu
/// advertises the extended spaces on every fp16-capable browser, tone-mapping or not,
/// and `display-p3` on every canvas whatever the panel — so picking on the
/// capability alone would charge an SDR sRGB user a double-width swapchain and a
/// gamut conversion for nothing. What the web can say about a display is two CSS
/// media queries, and they are exactly the two questions: `dynamic-range` and
/// `color-gamut`.
///
/// Decided once, because the engine's pipelines are compiled for one format (§6.4);
/// a display swapped afterwards is seen on the next load. The 8-bit fallback is a
/// non-sRGB format, as it always was — the media pass encodes the transfer itself, so
/// an `*Srgb` surface would encode it twice.
fn pick_surface(
    caps: &wgpu::SurfaceCapabilities,
    display: &wgpu::DisplayHdrInfo,
) -> (wgpu::TextureFormat, wgpu::SurfaceColorSpace) {
    let coarse = display.coarse.as_ref();
    let hdr = coarse.and_then(|c| c.high_dynamic_range).unwrap_or(false);
    let wide = matches!(
        coarse.and_then(|c| c.gamut),
        Some(wgpu::DisplayGamut::DisplayP3 | wgpu::DisplayGamut::Rec2020)
    );
    let f16 = wgpu::TextureFormat::Rgba16Float;
    let offers =
        |format, space: wgpu::SurfaceColorSpaces| caps.color_spaces(format).contains(space);

    if hdr && wide && offers(f16, wgpu::SurfaceColorSpaces::EXTENDED_DISPLAY_P3) {
        return (f16, wgpu::SurfaceColorSpace::ExtendedDisplayP3);
    }
    if hdr && offers(f16, wgpu::SurfaceColorSpaces::EXTENDED_SRGB) {
        return (f16, wgpu::SurfaceColorSpace::ExtendedSrgb);
    }
    let format = caps
        .formats
        .iter()
        .copied()
        .find(|f| !f.is_srgb())
        .unwrap_or(caps.formats[0]);
    // Wide gamut without the range: an 8-bit `display-p3` canvas, which is what a
    // laptop panel with P3 coverage and no HDR mode has to offer.
    if wide && offers(format, wgpu::SurfaceColorSpaces::DISPLAY_P3) {
        return (format, wgpu::SurfaceColorSpace::DisplayP3);
    }
    (format, wgpu::SurfaceColorSpace::Auto)
}

/// The transfer a canvas configured in `color_space` reads its texels in (§6.5) — off
/// the surface configuration, so the engine cannot be told another.
fn transfer_of(color_space: wgpu::SurfaceColorSpace) -> stark_engine::Transfer {
    use stark_engine::Transfer;
    match color_space {
        wgpu::SurfaceColorSpace::ExtendedSrgb => Transfer::ExtendedSrgb,
        wgpu::SurfaceColorSpace::ExtendedSrgbLinear => Transfer::Linear,
        wgpu::SurfaceColorSpace::DisplayP3 => Transfer::DisplayP3,
        wgpu::SurfaceColorSpace::ExtendedDisplayP3 => Transfer::ExtendedDisplayP3,
        _ => Transfer::Srgb,
    }
}

/// Why the app could not start (§5, `crate::failure`).
///
/// **A different fact from `ObservableState::gpu_failure`**, and the difference is
/// what earns it a type of its own: that one is a device that *died*, with a
/// document behind it that outlives it and is worth saving. This is a device that
/// never arrived — there is no engine, no document and nothing to offer. What the
/// two share is that the canvas will never take a mark, which is why both reports
/// are the same surface.
///
/// It was `expect` on all three arms until the review that named it. That is
/// defensible for `create_surface`, which fails only if the element is not a
/// canvas, and indefensible for the other two: a browser without WebGPU is the
/// single most likely way this app fails for a first-time visitor — Safari before
/// 26, Firefox without the flag, a blocklisted driver, any headless browser — and
/// a panic inside the startup task killed the task and nothing else, leaving the
/// chrome up over a blank canvas with the explanation in the console.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StartupFailure {
    /// The page's `<canvas>` would not give a WebGPU surface.
    Surface(String),
    /// No adapter answered — what a browser with no WebGPU at all looks like
    /// from here, and the common case by a wide margin.
    Adapter,
    /// An adapter answered but would not give a device at the limits the engine
    /// needs ([`GpuContext::minimum_required_limits`]).
    Device(String),
}

impl std::fmt::Display for StartupFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StartupFailure::Surface(why) => write!(f, "no WebGPU surface: {why}"),
            StartupFailure::Adapter => write!(f, "no WebGPU adapter"),
            StartupFailure::Device(why) => write!(f, "no WebGPU device: {why}"),
        }
    }
}

/// Asynchronously create the WebGPU device, configure the surface to the
/// canvas's current size, and build the engine (§7).
///
/// Fallible on all three of the browser's answers rather than panicking on them
/// — see [`StartupFailure`] for why that is not merely tidiness.
pub async fn init(canvas: Canvas) -> Result<Renderer, StartupFailure> {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::BROWSER_WEBGPU;
    let instance = wgpu::Instance::new(desc);

    let surface: wgpu::Surface<'static> = instance
        .create_surface(canvas.surface_target())
        .map_err(|e| StartupFailure::Surface(e.to_string()))?;

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..Default::default()
        })
        .await
        .map_err(|_| StartupFailure::Adapter)?;

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
        .map_err(|e| StartupFailure::Device(e.to_string()))?;

    let gpu = GpuContext::from_parts(device, queue);
    Ok(finish_init(canvas, instance, adapter, surface, gpu).await)
}

impl Renderer {
    /// Build a second [`Renderer`] on this one's device: a new surface bound to
    /// `canvas` plus an engine of its own that **shares** this engine's expensive
    /// state — every compiled pipeline, the imported brush shapes, and the decoded
    /// substrate and environment caches (`Engine::new_sharing`). The preview document
    /// stays fully isolated from the real one; it opens on this document's substrate,
    /// under this canvas's lighting and media parameters, so a stroke on it reads
    /// exactly as it would here — with nothing re-fetched and nothing re-decoded.
    ///
    /// Synchronous, and callable only once this renderer exists — which is also the
    /// only time it makes sense: a preview is a preview *of* this canvas. The caller
    /// should await a layout frame ([`platform::next_frame`](crate::platform::next_frame))
    /// before this, so the canvas
    /// measures as laid out rather than at its 300×150 intrinsic size; the measure
    /// here is still only a seed, corrected by [`Renderer::sync_to_canvas`] before
    /// anything is placed against it.
    ///
    /// Configured to this engine's own target format rather than a format picked
    /// from the new surface's capabilities, exactly as
    /// [`attach_overview`](Self::attach_overview) is and for the same reason: the
    /// shared pipelines are built for one format, and a second surface that chose
    /// differently would fail validation rather than merely look wrong.
    ///
    /// Errs where the browser refuses `canvas` a surface, as
    /// [`attach_overview`](Self::attach_overview) can; the caller decides what a
    /// missing preview looks like.
    pub fn shared(&self, canvas: Canvas) -> Result<Renderer, wgpu::CreateSurfaceError> {
        let (width, height) = canvas.laid_out_size();
        canvas.set_buffer_size(width, height);
        let surface: wgpu::Surface<'static> =
            self.instance.create_surface(canvas.surface_target())?;
        let caps = surface.get_capabilities(&self.adapter);
        // And to its color space: the engine encodes for one transfer (§6.5).
        let config = surface_config(
            &caps,
            self.desk.engine().target_format(),
            self.config.color_space,
            (width, height),
        );
        surface.configure(&self.desk.engine().gpu().device, &config);
        let desk = self.desk.sharing(Extent2::new(width, height));
        Ok(Renderer {
            canvas,
            instance: self.instance.clone(),
            adapter: self.adapter.clone(),
            surface,
            config,
            desk,
            overview: None,
            layer_thumbs: stark_engine::Offscreen::default(),
            frames_in_flight: Arc::new(AtomicU32::new(0)),
        })
    }
}

/// How every surface here is configured — the main canvas, the Navigator's, a
/// preview's — which differ only in format, color space and size.
fn surface_config(
    caps: &wgpu::SurfaceCapabilities,
    format: wgpu::TextureFormat,
    color_space: wgpu::SurfaceColorSpace,
    (width, height): (u32, u32),
) -> wgpu::SurfaceConfiguration {
    wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width,
        height,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
        color_space,
    }
}

/// Tail of [`init`]: size the drawing buffer, pick the surface format, configure,
/// and build the engine. (A *second* renderer never comes through here — it is built
/// synchronously by [`Renderer::shared`], on the first engine's format and state.)
async fn finish_init(
    canvas: Canvas,
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    surface: wgpu::Surface<'static>,
    gpu: GpuContext,
) -> Renderer {
    // Size the drawing buffer to the canvas's laid-out size (CSS pixels). We
    // measure the *element*, not the window, so an embedded/sub-window canvas
    // works too, and we do it here — after the async device setup and a layout
    // frame — rather than up front, where the unstyled 300×150 intrinsic size is
    // all there is to read.
    //
    // A frame is not a *guarantee* that the stylesheet (linked via
    // `document::Stylesheet`) has applied, though, so this is a seed and not the
    // answer: the caller re-reads the element with `Renderer::sync_to_canvas` just
    // before publishing the renderer, which is where the guarantee actually is.
    // Everything after that is handled by `onresize`.
    crate::platform::next_frame().await;
    let (width, height) = canvas.laid_out_size();
    canvas.set_buffer_size(width, height);

    let caps = surface.get_capabilities(&adapter);
    let (format, color_space) = pick_surface(&caps, &surface.display_hdr_info(&adapter));
    tracing::info!(?format, ?color_space, "canvas surface");

    let config = surface_config(&caps, format, color_space, (width, height));
    surface.configure(&gpu.device, &config);

    let engine = Engine::new(gpu, format, Extent2::new(width, height));
    let desk = Desk::new(engine, transfer_of(color_space));
    Renderer {
        canvas,
        instance,
        adapter,
        surface,
        config,
        desk,
        overview: None,
        layer_thumbs: stark_engine::Offscreen::default(),
        frames_in_flight: Arc::new(AtomicU32::new(0)),
    }
}
