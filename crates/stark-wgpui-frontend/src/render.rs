//! The wgpui surface the engine paints into (§6.4, §11).
//!
//! The web frontend's `Renderer` builds the device, binds a `wgpu::Surface` to a
//! `<canvas>` and configures it. None of that happens here: **wgpui owns the device**
//! and hands out a double-buffered pair of textures instead, so what this holds is a
//! [`WgpuSurfaceHandle`] and a [`Desk`] around an [`Engine`] built on the device
//! behind it. The engine renders straight into the back buffer and the swap is a
//! pointer swap — the same bargain the browser canvas makes, with no readback and no
//! encode.
//!
//! What is here is what the surface adds. The engine's own methods are reached through
//! [`Renderer::desk`], never re-spelled.

use stark_engine::Extent2;
use stark_engine::command::{InputCommand, ViewCommand};
use stark_engine::{Engine, GpuContext, Output, Transfer, ViewTransform};
use stark_ui::desk::Desk;
use wgpui::{WgpuSurfaceHandle, Window};

/// The format the engine renders through and the transfer the window reads it in
/// (§6.5), read off the swapchain wgpui configured (`vendor/wgpui/VENDORING.md`,
/// patch 6): the engine's texels are composited into it unconverted. **Never an
/// sRGB format**: the media pass encodes the transfer itself.
fn target_for(window: &Window) -> (wgpu::TextureFormat, Transfer) {
    match window.surface_color_space() {
        Some(wgpu::SurfaceColorSpace::ExtendedSrgbLinear) => {
            (wgpu::TextureFormat::Rgba16Float, Transfer::Linear)
        }
        _ => (wgpu::TextureFormat::Rgba8Unorm, Transfer::Srgb),
    }
}

/// One surface, one engine, and the resize that keeps them agreeing.
pub struct Renderer {
    surface: WgpuSurfaceHandle,
    /// The engine, and the shipped assets it has loaded by name.
    pub desk: Desk,
    /// The viewport the engine was last told about, in device px — see
    /// [`paint`](Self::paint), which is where it is corrected.
    viewport: (u32, u32),
    /// The navigator's miniature, once something has asked for one (`crate::navigator`).
    ///
    /// A **second surface**, not a picture: the engine renders the committed document
    /// straight into it and the swap is a pointer swap, exactly as the canvas's is —
    /// one document seen twice. So nothing here holds pixels, and a refresh costs a
    /// render rather than a render plus a readback.
    overview: Option<Overview>,
}

/// The miniature's surface and the pass-A attachments it renders through.
///
/// Its own [`Offscreen`](stark_engine::Offscreen) rather than the screen's, because
/// those are the size of the *target*: sharing them would have the miniature resize
/// the canvas's attachments away on every frame and back again on the next.
struct Overview {
    surface: WgpuSurfaceHandle,
    targets: stark_engine::Offscreen,
    /// The surface size the last render covered, in device px — the canvas keeps the
    /// same number under the same argument ([`viewport`](Renderer::viewport)).
    drawn: (u32, u32),
}

impl Renderer {
    /// Bind a surface to `window` and build an engine on the device behind it.
    ///
    /// `None` when wgpui is not on its wgpu renderer, which is the one platform
    /// answer that leaves nothing to paint with — the view reports it rather than
    /// panicking, for the reason the web frontend's `StartupFailure` gives.
    pub fn new(window: &Window) -> Option<Self> {
        let (width, height) = device_pixels(window);
        let (format, transfer) = target_for(window);
        let surface = window.create_wgpu_surface(width, height, format)?;
        // The engine is *given* its wgpu resources (CLAUDE.md), and here that is
        // forced rather than chosen: the handle carries a device and a queue and
        // nothing else. It also replaces wgpui's device callbacks with the engine's
        // (`GpuContext::from_parts`), which is the right way round — the engine is
        // what has to stop issuing work when the device dies.
        let gpu = GpuContext::from_parts(surface.device().clone(), surface.queue().clone());
        let mut engine = Engine::new(gpu, format, Extent2::new(width, height));
        // The transfer is the surface's from the first frame; `Desk::apply_output`
        // moves only the headroom.
        engine.process(ViewCommand::SetOutput(Output::new(transfer, 1.0)));
        Some(Self {
            surface,
            desk: Desk::new(engine, transfer),
            viewport: (width, height),
            overview: None,
        })
    }

    /// Send a command to the engine (§4). Its other `&mut` methods are reached through
    /// [`Renderer::desk`], and replacing the document is [`Desk::replace`]'s alone. No
    /// named `set_*` sits beside it, for the reason the web frontend's
    /// `Renderer::process` spells out.
    pub fn process(&mut self, command: impl Into<InputCommand>) {
        self.desk.engine_mut().process(command);
    }

    /// Render a picture and hand back a future for its readback (§15.6).
    ///
    /// The future does **not** borrow the renderer, which is the point: the caller
    /// can go back to painting while the GPU→CPU copy is in flight. A one-shot — the
    /// attachments are allocated for this render and dropped with it, so a large
    /// export does not park its buffers for the session.
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

    /// Sample the canvas color at `at` — the eyedropper (§18.0.2).
    ///
    /// A **request**, not a command (§4): it has to answer. The render happens now and
    /// the future is the readback alone, so it borrows nothing — which is what lets
    /// the window go on painting while the copy is in flight, exactly as
    /// [`export`](Self::export) does.
    pub fn pick_color(
        &mut self,
        at: stark_model::geom::Vec2,
        options: stark_engine::PickOptions,
    ) -> impl std::future::Future<Output = Option<[f32; 3]>> + use<> {
        self.desk.engine_mut().pick_color(at, options)
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

    // --- the navigator's miniature (§11) --------------------------------------

    /// What a miniature of the whole piece would be, at the largest size that fits
    /// `into` — **exactly what an export would write** (§15.6), because it is the same
    /// call: [`Engine::export_plan`] answers the rect, and the plan it returns *is*
    /// the view the miniature renders through. So the overview cannot come to
    /// disagree with the picture a file would hold.
    ///
    /// `None` on an unpainted, unframed canvas, where the rect the engine would fall
    /// back to is the *viewport* — a picture of the window presented as the piece.
    /// An unbounded canvas with nothing on it has no overview, and saying so is the
    /// honest answer.
    pub fn overview_plan(
        &self,
        frame: Option<stark_model::document::LayerId>,
        into: Extent2,
    ) -> Option<stark_engine::ExportPlan> {
        self.desk
            .engine()
            .export_plan(frame, stark_engine::ExportScale::Fit(into))
            .ok()
    }

    /// Draw the miniature `plan` describes into the overview surface and swap it.
    ///
    /// The surface is made on the first call, at the plan's own size; from then on the
    /// element resizes it from its own bounds, one frame behind — so the view is given
    /// the surface's *actual* size as its viewport rather than the plan's. The two
    /// differ by a pixel of rounding in the steady state and by a whole aspect on the
    /// frame after the piece is reshaped, and taking the target's word for it is what
    /// keeps that frame a slightly wider crop rather than a stretched picture. That
    /// resize also *discards* what was drawn, which is what
    /// [`overview_resized`](Self::overview_resized) is for.
    ///
    /// The **committed** document, over the substrate: an overview is a picture of the
    /// piece as it stands, and following the stroke in hand would mean compositing
    /// every tile in the document at pointer rate to show what the canvas beside it is
    /// already showing full size.
    pub fn paint_overview(&mut self, window: &Window, plan: &stark_engine::ExportPlan) -> bool {
        if self.overview.is_none() {
            // `None` where wgpui is not on its wgpu renderer, which is the same
            // answer the canvas's own surface gives and is reported the same way:
            // nothing to draw the miniature with, so it is simply not drawn.
            let Some(surface) =
                window.create_wgpu_surface(plan.size.width, plan.size.height, self.format())
            else {
                return false;
            };
            self.overview = Some(Overview {
                surface,
                targets: stark_engine::Offscreen::default(),
                // Nothing drawn into it yet, and no size it could have been drawn at.
                drawn: (0, 0),
            });
        }
        let Some(ov) = self.overview.as_mut() else {
            return false;
        };
        let Some((target, size)) = ov.surface.back_view_with_size() else {
            return false;
        };
        let mut view = plan.view();
        view.viewport = Extent2::new(size.0, size.1);
        self.desk.engine_mut().render_into(
            &mut ov.targets,
            &target,
            view,
            stark_engine::Background::Substrate,
            stark_engine::Rendered::Committed,
        );
        ov.drawn = size;
        ov.surface.swap_buffers();
        true
    }

    /// Whether the element has resized the miniature's surface out from under the
    /// picture drawn into it — the miniature's [`resized`](Self::resized), and
    /// sharper: a resized surface is two **new** textures, so what was drawn is not
    /// stretched, it is gone. Nothing else asks for it back, since the refresh policy
    /// is otherwise keyed on the document's revision and a resize does not move that.
    pub fn overview_resized(&self) -> bool {
        self.overview
            .as_ref()
            .is_some_and(|ov| ov.surface.size() != ov.drawn)
    }

    /// The miniature's handle, for the element that composites it — `None` until
    /// something has painted one.
    pub fn overview_surface(&self) -> Option<WgpuSurfaceHandle> {
        self.overview.as_ref().map(|ov| ov.surface.clone())
    }

    /// Let the miniature go: its surface leaves the registry with the last handle
    /// (`WgpuSurfaceHandle`), so putting the navigator away really does give the
    /// textures back rather than keeping a second copy of the piece on the GPU.
    pub fn drop_overview(&mut self) {
        self.overview = None;
    }

    /// The format every surface built on this device is configured in — the engine's
    /// own, since its texels are composited unconverted (§6.5).
    pub fn format(&self) -> wgpu::TextureFormat {
        self.surface.format()
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
            self.desk
                .engine_mut()
                .process(ViewCommand::Resize(Extent2::new(size.0, size.1)));
        }
        self.desk.engine_mut().render(&target);
        self.surface.swap_buffers();
    }
}

/// The brush editor's test canvas: a **third** surface, and a sibling engine on the
/// canvas's own device (§11, §6.2).
///
/// Built by *sharing* the main engine's expensive half (`Engine::on_shared`): the same
/// compiled pipelines, the same content-addressed brush assets, the same decoded
/// substrate and environment, around a fresh document of its own. So a stroke here
/// reads exactly as it will on the real canvas — which is a correctness argument
/// before it is an economy — and opening the dialog fetches and decodes nothing.
///
/// The one thing that does *not* ride in on the shared half is the substrate's tint,
/// which is document state and so is set once at construction.
///
/// It is the navigator's second surface again with one difference: that one is a
/// picture of the document this engine already holds, and this is a document of its
/// own — so it takes an engine rather than a render target.
pub struct Preview {
    surface: WgpuSurfaceHandle,
    engine: Engine,
    /// The viewport the engine was last told about, in device px. Corrected in
    /// [`paint`](Self::paint) for [`Renderer::paint`]'s reason: the buffer is what the
    /// render lands in, so a view that disagreed with it would put the test stroke
    /// somewhere other than under the pointer.
    viewport: (u32, u32),
}

impl Preview {
    /// Bind a surface `width`×`height` device px and build a sibling engine on it.
    ///
    /// `None` where wgpui is not on its wgpu renderer — the same answer the canvas's
    /// own surface gives, reported the same way: there is nothing to draw with, so
    /// nothing is drawn.
    pub fn new(donor: &Renderer, window: &Window, width: u32, height: u32) -> Option<Self> {
        let (width, height) = (width.max(1), height.max(1));
        let surface = window.create_wgpu_surface(width, height, donor.format())?;
        let donor = donor.desk.engine();
        let mut engine = Engine::on_shared(donor.shared(), Extent2::new(width, height));
        engine.process(stark_engine::command::DocCommand::SetSubstrateColor(
            donor.observe().substrate_color,
        ));
        Some(Self {
            surface,
            engine,
            viewport: (width, height),
        })
    }

    /// Send a command to the sibling engine — the only door, for
    /// [`Renderer::process`]'s reason (§4).
    pub fn process(&mut self, command: impl Into<InputCommand>) {
        self.engine.process(command);
    }

    /// The surface's size in device px, which is what the test stroke is laid out
    /// against.
    pub fn size(&self) -> (u32, u32) {
        self.viewport
    }

    /// The view a pointer position on this surface is mapped through.
    pub fn view(&self) -> ViewTransform {
        self.engine.view()
    }

    /// Replay a whole recorded stroke as one commit, with the jitter seed pinned
    /// (`Engine::replay_stroke_seeded`) so only the edited parameter moves between
    /// renders. `rope` is the §6.11 smoothing string, because the preview replays a
    /// recorded hand and has to show what the smoothing slider would do to it.
    pub fn replay_stroke(
        &mut self,
        samples: &[stark_engine::command::InputSample],
        seed: u64,
        rope: f32,
    ) -> bool {
        self.engine
            .replay_stroke_seeded(stark_engine::command::Tool::Brush, samples, seed, rope)
            .is_some()
    }

    /// The handle the element composites.
    pub fn surface(&self) -> WgpuSurfaceHandle {
        self.surface.clone()
    }

    /// Render into the back buffer and swap it to the front. Answers whether the
    /// viewport moved under it, which is what tells the caller the stroke has to be
    /// laid out again.
    pub fn paint(&mut self) -> bool {
        let Some((target, size)) = self.surface.back_view_with_size() else {
            return false;
        };
        let moved = size != self.viewport;
        if moved {
            self.viewport = size;
            self.engine
                .process(ViewCommand::Resize(Extent2::new(size.0, size.1)));
        }
        self.engine.render(&target);
        self.surface.swap_buffers();
        moved
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
