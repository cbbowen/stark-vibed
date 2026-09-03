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

use stark_chrome::brush_config::{BrushEffectType, MAX_FLOW, MAX_RADIUS, MIN_RADIUS};
use stark_chrome::commands::{Bindings, Command};
use stark_chrome::drags::{DragAction, DragBindings, DragButton, DragChord};
use stark_chrome::input as chrome_input;
use stark_chrome::keys::Mods;
use stark_engine::ObservableState;
use stark_engine::ViewTransform;
use stark_engine::command::{DocCommand, GestureCommand, InputSample, Tool};
use stark_model::Vec2;
use wgpui::{
    AnyElement, Context, FocusHandle, KeyDownEvent, MouseButton, MouseDownEvent, MouseMoveEvent,
    MouseUpEvent, Pixels, Point, Render, Window, div, prelude::*, rgb, wgpu_surface,
};

use crate::brush::Brush;
use crate::files::{self, Done};
use crate::layers::{self, Act};
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

/// How far one press of the bracket keys moves the brush's size, as a factor.
///
/// A ratio rather than a step, because size is perceived logarithmically: a pixel
/// added to a 4-px tip is a quarter of it and nothing at all to a 400-px one. The same
/// figure the web frontend steps by.
const SIZE_STEP: f32 = 1.1;

/// How much of the size range one horizontal pixel of a tuning drag spends, as an
/// exponent — so the knob moves multiplicatively, for `SIZE_STEP`'s reason.
///
/// `MIN_RADIUS..MAX_RADIUS` is about nine doublings, and this spends them over some
/// 900 px: a drag across a window covers the range once, and a short one is fine.
const TUNE_SIZE_PER_PX: f32 = 0.007;

/// How much flow one vertical pixel spends — the range over some 300 px, which is
/// shorter because there is far less of it to cross.
const TUNE_FLOW_PER_PX: f32 = 0.01;

/// What a press took hold of.
enum Held {
    /// A stroke on the canvas.
    Stroke,
    /// A knob on the panel, kept for the whole drag — see the module note.
    Knob(Knob),
    /// The layers panel's opacity track.
    Opacity,
    /// A **bound modifier drag** over the canvas (§18.1.9): the size sideways, the
    /// flow up and down, from where the press landed.
    Tune {
        /// Where the drag started, in logical px, and the tune it started from — so
        /// a long drag is one map from the press rather than a chain of steps, which
        /// is `stark_chrome::transform`'s rule applied to a knob.
        from: Point<Pixels>,
        size: f32,
        flow: f32,
    },
}

pub struct Canvas {
    /// `None` when wgpui is not on its wgpu renderer, so there is no device to paint
    /// with — reported on screen rather than as a panic.
    renderer: Option<Renderer>,
    /// The tool in hand and the library it can be swapped for.
    brush: Brush,
    /// What the pointer is holding, if anything.
    held: Option<Held>,
    /// This browser's chord table, and the drag table beside it — both shipped
    /// defaults for now: rebinding needs a settings surface, which is N8's.
    bindings: Bindings,
    drags: DragBindings,
    /// The keyboard needs somewhere to be focused, or nothing is dispatched at all.
    focus: FocusHandle,
    /// Whether the canvas owes the surface a frame.
    dirty: bool,
    /// Where the panel's controls were laid out, as of the last painted frame — the
    /// panel measures rather than predicts, and this is where it reports (`panel`).
    regions: Regions,
    /// The same, for the layers panel on the other side.
    layer_regions: layers::Regions,
    /// The engine's projection, refreshed after every command — what the layers
    /// panel draws and what a command's gate reads (§5).
    ///
    /// Kept rather than asked per frame: `observe()` walks the roster, and a frame
    /// that changed nothing would rebuild it for a panel that would draw the same.
    obs: Option<ObservableState>,
    /// The file this window holds, once one has been saved or opened. `None` for a
    /// document that has never been written — which is what the title says.
    path: Option<std::path::PathBuf>,
    /// The revision this client last wrote out. The other half of "is there anything
    /// to lose" (`stark_chrome::files::unsaved`); the engine supplies the first.
    written: u64,
    /// What the window's title bar last said. Kept so the title is set when it
    /// *changes* rather than every frame: a title is a platform call, and the frame
    /// loop runs whether or not the document moved.
    title: String,
    /// What went wrong, if anything, since the last act. Shown on the title bar for
    /// want of anywhere better — see [`Canvas::report`].
    failure: Option<String>,
    /// The file act in flight, if any. **Held rather than detached**: a wgpui `Task`
    /// cancels when it is dropped, and a save dropped mid-dialog is a file the user
    /// asked for and did not get.
    file_task: Option<wgpui::Task<()>>,
    /// Which groups this client has folded away. The panel's own state, not the
    /// document's: a collaborator's fold is theirs (§17.4).
    collapsed: std::collections::HashSet<stark_model::document::LayerId>,
    /// The clock `InputSample::time` is read off, and the raw reading it counts
    /// from. `quanta` rather than `std::time::Instant` (clippy.toml): this binary
    /// is native-only, but the clock the rest of the tree uses is the one that
    /// works in both places, and it is already compiled here through the engine's
    /// timing layer (§7.1).
    clock: quanta::Clock,
    epoch: u64,
}

impl Canvas {
    pub fn new(window: &mut Window, cx: &mut Context<'_, Self>) -> Self {
        let mut renderer = Renderer::new(window);
        let brush = Brush::default();
        if let Some(r) = renderer.as_mut() {
            r.process(brush.set());
        }
        let clock = quanta::Clock::new();
        let epoch = clock.raw();
        let focus = cx.focus_handle();
        let obs = renderer.as_ref().map(Renderer::observe);
        Self {
            renderer,
            brush,
            held: None,
            bindings: Bindings::default(),
            drags: DragBindings::default(),
            focus,
            // The first frame has a canvas nobody has painted yet.
            dirty: true,
            regions: Regions::default(),
            layer_regions: layers::Regions::default(),
            obs,
            path: None,
            written: 0,
            title: String::new(),
            failure: None,
            file_task: None,
            collapsed: std::collections::HashSet::new(),
            clock,
            epoch,
        }
    }

    fn press(&mut self, ev: &MouseDownEvent, window: &mut Window, cx: &mut Context<'_, Self>) {
        // The layers panel is on the right, so it is asked first for the same reason
        // the brush panel is: a press is paint only where neither column claims it.
        if let Some(region) = layers::hit(&self.layer_regions, ev.position) {
            if region == layers::Region::Opacity {
                self.held = Some(Held::Opacity);
                if let Some(f) = layers::opacity_at(&self.layer_regions, ev.position) {
                    self.set_opacity(f, cx);
                }
            } else {
                self.act(region, cx);
            }
            return;
        }
        if self.over_layers(window, ev.position) {
            return;
        }

        // Then the brush panel: its column is where a press stops being paint.
        match panel::hit(&self.regions, ev.position) {
            Some(Region::File(i)) => {
                if let Some(command) = panel::FILE_ACTS.get(i) {
                    self.run(*command, window, cx);
                }
                return;
            }
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

        // The drag table before the paint path, exactly as the web canvas asks it:
        // a modified press is a *gesture*, and which one is the table's answer rather
        // than a ladder of modifier tests here (§25.3).
        let mods = Mods {
            ctrl: ev.modifiers.control || ev.modifiers.platform,
            shift: ev.modifiers.shift,
            alt: ev.modifiers.alt,
        };
        let chord = DragChord {
            mods,
            button: DragButton::Left,
        };
        if self.drags.lookup(mods, DragButton::Left) == Some(DragAction::TuneBrush) {
            let _ = chord;
            self.held = Some(Held::Tune {
                from: ev.position,
                size: self.brush.tune.size,
                flow: self.brush.tune.flow,
            });
            return;
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
            Some(Held::Opacity) => {
                if let Some(f) = layers::opacity_at(&self.layer_regions, ev.position) {
                    self.set_opacity(f, cx);
                }
            }
            Some(Held::Tune { from, size, flow }) => {
                // Both knobs from the press rather than from the last move, so a long
                // drag is one map and rounding cannot walk over its length.
                let dx = f32::from(ev.position.x) - f32::from(from.x);
                let dy = f32::from(ev.position.y) - f32::from(from.y);
                self.brush.tune.size =
                    (size * (TUNE_SIZE_PER_PX * dx).exp()).clamp(MIN_RADIUS, MAX_RADIUS);
                // Up is more, which is the direction every slider in the app grows in
                // and the opposite of the screen's y.
                self.brush.tune.flow = (flow - dy * TUNE_FLOW_PER_PX).clamp(0.0, MAX_FLOW);
                self.send_brush(cx);
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
        // A committed stroke is a document change like any other: the roster it
        // may have added to has to reach the panel.
        if matches!(self.held.take(), Some(Held::Stroke)) {
            self.send(GestureCommand::End, cx);
        } else {
            self.repaint(cx);
        }
    }

    /// Whether a position is over the layers panel's column at all.
    ///
    /// The canvas ends where this begins, so a press it does not want is still not
    /// paint — the same bargain `panel::within` makes on the other side, measured
    /// from the right because that is the edge this column is pinned to.
    fn over_layers(&self, window: &Window, at: Point<Pixels>) -> bool {
        let right = f32::from(window.viewport_size().width);
        f32::from(at.x) >= right - layers::WIDTH
    }

    /// Do what a press on the layers panel means.
    ///
    /// The *meaning* is `layers::act`, which is a function over the rows so that it
    /// can be tested; what is here is the two things it cannot do — send the command,
    /// and fold a group, which is this client's own state rather than the document's.
    fn act(&mut self, region: layers::Region, cx: &mut Context<'_, Self>) {
        let rows = self.rows();
        let active = self.obs.as_ref().map(|o| o.active_layer);
        match layers::act(region, &rows, active) {
            Some(Act::Doc(command)) => self.send(command, cx),
            Some(Act::Peer(command)) => self.send(command, cx),
            Some(Act::Fold(id)) => {
                if !self.collapsed.remove(&id) {
                    self.collapsed.insert(id);
                }
                self.repaint(cx);
            }
            None => {}
        }
    }

    /// Set the selected layer's opacity, previewing per sample.
    ///
    /// One command per pointer move and one undo step for the whole drag is what
    /// `preview` buys the web app (§14.6); this sends the document command each time,
    /// which is honest but coarse — the engine coalesces nothing, so a drag is a run
    /// of history entries. The preview pair is a stage of its own.
    fn set_opacity(&mut self, opacity: f32, cx: &mut Context<'_, Self>) {
        let Some(id) = self.obs.as_ref().map(|o| o.active_layer) else {
            return;
        };
        self.send(DocCommand::SetLayerOpacity(id, opacity), cx);
    }

    /// The rows the layers panel draws, worked out by the tree.
    fn rows(&self) -> Vec<stark_chrome::layer_tree::Row> {
        match &self.obs {
            Some(o) => stark_chrome::layer_tree::rows(&o.layers, &self.collapsed),
            None => Vec::new(),
        }
    }

    /// Send a command and take the engine's answer back.
    ///
    /// **Every document change goes through here**, which is what keeps the
    /// projection and the panel in step: a command that moved state without
    /// refreshing `obs` would leave the panel drawing the state before it, which is
    /// the failure §4 names and the web frontend's `dispatch` exists to rule out.
    fn send(
        &mut self,
        command: impl Into<stark_engine::command::InputCommand>,
        cx: &mut Context<'_, Self>,
    ) {
        if let Some(r) = self.renderer.as_mut() {
            r.process(command);
            self.obs = Some(r.observe());
        }
        self.repaint(cx);
    }

    /// Whether the document holds committed work no file has (§8).
    fn unsaved(&self) -> bool {
        self.obs
            .as_ref()
            .is_some_and(|o| stark_chrome::files::unsaved(o.edited, o.doc_revision, self.written))
    }

    /// Write the document, asking for a path unless this window already has one.
    ///
    /// Saving *over* the file you opened is the first thing a real path buys — the web
    /// app cannot have it, because a download has nowhere to go back to. Its converse,
    /// a Save-As that forces the ask, is not here: the registry has no such act, and
    /// inventing one this frontend alone answers would put the two apps' vocabularies
    /// out of step for a dialog (§25).
    fn save(&mut self, window: &mut Window, cx: &mut Context<'_, Self>) {
        let Some(r) = self.renderer.as_ref() else {
            return;
        };
        let bytes = match files::save_bytes(r) {
            Ok(bytes) => bytes,
            Err(e) => return self.settle(Done::Failed(e), window, cx),
        };
        // The revision those bytes are of, read *before* anything asynchronous: by
        // the time a dialog answers the hand may have painted again, and marking that
        // revision written would call a stroke saved that no file holds.
        let revision = self.obs.as_ref().map_or(0, |o| o.doc_revision);
        if let Some(path) = self.path.clone() {
            let done = match files::write(&path, &bytes) {
                Ok(path) => Done::Saved { path, revision },
                Err(e) => Done::Failed(e),
            };
            return self.settle(done, window, cx);
        }
        // The dialogs are the *app's*, not the window's — one file picker at a time
        // per process is what every platform gives.
        let ask = cx.prompt_for_new_path(
            &self.directory(),
            Some(&stark_chrome::files::default_name()),
        );
        self.file_task = Some(cx.spawn_in(window, async move |this, cx| {
            let done = match ask.await {
                Ok(Ok(Some(path))) => match files::write(&path, &bytes) {
                    Ok(path) => Done::Saved { path, revision },
                    Err(e) => Done::Failed(e),
                },
                Ok(Ok(None)) => Done::Cancelled,
                Ok(Err(e)) => Done::Failed(format!("the save dialog failed: {e}")),
                // The sender went without answering — the window closed under the
                // dialog. Nothing to report and nothing to write.
                Err(_) => Done::Cancelled,
            };
            let _ = this.update_in(cx, |this, window, cx| this.settle(done, window, cx));
        }));
    }

    /// Replace the document with one read from disk.
    fn open(&mut self, window: &mut Window, cx: &mut Context<'_, Self>) {
        let ask = cx.prompt_for_paths(wgpui::PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: None,
        });
        self.file_task = Some(cx.spawn_in(window, async move |this, cx| {
            let done = match ask.await {
                Ok(Ok(Some(paths))) => match paths.into_iter().next() {
                    Some(path) => match files::read(&path) {
                        Ok(bytes) => Done::Opened { path, bytes },
                        Err(e) => Done::Failed(e),
                    },
                    None => Done::Cancelled,
                },
                Ok(Ok(None)) => Done::Cancelled,
                Ok(Err(e)) => Done::Failed(format!("the open dialog failed: {e}")),
                Err(_) => Done::Cancelled,
            };
            let _ = this.update_in(cx, |this, window, cx| this.settle(done, window, cx));
        }));
    }

    /// Write a picture of the document (§15.6).
    ///
    /// The render starts here and is *awaited* in the task, which is the borrow
    /// bargain `Engine::export` is built for: the future does not hold the renderer,
    /// so the window goes on painting while the GPU→CPU copy is in flight.
    fn export(&mut self, window: &mut Window, cx: &mut Context<'_, Self>) {
        let revision = self.obs.as_ref().map_or(0, |o| o.doc_revision);
        let (frame, scale, background, content) = files::EXPORT;
        let Some(r) = self.renderer.as_mut() else {
            return;
        };
        let render = match r.export(frame, scale, background, content) {
            Ok(render) => render,
            Err(e) => {
                let done = Done::Failed(format!("could not render the picture: {e}"));
                return self.settle(done, window, cx);
            }
        };
        let ask = cx.prompt_for_new_path(&self.directory(), Some("painting.png"));
        self.file_task = Some(cx.spawn_in(window, async move |this, cx| {
            let image = render.await;
            let done = match (ask.await, image) {
                (Ok(Ok(Some(path))), Ok(image)) => match files::encode(&image, &path) {
                    Ok(bytes) => match std::fs::write(&path, bytes) {
                        Ok(()) => Done::Exported { revision },
                        Err(e) => Done::Failed(format!("could not write {}: {e}", path.display())),
                    },
                    Err(e) => Done::Failed(e),
                },
                (Ok(Ok(None)), _) | (Err(_), _) => Done::Cancelled,
                (Ok(Err(e)), _) => Done::Failed(format!("the export dialog failed: {e}")),
                (_, Err(e)) => Done::Failed(format!("could not render the picture: {e}")),
            };
            let _ = this.update_in(cx, |this, window, cx| this.settle(done, window, cx));
        }));
    }

    /// Where a dialog should open: the last file's folder, or the working directory.
    fn directory(&self) -> std::path::PathBuf {
        self.path
            .as_ref()
            .and_then(|p| p.parent())
            .map(std::path::Path::to_path_buf)
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default())
    }

    /// Take a finished file act back into the view.
    ///
    /// **One place**, whichever door the act came through — the synchronous save over
    /// a known path and the three that wait on a dialog all end here, so what a
    /// success does to the title and the clean/dirty mark is written once.
    fn settle(&mut self, done: Done, window: &mut Window, cx: &mut Context<'_, Self>) {
        self.failure = None;
        match done {
            Done::Saved { path, revision } => {
                self.path = Some(path);
                self.written = revision;
            }
            Done::Opened { path, bytes } => {
                if self.load(&bytes) {
                    self.path = Some(path);
                    self.written = self.obs.as_ref().map_or(0, |o| o.doc_revision);
                }
            }
            // A picture is not the document, but it is a copy of the work — so it
            // settles the same question the unsaved guard asks (§15.6).
            Done::Exported { revision } => self.written = revision,
            Done::Cancelled => {}
            Done::Failed(why) => self.report(why),
        }
        self.retitle(window);
        self.repaint(cx);
    }

    /// Replay a loaded log over the open document (§8); `false` if it was refused.
    fn load(&mut self, bytes: &[u8]) -> bool {
        let file = match stark_model::DocumentFile::from_bytes(bytes) {
            Ok(file) => file,
            Err(e) => {
                self.report(format!("could not open that file: {e}"));
                return false;
            }
        };
        let Some(r) = self.renderer.as_mut() else {
            return false;
        };
        // What the file names but does not carry. This frontend has no catalog to
        // resolve those out of yet (`crate::files`), so anything owed is the end of
        // it — reported, with the painting on screen untouched, which is what makes a
        // refused file cost nothing.
        if !r.unresolved_content(&file).is_empty() {
            self.report("that painting uses content this build does not carry".to_string());
            return false;
        }
        if let Err(e) = r.load_document(&file) {
            self.report(format!("could not open that painting: {e}"));
            return false;
        }
        self.obs = Some(r.observe());
        // A load replaces the document wholesale, so everything the panel remembered
        // is stale — the folded groups above all, whose ids are gone.
        self.collapsed.clear();
        true
    }

    /// Say what went wrong, where a person will see it.
    ///
    /// The window title, for want of anywhere better: this frontend has no message
    /// surface, and a failure that reached only a log nobody is tailing is a failure
    /// nobody is told about. A proper report is a surface of its own (§25.7).
    fn report(&mut self, why: String) {
        self.failure = Some(why);
    }

    /// Put the file's name — or the last failure — on the window, if it has changed.
    fn retitle(&mut self, window: &mut Window) {
        let title = match &self.failure {
            Some(why) => format!("{why} — Stark"),
            None => files::window_title(self.path.as_deref(), self.unsaved()),
        };
        if title != self.title {
            window.set_window_title(&title);
            self.title = title;
        }
    }

    /// Run whatever the shipped chord table says this keystroke asks for.
    ///
    /// **The whole of the keyboard, and it is nine lines**, because the table is
    /// shared: what Ctrl+Z means was settled once (§25) and this frontend only has to
    /// say what a keystroke *is* (`crate::keys`) and what an act *does* below.
    fn key(&mut self, ev: &KeyDownEvent, window: &mut Window, cx: &mut Context<'_, Self>) {
        let Some(command) = self.bindings.lookup(&crate::keys::stroke(&ev.keystroke)) else {
            return;
        };
        self.run(command, window, cx);
    }

    /// Do what a command means here.
    ///
    /// A short list, and short *honestly*: the registry has thirty-odd acts and this
    /// frontend answers the handful below. What the rest need is a surface — a
    /// selection, a gradient bar, a settings page — and each arrives with its own
    /// stage (§11.2). An act with nothing to act on is left alone rather than given a
    /// no-op arm, so the day it lands the compiler has nothing to say and the reader
    /// does.
    fn run(&mut self, command: Command, window: &mut Window, cx: &mut Context<'_, Self>) {
        let doc = match command {
            Command::Undo => Some(DocCommand::Undo),
            Command::Redo => Some(DocCommand::Redo),
            _ => None,
        };
        match command {
            Command::BrushSmaller => self.step_size(1.0 / SIZE_STEP, cx),
            Command::BrushLarger => self.step_size(SIZE_STEP, cx),
            Command::SaveDocument => self.save(window, cx),
            Command::OpenDocument => self.open(window, cx),
            Command::ExportImage => self.export(window, cx),
            _ => {}
        }
        if let Some(doc) = doc {
            self.send(doc, cx);
        }
    }

    /// Step the brush's size by a factor, clamped to the range the panel offers.
    fn step_size(&mut self, factor: f32, cx: &mut Context<'_, Self>) {
        self.brush.tune.size = (self.brush.tune.size * factor).clamp(MIN_RADIUS, MAX_RADIUS);
        self.send_brush(cx);
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
        // The dirty mark follows the document rather than the last file act, so the
        // title is refreshed with the frame — cheaply, since `retitle` only calls the
        // platform when the words changed.
        self.retitle(window);

        let dragging = match self.held {
            Some(Held::Knob(k)) => Some(k),
            _ => None,
        };
        let chrome = panel::brush_panel(&self.brush, dragging, EFFECTS, &self.regions);
        let rows = self.rows();
        let roster = layers::layers_panel(self.obs.as_ref(), &rows, &self.layer_regions);

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
            // The keyboard answers whatever has focus, and nothing has it unless
            // something asks: an unfocused window dispatches no chord at all.
            .track_focus(&self.focus)
            .on_key_down(cx.listener(Self::key))
            .child(chrome)
            .child(
                div()
                    .flex_1()
                    .h_full()
                    .child(wgpu_surface(r.surface()).size_full()),
            )
            .child(roster)
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
