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

use stark_engine::ObservableState;
use stark_engine::ViewTransform;
use stark_engine::command::{
    DocCommand, GestureCommand, HoverReport, InputSample, Tool, ViewCommand,
};
use stark_model::AssetNeed;
use stark_model::document::{BrushShape, FillOp, GuideId, SelectionOp, ShapeAction};
use stark_model::geom::Vec2;
use stark_model::{AssetId, Srgb, SubstrateId};
use stark_pen::{Claim, Phase, Pose, Report, Tablet};
use stark_ui::assets;
use stark_ui::brush_config::{BrushEffectType, MAX_RADIUS, MIN_RADIUS};
use stark_ui::commands::{Bindings, Command, Gate, VisibilityToggle};
use stark_ui::drags::{DragAction, DragBindings, DragButton};
use stark_ui::input as chrome_input;
use stark_ui::keys::Mods;
use stark_ui::lighting as light;
use stark_ui::nav;
use stark_ui::panels::PanelId;
use stark_ui::prefs::{Hdr, Prefs};
use stark_ui::slots::{self, Grip};
use stark_ui::transform::{Bands, Family, Grab, Hint, Switch, TransformUi};
use wgpui::{
    AnyElement, Context, DispatchPhase, FocusHandle, KeyDownEvent, KeyUpEvent,
    ModifiersChangedEvent, MouseButton, MouseDownEvent, MouseExitEvent, MouseMoveEvent,
    MouseUpEvent, Pixels, Point, Render, ScrollDelta, ScrollWheelEvent, Subscription, Window,
    canvas, div, point, prelude::*, px, rgb, wgpu_surface,
};
use wgpui_component::button::{Button, ButtonVariants};
use wgpui_component::dialog::{DialogAction, DialogClose, DialogFooter};
use wgpui_component::input::{Input, InputState};
use wgpui_component::notification::Notification;
use wgpui_component::{Root, WindowExt};

use crate::brush::Brush;
use crate::brush_editor::{self, Editor};
use crate::collab::{self, Collab};
use crate::color;
use crate::controls::Controls;
use crate::files::{self, Done};
use crate::gallery;
use crate::guides;
use crate::layers::{self, Act};
use crate::lighting;
use crate::menu;
use crate::navigator;
use crate::palette;
use crate::panel::{self, Knob, Region, Regions, Side};
use crate::pick;
use crate::render::{Preview, Renderer};
use crate::select;
use crate::slots::Rack;
use crate::transform;

/// How far one press of the bracket keys moves the brush's size, as a factor.
///
/// A ratio rather than a step, because size is perceived logarithmically: a pixel
/// added to a 4-px tip is a quarter of it and nothing at all to a 400-px one. The same
/// figure the web frontend steps by.
const SIZE_STEP: f32 = 1.1;

/// Something the window has to say about the last act — a failure, or the one kind
/// of success that leaves nothing on screen. Queued by [`Canvas::report`] and
/// [`Canvas::say`] and handed to the widget layer's notifications on the next frame
/// ([`Canvas::render`]), since a notice is raised from places that hold no window.
enum Notice {
    Failed(String),
    Said(String),
}

impl From<Notice> for Notification {
    fn from(notice: Notice) -> Self {
        match notice {
            Notice::Failed(why) => Notification::error(why),
            Notice::Said(what) => Notification::info(what),
        }
    }
}

/// What a press took hold of.
enum Held {
    /// A stroke on the canvas.
    ///
    /// `restore` is the effect the brush wore before a stylus arrived tail-first: the
    /// eraser end swaps the effect for the length of one contact and puts it back
    /// (§18.1.8), which is the pen being turned over rather than the tool being
    /// changed — so nothing here takes the preset's name off it.
    Stroke { restore: Option<BrushEffectType> },
    /// A shape gesture — a marquee or a lasso — which is the *same* engine seam as a
    /// stroke and differs only in the tool it opened with and in fitting no curve
    /// (§6.8). `restore` is the action a held modifier borrowed for this one gesture,
    /// to be put back when it ends.
    Shape { restore: Option<ShapeAction> },
    /// One of the color picker's two controls, with what the press meant — see
    /// `stark_ui::color::Grab`, which decides that once and holds it.
    Pick {
        region: color::Region,
        grab: stark_ui::color::Grab,
    },
    /// A view drag — a pan or the scrubby zoom (§18.1.7). What it *is* was decided
    /// at the press and is held for the gesture, so letting go of the accelerator
    /// halfway through a zoom does not hand the canvas to the pan under a moving
    /// hand (`stark_ui::nav`).
    Navigate {
        mode: nav::Mode,
        last: Point<Pixels>,
    },
    /// A drag on the transform widget. The whole of what it does is
    /// `Grab::follow` — see `crate::transform`. Boxed: a warp grab carries two whole
    /// 4×4 meshes — where the drag started and the last shape the mesh could take —
    /// and a solved basis, which would otherwise be the size of every other thing a
    /// press can hold put together.
    Transform(Box<Grab>),
    /// The eyedropper (§18.0.2): the press samples the canvas instead of painting on
    /// it, and the drag keeps sampling — so a color is picked up without putting the
    /// brush down. Carries nothing: where a sample is taken is where the pointer is,
    /// and what it is taken with is the bar's ([`Canvas::sampler`]).
    Sample,
    /// A drag on the navigator's miniature: the view follows the pointer around the
    /// piece. Carries nothing — where the view goes is a pure function of where the
    /// pointer is over the picture (`crate::navigator`), so there is no start to hold.
    Overview,
    /// A stroke on the brush editor's test canvas (`crate::brush_editor`). Carries
    /// nothing: the sibling engine holds the gesture, exactly as the main one holds a
    /// stroke, and where the pointer is is read off the measured surface each move.
    PreviewStroke,
    /// A **bound modifier drag** over the canvas (§18.1.9): the size sideways, the
    /// flow up and down, from where the press landed.
    ///
    /// The gesture itself is `stark_ui::tune`, shared with the web frontend — where
    /// the press landed, the tune it landed on, and the one knob it has committed to.
    /// Both apps ran their own arithmetic here until it was, and neither the rates nor
    /// the axis lock agreed (§11.2).
    Tune(stark_ui::tune::Tune),
}

pub struct Canvas {
    /// `None` when wgpui is not on its wgpu renderer, so there is no device to paint
    /// with — reported on screen rather than as a panic.
    renderer: Option<Renderer>,
    /// The tool in hand and the library it can be swapped for.
    brush: Brush,
    /// The ten brushes under the hand (§18.1.8, `crate::slots`): what each digit
    /// holds, what is holding one down, and whether the rack is pinned up.
    ///
    /// **Not a member of `hidden`**, though its pin is a Window-menu row like the
    /// shelves': the rack floats over the painting rather than taking a column's room,
    /// so pinning it moves nothing about where the canvas begins ([`Canvas::origin`]).
    rack: Rack,
    slot_regions: crate::slots::Regions,
    /// What the pointer is holding, if anything.
    held: Option<Held>,
    /// The widget layer's dials and picker for the panels (§11.1), and the
    /// subscriptions that make them heard.
    controls: Controls,
    /// This client's chord table, and the drag table beside it — both read off the
    /// store at start (§25.6), so a table this client has kept is the one its chords
    /// and its presses answer. Rebinding *from here* still needs a settings surface,
    /// which is N8's; what the load buys today is that the two windows read one
    /// record the same way, and that the day the surface lands durability is already
    /// structural.
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
    /// Where the picker stands (`crate::color`) — held rather than read back off the
    /// brush, because a color coming through sRGB cannot say what hue a grey was.
    wheel: color::Wheel,
    /// What the eyedropper's next sample is taken with (§18.0.2): how far it sees,
    /// what the reach runs through, and how much canvas it averages.
    ///
    /// A field where the web app keeps three signals, which is the same value either
    /// way — how a frontend *stores* the options is its own, what they mean is
    /// `stark_ui::pick`'s (§11.2).
    sampler: stark_ui::pick::Sampler,
    /// Whether a sample is in flight. **One at a time**: a pick is a render plus an
    /// asynchronous readback and a picking drag asks for one per pointer move, so a
    /// move arriving while the last is still settling is dropped rather than queued
    /// ([`sample`](Canvas::sample)).
    sampling: bool,
    /// Its two pictures, kept between frames: a wheel is `FIELD_N²` gamut lookups.
    pictures: color::Pictures,
    /// Which shelves this client has folded to their title bar, remembered across
    /// sessions (`stark_ui::visibility`).
    ///
    /// Keyed by [`VisibilityToggle`] rather than by `PanelId` because the columns
    /// stack the navigator beside the panels and it folds like one — see
    /// `crate::visibility`.
    folded: std::collections::HashSet<VisibilityToggle>,
    /// Which it has put away entirely — the Window menu's answer (`crate::menu`),
    /// kept across sessions beside the fold above (`crate::visibility`).
    ///
    /// The one piece of chrome state the *canvas geometry* depends on. A column that
    /// is not built is room the surface takes, so where a press lands in the picture
    /// is a function of this set ([`Canvas::origin`]) — which is the difference
    /// between hiding a docked shelf and hiding a floating panel.
    hidden: std::collections::HashSet<VisibilityToggle>,
    /// Whether space is down — the modifier that turns a left drag into a pan
    /// (§18.1.7).
    ///
    /// Tracked rather than read off the event, because a `MouseDownEvent` carries
    /// the *modifier* keys and space is not one of them. So the keyboard is where it
    /// is learnt, which means it has to be un-learnt on focus loss too: a window that
    /// loses focus mid-hold never sees the key go up, and the next press would pan
    /// instead of paint.
    space: bool,
    /// The menu whose rows are showing, if any (`crate::menu`). Not remembered: a
    /// menu is open for the length of one decision.
    menu_open: Option<usize>,
    menu_regions: menu::Regions,
    /// The same, for the Select section, the transform bar, the two galleries and
    /// the three shelves that came after them.
    select_regions: select::Regions,
    /// The Select section's *acts*, which are not in the section: they are a bar over
    /// the canvas, up only while there is a selection (`crate::select`). A second
    /// list rather than a second enum — the two are built at different times into
    /// different places, and only ever one of them holds a `select::Region::Act`.
    select_bar_regions: select::Regions,
    color_regions: color::Regions,
    bar_regions: transform::Regions,
    gallery_regions: gallery::Regions,
    lighting_regions: lighting::Regions,
    guide_regions: guides::Regions,
    nav_regions: navigator::Regions,
    pick_regions: pick::Regions,
    /// The guide the Guides shelf's dressing acts on, if this client has taken one up.
    /// Resolved against the roster every frame (`guides::chosen`), so a guide removed
    /// under an undo leaves the tracks pointed at the newest rather than at nothing.
    guide: Option<GuideId>,
    /// Where the navigator's miniature sits in canvas space, and how large it is
    /// drawn — four numbers, because the picture itself is a surface on the GPU
    /// (`crate::navigator`).
    overview: Option<stark_ui::bounds::Overview>,
    /// The committed revision that miniature is a picture of, and when it was drawn.
    /// Together they are the whole of the refresh policy: draw when the document has
    /// moved, never under a live gesture, and at most once a settle.
    overview_at: u64,
    overview_when: f64,
    /// The two asset libraries this client keeps (§25.6) — the stamps and the
    /// substrates a person brought in, read from the store at start.
    shapes: Vec<assets::Entry>,
    substrates: Vec<assets::Entry>,
    /// The substrate the document is on, so a card can show as chosen. Read back from
    /// the engine would be better, but `ObservableState::substrate` is a
    /// `SubstrateId` and a card is keyed by the `AssetId` inside it.
    substrate: SubstrateId,
    /// What a press on the transform widget would take hold of, as of the last move
    /// — the cursor, and nothing else. Meaningless with no mode live.
    ///
    /// Not the hover *mark*, which is paint rather than chrome and is the engine's
    /// (§18.1.10, `Canvas::hover_to`): nothing here holds it.
    hint: Hint,
    /// The transform gesture in flight, if any (§16.6).
    ///
    /// **The mode is this one `Option`**, which is the web frontend's rule arrived at
    /// from the other side: over there four modes in four signals had to be kept
    /// mutually exclusive by every entry point remembering to, and the fix was one
    /// value that cannot hold two. This frontend has one mode so far, so "two at
    /// once" is not yet a state it could reach — but the shape is the one to grow
    /// into rather than a second `Option` beside this.
    mode: Option<TransformUi>,
    /// The engine's projection, refreshed after every command — what the layers
    /// panel draws and what a command's gate reads (§5).
    ///
    /// Kept rather than asked per frame: `observe()` walks the roster, and a frame
    /// that changed nothing would rebuild it for a panel that would draw the same.
    obs: Option<ObservableState>,
    /// This client's HDR choice (§6.5), from the record the web app keeps it in;
    /// the engine is told this met with the window (`Renderer::apply_hdr`).
    hdr: Hdr,
    /// The file this window holds, once one has been saved or opened. `None` for a
    /// document that has never been written — which is what the title says.
    path: Option<std::path::PathBuf>,
    /// The revision this client last wrote out. The other half of "is there anything
    /// to lose" (`stark_ui::files::unsaved`); the engine supplies the first.
    written: u64,
    /// What the window's title bar last said. Kept so the title is set when it
    /// *changes* rather than every frame: a title is a platform call, and the frame
    /// loop runs whether or not the document moved.
    title: String,
    /// What this window has to say about the last act, if anything, waiting for the
    /// frame that will show it — see [`Notice`].
    notice: Option<Notice>,
    /// Whether the command field has focus, which is exactly when its drop-down
    /// shows (`crate::palette`).
    searching: bool,
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
    /// The stylus, if this platform has one to give (§6.2, `stark-pen`).
    ///
    /// Attached whatever the machine has: a tablet with no backend and a backend with
    /// no digitizer both report nothing, and the mouse path below is unchanged by
    /// either — which is what keeps this one field rather than a mode.
    tablet: Tablet,
    /// What it has reported and this frame has not yet spent. A field so the drain
    /// costs no allocation per frame (`pump_pen`); empty between frames.
    pen: Vec<Report>,
    /// The brush editor, while it is open (`crate::brush_editor`).
    ///
    /// **The dialog is this one `Option`**, which is `mode`'s rule again one control
    /// down: closing it drops the sibling engine and its surface with it, so a dialog
    /// nobody has open costs no GPU memory rather than costing a texture pair for the
    /// session.
    editor: Option<Editor>,
    editor_regions: brush_editor::Regions,
    /// The one event that can take a key away without ever sending its keyup: the
    /// window losing focus. A hold left standing through an Alt+Tab would keep the
    /// borrowed brush for the rest of the session, with the key that would give it back
    /// now belonging to another window (§18.1.8) — the same class of bug the web
    /// frontend rules out on `blur`.
    ///
    /// Held rather than detached because a `Subscription` unsubscribes when it drops.
    _focus_out: Subscription,
    /// The shared session, if any, and where it stands (§12.4, `crate::collab`).
    ///
    /// One value rather than a phase beside a session, because they are one fact and a
    /// pair could come to hold half of each — a phase saying Shared with nothing to
    /// broadcast through is a client silently painting alone.
    collab: Collab,
}

impl Canvas {
    pub fn new(window: &mut Window, cx: &mut Context<'_, Self>) -> Self {
        let mut renderer = Renderer::new(window);
        // Before the first projection is read below, so the View menu opens knowing
        // whether its HDR row has anything to switch (§6.5).
        let hdr = stark_ui::storage::load::<Prefs>().map_or_else(Hdr::default, |p| p.hdr);
        if let Some(r) = renderer.as_mut() {
            r.apply_hdr(hdr, window.display_headroom());
        }
        // The shipped stamps go in before the first brush is chosen, because the
        // brush the app opens on may *be* one of them — a preset that resolved after
        // the first frame would paint one stroke round (`crate::brush`).
        //
        // Their ids are known without this (`stark_ui::assets::shipped_id`, hashed
        // at build time); what the import buys is the engine holding the bytes, so a
        // stroke can actually be stamped with one.
        let mut failure = None;
        if let Some(r) = renderer.as_mut() {
            for (row, png) in crate::assets::shipped_shape_files() {
                if let Err(e) = r.import_brush_id(png)
                    && failure.is_none()
                {
                    failure = Some(format!(
                        "the shipped shape “{}” did not load: {e}",
                        row.name
                    ));
                }
            }
        }
        // Both libraries, read to completion before the window is built. The backend
        // is synchronous file I/O (`crate::store`), so this parks on nothing — and
        // doing it here rather than in a task is what makes the roster below a fact
        // rather than a race.
        let shapes = pollster::block_on(assets::load::<assets::Shapes>());
        let substrates = pollster::block_on(assets::load::<assets::Substrates>());
        // The color the session opens on is the crate's, so the picker and the brush
        // start on one color rather than the picker showing something the first stroke
        // would not lay (`stark_ui::color::INITIAL_COLOR`).
        let wheel = color::Wheel::default();
        let mut brush = Brush::new(crate::assets::builtin_shapes());
        brush.tune.color = wheel.rgb(color::WHEEL_GAMUT);
        if let Some(r) = renderer.as_mut() {
            r.process(brush.set());
        }
        let clock = quanta::Clock::new();
        let epoch = clock.raw();
        // Onto the window winit made, which wgpui hands over as a raw handle and
        // nothing more — so this names no toolkit type and the crate behind it names
        // no `wgpui` one either.
        let tablet = Tablet::attach(&*window);
        let focus = cx.focus_handle();
        // **Focused before the first frame, not on the first press.** A key event is
        // dispatched down the path from the *focused* node, so with nothing focused
        // that path is the tree's root alone and this view's listeners are not on it —
        // every chord, and every frame the held modifiers owe (§18.0.2,
        // [`modifiers`](Self::modifiers)), waited for a click on the canvas to happen
        // first. Anything a field or a dialog focuses later is a descendant of this
        // view, so the path still runs through it; this is only about the state before
        // anything at all has been touched.
        focus.focus(window, cx);
        // Read off the shipped library rather than restated: a preset declares the
        // digit it ships on, so the rack and the panel's list are one table (§18.1.8).
        let rack = Rack::stored(&brush.library);
        let obs = renderer.as_ref().map(Renderer::observe);
        let controls = Controls::new(window, cx);
        // A hold has to end when the keyboard goes, and a window that has lost focus
        // sends no keyup — so the release is hung off focus leaving this view. A field
        // or dialog inside it is a descendant, and taking focus there is not leaving.
        let focus_out = cx.on_focus_out(&focus, window, |this, _, _window, cx| {
            this.release_slots(cx);
        });
        Self {
            renderer,
            brush,
            rack,
            slot_regions: crate::slots::Regions::default(),
            held: None,
            controls,
            bindings: Bindings::stored().unwrap_or_default(),
            // The offer's mark rides in the same record and is deliberately dropped:
            // this window makes no preset offer (there is no dialog for it), and a
            // field no surface reads would be a second authority over the one bit
            // §25.8 says is written when the dialog is *shown*.
            drags: stark_ui::drags::stored_drags()
                .map(|(table, _offer)| table)
                .unwrap_or_default(),
            focus,
            // The first frame has a canvas nobody has painted yet.
            dirty: true,
            regions: Regions::default(),
            layer_regions: layers::Regions::default(),
            wheel,
            sampler: stark_ui::pick::Sampler::default(),
            sampling: false,
            pictures: color::Pictures::default(),
            folded: crate::visibility::stored_folded(),
            hidden: crate::visibility::stored(),
            space: false,
            menu_open: None,
            menu_regions: menu::Regions::default(),
            select_regions: select::Regions::default(),
            select_bar_regions: select::Regions::default(),
            color_regions: color::Regions::default(),
            bar_regions: transform::Regions::default(),
            gallery_regions: gallery::Regions::default(),
            lighting_regions: lighting::Regions::default(),
            guide_regions: guides::Regions::default(),
            nav_regions: navigator::Regions::default(),
            pick_regions: pick::Regions::default(),
            guide: None,
            overview: None,
            overview_at: 0,
            // Behind the first settle, so the opening frame draws a miniature rather
            // than waiting a fifth of a second to admit there is a document.
            overview_when: f64::NEG_INFINITY,
            shapes,
            substrates,
            substrate: SubstrateId::Flat,
            hint: Hint::Move,
            mode: None,
            obs,
            hdr,
            path: None,
            written: 0,
            title: String::new(),
            notice: failure.map(Notice::Failed),
            searching: false,
            file_task: None,
            collapsed: std::collections::HashSet::new(),
            clock,
            epoch,
            tablet,
            pen: Vec::new(),
            editor: None,
            editor_regions: brush_editor::Regions::default(),
            _focus_out: focus_out,
            collab: Collab::default(),
        }
    }

    fn press(&mut self, ev: &MouseDownEvent, window: &mut Window, cx: &mut Context<'_, Self>) {
        // Read once, at the top: three of the branches below want it — the drag
        // table, the marquee's combine mode, and the picker's fine drag.
        let mods = mods_of(&ev.modifiers);

        // **A modal first, before even the menu bar.** The scrim is over the whole
        // window — the bar is dimmed under it — so a menu that opened there would be a
        // control on a surface that is not supposed to be reachable. Which is the same
        // two lines the menu itself spends below, one layer up.
        if self.editor.is_some() {
            self.press_editor(ev.position, window, cx);
            return;
        }

        // The menu bar next, and *before* the mode below: an open menu is over the
        // whole window, so a press it does not want is a press that closes it rather
        // than one that reaches the canvas. That is the whole of what "modal" means
        // here, and it is two lines rather than a catcher.
        match menu::hit(&self.menu_regions, ev.position) {
            Some(menu::Region::Title(i)) => {
                // The lit title closes it, which is the escape hatch every other
                // armed control in this app offers through the control that armed it.
                self.menu_open = if self.menu_open == Some(i) {
                    None
                } else {
                    Some(i)
                };
                return self.repaint(cx);
            }
            Some(menu::Region::Row(i, j)) => {
                self.menu_open = None;
                if let Some(command) = menu::command(i, j) {
                    self.run(command, window, cx);
                }
                return self.repaint(cx);
            }
            // The bar's own background, and anywhere else: both close an open menu.
            // A press on the bar that missed a title is not a press on the canvas
            // either, so it stops here rather than falling through to paint.
            Some(menu::Region::Bar) => {
                let was = self.menu_open.take();
                if was.is_some() {
                    return self.repaint(cx);
                }
                return;
            }
            None if self.menu_open.is_some() => {
                self.menu_open = None;
                return self.repaint(cx);
            }
            None => {}
        }

        // A live transform owns the canvas: its bar first, then the widget, and a
        // press that reached neither is still not paint. This is the web app's
        // full-viewport catcher without the catcher — one hit test in one place
        // rather than an element stacked over the surface (the module note).
        if let Some(ui) = self.mode {
            if let Some(region) = transform::hit(&self.bar_regions, ev.position) {
                self.bar_act(ui, region, cx);
                return;
            }
            if !self.over_chrome(window, ev.position)
                && let Some(view) = self.view()
            {
                let at = canvas_at(view, ev.position, self.origin(), window.scale_factor());
                let grab = Grab::take(ui, at, Bands::at(view.zoom));
                self.held = Some(Held::Transform(Box::new(grab)));
                return;
            }
        }

        // The selection's bar is over the canvas rather than over a column, so it is
        // asked before the columns and before anything else could call this paint.
        // It is mounted only when the mode's bar is not, so the two hit tests above
        // and here cannot both answer.
        match select::hit(&self.select_bar_regions, ev.position) {
            Some(select::Region::Act(i)) => {
                if let Some(command) = select::SELECT_ACTS.get(i) {
                    self.run(*command, window, cx);
                }
                return;
            }
            // The strip behind the chips, and nothing else reaches this list: a press
            // that missed a chip is still not a press on the picture.
            Some(_) => return,
            None => {}
        }

        // The eyedropper's bar takes the same edge as the selection's and is asked
        // beside it. The two are never up at once (`Canvas::render`), so the order
        // between these two tests carries no meaning.
        if let Some(region) = pick::hit(&self.pick_regions, ev.position) {
            if let Some(command) = pick::act(&mut self.sampler, region) {
                self.run(command, window, cx);
            }
            return self.repaint(cx);
        }

        // Every shelf of both columns, then the columns themselves. Order between the
        // hit tests is immaterial — the region lists are disjoint — but *all* of them
        // come before the catch-all below, which is what turns a press on a column
        // into nothing rather than into paint.
        if let Some(region) = layers::hit(&self.layer_regions, ev.position) {
            self.act(region, cx);
            return;
        }
        if let Some(region) = guides::hit(&self.guide_regions, ev.position) {
            self.guide_act(region, cx);
            return;
        }
        if let Some(region) = lighting::hit(&self.lighting_regions, ev.position) {
            self.light_act(region, window, cx);
            return;
        }
        if navigator::hit(&self.nav_regions, ev.position).is_some() {
            self.held = Some(Held::Overview);
            self.overview_to(ev.position, cx);
            return;
        }

        if let Some(region) = color::hit(&self.color_regions, ev.position) {
            let Some(at) = color::fraction_at(&self.color_regions, region, ev.position) else {
                return;
            };
            // Where the marker stands now, so a fine drag has somewhere to move
            // *from* — the press itself then picks nothing, which is the whole of
            // what makes a small adjustment possible.
            let held = match region {
                color::Region::Wheel => stark_ui::color::wheel_xy(self.wheel.hue, self.wheel.sat),
                color::Region::Track => (self.wheel.l, 0.5),
            };
            let grab = stark_ui::color::Grab::take(at, held, mods.shift);
            self.held = Some(Held::Pick { region, grab });
            self.pick(region, grab, at, cx);
            return;
        }

        // The galleries share the column too, and are asked before the Select section
        // for no reason beyond where they sit: the regions are disjoint.
        if let Some(region) = gallery::hit(&self.gallery_regions, ev.position) {
            self.gallery_act(region, window, cx);
            return;
        }

        // The Select section shares the brush panel's column, so it is asked with it.
        match select::hit(&self.select_regions, ev.position) {
            Some(select::Region::Tool(i)) => {
                if let Some(tool) = stark_ui::selection::SHAPE_TOOLS.get(i) {
                    self.run(select::tool_command(*tool), window, cx);
                }
                return;
            }
            Some(select::Region::Action(i)) => {
                if let Some(action) = stark_ui::selection::SHAPE_ACTIONS.get(i) {
                    // Picking what a shape *does* also hands back a tool to draw it
                    // with: all five answers are about a gesture that has not been
                    // made, and with the brush in hand there is nothing for one to be
                    // an answer about (§6.8).
                    self.send(ViewCommand::SetShapeAction(*action), cx);
                    self.arm_shape(cx);
                }
                return;
            }
            // The acts and the strip they sit on are the bar's, measured into its own
            // list above and never into this one.
            Some(select::Region::Act(_) | select::Region::Bar) => return,
            None => {}
        }

        // Then the brush panel: its column is where a press stops being paint.
        match panel::hit(&self.regions, ev.position) {
            Some(Region::Fold(what)) => {
                self.fold(what, cx);
                return;
            }
            Some(Region::Preset(i)) => {
                if let Some(name) = self.brush.library.get(i).map(|e| e.name.clone()) {
                    // A whole tool arriving from the library is what a held number is
                    // listening for: it counts as the hold's change even where it moves
                    // nothing, which is exactly the case of filling a slot with the
                    // brush already in hand (§18.1.8, `slots::Held::claim`).
                    self.claim_slot();
                    self.brush.wear(&name);
                    self.send_brush(cx);
                }
                return;
            }
            // Its own words and the registry's act (§25.1): a button here and a row in
            // the palette reach one command, so the two cannot come to mean different
            // things.
            Some(Region::Edit) => {
                self.run(Command::EditBrush, window, cx);
                return;
            }
            None if self.over_chrome(window, ev.position) => {
                // Somewhere on a column that is not a control. Not paint either.
                return;
            }
            None => {}
        }

        // The quick-brush rack, which stands over the canvas rather than beside it
        // (§18.1.8). Its rows record a rectangle only while it is **pinned**, so a
        // transient rack is never asked and the stroke it is standing over is never
        // swallowed — the same bargain the web frontend makes by granting the pointer
        // in its stylesheet.
        if let Some(region) = crate::slots::hit(&self.slot_regions, ev.position) {
            let now = self.elapsed();
            if self.rack.press(region, now) {
                self.repaint(cx);
            }
            return;
        }

        // Neither column claimed it, so this is the canvas — and what a press on the
        // canvas means is one function, because a stylus asks it too (`pump_pen`).
        self.open_canvas(ev.position, mods, None, window, cx);
    }

    /// Open whatever a press on the canvas itself means: a look around, a brush tune,
    /// or paint.
    ///
    /// **Two callers, and that is what it is for.** A mouse press arrives having got
    /// past every panel in [`press`](Self::press)'s ladder; a stylus contact arrives
    /// because the tablet was told which rectangle is the canvas and took only
    /// presses inside it (`stark_pen::Claim`). Which of the three a press means is a
    /// fact about the press rather than about the device that made it, so it is
    /// answered in one place — and the alternative was a second copy of this ladder
    /// that could disagree with the first about what a modifier does.
    ///
    /// `pen` is what the stylus reported, and it reaches three things: the sample's
    /// pressure and tilt, the resolution the fit is told to expect (§6.2), and
    /// whether the tail is the end facing the glass.
    fn open_canvas(
        &mut self,
        at: Point<Pixels>,
        mods: Mods,
        pen: Option<&Pose>,
        window: &mut Window,
        cx: &mut Context<'_, Self>,
    ) {
        // Read before anything borrows the renderer: where the canvas begins is a
        // question about the chrome, and the arms below are holding the engine.
        let origin = self.origin();
        // Navigation before paint, and before the drag table: a press that is
        // looking around is not a press on the picture, whatever else it would have
        // meant. Which press that is is `stark_ui::nav`'s answer.
        if let Some(mode) = nav::press(
            nav::Button::Left,
            screen_at(at, origin, window.scale_factor()),
            self.space,
            mods.ctrl,
        ) {
            self.held = Some(Held::Navigate { mode, last: at });
            // And this press is navigation, so the mark's promise of paint is over
            // (§18.1.10). One of the two places that clears it on a press — a press
            // that *paints* must not, or the stroke loses its run-up.
            self.clear_hover_mark(cx);
            return;
        }

        // The drag table before the paint path, exactly as the web canvas asks it:
        // a modified press is a *gesture*, and which one is the table's answer rather
        // than a ladder of modifier tests here (§25.3).
        match self.drags.lookup(mods, DragButton::Left) {
            Some(DragAction::TuneBrush) => {
                self.held = Some(Held::Tune(stark_ui::tune::Tune::press(
                    logical_at(at),
                    self.brush.tune,
                )));
                // The other one: the brush is moving rather than the pointer meaning
                // anything on the canvas (§18.1.9), so the mark under it stops being a
                // picture of the next stroke.
                self.clear_hover_mark(cx);
                return;
            }
            // The press samples the canvas instead of painting on it, and the drag
            // keeps sampling — the binding Clip Studio Paint and Rebelle both put on
            // Alt, so a color is picked up without putting the brush down (§18.0.2).
            //
            // `free` is what stands it down over a selection tool, where Alt is
            // already the subtract marquee (§6.8) and the press is *for* the shape —
            // the same answer the cursor and the bar were mounted on, asked of the
            // same value ([`pick_hand`](Self::pick_hand)). A declined press falls
            // through to the paint path exactly as an unbound chord does.
            Some(DragAction::PickColor) if DragAction::PickColor.claims(self.pick_hand()) => {
                self.held = Some(Held::Sample);
                // The press is not paint, so the mark promising it goes down with it
                // (§18.1.10).
                self.clear_hover_mark(cx);
                self.sample(at, window, cx);
                return;
            }
            // A sample this hand may not take falls through to the paint path exactly
            // as an unbound chord does — the arm above declined it, not the table.
            Some(DragAction::PickColor) => {}
            // The layer carry is a gesture this window has not got (§11.2, §16.11),
            // so its chord falls through to paint. Written out rather than left to a
            // `_`, because a fourth action added to the table has to be answered or
            // declined *here* — and because the chrome that stands down for this one
            // (`DragAction::shadows_paint`, the hover mark) is already promising
            // something this arm cannot yet deliver.
            Some(DragAction::PickAndTranslate) => {}
            None => {}
        }

        let tool = self.obs.as_ref().map_or(Tool::Brush, |o| o.tool);
        // A held modifier borrows the shape action for this one gesture — whether it
        // does is `stark_ui::selection`'s answer, and a `Some` is what has to be
        // put back on release.
        let shape_restore = if tool.is_selection() {
            let action = self
                .obs
                .as_ref()
                .map_or_else(Default::default, |o| o.shape_action);
            match stark_ui::selection::override_for(action, mods) {
                Some(next) => {
                    self.send(ViewCommand::SetShapeAction(next), cx);
                    Some(action)
                }
                None => None,
            }
        } else {
            None
        };

        // The stylus's tail erases while it is the end facing the glass (§18.1.8).
        // Sent *before* the gesture opens, because the engine takes the brush at
        // `Start` — and remembered rather than committed, because turning a pen over
        // is not choosing a different tool, so the preset keeps its name.
        //
        // Nothing is swapped without a renderer to swap it in: the gesture below bails
        // on the same condition, and a brush put into the eraser by a press that then
        // opened nothing is one no release would put back.
        let restore = if self.renderer.is_some()
            && pen.is_some_and(|p| p.inverted)
            && self.brush.config.effect != BrushEffectType::Erase
        {
            let was = std::mem::replace(&mut self.brush.config.effect, BrushEffectType::Erase);
            let command = self.brush.set();
            if let Some(r) = self.renderer.as_mut() {
                r.process(command);
            }
            Some(was)
        } else {
            None
        };

        let (scale, now) = (window.scale_factor(), self.elapsed());
        let smoothing = self.brush.config.smoothing;
        let Some(r) = self.renderer.as_mut() else {
            return;
        };
        let view = r.view();
        r.process(GestureCommand::Start {
            tool,
            sample: sample_at(view, at, origin, scale, now, pen),
            // Both are canvas-space lengths the frontend alone can state, and both
            // are mapped by `stark_ui::input` rather than here — which is the
            // point of that module: this frontend had its own copy of the rope's
            // constant and its own quadratic for exactly one commit (§11.2).
            //
            // Which device made the press is the other half `stark_ui::input` cannot
            // know, and it is a real difference: a mouse walks the screen in whole
            // pixels while a digitizer resolves well below one.
            tolerance: chrome_input::tolerance(view, resolution(pen)),
            // Zero for the shape tools, which fit no curve: a marquee's corner is
            // where the hand put it, and towing it would round the corner off.
            rope: if tool.is_selection() {
                0.0
            } else {
                chrome_input::rope(view, smoothing)
            },
        });
        self.held = Some(if tool.is_selection() {
            Held::Shape {
                restore: shape_restore,
            }
        } else {
            Held::Stroke { restore }
        });
        self.repaint(cx);
    }

    fn drag(&mut self, ev: &MouseMoveEvent, window: &mut Window, cx: &mut Context<'_, Self>) {
        self.move_to(ev.position, None, window, cx);
    }

    /// Follow whatever the press took hold of to `at`.
    ///
    /// [`open_canvas`](Self::open_canvas)'s other half, and split out for its reason:
    /// a gesture the stylus opened is one the stylus has to be able to move, and every
    /// arm below — the pan, the tune, the stroke — is the same work whichever device
    /// is asking.
    fn move_to(
        &mut self,
        at: Point<Pixels>,
        pen: Option<&Pose>,
        window: &mut Window,
        cx: &mut Context<'_, Self>,
    ) {
        // Sliding off the rack's trash backs out of the hold that would empty the slot
        // — what sliding off a button has always meant (§18.1.8). Asked before anything
        // else, because a press on it took hold of nothing else.
        if self.rack.moved(crate::slots::hit(&self.slot_regions, at)) {
            self.repaint(cx);
        }
        // [`open_canvas`](Self::open_canvas)'s reason, and one more: three of the arms
        // below want it, and a second reading is a second place to get it from.
        let origin = self.origin();
        // Once, before the match: the transform arm borrows `self.held` mutably and
        // so cannot ask `self` for it, and the resting arm wants the same value.
        let view = self.view();
        match self.held {
            Some(Held::Navigate { mode, last }) => {
                if let Some(command) = mode.moved(
                    screen_at(last, origin, window.scale_factor()),
                    screen_at(at, origin, window.scale_factor()),
                ) {
                    self.send(command, cx);
                }
                // The anchor moves with the hand for a pan and stays put for a zoom,
                // which is `Mode`'s own distinction — what this has to keep either way
                // is where the pointer was last seen.
                self.held = Some(Held::Navigate { mode, last: at });
            }
            Some(Held::Transform(ref mut grab)) => {
                // The view is read before the match (`view`), because the grab is
                // borrowed mutably here: it holds both the drag's start and the last
                // shape the family could express, so a long drag stays one map and
                // nothing outside has to hand either of them back
                // (`stark_ui::transform`).
                let Some(view) = view else { return };
                let held = canvas_at(view, at, origin, window.scale_factor());
                let next = grab.follow(held, Bands::at(view.zoom));
                self.compose(next, cx);
            }
            Some(Held::Pick { region, grab }) => {
                if let Some(f) = color::fraction_at(&self.color_regions, region, at) {
                    self.pick(region, grab, f, cx);
                }
            }
            // The drag goes on sampling, which is what makes this one gesture rather
            // than a press: `sample` drops a move that arrives while the last answer
            // is still settling.
            Some(Held::Sample) => self.sample(at, window, cx),
            // Held-and-dragged is one continuous request — "show me here" — which is
            // what makes the view follow the pointer instead of jumping to wherever it
            // is let go.
            Some(Held::Overview) => self.overview_to(at, cx),
            // A hand on the brush editor's test canvas, which is not on the document at
            // all: the sibling engine holds the gesture and the main one hears nothing
            // about it. Reachable by mouse only — the tablet is claimed to the canvas
            // rectangle, so a stylus never arrives over a dialog (`pen_claim`).
            Some(Held::PreviewStroke) => {
                let Some(pos) =
                    brush_editor::preview_at(&self.editor_regions, at, window.scale_factor())
                else {
                    return;
                };
                if let Some(editor) = self.editor.as_mut() {
                    editor.stroke_to(pos);
                }
                self.repaint(cx);
            }
            Some(Held::Tune(mut drag)) => {
                // The range is the in-force effect's (`BrushConfig::max_flow`), so a
                // full drag is a full knob whichever it is — and a liquify brush's
                // strength stops where its slider does rather than two thirds past it.
                let turn = drag.moved(logical_at(at), self.brush.config.max_flow());
                self.held = Some(Held::Tune(drag));
                if let Some(turn) = turn {
                    turn.write(&mut self.brush.tune);
                    self.send_brush(cx);
                }
            }
            Some(Held::Stroke { .. } | Held::Shape { .. }) => {
                let (scale, now) = (window.scale_factor(), self.elapsed());
                let Some(r) = self.renderer.as_mut() else {
                    return;
                };
                let view = r.view();
                r.process(GestureCommand::To {
                    sample: sample_at(view, at, origin, scale, now, pen),
                });
                self.repaint(cx);
            }
            // Resting. With a transform live, report what a press here would take
            // hold of, so the cursor says which of the three drags the widget is
            // offering at this point — the affine's rim, inside and outside are one
            // shape with three meanings, and nothing else distinguishes them.
            None => {
                if let (Some(ui), Some(view)) = (self.mode, view) {
                    let over = canvas_at(view, at, origin, window.scale_factor());
                    // `hint_at`, not a grab: a press on a warp surface solves a
                    // least-norm basis, and a hovering pointer asked for one of those
                    // per move to read three bits of it.
                    let hint = stark_ui::transform::hint_at(&ui, over, Bands::at(view.zoom));
                    if hint != self.hint {
                        self.hint = hint;
                        // The cursor is set during *paint*, so a changed hint owes a
                        // frame; an unchanged one owes nothing, which is what keeps a
                        // resting pointer from repainting the window.
                        self.repaint(cx);
                    }
                }
                // And the mark a press would lay, which is what a move with no gesture
                // behind it *is* (§18.1.10). A moving pointer therefore owes a frame
                // where the hint alone owed one only on a change — the price is
                // painting's, and it is paid by the hand: a window hears no move from a
                // hand that is still.
                self.hover_to(at, pen, window, cx);
            }
        }
    }

    /// End whatever the press took hold of — for a stroke, the one edge that commits
    /// an action (§4).
    fn release(&mut self, ev: &MouseUpEvent, _window: &mut Window, cx: &mut Context<'_, Self>) {
        // The rack's own click, and it is hit-tested afresh rather than remembered:
        // holding the trash removes the row while the pen is still down, so a release
        // has to land on whichever row has moved up under it — and pick nothing
        // (`slots::Rack::release`).
        if let Some(slot) = self
            .rack
            .release(crate::slots::hit(&self.slot_regions, ev.position))
        {
            self.pick_slot(slot, cx);
        }
        self.release_at(cx);
    }

    /// The same, for a lift that arrived off the tablet rather than off the mouse.
    ///
    /// Takes nothing but the context, which is why `release` could always have been
    /// written this way: what a release does depends on what is held and never on
    /// where the pointer was when it happened.
    fn release_at(&mut self, cx: &mut Context<'_, Self>) {
        match self.held.take() {
            // A committed stroke is a document change like any other: the roster it
            // may have added to has to reach the panel.
            Some(Held::Stroke { restore }) => {
                self.send(GestureCommand::End, cx);
                // The tail is off the glass, so the brush is what it was (§18.1.8).
                if let Some(effect) = restore {
                    self.brush.config.effect = effect;
                    self.send_brush(cx);
                }
            }
            Some(Held::Shape { restore }) => {
                self.send(GestureCommand::End, cx);
                // The borrowed action goes back *after* the gesture, which is also
                // after the gesture disarmed the tool (§6.8) — so this restores a
                // setting and re-arms nothing, which is the order that makes the
                // momentary rule hold under a modifier-drag.
                if let Some(action) = restore {
                    self.send(ViewCommand::SetShapeAction(action), cx);
                }
            }
            // A stroke on the test canvas commits to the sibling document and
            // becomes the stroke every later edit replays — the artist's own hand in
            // place of the seeded one (`crate::brush_editor`). A press that went
            // nowhere is not a stroke, and the one it interrupted has to be put back:
            // opening the gesture is what took it off the canvas.
            Some(Held::PreviewStroke) => {
                if self.editor.as_mut().is_some_and(Editor::end_stroke) {
                    self.repaint(cx);
                } else {
                    self.restroke(cx);
                }
            }
            // The sampler comes off the canvas and the answer to its last press does
            // not: the readback is detached, so a release cannot cancel the color the
            // press asked for ([`sample`](Self::sample)).
            Some(Held::Sample) => self.repaint(cx),
            // A transform is *not* committed on release: the gesture goes on being
            // composed until Done, which is what makes it one undo step however many
            // drags built it (§16.6).
            _ => self.repaint(cx),
        }
    }

    /// Lay the hover mark under a resting pointer (§18.1.10): the stroke a drag begun
    /// this instant would open, rendered where the press would land it — so the canvas
    /// says what the brush would do before the brush is put down, and says where the
    /// hand is while it is at it.
    ///
    /// What the report *contains* is `stark_ui::input::Hovering`, shared with the web
    /// canvas — the reach, and the full pressure a hovering hand does not report — and
    /// the window its heading is read from is the engine's (`Session::hover_to`). What
    /// is left here is where the pointer is and what this chrome has promised the press
    /// to.
    ///
    /// **Nothing on the paint path takes the mark down**, and that is a rule rather
    /// than an omission: a press takes the engine's hover window as the stroke's run-up
    /// (§6.2), and clearing the mark drops the window with it — the evidence the
    /// stroke's entry is smoothed through.
    ///
    /// **The mouse's alone so far.** `stark-pen` reports contact only, and over the
    /// claimed rectangle a hovering stylus's compatibility mouse message is swallowed
    /// (§11.3) — so a hovering pen lays no mark here yet. `pen` is threaded through
    /// rather than assumed away, because the lean a hovering pen would give the mark is
    /// owed rather than unwanted.
    fn hover_to(
        &mut self,
        at: Point<Pixels>,
        pen: Option<&Pose>,
        window: &Window,
        cx: &mut Context<'_, Self>,
    ) {
        // Everything a press here would mean instead, in [`press`](Self::press)'s own
        // order: a column is not the picture, a menu and the modal cover it, and a live
        // transform owns it. The mark comes *down* rather than merely not being
        // renewed — the one standing is the promise.
        if self.over_chrome(window, at)
            || self.menu_open.is_some()
            || self.editor.is_some()
            || self.mode.is_some()
        {
            return self.clear_hover_mark(cx);
        }
        let hand = chrome_input::Hovering {
            panning: self.space,
            // A held chord arms an act that reads the *shown* canvas back — the
            // eyedropper, and the layer carry on the day it lands. The mark is a
            // hypothesis about paint, so it has to be off the canvas before a press
            // can read one: the wrong color for the sample.
            shadowed: stark_ui::drags::armed(&self.drags, mods_of(&window.modifiers()))
                .is_some_and(DragAction::shadows_paint),
            // Reachable with **nothing held** — this is `move_to`'s resting arm — so a
            // sampler that is down cannot get here. The line that changes the day a
            // held touch resolves into one (§18.1.11).
            sampling: false,
            // No timeline in this frontend yet (§11.2).
            playing: false,
        };
        let (scale, now) = (window.scale_factor(), self.elapsed());
        let origin = self.origin();
        let Some(r) = self.renderer.as_ref() else {
            return;
        };
        let view = r.view();
        let sample = sample_at(view, at, origin, scale, now, pen);
        match hand.report(sample, chrome_input::tolerance(view, resolution(pen))) {
            Some(report) => self.send_hover(Some(report), cx),
            None => self.clear_hover_mark(cx),
        }
    }

    /// Watch for the pointer leaving the window, which is the one thing the mark needs
    /// and wgpui offers no element hook for (§18.1.10).
    ///
    /// A mark left standing under a cursor that is no longer there promises a press
    /// nobody is about to make — the web canvas takes it down on `pointerleave`, and a
    /// docked chrome catches most of the same cases by the move that lands on a column
    /// ([`hover_to`](Self::hover_to)). What is left over is the canvas's own edges,
    /// which is what this covers.
    ///
    /// A `canvas` of no size, because it draws nothing and is only somewhere to
    /// register from: `Window::on_mouse_event` is spent during *paint* and lasts one
    /// frame, so what it needs is a place in the tree that paints on every one.
    fn leaving(cx: &Context<'_, Self>) -> AnyElement {
        let this = cx.entity().downgrade();
        canvas(
            |_, _, _| (),
            move |_, (), window: &mut Window, _| {
                window.on_mouse_event(move |_: &MouseExitEvent, phase, _, cx| {
                    // Once per event, not once per phase.
                    if phase == DispatchPhase::Bubble {
                        this.update(cx, |canvas, cx| canvas.clear_hover_mark(cx))
                            .ok();
                    }
                });
            },
        )
        .w_0()
        .h_0()
        .into_any_element()
    }

    /// What this frontend knows about the hand that the drag table does not
    /// (`stark_ui::drags::Hand`).
    ///
    /// One place it is assembled, because three surfaces ask it — the cursor's
    /// promise, the bar's mounting and the press's own answer — and a promise made
    /// against one reading and kept against another is exactly the drift the shared
    /// predicate exists to stop.
    fn pick_hand(&self) -> stark_ui::drags::Hand {
        stark_ui::drags::Hand {
            panning: self.space,
            selecting: self.obs.as_ref().is_some_and(|o| o.tool.is_selection()),
            // No timeline in this frontend yet (§11.2) — the line that changes the day
            // there is one, and `Hand` carries the field so that day is one word.
            playing: false,
            sampling: matches!(self.held, Some(Held::Sample)),
            // Anything else the press already opened: a stroke, a pan, a knob, a
            // widget. The sampler is deliberately not among them — its own bar is
            // what `sampling` takes down, and the Color panel stays legible while it
            // is in use.
            busy: self.held.is_some() && !matches!(self.held, Some(Held::Sample)),
        }
    }

    /// Sample the canvas color under `at` and load the brush with it — the eyedropper
    /// (§18.0.2).
    ///
    /// **One sample at a time.** A pick is a render plus an asynchronous readback, and
    /// a picking drag asks for one per pointer move, so a move arriving while one is
    /// still in flight is *dropped rather than queued*: queueing would spend a GPU
    /// submit per move and let an older answer land after a newer one, and for a
    /// sampler being dragged only the latest answer matters anyway.
    fn sample(&mut self, at: Point<Pixels>, window: &Window, cx: &mut Context<'_, Self>) {
        if self.sampling {
            return;
        }
        // The *choice* is what the bar holds; which layer it means is resolved now,
        // against whichever layer is selected at the moment of the sample
        // (`stark_ui::pick::Sampler::options`).
        let options = self
            .sampler
            .options(self.obs.as_ref().map(|o| o.active_layer));
        let (origin, scale) = (self.origin(), window.scale_factor());
        let Some(r) = self.renderer.as_mut() else {
            return;
        };
        let view = r.view();
        let readback = r.pick_color(canvas_at(view, at, origin, scale), options);
        self.sampling = true;
        // Detached, which is the bargain `Renderer::pick_color` is shaped for: the
        // future holds no borrow of the renderer, so the window goes on painting
        // while the copy is in flight — and a release does not cancel the answer to
        // the press that asked for it.
        cx.spawn(async move |this, cx| {
            let picked = readback.await;
            let _ = this.update(cx, |this, cx| this.settle_sample(picked, cx));
        })
        .detach();
    }

    /// Take a finished sample into the brush.
    fn settle_sample(&mut self, picked: Option<[f32; 3]>, cx: &mut Context<'_, Self>) {
        self.sampling = false;
        // Nothing under the sampler leaves the brush as it was: bare canvas is the
        // substrate, not paint to pick up (§18.0.2).
        let Some(rgb) = picked else {
            return self.repaint(cx);
        };
        // The brush takes the sample **whole**, and the picker is only *seeded* from
        // it (`crate::color`) — the other way round from a typed color, which is a
        // wheel position being asked for. What comes off the canvas is paint that
        // exists, and in a Mixbox document (§6.7) rounding it to what a wheel can
        // produce is the difference between picking the mixture back up and picking a
        // display color.
        self.brush.tune.color = rgb;
        self.wheel = color::Wheel::of(color::WHEEL_GAMUT, rgb, self.wheel.hue);
        self.send_brush(cx);
    }

    /// Take the mark down, if one is up (§18.1.10).
    ///
    /// The peek is what makes this callable from anywhere: it runs on every move that
    /// lands on a column and on space's auto-repeat, and an idle call must spend
    /// neither a command nor a frame.
    fn clear_hover_mark(&mut self, cx: &mut Context<'_, Self>) {
        if self.renderer.as_ref().is_some_and(Renderer::hover_held) {
            self.send_hover(None, cx);
        }
    }

    /// The hover mark's own door, beside [`send`](Self::send) rather than through it
    /// (§18.1.10).
    ///
    /// The two things it does *not* do are the reason it exists. The projection cannot
    /// have moved — a hypothesis commits nothing — and presence never carries the mark
    /// (§17.9), so refreshing `obs` and broadcasting at pointer rate would be work for
    /// nobody. The frame is owed either way, because the mark is paint.
    fn send_hover(&mut self, report: Option<HoverReport>, cx: &mut Context<'_, Self>) {
        if let Some(r) = self.renderer.as_mut() {
            r.process(ViewCommand::PreviewHover(report));
        }
        self.repaint(cx);
    }

    /// Whether a position is over either column at all.
    ///
    /// The canvas is what is *between* them, so a press neither column's controls
    /// wanted is still not paint. The widths are this module's own (`panel::width`)
    /// rather than measured, for the reason [`Canvas::origin`] gives.
    fn over_chrome(&self, window: &Window, at: Point<Pixels>) -> bool {
        panel::within(
            at,
            panel::width(Side::Left, &self.hidden),
            panel::width(Side::Right, &self.hidden),
            f32::from(window.viewport_size().width),
        )
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
    pub(crate) fn set_opacity(&mut self, opacity: f32, cx: &mut Context<'_, Self>) {
        let Some(id) = self.obs.as_ref().map(|o| o.active_layer) else {
            return;
        };
        self.send(DocCommand::SetLayerOpacity(id, opacity), cx);
    }

    /// Set the selected layer's blend mode — the panel's drop-down (`crate::controls`).
    pub(crate) fn set_blend(
        &mut self,
        mode: stark_model::document::BlendMode,
        cx: &mut Context<'_, Self>,
    ) {
        let Some(id) = self.obs.as_ref().map(|o| o.active_layer) else {
            return;
        };
        self.send(DocCommand::SetLayerBlend(id, mode), cx);
    }

    // --- the Lighting shelf (§6.3, §6.4, §6.5) --------------------------------

    /// Move one of the Lighting shelf's tracks.
    ///
    /// The one place its three kinds of state are told apart (§4): the media
    /// parameters are a *view* setting, the substrate's scale is the **document's**,
    /// and the headroom is this client's own preference and reaches the engine only
    /// through the window's own capability (`Renderer::apply_hdr`).
    pub(crate) fn turn_light(&mut self, dial: light::Dial, v: f32, cx: &mut Context<'_, Self>) {
        match dial {
            light::Dial::Impasto | light::Dial::Texture | light::Dial::Gloss => {
                let mut media = self
                    .obs
                    .as_ref()
                    .map_or_else(stark_engine::MediaParams::default, |o| o.media);
                match dial {
                    light::Dial::Impasto => media.height_strength = v,
                    light::Dial::Texture => media.substrate_strength = v,
                    _ => media.specular = v,
                }
                self.send(ViewCommand::SetMediaParams(media), cx);
            }
            // Document state, and sent per sample: the engine coalesces nothing, so a
            // drag is a run of history entries. Honest but coarse, exactly as the
            // layer opacity above is, and the preview pair is the same stage of its
            // own for both.
            light::Dial::Scale => {
                let scale = stark_model::SubstrateScale::new(v.round().max(0.0) as u16);
                self.send(DocCommand::SetSubstrateScale(scale), cx);
            }
            // Shown but not kept: a headroom is written down when the hand comes off
            // it ([`settle_light`]), so one drag is one write rather than one a frame.
            light::Dial::Headroom => {
                self.hdr.headroom = v;
                self.apply_hdr(cx);
            }
        }
    }

    /// The end of a track's drag, for the one dial that has anything to do there.
    pub(crate) fn settle_light(&mut self, dial: light::Dial, _v: f32, _cx: &mut Context<'_, Self>) {
        if dial == light::Dial::Headroom {
            let mut prefs = stark_ui::storage::load::<Prefs>().unwrap_or_default();
            prefs.hdr = self.hdr;
            stark_ui::storage::save(&prefs);
        }
    }

    /// Tell the engine what the window is (§6.5) and show it — the read-modify half of
    /// the HDR switch, shared by the switch and by the headroom track.
    fn apply_hdr(&mut self, cx: &mut Context<'_, Self>) {
        // The display's own figure is a *window* question and this is not one; the
        // switch below re-asks it whenever the window can answer, and a track is only
        // ever mounted where it cannot (`lighting::dials`).
        let hdr = self.hdr;
        if let Some(r) = self.renderer.as_mut() {
            r.apply_hdr(hdr, None);
            self.obs = Some(r.observe());
        }
        self.repaint(cx);
    }

    /// Re-light the canvas (§6.3). A view setting: no stored pixel moves, only how the
    /// relief catches the light.
    ///
    /// The bytes go in first where this build has them and the engine has not — which
    /// is a decode and a prefilter, done once per light per session and never on the
    /// switch that follows. It is *not* a command (§4), which is why it is a call and
    /// the switch beside it is not.
    pub(crate) fn set_environment(
        &mut self,
        id: stark_engine::EnvironmentId,
        cx: &mut Context<'_, Self>,
    ) {
        let needed = self
            .renderer
            .as_ref()
            .is_some_and(|r| !r.environment_loaded(id));
        if needed
            && let Some(bytes) = lighting::environment_hdr(id)
            && let Some(r) = self.renderer.as_mut()
            && let Err(e) = r.register_environment(id, bytes.to_vec())
        {
            // The canvas keeps the light it has rather than switching to one that
            // will not decode — and says so, which is what this window has that the
            // web app's `tracing::warn` does not.
            return self.report(format!("that light will not load: {e}"));
        }
        self.send(ViewCommand::SetEnvironment(id), cx);
    }

    /// Do what a press on the Lighting shelf means.
    fn light_act(
        &mut self,
        region: lighting::Region,
        window: &mut Window,
        cx: &mut Context<'_, Self>,
    ) {
        match region {
            // The colour in hand, laid under the painting (§15.5) — see
            // `crate::lighting` for why the well takes rather than picks.
            lighting::Region::SubstrateColor => {
                let color = lighting::take_color(self.brush.tune.color);
                self.send(DocCommand::SetSubstrateColor(color), cx);
            }
            lighting::Region::Hdr => self.toggle_hdr(window, cx),
        }
    }

    // --- the Guides shelf (§20.5) ---------------------------------------------

    /// Move one of the Guides shelf's tracks: a whole edit of the camera in hand,
    /// because that is the shape `DocCommand::SetGuide` takes.
    pub(crate) fn turn_guide(
        &mut self,
        dial: stark_ui::guides::Dial,
        v: f32,
        cx: &mut Context<'_, Self>,
    ) {
        let Some((id, camera)) = self.held_guide() else {
            return;
        };
        self.send(DocCommand::SetGuide(id, dial.write(camera, v)), cx);
    }

    /// The guide this client has taken up, and its camera as the engine holds it now.
    ///
    /// Read back per edit rather than kept beside the choice, which is worth saying:
    /// the roster is the engine's projection, so this reports what the canvas is
    /// showing rather than what a copy here last recorded (§4).
    fn held_guide(&self) -> Option<(GuideId, stark_model::document::PerspectiveGuide)> {
        let o = self.obs.as_ref()?;
        let id = guides::chosen(Some(o), self.guide)?;
        o.guides.iter().find(|g| g.id == id).map(|g| (id, g.guide))
    }

    /// Do what a press on the Guides shelf means.
    ///
    /// The *meaning* is `guides::act`, a function over the roster so that it can be
    /// tested; what is here is the two things it cannot do — send the command, and
    /// take a guide up, which is this shelf's own state rather than the document's.
    fn guide_act(&mut self, region: guides::Region, cx: &mut Context<'_, Self>) {
        let guides: Vec<stark_engine::GuideInfo> = self
            .obs
            .as_ref()
            .map(|o| o.guides.to_vec())
            .unwrap_or_default();
        let taken = guides::chosen(self.obs.as_ref(), self.guide);
        // Where the artist is looking, which is where a new perspective is centred.
        let center = self.obs.as_ref().map_or(Vec2::ZERO, |o| o.view.center);
        let adding = region == guides::Region::Add;
        match guides::act(region, &guides, taken, center) {
            Some(guides::Act::Doc(command)) => {
                self.send(command, cx);
                // The engine mints no id for a guide — its identity is the id of the
                // action that added it (§20.5) — so a new one is *found* rather than
                // returned. It was appended, so it is the tail, and `send` has already
                // refreshed the projection.
                if adding && let Some(o) = self.obs.as_ref() {
                    self.guide = o.guides.last().map(|g| g.id);
                    // Drawn straight away: adding a guide is asking to see it, and an
                    // eye that had to be opened afterwards would make the act look
                    // like it had done nothing.
                    if let Some(id) = self.guide {
                        self.send(ViewCommand::SetGuideVisible(id, true), cx);
                    }
                }
            }
            Some(guides::Act::View(command)) => self.send(command, cx),
            Some(guides::Act::Take(id)) => {
                self.guide = Some(id);
                self.repaint(cx);
            }
            None => {}
        }
    }

    // --- the Navigator (§11) ---------------------------------------------------

    /// Point the view at what a press on the miniature landed on.
    fn overview_to(&mut self, at: Point<Pixels>, cx: &mut Context<'_, Self>) {
        let (Some(over), Some((fx, fy))) =
            (self.overview, navigator::fraction_at(&self.nav_regions, at))
        else {
            return;
        };
        self.send(ViewCommand::CenterOn(over.target(fx, fy)), cx);
    }

    /// Bring the miniature up to date, if it is on screen and owes a frame.
    ///
    /// The whole of the refresh policy, and every clause of it earns its place: the
    /// picture is of the **committed** document, so it is due when that revision
    /// moves; it is never drawn under a live gesture, because one refresh composites
    /// every tile in the document and mid-stroke is exactly where that is least
    /// affordable; and it is drawn at most once a settle, so a held undo collapses
    /// into one render rather than one a frame.
    ///
    /// A **resized** surface is the one thing that does not wait for the settle
    /// ([`Renderer::overview_resized`]): the picture is gone rather than stale, and
    /// the frame the element asked for on resizing is the one chance to draw it. Held
    /// to the same `quiet` all the same.
    fn refresh_overview(&mut self, window: &Window) {
        let Some(o) = self.obs.as_ref() else { return };
        // The **topmost** frame rather than the selected one: this is a permanent
        // readout of where you are in the piece, and "the piece" is what the frame on
        // top says it is (`stark_ui::bounds::piece_frame`).
        let frame = stark_ui::bounds::piece_frame(o);
        // Nothing painted and no frame: the rect the engine would fall back to is the
        // *viewport*, which for an overview would be a picture of the window
        // presented as the piece — and, since panning is not a change to the
        // document, one that then froze where it was rendered. An unbounded canvas
        // with nothing on it has no overview, and saying so is the honest answer.
        if frame.is_none() && o.bounds.tile_range().is_none() {
            self.overview = None;
            return;
        }
        let revision = o.doc_revision;
        let quiet = self.held.is_none() && !o.is_stroking;
        let scale = window.scale_factor();
        let now = self.elapsed();
        let plan = self
            .renderer
            .as_ref()
            .and_then(|r| r.overview_plan(frame, navigator::box_for(scale)));
        let Some(plan) = plan else {
            self.overview = None;
            return;
        };
        // The box the shelf lays out is the plan's whether or not this frame draws
        // into it — so the miniature keeps the piece's aspect from the first frame,
        // and the column does not change shape under a refresh.
        self.overview = Some(stark_ui::bounds::Overview::of(&plan, scale));
        // The first call sizes the surface from the plan and the element resizes it
        // from its own bounds a frame later, so the two disagree by a pixel of layout
        // rounding on the frame after the shelf appears — and every frame after that
        // agrees, which is why this settles rather than oscillating.
        let lost = self
            .renderer
            .as_ref()
            .is_some_and(Renderer::overview_resized);
        let moved = revision != self.overview_at || self.overview_when.is_infinite();
        let due = lost || (moved && now - self.overview_when >= stark_ui::bounds::SETTLE);
        if due
            && quiet
            && let Some(r) = self.renderer.as_mut()
            && r.paint_overview(window, &plan)
        {
            self.overview_at = revision;
            self.overview_when = now;
        }
    }

    /// The rows the layers panel draws, worked out by the tree.
    fn rows(&self) -> Vec<stark_ui::layer_tree::Row> {
        match &self.obs {
            Some(o) => stark_ui::layer_tree::rows(&o.layers, &self.collapsed),
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
        // Every document change goes through here, so this is where a shared session
        // hears about one — one seam for the projection and the wire alike (§12.4).
        self.broadcast();
        self.repaint(cx);
    }

    /// Whether the document holds committed work no file has (§8).
    fn unsaved(&self) -> bool {
        self.obs
            .as_ref()
            .is_some_and(|o| stark_ui::files::unsaved(o.edited, o.doc_revision, self.written))
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
        let ask = cx.prompt_for_new_path(&self.directory(), Some(&stark_ui::files::default_name()));
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
        // What the file names but does not carry. A lean file leaves out content it
        // expects the opener to have (§8's version 6), and this build **has** it: the
        // shipped images are in the binary, so settling an owed asset is a slice
        // rather than a fetch (`crate::assets`).
        let owed: Vec<_> = r
            .unresolved_content(&file)
            .into_iter()
            .filter_map(|need| Some((need, crate::assets::bytes_for(need.content())?)))
            .collect();
        for (need, png) in owed {
            let taken = match need {
                AssetNeed::Brush(_) => r.import_brush_id(png).map(|_| ()),
                AssetNeed::Substrate(id) => r.accept_substrate(SubstrateId::Image(id), png),
                // No build ships a picture: one is by definition something a person
                // brought in, so a catalog naming one is a catalog wrong about itself.
                AssetNeed::Picture(_) => Err("the shipped catalog holds no pictures".to_string()),
            };
            if let Err(e) = taken {
                self.report(format!("content this painting names would not load: {e}"));
                return false;
            }
        }
        // Anything still owed is content nobody here can produce. A file has no peer
        // to ask, so it is refused with the painting on screen untouched — which is
        // what makes a refused file cost nothing (§6.4).
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

    /// Say what went wrong, where a person will see it: a notification, raised on
    /// the next frame. A failure that reached only a log nobody is tailing is a
    /// failure nobody is told about.
    fn report(&mut self, why: String) {
        self.notice = Some(Notice::Failed(why));
    }

    /// Say something that went *right*, by the same means. The good news has to be
    /// said too: Share's whole product is a string on the clipboard, which leaves
    /// nothing at all on screen to say it worked (§12.4).
    fn say(&mut self, what: String) {
        self.notice = Some(Notice::Said(what));
    }

    /// Put the file's name on the window, if it changed.
    fn retitle(&mut self, window: &mut Window) {
        let title = files::window_title(
            self.path.as_deref(),
            self.unsaved(),
            self.collab.phase == collab::Phase::Shared,
        );
        if title != self.title {
            window.set_window_title(&title);
            self.title = title;
        }
    }

    /// Move the picker, and the brush with it.
    ///
    /// `grab` is what the press decided this gesture means and is applied *before*
    /// the position is read as a value — an ordinary drag is the pointer, a fine one
    /// is a fifth of its travel from where the marker stood
    /// (`stark_ui::color::Grab`).
    fn pick(
        &mut self,
        region: color::Region,
        grab: stark_ui::color::Grab,
        at: (f32, f32),
        cx: &mut Context<'_, Self>,
    ) {
        let (x, y) = grab.place(at);
        self.wheel = match region {
            color::Region::Wheel => self.wheel.at(x, y),
            color::Region::Track => color::Wheel {
                l: x.clamp(0.0, 1.0),
                ..self.wheel
            },
        };
        // The color is the hand's rather than the tool's (§18.1.8), so this moves the
        // transient half and leaves the preset's name on the brush.
        self.brush.tune.color = self.wheel.rgb(color::WHEEL_GAMUT);
        self.send_brush(cx);
    }

    /// Fold a panel away, or open it — and remember which.
    ///
    /// Written on the press rather than at shutdown, for `crate::window`'s reason
    /// inverted: a fold is one act a person performs deliberately, where a resize is
    /// a hundred frames of a drag. There is nothing here worth batching.
    fn fold(&mut self, what: VisibilityToggle, cx: &mut Context<'_, Self>) {
        if !self.folded.insert(what) {
            self.folded.remove(&what);
        }
        crate::visibility::persist(&self.hidden, &self.folded, self.rack.pinned);
        self.repaint(cx);
    }

    /// Whether a shelf is on screen at all.
    fn shown(&self, what: VisibilityToggle) -> bool {
        !self.hidden.contains(&what)
    }

    /// Whether its **body** is drawn — on screen, and not folded to its title bar.
    ///
    /// The one the view asks before building anything: a body that is not drawn is a
    /// body that is not built, which for the color wheel is `FIELD_N²` gamut lookups
    /// and for the navigator a composite of every tile in the document.
    fn drawn(&self, what: VisibilityToggle) -> bool {
        self.shown(what) && !self.folded.contains(&what)
    }

    /// The same, for a panel — which is most of them.
    fn panel_drawn(&self, id: PanelId) -> bool {
        self.drawn(VisibilityToggle::Panel(id))
    }

    /// Show a panel, or put it away — the Window menu's act (§25.5).
    ///
    /// Through the same writer as the fold above, because the two are one record: a
    /// panel that is not showing cannot also be folded, and two writers for one record
    /// is a record that can come to hold half of each (`stark_ui::visibility`).
    ///
    /// Nothing else has to happen here. A hidden panel is one the column does not
    /// build, and everything downstream of that — the room the canvas takes, where a
    /// press lands in the picture, where the stylus is captured — reads this same set
    /// ([`Canvas::origin`]) rather than being told.
    fn toggle_shelf(&mut self, what: VisibilityToggle, cx: &mut Context<'_, Self>) {
        // A chord could name something this frontend has not got, since the table is
        // the registry's and the registry knows nine (`stark_ui::commands`). Showing
        // one is not something this window can do, so it does nothing rather than
        // remembering a shelf it will never draw.
        if !crate::visibility::SHELVES.contains(&what) {
            return;
        }
        if !self.hidden.insert(what) {
            self.hidden.remove(&what);
        }
        crate::visibility::persist(&self.hidden, &self.folded, self.rack.pinned);
        self.repaint(cx);
    }

    // --- sharing (§12.4) -----------------------------------------------------

    /// Share this painting, and put the invitation on the clipboard.
    ///
    /// **The clipboard is the whole of the handing-over**, because this window has
    /// nothing else to hand a string with: there is no dialog to show a link in and no
    /// URL bar to keep one in (`crate::collab`). So the act ends by saying what it
    /// did, and what a person does next is paste.
    ///
    /// Pressed again while a session is live it mints a *fresh* link rather than doing
    /// nothing. A link names this peer and the members it could vouch were alive when
    /// it was made, so the one from an hour ago may name nobody who is still here —
    /// and a link that dials nothing is worse than no link at all.
    fn share(&mut self, window: &mut Window, cx: &mut Context<'_, Self>) {
        match self.collab.phase {
            // Binding and waiting for a relay takes a moment, and a second press
            // inside it would bind a second endpoint over the first.
            collab::Phase::Connecting => return,
            collab::Phase::Shared => {
                let Some(tx) = self.collab.broadcaster() else {
                    return;
                };
                self.collab.task = Some(cx.spawn_in(window, async move |this, cx| {
                    let done = collab::link(tx).await;
                    let _ = this
                        .update_in(cx, |this, window, cx| this.settle_session(done, window, cx));
                }));
                return;
            }
            collab::Phase::Solo => {}
        }
        let Some(r) = self.renderer.as_mut() else {
            return;
        };
        // The actor id derives from the endpoint's own key, and the shared log has to
        // carry it *before* the snapshot is served — so the engine is converted here
        // and the session is bound around what that produced. The identity is this
        // machine's persisted one, so sharing the same document twice is the same
        // author twice (`crate::identity`).
        let id = crate::identity::get();
        let actor = stark_net::actor_from_endpoint_id(id.secret.public());
        r.start_collaboration(stark_engine::Identity::new(actor, id.boot));
        let (doc, assets) = (r.document_file(), r.all_asset_bytes());
        self.obs = Some(r.observe());
        self.collab.phase = collab::Phase::Connecting;
        self.say("making a link\u{2026}".to_string());
        self.collab.task = Some(cx.spawn_in(window, async move |this, cx| {
            let done = collab::host(doc, assets).await;
            let _ = this.update_in(cx, |this, window, cx| this.settle_session(done, window, cx));
        }));
        self.repaint(cx);
    }

    /// Ask for the link to a session, and join it.
    ///
    /// A dialog with one field, as the web app has (§12.4), so the link can be pasted
    /// or typed. The clipboard is read first and offered as the field's value when
    /// what it holds is a link — the common case, and one keystroke fewer — but not
    /// acted on unasked: a paste is a thing a person does, not a thing a menu row
    /// reads over their shoulder. What *is* a link is `stark_ui::collab`'s to say —
    /// a whole URL and a bare ticket both are — and deciding that twice is how the
    /// two frontends would come to accept different things.
    fn join(&mut self, window: &mut Window, cx: &mut Context<'_, Self>) {
        match self.collab.phase {
            collab::Phase::Connecting => return,
            // Not silence: a row that looked available and did nothing is the failure
            // the menu's own rule is about (`crate::menu`).
            collab::Phase::Shared => {
                self.report("this canvas is already in a session".to_string());
                return self.repaint(cx);
            }
            collab::Phase::Solo => {}
        }
        let pasted = cx
            .read_from_clipboard()
            .and_then(|item| item.text())
            .filter(|text| stark_ui::collab::ticket_in(text).is_some());
        let field = cx.new(|cx| InputState::new(window, cx).placeholder("Paste a session link"));
        if let Some(link) = pasted {
            field.update(cx, |f, cx| f.set_value(link, window, cx));
        }
        let this = cx.weak_entity();
        window.open_dialog(cx, move |dialog, _, _| {
            let (this, field) = (this.clone(), field.clone());
            dialog
                .title("Join a session")
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap_2()
                        .child("The link the other painter shared. Anyone in the session can make one.")
                        .child(Input::new(&field)),
                )
                .footer(
                    DialogFooter::new()
                        .child(DialogClose::new().child(Button::new("cancel").label("Cancel")))
                        .child(DialogAction::new().child(Button::new("join").label("Join").primary())),
                )
                .on_ok(move |_, window, cx| {
                    let link = field.read(cx).value().to_string();
                    let _ = this.update(cx, |this, cx| this.join_link(link, window, cx));
                    true
                })
        });
    }

    /// Join the session a link names — the dialog's answer (`join`).
    fn join_link(&mut self, link: String, window: &mut Window, cx: &mut Context<'_, Self>) {
        self.take_focus(window, cx);
        if stark_ui::collab::ticket_in(&link).is_none() {
            self.report("that is not a session link".to_string());
            return self.repaint(cx);
        }
        self.collab.phase = collab::Phase::Connecting;
        self.say("joining\u{2026}".to_string());
        self.collab.task = Some(cx.spawn_in(window, async move |this, cx| {
            let done = collab::join(link).await;
            let _ = this.update_in(cx, |this, window, cx| this.settle_session(done, window, cx));
        }));
        self.repaint(cx);
    }

    /// Take a finished session act back into the view — [`Canvas::settle`]'s
    /// counterpart, and one place for the same reason: what a success does to the
    /// title and to the phase is written once, whichever act arrived at it.
    fn settle_session(
        &mut self,
        done: collab::Done,
        window: &mut Window,
        cx: &mut Context<'_, Self>,
    ) {
        match done {
            collab::Done::Started {
                session,
                events,
                link,
            } => {
                self.install_session(*session, events, window, cx);
                self.hand_over(link, window, cx);
            }
            collab::Done::Arrived {
                session,
                events,
                file,
                owed,
            } => {
                if !self.take_session_document(&file, &owed) {
                    // Refused, with the painting on screen untouched and the reason
                    // already reported. Dropping the session is what ends it.
                    self.collab.phase = collab::Phase::Solo;
                    self.retitle(window);
                    return self.repaint(cx);
                }
                // Everything this client had imported before it arrived, so a peer can
                // fetch whatever the joiner is about to paint with.
                let tx = session.broadcaster();
                if let Some(r) = self.renderer.as_ref() {
                    collab::seed(&tx, r.all_asset_bytes());
                }
                self.install_session(*session, events, window, cx);
                // No link is minted here, and none is needed: every member is a valid
                // entry point (§12.4), so the invitation a joiner passes on is the one
                // Share makes when it is asked.
                self.say("joined — Share hands the link on".to_string());
            }
            collab::Done::Link(link) => self.hand_over(link, window, cx),
            collab::Done::Failed(why) => {
                // A share converts the engine *before* it binds, so a bind that failed
                // leaves a document queueing broadcasts for a session that never
                // started. Put it back, which also hands its history back (§18.2.4).
                if self.collab.session.is_none()
                    && let Some(r) = self.renderer.as_mut()
                {
                    r.end_collaboration();
                    self.obs = Some(r.observe());
                }
                self.collab.phase = collab::Phase::Solo;
                self.report(why);
            }
        }
        self.retitle(window);
        self.repaint(cx);
    }

    /// Replace the document with a joined session's log, settling what the snapshot
    /// left out first; `false` if it was refused.
    ///
    /// The owed content goes in **before** the replay, which is the whole reason this
    /// is a step rather than a call: a substrate that is not registered when its
    /// `SetSubstrate` replays deposits every later stroke against the flat stand-in,
    /// and those pixels are stored (§6.4).
    fn take_session_document(
        &mut self,
        file: &stark_model::DocumentFile,
        owed: &[AssetNeed],
    ) -> bool {
        let id = crate::identity::get();
        let Some(r) = self.renderer.as_mut() else {
            return false;
        };
        if let Err(e) = collab::settle_owed(r, owed) {
            self.report(e);
            return false;
        }
        let actor = stark_net::actor_from_endpoint_id(id.secret.public());
        // Fallible, and refused here — before anything of this client's own document
        // has been disturbed. A session painted in a color space this build lacks is a
        // fact about *this build*, not about the link (§6.7).
        if let Err(e) = r.join_collaboration(file, stark_engine::Identity::new(actor, id.boot)) {
            self.report(format!("cannot join this session: {e}"));
            return false;
        }
        // Frame what arrived, which a file open does not need and this does: a view is
        // per-client and never sent (§18.1.2), so a joiner starts at the origin at 1:1
        // while the drawing they came to see can be anywhere on an unbounded canvas —
        // including entirely off their screen.
        r.process(ViewCommand::ShowPiece(None));
        self.obs = Some(r.observe());
        // The document is somebody else's wholesale, so everything the panel
        // remembered is stale — the folded groups above all, whose ids are gone.
        self.collapsed.clear();
        true
    }

    /// Hold the session and start the incoming pump.
    ///
    /// The pump is a **wgpui** task rather than a tokio one, which is the shape the
    /// whole module is arranged around: every event ends in the engine, the engine
    /// belongs to this view, and the channel the events arrive on needs no runtime to
    /// receive from (`crate::collab`).
    fn install_session(
        &mut self,
        session: stark_net::CollabSession,
        events: stark_net::Events,
        window: &mut Window,
        cx: &mut Context<'_, Self>,
    ) {
        self.collab.session = Some(session);
        self.collab.phase = collab::Phase::Shared;
        let mut events = collab::pump(events);
        // Replacing the pump drops the old one, and a dropped wgpui `Task` is a
        // cancelled one — which is what keeps a previous session's tail out of this
        // one.
        self.collab.pump = Some(cx.spawn_in(window, async move |this, cx| {
            while let Some(event) = events.recv().await {
                // The view is gone, so there is nothing left to feed.
                if this
                    .update_in(cx, |this, _, cx| this.take_remote(event, cx))
                    .is_err()
                {
                    return;
                }
            }
        }));
    }

    /// Put the invitation where a person can paste it, and show it.
    ///
    /// On the clipboard at once, since that is where it is going; and in a dialog,
    /// which is the surface the clipboard never was — the link can be read, copied
    /// again after something else has taken the clipboard, or shown to whoever is at
    /// the next desk. Its one button copies it again and closes.
    fn hand_over(&mut self, link: String, window: &mut Window, cx: &mut Context<'_, Self>) {
        cx.write_to_clipboard(wgpui::ClipboardItem::new_string(link.clone()));
        let field = cx.new(|cx| InputState::new(window, cx));
        field.update(cx, |f, cx| f.set_value(link.clone(), window, cx));
        let this = cx.weak_entity();
        window.open_dialog(cx, move |dialog, _, _| {
            let (this, field, link) = (this.clone(), field.clone(), link.clone());
            dialog
                .title("Share this canvas")
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap_2()
                        .child("Anyone who opens this link paints here with you. It is on the clipboard already.")
                        .child(Input::new(&field).readonly(true)),
                )
                .footer(DialogFooter::new().child(
                    DialogAction::new().child(Button::new("copy").label("Copy link").primary()),
                ))
                .on_ok(move |_, window, cx| {
                    cx.write_to_clipboard(wgpui::ClipboardItem::new_string(link.clone()));
                    let _ = this.update(cx, |this, cx| {
                        this.say("link copied".to_string());
                        this.take_focus(window, cx);
                        this.repaint(cx);
                    });
                    true
                })
        });
        self.say("link copied — anyone who opens it paints here with you".to_string());
        self.repaint(cx);
    }

    /// Give the keyboard back to the canvas — after a field or a dialog has had it.
    pub(crate) fn take_focus(&mut self, window: &mut Window, cx: &mut Context<'_, Self>) {
        self.focus.focus(window, cx);
    }

    /// Take a color typed into the notation field (`crate::controls`). Anything the
    /// field cannot read is left where it was; the next frame puts the brush's own
    /// notation back in it.
    pub(crate) fn take_hex(&mut self, text: &str, cx: &mut Context<'_, Self>) {
        let Some(rgb) = stark_ui::color::parse_color(text) else {
            return self.repaint(cx);
        };
        // Through the wheel, so the picker and the brush agree about what was typed
        // — and the hue survives a grey, which is why the wheel keeps one.
        self.wheel = color::Wheel::of(color::WHEEL_GAMUT, rgb, self.wheel.hue);
        self.brush.tune.color = self.wheel.rgb(color::WHEEL_GAMUT);
        self.send_brush(cx);
    }

    /// The command field gained or lost focus (`crate::palette`).
    pub(crate) fn set_searching(&mut self, on: bool, cx: &mut Context<'_, Self>) {
        self.searching = on;
        self.repaint(cx);
    }

    /// Enter in the command field: the first of the registry's answers that has
    /// something to act on, if any (`crate::palette`).
    pub(crate) fn pick_first(
        &mut self,
        query: &str,
        window: &mut Window,
        cx: &mut Context<'_, Self>,
    ) {
        // The first row the drop-down shows as **live**, which is both halves of what
        // dims one (`crate::palette`): a row this window cannot answer is skipped
        // here as well, or Enter would reach past what the eye was offered.
        let first = stark_ui::commands::search(query)
            .into_iter()
            .find(|c| c.enabled(self.obs.as_ref()) && answers(*c));
        if let Some(command) = first {
            self.pick_result(command, window, cx);
        }
    }

    /// Run what the command field settled on, and put the field away.
    pub(crate) fn pick_result(
        &mut self,
        command: Command,
        window: &mut Window,
        cx: &mut Context<'_, Self>,
    ) {
        self.controls
            .search
            .update(cx, |s, cx| s.set_value("", window, cx));
        self.searching = false;
        self.take_focus(window, cx);
        self.run(command, window, cx);
    }

    /// Feed one remote event into the engine and pay what it owes the screen.
    fn take_remote(&mut self, event: stark_net::RemoteEvent, cx: &mut Context<'_, Self>) {
        let now = self.elapsed();
        let Some(tx) = self.collab.broadcaster() else {
            return;
        };
        let Some(r) = self.renderer.as_mut() else {
            return;
        };
        let wake = collab::apply(r, &tx, event, now);
        // Re-read the projection only where the *document* moved: `observe` walks the
        // layer roster, and presence arrives at pointer rate from every peer at once.
        if wake.observe {
            self.obs = Some(r.observe());
        }
        // Requested, not painted inline: peer gesture frames arrive at ~30 Hz per
        // stroking peer, and the dirty latch is what folds all of it into one paint.
        if let Some(why) = wake.trouble {
            self.report(why);
        }
        if wake.repaint {
            self.repaint(cx);
        }
    }

    /// Put whatever the engine just committed on the wire (§12.4).
    ///
    /// **Inline on the dispatch path**, not spawned, and that is the one thing here
    /// worth getting right: broadcasting queues, and the session's own send task puts
    /// things on the wire in order. A task per dispatch let two commands in one frame
    /// race onto the sender, and every inversion cost a receiver a timeline resync.
    ///
    /// Free when solo — there is no sender, and the engine queues nothing.
    fn broadcast(&mut self) {
        let Some(tx) = self.collab.broadcaster() else {
            return;
        };
        let Some(r) = self.renderer.as_mut() else {
            return;
        };
        let trouble = collab::send(&tx, r.take_outbox());
        // Said where a person will see it rather than logged, which is this crate's
        // rule (`report`) and is right here for a reason of its own: work that stopped
        // reaching the session looks exactly like work that reached it, on both
        // canvases, until somebody compares them.
        if let Some(why) = trouble {
            self.report(why);
        }
    }

    /// Hand a just-imported asset to the session, so a peer can fetch what this client
    /// is about to reference (§12.4).
    ///
    /// The snapshot a peer was served carries what the *log* named. A shape or a
    /// substrate imported since is content only this client holds, and the action
    /// naming it parks on the far side until the bytes turn up — so they are offered
    /// where they enter the engine, which is the one place the id and the canonical
    /// bytes are both in hand.
    ///
    /// Nothing to do when solo, which is what lets the import paths call it
    /// unconditionally rather than each asking whether there is a session.
    fn offer(&mut self, need: AssetNeed) {
        let Some(tx) = self.collab.broadcaster() else {
            return;
        };
        let bytes = self.renderer.as_ref().and_then(|r| match need {
            AssetNeed::Substrate(id) => r.substrate_bytes(SubstrateId::Image(id)),
            _ => r.asset_bytes(need.content()),
        });
        if let Some(bytes) = bytes {
            collab::offer(&tx, need, bytes);
        }
    }

    /// Publish this client's presence, and expire peers who have gone quiet (§17.5).
    ///
    /// **On the frame loop rather than on a timer of its own**, which is the one thing
    /// this frontend has here that the web app does not: `render` already runs on the
    /// display's cadence (`window.request_animation_frame`), and the engine gates the
    /// work itself — `presence_due` is a `&self` comparison, so an idle shared session
    /// costs it per frame and takes no mutable borrow at all.
    fn tick_presence(&mut self, cx: &mut Context<'_, Self>) {
        if self.collab.phase != collab::Phase::Shared {
            return;
        }
        let now = self.elapsed();
        if !self.renderer.as_ref().is_some_and(|r| r.presence_due(now)) {
            return;
        }
        let Some(r) = self.renderer.as_mut() else {
            return;
        };
        let tick = r.take_presence(now);
        // The expiry may have taken a departed peer's live stroke off the canvas.
        // Nothing else would notice: the pump repaints for frames that *arrive*, and
        // this is precisely the case where they stopped.
        if tick.repaint {
            self.repaint(cx);
        }
        if let Some(frame) = tick.frame
            && let Some(tx) = self.collab.broadcaster()
        {
            collab::publish(tx, frame);
        }
    }

    /// Whether a command names something this window is currently *in* — `None` for
    /// an act, which is in no state at all.
    ///
    /// The web frontend's `commands::active`, asked of this window's own fields, and
    /// it is what a menu row's tick reads (`menu::bar`). One answer for the row and
    /// for the chord that reaches the same command, so a switch cannot move without
    /// the tick moving with it.
    fn active(&self, command: Command) -> Option<bool> {
        match command {
            Command::TogglePanel(id) => Some(self.shown(VisibilityToggle::Panel(id))),
            Command::ToggleNavigator => Some(self.shown(VisibilityToggle::Navigator)),
            Command::ToggleQuickBrushes => Some(self.rack.pinned),
            Command::ToggleHdr => Some(self.hdr.on),
            Command::SetPickScope(scope) => Some(self.sampler.scope == scope),
            _ => None,
        }
    }

    /// Where the canvas surface begins in the window, in logical px: what is left once
    /// the menu bar and whatever columns are up have taken theirs.
    ///
    /// **The one number every mapping from a pointer to the picture goes through**
    /// ([`screen_at`], [`canvas_at`]) and the one the stylus is captured against
    /// ([`Canvas::pen_claim`]). It is a *derived* layout rather than a measured one,
    /// which is the exception `panel::within` already was — the bar's height and each
    /// column's width are what the tree is told, so reading them back would be asking
    /// taffy to confirm an arithmetic this module did.
    fn origin(&self) -> Point<Pixels> {
        point(px(panel::width(Side::Left, &self.hidden)), px(menu::HEIGHT))
    }

    /// A middle-button press: the pan for a hand already on the mouse, whatever else
    /// is held down with it.
    fn press_middle(
        &mut self,
        ev: &MouseDownEvent,
        window: &mut Window,
        cx: &mut Context<'_, Self>,
    ) {
        let Some(mode) = nav::press(
            nav::Button::Middle,
            screen_at(ev.position, self.origin(), window.scale_factor()),
            self.space,
            false,
        ) else {
            return;
        };
        self.held = Some(Held::Navigate {
            mode,
            last: ev.position,
        });
        self.repaint(cx);
    }

    /// The wheel: a cursor-anchored zoom (§18.1.7).
    ///
    /// A canvas zooms where a document would scroll, which is the convention every
    /// raster editor shares — and the rate is the crate's, so a notch is worth the
    /// same in both apps.
    fn wheel(&mut self, ev: &ScrollWheelEvent, window: &mut Window, cx: &mut Context<'_, Self>) {
        // Nothing reaches the picture under a modal — the same claim the press ladder
        // makes, and owed here for a sharper reason: a wheel that zoomed the canvas
        // behind the brush editor would move the view a person cannot see moving.
        if self.editor.is_some() {
            return;
        }
        // A wheel over a column scrolls it; only the canvas zooms. Asked the way
        // every press is (`over_chrome`), so the two columns and the surface agree
        // about where each begins.
        if self.over_chrome(window, ev.position) {
            return;
        }
        let notches = match ev.delta {
            // A mouse reports whole notches; a trackpad reports pixels, which are
            // brought to the same scale rather than measured (`stark_ui::nav`).
            ScrollDelta::Lines(d) => d.y,
            ScrollDelta::Pixels(d) => f32::from(d.y) / nav::WHEEL_PIXELS_PER_NOTCH,
        };
        let anchor = screen_at(ev.position, self.origin(), window.scale_factor());
        if let Some(command) = nav::wheel(anchor, notches) {
            self.send(command, cx);
        }
    }

    /// Enter transform mode around whatever is selected (§16.6).
    ///
    /// Where the widget mounts and on which layer are `stark_ui::transform`'s
    /// answers, so the two frontends cannot come to disagree about what an unbounded
    /// selection means.
    fn begin_transform(&mut self, cx: &mut Context<'_, Self>) {
        let Some(o) = self.obs.as_ref() else { return };
        let Some(entry) = stark_ui::transform::entry(o) else {
            return;
        };
        let ui = stark_ui::transform::mount(
            entry.layer,
            Family::Free,
            entry.hull,
            Bands::at(o.view.zoom),
        );
        self.hold(ui, cx);
    }

    /// Replace what the mode is composing and **show** it.
    ///
    /// One door for every gesture, so the preview can never lag the state: a mutation
    /// that reached the mode without the preview would leave a picture on screen that
    /// "Done" would not reproduce.
    fn compose(&mut self, ui: TransformUi, cx: &mut Context<'_, Self>) {
        self.mode = Some(ui);
        self.send(
            ViewCommand::PreviewTransform(Some((ui.layer(), ui.map()))),
            cx,
        );
    }

    /// Replace what the mode is composing **without** showing it.
    ///
    /// For the two moves that compose nothing: entering, and switching to a family
    /// that carries the deformation across. Previewing an identity is not free — the
    /// preview resamples the selected paint (§16.6), so an entry that showed one would
    /// harden the selection's edge before the hand had done anything — and a carry's
    /// preview is by definition the one already on screen.
    fn hold(&mut self, ui: TransformUi, cx: &mut Context<'_, Self>) {
        self.mode = Some(ui);
        // Entering is the one edge that puts a widget over a canvas the hand may be
        // resting on, so it is where the mark comes down (§18.1.10) — every later
        // `compose` is behind a press that has already crossed this line.
        self.clear_hover_mark(cx);
        self.repaint(cx);
    }

    /// One of the transform bar's controls.
    fn bar_act(&mut self, ui: TransformUi, region: transform::Region, cx: &mut Context<'_, Self>) {
        match region {
            transform::Region::Family(i) => {
                if let Some((to, _)) = transform::FAMILIES.get(i) {
                    self.switch_family(ui, *to, cx);
                }
            }
            transform::Region::Flip(i) => {
                if let TransformUi::Affine { rect, ts } = ui {
                    let ts = if i == 0 {
                        ts.flipped_h()
                    } else {
                        ts.flipped_v()
                    };
                    self.compose(TransformUi::Affine { rect, ts }, cx);
                }
            }
            transform::Region::Act(i) => match transform::BAR_ACTS.get(i) {
                Some(Command::CancelMode) => self.cancel_mode(cx),
                Some(Command::FinishMode) => self.finish_mode(cx),
                _ => {}
            },
        }
    }

    /// Switch which family is composing — carrying the deformation when the new
    /// family holds it exactly, and committing it first when it cannot.
    fn switch_family(&mut self, ui: TransformUi, to: Family, cx: &mut Context<'_, Self>) {
        let bands = Bands::at(self.obs.as_ref().map_or(1.0, |o| o.view.zoom));
        match stark_ui::transform::switch(ui, to, bands) {
            Switch::Nothing => {}
            Switch::Carried(next) => self.compose(next, cx),
            Switch::Fresh(next) => self.hold(next, cx),
            Switch::Commit { map, then } => {
                // One honest undo step for what could not ride across, and the
                // commit clears the preview itself — so there is no frame showing
                // the document untransformed between the two.
                self.send(
                    DocCommand::Transform {
                        layer: ui.layer(),
                        map,
                    },
                    cx,
                );
                self.hold(then, cx);
            }
        }
    }

    /// Commit the gesture and leave the mode — the bar's Done, and Enter's.
    fn finish_mode(&mut self, cx: &mut Context<'_, Self>) {
        let Some(ui) = self.mode.take() else { return };
        if ui.is_identity() {
            // Nothing composed: drop the preview rather than spend an undo step on a
            // transform that would change no pixel.
            self.send(ViewCommand::PreviewTransform(None), cx);
        } else {
            self.send(
                DocCommand::Transform {
                    layer: ui.layer(),
                    map: ui.map(),
                },
                cx,
            );
        }
        self.held = None;
    }

    /// Leave the mode keeping nothing — the bar's Cancel, and Escape's.
    fn cancel_mode(&mut self, cx: &mut Context<'_, Self>) {
        self.leave_mode(cx);
        self.held = None;
    }

    /// Put down whatever is composing, dropping its preview and committing nothing —
    /// the web app's `modes::leave`, and the half of a cancel that is about the *mode*
    /// rather than about the hand.
    ///
    /// Split out for the callers that are not a cancel: the two gates that must not
    /// run under a composing mode ([`Canvas::run`]). Guarded, unlike
    /// [`cancel_mode`](Self::cancel_mode), which lets go of the pointer whatever it is
    /// holding: an undo pressed by the other hand mid-stroke must not orphan the
    /// gesture. The grab *is* dropped where there was a mode, because the only thing
    /// the pointer can hold while one composes is that mode's own grab
    /// ([`press`](Self::press)) — and it now grabs a mode that is gone.
    fn leave_mode(&mut self, cx: &mut Context<'_, Self>) {
        if self.mode.take().is_some() {
            self.send(ViewCommand::PreviewTransform(None), cx);
            self.held = None;
        }
    }

    /// Hand back a shape tool without naming one.
    ///
    /// The action row is what asks: picking what a shape *does* is a statement about
    /// a gesture that has not been made, and with the brush in hand there is nothing
    /// for it to be a statement about. Which of the three it hands back is not this
    /// frontend's to remember yet — the rectangle is what a marquee means when
    /// nothing says otherwise, and remembering the last one armed is a signal the web
    /// app keeps and this one has nowhere to.
    fn arm_shape(&mut self, cx: &mut Context<'_, Self>) {
        let tool = self.obs.as_ref().map_or(Tool::Brush, |o| o.tool);
        if !tool.is_selection() {
            self.send(ViewCommand::SetTool(Tool::SelectRect), cx);
        }
    }

    /// Move one of the Select section's dials.
    pub(crate) fn turn_dial(
        &mut self,
        dial: select::Dial,
        fraction: f32,
        cx: &mut Context<'_, Self>,
    ) {
        let v = dial.value_at(fraction);
        match dial {
            select::Dial::Feather => self.send(ViewCommand::SetSelectionFeather(v), cx),
            select::Dial::FillOpacity => self.send(ViewCommand::SetShapeOpacity(v), cx),
            // Previewed while the hand is on it and committed on release — the mask's
            // strength is the document's, so a drag that logged per sample would
            // spend a hundred undo steps crossing the track.
            select::Dial::MaskOpacity => {
                self.send(ViewCommand::PreviewSelectionOpacity(Some(v)), cx)
            }
        }
    }

    /// The end of a drag on a dial. The mask's strength was previewed for the length
    /// of it, and the release is what spends an action on it (§6.8); the other two
    /// were set as they moved, so there is nothing left to do for them.
    pub(crate) fn settle_dial(
        &mut self,
        dial: select::Dial,
        fraction: f32,
        cx: &mut Context<'_, Self>,
    ) {
        if dial == select::Dial::MaskOpacity {
            self.send(ViewCommand::PreviewSelectionOpacity(None), cx);
            self.send(DocCommand::SetSelectionOpacity(dial.value_at(fraction)), cx);
        }
    }

    /// The view the pointer is mapped through. `None` before there is a device —
    /// there is no identity to stand in for it, because a view carries the viewport
    /// and a made-up one would put every mapped point somewhere wrong.
    fn view(&self) -> Option<ViewTransform> {
        self.renderer.as_ref().map(Renderer::view)
    }

    /// What a press on one of the two galleries does.
    fn gallery_act(
        &mut self,
        region: gallery::Region,
        window: &mut Window,
        cx: &mut Context<'_, Self>,
    ) {
        match region {
            gallery::Region::Shape(i) => {
                let Some(row) = assets::SHIPPED_SHAPES.get(i) else {
                    return;
                };
                // The procedural one is a shape rather than an asset and needs no
                // image (§6.2) — the substrate arm below says the same of the smooth
                // canvas.
                //
                // **A stamp carries no hardness**, so leaving the round tip loses it
                // and coming back gives whatever the dial would have been reading
                // meanwhile (`Knob::Hardness::get`, which answers the renderer's own
                // fallback for a stamp). That is `BrushShape` being a sum rather than a
                // record — the hardness lives on the round arm — and the web app loses
                // it the same way.
                let Some(path) = row.path else {
                    let hardness = stark_ui::brush_editor::Knob::Hardness.get(&self.brush.config);
                    return self.wear_shape(BrushShape::Round { hardness }, cx);
                };
                // A shipped stamp's id is known without importing anything, and the
                // bytes went in at startup — so wearing one is a brush write and
                // nothing else (`crate::assets`).
                if let Some(id) = assets::shipped_id(path) {
                    self.wear_shape(BrushShape::Stamp(id), cx);
                }
            }
            gallery::Region::OwnShape(id) => {
                if let Some(id) = self.ensure_shape(id, cx) {
                    self.wear_shape(BrushShape::Stamp(id), cx);
                }
            }
            gallery::Region::Substrate(i) => {
                let Some(row) = assets::SHIPPED_SUBSTRATES.get(i) else {
                    return;
                };
                // The procedural one is its own id and needs no image (§6.4).
                let Some(path) = row.path else {
                    return self.wear_substrate(SubstrateId::Flat, cx);
                };
                let Some(png) = crate::assets::bundled(path) else {
                    return;
                };
                if let Some(id) = self.import_substrate(png, cx) {
                    self.wear_substrate(id, cx);
                }
            }
            gallery::Region::OwnSubstrate(id) => {
                let Some(png) = self
                    .substrates
                    .iter()
                    .find(|e| e.id == id)
                    .map(|e| e.png.clone())
                else {
                    return;
                };
                if let Some(id) = self.import_substrate(&png, cx) {
                    self.wear_substrate(id, cx);
                }
            }
            gallery::Region::Import(which) => self.import_asset(which, window, cx),
            gallery::Region::Remove(which, id) => self.forget_asset(which, id, cx),
        }
    }

    /// Put a shape on the brush, taking the preset's name off it: a stamp is what the
    /// tool *is* (§18.1.8).
    fn wear_shape(&mut self, shape: BrushShape, cx: &mut Context<'_, Self>) {
        self.brush.config.shape = shape;
        self.brush.tuned_off_preset();
        self.send_brush(cx);
    }

    /// Move the document onto a substrate — a logged action, so it undoes and
    /// replicates like any other edit. The painting is preserved; the paint already
    /// down re-reads against the new surface (§6.4).
    fn wear_substrate(&mut self, id: SubstrateId, cx: &mut Context<'_, Self>) {
        self.substrate = id;
        self.send(DocCommand::SetSubstrate(id), cx);
    }

    /// Get a substrate's height map into this document's engine, reporting a refusal
    /// where a person will see it.
    fn import_substrate(&mut self, png: &[u8], cx: &mut Context<'_, Self>) -> Option<SubstrateId> {
        match self.renderer.as_mut()?.import_substrate(png) {
            Ok(id) => {
                if let SubstrateId::Image(content) = id {
                    self.offer(AssetNeed::Substrate(content));
                }
                Some(id)
            }
            Err(e) => {
                self.report(format!("that substrate would not load: {e}"));
                self.repaint(cx);
                None
            }
        }
    }

    /// Make sure a library shape's bytes are in this document's engine, answering the
    /// id to reference it by — **healed** when the stored id predates a
    /// canonicalization change, which is the one case where what the library calls an
    /// asset and what the engine calls it can differ (§19).
    fn ensure_shape(&mut self, id: AssetId, cx: &mut Context<'_, Self>) -> Option<AssetId> {
        // Already here — imported in this session, or arrived with a loaded file.
        if self.renderer.as_ref()?.asset_bytes(id).is_some() {
            return Some(id);
        }
        let entry = self.shapes.iter().find(|e| e.id == id)?.clone();
        let actual = match self.renderer.as_ref()?.import_brush_id(&entry.png) {
            Ok(actual) => actual,
            Err(e) => {
                self.report(format!("“{}” would not load: {e}", entry.name));
                self.repaint(cx);
                return None;
            }
        };
        if actual != entry.id {
            if let Some(e) = self.shapes.iter_mut().find(|e| e.id == entry.id) {
                e.id = actual;
            }
            let rows = self.shapes.clone();
            pollster::block_on(assets::heal::<assets::Shapes>(
                &rows, entry.id, actual, &entry.png,
            ));
        }
        // The bytes have just entered the engine, so this is the moment a peer could
        // need them. The early return above needs no such offer: an asset the engine
        // already held was seeded when the session started.
        self.offer(AssetNeed::Brush(actual));
        Some(actual)
    }

    /// Open an image and put it in one of the two libraries.
    ///
    /// The dialog is held rather than detached for `crate::files`' reason: a wgpui
    /// `Task` cancels when it is dropped, and an import dropped mid-dialog is a file
    /// the user asked for and did not get.
    fn import_asset(
        &mut self,
        which: gallery::Which,
        window: &mut Window,
        cx: &mut Context<'_, Self>,
    ) {
        let ask = cx.prompt_for_paths(wgpui::PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: None,
        });
        self.file_task = Some(cx.spawn_in(window, async move |this, cx| {
            let picked = match ask.await {
                Ok(Ok(Some(paths))) => paths.into_iter().next(),
                Ok(Ok(None)) | Err(_) => None,
                Ok(Err(e)) => {
                    let _ = this.update_in(cx, |this, _, cx| {
                        this.report(format!("the import dialog failed: {e}"));
                        this.repaint(cx);
                    });
                    return;
                }
            };
            let Some(path) = picked else { return };
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            let read =
                std::fs::read(&path).map_err(|e| format!("could not read {}: {e}", path.display()));
            let _ = this.update_in(cx, |this, _, cx| match read {
                Ok(bytes) => this.take_asset(which, name, &bytes, cx),
                Err(e) => {
                    this.report(e);
                    this.repaint(cx);
                }
            });
        }));
    }

    /// Normalize an opened file into an entry, store it, and put it in hand.
    ///
    /// The decode is this frontend's and what the pixels *mean* is the crate's
    /// (`crate::assets`) — which is why the two branches below differ only in which
    /// of the pair they call and what they do with the result.
    fn take_asset(
        &mut self,
        which: gallery::Which,
        file_name: String,
        bytes: &[u8],
        cx: &mut Context<'_, Self>,
    ) {
        let name = stark_ui::library::display_name(
            &file_name,
            match which {
                gallery::Which::Shapes => "Imported shape",
                gallery::Which::Substrates => "Imported substrate",
            },
        );
        match which {
            gallery::Which::Shapes => {
                let (png, inverted) = match crate::assets::as_shape(bytes) {
                    Ok(v) => v,
                    Err(e) => return self.refuse(&file_name, e, cx),
                };
                let Some(r) = self.renderer.as_ref() else {
                    return;
                };
                let id = match r.import_brush_id(&png) {
                    Ok(id) => id,
                    Err(e) => return self.refuse(&file_name, e, cx),
                };
                // The engine's own canonical bytes, not the ones handed to it: the id
                // names *those*, and a library holding anything else would be a
                // library whose rows do not match its blobs.
                let canonical = r.asset_bytes(id).unwrap_or(png);
                self.keep::<assets::Shapes>(id, name, canonical);
                self.offer(AssetNeed::Brush(id));
                self.wear_shape(BrushShape::Stamp(id), cx);
                if inverted {
                    self.report(
                        "that image read as dark ink on light paper, so it was inverted — \
                         white now paints"
                            .to_string(),
                    );
                }
                self.repaint(cx);
            }
            gallery::Which::Substrates => {
                let png = match crate::assets::as_substrate(bytes) {
                    Ok(v) => v,
                    Err(e) => return self.refuse(&file_name, e, cx),
                };
                let Some(id) = self.import_substrate(&png, cx) else {
                    return;
                };
                let SubstrateId::Image(content) = id else {
                    return self.refuse(&file_name, "it holds no height field".to_string(), cx);
                };
                let canonical = self
                    .renderer
                    .as_ref()
                    .and_then(|r| r.substrate_bytes(id))
                    .unwrap_or(png);
                self.keep::<assets::Substrates>(content, name, canonical);
                self.wear_substrate(id, cx);
            }
        }
    }

    /// Put an entry in its library and on disk — **bytes before the row that names
    /// them**, which is `stark_ui::assets`' order and its reason.
    ///
    /// A repeat import is free and silent: content addressing means the id is already
    /// there, so the entry is not added twice.
    fn keep<K: assets::Kind>(&mut self, id: AssetId, name: String, png: Vec<u8>) {
        // Which list, taken from `K` rather than from a second argument beside it:
        // the type already says which store the bytes go to, and a parameter that
        // could disagree with it is a way for a shape to be filed as a substrate.
        let entries = if K::STORE == <assets::Shapes as assets::Kind>::STORE {
            &mut self.shapes
        } else {
            &mut self.substrates
        };
        if entries.iter().any(|e| e.id == id) {
            return;
        }
        pollster::block_on(assets::store_bytes::<K>(id, &png));
        entries.push(assets::Entry { name, png, id });
        assets::persist::<K>(entries);
    }

    /// Drop an entry from a library. **The row first, then the bytes** — the same rule
    /// read the other way: a crash between the two strands some bytes, which costs
    /// space, where the other order strands the row, which costs an asset that cannot
    /// be used.
    ///
    /// Paint already down is untouched: the engine's per-document store keeps every
    /// imported asset, and a save file bundles whatever its strokes reference.
    fn forget_asset(&mut self, which: gallery::Which, id: AssetId, cx: &mut Context<'_, Self>) {
        match which {
            gallery::Which::Shapes => {
                self.shapes.retain(|e| e.id != id);
                assets::persist::<assets::Shapes>(&self.shapes);
                pollster::block_on(assets::drop_bytes::<assets::Shapes>(id));
                if self.brush.config.shape == BrushShape::Stamp(id) {
                    self.wear_shape(BrushShape::default(), cx);
                }
            }
            gallery::Which::Substrates => {
                self.substrates.retain(|e| e.id != id);
                assets::persist::<assets::Substrates>(&self.substrates);
                pollster::block_on(assets::drop_bytes::<assets::Substrates>(id));
            }
        }
        self.repaint(cx);
    }

    /// Say why a file could not be taken, naming it.
    fn refuse(&mut self, file_name: &str, why: String, cx: &mut Context<'_, Self>) {
        self.report(format!("could not import “{file_name}”: {why}"));
        self.repaint(cx);
    }

    /// The shipped rows of one catalog, each paired with the id it resolves to.
    ///
    /// Every one of them resolves, always — the ids were hashed at build time and the
    /// bytes are in the binary — which is why this answers `Some` where the web app's
    /// equivalent has to allow for a fetch still in flight.
    fn shipped(
        rows: &'static [assets::Shipped],
    ) -> Vec<(&'static assets::Shipped, Option<AssetId>)> {
        rows.iter()
            .map(|row| (row, row.path.and_then(assets::shipped_id)))
            .collect()
    }

    /// Run whatever the shipped chord table says this keystroke asks for.
    ///
    /// **The whole of the keyboard, and it is nine lines**, because the table is
    /// shared: what Ctrl+Z means was settled once (§25) and this frontend only has to
    /// say what a keystroke *is* (`crate::keys`) and what an act *does* below.
    fn key(&mut self, ev: &KeyDownEvent, window: &mut Window, cx: &mut Context<'_, Self>) {
        // A field that has the keyboard has all of it: a letter typed into the
        // command search is not a chord, whatever the table says about that letter.
        if window.has_focused_input(cx) {
            return;
        }
        let stroke = crate::keys::stroke(&ev.keystroke);
        // Space is nobody's chord — the registry says so and claims it before the
        // table — because a frontend arms the pan off the key itself. Asked through
        // the shared reading so this app and that rule cannot come apart.
        if stark_ui::keys::is_space(&stroke) {
            self.space = true;
            // Space arms the pan, and a hover mark left standing would promise paint
            // the press will not make (§18.1.10). Self-guarding, so the key's
            // auto-repeat costs a peek and nothing else.
            self.clear_hover_mark(cx);
            return;
        }
        // The quick-brush rack, claimed before the chord table is consulted so a future
        // row on a digit could never shadow it. A digit is not a chord: it is a *hold*,
        // owning both edges of its key (§18.1.8); it is read off the physical row
        // (`crate::keys::code_of`) so a layout that types something else there still has
        // a rack; and **Shift is tolerated**, since on most layouts it is what the digit
        // row types under and a hand resting on it should not silently disarm the rack.
        // Alt is not: bare Alt is the eyedropper's, and only a bare digit is ours.
        //
        // `hold_slot` ignores a press while a hold is in flight, which is what makes the
        // key's own auto-repeat harmless, and it is what counts a digit pressed twice in
        // a beat (`slots::Taps`) — so nothing here keeps time.
        if !stroke.mods.ctrl
            && !stroke.mods.alt
            && let Some(slot) = slots::of_code(stroke.code)
        {
            self.hold_slot(slot, Grip::Key, cx);
            return;
        }
        let Some(command) = self.bindings.lookup(&stroke) else {
            return;
        };
        // Escape shuts an open menu before it cancels anything else. The chord table
        // says Escape means `CancelMode`, and it still does — what this adds is that
        // the nearest thing to cancel is the menu the hand just opened.
        if command == Command::CancelMode && self.menu_open.take().is_some() {
            return self.repaint(cx);
        }
        self.run(command, window, cx);
    }

    /// The modifiers moved under a hand that is holding nothing.
    ///
    /// The eyedropper's bar and its cursor are mounted on the chord being *held*
    /// (§18.0.2), and that is the whole discoverability of a modifier binding — so a
    /// modifier going down or coming up owes the window a frame. Unconditional,
    /// because a modifier is pressed at the rate a hand moves: the frame it costs is
    /// the frame the bar needs, and noticing a *change* instead would mean this view
    /// keeping a second copy of what `Window::modifiers` already holds.
    fn modifiers(
        &mut self,
        ev: &ModifiersChangedEvent,
        _window: &mut Window,
        cx: &mut Context<'_, Self>,
    ) {
        // The mark under the cursor is a promise of paint, and a chord that arms the
        // sampler has just taken it back (§18.1.10) — the same thing space does one
        // handler down, and self-guarding for its reason.
        if self.pick_hand().armed(&self.drags, mods_of(&ev.modifiers)) {
            self.clear_hover_mark(cx);
        }
        self.repaint(cx);
    }

    /// Space going up, and the one other thing that ends a hold: losing focus.
    ///
    /// A key-up is not a chord — the table answers presses — so this reaches nothing
    /// else. It exists because the modifier that makes a left drag a pan is a *key*,
    /// and a key that is never seen to rise stays down for good.
    fn key_up(&mut self, ev: &KeyUpEvent, _window: &mut Window, cx: &mut Context<'_, Self>) {
        let stroke = crate::keys::stroke(&ev.keystroke);
        if stark_ui::keys::is_space(&stroke) {
            self.space = false;
        }
        // The rack's release, named by the slot it lets go of — so a hand rolling from 3
        // to 4 and off 4 first does not end the hold 3 still has (§18.1.8). Unguarded by
        // the focused-field test the keydown makes, and for the reason the web frontend
        // leaves its own unguarded: focus can move between a press and its release, and
        // a release that never arrived would leave the brush swapped.
        if let Some(slot) = slots::of_code(stroke.code) {
            self.release_slot(slot, Grip::Key, cx);
        }
    }

    /// Do what a command means here, if this window can and may.
    ///
    /// A short list, and short *honestly*: the registry has thirty-odd acts and this
    /// frontend answers the handful below. Which those are is [`answers`]', so the
    /// list is a total function rather than a match that quietly ends in `_ => {}` —
    /// which is what let the palette offer a dead act undimmed.
    fn run(&mut self, command: Command, window: &mut Window, cx: &mut Context<'_, Self>) {
        // An act this window has no answer for is refused at the door rather than
        // falling off the end of the match, so the palette's dimming and this are one
        // question asked once (`answers`).
        if !answers(command) {
            return;
        }
        // **The act's own gate, off the registry** (§25.2) — not `Command::enabled`,
        // which is presentation and says so: a row greys because there is nothing to
        // undo, and the day it greys for a reason that is only about the screen, a
        // `run` that asked it would silently refuse the act.
        match command.gate() {
            // View, brush and chrome acts, which commit nothing.
            Gate::Free => {}
            Gate::Edit => {
                if !self.may_edit() {
                    return;
                }
            }
            // The composing half is not *refused* — the act replaces the mode rather
            // than being turned away by it (§20.5) — so the replacing is done here.
            // The web app gets it for free, since its guide edit is itself a mode and
            // `modes::enter` leaves the last one; `guide_act` composes nothing, so
            // without this the commit would land under a preview computed against the
            // document it moves. The playhead half has nothing to refuse on: no
            // timeline yet (§11.2).
            Gate::EditReplacingMode => self.leave_mode(cx),
            // These two *resolve* rather than refuse — nothing on screen says undo is
            // unavailable, so a silent refusal would read as a broken keyboard. There
            // is no playback to stop, so putting the composition down is the whole of
            // it, and it has to happen for the same reason as above.
            Gate::History => self.leave_mode(cx),
        }
        let doc = match command {
            Command::Undo => Some(DocCommand::Undo),
            Command::Redo => Some(DocCommand::Redo),
            // Covering everything *is* selecting nothing, so Ctrl+A and Ctrl+D are
            // one act (§6.8) — which is the registry's claim, and this is it honoured
            // rather than restated.
            Command::Deselect => Some(DocCommand::Select(SelectionOp::select_all())),
            Command::InvertSelection => Some(DocCommand::InvertSelection),
            Command::FloatSelection => self.obs.as_ref().map(|o| DocCommand::FloatSelection {
                layer: o.active_layer,
            }),
            // The color comes off the brush, which is the same choice a Fill *gesture*
            // makes: a fill lays the paint in hand. How far it covers is not a
            // question this act asks — it fills the selection, so the selection's own
            // coverage answers it.
            Command::FillSelection => self.obs.as_ref().map(|o| DocCommand::Fill {
                layer: o.active_layer,
                op: FillOp::of_selection(Srgb::new(self.brush.tune.color)),
            }),
            _ => None,
        };
        match command {
            Command::BrushSmaller => self.step_size(1.0 / SIZE_STEP, cx),
            Command::BrushLarger => self.step_size(SIZE_STEP, cx),
            Command::SaveDocument => self.save(window, cx),
            Command::OpenDocument => self.open(window, cx),
            Command::ExportImage => self.export(window, cx),
            Command::Share => self.share(window, cx),
            Command::Join => self.join(window, cx),
            Command::SelectRect => self.arm_tool(Tool::SelectRect, cx),
            Command::SelectEllipse => self.arm_tool(Tool::SelectEllipse, cx),
            Command::SelectLasso => self.arm_tool(Tool::SelectLasso, cx),
            Command::Transform => self.begin_transform(cx),
            // Both put the dialog away while one is up, and neither reaches the
            // transform mode then: a modal is what the keyboard is addressing, which
            // is the same claim the press ladder makes two screens up. There is no
            // "cancel" of a brush edit to distinguish them by — every track writes
            // straight through, so Escape and Done are one act (`open_editor`).
            Command::CancelMode if self.editor.is_some() => self.close_editor(cx),
            Command::FinishMode if self.editor.is_some() => self.close_editor(cx),
            Command::CancelMode => self.cancel_mode(cx),
            Command::FinishMode => self.finish_mode(cx),
            Command::ToggleHdr => self.toggle_hdr(window, cx),
            Command::TogglePanel(id) => self.toggle_shelf(VisibilityToggle::Panel(id), cx),
            Command::ToggleNavigator => self.toggle_shelf(VisibilityToggle::Navigator, cx),
            Command::ToggleQuickBrushes => self.pin_rack(!self.rack.pinned, cx),
            Command::AddPerspective => self.guide_act(guides::Region::Add, cx),
            // Setting, never cycling — so the chip the bar lights, the chord held
            // under the modifier that raised it and the palette row are one act.
            Command::SetPickScope(scope) => {
                self.sampler.scope = scope;
                self.repaint(cx);
            }
            Command::EditBrush => self.open_editor(window, cx),
            // The acts whose whole answer is the `doc` command above, written out so
            // that a command turned `true` in [`answers`] and given no arm falls off
            // the end of a list a reader can check rather than into a silent `_`.
            Command::Undo
            | Command::Redo
            | Command::Deselect
            | Command::InvertSelection
            | Command::FloatSelection
            | Command::FillSelection => {}
            // Turned away by [`answers`] before the match was reached.
            _ => {}
        }
        if let Some(doc) = doc {
            self.send(doc, cx);
        }
    }

    /// Whether a **document edit** may be accepted right now — this window's answer to
    /// [`Gate::Edit`], which is the registry's question (§25.2).
    ///
    /// One of the two halves so far. A transform's preview is computed against the
    /// committed document, so an edit laid under one would move the wrong region on
    /// Done — which is the bug §25.2 names, and which this window had until the
    /// classification came down. The other half is the playhead, and there is no
    /// timeline here yet (§11.2): it arrives as one more `&&`.
    fn may_edit(&self) -> bool {
        self.mode.is_none()
    }

    // --- the brush editor (§6.2, `crate::brush_editor`) -----------------------

    /// Raise the dialog, building its test canvas on the main engine's own device.
    ///
    /// Nothing is stashed and nothing is restored on close: every track writes straight
    /// through to the brush in hand, so the dialog *is* the brush being edited and Done
    /// has nothing to commit. That is the web app's bargain too, and it is what makes
    /// the preview honest — what is on the test canvas is what the next stroke lays.
    fn open_editor(&mut self, window: &mut Window, cx: &mut Context<'_, Self>) {
        if self.editor.is_some() {
            return;
        }
        // A nominal size: the element resizes the surface from its own laid-out bounds
        // one frame later, and the stroke is re-laid then (`refresh_editor`). Guessing
        // the layout here would be the arithmetic §11.2 N2 deleted.
        let scale = window.scale_factor();
        let px = |v: f32| ((v * scale).round() as u32).max(1);
        let preview = self.renderer.as_ref().and_then(|r| {
            Preview::new(
                r,
                window,
                px(brush_editor::PREVIEW_WIDTH),
                px(brush_editor::SHEET_HEIGHT),
            )
        });
        let mut editor = Editor::new(preview);
        editor.lay_reference();
        editor.restroke(&self.brush);
        self.editor = Some(editor);
        // The modal is over the whole window, so anything the canvas was promising is
        // no longer on offer — including the mark, which a chord can leave standing
        // under the scrim with no move to notice it (§18.1.10).
        self.clear_hover_mark(cx);
        self.repaint(cx);
    }

    /// Put it away, and give the test canvas's surface and document back with it.
    fn close_editor(&mut self, cx: &mut Context<'_, Self>) {
        self.editor = None;
        self.repaint(cx);
    }

    /// What the editor's rows are built from: the brush, and the two document facts
    /// they depend on — the color space its channels are in (§6.7) and whether the
    /// canvas has a tooth to catch on (§6.4).
    fn editor_shown(&self) -> stark_ui::brush_editor::Shown {
        stark_ui::brush_editor::Shown {
            brush: self.brush.config,
            tune: self.brush.tune,
            space: self
                .obs
                .as_ref()
                .map_or(stark_model::ColorSpaceId::Oklab, |o| o.color_space),
            substrate: self.substrate,
        }
    }

    /// A press anywhere while the dialog is up.
    fn press_editor(&mut self, at: Point<Pixels>, window: &mut Window, cx: &mut Context<'_, Self>) {
        // The stamp gallery is inside the dialog while one is open — the shelf that
        // usually holds it is behind the scrim — so its own list is asked first, its
        // cards being nested inside the sheet's rectangle either way.
        if let Some(region) = gallery::hit(&self.gallery_regions, at) {
            self.gallery_act(region, window, cx);
            self.restroke(cx);
            return;
        }
        let Some(region) = brush_editor::hit(&self.editor_regions, at) else {
            // The scrim: outside the sheet is out of the dialog, which is the answer
            // every overlay in this toolkit gives (`wgpui_component::dialog`).
            return self.close_editor(cx);
        };
        match region {
            brush_editor::Region::Done => self.close_editor(cx),
            // A press on the panel that hit no control. Not paint, and not a close
            // either — this is the whole of what makes the dialog modal.
            brush_editor::Region::Sheet => {}
            brush_editor::Region::Preview => {
                let Some(pos) =
                    brush_editor::preview_at(&self.editor_regions, at, window.scale_factor())
                else {
                    return;
                };
                let rope = self
                    .editor
                    .as_ref()
                    .and_then(|e| e.preview.as_ref())
                    .map_or(0.0, |p| {
                        stark_ui::input::rope(p.view(), self.brush.config.smoothing)
                    });
                if let Some(editor) = self.editor.as_mut() {
                    editor.start_stroke(pos, rope);
                }
                self.held = Some(Held::PreviewStroke);
                self.repaint(cx);
            }
            brush_editor::Region::Reset => {
                if let Some(editor) = self.editor.as_mut() {
                    editor.reset_stroke();
                }
                self.restroke(cx);
            }
            brush_editor::Region::Fold(section) => {
                if let Some(editor) = self.editor.as_mut() {
                    editor.fold(section);
                }
                self.repaint(cx);
            }
            brush_editor::Region::More(section) => {
                if let Some(editor) = self.editor.as_mut() {
                    editor.toggle_more(section);
                }
                self.repaint(cx);
            }
            brush_editor::Region::Mapping(row) => {
                if let Some(editor) = self.editor.as_mut() {
                    editor.toggle_mapping(row);
                }
                self.repaint(cx);
            }
            // One field moves and nothing is forgotten: the configuration carries every
            // effect (`BrushConfig`), so switching to Erase and back costs a tuned
            // smudge none of its axes.
            brush_editor::Region::Effect(effect) => {
                self.brush.config.effect = effect;
                self.edited(cx);
            }
            brush_editor::Region::Orientation(orientation) => {
                self.brush.config.orientation = orientation;
                self.edited(cx);
            }
            brush_editor::Region::Noise(noise) => {
                self.brush.config.color_dynamics.noise = noise;
                self.edited(cx);
            }
            brush_editor::Region::Source(row, source) => {
                stark_ui::brush_editor::set_source(&mut self.brush.config, row, source);
                self.edited(cx);
            }
        }
    }

    /// Move one of the editor's modulatable rows, `fraction` along whatever range it
    /// has *this* frame (`crate::controls`).
    pub(crate) fn turn_mod_row(
        &mut self,
        row: stark_ui::brush_editor::ModRow,
        fraction: f32,
        cx: &mut Context<'_, Self>,
    ) {
        let (lo, hi) = row.range(&self.brush.config, self.brush.tune);
        let value = lo + fraction.clamp(0.0, 1.0) * (hi - lo);
        let mut tune = self.brush.tune;
        row.set(&mut self.brush.config, &mut tune, value);
        self.brush.tune = tune;
        // Only the durable rows take the preset's name off: the size and the flow are
        // the hand's, and working a brush at another size is the same tool (§18.1.8).
        if row.durable() {
            self.brush.tuned_off_preset();
        }
        self.send_brush(cx);
        self.restroke(cx);
    }

    /// Move one of its plain ones.
    pub(crate) fn turn_editor_knob(
        &mut self,
        knob: stark_ui::brush_editor::Knob,
        fraction: f32,
        cx: &mut Context<'_, Self>,
    ) {
        let (lo, hi) = knob.range();
        knob.set(
            &mut self.brush.config,
            lo + fraction.clamp(0.0, 1.0) * (hi - lo),
        );
        self.edited(cx);
    }

    /// Move one of the open mapping's two shape knobs.
    ///
    /// Which mapping that is, is resolved here rather than captured by the track: a
    /// subscription is `'static` and the row it writes into is whichever one is open on
    /// the frame the drag lands.
    pub(crate) fn turn_mapping(
        &mut self,
        shape: brush_editor::Shape,
        v: f32,
        cx: &mut Context<'_, Self>,
    ) {
        let Some(row) = self.editor.as_ref().and_then(Editor::open_mapping) else {
            return;
        };
        let Some(mapping) = row.slot(&mut self.brush.config) else {
            return;
        };
        shape.set(mapping, v);
        self.edited(cx);
    }

    /// A durable edit: the tool is no longer *the* preset, the engine is told, and the
    /// test canvas is re-stroked.
    fn edited(&mut self, cx: &mut Context<'_, Self>) {
        self.brush.tuned_off_preset();
        self.send_brush(cx);
        self.restroke(cx);
    }

    /// Re-render the test stroke with the brush as it now stands, if a dialog is up.
    ///
    /// Unthrottled, unlike the web app's: a track emits at most one change per frame
    /// and the replay is one commit, so the frame loop already is the throttle
    /// (`crate::brush_editor`).
    fn restroke(&mut self, cx: &mut Context<'_, Self>) {
        let brush = &self.brush;
        if let Some(editor) = self.editor.as_mut() {
            editor.restroke(brush);
        }
        self.repaint(cx);
    }

    /// Draw the test canvas, and re-lay the seeded stroke if the element has resized
    /// the surface out from under it.
    ///
    /// The resize happens in prepaint, so the frame that caused it drew the old picture
    /// stretched — and a resized surface is two *new* textures rather than a stretch, so
    /// what was on it is gone. Nothing else would ask for it back, which is the same
    /// clause the navigator's miniature owes (`Renderer::overview_resized`).
    fn refresh_editor(&mut self) {
        let brush = &self.brush;
        let Some(editor) = self.editor.as_mut() else {
            return;
        };
        let Some(p) = editor.preview.as_mut() else {
            return;
        };
        if p.paint() {
            editor.relay_after_resize();
            editor.restroke(brush);
            if let Some(p) = editor.preview.as_mut() {
                p.paint();
            }
        }
    }

    /// Flip the HDR switch (§6.5): show it, and keep it. Ungated — a view act.
    /// Read-modify-write, so the preferences this frontend does not apply survive.
    fn toggle_hdr(&mut self, window: &Window, cx: &mut Context<'_, Self>) {
        self.hdr.on = !self.hdr.on;
        let display = window.display_headroom();
        if let Some(r) = self.renderer.as_mut() {
            r.apply_hdr(self.hdr, display);
            self.obs = Some(r.observe());
        }
        let mut prefs = stark_ui::storage::load::<Prefs>().unwrap_or_default();
        prefs.hdr = self.hdr;
        stark_ui::storage::save(&prefs);
        self.repaint(cx);
    }

    /// Arm a shape tool, or put it down if it is the one already in hand.
    fn arm_tool(&mut self, tool: Tool, cx: &mut Context<'_, Self>) {
        let current = self.obs.as_ref().map_or(Tool::Brush, |o| o.tool);
        self.send(
            ViewCommand::SetTool(stark_ui::selection::arm(current, tool)),
            cx,
        );
    }

    /// Step the brush's size by a factor, clamped to the range the panel offers.
    fn step_size(&mut self, factor: f32, cx: &mut Context<'_, Self>) {
        self.brush.tune.size = (self.brush.tune.size * factor).clamp(MIN_RADIUS, MAX_RADIUS);
        self.send_brush(cx);
    }

    /// Move a knob and put the changed brush in the engine's hand.
    pub(crate) fn turn(&mut self, knob: Knob, fraction: f32, cx: &mut Context<'_, Self>) {
        panel::drag_knob(&mut self.brush, knob, fraction);
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

    // --- the quick-brush rack (§18.1.8, `crate::slots`) -----------------------
    //
    // The rule is `stark_ui::slots`', shared with the web frontend. What is here is
    // what only this window can do: reach the live brush, keep the four values, and ask
    // for a frame.

    /// Put `config` on at `tune`, keeping the colour in hand, and tell the engine — the
    /// one door every swap comes through, in both directions (`Brush::put_on`).
    fn wear(
        &mut self,
        config: stark_ui::brush_config::BrushConfig,
        tune: stark_ui::brush_config::Transient,
        from: Option<String>,
        cx: &mut Context<'_, Self>,
    ) {
        self.brush.put_on(config, tune, from);
        self.send_brush(cx);
    }

    /// Begin holding `slot`.
    ///
    /// Ignored when a hold is already in flight, which is what makes it safe to call on
    /// every keydown: a held key repeats at the system's rate and each repeat is another
    /// keydown. The one exception is `Grip::displaces`' — an act over a posture.
    ///
    /// A slot with nothing in it still enters the hold rather than declining: the hold
    /// *is* the arming, and holding an empty number while clicking a preset is how the
    /// number gets its first brush.
    fn hold_slot(&mut self, slot: slots::Digit, grip: Grip, cx: &mut Context<'_, Self>) {
        if let Some(held) = self.rack.held.as_ref() {
            match held.displaced_by(grip) {
                Some((slot, grip)) => self.release_slot(slot, grip, cx),
                None => return,
            }
        }
        // Counted below the guard above, so a held key's repeats are never presses; and
        // for keys alone, since a tail is on the glass or off it and two dabs of it are
        // two erase strokes.
        let picked = grip == Grip::Key && self.rack.taps.press(slot, self.elapsed());
        let mut hold = slots::Held::open(
            slot,
            grip,
            self.brush.worn(),
            self.brush.from.clone(),
            picked,
        );
        // The slot's brush as it is *now* — its preset looked up live, at the slot's own
        // size and flow. A binding the library cannot answer is an empty slot, and an
        // empty slot is held without a swap.
        let bound = self.rack.brushes[slot.as_index()].clone();
        if let Some(bound) = bound
            && let Some((config, tune)) = slots::resolve(&self.brush.library, &bound)
        {
            self.wear(config, tune, Some(bound.preset), cx);
            // Read back rather than assumed: what the app now holds is what the release
            // has to compare against.
            hold.enter(self.brush.tune);
        }
        self.rack.held = Some(hold);
        self.repaint(cx);
    }

    /// End the hold on `slot`, if `grip` is what is holding it: keep whatever was
    /// changed, and put the displaced brush back (`slots::Held::settle`).
    fn release_slot(&mut self, slot: slots::Digit, grip: Grip, cx: &mut Context<'_, Self>) {
        let Some(held) = self.rack.held.take_if(|held| held.ends_on(slot, grip)) else {
            return;
        };
        let (kept, back) = held.settle(self.brush.tune, self.brush.from.as_deref());
        if let Some(bound) = kept {
            self.assign_slot(held.slot(), bound);
        }
        // Back through the door it left by, with the name it had: the hold borrowed the
        // hand, and a preset chosen *during* it went to the slot, not to this. Or not
        // back at all, for a double-tap's hold, whose whole point is that the swap
        // stands.
        match back {
            Some((config, tune)) => self.wear(config, tune, held.base_from(), cx),
            None => self.repaint(cx),
        }
    }

    /// End whatever hold is in flight, whoever made it — for the one event that can take
    /// a key away without ever sending its keyup: the window losing focus.
    fn release_slots(&mut self, cx: &mut Context<'_, Self>) {
        if let Some((slot, grip)) = self.rack.held.as_ref().map(|h| (h.slot(), h.grip())) {
            self.release_slot(slot, grip, cx);
        }
    }

    /// Say that a whole tool was just put on deliberately, so a hold in flight keeps
    /// what is live when it ends whether or not that moved anything.
    ///
    /// Raised by the two acts that mean *the artist chose a tool from a library* — a
    /// preset row clicked and a rack row clicked — and by nothing else. A knob turned
    /// needs no such word: it changed a value, and `settle`'s comparison sees that.
    /// Never from inside [`wear`](Self::wear), which the hold uses itself in both
    /// directions and which would therefore make every hold claim itself on the way in.
    fn claim_slot(&mut self) {
        if let Some(held) = self.rack.held.as_mut() {
            held.claim();
        }
    }

    /// Make `slot`'s brush the live one for good — what clicking a row of the pinned
    /// rack does, and the only way to a slot for a hand with no keyboard under it.
    ///
    /// Tapping the number twice arrives at the same place by another route: the second
    /// press enters a hold whose release keeps the slot's brush rather than putting the
    /// displaced one back (`slots::Held`), so this is not called for it and there is no
    /// second path to one outcome.
    fn pick_slot(&mut self, slot: slots::Digit, cx: &mut Context<'_, Self>) {
        let Some(bound) = self.rack.brushes[slot.as_index()].clone() else {
            return;
        };
        // A binding the library cannot answer is an empty row, and an empty row's click
        // puts on nothing.
        let Some((config, tune)) = slots::resolve(&self.brush.library, &bound) else {
            return;
        };
        self.claim_slot();
        self.wear(config, tune, Some(bound.preset), cx);
    }

    /// Bind `slot` and write the rack down.
    fn assign_slot(&mut self, slot: slots::Digit, bound: slots::QuickBrush) {
        slots::assign(&mut self.rack.brushes, slot, bound);
        slots::persist(&self.rack.brushes);
    }

    /// Empty `slot` and write the rack down — the trash on a pinned row, held until its
    /// fill closes (`crate::slots`).
    ///
    /// The live brush is untouched, exactly as removing a preset would leave it: what
    /// goes is the *binding*, not the tool.
    fn clear_slot(&mut self, slot: slots::Digit, cx: &mut Context<'_, Self>) {
        if slots::clear(&mut self.rack.brushes, slot) {
            slots::persist(&self.rack.brushes);
            self.repaint(cx);
        }
    }

    /// Pin the rack up or put it away — the Window menu's act, and **the only thing that
    /// writes `Rack::pinned`**, which is what makes durability structural rather than a
    /// line the menu row has to remember.
    ///
    /// Pinning is not the same question as the rack being *up*: while a number is held
    /// it shows regardless, and what the pin buys is a rack that stays and takes clicks.
    fn pin_rack(&mut self, pinned: bool, cx: &mut Context<'_, Self>) {
        if self.rack.pinned == pinned {
            return;
        }
        self.rack.pinned = pinned;
        crate::visibility::persist(&self.hidden, &self.folded, self.rack.pinned);
        self.repaint(cx);
    }

    /// Seconds since the window opened, for `InputSample::time` — which the stroke
    /// dynamics read as velocity and the timelapse (§8) replays against.
    fn elapsed(&self) -> f64 {
        self.clock.delta(self.epoch, self.clock.raw()).as_secs_f64()
    }

    /// Where a stylus press opens a stroke, in the physical px a tablet measures in.
    ///
    /// The canvas is what is left once both columns and the menu bar have taken
    /// theirs — named from the same constants the hit tests use rather than measured a
    /// second way, because the half of this that could be wrong is the half deciding
    /// whether a button can still be pressed (`stark_pen::Claim`).
    ///
    /// **Off entirely with a menu open or a transform live.** Both put something over
    /// the canvas that a press means instead, both are already answered by the mouse
    /// path, and a claim would be what took the press away from it. Neither wants
    /// anything a stylus adds: a transform handle does not care what a nib weighs.
    fn pen_claim(&self, window: &Window) -> Claim {
        let scale = window.scale_factor();
        let size = window.viewport_size();
        let origin = self.origin();
        Claim {
            rect: stark_pen::Rect {
                left: f32::from(origin.x) * scale,
                top: f32::from(origin.y) * scale,
                right: (f32::from(size.width) - panel::width(Side::Right, &self.hidden)) * scale,
                bottom: f32::from(size.height) * scale,
            },
            enabled: self.mode.is_none() && self.menu_open.is_none(),
        }
    }

    /// Spend everything the stylus reported since the last frame.
    ///
    /// A loop rather than a read, and that is the point: the reports are the whole of
    /// what the digitizer made between two frames at the spacing the hand made them,
    /// so a fast pen puts several samples through here per frame. Reading one would
    /// cap every stroke at the display's rate whatever the hardware resolved — the
    /// same trap `getCoalescedEvents` keeps the web frontend out of (§11.2).
    ///
    /// The buffer is a field so the drain costs no allocation: taken out for the
    /// length of the loop, because the arms below need `self` mutably, and put back
    /// empty.
    fn pump_pen(&mut self, window: &mut Window, cx: &mut Context<'_, Self>) {
        if !self.tablet.attached() {
            return;
        }
        let mut reports = std::mem::take(&mut self.pen);
        self.tablet.drain(&mut reports);
        let scale = window.scale_factor();
        // The keyboard as it stands now rather than as it stood at the report: a
        // pointer message carries no modifier this frontend binds anything to, and the
        // window is the one place that knows. A chord released mid-stroke therefore
        // reads as held for that stroke, which is what the mouse path does too — the
        // press is where a gesture is decided (§25.3).
        let mods = mods_of(&window.modifiers());
        for report in reports.drain(..) {
            // A tablet measures in the device's px and the chrome is laid out in
            // logical ones. The scale factor is the whole of the difference and this
            // is the one place it is spent, so everything below is in the units the
            // mouse path already speaks (§11.1).
            let at = point(
                px(report.pose.position[0] / scale),
                px(report.pose.position[1] / scale),
            );
            match report.phase {
                Phase::Down => self.open_canvas(at, mods, Some(&report.pose), window, cx),
                Phase::Move => self.move_to(at, Some(&report.pose), window, cx),
                Phase::Up => self.release_at(cx),
            }
        }
        self.pen = reports;
    }

    /// Note that the frame has changed. `notify` schedules it; `dirty` is what that
    /// frame reads to decide whether the *engine* has to render at all.
    pub(crate) fn repaint(&mut self, cx: &mut Context<'_, Self>) {
        self.dirty = true;
        cx.notify();
    }
}

impl Render for Canvas {
    fn render(&mut self, window: &mut Window, cx: &mut Context<'_, Self>) -> impl IntoElement {
        // A shared session is the one thing here that needs the frame loop to keep
        // turning on its own: presence rides it rather than a timer (`tick_presence`),
        // so an idle client that stopped asking would be expired by its peers.
        //
        // **Asked for only then.** wgpui honours the request now — a queued animation
        // frame chains the next one — where it used to sit unread until unrelated
        // input woke the loop, so an unconditional ask is an event loop that never
        // sleeps. Nothing else here wants the cadence: the stylus wakes the window
        // itself (`stark_pen`), a peer's frames arrive on a task, a resize is a frame
        // the surface element asks for, and everything a command touches goes through
        // `repaint`.
        if self.collab.phase == collab::Phase::Shared {
            window.request_animation_frame();
        }
        // The rack's one clock: the trash held down empties its slot when the fill it
        // is drawn from closes, and both are read off this same elapsed value
        // (`crate::slots::CLEAR_HOLD`) so what the disc shows and what happens cannot
        // come apart. A frame is asked for while it runs, since nothing else moves.
        let now = self.elapsed();
        if let Some(slot) = self.rack.armed_out(now) {
            self.clear_slot(slot, cx);
        }
        if self.rack.clearing() {
            window.request_animation_frame();
        }
        // The stylus first, before anything reads the document: what it reported since
        // the last frame is a press, a run of samples and a lift, and the engine has to
        // have them before `paint` below renders the answer — which is what keeps a
        // stroke on the frame the hand made it rather than the one after.
        self.pump_pen(window, cx);
        // Then say where a press *is* paint, off the same constants the hit tests
        // measure against (`pen_claim`).
        self.tablet.claim(self.pen_claim(window));
        // Then this client's presence, on the same frame the hand made (§17.5). Free
        // when solo, and nearly free when shared and idle — see `tick_presence`.
        self.tick_presence(cx);
        // The dirty mark follows the document rather than the last file act, so the
        // title is refreshed with the frame — cheaply, since `retitle` only calls the
        // platform when the words changed.
        self.retitle(window);
        // What the last act had to say, handed to the widget layer's notifications
        // *after* this frame: they hang off the `Root` that is drawing this view,
        // and a layer cannot be pushed onto it mid-draw.
        if let Some(notice) = self.notice.take() {
            cx.defer_in(window, move |_, window, cx| {
                window.push_notification(notice, cx);
            });
        }

        // **Every region list is cleared by the frame, not by the control that fills
        // it.** A panel the Window menu has put away is not built at all, so a builder
        // that cleared its own list would leave the last frame's rectangles standing —
        // controls a press still finds and the artist can no longer see. The transform
        // bar was cleared here for that reason from the start; a panel that can be
        // hidden made it every list's rule.
        //
        // Nothing is lost by clearing here rather than there: prepaint refills them
        // after this returns, so what a press between two frames reads is the layout
        // that is actually on screen.
        self.menu_regions.borrow_mut().clear();
        self.regions.borrow_mut().clear();
        self.color_regions.borrow_mut().clear();
        self.select_regions.borrow_mut().clear();
        self.select_bar_regions.borrow_mut().clear();
        self.gallery_regions.borrow_mut().clear();
        self.layer_regions.borrow_mut().clear();
        self.bar_regions.borrow_mut().clear();
        self.lighting_regions.borrow_mut().clear();
        self.guide_regions.borrow_mut().clear();
        self.nav_regions.borrow_mut().clear();
        self.pick_regions.borrow_mut().clear();
        self.editor_regions.borrow_mut().clear();
        self.slot_regions.borrow_mut().clear();
        // The miniature, before anything is built: it is a *second render* rather
        // than an element, and what it produces — the box it fills — is what the
        // shelf below is laid out to. Put away, its surface goes with it, which is
        // the whole of what hiding the navigator gives back (`Renderer::drop_overview`).
        if self.drawn(VisibilityToggle::Navigator) {
            self.refresh_overview(window);
        } else if let Some(r) = self.renderer.as_mut() {
            r.drop_overview();
            self.overview = None;
            // Back to "never drawn", so showing the navigator again renders rather
            // than deciding the document has not moved since the surface it was
            // holding was let go.
            self.overview_when = f64::NEG_INFINITY;
        }
        // The test canvas, before anything is built and for the miniature's reason: it
        // is a *second render* rather than an element, and the frame after the dialog
        // opens is the one that learns how large the element actually laid its surface
        // out (`refresh_editor`).
        self.refresh_editor();
        let shape_rows = Self::shipped(assets::SHIPPED_SHAPES);
        let substrate_rows = Self::shipped(assets::SHIPPED_SUBSTRATES);
        let held_shape = match self.brush.config.shape {
            BrushShape::Stamp(id) => Some(id),
            BrushShape::Round { .. } => None,
        };
        let held_substrate = match self.substrate {
            SubstrateId::Image(id) => Some(id),
            SubstrateId::Flat => None,
        };
        // A card's bytes come from the engine first and the shipped table second, on
        // `ensure`'s order: a stamp imported in an earlier session is only in the
        // library until it is picked, and a shipped one is in the binary either way.
        let engine_bytes = |id| {
            self.renderer
                .as_ref()
                .and_then(|r| r.asset_bytes(id))
                .or_else(|| crate::assets::bytes_for(id).map(<[u8]>::to_vec))
        };
        // The stamp gallery is the **editor's**, so it is built only while one is open:
        // what a shape *is* is the tool, and the shelf beside the canvas is where a
        // brush is worked rather than made (`crate::brush_editor`). Worth the test
        // here rather than inside, because a card's bytes are asked of the engine per
        // row per frame.
        let editor_open = self.editor.is_some();
        let shapes = editor_open.then(|| {
            gallery::gallery::<assets::Shapes>(
                gallery::Which::Shapes,
                "Shapes",
                gallery::Shown {
                    rows: &shape_rows,
                    own: &self.shapes,
                    current: held_shape,
                },
                engine_bytes,
                &self.gallery_regions,
            )
        });
        let substrate_bytes = |id| {
            self.renderer
                .as_ref()
                .and_then(|r| r.substrate_bytes(SubstrateId::Image(id)))
                .or_else(|| crate::assets::bytes_for(id).map(<[u8]>::to_vec))
        };
        // Headed by what it *is* rather than by what it holds: the shelf around it is
        // the light, and this is the surface being lit (`stark_ui::icons::SURFACE`).
        let substrates = self.panel_drawn(PanelId::Lighting).then(|| {
            gallery::gallery::<assets::Substrates>(
                gallery::Which::Substrates,
                "Surface",
                gallery::Shown {
                    rows: &substrate_rows,
                    own: &self.substrates,
                    current: held_substrate,
                },
                substrate_bytes,
                &self.gallery_regions,
            )
        });
        // What the *window* can say about the display (§6.5), read before anything
        // borrows the renderer: the Lighting shelf shows the HDR switch only where
        // there is more than white to show, and stands a headroom track in only where
        // the platform will not state its own.
        let hdr_capable = self.renderer.as_ref().is_some_and(Renderer::hdr_capable);
        let display_headroom = window.display_headroom();
        // Built before the panels so its regions are recorded first — which does not
        // matter for the hit test (the lists are separate) but keeps the bar's own
        // drop-down measured on the frame it opens.
        let query = self.controls.search.read(cx).value();
        let results: Vec<Command> = if self.searching && !query.trim().is_empty() {
            stark_ui::commands::search(&query)
                .into_iter()
                .take(palette::SHOWN)
                .collect()
        } else {
            Vec::new()
        };
        let this = cx.weak_entity();
        let run: palette::Run = std::rc::Rc::new(move |command, window, cx| {
            let _ = this.update(cx, |this, cx| this.pick_result(command, window, cx));
        });
        let search = palette::field(palette::Search {
            field: &self.controls.search,
            open: self.searching,
            results: &results,
            obs: self.obs.as_ref(),
            bindings: &self.bindings,
            run,
        })
        .into_any_element();
        let menubar = menu::bar(
            self.menu_open,
            self.obs.as_ref(),
            &self.bindings,
            &|command| self.active(command),
            &self.menu_regions,
            search,
        );
        // The widget layer's dials are told what the model says before the shelves
        // that show them are built (`crate::controls`).
        let rows = self.rows();
        let active = self.obs.as_ref().map(|o| o.active_layer);
        let chosen = active.and_then(|id| rows.iter().find(|r| r.info.id == id));
        // What the editor's two dozen tracks are settled from — `None` with no dialog
        // up, which is what keeps them free the rest of the time (`crate::controls`).
        let shown = editor_open.then(|| self.editor_shown());
        let mapping = self
            .editor
            .as_ref()
            .and_then(Editor::open_mapping)
            .and_then(|row| row.of(&self.brush.config));
        self.controls.sync(
            crate::controls::Sync {
                brush: &self.brush,
                obs: self.obs.as_ref(),
                opacity: chosen.map_or(1.0, |r| r.info.opacity),
                blend: chosen.map_or(stark_model::document::BlendMode::Normal, |r| r.info.blend),
                hdr: self.hdr,
                guide: self.held_guide().map(|(_, camera)| camera),
                editor: shown.as_ref(),
                mapping,
            },
            window,
            cx,
        );

        // **A body is built only where it is drawn.** A shelf the Window menu has put
        // away, or one folded to its title bar, gets a `None` here and costs nothing —
        // which for the color wheel is `FIELD_N²` gamut lookups a frame and for the
        // layers panel a walk of the roster.
        //
        // The two galleries are already `Option`s — built above, where the bytes for
        // their cards are asked for — and are `take`n below because an element is not
        // `Clone` and the loop cannot prove to the compiler that each is taken once.
        let (mut shapes, mut substrates) = (shapes, substrates);
        let mut left = Vec::new();
        let mut right = Vec::new();
        for what in crate::visibility::SHELVES {
            if !self.shown(what) {
                continue;
            }
            let body: Option<AnyElement> = match what {
                VisibilityToggle::Panel(PanelId::Brush) => {
                    self.panel_drawn(PanelId::Brush).then(|| {
                        panel::brush_body(&self.brush, &self.controls, &self.regions)
                            .into_any_element()
                    })
                }
                VisibilityToggle::Panel(PanelId::Select) => {
                    self.panel_drawn(PanelId::Select).then(|| {
                        select::select_body(
                            self.obs.as_ref(),
                            &self.bindings,
                            &self.controls,
                            &self.select_regions,
                        )
                        .into_any_element()
                    })
                }
                VisibilityToggle::Panel(PanelId::Lighting) => {
                    self.panel_drawn(PanelId::Lighting).then(|| {
                        lighting::lighting_body(
                            lighting::Shown {
                                obs: self.obs.as_ref(),
                                hdr: self.hdr,
                                hdr_capable,
                                display_headroom,
                                substrates: substrates.take(),
                            },
                            &self.controls,
                            &self.lighting_regions,
                        )
                        .into_any_element()
                    })
                }
                VisibilityToggle::Panel(PanelId::Guides) => {
                    self.panel_drawn(PanelId::Guides).then(|| {
                        guides::guides_body(
                            self.obs.as_ref(),
                            self.guide,
                            &self.bindings,
                            &self.controls,
                            &self.guide_regions,
                        )
                        .into_any_element()
                    })
                }
                VisibilityToggle::Navigator => self.drawn(VisibilityToggle::Navigator).then(|| {
                    navigator::navigator_body(
                        self.overview,
                        self.renderer.as_ref().and_then(Renderer::overview_surface),
                        self.obs.as_ref().map(|o| o.view),
                        &self.nav_regions,
                    )
                    .into_any_element()
                }),
                VisibilityToggle::Panel(PanelId::Color) => {
                    self.panel_drawn(PanelId::Color).then(|| {
                        color::color_panel(
                            self.wheel,
                            &mut self.pictures,
                            &self.controls.hex,
                            &self.color_regions,
                        )
                        .into_any_element()
                    })
                }
                VisibilityToggle::Panel(PanelId::Layers) => {
                    self.panel_drawn(PanelId::Layers).then(|| {
                        layers::layers_body(
                            self.obs.as_ref(),
                            &rows,
                            &self.bindings,
                            &self.controls,
                            &self.layer_regions,
                        )
                        .into_any_element()
                    })
                }
                // Every other entry of the vocabulary is one this frontend does not
                // draw, and `visibility::SHELVES` is what says so — this arm is
                // unreachable through that list, and is a `continue` rather than a
                // panic because a stored record is not a place to assert from.
                _ => continue,
            };
            let shelf = panel::Shelf { what, body };
            if crate::visibility::LEFT.contains(&what) {
                left.push(shelf);
            } else {
                right.push(shelf);
            }
        }
        // Built only where there is a column to put them in: with every shelf on one
        // side hidden the tree has no child there at all, and the surface flexes into
        // the room — which is the same fact `Canvas::origin` states.
        let column = (!left.is_empty())
            .then(|| panel::column(Side::Left, &self.hidden, &self.regions, left));
        let roster = (!right.is_empty())
            .then(|| panel::column(Side::Right, &self.hidden, &self.regions, right));
        // The mode's two pieces are built here, where `self` is still borrowable —
        // the surface below takes a mutable borrow of the renderer that outlives the
        // rest of the tree.
        let mode = self.mode;
        let (bar, overlay) = match mode {
            Some(ui) => (
                Some(transform::bar(ui, &self.bindings, &self.bar_regions)),
                self.view()
                    .map(|view| transform::overlay(ui, view, window.scale_factor(), self.hint)),
            ),
            None => (None, None),
        };
        // The eyedropper's own bar, and the cursor that goes with it (§18.0.2). Both
        // stand on the chord being *held* rather than on anything the document says,
        // so both come and go with the modifier — which is the whole discoverability
        // of a binding nobody could otherwise find.
        let hand = self.pick_hand();
        let held = mods_of(&window.modifiers());
        let armed = mode.is_none() && hand.armed(&self.drags, held);
        let pick_bar = (mode.is_none() && hand.shows_options(&self.drags, held))
            .then(|| pick::bar(self.sampler, &self.bindings, &self.pick_regions));
        let pick_cursor = armed.then(pick::cursor);
        // The selection's own bar takes the same edge, and stands down rather than
        // stacking under the mode's: a transform owns the canvas, and an act on the
        // whole selection reaching under one would move the wrong region on Done.
        // (The web app recedes its bar instead — dimmed and inert, so the place Done
        // returns to stays visible — which is a design this frontend has not got.)
        //
        // It yields the edge to the sampler's bar too, and that one is the other way
        // round from the mode: not "a press here means something else" but "a press
        // here has not happened yet" — a bar about the next press outranks one about
        // paint that is already there, and it is gone again the instant the modifier
        // is.
        let select_bar =
            (mode.is_none() && pick_bar.is_none() && select::bar_mounted(self.obs.as_ref()))
                .then(|| select::selection_bar(&self.bindings, &self.select_bar_regions));

        // The quick-brush rack (§18.1.8), built only while there is one to draw — held
        // by a key, or pinned up by the Window menu. Its rows are resolved against the
        // library here, a slot being a name until something looks it up
        // (`stark_ui::slots::rows`).
        let rack = self.rack.up().then(|| {
            let rows = stark_ui::slots::rows(stark_ui::slots::View {
                rack: &self.rack.brushes,
                library: &self.brush.library,
                // A **key** hold alone: the pen's tail holds a slot for the length of
                // every erase stroke, and a rack flying in and out of the corner of the
                // eye on each one is noise answering a question nobody asked.
                holding: self.rack.held.as_ref().filter(|h| h.by_key()),
                live: self.brush.worn(),
                in_hand: self.brush.from.as_deref(),
            });
            crate::slots::rack(&self.rack, &rows, now, &self.slot_regions)
        });

        // Built while `self` is still borrowable, like the mode's two pieces above: the
        // surface below takes a mutable borrow of the renderer that outlives the rest
        // of the tree.
        let editor = self.editor.as_ref().zip(shown.as_ref()).map(|(e, shown)| {
            brush_editor::modal(
                brush_editor::Dressing {
                    shown,
                    editor: e,
                    controls: &self.controls,
                    shapes: shapes.take(),
                    from: self.brush.from.as_deref(),
                },
                &self.editor_regions,
            )
        });

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
            .flex_col()
            // The keyboard answers whatever has focus, and nothing has it unless
            // something asks: an unfocused window dispatches no chord at all.
            .track_focus(&self.focus)
            .on_key_down(cx.listener(Self::key))
            .on_key_up(cx.listener(Self::key_up))
            .on_modifiers_changed(cx.listener(Self::modifiers))
            .child(menubar)
            .child(
                div()
                    .flex_1()
                    .min_h_0()
                    .flex()
                    .children(column)
                    .child(
                        div()
                            .relative()
                            .flex_1()
                            .h_full()
                            .child(Self::leaving(cx))
                            .child(wgpu_surface(r.surface()).size_full())
                            // Under everything else on the canvas, so a bar's own
                            // chips keep the pointer they ask for where the two
                            // overlap.
                            .children(pick_cursor)
                            // Over the surface rather than beside it: the widget is drawn in
                            // canvas space and the surface is what canvas space maps onto, so
                            // the overlay's own bounds are the frame the mapping lands in.
                            .children(overlay)
                            .children(rack)
                            .children(bar)
                            .children(select_bar)
                            .children(pick_bar),
                    )
                    .children(roster),
            )
            // Over both columns and the surface between them, because it is over the
            // *window*: the dialog is about the brush rather than about the picture, and
            // a modal that left a column pressable would be one whose Done is not the
            // only way out (`crate::brush_editor`).
            .children(editor)
            // The widget layer's own layers — sheets, dialogs, notifications — over
            // everything this view draws. `Root` holds them but leaves their place
            // in the tree to the view it wraps, which is the one thing that knows
            // where "over everything" is. Each occludes what it covers, so a press
            // on a dialog is not also a press on the canvas under it.
            .children(Root::render_sheet_layer(window, cx))
            .children(Root::render_dialog_layer(window, cx))
            .children(Root::render_notification_layer(window, cx))
            .on_mouse_down(MouseButton::Left, cx.listener(Self::press))
            .on_mouse_down(MouseButton::Middle, cx.listener(Self::press_middle))
            .on_mouse_up(MouseButton::Middle, cx.listener(Self::release))
            .on_mouse_up_out(MouseButton::Middle, cx.listener(Self::release))
            .on_scroll_wheel(cx.listener(Self::wheel))
            .on_mouse_move(cx.listener(Self::drag))
            .on_mouse_up(MouseButton::Left, cx.listener(Self::release))
            // A release the view never saw still ends what it was holding: the
            // pointer can leave the window mid-drag, and the alternative is a gesture
            // the engine never closes and a knob that keeps following the mouse.
            .on_mouse_up_out(MouseButton::Left, cx.listener(Self::release))
            .into_any_element()
    }
}

/// The held modifiers as the shared tables spell a chord.
///
/// One reading, because four callers want it — the press, the stylus pump, the hover
/// and the frame — and a second is a second chance to spell `platform` wrong.
fn mods_of(m: &wgpui::Modifiers) -> Mods {
    Mods {
        ctrl: m.control || m.platform,
        shift: m.shift,
        alt: m.alt,
    }
}

/// A pointer position mapped to a canvas-space sample.
///
/// `position` is window-relative and **logical**, while the view is denominated in
/// the surface's device px — so the scale factor is the whole of the conversion.
///
/// The chrome's own rectangle comes off first ([`Canvas::origin`]): the surface begins
/// where the bar and the column end, and `screen_to_canvas` maps out of the
/// *surface's* space rather than the window's.
fn sample_at(
    view: ViewTransform,
    position: Point<Pixels>,
    origin: Point<Pixels>,
    scale: f32,
    time: f64,
    pen: Option<&Pose>,
) -> InputSample {
    InputSample {
        pos: canvas_at(view, position, origin, scale),
        // A mouse is always pressed home (`ModSource::Pressure`) and reports no tilt
        // at all. A stylus answers both, in the units the engine measures in —
        // `stark-pen` normalizes, for the reason it gives: the ranges are the
        // platform's, and a frontend dividing by them would be a second copy of a fact
        // that crate already holds.
        pressure: pen.map_or(1.0, |p| p.pressure),
        tilt: pen.map_or(Vec2::ZERO, |p| Vec2::new(p.tilt[0], p.tilt[1])),
        // **The stylus's own clock where a stylus made the report**, rather than a
        // conversion onto this window's. The fitter re-bases every channel time to the
        // first sample of the gesture it is fitting (`path::fit`), so what has to hold
        // is that one gesture is timed by one device — and it is, because a gesture the
        // tablet opened is one the tablet also closes.
        time: pen.map_or(time, |p| p.time),
    }
}

/// What the device that made a gesture resolves position to, in this surface's device
/// px — the half of `stark_ui::input::tolerance` only a frontend can answer.
fn resolution(pen: Option<&Pose>) -> f32 {
    if pen.is_some() {
        chrome_input::PEN_RESOLUTION
    } else {
        chrome_input::MOUSE_RESOLUTION
    }
}

/// A window position in the **screen** px the view is denominated in — logical px
/// times the display's scale factor, measured from where the surface begins.
///
/// `origin` is [`Canvas::origin`], and it is both coordinates rather than the panel's
/// width alone: the surface sits under the menu bar as well as beside the column, and
/// a pointer read from the window's corner was every stroke landing the bar's height
/// above the nib.
///
/// Not `canvas_at`: a pan is a screen-space delta and a zoom anchors on a screen-space
/// point, so both stay in the frame the surface renders in rather than the one the
/// paint lives in (`ViewCommand::Pan`, `ViewCommand::Zoom`).
fn screen_at(position: Point<Pixels>, origin: Point<Pixels>, scale: f32) -> Vec2 {
    let x = f32::from(position.x) - f32::from(origin.x);
    let y = f32::from(position.y) - f32::from(origin.y);
    Vec2::new(x * scale, y * scale)
}

/// A window position as a plain vector of **logical** px.
///
/// Neither [`screen_at`] nor [`canvas_at`]: the brush-tuning drag is denominated in
/// what the hand travelled across the glass, not in what the view is showing
/// (`stark_ui::tune`), so it is measured where the window reports rather than through
/// the origin or the zoom. The web frontend's page coordinates are the same quantity.
fn logical_at(position: Point<Pixels>) -> Vec2 {
    Vec2::new(f32::from(position.x), f32::from(position.y))
}

/// A window position in canvas px.
///
/// Split out of [`sample_at`] because the transform widget wants the point without
/// the pen fields around it — and because one mapping is the whole of what keeps the
/// widget under the pointer that grabbed it. Through [`screen_at`] rather than beside
/// it, so the two cannot come to disagree about where the surface starts.
fn canvas_at(
    view: ViewTransform,
    position: Point<Pixels>,
    origin: Point<Pixels>,
    scale: f32,
) -> Vec2 {
    view.screen_to_canvas(screen_at(position, origin, scale))
}

/// Which acts this frontend answers ([`Canvas::run`]).
///
/// **Total** — a new command does not compile until somebody says whether this window
/// can do it, which is what the palette needs and a hand-kept list could not give it:
/// the palette reaches all of `commands::ALL`, so it was offering acts that ran
/// nothing, undimmed. The menus are short *because* of this list rather than beside
/// it (`crate::menu`).
///
/// An act with nothing to act on is `false` rather than given a no-op arm, so the day
/// its surface lands (§11.2) the compiler has nothing to say and the reader does.
pub fn answers(command: Command) -> bool {
    match command {
        // The document acts this window sends to the engine, and the history pair.
        Command::Undo
        | Command::Redo
        | Command::Deselect
        | Command::InvertSelection
        | Command::FloatSelection
        | Command::FillSelection
        | Command::Transform
        // The three shape tools, and the two steps on the brush.
        | Command::SelectRect
        | Command::SelectEllipse
        | Command::SelectLasso
        | Command::BrushSmaller
        | Command::BrushLarger
        // The document in and out of this window (§12.4, `crate::menu`).
        | Command::OpenDocument
        | Command::SaveDocument
        | Command::ExportImage
        | Command::Share
        | Command::Join
        // The chrome: what is on screen, what a sample sees, and the two ladders.
        | Command::ToggleHdr
        | Command::TogglePanel(_)
        | Command::ToggleNavigator
        | Command::ToggleQuickBrushes
        | Command::SetPickScope(_)
        | Command::EditBrush
        | Command::AddPerspective
        | Command::CancelMode
        | Command::FinishMode => true,
        // A surface this window has not got: no new-document dialog, no clipboard or
        // file import (§23), no gradient bar, no layer or frame stack to add to, no
        // preset-name dialog, no settings page, no timing readout, no credits — and no
        // timeline at all, which is why `ToggleTimeline` is not a Window-menu row
        // either (`crate::menu`). `MirrorView` is the one that is merely not written
        // yet.
        Command::NewDocument
        | Command::ImportImage
        | Command::MirrorView
        | Command::GradientFill
        | Command::AddLayer
        | Command::AddFrame
        | Command::SavePreset
        | Command::Settings
        | Command::ToggleTimeline
        | Command::TimingStats
        | Command::Credits => false,
    }
}

/// What the window shows when there is no wgpu device to paint with.
///
/// Names its own two colours rather than `crate::style`'s: there is no chrome on this
/// screen to be consistent with, and a restyle of the panels has nothing to say about
/// the page that stands in for all of them.
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
