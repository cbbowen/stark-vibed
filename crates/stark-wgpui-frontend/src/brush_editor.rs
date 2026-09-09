//! The brush editor: a modal over the whole window, with a live test stroke down its
//! right-hand column (§6.2, §11.2).
//!
//! The Brush shelf keeps the two knobs a hand moves all day — the size and the flow,
//! which are the transient's (§18.1.8) — and everything that says what the tool *is*
//! moved here. That is the same line the web app draws, and it is the durable/transient
//! split showing through the chrome: the panel is where a brush is *worked*, and this
//! is where one is *made*.
//!
//! **What the dialog shows is `stark_ui::brush_editor`'s.** Which parameter is in which
//! group, over what range, and which rows a liquify brush does not get are facts about
//! the engine, not about a toolkit — so they are shared with the web app and what is
//! here is the markup and the measuring.
//!
//! # The test stroke
//!
//! The preview is a **third surface** and a sibling engine on the canvas's own device
//! (`render::Preview`): same pipelines, same stamps, same substrate under the same
//! light, a document of its own. A fixed red band is laid across it once, and the test
//! stroke — a seeded S-curve with a pressure bell and a ramping tilt, or whatever the
//! artist last drew on it — is undone and replayed as a **single commit** on every
//! edit, with its jitter seed pinned so only the changed parameter moves.
//!
//! **No throttle, unlike the web app's.** Over there a slider fires an `input` event
//! per pointer sample and each one dispatches, repaints and re-renders the dialog, so
//! the edits are batched to one per 50 ms. Here a track emits at most one change per
//! frame and the replay is about a frame's worth of GPU, so the frame loop already is
//! the throttle — and a throttle on top of it would only make the stroke lag the thumb.
//!
//! # Where the controls are
//!
//! Measured, like every other control in this chrome (`crate::panel`): each carries a
//! zero-cost probe whose prepaint writes its laid-out bounds into [`Regions`], and the
//! press ladder reads that. The one addition is [`Region::Sheet`], a probe over the
//! whole panel — a press inside the dialog that hit no control is *not* a press on the
//! canvas under it, and that is the whole of what "modal" means here.

use std::collections::HashSet;

use stark_engine::ViewTransform;
use stark_engine::command::InputSample;
use stark_model::document::{ModSource, NoiseKind, OrientationSource};
use stark_model::geom::Vec2;
use stark_ui::brush_config::BrushEffectType;
use stark_ui::brush_editor::{ModRow, Row, Section, Shown, TestStroke};
use wgpui::{
    AnyElement, Bounds, Entity, IntoElement, PathBuilder, Pixels, Point, SharedString,
    WgpuSurfaceHandle, canvas, div, point, prelude::*, px, rgb, rgba, wgpu_surface,
};
use wgpui_component::slider::SliderState;

use crate::controls::Controls;
use crate::render::Preview;
use crate::style::{self, StyleExt};

/// The dialog's own size in logical px. Fixed rather than a fraction of the window:
/// the left column is a list of tracks that want a readable width and no more, and the
/// right one is a stroke that wants height. A dialog that grew with the window would
/// spend both on nothing.
const SHEET_WIDTH: f32 = 900.0;
pub const SHEET_HEIGHT: f32 = 780.0;

/// The test canvas's column. Tall and narrow because a stroke is long and thin: the
/// column buys a longer run of paint than a letterbox strip of the same area would,
/// and it leaves the settings a full-width column of their own.
pub const PREVIEW_WIDTH: f32 = 300.0;

/// The room a track's figure keeps, logical px — `crate::panel::READOUT_WIDTH`'s twin,
/// wider because this dialog prints tapers in radii and a frequency in tiles.
const READOUT_WIDTH: f32 = 34.0;

/// The response plot beside an open mapping's two knobs, logical px.
const CURVE_WIDTH: f32 = 64.0;
const CURVE_HEIGHT: f32 = 34.0;
const CURVE_STROKE: f32 = 1.5;

/// Which control a press landed on.
///
/// Each variant carries the *thing* rather than an index into a list, so a row that
/// moves as a section folds cannot come to mean a different act.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Region {
    /// The panel itself. Recorded first and read last ([`hit`] scans in reverse), so
    /// every control in the dialog wins over it and a press that hit none of them stops
    /// here rather than reaching the canvas.
    Sheet,
    /// Keep the brush and put the dialog away.
    Done,
    /// Restore the default test stroke.
    Reset,
    /// The test canvas — draw on it to replace the stroke.
    Preview,
    /// A group's title bar.
    Fold(Section),
    /// A group's "Show more".
    More(Section),
    /// One of the four effect chips.
    Effect(BrushEffectType),
    /// One of the four noise chips.
    Noise(NoiseKind),
    /// One of the two orientation chips.
    Orientation(OrientationSource),
    /// The chip that opens a row's pen mapping.
    Mapping(ModRow),
    /// A source chip inside an open mapping — `None` is "Off".
    Source(ModRow, Option<ModSource>),
}

/// Which of an open mapping's two shape knobs a track is.
///
/// A value rather than a `&mut` accessor, because the track's subscription outlives the
/// borrow it would need: the closure is `'static` and the mapping it writes into is
/// resolved on the frame the drag lands, out of whichever row is open then.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Shape {
    /// What survives a feather touch, or an upright pen — which is what makes a
    /// tilt-driven brush usable with a mouse, since a mouse reports no tilt at all.
    Floor,
    /// Negative is late, positive early, 0 exactly linear.
    Curve,
}

impl Shape {
    /// Write it into `mapping`. The response arrives already mapped out of the track's
    /// fraction, so this is the one place either number is stored.
    pub fn set(self, mapping: &mut stark_model::document::Modulation, v: f32) {
        match self {
            Self::Floor => mapping.floor = v,
            Self::Curve => mapping.curve = v,
        }
    }
}

/// Where each control was laid out, as of the last frame that painted.
pub type Regions = std::rc::Rc<std::cell::RefCell<Vec<(Region, Bounds<Pixels>)>>>;

/// Which control a press landed on.
///
/// **Reversed**, unlike the panel's: the controls here nest — a chip inside a row
/// inside a section inside the sheet — and the list is built parent-first, so the last
/// rectangle containing the point is the innermost one.
pub fn hit(regions: &Regions, at: Point<Pixels>) -> Option<Region> {
    regions
        .borrow()
        .iter()
        .rev()
        .find(|(_, bounds)| bounds.contains(&at))
        .map(|(region, _)| *region)
}

/// Where a press on the test canvas is, in the **device px** its view is denominated
/// in — `None` when the press was not on it, or when no frame has measured it yet.
pub fn preview_at(regions: &Regions, at: Point<Pixels>, scale: f32) -> Option<Vec2> {
    let bounds = regions
        .borrow()
        .iter()
        .find(|(region, _)| *region == Region::Preview)
        .map(|(_, bounds)| *bounds)?;
    Some(Vec2::new(
        (f32::from(at.x) - f32::from(bounds.origin.x)) * scale,
        (f32::from(at.y) - f32::from(bounds.origin.y)) * scale,
    ))
}

fn probe(regions: &Regions, region: Region) -> impl IntoElement {
    let regions = regions.clone();
    canvas(
        move |bounds, _, _| regions.borrow_mut().push((region, bounds)),
        |_, (), _, _| {},
    )
    // All four insets rather than `size_full`: an absolutely-positioned child with
    // only a size is placed wherever the flow had reached, which put the transform
    // overlay a viewport below the window for exactly one build (§11.2, N6).
    .absolute()
    .top_0()
    .left_0()
    .right_0()
    .bottom_0()
}

/// The editor while it is open — the sibling engine, the stroke it is showing, and
/// what the artist has folded.
///
/// Dropped whole when the dialog closes, which is what actually gives the preview's
/// surface and its document back: a `WgpuSurfaceHandle` leaves the registry with the
/// last handle, exactly as the navigator's does.
pub struct Editor {
    /// The test canvas. `None` where wgpui is not on its wgpu renderer — the dialog
    /// still opens and every knob still works, it simply shows no stroke.
    pub preview: Option<Preview>,
    /// The test stroke and what a hand on the test canvas does to it — the whole of
    /// it, since the six values and five transitions are `stark_ui`'s and the web
    /// frontend carries the same one (§11.2). What is this frontend's is the engine
    /// calls around it.
    stroke: TestStroke,
    /// Which groups are folded to their title bar. The specialised two start shut
    /// (`Section::open_by_default`).
    shut: HashSet<Section>,
    /// Which groups have their "Show more" open.
    more: HashSet<Section>,
    /// Whose pen mapping is being edited, at most one at a time — so the dialog grows
    /// by one sub-row while a mapping is open and by nothing otherwise.
    mapping: Option<ModRow>,
}

impl Editor {
    /// Open on `preview`, with the everyday groups unfolded.
    pub fn new(preview: Option<Preview>) -> Self {
        Self {
            preview,
            stroke: TestStroke::default(),
            shut: stark_ui::brush_editor::SECTIONS
                .into_iter()
                .filter(|s| !s.open_by_default())
                .collect(),
            more: HashSet::new(),
            mapping: None,
        }
    }

    /// The mapping whose sub-row is open, if any.
    pub fn open_mapping(&self) -> Option<ModRow> {
        self.mapping
    }

    /// Open a row's mapping, or shut it if it is the one already open — the escape
    /// hatch every armed control in this app offers through the control that armed it.
    pub fn toggle_mapping(&mut self, row: ModRow) {
        self.mapping = if self.mapping == Some(row) {
            None
        } else {
            Some(row)
        };
    }

    /// Fold a group away, or unfold it.
    pub fn fold(&mut self, section: Section) {
        if !self.shut.remove(&section) {
            self.shut.insert(section);
        }
    }

    /// Show a group's rarely-touched rows, or put them back.
    pub fn toggle_more(&mut self, section: Section) {
        if !self.more.remove(&section) {
            self.more.insert(section);
        }
    }

    /// Lay the fixed reference band, then seed the default test stroke.
    ///
    /// The band is committed **once**, beneath everything the dialog replays, so the
    /// single undo a re-stroke spends never reaches it — which is what lets a smudge,
    /// a knife or a liquify brush be shown working paint that is already there.
    pub fn lay_reference(&mut self) {
        let Some(p) = self.preview.as_mut() else {
            return;
        };
        let (w, h) = size_of(p);
        let view = p.view();
        p.process(stark_engine::command::ViewCommand::SetBrush {
            brush: stark_ui::brush_editor::reference_brush(),
            color: stark_ui::brush_editor::REFERENCE_COLOR,
        });
        let samples = stark_ui::brush_editor::reference_stroke(w, h, view);
        p.replay_stroke(&samples, stark_ui::brush_editor::PREVIEW_STROKE_SEED, 0.0);
        self.stroke.seed(w, h, view);
    }

    /// Put the seeded stroke back, whatever the artist drew over it.
    pub fn reset_stroke(&mut self) {
        let Some(p) = self.preview.as_ref() else {
            return;
        };
        let (w, h) = size_of(p);
        self.stroke.seed(w, h, p.view());
    }

    /// Re-lay the seeded stroke to a surface that has changed size, so the default
    /// keeps running the length of the column instead of ending short of it — and
    /// leave the artist's own alone, which is `TestStroke`'s rule.
    pub fn relay_after_resize(&mut self) {
        let Some(p) = self.preview.as_ref() else {
            return;
        };
        let (w, h) = size_of(p);
        self.stroke.relay_after_resize(w, h, p.view());
    }

    /// Undo the committed test stroke, push `brush`, and replay the recorded hand as a
    /// single commit.
    ///
    /// One render rather than one per sample: an interactive gesture folds its samples
    /// once a frame, and a replay that asked for the same would be O(n²) across the
    /// stroke and would starve the GPU for the length of every drag.
    ///
    /// A no-op while a hand is on the test canvas — what is on screen then is the
    /// stroke being drawn, and replacing it mid-gesture would take it out from under
    /// the pointer.
    pub fn restroke(&mut self, brush: &crate::brush::Brush) {
        if self.stroke.drawing() {
            return;
        }
        let Some(p) = self.preview.as_mut() else {
            return;
        };
        if self.stroke.needs_undo() {
            p.process(stark_engine::command::DocCommand::Undo);
        }
        // The test stroke wears the fixed preview gold whatever the palette holds, so
        // it reads over the red band beneath it; the effect's own opacity is untouched,
        // which is what keeps the Opacity track honest. A stamp needs no handing over:
        // the sibling engine shares the content-addressed asset store, so whatever the
        // brush names is already here.
        let mut tune = brush.tune;
        tune.color = stark_ui::brush_editor::PREVIEW_STROKE_COLOR;
        p.process(stark_engine::command::ViewCommand::SetBrush {
            brush: brush.config.params(tune),
            color: tune.color,
        });
        // The §6.11 rope the Smoothing track means *on this canvas*: the recorded test
        // stroke is a hand, and replaying it through the tow is what lets the track
        // show its work on the stroke beside it.
        let rope = stark_ui::input::rope(p.view(), brush.config.smoothing);
        let committed = p.replay_stroke(
            self.stroke.samples(),
            stark_ui::brush_editor::PREVIEW_STROKE_SEED,
            rope,
        );
        self.stroke.replayed(committed);
    }

    /// Begin a stroke the artist is drawing on the test canvas: clear the committed
    /// one and start recording.
    pub fn start_stroke(&mut self, at: Vec2, rope: f32) {
        let Some(p) = self.preview.as_mut() else {
            return;
        };
        let sample = sample_at(p.view(), at);
        // The committed test stroke comes off before the new one goes on, and whether
        // there is one to take off is the stroke's own answer (`TestStroke::begin`).
        if self.stroke.begin(sample) {
            p.process(stark_engine::command::DocCommand::Undo);
        }
        p.process(stark_engine::command::GestureCommand::Start {
            tool: stark_engine::command::Tool::Brush,
            sample,
            // Nothing to declare: a mouse on a laid-out element resolves to the px it
            // reports, and this frontend takes no stylus reports over the dialog.
            tolerance: stark_engine::path::DEFAULT_TOLERANCE,
            // Towed like the main canvas (§6.11): the preview is where a brush is felt
            // out, so drawing on it under smoothing has to feel like the brush rather
            // than like the brush with its string cut.
            rope,
        });
    }

    /// Extend it.
    pub fn stroke_to(&mut self, at: Vec2) {
        if !self.stroke.drawing() {
            return;
        }
        let Some(p) = self.preview.as_mut() else {
            return;
        };
        let sample = sample_at(p.view(), at);
        p.process(stark_engine::command::GestureCommand::To { sample });
        self.stroke.extend(sample);
    }

    /// Commit it as the new test stroke, so every later edit replays what the artist
    /// drew rather than the seeded default.
    ///
    /// Answers whether the hand actually drew one. **A tap is not a stroke**
    /// (`TestStroke::end`, which carries the argument) — so a press that went nowhere
    /// leaves the stroke that was there and the caller replays it.
    #[must_use]
    pub fn end_stroke(&mut self) -> bool {
        if !self.stroke.drawing() {
            return false;
        }
        if let Some(p) = self.preview.as_mut() {
            p.process(stark_engine::command::GestureCommand::End);
        }
        self.stroke.end()
    }
}

/// A pointer position on the test canvas as an input sample.
///
/// A mouse is always pressed home and reports no tilt, exactly as on the main canvas
/// (`canvas::sample_at`) — the pen-driven rows are shown their work by the *seeded*
/// stroke, which carries a pressure bell and a ramping lean for that reason.
fn sample_at(view: ViewTransform, at: Vec2) -> InputSample {
    InputSample {
        pos: view.screen_to_canvas(at),
        pressure: 1.0,
        ..Default::default()
    }
}

/// The preview surface's size as a pair of floats, which is what the stroke geometry
/// is laid out in.
fn size_of(p: &Preview) -> (f32, f32) {
    let (w, h) = p.size();
    (w as f32, h as f32)
}

// --- the markup -----------------------------------------------------------

/// What the dialog needs to draw itself, gathered by the view.
pub struct Dressing<'a> {
    /// The brush being edited, and the two document facts its rows depend on.
    pub shown: &'a Shown,
    pub editor: &'a Editor,
    pub controls: &'a Controls,
    /// The stamp gallery, built once by the view and handed to whichever surface wants
    /// it — while the dialog is up it is this one, since the shelf that usually holds
    /// it is behind the scrim and cannot be pressed.
    pub shapes: Option<AnyElement>,
    /// Which preset the brush descends from, so the header can say what is being
    /// tuned. `None` once a durable knob has been moved off it.
    pub from: Option<&'a str>,
}

/// The whole modal: a scrim over the window, and the sheet on it.
pub fn modal(dressing: Dressing<'_>, regions: &Regions) -> AnyElement {
    let Dressing {
        shown,
        editor,
        controls,
        mut shapes,
        from,
    } = dressing;
    let surface = editor.preview.as_ref().map(Preview::surface);
    let body = div()
        // Scrolls, so a brush whose groups are all unfolded is one whose last group
        // exists — the same answer the docked columns give (`crate::panel::column`).
        // An id, because a scroll position is state and state has to belong to
        // something.
        .id("brush-editor-body")
        .flex_1()
        .min_h_0()
        .overflow_y_scroll()
        .flex()
        .flex_col()
        .gap_2()
        .children(
            stark_ui::brush_editor::SECTIONS
                .into_iter()
                .filter(|s| s.mounted(&shown.brush))
                .map(|section| group(section, shown, editor, controls, &mut shapes, regions)),
        );
    div()
        // The scrim. All four insets, and a ground with alpha rather than a solid one:
        // what is being tuned is a brush, and the canvas it will paint on staying
        // visible behind the dialog is the point.
        .absolute()
        .top_0()
        .left_0()
        .right_0()
        .bottom_0()
        .flex()
        .items_center()
        .justify_center()
        .bg(rgba(style::SCRIM))
        .child(
            div()
                .relative()
                .w(px(SHEET_WIDTH))
                .h(px(SHEET_HEIGHT))
                .flex()
                .flex_col()
                .rounded_md()
                .overflow_hidden()
                .border_1()
                .border_color(rgb(style::EDGE))
                .bg(rgb(style::PANEL))
                .text_color(rgb(style::INK_LIT))
                // First child, so every control laid out after it wins the reverse
                // scan in `hit`.
                .child(probe(regions, Region::Sheet))
                .child(header(from, regions))
                .child(
                    div()
                        .flex_1()
                        .min_h_0()
                        .flex()
                        .child(div().flex_1().min_w_0().p_3().flex().flex_col().child(body))
                        .child(preview_column(surface, regions)),
                ),
        )
        .into_any_element()
}

/// The title row, and the one way out of the dialog.
///
/// One button rather than the web app's three: keeping a brush under a name is a
/// record and a dialog of its own, and this frontend has neither yet
/// (`crate::brush::Brush::library`). What it does have is the same claim Done makes —
/// the brush in hand is already the edited one, because every track writes straight
/// through to it.
fn header(from: Option<&str>, regions: &Regions) -> impl IntoElement {
    let title = div()
        .flex()
        .items_baseline()
        .gap_2()
        .child(div().text_color(rgb(style::INK_LIT)).child("Brush"))
        // Which preset the brush descends from, said where it can be read rather than
        // deduced from the row that is still lit two columns away.
        .children(from.map(|name| div().caption().child(SharedString::from(name.to_string()))));
    div()
        .flex()
        .items_center()
        .justify_between()
        .px_3()
        .py_2()
        .border_b_1()
        .border_color(rgb(style::EDGE))
        .child(title)
        .child(style::tip(
            div()
                .id("brush-editor-done")
                .chip()
                .flex()
                .items_center()
                .gap_1()
                .px_2()
                .py_1()
                .lit(true)
                .child(probe(regions, Region::Done))
                .child(crate::icons::icon(stark_ui::icons::DONE, style::INK_LIT))
                .child("Done"),
            "Keep the brush in hand and close",
        ))
}

/// The test canvas, the sentence that says it can be drawn on, and the turn-back mark.
fn preview_column(surface: Option<WgpuSurfaceHandle>, regions: &Regions) -> impl IntoElement {
    div()
        .w(px(PREVIEW_WIDTH))
        .flex_none()
        .h_full()
        .relative()
        .border_l_1()
        .border_color(rgb(style::EDGE))
        .bg(rgb(style::WELL))
        .child(probe(regions, Region::Preview))
        .children(surface.map(|handle| wgpu_surface(handle).size_full()))
        // **One overlay, with all four insets.** An absolutely-positioned child given
        // only a corner is placed wherever the flow had reached (§11.2, N6), which put
        // both of these below the column: invisible, and — since a probe goes with
        // them — a Reset the press ladder could not find, so pressing it drew a
        // one-point stroke on the test canvas instead. The two marks share one
        // full-size box now and are placed *inside* it by ordinary flex.
        .child(
            div()
                .absolute()
                .top_0()
                .left_0()
                .right_0()
                .bottom_0()
                .p_2()
                .flex()
                .flex_col()
                .justify_between()
                .child(
                    div().flex().justify_end().child(
                        // The turning arrow Undo wears, because putting the preview
                        // back *is* an undo — narrowed to the one stroke this dialog
                        // owns.
                        style::tip(
                            div()
                                .id("brush-editor-reset")
                                .chip()
                                .flex()
                                .items_center()
                                .justify_center()
                                .w(px(22.))
                                .h(px(22.))
                                .resting()
                                .child(probe(regions, Region::Reset))
                                .child(crate::icons::icon(stark_ui::icons::RESET, style::INK_MARK)),
                            "Restore the default test stroke",
                        ),
                    ),
                )
                .child(
                    // On its own dark pill rather than bare: the test canvas wears the
                    // *document's* substrate colour, which can be anything, and a
                    // caption that reads on white disappears on a dark ground. The
                    // transform bar's translucent panel, for the transform bar's
                    // reason — the stroke under it stays visible.
                    div().flex().child(
                        div()
                            .px_1p5()
                            .py_0p5()
                            .rounded_sm()
                            .bg(rgba(style::PANEL_OVER_CANVAS))
                            .caption()
                            .child("Test stroke \u{2014} draw here to replace it"),
                    ),
                ),
        )
}

/// One group: a title bar that folds it, and — unfolded — its sentence and its rows.
fn group(
    section: Section,
    shown: &Shown,
    editor: &Editor,
    controls: &Controls,
    shapes: &mut Option<AnyElement>,
    regions: &Regions,
) -> AnyElement {
    let open = !editor.shut.contains(&section);
    let title = style::tip(
        div()
            .id(SharedString::from(format!(
                "brush-editor-{}",
                section.key()
            )))
            .relative()
            .flex()
            .items_center()
            .gap_2()
            .cursor_pointer()
            .child(probe(regions, Region::Fold(section)))
            .child(crate::icons::icon(section.glyph(), style::INK_LABEL))
            .child(
                div()
                    .flex_1()
                    .text_sm()
                    .text_color(rgb(style::INK_LIT))
                    .child(section.title(&shown.brush)),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(style::INK_FOLD))
                    // Down for open, right for folded — which way the content lies, not
                    // which way pressing it would go.
                    .child(if open { "\u{25be}" } else { "\u{25b8}" }),
            ),
        if open { "Fold away" } else { "Unfold" },
    );
    let more_open = editor.more.contains(&section);
    let more_rows = section.more(shown);
    let body = open.then(|| {
        div()
            .flex()
            .flex_col()
            .gap_1()
            .pl_1()
            .child(div().caption().child(section.desc(&shown.brush)))
            .children(
                section
                    .rows(shown)
                    .into_iter()
                    .map(|row| line(row, shown, editor, controls, shapes, regions)),
            )
            .children((!more_rows.is_empty()).then(|| {
                div()
                    .id(SharedString::from(format!("more-{}", section.key())))
                    .relative()
                    .caption()
                    .cursor_pointer()
                    .py_0p5()
                    .child(probe(regions, Region::More(section)))
                    .child(if more_open {
                        "\u{25be} Fewer"
                    } else {
                        "\u{25b8} Show more"
                    })
            }))
            .children(more_open.then(|| {
                div().flex().flex_col().gap_1().children(
                    more_rows
                        .into_iter()
                        .map(|row| line(row, shown, editor, controls, shapes, regions)),
                )
            }))
    });
    div()
        .flex()
        .flex_col()
        .gap_1()
        .pt_1()
        .child(title)
        .children(body)
        .into_any_element()
}

/// One row of a group.
fn line(
    row: Row,
    shown: &Shown,
    editor: &Editor,
    controls: &Controls,
    shapes: &mut Option<AnyElement>,
    regions: &Regions,
) -> AnyElement {
    match row {
        Row::Shapes => shapes.take().unwrap_or_else(|| div().into_any_element()),
        Row::Orientation => chips(
            [OrientationSource::FollowStroke, OrientationSource::Pen]
                .into_iter()
                .map(|o| {
                    (
                        Region::Orientation(o),
                        orientation_label(o),
                        shown.brush.orientation == o,
                        orientation_tip(o),
                    )
                }),
            regions,
        ),
        Row::Effects => chips(
            [
                BrushEffectType::Paint,
                BrushEffectType::Wet,
                BrushEffectType::Erase,
                BrushEffectType::Liquify,
            ]
            .into_iter()
            .map(|e| {
                (
                    Region::Effect(e),
                    effect_label(e),
                    shown.brush.effect == e,
                    effect_tip(e),
                )
            }),
            regions,
        ),
        Row::Noise => chips(
            stark_ui::brush_editor::NOISE_KINDS.into_iter().map(|k| {
                (
                    Region::Noise(k),
                    stark_ui::brush_editor::noise_label(k),
                    shown.brush.color_dynamics.noise == k,
                    "The field the color wanders across",
                )
            }),
            regions,
        ),
        Row::Note(note) => div()
            .text_xs()
            .text_color(rgb(style::INK_LABEL))
            .px_1()
            .py_1()
            .rounded_sm()
            .bg(rgb(style::WELL))
            .child(note.text())
            .into_any_element(),
        Row::Knob(knob) => track(
            controls.editor_knob(knob),
            knob.glyph(),
            knob.label(shown.space),
            figure(knob.get(&shown.brush)),
        )
        .into_any_element(),
        Row::Mod(m) => mod_row(m, shown, editor, controls, regions),
    }
}

/// A parameter track with its **pen mapping** hung off the end (§6.2): the track
/// exactly as a plain one, plus a chip naming what drives it, which opens the
/// mapping's own controls in place.
///
/// The chip is the whole design decision. A brush with no mapping on this row reads as
/// one word and one track, as it always did; the mapping is a second line only while
/// it is being edited, and only ever one row at a time. The alternative — a "pen
/// response" group listing every target — puts the control a fold away from the
/// parameter it drives, so reading a brush means holding two lists against each other.
fn mod_row(
    m: ModRow,
    shown: &Shown,
    editor: &Editor,
    controls: &Controls,
    regions: &Regions,
) -> AnyElement {
    let mapping = m.of(&shown.brush);
    let open = editor.mapping == Some(m);
    let value = m.get(&shown.brush, shown.tune);
    let row = div()
        .flex()
        .items_center()
        .gap_2()
        .child(div().flex_1().min_w_0().child(track(
            controls.editor_mod(m),
            m.glyph(),
            m.label(&shown.brush),
            if m == ModRow::Size {
                // Whole px: a radius is a length, and two decimals of one is a
                // figure nobody reads.
                format!("{value:.0}")
            } else {
                figure(value)
            },
        )))
        .child(style::tip(
            div()
                .id(SharedString::from(format!("mod-{m:?}")))
                .chip()
                .flex()
                .items_center()
                .justify_center()
                .px_1p5()
                .py_0p5()
                .lit(mapping.is_some() || open)
                .child(probe(regions, Region::Mapping(m)))
                .child(match mapping {
                    Some(mapping) => div()
                        .child(stark_ui::brush_editor::source_label(mapping.source))
                        .into_any_element(),
                    None => crate::icons::icon(stark_ui::icons::MODULATE, style::INK_MARK)
                        .into_any_element(),
                }),
            "What the pen drives this with",
        ));
    let panel = open.then(|| {
        div()
            .flex()
            .flex_col()
            .gap_1()
            .ml_3()
            .pl_2()
            .py_1()
            .border_l_1()
            .border_color(rgb(style::RULE))
            .child(chips(
                std::iter::once((
                    Region::Source(m, None),
                    "Off",
                    mapping.is_none(),
                    "Nothing drives this",
                ))
                .chain(stark_ui::brush_editor::SOURCES.into_iter().map(
                    |source| {
                        (
                            Region::Source(m, Some(source)),
                            stark_ui::brush_editor::source_label(source),
                            mapping.is_some_and(|mapping| mapping.source == source),
                            source_tip(source),
                        )
                    },
                )),
                regions,
            ))
            .children(mapping.map(|mapping| {
                // The two shape knobs, and the curve they describe drawn beside them —
                // the factor is what actually reaches the renderer, and a number pair
                // for a curve is the one thing a picture reads better.
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .flex()
                            .flex_col()
                            .child(track(
                                &controls.mod_floor,
                                None,
                                "At zero",
                                figure(mapping.floor),
                            ))
                            .child(track(
                                &controls.mod_curve,
                                None,
                                "Response",
                                figure(mapping.curve),
                            )),
                    )
                    .child(curve_plot(mapping))
            }))
    });
    div()
        .flex()
        .flex_col()
        .child(row)
        .children(panel)
        .into_any_element()
}

/// One track: an optional mark, the word, the trough and the figure it stands at.
///
/// Words rather than marks alone, unlike the columns (`crate::panel`): a docked column
/// never had room for a label and a dialog opened for one job has nothing else to spend
/// its width on. Two dozen unlabelled troughs would be a dialog nobody can read.
fn track(
    state: &Entity<SliderState>,
    glyph: Option<stark_ui::icons::Icon>,
    label: &'static str,
    readout: String,
) -> impl IntoElement {
    div()
        .flex()
        .items_center()
        .gap_2()
        .py_0p5()
        .children(glyph.map(|g| crate::icons::icon(g, style::INK_MARK)))
        .child(div().w(px(LABEL_WIDTH)).flex_none().caption().child(label))
        .child(
            div()
                .flex_1()
                .min_w_0()
                .child(wgpui_component::slider::Slider::new(state).w_full()),
        )
        .child(
            // The figure keeps its own column so the troughs all end in the same place
            // however wide the number is.
            div()
                .w(px(READOUT_WIDTH))
                .flex_none()
                .text_right()
                .caption()
                .child(SharedString::from(readout)),
        )
}

/// The room a track's word keeps, logical px — the longest of them is
/// "Scale → across stroke", and a column that fits it is a column no row wraps in.
const LABEL_WIDTH: f32 = 124.0;

/// How the dialog prints a value: enough digits to see a change, no more.
fn figure(v: f32) -> String {
    format!("{v:.2}")
}

/// A run of two-state chips, each with its own region and hover.
fn chips<'a>(
    items: impl Iterator<Item = (Region, &'a str, bool, &'static str)>,
    regions: &Regions,
) -> AnyElement {
    div()
        .flex()
        .flex_wrap()
        .gap_1()
        .py_0p5()
        .children(items.map(|(region, label, on, tip)| {
            style::tip(
                div()
                    .id(SharedString::from(format!("{region:?}")))
                    .chip()
                    .px_2()
                    .py_1()
                    .lit(on)
                    .child(probe(regions, region))
                    .child(SharedString::from(label.to_string())),
                tip,
            )
        }))
        .into_any_element()
}

/// The mapping's response, painted: input left → right, the factor it multiplies the
/// parameter by bottom → top.
///
/// The points are `stark_ui::brush_editor::curve_points`, which samples
/// `Modulation::factor` itself rather than redrawing the formula — so the picture
/// cannot disagree with the renderer, about the floor or about the clamps. What is this
/// frontend's is only the box: the web app draws the same numbers into an SVG viewbox.
fn curve_plot(mapping: stark_model::document::Modulation) -> impl IntoElement {
    let points = stark_ui::brush_editor::curve_points(mapping);
    div()
        .w(px(CURVE_WIDTH))
        .h(px(CURVE_HEIGHT))
        .flex_none()
        .rounded_sm()
        .bg(rgb(style::WELL))
        .child(
            canvas(
                move |_, _, _| {},
                move |bounds: Bounds<Pixels>, (), window, _| {
                    // A hair of padding, so a factor pinned at 0 or 1 is a line inside
                    // the box rather than half a line on its edge.
                    let pad = CURVE_STROKE;
                    let w = f32::from(bounds.size.width) - 2.0 * pad;
                    let h = f32::from(bounds.size.height) - 2.0 * pad;
                    let at = |(x, y): (f32, f32)| {
                        point(
                            bounds.origin.x + px(pad + x * w),
                            bounds.origin.y + px(pad + (1.0 - y) * h),
                        )
                    };
                    let Some((first, rest)) = points.split_first() else {
                        return;
                    };
                    let mut path = PathBuilder::stroke(px(CURVE_STROKE));
                    path.move_to(at(*first));
                    for p in rest {
                        path.line_to(at(*p));
                    }
                    if let Ok(path) = path.build() {
                        window.paint_path(path, rgb(style::ACCENT));
                    }
                },
            )
            .size_full(),
        )
}

/// The word an orientation chip wears, and what the hover says it does.
fn orientation_label(o: OrientationSource) -> &'static str {
    match o {
        OrientationSource::FollowStroke => "Follow stroke",
        OrientationSource::Pen => "Pen angle",
    }
}

fn orientation_tip(o: OrientationSource) -> &'static str {
    match o {
        OrientationSource::FollowStroke => {
            "The footprint turns with the travel \u{2014} a nib that follows the line"
        }
        OrientationSource::Pen => {
            "The footprint follows the pen's lean \u{2014} a real conical tip"
        }
    }
}

/// The word an effect chip wears. The marks are the panel's, where a chip has no room
/// for a word; here the word leads, because the chip is what names the *group* under it.
fn effect_label(effect: BrushEffectType) -> &'static str {
    match effect {
        BrushEffectType::Paint => "Paint",
        BrushEffectType::Wet => "Wet",
        BrushEffectType::Erase => "Erase",
        BrushEffectType::Liquify => "Liquify",
    }
}

fn effect_tip(effect: BrushEffectType) -> &'static str {
    match effect {
        BrushEffectType::Paint => "Paint \u{2014} lay the colour in hand",
        BrushEffectType::Wet => "Wet \u{2014} move and mix the paint already on the canvas",
        BrushEffectType::Erase => "Erase \u{2014} take paint away where the tip passes",
        BrushEffectType::Liquify => "Liquify \u{2014} push the paint about without adding any",
    }
}

fn source_tip(source: ModSource) -> &'static str {
    match source {
        ModSource::Pressure => "How hard the pen is pressed",
        ModSource::Tilt => "How far the pen is leaned over",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpui::size;

    fn at(x: f32, y: f32) -> Point<Pixels> {
        point(px(x), px(y))
    }

    fn measured(rows: &[(Region, f32, f32, f32, f32)]) -> Regions {
        let regions = Regions::default();
        for (region, x, y, w, h) in rows {
            regions.borrow_mut().push((
                *region,
                Bounds {
                    origin: point(px(*x), px(*y)),
                    size: size(px(*w), px(*h)),
                },
            ));
        }
        regions
    }

    /// The sheet is recorded first and every control after it, so a press on a control
    /// finds the control — and a press inside the dialog that hit none of them finds the
    /// sheet rather than falling through to the canvas.
    #[test]
    fn a_control_wins_over_the_sheet_it_is_on() {
        let regions = measured(&[
            (Region::Sheet, 100.0, 100.0, 400.0, 400.0),
            (Region::Done, 420.0, 110.0, 60.0, 24.0),
        ]);
        assert_eq!(hit(&regions, at(440.0, 120.0)), Some(Region::Done));
        assert_eq!(hit(&regions, at(200.0, 300.0)), Some(Region::Sheet));
        assert_eq!(hit(&regions, at(50.0, 50.0)), None, "the scrim is nobody's");
    }

    /// A press on the test canvas maps into the surface's own device px, measured from
    /// where the element was laid out.
    #[test]
    fn a_press_on_the_test_canvas_lands_in_its_own_frame() {
        let regions = measured(&[(Region::Preview, 600.0, 140.0, 300.0, 500.0)]);
        let got = preview_at(&regions, at(610.0, 160.0), 2.0).expect("measured");
        assert_eq!(got, Vec2::new(20.0, 40.0));
        // Nothing measured is nothing to map into, rather than a guess at the origin.
        assert!(preview_at(&Regions::default(), at(610.0, 160.0), 1.0).is_none());
    }

    /// Folding is a toggle, and the specialised groups open shut.
    #[test]
    fn the_everyday_groups_open_unfolded() {
        let mut editor = Editor::new(None);
        assert!(!editor.shut.contains(&Section::Tip));
        assert!(!editor.shut.contains(&Section::Effect));
        assert!(editor.shut.contains(&Section::Color));
        assert!(editor.shut.contains(&Section::Wet));
        editor.fold(Section::Tip);
        assert!(editor.shut.contains(&Section::Tip));
        editor.fold(Section::Tip);
        assert!(!editor.shut.contains(&Section::Tip));
    }

    /// Only ever one mapping open, and pressing the open one shuts it.
    #[test]
    fn one_mapping_is_open_at_a_time() {
        let mut editor = Editor::new(None);
        assert_eq!(editor.open_mapping(), None);
        editor.toggle_mapping(ModRow::Size);
        assert_eq!(editor.open_mapping(), Some(ModRow::Size));
        editor.toggle_mapping(ModRow::Flow);
        assert_eq!(editor.open_mapping(), Some(ModRow::Flow));
        editor.toggle_mapping(ModRow::Flow);
        assert_eq!(editor.open_mapping(), None);
    }
}
