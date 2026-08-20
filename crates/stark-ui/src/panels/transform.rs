//! The transform mode's chrome (§16.6, §16.8, §16.9): one bar, three
//! families, each with a control surface whose **shape is the deformation**:
//!
//! - **Free** (the affine) — an ellipse, not a box of handles: the image of a
//!   reference circle under the accumulated linear map. Drag inside to
//!   translate, on the rim to rotate/scale, outside to stretch/shear — the
//!   grabbed point follows the pointer exactly in every family.
//! - **Perspective** — the image of the source rect is a quad with corner
//!   handles; the map is *defined* as "the homography putting the corners
//!   where the hand put them", so the widget cannot disagree with the paint.
//!   Edges shift whole sides (the foreshortening gesture); inside translates.
//!   The receding grid drawn through the quad is the transformed space itself.
//! - **Warp** — a 4×4 control mesh whose drawn curves are sampled from the
//!   same smooth surface the paint resamples through. Drag a point, or grab
//!   the surface anywhere: the least-norm control move keeps the grabbed
//!   *paint* under the pointer — the hand holds the painting, not a handle.
//!
//! The bar selects the family. A switch **carries** the deformation when the
//! new family contains the old one exactly (affine ⊂ perspective, and the
//! smooth mesh reproduces any affine); otherwise it **commits** first — one
//! honest undo step — and reopens fresh around the moved paint, because a
//! lossy carry would silently change what "Done" produces.
//!
//! Everything before "Done" is a **lossless preview**: every change previews
//! through `preview::TRANSFORM` — the same renderer the commit
//! runs, over the committed tiles, so what is on screen is exactly what "Done"
//! will produce. "Done" commits a single `DocCommand::Transform`: one undo
//! step per gesture, like the frame drag (§15.7).
//!
//! While the mode is active the full-viewport catcher owns the pointer, so a
//! stray drag cannot paint — but navigation still works: middle-drag and
//! space-drag pan, the wheel zooms (see `input::Nav`). All gesture maths is in
//! canvas space, so panning or zooming mid-gesture cannot corrupt it.

use dioxus::html::input_data::MouseButton;
use dioxus::prelude::*;

use super::frame::{content_rect, view_rect};
use crate::commands::{self, Command};
use crate::gesture::{
    MeshRegion, PerspectiveUi, QuadRegion, TransformRegion, TransformState, TransformUi, WARP_GRID,
    WarpUi,
};
use crate::icons::{self, icon, label};
use crate::input::{Nav, page_xy};
use crate::layout::chrome_class;
use crate::preview;
use crate::state::AppState;
use crate::widgets::CommandButton;
use stark_engine::ViewTransform;
use stark_model::document::TransformMap;
use stark_model::geom::Vec2;

/// Half-width of the rim's / an edge's grab band, screen px — converted to
/// canvas px by the zoom, so it is equally grabbable at any magnification.
const RIM_BAND_PX: f32 = 10.0;

/// Grab radius of a corner or control-point handle, screen px.
const HANDLE_PX: f32 = 14.0;

/// Screen-px floor for the widget's radius at entry, so a hairline selection
/// still mounts a circle with an inside to translate by.
const MIN_RADIUS_PX: f32 = 28.0;

/// Screen-px floor for a perspective/warp source rect's extent at entry — a
/// hairline hull still mounts a quad with corners apart enough to grab.
const MIN_RECT_PX: f32 = 56.0;

/// Pointer travel below which a gesture reads as a jiggle and snaps back to its
/// start (screen px): an accidental touch must never resample the paint.
const SNAP_PX: f32 = 2.0;

/// Enter transform mode around the current selection, in the Free (affine)
/// family.
///
/// The widget mounts around the selection's analytic hull; an unbounded
/// selection (select-all, an inversion) falls back to the painted content's
/// bounds — the whole layer is what an unbounded selection holds — and an
/// empty canvas to the view, so the widget always exists. The target layer is
/// the active one, or the topmost paintable layer when a matte is selected (a
/// matte refuses transforms the same way it refuses strokes).
pub fn begin_transform(state: AppState) {
    // One composing mode at a time (`crate::modes`): whatever was in hand is put
    // down before this takes the canvas, so the case of two catchers over one
    // pointer never arises.
    crate::modes::leave(state);
    let obs = state.obs.read();
    let Some(o) = obs.as_ref() else { return };
    let layer = o
        .layers
        .iter()
        .find(|l| l.id == o.active_layer && l.is_paintable())
        .or_else(|| o.layers.iter().rev().find(|l| l.is_paintable()))
        .map(|l| l.id);
    let Some(layer) = layer else { return };
    let hull = o
        .selection_hull
        .or_else(|| content_rect(o))
        .unwrap_or_else(|| view_rect(o));
    let min_radius = MIN_RADIUS_PX / o.view.zoom;
    let rect = inflate(hull, MIN_RECT_PX / o.view.zoom);
    drop(obs);
    let mut mode = state.transform;
    mode.set(Some(TransformUi::Affine {
        rect,
        ts: TransformState::begin(layer, hull, min_radius),
    }));
}

/// `rect`, grown symmetrically wherever an axis is thinner than `min` — the
/// perspective/warp source rect must have corners a hand can tell apart.
fn inflate(rect: (Vec2, Vec2), min: f32) -> (Vec2, Vec2) {
    let (mut lo, mut hi) = rect;
    for axis in 0..2 {
        let (a, b) = (lo[axis], hi[axis]);
        if b - a < min {
            let pad = (min - (b - a)) * 0.5;
            lo[axis] = a - pad;
            hi[axis] = b + pad;
        }
    }
    (lo, hi)
}

/// Update the gesture and show its consequence — every mutation funnels through
/// here, so the preview can never lag the state.
fn update(state: AppState, ui: TransformUi) {
    let mut mode = state.transform;
    mode.set(Some(ui));
    preview::TRANSFORM.show(state, (ui.layer(), ui.map()));
}

/// The three chips on the bar. A selector enum of its own (rather than
/// matching on `TransformUi`) so the bar can name a family the gesture is not
/// currently in.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Family {
    Free,
    Perspective,
    Warp,
}

fn family_of(ui: &TransformUi) -> Family {
    match ui {
        TransformUi::Affine { .. } => Family::Free,
        TransformUi::Perspective(_) => Family::Perspective,
        TransformUi::Warp(_) => Family::Warp,
    }
}

/// Switch the composing family. Carries the accumulated deformation when the
/// new family holds it exactly; otherwise commits it (one undo step) and
/// reopens fresh around the moved paint — never a silent approximation.
fn switch_family(state: AppState, ui: TransformUi, to: Family) {
    if family_of(&ui) == to {
        return;
    }
    let layer = ui.layer();
    let fresh = |rect: (Vec2, Vec2), zoom: f32| match to {
        Family::Free => TransformUi::Affine {
            rect,
            ts: TransformState::begin(layer, rect, MIN_RADIUS_PX / zoom),
        },
        Family::Perspective => TransformUi::Perspective(PerspectiveUi::begin(
            layer,
            inflate(rect, MIN_RECT_PX / zoom),
        )),
        Family::Warp => TransformUi::Warp(WarpUi::begin(layer, inflate(rect, MIN_RECT_PX / zoom))),
    };
    let zoom = state
        .obs
        .read()
        .as_ref()
        .map(|o| o.view.zoom)
        .unwrap_or(1.0);

    // Lossless carries: an orientation-preserving affine is exactly a
    // parallelogram perspective, and exactly a mesh whose smooth surface
    // reproduces it (cubic interpolation reproduces affine functions).
    if let TransformUi::Affine { rect, ts } = ui {
        let affine = ts.affine();
        if affine.matrix2.determinant() > 0.0 {
            let carried = match to {
                Family::Free => unreachable!("same family returned above"),
                Family::Perspective => {
                    let mut p = PerspectiveUi::begin(layer, inflate(rect, MIN_RECT_PX / zoom));
                    p.corners = p.corners.map(|c| affine.transform_point2(c));
                    TransformUi::Perspective(p)
                }
                Family::Warp => {
                    let mut w = WarpUi::begin(layer, inflate(rect, MIN_RECT_PX / zoom));
                    for pt in &mut w.points {
                        *pt = affine.transform_point2(*pt);
                    }
                    TransformUi::Warp(w)
                }
            };
            if carried.map().usable() {
                update(state, carried);
                return;
            }
        }
        // A mirrored or degenerate affine has no image in the other families
        // (their maps preserve orientation): commit it first, below.
    }

    if ui.is_identity() {
        // Nothing composed yet: switching is free, around the same paint.
        let rect = match ui {
            TransformUi::Affine { rect, .. } => rect,
            TransformUi::Perspective(p) => p.rect,
            TransformUi::Warp(w) => w.rect,
        };
        let next = fresh(rect, zoom);
        let mut mode = state.transform;
        mode.set(Some(next));
        return;
    }

    // The deformation cannot ride into the new family exactly: commit it —
    // one honest undo step — and reopen around where the paint now is.
    let map = ui.map();
    let moved = match &map {
        TransformMap::Affine(a) => {
            let rect = match ui {
                TransformUi::Affine { rect, .. } => rect,
                _ => unreachable!("only the affine family reaches here with an affine map"),
            };
            let corners =
                stark_model::document::rect_corners(rect.0, rect.1).map(|c| a.transform_point2(c));
            let lo = corners.iter().fold(corners[0], |m, p| m.min(*p));
            let hi = corners.iter().fold(corners[0], |m, p| m.max(*p));
            (lo, hi)
        }
        TransformMap::Perspective(p) => p.image_aabb(),
        TransformMap::Warp(w) => w.image_aabb().unwrap_or((w.min, w.max)),
    };
    preview::TRANSFORM.commit(state, (layer, map));
    let next = fresh(moved, zoom);
    let mut mode = state.transform;
    mode.set(Some(next));
}

/// Commit the gesture and leave the mode — the bar's "Done", and Enter's
/// (`crate::modes::finish`). An identity transform just drops the preview
/// rather than spending an undo step on a no-op.
pub fn finish(state: AppState) {
    let Some(ui) = *state.transform.peek() else {
        return;
    };
    if ui.is_identity() {
        preview::TRANSFORM.clear(state);
    } else {
        // The commit clears the preview itself, so there is no intermediate
        // frame showing the untransformed document.
        preview::TRANSFORM.commit(state, (ui.layer(), ui.map()));
    }
    let mut mode = state.transform;
    mode.set(None);
}

/// The transform bar: the family selector, the affine's two flips, Cancel and
/// "Done". Mounted only while the gesture is in flight, in the same bottom
/// column as the selection and frame bars — wearing the composing register
/// (`mode-bar`) those two do not, because this bar fronts a catcher that has
/// taken the pointer away from painting (MODAL_DESIGN.md).
#[component]
pub fn TransformBar() -> Element {
    let state = use_context::<AppState>();
    let Some(ui) = *state.transform.read() else {
        return rsx! {};
    };
    let family = family_of(&ui);
    let chip = |on: bool| if on { "chip active" } else { "chip" };

    rsx! {
        div { class: chrome_class(state, "transform-bar mode-bar"),
            // The same mark the selection bar's Transform chip wears — this bar stands
            // in for that one for the gesture's duration, so it carries the glyph of
            // the button that raised it.
            span { class: "bar-label",
                {icon(icons::TRANSFORM)}
                {label("Transform")}
            }

            span { class: "bar-sep" }

            // The three families. Switching carries the deformation when the new
            // family holds it exactly (free → perspective, free → warp), and
            // commits it first when it cannot — never a silent approximation.
            button {
                class: chip(family == Family::Free),
                title: "Move, scale, rotate, shear — the ellipse widget",
                onclick: move |_| switch_family(state, ui, Family::Free),
                {icon(icons::TRANSFORM)}
                {label("Free")}
            }
            button {
                class: chip(family == Family::Perspective),
                title: "Drag the corners into a perspective (§16.8)",
                onclick: move |_| switch_family(state, ui, Family::Perspective),
                {icon(icons::PERSPECTIVE)}
                {label("Perspective")}
            }
            button {
                class: chip(family == Family::Warp),
                title: "Bend the paint through a mesh (§16.9)",
                onclick: move |_| switch_family(state, ui, Family::Warp),
                {icon(icons::WARP)}
                {label("Warp")}
            }
            if family == Family::Free {
                span { class: "bar-sep" }
                // The axis was already the only thing distinguishing these two buttons,
                // and it is carried by the glyph: a picture of the mirroring itself.
                button {
                    class: "chip",
                    title: "Mirror left \u{2194} right",
                    onclick: move |_| {
                        if let TransformUi::Affine { rect, ts } = ui {
                            update(state, TransformUi::Affine { rect, ts: ts.flipped_h() });
                        }
                    },
                    {icon(icons::FLIP_H)}
                    {label("Flip")}
                }
                button {
                    class: "chip",
                    title: "Mirror top \u{2195} bottom",
                    onclick: move |_| {
                        if let TransformUi::Affine { rect, ts } = ui {
                            update(state, TransformUi::Affine { rect, ts: ts.flipped_v() });
                        }
                    },
                    {icon(icons::FLIP_V)}
                    {label("Flip")}
                }
            }
            span { class: "bar-sep" }
            // The way out that keeps nothing — the act `modes::leave` performs
            // for every other entry point, finally offered as itself. Worn
            // whole off the registry, Esc advertisement included.
            CommandButton { command: Command::CancelMode }
            button {
                class: "chip",
                title: commands::advertised(
                    "Apply the transform \u{2014} one undo step",
                    Command::FinishMode,
                    &state.bindings.read(),
                ),
                onclick: move |_| finish(state),
                {icon(icons::DONE)}
                {label("Done")}
            }
        }
    }
}

/// An in-flight drag: which family and region it started in (locked at the
/// press — crossing a boundary mid-drag must not change what the hand is
/// doing), where it started in canvas px, and the state as it was then.
/// Everything is recomputed from the start rather than accumulated, so
/// rounding cannot drift over a long drag; it lives in canvas space, so
/// panning or zooming mid-gesture cannot corrupt it.
#[derive(Clone, Copy)]
enum Drag {
    Affine {
        region: TransformRegion,
        from: Vec2,
        rect: (Vec2, Vec2),
        start: TransformState,
    },
    Quad {
        region: QuadRegion,
        from: Vec2,
        start: PerspectiveUi,
    },
    Mesh {
        region: MeshRegion,
        from: Vec2,
        start: WarpUi,
        /// Surface-grab data, computed once at the press: the basis weights at
        /// the grabbed grid fraction and their squared norm (§16.9).
        basis: [f32; WARP_GRID * WARP_GRID],
        norm: f32,
    },
}

/// The transform widget: a full-viewport catcher that owns every pointer event
/// and classifies it against the current family's control surface (the maths
/// lives on [`TransformState`] / [`PerspectiveUi`] / [`WarpUi`]), plus the
/// purely visual chrome — the affine's ellipse, or an SVG of the quad/mesh
/// whose lines are the deformation itself. No per-handle DOM: the whole
/// viewport is the control surface.
#[component]
pub fn TransformOverlay() -> Element {
    let state = use_context::<AppState>();
    let mut drag = use_signal(|| None::<Drag>);
    // The cursor the resting pointer has earned, for feedback only.
    let mut hover = use_signal(|| "");
    // The canvas's own navigation bindings, live on the catcher: composing a
    // transform must not cost the view (see `input::Nav`).
    let nav = Nav::use_nav(state);

    let Some(ui) = *state.transform.read() else {
        return rsx! {};
    };
    let view = match state.obs.read().as_ref() {
        Some(o) => o.view,
        None => return rsx! {},
    };

    let to_canvas = move |e: &Event<PointerData>| view.screen_to_canvas(page_xy(e));
    let band = RIM_BAND_PX / view.zoom;
    let grab = HANDLE_PX / view.zoom;
    let snap = SNAP_PX / view.zoom;

    let classify = move |pc: Vec2| -> (Drag, &'static str) {
        match ui {
            TransformUi::Affine { rect, ts } => {
                let region = ts.region(pc, band);
                let cursor = match region {
                    TransformRegion::Inside => "cursor: move;",
                    TransformRegion::Rim => "cursor: grab;",
                    TransformRegion::Outside => "cursor: crosshair;",
                };
                (
                    Drag::Affine {
                        region,
                        from: pc,
                        rect,
                        start: ts,
                    },
                    cursor,
                )
            }
            TransformUi::Perspective(p) => {
                let region = p.region(pc, grab, band);
                let cursor = match region {
                    QuadRegion::Corner(_) => "cursor: grab;",
                    QuadRegion::Edge(..) => "cursor: grab;",
                    QuadRegion::Inside | QuadRegion::Outside => "cursor: move;",
                };
                (
                    Drag::Quad {
                        region,
                        from: pc,
                        start: p,
                    },
                    cursor,
                )
            }
            TransformUi::Warp(w) => {
                let region = w.region(pc, grab);
                let (basis, norm) = match region {
                    MeshRegion::Inside => {
                        let (_, basis, norm) = w.grab(pc);
                        (basis, norm)
                    }
                    _ => ([0.0; WARP_GRID * WARP_GRID], 1.0),
                };
                let cursor = match region {
                    MeshRegion::Point(_) => "cursor: grab;",
                    MeshRegion::Inside => "cursor: grab;",
                    MeshRegion::Outside => "cursor: move;",
                };
                (
                    Drag::Mesh {
                        region,
                        from: pc,
                        start: w,
                        basis,
                        norm,
                    },
                    cursor,
                )
            }
        }
    };

    let mut follow = move |e: &Event<PointerData>| {
        if nav.advance(e) {
            return;
        }
        let pc = to_canvas(e);
        let Some(d) = drag() else {
            // Resting: report what a press here would do, for the cursor.
            hover.set(classify(pc).1);
            return;
        };
        // The current state, for the validity clamps to hold at.
        let current = (*state.transform.peek()).unwrap_or(ui);
        match d {
            Drag::Affine {
                region,
                from,
                rect,
                start,
            } => {
                let ts = match region {
                    TransformRegion::Inside => start.translated(from, pc, snap),
                    TransformRegion::Rim => start.turned_scaled(from, pc, snap),
                    TransformRegion::Outside => start.stretched(from, pc, snap),
                };
                update(state, TransformUi::Affine { rect, ts });
            }
            Drag::Quad {
                region,
                from,
                start,
            } => {
                let cur = match current {
                    TransformUi::Perspective(p) => p,
                    _ => start,
                };
                let delta = pc - from;
                let next = match region {
                    QuadRegion::Corner(i) => {
                        PerspectiveUi::corner_dragged(start, cur, i, delta, snap)
                    }
                    QuadRegion::Edge(a, b) => {
                        PerspectiveUi::edge_dragged(start, cur, (a, b), delta, snap)
                    }
                    QuadRegion::Inside | QuadRegion::Outside => {
                        PerspectiveUi::translated(start, cur, delta, snap)
                    }
                };
                update(state, TransformUi::Perspective(next));
            }
            Drag::Mesh {
                region,
                from,
                start,
                basis,
                norm,
            } => {
                let cur = match current {
                    TransformUi::Warp(w) => w,
                    _ => start,
                };
                let delta = pc - from;
                let next = match region {
                    MeshRegion::Point(i) => WarpUi::point_dragged(start, cur, i, delta, snap),
                    MeshRegion::Inside => {
                        WarpUi::surface_dragged(start, cur, &basis, norm, delta, snap)
                    }
                    MeshRegion::Outside => WarpUi::translated(start, cur, delta, snap),
                };
                update(state, TransformUi::Warp(next));
            }
        }
    };
    let mut finish = move |e: &Event<PointerData>| {
        follow(e);
        nav.stop();
        drag.set(None);
    };

    // The cursor announces the region the press would grab; the pan class wins
    // while space is held (an inline cursor would override it, so none is set).
    let panning = (state.space_down)();
    let catcher_class = if panning {
        "transform-catcher pan"
    } else {
        "transform-catcher"
    };
    let cursor = match (panning, drag()) {
        (true, _) => "",
        (_, Some(_)) => "cursor: grabbing;",
        (_, None) => hover(),
    };

    rsx! {
        div {
            class: "{catcher_class}",
            style: "{cursor}",
            onpointerdown: move |e| {
                if nav.begin(&e) {
                    // A second finger turns the drag into navigation (§18.1.7).
                    // The preview it had built stands — a transform commits on
                    // "Done", not on release, so letting go to look around costs
                    // nothing.
                    drag.set(None);
                    return;
                }
                if e.trigger_button() != Some(MouseButton::Primary) {
                    return;
                }
                e.stop_propagation();
                crate::platform::capture_pointer(&e);
                let pc = to_canvas(&e);
                drag.set(Some(classify(pc).0));
            },
            onpointermove: move |e| follow(&e),
            // Fingers still on the glass mean the gesture is not over — see the
            // canvas's own release handler.
            onpointerup: move |e| if !nav.release(&e) { finish(&e) },
            onpointercancel: move |e| if !nav.release(&e) { nav.stop(); drag.set(None); },
            onwheel: move |e| nav.wheel(e),
        }

        {match ui {
            TransformUi::Affine { ts, .. } => affine_ellipse(state, ts, view),
            TransformUi::Perspective(p) => quad_overlay(state, p, view),
            TransformUi::Warp(w) => mesh_overlay(state, w, view),
        }}
    }
}

/// The affine family's widget: a circle of the reference radius, deformed by
/// the linear map via CSS about its centre — the same composition the affine
/// applies to the paint, so the widget and the preview cannot disagree. It
/// stays a circle exactly as long as the transform is a similarity;
/// eccentricity *is* the distortion.
///
/// Plain functions rather than `#[component]`s, all three: they are chosen by
/// a `match` (a component would be conditionally-mounted hooks) and their
/// inputs are the parent's already-read state, not props to diff.
fn affine_ellipse(state: AppState, ts: TransformState, view: ViewTransform) -> Element {
    let cs = view.canvas_to_screen(ts.center);
    let r = ts.radius * view.zoom;
    // The gesture's linear map is canvas-space; what CSS draws is on screen, so the
    // view's own orientation composes onto it. The zoom is already in `r`, and the
    // orientation carries no scale, so this is the whole difference — a turned or
    // mirrored canvas turns and mirrors the widget with the paint it stands for.
    let l = view.orientation() * ts.linear;
    let ellipse_style = format!(
        "left: {}px; top: {}px; width: {}px; height: {}px; \
         transform: matrix({}, {}, {}, {}, 0, 0);",
        cs.x - r,
        cs.y - r,
        2.0 * r,
        2.0 * r,
        l.x_axis.x,
        l.x_axis.y,
        l.y_axis.x,
        l.y_axis.y,
    );
    rsx! {
        div {
            class: chrome_class(state, "transform-ellipse"),
            style: "{ellipse_style}",
            // The north dot rides the same CSS transform as its parent, so it
            // marks the reference ellipse's "up" wherever the deformation has
            // carried it — without it a rotated circle looks unrotated.
            div { class: "transform-north" }
        }
    }
}

/// One SVG polyline through screen-space points.
fn polyline(points: impl Iterator<Item = Vec2>) -> String {
    let mut d = String::new();
    for (i, p) in points.enumerate() {
        d.push(if i == 0 { 'M' } else { 'L' });
        d.push_str(&format!("{:.2} {:.2} ", p.x, p.y));
    }
    d
}

/// The perspective family's widget: the quad, the receding grid inside it
/// (the images of the source rect's thirds — straight under a homography, so
/// two endpoints each), and the four corner handles.
fn quad_overlay(state: AppState, p: PerspectiveUi, view: ViewTransform) -> Element {
    let to_screen = move |c: Vec2| view.canvas_to_screen(c);
    let (w, h) = (view.viewport.width, view.viewport.height);

    // The quad boundary, in corner order 0 → 1 → 3 → 2.
    let b = [p.corners[0], p.corners[1], p.corners[3], p.corners[2]];
    let outline = polyline(b.iter().copied().chain([b[0]]).map(to_screen)) + "Z";

    // The receding grid: the images of the rect's thirds. Under a homography a
    // line stays a line, so endpoints on opposite edges suffice — and because
    // the map is exact, the grid converging toward its vanishing points is not
    // an illustration of the transform; it is the transform.
    let mut grid = String::new();
    if let Some(f) = p.map().forward() {
        let (lo, hi) = p.rect;
        for i in 1..3 {
            let t = i as f32 / 3.0;
            let x = lo.x + (hi.x - lo.x) * t;
            let y = lo.y + (hi.y - lo.y) * t;
            grid += &polyline(
                [Vec2::new(x, lo.y), Vec2::new(x, hi.y)]
                    .into_iter()
                    .map(|c| to_screen(f.apply(c))),
            );
            grid += &polyline(
                [Vec2::new(lo.x, y), Vec2::new(hi.x, y)]
                    .into_iter()
                    .map(|c| to_screen(f.apply(c))),
            );
        }
    }

    rsx! {
        svg {
            class: chrome_class(state, "transform-svg"),
            width: "{w}",
            height: "{h}",
            view_box: "0 0 {w} {h}",
            path { class: "transform-grid", d: "{grid}" }
            path { class: "transform-outline", d: "{outline}" }
            for c in p.corners.iter() {
                circle {
                    class: "transform-handle",
                    cx: "{to_screen(*c).x}",
                    cy: "{to_screen(*c).y}",
                    r: "5",
                }
            }
        }
    }
}

/// The warp family's widget: the mesh curves sampled from the engine's own
/// smooth surface — the very map the paint resamples through — plus the 16
/// control points. A straight grid says "untouched"; every bend in the drawn
/// curves is a bend the paint has taken.
fn mesh_overlay(state: AppState, w: WarpUi, view: ViewTransform) -> Element {
    let to_screen = move |c: Vec2| view.canvas_to_screen(c);
    let (vw, vh) = (view.viewport.width, view.viewport.height);
    let map = w.map();

    // One curve per control row and column, sampled densely enough that the
    // cubic reads as a curve at any deformation.
    const SAMPLES: usize = 24;
    let mut lines = String::new();
    // Once for the whole overlay rather than once per sample: this is
    // `WARP_GRID * 2 * (SAMPLES + 1)` evaluations — 400 at an 8-wide grid — and
    // it redraws every frame of a drag.
    let surface = map.prepared();
    for k in 0..WARP_GRID {
        let t = k as f32 / (WARP_GRID - 1) as f32;
        lines += &polyline((0..=SAMPLES).map(|s| {
            let u = s as f32 / SAMPLES as f32;
            to_screen(surface.eval(Vec2::new(u, t)))
        }));
        lines += &polyline((0..=SAMPLES).map(|s| {
            let u = s as f32 / SAMPLES as f32;
            to_screen(surface.eval(Vec2::new(t, u)))
        }));
    }

    rsx! {
        svg {
            class: chrome_class(state, "transform-svg"),
            width: "{vw}",
            height: "{vh}",
            view_box: "0 0 {vw} {vh}",
            path { class: "transform-mesh", d: "{lines}" }
            for pt in w.points.iter() {
                circle {
                    class: "transform-handle",
                    cx: "{to_screen(*pt).x}",
                    cy: "{to_screen(*pt).y}",
                    r: "5",
                }
            }
        }
    }
}
