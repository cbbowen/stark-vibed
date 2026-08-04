//! The frame: its bottom bar, its "add" button, and its on-canvas handles
//! (§15.7).
//!
//! A frame is a **matte layer** — a region filled with a flat colour — so almost
//! everything about it is already the Layers panel's job: visibility, opacity (the
//! crop scrim), reordering, delete. What is left is the part that is about
//! *composition*: sizing it to an aspect, fitting it to what you have painted, and
//! dragging its edges.
//!
//! That part has no permanent panel. It lives in a bar mounted only while a frame
//! is **selected**, on the same argument the selection bar is made this way
//! (§6.8): controls meaningless without a frame should be absent rather
//! than greyed out, and a bar that is simply present or absent says "you are
//! composing" more directly than a mode indicator would. Creating a frame is a
//! button in the Layers panel, because a frame *is* a layer.
//!
//! The frame **clips nothing**. Paint runs past it and the matte covers the
//! overshoot, so re-cropping later is free — that is the whole reason the infinite
//! canvas earns its keep (§15.1).

use dioxus::prelude::*;

use crate::icons::{self, icon, label};
use crate::input::page_xy;
use crate::layout::chrome_class;
use crate::state::{AppState, dispatch};
use stark_core::command::{DocCommand, PeerCommand, ViewCommand};
use stark_core::document::MatteRegion;
use stark_core::geom::Vec2;
use stark_core::{LayerInfo, MatteInfo};

/// The frame's default fill: a near-black mat board. Dark reads as "not the
/// piece" against almost any painting, which is what a crop scrim is for.
const DEFAULT_MATTE: [f32; 3] = [0.06, 0.06, 0.07];

/// Aspect presets, as width:height.
const ASPECTS: [(&str, f32); 4] = [
    ("1:1", 1.0),
    ("4:5", 0.8),
    ("3:2", 1.5),
    ("16:9", 16.0 / 9.0),
];

/// What the aspect drop-down shows when the frame matches no preset — a real
/// state, since dragging a handle lands on an arbitrary ratio and the control
/// should say so rather than lie about the nearest preset.
const CUSTOM: &str = "Custom";

/// The preset this frame's ratio matches, if any. Tolerance is relative, so it
/// holds at any size; loose enough that a handle dragged to visually 16:9 reads as
/// 16:9 rather than flicking to "Custom" on a sub-pixel difference.
fn matched_aspect(matte: MatteInfo) -> &'static str {
    let (w, h) = (matte.width(), matte.height());
    if h.abs() < 1e-3 {
        return CUSTOM;
    }
    let ratio = w / h;
    ASPECTS
        .iter()
        .find(|(_, a)| (ratio - a).abs() <= a * 0.005)
        .map_or(CUSTOM, |(label, _)| label)
}

/// The frame being composed, if the **selected layer** is one.
///
/// There is deliberately no separate frame-selection state. `active_layer` is the
/// selected layer and a matte may be it (§15.7), so the frame bar and
/// the on-canvas handles key off the same thing the Layers panel highlights. That
/// is what makes exactly one row highlighted at a time a *consequence* rather than
/// a rule to enforce — and it means a frame that is removed, undone, or replaced
/// by a document load stops being composed with no invalidation to remember.
pub fn selected_frame(state: AppState) -> Option<(LayerInfo, MatteInfo)> {
    let obs = state.obs.read();
    let o = obs.as_ref()?;
    o.layers
        .iter()
        .find(|l| l.id == o.active_layer)
        .and_then(|l| l.matte.map(|m| (l.clone(), m)))
}

/// Stop composing: select the topmost paint layer instead. Used by the frame bar's
/// "Done", since a frame is only "deselected" by selecting something else.
pub fn done_composing(state: AppState) {
    let next = state.obs.read().as_ref().and_then(|o| {
        o.layers
            .iter()
            .rev()
            .find(|l| l.is_paintable())
            .map(|l| l.id)
    });
    if let Some(id) = next {
        dispatch(state, PeerCommand::SetActiveLayer(id));
    }
}

/// The canvas-space rect a new frame should take: the painted content if there is
/// any, otherwise what the viewport currently shows. Both are "frame what I am
/// looking at", which is the only sensible default for an unbounded canvas.
fn default_rect(state: AppState) -> (Vec2, Vec2) {
    let obs = state.obs.read();
    let Some(o) = obs.as_ref() else {
        return (Vec2::new(-256.0, -256.0), Vec2::new(256.0, 256.0));
    };
    content_rect(o).unwrap_or_else(|| view_rect(o))
}

/// The painted content's canvas-space bounds, inset to the populated tiles.
pub(crate) fn content_rect(o: &stark_core::ObservableState) -> Option<(Vec2, Vec2)> {
    let (min, max) = o.bounds.tile_range()?;
    let t = stark_core::geom::TILE_SIZE as f32;
    Some((
        Vec2::new(min.x as f32 * t, min.y as f32 * t),
        Vec2::new((max.x + 1) as f32 * t, (max.y + 1) as f32 * t),
    ))
}

/// What the viewport currently shows, in canvas px, inset a little so a frame made
/// from it is visibly a frame rather than flush with the window edge.
pub(crate) fn view_rect(o: &stark_core::ObservableState) -> (Vec2, Vec2) {
    // The canvas-space bound of what is on screen — which under a turned canvas
    // covers a little more than the window really shows, and that is the right way
    // round for "frame what I am looking at".
    let (min, max) = o.view.visible_bounds();
    let inset = (max - min) * 0.06;
    (min + inset, max - inset)
}

/// Reshape `rect` to `aspect` about its centre, preserving its area so switching
/// presets neither grows nor shrinks the piece.
fn to_aspect(min: Vec2, max: Vec2, aspect: f32) -> (Vec2, Vec2) {
    let center = (min + max) * 0.5;
    let area = ((max.x - min.x) * (max.y - min.y)).max(1.0);
    let h = (area / aspect).sqrt();
    let half = Vec2::new(aspect * h, h) * 0.5;
    (center - half, center + half)
}

/// The "+ Frame" button, for the Layers panel's header — a frame *is* a layer, so
/// that is where making one belongs. The new frame is selected immediately, so its
/// bar and handles come up without a second click.
#[component]
pub fn AddFrameButton() -> Element {
    let state = use_context::<AppState>();
    rsx! {
        button {
            class: "layer-add",
            title: "Add a frame: it marks what the piece is, and clips nothing — \
                    paint past it and re-crop whenever you like",
            onclick: move |_| {
                let (min, max) = default_rect(state);
                dispatch(state, DocCommand::AddMatte {
                    carrier: None,
                    above: None,
                    region: MatteRegion::OutsideRect { min, max },
                    color: DEFAULT_MATTE,
                });
                // Select it, so its bar and handles come up without a second click.
                // `AddMatte` mints the id engine-side, so the new frame is the
                // topmost matte in the layer stack that came back.
                let new_id = state
                    .obs
                    .read()
                    .as_ref()
                    .and_then(|o| o.layers.iter().rev().find(|l| l.matte.is_some()).map(|l| l.id));
                if let Some(id) = new_id {
                    dispatch(state, PeerCommand::SetActiveLayer(id));
                }
            },
            // The glyph carries the plus the label used to, so the word is just the
            // noun: "+ Frame" beside a bordered-square-with-a-plus said it twice.
            {icon(icons::ADD_FRAME)}
            {label("Frame")}
        }
    }
}

/// The frame's composition controls, in a bar at the bottom of the screen. Mounted
/// only while a frame is selected — see the module docs for why that is the whole
/// interaction model rather than a panel with an empty state.
#[component]
pub fn FrameBar() -> Element {
    let state = use_context::<AppState>();
    let Some((info, matte)) = selected_frame(state) else {
        return rsx! {};
    };
    let (w, h) = (matte.width(), matte.height());
    let current_aspect = matched_aspect(matte);
    let set_rect = move |min: Vec2, max: Vec2| {
        dispatch(state, DocCommand::SetMatteRect(info.id, min, max));
    };

    rsx! {
        div { class: chrome_class(state, "frame-bar"),
            // The glyph rides the bar's *label*, not one of its buttons: no single
            // control here is "crop" — sizing to an aspect, fitting to the art and
            // fitting to the view are three ways of doing it — so what the mark
            // identifies is the bar, and through it the mode you are in.
            span { class: "bar-label",
                {icon(icons::FRAME)}
                {label("Frame")}
            }
            span { class: "frame-dim",
                "{w.round() as i64} \u{00D7} {h.round() as i64}"
                span { class: "frame-unit", " px" }
            }

            span { class: "bar-sep" }

            // Reads the frame's *current* ratio and reshapes to a chosen one, so it
            // is a state readout rather than a row of fire-and-forget buttons.
            select {
                class: "select frame-aspect",
                title: "Reshape the frame to this aspect, keeping its area",
                onchange: move |e| {
                    if let Some((_, a)) = ASPECTS.iter().find(|(l, _)| *l == e.value()) {
                        let (min, max) = to_aspect(matte.min, matte.max, *a);
                        set_rect(min, max);
                    }
                },
                // Only offered while it is what the frame actually is: picking
                // "Custom" could not mean anything, since there is no ratio to
                // reshape *to*.
                if current_aspect == CUSTOM {
                    option { value: CUSTOM, selected: true, "{CUSTOM}" }
                }
                for (label, _) in ASPECTS {
                    option { value: label, selected: current_aspect == label, "{label}" }
                }
            }

            span { class: "bar-sep" }

            button {
                class: "chip",
                title: "Fit the frame to everything painted so far",
                onclick: move |_| {
                    let rect = state.obs.read().as_ref().and_then(content_rect);
                    if let Some((min, max)) = rect { set_rect(min, max); }
                },
                "Fit to art"
            }
            button {
                class: "chip",
                title: "Fit the frame to the current view",
                onclick: move |_| {
                    let rect = state.obs.read().as_ref().map(view_rect);
                    if let Some((min, max)) = rect { set_rect(min, max); }
                },
                "Fit to view"
            }

            span { class: "bar-sep" }

            // Only the fill lives here — it is the one thing that is *about the
            // frame* rather than about a layer. Opacity (the crop scrim) and
            // removal are ordinary layer properties, so they belong to the Layers
            // panel's single set of controls for whatever is selected, and having
            // them in two places was the duplication this replaced.
            input {
                class: "frame-swatch",
                r#type: "color",
                title: "Frame colour",
                value: "{hex(matte.color)}",
                oninput: move |e| {
                    if let Some(rgb) = parse_hex(&e.value()) {
                        dispatch(state, DocCommand::SetMatteColor(info.id, rgb));
                    }
                },
            }

            span { class: "bar-sep" }

            button {
                class: "chip",
                title: "Stop composing and go back to painting (the frame stays)",
                onclick: move |_| done_composing(state),
                {icon(icons::DONE)}
                {label("Done")}
            }
        }
    }
}

/// sRGB triple → `#rrggbb`, for the colour input.
fn hex(c: [f32; 3]) -> String {
    let b = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("#{:02x}{:02x}{:02x}", b(c[0]), b(c[1]), b(c[2]))
}

/// `#rrggbb` → sRGB triple.
fn parse_hex(s: &str) -> Option<[f32; 3]> {
    let s = s.strip_prefix('#')?;
    if s.len() != 6 {
        return None;
    }
    let c = |i: usize| {
        u8::from_str_radix(&s[i..i + 2], 16)
            .ok()
            .map(|v| v as f32 / 255.0)
    };
    Some([c(0)?, c(2)?, c(4)?])
}

// --- on-canvas handles ---------------------------------------------------

/// Which part of the frame a drag has hold of. The eight edge/corner handles plus
/// the frame itself (moved by its edges, never its interior — the interior is
/// where you paint).
#[derive(Clone, Copy, PartialEq)]
pub enum Grip {
    N,
    S,
    E,
    W,
    Nw,
    Ne,
    Sw,
    Se,
    Move,
}

impl Grip {
    /// `(class suffix, cursor)`. Placement itself lives in the stylesheet — edges
    /// are thin bars, corners are squares, all positioned against the frame box's
    /// own edges, so nothing here needs measuring.
    pub(crate) fn spec(self) -> (&'static str, &'static str) {
        match self {
            Grip::N => ("n", "n-resize"),
            Grip::S => ("s", "s-resize"),
            Grip::E => ("e", "e-resize"),
            Grip::W => ("w", "w-resize"),
            Grip::Nw => ("nw", "nw-resize"),
            Grip::Ne => ("ne", "ne-resize"),
            Grip::Sw => ("sw", "sw-resize"),
            Grip::Se => ("se", "se-resize"),
            Grip::Move => ("move", "move"),
        }
    }

    /// Every grip, including the move pill. `Move` is a small handle sitting
    /// *outside* the top edge rather than the frame's interior: dragging the
    /// inside is what every other crop tool uses to move a frame, and it is the
    /// one thing this frame cannot borrow, because the inside is where you paint.
    pub(crate) const ALL: [Grip; 9] = [
        Grip::N,
        Grip::S,
        Grip::E,
        Grip::W,
        Grip::Nw,
        Grip::Ne,
        Grip::Sw,
        Grip::Se,
        Grip::Move,
    ];

    /// Apply a canvas-space pointer delta to the rect this grip was pressed on.
    /// Edges move one side, corners two, `Move` translates. The rect is normalized
    /// afterwards, so dragging a side past its opposite flips rather than inverts.
    pub(crate) fn apply(self, (min, max): (Vec2, Vec2), d: Vec2) -> (Vec2, Vec2) {
        let (mut lo, mut hi) = (min, max);
        let (north, south) = (
            matches!(self, Grip::N | Grip::Nw | Grip::Ne),
            matches!(self, Grip::S | Grip::Sw | Grip::Se),
        );
        let (west, east) = (
            matches!(self, Grip::W | Grip::Nw | Grip::Sw),
            matches!(self, Grip::E | Grip::Ne | Grip::Se),
        );
        if north {
            lo.y += d.y;
        }
        if south {
            hi.y += d.y;
        }
        if west {
            lo.x += d.x;
        }
        if east {
            hi.x += d.x;
        }
        if self == Grip::Move {
            lo += d;
            hi += d;
        }
        (
            Vec2::new(lo.x.min(hi.x), lo.y.min(hi.y)),
            Vec2::new(lo.x.max(hi.x), lo.y.max(hi.y)),
        )
    }
}

/// An in-flight handle drag: which grip, where the pointer went down (screen px),
/// and the rect as it was when it did. Deltas are taken from the *start* rect
/// rather than accumulated, so rounding cannot drift over a long drag.
#[derive(Clone, Copy)]
pub struct FrameDrag {
    grip: Grip,
    origin: Vec2,
    start: (Vec2, Vec2),
}

/// The frame's edges and handles, drawn over the canvas (§15.7).
///
/// Mounted only while a frame is **selected**, together with [`FrameBar`] — one
/// state drives both, so the bar being up is exactly the promise that the handles
/// are live. Handles have no business sitting over the painting the rest of the
/// time.
///
/// The interior is deliberately **not** interactive (`pointer-events: none` on the
/// box, `auto` only on the handles): the inside of the frame is exactly where you
/// paint, so it must pass every pointer event straight through to the canvas. The
/// frame is moved by the pill above its top edge.
#[component]
pub fn FrameOverlay() -> Element {
    let state = use_context::<AppState>();
    let mut drag = use_signal(|| None::<FrameDrag>);

    let Some((info, matte)) = selected_frame(state) else {
        return rsx! {};
    };
    let view = match state.obs.read().as_ref() {
        Some(o) => o.view,
        None => return rsx! {},
    };

    // The box is laid out around the frame's *centre* and then turned, rather than
    // pinned by its top-left corner: the canvas can be rotated and mirrored
    // (§18.1.2), and a rect described by two opposite corners only
    // stays a rect while it is axis-aligned. Handles are placed as percentages of the
    // box, so they ride the turn with it and nothing else needs measuring.
    let center = view.canvas_to_screen((matte.min + matte.max) * 0.5);
    let (w, h) = (
        (matte.max.x - matte.min.x) * view.zoom,
        (matte.max.y - matte.min.y) * view.zoom,
    );
    // `transform-origin` is the box's centre by default, which is exactly the pivot
    // the canvas turns about. A mirrored view has determinant −1 here, so the grips
    // mirror too — the north-west handle really is at the north-west of the picture.
    let o = view.orientation();
    let box_style = format!(
        "left: {}px; top: {}px; width: {}px; height: {}px;          transform: matrix({}, {}, {}, {}, 0, 0);",
        center.x - 0.5 * w,
        center.y - 0.5 * h,
        w,
        h,
        o.x_axis.x,
        o.x_axis.y,
        o.y_axis.x,
        o.y_axis.y,
    );

    // A pointer delta in screen px, carried back through the whole view — a turned
    // canvas sends a drag to the right somewhere other than +x.
    let to_canvas = move |screen: Vec2, origin: Vec2| view.canvas_delta(screen - origin);

    rsx! {
        div {
            class: chrome_class(state, "frame-overlay"),
            style: "{box_style}",

            for grip in Grip::ALL {
                {
                    let (suffix, cursor) = grip.spec();
                    rsx! {
                        div {
                            key: "{suffix}",
                            class: "frame-grip frame-grip-{suffix}",
                            style: "cursor: {cursor};",
                            onpointerdown: move |e| {
                                // Never let a grip press reach the canvas beneath —
                                // it would start a stroke under the frame chrome.
                                e.stop_propagation();
                                crate::platform::capture_pointer(&e);
                                drag.set(Some(FrameDrag {
                                    grip,
                                    origin: page_xy(&e),
                                    start: (matte.min, matte.max),
                                }));
                            },
                            onpointermove: move |e| {
                                let Some(d) = drag() else { return };
                                let (min, max) =
                                    d.grip.apply(d.start, to_canvas(page_xy(&e), d.origin));
                                // Previewed, not committed: one undo step per drag,
                                // not one per pointer move (§15.7).
                                dispatch(
                                    state,
                                    ViewCommand::PreviewMatteRect(Some((info.id, min, max))),
                                );
                            },
                            onpointerup: move |e| {
                                let Some(d) = drag.take() else { return };
                                let (min, max) =
                                    d.grip.apply(d.start, to_canvas(page_xy(&e), d.origin));
                                dispatch(state, DocCommand::SetMatteRect(info.id, min, max));
                            },
                            onpointercancel: move |_| {
                                if drag.take().is_some() {
                                    dispatch(state, ViewCommand::PreviewMatteRect(None));
                                }
                            },
                        }
                    }
                }
            }
        }
    }
}
