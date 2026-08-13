//! Floating-panel layout: identity, order, visibility, the title-bar drag, and the
//! fade that gets all the floating chrome out of the way mid-gesture
//! (§11).
//!
//! The drag math deliberately never reads the layout it is mutating: panel
//! positions are measured once at drag start and everything after is derived from
//! the live pointer, so a sliding neighbour cannot feed back into the decision that
//! moved it.

use std::collections::{HashMap, HashSet};

use dioxus::prelude::*;

use crate::icons::{self, icon};
use crate::panels::{
    BrushPanel, ColorPanel, GradientsPanel, GuidesPanel, LayerPanel, LightingPanel, NavigatorPanel,
    SelectPanel,
};
use crate::platform;
use crate::state::AppState;

/// Identity of a floating tool panel. The set is fixed; `PanelLayout` tracks their
/// order and which are open (§11).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum PanelId {
    Navigator,
    Color,
    Brush,
    Select,
    Gradients,
    Layers,
    Guides,
    Lighting,
}

impl PanelId {
    /// Every panel, in the default top-to-bottom order. The navigator leads: it is
    /// the only one that says where you *are* rather than what the next stroke will
    /// be, and it is read at a glance rather than operated.
    pub const ALL: [PanelId; 8] = [
        PanelId::Navigator,
        PanelId::Color,
        PanelId::Brush,
        PanelId::Select,
        PanelId::Gradients,
        PanelId::Layers,
        PanelId::Guides,
        PanelId::Lighting,
    ];

    /// The panels that start closed: lighting is a scene-setup control, a guide
    /// is reached for when a drawing calls for one, and a gradient library is
    /// visited between passages rather than lived in — none is touched
    /// mid-painting, so all three stay out of the stack until asked for.
    pub const CLOSED_BY_DEFAULT: [PanelId; 3] =
        [PanelId::Gradients, PanelId::Guides, PanelId::Lighting];

    /// The panel's title-bar label.
    pub fn title(self) -> &'static str {
        match self {
            PanelId::Navigator => "Navigator",
            PanelId::Color => "Color",
            PanelId::Brush => "Brush",
            PanelId::Select => "Select",
            PanelId::Gradients => "Gradients",
            PanelId::Layers => "Layers",
            PanelId::Guides => "Drawing Guides",
            PanelId::Lighting => "Lighting",
        }
    }

    /// The panel's mark, worn by its title bar and by the entry that reopens it in
    /// the Panels menu.
    ///
    /// A stack of panels is read down its left edge, and a column of seven words is
    /// read one word at a time; a column of seven marks is read at once. That is the
    /// whole argument — the glyph is not decoration on the title, it is what makes
    /// the stack scannable at the glance the title bar is actually given.
    ///
    /// Three of these are shared with a bar or a dialog elsewhere (see
    /// [`crate::icons`]), on purpose: a panel and the bar that serves it are two
    /// views of one subject, so they are one mark.
    pub fn glyph(self) -> &'static str {
        match self {
            PanelId::Navigator => icons::NAVIGATOR,
            PanelId::Color => icons::COLOR,
            PanelId::Brush => icons::BRUSH,
            PanelId::Select => icons::SELECTION,
            PanelId::Gradients => icons::GRADIENT,
            PanelId::Layers => icons::LAYERS,
            PanelId::Guides => icons::PERSPECTIVE_GRID,
            PanelId::Lighting => icons::LIGHTING,
        }
    }

    /// The height a panel opens at, in px — and, by being `Some`, that it is
    /// **vertically resizable**: it gets a grip on its bottom edge and its content
    /// is laid out to fill whatever height the user leaves it at.
    ///
    /// One method rather than a `resizable()` flag beside a `default_height()`,
    /// because a panel that can be resized is exactly a panel whose height the
    /// layout owns; two sources for that would be one to get out of step.
    ///
    /// Everything else hugs its controls, which is the right answer for a fixed set
    /// of knobs — there is nothing to give the extra room to. Only a panel holding a
    /// list the user grows (Brush, via its presets) has an appetite for height.
    pub fn default_height(self) -> Option<f32> {
        match self {
            // Tall enough for the quick controls plus four or five presets — a library
            // worth scrolling rather than a slot — and no taller, because the panel
            // stack is a column and every pixel here is one the panels under it lose.
            PanelId::Brush => Some(340.0),
            // The other panel holding a list the user grows (§22.3): its header
            // plus four or five strips, scrolling past that.
            PanelId::Gradients => Some(280.0),
            _ => None,
        }
    }
}

/// The shortest a resizable panel may be dragged: enough for the Brush panel's fixed
/// controls plus a row of the list under them, so the grip cannot fold a panel into a
/// sliver that shows nothing and is hard to grab back.
const MIN_PANEL_HEIGHT: f32 = 260.0;

/// Shared `Copy` layout state for the floating panels: their display order, which are
/// hidden, and the two in-flight gestures. Closed panels stay in `order` (so reopening
/// restores their slot); the stack renders `order` minus `hidden`. Provided via context
/// to the panel chrome and the menu.
///
/// No panel geometry is kept here — a drag reads it off the DOM at the moment it starts
/// ([`platform::panel_boxes`]), so there is no cached measurement to fall out of date.
#[derive(Clone, Copy)]
pub struct PanelLayout {
    pub order: Signal<Vec<PanelId>>,
    pub hidden: Signal<HashSet<PanelId>>,
    pub drag: Signal<Option<DragState>>,
    /// The current height of each resizable panel ([`PanelId::default_height`]),
    /// seeded with those defaults by [`PanelLayout::default_heights`]. A panel that
    /// is not resizable never appears here and never gets a height at all.
    pub heights: Signal<HashMap<PanelId, f32>>,
    /// The in-flight bottom-edge resize, if any.
    pub resize: Signal<Option<ResizeState>>,
}

impl PanelLayout {
    /// The starting heights: every resizable panel at its default. Seeded up front
    /// rather than measured on first drag, so a resize is pure arithmetic on a number
    /// the layout already knows (and the grip responds on the first pixel of the
    /// drag, with no async measurement to wait for).
    pub fn default_heights() -> HashMap<PanelId, f32> {
        PanelId::ALL
            .iter()
            .filter_map(|id| id.default_height().map(|h| (*id, h)))
            .collect()
    }
}

/// An in-flight resize of a panel's bottom edge. Like [`DragState`], the height is
/// derived from where the pointer started rather than from the panel's live box, so
/// the element being resized can never feed back into the size being computed.
#[derive(Clone, Copy, PartialEq)]
pub struct ResizeState {
    id: PanelId,
    /// Pointer Y at grab (client px).
    anchor_y: f32,
    /// The panel's height at grab.
    start_h: f32,
}

/// An in-flight title-bar drag. `panels` is the visible panels' `(id, top, height)`
/// measured once at drag start (client px, top-to-bottom); everything else is derived
/// from the live pointer so the math never feeds back on the shifting layout.
///
/// This state exists **only while a pointer is down on a title bar** — [`drag_end`]
/// commits the reorder and clears it in the same write. It used to outlive the gesture by
/// 180ms so the dropped panel could ease into its slot, with a timer committing the order
/// once it landed; when that timer failed to fire the panels were left wearing the
/// preview's transforms over a stack that had never reordered, which is a whole panel of
/// displacement and no way back. A gesture the browser itself delimits cannot get stuck
/// that way, and that is worth more than the ease (§11).
#[derive(Clone, PartialEq)]
pub struct DragState {
    id: PanelId,
    from: usize,
    panels: Vec<(PanelId, f32, f32)>,
    height: f32,
    gap: f32,
    anchor_y: f32,
    pointer_y: f32,
}

impl DragState {
    /// How far a neighbour slides to open/close the dragged panel's slot: its full slot
    /// extent (height + one inter-panel gap).
    fn step(&self) -> f32 {
        self.height + self.gap
    }

    /// The dragged panel's current top / bottom Y (original edge + pointer delta).
    fn dragged_top(&self) -> f32 {
        self.panels[self.from].1 + (self.pointer_y - self.anchor_y)
    }
    fn dragged_bottom(&self) -> f32 {
        self.dragged_top() + self.height
    }

    /// The vertical offset to render panel `id` at. The dragged panel follows the pointer;
    /// the others slide by ±`step` to open the landing slot. A neighbour yields once the
    /// dragged panel's **leading edge** — its top going up, its bottom going down —
    /// crosses that neighbour's centre, so a panel can always be dragged all the way to
    /// the top or bottom.
    fn offset(&self, id: PanelId) -> f32 {
        if id == self.id {
            return self.pointer_y - self.anchor_y;
        }
        let Some(k) = self.panels.iter().position(|p| p.0 == id) else {
            return 0.0;
        };
        let center = self.panels[k].1 + self.panels[k].2 * 0.5;
        if k > self.from && self.dragged_bottom() > center {
            -self.step()
        } else if k < self.from && self.dragged_top() < center {
            self.step()
        } else {
            0.0
        }
    }

    /// Insertion index among the visible panels for the current pointer position — the
    /// count of neighbours that now sit above the dragged panel (leading-edge rule).
    fn insert_index(&self) -> usize {
        let (top, bottom) = (self.dragged_top(), self.dragged_bottom());
        self.panels
            .iter()
            .enumerate()
            .filter(|(k, p)| {
                if *k == self.from {
                    return false;
                }
                let center = p.1 + p.2 * 0.5;
                if *k < self.from {
                    top >= center
                } else {
                    bottom > center
                }
            })
            .count()
    }
}

/// The class list for a floating-chrome container (the panel stack, the command rail,
/// the selection bar): its own class plus `chrome`, and `dimmed` while a canvas gesture
/// is in flight.
///
/// Every one of them sits over the canvas and none of them is what the user is looking
/// at mid-stroke, so they all fade together — the screen goes back to being the
/// painting, and comes back the moment the gesture ends. The fade is CSS
/// ([`crate::state::AppState::canvas_active`] only toggles the class), so the chrome
/// stays laid out where it was and nothing reflows on the way in or out.
pub fn chrome_class(state: AppState, base: &str) -> String {
    if (state.canvas_active)() {
        format!("{base} chrome dimmed")
    } else {
        format!("{base} chrome")
    }
}

/// The floating tool panels, top-right. Renders the open panels in a **fixed** sequence —
/// [`PanelId::ALL`], always — and states the user's order as a flex `order` on each one.
///
/// So the stack's order is not in the DOM: slot *k* renders panel *k* for the life of the
/// app, and a reorder changes an integer rather than moving anything. What that buys is
/// that a panel keeps its own element and its own subtree — the navigator's canvas and the
/// layer list survive being dragged up the stack, where reordering the children would
/// rebuild whichever panels changed slots.
///
/// It costs the stack `:first-child` / `:last-child`, since the first child is now
/// whichever panel leads `PanelId::ALL` rather than the top of the column; `Panel` names
/// the ends with `.stack-first` / `.stack-last` instead.
///
/// A stack no taller than its panels, and a stack taller than the window scrolls: both
/// are the stylesheet's job alone (`.panel-stack`) — no wheel handler here, and no box
/// beyond the panels for a canvas press to disappear into. The canvas's zoom hangs off
/// the `<canvas>` element, which is this stack's *sibling*, so a wheel spent over it is
/// unable to reach it; adding a handler to suppress a zoom that cannot happen would be
/// a second, quieter claim about the DOM shape for the first one to fall out of step
/// with.
#[component]
pub fn PanelStack() -> Element {
    let layout = use_context::<PanelLayout>();
    let state = use_context::<AppState>();
    let hidden = (layout.hidden)();
    // The open panels top to bottom — the order the *user* sees, which each panel then
    // carries as its own `order` slot. Everything below iterates the constant instead.
    let visible: Vec<PanelId> = (layout.order)()
        .into_iter()
        .filter(|id| !hidden.contains(id))
        .collect();
    // Every panel closed is no stack at all, not an empty one. An empty stack is still a
    // box over the canvas — its padding alone is a strip across the top-right corner —
    // and chrome the user cannot see must not be able to take a press aimed at the
    // painting. The same reasoning the stylesheet applies to the stack's height.
    if visible.is_empty() {
        return rsx! {};
    }
    let count = visible.len();
    // The drag preview is resolved to a number here, alongside each panel's slot, so a
    // panel is handed its offset rather than reading the gesture itself. Only the panels
    // whose offset actually changed then re-render as the pointer moves.
    let drag = (layout.drag)();
    let live = drag.as_ref();
    rsx! {
        div { class: chrome_class(state, "panel-stack"),
            for id in PanelId::ALL {
                if let Some(slot) = visible.iter().position(|p| *p == id) {
                    Panel {
                        id,
                        slot,
                        count,
                        offset: live.map(|d| d.offset(id)),
                        dragging: live.is_some_and(|d| d.id == id),
                        match id {
                            PanelId::Navigator => rsx! { NavigatorPanel {} },
                            PanelId::Color => rsx! { ColorPanel {} },
                            PanelId::Brush => rsx! { BrushPanel {} },
                            PanelId::Select => rsx! { SelectPanel {} },
                            PanelId::Gradients => rsx! { GradientsPanel {} },
                            PanelId::Guides => rsx! { GuidesPanel {} },
                            PanelId::Lighting => rsx! { LightingPanel {} },
                            PanelId::Layers => rsx! { LayerPanel {} },
                        }
                    }
                }
            }
        }
    }
}

/// Unified panel chrome: a header (title = drag handle + close button) over the panel's
/// controls. The ✕ closes the panel (the "Panels" menu reopens it). During a drag the
/// dragged panel follows the pointer and the others slide to open its landing slot; the
/// slide transition is applied inline *only while dragging*, so on release every panel
/// snaps straight to the freshly-reordered layout with no transition glitch.
///
/// A panel with a [`PanelId::default_height`] also gets an explicit height and a grip on
/// its bottom edge. The height is set here rather than in the stylesheet because it is
/// live state; what the stylesheet owns is what the panel does with it — see
/// `.panel.resizable`, which turns the panel into a column so its list can take the slack.
///
/// `slot` is where this panel sits in the stack, `count` how many are open. The first is a
/// flex `order`, which is the whole of how the stack is ordered ([`PanelStack`]); the two
/// together name the ends of the column, which minimal mode rounds. Those have to be
/// classes rather than `:first-child` / `:last-child` — the DOM child order is the fixed
/// one now, so the first child is whichever panel happens to lead `PanelId::ALL`.
///
/// `offset` is the drag preview — `Some(dy)` while a drag is in flight, `None` at rest.
/// The panel is handed it rather than reading the gesture, so the offsets and the slots
/// are decided together in [`PanelStack`] and only the panels that actually move
/// re-render as the pointer travels.
#[component]
pub fn Panel(
    id: PanelId,
    slot: usize,
    count: usize,
    offset: Option<f32>,
    dragging: bool,
    children: Element,
) -> Element {
    let layout = use_context::<PanelLayout>();
    // `map` on the default, so a panel that is not resizable never reads either signal
    // and so never re-renders for someone else's resize.
    let height = id
        .default_height()
        .map(|d| layout.heights.read().get(&id).copied().unwrap_or(d));
    let resizing = height.is_some() && layout.resize.read().is_some_and(|r| r.id == id);

    let mut class = String::from("panel");
    if height.is_some() {
        class.push_str(" resizable");
    }
    if dragging {
        class.push_str(" dragging");
    }
    if resizing {
        class.push_str(" resizing");
    }
    if slot == 0 {
        class.push_str(" stack-first");
    }
    if slot + 1 == count {
        class.push_str(" stack-last");
    }

    // **Every declaration, every render, including the ones that are "off".**
    //
    // Inline styles are applied property by property, not by replacing the attribute, so a
    // declaration left out of this string is not removed from the element — it keeps
    // whatever value the last render that *did* mention it gave it. Omitting `transform`
    // when no drag is in flight therefore does not clear the drag's transform: it strands
    // it, over a stack that has since reordered, which is the whole of this bug
    // (`2026-08-03`). Writing the resting values explicitly is what clears them.
    //
    // The rule is easy to break by adding a conditional declaration here, and the breakage
    // is invisible until some *other* state change makes the stale value wrong.
    let h = match height {
        Some(h) => format!("{h}px"),
        None => "auto".to_string(),
    };
    // The panel under the pointer tracks it 1:1; the neighbours slide; at rest there is
    // nothing to animate, and the reorder lands in the same frame as the reset.
    let (dy, trans) = match offset {
        Some(dy) if dragging => (dy, "none"),
        Some(dy) => (dy, "transform 180ms ease"),
        None => (0.0, "none"),
    };
    let style =
        format!("order: {slot}; height: {h}; transform: translateY({dy}px); transition: {trans};");
    rsx! {
        div {
            class,
            style,
            // Which panel this element is, for `platform::panel_boxes` to read back. The
            // drag measures the DOM and writes `order`, so it needs the two to agree; this
            // is what lets it check rather than assume.
            "data-panel": "{id:?}",
            div { class: "panel-header",
                // The mark is inside the drag handle rather than beside it: the whole
                // title *is* the grip, and a glyph sitting outside it would be the one
                // part of the header that looks draggable and is not.
                div {
                    class: "panel-title",
                    // The grip captures the pointer and owns the whole gesture, as every
                    // other drag in the app does (`platform::capture_pointer`). Capture is
                    // what makes the release certain — it is delivered to the capturing
                    // element whatever the pointer is over by then, and this is a drag
                    // where the thing under the pointer moves as you drag it. Left to
                    // bubble to the app root instead, the release did not arrive at all.
                    onpointerdown: move |e| {
                        platform::capture_pointer(&e);
                        start_drag(layout, id, &e);
                    },
                    onpointermove: move |e| drag_move(layout, &e),
                    onpointerup: move |_| drag_end(layout),
                    onpointercancel: move |_| drag_end(layout),
                    {icon(id.glyph())}
                    "{id.title()}"
                }
                button {
                    class: "panel-close",
                    title: "Close panel",
                    onclick: move |_| {
                        let mut hidden = layout.hidden;
                        hidden.write().insert(id);
                    },
                    {icon(icons::CLOSE)}
                }
            }
            {children}
            if height.is_some() {
                // Sits in the panel's own bottom padding, so it costs no layout: the
                // grip is a place to press, and the bar it draws only appears once the
                // pointer is over the panel that owns it.
                div {
                    class: "panel-resize",
                    title: "Drag to resize",
                    onpointerdown: move |e| start_resize(layout, id, &e),
                }
            }
        }
    }
}

/// Begin resizing panel `id` from its bottom edge. Unlike a reorder drag there is
/// nothing to measure — the layout already holds the height — so the grip answers on
/// the first pixel of movement.
pub fn start_resize(layout: PanelLayout, id: PanelId, e: &Event<PointerData>) {
    let Some(default) = id.default_height() else {
        return;
    };
    let start_h = layout.heights.peek().get(&id).copied().unwrap_or(default);
    let mut resize = layout.resize;
    resize.set(Some(ResizeState {
        id,
        anchor_y: e.client_coordinates().y as f32,
        start_h,
    }));
}

/// Track the pointer for an in-flight panel resize (no-op when idle). Writes only the
/// resized panel's height, so nothing else in the stack re-renders as it grows.
pub fn resize_move(layout: PanelLayout, e: &Event<PointerData>) {
    let Some(r) = *layout.resize.peek() else {
        return;
    };
    let h = (r.start_h + (e.client_coordinates().y as f32 - r.anchor_y)).max(MIN_PANEL_HEIGHT);
    let mut heights = layout.heights;
    heights.write().insert(r.id, h);
}

/// End a panel resize. The height is already committed (every move wrote it), so this
/// only disarms — there is nothing to settle and nothing to undo.
pub fn resize_end(layout: PanelLayout) {
    if layout.resize.peek().is_none() {
        return;
    }
    let mut resize = layout.resize;
    resize.set(None);
}

/// Begin dragging panel `id` by its title bar: measure the visible panels and arm the
/// drag. The pointer tracking and the reorder follow in [`drag_move`] / [`drag_end`].
///
/// Every box comes from [`platform::panel_boxes`] and is matched to its panel **by the id
/// on the same element**, never by position — the stack's children are in a fixed sequence,
/// not the user's order ([`PanelStack`]).
///
/// Synchronous, so the drag is armed on the press rather than a JS round trip later —
/// which also means no pointer movement is dropped between the two.
pub fn start_drag(layout: PanelLayout, id: PanelId, e: &Event<PointerData>) {
    let hidden = layout.hidden.peek().clone();
    let visible: Vec<PanelId> = layout
        .order
        .peek()
        .iter()
        .copied()
        .filter(|p| !hidden.contains(p))
        .collect();
    let boxes = platform::panel_boxes();
    // Each panel's own box, found by the id it wears. `None` if the DOM has not caught up
    // with `order` — a press in the frame before a newly-opened panel has mounted, or
    // off-wasm, where there is no DOM to read at all.
    let Some(panels) = visible
        .iter()
        .map(|p| {
            let key = format!("{p:?}");
            boxes
                .iter()
                .find(|(k, ..)| *k == key)
                .map(|&(_, top, h)| (*p, top, h))
        })
        .collect::<Option<Vec<(PanelId, f32, f32)>>>()
    else {
        return;
    };
    let Some(from) = visible.iter().position(|p| *p == id) else {
        return;
    };
    let height = panels[from].2;
    // The inter-panel gap (so a slide closes the slot exactly): the space between the
    // first two panels, or 0 if there's only one (then nothing can reorder anyway).
    let gap = if panels.len() > 1 {
        (panels[1].1 - panels[0].1 - panels[0].2).max(0.0)
    } else {
        0.0
    };
    let anchor_y = e.client_coordinates().y as f32;
    let mut drag = layout.drag;
    drag.set(Some(DragState {
        id,
        from,
        panels,
        height,
        gap,
        anchor_y,
        pointer_y: anchor_y,
    }));
}

/// Track the pointer for an in-flight panel drag (no-op when idle — the check is what
/// keeps every pointer move over the app from dirtying the whole stack).
pub fn drag_move(layout: PanelLayout, e: &Event<PointerData>) {
    let armed = layout.drag.peek().is_some();
    if !armed {
        return;
    }
    let y = e.client_coordinates().y as f32;
    let mut drag = layout.drag;
    if let Some(d) = drag.write().as_mut() {
        d.pointer_y = y;
    }
}

/// End a panel drag: disarm, then write the panel's landing slot into `order`. No-op if
/// no drag is active.
///
/// **The disarm goes first.** A panel's offset is a transform stated against the layout as
/// it stood when the drag began, so a frame carrying the new `order` while the transforms
/// are still on would be the reorder applied twice — every affected panel a slot from where
/// it belongs. Cleared first, the worst an in-between frame can show is the stack exactly as
/// it was before the gesture: a glitch the next render corrects rather than a wrong layout.
/// Taking the state *out* of the signal is what disarms it, so that is one write and not
/// two things to keep in step.
///
/// Nothing is deferred here on purpose. The dragged panel used to ease into its slot over
/// 180ms with a timer committing the order afterwards — a settle whose timer never fired,
/// and a dropped panel is the one moment in the gesture where the user is looking at the
/// *slot*, not at the panel.
pub fn drag_end(layout: PanelLayout) {
    let mut drag = layout.drag;
    let taken = drag.write().take();
    let Some(d) = taken else {
        return;
    };
    let ins = d.insert_index();
    let hidden = layout.hidden.peek().clone();
    let mut order = layout.order;
    {
        let mut ord = order.write();
        ord.retain(|p| *p != d.id);
        // Insert before the visible panel currently at index `ins` (hidden panels keep
        // their slots), or at the end.
        let visible: Vec<usize> = ord
            .iter()
            .enumerate()
            .filter(|(_, p)| !hidden.contains(p))
            .map(|(i, _)| i)
            .collect();
        let at = visible.get(ins).copied().unwrap_or(ord.len());
        ord.insert(at, d.id);
    }
}
