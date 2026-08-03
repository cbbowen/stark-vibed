//! Floating-panel layout: identity, order, visibility, the title-bar drag, and the
//! fade that gets all the floating chrome out of the way mid-gesture
//! (§11).
//!
//! The drag math deliberately never reads the layout it is mutating: panel
//! positions are measured once at drag start and everything after is derived from
//! the live pointer, so a sliding neighbour cannot feed back into the decision that
//! moved it.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use dioxus::prelude::*;

use crate::icons::{self, icon};
use crate::panels::{
    BrushPanel, ColorPanel, GuidesPanel, LayerPanel, LightingPanel, NavigatorPanel, SelectPanel,
};
use crate::platform::sleep_ms;
use crate::state::AppState;

/// Identity of a floating tool panel. The set is fixed; `PanelLayout` tracks their
/// order and which are open (§11).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum PanelId {
    Navigator,
    Color,
    Brush,
    Select,
    Layers,
    Guides,
    Lighting,
}

impl PanelId {
    /// Every panel, in the default top-to-bottom order. The navigator leads: it is
    /// the only one that says where you *are* rather than what the next stroke will
    /// be, and it is read at a glance rather than operated.
    pub const ALL: [PanelId; 7] = [
        PanelId::Navigator,
        PanelId::Color,
        PanelId::Brush,
        PanelId::Select,
        PanelId::Layers,
        PanelId::Guides,
        PanelId::Lighting,
    ];

    /// The panels that start closed: lighting is a scene-setup control, and a guide
    /// is reached for when a drawing calls for one — neither is touched mid-painting,
    /// so both stay out of the stack until asked for.
    pub const CLOSED_BY_DEFAULT: [PanelId; 2] = [PanelId::Guides, PanelId::Lighting];

    /// The panel's title-bar label.
    pub fn title(self) -> &'static str {
        match self {
            PanelId::Navigator => "Navigator",
            PanelId::Color => "Color",
            PanelId::Brush => "Brush",
            PanelId::Select => "Select",
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
            PanelId::Layers => icons::LAYERS,
            PanelId::Guides => icons::PERSPECTIVE,
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
            _ => None,
        }
    }
}

/// The shortest a resizable panel may be dragged: enough for the Brush panel's fixed
/// controls plus a row of the list under them, so the grip cannot fold a panel into a
/// sliver that shows nothing and is hard to grab back.
const MIN_PANEL_HEIGHT: f32 = 260.0;

/// Shared `Copy` layout state for the floating panels: their display order, which are
/// hidden, the in-flight title-bar drag, and each panel's mounted node (for measuring).
/// Closed panels stay in `order` (so reopening restores their slot); the stack renders
/// `order` minus `hidden`. Provided via context to the panel chrome and the menu.
#[derive(Clone, Copy)]
pub struct PanelLayout {
    pub order: Signal<Vec<PanelId>>,
    pub hidden: Signal<HashSet<PanelId>>,
    pub drag: Signal<Option<DragState>>,
    /// Each visible panel's mounted node, so a drag can measure their positions.
    pub refs: Signal<HashMap<PanelId, Rc<MountedData>>>,
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
/// from the live pointer so the math never feeds back on the shifting layout. Once the
/// pointer is released, `release` holds the dragged panel's target offset and it settles
/// there (sliding back to 0 if nothing reordered) before the new order is committed.
#[derive(Clone, PartialEq)]
pub struct DragState {
    id: PanelId,
    from: usize,
    panels: Vec<(PanelId, f32, f32)>,
    height: f32,
    gap: f32,
    anchor_y: f32,
    pointer_y: f32,
    release: Option<f32>,
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

    /// The vertical offset to render panel `id` at. The dragged panel follows the pointer
    /// (or eases to its settle target on release); the others slide by ±`step` to open
    /// the landing slot. A neighbour yields once the dragged panel's **leading edge** —
    /// its top going up, its bottom going down — crosses that neighbour's centre, so a
    /// panel can always be dragged all the way to the top or bottom.
    fn offset(&self, id: PanelId) -> f32 {
        if id == self.id {
            return self.release.unwrap_or(self.pointer_y - self.anchor_y);
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

    /// The dragged panel's offset from its original slot to its final slot (0 if the
    /// order is unchanged), so it can ease into place on release. Sums the slot extents
    /// of the panels it jumps over — using their own heights, since they need not match.
    fn target_offset(&self) -> f32 {
        let ins = self.insert_index();
        if ins == self.from {
            return 0.0;
        }
        let others: Vec<f32> = self
            .panels
            .iter()
            .filter(|p| p.0 != self.id)
            .map(|p| p.2 + self.gap)
            .collect();
        let sum = |r: std::ops::Range<usize>| others[r].iter().sum::<f32>();
        sum(0..ins) - sum(0..self.from)
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

/// The floating tool panels, top-right. Data-driven: renders `layout.order` minus the
/// hidden set, each wrapped in the unified [`Panel`] chrome (keyed by id so reordering
/// moves nodes rather than recreating them — preserves per-panel state and, later,
/// enables the drag animation).
///
/// A stack taller than the window scrolls, and that is the stylesheet's job alone
/// (`.panel-stack`) — no wheel handler here. The canvas's zoom hangs off the `<canvas>`
/// element, which is this stack's *sibling*, so a wheel spent over a panel is already
/// unable to reach it; adding a handler to suppress a zoom that cannot happen would be
/// a second, quieter claim about the DOM shape for the first one to fall out of step
/// with.
#[component]
pub fn PanelStack() -> Element {
    let layout = use_context::<PanelLayout>();
    let state = use_context::<AppState>();
    let hidden = (layout.hidden)();
    rsx! {
        div { class: chrome_class(state, "panel-stack"),
            for id in (layout.order)() {
                if !hidden.contains(&id) {
                    Panel { key: "{id:?}", id,
                        match id {
                            PanelId::Navigator => rsx! { NavigatorPanel {} },
                            PanelId::Color => rsx! { ColorPanel {} },
                            PanelId::Brush => rsx! { BrushPanel {} },
                            PanelId::Select => rsx! { SelectPanel {} },
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
#[component]
pub fn Panel(id: PanelId, children: Element) -> Element {
    let layout = use_context::<PanelLayout>();
    let drag = (layout.drag)();
    let dragging = drag.as_ref().is_some_and(|d| d.id == id);
    let dy = drag.as_ref().map_or(0.0, |d| d.offset(id));
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

    let mut style = String::new();
    if let Some(h) = height {
        style.push_str(&format!("height: {h}px;"));
    }
    if let Some(d) = &drag {
        // Track the pointer 1:1 only while actively dragging this panel; the sliding
        // neighbours — and the dragged panel as it settles on release — transition.
        let tracking = d.id == id && d.release.is_none();
        let trans = if tracking {
            "none"
        } else {
            "transform 180ms ease"
        };
        style.push_str(&format!(
            "transform: translateY({dy}px); transition: {trans};"
        ));
    }
    rsx! {
        div {
            class,
            style,
            onmounted: move |e| {
                let mut refs = layout.refs;
                refs.write().insert(id, e.data());
            },
            div { class: "panel-header",
                // The mark is inside the drag handle rather than beside it: the whole
                // title *is* the grip, and a glyph sitting outside it would be the one
                // part of the header that looks draggable and is not.
                div {
                    class: "panel-title",
                    onpointerdown: move |e| start_drag(layout, id, &e),
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

/// Begin dragging panel `id` by its title bar. Measures the visible panels' positions
/// (async, via their mounted nodes) and arms the drag; the actual pointer tracking +
/// reorder happen in [`drag_move`] / [`drag_end`] at the app root.
pub fn start_drag(layout: PanelLayout, id: PanelId, e: &Event<PointerData>) {
    let anchor_y = e.client_coordinates().y as f32;
    let order = layout.order.peek().clone();
    let hidden = layout.hidden.peek().clone();
    let refs = layout.refs.peek().clone();
    let mounted: Vec<(PanelId, Rc<MountedData>)> = order
        .into_iter()
        .filter(|p| !hidden.contains(p))
        .filter_map(|p| refs.get(&p).map(|m| (p, m.clone())))
        .collect();
    let mut drag = layout.drag;
    spawn(async move {
        let mut panels = Vec::with_capacity(mounted.len());
        for (pid, m) in &mounted {
            if let Ok(rect) = m.get_client_rect().await {
                panels.push((*pid, rect.origin.y as f32, rect.size.height as f32));
            }
        }
        let Some(from) = panels.iter().position(|p| p.0 == id) else {
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
        drag.set(Some(DragState {
            id,
            from,
            panels,
            height,
            gap,
            anchor_y,
            pointer_y: anchor_y,
            release: None,
        }));
    });
}

/// Track the pointer for an in-flight panel drag (no-op when idle or already settling).
pub fn drag_move(layout: PanelLayout, e: &Event<PointerData>) {
    if !matches!(layout.drag.peek().as_ref(), Some(d) if d.release.is_none()) {
        return;
    }
    let y = e.client_coordinates().y as f32;
    let mut drag = layout.drag;
    if let Some(d) = drag.write().as_mut() {
        d.pointer_y = y;
    }
}

/// Release a panel drag: enter the settle state (the dragged panel eases to its final
/// slot — back to 0 if nothing reordered), then commit the new order once it lands.
/// No-op if no drag is active or one is already settling.
pub fn drag_end(layout: PanelLayout) {
    let target = match layout.drag.peek().as_ref() {
        Some(d) if d.release.is_none() => d.target_offset(),
        _ => return,
    };
    let mut drag = layout.drag;
    if let Some(d) = drag.write().as_mut() {
        d.release = Some(target);
    }
    spawn(async move {
        sleep_ms(180).await;
        commit_drag(layout);
    });
}

/// Commit a settled drag: write the new order and disarm. Skips if a fresh drag has
/// replaced the settling one in the meantime.
pub fn commit_drag(layout: PanelLayout) {
    let Some(d) = layout.drag.peek().clone() else {
        return;
    };
    if d.release.is_none() {
        return; // a new drag started during the settle — leave it be
    }
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
    let mut drag = layout.drag;
    drag.set(None);
}
