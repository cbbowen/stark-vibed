//! Floating-panel layout: identity, order, visibility, the title-bar drag, and the
//! fade that gets all the floating chrome out of the way mid-gesture
//! (DESIGN.md §11).
//!
//! The drag math deliberately never reads the layout it is mutating: panel
//! positions are measured once at drag start and everything after is derived from
//! the live pointer, so a sliding neighbour cannot feed back into the decision that
//! moved it.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use dioxus::prelude::*;

use crate::panels::{BrushPanel, ColorPanel, LayerPanel, LightingPanel, SelectPanel};
use crate::platform::sleep_ms;
use crate::state::AppState;

/// Identity of a floating tool panel. The set is fixed; `PanelLayout` tracks their
/// order and which are open (DESIGN.md §11).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum PanelId {
    Color,
    Brush,
    Select,
    Lighting,
    Layers,
}

impl PanelId {
    /// Every panel, in the default top-to-bottom order.
    pub const ALL: [PanelId; 5] = [
        PanelId::Color,
        PanelId::Brush,
        PanelId::Select,
        PanelId::Lighting,
        PanelId::Layers,
    ];

    /// The panel's title-bar label.
    pub fn title(self) -> &'static str {
        match self {
            PanelId::Color => "Color",
            PanelId::Brush => "Brush",
            PanelId::Select => "Select",
            PanelId::Lighting => "Lighting",
            PanelId::Layers => "Layers",
        }
    }
}

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
                            PanelId::Color => rsx! { ColorPanel {} },
                            PanelId::Brush => rsx! { BrushPanel {} },
                            PanelId::Select => rsx! { SelectPanel {} },
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
#[component]
pub fn Panel(id: PanelId, children: Element) -> Element {
    let layout = use_context::<PanelLayout>();
    let drag = (layout.drag)();
    let dragging = drag.as_ref().is_some_and(|d| d.id == id);
    let dy = drag.as_ref().map_or(0.0, |d| d.offset(id));
    let class = if dragging { "panel dragging" } else { "panel" };
    let style = match &drag {
        None => String::new(),
        Some(d) => {
            // Track the pointer 1:1 only while actively dragging this panel; the sliding
            // neighbours — and the dragged panel as it settles on release — transition.
            let tracking = d.id == id && d.release.is_none();
            let trans = if tracking {
                "none"
            } else {
                "transform 180ms ease"
            };
            format!("transform: translateY({dy}px); transition: {trans};")
        }
    };
    rsx! {
        div {
            class,
            style,
            onmounted: move |e| {
                let mut refs = layout.refs;
                refs.write().insert(id, e.data());
            },
            div { class: "panel-header",
                div {
                    class: "panel-title",
                    onpointerdown: move |e| start_drag(layout, id, &e),
                    "{id.title()}"
                }
                button {
                    class: "panel-close",
                    title: "Close panel",
                    onclick: move |_| {
                        let mut hidden = layout.hidden;
                        hidden.write().insert(id);
                    },
                    "\u{2715}"
                }
            }
            {children}
        }
    }
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
