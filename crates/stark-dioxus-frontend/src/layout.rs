//! Floating-panel layout: identity, order, visibility, the title-bar drag, and the
//! fade that gets the floating chrome out of the way mid-gesture (§11).
//!
//! The title-bar drag is the row drag the layer tree and the guide list use
//! (`panels::reorder`), over a flat list. Panel boxes are measured once at the press and
//! everything after derives from the pointer, so a sliding neighbour cannot feed back
//! into the move ([`Grab`]).
//!
//! # Which panels are open follows the browser
//!
//! Every panel starts closed, and what is open is kept per browser (`crate::visibility`,
//! §25.6); the tour keeps "none" from meaning "hidden" (§24.5). [`set_open`] is the only
//! writer of [`PanelLayout::hidden`] and persists on every change, so a new way to close
//! a panel is durable without thinking about storage.

use std::collections::{HashMap, HashSet};

use dioxus::prelude::*;

use crate::icons::icon;
use crate::panels::reorder::{self, RowKey};
use crate::panels::{BrushPanel, ColorPanel, GuidesPanel, LayerPanel, LightingPanel, SelectPanel};
use crate::platform;
use crate::state::{AppState, root_signal};
use stark_ui::panels::PanelId;
use stark_ui::reorder::{Grab, Motion, Slide};
use strum::VariantArray;

/// The panel's mark, worn by its title bar and its visibility-menu entry
/// (`stark_ui::commands::VisibilityToggle`). A free function for the orphan rule; which
/// mark it is belongs to the crate (§11.2 N8).
pub fn panel_glyph(id: PanelId) -> stark_ui::icons::Icon {
    id.glyph()
}

/// The shortest a resizable panel may be dragged.
const MIN_PANEL_HEIGHT: f32 = 140.0;

/// Shared `Copy` layout state for the floating panels. Closed panels keep their slot in
/// `order`; the stack renders `order` minus `hidden`. A field of [`AppState`], so registry
/// commands reach it.
///
/// No panel geometry is kept: a drag reads it off the DOM when it starts
/// ([`platform::panel_boxes`]).
#[derive(Clone, Copy)]
pub struct PanelLayout {
    pub order: Signal<Vec<PanelId>>,
    pub hidden: Signal<HashSet<PanelId>>,
    /// Which panels are **folded to their title bar** ([`toggle_collapse`]). Distinct from
    /// [`hidden`](Self::hidden): a folded panel is still in use, like a Brush panel whose
    /// quick slots write into it.
    pub collapsed: Signal<HashSet<PanelId>>,
    /// The in-flight title-bar drag — the shared row grab (`panels::reorder`).
    pub drag: Signal<Option<Grab>>,
    /// The current height of each resizable panel ([`PanelId::default_height`]), seeded by
    /// [`PanelLayout::default_heights`]. A panel that is not resizable never appears.
    pub heights: Signal<HashMap<PanelId, f32>>,
    /// The in-flight bottom-edge resize, if any.
    pub resize: Signal<Option<ResizeState>>,
    /// Where the stack is scrolled to ([`Scroll`]). The one cached measurement here: the
    /// browser moves it unasked, so it is re-read on `scroll`, on pointer entry, and after
    /// any render that could change the column's height ([`PanelStack`]).
    pub scroll: Signal<Scroll>,
    /// The in-flight drag of the rail's thumb, if any.
    pub thumb: Signal<Option<ThumbGrab>>,
}

impl PanelLayout {
    /// Its signals, root-owned (`state::root_signal`), in the state a stack opens with.
    pub(crate) fn new() -> Self {
        Self {
            order: root_signal(|| PanelId::VARIANTS.to_vec()),
            // Read before the first render, so the stack the artist left is the first one drawn
            // (`crate::visibility`, §25.6).
            hidden: root_signal(stark_ui::visibility::stored_hidden),
            collapsed: root_signal(stark_ui::visibility::stored_collapsed),
            drag: root_signal(|| None),
            heights: root_signal(PanelLayout::default_heights),
            resize: root_signal(|| None),
            scroll: root_signal(Default::default),
            thumb: root_signal(|| None),
        }
    }
}

/// The panel stack's scroll geometry, in CSS px. Everything derived from it is derived
/// here, so the thumb's drawing and its drag cannot disagree.
#[derive(Clone, Copy, PartialEq, Default, Debug)]
pub struct Scroll {
    /// `scrollTop`.
    pub top: f32,
    /// `scrollHeight` — the whole column, including what is off the ends.
    pub content: f32,
    /// `clientHeight` — the part of it on screen, which is also the rail's length.
    pub view: f32,
}

/// The shortest the rail's thumb may be drawn, in px, so it stays grabbable: a tablet has
/// no wheel.
const MIN_THUMB: f32 = 28.0;

impl Scroll {
    /// Whether there is anything to scroll; the rail is not in the DOM otherwise. A pixel of
    /// slack, so the browser's rounding of fractional heights cannot flicker the rail.
    pub fn overflows(self) -> bool {
        self.content > self.view + 1.0
    }

    /// The thumb as `(top, height)` in px down the rail: as tall a share of the rail
    /// as the view is of the content, at the place the scroll has reached.
    pub fn thumb(self) -> (f32, f32) {
        let height = (self.view / self.content * self.view).clamp(MIN_THUMB, self.view);
        // Zero-safe: the caller checked `overflows`, and the clamp above cannot exceed the rail.
        let travel = (self.view - height).max(0.0);
        let scrollable = (self.content - self.view).max(1.0);
        let top = (self.top / scrollable * travel).clamp(0.0, travel);
        (top, height)
    }

    /// Where a thumb dragged `dy` px from `start` would leave the stack: the inverse of
    /// [`thumb`](Self::thumb), so the thumb stays under the pointer.
    pub fn scrolled_by(self, start: f32, dy: f32) -> f32 {
        let (_, height) = self.thumb();
        let travel = (self.view - height).max(1.0);
        let scrollable = (self.content - self.view).max(0.0);
        (start + dy / travel * scrollable).clamp(0.0, scrollable)
    }
}

/// An in-flight drag of the rail's thumb, derived from where the pointer started so the
/// thumb being moved cannot feed back into the move.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ThumbGrab {
    /// Pointer Y at grab (client px).
    anchor_y: f32,
    /// `scrollTop` at grab.
    start_top: f32,
}

impl PanelLayout {
    /// Every resizable panel at its default height. Seeded rather than measured, so a resize
    /// answers on its first pixel.
    pub fn default_heights() -> HashMap<PanelId, f32> {
        PanelId::VARIANTS
            .iter()
            .filter_map(|id| id.default_height().map(|h| (*id, h)))
            .collect()
    }
}

/// An in-flight resize of a panel's bottom edge, derived from where the pointer started,
/// as [`ThumbGrab`] is.
#[derive(Clone, Copy, PartialEq)]
pub struct ResizeState {
    id: PanelId,
    /// Pointer Y at grab (client px).
    anchor_y: f32,
    /// The panel's height at grab.
    start_h: f32,
}

/// Where an in-flight drag would land: the slide, and how far the hand has taken it.
///
/// `None` when a panel has no box, which is what a stack that changed under the pointer
/// looks like; the stack then draws at rest and the release commits nothing
/// (`Grab::resolve`). The travelling block is always one panel, so there is no depth.
fn landing(visible: &[PanelId], drag: &Grab) -> Option<(Slide, f32)> {
    let keys: Vec<String> = visible.iter().copied().map(panel_key).collect();
    let (from, boxes) = drag.resolve(&keys)?;
    let (_, dy) = drag.delta();
    Some((Slide::resolve(&boxes, (from, from), dy)?, dy))
}

/// The identity a panel wears in the DOM and a drag resolves against: the `data-panel`
/// attribute, `reorder::RowKey::Panel`, [`landing`] and the tour's anchor
/// (`tutor::Anchor`). One function, because a box matched to the wrong panel fails
/// silently (§11).
pub fn panel_key(id: PanelId) -> String {
    format!("{id:?}")
}

/// Whether floating chrome (the panel stack, the command rail, the bars) is faded right
/// now: a canvas gesture is in flight and this browser fades
/// ([`ChromeHiding`](stark_ui::prefs::ChromeHiding)).
///
/// The fade is CSS, so nothing reflows. A container wears `chrome` and, under this,
/// `dimmed`. The gesture is read first, so idle chrome subscribes to nothing else.
pub fn chrome_dimmed(state: AppState) -> bool {
    (state.canvas_active)() && (state.chrome_hiding)().fades()
}

/// [`chrome_dimmed`], or the stack standing down after a gesture until the pointer
/// reaches its column ([`standing_down`]). One `dimmed` class for both.
fn stack_dimmed(state: AppState) -> bool {
    chrome_dimmed(state) || standing_down(state)
}

/// Whether the stack is asleep **and nothing is holding it up**
/// ([`Signals::panels_asleep`](crate::state::Signals::panels_asleep)).
///
/// The tour holds it up while a card points into it (§24.3), or a stroke would sleep the
/// panel out from under the card's arrow. Shared by the fade and the wake slice, which
/// must not be mounted over a stack that is on screen.
fn standing_down(state: AppState) -> bool {
    // Short-circuits, so an awake stack subscribes to neither the setting nor the tour. The
    // setting is asked here too, so a sleeping stack wakes when the artist switches to a
    // mode that does not sleep. Safe, because falling asleep is itself a write of
    // `panels_asleep`.
    (state.panels_asleep)()
        && (state.chrome_hiding)().sleeps()
        && !crate::tutor::holding_panels(state)
}

/// Wake the stack: from the wake slice ([`PanelStack`]), [`open_panel`], and the tour
/// putting away a card that pointed into it (§24.3). Idempotent.
pub fn wake_panels(state: AppState) {
    let mut asleep = state.panels_asleep;
    // Into a `bool` before the write: a signal read held across a write of itself panics.
    // Guarded, since `set` dirties readers whatever it is handed and this runs per move.
    let sleeping = *asleep.peek();
    if sleeping {
        asleep.set(false);
    }
}

/// The wake slice's handler: reaching into the column is a **deed** as well as a wake
/// (§24.2). Not reported inside [`wake_panels`], whose other callers are not gestures.
/// Reported before the wake, so the tour reads the panels as still asleep.
fn reach_for_panels(state: AppState) {
    crate::tutor::did(state, crate::tutor::Deed::WokePanels);
    wake_panels(state);
}

/// Stand the panels down, as the end of a canvas gesture does
/// ([`Signals::panels_asleep`](crate::state::Signals::panels_asleep)). Public for the
/// tour, whose wake lesson points at a slice that exists only while they sleep (§24.3).
/// Idempotent.
pub fn sleep_panels(state: AppState) {
    // The one door, so the setting is asked here rather than by its callers
    // (`input::end_interaction` and the tour).
    if !state.prefs.peek().chrome_hiding.sleeps() {
        return;
    }
    let mut asleep = state.panels_asleep;
    // Into a `bool` before the write, as in `wake_panels`.
    let awake = !*asleep.peek();
    if awake {
        asleep.set(true);
    }
}

/// Open or close `id`, and remember it. **The only writer of [`PanelLayout::hidden`]**,
/// so durability is structural, as `prefs::set` makes it for the preferences.
///
/// Guarded on the set changing, since a `Signal` write wakes readers even when unchanged,
/// and the tour opens panels that are often already open (§24.3). Answers whether it
/// moved, so the tour ignores a no-op (§24.2).
fn set_open(state: AppState, layout: PanelLayout, id: PanelId, open: bool) -> bool {
    let mut hidden = layout.hidden;
    // Into a local before the write: a read guard held across one panics.
    let was_open = !hidden.peek().contains(&id);
    if was_open == open {
        return false;
    }
    if open {
        hidden.write().remove(&id);
    } else {
        hidden.write().insert(id);
    }
    crate::visibility::persist(state);
    true
}

/// Fold `id` to its title bar, or unfold it: what clicking a title does ([`Panel`]).
///
/// Not a close: the panel keeps its slot, height and drag order. The content is hidden by
/// the stylesheet rather than unmounted, so its scroll and measurements survive.
pub fn toggle_collapse(state: AppState, layout: PanelLayout, id: PanelId) {
    let mut collapsed = layout.collapsed;
    // Bound before the write: a guard in a `match` or `if let` scrutinee stays live
    // through arms that write.
    let folded = collapsed.peek().contains(&id);
    if folded {
        collapsed.write().remove(&id);
    } else {
        collapsed.write().insert(id);
    }
    crate::visibility::persist(state);
}

/// Close `id`, and remember it — a panel's own ✕ ([`Panel`]) and the closing half of
/// its row in the visibility menu.
pub fn close_panel(state: AppState, layout: PanelLayout, id: PanelId) {
    if set_open(state, layout, id, false) {
        // Only when a panel actually closed: the tour explains where it went (§24.5).
        crate::tutor::did(state, crate::tutor::Deed::ClosedPanel);
    }
}

/// Show `id`, unfolded, and wake the stack. **The only way a panel is opened**, so no call
/// site can un-hide a panel into a sleeping stack.
pub fn open_panel(state: AppState, layout: PanelLayout, id: PanelId) {
    set_open(state, layout, id, true);
    // Unfolded, since opening is a request to see the panel; a tour card pointing into a
    // folded one would point at a closed lid (§24.3).
    let mut collapsed = layout.collapsed;
    // Read out before the write.
    let folded = collapsed.peek().contains(&id);
    if folded {
        collapsed.write().remove(&id);
        crate::visibility::persist(state);
    }
    wake_panels(state);
}

/// Show `id` if it is hidden, hide it if not: its row in the visibility menu
/// (`stark_ui::commands::VisibilityToggle::Panel`). Only the opening half wakes the stack.
pub fn toggle_panel(state: AppState, layout: PanelLayout, id: PanelId) {
    // Bound before the write: a guard in a `match` or `if let` scrutinee stays live
    // through arms that write.
    let was_hidden = layout.hidden.peek().contains(&id);
    if was_hidden {
        open_panel(state, layout, id);
    } else {
        close_panel(state, layout, id);
    }
}

/// The floating tool panels, top-right, rendered in the fixed `PanelId::VARIANTS`
/// sequence with the user's order as each panel's flex `order`.
///
/// A reorder changes an integer, so a dragged panel keeps its element and subtree (the
/// layer list's scroll). It costs `:first-child` / `:last-child`, so `Panel` marks the
/// ends with `.stack-first` / `.stack-last`.
///
/// Height and overflow are the stylesheet's (`.panel-stack`). The canvas's zoom hangs off
/// the `<canvas>`, a sibling, so a wheel over the stack cannot zoom and needs no handler.
#[component]
pub fn PanelStack() -> Element {
    let state = use_context::<AppState>();
    let layout = state.panels;
    // Re-measure after any render that could change the column's shape; effects run after
    // the patch. Content growing inside a panel is caught on pointer entry instead.
    //
    // Above the early return: hooks must run in the same order every render.
    use_effect(move || {
        let _ = (layout.hidden)().len();
        let _ = (layout.collapsed)().len();
        let _ = (layout.heights)().len();
        let _ = (layout.order)().len();
        measure_scroll(layout);
    });
    let hidden = (layout.hidden)();
    // The open panels in the user's order. Reactive, unlike `visible()`, which peeks.
    let open: Vec<PanelId> = (layout.order)()
        .into_iter()
        .filter(|id| !hidden.contains(id))
        .collect();
    // No open panel is no stack: an empty stack's padding would still take presses aimed at
    // the painting.
    if open.is_empty() {
        return rsx! {};
    }
    let count = open.len();
    // Resolved here, so only panels whose motion changed re-render as the pointer moves.
    let drag = (layout.drag)();
    let land = drag
        .as_ref()
        .filter(|d| d.live())
        .and_then(|d| landing(&open, d));
    // Whether the wake slice is mounted (see below).
    let reachable = standing_down(state) && !(state.canvas_active)();
    rsx! {
        // The slice the pointer reaches into to bring the panels back (§11): full column height,
        // since the stack itself is only as tall as its panels.
        //
        // Mounted only while asleep and with the canvas out of hand: a stroke's moves go to
        // whatever is under the pointer, so a box live mid-stroke would take the release that
        // ends it. A press here wakes the panels instead of painting, which is the only way a
        // touch that cannot hover asks for them. A wheel wakes them too, rather than vanishing
        // into the box.
        if reachable {
            div {
                class: "panel-wake",
                onpointermove: move |_| reach_for_panels(state),
                onpointerdown: move |_| reach_for_panels(state),
                onwheel: move |_| reach_for_panels(state),
            }
        }
        div {
            class: "panel-stack chrome",
            class: if stack_dimmed(state) { "dimmed" },
            // The rail's numbers: on the column's own scroll, and on pointer entry for content that
            // grew unseen.
            onscroll: move |_| measure_scroll(layout),
            onpointerenter: move |_| measure_scroll(layout),
            for id in PanelId::VARIANTS.iter().copied() {
                if let Some(slot) = open.iter().position(|p| *p == id) {
                    Panel {
                        id,
                        slot,
                        count,
                        // One panel, travelling only vertically: no x.
                        motion: land.map_or_else(
                            Motion::default,
                            |(s, dy)| s.motion(slot, (0.0, dy)),
                        ),
                        match id {
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
        PanelScrollbar {}
    }
}

/// The panel stack's own scrollbar, in the column's right padding, present only while
/// the column overflows (§11).
///
/// Drawn by the app because Blink's native scrollbar takes its width from the content
/// box, narrowing every panel on overflow, and a pen has no wheel. Shown on
/// `.panel-stack:hover`, which needs the rail as the stack's *next* sibling, on its own
/// hover, and while dragged (`.dragging`). Not a child, since a positioned child of a
/// scroller scrolls with it; its box is written inline from the measurement instead.
#[component]
fn PanelScrollbar() -> Element {
    let state = use_context::<AppState>();
    let layout = state.panels;
    let scroll = (layout.scroll)();
    let dragging = (layout.thumb)().is_some();
    // Nothing to scroll is no rail.
    if !scroll.overflows() {
        return rsx! {};
    }
    let (top, height) = scroll.thumb();
    rsx! {
        div {
            // `chrome` also stops a faded rail catching a stroke that strays into the column.
            class: "panel-scroll chrome",
            class: if chrome_dimmed(state) { "dimmed" },
            // A thumb that vanished mid-drag would leave the hand steering nothing.
            class: if dragging { "dragging" },
            style: "top: {PANEL_INSET}px; height: {scroll.view}px;",
            div {
                class: "panel-scroll-thumb",
                style: "top: {top}px; height: {height}px;",
                // Captured, so the release arrives here whatever the pointer is over by then.
                onpointerdown: move |e| {
                    platform::capture_pointer(&e);
                    start_thumb(layout, &e);
                },
                onpointermove: move |e| thumb_move(layout, &e),
                onpointerup: move |_| thumb_end(layout),
                onpointercancel: move |_| thumb_end(layout),
            }
        }
    }
}

/// The stack's inset from the window's corner, in px: the stylesheet's `--panel-inset`,
/// needed here because the rail's `top` is inline ([`PanelScrollbar`]).
const PANEL_INSET: f32 = 4.0;

/// Unified panel chrome: a header (the title is the drag handle and the fold; ✕ closes)
/// over the panel's controls.
///
/// A panel with a [`PanelId::default_height`] gets an inline height, since it is live
/// state, and a bottom-edge grip (`.panel.resizable`). `slot` is its flex `order`
/// ([`PanelStack`]); `slot` and `count` name the column's ends as classes. `motion` is
/// the drag preview, resolved in [`PanelStack`] so only moving panels re-render; at rest
/// it is `Motion::default`, written out (see below).
#[component]
pub fn Panel(id: PanelId, slot: usize, count: usize, motion: Motion, children: Element) -> Element {
    let state = use_context::<AppState>();
    let layout = state.panels;
    // A panel that is not resizable never reads `heights` or `resize`, so another panel's
    // resize does not re-render it. Folded, a panel loses
    // its height too: a 340px panel must not fold into a 340px header.
    let folded = layout.collapsed.read().contains(&id);
    let height = id
        .default_height()
        .filter(|_| !folded)
        .map(|d| layout.heights.read().get(&id).copied().unwrap_or(d));
    let resizing = height.is_some() && layout.resize.read().is_some_and(|r| r.id == id);

    // **Every declaration, every render, including the "off" ones**: inline styles apply per
    // property, so an omitted `transform` is stranded rather than cleared. The
    // `transform`/`transition` pair is `panels::reorder::css`'s, which has the test.
    let h = match height {
        Some(h) => format!("{h}px"),
        None => "auto".to_string(),
    };
    let style = format!(
        "order: {slot}; height: {h}; {}",
        crate::panels::reorder::css(motion)
    );
    rsx! {
        div {
            class: "panel",
            class: if height.is_some() { "resizable" },
            class: if folded { "collapsed" },
            class: if motion.lifted { "dragging" },
            class: if resizing { "resizing" },
            class: if slot == 0 { "stack-first" },
            class: if slot + 1 == count { "stack-last" },
            style,
            // For `platform::panel_boxes`: a drag checks DOM boxes against panels by `panel_key`.
            "data-panel": "{panel_key(id)}",
            div { class: "panel-header",
                // The mark is inside the grip: the whole title is draggable.
                div {
                    class: "panel-title",
                    title: if folded { "Click to unfold, or drag to reorder" } else { "Click to fold, or drag to reorder" },
                    // The roster rows' press and follow (`panels::reorder`); the grip captures
                    // the pointer.
                    onpointerdown: move |e| reorder::press(layout.drag, RowKey::Panel(id), &e),
                    onpointermove: move |e| reorder::follow(layout.drag, &e),
                    // A press that travelled lands; one that did not folds the panel.
                    onpointerup: move |_| release_title(state, layout, id),
                    // A *cancel* is neither: the browser took the gesture away, and a
                    // gesture nobody finished must not be read as a click.
                    onpointercancel: move |_| drag_end(layout, id),
                    // Refuse the browser's own drag, which would take the reorder's press. The
                    // stack is also unselectable (`.panel-stack`).
                    ondragstart: move |e| e.prevent_default(),
                    {icon(crate::layout::panel_glyph(id))}
                    "{id.title()}"
                    // The fold caret, inside the grip, rotated to point where the content
                    // goes. The layer tree's fold (`stark_ui::icons::FOLD_OPEN`) cannot turn,
                    // since its rows are above it in both states.
                    span { class: "panel-fold", {icon(stark_ui::icons::FOLD_OPEN)} }
                }
                button {
                    class: "panel-close",
                    title: "Close panel",
                    onclick: move |_| close_panel(state, layout, id),
                    {icon(stark_ui::icons::CLOSE)}
                }
            }
            // The content as one ancestor to hide when folded: minimal mode re-lays some
            // children by name (`.app-root.minimal .panel .slider-row.marked`), which beats a
            // `display: none` on them on specificity. `display: contents`, so it is not a box.
            div { class: "panel-body", {children} }
            if height.is_some() {
                // In the panel's bottom padding, so the grip costs no layout.
                div {
                    class: "panel-resize",
                    title: "Drag to resize",
                    onpointerdown: move |e| start_resize(layout, id, &e),
                }
            }
        }
    }
}

/// The title bar's release: a press that travelled lands the reorder, one that did not
/// folds the panel ([`toggle_collapse`]) — told apart by distance, as a quick slot's tap
/// and hold are by time (§18.1.8). The threshold is the drag's own
/// ([`GRAB_SLOP`](crate::panels::reorder)), so no press does neither.
///
/// Asked before [`drag_end`], which spends the grab. A release with no grab folds nothing.
pub fn release_title(state: AppState, layout: PanelLayout, id: PanelId) {
    let (pressed, dragged) = {
        let grab = layout.drag.peek();
        let live = grab.as_ref().filter(|d| !d.over());
        (live.is_some(), live.is_some_and(|d| d.live()))
    };
    drag_end(layout, id);
    if pressed && !dragged {
        toggle_collapse(state, layout, id);
    }
}

/// Re-read the stack's scroll geometry into [`PanelLayout::scroll`], writing only when it
/// moved, so it can be called freely.
pub fn measure_scroll(layout: PanelLayout) {
    let Some((top, content, view)) = platform::stack_scroll() else {
        return;
    };
    let now = Scroll { top, content, view };
    let mut scroll = layout.scroll;
    // Read out before the write.
    let was = *scroll.peek();
    if was != now {
        scroll.set(now);
    }
}

/// Begin dragging the rail's thumb.
pub fn start_thumb(layout: PanelLayout, e: &Event<PointerData>) {
    let start_top = layout.scroll.peek().top;
    let mut thumb = layout.thumb;
    thumb.set(Some(ThumbGrab {
        anchor_y: e.client_coordinates().y as f32,
        start_top,
    }));
}

/// Track an in-flight thumb drag (no-op when idle). Scrolls the element and re-measures,
/// so the browser's clamping keeps the rail and the wheel in agreement.
pub fn thumb_move(layout: PanelLayout, e: &Event<PointerData>) {
    let Some(grab) = *layout.thumb.peek() else {
        return;
    };
    let scroll = *layout.scroll.peek();
    let dy = e.client_coordinates().y as f32 - grab.anchor_y;
    platform::set_stack_scroll(scroll.scrolled_by(grab.start_top, dy));
    measure_scroll(layout);
}

/// End a thumb drag. Nothing to commit — every move already scrolled — so this only
/// disarms.
pub fn thumb_end(layout: PanelLayout) {
    if layout.thumb.peek().is_none() {
        return;
    }
    let mut thumb = layout.thumb;
    thumb.set(None);
}

/// Begin resizing panel `id` from its bottom edge.
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

/// End a panel resize. Every move already wrote the height, so this only disarms.
pub fn resize_end(layout: PanelLayout) {
    if layout.resize.peek().is_none() {
        return;
    }
    let mut resize = layout.resize;
    resize.set(None);
}

/// The open panels, top to bottom: what every box, slot and landing is counted in.
fn visible(layout: PanelLayout) -> Vec<PanelId> {
    let hidden = layout.hidden.peek().clone();
    layout
        .order
        .peek()
        .iter()
        .copied()
        .filter(|p| !hidden.contains(p))
        .collect()
}

/// End a panel drag: disarm, then write the landing slot into `order`. No-op if no drag
/// is active or it never travelled.
///
/// **The disarm goes first**: the transforms are stated against the pre-drag layout, so a
/// frame carrying the new `order` with the transforms still on would apply it twice.
/// [`Grab::spend`] is terminal, so a press cannot commit twice. Nothing is deferred: a
/// settle timer that never fired would strand the layout.
pub fn drag_end(layout: PanelLayout, id: PanelId) {
    let open = visible(layout);
    // Resolved before `spend`, which ends the grab.
    let land = layout
        .drag
        .peek()
        .as_ref()
        .filter(|d| d.live())
        .and_then(|d| landing(&open, d));
    let mut drag = layout.drag;
    if let Some(d) = drag.write().as_mut() {
        d.spend();
    }
    // A press that never travelled is not a drag, and a drop back in the slot it came
    // out of is not a move.
    let Some((slide, _)) = land.filter(|(s, _)| !s.inert()) else {
        return;
    };
    let hidden = layout.hidden.peek().clone();
    let mut order = layout.order;
    let mut ord = order.write();
    ord.retain(|p| *p != id);
    // Insert before the visible panel now at `slide.gap` (for a flat list the gap is the
    // insertion index), or at the end; hidden panels keep their slots.
    let slots: Vec<usize> = ord
        .iter()
        .enumerate()
        .filter(|(_, p)| !hidden.contains(p))
        .map(|(i, _)| i)
        .collect();
    let at = slots.get(slide.gap).copied().unwrap_or(ord.len());
    ord.insert(at, id);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A panel's height and the gap between two; only their consistency matters, which makes
    /// a drag of `STEP` one slot.
    const H: f32 = 120.0;
    const GAP: f32 = 10.0;
    const STEP: f32 = H + GAP;
    const TOP: f32 = 64.0;

    /// Three open panels and the boxes `platform::panel_boxes` would report for them
    /// — **keyed the way the DOM keys them**, which is the round trip under test.
    fn stack() -> (Vec<PanelId>, Vec<(String, f32, f32)>) {
        let open = vec![PanelId::Color, PanelId::Brush, PanelId::Select];
        let boxes = open
            .iter()
            .enumerate()
            .map(|(i, id)| (panel_key(*id), TOP + i as f32 * STEP, H))
            .collect();
        (open, boxes)
    }

    /// Drag `id` by `dy`, far enough to be a drag, and resolve it.
    fn dragged(open: &[PanelId], boxes: Vec<(String, f32, f32)>, id: PanelId, dy: f32) -> Slide {
        let mut grab = Grab::begin(panel_key(id), boxes, (0.0, 0.0));
        grab.track((0.0, dy), true);
        landing(open, &grab).expect("resolves").0
    }

    /// The DOM identity, the grab key and the landing's list are one string, or a drag
    /// measures a panel through its neighbour's box.
    #[test]
    fn a_grab_resolves_against_the_key_the_panel_wears() {
        let (open, boxes) = stack();
        let grab = Grab::begin(panel_key(PanelId::Brush), boxes, (0.0, 0.0));
        let keys: Vec<String> = open.iter().copied().map(panel_key).collect();
        assert_eq!(grab.resolve(&keys).map(|(i, _)| i), Some(1));
    }

    /// A panel dragged down past its neighbour lands in that neighbour's slot, and
    /// one dragged clear to the end lands at the end.
    #[test]
    fn a_panel_lands_in_the_slot_it_was_dragged_to() {
        let (open, boxes) = stack();
        for (dy, gap) in [(0.0, 0), (STEP, 1), (2.0 * STEP, 2), (9.0 * STEP, 2)] {
            let slide = dragged(&open, boxes.clone(), PanelId::Color, dy);
            assert_eq!(slide.gap, gap, "{dy}px down");
        }
    }

    /// A drag that goes nowhere writes no order: neither a press under the slop nor a drag
    /// back to where it started.
    #[test]
    fn a_drag_that_goes_nowhere_commits_nothing() {
        let (open, boxes) = stack();
        let mut grab = Grab::begin(panel_key(PanelId::Brush), boxes.clone(), (0.0, 0.0));
        grab.track((0.0, 1.0), true);
        assert!(!grab.live(), "a press under the slop is still a press");
        assert!(
            dragged(&open, boxes, PanelId::Brush, 0.0).inert(),
            "and a drag back to where it started is not a move"
        );
    }

    /// A panel with no measured box abandons the gesture: a stack that changed mid-drag.
    #[test]
    fn an_unmeasured_panel_abandons_the_gesture() {
        let (mut open, boxes) = stack();
        let mut grab = Grab::begin(panel_key(PanelId::Color), boxes, (0.0, 0.0));
        grab.track((0.0, 4.0 * STEP), true);
        open.push(PanelId::Layers);
        assert!(landing(&open, &grab).is_none(), "Layers has no box");
    }

    /// The name a panel is stored under is the name it wears in the DOM. Holds by
    /// construction today; a `rename_all` or a hand-written `panel_key` would make the stack
    /// silently forget itself between visits.
    #[test]
    fn a_panel_is_one_name_in_the_dom_and_in_the_store() {
        for &id in PanelId::VARIANTS {
            let stored = serde_json::to_string(&id).unwrap();
            assert_eq!(stored, format!("\"{}\"", panel_key(id)));
            assert_eq!(serde_json::from_str::<PanelId>(&stored).unwrap(), id);
        }
        assert!(serde_json::from_str::<PanelId>("\"Atmosphere\"").is_err());
    }

    /// The thumb says what share of the column is showing, and where in it — the two
    /// facts a rail is for.
    #[test]
    fn the_thumb_is_the_share_of_the_column_that_shows() {
        // Twice as much content as fits: half a rail of thumb, at the top.
        let at_top = Scroll {
            top: 0.0,
            content: 800.0,
            view: 400.0,
        };
        assert!(at_top.overflows());
        let (top, height) = at_top.thumb();
        assert_eq!((top, height), (0.0, 200.0));
        // Scrolled to the end, the thumb is against the end.
        let at_end = Scroll {
            top: 400.0,
            ..at_top
        };
        let (top, height) = at_end.thumb();
        assert_eq!((top, height), (200.0, 200.0));
        // A column that fits has no rail at all.
        assert!(
            !Scroll {
                top: 0.0,
                content: 400.0,
                view: 400.0,
            }
            .overflows()
        );
    }

    /// A very long column still gets a grabbable thumb.
    #[test]
    #[expect(
        clippy::float_cmp_const,
        reason = "the floor lands on MIN_THUMB itself, so the assertion is identity rather than proximity"
    )]
    fn a_long_column_keeps_a_grabbable_thumb() {
        let scroll = Scroll {
            top: 0.0,
            content: 40_000.0,
            view: 400.0,
        };
        let (_, height) = scroll.thumb();
        assert_eq!(height, MIN_THUMB);
    }

    /// Dragging the thumb the length of its travel scrolls the column the length of
    /// its overflow: the thumb stays under the pointer rather than drifting off it.
    #[test]
    fn a_thumb_drag_inverts_where_the_thumb_is_drawn() {
        let scroll = Scroll {
            top: 0.0,
            content: 1200.0,
            view: 400.0,
        };
        let (_, height) = scroll.thumb();
        let travel = scroll.view - height;
        assert_eq!(scroll.scrolled_by(0.0, travel), 800.0, "to the very end");
        assert_eq!(scroll.scrolled_by(0.0, travel * 0.5), 400.0, "halfway");
        // And neither end can be dragged past.
        assert_eq!(scroll.scrolled_by(0.0, -50.0), 0.0);
        assert_eq!(scroll.scrolled_by(800.0, 50.0), 800.0);
    }
}
