//! Floating-panel layout: identity, order, visibility, the title-bar drag, and the
//! fade that gets all the floating chrome out of the way mid-gesture
//! (§11).
//!
//! The title-bar drag **is** the row drag the layer tree and the guide list use
//! (`panels::reorder`), and this file is the third caller rather than a third copy
//! of it. It predated the extraction and kept its own `DragState` for a while:
//! the same measured-at-the-press boxes, the same slot opening under the hand, the
//! same leading-edge rule stated in the same words — and, being the copy with no
//! tests, the one where the two could quietly stop agreeing. A stack of panels is
//! a flat list, which is the *simplest* case of what that module already does for a
//! tree: the travelling block is one row, and `Slide::gap` is the insertion index
//! with no depth to resolve.
//!
//! The drag math deliberately never reads the layout it is mutating: panel
//! positions are measured once at drag start and everything after is derived from
//! the live pointer, so a sliding neighbour cannot feed back into the decision that
//! moved it. That property now lives in [`Grab`] and is tested there.
//!
//! # Which panels are open follows the browser
//!
//! **Every panel starts closed**, and what is open is this browser's, kept between
//! visits the way the shape and preset libraries are (`crate::storage`). The
//! opening screen is therefore the painting and nothing else, and the panels that
//! come back are the ones the artist actually reached for — a stack assembled by
//! use rather than a default arrangement everybody has to disassemble.
//!
//! The two halves of that are one decision. A set of panels chosen for you is only
//! tolerable because it resets every visit; once the choice sticks, the honest
//! starting point is none. And the tour is what keeps "none" from meaning "hidden"
//! — the Color panel arrives on the second stroke and the wake gesture is explained
//! on the third (§24.5).
//!
//! Durability is structural rather than remembered: [`set_open`] is the only thing
//! that writes [`PanelLayout::hidden`], and it persists after every change, so a
//! new way to close a panel is durable without its author thinking about storage.
//! The same move `settings::SettingToggle` makes for the preferences.

use std::collections::{HashMap, HashSet};

use dioxus::prelude::*;

use crate::icons::{self, icon};
use crate::panels::reorder::{Grab, Motion, Slide};
use crate::panels::{BrushPanel, ColorPanel, GuidesPanel, LayerPanel, LightingPanel, SelectPanel};
use crate::platform;
use crate::state::AppState;

/// Identity of a floating tool panel. The set is fixed; `PanelLayout` tracks their
/// order and which are open (§11).
///
/// Serde, because a panel is named in two stored records — which panels this browser
/// left open, and a rebinding of `Command::TogglePanel` — and the derive spells a
/// variant exactly as `Debug` does. So the stored name, the `data-panel` attribute and
/// the drag key ([`panel_key`]) are one word by construction, and a variant renamed
/// costs the stored row rather than mis-matching it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub enum PanelId {
    Color,
    Brush,
    Select,
    Layers,
    Guides,
    Lighting,
}

impl PanelId {
    /// Every panel, in the default top-to-bottom order. Color leads: it is what the
    /// next stroke is made of, and the one panel an artist reaches for between
    /// nearly every pair of them.
    pub const ALL: [PanelId; 6] = [
        PanelId::Color,
        PanelId::Brush,
        PanelId::Select,
        PanelId::Layers,
        PanelId::Guides,
        PanelId::Lighting,
    ];

    /// The panel's title-bar label.
    pub fn title(self) -> &'static str {
        match self {
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
            PanelId::Color => icons::COLOR,
            PanelId::Brush => icons::BRUSH,
            PanelId::Select => icons::SELECTION,
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
            _ => None,
        }
    }
}

/// The shortest a resizable panel may be dragged.
const MIN_PANEL_HEIGHT: f32 = 140.0;

/// Shared `Copy` layout state for the floating panels: their display order, which are
/// hidden, and the two in-flight gestures. Closed panels stay in `order` (so reopening
/// restores their slot); the stack renders `order` minus `hidden`. A field of
/// [`AppState`] (`state.panels`) rather than a context of its own, because the panel
/// toggles are registry commands now (`commands::Command::TogglePanel`) and a command
/// reaches everything it acts on through that one handle.
///
/// No panel geometry is kept here — a drag reads it off the DOM at the moment it starts
/// ([`platform::panel_boxes`]), so there is no cached measurement to fall out of date.
#[derive(Clone, Copy)]
pub struct PanelLayout {
    pub order: Signal<Vec<PanelId>>,
    pub hidden: Signal<HashSet<PanelId>>,
    /// Which panels are **folded to their title bar** — open, in their slot, showing
    /// their header and nothing else ([`toggle_collapse`]).
    ///
    /// Separate from [`hidden`](Self::hidden) because they answer different questions:
    /// a closed panel is one the artist is not using, and a collapsed one is a panel
    /// they are using and do not need to *see* right now — a Brush panel folded away
    /// while its quick brushes are on the keyboard is still the panel a slot writes
    /// into. Stored in the same table for the same reason (see [`set_open`]).
    pub collapsed: Signal<HashSet<PanelId>>,
    /// The in-flight title-bar drag — the shared row grab (`panels::reorder`).
    pub drag: Signal<Option<Grab>>,
    /// The current height of each resizable panel ([`PanelId::default_height`]),
    /// seeded with those defaults by [`PanelLayout::default_heights`]. A panel that
    /// is not resizable never appears here and never gets a height at all.
    pub heights: Signal<HashMap<PanelId, f32>>,
    /// The in-flight bottom-edge resize, if any.
    pub resize: Signal<Option<ResizeState>>,
    /// Where the stack is scrolled to, as the rail draws it ([`Scroll`]).
    ///
    /// A cached measurement, unlike everything else here — the *element* owns this
    /// number and the browser moves it without asking (a wheel, a keystroke, a
    /// focus). So it is re-read rather than kept: on the stack's own `scroll` event,
    /// on the pointer arriving in the column, and after any render that could have
    /// changed the column's height ([`PanelStack`]).
    pub scroll: Signal<Scroll>,
    /// The in-flight drag of the rail's thumb, if any.
    pub thumb: Signal<Option<ThumbGrab>>,
}

/// The panel stack's scroll geometry, in CSS px: how far it is scrolled, how tall its
/// content is, and how much of it is showing.
///
/// Enough to draw the rail and no more. It is a *measurement* and everything derived
/// from it is derived here, so the component that draws the thumb and the drag that
/// moves it cannot come to disagree about where the thumb is.
#[derive(Clone, Copy, PartialEq, Default, Debug)]
pub struct Scroll {
    /// `scrollTop`.
    pub top: f32,
    /// `scrollHeight` — the whole column, including what is off the ends.
    pub content: f32,
    /// `clientHeight` — the part of it on screen, which is also the rail's length.
    pub view: f32,
}

/// The shortest the rail's thumb may be drawn. A very long column would otherwise
/// give a thumb of a few pixels, which says how much is off screen at the cost of
/// being unable to be grabbed — and being grabbable is the whole reason this rail
/// exists (a tablet has no wheel).
const MIN_THUMB: f32 = 28.0;

impl Scroll {
    /// Whether there is anything to scroll. The rail is not in the DOM otherwise: it
    /// would be a control that cannot do anything, standing over the painting.
    ///
    /// A pixel of slack, because both numbers are the browser's rounding of a layout
    /// that includes fractional panel heights, and a rail that flickered on and off
    /// as a slider moved would be worse than no rail.
    pub fn overflows(self) -> bool {
        self.content > self.view + 1.0
    }

    /// The thumb as `(top, height)` in px down the rail: as tall a share of the rail
    /// as the view is of the content, at the place the scroll has reached.
    pub fn thumb(self) -> (f32, f32) {
        let height = (self.view / self.content * self.view).clamp(MIN_THUMB, self.view);
        // What the thumb has left to travel, against what the content has. Both are
        // zero-safe: `overflows` is what the caller checked, and the clamp above
        // cannot exceed the rail.
        let travel = (self.view - height).max(0.0);
        let scrollable = (self.content - self.view).max(1.0);
        let top = (self.top / scrollable * travel).clamp(0.0, travel);
        (top, height)
    }

    /// Where a thumb dragged `dy` px from `start` would leave the stack.
    ///
    /// The inverse of [`thumb`](Self::thumb)'s placement, which is what makes the drag
    /// track the hand exactly: the thumb ends up under the pointer rather than
    /// somewhere proportional to it.
    pub fn scrolled_by(self, start: f32, dy: f32) -> f32 {
        let (_, height) = self.thumb();
        let travel = (self.view - height).max(1.0);
        let scrollable = (self.content - self.view).max(0.0);
        (start + dy / travel * scrollable).clamp(0.0, scrollable)
    }
}

/// An in-flight drag of the rail's thumb. Like every other drag here, the answer is
/// derived from where the pointer *started* rather than from the thumb's live
/// position, so the thing being moved cannot feed back into the move.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ThumbGrab {
    /// Pointer Y at grab (client px).
    anchor_y: f32,
    /// `scrollTop` at grab.
    start_top: f32,
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

/// Where an in-flight drag would land, resolved against the panels as they stand
/// now: which rows travel and how far the hand has taken them.
///
/// `None` when the drag cannot be resolved — a panel without a box, which is what
/// a stack that changed under the pointer looks like. The stack then draws itself
/// at rest and the release commits nothing, rather than acting on geometry that
/// describes an arrangement that is gone (`Grab::resolve`).
///
/// The travelling block is a single panel, which is the whole difference between
/// this and the layer tree: a panel carries nothing, so there is no subtree to
/// reach back over and no depth for the pointer's *x* to choose among.
fn landing(visible: &[PanelId], drag: &Grab) -> Option<(Slide, f32)> {
    let keys: Vec<String> = visible.iter().copied().map(panel_key).collect();
    let (from, boxes) = drag.resolve(&keys)?;
    let (_, dy) = drag.delta();
    Some((Slide::resolve(&boxes, (from, from), dy)?, dy))
}

/// The identity a panel wears in the DOM, and the one a drag resolves against.
///
/// One function because it is asked in four places — the `data-panel` attribute
/// [`Panel`] writes, the key [`start_drag`] grabs by, the list [`landing`] matches,
/// and the selector the guided tour's card points at
/// ([`tutor::Anchor`](crate::tutor::Anchor)) — and a box matched to the wrong panel
/// is measured in silence: any box is a plausible box, whichever element it came
/// from (§11). Stated once, the four cannot disagree; stated four times, `{id:?}`
/// was the agreement.
pub fn panel_key(id: PanelId) -> String {
    format!("{id:?}")
}

/// What the floating chrome does while the canvas is in hand — this browser's own
/// choice (`crate::prefs`, the ⚙ dialog's APPEARANCE section).
///
/// There were two mechanisms here and no switch: every floating container fades while
/// a gesture is in flight, and the panel *stack* then stays down afterwards until the
/// pointer reaches into its column. Which of those is wanted turns out to be a fact
/// about the hardware rather than a matter of taste — a pen on a tablet crosses the
/// panel column on the way to everything, so the wake gesture costs it nothing and it
/// gets the whole window back; a mouse reaching for a slider between strokes pays the
/// reach every time. The one thing that could not stay is the assumption.
///
/// Three states rather than two switches, because the third combination does not
/// exist: a stack that stays down after a gesture it never faded for is a panel
/// vanishing at the moment the artist stopped painting, which is nobody's preference.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(from = "String", into = "String")]
pub enum ChromeHiding {
    /// Never get out of the way: the chrome is where it was put, whatever the hand is
    /// doing.
    Never,
    /// Fade for the length of a canvas gesture and come straight back when it ends.
    WhilePainting,
    /// Fade for the gesture, and leave the panel stack down afterwards until the
    /// pointer reaches into its column (§11). The default, and what Stark did before
    /// there was anything to choose.
    #[default]
    AfterPainting,
}

impl ChromeHiding {
    /// Whether a canvas gesture takes the chrome with it.
    fn fades(self) -> bool {
        !matches!(self, Self::Never)
    }

    /// Whether the panel stack stays down once the gesture has ended.
    ///
    /// Public because it is not only the fade's question: it decides whether
    /// [`sleep_panels`] does anything at all, and the tour asks it about the one
    /// lesson whose whole subject is this gesture (§24).
    pub fn sleeps(self) -> bool {
        matches!(self, Self::AfterPainting)
    }

    /// The name this is stored and chosen by — one vocabulary for the store and the
    /// dialog, so a row of the dialog cannot come to mean a value nothing reads.
    pub fn key(self) -> &'static str {
        match self {
            Self::Never => "never",
            Self::WhilePainting => "while-painting",
            Self::AfterPainting => "after-painting",
        }
    }
}

impl From<String> for ChromeHiding {
    /// A name this build does not know reads as the default — a preference written by
    /// a later version, or a damaged one. Lenient on purpose: `Prefs` is one JSON blob
    /// (`crate::prefs`), so an enum that refused an unknown variant would take every
    /// *other* setting down with it rather than costing its own.
    fn from(name: String) -> Self {
        [Self::Never, Self::WhilePainting, Self::AfterPainting]
            .into_iter()
            .find(|c| c.key() == name)
            .unwrap_or_default()
    }
}

impl From<ChromeHiding> for String {
    fn from(c: ChromeHiding) -> Self {
        c.key().to_string()
    }
}

/// The class list for a floating-chrome container (the panel stack, the command rail,
/// the selection bar): its own class plus `chrome`, and `dimmed` while a canvas gesture
/// is in flight *and* this browser has asked for that ([`ChromeHiding`]).
///
/// Every one of them sits over the canvas and none of them is what the user is looking
/// at mid-stroke, so they all fade together — the screen goes back to being the
/// painting, and comes back the moment the gesture ends. The fade is CSS
/// ([`crate::state::AppState::canvas_active`] only toggles the class), so the chrome
/// stays laid out where it was and nothing reflows on the way in or out.
pub fn chrome_class(state: AppState, base: &str) -> String {
    dim_class(base, fading(state))
}

/// Whether a gesture is taking the chrome with it right now — the gesture and the
/// setting together, since "Always show" is the answer no to both.
///
/// The gesture is read first, so chrome standing idle subscribes to nothing else: the
/// setting cannot change what an idle container looks like, and every container in the
/// app calls this.
fn fading(state: AppState) -> bool {
    (state.canvas_active)() && (state.chrome_hiding)().fades()
}

/// `base`, plus `chrome`, plus `dimmed` when it is `out` of the way. The two class
/// names the fade is written against, stated once — there are two reasons a container
/// can be faded now (see [`stack_class`]) and they must not each spell it.
fn dim_class(base: &str, out: bool) -> String {
    if out {
        format!("{base} chrome dimmed")
    } else {
        format!("{base} chrome")
    }
}

/// [`chrome_class`] for the panel stack, which fades for a second reason: it stays
/// out of the way after the gesture ends, until the pointer reaches into its column
/// ([`AppState::panels_asleep`](crate::state::AppState::panels_asleep)).
///
/// One `dimmed` for both, deliberately — mid-gesture and asleep are the same fact
/// about the stack (it is not what the screen is for right now), and giving the second
/// one a class of its own would be two ways to be invisible for the stylesheet to keep
/// in step.
fn stack_class(state: AppState) -> String {
    dim_class("panel-stack", fading(state) || standing_down(state))
}

/// Whether the stack is out of the way **and nothing is holding it up** — the
/// question both the fade and the wake slice actually want, rather than
/// [`AppState::panels_asleep`](crate::state::AppState::panels_asleep) raw.
///
/// The tour holds it up for as long as a card is pointing into it (§24.3). Without
/// that, the first stroke after a lesson appeared would put the panels back to sleep
/// underneath it and leave an arrow aimed at nothing — the card fades for the gesture
/// like all the chrome and comes back, and the panel it is about would not.
///
/// Asked in **two** places and so stated once: the class that fades the stack, and
/// the slice that exists to wake it. A slice mounted over a stack that is plainly on
/// screen would be an invisible box taking presses to perform a wake that has already
/// happened, which is precisely the thing `PanelStack`'s own comment argues must not
/// exist.
fn standing_down(state: AppState) -> bool {
    // Short-circuits, so a stack that is awake never reads the setting or the tour at
    // all and the component does not subscribe to either — and it does not need to,
    // because coming back from asleep is itself a write of the signal on the left.
    //
    // The setting is asked here as well as at [`sleep_panels`], and not as a belt: a
    // stack already asleep when the artist switches to a mode that does not sleep has
    // to come back, and this is the read that says so.
    (state.panels_asleep)()
        && (state.chrome_hiding)().sleeps()
        && !crate::tutor::holding_panels(state)
}

/// Wake the stack: whatever it was still standing down from, the panels are wanted.
///
/// Called from the slice the pointer reaches into ([`PanelStack`]), from
/// [`open_panel`], and by the tour when it puts away a card that was pointing into
/// the stack — so acknowledging a lesson leaves the panel it was about on screen
/// rather than snapping it out from under the hand (§24.3). Idempotent, and free when
/// it is already awake — a signal set to the value it holds wakes no reader.
pub fn wake_panels(state: AppState) {
    let mut asleep = state.panels_asleep;
    // Read out into a `bool` first, so the borrow the peek takes is over before the
    // write — a signal read left inline in the condition is a panic in a handler that
    // then writes it. And guarded rather than set flat: `set` marks its readers dirty
    // whatever it is handed, and this runs on every pointer move across the slice.
    let sleeping = *asleep.peek();
    if sleeping {
        asleep.set(false);
    }
}

/// The wake slice's own three handlers: the artist reaching into the column, which is
/// a **deed** as well as a wake (§24.2).
///
/// Separate from [`wake_panels`] rather than reported inside it, and the difference is
/// the whole of what the deed means: [`open_panel`] wakes the stack, and so does the
/// tour letting go of a card it was holding it up for — neither of which is anybody
/// reaching for anything, and the second is the very lesson this deed answers taking
/// itself down. The deed is the *gesture*, so it is reported where the gesture lands.
///
/// Reported before the wake, so the tour reads the panels as still asleep — which is
/// what its card is standing in front of. It is at most one deed per sleep, since the
/// first event here takes the slice out of the DOM.
fn reach_for_panels(state: AppState) {
    crate::tutor::did(state, crate::tutor::Deed::WokePanels);
    wake_panels(state);
}

/// Stand the panels down, as the end of a canvas gesture does
/// ([`AppState::panels_asleep`](crate::state::AppState::panels_asleep)).
///
/// [`wake_panels`]'s counterpart, and public where that one is not, because it has
/// an outside caller and waking has none: the tour's lesson about the wake gesture
/// points at a slice of the window that is only in the DOM while the panels are
/// standing down, so it makes sure they are (§24.3). Idempotent, and free when they
/// already are.
pub fn sleep_panels(state: AppState) {
    // **The one door**, which is why the setting is asked here rather than at the two
    // callers: the release that ends a stroke (`input::end_interaction`) and the tour
    // revealing the slice its lesson points at. A browser that has asked the panels to
    // stay put has asked both of them.
    if !state.chrome_hiding.peek().sleeps() {
        return;
    }
    let mut asleep = state.panels_asleep;
    // Read out into a `bool` before the write, as `wake_panels` does and for the
    // same two reasons: a signal read left live across a write of itself panics, and
    // a `set` marks its readers dirty whatever it is handed.
    let awake = !*asleep.peek();
    if awake {
        asleep.set(true);
    }
}

/// One **open** panel, as this browser left it (`crate::storage`).
///
/// Only the open ones are written, and the difference is what a panel added in a later
/// release does: it is absent from every stored row, so it arrives closed like
/// everything else rather than appearing unbidden in the stack of every existing user.
/// Folding is a field on the row rather than a record of its own, because it is the
/// same fact about the same panel — and a panel is only ever folded while it is open,
/// so there is no row this could appear on alone. `#[serde(default)]` makes the field
/// optional in both directions across a version.
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct StoredPanel {
    panel: PanelId,
    #[serde(default)]
    collapsed: bool,
}

impl crate::storage::Entry for StoredPanel {
    const STORE: crate::storage::Store = crate::storage::Store::Panels;
}

/// The panels this browser last had open, as [`PanelLayout::hidden`] — or every
/// panel hidden, which is both the never-visited case and the damaged-store one.
///
/// A row naming a panel this build no longer has costs that row and not the layout
/// (`storage::load_list`).
pub fn stored_hidden() -> HashSet<PanelId> {
    let open: HashSet<PanelId> = stored_panels().map(|p| p.panel).collect();
    PanelId::ALL
        .into_iter()
        .filter(|id| !open.contains(id))
        .collect()
}

/// The panels this browser last had **folded** ([`PanelLayout::collapsed`]).
///
/// Read from the same rows as [`stored_hidden`], since it is the same fact about
/// the same panel; a panel that is not open cannot appear here at all.
pub fn stored_collapsed() -> HashSet<PanelId> {
    stored_panels()
        .filter(|p| p.collapsed)
        .map(|p| p.panel)
        .collect()
}

fn stored_panels() -> impl Iterator<Item = StoredPanel> {
    crate::storage::load_list::<StoredPanel>()
        .unwrap_or_default()
        .into_iter()
}

/// Open or close `id`, and remember it. **The only thing that writes
/// [`PanelLayout::hidden`]**, which is what makes durability structural rather than
/// a line every call site has to remember — the move
/// `settings::SettingToggle` makes for the preferences.
///
/// The write is guarded on the set actually changing, since this runs from
/// [`open_panel`], which the tour calls for a panel that is very often already open
/// (§24.3) — and a `Signal` write dirties every subscriber whether or not the value
/// moved.
///
/// Answers whether it **moved**, which is what keeps the tour from reading a panel
/// opened onto a panel that was already open as anything at all (§24.2).
fn set_open(layout: PanelLayout, id: PanelId, open: bool) -> bool {
    let mut hidden = layout.hidden;
    // Into a local before the write: a read guard held across one is the shape that
    // has borrow-panicked in this crate before.
    let was_open = !hidden.peek().contains(&id);
    if was_open == open {
        return false;
    }
    if open {
        hidden.write().remove(&id);
    } else {
        hidden.write().insert(id);
    }
    persist(layout);
    true
}

/// Write the stack's durable half: a [`StoredPanel`] per open panel.
///
/// One writer for both facts, called by everything that changes either — which is
/// what keeps durability structural (see [`set_open`]): a new way to open, close or
/// fold a panel is remembered without its author thinking about storage.
fn persist(layout: PanelLayout) {
    let hidden = layout.hidden.peek().clone();
    let collapsed = layout.collapsed.peek().clone();
    let open: Vec<StoredPanel> = PanelId::ALL
        .into_iter()
        .filter(|id| !hidden.contains(id))
        .map(|panel| StoredPanel {
            panel,
            collapsed: collapsed.contains(&panel),
        })
        .collect();
    crate::storage::save_list(&open);
}

/// Fold `id` to its title bar, or unfold it — what clicking a title does
/// ([`Panel`]), and remembered like everything else about the stack.
///
/// A fold is deliberately *not* a close. The panel keeps its slot, its height and
/// its place in the drag order, and everything inside it goes on existing — the layer
/// list keeps its scroll, the preset library its place in the roster — because the
/// point of folding is to spend less of the column on a panel you are still working
/// with. Which is also why the content is hidden by the stylesheet rather than left
/// out of the render: an unmounted panel would rebuild itself, and everything it had
/// scrolled to or measured, on every unfold.
pub fn toggle_collapse(layout: PanelLayout, id: PanelId) {
    let mut collapsed = layout.collapsed;
    // Answered into a `bool` first: a `peek` in the condition stays borrowed through
    // the body, and both arms write the very signal being read.
    let folded = collapsed.peek().contains(&id);
    if folded {
        collapsed.write().remove(&id);
    } else {
        collapsed.write().insert(id);
    }
    persist(layout);
}

/// Close `id`, and remember it — a panel's own ✕ ([`Panel`]) and the closing half of
/// the Panels menu.
pub fn close_panel(state: AppState, layout: PanelLayout, id: PanelId) {
    if set_open(layout, id, false) {
        // Only where a panel actually went away. The tour answers the question this
        // raises — *where did it go?* — and answering it about a panel that was
        // already closed would be answering nobody (§24.5).
        crate::tutor::did(state, crate::tutor::Deed::ClosedPanel);
    }
}

/// Show `id`, and wake the stack. **The only way a panel is opened**, which is what
/// makes the second half structural: a panel un-hidden into a sleeping stack would be
/// a menu entry that ticks itself and changes nothing on screen, and every call site
/// that opens one would have to remember the same line.
pub fn open_panel(state: AppState, layout: PanelLayout, id: PanelId) {
    set_open(layout, id, true);
    // And unfolded, for the reason the wake below happens: opening is a request to
    // *see* a panel, and one that came back as a bare title bar would be a menu entry
    // that ticks itself and shows nothing — which is exactly what folding one and
    // closing it would otherwise leave for next time. The tour depends on it too: a
    // card pointing into a folded panel is pointing at a closed lid (§24.3).
    let mut collapsed = layout.collapsed;
    // Read out before the write, as everything else here does.
    let folded = collapsed.peek().contains(&id);
    if folded {
        collapsed.write().remove(&id);
        persist(layout);
    }
    wake_panels(state);
}

/// Show `id` if it is hidden, hide it if it is not — the Panels menu's entry.
///
/// The wake goes through [`open_panel`], so it happens on the half of the toggle that
/// opens and not on the half that closes.
pub fn toggle_panel(state: AppState, layout: PanelLayout, id: PanelId) {
    // Answered into a `bool` before the branch rather than peeked in the condition
    // itself: a signal read in an `if` stays borrowed for the whole of its body, and
    // both arms below write that same signal — which would be a panic in a handler
    // that had already ticked the menu entry.
    let was_hidden = layout.hidden.peek().contains(&id);
    if was_hidden {
        open_panel(state, layout, id);
    } else {
        close_panel(state, layout, id);
    }
}

/// The floating tool panels, top-right. Renders the open panels in a **fixed** sequence —
/// [`PanelId::ALL`], always — and states the user's order as a flex `order` on each one.
///
/// So the stack's order is not in the DOM: slot *k* renders panel *k* for the life of the
/// app, and a reorder changes an integer rather than moving anything. What that buys is
/// that a panel keeps its own element and its own subtree — the layer list's scroll and
/// the preset library's survive being dragged up the stack, where reordering the children
/// would rebuild whichever panels changed slots.
///
/// It costs the stack `:first-child` / `:last-child`, since the first child is now
/// whichever panel leads `PanelId::ALL` rather than the top of the column; `Panel` names
/// the ends with `.stack-first` / `.stack-last` instead.
///
/// A stack no taller than its panels, and a stack taller than the window scrolls: both
/// are the stylesheet's job alone (`.panel-stack`). The canvas's zoom hangs off the
/// `<canvas>` element, which is this stack's *sibling*, so a wheel spent over the stack
/// is unable to reach it; adding a handler to suppress a zoom that cannot happen would
/// be a second, quieter claim about the DOM shape for the first one to fall out of step
/// with.
///
/// The one box here beyond the panels is `.panel-wake`, and it exists only while they
/// are asleep — see the comment on it below, which is where the reasoning about what a
/// box over the painting may and may not take belongs.
#[component]
pub fn PanelStack() -> Element {
    let state = use_context::<AppState>();
    let layout = state.panels;
    // Re-measure whenever the column's *shape* could have changed. Effects run after
    // the DOM is patched, which is the only moment the new height exists to be read —
    // and reading these three signals is what makes it a rule rather than a line every
    // panel-opening, panel-folding, panel-resizing call site has to remember. What it
    // cannot see is a panel that grew its own content (a layer added, a preset saved),
    // which is what the pointer arriving in the column re-measures for.
    //
    // Before the early return below, because hooks run in the same order every render
    // or Dioxus loses track of which is which.
    use_effect(move || {
        let _ = (layout.hidden)().len();
        let _ = (layout.collapsed)().len();
        let _ = (layout.heights)().len();
        let _ = (layout.order)().len();
        measure_scroll(layout);
    });
    let hidden = (layout.hidden)();
    // The open panels top to bottom — the order the *user* sees, which each panel then
    // carries as its own `order` slot. Everything below iterates the constant instead.
    //
    // Read reactively here rather than through `visible()`, which peeks: this is a
    // render and the stack must follow both signals.
    let open: Vec<PanelId> = (layout.order)()
        .into_iter()
        .filter(|id| !hidden.contains(id))
        .collect();
    // Every panel closed is no stack at all, not an empty one. An empty stack is still a
    // box over the canvas — its padding alone is a strip across the top-right corner —
    // and chrome the user cannot see must not be able to take a press aimed at the
    // painting. The same reasoning the stylesheet applies to the stack's height.
    if open.is_empty() {
        return rsx! {};
    }
    let count = open.len();
    // The drag preview is resolved to a `Motion` here, alongside each panel's slot, so
    // a panel is handed how to draw itself rather than reading the gesture itself. Only
    // the panels whose motion actually changed then re-render as the pointer moves.
    let drag = (layout.drag)();
    let land = drag
        .as_ref()
        .filter(|d| d.live())
        .and_then(|d| landing(&open, d));
    // Whether the slice below is standing by. Asleep says the panels are waiting to be
    // asked for; `!canvas_active` is what keeps the box out of a gesture's way, and it
    // is the whole safety argument for having one at all (below).
    let reachable = standing_down(state) && !(state.canvas_active)();
    rsx! {
        // **The slice the pointer reaches into to bring the panels back** (§11).
        //
        // It cannot be the stack itself: the stack is exactly as tall as the panels in
        // it and must stay that way for its scroller to work, so hovering *it* would
        // answer near the top of the window and not at the foot — "reach for the panel
        // you cannot see" is not a thing a hand can aim at. The column is what the user
        // means by where the tools are, and a column is full height.
        //
        // Which makes this the invisible box over the painting that `.panel-stack`'s own
        // comment forbids, so it is worth being exact about the two things that make it
        // admissible.
        //
        // **It is not here unless the panels are asleep and the canvas is out of hand.**
        // A gesture in flight is the case that matters: a stroke's moves are delivered
        // to whatever is under the pointer (nothing captures — the canvas is
        // full-window, and faded chrome takes no events, which is what lets a stroke
        // stray under a panel and keep painting), so a box that was live mid-stroke
        // would take the moves that crossed into this column *and the release that ends
        // the stroke* — leaving a gesture in flight with nothing left to end it. Awake,
        // it is not in the DOM at all and can take nothing.
        //
        // **What it does take, it answers.** The first press in this column while the
        // panels are asleep brings them back instead of painting — one press, and only
        // for a device that arrives without hovering first. A pointer that hovers has
        // already woken them by moving in, so it never reaches this case; a finger,
        // which does not hover, taps once to ask for the panels, and that is the only
        // way a touch-only hand *can* ask — a sleeping stack takes no taps either. The
        // wheel is here for the same reason and not for the stack's: a notch of zoom
        // spent in this column would otherwise fall into the box and be silently lost.
        if reachable {
            div {
                class: "panel-wake",
                onpointermove: move |_| reach_for_panels(state),
                onpointerdown: move |_| reach_for_panels(state),
                onwheel: move |_| reach_for_panels(state),
            }
        }
        div {
            class: stack_class(state),
            // Where the rail's numbers come from. The scroll event is the browser
            // telling us it moved the column — a wheel, a keystroke, the thumb below
            // — and the pointer arriving is the one moment a rail that is about to be
            // *shown* has to be right, which covers every way a panel's own content
            // grew while nobody was looking.
            onscroll: move |_| measure_scroll(layout),
            onpointerenter: move |_| measure_scroll(layout),
            for id in PanelId::ALL {
                if let Some(slot) = open.iter().position(|p| *p == id) {
                    Panel {
                        id,
                        slot,
                        count,
                        // The block is one panel and it only ever travels vertically,
                        // so the shift handed to `Slide` has no x.
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

/// The panel stack's own scrollbar: a thumb in the strip of padding down the column's
/// right edge, in the DOM only while the column has more in it than fits (§11).
///
/// **Why the app draws one rather than styling the browser's.** A native scrollbar in
/// Blink takes its width out of the *content* box, so the moment the column overflowed
/// every panel in it would narrow by ten pixels — and widen again when a panel closed.
/// This sits over the padding the panels already clear, so nothing moves when it
/// appears. The wheel was the only way to reach the rest of the column before it, which
/// is fine for a mouse and no use at all to the hand this application is for: a pen on
/// a tablet has no wheel, and "scroll" for that hand means *drag the bar*.
///
/// Shown while the pointer is in the column and hidden the moment it leaves — the
/// stylesheet's job, off `.panel-stack:hover` (the rail is its next sibling, which is
/// what that selector needs) and off the rail's own hover, since a pointer on the rail
/// is not on the stack. A drag keeps it up whatever the pointer is over, which is what
/// `.dragging` is for: a thumb that vanished mid-drag would leave the hand steering
/// something it cannot see.
///
/// It sits **beside** the stack rather than inside it because an absolutely-positioned
/// child of a scroll container scrolls with the content — the rail would slide off the
/// top the moment it was used. Which is also why its own top and height are written
/// inline from the measurement: as a sibling it has no way to inherit the column's
/// box.
#[component]
fn PanelScrollbar() -> Element {
    let state = use_context::<AppState>();
    let layout = state.panels;
    let scroll = (layout.scroll)();
    let dragging = (layout.thumb)().is_some();
    // Nothing to scroll is no rail: a control that cannot do anything, standing over
    // the painting, is the thing `.panel-stack`'s own comment forbids.
    if !scroll.overflows() {
        return rsx! {};
    }
    let (top, height) = scroll.thumb();
    // Fades with the rest of the floating chrome, and for its reason: mid-gesture the
    // screen goes back to being the painting. `chrome` also stops it taking the
    // pointer while faded, so a stroke that strays into the column cannot be caught by
    // a bar the artist cannot see.
    let mut class = chrome_class(state, "panel-scroll");
    if dragging {
        class.push_str(" dragging");
    }
    rsx! {
        div {
            class,
            style: "top: {PANEL_INSET}px; height: {scroll.view}px;",
            div {
                class: "panel-scroll-thumb",
                style: "top: {top}px; height: {height}px;",
                // The same capture every other drag in the app takes: the release is
                // delivered to the capturing element whatever the pointer is over by
                // then, and this is a drag whose target moves under the hand.
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

/// The stack's inset from the window's corner, which the rail is placed by — the
/// stylesheet's `--panel-inset`, in the one place the host has to know it.
///
/// Stated here because the rail's `top` is inline (see [`PanelScrollbar`]) and there
/// is no way to write "the same 4px the column uses" in an inline style. Anything
/// else about where it sits is the stylesheet's.
const PANEL_INSET: f32 = 4.0;

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
/// `motion` is the drag preview — where this panel is drawn relative to where it
/// belongs, and whether it is the one in flight (`panels::reorder`). The panel is
/// handed it rather than reading the gesture, so the motions and the slots are
/// decided together in [`PanelStack`] and only the panels that actually move
/// re-render as the pointer travels. At rest it is `Motion::default`, which is the
/// resting state written out rather than the absence of one — see below.
#[component]
pub fn Panel(id: PanelId, slot: usize, count: usize, motion: Motion, children: Element) -> Element {
    let state = use_context::<AppState>();
    let layout = state.panels;
    // `map` on the default, so a panel that is not resizable never reads either signal
    // and so never re-renders for someone else's resize.
    // Folded to its title bar, which takes its height away with its content: a panel
    // the grip left 340px tall must not fold into a 340px header.
    let folded = layout.collapsed.read().contains(&id);
    let height = id
        .default_height()
        .filter(|_| !folded)
        .map(|d| layout.heights.read().get(&id).copied().unwrap_or(d));
    let resizing = height.is_some() && layout.resize.read().is_some_and(|r| r.id == id);

    let mut class = String::from("panel");
    if height.is_some() {
        class.push_str(" resizable");
    }
    if folded {
        class.push_str(" collapsed");
    }
    class.push_str(motion.class());
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
    // is invisible until some *other* state change makes the stale value wrong. The
    // `transform`/`transition` pair is `Motion::css`'s, which holds the same rule for
    // the layer tree and the guide list — and has the test that pins it.
    let h = match height {
        Some(h) => format!("{h}px"),
        None => "auto".to_string(),
    };
    let style = format!("order: {slot}; height: {h}; {}", motion.css());
    rsx! {
        div {
            class,
            style,
            // Which panel this element is, for `platform::panel_boxes` to read back. The
            // drag measures the DOM and writes `order`, so it needs the two to agree; this
            // is what lets it check rather than assume — and it is `panel_key`, not a
            // fourth spelling of the same format.
            "data-panel": "{panel_key(id)}",
            div { class: "panel-header",
                // The mark is inside the drag handle rather than beside it: the whole
                // title *is* the grip, and a glyph sitting outside it would be the one
                // part of the header that looks draggable and is not.
                div {
                    class: "panel-title",
                    title: if folded { "Click to unfold, or drag to reorder" } else { "Click to fold, or drag to reorder" },
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
                    // A press that travelled lands; one that did not folds the panel.
                    onpointerup: move |_| release_title(layout, id),
                    // A *cancel* is neither: the browser took the gesture away, and a
                    // gesture nobody finished must not be read as a click.
                    onpointercancel: move |_| drag_end(layout, id),
                    // The browser's own drag, refused. Nothing in the stack is
                    // selectable any more (`.panel-stack` in the stylesheet), which is
                    // what actually fixed a grip that dragged a ghost of the panel's
                    // words instead of the panel — this is the belt: a native drag
                    // starting *from the grip* would take the reorder's press with it,
                    // whatever the selection is.
                    ondragstart: move |e| e.prevent_default(),
                    {icon(id.glyph())}
                    "{id.title()}"
                    // Which way this panel is folded, at the far end of the grip —
                    // inside it, so the mark is part of what you click rather than a
                    // second control beside it. One glyph rotated rather than a pair:
                    // unlike the layer tree's fold (`icons::FOLD_OPEN`), which points at
                    // rows drawn *above* it in both states and so cannot turn, a panel's
                    // content is below its bar and the caret means the direction it will
                    // go.
                    span { class: "panel-fold", {icon(icons::FOLD_OPEN)} }
                }
                button {
                    class: "panel-close",
                    title: "Close panel",
                    onclick: move |_| close_panel(state, layout, id),
                    {icon(icons::CLOSE)}
                }
            }
            // The panel's content as **one** thing, so folding it away is one rule
            // about an ancestor rather than a rule about each of the children.
            //
            // That distinction is the whole reason the wrapper exists. Minimal mode
            // re-lays some of those children by name (`.app-root.minimal .panel
            // .slider-row.marked`), and a `display: none` on the same elements loses
            // to it on specificity — a folded Brush panel went on showing its sliders.
            // A hidden *ancestor* cannot be undone by any rule about a descendant, so
            // this cannot be beaten by the next such addition either. The wrapper is
            // `display: contents`, so it is a selector and not a box: its children go
            // on being flex items of the panel exactly as they were.
            div { class: "panel-body", {children} }
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

/// The title bar's release: a press that travelled lands the reorder, and one that
/// did not folds the panel to its bar ([`toggle_collapse`]).
///
/// The two share a press because a title is one grip, and which gesture it was is a
/// question only the pointer can answer — the same "a tap and a hold are one keystroke
/// told apart by how long it lasted" argument the quick slots make (§18.1.8), here
/// told apart by how *far* it went. [`GRAB_SLOP`](crate::panels::reorder) is that
/// distance, and it is the drag's own: a fold that happened at a different threshold
/// than the one a drag begins at would leave presses in between that did neither.
///
/// Asked **before** [`drag_end`], which spends the grab and so answers the question
/// away. A release with no grab in flight folds nothing: it belongs to a press this
/// element never heard, which is a gesture that started somewhere else.
pub fn release_title(layout: PanelLayout, id: PanelId) {
    let (pressed, dragged) = {
        let grab = layout.drag.peek();
        let live = grab.as_ref().filter(|d| !d.over());
        (live.is_some(), live.is_some_and(|d| d.live()))
    };
    drag_end(layout, id);
    if pressed && !dragged {
        toggle_collapse(layout, id);
    }
}

/// Re-read the stack's scroll geometry into [`PanelLayout::scroll`].
///
/// Guarded on the value actually moving, which is what lets this be called freely —
/// from the scroll event, from the pointer arriving, and from the effect that watches
/// the column's shape. A `Signal` write marks every reader dirty whether or not the
/// value changed, and the reader here is the stack.
pub fn measure_scroll(layout: PanelLayout) {
    let Some((top, content, view)) = platform::stack_scroll() else {
        return;
    };
    let now = Scroll { top, content, view };
    let mut scroll = layout.scroll;
    // Read out before the write, as everything else in this module does.
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

/// Track an in-flight thumb drag (no-op when idle).
///
/// Scrolls the element and then re-measures, rather than writing a position of our
/// own: the browser clamps `scrollTop` at both ends, so the rail and the wheel cannot
/// come to disagree about where the column is.
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

/// The open panels, top to bottom — the order the user sees, which is what every
/// box, slot and landing here is counted in.
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

/// Begin dragging panel `id` by its title bar: measure the stack and arm the grab.
/// The pointer tracking and the reorder follow in [`drag_move`] / [`drag_end`].
///
/// Every box comes from [`platform::panel_boxes`] and is matched to its panel **by
/// the id on the same element**, never by position — the stack's children are in a
/// fixed sequence, not the user's order ([`PanelStack`]). That matching is
/// `Grab`'s, which is why nothing is resolved here: the measurement is taken whole
/// and interpreted at the moment it is used, against the list as it stands then.
///
/// Synchronous, so the drag is armed on the press rather than a JS round trip later —
/// which also means no pointer movement is dropped between the two.
pub fn start_drag(layout: PanelLayout, id: PanelId, e: &Event<PointerData>) {
    let p = e.client_coordinates();
    let mut drag = layout.drag;
    drag.set(Some(Grab::begin(
        panel_key(id),
        platform::panel_boxes(),
        (p.x as f32, p.y as f32),
    )));
}

/// Track the pointer for an in-flight panel drag (no-op when idle — the check is what
/// keeps every pointer move over the app from dirtying the whole stack).
///
/// `held` is passed on for the reason the layer tree passes it: a title bar is a
/// thing the pointer merely travels over as well as a grip, so a release this
/// handler never heard about must end the gesture rather than leave the panel
/// following an unpressed pointer around the screen (`Grab::track`).
pub fn drag_move(layout: PanelLayout, e: &Event<PointerData>) {
    if layout.drag.peek().as_ref().is_none_or(Grab::over) {
        return;
    }
    let p = e.client_coordinates();
    let held = !e.held_buttons().is_empty();
    let mut drag = layout.drag;
    if let Some(d) = drag.write().as_mut() {
        d.track((p.x as f32, p.y as f32), held);
    }
}

/// End a panel drag: disarm, then write the panel's landing slot into `order`. No-op if
/// no drag is active or if it never travelled far enough to be one.
///
/// **The disarm goes first.** A panel's offset is a transform stated against the layout as
/// it stood when the drag began, so a frame carrying the new `order` while the transforms
/// are still on would be the reorder applied twice — every affected panel a slot from where
/// it belongs. Cleared first, the worst an in-between frame can show is the stack exactly as
/// it was before the gesture: a glitch the next render corrects rather than a wrong layout.
/// [`Grab::spend`] is that disarm, and it is terminal — nothing brings a spent grab
/// back, so this cannot commit the same press twice.
///
/// Nothing is deferred here on purpose. Easing the dragged panel into its slot would
/// put a timer in charge of committing the order, and a settle whose timer never fires
/// strands the layout — at the one moment in the gesture where the user is looking at
/// the *slot*, not at the panel.
pub fn drag_end(layout: PanelLayout, id: PanelId) {
    let open = visible(layout);
    // Resolved before the disarm, because the landing is stated against the stack as
    // it stood at the press and `spend` is what stops it being drawn.
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
    // Insert before the visible panel currently at index `slide.gap` (hidden panels
    // keep their slots), or at the end. For a flat list the count of rows that end up
    // above the block *is* the insertion index.
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

    /// A panel's height and the gap between two of them. The numbers do not matter —
    /// every answer is a comparison against a panel's centre — only that they are
    /// consistent, which is what makes a drag of `STEP` "one slot down".
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

    /// The identity a panel wears in the DOM, the key a grab is armed with, and the
    /// list a landing matches are one string — which is the only thing standing
    /// between a drag and measuring a panel through its neighbour's box.
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

    /// A drag that goes nowhere is not a move, so the release writes no order. Both
    /// halves matter: a press under the slop is not a drag at all, and a drag that
    /// wandered out and came back is a drag that landed where it started.
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

    /// A panel whose box was never measured abandons the gesture rather than
    /// guessing — what a stack that changed under the pointer looks like (a panel
    /// closed by the Panels menu mid-drag).
    #[test]
    fn an_unmeasured_panel_abandons_the_gesture() {
        let (mut open, boxes) = stack();
        let mut grab = Grab::begin(panel_key(PanelId::Color), boxes, (0.0, 0.0));
        grab.track((0.0, 4.0 * STEP), true);
        open.push(PanelId::Layers);
        assert!(landing(&open, &grab).is_none(), "Layers has no box");
    }

    /// The name a panel is stored under is the name it wears in the DOM.
    ///
    /// Both come off the variant — serde's for the store, `Debug`'s for the attribute
    /// and the drag — so this cannot fail today; it is here because the day somebody
    /// gives `PanelId` a `rename_all` or hand-writes `panel_key`, the symptom is a
    /// stack that silently forgets itself between visits rather than an error.
    #[test]
    fn a_panel_is_one_name_in_the_dom_and_in_the_store() {
        for id in PanelId::ALL {
            let stored = serde_json::to_string(&id).unwrap();
            assert_eq!(stored, format!("\"{}\"", panel_key(id)));
            assert_eq!(serde_json::from_str::<PanelId>(&stored).unwrap(), id);
        }
        assert!(serde_json::from_str::<PanelId>("\"Atmosphere\"").is_err());
    }

    /// A folded panel is a **field** on its own row, so the two states cross a version
    /// in both directions: a row written before folding existed reads as an open
    /// panel, and one written by this build reads as an open panel to a version that
    /// knows only the name.
    #[test]
    fn a_stored_row_carries_whether_the_panel_was_folded() {
        let read =
            |json: &str| serde_json::from_str::<StoredPanel>(json).map(|p| (p.panel, p.collapsed));
        assert_eq!(
            read(r#"{"panel":"Brush","collapsed":true}"#).unwrap(),
            (PanelId::Brush, true)
        );
        // A row from before folding existed, and one from a build that has since
        // dropped it: the name alone is an open panel.
        assert_eq!(
            read(r#"{"panel":"Layers"}"#).unwrap(),
            (PanelId::Layers, false)
        );
        // And a panel this build lacks is unreadable as a row, which is what puts it
        // in reach of the drop `storage::load_list` does.
        assert!(read(r#"{"panel":"Atmosphere","collapsed":true}"#).is_err());
    }

    /// The stored name of a hiding mode is what the dialog offers and what
    /// [`ChromeHiding::from`] reads back, and a name from a version that knows more
    /// modes than this one reads as the default rather than refusing — which would
    /// take every *other* preference in the blob down with it (`crate::prefs`).
    #[test]
    fn a_chrome_mode_round_trips_through_its_stored_name() {
        for mode in [
            ChromeHiding::Never,
            ChromeHiding::WhilePainting,
            ChromeHiding::AfterPainting,
        ] {
            assert_eq!(ChromeHiding::from(mode.key().to_string()), mode);
        }
        assert_eq!(
            ChromeHiding::from("hide-on-tuesdays".to_string()),
            ChromeHiding::default(),
        );
        // The default is what Stark did before there was a choice, which is what keeps
        // a browser that stored its preferences before this field existed where it was.
        assert!(ChromeHiding::default().sleeps());
        assert!(ChromeHiding::default().fades());
        assert!(!ChromeHiding::Never.fades(), "never means never");
        assert!(
            !ChromeHiding::WhilePainting.sleeps(),
            "the stack comes straight back",
        );
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

    /// A very long column still gets a thumb that can be grabbed — the whole reason
    /// the rail exists is a hand with no wheel.
    #[test]
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

    /// A stored layout names the panels that are **open**, so one this build does
    /// not have costs its own row, and one it has that the record never mentioned
    /// stays closed rather than appearing unbidden.
    #[test]
    fn an_unknown_panel_costs_its_own_row() {
        let json = r#"[{"panel":"Color"},{"panel":"Atmosphere"},{"panel":"Brush"}]"#;
        let open: std::collections::HashSet<PanelId> =
            serde_json::from_str::<Vec<serde_json::Value>>(json)
                .unwrap()
                .into_iter()
                .filter_map(|v| serde_json::from_value::<StoredPanel>(v).ok())
                .map(|p| p.panel)
                .collect();
        let hidden: HashSet<PanelId> = PanelId::ALL
            .into_iter()
            .filter(|id| !open.contains(id))
            .collect();
        assert!(!hidden.contains(&PanelId::Color));
        assert!(!hidden.contains(&PanelId::Brush));
        assert!(hidden.contains(&PanelId::Layers), "never stored, so closed");
        assert_eq!(hidden.len(), PanelId::ALL.len() - 2);
    }
}
