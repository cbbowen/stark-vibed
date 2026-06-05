# First-class, unified Panels (drag-reorder · close · reopen)

## Context

The four floating tool panels — **Color, Brush, Lighting, Layers** — are today hand-
listed in a static `div.panel-stack` ([main.rs:121](crates/stark-ui/src/main.rs#L121))
and each wraps its controls in a dumb `Panel { title, children }`
([main.rs:752](crates/stark-ui/src/main.rs#L752)) that only renders a title + body.
There's no way to reorder, hide, or restore them. The user wants Panels to be a
first-class element: (1) **drag the title bar to reorder**, with the others animating
into their new spots; (2) **close** via a top-right button; (3) **reopen** from a
dedicated menu. We'll build this incrementally so each phase is independently testable.

Work is confined to the web UI crate (`crates/stark-ui`): `src/main.rs` and
`assets/stark.css`. No engine/core changes.

## Design overview

Introduce a panel **identity + shared layout state**, make the stack **data-driven**,
and give the unified `Panel` a real header (drag handle + close button).

- `PanelId` enum `{ Color, Brush, Lighting, Layers }` with `ALL` and `title()`.
- A `PanelLayout` context (a `Copy` struct of signals), provided in `app()`:
	- `order: Signal<Vec<PanelId>>` — display order (defaults to `ALL`).
	- `hidden: Signal<HashSet<PanelId>>` — closed panels (kept in `order` so reopening
		restores their slot).
	- `drag: Signal<Option<DragState>>` — the in-flight drag (phases 2–3).
	- `refs: Signal<HashMap<PanelId, Rc<MountedData>>>` — each panel's mounted node, for
		measuring (phases 2–3).
- Content components (`ColorPanel`/`BrushPanel`/`LightingPanel`/`LayerPanel`) stop
	wrapping themselves in `Panel` and **return just their inner controls**. A new
	`PanelStack` iterates `order`, skips `hidden`, and renders each id through the
	unified `Panel` wrapper (keyed by id so Dioxus preserves node identity while
	reordering — needed for the animation), matching `id → content component`.

## Phase 1 — Unify + close + reopen (no drag yet)

- Add `PanelId` and `PanelLayout`; `use_context_provider` it in `app()`.
- Replace the static stack body with `PanelStack {}`.
- `Panel` becomes `Panel { id, children }`: a `.panel-header` row with `.panel-title`
	(left) and a `.panel-close` ✕ button (right) that does `hidden.write().insert(id)`.
	Title text comes from `id.title()`.
- Refactor the four content components to drop their own `Panel { title, .. }` wrapper.
- **Dedicated reopen menu:** add a second `MenubarMenu` (a "Panels" trigger) to the
	existing menubar in `CommandRail` ([main.rs:580](crates/stark-ui/src/main.rs#L580)),
	one `MenubarItem` per `PanelId::ALL` toggling membership in `hidden`; show a ✓ when
	visible. Reuses the existing `components::menubar` wrappers — no new menu code.
- **CSS** (`stark.css`): `.panel-header` (flex, space-between; drag-handle cursor on the
	title), `.panel-close` (subtle ✕, hover highlight). Keep the `.panel`/`.panel-title`
	look; the old `.panel-title { margin-bottom }` moves onto the header.
- **Outcome:** every panel has a ✕; the Panels menu hides/restores them; order is data-
	driven (still fixed).

## Phase 2 — Drag the title bar to reorder (functional)

- `Panel` root `div` gets `onmounted` → store `Rc<MountedData>` in `layout.refs` keyed by
	id; the `.panel-title` gets `onpointerdown` → start a drag.
- `DragState { id, pointer_y, grab_offset, panels: Vec<(PanelId, top, height)>, target }`.
	On pointerdown, `spawn` an async measure of every visible panel via
	`MountedData::get_client_rect().await` (confirm the exact method in this Dioxus build)
	to fill `panels`; set `drag`.
- Drive the drag at the **`app-root`** level (events bubble there even over the canvas):
	add `onpointermove` (update `pointer_y`; recompute the `target` index by comparing the
	dragged panel's current center against the others' midpoints, with hysteresis) and
	`onpointerup` (commit: move `id` to `target` in `order`; clear `drag`). The canvas
	won't paint/pan during this — its handlers are gated on `drawing()`/`panning()`, which
	a title-bar drag never sets.
- The dragged panel lifts: inline `style` `transform: translateY(dy); z-index:…` + a
	`.dragging` class (shadow/scale). A simple **drop indicator** (a 2px accent line at the
	`target` boundary) shows where it will land.
- **Outcome:** drag a panel up/down; it follows the pointer; releasing reorders.

## Phase 3 — The pleasant shift animation (FLIP)

Replace the static drop indicator with the other panels **sliding** to make room.

- Reorder `order` **live** as the drag's `target` changes (not just on drop).
- Animate via **FLIP**: keep `prev_top: HashMap<PanelId, f64>`. A `use_effect` keyed on
	`order` runs after each reorder render: measure new tops (from `refs`); for each
	non-dragged panel set its transform to `translateY(prev − new)` with **no** transition
	(Invert), then `spawn` a task that on the next frame sets it to `0` with `transition:
	transform 180ms ease` (Play). Store new tops as `prev_top`.
- Per-panel transform is a small signal the `Panel` reads (e.g. a `HashMap<PanelId,
	Signal<f32>>` in `PanelLayout`). The dragged panel is excluded (it tracks the pointer);
	on drop it animates from its lifted transform to `0` (settle) with the same transition.
- `.panel { transition: transform 180ms ease }`, suppressed (`transition: none`) on the
	actively-dragged panel so it tracks the pointer 1:1.
- **Outcome:** dragging visibly pushes neighbors aside; everything settles on drop.

## Key files

- [crates/stark-ui/src/main.rs](crates/stark-ui/src/main.rs) — `PanelId`/`PanelLayout`,
	`PanelStack`, the rewritten `Panel`, the four content components (un-wrap), the
	"Panels" menu in `CommandRail`, app-root drag handlers.
- [crates/stark-ui/assets/stark.css](crates/stark-ui/assets/stark.css) — `.panel-header`,
	`.panel-close`, `.panel.dragging`, transforms/transitions.
- Reuse `components::menubar::{Menubar, MenubarMenu, MenubarTrigger, MenubarContent,
	MenubarItem}` (already imported) for the reopen menu.

## Verification

After each phase: `cargo build -p stark-ui --target wasm32-unknown-unknown` stays green
(the primary gate — this crate has no unit tests) and `cargo clippy -p stark-ui --target
wasm32-unknown-unknown` clean (ignoring pre-existing `environment`/`vendor` warnings).
Then `dx serve` (or the user's run flow) and exercise by hand:
- Ph1: each panel shows a ✕; closing hides it; the Panels menu shows ✓s and restores a
	closed panel into its original slot.
- Ph2: dragging a title bar lifts the panel and follows the pointer; dropping reorders;
	order persists; the canvas neither paints nor pans during a panel drag.
- Ph3: neighbors slide to make room as you drag and the dropped panel settles smoothly;
	no flicker when crossing boundaries.

## Notes / risks

- **Measuring:** confirm the `MountedData` measurement method in this Dioxus 0.7 build
	(`get_client_rect`); it's async, so measures run inside `spawn`. If the API differs,
	fall back to a tiny `web-sys` `getBoundingClientRect` (web-sys is already a dep), gated
	to `wasm32`.
- **Keying:** the `key:` on each `Panel` in `PanelStack` must be the stable `id` so
	reordering moves existing nodes (preserving each panel's internal signal state and
	enabling FLIP) instead of recreating them.
- Phases 2–3 are the only non-trivial part; Phase 1 is a safe refactor that already
	delivers close/reopen. We can stop after any phase with a coherent result.
