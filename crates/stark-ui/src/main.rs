//! Stark's Dioxus **web** frontend (DESIGN.md §11).
//!
//! The backend runs in WASM and paints through a WebGPU surface bound to the
//! page's `<canvas>` — the engine renders straight into the surface texture
//! after each command, with no GPU→CPU readback. The canvas fills the window;
//! unobtrusive floating panels (color, brush, layers) sit on top.
//!
//! Run with `dx serve --web -p stark-ui` in a WebGPU-capable browser.

mod brush_editor;
mod collab;
mod components;
mod render;

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use dioxus::html::geometry::ElementPoint;
use dioxus::html::input_data::MouseButton;
use dioxus::html::{Key, Modifiers};
use dioxus::prelude::*;

use brush_editor::BrushEditorModal;
use components::menubar::{Menubar, MenubarContent, MenubarItem, MenubarMenu, MenubarTrigger};
use render::{Renderer, BG, CANVAS_ID};
use stark_core::document::{
    BrushDynamics, BrushParams, BrushShape, OrientationSource, Tool,
};
use stark_core::color::{oklab_to_srgb, srgb_to_oklab};
use stark_core::geom::Vec2;
use stark_core::{
    ColorSpaceId, EnvironmentId, InputCommand, InputSample, LayerInfo, MediaParams,
    ObservableState, SurfaceId,
};

/// Built-in assets, bundled as static files and **fetched at runtime** so they
/// stay out of the wasm binary (DESIGN.md §6.6). The engine is handed the bytes.
const BRISTLE_BRUSH: Asset = asset!("/assets/shape/WornBristles.png");
const SURFACE_LINEN: Asset = asset!("/assets/surface/Linen.png");
const ENV_FERNDALE: Asset = asset!("/assets/environment/ferndale_studio_11_1k.hdr");

/// The selectable canvas surfaces, in display order (DESIGN.md §6.4). Adding a
/// surface = one row here (plus its asset fetch in [`set_surface`]); the Lighting
/// panel's drop-down renders this table.
const SURFACES: &[(SurfaceId, &str)] = &[
    (SurfaceId::Flat, "Smooth"),
    (SurfaceId::Linen, "Linen"),
];

/// Oklab a/b picker field, on screen (px) — a square `a`×`b` plane at the current `L`.
const FIELD_PX: f32 = 220.0;
/// Oklab `L` slider height, on screen (px).
const L_H: f32 = 220.0;
/// Half-extent of the `a`/`b` axes shown in the field. Symmetric, so it covers most of
/// the sRGB gamut (blue reaches b ≈ −0.31); out-of-gamut corners clamp.
const AB: f32 = 0.32;
/// Rendered resolution of the a/b field BMP (CSS scales it to `FIELD_PX`, smoothly —
/// the plane is low-frequency, and a small BMP keeps the data URL cheap to regenerate
/// while dragging `L`). `N·3` is a multiple of 4, so BMP rows need no padding.
const FIELD_N: usize = 96;

/// Ternary pad triangle, on screen (px): width matches the colour field, height makes
/// it equilateral (`w·√3/2`). Mirrored by `.ternary`/`.ternary-tri` in stark.css.
const TRI_W: f32 = 220.0;
const TRI_H: f32 = 190.0;
/// Vertical room above/below the ternary triangle for its vertex labels (px).
const TRI_LBL: f32 = 16.0;

/// The UI's global stylesheet — panel chrome (shared CSS custom properties) plus
/// every component class referenced below. Linked once by [`app`] so the rsx!
/// blocks carry class names, not inline styles. Custom properties are global, so
/// the css_module menubar styles pick up `--panel-shadow` / `--panel-background` too.
static STARK_CSS: Asset = asset!("/assets/stark.css");

fn main() {
    #[cfg(target_arch = "wasm32")]
    {
        console_error_panic_hook::set_once();
        // Route `tracing` events (engine + UI) to the browser console.
        tracing_wasm::set_as_global_default();
    }
    dioxus::launch(app);
}

/// Identity of a floating tool panel. The set is fixed; `PanelLayout` tracks their
/// order and which are open (DESIGN.md §11).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum PanelId {
    Color,
    Brush,
    Lighting,
    Layers,
}

impl PanelId {
    /// Every panel, in the default top-to-bottom order.
    const ALL: [PanelId; 4] = [PanelId::Color, PanelId::Brush, PanelId::Lighting, PanelId::Layers];

    /// The panel's title-bar label.
    fn title(self) -> &'static str {
        match self {
            PanelId::Color => "Color",
            PanelId::Brush => "Brush",
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
struct PanelLayout {
    order: Signal<Vec<PanelId>>,
    hidden: Signal<HashSet<PanelId>>,
    drag: Signal<Option<DragState>>,
    /// Each visible panel's mounted node, so a drag can measure their positions.
    refs: Signal<HashMap<PanelId, Rc<MountedData>>>,
}

/// An in-flight title-bar drag. `panels` is the visible panels' `(id, top, height)`
/// measured once at drag start (client px, top-to-bottom); everything else is derived
/// from the live pointer so the math never feeds back on the shifting layout. Once the
/// pointer is released, `release` holds the dragged panel's target offset and it settles
/// there (sliding back to 0 if nothing reordered) before the new order is committed.
#[derive(Clone, PartialEq)]
struct DragState {
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
                if *k < self.from { top >= center } else { bottom > center }
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

/// Shared `Copy` handle to the app's signals.
#[derive(Clone, Copy)]
struct AppState {
    /// Surface + engine, built asynchronously once the canvas mounts. `None`
    /// until WebGPU init completes. Not `Send` — lives in unsync storage.
    renderer: Signal<Option<Renderer>>,
    /// UI-facing engine projection, refreshed after each command.
    obs: Signal<Option<ObservableState>>,
    /// Whether the user is holding space.
    space_down: Signal<bool>,
    /// Whether the brush editor dialog is open (rendered at the app root so its
    /// backdrop escapes the panels' `backdrop-filter` containing blocks).
    brush_editor_open: Signal<bool>,
    /// The live shared-drawing session, if any (DESIGN.md §12). `!Send` iroh
    /// handles live in unsync storage beside the renderer.
    collab_session: Signal<Option<stark_net::CollabSession>>,
    /// The shareable ticket string, while hosting/joined.
    collab_ticket: Signal<Option<String>>,
    /// Where the session lifecycle stands (drives the dialog + rail badge).
    collab_phase: Signal<collab::CollabPhase>,
    /// The last share/join failure, surfaced in the dialog.
    collab_error: Signal<Option<String>>,
}

fn app() -> Element {
    let renderer = use_signal(|| None::<Renderer>);
    let obs = use_signal(|| None::<ObservableState>);
    let space_down = use_signal(|| false);
    let brush_editor_open = use_signal(|| false);
    let collab_session = use_signal(|| None::<stark_net::CollabSession>);
    let collab_ticket = use_signal(|| None::<String>);
    let collab_phase = use_signal(collab::CollabPhase::default);
    let collab_error = use_signal(|| None::<String>);
    let state = AppState {
        renderer,
        obs,
        space_down,
        brush_editor_open,
        collab_session,
        collab_ticket,
        collab_phase,
        collab_error,
    };
    use_context_provider(|| state);

    // Floating-panel layout: order + which are open. Provided so the panel chrome and
    // the "Panels" menu can reorder/close/restore them.
    let panels = PanelLayout {
        order: use_signal(|| PanelId::ALL.to_vec()),
        hidden: use_signal(HashSet::new),
        drag: use_signal(|| None),
        refs: use_signal(HashMap::new),
    };
    use_context_provider(|| panels);

    use_hook(|| {
        let mut renderer = renderer;
        let mut obs = obs;
        spawn(async move {
            let mut r = render::init(render::canvas_element(CANVAS_ID)).await;
            // Fetch the built-in brush at runtime (kept out of the wasm binary)
            // and import it once, so the Bristles chip is ready (DESIGN.md §6.6).
            if let Ok(bytes) = dioxus::asset_resolver::read_asset_bytes(BRISTLE_BRUSH).await {
                r.load_bristle(&bytes);
            }
            // Fetch the studio HDR and light the canvas with it (DESIGN.md §6.3);
            // until then the procedural studio environment is used.
            if let Ok(bytes) = dioxus::asset_resolver::read_asset_bytes(ENV_FERNDALE).await {
                r.register_environment(EnvironmentId::Ferndale, bytes);
                r.set_environment(EnvironmentId::Ferndale);
            }
            r.paint();
            obs.set(Some(r.observe()));
            renderer.set(Some(r));

            // A `#stark…` fragment in the page URL is a session invitation:
            // join it now that the engine is up (DESIGN.md §12.4).
            if let Some(ticket) = collab::url_ticket() {
                tracing::info!("joining shared session from URL fragment");
                collab::join(state, ticket);
            }
        });
    });

    rsx! {
        document::Stylesheet { href: STARK_CSS }

        div {
            class: "app-root",
            tabindex: "0",
            autofocus: true,
            onkeydown: move |e| handle_keydown(state, &e),
            onkeyup: move |e| handle_keyup(state, &e),
            // A panel drag is driven here (events bubble up even over the canvas), so it
            // keeps tracking wherever the pointer goes. No-op unless a drag is active;
            // leaving the window commits it so it can't get stuck.
            onpointermove: move |e| drag_move(panels, &e),
            onpointerup: move |_| drag_end(panels),
            onpointerleave: move |_| drag_end(panels),

            Canvas {}

            // Left command rail: rarely-used document commands, tucked away.
            CommandRail {}

            // Floating tool panels, stacked top-right — order + visibility are data-driven.
            PanelStack {}

            // The brush editor dialog (mounted only while open, so each open
            // re-inits its preview against the current canvas look).
            if (state.brush_editor_open)() {
                BrushEditorModal {
                    on_close: move |_| {
                        let mut open = state.brush_editor_open;
                        open.set(false);
                    }
                }
            }
        }
    }
}

/// The full-window painting surface (a WebGPU canvas the engine draws into).
#[component]
fn Canvas() -> Element {
    let state = use_context::<AppState>();
    let mut drawing = use_signal(|| false);
    let mut panning = use_signal(|| false);
    let mut last_position = use_signal(|| None::<Vec2>);

    rsx! {
        canvas {
            id: "{CANVAS_ID}",
            class: "paint-canvas",
            onresize: move |e| {
                if let Ok(size) = e.get_content_box_size() {
                    resize(state, size.width as u32, size.height as u32);
                }
            },
            // Strokes and pans capture the pointer (like the pads/pickers): leaving the
            // window mid-stroke keeps painting — the infinite canvas extends past the
            // viewport anyway — and the interaction ends on release/cancel, never by
            // crossing the canvas edge.
            onpointerdown: move |e| {
                match e.trigger_button() {
                    Some(MouseButton::Primary) => {
                        capture_pointer(&e);
                        if (state.space_down)() {
                            panning.set(true);
                        } else {
                            dispatch(state, InputCommand::StartStroke { tool: Tool::Brush, sample: sample(state, &e) });
                            drawing.set(true);
                        }
                    }
                    Some(MouseButton::Auxiliary) => {
                        e.prevent_default(); // suppress middle-click autoscroll
                        capture_pointer(&e);
                        panning.set(true);
                    }
                    _ => {}
                }
            },
            onpointermove: move |e| {
                if drawing() {
                    dispatch(state, InputCommand::StrokeTo { sample: sample(state, &e) });
                } else if panning() && let Some(l) = last_position() {
                    dispatch(state, InputCommand::Pan { delta: elem_xy(&e) - l });
                }
                last_position.set(Some(elem_xy(&e)));
            },
            onpointerup: move |_| end_interaction(state, &mut drawing, &mut panning),
            onpointercancel: move |_| {
                end_interaction(state, &mut drawing, &mut panning);
                last_position.set(None);
            },
            onwheel: move |e| {
                e.prevent_default();
                let dy = e.delta().strip_units().y;
                if dy != 0.0 {
                    let factor = if dy < 0.0 { 1.15 } else { 1.0 / 1.15 };
                    let c = e.element_coordinates();
                    dispatch(state, InputCommand::Zoom { anchor: Vec2::new(c.x as f32, c.y as f32), factor });
                }
            },
        }
    }
}

#[component]
fn ColorPanel() -> Element {
    let state = use_context::<AppState>();
    // Seed from the brush's current colour (peek → no re-render on every paint).
    let init = state
        .obs
        .peek()
        .as_ref()
        .map(|o| o.brush.color)
        .unwrap_or([0.85, 0.15, 0.1, 1.0]);
    rsx! {
        OklabPicker {
            init: [init[0], init[1], init[2]],
            onchange: move |rgb: [f32; 3]| {
                update_brush(state, move |br| {
                    br.color = [rgb[0], rgb[1], rgb[2], br.color[3]];
                });
            },
        }
    }
}

/// Reusable Oklab colour picker: a vertical `L` slider + a 2D `a`/`b` field. Seeds its
/// Oklab state from `init` (straight sRGB) when mounted and reports every pick through
/// `onchange` as straight sRGB, gamut-clamped. Signals are `Copy`, so they can be handed
/// to several event closures and the free helpers below. Used by the Color panel (brush
/// colour) and the Lighting panel's canvas-colour pop-out.
#[component]
fn OklabPicker(init: [f32; 3], onchange: EventHandler<[f32; 3]>) -> Element {
    let lab = srgb_to_oklab([init[0], init[1], init[2], 1.0]);
    let l = use_signal(|| lab[0]);
    let a = use_signal(|| lab[1]);
    let b = use_signal(|| lab[2]);
    let mut picking_ab = use_signal(|| false);
    let mut picking_l = use_signal(|| false);

    // The a/b field is the colour plane at the current `L`; it only depends on `L`, so
    // memoize it (no rebuild while dragging in the field, which moves only `a`/`b`).
    let field = use_memo(move || ab_field_data_url(l()));

    let ax = (a() / AB * 0.5 + 0.5) * FIELD_PX; // a: −AB→left, +AB→right
    let by = (0.5 - b() / AB * 0.5) * FIELD_PX; // b: +AB→top (warm), −AB→bottom (cool)
    let ly = (1.0 - l()) * L_H; // L: 1→top, 0→bottom
    // Exact 1-D oklab gradient for the L track at the current chroma (CSS interpolates
    // in oklab when asked, so the ramp is perceptually even).
    let l_grad =
        format!("linear-gradient(in oklab to top, oklab(0 {a:.4} {b:.4}), oklab(1 {a:.4} {b:.4}))", a = a(), b = b());

    rsx! {
        div { class: "color-pick",
            div {
                class: "ab-field",
                style: "background-image: {field()};",
                // Pointer capture: the drag keeps tracking while the button is held,
                // even outside the field (picks clamp to the gamut box).
                onpointerdown: move |e| { capture_pointer(&e); picking_ab.set(true); pick_ab(onchange, a, b, l, &e); },
                onpointermove: move |e| { if picking_ab() { pick_ab(onchange, a, b, l, &e); } },
                onpointerup: move |_| picking_ab.set(false),
                onpointercancel: move |_| picking_ab.set(false),
                div { class: "ab-marker", style: "left:{ax}px; top:{by}px;" }
            }
            div {
                class: "l-slider",
                style: "background: {l_grad};",
                onpointerdown: move |e| { capture_pointer(&e); picking_l.set(true); pick_l(onchange, l, a, b, &e); },
                onpointermove: move |e| { if picking_l() { pick_l(onchange, l, a, b, &e); } },
                onpointerup: move |_| picking_l.set(false),
                onpointercancel: move |_| picking_l.set(false),
                div { class: "l-marker", style: "top:{ly}px;" }
            }
        }
    }
}

/// Report the current Oklab `(L, a, b)` through `onchange` as straight sRGB.
/// Out-of-gamut points clamp to sRGB.
fn apply_color(onchange: EventHandler<[f32; 3]>, l: Signal<f32>, a: Signal<f32>, b: Signal<f32>) {
    let rgba = oklab_to_srgb([l(), a(), b(), 1.0]);
    onchange.call([
        rgba[0].clamp(0.0, 1.0),
        rgba[1].clamp(0.0, 1.0),
        rgba[2].clamp(0.0, 1.0),
    ]);
}

/// Set `a`/`b` from a pointer position over the field (warm/+b at top), then apply.
fn pick_ab(onchange: EventHandler<[f32; 3]>, mut a: Signal<f32>, mut b: Signal<f32>, l: Signal<f32>, e: &Event<PointerData>) {
    let c = e.element_coordinates();
    a.set(((c.x as f32 / FIELD_PX) * 2.0 - 1.0).clamp(-1.0, 1.0) * AB);
    b.set((1.0 - (c.y as f32 / FIELD_PX) * 2.0).clamp(-1.0, 1.0) * AB);
    apply_color(onchange, l, a, b);
}

/// Set `L` from a pointer position over the vertical slider (top = light), then apply.
fn pick_l(onchange: EventHandler<[f32; 3]>, mut l: Signal<f32>, a: Signal<f32>, b: Signal<f32>, e: &Event<PointerData>) {
    let c = e.element_coordinates();
    l.set((1.0 - c.y as f32 / L_H).clamp(0.0, 1.0));
    apply_color(onchange, l, a, b);
}

/// Render the Oklab `a`/`b` colour plane at lightness `l` as a small 24-bit BMP
/// `data:` URL (CSS scales it up). `a` runs left→right (−AB→+AB), `b` runs top→bottom
/// (+AB→−AB), so warm colours sit at the top and cool at the bottom. Out-of-gamut
/// colours clamp to sRGB. Cheap enough to recompute whenever `L` changes.
fn ab_field_data_url(l: f32) -> String {
    let n = FIELD_N;
    let pixels = n * n * 3;
    let mut bmp = Vec::with_capacity(54 + pixels);
    // BITMAPFILEHEADER (14) + BITMAPINFOHEADER (40).
    bmp.extend_from_slice(b"BM");
    bmp.extend_from_slice(&((54 + pixels) as u32).to_le_bytes());
    bmp.extend_from_slice(&0u32.to_le_bytes()); // reserved
    bmp.extend_from_slice(&54u32.to_le_bytes()); // pixel data offset
    bmp.extend_from_slice(&40u32.to_le_bytes()); // info header size
    bmp.extend_from_slice(&(n as i32).to_le_bytes()); // width
    bmp.extend_from_slice(&(n as i32).to_le_bytes()); // height (+ → bottom-up)
    bmp.extend_from_slice(&1u16.to_le_bytes()); // planes
    bmp.extend_from_slice(&24u16.to_le_bytes()); // bpp
    bmp.extend_from_slice(&0u32.to_le_bytes()); // BI_RGB
    bmp.extend_from_slice(&(pixels as u32).to_le_bytes());
    bmp.extend_from_slice(&2835i32.to_le_bytes()); // 72 dpi
    bmp.extend_from_slice(&2835i32.to_le_bytes());
    bmp.extend_from_slice(&0u32.to_le_bytes()); // colours used
    bmp.extend_from_slice(&0u32.to_le_bytes()); // important
    let last = (n - 1) as f32;
    for row in 0..n {
        let y = n - 1 - row; // BMP rows are bottom-up; `y` is from the top
        let bb = AB * (1.0 - 2.0 * y as f32 / last); // top → +AB (warm)
        for x in 0..n {
            let aa = AB * (2.0 * x as f32 / last - 1.0); // left → −AB
            let rgb = oklab_to_srgb([l, aa, bb, 1.0]);
            let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            bmp.push(q(rgb[2])); // B
            bmp.push(q(rgb[1])); // G
            bmp.push(q(rgb[0])); // R
        }
    }
    format!("url(data:image/bmp;base64,{})", base64(&bmp))
}

/// Standard base64 (with padding) — small, so a data URL stays dependency-free.
fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let n = (chunk[0] as u32) << 16
            | (*chunk.get(1).unwrap_or(&0) as u32) << 8
            | (*chunk.get(2).unwrap_or(&0) as u32);
        out.push(T[(n >> 18 & 63) as usize] as char);
        out.push(T[(n >> 12 & 63) as usize] as char);
        out.push(if chunk.len() > 1 { T[(n >> 6 & 63) as usize] as char } else { '=' });
        out.push(if chunk.len() > 2 { T[(n & 63) as usize] as char } else { '=' });
    }
    out
}

/// The floating Brush panel: the everyday quick controls (shape, size, opacity,
/// rate). Everything else — the full grouped parameter set with a live test
/// stroke — lives in the brush editor dialog ("Edit brush…").
#[component]
fn BrushPanel() -> Element {
    let state = use_context::<AppState>();
    let brush = state
        .obs
        .read()
        .as_ref()
        .map(|o| o.brush)
        .unwrap_or_default();
    let is_round = matches!(brush.shape, BrushShape::Round);

    let chip = |active: bool| if active { "chip active" } else { "chip" };

    rsx! {
        div { class: "brush-shapes",
            button {
                class: chip(is_round),
                onclick: move |_| set_shape(state, BrushShape::Round, 0.25),
                "Round"
            }
            button {
                class: chip(!is_round),
                onclick: move |_| set_bristles(state),
                "Bristles"
            }
        }
        Slider { label: "Size", min: 1.0, max: 120.0, value: brush.radius,
            oninput: move |v| update_brush(state, move |b| b.radius = v) }
        Slider { label: "Opacity", min: 0.0, max: 1.0, value: brush.color[3],
            oninput: move |v| update_brush(state, move |b| b.color[3] = v) }
        Slider { label: "Rate", min: 0.05, max: 1.0, value: brush.flow,
            oninput: move |v| update_brush(state, move |b| b.flow = v) }
        button {
            class: "be-open",
            onclick: move |_| {
                let mut open = state.brush_editor_open;
                open.set(true);
            },
            "Edit brush\u{2026}"
        }
    }
}

/// Switch to a shape, also setting a sensible default spacing for it.
fn set_shape(state: AppState, shape: BrushShape, spacing: f32) {
    update_brush(state, move |b| {
        b.shape = shape;
        b.spacing = spacing;
    });
}

/// Set what orients the brush shape as it sweeps (DESIGN.md §6.6).
fn set_orientation(state: AppState, orientation: OrientationSource) {
    update_brush(state, move |b| b.orientation = orientation);
}

/// Edit the unified brush dynamics in place (DESIGN.md §6.2).
fn set_dyn(state: AppState, f: impl FnOnce(&mut BrushDynamics)) {
    update_brush(state, move |b| f(&mut b.dynamics));
}

/// Reset to the everyday brush: lay the brush's own paint, manipulate nothing.
fn set_brush_preset(state: AppState) {
    set_dyn(state, |d| *d = BrushDynamics::default());
}

/// The palette knife (DESIGN.md §6.2): no own paint (`add = 0`), a finite pre-`charge` it
/// carries, pen pressure fully drives the scrape (`load` + `load_pressure`), and pen tilt
/// toward the motion fully drives the `deposit` (`deposit_tilt`). A hard edge + tooth so it
/// reads as a blade riding the weave.
fn set_knife(state: AppState) {
    update_brush(state, |b| {
        b.shape = BrushShape::Round;
        b.hardness = 0.9;
        b.tooth = 0.7;
        b.dynamics = BrushDynamics {
            add: 0.0,
            lift: 1.0,
            deposit: 0.6,
            charge: 0.5,
            load_pressure: 1.0,
            deposit_tilt: 1.0,
            drag: 0.0,
            bleed: 0.0,
            ridge: 0.0,
        };
    });
}

/// Select the built-in bristle brush. It's fetched + imported once at startup
/// (DESIGN.md §6.6), so this is a no-op until those bytes have loaded.
fn set_bristles(state: AppState) {
    let id = state.renderer.read().as_ref().and_then(|r| r.bristle());
    let Some(id) = id else { return };
    set_shape(state, BrushShape::Stamp(id), 0.08);
}

#[component]
fn LayerPanel() -> Element {
    let state = use_context::<AppState>();
    let layers = state
        .obs
        .read()
        .as_ref()
        .map(|o| o.layers.clone())
        .unwrap_or_default();

    rsx! {
        div { class: "layer-header",
            button {
                class: "layer-add",
                onclick: move |_| dispatch(state, InputCommand::AddLayer { above: None }),
                "+ Add"
            }
        }
        for info in layers.iter().rev().cloned() {
            LayerRow { info }
        }
    }
}

#[component]
fn LayerRow(info: LayerInfo) -> Element {
    let state = use_context::<AppState>();
    let active = state
        .obs
        .read()
        .as_ref()
        .map(|o| o.active_layer == info.id)
        .unwrap_or(false);
    let row_class = if active { "layer-row active" } else { "layer-row" };

    rsx! {
        div {
            class: row_class,
            div { class: "row",
                input {
                    r#type: "checkbox",
                    checked: info.visible,
                    onchange: move |_| dispatch(state, InputCommand::SetLayerVisible(info.id, !info.visible)),
                }
                button {
                    class: "layer-name",
                    onclick: move |_| dispatch(state, InputCommand::SetActiveLayer(info.id)),
                    "Layer {info.id.0}"
                }
            }
            input {
                class: "slider",
                r#type: "range", min: "0", max: "100",
                value: "{(info.opacity * 100.0) as i32}",
                oninput: move |e| {
                    if let Ok(v) = e.value().parse::<f32>() {
                        dispatch(state, InputCommand::SetLayerOpacity(info.id, v / 100.0));
                    }
                },
            }
        }
    }
}

/// Lighting controls for the image-based-lighting media pass (DESIGN.md §6.3).
/// The canvas is lit by the studio HDR environment; these tune how it reads.
#[component]
fn LightingPanel() -> Element {
    let state = use_context::<AppState>();
    // Seeded from the engine defaults; this panel owns the live values (lighting is
    // a view setting, not part of the observable document state).
    let media = use_signal(MediaParams::default);
    let p = media();
    // The canvas substrate colour (straight sRGB), shown as a swatch that pops out an
    // Oklab picker. Like the sliders, a view setting owned here (`Renderer::set_background`).
    let mut bg = use_signal(|| [BG.r as f32, BG.g as f32, BG.b as f32]);
    let mut show_bg_picker = use_signal(|| false);
    let c = bg();
    let swatch = format!(
        "background: rgb({:.1}% {:.1}% {:.1}%);",
        c[0] * 100.0,
        c[1] * 100.0,
        c[2] * 100.0
    );
    // The canvas surface (weave), switchable in place — the document is preserved;
    // existing paint re-reads against the new bump (DESIGN.md §6.4). Reading the
    // renderer signal keeps the drop-down in sync after the async switch lands.
    let surf = state
        .renderer
        .read()
        .as_ref()
        .map(|r| r.surface())
        .unwrap_or_default();

    rsx! {
        Slider { label: "Exposure", min: 0.1, max: 2.0, value: p.exposure,
            oninput: move |v| update_media(state, media, move |m| m.exposure = v) }
        Slider { label: "Relief", min: 0.0, max: 0.6, value: p.height_strength,
            oninput: move |v| update_media(state, media, move |m| m.height_strength = v) }
        Slider { label: "Weave", min: 0.0, max: 1.5, value: p.surface_strength,
            oninput: move |v| update_media(state, media, move |m| m.surface_strength = v) }
        Slider { label: "Wet gloss", min: 0.0, max: 0.35, value: p.specular,
            oninput: move |v| update_media(state, media, move |m| m.specular = v) }
        div { class: "slider-row",
            div { class: "slider-label", "Canvas" }
            button {
                class: "swatch",
                style: "{swatch}",
                onclick: move |_| show_bg_picker.set(!show_bg_picker()),
            }
        }
        div { class: "slider-row",
            div { class: "slider-label", "Surface" }
            select {
                class: "select",
                onchange: move |e| {
                    if let Some((id, _)) = SURFACES.iter().find(|(s, _)| format!("{s:?}") == e.value()) {
                        set_surface(state, *id);
                    }
                },
                for (id, name) in SURFACES.iter().copied() {
                    option { value: "{id:?}", selected: surf == id, "{name}" }
                }
            }
        }
        // Pop-out colour selector: mounted only while open, so the picker re-seeds from
        // the current colour each time. Positioned by `.color-popout` (flies out beside
        // the panel, whose `.panel` is the nearest positioned ancestor).
        if show_bg_picker() {
            div { class: "color-popout",
                OklabPicker {
                    init: bg(),
                    onchange: move |rgb: [f32; 3]| {
                        bg.set(rgb);
                        update_background(state, rgb);
                    },
                }
            }
        }
    }
}

/// Mutate the lighting params in place, push them to the engine, and repaint.
fn update_media(state: AppState, mut media: Signal<MediaParams>, f: impl FnOnce(&mut MediaParams)) {
    let mut p = media();
    f(&mut p);
    media.set(p);
    let mut renderer = state.renderer;
    let mut guard = renderer.write();
    if let Some(r) = guard.as_mut() {
        r.set_media_params(p);
        r.paint();
    }
}

/// Set the canvas substrate colour (straight sRGB, a view setting) and repaint.
fn update_background(state: AppState, rgb: [f32; 3]) {
    let mut renderer = state.renderer;
    let mut guard = renderer.write();
    if let Some(r) = guard.as_mut() {
        r.set_background(rgb);
        r.paint();
    }
}

/// The bundled asset behind an image-backed surface (`None` for procedural ones,
/// which need no bytes). The one place to map a new [`SURFACES`] row to its file.
fn surface_asset(id: SurfaceId) -> Option<Asset> {
    match id {
        SurfaceId::Flat => None,
        SurfaceId::Linen => Some(SURFACE_LINEN),
    }
}

/// Switch the canvas surface in place and repaint — the document is preserved;
/// existing paint re-reads against the new weave (DESIGN.md §6.4). Image-backed
/// surfaces are fetched on first use (the bump maps stay out of the wasm binary),
/// so this runs async, like `new_document`'s fetch.
fn set_surface(state: AppState, id: SurfaceId) {
    let mut renderer = state.renderer;
    spawn(async move {
        let needs_bytes = renderer
            .read()
            .as_ref()
            .is_some_and(|r| !r.surface_loaded(id));
        if needs_bytes && let Some(asset) = surface_asset(id) {
            tracing::info!(surface = ?id, url = %asset, "fetching surface asset");
            match dioxus::asset_resolver::read_asset_bytes(asset).await {
                Ok(bytes) => {
                    if let Some(r) = renderer.write().as_mut() {
                        r.register_surface(id, bytes);
                    }
                }
                Err(e) => {
                    tracing::warn!("surface fetch failed: {e}");
                    return;
                }
            }
        }
        if let Some(r) = renderer.write().as_mut() {
            r.set_surface(id);
            r.paint();
        }
    });
}

/// A vertical menu rail on the far left for uncommon or keyboard-driven commands
/// (DESIGN.md §11). Built on the vendored `menubar` component; the dropdown flies
/// out to the right. Undo/Redo live here purely to advertise their Ctrl+Z / Ctrl+Y
/// shortcuts (the everyday way to invoke them); "New document…" opens a modal.
#[component]
fn CommandRail() -> Element {
    let state = use_context::<AppState>();
    let layout = use_context::<PanelLayout>();
    let mut show_new_doc = use_signal(|| false);
    let mut show_session = use_signal(|| false);
    let live = (state.collab_phase)() == collab::CollabPhase::Shared;
    let (can_undo, can_redo) = state
        .obs
        .read()
        .as_ref()
        .map(|o| (o.can_undo, o.can_redo))
        .unwrap_or((false, false));
    let hidden = (layout.hidden)();

    rsx! {
        div { class: "command-rail",
            Menubar {
                MenubarMenu { index: 0usize,
                    // ☰ — the catch-all menu for infrequent commands.
                    MenubarTrigger { "\u{2630}" }
                    MenubarContent {
                        MenubarItem {
                            index: 2usize,
                            value: "new-document".to_string(),
                            on_select: move |_| show_new_doc.set(true),
                            span { "New document…" }
                        }
                        MenubarItem {
                            index: 3usize,
                            value: "shared-drawing".to_string(),
                            on_select: move |_| show_session.set(true),
                            span { if live { "Shared drawing \u{25CF}" } else { "Shared drawing…" } }
                        }
                        MenubarItem {
                            index: 0usize,
                            value: "undo".to_string(),
                            disabled: !can_undo,
                            on_select: move |_| dispatch(state, InputCommand::Undo),
                            span { "Undo" }
                            span { class: "menu-shortcut", "Ctrl+Z" }
                        }
                        MenubarItem {
                            index: 1usize,
                            value: "redo".to_string(),
                            disabled: !can_redo,
                            on_select: move |_| dispatch(state, InputCommand::Redo),
                            span { "Redo" }
                            span { class: "menu-shortcut", "Ctrl+Y" }
                        }
                    }
                }
                MenubarMenu { index: 1usize,
                    // ▤ — toggle which floating panels are shown.
                    MenubarTrigger { "\u{25A4}" }
                    MenubarContent {
                        for (i, id) in PanelId::ALL.into_iter().enumerate() {
                            MenubarItem {
                                index: i,
                                value: format!("panel-{id:?}"),
                                on_select: move |_| {
                                    let mut hidden = layout.hidden;
                                    let mut h = hidden.write();
                                    if !h.remove(&id) { h.insert(id); }
                                },
                                span { "{id.title()}" }
                                span { class: "menu-check",
                                    if hidden.contains(&id) { "" } else { "\u{2713}" }
                                }
                            }
                        }
                    }
                }
            }
        }
        if show_new_doc() {
            NewDocumentModal { on_close: move |_| show_new_doc.set(false) }
        }
        if show_session() {
            collab::SessionModal { on_close: move |_| show_session.set(false) }
        }
    }
}

/// Modal for starting a fresh document. Today it carries the color-space choice
/// (DESIGN.md §6.7); it's a dialog so more document settings can join it later.
#[component]
fn NewDocumentModal(on_close: EventHandler<()>) -> Element {
    let state = use_context::<AppState>();
    let current = state
        .renderer
        .read()
        .as_ref()
        .map(|r| r.color_space())
        .unwrap_or(ColorSpaceId::Oklab);
    let choice = use_signal(|| current);

    let current_surface = state
        .renderer
        .read()
        .as_ref()
        .map(|r| r.surface())
        .unwrap_or_default();
    let surf_choice = use_signal(|| current_surface);

    // One selectable color-space card; `selected` toggles the highlight.
    let card = |id: ColorSpaceId, title: &str, desc: &str| {
        let class = if choice() == id { "space-card selected" } else { "space-card" };
        rsx! {
            div {
                class,
                onclick: move |_| { let mut choice = choice; choice.set(id); },
                div { class: "space-card-title", "{title}" }
                div { class: "space-card-desc", "{desc}" }
            }
        }
    };

    // Same card, for the canvas surface choice.
    let scard = |id: SurfaceId, title: &str, desc: &str| {
        let class = if surf_choice() == id { "space-card selected" } else { "space-card" };
        rsx! {
            div {
                class,
                onclick: move |_| { let mut c = surf_choice; c.set(id); },
                div { class: "space-card-title", "{title}" }
                div { class: "space-card-desc", "{desc}" }
            }
        }
    };

    rsx! {
        // Dimmed backdrop; click outside the dialog to dismiss.
        div {
            class: "modal-backdrop",
            onclick: move |_| on_close.call(()),
            div {
                class: "modal-dialog",
                onclick: move |e| e.stop_propagation(),

                div { class: "modal-title", "New Document" }
                div { class: "modal-subtitle", "Starting a new document replaces the current canvas." }

                div { class: "modal-section-label", "COLOR SPACE" }
                {card(ColorSpaceId::Oklab, "Oklab", "Perceptual color with smooth, predictable blending. The standard choice for digital painting.")}
                {card(ColorSpaceId::Mixbox, "Mixbox", "Realistic pigment mixing (Mixbox): blue + yellow makes green, like real paint. For natural media.")}

                div { class: "modal-section-label", "SURFACE" }
                {scard(SurfaceId::Flat, "Smooth", "A perfectly smooth surface — paint lies flat, no canvas texture.")}
                {scard(SurfaceId::Linen, "Canvas", "Linen weave: dry strokes catch on the tooth and the weave catches the light.")}

                div { class: "modal-actions",
                    button {
                        class: "btn btn-secondary",
                        onclick: move |_| on_close.call(()),
                        "Cancel"
                    }
                    button {
                        class: "btn btn-primary",
                        onclick: move |_| new_document(state, choice(), surf_choice(), on_close),
                        "Create"
                    }
                }
            }
        }
    }
}

/// Replace the document with a fresh one in the chosen color space and surface,
/// then repaint. Image-backed surfaces are fetched on first use (the large bump
/// maps stay out of the wasm binary — DESIGN.md §6.6), so this runs async.
///
/// It owns closing the modal (`on_close`), calling it only once the work is done.
/// Crucial: `spawn` ties the task to the *calling component's* scope, so if the
/// modal closed synchronously first, unmounting it would cancel this task mid-
/// fetch (silently — the symptom: the surface never applies). Keeping the modal
/// mounted until completion keeps the task alive.
fn new_document(state: AppState, color: ColorSpaceId, surface: SurfaceId, on_close: EventHandler<()>) {
    let mut renderer = state.renderer;
    let mut obs = state.obs;
    // Replacing the document abandons any shared session (and clears the
    // ticket from the URL) — the fresh canvas is private until re-shared.
    collab::leave(state);
    spawn(async move {
        // Fetch + register the surface bytes the first time it's chosen
        // (procedural surfaces have no asset — see `surface_asset`).
        let needs_bytes = renderer
            .read()
            .as_ref()
            .is_some_and(|r| !r.surface_loaded(surface));
        if needs_bytes && let Some(asset) = surface_asset(surface) {
            tracing::info!(?surface, url = %asset, "fetching surface asset");
            match dioxus::asset_resolver::read_asset_bytes(asset).await {
                Ok(bytes) => {
                    tracing::info!(?surface, bytes = bytes.len(), "surface fetched; registering");
                    if let Some(r) = renderer.write().as_mut() {
                        r.register_surface(surface, bytes);
                    }
                }
                Err(e) => {
                    tracing::warn!("surface fetch failed: {e}");
                    on_close.call(());
                    return;
                }
            }
        }

        if let Some(r) = renderer.write().as_mut() {
            r.set_color_space(color);
            r.set_surface(surface);
            r.paint();
            obs.set(Some(r.observe()));
        }
        tracing::info!(?color, ?surface, "new document ready");
        on_close.call(());
    });
}

// --- reusable chrome ---

/// The floating tool panels, top-right. Data-driven: renders `layout.order` minus the
/// hidden set, each wrapped in the unified [`Panel`] chrome (keyed by id so reordering
/// moves nodes rather than recreating them — preserves per-panel state and, later,
/// enables the drag animation).
#[component]
fn PanelStack() -> Element {
    let layout = use_context::<PanelLayout>();
    let hidden = (layout.hidden)();
    rsx! {
        div { class: "panel-stack",
            for id in (layout.order)() {
                if !hidden.contains(&id) {
                    Panel { key: "{id:?}", id,
                        match id {
                            PanelId::Color => rsx! { ColorPanel {} },
                            PanelId::Brush => rsx! { BrushPanel {} },
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
fn Panel(id: PanelId, children: Element) -> Element {
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
            let trans = if tracking { "none" } else { "transform 180ms ease" };
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
fn start_drag(layout: PanelLayout, id: PanelId, e: &Event<PointerData>) {
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
fn drag_move(layout: PanelLayout, e: &Event<PointerData>) {
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
fn drag_end(layout: PanelLayout) {
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
fn commit_drag(layout: PanelLayout) {
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

/// Resolve after `ms` milliseconds (so a settle animation can finish before the order is
/// committed). Browser `setTimeout` on web; a no-op off-wasm.
#[cfg(target_arch = "wasm32")]
async fn sleep_ms(ms: i32) {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        let _ = web_sys::window()
            .expect("window")
            .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, ms);
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}
#[cfg(not(target_arch = "wasm32"))]
async fn sleep_ms(_ms: i32) {}

/// Capture the pointer for the element under `e`, so the in-progress drag keeps
/// streaming move/up events to it while the button is held — even after the pointer
/// leaves the element. The capture releases automatically on pointer-up, which is
/// guaranteed to be delivered to the capturing element.
#[cfg(target_arch = "wasm32")]
fn capture_pointer(e: &Event<PointerData>) {
    use dioxus::web::WebEventExt;
    use wasm_bindgen::JsCast;
    if let Some(ev) = e.try_as_web_event()
        && let Some(target) = ev.target().and_then(|t| t.dyn_into::<web_sys::Element>().ok())
    {
        let _ = target.set_pointer_capture(ev.pointer_id());
    }
}
#[cfg(not(target_arch = "wasm32"))]
fn capture_pointer(_e: &Event<PointerData>) {}

#[component]
fn Slider(label: String, min: f32, max: f32, value: f32, oninput: EventHandler<f32>) -> Element {
    rsx! {
        div { class: "slider-row",
            div { class: "slider-label", "{label}" }
            input {
                class: "slider",
                r#type: "range", min: "{min}", max: "{max}", step: "any", value: "{value}",
                oninput: move |e| {
                    if let Ok(v) = e.value().parse::<f32>() { oninput.call(v); }
                },
            }
        }
    }
}

/// Ternary (barycentric) pad: drag a marker inside a triangle whose vertices are the
/// pure axes (`labels[0]` top, `labels[1]` bottom-left, `labels[2]` bottom-right).
/// Reports the marker's barycentric coordinates — weights ≥ 0 summing to 1 — so three
/// knobs whose common scale is redundant (overactuated against a separate rate/strength
/// control) collapse to the two degrees of freedom that matter. Controlled: the marker
/// tracks `value` (normalized defensively, so legacy non-normalized params display
/// sensibly) and every drag reports through `onchange`. Used for the dry brush's
/// add/lift/deposit; the wet brush's mix gets the same treatment (DESIGN.md §6.2).
#[component]
fn TernaryPad(labels: [String; 3], value: [f32; 3], onchange: EventHandler<[f32; 3]>) -> Element {
    let mut picking = use_signal(|| false);

    // Marker position from the (normalized) weights: p = Σ wᵢ·Vᵢ over the triangle's
    // vertices V₀=(W/2, 0), V₁=(0, H), V₂=(W, H), shifted down by the label band.
    let s: f32 = value.iter().sum();
    let v = if s > 1e-4 { value.map(|x| x / s) } else { [1.0, 0.0, 0.0] };
    let mx = v[0] * TRI_W * 0.5 + v[2] * TRI_W;
    let my = (v[1] + v[2]) * TRI_H + TRI_LBL;

    let pick = move |e: &Event<PointerData>| {
        let c = e.element_coordinates();
        onchange.call(ternary_weights(c.x as f32, c.y as f32 - TRI_LBL));
    };

    rsx! {
        div {
            class: "ternary",
            // Pointer capture keeps the drag streaming here while the button is held,
            // even outside the pad (weights clamp onto the simplex); the drag ends on
            // up/cancel, never on leaving the bounds.
            onpointerdown: move |e| { capture_pointer(&e); picking.set(true); pick(&e); },
            onpointermove: move |e| { if picking() { pick(&e); } },
            onpointerup: move |_| picking.set(false),
            onpointercancel: move |_| picking.set(false),
            div { class: "ternary-tri" }
            div { class: "ternary-label ternary-top", "{labels[0]}" }
            div { class: "ternary-label ternary-left", "{labels[1]}" }
            div { class: "ternary-label ternary-right", "{labels[2]}" }
            div { class: "ternary-marker", style: "left:{mx}px; top:{my}px;" }
        }
    }
}

/// Barycentric weights of a pointer position in the ternary triangle's local space
/// (origin at the label band's bottom-left, vertices as in [`TernaryPad`]). Positions
/// outside the triangle clamp onto it: negative weights drop to 0 and the rest
/// renormalize — so dragging past an edge or vertex pins the opposite weights to
/// exactly 0, which is also how a pure single- or two-axis mix is dialled in.
fn ternary_weights(px: f32, py: f32) -> [f32; 3] {
    let (x0, y0) = (TRI_W * 0.5, 0.0f32);
    let (x1, y1) = (0.0f32, TRI_H);
    let (x2, y2) = (TRI_W, TRI_H);
    let denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
    let w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom;
    let w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom;
    let w = [w0, w1, 1.0 - w0 - w1].map(|x| x.max(0.0));
    let s: f32 = w.iter().sum();
    if s > 0.0 { w.map(|x| x / s) } else { [1.0, 0.0, 0.0] }
}

// --- command dispatch ---

/// Apply a command, repaint the surface, and refresh the observable snapshot.
/// In a shared session, whatever the command committed is then broadcast.
fn dispatch(state: AppState, command: InputCommand) {
    let mut renderer = state.renderer;
    let mut obs = state.obs;
    {
        let mut guard = renderer.write();
        if let Some(r) = guard.as_mut() {
            r.process(command);
            r.paint();
            obs.set(Some(r.observe()));
        }
    }
    collab::flush_outbox(state);
}

/// Resize the surface/engine, then repaint.
fn resize(state: AppState, width: u32, height: u32) {
    let mut renderer = state.renderer;
    let mut obs = state.obs;
    let mut guard = renderer.write();
    if let Some(r) = guard.as_mut() {
        r.resize(width, height);
        r.paint();
        obs.set(Some(r.observe()));
    }
}

/// Read the current brush, mutate a copy, and commit it (releasing the `obs`
/// read guard before `dispatch` writes — avoids an AlreadyBorrowed panic).
fn update_brush(state: AppState, f: impl FnOnce(&mut BrushParams)) {
    let brush = state.obs.read().as_ref().map(|o| o.brush);
    if let Some(mut brush) = brush {
        f(&mut brush);
        dispatch(state, InputCommand::SetBrush(brush));
    }
}

fn handle_keydown(mut state: AppState, e: &Event<KeyboardData>) {
    match e.key() {
        Key::Character(c) if c.eq_ignore_ascii_case(" ") => {
            state.space_down.set(true);
            e.prevent_default();
        }
        _ => {}
    }

    let m = e.modifiers();
    if !(m.contains(Modifiers::CONTROL) || m.contains(Modifiers::META)) {
        return;
    }
    match e.key() {
        Key::Character(c) if c.eq_ignore_ascii_case("z") => {
            let cmd = if m.contains(Modifiers::SHIFT) {
                InputCommand::Redo
            } else {
                InputCommand::Undo
            };
            dispatch(state, cmd);
            e.prevent_default();
        }
        Key::Character(c) if c.eq_ignore_ascii_case("y") => dispatch(state, InputCommand::Redo),
        _ => {}
    }
}

fn handle_keyup(mut state: AppState, e: &Event<KeyboardData>) {
    match e.key() {
        Key::Character(c) if c.eq_ignore_ascii_case(" ") => {
            state.space_down.set(false);
            e.prevent_default();
        }
        _ => {}
    }
}

/// Pointer position within an element, in CSS pixels.
fn elem_xy(e: &Event<PointerData>) -> Vec2 {
    let ElementPoint { x, y, .. } = e.element_coordinates();
    Vec2::new(x as f32, y as f32)
}

/// Map an element-relative pointer position to a canvas-space input sample.
fn sample(state: AppState, e: &Event<PointerData>) -> InputSample {
    let view = state
        .renderer
        .read()
        .as_ref()
        .map(|r| r.view())
        .expect("renderer ready during input");
    InputSample {
        pos: view.screen_to_canvas(elem_xy(&e)),
        pressure: e.pressure(),
        // Pen tilt (degrees from vertical, ±90 per axis) → a canvas-space lean vector. The
        // palette knife's deposit reads its component along the stroke direction (DESIGN
        // §6.2); a mouse reports (0, 0), so the deposit falls back to its constant rate.
        tilt: Vec2::new(e.tilt_x() as f32, e.tilt_y() as f32) / 90.0,
        ..Default::default()
    }
}

/// End any in-progress stroke or pan.
fn end_interaction(
    state: AppState,
    drawing: &mut Signal<bool>,
    panning: &mut Signal<bool>,
) {
    if drawing() {
        dispatch(state, InputCommand::EndStroke);
        drawing.set(false);
    }
    panning.set(false);
}

