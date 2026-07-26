//! The floating Color panel: an Oklab picker over an `a`/`b` plane at a chosen `L`.

use dioxus::prelude::*;

use crate::platform::capture_pointer;
use crate::state::{AppState, update_brush};
use stark_core::color::{oklab_to_srgb, srgb_to_oklab};

#[component]
pub fn ColorPanel() -> Element {
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

/// Reusable Oklab colour picker: a vertical `L` slider + a 2D `a`/`b` field. Seeds its
/// Oklab state from `init` (straight sRGB) when mounted and reports every pick through
/// `onchange` as straight sRGB, gamut-clamped. Signals are `Copy`, so they can be handed
/// to several event closures and the free helpers below. Used by the Color panel (brush
/// colour) and the Lighting panel's canvas-colour pop-out.
#[component]
pub fn OklabPicker(init: [f32; 3], onchange: EventHandler<[f32; 3]>) -> Element {
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
    let l_grad = format!(
        "linear-gradient(in oklab to top, oklab(0 {a:.4} {b:.4}), oklab(1 {a:.4} {b:.4}))",
        a = a(),
        b = b()
    );

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
fn pick_ab(
    onchange: EventHandler<[f32; 3]>,
    mut a: Signal<f32>,
    mut b: Signal<f32>,
    l: Signal<f32>,
    e: &Event<PointerData>,
) {
    let c = e.element_coordinates();
    a.set(((c.x as f32 / FIELD_PX) * 2.0 - 1.0).clamp(-1.0, 1.0) * AB);
    b.set((1.0 - (c.y as f32 / FIELD_PX) * 2.0).clamp(-1.0, 1.0) * AB);
    apply_color(onchange, l, a, b);
}

/// Set `L` from a pointer position over the vertical slider (top = light), then apply.
fn pick_l(
    onchange: EventHandler<[f32; 3]>,
    mut l: Signal<f32>,
    a: Signal<f32>,
    b: Signal<f32>,
    e: &Event<PointerData>,
) {
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
        out.push(if chunk.len() > 1 {
            T[(n >> 6 & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            T[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}
