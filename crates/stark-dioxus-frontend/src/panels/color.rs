//! The floating Color panel: an Oklab picker — a wheel carrying every color the
//! display can hold at a chosen lightness, and the lightness beside it.
//!
//! The eyedropper writes the same brush color this picker does, but its *options*
//! are not here — they live in a bar that comes up on Alt
//! ([`crate::panels::pick`]), since it is a modifier rather than a tool. What the
//! panel owes it is `seed`: a pick sets the color from outside the picker, and the
//! markers have to follow.

use dioxus::prelude::*;

use crate::platform::{capture_pointer, pointer_fraction};
use crate::state::{AppState, update_brush};
use stark_model::color::Gamut;
use stark_ui::color::{
    FIELD_N, Grab, RAMP_N, ab_field_rgb, notation_of, on_wheel, parse_color, ramp_rgb, wheel_color,
    wheel_rgb, wheel_xy,
};

/// The gamut the wheel is fitted to (§6.5) — **the picture carrier's, not the
/// canvas's**. The engine paints in any gamut the surface has, and this control's
/// pictures are `data:` URLs a browser reads as sRGB, so a wider rim here would draw
/// an outer ring of colors clamped on their way to the screen — the very thing the
/// fit exists to remove. A wide color is reachable meanwhile by typing it
/// (`stark_ui::color::parse_color`); a tagged carrier is what would move this
/// (§11, the wide-gamut wheel).
const WHEEL_GAMUT: Gamut = Gamut::Srgb;

#[component]
pub fn ColorPanel() -> Element {
    let state = use_context::<AppState>();
    // Seed from the hand's current color (peek → no re-render on every paint).
    // The hand's is the paint side's, whatever effect is in force — which is
    // what lets a color picked while the eraser is held land somewhere
    // (`BrushConfig::paint`).
    let init = state.transient.peek().color;
    // Read reactively, unlike the color: this is how a pick — which sets the color
    // from outside the picker — gets the markers to move (see `AppState::color_epoch`).
    let seed = (state.color_epoch)();

    rsx! {
        OklabPicker {
            init,
            seed,
            onchange: move |rgb: [f32; 3]| {
                update_brush(state, move |_, t| t.color = rgb);
            },
        }
    }
}

// The picker's on-screen sizes are the stylesheet's alone (`.color-wheel`,
// `.l-slider`): the markers are placed in percentages and a pick reads the
// pointer as a fraction of the element it landed in (`platform::pointer_fraction`),
// so nothing here has to mirror a px value — which is what lets each of the
// picker's homes size the wheel with one CSS rule and no second copy to drift.

/// What a press means, given where the marker stood — the crate's, and its `place`
/// is the whole of a drag. What stays here is reading the modifier off a DOM event,
/// which is the only part of it this frontend knows.
fn grab(e: &Event<PointerData>, held: (f32, f32)) -> Grab {
    let at = pointer_fraction(e).unwrap_or(held);
    Grab::take(at, held, e.modifiers().contains(Modifiers::SHIFT))
}

/// Reusable Oklab color picker: a wheel of hue and chroma at one lightness, a
/// horizontal `L` slider under it, and a readout of what has been chosen. Seeds its
/// state from `init` (straight sRGB) when mounted and reports every pick through
/// `onchange` as straight sRGB. Signals are `Copy`, so they can be handed to several
/// event closures and the free helpers below. Used by the Color panel (brush color),
/// the frame bar's matte well and the Lighting panel's canvas-color pop-out.
///
/// **The wheel is fitted to the gamut, and that is the whole of why it is a wheel.**
/// The picker used to draw a fixed square of the Oklab `a`/`b` plane, ±0.32 on each
/// axis, which is the box the *whole* gamut fits in — but one lightness is a thin
/// slice of that box, and the rest of it is colors sRGB cannot show, drawn clamped.
/// At `L = 0.61` the slice is 28% of the square; at `L = 0.2`, 3.8%. So most of the
/// picker was a flat wash answering every position with the same color, and
/// everything the artist was choosing between was crowded into what was left —
/// which is exactly the complaint, *it moves too fast and I cannot see what I have*.
/// Here the radius is chroma as a fraction of what this lightness and this hue can
/// hold, so the rim *is* the gamut boundary, every point inside it is a distinct
/// color the display can show, and the same panel gives the choice between 2.8× and
/// 20× the room.
///
/// Two things follow that are worth having on their own. Dragging `L` now travels
/// along a hue at constant relative chroma instead of walking a fixed `(a, b)` in
/// and out of gamut, which is the move a painter makes constantly — *the same color,
/// lighter*. And no state this picker can hold is out of gamut, so nothing it shows
/// is a lie about what the brush will lay down.
///
/// `oncommit` fires once when the pointer is released, with the final color. The
/// two exist separately because a drag reports a color per pointer *move* while
/// being one edit: a caller feeding history hangs the unlogged preview off
/// `onchange` and the single commit off `oncommit`. Omitting it is right for a
/// caller whose color is not historized at all — the brush's, which is view state.
///
/// `seed` re-seeds the markers from `init`. The picker is *seeded* rather than
/// driven — it holds a hue that survives a trip to the achromatic axis, and `init`
/// comes back through sRGB, which cannot say what hue a grey is — so a caller that
/// sets the color some other way (the eyedropper) has to say so. Keyed on a counter
/// rather than on `init` itself, and deliberately: reseeding whenever the color
/// changed would spin a marker the user has dragged to the centre back to hue zero,
/// under their own cursor.
#[component]
pub fn OklabPicker(
    init: [f32; 3],
    onchange: EventHandler<[f32; 3]>,
    #[props(default)] oncommit: Option<EventHandler<[f32; 3]>>,
    #[props(default)] seed: u64,
) -> Element {
    let (il, ih, is) = on_wheel(WHEEL_GAMUT, init, 0.0);
    let mut l = use_signal(|| il);
    let mut hue = use_signal(|| ih);
    let mut sat = use_signal(|| is);
    let mut wheel_grab = use_signal(|| None::<Grab>);
    let mut l_grab = use_signal(|| None::<Grab>);
    // What is in the hex field while it is being typed in. `None` — the resting
    // state — is the field showing the color, live, including mid-drag.
    let mut draft = use_signal(|| None::<String>);

    // `init` is a plain prop, so `use_reactive!` is what makes a change in it visible
    // to an effect at all. Both are dependencies, but only a moved `seed` reseeds:
    // `init` is here so the effect reads the color of the render it fires on rather
    // than the one it was created on.
    let mut seeded = use_signal(|| seed);
    use_effect(use_reactive!(|seed, init| {
        if seed == *seeded.peek() {
            return;
        }
        seeded.set(seed);
        let (nl, nh, ns) = on_wheel(WHEEL_GAMUT, init, *hue.peek());
        l.set(nl);
        hue.set(nh);
        sat.set(ns);
    }));

    // The wheel is the gamut at the current `L`, so it only depends on `L` — memoize
    // it, and no drag on the wheel itself rebuilds it. The ramp is the other way
    // round: it is this hue at this chroma fraction, drawn through every lightness,
    // so it follows the wheel and not the slider.
    let wheel = use_memo(move || wheel_data_url(l()));
    let ramp = use_memo(move || l_ramp_data_url(hue(), sat()));

    // Percentages of each control's own box, whatever size the stylesheet gave it.
    let (mx, my) = wheel_xy(hue(), sat());
    let (wx, wy) = (mx * 100.0, my * 100.0);
    let lx = l() * 100.0; // L: 0→left, 1→right
    let rgb = wheel_color(WHEEL_GAMUT, l(), hue(), sat());
    let well = format!(
        "background: rgb({:.2}% {:.2}% {:.2}%);",
        rgb[0] * 100.0,
        rgb[1] * 100.0,
        rgb[2] * 100.0
    );
    let shown = draft().unwrap_or_else(|| notation_of(rgb));

    rsx! {
        div { class: "color-pick",
            div {
                class: "color-wheel",
                style: "background-image: {wheel()};",
                // The cursor goes away for the duration of a drag: it is an arrow
                // sitting on the one pixel the whole control is about. Under pointer
                // capture the hand cannot lose the wheel, so there is nothing left
                // for it to do that the marker does not.
                "data-picking": "{wheel_grab().is_some()}",
                // Pointer capture: the drag keeps tracking while the button is held,
                // even outside the wheel (picks past the rim slide along it).
                onpointerdown: move |e| {
                    capture_pointer(&e);
                    let g = grab(&e, wheel_xy(hue(), sat()));
                    wheel_grab.set(Some(g));
                    pick_wheel(onchange, hue, sat, l, g, &e);
                },
                onpointermove: move |e| {
                    if let Some(g) = wheel_grab() { pick_wheel(onchange, hue, sat, l, g, &e); }
                },
                onpointerup: move |_| end_pick(oncommit, wheel_grab, l, hue, sat),
                onpointercancel: move |_| end_pick(oncommit, wheel_grab, l, hue, sat),
                div { class: "wheel-marker", style: "left:{wx}%; top:{wy}%;" }
            }
            div {
                class: "l-slider",
                style: "background-image: {ramp()};",
                "data-picking": "{l_grab().is_some()}",
                onpointerdown: move |e| {
                    capture_pointer(&e);
                    let g = grab(&e, (l(), 0.5));
                    l_grab.set(Some(g));
                    pick_l(onchange, l, hue, sat, g, &e);
                },
                onpointermove: move |e| {
                    if let Some(g) = l_grab() { pick_l(onchange, l, hue, sat, g, &e); }
                },
                onpointerup: move |_| end_pick(oncommit, l_grab, l, hue, sat),
                onpointercancel: move |_| end_pick(oncommit, l_grab, l, hue, sat),
                div { class: "l-marker", style: "left:{lx}%;" }
            }
            // What was picked, said twice: as a patch big enough to judge, and as the
            // number to type when judging is not the point. The patch is the honest
            // answer to *what am I holding* — the marker sits on the color it names, and
            // a 12px ring around a pixel is not a sample of anything.
            input {
                class: "color-hex",
                style: "{well}",
                r#type: "text",
                spellcheck: false,
                autocomplete: "off",
                title: "Type a color: #rgb or #rrggbb",
                value: "{shown}",
                oninput: move |e| draft.set(Some(e.value())),
                // Blur commits (clicking away is an ordinary way to be finished);
                // Enter commits directly, and Escape abandons by dropping the draft
                // so the blur behind it has nothing left to send. Text that does not
                // parse is abandoned the same way — the field goes back to showing
                // the color the moment it stops being typed in, which says *no*
                // without a second control to say it with.
                onblur: move |_| commit_hex(onchange, oncommit, draft, l, hue, sat),
                onkeydown: move |e| match e.key() {
                    Key::Enter => commit_hex(onchange, oncommit, draft, l, hue, sat),
                    Key::Escape => draft.set(None),
                    _ => {}
                },
            }
        }
    }
}

/// Report the current wheel position through `handler` as straight sRGB.
fn apply_color(
    handler: EventHandler<[f32; 3]>,
    l: Signal<f32>,
    hue: Signal<f32>,
    sat: Signal<f32>,
) {
    handler.call(wheel_color(WHEEL_GAMUT, l(), hue(), sat()));
}

/// End a drag on `grab`, reporting the settled color through `oncommit` once —
/// the caller's cue to turn a run of `onchange` previews into one committed edit.
/// A no-op when no drag was in progress, so a stray release commits nothing.
///
/// Shared by `onpointerup` and `onpointercancel`: a cancelled color pick still
/// commits, unlike a cancelled geometry drag, because every instant of it is a
/// color the user chose and is already looking at — and because discarding it
/// would strand the caller's preview with no commit to supersede it.
fn end_pick(
    oncommit: Option<EventHandler<[f32; 3]>>,
    mut grab: Signal<Option<Grab>>,
    l: Signal<f32>,
    hue: Signal<f32>,
    sat: Signal<f32>,
) {
    if grab.write().take().is_none() {
        return;
    }
    if let Some(oncommit) = oncommit {
        apply_color(oncommit, l, hue, sat);
    }
}

/// Set hue and chroma from a pointer position over the wheel, then apply. Past the
/// rim the pick slides along it: there is no more chroma out there to mean.
fn pick_wheel(
    onchange: EventHandler<[f32; 3]>,
    mut hue: Signal<f32>,
    mut sat: Signal<f32>,
    l: Signal<f32>,
    grab: Grab,
    e: &Event<PointerData>,
) {
    let Some(p) = pointer_fraction(e) else {
        return;
    };
    let (fx, fy) = grab.place(p);
    let (dx, dy) = (fx * 2.0 - 1.0, 1.0 - fy * 2.0);
    let r = (dx * dx + dy * dy).sqrt();
    // Crossing the centre must not spin the hue. There is no direction to read
    // there, every direction is the same grey, and the one the artist came in on is
    // the one they keep — which is what makes a pass through the middle a way to
    // *desaturate* rather than a way to lose the hue.
    if r > 1e-4 {
        hue.set(dy.atan2(dx));
    }
    sat.set(r.min(1.0));
    apply_color(onchange, l, hue, sat);
}

/// Set `L` from a pointer position over the vertical slider (top = light), then apply.
fn pick_l(
    onchange: EventHandler<[f32; 3]>,
    mut l: Signal<f32>,
    hue: Signal<f32>,
    sat: Signal<f32>,
    grab: Grab,
    e: &Event<PointerData>,
) {
    let Some(p) = pointer_fraction(e) else {
        return;
    };
    l.set(grab.place(p).0.clamp(0.0, 1.0));
    apply_color(onchange, l, hue, sat);
}

/// Take what has been typed in the hex field as the whole color, previewing and
/// committing it in one go — a typed color is not a drag, so there is no run of
/// previews for a commit to close.
///
/// Clears the draft either way, so a field that cannot be parsed goes back to
/// showing the color rather than sitting there wrong.
fn commit_hex(
    onchange: EventHandler<[f32; 3]>,
    oncommit: Option<EventHandler<[f32; 3]>>,
    mut draft: Signal<Option<String>>,
    mut l: Signal<f32>,
    mut hue: Signal<f32>,
    mut sat: Signal<f32>,
) {
    // Taken out of the signal in its own statement: a `let … else` over a borrow of
    // the signal would keep that borrow alive across the `else`, and the writes
    // below would find it still held.
    let typed = draft.write().take();
    let Some(rgb) = typed.as_deref().and_then(parse_color) else {
        return;
    };
    let (nl, nh, ns) = on_wheel(WHEEL_GAMUT, rgb, hue());
    l.set(nl);
    hue.set(nh);
    sat.set(ns);
    apply_color(onchange, l, hue, sat);
    if let Some(oncommit) = oncommit {
        apply_color(oncommit, l, hue, sat);
    }
}

// --- the pictures ----------------------------------------------------------------
//
// The numbers are `stark_ui::color`'s — a wheel is a picture of which colors
// exist, and two answers to that would be two apps. What is left here is the
// carrier: a `data:` URL, because a `background-image` takes one.

/// One buffer of straight sRGB as a 24-bit BMP `data:` URL, `w`×`h`, rows from the
/// top (BMP rows run the other way, and that is this function's business rather than
/// its callers').
///
/// A BMP rather than a PNG because there is nothing to compress and no encoder to
/// carry: the header is fourteen fields and the body is the pixels. Rows are padded
/// to four bytes, which is what lets a caller choose any width — the ramp is one
/// pixel tall.
fn bmp_data_url(w: usize, h: usize, rgb: &[u8]) -> String {
    let stride = (w * 3 + 3) & !3;
    let bytes = stride * h;
    let mut bmp = Vec::with_capacity(54 + bytes);
    // BITMAPFILEHEADER (14) + BITMAPINFOHEADER (40).
    bmp.extend_from_slice(b"BM");
    bmp.extend_from_slice(&((54 + bytes) as u32).to_le_bytes());
    bmp.extend_from_slice(&0u32.to_le_bytes()); // reserved
    bmp.extend_from_slice(&54u32.to_le_bytes()); // pixel data offset
    bmp.extend_from_slice(&40u32.to_le_bytes()); // info header size
    bmp.extend_from_slice(&(w as i32).to_le_bytes()); // width
    bmp.extend_from_slice(&(h as i32).to_le_bytes()); // height (+ → bottom-up)
    bmp.extend_from_slice(&1u16.to_le_bytes()); // planes
    bmp.extend_from_slice(&24u16.to_le_bytes()); // bpp
    bmp.extend_from_slice(&0u32.to_le_bytes()); // BI_RGB
    bmp.extend_from_slice(&(bytes as u32).to_le_bytes());
    bmp.extend_from_slice(&2835i32.to_le_bytes()); // 72 dpi
    bmp.extend_from_slice(&2835i32.to_le_bytes());
    bmp.extend_from_slice(&0u32.to_le_bytes()); // colors used
    bmp.extend_from_slice(&0u32.to_le_bytes()); // important
    for row in 0..h {
        let y = h - 1 - row; // BMP rows are bottom-up; the buffer runs from the top
        for x in 0..w {
            let i = (y * w + x) * 3;
            bmp.push(rgb[i + 2]); // B
            bmp.push(rgb[i + 1]); // G
            bmp.push(rgb[i]); // R
        }
        bmp.resize(bmp.len() + stride - w * 3, 0);
    }
    format!(
        "url(data:image/bmp;base64,{})",
        crate::platform::base64_encode(&bmp)
    )
}

/// The picker's wheel at lightness `l`.
fn wheel_data_url(l: f32) -> String {
    bmp_data_url(FIELD_N, FIELD_N, &wheel_rgb(WHEEL_GAMUT, l))
}

/// The `L` slider's track at this hue and relative chroma.
fn l_ramp_data_url(hue: f32, sat: f32) -> String {
    bmp_data_url(RAMP_N, 1, &ramp_rgb(WHEEL_GAMUT, hue, sat))
}

/// The flat `a`/`b` plane a filter's chroma dial is drawn over (§21.5).
pub(super) fn ab_field_data_url(l: f32, ab: f32) -> String {
    bmp_data_url(FIELD_N, FIELD_N, &ab_field_rgb(l, ab))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A BMP is the size its own header says, whatever width it was asked for — row
    /// padding is the one way to get that wrong, and a one-pixel-wide ramp needs it.
    ///
    /// The only test left here: everything this panel used to check about the *wheel*
    /// went down with the wheel (`stark_ui::color`), and what is left is the
    /// carrier.
    #[test]
    fn a_bmp_is_the_size_its_header_says() {
        for (w, h) in [(1, 128), (2, 3), (96, 96)] {
            let url = bmp_data_url(w, h, &vec![128; w * h * 3]);
            let b64 = url
                .trim_start_matches("url(data:image/bmp;base64,")
                .trim_end_matches(')');
            let stride = (w * 3 + 3) & !3;
            // Base64 is four characters per three bytes, padded up.
            assert_eq!(b64.len(), (54 + stride * h).div_ceil(3) * 4, "{w}x{h}");
        }
    }
}
