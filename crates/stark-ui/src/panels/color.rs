//! The floating Color panel: an Oklab picker — a wheel carrying every color the
//! display can hold at a chosen lightness, and the lightness beside it.
//!
//! The eyedropper writes the same brush color this picker does, but its *options*
//! are not here — they live in a bar that comes up on Alt
//! ([`crate::panels::pick`]), since it is a modifier rather than a tool. What the
//! panel owes it is `seed`: a pick sets the color from outside the picker, and the
//! markers have to follow.

use std::f32::consts::TAU;

use dioxus::prelude::*;

use crate::platform::{capture_pointer, pointer_fraction};
use crate::state::{AppState, update_brush};
use stark_model::color::{oklab_to_linear_srgb, oklab_to_srgb, srgb_to_oklab};

/// The color a session starts on. The panel mounts before the engine exists, so
/// this is both the picker's fallback seed *and* what `main`'s startup seeding
/// pushes into the engine — the two have to agree, or the picker would show a
/// color the brush does not have and the first stroke would come out black.
pub const INITIAL_COLOR: [f32; 3] = [0.61, 0.04, 0.02];

#[component]
pub fn ColorPanel() -> Element {
    let state = use_context::<AppState>();
    // Seed from the brush's current color (peek → no re-render on every paint).
    let init = state
        .obs
        .peek()
        .as_ref()
        .map(|o| [o.brush.color[0], o.brush.color[1], o.brush.color[2]])
        .unwrap_or(INITIAL_COLOR);
    // Read reactively, unlike the color: this is how a pick — which sets the color
    // from outside the picker — gets the markers to move (see `AppState::color_epoch`).
    let seed = (state.color_epoch)();

    rsx! {
        OklabPicker {
            init,
            seed,
            onchange: move |rgb: [f32; 3]| {
                update_brush(state, move |br| {
                    br.color = [rgb[0], rgb[1], rgb[2], br.color[3]];
                });
            },
        }
    }
}

// The picker's on-screen sizes are the stylesheet's alone (`.color-wheel`,
// `.l-slider`): the markers are placed in percentages and a pick reads the
// pointer as a fraction of the element it landed in (`platform::pointer_fraction`),
// so nothing here has to mirror a px value — which is what lets each of the
// picker's homes size the wheel with one CSS rule and no second copy to drift.

/// Rendered resolution of the wheel BMP (CSS scales it to the wheel's size,
/// smoothly — the plane is low-frequency, and a small BMP keeps the data URL cheap
/// to regenerate while dragging `L`).
const FIELD_N: usize = 96;

/// Rendered height of the `L` ramp BMP — one pixel wide, since the track carries no
/// horizontal variation and CSS stretches it across.
const RAMP_N: usize = 128;

// --- the sRGB gamut, which is what the wheel is a picture of ---------------------

/// How far past the achromatic axis sRGB could possibly reach in Oklab chroma — the
/// outer bracket [`max_chroma`] searches in. The real maximum over the whole gamut
/// is ≈0.323 (a saturated blue), so this is outside it at *every* lightness, which
/// is the only thing the search asks of it.
const CHROMA_CEILING: f32 = 0.45;

/// Halvings behind [`max_chroma`]. Twenty of them land within 5e-7 of the boundary,
/// three orders under the 1/255 the display quantizes to, and a fixed count keeps a
/// wheel a pure function of its lightness.
const CHROMA_STEPS: u32 = 20;

/// How many hues [`rim_table`] samples the boundary at. 512 is a step of 0.7°.
///
/// The rim is smooth in hue nearly everywhere, and one place it is not: **a slice at
/// constant lightness has corners, because the sRGB cube has them**. At `L = 0.45`
/// the rim runs 0.154 at −104° up to 0.240 at −96.5° and then turns hard, reaching
/// 0.313 — the blue primary itself — half a degree later, before easing back down.
/// The wheel wears that as a crease: a narrow wedge of deep blue with a hard edge,
/// visible in the picture and steep to drag across.
///
/// It is kept rather than smoothed away, because the two ways out are worse. Shaving
/// the corner to make the rim gentle puts `#0000ff` outside the wheel — a primary no
/// drag can reach. Rounding *outward* instead puts a band of clamped color back
/// inside the rim, which is the thing this picker exists to get rid of. And the
/// crease costs less than it looks: chroma per pixel is bounded by `rim / 124px`
/// everywhere except across that one ray, which is under the flat `0.64 / 248px` the
/// square field this replaced had *everywhere*. The known third answer is Ottosson's
/// okhsl, which normalizes against a hue-smooth cusp instead of the slice and
/// accepts a little clamping in exchange — a hundred and fifty lines of fitted
/// polynomial, and a lightness axis that is no longer Oklab's `L`.
const RIM_N: usize = 512;

/// How far outside a linear channel [`in_srgb`] still calls a color showable.
///
/// **A gamut slice at constant lightness is not star-shaped about the achromatic
/// axis**, which a search out from the centre has to be told. Oklab is a cube root
/// away from linear sRGB, so a straight line at constant `L` is a curve through the
/// sRGB cube, and along the hue of the blue primary that curve dips out of the cube
/// and back in: at `L = 0.452` the red channel falls to −0.0007 between chroma 0.266
/// and 0.313, then returns for the corner that *is* `#0000ff`. Bisection from the
/// centre stops at the first crossing, and the blue primary is stranded on the far
/// side of a gap 0.0006 wide — 11.6% short on chroma, with no drag able to reach it.
///
/// A thousandth of a channel bridges the gap: over a 26³ sample of sRGB, nothing is
/// unreachable at `0.001` and 44 colors are at `0` (two at `0.0005`). What it costs
/// is that a color on the rim may be clamped, by at most 4/255 in one channel and
/// only where that channel is already near zero — the deep blues. That is the whole
/// price, it is measured rather than hoped for, and `wheel_color`'s clamp is what
/// collects it.
const GAMUT_BRIDGE: f32 = 0.001;

/// Whether Oklab `(l, a, b)` is a color sRGB can show, give or take
/// [`GAMUT_BRIDGE`].
fn in_srgb(l: f32, a: f32, b: f32) -> bool {
    oklab_to_linear_srgb([l, a, b])
        .iter()
        .all(|c| (-GAMUT_BRIDGE..=1.0 + GAMUT_BRIDGE).contains(c))
}

/// The most chroma sRGB holds at lightness `l` in the direction `hue` — the gamut's
/// rim, which is what the wheel's edge *is*.
///
/// By bisection rather than by Ottosson's analytic approximation of the same
/// boundary, for one reason: the search only ever moves its lower bracket to a point
/// it has *tested*, so it answers with the gamut this build's own conversion has
/// rather than with a curve fitted to somebody else's. The cost is nothing the
/// picker can feel — a wheel asks this once per hue step ([`rim_table`]), not once
/// per pixel, and a pick asks it once. What it is told about the one place the
/// bisection's assumption fails is [`GAMUT_BRIDGE`].
fn max_chroma(l: f32, hue: f32) -> f32 {
    let (sin, cos) = hue.sin_cos();
    let (mut lo, mut hi) = (0.0f32, CHROMA_CEILING);
    for _ in 0..CHROMA_STEPS {
        let mid = 0.5 * (lo + hi);
        if in_srgb(l, cos * mid, sin * mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// The whole rim at one lightness: [`RIM_N`] chromas around the circle.
///
/// Only the *drawing* goes through the table; every color the picker reports asks
/// [`max_chroma`] directly. That is the cheap half of a measured trade — a wheel
/// costs 0.8ms through the table and 2.9ms with a bisection per pixel, and the
/// second one is most of a frame on the machine this ships to, spent while the `L`
/// slider is being dragged. What the table costs back is that a drawn pixel and the
/// color reported for it can disagree by the interpolation error over one 0.7° step:
/// nothing anywhere the rim is smooth, and under a pixel's worth at the one place it
/// is not (see [`RIM_N`]).
fn rim_table(l: f32) -> Vec<f32> {
    (0..RIM_N)
        .map(|i| max_chroma(l, TAU * i as f32 / RIM_N as f32))
        .collect()
}

/// The rim between two of [`rim_table`]'s samples, linearly and wrapping.
fn rim_at(rim: &[f32], hue: f32) -> f32 {
    let t = hue.rem_euclid(TAU) / TAU * RIM_N as f32;
    let i = t.floor();
    let f = t - i;
    let i = (i as usize) % RIM_N;
    rim[i] * (1.0 - f) + rim[(i + 1) % RIM_N] * f
}

// --- the wheel's coordinates -----------------------------------------------------

/// Where a straight-sRGB color sits on the wheel: its lightness, its hue, and how
/// much of the chroma available *at that lightness and that hue* it spends — `0` at
/// the centre, `1` on the rim.
///
/// `keep` is the hue to hold on to for a color that has none. A grey is every hue at
/// once, so sRGB cannot say which one the artist was on, and the direction they last
/// chose is the only answer that does not spin the marker for them.
fn on_wheel(rgb: [f32; 3], keep: f32) -> (f32, f32, f32) {
    let [l, a, b, _] = srgb_to_oklab([rgb[0], rgb[1], rgb[2], 1.0]);
    let c = (a * a + b * b).sqrt();
    if c <= 1e-6 {
        return (l, keep, 0.0);
    }
    let hue = b.atan2(a);
    let rim = max_chroma(l, hue);
    (l, hue, if rim > 1e-6 { (c / rim).min(1.0) } else { 0.0 })
}

/// The color a wheel position *is*, as straight sRGB.
///
/// The clamp collects the two hairs the fit leaves: [`GAMUT_BRIDGE`], and the ULP at
/// the ends of the lightness axis, where the conversion matrices' rows sum to
/// 1.00000004 and white comes back a shade over. Neither is the fit failing — a
/// position with `sat ≤ 1` is a color the display has.
fn wheel_color(l: f32, hue: f32, sat: f32) -> [f32; 3] {
    let (sin, cos) = hue.sin_cos();
    let c = sat * max_chroma(l, hue);
    let rgba = oklab_to_srgb([l, c * cos, c * sin, 1.0]);
    [
        rgba[0].clamp(0.0, 1.0),
        rgba[1].clamp(0.0, 1.0),
        rgba[2].clamp(0.0, 1.0),
    ]
}

/// Where `(hue, sat)` sits in the wheel's box, as fractions of it — the marker's
/// place, and the point a fine drag moves *from*. `+a` runs right and `+b` up, the
/// orientation the flat picture of the same plane keeps ([`ab_field_data_url`]), so
/// warm sits at the top of both.
fn wheel_xy(hue: f32, sat: f32) -> (f32, f32) {
    let (sin, cos) = hue.sin_cos();
    (0.5 + 0.5 * sat * cos, 0.5 - 0.5 * sat * sin)
}

/// How much of the pointer's travel a fine drag spends. A fifth: the whole width of
/// the wheel then covers a fifth of it, which is the difference between landing in
/// *that red* and landing somewhere in the reds.
const FINE_GAIN: f32 = 0.2;

/// What a pointer sample on one of the picker's two controls means.
///
/// Decided once, on pointer-down, and held for the whole gesture — the reason the
/// filter dial's `Grab` is decided once too: a drag that changed its mind about what
/// the pointer meant halfway through would rewrite a value the hand was not on.
#[derive(Copy, Clone)]
enum Grab {
    /// The pointer *is* the value: where it lands is what is picked, so a press with
    /// no travel is already a complete pick.
    At,
    /// Shift: the value moves *with* the pointer at [`FINE_GAIN`], from where it
    /// already stood. The press picks nothing at all, which is the point — the hand
    /// gets the control's whole width to spend on a fraction of its range, and the
    /// color under the marker does not jump away before the adjustment starts.
    ///
    /// `from` is the pointer fraction the press landed at, `held` the marker's place
    /// at that moment; every later sample is `held` plus the geared-down travel.
    Fine { from: (f32, f32), held: (f32, f32) },
}

impl Grab {
    /// What this press means, given where the marker stood when it landed.
    fn press(e: &Event<PointerData>, held: (f32, f32)) -> Grab {
        match pointer_fraction(e) {
            Some(from) if e.modifiers().contains(Modifiers::SHIFT) => Grab::Fine { from, held },
            _ => Grab::At,
        }
    }

    /// Where in the control's box this sample points, as a fraction of it.
    fn place(self, p: (f32, f32)) -> (f32, f32) {
        match self {
            Grab::At => p,
            Grab::Fine { from, held } => (
                held.0 + (p.0 - from.0) * FINE_GAIN,
                held.1 + (p.1 - from.1) * FINE_GAIN,
            ),
        }
    }
}

/// Reusable Oklab color picker: a wheel of hue and chroma at one lightness, a
/// vertical `L` slider beside it, and a readout of what has been chosen. Seeds its
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
    let (il, ih, is) = on_wheel(init, 0.0);
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
        let (nl, nh, ns) = on_wheel(init, *hue.peek());
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
    let ly = (1.0 - l()) * 100.0; // L: 1→top, 0→bottom
    let rgb = wheel_color(l(), hue(), sat());
    let well = format!(
        "background: rgb({:.2}% {:.2}% {:.2}%);",
        rgb[0] * 100.0,
        rgb[1] * 100.0,
        rgb[2] * 100.0
    );
    let shown = draft().unwrap_or_else(|| hex_of(rgb));

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
                    let g = Grab::press(&e, wheel_xy(hue(), sat()));
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
                    let g = Grab::press(&e, (0.5, 1.0 - l()));
                    l_grab.set(Some(g));
                    pick_l(onchange, l, hue, sat, g, &e);
                },
                onpointermove: move |e| {
                    if let Some(g) = l_grab() { pick_l(onchange, l, hue, sat, g, &e); }
                },
                onpointerup: move |_| end_pick(oncommit, l_grab, l, hue, sat),
                onpointercancel: move |_| end_pick(oncommit, l_grab, l, hue, sat),
                div { class: "l-marker", style: "top:{ly}%;" }
            }
        }
        // What was picked, said twice: as a patch big enough to judge, and as the
        // number to type when judging is not the point. The patch is the honest
        // answer to *what am I holding* — the marker sits on the color it names, and
        // a 12px ring around a pixel is not a sample of anything.
        div { class: "color-readout",
            div { class: "color-well", style: "{well}" }
            input {
                class: "color-hex",
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
    handler.call(wheel_color(l(), hue(), sat()));
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
    l.set((1.0 - grab.place(p).1).clamp(0.0, 1.0));
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
    let Some(rgb) = typed.as_deref().and_then(parse_hex) else {
        return;
    };
    let (nl, nh, ns) = on_wheel(rgb, hue());
    l.set(nl);
    hue.set(nh);
    sat.set(ns);
    apply_color(onchange, l, hue, sat);
    if let Some(oncommit) = oncommit {
        apply_color(oncommit, l, hue, sat);
    }
}

/// A straight-sRGB color as `#rrggbb`, at the display's own precision.
fn hex_of(rgb: [f32; 3]) -> String {
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("#{:02x}{:02x}{:02x}", q(rgb[0]), q(rgb[1]), q(rgb[2]))
}

/// `#rgb`, `#rrggbb` or either without the hash, as straight sRGB. `None` for
/// anything else — including a half-typed one, which is what the field holds most of
/// the time it is being used.
fn parse_hex(s: &str) -> Option<[f32; 3]> {
    let s = s.trim();
    let s = s.strip_prefix('#').unwrap_or(s);
    if !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let byte = |i: usize| u8::from_str_radix(&s[i..i + 2], 16).ok();
    // `0x7` is `0x77`: a short code names the byte whose halves match, so the two
    // spellings of the same color agree rather than differing by a 17th.
    let nibble = |i: usize| u8::from_str_radix(&s[i..i + 1], 16).ok().map(|v| v * 17);
    let rgb = match s.len() {
        3 => [nibble(0)?, nibble(1)?, nibble(2)?],
        6 => [byte(0)?, byte(2)?, byte(4)?],
        _ => return None,
    };
    Some(rgb.map(|v| v as f32 / 255.0))
}

// --- the pictures ----------------------------------------------------------------

/// One small 24-bit BMP as a `data:` URL: `w`×`h`, each pixel the straight sRGB
/// `px(x, y)` gives for it, with `y` counted from the *top* (BMP rows run the other
/// way, and that is this function's business rather than its callers').
///
/// A BMP rather than a PNG because there is nothing to compress and no encoder to
/// carry: the header is fourteen fields and the body is the pixels. Rows are padded
/// to four bytes, which is what lets a caller choose any width — the ramp is one
/// pixel wide.
fn bmp_data_url(w: usize, h: usize, px: impl Fn(usize, usize) -> [f32; 3]) -> String {
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
        let y = h - 1 - row; // BMP rows are bottom-up; `y` is from the top
        for x in 0..w {
            let rgb = px(x, y);
            let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            bmp.push(q(rgb[2])); // B
            bmp.push(q(rgb[1])); // G
            bmp.push(q(rgb[0])); // R
        }
        bmp.resize(bmp.len() + stride - w * 3, 0);
    }
    format!(
        "url(data:image/bmp;base64,{})",
        crate::platform::base64_encode(&bmp)
    )
}

/// The picker's wheel at lightness `l`: hue by direction, chroma by distance as a
/// fraction of [`max_chroma`] in that direction, so the unit circle is the sRGB
/// boundary and every pixel inside it is a color the display can show.
///
/// Square, and the corners past the rim carry the rim's own color: the element is
/// clipped to a circle by the stylesheet, so those pixels are never seen — but they
/// are what the scaler mixes into the edge ones, and a corner left black would draw
/// a dark fringe all the way round.
fn wheel_data_url(l: f32) -> String {
    let rim = rim_table(l);
    let last = (FIELD_N - 1) as f32;
    bmp_data_url(FIELD_N, FIELD_N, |x, y| {
        let dx = 2.0 * x as f32 / last - 1.0;
        let dy = 1.0 - 2.0 * y as f32 / last;
        let r = (dx * dx + dy * dy).sqrt();
        let (ux, uy) = if r > 1e-6 {
            (dx / r, dy / r)
        } else {
            (1.0, 0.0)
        };
        let c = rim_at(&rim, dy.atan2(dx)) * r.min(1.0);
        let rgba = oklab_to_srgb([l, c * ux, c * uy, 1.0]);
        [rgba[0], rgba[1], rgba[2]]
    })
}

/// The `L` slider's track: this hue at this fraction of the chroma each lightness
/// can hold, black at the bottom to white at the top.
///
/// Drawn rather than handed to CSS as a `linear-gradient(in oklab …)`, which is what
/// it used to be. A CSS ramp interpolates `(a, b)` linearly, so with a saturated
/// color it leaves the gamut immediately and both ends of the track go flat under
/// the clamp — the slider stops answering exactly where the artist is looking for a
/// highlight or a shadow. Fitting each row to its own lightness is the same fix the
/// wheel makes, on the other axis, and what it draws is not a gradient of anything.
fn l_ramp_data_url(hue: f32, sat: f32) -> String {
    let last = (RAMP_N - 1) as f32;
    bmp_data_url(1, RAMP_N, |_, y| {
        wheel_color(1.0 - y as f32 / last, hue, sat)
    })
}

/// Render the Oklab `a`/`b` plane at lightness `l` as a small BMP `data:` URL, flat:
/// `a` runs left→right (−`ab`→+`ab`), `b` runs bottom→top, so warm colors sit at the
/// top. Out-of-gamut colors clamp to sRGB.
///
/// This is the *other* picture of the plane, and it is deliberately not the picker's.
/// The filter bar's chroma dial draws an affine map of the `(a, b)` plane — a circle
/// of reference hues and where the filter sends them (§21.5) — so its background has
/// to be that plane, flat and undistorted, or the handle would sit over a picture it
/// disagrees with. The picker's wheel is fitted to the gamut and so is not flat; the
/// two still agree about orientation, which is the part a reader carries between
/// them.
pub(super) fn ab_field_data_url(l: f32, ab: f32) -> String {
    let last = (FIELD_N - 1) as f32;
    bmp_data_url(FIELD_N, FIELD_N, |x, y| {
        let aa = ab * (2.0 * x as f32 / last - 1.0);
        let bb = ab * (1.0 - 2.0 * y as f32 / last);
        let rgba = oklab_to_srgb([l, aa, bb, 1.0]);
        [rgba[0], rgba[1], rgba[2]]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The point of the fit: every position the wheel can hold is a color the
    /// display shows as asked. Stated as what the artist would see — how far the
    /// clamp in [`wheel_color`] has to move the color — rather than as a bound on
    /// the linear channels, because that is the number [`GAMUT_BRIDGE`] is spending.
    #[test]
    fn no_wheel_position_needs_more_than_a_hair_of_clamping() {
        for li in 0..=20 {
            let l = li as f32 / 20.0;
            for hi in 0..72 {
                let hue = TAU * hi as f32 / 72.0;
                for si in 0..=10 {
                    let sat = si as f32 / 10.0;
                    let (sin, cos) = hue.sin_cos();
                    let c = sat * max_chroma(l, hue);
                    let raw = oklab_to_srgb([l, c * cos, c * sin, 1.0]);
                    let got = wheel_color(l, hue, sat);
                    let moved = (0..3).map(|i| (raw[i] - got[i]).abs()).fold(0.0, f32::max);
                    assert!(
                        moved <= 4.0 / 255.0,
                        "l {l} hue {hue} sat {sat}: clamp moved {} levels",
                        moved * 255.0
                    );
                }
            }
        }
    }

    /// A color goes onto the wheel and comes back the same color — within a
    /// quantization step, which is all the readout that reports it can carry.
    #[test]
    fn a_color_survives_the_wheel() {
        for rgb in [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.61, 0.04, 0.02],
            [0.2, 0.45, 0.7],
        ] {
            let (l, hue, sat) = on_wheel(rgb, 0.0);
            let back = wheel_color(l, hue, sat);
            assert!(
                rgb.iter()
                    .zip(back)
                    .all(|(a, b)| (a - b).abs() < 1.0 / 255.0),
                "{rgb:?} → (l {l}, hue {hue}, sat {sat}) → {back:?}"
            );
        }
    }

    /// The most saturated colors sRGB has are *on* the rim, rather than stranded
    /// outside it where no drag reaches them. Within a percent, which is the width
    /// of the gap [`GAMUT_BRIDGE`] crosses — and a rim position renders as the
    /// primary either way, which the round trip above is what actually checks.
    #[test]
    fn the_primaries_reach_the_rim() {
        for rgb in [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ] {
            let (_, _, sat) = on_wheel(rgb, 0.0);
            assert!(sat > 0.99, "{rgb:?} sat {sat}");
        }
    }

    /// A grey keeps the hue it is given: sRGB cannot say which one it was, and
    /// inventing zero would spin the marker every time a drag crossed the centre.
    #[test]
    fn a_grey_keeps_the_hue_it_is_given() {
        let (_, hue, sat) = on_wheel([0.5, 0.5, 0.5], 1.25);
        assert_eq!(hue, 1.25);
        assert_eq!(sat, 0.0);
    }

    /// The rim table wraps rather than falling off its end.
    #[test]
    fn the_rim_wraps() {
        let rim = rim_table(0.6);
        for hue in [-TAU, -0.3, 0.0, 0.3, TAU, TAU + 0.3] {
            let exact = max_chroma(0.6, hue);
            let table = rim_at(&rim, hue);
            assert!(
                (table - exact).abs() < 1e-3,
                "hue {hue}: table {table} vs exact {exact}"
            );
        }
    }

    #[test]
    fn hex_round_trips() {
        for (text, rgb) in [
            ("#000000", [0.0, 0.0, 0.0]),
            ("#ffffff", [1.0, 1.0, 1.0]),
            ("#ff0000", [1.0, 0.0, 0.0]),
        ] {
            assert_eq!(parse_hex(text), Some(rgb));
            assert_eq!(hex_of(rgb), text);
        }
        // A short code is the byte whose halves match, and the hash is optional.
        assert_eq!(parse_hex("#abc"), parse_hex("aabbcc"));
        assert_eq!(parse_hex(" #ABC "), parse_hex("#abc"));
        for bad in ["", "#", "#12", "#12345", "#gggggg", "12345678"] {
            assert_eq!(parse_hex(bad), None, "{bad:?}");
        }
    }

    /// A BMP is the size its own header says, whatever width it was asked for — row
    /// padding is the one way to get that wrong, and a one-pixel-wide ramp needs it.
    #[test]
    fn a_bmp_is_the_size_its_header_says() {
        for (w, h) in [(1, 128), (2, 3), (96, 96)] {
            let url = bmp_data_url(w, h, |_, _| [0.5, 0.25, 0.125]);
            let b64 = url
                .trim_start_matches("url(data:image/bmp;base64,")
                .trim_end_matches(')');
            let stride = (w * 3 + 3) & !3;
            // Base64 is four characters per three bytes, padded up.
            assert_eq!(b64.len(), (54 + stride * h).div_ceil(3) * 4, "{w}x{h}");
        }
    }
}
