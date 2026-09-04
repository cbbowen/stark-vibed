//! The Color panel: an Oklab wheel carrying every color the display can hold at a
//! chosen lightness, and the lightness beside it (§6.7, §11.2 N8).
//!
//! **Nothing about which colors exist is decided here.** The gamut's rim, the fit that
//! makes the wheel a wheel, the pictures of both and what a fine drag spends are
//! `stark_ui::color`, measured constants and all. What is here is a toolkit's
//! half: two textures, two hit regions, and a marker.
//!
//! The picker is **seeded, not driven**. It holds a hue that survives a trip to the
//! achromatic axis, and a color coming back through sRGB cannot say what hue a grey
//! was — so the wheel's `(l, hue, sat)` is the state, and the brush's color is what
//! that state *produces*. Reading the brush back every frame would spin the marker to
//! hue zero under the hand the moment a drag crossed the centre.

use std::sync::Arc;

use stark_ui::color::{self, FIELD_N, RAMP_N};
use wgpui::{
    Bounds, ImageSource, IntoElement, Pixels, Point, RenderImage, canvas, div, img, prelude::*, px,
    rgb,
};

/// The gamut the wheel is fitted to (§6.5) — **the picture carrier's, not the
/// window's**. The window's swapchain may be scRGB and the engine paints in any
/// gamut it has; these pictures are RGBA8 sprites wgpui's shaders read as sRGB, so a
/// wider rim would draw an outer ring of colors clamped on their way to the screen.
/// A wide color is reachable meanwhile by typing it
/// (`stark_ui::color::parse_color`); a wide carrier is what would move this
/// (§11.2, the wide-gamut wheel).
const WHEEL_GAMUT: stark_model::color::Gamut = stark_model::color::Gamut::Srgb;

/// The wheel's side, logical px — and the ramp's width, so the two controls line up
/// in a column narrower than the panel.
const WHEEL: f32 = 168.0;

/// The `L` track's height.
const TRACK: f32 = 16.0;

/// The marker's radius, logical px.
const MARK: f32 = 5.0;

/// Where the picker stands: a lightness, a hue and how much of the chroma available
/// at that lightness and hue it spends.
///
/// The three the crate's wheel is parametrized by, held rather than derived — see the
/// module note on why a grey has to keep its hue.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Wheel {
    pub l: f32,
    pub hue: f32,
    pub sat: f32,
}

impl Default for Wheel {
    /// The color a session starts on, put on the wheel.
    fn default() -> Self {
        Self::of(color::INITIAL_COLOR, 0.0)
    }
}

impl Wheel {
    /// Where `rgb` sits, keeping `hue` for a color that has none.
    pub fn of(rgb: [f32; 3], hue: f32) -> Self {
        let (l, hue, sat) = color::on_wheel(WHEEL_GAMUT, rgb, hue);
        Self { l, hue, sat }
    }

    /// The straight-sRGB color this position *is*.
    pub fn rgb(self) -> [f32; 3] {
        color::wheel_color(WHEEL_GAMUT, self.l, self.hue, self.sat)
    }
}

/// Which of the picker's two controls a press landed on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Region {
    /// The wheel: hue by direction, chroma by distance.
    Wheel,
    /// The lightness track under it.
    Track,
}

/// Where the two controls were laid out — `crate::panel`'s device, for its reason.
pub type Regions = std::rc::Rc<std::cell::RefCell<Vec<(Region, Bounds<Pixels>)>>>;

fn probe(regions: &Regions, region: Region) -> impl IntoElement {
    let regions = regions.clone();
    canvas(
        move |bounds, _, _| regions.borrow_mut().push((region, bounds)),
        |_, (), _, _| {},
    )
    .absolute()
    .top_0()
    .left_0()
    .right_0()
    .bottom_0()
}

/// Which control a press landed on.
pub fn hit(regions: &Regions, at: Point<Pixels>) -> Option<Region> {
    regions
        .borrow()
        .iter()
        .find(|(_, bounds)| bounds.contains(&at))
        .map(|(region, _)| *region)
}

/// Where in a control's own box a position sits, as fractions of it — clamped, so a
/// drag that has left the control keeps moving the value it took hold of.
pub fn fraction_at(regions: &Regions, region: Region, at: Point<Pixels>) -> Option<(f32, f32)> {
    let bounds = regions
        .borrow()
        .iter()
        .find(|(r, _)| *r == region)
        .map(|(_, b)| *b)?;
    let (w, h) = (f32::from(bounds.size.width), f32::from(bounds.size.height));
    (w > 0.0 && h > 0.0).then(|| {
        (
            ((f32::from(at.x) - f32::from(bounds.origin.x)) / w).clamp(0.0, 1.0),
            ((f32::from(at.y) - f32::from(bounds.origin.y)) / h).clamp(0.0, 1.0),
        )
    })
}

/// The wheel the fraction `(x, y)` names, at the lightness in hand.
///
/// Only hue and saturation: the wheel is one lightness, and moving `L` is the track's
/// job. A press outside the rim lands *on* it rather than off the control, which is
/// what makes the most saturated colors reachable at the edge of a fast drag.
pub fn wheel_at(held: Wheel, x: f32, y: f32) -> Wheel {
    let (hue, sat) = color::wheel_at(x, y);
    Wheel {
        // A press at the exact centre has no direction, so it keeps the one in hand
        // rather than snapping to zero — `on_wheel`'s rule, on the other side.
        hue: if sat > 1e-4 { hue } else { held.hue },
        sat,
        ..held
    }
}

/// The picker's own cached pictures.
///
/// One of each, keyed by what it is a picture *of*: the wheel changes with every step
/// of an `L` drag and the track with every step of a wheel drag, so a table keyed by
/// value would grow for the length of a gesture and never be asked twice. Holding the
/// last one is the whole of what a cache can do here — and it is worth doing, because
/// the alternative is `FIELD_N²` gamut lookups per frame.
#[derive(Default)]
pub struct Pictures {
    wheel: Option<(u32, Arc<RenderImage>)>,
    track: Option<((u32, u32), Arc<RenderImage>)>,
}

impl Pictures {
    /// The wheel at this lightness, built if the last one was of another.
    ///
    /// Keyed on the *quantized* lightness: the picture is 96 texels of a low-frequency
    /// plane, so two lightnesses a 255th apart draw the same thing, and a drag that
    /// crosses one step is what should cost a rebuild.
    fn wheel(&mut self, l: f32) -> Option<Arc<RenderImage>> {
        let key = (l.clamp(0.0, 1.0) * 255.0).round() as u32;
        if self.wheel.as_ref().is_none_or(|(k, _)| *k != key) {
            let picture = texture(
                FIELD_N,
                FIELD_N,
                &color::wheel_rgb(WHEEL_GAMUT, key as f32 / 255.0),
                inside_the_rim,
            )?;
            self.wheel = Some((key, picture));
        }
        self.wheel.as_ref().map(|(_, p)| p.clone())
    }

    /// The lightness track at this hue and relative chroma.
    fn track(&mut self, hue: f32, sat: f32) -> Option<Arc<RenderImage>> {
        let key = (
            (hue.rem_euclid(std::f32::consts::TAU) * 40.0).round() as u32,
            (sat.clamp(0.0, 1.0) * 255.0).round() as u32,
        );
        if self.track.as_ref().is_none_or(|(k, _)| *k != key) {
            let picture = texture(
                RAMP_N,
                1,
                &color::ramp_rgb(WHEEL_GAMUT, hue, sat),
                |_, _| 255,
            )?;
            self.track = Some((key, picture));
        }
        self.track.as_ref().map(|(_, p)| p.clone())
    }
}

/// A buffer of straight sRGB as a texture this toolkit can draw, with `alpha`
/// deciding each texel's coverage from its place in the picture.
///
/// **RGBA**, which is what the polychrome atlas has always been — though wgpui said
/// otherwise until this wheel proved it did not: the picture went up with red and
/// blue exchanged, and the marker sat on a blue the readout called `#9c0a05`. The
/// vendored fix is patch 4 (`vendor/wgpui/VENDORING.md`), and this is what agrees
/// with it.
///
/// The alpha is how the wheel becomes a circle. Clipping the element would be the
/// obvious way and is the web frontend's, but it leans on the toolkit rounding a
/// *child image*; cutting the picture is the same result decided by the thing that
/// knows the geometry, and it gets an antialiased rim for free where a clip gets the
/// element's own.
fn texture(
    w: usize,
    h: usize,
    rgb: &[u8],
    alpha: impl Fn(usize, usize) -> u8,
) -> Option<Arc<RenderImage>> {
    let rgba: Vec<u8> = rgb
        .as_chunks::<3>()
        .0
        .iter()
        .enumerate()
        .flat_map(|(i, p)| [p[0], p[1], p[2], alpha(i % w, i / w)])
        .collect();
    let buffer = image::RgbaImage::from_raw(w as u32, h as u32, rgba)?;
    Some(Arc::new(RenderImage::new(vec![image::Frame::new(buffer)])))
}

/// Opaque inside the unit circle, clear outside it, with one texel of ramp between —
/// which is what turns the square picture of the wheel into a wheel.
fn inside_the_rim(x: usize, y: usize) -> u8 {
    let last = (FIELD_N - 1) as f32;
    let dx = 2.0 * x as f32 / last - 1.0;
    let dy = 2.0 * y as f32 / last - 1.0;
    let r = (dx * dx + dy * dy).sqrt();
    // The ramp is a texel of the *picture*, which is scaled up — so the edge it draws
    // is soft at the size the panel shows, which is what an unclipped circle needs.
    let t = ((1.0 - r) * last * 0.5 + 0.5).clamp(0.0, 1.0);
    (t * 255.0) as u8
}

/// Build the panel.
pub fn color_panel(
    wheel: Wheel,
    pictures: &mut Pictures,
    regions: &Regions,
) -> impl IntoElement + use<> {
    regions.borrow_mut().clear();
    let (mx, my) = color::wheel_xy(wheel.hue, wheel.sat);
    let rgb_now = wheel.rgb();
    let swatch = ((rgb_now[0] * 255.0) as u32) << 16
        | ((rgb_now[1] * 255.0) as u32) << 8
        | (rgb_now[2] * 255.0) as u32;
    div()
        .flex()
        .flex_col()
        .gap_2()
        .items_center()
        .child(
            div()
                .relative()
                .size(px(WHEEL))
                // No clip: the picture cuts itself (`inside_the_rim`), so what is
                // round is the thing that knows where the rim is.
                .bg(rgb(0x1e2124))
                .child(probe(regions, Region::Wheel))
                .children(
                    pictures
                        .wheel(wheel.l)
                        .map(|picture| img(ImageSource::Render(picture)).size(px(WHEEL))),
                )
                .child(marker(mx * WHEEL, my * WHEEL)),
        )
        .child(
            div()
                .relative()
                .w(px(WHEEL))
                .h(px(TRACK))
                .rounded_sm()
                .overflow_hidden()
                .bg(rgb(0x14161a))
                .child(probe(regions, Region::Track))
                .children(
                    pictures
                        .track(wheel.hue, wheel.sat)
                        .map(|picture| img(ImageSource::Render(picture)).w(px(WHEEL)).h(px(TRACK))),
                )
                .child(marker(wheel.l * WHEEL, TRACK / 2.0)),
        )
        .child(
            div()
                .flex()
                .w(px(WHEEL))
                .items_center()
                .gap_2()
                .child(
                    div()
                        .size(px(TRACK))
                        .rounded_sm()
                        .border_1()
                        .border_color(rgb(0x35393d))
                        .bg(rgb(swatch)),
                )
                .child(
                    div()
                        .text_xs()
                        .text_color(rgb(0x9aa0a6))
                        .child(color::notation_of(rgb_now)),
                ),
        )
}

/// The ring that says where the picker stands.
///
/// A ring rather than a dot, so the color *under* it is what is judged — the whole
/// point of a picker being a picture. White with a dark outline, which reads on both
/// ends of the lightness axis without knowing which end it is on.
fn marker(x: f32, y: f32) -> impl IntoElement {
    div()
        .absolute()
        .left(px(x - MARK))
        .top(px(y - MARK))
        .size(px(MARK * 2.0))
        .rounded(px(MARK))
        .border_2()
        .border_color(rgb(0xffffff))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A press at the exact centre keeps the hue in hand rather than snapping to
    /// zero — the same rule `on_wheel` follows for a grey, on the other side of the
    /// same problem.
    #[test]
    fn the_centre_keeps_the_hue_it_had() {
        let held = Wheel {
            l: 0.5,
            hue: 1.25,
            sat: 0.8,
        };
        let at_centre = wheel_at(held, 0.5, 0.5);
        assert_eq!(at_centre.hue, 1.25);
        assert!(at_centre.sat < 1e-4);
    }

    /// A press outside the rim lands on it, so the most saturated colors are reachable
    /// at the edge of a fast drag rather than only by stopping exactly on the line.
    #[test]
    fn a_press_outside_the_rim_lands_on_it() {
        let held = Wheel::default();
        assert_eq!(wheel_at(held, 1.0, 0.0).sat, 1.0);
    }

    /// The wheel a color puts the picker on produces that color back.
    #[test]
    fn a_color_and_its_wheel_agree() {
        let w = Wheel::of([0.2, 0.45, 0.7], 0.0);
        let back = w.rgb();
        for i in 0..3 {
            assert!((back[i] - [0.2, 0.45, 0.7][i]).abs() < 1.0 / 255.0);
        }
    }

    /// The pictures are rebuilt when what they are a picture of moves, and not
    /// otherwise — a wheel is `FIELD_N²` gamut lookups, spent per frame if this is
    /// wrong.
    #[test]
    fn a_picture_is_kept_until_its_subject_moves() {
        let mut pictures = Pictures::default();
        let first = pictures.wheel(0.5).expect("a wheel builds");
        // The same lightness, and one under the quantization step: the same picture.
        assert!(Arc::ptr_eq(&first, &pictures.wheel(0.5).unwrap()));
        assert!(Arc::ptr_eq(
            &first,
            &pictures.wheel(0.5 + 1.0 / 1000.0).unwrap()
        ));
        assert!(!Arc::ptr_eq(&first, &pictures.wheel(0.9).unwrap()));
    }
}
