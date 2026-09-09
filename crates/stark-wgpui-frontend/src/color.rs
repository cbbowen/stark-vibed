//! The Color panel: an Oklab wheel carrying every color the display can hold at a
//! chosen lightness, and the lightness beside it (§6.7, §11.2 N8).
//!
//! **Nothing about which colors exist is decided here.** The gamut's rim, the fit that
//! makes the wheel a wheel, the pictures of both, what a fine drag spends and the
//! picker's own state ([`Wheel`]) are `stark_ui::color`, measured constants and all.
//! What is here is a toolkit's half: two textures, two hit regions, and a marker.

use std::sync::Arc;

use stark_ui::color::{self, FIELD_N, RAMP_N};
use wgpui::{
    Bounds, ImageSource, IntoElement, Pixels, Point, RenderImage, canvas, div, img, prelude::*, px,
    rgb,
};

use wgpui::Entity;
use wgpui_component::Sizable;
use wgpui_component::input::{Input, InputState};

use crate::style;

/// The gamut the wheel is fitted to (§6.5) — **the picture carrier's, not the
/// window's**. The window's swapchain may be scRGB and the engine paints in any
/// gamut it has; these pictures are RGBA8 sprites wgpui's shaders read as sRGB, so a
/// wider rim would draw an outer ring of colors clamped on their way to the screen.
/// A wide color is reachable meanwhile by typing it
/// (`stark_ui::color::parse_color`); a wide carrier is what would move this
/// (§11.2, the wide-gamut wheel).
pub const WHEEL_GAMUT: stark_model::color::Gamut = stark_model::color::Gamut::Srgb;

/// The wheel's side, logical px — and the ramp's width, and the hex row's, so the
/// three controls are one edge-to-edge column.
///
/// The shelf's whole width rather than a figure of its own (`crate::panel`): a picker
/// is a picture read by eye, so every px of the column is resolution it can spend, and
/// one narrower than the layer rows under it reads as a control that failed to lay
/// out. The rim's antialiasing is a texel of [`FIELD_N`] scaled to this, so it softens
/// as this grows — which is what `inside_the_rim` is trading for an unclipped circle.
const WHEEL: f32 = crate::panel::RIGHT_CONTENT;

/// The `L` track's height.
const TRACK: f32 = 16.0;

/// The marker's radius, logical px.
const MARK: f32 = 5.0;

/// The picker's state, which is the crate's (`stark_ui::color::Wheel`): a lightness,
/// a hue, and how much of the chroma available at that lightness and hue it spends.
///
/// Re-exported rather than wrapped, so the one thing this frontend adds — which gamut
/// its picture carrier has — stays a constant the call sites pass rather than a second
/// type with the same three fields.
pub use stark_ui::color::Wheel;

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
    hex: &Entity<InputState>,
    regions: &Regions,
) -> impl IntoElement + use<> {
    let (mx, my) = color::wheel_xy(wheel.hue, wheel.sat);
    let rgb_now = wheel.rgb(WHEEL_GAMUT);
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
                .bg(rgb(style::PANEL))
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
                .bg(rgb(style::WELL))
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
                        .border_color(rgb(style::EDGE))
                        .bg(rgb(swatch)),
                )
                // The notation, as a field rather than a readout: what the picker
                // stands on can be copied out, and a color can be brought in by
                // typing it (`crate::controls` hears the field).
                .child(Input::new(hex).xsmall().flex_1()),
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
