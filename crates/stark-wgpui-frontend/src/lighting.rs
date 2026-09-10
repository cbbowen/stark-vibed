//! The Lighting shelf: the image-based-lighting media pass, the canvas substrate, and
//! the display in front of them (§6.3, §6.4, §6.5).
//!
//! The web app's Lighting panel as a docked column draws one. Everything it decides is
//! the engine's, so what is here is five tracks, a well, a drop-down and a switch,
//! each wearing its mark and saying the rest on hover (`crate::panel`). **Which** five,
//! over what ranges, and which rooms the drop-down lists are
//! `stark_ui::lighting`'s — they were this file's copy of the web panel's until there
//! were two panels to disagree.
//!
//! **The substrate gallery moved here** from the Brush shelf, where it was only
//! because there was one column: what a substrate does is catch the light (§6.4), and
//! the scale dial under it is meaningless without one. The stamp gallery stayed with
//! the brush for the mirror-image reason — a shape *is* the tool.
//!
//! **The canvas colour is taken rather than picked.** The web app flies an Oklab
//! picker out beside this panel because its Color panel is a column away and may be
//! shut; this window's is three inches away in the other column, so the well lays the
//! colour in hand and there is no second picker onto the same wheel.

use stark_engine::{EnvironmentId, ObservableState};
use stark_model::{Srgb, SubstrateId};
use stark_ui::lighting::{Dial, dials};
use stark_ui::prefs::Hdr;
use wgpui::{AnyElement, Bounds, IntoElement, Pixels, Point, canvas, div, prelude::*, px, rgb};
use wgpui_component::select::Select;

use crate::controls::Controls;
use crate::style::{self, StyleExt};

/// The HDR behind an image-backed environment, or `None` for the procedural
/// `Neutral`, which is generated on the GPU side and needs no bytes.
///
/// `include_bytes!`, on `crate::assets`' argument and pointed at the same directory:
/// a native binary is installed once, so fetching would mean an install layout to get
/// right and a `cargo run` that cannot find its own lights.
pub fn environment_hdr(id: EnvironmentId) -> Option<&'static [u8]> {
    Some(match id {
        EnvironmentId::Neutral => return None,
        EnvironmentId::Ferndale => {
            include_bytes!(
                "../../stark-dioxus-frontend/assets/environment/ferndale_studio_11_1k.hdr"
            )
        }
        EnvironmentId::BloemHill => {
            include_bytes!("../../stark-dioxus-frontend/assets/environment/bloem_hill_01_1k.hdr")
        }
        EnvironmentId::KloofendalOvercast => include_bytes!(
            "../../stark-dioxus-frontend/assets/environment/kloofendal_overcast_puresky_1k.hdr"
        ),
        EnvironmentId::QwantaniDusk => include_bytes!(
            "../../stark-dioxus-frontend/assets/environment/qwantani_dusk_2_puresky_1k.hdr"
        ),
    })
}

/// Which control a press on the shelf landed on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Region {
    /// The well that lays the colour in hand under the painting (§15.5).
    SubstrateColor,
    /// The HDR switch (§6.5).
    Hdr,
}

/// Where each was laid out — `crate::panel`'s device, for its reason.
pub type Regions = std::rc::Rc<std::cell::RefCell<Vec<(Region, Bounds<Pixels>)>>>;

fn probe(regions: &Regions, region: Region) -> impl IntoElement {
    let regions = regions.clone();
    canvas(
        move |bounds, _, _| regions.borrow_mut().push((region, bounds)),
        |_, (), _, _| {},
    )
    .absolute()
    .size_full()
}

/// Which control a press landed on.
pub fn hit(regions: &Regions, at: Point<Pixels>) -> Option<Region> {
    regions
        .borrow()
        .iter()
        .find(|(_, bounds)| bounds.contains(&at))
        .map(|(region, _)| *region)
}

/// What the shelf needs that is neither the projection nor a track.
pub struct Shown<'a> {
    pub obs: Option<&'a ObservableState>,
    /// This client's HDR choice, and whether the window in front of it can show
    /// anything above white at all.
    pub hdr: Hdr,
    pub hdr_capable: bool,
    pub display_headroom: Option<f32>,
    /// The substrate gallery, built by the view (`crate::gallery`). An `Option` for
    /// `panel::brush_body`'s reason: an element is built once and handed to whichever
    /// shelf wants it.
    pub substrates: Option<AnyElement>,
}

/// Build the shelf's body.
pub fn lighting_body(
    shown: Shown<'_>,
    controls: &Controls,
    regions: &Regions,
) -> impl IntoElement + use<> {
    let Shown {
        obs,
        hdr,
        hdr_capable,
        display_headroom,
        substrates,
    } = shown;
    let color = obs.map_or(stark_engine::document::DEFAULT_SUBSTRATE_COLOR, |o| {
        o.substrate_color
    });
    let swatch = packed(color.get());
    let flat = obs.is_some_and(|o| o.substrate == SubstrateId::Flat);
    let shown_dials = dials(hdr, hdr_capable, display_headroom);

    div()
        .flex()
        .flex_col()
        .gap_1()
        // The three media tracks, then the ground they light.
        .children(
            shown_dials
                .iter()
                .filter(|d| matches!(d, Dial::Impasto | Dial::Texture | Dial::Gloss))
                .map(|dial| track(*dial, obs, hdr, controls)),
        )
        // The colour under everything (§15.5), as a well that takes the colour the
        // brush is holding — see the module note on why there is no second picker.
        .child(
            div()
                .flex()
                .items_center()
                .gap_2()
                .py_0p5()
                .child(crate::icons::icon(stark_ui::icons::CANVAS, style::INK_MARK))
                .child(style::tip(
                    div()
                        .id("substrate-color")
                        .relative()
                        .flex_1()
                        .h(px(16.))
                        .rounded_sm()
                        .border_1()
                        .border_color(rgb(style::EDGE))
                        .cursor_pointer()
                        .bg(rgb(swatch))
                        .child(probe(regions, Region::SubstrateColor)),
                    "Canvas colour \u{2014} press to lay the colour in hand under the painting",
                )),
        )
        .children(substrates)
        // Inert on the procedural surface, whose height is a constant: there is no
        // substrate to size, and a live track would claim otherwise.
        .children(
            shown_dials
                .iter()
                .filter(|d| **d == Dial::Scale && !flat)
                .map(|dial| track(*dial, obs, hdr, controls)),
        )
        // Which sky the canvas is under (§6.3).
        .child(
            div()
                .flex()
                .items_center()
                .gap_2()
                .pt_1()
                .child(crate::icons::icon(stark_ui::icons::LIGHT, style::INK_MARK))
                .child(style::tip(
                    div()
                        .id("environment")
                        .flex_1()
                        .child(Select::new(&controls.environment).w_full()),
                    "Light \u{2014} the room the canvas is lit by",
                )),
        )
        // The HDR switch (§6.5): off is the picture an export makes. Only on a window
        // that can show more than white.
        //
        // The chip *is* the mark, lit or not, rather than a mark beside a word saying
        // which — one glyph rather than the same glyph twice, and the state is kept
        // where every other chip in the chrome keeps it (`style::StyleExt::lit`).
        .children(hdr_capable.then(|| {
            style::tip(
                div()
                    .id("hdr")
                    .relative()
                    .chip()
                    .flex()
                    .items_center()
                    .justify_center()
                    .py_1p5()
                    .mt_1()
                    .lit(hdr.on)
                    .child(probe(regions, Region::Hdr))
                    .child(crate::icons::icon(
                        stark_ui::icons::HDR,
                        if hdr.on {
                            style::INK_LIT
                        } else {
                            style::INK_MARK
                        },
                    )),
                "HDR \u{2014} let the picture go above white; off is what an export writes",
            )
        }))
        .children(
            shown_dials
                .iter()
                .filter(|d| **d == Dial::Headroom)
                .map(|dial| track(*dial, obs, hdr, controls)),
        )
}

/// One of the shelf's tracks.
fn track(
    dial: Dial,
    obs: Option<&ObservableState>,
    hdr: Hdr,
    controls: &Controls,
) -> impl IntoElement + use<> {
    let v = dial.read(obs, hdr);
    crate::panel::Slider::new(
        dial.glyph(),
        dial.tip(),
        dial.readout(v),
        controls.light(dial),
    )
}

/// A straight-sRGB triple as the packed value the toolkit paints with.
fn packed(rgb: [f32; 3]) -> u32 {
    let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u32;
    (byte(rgb[0]) << 16) | (byte(rgb[1]) << 8) | byte(rgb[2])
}

/// The colour the well would lay, given the colour in hand.
///
/// A function so the press is testable, which is worth one line here: it is the whole
/// of what this shelf's one destructive-looking act does, and it is document state
/// (§15.5) rather than a view setting.
pub fn take_color(in_hand: [f32; 3]) -> Srgb {
    Srgb::new(in_hand)
}

#[cfg(test)]
mod tests {
    use super::*;
    use stark_ui::lighting::ENVIRONMENTS;

    /// Every environment this shelf offers is one this build can actually light the
    /// canvas with: the procedural one needs no bytes, and each of the others has its
    /// HDR in the binary. A row whose file went missing would be a light that switches
    /// to nothing.
    #[test]
    fn every_light_offered_is_one_this_build_carries() {
        for (id, name) in ENVIRONMENTS {
            assert!(!name.is_empty());
            assert_eq!(
                environment_hdr(*id).is_none(),
                *id == EnvironmentId::Neutral,
                "{id:?} is the one procedural light, or it has bytes"
            );
        }
    }

    /// A colour survives the trip through the well, which is the whole of the act.
    #[test]
    fn the_well_lays_the_colour_in_hand() {
        assert_eq!(take_color([0.2, 0.4, 0.6]).get(), [0.2, 0.4, 0.6]);
    }
}
