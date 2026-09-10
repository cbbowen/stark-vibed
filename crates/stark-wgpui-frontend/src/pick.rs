//! The eyedropper's chrome (§18.0.2): the options bar the chord raises, and the
//! cursor that says what a press is about to do.
//!
//! **Nothing about the sampler is decided here.** Which reaches there are, what the
//! group fence lets in, how a choice resolves against the layer selected now and
//! whether the chord is even armed are all `stark_ui::pick`; the three reaches wear
//! the registry's own words and marks (`Command::SetPickScope`). What is this
//! module's is where the bar sits and how it is measured — `crate::select`'s split,
//! for its reason.
//!
//! # Why a bar, and why it is mounted rather than dimmed
//!
//! The eyedropper is not a tool you switch to: the chord over the brush *is* the
//! binding, as in Clip Studio Paint and Rebelle, so it has no resting state for a
//! panel to occupy. Coming up on the modifier is also the whole discoverability of a
//! modifier binding — press Alt and the options appear — and it goes down again the
//! moment the drag starts, because from then on the thing to look at is the canvas
//! and the color coming off it.
//!
//! The web app floats its bar beside the cursor; this one takes the **top of the
//! canvas**, which is where every other bar in this frontend goes (`crate::transform`,
//! `crate::select`) and the one edge with nothing above it.

use stark_ui::commands::{Bindings, Command, PickScope};
use stark_ui::pick::{PATCHES, Sampler, patch_word};
use strum::VariantArray;
use wgpui::{
    Bounds, HitboxBehavior, IntoElement, Pixels, Point, SharedString, canvas, div, prelude::*, rgb,
    rgba,
};

use crate::style::{self, StyleExt};

/// Which of the bar's controls a press landed on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Region {
    /// One of `PickScope::VARIANTS`, by index.
    Scope(usize),
    /// The group fence.
    Group,
    /// One of [`PATCHES`], by index.
    Patch(usize),
    /// The strip behind the chips. A press that missed a chip is still not a press on
    /// the picture — and over the canvas there is nothing below it to say so.
    Bar,
}

/// Where the bar's controls were laid out — `crate::panel`'s device, for its reason.
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
///
/// Innermost first (`.rev()`), which is `crate::select`'s rule and its reason: the
/// strip is probed as the bar's first child and contains every chip on it, so reading
/// forwards would hand back the ground under a chip that was pressed.
pub fn hit(regions: &Regions, at: Point<Pixels>) -> Option<Region> {
    regions
        .borrow()
        .iter()
        .rev()
        .find(|(_, bounds)| bounds.contains(&at))
        .map(|(region, _)| *region)
}

/// What a press on `region` makes of the sampler, or the command it asks for.
///
/// A function over the value rather than a match in the view, so what a chip *does*
/// is testable without a window — the same split `layers::act` makes.
pub fn act(sampler: &mut Sampler, region: Region) -> Option<Command> {
    match region {
        // The reach is the registry's act rather than a write here: the chip, the
        // Alt+Q/A/Z chord and the palette row must not describe one reach three ways.
        Region::Scope(i) => PickScope::VARIANTS
            .get(i)
            .copied()
            .map(Command::SetPickScope),
        Region::Group => {
            sampler.group_only = !sampler.group_only;
            None
        }
        Region::Patch(i) => {
            if let Some(radius) = PATCHES.get(i) {
                sampler.radius = *radius;
            }
            None
        }
        Region::Bar => None,
    }
}

/// The options bar, along the top of the canvas.
///
/// The same edge and ground as the other two bars, and it takes the edge from the
/// selection's while it is up: this one is about the press that is *about* to happen,
/// and it is gone again as soon as the modifier is (`Canvas::pick_hand`).
pub fn bar(sampler: Sampler, bindings: &Bindings, regions: &Regions) -> impl IntoElement {
    div()
        .absolute()
        .top_0()
        .left_0()
        .right_0()
        .flex()
        .items_center()
        .gap_1()
        .p_2()
        .bg(rgba(style::PANEL_OVER_CANVAS))
        .border_b_1()
        .border_color(rgb(style::EDGE))
        .text_color(rgb(style::INK_LIT))
        .child(probe(regions, Region::Bar))
        // The tool the chord has just armed, drawn as well as named: this bar exists
        // to make a modifier binding discoverable, and a picture of the eyedropper
        // appearing on the canvas is the shortest version of that argument.
        .child(
            div()
                .flex()
                .items_center()
                .gap_1()
                .py_1()
                .px_2()
                .caption()
                .child(crate::icons::icon(
                    stark_ui::icons::EYEDROPPER,
                    style::INK_LABEL,
                ))
                .child("Eyedropper"),
        )
        // How far the sample sees. `PickScope::VARIANTS` is the ordering — one layer, the
        // layers beneath it, then all of them — so the run reads as one question
        // rather than as three unrelated chips.
        .child(
            div()
                .flex()
                .gap_1()
                .children(PickScope::VARIANTS.iter().enumerate().map(|(i, scope)| {
                    let command = Command::SetPickScope(*scope);
                    // Whole off the registry: the mark, the terse word, the sentence
                    // in the hover and the chord it advertises — Alt+Q / Alt+A /
                    // Alt+Z reach these same three acts without letting go of the
                    // modifier that raised the bar.
                    marked(
                        format!("scope-{scope:?}").into(),
                        probe(regions, Region::Scope(i)),
                        command.icon(),
                        command.word(),
                        *scope == sampler.scope,
                        command.tooltip(bindings),
                    )
                })),
        )
        // The fence the reach runs inside — beside the run rather than a fourth
        // position in it, because it composes with every reach instead of competing
        // with them. The canvas color arrives exactly when it comes down: the canvas
        // is a fact about the picture, not about any group of paint.
        .child(marked(
            "group-only".into(),
            probe(regions, Region::Group),
            stark_ui::icons::GROUP_ONLY,
            "Group",
            sampler.group_only,
            stark_ui::pick::GROUP_TIP.to_string(),
        ))
        // How much canvas one sample averages. The words are the radii spelled out
        // (`stark_ui::pick::patch_word`), so a chip cannot offer a square the engine
        // would not take.
        .child(
            div()
                .flex()
                .gap_1()
                .children(PATCHES.iter().enumerate().map(|(i, radius)| {
                    let word = patch_word(*radius);
                    let chip = div()
                        .id(SharedString::from(format!("patch-{radius}")))
                        .chip()
                        .py_1()
                        .px_2()
                        .lit(*radius == sampler.radius)
                        .child(probe(regions, Region::Patch(i)))
                        .child(word);
                    style::tip(chip, "How much canvas one sample averages")
                })),
        )
}

/// One chip wearing its mark and its word, with the sentence in the hover — the
/// selection bar's chip, since a bar is as wide as the canvas and the room the
/// columns never had is here.
fn marked(
    id: SharedString,
    probe: impl IntoElement,
    mark: stark_ui::icons::Icon,
    word: &str,
    lit: bool,
    tip: String,
) -> impl IntoElement {
    let ink = if lit { style::INK_LIT } else { style::INK_MARK };
    let chip = div()
        .id(id)
        .chip()
        .flex()
        .items_center()
        .gap_1()
        .py_1()
        .px_2()
        .lit(lit)
        .child(probe)
        .child(crate::icons::icon(mark, ink))
        .child(word.to_string());
    style::tip(chip, tip)
}

/// The cursor the canvas wears while the sampler is armed — the whole
/// discoverability of a modifier binding, which is why it changes on the *key* rather
/// than on the press.
///
/// A crosshair rather than a drawn dropper: this toolkit offers the platform's
/// cursors and no custom bitmap, and the crosshair is the same fallback the web app's
/// rule names behind its inline-SVG dropper. Mounted before the bars so their own
/// chips keep the pointer they ask for.
pub fn cursor() -> impl IntoElement {
    canvas(
        move |bounds, window, _| window.insert_hitbox(bounds, HitboxBehavior::Normal),
        move |_, hitbox, window, _| {
            window.set_cursor_style(wgpui::CursorStyle::Crosshair, &hitbox);
        },
    )
    .absolute()
    .top_0()
    .left_0()
    .right_0()
    .bottom_0()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reach is asked for as the registry's act, never written into the sampler
    /// here — so the chip and the chord land on one thing.
    #[test]
    fn a_reach_is_the_registrys_act() {
        let mut sampler = Sampler::default();
        for (i, scope) in PickScope::VARIANTS.iter().enumerate() {
            assert_eq!(
                act(&mut sampler, Region::Scope(i)),
                Some(Command::SetPickScope(*scope))
            );
        }
        assert_eq!(sampler, Sampler::default(), "a chip wrote the reach itself");
    }

    /// The fence toggles and the patches set, both without asking for a command:
    /// neither is a nameable act, and inventing one for them would put two rows in
    /// the palette that no chord could usefully reach.
    #[test]
    fn the_fence_and_the_patch_are_the_bars_own() {
        let mut sampler = Sampler::default();
        assert_eq!(act(&mut sampler, Region::Group), None);
        assert!(!sampler.group_only);
        assert_eq!(act(&mut sampler, Region::Group), None);
        assert!(sampler.group_only);
        for (i, radius) in PATCHES.iter().enumerate() {
            assert_eq!(act(&mut sampler, Region::Patch(i)), None);
            assert_eq!(sampler.radius, *radius);
        }
    }

    /// The strip behind the chips does nothing at all — it exists so a press that
    /// missed a chip stops here rather than reaching the painting under it.
    #[test]
    fn the_strip_is_not_a_control() {
        let mut sampler = Sampler::default();
        assert_eq!(act(&mut sampler, Region::Bar), None);
        assert_eq!(sampler, Sampler::default());
    }
}
