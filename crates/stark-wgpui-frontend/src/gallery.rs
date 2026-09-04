//! The two asset galleries: brush stamps and canvas substrates (§6.4, §6.6, §11.2 N7).
//!
//! One module for two libraries, because they are one object twice over — which is
//! `stark_ui::assets`' claim, and this is it taken at its word: the rows, the
//! cards, the hit regions and the import are written once and instantiated at
//! [`Shapes`](stark_ui::assets::Shapes) and
//! [`Substrates`](stark_ui::assets::Substrates).
//!
//! What is this frontend's is the *carrier*. A card's picture — which texels, and
//! whether they are coverage or height — is the crate's; turning them into something
//! a toolkit can draw is not, and here that means a texture rather than the web's
//! data URL. The cache in front of it is the crate's too, generic in exactly that
//! difference (`stark_ui::library::Thumbs`).

use std::sync::Arc;

use stark_model::AssetId;
use stark_ui::assets::{Entry, Kind, Shapes, Shipped};
use stark_ui::library::Thumbs;
use wgpui::{
    Bounds, ImageSource, IntoElement, Pixels, Point, RenderImage, canvas, div, img, prelude::*, px,
    rgb,
};

/// The width of a card, logical px. Small enough that a row of them fits the panel's
/// column, large enough that a stamp's silhouette is legible.
const CARD: f32 = 46.0;

/// Each library's cards, keyed by the id each is a picture of.
///
/// **Two caches, never one.** The same grayscale PNG canonicalizes to one id under
/// both readings and means two different fields, so a shared table would hand a
/// substrate the picture of a stamp (`stark_ui::assets`).
static SHAPE_CARDS: Thumbs<Arc<RenderImage>> = Thumbs::new();
static SUBSTRATE_CARDS: Thumbs<Arc<RenderImage>> = Thumbs::new();

/// Which cache a library draws out of — the one thing [`Kind`] does not already say,
/// because a `static` cannot hang off a type parameter.
fn cards<K: Kind>() -> &'static Thumbs<Arc<RenderImage>> {
    if K::STORE == Shapes::STORE {
        &SHAPE_CARDS
    } else {
        &SUBSTRATE_CARDS
    }
}

/// The card for `id`, built from `png` the first time it is asked for.
///
/// An id *names* a field, so a card is a pure function of the id and of which library
/// is drawing it: there is nothing to invalidate and nothing to evict.
pub fn card<K: Kind>(id: AssetId, png: &[u8]) -> Option<Arc<RenderImage>> {
    if let Some(hit) = cards::<K>().get(id) {
        return Some(hit);
    }
    let (width, height, rgba) = crate::assets::card_rgba::<K>(png)?;
    // Straight through, RGBA: the polychrome atlas is `Rgba8Unorm`. This swapped for
    // one stage, taking wgpui's doc at its word, and nothing could show it — an asset
    // card is grey, and grey survives exchanging red and blue exactly. The colour
    // wheel caught it, and the vendored crate is fixed (patch 4).
    let buffer = image::RgbaImage::from_raw(width, height, rgba)?;
    let image = Arc::new(RenderImage::new(vec![image::Frame::new(buffer)]));
    cards::<K>().put(id, image.clone());
    Some(image)
}

/// Which of a gallery's cards a press landed on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Region {
    /// A shipped brush shape, by its row in `SHIPPED_SHAPES`.
    Shape(usize),
    /// One the user imported, by content id.
    OwnShape(AssetId),
    /// A shipped substrate, by its row in `SHIPPED_SUBSTRATES`.
    Substrate(usize),
    /// One the user imported.
    OwnSubstrate(AssetId),
    /// The button that opens a file for one of the two libraries.
    Import(Which),
    /// The button that drops the hovered entry from its library.
    Remove(Which, AssetId),
}

/// Which of the two galleries a region belongs to — the runtime half of [`Kind`],
/// for the places a value is what is available rather than a type.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Which {
    Shapes,
    Substrates,
}

/// Where each card was laid out — `crate::panel`'s device, for its reason.
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

/// Which card a press landed on.
pub fn hit(regions: &Regions, at: Point<Pixels>) -> Option<Region> {
    // Reversed, so the remove mark on top of a card wins over the card under it: the
    // two overlap by construction and the list is built parent-first.
    regions
        .borrow()
        .iter()
        .rev()
        .find(|(_, bounds)| bounds.contains(&at))
        .map(|(region, _)| *region)
}

/// What a gallery needs to draw itself, gathered by the view.
///
/// A borrowed slice of (id, picture) pairs rather than the library itself, because
/// what a card is a picture of comes from the *engine* first and the library second —
/// a shipped asset is only ever in the engine, and one imported in an earlier session
/// is only in the library until it is picked.
pub struct Shown<'a> {
    /// The shipped rows, each with the id it resolves to.
    pub rows: &'a [(&'static Shipped, Option<AssetId>)],
    /// The user's own entries.
    pub own: &'a [Entry],
    /// What is in hand, so one card can show as chosen.
    pub current: Option<AssetId>,
}

/// One gallery: a heading, a wrapped run of cards, and a way to add to it.
pub fn gallery<K: Kind>(
    which: Which,
    heading: &'static str,
    shown: Shown<'_>,
    bytes: impl Fn(AssetId) -> Option<Vec<u8>>,
    regions: &Regions,
    // **Erased, not `impl IntoElement`.** An opaque return type captures every type
    // parameter in scope, including the closure's — so the element would go on
    // borrowing whatever `bytes` borrowed, which for a caller reaching the renderer
    // through `&self` is the difference between one shared borrow and a frame that
    // cannot paint. The tree is built here; the closure is finished with here.
) -> wgpui::AnyElement {
    let shipped_region = |i: usize| match which {
        Which::Shapes => Region::Shape(i),
        Which::Substrates => Region::Substrate(i),
    };
    let own_region = |id: AssetId| match which {
        Which::Shapes => Region::OwnShape(id),
        Which::Substrates => Region::OwnSubstrate(id),
    };
    div()
        .flex()
        .flex_col()
        .gap_1()
        .child(
            div()
                .flex()
                .justify_between()
                .items_center()
                .pt_2()
                .child(div().text_sm().text_color(rgb(0x9aa0a6)).child(heading))
                .child(
                    div()
                        .relative()
                        .py_1()
                        .px_2()
                        .rounded_sm()
                        .bg(rgb(0x2a2d31))
                        .text_xs()
                        .text_color(rgb(0xb0b4b8))
                        .cursor_pointer()
                        .child(probe(regions, Region::Import(which)))
                        .child("Import\u{2026}"),
                ),
        )
        .child(
            div()
                .flex()
                .flex_wrap()
                .gap_1()
                .children(shown.rows.iter().enumerate().map(|(i, (row, id))| {
                    // A shipped row with no file is the procedural one, which has no
                    // field to draw: it wears its initial rather than a blank card.
                    let picture = id
                        .filter(|_| row.path.is_some())
                        .and_then(|id| bytes(id).and_then(|png| card::<K>(id, &png)));
                    face(
                        probe(regions, shipped_region(i)),
                        picture,
                        row.name,
                        *id == shown.current && id.is_some(),
                        None,
                    )
                }))
                .children(shown.own.iter().map(|entry| {
                    let picture = card::<K>(entry.id, &entry.png);
                    face(
                        probe(regions, own_region(entry.id)),
                        picture,
                        &entry.name,
                        Some(entry.id) == shown.current,
                        Some(probe(regions, Region::Remove(which, entry.id)).into_any_element()),
                    )
                })),
        )
        .into_any_element()
}

/// One card: the picture, the name under it, and — for an entry the user owns — the
/// mark that drops it.
fn face(
    probe: impl IntoElement,
    picture: Option<Arc<RenderImage>>,
    name: &str,
    chosen: bool,
    remove: Option<wgpui::AnyElement>,
) -> impl IntoElement {
    div()
        .relative()
        .w(px(CARD))
        .flex()
        .flex_col()
        .items_center()
        .gap_1()
        .cursor_pointer()
        .child(probe)
        .child(
            div()
                .relative()
                .size(px(CARD))
                .rounded_sm()
                // A card sets the ground its picture sits on, which is what makes a
                // stamp's coverage legible: white ink over a dark square.
                .bg(rgb(0x14161a))
                .border_1()
                .border_color(if chosen { rgb(0x5b9dd9) } else { rgb(0x35393d) })
                .children(picture.map(|image| img(ImageSource::Render(image)).size(px(CARD - 2.))))
                .children(remove.map(|mark| {
                    div()
                        .absolute()
                        .top_0()
                        .right_0()
                        .px_1()
                        .text_xs()
                        .text_color(rgb(0x9aa0a6))
                        .child(mark)
                        .child("\u{00d7}")
                })),
        )
        .child(
            // Truncated by the card's width rather than by a middle ellipsis: two
            // stamps whose names share a prefix are told apart by their pictures,
            // which is the whole reason a gallery is cards and not a list.
            div()
                .w_full()
                .text_xs()
                .text_center()
                .text_color(if chosen { rgb(0xe8eaed) } else { rgb(0x9aa0a6) })
                .overflow_hidden()
                .child(name.to_string()),
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use stark_ui::assets::Substrates;

    /// The two libraries draw out of two caches. One would hand a substrate the
    /// picture of a stamp, since a grayscale PNG lands on one id under both readings.
    #[test]
    fn each_library_has_its_own_cache() {
        assert!(!std::ptr::eq(cards::<Shapes>(), cards::<Substrates>()));
    }

    /// A card's texels come back as BGRA of the right length for its size.
    #[test]
    fn a_card_is_built_at_the_size_its_field_reduced_to() {
        let png = crate::assets::bundled("shape/Pencil.png").expect("the pencil ships");
        let (w, h, rgba) = crate::assets::card_rgba::<Shapes>(png).expect("it reduces");
        assert_eq!(rgba.len(), (w * h * 4) as usize);
        assert!(w <= stark_ui::library::THUMB_DIM);
        assert!(h <= stark_ui::library::THUMB_DIM);
    }

    /// A stamp's card is white ink with the coverage in alpha, so the panel shows
    /// through where it lays nothing; a substrate's is opaque, because it has no gaps.
    #[test]
    fn the_two_readings_carry_their_field_in_different_channels() {
        let shape = crate::assets::bundled("shape/Pencil.png").expect("the pencil ships");
        let (_, _, ink) = crate::assets::card_rgba::<Shapes>(shape).expect("it reduces");
        assert!(
            ink.as_chunks::<4>().0.iter().all(|p| p[..3] == [255; 3]),
            "a stamp's card is a constant ink"
        );
        assert!(
            ink.as_chunks::<4>().0.iter().any(|p| p[3] != 255),
            "and its field is in the alpha channel"
        );

        let ground = crate::assets::bundled("substrate/Rough.png").expect("rough ships");
        let (_, _, height) = crate::assets::card_rgba::<Substrates>(ground).expect("it reduces");
        assert!(
            height.as_chunks::<4>().0.iter().all(|p| p[3] == 255),
            "a substrate's card has no holes in it"
        );
    }
}
