//! Where each control was laid out — how a panel answers which control a press landed
//! on (§11.2).
//!
//! **A panel does not compute its own geometry, it reports it.** The first cut of the
//! brush panel hand-derived every offset from the Tailwind-shaped spacing the tree asks
//! for — and was wrong: a press where the arithmetic said "Airbrush" selected Hard
//! Eraser, because the guessed row pitch was 26 px and Taffy had laid out 39. Two
//! descriptions of one layout is exactly the drift `stark-ui` exists to stop, one
//! scale down, and the answer is the same: have one of them ask the other.
//!
//! So each control carries a zero-cost [`probe`] whose *prepaint* writes its laid-out
//! bounds into a [`Regions`]. The view clears the list each frame before building the
//! tree and reads it on the next press — layout is stable between frames, and a press
//! before the first prepaint simply finds nothing, which is the same answer as a press
//! on empty panel.

use wgpui::{Bounds, IntoElement, Pixels, Point, Styled, canvas};

/// Where each control was laid out, as of the last frame that painted, each rectangle
/// tagged with the module's own `R` saying which control it is.
///
/// `Rc<RefCell<..>>` because a prepaint closure is `'static` and the view outlives it:
/// wgpui hands the closure no context to write through, so what it writes into has to
/// be something it owns a handle to.
pub type Regions<R> = std::rc::Rc<std::cell::RefCell<Vec<(R, Bounds<Pixels>)>>>;

/// An invisible element that records its parent's bounds as `region`'s.
///
/// Absolute, on all four insets, so it takes the whole of the control it sits in and
/// adds nothing to the layout — four insets rather than `size_full`, because an
/// absolutely-positioned child with only a size is placed wherever the flow had
/// reached, which put the transform overlay a viewport below the window for exactly
/// one build (§11.2, N6). Its paint is empty: what is wanted is the *measurement*, and
/// the control draws itself.
pub fn probe<R: Copy + 'static>(regions: &Regions<R>, region: R) -> impl IntoElement {
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
/// **The last rectangle containing the point wins.** Controls can nest — a chip inside
/// a row inside a section inside the sheet — and the list is built parent-first, so
/// the last one is the innermost; where nothing nests, there is only one.
pub fn hit<R: Copy>(regions: &Regions<R>, at: Point<Pixels>) -> Option<R> {
    regions
        .borrow()
        .iter()
        .rev()
        .find(|(_, bounds)| bounds.contains(&at))
        .map(|(region, _)| *region)
}

/// Where `region` was laid out, if a frame has measured it.
pub fn bounds_of<R: PartialEq>(regions: &Regions<R>, region: R) -> Option<Bounds<Pixels>> {
    regions
        .borrow()
        .iter()
        .find(|(r, _)| *r == region)
        .map(|(_, bounds)| *bounds)
}

/// Where in `bounds` a position sits, as fractions of it — clamped, so a drag that has
/// left the control keeps moving the value it took hold of. `None` for a box with no
/// area.
pub fn fraction_in(bounds: Bounds<Pixels>, at: Point<Pixels>) -> Option<(f32, f32)> {
    let (w, h) = (f32::from(bounds.size.width), f32::from(bounds.size.height));
    (w > 0.0 && h > 0.0).then(|| {
        (
            ((f32::from(at.x) - f32::from(bounds.origin.x)) / w).clamp(0.0, 1.0),
            ((f32::from(at.y) - f32::from(bounds.origin.y)) / h).clamp(0.0, 1.0),
        )
    })
}
