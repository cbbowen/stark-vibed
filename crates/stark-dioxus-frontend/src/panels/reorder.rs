//! The pieces of the list-drag gesture that cannot leave a frontend: the grip a row is
//! picked up by, and writing a row's motion as CSS.
//!
//! Everything else — [`Grab`](stark_ui::reorder::Grab), the column arithmetic and what a displaced row does
//! about it — is `stark_ui::reorder`, because it is arithmetic over boxes and the hand.
//! These read the DOM, hold a `Signal` and emit a stylesheet dialect, which is the
//! definition of chrome (§11.2).

use dioxus::prelude::*;

use crate::platform::{capture_pointer, guide_boxes, layer_boxes};
use stark_model::document::LayerId;
use stark_ui::reorder::{Grab, Motion};

/// How long a displaced row takes to reach its new place. Long enough to be followed
/// by eye, short enough that the list has settled by the time the hand arrives.
const SLIDE_MS: u32 = 180;

/// A row's own two declarations — **both of them, every render, including the ones
/// that are "off"**.
///
/// Inline styles are applied property by property rather than by replacing the
/// attribute, so a declaration left out of a render is not cleared: it keeps whatever
/// the last render that *did* mention it gave it. That is how a dropped row was once
/// left wearing a preview's transform over a list that had since reordered
/// (`layout::Panel` carries the same scar). Writing the pair from one place is one
/// place for the rule to hold rather than one per panel.
///
/// A free function rather than a method on [`Motion`], because the orphan rule puts
/// one out of reach here — and because the shift, the lift and the ease are what a
/// native list would want too, while the declarations are this frontend's alone.
pub fn css(motion: Motion) -> String {
    let (dx, dy) = motion.shift;
    let ease = if motion.live && !motion.lifted {
        format!("transform {SLIDE_MS}ms ease")
    } else {
        "none".to_string()
    };
    format!("transform: translate({dx}px, {dy}px); transition: {ease};")
}

/// The row a [`Grip`] picks up — and so the list a press measures, which is why the
/// two are one value: a key cannot be looked up among another list's boxes.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum RowKey {
    /// A layer, by id — the tree's rows wear it as `data-layer`.
    Layer(LayerId),
    /// A guide, by its place in the roster — the rows wear it as `data-guide`, since a
    /// drag over drawn rows has no other name for one (`panels::guides`).
    Guide(usize),
}

impl RowKey {
    fn key(self) -> String {
        match self {
            RowKey::Layer(id) => id.to_string(),
            RowKey::Guide(index) => index.to_string(),
        }
    }

    fn boxes(self) -> Vec<(String, f32, f32)> {
        match self {
            RowKey::Layer(_) => layer_boxes(),
            RowKey::Guide(_) => guide_boxes(),
        }
    }
}

/// A roster row's name, which **is** its grip, as a panel's title is (`layout::Panel`):
/// the thing you would reach for to move a row is the row. Three gestures share its
/// press — a click selects, a double-click renames, and a press that travels is a move —
/// so the drag arms on the press and only *becomes* one once the pointer has said so.
///
/// All of that is here rather than in each roster: the capture that makes the release
/// certain (it is delivered to the capturing element whatever the pointer is over, and
/// here everything under the pointer moves), the armed check that keeps a hover from
/// dirtying the list, the disarm before the list is written ([`landed`]), and the click a
/// finished drag leaves behind ([`claimed`]). `onland` hears only a drag that went live.
#[component]
pub fn Grip(
    class: &'static str,
    title: &'static str,
    row: RowKey,
    drag: Signal<Option<Grab>>,
    onclick: EventHandler<()>,
    ondoubleclick: EventHandler<()>,
    onland: EventHandler<()>,
    children: Element,
) -> Element {
    rsx! {
        button {
            class,
            title,
            onclick: move |_| {
                if !claimed(&mut drag) {
                    onclick.call(());
                }
            },
            ondoubleclick: move |_| ondoubleclick.call(()),
            onpointerdown: move |e: Event<PointerData>| {
                capture_pointer(&e);
                let p = e.client_coordinates();
                drag.set(Some(Grab::begin(row.key(), row.boxes(), (p.x as f32, p.y as f32))));
            },
            onpointermove: move |e: Event<PointerData>| {
                // A finished grab is not armed: it is a receipt waiting for its click.
                if drag.peek().as_ref().is_none_or(Grab::over) {
                    return;
                }
                let p = e.client_coordinates();
                // Whether the press is still down — the name is also a thing hovered, and
                // a release this row never heard must not leave a drag to steer
                // (`Grab::track`).
                let held = !e.held_buttons().is_empty();
                if let Some(d) = drag.write().as_mut() {
                    d.track((p.x as f32, p.y as f32), held);
                }
            },
            onpointerup: move |_| {
                if landed(&mut drag) {
                    onland.call(());
                }
            },
            // The browser taking the gesture, or a pen leaving the tablet, ends it too.
            onpointercancel: move |_| {
                if landed(&mut drag) {
                    onland.call(());
                }
            },
            {children}
        }
    }
}

/// Whether the release a [`Grip`] heard ends a drag.
///
/// A press that never travelled is a click the browser is about to send, and its grab
/// is dropped. A drag is **spent** — before anyone writes the list, since a row's shift
/// is stated against the list as it stood at the press, and a frame carrying the new
/// order with the shifts still on would be the move applied twice. Spent rather than
/// dropped, so [`claimed`] can recognise the click behind the release.
fn landed(drag: &mut Signal<Option<Grab>>) -> bool {
    let live = drag.peek().as_ref().is_some_and(Grab::live);
    if !live {
        drag.set(None);
    } else if let Some(d) = drag.write().as_mut() {
        d.spend();
    }
    live
}

/// Whether the `click` the browser has just sent belongs to a drag that already
/// landed — and clears the grab either way, so the next press starts clean.
///
/// A row that both clicks and drags gets a click after *every* release, drag or not,
/// and on a list that has just reordered itself that click is aimed at a row which is
/// no longer the one that was pressed. Swallowing it is what keeps a move from also
/// being a selection of whatever took the moved row's place.
///
/// A press always overwrites the grab, so a finished one cannot outlive the gesture
/// that left it even if no click ever arrives — and while it waits it is inert, which
/// is [`Grab::spend`]'s job rather than this one's. It was not always: a finished grab
/// that a hover could re-arm turned "no click arrived" from a thing that costs nothing
/// into a row that follows the pointer until the panel is closed (`2026-08-09`).
fn claimed(grab: &mut Signal<Option<Grab>>) -> bool {
    let over = grab.peek().as_ref().is_some_and(Grab::over);
    if over {
        grab.set(None);
    }
    over
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every row's transform is written on every render, including at rest — the
    /// declaration a render leaves out is the one that goes stale.
    #[test]
    fn a_resting_row_still_states_its_transform() {
        let css = css(Motion::default());
        assert!(css.contains("translate(0px, 0px)"), "{css}");
        assert!(css.contains("transition: none"), "{css}");
    }
}
