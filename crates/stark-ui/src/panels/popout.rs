//! Where a panel's pop-out is drawn (§25.7).
//!
//! A bar draws its own: nothing clips `.bottom-bars`, so the frame bar's colour
//! picker hangs off the well in the markup and needs no coordinates at all. A panel
//! cannot. The stack is a scroll container that clips (`overflow-y: auto`,
//! `overflow-x: clip`) and every panel in it carries a `backdrop-filter`, which makes
//! a containing block — so a surface flown out of a panel row is clipped at the
//! column's edge whether it is `absolute` or `fixed`. There is no arrangement of the
//! markup that gets it out.
//!
//! So it is mounted here instead, at the app root, and *placed*: the row it belongs
//! to is measured and the surface is stood beside it (`crate::anchor`, which is the
//! same machinery the tour's card is placed with, §24.3). One component for all of
//! them, because the measurement, what keeps it true as the column moves under it and
//! the choice of which way to grow are the same three answers whatever is inside.
//!
//! What is *inside* is not this module's. `PopoutId::in_stack` says which pop-outs
//! are the stack's, [`StackPopouts`] says where each goes, and the body of each is a
//! component owned by the feature it belongs to — so a pop-out is a way of showing a
//! control rather than a place a control has to move to in order to be shown.

use dioxus::prelude::*;

use crate::anchor;
use crate::layout::chrome_class;
use crate::platform::{self, ElementBox};
use crate::state::AppState;
use crate::widgets::{PopoutId, close_popout};

/// Every pop-out that flies out of the panel stack, mounted at the app root for the
/// life of the page and empty whenever none is open.
///
/// **Mounted for the life of the page** rather than with the panel that opens one —
/// the bargain `tutor::TutorCard` makes, and for its reason: the effect below is what
/// places a pop-out, and an effect unmounted out from under the surface it is placing
/// leaves that surface wherever it last was.
///
/// One component for all of them rather than one each, because the placing is the
/// same three answers whatever is inside: which row, how it stays on that row, and
/// which way it grows out of it.
#[component]
pub fn StackPopouts() -> Element {
    let state = use_context::<AppState>();
    let mut at = use_signal(|| None::<ElementBox>);

    // Place the open pop-out, and keep it placed.
    use_effect(move || {
        let open = (state.popout)();
        // **A canvas gesture closes it.** A pop-out flown out of the stack stands over
        // the painting, so the moment the artist goes back to painting it is in the
        // way — and the press that says so is a stroke, not a dismissal. Closing on
        // the gesture rather than catching the press is the whole point: a catcher
        // over the canvas would eat that first stroke, which is worse than the
        // pop-out staying up (`widgets::PopoutId`, on the light dismiss still owed).
        if (state.canvas_active)() && open.and_then(PopoutId::in_stack).is_some() {
            close_popout(state);
            return;
        }
        let Some(selector) = open.and_then(PopoutId::in_stack) else {
            // Nothing of ours is open, and there is no loop running to say so.
            if at.peek().is_some() {
                at.set(None);
            }
            return;
        };
        // Every frame until it closes, because everything that moves the row moves it
        // silently — see [`anchor::follow`].
        anchor::follow(
            at,
            move || showing_row(selector),
            move || *state.popout.peek() == open,
        );
    });

    let Some(id) = (state.popout)() else {
        return rsx! {};
    };
    if id.in_stack().is_none() {
        return rsx! {};
    }
    // No box yet, or the row went out from under it — the panel closed, the column
    // scrolled it away. Drawing nothing is the honest answer, and the effect above is
    // still watching, so it comes back if the row does.
    let Some(at) = at() else {
        return rsx! {};
    };

    // **Which way it grows out of its row** — the one question here the stylesheet
    // cannot answer for itself. The horizontal is the panel column's own geometry,
    // which `.stack-popout` states in `calc` without measuring anything; the vertical
    // is a fact about one row, and a branch, and `calc` cannot branch.
    //
    // Down from the row's top or up from its bottom, whichever the window has more
    // of. Not *centred* on the row, which is the obvious third answer and the wrong
    // one: a surface centred on a row may only be twice the room on its narrower
    // side, so a pop-out beside the second row of the column would be capped at a few
    // hundred pixels in exactly the case where the whole screen below it was free.
    // Both arms cap what is left ([`anchor::room_below`]) and the surface scrolls
    // inside the cap rather than running off the edge.
    let below = platform::viewport_height() - at.top;
    let place = if below >= at.bottom() {
        format!("top: {:.1}px; {}", at.top, anchor::room_below(at.top))
    } else {
        format!(
            "top: {:.1}px; transform: translateY(-100%); {}",
            at.bottom(),
            anchor::room_above(at.bottom()),
        )
    };

    rsx! {
        div {
            class: chrome_class(state, "stack-popout"),
            style: "{place}",
            match id {
                PopoutId::SubstrateColor => rsx! { super::lighting::SubstrateColorPicker {} },
                PopoutId::SubstrateGallery => rsx! { crate::substrates::SubstrateGallery {} },
                // The ones drawn where they are opened — a bar's own, the rail's
                // — which `in_stack` has already returned for above.
                PopoutId::VisibilityMenu
                | PopoutId::Parcel
                | PopoutId::GradientLibrary => rsx! {},
            }
        }
    }
}

/// The row a stack pop-out belongs to, **and `None` while it is not showing**.
///
/// A row scrolled out of the column still has a box — the stack clips it, it does not
/// move it — and standing a surface beside that box puts the surface off the screen
/// entirely, pointing at something the artist cannot see. So the answer is qualified
/// by the scroller: a pop-out is on screen exactly while the row it flew out of is,
/// and it comes back when the row is scrolled back, because this is asked again every
/// frame.
///
/// Overlap rather than *containment*, so a row half over the column's edge keeps its
/// pop-out: the alternative flickers the surface off and on through the middle of
/// every scroll that passes it.
fn showing_row(selector: &'static str) -> Option<ElementBox> {
    let row = platform::anchor_box(selector)?;
    let stack = platform::anchor_box(".panel-stack")?;
    (row.bottom() > stack.top && row.top < stack.bottom()).then_some(row)
}
