//! The lesson card (§24.3): where it stands against its anchor, what it opens, and when it
//! comes down.

use dioxus::prelude::*;
use stark_ui::panels::PanelId;
use strum::VariantArray;

use super::lessons::{Anchor, LESSONS, Side};
use super::reader::Effects;
use crate::anchor::{self, GAP};
use crate::dialogs::DialogId;
use crate::icons::icon;
use crate::layout::{PanelLayout, chrome_dimmed, open_panel, panel_key};
use crate::platform::{self, ElementBox};
use crate::state::{AppState, use_pref};

/// How far down its anchor a [`Side::Inside`] card sits, as a fraction of the anchor's
/// height: far enough to point into the picture rather than at its top edge, and not so far
/// that the card covers the middle of the work.
const INSIDE_DEPTH: f32 = 0.25;

impl Anchor {
    /// The selector that finds this anchor's element.
    ///
    /// Built from [`panel_key`] rather than written out, so a panel and the card
    /// pointing at it cannot come to disagree about what a panel is called in the
    /// DOM — a disagreement that shows as a card in the corner of the window with
    /// nothing beside it to explain.
    fn selector(self) -> String {
        match self {
            Anchor::Panel(id) => {
                format!(".panel-stack > .panel[data-panel=\"{}\"]", panel_key(id))
            }
            Anchor::PanelColumn => ".panel-wake".to_string(),
            Anchor::QuickSlots => ".slot-overlay".to_string(),
            Anchor::Navigator => ".navigator-overlay".to_string(),
            Anchor::CommandRail => ".command-rail".to_string(),
            // By the id the app already gives it, rather than by its class: the
            // canvas is named once (`render::CANVAS_ID`) and this is that name,
            // so there is no second spelling to fall out of step.
            Anchor::Canvas => format!("#{}", crate::render::CANVAS_ID),
            // From the editor's own naming, so a section renamed on screen keeps
            // its anchor and a section deleted stops compiling on both sides at
            // once (`brush_editor::BrushPart`).
            Anchor::BrushEditor(part) => format!("[data-be=\"{}\"]", part.key()),
            Anchor::TimelineBar => ".timeline-bar".to_string(),
        }
    }

    /// Whether this anchor is still *meant* to be on screen.
    ///
    /// Deliberately not the same question as "did it measure". A measurement comes
    /// back `None` for a frame while the browser lays out a panel that has only
    /// just opened, and dismissing a lesson on that would be dismissing every
    /// lesson the moment it appeared. This asks the app's own state instead, so the
    /// one thing it answers `false` to is the user having **closed** the thing the
    /// card is about — which is an answer to the card, and is taken as one.
    fn on_screen(self, state: AppState, layout: PanelLayout) -> bool {
        match self {
            Anchor::Panel(id) => !layout.hidden.read().contains(&id),
            // Asleep, with something to wake. **The wake is the dismissal** — reach
            // into the column and the panels come back, which is the lesson done and
            // is a better acknowledgement than the button. Deliberately *not* also
            // testing `canvas_active`, which is the third thing the slice itself
            // wants: painting hides the card for the length of the stroke (the
            // measurement finds nothing) and must not end the lesson, since starting
            // a stroke is not an answer to it.
            Anchor::PanelColumn => {
                let asleep = (state.panels_asleep)();
                let hidden = layout.hidden.read();
                asleep && PanelId::VARIANTS.iter().any(|id| !hidden.contains(id))
            }
            Anchor::QuickSlots => (state.slots.pinned)(),
            Anchor::Navigator => (state.navigator)(),
            // Always. Both are mounted for the life of the page and neither has a
            // control that puts it away, so these lessons are dismissed the ordinary
            // way and by nothing else.
            Anchor::CommandRail | Anchor::Canvas => true,
            // Exactly as long as the dialog is up. Closing it mid-series is an
            // answer to the card on screen and leaves the rest of the series owed
            // for the next time it is opened (`Tour::abandon`).
            Anchor::BrushEditor(_) => {
                crate::dialogs::is_open(state, crate::dialogs::DialogId::BrushEditor)
            }
            Anchor::TimelineBar => (state.timeline.open)(),
        }
    }

    /// Put this anchor on screen, so there is something for the card to point at.
    ///
    /// Derived from the anchor rather than named separately on the lesson: "what it
    /// points at" and "what it opens" are one fact, and a lesson able to state them
    /// differently is a lesson that could open the Color panel and point at the
    /// Brush one.
    fn reveal(self, state: AppState, layout: PanelLayout) {
        match self {
            Anchor::Panel(id) => open_panel(state, layout, id),
            // The slice exists only while the panels are standing down, so what
            // "reveal" means here is to put them there. A no-op in practice — a
            // lesson is promoted with the canvas out of hand, and the release that
            // freed it is what set this — but written out rather than relied upon,
            // so the card cannot be shown pointing at a box that is not in the DOM.
            Anchor::PanelColumn => crate::layout::sleep_panels(state),
            Anchor::QuickSlots => {
                let mut pinned = state.slots.pinned;
                pinned.set(true);
            }
            Anchor::Navigator => crate::navigator::set_open(state, true),
            // Nothing to do. The rail and the canvas are always there, and the editor
            // is up already — opening it is the deed that brings these due, and opening
            // it *for* somebody would be the tour taking the screen.
            Anchor::CommandRail | Anchor::Canvas | Anchor::BrushEditor(_) => {}
            Anchor::TimelineBar => crate::panels::timeline::set_open(state, true),
        }
    }
}

/// Whether the dialog on top of the stack covers a card pointing at `anchor`: every
/// dialog does, except the brush editor for a card pointing into it. Only the top is
/// asked, because a dialog opened over the editor covers its parts as well.
fn covered(top: Option<DialogId>, anchor: Anchor) -> bool {
    match top {
        None => false,
        Some(DialogId::BrushEditor) => !anchor.inside_dialog(),
        Some(_) => true,
    }
}

/// Put the lesson waiting on screen.
fn show(state: AppState) {
    super::step(state, |tour| {
        tour.show();
        Effects::default()
    });
}

/// Acknowledge lesson `i` by its button, which brings the next lesson its deed owes.
fn dismiss(state: AppState, i: usize) {
    let chrome = state.prefs.peek().chrome_hiding;
    super::step(state, |tour| tour.dismiss(i, chrome));
}

/// The lesson card: one at a time, floating beside the thing it points at.
///
/// Mounted at the app root for the life of the page and empty whenever no lesson is
/// showing, so its two effects — one promoting a lesson due, one measuring the anchor of
/// the lesson shown — are never unmounted from under a lesson in flight.
#[component]
pub fn TutorCard() -> Element {
    let state = use_context::<AppState>();
    let layout = state.panels;
    let mut anchored = use_signal(|| None::<ElementBox>);

    // A resize moves everything a card could point at. Bound once, like the app's other
    // window listeners: this component never unmounts.
    use_hook(|| {
        let mut epoch = state.tutor.epoch;
        platform::on_window_event("resize", move |_| {
            let n = *epoch.peek();
            epoch.set(n + 1);
        });
    });

    // The next lesson also depends on the chrome-hiding setting, which moves outside any step.
    use_effect(move || {
        let _ = (state.chrome_hiding)();
        super::publish(state);
    });

    // Show a lesson that has come due, once the screen is the user's again. Each condition
    // is a claim that the card would be *wrong* now: mid-gesture, a panel opened is put
    // back to sleep by the release; a composing mode owns the whole window
    // (`crate::modes`); and a dialog covers everything a card could point at.
    let tips = use_pref(state, |p| p.tips);
    use_effect(move || {
        let Some(i) = (state.tutor.due)() else { return };
        let Some(lesson) = LESSONS.get(i) else { return };
        // The switch as the way out as well as the way in: `due` can be set by the dismiss
        // chain or left from before tips went off, and a subscribing read offers it as soon
        // as they are back on.
        if !tips() {
            return;
        }
        let dialog = covered(state.dialogs.read().last().copied(), lesson.anchor);
        let busy = (state.canvas_active)() || dialog || crate::modes::composing(state).is_some();
        if busy {
            return;
        }
        lesson.anchor.reveal(state, layout);
        show(state);
    });

    // Measure whatever the lesson on screen points at, retrying for the frames a reveal
    // takes to lay out (`anchor::measure`). It follows the panel order and the hidden set as
    // well as the lesson, since both move the column the card points into — and those reads
    // are also what the dismissal below watches.
    use_effect(move || {
        let showing = (state.tutor.showing)();
        let _ = (layout.order)();
        let _ = (state.tutor.epoch)();
        // The anchors that come and go on their own: the wake slice is in the DOM only while
        // the panels are asleep and the canvas is out of hand (`layout::PanelStack`), the
        // rack only while pinned, the editor's parts only while it is open.
        let _ = (state.canvas_active)();
        let _ = (state.slots.pinned)();
        let _ = crate::dialogs::is_open(state, crate::dialogs::DialogId::BrushEditor);
        // Closing the thing a card is about answers the card, and latching it instead would
        // end the tour silently at whichever tip the artist closed a panel under. Asked of
        // the app's state rather than the DOM, for `Anchor::on_screen`'s reason.
        if let Some(i) = showing
            && let Some(lesson) = LESSONS.get(i)
            && !lesson.anchor.on_screen(state, layout)
        {
            super::step(state, |tour| tour.abandon(i));
            return;
        }
        let (Some(i), Some(selector)) = (
            showing,
            showing
                .and_then(|i| LESSONS.get(i))
                .map(|l| l.anchor.selector()),
        ) else {
            anchored.set(None);
            return;
        };
        // The retry, and the guard that stops a measurement in flight writing a box that
        // belongs to the lesson *this* one was replaced by (`anchor::measure`).
        anchor::measure(
            selector,
            anchored,
            move || *state.tutor.showing.peek() == Some(i),
            anchor::TRIES,
        );
    });

    let Some(i) = (state.tutor.showing)() else {
        return rsx! {};
    };
    let Some(lesson) = LESSONS.get(i) else {
        return rsx! {};
    };
    // Nothing to point at — the anchor went, or the DOM has not caught up. The effect above
    // is still watching, so the card comes back if the anchor does.
    let Some(at) = anchored() else {
        return rsx! {};
    };
    // Whether acknowledging this one brings another, which the button says.
    let more = (state.tutor.brings_another)();

    // Placed against the anchor's own edges, with no reading of the viewport and no
    // measurement of the card itself: the translate does the work that knowing the
    // card's width would otherwise be needed for.
    let (side, place) = match lesson.side {
        Side::LeftAtTop => (
            "side-left",
            format!(
                "left: {:.1}px; top: {:.1}px; transform: translateX(-100%); {}",
                at.left - GAP,
                at.top,
                anchor::room_left(at.left - GAP),
            ),
        ),
        Side::LeftAtMiddle => (
            "side-left-middle",
            format!(
                "left: {:.1}px; top: {:.1}px; transform: translate(-100%, -50%); {}",
                at.left - GAP,
                at.mid_y(),
                anchor::room_left(at.left - GAP),
            ),
        ),
        Side::RightAtTop => (
            "side-right",
            format!(
                "left: {:.1}px; top: {:.1}px; {}",
                at.right() + GAP,
                at.top,
                anchor::room_right(at.right() + GAP),
            ),
        ),
        Side::RightAtMiddle => (
            "side-right-middle",
            format!(
                "left: {:.1}px; top: {:.1}px; transform: translateY(-50%); {}",
                at.right() + GAP,
                at.mid_y(),
                anchor::room_right(at.right() + GAP),
            ),
        ),
        Side::Inside => (
            "side-inside",
            format!(
                "left: {:.1}px; top: {:.1}px; transform: translateX(-50%); {}",
                at.mid_x(),
                at.top + at.height * INSIDE_DEPTH,
                anchor::room_about(at.mid_x()),
            ),
        ),
        // The one placement that hands the stylesheet a measurement as well as a
        // position: where its arrow goes depends on how tall the anchor is, and this
        // is the only side whose anchor is sized by the artwork (`RightAtBottom`).
        //
        // A custom property is safe to write from one arm only, where a plain
        // declaration would not be: an inline style is applied property by property
        // in Dioxus, so a name this arm sets is *stranded* on the element when
        // another arm renders next. Nothing reads it there — the rule that does is
        // turned on by the side's class, and the class is rewritten whole.
        Side::RightAtBottom => (
            "side-right-bottom",
            format!(
                "left: {:.1}px; top: {:.1}px; transform: translateY(-100%); \
                 --tutor-reach: {:.1}px; {}",
                at.right() + GAP,
                at.bottom(),
                at.height * 0.5,
                anchor::room_right(at.right() + GAP),
            ),
        ),
        Side::Above => (
            "side-above",
            format!(
                "left: {:.1}px; top: {:.1}px; transform: translate(-50%, -100%); {}",
                at.mid_x(),
                at.top - GAP,
                anchor::room_about(at.mid_x()),
            ),
        ),
    };

    rsx! {
        div {
            class: "tutor-card chrome {side}",
            class: if chrome_dimmed(state) { "dimmed" },
            class: if lesson.anchor.inside_dialog() { "over-dialog" },
            style: "{place}",
            div { class: "tutor-head",
                span { class: "tutor-mark", {icon(stark_ui::icons::TOUR)} }
                span { class: "tutor-title", "{lesson.title}" }
            }
            div { class: "tutor-body", "{lesson.body}" }
            div { class: "tutor-actions",
                button {
                    class: "tutor-quiet",
                    title: "Stop showing tips. Settings has the switch to turn them back on.",
                    onclick: move |_| {
                        // Through the same door as the dialog's switch, which takes
                        // this card down without marking it given (`tutor::switch_off`).
                        crate::prefs::set(state, |p| p.tips = false);
                    },
                    "Stop tips"
                }
                button {
                    class: if more { "chip tutor-done tutor-next" } else { "chip tutor-done" },
                    onclick: move |_| dismiss(state, i),
                    {icon(if more { stark_ui::icons::NEXT } else { stark_ui::icons::DONE })}
                    if more { "Next" } else { "Got it" }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brush_editor::BrushPart;

    #[test]
    fn every_dialog_covers_a_card_but_the_editor_it_points_into() {
        let inside = Anchor::BrushEditor(BrushPart::Preview);
        assert!(!covered(None, Anchor::Canvas));
        assert!(covered(Some(DialogId::Settings), Anchor::Canvas));
        assert!(covered(Some(DialogId::BrushEditor), Anchor::Canvas));
        assert!(!covered(Some(DialogId::BrushEditor), inside));
        assert!(
            covered(Some(DialogId::PresetSave), inside),
            "a dialog opened over the editor covers its parts"
        );
    }
}
