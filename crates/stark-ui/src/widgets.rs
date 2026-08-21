//! Small reusable controls shared by the panels, the dialogs and the brush editor.

use dioxus::prelude::*;

use crate::commands::Command;
use crate::icons::{icon, label as label_span};
use crate::state::AppState;

/// A button that runs a [`Command`], wearing the command's own mark, word and
/// tooltip (`crate::commands`) — so a control and the act it reaches cannot
/// describe each other differently, and a chord the act gains is advertised
/// here without the button changing.
///
/// `class` stays the call site's because the registry deliberately says nothing
/// about *where* a command is drawn: the same act is a `chip` on a bar and a
/// `layer-add` in a panel header, and the stylesheet keys on the slot, not the
/// act. What a call site may **not** vary is what the button says or does — a
/// site needing that (the Fill chip's paint-tinted bucket) writes its own
/// `button` and still reads the words off the command.
///
/// Whether the button is **lit** is on that second list, so it comes off
/// [`Command::active`] rather than from a prop: a chip showing that its act is
/// live right now — the armed shape tool (§6.8) — is saying something about
/// the act, and a call site that computed it would be the second copy of an
/// answer the menu tick and the palette's blue mark already read from the
/// registry. A command with no such state (`None`) is never lit, which is
/// every act on a bar today.
#[component]
pub fn CommandButton(
    command: Command,
    #[props(default = String::from("chip"))] class: String,
) -> Element {
    let state = use_context::<AppState>();
    // A memo, for `CmdItem`'s reason: `active` reads the projection, which
    // moves at pointer rate during a stroke, and this button's answer is one
    // bool that almost never changes. Re-render on the bool, not on the read.
    let lit = use_memo(move || command.active(state) == Some(true));
    rsx! {
        button {
            class: "{class}",
            class: if lit() { "active" },
            title: command.tooltip(&state.bindings.read()),
            onclick: move |_| command.run(state),
            {icon(command.icon())}
            {label_span(command.word())}
        }
    }
}

/// A labelled range control.
///
/// `glyph` is an `Option` because the brush editor's dense parameter list has not been
/// marked yet, **not** because a slider is expected to go without one. A control's mark
/// is the half of it that survives its label, so anything reachable wants one; a `None`
/// here is a row that would be blank if the words were hidden, and is a to-do rather
/// than a decision (see [`crate::icons::SIZE`]).
///
/// Which is exactly why the word is wrapped as hideable ([`crate::icons::label`]) only
/// when there *is* a mark to fall back on. An unmarked slider keeps its word in minimal
/// mode — not as a special case, but because the two facts are one fact here, and a
/// component that reads them off each other cannot be given the wrong pair. The rows
/// still to be marked therefore stay legible in the meantime instead of turning into a
/// column of anonymous tracks.
///
/// `marked` on the row carries that same fact out to the stylesheet, which needs it for
/// a second reason: in minimal mode a marked row folds onto **one line**, its glyph to
/// the left of the track instead of over it, which is where the mode's vertical saving
/// in the panel stack actually comes from. A row that kept its word cannot fold — the
/// words differ in length, so the tracks would start at a ragged left edge — and it
/// does not have to, because the class it would need is the one it does not get.
#[component]
pub fn Slider(
    label: String,
    #[props(default)] glyph: Option<&'static str>,
    min: f32,
    max: f32,
    value: f32,
    oninput: EventHandler<f32>,
) -> Element {
    rsx! {
        div { class: if glyph.is_some() { "slider-row marked" } else { "slider-row" },
            div { class: "slider-label",
                match glyph {
                    Some(glyph) => rsx! { {icon(glyph)} {label_span(&label)} },
                    None => rsx! { "{label}" },
                }
            }
            input {
                class: "slider",
                r#type: "range", min: "{min}", max: "{max}", step: "any", value: "{value}",
                oninput: move |e| {
                    if let Ok(v) = e.value().parse::<f32>() { oninput.call(v); }
                },
            }
        }
    }
}

/// The shell every dialog floats in — the dimmed backdrop, the box on it, and
/// the one place the press-outside-to-dismiss rule is written (§25.7).
///
/// `class` is the box's extra classes (`modal-wide`, `be-dialog`) and the spread
/// attributes land on the box too, so a dialog the tutor anchors to keeps its own
/// mark. `on_close` is an `Option` because one dialog has no way out: the
/// GPU-failure notice ([`crate::failure`]) covers a canvas that cannot be drawn
/// any more, and there is nothing behind it to go back to.
///
/// **Why the rule cannot be a bare `onclick` on the backdrop.** A menu row acts on
/// `pointerdown` (dioxus-primitives' `MenubarItem` does, deliberately — see
/// `panels::filter::AddFilterButton` for the race it wins), so a dialog is mounted
/// while the pointer that opened it is still down. A pen, like a touch, is a
/// *direct-manipulation* device: the browser withholds the whole compatibility
/// mouse sequence for the gesture and hit-tests it fresh **at the release point** —
/// so the `mousedown`, `mouseup` and `click` of the very press that opened the
/// dialog are all delivered to the backdrop that press created. A backdrop
/// dismissing on any click dismisses itself in the act of opening, which is what
/// every dialog in the app did under a pen. A mouse is dispatched as it goes and
/// generates no click at all when its press target has been removed, so this was
/// invisible to every mouse the app was built with.
///
/// The rule that rules the class out: **a click dismisses only if this backdrop
/// also heard the press it belongs to.** `pointerdown` is the one event in that
/// deferred burst the browser does not re-target — it had already been delivered,
/// to a menu row, before the backdrop existed. The box stops both events on the
/// way up, which is what makes "armed" mean the press landed on the *backdrop*: a
/// slider dragged out of the dialog and let go over the dim stops reading as
/// dismissal too. Stopping them costs nothing above: the one listener that must
/// hear every press whatever it lands on binds in the capture phase for exactly
/// that reason ([`crate::platform::on_window_pointer`]).
#[component]
pub fn Modal(
    #[props(default = String::new())] class: String,
    on_close: Option<EventHandler<()>>,
    children: Element,
    #[props(extends = GlobalAttributes)] attributes: Vec<Attribute>,
) -> Element {
    let mut armed = use_signal(|| false);
    rsx! {
        div {
            class: "modal-backdrop",
            onpointerdown: move |_| armed.set(true),
            // Terminal, like every other disarm: a grip that hears a cancel is
            // over, and a stale arm would be spent on whatever click came next.
            onpointercancel: move |_| armed.set(false),
            onclick: move |_| {
                // Bound, not read in the `if` condition: the read would still be
                // held through the body, which writes the same signal.
                let heard_the_press = armed();
                armed.set(false);
                if let (true, Some(on_close)) = (heard_the_press, on_close) {
                    on_close.call(());
                }
            },
            div {
                class: "modal-dialog {class}",
                onpointerdown: move |e| e.stop_propagation(),
                onclick: move |e| e.stop_propagation(),
                ..attributes,
                {children}
            }
        }
    }
}
