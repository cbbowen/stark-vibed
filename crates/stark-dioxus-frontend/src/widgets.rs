//! Small reusable controls shared by the panels, the dialogs and the brush editor.

use crate::commands;
use dioxus::prelude::*;

use crate::icons::{icon, label as label_span};
use crate::state::{AppState, use_obs_opt};
use stark_chrome::commands::Command;

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
/// answer the lit mark a menu row and a palette row both wear already reads
/// from the registry. A command with no such state (`None`) is never lit,
/// which is every act on a bar today.
///
/// Whether it is **greyed** comes off [`Command::enabled`] the same way, and
/// for the same reason the menu's rows and the rail's read it there: a bar can
/// stand before the thing its acts need exists — the selection bar from the
/// moment a shape tool is armed (§6.8) — and a chip that can be pressed to do
/// nothing reads as broken. The act's own gate is still `run`'s; this is
/// presentation, as `enabled` says.
#[component]
pub fn CommandButton(
    command: Command,
    #[props(default = String::from("chip"))] class: String,
) -> Element {
    let state = use_context::<AppState>();
    // One memo, for `CmdItem`'s reason: both answers read the projection, which
    // moves at pointer rate during a stroke, and this button's pair of bools
    // almost never changes. Re-render on the bools, not on the read.
    let look = use_obs_opt(state, move |o| {
        (
            command.enabled(o),
            commands::active(command, state) == Some(true),
        )
    });
    let (enabled, lit) = look();
    rsx! {
        button {
            class: "{class}",
            class: if lit { "active" },
            disabled: !enabled,
            title: command.tooltip(&state.bindings.read()),
            onclick: move |_| commands::run(command, state),
            {icon(commands::icon(command))}
            {label_span(command.word())}
        }
    }
}

/// The filled share of a range control, as the inline `--fill` custom property
/// the track's gradient is drawn from (`.slider` in stark.css).
///
/// Inline because it is the one part of the slider's look only the control
/// knows: the track shows how far along its range the value sits, a browser
/// paints that only for a *native* range — in the platform's accent blue, which
/// the neutral chrome gave up — and CSS alone cannot see the value. Every range
/// input wearing `.slider` passes through here, the raw call sites as well as
/// [`Slider`]; the stylesheet's fallback (an empty track) is what a site that
/// forgets shows.
pub fn slider_fill(min: f32, max: f32, value: f32) -> String {
    let pct = if max > min {
        ((value - min) / (max - min) * 100.0).clamp(0.0, 100.0)
    } else {
        0.0
    };
    format!("--fill: {pct}%")
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
///
/// `onsettle` is the other half of a control whose value is **document state**: such
/// a slider previews per sample and lays its answer down once, and this is where the
/// three events that can end a drag are wired (see
/// [`Preview::settle`](crate::preview::Preview::settle) for why it takes all three).
/// A slider setting view state — most of them — leaves it out and has nothing to
/// settle.
///
/// `disabled` greys the track out (`.slider:disabled`) for a value that has
/// nothing to act on right now — the way a chip is disabled, and for the same
/// reason: a control that can be moved and does nothing reads as broken.
#[component]
pub fn Slider(
    label: String,
    #[props(default)] glyph: Option<&'static str>,
    min: f32,
    max: f32,
    value: f32,
    oninput: EventHandler<f32>,
    #[props(default)] onsettle: Option<EventHandler<()>>,
    #[props(default)] disabled: bool,
) -> Element {
    let settle = move || {
        if let Some(h) = &onsettle {
            h.call(());
        }
    };
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
                style: slider_fill(min, max, value),
                r#type: "range", min: "{min}", max: "{max}", step: "any", value: "{value}",
                disabled,
                oninput: move |e| {
                    if let Ok(v) = e.value().parse::<f32>() { oninput.call(v); }
                },
                onchange: move |_| settle(),
                onpointerup: move |_| settle(),
                onpointercancel: move |_| settle(),
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
/// `pointerdown`, deliberately (see `panels::filter::AddFilterButton` for the race
/// it wins), so a dialog is mounted while the pointer that opened it is still
/// down. A pen, like a touch, is a
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

/// The pop-outs the chrome can fly open, and the one place a surface that is
/// neither a panel nor a dialog is named (§25.7).
///
/// **One at a time, in one signal** ([`AppState::popout`](crate::state::AppState)),
/// on `modes::Composing`'s argument: two open at once is a state nothing wants
/// and nothing should have to prevent. They were `use_signal(|| false)` locals of
/// the surfaces that draw them, which made them invisible to the app — and in
/// particular to Escape, whose ladder knows the dialogs, the composing modes, the
/// composing layers and Timeline mode, and could not see a pop-out standing over
/// all of them (`commands::escape`). That is why the rail's menu is on this list
/// though no well opens it: reaching it from the ladder is the only way its
/// Escape is not a *second* handler for a keystroke the window already hears.
///
/// **Not a [`Dialogs`](crate::state::Dialogs) flag**, though the machinery would
/// have fitted: that list is also what stands `FinishMode` down
/// (`commands::dialog_open`), and the gradient library is opened *from* the
/// gradient bar while a fill is composing — so a pop-out on that list would take
/// Enter's "Done" away for as long as the library was open. It gets a rung of its
/// own, above the dialogs, and nothing else changes.
///
/// # What is still owed
///
/// **Light dismiss, for the ones flown out of a bar or a panel.** A press outside
/// a pop-out should close it and only the rail's does — which it can because it
/// holds the keyboard, so `focusout` answers the question for it
/// (`rail::VisibilityMenu`). Nothing in a bar does, and the fix there is not a
/// component: the catcher has to be root-mounted the way [`Modal`]'s backdrop is,
/// because `.bottom-bars` carries a `transform` and every bar and panel a
/// `backdrop-filter`, and each of those makes a containing block that a
/// `position: fixed` catcher rendered inside them cannot escape. It is also the
/// one part of this that cannot be got right by reading — where the catcher sits
/// among the z-indices decides which presses it eats, and eating a canvas press
/// would be worse than the bug — so it wants a browser rather than an argument.
///
/// **The one press it would eat is already handled**, which is what makes the rest
/// of it merely owed rather than urgent: a pop-out flown out of the stack stands over
/// the painting, and the press that matters there is the artist going back to
/// painting. `panels::popout` closes on the *gesture* instead of catching the press,
/// so the stroke that dismisses one also paints (`StackPopouts`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PopoutId {
    /// The rail's map of what is on screen (§25.5). The one member that is not a
    /// well's: it is here for the rung alone, since a menu that answered Escape
    /// at the element level would be a second actor on a keystroke the window is
    /// already hearing (`rail::VisibilityMenu`, `input::keys`).
    VisibilityMenu,
    /// The frame bar's matte-colour picker (§15.4).
    Parcel,
    /// The gradient library, flown out of a bar's ramp well (§22.3).
    GradientLibrary,
    /// The Lighting panel's canvas-colour picker (§6.4).
    SubstrateColor,
    /// The Lighting panel's surface gallery (§6.4).
    SubstrateGallery,
}

impl PopoutId {
    /// The control this pop-out flies out of, as a selector — and `None` for one
    /// that is drawn in place inside the bar that owns it.
    ///
    /// **The answer is where the pop-out is mounted**, which is the whole of the
    /// difference between the two kinds. A bar can draw its own: nothing clips
    /// `.bottom-bars`, so the picker hangs off the well in the markup and needs no
    /// coordinates. A panel cannot: the stack is a scroll container that clips, and
    /// every panel in it carries a `backdrop-filter`, so a surface flown out of a
    /// panel row has to be mounted at the app root and *placed* — which means being
    /// told which row, and that is what this selector is for
    /// (`panels::popout::StackPopouts`, `crate::anchor`).
    ///
    /// The row rather than the well inside it, deliberately: the row spans the
    /// panel's whole content width, so its left edge is the panel's own and the
    /// pop-out's distance from the column is a fact about the column rather than
    /// about which control in the row happened to be pressed.
    pub fn in_stack(self) -> Option<&'static str> {
        match self {
            PopoutId::VisibilityMenu | PopoutId::Parcel | PopoutId::GradientLibrary => None,
            PopoutId::SubstrateColor => Some("[data-popout=\"substrate-color\"]"),
            PopoutId::SubstrateGallery => Some("[data-popout=\"substrate-gallery\"]"),
        }
    }
}

/// Whether `id` is the pop-out currently open. Subscribing — the caller is the
/// bar that mounts it.
pub fn popout_open(state: AppState, id: PopoutId) -> bool {
    *state.popout.read() == Some(id)
}

/// Open `id`, closing whichever pop-out was open. Toggles, since every one of
/// these is opened by a press on the well it flies out of.
pub fn toggle_popout(state: AppState, id: PopoutId) {
    let mut open = state.popout;
    let was = *open.peek();
    open.set(if was == Some(id) { None } else { Some(id) });
}

/// Close `id` if it is the one open — what a surface that light-dismisses calls
/// when it hears the press leave it (`rail::VisibilityMenu`).
///
/// Guarded, where [`close_popout`] is not, because that press is often the one
/// opening the *next* pop-out: the focus leaves before the new well's click
/// lands, so an unguarded close would take down whatever had just come up.
///
/// `peek` rather than `read`, like [`toggle_popout`] above: this is called from a
/// handler, and subscribing one to the signal it is in the middle of writing is
/// how a read ends up live across a write of itself (`visibility::persist`).
pub fn close_popout_of(state: AppState, id: PopoutId) {
    let mut open = state.popout;
    if *open.peek() == Some(id) {
        open.set(None);
    }
}

/// Close whatever is open; `true` if anything was — what Escape's first rung
/// asks (`commands::escape`).
///
/// Also what a bar calls on its way out: a pop-out is drawn inside the bar that
/// owns it, so a bar that unmounts takes the pop-out off the screen without
/// clearing the flag, and the next time that bar came up the library would be
/// standing open on it.
pub fn close_popout(state: AppState) -> bool {
    let mut open = state.popout;
    let was = open.peek().is_some();
    if was {
        open.set(None);
    }
    was
}
